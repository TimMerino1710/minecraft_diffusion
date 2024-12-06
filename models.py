from pathlib import Path
from functools import partial
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vision_utils
import lpips
from torchinfo import summary
import torch.distributions as dists

#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        #   so z is coming in as (batch, channel, height, width)
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #   this gives a (batch_size * height * width, codebook_size) tensor, where each element coresponds to ther squared euclidian distance b/t i-th input and jth embedding
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)   # is this an output metric? mean distance between z and embeddings

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  #   gets the index of the closest embedding for each input data point (shape is (batch_size * height * width, 1))
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)    #   creates a tensor of zeros with shape (batch_size * height * width, codebook_size), puts it on the same device as z
        min_encodings.scatter_(1, min_encoding_indices, 1)  #   sets the value at the index of the closest embedding to 1 (one-hot encoding)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  #   multiply the one-hot encodings by the embeddings to get the quantized latent vectors
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)    #   two loss terms. First is the reconstruction loss b/t quantized latent vectors and input data. second is the commitment loss
        # preserve gradients
        #   This is the straight-through estimator. It allows the gradients to flow through the quantized latent vectors during backpropagation
        #   Detach the diff between quantized and original z, then add it to z. This is the same as zq, but has gradients only w.r.t z. Allows gradients to pass through as if z_q was z
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)   #   mean of the one-hot encodings acriss dim 0 - so every item will be how frequently that encoding happens across all samples in the batch
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))  #   perplexity measures how spread out the encoding usage is. Higher perplexity is more uniform distribution (more uncertainty), which I think is good?
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
            }

    def get_codebook_entry(self, indices, shape):   #   get the embedding for a given index (used at inference etc)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q
    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0) #   padding the right and the bottom with 0s
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x
    
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)   #   divides the channels into 32 groups, and normalizes each group. More effective for smaller batch size than batch norm

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)   #  swish activation function, compiled using torch.jit.script. Smooth, non-linear activation function, works better than ReLu in some cases. swish (x) = x * sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)    #   Flattening the spatial dimensions
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw (flattening spatial dimensions)
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]   these are the attention weights (or attn scores)
        w_ = w_ * (int(c)**(-0.5))  #   scaling the attention weights by the sqrt of the number of channels to stabilize training
        w_ = F.softmax(w_, dim=2)  #   softmax over the last dimension (which is the keys right now)

        # attend to values
        v = v.reshape(b, c, h*w)    #   flattening spatial dimensions
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)       swapping k and q dimensions
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]      # weighing the values by the attention weights
        h_ = h_.reshape(b, c, h, w)     # reshape back into spatial dimensions

        h_ = self.proj_out(h_)      #   project the attended values back to the original number of channels

        return x+h_     #   add the attended values to the original input (residual)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))   #   initial convolution, nf filters, 3x3 kernel, stride 1, padding 1

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):   #  for each resolution in num_resulutions
            block_in_ch = nf * in_ch_mult[i]        #   input channels to the block is nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]      #   output channels is nf * ch_mult[i]
            for _ in range(self.num_res_blocks):        #   for each res block
                blocks.append(ResBlock(block_in_ch, block_out_ch))      #   add a res block
                block_in_ch = block_out_ch      #   set the input channels to the output channels
                if curr_res in attn_resolutions:        #   if the current resolution is in the attn_resolutions
                    blocks.append(AttnBlock(block_in_ch))       #   add an attention block

            if i != self.num_resolutions - 1:       #   if this isn't the last resolution
                blocks.append(Downsample(block_in_ch))      #   add a downsampling block
                curr_res = curr_res // 2        #   halve the resolution to account for the downsampling block

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))       #   final blocks: res block, attn block, res block
        blocks.append(AttnBlock(block_in_ch))       
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))       #   normalize the output
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        # print all of the blocks for debugging
        # print(f'blocks: {blocks}')
        
        self.blocks = nn.ModuleList(blocks)

        # summary(self, (in_channels, resolution, resolution))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = H.res_blocks
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.in_channels = H.emb_dim
        self.out_channels = H.n_channels
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        print(f'resolution: {self.resolution}, num_resolutions: {self.num_resolutions}, num_res_blocks: {self.num_res_blocks}, attn_resolutions: {self.attn_resolutions}, in_channels: {self.in_channels}, out_channels: {self.out_channels}, block_in_ch: {block_in_ch}, curr_res: {curr_res}')

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # used for calculating ELBO - fine tuned after training
        self.logsigma = nn.Sequential(
                            nn.Conv2d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
                        ).cuda()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def probabilistic(self, x):
        with torch.no_grad():
            for block in self.blocks[:-1]:
                x = block(x)
            mu = self.blocks[-1](x)
        logsigma = self.logsigma(x)
        return mu, logsigma
    
# patch based discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_layers=3):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        qy = F.softmax(logits, dim=1)

        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }
class EMAQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, decay=0.99, eps=1e-5):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.decay = decay
        self.eps = eps

        # Initialize embeddings with randn, transposed from original
        embed = torch.randn(emb_dim, codebook_size).t()
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        # Save input shape and flatten
        b, c, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, self.emb_dim)
        
        # Calculate distances
        dist = (
            z_flattened.pow(2).sum(1, keepdim=True)
            - 2 * z_flattened @ self.embedding.t()
            + self.embedding.pow(2).sum(1, keepdim=True).t()
        )
        
        # Get closest encodings
        _, min_encoding_indices = (-dist).max(1)
        min_encodings = F.one_hot(min_encoding_indices, self.codebook_size).type(z_flattened.dtype)
        
        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding)
        
        # EMA updates during training
        if self.training:
            embed_onehot_sum = min_encodings.sum(0)
            embed_sum = z_flattened.transpose(0, 1) @ min_encodings
            
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum.t(), alpha=1 - self.decay
            )

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            )
            
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
            
        # Reshape z_q and apply straight-through estimator
        z_q = z_q.view(b, h, w, c)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Calculate perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, torch.tensor(0.0, device=z.device), {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices.view(b, h, w),
            "mean_distance": dist.mean()
        }

    def get_codebook_entry(self, indices, shape):
        min_encodings = F.one_hot(indices, self.codebook_size).type(torch.float)
        z_q = torch.matmul(min_encodings, self.embedding)

        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q
    
class VQAutoEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.codebook_size = H.codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.quantizer_type = H.quantizer
        self.beta = H.beta
        self.gumbel_num_hiddens = H.emb_dim
        self.straight_through = H.gumbel_straight_through
        self.kl_weight = H.gumbel_kl_weight
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        if self.quantizer_type == "nearest":
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        elif self.quantizer_type == "ema": 
            self.quantize = EMAQuantizer(self.codebook_size, self.embed_dim).to('cuda')
        self.generator = Generator(H)

        if False:
            print("Encoder")
            # print(f'encoder input shape: {H.batch_size, H.n_channels, 64, 64}')
            print(summary(self.encoder, (H.batch_size, H.n_channels, 64, 64)))
            print("Quantizer")
            print(summary(self.quantize, (H.batch_size, H.emb_dim, 4, 4)))
            print("Generator")
            print(summary(self.generator, (H.batch_size, H.emb_dim, 4, 4)))
        

    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats

    def probabilistic(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            quant, _, quant_stats = self.quantize(x)
        mu, logsigma = self.generator.probabilistic(quant)
        return mu, logsigma, quant_stats
    
def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

@torch.jit.script
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class VQGAN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ae = VQAutoEncoder(H)
        self.disc = Discriminator(
            H.n_channels,
            H.ndf,
            n_layers=H.disc_layers
        ).to(device)
        self.use_perceptual = H.dataset not in ['maps']
        if self.use_perceptual:
            self.perceptual = lpips.LPIPS(net="vgg").to(device)
            self.perceptual_weight = H.perceptual_weight
        else:
            self.perceptual_weight = 0.0
        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.diff_aug = H.diff_aug
        self.policy = "color,translation"

        print("Discriminator")
        # print(summary(self.disc, (H.batch_size, H.n_channels, H.img_size, H.img_size)))

    def train_iter(self, x, step):  #   editing this to remove diff aug and gumbel
        stats = {}
        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        if self.use_perceptual:
            p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0, device=x.device)
            nll_loss = recon_loss
        nll_loss = torch.mean(nll_loss)


        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["d_weight"] = d_weight
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"].item()
        if step > self.disc_start_step:
            logits_real = self.disc(x.contiguous().detach())
            logits_fake = self.disc(x_hat.contiguous().detach())  # detach so that generator isn"t also updated
            d_loss = hinge_d_loss(logits_real, logits_fake)
            stats["d_loss"] = d_loss

        return x_hat, stats

    @torch.no_grad()
    def val_iter(self, x, step):
        stats = {}
        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)

        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

        return x_hat, stats

    def probabilistic(self, x):
        stats = {}

        mu, logsigma, quant_stats = self.ae.probabilistic(x)
        recon = 0.5 * torch.exp(2*torch.log(torch.abs(x - mu)) - 2*logsigma)
        if torch.isnan(recon.mean()):
            print("nan detected in probabilsitic VQGAN")
        nll = recon + logsigma + 0.5*np.log(2*np.pi)
        stats['nll'] = nll.mean(0).sum() / (np.log(2) * np.prod(x.shape[1:]))
        stats['nll_raw'] = nll.sum((1, 2, 3))
        stats['latent_ids'] = quant_stats['min_encoding_indices'].squeeze(1).reshape(x.shape[0], -1)
        x_hat = mu + 0.5*torch.exp(logsigma)*torch.randn_like(logsigma)

        return x_hat, stats
    
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, H):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head
        self.causal = True if H.sampler == 'autoregressive' else False
        if self.causal:
            block_size = np.prod(H.latent_shape)
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal and layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present
    
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        self.attn = CausalSelfAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x

class Transformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H):
        super().__init__()

        self.vocab_size = H.codebook_size + 1
        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == 'autoregressive'
        if self.causal:
            self.vocab_size = H.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.block_size, self.n_embd))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        if self.causal:
            token_embeddings = torch.cat(
                (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings),
                dim=1
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
class Sampler(nn.Module):
    def __init__(self, H, embedding_weight):
        super().__init__()
        self.latent_shape = H.latent_shape
        self.emb_dim = H.emb_dim
        self.codebook_size = H.codebook_size
        self.embedding_weight = embedding_weight
        self.embedding_weight.requires_grad = False
        self.n_samples = H.n_samples

    def train_iter(self, x, x_target, step):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def class_conditional_train_iter(self, x, y):
        raise NotImplementedError()

    def class_conditional_sample(n_samples, y):
        raise NotImplementedError()

    def embed(self, z):
        with torch.no_grad():
            z_flattened = z.view(-1, self.codebook_size)  # B*H*W, codebook_size
            embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                z.size(0),
                self.latent_shape[1],
                self.latent_shape[2],
                self.emb_dim
            ).permute(0, 3, 1, 2).contiguous()

        return embedded
    
class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        self.aux_weight = aux_weight
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))

        assert self.mask_schedule in ['random', 'fixed']

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError
        
    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id        #   set random tokens to mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1    #   set positions that are not masked to -1 - copy of x_0 with -1 where there is no mask
        return x_t, x_0_ignore, mask

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def _train_loss(self, x_0):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise

        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        return loss.mean(), vb_loss.mean()

    def sample(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_t = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id     #   initialize x_t to be an array with all mask_ids
        unmasked = torch.zeros_like(x_t, device=device).bool()     #   initialize unmasked to be an array of all False . This keeps track of which elements have been unmasked
        sample_steps = list(range(1, sample_steps+1))    #   sample_steps is a list of integers from 1 to sample_steps

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)    #   set t to be an array of the current timestep for each sample

            # where to unmask
            #   create an array of random numbers between 0 and 1 for each element in x_t
            #   if the random number is less than 1/t, then the element is unmasked
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)  
            # don't unmask somewhere already unmasked
            #   create an array of changes that are already unmasked
            #   this is done by performing a bitwise XOR on changes and unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            #   update the unmasked array with the pixels that will be unmasked this timestep (either already unmasked or newly unmasked)
            unmasked = torch.bitwise_or(unmasked, changes)

            #   Use our denoiser to predict the original input from our noisy input x_t. This x_t is being updated each iteration
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            #       Create a categorical distribution over from the logits
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            #       Sample from the distribution. Get predicted x_0 by sampling from the logits distribution (dist over which token is most likely)
            x_0_hat = x_0_dist.sample().long()
            #       Update x_t with the sampled values at the positions we are unmasking
            x_t[changes] = x_0_hat[changes]

        return x_t

    def sample_mlm(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        sample_steps = np.linspace(1, self.num_timesteps, num=sample_steps).astype(np.long)

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
            elbo += cross_entropy_loss / t
        return elbo

    def train_iter(self, x):
        loss, vb_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss}
        return stats

    # def sample_shape(self, shape, num_samples, time_steps=1000, step=1, temp=0.8):
    #     device = 'cuda'
    #     x_t = torch.ones((num_samples,) + shape, device=device).long() * self.mask_id
    #     x_lim, y_lim = shape[0] - self.shape[1], shape[1] - self.shape[2]

    #     unmasked = torch.zeros_like(x_t, device=device).bool()

    #     autoregressive_step = 0
    #     for t in tqdm(list(reversed(list(range(1, time_steps+1))))):
    #         t = torch.full((num_samples,), t, device='cuda', dtype=torch.long)

    #         unmasking_method = 'autoregressive'
    #         if unmasking_method == 'random':
    #             # where to unmask
    #             changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1).unsqueeze(-1)
    #             # don't unmask somewhere already unmasked
    #             changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
    #             # update mask with changes
    #             unmasked = torch.bitwise_or(unmasked, changes)
    #         elif unmasking_method == 'autoregressive':
    #             changes = torch.zeros(x_t.shape, device=device).bool()
    #             index = (int(autoregressive_step / shape[1]), autoregressive_step % shape[1])
    #             changes[:, index[0], index[1]] = True
    #             unmasked = torch.bitwise_or(unmasked, changes)
    #             autoregressive_step += 1

    #         # keep track of PoE probabilities
    #         x_0_probs = torch.zeros((num_samples,) + shape + (self.codebook_size,), device='cuda')
    #         # keep track of counts
    #         count = torch.zeros((num_samples,) + shape, device='cuda')

    #         # TODO: Monte carlo approximate this instead
    #         for i in range(0, x_lim+1, step):
    #             for j in range(0, y_lim+1, step):
    #                 # collect local noisy area
    #                 x_t_part = x_t[:, i:i+self.shape[1], j:j+self.shape[2]]

    #                 # increment count
    #                 count[:, i:i+self.shape[1], j:j+self.shape[2]] += 1.0

    #                 # flatten
    #                 x_t_part = x_t_part.reshape(x_t_part.size(0), -1)

    #                 # denoise
    #                 x_0_logits_part = self._denoise_fn(x_t_part, t=t)

    #                 # unflatten
    #                 x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], self.shape[2], -1)

    #                 # multiply probabilities
    #                 # for mixture
    #                 x_0_probs[:, i:i+self.shape[1], j:j+self.shape[2]] += torch.softmax(x_0_logits_part, dim=-1)

    #         # Mixture with Temperature
    #         x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
    #         C = torch.tensor(x_0_probs.size(-1)).float()
    #         x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

    #         x_0_dist = dists.Categorical(probs=x_0_probs)
    #         x_0_hat = x_0_dist.sample().long()

    #         # update x_0 where anything has been masked
    #         x_t[changes] = x_0_hat[changes]

    #     return x_t