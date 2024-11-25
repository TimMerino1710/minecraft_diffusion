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
#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        # Add tracking for usage
        self.register_buffer('usage_count', torch.zeros(self.codebook_size))
        self.register_buffer('last_reset', torch.zeros(self.codebook_size))
        self.reset_threshold = 1  # Number of epochs before resetting unused vectors

    def reset_unused_codes(self, epoch):
        """Reset any embeddings that weren't used in the last epoch"""
        unused_indices = torch.where(self.usage_count == 0)[0]
    
        if len(unused_indices) > 0:
            # Reinitialize unused embeddings
            self.embedding.weight.data[unused_indices].uniform_(
            -1.0 / self.codebook_size, 
            1.0 / self.codebook_size
            )
            
            # Reset usage count for next epoch
            self.usage_count.zero_()  # Reset all counts for next epoch
            
            return len(unused_indices)
        return 0

    def forward(self, z):
        # z shape: (batch, channel, height, width, depth)
        # Reshape z -> (batch, height, width, depth, channel) and flatten
        z = z.permute(0, 2, 3, 4, 1).contiguous()
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
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

         # Update usage counts
        unique_indices = torch.unique(min_encoding_indices)
        self.usage_count[unique_indices] += 1
        
        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
            }

    def get_codebook_entry(self, indices, shape):
        # indices shape: (batch_size * height * width * depth)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)
        
        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        
        if shape is not None:
            # Reshape to (batch, height, width, depth, emb_dim)
            z_q = z_q.view(shape)
            # Permute to (batch, emb_dim, height, width, depth)
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            
        return z_q

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
        b, c, h, w, d = z.shape  # Now includes depth dimension
        z_flattened = z.permute(0, 2, 3, 4, 1).reshape(-1, self.emb_dim)
        
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
        z_q = z_q.view(b, h, w, d, c)  # Added depth dimension
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, D]
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Calculate perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, torch.tensor(0.0, device=z.device), {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices.view(b, h, w, d),
            "mean_distance": dist.mean()
        }

    def get_codebook_entry(self, indices, shape):
        min_encodings = F.one_hot(indices, self.codebook_size).type(torch.float)
        z_q = torch.matmul(min_encodings, self.embedding)

        if shape is not None:
            z_q = z_q.view(shape).permute(0, 4, 1, 2, 3).contiguous()

        return z_q
    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)  # Padding for all 3 dimensions
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0) #   padding the right and the bottom with 0s
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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
        # Convert all 2D convolutions to 3D
        self.q = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv3d(
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
        b, c, h, w, d = q.shape
        q = q.reshape(b, c, h*w*d)    # Flatten all spatial dimensions
        q = q.permute(0, 2, 1)        # b, hwd, c
        k = k.reshape(b, c, h*w*d)    # b, c, hwd
        w_ = torch.bmm(q, k)          # b, hwd, hwd    
        w_ = w_ * (int(c)**(-0.5))    # Scale dot products
        w_ = F.softmax(w_, dim=2)     # Softmax over spatial positions

        # attend to values
        v = v.reshape(b, c, h*w*d)
        w_ = w_.permute(0, 2, 1)      # b, hwd, hwd (first hwd of k, second of q)
        h_ = torch.bmm(v, w_)         # b, c, hwd
        h_ = h_.reshape(b, c, h, w, d) # Restore spatial structure

        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # Initial convolution - now 3D
        blocks.append(
            nn.Conv3d(
                in_channels, 
                nf, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        )

        # Residual and downsampling blocks, with attention on specified resolutions
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            
            # Add ResBlocks
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                
                # Add attention if we're at the right resolution
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            # Add downsampling block if not the last resolution
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # Final blocks
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # Normalize and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(
            nn.Conv3d(
                block_in_ch, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)

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

        print(f'resolution: {self.resolution}, num_resolutions: {self.num_resolutions}, '
              f'num_res_blocks: {self.num_res_blocks}, attn_resolutions: {self.attn_resolutions}, '
              f'in_channels: {self.in_channels}, out_channels: {self.out_channels}, '
              f'block_in_ch: {block_in_ch}, curr_res: {curr_res}')

        blocks = []
        # Initial conv - now 3D
        blocks.append(nn.Conv3d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # Non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # Upsampling blocks
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

        # Final processing
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv3d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # Used for calculating ELBO - fine tuned after training
        self.logsigma = nn.Sequential(
            nn.Conv3d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
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

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_layers=3):
        """
        Parameters:
            nc (int): Number of input channels (block types)
            ndf (int): Number of discriminator filters in first conv layer
            n_layers (int): Number of conv layers
        """
        super().__init__()

        layers = [
            # Initial layer
            nn.Conv3d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        # Gradually increase the number of filters
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv3d( ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers
        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv3d( ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer - produces one value per 3D patch
        layers += [
            nn.Conv3d( ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input of shape [B, C, H, W, D]
        Returns:
            Tensor: Discriminator scores for each 3D patch
        """
        return self.main(x)

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
        elif self.quantizer_type == "ema": 
            self.quantize = EMAQuantizer(self.codebook_size, self.embed_dim)
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
        )
        self.perceptual_weight = 0.0

        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.diff_aug = H.diff_aug
        self.policy = "translation"

        # print("Discriminator")
        # print(summary(self.disc, (H.batch_size, H.n_channels, H.img_size, H.img_size)))

    def train_iter(self, x, step):  #   editing this to remove diff aug and gumbel
        stats = {}
        x_hat, codebook_loss, quant_stats = self.ae(x)
        
        # print(f'x size: {x.size()}')
        # print(f'x_hate size: {x_hat.size()}')

        #TODO: This is used for images, swapped to CE for minecraft data
        # get recon/perceptual loss
        # recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        # nll_loss = recon_loss

        # Apply softmax across block type dimension
        # x_hat_probs = F.softmax(x_hat, dim=1)
        
        # # Cross entropy loss (x is already one-hot encoded)
        # recon_loss = F.cross_entropy(
        #     x_hat.view(-1, x_hat.size(1)),  # [B*H*W*D, num_blocks]
        #     torch.argmax(x, dim=1).view(-1)  # [B*H*W*D]
        # )
        
        recon_loss = F.cross_entropy(
            x_hat.contiguous(),  # [B*H*W*D, num_blocks]
            torch.argmax(x.contiguous(), dim=1)  # [B*H*W*D]
        )
        
        nll_loss = recon_loss
        
        nll_loss = torch.mean(nll_loss)


        # update generator
        logits_fake = self.disc(x_hat)
        # logits_fake = self.disc(x_hat_probs)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = 0.0
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