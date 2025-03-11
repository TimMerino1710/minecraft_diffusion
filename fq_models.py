from pathlib import Path
from functools import partial
from collections import defaultdict
from collections import OrderedDict

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vision_utils
import lpips
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
import os
from einops import rearrange

def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

class FQVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        # Same initialization as original
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # reshape z -> (batch, height, width, depth, channel) and flatten
        z = torch.einsum('b c h w d -> b h w d c', z).contiguous()  # Changed permute to handle 3D
        z_flattened = z.view(-1, self.e_dim)

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)

        # Rest of the function remains the same
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w d c -> b c h w d', z_q)  # Changed permute to handle 3D

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width, depth) if channel_first else (batch, height, width, depth, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[4], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q
    
class FactorizedEMAQuantizer(nn.Module):
    def __init__(self, structure_codebook_size, style_codebook_size, emb_dim, decay=0.99):
        super().__init__()
        self.structure_codebook_size = structure_codebook_size
        self.style_codebook_size = style_codebook_size
        self.emb_dim = emb_dim
        self.decay = decay

        # Structure codebook
        self.register_buffer('structure_cluster_size', torch.zeros(structure_codebook_size))
        self.register_buffer('structure_embedding_avg', torch.zeros(structure_codebook_size, emb_dim))
        self.register_buffer('structure_embedding', torch.randn(structure_codebook_size, emb_dim))
        
        # Style codebook
        self.register_buffer('style_cluster_size', torch.zeros(style_codebook_size))
        self.register_buffer('style_embedding_avg', torch.zeros(style_codebook_size, emb_dim))
        self.register_buffer('style_embedding', torch.randn(style_codebook_size, emb_dim))

    def forward(self, z):
        # Quantize with structure codebook
        struct_encoding_indices, struct_encodings = self.quantize(
            z, 
            self.structure_embedding,
            self.structure_cluster_size,
            self.structure_embedding_avg,
            self.structure_codebook_size
        )
        
        # Quantize with style codebook
        style_encoding_indices, style_encodings = self.quantize(
            z,
            self.style_embedding,
            self.style_cluster_size,
            self.style_embedding_avg,
            self.style_codebook_size
        )

        # Calculate disentanglement loss
        struct_norm = F.normalize(struct_encodings, dim=-1)
        style_norm = F.normalize(style_encodings, dim=-1)
        disentangle_loss = torch.mean((struct_norm * style_norm).sum(-1) ** 2)

        # Average the encodings since they're in the same space
        z_q = (struct_encodings + style_encodings) / 2

        return z_q, disentangle_loss, {
            "struct_indices": struct_encoding_indices,
            "style_indices": style_encoding_indices
        }

    def quantize(self, z, embedding, cluster_size, embedding_avg, n_codes):
        # Reshape z -> (batch, height, width, depth, channel)
        z = torch.einsum('b c h w d -> b h w d c', z)
        z_flattened = z.reshape(-1, self.emb_dim)
        
        # Distances to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - \
            2 * torch.einsum('bd,nd->bn', z_flattened, embedding)
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(d, dim=1)
        encodings = F.one_hot(encoding_indices, n_codes).type_as(z_flattened)
        
        # EMA update of embeddings
        if self.training:
            n_total = encodings.sum(0)
            cluster_size.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            
            dw = torch.einsum('bn,bd->nd', encodings, z_flattened)
            embedding_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = cluster_size.sum()
            cluster_size_balanced = (cluster_size + 1e-5) / (n + n_codes * 1e-5) * n
            
            embedding.data = embedding_avg / cluster_size_balanced.unsqueeze(1)
        
        # Quantize z
        z_q = torch.matmul(encodings, embedding)
        z_q = z_q.view(z.shape)
        
        # Reshape back
        z_q = torch.einsum('b h w d c -> b c h w d', z_q)
        
        return encoding_indices, z_q

    def get_codebook_entry(self, indices, shape, codebook="structure"):
        # Select appropriate codebook
        embedding = self.structure_embedding if codebook == "structure" else self.style_embedding
        
        # Get quantized latents
        z_q = embedding[indices]

        if shape is not None:
            z_q = z_q.view(shape)

        return z_q
    
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x
    
class FactorizedAdapter(nn.Module):
    def __init__(self, down_factor):
        super().__init__()

        # Modified for 3D: grid_size now represents volume size
        self.grid_size = 24 // down_factor  # volume size // down-sample ratio
        self.width = 256  # same dim as VQ encoder output
        self.num_layers = 6
        self.num_heads = 8

        scale = self.width ** -0.5
        # Modified for 3D: positional embedding now handles cubic volume
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size ** 3, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm(self.width)

    def forward(self, x):
        # Modified for 3D: reshape from 5D to sequence
        h = x.shape[-1]  # depth dimension
        x = rearrange(x, 'b c h w d -> b (h w d) c')  # flatten 3D volume to sequence

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for transformer in self.transformer:
            x = transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        # Modified for 3D: reshape back to 5D
        x = rearrange(
            x, 
            'b (h w d) c -> b c h w d', 
            h=self.grid_size, 
            w=self.grid_size, 
            d=self.grid_size
        )

        return x
    
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
    
def normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True) #   divides the channels into 32 groups, and normalizes each group. More effective for smaller batch size than batch norm

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)   #  swish activation function, compiled using torch.jit.script. Smooth, non-linear activation function, works better than ReLu in some cases. swish (x) = x * sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_groups=32):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels, num_groups)  # Pass num_groups here
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels, num_groups)  # Pass num_groups here
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
    
def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_in = nn.Conv3d(in_channels, nf, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)

        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = nf * in_ch_mult[i_level]
            block_out = nf * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResBlock(block_in, block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResBlock(block_in, block_in))
        self.mid.append(AttnBlock(block_in))
        self.mid.append(ResBlock(block_in, block_in))


        if self.num_resolutions == 5:
            down_factor = 16
        elif self.num_resolutions == 4:
            down_factor = 8
        elif self.num_resolutions == 3:
            down_factor = 4
        else:
            raise NotImplementedError
        
        # semantic head
        self.style_head = nn.ModuleList()
        self.style_head.append(FactorizedAdapter(down_factor))

        # structural details head
        self.structure_head = nn.ModuleList()
        self.structure_head.append(FactorizedAdapter(down_factor))

        # end
        self.norm_out_style = Normalize(block_in)
        self.conv_out_style = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm_out_struct = Normalize(block_in)
        self.conv_out_struct = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
        # blocks = []
        # # Initial convolution - now 3D
        # blocks.append(
        #     nn.Conv3d(
        #         in_channels, 
        #         nf, 
        #         kernel_size=3, 
        #         stride=1, 
        #         padding=1
        #     )
        # )

        # # Residual and downsampling blocks, with attention on specified resolutions
        # for i in range(self.num_resolutions):
        #     block_in_ch = nf * in_ch_mult[i]
        #     block_out_ch = nf * ch_mult[i]
            
        #     # Add ResBlocks
        #     for _ in range(self.num_res_blocks):
        #         blocks.append(ResBlock(block_in_ch, block_out_ch))
        #         block_in_ch = block_out_ch
                
        #         # Add attention if we're at the right resolution
        #         if curr_res in attn_resolutions:
        #             blocks.append(AttnBlock(block_in_ch))

        #     # Add downsampling block if not the last resolution
        #     if i != self.num_resolutions - 1:
        #         blocks.append(Downsample(block_in_ch))
        #         curr_res = curr_res // 2

        # # Final blocks
        # blocks.append(ResBlock(block_in_ch, block_in_ch))
        # blocks.append(AttnBlock(block_in_ch))
        # blocks.append(ResBlock(block_in_ch, block_in_ch))

        # # Normalize and convert to latent size
        # blocks.append(normalize(block_in_ch))
        # blocks.append(
        #     nn.Conv3d(
        #         block_in_ch, 
        #         out_channels, 
        #         kernel_size=3, 
        #         stride=1, 
        #         padding=1
        #     )
        # )


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        h_style = h
        h_struct = h

        # style head
        for blk in self.style_head:
            h_style = blk(h_style)

        h_style = self.norm_out_style(h_style)
        h_style = nonlinearity(h_style)
        h_style = self.conv_out_style(h_style)

        # structure head
        for blk in self.structure_head:
            h_struct = blk(h_struct)

        h_struct = self.norm_out_struct(h_struct)
        h_struct = nonlinearity(h_struct)
        h_struct = self.conv_out_struct(h_struct)

        return h_style, h_struct
        # for block in self.blocks:
        #     x = block(x)
        # return x

class Generator(nn.Module):
    def __init__(self, H, z_channels=256):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = H.res_blocks
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.in_channels = H.emb_dim
        self.out_channels = H.n_channels

        block_in = self.nf * self.ch_mult[self.num_resolutions-1]


        # z to block_in
        # self.conv_in = nn.Conv3d(z_channels * 2, block_in, kernel_size=3, stride=1, padding=1)
        #TODO: trying addition instead of concat
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResBlock(block_in, block_in))
        # self.mid.append(AttnBlock(block_in))
        self.mid.append(ResBlock(block_in, block_in))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = self.nf * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResBlock(block_in, block_out))
                block_in = block_out
                # if i_level == self.num_resolutions - 1:
                #     attn_block.append(AttnBlock(block_in))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, H.n_channels, kernel_size=3, stride=1, padding=1)


    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        #TODO: trying addition instead of concat
        B, C, H, W, D = z.shape
        z_style = z[:, :C//2]  # First half of channels
        z_struct = z[:, C//2:]  # Second half of channels
        
        # Add the vectors
        z_add = z_style + z_struct

        # z to block_in
        h = self.conv_in(z_add)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

class TwoStageGenerator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = H.res_blocks
        self.combine_method = H.combine_method
        

        # First stage: structure codes -> binary map
        block_in = self.nf * self.ch_mult[self.num_resolutions-1]
        
        # Structure decoder (similar to current Generator but outputs single channel)
        self.struct_conv_in = nn.Conv3d(H.z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # Middle blocks for structure
        self.struct_mid = nn.ModuleList([
            ResBlock(block_in, block_in),
            ResBlock(block_in, block_in)
        ])
        
        # Upsampling blocks for structure
        self.struct_up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            up_block = nn.Module()
            res_block = nn.ModuleList()
            block_out = self.nf * self.ch_mult[i_level]
            
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResBlock(block_in, block_out))
                block_in = block_out
                
            if i_level != 0:
                up_block.upsample = Upsample(block_in)
            
            up_block.res = res_block
            self.struct_up_blocks.append(up_block)
            
        # Final layers for binary output
        self.struct_norm_out = Normalize(block_in)
        self.struct_conv_out = nn.Conv3d(block_in, 1, kernel_size=3, stride=1, padding=1)
        
       # Calculate combined channels
        if self.combine_method == 'concat':
            combined_channels = 1 + H.z_channels  # 33 channels
        else:  # multiply
            combined_channels = block_in

        # Just use a single conv layer to map from combined_channels to block_in
        self.initial_conv = nn.Conv3d(combined_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Then process with regular blocks using standard grouping
        self.final_blocks = nn.ModuleList([
            ResBlock(block_in, block_in),  # Now everything uses block_in channels
            ResBlock(block_in, block_in)
        ])

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, H.n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # Split input into structure and style codes
        B, C, H, W, D = z.shape
        z_struct = z[:, :C//2]  # First half of channels
        z_style = z[:, C//2:]   # Second half of channels
        
        # Structure path
        h_struct = self.struct_conv_in(z_struct)
        
        for mid_block in self.struct_mid:
            h_struct = mid_block(h_struct)
            
        for block in self.struct_up_blocks:
            for res_block in block.res:
                h_struct = res_block(h_struct)
            if hasattr(block, 'upsample'):
                h_struct = block.upsample(h_struct)
                
        h_struct = self.struct_norm_out(h_struct)
        h_struct = nonlinearity(h_struct)
        binary_out = self.struct_conv_out(h_struct)
        binary_out = torch.sigmoid(binary_out)  # Convert to probabilities
        
        # Simply upsample style codes to match spatial dimensions
        h_style = F.interpolate(z_style, size=(24, 24, 24), mode='trilinear', align_corners=False)
        
        # After combining binary_out and style
        if self.combine_method == 'concat':
            h_combined = torch.cat([binary_out, h_style], dim=1)
        else:  # multiply
            h_combined = binary_out * h_style

        # First map to block_in channels
        h_combined = self.initial_conv(h_combined)

        # Then process with regular blocks
        for block in self.final_blocks:
            h_combined = block(h_combined)
            
        h_combined = self.norm_out(h_combined)
        h_combined = nonlinearity(h_combined)
        out = self.conv_out(h_combined)
        
        return out, binary_out  # Return both final output and binary reconstruction

class PatchGAN3DDiscriminator(nn.Module):
    """3D PatchGAN discriminator adapted for Minecraft voxel data"""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        Parameters:
            input_nc (int)  -- number of input channels (block types)
            ndf (int)       -- number of filters in first conv layer
            n_layers (int)  -- number of conv layers
        """
        super().__init__()
        norm_layer = nn.BatchNorm3d

        
        use_bias = norm_layer != nn.BatchNorm3d

        kw = 4  # kernel size
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.main = nn.Sequential(*sequence)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv3d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        return self.main(input)
    
class Discriminator3D(nn.Module):
    def __init__(self, input_nc=43, ndf=64, n_layers=3):
        """Simple 3D convolutional discriminator
        
        Args:
            input_nc (int): Number of input channels (number of block types)
            ndf (int): Number of filters in first conv layer
            n_layers (int): Number of conv layers
        """
        super().__init__()
        
        # Initial convolution
        layers = [
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Increasing number of filters with each layer
        current_channels = ndf
        for i in range(n_layers - 1):
            next_channels = min(current_channels * 2, 512)
            layers.extend([
                nn.Conv3d(current_channels, next_channels, 
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_channels = next_channels
        
        # Final layers
        layers.extend([
            nn.Conv3d(current_channels, current_channels,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm3d(current_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(current_channels, 1, kernel_size=4, stride=1, padding=1)
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
                where C is number of block types (one-hot encoded)
        Returns:
            Tensor of shape [B, 1, D', H', W'] containing realness scores
        """
        return self.model(x)

class BiomeClassifier(nn.Module):
    def __init__(self, num_block_types, num_biomes, feature_dim=256):
        super(BiomeClassifier, self).__init__()
        
        # Initial embedding layer for one-hot encoded blocks
        self.block_proj = nn.Conv3d(num_block_types, 64, kernel_size=1)
        
        # Encoder layers that will downsample to 6x6x6 spatial dimensions
        self.encoder = nn.Sequential(
            # Layer 1: 24x24x24 -> 12x12x12
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            # Layer 2: 12x12x12 -> 6x6x6
            nn.Conv3d(128, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(),
        )
        
        # Self-attention block at 6x6x6 resolution
        self.attention = AttnBlock(feature_dim)
        
        # Additional convolution after attention for feature extraction
        self.feature_conv = nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1)
        
        # Biome prediction head
        self.biome_head = nn.Conv3d(feature_dim, num_biomes, kernel_size=1)
        
        # Upsampling layer to get back to original resolution
        self.upsample = nn.Upsample(size=(24, 24, 24), mode='trilinear', align_corners=False)
        
    def get_intermediate_features(self, x):
        """Get the features after attention for representation learning"""
        x = self.block_proj(x)
        x = self.encoder(x)
        x = self.attention(x)
        return self.feature_conv(x)
        
    def forward(self, x, return_features=False):
        # x shape: (batch_size, num_blocks, 24, 24, 24)
        
        # Initial projection and encoding
        x = self.block_proj(x)
        x = self.encoder(x)  # (batch_size, feature_dim, 6, 6, 6)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Get features for representation learning
        features = self.feature_conv(x)
        
        # Predict biomes and upsample
        biome_logits = self.biome_head(features)  # (batch_size, num_biomes, 6, 6, 6)
        biome_logits = self.upsample(biome_logits)  # (batch_size, num_biomes, 24, 24, 24)
        
        if return_features:
            return biome_logits, features
        return biome_logits

class BiomeFeatureModel(nn.Module):
    def __init__(self, biome_classifier_path):
        super().__init__()
        self.biome_classifier = BiomeClassifier(
            num_block_types=43,
            num_biomes=14,
            feature_dim=256
        ).cuda()
        
        # Load pretrained weights for biome classifier
        self.biome_classifier.load_state_dict(torch.load(biome_classifier_path))
        self.biome_classifier.eval()
        for param in self.biome_classifier.parameters():
            param.requires_grad = False


    # @torch.no_grad()
    def forward(self, inputs, style_features):
        # Get biome features from real input
        biome_features = self.biome_classifier.get_intermediate_features(inputs)  # [B, C, 6, 6, 6]
        B, C, H, W, D = biome_features.shape
        biome_features = biome_features.permute(0, 2, 3, 4, 1)  # [B, 6, 6, 6, C]
        biome_features = biome_features.reshape(B, H*W*D, C)    # [B, 216, C]
        
        # Normalize features
        # biome_features = F.normalize(biome_features, dim=-1).detach()
        # style_features = F.normalize(style_features, dim=-1)

        biome_features = biome_features.detach()

        # Print feature statistics
        # print("Biome features mean/std:", biome_features.mean().item(), biome_features.std().item())
        # print("Style features mean/std:", style_features.mean().item(), style_features.std().item())
        
        # # Simple cosine similarity loss
        loss = 1 - F.cosine_similarity(style_features, biome_features, dim=-1).mean()
        # or MSE loss
        # loss = F.mse_loss(style_features, biome_features)
        return loss
    # @torch.no_grad()
    # def forward(self, inputs, style_features):
    #     # Extract biome features from real input using pretrained classifier
    #     biome_features = self.biome_classifier.get_intermediate_features(inputs)  # [B, C, 6, 6, 6]
    #     B, C, H, W, D = biome_features.shape
    #     biome_features = biome_features.permute(0, 2, 3, 4, 1)  # [B, 6, 6, 6, C]
    #     biome_features = biome_features.reshape(B, H*W*D, C)    # [B, 216, C]
        
    #     # Normalize both features
    #     biome_features = F.normalize(biome_features, p=2, dim=-1).detach()  # Fixed target
    #     style_features = F.normalize(style_features, p=2, dim=-1)  # These will be optimized
        
    #     # Compute similarity matrix and InfoNCE loss
    #     loss_mat = torch.bmm(style_features, biome_features.transpose(1, 2))
    #     loss_mat = loss_mat.exp()
        
    #     loss_diag = torch.diagonal(loss_mat, dim1=1, dim2=2)  # [B, 216]
    #     loss_denom = loss_mat.sum(dim=2)  # [B, 216]
        
    #     loss_InfoNCE = -(loss_diag / loss_denom).log().mean()
        
    #     return loss_InfoNCE
    # @torch.no_grad()
    # def forward(self, inputs, semantic_feat):
    #     # Get features from biome classifier (B, C, 6, 6, 6)
    #     real_features = self.biome_classifier.get_intermediate_features(inputs)
        
    #     # Print shapes for debugging
    #     print("Before reshape:")
    #     print("real_features:", real_features.shape)
    #     print("semantic_feat:", semantic_feat.shape)
        
    #     # Reshape to (B, N, C) where N = 6*6*6
    #     B, C, H, W, D = real_features.shape
    #     real_features = real_features.view(B, C, -1).permute(0, 2, 1)  # B, 216, C
        
    #     print("After reshape:")
    #     print("real_features:", real_features.shape)
    #     print("semantic_feat:", semantic_feat.shape)
        
    #     # Normalize both feature sets
    #     real_features = F.normalize(real_features, p=2, dim=-1)
    #     semantic_feat = F.normalize(semantic_feat, p=2, dim=-1)
        
    #     # Matrix multiplication should be:
    #     # (B, 216, C) @ (B, 216, C).transpose(-2, -1) -> (B, 216, 216)
    #     loss_mat = (semantic_feat @ real_features.detach().mT)  # Use mT and proper dimensions
    #     loss_diag = loss_mat.diag()
    #     loss_denom = loss_mat.sum(1)
    #     loss_InfoNCE = -(loss_diag / loss_denom).log().mean()
        
    #     return loss_InfoNCE


class VQLossDualCodebook(nn.Module):
    def __init__(self, H):
        super().__init__()
        # Discriminator parameters
        if H.disc_type == 'conv':
            # Use simpler 3D discriminator
            self.discriminator = Discriminator3D(
                input_nc=H.n_channels,  # number of block types
                ndf=H.ndf  # base number of filters
            ).cuda()
        elif H.disc_type == 'patch':
            self.discriminator = PatchGAN3DDiscriminator(
                input_nc=H.n_channels,
                ndf=H.ndf,
                n_layers=H.disc_layers
            ).cuda()
        
        # Initialize BiomeClassifier for feature extraction
        self.biome_feature_model = BiomeFeatureModel(H.biome_classifier_path)
        
        # Loss weights and parameters
        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.disc_weight_min = 0.0
        self.disc_adaptive_weight = H.disc_adaptive_weight
        self.reconstruction_weight = H.reconstruction_weight
        self.codebook_weight = H.codebook_weight
        self.biome_weight = H.biome_weight
        self.disentanglement_ratio = H.disentanglement_ratio
        self.binary_recon_weight = H.binary_reconstruction_weight
        
        # Loss functions
        self.disc_loss = non_saturating_d_loss
        self.gen_loss = hinge_gen_loss

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        """Gradually adopt weight after threshold step"""
        if global_step < threshold:
            weight = value
        return weight
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """Dynamically adjust discriminator weight to balance with other losses"""
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, self.disc_weight_min, self.disc_weight_max).detach()
            return d_weight
        return 1.0

    
    def compute_biome_feature_loss(self, inputs, reconstructions):
        """Compute feature matching loss using BiomeClassifier's intermediate features"""
        with torch.no_grad():
            real_features = self.biome_classifier.get_intermediate_features(inputs)
        fake_features = self.biome_classifier.get_intermediate_features(reconstructions)
        
        # Normalize features
        real_features = F.normalize(real_features, p=2, dim=1)
        fake_features = F.normalize(fake_features, p=2, dim=1)
        
        # Compute feature matching loss
        feature_loss = F.mse_loss(fake_features, real_features)
        
        return feature_loss

    def forward(self, codebook_loss_style, codebook_loss_struct,
                inputs, reconstructions, disentangle_loss, biome_feat,
                optimizer_idx, global_step, last_layer=None, binary_out=None, binary_target=None):
        
        if optimizer_idx == 0:
            rec_loss = F.cross_entropy(
                reconstructions.contiguous(), 
                torch.argmax(inputs, dim=1).contiguous()
            ) * self.reconstruction_weight

            # Binary reconstruction loss (if in two-stage mode)
            binary_recon_loss = 0.0
            if binary_out is not None and binary_target is not None:
                binary_recon_loss = F.binary_cross_entropy(
                    binary_out.squeeze(1),
                    binary_target
                ) * self.binary_recon_weight
                        
            # Codebook losses
            style_loss = sum(codebook_loss_style[:3]) * self.codebook_weight
            struct_loss = sum(codebook_loss_struct[:3]) * self.codebook_weight
            
            # Biome feature loss using InfoNCE
            if biome_feat is not None:
                biome_feat_loss = self.biome_feature_model(inputs.contiguous(), biome_feat)
                biome_feat_loss = self.biome_weight * biome_feat_loss
            else:
                biome_feat_loss = 0.0
            
            # Disentanglement loss
            disent_loss = self.disentanglement_ratio * disentangle_loss if disentangle_loss is not None else 0.0
            
            # Generator adversarial loss with adaptive weight
            disc_weight = self.adopt_weight(self.disc_weight_max, global_step, 
                                          threshold=self.disc_start_step, value=0.0)
            
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = self.gen_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = rec_loss + biome_feat_loss + binary_recon_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, g_loss, last_layer)
                g_loss = g_loss * disc_weight * disc_adaptive_weight
            else:
                g_loss = g_loss * disc_weight
            
            # Total loss
            loss = rec_loss + style_loss + struct_loss + biome_feat_loss + disent_loss + g_loss + binary_recon_loss
            
            return {
                'loss': loss,
                'rec_loss': rec_loss,
                'binary_rec_loss': binary_recon_loss,
                'style_loss': style_loss,
                'struct_loss': struct_loss,
                'biome_feat_loss': biome_feat_loss,
                'disent_loss': disent_loss,
                'g_loss': g_loss,
                'disc_weight': disc_weight,
                'disc_adaptive_weight': disc_adaptive_weight if self.disc_adaptive_weight else 1.0,
                'codebook_usage_style': codebook_loss_style[3],
                'codebook_usage_struct': codebook_loss_struct[3]
            }
            
        # Discriminator update
        elif optimizer_idx == 1:
            # Get discriminator predictions
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            # Calculate discriminator loss with weight adoption
            disc_weight = self.adopt_weight(self.disc_weight_max, global_step, 
                                          threshold=self.disc_start_step, value=0.0)
            d_loss = self.disc_loss(logits_real, logits_fake) * disc_weight
            
            return {
                'd_loss': d_loss,
                'logits_real': logits_real.mean(),
                'logits_fake': logits_fake.mean(),
                'disc_weight': disc_weight
            }
        

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))

    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_gen_loss(logits_fake):
    return -torch.mean(logits_fake)

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class FeatPredHead(nn.Module):
    def __init__(self, input_dim=256, down_factor=16):
        super().__init__()
        self.grid_size = 24 // down_factor
        self.width = 256
        self.num_layers = 3
        self.num_heads = 8

        self.upscale = nn.Sequential(
            nn.Linear(input_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width)
        )

        scale = self.width ** -0.5
        # Remove class embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size ** 3, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm(self.width)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        x = self.upscale(x)

        # No class token addition
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x  # Shape will be [B, 216, C]
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class FeatPredHead(nn.Module):
    def __init__(self, input_dim=256, down_factor=16):
        super().__init__()
        self.grid_size = 24 // down_factor
        self.width = 256
        self.num_layers = 3
        self.num_heads = 8

        self.upscale = nn.Sequential(
            nn.Linear(input_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width)
        )

        scale = self.width ** -0.5
        # Remove class embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size ** 3, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm(self.width)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        x = self.upscale(x)

        # No class token addition
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x  # Shape will be [B, 216, C]
    
class FQModel(nn.Module):
    def __init__(self, H):
        super().__init__()
        # Basic parameters
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.struct_codebook_size = H.struct_codebook_size
        self.style_codebook_size = H.style_codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.resolution = H.img_size
        self.z_channels = H.z_channels
        self.with_biome_supervision = H.with_biome_supervision
        self.with_disentanglement = H.with_disentanglement
        self.disentanglement_ratio = H.disentanglement_ratio
        self.two_stage_decoder = H.two_stage_decoder
        
        # Two head encoder
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution
        )

        # Quantizer for style head (semantic)
        self.quantize_style = FQVectorQuantizer(
            # self.codebook_size, 
            self.style_codebook_size, 
            self.embed_dim,
            H.beta, 
            H.entropy_loss_ratio,
            H.codebook_l2_norm, 
            H.codebook_show_usage
        )
        self.quant_conv_style = nn.Conv3d(self.z_channels, self.embed_dim, 1)

        # Quantizer for structural head (visual)
        self.quantize_struct = FQVectorQuantizer(
            self.struct_codebook_size, 
            self.embed_dim,
            H.beta, 
            H.entropy_loss_ratio,
            H.codebook_l2_norm, 
            H.codebook_show_usage
        )
        self.quant_conv_struct = nn.Conv3d(self.z_channels, self.embed_dim, 1)

        # Pixel decoder
        input_dim = self.embed_dim * 2  # Combined dimension from both codebooks
        self.post_quant_conv = nn.Conv3d(input_dim, self.z_channels, 1)
        # self.decoder = Generator(H, z_channels=self.z_channels)
        # Choose decoder type based on hyperparameter
        if H.two_stage_decoder:
            self.decoder = TwoStageGenerator(H)
        else:
            self.decoder = Generator(H, z_channels=self.z_channels)

        # Determine downsampling factor
        if self.num_resolutions == 5:
            down_factor = 16
        elif self.num_resolutions == 4:
            down_factor = 8
        elif self.num_resolutions == 3:
            down_factor = 4
        else:
            raise NotImplementedError

        # Biome prediction head for style representation learning
        if H.with_biome_supervision:
            print("Include feature prediction head for biome supervision")
            self.feat_pred_head = FeatPredHead(input_dim=self.embed_dim, down_factor=down_factor)
        else:
            print("NO biome supervision")

        if H.with_disentanglement:
            print("Disentangle Ratio: ", H.disentanglement_ratio)
        else:
            print("No Disentangle Regularization")

    def compute_disentangle_loss(self, quant_struct, quant_style):
        # Reshape from 5D to 2D
        quant_struct = rearrange(quant_struct, 'b c h w d -> (b h w d) c')
        quant_style = rearrange(quant_style, 'b c h w d -> (b h w d) c')

        # Normalize the vectors
        quant_struct = F.normalize(quant_struct, p=2, dim=-1)
        quant_style = F.normalize(quant_style, p=2, dim=-1)

        # Compute dot product and loss
        dot_product = torch.sum(quant_struct * quant_style, dim=1)
        loss = torch.mean(dot_product ** 2) * self.disentanglement_ratio

        return loss

    def forward(self, input):
        # Get both style and structure encodings
        h_style, h_struct = self.encoder(input)
        h_style = self.quant_conv_style(h_style)
        h_struct = self.quant_conv_struct(h_struct)

        # Quantize both paths
        quant_style, emb_loss_style, _ = self.quantize_style(h_style)
        quant_struct, emb_loss_struct, _ = self.quantize_struct(h_struct)

        # Biome feature prediction if enabled
        if self.with_biome_supervision:
            style_feat = self.feat_pred_head(quant_style)
        else:
            style_feat = None

        # Compute disentanglement loss if enabled
        if self.with_disentanglement:
            disentangle_loss = self.compute_disentangle_loss(quant_struct, quant_style)
        else:
            disentangle_loss = 0

        # Combine quantized representations and decode
        # quant = torch.cat([quant_struct, quant_style], dim=1)
        # dec = self.decoder(quant)

        # Combine quantized representations and decode
        quant = torch.cat([quant_struct, quant_style], dim=1)
        
        if self.two_stage_decoder:
            dec, binary_out = self.decoder(quant)
            # Create binary target (you'll need to implement this based on your data)
            # binary_target = self.create_binary_target(input)
            # binary_loss = F.binary_cross_entropy(binary_out, binary_target)
            return dec, binary_out,  emb_loss_style, emb_loss_struct, disentangle_loss, style_feat
        else:
            dec = self.decoder(quant)
            return dec, emb_loss_style, emb_loss_struct, disentangle_loss, style_feat
        
        # return dec, emb_loss_style, emb_loss_struct, disentangle_loss, style_feat
    
class HparamsBase(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

class HparamsBase(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

class HparamsFQGAN(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        
        if self.dataset == 'minecraft':
            # Existing parameters
            self.batch_size = 8
            self.img_size = 24
            self.n_channels = 43
            self.nf = 64
            self.ndf = 64
            self.res_blocks = 2
            self.latent_shape = [1, 6, 6, 6]
            
            # New parameters for dual codebook architecture
            self.struct_codebook_size = 32  # Size of each codebook
            self.style_codebook_size = 20  # Size of each codebook
            self.emb_dim = 32  # Embedding dimension
            self.z_channels = 32  # Bottleneck channels
            self.ch_mult = [1, 2, 4]  # Channel multipliers for progressive downsampling
            self.num_resolutions = len(self.ch_mult)
            self.attn_resolutions = [6]  # Resolutions at which to apply attention
            
            # Loss weights and parameters
            self.disc_type = 'conv'
            self.disc_weight_max = 0.5  # Weight for discriminator loss
            self.disc_weight_min = 0.0  # Weight for discriminator loss
            self.disc_adaptive_weight = True  # Enable adaptive weighting
            self.disc_start_step = 7500  # Step to start discriminator training
            self.reconstruction_weight = 1.0  # Weight for reconstruction loss
            self.codebook_weight = 1.0  # Weight for codebook loss
            self.biome_weight = 1.0  # Weight for biome feature prediction
            self.disentanglement_ratio = 0.5  # Weight for disentanglement loss
            
            # Codebook specific parameters
            self.beta = 0.5  # Commitment loss coefficient
            self.entropy_loss_ratio = 0.05  # For codebook entropy regularization
            self.codebook_l2_norm = True  # Whether to L2 normalize codebook entries
            self.codebook_show_usage = True  # Track codebook usage statistics
            
            # Training parameters
            self.lr = 1e-4  # Learning rate
            self.beta1 = 0.9  # Adam beta1
            self.beta2 = 0.95  # Adam beta2
            self.disc_layers = 3  # Number of discriminator layers
            self.train_steps = 10000
            self.start_step = 0
            
            self.transformer_dim = self.emb_dim  # Make transformer dim match embedding dim
            self.num_heads = 8  # Number of attention heads
            
            # Feature prediction parameters
            self.with_biome_supervision = True  # Enable biome feature prediction
            self.with_disentanglement = True  # Enable disentanglement loss
            
            # Logging parameters (if not already present)
            self.steps_per_log = 150
            self.steps_per_checkpoint = 500
            self.steps_per_display_output = 500
            self.steps_per_save_output = 500
            self.steps_per_validation = 150
            self.val_samples_to_save = 16
            self.val_samples_to_display = 4
            self.visdom_port = 8097
            
            # Two stage decoder stuff
            self.binary_reconstruction_weight = 1
            self.two_stage_decoder = True
            self.combine_method = 'concat'

            self.num_biomes = 11  # Number of biome classes
            self.biome_feat_dim = 256  # Dimension of biome features
            self.biome_classifier_path = 'best_biome_classifier_airprocessed.pt'
        else:
            raise KeyError(f'Defaults not defined for dataset: {self.dataset}')