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
from torch.nn.utils import spectral_norm

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
    
class FQEMAVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, decay=0.99, eps=1e-5, l2_norm=False):
        super().__init__()
        self.n_e = n_e  # Number of embeddings
        self.e_dim = e_dim  # Embedding dimension
        self.decay = decay  # EMA decay factor (higher = slower updates)
        self.eps = eps  # Small constant for numerical stability
        self.l2_norm = l2_norm  # Whether to normalize embeddings
        
        # Initialize embeddings - use uniform initialization like in FQVectorQuantizer
        embedding = torch.randn(n_e, e_dim)
        if l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        # Initialize EMA tracking variables
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('embedding', embedding)
        self.register_buffer('codebook_used', torch.zeros(65536))  # For usage tracking

    def forward(self, z):
        # Reshape z -> (batch, height, width, depth, channel) and flatten
        z = torch.einsum('b c h w d -> b h w d c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Apply L2 normalization if specified
        if self.l2_norm:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding, p=2, dim=-1)
        else:
            embedding = self.embedding
        
        # Compute distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - \
            2 * torch.einsum('bd,nd->bn', z_flattened, embedding)
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(d, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_e).type_as(z_flattened)
        
        # Track codebook usage
        if self.training:
            cur_len = encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
        else:
            codebook_usage = 0
        
        # EMA update of embeddings - only in training mode
        if self.training:
            n_total = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            
            # Update embedding average
            dw = torch.matmul(encodings.t(), z_flattened)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Update the embedding with EMA
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            
            # Update embedding weights with normalized averages
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
            
            # Apply L2 norm to updated embeddings if needed
            if self.l2_norm:
                self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        # Get quantized latent vectors
        z_q = torch.matmul(encodings, embedding)
        z_q = z_q.view(z.shape)
        
        # IMPORTANT: Implement straight-through estimator
        # This is crucial for gradient flow in the encoder
        z_q = z + (z_q - z).detach()
        
        # Reshape back to match original input shape
        z_q = torch.einsum('b h w d c -> b c h w d', z_q)
        
        # Return in the same format as FQVectorQuantizer, but with zeros for losses
        # The EMA update doesn't need explicit gradient-based losses
        vq_loss = torch.tensor(0.0, device=z.device)
        commit_loss = torch.tensor(0.0, device=z.device)
        entropy_loss = torch.tensor(0.0, device=z.device)
        
        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (None, None, encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # Get quantized latents
        embedding = self.embedding
        if self.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)
            
        z_q = embedding[indices]

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[4], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            else:
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
    def __init__(self, resolution, down_factor):
        super().__init__()

        # Modified for 3D: grid_size now represents volume size
        self.grid_size = resolution // down_factor  # volume size // down-sample ratio
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
    def __init__(self, in_channels, padding_mode='constant'):
        super().__init__()
        self.padding_mode = padding_mode
        self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)  # Padding for all 3 dimensions
        # x = torch.nn.functional.pad(x, pad, mode="constant", value=0) #   padding the right and the bottom with 0s
        x = torch.nn.functional.pad(x, pad, mode=self.padding_mode) #   padding the right and the bottom with 0s
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

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
    def __init__(self, in_channels, out_channels=None, num_groups=32, padding_mode='zeros'):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels, num_groups)  # Pass num_groups here
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.norm2 = normalize(out_channels, num_groups)  # Pass num_groups here
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
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
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, padding_mode='zeros'):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_in = nn.Conv3d(in_channels, nf, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

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
                res_block.append(ResBlock(block_in, block_out, padding_mode=padding_mode))
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
        self.mid.append(ResBlock(block_in, block_in, padding_mode=padding_mode))
        self.mid.append(AttnBlock(block_in))
        self.mid.append(ResBlock(block_in, block_in, padding_mode=padding_mode))


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
        self.style_head.append(FactorizedAdapter(self.resolution, down_factor))

        # structural details head
        self.structure_head = nn.ModuleList()
        self.structure_head.append(FactorizedAdapter(self.resolution, down_factor))

        # end
        self.norm_out_style = Normalize(block_in)
        self.conv_out_style = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.norm_out_struct = Normalize(block_in)
        self.conv_out_struct = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
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
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in))
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
        self.resolution = H.img_size
        self.detach = H.detach_binary_recon

        if self.detach:
            print("Detaching binary reconstruction from comp graph for final loss")

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
        h_style = F.interpolate(z_style, size=(self.resolution, self.resolution, self.resolution), mode='trilinear', align_corners=False)
        
        if self.detach:
            detached_binary_out = binary_out.detach()
            # --- Combine (Identical) ---
            if self.combine_method == 'concat':
                h_combined = torch.cat([detached_binary_out, h_style], dim=1)
            else:
                h_combined = detached_binary_out * h_style
        else:
            if self.combine_method == 'concat':
                h_combined = torch.cat([binary_out, h_style], dim=1)
            else:
                h_combined = binary_out * h_style
        # # After combining binary_out and style
        # if self.combine_method == 'concat':
        #     h_combined = torch.cat([binary_out, h_style], dim=1)
        # else:  # multiply
        #     h_combined = binary_out * h_style

        # First map to block_in channels
        h_combined = self.initial_conv(h_combined)

        # Then process with regular blocks
        for block in self.final_blocks:
            h_combined = block(h_combined)
            
        h_combined = self.norm_out(h_combined)
        h_combined = nonlinearity(h_combined)
        out = self.conv_out(h_combined)
        
        return out, binary_out  # Return both final output and binary reconstruction

class DumbTwoStageGenerator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        # Note: H.res_blocks is used for the final stage, keep it.
        self.num_res_blocks = H.res_blocks
        self.combine_method = H.combine_method
        self.resolution = H.img_size
        self.emb_dim = H.emb_dim # Dimension of z_struct / z_style

        # --- Simplified First Stage ---
        # Directly map upsampled structure codes to binary output
        # Input channels = H.emb_dim (dimension of z_struct)
        # Output channels = 1
        self.struct_to_binary_conv = nn.Conv3d(
            self.emb_dim, 1, kernel_size=3, stride=1, padding=1
        )
        # Removed: struct_conv_in, struct_mid, struct_up_blocks, struct_norm_out, struct_conv_out

        # --- Second Stage (Identical Logic to Original) ---
        # This defines the channel depth expected by the final ResBlocks
        # Uses the same logic as the original to determine this depth
        block_in_final_stage = self.nf * self.ch_mult[self.num_resolutions-1]

        # Calculate combined channels based on combine_method and input dims
        if self.combine_method == 'concat':
            # 1 channel from binary_out + emb_dim channels from h_style
            combined_channels = 1 + self.emb_dim
        else:  # multiply
            # binary_out * h_style -> channel dim remains emb_dim
            combined_channels = self.emb_dim

        # Initial conv for the second stage
        self.initial_conv = nn.Conv3d(
            combined_channels,
            block_in_final_stage,
            kernel_size=3, stride=1, padding=1
        )

        # Final processing blocks (same as original)
        self.final_blocks = nn.ModuleList([
            ResBlock(block_in_final_stage, block_in_final_stage)
            for _ in range(self.num_res_blocks) # Using H.res_blocks here
        ])

        # Final output layers (same as original)
        self.norm_out = Normalize(block_in_final_stage) # Assuming Normalize is GroupNorm
        self.conv_out = nn.Conv3d(
            block_in_final_stage, H.n_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # Input z shape: [B, C, H', W', D'] where C = 2 * emb_dim
        # H', W', D' are the latent spatial dimensions
        B, C, H_latent, W_latent, D_latent = z.shape

        # Split input into structure and style codes
        # Each has shape [B, emb_dim, H', W', D']
        z_struct = z[:, :self.emb_dim]
        z_style = z[:, self.emb_dim:]

        # --- Simplified Structure Path ---
        # 1. Upsample structure codes directly to target resolution
        h_struct_upsampled = F.interpolate(
            z_struct,
            size=(self.resolution, self.resolution, self.resolution),
            mode='trilinear', # Using trilinear for smoother interpolation pre-conv
            align_corners=False
        ) # Shape: [B, emb_dim, resolution, resolution, resolution]

        # 2. Apply single conv layer to get binary logits
        binary_logits = self.struct_to_binary_conv(h_struct_upsampled)
        # Shape: [B, 1, resolution, resolution, resolution]

        # 3. Apply sigmoid
        binary_out = torch.sigmoid(binary_logits)
        # Shape: [B, 1, resolution, resolution, resolution]

        # --- Style Path (Identical to Original) ---
        # Upsample style codes to match spatial dimensions
        h_style = F.interpolate(
            z_style,
            size=(self.resolution, self.resolution, self.resolution),
            mode='trilinear',
            align_corners=False
        ) # Shape: [B, emb_dim, resolution, resolution, resolution]

        # --- Combine (Identical to Original) ---
        if self.combine_method == 'concat':
            h_combined = torch.cat([binary_out, h_style], dim=1)
            # Shape: [B, 1 + emb_dim, resolution, resolution, resolution]
        else:  # multiply
            # Broadcasting handles the shapes: [B, 1, ...] * [B, emb_dim, ...]
            h_combined = binary_out * h_style
            # Shape: [B, emb_dim, resolution, resolution, resolution]

        # --- Final Stage (Identical to Original) ---
        # Map combined representation to the input depth for final blocks
        h_combined = self.initial_conv(h_combined)

        # Process with final ResBlocks
        for block in self.final_blocks:
            h_combined = block(h_combined)

        # Final normalization, nonlinearity, and output convolution
        h_combined = self.norm_out(h_combined)
        h_combined = nonlinearity(h_combined) # Assuming nonlinearity is swish
        out = self.conv_out(h_combined)
        # Shape: [B, n_channels, resolution, resolution, resolution]

        return out, binary_out
    
class SlightlyLessDumbTwoStageGenerator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = H.res_blocks # Used for final stage
        self.combine_method = H.combine_method
        self.resolution = H.img_size
        self.emb_dim = H.emb_dim
        self.padding_mode = H.padding_mode
        self.detach = H.detach_binary_recon

        if self.detach:
            print("Detaching binary reconstruction from comp graph for final loss")
        # --- Slightly Enhanced First Stage ---
        # Still simpler than original, but with a bit more capacity

        # 1. Optional: Add a ResBlock after upsampling
        #    Use the embedding dimension as the channel count here.
        # self.struct_resblock = ResBlock(self.emb_dim, self.emb_dim, padding_mode=self.padding_mode)
        self.struct_resblock = ResBlock(self.emb_dim, self.emb_dim, padding_mode=self.padding_mode, num_groups=self.emb_dim // 2)

        # 2. Add Norm and Nonlinearity before the final conv
        # self.struct_norm_out = Normalize(self.emb_dim) # Use Normalize/GroupNorm
        self.struct_norm_out = nn.GroupNorm(num_groups=self.emb_dim // 2, num_channels=self.emb_dim, eps=1e-6, affine=True)

        # 3. Final Conv to binary output (kernel 3x3x3 might be better than 1x1x1)
        self.struct_to_binary_conv = nn.Conv3d(self.emb_dim, 1, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

        # --- Second Stage (Identical Logic to Original Dumb Version) ---
        block_in_final_stage = self.nf * self.ch_mult[self.num_resolutions-1]
        if self.combine_method == 'concat':
            combined_channels = 1 + self.emb_dim
        else:
            combined_channels = self.emb_dim

        self.initial_conv = nn.Conv3d(
            combined_channels, block_in_final_stage, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode
        )
        self.final_blocks = nn.ModuleList([
            ResBlock(block_in_final_stage, block_in_final_stage, padding_mode=self.padding_mode)
            for _ in range(self.num_res_blocks)
        ])
        self.norm_out = Normalize(block_in_final_stage)
        self.conv_out = nn.Conv3d(
            block_in_final_stage, H.n_channels, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode
        )

    def forward(self, z):
        B, C, H_latent, W_latent, D_latent = z.shape
        z_struct = z[:, :self.emb_dim]
        z_style = z[:, self.emb_dim:]

        # --- Slightly Enhanced Structure Path ---
        # 1. Upsample structure codes
        h_struct_upsampled = F.interpolate(
            z_struct,
            size=(self.resolution, self.resolution, self.resolution),
            mode='trilinear',
            align_corners=False
        ) # Shape: [B, emb_dim, resolution, resolution, resolution]

        # 1.5 Apply ResBlock
        h_struct_proc = self.struct_resblock(h_struct_upsampled)

        # 2. Apply Norm and Nonlinearity
        h_struct_proc = self.struct_norm_out(h_struct_proc)
        h_struct_proc = nonlinearity(h_struct_proc) # Apply swish

        # 3. Apply final conv layer to get binary logits
        binary_logits = self.struct_to_binary_conv(h_struct_proc)

        # 4. Apply sigmoid
        binary_out = torch.sigmoid(binary_logits)

        # --- Style Path (Identical) ---
        
        h_style = F.interpolate(
            z_style, size=(self.resolution, self.resolution, self.resolution),
            mode='trilinear', align_corners=False
        )

        if self.detach:
            detached_binary_out = binary_out.detach()
            # --- Combine (Identical) ---
            if self.combine_method == 'concat':
                h_combined = torch.cat([detached_binary_out, h_style], dim=1)
            else:
                h_combined = detached_binary_out * h_style
        else:
            if self.combine_method == 'concat':
                h_combined = torch.cat([binary_out, h_style], dim=1)
            else:
                h_combined = binary_out * h_style

        # --- Final Stage (Identical) ---
        h_combined = self.initial_conv(h_combined)
        for block in self.final_blocks:
            h_combined = block(h_combined)
        h_combined = self.norm_out(h_combined)
        h_combined = nonlinearity(h_combined)
        out = self.conv_out(h_combined)

        return out, binary_out
    
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
    
class SpectralDiscriminator3D(nn.Module):
    """
    3D PatchGAN-style discriminator using Spectral Normalization.
    Adapted from the diffusers library example and previous 3D discriminators.
    Operates on 3D voxel data (e.g., Minecraft chunks).
    """
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        Parameters:
            input_nc (int): Number of input channels (e.g., number of block types in one-hot encoding).
            ndf (int): Number of filters in the first convolutional layer.
            n_layers (int): Number of convolutional layers in the main path (excluding the final output layer).
                            Total depth is n_layers + 1 convolutional layers.
        """
        super().__init__()

        # Using kernel size 4, stride 2, padding 1, similar to original PatchGAN
        kw = 4
        padw = 1

        # Initial layer: Conv3d + LeakyReLU
        # No normalization recommended immediately after input by some practices
        sequence = [
            spectral_norm(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Intermediate layers: Conv3d + InstanceNorm3d (optional) + LeakyReLU
        # Increasing number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            # Cap the multiplier at 8 (equivalent to ndf * 8 channels max)
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                                         kernel_size=kw, stride=2, padding=padw)),
                # InstanceNorm is often preferred over BatchNorm in GANs, especially with SpectralNorm
                # You might experiment with removing this InstanceNorm layer as SpectralNorm provides stabilization
                nn.InstanceNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Final layer: Conv3d to output a 1-channel prediction map (logits)
        # Use stride 1 for the final convolutional layer before outputting patch scores
        nf_mult_prev = nf_mult
        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult_prev, 1, kernel_size=kw, stride=1, padding=padw))
            # No Sigmoid activation here - output raw logits for losses like BCEWithLogitsLoss or Hinge Loss
        ]

        self.main = nn.Sequential(*sequence)
        self.apply(self._init_weights) # Apply custom weight initialization

    def _init_weights(self, module):
        # Initialize Conv layers similar to DCGAN/PatchGAN practices
        if isinstance(module, nn.Conv3d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                 nn.init.constant_(module.bias.data, 0)
        # Initialize InstanceNorm layers
        elif isinstance(module, nn.InstanceNorm3d):
             if module.weight is not None:
                 nn.init.normal_(module.weight.data, 1.0, 0.02)
             if module.bias is not None:
                 nn.init.constant_(module.bias.data, 0)


    def forward(self, input):
        """
        Args:
            input: Input tensor of shape [B, C, D, H, W]
                   where C is input_nc (number of block types, typically one-hot encoded)
                   D, H, W are the depth, height, width of the 3D volume.
        Returns:
            Tensor of shape [B, 1, D', H', W'] containing patch-wise realness scores (logits).
            The spatial dimensions (D', H', W') depend on the input size, kernel size, stride, and padding.
        """
        return self.main(input)

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
def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class VQLossDualCodebook(nn.Module):
    def __init__(self, H):
        super().__init__()
        # Discriminator parameters
        if H.disc_type == 'conv':
            # Use simpler 3D discriminator
            # self.discriminator = Discriminator3D(
            #     input_nc=H.n_channels,  # number of block types
            #     ndf=H.ndf  # base number of filters
            # ).cuda()
            self.discriminator = SpectralDiscriminator3D(
                input_nc=H.n_channels,  # number of block types
                ndf=H.ndf  # base number of filters
            ).cuda()
        elif H.disc_type == 'patch':
            self.discriminator = PatchGAN3DDiscriminator(
                input_nc=H.n_channels,
                ndf=H.ndf,
                n_layers=H.disc_layers
            ).cuda()
        
        if H.with_biome_supervision:
            # Initialize BiomeClassifier for feature extraction
            self.biome_feature_model = BiomeFeatureModel(H.biome_classifier_path)
        
        # Loss weights and parameters
        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.disc_weight_min = 0.0
        self.disc_weight = 0.5
        self.disc_adaptive_weight = H.disc_adaptive_weight
        self.reconstruction_weight = H.reconstruction_weight
        self.codebook_weight = H.codebook_weight
        self.biome_weight = H.biome_weight
        self.disentanglement_ratio = H.disentanglement_ratio
        self.binary_recon_weight = H.binary_reconstruction_weight
        self.with_struct_consistency = H.with_struct_consistency
        self.struct_consistency_weight = H.struct_consistency_weight
        self.n_channels = H.n_channels # Number of block types (classes)
        self.weights = torch.ones(self.n_channels) # Initialize with ones (default CE behavior)
        self.block_weighting = getattr(H, 'block_weighting', False) # Default to False if not set

        # <<< Added Gumbel parameters >>>
        self.disc_gumbel = H.disc_gumbel
        self.gumbel_tau = H.gumbel_tau
        self.gumbel_hard = H.gumbel_hard
        self.gumbel_anneal = True
        self.gumbel_tau_init = 1.0 # Initial temperature
        self.gumbel_tau_final = 0.1 # Final temperature to anneal towards
        self.gumbel_anneal_steps = 10000 # Number of steps over which to anneal

        # cycle consistency loss
        self.with_cycle_consistency = getattr(H, 'with_cycle_consistency', False)
        self.cycle_consistency_weight = getattr(H, 'cycle_consistency_weight', 1.0)
        self.cycle_start_step = getattr(H, 'cycle_start_step', 0)

        print(f'With cycle consistency: {self.with_cycle_consistency} weight: {self.cycle_consistency_weight}')

        self.disc_argmax_ste = H.disc_argmax_ste
        if self.disc_gumbel:
            print("Using gumbell sampling for disc input")
        elif self.disc_argmax_ste:
            print("Using argmax + straight through estimator for disc input")
        # <<< End Added Gumbel parameters >>>


        if self.block_weighting:
            weighted_block_indices = getattr(H, 'weighted_block_indices', [])
            weighted_block_amount = getattr(H, 'weighted_block_amount', 1.0) # Default weight is 1 if not specified

            if not weighted_block_indices:
                print("Warning: block_weighting is True, but weighted_block_indices is empty. Weights remain uniform.")
            else:
                print(f"Applying block weighting: amount={weighted_block_amount} to indices={weighted_block_indices}")
                # Ensure indices are valid
                valid_indices = [idx for idx in weighted_block_indices if 0 <= idx < self.n_channels]
                if len(valid_indices) < len(weighted_block_indices):
                    print(f"Warning: Some provided weighted_block_indices were out of range [0, {self.n_channels - 1}] and were ignored.")

                if valid_indices:
                    indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
                    # Use scatter_ to modify weights in place
                    self.weights.scatter_(0, indices_tensor, weighted_block_amount)
        else:
             print("Block weighting disabled. Using uniform weights for Cross Entropy.")


        
        # Loss functions
        self.disc_loss = hinge_d_loss
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
                optimizer_idx, global_step, last_layer=None, binary_out=None, binary_target=None, struct_consistency_loss=None, cycle_consistency_loss=None):
        
        device = reconstructions.device
        # Move weights to the correct device if they aren't already there
        weights_on_device = self.weights.to(device)

        # anneal tau for gumbel
        current_gumbel_tau = self.gumbel_tau
        if self.gumbel_anneal and self.disc_gumbel: # Only anneal if enabled
                # Prevent division by zero/invalid ops if init/final are same or steps=0
                if self.gumbel_tau_init > self.gumbel_tau_final and self.gumbel_anneal_steps > 0:
                    # Calculate decay rate for exponential decay
                    decay_rate = (self.gumbel_tau_final / self.gumbel_tau_init)**(1.0 / self.gumbel_anneal_steps)
                    # Calculate annealed tau, ensuring it doesn't drop below final value
                    current_gumbel_tau = max(
                        self.gumbel_tau_final,
                        self.gumbel_tau_init * (decay_rate ** global_step)
                    )
                else:
                    # If init <= final or steps = 0, just use final tau if past step 0, else init
                    current_gumbel_tau = self.gumbel_tau_init
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = F.cross_entropy(
                reconstructions.contiguous(),
                torch.argmax(inputs, dim=1).contiguous().long(), # Ensure target is LongTensor
                weight=weights_on_device # Pass the weights here
            ) * self.reconstruction_weight

            # Binary reconstruction loss (if in two-stage mode)
            binary_recon_loss = 0.0
            if binary_out is not None and binary_target is not None:
                # ---- START DEBUG ----
                squeezed_binary_out = binary_out.squeeze(1)
                print(f"DEBUG: binary_out.squeeze(1) min: {squeezed_binary_out.min().item()}, max: {squeezed_binary_out.max().item()}")
                if torch.any(squeezed_binary_out < 0) or torch.any(squeezed_binary_out > 1):
                    print("ERROR: binary_out.squeeze(1) is outside the [0, 1] range!")
                # ---- END DEBUG ----

                binary_recon_loss = F.binary_cross_entropy(
                    squeezed_binary_out, # Use the squeezed version
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
            
            # <<< Apply Gumbel-Softmax for Adversarial Loss >>>
            if self.disc_gumbel:
                fake_input_disc = F.gumbel_softmax(reconstructions.contiguous(), tau=current_gumbel_tau, hard=self.gumbel_hard, dim=1)
                # logits_fake = self.discriminator(recons_gumbel)
            elif self.disc_argmax_ste:
                with torch.no_grad(): # Don't track gradients for argmax/one_hot
                    # Forward pass: Apply argmax and convert to one-hot
                    indices = torch.argmax(reconstructions, dim=1)
                    y_hard = F.one_hot(indices, num_classes=reconstructions.shape[1]).float()
                    # Permute channels: [B, H, W, D, C] -> [B, C, H, W, D]
                    y_hard = y_hard.permute(0, 4, 1, 2, 3).contiguous()

                # Backward pass: Use STE trick
                # Add the difference between original logits and detached logits.
                # Gradients will flow back through `reconstructions`.
                fake_input_disc = y_hard + (reconstructions - reconstructions.detach())
            else:
                fake_input_disc = reconstructions.contiguous()

            logits_fake = self.discriminator(fake_input_disc)
            g_loss = self.gen_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                # null_loss = rec_loss + biome_feat_loss + binary_recon_loss
                null_loss = rec_loss * self.reconstruction_weight
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, g_loss, last_layer)
                # g_loss = g_loss * disc_weight * disc_adaptive_weight
            else:
                # g_loss = g_loss * disc_weight
                disc_adaptive_weight = 1

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start_step)
             # Add structure consistency loss if enabled
            struct_consistency = 0.0
            if self.with_struct_consistency and struct_consistency_loss is not None:
                struct_consistency = struct_consistency_loss * self.struct_consistency_weight
            
            # Cycle Consistency Loss term
            cycle_consistency_loss_term = 0.0
            if self.with_cycle_consistency and cycle_consistency_loss is not None and global_step > self.cycle_start_step:
                cycle_consistency_loss_term = cycle_consistency_loss * self.cycle_consistency_weight
            # fixed this to match the original impl
            # Total loss
            loss = rec_loss * self.reconstruction_weight + \
                style_loss + struct_loss + \
                biome_feat_loss + disent_loss + \
                binary_recon_loss + struct_consistency + \
                cycle_consistency_loss_term
                # disc_adaptive_weight * disc_weight * g_loss + \
                
            
            return {
                'loss': loss,
                'rec_loss': rec_loss,
                'binary_rec_loss': binary_recon_loss,
                'style_loss': style_loss,
                'struct_loss': struct_loss,
                'biome_feat_loss': biome_feat_loss,
                'disent_loss': disent_loss,
                'struct_consistency_loss': struct_consistency,
                'g_loss': g_loss,
                'disc_weight': disc_weight,
                'disc_adaptive_weight': disc_adaptive_weight if self.disc_adaptive_weight else 1.0,
                'codebook_usage_style': codebook_loss_style[3],
                'codebook_usage_struct': codebook_loss_struct[3],
                'cycle_consistency_loss': cycle_consistency_loss_term
            }
            
        # Discriminator update
        elif optimizer_idx == 1:
            # Get discriminator predictions
            logits_real = self.discriminator(inputs.contiguous().detach())

            # <<< Apply Gumbel-Softmax for Discriminator Fake Input >>>
            # Apply Gumbel-Softmax to detached logits
            # No gradients needed back to the generator here
            recons_detached = reconstructions.contiguous().detach()
            if self.disc_gumbel:
                # Gumbel-Softmax on detached reconstructions
                fake_input_disc_detached = F.gumbel_softmax(recons_detached, tau=current_gumbel_tau, hard=self.gumbel_hard, dim=1)
            elif self.disc_argmax_ste:
                # Argmax + OneHot on detached reconstructions (no STE needed here)
                indices = torch.argmax(recons_detached, dim=1)
                fake_input_disc_detached = F.one_hot(indices, num_classes=reconstructions.shape[1]).float()
                # Permute channels: [B, H, W, D, C] -> [B, C, H, W, D]
                fake_input_disc_detached = fake_input_disc_detached.permute(0, 4, 1, 2, 3).contiguous()
            else:
                # Raw detached reconstructions
                fake_input_disc_detached = recons_detached

            logits_fake = self.discriminator(fake_input_disc_detached)

            # Calculate discriminator loss with weight adoption
            disc_weight = self.adopt_weight(self.disc_weight, global_step, threshold=self.disc_start_step)
            d_loss = self.disc_loss(logits_real, logits_fake) * disc_weight

            if global_step % 100 == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                print(f"(Discriminator) "
                            f"discriminator_adv_loss: {d_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}"
                            f"discriminator weight: {disc_weight}"
                            f"Current gumbel tau: {current_gumbel_tau}")
            return {
                'd_loss': d_loss,
                'logits_real': logits_real.mean(),
                'logits_fake': logits_fake.mean(),
                'disc_weight': disc_weight
            }

def pre_quantization_consistency_loss(input_blocks, z_e):
    """Apply consistency loss to embeddings before quantization"""
    # Get binary blocks
    binary_blocks = (torch.argmax(input_blocks, dim=1) != 0).float()
    batch_size = binary_blocks.shape[0]
    
    # Reshape to match the latent resolution
    chunks = binary_blocks.reshape(batch_size, 6, 4, 6, 4, 6, 4)
    chunk_patterns = chunks.reshape(batch_size, 6, 6, 6, 64)
    
    # Get pre-quantization embeddings
    z_e_reshaped = z_e.permute(0, 2, 3, 4, 1)  # [B, 6, 6, 6, C]
    
    # For each pair of similar patterns, make embeddings more similar
    flat_patterns = chunk_patterns.reshape(-1, 64)
    flat_embeddings = z_e_reshaped.reshape(-1, z_e.shape[1])
    
    # Normalize for cosine similarity
    norm_patterns = F.normalize(flat_patterns, p=2, dim=1)
    norm_embeddings = F.normalize(flat_embeddings, p=2, dim=1)
    
    # Calculate pattern similarities
    pattern_sim = torch.mm(norm_patterns, norm_patterns.t())
    
    # Only consider highly similar patterns
    threshold = 0.5
    mask = (pattern_sim > threshold) & (pattern_sim < 0.99)
    
    # Skip if no similar pairs
    if mask.sum() == 0:
        return torch.tensor(0.0, device=input_blocks.device)
    
    # Get indices of similar pairs
    row_idx, col_idx = torch.where(mask)
    
    # Calculate embedding distances for similar pattern pairs
    emb_dists = 1.0 - F.cosine_similarity(
        norm_embeddings[row_idx], norm_embeddings[col_idx], dim=1
    )
    
    # Weight by pattern similarity - more similar patterns should have more similar embeddings
    weights = pattern_sim[row_idx, col_idx]
    
    # Return weighted loss
    return (emb_dists * weights).mean()

# New hamming distance based consistency loss
def pre_quant_hamming_consistency_loss(input_blocks, z_e_struct, similarity_threshold=0.92):
    """
    Apply consistency loss to pre-quantization embeddings based on Hamming similarity of binary patterns.

    Args:
        input_blocks (torch.Tensor): The original input tensor [B, C, H, W, D].
        z_e_struct (torch.Tensor): The pre-quantization embeddings from the encoder's structure head [B, C', H', W', D'].
        similarity_threshold (float): Hamming similarity threshold (1 - normalized_hamming_dist).

    Returns:
        torch.Tensor: The consistency loss value.
    """
    # 1. Extract binary patterns
    binary_blocks = (torch.argmax(input_blocks, dim=1) != 0).float() # [B, 24, 24, 24]
    batch_size = binary_blocks.shape[0]
    # Assuming latent grid is 6x6x6, chunks are 4x4x4
    chunks = binary_blocks.reshape(batch_size, 6, 4, 6, 4, 6, 4)
    # [B, 6, 6, 6, 64]
    chunk_patterns = chunks.reshape(batch_size, 6, 6, 6, 64)
    # Flatten patterns: [N, 64] where N = B * 6 * 6 * 6 = B * 216
    flat_patterns = chunk_patterns.reshape(-1, 64)
    N = flat_patterns.shape[0]

    # 2. Calculate pairwise Hamming similarity
    patterns_float = flat_patterns.float()
    hamming_dist_matrix = patterns_float @ (1 - patterns_float).T + (1 - patterns_float) @ patterns_float.T
    normalized_hamming_dist = hamming_dist_matrix / 64.0
    hamming_sim_matrix = 1.0 - normalized_hamming_dist

    # 3. Identify pairs with high similarity (excluding self-similarity)
    mask_noself = ~torch.eye(N, dtype=torch.bool, device=input_blocks.device)
    mask_similar = hamming_sim_matrix > similarity_threshold
    final_mask = mask_similar & mask_noself

    # 4. If no similar pairs, return zero loss
    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=input_blocks.device, requires_grad=True) # Still need grad requirement

    # Get indices of the similar pairs
    row_idx, col_idx = torch.where(final_mask)

    # 5. Get corresponding pre-quantization embeddings
    # Reshape z_e_struct: [B, C', H', W', D'] -> [B, H', W', D', C'] -> [N, C']
    z_e_struct_permuted = z_e_struct.permute(0, 2, 3, 4, 1)
    flat_embeddings = z_e_struct_permuted.reshape(N, -1) # Shape [N, C']

    # Get embeddings for the similar pairs
    embeddings1 = flat_embeddings[row_idx]
    embeddings2 = flat_embeddings[col_idx]

    # Normalize embeddings for cosine distance calculation
    norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # 6. Calculate loss: Encourage embeddings to be similar for similar patterns
    # Use cosine distance (1 - cosine_similarity)
    embedding_distance = 1.0 - F.cosine_similarity(norm_embeddings1, norm_embeddings2, dim=1)

    # Weight the loss by the Hamming similarity of the patterns
    weights = hamming_sim_matrix[row_idx, col_idx].detach() # Detach weights to avoid odd gradient paths

    # Weighted mean distance
    loss = (embedding_distance * weights).sum() / (weights.sum() + 1e-8)

    return loss



def structure_consistency_loss(input_blocks, struct_indices):
    # Convert one-hot encoded blocks to binary (1 for any block, 0 for air)
    # Assuming air is at index 0 in your one-hot encoding
    binary_blocks = (torch.argmax(input_blocks, dim=1) != 0).float()  # [B, 24, 24, 24]
    batch_size = binary_blocks.shape[0]
    
    # Reshape to separate each 4×4×4 chunk
    # [B, 6, 4, 6, 4, 6, 4] - preserving the spatial arrangement
    chunks = binary_blocks.reshape(batch_size, 6, 4, 6, 4, 6, 4)
    
    # Flatten each 4×4×4 chunk into a 64-dimensional binary vector
    # [B, 6, 6, 6, 64] where each vector is the flattened 4×4×4 pattern
    chunk_patterns = chunks.reshape(batch_size, 6, 6, 6, 64)
    
    # Get structure indices
    struct_indices = struct_indices.reshape(batch_size, 6, 6, 6)
    
    # Calculate pairwise pattern similarity across all chunks in the batch
    # Reshape to [B*216, 64]
    flat_patterns = chunk_patterns.reshape(-1, 64)
    
    # Calculate pairwise cosine similarity
    # Normalize patterns
    norm_patterns = F.normalize(flat_patterns, p=2, dim=1)
    # Compute similarity matrix [B*216, B*216]
    similarity = torch.mm(norm_patterns, norm_patterns.t())
    
    # Get the corresponding flattened indices [B*216]
    flat_indices = struct_indices.reshape(-1)
    
    # Find pairs with high similarity (above threshold)
    threshold = 0.5  # Can be adjusted
    similar_pairs = (similarity > threshold) & (similarity < 1.0)  # Exclude self-similarity
    
    if similar_pairs.sum() == 0:
        return torch.tensor(0.0, device=input_blocks.device)
    
    # For each similar pair, penalize different codes
    # Get pairs of indices for similar patterns
    row_indices, col_indices = torch.where(similar_pairs)
    idx_pairs = torch.stack([flat_indices[row_indices], flat_indices[col_indices]], dim=1)
    
    # Calculate if codes are different (1 if different, 0 if same)
    code_diff = (idx_pairs[:, 0] != idx_pairs[:, 1]).float()
    
    # Weight by similarity (more similar patterns should have more similar codes)
    pair_similarities = similarity[row_indices, col_indices]
    
    # Weighted average of code differences - penalizes different codes for similar patterns
    loss = (code_diff * pair_similarities).mean()
    
    return loss

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

def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class FeatPredHead(nn.Module):
    def __init__(self, resolution, input_dim=256, down_factor=16):
        super().__init__()
        self.grid_size = resolution // down_factor
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
        self.with_struct_consistency = H.with_struct_consistency
        self.use_dumb_decoder = getattr(H, 'use_dumb_decoder', False) # Add Hparam flag
        self.padding_mode = getattr(H, 'padding_mode', 'zeros')
        self.ema_decay = getattr(H, 'ema_decay', 0.99)


        # cycle consistency loss
        self.with_cycle_consistency = getattr(H, 'with_cycle_consistency', False)
        self.cycle_consistency_type = getattr(H, 'cycle_consistency_type', 'None')
        self.disc_gumbel_for_cycle_input = getattr(H, 'disc_gumbel_for_cycle_input', False)
        self.ema_decay = getattr(H, 'ema_decay', 0.99)
        print(f'using padding mode: {self.padding_mode}')
        print(f'With cycle consistency: {self.with_cycle_consistency} type: {self.cycle_consistency_type}, using gumbel: {self.disc_gumbel_for_cycle_input}')
        
        # Two head encoder
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            padding_mode=self.padding_mode
        )

        # Quantizer for style head (semantic)
        if H.quantizer_type == 'ema':
            self.quantize_style = FQEMAVectorQuantizer(
                self.style_codebook_size,
                self.embed_dim,
                decay=self.ema_decay
            )
            self.quantize_struct = FQEMAVectorQuantizer(
                self.struct_codebook_size, 
                self.embed_dim,
                decay=self.ema_decay
            )
            print("Using EMA quantizer")
        else:
            self.quantize_style = FQVectorQuantizer(
                # self.codebook_size, 
                self.style_codebook_size, 
                self.embed_dim,
                H.beta, 
                H.entropy_loss_ratio,
                H.codebook_l2_norm, 
                H.codebook_show_usage
            )
            # Quantizer for structural head (visual)
            self.quantize_struct = FQVectorQuantizer(
                self.struct_codebook_size, 
                self.embed_dim,
                H.beta, 
                H.entropy_loss_ratio,
                H.codebook_l2_norm, 
                H.codebook_show_usage
            )
        self.quant_conv_style = nn.Conv3d(self.z_channels, self.embed_dim, 1, padding_mode=self.padding_mode)
        self.quant_conv_struct = nn.Conv3d(self.z_channels, self.embed_dim, 1, padding_mode=self.padding_mode)

        # Pixel decoder
        input_dim = self.embed_dim * 2  # Combined dimension from both codebooks
        self.post_quant_conv = nn.Conv3d(input_dim, self.z_channels, 1)
        # self.decoder = Generator(H, z_channels=self.z_channels)
        # Choose decoder type based on hyperparameter
        if H.two_stage_decoder:
            if self.use_dumb_decoder:
                print("Using SlightlyLessDumbTwoStageGenerator")
                # self.decoder = DumbTwoStageGenerator(H)
                self.decoder = SlightlyLessDumbTwoStageGenerator(H)
            else:
                print("Using TwoStageGenerator")
                self.decoder = TwoStageGenerator(H)
        else:
            print("Using standard Generator")
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
            self.feat_pred_head = FeatPredHead(resolution=self.resolution, input_dim=self.embed_dim, down_factor=down_factor)
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
        h_style_raw, h_struct_raw = self.encoder(input)
        h_style = self.quant_conv_style(h_style_raw)
        h_struct = self.quant_conv_struct(h_struct_raw)

        # Quantize both paths
        quant_style, emb_loss_style, indices_style = self.quantize_style(h_style)
        quant_struct, emb_loss_struct, indices_struct = self.quantize_struct(h_struct)

        # ---- START DEBUG NaN checks for quantizer outputs ----
        if torch.isnan(quant_style).any():
            print(f"!!! NaN DETECTED IN quant_style!!!")
        if torch.isnan(quant_struct).any():
            print(f"!!! NaN DETECTED IN quant_struct!!!")

        # Biome feature prediction if enabled
        if self.with_biome_supervision:
            style_feat = self.feat_pred_head(quant_style)
        else:
            style_feat = None

        # Structural consistency loss, if enabled
        struct_consistency_loss = None
        if self.with_struct_consistency:
            # struct_consistency_loss = structure_consistency_loss(input, indices_struct[2])
            # struct_consistency_loss = pre_quantization_consistency_loss(input, h_struct)
            struct_consistency_loss = pre_quant_hamming_consistency_loss(input, h_struct, similarity_threshold=0.72)
            
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
        # ---- START DEBUG NaN check for combined quant ----
        if torch.isnan(quant).any():
            print(f"!!! NaN DETECTED IN quant (input to decoder) at step {global_step} !!!")
        if self.two_stage_decoder:
            dec_logits, binary_out = self.decoder(quant)
            # Create binary target (you'll need to implement this based on your data)
            # binary_target = self.create_binary_target(input)
            # binary_loss = F.binary_cross_entropy(binary_out, binary_target)
            # return dec, binary_out,  emb_loss_style, emb_loss_struct, disentangle_loss, style_feat, struct_consistency_loss
        else:
            dec_logits = self.decoder(quant)
            # return dec, emb_loss_style, emb_loss_struct, disentangle_loss, style_feat, struct_consistency_loss
        
        if torch.isnan(dec_logits).any():
            print(f"!!! NaN DETECTED IN dec_logits!!!")
        if binary_out is not None and torch.isnan(binary_out).any():
            print(f"!!! NaN DETECTED IN binary_out at step (inside FQModel.forward) !!!")
        # Cycle Consistency Loss Calculation
        cycle_consistency_loss_val = None
        if self.with_cycle_consistency:
            input_for_cycle_encoder = None
            # Process decoder output (dec_logits) to be suitable for encoder input
            dec_logits_detached = dec_logits.detach()
            if self.disc_gumbel_for_cycle_input: # Align with discriminator input processing
                input_for_cycle_encoder = F.gumbel_softmax(dec_logits_detached.contiguous(), tau=1, hard=True, dim=1)
            else: # Default to Argmax + STE
                # Calculate y_hard (discrete one-hot version for the forward pass of the cycle encoder)
                pred_indices_cycle = torch.argmax(dec_logits_detached, dim=1) 
                y_hard_cycle = F.one_hot(pred_indices_cycle, num_classes=self.in_channels).permute(0, 4, 1, 2, 3).float()
                
                # Create the STE version: y_eff = y_hard + x_orig_logits - x_orig_logits.detach()
                # This ensures y_hard_cycle is used in the forward pass of the cycle encoder,
                # while gradients flow back to dec_logits in the backward pass.
                # input_for_cycle_encoder = y_hard_cycle + (dec_logits - dec_logits.detach())
                input_for_cycle_encoder = y_hard_cycle
            
            # Re-encode the processed reconstruction (Cycle Pass)
            h_style_cycle_raw, h_struct_cycle_raw = self.encoder(input_for_cycle_encoder)
            
            if self.cycle_consistency_type == 'post_quant_conv':
                h_style_cycle_post_conv = self.quant_conv_style(h_style_cycle_raw)
                h_struct_cycle_post_conv = self.quant_conv_struct(h_struct_cycle_raw)
                
                loss_cycle_style = F.mse_loss(h_style_cycle_post_conv, h_style.detach())
                loss_cycle_struct = F.mse_loss(h_struct_cycle_post_conv, h_struct.detach())
                cycle_consistency_loss_val = loss_cycle_style + loss_cycle_struct
                # cycle_consistency_loss_val = loss_cycle_struct
            elif self.cycle_consistency_type == 'pre_quant':
                loss_cycle_style = F.mse_loss(h_style_cycle_raw, h_style_raw.detach())
                loss_cycle_struct = F.mse_loss(h_struct_cycle_raw, h_struct_raw.detach())
                cycle_consistency_loss_val = loss_cycle_style + loss_cycle_struct
                # cycle_consistency_loss_val = loss_cycle_struct
            # Add other types (e.g., 'post_quant' comparing VQ outputs) if needed later
        if self.two_stage_decoder:
            return dec_logits, binary_out, emb_loss_style, emb_loss_struct, disentangle_loss, style_feat, struct_consistency_loss, cycle_consistency_loss_val
        else:
            return dec_logits, emb_loss_style, emb_loss_struct, disentangle_loss, style_feat, struct_consistency_loss, cycle_consistency_loss_val
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
            self.n_channels = 42
            self.nf = 64
            self.ndf = 64
            self.res_blocks = 2
            self.latent_shape = [1, 6, 6, 6]
            
            # Structure consistency parameters
            self.with_struct_consistency = False  # Enable structure consistency regularization
            self.struct_consistency_weight = 2.0  # Weight for structure consistency loss
            self.struct_consistency_threshold = 0.72  # Similarity threshold

            # New parameters for dual codebook architecture
            self.struct_codebook_size = 20  # Size of each codebook
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
            self.disc_start_step = 10000  # Step to start discriminator training
            self.reconstruction_weight = 1.0  # Weight for reconstruction loss
            self.codebook_weight = 1.0  # Weight for codebook loss
            self.biome_weight = 1.0  # Weight for biome feature prediction
            self.disentanglement_ratio = 0.5  # Weight for disentanglement loss
            
            # Codebook specific parameters
            self.quantizer_type = 'ema'
            self.beta = 0.5  # Commitment loss coefficient
            # self.entropy_loss_ratio = 0.05  # For codebook entropy regularization
            self.entropy_loss_ratio = 0.2
            self.codebook_l2_norm = True  # Whether to L2 normalize codebook entries
            self.codebook_show_usage = True  # Track codebook usage statistics
            self.ema_decay = 0.99

            
            # Training parameters
            self.lr = 1e-4  # Learning rate
            self.beta1 = 0.9  # Adam beta1
            self.beta2 = 0.95  # Adam beta2
            self.disc_layers = 3  # Number of discriminator layers
            self.train_steps = 15000
            self.start_step = 0
            
            self.transformer_dim = self.emb_dim  # Make transformer dim match embedding dim
            self.num_heads = 8  # Number of attention heads
            
            # Feature prediction parameters
            self.with_biome_supervision = False  # Enable biome feature prediction
            self.with_disentanglement = True  # Enable disentanglement loss
            
            # Logging parameters (if not already present)
            self.steps_per_log = 150
            self.steps_per_checkpoint = 1000
            self.steps_per_display_output = 500
            self.steps_per_save_output = 500
            self.steps_per_validation = 150
            self.val_samples_to_save = 16
            self.val_samples_to_display = 4
            self.visdom_port = 8097
            
            # Two stage decoder stuff
            self.binary_reconstruction_weight = 1
            self.two_stage_decoder = True
            self.use_dumb_decoder = False
            self.combine_method = 'concat'

            # Weighted recon loss:
            self.block_weighting = True
            self.weighted_block_amount = 3.0
            self.weighted_block_indices = []

            self.disc_gumbel = True
            self.gumbel_tau = 1
            self.gumbel_hard = True
            self.disc_argmax_ste = False
             
            self.num_biomes = 11  # Number of biome classes
            self.biome_feat_dim = 256  # Dimension of biome features
            self.biome_classifier_path = 'best_biome_classifier_airprocessed.pt'
        else:
            raise KeyError(f'Defaults not defined for dataset: {self.dataset}')