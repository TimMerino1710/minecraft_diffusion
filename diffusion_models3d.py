from functools import partial
from collections import defaultdict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchinfo import summary
import torch.distributions as dists
import copy
import time
import random

from torchinfo import summary


## Causal Self Attn
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


## Transformer block
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

## Transformer Model
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
        # print(f'token embeddings shape: {token_embeddings.shape}')
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

## Sampler Module
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

    # def embed(self, z):
    #     with torch.no_grad():
    #         z_flattened = z.view(-1, self.codebook_size)
    #         embedded = torch.matmul(z_flattened, self.embedding_weight).view(
    #             z.size(0),
    #             self.latent_shape[1],  # D
    #             self.latent_shape[2],  # H 
    #             self.latent_shape[3],  # W
    #             self.emb_dim
    #         ).permute(0, 4, 1, 2, 3).contiguous()

    #     return embedded
    def embed(self, z):
        with torch.no_grad():
            z_flattened = z.view(-1, self.codebook_size)
            embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                z.size(0),
                z.size(1),  # D
                z.size(2),  # H 
                z.size(3),  # W
                self.emb_dim
            ).permute(0, 4, 1, 2, 3).contiguous()

        return embedded

## Absorbing Diffusion Model
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

    def sample_with_intermediates(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_t = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, sample_steps+1))
        
        # Lists to store intermediate states
        states = [x_t.clone()]  # Include initial state
        masks = [unmasked.clone()]  # Include initial mask

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            unmasked = torch.bitwise_or(unmasked, changes)

            # Use denoiser to predict x_0
            x_0_logits = self._denoise_fn(x_t, t=t)
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

            # Store intermediate state and mask
            states.append(x_t.clone())
            masks.append(unmasked.clone())

        return {
            'states': states,  # List of tensors showing progression
            'masks': masks,    # List of boolean masks showing what's revealed
            'final': x_t      # Final result
        }
    
    def sample_with_inpainting(self, base_latent, slice_coords, sample_steps=None, temp=1.0):
        """
        Perform inpainting on specific coordinates of a base latent.
        
        Args:
            base_latent (torch.Tensor): Starting latent tensor [D, H, W]
            slice_coords (list): List of (x, y, z) coordinates to inpaint
            sample_steps (int): Number of denoising steps
            temp (float): Temperature for sampling
        """
        device = 'cuda'
        # Add batch dimension if not present
        if len(base_latent.shape) == 3:
            base_latent = base_latent.unsqueeze(0)
        
        # Create initial x_t as copy of base_latent
        x_t = base_latent.clone().to(device)
        b = x_t.size(0)  # Batch size
        
        # Create mask for tracking what's been unmasked
        unmasked = torch.ones_like(x_t, device=device, dtype=torch.bool)
        
        # Mask out the specified coordinates
        for coord in slice_coords:
            x_t[:, coord[0], coord[1], coord[2]] = self.mask_id
            unmasked[:, coord[0], coord[1], coord[2]] = False
        
        # Create coordinate mask for what we want to inpaint
        inpaint_mask = torch.zeros_like(x_t, device=device, dtype=torch.bool)
        for coord in slice_coords:
            inpaint_mask[:, coord[0], coord[1], coord[2]] = True
        
        sample_steps = list(range(1, sample_steps+1))
        
        for t in reversed(sample_steps):
            print(f'Inpainting timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Only unmask within our inpainting region
            changes = (torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)) & inpaint_mask
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            unmasked = torch.bitwise_or(unmasked, changes)
            
            # Flatten for denoising
            x_t_flat = x_t.reshape(b, -1)
            
            # Get predictions for all tokens
            x_0_logits = self._denoise_fn(x_t_flat, t=t)
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            
            # Only update the tokens we want to inpaint
            x_t_flat[changes.reshape(b, -1)] = x_0_hat[changes.reshape(b, -1)]
            x_t = x_t_flat.reshape(x_t.shape)
        
        return x_t

    def sample_with_outpainting(self, base_latent, preserve_coords, sample_steps=None, temp=1.0):
        """
        Perform outpainting by preserving specific coordinates and generating everything else.
        
        Args:
            base_latent (torch.Tensor): Starting latent tensor [D, H, W]
            preserve_coords (list): List of (x, y, z) coordinates to preserve
            sample_steps (int): Number of denoising steps
            temp (float): Temperature for sampling
        """
        device = 'cuda'
        # Add batch dimension if not present
        if len(base_latent.shape) == 3:
            base_latent = base_latent.unsqueeze(0)
        
        # Create initial x_t filled with mask tokens
        x_t = torch.ones_like(base_latent, device=device).long() * self.mask_id
        b = x_t.size(0)  # Batch size
        
        # Create mask for tracking what's been unmasked
        unmasked = torch.zeros_like(x_t, device=device, dtype=torch.bool)
        
        # Preserve the specified coordinates from base_latent
        for coord in preserve_coords:
            x_t[:, coord[0], coord[1], coord[2]] = base_latent[:, coord[0], coord[1], coord[2]]
            unmasked[:, coord[0], coord[1], coord[2]] = True
        
        # Create coordinate mask for what we want to generate (inverse of preserve mask)
        outpaint_mask = torch.ones_like(x_t, device=device, dtype=torch.bool)
        for coord in preserve_coords:
            outpaint_mask[:, coord[0], coord[1], coord[2]] = False
        
        sample_steps = list(range(1, sample_steps+1))
        
        for t in reversed(sample_steps):
            print(f'Outpainting timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Only unmask outside preserved region
            changes = (torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)) & outpaint_mask
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            unmasked = torch.bitwise_or(unmasked, changes)
            
            # Flatten for denoising
            x_t_flat = x_t.reshape(b, -1)
            
            # Get predictions for all tokens
            x_0_logits = self._denoise_fn(x_t_flat, t=t)
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            
            # Only update the tokens we want to generate
            x_t_flat[changes.reshape(b, -1)] = x_0_hat[changes.reshape(b, -1)]
            x_t = x_t_flat.reshape(x_t.shape)
        
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