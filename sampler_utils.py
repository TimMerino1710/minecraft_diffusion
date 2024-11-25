import os
import torch
from tqdm import tqdm
from log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion


def get_sampler(H, embedding_weight):

    if H.sampler == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        sampler = AbsorbingDiffusion(
            H, denoise_fn, H.codebook_size, embedding_weight)

    return sampler


@torch.no_grad()
def get_samples(H, generator, sampler):

    
    if H.sampler == "absorbing":
        if H.sample_type == "diffusion":
            latents = sampler.sample(sample_steps=H.sample_steps, temp=H.temp)
        else:
            latents = sampler.sample_mlm(temp=H.temp, sample_steps=H.sample_steps)

    elif H.sampler == "autoregressive":
        latents = sampler.sample(H.temp)

    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images

@torch.no_grad()
def get_samples3d(H, generator, sampler):

    
    if H.sampler == "absorbing":
        if H.sample_type == "diffusion":
            latents = sampler.sample(sample_steps=H.sample_steps, temp=H.temp)
        else:
            latents = sampler.sample_mlm(temp=H.temp, sample_steps=H.sample_steps)

    elif H.sampler == "autoregressive":
        latents = sampler.sample(H.temp)

    latents_one_hot = latent_ids_to_onehot3d(latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)

def latent_ids_to_onehot3d(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        latent_shape[3],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


@torch.no_grad()
def generate_latent_ids(H, ae, train_loader, val_loader=None):
    train_latent_ids = generate_latents_from_loader(H, ae, train_loader)
    if val_loader is not None:
        val_latent_ids = generate_latents_from_loader(H, ae, val_loader)
    else:
        val_latent_ids = None

    save_latents(H, train_latent_ids, val_latent_ids)

@torch.no_grad()
def generate_latent_ids3d(H, ae, train_loader, val_loader=None):
    train_latent_ids = generate_latents_from_loader3d(H, ae, train_loader)
    if val_loader is not None:
        val_latent_ids = generate_latents_from_loader3d(H, ae, val_loader)
    else:
        val_latent_ids = None

    save_latents(H, train_latent_ids, val_latent_ids)


def generate_latents_from_loader(H, autoencoder, dataloader):
    latent_ids = []
    for batch in tqdm(dataloader):
        # Handle both cases: (data, label) and just data
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.cuda()
        latents = autoencoder.encoder(x)  # B, emb_dim, H, W

        latents = latents.permute(0, 2, 3, 1).contiguous()  # B, H, W, emb_dim
        latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim

        # Get embedding weights - handle both Parameter and Buffer cases
        if hasattr(autoencoder.quantize, 'embedding'):
            embedding = autoencoder.quantize.embedding
        elif hasattr(autoencoder.quantize, 'embedding.weight'):
            embedding = autoencoder.quantize.embedding.weight
        else:
            raise AttributeError("Could not find embedding in quantizer")

        
        # Calculate distances
        distances = (latents_flattened ** 2).sum(dim=1, keepdim=True) + \
            (embedding**2).sum(1) - \
            2 * torch.matmul(latents_flattened, embedding.t())

        min_encoding_indices = torch.argmin(distances, dim=1)

        latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous())
    return torch.cat(latent_ids, dim=0)

def generate_latents_from_loader3d(H, autoencoder, dataloader):
    latent_ids = []
    for batch in tqdm(dataloader):
        # Handle both cases: (data, label) and just data
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.cuda()
        latents = autoencoder.encoder(x)  # B, emb_dim, H, W

        latents = latents.permute(0, 2, 3, 4, 1).contiguous()  # B, H, W, emb_dim
        latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim

        # Get embedding weights - handle both Parameter and Buffer cases
        if hasattr(autoencoder.quantize, 'embedding'):
            embedding = autoencoder.quantize.embedding
        elif hasattr(autoencoder.quantize, 'embedding.weight'):
            embedding = autoencoder.quantize.embedding.weight
        else:
            raise AttributeError("Could not find embedding in quantizer")

        
        # Calculate distances
        distances = (latents_flattened ** 2).sum(dim=1, keepdim=True) + \
            (embedding**2).sum(1) - \
            2 * torch.matmul(latents_flattened, embedding.t())

        min_encoding_indices = torch.argmin(distances, dim=1)

        latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous())
    return torch.cat(latent_ids, dim=0)


@torch.no_grad()
def get_latent_loaders(H, get_validation_loader=True, shuffle=True):
    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""

    train_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_{H.log_dir}_train_latents{latents_fp_suffix}.pt"
    train_latent_ids = torch.load(train_latents_fp)
    train_latent_loader = torch.utils.data.DataLoader(train_latent_ids, batch_size=H.batch_size, shuffle=shuffle)

    if get_validation_loader:
        val_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_{H.log_dir}_val_latents{latents_fp_suffix}.pt"
        val_latent_ids = torch.load(val_latents_fp)
        val_latent_loader = torch.utils.data.DataLoader(val_latent_ids, batch_size=H.batch_size, shuffle=shuffle)
    else:
        val_latent_loader = None

    return train_latent_loader, val_latent_loader


# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"../model_logs/{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th"
    if not os.path.exists(ae_load_path):
        ae_load_path = f"../model_logs/{H.ae_load_dir}/saved_models/vqgan_{H.ae_load_step}.th"
    log(f"Loading VQGAN from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")
    
    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]

    return state_dict

# def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
#     state_dict = {}
#     # default to loading ema models first
#     ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th"
#     if not os.path.exists(ae_load_path):
#         ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_{H.ae_load_step}.th"
#     log(f"Loading VQGAN from {ae_load_path}")
#     full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")
    
#     for key in full_vqgan_state_dict:
#         for component in components_list:
#             if component in key:
#                 new_key = key  # Keep the original key
#                 if remove_component_from_key:
#                     new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

#                 state_dict[new_key] = full_vqgan_state_dict[key]

#     return state_dict