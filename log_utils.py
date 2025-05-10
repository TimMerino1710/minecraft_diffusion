import logging
import numpy as np
import os
import torch
import torchvision
# import visdom
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

import json

def log_hparams_to_json(hparams, log_dir, filename="hparams.json"):
    """
    Log hyperparameters to a JSON file.

    Args:
        hparams (dict): Dictionary containing hyperparameters.
        log_dir (str): Directory to save the JSON file.
        filename (str, optional): Name of the JSON file. Defaults to "hparams.json".
    """
    log_dir = "../model_logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)
    
    with open(file_path, 'w') as f:
        json.dump(hparams, f, indent=4)
    
    log(f"Hyperparameters logged to {file_path}")

def config_log(log_dir, filename="log.txt"):
    log_dir = "../model_logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def log(output):
    logging.info(output)
    print(output)


def log_stats(step, stats):
    log_str = f"Step: {step}  "
    for stat in stats:
        if "latent_ids" not in stat:
            try:
                log_str += f"{stat}: {stats[stat]:.4f}  "
            except TypeError:
                log_str += f"{stat}: {stats[stat].mean().item():.4f}  "

    log(log_str)


def start_training_log(hparams):
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def save_model(model, model_save_name, step, log_dir):
    log_dir = "../model_logs/" + log_dir + "/saved_models"
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{model_save_name}_{step}.th"
    log(f"Saving {model_save_name} to {model_save_name}_{str(step)}.th")
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir, strict=False):
    log(f"Loading {model_load_name}_{str(step)}.th")
    log_dir = "../model_logs/" + log_dir + "/saved_models"
    try:
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
            strict=strict,
        )
    except TypeError:  # for some reason optimisers don't liek the strict keyword
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
        )

    return model


def display_images(vis, images, H, win_name=None, is_original=False):
    if win_name is None:
        win_name = f"{H.model}_{'original' if is_original else 'reconstructed'}_images"
    images = torchvision.utils.make_grid(images.clamp(0, 1), nrow=int(np.sqrt(images.size(0))), padding=0)
    vis.image(images, win=win_name, opts=dict(title=win_name))

def display_maps(vis, images, H, visualizer, win_name=None, is_original=False):
    if win_name is None:
        win_name = f"{H.model}_{'original' if is_original else 'reconstructed'}_images"
    
    # Convert maps to images using visualizer
    image_list = []
    for i in range(min(images.size(0), 16)):  # Display up to 16 maps in a grid
        img = visualizer.map_to_image(images[i])
        # Convert PIL image to numpy array and change to CHW format for visdom
        img_np = np.array(img).transpose(2, 0, 1)
        image_list.append(img_np)
    
    # Stack images and create grid
    images = np.stack(image_list)
    images = torch.from_numpy(images).float()
    grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(len(image_list))), padding=0)
    
    # Display in visdom
    vis.image(grid, win=win_name, opts=dict(title=win_name))

def save_images(images, im_name, step, log_dir, save_individually=False, is_original=False):
    log_dir = "../model_logs/" + log_dir + "/images"
    os.makedirs(log_dir, exist_ok=True)
    prefix = "original_" if is_original else ""
    if save_individually:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/{prefix}{im_name}_{step}_{idx}.png")
    else:
        torchvision.utils.save_image(
            torch.clamp(images, 0, 1),
            f"{log_dir}/{prefix}{im_name}_{step}.png",
            nrow=int(np.sqrt(images.shape[0])),
            padding=0
        )

def save_maps(images, im_name, step, log_dir, visualizer, save_individually=False, is_original=False):
    """
    Save maps as images using the visualizer.
    
    Args:
        images: Tensor of maps in format (B, C, H, W)
        im_name: Base name for saved files
        step: Current training step
        log_dir: Directory to save images
        visualizer: MapVisualizer instance
        save_individually: Whether to save each map separately
        is_original: Whether these are original or reconstructed maps
    """
    log_dir = "../model_logs/" + log_dir + "/images"
    os.makedirs(log_dir, exist_ok=True)
    prefix = "original_" if is_original else ""
    
    if save_individually:
        # Save each map separately
        for idx in range(len(images)):
            img = visualizer.map_to_image(images[idx])
            img.save(f"{log_dir}/{prefix}{im_name}_{step}_{idx}.png")
    else:
        # Create a grid of maps
        image_list = []
        for i in range(min(images.size(0), 16)):  # Limit to 16 maps for grid
            img = visualizer.map_to_image(images[i])
            img_np = np.array(img).transpose(2, 0, 1)
            image_list.append(img_np)
        
        # Create and save grid
        grid_images = np.stack(image_list)
        grid_images = torch.from_numpy(grid_images).float()
        grid = torchvision.utils.make_grid(
            grid_images,
            nrow=int(np.sqrt(len(image_list))),
            padding=0
        )
        
        # Convert back to PIL and save
        grid_image = torchvision.transforms.ToPILImage()(grid)
        grid_image.save(f"{log_dir}/{prefix}{im_name}_{step}.png")


def save_latents(H, train_latent_ids, val_latent_ids):
    save_dir = "latents/"
    os.makedirs(save_dir, exist_ok=True)

    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""
    train_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_{H.log_dir}_train_latents{latents_fp_suffix}.pt"
    val_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_{H.log_dir}_val_latents{latents_fp_suffix}.pt"
        
    torch.save(train_latent_ids, train_latents_fp)
    torch.save(val_latent_ids, val_latents_fp)


def save_stats(H, stats, step):
    save_dir = f"../model_logs/{H.log_dir}/saved_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"../model_logs/{H.log_dir}/saved_stats/stats_{step}"
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)


def load_stats(H, step):
    load_path = f"../model_logs/{H.load_dir}/saved_stats/stats_{step}"
    stats = torch.load(load_path)
    return stats


def set_up_visdom(H):
    server = H.visdom_server
    try:
        if server:
            vis = visdom.Visdom(server=server, port=H.visdom_port)
        else:
            vis = visdom.Visdom(port=H.visdom_port)
        return vis

    except Exception:
        log_str = "Failed to set up visdom server - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)

def display_minecraft(vis, mc_visualizer, data, mc_dataset, win_name="minecraft_display", title="Minecraft Chunks", nrow=4, save_path=None):
    """
    Display or save multiple minecraft chunks.
    
    Args:
        vis: Visdom instance (can be None if only saving)
        data: Tensor of shape [B, 256, 20, 20, 20] or [B, 20, 20, 20]
        win_name: Window name for visdom
        title: Title for the plot
        nrow: Number of images per row
        save_path: If provided, saves the figure to this path
    """
    # Convert to original block IDs for visualization
    data = mc_dataset.convert_to_original_blocks(data)
    # Convert to one-hot if needed
    if len(data.shape) == 4:  # [B, 20, 20, 20]
        data = F.one_hot(data.long(), num_classes=256).permute(0, 4, 1, 2, 3).float()
    
    # Create figure with subplots
    batch_size = min(data.shape[0], 16)  # Display up to 16 chunks
    ncols = nrow
    nrows = (batch_size + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    fig.suptitle(title)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        mc_visualizer.visualize_chunk(data[i], ax)
        ax.set_title(f'Chunk {i}')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Display in visdom if instance provided
    if vis is not None:
        # Convert matplotlib figure to numpy array for visdom
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(height, width, 3)
        
        vis.image(
            img_array.transpose(2, 0, 1),  # Convert to CHW format
            win=win_name,
            opts=dict(
                title=title,
                caption=f'Batch of {batch_size} chunks'
            )
        )
    
    plt.close(fig)

def save_minecraft(data, mc_visualizer, mc_dataset, save_path, nrow=4, title="Minecraft Chunks"):
    """
    Save multiple minecraft chunks to a file.
    
    Args:
        data: Tensor of shape [B, 256, 20, 20, 20] or [B, 20, 20, 20]
        save_path: Path to save the image
        nrow: Number of images per row
        title: Title for the plot
    """
    # Convert to original block IDs for visualization
    data = mc_dataset.convert_to_original_blocks(data)
    # Convert to one-hot if needed
    if len(data.shape) == 4:  # [B, 20, 20, 20]
        data = F.one_hot(data.long(), num_classes=256).permute(0, 4, 1, 2, 3).float()
    
    # Create figure with subplots
    batch_size = min(data.shape[0], 16)  # Save up to 16 chunks
    ncols = nrow
    nrows = (batch_size + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    fig.suptitle(title)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        mc_visualizer.visualize_chunk(data[i], ax)
        ax.set_title(f'Chunk {i}')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
