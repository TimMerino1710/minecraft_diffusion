from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path





import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def rotate_voxels_90(voxels, k=1):
    """
    Rotate voxels around Y axis by k*90 degrees
    Args:
        voxels: tensor of shape [B, C, H, W, D] or [C, H, W, D]
        k: number of 90 degree rotations (1 = 90°, 2 = 180°, 3 = 270°)
    Returns:
        Rotated voxels
    """
    # Handle both batched and unbatched inputs
    if len(voxels.shape) == 5:  # Batched [B, C, H, W, D]
        # Rotate around Y (height) axis by swapping width and depth dimensions
        return torch.rot90(voxels, k=k, dims=(2, 4))
    elif len(voxels.shape) == 4:  # Unbatched [C, H, W, D]
        return torch.rot90(voxels, k=k, dims=(1, 3))
    elif len(voxels.shape) == 3:  # Unbatched [H, W, D]
        return torch.rot90(voxels, k=k, dims=(0, 2))
    else:
        raise ValueError(f"Unexpected voxel shape: {voxels.shape}")
    
class MinecraftRotationAugmentation:
    def __init__(self, p=0.75):
        """
        Args:
            p: probability of applying rotation (0-1)
        """
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # Randomly choose rotation: 90°, 180°, or 270°
            k = random.randint(1, 3)
            return rotate_voxels_90(x, k)
        return x

class MinecraftDataset(Dataset):
    def __init__(self, data_path, augment=True, rotation_prob=0.75):
        data_path = Path(data_path)
        self.augment = augment
        self.rotation_prob = rotation_prob
        # Try to load processed data first
        # assert processed_data_path.exists() and mappings_path.exists()

        print("Loading pre-processed data...")
        processed_data = torch.load(data_path)
        # Only keep the chunks, discard biome data
        self.processed_chunks = processed_data['chunks']
        # Delete the biomes to free memory
        del processed_data['biomes']
        del processed_data
        
        
        print(f"Loaded {len(self.processed_chunks)} chunks of size {self.processed_chunks.shape[1:]}")
        print(f"Number of unique block types: {self.processed_chunks.shape[1]}")
        print(f'Unique blocks: {torch.unique(torch.argmax(self.processed_chunks, dim=1)).tolist()}')

    def __getitem__(self, idx):
        chunk = self.processed_chunks[idx]
        
        # Apply random rotation augmentation during training
        if self.augment and random.random() < self.rotation_prob:
            # Randomly choose rotation: 90°, 180°, or 270°
            k = random.randint(1, 3)
            chunk = rotate_voxels_90(chunk, k)
            
        return chunk
    
    def __len__(self):
        return len(self.processed_chunks)

def get_minecraft_dataloaders(data_path, batch_size=32, val_split=0.1, num_workers=0, save_val_path=None, augment=True):
    """
    Creates training and validation dataloaders for Minecraft chunks.
    """
    # Create dataset
    dataset = MinecraftDataset(data_path)
    
    # Split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    val_dataset.dataset.augment = False

    # Save validation data if path provided
    if save_val_path:
        print(f'saving validation dataset to file: {save_val_path}')
        # Extract validation samples
        val_samples = torch.stack([dataset.processed_chunks[i] for i in val_dataset.indices])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_val_path), exist_ok=True)
        
        # Save validation data
        torch.save({
            'data': val_samples,
            'indices': val_dataset.indices
        }, save_val_path)
        print(f"Saved validation data to {save_val_path}")
    
    # Create dataloaders with memory pinning
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    
    print(f"\nDataloader details:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


# class MinecraftVisualizer:
#     def __init__(self):
#         """Initialize the visualizer with the same block color mappings"""
#         self.blocks_to_cols = {
#             0: (0.5, 0.25, 0.0),    # light brown
#             29: "#006400",
#             38: "#B8860B",
#             60: "brown",
#             92: "gold",
#             93: "green",
#             115: "brown",
#             119: "forestgreen",
#             120: "forestgreen",
#             194: "yellow",
#             217: "gray",
#             227: "#90EE90",
#             240: "blue",
#             40: "#2F4F4F",
#             62: "#228B22",
#             108: "#BEBEBE",
#             131: "saddlebrown",
#             132: "saddlebrown",
#             95: "lightgray",
#             243: "wheat",
#             197: "limegreen",
#             166: "orange",
#             167: "#FF8C00",
#             184: "#FFA07A",
#             195: "tan",
#             250: "white",
#             251: "gold",
#         }

#     def visualize_chunk(self, voxels, ax=None):
#         """
#         Create a 3D visualization of a Minecraft chunk using the original plotting logic.
        
#         Args:
#             voxels: torch.Tensor [C,H,W,D] (one-hot) or numpy.ndarray [H,W,D] (block IDs)
#             ax: Optional matplotlib axis
#         """
#         # Convert one-hot to block IDs if needed
#         if isinstance(voxels, torch.Tensor):
#             if voxels.dim() == 4:  # One-hot encoded [C,H,W,D]
#                 voxels = voxels.detach().cpu()
#                 voxels = torch.argmax(voxels, dim=0).numpy()
#             else:
#                 voxels = voxels.detach().cpu().numpy()

#         # Apply the same transformations as original
#         voxels = voxels.transpose(2, 0, 1)
#         # Rotate the voxels 90 degrees around the height axis
#         voxels = np.rot90(voxels, 1, (0, 1))

#         # Create axis if not provided
#         if ax is None:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')

#         # Plot non-air blocks
#         other_vox = (voxels != 5) & (voxels != -1)
        
#         # Plot each block type with its color
#         for block_id in np.unique(voxels[other_vox]):
#             if block_id not in self.blocks_to_cols:
#                 # print(f"Unknown block id: {block_id}")
#                 continue
#             ax.voxels(voxels == block_id, facecolors=self.blocks_to_cols[int(block_id)])
#             other_vox = other_vox & (voxels != block_id)

#         # Plot remaining blocks in red with black edges
#         ax.voxels(other_vox, edgecolor="k", facecolor="red")
        
#         return ax

        
# def visualize_images(data_loader):
#     data_iter = iter(data_loader)
#     images, _ = next(data_iter)
#     images = images.numpy()  # Convert images to numpy for visualization

#     fig, axes = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
#     for i, ax in enumerate(axes.flat):
#         img = images[i].squeeze()  # Remove channel dimension
#         ax.imshow(img, cmap='gray')  # Display image in grayscale
#         ax.axis('off')  # Hide axes ticks
#     plt.show()


# So that our conversion back to the original indexes isn't tied to the dataset, we can just load it from file
class BlockConverter:
    def __init__(self, index_to_block_map=None, block_to_index_map=None):
        """
        Initialize with pre-computed mappings
        Can be initialized either with saved mappings or by creating new ones
        """
        self.index_to_block = index_to_block_map
        self.block_to_index = block_to_index_map
    
    @classmethod
    def from_dataset(cls, data_path):
        """Create mappings from a dataset file"""
        chunks = torch.from_numpy(np.load(data_path))
        unique_blocks = torch.unique(chunks).numpy()
        
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        return cls(index_to_block, block_to_index)
    
    @classmethod
    def load_mappings(cls, path):
        """Load pre-saved mappings"""
        mappings = torch.load(path)
        return cls(mappings['index_to_block'], mappings['block_to_index'])
    
    def save_mappings(self, path):
        """Save mappings for later use"""
        torch.save({
            'index_to_block': self.index_to_block,
            'block_to_index': self.block_to_index
        }, path)
    
    def convert_to_original_blocks(self, data):
        """Convert from indices back to original block IDs"""
        if len(data.shape) == 5:  # [B, C, H, W, D] or [C, H, W, D]
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        return torch.tensor([[[[self.index_to_block[int(b)] 
                              for b in row]
                             for row in layer]
                            for layer in slice_]
                           for slice_ in data])
    
class BlockBiomeConverter:
    def __init__(self, block_mappings=None, biome_mappings=None):
        """
        Initialize with pre-computed mappings for both blocks and biomes
        
        Args:
            block_mappings: dict containing 'index_to_block' and 'block_to_index'
            biome_mappings: dict containing 'index_to_biome' and 'biome_to_index'
        """
        self.index_to_block = block_mappings['index_to_block'] if block_mappings else None
        self.block_to_index = block_mappings['block_to_index'] if block_mappings else None
        self.index_to_biome = biome_mappings['index_to_biome'] if biome_mappings else None
        self.biome_to_index = biome_mappings['biome_to_index'] if biome_mappings else None
    
    @classmethod
    def from_dataset(cls, data_path):
        """Create mappings from a dataset file"""
        data = np.load(data_path, allow_pickle=True)
        voxels = data['voxels']
        biomes = data['biomes']
        
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}
        
        block_mappings = {'index_to_block': index_to_block, 'block_to_index': block_to_index}
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def from_arrays(cls, voxels, biomes):
        """Create mappings directly from numpy arrays"""
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}
        
        block_mappings = {'index_to_block': index_to_block, 'block_to_index': block_to_index}
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def load_mappings(cls, path):
        """Load pre-saved mappings"""
        mappings = torch.load(path)
        return cls(mappings['block_mappings'], mappings['biome_mappings'])
    
    def save_mappings(self, path):
        """Save mappings for later use"""
        torch.save({
            'block_mappings': {
                'index_to_block': self.index_to_block,
                'block_to_index': self.block_to_index
            },
            'biome_mappings': {
                'index_to_biome': self.index_to_biome,
                'biome_to_index': self.biome_to_index
            }
        }, path)
    
    def convert_to_original_blocks(self, data):
        """
        Convert from indices back to original block IDs.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_blocks), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original blocks
        if len(data.shape) == 4:  # Batch dimension present
            return torch.tensor([[[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in slice_]
                                for slice_ in data])
        else:  # No batch dimension
            return torch.tensor([[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in data])

    def convert_to_original_biomes(self, data):
        """
        Convert from indices back to original biome strings.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded biomes [B, C, H, W, D] or [C, H, W, D]
                - indexed biomes [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of original biome strings with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_biomes), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.biome_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original biomes
        if len(data.shape) == 4:  # Batch dimension present
            return np.array([[[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in data])