from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np



def get_mnist_dataloaders(batch_size, img_size, val_split=0.1):
    # Transformations applied on each image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize image to the specified size
        transforms.ToTensor(),  # Transform the image to a torch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalizing the dataset using mean and std
    ])

    # Loading MNIST dataset from torchvision.datasets
    full_dataset = datasets.MNIST(
        root='./data',  # Directory where the data is located/downloaded
        train=True,  # Use training data
        download=True,  # Download if it's not already downloaded
        transform=transform  # Apply the transformations defined above
    )

    # Splitting dataset into train and validation
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Creating dataloaders for training and validation
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle the training data
        num_workers=2  # Number of workers for loading data
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle validation data
        num_workers=2  # Number of workers for loading data
    )

    return train_loader, val_loader

class MinecraftDataset(Dataset):
    def __init__(self, data_path):
        # Load data
        chunks = torch.from_numpy(np.load(data_path))
        
        # Create block type mappings
        unique_blocks = torch.unique(chunks).numpy()
        self.num_block_types = len(unique_blocks)
        
        # Create mappings
        self.block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        self.index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Convert entire dataset at once to new indices
        self.chunks = torch.tensor([[[[self.block_to_index[int(b)] 
                                     for b in row] 
                                    for row in layer]
                                   for layer in slice_]
                                  for slice_ in chunks])
        
        # Store air block index
        self.air_idx = self.block_to_index[5]
        self.target_size = 24

        # Pre-process all chunks (pad and one-hot encode)
        pad_size = self.target_size - self.chunks.size(-1)
        padded_chunks = F.pad(self.chunks, 
                            (0, pad_size, 0, pad_size, 0, pad_size), 
                            value=self.air_idx)
        # Convert to one-hot [N, C, H, W, D]
        self.processed_chunks = F.one_hot(
            padded_chunks.long(), 
            num_classes=self.num_block_types
        ).permute(0, 4, 1, 2, 3).float()
        
        print(f"Loaded {len(self.chunks)} chunks of size {self.chunks.shape[1:]}")
        print(f"Number of unique block types: {self.num_block_types}")
        print(f'unique blocks original: {np.unique(chunks.numpy())}')
        print(f'unique blocks after: {np.unique(self.chunks.numpy())}')
        print(f"Block mapping: {self.block_to_index}")
        print(f"Air block is now index: {self.air_idx}")

    def __getitem__(self, idx):
        return self.processed_chunks[idx]

    def convert_to_original_blocks(self, data):
        """Convert from indices back to original block IDs"""
        if len(data.shape) == 5:  # [B, C, H, W, D] or [C, H, W, D]
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        return torch.tensor([[[[self.index_to_block[int(b)] 
                              for b in row]
                             for row in layer]
                            for layer in slice_]
                           for slice_ in data])

    def __len__(self):
        return len(self.chunks)



def get_minecraft_dataloaders(data_path, batch_size=32, val_split=0.1, num_workers=4):
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

    # Store conversion function in the loaders
    train_loader.convert_to_original_blocks = dataset.convert_to_original_blocks
    val_loader.convert_to_original_blocks = dataset.convert_to_original_blocks
    
    print(f"\nDataloader details:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


class MinecraftVisualizer:
    def __init__(self):
        """Initialize the visualizer with the same block color mappings"""
        self.blocks_to_cols = {
            0: (0.5, 0.25, 0.0),    # light brown
            29: "#006400",
            38: "#B8860B",
            60: "brown",
            92: "gold",
            93: "green",
            115: "brown",
            119: "forestgreen",
            120: "forestgreen",
            194: "yellow",
            217: "gray",
            227: "#90EE90",
            240: "blue",
            40: "#2F4F4F",
            62: "#228B22",
            108: "#BEBEBE",
            131: "saddlebrown",
            132: "saddlebrown",
            95: "lightgray",
            243: "wheat",
            197: "limegreen",
            166: "orange",
            167: "#FF8C00",
            184: "#FFA07A",
            195: "tan",
            250: "white",
            251: "gold",
        }

    def visualize_chunk(self, voxels, ax=None):
        """
        Create a 3D visualization of a Minecraft chunk using the original plotting logic.
        
        Args:
            voxels: torch.Tensor [C,H,W,D] (one-hot) or numpy.ndarray [H,W,D] (block IDs)
            ax: Optional matplotlib axis
        """
        # Convert one-hot to block IDs if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:  # One-hot encoded [C,H,W,D]
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()

        # Apply the same transformations as original
        voxels = voxels.transpose(2, 0, 1)
        # Rotate the voxels 90 degrees around the height axis
        voxels = np.rot90(voxels, 1, (0, 1))

        # Create axis if not provided
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot non-air blocks
        other_vox = (voxels != 5) & (voxels != -1)
        
        # Plot each block type with its color
        for block_id in np.unique(voxels[other_vox]):
            if block_id not in self.blocks_to_cols:
                # print(f"Unknown block id: {block_id}")
                continue
            ax.voxels(voxels == block_id, facecolors=self.blocks_to_cols[int(block_id)])
            other_vox = other_vox & (voxels != block_id)

        # Plot remaining blocks in red with black edges
        ax.voxels(other_vox, edgecolor="k", facecolor="red")
        
        return ax

        
def visualize_images(data_loader):
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.numpy()  # Convert images to numpy for visualization

    fig, axes = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()  # Remove channel dimension
        ax.imshow(img, cmap='gray')  # Display image in grayscale
        ax.axis('off')  # Hide axes ticks
    plt.show()