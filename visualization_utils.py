import torch
import numpy as np
import matplotlib.pyplot as plt

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

        # Get dimensions after rotation
        depth, height, width = voxels.shape
        
        # Calculate scaling factors based on 6x6x6 being the default size
        default_size = 6
        scale_x = depth / default_size
        scale_y = height / default_size
        scale_z = width / default_size
        
        # Create axis if not provided
        if ax is None:
            # Scale figure size based on dimensions
            fig = plt.figure(figsize=(8 * scale_x, 8 * scale_y))
            ax = fig.add_subplot(111, projection='3d')

        # Adjust the axis aspect ratio to match the data dimensions
        ax.set_box_aspect((scale_x, scale_y, scale_z))

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