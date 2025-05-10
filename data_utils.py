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
def combine_chunk_files(input_directory, output_file_path):
    """
    Combines multiple Minecraft chunk NPZ files from a directory into a single NPZ file.
    
    Args:
        input_directory (str or Path): Path to directory containing NPZ files with voxels and biome data
        output_file_path (str or Path): Path where the combined data will be saved
    
    Returns:
        dict: Information about the combined data including counts and shapes
    """
    input_directory = Path(input_directory)
    output_file_path = Path(output_file_path)
    
    # Ensure input directory exists
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_directory}")
    
    # Find all NPZ files in the directory
    npz_files = list(input_directory.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in directory: {input_directory}")
    
    print(f"Found {len(npz_files)} NPZ files in {input_directory}")
    
    # Load the first file to determine shapes
    first_data = np.load(npz_files[0], allow_pickle=True)
    voxel_shape = first_data['voxels'].shape
    biome_shape = first_data['biome'].shape if 'biome' in first_data else None
    
    # Prepare arrays to hold all data
    all_voxels = []
    all_biomes = []
    
    # Process all files
    for i, file_path in enumerate(npz_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(npz_files)}")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            voxels = data['voxels']
            
            # Verify shape consistency
            if voxels.shape != voxel_shape:
                print(f"Warning: Skipping file {file_path} due to shape mismatch. "
                      f"Expected {voxel_shape}, got {voxels.shape}")
                continue
            
            all_voxels.append(voxels)
            
            # Handle biomes if present
            if 'biome' in data:
                biomes = data['biome']
                if biome_shape is None:
                    biome_shape = biomes.shape
                if biomes.shape != biome_shape:
                    print(f"Warning: Biome shape mismatch in {file_path}. "
                          f"Expected {biome_shape}, got {biomes.shape}")
                all_biomes.append(biomes)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Stack all data
    combined_voxels = np.stack(all_voxels)
    combined_biomes = np.stack(all_biomes) if all_biomes else None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_file_path.parent, exist_ok=True)
    
    # Save combined data
    if combined_biomes is not None:
        np.savez(output_file_path, chunks=combined_voxels, biomes=combined_biomes)
        print(f"Saved combined data to {output_file_path}")
        print(f"Combined voxel shape: {combined_voxels.shape}")
        print(f"Combined biome shape: {combined_biomes.shape}")
        return {
            "num_chunks": len(all_voxels),
            "voxel_shape": combined_voxels.shape,
            "biome_shape": combined_biomes.shape
        }
    else:
        np.savez(output_file_path, chunks=combined_voxels)
        print(f"Saved combined voxel data to {output_file_path} (no biome data found)")
        print(f"Combined voxel shape: {combined_voxels.shape}")
        return {
            "num_chunks": len(all_voxels),
            "voxel_shape": combined_voxels.shape,
            "biome_shape": None
        }
    
    
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
        self.block_to_str =  {
            0: "ACACIA_DOOR",
            1: "ACACIA_FENCE",
            2: "ACACIA_FENCE_GATE",
            3: "ACACIA_STAIRS",
            4: "ACTIVATOR_RAIL",
            5: "AIR",
            6: "ANVIL",
            7: "BARRIER",
            8: "BEACON",
            9: "BED",
            10: "BEDROCK",
            11: "BEETROOTS",
            12: "BIRCH_DOOR",
            13: "BIRCH_FENCE",
            14: "BIRCH_FENCE_GATE",
            15: "BIRCH_STAIRS",
            16: "BLACK_GLAZED_TERRACOTTA",
            17: "BLACK_SHULKER_BOX",
            18: "BLUE_GLAZED_TERRACOTTA",
            19: "BLUE_SHULKER_BOX",
            20: "BONE_BLOCK",
            21: "BOOKSHELF",
            22: "BREWING_STAND",
            23: "BRICK_BLOCK",
            24: "BRICK_STAIRS",
            25: "BROWN_GLAZED_TERRACOTTA",
            26: "BROWN_MUSHROOM",
            27: "BROWN_MUSHROOM_BLOCK",
            28: "BROWN_SHULKER_BOX",
            29: "CACTUS",
            30: "CAKE",
            31: "CARPET",
            32: "CARROTS",
            33: "CAULDRON",
            34: "CHAIN_COMMAND_BLOCK",
            35: "CHEST",
            36: "CHORUS_FLOWER",
            37: "CHORUS_PLANT",
            38: "CLAY",
            39: "COAL_BLOCK",
            40: "COAL_ORE",
            41: "COBBLESTONE",
            42: "COBBLESTONE_WALL",
            43: "COCOA",
            44: "COMMAND_BLOCK",
            45: "CONCRETE",
            46: "CONCRETE_POWDER",
            47: "CRAFTING_TABLE",
            48: "CYAN_GLAZED_TERRACOTTA",
            49: "CYAN_SHULKER_BOX",
            50: "DARK_OAK_DOOR",
            51: "DARK_OAK_FENCE",
            52: "DARK_OAK_FENCE_GATE",
            53: "DARK_OAK_STAIRS",
            54: "DAYLIGHT_DETECTOR",
            55: "DAYLIGHT_DETECTOR_INVERTED",
            56: "DEADBUSH",
            57: "DETECTOR_RAIL",
            58: "DIAMOND_BLOCK",
            59: "DIAMOND_ORE",
            60: "DIRT",
            61: "DISPENSER",
            62: "DOUBLE_PLANT",
            63: "DOUBLE_STONE_SLAB",
            64: "DOUBLE_STONE_SLAB2",
            65: "DOUBLE_WOODEN_SLAB",
            66: "DRAGON_EGG",
            67: "DROPPER",
            68: "EMERALD_BLOCK",
            69: "EMERALD_ORE",
            70: "ENCHANTING_TABLE",
            71: "ENDER_CHEST",
            72: "END_BRICKS",
            73: "END_GATEWAY",
            74: "END_PORTAL",
            75: "END_PORTAL_FRAME",
            76: "END_ROD",
            77: "END_STONE",
            78: "FARMLAND",
            79: "FENCE",
            80: "FENCE_GATE",
            81: "FIRE",
            82: "FLOWER_POT",
            83: "FLOWING_LAVA",
            84: "FLOWING_WATER",
            85: "FROSTED_ICE",
            86: "FURNACE",
            87: "GLASS",
            88: "GLASS_PANE",
            89: "GLOWSTONE",
            90: "GOLDEN_RAIL",
            91: "GOLD_BLOCK",
            92: "GOLD_ORE",
            93: "GRASS",
            94: "GRASS_PATH",
            95: "GRAVEL",
            96: "GRAY_GLAZED_TERRACOTTA",
            97: "GRAY_SHULKER_BOX",
            98: "GREEN_GLAZED_TERRACOTTA",
            99: "GREEN_SHULKER_BOX",
            100: "HARDENED_CLAY",
            101: "HAY_BLOCK",
            102: "HEAVY_WEIGHTED_PRESSURE_PLATE",
            103: "HOPPER",
            104: "ICE",
            105: "IRON_BARS",
            106: "IRON_BLOCK",
            107: "IRON_DOOR",
            108: "IRON_ORE",
            109: "IRON_TRAPDOOR",
            110: "JUKEBOX",
            111: "JUNGLE_DOOR",
            112: "JUNGLE_FENCE",
            113: "JUNGLE_FENCE_GATE",
            114: "JUNGLE_STAIRS",
            115: "LADDER",
            116: "LAPIS_BLOCK",
            117: "LAPIS_ORE",
            118: "LAVA",
            119: "LEAVES",
            120: "LEAVES2",
            121: "LEVER",
            122: "LIGHT_BLUE_GLAZED_TERRACOTTA",
            123: "LIGHT_BLUE_SHULKER_BOX",
            124: "LIGHT_WEIGHTED_PRESSURE_PLATE",
            125: "LIME_GLAZED_TERRACOTTA",
            126: "LIME_SHULKER_BOX",
            127: "LIT_FURNACE",
            128: "LIT_PUMPKIN",
            129: "LIT_REDSTONE_LAMP",
            130: "LIT_REDSTONE_ORE",
            131: "LOG",
            132: "LOG2",
            133: "MAGENTA_GLAZED_TERRACOTTA",
            134: "MAGENTA_SHULKER_BOX",
            135: "MAGMA",
            136: "MELON_BLOCK",
            137: "MELON_STEM",
            138: "MOB_SPAWNER",
            139: "MONSTER_EGG",
            140: "MOSSY_COBBLESTONE",
            141: "MYCELIUM",
            142: "NETHERRACK",
            143: "NETHER_BRICK",
            144: "NETHER_BRICK_FENCE",
            145: "NETHER_BRICK_STAIRS",
            146: "NETHER_WART",
            147: "NETHER_WART_BLOCK",
            148: "NOTEBLOCK",
            149: "OAK_STAIRS",
            150: "OBSERVER",
            151: "OBSIDIAN",
            152: "ORANGE_GLAZED_TERRACOTTA",
            153: "ORANGE_SHULKER_BOX",
            154: "PACKED_ICE",
            155: "PINK_GLAZED_TERRACOTTA",
            156: "PINK_SHULKER_BOX",
            157: "PISTON",
            158: "PISTON_EXTENSION",
            159: "PISTON_HEAD",
            160: "PLANKS",
            161: "PORTAL",
            162: "POTATOES",
            163: "POWERED_COMPARATOR",
            164: "POWERED_REPEATER",
            165: "PRISMARINE",
            166: "PUMPKIN",
            167: "PUMPKIN_STEM",
            168: "PURPLE_GLAZED_TERRACOTTA",
            169: "PURPLE_SHULKER_BOX",
            170: "PURPUR_BLOCK",
            171: "PURPUR_DOUBLE_SLAB",
            172: "PURPUR_PILLAR",
            173: "PURPUR_SLAB",
            174: "PURPUR_STAIRS",
            175: "QUARTZ_BLOCK",
            176: "QUARTZ_ORE",
            177: "QUARTZ_STAIRS",
            178: "RAIL",
            179: "REDSTONE_BLOCK",
            180: "REDSTONE_LAMP",
            181: "REDSTONE_ORE",
            182: "REDSTONE_TORCH",
            183: "REDSTONE_WIRE",
            184: "RED_FLOWER",
            185: "RED_GLAZED_TERRACOTTA",
            186: "RED_MUSHROOM",
            187: "RED_MUSHROOM_BLOCK",
            188: "RED_NETHER_BRICK",
            189: "RED_SANDSTONE",
            190: "RED_SANDSTONE_STAIRS",
            191: "RED_SHULKER_BOX",
            192: "REEDS",
            193: "REPEATING_COMMAND_BLOCK",
            194: "SAND",
            195: "SANDSTONE",
            196: "SANDSTONE_STAIRS",
            197: "SAPLING",
            198: "SEA_LANTERN",
            199: "SILVER_GLAZED_TERRACOTTA",
            200: "SILVER_SHULKER_BOX",
            201: "SKULL",
            202: "SLIME",
            203: "SNOW",
            204: "SNOW_LAYER",
            205: "SOUL_SAND",
            206: "SPONGE",
            207: "SPRUCE_DOOR",
            208: "SPRUCE_FENCE",
            209: "SPRUCE_FENCE_GATE",
            210: "SPRUCE_STAIRS",
            211: "STAINED_GLASS",
            212: "STAINED_GLASS_PANE",
            213: "STAINED_HARDENED_CLAY",
            214: "STANDING_BANNER",
            215: "STANDING_SIGN",
            216: "STICKY_PISTON",
            217: "STONE",
            218: "STONEBRICK",
            219: "STONE_BRICK_STAIRS",
            220: "STONE_BUTTON",
            221: "STONE_PRESSURE_PLATE",
            222: "STONE_SLAB",
            223: "STONE_SLAB2",
            224: "STONE_STAIRS",
            225: "STRUCTURE_BLOCK",
            226: "STRUCTURE_VOID",
            227: "TALLGRASS",
            228: "TNT",
            229: "TORCH",
            230: "TRAPDOOR",
            231: "TRAPPED_CHEST",
            232: "TRIPWIRE",
            233: "TRIPWIRE_HOOK",
            234: "UNLIT_REDSTONE_TORCH",
            235: "UNPOWERED_COMPARATOR",
            236: "UNPOWERED_REPEATER",
            237: "VINE",
            238: "WALL_BANNER",
            239: "WALL_SIGN",
            240: "WATER",
            241: "WATERLILY",
            242: "WEB",
            243: "WHEAT",
            244: "WHITE_GLAZED_TERRACOTTA",
            245: "WHITE_SHULKER_BOX",
            246: "WOODEN_BUTTON",
            247: "WOODEN_DOOR",
            248: "WOODEN_PRESSURE_PLATE",
            249: "WOODEN_SLAB",
            250: "WOOL",
            251: "YELLOW_FLOWER",
            252: "YELLOW_GLAZED_TERRACOTTA",
            253: "YELLOW_SHULKER_BOX"
        }
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

    def get_block_name_from_index(self, index):
        """
        Convert a single index to the corresponding block name.
        
        Args:
            index: int - the index of the block type
        Returns:
            str - the name of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        block_id = self.index_to_block[index]
        return self.block_to_str.get(block_id, f"UNKNOWN_BLOCK_{block_id}")
    
    def get_block_id_from_index(self, index):
        """
        Convert a single index to the corresponding block ID.
        
        Args:
            index: int - the index of the block type
        Returns:
            int - the ID of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        return self.index_to_block[index]
    
    def convert_to_block_names(self, data):
        """
        Convert from indices or IDs to block names.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
                - block ID blocks [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of block names with shape [B, H, W, D] or [H, W, D]
        """
        # First convert to block IDs if necessary
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            # One-hot encoded, convert to indices first
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
            # Then convert indices to block IDs
            data = self.convert_to_original_blocks(data)
        elif self.index_to_block is not None and (
            (len(data.shape) == 4 and torch.max(data) < len(self.index_to_block)) or
            (len(data.shape) == 3 and torch.max(data) < len(self.index_to_block))
        ):
            # These are indices, convert to block IDs
            data = self.convert_to_original_blocks(data)
        
        # Now data contains block IDs, convert to names
        if data.dim() == 4:  # Batch dimension present
            return np.array([[[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in data])