import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import colorsys


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
    

import pyvista as pv
import matplotlib.cm as cm  # Add this import


class MinecraftVisualizerPyVista:
    def __init__(self):
        """Initialize with same block color mappings"""
        self.blocks_to_cols = {
            0: (0.5, 0.25, 0.0),    # light brown
            10: 'black', # bedrock
            29: "#006400", # cacutus
            38: "#B8860B",  # clay
            60: "brown",  # dirt
            92: "gold",  # gold ore
            93: "green",  # grass
            115: "brown",  # ladder...?
            119: (.02, .28, .16, 0.9),  # transparent forest green (RGBA) for leaves
            120: (.02, .28, .16, 0.9),  # leaves2
            194: "yellow",  # sand
            217: "gray",  # stone
            240: (0.0, 0.0, 1.0, 0.4),  # water
            227: (0.0, 1.0, 0.0, .3), # tall grass
            237: (0.33, 0.7, 0.33, 0.3), # vine
            40: "#2F4F4F",  # coal ore
            62: "#228B22",  # double plant
            108: "#BEBEBE",  # iron ore
            131: "saddlebrown",  # log1
            132: "saddlebrown",  #log2
            95: "lightgray",  # gravel
            243: "wheat",  # wheat
            197: "limegreen",  # sapling
            166: "orange",  #pumpkin
            167: "#FF8C00",  # pumpkin stem
            184: "#FFA07A",  # red flower
            195: "tan",  # sandstone
            250: "white",  #wool 
            251: "gold",   #yellow flower
        }
        try:
            import panel as pn
            pn.extension('vtk')
            pv.set_jupyter_backend('trame')
        except ImportError:
            print("Please install panel with: pip install panel")
        
    def visualize_chunk(self, voxels, highlight_latents=None, plotter=None, interactive=False, show_axis=True, wireframe_highlight=False):
        """Visualize a single chunk with optional latent space highlighting"""
        # Convert to numpy if needed
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
                
        # Create grid
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")
        
        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)
        
        # Remove existing lights
        plotter.remove_all_lights()
        
        # Add the three-point lighting setup
        plotter.add_light(pv.Light(
            position=(1, -1, 1),
            intensity=1.0,
            color='white'
        ))
        
        plotter.add_light(pv.Light(
            position=(-1, 1, 0.5),
            intensity=0.5,
            color='white'
        ))
        
        plotter.add_light(pv.Light(
            position=(-0.5, -0.5, -1),
            intensity=0.3,
            color='white'
        ))
        
        # Plot each block type
        mask = (voxels != 5) & (voxels != -1)
        unique_blocks = np.unique(voxels[mask])
        
        for block_id in unique_blocks:
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)
            
        # Add highlight boxes if specified
        if highlight_latents is not None:
            coords = np.array(highlight_latents)
            scale = 4
            
            # Find the bounds of all coordinates
            d_coords = coords[:, 0]
            h_coords = coords[:, 1]
            w_coords = coords[:, 2]
            
            # Create bounds for the cube
            bounds = (
                abs(5 - d_coords.max()) * scale, abs(5 - d_coords.min() + 1) * scale,  # X bounds
                w_coords.min() * scale, (w_coords.max() + 1) * scale,                  # Y bounds
                h_coords.min() * scale, (h_coords.max() + 1) * scale                   # Z bounds
            )
            
            # Add transparent faces
            cube_faces = pv.Cube(bounds=bounds)
            plotter.add_mesh(cube_faces, 
                            color='red',
                            opacity=0.0 if wireframe_highlight else 0.1,
                            show_edges=False,
                            lighting=False)
            
            # Add opaque edges as a separate wireframe
            cube_edges = pv.Cube(bounds=bounds)
            plotter.add_mesh(cube_edges,
                            color='red',
                            style='wireframe',
                            line_width=1,
                            opacity=.5,
                            lighting=False)
                

        # Add dummy cube for bounds
        outline = pv.Cube(bounds=(0, 24, 0, 24, 0, 24))
        plotter.add_mesh(outline, opacity=0.0)
        
        # Add bounds with consistent settings
        if show_axis:
            plotter.show_bounds(
                grid='back',
                location='back',
                font_size=8,
                bold=False,
                font_family='arial',
                use_2d=False,
                bounds=[0, 24, 0, 24, 0, 24],
                axes_ranges=[0, 24, 0, 24, 0, 24],
                padding=0.0,
                n_xlabels=2,
                n_ylabels=2,
                n_zlabels=2
            )
        
        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)
        
        return plotter
    
    def visualize_interactive(self, voxels):
        # Convert to numpy if needed
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
                
        # Create grid
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")
        
        # Create plotter
        plotter = pv.Plotter(notebook=True)
        
        # Remove existing lights
        plotter.remove_all_lights()
        
        # Add custom lights
        # Main light from top-front-right (sun-like)
        # Add a headlight (light from camera position)
        # Key light (main light, 45 degrees from front-right)
        plotter.add_light(pv.Light(
            position=(1, -1, 1),
            intensity=1.0,
            color='white'
        ))
        
        # Fill light (softer light from opposite side)
        plotter.add_light(pv.Light(
            position=(-1, 1, 0.5),
            intensity=0.5,
            color='white'
    ))
        
        # Back light (rim lighting from behind)
        plotter.add_light(pv.Light(
            position=(-0.5, -0.5, -1),
            intensity=0.3,
            color='white'
    ))
        
        # Plot each block type
        mask = (voxels != 5) & (voxels != -1)
        unique_blocks = np.unique(voxels[mask])
        
        for block_id in unique_blocks:
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,   # Thin edges
                        edge_opacity=0.2,
                        lighting=True)
        
        # Add a dummy cube to force the bounds
        outline = pv.Cube(bounds=(0, 24, 0, 24, 0, 24))
        plotter.add_mesh(outline, opacity=0.0)  # Invisible cube to set bounds
        
        # Add clean axes with consistent range
        plotter.show_bounds(
            grid='back',
            location='back',
            # all_edges=True,
            # ticks=None,
            font_size=8,
            bold=False,
            font_family='arial',
            use_2d=False,
            bounds=[0, 24, 0, 24, 0, 24],
            axes_ranges=[0, 24, 0, 24, 0, 24],
            padding=0.0,
            n_xlabels=2,
            n_ylabels=2,
            n_zlabels=2,
            # show_xlabels=False,
            # show_ylabels=False,
            # show_zlabels=False
        )
        
        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)
        
        return plotter
    
    def visualize_latent_space(self, latents, latent_type="style", plotter=None):
        """
        Interactive visualization of latent codes in their spatial positions.
        """
        # Handle batch dimension if present
        if isinstance(latents, torch.Tensor):
            if latents.dim() == 4:  # [1,6,6,6]
                latents = latents.squeeze(0)
            latents = latents.detach().cpu().numpy()
        
        if plotter is None:
            plotter = pv.Plotter(notebook=True)
        
        # Add title to the plot
        plotter.add_title(f"{latent_type.capitalize()} Latent Codes", font_size=16)
        
        # Store cubes and text separately
        cubes_by_layer = {i: [] for i in range(6)}
        text_actors = {}  # Store one text actor per layer
        
        # Create colormap
        import matplotlib.cm as cm
        max_code = np.max(latents)
        min_code = np.min(latents)
        norm = plt.Normalize(min_code, max_code)
        cmap = cm.viridis  # You can try other colormaps like 'plasma', 'magma', etc.

        # Create all cubes and text for all layers
        for layer in range(6):  # This is the height (H)
            # Create all text for this layer at once
            centers = []
            labels = []
            for d in range(6):  # This is depth (D)
                for w in range(6):  # This is width (W)
                    # Transform coordinates for visualization while keeping original indices for values
                    x = abs(5 - d)  # Transform depth to x-coordinate
                    # Get color for this latent code
                    latent_value = latents[d, layer, w]
                    rgba_color = cmap(norm(latent_value))

                    cube = pv.Cube(
                        bounds=(
                            x * 4, (x + 1) * 4,        # X (transformed from depth)
                            w * 4, (w + 1) * 4,        # Y (width)
                            layer * 4, (layer + 1) * 4  # Z (height)
                        )
                    )
                    
                    # Add cube
                    actor = plotter.add_mesh(cube,
                                        color=rgba_color[:3],
                                        opacity=0.1,
                                        show_edges=True,
                                        edge_color='blue',
                                        line_width=2)
                    cubes_by_layer[layer].append(actor)
                    
                    # Collect center and label for text
                    centers.append(cube.center)
                    labels.append(f"{latents[d, layer, w]}")  # [D,H,W] ordering
            
            # Create text actor for this layer
            text_actor = plotter.add_point_labels(
                centers,
                labels,
                font_size=16,
                always_visible=True,
                shape_opacity=0.0,
                text_color='black'
            )
            text_actors[layer] = text_actor
            
            # Hide all layers except first
            if layer != 0:
                for cube in cubes_by_layer[layer]:
                    cube.visibility = False
                plotter.remove_actor(text_actor)
        
        def update_layer(value):
            layer = int(value)
            # Update cube visibility
            for l in range(6):
                for cube in cubes_by_layer[l]:
                    cube.visibility = (l == layer)
                
                # Update text visibility
                if l == layer:
                    plotter.add_actor(text_actors[l])
                else:
                    plotter.remove_actor(text_actors[l])
            
            plotter.render()
        
        # Add slider widget
        slider = plotter.add_slider_widget(
            update_layer,
            [0, 5],
            value=0,
            title='Layer',
            pointa=(0.025, 0.1),
            pointb=(0.225, 0.1),
            style='modern',
            fmt='%0.0f'
        )
        
        # Set initial camera position
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.2)
        
        return plotter
    
    def create_voxel_gif(self, voxel_sequence, output_path='voxel_animation.gif', duration=2.0, highlight_latents=None, show_axis=True, transparent_background=False, fps=2, wireframe_highlight=False):
        """
        Create an animated visualization using PIL with proper frame disposal.
        """
        from PIL import Image
        
        frames = []
        
        # Create frames
        for voxels in voxel_sequence:
            plotter = pv.Plotter(off_screen=True)
            if highlight_latents is not None:
                plotter = self.visualize_chunk(voxels, plotter=plotter, highlight_latents=highlight_latents, show_axis=show_axis, wireframe_highlight=wireframe_highlight)
            else:
                plotter = self.visualize_chunk(voxels, plotter=plotter, show_axis=show_axis)
            
            frame = plotter.screenshot(transparent_background=transparent_background, return_img=True)
            frames.append(Image.fromarray(frame))
            plotter.close()

        # Save as GIF with disposal method
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),  # Duration in milliseconds
            loop=0,
            disposal=2,  # 2 = restore to background color
            transparency=0,  # Index of transparent color
            optimize=False  # Don't optimize to preserve transparency
        )
        
        return output_path
        
    def visualize_latent_space_with_blocks(self, latents, block_data, latent_type="style", plotter=None):
        """
        Interactive visualization of latent codes next to minecraft blocks.
        
        Args:
            latents: Tensor of latent indices [6,6,6] or [1,6,6,6]
            block_data: Tensor of minecraft blocks
            latent_type: "style" or "struct" for color scheme
            plotter: Optional existing plotter
        """
        if plotter is None:
            plotter = pv.Plotter(notebook=True)
        
        # Handle batch dimension if present
        if isinstance(latents, torch.Tensor):
            if latents.dim() == 4:
                latents = latents.squeeze(0)
            latents = latents.detach().cpu().numpy()
        
        # Add title to the plot
        plotter.add_title(f"{latent_type.capitalize()} Latent Codes", font_size=16)
        
        # Store cubes and text separately
        cubes_by_layer = {i: [] for i in range(6)}
        text_actors = {}
        
        # Create colormap
        import matplotlib.cm as cm
        max_code = np.max(latents)
        min_code = np.min(latents)
        norm = plt.Normalize(min_code, max_code)
        cmap = cm.viridis
        
        # Create all cubes and text for latent visualization (offset in x direction)
        offset = 24  # Space between visualizations
        for layer in range(6):
            centers = []
            labels = []
            for d in range(6):
                for w in range(6):
                    x = abs(5 - d)
                    latent_value = latents[d, layer, w]
                    rgba_color = cmap(norm(latent_value))
                    
                    cube = pv.Cube(bounds=(
                        x * 4 + offset, (x + 1) * 4 + offset,  # Offset in x direction
                        w * 4, (w + 1) * 4,
                        layer * 4, (layer + 1) * 4
                    ))
                    
                    actor = plotter.add_mesh(cube,
                                        color=rgba_color[:3],
                                        opacity=0.8,
                                        show_edges=True,
                                        edge_color='black',
                                        line_width=1)
                    cubes_by_layer[layer].append(actor)
                    
                    centers.append(cube.center)
                    labels.append(f"{latents[d, layer, w]}")
            
            text_actor = plotter.add_point_labels(
                centers,
                labels,
                font_size=16,
                always_visible=True,
                shape_opacity=0.0,
                text_color='black'
            )
            text_actors[layer] = text_actor
            
            if layer != 0:
                for cube in cubes_by_layer[layer]:
                    cube.visibility = False
                plotter.remove_actor(text_actor)
        
        # Add minecraft visualization
        self.visualize_chunk(block_data, plotter=plotter)
        
        def update_layer(value):
            layer = int(value)
            for l in range(6):
                for cube in cubes_by_layer[l]:
                    cube.visibility = (l == layer)
                
                if l == layer:
                    plotter.add_actor(text_actors[l])
                else:
                    plotter.remove_actor(text_actors[l])
            
            plotter.render()
        
        # Add slider widget
        slider = plotter.add_slider_widget(
            update_layer,
            [0, 5],
            value=0,
            title='Layer',
            pointa=(0.025, 0.1),
            pointb=(0.225, 0.1),
            style='modern',
            fmt='%0.0f'
        )
        
        # # Add colorbar
        # plotter.add_scalar_bar(title=f'{latent_type.capitalize()} Latent Values',
        #                     n_labels=5,
        #                     mapper=plt.cm.ScalarMappable(norm=norm, cmap=cmap))
        
        # Set initial camera position
        plotter.camera_position = 'iso'
        plotter.camera.zoom(0.8)  # Zoom out a bit to see both visualizations
        
        return plotter
    
    def visualize_both_codes_with_blocks(self, style_latents, struct_latents, block_data, plotter=None):
        """
        More efficient visualization of style and structure codes next to minecraft blocks.
        Uses pre-computed actors and optimized updates.
        """
        if plotter is None:
            plotter = pv.Plotter(notebook=False)
        
        # Handle batch dimensions and convert to numpy
        if isinstance(style_latents, torch.Tensor):
            if style_latents.dim() == 4:
                style_latents = style_latents.squeeze(0)
            style_latents = style_latents.detach().cpu().numpy()
        if isinstance(struct_latents, torch.Tensor):
            if struct_latents.dim() == 4:
                struct_latents = struct_latents.squeeze(0)
            struct_latents = struct_latents.detach().cpu().numpy()
        
        # Add title
        plotter.add_title("Style and Structure Latent Codes", font_size=16)
        
        # Create colormaps
        import matplotlib.cm as cm
        style_norm = plt.Normalize(np.min(style_latents), np.max(style_latents))
        struct_norm = plt.Normalize(np.min(struct_latents), np.max(struct_latents))
        style_cmap = cm.viridis
        struct_cmap = cm.plasma
        
        # Pre-compute all actors for each layer
        style_actors = {i: [] for i in range(6)}
        struct_actors = {i: [] for i in range(6)}
        style_texts = {}
        struct_texts = {}
        
        # Create all actors at once
        for layer in range(6):
            style_centers = []
            struct_centers = []
            style_labels = []
            struct_labels = []
            
            for d in range(6):
                for w in range(6):
                    x = abs(5 - d)
                    
                    # Style latents (left side)
                    style_value = style_latents[d, layer, w]
                    style_color = style_cmap(style_norm(style_value))
                    style_cube = pv.Cube(bounds=(
                        x * 4 + 24, (x + 1) * 4 + 24,
                        w * 4, (w + 1) * 4,
                        layer * 4, (layer + 1) * 4
                    ))
                    style_actor = plotter.add_mesh(
                        style_cube,
                        color=style_color[:3],
                        opacity=0.8,
                        show_edges=True,
                        edge_color='black',
                        line_width=1
                    )
                    style_actors[layer].append(style_actor)
                    style_centers.append(style_cube.center)
                    style_labels.append(f"{style_value}")
                    
                    # Structure latents (right side)
                    struct_value = struct_latents[d, layer, w]
                    struct_color = struct_cmap(struct_norm(struct_value))
                    struct_cube = pv.Cube(bounds=(
                        x * 4, (x + 1) * 4,
                        w * 4 + 24, (w + 1) * 4 + 24,
                        layer * 4, (layer + 1) * 4
                    ))
                    struct_actor = plotter.add_mesh(
                        struct_cube,
                        color=struct_color[:3],
                        opacity=0.8,
                        show_edges=True,
                        edge_color='black',
                        line_width=1
                    )
                    struct_actors[layer].append(struct_actor)
                    struct_centers.append(struct_cube.center)
                    struct_labels.append(f"{struct_value}")
            
            # Create text actors for each layer
            style_text = plotter.add_point_labels(
                style_centers,
                style_labels,
                font_size=16,
                always_visible=True,
                shape_opacity=0.0,
                text_color='black'
            )
            struct_text = plotter.add_point_labels(
                struct_centers,
                struct_labels,
                font_size=16,
                always_visible=True,
                shape_opacity=0.0,
                text_color='black'
            )
            
            style_texts[layer] = style_text
            struct_texts[layer] = struct_text
            
            # Hide all layers except first
            if layer != 0:
                for actor in style_actors[layer]:
                    actor.visibility = False
                for actor in struct_actors[layer]:
                    actor.visibility = False
                plotter.remove_actor(style_text)
                plotter.remove_actor(struct_text)
        
        # Add static labels
        for label, pos, rot in [
            ("Style codes", (50, 12, 0), 90),
            ("Structure codes", (12, 50, 0), 180)
        ]:
            text = pv.Text3D(label, depth=0.3, height=2)
            text = text.rotate_z(rot)
            text = text.translate(pos)
            plotter.add_mesh(text, color='black')
        
        # Add minecraft visualization
        self.visualize_chunk(block_data, plotter=plotter)
        
        def update_layer(value):
            layer = int(value)
            # Use batch operations where possible
            for l in range(6):
                is_visible = (l == layer)
                # Update cube visibility
                for actor in style_actors[l]:
                    actor.visibility = is_visible
                for actor in struct_actors[l]:
                    actor.visibility = is_visible
                
                # Update text visibility
                if is_visible:
                    plotter.add_actor(style_texts[l])
                    plotter.add_actor(struct_texts[l])
                else:
                    plotter.remove_actor(style_texts[l])
                    plotter.remove_actor(struct_texts[l])
            
            plotter.render()
        
        # Add slider
        plotter.add_slider_widget(
            update_layer,
            [0, 5],
            value=0,
            title='Layer',
            pointa=(0.025, 0.1),
            pointb=(0.225, 0.1),
            style='modern',
            fmt='%0.0f'
        )
        
        # Set camera
        plotter.camera_position = 'iso'
        plotter.camera.zoom(0.7)
        
        return plotter
    
    def visualize_chunk_with_biomes(self, voxels, biomes, plotter=None, interactive=False):
        """
        3D visualization of a Minecraft chunk with biome overlay using PyVista.
        
        Args:
            voxels: numpy array/torch.Tensor of block IDs
            biomes: numpy array/torch.Tensor of biome strings
            plotter: Optional existing PyVista plotter
            interactive: Whether to create an interactive display
        """
        # Convert tensors to numpy if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:  # One-hot encoded
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()
        if isinstance(biomes, torch.Tensor):
            biomes = biomes.detach().cpu().numpy()

        # Apply the same transformations to both arrays
        voxels = voxels.transpose(2, 0, 1)
        voxels = np.rot90(voxels, 1, (0, 1))
        biomes = biomes.transpose(2, 0, 1)
        biomes = np.rot90(biomes, 1, (0, 1))

        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)

        # Remove existing lights and add three-point lighting
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
        plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
        plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))

        # First plot the regular blocks
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")

        # Plot each block type
        mask = (voxels != 5) & (voxels != -1)
        unique_blocks = np.unique(voxels[mask])
        
        for block_id in unique_blocks:
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)

        # Create a colormap for biomes using distinct RGB values
        unique_biomes = np.unique(biomes)
        num_biomes = len(unique_biomes)
        
        # Generate distinct colors using HSV color space
        hsv_colors = [(i/num_biomes, 0.8, 0.8) for i in range(num_biomes)]
        rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
        biome_color_map = dict(zip(unique_biomes, rgb_colors))
        
        # Plot biome overlay
        biome_grid = pv.ImageData()
        biome_grid.dimensions = np.array(biomes.shape) + 1
        
        legend_entries = []
        
        for biome in unique_biomes:
            # Create mask for this biome
            biome_mask = (biomes == biome).astype(float)
            biome_grid.cell_data["values"] = biome_mask.flatten(order="F")
            
            # Get RGB color for this biome
            rgb_color = biome_color_map[biome]
            
            # Add semi-transparent overlay
            threshold = biome_grid.threshold([0.5, 1.5])
            plotter.add_mesh(threshold,
                            color=rgb_color,
                            opacity=0.2,
                            show_edges=False,
                            lighting=False)
            
            # Add to legend entries - IMPORTANT: first item must be the RGB color
            legend_entries.append([str(biome), rgb_color])

        # Add dummy cube for bounds
        outline = pv.Cube(bounds=(0, 24, 0, 24, 0, 24))
        plotter.add_mesh(outline, opacity=0.0)

        # Add bounds with consistent settings
        plotter.show_bounds(
            grid='back',
            location='back',
            font_size=8,
            bold=False,
            font_family='arial',
            use_2d=False,
            bounds=[0, 24, 0, 24, 0, 24],
            axes_ranges=[0, 24, 0, 24, 0, 24],
            padding=0.0,
            n_xlabels=2,
            n_ylabels=2,
            n_zlabels=2
        )

        # Add legend if we have entries
        if legend_entries:
            plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9, 0.3))

        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)

        return plotter
    
    def visualize_latent_blocks(self, voxels, latent_coords, plotter=None, interactive=False, show_axis=True):
        """
        Visualize only the blocks corresponding to specific latent coordinates.
        
        Args:
            voxels: torch.Tensor [C,H,W,D] (one-hot) or numpy.ndarray [H,W,D] (block IDs)
            latent_coords: list of (d, h, w) coordinate tuples in the 6x6x6 latent space
            plotter: Optional existing plotter
            interactive: Whether to create an interactive display
            
        Returns:
            plotter: PyVista plotter object
        """
        # Convert to numpy if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:  # One-hot encoded [C,H,W,D]
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()
                
        # Apply the same transformations as original
        voxels = voxels.transpose(2, 0, 1)
        voxels = np.rot90(voxels, 1, (0, 1))
        
        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)
        
        # Remove existing lights
        plotter.remove_all_lights()
        
        # Add the three-point lighting setup
        plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
        plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
        plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))
        
        # Create a mask for all selected regions
        mask = np.zeros_like(voxels, dtype=bool)
        coords = np.array(latent_coords)
        scale = 4  # Each latent coordinate corresponds to a 4x4x4 block region
        
        # Extract coordinates
        d_coords = coords[:, 0]
        h_coords = coords[:, 1]
        w_coords = coords[:, 2]
        
        # Calculate bounds using the same transformation as highlight_latents
        x_start = abs(5 - d_coords.max()) * scale
        x_end = abs(5 - d_coords.min() + 1) * scale
        y_start = w_coords.min() * scale
        y_end = (w_coords.max() + 1) * scale
        z_start = h_coords.min() * scale
        z_end = (h_coords.max() + 1) * scale
        
        # Set the mask for the selected region
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
        
        # Create grid for the selected blocks only
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.cell_data["values"] = voxels.flatten(order="F")
        
        # Plot only the blocks within the selected regions
        unique_blocks = np.unique(voxels[mask])
        for block_id in unique_blocks:
            if block_id in [5, -1]:  # Skip air blocks
                continue
                
            # Create threshold for this block type within the masked region
            block_mask = (voxels == block_id) & mask
            if not block_mask.any():
                continue
                
            grid.cell_data["values"] = block_mask.flatten(order="F")
            threshold = grid.threshold([0.5, 1.5])
            
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)
        
        # Add dummy cube for bounds
        outline = pv.Cube(bounds=(0, 24, 0, 24, 0, 24))
        plotter.add_mesh(outline, opacity=0.0)
        
        # Add bounds with consistent settings
        if show_axis:
            plotter.show_bounds(
                grid='back',
                location='back',
                font_size=8,
                bold=False,
                font_family='arial',
                use_2d=False,
                bounds=[0, 24, 0, 24, 0, 24],
                axes_ranges=[0, 24, 0, 24, 0, 24],
                padding=0.0,
                n_xlabels=2,
                n_ylabels=2,
                n_zlabels=2
            )
            
        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1)
        
        return plotter
    
    def visualize_isolated_latent_blocks(self, voxels, latent_coords, plotter=None, interactive=False, show_axis=True):
        """
        Visualize blocks from specific latent coordinates isolated from their original position.
        Extracts the blocks and renders them centered at the origin for closer inspection.
        
        Args:
            voxels: torch.Tensor [C,H,W,D] (one-hot) or numpy.ndarray [H,W,D] (block IDs)
            latent_coords: list of (d, h, w) coordinate tuples in the 6x6x6 latent space
            plotter: Optional existing plotter
            interactive: Whether to create an interactive display
            
        Returns:
            plotter: PyVista plotter object
        """
        # Convert to numpy if needed
        if isinstance(voxels, torch.Tensor):
            if voxels.dim() == 4:
                voxels = voxels.detach().cpu()
                voxels = torch.argmax(voxels, dim=0).numpy()
            else:
                voxels = voxels.detach().cpu().numpy()
                
        # Apply the same transformations as original
        voxels = voxels.transpose(2, 0, 1)
        voxels = np.rot90(voxels, 1, (0, 1))
        
        # Create plotter if not provided
        if plotter is None:
            if interactive:
                plotter = pv.Plotter(notebook=True)
            else:
                plotter = pv.Plotter(off_screen=True)
        
        # Remove existing lights and add three-point lighting
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
        plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
        plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))
        
        # Extract coordinates
        coords = np.array(latent_coords)
        scale = 4
        
        d_coords = coords[:, 0]
        h_coords = coords[:, 1]
        w_coords = coords[:, 2]
        
        # Calculate original bounds
        x_start = abs(5 - d_coords.max()) * scale
        x_end = abs(5 - d_coords.min() + 1) * scale
        y_start = w_coords.min() * scale
        y_end = (w_coords.max() + 1) * scale
        z_start = h_coords.min() * scale
        z_end = (h_coords.max() + 1) * scale
        
        # Extract the subvolume
        subvolume = voxels[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Create grid for the extracted blocks
        grid = pv.ImageData()
        grid.dimensions = np.array(subvolume.shape) + 1
        grid.cell_data["values"] = subvolume.flatten(order="F")
        
        # Plot each block type in the extracted region
        unique_blocks = np.unique(subvolume)
        for block_id in unique_blocks:
            if block_id in [5, -1]:  # Skip air blocks
                continue
                
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            
            if block_id in self.blocks_to_cols:
                color = self.blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                color = (1.0, 0.0, 0.0)
                opacity = 0.2
            
            plotter.add_mesh(threshold, 
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                        edge_color='black',
                        line_width=.2,
                        edge_opacity=0.2,
                        lighting=True)
        
        # Add bounds with consistent settings scaled to the subvolume size
        size = max(subvolume.shape)
        if show_axis:
            plotter.show_bounds(
                grid='back',
                location='back',
                font_size=8,
                bold=False,
                font_family='arial',
                use_2d=False,
                bounds=[0, size, 0, size, 0, size],
                axes_ranges=[0, size, 0, size, 0, size],
                padding=0.0,
                n_xlabels=2,
                n_ylabels=2,
                n_zlabels=2
            )
        
        # Set camera position and zoom
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.5)  # Zoom in a bit more since we're looking at a smaller region
        
        return plotter
    
def display_minecraft_pyvista(vis, mc_visualizer, data, win_name="minecraft_display", title="Minecraft Chunks", nrow=4, save_path=None):
    """
    Display or save multiple minecraft chunks using PyVista.
    """
    # Convert to one-hot if needed
    if len(data.shape) == 4:  # [B, 20, 20, 20]
        data = F.one_hot(data.long(), num_classes=256).permute(0, 4, 1, 2, 3).float()
    
    # Create figure with subplots
    batch_size = min(data.shape[0], 16)  # Display up to 16 chunks
    ncols = nrow
    nrows = (batch_size + ncols - 1) // ncols
    
    # Calculate the size of the combined image
    single_size = 400  # Size of each subplot in pixels
    
    # Create a list to store individual chunk images
    chunk_images = []
    
    for i in range(batch_size):
        # Use the visualizer's method to create the plot
        plotter = mc_visualizer.visualize_chunk(data[i])
        
        # Render to image
        img = plotter.screenshot(window_size=(single_size, single_size), 
                               transparent_background=True, 
                               return_img=True)
        chunk_images.append(img)
        plotter.close()
    
    # Combine images into a grid
    grid_rows = []
    for row in range(nrows):
        row_images = chunk_images[row * ncols : (row + 1) * ncols]
        # Pad the last row if needed
        while len(row_images) < ncols:
            row_images.append(np.zeros_like(chunk_images[0]))
        grid_rows.append(np.concatenate(row_images, axis=1))
    
    combined_img = np.concatenate(grid_rows, axis=0)
    
    # Save if path provided
     # Save if path provided
    if save_path:
        # Just save directly without trying to create directories
        plt.imsave(save_path, combined_img)
    
    # Display in visdom if instance provided
    if vis is not None:
        vis.image(
            combined_img.transpose(2, 0, 1),  # Convert to CHW format
            win=win_name,
            opts=dict(
                title=title,
                caption=f'Batch of {batch_size} chunks'
            )
        )


