import matplotlib.pyplot as plt


__all__ = ("visualize_channels_over_time",)


channel_names = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
       'VIS006', 'VIS008', 'WV_062', 'WV_073']


def visualize_channels_over_time(images,  batch_idx=0, figsize=(12, 8), cmap='viridis', diff=False):
    """
    Visualize multi-channel images over time, handling both single and multiple timesteps.
    
    Args:
        images: Tensor of shape (batch, channels, time, height, width)
        channel_names: List of names for each channel
        batch_idx: Which batch element to visualize
        figsize: Size of the figure
        cmap: Colormap to use for visualization
    """
    n_channels = images.shape[1]
    n_timesteps = images.shape[2]
    

    
    # Create a grid of subplots
    if n_timesteps == 1:
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize)
        axes = axes.reshape(-1, 1)  # Reshape to 2D array for consistent indexing
    else:
        fig, axes = plt.subplots(n_channels, n_timesteps, figsize=figsize)
    
    # Set the spacing between subplots
    plt.subplots_adjust(hspace=0.2, wspace=-0.5)
    
    # Iterate through channels and timesteps
    for channel in range(n_channels):
        for timestep in range(n_timesteps):
            # Get the current image
            img = images[batch_idx, channel, timestep].numpy()
            
            # Normalize the image for better visualization
            # img_min = img.min()
            # img_max = img.max()
            # if img_max > img_min:
            #     img = (img - img_min) / (img_max - img_min)
            
            # Plot the image
            if diff:
                im = axes[channel, timestep].imshow(img, cmap='RdBu', vmin=-1, vmax=1)
            else:
                im = axes[channel, timestep].imshow(img, cmap=cmap, origin='lower')
            axes[channel, timestep].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[channel, timestep], fraction=0.046, pad=0.04)
            
            # Add titles
            if channel == 0:
                extra = ' (predicted)' if (timestep-3) == 0 else ''
                axes[channel, timestep].set_title(f'frame {timestep-3}' + extra)
            if timestep == 0:
                axes[channel, timestep].text(-10, 32, channel_names[channel], 
                                          rotation=0, ha='right', va='center')

    return fig
