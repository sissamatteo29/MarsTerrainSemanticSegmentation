import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np



def plot_image_and_mask(image, mask, figsize=(15, 22), cmap='gray'):
    """
    Helper function to plot a single image and its corresponding mask.
    """

    # Define a colormap to simplify the visualisation of the classes on the masks
    # Background (0) -> black
    # Soil (1) -> red
    # Bedrock (2) -> blue
    # Sand (3) -> green
    # Big Rock (4) -> yellow
    mask_colormap = mpl.colors.ListedColormap(["black", "red", "blue", "green", "yellow"])

    figure = plt.figure(figsize=figsize)

    axes_array = figure.subplots(1, 2)
    axes_array = axes_array.flatten()
    axes_array[0].imshow(image, cmap=cmap, vmin=0, vmax=255)
    axes_array[0].axis('off')
    # Plot original mask
    axes_array[1].imshow(mask, cmap=mask_colormap, vmin=0, vmax=4) 
    axes_array[1].axis('off')

    plt.show()





def plot_images_and_masks(images, masks, figsize=(15, 22), cmap='gray'):

    """
    Helper function to plot images and masks pairs.

    Notice: when plotting grayscale or masks, it is necessary to specify vmin and vmax to the plotting function
    so that no rescaling is performed on the input data by the library
    
    """
    
    # Define a colormap to simplify the visualisation of the classes on the masks
    # Background (0) -> black
    # Soil (1) -> red
    # Bedrock (2) -> blue
    # Sand (3) -> green
    # Big Rock (4) -> yellow
    mask_colormap = mpl.colors.ListedColormap(["black", "red", "blue", "green", "yellow"])

    figure = plt.figure(figsize=figsize)

    if isinstance(images, list):
        images = np.array(images)
        masks = np.array(masks)

    if images.ndim == 2:    # Single image
        axes_array = figure.subplots(1, 2)
        axes_array = axes_array.flatten()
        axes_array[0].imshow(images, cmap=cmap, vmin=0, vmax=255)
        axes_array[0].axis('off')
        # Plot original mask
        axes_array[1].imshow(masks, cmap=mask_colormap, vmin=0, vmax=4) 
        axes_array[1].axis('off')
    else: 
        image_count = len(images)
        axes_array = figure.subplots(image_count, 2)
        axes_array = axes_array.flatten()
        for img_index, axes_index in zip(range(0, image_count), range(0, image_count * 2, 2)):
            axes_array[axes_index].imshow(images[img_index], cmap=cmap, vmin=0, vmax=255)
            axes_array[axes_index].axis('off')
            # Plot original mask
            axes_array[axes_index + 1].imshow(masks[img_index], cmap=mask_colormap, vmin=0, vmax=4) 
            axes_array[axes_index + 1].axis('off')

    plt.show()




def plot_images_and_masks_augmented(aug_imgs, aug_msks, original_imgs, original_msks, grid=(10, 4), figsize=(20, 30), cmap='gray'):

    """
    Helper function to verify the correct functioning of augmentation or image transformations.
    The aug_imgs and aug_msks parameters are the array-like structures containing the transformed images and masks, while 
    original_imgs and original_msks are the original images from the data set.
    By default, the function works well with 10 images-masks pairs (the defaults values are set for this purpose), 
    but it is possible to change the parameters to adapt it to a varying size of inputs.

    Notice: when plotting grayscale or masks, it is necessary to specify vmin and vmax to the plotting function
    so that no rescaling is performed on the input data by the library
    
    """
    
    # Define a colormap to simplify the visualisation of the classes on the masks
    # Background (0) -> black
    # Soil (1) -> red
    # Bedrock (2) -> blue
    # Sand (3) -> green
    # Big Rock (4) -> yellow
    mask_colormap = mpl.colors.ListedColormap(["black", "red", "blue", "green", "yellow"])

    image_count = len(original_imgs)
    
    figure = plt.figure(figsize=figsize)
    axes_array = figure.subplots(grid[0], grid[1])
    axes_array = axes_array.flatten()
    

    for img_index, ax_index  in zip(range(0, image_count), range(0, image_count * 4, 4)):
        # Plot original image
        axes_array[ax_index].imshow(original_imgs[img_index], cmap=cmap, vmin=0, vmax=255)
        axes_array[ax_index].axis('off')
        # Plot original mask
        axes_array[ax_index + 1].imshow(original_msks[img_index], cmap=mask_colormap, vmin=0, vmax=4) 
        axes_array[ax_index + 1].axis('off')
        # Plot augmented image
        axes_array[ax_index + 2].imshow(aug_imgs[img_index], cmap=cmap, vmin=0, vmax=255)
        axes_array[ax_index + 2].axis('off')
        # Plot "augmented" mask
        axes_array[ax_index + 3].imshow(aug_msks[img_index], cmap=mask_colormap, vmin=0, vmax=4) 
        axes_array[ax_index + 3].axis('off')

    plt.show()
