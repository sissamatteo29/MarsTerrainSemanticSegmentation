import numpy as np
import matplotlib as mpl
import albumentations as A
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Setting the seeds for reproducibility
seed = 42
np.random.seed(42)
tf.random.set_seed(42)

"""
All the functions are built in such a way to be compatible with the tensorflow Dataset API.
This is the reason why all functions are annotated with @tf.py_function(...)
To make this integration process possible, some data conversion steps are required between tensors and
numpy arrays (in and out of the functions).
"""


def apply_geometric_transform(image, mask):
    """
    Simple helper function that takes in a single image and the corresponding mask as tensors 
    and applies a series of geometric transformations on both. It returns the image and the mask as tensors (after transformations).
    """

    transformation = A.Compose(
        [
            A.Rotate(p=0.9),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.9),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.ElasticTransform(alpha=10, sigma=10, interpolation=cv2.INTER_NEAREST, p=0.6),   # alpha for distortion level, sigma for smoothness
        ]
    )
    
    # The application of the transformation returns a dictionary with two keys
    # "image" : modified image
    # "mask" : modified mask
    result_dict = transformation(image=image.numpy(), mask=mask.numpy())
    result_image = result_dict["image"]
    result_mask = result_dict["mask"]
    
    # Turn results back into tensors
    result_image = tf.convert_to_tensor(result_image, dtype=tf.uint8)
    result_mask = tf.convert_to_tensor(result_mask, dtype=tf.uint8)

    result_image = tf.ensure_shape(result_image, (64, 128))
    result_mask = tf.ensure_shape(result_mask, (64, 128))


    return result_image, result_mask


def map_geometric_transform(image, mask):
    result_image, result_mask = tf.py_function(
        func=apply_geometric_transform,
        inp=[image, mask],
        Tout=[tf.uint8, tf.uint8]
    )
    # Ensure static shapes after tf.py_function
    result_image.set_shape((64, 128))
    result_mask.set_shape((64, 128))
    return result_image, result_mask





def apply_intensity_transform(image, mask):
    """
    Simple helper function that takes in a single image and the corresponding mask as tensors 
    and applies a series of colour/intensity transformations on both. It returns the image and the mask as tensors (after transformations).

    Notice: for these types of transformations, the Albumentations library expects the input images in two possible formats:
    - floating point values in the range [0.0, 1.0]
    - integers in the range [0, 255]
    It is therefore necessary to cast the images to hold integer pixel values before passing them in.
    """
    transformation = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.9),
            A.CLAHE(p=0.7),
            A.Sharpen(p=0.5),
            A.MotionBlur(p=0.1),
        ]
    )

    # The application of the transformation returns a dictionary with two keys
    # "image" : modified image
    # "mask" : modified mask
    result_dict = transformation(image=image.numpy(), mask=mask.numpy())
    result_image = result_dict["image"]
    result_mask = result_dict["mask"]

    # Turn results back into tensors
    result_image = tf.convert_to_tensor(result_image, dtype=tf.uint8)
    result_mask = tf.convert_to_tensor(result_mask, dtype=tf.uint8)

    result_image = tf.ensure_shape(result_image, (64, 128))
    result_mask = tf.ensure_shape(result_mask, (64, 128))
    
    return result_image, result_mask


def map_intensity_transform(image, mask):
    result_image, result_mask = tf.py_function(
        func=apply_intensity_transform,
        inp=[image, mask],
        Tout=[tf.uint8, tf.uint8]
    )
    # Ensure static shapes after tf.py_function
    result_image.set_shape((64, 128))
    result_mask.set_shape((64, 128))
    return result_image, result_mask

    




def apply_total_transform(image, mask):
    """
    Simple helper function that takes in a single image and the corresponding mask as tensors 
    and applies a combination of geometric and pixel intensity transformations. 
    It returns the image and the mask as tensors (after transformations).

    Notice: since many different types of augmentation steps are listed, the parameters 
    are tuned to mitigate and control the effects of each component of the pipeline (eccessive 
    augmentation is not advisable).
    """
    transformation = A.Compose(
        [
            A.SomeOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.CLAHE(p=0.5, clip_limit=1.1),
                    A.Sharpen(p=0.3, alpha=(0.1,0.25), lightness=(0.9,1)),
                    A.MotionBlur(p=0.1, blur_limit=5)
                ],
                n=1,
                replace=False
            ),
            A.SomeOf(
                [
                    A.Rotate(p=1),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1),
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.ElasticTransform(alpha=5, sigma=5, interpolation=cv2.INTER_NEAREST, p=0.3),   # alpha for distortion level, sigma for smoothness
                ],
                n=2,
                replace=False 
            ), 
        ]
    )
    
    # The application of the transformation returns a dictionary with two keys
    # "image" : modified image
    # "mask" : modified mask
    result_dict = transformation(image=image.numpy(), mask=mask.numpy())
    result_image = result_dict["image"]
    result_mask = result_dict["mask"]

    # Turn results back into tensors
    result_image = tf.convert_to_tensor(result_image, dtype=tf.uint8)
    result_mask = tf.convert_to_tensor(result_mask, dtype=tf.uint8)

    result_image = tf.ensure_shape(result_image, (64, 128))
    result_mask = tf.ensure_shape(result_mask, (64, 128))
    
    return result_image, result_mask


def map_total_transform(image, mask):
    result_image, result_mask = tf.py_function(
        func=apply_total_transform,
        inp=[image, mask],
        Tout=[tf.uint8, tf.uint8]
    )
    # Ensure static shapes after tf.py_function
    result_image.set_shape((64, 128))
    result_mask.set_shape((64, 128))
    return result_image, result_mask





def resize_crop(image, mask):
    """
    Resize a cropped region of class 4 from the original image to the target size.
    
    Args:
    - image (numpy array): The original image (height x width).
    - class_mask (numpy array): Segmentation mask with class 4 region.
    - target_size (tuple): Desired size after resizing (height, width).

    Returns:
    - region_resized (numpy array), mask_resized (numpy array)
    - or False, False if no class 4 region is found.
    """
    
    # Step 1: Find the bounding box of the class 4 region (yellow class)
    y_indices, x_indices = np.where(mask == 4)  # Locate all pixels of class 4
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        # If there are no class 4 pixels, return False
        return False, False

    # Step 2: Get the bounding box of the class 4 region
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Crop the image and the mask to the bounding box of class 4
    class_4_region = image[y_min:y_max+1, x_min:x_max+1]
    class_4_mask = mask[y_min:y_max+1, x_min:x_max+1]

    
    # Step 3: Resize the cropped class 4 region to the target size (64x128)
    target_size=(128, 64)
    region_resized = cv2.resize(class_4_region, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(class_4_mask, target_size, interpolation=cv2.INTER_LINEAR)
    
    return region_resized, mask_resized





def dataset_to_list(dataset):
    """
    Helper function that takes as input a tf.Dataset of images and masks and returns two lists only containing
    the numpy arrays of images and masks (unpacks both the Dataset layer and the Tensor layer).
    """
    image_list = []
    mask_list = []

    for img, mask in dataset.as_numpy_iterator():
        image_list.append(img)
        mask_list.append(mask)

    return image_list, mask_list




def plot_images_and_masks(images, masks, titles=None, figsize=(15, 22), cmap='gray'):

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
    colors = ["black", "red", "blue", "green", "yellow"]
    bounds = [0, 1, 2, 3, 4, 5]  # Bounds correspond to the intervals between numbers.
    norm = mpl.colors.BoundaryNorm(bounds, len(colors))

    # Create the colormap
    mask_colormap = mpl.colors.ListedColormap(colors)

    figure = plt.figure(figsize=figsize)

    if images.ndim == 2:    # Single image
        axes_array = figure.subplots(1, 2)
        axes_array = axes_array.flatten()
        axes_array[0].imshow(images, cmap=cmap, vmin=0, vmax=255)
        axes_array[0].set_title(titles[0]) if titles is not None else None
        axes_array[0].axis('off')
        # Plot original mask
        axes_array[1].imshow(masks, cmap=mask_colormap, norm=norm, vmin=0, vmax=4) 
        axes_array[1].axis('off')
        axes_array[1].set_title(titles[0]) if titles is not None else None
    else: 
        image_count = len(images)
        axes_array = figure.subplots(image_count, 2)
        axes_array = axes_array.flatten()
        for img_index, axes_index in zip(range(0, image_count), range(0, image_count * 2, 2)):
            axes_array[axes_index].imshow(images[img_index], cmap=cmap, vmin=0, vmax=255)
            axes_array[axes_index].set_title(titles[img_index]) if titles is not None else None
            axes_array[axes_index].axis('off')
            # Plot original mask
            axes_array[axes_index + 1].imshow(masks[img_index], cmap=mask_colormap, vmin=0, vmax=4) 
            axes_array[axes_index + 1].axis('off')
            axes_array[axes_index + 1].set_title(titles[img_index]) if titles is not None else None

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
        
