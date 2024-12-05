import numpy as np
import matplotlib as mpl
import albumentations as A
import cv2
import tensorflow as tf

"""
All the functions are built in such a way to be compatible with the tensorflow Dataset API.
This is the reason why all functions are annotated with @tf.py_function(...)
To make this integration process possible, some data conversion steps are required between tensors and
numpy arrays (in and out of the functions).
"""


@tf.py_function(Tout=[tf.uint8, tf.uint8])
def apply_geometric_transform(images, masks):
    """
    Simple helper function that takes in an array of images and the corresponding masks
    and applies a series of geometric transformations on both. It returns the two arrays of transformed
    images and masks as nparrays.
    """

    transformation = A.Compose(
        [
            A.Rotate(p=0.9),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.9),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.ElasticTransform(alpha=10, sigma=10, interpolation=cv2.INTER_NEAREST, p=0.6),   # alpha for distortion level, sigma for smoothness
            A.Perspective(),
            A.GridDistortion()
        ]
    )
    
    images=images.numpy()
    masks = masks.numpy()
    n_images = len(images)
    result_images = []
    result_masks = []
    
    for index in range(0, n_images):
        # The application of the transformation returns a dictionary with two keys
        # "image" : modified image
        # "mask" : modified mask
        result_dict = transformation(image=images[index], mask=masks[index])
        result_images.append(result_dict["image"])
        result_masks.append(result_dict["mask"])

    return np.array(result_images), np.array(result_masks)
        


@tf.py_function(Tout=[tf.uint8, tf.uint8])
def apply_intensity_transform(images, masks):
    """
    Simple helper function that takes in an array of images and the corresponding masks
    and applies a series of "intensity" transformations, which modify pixel values to achieve
    common "colour distortions". It returns the two arrays of transformed
    images and masks as nparrays.

    Notice: for these types of transformations, the library expects the input images in two possible formats:
    - floating point values in the range [0.0, 1.0]
    - integers in the range [0, 255]
    It is therefore necessary to cast the images to hold integer pixel values before passing them in.
    """
    transformation = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.9),
            A.CLAHE(p=0.7),
            A.Sharpen(p=0.5),
            A.MotionBlur(p=0.6),
        ]
    )

    images=images.numpy()
    masks = masks.numpy()
    n_images = len(images)
    result_images = []
    result_masks = []
    
    for index in range(0, n_images):
        # The application of the transformation returns a dictionary with two keys
        # "image" : modified image
        # "mask" : modified mask
        result_dict = transformation(image=images[index], mask=masks[index])
        result_images.append(result_dict["image"])
        result_masks.append(result_dict["mask"])

    return result_images, result_masks


    

@tf.py_function(Tout=[tf.uint8, tf.uint8])
def apply_total_transform(images, masks):
    """
    Simple helper function that takes in an array of images and the corresponding masks
    and applies a combination of geometric and pixel intensity transformations. 
    It returns the two arrays of transformed images and masks as nparrays.

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
                    A.MotionBlur(p=0.4, blur_limit=5)
                ],
                n=2,
                replace=False
            ),
            A.SomeOf(
                [
                    A.Rotate(p=1),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1),
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.ElasticTransform(alpha=5, sigma=5, interpolation=cv2.INTER_NEAREST, p=0.3),   # alpha for distortion level, sigma for smoothness
                    A.Perspective(p=0.3),
                    A.GridDistortion(p=0.3)
                ],
                n=3,
                replace=False 
            ), 
        ]
    )

    images=images.numpy()
    masks = masks.numpy()
    n_images = len(images)
    result_images = []
    result_masks = []
    
    for index in range(0, n_images):
        # The application of the transformation returns a dictionary with two keys
        # "image" : modified image
        # "mask" : modified mask
        result_dict = transformation(image=images[index], mask=masks[index])
        result_images.append(result_dict["image"])
        result_masks.append(result_dict["mask"])

    return result_images, result_masks






def plot_images_and_masks(aug_imgs, aug_msks, original_imgs, original_msks, grid=(10, 4), figsize=(20, 30), cmap='gray'):

    """
    Helper function to verify the correct functioning of augmentation or image transformations.
    The aug_imgs and aug_msks parameters are the arrays of transformed images and masks, while original_imgs and original_msks
    are the original images from the data set.
    By default, the function works well with 10 images, but it is possible to change the parameters to adapt it to 
    a varying size of inputs.

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
    
    figure = mpl.pyplot.figure(figsize=figsize)
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

