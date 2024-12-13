import numpy as np

import albumentations as A
import cv2
import tensorflow as tf


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




def apply_geometric_transform_light(image, mask):
    """
    Simple helper function that takes in a single image and the corresponding mask as tensors 
    and applies a series of geometric transformations on both. It returns the image and the mask as tensors (after transformations).

    This has been specifically designed to handle augmentation for class 4 (light augmentation).
    """

    transformation = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
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


def map_geometric_transform_light(image, mask):
    result_image, result_mask = tf.py_function(
        func=apply_geometric_transform_light,
        inp=[image, mask],
        Tout=[tf.uint8, tf.uint8]
    )
    # Ensure static shapes after tf.py_function
    result_image.set_shape((64, 128))
    result_mask.set_shape((64, 128))
    return result_image, result_mask




def apply_total_transform_light(image, mask):
    """
    Simple helper function that takes in a single image and the corresponding mask as tensors 
    and applies a combination of geometric and pixel intensity transformations. 
    It returns the image and the mask as tensors (after transformations).

    Notice: specifically designed to augment class 4 (very light pipeline)
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
                    A.Rotate(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
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


def map_total_transform_light(image, mask):
    result_image, result_mask = tf.py_function(
        func=apply_total_transform_light,
        inp=[image, mask],
        Tout=[tf.uint8, tf.uint8]
    )
    # Ensure static shapes after tf.py_function
    result_image.set_shape((64, 128))
    result_mask.set_shape((64, 128))
    return result_image, result_mask
