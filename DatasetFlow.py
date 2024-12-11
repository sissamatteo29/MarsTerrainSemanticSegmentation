import tensorflow as tf
import numpy as np
import AugmentationHelper

NUM_CLASSES = 5
BATCH_SIZE = 32


def configure(num_classes=None, batch_size=None):
    """
    Configure the parameters for this module if different from the default ones
    """
    global NUM_CLASSES, BATCH_SIZE
    if num_classes is not None:
        NUM_CLASSES = num_classes
    if batch_size is not None:
        BATCH_SIZE = batch_size



def one_hot_encoding(image, mask):
    """
    Turn the masks into one-hot-encoded with NUM_CLASSES depth
    """
    tf.ensure_shape(mask, (64, 128))
    mask = tf.one_hot(mask, depth=NUM_CLASSES, dtype=tf.uint8)
    return image, mask



def reshape_input(image, mask):
    """
    Reshape the input images by adding a third dimension (channel dimension)
    """
    reshaped_image = tf.expand_dims(image, axis=-1)
    return reshaped_image, mask



def revert_data_flow(dataset):
    """ 
    This function takes a dataset that is supposed to be preprocessed with the function data_flow and returns the dataset
    with the reversed functions applied
    """

    # Reverse function to map on the dataset
    def reverse_pre_processing(image, mask):
        image = tf.squeeze(image, axis=-1)
        image = tf.cast(image, tf.uint8)

        mask = tf.argmax(mask, axis=-1)
        mask = tf.cast(mask, tf.uint8)

        return image, mask    

    # Unbatch the dataset
    dataset = dataset.unbatch()

    # Map the reverse function
    dataset = dataset.map(reverse_pre_processing)

    return dataset
    


def data_flow(dataset, augment=True):
    """
    Define the entire dataflow to be applied on the dataset. For consistency, this function can be applied to both the 
    training and validation set, tuning the augment parameter correctly
    """

    is_validation = "validation"

    if augment:
        is_validation = "training"
        print()
        print("FUNCTION DATA FLOW - Starting augmentation")
        aug_geometric_1 = dataset.map(AugmentationHelper.map_geometric_transform, num_parallel_calls=tf.data.AUTOTUNE)
        aug_geometric_2 = dataset.map(AugmentationHelper.map_geometric_transform, num_parallel_calls=tf.data.AUTOTUNE)
        aug_intensity = dataset.map(AugmentationHelper.map_intensity_transform, num_parallel_calls=tf.data.AUTOTUNE)
        aug_total_1 = dataset.map(AugmentationHelper.map_total_transform, num_parallel_calls=tf.data.AUTOTUNE)
        aug_total_2 = dataset.map(AugmentationHelper.map_total_transform, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.concatenate(aug_geometric_1).concatenate(aug_geometric_2).concatenate(aug_intensity).concatenate(aug_total_1).concatenate(aug_total_2)

    # Reshape the images to have a third dimension (models expect this shape)
    dataset = dataset.map(reshape_input, num_parallel_calls=tf.data.AUTOTUNE)

    # Modify masks to be one hot encoded
    dataset = dataset.map(one_hot_encoding, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the result
    dataset = dataset.shuffle(buffer_size=tf.data.experimental.cardinality(dataset))
    
    print(f"FUNCTION DATA FLOW - Produced the {is_validation} dataset with {tf.data.experimental.cardinality(dataset)} images")    
    
    # Batch the dataset
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"FUNCTION DATA FLOW - Produced the {is_validation} dataset with {tf.data.experimental.cardinality(dataset)} batches")    

    return dataset



########################
# DEBUGGING PURPOSES   #
########################
def print_dataset_shape(dataset):
    """
    Helper function to debug the internal shape of a dataset
    """
    print()
    print("FUNCTION PRINT DATASET SHAPE")
    for images, masks in dataset.take(1):
        print(f"Image(s) shape: {images.shape}")
        print(f"Mask(s) shape: {masks.shape}")




def revert_flow_and_plot(train_dataset):
    
    revert_dataset = revert_data_flow(train_dataset)

    sample_images = []
    sample_masks = []
    for image, mask in revert_dataset.take(10):
        sample_images.append(image.numpy())
        sample_masks.append(mask.numpy())

    sample_images = np.array(sample_images)
    sample_masks = np.array(sample_masks)

    # Plot with helper function
    AugmentationHelper.plot_images_and_masks(sample_images, sample_masks)