import tensorflow as tf
import numpy as np
import random
import AugmentationHelper
import Preprocessing
import DatasetFlow

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



def merge_datasets(orig_images, orig_masks, aug_images, aug_masks):
    """
    Merge two datasets into one
    """
    print("FUNCTION MERGE DATASETS - Merging datasets")
    print("--------------------------------------------")

    print(f"Shape before merging: {orig_images.shape}")
    # Concat original and augmented images and masks
    all_images = np.concatenate((orig_images, aug_images))
    all_masks = np.concatenate((orig_masks, aug_masks))

    # Suffle the dataset
    indices = np.arange(all_images.shape[0])
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_masks = all_masks[indices]

    print(f"Shape after merging: {all_images.shape}")

    return all_images, all_masks


def balance_dataset(orig_images, orig_masks):
    """
    Balance the dataset by adding samples from orig_images and orig_masks to class4_images and class4_masks
    until the difference between class proportions is below a certain threshold (excluding class 0).
    """
    print("FUNCTION BALANCE DATASET - Balancing dataset")
    print("--------------------------------------------")

    # Separate the samples per classes
    class_idx_dict = separate_samples_per_classes(orig_images, orig_masks)
    
    # Threshold for the difference between class proportions
    threshold_strict = 1e-5 # 0.00001
    threshold_broad = 1

    # Initializations
    merged_images = []
    merged_masks = []
    diff = {cls: threshold_strict for cls in range(1, NUM_CLASSES)}
    completed_classes = set()  # Set to track classes with no samples left

    while max(diff.values()) >= threshold_strict or len(merged_images) < 2000:  # to have at least 2000 samples
            
        # Find the class with the largest difference that has samples available
        valid_classes = [cls for cls in diff if cls not in completed_classes and len(class_idx_dict[cls]) >= 0]
        
        if not valid_classes:  # If no valid classes are left to process, break
            break

        if len(valid_classes) < 4 and max(diff.values()) > threshold_broad:
            # After class 4 samples are added, the difference starts to grow instead of decreasing, so we break the loop
            break

        
        max_diff_class = max(valid_classes, key=diff.get)

        # print(valid_classes)

        # print(f"Adding samples from class {max_diff_class} to balance the dataset")

        # Get and delete last sample from the class with the largest difference
        sample_idx = class_idx_dict[max_diff_class].pop()

        # Add the sample to the merged dataset
        merged_images.append(orig_images[sample_idx])
        merged_masks.append(orig_masks[sample_idx])

        # Update the class distribution
        proportions = Preprocessing.compute_class_distribution(merged_masks, NUM_CLASSES)
        diff = calculate_proportion_diff(proportions)

        # If this class has no more samples, mark it as completed
        if len(class_idx_dict[max_diff_class]) == 0:
            completed_classes.add(max_diff_class)

        # print("Current difference:", diff)
        # print("Current number of samples:", len(merged_images))
        # print()
        

    # Cast the lists to numpy arrays
    merged_images = np.array(merged_images)
    merged_masks = np.array(merged_masks)

    for key in class_idx_dict:
        print(f"Class {key} has {len(class_idx_dict[key])} samples left, but couldn't be added so classes stay balanced.")
    

    print("Completed classes:", completed_classes)

    print("Balancing complete.")
    print(f"Shape of the balanced dataset: {merged_images.shape}")
    
    return merged_images, merged_masks


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



# Helper functions
def calculate_proportion_diff(proportions):
    '''
    Function to calculate the difference between proportions of each class (excluding class 0)
    '''
    total_samples = sum([count for class_id, count in proportions.items() if class_id != 0])
    target_count = total_samples / (NUM_CLASSES - 1)
    
    # Calculate the difference from the target for each class (excluding class 0)
    diff = {cls: (target_count - proportions[cls]) for cls in range(1, NUM_CLASSES)}
    return diff






def separate_samples_per_classes(orig_images, orig_masks):
    """
    Separate the samples per classes
    """
    class_idx_dict = {0:[],1:[],2:[],3:[],4:[]}

    # Separate the original samples into classes
    for sample_idx in range(orig_images.shape[0]):
        # Get the mask
        mask = orig_masks[sample_idx]
        
        # Get predominant class in sample
        class_counts = np.bincount(mask.flatten(), minlength=5)
        class_counts[0] = 0  # Ignore class 0
        predominant_class = np.argmax(class_counts)

        # Add the sample to the corresponding class
        if predominant_class == 1:
            class_idx_dict[1].append(sample_idx)
        elif predominant_class == 2:
            class_idx_dict[2].append(sample_idx)
        elif predominant_class == 3:
            class_idx_dict[3].append(sample_idx)
        elif predominant_class == 4:
            class_idx_dict[4].append(sample_idx)
        else:
            class_idx_dict[0].append(sample_idx)     

    return class_idx_dict
