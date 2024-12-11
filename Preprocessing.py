import numpy as np
import tensorflow as tf
import keras
import sklearn


def load_data(file_path):
    """
    Data loader function tailored for this specific dataset, it automatically unpacks the train and test sets
    and prints some statistics.
    """

    # Load the NpzFile
    data = np.load(file_path)

    # Inspect content of the NpzFile: contains two keys "training_set", "test_set"
    print(data.keys())

    # Extract training_set and test_set
    training_set = data.get("training_set")
    test_set = data.get("test_set")

    # Print the shapes of both sets
    print(f"The shape of the training set: {training_set.shape}")
    print(f"The shape of the test set: {test_set.shape}")

    # Split training set into input images and masks
    train_images = training_set[:, 0, :, :]
    train_masks = training_set[:, 1, :, :]

    print()
    print("FUNCTION LOAD DATA")
    # Analyse the values of the grayscale images in the training set
    print("---------------------------------------------------")
    print("TRAINING SET INPUT IMAGES ANALYSIS")
    print(f"The shape of the input images: {train_images.shape}")
    print(f"Max pixel value: {train_images.max()}")
    print(f"Min pixel value: {train_images.min()}")
    print(f"Data type to encode pixel values: {train_images.dtype}")

    # Analyse the values of the masks in the training set
    print("---------------------------------------------------")
    print("TRAINING SET OUTPUT MASKS ANALYSIS")
    print(f"The shape of the output masks: {train_masks.shape}")
    print(f"Max pixel value: {train_masks.max()}")
    print(f"Min pixel value: {train_masks.min()}")
    print(f"Data type to encode pixel values: {train_masks.dtype}")

    # Analyse the values of the grayscale images in the test set
    print("---------------------------------------------------")
    print("TEST SET ANALYSIS")
    print(f"Max pixel value: {test_set.max()}")
    print(f"Min pixel value: {test_set.min()}")
    print(f"Data type to encode pixel values: {test_set.dtype}")

    return train_images, train_masks, test_set



def split_train_data(train_images, train_masks, validation_ratio=0.1):

    """
    Splitter of the dataset into a train and validation sets, it returns the results already
    encapsulated into a tf.Dataset class for further processing.
    """

    initial_data = (train_images, train_masks)

    # Function offered by Keras utils to split the initial data and generate datasets automatically
    train_dataset, validation_dataset = keras.utils.split_dataset(
        initial_data, 
        left_size=1-validation_ratio, 
        right_size=validation_ratio,
        shuffle=True
        )

    # Extract some metrics
    TRAIN_SIZE = tf.data.experimental.cardinality(train_dataset)
    VALIDATION_SIZE = tf.data.experimental.cardinality(validation_dataset)

    print()
    print("FUNCTION SPLIT TRAIN DATA")
    print(f"Training dataset type specification: {train_dataset.element_spec}")
    print(f"Validation dataset type specification: {validation_dataset.element_spec}")

    print(f"Training dataset size {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Validation dataset size {tf.data.experimental.cardinality(validation_dataset)}")

    return train_dataset, validation_dataset




def compute_class_distribution(masks, num_classes):
    """
    Computes the percentage distribution of classes in the dataset.
    
    Args:
        masks: A 3D numpy array of shape (num_samples, height, width) containing class indices (notice: not one_hot_encoded)
        num_classes: Total number of classes (e.g., 5 for your dataset).
    
    Returns:
        A dictionary with class indices as keys and percentage distributions as values.
    """
    # Flatten the masks into a 1D array
    flattened_masks = masks.flatten()  # Shape: [num_samples * height * width]
    
    # Count occurrences of each class
    class_counts = np.bincount(flattened_masks, minlength=num_classes)  # Ensure all classes are included
    
    # Total number of pixels
    total_pixels = flattened_masks.size
    
    # Calculate percentage for each class
    class_percentages = (class_counts / total_pixels) * 100

    result_dictionary = {class_id: percentage for class_id, percentage in enumerate(class_percentages)}

    print()
    print("FUNCTION COMPUTE CLASS DISTRIBUTION")
    print("Class Distribution (%):")
    for class_id, percentage in result_dictionary.items():
        print(f"Class {class_id}: {percentage:.2f}%")
    
    # Return as a dictionary
    return result_dictionary




def compute_class_weights(train_dataset): 
    """
    Helper function that takes a complete train dataset (tf.Dataset) containing image-mask pairs (masks not one_hot_encoded)
    and produces the class weights to use for the loss functions based on the distribution of classes over the pixel
    values.
    """
    
    # Generate a numpy array of masks
    train_masks_list = []
    for _, mask in train_dataset.as_numpy_iterator():
        train_masks_list.append(mask)

    train_masks_array = np.array(train_masks_list)

    # Flatten masks to compute global class frequencies
    flat_masks = train_masks_array.flatten()

    # Compute class weights
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(flat_masks),
        y=flat_masks
    )

    print()
    print("FUNCTION COMPUTE CLASS WEIGHTS")

    # Convert to dictionary
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class Weights:", class_weights_dict)

    return class_weights




