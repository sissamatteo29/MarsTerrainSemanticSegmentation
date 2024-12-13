import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
import sklearn
import cv2


def load_data(file_path, remove_bg_percentage=0, remove_outliers=False):
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

    if (remove_bg_percentage > 0):
        train_images, train_masks = remove_background_images(train_images, train_masks, remove_bg_percentage)

    if remove_outliers:
        train_images, train_masks = remove_outliers(train_images, train_masks)

    return train_images, train_masks, test_set



def remove_background_images(images, masks, background_percentage):
    """
    This function removes the images and masks that have a `background_percentage`
    of background class pixels.
    """
    print()
    print("FUNCTION REMOVE BACKGROUND IMAGES")
    print("---------------------------------------------------")

    # Find all indices of masks with a high proportion of background pixels
    black_mask_indices = []
    for i, mask in enumerate(masks):
        black_prop = np.sum(mask == 0) / mask.size # Proportion of black pixels
        if black_prop >= background_percentage: # and (np.any(mask != 4)): # Exclude class 4 masks
            black_mask_indices.append(i)
            
    # delete images and masks with high proportion of background pixels
    images_clean = np.delete(images, black_mask_indices, axis=0)
    masks_clean = np.delete(masks, black_mask_indices, axis=0)


    print("Shape of the images and masks after backgroun removal:")
    print(images_clean.shape, masks_clean.shape)

    print(f"Number of images removed: {len(black_mask_indices)}")
    print()
 
    return images_clean, masks_clean


def remove_outliers(images, masks):
    """
    This function removes the images and masks that DBSCAN identifies as outliers.
    """
    print()
    print("FUNCTION REMOVE OUTLIERS")
    print("---------------------------------------------------")

    # Apply DBSCAN
    epsilon = 400  # Radius of the neighborhood
    min_samples = 5  # Minimum points required to form a cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    features_pca_clean = pca_distribution(images)
    clusters = dbscan.fit_predict(features_pca_clean)

    # Get indices of points outside of cluster 0
    non_cluster_0_indices = np.where(clusters != 0)[0]

    # keep only the clean samples from x and y
    images_clean = np.delete(images, non_cluster_0_indices, axis=0)
    masks_clean = np.delete(masks, non_cluster_0_indices, axis=0)


    print("Shape of the images and masks after outlier removal:")
    print(images_clean.shape, masks_clean.shape)

    print(f"Number of images removed: {len(non_cluster_0_indices)}")
    print()
 
    return images_clean, masks_clean




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

    if isinstance(masks, list):
        masks = np.array(masks)


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





def compute_class_weights_including_0(train_dataset): 
    """
    Helper function that takes a complete train dataset (tf.Dataset) containing image-mask pairs (masks not one_hot_encoded)
    and produces the class weights to use for the loss functions based on the distribution of classes over the pixel
    values.

    Notice two things:
    - The calculation of the weights is performed also considering the pixels of the background (they count in the total number of pixels)
    - The weight for class 0 is set to 0 before returning the list of values
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

    # Set weight for class 0 to zero
    class_weights[0] = 0

    

    return class_weights




def compute_class_weights_excluding_0(train_dataset):
    """
    Compute class weights excluding class 0 pixels entirely from the calculation. Here the pixels with value 0 are
    not even included in the total count of pixels.

    Args:
    - train_dataset: Iterable of 2D masks (numpy arrays), where each mask contains pixel class indices.
    - num_classes: Total number of classes in the dataset (including class 0).

    Returns:
    - class_weights: List of normalized weights for each class, with the weight for class 0 set to 0.
    """

    # Generate a numpy array of masks
    train_masks_list = []
    for _, mask in train_dataset.as_numpy_iterator():
        train_masks_list.append(mask)

    train_masks_array = np.array(train_masks_list)

    # Initialize a list to count pixels for each class (excluding class 0)
    class_pixel_counts = np.zeros(5, dtype=np.int64)

    # Iterate over the dataset and count pixels for each class
    total_non_background_pixels = 0
    for mask in train_masks_array:
        unique_classes, counts = np.unique(mask, return_counts=True)
        for class_n, count in zip(unique_classes, counts):
            if class_n != 0:  # Ignore class 0
                class_pixel_counts[class_n] += count
                total_non_background_pixels += count

    # Compute weights for each class (inverse frequency), ignoring class 0
    class_weights = np.zeros(5, dtype=np.float32)  # Start with all zeros
    for class_n in range(1, 5):  # Start from class 1
        expected_samples_per_class_n = total_non_background_pixels / 4
        class_weights[class_n] = expected_samples_per_class_n / class_pixel_counts[class_n]

    return class_weights




def compute_class_weights(train_dataset, include_0=False, normalise=False):
    """
    Function wrapping together the functionalities of the previous two functions
    """
    class_weights = []
    if include_0:
        class_weights = compute_class_weights_including_0(train_dataset)
    else:
        class_weights = compute_class_weights_excluding_0(train_dataset)

    if normalise:
        class_weights = class_weights / np.max(class_weights[1:]) 
    
    # Print result (as a dictionary)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print()
    print("FUNCTION COMPUTE CLASS WEIGHTS")
    print("Class Weights:", class_weights_dict)

    return class_weights





def resize_crop(image, mask, class_label=4, min_class_coverage=0.05, min_patch_size=20, target_size=(128, 64), interpolation=cv2.INTER_LINEAR):
    """
    Extract and resize a region of the image containing the target class with sufficient coverage.

    Args:
    - image (numpy array): Original image (height x width x channels).
    - mask (numpy array): Segmentation mask (height x width).
    - class_label (int): Class to focus on.
    - min_class_coverage (float): Minimum percentage of region covered by the target class.
    - min_patch_size (int): Minimum size of the bounding box for extraction.
    - target_size (tuple): Target size to resize the region.
    - interpolation: Interpolation method for resizing (default cv2.INTER_LINEAR).

    Returns:
    - resized_image (numpy array), resized_mask (numpy array): Resized image and mask with target class emphasis.
    - None, None: If no suitable region is found.

    Notice: the interpolation method passed as a parameter will only be applied to the images, not to the masks
    which contain discrete values. Nearest Neighbour interpolation is used for masks always.
    Moreover, the target_size is injected in this order because it is the way cv2 expects it
    """
    # Step 1: Locate all pixels belonging to the target class
    y_indices, x_indices = np.where(mask == class_label)

    if len(y_indices) == 0 or len(x_indices) == 0:
        # No pixels of the target class
        return None, None

    # Step 2: Compute bounding box around target class pixels
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    # Step 3: Ensure the bounding box meets minimum patch size
    bbox_height, bbox_width = y_max - y_min + 1, x_max - x_min + 1
    if bbox_height < min_patch_size or bbox_width < min_patch_size:
        return None, None

    # Step 4: Extract the region and check class coverage
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    class_coverage = np.sum(cropped_mask == class_label) / (bbox_height * bbox_width)

    if class_coverage < min_class_coverage:
        # Reject patches with insufficient coverage of the target class
        return None, None

    # Step 5: Resize the extracted region to the target size
    resized_image = cv2.resize(cropped_image, target_size, interpolation=interpolation)
    resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return resized_image, resized_mask




def resize_crop_total(images, masks):
    """
    Function to extract from a set of images and a set of masks patches of class 4 and resize them to original input size
    Specifically designed for class 4 balancing procedure
    """
    result_images = []
    result_masks = []

    for image, mask in zip(images, masks):
        cropped_image, cropped_mask = resize_crop(image, mask)
        if cropped_image is not None and cropped_mask is not None:
            result_images.append(cropped_image)
            result_masks.append(cropped_mask)

    
    result_images = np.array(result_images)
    result_masks = np.array(result_masks)

    print()
    print("FUNCTION RESIZE AND CROP TOTAL")
    print(f"Produced a total number of images equal to {result_images.shape[0]}")

    return result_images, result_masks





def extract_relevant_patches_single_image(image, mask, class_label=4, patch_size=32, stride=10, min_class_coverage=0.2):
    """
    Extract relevant patches from an image and mask, focusing on regions with a significant presence of a specific class.

    Args:
    - image: Input image (H, W).
    - mask: Corresponding mask (H, W).
    - class_label: Class of interest for patch extraction.
    - patch_size: Size of the square patch.
    - stride: Stride for sliding window.
    - min_class_coverage: Minimum percentage of the patch area covered by the class (default 10%).

    Returns:
    - patches: List of extracted image patches.
    - patch_masks: List of extracted mask patches.
    """
    patches = []
    patch_masks = []
    height, width = mask.shape[:2]
    
    # Sliding window to extract patches
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Extract patch
            img_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            
            # Compute the percentage of the patch covered by the target class
            class_coverage = np.sum(mask_patch == class_label) / (patch_size * patch_size)
            
            # Include patch if it meets the coverage threshold
            if class_coverage >= min_class_coverage:
                patches.append(img_patch)
                patch_masks.append(mask_patch)
    
    return patches, patch_masks





def extract_relevant_patches_dataset(images, masks, patch_size=32, stride=10, min_class_coverage=0.2):
    """
    Extract relevant patches from a dataset of images and masks.

    Args:
    - images: List or array of images (each image is H x W x C).
    - masks: List or array of corresponding masks (each mask is H x W).
    - class_label: Class of interest for patch extraction (default is 4).
    - patch_size: Size of the square patch (default is 32).
    - stride: Stride for sliding window (default is 10).
    - min_class_coverage: Minimum percentage of the patch area covered by the class (default is 0.2).

    Returns:
    - all_patches: List of all image patches across the dataset.
    - all_patch_masks: List of all mask patches across the dataset.
    """
    all_patches = []
    all_patch_masks = []

    for image, mask in zip(images, masks):
        # Extract relevant patches for each image-mask pair
        patches, patch_masks = extract_relevant_patches_single_image(
            image=image,
            mask=mask,
            patch_size=patch_size,
            stride=stride,
            min_class_coverage=min_class_coverage,
        )
        
        # Append the patches and masks to the full list
        all_patches.extend(patches)
        all_patch_masks.extend(patch_masks)


    img_shape = 1
    if len(all_patches) > 0:
        img_shape = all_patches[0].shape
    else:
        img_shape = 0

    print()
    print("FUNCTION EXTRACT RELEVANT PATCHES DATASET")
    print(f"Found {len(all_patches)} patches with size {img_shape}")

    return all_patches, all_patch_masks





def relevant_patches_resized_total(patches_images, patches_masks, 
                                    target_size=(128, 64), interpolation=cv2.INTER_LINEAR):
    """
    This function takes a set of patches (images - masks pairs) and resizes them to the target size

    Notice: the interpolation method passed as a parameter will only be applied to the images, not to the masks
    which contain discrete values. Nearest Neighbour interpolation is used for masks always
    """

    resized_patches = []
    resized_patches_masks = []
    for patch, patch_mask in zip(patches_images, patches_masks):
        resized_image = cv2.resize(patch, target_size, interpolation=interpolation)
        resized_mask = cv2.resize(patch_mask, target_size, interpolation=cv2.INTER_NEAREST)    # Pay attention that for discrete labels NN has to be used
        resized_patches.append(resized_image)
        resized_patches_masks.append(resized_mask)

    return resized_patches, resized_patches_masks





def tile_patch_to_target_size(image_patch, mask_patch, target_size=(64, 128)):
    """
    Tile smaller image and mask patches to match the target size.

    Args:
    - image_patch: Input image patch (H, W).
    - mask_patch: Input mask patch (H, W).
    - target_size: Tuple specifying the target size (height, width).

    Returns:
    - tiled_image: Image tiled to the target size.
    - tiled_mask: Mask tiled to the target size.
    """
    patch_height, patch_width = image_patch.shape[:2]
    target_height, target_width = target_size

    # Calculate the number of repetitions needed
    vertical_repeats = -(-target_height // patch_height)  # Ceiling division
    horizontal_repeats = -(-target_width // patch_width)  # Ceiling division

    # Tile the image patch
    tiled_image = np.tile(
        image_patch, 
        (vertical_repeats, horizontal_repeats)  # For image (H, W)
    )
    
    # Tile the mask patch
    tiled_mask = np.tile(
        mask_patch, 
        (vertical_repeats, horizontal_repeats)  # For mask (H, W)
    )

    # Crop both to the exact target size
    tiled_image_cropped = tiled_image[:target_height, :target_width]
    tiled_mask_cropped = tiled_mask[:target_height, :target_width]

    return tiled_image_cropped, tiled_mask_cropped





def produce_tile_patch_images_total(patches_images, patches_masks, target_size=(64,128)):
    """
    Helper function that takes a set of image and mask pairs and for each patch it produces a 
    tiled version that is compliant to the original image size (64, 128)
    """

    tiled_images = []
    tiled_masks = []
    for patch, patch_mask in zip(patches_images, patches_masks):
        # Tile the patches
        tiled_image, tiled_mask = tile_patch_to_target_size(patch, patch_mask, target_size=target_size)
        tiled_images.append(tiled_image)
        tiled_masks.append(tiled_mask)

    return tiled_images, tiled_masks





def produce_tile_patch_images_mixed(patches_images, patches_masks, target_size=(64, 128), patch_size=32, target_num_images=100):
    """
    Generate tiled images by mixing patches from different input images.

    Args:
    - images: List or array of input images.
    - masks: List or array of corresponding masks.
    - target_size: Desired output image size (height, width).
    - patch_size: Size of the square patch for extraction.
    - stride: Stride for sliding window in patch extraction.
    - min_class_coverage: Minimum percentage of the patch area covered by the target class.
    - target_num_images: Number of tiled images to produce.

    Returns:
    - tiled_images: List of tiled images of size `target_size`.
    - tiled_masks: List of tiled masks of size `target_size`.
    """

    tiled_images = []
    tiled_masks = []

    # Compute the number of patches needed to fill the target size
    target_height, target_width = target_size
    patches_per_row = -(-target_width // patch_size)  # Ceiling division
    patches_per_col = -(-target_height // patch_size)  # Ceiling division
    patches_needed = patches_per_row * patches_per_col

    print(f"Tiling target size: {target_size}, requires {patches_needed} patches per image.")

    # Randomly sample patches and tile them together
    for _ in range(target_num_images):
        # Randomly select patches for this image
        selected_indices = np.random.choice(len(patches_images), patches_needed, replace=True)
        selected_patches = [patches_images[i] for i in selected_indices]
        selected_patch_masks = [patches_masks[i] for i in selected_indices]

        # Create a blank canvas for the tiled image and mask
        tiled_image = np.zeros((target_height, target_width), dtype=patches_images[0].dtype)
        tiled_mask = np.zeros((target_height, target_width), dtype=patches_masks[0].dtype)

        # Tile the patches
        for row in range(patches_per_col):
            for col in range(patches_per_row):
                patch_idx = row * patches_per_row + col
                patch = selected_patches[patch_idx]
                patch_mask = selected_patch_masks[patch_idx]

                # Determine the placement bounds
                y_start = row * patch_size
                y_end = min(y_start + patch_size, target_height)
                x_start = col * patch_size
                x_end = min(x_start + patch_size, target_width)

                # Determine the crop bounds for the patch
                patch_y_end = y_end - y_start
                patch_x_end = x_end - x_start

                # Place the patch on the canvas
                tiled_image[y_start:y_end, x_start:x_end] = patch[:patch_y_end, :patch_x_end]
                tiled_mask[y_start:y_end, x_start:x_end] = patch_mask[:patch_y_end, :patch_x_end]

        # Append the tiled image and mask to the list
        tiled_images.append(tiled_image)
        tiled_masks.append(tiled_mask)

    print(f"Finished procedure, returning {len(tiled_images)} images")

    return tiled_images, tiled_masks


    



def extraction_class_4_samples(images, masks,
                                patch_size=32, stride=10, min_class_coverage=0.2,
                                interpolation=cv2.INTER_LINEAR, target_num_images_mixed_tiling=100):
    """
    Wrapping function that takes in all necessary parameters and applies all functions defined above to extract
    as many samples as possible from class 4 (with a certain variability).
    """

    # First extract all relevant patches from the dataset for class 4
    all_patches, all_patches_masks = extract_relevant_patches_dataset(images, masks, patch_size=patch_size, stride=stride, min_class_coverage=min_class_coverage)

    # Resulting lists
    total_images = []
    total_masks = []

    resize_crop_images, resize_crop_masks = resize_crop_total(images, masks)


    patches_resized_images, patches_resized_masks = relevant_patches_resized_total(all_patches, all_patches_masks, 
                                                                                    interpolation=interpolation)
    

    tiled_patches_images, tiled_patches_masks = produce_tile_patch_images_total(all_patches, all_patches_masks)
                                                                                

    tiled_patches_images_mixed, tiled_patches_masks_mixed = produce_tile_patch_images_mixed(all_patches, all_patches_masks, 
                                                                                            patch_size=patch_size,
                                                                                            target_num_images=target_num_images_mixed_tiling)
    

    total_images.extend(resize_crop_images)
    total_masks.extend(resize_crop_masks) 

    total_images.extend(patches_resized_images)
    total_masks.extend(patches_resized_masks)

    total_images.extend(tiled_patches_images)
    total_masks.extend(tiled_patches_masks)

    total_images.extend(tiled_patches_images_mixed)
    total_masks.extend(tiled_patches_masks_mixed)

    print()
    print("FUNCTION EXTRACTION CLASS 4 SAMPLES")
    print(f"The final size of the extracted images is {len(total_images)}")

    return total_images, total_masks
    







