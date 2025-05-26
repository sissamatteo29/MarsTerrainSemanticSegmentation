import tensorflow as tf
import keras
import numpy as np
import random

# import sys
# sys.path.append("/kaggle/input/")

import Preprocessing
import Plot
import DatasetFlow
import Model
import Losses
import Metrics
import Logger
import Analysis
import Submission
import AugmentationHelper

keras.utils.set_random_seed(13)


# Load data (already split into train set and test set)
train_images, train_masks, test_set = Preprocessing.load_data("./Datasets/class4_samples.npz")

# resized_crop_images, resized_crop_masks = Preprocessing.resize_crop_total(train_images, train_masks)

# Plot.plot_images_and_masks(resized_crop_images[0:10], resized_crop_masks[0:10])

images, masks = Preprocessing.extraction_class_4_samples(train_images, train_masks)

Preprocessing.compute_class_distribution(masks, num_classes=5)

# Zip the two lists together
combined = list(zip(images, masks))

# Shuffle the combined list
random.shuffle(combined)

# Unzip back into two lists
shuffled_images, shuffled_masks = zip(*combined)

# Convert back to lists (optional, since zip returns tuples)
images = list(shuffled_images)
masks = list(shuffled_masks)


for index in range(0, 450, 3):
    Plot.plot_images_and_masks(images[index:index+3], masks[index:index+3])

    #patches_img, patches_masks = Preprocessing.relevant_patches_resized(img, mask)
    #Plot.plot_images_and_masks_variable_size(patches_img, patches_masks)
        
    # patches_img, patches_masks = Preprocessing.produce_tile_patch_images(img, mask)
    # Plot.plot_images_and_masks_variable_size(patches_img, patches_masks)
    


        




