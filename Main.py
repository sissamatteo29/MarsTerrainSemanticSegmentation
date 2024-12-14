# Import libraries
from datetime import datetime
import tensorflow as tf
import keras
import numpy as np

# Kaggle configuration
# import sys
# sys.path.append("/kaggle/input/aug-pipeline") # sys.path.append("/kaggle/input/")

# Import custom modules
import Preprocessing
import DatasetFlow
import Model
import Losses
import Metrics
import Logger
import Analysis
import Submission
import AugmentationHelper

# Constants
NUM_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 30
EPOCHS = 1000
DATA_PATH = "/kaggle/input/mars-data/ds_no_aliens.npz" #"data/ds_no_aliens.npz"
CLASS4_PATH = "/kaggle/input/mars-data/class4_samples.npz" #"data/class4_samples.npz" 

# Set random seed for reproducibility
keras.utils.set_random_seed(13)

# Load data (already split into train set and test set)
train_images, train_masks, test_set = Preprocessing.load_data(
    DATA_PATH, remove_bg_percentage=0.95, remove_outliers=False
)


# Class 4 augmentation pipeline version 2
class4_images, class4_masks, _ = Preprocessing.load_data(CLASS4_PATH)
class4_images, class4_masks = Preprocessing.extraction_class_4_samples(class4_images, class4_masks, target_num_images_mixed_tiling=700)
# Class 4 augmentation pipeline version 3
#dataset_class4 = tf.data.Dataset.from_tensor_slices((class4_images, class4_masks))
#aug_class4 = dataset_class4.map(AugmentationHelper.map_geometric_transform_light, num_parallel_calls=tf.data.AUTOTUNE)
#dataset_class4.concatenate(aug_class4)
#class4_images, class4_masks = Analysis.dataset_to_array(dataset_class4)

# Merge class 4 augmentation with the original training data 
train_images, train_masks = DatasetFlow.merge_datasets(
    train_images, train_masks, class4_images, class4_masks
)

# Verify distribution of classes before balancing
Preprocessing.compute_class_distribution(train_masks, NUM_CLASSES)

# Version 3
# Build a dataset with the most representative samples of all classes until classes are balanced 
#train_images, train_masks = DatasetFlow.balance_dataset(train_images, train_masks)


# Split into train and validation, generating Dataset classes
train_dataset, validation_dataset = Preprocessing.split_train_data(
    train_images, train_masks
)

# Verify distribution of classes over the training dataset
Preprocessing.compute_class_distribution(train_masks, NUM_CLASSES)

# Compute weights based on distribution of classes
class_weights = Preprocessing.compute_class_weights(
    train_dataset, include_0=False, normalise=False
)
class_weights_normalised = Preprocessing.compute_class_weights(
    train_dataset, include_0=False, normalise=True
)

# Setup dataflow for the training and validation datasets (augmentation, shuffling, resizing, one_hot_encoding)
train_dataset = DatasetFlow.data_flow(train_dataset, augment=False)
validation_dataset = DatasetFlow.data_flow(validation_dataset, augment=False)

# Verify shapes
DatasetFlow.print_dataset_shape(train_dataset)
DatasetFlow.print_dataset_shape(validation_dataset)

# Build model
model = Model.dense_u_net()

# Compile the model
print("Compiling model...")

model.compile(
    loss=Losses.weighted_focal_loss(class_weights),
    optimizer=tf.keras.optimizers.AdamW(LEARNING_RATE),
    metrics=[
        Metrics.pixel_accuracy_exclude_class_0,
        Metrics.mean_iou,
        Metrics.background_iou,
        Metrics.class_1_iou,
        Metrics.class_2_iou,
        Metrics.class_3_iou,
        Metrics.class_4_iou,
    ],
)

print("Model compiled!")


# Setup callbacks
metrics_logger = Logger.MetricsLoggerCallback()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_iou", mode="max", patience=PATIENCE, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, mode="auto", min_lr=1e-5
)


history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr, metrics_logger],
    verbose=1,
).history

# Calculate and print the final validation accuracy
final_val_meanIoU = round(max(history["val_mean_iou"]) * 100, 2)
print(f"Final validation Mean Intersection Over Union: {final_val_meanIoU}%")


# Save the trained model to a file with the accuracy included in the filename
model_filename = "model.keras"
model.save(model_filename)


metrics_to_plot = [
    "loss",
    "mean_iou",
    "background_iou",
    "class_1_iou",
    "class_2_iou",
    "class_3_iou",
    "class_4_iou",
]

Analysis.plot_training_results(history, training_metrics=metrics_to_plot)
Analysis.plot_confusion_matrix(model, validation_dataset)

filename = f'submission_{datetime.now().strftime("%y%m%d_%H%M%S")}.csv'
Submission.prepare_submission(model_filename, test_set, filename)
