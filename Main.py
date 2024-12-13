

import tensorflow as tf
import keras
import numpy as np


# import sys
# sys.path.append("/kaggle/input/")

import Preprocessing
import DatasetFlow
import Model
import Losses
import Metrics
import Logger
import Analysis
import Submission
# import AugmentationHelper

keras.utils.set_random_seed(13)

NUM_CLASSES = 5

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

PATIENCE = 30

EPOCHS = 1

# Load data (already split into train set and test set)
train_images, train_masks, test_set = Preprocessing.load_data("./ds_no_aliens.npz", remove_bg_percentage=0.95, remove_outliers=False)

# Split into train and validation, generating Dataset classes
train_dataset, validation_dataset = Preprocessing.split_train_data(train_images, train_masks)

# Verify distribution of classes over the training dataset
Preprocessing.compute_class_distribution(train_masks, 5)

# Compute weights based on distribution of classes
class_weights = Preprocessing.compute_class_weights(train_dataset, include_0=False, normalise=False)
class_weights_normalised = Preprocessing.compute_class_weights(train_dataset, include_0=False, normalise=True)

# Setup dataflow for the training and validation datasets (augmentation, shuffling, resizing, one_hot_encoding)
train_dataset = DatasetFlow.data_flow(train_dataset, augment=False)
validation_dataset = DatasetFlow.data_flow(validation_dataset, augment=False)

# Verify shapes
DatasetFlow.print_dataset_shape(train_dataset)
DatasetFlow.print_dataset_shape(validation_dataset)

# Build model
model = Model.u_net()

# Compile the model
print("Compiling model...")

model.compile(
    loss=Losses.weighted_categorical_crossentropy(class_weights),
    optimizer=tf.keras.optimizers.AdamW(LEARNING_RATE),
    metrics=[
        Metrics.pixel_accuracy_exclude_class_0, 
        Metrics.mean_iou, 
        Metrics.background_iou, 
        Metrics.class_1_iou, 
        Metrics.class_2_iou, 
        Metrics.class_3_iou, 
        Metrics.class_4_iou
    ]
)

print("Model compiled!")


# Setup callbacks

metrics_logger = Logger.MetricsLoggerCallback()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mean_iou',
    mode='max',
    patience=PATIENCE,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    mode="auto",
    min_lr=1e-5
)



history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr, metrics_logger],
    verbose=1
).history

# Calculate and print the final validation accuracy
final_val_meanIoU = round(max(history['val_mean_iou'])* 100, 2)
print(f'Final validation Mean Intersection Over Union: {final_val_meanIoU}%')


# Save the trained model to a file with the accuracy included in the filename
model_filename = 'model.keras'
model.save(model_filename)


metrics_to_plot = [
    'loss',
    'mean_iou',
    'background_iou',
    'class_1_iou',
    'class_2_iou',
    'class_3_iou',
    'class_4_iou',
]

Analysis.plot_training_results(history, training_metrics=metrics_to_plot)
Analysis.plot_confusion_matrix(model, validation_dataset)

Submission.prepare_submission(model_filename, test_set, "submission.csv")











