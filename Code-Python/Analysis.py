import tensorflow as tf
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns


NUM_CLASSES = 5
METRICS_TO_PLOT = ['loss', 'mean_iou']

def configure(num_classes=None, metrics_to_plot=None):
    """
    Configure the parameters for this module if different from the default ones
    """
    global NUM_CLASSES, METRICS_TO_PLOT

    if num_classes is not None:
        NUM_CLASSES = num_classes

    if metrics_to_plot is not None:
        METRICS_TO_PLOT = metrics_to_plot





def dataset_to_array(dataset):
    """
    Helper function that takes as input a tf.Dataset of images and masks and returns two lists only containing
    the numpy arrays of images and masks (unpacks both the Dataset layer and the Tensor layer).
    """
    image_list = []
    mask_list = []

    for img, mask in dataset.as_numpy_iterator():
        image_list.append(img)
        mask_list.append(mask)

    image_list = np.array(image_list)
    mask_list = np.array(mask_list)

    return image_list, mask_list




def extract_random(dataset, images_to_extract=10):
    """
    Helper function that takes in a dataset of images-mask pairs and extracts some random images and masks.
    It returns them all in two separate numpy arrays.
    """

    images_in_dataset = tf.data.experimental.cardinality(dataset)

    random_indices = np.random.randint(0, images_in_dataset, size=images_to_extract)

    image_list, mask_list = dataset_to_array(dataset)

    random_images = []
    random_masks = []
    for index in random_indices:
        random_images.append(image_list[index])
        random_masks.append(mask_list[index])

    random_images = np.array(random_images)
    random_masks = np.array(random_masks)

    return random_images, random_masks






def plot_confusion_matrix(model, dataset_to_predict, num_classes=NUM_CLASSES):
    """
    Helper function that plots the confusion matrix for a model computing the predictions over the provided dataset.
    """
    # Initialize arrays for true and predicted labels
    all_true_labels = []
    all_pred_labels = []

    # Iterate over the validation dataset
    for x_batch, y_true_batch in dataset_to_predict:
        # Get model predictions
        y_pred_batch = model.predict(x_batch)
        # Convert predictions and ground truth to integer labels
        y_pred_batch = tf.argmax(y_pred_batch, axis=-1).numpy().flatten()
        y_true_batch = tf.argmax(y_true_batch, axis=-1).numpy().flatten()
        # Append to overall lists
        all_true_labels.extend(y_true_batch)
        all_pred_labels.extend(y_pred_batch)

    # Compute the confusion matrix
    cm = sklearn.metrics.confusion_matrix(all_true_labels, all_pred_labels, labels=range(num_classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()




def plot_training_results(history, training_metrics=None):
    """
    Function to be used at the end of the training process to plot some metrics extracted from the history object
    (returned by model.fit). The training_metrics parameter can be used to specify the names of the metrics
    that the user wants to be plotted.
    """

    if training_metrics is None:
        training_metrics = METRICS_TO_PLOT

    validation_metrics = []
    for metric in training_metrics:
        validation_metrics.append("val_" + metric)

    for training_metric, validation_metric in zip(METRICS_TO_PLOT, validation_metrics):
        plt.figure(figsize=(18, 3))
        plt.plot(history[training_metric], label='Training', alpha=0.8, color='#ff7f0e', linewidth=2)
        plt.plot(history[validation_metric], label='Validation', alpha=0.9, color='#5a9aa5', linewidth=2)
        plt.title(training_metric)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()



