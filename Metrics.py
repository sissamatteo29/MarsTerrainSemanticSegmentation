
import tensorflow as tf

NUM_CLASSES = 5


def configure(num_classes=None):
    """
    Configure the parameters for this module if different from the default ones
    """
    global NUM_CLASSES
    if num_classes is not None:
        NUM_CLASSES = num_classes



def mean_iou(y_true, y_pred):
    """
    Takes two input tensors that are a batch of true masks and the corresponding batch of predicted masks.
    The true masks are one-hot encoded, the predicted ones are probability distributions over the classes
    The function computes the mean IoU for the batch of masks.

    Notice: this function completely skips the computation for class 0, as it is not relevant for the test set evaluation
    (the for loop starts from class 1).
    """
    # Convert predictions to discrete labels
    y_pred = tf.argmax(y_pred, axis=-1)  # Shape: (batch_size, height, width)
    y_true = tf.argmax(y_true, axis=-1)  # Shape: (batch_size, height, width)
    
    iou_scores = []
    for i in range(1, NUM_CLASSES):
        # Compute IoU for class `i`
        intersection = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
        union = tf.reduce_sum(tf.cast((y_true == i) | (y_pred == i), tf.float32))
        iou = tf.cond(union > 0, lambda: intersection / union, lambda: tf.constant(0.0))
        iou_scores.append(iou)

    # Return mean IoU over all classes
    return tf.reduce_mean(iou_scores)



def pixel_accuracy_exclude_class_0(y_true, y_pred):
    """
    Compute pixel accuracy excluding class 0 for semantic segmentation.
    Args:
        y_true: Ground truth one-hot encoded labels, shape (batch_size, height, width, num_classes).
        y_pred: Predicted probabilities, shape (batch_size, height, width, num_classes).
    Returns:
        Pixel accuracy (excluding class 0) as a scalar tensor.
    """
    # Convert predictions to discrete labels
    y_pred = tf.argmax(y_pred, axis=-1)  # Shape: (batch_size, height, width)
    y_true = tf.argmax(y_true, axis=-1)  # Shape: (batch_size, height, width)

    # Create a mask to exclude class 0
    mask = tf.cast(y_true != 0, tf.float32)  # Shape: (batch_size, height, width)

    # Calculate correct predictions (excluding class 0)
    correct = tf.reduce_sum(tf.cast((y_pred == y_true), tf.float32) * mask)

    # Calculate total pixels (excluding class 0)
    total = tf.reduce_sum(mask)

    # Avoid division by zero
    accuracy = tf.cond(total > 0, lambda: correct / total, lambda: tf.constant(0.0))
    return accuracy



def class_wise_iou(y_true, y_pred, class_idx):
    """
    Computes IoU for a specific class over the entire batch.
    Args:
        y_true: Ground truth one-hot encoded labels.
        y_pred: Predicted probabilities.
        class_idx: Index of the class to compute IoU for.
    """
    y_true = tf.argmax(y_true, axis=-1)  # Convert to integer labels
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert to integer labels

    intersection = tf.reduce_sum(tf.cast((y_true == class_idx) & (y_pred == class_idx), tf.float32))
    union = tf.reduce_sum(tf.cast((y_true == class_idx) | (y_pred == class_idx), tf.float32))

    return tf.cond(union > 0, lambda: intersection / union, lambda: tf.constant(0.0))

# Functions to be passed to the metrics argument of model.fit
def background_iou(y_true, y_pred):
    return class_wise_iou(y_true, y_pred, 0)

def class_1_iou(y_true, y_pred):
    return class_wise_iou(y_true, y_pred, 1)
    
def class_2_iou(y_true, y_pred):
    return class_wise_iou(y_true, y_pred, 2)
    
def class_3_iou(y_true, y_pred):
    return class_wise_iou(y_true, y_pred, 3)
    
def class_4_iou(y_true, y_pred):
    return class_wise_iou(y_true, y_pred, 4)




