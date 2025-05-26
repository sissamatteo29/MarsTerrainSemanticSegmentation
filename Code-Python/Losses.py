import tensorflow as tf



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of categorical crossentropy.

    Args:
        weights: A tensor or list of shape (num_classes,) representing the weight for each class.

    Returns:
        A loss function for use in Keras `model.compile`.

    Notice: to make sure this loss function doesn't count the contribution of class 0, simply
    pass a set of weights with 0 weight for class 0.
    """
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute the weighted cross-entropy
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred) * weights, axis=-1)
        
        # Return mean loss for the batch
        return tf.reduce_mean(loss)

    return loss




def dice_loss_exclude_0(y_true, y_pred, smooth=1e-6):

    """
    Implementation of the dice loss excluding the contribution of class 0 (background)
    This function first computes the dice coefficient for each class over the entire batch (similar to the IoU metric).
    Then it produces an average of the dice coefficients over all classes.
    It returns the loss as 1 - average.
    """

    # Compute intersection and union over the entire batch of images (this step reduces the batch to a single
    # vector of intersection and unions for each class)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2])

    # Compute Dice coefficient for each class
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)

    # Average Dice coefficient across classes for each sample
    # Notice: index 0 is excluded from the mean as the background predictions do not contribute to the loss
    mean_dice_coefficient = tf.reduce_mean(dice_coeff[1:], axis=-1)

    # Compute mean Dice loss across the batch
    return 1 - mean_dice_coefficient





def weighted_focal_loss(weights=None, gamma=2.0, alpha=0.5):
    """ 
    Wrapper for model.compile 
    Notice: the weights can be passed in as normalised or not, since the reduction is with the sum, if the weights are high, they will be internally
    normalised.
    """

    def focal_loss(y_true, y_pred):
        """
        Compute the Focal Loss, the reduction steps in this function are handled in this way:
        - Sum all focal coefficients over an image
        - Reduce over the batch averaging
        To make the loss function more contained, it is necessary to normalise the weights and a coefficient alpha is provided to empirically
        reduce the loss if needed
    
        Args:
        - y_true: Ground truth tensor, one-hot encoded. Shape: [batch_size, height, width, num_classes].
        - y_pred: Predicted tensor (logits or probabilities). Shape: [batch_size, height, width, num_classes].
        - weights: Balancing factor for classes (default 0.25).
        - gamma: Focusing parameter (default 2.0).
        - reduction: 'mean' or 'sum' to specify the aggregation method.
    
        Returns:
        - Focal loss value.
    
        Notice: to exclude class 0 from the computation, simply set a weight of 0 when passing in weights
        """
    
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    
        # Compute cross-entropy loss, Shape: [batch_size, height, width, num_classes]
        ce_loss = -y_true * tf.math.log(y_pred)
    
        # Compute the focal weigths
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, gamma)
    
        if weights is not None:
            class_weights = tf.convert_to_tensor(weights, dtype=tf.float32)  # Ensure weights is a tensor
            class_weights = tf.reshape(weights, [1, 1, 1, -1])  # Broadcast to match shape
            # Notice: scale the weights to prevent too high loss
            class_weights = tf.cond(
                tf.reduce_max(class_weights) > 5,
                lambda: class_weights / tf.reduce_sum(class_weights),  # If condition is True
                lambda: class_weights  # If condition is False
            )
            class_weights = tf.cast(class_weights, tf.float32)
            ce_loss = ce_loss * class_weights  # Apply class weights
    
        # Apply focal weight to the cross-entropy loss
        focal_loss = focal_weight[..., tf.newaxis] * ce_loss  # Shape: [batch_size, height, width, num_classes]
    
        # Sum over classes dimension
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)  # Shape: [batch_size, height, width]
    
        # Aggregate spatial dimensions (height and width)
        focal_loss_per_image = tf.reduce_sum(focal_loss, axis=[1, 2])  # Shape: [batch_size]
    
        # Aggregate across the batch
        return alpha * tf.reduce_mean(focal_loss_per_image)  # Single scalar


    return focal_loss
