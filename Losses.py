import tensorflow as tf



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of categorical crossentropy.

    Args:
        weights: A tensor or list of shape (num_classes,) representing the weight for each class.

    Returns:
        A loss function for use in Keras `model.compile`.
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