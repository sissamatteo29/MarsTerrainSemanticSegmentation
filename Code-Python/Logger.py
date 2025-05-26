
import tensorflow as tf

class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log training metrics
        print()
        print()
        print(f"Epoch {epoch + 1}:")
        print(f"  Training Loss: {logs.get('loss'):.4f}")
        print(f"  Training Mean IoU: {logs.get('mean_iou'):.4f}")
        print(f"  Training Pixel Accuracy (excluding class 0): {logs.get('pixel_accuracy_exclude_class_0'):.4f}")
        print(f"  Training Background IoU: {logs.get('background_iou'):.4f}")
        print(f"  Training Class 1 IoU: {logs.get('class_1_iou'):.4f}")
        print(f"  Training Class 2 IoU: {logs.get('class_2_iou'):.4f}")
        print(f"  Training Class 3 IoU: {logs.get('class_3_iou'):.4f}")
        print(f"  Training Class 4 IoU: {logs.get('class_4_iou'):.4f}")

        # Log validation metrics
        print()
        print(f"  Validation Loss: {logs.get('val_loss'):.4f}")
        print(f"  Validation Mean IoU: {logs.get('val_mean_iou'):.4f}")
        print(f"  Validation Pixel Accuracy (excluding class 0): {logs.get('val_pixel_accuracy_exclude_class_0'):.4f}")
        print(f"  Validation Background IoU: {logs.get('val_background_iou'):.4f}")
        print(f"  Validation Class 1 IoU: {logs.get('val_class_1_iou'):.4f}")
        print(f"  Validation Class 2 IoU: {logs.get('val_class_2_iou'):.4f}")
        print(f"  Validation Class 3 IoU: {logs.get('val_class_3_iou'):.4f}")
        print(f"  Validation Class 4 IoU: {logs.get('val_class_4_iou'):.4f}")
        print()