# Segmentation Model Library

To make the process of testing different models, parameters and techniques consistent among team members, the code base has been organised into python files (libraries) that served as the main source of code for all notebooks. In this way, all notebooks could execute the latest version of the code with all enhancements and improvements. All team mates contributed to the construction of this code base, tailored to the needs of this project.  
 To facilitate the navigation of the library, this README file briefly lists the purpose and content of each python file.

---

## **Files and Functions**

### 1. `Submission.py`
- **Purpose**: Prepares test predictions for submission.
- **Functions**:
  - `prepare_submission`: Loads a model, predicts on test data, and saves predictions (csv file).

---

### 2. `Losses.py`
- **Purpose**: Contains custom loss functions for model training.
- **Functions**:
  - `weighted_categorical_crossentropy`
  - `dice_loss_exclude_0`: Computes Dice loss excluding the background.
  - `weighted_focal_loss_1/2`: Implements focal loss with different configurations.
  - Hybrid loss combinations (e.g., `crossentropy_and_dice`).

---

### 3. `Analysis.py`
- **Purpose**: Analyzes training and validation metrics and results.
- **Functions**:
  - `plot_confusion_matrix`: Visualizes confusion matrices.
  - `plot_training_results`: Plots training history for metrics.

---

### 4. `Logger.py`
- **Purpose**: Logs training and validation metrics during model training.
- **Functions**:
  - `MetricsLoggerCallback`: Custom callback for detailed metric logging.

---

### 5. `Plot.py`
- **Purpose**: Visualizes images, masks, and augmentations.
- **Functions**:
  - `plot_image_and_mask`: Displays a single image-mask pair.
  - `plot_images_and_masks`: Visualizes multiple images and masks.
  - `plot_images_and_masks_augmented`: Compares original and augmented data (as image - mask pairs).

---

### 6. `DatasetFlow.py`
- **Purpose**: Prepares and augments datasets.
- **Functions**:
  - `data_flow`: Defines the entire preprocessing pipeline.
  - `balance_dataset`: Balances datasets across classes.
  - `merge_datasets`: Combines datasets into a single pipeline.

---

### 7. `Main.py`
- **Purpose**: Orchestrates the training pipeline.
- **Highlights**:
  - Data loading, augmentation, and model compilation.
  - Training loop with metrics and callbacks.
  - Saves the trained model and generates a submission file.

---

### 8. `Model.py`
- **Purpose**: Defines the model architectures.
- **Functions**:
  - `u_net`: Classic UNet architecture.
  - `attention_u_net`: UNet with attention mechanisms.
  - `dense_u_net`: UNet with dense connectivity.

---

### 9. `Preprocessing.py`
- **Purpose**: Provides data preprocessing and patch extraction.
- **Functions**:
  - `load_data`: Loads the dataset from a `.npz` file and splits it into training and testing sets.
  - `remove_background_images`: Filters out images and masks with a high percentage of background pixels.
  - `compute_class_distribution`: Calculates and prints the percentage distribution of each class in the dataset.
  - `compute_class_weights`: Computes class weights for loss functions based on class distribution.
  - `resize_crop`: Resizes and crops regions with specific classes.
  - `extract_representative_for_class`: Extracts representative samples for a class.
  - `split_train_data`: Splits data into training and validation sets.
  - `extract_relevant_patches_dataset`: Extracts patches from alls image-mask pairs in a dataset with sufficient coverage of a specified class.
  - `extraction_class_4_samples`: A pipeline function that combines various methods to extract and process samples for Class 4.

---

### 10. `AugmentationHelper.py`
- **Purpose**: Implements augmentation utilities (Albumentations library).
- **Functions**:
  - `map_geometric_transform`: Applies geometric augmentations.
  - `map_intensity_transform`: Adjusts image intensity and colors.
  - `map_total_transform`: Combines geometric and intensity transformations.

---

### 11. `Metrics.py`
- **Purpose**: Defines metrics for model evaluation.
- **Functions**:
  - `mean_iou`: Computes mean IoU, excluding the background.
  - `pixel_accuracy_exclude_class_0`: Measures pixel accuracy excluding class 0.
  - Class-wise IoU metrics for each terrain class.