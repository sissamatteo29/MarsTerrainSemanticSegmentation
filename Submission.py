import numpy as np
import keras
import pandas as pd


def y_to_df(y) -> pd.DataFrame:
    """Converts segmentation predictions into a DataFrame format for Kaggle."""
    n_samples = len(y)
    y_flat = y.reshape(n_samples, -1)
    df = pd.DataFrame(y_flat)
    df["id"] = np.arange(n_samples)
    cols = ["id"] + [col for col in df.columns if col != "id"]
    return df[cols]

def prepare_submission(model_filename, test_set, output_name):

    # Load model
    model = keras.models.load_model(model_filename, compile=False)
    print(f"Model loaded from {model_filename}")

    # Make predictions (and turn them into sparse labels)
    preds = model.predict(test_set)
    print(f"Predictions shape before argmax: {preds.shape}")
    preds = np.argmax(preds, axis=-1)
    print(f"Predictions shape: {preds.shape}")

    submission_df = y_to_df(preds)
    submission_df.to_csv(output_name, index=False)
    print(f"Test prediction file saved as {output_name}")
