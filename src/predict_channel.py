"""
predict_channel.py
------------------
Load the trained ML model and predict whether a channel is **busy** or **free**
given signal characteristics (SNR, signal_power, frequency, channel_load).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Feature columns the model was trained on
FEATURE_COLS = ["SNR", "signal_power", "frequency", "channel_load"]


def load_model(model_path: str | None = None):
    """Load the trained Random Forest model from disk."""
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "trained_model.pkl")
    return joblib.load(model_path)


def predict_channel_status(
    model,
    snr: float,
    signal_power: float,
    frequency: float,
    channel_load: int,
) -> str:
    """
    Predict whether a channel is 'busy' or 'free'.

    Parameters
    ----------
    model : sklearn estimator
    snr : float          - Signal-to-Noise Ratio (dB)
    signal_power : float - received signal power
    frequency : float    - channel frequency band (MHz)
    channel_load : int   - current number of devices on the channel

    Returns
    -------
    str - "busy" or "free"
    """
    features = pd.DataFrame(
        [[snr, signal_power, frequency, channel_load]],
        columns=FEATURE_COLS,
    )
    prediction = model.predict(features)[0]
    return prediction


def predict_channel_proba(
    model,
    snr: float,
    signal_power: float,
    frequency: float,
    channel_load: int,
) -> dict:
    """
    Return probability estimates for each class.

    Returns
    -------
    dict  e.g. {"busy": 0.72, "free": 0.28}
    """
    features = pd.DataFrame(
        [[snr, signal_power, frequency, channel_load]],
        columns=FEATURE_COLS,
    )
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    return {cls: float(p) for cls, p in zip(classes, proba)}


def predict_batch(model, feature_df: pd.DataFrame) -> np.ndarray:
    """Predict for multiple rows at once. Returns array of 'busy'/'free'."""
    return model.predict(feature_df[FEATURE_COLS])


# -- quick test --------------------------------------------------------------
if __name__ == "__main__":
    m = load_model()
    result = predict_channel_status(m, snr=-55.0, signal_power=-55.0, frequency=1800, channel_load=7)
    print(f"Prediction: {result}")
    proba = predict_channel_proba(m, snr=-55.0, signal_power=-55.0, frequency=1800, channel_load=7)
    print(f"Probabilities: {proba}")
