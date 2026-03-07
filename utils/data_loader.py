"""
data_loader.py
--------------
Utility to load and preprocess the cognitive radio spectrum dataset.

The raw dataset contains columns like power_dB, PU_Presence,
PU_Signal_Strength, spectral_entropy, and Frequency_Band.
This module maps them to a simplified schema:
    SNR, signal_power, frequency, label (busy / free)
"""

import os
import pandas as pd


def load_spectrum_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the spectrum dataset and return a cleaned DataFrame with
    columns: SNR, signal_power, frequency, spectral_entropy, label.

    Parameters
    ----------
    csv_path : str, optional
        Absolute or relative path to the CSV file.  Defaults to
        ``data/spectrum_dataset.csv`` relative to the project root.

    Returns
    -------
    pd.DataFrame
    """
    if csv_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "data", "spectrum_dataset.csv")

    df = pd.read_csv(csv_path)

    # --- column mapping --------------------------------------------------
    # The CSV may have original names OR already-renamed names
    original_cols = {"power_dB", "PU_Signal_Strength", "Frequency_Band", "PU_Presence"}
    renamed_cols  = {"SNR", "signal_power", "frequency", "label"}

    if original_cols.issubset(df.columns):
        rename_map = {
            "power_dB": "SNR",
            "PU_Signal_Strength": "signal_power",
            "Frequency_Band": "frequency",
            "spectral_entropy": "spectral_entropy",
            "PU_Presence": "label",
        }
        cols_to_keep = list(rename_map.keys())
        df = df[cols_to_keep].copy()
        df.rename(columns=rename_map, inplace=True)
    else:
        # Already has renamed columns
        cols_to_keep = ["SNR", "signal_power", "frequency", "spectral_entropy", "label"]
        df = df[cols_to_keep].copy()

    # --- cleaning ---------------------------------------------------------
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert label: 0 → "free", 1 → "busy" (if numeric)
    if df["label"].dtype in ("int64", "float64", "int32", "float32"):
        df["label"] = df["label"].map({0: "free", 1: "busy"})

    return df


def get_features_and_labels(df: pd.DataFrame):
    """
    Split the DataFrame into feature matrix X and label vector y.

    Returns
    -------
    X : pd.DataFrame   - numeric features (SNR, signal_power, frequency, channel_load)
    y : pd.Series       - string labels ("busy" / "free")
    """
    feature_cols = ["SNR", "signal_power", "frequency", "channel_load"]
    # Only use columns that exist in the dataframe
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df["label"]
    return X, y


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_spectrum_data()
    print(f"Dataset shape: {data.shape}")
    print(data.head(10))
    print(f"\nLabel distribution:\n{data['label'].value_counts()}")
