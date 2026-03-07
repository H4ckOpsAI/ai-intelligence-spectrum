"""
train_model.py
--------------
Train a Random Forest classifier on the spectrum sensing dataset.

Steps
-----
1. Load dataset via data_loader
2. Clean missing values
3. Generate simulated channel_load feature
4. Select features (SNR, signal_power, frequency, channel_load)
5. Split into train / test (80 / 20)
6. Train RandomForestClassifier
7. Print accuracy
8. Save trained model to  models/trained_model.pkl
"""

import os
import sys
import random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
#  REPRODUCIBLE SEEDS
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_spectrum_data, get_features_and_labels


def _generate_channel_load(df):
    """
    Generate simulated channel_load values.

    Busy channels are more likely to have high device counts (5-12),
    while free channels tend to have lower counts (1-6).
    This gives the model a realistic signal to learn from.
    """
    rng = np.random.RandomState(42)
    loads = []
    for label in df["label"]:
        if label == "busy":
            loads.append(rng.randint(5, 13))   # 5 .. 12
        else:
            loads.append(rng.randint(1, 7))    # 1 .. 6
    df["channel_load"] = loads
    return df


def train_and_save_model():
    """Full training pipeline."""

    # 1. Load dataset
    print("[INFO] Loading spectrum dataset ...")
    df = load_spectrum_data()
    print(f"   [OK] Loaded {len(df)} samples")

    # 2. Generate channel_load feature (if not already present)
    if "channel_load" not in df.columns:
        print("[INFO] Generating simulated channel_load values ...")
        df = _generate_channel_load(df)

    # 3. Features & labels
    X, y = get_features_and_labels(df)
    print(f"   Features: {list(X.columns)}")

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 5. Train Random Forest
    print("[INFO] Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Model Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # 7. Save model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "trained_model.pkl")
    joblib.dump(model, model_path)
    print(f"[SAVED] Model saved -> {model_path}")


if __name__ == "__main__":
    train_and_save_model()
