"""
Model evaluation: calibration, log loss, Brier score.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def evaluate(model_path: str, test_features_path: str) -> None:
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    model = artefact["model"]
    feature_cols = artefact["feature_cols"]

    df = pd.read_csv(test_features_path, parse_dates=["date"])
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    probs = model.predict_proba(X)[:, 1]

    print(f"Log loss:    {log_loss(y, probs):.4f}")
    print(f"Brier score: {brier_score_loss(y, probs):.4f}")

    # Calibration plot
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/calibration_curve.png", dpi=150)
    print("Calibration curve saved to output/calibration_curve.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_path> <test_features_csv>")
    else:
        evaluate(sys.argv[1], sys.argv[2])
