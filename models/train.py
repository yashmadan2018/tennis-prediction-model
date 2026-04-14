"""
Model training: XGBoost win probability classifier.

Trains on historical match feature rows produced by features/pipeline.py.
Saves model artefacts to models/saved/.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

LABEL_COL = "label"  # 1 = player_a won, 0 = player_b won

FEATURE_COLS = [
    "elo_diff",
    "a_first_serve_pct", "a_first_srv_won_pct", "a_second_srv_won_pct",
    "a_hold_pct", "a_bp_save_pct",
    "a_first_ret_won_pct", "a_second_ret_won_pct", "a_break_pct",
    "b_first_serve_pct", "b_first_srv_won_pct", "b_second_srv_won_pct",
    "b_hold_pct", "b_bp_save_pct",
    "b_first_ret_won_pct", "b_second_ret_won_pct", "b_break_pct",
    "a_form_score", "b_form_score",
    "h2h_win_rate",
    "best_of_5",
    "a_days_rest", "b_days_rest",
    "a_prev_match_minutes", "b_prev_match_minutes",
    "a_venue_win_rate", "b_venue_win_rate",
    "a_rank_current", "b_rank_current",
    "a_rank_delta_3mo", "b_rank_delta_3mo",
    "lefty_righty_matchup",
    "a_injury_flag", "b_injury_flag",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
}


def train(features_path: str, model_name: str = "xgb_base") -> None:
    df = pd.read_csv(features_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present_cols].astype(float)
    y = df[LABEL_COL].astype(int)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X, y)

    out_path = MODEL_DIR / f"{model_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": calibrated, "feature_cols": present_cols}, f)

    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <features_csv_path>")
    else:
        train(sys.argv[1])
