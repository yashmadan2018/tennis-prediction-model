"""
Ensemble model: XGBoost + Logistic Regression + MLP.

Three calibrated classifiers combined via a weighted average.  Weights
are tuned per-context:
  • ATP / WTA Tour : XGB 0.50  LR 0.25  MLP 0.25
  • Challenger     : XGB 0.35  LR 0.40  MLP 0.25  (LR up-weighted for sparse data)

Architecture
------------
  XGBoost  – already in models/saved/xgb_calibrated.pkl
  LR       – Pipeline(SimpleImputer → StandardScaler → LogisticRegression(L2))
               wrapped in CalibratedClassifierCV(method='isotonic', cv=5)
  MLP      – Pipeline(SimpleImputer → StandardScaler → MLPClassifier)
               hidden=(64,32,16), relu, early_stopping as regularisation
               wrapped in CalibratedClassifierCV(method='isotonic', cv=5)

All three are trained on the 2015-2021 training set (via cv=5 calibration on
the full train+val 2015-2023 for the saved artefacts).

Usage
-----
  # Train LR + MLP, save, print comparison table
  python -m models.ensemble

  # Faster run (cv=2, smaller MLP) — useful for smoke tests
  python -m models.ensemble --fast

  # Skip final retraining on train+val — only compare on val
  python -m models.ensemble --compare-only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── ensemble weights ───────────────────────────────────────────────────────────

WEIGHTS: dict[str, dict[str, float]] = {
    "atp":        {"xgb": 0.50, "lr": 0.25, "mlp": 0.25},
    "wta":        {"xgb": 0.50, "lr": 0.25, "mlp": 0.25},
    "challenger": {"xgb": 0.35, "lr": 0.40, "mlp": 0.25},
}
_DEFAULT_WEIGHTS = WEIGHTS["atp"]


def get_weights(tour: str = "atp", tourney_level: str = "A") -> dict[str, float]:
    """Return weight dict for given tour / tournament level."""
    t = tour.lower()
    if tourney_level == "C" or "challenger" in t:
        return WEIGHTS["challenger"]
    if "wta" in t:
        return WEIGHTS["wta"]
    return WEIGHTS["atp"]


# ── EnsembleModel ──────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted average of XGBoost, Logistic Regression, and MLP.

    sklearn-compatible: predict_proba(X) returns shape (n, 2) and can be
    used as a drop-in replacement for CalibratedClassifierCV everywhere,
    including BootstrapCI which calls model.predict_proba(X_df)[:, 1].

    For context-weighted predictions (Challenger vs Tour), pass tour and
    tourney_level keyword arguments.
    """

    def __init__(
        self,
        xgb_model,
        lr_model,
        mlp_model,
        feat_cols: list[str],
    ) -> None:
        self.xgb       = xgb_model
        self.lr        = lr_model
        self.mlp       = mlp_model
        self.feat_cols = feat_cols

    # ── internal helpers ───────────────────────────────────────────────────────

    def _p1(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return P(class=1) for each component model."""
        return (
            self.xgb.predict_proba(X)[:, 1],
            self.lr.predict_proba(X)[:, 1],
            self.mlp.predict_proba(X)[:, 1],
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: pd.DataFrame,
        tour: str = "atp",
        tourney_level: str = "A",
    ) -> np.ndarray:
        """
        Weighted ensemble probability.

        Returns ndarray of shape (n, 2) where column 1 = P(player_a wins).
        Default weights are ATP Tour; pass tour/tourney_level for context
        (e.g. tour='atp', tourney_level='C' → Challenger weights).
        """
        w = get_weights(tour, tourney_level)
        p_xgb, p_lr, p_mlp = self._p1(X)
        p1 = w["xgb"] * p_xgb + w["lr"] * p_lr + w["mlp"] * p_mlp
        return np.column_stack([1.0 - p1, p1])

    def predict_all(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Return individual model probabilities and ensemble probability.

        Useful for inspection and debugging.
        """
        p_xgb, p_lr, p_mlp = self._p1(X)
        w = _DEFAULT_WEIGHTS
        p_ens = w["xgb"] * p_xgb + w["lr"] * p_lr + w["mlp"] * p_mlp
        return {"xgb": p_xgb, "lr": p_lr, "mlp": p_mlp, "ensemble": p_ens}

    # ── persistence ────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "EnsembleModel":
        """Load all three models from models/saved/ and construct EnsembleModel."""
        xgb_path = MODEL_DIR / "xgb_calibrated.pkl"
        lr_path  = MODEL_DIR / "lr_calibrated.pkl"
        mlp_path = MODEL_DIR / "mlp_calibrated.pkl"

        for p in (xgb_path, lr_path, mlp_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p.name} not found. Run: python -m models.ensemble"
                )

        with open(xgb_path, "rb") as f:
            xgb_bundle = pickle.load(f)
        with open(lr_path, "rb") as f:
            lr_bundle = pickle.load(f)
        with open(mlp_path, "rb") as f:
            mlp_bundle = pickle.load(f)

        feat_cols = xgb_bundle["feature_cols"]
        ensemble  = cls(
            xgb_model = xgb_bundle["model"],
            lr_model  = lr_bundle["model"],
            mlp_model = mlp_bundle["model"],
            feat_cols = feat_cols,
        )
        print(
            f"[ensemble] Loaded XGB + LR + MLP  "
            f"({len(feat_cols)} features)"
        )
        return ensemble


# ── pipeline factories ─────────────────────────────────────────────────────────

def _lr_pipeline() -> Pipeline:
    """LR pipeline: impute NaN → z-score normalise → L2 logistic regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )),
    ])


def _mlp_pipeline(fast: bool = False) -> Pipeline:
    """
    MLP pipeline: impute → normalise → 3-layer MLP.

    hidden_layer_sizes=(64, 32, 16), relu, early_stopping as a
    regularisation proxy (stops when val loss doesn't improve for
    n_iter_no_change consecutive epochs — approximates dropout in effect).
    """
    layers = (32, 16) if fast else (64, 32, 16)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
            random_state=42,
        )),
    ])


# ── training ──────────────────────────────────────────────────────────────────

def train_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Train calibrated LR on X_train.

    CalibratedClassifierCV(cv=cv) fits cv base models on cv-1 folds and
    uses each held-out fold for isotonic calibration.  No val-set leakage.
    """
    print(f"[ensemble] Training LR (cv={cv})...")
    t0  = time.time()
    cal = CalibratedClassifierCV(_lr_pipeline(), method="isotonic", cv=cv)
    cal.fit(X_train, y_train)
    print(f"[ensemble] LR done  ({time.time()-t0:.1f}s)")
    return cal


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    fast: bool = False,
) -> CalibratedClassifierCV:
    """
    Train calibrated MLP on X_train.

    early_stopping inside MLPClassifier acts as regularisation — each fold
    holds out 10% internally and stops when validation loss plateaus.
    """
    layers = "32-16" if fast else "64-32-16"
    print(f"[ensemble] Training MLP (cv={cv}, hidden={layers}, early_stopping)...")
    t0  = time.time()
    cal = CalibratedClassifierCV(_mlp_pipeline(fast=fast), method="isotonic", cv=cv)
    cal.fit(X_train, y_train)
    print(f"[ensemble] MLP done  ({time.time()-t0:.1f}s)")
    return cal


# ── save helpers ───────────────────────────────────────────────────────────────

def _save(model, name: str, feat_cols: list[str], val_brier: float) -> None:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols}, f)
    print(f"[ensemble] Saved {name} → {path}  (val_brier={val_brier:.5f})")


def _update_feature_json(lr_brier: float, mlp_brier: float, ens_brier: float) -> None:
    """Append ensemble metrics to feature_list.json."""
    path = MODEL_DIR / "feature_list.json"
    if not path.exists():
        return
    with open(path) as f:
        meta = json.load(f)
    meta["lr_val_brier"]  = round(lr_brier,  6)
    meta["mlp_val_brier"] = round(mlp_brier, 6)
    meta["ens_val_brier"] = round(ens_brier, 6)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ensemble] Updated feature_list.json with ensemble metrics.")


# ── comparison table ───────────────────────────────────────────────────────────

def print_comparison(
    X_tr:  pd.DataFrame,
    y_tr:  pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fast:  bool = False,
) -> dict[str, float]:
    """
    Train LR and MLP on X_tr (2015-2021, uncalibrated for speed),
    evaluate all models on X_val (2022-2023).  XGBoost uses the saved
    calibrated model (trained on train+val — noted in the table).

    Returns dict: {"xgb": brier, "lr": brier, "mlp": brier, "ensemble": brier}
    """
    print("\n[ensemble] ── Model Comparison (val set 2022-2023) ──────────")

    # ── XGBoost (saved, calibrated, trained on train+val) ─────────────────────
    with open(MODEL_DIR / "xgb_calibrated.pkl", "rb") as f:
        xgb_bundle = pickle.load(f)
    xgb_model = xgb_bundle["model"]
    p_xgb     = xgb_model.predict_proba(X_val)[:, 1]
    bs_xgb    = float(brier_score_loss(y_val, p_xgb))
    ll_xgb    = float(log_loss(y_val, p_xgb))
    print(f"  XGBoost (calibrated*)  : Brier={bs_xgb:.5f}  log-loss={ll_xgb:.5f}")

    # ── Logistic Regression (train-only, uncalibrated) ─────────────────────────
    print("  Fitting LR on train set...")
    t0    = time.time()
    lr_cmp = _lr_pipeline()
    lr_cmp.fit(X_tr, y_tr)
    p_lr  = lr_cmp.predict_proba(X_val)[:, 1]
    bs_lr = float(brier_score_loss(y_val, p_lr))
    ll_lr = float(log_loss(y_val, p_lr))
    print(f"  Logistic Regression    : Brier={bs_lr:.5f}  log-loss={ll_lr:.5f}  ({time.time()-t0:.1f}s)")

    # ── MLP (train-only, uncalibrated, early_stopping) ─────────────────────────
    layers = "32-16" if fast else "64-32-16"
    print(f"  Fitting MLP ({layers}) on train set...")
    t0     = time.time()
    mlp_cmp = _mlp_pipeline(fast=fast)
    mlp_cmp.fit(X_tr, y_tr)
    p_mlp  = mlp_cmp.predict_proba(X_val)[:, 1]
    bs_mlp = float(brier_score_loss(y_val, p_mlp))
    ll_mlp = float(log_loss(y_val, p_mlp))
    print(f"  MLP ({layers})         : Brier={bs_mlp:.5f}  log-loss={ll_mlp:.5f}  ({time.time()-t0:.1f}s)")

    # ── Ensemble (calibrated XGB + uncalibrated LR/MLP) ────────────────────────
    w     = _DEFAULT_WEIGHTS
    p_ens = w["xgb"] * p_xgb + w["lr"] * p_lr + w["mlp"] * p_mlp
    bs_ens = float(brier_score_loss(y_val, p_ens))
    ll_ens = float(log_loss(y_val, p_ens))
    print(f"  Ensemble (ATP weights) : Brier={bs_ens:.5f}  log-loss={ll_ens:.5f}")

    print(f"\n  * XGBoost trained on train+val 2015-2023; "
          f"LR/MLP trained on train-only 2015-2021.")

    scores = {"xgb": bs_xgb, "lr": bs_lr, "mlp": bs_mlp, "ensemble": bs_ens}
    best_name = min(scores, key=scores.__getitem__)

    print(f"\n  ─────────────────────────────────────────────────────────")
    print(f"  {'Model':<25}  Brier     Δ vs XGB")
    for name, bs in scores.items():
        delta = bs - bs_xgb
        tag   = "  ← best" if name == best_name else ""
        print(f"  {name:<25}  {bs:.5f}   {delta:+.5f}{tag}")
    print()

    return scores


# ── main entry point ───────────────────────────────────────────────────────────

def main(fast: bool = False, compare_only: bool = False) -> None:
    sys.path.insert(0, str(ROOT))
    from models.train import prepare_splits, drop_no_elo_rows, FEATURE_COLS

    FEATURES_CSV = ROOT / "data" / "processed" / "train_features.csv"
    if not FEATURES_CSV.exists():
        print("[ensemble] ERROR: train_features.csv not found. "
              "Run: python -m models.train")
        sys.exit(1)

    print(f"[ensemble] Loading feature dataset...")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    print(f"[ensemble] {len(df):,} rows loaded")

    df        = drop_no_elo_rows(df)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    X_tr, y_tr, X_val, y_val, _X_te, _y_te = prepare_splits(df)
    X_tr  = X_tr[feat_cols]
    X_val = X_val[feat_cols]

    cv = 2 if fast else 5

    # ── Step 1: comparison table ───────────────────────────────────────────────
    scores = print_comparison(X_tr, y_tr, X_val, y_val, fast=fast)

    if compare_only:
        print("[ensemble] --compare-only: skipping final model training.")
        return

    # ── Step 2: train final calibrated models on train+val ────────────────────
    X_combined = pd.concat([X_tr, X_val], ignore_index=True)
    y_combined = pd.concat([y_tr, y_val], ignore_index=True)

    print(f"\n[ensemble] Training final models on train+val "
          f"({len(X_combined):,} rows, cv={cv})...")

    lr_final  = train_lr(X_combined, y_combined, cv=cv)
    mlp_final = train_mlp(X_combined, y_combined, cv=cv, fast=fast)

    # ── Step 3: evaluate final models on val ──────────────────────────────────
    print("\n[ensemble] ── Final model val-set metrics (train+val → val) ──")
    with open(MODEL_DIR / "xgb_calibrated.pkl", "rb") as f:
        xgb_bundle = pickle.load(f)
    xgb_model = xgb_bundle["model"]

    p_xgb_f = xgb_model.predict_proba(X_val)[:, 1]
    p_lr_f  = lr_final.predict_proba(X_val)[:, 1]
    p_mlp_f = mlp_final.predict_proba(X_val)[:, 1]

    w       = _DEFAULT_WEIGHTS
    p_ens_f = w["xgb"]*p_xgb_f + w["lr"]*p_lr_f + w["mlp"]*p_mlp_f

    bs_lr_f  = float(brier_score_loss(y_val, p_lr_f))
    bs_mlp_f = float(brier_score_loss(y_val, p_mlp_f))
    bs_ens_f = float(brier_score_loss(y_val, p_ens_f))
    bs_xgb_f = float(brier_score_loss(y_val, p_xgb_f))

    print(f"  XGBoost  : {bs_xgb_f:.5f}")
    print(f"  LR       : {bs_lr_f:.5f}")
    print(f"  MLP      : {bs_mlp_f:.5f}")
    print(f"  Ensemble : {bs_ens_f:.5f}")

    # ── Step 4: save ───────────────────────────────────────────────────────────
    _save(lr_final,  "lr_calibrated",  feat_cols, bs_lr_f)
    _save(mlp_final, "mlp_calibrated", feat_cols, bs_mlp_f)
    _update_feature_json(bs_lr_f, bs_mlp_f, bs_ens_f)

    print(f"\n[ensemble] Done. Run predict.py to use the ensemble.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble (LR + MLP + XGB)")
    parser.add_argument("--fast",         action="store_true",
                        help="cv=2, smaller MLP — faster smoke test")
    parser.add_argument("--compare-only", action="store_true",
                        help="Print comparison table only; don't save models")
    args = parser.parse_args()

    main(fast=args.fast, compare_only=args.compare_only)
