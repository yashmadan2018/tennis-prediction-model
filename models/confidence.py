"""
Confidence intervals for win probability predictions.

Method: distance-ratio k-NN CI on the 2022-2023 validation set.

For each new prediction:
  1. Find the k=30 closest validation-set matches in normalised feature space.
  2. Compute mean_nn_dist = average distance to those k neighbours.
  3. ratio = mean_nn_dist / ref_dist
       ref_dist: precomputed median LOO k-NN distance for the val set
       — anchors what "well-covered" looks like.
  4. half_width = clip(0.045 × exp(4 × (ratio − 1)),  0.015, 0.25)
  5. CI = [prob_a − half_width, prob_a + half_width], clipped to [0, 1].

Why this gives sensible widths
-------------------------------
  • ratio < 1: prediction is in a DENSE region of feature space (many similar
    historical matches) → narrow CI → SHARP.
  • ratio ≈ 1: average coverage → MODERATE.
  • ratio > 1: sparse region, model must extrapolate → wide CI → WIDE.
  In practice: hard-court matches sit below the reference (more training data),
  grass matches sit above it (fewer matches, sparse coverage) — matching
  intuition about surface data richness.

  P_nn.std() was tried but fails: all k=30 neighbourhoods produce std ≈ 0.085
  regardless of distance, because the model's local variance is similar
  everywhere.  The distance ratio is the informative signal.

Tiers
-----
  SHARP    : width < 0.08   — act on these
  MODERATE : width 0.08–0.15 — act with awareness
  WIDE     : width > 0.15   — model is extrapolating; use as context only

Precomputation
--------------
BootstrapCI loads the val-set features and predictions once at instantiation
and precomputes the reference distance (~2 s).  Subsequent calls to .compute()
are pure numpy — no model calls, no I/O.  Speed: ~5 ms per call.

Usage
-----
  from models.confidence import BootstrapCI
  ci = BootstrapCI(model, feat_cols)         # ~2s on first load
  result = ci.compute(X_vec, prob_a=prob)    # numpy array, same order as feat_cols
  # → {"prob_low": 0.481, "prob_high": 0.565, "confidence_width": 0.084,
  #    "confidence_tier": "MODERATE"}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "train_features.csv"

# ── tier thresholds ────────────────────────────────────────────────────────────

SHARP_THRESHOLD    = 0.08
MODERATE_THRESHOLD = 0.15

TIER_SHARP    = "SHARP"
TIER_MODERATE = "MODERATE"
TIER_WIDE     = "WIDE"


def confidence_tier(width: float) -> str:
    if width < SHARP_THRESHOLD:
        return TIER_SHARP
    if width < MODERATE_THRESHOLD:
        return TIER_MODERATE
    return TIER_WIDE


# ── bootstrap CI ───────────────────────────────────────────────────────────────

class BootstrapCI:
    """
    Local k-NN neighbourhood confidence interval estimator.

    Parameters
    ----------
    model        : fitted CalibratedClassifierCV (or any sklearn estimator)
    feat_cols    : ordered list of feature column names
    k_neighbors  : neighbourhood size (default 30)
    seed         : random seed for reproducibility
    val_path     : override path to train_features.csv
    """

    def __init__(
        self,
        model,
        feat_cols:   list[str],
        k_neighbors: int = 30,
        seed:        int = 42,
        val_path:    Path | None = None,
    ) -> None:
        self.k   = k_neighbors
        self.rng = np.random.default_rng(seed)

        # ── load 2022-2023 validation rows ────────────────────────────────────
        src = val_path or FEATURES_CSV
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found. Run python -m models.train first."
            )

        df = pd.read_csv(src, parse_dates=["date"])
        val_mask = (df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2023)
        val_df   = df[val_mask].copy()

        if len(val_df) == 0:
            raise ValueError("No 2022-2023 rows found in train_features.csv.")

        # Feature matrix
        X_val = val_df[[c for c in feat_cols if c in val_df.columns]].apply(
            pd.to_numeric, errors="coerce"
        )
        for c in feat_cols:
            if c not in X_val.columns:
                X_val[c] = np.nan
        X_val = X_val[feat_cols].values  # (n_val, n_feat)

        # ── precompute model predictions on the val set ───────────────────────
        X_df = pd.DataFrame(X_val, columns=feat_cols)
        self.P_val: np.ndarray = model.predict_proba(X_df)[:, 1]  # (n_val,)

        # ── normalisation stats for distance computation ──────────────────────
        self._feat_mean: np.ndarray = np.nanmean(X_val, axis=0)
        self._feat_std:  np.ndarray = np.nanstd(X_val,  axis=0)

        # Sanitise: all-NaN features get mean=0 and std=1 so they contribute
        # zero distance (NaN < 1e-8 is False in numpy — must handle explicitly)
        self._feat_mean = np.where(np.isnan(self._feat_mean), 0.0, self._feat_mean)
        self._feat_std  = np.where(np.isnan(self._feat_std),  1.0, self._feat_std)
        self._feat_std[self._feat_std < 1e-8] = 1.0   # avoid /0 for constant features

        # Normalised val matrix (NaN → feature mean = 0 for all-NaN cols)
        X_filled = np.where(np.isnan(X_val), self._feat_mean, X_val)
        self._X_val_norm: np.ndarray = (
            (X_filled - self._feat_mean) / self._feat_std
        )

        # ── reference distance (LOO k-NN on a val-set sample) ────────────────
        # Median of mean-k-NN-distance across 200 random val-set points
        # (leave-one-out, so self is excluded).  Captures the "typical" density
        # of a well-represented prediction and anchors the width formula.
        rng_ref   = np.random.default_rng(seed)
        n_ref     = min(200, len(self._X_val_norm))
        ref_idx   = rng_ref.choice(len(self._X_val_norm), size=n_ref, replace=False)
        ref_dists = []
        for i in ref_idx:
            d = np.linalg.norm(self._X_val_norm - self._X_val_norm[i], axis=1)
            d[i] = np.inf   # exclude self
            kk = min(k_neighbors, len(d) - 1)
            ref_dists.append(float(np.partition(d, kk - 1)[:kk].mean()))
        self._ref_dist: float = float(np.median(ref_dists))

        print(
            f"[ci] BootstrapCI ready: {len(self.P_val):,} val rows, "
            f"k={k_neighbors}, ref_dist={self._ref_dist:.3f}"
        )

    # ── per-prediction CI ─────────────────────────────────────────────────────

    def compute(self, X_vec: np.ndarray, prob_a: float | None = None) -> dict:
        """
        Compute confidence interval for a single prediction.

        Parameters
        ----------
        X_vec  : 1-D numpy array with one value per feature (same order as
                 feat_cols passed at construction).  NaN is allowed.
        prob_a : point prediction from the model (optional but recommended).
                 When provided the CI is CENTRED on prob_a; the neighbourhood
                 std supplies only the WIDTH.
                 When omitted the CI is centred on the neighbourhood mean.

        Algorithm
        ---------
        1. Find the k=30 nearest val-set matches in normalised feature space.
        2. Compute mean distance to those k neighbours (mean_nn_dist).
        3. ratio = mean_nn_dist / ref_dist
             ref_dist: precomputed median LOO k-NN distance across the val set,
             anchoring what "typical well-covered" looks like.
        4. half_width = 0.045 × exp(4 × (ratio − 1)), clipped to [0.015, 0.25]
             — exponential in the distance ratio:
               ratio < 1 → prediction in a dense region → narrow CI → SHARP
               ratio ≈ 1 → average coverage → MODERATE
               ratio > 1 → sparse region, model extrapolating → wide CI → WIDE
        5. CI = [prob_a − half_width, prob_a + half_width], clipped to [0,1].

        Using P_nn.std() was tried first but fails to differentiate cases:
        all k=30 neighbourhoods produce P_nn.std() ≈ 0.085 regardless of
        distance, because the model's local variance is similar everywhere.
        The distance ratio is the informative signal.

        Returns
        -------
        dict with keys:
            prob_low          float  lower bound (centred on prob_a)
            prob_high         float  upper bound
            confidence_width  float  prob_high − prob_low  (= 2 × half_width)
            confidence_tier   str    SHARP | MODERATE | WIDE
        """
        # Normalise input (replace NaN with training mean)
        x_filled = np.where(np.isnan(X_vec), self._feat_mean, X_vec)
        x_norm   = (x_filled - self._feat_mean) / self._feat_std

        # k nearest neighbours by Euclidean distance in normalised space
        dists  = np.linalg.norm(self._X_val_norm - x_norm, axis=1)
        k      = min(self.k, len(dists))
        nn_idx = np.argpartition(dists, k - 1)[:k]

        # Distance-ratio width: how far we had to reach for k neighbours
        # relative to the typical reach in a well-covered region of the val set.
        mean_nn_dist = float(dists[nn_idx].mean())
        ratio        = mean_nn_dist / max(self._ref_dist, 1e-6)
        half_width   = float(np.clip(0.045 * np.exp(4.0 * (ratio - 1.0)), 0.015, 0.25))

        if prob_a is not None:
            centre = prob_a
        else:
            centre = float(self.P_val[nn_idx].mean())

        p5  = max(0.0, centre - half_width)
        p95 = min(1.0, centre + half_width)

        width = float(p95 - p5)

        return {
            "prob_low":         round(p5,    4),
            "prob_high":        round(p95,   4),
            "confidence_width": round(width, 4),
            "confidence_tier":  confidence_tier(width),
        }

    def compute_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Batch version: compute CI for every row in X.
        Returns DataFrame with the four CI columns.
        """
        rows = [self.compute(row) for row in X.values]
        return pd.DataFrame(rows, index=X.index)


# ── convenience ────────────────────────────────────────────────────────────────

def format_ci(
    player_name: str,
    prob:        float,
    prob_low:    float,
    prob_high:   float,
    tier:        str,
) -> str:
    """
    Format a single player's probability line.

    Example output:
        Sinner 52.3% [48.1%–56.5%] SHARP
    """
    return (
        f"{player_name} {prob:.1%} "
        f"[{prob_low:.1%}–{prob_high:.1%}] "
        f"{tier}"
    )
