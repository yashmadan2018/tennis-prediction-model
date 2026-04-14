"""
Model evaluation on the held-out 2024 test set.

Never used for tuning — pure evaluation only.

Outputs
-------
  output/calibration_curve.png   10-bucket calibration diagram
  output/feature_importance.png  top-20 XGBoost feature importances
  output/evaluation_report.txt   full text report (mirrors stdout)

Usage
-----
  python -m models.evaluate
"""

from __future__ import annotations

import io
import json
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
FEATURES_CSV = ROOT / "data"    / "processed" / "train_features.csv"
MODEL_PKL    = ROOT / "models"  / "saved"     / "xgb_calibrated.pkl"
FEATURE_JSON = ROOT / "models"  / "saved"     / "feature_list.json"
OUTPUT_DIR   = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_PATH  = OUTPUT_DIR / "evaluation_report.txt"
CAL_CURVE    = OUTPUT_DIR / "calibration_curve.png"
FEAT_IMP     = OUTPUT_DIR / "feature_importance.png"

LABEL_COL    = "label"

# ── tourney level labels ───────────────────────────────────────────────────────

LEVEL_LABEL = {
    5: "Grand Slam",
    4: "Masters / Premier",
    3: "ATP 500 / WTA Premier",
    2: "ATP 250 / WTA Intl",
    1: "Challenger / ITF",
}


# ── load artefacts ─────────────────────────────────────────────────────────────

def load_model_and_features() -> tuple:
    """Load pickled model and feature list. Returns (model, feat_cols, meta)."""
    with open(MODEL_PKL, "rb") as f:
        artefact = pickle.load(f)
    model     = artefact["model"]
    feat_cols = artefact["feature_cols"]

    with open(FEATURE_JSON) as f:
        meta = json.load(f)

    return model, feat_cols, meta


def load_test_set(feat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the 2024 test split.
    Returns (raw_df, X, y) where raw_df retains all columns for breakdowns.
    """
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    test = df[df["date"].dt.year >= 2024].copy().reset_index(drop=True)

    present = [c for c in feat_cols if c in test.columns]
    X = test[present].apply(pd.to_numeric, errors="coerce")
    y = test[LABEL_COL].astype(int)

    return test, X, y


# ── core metrics ───────────────────────────────────────────────────────────────

def core_metrics(y: pd.Series, probs: np.ndarray) -> dict:
    bs  = brier_score_loss(y, probs)
    ll  = log_loss(y, probs)
    acc = float(((probs >= 0.5) == y).mean())
    return {"brier": bs, "logloss": ll, "accuracy": acc, "n": len(y)}


# ── confidence-threshold accuracy ─────────────────────────────────────────────

def threshold_accuracy(
    y: pd.Series,
    probs: np.ndarray,
    thresholds: list[float] = [0.60, 0.65, 0.70, 0.75, 0.80],
) -> list[dict]:
    """
    For each threshold, report accuracy and sample size on the subset of
    matches where the model's confidence (max(p, 1-p)) >= threshold.
    """
    conf = np.maximum(probs, 1 - probs)
    correct = (probs >= 0.5) == np.array(y)
    rows = []
    for t in thresholds:
        mask = conf >= t
        n    = int(mask.sum())
        acc  = float(correct[mask].mean()) if n > 0 else np.nan
        rows.append({"threshold": t, "n": n, "pct_of_total": n / len(y), "accuracy": acc})
    return rows


# ── surface breakdown ──────────────────────────────────────────────────────────

def surface_breakdown(
    raw: pd.DataFrame, y: pd.Series, probs: np.ndarray
) -> list[dict]:
    rows = []
    for surf in ["hard", "clay", "grass"]:
        mask = raw["surface"].str.lower() == surf
        if mask.sum() < 10:
            continue
        m = core_metrics(y[mask], probs[mask])
        m["surface"] = surf.capitalize()
        rows.append(m)
    return rows


# ── tournament level breakdown ─────────────────────────────────────────────────

def level_breakdown(
    raw: pd.DataFrame, y: pd.Series, probs: np.ndarray
) -> list[dict]:
    if "tourney_level_ord" not in raw.columns:
        return []
    rows = []
    for lvl in sorted(raw["tourney_level_ord"].dropna().unique(), reverse=True):
        mask = raw["tourney_level_ord"] == lvl
        if mask.sum() < 10:
            continue
        m = core_metrics(y[mask], probs[mask])
        m["level"] = LEVEL_LABEL.get(int(lvl), f"Level {int(lvl)}")
        m["level_ord"] = int(lvl)
        rows.append(m)
    return rows


# ── calibration curve ─────────────────────────────────────────────────────────

def plot_calibration(
    y: pd.Series,
    probs: np.ndarray,
    out_path: Path = CAL_CURVE,
    n_bins: int = 10,
) -> list[dict]:
    """
    Plot calibration curve (reliability diagram) and return bin data.
    """
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=n_bins,
                                             strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: reliability diagram ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color="#2176AE", lw=2,
            markersize=7, label="Model")

    # Shade gap from perfect
    ax.fill_between(prob_pred, prob_pred, prob_true,
                    alpha=0.12, color="#E84855")

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Observed win rate", fontsize=12)
    ax.set_title("Calibration curve — 2024 test set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # ── right: probability histogram ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(probs[np.array(y) == 1], bins=20, alpha=0.65, color="#2176AE",
             label="Player A won", density=True)
    ax2.hist(probs[np.array(y) == 0], bins=20, alpha=0.65, color="#E84855",
             label="Player B won", density=True)
    ax2.set_xlabel("Predicted P(player A wins)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Predicted probability distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Build bin table
    bins = []
    for pp, pt in zip(prob_pred, prob_true):
        bins.append({"pred_prob": pp, "actual_win_rate": pt,
                     "gap": pt - pp})
    return bins


# ── feature importance chart ───────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feat_cols: list[str],
    out_path: Path = FEAT_IMP,
    top_n: int = 20,
) -> list[dict]:
    """
    Average feature importances across all XGBoost base learners in the
    CalibratedClassifierCV ensemble.
    """
    importances = np.zeros(len(feat_cols))
    n_est = 0
    for estimator in model.calibrated_classifiers_:
        base = estimator.estimator
        if hasattr(base, "feature_importances_"):
            importances += base.feature_importances_
            n_est += 1
    if n_est == 0:
        print("[eval] Could not extract importances from model.")
        return []
    importances /= n_est

    ranked = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)
    top    = ranked[:top_n]
    names  = [r[0] for r in top]
    vals   = [r[1] for r in top]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(names)), vals[::-1],
                   color="#2176AE", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Mean feature importance (gain)", fontsize=12)
    ax.set_title(f"Top {top_n} feature importances — 2024 test set",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    # Value labels
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return [{"feature": n, "importance": v} for n, v in ranked[:top_n]]


# ── report builder ─────────────────────────────────────────────────────────────

def build_report(
    meta: dict,
    overall: dict,
    thresh_rows: list[dict],
    surf_rows: list[dict],
    lvl_rows: list[dict],
    cal_bins: list[dict],
    feat_rows: list[dict],
) -> str:
    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    def row(label: str, value) -> None:
        lines.append(f"  {label:<40} {value}")

    # ── header ─────────────────────────────────────────────────────────────────
    lines.append("Tennis Prediction Model — Evaluation Report")
    lines.append("Test set: 2024 (held out, never used during training/tuning)")
    lines.append(f"Model: models/saved/xgb_calibrated.pkl")
    lines.append(f"Val Brier (2022-23, from training): {meta.get('val_brier', 'n/a'):.5f}")

    # ── overall ────────────────────────────────────────────────────────────────
    h("OVERALL — 2024 TEST SET")
    row("Matches",              f"{overall['n']:,}")
    row("Brier score",          f"{overall['brier']:.5f}")
    row("Log-loss",             f"{overall['logloss']:.5f}")
    row("Accuracy (>50% conf)", f"{overall['accuracy']*100:.2f}%")

    # ── confidence thresholds ──────────────────────────────────────────────────
    h("CONFIDENCE THRESHOLD ACCURACY")
    lines.append(f"  {'Threshold':<12} {'N':>7}  {'% of total':>10}  {'Accuracy':>10}")
    lines.append("  " + "-" * 46)
    for r in thresh_rows:
        acc_str = f"{r['accuracy']*100:.2f}%" if not np.isnan(r['accuracy']) else "  n/a"
        lines.append(f"  {r['threshold']*100:.0f}%+{'':<8} "
                     f"{r['n']:>7,}  "
                     f"{r['pct_of_total']*100:>9.1f}%  "
                     f"{acc_str:>10}")

    # ── surface breakdown ──────────────────────────────────────────────────────
    h("BY SURFACE")
    lines.append(f"  {'Surface':<12} {'N':>7}  {'Brier':>8}  {'Log-loss':>10}  {'Accuracy':>10}")
    lines.append("  " + "-" * 55)
    for r in surf_rows:
        lines.append(f"  {r['surface']:<12} "
                     f"{r['n']:>7,}  "
                     f"{r['brier']:>8.5f}  "
                     f"{r['logloss']:>10.5f}  "
                     f"{r['accuracy']*100:>9.2f}%")

    # ── tournament level breakdown ─────────────────────────────────────────────
    h("BY TOURNAMENT LEVEL")
    lines.append(f"  {'Level':<25} {'N':>7}  {'Brier':>8}  {'Log-loss':>10}  {'Accuracy':>10}")
    lines.append("  " + "-" * 66)
    for r in sorted(lvl_rows, key=lambda x: -x["level_ord"]):
        lines.append(f"  {r['level']:<25} "
                     f"{r['n']:>7,}  "
                     f"{r['brier']:>8.5f}  "
                     f"{r['logloss']:>10.5f}  "
                     f"{r['accuracy']*100:>9.2f}%")

    # ── calibration bins ───────────────────────────────────────────────────────
    h("CALIBRATION — PREDICTED VS ACTUAL (10 BUCKETS)")
    lines.append(f"  {'Pred prob':>10}  {'Actual rate':>12}  {'Gap':>8}")
    lines.append("  " + "-" * 36)
    for b in cal_bins:
        gap_str = f"{b['gap']:+.4f}"
        lines.append(f"  {b['pred_prob']*100:>9.1f}%  "
                     f"{b['actual_win_rate']*100:>11.1f}%  "
                     f"{gap_str:>8}")

    # ── feature importances ────────────────────────────────────────────────────
    h("TOP 20 FEATURE IMPORTANCES")
    for i, r in enumerate(feat_rows, 1):
        bar = "█" * int(r["importance"] * 400)
        lines.append(f"  {i:>2}. {r['feature']:<40}  {r['importance']:.4f}  {bar}")

    # ── artefact paths ─────────────────────────────────────────────────────────
    h("OUTPUT ARTEFACTS")
    lines.append(f"  Calibration curve:   output/calibration_curve.png")
    lines.append(f"  Feature importance:  output/feature_importance.png")
    lines.append(f"  This report:         output/evaluation_report.txt")

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[eval] Loading model and feature list...")
    model, feat_cols, meta = load_model_and_features()

    print("[eval] Loading 2024 test set...")
    raw, X, y = load_test_set(feat_cols)
    print(f"[eval] Test set: {len(X):,} matches, {len(feat_cols)} features")

    print("[eval] Running inference...")
    probs = model.predict_proba(X)[:, 1]

    # ── metrics ────────────────────────────────────────────────────────────────
    print("[eval] Computing metrics...")
    overall     = core_metrics(y, probs)
    thresh_rows = threshold_accuracy(y, probs)
    surf_rows   = surface_breakdown(raw, y, probs)
    lvl_rows    = level_breakdown(raw, y, probs)

    # ── charts ─────────────────────────────────────────────────────────────────
    print("[eval] Plotting calibration curve...")
    cal_bins = plot_calibration(y, probs)

    print("[eval] Plotting feature importances...")
    feat_rows = plot_feature_importance(model, feat_cols)

    # ── build and print report ─────────────────────────────────────────────────
    report = build_report(meta, overall, thresh_rows, surf_rows,
                          lvl_rows, cal_bins, feat_rows)

    # Tee to stdout and file simultaneously
    tee = io.StringIO()
    for line in report.splitlines():
        print(line)
        tee.write(line + "\n")

    REPORT_PATH.write_text(tee.getvalue())
    print(f"\n[eval] Report saved → {REPORT_PATH}")
    print(f"[eval] Calibration curve → {CAL_CURVE}")
    print(f"[eval] Feature importance → {FEAT_IMP}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
