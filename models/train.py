"""
Model training: XGBoost win probability classifier with isotonic calibration.

Pipeline
--------
1. build_training_dataset()   iterate all historical matches through pipeline.py,
                              frame player A as the higher-ranked player,
                              cache to data/processed/train_features.csv
2. prepare_splits()           train 2015-2021 / val 2022-2023 / hold 2024
3. run_grid_search()          time-respecting grid search (train→val, never touch 2024)
4. train_final()              refit best params on train+val combined, calibrate
5. save artefacts             models/saved/xgb_calibrated.pkl + feature_list.json

Framing convention
------------------
Player A is ALWAYS the higher-ranked player (lower rank number).
If ranks are unavailable for both, higher pre-match surface Elo is used.
Label = 1 if player A won, 0 if player B won.
This gives ~50/50 class balance (favourites win ~60% so slight imbalance is real).

Usage
-----
  # First run: builds feature dataset (~30-90 min depending on hardware), then trains
  python -m models.train

  # Rebuild feature cache (e.g. after pipeline changes)
  python -m models.train --rebuild

  # Quick smoke test on 2000 matches (sanity check imports & shapes)
  python -m models.train --sample 2000

  # Skip grid search, use fixed params
  python -m models.train --no-grid
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
PROCESSED    = ROOT / "data" / "processed"
MODEL_DIR    = ROOT / "models" / "saved"
FEATURES_CSV = PROCESSED / "train_features.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── feature columns ────────────────────────────────────────────────────────────
# Defined once here — exactly what gets written to feature_list.json
# and loaded by predict.py. Categorical-only cols (style, hand, h2h_last3)
# are excluded; availability/metadata flags excluded.

FEATURE_COLS: list[str] = [

    # ── Tier 1: Elo ─────────────────────────────────────────────────────────
    "elo_diff",

    # ── Tier 1: Serve/return — player A ─────────────────────────────────────
    "a_hold_pct", "a_break_pct",
    "a_first_serve_pct", "a_first_srv_won_pct", "a_second_srv_won_pct",
    "a_first_ret_won_pct", "a_second_ret_won_pct",
    "a_bp_save_pct", "a_bp_conv_pct",

    # ── Tier 1: Serve/return — player B ─────────────────────────────────────
    "b_hold_pct", "b_break_pct",
    "b_first_serve_pct", "b_first_srv_won_pct", "b_second_srv_won_pct",
    "b_first_ret_won_pct", "b_second_ret_won_pct",
    "b_bp_save_pct", "b_bp_conv_pct",

    # ── Tier 1: Form — player A ──────────────────────────────────────────────
    "a_form_weighted_win_pct", "a_form_surface_win_pct",
    "a_form_top10_win_pct", "a_form_top30_win_pct",
    "a_form_streak", "a_form_titles_6m", "a_form_finals_6m",

    # ── Tier 1: Form — player B ──────────────────────────────────────────────
    "b_form_weighted_win_pct", "b_form_surface_win_pct",
    "b_form_top10_win_pct", "b_form_top30_win_pct",
    "b_form_streak", "b_form_titles_6m", "b_form_finals_6m",

    # ── Tier 1: H2H ──────────────────────────────────────────────────────────
    "h2h_win_pct_overall", "h2h_win_pct_surface",
    "h2h_weighted_win_pct", "h2h_momentum", "h2h_avg_sets", "h2h_n_overall",

    # ── Tier 1: Match format ─────────────────────────────────────────────────
    "best_of_5",

    # ── Tier 2: Court speed ──────────────────────────────────────────────────
    "court_speed_index",

    # ── Tier 2: Rally profile ────────────────────────────────────────────────
    "rally_0_4_delta", "rally_5_8_delta", "rally_9plus_delta",

    # ── Tier 2: Ace rates ────────────────────────────────────────────────────
    "a_ace_rate", "b_ace_rate", "ace_rate_delta",

    # ── Tier 2: Hand matchup ─────────────────────────────────────────────────
    "is_lefty_righty",
    "a_win_pct_vs_lefty", "a_win_pct_vs_righty",
    "b_win_pct_vs_lefty", "b_win_pct_vs_righty",

    # ── Tier 2: Venue history ─────────────────────────────────────────────────
    "a_venue_win_pct", "a_venue_round_avg", "a_venue_titles",
    "b_venue_win_pct", "b_venue_round_avg", "b_venue_titles",

    # ── Tier 2: Ranking ──────────────────────────────────────────────────────
    "a_rank_current", "b_rank_current",
    "a_trajectory_3m", "b_trajectory_3m",
    "a_trajectory_6m", "b_trajectory_6m",

    # ── Tier 2: Defending pressure ───────────────────────────────────────────
    "a_defending_pressure", "b_defending_pressure",

    # ── Tier 2: Fatigue / scheduling ─────────────────────────────────────────
    "a_days_rest", "b_days_rest",
    "a_prev_match_minutes", "b_prev_match_minutes",
    "a_matches_last_14_days", "b_matches_last_14_days",
    "a_timezone_shift", "b_timezone_shift",

    # ── Tier 2: Tournament / round context ───────────────────────────────────
    "tourney_level_ord", "is_grand_slam", "is_masters", "is_challenger",
    "round_stage", "is_late_round",

    # ── Tier 3: Injury proxies ────────────────────────────────────────────────
    "a_injury_flag", "b_injury_flag",
    "a_injury_severity", "b_injury_severity",
    "a_duration_spike", "b_duration_spike",
    "a_multi_retirement", "b_multi_retirement",

    # ── Market (optional — NaN when no odds available) ────────────────────────
    "opening_implied_a", "movement_magnitude", "sharp_flag",
]

LABEL_COL = "label"   # 1 = player A (higher ranked) won

# ── Elo filter floor ───────────────────────────────────────────────────────────
# Drop rows where BOTH players have default Elo (neither has any prior match).
# A single default value (1500.0 exactly) is fine — one new player vs established one.
ELO_DEFAULT = 1500.0

# ── grid search space ─────────────────────────────────────────────────────────

PARAM_GRID = [
    {"max_depth": d, "learning_rate": lr, "n_estimators": n, "min_child_weight": mcw}
    for d   in [3, 5, 7]
    for lr  in [0.05, 0.1]
    for n   in [200, 500]
    for mcw in [1, 3]
]  # 3 × 2 × 2 × 2 = 24 combinations

FIXED_PARAMS = {
    "max_depth": 5, "learning_rate": 0.05,
    "n_estimators": 500, "min_child_weight": 3,
}

XGB_BASE = {
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


# ── dataset builder ───────────────────────────────────────────────────────────

def _determine_player_a(row: pd.Series, elo_index: dict) -> tuple[bool, bool]:
    """
    Return (swap, label):
      swap  = True  → we need to swap A/B so player A is the higher-ranked player
      label = 1 if the winner ends up as player A after the swap

    Winner in Sackmann = always the player in winner_* columns.
    Higher ranked = lower rank number (rank 5 > rank 50).
    Tiebreak uses surface Elo when ranks are missing.
    """
    from features.elo import get_elo_at_date

    w_rank = row.get("winner_rank")
    l_rank = row.get("loser_rank")
    date   = row["date"]
    surf   = str(row.get("surface", "hard")).lower()
    w_id   = int(row["winner_id"])
    l_id   = int(row["loser_id"])

    w_rank = float(w_rank) if pd.notna(w_rank) else np.nan
    l_rank = float(l_rank) if pd.notna(l_rank) else np.nan

    if not np.isnan(w_rank) and not np.isnan(l_rank):
        # lower rank number = better player → winner_rank < loser_rank means winner is higher ranked
        winner_is_higher = w_rank <= l_rank
    elif not np.isnan(w_rank):
        winner_is_higher = True   # ranked player vs unranked → ranked is "higher"
    elif not np.isnan(l_rank):
        winner_is_higher = False
    else:
        # Neither ranked — fall back to Elo
        w_elo = get_elo_at_date(elo_index, w_id, surf, date)
        l_elo = get_elo_at_date(elo_index, l_id, surf, date)
        winner_is_higher = w_elo >= l_elo

    # If winner is higher-ranked: no swap needed, label = 1 (player A = winner = won)
    # If loser is higher-ranked:  swap so player A = loser,  label = 0 (player A = loser = lost)
    if winner_is_higher:
        return False, 1   # no swap; A=winner won
    else:
        return True, 0    # swap;    A=loser  lost


def _build_player_match_index(matches: pd.DataFrame) -> dict[int, list[int]]:
    """
    Pre-build a player_id → [integer row positions] mapping in one O(n) pass.
    Each entry is the sorted list of positional indices (iloc) in `matches`
    where that player appears as winner or loser.
    """
    from collections import defaultdict
    idx: dict[int, list[int]] = defaultdict(list)
    winner_ids = matches["winner_id"].to_numpy()
    loser_ids  = matches["loser_id"].to_numpy()
    for pos, (wid, lid) in enumerate(zip(winner_ids, loser_ids)):
        idx[int(wid)].append(pos)
        idx[int(lid)].append(pos)
    return dict(idx)


# ── multiprocessing worker ────────────────────────────────────────────────────
# These must be module-level so they're picklable under the 'spawn' start method.

_worker_ctx        = None   # PipelineContext — loaded once per worker
_worker_matches    = None   # full sorted matches DataFrame
_worker_player_idx = None   # player_id → [row positions]


def _worker_init(root_str: str) -> None:
    """
    Initialiser for each worker process.
    Loads PipelineContext from cached disk files (elo_history.csv,
    serve_return_stats.csv) — fast because nothing is recomputed.
    Suppresses per-worker print noise.
    """
    import io, contextlib
    global _worker_ctx, _worker_matches, _worker_player_idx

    sys.path.insert(0, root_str)

    # Silence the [pipeline] loading messages from workers
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from features.pipeline import PipelineContext
        _worker_ctx = PipelineContext.load()

    _worker_matches = _worker_ctx.matches.sort_values("date").reset_index(drop=True)
    _worker_player_idx = _build_player_match_index(_worker_matches)


def _process_chunk(positions: list[int]) -> tuple[list[dict], int]:
    """
    Process a chunk of match row positions.
    Returns (feature_rows, n_skipped).
    Each worker has its own PipelineContext — no shared mutable state.
    """
    from features.pipeline import build_feature_row

    rows: list[dict] = []
    skipped = 0

    for pos in positions:
        match = _worker_matches.iloc[pos]
        try:
            swap, label = _determine_player_a(match, _worker_ctx.elo_index)

            if swap:
                a_id   = int(match["loser_id"])
                b_id   = int(match["winner_id"])
                a_name = str(match.get("loser_name", ""))
                b_name = str(match.get("winner_name", ""))
            else:
                a_id   = int(match["winner_id"])
                b_id   = int(match["loser_id"])
                a_name = str(match.get("winner_name", ""))
                b_name = str(match.get("loser_name", ""))

            ab_positions = sorted(
                set(_worker_player_idx.get(a_id, []))
                | set(_worker_player_idx.get(b_id, []))
            )
            _worker_ctx.matches = _worker_matches.iloc[ab_positions]

            feat = build_feature_row(
                _worker_ctx,
                player_a_id   = a_id,
                player_b_id   = b_id,
                player_a_name = a_name,
                player_b_name = b_name,
                surface       = str(match.get("surface", "hard")),
                tournament    = str(match.get("tourney_name", "")),
                match_date    = pd.Timestamp(match["date"]),
                best_of       = int(match.get("best_of", 3)),
                round_str     = str(match.get("round", "R32")),
                tourney_level = str(match.get("tourney_level", "A")),
                tour          = str(match.get("tour", "atp")),
            )
            feat[LABEL_COL] = label
            rows.append(feat)

        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"  [worker-{os.getpid()}] pos {pos} skipped: {exc}",
                      flush=True)
        finally:
            _worker_ctx.matches = _worker_matches

    return rows, skipped


def build_training_dataset(
    ctx,
    sample_n:   int | None = None,
    n_workers:  int = min(8, mp.cpu_count()),
    output_path: Path = FEATURES_CSV,
) -> pd.DataFrame:
    """
    Build the training feature dataset using multiprocessing.

    Each worker loads PipelineContext independently from cached disk files,
    builds its own player-match index, and processes its assigned chunk.
    Only a list of integer positions is sent to each worker; nothing large
    (DataFrames, dicts) is pickled across the process boundary.

    Parameters
    ----------
    ctx        : PipelineContext already loaded in the main process
                 (used only to get the sorted matches — workers reload independently)
    sample_n   : if set, only process the first N matches (smoke test)
    n_workers  : number of parallel workers (default: min(8, cpu_count))
    output_path: cache destination
    """
    full_matches = ctx.matches.sort_values("date").reset_index(drop=True)

    if sample_n is not None:
        full_matches = full_matches.head(sample_n)
        print(f"[train] Sample mode: using {len(full_matches):,} matches")

    n_total    = len(full_matches)
    all_pos    = list(range(n_total))

    # Split positions into evenly-sized chunks — one per worker
    chunk_size = max(1, n_total // n_workers)
    chunks     = [all_pos[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    actual_workers = min(n_workers, len(chunks))

    print(f"[train] Building {n_total:,} feature rows "
          f"across {actual_workers} workers "
          f"(~{chunk_size:,} rows each)...", flush=True)
    t0 = time.time()

    ctx_mp = mp.get_context("spawn")
    with ctx_mp.Pool(
        processes   = actual_workers,
        initializer = _worker_init,
        initargs    = (str(ROOT),),
    ) as pool:
        results = pool.map(_process_chunk, chunks)

    elapsed = time.time() - t0

    # Merge results preserving chronological order (chunks are already ordered)
    all_rows: list[dict] = []
    total_skipped = 0
    for rows, skipped in results:
        all_rows.extend(rows)
        total_skipped += skipped

    rate = len(all_rows) / elapsed if elapsed > 0 else 0
    print(f"[train] Done: {len(all_rows):,} rows built, "
          f"{total_skipped} skipped  "
          f"({elapsed/60:.1f} min  {rate:.0f} rows/s)", flush=True)

    df = pd.DataFrame(all_rows)
    # Re-sort by date — chunks are ordered but parallel completion may interleave
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[train] Feature dataset saved → {output_path}", flush=True)
    return df


# ── split helper ──────────────────────────────────────────────────────────────

def prepare_splits(
    df: pd.DataFrame,
    rolling: bool = False,
    rolling_train_years: int = 5,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """
    Temporal splits.

    Fixed mode (default):
      train : 2015-01-01 … 2021-12-31
      val   : 2022-01-01 … 2023-12-31
      test  : 2024-01-01 … (HELD OUT)

    Rolling mode (--rolling flag):
      Shifts the window each time this runs so the model always trains on
      the most recent data.  Using current_year C:
        test  : C           (held out)
        val   : C-1
        train : (C - rolling_train_years - 1) … (C-2)   [default 5 years]

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    import datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if rolling:
        C = datetime.date.today().year
        test_year  = C
        val_year   = C - 1
        train_end  = C - 2
        train_start = C - 2 - rolling_train_years + 1   # inclusive
        print(f"[train] Rolling split: train {train_start}–{train_end}  "
              f"val {val_year}  test {test_year} (held out)")
        train_mask = (df["date"].dt.year >= train_start) & (df["date"].dt.year <= train_end)
        val_mask   = df["date"].dt.year == val_year
        test_mask  = df["date"].dt.year >= test_year
    else:
        train_mask = df["date"].dt.year <= 2021
        val_mask   = (df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2023)
        test_mask  = df["date"].dt.year >= 2024

    # Resolve feature columns actually present
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    def _split(mask):
        sub = df[mask].copy()
        X = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
        y = sub[LABEL_COL].astype(int)
        return X, y

    X_tr, y_tr   = _split(train_mask)
    X_val, y_val = _split(val_mask)
    X_te, y_te   = _split(test_mask)

    print(f"[train] Split sizes:")
    print(f"  train  {len(X_tr):>7,}  ({y_tr.mean():.3f} base rate)")
    print(f"  val    {len(X_val):>7,}  ({y_val.mean():.3f} base rate)")
    print(f"  test   {len(X_te):>7,}  (HELD OUT — not touched until evaluate.py)")

    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ── Elo filter ────────────────────────────────────────────────────────────────

def drop_no_elo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where elo_diff is NaN (both players at exact default —
    no surface Elo history for either). A single default is fine.
    """
    if "elo_diff" not in df.columns:
        return df
    before = len(df)
    df = df[df["elo_diff"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[train] Dropped {dropped:,} rows with missing Elo diff "
              f"({dropped/before*100:.2f}%)")
    return df


# ── grid search ───────────────────────────────────────────────────────────────

def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    feat_cols: list[str],
) -> dict:
    """
    Manual grid search over PARAM_GRID evaluated on the held-out val set.
    Returns the best parameter dict.
    """
    print(f"\n[train] Grid search: {len(PARAM_GRID)} combinations "
          f"(train={len(X_train):,}, val={len(X_val):,})")

    best_params  = None
    best_brier   = np.inf
    results: list[dict] = []

    for idx, params in enumerate(PARAM_GRID):
        p = {**XGB_BASE, **params}
        model = xgb.XGBClassifier(**p, verbosity=0)

        # Fit on train, evaluate on val — no calibration during grid search
        # (calibration needs its own hold-out; here we just compare raw scores)
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]

        bs  = brier_score_loss(y_val, prob_val)
        ll  = log_loss(y_val, prob_val)

        results.append({**params, "brier": bs, "logloss": ll})

        if bs < best_brier:
            best_brier  = bs
            best_params = params

        print(f"  [{idx+1:>2}/{len(PARAM_GRID)}]  "
              f"depth={params['max_depth']}  "
              f"lr={params['learning_rate']}  "
              f"trees={params['n_estimators']:>3}  "
              f"mcw={params['min_child_weight']}  "
              f"→ brier={bs:.5f}  ll={ll:.5f}"
              + ("  ← best" if params == best_params else ""))

    print(f"\n[train] Best params: {best_params}  (val brier={best_brier:.5f})")
    return best_params


# ── final training ────────────────────────────────────────────────────────────

def train_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    params:  dict,
    feat_cols: list[str],
) -> CalibratedClassifierCV:
    """
    Refit best params on train+val combined, then wrap with isotonic calibration.
    Uses internal 5-fold CV for calibration (no data leakage from test set).
    """
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    print(f"\n[train] Final fit on train+val ({len(X_combined):,} rows) "
          f"with params: {params}")

    base = xgb.XGBClassifier(**{**XGB_BASE, **params}, verbosity=0)

    # cv=5 means CalibratedClassifierCV fits 5 base models on 4/5 of the data
    # and calibrates on the held-out 1/5 — all within train+val only.
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_combined, y_combined)

    return calibrated


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate(
    model: CalibratedClassifierCV,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "val",
) -> dict:
    prob = model.predict_proba(X)[:, 1]
    bs   = brier_score_loss(y, prob)
    ll   = log_loss(y, prob)
    print(f"[{label}] Brier score = {bs:.5f}   log-loss = {ll:.5f}")
    return {"brier": bs, "logloss": ll}


def print_feature_importances(
    model: CalibratedClassifierCV,
    feat_cols: list[str],
    top_n: int = 20,
) -> None:
    """
    Extract average feature importances from XGBoost base learners inside
    the CalibratedClassifierCV ensemble.
    """
    importances = np.zeros(len(feat_cols))
    n_estimators = 0

    for estimator in model.calibrated_classifiers_:
        base = estimator.estimator
        if hasattr(base, "feature_importances_"):
            importances += base.feature_importances_
            n_estimators += 1

    if n_estimators == 0:
        print("[train] Could not extract feature importances.")
        return

    importances /= n_estimators
    ranked = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)

    print(f"\n[train] Top {top_n} feature importances (avg over {n_estimators} calibrators):")
    for rank, (name, imp) in enumerate(ranked[:top_n], 1):
        bar = "█" * int(imp * 400)
        print(f"  {rank:>2}. {name:<40}  {imp:.4f}  {bar}")


# ── save artefacts ────────────────────────────────────────────────────────────

def save_artefacts(
    model:     CalibratedClassifierCV,
    feat_cols: list[str],
    metrics:   dict,
) -> None:
    model_path   = MODEL_DIR / "xgb_calibrated.pkl"
    feature_path = MODEL_DIR / "feature_list.json"

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols}, f)

    with open(feature_path, "w") as f:
        json.dump({
            "feature_cols": feat_cols,
            "n_features":   len(feat_cols),
            "val_brier":    round(metrics.get("brier", np.nan), 6),
            "val_logloss":  round(metrics.get("logloss", np.nan), 6),
        }, f, indent=2)

    print(f"\n[train] Model    saved → {model_path}")
    print(f"[train] Features saved → {feature_path}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main(
    rebuild:   bool = False,
    sample_n:  int | None = None,
    no_grid:   bool = False,
    n_workers: int = min(8, mp.cpu_count()),
    rolling:   bool = False,
) -> None:
    sys.path.insert(0, str(ROOT))
    from features.pipeline import PipelineContext

    # ── 1. Load or build feature dataset ──────────────────────────────────────
    if not FEATURES_CSV.exists() or rebuild or sample_n is not None:
        print("[train] Loading pipeline context...")
        ctx = PipelineContext.load()
        df  = build_training_dataset(ctx, sample_n=sample_n, n_workers=n_workers)
    else:
        print(f"[train] Loading cached feature dataset from {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        print(f"[train] Loaded {len(df):,} rows")

    # ── 2. Filter unusable rows ────────────────────────────────────────────────
    df = drop_no_elo_rows(df)

    # ── 3. Resolve feature columns present in this dataset ────────────────────
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"[train] {len(missing_cols)} feature cols not in dataset "
              f"(will be treated as NaN): {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

    print(f"[train] Using {len(feat_cols)} features, {len(df):,} rows")

    # ── 4. Temporal split ──────────────────────────────────────────────────────
    X_tr, y_tr, X_val, y_val, X_te, y_te = prepare_splits(df, rolling=rolling)

    if len(X_tr) == 0:
        print("[train] ERROR: empty training set. "
              "Check date range in the feature CSV.")
        sys.exit(1)

    if len(X_val) == 0:
        print("[train] WARNING: validation set is empty "
              "(all rows are before 2022 — smoke test mode).")
        print("[train] Training on available rows; skipping grid search and val metrics.")
        feat_cols = [c for c in feat_cols if c in X_tr.columns]
        X_tr = X_tr[feat_cols]
        base = xgb.XGBClassifier(**{**XGB_BASE, **FIXED_PARAMS}, verbosity=0)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
        calibrated.fit(X_tr, y_tr)
        print_feature_importances(calibrated, feat_cols)
        save_artefacts(calibrated, feat_cols, {})
        return

    # Re-filter feat_cols to what's actually in the split DataFrames
    feat_cols = [c for c in feat_cols if c in X_tr.columns]

    X_tr  = X_tr[feat_cols]
    X_val = X_val[feat_cols]
    X_te  = X_te[feat_cols]

    # ── 5. Hyperparameter selection ────────────────────────────────────────────
    if no_grid:
        best_params = FIXED_PARAMS
        print(f"[train] Grid search skipped — using fixed params: {best_params}")
    else:
        best_params = run_grid_search(X_tr, y_tr, X_val, y_val, feat_cols)

    # ── 6. Final model: refit on train+val with isotonic calibration ───────────
    final_model = train_final(X_tr, y_tr, X_val, y_val, best_params, feat_cols)

    # ── 7. Report val metrics (test set is NEVER touched here) ─────────────────
    print("\n── Validation metrics (train+val model evaluated on val portion) ──")
    val_model_only = xgb.XGBClassifier(**{**XGB_BASE, **best_params}, verbosity=0)
    val_model_only.fit(X_tr, y_tr)
    prob_val_uncal = val_model_only.predict_proba(X_val)[:, 1]
    bs_uncal  = brier_score_loss(y_val, prob_val_uncal)
    ll_uncal  = log_loss(y_val, prob_val_uncal)
    print(f"  Uncalibrated  — Brier: {bs_uncal:.5f}   log-loss: {ll_uncal:.5f}")

    prob_val_cal = final_model.predict_proba(X_val)[:, 1]
    bs_cal  = brier_score_loss(y_val, prob_val_cal)
    ll_cal  = log_loss(y_val, prob_val_cal)
    print(f"  Calibrated    — Brier: {bs_cal:.5f}   log-loss: {ll_cal:.5f}")

    val_metrics = {"brier": bs_cal, "logloss": ll_cal}

    print(f"\n  Test set ({len(X_te):,} rows) held out — run evaluate.py for final numbers.")

    # ── 8. Feature importances ──────────────────────────────────────────────────
    print_feature_importances(final_model, feat_cols)

    # ── 9. Save ────────────────────────────────────────────────────────────────
    save_artefacts(final_model, feat_cols, val_metrics)


if __name__ == "__main__":
    # Must be set before any Pool is created; 'spawn' is safe on macOS/Linux.
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train tennis win probability model")
    parser.add_argument("--rebuild",   action="store_true",
                        help="Rebuild feature cache even if it exists")
    parser.add_argument("--sample",    type=int, default=None,
                        help="Only process N matches (smoke test)")
    parser.add_argument("--no-grid",   action="store_true",
                        help="Skip grid search, use fixed params")
    parser.add_argument("--rolling",   action="store_true",
                        help="Use rolling 5-year window split instead of fixed 2015-2024 split")
    parser.add_argument("--workers",   type=int, default=min(8, mp.cpu_count()),
                        help=f"Parallel workers for feature build (default: {min(8, mp.cpu_count())})")
    args = parser.parse_args()

    main(rebuild=args.rebuild, sample_n=args.sample, no_grid=args.no_grid,
         n_workers=args.workers, rolling=args.rolling)
