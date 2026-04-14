"""
Recent form features.

Rolling window of last 15 matches (regardless of surface), weighted by
recency × opponent quality. Also reports surface-specific form within
that same window.

Weighting
---------
  recency_weight_i  = DECAY^i            (i=0 = most recent match)
  quality_weight_i  = opp_elo_i / 1500   (normalised around mean Elo)
  combined_weight_i = recency_weight_i * quality_weight_i

Weighted win% = Σ(combined_w * result) / Σ(combined_w)
A win vs a 2000-Elo opponent counts ~33% more than vs a 1500-Elo opponent.

Stats returned
--------------
  form_weighted_win_pct       combined recency+quality weighted win%
  form_surface_win_pct        same weighting but only matches on current surface
  form_top10_win_pct          win% vs opponents ranked ≤10 at match time
  form_top30_win_pct          win% vs opponents ranked ≤30 at match time
  form_streak                 signed streak: +3 = 3W in a row, -2 = 2L in a row
  form_avg_duration           mean match minutes over last 10 matches
  form_titles_6m              tournament titles in last 6 months
  form_finals_6m              finals appearances (W or L) in last 6 months
  form_n_matches              matches in the window (≤15)
  form_n_surface              surface matches in the window

Usage
-----
  from features.form import get_form_features
  feats = get_form_features(player_id, surface, date, elo_index, name_to_id)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly (python3 features/form.py) as well as package import
try:
    from features.elo import get_elo_at_date, DEFAULT_ELO
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features.elo import get_elo_at_date, DEFAULT_ELO

WINDOW      = 15
DECAY       = 0.94
DURATION_W  = 10   # match window for average duration
TITLES_DAYS = 182  # ~6 months

TOP_10_RANK = 10
TOP_30_RANK = 30


# ── helpers ──────────────────────────────────────────────────────────────────

def _recency_weights(n: int) -> np.ndarray:
    """weight_i = DECAY^i for i=0..n-1 (index 0 = most recent)."""
    return np.array([DECAY ** i for i in range(n)], dtype=float)


def _weighted_win_pct(results: list[int], weights: np.ndarray) -> float:
    """Σ(w * result) / Σ(w). Returns NaN if weight sum is 0."""
    w = weights[: len(results)]
    total = w.sum()
    if total == 0:
        return np.nan
    return float(np.dot(results, w) / total)


def _streak(results_newest_first: list[int]) -> int:
    """
    Current consecutive win (+) or loss (-) streak.
    results_newest_first[0] is the most recent match.
    Returns 0 if no matches.
    """
    if not results_newest_first:
        return 0
    sign   = 1 if results_newest_first[0] == 1 else -1
    count  = 0
    target = results_newest_first[0]
    for r in results_newest_first:
        if r == target:
            count += 1
        else:
            break
    return sign * count


# ── main function ─────────────────────────────────────────────────────────────

def get_form_features(
    matches: pd.DataFrame,
    player_id: int,
    surface: str,
    date: pd.Timestamp,
    elo_index: dict,
    name_to_id: dict[str, int] | None = None,
) -> dict:
    """
    Return form features for player_id as of date (exclusive).

    Parameters
    ----------
    matches     : processed matches DataFrame (from data_loader)
    player_id   : numeric Sackmann player ID
    surface     : surface of the upcoming match (for surface-specific form)
    date        : prediction date — all matches must be strictly before this
    elo_index   : built by build_elo_index() — used for opponent quality weights
    name_to_id  : player_name -> player_id — resolves opponent IDs for Elo lookups

    Returns
    -------
    dict of form stats (see module docstring)
    """
    surface = surface.lower()

    # ── pull last WINDOW matches ──────────────────────────────────────────────
    as_winner = matches[
        (matches["winner_id"] == player_id) & (matches["date"] < date)
    ].copy()
    as_winner["result"]    = 1
    as_winner["opp_id"]    = as_winner["loser_id"]
    as_winner["opp_rank"]  = as_winner["loser_rank"]
    as_winner["opp_name"]  = as_winner["loser_name"]

    as_loser = matches[
        (matches["loser_id"] == player_id) & (matches["date"] < date)
    ].copy()
    as_loser["result"]    = 0
    as_loser["opp_id"]    = as_loser["winner_id"]
    as_loser["opp_rank"]  = as_loser["winner_rank"]
    as_loser["opp_name"]  = as_loser["winner_name"]

    combined = (
        pd.concat([as_winner, as_loser], ignore_index=True)
        .sort_values("date", ascending=False)   # newest first
        .head(WINDOW)
        .reset_index(drop=True)
    )

    n = len(combined)
    if n == 0:
        return _empty_form()

    # ── opponent Elo lookup ───────────────────────────────────────────────────
    opp_elos = np.full(n, DEFAULT_ELO, dtype=float)
    for i, row in enumerate(combined.itertuples(index=False)):
        opp_id = int(row.opp_id) if pd.notna(row.opp_id) else None
        if opp_id is None and name_to_id:
            opp_id = name_to_id.get(str(row.opp_name))
        if opp_id is not None:
            opp_surf = str(row.surface).lower()
            opp_elos[i] = get_elo_at_date(elo_index, opp_id, opp_surf, row.date)

    quality_weights = opp_elos / DEFAULT_ELO
    recency         = _recency_weights(n)
    combined_w      = recency * quality_weights

    results = combined["result"].tolist()

    # ── overall weighted win% ─────────────────────────────────────────────────
    form_weighted_win_pct = _weighted_win_pct(results, combined_w)

    # ── surface-specific weighted win% ───────────────────────────────────────
    surf_mask = combined["surface"].str.lower() == surface
    surf_idx  = combined[surf_mask].index.tolist()
    if surf_idx:
        # Reset recency within the surface subset (index 0 = most recent surface match)
        surf_results = [results[i] for i in surf_idx]
        surf_recency = _recency_weights(len(surf_idx))
        surf_quality = quality_weights[surf_idx]
        surf_w       = surf_recency * surf_quality
        form_surface_win_pct = _weighted_win_pct(surf_results, surf_w)
        form_n_surface = len(surf_idx)
    else:
        form_surface_win_pct = np.nan
        form_n_surface = 0

    # ── top-10 / top-30 win% (unweighted — sample is already tiny) ───────────
    def _tier_win_pct(max_rank: int) -> float:
        mask = combined["opp_rank"] <= max_rank
        if mask.sum() == 0:
            return np.nan
        return float(combined.loc[mask, "result"].mean())

    form_top10_win_pct = _tier_win_pct(TOP_10_RANK)
    form_top30_win_pct = _tier_win_pct(TOP_30_RANK)

    # ── streak ────────────────────────────────────────────────────────────────
    form_streak = _streak(results)   # results already newest-first

    # ── average match duration (last DURATION_W matches) ─────────────────────
    dur_window = combined.head(DURATION_W)
    durations  = pd.to_numeric(dur_window["minutes"], errors="coerce").dropna()
    form_avg_duration = float(durations.mean()) if len(durations) > 0 else np.nan

    # ── titles and finals in last 6 months ───────────────────────────────────
    cutoff_6m = date - pd.Timedelta(days=TITLES_DAYS)
    recent_finals = matches[
        ((matches["winner_id"] == player_id) | (matches["loser_id"] == player_id))
        & (matches["round"] == "F")
        & (matches["date"] >= cutoff_6m)
        & (matches["date"] <  date)
    ]
    form_titles_6m = int((recent_finals["winner_id"] == player_id).sum())
    form_finals_6m = len(recent_finals)

    return {
        "form_weighted_win_pct":  form_weighted_win_pct,
        "form_surface_win_pct":   form_surface_win_pct,
        "form_top10_win_pct":     form_top10_win_pct,
        "form_top30_win_pct":     form_top30_win_pct,
        "form_streak":            form_streak,
        "form_avg_duration":      form_avg_duration,
        "form_titles_6m":         form_titles_6m,
        "form_finals_6m":         form_finals_6m,
        "form_n_matches":         n,
        "form_n_surface":         form_n_surface,
    }


def _empty_form() -> dict:
    return {
        "form_weighted_win_pct":  np.nan,
        "form_surface_win_pct":   np.nan,
        "form_top10_win_pct":     np.nan,
        "form_top30_win_pct":     np.nan,
        "form_streak":            0,
        "form_avg_duration":      np.nan,
        "form_titles_6m":         0,
        "form_finals_6m":         0,
        "form_n_matches":         0,
        "form_n_surface":         0,
    }


# ── CLI / sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.data_loader import load_processed
    from features.elo import load_elo_history, build_elo_index

    matches = load_processed()
    elo_history = load_elo_history()
    elo_index   = build_elo_index(elo_history)

    name_to_id: dict[str, int] = {}
    for r in matches[["winner_name","winner_id"]].itertuples(index=False):
        name_to_id[r.winner_name] = int(r.winner_id)
    for r in matches[["loser_name","loser_id"]].itertuples(index=False):
        name_to_id[r.loser_name] = int(r.loser_id)

    ALCARAZ_ID = 207989
    RUNE_ID    = 208029
    PRED_DATE  = pd.Timestamp("2023-07-03")   # just before Wimbledon 2023

    print("=" * 60)
    print(f"Carlos Alcaraz  — form as of {PRED_DATE.date()}")
    print("=" * 60)
    feats_a = get_form_features(matches, ALCARAZ_ID, "grass", PRED_DATE, elo_index, name_to_id)
    for k, v in feats_a.items():
        print(f"  {k:<30} {v}")

    print()
    print("=" * 60)
    print(f"Holger Rune     — form as of {PRED_DATE.date()}")
    print("=" * 60)
    feats_r = get_form_features(matches, RUNE_ID, "grass", PRED_DATE, elo_index, name_to_id)
    for k, v in feats_r.items():
        print(f"  {k:<30} {v}")

    print()
    print(f"Alcaraz weighted_win_pct={feats_a['form_weighted_win_pct']:.3f}  "
          f"Rune weighted_win_pct={feats_r['form_weighted_win_pct']:.3f}")
    print(f"Alcaraz streak={feats_a['form_streak']}  "
          f"Rune streak={feats_r['form_streak']}")
    print(f"Alcaraz titles_6m={feats_a['form_titles_6m']}  "
          f"Rune titles_6m={feats_r['form_titles_6m']}")
