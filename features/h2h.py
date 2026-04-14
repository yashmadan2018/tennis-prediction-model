"""
Head-to-head feature extraction.

Rules
-----
  - Filter to last 4 years from the prediction date (never include the match itself)
  - Surface-specific H2H computed separately from overall H2H
  - Minimum 3 matches required before H2H is used as a feature
  - If surface H2H < threshold but overall >= threshold: use overall, set
    surface_h2h_unavailable=True so pipeline knows to weight it lower
  - If overall also < threshold: h2h_available=False, pipeline falls back to
    Elo + serve/return entirely

Recency weighting
-----------------
Within the 4-year window, matches are weighted by recency using the same
exponential decay as serve_return.py: weight_i = DECAY^i (i=0 = most recent).
Weighted win% is more robust than raw win% for small samples.

Stats returned
--------------
  h2h_wins_overall          player_a wins in 4yr window
  h2h_losses_overall        player_a losses in 4yr window
  h2h_wins_surface          wins on same surface
  h2h_losses_surface        losses on same surface
  h2h_win_pct_overall       raw win% overall (NaN if below threshold)
  h2h_win_pct_surface       raw win% on surface (NaN if below threshold)
  h2h_weighted_win_pct      recency-weighted win% (uses surface if available, else overall)
  h2h_last3                 W/L string, most recent first (e.g. 'LLW')
  h2h_avg_sets              average sets played across H2H matches (competitiveness proxy)
  h2h_momentum              1 if player_a won the most recent meeting, else 0
  h2h_n_overall             total H2H matches in window
  h2h_n_surface             H2H matches on same surface in window
  h2h_available             True if overall >= MIN_MATCHES
  h2h_surface_available     True if surface >= MIN_MATCHES
  surface_h2h_unavailable   True when surface < threshold but overall is used as fallback
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

H2H_YEARS = 4
H2H_MIN_MATCHES = 3
DECAY = 0.94          # matches serve_return.py recency decay


# ── set counting ─────────────────────────────────────────────────────────────

_SET_RE = re.compile(r"\d+-\d+(?:\(\d+\))?")
_SUFFIX_RE = re.compile(r"\s*(RET|W/O|WO|DEF|ABD|Retired|Walkover)\b.*", re.IGNORECASE)


def _count_sets(score: str) -> float:
    """Parse a Sackmann score string and return number of sets played."""
    if not isinstance(score, str) or not score.strip():
        return np.nan
    clean = _SUFFIX_RE.sub("", score.strip())
    sets = _SET_RE.findall(clean)
    return float(len(sets)) if sets else np.nan


# ── recency-weighted win% ────────────────────────────────────────────────────

def _weighted_win_pct(results: list[int]) -> float:
    """
    Compute recency-weighted win%.
    results: list of 1/0 ordered newest-first.
    weight_i = DECAY^i
    """
    if not results:
        return np.nan
    weights = np.array([DECAY ** i for i in range(len(results))])
    return float(np.dot(results, weights) / weights.sum())


# ── main function ─────────────────────────────────────────────────────────────

def get_h2h_features(
    matches: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    surface: str,
    date: pd.Timestamp,
) -> dict:
    """
    Return H2H features for player_a vs player_b as of date (exclusive).

    Parameters
    ----------
    matches     : processed matches DataFrame (from data_loader)
    player_a_id : numeric Sackmann ID for player A
    player_b_id : numeric Sackmann ID for player B
    surface     : 'hard' | 'clay' | 'grass'
    date        : prediction date — all H2H matches must be strictly before this

    Returns
    -------
    dict with all H2H stats (see module docstring for field list)
    """
    surface = surface.lower()
    cutoff = date - pd.DateOffset(years=H2H_YEARS)

    # ── pull all qualifying H2H matches ──────────────────────────────────────
    a_beats_b = matches[
        (matches["winner_id"] == player_a_id)
        & (matches["loser_id"]  == player_b_id)
        & (matches["date"] >= cutoff)
        & (matches["date"] <  date)
    ].copy()
    a_beats_b["result_a"] = 1

    b_beats_a = matches[
        (matches["winner_id"] == player_b_id)
        & (matches["loser_id"]  == player_a_id)
        & (matches["date"] >= cutoff)
        & (matches["date"] <  date)
    ].copy()
    b_beats_a["result_a"] = 0

    all_h2h = (
        pd.concat([a_beats_b, b_beats_a], ignore_index=True)
        .sort_values("date", ascending=False)   # newest first
        .reset_index(drop=True)
    )

    # ── surface subset ────────────────────────────────────────────────────────
    surf_h2h = all_h2h[all_h2h["surface"] == surface].reset_index(drop=True)

    n_overall = len(all_h2h)
    n_surface = len(surf_h2h)

    # ── availability flags ────────────────────────────────────────────────────
    h2h_available         = n_overall >= H2H_MIN_MATCHES
    h2h_surface_available = n_surface >= H2H_MIN_MATCHES
    surface_h2h_unavailable = h2h_available and not h2h_surface_available

    # ── base result ──────────────────────────────────────────────────────────
    result = {
        "h2h_available":          h2h_available,
        "h2h_surface_available":  h2h_surface_available,
        "surface_h2h_unavailable": surface_h2h_unavailable,
        "h2h_n_overall":          n_overall,
        "h2h_n_surface":          n_surface,
    }

    # ── overall stats ─────────────────────────────────────────────────────────
    if n_overall == 0:
        result.update({
            "h2h_wins_overall":         0,
            "h2h_losses_overall":       0,
            "h2h_win_pct_overall":      np.nan,
            "h2h_weighted_win_pct":     np.nan,
            "h2h_last3":                "",
            "h2h_avg_sets":             np.nan,
            "h2h_momentum":             np.nan,
            "h2h_wins_surface":         0,
            "h2h_losses_surface":       0,
            "h2h_win_pct_surface":      np.nan,
        })
        return result

    wins_overall   = int(all_h2h["result_a"].sum())
    losses_overall = n_overall - wins_overall

    win_pct_overall = (wins_overall / n_overall) if h2h_available else np.nan

    # Recency-weighted win% — uses surface subset if available, overall otherwise
    active_set = surf_h2h if h2h_surface_available else all_h2h
    weighted_win_pct = _weighted_win_pct(active_set["result_a"].tolist())

    # Last 3 results (newest first, W/L string)
    last3_results = all_h2h["result_a"].head(3).tolist()
    last3_str = "".join("W" if r == 1 else "L" for r in last3_results)

    # Average sets across all H2H matches
    set_counts = all_h2h["score"].apply(_count_sets).dropna()
    avg_sets = float(set_counts.mean()) if len(set_counts) > 0 else np.nan

    # Momentum: did player_a win the most recent meeting?
    momentum = int(all_h2h.iloc[0]["result_a"])

    result.update({
        "h2h_wins_overall":     wins_overall,
        "h2h_losses_overall":   losses_overall,
        "h2h_win_pct_overall":  win_pct_overall,
        "h2h_weighted_win_pct": weighted_win_pct,
        "h2h_last3":            last3_str,
        "h2h_avg_sets":         avg_sets,
        "h2h_momentum":         momentum,
    })

    # ── surface stats ─────────────────────────────────────────────────────────
    if n_surface == 0:
        result.update({
            "h2h_wins_surface":    0,
            "h2h_losses_surface":  0,
            "h2h_win_pct_surface": np.nan,
        })
        return result

    wins_surface   = int(surf_h2h["result_a"].sum())
    losses_surface = n_surface - wins_surface
    win_pct_surface = (wins_surface / n_surface) if h2h_surface_available else np.nan

    result.update({
        "h2h_wins_surface":    wins_surface,
        "h2h_losses_surface":  losses_surface,
        "h2h_win_pct_surface": win_pct_surface,
    })

    return result


# ── pipeline wiring convenience ───────────────────────────────────────────────

def h2h_feature_cols() -> list[str]:
    """All keys returned by get_h2h_features — useful for model feature selection."""
    return [
        "h2h_wins_overall", "h2h_losses_overall",
        "h2h_wins_surface", "h2h_losses_surface",
        "h2h_win_pct_overall", "h2h_win_pct_surface",
        "h2h_weighted_win_pct",
        "h2h_last3", "h2h_avg_sets", "h2h_momentum",
        "h2h_n_overall", "h2h_n_surface",
        "h2h_available", "h2h_surface_available", "surface_h2h_unavailable",
    ]


# ── CLI / sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    matches = load_processed()

    DJOKOVIC_ID = 104925
    NADAL_ID    = 104745

    # Use the day after Djokovic vs Nadal 2022 Roland Garros SF (2022-05-23)
    # as our prediction date — so that match is included in the 4yr window
    pred_date = pd.Timestamp("2023-01-01")

    print("=" * 60)
    print(f"Djokovic (A) vs Nadal (B) on clay — as of {pred_date.date()}")
    print("=" * 60)

    feats = get_h2h_features(matches, DJOKOVIC_ID, NADAL_ID, "clay", pred_date)
    for k, v in feats.items():
        print(f"  {k:<30} {v}")

    print()
    print("=" * 60)
    print(f"Djokovic (A) vs Nadal (B) on hard — as of {pred_date.date()}")
    print("=" * 60)
    feats_hard = get_h2h_features(matches, DJOKOVIC_ID, NADAL_ID, "hard", pred_date)
    for k, v in feats_hard.items():
        print(f"  {k:<30} {v}")

    # Edge case: players with no H2H
    print()
    print("=" * 60)
    print("No-H2H fallback (unknown vs unknown)")
    print("=" * 60)
    feats_empty = get_h2h_features(matches, 999991, 999992, "hard", pred_date)
    for k, v in feats_empty.items():
        print(f"  {k:<30} {v}")
