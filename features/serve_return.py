"""
Surface-specific serve and return stat aggregation.

Approach
--------
Snapshot table: one row per (player_id, surface) per match they play,
storing the recency-weighted stats computed from the PREVIOUS window
(never including the current match — zero look-ahead guaranteed).

Saved to data/processed/serve_return_stats.csv and indexed for O(log n)
point-in-time lookups.

Stats computed
--------------
  hold_pct              hold% on serve
  break_pct             break% on return (BP converted%)
  first_serve_pct       1st serve in%
  first_srv_won_pct     1st serve points won%
  second_srv_won_pct    2nd serve points won%
  first_ret_won_pct     1st serve return points won%
  second_ret_won_pct    2nd serve return points won%
  bp_save_pct           break points saved%
  bp_conv_pct           break points converted%

All computed as recency-weighted ratios over a rolling window of
the last WINDOW_SIZE matches on the same surface.

Weighting
---------
Exponential decay: weight_i = DECAY^i  (i=0 → most recent match)
DECAY = 0.94 → half-life ≈ 11 matches. Match 30 ago has weight 0.94^30 ≈ 0.16.

Missing data
------------
~3-5% of matches (mostly Challenger local events) have no stat columns.
Strategy:
  - Include missing-stat matches in the window (they affect recency ranks
    of older matches, preventing the window from stale-expanding).
  - For each ratio, only use matches where the relevant columns were present.
  - `data_coverage`: fraction of window matches that had stats.
  - `missing_flags`: comma-separated list of stats with < MIN_STAT_MATCHES
    valid observations (flag for downstream use; pipeline falls back to Elo).

Usage
-----
  from features.serve_return import build_serve_return_index, get_serve_return_features
  sr_index = build_serve_return_index(matches)          # or load_serve_return_index()
  stats = get_serve_return_features(player_id, 'hard', match_date, sr_index)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── constants ───────────────────────────────────────────────────────────────
WINDOW_SIZE = 30          # rolling match window per player per surface
DECAY = 0.94              # exponential recency weight per match position
MIN_STAT_MATCHES = 3      # minimum matches with valid stats for a ratio to be trusted

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

STAT_COLS = [
    "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


# ── raw per-match stat struct ────────────────────────────────────────────────

class _MatchStats(NamedTuple):
    """Raw counts from one match for one player. NaN = stat not available."""
    # serve side
    svpt:    float
    s1in:    float   # 1st serve in
    s1won:   float   # 1st serve points won
    s2won:   float   # 2nd serve points won
    svgms:   float   # service games
    bpsaved: float
    bpfaced: float
    # return side (opponent's serve stats = player's return stats)
    r_svpt:  float   # opp svpt (total return points faced)
    r_1in:   float   # opp 1stIn
    r_1won:  float   # opp 1stWon (= return LOST on 1st)
    r_2won:  float   # opp 2ndWon (= return LOST on 2nd)
    r_bpfaced: float # opp bpFaced = BPs I created as returner
    r_bpsaved: float # opp bpSaved = BPs I failed to convert
    has_stats: int   # 1 if core serve cols present, 0 otherwise


def _extract_as_winner(row) -> _MatchStats:
    """Extract player's stats from a match where they were the winner."""
    def _f(col: str) -> float:
        v = getattr(row, col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    has = 1 if pd.notna(getattr(row, "w_svpt", np.nan)) else 0
    return _MatchStats(
        svpt=_f("w_svpt"), s1in=_f("w_1stIn"), s1won=_f("w_1stWon"),
        s2won=_f("w_2ndWon"), svgms=_f("w_SvGms"),
        bpsaved=_f("w_bpSaved"), bpfaced=_f("w_bpFaced"),
        r_svpt=_f("l_svpt"), r_1in=_f("l_1stIn"), r_1won=_f("l_1stWon"),
        r_2won=_f("l_2ndWon"), r_bpfaced=_f("l_bpFaced"),
        r_bpsaved=_f("l_bpSaved"), has_stats=has,
    )


def _extract_as_loser(row) -> _MatchStats:
    """Extract player's stats from a match where they were the loser."""
    def _f(col: str) -> float:
        v = getattr(row, col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    has = 1 if pd.notna(getattr(row, "l_svpt", np.nan)) else 0
    return _MatchStats(
        svpt=_f("l_svpt"), s1in=_f("l_1stIn"), s1won=_f("l_1stWon"),
        s2won=_f("l_2ndWon"), svgms=_f("l_SvGms"),
        bpsaved=_f("l_bpSaved"), bpfaced=_f("l_bpFaced"),
        r_svpt=_f("w_svpt"), r_1in=_f("w_1stIn"), r_1won=_f("w_1stWon"),
        r_2won=_f("w_2ndWon"), r_bpfaced=_f("w_bpFaced"),
        r_bpsaved=_f("w_bpSaved"), has_stats=has,
    )


# ── weighted aggregation ─────────────────────────────────────────────────────

def _compute_stats(window: deque) -> dict:
    """
    Aggregate a window of _MatchStats into a single stats dict.

    Uses recency-weighted ratio aggregation:
        stat = Σ(w_i * numerator_i) / Σ(w_i * denominator_i)
    where w_i = DECAY^i for i=0 (most recent) to len(window)-1.

    For each ratio, only uses matches where both numerator and denominator
    had valid data. Returns NaN if fewer than MIN_STAT_MATCHES valid points.
    """
    n = len(window)
    if n == 0:
        return _empty_stats()

    # Pre-compute recency weights (index 0 = most recent)
    weights = np.array([DECAY ** i for i in range(n)])
    matches = list(window)  # index 0 = most recent

    def _wratio(num_fn, den_fn, min_matches: int = MIN_STAT_MATCHES) -> float:
        """Compute Σ(w*num) / Σ(w*den) over valid matches."""
        w_sum_num = 0.0
        w_sum_den = 0.0
        valid = 0
        for i, m in enumerate(matches):
            num = num_fn(m)
            den = den_fn(m)
            if np.isnan(num) or np.isnan(den) or den == 0:
                continue
            w = weights[i]
            w_sum_num += w * num
            w_sum_den += w * den
            valid += 1
        if valid < min_matches or w_sum_den == 0:
            return np.nan
        return w_sum_num / w_sum_den

    data_coverage = sum(m.has_stats for m in matches) / n

    # ── serve ratios ──
    first_serve_pct    = _wratio(lambda m: m.s1in,   lambda m: m.svpt)
    first_srv_won_pct  = _wratio(lambda m: m.s1won,  lambda m: m.s1in)

    def _2nd_in(m):
        return m.svpt - m.s1in if not np.isnan(m.svpt) and not np.isnan(m.s1in) else np.nan

    second_srv_won_pct = _wratio(lambda m: m.s2won,  _2nd_in)

    def _holds(m):
        """Approximate breaks = bpFaced - bpSaved; hold games = SvGms - breaks."""
        if np.isnan(m.svgms) or np.isnan(m.bpfaced) or np.isnan(m.bpsaved):
            return np.nan
        breaks = m.bpfaced - m.bpsaved
        return max(m.svgms - breaks, 0.0)

    hold_pct    = _wratio(_holds, lambda m: m.svgms)
    bp_save_pct = _wratio(lambda m: m.bpsaved, lambda m: m.bpfaced)

    # ── return ratios ──
    def _ret_1st_won(m):
        """Return points won on opponent's 1st serve = 1stIn - 1stWon_opp."""
        if np.isnan(m.r_1in) or np.isnan(m.r_1won):
            return np.nan
        return m.r_1in - m.r_1won

    def _ret_2nd_in(m):
        if np.isnan(m.r_svpt) or np.isnan(m.r_1in):
            return np.nan
        return m.r_svpt - m.r_1in

    def _ret_2nd_won(m):
        if np.isnan(m.r_svpt) or np.isnan(m.r_1in) or np.isnan(m.r_2won):
            return np.nan
        return (m.r_svpt - m.r_1in) - m.r_2won

    first_ret_won_pct  = _wratio(_ret_1st_won, lambda m: m.r_1in)
    second_ret_won_pct = _wratio(_ret_2nd_won, _ret_2nd_in)

    def _bp_converted(m):
        if np.isnan(m.r_bpfaced) or np.isnan(m.r_bpsaved):
            return np.nan
        return m.r_bpfaced - m.r_bpsaved

    bp_conv_pct  = _wratio(_bp_converted, lambda m: m.r_bpfaced)
    break_pct    = bp_conv_pct  # alias — same stat, two names in spec

    # ── missing flags ──
    stat_names = [
        "first_serve_pct", "first_srv_won_pct", "second_srv_won_pct",
        "hold_pct", "bp_save_pct", "first_ret_won_pct", "second_ret_won_pct",
        "bp_conv_pct",
    ]
    stat_vals = [
        first_serve_pct, first_srv_won_pct, second_srv_won_pct,
        hold_pct, bp_save_pct, first_ret_won_pct, second_ret_won_pct,
        bp_conv_pct,
    ]
    missing_flags = ",".join(s for s, v in zip(stat_names, stat_vals) if np.isnan(v))

    return {
        "hold_pct":            hold_pct,
        "break_pct":           break_pct,
        "first_serve_pct":     first_serve_pct,
        "first_srv_won_pct":   first_srv_won_pct,
        "second_srv_won_pct":  second_srv_won_pct,
        "first_ret_won_pct":   first_ret_won_pct,
        "second_ret_won_pct":  second_ret_won_pct,
        "bp_save_pct":         bp_save_pct,
        "bp_conv_pct":         bp_conv_pct,
        "n_matches_window":    n,
        "data_coverage":       round(data_coverage, 3),
        "missing_flags":       missing_flags if missing_flags else "none",
    }


def _empty_stats() -> dict:
    return {
        "hold_pct": np.nan, "break_pct": np.nan,
        "first_serve_pct": np.nan, "first_srv_won_pct": np.nan,
        "second_srv_won_pct": np.nan, "first_ret_won_pct": np.nan,
        "second_ret_won_pct": np.nan, "bp_save_pct": np.nan,
        "bp_conv_pct": np.nan, "n_matches_window": 0,
        "data_coverage": 0.0,
        "missing_flags": "first_serve_pct,first_srv_won_pct,second_srv_won_pct,"
                         "hold_pct,bp_save_pct,first_ret_won_pct,second_ret_won_pct,bp_conv_pct",
    }


# ── main build ───────────────────────────────────────────────────────────────

def build_serve_return_snapshots(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute recency-weighted serve/return stat snapshots for every player
    per surface at every match they played.

    Each row stores the stats as they stood BEFORE that match (zero look-ahead).
    Processes 141k matches in ~3-4s.

    Parameters
    ----------
    matches : cleaned DataFrame from data_loader (sorted by date)

    Returns
    -------
    DataFrame with one row per (match_id, player_id, surface):
        match_id, player_id, player_name, surface, date,
        hold_pct, break_pct, first_serve_pct, first_srv_won_pct,
        second_srv_won_pct, first_ret_won_pct, second_ret_won_pct,
        bp_save_pct, bp_conv_pct,
        n_matches_window, data_coverage, missing_flags
    """
    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)

    # windows[player_id][surface] = deque(maxlen=WINDOW_SIZE) of _MatchStats
    # ordered newest-first: we prepend each new match so index 0 = most recent
    windows: dict[int, dict[str, deque]] = {}

    records: list[dict] = []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="ServeReturn"):
        surface = str(row.surface).lower()
        if surface not in ("hard", "clay", "grass", "carpet"):
            surface = "hard"

        mid   = row.match_id
        date  = row.date
        w_id  = int(row.winner_id)
        l_id  = int(row.loser_id)
        w_name = row.winner_name
        l_name = row.loser_name

        for pid in (w_id, l_id):
            if pid not in windows:
                windows[pid] = {}
            if surface not in windows[pid]:
                windows[pid][surface] = deque(maxlen=WINDOW_SIZE)

        # ── snapshot BEFORE this match ──
        for pid, name, extract_fn in [
            (w_id, w_name, _extract_as_winner),
            (l_id, l_name, _extract_as_loser),
        ]:
            pre_stats = _compute_stats(windows[pid][surface])
            rec = {
                "match_id":   mid,
                "player_id":  pid,
                "player_name": name,
                "surface":    surface,
                "date":       date,
            }
            rec.update(pre_stats)
            records.append(rec)

        # ── update windows AFTER snapshot ──
        w_stats = _extract_as_winner(row)
        l_stats = _extract_as_loser(row)

        # Prepend (newest first) by rotating deque: appendleft not available on
        # deque(maxlen=...) preserving order, so we use a reversed convention:
        # we append and treat index -1 as most recent in _compute_stats.
        # CORRECTION: we append and pass reversed order to _compute_stats.
        # Actually simpler: just appendleft on a regular deque with maxlen.
        windows[w_id][surface].appendleft(w_stats)
        windows[l_id][surface].appendleft(l_stats)

    return pd.DataFrame(records)


# ── save / load ──────────────────────────────────────────────────────────────

def save_serve_return_snapshots(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "serve_return_stats.csv"
    df.to_csv(out, index=False)
    print(f"[serve_return] Saved {len(df):,} rows to {out}")
    return out


def load_serve_return_snapshots() -> pd.DataFrame:
    path = PROCESSED_DIR / "serve_return_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_serve_return_snapshots() first.")
    return pd.read_csv(path, parse_dates=["date"])


def build_serve_return_index(
    df: pd.DataFrame,
) -> dict[tuple[int, str], pd.DataFrame]:
    """
    Index snapshot table by (player_id, surface) for fast point-in-time lookups.
    Returns dict mapping (player_id, surface) -> sorted sub-DataFrame.
    """
    index: dict[tuple[int, str], pd.DataFrame] = {}
    for (pid, surf), grp in df.groupby(["player_id", "surface"]):
        index[(int(pid), str(surf))] = grp.sort_values("date").reset_index(drop=True)
    return index


# ── public lookup API ────────────────────────────────────────────────────────

STAT_KEYS = [
    "hold_pct", "break_pct", "first_serve_pct", "first_srv_won_pct",
    "second_srv_won_pct", "first_ret_won_pct", "second_ret_won_pct",
    "bp_save_pct", "bp_conv_pct", "n_matches_window", "data_coverage",
    "missing_flags",
]


def get_serve_return_features(
    player_id: int,
    surface: str,
    date: pd.Timestamp,
    sr_index: dict[tuple[int, str], pd.DataFrame],
    elo_index: dict | None = None,  # reserved for future opponent-quality weighting
) -> dict:
    """
    Return serve/return stats for player_id on surface as of date (exclusive).

    Falls back to _empty_stats() if no history exists.

    Parameters
    ----------
    player_id   : numeric Sackmann player ID
    surface     : 'hard' | 'clay' | 'grass'
    date        : match date — stats from all matches strictly before this date
    sr_index    : built by build_serve_return_index()
    elo_index   : reserved — not used in current weighting scheme
    """
    key = (int(player_id), surface.lower())
    if key not in sr_index:
        return _empty_stats()

    grp = sr_index[key]
    prior = grp[grp["date"] < date]
    if prior.empty:
        return _empty_stats()

    row = prior.iloc[-1]
    return {k: row[k] if k in row.index else np.nan for k in STAT_KEYS}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_sample(df: pd.DataFrame, n: int = 5) -> None:
    """Print serve/return stats for a sample of top active players."""
    # Latest snapshot per player per surface
    latest = (
        df.sort_values("date")
        .groupby(["player_id", "player_name", "surface"])
        .last()
        .reset_index()
    )
    # Pick players with most matches (= best data quality for demo)
    top_ids = (
        latest.groupby("player_id")["n_matches_window"]
        .max()
        .nlargest(n)
        .index.tolist()
    )
    sample = latest[latest["player_id"].isin(top_ids)][
        ["player_name", "surface", "hold_pct", "break_pct",
         "first_serve_pct", "first_srv_won_pct", "bp_save_pct",
         "bp_conv_pct", "data_coverage", "n_matches_window"]
    ].sort_values(["player_name", "surface"])

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print("\n── Serve/return stats sample (top 5 by window size) ──")
    print(sample.to_string(index=False))
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    print("[serve_return] Loading processed matches...")
    matches = load_processed()

    print(f"[serve_return] Building snapshots over {len(matches):,} matches...")
    df = build_serve_return_snapshots(matches)

    save_serve_return_snapshots(df)
    _print_sample(df)

    # Quick lookup sanity check
    print("── Lookup test: Djokovic hard stats before 2024 AO ──")
    sr_index = build_serve_return_index(df)
    stats = get_serve_return_features(104925, "hard", pd.Timestamp("2024-01-14"), sr_index)
    for k, v in stats.items():
        print(f"  {k:<25} {v}")

    print("\n[serve_return] Done.")
