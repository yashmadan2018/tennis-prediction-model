"""
Data loader for Jeff Sackmann tennis CSVs.

Loads ATP Tour, ATP Challenger, and WTA match files from data/raw/,
merges them into a single cleaned DataFrame, and saves to data/processed/matches.csv.

Raw files are never modified.

Usage:
    python utils/data_loader.py
    # or from Python:
    from utils.data_loader import load_matches
    df = load_matches()
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Columns kept from raw files (superset — missing cols get NaN)
KEEP_COLS = [
    "tourney_id", "tourney_name", "surface", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_hand", "winner_ht",
    "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "winner_seed", "winner_entry",
    "loser_id", "loser_name", "loser_hand", "loser_ht",
    "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    "loser_seed", "loser_entry",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

SURFACE_MAP = {
    "Hard": "hard",
    "Clay": "clay",
    "Grass": "grass",
    "Carpet": "carpet",
}

NUMERIC_COLS = [
    "best_of", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
    "winner_age", "loser_age",
    "winner_ht", "loser_ht",
]


def _load_file(path: str, tour: str) -> pd.DataFrame:
    """Load a single CSV and tag with tour identifier."""
    df = pd.read_csv(path, low_memory=False)

    # Keep only columns that exist
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Add missing keep-cols as NaN so downstream code can rely on them
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["tour"] = tour
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transforms — no raw data is touched."""

    # Parse date
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.drop(columns=["tourney_date"])

    # Drop rows with unparseable dates
    df = df.dropna(subset=["date"])

    # Normalise surface
    df["surface"] = df["surface"].map(SURFACE_MAP).fillna(df["surface"].str.lower())

    # Strip / title-case player names
    df["winner_name"] = df["winner_name"].astype(str).str.strip().str.title()
    df["loser_name"] = df["loser_name"].astype(str).str.strip().str.title()

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop walkovers and retirements in score that have no stats
    # (keep retirements — they're injury signals)
    df = df[df["winner_name"].str.len() > 0]
    df = df[df["loser_name"].str.len() > 0]

    # Stable match identifier
    df["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)

    # Sort chronologically
    df = df.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)

    return df


def load_matches(
    tours: list[str] | None = None,
    start_year: int = 2015,
    end_year: int = 2024,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load and merge all match CSVs from data/raw/.

    Parameters
    ----------
    tours : list of 'atp', 'challenger', 'wta' (default: all three)
    start_year : first year to include
    end_year : last year to include
    save : whether to write merged file to data/processed/matches.csv

    Returns
    -------
    pd.DataFrame with all matches merged and cleaned
    """
    if tours is None:
        tours = ["atp", "challenger", "wta", "wta125"]

    patterns = {
        "atp":        "atp_matches_[0-9][0-9][0-9][0-9].csv",
        "challenger":  "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
        "wta":        "wta_matches_[0-9][0-9][0-9][0-9].csv",
        "wta125":     "wta_matches_qual_itf_[0-9][0-9][0-9][0-9].csv",
    }

    frames = []
    for tour in tours:
        pattern = str(RAW_DIR / patterns[tour])
        files = sorted(glob.glob(pattern))
        files = [
            f for f in files
            if start_year <= int(Path(f).stem.split("_")[-1]) <= end_year
        ]
        if not files:
            print(f"[data_loader] No files found for tour={tour} in {RAW_DIR}")
            continue

        print(f"[data_loader] Loading {len(files)} {tour.upper()} files...")
        tour_frames = []
        for f in tqdm(files, desc=tour):
            try:
                tour_frames.append(_load_file(f, tour))
            except Exception as e:
                print(f"  WARNING: failed to load {f}: {e}")

        if tour_frames:
            frames.append(pd.concat(tour_frames, ignore_index=True))

    if not frames:
        raise RuntimeError("No data loaded. Check that data/raw/ contains CSV files.")

    df = pd.concat(frames, ignore_index=True)
    df = _clean(df)

    print(f"\n[data_loader] Loaded {len(df):,} matches "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"  Tours: {df['tour'].value_counts().to_dict()}")
    print(f"  Surfaces: {df['surface'].value_counts().to_dict()}")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "matches.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[data_loader] Saved to {out_path}")

    return df


def load_processed() -> pd.DataFrame:
    """Load the pre-merged processed file (faster than re-loading raw)."""
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run load_matches() first to generate it."
        )
    return pd.read_csv(path, parse_dates=["date"], low_memory=False)


if __name__ == "__main__":
    df = load_matches()
    print(df.head())
    print(df.dtypes)
