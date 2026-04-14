"""
Court Speed Index (CPI) — derived from match statistics.

Since no Sackmann flat file provides court speed, we derive it per-tournament
per-year from the actual ball-in-play data:

  CPI = surface_baseline + SCALE * z_score

  z_score is computed within the surface (hard / clay / grass separately)
  using the combined ace rate and first-serve-won % of that tournament.

Surface baselines (calibrated to approximate published CPI values):
  hard   → 75   (medium-fast)
  clay   → 40   (slow)
  grass  → 110  (fast)
  carpet → 80   (medium-fast indoor)

Scale: ± 20 per standard deviation within the surface.

Hard-court Elo split threshold: CPI >= HARD_FAST_THRESHOLD → "hard_fast"
                                 CPI <  HARD_FAST_THRESHOLD → "hard_slow"

The HARD_FAST_THRESHOLD of 70 sits about 0.25 std devs below the hard mean,
so roughly 60% of hard matches land in hard_fast and 40% in hard_slow —
separating fast outdoor slabs (Miami, Indian Wells, AO) from slower indoor
or sub-average hard surfaces.

Usage
-----
  from utils.court_speed import build_court_speed_index, get_court_speed
  lookup = build_court_speed_index(matches)   # once at load time
  cpi    = get_court_speed(lookup, "Roland Garros", 2023)  # → ~38.4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

SURFACE_BASELINE: dict[str, float] = {
    "hard":    75.0,
    "clay":    40.0,
    "grass":  110.0,
    "carpet":  80.0,
}
SURFACE_SCALE:    float = 20.0    # CPI points per within-surface std dev
HARD_FAST_THRESHOLD: float = 70.0  # CPI >= this → "hard_fast"

# Minimum matches needed to compute a reliable CPI for a tournament-year
MIN_MATCHES_CPI: int = 8

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CPI_PATH = RAW_DIR / "court_pace_index.csv"


# ── computation ────────────────────────────────────────────────────────────────

def compute_cpi_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-tournament per-year CPI from match data.

    Normalisation is done within (surface, tour_group) to prevent ATP's
    higher ace rates from pushing all WTA hard courts into "hard_slow":
      tour_group = "atp"  for ATP main + Challenger matches
      tour_group = "wta"  for WTA main + WTA 125/ITF

    Inputs needed: surface, tourney_name, date, tour,
                   w_ace, w_svpt, l_ace, l_svpt,
                   w_1stIn, w_1stWon, l_1stIn, l_1stWon

    Returns DataFrame with columns:
        tourney_name, year, surface, tour_group, n_matches,
        avg_ace_rate, avg_fsw, cpi
    """
    df = matches.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["surface_norm"] = df["surface"].str.lower().fillna("hard").replace(
        {"hard": "hard", "clay": "clay", "grass": "grass", "carpet": "carpet"}
    ).fillna("hard")

    # Tour group: ATP/Challenger vs WTA/WTA125
    df["tour_group"] = df["tour"].str.lower().map(
        lambda t: "atp" if t in ("atp", "challenger") else "wta"
    ).fillna("atp")

    # Per-match totals for both serve sides
    for col_pair, out_col in [
        (("w_ace",    "l_ace"),    "match_aces"),
        (("w_svpt",   "l_svpt"),   "match_svpt"),
        (("w_1stIn",  "l_1stIn"),  "match_1stin"),
        (("w_1stWon", "l_1stWon"), "match_1stwon"),
    ]:
        df[out_col] = (
            pd.to_numeric(df[col_pair[0]], errors="coerce").fillna(0) +
            pd.to_numeric(df[col_pair[1]], errors="coerce").fillna(0)
        )

    # Aggregate by tournament + year + tour_group
    grp = df.groupby(
        ["tourney_name", "year", "surface_norm", "tour_group"]
    ).agg(
        n_matches    = ("match_id",     "count"),
        total_aces   = ("match_aces",   "sum"),
        total_svpt   = ("match_svpt",   "sum"),
        total_1stin  = ("match_1stin",  "sum"),
        total_1stwon = ("match_1stwon", "sum"),
    ).reset_index()

    grp = grp[grp["n_matches"] >= MIN_MATCHES_CPI].copy()

    grp["avg_ace_rate"] = np.where(
        grp["total_svpt"]  > 0, grp["total_aces"]   / grp["total_svpt"],  np.nan)
    grp["avg_fsw"]       = np.where(
        grp["total_1stin"] > 0, grp["total_1stwon"] / grp["total_1stin"], np.nan)

    # Combined raw = 60% ace rate + 40% first-serve won %
    grp["raw"] = 0.6 * grp["avg_ace_rate"] + 0.4 * grp["avg_fsw"]

    # Z-score within (surface, tour_group) — keeps ATP and WTA on comparable scales
    group_stats = (
        grp.groupby(["surface_norm", "tour_group"])["raw"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "grp_mean", "std": "grp_std"})
    )
    grp = grp.join(group_stats, on=["surface_norm", "tour_group"])
    grp["grp_std"] = grp["grp_std"].fillna(0.01)

    grp["z"]   = (grp["raw"] - grp["grp_mean"]) / grp["grp_std"]
    grp["cpi"] = (grp["surface_norm"].map(SURFACE_BASELINE) + SURFACE_SCALE * grp["z"]).round(1)

    result = grp[[
        "tourney_name", "year", "surface_norm", "tour_group", "n_matches",
        "avg_ace_rate", "avg_fsw", "cpi",
    ]].rename(columns={"surface_norm": "surface"}).sort_values(
        ["surface", "tour_group", "cpi"], ascending=[True, True, False]
    ).reset_index(drop=True)

    return result


def build_court_speed_index(matches: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    """
    Compute CPI table from matches and return a fast lookup dict.

    Key   : (tourney_name_lower, year, tour_group)   tour_group = 'atp' | 'wta'
    Value : CPI float

    Falls back to surface average when a tournament-year is not in the table.
    """
    cpi_df = compute_cpi_table(matches)
    save_court_pace_index(cpi_df)

    lookup: dict[tuple[str, int, str], float] = {}
    for row in cpi_df.itertuples(index=False):
        key = (str(row.tourney_name).lower(), int(row.year), str(row.tour_group))
        lookup[key] = float(row.cpi)
    return lookup


def get_court_speed(
    lookup: dict[tuple[str, int, str], float],
    tournament: str,
    date: pd.Timestamp,
    surface: str = "hard",
    tour: str = "atp",
) -> float:
    """
    Return CPI for a specific tournament + year + tour.

    Fallback chain:
    1. Exact (tournament, year, tour_group)
    2. Adjacent years ±1, ±2
    3. Surface baseline (SURFACE_BASELINE[surface])

    Parameters
    ----------
    lookup     : built by build_court_speed_index()
    tournament : tourney_name string
    date       : match date (year extracted from this)
    surface    : 'hard' | 'clay' | 'grass' — fallback baseline key
    tour       : 'atp' | 'challenger' | 'wta' | 'wta125' — mapped to tour_group
    """
    year       = int(pd.Timestamp(date).year)
    tour_group = "atp" if tour.lower() in ("atp", "challenger") else "wta"
    tname      = tournament.lower()

    key = (tname, year, tour_group)
    if key in lookup:
        return lookup[key]

    for dy in (1, -1, 2, -2):
        alt = (tname, year + dy, tour_group)
        if alt in lookup:
            return lookup[alt]

    return SURFACE_BASELINE.get(surface.lower(), 75.0)


def is_hard_fast(cpi: float) -> bool:
    """True if the court speed qualifies as hard_fast."""
    return cpi >= HARD_FAST_THRESHOLD


def elo_surface_for_match(surface: str, cpi: float) -> str:
    """
    Map (surface, cpi) to the Elo bucket to use.

    Non-hard surfaces are unchanged.
    Hard courts split at HARD_FAST_THRESHOLD.
    """
    if surface == "hard":
        return "hard_fast" if cpi >= HARD_FAST_THRESHOLD else "hard_slow"
    return surface


# ── save / load ────────────────────────────────────────────────────────────────

def save_court_pace_index(cpi_df: pd.DataFrame) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cpi_df.to_csv(CPI_PATH, index=False)
    print(f"[court_speed] Saved {len(cpi_df):,} tournament-year CPI rows → {CPI_PATH}")
    return CPI_PATH


def load_court_pace_index() -> pd.DataFrame:
    if not CPI_PATH.exists():
        raise FileNotFoundError(
            f"{CPI_PATH} not found. Run build_court_speed_index(matches) first."
        )
    return pd.read_csv(CPI_PATH)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    matches = load_processed()
    cpi_df  = compute_cpi_table(matches)
    save_court_pace_index(cpi_df)

    print("\n── Slowest 10 (clay) ──")
    print(cpi_df[cpi_df["surface"] == "clay"].sort_values("cpi").head(10)
          [["tourney_name", "year", "cpi"]].to_string(index=False))

    print("\n── Hardest split — hard_slow (<70) sample ──")
    hard = cpi_df[cpi_df["surface"] == "hard"]
    print(hard[hard["cpi"] < HARD_FAST_THRESHOLD].sort_values("cpi")
          [["tourney_name", "year", "cpi"]].head(10).to_string(index=False))

    print("\n── Hardest split — hard_fast (≥70) sample ──")
    print(hard[hard["cpi"] >= HARD_FAST_THRESHOLD].sort_values("cpi", ascending=False)
          [["tourney_name", "year", "cpi"]].head(10).to_string(index=False))

    print("\n── Fastest 10 (grass) ──")
    print(cpi_df[cpi_df["surface"] == "grass"].sort_values("cpi", ascending=False).head(10)
          [["tourney_name", "year", "cpi"]].to_string(index=False))

    print(f"\n── Hard split: {(hard['cpi'] >= HARD_FAST_THRESHOLD).mean():.1%} fast, "
          f"{(hard['cpi'] < HARD_FAST_THRESHOLD).mean():.1%} slow ──")
