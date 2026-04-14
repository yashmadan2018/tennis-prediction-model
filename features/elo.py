"""
Surface-specific Elo computation.

Design:
  - Separate Elo per (player_id, surface). Hard courts are split into
    hard_fast (CPI ≥ 70) and hard_slow (CPI < 70) using the Court Pace
    Index derived in utils/court_speed.py.  Clay and grass are unchanged.
  - K-factor varies by tourney_level to reflect result variance:
      Grand Slam (G)         : K=20  — low variance, prestige inflation
      Masters / PM (M, PM)   : K=28
      ATP 250/500 / WTA (A,I,P,W,F,D,O) : K=32
      Challenger / ITF (C)   : K=40  — high variance
  - Elo updates chronologically, match by match — zero look-ahead.
  - Natural decay: if a player hasn't played a surface in 6+ months,
    their surface Elo is partially regressed toward their cross-surface
    mean Elo before the next match is processed on that surface.
    Decay amount scales linearly from 0 at 6 months to full regression
    at 24 months.

Output schema (one row per player per match):
    match_id, player_id, player_name, surface, elo_before, elo_after, date
    (surface is 'hard_fast', 'hard_slow', 'clay', 'grass', or 'carpet')
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_ELO = 1500.0
DECAY_START_MONTHS = 6       # no decay below this gap
DECAY_FULL_MONTHS = 24       # full regression at this gap
DECAY_FRACTION = 0.5         # max fraction to regress toward cross-surface mean

K_BY_LEVEL: dict[str, float] = {
    "G": 20.0,   # Grand Slam
    "M": 28.0,   # Masters 1000
    "PM": 28.0,  # WTA Premier Mandatory
    "F": 28.0,   # Year-end Finals
    "A": 32.0,   # ATP 250 / 500
    "I": 32.0,   # WTA International
    "P": 32.0,   # WTA Premier
    "W": 32.0,   # WTA (generic)
    "D": 32.0,   # Davis Cup / Fed Cup
    "O": 32.0,   # Olympics
    "C": 40.0,   # Challenger / ITF
}
DEFAULT_K = 32.0

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── core Elo math ──────────────────────────────────────────────────────────

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _update(ra: float, rb: float, k: float) -> tuple[float, float]:
    """Winner gets score=1, loser gets score=0."""
    e = _expected(ra, rb)
    return ra + k * (1.0 - e), rb + k * (0.0 - (1.0 - e))


# ── decay helper ───────────────────────────────────────────────────────────

def _decay_factor(gap_days: float) -> float:
    """
    Returns fraction [0, DECAY_FRACTION] to regress toward cross-surface mean.
    0 for gaps < DECAY_START_MONTHS, linearly rises to DECAY_FRACTION at
    DECAY_FULL_MONTHS, capped there.
    """
    gap_months = gap_days / 30.44
    if gap_months < DECAY_START_MONTHS:
        return 0.0
    t = min((gap_months - DECAY_START_MONTHS) / (DECAY_FULL_MONTHS - DECAY_START_MONTHS), 1.0)
    return t * DECAY_FRACTION


def _cross_surface_mean(ratings: dict[str, float]) -> float:
    """Mean Elo across all surfaces a player has a rating for."""
    if not ratings:
        return DEFAULT_ELO
    return float(np.mean(list(ratings.values())))


# ── main computation ────────────────────────────────────────────────────────

def compute_surface_elo(
    matches: pd.DataFrame,
    court_speed_lookup: dict | None = None,
) -> pd.DataFrame:
    """
    Compute surface-specific Elo for every player across all matches.

    Hard courts are split into hard_fast / hard_slow when court_speed_lookup
    is provided (built by utils.court_speed.build_court_speed_index).
    Without it, all hard courts share one Elo bucket (backward-compat mode).

    Parameters
    ----------
    matches            : cleaned DataFrame from data_loader
    court_speed_lookup : dict keyed (tourney_name_lower, year, tour_group) → CPI

    Returns
    -------
    DataFrame with columns:
        match_id, player_id, player_name, surface,
        elo_before, elo_after, date
    """
    from utils.court_speed import elo_surface_for_match, get_court_speed

    matches = matches.sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)

    # Per-player, per-surface state
    ratings: dict[int, dict[str, float]] = {}     # player_id -> surface -> elo
    last_played: dict[int, dict[str, pd.Timestamp]] = {}  # player_id -> surface -> date

    records: list[dict] = []

    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="Elo"):
        raw_surface = str(row.surface).lower()
        if raw_surface not in ("hard", "clay", "grass", "carpet"):
            raw_surface = "hard"

        # Optionally split hard into hard_fast / hard_slow
        if court_speed_lookup is not None and raw_surface == "hard":
            tour = str(getattr(row, "tour", "atp")).lower()
            cpi  = get_court_speed(
                court_speed_lookup,
                str(getattr(row, "tourney_name", "")),
                row.date,
                raw_surface,
                tour,
            )
            surface = elo_surface_for_match(raw_surface, cpi)
        else:
            surface = raw_surface

        k = K_BY_LEVEL.get(str(row.tourney_level), DEFAULT_K)
        mid = row.match_id

        w_id: int = int(row.winner_id)
        l_id: int = int(row.loser_id)
        w_name: str = row.winner_name
        l_name: str = row.loser_name
        date: pd.Timestamp = row.date

        # Ensure dicts exist
        for pid in (w_id, l_id):
            if pid not in ratings:
                ratings[pid] = {}
                last_played[pid] = {}

        # Apply decay before reading pre-match rating
        for pid in (w_id, l_id):
            if surface in last_played[pid]:
                gap = (date - last_played[pid][surface]).days
                frac = _decay_factor(gap)
                if frac > 0.0:
                    target = _cross_surface_mean(ratings[pid])
                    ratings[pid][surface] += frac * (target - ratings[pid][surface])
            # Initialise if first appearance on this surface
            if surface not in ratings[pid]:
                ratings[pid][surface] = DEFAULT_ELO

        r_w = ratings[w_id][surface]
        r_l = ratings[l_id][surface]

        new_w, new_l = _update(r_w, r_l, k)

        records.append({
            "match_id": mid, "player_id": w_id, "player_name": w_name,
            "surface": surface, "elo_before": r_w, "elo_after": new_w, "date": date,
        })
        records.append({
            "match_id": mid, "player_id": l_id, "player_name": l_name,
            "surface": surface, "elo_before": r_l, "elo_after": new_l, "date": date,
        })

        ratings[w_id][surface] = new_w
        ratings[l_id][surface] = new_l
        last_played[w_id][surface] = date
        last_played[l_id][surface] = date

    return pd.DataFrame(records)


# ── fast point-in-time lookup ───────────────────────────────────────────────

def build_elo_index(elo_history: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    """
    Pre-index elo_history by (player_id, surface) for O(log n) lookups.
    Returns a dict mapping (player_id, surface) -> sorted sub-DataFrame.
    """
    index: dict[tuple[int, str], pd.DataFrame] = {}
    for (pid, surf), grp in elo_history.groupby(["player_id", "surface"]):
        index[(int(pid), surf)] = grp.sort_values("date").reset_index(drop=True)
    return index


def get_elo_at_date(
    index: dict[tuple[int, str], pd.DataFrame],
    player_id: int,
    surface: str,
    date: pd.Timestamp,
    court_speed: float | None = None,
) -> float:
    """
    Return the elo_after value for player_id on surface from the last match
    strictly before `date`.

    Hard-court CPI split
    --------------------
    When `court_speed` is provided and surface == "hard", look up the
    appropriate hard_fast / hard_slow bucket.  Fallback chain:
      1. hard_fast or hard_slow (whichever applies given CPI)
      2. The other hard bucket (cross-speed bleed is real — most players
         move between fast and slow hard)
      3. DEFAULT_ELO

    Non-hard surfaces (clay, grass, carpet) are unaffected.
    """
    from utils.court_speed import elo_surface_for_match, HARD_FAST_THRESHOLD

    pid = int(player_id)
    surf = surface.lower()

    if surf == "hard" and court_speed is not None:
        primary   = elo_surface_for_match("hard", court_speed)          # hard_fast or hard_slow
        secondary = "hard_slow" if primary == "hard_fast" else "hard_fast"
        fallbacks = [primary, secondary, "hard"]
    else:
        fallbacks = [surf]

    for key_surf in fallbacks:
        key = (pid, key_surf)
        if key not in index:
            continue
        grp   = index[key]
        prior = grp[grp["date"] < date]
        if not prior.empty:
            return float(prior.iloc[-1]["elo_after"])

    return DEFAULT_ELO


# ── save / load ─────────────────────────────────────────────────────────────

def save_elo_history(elo_history: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "elo_history.csv"
    elo_history.to_csv(out, index=False)
    print(f"[elo] Saved {len(elo_history):,} rows to {out}")
    return out


def load_elo_history() -> pd.DataFrame:
    path = PROCESSED_DIR / "elo_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run compute_surface_elo() first.")
    return pd.read_csv(path, parse_dates=["date"])


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_sample(elo_history: pd.DataFrame, n_players: int = 5) -> None:
    """Print current surface Elo for a sample of players."""
    # "Current" = highest date per (player_id, surface)
    latest = (
        elo_history.sort_values("date")
        .groupby(["player_id", "player_name", "surface"])
        .last()
        .reset_index()[["player_id", "player_name", "surface", "elo_after"]]
    )
    # Pick 5 players with the highest peak Elo (interesting sample)
    top_ids = (
        latest.groupby("player_id")["elo_after"]
        .max()
        .nlargest(n_players)
        .index.tolist()
    )
    sample = latest[latest["player_id"].isin(top_ids)].pivot_table(
        index=["player_id", "player_name"],
        columns="surface",
        values="elo_after",
    ).round(1)
    print("\n── Current surface Elo (top 5 by peak rating) ──")
    print(sample.to_string())
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    print("[elo] Loading processed matches...")
    matches = load_processed()

    # Build a stable match_id if not already present
    if "match_id" not in matches.columns:
        matches["match_id"] = matches["tourney_id"].astype(str) + "_" + matches["match_num"].astype(str)

    print(f"[elo] Computing surface Elo over {len(matches):,} matches...")
    elo_history = compute_surface_elo(matches)

    save_elo_history(elo_history)
    _print_sample(elo_history)

    print("[elo] Done.")
