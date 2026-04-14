"""
Context features: venue history, ranking trajectory, defending points pressure,
scheduling/fatigue, tournament level, round encoding.

All lookups are strictly before the prediction date — zero look-ahead.

Stats returned
--------------
Venue history (last 4 editions):
  venue_win_pct           match win% at this tournament
  venue_round_avg         average round reached (1=R128 … 7=F, 8=W/title)
  venue_titles            titles won at this tournament in window
  venue_appearances       editions appeared in
  venue_insufficient      True if < 2 appearances (flag, not disqualifier)

Ranking trajectory:
  rank_current            most recent known rank
  trajectory_3m           current − rank_3m_ago  (negative = improving)
  trajectory_6m           current − rank_6m_ago

Defending pressure:
  defending_pressure      0/1/2 — 0=none, 1=reached QF/SF, 2=reached F/won
  defending_round         furthest round reached at this tournament last year

Fatigue / scheduling:
  days_rest               days since last match
  prev_match_minutes      duration of last match (sets_played proxy if missing)
  matches_last_14_days    matches in last 14 days (including current tournament)
  timezone_shift          1 if last tournament was 5+ TZ hours away, else 0

Tournament level:
  tourney_level_ord       ordinal: GS=5, M1000=4, ATP500/PM=3, ATP250/P/I=2, Challenger=1
  is_grand_slam           binary
  is_masters              binary
  is_challenger           binary

Round encoding:
  round_stage             early=1 (R128/R64/R32), middle=2 (R16/QF), late=3 (SF/F)
  is_late_round           binary (SF or F)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

VENUE_EDITIONS    = 4
VENUE_MIN_APPS    = 2
FATIGUE_WINDOW    = 14   # days
TRAJ_MONTHS_3     = 3
TRAJ_MONTHS_6     = 6

ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    # qualifier / round-robin rows treated as 0 (don't count as main-draw deep runs)
    "Q1": 0, "Q2": 0, "Q3": 0, "RR": 3, "BR": 0,
}

LEVEL_ORD = {
    "G": 5,    # Grand Slam
    "M": 4,    # ATP Masters 1000
    "PM": 3,   # WTA Premier Mandatory
    "A": 2,    # ATP 250 / 500
    "P": 2,    # WTA Premier
    "I": 2,    # WTA International
    "W": 2,    # WTA (generic)
    "F": 4,    # Year-end Finals (treat as Masters level)
    "D": 2,    # Davis Cup
    "O": 3,    # Olympics
    "C": 1,    # Challenger / ITF
}

# Approximate UTC offset for major tournaments (used for timezone shift flag)
# Format: tourney_name (as it appears in Sackmann data) -> UTC offset (hours)
TOURNEY_TZ: dict[str, int] = {
    # Grand Slams
    "Australian Open":          11,
    "Roland Garros":             2,
    "Wimbledon":                 1,
    "Us Open":                  -4,
    # ATP Masters 1000
    "Indian Wells Masters":     -7,
    "Miami Open":               -4,
    "Monte Carlo Masters":       2,
    "Madrid Masters":            2,
    "Rome Masters":              2,
    "Canada Masters":           -4,
    "Rogers Cup":               -4,
    "Cincinnati Masters":       -4,
    "Western & Southern Open":  -4,
    "Shanghai Masters":          8,
    "Paris Masters":             1,
    "Atp Finals":                1,
    "Nitto Atp Finals":          1,
    # ATP 500 / 250
    "Dubai":                     4,
    "Rotterdam":                 1,
    "Acapulco":                 -6,
    "Barcelona":                 2,
    "Hamburg":                   2,
    "Washington":               -4,
    "Vienna":                    1,
    "Basel":                     1,
    "Beijing":                   8,
    "Tokyo":                     9,
    "Seoul":                     9,
    "Eastbourne":                1,
    "Queen'S Club":              1,
    "Halle":                     2,
    "Stuttgart":                 2,
    "Metz":                      2,
    "Lyon":                      2,
    "Marseille":                 1,
    # WTA
    "Wta Finals":                8,
    "Miami":                    -4,
    "Doha":                      3,
    "Riyadh":                    3,
    # Other ATP
    "Brisbane":                 10,
    "Sydney":                   11,
    "Adelaide":                  9,
    "Auckland":                 13,
    "Doha":                      3,
    "Buenos Aires":             -3,
    "Rio De Janeiro":           -3,
    "Bucharest":                 3,
    "Sofia":                     2,
    "Montpellier":               1,
    "Pune":                      5,
    "Chennai":                   5,
    "Kuala Lumpur":              8,
    "Hong Kong":                 8,
    "Chengdu":                   8,
    "Zhuhai":                    8,
}
TZ_SHIFT_THRESHOLD = 5   # hours


def _tourney_tz(name: str) -> int | None:
    """Return UTC offset for a tournament name, case-insensitive. None if unknown."""
    name_lc = name.strip().lower()
    for k, v in TOURNEY_TZ.items():
        if k.lower() == name_lc:
            return v
    return None


# ── venue history ─────────────────────────────────────────────────────────────

def _venue_history(
    matches: pd.DataFrame,
    player_id: int,
    tournament: str,
    date: pd.Timestamp,
    editions: int = VENUE_EDITIONS,
) -> dict:
    """
    Win%, average round reached, titles, appearances at this tournament
    over the last `editions` calendar years before `date`.
    """
    subset = matches[
        ((matches["winner_id"] == player_id) | (matches["loser_id"] == player_id))
        & (matches["tourney_name"].str.strip().str.lower() == tournament.strip().lower())
        & (matches["date"] < date)
    ].copy()

    if subset.empty:
        return {
            "venue_win_pct":      np.nan,
            "venue_round_avg":    np.nan,
            "venue_titles":       0,
            "venue_appearances":  0,
            "venue_insufficient": True,
        }

    # Keep last N editions (by calendar year)
    years = sorted(subset["date"].dt.year.unique())[-editions:]
    subset = subset[subset["date"].dt.year.isin(years)].copy()
    subset["round_val"]  = subset["round"].map(ROUND_ORDER).fillna(0)
    subset["player_won"] = (subset["winner_id"] == player_id).astype(int)

    wins        = int(subset["player_won"].sum())
    total       = len(subset)
    win_pct     = wins / total if total > 0 else np.nan

    # Best (deepest) round per edition
    deepest_per_year = subset.groupby(subset["date"].dt.year)["round_val"].max()
    round_avg        = float(deepest_per_year.mean()) if len(deepest_per_year) > 0 else np.nan

    # Titles: player won the F round
    titles = int(
        subset[(subset["round"] == "F") & (subset["player_won"] == 1)].shape[0]
    )

    appearances = int(len(deepest_per_year))

    return {
        "venue_win_pct":      win_pct,
        "venue_round_avg":    round_avg,
        "venue_titles":       titles,
        "venue_appearances":  appearances,
        "venue_insufficient": appearances < VENUE_MIN_APPS,
    }


# ── ranking trajectory ────────────────────────────────────────────────────────

def _rank_at(
    matches: pd.DataFrame,
    player_id: int,
    as_of: pd.Timestamp,
) -> float:
    """Most recent known rank for player_id at or before `as_of`."""
    w = matches[(matches["winner_id"] == player_id) & (matches["date"] <= as_of)
                & matches["winner_rank"].notna()][["date", "winner_rank"]].rename(
                    columns={"winner_rank": "rank"})
    l = matches[(matches["loser_id"] == player_id) & (matches["date"] <= as_of)
                & matches["loser_rank"].notna()][["date", "loser_rank"]].rename(
                    columns={"loser_rank": "rank"})
    combined = pd.concat([w, l]).sort_values("date")
    if combined.empty:
        return np.nan
    return float(combined.iloc[-1]["rank"])


def _ranking_trajectory(
    matches: pd.DataFrame,
    player_id: int,
    date: pd.Timestamp,
) -> dict:
    """
    Current rank and trajectory vs 3 and 6 months ago.
    Negative delta = improving (rank number dropped), positive = declining.
    """
    current = _rank_at(matches, player_id, date)
    rank_3m = _rank_at(matches, player_id, date - pd.DateOffset(months=TRAJ_MONTHS_3))
    rank_6m = _rank_at(matches, player_id, date - pd.DateOffset(months=TRAJ_MONTHS_6))

    traj_3m = float(current - rank_3m) if not (np.isnan(current) or np.isnan(rank_3m)) else np.nan
    traj_6m = float(current - rank_6m) if not (np.isnan(current) or np.isnan(rank_6m)) else np.nan

    return {
        "rank_current":  current,
        "trajectory_3m": traj_3m,
        "trajectory_6m": traj_6m,
    }


# ── defending points pressure ─────────────────────────────────────────────────

def _defending_pressure(
    matches: pd.DataFrame,
    player_id: int,
    tournament: str,
    date: pd.Timestamp,
) -> dict:
    """
    Check the player's result at this tournament in the prior calendar year.
    defending_pressure: 0=none/early exit, 1=QF/SF, 2=F/title
    """
    prior_year = date.year - 1
    prev = matches[
        ((matches["winner_id"] == player_id) | (matches["loser_id"] == player_id))
        & (matches["tourney_name"].str.strip().str.lower() == tournament.strip().lower())
        & (matches["date"].dt.year == prior_year)
    ].copy()

    if prev.empty:
        return {"defending_pressure": 0, "defending_round": "none"}

    prev["round_val"] = prev["round"].map(ROUND_ORDER).fillna(0)
    deepest_round_val = int(prev["round_val"].max())
    deepest_round     = prev.loc[prev["round_val"].idxmax(), "round"]

    # Was the player still alive at the deepest round (winner) or did they lose there?
    # Either way they "reached" it — that's what generates defending pressure.
    if deepest_round_val >= 7:      # F or better (won it)
        pressure = 2
    elif deepest_round_val >= 5:    # QF or SF
        pressure = 1
    else:
        pressure = 0

    return {
        "defending_pressure": pressure,
        "defending_round":    str(deepest_round),
    }


# ── fatigue / scheduling ──────────────────────────────────────────────────────

def _fatigue(
    matches: pd.DataFrame,
    player_id: int,
    tournament: str,
    date: pd.Timestamp,
) -> dict:
    """
    days_rest, prev_match_minutes (or sets proxy), matches_last_14_days,
    timezone_shift flag.
    """
    recent = matches[
        ((matches["winner_id"] == player_id) | (matches["loser_id"] == player_id))
        & (matches["date"] < date)
    ].sort_values("date")

    if recent.empty:
        return {
            "days_rest":           np.nan,
            "prev_match_minutes":  np.nan,
            "matches_last_14_days": 0,
            "timezone_shift":      0,
        }

    last = recent.iloc[-1]
    days_rest = int((date - last["date"]).days)

    # Duration: use minutes if available, else estimate from sets played
    mins = last.get("minutes", np.nan)
    if pd.notna(mins) and float(mins) > 0:
        prev_minutes = float(mins)
    else:
        # Estimate from score string: count sets as proxy
        score = str(last.get("score", ""))
        import re
        n_sets = len(re.findall(r"\d+-\d+", score))
        prev_minutes = float(n_sets * 40) if n_sets > 0 else np.nan

    # Matches in last 14 days (excluding date itself)
    cutoff_14 = date - pd.Timedelta(days=FATIGUE_WINDOW)
    m14 = int(((recent["date"] >= cutoff_14) & (recent["date"] < date)).sum())

    # Timezone shift: compare last tournament to current tournament
    last_tourney = str(last.get("tourney_name", ""))
    tz_last    = _tourney_tz(last_tourney)
    tz_current = _tourney_tz(tournament)
    if tz_last is not None and tz_current is not None:
        tz_shift = int(abs(tz_current - tz_last) >= TZ_SHIFT_THRESHOLD)
    else:
        tz_shift = 0   # unknown → conservative, assume no shift

    return {
        "days_rest":            days_rest,
        "prev_match_minutes":   prev_minutes,
        "matches_last_14_days": m14,
        "timezone_shift":       tz_shift,
    }


# ── tournament / round encoding ───────────────────────────────────────────────

def _tourney_encoding(level: str) -> dict:
    ord_val = LEVEL_ORD.get(str(level).upper(), 2)
    return {
        "tourney_level_ord": ord_val,
        "is_grand_slam":     int(str(level).upper() == "G"),
        "is_masters":        int(str(level).upper() in ("M", "PM", "F")),
        "is_challenger":     int(str(level).upper() == "C"),
    }


def _round_encoding(round_str: str) -> dict:
    r = str(round_str).upper()
    if r in ("R128", "R64", "R32"):
        stage = 1
    elif r in ("R16", "QF"):
        stage = 2
    else:          # SF, F, RR
        stage = 3
    return {
        "round_stage":    stage,
        "is_late_round":  int(stage == 3),
    }


# ── main function ─────────────────────────────────────────────────────────────

def get_context_features(
    matches: pd.DataFrame,
    player_id: int,
    tournament: str,
    surface: str,
    date: pd.Timestamp,
    round_str: str,
    tourney_level: str = "A",
) -> dict:
    """
    Return all context features for player_id at a given match.

    Parameters
    ----------
    matches       : processed matches DataFrame
    player_id     : numeric Sackmann player ID
    tournament    : tourney_name string (e.g. 'Roland Garros')
    surface       : 'hard' | 'clay' | 'grass'
    date          : match date (strictly before for all lookbacks)
    round_str     : round of the match (e.g. 'QF', 'R32')
    tourney_level : Sackmann tourney_level code (e.g. 'G', 'M', 'A', 'C')
    """
    out: dict = {}

    out.update(_venue_history(matches, player_id, tournament, date))
    out.update(_ranking_trajectory(matches, player_id, date))
    out.update(_defending_pressure(matches, player_id, tournament, date))
    out.update(_fatigue(matches, player_id, tournament, date))
    out.update(_tourney_encoding(tourney_level))
    out.update(_round_encoding(round_str))

    return out


# ── CLI / sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    matches = load_processed()

    NADAL_ID = 104745
    # Prediction date = first match day of RG 2023 (Nadal withdrew, so we predict
    # as if he were playing R128). Defending champion from 2022.
    PRED_DATE = pd.Timestamp("2023-05-28")

    print("=" * 65)
    print(f"Rafael Nadal at Roland Garros — as of {PRED_DATE.date()} (R128)")
    print("=" * 65)
    feats = get_context_features(
        matches, NADAL_ID,
        tournament="Roland Garros",
        surface="clay",
        date=PRED_DATE,
        round_str="R128",
        tourney_level="G",
    )
    for k, v in feats.items():
        print(f"  {k:<28} {v}")

    print()
    # Second check: a journeyman challenger player with no RG history
    print("=" * 65)
    print("Unknown Challenger player at Roland Garros (no history)")
    print("=" * 65)
    feats2 = get_context_features(
        matches, 999999,
        tournament="Roland Garros",
        surface="clay",
        date=PRED_DATE,
        round_str="R128",
        tourney_level="G",
    )
    for k, v in feats2.items():
        print(f"  {k:<28} {v}")
