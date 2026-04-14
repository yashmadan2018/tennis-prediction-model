"""
Injury flag from observable proxies only (Tier 3 — qualitative multiplier).

Philosophy
----------
No injury data exists in Sackmann CSVs. All signals are inferred from
match outcomes and timing patterns. The flag is a MULTIPLIER on the model's
win probability — it does not feed into the model as a raw feature score.

Observable proxies tracked
--------------------------
1. retirement_last_match   Player was the loser in a match where score contains
                           'RET' (they retired mid-match). Most acute signal.
2. wo_last_tournament      Player issued a walkover (W/O as loser) — withdrew
                           before a match. Signals a known injury at the time.
3. duration_spike          Last match was DURATION_SPIKE_Z z-scores above the
                           player's rolling average. Extended effort = physical stress.
4. multi_retirement        Player has retired in 2+ of their last LOOKBACK matches.
                           Recurring retirements = chronic concern.
5. early_retirement        Player retired before completing a full set in their
                           last match (score has only 0 or 1 completed sets before
                           RET). Suggests injury was severe, not tactical.

Severity scale
--------------
0 — no signals
1 — one mild signal (duration spike or single retirement)
2 — retirement + one other signal, or wo_last_tournament
3 — multiple concurrent signals

Multiplier
----------
severity 0 → 1.00 (no adjustment)
severity 1 → 0.95
severity 2 → 0.88
severity 3 → 0.80

Usage
-----
  from features.injury import get_injury_features
  feats = get_injury_features(player_id, date, matches)
  # feats['injury_multiplier'] applied to win_prob before output
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

LOOKBACK_MATCHES     = 10    # window for multi-retirement check
DURATION_CAP_MINUTES = 400   # cap outliers (erroneous entries like 4756)
DURATION_SPIKE_Z     = 1.8   # z-score threshold for "unusually long match"
DURATION_MIN_SAMPLE  = 4     # minimum prior matches needed to compute z-score

SEVERITY_MULTIPLIER = {0: 1.00, 1: 0.95, 2: 0.88, 3: 0.80}

_RET_RE = re.compile(r"\bRET\b", re.IGNORECASE)
_WO_RE  = re.compile(r"\bW/?O\b",  re.IGNORECASE)

# Sets-completed regex: count set scores (digits-digits) before RET
_SET_RE = re.compile(r"\d+-\d+(?:\(\d+\))?")


# ── score parsing helpers ────────────────────────────────────────────────────

def _is_retirement(score: str) -> bool:
    return isinstance(score, str) and bool(_RET_RE.search(score))

def _is_walkover(score: str) -> bool:
    return isinstance(score, str) and bool(_WO_RE.search(score))

def _sets_completed_before_ret(score: str) -> int:
    """
    Count fully played sets before the RET marker.
    e.g. '6-3 3-1 RET' → 1 full set completed before retirement.
    '3-1 RET' → 0 full sets (retired inside first set).
    """
    if not isinstance(score, str):
        return 0
    # Strip everything from RET onward
    pre = _RET_RE.split(score)[0]
    sets = _SET_RE.findall(pre)
    # A "complete" set has both scores; check if winner reached 6+ or tiebreak
    complete = 0
    for s in sets:
        parts = re.split(r"[-()]", s)
        nums = [int(x) for x in parts if x.isdigit()]
        if len(nums) >= 2 and max(nums) >= 6:
            complete += 1
    return complete


# ── duration z-score helper ──────────────────────────────────────────────────

def _duration_zscore(
    player_matches: pd.DataFrame,
    last_idx: int,
) -> float:
    """
    Z-score of the last match duration vs the player's prior rolling average.
    Returns NaN if insufficient history or minutes missing.
    """
    mins = (
        pd.to_numeric(player_matches["minutes"], errors="coerce")
        .clip(upper=DURATION_CAP_MINUTES)
    )
    if mins.isna().all():
        return np.nan

    # All matches except the last
    prior = mins.iloc[:last_idx].dropna()
    last  = mins.iloc[last_idx]

    if pd.isna(last) or len(prior) < DURATION_MIN_SAMPLE:
        return np.nan

    mu  = prior.mean()
    std = prior.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan

    return float((last - mu) / std)


# ── main function ─────────────────────────────────────────────────────────────

def get_injury_features(
    matches: pd.DataFrame,
    player_id: int,
    date: pd.Timestamp,
) -> dict:
    """
    Return injury proxy features for player_id strictly before `date`.

    Parameters
    ----------
    matches   : processed matches DataFrame (from data_loader)
    player_id : numeric Sackmann player ID
    date      : prediction date — all lookbacks are strictly before this

    Returns
    -------
    dict with:
      injury_flag           binary — 1 if any signal triggered
      injury_severity       0–3 scale
      injury_multiplier     suggested win-probability adjustment factor
      ret_last_match        1 if player retired in their most recent match
      wo_last_tournament    1 if player issued a W/O in their last tournament
      duration_spike        1 if last match was unusually long
      multi_retirement      1 if player retired in 2+ of last LOOKBACK matches
      early_retirement      1 if retirement happened before completing a set
      injury_signals        comma-separated list of triggered signal names
      ret_count_lookback    raw retirement count in last LOOKBACK matches
      last_match_z_score    z-score of last match duration (NaN if unavailable)
    """
    # Pull all matches for this player before prediction date, sorted oldest→newest
    player_matches = matches[
        ((matches["winner_id"] == player_id) | (matches["loser_id"] == player_id))
        & (matches["date"] < date)
    ].sort_values("date").reset_index(drop=True)

    empty = {
        "injury_flag":         0,
        "injury_severity":     0,
        "injury_multiplier":   1.00,
        "ret_last_match":      0,
        "wo_last_tournament":  0,
        "duration_spike":      0,
        "multi_retirement":    0,
        "early_retirement":    0,
        "injury_signals":      "none",
        "ret_count_lookback":  0,
        "last_match_z_score":  np.nan,
    }

    if player_matches.empty:
        return empty

    last      = player_matches.iloc[-1]
    last_idx  = len(player_matches) - 1
    lookback  = player_matches.tail(LOOKBACK_MATCHES)

    signals: list[str] = []

    # ── Signal 1: retirement in last match ──────────────────────────────────
    # Player must be the loser (they were the one who retired)
    ret_last = int(
        _is_retirement(str(last.get("score", "")))
        and int(last.get("loser_id", -1)) == player_id
    )
    if ret_last:
        signals.append("ret_last_match")

    # ── Signal 2: walkover given (withdrew from last tournament match) ────────
    # Find any W/O where this player was the loser within the last tournament
    last_tourney = str(last.get("tourney_id", ""))
    last_tourney_matches = player_matches[
        player_matches["tourney_id"].astype(str) == last_tourney
    ]
    wo_given = int(
        last_tourney_matches.apply(
            lambda r: _is_walkover(str(r.get("score", "")))
                      and int(r.get("loser_id", -1)) == player_id,
            axis=1,
        ).any()
    )
    if wo_given:
        signals.append("wo_last_tournament")

    # ── Signal 3: duration spike ──────────────────────────────────────────────
    z = _duration_zscore(player_matches, last_idx)
    dur_spike = int(not np.isnan(z) and z >= DURATION_SPIKE_Z)
    if dur_spike:
        signals.append("duration_spike")

    # ── Signal 4: multiple retirements in lookback window ────────────────────
    ret_count = int(lookback.apply(
        lambda r: _is_retirement(str(r.get("score", "")))
                  and int(r.get("loser_id", -1)) == player_id,
        axis=1,
    ).sum())
    multi_ret = int(ret_count >= 2)
    if multi_ret:
        signals.append("multi_retirement")

    # ── Signal 5: early retirement (within first set) ─────────────────────────
    early_ret = 0
    if ret_last:
        sets_done = _sets_completed_before_ret(str(last.get("score", "")))
        early_ret = int(sets_done == 0)
        if early_ret:
            signals.append("early_retirement")

    # ── Severity and multiplier ───────────────────────────────────────────────
    n_signals = len(signals)

    if n_signals == 0:
        severity = 0
    elif wo_given or (ret_last and early_ret):
        # Acute signals — jump straight to severity 2+
        severity = 2 + int(n_signals >= 3)
    elif n_signals == 1 and dur_spike and not ret_last:
        severity = 1   # duration spike alone is mild
    elif n_signals == 1:
        severity = 1
    elif n_signals == 2:
        severity = 2
    else:
        severity = 3

    severity = min(severity, 3)
    multiplier = SEVERITY_MULTIPLIER[severity]

    return {
        "injury_flag":         int(n_signals > 0),
        "injury_severity":     severity,
        "injury_multiplier":   multiplier,
        "ret_last_match":      ret_last,
        "wo_last_tournament":  wo_given,
        "duration_spike":      dur_spike,
        "multi_retirement":    multi_ret,
        "early_retirement":    early_ret,
        "injury_signals":      ",".join(signals) if signals else "none",
        "ret_count_lookback":  ret_count,
        "last_match_z_score":  round(z, 3) if not np.isnan(z) else np.nan,
    }


# ── CLI / sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_processed

    matches = load_processed()

    # ── Test 1: Nadal before 2022 Wimbledon (withdrew with abdominal injury) ──
    # He played Wimbledon SF, then withdrew. The match before withdrawal
    # was a semi vs Kyrgios that he retired from.
    NADAL_ID  = 104745
    pred_date = pd.Timestamp("2022-07-08")   # day of Wimbledon SF (withdrew)

    print("=" * 60)
    print(f"Nadal — {pred_date.date()} (withdrew from Wimbledon SF)")
    print("=" * 60)
    feats = get_injury_features(matches, NADAL_ID, pred_date)
    for k, v in feats.items():
        print(f"  {k:<28} {v}")

    # ── Test 2: Djokovic before 2021 Australian Open — no injury ──────────────
    DJOK_ID   = 104925
    pred_date2 = pd.Timestamp("2021-02-08")

    print()
    print("=" * 60)
    print(f"Djokovic — {pred_date2.date()} (healthy, AO 2021 mid-run)")
    print("=" * 60)
    feats2 = get_injury_features(matches, DJOK_ID, pred_date2)
    for k, v in feats2.items():
        print(f"  {k:<28} {v}")

    # ── Test 3: A player who retired in last match ────────────────────────────
    # Find a real case
    ret_matches = matches[
        matches["score"].str.contains("RET", na=False, case=False)
    ].sort_values("date").tail(5)

    print()
    print("=" * 60)
    print("Recent real retirement — verify signal triggers")
    print("=" * 60)
    for _, row in ret_matches.iterrows():
        pid   = int(row["loser_id"])
        pname = row["loser_name"]
        pdate = row["date"] + pd.Timedelta(days=3)
        f = get_injury_features(matches, pid, pdate)
        if f["ret_last_match"]:
            print(f"  Player: {pname}  date: {row['date'].date()}  score: {row['score']}")
            print(f"  -> ret_last={f['ret_last_match']}  severity={f['injury_severity']}"
                  f"  multiplier={f['injury_multiplier']}  z={f['last_match_z_score']}")
            print(f"     signals: {f['injury_signals']}")
            break
