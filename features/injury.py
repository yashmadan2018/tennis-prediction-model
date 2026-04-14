"""
Injury flag from observable proxies only (Tier 3 — qualitative multiplier).

Signals tracked:
  - Retirement in last match (score contains 'RET')
  - Medical timeout taken (MTO) — if flagged in data
  - Match duration spike vs player rolling average
  - Late withdrawal from this tournament (manual flag)

Injury flag is a multiplier applied to the model probability,
NOT a direct score input.
"""

import pandas as pd
import numpy as np

DURATION_SPIKE_THRESHOLD = 1.4  # 40% longer than rolling average


def get_injury_flag(
    matches: pd.DataFrame,
    player: str,
    as_of_date: pd.Timestamp,
    lookback_matches: int = 5,
) -> dict:
    """
    Scan recent matches for injury proxies.

    Returns:
        injury_flag: 1 if any proxy triggered, 0 otherwise
        injury_signal: human-readable description of triggered signals
    """
    player_matches = matches[
        ((matches["winner_name"] == player) | (matches["loser_name"] == player))
        & (matches["date"] < as_of_date)
    ].sort_values("date").tail(lookback_matches)

    if player_matches.empty:
        return {"injury_flag": 0, "injury_signal": "none"}

    signals = []
    last = player_matches.iloc[-1]

    # Signal 1: retirement in last match
    score_col = "score" if "score" in last.index else None
    if score_col and isinstance(last[score_col], str) and "RET" in last[score_col].upper():
        signals.append("retirement_last_match")

    # Signal 2: match duration spike
    if "minutes" in player_matches.columns:
        durations = player_matches["minutes"].dropna()
        if len(durations) >= 2:
            rolling_avg = durations.iloc[:-1].mean()
            last_duration = durations.iloc[-1]
            if rolling_avg > 0 and last_duration > rolling_avg * DURATION_SPIKE_THRESHOLD:
                signals.append("duration_spike")

    flag = 1 if signals else 0
    return {
        "injury_flag": flag,
        "injury_signal": ",".join(signals) if signals else "none",
    }
