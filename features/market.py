"""
Market layer: odds parsing, implied probability, line movement, CLV.

Covers:
  - Convert decimal odds to implied probability (with vig removal)
  - Opening vs closing comparison
  - Sharp-sided movement detection (closing toward underdog = sharp)
  - Model prob vs closing implied delta (market edge)
  - CLV logging helpers
"""

import numpy as np
import pandas as pd


def decimal_to_implied(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Convert decimal odds to vig-removed implied probabilities.
    Uses basic proportional vig removal.
    """
    raw_a = 1 / odds_a
    raw_b = 1 / odds_b
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def line_movement(
    opening_odds_a: float,
    closing_odds_a: float,
) -> dict:
    """
    Characterise line movement for player A.

    direction: 'toward_a' | 'toward_b' | 'flat'
    sharp_flag: True if closing moved toward the underdog (sharp money signal)
    """
    open_impl, _ = decimal_to_implied(opening_odds_a, 1 / (1 - 1 / opening_odds_a))
    close_impl, _ = decimal_to_implied(closing_odds_a, 1 / (1 - 1 / closing_odds_a))

    delta = close_impl - open_impl
    threshold = 0.005  # 0.5pp minimum movement to call direction

    if abs(delta) < threshold:
        direction = "flat"
    elif delta > 0:
        direction = "toward_a"
    else:
        direction = "toward_b"

    # Sharp flag: favourite shortens → public; underdog shortens → sharp
    # If player A opened as favourite (impl > 0.5) and line moved toward B → sharp on B
    # If player A opened as underdog (impl < 0.5) and line moved toward A → sharp on A
    a_is_favourite = open_impl > 0.5
    sharp_flag = (a_is_favourite and direction == "toward_b") or \
                 (not a_is_favourite and direction == "toward_a")

    return {
        "line_delta_a": delta,
        "line_direction": direction,
        "sharp_flag": int(sharp_flag),
    }


def compute_market_edge(model_prob_a: float, closing_implied_a: float) -> float:
    """Raw delta between model probability and closing implied probability."""
    return model_prob_a - closing_implied_a


def build_clv_record(
    match_id: str,
    date: pd.Timestamp,
    player_a: str,
    predicted_prob: float,
    opening_odds_a: float,
    opening_odds_b: float,
    closing_odds_a: float,
    closing_odds_b: float,
) -> dict:
    """Build a CLV tracker row (result filled post-match)."""
    opening_implied, _ = decimal_to_implied(opening_odds_a, opening_odds_b)
    closing_implied, _ = decimal_to_implied(closing_odds_a, closing_odds_b)
    clv = predicted_prob - closing_implied

    return {
        "match_id": match_id,
        "date": date,
        "player_a": player_a,
        "predicted_prob": predicted_prob,
        "opening_implied": opening_implied,
        "closing_implied": closing_implied,
        "clv": clv,
        "result": np.nan,  # filled after match
    }
