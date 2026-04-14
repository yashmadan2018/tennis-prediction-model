"""
Market layer: odds parsing, implied probability, line movement, CLV.

Philosophy
----------
Closing line value (CLV) is the gold standard for evaluating prediction quality.
A model that consistently beats the closing line is extracting real edge — not noise.

Odds handling
-------------
- Decimal: 1.91 means you get $1.91 back per $1 staked (includes principal)
- American: -150 means stake $150 to win $100; +130 means stake $100 to win $130
- All formats converted to decimal internally before any calculation.

Vig removal
-----------
Raw implied probabilities from decimal odds sum to > 1.0 (the book's margin).
Multiplicative method: divide each raw implied by their sum.
  raw_a = 1 / odds_a
  raw_b = 1 / odds_b
  vig   = raw_a + raw_b       # e.g. 1.053 → 5.3% margin
  true_a = raw_a / vig
  true_b = raw_b / vig

This is preferred over additive subtraction because it distributes margin
proportionally — if the book is 60/40 before vig, it stays 60/40 after.

Sharp movement signal
---------------------
Public money shortens the favourite. Sharp money shortens the underdog.
sharp_flag = True when line moved ≥ SHARP_THRESHOLD pp toward the underdog.

Reverse line movement
---------------------
If public % heavily backs player A but line moves toward B, that's sharp action.
Only computed when public betting percentage data is supplied.

CLV delta
---------
clv_delta = model_prob_a - closing_implied_a
Positive = model sees player A as more likely to win than the market at close.
Negative = model is behind the market (wrong way, or market has better info).

Usage
-----
  from features.market import get_market_features, log_clv, american_to_decimal

  feats = get_market_features(
      opening_odds_a=1.67, opening_odds_b=2.30,
      closing_odds_a=1.56, closing_odds_b=2.50,
  )
  log_clv(clv_row, output_path)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

SHARP_THRESHOLD = 0.03        # 3 pp movement toward underdog = sharp flag
FLAT_THRESHOLD  = 0.005       # movements below 0.5 pp treated as noise / flat

CLV_PATH = Path(__file__).parent.parent / "output" / "clv_tracker.csv"

CLV_COLUMNS = [
    "date", "tournament", "surface",
    "player_a", "player_b",
    "model_prob_a",
    "opening_implied_a", "closing_implied_a",
    "sharp_flag", "movement_magnitude",
    "clv_delta",
    "result",
]


# ── odds conversion ────────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    """
    Convert American moneyline odds to decimal odds.

    Examples
    --------
    american_to_decimal(-150) → 1.6667
    american_to_decimal(+130) → 2.30
    american_to_decimal(-110) → 1.9091
    """
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    else:
        raise ValueError(
            f"American odds value {american} is invalid — "
            "must be ≥ +100 or ≤ -100 (e.g. +130, -150)."
        )


def to_decimal(odds: float, fmt: str = "decimal") -> float:
    """
    Normalise odds to decimal format.

    Parameters
    ----------
    odds : raw odds value
    fmt  : 'decimal' | 'american'
    """
    if fmt == "american":
        return american_to_decimal(odds)
    if fmt == "decimal":
        if odds < 1.0:
            raise ValueError(f"Decimal odds must be ≥ 1.0, got {odds}")
        return float(odds)
    raise ValueError(f"Unknown odds format '{fmt}'. Use 'decimal' or 'american'.")


# ── vig removal ───────────────────────────────────────────────────────────────

def remove_vig(odds_a: float, odds_b: float) -> tuple[float, float, float]:
    """
    Convert decimal odds to vig-removed implied probabilities.

    Uses the multiplicative method: proportional vig distribution.

    Parameters
    ----------
    odds_a, odds_b : decimal odds (must be ≥ 1.01)

    Returns
    -------
    (true_implied_a, true_implied_b, vig_pct)
    true_implied_a + true_implied_b == 1.0
    vig_pct: book margin as a percentage (e.g. 5.3 means 5.3%)
    """
    if odds_a < 1.01 or odds_b < 1.01:
        raise ValueError(f"Decimal odds must be ≥ 1.01, got {odds_a}, {odds_b}")

    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    total = raw_a + raw_b

    true_a = raw_a / total
    true_b = raw_b / total
    vig_pct = (total - 1.0) * 100.0

    return true_a, true_b, vig_pct


# kept for backward compatibility with pipeline.py import
def decimal_to_implied(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Thin wrapper — returns (true_implied_a, true_implied_b) only."""
    a, b, _ = remove_vig(odds_a, odds_b)
    return a, b


# ── line movement ─────────────────────────────────────────────────────────────

def _movement_direction(delta: float) -> int:
    """
    +1 = line moved toward player A (A shortened / became more likely)
    -1 = line moved toward player B
     0 = flat / noise
    """
    if delta > FLAT_THRESHOLD:
        return 1
    if delta < -FLAT_THRESHOLD:
        return -1
    return 0


def line_movement(
    opening_odds_a: float,
    opening_odds_b: float,
    closing_odds_a: float,
    closing_odds_b: float,
) -> dict:
    """
    Characterise line movement given opening and closing decimal odds for both players.

    Returns
    -------
    movement_magnitude   : abs change in vig-removed implied prob for player A (0–1)
    movement_direction   : +1 toward A, -1 toward B, 0 flat
    sharp_flag           : 1 if ≥ SHARP_THRESHOLD pp moved toward the underdog
    line_delta_a         : signed change in vig-removed implied prob for A (close - open)
    open_vig_pct         : opening book margin %
    close_vig_pct        : closing book margin %
    """
    open_a,  open_b,  open_vig  = remove_vig(opening_odds_a, opening_odds_b)
    close_a, close_b, close_vig = remove_vig(closing_odds_a, closing_odds_b)

    delta = close_a - open_a          # positive → A shortened (favourite getting money)
    direction = _movement_direction(delta)

    # Opening favourite: raw implied > 0.5
    a_opened_favourite = open_a > 0.5

    # Sharp: line moved toward the underdog by ≥ threshold
    if a_opened_favourite:
        # sharp = line moved toward B (the underdog) despite A being favourite
        sharp = int(direction == -1 and abs(delta) >= SHARP_THRESHOLD)
    else:
        # A is the underdog; sharp = line moved toward A
        sharp = int(direction == 1 and abs(delta) >= SHARP_THRESHOLD)

    return {
        "movement_magnitude": round(abs(delta), 4),
        "movement_direction": direction,
        "sharp_flag":         sharp,
        "line_delta_a":       round(delta, 4),
        "open_vig_pct":       round(open_vig, 3),
        "close_vig_pct":      round(close_vig, 3),
    }


# ── reverse line movement ─────────────────────────────────────────────────────

def reverse_line_movement(
    public_pct_a: Optional[float],
    movement_direction: int,
) -> Optional[int]:
    """
    Return 1 if public heavily backs A but line moves toward B (sharp on B),
    or if public heavily backs B but line moves toward A (sharp on A).
    Returns None if public_pct_a is unavailable.

    Parameters
    ----------
    public_pct_a     : fraction of public bets on player A (0.0–1.0), or None
    movement_direction : from line_movement() — +1, -1, or 0
    """
    if public_pct_a is None:
        return None

    PUBLIC_THRESHOLD = 0.60  # ≥ 60% public bets = "heavily backed"

    if public_pct_a >= PUBLIC_THRESHOLD and movement_direction == -1:
        return 1  # public on A, line moved to B → sharp on B
    if public_pct_a <= (1.0 - PUBLIC_THRESHOLD) and movement_direction == 1:
        return 1  # public on B, line moved to A → sharp on A
    return 0


# ── main feature function ─────────────────────────────────────────────────────

def get_market_features(
    opening_odds_a: float,
    opening_odds_b: float,
    closing_odds_a: Optional[float] = None,
    closing_odds_b: Optional[float] = None,
    model_prob_a:   Optional[float] = None,
    public_pct_a:   Optional[float] = None,
    odds_format:    str = "decimal",
) -> dict:
    """
    Compute all market features for one match.

    Parameters
    ----------
    opening_odds_a/b : odds at market open
    closing_odds_a/b : odds at close — if None, closing features are NaN
    model_prob_a     : model's win probability for player A (optional, for CLV delta)
    public_pct_a     : fraction of public bets on A — for reverse line movement
    odds_format      : 'decimal' | 'american' — applied to all four inputs

    Returns
    -------
    dict with keys:
      opening_implied_a    vig-removed opening implied prob for A
      opening_implied_b
      opening_vig_pct      book margin at open
      closing_implied_a    vig-removed closing implied prob (NaN if unavailable)
      closing_implied_b
      closing_vig_pct
      closing_unavailable  1 if closing odds not supplied
      movement_magnitude   abs pp change open→close
      movement_direction   +1/-1/0
      sharp_flag           1 if sharp-money signal
      line_delta_a         signed pp change for A
      open_vig_pct         alias for opening_vig_pct (pipeline compat)
      close_vig_pct        alias
      reverse_line_movement  1/0/None
      clv_delta            model_prob_a - closing_implied_a (NaN if either missing)
    """
    # When opening odds are absent return all-NaN (no odds data for this match)
    opening_available = opening_odds_a is not None and opening_odds_b is not None
    if not opening_available:
        return {
            "opening_implied_a":     np.nan,
            "opening_implied_b":     np.nan,
            "opening_vig_pct":       np.nan,
            "closing_implied_a":     np.nan,
            "closing_implied_b":     np.nan,
            "closing_vig_pct":       np.nan,
            "closing_unavailable":   1,
            "movement_magnitude":    np.nan,
            "movement_direction":    np.nan,
            "sharp_flag":            np.nan,
            "line_delta_a":          np.nan,
            "open_vig_pct":          np.nan,
            "close_vig_pct":         np.nan,
            "reverse_line_movement": None,
            "clv_delta":             np.nan,
        }

    # Normalise to decimal
    oa = to_decimal(opening_odds_a, odds_format)
    ob = to_decimal(opening_odds_b, odds_format)

    open_a, open_b, open_vig = remove_vig(oa, ob)

    result: dict = {
        "opening_implied_a":  round(open_a, 4),
        "opening_implied_b":  round(open_b, 4),
        "opening_vig_pct":    round(open_vig, 3),
    }

    closing_available = closing_odds_a is not None and closing_odds_b is not None

    if closing_available:
        ca = to_decimal(closing_odds_a, odds_format)
        cb = to_decimal(closing_odds_b, odds_format)
        close_a, close_b, close_vig = remove_vig(ca, cb)

        mv = line_movement(oa, ob, ca, cb)

        result.update({
            "closing_implied_a":     round(close_a, 4),
            "closing_implied_b":     round(close_b, 4),
            "closing_vig_pct":       round(close_vig, 3),
            "closing_unavailable":   0,
            "movement_magnitude":    mv["movement_magnitude"],
            "movement_direction":    mv["movement_direction"],
            "sharp_flag":            mv["sharp_flag"],
            "line_delta_a":          mv["line_delta_a"],
            "open_vig_pct":          mv["open_vig_pct"],
            "close_vig_pct":         mv["close_vig_pct"],
            "reverse_line_movement": reverse_line_movement(public_pct_a, mv["movement_direction"]),
            "clv_delta":             round(model_prob_a - close_a, 4)
                                     if model_prob_a is not None else np.nan,
        })
    else:
        result.update({
            "closing_implied_a":     np.nan,
            "closing_implied_b":     np.nan,
            "closing_vig_pct":       np.nan,
            "closing_unavailable":   1,
            "movement_magnitude":    np.nan,
            "movement_direction":    np.nan,
            "sharp_flag":            np.nan,
            "line_delta_a":          np.nan,
            "open_vig_pct":          round(open_vig, 3),
            "close_vig_pct":         np.nan,
            "reverse_line_movement": None,
            "clv_delta":             np.nan,
        })

    return result


# ── CLV logging ───────────────────────────────────────────────────────────────

def log_clv(
    date:              pd.Timestamp | str,
    tournament:        str,
    surface:           str,
    player_a:          str,
    player_b:          str,
    model_prob_a:      float,
    opening_odds_a:    float,
    opening_odds_b:    float,
    closing_odds_a:    Optional[float] = None,
    closing_odds_b:    Optional[float] = None,
    odds_format:       str = "decimal",
    output_path:       Path | str = CLV_PATH,
) -> dict:
    """
    Compute CLV metrics and append one row to clv_tracker.csv.

    Returns the dict that was written (for inspection / testing).
    Result column is left blank — filled post-match via update_clv_result().
    """
    feats = get_market_features(
        opening_odds_a, opening_odds_b,
        closing_odds_a, closing_odds_b,
        model_prob_a=model_prob_a,
        odds_format=odds_format,
    )

    row = {
        "date":               str(date)[:10],
        "tournament":         tournament,
        "surface":            surface,
        "player_a":           player_a,
        "player_b":           player_b,
        "model_prob_a":       round(model_prob_a, 4),
        "opening_implied_a":  feats["opening_implied_a"],
        "closing_implied_a":  feats["closing_implied_a"],
        "sharp_flag":         feats["sharp_flag"],
        "movement_magnitude": feats["movement_magnitude"],
        "clv_delta":          feats["clv_delta"],
        "result":             "",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def update_clv_result(
    output_path: Path | str,
    date: str,
    player_a: str,
    result: str,   # 'W' or 'L' from player A's perspective
) -> int:
    """
    Fill in the result column for a previously logged CLV row.
    Matches on date + player_a. Returns number of rows updated.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return 0

    df = pd.read_csv(output_path)
    mask = (df["date"].astype(str).str[:10] == date[:10]) & (df["player_a"] == player_a)
    n = int(mask.sum())
    df.loc[mask, "result"] = result
    df.to_csv(output_path, index=False)
    return n


# ── CLI sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Sanity check: American odds -150/+130 open → -180/+155 close")
    print("=" * 65)

    feats = get_market_features(
        opening_odds_a=-150, opening_odds_b=+130,
        closing_odds_a=-180, closing_odds_b=+155,
        model_prob_a=0.65,
        odds_format="american",
    )
    for k, v in feats.items():
        print(f"  {k:<28} {v}")

    # ── manual verification ────────────────────────────────────────────────────
    # American -150 → decimal 1.6667 → raw_impl = 0.6000
    # American +130 → decimal 2.30   → raw_impl = 0.4348
    # sum = 1.0348  vig = 3.48%
    # true_a = 0.6000 / 1.0348 = 0.5797
    # true_b = 0.4348 / 1.0348 = 0.4202
    #
    # American -180 → decimal 1.5556 → raw_impl = 0.6429
    # American +155 → decimal 2.55   → raw_impl = 0.3922
    # sum = 1.0351  vig = 3.51%
    # true_a_close = 0.6429 / 1.0351 = 0.6210
    # true_b_close = 0.3922 / 1.0351 = 0.3790
    #
    # delta = 0.6210 - 0.5797 = +0.0413 → line moved toward A (favourite shortened)
    # A opened as favourite (0.5797 > 0.5); line moved TOWARD A → public signal, NOT sharp
    # sharp_flag should be 0
    #
    # model_prob_a = 0.65; closing_implied_a ≈ 0.621 → clv_delta ≈ +0.029
    print()
    print("Expected:")
    print("  opening_implied_a       ~0.5797")
    print("  closing_implied_a       ~0.6210")
    print("  line moved toward A     (direction = +1)")
    print("  sharp_flag              0  (favourite shortened = public money)")
    print("  clv_delta               ~+0.029  (model above market)")
    print()

    # ── second check: sharp flag trigger ──────────────────────────────────────
    print("=" * 65)
    print("Sharp-flag check: favourite shortens at open but UNDERDOG closes shorter")
    print("Opening -200/+170  →  Closing -150/+120  (underdog shortened)")
    print("=" * 65)

    feats2 = get_market_features(
        opening_odds_a=-200, opening_odds_b=+170,
        closing_odds_a=-150, closing_odds_b=+120,
        odds_format="american",
    )
    for k, v in feats2.items():
        print(f"  {k:<28} {v}")
    print()
    print("Expected:")
    print("  A opened as favourite (>0.5); line moved toward B → sharp_flag = 1")
    print()

    # ── closing unavailable ────────────────────────────────────────────────────
    print("=" * 65)
    print("Closing-unavailable check: only opening odds supplied")
    print("=" * 65)
    feats3 = get_market_features(
        opening_odds_a=1.80, opening_odds_b=2.10,
    )
    print(f"  closing_unavailable     {feats3['closing_unavailable']}")
    print(f"  movement_magnitude      {feats3['movement_magnitude']}")
    print(f"  sharp_flag              {feats3['sharp_flag']}")
    print()
    print("Expected: closing_unavailable=1, movement/sharp=NaN")
    print()

    # ── CLV log test ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("CLV log test: writing to /tmp/clv_test.csv")
    print("=" * 65)
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mktemp(suffix=".csv"))
    logged = log_clv(
        date="2024-06-15",
        tournament="Wimbledon",
        surface="grass",
        player_a="Djokovic N.",
        player_b="Alcaraz C.",
        model_prob_a=0.52,
        opening_odds_a=-120,
        opening_odds_b=+100,
        closing_odds_a=-135,
        closing_odds_b=+115,
        odds_format="american",
        output_path=tmp,
    )
    print(f"  Logged row: {logged}")
    written = pd.read_csv(tmp)
    print(f"  CSV contents:\n{written.to_string(index=False)}")
    tmp.unlink()
