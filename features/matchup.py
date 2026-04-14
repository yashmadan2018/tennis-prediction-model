"""
Playing style matchup features.

Since Sackmann CSVs contain no shot-by-shot rally data, all rally-length
profiles are derived analytically from serve/return aggregate stats:

  rally_0_4_score  ← ace rate + 1st serve won%    (serve dominance = short points)
  rally_5_8_score  ← hold% + break% blend          (transition zone)
  rally_9plus_score← 2nd return won% + 1st ret won% (groundstroke/retrieval edge)

These are "tendency scores", not literal win% — they quantify how much a player
skews toward winning short/medium/long points relative to the tour average.
The model sees deltas (A − B), so absolute scale cancels.

Style classification uses thresholds on:
  hold%, 1st-serve-won%, ace_rate, 2nd-return-won%
Buckets: big_server | baseliner_aggressive | counterpuncher | all_court | serve_and_volley

Lefty/righty splits are computed from match-level hand columns over last 4 years.

Output
------
  get_matchup_features(player_a_id, player_b_id, surface, date,
                       matches, sr_index, elo_index) → dict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features.serve_return import get_serve_return_features, STAT_KEYS
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features.serve_return import get_serve_return_features, STAT_KEYS

# ── constants ─────────────────────────────────────────────────────────────────

HAND_YEARS  = 4     # lookback for hand-split win rates
ACE_WINDOW  = 30    # matches for rolling ace rate

# Tour-average ace rates (hard calibration baseline)
TOUR_AVG_ACE_RATE = {"hard": 0.082, "clay": 0.048, "grass": 0.092, "carpet": 0.090}

# Style-classification thresholds (calibrated to hard-court tour averages)
# Grass shifts these up ~3pp naturally via higher ball-bounce variance
BIG_SERVER_ACE_MIN        = 0.130   # ~1.6× tour avg on hard
BIG_SERVER_FIRST_WON_MIN  = 0.760
BIG_SERVER_HOLD_MIN       = 0.720
SNV_ACE_MIN               = 0.110   # serve-and-volley: slightly lower ace thresh
SNV_HOLD_MIN              = 0.730
AGGRESSIVE_BREAK_MIN      = 0.390
AGGRESSIVE_FIRST_RET_MIN  = 0.310
COUNTER_SECOND_RET_MIN    = 0.540
COUNTER_HOLD_MAX          = 0.660


# ── ace rate helper ──────────────────────────────────────────────────────────

def _compute_ace_rate(
    matches: pd.DataFrame,
    player_id: int,
    surface: str,
    date: pd.Timestamp,
    window: int = ACE_WINDOW,
) -> float:
    """Rolling ace rate for a player on a surface (last `window` surface matches)."""
    surface = surface.lower()
    as_w = matches[
        (matches["winner_id"] == player_id) & (matches["surface"] == surface)
        & (matches["date"] < date) & matches["w_ace"].notna()
    ].tail(window)
    as_l = matches[
        (matches["loser_id"] == player_id) & (matches["surface"] == surface)
        & (matches["date"] < date) & matches["l_ace"].notna()
    ].tail(window)

    total_aces = as_w["w_ace"].sum() + as_l["l_ace"].sum()
    total_svpt = as_w["w_svpt"].sum() + as_l["l_svpt"].sum()
    if total_svpt == 0:
        return TOUR_AVG_ACE_RATE.get(surface, 0.082)
    return float(total_aces / total_svpt)


def _get_player_hand(
    matches: pd.DataFrame,
    player_id: int,
) -> str:
    """Return the most-observed hand (R/L) for a player; 'U' if unknown."""
    w_hand = matches[matches["winner_id"] == player_id]["winner_hand"]
    l_hand = matches[matches["loser_id"]  == player_id]["loser_hand"]
    combined = pd.concat([w_hand, l_hand])
    valid    = combined[combined.isin(["R", "L"])]
    if valid.empty:
        return "U"
    return str(valid.mode().iloc[0])


# ── rally profile estimation ─────────────────────────────────────────────────

def _estimate_rally_profile(
    sr: dict,
    ace_rate: float,
    surface: str,
) -> dict:
    """
    Estimate per-player rally-length tendency scores from aggregate serve/return stats.

    Returns scores in [0, 1] — higher = stronger tendency to win points
    of that rally length.  Absolute values are not win%; use deltas (A−B)
    in the model.

    rally_0_4_score  : dominated by serve speed / ace rate
    rally_5_8_score  : transition zone — hold + break blend
    rally_9plus_score: long-rally player — 2nd return won + retrieval
    """
    tour_ace = TOUR_AVG_ACE_RATE.get(surface.lower(), 0.082)
    norm_ace = ace_rate / tour_ace if tour_ace > 0 else 1.0   # 1.0 = tour average

    fsw  = sr.get("first_srv_won_pct",  np.nan)
    ssw  = sr.get("second_srv_won_pct", np.nan)
    hold = sr.get("hold_pct",           np.nan)
    brk  = sr.get("break_pct",          np.nan)
    r1   = sr.get("first_ret_won_pct",  np.nan)
    r2   = sr.get("second_ret_won_pct", np.nan)

    def _safe(*vals, weights=None):
        """Weighted mean of non-NaN values; NaN if all missing."""
        pairs = [(v, w) for v, w in zip(vals, weights or [1]*len(vals)) if not np.isnan(v)]
        if not pairs:
            return np.nan
        num = sum(v * w for v, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den

    # 0-4 shots: aces + fast serve-side wins dominate
    rally_0_4   = _safe(norm_ace * 0.5, fsw, weights=[0.40, 0.60])

    # 5-8 shots: balanced — hold and break ability; second serve quality
    rally_5_8   = _safe(hold, brk, ssw, weights=[0.40, 0.35, 0.25])

    # 9+ shots: return quality / endurance; second return is most diagnostic
    rally_9plus = _safe(r2, r1, ssw, weights=[0.45, 0.35, 0.20])

    return {
        "rally_0_4_score":   rally_0_4,
        "rally_5_8_score":   rally_5_8,
        "rally_9plus_score": rally_9plus,
    }


# ── style classification ──────────────────────────────────────────────────────

STYLE_UNKNOWN = "all_court"

def _classify_style(
    sr: dict,
    ace_rate: float,
    surface: str,
) -> str:
    """
    Classify a player into a style bucket from their serve/return stats.

    Priority order: big_server > serve_and_volley > baseliner_aggressive
                    > counterpuncher > all_court
    Thresholds are calibrated to hard-court tour averages; grass naturally
    pushes players toward big_server / serve_and_volley due to faster pace.
    """
    hold = sr.get("hold_pct",           np.nan)
    fsw  = sr.get("first_srv_won_pct",  np.nan)
    brk  = sr.get("break_pct",          np.nan)
    r1   = sr.get("first_ret_won_pct",  np.nan)
    r2   = sr.get("second_ret_won_pct", np.nan)

    def _ok(val: float) -> bool:
        return not np.isnan(val)

    # big_server: dominant ace rate + 1st serve won% + holds
    if (_ok(hold) and _ok(fsw) and _ok(ace_rate)
            and ace_rate >= BIG_SERVER_ACE_MIN
            and fsw      >= BIG_SERVER_FIRST_WON_MIN
            and hold     >= BIG_SERVER_HOLD_MIN):
        return "big_server"

    # serve_and_volley: high hold + high ace rate (slightly lower thresholds)
    # Detected on any surface but most common on grass
    if (_ok(hold) and _ok(ace_rate)
            and ace_rate >= SNV_ACE_MIN
            and hold     >= SNV_HOLD_MIN):
        return "serve_and_volley"

    # baseliner_aggressive: attacks from the baseline; strong break%
    if (_ok(brk) and _ok(r1)
            and brk >= AGGRESSIVE_BREAK_MIN
            and r1  >= AGGRESSIVE_FIRST_RET_MIN):
        return "baseliner_aggressive"

    # counterpuncher: lower hold, wins by outlasting opponents (high 2nd ret)
    if (_ok(r2) and _ok(hold)
            and r2   >= COUNTER_SECOND_RET_MIN
            and hold <= COUNTER_HOLD_MAX):
        return "counterpuncher"

    return STYLE_UNKNOWN


# ── hand split win rates ──────────────────────────────────────────────────────

def _hand_split_win_rates(
    matches: pd.DataFrame,
    player_id: int,
    surface: str,
    date: pd.Timestamp,
    years: int = HAND_YEARS,
) -> dict:
    """
    Win% against lefties and righties for a player on a surface over last N years.
    Returns NaN where sample is zero.
    """
    surface = surface.lower()
    cutoff  = date - pd.DateOffset(years=years)

    as_w = matches[
        (matches["winner_id"] == player_id) & (matches["surface"] == surface)
        & (matches["date"] >= cutoff) & (matches["date"] < date)
    ].copy()
    as_w["result"]   = 1
    as_w["opp_hand"] = as_w["loser_hand"]

    as_l = matches[
        (matches["loser_id"] == player_id) & (matches["surface"] == surface)
        & (matches["date"] >= cutoff) & (matches["date"] < date)
    ].copy()
    as_l["result"]   = 0
    as_l["opp_hand"] = as_l["winner_hand"]

    all_m = pd.concat([as_w, as_l], ignore_index=True)

    def _win_vs(hand: str) -> float:
        sub = all_m[all_m["opp_hand"] == hand]
        if sub.empty:
            return np.nan
        return float(sub["result"].mean())

    return {
        "win_pct_vs_lefty":  _win_vs("L"),
        "win_pct_vs_righty": _win_vs("R"),
    }


# ── main function ─────────────────────────────────────────────────────────────

def get_matchup_features(
    player_a_id:  int,
    player_b_id:  int,
    surface:      str,
    date:         pd.Timestamp,
    matches:      pd.DataFrame,
    sr_index:     dict,
    elo_index:    dict | None = None,
) -> dict:
    """
    Return matchup features for player_a vs player_b.

    Parameters
    ----------
    player_a_id / player_b_id : numeric Sackmann IDs
    surface   : 'hard' | 'clay' | 'grass'
    date      : prediction date (all lookbacks are strictly before this)
    matches   : processed matches DataFrame (for ace rate + hand lookups)
    sr_index  : built by build_serve_return_index()
    elo_index : (unused in current version; reserved for interaction effects)

    Returns
    -------
    dict with rally deltas, style buckets, hand flags, hand-split win rates
    """
    surface = surface.lower()

    # ── serve/return stats ────────────────────────────────────────────────────
    sr_a = get_serve_return_features(player_a_id, surface, date, sr_index)
    sr_b = get_serve_return_features(player_b_id, surface, date, sr_index)

    # ── ace rates ────────────────────────────────────────────────────────────
    ace_a = _compute_ace_rate(matches, player_a_id, surface, date)
    ace_b = _compute_ace_rate(matches, player_b_id, surface, date)

    # ── rally profiles ────────────────────────────────────────────────────────
    rally_a = _estimate_rally_profile(sr_a, ace_a, surface)
    rally_b = _estimate_rally_profile(sr_b, ace_b, surface)

    # ── style classification ──────────────────────────────────────────────────
    style_a = _classify_style(sr_a, ace_a, surface)
    style_b = _classify_style(sr_b, ace_b, surface)
    style_matchup = f"{style_a}_vs_{style_b}"

    # ── hand flags ────────────────────────────────────────────────────────────
    hand_a = _get_player_hand(matches, player_a_id)
    hand_b = _get_player_hand(matches, player_b_id)
    is_lefty_righty = int({hand_a, hand_b} == {"L", "R"})

    hand_split_a = _hand_split_win_rates(matches, player_a_id, surface, date)
    hand_split_b = _hand_split_win_rates(matches, player_b_id, surface, date)

    # ── build output ──────────────────────────────────────────────────────────
    row: dict = {}

    # Rally scores per player (useful for inspection)
    row["a_rally_0_4_score"]   = rally_a["rally_0_4_score"]
    row["a_rally_5_8_score"]   = rally_a["rally_5_8_score"]
    row["a_rally_9plus_score"] = rally_a["rally_9plus_score"]
    row["b_rally_0_4_score"]   = rally_b["rally_0_4_score"]
    row["b_rally_5_8_score"]   = rally_b["rally_5_8_score"]
    row["b_rally_9plus_score"] = rally_b["rally_9plus_score"]

    # Rally delta features — these are what the model actually uses
    def _delta(key: str) -> float:
        va = rally_a.get(key, np.nan)
        vb = rally_b.get(key, np.nan)
        if np.isnan(va) or np.isnan(vb):
            return np.nan
        return float(va - vb)

    row["rally_0_4_delta"]   = _delta("rally_0_4_score")
    row["rally_5_8_delta"]   = _delta("rally_5_8_score")
    row["rally_9plus_delta"] = _delta("rally_9plus_score")

    # Ace rates (useful standalone feature too)
    row["a_ace_rate"] = ace_a
    row["b_ace_rate"] = ace_b
    row["ace_rate_delta"] = ace_a - ace_b

    # Style labels and matchup interaction
    row["a_style"]       = style_a
    row["b_style"]       = style_b
    row["style_matchup"] = style_matchup

    # Hand flags
    row["a_hand"]            = hand_a
    row["b_hand"]            = hand_b
    row["is_lefty_righty"]   = is_lefty_righty

    row["a_win_pct_vs_lefty"]   = hand_split_a["win_pct_vs_lefty"]
    row["a_win_pct_vs_righty"]  = hand_split_a["win_pct_vs_righty"]
    row["b_win_pct_vs_lefty"]   = hand_split_b["win_pct_vs_lefty"]
    row["b_win_pct_vs_righty"]  = hand_split_b["win_pct_vs_righty"]

    return row


# ── CLI / sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.data_loader import load_processed
    from features.serve_return import load_serve_return_snapshots, build_serve_return_index

    matches   = load_processed()
    sr_df     = load_serve_return_snapshots()
    sr_index  = build_serve_return_index(sr_df)

    ISNER_ID    = 104545
    DIMITROV_ID = 105777
    PRED_DATE   = pd.Timestamp("2019-07-01")  # Wimbledon 2019

    print("=" * 65)
    print(f"Isner (A) vs Dimitrov (B) on grass — {PRED_DATE.date()}")
    print("=" * 65)
    feats = get_matchup_features(
        ISNER_ID, DIMITROV_ID, "grass", PRED_DATE, matches, sr_index
    )
    for k, v in feats.items():
        print(f"  {k:<30} {v}")

    print()
    print(f"  ace_rate  Isner={feats['a_ace_rate']:.3f}  Dimitrov={feats['b_ace_rate']:.3f}")
    print(f"  rally_0_4 Isner={feats['a_rally_0_4_score']:.3f}  Dimitrov={feats['b_rally_0_4_score']:.3f}  delta={feats['rally_0_4_delta']:+.3f}")
    print(f"  rally_9+  Isner={feats['a_rally_9plus_score']:.3f}  Dimitrov={feats['b_rally_9plus_score']:.3f}  delta={feats['rally_9plus_delta']:+.3f}")
    print(f"  styles    Isner={feats['a_style']}  Dimitrov={feats['b_style']}")
    print(f"  matchup   {feats['style_matchup']}")
