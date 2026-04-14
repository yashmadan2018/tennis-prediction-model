"""
Feature pipeline: assembles all feature tiers into one row per match.

Tier 1 (core): surface Elo, serve/return stats, recent form, H2H, match format
Tier 2 (structural): matchup style, venue history, ranking trajectory, fatigue
Tier 3 (situational): injury flag, market features

Usage
-----
# One-time setup (load pre-computed artefacts)
from features.pipeline import PipelineContext, build_feature_row
ctx = PipelineContext.load()

# Per-match inference
row = build_feature_row(ctx, player_a_id=104925, player_b_id=106421, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from features.elo import (
    build_elo_index,
    compute_surface_elo,
    get_elo_at_date,
    load_elo_history,
    save_elo_history,
)
from features.serve_return import (
    build_serve_return_index,
    build_serve_return_snapshots,
    get_serve_return_features,
    load_serve_return_snapshots,
    save_serve_return_snapshots,
)
from features.form import get_form_features
from features.h2h import get_h2h_features
from features.matchup import get_matchup_features
from features.context import get_context_features
from features.injury import get_injury_features
from features.market import get_market_features, decimal_to_implied

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@dataclass
class PipelineContext:
    """
    Holds all pre-loaded, pre-indexed artefacts needed to build feature rows.
    Load once, reuse for every prediction to avoid repeated I/O.
    """
    matches: pd.DataFrame
    elo_index: dict                        # built by build_elo_index()
    sr_index: dict                         # built by build_serve_return_index()
    name_to_id: dict[str, int] = field(default_factory=dict)  # player_name -> player_id
    court_speed_lookup: dict = field(default_factory=dict)     # (name_lower, year, tour_group) → CPI

    @classmethod
    def load(cls, recompute_elo: bool = False) -> "PipelineContext":
        """
        Load processed matches and Elo history from disk.
        Set recompute_elo=True to regenerate elo_history.csv from scratch.
        """
        from utils.data_loader import load_processed

        print("[pipeline] Loading processed matches...")
        matches = load_processed()

        if recompute_elo or not (PROCESSED_DIR / "elo_history.csv").exists():
            print("[pipeline] Computing surface Elo (this takes ~2s)...")
            elo_history = compute_surface_elo(matches)
            save_elo_history(elo_history)
        else:
            print("[pipeline] Loading cached Elo history...")
            elo_history = load_elo_history()

        print("[pipeline] Building Elo index...")
        elo_index = build_elo_index(elo_history)

        # Serve/return snapshot index
        sr_path = PROCESSED_DIR / "serve_return_stats.csv"
        if not sr_path.exists():
            print("[pipeline] Computing serve/return snapshots (one-time, ~60s)...")
            sr_df = build_serve_return_snapshots(matches)
            save_serve_return_snapshots(sr_df)
        else:
            print("[pipeline] Loading cached serve/return snapshots...")
            sr_df = load_serve_return_snapshots()
        sr_index = build_serve_return_index(sr_df)

        # name -> id lookup (last known mapping wins — names are stable in Sackmann data)
        name_to_id: dict[str, int] = {}
        for row in matches[["winner_name", "winner_id"]].itertuples(index=False):
            name_to_id[row.winner_name] = int(row.winner_id)
        for row in matches[["loser_name", "loser_id"]].itertuples(index=False):
            name_to_id[row.loser_name] = int(row.loser_id)

        # Court speed index — derived from match data once and cached
        print("[pipeline] Building court speed index...")
        from utils.court_speed import build_court_speed_index
        court_speed_lookup = build_court_speed_index(matches)

        print("[pipeline] Context ready.")
        return cls(matches=matches, elo_index=elo_index, sr_index=sr_index,
                   name_to_id=name_to_id, court_speed_lookup=court_speed_lookup)



def build_feature_row(
    ctx: PipelineContext,
    player_a_id: int,
    player_b_id: int,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    tournament: str,
    match_date: pd.Timestamp,
    best_of: int = 3,
    round_str: str = "R32",
    tourney_level: str = "A",
    tour: str = "atp",
    opening_odds_a: float | None = None,
    opening_odds_b: float | None = None,
    closing_odds_a: float | None = None,
    closing_odds_b: float | None = None,
) -> dict:
    """
    Build a single flat feature row for player_a vs player_b.
    All features are computed as of match_date — zero look-ahead.

    Parameters
    ----------
    ctx             : PipelineContext loaded via PipelineContext.load()
    player_a_id     : numeric player ID (from Sackmann data)
    player_b_id     : numeric player ID
    player_a_name   : display name (used for H2H, venue lookups)
    player_b_name   : display name
    surface         : 'hard' | 'clay' | 'grass'
    tournament      : tourney_name string
    match_date      : date of the match
    best_of         : 3 or 5
    hand_a/b        : 'R' or 'L'
    *_odds_*        : decimal odds (optional — skip market features if None)
    """
    from utils.court_speed import get_court_speed

    matches = ctx.matches
    elo_index = ctx.elo_index
    surface = surface.lower()

    # Resolve numeric court speed for this match — used by Elo lookup and matchup
    court_speed = get_court_speed(
        ctx.court_speed_lookup, tournament, match_date, surface, tour
    ) if ctx.court_speed_lookup else None

    row: dict = {
        "player_a": player_a_name, "player_b": player_b_name,
        "player_a_id": player_a_id, "player_b_id": player_b_id,
        "surface": surface, "tournament": tournament,
        "date": match_date, "best_of": best_of,
    }

    # ── TIER 1: CORE ──────────────────────────────────────────────────────

    row["elo_a"] = get_elo_at_date(elo_index, player_a_id, surface, match_date, court_speed=court_speed)
    row["elo_b"] = get_elo_at_date(elo_index, player_b_id, surface, match_date, court_speed=court_speed)
    row["elo_diff"] = row["elo_a"] - row["elo_b"]

    sr_a = get_serve_return_features(player_a_id, surface, match_date, ctx.sr_index)
    sr_b = get_serve_return_features(player_b_id, surface, match_date, ctx.sr_index)
    for k, v in sr_a.items():
        row[f"a_{k}"] = v
    for k, v in sr_b.items():
        row[f"b_{k}"] = v

    name_to_id = ctx.name_to_id
    form_a = get_form_features(matches, player_a_id, surface, match_date, elo_index, name_to_id)
    form_b = get_form_features(matches, player_b_id, surface, match_date, elo_index, name_to_id)
    for k, v in form_a.items():
        row[f"a_{k}"] = v
    for k, v in form_b.items():
        row[f"b_{k}"] = v

    h2h = get_h2h_features(matches, player_a_id, player_b_id, surface, match_date)
    row.update(h2h)

    row["best_of_5"] = int(best_of == 5)

    # ── TIER 2: STRUCTURAL EDGE ───────────────────────────────────────────

    matchup = get_matchup_features(
        player_a_id, player_b_id, surface, match_date,
        matches, ctx.sr_index, ctx.elo_index,
        court_speed_index=court_speed,
    )
    row.update(matchup)

    ctx_a = get_context_features(
        matches, player_a_id, tournament, surface, match_date, round_str, tourney_level
    )
    ctx_b = get_context_features(
        matches, player_b_id, tournament, surface, match_date, round_str, tourney_level
    )
    # Tournament-level and round encoding are match-level (same for both) — store once
    for shared_key in ("tourney_level_ord", "is_grand_slam", "is_masters",
                       "is_challenger", "round_stage", "is_late_round"):
        row[shared_key] = ctx_a.pop(shared_key, np.nan)
        ctx_b.pop(shared_key, None)
    for k, v in ctx_a.items():
        row[f"a_{k}"] = v
    for k, v in ctx_b.items():
        row[f"b_{k}"] = v

    # ── TIER 3: SITUATIONAL OVERLAY ───────────────────────────────────────

    inj_a = get_injury_features(matches, player_a_id, match_date)
    inj_b = get_injury_features(matches, player_b_id, match_date)
    for k, v in inj_a.items():
        row[f"a_{k}"] = v
    for k, v in inj_b.items():
        row[f"b_{k}"] = v

    # ── MARKET LAYER (optional) ───────────────────────────────────────────

    mkt = get_market_features(
        opening_odds_a=opening_odds_a,
        opening_odds_b=opening_odds_b,
        closing_odds_a=closing_odds_a,
        closing_odds_b=closing_odds_b,
    )
    row.update(mkt)

    return row
