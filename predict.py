"""
End-to-end prediction runner.

For every upcoming match in the odds feed:
  1. Match player names → Sackmann player IDs (exact → normalised → fuzzy)
  2. Build feature row via pipeline.py
  3. Run model inference → P(player_a wins)
  4. Compute market edge vs Pinnacle implied probability
  5. Log to output/predictions.csv and output/clv_tracker.csv

Usage
-----
  # Live run (fetches fresh odds — requires ODDS_API_KEY env var)
  python predict.py

  # Use the latest cached odds snapshot (no API call)
  python predict.py --offline

  # Use a specific snapshot
  python predict.py --snapshot data/odds/odds_snapshot_20250101_120000.csv

  # Dry run: print predictions without writing to disk
  python predict.py --dry-run

  # Single match (no odds needed)
  python predict.py --match "Carlos Alcaraz" "Novak Djokovic" --surface hard \
                    --tournament "Australian Open" --date 2025-01-20 \
                    --odds-a 1.65 --odds-b 2.35
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PREDICTIONS_CSV = ROOT / "output" / "predictions.csv"
CLV_CSV         = ROOT / "output" / "clv_tracker.csv"
MODEL_PKL       = ROOT / "models" / "saved" / "xgb_calibrated.pkl"
FEATURE_JSON    = ROOT / "models" / "saved" / "feature_list.json"
LR_PKL          = ROOT / "models" / "saved" / "lr_calibrated.pkl"
MLP_PKL         = ROOT / "models" / "saved" / "mlp_calibrated.pkl"

(ROOT / "output").mkdir(exist_ok=True)

# ── confidence label ───────────────────────────────────────────────────────────

def _confidence_label(prob: float) -> str:
    conf = max(prob, 1 - prob)
    if conf >= 0.75:
        return "high"
    if conf >= 0.65:
        return "medium"
    return "low"


# ── player name matching ───────────────────────────────────────────────────────

def _normalize_for_match(name: str) -> str:
    """Lowercase, strip accents, remove punctuation — for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()


def build_name_lookup(name_to_id: dict[str, int]) -> dict[str, int]:
    """
    Build a secondary lookup keyed on _normalize_for_match(name).
    Preserves original name_to_id as primary; this is the accent-stripped fallback.
    """
    return {_normalize_for_match(k): v for k, v in name_to_id.items()}


def resolve_player_id(
    api_name: str,
    name_to_id: dict[str, int],
    norm_lookup: dict[str, int],
    cutoff: float = 0.82,
) -> int | None:
    """
    Resolve an API player name to a Sackmann player ID.

    Resolution order:
      1. Exact match (title-cased)
      2. Normalised match (accent-stripped, lowercased)
      3. difflib fuzzy match against normalised keys (cutoff 0.82)
    """
    # 1. Exact
    titled = api_name.strip().title()
    if titled in name_to_id:
        return name_to_id[titled]

    # 2. Normalised
    normed = _normalize_for_match(api_name)
    if normed in norm_lookup:
        return norm_lookup[normed]

    # 3. Fuzzy
    matches = difflib.get_close_matches(normed, norm_lookup.keys(),
                                        n=1, cutoff=cutoff)
    if matches:
        return norm_lookup[matches[0]]

    return None


# ── model loader ───────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """
    Returns (model, feat_cols, meta, ci).

    If all three ensemble models exist (xgb + lr + mlp), loads an EnsembleModel
    that applies tour-specific weights.  Falls back to XGBoost-only when LR/MLP
    have not yet been trained.

    BootstrapCI is precomputed once on the 2022-2023 val set — all subsequent
    ci.compute() calls are pure numpy (~5 ms each).
    """
    from models.confidence import BootstrapCI

    with open(FEATURE_JSON) as f:
        meta = json.load(f)

    if LR_PKL.exists() and MLP_PKL.exists():
        from models.ensemble import EnsembleModel
        model     = EnsembleModel.load()
        feat_cols = model.feat_cols
        print("[predict] Ensemble model loaded (XGB + LR + MLP)")
    else:
        with open(MODEL_PKL, "rb") as f:
            artefact = pickle.load(f)
        model     = artefact["model"]
        feat_cols = artefact["feature_cols"]
        print("[predict] XGBoost-only model loaded (run models/ensemble.py to enable ensemble)")

    ci = BootstrapCI(model, feat_cols)
    return model, feat_cols, meta, ci


# ── single-match prediction ────────────────────────────────────────────────────

def predict_match(
    ctx,
    model,
    feat_cols: list[str],
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
    ci=None,   # BootstrapCI instance (optional — skipped if None)
) -> dict:
    """
    Build features and run inference for a single match.
    Returns a result dict with prob, edge, confidence, and all logging fields.
    """
    from features.pipeline import build_feature_row

    feat_row = build_feature_row(
        ctx,
        player_a_id   = player_a_id,
        player_b_id   = player_b_id,
        player_a_name = player_a_name,
        player_b_name = player_b_name,
        surface       = surface,
        tournament    = tournament,
        match_date    = match_date,
        best_of       = best_of,
        round_str     = round_str,
        tourney_level = tourney_level,
        tour          = tour,
        opening_odds_a = opening_odds_a,
        opening_odds_b = opening_odds_b,
        closing_odds_a = closing_odds_a,
        closing_odds_b = closing_odds_b,
    )

    # Build feature vector — fill missing features with NaN
    X = pd.DataFrame([feat_row])[
        [c for c in feat_cols if c in feat_row]
    ].apply(pd.to_numeric, errors="coerce")

    # Pad any missing feature columns with NaN
    for c in feat_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feat_cols]

    # Use tour-weighted ensemble probability when available
    from models.ensemble import EnsembleModel
    if isinstance(model, EnsembleModel):
        prob_a = float(model.predict_proba(X, tour=tour, tourney_level=tourney_level)[0, 1])
    else:
        prob_a = float(model.predict_proba(X)[0, 1])
    prob_b = 1.0 - prob_a

    # ── confidence interval ───────────────────────────────────────────────────
    ci_result: dict = {}
    if ci is not None:
        ci_result = ci.compute(X.values[0], prob_a=prob_a)

    # Market edge: model prob − implied prob (from closing if available, else opening)
    edge_a = np.nan
    if closing_odds_a is not None and closing_odds_b is not None:
        total_implied = 1 / closing_odds_a + 1 / closing_odds_b
        implied_a = (1 / closing_odds_a) / total_implied
        edge_a = prob_a - implied_a
    elif opening_odds_a is not None and opening_odds_b is not None:
        total_implied = 1 / opening_odds_a + 1 / opening_odds_b
        implied_a = (1 / opening_odds_a) / total_implied
        edge_a = prob_a - implied_a

    # Top feature drivers (Elo, serve/return, form summary)
    drivers = _key_drivers(feat_row)

    return {
        "player_a":          player_a_name,
        "player_b":          player_b_name,
        "player_a_id":       player_a_id,
        "player_b_id":       player_b_id,
        "surface":           surface,
        "tournament":        tournament,
        "date":              str(match_date.date()),
        "prob_a":            round(prob_a, 4),
        "prob_b":            round(prob_b, 4),
        "prob_low":          ci_result.get("prob_low"),
        "prob_high":         ci_result.get("prob_high"),
        "confidence_width":  ci_result.get("confidence_width"),
        "confidence_tier":   ci_result.get("confidence_tier"),
        "confidence":        _confidence_label(prob_a),
        "model_edge":        round(float(edge_a), 4) if not np.isnan(edge_a) else None,
        "opening_odds_a":    opening_odds_a,
        "opening_odds_b":    opening_odds_b,
        "closing_odds_a":    closing_odds_a,
        "closing_odds_b":    closing_odds_b,
        "key_drivers":       drivers,
        "feat_row":          feat_row,   # kept for CLV logging, stripped before CSV write
    }


def _key_drivers(feat_row: dict) -> str:
    """
    Summarise the top 3 prediction drivers as a human-readable string.
    Based on the most diagnostic feature values.
    """
    parts = []

    elo_diff = feat_row.get("elo_diff", np.nan)
    if not np.isnan(elo_diff) and abs(elo_diff) > 30:
        direction = "Elo +" if elo_diff > 0 else "Elo "
        parts.append(f"{direction}{elo_diff:+.0f}")

    form_a = feat_row.get("a_form_weighted_win_pct", np.nan)
    form_b = feat_row.get("b_form_weighted_win_pct", np.nan)
    if not (np.isnan(form_a) or np.isnan(form_b)):
        diff = form_a - form_b
        if abs(diff) > 0.08:
            parts.append(f"form {diff:+.2f}")

    h2h = feat_row.get("h2h_win_pct_surface", np.nan)
    if not np.isnan(h2h) and abs(h2h - 0.5) > 0.15:
        parts.append(f"H2H {h2h:.0%}")

    hold_a = feat_row.get("a_hold_pct", np.nan)
    hold_b = feat_row.get("b_hold_pct", np.nan)
    if not (np.isnan(hold_a) or np.isnan(hold_b)):
        diff = hold_a - hold_b
        if abs(diff) > 0.06:
            parts.append(f"hold {diff:+.2f}")

    return " | ".join(parts) if parts else "balanced"


# ── batch prediction from odds DataFrame ──────────────────────────────────────

def run_predictions(
    ctx,
    model,
    feat_cols: list[str],
    odds_df: pd.DataFrame,
    dry_run: bool = False,
    ci=None,
) -> pd.DataFrame:
    """
    Run predictions for all matches in odds_df.

    Parameters
    ----------
    ctx       : PipelineContext
    model     : calibrated XGBoost model
    feat_cols : feature column list from feature_list.json
    odds_df   : output of OddsClient.fetch_tennis_odds() (best-bookmaker rows)
    dry_run   : if True, don't write to disk
    ci        : BootstrapCI instance for confidence intervals (optional)

    Returns
    -------
    DataFrame of prediction results (one row per match)
    """
    from utils.odds_fetcher import best_bookmaker_row

    norm_lookup = build_name_lookup(ctx.name_to_id)

    # One row per event (best bookmaker)
    events = best_bookmaker_row(odds_df)

    results: list[dict] = []
    skipped: list[str]  = []

    for _, ev in events.iterrows():
        a_name = str(ev["player_a"])
        b_name = str(ev["player_b"])

        a_id = resolve_player_id(a_name, ctx.name_to_id, norm_lookup)
        b_id = resolve_player_id(b_name, ctx.name_to_id, norm_lookup)

        if a_id is None or b_id is None:
            missing = [n for n, i in [(a_name, a_id), (b_name, b_id)] if i is None]
            skipped.append(f"{a_name} vs {b_name} — unresolved: {missing}")
            continue

        match_date = pd.Timestamp(ev["commence_time"]).tz_localize(None) \
                     if ev["commence_time"].tzinfo else pd.Timestamp(ev["commence_time"])

        try:
            result = predict_match(
                ctx, model, feat_cols,
                player_a_id    = a_id,
                player_b_id    = b_id,
                player_a_name  = a_name,
                player_b_name  = b_name,
                surface        = str(ev["surface"]),
                tournament     = str(ev["tournament"]),
                match_date     = match_date,
                best_of        = int(ev.get("best_of", 3)),
                tourney_level  = str(ev.get("tourney_level", "A")),
                tour           = str(ev.get("tour", "atp")),
                opening_odds_a = float(ev["odds_a"]),
                opening_odds_b = float(ev["odds_b"]),
                ci             = ci,
            )
            results.append(result)
        except Exception as exc:
            skipped.append(f"{a_name} vs {b_name} — error: {exc}")

    if skipped:
        print(f"\n[predict] Skipped {len(skipped)} matches:")
        for s in skipped:
            print(f"  {s}")

    if not results:
        print("[predict] No predictions generated.")
        return pd.DataFrame()

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "feat_row"}
                       for r in results])

    if not dry_run:
        _log_predictions(df)
        _log_clv(results)

    return df


# ── logging ────────────────────────────────────────────────────────────────────

def _log_predictions(df: pd.DataFrame) -> None:
    """Append to output/predictions.csv."""
    cols = [
        "date", "tournament", "surface",
        "player_a", "player_b",
        "prob_a", "prob_b",
        "prob_low", "prob_high", "confidence_width", "confidence_tier",
        "confidence", "model_edge",
        "opening_odds_a", "opening_odds_b",
        "closing_odds_a", "closing_odds_b",
        "key_drivers",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    if PREDICTIONS_CSV.exists():
        out.to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)
    else:
        out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[predict] Logged {len(out)} predictions → {PREDICTIONS_CSV}")


def _log_clv(results: list[dict]) -> None:
    """Append opening entries to output/clv_tracker.csv (result filled post-match)."""
    rows = []
    for r in results:
        feat = r.get("feat_row", {})
        open_implied_a = np.nan
        if r["opening_odds_a"] and r["opening_odds_b"]:
            total = 1 / r["opening_odds_a"] + 1 / r["opening_odds_b"]
            open_implied_a = round((1 / r["opening_odds_a"]) / total, 4)

        rows.append({
            "date":              r["date"],
            "tournament":        r["tournament"],
            "surface":           r["surface"],
            "player_a":          r["player_a"],
            "player_b":          r["player_b"],
            "model_prob_a":      r["prob_a"],
            "opening_implied_a": open_implied_a,
            "closing_implied_a": np.nan,   # filled post-match via update_clv_result()
            "sharp_flag":        feat.get("sharp_flag", np.nan),
            "movement_magnitude": feat.get("movement_magnitude", np.nan),
            "clv_delta":         np.nan,   # filled post-match
            "result":            np.nan,   # filled post-match
        })

    clv_df = pd.DataFrame(rows)
    if CLV_CSV.exists():
        clv_df.to_csv(CLV_CSV, mode="a", header=False, index=False)
    else:
        clv_df.to_csv(CLV_CSV, index=False)
    print(f"[predict] Logged {len(clv_df)} CLV entries → {CLV_CSV}")


# ── display ────────────────────────────────────────────────────────────────────

def print_predictions(df: pd.DataFrame) -> None:
    """Pretty-print prediction results to stdout."""
    from models.confidence import format_ci, TIER_SHARP, TIER_MODERATE

    if df.empty:
        return

    has_ci = "prob_low" in df.columns and df["prob_low"].notna().any()

    print(f"\n{'='*72}")
    print(f"  TENNIS PREDICTIONS  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")

    # Sort by confidence tier (SHARP first) then by confidence width ascending
    display = df.copy()
    if has_ci:
        tier_order = {"SHARP": 0, "MODERATE": 1, "WIDE": 2}
        display["_tier_ord"] = display["confidence_tier"].map(tier_order).fillna(3)
        display["_width"]    = pd.to_numeric(display.get("confidence_width"), errors="coerce")
        display = display.sort_values(["_tier_ord", "_width"]).drop(
            columns=["_tier_ord", "_width"]
        )

    for _, row in display.iterrows():
        prob_a = float(row["prob_a"])
        prob_b = float(row["prob_b"])

        # Probability lines
        if has_ci and pd.notna(row.get("prob_low")):
            pl  = float(row["prob_low"])
            ph  = float(row["prob_high"])
            tier = str(row.get("confidence_tier", ""))
            # CI is for player A; invert for player B
            line_a = format_ci(row["player_a"], prob_a, pl,       ph,       tier)
            line_b = format_ci(row["player_b"], prob_b, 1.0 - ph, 1.0 - pl, tier)
        else:
            line_a = f"{row['player_a']} {prob_a:.1%}"
            line_b = f"{row['player_b']} {prob_b:.1%}"

        edge_str = ""
        if pd.notna(row.get("model_edge")):
            edge_str = f"  edge {row['model_edge']:+.3f}"

        odds_str = ""
        if pd.notna(row.get("opening_odds_a")):
            odds_str = f"  [{row['opening_odds_a']:.2f} / {row['opening_odds_b']:.2f}]"

        print(f"\n  {row['tournament']} ({str(row['surface']).upper()})")
        print(f"  {line_a}")
        print(f"  {line_b}")
        if edge_str or odds_str:
            print(f"  {(edge_str + odds_str).strip()}")
        print(f"  Drivers: {row.get('key_drivers', '')}")

    # ── summary footer ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {len(df)} matches", end="")
    if has_ci and "confidence_tier" in df.columns:
        sharp    = (df["confidence_tier"] == TIER_SHARP).sum()
        moderate = (df["confidence_tier"] == TIER_MODERATE).sum()
        wide     = (df["confidence_tier"] == df["confidence_tier"].map(
                        lambda x: x if x == "WIDE" else "")).sum()
        wide     = (df["confidence_tier"] == "WIDE").sum()
        print(f"  |  SHARP: {sharp}  MODERATE: {moderate}  WIDE: {wide}", end="")
    print()
    if "model_edge" in df.columns:
        edges = pd.to_numeric(df["model_edge"], errors="coerce").dropna()
        if not edges.empty:
            print(f"  Positive edge (>3pp): {(edges > 0.03).sum()} matches")
    print(f"{'='*72}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tennis win probability predictor")
    parser.add_argument("--offline",       action="store_true",
                        help="Use latest cached odds snapshot (no API call)")
    parser.add_argument("--snapshot",      type=str, default=None,
                        help="Path to specific odds snapshot CSV")
    parser.add_argument("--all-upcoming",  action="store_true",
                        help="Fetch all upcoming matches (not just tomorrow)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print predictions without writing to disk")
    parser.add_argument("--bookmakers",    nargs="+", default=None,
                        help="Bookmaker keys to request (default: all in region)")

    # Single-match override
    parser.add_argument("--match",    nargs=2, metavar=("PLAYER_A", "PLAYER_B"),
                        help="Single match override (bypasses odds feed)")
    parser.add_argument("--surface",      default="hard")
    parser.add_argument("--tournament",   default="Unknown")
    parser.add_argument("--date",         default=None,
                        help="Match date YYYY-MM-DD (default: today)")
    parser.add_argument("--tour",         default="atp")
    parser.add_argument("--best-of",      type=int, default=3)
    parser.add_argument("--odds-a",       type=float, default=None)
    parser.add_argument("--odds-b",       type=float, default=None)

    args = parser.parse_args()

    print("[predict] Loading model...")
    model, feat_cols, meta, ci = load_model()

    print("[predict] Loading pipeline context...")
    from features.pipeline import PipelineContext
    ctx = PipelineContext.load()

    # ── single-match mode ──────────────────────────────────────────────────────
    if args.match:
        a_name, b_name = args.match
        norm_lookup = build_name_lookup(ctx.name_to_id)
        a_id = resolve_player_id(a_name, ctx.name_to_id, norm_lookup)
        b_id = resolve_player_id(b_name, ctx.name_to_id, norm_lookup)

        if a_id is None:
            print(f"[predict] ERROR: could not resolve player ID for '{a_name}'")
            sys.exit(1)
        if b_id is None:
            print(f"[predict] ERROR: could not resolve player ID for '{b_name}'")
            sys.exit(1)

        match_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today()

        result = predict_match(
            ctx, model, feat_cols,
            player_a_id   = a_id,
            player_b_id   = b_id,
            player_a_name = a_name,
            player_b_name = b_name,
            surface       = args.surface,
            tournament    = args.tournament,
            match_date    = match_date,
            best_of       = args.best_of,
            tour          = args.tour,
            opening_odds_a = args.odds_a,
            opening_odds_b = args.odds_b,
            ci             = ci,
        )

        df = pd.DataFrame([{k: v for k, v in result.items() if k != "feat_row"}])
        print_predictions(df)

        if not args.dry_run:
            _log_predictions(df)
            _log_clv([result])
        return

    # ── batch mode from odds feed ──────────────────────────────────────────────
    if args.snapshot:
        odds_df = pd.read_csv(args.snapshot, parse_dates=["commence_time"])
        print(f"[predict] Loaded snapshot: {args.snapshot}  ({len(odds_df):,} rows)")

    elif args.offline:
        from utils.odds_fetcher import load_latest_snapshot
        odds_df = load_latest_snapshot()

    elif args.all_upcoming:
        from utils.odds_fetcher import OddsClient
        print("[predict] Fetching all upcoming matches...")
        client  = OddsClient(api_key=os.environ.get("ODDS_API_KEY"))
        odds_df = client.fetch_tennis_odds(bookmakers=args.bookmakers, save=True)

    else:
        # Default: generate tomorrow's slate and predict on it
        from utils.slate_generator import get_tomorrow_slate, slate_to_odds_df
        print("[predict] Generating tomorrow's match slate...")
        slate   = get_tomorrow_slate(bookmakers=args.bookmakers, save=True)
        odds_df = slate_to_odds_df(slate) if not slate.empty else slate

    if odds_df.empty:
        print("[predict] No odds data — nothing to predict.")
        sys.exit(0)

    n_events = odds_df["event_id"].nunique() if "event_id" in odds_df.columns else len(odds_df)
    print(f"[predict] Running predictions for {n_events} matches...")
    results_df = run_predictions(ctx, model, feat_cols, odds_df,
                                 dry_run=args.dry_run, ci=ci)
    print_predictions(results_df)


if __name__ == "__main__":
    main()
