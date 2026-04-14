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
    """Returns (model, feat_cols)."""
    with open(MODEL_PKL, "rb") as f:
        artefact = pickle.load(f)
    with open(FEATURE_JSON) as f:
        meta = json.load(f)
    return artefact["model"], artefact["feature_cols"], meta


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

    prob_a = float(model.predict_proba(X)[0, 1])
    prob_b = 1.0 - prob_a

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
        "player_a":      player_a_name,
        "player_b":      player_b_name,
        "player_a_id":   player_a_id,
        "player_b_id":   player_b_id,
        "surface":       surface,
        "tournament":    tournament,
        "date":          str(match_date.date()),
        "prob_a":        round(prob_a, 4),
        "prob_b":        round(prob_b, 4),
        "confidence":    _confidence_label(prob_a),
        "model_edge":    round(float(edge_a), 4) if not np.isnan(edge_a) else None,
        "opening_odds_a": opening_odds_a,
        "opening_odds_b": opening_odds_b,
        "closing_odds_a": closing_odds_a,
        "closing_odds_b": closing_odds_b,
        "key_drivers":   drivers,
        "feat_row":      feat_row,   # kept for CLV logging, stripped before CSV write
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
        "prob_a", "prob_b", "confidence", "model_edge",
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
    if df.empty:
        return

    print(f"\n{'='*72}")
    print(f"  TENNIS PREDICTIONS  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")

    for _, row in df.sort_values("prob_a", ascending=False).iterrows():
        edge_str = f"  edge {row['model_edge']:+.3f}" if pd.notna(row.get("model_edge")) else ""
        odds_str = ""
        if pd.notna(row.get("opening_odds_a")):
            odds_str = f"  [{row['opening_odds_a']:.2f} / {row['opening_odds_b']:.2f}]"

        print(f"\n  {row['tournament']} ({row['surface'].upper()})")
        print(f"  {row['player_a']} vs {row['player_b']}")
        print(f"  P(A wins): {row['prob_a']:.1%}   P(B wins): {row['prob_b']:.1%}"
              f"   conf={row['confidence']}{edge_str}{odds_str}")
        print(f"  Drivers: {row.get('key_drivers', '')}")

    print(f"\n{'='*72}")
    print(f"  {len(df)} matches  |  "
          f"High conf: {(df['confidence']=='high').sum()}  |  "
          f"Medium: {(df['confidence']=='medium').sum()}  |  "
          f"Low: {(df['confidence']=='low').sum()}")
    if "model_edge" in df.columns:
        edges = df["model_edge"].dropna()
        if not edges.empty:
            pos_edge = (edges > 0.03).sum()
            print(f"  Positive edge (>3pp): {pos_edge} matches")
    print(f"{'='*72}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tennis win probability predictor")
    parser.add_argument("--offline",    action="store_true",
                        help="Use latest cached odds snapshot (no API call)")
    parser.add_argument("--snapshot",   type=str, default=None,
                        help="Path to specific odds snapshot CSV")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print predictions without writing to disk")
    parser.add_argument("--bookmakers", nargs="+", default=None,
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
    model, feat_cols, meta = load_model()

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

    else:
        from utils.odds_fetcher import OddsClient
        print("[predict] Fetching live odds...")
        client = OddsClient()
        odds_df = client.fetch_tennis_odds(
            bookmakers = args.bookmakers,
            save       = True,
        )

    if odds_df.empty:
        print("[predict] No odds data — nothing to predict.")
        sys.exit(0)

    print(f"[predict] Running predictions for {odds_df['event_id'].nunique()} matches...")
    results_df = run_predictions(ctx, model, feat_cols, odds_df,
                                 dry_run=args.dry_run)
    print_predictions(results_df)


if __name__ == "__main__":
    main()
