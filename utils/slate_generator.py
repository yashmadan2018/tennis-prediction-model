"""
Tomorrow's match slate generator.

Pulls upcoming tennis matches from The Odds API, filters to those
commencing tomorrow (local date), and writes a clean CSV to
output/slate_YYYY-MM-DD.csv.

Output columns
--------------
  match_id        : {tour}_{event_id[:10]}
  player_a        : title-cased name
  player_b        : title-cased name
  surface         : hard | clay | grass
  tournament      : human-readable tournament name
  tour            : atp | wta
  round           : inferred best-effort (Unknown if can't determine)
  best_of         : 3 or 5
  tourney_level   : G | M | A | P | PM | I | C
  commence_time   : ISO 8601 UTC
  opening_odds_a  : decimal odds player A (best available book)
  opening_odds_b  : decimal odds player B

Usage
-----
  python utils/slate_generator.py               # tomorrow's slate
  python utils/slate_generator.py --date 2025-06-01  # specific date
  python utils/slate_generator.py --all         # all upcoming matches
  python utils/slate_generator.py --load        # print latest saved slate

Integration
-----------
  from utils.slate_generator import get_tomorrow_slate
  df = get_tomorrow_slate()   # returns normalized DataFrame
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SLATE_COLS = [
    "match_id", "event_id",
    "player_a", "player_b",
    "surface", "tournament", "tour",
    "round", "best_of", "tourney_level",
    "commence_time",
    "opening_odds_a", "opening_odds_b",
    "bookmaker",
]


# ── round inference ────────────────────────────────────────────────────────────
# The Odds API h2h endpoint doesn't include round information.
# We infer a best-effort round from the number of days into a known
# tournament schedule.  Falls back to "Unknown".

_SLAM_ROUNDS_BY_DAY = {
    1: "R128", 2: "R128",
    3: "R64",  4: "R64",
    5: "R32",  6: "R32",
    7: "R16",  8: "R16",
    9: "QF",   10: "QF",
    11: "SF",  12: "SF",
    13: "F",
}
_ATP500_ROUNDS_BY_DAY = {
    1: "R32", 2: "R32",
    3: "R16", 4: "R16",
    5: "QF",  6: "QF",
    7: "SF",  8: "F",
}
_ATP250_ROUNDS_BY_DAY = {
    1: "R32", 2: "R16",
    3: "QF",  4: "SF",
    5: "F",
}


def _infer_round(tourney_level: str, day_of_tournament: int | None) -> str:
    """Best-effort round from tournament level and day number (1-indexed)."""
    if day_of_tournament is None:
        return "Unknown"
    d = int(day_of_tournament)
    if tourney_level == "G":
        return _SLAM_ROUNDS_BY_DAY.get(d, "Unknown")
    if tourney_level in ("M", "PM"):
        return _ATP500_ROUNDS_BY_DAY.get(d, "Unknown")
    return _ATP250_ROUNDS_BY_DAY.get(d, "Unknown")


# ── slate builder ──────────────────────────────────────────────────────────────

def _build_slate(odds_df: pd.DataFrame, target_date: date | None) -> pd.DataFrame:
    """
    Filter odds_df to matches commencing on target_date (local system time)
    and format as a clean slate DataFrame.

    Dates are compared in the local system timezone so that matches near
    midnight UTC are bucketed correctly for the user's locale.
    Pass target_date=None to skip date filtering (return all rows).
    """
    if odds_df.empty:
        return pd.DataFrame(columns=SLATE_COLS)

    df = odds_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["commence_time"]):
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    # Convert to local system time before extracting the date
    import datetime as _dt
    local_tz = _dt.datetime.now(_dt.timezone.utc).astimezone().tzinfo
    df["commence_date"] = df["commence_time"].dt.tz_convert(local_tz).dt.date

    if target_date is not None:
        slate = df[df["commence_date"] == target_date].copy()
    else:
        slate = df.copy()
    if slate.empty:
        return pd.DataFrame(columns=SLATE_COLS)

    # Build match_id
    slate["match_id"] = (
        slate["tour"].str.lower() + "_" +
        slate["event_id"].astype(str).str[:10]
    )

    # Round — infer from tourney_level (no day info available from API)
    slate["round"] = slate["tourney_level"].apply(
        lambda lvl: _infer_round(str(lvl), None)
    )

    # Rename odds columns
    slate = slate.rename(columns={
        "odds_a": "opening_odds_a",
        "odds_b": "opening_odds_b",
    })

    # Select and order
    present = [c for c in SLATE_COLS if c in slate.columns]
    slate   = slate[present].reset_index(drop=True)
    return slate


# ── main entry points ──────────────────────────────────────────────────────────

def get_slate(
    target_date: date | str | None = None,
    api_key: str | None = None,
    bookmakers: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch and return the match slate for target_date (default: tomorrow).

    Parameters
    ----------
    target_date : date or 'YYYY-MM-DD' string (default: tomorrow UTC)
    api_key     : overrides ODDS_API_KEY env var
    bookmakers  : bookmaker keys to request (default: all in region)
    save        : write to output/slate_YYYY-MM-DD.csv

    Returns
    -------
    DataFrame with SLATE_COLS columns, one row per match.
    """
    sys.path.insert(0, str(ROOT))
    from utils.odds_fetcher import OddsClient, best_bookmaker_row

    if target_date is None:
        target_date = date.today()   # default: today (local date)
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    date_str = target_date.isoformat()
    print(f"[slate] Fetching slate for {date_str}...")

    client   = OddsClient(api_key=api_key)
    odds_raw = client.fetch_tennis_odds(bookmakers=bookmakers, save=True)

    if odds_raw.empty:
        print(f"[slate] No odds data returned — empty slate for {date_str}.")
        return pd.DataFrame(columns=SLATE_COLS)

    # One row per event (best bookmaker)
    odds_best = best_bookmaker_row(odds_raw)
    slate     = _build_slate(odds_best, target_date)

    if slate.empty:
        print(f"[slate] No matches found commencing on {date_str}.")
        return slate

    if save:
        out_path = OUTPUT_DIR / f"slate_{date_str}.csv"
        slate.to_csv(out_path, index=False)
        print(f"[slate] Saved {len(slate)} matches → {out_path}")

    return slate


def get_tomorrow_slate(
    api_key: str | None = None,
    bookmakers: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: slate for tomorrow."""
    return get_slate(
        target_date = date.today() + timedelta(days=1),
        api_key     = api_key,
        bookmakers  = bookmakers,
        save        = save,
    )


def get_all_upcoming_slate(
    api_key: str | None = None,
    bookmakers: list[str] | None = None,
    save: bool = False,
) -> pd.DataFrame:
    """Return all upcoming matches (not filtered to a single date)."""
    sys.path.insert(0, str(ROOT))
    from utils.odds_fetcher import OddsClient, best_bookmaker_row

    client   = OddsClient(api_key=api_key)
    odds_raw = client.fetch_tennis_odds(bookmakers=bookmakers, save=save)
    if odds_raw.empty:
        return pd.DataFrame(columns=SLATE_COLS)

    odds_best = best_bookmaker_row(odds_raw)
    return _build_slate(odds_best, target_date=None)   # no date filter


def load_slate(date_str: str | None = None) -> pd.DataFrame:
    """
    Load a previously saved slate CSV from output/.

    Parameters
    ----------
    date_str : 'YYYY-MM-DD' (default: today's date, then yesterday's as fallback)
    """
    if date_str:
        path = OUTPUT_DIR / f"slate_{date_str}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No slate found for {date_str}: {path}")
        return pd.read_csv(path, parse_dates=["commence_time"])

    # Search for most recent
    slates = sorted(OUTPUT_DIR.glob("slate_????-??-??.csv"))
    if not slates:
        raise FileNotFoundError(f"No slate files found in {OUTPUT_DIR}")
    path = slates[-1]
    print(f"[slate] Loading: {path.name}")
    return pd.read_csv(path, parse_dates=["commence_time"])


def slate_to_odds_df(slate: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a slate DataFrame back to the odds_df format expected by
    predict.run_predictions() — adds required columns with defaults.
    """
    df = slate.copy()
    rename = {
        "opening_odds_a": "odds_a",
        "opening_odds_b": "odds_b",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure required columns exist
    for col, default in [
        ("bookmaker",     "unknown"),
        ("snapshot_time", pd.Timestamp.utcnow().isoformat()),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


# ── no-date filter version of _build_slate ─────────────────────────────────────

def _build_slate(odds_df: pd.DataFrame, target_date) -> pd.DataFrame:  # noqa: F811
    """Overload: if target_date is None, return all rows."""
    if odds_df.empty:
        return pd.DataFrame(columns=SLATE_COLS)

    df = odds_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["commence_time"]):
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    if target_date is not None:
        df["commence_date"] = df["commence_time"].dt.tz_convert("UTC").dt.date
        df = df[df["commence_date"] == target_date]

    if df.empty:
        return pd.DataFrame(columns=SLATE_COLS)

    df["match_id"] = (
        df["tour"].str.lower() + "_" +
        df["event_id"].astype(str).str[:10]
    )
    df["round"] = df["tourney_level"].apply(lambda lvl: _infer_round(str(lvl), None))
    df = df.rename(columns={"odds_a": "opening_odds_a", "odds_b": "opening_odds_b"})

    present = [c for c in SLATE_COLS if c in df.columns]
    return df[present].reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tomorrow's tennis match slate")
    parser.add_argument("--date",  type=str, default=None,
                        help="Target date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--all",   action="store_true",
                        help="Fetch all upcoming matches, not just tomorrow")
    parser.add_argument("--load",  action="store_true",
                        help="Load and print the latest saved slate (no API call)")
    parser.add_argument("--bookmakers", nargs="+", default=None,
                        help="Bookmaker keys to request")
    args = parser.parse_args()

    if args.load:
        slate = load_slate(args.date)
    elif args.all:
        slate = get_all_upcoming_slate(bookmakers=args.bookmakers)
    else:
        target = date.fromisoformat(args.date) if args.date else None
        slate  = get_slate(target_date=target, bookmakers=args.bookmakers)

    if slate.empty:
        print("[slate] Empty slate.")
        return

    print(f"\n── Slate ({len(slate)} matches) ──────────────────────────────────")
    for _, row in slate.iterrows():
        odds_str = ""
        if "opening_odds_a" in row and pd.notna(row["opening_odds_a"]):
            odds_str = f"  [{row['opening_odds_a']:.2f} / {row['opening_odds_b']:.2f}]"
        print(f"  {row.get('tournament','?')} ({row.get('surface','?').upper()}) "
              f"| {row['player_a']} vs {row['player_b']}{odds_str}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
