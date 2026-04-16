"""
utils/live_poller.py
────────────────────
Live odds polling engine.

Polls The Odds API on a per-game schedule based on time-to-game, and
re-fetches commence_time on every poll cycle so that late starts (common
in tennis) are detected and the polling bucket is updated automatically.

Polling buckets
───────────────
  >12hr      → every 60 min
  1hr–12hr   → every 15 min
  <1hr       → every  2 min  (also used once game has started)
  active     → every  2 min  (past commence_time, within +3hr grace window)
  expired    → skipped        (commence_time + 3hr has passed)

Time-update detection
─────────────────────
  On every poll the API's commence_time is compared to the stored value.
  If it has shifted, the tracker is updated and a log line is emitted:
    ⏰ Time update: Alcaraz vs Djokovic moved from 14:00 → 15:34

Outputs
───────
  output/line_snapshots.csv   — odds at every poll point
  output/game_poll_tracker.json — current tracker state (resumable)

Usage
─────
  # Run indefinitely (Ctrl-C to stop)
  python -m utils.live_poller

  # Run for a fixed number of seconds
  python -m utils.live_poller --duration 3600

  # Dry-run: show which games would be tracked, then exit
  python -m utils.live_poller --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRACKER_PATH   = OUTPUT_DIR / "game_poll_tracker.json"
SNAPSHOTS_PATH = OUTPUT_DIR / "line_snapshots.csv"

# ── polling intervals per bucket (seconds) ────────────────────────────────────
BUCKET_INTERVALS: dict[str, int] = {
    ">12hr":    3600,   # 60 min
    "1hr-12hr":  900,   # 15 min
    "<1hr":      120,   #  2 min
    "active":    120,   #  2 min (game in progress / grace window)
}

GRACE_HOURS = 3          # keep polling this many hours after commence_time
LOOP_SLEEP  = 30         # main loop heartbeat (seconds)

# ── snapshot CSV columns ──────────────────────────────────────────────────────
SNAPSHOT_COLS = [
    "poll_time", "event_id", "player_a", "player_b",
    "tournament", "surface", "tour",
    "commence_time",          # as returned by the API this cycle
    "odds_a", "odds_b", "bookmaker",
    "bucket",
]


# ── bucket helpers ────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _bucket(commence: datetime) -> str:
    """
    Classify a game into a polling bucket based on its commence_time.
    Returns 'expired' if the game is past the grace window.
    """
    now  = _now_utc()
    diff = (commence - now).total_seconds()

    if diff > 12 * 3600:
        return ">12hr"
    if diff > 3600:
        return "1hr-12hr"
    if diff > 0:
        return "<1hr"
    # Past commence_time
    elapsed = (now - commence).total_seconds()
    if elapsed < GRACE_HOURS * 3600:
        return "active"
    return "expired"


def _next_poll_time(bucket: str, last_poll: datetime | None = None) -> datetime:
    """Compute the next poll datetime for a given bucket."""
    base     = last_poll or _now_utc()
    interval = BUCKET_INTERVALS.get(bucket, 3600)
    return base + timedelta(seconds=interval)


# ── tracker persistence ───────────────────────────────────────────────────────

def _load_tracker() -> dict[str, dict]:
    """Load tracker from disk. Returns {} if not found or corrupt."""
    if not TRACKER_PATH.exists():
        return {}
    try:
        raw = json.loads(TRACKER_PATH.read_text())
        # Deserialise datetime strings
        for entry in raw.values():
            for key in ("commence_time", "last_poll", "next_poll"):
                if entry.get(key):
                    entry[key] = datetime.fromisoformat(entry[key])
        return raw
    except Exception:
        return {}


def _save_tracker(tracker: dict[str, dict]) -> None:
    """Persist tracker to disk."""
    serialisable = {}
    for event_id, entry in tracker.items():
        row = dict(entry)
        for key in ("commence_time", "last_poll", "next_poll"):
            if isinstance(row.get(key), datetime):
                row[key] = row[key].isoformat()
        serialisable[event_id] = row
    TRACKER_PATH.write_text(json.dumps(serialisable, indent=2))


# ── snapshot writer ───────────────────────────────────────────────────────────

def _write_snapshot(rows: list[dict]) -> None:
    """Append rows to line_snapshots.csv."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLS)
    write_header = not SNAPSHOTS_PATH.exists()
    df.to_csv(SNAPSHOTS_PATH, mode="a", index=False, header=write_header)


# ── odds fetch helpers ────────────────────────────────────────────────────────

def _fetch_all_odds(client: Any) -> pd.DataFrame:
    """
    Fetch fresh odds for all active tennis sport keys.
    Returns a DataFrame in the same format as OddsClient.fetch_tennis_odds().
    """
    return client.fetch_tennis_odds(save=False)


def _extract_commence(odds_df: pd.DataFrame, event_id: str) -> datetime | None:
    """
    Pull the latest commence_time for event_id from a fresh odds snapshot.
    Returns None if the event is no longer in the feed (completed / delisted).
    """
    match = odds_df[odds_df["event_id"] == event_id]
    if match.empty:
        return None
    ct = match.iloc[0]["commence_time"]
    if pd.isna(ct):
        return None
    if isinstance(ct, datetime):
        return ct.astimezone(timezone.utc) if ct.tzinfo else ct.replace(tzinfo=timezone.utc)
    return pd.Timestamp(ct).tz_convert("UTC").to_pydatetime()


def _best_odds_row(odds_df: pd.DataFrame, event_id: str) -> dict | None:
    """
    Return best-book odds row for event_id from an already-fetched DataFrame.
    Uses the same preferred-book ranking as best_bookmaker_row().
    """
    sys.path.insert(0, str(ROOT))
    from utils.odds_fetcher import best_bookmaker_row

    subset = odds_df[odds_df["event_id"] == event_id]
    if subset.empty:
        return None
    best = best_bookmaker_row(subset)
    if best.empty:
        return None
    return best.iloc[0].to_dict()


# ── core poll cycle ───────────────────────────────────────────────────────────

def _poll_game(
    event_id: str,
    tracker: dict[str, dict],
    odds_df: pd.DataFrame,
) -> dict | None:
    """
    Process one game in the tracker against fresh odds_df.

    - Detects commence_time changes and logs them.
    - Updates the tracker entry in-place.
    - Returns a snapshot dict to be written, or None if no data.
    """
    entry          = tracker[event_id]
    stored_commence = entry["commence_time"]
    player_a       = entry.get("player_a", "?")
    player_b       = entry.get("player_b", "?")

    # ── 1. Re-fetch commence_time from API response ───────────────────────────
    fresh_commence = _extract_commence(odds_df, event_id)

    if fresh_commence is not None and fresh_commence != stored_commence:
        old_str = stored_commence.strftime("%H:%M") if stored_commence else "?"
        new_str = fresh_commence.strftime("%H:%M")
        print(
            f"⏰ Time update: {player_a} vs {player_b} "
            f"moved from {old_str} → {new_str}"
        )
        entry["commence_time"] = fresh_commence
        stored_commence        = fresh_commence

    # ── 2. Recompute bucket from updated commence_time ────────────────────────
    new_bucket = _bucket(stored_commence)
    old_bucket = entry.get("bucket", "")

    if new_bucket != old_bucket:
        if new_bucket != "expired":
            print(
                f"[poll] {player_a} vs {player_b}: "
                f"bucket {old_bucket} → {new_bucket} "
                f"(interval now {BUCKET_INTERVALS.get(new_bucket, '?')}s)"
            )
        entry["bucket"] = new_bucket

    # ── 3. Skip expired games ─────────────────────────────────────────────────
    if new_bucket == "expired":
        return None

    # ── 4. Pull best odds row ─────────────────────────────────────────────────
    odds_row = _best_odds_row(odds_df, event_id)

    now = _now_utc()
    entry["last_poll"] = now
    entry["next_poll"] = _next_poll_time(new_bucket, now)

    if odds_row is None:
        return None

    # Update stored odds
    entry["odds_a"]    = odds_row.get("odds_a",    entry.get("odds_a"))
    entry["odds_b"]    = odds_row.get("odds_b",    entry.get("odds_b"))
    entry["bookmaker"] = odds_row.get("bookmaker", entry.get("bookmaker", ""))

    return {
        "poll_time":    now.isoformat(),
        "event_id":     event_id,
        "player_a":     player_a,
        "player_b":     player_b,
        "tournament":   entry.get("tournament", ""),
        "surface":      entry.get("surface", ""),
        "tour":         entry.get("tour", ""),
        "commence_time": stored_commence.isoformat(),
        "odds_a":        entry["odds_a"],
        "odds_b":        entry["odds_b"],
        "bookmaker":     entry["bookmaker"],
        "bucket":        new_bucket,
    }


# ── slate → tracker initialisation ───────────────────────────────────────────

def _seed_tracker(
    tracker: dict[str, dict],
    odds_df: pd.DataFrame,
) -> int:
    """
    Add any new events from odds_df that aren't already in the tracker.
    Returns the number of new games added.
    """
    sys.path.insert(0, str(ROOT))
    from utils.odds_fetcher import best_bookmaker_row

    best = best_bookmaker_row(odds_df)
    added = 0

    for _, row in best.iterrows():
        eid = str(row["event_id"])
        if eid in tracker:
            continue

        ct_raw = row.get("commence_time")
        if pd.isna(ct_raw):
            continue
        ct = pd.Timestamp(ct_raw).tz_convert("UTC").to_pydatetime()

        bkt = _bucket(ct)
        if bkt == "expired":
            continue

        tracker[eid] = {
            "event_id":    eid,
            "player_a":    row.get("player_a", "?"),
            "player_b":    row.get("player_b", "?"),
            "tournament":  row.get("tournament", ""),
            "surface":     row.get("surface", ""),
            "tour":        row.get("tour", ""),
            "commence_time": ct,
            "odds_a":      row.get("odds_a"),
            "odds_b":      row.get("odds_b"),
            "bookmaker":   row.get("bookmaker", ""),
            "bucket":      bkt,
            "last_poll":   None,
            "next_poll":   _now_utc(),   # poll immediately on first cycle
        }
        print(
            f"[poll] Added  {row.get('player_a','?')} vs {row.get('player_b','?')} "
            f"  [{bkt}]  {ct.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        added += 1

    return added


# ── main polling loop ─────────────────────────────────────────────────────────

def run_poller(
    duration: float | None = None,
    dry_run:  bool         = False,
    api_key:  str | None   = None,
) -> None:
    """
    Run the live polling loop.

    Parameters
    ----------
    duration : stop after this many seconds (None = run forever)
    dry_run  : print initial tracker, then exit without polling
    api_key  : override ODDS_API_KEY env var
    """
    sys.path.insert(0, str(ROOT))
    from utils.odds_fetcher import OddsClient

    client  = OddsClient(api_key=api_key)
    tracker = _load_tracker()

    print(f"[poll] Starting live poller  (grace={GRACE_HOURS}hr, loop={LOOP_SLEEP}s)")
    print(f"[poll] Snapshots → {SNAPSHOTS_PATH}")
    print(f"[poll] Tracker   → {TRACKER_PATH}")
    print()

    # Initial seed
    print("[poll] Fetching initial odds...")
    odds_df = _fetch_all_odds(client)
    if not odds_df.empty:
        n = _seed_tracker(tracker, odds_df)
        print(f"[poll] {n} new game(s) added.  Tracker size: {len(tracker)}")
    else:
        print("[poll] No odds data returned from API.")

    _save_tracker(tracker)

    if dry_run:
        print()
        active = {k: v for k, v in tracker.items() if v.get("bucket") != "expired"}
        print(f"[poll] DRY-RUN — {len(active)} game(s) would be tracked:\n")
        for entry in active.values():
            ct  = entry["commence_time"]
            ct_str = ct.strftime("%Y-%m-%d %H:%M UTC") if isinstance(ct, datetime) else str(ct)
            print(
                f"  {entry['player_a']:25s} vs {entry['player_b']:25s}"
                f"  [{entry['bucket']:10s}]  {ct_str}"
            )
        return

    start_time = time.monotonic()

    while True:
        if duration is not None and (time.monotonic() - start_time) >= duration:
            print("[poll] Duration reached — stopping.")
            break

        now = _now_utc()

        # Determine which games are due for a poll this cycle
        due = [
            eid for eid, entry in tracker.items()
            if entry.get("bucket") != "expired"
            and (entry.get("next_poll") or now) <= now
        ]

        if due:
            print(f"[poll] {now.strftime('%H:%M:%S UTC')} — polling {len(due)} game(s)...")
            # Re-fetch odds once for this cycle (shared across all due games)
            try:
                odds_df = _fetch_all_odds(client)
            except Exception as exc:
                print(f"[poll] ERROR fetching odds: {exc}")
                odds_df = pd.DataFrame()

            if not odds_df.empty:
                # Seed any new games that appeared since last cycle
                _seed_tracker(tracker, odds_df)

            snapshots = []
            for eid in due:
                if eid not in tracker:
                    continue
                snap = _poll_game(eid, tracker, odds_df)
                if snap:
                    snapshots.append(snap)

            _write_snapshot(snapshots)
            _save_tracker(tracker)

            # Prune expired games from memory
            expired = [eid for eid, e in tracker.items() if e.get("bucket") == "expired"]
            for eid in expired:
                e = tracker.pop(eid)
                print(f"[poll] Expired: {e.get('player_a','?')} vs {e.get('player_b','?')}")

        time.sleep(LOOP_SLEEP)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live tennis odds poller")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run forever)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print tracker state and exit without polling")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Override ODDS_API_KEY env var")
    args = parser.parse_args()

    # Load .env
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    run_poller(
        duration = args.duration,
        dry_run  = args.dry_run,
        api_key  = args.api_key,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
