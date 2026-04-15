"""
utils/alert_runner.py
─────────────────────
Load predictions.csv, filter to edge matches, send SMS alerts for new ones.

Deduplication: tracks sent alerts in output/sent_alerts.csv.
A match is considered already-alerted if the same (date, player_a, player_b)
tuple appears in sent_alerts.csv from today.

Usage
-----
  # Send real SMS for all new edge alerts today
  python -m utils.alert_runner

  # Print what would be sent without hitting Twilio
  python -m utils.alert_runner --dry-run

  # Override edge threshold (% points as integer)
  python -m utils.alert_runner --threshold 6

  # Point at a specific predictions file
  python -m utils.alert_runner --predictions /path/to/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT            = Path(__file__).parent.parent
PREDICTIONS_CSV = ROOT / "output" / "predictions.csv"
SENT_ALERTS_CSV = ROOT / "output" / "sent_alerts.csv"
CONFIG_JSON     = ROOT / "config.json"

_SENT_COLS = ["sent_date", "match_date", "player_a", "player_b", "tournament"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config_threshold(default: float = 4.0) -> float:
    """Read edge_threshold from config.json; fall back to default."""
    if CONFIG_JSON.exists():
        try:
            with open(CONFIG_JSON) as f:
                return float(json.load(f).get("edge_threshold", default))
        except Exception:
            pass
    return default


def _load_sent_today() -> set[tuple]:
    """
    Return set of (match_date_str, player_a, player_b) already alerted today.
    """
    today = date.today().isoformat()
    if not SENT_ALERTS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(SENT_ALERTS_CSV, dtype=str)
        today_rows = df[df["sent_date"] == today]
        # Normalise match_date to YYYY-MM-DD to match _date_str() output
        return set(
            zip(today_rows["match_date"].str[:10],
                today_rows["player_a"],
                today_rows["player_b"])
        )
    except Exception:
        return set()


def _record_sent(match: dict) -> None:
    """Append one row to sent_alerts.csv."""
    row = pd.DataFrame([{
        "sent_date":   date.today().isoformat(),
        "match_date":  str(match.get("date", "")),
        "player_a":    str(match.get("player_a", "")),
        "player_b":    str(match.get("player_b", "")),
        "tournament":  str(match.get("tournament", "")),
        "edge":        match.get("model_edge", ""),
        "prob_a":      match.get("prob_a", ""),
    }])
    if SENT_ALERTS_CSV.exists():
        row.to_csv(SENT_ALERTS_CSV, mode="a", header=False, index=False)
    else:
        SENT_ALERTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(SENT_ALERTS_CSV, index=False)


def _date_str(val) -> str:
    """Normalise any date-like value to YYYY-MM-DD string."""
    try:
        import pandas as pd
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)[:10]   # best-effort slice


def _is_already_sent(match: dict, sent_today: set[tuple]) -> bool:
    key = (_date_str(match.get("date", "")),
           str(match.get("player_a", "")),
           str(match.get("player_b", "")))
    return key in sent_today


# ── core runner ───────────────────────────────────────────────────────────────

def run_alerts(
    predictions_path: Path = PREDICTIONS_CSV,
    threshold_pct: float | None = None,
    dry_run: bool = False,
    today_only: bool = True,
) -> int:
    """
    Send alerts for all qualifying predictions.

    Parameters
    ----------
    predictions_path : path to predictions CSV
    threshold_pct    : edge % threshold (e.g. 4.0 means 4%).  Reads config.json if None.
    dry_run          : print instead of sending SMS
    today_only       : only alert on today's matches (default True)

    Returns
    -------
    Number of alerts sent (or that would have been sent in dry-run).
    """
    if threshold_pct is None:
        threshold_pct = _load_config_threshold()

    threshold = threshold_pct / 100.0   # convert % → fraction

    # ── load predictions ──────────────────────────────────────────────────────
    if not predictions_path.exists():
        print(f"[alerts] No predictions file at {predictions_path}")
        return 0

    df = pd.read_csv(predictions_path)
    if df.empty:
        print("[alerts] predictions.csv is empty — nothing to alert.")
        return 0

    # Numeric coerce
    df["model_edge"] = pd.to_numeric(df.get("model_edge"), errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── filter: today's matches only ─────────────────────────────────────────
    if today_only and "date" in df.columns:
        today_ts = pd.Timestamp(date.today())
        df = df[df["date"].dt.date == today_ts.date()]
        if df.empty:
            print("[alerts] No predictions for today.")
            return 0

    # ── filter: edge above threshold ─────────────────────────────────────────
    qualifying = df[df["model_edge"].notna() & (df["model_edge"] >= threshold)].copy()

    if qualifying.empty:
        print(
            f"[alerts] No matches above edge threshold "
            f"({threshold_pct:.0f}%). Nothing to send."
        )
        return 0

    print(
        f"[alerts] {len(qualifying)} match(es) above "
        f"{threshold_pct:.0f}% edge threshold."
    )

    # ── deduplication ─────────────────────────────────────────────────────────
    sent_today = _load_sent_today()

    # ── load credentials once ─────────────────────────────────────────────────
    from utils.alerts import send_alert, load_credentials
    creds = load_credentials() if not dry_run else None

    if not dry_run and creds and not creds.valid():
        print(
            "[alerts] WARNING: Twilio credentials missing or incomplete.\n"
            "         Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER in .env or Streamlit secrets."
        )

    # ── send / dry-run ───────────────────────────────────────────────────────
    n_sent = 0
    n_skipped = 0

    for _, row in qualifying.iterrows():
        match = row.to_dict()

        if _is_already_sent(match, sent_today):
            player_a = match.get("player_a", "?")
            player_b = match.get("player_b", "?")
            print(f"[alerts] Already alerted today → {player_a} vs {player_b} (skipped)")
            n_skipped += 1
            continue

        ok = send_alert(match, creds=creds, dry_run=dry_run)

        if ok:
            n_sent += 1
            if not dry_run:
                _record_sent(match)

    print(
        f"[alerts] Done — {n_sent} alert(s) sent, "
        f"{n_skipped} duplicate(s) skipped."
    )
    return n_sent


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send SMS edge alerts for qualifying tennis predictions"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print alerts to terminal instead of sending SMS",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        metavar="PCT",
        help="Edge threshold in %% (e.g. 6 means 6%%). Reads config.json if omitted.",
    )
    parser.add_argument(
        "--predictions", type=str, default=None,
        help="Path to predictions CSV (default: output/predictions.csv)",
    )
    parser.add_argument(
        "--all-dates", action="store_true",
        help="Alert on all dates in the file, not just today",
    )
    args = parser.parse_args()

    preds_path = Path(args.predictions) if args.predictions else PREDICTIONS_CSV

    run_alerts(
        predictions_path = preds_path,
        threshold_pct    = args.threshold,
        dry_run          = args.dry_run,
        today_only       = not args.all_dates,
    )


if __name__ == "__main__":
    main()
