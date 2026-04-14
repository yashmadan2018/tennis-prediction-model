"""
The Odds API client for live tennis match odds.

API docs  : https://the-odds-api.com/liveapi/guides/v4/
Free tier : 500 requests/month — every response is cached to data/odds/
            so re-runs within the same session don't burn quota.

Authentication
--------------
Set env var ODDS_API_KEY, or pass api_key= to OddsClient().

Usage
-----
  from utils.odds_fetcher import OddsClient
  client = OddsClient()                        # reads ODDS_API_KEY from env
  df = client.fetch_tennis_odds()              # all upcoming matches
  df = client.fetch_tennis_odds(bookmakers=["pinnacle", "betfair_ex_eu"])
  client.print_quota()                         # remaining API requests

Output DataFrame columns
------------------------
  event_id, sport_key, tournament, surface, tour,
  commence_time, player_a, player_b,
  odds_a, odds_b, bookmaker, snapshot_time
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
ODDS_DIR  = ROOT / "data" / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# ── API constants ──────────────────────────────────────────────────────────────

BASE_URL   = "https://api.the-odds-api.com/v4"
REGIONS    = "eu"          # eu gives Pinnacle; us gives US books
MARKETS    = "h2h"         # head-to-head moneyline only
ODDS_FMT   = "decimal"

# All bookmakers we'll try in preference order
PREFERRED_BOOKS = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk", "unibet_eu",
                   "williamhill", "bet365", "draftkings", "fanduel"]

# ── tournament → surface lookup ────────────────────────────────────────────────
# Keyed on lowercased substrings found in sport_key or sport_title.

_CLAY_KEYS = {
    "french_open", "roland_garros", "monte_carlo", "monte carlo",
    "madrid", "barcelona", "rome", "hamburg", "munich",
    "estoril", "bucharest", "bastad", "umag", "gstaad",
    "kitzbuhel", "kitzbühel", "metz", "lyon", "marrakech",
    "casablanca", "istanbul", "prague", "budapest", "rabat",
    "geneva", "parma", "sttropez", "st tropez", "clay",
}
_GRASS_KEYS = {
    "wimbledon", "queens", "queen's", "halle", "eastbourne",
    "'s-hertogenbosch", "hertogenbosch", "birmingham", "nottingham",
    "grass",
}
_HARD_INDOOR_KEYS = {
    "rotterdam", "marseille", "montpellier", "sofia", "singapore",
    "dubai indoor", "indoor",
}

# sport_key suffixes that encode the surface directly
_SPORT_KEY_SURFACE: dict[str, str] = {
    "french_open":        "clay",
    "roland_garros":      "clay",
    "wimbledon":          "grass",
    "us_open":            "hard",
    "australian_open":    "hard",
}

# Grand Slam sport key fragments
_GRAND_SLAMS = {"french_open", "roland_garros", "wimbledon", "us_open", "australian_open"}


def _surface_from_tournament(sport_key: str, sport_title: str) -> str:
    """Infer surface from sport_key / sport_title. Defaults to 'hard'."""
    combined = f"{sport_key} {sport_title}".lower()

    for frag, surf in _SPORT_KEY_SURFACE.items():
        if frag in combined:
            return surf

    for kw in _CLAY_KEYS:
        if kw in combined:
            return "clay"
    for kw in _GRASS_KEYS:
        if kw in combined:
            return "grass"

    return "hard"


def _tour_from_sport_key(sport_key: str) -> str:
    """'atp' | 'wta' | 'atp' (default)."""
    sk = sport_key.lower()
    if "_wta" in sk or sk.startswith("wta"):
        return "wta"
    return "atp"


def _is_grand_slam(sport_key: str) -> bool:
    sk = sport_key.lower()
    return any(gs in sk for gs in _GRAND_SLAMS)


def _best_of(sport_key: str, tour: str) -> int:
    """Grand Slam ATP = best of 5; everything else = 3."""
    if _is_grand_slam(sport_key) and tour == "atp":
        return 5
    return 3


def _tourney_level(sport_key: str, tour: str) -> str:
    """Approximate tourney_level code used in pipeline."""
    if _is_grand_slam(sport_key):
        return "G"
    sk = sport_key.lower()
    masters_kw = {
        "indian_wells", "miami", "monte_carlo", "madrid", "rome",
        "canadian_open", "rogers_cup", "cincinnati", "shanghai",
        "paris_masters", "atp_finals",
    }
    if any(kw in sk for kw in masters_kw):
        return "M"
    if tour == "wta":
        if any(kw in sk for kw in {"mandatory", "premier_mandatory"}):
            return "PM"
        if any(kw in sk for kw in {"premier", "500"}):
            return "P"
        return "I"
    return "A"


# ── client ─────────────────────────────────────────────────────────────────────

class OddsClient:
    """
    Thin wrapper around The Odds API v4.

    Parameters
    ----------
    api_key  : API key — reads ODDS_API_KEY env var if omitted
    timeout  : HTTP timeout in seconds
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 15,
    ) -> None:
        key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not key:
            raise ValueError(
                "No API key found. Set ODDS_API_KEY env var or pass api_key=."
            )
        self._key     = key
        self._timeout = timeout
        self._remaining: int | None = None   # updated after each call

    # ── low-level request ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """GET request; updates quota tracker from response headers."""
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        p   = {"apiKey": self._key, **(params or {})}
        resp = requests.get(url, params=p, timeout=self._timeout)
        resp.raise_for_status()

        # Track remaining quota
        rem = resp.headers.get("x-requests-remaining")
        if rem is not None:
            self._remaining = int(rem)

        return resp.json()

    # ── sports list ───────────────────────────────────────────────────────────

    def get_sports(self, active_only: bool = True) -> list[dict]:
        """Return all sports available in the API."""
        data = self._get("sports", {"all": "false" if active_only else "true"})
        return data

    def get_tennis_sport_keys(self) -> list[str]:
        """Return all currently-active tennis sport keys."""
        sports = self.get_sports(active_only=True)
        return [s["key"] for s in sports if "tennis" in s["key"].lower()]

    # ── odds fetch ────────────────────────────────────────────────────────────

    def get_odds_for_sport(
        self,
        sport_key: str,
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch raw odds JSON for a single sport key.
        Returns list of event dicts (may be empty if no upcoming matches).
        """
        params: dict = {
            "regions":    REGIONS,
            "markets":    MARKETS,
            "oddsFormat": ODDS_FMT,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            return self._get(f"sports/{sport_key}/odds", params) or []
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 422:
                return []   # sport key exists but no odds right now
            raise

    # ── main fetch ────────────────────────────────────────────────────────────

    def fetch_tennis_odds(
        self,
        bookmakers: list[str] | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all upcoming tennis matches with odds from The Odds API.

        Parameters
        ----------
        bookmakers : list of bookmaker keys to request (default: all via REGIONS=eu)
        save       : if True, raw JSON cached to data/odds/ and normalized
                     DataFrame saved as odds_snapshot_<timestamp>.csv

        Returns
        -------
        DataFrame with one row per (event × bookmaker):
            event_id, sport_key, tournament, surface, tour,
            commence_time, player_a, player_b, odds_a, odds_b,
            bookmaker, snapshot_time
        """
        sport_keys = self.get_tennis_sport_keys()
        if not sport_keys:
            print("[odds] No active tennis sport keys found.")
            return pd.DataFrame()

        print(f"[odds] Found {len(sport_keys)} active tennis sport keys: {sport_keys}")

        all_events: list[dict] = []
        for sk in sport_keys:
            events = self.get_odds_for_sport(sk, bookmakers=bookmakers)
            all_events.extend(events)
            time.sleep(0.2)   # gentle rate limiting

        if not all_events:
            print("[odds] No upcoming tennis matches with odds found.")
            return pd.DataFrame()

        snapshot_time = datetime.now(timezone.utc)

        if save:
            ts   = snapshot_time.strftime("%Y%m%d_%H%M%S")
            raw_path = ODDS_DIR / f"raw_odds_{ts}.json"
            raw_path.write_text(json.dumps(all_events, indent=2))
            print(f"[odds] Raw JSON saved → {raw_path}")

        df = _normalize_events(all_events, snapshot_time)

        if save and not df.empty:
            csv_path = ODDS_DIR / f"odds_snapshot_{ts}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[odds] Snapshot saved → {csv_path}  ({len(df):,} rows)")

        if self._remaining is not None:
            print(f"[odds] API quota remaining: {self._remaining} requests")

        return df

    # ── quota ─────────────────────────────────────────────────────────────────

    def print_quota(self) -> None:
        """Print remaining API request quota (makes one lightweight call)."""
        self.get_sports()
        if self._remaining is not None:
            print(f"[odds] Remaining requests: {self._remaining}")
        else:
            print("[odds] Quota info unavailable.")


# ── normalisation ──────────────────────────────────────────────────────────────

def _normalize_events(events: list[dict], snapshot_time: datetime) -> pd.DataFrame:
    """
    Flatten raw API event list into one row per (event × bookmaker).
    Picks the best available bookmaker per event using PREFERRED_BOOKS order.
    """
    rows: list[dict] = []

    for ev in events:
        sport_key   = ev.get("sport_key", "")
        sport_title = ev.get("sport_title", "")
        event_id    = ev.get("id", "")
        commence    = ev.get("commence_time", "")
        player_a    = str(ev.get("home_team", "")).strip().title()
        player_b    = str(ev.get("away_team", "")).strip().title()

        surface = _surface_from_tournament(sport_key, sport_title)
        tour    = _tour_from_sport_key(sport_key)

        bookmakers: list[dict] = ev.get("bookmakers", [])
        if not bookmakers:
            continue

        for bm in bookmakers:
            bm_key = bm.get("key", "")
            markets = bm.get("markets", [])
            h2h = next((m for m in markets if m.get("key") == "h2h"), None)
            if h2h is None:
                continue

            outcomes = {o["name"].strip().title(): o["price"]
                        for o in h2h.get("outcomes", [])}

            odds_a = outcomes.get(player_a)
            odds_b = outcomes.get(player_b)
            if odds_a is None or odds_b is None:
                # API uses home/away order — try swapping
                names = list(outcomes.keys())
                if len(names) == 2:
                    odds_a = outcomes.get(names[0])
                    odds_b = outcomes.get(names[1])
                    player_a = names[0]
                    player_b = names[1]

            if odds_a is None or odds_b is None:
                continue

            rows.append({
                "event_id":      event_id,
                "sport_key":     sport_key,
                "tournament":    sport_title,
                "surface":       surface,
                "tour":          tour,
                "best_of":       _best_of(sport_key, tour),
                "tourney_level": _tourney_level(sport_key, tour),
                "commence_time": commence,
                "player_a":      player_a,
                "player_b":      player_b,
                "odds_a":        float(odds_a),
                "odds_b":        float(odds_b),
                "bookmaker":     bm_key,
                "snapshot_time": snapshot_time.isoformat(),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df = df.sort_values("commence_time").reset_index(drop=True)
    return df


# ── snapshot helpers ───────────────────────────────────────────────────────────

def load_latest_snapshot() -> pd.DataFrame:
    """Load the most recently saved odds snapshot CSV."""
    csvs = sorted(ODDS_DIR.glob("odds_snapshot_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No odds snapshots in {ODDS_DIR}. Run OddsClient().fetch_tennis_odds() first."
        )
    path = csvs[-1]
    print(f"[odds] Loading snapshot: {path.name}")
    df = pd.read_csv(path, parse_dates=["commence_time"])
    return df


def best_bookmaker_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event_id keep a single row — the best available bookmaker
    in PREFERRED_BOOKS order (Pinnacle first).  Events not covered by
    any preferred book are kept with whichever bookmaker was returned.
    """
    if df.empty:
        return df

    book_rank = {b: i for i, b in enumerate(PREFERRED_BOOKS)}
    df = df.copy()
    df["_brank"] = df["bookmaker"].map(book_rank).fillna(len(PREFERRED_BOOKS))
    df = (
        df.sort_values("_brank")
          .groupby("event_id", sort=False)
          .first()
          .reset_index()
          .drop(columns=["_brank"])
    )
    return df


# ── line movement across snapshots ────────────────────────────────────────────

def compute_line_movement(
    opening_df: pd.DataFrame,
    closing_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join two snapshots on event_id + bookmaker and compute line movement.

    Returns DataFrame with opening_odds_a/b, closing_odds_a/b, and
    movement columns ready to feed into features.market.get_market_features().
    """
    open_  = best_bookmaker_row(opening_df)[
        ["event_id", "player_a", "player_b", "odds_a", "odds_b",
         "surface", "tournament", "tour", "best_of", "tourney_level",
         "commence_time"]
    ].rename(columns={"odds_a": "opening_odds_a", "odds_b": "opening_odds_b"})

    close_ = best_bookmaker_row(closing_df)[
        ["event_id", "odds_a", "odds_b"]
    ].rename(columns={"odds_a": "closing_odds_a", "odds_b": "closing_odds_b"})

    merged = open_.merge(close_, on="event_id", how="left")
    return merged


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    client = OddsClient()

    print("[odds] Fetching all active tennis sport keys...")
    keys = client.get_tennis_sport_keys()
    print(f"  {keys}\n")

    print("[odds] Fetching tennis odds...")
    df = client.fetch_tennis_odds()

    if df.empty:
        print("[odds] No upcoming matches found (off-season or no API key).")
        sys.exit(0)

    print(f"\n[odds] {len(df):,} rows fetched.")
    print(df[["tournament", "surface", "player_a", "player_b",
              "odds_a", "odds_b", "bookmaker"]].to_string(index=False))
