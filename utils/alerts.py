"""
utils/alerts.py
───────────────
Pushover push-notification alert sender for edge predictions.

Reads credentials from (in priority order):
  1. st.secrets  (Streamlit Cloud)
  2. Environment variables
  3. .env file in the project root

Usage
-----
  from utils.alerts import send_alert, load_credentials, AlertCredentials

  creds = load_credentials()
  send_alert(match_row, creds, dry_run=False)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


# ── credentials ───────────────────────────────────────────────────────────────

@dataclass
class AlertCredentials:
    user_key:  str
    api_token: str

    def valid(self) -> bool:
        return bool(self.user_key and self.api_token)


def _load_dotenv() -> dict[str, str]:
    """Parse .env file in project root into a dict. Never raises."""
    env_path = ROOT / ".env"
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    except Exception:
        pass
    return result


def load_credentials() -> AlertCredentials:
    """
    Load Pushover credentials. Priority:
      1. st.secrets (Streamlit Cloud runtime)
      2. Environment variables
      3. .env file
    """
    # 1 — Streamlit secrets (only available inside a Streamlit session)
    try:
        import streamlit as st
        secrets = st.secrets
        return AlertCredentials(
            user_key  = secrets.get("PUSHOVER_USER_KEY",  ""),
            api_token = secrets.get("PUSHOVER_API_TOKEN", ""),
        )
    except Exception:
        pass

    # 2 & 3 — env vars (possibly injected from .env below)
    env = _load_dotenv()
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v

    return AlertCredentials(
        user_key  = os.environ.get("PUSHOVER_USER_KEY",  ""),
        api_token = os.environ.get("PUSHOVER_API_TOKEN", ""),
    )


# ── message formatting ────────────────────────────────────────────────────────

def format_alert_message(match: dict) -> str:
    """
    Build the notification body from a predictions row.

    Expected keys (all optional with sensible fallbacks):
      player_a, player_b, surface, tournament,
      prob_a, model_edge, sharp_flag, opening_odds_a, opening_odds_b
    """
    player_a   = str(match.get("player_a",   "Player A"))
    player_b   = str(match.get("player_b",   "Player B"))
    surface    = str(match.get("surface",    "?")).upper()
    tournament = str(match.get("tournament", "Unknown"))

    prob_a = match.get("prob_a")
    edge   = match.get("model_edge")
    sharp  = match.get("sharp_flag")

    # Market implied from opening decimal odds
    oa = match.get("opening_odds_a")
    ob = match.get("opening_odds_b")
    if oa and ob:
        try:
            total = 1 / float(oa) + 1 / float(ob)
            market_pct = f"{(1 / float(oa)) / total:.0%}"
        except (ZeroDivisionError, TypeError):
            market_pct = "N/A"
    else:
        market_pct = "N/A"

    model_pct  = f"{float(prob_a):.0%}" if prob_a is not None else "N/A"
    edge_pct   = f"+{float(edge)*100:.1f}%" if edge is not None else "N/A"
    sharp_icon = "✓" if str(sharp) in ("1", "1.0", "True") else "✗"

    return (
        f"EDGE ALERT: {player_a} vs {player_b} "
        f"— {surface} "
        f"— Model {model_pct} vs Market {market_pct} "
        f"— Edge {edge_pct} "
        f"— Sharp {sharp_icon} "
        f"— {tournament}"
    )


# ── send ──────────────────────────────────────────────────────────────────────

def send_alert(
    match: dict,
    creds: Optional[AlertCredentials] = None,
    dry_run: bool = False,
) -> bool:
    """
    Send (or print, in dry-run mode) a Pushover notification for one match.

    Parameters
    ----------
    match    : dict or pd.Series row from predictions.csv
    creds    : AlertCredentials; loaded automatically if None
    dry_run  : if True, print the message instead of sending

    Returns
    -------
    True if sent (or would have been sent in dry-run), False on error.
    """
    if creds is None:
        creds = load_credentials()

    body = format_alert_message(match)

    if dry_run:
        player_a = match.get("player_a", "?")
        player_b = match.get("player_b", "?")
        print(f"[alerts] DRY-RUN → {player_a} vs {player_b}")
        print(f"         {body}")
        return True

    if not creds.valid():
        print(
            "[alerts] ERROR: Pushover credentials incomplete. "
            "Set PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN."
        )
        return False

    try:
        import urllib.request
        import urllib.parse

        data = urllib.parse.urlencode({
            "token":   creds.api_token,
            "user":    creds.user_key,
            "message": body,
            "title":   "Tennis Edge Alert",
        }).encode()

        req = urllib.request.Request(PUSHOVER_API_URL, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()

        if status == 200:
            print(f"[alerts] Pushover sent  "
                  f"{match.get('player_a','?')} vs {match.get('player_b','?')}")
            return True
        else:
            print(f"[alerts] ERROR: Pushover returned HTTP {status}")
            return False

    except Exception as exc:
        print(f"[alerts] ERROR sending Pushover notification: {exc}")
        return False
