"""
utils/alerts.py
───────────────
Twilio SMS alert sender for edge predictions.

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


# ── credentials ───────────────────────────────────────────────────────────────

@dataclass
class AlertCredentials:
    account_sid:  str
    auth_token:   str
    from_number:  str   # Twilio number, e.g. "+15005550006"
    to_number:    str   # Your mobile number, e.g. "+14155550123"

    def valid(self) -> bool:
        return all([self.account_sid, self.auth_token,
                    self.from_number, self.to_number])


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
    Load Twilio credentials. Priority:
      1. st.secrets (Streamlit Cloud runtime)
      2. Environment variables
      3. .env file
    """
    # 1 — Streamlit secrets (only available inside a Streamlit session)
    try:
        import streamlit as st
        secrets = st.secrets
        return AlertCredentials(
            account_sid = secrets.get("TWILIO_ACCOUNT_SID",  ""),
            auth_token  = secrets.get("TWILIO_AUTH_TOKEN",   ""),
            from_number = secrets.get("TWILIO_FROM_NUMBER",  ""),
            to_number   = secrets.get("TWILIO_TO_NUMBER",    ""),
        )
    except Exception:
        pass

    # 2 & 3 — env vars (possibly injected from .env below)
    env = _load_dotenv()
    # Merge dotenv into os.environ without overwriting existing vars
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v

    return AlertCredentials(
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID",  ""),
        auth_token  = os.environ.get("TWILIO_AUTH_TOKEN",   ""),
        from_number = os.environ.get("TWILIO_FROM_NUMBER",  ""),
        to_number   = os.environ.get("TWILIO_TO_NUMBER",    ""),
    )


# ── message formatting ────────────────────────────────────────────────────────

def format_alert_message(match: dict) -> str:
    """
    Build the SMS body from a predictions row.

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
    Send (or print, in dry-run mode) an SMS alert for one match.

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
            "[alerts] ERROR: Twilio credentials incomplete. "
            "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER."
        )
        return False

    try:
        from twilio.rest import Client          # type: ignore[import]
    except ImportError:
        print("[alerts] ERROR: twilio package not installed. Run: pip install twilio")
        return False

    try:
        client = Client(creds.account_sid, creds.auth_token)
        message = client.messages.create(
            body = body,
            from_ = creds.from_number,
            to    = creds.to_number,
        )
        print(f"[alerts] SMS sent  SID={message.sid}  "
              f"{match.get('player_a','?')} vs {match.get('player_b','?')}")
        return True
    except Exception as exc:
        print(f"[alerts] ERROR sending SMS: {exc}")
        return False
