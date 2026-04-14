"""
Shared utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def normalize_surface(surface: str) -> str:
    s = surface.strip().lower()
    if s in ("hard", "indoor hard", "outdoor hard"):
        return "hard"
    if s in ("clay", "red clay"):
        return "clay"
    if s in ("grass",):
        return "grass"
    if s in ("carpet",):
        return "carpet"
    return s


def normalize_name(name: str) -> str:
    """Standardise player name: title case, strip whitespace."""
    return name.strip().title()


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return np.nan
    return 1 / decimal_odds


def append_to_csv(row: dict, path: str) -> None:
    """Append a single dict as a row to a CSV (creates file with header if missing)."""
    p = Path(path)
    df_new = pd.DataFrame([row])
    if p.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)
