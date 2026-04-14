"""
Nightly result logger.

Fills in the `result` column of output/clv_tracker.csv by matching
logged predictions against actual Sackmann match CSVs.

Algorithm
---------
1. Load clv_tracker.csv — isolate rows where result is NaN.
2. Determine which raw CSV files cover the unresolved date range;
   check for files added / modified since last run.
3. Load those Sackmann matches into a single lookup DataFrame.
4. For each unresolved prediction:
     a. Filter candidates to ±2 day window around prediction date.
     b. Resolve player_a and player_b names against winner/loser columns
        using the same three-tier resolution as predict.py
        (exact → accent-stripped → difflib fuzzy at 0.82 cutoff).
     c. Fill result=1 if player_a won, result=0 if player_b won.
     d. Also compute clv_delta = model_prob_a − closing_implied_a
        (falls back to opening_implied_a when closing is absent).
5. Write updated clv_tracker.csv in-place.
6. Append unmatched rows to output/unmatched_results.csv.
7. Persist state (last_run, per-file mtimes) to avoid redundant I/O.

Usage
-----
  python utils/result_logger.py                # normal nightly run
  python utils/result_logger.py --dry-run      # show matches without writing
  python utils/result_logger.py --force        # ignore state; rescan all files
  python utils/result_logger.py --since 2025-01-01  # only process predictions on/after date
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT           = Path(__file__).parent.parent
RAW_DIR        = ROOT / "data" / "raw"
PROCESSED_DIR  = ROOT / "data" / "processed"
OUTPUT_DIR     = ROOT / "output"
CLV_CSV        = OUTPUT_DIR / "clv_tracker.csv"
UNMATCHED_CSV  = OUTPUT_DIR / "unmatched_results.csv"
STATE_FILE     = PROCESSED_DIR / "result_logger_state.json"

DATE_WINDOW_DAYS = 2   # look ±N days around prediction date for a match
FUZZY_CUTOFF     = 0.82

# Raw file patterns to scan
RAW_PATTERNS = [
    "atp_matches_[0-9][0-9][0-9][0-9].csv",
    "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
    "wta_matches_[0-9][0-9][0-9][0-9].csv",
    "wta_matches_qual_itf_[0-9][0-9][0-9][0-9].csv",
]


# ── name normalisation (mirrors predict.py) ────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase, strip accents and punctuation."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_ = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_.lower().strip()


def _resolve_name(
    api_name: str,
    exact_set: set[str],
    norm_map: dict[str, str],   # normalized → canonical
    cutoff: float = FUZZY_CUTOFF,
) -> str | None:
    """
    Return the canonical Sackmann name matching api_name, or None.
    Three-tier: exact (title-cased) → accent-stripped → difflib fuzzy.
    """
    titled = api_name.strip().title()
    if titled in exact_set:
        return titled

    normed = _normalize(api_name)
    if normed in norm_map:
        return norm_map[normed]

    matches = difflib.get_close_matches(normed, norm_map.keys(), n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]]

    return None


def _build_name_sets(sackmann: pd.DataFrame) -> tuple[set[str], dict[str, str]]:
    """Build exact_set and norm_map from a Sackmann results DataFrame."""
    all_names: set[str] = set()
    for col in ("winner_name", "loser_name"):
        if col in sackmann.columns:
            all_names.update(sackmann[col].dropna().str.strip().str.title().unique())
    exact_set = all_names
    norm_map  = {_normalize(n): n for n in all_names}
    return exact_set, norm_map


# ── state management ───────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run": None, "file_mtimes": {}}


def _save_state(state: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _all_raw_files() -> list[Path]:
    """Return sorted list of all Sackmann raw CSV files."""
    import glob as _glob
    files = []
    for pat in RAW_PATTERNS:
        files.extend(Path(p) for p in sorted(_glob.glob(str(RAW_DIR / pat))))
    return files


def _files_to_scan(
    state: dict,
    unresolved_dates: pd.Series,
    force: bool = False,
) -> list[Path]:
    """
    Determine which raw files to load:
      • Any file covering the year range of unresolved predictions.
      • Any file that is new or has a modified mtime since last run.
    """
    all_files = _all_raw_files()
    if not all_files:
        return []

    if force:
        return all_files

    old_mtimes: dict[str, float] = state.get("file_mtimes", {})

    # Year range of unresolved predictions
    if not unresolved_dates.empty:
        min_year = int(unresolved_dates.dt.year.min())
        max_year = int(unresolved_dates.dt.year.max())
    else:
        min_year = max_year = 9999   # no unresolved → still pick up new/changed files

    needed: list[Path] = []
    for f in all_files:
        # Detect year from filename (last 4-digit segment before .csv)
        stem = f.stem
        digits = [p for p in stem.split("_") if p.isdigit() and len(p) == 4]
        file_year = int(digits[-1]) if digits else 0

        in_range  = (min_year <= file_year <= max_year)
        mtime     = f.stat().st_mtime
        is_new    = str(f.name) not in old_mtimes
        is_updated = not is_new and old_mtimes[str(f.name)] < mtime

        if in_range or is_new or is_updated:
            needed.append(f)

    return needed


# ── Sackmann loader ────────────────────────────────────────────────────────────

def _load_sackmann(files: list[Path]) -> pd.DataFrame:
    """Load winner/loser names + dates from a list of raw CSVs."""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in {
                "tourney_id", "tourney_name", "tourney_date", "match_num",
                "winner_name", "loser_name", "score",
            })
            frames.append(df)
        except Exception as e:
            print(f"  [result_logger] WARNING: could not read {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(
        combined["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    combined = combined.dropna(subset=["date"])
    combined["winner_name"] = combined["winner_name"].astype(str).str.strip().str.title()
    combined["loser_name"]  = combined["loser_name"].astype(str).str.strip().str.title()
    combined["tourney_name_lower"] = combined["tourney_name"].str.lower().str.strip()
    return combined.sort_values("date").reset_index(drop=True)


# ── tournament name similarity ─────────────────────────────────────────────────

def _tourney_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio between two lowercased tournament names."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── core matching ──────────────────────────────────────────────────────────────

def _match_prediction(
    row: pd.Series,
    sackmann: pd.DataFrame,
    exact_set: set[str],
    norm_map: dict[str, str],
) -> tuple[int | None, str]:
    """
    Attempt to match one CLV prediction row to a Sackmann match.

    Returns (result, method_description):
      result = 1  (player_a won)
      result = 0  (player_b won)
      result = None (no match found)
    """
    pred_date    = pd.Timestamp(row["date"])
    player_a_raw = str(row["player_a"])
    player_b_raw = str(row["player_b"])
    tourney_raw  = str(row.get("tournament", ""))

    # ── narrow candidates by date window ───────────────────────────────────────
    lo = pred_date - timedelta(days=DATE_WINDOW_DAYS)
    hi = pred_date + timedelta(days=DATE_WINDOW_DAYS)
    cands = sackmann[(sackmann["date"] >= lo) & (sackmann["date"] <= hi)].copy()
    if cands.empty:
        return None, "no_candidates_in_window"

    # ── resolve predicted names to Sackmann canonical names ───────────────────
    a_canon = _resolve_name(player_a_raw, exact_set, norm_map)
    b_canon = _resolve_name(player_b_raw, exact_set, norm_map)

    if a_canon is None and b_canon is None:
        return None, f"unresolved_both: {player_a_raw!r}, {player_b_raw!r}"
    if a_canon is None:
        return None, f"unresolved_a: {player_a_raw!r}"
    if b_canon is None:
        return None, f"unresolved_b: {player_b_raw!r}"

    # ── find the specific matchup ──────────────────────────────────────────────
    mask = (
        ((cands["winner_name"] == a_canon) & (cands["loser_name"] == b_canon)) |
        ((cands["winner_name"] == b_canon) & (cands["loser_name"] == a_canon))
    )
    found = cands[mask]

    # ── optional tournament disambiguation ────────────────────────────────────
    if len(found) > 1 and tourney_raw:
        scores = found["tourney_name_lower"].apply(
            lambda t: _tourney_sim(tourney_raw, t)
        )
        best = scores.idxmax()
        found = found.loc[[best]]

    if found.empty:
        return None, f"no_matchup_found: {a_canon!r} vs {b_canon!r}"

    match_row = found.iloc[0]
    result = 1 if match_row["winner_name"] == a_canon else 0
    method = f"matched ({a_canon!r} vs {b_canon!r}, date={match_row['date'].date()})"
    return result, method


# ── CLV delta ──────────────────────────────────────────────────────────────────

def _compute_clv_delta(row: pd.Series) -> float:
    """model_prob_a − closing_implied_a (falls back to opening if closing absent)."""
    model_prob = row.get("model_prob_a", np.nan)
    closing    = row.get("closing_implied_a", np.nan)
    opening    = row.get("opening_implied_a", np.nan)

    if pd.isna(model_prob):
        return np.nan
    baseline = closing if not pd.isna(closing) else opening
    if pd.isna(baseline):
        return np.nan
    return round(float(model_prob) - float(baseline), 4)


# ── main function ──────────────────────────────────────────────────────────────

def fill_clv_results(
    dry_run:   bool = False,
    force:     bool = False,
    since:     str  | None = None,
    verbose:   bool = True,
) -> dict:
    """
    Scan Sackmann CSVs and fill result + clv_delta for unresolved CLV rows.

    Returns summary dict: {filled, unmatched, skipped, total_unresolved}
    """
    from datetime import datetime, timezone

    # ── load CLV tracker ───────────────────────────────────────────────────────
    if not CLV_CSV.exists():
        print("[result_logger] clv_tracker.csv not found — nothing to fill.")
        return {"filled": 0, "unmatched": 0, "skipped": 0, "total_unresolved": 0}

    clv = pd.read_csv(CLV_CSV)
    # Normalise result column
    clv["result"] = pd.to_numeric(clv["result"], errors="coerce")

    unresolved_mask = clv["result"].isna()

    if since:
        clv["date"] = pd.to_datetime(clv["date"], errors="coerce")
        since_ts = pd.Timestamp(since)
        unresolved_mask = unresolved_mask & (clv["date"] >= since_ts)

    unresolved = clv[unresolved_mask].copy()
    print(f"[result_logger] {len(unresolved)} unresolved predictions "
          f"(of {len(clv)} total)")

    if unresolved.empty:
        print("[result_logger] Nothing to fill.")
        return {"filled": 0, "unmatched": 0, "skipped": 0, "total_unresolved": 0}

    unresolved["date"] = pd.to_datetime(unresolved["date"], errors="coerce")

    # ── determine which raw files to scan ─────────────────────────────────────
    state = _load_state()
    files = _files_to_scan(state, unresolved["date"], force=force)
    if not files:
        print("[result_logger] No raw files to scan for the relevant date range.")
        return {"filled": 0, "unmatched": 0, "skipped": 0,
                "total_unresolved": len(unresolved)}

    print(f"[result_logger] Scanning {len(files)} raw file(s): "
          f"{[f.name for f in files]}")

    # ── load Sackmann matches ──────────────────────────────────────────────────
    sackmann    = _load_sackmann(files)
    exact_set, norm_map = _build_name_sets(sackmann)
    print(f"[result_logger] Loaded {len(sackmann):,} Sackmann matches "
          f"({len(exact_set)} unique player names)")

    # ── match each prediction ──────────────────────────────────────────────────
    filled_indices: list[int] = []
    unmatched_rows: list[dict] = []
    results_map: dict[int, int] = {}   # clv.index → result value

    for idx, row in unresolved.iterrows():
        result, method = _match_prediction(row, sackmann, exact_set, norm_map)
        if result is not None:
            results_map[idx] = result
            filled_indices.append(idx)
            if verbose:
                print(f"  FILLED  idx={idx}  {row['player_a']} vs {row['player_b']}"
                      f"  → result={result}  [{method}]")
        else:
            unmatched_rows.append({
                **row.to_dict(),
                "reason": method,
            })
            if verbose:
                print(f"  SKIP    idx={idx}  {row['player_a']} vs {row['player_b']}"
                      f"  → {method}")

    # ── apply fills ───────────────────────────────────────────────────────────
    if not dry_run:
        clv["date"] = pd.to_datetime(clv["date"], errors="coerce")
        for idx, result in results_map.items():
            clv.at[idx, "result"] = result
            # Fill clv_delta while we're here
            if pd.isna(clv.at[idx, "clv_delta"]):
                clv.at[idx, "clv_delta"] = _compute_clv_delta(clv.loc[idx])

        clv.to_csv(CLV_CSV, index=False)
        print(f"\n[result_logger] Updated {CLV_CSV} "
              f"({len(results_map)} results filled)")

        # Append unmatched rows
        if unmatched_rows:
            um_df = pd.DataFrame(unmatched_rows)
            if UNMATCHED_CSV.exists():
                um_df.to_csv(UNMATCHED_CSV, mode="a", header=False, index=False)
            else:
                um_df.to_csv(UNMATCHED_CSV, index=False)
            print(f"[result_logger] {len(unmatched_rows)} unmatched → {UNMATCHED_CSV}")

        # Persist state
        new_state = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "file_mtimes": {
                f.name: f.stat().st_mtime for f in _all_raw_files()
            },
        }
        _save_state(new_state)
    else:
        print(f"\n[result_logger] DRY RUN — no files written.")
        print(f"  Would fill  : {len(results_map)}")
        print(f"  Unmatched   : {len(unmatched_rows)}")

    summary = {
        "filled":           len(results_map),
        "unmatched":        len(unmatched_rows),
        "skipped":          0,
        "total_unresolved": len(unresolved),
    }
    print(f"\n[result_logger] Summary: {summary}")
    return summary


# ── CLV report ─────────────────────────────────────────────────────────────────

def print_clv_report() -> None:
    """Print a quick CLV performance summary from the tracker."""
    if not CLV_CSV.exists():
        print("[result_logger] No CLV tracker found.")
        return

    clv = pd.read_csv(CLV_CSV)
    clv["result"] = pd.to_numeric(clv["result"], errors="coerce")
    filled = clv[clv["result"].notna()]

    if filled.empty:
        print("[result_logger] No filled results yet — run after matches complete.")
        return

    print(f"\n── CLV Performance Report ──────────────────────────────────")
    print(f"  Predictions logged  : {len(clv)}")
    print(f"  Results filled      : {len(filled)}")
    print(f"  Actual win rate (A) : {filled['result'].mean():.3f}")

    prob_col = "model_prob_a" if "model_prob_a" in filled.columns else None
    if prob_col:
        acc = ((filled[prob_col] >= 0.5) == filled["result"]).mean()
        print(f"  Model accuracy      : {acc:.3f}")

    clv_col = "clv_delta" if "clv_delta" in filled.columns else None
    if clv_col:
        clv_vals = pd.to_numeric(filled[clv_col], errors="coerce").dropna()
        if not clv_vals.empty:
            print(f"  Mean CLV delta      : {clv_vals.mean():+.4f}")
            print(f"  Positive CLV rows   : {(clv_vals > 0).sum()} / {len(clv_vals)}")

    print(f"────────────────────────────────────────────────────────────\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly result logger for CLV tracker")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be filled without writing")
    parser.add_argument("--force",    action="store_true",
                        help="Ignore state file; rescan all raw files")
    parser.add_argument("--since",    type=str, default=None,
                        help="Only process predictions on/after this date (YYYY-MM-DD)")
    parser.add_argument("--report",   action="store_true",
                        help="Print CLV performance report and exit")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-match verbose output")
    args = parser.parse_args()

    if args.report:
        print_clv_report()
        return

    fill_clv_results(
        dry_run = args.dry_run,
        force   = args.force,
        since   = args.since,
        verbose = not args.quiet,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
