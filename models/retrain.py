"""
models/retrain.py
─────────────────
Quarterly retraining orchestrator.

Steps
-----
  1. Download fresh Sackmann ATP + WTA CSVs into data/raw/
  2. Rebuild processed match dataset (data_loader.py)
  3. Retrain XGBoost on rolling 5-year window  (train.py --rebuild --rolling --no-grid)
  4. Retrain LR + MLP ensemble  (ensemble.py)
  5. Compare new val Brier vs previous — only replace if improved
  6. Write output/retrain_log.json with full audit trail
  7. Exit 0 on improvement (workflow commits), exit 2 on no-improvement (workflow skips commit)

Usage
-----
  # Full run (downloads data, trains, compares)
  python3 models/retrain.py

  # Skip the data download step (use whatever is in data/raw/)
  python3 models/retrain.py --skip-download

  # Dry run: print plan, don't train or commit anything
  python3 models/retrain.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
SAVED_DIR   = ROOT / "models" / "saved"
RAW_DIR     = ROOT / "data" / "raw"
OUTPUT_DIR  = ROOT / "output"
FEAT_JSON   = SAVED_DIR / "feature_list.json"
RETRAIN_LOG = OUTPUT_DIR / "retrain_log.json"

SACKMANN_REPOS = {
    "tennis_atp": "https://github.com/JeffSackmann/tennis_atp.git",
    "tennis_wta": "https://github.com/JeffSackmann/tennis_wta.git",
}

MODEL_FILES = [
    "xgb_calibrated.pkl",
    "lr_calibrated.pkl",
    "mlp_calibrated.pkl",
    "feature_list.json",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command; raise on non-zero exit."""
    print(f"\n[retrain] ── {label} ──")
    print(f"[retrain] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"[retrain] FAILED ({result.returncode}): {' '.join(cmd)}"
        )


def _read_brier(feat_json: Path) -> float | None:
    """Read val_brier from feature_list.json; return None if missing."""
    if not feat_json.exists():
        return None
    try:
        with open(feat_json) as f:
            meta = json.load(f)
        # Prefer ensemble val Brier (ens_val_brier); fall back to xgb val_brier
        return float(meta.get("ens_val_brier") or meta.get("val_brier") or float("nan"))
    except Exception:
        return None


def _write_log(entry: dict) -> None:
    """Append one entry to retrain_log.json (list of run records)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if RETRAIN_LOG.exists():
        try:
            with open(RETRAIN_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[retrain] Log written → {RETRAIN_LOG}")


# ── download ──────────────────────────────────────────────────────────────────

def download_sackmann_data() -> None:
    """
    Clone Sackmann repos into data/raw/ or pull if they already exist.
    Each repo directory is placed at data/raw/<repo_name>/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for repo_name, url in SACKMANN_REPOS.items():
        dest = RAW_DIR / repo_name
        if dest.exists():
            print(f"[retrain] Pulling {repo_name}...")
            _run(["git", "-C", str(dest), "pull", "--ff-only"], f"pull {repo_name}")
        else:
            print(f"[retrain] Cloning {repo_name}...")
            _run(["git", "clone", "--depth=1", url, str(dest)], f"clone {repo_name}")


# ── backup / restore ──────────────────────────────────────────────────────────

def _backup_models(tmp_dir: Path) -> None:
    """Copy current saved models to tmp_dir."""
    for fname in MODEL_FILES:
        src = SAVED_DIR / fname
        if src.exists():
            shutil.copy2(src, tmp_dir / fname)


def _restore_models(tmp_dir: Path) -> None:
    """Restore models from tmp_dir back to SAVED_DIR."""
    for fname in MODEL_FILES:
        src = tmp_dir / fname
        if src.exists():
            shutil.copy2(src, SAVED_DIR / fname)
            print(f"[retrain] Restored {fname} from backup.")


# ── main ──────────────────────────────────────────────────────────────────────

def run_retrain(skip_download: bool = False, dry_run: bool = False) -> int:
    """
    Returns
    -------
    0  — new model is better (or equal); caller should commit updated files
    2  — new model is worse; old model kept; caller should skip commit
    """
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n[retrain] ════ Quarterly retrain started  {ts} ════")

    # ── 0. Dry-run guard ──────────────────────────────────────────────────────
    if dry_run:
        old_brier = _read_brier(FEAT_JSON)
        print(f"[retrain] DRY-RUN mode — no training will run.")
        print(f"[retrain] Current val Brier: {old_brier}")
        print(f"[retrain] Would: download data, rebuild features, retrain, compare.")
        return 0

    # ── 1. Read current (old) Brier before anything changes ───────────────────
    old_brier = _read_brier(FEAT_JSON)
    print(f"[retrain] Previous val Brier: {old_brier}")

    # Keep the backup dir alive through the Brier comparison so we can restore if needed
    backup_tmp = tempfile.mkdtemp(prefix="retrain_backup_")
    tmp_dir = Path(backup_tmp)
    _backup_models(tmp_dir)
    print(f"[retrain] Models backed up to {tmp_dir}")

    try:
        # ── 2. Download fresh data ─────────────────────────────────────────────
        if not skip_download:
            download_sackmann_data()
        else:
            print("[retrain] --skip-download: using existing data/raw/")

        # ── 3. Rebuild processed matches ───────────────────────────────────────
        _run([sys.executable, "utils/data_loader.py"], "rebuild processed data")

        # ── 4. Retrain XGBoost (rolling 5-year window, skip grid search) ──────
        _run(
            [sys.executable, "-m", "models.train",
             "--rebuild", "--rolling", "--no-grid"],
            "train XGBoost",
        )

        # ── 5. Retrain LR + MLP ensemble ───────────────────────────────────────
        _run(
            [sys.executable, "-m", "models.ensemble"],
            "train LR + MLP ensemble",
        )

    except Exception as exc:
        print(f"\n[retrain] ERROR during training: {exc}")
        print("[retrain] Restoring previous models from backup...")
        _restore_models(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _write_log({
            "timestamp":  ts,
            "status":     "error",
            "error":      str(exc),
            "old_brier":  old_brier,
            "new_brier":  None,
        })
        return 1   # hard failure

    # ── 6. Read new Brier ─────────────────────────────────────────────────────
    new_brier = _read_brier(FEAT_JSON)
    print(f"\n[retrain] New val Brier: {new_brier}")

    # ── 7. Compare ────────────────────────────────────────────────────────────
    if old_brier is None:
        improved  = True
        delta_str = "no baseline"
    elif new_brier is None:
        improved  = False
        delta_str = "new Brier unreadable"
    else:
        improved  = new_brier <= old_brier
        delta     = new_brier - old_brier
        delta_str = f"{delta:+.6f}"

    status = "improved" if improved else "no_improvement"
    print(f"[retrain] Δ Brier: {delta_str}  →  {status.upper()}")

    if not improved:
        # Restore the old (better) models from the backup we kept alive
        print("[retrain] New model is worse — restoring previous models from backup.")
        _restore_models(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _write_log({
            "timestamp":  ts,
            "status":     status,
            "old_brier":  old_brier,
            "new_brier":  new_brier,
            "delta":      new_brier - old_brier if (new_brier and old_brier) else None,
            "action":     "kept_old_model",
        })
        print("[retrain] ⚠️  WARNING: new model regressed. Old model kept.")
        return 2

    # ── 8. Accept new model ───────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _write_log({
        "timestamp":  ts,
        "status":     status,
        "old_brier":  old_brier,
        "new_brier":  new_brier,
        "delta":      (new_brier - old_brier) if (new_brier and old_brier) else None,
        "action":     "replaced_model",
    })

    print(f"\n[retrain] ✅  New model accepted  (Brier {old_brier} → {new_brier})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarterly retraining orchestrator"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip git clone/pull of Sackmann repos (use existing data/raw/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and current Brier; do not train",
    )
    args = parser.parse_args()

    exit_code = run_retrain(
        skip_download = args.skip_download,
        dry_run       = args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
