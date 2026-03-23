"""
Run tracking system — links every live run to the exact code that produced it.

On each live run:
  1. Snapshots the git state (commit hash + uncommitted changes as a patch)
  2. Copies config.yaml and model hyperparams
  3. Creates a git tag `run/<run_id>` for easy checkout
  4. At run end, parses the trade log to record aggregate results

Usage in live scripts:
    from src.run_tracker import start_run, end_run

    run_id = start_run("run_live_simple.py", vars(args))
    # ... trading loop ...
    end_run(run_id, trade_log_path=TRADE_LOG_PATH)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path("logs/runs")


def _git(*cmd: str) -> str:
    """Run a git command and return stripped stdout. Returns '' on failure."""
    try:
        return subprocess.check_output(
            ["git"] + list(cmd),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _is_dirty() -> bool:
    """True if working tree or index has uncommitted changes."""
    return subprocess.call(
        ["git", "diff", "--quiet", "HEAD"],
        stderr=subprocess.DEVNULL,
    ) != 0


def _collect_hyperparams(run_dir: Path) -> None:
    """Copy all best.json files from logs/mega_tune/ into the run snapshot."""
    mega_dir = Path("logs/mega_tune")
    if not mega_dir.exists():
        return
    hp_dir = run_dir / "hyperparams"
    # Find the latest tuning timestamp directory
    tune_dirs = sorted(
        [d for d in mega_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not tune_dirs:
        return
    latest = tune_dirs[0]
    for model_dir in latest.iterdir():
        best = model_dir / "best.json"
        if best.exists():
            hp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(best, hp_dir / f"{model_dir.name}_best.json")


def start_run(
    script_name: str,
    args: dict | None = None,
) -> str:
    """Call at the start of any live run. Returns a run_id string.

    Creates:
      logs/runs/<run_id>/
        meta.json          — run metadata + git info
        config.yaml        — frozen copy of config
        uncommitted.patch  — diff of uncommitted changes (if any)
        hyperparams/       — copies of best.json files
    Also creates git tag  run/<run_id>  pointing to the exact code state.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- git state ----
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    short_hash = _git("rev-parse", "--short", "HEAD")
    dirty = _is_dirty()

    stash_hash = ""
    if dirty:
        # Save the full diff as a human-readable patch
        diff = _git("diff", "HEAD")
        if diff:
            (run_dir / "uncommitted.patch").write_text(diff)

        # Create a stash commit object (does NOT modify working tree or index)
        stash_hash = _git("stash", "create")

    # Create git tag for easy reference
    tag_target = stash_hash if stash_hash else commit
    tag_name = f"run/{run_id}"
    if tag_target:
        _git("tag", tag_name, tag_target)

    # ---- config snapshot ----
    config_path = Path("config.yaml")
    if config_path.exists():
        shutil.copy(config_path, run_dir / "config.yaml")

    # ---- hyperparams snapshot ----
    _collect_hyperparams(run_dir)

    # ---- metadata ----
    meta = {
        "run_id": run_id,
        "script": script_name,
        "args": _sanitize_args(args or {}),
        "git_commit": commit,
        "git_commit_short": short_hash,
        "git_branch": branch,
        "git_tag": tag_name,
        "git_stash": stash_hash,
        "has_uncommitted_changes": dirty,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Run tracked: {run_id} (git: {short_hash}{'*' if dirty else ''})")
    return run_id


def end_run(
    run_id: str,
    trade_log_path: Path | str | None = None,
    extra_results: dict | None = None,
) -> None:
    """Call at the end of a run to record results.

    Parses the trade log CSV to compute aggregate stats and writes them
    into meta.json.
    """
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        print(f"Warning: run {run_id} metadata not found, skipping end_run.")
        return

    meta = json.loads(meta_path.read_text())

    # Parse trade log for results
    results = _parse_trade_log(trade_log_path) if trade_log_path else {}
    if extra_results:
        results.update(extra_results)

    # Copy the trade log into the run directory for archival
    if trade_log_path and Path(trade_log_path).exists():
        shutil.copy(trade_log_path, run_dir / "trade_log.csv")

    meta["ended_at"] = datetime.now(timezone.utc).isoformat()
    meta["status"] = "completed"
    meta["results"] = results

    # Compute duration
    started = datetime.fromisoformat(meta["started_at"])
    ended = datetime.fromisoformat(meta["ended_at"])
    meta["duration_minutes"] = round((ended - started).total_seconds() / 60, 1)

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Update the flat registry for quick lookups
    _update_registry(meta)

    _print_summary(meta)


def _parse_trade_log(path: Path | str | None) -> dict:
    """Parse a trade log CSV and return aggregate stats."""
    path = Path(path) if path else None
    if not path or not path.exists():
        return {}

    trades_placed = 0
    trades_filled = 0
    trades_no_fill = 0
    trades_recommended = 0
    total_spent = 0.0
    total_contracts = 0
    tickers_traded: set[str] = set()
    cities_traded: set[str] = set()

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                action = row.get("action", "")
                status = row.get("status", "")
                spend = float(row.get("size_dollars", 0) or 0)
                contracts = int(float(row.get("contract_count", 0) or 0))

                if action == "place":
                    trades_placed += 1
                    if status in ("placed", "pending"):
                        trades_filled += 1
                        total_spent += spend
                        total_contracts += contracts
                        tickers_traded.add(row.get("market_ticker", ""))
                        cities_traded.add(row.get("city", ""))
                    elif status == "no_fill":
                        trades_no_fill += 1
                elif action == "recommend":
                    trades_recommended += 1
    except Exception as e:
        return {"parse_error": str(e)}

    return {
        "trades_placed": trades_placed,
        "trades_filled": trades_filled,
        "trades_no_fill": trades_no_fill,
        "trades_recommended": trades_recommended,
        "total_spent": round(total_spent, 2),
        "total_contracts": total_contracts,
        "unique_tickers": len(tickers_traded),
        "unique_cities": len(cities_traded - {""}),
        "tickers": sorted(tickers_traded - {""}),
        "cities": sorted(cities_traded - {""}),
    }


def _update_registry(meta: dict) -> None:
    """Append/update a flat JSON registry file for quick listing."""
    registry_path = RUNS_DIR / "registry.json"
    registry: list[dict] = []
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except (json.JSONDecodeError, ValueError):
            registry = []

    # Remove existing entry for this run_id (in case of re-run)
    registry = [r for r in registry if r.get("run_id") != meta["run_id"]]

    # Add summary entry
    results = meta.get("results", {})
    registry.append({
        "run_id": meta["run_id"],
        "script": meta.get("script", ""),
        "git_commit_short": meta.get("git_commit_short", ""),
        "git_branch": meta.get("git_branch", ""),
        "dirty": meta.get("has_uncommitted_changes", False),
        "started_at": meta.get("started_at", ""),
        "ended_at": meta.get("ended_at", ""),
        "duration_minutes": meta.get("duration_minutes", 0),
        "trades_filled": results.get("trades_filled", 0),
        "total_spent": results.get("total_spent", 0),
        "unique_cities": results.get("unique_cities", 0),
        "status": meta.get("status", ""),
    })

    registry_path.write_text(json.dumps(registry, indent=2) + "\n")


def _print_summary(meta: dict) -> None:
    """Print a one-line summary of the completed run."""
    r = meta.get("results", {})
    filled = r.get("trades_filled", 0)
    spent = r.get("total_spent", 0)
    cities = r.get("unique_cities", 0)
    dur = meta.get("duration_minutes", 0)
    tag = meta.get("git_tag", "")
    print(
        f"Run {meta['run_id']} complete: "
        f"{filled} fills, ${spent:.2f} spent, "
        f"{cities} cities, {dur}m duration  "
        f"[restore: git stash apply {tag}]"
        if meta.get("has_uncommitted_changes")
        else f"Run {meta['run_id']} complete: "
        f"{filled} fills, ${spent:.2f} spent, "
        f"{cities} cities, {dur}m duration  "
        f"[restore: git checkout {tag}]"
    )


def _sanitize_args(args: dict) -> dict:
    """Make args JSON-serializable."""
    clean = {}
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = [str(x) for x in v]
        else:
            clean[k] = str(v)
    return clean
