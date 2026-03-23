#!/usr/bin/env python3
"""List all tracked live runs with their results.

Usage:
    python scripts/list_runs.py              # last 20 runs
    python scripts/list_runs.py --all        # all runs
    python scripts/list_runs.py --json       # raw JSON output
    python scripts/list_runs.py <run_id>     # details for one run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNS_DIR = Path("logs/runs")


def _load_registry() -> list[dict]:
    reg_path = RUNS_DIR / "registry.json"
    if not reg_path.exists():
        return []
    try:
        return json.loads(reg_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def _show_detail(run_id: str) -> None:
    meta_path = RUNS_DIR / run_id / "meta.json"
    if not meta_path.exists():
        print(f"Run {run_id} not found.")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    print(json.dumps(meta, indent=2))

    # Show restore command
    print("\n--- Restore this run's code ---")
    if meta.get("has_uncommitted_changes"):
        print(f"  git checkout {meta['git_commit']}")
        patch = RUNS_DIR / run_id / "uncommitted.patch"
        if patch.exists():
            print(f"  git apply {patch}")
        print(f"  # or: git stash apply {meta.get('git_tag', '')}")
    else:
        print(f"  git checkout {meta.get('git_tag', meta.get('git_commit', ''))}")


def _show_table(runs: list[dict]) -> None:
    if not runs:
        print("No tracked runs yet. Run a live script to start tracking.")
        return

    # Header
    print(
        f"{'RUN ID':<17} {'SCRIPT':<22} {'GIT':<10} "
        f"{'FILLS':>5} {'SPENT':>8} {'CITIES':>6} {'DUR':>6} {'STATUS':<10}"
    )
    print("-" * 90)

    for r in runs:
        git = r.get("git_commit_short", "?")
        if r.get("dirty"):
            git += "*"
        script = r.get("script", "")
        # Shorten script name
        if "/" in script:
            script = script.rsplit("/", 1)[-1]
        if script.endswith(".py"):
            script = script[:-3]

        dur = r.get("duration_minutes", 0)
        dur_str = f"{dur:.0f}m" if dur else "-"

        print(
            f"{r.get('run_id', '?'):<17} {script:<22} {git:<10} "
            f"{r.get('trades_filled', 0):>5} "
            f"${r.get('total_spent', 0):>7.2f} "
            f"{r.get('unique_cities', 0):>6} "
            f"{dur_str:>6} "
            f"{r.get('status', '?'):<10}"
        )


def main():
    parser = argparse.ArgumentParser(description="List tracked live runs")
    parser.add_argument("run_id", nargs="?", help="Show details for a specific run")
    parser.add_argument("--all", action="store_true", help="Show all runs (default: last 20)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.run_id:
        _show_detail(args.run_id)
        return

    runs = _load_registry()

    if args.json:
        print(json.dumps(runs, indent=2))
        return

    if not args.all:
        runs = runs[-20:]

    _show_table(runs)


if __name__ == "__main__":
    main()
