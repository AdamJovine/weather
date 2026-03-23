#!/usr/bin/env python3
"""Restore the code state from a tracked live run.

Usage:
    python scripts/restore_run.py <run_id>              # restore to a new branch
    python scripts/restore_run.py <run_id> --diff-only  # just show what changed
    python scripts/restore_run.py <run_id> --worktree   # restore in a git worktree (isolated copy)

This creates a new branch `replay/<run_id>` with the exact code used in that run,
so you can test/re-run without disturbing your current work on main.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNS_DIR = Path("logs/runs")


def _git(*cmd: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git"] + list(cmd),
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"git {' '.join(cmd)} failed: {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Restore code from a tracked run")
    parser.add_argument("run_id", help="The run ID to restore (e.g. 20260323_091500)")
    parser.add_argument("--diff-only", action="store_true", help="Just show the uncommitted diff, don't restore")
    parser.add_argument("--worktree", action="store_true", help="Create an isolated git worktree instead of switching branches")
    args = parser.parse_args()

    meta_path = RUNS_DIR / args.run_id / "meta.json"
    if not meta_path.exists():
        print(f"Run {args.run_id} not found at {meta_path}")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    commit = meta["git_commit"]
    dirty = meta.get("has_uncommitted_changes", False)
    patch_path = RUNS_DIR / args.run_id / "uncommitted.patch"

    print(f"Run:    {args.run_id}")
    print(f"Script: {meta.get('script', '?')}")
    print(f"Commit: {commit} ({meta.get('git_branch', '?')})")
    print(f"Dirty:  {'yes' if dirty else 'no'}")
    if meta.get("results"):
        r = meta["results"]
        print(f"Result: {r.get('trades_filled', 0)} fills, ${r.get('total_spent', 0):.2f} spent")
    print()

    if args.diff_only:
        if dirty and patch_path.exists():
            print("--- Uncommitted changes at time of run ---")
            print(patch_path.read_text())
        else:
            print("No uncommitted changes for this run.")
        return

    branch_name = f"replay/{args.run_id}"

    if args.worktree:
        wt_path = Path(f"worktrees/{args.run_id}")
        print(f"Creating worktree at {wt_path} ...")
        _git("worktree", "add", "-b", branch_name, str(wt_path), commit)
        if dirty and patch_path.exists():
            subprocess.run(
                ["git", "apply", str(patch_path.resolve())],
                cwd=str(wt_path),
                check=True,
            )
            print(f"Applied uncommitted patch.")
        # Copy config + hyperparams
        config_snap = RUNS_DIR / args.run_id / "config.yaml"
        if config_snap.exists():
            import shutil
            shutil.copy(config_snap, wt_path / "config.yaml")
            print("Restored config.yaml from run snapshot.")
        print(f"\nWorktree ready at: {wt_path}")
        print(f"  cd {wt_path} && python {meta.get('script', 'run_live.py')}")
        return

    # Check for clean working tree
    status = _git("status", "--porcelain", check=False)
    if status:
        print("ERROR: You have uncommitted changes. Commit or stash them first.")
        print("  git stash")
        print(f"  python scripts/restore_run.py {args.run_id}")
        print("  # ... test the old run ...")
        print("  git checkout main && git stash pop")
        sys.exit(1)

    print(f"Creating branch: {branch_name}")
    _git("checkout", "-b", branch_name, commit)

    if dirty and patch_path.exists():
        print("Applying uncommitted changes from run...")
        _git("apply", str(patch_path.resolve()))
        print("Applied uncommitted patch.")

    # Restore config snapshot
    config_snap = RUNS_DIR / args.run_id / "config.yaml"
    if config_snap.exists():
        import shutil
        shutil.copy(config_snap, "config.yaml")
        print("Restored config.yaml from run snapshot.")

    print(f"\nCode restored. You're now on branch '{branch_name}'.")
    print(f"To re-run:  python {meta.get('script', 'run_live.py')}")
    print(f"To go back: git checkout main")


if __name__ == "__main__":
    main()
