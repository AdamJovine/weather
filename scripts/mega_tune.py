"""
Mega hyperparameter tuner: round-robin Bayesian Optimization across all models.

Design principles
─────────────────
  Round-robin:  one BO iteration per model per round so no single slow model
                blocks progress on all others.
  Disk-first:   every trial result is written to disk the moment it completes.
                Only the GP state (tiny arrays) is kept in memory per model.
                If the process dies, all completed trials are safe.
  Blacklist:    any model whose evaluation takes > 5 minutes is permanently
                skipped for the rest of the run.

Evaluation settings (fixed, not tuned)
───────────────────────────────────────
  sessions_per_day  = 0   → all available hourly candles (≈ 30-min intervals)
  refit_every       = 1   → daily model refresh with latest observations
  max_bet_dollars   = 10  → hard $10 cap per trade per session
  max_session_trades= 4   → four bets per candle / interval (matches live)
  fee_rate          = 0.02→ 2% Kalshi fee on winnings

Per-model tunable parameters (in addition to shared params)
────────────────────────────────────────────────────────────
  See tune_hyperparams.SEARCH_SPACES or run --list.

Output layout
─────────────
  logs/mega_tune/
    progress.json               ← updated after every round
    summary.json                ← final cross-model leaderboard
    {ModelName}/
      trial_0001.json           ← params + roi + year breakdown + elapsed
      trial_0002.json
      ...
      best.json                 ← current best params (updated as improved)
      blacklisted.json          ← written if model is blacklisted

Prerequisites:
    python scripts/update_data.py   (run automatically unless --skip-update)

Usage:
    python scripts/mega_tune.py
    python scripts/mega_tune.py --n-rounds 50 --n-init 10
    python scripts/mega_tune.py --models GBResidual ARD --skip-update
    python scripts/mega_tune.py --start 2025-12-01 --end 2026-03-15
    python scripts/mega_tune.py --list
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for tune_hyperparams

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import tune_hyperparams as th
from backtesting.config import BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.market_data import KalshiPriceStore
from src.app_config import cfg as _cfg

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

TIMEOUT_SECS = _cfg.mega_tune.timeout_secs
OUT_ROOT     = Path(_cfg.paths.tune_output)


# EvalConfig is an alias for th.TuneConfig — kept for readability in this file.
EvalConfig = th.TuneConfig


# ── Per-model BO state ────────────────────────────────────────────────────────

class ModelState:
    """Holds GP state, observation history, and output path for one model."""

    def __init__(self, name: str, space: list, n_rounds: int, rng: np.random.Generator,
                 out_dir: Path) -> None:
        self.name      = name
        self.space     = space
        self.n_dims    = len(space)
        self.out_dir   = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-allocate observation arrays (max n_rounds rows)
        self.X_obs = np.zeros((n_rounds, self.n_dims))
        self.y_obs = np.full(n_rounds, np.nan)
        self.n_evals = 0

        # Pre-generate LHC points for the random-init phase
        self._lhc_X = th.latin_hypercube(n_rounds, self.n_dims, rng)

        kernel = Matern(nu=2.5, length_scale=np.ones(self.n_dims),
                        length_scale_bounds=(1e-3, 10.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-3, normalize_y=True, n_restarts_optimizer=3,
        )

        self.best_pnl:    float = -np.inf
        self.best_params: dict  = {}
        self.blacklisted: bool  = False

    @property
    def is_active(self) -> bool:
        return not self.blacklisted

    def pick_next_x(self, round_idx: int, n_init: int,
                    rng: np.random.Generator) -> np.ndarray:
        """Return next point in [0,1]^d: LHC during init, GP-EI afterward."""
        if round_idx < n_init:
            return self._lhc_X[round_idx]

        mask = np.isfinite(self.y_obs[:self.n_evals])
        if mask.sum() < 2:
            return rng.uniform(size=self.n_dims)

        self.gp.fit(self.X_obs[:self.n_evals][mask], self.y_obs[:self.n_evals][mask])
        X_cand = rng.uniform(size=(5000, self.n_dims))
        ei = th.expected_improvement(
            X_cand, self.y_obs[:self.n_evals][mask], self.gp, xi=0.01,
        )
        return X_cand[int(np.argmax(ei))]

    def record(self, x: np.ndarray, pnl: float) -> None:
        self.X_obs[self.n_evals] = x
        self.y_obs[self.n_evals] = pnl
        self.n_evals += 1
        # Flush GP state to disk so no observations are memory-only
        np.save(self.out_dir / "X_obs.npy", self.X_obs[:self.n_evals])
        np.save(self.out_dir / "y_obs.npy", self.y_obs[:self.n_evals])

    def load_state(self) -> None:
        """Restore GP observations and best params from disk (for --resume)."""
        x_path    = self.out_dir / "X_obs.npy"
        y_path    = self.out_dir / "y_obs.npy"
        best_path = self.out_dir / "best.json"
        bl_path   = self.out_dir / "blacklisted.json"

        if x_path.exists() and y_path.exists():
            X = np.load(x_path)
            y = np.load(y_path)
            n = len(y)
            cap = self.X_obs.shape[0]
            n = min(n, cap)
            self.X_obs[:n] = X[:n]
            self.y_obs[:n] = y[:n]
            self.n_evals   = n

        if best_path.exists():
            best = json.load(open(best_path))
            self.best_pnl    = float(best.get("total_pnl", -np.inf))
            self.best_params = best.get("params", {})

        if bl_path.exists():
            self.blacklisted = True


# ── Disk I/O helpers (called immediately after each eval) ────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_trial(state: ModelState, params: dict, pnl: float, year_pnls: dict,
               elapsed: float, round_idx: int) -> None:
    record = {
        "trial":     state.n_evals,   # already incremented by record()
        "round":     round_idx + 1,
        "timestamp": _now_iso(),
        "elapsed_s": round(elapsed, 1),
        "total_pnl": round(float(pnl), 2) if np.isfinite(pnl) else None,
        "year_pnls": {str(k): round(float(v), 2) for k, v in year_pnls.items()},
        "params":    {k: (round(float(v), 8) if isinstance(v, float) else int(v))
                      for k, v in params.items()},
    }
    path = state.out_dir / f"trial_{state.n_evals:04d}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def save_best(state: ModelState) -> None:
    record = {
        "timestamp":   _now_iso(),
        "total_pnl":   round(state.best_pnl, 2),
        "n_evals":     state.n_evals,
        "params":      {k: (round(float(v), 8) if isinstance(v, float) else int(v))
                        for k, v in state.best_params.items()},
    }
    with open(state.out_dir / "best.json", "w") as f:
        json.dump(record, f, indent=2)


def save_blacklisted(state: ModelState, reason: str) -> None:
    record = {"timestamp": _now_iso(), "reason": reason}
    with open(state.out_dir / "blacklisted.json", "w") as f:
        json.dump(record, f, indent=2)


def save_progress(states: list[ModelState], n_rounds_done: int,
                  n_rounds_total: int, run_dir: Path) -> None:
    rows = []
    for s in states:
        valid = s.y_obs[:s.n_evals][np.isfinite(s.y_obs[:s.n_evals])]
        rows.append({
            "model":       s.name,
            "n_evals":     s.n_evals,
            "best_pnl":    round(s.best_pnl, 2) if np.isfinite(s.best_pnl) else None,
            "mean_pnl":    round(float(valid.mean()), 2) if len(valid) else None,
            "blacklisted": s.blacklisted,
        })
    rows.sort(key=lambda r: (r["best_pnl"] or -999), reverse=True)
    doc = {
        "timestamp":      _now_iso(),
        "rounds_done":    n_rounds_done,
        "rounds_total":   n_rounds_total,
        "models":         rows,
    }
    with open(run_dir / "progress.json", "w") as f:
        json.dump(doc, f, indent=2)


def save_summary(states: list[ModelState], run_dir: Path) -> None:
    rows = []
    for s in states:
        valid = s.y_obs[:s.n_evals][np.isfinite(s.y_obs[:s.n_evals])]
        rows.append({
            "model":        s.name,
            "n_evals":      s.n_evals,
            "best_pnl":     round(s.best_pnl, 2) if np.isfinite(s.best_pnl) else None,
            "mean_pnl":     round(float(valid.mean()), 2) if len(valid) else None,
            "std_pnl":      round(float(valid.std()),  2) if len(valid) > 1 else None,
            "pct_positive": round(float((valid > 0).mean()), 4) if len(valid) else None,
            "blacklisted":  s.blacklisted,
            "best_params":  {k: (round(float(v), 8) if isinstance(v, float) else int(v))
                             for k, v in s.best_params.items()},
        })
    rows.sort(key=lambda r: (r["best_pnl"] or -999), reverse=True)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)


# ── Update-data helpers ───────────────────────────────────────────────────────

def _update_data_running() -> bool:
    """Return True if an update_data.py process is currently running."""
    result = subprocess.run(
        ["pgrep", "-f", "update_data.py"],
        capture_output=True,
    )
    return result.returncode == 0


def wait_for_update_data(poll_secs: int = 30) -> None:
    """Block until no update_data.py process is detected."""
    if not _update_data_running():
        print("No update_data.py process detected — proceeding immediately.")
        return
    print("update_data.py is running. Waiting for it to finish...")
    while _update_data_running():
        print(f"  still running... checking again in {poll_secs}s", flush=True)
        time.sleep(poll_secs)
    print("update_data.py finished. Proceeding.\n")


# ── Worker process helpers ────────────────────────────────────────────────────
# dataset and price_store are large; initialise them once per worker process
# so they aren't re-pickled on every trial submission.

_w_dataset     = None
_w_price_store = None
_w_eval_cfg    = None
_w_trades_dir  = None   # root dir under which trades CSVs are written per trial


def _worker_init(dataset, price_store, eval_cfg, trades_dir) -> None:
    global _w_dataset, _w_price_store, _w_eval_cfg, _w_trades_dir
    _w_dataset     = dataset
    _w_price_store = price_store
    _w_eval_cfg    = eval_cfg
    _w_trades_dir  = trades_dir


def _worker_call(model_name: str, params: dict, trial_num: int) -> tuple[float, dict, float]:
    """Run one trial in a worker process. Returns (pnl, year_pnls, elapsed_s)."""
    t0 = time.time()
    pnl, year_pnls, run = th.evaluate_params(
        model_name, params, _w_dataset, _w_price_store, _w_eval_cfg,
        return_run=True,
    )
    if _w_trades_dir is not None and run is not None and not run.trades.empty:
        out = Path(_w_trades_dir) / model_name / f"trial_{trial_num:04d}_trades.csv"
        run.trades.to_csv(out, index=False)
    return pnl, year_pnls, time.time() - t0


# ── Round-robin BO loop ───────────────────────────────────────────────────────

def run_mega_tune(
    models:      list[ModelState],
    dataset,
    price_store: KalshiPriceStore,
    eval_cfg:    EvalConfig,
    n_rounds:    int,
    n_init:      int,
    rng:         np.random.Generator,
    run_dir:     Path,
    n_jobs:      int = 4,
    start_round: int = 0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_worker_init,
        initargs=(dataset, price_store, eval_cfg, run_dir),
    ) as pool:
        for round_idx in range(start_round, n_rounds):
            active = [m for m in models if m.is_active]
            if not active:
                print("\nAll models blacklisted — stopping early.")
                break

            sep = "=" * 70
            phase = "init" if round_idx < n_init else "BO"
            print(f"\n{sep}")
            print(f"  Round {round_idx + 1}/{n_rounds}  |  "
                  f"{len(active)} active model(s)  |  "
                  f"{phase} phase  |  {n_jobs} parallel workers")
            print(sep)

            # Pick next points for all active models before submitting
            # (GP state is only updated after results come back)
            submissions: list[tuple] = []   # (future, state, x_next, params)
            for state in active:
                x_next = state.pick_next_x(round_idx, n_init, rng)
                params = th.decode(x_next, state.space)
                params_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
                print(f"  [{state.name}] trial {state.n_evals + 1}  ({phase})")
                print(f"    {params_str}", flush=True)
                future = pool.submit(_worker_call, state.name, params, state.n_evals + 1)
                submissions.append((future, state, x_next, params))

            # Collect results as workers finish (order may differ from submission)
            future_map = {f: (s, x, p) for f, s, x, p in submissions}
            for future in as_completed(future_map):
                state, x_next, params = future_map[future]
                try:
                    pnl, year_pnls, elapsed = future.result()
                except Exception as exc:
                    pnl, year_pnls, elapsed = -np.inf, {}, 0.0
                    print(f"  [{state.name}] exception: {exc}")

                # ── blacklist if too slow ──────────────────────────────────
                if elapsed > TIMEOUT_SECS:
                    reason = f"eval took {elapsed:.0f}s > {TIMEOUT_SECS}s limit"
                    print(f"  [{state.name}] BLACKLISTED — {reason}")
                    state.blacklisted = True
                    save_blacklisted(state, reason)

                # ── record and persist immediately ─────────────────────────
                state.record(x_next, pnl)
                save_trial(state, params, pnl, year_pnls, elapsed, round_idx)

                if pnl > state.best_pnl:
                    state.best_pnl = pnl
                    state.best_params = params
                    save_best(state)
                    marker = " ← new best"
                else:
                    marker = ""

                pnl_str = f"${pnl:+.2f}" if np.isfinite(pnl) else "FAIL"
                print(f"  [{state.name}] PnL={pnl_str}  "
                      f"best=${state.best_pnl:+.2f}  ({elapsed:.1f}s){marker}")
                if year_pnls:
                    yr_str = "  ".join(
                        f"{yr}:${p:+.2f}" for yr, p in sorted(year_pnls.items())
                    )
                    print(f"             per-year: {yr_str}", flush=True)

            # ── after every round: flush progress and summary to disk ──────
            save_progress(models, round_idx + 1, n_rounds, run_dir)
            save_summary(models, run_dir)

    sep = "=" * 70
    print(f"\n{sep}")
    print("FINAL LEADERBOARD  (sorted by best PnL)")
    print(sep)
    ranked = sorted([m for m in models if m.n_evals > 0],
                    key=lambda m: m.best_pnl, reverse=True)
    for i, m in enumerate(ranked, 1):
        valid = m.y_obs[:m.n_evals][np.isfinite(m.y_obs[:m.n_evals])]
        flag = " [BLACKLISTED]" if m.blacklisted else ""
        print(f"  {i:>2}. {m.name:<18}  best=${m.best_pnl:+.2f}  "
              f"mean=${valid.mean():+.2f}  evals={m.n_evals}{flag}")
    print(sep)
    print(f"\n  Results → {run_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/mega_tune.py",
        description="Round-robin Bayesian Optimization across all temperature models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", default=None,
                   choices=sorted(th.SEARCH_SPACES.keys()), metavar="MODEL",
                   help="Subset of models to tune. Default: all.")
    _mt = _cfg.mega_tune
    p.add_argument("--n-rounds", type=int, default=_mt.n_rounds,
                   help=f"Total BO rounds (each model gets one eval per round). [{_mt.n_rounds}]")
    p.add_argument("--n-init", type=int, default=_mt.n_init,
                   help=f"Random init rounds before GP-EI kicks in. [{_mt.n_init}]")
    p.add_argument("--seed", type=int, default=_mt.seed)
    p.add_argument("--start", default=_mt.start_date, metavar="YYYY-MM-DD",
                   help=f"Backtest window start. [{_mt.start_date}]")
    p.add_argument("--end",   default=_mt.end_date, metavar="YYYY-MM-DD",
                   help=f"Backtest window end.   [{_mt.end_date}]")
    p.add_argument("--max-bet-dollars", type=float, default=_mt.max_bet_dollars, metavar="$",
                   help=f"Hard dollar cap per trade per session. [{_mt.max_bet_dollars}]")
    p.add_argument("--max-session-trades", type=int, default=_mt.max_session_trades,
                   help=f"Max trades per candle / interval. [{_mt.max_session_trades}]")
    p.add_argument("--max-daily-trades", type=int, default=_mt.max_daily_trades)
    p.add_argument("--refit-every", type=int, default=_mt.refit_every, metavar="DAYS",
                   help=f"Model refit cadence. 1=daily. [{_mt.refit_every}]")
    p.add_argument("--fee-rate", type=float, default=_mt.fee_rate)
    p.add_argument("--pessimistic-pricing", action="store_true",
                   help="Buy at candle high, sell at candle low (worst-case fill simulation).")
    p.add_argument("--recency-halflife", type=int, default=None, metavar="DAYS",
                   help="Exponential recency weighting: trades N days ago count half as much. "
                        "None = flat. Try 90 to weight current season 4x over 6 months ago.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip data quality validation.")

    update_grp = p.add_mutually_exclusive_group()
    update_grp.add_argument("--skip-update", action="store_true",
                            help="Skip update_data.py entirely.")
    update_grp.add_argument("--wait-for-update", action="store_true",
                            help=(
                                "Don't launch update_data.py — instead wait for "
                                "an already-running instance to finish, then proceed."
                            ))
    p.add_argument("--n-jobs", type=int, default=_cfg.mega_tune.n_jobs,
                   help=f"Parallel worker processes. [{_cfg.mega_tune.n_jobs}]")
    p.add_argument("--list", action="store_true",
                   help="Print models + search spaces then exit.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last completed round (reads progress.json + X/y_obs.npy).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.list:
        print("Models and their tunable parameters:\n")
        for name, space in sorted(th.SEARCH_SPACES.items()):
            print(f"  {name}")
            for pname, scale, lo, hi in space:
                print(f"    {pname:<22} {scale:<8} [{lo}, {hi}]")
        return

    models_to_run = args.models or sorted(th.SEARCH_SPACES.keys())

    # ── Step 1: refresh data ──────────────────────────────────────────────────
    if args.skip_update:
        print("Skipping update_data.py (--skip-update)")
    elif args.wait_for_update:
        wait_for_update_data()
    else:
        print("Running scripts/update_data.py ...")
        result = subprocess.run(
            [sys.executable, "scripts/update_data.py"],
            check=False,
        )
        if result.returncode != 0:
            print("WARNING: update_data.py returned non-zero — continuing anyway.")

    # ── Step 2: load data once (shared across all models and all trials) ──────
    print("\nLoading Kalshi price history...")
    price_store = KalshiPriceStore().load()
    print(price_store.coverage_summary())

    _mt = _cfg.mega_tune
    eval_cfg = EvalConfig(
        start_date=args.start,
        end_date=args.end,
        sessions_per_day=0,         # all candles → ~30-min granularity
        max_bet_dollars=args.max_bet_dollars,
        max_session_trades=args.max_session_trades,
        max_daily_trades=args.max_daily_trades,
        refit_every=args.refit_every,
        fee_rate=args.fee_rate,
        cash_reserve_fraction=_mt.cash_reserve_fraction,
        rotation_min_edge_gain=_mt.rotation_min_edge_gain,
        initial_bankroll=_mt.initial_bankroll,
        pessimistic_pricing=args.pessimistic_pricing,
        recency_halflife_days=args.recency_halflife,
    )

    print("\nLoading feature data...")
    bootstrap_cfg = BacktestConfig(
        model_name="BayesianRidge",
        start_date=eval_cfg.start_date,
        end_date=eval_cfg.end_date,
    )
    dataset = DataLoader().load(
        bootstrap_cfg, skip_validation=args.no_validate,
    )

    # ── Step 3: initialise per-model BO state ─────────────────────────────────
    rng = np.random.default_rng(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_init = min(args.n_init, args.n_rounds)

    # ── Step 3b: restore prior state if resuming ──────────────────────────────
    start_round = 0
    if args.resume:
        # Find the most recent timestamped run directory
        existing = sorted(OUT_ROOT.glob("20??????_??????"))
        if existing:
            run_dir = existing[-1]
            progress_path = run_dir / "progress.json"
            if progress_path.exists():
                prog = json.load(open(progress_path))
                start_round = int(prog.get("rounds_done", 0))
                print(f"Resuming run {run_dir.name} from round "
                      f"{start_round}/{args.n_rounds} "
                      f"(read from {progress_path})")
            else:
                print(f"Found run dir {run_dir.name} but no progress.json — starting fresh.")
                run_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            print("--resume specified but no prior run dirs found — starting fresh.")
            run_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    model_states = [
        ModelState(
            name=name,
            space=th.SEARCH_SPACES[name],
            n_rounds=args.n_rounds,
            rng=rng,
            out_dir=run_dir / name,
        )
        for name in models_to_run
    ]

    if args.resume:
        for state in model_states:
            state.load_state()
            if state.n_evals:
                flag = " [BLACKLISTED]" if state.blacklisted else ""
                print(f"  {state.name}: loaded {state.n_evals} obs, "
                      f"best=${state.best_pnl:+.2f}{flag}")
        print()

    if start_round >= args.n_rounds:
        print(f"Already completed {start_round}/{args.n_rounds} rounds — nothing to do.")
        print("Use --n-rounds N with N > {start_round} to run more rounds.")
        return

    print(f"\n{'='*70}")
    print(f"MEGA TUNE{'  (RESUMED)' if args.resume else ''}")
    print(f"  Models:        {', '.join(models_to_run)}")
    print(f"  Rounds:        {start_round} done → {args.n_rounds} total  (init: {n_init})")
    print(f"  Window:        {eval_cfg.start_date} → {eval_cfg.end_date}")
    print(f"  Sessions/day:  all  |  refit_every: {eval_cfg.refit_every}d")
    print(f"  Max bet:       ${eval_cfg.max_bet_dollars}/trade  |  "
          f"{eval_cfg.max_session_trades}/session  |  "
          f"{eval_cfg.max_daily_trades}/day")
    print(f"  Fee:           {eval_cfg.fee_rate:.0%}  |  "
          f"Timeout: {TIMEOUT_SECS}s  |  Workers: {args.n_jobs}")
    print(f"  Cash reserve:  {eval_cfg.cash_reserve_fraction:.0%}  |  "
          f"Rotation min edge gain: {eval_cfg.rotation_min_edge_gain:.0%}  |  "
          f"Initial bankroll: ${eval_cfg.initial_bankroll:.2f}")
    print(f"  Output:        {run_dir}/")
    print(f"{'='*70}\n")

    # ── Step 4: run ───────────────────────────────────────────────────────────
    run_mega_tune(
        models=model_states,
        dataset=dataset,
        price_store=price_store,
        eval_cfg=eval_cfg,
        n_rounds=args.n_rounds,
        n_init=n_init,
        rng=rng,
        run_dir=run_dir,
        n_jobs=args.n_jobs,
        start_round=start_round,
    )


if __name__ == "__main__":
    main()
