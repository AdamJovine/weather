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
  max_session_trades= 1   → one bet per candle / interval
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
from dataclasses import dataclass, field
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
from backtesting.engine import BacktestEngine
from backtesting.market_data import KalshiPriceStore

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

TIMEOUT_SECS   = 300       # blacklist model if single eval > 5 min
OUT_ROOT       = Path("logs/mega_tune")


# ── Evaluation config (fixed for all trials) ──────────────────────────────────

@dataclass
class EvalConfig:
    """
    Backtest settings that are fixed across all BO trials.
    These are not themselves tuned — they define the evaluation protocol.
    """
    start_date:          str   = "2025-12-01"
    end_date:            str   = "2026-03-15"   # ~100 trading days
    sessions_per_day:    int   = 0              # all candles → ~30-min granularity
    refit_every:         int   = 1              # model refreshes every day
    max_bet_dollars:     float = 10.0           # $10 hard cap per trade
    max_session_trades:  int   = 1              # 1 bet per candle / interval
    max_daily_trades:    int   = 100            # generous daily cap
    min_train_rows:      int   = 365
    fee_rate:            float = 0.02


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

        self.best_roi:    float = -np.inf
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

    def record(self, x: np.ndarray, roi: float) -> None:
        self.X_obs[self.n_evals] = x
        self.y_obs[self.n_evals] = roi
        self.n_evals += 1


# ── Disk I/O helpers (called immediately after each eval) ────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_trial(state: ModelState, params: dict, roi: float, year_rois: dict,
               elapsed: float, round_idx: int) -> None:
    record = {
        "trial":     state.n_evals,   # already incremented by record()
        "round":     round_idx + 1,
        "timestamp": _now_iso(),
        "elapsed_s": round(elapsed, 1),
        "roi":       round(float(roi), 6) if np.isfinite(roi) else None,
        "year_rois": {str(k): round(float(v), 6) for k, v in year_rois.items()},
        "params":    {k: (round(float(v), 8) if isinstance(v, float) else int(v))
                      for k, v in params.items()},
    }
    path = state.out_dir / f"trial_{state.n_evals:04d}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def save_best(state: ModelState) -> None:
    record = {
        "timestamp":   _now_iso(),
        "roi":         round(state.best_roi, 6),
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
                  n_rounds_total: int) -> None:
    rows = []
    for s in states:
        valid = s.y_obs[:s.n_evals][np.isfinite(s.y_obs[:s.n_evals])]
        rows.append({
            "model":       s.name,
            "n_evals":     s.n_evals,
            "best_roi":    round(s.best_roi, 6) if np.isfinite(s.best_roi) else None,
            "mean_roi":    round(float(valid.mean()), 6) if len(valid) else None,
            "blacklisted": s.blacklisted,
        })
    rows.sort(key=lambda r: (r["best_roi"] or -999), reverse=True)
    doc = {
        "timestamp":      _now_iso(),
        "rounds_done":    n_rounds_done,
        "rounds_total":   n_rounds_total,
        "models":         rows,
    }
    with open(OUT_ROOT / "progress.json", "w") as f:
        json.dump(doc, f, indent=2)


def save_summary(states: list[ModelState]) -> None:
    rows = []
    for s in states:
        valid = s.y_obs[:s.n_evals][np.isfinite(s.y_obs[:s.n_evals])]
        rows.append({
            "model":        s.name,
            "n_evals":      s.n_evals,
            "best_roi":     round(s.best_roi, 6) if np.isfinite(s.best_roi) else None,
            "mean_roi":     round(float(valid.mean()), 6) if len(valid) else None,
            "std_roi":      round(float(valid.std()),  6) if len(valid) > 1 else None,
            "pct_positive": round(float((valid > 0).mean()), 4) if len(valid) else None,
            "blacklisted":  s.blacklisted,
            "best_params":  {k: (round(float(v), 8) if isinstance(v, float) else int(v))
                             for k, v in s.best_params.items()},
        })
    rows.sort(key=lambda r: (r["best_roi"] or -999), reverse=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)


# ── Single-trial evaluator ────────────────────────────────────────────────────

def evaluate_trial(
    model_name: str,
    params: dict,
    dataset,
    price_store: KalshiPriceStore,
    eval_cfg: EvalConfig,
) -> tuple[float, dict]:
    """
    Run one full backtest for the given params and return (roi, year_rois).
    Returns (-1.0, {}) on any failure.
    spread_alpha is patched at the module level (single-threaded; always restored).
    """
    import src.model as _mmod
    orig_spread = _mmod.SPREAD_ALPHA
    _mmod.SPREAD_ALPHA = float(params.get("spread_alpha", orig_spread))

    try:
        cfg = BacktestConfig(
            model_name=model_name,
            lookback=int(params.get("lookback", 730)),
            kelly_fraction=float(params.get("kelly_fraction", 0.33)),
            max_bet_fraction=float(params.get("max_bet_frac", 0.05)),
            max_bet_dollars=eval_cfg.max_bet_dollars,
            min_edge=float(params.get("min_edge", 0.05)),
            n_runs=1,
            seed=42,
            start_date=eval_cfg.start_date,
            end_date=eval_cfg.end_date,
            sessions_per_day=eval_cfg.sessions_per_day,
            max_daily_trades=eval_cfg.max_daily_trades,
            max_session_trades=eval_cfg.max_session_trades,
            refit_every=eval_cfg.refit_every,
            min_train_rows=eval_cfg.min_train_rows,
            fee_rate=eval_cfg.fee_rate,
            verbose=False,
        )
        _params = dict(params)
        engine = BacktestEngine(
            dataset, cfg, price_store,
            model_factory=lambda: th.make_model(model_name, _params),
        )
        results = engine.run()
    except Exception:
        return -1.0, {}
    finally:
        _mmod.SPREAD_ALPHA = orig_spread

    run = results.runs[0]
    return run.roi, th._year_rois(run.trades)


# ── Round-robin BO loop ───────────────────────────────────────────────────────

def run_mega_tune(
    models:      list[ModelState],
    dataset,
    price_store: KalshiPriceStore,
    eval_cfg:    EvalConfig,
    n_rounds:    int,
    n_init:      int,
    rng:         np.random.Generator,
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for round_idx in range(n_rounds):
        active = [m for m in models if m.is_active]
        if not active:
            print("\nAll models blacklisted — stopping early.")
            break

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Round {round_idx + 1}/{n_rounds}  |  "
              f"{len(active)} active model(s)  |  "
              f"{'init' if round_idx < n_init else 'BO'} phase")
        print(sep)

        for state in active:
            x_next = state.pick_next_x(round_idx, n_init, rng)
            params = th.decode(x_next, state.space)

            phase = "init" if round_idx < n_init else "GP-EI"
            print(f"\n  [{state.name}] trial {state.n_evals + 1}  ({phase})",
                  flush=True)
            params_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
            print(f"    {params_str}")

            t0 = time.time()
            try:
                roi, year_rois = evaluate_trial(
                    state.name, params, dataset, price_store, eval_cfg,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  [{state.name}] exception ({elapsed:.0f}s): {exc}")
                roi, year_rois = -1.0, {}

            elapsed = time.time() - t0

            # ── blacklist if too slow ──────────────────────────────────────
            if elapsed > TIMEOUT_SECS:
                reason = f"eval took {elapsed:.0f}s > {TIMEOUT_SECS}s limit"
                print(f"  [{state.name}] BLACKLISTED — {reason}")
                state.blacklisted = True
                save_blacklisted(state, reason)
                # still record this result (model may have valid roi)

            # ── record observation and save to disk immediately ────────────
            state.record(x_next, roi)

            save_trial(state, params, roi, year_rois, elapsed, round_idx)

            if roi > state.best_roi:
                state.best_roi = roi
                state.best_params = params
                save_best(state)
                marker = " ← new best"
            else:
                marker = ""

            roi_str = f"{roi:+.4f}" if np.isfinite(roi) else "FAIL"
            print(f"  [{state.name}] ROI={roi_str}  "
                  f"best={state.best_roi:+.4f}  ({elapsed:.1f}s){marker}")
            if year_rois:
                yr_str = "  ".join(
                    f"{yr}:{r:+.3f}" for yr, r in sorted(year_rois.items())
                )
                print(f"             per-year: {yr_str}")

        # ── after every round: update progress file ────────────────────────
        save_progress(models, round_idx + 1, n_rounds)

    # ── final summary ─────────────────────────────────────────────────────────
    save_summary(models)

    sep = "=" * 70
    print(f"\n{sep}")
    print("FINAL LEADERBOARD  (sorted by best ROI)")
    print(sep)
    ranked = sorted([m for m in models if m.n_evals > 0],
                    key=lambda m: m.best_roi, reverse=True)
    for i, m in enumerate(ranked, 1):
        valid = m.y_obs[:m.n_evals][np.isfinite(m.y_obs[:m.n_evals])]
        flag = " [BLACKLISTED]" if m.blacklisted else ""
        print(f"  {i:>2}. {m.name:<18}  best={m.best_roi:+.4f}  "
              f"mean={valid.mean():+.4f}  evals={m.n_evals}{flag}")
    print(sep)
    print(f"\n  Results → {OUT_ROOT}/")


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
    p.add_argument("--n-rounds", type=int, default=50,
                   help="Total BO rounds (each model gets one eval per round). [50]")
    p.add_argument("--n-init", type=int, default=10,
                   help="Random init rounds before GP-EI kicks in. [10]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", default="2025-12-01", metavar="YYYY-MM-DD",
                   help="Backtest window start. [2025-12-01]")
    p.add_argument("--end",   default="2026-03-15", metavar="YYYY-MM-DD",
                   help="Backtest window end.   [2026-03-15]")
    p.add_argument("--max-bet-dollars", type=float, default=10.0, metavar="$",
                   help="Hard dollar cap per trade per session. [10]")
    p.add_argument("--max-session-trades", type=int, default=1,
                   help="Max trades per candle / 30-min interval. [1]")
    p.add_argument("--max-daily-trades", type=int, default=100)
    p.add_argument("--refit-every", type=int, default=1, metavar="DAYS",
                   help="Model refit cadence. 1=daily. [1]")
    p.add_argument("--fee-rate", type=float, default=0.02)
    p.add_argument("--no-validate", action="store_true",
                   help="Skip data quality validation.")
    p.add_argument("--skip-update", action="store_true",
                   help="Skip running scripts/update_data.py at startup.")
    p.add_argument("--list", action="store_true",
                   help="Print models + search spaces then exit.")
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
    if not args.skip_update:
        print("Running scripts/update_data.py ...")
        result = subprocess.run(
            [sys.executable, "scripts/update_data.py"],
            check=False,
        )
        if result.returncode != 0:
            print("WARNING: update_data.py returned non-zero — continuing anyway.")
    else:
        print("Skipping update_data.py (--skip-update)")

    # ── Step 2: load data once (shared across all models and all trials) ──────
    print("\nLoading Kalshi price history...")
    price_store = KalshiPriceStore(data_dir="data").load()
    print(price_store.coverage_summary())

    eval_cfg = EvalConfig(
        start_date=args.start,
        end_date=args.end,
        max_bet_dollars=args.max_bet_dollars,
        max_session_trades=args.max_session_trades,
        max_daily_trades=args.max_daily_trades,
        refit_every=args.refit_every,
        fee_rate=args.fee_rate,
    )

    print("\nLoading feature data...")
    bootstrap_cfg = BacktestConfig(
        model_name="BayesianRidge",
        start_date=eval_cfg.start_date,
        end_date=eval_cfg.end_date,
    )
    dataset = DataLoader(data_dir="data").load(
        bootstrap_cfg, skip_validation=args.no_validate,
    )

    # ── Step 3: initialise per-model BO state ─────────────────────────────────
    rng = np.random.default_rng(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    model_states = [
        ModelState(
            name=name,
            space=th.SEARCH_SPACES[name],
            n_rounds=args.n_rounds,
            rng=rng,
            out_dir=OUT_ROOT / name,
        )
        for name in models_to_run
    ]

    n_init = min(args.n_init, args.n_rounds)

    print(f"\n{'='*70}")
    print(f"MEGA TUNE")
    print(f"  Models:        {', '.join(models_to_run)}")
    print(f"  Rounds:        {args.n_rounds}  (init: {n_init})")
    print(f"  Window:        {eval_cfg.start_date} → {eval_cfg.end_date}")
    print(f"  Sessions/day:  all  |  refit_every: {eval_cfg.refit_every}d")
    print(f"  Max bet:       ${eval_cfg.max_bet_dollars}/trade  |  "
          f"{eval_cfg.max_session_trades}/session  |  "
          f"{eval_cfg.max_daily_trades}/day")
    print(f"  Fee:           {eval_cfg.fee_rate:.0%}  |  "
          f"Timeout: {TIMEOUT_SECS}s")
    print(f"  Output:        {OUT_ROOT}/")
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
    )


if __name__ == "__main__":
    main()
