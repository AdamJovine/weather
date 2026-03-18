"""
Bayesian Hyperparameter Optimization for weather temperature models.

Objective: walk-forward ROI evaluated by BacktestEngine against real Kalshi
historical prices. Every trial is exactly as principled as a real backtest —
no synthetic prices, no subsampled rows.

For speed during tuning, trials use:
  --refit-every  14    (bi-weekly model refits instead of daily)
  --sessions-per-day 1 (first hourly snapshot per day only)

These can be loosened with --refit-every 1 --sessions-per-day 0 for a
higher-fidelity (slower) objective.

Prerequisites:
    python scripts/download_kalshi_history.py   # real price history

Usage:
    python scripts/tune_hyperparams.py --model GBResidual
    python scripts/tune_hyperparams.py --model NGBoost --n-iter 30
    python scripts/tune_hyperparams.py --model all             # all models sequentially
    python scripts/tune_hyperparams.py --list                  # list models + their params

Available models:
    ARD, BaggingRidge, BayesianRidge, ExtraTrees, GBResidual,
    InteractionBayes, KernelRidge, NGBoost, QuantileGB, RandomForest
"""

import sys
import argparse
import json
import time
import warnings
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from backtesting.config import BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.engine import BacktestEngine
from backtesting.market_data import KalshiPriceStore
from backtesting.results import RunResult
from src.app_config import cfg as _cfg

warnings.filterwarnings("ignore")


# ── Tuning-run settings (fixed across all trials) ────────────────────────────

_tc = _cfg.tuning


@dataclass
class TuneConfig:
    """
    Backtest settings that are fixed for all BO trials — not themselves tuned.

    refit_every and sessions_per_day control the speed/fidelity tradeoff:
      faster  → refit_every=14, sessions_per_day=1
      slower  → refit_every=1,  sessions_per_day=0 (all candles)
    """
    start_date: str = _tc.start_date
    end_date: str = _tc.end_date
    refit_every: int = _tc.refit_every      # model refit cadence in calendar days
    sessions_per_day: int = _tc.sessions_per_day  # 1 = first candle per day; 0 = all candles
    max_daily_trades: int = _tc.max_daily_trades
    max_session_trades: int = _tc.max_session_trades
    min_train_rows: int = _tc.min_train_rows
    max_bet_dollars: Optional[float] = None  # hard dollar cap per trade (None = no cap)
    fee_rate: float = _tc.fee_rate           # Kalshi fee fraction on winnings
    cash_reserve_fraction: float = _tc.cash_reserve_fraction
    rotation_min_edge_gain: float = _tc.rotation_min_edge_gain
    initial_bankroll: float = _tc.initial_bankroll
    pessimistic_pricing: bool = False
    recency_halflife_days: Optional[int] = None  # None = flat weighting; N = trades N days ago count half as much


# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Each entry: (param_name, scale, low, high)
#   "log"    → GP operates on log10(value)
#   "linear" → GP operates on raw value
#   "int"    → integer; rounded when decoding from unit hypercube

_SHARED_PARAMS = [
    # Training
    ("lookback",        "int",    180, 2500),   # rows of history to train on
    # Sizing
    ("kelly_fraction",  "linear", 0.05,  1.0),
    ("max_bet_frac",    "linear", 0.01,  0.20),
    # Edge
    ("min_edge",        "linear", 0.00,  0.15),
    # Confidence — model must be >= 0.5 + min_confidence certain; controls win rate
    ("min_confidence",  "linear", 0.00,  0.35),
    # Uncertainty — widens model's predictive sigma on high-spread days
    ("spread_alpha",    "linear", 0.00,  1.00),
    # Probability band — skip near-certain outcomes
    ("min_fair_p",      "linear", 0.02,  0.20),
    ("max_fair_p",      "linear", 0.80,  0.98),
    ("sigma_floor",     "linear", 2.0,   7.0),   # minimum predictive sigma — prevents overconfidence on tail events
]

SEARCH_SPACES = {
    "BayesianRidge": [
        ("alpha_1",  "log", 1e-8, 1e-2),
        ("alpha_2",  "log", 1e-8, 1e-2),
        ("lambda_1", "log", 1e-8, 1e-2),
        ("lambda_2", "log", 1e-8, 1e-2),
    ] + _SHARED_PARAMS,

    "ARD": [
        ("alpha_1",  "log", 1e-8, 1e-2),
        ("alpha_2",  "log", 1e-8, 1e-2),
        ("lambda_1", "log", 1e-8, 1e-2),
        ("lambda_2", "log", 1e-8, 1e-2),
    ] + _SHARED_PARAMS,

    "InteractionBayes": [
        ("alpha_1",  "log", 1e-8, 1e-2),
        ("alpha_2",  "log", 1e-8, 1e-2),
        ("lambda_1", "log", 1e-8, 1e-2),
        ("lambda_2", "log", 1e-8, 1e-2),
    ] + _SHARED_PARAMS,

    "NGBoost": [
        ("n_estimators",   "int",     50,  400),
        ("learning_rate",  "log",   0.01,  0.3),
        ("epistemic_frac", "linear", 0.05, 0.5),
    ] + _SHARED_PARAMS,

    "RandomForest": [
        ("n_estimators", "int",  20, 200),
        ("max_depth",    "int",   3,  15),
    ] + _SHARED_PARAMS,

    "ExtraTrees": [
        ("n_estimators", "int",  20, 200),
        ("max_depth",    "int",   3,  15),
    ] + _SHARED_PARAMS,

    "QuantileGB": [
        ("max_iter", "int", 30, 300),
    ] + _SHARED_PARAMS,

    "GBResidual": [
        ("n_estimators", "int",  30, 300),
        ("max_depth",    "int",   2,   8),
    ] + _SHARED_PARAMS,

    "BaggingRidge": [
        ("n_estimators", "int",   10, 100),
        ("ridge_alpha",  "log", 0.01, 100.0),
    ] + _SHARED_PARAMS,

    "KernelRidge": [
        ("alpha", "log",  0.01, 10.0),
        ("gamma", "log", 0.001,  1.0),
        ("n_boot", "int",    5,   30),
    ] + _SHARED_PARAMS,
}


# ── Param space encode / decode ───────────────────────────────────────────────

def encode(params: dict, space: list) -> np.ndarray:
    """Map a params dict to a point in [0, 1]^d."""
    x = []
    for name, scale, lo, hi in space:
        v = params[name]
        if scale == "log":
            x.append((np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)))
        else:
            x.append((v - lo) / (hi - lo))
    return np.array(x)


def decode(x: np.ndarray, space: list) -> dict:
    """Map a point in [0, 1]^d back to a params dict."""
    params = {}
    for xi, (name, scale, lo, hi) in zip(x, space):
        xi = float(np.clip(xi, 0.0, 1.0))
        if scale == "log":
            v = 10 ** (xi * (np.log10(hi) - np.log10(lo)) + np.log10(lo))
        elif scale == "int":
            v = int(round(xi * (hi - lo) + lo))
        else:
            v = xi * (hi - lo) + lo
        params[name] = v
    return params


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube samples in [0, 1]^d."""
    pts = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.uniform(size=n)) / n
    return pts


# ── Expected Improvement acquisition ──────────────────────────────────────────

def expected_improvement(
    X_cand: np.ndarray,
    y_obs: np.ndarray,
    gp: GaussianProcessRegressor,
    xi: float = 0.01,
) -> np.ndarray:
    """EI = E[max(f(x) - (y_best + xi), 0)] under the GP posterior."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    y_best = np.max(y_obs)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(model_name: str, params: dict):
    """
    Instantiate a temperature model and inject trial-specific hyperparameters.
    The returned instance is ready for .fit().

    spread_alpha is NOT injected here — it is patched at the src.model module
    level in evaluate_params() (single-threaded; always restored in finally).
    """
    from src.model import (
        BayesianTempModel, ARDTempModel, InteractionBayesModel,
        NGBoostTempModel, RandomForestModel, ExtraTreesModel,
        QuantileGBModel, GBResidualModel, BaggingRidgeModel, KernelRidgeModel,
    )

    if model_name in ("BayesianRidge", "InteractionBayes"):
        from sklearn.linear_model import BayesianRidge as _BR
        br = _BR(
            alpha_1=params.get("alpha_1", 1e-6),
            alpha_2=params.get("alpha_2", 1e-6),
            lambda_1=params.get("lambda_1", 1e-6),
            lambda_2=params.get("lambda_2", 1e-6),
        )
        if model_name == "BayesianRidge":
            m = BayesianTempModel()
            m._model = br
        else:
            m = InteractionBayesModel()
            m._model = br

    elif model_name == "ARD":
        from sklearn.linear_model import ARDRegression
        m = ARDTempModel()
        m._model = ARDRegression(
            alpha_1=params.get("alpha_1", 1e-6),
            alpha_2=params.get("alpha_2", 1e-6),
            lambda_1=params.get("lambda_1", 1e-6),
            lambda_2=params.get("lambda_2", 1e-6),
        )

    elif model_name == "NGBoost":
        m = NGBoostTempModel(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=params.get("learning_rate", 0.05),
        )
        m.EPISTEMIC_FRAC = params.get("epistemic_frac", 0.25)

    elif model_name == "RandomForest":
        m = RandomForestModel(
            n_estimators=int(params.get("n_estimators", 40)),
            max_depth=int(params.get("max_depth", 8)),
        )

    elif model_name == "ExtraTrees":
        m = ExtraTreesModel(
            n_estimators=int(params.get("n_estimators", 40)),
            max_depth=int(params.get("max_depth", 8)),
        )

    elif model_name == "QuantileGB":
        m = QuantileGBModel(max_iter=int(params.get("max_iter", 80)))

    elif model_name == "GBResidual":
        m = GBResidualModel(
            max_iter=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
        )

    elif model_name == "BaggingRidge":
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import BaggingRegressor
        n_est = int(params.get("n_estimators", 30))
        m = BaggingRidgeModel(n_estimators=n_est)
        m._model = BaggingRegressor(
            estimator=Ridge(alpha=params.get("ridge_alpha", 1.0)),
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "KernelRidge":
        m = KernelRidgeModel(
            alpha=params.get("alpha", 0.5),
            gamma=params.get("gamma", None),
        )
        m.N_BOOT = int(params.get("n_boot", 15))

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    m._sigma_floor = float(params.get("sigma_floor", 4.0))
    return m


# ── Trial evaluator ───────────────────────────────────────────────────────────

def evaluate_params(
    model_name: str,
    params: dict,
    dataset,
    price_store: KalshiPriceStore,
    tune_cfg: TuneConfig,
    return_run: bool = False,
) -> "tuple[float, dict] | tuple[float, dict, RunResult]":
    """
    Build a BacktestConfig from trial params, run one backtest via
    BacktestEngine, and return (total_pnl, per_year_pnl_dict).

    spread_alpha is patched at the src.model module level so it affects
    the model's aleatoric sigma computation. Always restored in finally.
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
            max_bet_dollars=tune_cfg.max_bet_dollars,
            min_edge=float(params.get("min_edge", 0.05)),
            min_confidence=float(params.get("min_confidence", 0.0)),
            min_fair_p=float(params.get("min_fair_p", 0.05)),
            max_fair_p=float(params.get("max_fair_p", 0.95)),
            fee_rate=tune_cfg.fee_rate,
            n_runs=1,
            seed=42,
            start_date=tune_cfg.start_date,
            end_date=tune_cfg.end_date,
            sessions_per_day=tune_cfg.sessions_per_day,
            max_daily_trades=tune_cfg.max_daily_trades,
            max_session_trades=tune_cfg.max_session_trades,
            refit_every=tune_cfg.refit_every,
            min_train_rows=tune_cfg.min_train_rows,
            verbose=False,
            cash_reserve_fraction=tune_cfg.cash_reserve_fraction,
            rotation_min_edge_gain=tune_cfg.rotation_min_edge_gain,
            initial_bankroll=tune_cfg.initial_bankroll,
            pessimistic_pricing=tune_cfg.pessimistic_pricing,
        )
        # Capture params in a local so the lambda closure is stable
        _params = dict(params)
        engine = BacktestEngine(
            dataset,
            cfg,
            price_store,
            model_factory=lambda: make_model(model_name, _params),
        )
        results = engine.run()
    except Exception:
        empty = ({},)
        return (-np.inf, *empty, None) if return_run else (-np.inf, {})
    finally:
        _mmod.SPREAD_ALPHA = orig_spread

    run = results.runs[0]
    year_pnls = _year_pnls(run.trades)

    if tune_cfg.recency_halflife_days is not None and not run.trades.empty:
        trades = run.trades.copy()
        ref_date = pd.to_datetime(trades["date"]).max()
        days_ago = (ref_date - pd.to_datetime(trades["date"])).dt.days.values
        decay = np.log(2) / tune_cfg.recency_halflife_days
        weights = np.exp(-decay * days_ago)
        score = float((trades["pnl"].values * weights).sum())
    else:
        score = run.total_pnl

    if return_run:
        return score, year_pnls, run
    return score, year_pnls


def _year_pnls(trades: pd.DataFrame) -> dict:
    """Return absolute PnL per calendar year."""
    if trades.empty:
        return {}
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["date"]).dt.year
    return {int(yr): float(grp["pnl"].sum()) for yr, grp in trades.groupby("year")}


# ── Bayesian Optimization loop ────────────────────────────────────────────────

def run_bo(
    model_name: str,
    dataset,
    price_store: KalshiPriceStore,
    tune_cfg: TuneConfig,
    n_iter: int = 50,
    n_init: int = 10,
    xi: float = 0.01,
    seed: int = 0,
) -> dict:
    """
    Run Bayesian Optimization for model_name.
    Returns dict: {best_params, best_pnl, history}.
    """
    space = SEARCH_SPACES[model_name]
    d = len(space)
    rng = np.random.default_rng(seed)

    kernel = Matern(nu=2.5, length_scale=np.ones(d), length_scale_bounds=(1e-3, 10.0))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    X_obs = latin_hypercube(n_init, d, rng)
    y_obs = np.full(n_init, np.nan)
    history = []

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"Tuning: {model_name}  |  {n_iter} evals  ({n_init} random init)")
    print(
        f"Period: {tune_cfg.start_date} → {tune_cfg.end_date}  |  "
        f"refit_every={tune_cfg.refit_every}d  |  "
        f"sessions/day={'all' if tune_cfg.sessions_per_day == 0 else tune_cfg.sessions_per_day}"
    )
    print(sep)
    print(f"{'iter':>4}  {'pnl':>10}  {'best':>10}  params")
    print("-" * 64)

    for it in range(n_iter):
        if it < n_init:
            x_next = X_obs[it]
        else:
            mask = np.isfinite(y_obs)
            if mask.sum() >= 2:
                gp.fit(X_obs[mask], y_obs[mask])
                X_cand = rng.uniform(size=(8000, d))
                ei = expected_improvement(X_cand, y_obs[mask], gp, xi=xi)
                x_next = X_cand[np.argmax(ei)]
            else:
                x_next = rng.uniform(size=d)

        params = decode(x_next, space)
        t0 = time.time()
        pnl, year_pnls = evaluate_params(model_name, params, dataset, price_store, tune_cfg)
        elapsed = time.time() - t0

        if it >= n_init:
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, pnl)
        else:
            y_obs[it] = pnl

        history.append({"params": params, "pnl": pnl})
        best_pnl = float(np.nanmax(y_obs))
        params_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
        print(f"{it+1:>4}  {pnl:>10.2f}  {best_pnl:>10.2f}  {params_str}  ({elapsed:.0f}s)")

        if year_pnls:
            yr_str = "  ".join(f"{yr}:${p:.2f}" for yr, p in sorted(year_pnls.items()))
            print(f"      per-year → {yr_str}")

    best_idx = int(np.nanargmax(y_obs))
    best_params = history[best_idx]["params"]
    best_pnl = history[best_idx]["pnl"]

    print(f"\nBest PnL: ${best_pnl:.2f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k:<20} {v:.6g}")

    valid = np.array([r for r in y_obs if np.isfinite(r)])
    print(f"\n=== PnL distribution across {len(valid)} evals ===")
    for label, p in [("min", 0), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("max", 100)]:
        print(f"  {label}:  ${np.percentile(valid, p):.2f}")
    print(f"  mean: ${valid.mean():.2f}  |  std: ${valid.std():.2f}  |  % > 0: {(valid > 0).mean():.1%}")

    return {"best_params": best_params, "best_pnl": best_pnl, "history": history}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning against real Kalshi market prices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="BayesianRidge",
        choices=sorted(SEARCH_SPACES.keys()) + ["all"],
        metavar="MODEL",
        help=(
            "Model to tune. Choices: "
            + ", ".join(sorted(SEARCH_SPACES.keys()))
            + ", all  [default: BayesianRidge]"
        ),
    )
    parser.add_argument("--n-iter",  type=int,   default=50,
                        help="Total BO evaluations (including random init)")
    parser.add_argument("--n-init",  type=int,   default=10,
                        help="Random Latin-hypercube evals before BO starts")
    parser.add_argument("--xi",      type=float, default=0.01,
                        help="EI exploration bonus")
    parser.add_argument("--seed",    type=int,   default=0)
    parser.add_argument("--start",   default=_tc.start_date,
                        help="Backtest start date for tuning trials")
    parser.add_argument("--end",     default=_tc.end_date,
                        help="Backtest end date for tuning trials")
    parser.add_argument("--refit-every", type=int, default=_tc.refit_every,
                        help="Model refit cadence in days (larger = faster trials)")
    parser.add_argument("--sessions-per-day", type=int, default=_tc.sessions_per_day,
                        help="Price candles per day (1=fast, 0=all)")
    parser.add_argument("--recency-halflife", type=int, default=None, metavar="DAYS",
                        help="Exponential recency weighting: trades N days ago count half as much. "
                             "None = flat. Try 90 to weight current season 4x over 6 months ago.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip data quality validation")
    parser.add_argument("--list",    action="store_true",
                        help="List models and their search spaces, then exit")
    args = parser.parse_args()

    if args.list:
        print("Available models and their tunable parameters:\n")
        for name, space in SEARCH_SPACES.items():
            print(f"  {name}")
            for pname, scale, lo, hi in space:
                print(f"    {pname:<20} {scale:<8} [{lo}, {hi}]")
        return

    models_to_run = list(SEARCH_SPACES.keys()) if args.model == "all" else [args.model]
    for m in models_to_run:
        if m not in SEARCH_SPACES:
            print(f"Unknown model: {m!r}. Use --list.")
            sys.exit(1)

    tune_cfg = TuneConfig(
        start_date=args.start,
        end_date=args.end,
        refit_every=args.refit_every,
        sessions_per_day=args.sessions_per_day,
        recency_halflife_days=args.recency_halflife,
    )

    # ── Load data once — shared across all trials and all models ──────────────
    print("Loading Kalshi price history...")
    price_store = KalshiPriceStore().load()
    print(price_store.coverage_summary())

    print("\nLoading feature data...")
    bootstrap_cfg = BacktestConfig(model_name="BayesianRidge",
                                   start_date=tune_cfg.start_date,
                                   end_date=tune_cfg.end_date)
    dataset = DataLoader().load(bootstrap_cfg, skip_validation=args.no_validate)

    Path("logs").mkdir(exist_ok=True)
    all_results = {}

    for model_name in models_to_run:
        result = run_bo(
            model_name=model_name,
            dataset=dataset,
            price_store=price_store,
            tune_cfg=tune_cfg,
            n_iter=args.n_iter,
            n_init=min(args.n_init, args.n_iter),
            xi=args.xi,
            seed=args.seed,
        )
        all_results[model_name] = {
            "best_params": result["best_params"],
            "best_pnl": result["best_pnl"],
        }

        # Save trade log for the winning params
        print(f"\nReplaying best params for {model_name}...")
        pnl, _, best_run = evaluate_params(
            model_name, result["best_params"], dataset, price_store, tune_cfg,
            return_run=True,
        )
        if best_run is not None and not best_run.trades.empty:
            trades_path = Path(f"logs/tuning_trades_{model_name}.csv")
            best_run.trades.to_csv(trades_path, index=False)
            win_rate = (best_run.trades["pnl"] > 0).mean()
            print(
                f"  Trades → {trades_path}  "
                f"({len(best_run.trades)} trades, {win_rate:.1%} win, PnL=${pnl:.2f})"
            )

    # Merge into the shared JSON (preserves results from previous tuning runs)
    out_path = Path("logs/tuned_hyperparams.json")
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nBest params → {out_path}")

    if len(models_to_run) > 1:
        print("\n=== Summary (ranked by PnL) ===")
        for name, r in sorted(all_results.items(), key=lambda x: -x[1]["best_pnl"]):
            print(f"  {name:<20}  PnL=${r['best_pnl']:.2f}")


if __name__ == "__main__":
    main()
