"""
Bayesian Hyperparameter Optimization for weather temperature models.

Gaussian Process surrogate + Expected Improvement (EI) acquisition.
Objective: walk-forward ROI (same setup as compare_models.py).

Usage:
  python scripts/tune_hyperparams.py                        # BayesianRidge, 25 evals
  python scripts/tune_hyperparams.py --model NGBoost --n-iter 30
  python scripts/tune_hyperparams.py --model all            # tune all models sequentially
  python scripts/tune_hyperparams.py --list                 # list available models

Available models: BayesianRidge, ARD, NGBoost, RandomForest, ExtraTrees,
                  QuantileGB, GBResidual, BaggingRidge, KernelRidge
"""

import sys
import argparse
import time
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.features import build_feature_table
from src.config import TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Backtest parameters (mirrors compare_models.py for comparability)
# ------------------------------------------------------------------
EDGE_THRESHOLD    = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER
BANKROLL_START    = 500.0
THRESHOLD_OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
MIN_TRAIN_ROWS    = 365
MKT_SPREAD_ALPHA  = 0.3     # market sigma scaling; fixed, not tuned here
N_THOMPSON_DRAWS  = 30
EVAL_STEP         = 20      # evaluate every Nth row in walk-forward

# ------------------------------------------------------------------
# Hyperparameter search spaces
# Each param tuple: (name, scale, low, high)
#   "log"    → GP operates on log10(value); low/high are raw values
#   "linear" → GP operates on value directly
#   "int"    → integer linear; rounded when decoding
# ------------------------------------------------------------------
# Shared params appended to every model's search space.
# Training: how much history and how to weight it.
#   lookback       – rows of training history to keep (2500 ≈ no limit)
#   decay_halflife – exp recency weighting: weight ∝ exp(-ln2/h * age_days)
#                    720 ≈ uniform; ignored for BayesianRidge/ARD/KernelRidge
#                    (those estimators don't support sample_weight)
# Trading: bet sizing and contract selectivity.
#   min_p_sel – fraction of Thompson draws that must agree before betting
#   min_edge  – extra edge buffer on top of base EDGE_THRESHOLD
_SHARED_PARAMS = [
    ("lookback",       "int",     180, 2500),
    ("decay_halflife", "int",      20,  720),
    ("kelly_fraction", "linear",  0.05,  1.0),
    ("max_bet_frac",   "linear",  0.01,  0.20),
    ("min_p_sel",      "linear",  0.05,  0.80),
    ("min_edge",       "linear",  0.0,   0.15),
]

SEARCH_SPACES = {
    "BayesianRidge": [
        ("alpha_1",      "log",    1e-8,  1e-2),
        ("alpha_2",      "log",    1e-8,  1e-2),
        ("lambda_1",     "log",    1e-8,  1e-2),
        ("lambda_2",     "log",    1e-8,  1e-2),
        ("spread_alpha", "linear", 0.0,   1.0),
    ] + _SHARED_PARAMS,
    "ARD": [
        ("alpha_1",      "log",    1e-8,  1e-2),
        ("alpha_2",      "log",    1e-8,  1e-2),
        ("lambda_1",     "log",    1e-8,  1e-2),
        ("lambda_2",     "log",    1e-8,  1e-2),
        ("spread_alpha", "linear", 0.0,   1.0),
    ] + _SHARED_PARAMS,
    "NGBoost": [
        ("n_estimators",   "int",    50,   400),
        ("learning_rate",  "log",  0.01,   0.3),
        ("epistemic_frac", "linear",0.05,  0.5),
        ("spread_alpha",   "linear",0.0,   1.0),
    ] + _SHARED_PARAMS,
    "RandomForest": [
        ("n_estimators", "int",    20,  200),
        ("max_depth",    "int",     3,   15),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
    "ExtraTrees": [
        ("n_estimators", "int",    20,  200),
        ("max_depth",    "int",     3,   15),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
    "QuantileGB": [
        ("max_iter",     "int",    30,  300),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
    "GBResidual": [
        ("n_estimators", "int",    30,  300),
        ("max_depth",    "int",     2,    8),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
    "BaggingRidge": [
        ("n_estimators", "int",    10,  100),
        ("ridge_alpha",  "log",  0.01, 100.0),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
    "KernelRidge": [
        ("alpha",        "log",  0.01,  10.0),
        ("gamma",        "log", 0.001,   1.0),
        ("n_boot",       "int",     5,    30),
        ("spread_alpha", "linear", 0.0,  1.0),
    ] + _SHARED_PARAMS,
}


# ------------------------------------------------------------------
# Param space encode / decode  (maps params ↔ [0,1]^d for GP)
# ------------------------------------------------------------------

def encode(params: dict, space: list) -> np.ndarray:
    """Map a params dict to a unit-hypercube point."""
    x = []
    for name, scale, lo, hi in space:
        v = params[name]
        if scale == "log":
            x.append((np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)))
        else:
            x.append((v - lo) / (hi - lo))
    return np.array(x)


def decode(x: np.ndarray, space: list) -> dict:
    """Map a unit-hypercube point back to a params dict."""
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
    """Latin-hypercube samples in [0,1]^d."""
    pts = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.uniform(size=n)) / n
    return pts


# ------------------------------------------------------------------
# Expected Improvement acquisition
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Shared backtest helpers (same logic as compare_models.py)
# ------------------------------------------------------------------

def climo_prob_geq(threshold, climo_mean, climo_sigma, spread=0.0):
    market_sigma = climo_sigma * (1.0 + MKT_SPREAD_ALPHA * spread)
    return float(sp_norm.sf(threshold - 0.5, loc=climo_mean, scale=market_sigma))


def kelly_size(edge, bankroll, kelly_fraction, max_bet_frac):
    if edge <= 0:
        return 0.0
    return min(bankroll * max_bet_frac, bankroll * edge * kelly_fraction)


def thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices):
    mu_draws    = np.random.normal(mu_hat, max(epistemic_std, 0.01), N_THOMPSON_DRAWS)
    total_sigma = float(np.sqrt(epistemic_std**2 + aleatoric**2))
    counts: dict = {}

    for mu_s in mu_draws:
        best_edge, best_key = EDGE_THRESHOLD, None
        for threshold, mkt_p in zip(thresholds, mkt_prices):
            model_p  = float(sp_norm.sf(threshold - 0.5, loc=mu_s, scale=aleatoric))
            yes_edge = model_p - mkt_p
            no_edge  = mkt_p - model_p
            if yes_edge > best_edge and model_p <= 0.90:
                best_edge, best_key = yes_edge, (threshold, "yes")
            if no_edge > best_edge and (1 - model_p) <= 0.90:
                best_edge, best_key = no_edge, (threshold, "no")
        if best_key:
            counts[best_key] = counts.get(best_key, 0) + 1

    if not counts:
        return []

    trades = []
    for (threshold, side), count in counts.items():
        p_sel       = count / N_THOMPSON_DRAWS
        mkt_p       = mkt_prices[thresholds.index(threshold)]
        exp_model_p = float(sp_norm.sf(threshold - 0.5, loc=mu_hat, scale=total_sigma))
        exp_edge    = (exp_model_p - mkt_p) if side == "yes" else (mkt_p - exp_model_p)
        if p_sel > 0.80 and exp_edge < 0.10:
            continue
        if exp_edge > 0:
            trades.append({"threshold": threshold, "side": side,
                           "mkt_p": mkt_p, "exp_edge": exp_edge,
                           "p_sel": p_sel, "ei": p_sel * exp_edge})
    trades.sort(key=lambda x: -x["ei"])
    return trades


# ------------------------------------------------------------------
# Model factory: build an instance with the trial's hyperparams.
# spread_alpha is patched via monkeypatching src.model.SPREAD_ALPHA
# (single-threaded, so safe to restore in a finally block).
# ------------------------------------------------------------------

def make_model(model_name: str, params: dict):
    from src.model import (
        BayesianTempModel, ARDTempModel, NGBoostTempModel,
        RandomForestModel, ExtraTreesModel, QuantileGBModel,
        GBResidualModel, BaggingRidgeModel, KernelRidgeModel,
    )

    if model_name == "BayesianRidge":
        from sklearn.linear_model import BayesianRidge
        m = BayesianTempModel()
        m._model = BayesianRidge(
            alpha_1=params.get("alpha_1", 1e-6),
            alpha_2=params.get("alpha_2", 1e-6),
            lambda_1=params.get("lambda_1", 1e-6),
            lambda_2=params.get("lambda_2", 1e-6),
        )

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
            n_estimators=int(params.get("n_estimators", 100)),
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
        raise ValueError(f"Unknown model: {model_name!r}. Use --list to see options.")

    # Inject training params — read by _fit_core via getattr
    m._lookback       = int(params.get("lookback",       0))
    m._decay_halflife = float(params.get("decay_halflife", 0))

    return m


# ------------------------------------------------------------------
# Walk-forward evaluator: returns ROI for a given params dict
# ------------------------------------------------------------------

def evaluate_params(
    model_name: str,
    params: dict,
    df: pd.DataFrame,
    climo_sigma_map: dict,
    return_trades: bool = False,
) -> "float | tuple[float, pd.DataFrame]":
    """Run walk-forward simulation; return ROI (or (ROI, trades_df) if return_trades=True).
    Returns -1.0 on total failure."""
    import src.model as _mmod
    from src.model import FEATURES

    # Patch SPREAD_ALPHA for this trial (affects all models' _aleatoric)
    orig_spread = _mmod.SPREAD_ALPHA
    _mmod.SPREAD_ALPHA = float(params.get("spread_alpha", orig_spread))

    all_trades = []
    try:
        for city in df["city"].unique():
            city_df     = df[df["city"] == city].copy().reset_index(drop=True)
            climo_sigma = climo_sigma_map.get(city, 10.0)

            for i in range(MIN_TRAIN_ROWS, len(city_df), EVAL_STEP):
                test_row = city_df.iloc[i]
                if pd.isna(test_row["y_tmax"]):
                    continue
                if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                    continue

                train_df = city_df.iloc[:i]
                model    = make_model(model_name, params)
                try:
                    model.fit(train_df)
                    mu_arr, ep_arr, al_arr = model.predict_with_uncertainty(city_df.iloc[[i]])
                except Exception:
                    continue

                mu_hat        = float(mu_arr[0])
                epistemic_std = float(ep_arr[0])
                aleatoric     = float(al_arr[0])
                spread_val    = float(test_row.get("ensemble_spread", 0) or 0)
                y_tmax        = float(test_row["y_tmax"])
                climo_mean    = float(test_row["climo_mean_doy"])

                thresholds = [int(round(climo_mean + off)) for off in THRESHOLD_OFFSETS]
                mkt_prices = [
                    climo_prob_geq(t, climo_mean, climo_sigma, spread_val)
                    for t in thresholds
                ]

                candidates = thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices)
                kf       = params.get("kelly_fraction", 0.25)
                mbf      = params.get("max_bet_frac",   0.05)
                min_psel = params.get("min_p_sel",      0.0)
                min_edge = EDGE_THRESHOLD + params.get("min_edge", 0.0)
                for trade_info in candidates[:1]:
                    if trade_info["p_sel"] < min_psel:
                        continue
                    if trade_info["exp_edge"] < min_edge:
                        continue
                    side    = trade_info["side"]
                    mkt_p   = trade_info["mkt_p"]
                    edge    = trade_info["exp_edge"]
                    outcome = 1 if y_tmax >= trade_info["threshold"] else 0
                    size    = kelly_size(edge, BANKROLL_START, kf, mbf)
                    pnl     = (
                        size * (outcome - mkt_p)
                        if side == "yes"
                        else size * ((1 - outcome) - (1 - mkt_p))
                    )
                    trade = {
                        "year": pd.to_datetime(test_row["date"]).year,
                        "size": size,
                        "pnl":  pnl,
                    }
                    if return_trades:
                        trade.update({
                            "date":      test_row["date"],
                            "city":      city,
                            "threshold": trade_info["threshold"],
                            "side":      side,
                            "mkt_p":     mkt_p,
                            "edge":      edge,
                            "p_sel":     trade_info["p_sel"],
                            "ei":        trade_info["ei"],
                            "mu_hat":    mu_hat,
                            "outcome":   outcome if side == "yes" else 1 - outcome,
                        })
                    all_trades.append(trade)

    finally:
        _mmod.SPREAD_ALPHA = orig_spread

    if not all_trades:
        return (-1.0, {}, pd.DataFrame()) if return_trades else (-1.0, {})
    tdf           = pd.DataFrame(all_trades)
    total_wagered = tdf["size"].sum()
    roi           = float(tdf["pnl"].sum() / total_wagered) if total_wagered > 0 else -1.0

    # Per-year ROI breakdown
    year_rois = {}
    for yr, grp in tdf.groupby("year"):
        w = grp["size"].sum()
        year_rois[int(yr)] = float(grp["pnl"].sum() / w) if w > 0 else 0.0

    if return_trades:
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.sort_values("date").reset_index(drop=True)
        return roi, year_rois, tdf
    return roi, year_rois


# ------------------------------------------------------------------
# Bayesian Optimization loop
# ------------------------------------------------------------------

def run_bo(
    model_name: str,
    df: pd.DataFrame,
    climo_sigma_map: dict,
    n_iter: int = 25,
    n_init: int = 5,
    xi: float = 0.01,
    seed: int = 0,
) -> dict:
    """
    Run BO for `model_name`.  Returns a dict with keys:
        best_params, best_roi, history (list of {params, roi} dicts)
    """
    space = SEARCH_SPACES[model_name]
    d     = len(space)
    rng   = np.random.default_rng(seed)

    # GP surrogate: Matérn 5/2 kernel + noise
    kernel = Matern(nu=2.5, length_scale=np.ones(d), length_scale_bounds=(1e-3, 10.0))
    gp     = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,          # observation noise
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    X_obs = latin_hypercube(n_init, d, rng)
    y_obs = np.full(n_init, np.nan)
    history = []

    print(f"\n{'='*60}")
    print(f"Tuning: {model_name}  |  {n_iter} total evals  ({n_init} random init)")
    print(f"{'='*60}")
    param_names = [s[0] for s in space]
    print(f"{'iter':>4}  {'roi':>8}  {'best':>8}  params")
    print("-" * 60)

    for it in range(n_iter):
        if it < n_init:
            x_next = X_obs[it]
        else:
            # Fit GP on valid observations
            mask   = np.isfinite(y_obs)
            if mask.sum() >= 2:
                gp.fit(X_obs[mask], y_obs[mask])
                # Sample 8000 candidates; pick argmax EI
                X_cand = rng.uniform(size=(8000, d))
                ei     = expected_improvement(X_cand, y_obs[mask], gp, xi=xi)
                x_next = X_cand[np.argmax(ei)]
            else:
                x_next = rng.uniform(size=d)

        params = decode(x_next, space)
        t0     = time.time()
        roi, year_rois = evaluate_params(model_name, params, df, climo_sigma_map)
        elapsed = time.time() - t0

        if it >= n_init:
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, roi)
        else:
            y_obs[it] = roi

        history.append({"params": params, "roi": roi})
        best_roi   = np.nanmax(y_obs)
        params_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
        yr_vals   = list(year_rois.values())
        yr_by_yr  = "  ".join(f"{yr}:{r:.3f}" for yr, r in sorted(year_rois.items()))
        print(f"{it+1:>4}  {roi:>8.4f}  {best_roi:>8.4f}  {params_str}  ({elapsed:.0f}s)")
        print(f"      per-year ROI → {yr_by_yr}")
        print(f"      year stats   → mean:{np.mean(yr_vals):.4f}  "
              f"std:{np.std(yr_vals):.4f}  "
              f"min:{np.min(yr_vals):.4f}  "
              f"max:{np.max(yr_vals):.4f}")

    best_idx    = int(np.nanargmax(y_obs))
    best_params = history[best_idx]["params"]
    best_roi    = history[best_idx]["roi"]

    print(f"\nBest ROI: {best_roi:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6g}")

    valid_rois = np.array([r for r in y_obs if np.isfinite(r)])
    print(f"\n=== ROI distribution across {len(valid_rois)} evals ===")
    print(f"  mean:  {valid_rois.mean():.4f}")
    print(f"  std:   {valid_rois.std():.4f}")
    print(f"  min:   {valid_rois.min():.4f}")
    for p in (25, 50, 75, 90):
        print(f"  p{p:<3}:  {np.percentile(valid_rois, p):.4f}")
    print(f"  max:   {valid_rois.max():.4f}")
    print(f"  % > 0: {(valid_rois > 0).mean():.1%}")

    return {"best_params": best_params, "best_roi": best_roi, "history": history}


# ------------------------------------------------------------------
# Data loading (shared across all model tuning runs)
# ------------------------------------------------------------------

def load_data():
    hist      = pd.read_csv("data/historical_tmax.csv")
    gfs_path  = Path("data/forecasts/openmeteo_forecast_history.csv")
    idx_path  = Path("data/climate_indices.csv")
    gefs_path = Path("data/forecasts/gefs_spread.csv")

    gfs_df  = pd.read_csv(gfs_path)  if gfs_path.exists()  else None
    idx_df  = pd.read_csv(idx_path)  if idx_path.exists()  else None
    gefs_df = pd.read_csv(gefs_path) if gefs_path.exists() else None

    forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=idx_df, gefs_df=gefs_df)
    df["forecast_high"]        = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    climo_sigma_map = {}
    for city in df["city"].unique():
        cdf = df[df["city"] == city].dropna(subset=["y_tmax", "climo_mean_doy"])
        climo_sigma_map[city] = float((cdf["y_tmax"] - cdf["climo_mean_doy"]).std())

    return df, climo_sigma_map


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter tuning for temperature models.")
    parser.add_argument("--model",  default="BayesianRidge",
                        help="Model to tune, or 'all' (default: BayesianRidge)")
    parser.add_argument("--n-iter", type=int, default=50,
                        help="Total evaluations including random init (default: 50)")
    parser.add_argument("--n-init", type=int, default=10,
                        help="Random initialization evals before BO starts (default: 10)")
    parser.add_argument("--xi",     type=float, default=0.01,
                        help="EI exploration bonus xi (default: 0.01)")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--list",   action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, space in SEARCH_SPACES.items():
            param_str = ", ".join(s[0] for s in space)
            print(f"  {name:<20} [{param_str}]")
        return

    models_to_run = list(SEARCH_SPACES.keys()) if args.model == "all" else [args.model]
    for m in models_to_run:
        if m not in SEARCH_SPACES:
            print(f"Unknown model: {m!r}. Use --list to see options.")
            sys.exit(1)

    print("Loading data...")
    df, climo_sigma_map = load_data()
    print(f"Loaded {len(df)} rows across {df['city'].nunique()} cities.")

    Path("logs").mkdir(exist_ok=True)
    all_results = {}

    for model_name in models_to_run:
        result = run_bo(
            model_name, df, climo_sigma_map,
            n_iter=args.n_iter,
            n_init=min(args.n_init, args.n_iter),
            xi=args.xi,
            seed=args.seed,
        )
        all_results[model_name] = {
            "best_params": result["best_params"],
            "best_roi":    result["best_roi"],
        }

        # Save trade history for the best params so it can be inspected later
        print(f"\nReplaying best params to save trade history...")
        _, _, trades_df = evaluate_params(
            model_name, result["best_params"], df, climo_sigma_map, return_trades=True
        )
        if len(trades_df):
            trades_path = Path(f"logs/tuning_trades_{model_name}.csv")
            trades_df.to_csv(trades_path, index=False)
            win_rate = (trades_df["pnl"] > 0).mean()
            print(f"Trade history saved to {trades_path}  "
                  f"({len(trades_df)} trades, {win_rate:.1%} win rate)")

    out_path = Path("logs/tuned_hyperparams.json")
    # Merge with any existing results
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if len(models_to_run) > 1:
        print("\n=== Summary (ranked by ROI) ===")
        rows = sorted(all_results.items(), key=lambda x: -x[1]["best_roi"])
        for name, r in rows:
            print(f"  {name:<20}  ROI={r['best_roi']:.4f}")


if __name__ == "__main__":
    main()
