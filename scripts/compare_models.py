"""
Walk-forward PnL comparison across all 6 temperature uncertainty models.

Each model is evaluated with identical Thompson Sampling contract selection,
spread-aware market pricing, and Kelly sizing.  Results are ranked by ROI.

Run from project root:
  python scripts/compare_models.py

Models compared:
  BayesianTempModel    – Bayesian Ridge, analytic posterior
  RandomForestModel    – Random Forest, tree-variance epistemic
  ExtraTreesModel      – Extra-Trees, higher-variance tree diversity
  QuantileGBModel      – HistGradientBoosting quantile interval
  GBResidualModel      – GradientBoosting + separate residual predictor
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.features import build_feature_table
from src.config import KELLY_FRACTION, MAX_BET_FRACTION, TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER
from src.model import (
    FEATURES,
    BayesianTempModel,
    ARDTempModel,
    InteractionBayesModel,
    KernelRidgeModel,
    NGBoostTempModel,
    RandomForestModel,
    ExtraTreesModel,
    QuantileGBModel,
    GBResidualModel,
)

EDGE_THRESHOLD    = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER
BANKROLL_START    = 500.0
THRESHOLD_OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
MIN_TRAIN_ROWS    = 365
SPREAD_ALPHA      = 0.3
N_THOMPSON_DRAWS  = 30
EVAL_STEP         = 30  # evaluate every Nth row to speed up walk-forward

ALL_MODELS = [
    ("BayesianRidge",      BayesianTempModel),
    ("ARD",                ARDTempModel),
    ("Interaction+Bayes",  InteractionBayesModel),
    ("KernelRidge",        KernelRidgeModel),
    ("NGBoost",            NGBoostTempModel),
    ("RandomForest",       RandomForestModel),
    ("ExtraTrees",         ExtraTreesModel),
    ("QuantileGB",         QuantileGBModel),
    ("GBResidual",         GBResidualModel),
]


# ------------------------------------------------------------------
# Shared helpers (same as run_pnl_backtest.py)
# ------------------------------------------------------------------

def climo_prob_geq(threshold, climo_mean, climo_sigma, spread=0.0):
    market_sigma = climo_sigma * (1.0 + SPREAD_ALPHA * spread)
    return float(norm.sf(threshold - 0.5, loc=climo_mean, scale=market_sigma))


def kelly_size(edge, bankroll):
    if edge <= 0:
        return 0.0
    return min(bankroll * MAX_BET_FRACTION, bankroll * edge * KELLY_FRACTION)


def thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices):
    mu_draws = np.random.normal(mu_hat, max(epistemic_std, 0.01), N_THOMPSON_DRAWS)
    total_sigma = float(np.sqrt(epistemic_std**2 + aleatoric**2))
    counts = {}

    for mu_s in mu_draws:
        best_edge = EDGE_THRESHOLD
        best_key  = None
        for threshold, mkt_p in zip(thresholds, mkt_prices):
            model_p  = float(norm.sf(threshold - 0.5, loc=mu_s, scale=aleatoric))
            yes_edge = model_p - mkt_p
            no_edge  = mkt_p - model_p
            if yes_edge > best_edge and model_p <= 0.90:
                best_edge = yes_edge
                best_key  = (threshold, "yes")
            if no_edge > best_edge and (1 - model_p) <= 0.90:
                best_edge = no_edge
                best_key  = (threshold, "no")
        if best_key:
            counts[best_key] = counts.get(best_key, 0) + 1

    if not counts:
        return []

    trades = []
    for (threshold, side), count in counts.items():
        p_sel = count / N_THOMPSON_DRAWS
        mkt_p = mkt_prices[thresholds.index(threshold)]
        exp_model_p = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=total_sigma))
        exp_edge    = (exp_model_p - mkt_p) if side == "yes" else (mkt_p - exp_model_p)
        if exp_edge > 0:
            trades.append({"threshold": threshold, "side": side,
                           "mkt_p": mkt_p, "exp_edge": exp_edge,
                           "ei": p_sel * exp_edge})
    trades.sort(key=lambda x: -x["ei"])
    return trades


# ------------------------------------------------------------------
# Walk-forward loop for one model class
# ------------------------------------------------------------------

def run_model(name, model_class, df, climo_sigma_map):
    all_trades = []

    for city in df["city"].unique():
        city_df = df[df["city"] == city].copy().reset_index(drop=True)
        climo_sigma = climo_sigma_map.get(city, 10.0)

        for i in range(MIN_TRAIN_ROWS, len(city_df), EVAL_STEP):
            test_row = city_df.iloc[i]
            if pd.isna(test_row["y_tmax"]):
                continue
            if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                continue

            train_df = city_df.iloc[:i]
            model = model_class()
            try:
                model.fit(train_df)
            except Exception:
                continue

            try:
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
            mkt_prices = [climo_prob_geq(t, climo_mean, climo_sigma, spread_val) for t in thresholds]

            candidates = thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices)

            for trade_info in candidates[:1]:
                threshold = trade_info["threshold"]
                side      = trade_info["side"]
                mkt_p     = trade_info["mkt_p"]
                edge      = trade_info["exp_edge"]
                raw_p     = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=aleatoric))
                outcome   = 1 if y_tmax >= threshold else 0
                size      = kelly_size(edge, BANKROLL_START)
                pnl       = size * (outcome - mkt_p) if side == "yes" else size * ((1 - outcome) - (1 - mkt_p))

                all_trades.append({
                    "date":      test_row["date"],
                    "city":      city,
                    "threshold": threshold,
                    "side":      side,
                    "edge":      edge,
                    "size":      size,
                    "outcome":   outcome if side == "yes" else 1 - outcome,
                    "pnl":       pnl,
                })

    if not all_trades:
        return None
    return pd.DataFrame(all_trades)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    hist      = pd.read_csv("data/historical_tmax.csv")
    gfs_path  = Path("data/forecasts/openmeteo_forecast_history.csv")
    gfs_df    = pd.read_csv(gfs_path) if gfs_path.exists() else None
    idx_path  = Path("data/climate_indices.csv")
    idx_df    = pd.read_csv(idx_path) if idx_path.exists() else None
    gefs_path = Path("data/forecasts/gefs_spread.csv")
    gefs_df   = pd.read_csv(gefs_path) if gefs_path.exists() else None

    forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=idx_df, gefs_df=gefs_df)
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    climo_sigma_map = {}
    for city in df["city"].unique():
        cdf = df[df["city"] == city].dropna(subset=["y_tmax", "climo_mean_doy"])
        climo_sigma_map[city] = float((cdf["y_tmax"] - cdf["climo_mean_doy"]).std())

    results = []
    for name, model_class in ALL_MODELS:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        t0 = time.time()
        trades_df = run_model(name, model_class, df, climo_sigma_map)
        elapsed = time.time() - t0

        if trades_df is None or trades_df.empty:
            print(f"  No trades generated.")
            continue

        n_trades     = len(trades_df)
        win_rate     = (trades_df["pnl"] > 0).mean()
        total_pnl    = trades_df["pnl"].sum()
        total_wagered = trades_df["size"].sum()
        roi          = total_pnl / total_wagered if total_wagered > 0 else 0
        avg_edge     = trades_df["edge"].mean()

        print(f"  trades={n_trades:,}  win={win_rate:.1%}  edge={avg_edge:.3f}  "
              f"wagered=${total_wagered:,.0f}  PnL=${total_pnl:,.0f}  ROI={roi:.2%}  "
              f"({elapsed:.0f}s)")

        results.append({
            "model":     name,
            "trades":    n_trades,
            "win_rate":  win_rate,
            "avg_edge":  avg_edge,
            "wagered":   total_wagered,
            "pnl":       total_pnl,
            "roi":       roi,
            "time_s":    elapsed,
        })

    if not results:
        print("No results.")
        return

    print(f"\n{'='*70}")
    print("MODEL COMPARISON (ranked by ROI)")
    print(f"{'='*70}")
    res_df = pd.DataFrame(results).sort_values("roi", ascending=False)
    res_df["win_rate"] = res_df["win_rate"].map("{:.1%}".format)
    res_df["roi"]      = res_df["roi"].map("{:.2%}".format)
    res_df["avg_edge"] = res_df["avg_edge"].map("{:.3f}".format)
    res_df["pnl"]      = res_df["pnl"].map("${:,.0f}".format)
    res_df["wagered"]  = res_df["wagered"].map("${:,.0f}".format)
    res_df["time_s"]   = res_df["time_s"].map("{:.0f}s".format)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
