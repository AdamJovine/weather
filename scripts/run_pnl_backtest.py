"""
PnL backtest: simulates trading geq/leq contracts using walk-forward model predictions
vs a climatology-based "market" price.

Methodology:
  - Market price for a geq(k) contract = P(tmax >= k | climatology mean & sigma)
    * sigma is scaled by ensemble_spread (implied-vol-style) so the market is
      also uncertain on days when NWP models disagree.
  - Model uses BayesianTempModel (Bayesian Ridge, GP-proxy) which provides:
      mu:          posterior predictive mean
      total_sigma: sqrt(epistemic² + aleatoric²)  — full predictive uncertainty
  - Thompson Sampling for contract selection:
      draw N_DRAWS samples of mu from BayesianRidge weight posterior,
      find the best (highest-edge) contract under each draw,
      accumulate Expected Improvement = p_selected * expected_edge,
      bet on the contract with the highest Expected Improvement.
  - Sizing: fractional Kelly, capped at MAX_BET_FRACTION of bankroll
  - PnL per dollar bet: (outcome - market_price)  [binary payoff at market odds]

Thresholds simulated: market_mean +/- {3, 6, 9, 12} degrees
These approximate the bracket of contracts Kalshi offers around the expected temp.

Run from project root:
  python scripts/run_pnl_backtest.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.features import build_feature_table
from src.config import KELLY_FRACTION, MAX_BET_FRACTION, TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

EDGE_THRESHOLD    = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER  # 0.05
BANKROLL_START    = 500.0
THRESHOLD_OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]    # degrees from climo mean
MIN_TRAIN_ROWS    = 365
SPREAD_ALPHA      = 0.3   # how much ensemble_spread widens the market sigma
N_THOMPSON_DRAWS  = 100   # posterior samples per prediction for Thompson Sampling

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def climo_prob_geq(
    threshold: int,
    climo_mean: float,
    climo_sigma: float,
    ensemble_spread: float = 0.0,
) -> float:
    """
    P(tmax >= threshold) under the market's Gaussian model.
    Market sigma is widened by ensemble_spread (implied-vol proxy):
      market_sigma = climo_sigma * (1 + SPREAD_ALPHA * ensemble_spread)
    """
    market_sigma = climo_sigma * (1.0 + SPREAD_ALPHA * ensemble_spread)
    return float(norm.sf(threshold - 0.5, loc=climo_mean, scale=market_sigma))


def thompson_select_contracts(
    mu_hat: float,
    epistemic_std: float,
    aleatoric_sigma: float,
    thresholds: list,
    mkt_prices: list,
) -> list[dict]:
    """
    Thompson Sampling acquisition: draw N_THOMPSON_DRAWS samples of mu from the
    Bayesian posterior, identify the best contract under each draw, then
    return a ranked list by Expected Improvement = p_selected * expected_edge.

    Returns a list of trade dicts (can be empty if no draw finds positive edge).
    The caller should bet on trades[:1] — the single highest-EI contract.
    """
    mu_draws = np.random.normal(mu_hat, epistemic_std, N_THOMPSON_DRAWS)
    total_sigma = np.sqrt(epistemic_std**2 + aleatoric_sigma**2)

    # Tally how often each (threshold, side) wins the best-edge draw
    counts: dict[tuple, int] = {}

    for mu_s in mu_draws:
        best_edge = EDGE_THRESHOLD  # minimum bar to be worth selecting
        best_key = None
        for threshold, mkt_p in zip(thresholds, mkt_prices):
            model_p = float(norm.sf(threshold - 0.5, loc=mu_s, scale=aleatoric_sigma))
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
        p_sel   = count / N_THOMPSON_DRAWS
        mkt_p   = mkt_prices[thresholds.index(threshold)]
        # Expected edge under the full predictive posterior
        if side == "yes":
            exp_model_p = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=total_sigma))
            exp_edge    = exp_model_p - mkt_p
        else:
            exp_model_p = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=total_sigma))
            exp_edge    = mkt_p - exp_model_p
        # Drop high-confidence bets with low edge: model is certain but the
        # market has already priced it in — not worth the fee.
        if p_sel > 0.80 and exp_edge < 0.10:
            continue
        if exp_edge > 0:
            trades.append({
                "threshold": threshold,
                "side":      side,
                "mkt_p":     mkt_p,
                "exp_edge":  exp_edge,
                "p_sel":     p_sel,
                "ei":        p_sel * exp_edge,   # Expected Improvement
            })

    trades.sort(key=lambda x: -x["ei"])
    return trades


def kelly_size(edge: float, bankroll: float) -> float:
    if edge <= 0:
        return 0.0
    return min(bankroll * MAX_BET_FRACTION, bankroll * edge * KELLY_FRACTION)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # 1. Load data
    hist_path = Path("data/historical_tmax.csv")
    if not hist_path.exists():
        print("Run scripts/download_history.py first.")
        return

    hist     = pd.read_csv(hist_path)
    gfs_path = Path("data/forecasts/openmeteo_forecast_history.csv")
    gfs_df   = pd.read_csv(gfs_path) if gfs_path.exists() else None
    if gfs_df is not None:
        print(f"GFS forecast data loaded: {len(gfs_df)} rows.")
    else:
        print("No GFS data found — falling back to climatology as forecast.")

    indices_path = Path("data/climate_indices.csv")
    indices_df   = pd.read_csv(indices_path) if indices_path.exists() else None
    if indices_df is not None:
        print(f"Climate indices loaded: {len(indices_df)} rows.")
    else:
        print("No climate_indices.csv — AO/ONI features will be missing.")

    gefs_path = Path("data/forecasts/gefs_spread.csv")
    gefs_df   = pd.read_csv(gefs_path) if gefs_path.exists() else None
    if gefs_df is not None:
        print(f"GEFS spread data loaded: {len(gefs_df)} rows.")
    else:
        print("No gefs_spread.csv — gefs_spread will fall back to ensemble_spread.")

    forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=indices_df,
                             gefs_df=gefs_df)

    # Fill any remaining gaps (dates with no GFS and no observed lag) with climo
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    # 2. Load calibrator
    cal_path = Path("data/calibrator_geq72.pkl")
    calibrator = None
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        print("Calibrator loaded.")
    else:
        print("No calibrator — using raw model probabilities.")

    # 3. Compute per-city climo sigma as std of tmax residuals from doy mean.
    #    This is what a seasonal market would use — not the full annual spread.
    climo_sigma_map = {}
    for city in df["city"].unique():
        city_df_tmp = df[df["city"] == city].dropna(subset=["y_tmax", "climo_mean_doy"])
        residuals = city_df_tmp["y_tmax"] - city_df_tmp["climo_mean_doy"]
        climo_sigma_map[city] = float(residuals.std())
    print("Climo sigmas (residual from doy mean):", {c: round(s, 2) for c, s in climo_sigma_map.items()})

    # 4. Walk-forward PnL simulation using BayesianTempModel + Thompson Sampling
    from src.model import InteractionBayesModel as BayesianTempModel, FEATURES

    all_trades = []
    cities = df["city"].unique()

    for city in cities:
        city_df = df[df["city"] == city].copy().reset_index(drop=True)
        climo_sigma = climo_sigma_map.get(city, 10.0)
        print(f"\n{city}: {len(city_df)} rows, climo_sigma={climo_sigma:.1f}°F")

        for i in range(MIN_TRAIN_ROWS, len(city_df)):
            test_row = city_df.iloc[i]
            if pd.isna(test_row["y_tmax"]):
                continue
            if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                continue

            train_df = city_df.iloc[:i]
            model = BayesianTempModel()
            try:
                model.fit(train_df)
            except ValueError:
                continue

            mu_arr, epistemic_arr, aleatoric_arr = model.predict_with_uncertainty(city_df.iloc[[i]])
            mu_hat        = float(mu_arr[0])
            epistemic_std = float(epistemic_arr[0])
            aleatoric     = float(aleatoric_arr[0])
            spread_val    = float(test_row.get("ensemble_spread", 0) or 0)

            y_tmax     = float(test_row["y_tmax"])
            climo_mean = float(test_row["climo_mean_doy"])
            date_val   = test_row["date"]

            # Build threshold list and corresponding market prices (spread-aware)
            thresholds = [int(round(climo_mean + off)) for off in THRESHOLD_OFFSETS]
            mkt_prices = [
                climo_prob_geq(t, climo_mean, climo_sigma, spread_val)
                for t in thresholds
            ]

            # Thompson Sampling: pick the single best contract by Expected Improvement
            candidates = thompson_select_contracts(
                mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices
            )

            for trade_info in candidates[:1]:   # bet only the top-EI contract
                threshold = trade_info["threshold"]
                side      = trade_info["side"]
                mkt_p     = trade_info["mkt_p"]
                edge      = trade_info["exp_edge"]

                # Point-estimate model_p for calibration
                raw_p = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=aleatoric))
                cal_p = calibrator.predict(np.array([raw_p]))[0] if calibrator else raw_p

                raw_outcome = 1 if y_tmax >= threshold else 0

                # size/pnl deferred — computed in the compounding pass below
                all_trades.append({
                    "date":        date_val, "city": city,
                    "threshold":   threshold, "side": side,
                    "mkt_p":       mkt_p,
                    "raw_p":       raw_p,    # raw model prob before calibration
                    "cal_p":       cal_p,
                    "edge":        edge,
                    "raw_outcome": raw_outcome,
                    "ei":          trade_info["ei"],
                    "p_sel":       trade_info["p_sel"],
                })

    if not all_trades:
        print("No trades generated — try lowering EDGE_THRESHOLD.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df = trades_df.sort_values("date").reset_index(drop=True)

    # Compounding bankroll pass — bets are sized off the running balance (date-ordered)
    bankroll = BANKROLL_START
    sizes, pnls, bankrolls, outcomes = [], [], [], []
    for _, row in trades_df.iterrows():
        size = kelly_size(row["edge"], bankroll)
        if row["side"] == "yes":
            pnl     = size * (row["raw_outcome"] - row["mkt_p"])
            outcome = int(row["raw_outcome"])
        else:
            pnl     = size * ((1 - row["raw_outcome"]) - (1 - row["mkt_p"]))
            outcome = 1 - int(row["raw_outcome"])
        bankroll = max(bankroll + pnl, 1.0)   # floor at $1 to avoid total ruin
        sizes.append(size)
        pnls.append(pnl)
        bankrolls.append(bankroll)
        outcomes.append(outcome)

    trades_df["size"]     = sizes
    trades_df["pnl"]      = pnls
    trades_df["bankroll"] = bankrolls
    trades_df["outcome"]  = outcomes
    trades_df.drop(columns=["raw_outcome"], inplace=True)

    trades_df.to_csv("logs/pnl_backtest_trades.csv", index=False)
    print(f"\nTotal simulated trades: {len(trades_df)}")

    # 5. Summary stats
    print("\n=== PnL Summary ===")
    total_pnl    = trades_df["pnl"].sum()
    win_rate     = (trades_df["pnl"] > 0).mean()
    avg_edge     = trades_df["edge"].mean()
    total_wagered = trades_df["size"].sum()
    roi          = total_pnl / total_wagered if total_wagered > 0 else 0

    print(f"Total trades:   {len(trades_df)}")
    print(f"Win rate:       {win_rate:.1%}")
    print(f"Avg edge:       {avg_edge:.3f}")
    print(f"Total wagered:  ${total_wagered:,.2f}")
    print(f"Total PnL:      ${total_pnl:,.2f}")
    print(f"ROI:            {roi:.2%}")

    print("\nPnL by city:")
    by_city = trades_df.groupby("city")[["pnl", "size"]].sum()
    by_city["roi"] = by_city["pnl"] / by_city["size"]
    print(by_city.round(2).to_string())

    print("\nPnL by year:")
    trades_df["year"] = trades_df["date"].dt.year
    by_year = trades_df.groupby("year")[["pnl", "size"]].sum()
    by_year["roi"] = by_year["pnl"] / by_year["size"]
    print(by_year.round(2).to_string())

    # 6. Refit calibrator on the new backtest raw probabilities
    from sklearn.isotonic import IsotonicRegression
    print("\nRefitting calibrator on backtest raw_p vs outcomes...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(trades_df["raw_p"].values, trades_df["outcome"].values)
    cal_path = Path("data/calibrator_geq72.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"Calibrator saved to {cal_path}  ({len(trades_df)} training rows)")

    # 7. Cumulative PnL chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Overall bankroll growth (compounding)
    axes[0].plot(trades_df["date"], trades_df["bankroll"], linewidth=1.5)
    axes[0].axhline(BANKROLL_START, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"Bankroll (Compounding) — Walk-Forward Backtest  [start: ${BANKROLL_START:,.0f}]")
    axes[0].set_ylabel("Dollars")
    axes[0].grid(True, alpha=0.3)

    # Per-city cumulative PnL
    for city in trades_df["city"].unique():
        city_trades = trades_df[trades_df["city"] == city].copy()
        city_trades["cum_pnl"] = city_trades["pnl"].cumsum()
        axes[1].plot(city_trades["date"], city_trades["cum_pnl"], label=city, linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_title("Cumulative PnL by City")
    axes[1].set_ylabel("Dollars")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "logs/pnl_backtest.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
