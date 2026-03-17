"""
Replay best hyperparameters with a LIVE COMPOUNDING bankroll.

Two-phase approach:
  Phase 1 — collect all trade signals across all cities (no sizing yet)
  Phase 2 — sort by date, apply compounding:
             same-day trades are all sized off the start-of-day bankroll
             (they're placed simultaneously before outcomes are known),
             then all PnL settles at end of day and bankroll updates.

Reports: final balance, CAGR, max drawdown.
"""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from src.features import build_feature_table
import src.model as _mmod
from src.model import BayesianTempModel, FEATURES
from src.config import TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER

# ── Hyperparameters ────────────────────────────────────────────────
PARAMS = dict(
    alpha_1        = 6.11479e-05,
    alpha_2        = 4.86713e-06,
    lambda_1       = 2.5343e-05,
    lambda_2       = 3.44075e-07,
    spread_alpha   = 0.951836,
    lookback       = 2052,
    decay_halflife = 39,
    kelly_fraction = 0.40,
    max_bet_frac   = 0.25,
    min_p_sel      = 0.40,
    min_edge       = 0.05,
)

BANKROLL_START    = 500.0
MIN_TRAIN_ROWS    = 365
THRESHOLD_OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
N_THOMPSON_DRAWS  = 100
EDGE_THRESHOLD    = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER + PARAMS["min_edge"]


def climo_prob_geq(threshold, climo_mean, climo_sigma, spread=0.0):
    mkt_sigma = climo_sigma * (1.0 + PARAMS["spread_alpha"] * spread)
    return float(norm.sf(threshold - 0.5, loc=climo_mean, scale=mkt_sigma))


def thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices):
    mu_draws    = np.random.normal(mu_hat, max(epistemic_std, 0.01), N_THOMPSON_DRAWS)
    total_sigma = float(np.sqrt(epistemic_std**2 + aleatoric**2))
    counts: dict = {}
    for mu_s in mu_draws:
        best_edge, best_key = EDGE_THRESHOLD, None
        for threshold, mkt_p in zip(thresholds, mkt_prices):
            model_p  = float(norm.sf(threshold - 0.5, loc=mu_s, scale=aleatoric))
            yes_edge = model_p - mkt_p
            no_edge  = mkt_p - model_p
            if yes_edge > best_edge and model_p <= 0.90:
                best_edge, best_key = yes_edge, (threshold, "yes")
            if no_edge  > best_edge and (1 - model_p) <= 0.90:
                best_edge, best_key = no_edge,  (threshold, "no")
        if best_key:
            counts[best_key] = counts.get(best_key, 0) + 1
    if not counts:
        return []
    trades = []
    for (threshold, side), count in counts.items():
        p_sel       = count / N_THOMPSON_DRAWS
        mkt_p       = mkt_prices[thresholds.index(threshold)]
        exp_model_p = float(norm.sf(threshold - 0.5, loc=mu_hat, scale=total_sigma))
        exp_edge    = (exp_model_p - mkt_p) if side == "yes" else (mkt_p - exp_model_p)
        if exp_edge > 0:
            trades.append(dict(threshold=threshold, side=side, mkt_p=mkt_p,
                               exp_edge=exp_edge, p_sel=p_sel, ei=p_sel * exp_edge))
    trades.sort(key=lambda x: -x["ei"])
    return trades


def kelly_size(edge, bankroll):
    if edge <= 0:
        return 0.0
    return min(bankroll * PARAMS["max_bet_frac"], bankroll * edge * PARAMS["kelly_fraction"])


def make_model():
    m = BayesianTempModel()
    m._model = BayesianRidge(
        alpha_1=PARAMS["alpha_1"], alpha_2=PARAMS["alpha_2"],
        lambda_1=PARAMS["lambda_1"], lambda_2=PARAMS["lambda_2"],
    )
    m._lookback       = int(PARAMS["lookback"])
    m._decay_halflife = float(PARAMS["decay_halflife"])
    return m


def main():
    # ── Load data ──────────────────────────────────────────────────
    hist    = pd.read_csv("data/historical_tmax.csv")
    gfs_df  = pd.read_csv("data/forecasts/openmeteo_forecast_history.csv")
    idx_df  = pd.read_csv("data/climate_indices.csv")        if Path("data/climate_indices.csv").exists()          else None
    gefs_df = pd.read_csv("data/forecasts/gefs_spread.csv")  if Path("data/forecasts/gefs_spread.csv").exists()   else None

    forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=idx_df, gefs_df=gefs_df)
    df["forecast_high"]        = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    climo_sigma_map = {}
    for city in df["city"].unique():
        cdf = df[df["city"] == city].dropna(subset=["y_tmax", "climo_mean_doy"])
        climo_sigma_map[city] = float((cdf["y_tmax"] - cdf["climo_mean_doy"]).std())

    _mmod.SPREAD_ALPHA = PARAMS["spread_alpha"]

    # ── Phase 1: collect raw signals (no sizing) ───────────────────
    print("Phase 1: generating signals across all cities...")
    raw_signals = []

    for city in sorted(df["city"].unique()):
        city_df     = df[df["city"] == city].copy().reset_index(drop=True)
        climo_sigma = climo_sigma_map[city]
        print(f"  {city}")

        for i in range(MIN_TRAIN_ROWS, len(city_df)):
            test_row = city_df.iloc[i]
            if pd.isna(test_row["y_tmax"]):
                continue
            if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                continue

            train_df = city_df.iloc[:i]
            model    = make_model()
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
            mkt_prices = [climo_prob_geq(t, climo_mean, climo_sigma, spread_val) for t in thresholds]
            candidates = thompson_select(mu_hat, epistemic_std, aleatoric, thresholds, mkt_prices)

            for trade_info in candidates[:2]:
                if trade_info["p_sel"]    < PARAMS["min_p_sel"]:
                    continue
                if trade_info["exp_edge"] < EDGE_THRESHOLD:
                    continue
                raw_outcome = 1 if y_tmax >= trade_info["threshold"] else 0
                raw_signals.append(dict(
                    date      = pd.Timestamp(test_row["date"]),
                    city      = city,
                    threshold = trade_info["threshold"],
                    side      = trade_info["side"],
                    mkt_p     = trade_info["mkt_p"],
                    edge      = trade_info["exp_edge"],
                    p_sel     = trade_info["p_sel"],
                    ei        = trade_info["ei"],
                    mu_hat    = mu_hat,
                    y_tmax    = y_tmax,
                    raw_outcome = raw_outcome,
                ))

    print(f"\nPhase 1 complete: {len(raw_signals)} signals across all cities.\n")

    # ── Phase 2: apply compounding bankroll ────────────────────────
    signals_df = pd.DataFrame(raw_signals).sort_values("date").reset_index(drop=True)

    # ── Phase 2: log-space compounding ────────────────────────────
    # Bankroll can grow astronomically — track log10(bankroll) to avoid overflow.
    # Same-day trades are all sized off the start-of-day bankroll (simultaneous entry),
    # so we sum fractional returns for the day, then do one log update.

    log10_br      = math.log10(BANKROLL_START)   # tracks log10(bankroll)
    peak_log10_br = log10_br
    max_drawdown  = 0.0   # in log10 terms: peak_log10 - trough_log10
    result_rows   = []
    trade_num     = 0

    print(f"{'='*105}")
    print(f"COMPOUNDING REPLAY (log-space)  |  start: ${BANKROLL_START:,.2f}  |  edge threshold: {EDGE_THRESHOLD:.3f}")
    print(f"  kelly={PARAMS['kelly_fraction']:.2f}  max_bet={PARAMS['max_bet_frac']:.2f}  "
          f"min_p_sel={PARAMS['min_p_sel']:.2f}  spread_alpha={PARAMS['spread_alpha']:.3f}")
    print(f"{'='*105}")
    print(f"{'#':>6}  {'Date':<12} {'City':<16} {'Side':<4} {'Thr':>4} "
          f"{'MktP':>6} {'Edge':>6} {'pSel':>5} {'BetFrac':>8} {'RetFrac':>8} {'log10($)':>9}  Win?")
    print("-" * 105)

    for date, day_df in signals_df.groupby("date"):
        day_frac_return = 0.0   # sum of (bet_fraction * payoff_factor) for all trades today

        day_trades_buf = []
        for _, row in day_df.iterrows():
            # fraction of bankroll to bet (capped by max_bet_frac)
            f = min(PARAMS["max_bet_frac"], row["edge"] * PARAMS["kelly_fraction"])

            if row["side"] == "yes":
                payoff_frac = f * (row["raw_outcome"] - row["mkt_p"])
                adj_outcome = int(row["raw_outcome"])
            else:
                payoff_frac = f * ((1 - row["raw_outcome"]) - (1 - row["mkt_p"]))
                adj_outcome = 1 - int(row["raw_outcome"])

            day_frac_return += payoff_frac
            trade_num       += 1
            win              = adj_outcome == 1

            day_trades_buf.append(dict(
                date=date, city=row["city"], threshold=row["threshold"], side=row["side"],
                mkt_p=row["mkt_p"], edge=row["edge"], p_sel=row["p_sel"],
                mu_hat=row["mu_hat"], y_tmax=row["y_tmax"],
                bet_frac=f, payoff_frac=payoff_frac, outcome=adj_outcome,
            ))

            # log10(bankroll) after this day's full settlement (approximate mid-day display)
            approx_log10 = log10_br + math.log10(max(1 + day_frac_return, 1e-300))

            print(f"{trade_num:>6}  {str(date.date()):<12} {row['city']:<16} {row['side']:<4} "
                  f"{row['threshold']:>4} {row['mkt_p']:>6.3f} {row['edge']:>6.3f} "
                  f"{row['p_sel']:>5.2f} {f:>8.3f} {payoff_frac:>+8.4f} {approx_log10:>9.3f}  "
                  f"{'W' if win else 'L'}")

        # End of day: update log bankroll
        new_log10_br = log10_br + math.log10(max(1 + day_frac_return, 1e-300))
        log10_br = new_log10_br

        if log10_br > peak_log10_br:
            peak_log10_br = log10_br
        dd = peak_log10_br - log10_br   # drawdown in log10 units
        if dd > max_drawdown:
            max_drawdown = dd

        for t in day_trades_buf:
            t["log10_bankroll"] = log10_br
            result_rows.append(t)

        if log10_br < -6:   # bankroll < $0.000001 — effectively ruined
            print("*** BANKROLL RUINED ***")
            break

    print("=" * 105)

    if not result_rows:
        print("No trades.")
        return

    trades_df = pd.DataFrame(result_rows)
    trades_df["date"] = pd.to_datetime(trades_df["date"])

    # ── Summary ────────────────────────────────────────────────────
    start_date  = trades_df["date"].min()
    end_date    = trades_df["date"].max()
    years       = (end_date - start_date).days / 365.25

    # CAGR from log10: (log10_final - log10_start) / years = log10(CAGR+1)
    log10_gain  = log10_br - math.log10(BANKROLL_START)
    cagr        = 10 ** (log10_gain / years) - 1
    win_rate    = (trades_df["payoff_frac"] > 0).mean()
    avg_edge    = trades_df["edge"].mean()

    # Max drawdown as a fraction: 10^(-dd_log10) - 1
    max_dd_pct  = 1 - 10 ** (-max_drawdown)

    print(f"\n=== SUMMARY ===")
    print(f"Starting bankroll:    ${BANKROLL_START:,.2f}")
    print(f"Final log10(bankroll): {log10_br:.2f}  →  10^{log10_br:.1f}")
    if log10_br < 15:
        print(f"Final bankroll:       ${10**log10_br:,.2f}")
    else:
        print(f"Final bankroll:       ~$10^{log10_br:.1f}  (astronomical)")
    print(f"CAGR:                 {cagr:.1%}  ({cagr:.2e}x per year)")
    print(f"Max drawdown:         {max_dd_pct:.2%}  ({max_drawdown:.3f} log10 units)")
    print(f"Total trades:         {len(trades_df):,}")
    print(f"Win rate:             {win_rate:.1%}")
    print(f"Avg edge:             {avg_edge:.3f}")
    print(f"Period:               {start_date.date()} → {end_date.date()}  ({years:.1f} years)")
    print(f"\nYears to reach:")
    for target in [1_000, 10_000, 100_000, 1_000_000, 1_000_000_000]:
        log_target = math.log10(target)
        if log_target > math.log10(BANKROLL_START):
            yrs = (log_target - math.log10(BANKROLL_START)) / (log10_gain / years)
            print(f"  ${target:>15,.0f}  →  {yrs:.2f} years")

    print("\nBy city (fractional PnL contribution):")
    by_city = trades_df.groupby("city").agg(
        trades=("payoff_frac", "count"),
        win_rate=("outcome", "mean"),
        avg_edge=("edge", "mean"),
        avg_bet_frac=("bet_frac", "mean"),
        total_payoff_frac=("payoff_frac", "sum"),
    ).round(4)
    print(by_city.to_string())

    # ── Save ───────────────────────────────────────────────────────
    out = Path("logs/replay_trades.csv")
    trades_df.to_csv(out, index=False)
    print(f"\nTrades saved to {out}")

    # ── Chart: log10 bankroll over time ───────────────────────────
    daily_log10 = trades_df.groupby("date")["log10_bankroll"].last().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(daily_log10["date"], daily_log10["log10_bankroll"], linewidth=1.5)
    axes[0].axhline(math.log10(BANKROLL_START), color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"log₁₀(Bankroll)  |  CAGR {cagr:.1%}  |  Max DD {max_dd_pct:.2%}")
    axes[0].set_ylabel("log₁₀(Bankroll $)")
    axes[0].grid(True, alpha=0.3)

    # Daily fractional return
    daily_ret = trades_df.groupby("date")["payoff_frac"].sum()
    axes[1].bar(daily_ret.index, daily_ret.values, linewidth=0, width=1,
                color=["green" if r > 0 else "red" for r in daily_ret.values], alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Daily Fractional Return (sum of bet_frac × payoff across all trades)")
    axes[1].set_ylabel("Fractional Return")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart = "logs/replay_pnl.png"
    plt.savefig(chart, dpi=150)
    print(f"Chart saved to {chart}")


if __name__ == "__main__":
    main()
