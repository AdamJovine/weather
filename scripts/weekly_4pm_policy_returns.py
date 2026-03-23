#!/usr/bin/env python3
"""
Run a 1-week backtest of the simple mispricing policy and report 4pm-to-4pm daily returns.

Fee model used here (as requested):
  - 2% entry fee
  - 2% exit fee
So each trade applies both legs (round-trip fee impact).
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_simple_mispricing_bot import (
    _apply_probability_safety,
    _build_daily_distribution,
    _compute_yes_prob,
    _load_market_rows,
    _pick_entries,
)


def _settle_round_trip_fee(
    entries: pd.DataFrame,
    stake: float,
    entry_fee_rate: float,
    exit_fee_rate: float,
) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()

    t = entries.copy()
    t["stake"] = float(stake)
    t["contracts"] = t["stake"] / t["price_chosen"]
    # Cash paid when entering (stake plus entry fee).
    t["entry_cost"] = t["stake"] * (1.0 + float(entry_fee_rate))
    # Cash received at settlement (if wins), net of exit fee.
    t["exit_value"] = t["contracts"] * t["outcome_chosen"] * (1.0 - float(exit_fee_rate))
    t["pnl"] = t["exit_value"] - t["entry_cost"]
    return t


def _to_4pm_session_day(ts_et: pd.Series) -> pd.Series:
    # Session starts at 4:00 PM ET and ends just before next 4:00 PM ET.
    # Shift back 16 hours, then take date as the session label.
    shifted = ts_et - pd.Timedelta(hours=16)
    return shifted.dt.date


def _summarize_by_session(trades: pd.DataFrame) -> pd.DataFrame:
    out = (
        trades.groupby("session_day")
        .agg(
            trades=("pnl", "size"),
            notional_stake=("stake", "sum"),
            entry_cost_total=("entry_cost", "sum"),
            exit_value_total=("exit_value", "sum"),
            pnl_total=("pnl", "sum"),
            win_rate=("outcome_chosen", "mean"),
        )
        .reset_index()
        .sort_values("session_day")
    )
    out["roi_on_entry_cost"] = out["pnl_total"] / out["entry_cost_total"]
    out["cum_pnl"] = out["pnl_total"].cumsum()
    return out


def _plot_daily_roi(df: pd.DataFrame, out_png: Path) -> None:
    x = [d.isoformat() for d in df["session_day"]]
    y = 100.0 * df["roi_on_entry_cost"]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in y]
    ax.bar(x, y, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_title("Policy ROI by 4pm-to-4pm Session (with 2% in + 2% out fees)")
    ax.set_xlabel("Session Start Date (ET, 4pm)")
    ax.set_ylabel("ROI (%) on Entry Cost")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD; default latest settlement date in filtered data")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--cities", default=None, help="Comma list; default all cities")
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--entry-fee-rate", type=float, default=0.02)
    ap.add_argument("--exit-fee-rate", type=float, default=0.02)
    ap.add_argument("--check-minutes", type=int, default=5)
    ap.add_argument("--half-spread", type=float, default=0.0)
    ap.add_argument("--min-volume", type=float, default=10.0)
    ap.add_argument("--min-price", type=float, default=0.05)
    ap.add_argument("--max-price", type=float, default=0.95)
    ap.add_argument("--allow-settlement-day", action="store_true")
    ap.add_argument("--side-mode", choices=["best", "yes", "no"], default="no")
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--uncertainty-z", type=float, default=1.0)
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--min-hist-days", type=int, default=45)
    ap.add_argument("--sigma-floor", type=float, default=3.0)
    ap.add_argument("--spread-alpha", type=float, default=0.20)
    ap.add_argument("--ecmwf-weight", type=float, default=0.50)
    ap.add_argument("--prob-floor", type=float, default=0.03)
    ap.add_argument("--prob-ceil", type=float, default=0.97)
    ap.add_argument("--confidence-shrink-k", type=float, default=60.0)
    ap.add_argument("--min-minutes-between-ticker-trades", type=int, default=0)
    ap.add_argument("--max-trades-per-ticker-day", type=int, default=3)
    ap.add_argument("--max-daily-trades", type=int, default=0)
    ap.add_argument("--max-city-daily-trades", type=int, default=0)
    ap.add_argument("--csv-out", default="logs/weekly_4pm_policy_returns.csv")
    ap.add_argument("--plot-out", default="logs/weekly_4pm_policy_returns.png")
    ap.add_argument("--trades-out", default="logs/weekly_4pm_policy_trades.csv")
    args = ap.parse_args()

    dist_daily = _build_daily_distribution(
        db_path=args.db,
        lookback_days=args.lookback_days,
        min_hist_days=args.min_hist_days,
        sigma_floor=args.sigma_floor,
        spread_alpha=args.spread_alpha,
        ecmwf_weight=args.ecmwf_weight,
    )
    mkt = _load_market_rows(
        db_path=args.db,
        min_volume=args.min_volume,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    mkt = mkt.merge(
        dist_daily,
        left_on=["city", "settlement_date"],
        right_on=["city", "date"],
        how="left",
    )
    mkt = mkt[mkt["mu"].notna() & mkt["sigma"].notna()].copy()

    if args.cities:
        city_set = {x.strip() for x in args.cities.split(",") if x.strip()}
        mkt = mkt[mkt["city"].isin(city_set)].copy()

    if args.end:
        end_date = pd.to_datetime(args.end).date()
    else:
        end_date = pd.to_datetime(mkt["settlement_date"]).dt.date.max()
    start_date = end_date - timedelta(days=max(1, int(args.days)) - 1)

    mkt = mkt[
        (pd.to_datetime(mkt["settlement_date"]).dt.date >= start_date)
        & (pd.to_datetime(mkt["settlement_date"]).dt.date <= end_date)
    ].copy()

    mkt["trade_date_et"] = mkt["ts_et"].dt.date
    mkt["settle_date"] = pd.to_datetime(mkt["settlement_date"]).dt.date
    if not args.allow_settlement_day:
        mkt = mkt[mkt["trade_date_et"] < mkt["settle_date"]].copy()
    if mkt.empty:
        raise RuntimeError("No market rows remain after filters.")

    mkt["p_yes_model"] = _compute_yes_prob(mkt)
    mkt = mkt[mkt["p_yes_model"].notna()].copy()
    mkt = _apply_probability_safety(
        mkt,
        prob_floor=args.prob_floor,
        prob_ceil=args.prob_ceil,
        confidence_shrink_k=args.confidence_shrink_k,
    )

    entries = _pick_entries(
        mkt,
        check_minutes=args.check_minutes,
        side_mode=args.side_mode,
        min_edge=args.min_edge,
        uncertainty_z=args.uncertainty_z,
        half_spread=args.half_spread,
        min_minutes_between_ticker_trades=args.min_minutes_between_ticker_trades,
        max_trades_per_ticker_day=args.max_trades_per_ticker_day,
        max_daily_trades=args.max_daily_trades,
        max_city_daily_trades=args.max_city_daily_trades,
    )
    if entries.empty:
        raise RuntimeError("No trades triggered under current settings.")

    trades = _settle_round_trip_fee(
        entries=entries,
        stake=args.stake,
        entry_fee_rate=args.entry_fee_rate,
        exit_fee_rate=args.exit_fee_rate,
    )
    trades["session_day"] = _to_4pm_session_day(trades["ts_et"])

    daily = _summarize_by_session(trades)

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(csv_out, index=False)

    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_out, index=False)

    plot_out = Path(args.plot_out)
    _plot_daily_roi(daily, plot_out)

    print("Weekly 4pm Session Backtest")
    print(f"Window: {start_date} to {end_date}")
    print(f"Trades: {len(trades)} | Tickers: {trades['ticker'].nunique()}")
    print(f"Side mode: {args.side_mode} | Check interval: {args.check_minutes} min")
    print(
        f"Fees: entry={args.entry_fee_rate:.2%}, exit={args.exit_fee_rate:.2%}, "
        f"round-trip multiplier={(1.0 + args.entry_fee_rate) * (1.0 + args.exit_fee_rate):.6f}"
    )
    print()
    print(daily.to_string(index=False))
    print()
    print(f"Daily CSV: {csv_out}")
    print(f"Trades CSV: {trades_out}")
    print(f"Plot: {plot_out}")


if __name__ == "__main__":
    main()
