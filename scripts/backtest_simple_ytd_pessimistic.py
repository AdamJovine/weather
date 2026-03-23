#!/usr/bin/env python3
"""
Year-to-date pessimistic backtest for the simple live mispricing algorithm.

Execution assumptions (worst-case):
  - Buys execute at the candle high on YES.
  - For NO buys, price is derived from the worst YES low complement: 1 - low_yes.
  - No optimistic intraday exits; trades are held to settlement.

The script mirrors current live-style defaults (stake, min-edge, uncertainty-z,
probability safety, side mode, per-ticker cap) and reports daily ROI.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract
from scripts.backtest_simple_mispricing_bot import (
    _apply_probability_safety,
    _build_daily_distribution,
    _compute_yes_prob,
    _compute_yes_prob_empirical,
)


def _safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num) / float(den)


def _load_market_rows(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                c.ticker,
                c.ts,
                c.close_dollars,
                c.high_dollars,
                c.low_dollars,
                c.volume,
                m.city,
                m.settlement_date,
                m.title,
                w.tmax
            FROM kalshi_candles c
            JOIN kalshi_markets m
              ON m.ticker = c.ticker
            LEFT JOIN weather_daily w
              ON w.city = m.city AND w.date = m.settlement_date
            """,
            conn,
        )

    if df.empty:
        return df

    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["ts_et"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["trade_date_et"] = df["ts_et"].dt.date
    df["settle_date"] = df["settlement_date"].dt.date
    df = df[df["trade_date_et"] <= df["settle_date"]].copy()
    df = df[df["tmax"].notna()].copy()

    parsed = []
    for title in df["title"].astype(str):
        try:
            parsed.append(parse_contract(title, ""))
        except Exception:
            parsed.append(None)
    df["contract"] = parsed
    df = df[df["contract"].notna()].copy()

    df["market_type"] = [str(c["market_type"]) for c in df["contract"]]
    df["threshold"] = [float(c.get("threshold", np.nan)) for c in df["contract"]]
    df["low"] = [float(c.get("low", np.nan)) for c in df["contract"]]
    df["high"] = [float(c.get("high", np.nan)) for c in df["contract"]]
    df["yes_outcome_weather"] = [
        contract_yes_outcome(c, float(t))
        for c, t in zip(df["contract"], df["tmax"])
    ]
    df["no_outcome_weather"] = 1 - df["yes_outcome_weather"]
    df["yes_outcome"] = df["yes_outcome_weather"]
    df["no_outcome"] = df["no_outcome_weather"]
    return df


def _load_final_close_by_ticker(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT c.ticker, c.close_dollars AS final_close_dollars
            FROM (
                SELECT ticker, MAX(ts) AS max_ts
                FROM kalshi_candles
                GROUP BY ticker
            ) x
            JOIN kalshi_candles c
              ON c.ticker = x.ticker AND c.ts = x.max_ts
            """,
            conn,
        )
    return df


def _prepare_rows(df: pd.DataFrame, check_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    slot = int(max(1, check_minutes)) * 60
    out = df.copy()
    out["check_slot"] = (out["ts"] // slot).astype(np.int64)
    out = (
        out.sort_values(["ticker", "ts"])
        .groupby(["ticker", "check_slot"], as_index=False)
        .first()
    )
    return out


def _run_backtest(
    rows: pd.DataFrame,
    side_mode: str,
    min_edge: float,
    uncertainty_z: float,
    stake_per_trade: float,
    max_per_market: float,
    fee_rate: float,
    max_edge: float = 1.0,
    pessimistic_pricing: bool = True,
    skip_edge_lo: float = 0.0,
    skip_edge_hi: float = 0.0,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    if pessimistic_pricing:
        # Pessimistic entry prices:
        # - YES buy at candle high
        # - NO buy at complement of candle YES low
        high_yes = rows["high_dollars"].astype(float).fillna(rows["close_dollars"].astype(float))
        low_yes = rows["low_dollars"].astype(float).fillna(rows["close_dollars"].astype(float))
        rows["yes_ask_pess"] = high_yes.clip(lower=0.01, upper=0.99)
        rows["no_ask_pess"] = (1.0 - low_yes).clip(lower=0.01, upper=0.99)
    else:
        # Mid-price: use candle close
        close = rows["close_dollars"].astype(float)
        rows["yes_ask_pess"] = close.clip(lower=0.01, upper=0.99)
        rows["no_ask_pess"] = (1.0 - close).clip(lower=0.01, upper=0.99)

    rows["edge_yes"] = rows["p_yes_used"] - rows["yes_ask_pess"]
    rows["edge_no"] = (1.0 - rows["p_yes_used"]) - rows["no_ask_pess"]
    n_eff = rows["n_hist"].astype(float).clip(lower=1.0)
    p = rows["p_yes_used"].astype(float).clip(0.0, 1.0)
    rows["effective_min_edge"] = float(min_edge) + float(uncertainty_z) * np.sqrt((p * (1.0 - p)) / n_eff)

    if side_mode == "yes":
        rows["side"] = "yes"
        rows["edge_chosen"] = rows["edge_yes"]
        rows["ask_chosen"] = rows["yes_ask_pess"]
        rows["outcome_chosen"] = rows["yes_outcome"]
    elif side_mode == "no":
        rows["side"] = "no"
        rows["edge_chosen"] = rows["edge_no"]
        rows["ask_chosen"] = rows["no_ask_pess"]
        rows["outcome_chosen"] = rows["no_outcome"]
    else:
        choose_yes = rows["edge_yes"] >= rows["edge_no"]
        rows["side"] = np.where(choose_yes, "yes", "no")
        rows["edge_chosen"] = np.where(choose_yes, rows["edge_yes"], rows["edge_no"])
        rows["ask_chosen"] = np.where(choose_yes, rows["yes_ask_pess"], rows["no_ask_pess"])
        rows["outcome_chosen"] = np.where(choose_yes, rows["yes_outcome"], rows["no_outcome"])

    edge_ok = (rows["edge_chosen"] >= rows["effective_min_edge"]) & (rows["edge_chosen"] <= float(max_edge))
    if float(skip_edge_lo) > 0 and float(skip_edge_hi) > 0:
        edge_ok &= ~((rows["edge_chosen"] > float(skip_edge_lo)) & (rows["edge_chosen"] < float(skip_edge_hi)))
    rows = rows[edge_ok].copy()
    if rows.empty:
        return rows

    rows = rows.sort_values(["ts", "ticker"]).copy()
    ticker_spend: dict[str, float] = {}
    kept: list[dict] = []

    for _, r in rows.iterrows():
        ask = float(r["ask_chosen"])
        contracts = max(1, int(float(stake_per_trade) / ask))
        while contracts > 1 and (float(contracts) * ask) > (float(stake_per_trade) + 1e-9):
            contracts -= 1
        est_spend = float(contracts) * ask
        ticker = str(r["ticker"])
        used = float(ticker_spend.get(ticker, 0.0))
        if used + est_spend > float(max_per_market) + 1e-9:
            continue

        ticker_spend[ticker] = used + est_spend
        rec = dict(r)
        rec["contracts"] = float(contracts)
        rec["stake"] = est_spend
        kept.append(rec)

    if not kept:
        return pd.DataFrame()

    out = pd.DataFrame(kept)
    gross = out["contracts"] * (out["outcome_chosen"].astype(float) - out["ask_chosen"].astype(float))
    out["pnl"] = gross - float(fee_rate) * gross.clip(lower=0.0)
    out["settlement_date_only"] = pd.to_datetime(out["settlement_date"]).dt.date
    out["trade_date_only"] = pd.to_datetime(out["trade_date_et"]).dt.date
    out["roi_trade"] = np.where(out["stake"] > 0, out["pnl"] / out["stake"], 0.0)
    return out


def _daily_roi(trades: pd.DataFrame, by_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[by_col, "trades", "stake_total", "pnl_total", "roi"])
    d = (
        trades.groupby(by_col)
        .agg(
            trades=("pnl", "size"),
            stake_total=("stake", "sum"),
            pnl_total=("pnl", "sum"),
        )
        .reset_index()
        .sort_values(by_col)
    )
    d["roi"] = d["pnl_total"] / d["stake_total"]
    return d


def main() -> None:
    today = date.today()
    ytd_start = date(today.year, 1, 1)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db")
    ap.add_argument("--start", default=str(ytd_start), help="YYYY-MM-DD (default: Jan 1 this year)")
    ap.add_argument("--end", default=str(today), help="YYYY-MM-DD (default: today)")
    ap.add_argument("--cities", default=None, help="Optional comma list")
    ap.add_argument("--side-mode", choices=["best", "yes", "no"], default="no")
    ap.add_argument("--stake-per-trade", type=float, default=5.0)
    ap.add_argument("--max-per-market", type=float, default=15.0)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--uncertainty-z", type=float, default=1.0)
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--min-hist-days", type=int, default=45)
    ap.add_argument("--sigma-floor", type=float, default=3.0)
    ap.add_argument("--spread-alpha", type=float, default=0.20)
    ap.add_argument("--ecmwf-weight", type=float, default=0.50)
    ap.add_argument("--nbm-weight", type=float, default=0.30, help="Weight for NWS NBM forecast when available (0=disabled).")
    ap.add_argument("--disagree-alpha", type=float, default=0.15, help="Sigma inflation per degree of GFS/ECMWF disagreement.")
    ap.add_argument("--t850-sigma-alpha", type=float, default=0.0, help="Sigma inflation based on 850hPa temperature anomaly.")
    ap.add_argument("--prob-floor", type=float, default=0.03)
    ap.add_argument("--prob-ceil", type=float, default=0.97)
    ap.add_argument("--confidence-shrink-k", type=float, default=0.0)
    ap.add_argument("--max-edge", type=float, default=1.0, help="Skip trades with edge above this (0=disabled).")
    ap.add_argument("--skip-edge-lo", type=float, default=0.05, help="Lower bound of edge skip zone (0=disabled).")
    ap.add_argument("--skip-edge-hi", type=float, default=0.20, help="Upper bound of edge skip zone (0=disabled).")
    ap.add_argument("--pessimistic-pricing", action=argparse.BooleanOptionalAction, default=True, help="Use pessimistic entry prices (candle high/low). Disable for mid-price (close).")
    ap.add_argument("--check-minutes", type=int, default=1)
    ap.add_argument("--fee-rate", type=float, default=0.02)
    ap.add_argument(
        "--same-day-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only trade markets that settle on the same ET date as the trade timestamp.",
    )
    ap.add_argument(
        "--max-days-before-settlement",
        type=int,
        default=2,
        help="Only trade within N days of settlement (0=no filter). Ignored when --same-day-only is enabled.",
    )
    ap.add_argument(
        "--settlement-source",
        choices=["market", "weather"],
        default="market",
        help="How to determine settled outcome: market=final candle close (>=0.5 YES), weather=weather_daily tmax.",
    )
    ap.add_argument("--use-empirical", action=argparse.BooleanOptionalAction, default=True,
                    help="Use empirical error CDF instead of Normal CDF for probabilities.")
    ap.add_argument("--daily-csv", default="logs/simple_pessimistic_daily_roi_ytd.csv")
    ap.add_argument("--trades-csv", default="logs/simple_pessimistic_trades_ytd.csv")
    args = ap.parse_args()

    build_result = _build_daily_distribution(
        db_path=args.db,
        lookback_days=args.lookback_days,
        min_hist_days=args.min_hist_days,
        sigma_floor=args.sigma_floor,
        spread_alpha=args.spread_alpha,
        ecmwf_weight=args.ecmwf_weight,
        nbm_weight=float(args.nbm_weight),
        disagree_alpha=float(args.disagree_alpha),
        t850_sigma_alpha=float(args.t850_sigma_alpha),
        return_errors=bool(args.use_empirical),
    )
    if bool(args.use_empirical):
        dist_daily, errors_dict = build_result
    else:
        dist_daily = build_result
        errors_dict = None
    mkt = _load_market_rows(args.db)
    if mkt.empty:
        raise RuntimeError("No market rows loaded from DB.")

    final_close = _load_final_close_by_ticker(args.db)
    mkt = mkt.merge(final_close, on="ticker", how="left")
    if str(args.settlement_source) == "market":
        if mkt["final_close_dollars"].notna().sum() == 0:
            raise RuntimeError("No final close prices available to resolve market outcomes.")
        yes_from_market = (mkt["final_close_dollars"].astype(float) >= 0.5).astype(float)
        mkt["yes_outcome"] = np.where(
            mkt["final_close_dollars"].notna(),
            yes_from_market,
            mkt["yes_outcome_weather"].astype(float),
        ).astype(int)
        mkt["no_outcome"] = 1 - mkt["yes_outcome"]

    mkt = mkt.merge(
        dist_daily,
        left_on=["city", "settlement_date"],
        right_on=["city", "date"],
        how="left",
    )
    mkt = mkt[mkt["mu"].notna() & mkt["sigma"].notna()].copy()
    mkt = mkt[
        (mkt["settlement_date"] >= pd.to_datetime(args.start))
        & (mkt["settlement_date"] <= pd.to_datetime(args.end))
    ].copy()
    if args.cities:
        city_set = {c.strip() for c in str(args.cities).split(",") if c.strip()}
        if city_set:
            mkt = mkt[mkt["city"].isin(city_set)].copy()
    if mkt.empty:
        raise RuntimeError("No rows remain after date/city filters.")

    if errors_dict is not None:
        mkt["p_yes_model"] = _compute_yes_prob_empirical(mkt, errors_dict)
    else:
        mkt["p_yes_model"] = _compute_yes_prob(mkt)
    mkt = mkt[mkt["p_yes_model"].notna()].copy()
    mkt = _apply_probability_safety(
        mkt,
        prob_floor=float(args.prob_floor),
        prob_ceil=float(args.prob_ceil),
        confidence_shrink_k=float(args.confidence_shrink_k),
    )
    mkt["_days_to_settle"] = (
        pd.to_datetime(mkt["settle_date"]) - pd.to_datetime(mkt["trade_date_et"])
    ).dt.days
    if bool(args.same_day_only):
        mkt = mkt[mkt["_days_to_settle"] == 0].copy()
    elif int(args.max_days_before_settlement) > 0:
        mkt = mkt[mkt["_days_to_settle"] <= int(args.max_days_before_settlement)].copy()
    mkt = _prepare_rows(mkt, check_minutes=int(args.check_minutes))

    trades = _run_backtest(
        rows=mkt,
        side_mode=str(args.side_mode),
        min_edge=float(args.min_edge),
        uncertainty_z=float(args.uncertainty_z),
        stake_per_trade=float(args.stake_per_trade),
        max_per_market=float(args.max_per_market),
        fee_rate=float(args.fee_rate),
        max_edge=float(args.max_edge),
        pessimistic_pricing=bool(args.pessimistic_pricing),
        skip_edge_lo=float(args.skip_edge_lo),
        skip_edge_hi=float(args.skip_edge_hi),
    )

    if trades.empty:
        print("No trades triggered under current settings.")
        return

    daily_settle = _daily_roi(trades, by_col="settlement_date_only")
    overall_roi = _safe_div(float(trades["pnl"].sum()), float(trades["stake"].sum()))

    print("Simple Algorithm YTD Pessimistic Backtest")
    print(f"Range: {args.start} to {args.end}")
    print(f"Probability model: {'empirical CDF' if args.use_empirical else 'Normal CDF'}")
    print(f"Settlement source: {args.settlement_source}")
    if bool(args.same_day_only):
        print("Entry window: same-day only")
    elif int(args.max_days_before_settlement) > 0:
        print(f"Entry window: <= {int(args.max_days_before_settlement)} day(s) before settlement")
    else:
        print("Entry window: unrestricted")
    if str(args.settlement_source) == "market":
        cmp = (
            mkt[["ticker", "yes_outcome", "yes_outcome_weather"]]
            .drop_duplicates(subset=["ticker"])
            .copy()
        )
        if not cmp.empty:
            agree = (cmp["yes_outcome"] == cmp["yes_outcome_weather"]).mean()
            print(
                "Weather-vs-market outcome agreement: "
                f"{agree:.2%} ({len(cmp)} tickers)"
            )
    print(f"Trades: {len(trades)} | Tickers: {trades['ticker'].nunique()}")
    print(
        f"Total stake=${float(trades['stake'].sum()):,.2f} "
        f"P&L=${float(trades['pnl'].sum()):,.2f} ROI={overall_roi:+.2%}"
    )
    print("\nDaily ROI by settlement date:")
    print(
        daily_settle.rename(columns={"settlement_date_only": "date"}).to_string(
            index=False,
            formatters={
                "stake_total": "{:,.2f}".format,
                "pnl_total": "{:,.2f}".format,
                "roi": "{:+.2%}".format,
            },
        )
    )

    daily_out = Path(args.daily_csv)
    trades_out = Path(args.trades_csv)
    daily_out.parent.mkdir(parents=True, exist_ok=True)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    daily_settle.rename(columns={"settlement_date_only": "date"}).to_csv(daily_out, index=False)
    trades.to_csv(trades_out, index=False)
    print(f"\nWrote daily ROI: {daily_out}")
    print(f"Wrote trades: {trades_out}")


if __name__ == "__main__":
    main()
