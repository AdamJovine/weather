#!/usr/bin/env python3
"""
Compute ROI by 3-month period (calendar quarter), city, and bet type.

Bet types combine exact market payout criterion and side:
  - <x
  - x-y
  - >z
with YES and NO settled separately.

Output CSV columns:
  city, bet, market_type, side, quarter, period_start, period_end,
  trades, stake_total, pnl_total, roi, win_rate
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import pandas as pd

# Ensure `src/` imports work when running as a script from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract


def _contract_value(contract: dict) -> tuple[str, str]:
    mtype = str(contract["market_type"])
    if mtype == "range":
        return mtype, f"{int(contract['low'])}-{int(contract['high'])}"
    if mtype == "gt":
        return mtype, f">{int(contract['threshold'])}"
    if mtype == "lt":
        return mtype, f"<{int(contract['threshold'])}"
    if mtype == "geq":
        return mtype, f">={int(contract['threshold'])}"
    if mtype == "leq":
        return mtype, f"<={int(contract['threshold'])}"
    return mtype, mtype


def _load_history(db_path: str, min_volume: float) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT c.ticker, c.ts, c.close_dollars, c.volume,
                   m.city, m.settlement_date, m.title,
                   w.tmax
            FROM kalshi_candles c
            JOIN kalshi_markets m ON m.ticker = c.ticker
            LEFT JOIN weather_daily w
              ON w.city = m.city AND w.date = m.settlement_date
            """,
            conn,
        )

    df = df[df["close_dollars"].between(0.01, 0.99)].copy()
    df = df[df["volume"] >= float(min_volume)].copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    parsed = []
    for title in df["title"].astype(str):
        try:
            parsed.append(parse_contract(title, ""))
        except Exception:
            parsed.append(None)
    df["contract"] = parsed
    df = df[df["contract"].notna()].copy()
    df["market_type"] = [c["market_type"] for c in df["contract"]]
    values = [_contract_value(c) for c in df["contract"]]
    df["market_type"] = [v[0] for v in values]
    df["bet"] = [v[1] for v in values]
    return df


def _open_rows(df: pd.DataFrame, strict_band_only: bool) -> pd.DataFrame:
    open_ts = (
        df.groupby(["city", "settlement_date"])["ts"]
        .min()
        .rename("open_ts")
        .reset_index()
    )
    open_df = df.merge(open_ts, on=["city", "settlement_date"], how="inner")
    open_df = open_df[open_df["ts"] == open_df["open_ts"]].copy()

    meta = (
        open_df.groupby(["city", "settlement_date"])
        .agg(
            n_open_contracts=("ticker", "size"),
            open_types=("market_type", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    open_df = open_df.merge(meta, on=["city", "settlement_date"], how="left")

    if strict_band_only:
        open_df = open_df[
            (open_df["n_open_contracts"] >= 2)
        ].copy()
    return open_df


def _settle_trades(
    chosen: pd.DataFrame,
    stake: float,
    fee_rate: float,
    half_spread: float,
    side: str,
) -> pd.DataFrame:
    settled = chosen[chosen["tmax"].notna()].copy()
    if settled.empty:
        return settled

    settled["yes_outcome"] = [
        contract_yes_outcome(c, float(t))
        for c, t in zip(settled["contract"], settled["tmax"])
    ]
    settled["no_outcome"] = 1 - settled["yes_outcome"]

    outputs: list[pd.DataFrame] = []
    sides = ["yes", "no"] if side == "both" else [side]
    for bet_side in sides:
        part = settled.copy()
        part["side"] = bet_side
        if bet_side == "yes":
            part["outcome"] = part["yes_outcome"]
            part["price"] = (
                part["close_dollars"] + float(half_spread)
            ).clip(lower=0.01, upper=0.99)
        else:
            part["outcome"] = part["no_outcome"]
            part["price"] = (
                1.0 - part["close_dollars"] + float(half_spread)
            ).clip(lower=0.01, upper=0.99)

        part["stake"] = float(stake)
        part["contracts"] = part["stake"] / part["price"]
        gross_pnl = part["contracts"] * (part["outcome"] - part["price"])
        part["pnl"] = gross_pnl - float(fee_rate) * gross_pnl.clip(lower=0.0)
        outputs.append(part)

    return pd.concat(outputs, ignore_index=True)


def _quarterly_summary(settled: pd.DataFrame) -> pd.DataFrame:
    if settled.empty:
        return pd.DataFrame(
            columns=[
                "city", "bet", "market_type", "side", "quarter", "period_start", "period_end",
                "trades", "stake_total", "pnl_total", "roi", "win_rate",
            ]
        )

    s = settled.copy()
    q = s["settlement_date"].dt.to_period("Q")
    s["quarter"] = q.astype(str)
    s["period_start"] = q.dt.start_time.dt.date.astype(str)
    s["period_end"] = q.dt.end_time.dt.date.astype(str)

    out = (
        s.groupby(["city", "bet", "market_type", "side", "quarter", "period_start", "period_end"])
        .agg(
            trades=("pnl", "size"),
            stake_total=("stake", "sum"),
            pnl_total=("pnl", "sum"),
            win_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    out["roi"] = out["pnl_total"] / out["stake_total"]
    out = out[
        [
            "city",
            "bet",
            "market_type",
            "side",
            "quarter",
            "period_start",
            "period_end",
            "trades",
            "stake_total",
            "pnl_total",
            "win_rate",
            "roi",
        ]
    ]
    return out.sort_values(["city", "bet", "side", "period_start"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db", help="Path to SQLite DB")
    ap.add_argument("--stake", type=float, default=1.0, help="Dollars staked per trade")
    ap.add_argument("--fee-rate", type=float, default=0.02, help="Fee on winnings")
    ap.add_argument("--half-spread", type=float, default=0.02, help="Additive ask spread")
    ap.add_argument("--min-volume", type=float, default=10.0, help="Minimum candle volume")
    ap.add_argument(
        "--bet-values",
        default=None,
        help="Optional bet filter list: <65,65-66,>91",
    )
    ap.add_argument(
        "--side",
        choices=["yes", "no", "both"],
        default="both",
        help="Side filter",
    )
    ap.add_argument("--start", default=None, help="Optional start settlement date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional end settlement date YYYY-MM-DD")
    ap.add_argument("--city", default=None, help="Optional city filter")
    ap.add_argument("--cities", default=None, help="Optional city list: Miami,Houston,...")
    ap.add_argument(
        "--include-nonband-days",
        action="store_true",
        help="Include days without clear multi-band structure",
    )
    ap.add_argument(
        "--csv-out",
        default="logs/roi_by_3month_city_bet.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    df = _load_history(args.db, min_volume=args.min_volume)

    city_set: set[str] = set()
    if args.city:
        city_set.add(args.city.strip())
    if args.cities:
        city_set.update({c.strip() for c in args.cities.split(",") if c.strip()})
    if city_set:
        df = df[df["city"].isin(city_set)].copy()

    if args.start:
        start = pd.to_datetime(args.start)
        df = df[df["settlement_date"] >= start].copy()
    if args.end:
        end = pd.to_datetime(args.end)
        df = df[df["settlement_date"] <= end].copy()

    if df.empty:
        raise RuntimeError("No rows remain after filters.")

    open_df = _open_rows(df, strict_band_only=not args.include_nonband_days)
    chosen = open_df.copy()
    if args.bet_values:
        bet_set = {s.strip() for s in args.bet_values.split(",") if s.strip()}
        chosen = chosen[chosen["bet"].isin(bet_set)].copy()

    settled = _settle_trades(
        chosen,
        stake=args.stake,
        fee_rate=args.fee_rate,
        half_spread=args.half_spread,
        side=args.side,
    )
    if settled.empty:
        raise RuntimeError("No settled rows remain after filters.")

    out = _quarterly_summary(settled)
    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Rows written: {len(out)}")
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
