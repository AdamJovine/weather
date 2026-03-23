#!/usr/bin/env python3
"""
Simple policy study on historical Kalshi weather markets at market open.

Policies:
  - bottom_no : Bet NO on the lowest available temperature bucket
  - top_no    : Bet NO on the highest available temperature bucket
  - both      : Place both bets each city/day (bottom_no + top_no)

Pricing model:
  no_price = clip(1 - yes_close + half_spread, 0.01, 0.99)
  PnL      = contracts * (no_outcome - no_price) - fee_rate * max(gross_pnl, 0)
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

# Ensure `src/` imports work when running as a script from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract


@dataclass
class StudyStats:
    trades: int
    unsettled_skipped: int
    stake_total: float
    pnl_total: float
    roi: float
    win_rate: float


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

    # Mirror backtest store sanity bounds and liquidity filter.
    df = df[df["close_dollars"].between(0.01, 0.99)].copy()
    df = df[df["volume"] >= float(min_volume)].copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

    parsed = []
    for title in df["title"].astype(str):
        try:
            parsed.append(parse_contract(title, ""))
        except Exception:
            parsed.append(None)
    df["contract"] = parsed
    df = df[df["contract"].notna()].copy()
    df["market_type"] = [c["market_type"] for c in df["contract"]]
    return df


def _low_key(contract: dict) -> int:
    # Smaller key = lower bucket.
    if contract["market_type"] in {"leq", "lt"}:
        return -10**9
    if contract["market_type"] == "range":
        return int(contract["low"])
    if contract["market_type"] in {"geq", "gt"}:
        return int(contract["threshold"])
    return 10**9


def _high_key(contract: dict) -> int:
    # Larger key = higher bucket.
    if contract["market_type"] in {"geq", "gt"}:
        return 10**9
    if contract["market_type"] == "range":
        return int(contract["high"])
    if contract["market_type"] in {"leq", "lt"}:
        return int(contract["threshold"])
    return -10**9


def _open_rows(df: pd.DataFrame, strict_band_only: bool) -> pd.DataFrame:
    df = df.copy()
    df["band_low_key"] = [_low_key(c) for c in df["contract"]]
    df["band_high_key"] = [_high_key(c) for c in df["contract"]]

    # "Market open" proxy: earliest available candle per (city, settlement_date).
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
        # Keep only city/dates with clear multi-band structure.
        open_df = open_df[
            (open_df["n_open_contracts"] >= 2)
            & open_df["open_types"].str.contains("range", regex=False)
        ].copy()
    return open_df


def _pick_policy_rows(open_df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy not in {"bottom_no", "top_no", "both"}:
        raise ValueError(f"Unsupported policy: {policy}")

    picks = []
    if policy in {"bottom_no", "both"}:
        bottom = (
            open_df.sort_values(
                ["city", "settlement_date", "band_low_key", "ticker"],
                ascending=[True, True, True, True],
            )
            .groupby(["city", "settlement_date"], as_index=False)
            .first()
        )
        bottom["policy"] = "bottom_no"
        picks.append(bottom)

    if policy in {"top_no", "both"}:
        top = (
            open_df.sort_values(
                ["city", "settlement_date", "band_high_key", "ticker"],
                ascending=[True, True, False, True],
            )
            .groupby(["city", "settlement_date"], as_index=False)
            .first()
        )
        top["policy"] = "top_no"
        picks.append(top)

    if not picks:
        return pd.DataFrame(columns=open_df.columns.tolist() + ["policy"])
    return pd.concat(picks, ignore_index=True)


def _settle_no_trades(
    chosen: pd.DataFrame,
    stake: float,
    fee_rate: float,
    half_spread: float,
) -> tuple[pd.DataFrame, int]:
    unsettled_skipped = int(chosen["tmax"].isna().sum())
    settled = chosen[chosen["tmax"].notna()].copy()
    if settled.empty:
        return settled, unsettled_skipped

    settled["yes_outcome"] = [
        contract_yes_outcome(c, float(t))
        for c, t in zip(settled["contract"], settled["tmax"])
    ]
    settled["no_outcome"] = 1 - settled["yes_outcome"]

    settled["no_price"] = (
        1.0 - settled["close_dollars"] + float(half_spread)
    ).clip(lower=0.01, upper=0.99)

    settled["stake"] = float(stake)
    settled["contracts"] = settled["stake"] / settled["no_price"]
    gross_pnl = settled["contracts"] * (settled["no_outcome"] - settled["no_price"])
    settled["pnl"] = gross_pnl - float(fee_rate) * gross_pnl.clip(lower=0.0)
    settled["year"] = pd.to_datetime(settled["settlement_date"]).dt.year
    return settled, unsettled_skipped


def _stats(trades: pd.DataFrame, unsettled_skipped: int) -> StudyStats:
    if trades.empty:
        return StudyStats(
            trades=0,
            unsettled_skipped=unsettled_skipped,
            stake_total=0.0,
            pnl_total=0.0,
            roi=0.0,
            win_rate=0.0,
        )
    stake_total = float(trades["stake"].sum())
    pnl_total = float(trades["pnl"].sum())
    return StudyStats(
        trades=int(len(trades)),
        unsettled_skipped=unsettled_skipped,
        stake_total=stake_total,
        pnl_total=pnl_total,
        roi=(pnl_total / stake_total) if stake_total else 0.0,
        win_rate=float(trades["no_outcome"].mean()),
    )


def _summary_table(trades: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=by + ["trades", "stake_total", "pnl_total", "roi", "win_rate"])
    out = (
        trades.groupby(by)
        .agg(
            trades=("pnl", "size"),
            stake_total=("stake", "sum"),
            pnl_total=("pnl", "sum"),
            win_rate=("no_outcome", "mean"),
        )
        .reset_index()
    )
    out["roi"] = out["pnl_total"] / out["stake_total"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db", help="Path to SQLite DB")
    ap.add_argument("--stake", type=float, default=1.0, help="Dollars staked per trade")
    ap.add_argument("--fee-rate", type=float, default=0.02, help="Fee on winnings")
    ap.add_argument("--half-spread", type=float, default=0.02, help="Additive ask spread")
    ap.add_argument("--min-volume", type=float, default=10.0, help="Minimum candle volume")
    ap.add_argument(
        "--policy",
        choices=["bottom_no", "top_no", "both"],
        default="bottom_no",
        help="Policy to backtest",
    )
    ap.add_argument("--start", default=None, help="Optional start settlement date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional end settlement date YYYY-MM-DD")
    ap.add_argument("--city", default=None, help="Optional city filter")
    ap.add_argument(
        "--cities",
        default=None,
        help="Optional city basket filter, comma list like Miami,Houston,Austin",
    )
    ap.add_argument(
        "--months",
        default=None,
        help="Optional month filter, comma list like 6,7,8",
    )
    ap.add_argument("--no-price-min", type=float, default=None, help="Optional minimum NO ask")
    ap.add_argument("--no-price-max", type=float, default=None, help="Optional maximum NO ask")
    ap.add_argument(
        "--include-nonband-days",
        action="store_true",
        help="Include days without clear multi-band structure",
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write settled trade-level rows",
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
        start = pd.to_datetime(args.start).date()
        df = df[df["settlement_date"] >= start].copy()
    if args.end:
        end = pd.to_datetime(args.end).date()
        df = df[df["settlement_date"] <= end].copy()
    if df.empty:
        raise RuntimeError("No rows remain after filters.")

    open_df = _open_rows(df, strict_band_only=not args.include_nonband_days)
    chosen = _pick_policy_rows(open_df, policy=args.policy)
    settled, unsettled_skipped = _settle_no_trades(
        chosen,
        stake=args.stake,
        fee_rate=args.fee_rate,
        half_spread=args.half_spread,
    )
    if args.months:
        month_set = {
            int(x.strip())
            for x in args.months.split(",")
            if x.strip()
        }
        settled = settled[pd.to_datetime(settled["settlement_date"]).dt.month.isin(month_set)].copy()
    if args.no_price_min is not None:
        settled = settled[settled["no_price"] >= float(args.no_price_min)].copy()
    if args.no_price_max is not None:
        settled = settled[settled["no_price"] <= float(args.no_price_max)].copy()
    if settled.empty:
        raise RuntimeError("No settled rows remain after filters.")

    stats = _stats(settled, unsettled_skipped)
    date_min = str(settled["settlement_date"].min())
    date_max = str(settled["settlement_date"].max())
    mode = "strict band-only" if not args.include_nonband_days else "all parsed days"

    print(f"Mode: {mode}")
    print(f"Policy: {args.policy}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Cities: {settled['city'].nunique()}")
    print(f"Trades: {stats.trades}")
    print(f"Unsettled skipped: {stats.unsettled_skipped}")
    print(f"Stake/trade: ${args.stake:.2f}")
    print(f"Fee rate: {args.fee_rate:.2%}  |  Half spread: ${args.half_spread:.2f}")
    print(f"Total staked: ${stats.stake_total:,.2f}")
    print(f"Total PnL: ${stats.pnl_total:,.2f}")
    print(f"ROI: {stats.roi:+.2%}")
    print(f"NO win rate: {stats.win_rate:.2%}")

    by_policy = _summary_table(settled, by=["policy"])
    by_year = _summary_table(settled, by=["year", "policy"])
    by_city = _summary_table(settled, by=["city", "policy"])

    if not by_policy.empty:
        print("\nOverall by policy:")
        print(by_policy.round(4).to_string(index=False))
    if not by_year.empty:
        print("\nYearly by policy:")
        print(by_year.sort_values(["year", "policy"]).round(4).to_string(index=False))
    if not by_city.empty:
        print("\nCity by policy:")
        print(by_city.sort_values(["policy", "roi"], ascending=[True, False]).round(4).to_string(index=False))

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        settled.to_csv(out, index=False)
        print(f"\nWrote settled trades: {out}")


if __name__ == "__main__":
    main()
