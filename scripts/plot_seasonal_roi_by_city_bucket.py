#!/usr/bin/env python3
"""
Plot seasonal ROI by exact market bucket for each city.

Each plot is one city-season pair, with ROI bars by bucket value (<x, x-y, >z),
and YES/NO shown separately.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

# Ensure `src/` imports work when running as a script from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract


SEASON_ORDER = ["winter", "spring", "summer", "fall"]
SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}


def _contract_value_and_sort(contract: dict) -> tuple[str, str, int, int]:
    mtype = str(contract["market_type"])
    if mtype == "range":
        low = int(contract["low"])
        high = int(contract["high"])
        return mtype, f"{low}-{high}", 1, low
    if mtype == "gt":
        t = int(contract["threshold"])
        return mtype, f">{t}", 2, t
    if mtype == "lt":
        t = int(contract["threshold"])
        return mtype, f"<{t}", 0, t
    if mtype == "geq":
        t = int(contract["threshold"])
        return mtype, f">={t}", 2, t
    if mtype == "leq":
        t = int(contract["threshold"])
        return mtype, f"<={t}", 0, t
    return mtype, mtype, 3, 10**9


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

    contract_info = [_contract_value_and_sort(c) for c in df["contract"]]
    df["market_type"] = [x[0] for x in contract_info]
    df["bet"] = [x[1] for x in contract_info]
    df["bet_class"] = [x[2] for x in contract_info]
    df["bet_value"] = [x[3] for x in contract_info]
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

    if strict_band_only:
        meta = (
            open_df.groupby(["city", "settlement_date"])["ticker"]
            .size()
            .rename("n_open_contracts")
            .reset_index()
        )
        open_df = open_df.merge(meta, on=["city", "settlement_date"], how="left")
        open_df = open_df[open_df["n_open_contracts"] >= 2].copy()
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


def _seasonal_summary(settled: pd.DataFrame) -> pd.DataFrame:
    if settled.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "season",
                "bet",
                "market_type",
                "bet_class",
                "bet_value",
                "side",
                "trades",
                "stake_total",
                "pnl_total",
                "win_rate",
                "roi",
            ]
        )

    s = settled.copy()
    s["season"] = s["settlement_date"].dt.month.map(SEASON_MAP)
    out = (
        s.groupby(
            ["city", "season", "bet", "market_type", "bet_class", "bet_value", "side"]
        )
        .agg(
            trades=("pnl", "size"),
            stake_total=("stake", "sum"),
            pnl_total=("pnl", "sum"),
            win_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    out["roi"] = out["pnl_total"] / out["stake_total"]
    return out.sort_values(
        ["city", "season", "bet_class", "bet_value", "bet", "side"]
    ).reset_index(drop=True)


def _plot_city_season(df_cs: pd.DataFrame, city: str, season: str, out_dir: Path) -> Path:
    city_slug = city.lower().replace(" ", "_")
    out_path = out_dir / f"{city_slug}_{season}_roi_by_bucket.png"

    if df_cs.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.0))
        ax.axis("off")
        ax.set_title(f"{city} - {season.title()} - ROI by Bucket")
        ax.text(
            0.5,
            0.5,
            "No data for this city-season under current filters.",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    bets_order = (
        df_cs[["bet", "bet_class", "bet_value"]]
        .drop_duplicates()
        .sort_values(["bet_class", "bet_value", "bet"])["bet"]
        .tolist()
    )
    n_bets = len(bets_order)
    width = min(max(10.0, 0.32 * n_bets + 4.0), 30.0)

    fig, axes = plt.subplots(2, 1, figsize=(width, 8.0), sharex=True)
    side_cfg = [
        ("yes", axes[0], "#1f77b4"),
        ("no", axes[1], "#d62728"),
    ]

    for side, ax, color in side_cfg:
        d = df_cs[df_cs["side"] == side].copy()
        d = d.set_index("bet").reindex(bets_order).reset_index()
        ax.bar(d["bet"], d["roi"], color=color, alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel(f"{side.upper()} ROI")
        ax.grid(axis="y", alpha=0.25)

    axes[1].set_xlabel("Bucket Value")
    axes[1].tick_params(axis="x", rotation=90)
    fig.suptitle(f"{city} - {season.title()} - ROI by Bucket")
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db", help="Path to SQLite DB")
    ap.add_argument("--stake", type=float, default=1.0, help="Dollars staked per trade")
    ap.add_argument("--fee-rate", type=float, default=0.02, help="Fee on winnings")
    ap.add_argument("--half-spread", type=float, default=0.02, help="Additive ask spread")
    ap.add_argument("--min-volume", type=float, default=10.0, help="Minimum candle volume")
    ap.add_argument(
        "--side",
        choices=["yes", "no", "both"],
        default="both",
        help="Side filter",
    )
    ap.add_argument("--min-trades", type=int, default=1, help="Min trades per bucket to plot")
    ap.add_argument("--start", default=None, help="Optional start settlement date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional end settlement date YYYY-MM-DD")
    ap.add_argument("--city", default=None, help="Optional city filter")
    ap.add_argument("--cities", default=None, help="Optional city list: Miami,Houston,...")
    ap.add_argument(
        "--include-nonband-days",
        action="store_true",
        help="Include days without at least two contracts open",
    )
    ap.add_argument(
        "--csv-out",
        default="logs/seasonal_roi_by_city_bucket.csv",
        help="Output CSV path for seasonal aggregates",
    )
    ap.add_argument(
        "--out-dir",
        default="logs/seasonal_bucket_roi_plots",
        help="Directory for output plot images",
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
    settled = _settle_trades(
        open_df,
        stake=args.stake,
        fee_rate=args.fee_rate,
        half_spread=args.half_spread,
        side=args.side,
    )
    if settled.empty:
        raise RuntimeError("No settled rows remain after filters.")

    summary = _seasonal_summary(settled)
    summary = summary[summary["trades"] >= int(args.min_trades)].copy()
    if summary.empty:
        raise RuntimeError("No seasonal rows remain after min-trades filter.")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for city in sorted(open_df["city"].unique()):
        city_df = summary[summary["city"] == city]
        for season in SEASON_ORDER:
            df_cs = city_df[city_df["season"] == season].copy()
            saved.append(_plot_city_season(df_cs, city=city, season=season, out_dir=out_dir))

    print(f"Rows written: {len(summary)}")
    print(f"CSV: {csv_path}")
    print(f"Plots written: {len(saved)}")
    print(f"Plot dir: {out_dir}")


if __name__ == "__main__":
    main()
