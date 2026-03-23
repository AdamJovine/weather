#!/usr/bin/env python3
"""
Scan simple weather-market strategies with walk-forward robustness checks.

Two execution styles are analyzed:
  1) open_per_market  : first available candle per ticker
  2) first_per_hour   : first available candle per (ticker, hour_et)

Each strategy is a simple filter on:
  - city
  - side (yes/no)
  - market_type (lt/range/gt)
  - optional hour_et

For each rule family, this script:
  - runs rolling walk-forward folds (train years < test year)
  - computes fold-level ROI stability metrics
  - bootstraps confidence intervals on mean fold ROI
  - applies Benjamini-Hochberg FDR correction

Outputs CSVs in logs/ by default.
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure `src/` imports work when running from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract


def _load_base(
    db_path: str,
    min_volume: float,
    min_price: float,
    max_price: float,
    tz: str,
) -> pd.DataFrame:
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

    df = df[df["volume"] >= float(min_volume)].copy()
    df = df[df["close_dollars"].between(float(min_price), float(max_price))].copy()
    df = df[df["tmax"].notna()].copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["year"] = df["settlement_date"].dt.year.astype(int)
    df["ts_et"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz)
    df["hour_et"] = df["ts_et"].dt.hour.astype(int)

    parsed = []
    for title in df["title"].astype(str):
        try:
            parsed.append(parse_contract(title, ""))
        except Exception:
            parsed.append(None)
    df["contract"] = parsed
    df = df[df["contract"].notna()].copy()

    df["market_type"] = [c["market_type"] for c in df["contract"]]
    df["yes_outcome"] = [
        contract_yes_outcome(c, float(t))
        for c, t in zip(df["contract"], df["tmax"])
    ]
    df["no_outcome"] = 1 - df["yes_outcome"]
    return df


def _attach_side_pnl(df: pd.DataFrame, half_spread: float, fee_rate: float) -> pd.DataFrame:
    out = df.copy()
    out["yes_price"] = (out["close_dollars"] + float(half_spread)).clip(lower=0.01, upper=0.99)
    out["no_price"] = (1.0 - out["close_dollars"] + float(half_spread)).clip(lower=0.01, upper=0.99)

    yes_gross = (1.0 / out["yes_price"]) * (out["yes_outcome"] - out["yes_price"])
    no_gross = (1.0 / out["no_price"]) * (out["no_outcome"] - out["no_price"])
    out["yes_pnl"] = yes_gross - float(fee_rate) * yes_gross.clip(lower=0.0)
    out["no_pnl"] = no_gross - float(fee_rate) * no_gross.clip(lower=0.0)
    return out


def _to_long_sides(df: pd.DataFrame) -> pd.DataFrame:
    yes = df[["ticker", "city", "year", "hour_et", "market_type", "yes_pnl", "yes_price", "yes_outcome"]].copy()
    yes = yes.rename(columns={"yes_pnl": "pnl", "yes_price": "price", "yes_outcome": "win"})
    yes["side"] = "yes"

    no = df[["ticker", "city", "year", "hour_et", "market_type", "no_pnl", "no_price", "no_outcome"]].copy()
    no = no.rename(columns={"no_pnl": "pnl", "no_price": "price", "no_outcome": "win"})
    no["side"] = "no"

    return pd.concat([yes, no], ignore_index=True)


def _safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return float("nan")
    return float(num) / float(den)


def _benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """Return BH-adjusted q-values for a vector of p-values."""
    arr = pvals.to_numpy(dtype=float)
    q = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return pd.Series(q, index=pvals.index)

    pv = arr[idx]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    unsorted = np.empty_like(q_ranked)
    unsorted[order] = q_ranked
    q[idx] = unsorted
    return pd.Series(q, index=pvals.index)


def _rule_seed(base_seed: int, group_key: tuple) -> int:
    s = "|".join(str(x) for x in group_key)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int((int(base_seed) + int(h, 16)) % (2**32 - 1))


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Bootstrap mean CI and one-sided p-value for mean <= 0."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        mean = float(arr[0])
        p_nonpos = 1.0 if mean <= 0.0 else 0.0
        return mean, mean, mean, p_nonpos

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(arr), size=(int(n_bootstrap), len(arr)))
    draws = arr[idx].mean(axis=1)
    alpha = 1.0 - float(ci_level)
    low = float(np.quantile(draws, alpha / 2.0))
    high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    p_nonpos = float(np.mean(draws <= 0.0))
    return float(arr.mean()), low, high, p_nonpos


def _build_walk_forward_folds(
    long_df: pd.DataFrame,
    group_cols: list[str],
    test_years: list[int],
    min_train_years: int,
    min_train_trades: int,
    min_test_trades: int,
) -> pd.DataFrame:
    yearly = (
        long_df.groupby(group_cols + ["year"])
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            win_sum=("win", "sum"),
            price_sum=("price", "sum"),
        )
        .reset_index()
    )
    if yearly.empty:
        return pd.DataFrame()

    yearly = yearly.sort_values(group_cols + ["year"]).reset_index(drop=True)
    grp = yearly.groupby(group_cols, sort=False)

    yearly["train_years"] = grp.cumcount()
    yearly["train_trades"] = grp["trades"].cumsum() - yearly["trades"]
    yearly["train_pnl"] = grp["pnl"].cumsum() - yearly["pnl"]
    yearly["train_win_sum"] = grp["win_sum"].cumsum() - yearly["win_sum"]
    yearly["train_price_sum"] = grp["price_sum"].cumsum() - yearly["price_sum"]

    yearly["train_roi"] = yearly["train_pnl"] / yearly["train_trades"]
    yearly["train_win_rate"] = yearly["train_win_sum"] / yearly["train_trades"]
    yearly["train_avg_price"] = yearly["train_price_sum"] / yearly["train_trades"]

    yearly["test_year"] = yearly["year"]
    yearly["test_trades"] = yearly["trades"]
    yearly["test_pnl"] = yearly["pnl"]
    yearly["test_roi"] = yearly["pnl"] / yearly["trades"]
    yearly["test_win_rate"] = yearly["win_sum"] / yearly["trades"]
    yearly["test_avg_price"] = yearly["price_sum"] / yearly["trades"]

    folds = yearly[
        yearly["test_year"].isin(test_years)
        & (yearly["train_years"] >= int(min_train_years))
        & (yearly["train_trades"] >= int(min_train_trades))
        & (yearly["test_trades"] >= int(min_test_trades))
    ].copy()

    cols = (
        group_cols
        + [
            "test_year",
            "train_years",
            "train_trades",
            "train_pnl",
            "train_roi",
            "train_win_rate",
            "train_avg_price",
            "test_trades",
            "test_pnl",
            "test_roi",
            "test_win_rate",
            "test_avg_price",
        ]
    )
    return folds[cols].reset_index(drop=True)


def _aggregate_walk_forward(
    folds: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()

    rows = []
    for key, sub in folds.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)

        fold_rois = sub["test_roi"].to_numpy(dtype=float)
        mean_boot, ci_low, ci_high, p_nonpos = _bootstrap_mean_ci(
            fold_rois,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=_rule_seed(seed, key),
        )

        fold_std = float(sub["test_roi"].std(ddof=0))
        fold_med = float(sub["test_roi"].median())
        fold_mean = float(sub["test_roi"].mean())
        pos_folds = int((sub["test_roi"] > 0).sum())
        n_folds = int(len(sub))
        test_trades_total = float(sub["test_trades"].sum())
        test_pnl_total = float(sub["test_pnl"].sum())

        rec = dict(zip(group_cols, key))
        rec.update(
            {
                "n_folds": n_folds,
                "first_test_year": int(sub["test_year"].min()),
                "last_test_year": int(sub["test_year"].max()),
                "positive_folds": pos_folds,
                "positive_fold_frac": _safe_div(pos_folds, n_folds),
                "train_trades_median": float(sub["train_trades"].median()),
                "train_roi_median": float(sub["train_roi"].median()),
                "test_trades_total": test_trades_total,
                "test_pnl_total": test_pnl_total,
                "test_roi_pooled": _safe_div(test_pnl_total, test_trades_total),
                "test_roi_mean_fold": fold_mean,
                "test_roi_median_fold": fold_med,
                "test_roi_std_fold": fold_std,
                "bootstrap_mean_fold_roi": mean_boot,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_boot_nonpos": p_nonpos,
                "score_stability": fold_med - 0.5 * fold_std,
            }
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = _benjamini_hochberg(out["p_boot_nonpos"])
    return out.sort_values(
        ["score_stability", "test_roi_pooled", "test_trades_total"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _flag_robust_rules(
    agg: pd.DataFrame,
    min_folds: int,
    min_positive_fold_frac: float,
    fdr_alpha: float,
    min_test_total_trades: int,
) -> pd.DataFrame:
    if agg.empty:
        return agg

    out = agg.copy()
    out["robust_rule"] = (
        (out["n_folds"] >= int(min_folds))
        & (out["positive_fold_frac"] >= float(min_positive_fold_frac))
        & (out["ci_low"] > 0.0)
        & (out["q_value"] <= float(fdr_alpha))
        & (out["test_roi_pooled"] > 0.0)
        & (out["test_trades_total"] >= float(min_test_total_trades))
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db", help="Path to SQLite DB")
    ap.add_argument(
        "--split-year",
        type=int,
        default=2026,
        help="Last walk-forward test year (inclusive)",
    )
    ap.add_argument(
        "--start-test-year",
        type=int,
        default=None,
        help="First walk-forward test year (defaults to second year in data)",
    )
    ap.add_argument("--min-volume", type=float, default=10.0, help="Min candle volume")
    ap.add_argument("--min-price", type=float, default=0.05, help="Min close_dollars")
    ap.add_argument("--max-price", type=float, default=0.95, help="Max close_dollars")
    ap.add_argument("--half-spread", type=float, default=0.02, help="Additive spread to ask")
    ap.add_argument("--fee-rate", type=float, default=0.02, help="Fee rate on winnings")
    ap.add_argument("--tz", default="America/New_York", help="Timezone for hour bucketing")
    ap.add_argument("--min-train", type=int, default=150, help="Min train trades per rule")
    ap.add_argument("--min-test", type=int, default=40, help="Min test trades per rule")
    ap.add_argument("--min-train-years", type=int, default=1, help="Min train years in a fold")
    ap.add_argument("--min-folds", type=int, default=2, help="Min valid folds per rule")
    ap.add_argument(
        "--min-positive-fold-frac",
        type=float,
        default=0.60,
        help="Min fraction of folds with positive test ROI",
    )
    ap.add_argument(
        "--min-test-total-trades",
        type=int,
        default=200,
        help="Min total test trades across folds",
    )
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws per rule")
    ap.add_argument("--ci-level", type=float, default=0.95, help="Bootstrap CI level")
    ap.add_argument("--fdr-alpha", type=float, default=0.10, help="FDR threshold")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    ap.add_argument(
        "--out-prefix",
        default="logs/simple_strategy_scan",
        help="Prefix for output CSV files",
    )
    args = ap.parse_args()

    base = _load_base(
        db_path=args.db,
        min_volume=args.min_volume,
        min_price=args.min_price,
        max_price=args.max_price,
        tz=args.tz,
    )
    base = _attach_side_pnl(base, half_spread=args.half_spread, fee_rate=args.fee_rate)

    open_entry = base.sort_values(["ticker", "ts"]).groupby("ticker", as_index=False).first()
    hourly_entry = (
        base.sort_values(["ticker", "ts"])
        .groupby(["ticker", "hour_et"], as_index=False)
        .first()
    )

    scans = {
        "open_per_market": _to_long_sides(open_entry),
        "first_per_hour": _to_long_sides(hourly_entry),
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for scan_name, long_df in scans.items():
        years = sorted(int(y) for y in long_df["year"].dropna().unique())
        if len(years) < 2:
            print(f"\nScan: {scan_name} | skipped: need at least 2 years of data")
            continue

        start_test_year = int(args.start_test_year) if args.start_test_year is not None else years[1]
        end_test_year = int(args.split_year)
        test_years = [y for y in years if start_test_year <= y <= end_test_year]
        if not test_years:
            print(f"\nScan: {scan_name} | skipped: no test years in range")
            continue

        templates = [
            ("city_side_type", ["city", "side", "market_type"], int(args.min_train), int(args.min_test)),
            (
                "side_type_hour",
                ["side", "market_type", "hour_et"],
                max(120, int(args.min_train // 2)),
                max(30, int(args.min_test // 2)),
            ),
            (
                "city_side_type_hour",
                ["city", "side", "market_type", "hour_et"],
                max(80, int(args.min_train // 2)),
                max(20, int(args.min_test // 2)),
            ),
        ]

        print(f"\nScan: {scan_name}")
        print(f"Rows: {len(long_df):,} | test years: {test_years[0]}-{test_years[-1]} ({len(test_years)} folds max)")

        for template_name, cols, min_train_trades, min_test_trades in templates:
            folds = _build_walk_forward_folds(
                long_df=long_df,
                group_cols=cols,
                test_years=test_years,
                min_train_years=int(args.min_train_years),
                min_train_trades=min_train_trades,
                min_test_trades=min_test_trades,
            )
            agg = _aggregate_walk_forward(
                folds=folds,
                group_cols=cols,
                n_bootstrap=int(args.bootstrap),
                ci_level=float(args.ci_level),
                seed=int(args.seed),
            )
            agg = _flag_robust_rules(
                agg,
                min_folds=int(args.min_folds),
                min_positive_fold_frac=float(args.min_positive_fold_frac),
                fdr_alpha=float(args.fdr_alpha),
                min_test_total_trades=int(args.min_test_total_trades),
            )

            p_agg = out_prefix.with_name(f"{out_prefix.name}_{scan_name}_{template_name}_wf.csv")
            p_folds = out_prefix.with_name(f"{out_prefix.name}_{scan_name}_{template_name}_folds.csv")
            agg.to_csv(p_agg, index=False)
            folds.to_csv(p_folds, index=False)

            robust = agg[agg.get("robust_rule", False)].copy()
            robust = robust.sort_values(
                ["score_stability", "test_roi_pooled", "test_trades_total"],
                ascending=[False, False, False],
            )

            print(f"\nTemplate: {template_name}")
            print(f"Rules: {len(agg):,} | robust: {len(robust):,}")
            print(f"CSV: {p_agg}")
            print(f"CSV: {p_folds}")
            if not robust.empty:
                print("Top robust rules:")
                print(robust.head(10).to_string(index=False))
            else:
                print("Top robust rules: none")


if __name__ == "__main__":
    main()
