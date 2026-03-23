#!/usr/bin/env python3
"""
Hyperparameter search using pessimistic backtest assumptions.

Matches live-bot execution as closely as possible:
  - Pessimistic entry pricing: YES buy at candle high, NO buy at 1 - candle low.
  - Market settlement: final candle close >= 0.5 determines outcome.
  - Integer contracts only (no fractional).
  - Per-ticker dollar cap.
  - Same-day + next-day entry window (max_days_before_settlement=1).

Loads data once, then iterates parameter combos in-process for speed.
"""
from __future__ import annotations

import itertools
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_simple_mispricing_bot import (
    _build_daily_distribution,
    _compute_yes_prob,
    _compute_yes_prob_empirical,
    _apply_probability_safety,
    _summary,
)
from scripts.backtest_simple_ytd_pessimistic import (
    _load_market_rows,
    _load_final_close_by_ticker,
    _prepare_rows,
    _run_backtest,
)


# ── Fixed parameters ──────────────────────────────────────────────────
FIXED = dict(
    db="data/weather.db",
    fee_rate=0.02,
    check_minutes=1,
    prob_floor=0.03,
    prob_ceil=0.97,
    lookback_days=180,
    min_hist_days=45,
    sigma_floor=3.0,
    pessimistic_pricing=True,
    max_days_before_settlement=1,  # same-day + next-day (matches --trade-today --include-tomorrow)
)

# ── Search grid ───────────────────────────────────────────────────────
GRID = dict(
    ecmwf_weight=[0.50],
    spread_alpha=[0.20],
    confidence_shrink_k=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    use_empirical=[False, True],
    min_edge=[0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    max_edge=[0.45],
    uncertainty_z=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    side_mode=["best", "yes", "no"],
    stake_per_trade=[2.0],
    max_per_market=[8.0],
)


def _run_one(
    mkt_prepared: pd.DataFrame,
    dist_daily: pd.DataFrame,
    errors_dict: dict | None,
    final_close: pd.DataFrame,
    use_empirical: bool,
    confidence_shrink_k: float,
    min_edge: float,
    max_edge: float,
    uncertainty_z: float,
    side_mode: str,
    stake_per_trade: float,
    max_per_market: float,
) -> dict | None:
    """Run one backtest combo given pre-built distribution and market data."""
    mkt = mkt_prepared.merge(
        dist_daily,
        left_on=["city", "settlement_date"],
        right_on=["city", "date"],
        how="left",
    )
    mkt = mkt[mkt["mu"].notna() & mkt["sigma"].notna()].copy()
    if mkt.empty:
        return None

    # Apply market settlement
    mkt = mkt.merge(final_close, on="ticker", how="left")
    yes_from_market = (mkt["final_close_dollars"].astype(float) >= 0.5).astype(float)
    mkt["yes_outcome"] = np.where(
        mkt["final_close_dollars"].notna(),
        yes_from_market,
        mkt["yes_outcome_weather"].astype(float),
    ).astype(int)
    mkt["no_outcome"] = 1 - mkt["yes_outcome"]

    if use_empirical and errors_dict:
        mkt["p_yes_model"] = _compute_yes_prob_empirical(mkt, errors_dict)
    else:
        mkt["p_yes_model"] = _compute_yes_prob(mkt)
    mkt = mkt[mkt["p_yes_model"].notna()].copy()
    if mkt.empty:
        return None

    mkt = _apply_probability_safety(
        mkt,
        prob_floor=FIXED["prob_floor"],
        prob_ceil=FIXED["prob_ceil"],
        confidence_shrink_k=confidence_shrink_k,
    )

    trades = _run_backtest(
        rows=mkt,
        side_mode=side_mode,
        min_edge=min_edge,
        uncertainty_z=uncertainty_z,
        stake_per_trade=stake_per_trade,
        max_per_market=max_per_market,
        fee_rate=FIXED["fee_rate"],
        max_edge=max_edge,
        pessimistic_pricing=FIXED["pessimistic_pricing"],
    )
    if trades.empty:
        return None

    overall = _summary(trades, by=[])
    by_side = _summary(trades, by=["side"])
    by_city = _summary(trades, by=["city"])

    side_dict = {}
    for _, row in by_side.iterrows():
        s = row["side"]
        side_dict[f"roi_{s}"] = row["roi"]
        side_dict[f"wr_{s}"] = row["win_rate"]
        side_dict[f"n_{s}"] = int(row["trades"])

    city_roi = {row["city"]: round(row["roi"], 4) for _, row in by_city.iterrows()}

    return {
        "trades": int(overall["trades"].iloc[0]),
        "roi": float(overall["roi"].iloc[0]),
        "win_rate": float(overall["win_rate"].iloc[0]),
        "pnl": float(overall["pnl_total"].iloc[0]),
        **side_dict,
        "city_roi": city_roi,
    }


def main() -> None:
    db = FIXED["db"]
    print("Loading market data (with high/low for pessimistic pricing) …")
    mkt_base = _load_market_rows(db_path=db)
    print(f"  {len(mkt_base):,} market rows loaded.")

    # Filter to max_days_before_settlement
    mkt_base["_days_to_settle"] = (
        pd.to_datetime(mkt_base["settle_date"]) - pd.to_datetime(mkt_base["trade_date_et"])
    ).dt.days
    mkt_base = mkt_base[mkt_base["_days_to_settle"] <= FIXED["max_days_before_settlement"]].copy()
    print(f"  {len(mkt_base):,} rows after filtering to <={FIXED['max_days_before_settlement']} days before settlement.")

    # Prepare rows (slot dedup)
    mkt_base = _prepare_rows(mkt_base, check_minutes=FIXED["check_minutes"])
    print(f"  {len(mkt_base):,} rows after slot dedup.")

    # Load final close for market settlement
    final_close = _load_final_close_by_ticker(db_path=db)
    print(f"  {len(final_close):,} tickers with final close prices.")

    # Pre-compute distributions for each (ecmwf_weight, spread_alpha) combo.
    combo_keys = list(itertools.product(GRID["ecmwf_weight"], GRID["spread_alpha"]))
    dist_cache: dict[tuple, tuple[pd.DataFrame, dict | None]] = {}

    print(f"Building {len(combo_keys)} distribution variant(s) …")
    for ew, sa in combo_keys:
        t0 = time.time()
        result = _build_daily_distribution(
            db_path=db,
            lookback_days=FIXED["lookback_days"],
            min_hist_days=FIXED["min_hist_days"],
            sigma_floor=FIXED["sigma_floor"],
            spread_alpha=sa,
            ecmwf_weight=ew,
            return_errors=True,
        )
        dist_daily, errors_dict = result
        dist_cache[(ew, sa)] = (dist_daily, errors_dict)
        elapsed = time.time() - t0
        print(f"  ecmwf={ew:.2f} spread_alpha={sa:.2f}  ({elapsed:.1f}s)")

    # Run all combos.
    all_combos = list(itertools.product(
        GRID["ecmwf_weight"],
        GRID["spread_alpha"],
        GRID["confidence_shrink_k"],
        GRID["use_empirical"],
        GRID["min_edge"],
        GRID["max_edge"],
        GRID["uncertainty_z"],
        GRID["side_mode"],
        GRID["stake_per_trade"],
        GRID["max_per_market"],
    ))
    print(f"\nRunning {len(all_combos)} backtest combos …\n")

    results = []
    for i, (ew, sa, sk, emp, me, mxe, uz, sm, spt, mpm) in enumerate(all_combos, 1):
        dist_daily, errors_dict = dist_cache[(ew, sa)]
        t0 = time.time()
        out = _run_one(
            mkt_prepared=mkt_base,
            dist_daily=dist_daily,
            errors_dict=errors_dict,
            final_close=final_close,
            use_empirical=emp,
            confidence_shrink_k=sk,
            min_edge=me,
            max_edge=mxe,
            uncertainty_z=uz,
            side_mode=sm,
            stake_per_trade=spt,
            max_per_market=mpm,
        )
        elapsed = time.time() - t0
        label = (
            f"shrink={sk:.0f} emp={emp} min_e={me:.2f} uz={uz:.1f} side={sm}"
        )
        if out is None:
            if i % 50 == 0:
                print(f"  [{i}/{len(all_combos)}] {label}  → no trades  ({elapsed:.1f}s)")
            continue

        row = {
            "ecmwf_weight": ew,
            "spread_alpha": sa,
            "shrink_k": sk,
            "empirical": emp,
            "min_edge": me,
            "max_edge": mxe,
            "uncertainty_z": uz,
            "side_mode": sm,
            "stake_per_trade": spt,
            "max_per_market": mpm,
            **{k: v for k, v in out.items() if k != "city_roi"},
        }
        for city, croi in out.get("city_roi", {}).items():
            row[f"roi_{city}"] = croi
        results.append(row)

        if i % 20 == 0 or out["roi"] > 0.05:
            print(
                f"  [{i}/{len(all_combos)}] {label}  "
                f"ROI={out['roi']:+.1%}  WR={out['win_rate']:.1%}  "
                f"N={out['trades']}  PnL=${out['pnl']:.0f}  ({elapsed:.1f}s)"
            )

    if not results:
        print("\nNo combos produced trades.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 120)
    print("TOP 30 BY ROI")
    print("=" * 120)
    cols = ["shrink_k", "empirical", "min_edge", "uncertainty_z", "side_mode",
            "trades", "roi", "win_rate", "pnl"]
    for extra in ["roi_yes", "wr_yes", "n_yes", "roi_no", "wr_no", "n_no"]:
        if extra in df.columns:
            cols.append(extra)
    print(df[cols].head(30).to_string(index=False))

    print("\n" + "=" * 120)
    print("TOP 10 BY PNL (absolute profit)")
    print("=" * 120)
    df_by_pnl = df.sort_values("pnl", ascending=False)
    print(df_by_pnl[cols].head(10).to_string(index=False))

    print("\n" + "=" * 120)
    print("BOTTOM 5 BY ROI")
    print("=" * 120)
    print(df[cols].tail(5).to_string(index=False))

    # Best combo city breakdown
    best = df.iloc[0]
    print(f"\n{'='*120}")
    print(f"BEST COMBO (by ROI): shrink_k={best['shrink_k']:.0f} empirical={best['empirical']} "
          f"min_edge={best['min_edge']:.2f} uncertainty_z={best['uncertainty_z']:.1f} "
          f"side_mode={best['side_mode']}")
    print(f"  ROI={best['roi']:+.2%}  WR={best['win_rate']:.1%}  Trades={int(best['trades'])}  PnL=${best['pnl']:.2f}")
    city_cols = [c for c in df.columns if c.startswith("roi_") and c not in ("roi_yes", "roi_no")]
    if city_cols:
        print("  City ROIs:")
        for c in sorted(city_cols):
            city_name = c.replace("roi_", "")
            val = best.get(c)
            if pd.notna(val):
                print(f"    {city_name}: {val:+.1%}")

    # Best combo by PnL
    best_pnl = df_by_pnl.iloc[0]
    print(f"\nBEST COMBO (by PnL): shrink_k={best_pnl['shrink_k']:.0f} empirical={best_pnl['empirical']} "
          f"min_edge={best_pnl['min_edge']:.2f} uncertainty_z={best_pnl['uncertainty_z']:.1f} "
          f"side_mode={best_pnl['side_mode']}")
    print(f"  ROI={best_pnl['roi']:+.2%}  WR={best_pnl['win_rate']:.1%}  Trades={int(best_pnl['trades'])}  PnL=${best_pnl['pnl']:.2f}")
    if city_cols:
        print("  City ROIs:")
        for c in sorted(city_cols):
            city_name = c.replace("roi_", "")
            val = best_pnl.get(c)
            if pd.notna(val):
                print(f"    {city_name}: {val:+.1%}")

    # Save full results
    out_path = Path("logs/hyperparam_search_pessimistic.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
