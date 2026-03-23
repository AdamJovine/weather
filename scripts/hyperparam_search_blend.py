#!/usr/bin/env python3
"""
Hyperparameter search over forecast-blend, spread_alpha, confidence-shrink-k,
and use-empirical.

Loads data once, then iterates parameter combos in-process for speed.
"""
from __future__ import annotations

import itertools
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
    _load_market_rows,
    _compute_yes_prob,
    _compute_yes_prob_empirical,
    _apply_probability_safety,
    _pick_entries,
    _settle,
    _summary,
)
from src.strategy import parse_contract, contract_yes_outcome

# ── Fixed parameters (from previous optimisation) ──────────────────────
FIXED = dict(
    db="data/weather.db",
    min_volume=10.0,
    min_price=0.05,
    max_price=0.95,
    stake=1.0,
    fee_rate=0.02,
    half_spread=0.02,
    check_minutes=5,
    side_mode="best",
    min_edge=0.15,
    uncertainty_z=1.0,
    prob_floor=0.03,
    prob_ceil=0.97,
    min_minutes_between_ticker_trades=0,
    max_trades_per_ticker_day=3,
    max_daily_trades=0,
    max_city_daily_trades=0,
    lookback_days=180,
    min_hist_days=45,
    sigma_floor=3.0,
)

# ── Search grid ─────────────────────────────────────────────────────────
GRID = dict(
    ecmwf_weight=[0.30, 0.50, 0.70],
    spread_alpha=[0.0, 0.10, 0.20, 0.35, 0.50],
    confidence_shrink_k=[0.0, 10.0, 30.0, 60.0],
    use_empirical=[False, True],
)


def _run_one(
    mkt_base: pd.DataFrame,
    dist_daily: pd.DataFrame,
    errors_dict: dict | None,
    use_empirical: bool,
    confidence_shrink_k: float,
) -> dict | None:
    """Run one backtest combo given pre-built distribution and market data."""
    mkt = mkt_base.merge(
        dist_daily,
        left_on=["city", "settlement_date"],
        right_on=["city", "date"],
        how="left",
    )
    mkt = mkt[mkt["mu"].notna() & mkt["sigma"].notna()].copy()

    mkt["trade_date_et"] = mkt["ts_et"].dt.date
    mkt["settle_date"] = mkt["settlement_date"].dt.date
    mkt = mkt[mkt["trade_date_et"] < mkt["settle_date"]].copy()
    if mkt.empty:
        return None

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

    entries = _pick_entries(
        mkt,
        check_minutes=FIXED["check_minutes"],
        side_mode=FIXED["side_mode"],
        min_edge=FIXED["min_edge"],
        uncertainty_z=FIXED["uncertainty_z"],
        half_spread=FIXED["half_spread"],
        min_minutes_between_ticker_trades=FIXED["min_minutes_between_ticker_trades"],
        max_trades_per_ticker_day=FIXED["max_trades_per_ticker_day"],
        max_daily_trades=FIXED["max_daily_trades"],
        max_city_daily_trades=FIXED["max_city_daily_trades"],
    )
    trades = _settle(entries, stake=FIXED["stake"], fee_rate=FIXED["fee_rate"])
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
    print("Loading market data …")
    mkt_base = _load_market_rows(
        db_path=db,
        min_volume=FIXED["min_volume"],
        min_price=FIXED["min_price"],
        max_price=FIXED["max_price"],
    )
    print(f"  {len(mkt_base):,} market rows loaded.")

    # Pre-compute distributions for each (ecmwf_weight, spread_alpha) combo.
    # Cache them so we don't rebuild for each shrink_k / empirical toggle.
    combo_keys = list(itertools.product(GRID["ecmwf_weight"], GRID["spread_alpha"]))
    dist_cache: dict[tuple, tuple[pd.DataFrame, dict | None]] = {}

    print(f"Building {len(combo_keys)} distribution variants …")
    for ew, sa in combo_keys:
        t0 = time.time()
        result = _build_daily_distribution(
            db_path=db,
            lookback_days=FIXED["lookback_days"],
            min_hist_days=FIXED["min_hist_days"],
            sigma_floor=FIXED["sigma_floor"],
            spread_alpha=sa,
            ecmwf_weight=ew,
            return_errors=True,  # always build errors so we can toggle empirical
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
    ))
    print(f"\nRunning {len(all_combos)} backtest combos …\n")

    results = []
    for i, (ew, sa, sk, emp) in enumerate(all_combos, 1):
        dist_daily, errors_dict = dist_cache[(ew, sa)]
        t0 = time.time()
        out = _run_one(
            mkt_base=mkt_base,
            dist_daily=dist_daily,
            errors_dict=errors_dict,
            use_empirical=emp,
            confidence_shrink_k=sk,
        )
        elapsed = time.time() - t0
        label = f"ecmwf={ew:.2f} spread={sa:.2f} shrink_k={sk:.0f} emp={emp}"
        if out is None:
            print(f"  [{i}/{len(all_combos)}] {label}  → no trades  ({elapsed:.1f}s)")
            continue

        row = {
            "ecmwf_weight": ew,
            "spread_alpha": sa,
            "shrink_k": sk,
            "empirical": emp,
            **{k: v for k, v in out.items() if k != "city_roi"},
        }
        # Flatten city ROIs
        for city, croi in out.get("city_roi", {}).items():
            row[f"roi_{city}"] = croi
        results.append(row)

        print(
            f"  [{i}/{len(all_combos)}] {label}  "
            f"ROI={out['roi']:+.1%}  WR={out['win_rate']:.1%}  "
            f"N={out['trades']}  ({elapsed:.1f}s)"
        )

    if not results:
        print("\nNo combos produced trades.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("TOP 20 BY ROI")
    print("=" * 100)
    cols = ["ecmwf_weight", "spread_alpha", "shrink_k", "empirical",
            "trades", "roi", "win_rate", "pnl"]
    for extra in ["roi_yes", "wr_yes", "n_yes", "roi_no", "wr_no", "n_no"]:
        if extra in df.columns:
            cols.append(extra)
    print(df[cols].head(20).to_string(index=False))

    print("\n" + "=" * 100)
    print("BOTTOM 5 BY ROI")
    print("=" * 100)
    print(df[cols].tail(5).to_string(index=False))

    # Best combo city breakdown
    best = df.iloc[0]
    print(f"\n{'='*100}")
    print(f"BEST COMBO: ecmwf={best['ecmwf_weight']:.2f} spread={best['spread_alpha']:.2f} "
          f"shrink_k={best['shrink_k']:.0f} empirical={best['empirical']}")
    print(f"  ROI={best['roi']:+.1%}  WR={best['win_rate']:.1%}  Trades={int(best['trades'])}")
    city_cols = [c for c in df.columns if c.startswith("roi_") and c not in ("roi_yes", "roi_no")]
    if city_cols:
        print("  City ROIs:")
        for c in sorted(city_cols):
            city_name = c.replace("roi_", "")
            print(f"    {city_name}: {best[c]:+.1%}")

    # Save full results
    out_path = Path("logs/hyperparam_search_blend.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
