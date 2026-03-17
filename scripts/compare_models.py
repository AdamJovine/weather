"""
Compare all temperature models head-to-head using the backtesting framework.

Each model is run with default hyperparameters over the same date window,
then ranked by ROI, Sharpe, and win-rate.  Results are written to
logs/model_comparison.csv and printed as a sorted table.

Prerequisites:
    python scripts/download_kalshi_history.py   # real price history

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --start 2025-12-01 --end 2025-12-31
    python scripts/compare_models.py --models BayesianRidge NGBoost ExtraTrees
    python scripts/compare_models.py --refit-every 1 --sessions-per-day 0  # high-fidelity
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

warnings.filterwarnings("ignore")

from backtesting.config import BacktestConfig, MODEL_REGISTRY
from backtesting.data_loader import DataLoader
from backtesting.engine import BacktestEngine
from backtesting.market_data import KalshiPriceStore


def run_model(
    model_name: str,
    dataset,
    price_store: KalshiPriceStore,
    cfg: BacktestConfig,
) -> dict:
    """Run one model and return its summary metrics."""
    model_cfg = BacktestConfig(
        model_name=model_name,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        sessions_per_day=cfg.sessions_per_day,
        max_daily_trades=cfg.max_daily_trades,
        max_session_trades=cfg.max_session_trades,
        refit_every=cfg.refit_every,
        min_train_rows=cfg.min_train_rows,
        lookback=cfg.lookback,
        initial_bankroll=cfg.initial_bankroll,
        kelly_fraction=cfg.kelly_fraction,
        max_bet_fraction=cfg.max_bet_fraction,
        fee_rate=cfg.fee_rate,
        min_edge=cfg.min_edge,
        cities=cfg.cities,
        output_dir=cfg.output_dir,
    )

    engine = BacktestEngine(dataset, model_cfg, price_store)
    results = engine.run()
    r = results.runs[0]
    return {
        "model":          model_name,
        "n_trades":       r.total_pnl and len(r.trades),  # 0 if empty
        "total_pnl":      round(r.total_pnl, 2),
        "roi":            round(r.roi, 4),
        "win_rate":       round(r.win_rate, 4),
        "avg_edge":       round(r.avg_edge, 4),
        "sharpe":         round(r.sharpe, 3),
        "max_drawdown":   round(r.max_drawdown, 2),
        "final_bankroll": round(r.final_bankroll, 2),
        "cagr":           round(r.cagr, 4),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/compare_models.py",
        description="Head-to-head comparison of all temperature models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start", default="2025-12-01", metavar="YYYY-MM-DD")
    p.add_argument("--end",   default="2025-12-31", metavar="YYYY-MM-DD")
    p.add_argument(
        "--models", nargs="+", default=None,
        choices=sorted(MODEL_REGISTRY.keys()), metavar="MODEL",
        help="Subset of models to run. Default: all.",
    )
    p.add_argument("--sessions-per-day", type=int, default=1, metavar="N",
                   help="0 = all candles. [default: 1]")
    p.add_argument("--refit-every", type=int, default=7, metavar="DAYS",
                   help="[default: 7]")
    p.add_argument("--lookback", type=int, default=730, metavar="ROWS")
    p.add_argument("--min-train-rows", type=int, default=365, metavar="N")
    p.add_argument("--max-daily-trades", type=int, default=8)
    p.add_argument("--max-session-trades", type=int, default=4)
    p.add_argument("--bankroll", type=float, default=1000.0, metavar="$")
    p.add_argument("--kelly", type=float, default=0.33, metavar="F")
    p.add_argument("--max-bet-frac", type=float, default=0.05, metavar="F")
    p.add_argument("--fee-rate", type=float, default=0.02, metavar="F")
    p.add_argument("--min-edge", type=float, default=0.05, metavar="F")
    p.add_argument("--cities", nargs="+", default=None, metavar="CITY")
    p.add_argument("--output-dir", default="logs", metavar="DIR")
    p.add_argument("--no-validate", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    models_to_run = args.models or sorted(MODEL_REGISTRY.keys())

    # Shared config template (model_name overridden per run)
    cfg = BacktestConfig(
        model_name="BayesianRidge",  # placeholder; overridden per model
        start_date=args.start,
        end_date=args.end,
        sessions_per_day=args.sessions_per_day,
        max_daily_trades=args.max_daily_trades,
        max_session_trades=args.max_session_trades,
        refit_every=args.refit_every,
        min_train_rows=args.min_train_rows,
        lookback=args.lookback,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_bet_fraction=args.max_bet_frac,
        fee_rate=args.fee_rate,
        min_edge=args.min_edge,
        cities=args.cities,
        output_dir=args.output_dir,
    )

    print("Loading price history...")
    price_store = KalshiPriceStore(data_dir="data").load()
    print(price_store.coverage_summary())

    print("Loading feature data...")
    loader = DataLoader(data_dir="data")
    dataset = loader.load(cfg, skip_validation=args.no_validate)
    print()

    sep = "=" * 62
    print(sep)
    print(f"MODEL COMPARISON  {args.start} → {args.end}")
    print(f"Sessions/day: {args.sessions_per_day}  |  "
          f"refit every: {args.refit_every}d  |  "
          f"fee: {args.fee_rate:.0%}  |  "
          f"min_edge: {args.min_edge}")
    print(sep)

    rows = []
    for model_name in models_to_run:
        print(f"  Running {model_name}...", end="", flush=True)
        t0 = time.time()
        try:
            row = run_model(model_name, dataset, price_store, cfg)
            elapsed = time.time() - t0
            print(f"  {elapsed:.1f}s  →  ROI={row['roi']:+.2%}  "
                  f"Sharpe={row['sharpe']:.3f}  trades={row['n_trades']}")
            rows.append(row)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {exc}")

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)
    df.index += 1  # rank from 1

    print(f"\n{sep}")
    print("RESULTS  (sorted by ROI, after 2% Kalshi fee)")
    print(sep)
    print(df.to_string())
    print(sep)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_comparison.csv"
    df.to_csv(path, index_label="rank")
    print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()
