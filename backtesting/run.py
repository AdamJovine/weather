"""
Walk-forward PnL backtest against real Kalshi historical prices — CLI entry point.

Prerequisites:
    # Download real Kalshi price history first (done once, incremental after that)
    python scripts/download_kalshi_history.py

Usage:
    python -m backtesting.run [OPTIONS]

Quick examples:
    # 1 run, default settings
    python -m backtesting.run

    # 20 runs to measure outcome variance across random seeds
    python -m backtesting.run --runs 20 --start 2023-01-01 --end 2024-12-31

    # All hourly sessions, cap at 2 trades/session, 8 total/day
    python -m backtesting.run --sessions-per-day 0 --max-session-trades 2 --max-daily-trades 8

    # First snapshot only (one trade opportunity per city per day)
    python -m backtesting.run --sessions-per-day 1 --max-session-trades 1

    # Tight edge filter, slow model, weekly refits
    python -m backtesting.run --min-edge 0.10 --model NGBoost --refit-every 7

    # Single city
    python -m backtesting.run --cities "New York"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python -m backtesting.run` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.config import BacktestConfig, MODEL_REGISTRY
from backtesting.data_loader import DataLoader
from backtesting.engine import BacktestEngine
from backtesting.market_data import KalshiPriceStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtesting.run",
        description="Walk-forward PnL backtest for Kalshi weather temperature models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── run control ───────────────────────────────────────────────────────────
    g = p.add_argument_group("Run control")
    g.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help=(
            "Number of independent runs. Each run uses a different random seed "
            "(base_seed + run_index), so outcomes vary only by Thompson Sampling "
            "noise. Use --runs > 1 to get a distribution of results and "
            "confidence intervals on ROI / Sharpe. [default: 1]"
        ),
    )
    g.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Base random seed. Run k uses seed+k. [default: 42]",
    )

    # ── date range ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Date range")
    g.add_argument(
        "--start", type=str, default="2020-01-01", metavar="YYYY-MM-DD",
        help="First date to include in the test window. [default: 2020-01-01]",
    )
    g.add_argument(
        "--end", type=str, default="2024-12-31", metavar="YYYY-MM-DD",
        help="Last date to include in the test window. [default: 2024-12-31]",
    )

    # ── trading frequency & limits ────────────────────────────────────────────
    g = p.add_argument_group("Trading frequency and limits")
    g.add_argument(
        "--sessions-per-day", type=int, default=0, metavar="N",
        help=(
            "How many real hourly price snapshots to evaluate per day. "
            "0 = use all available candles (~12–20/day). "
            "N > 0 = subsample N evenly-spaced snapshots. "
            "Fewer sessions = less noise, more sessions = more fill opportunities. "
            "[default: 0 — all]"
        ),
    )
    g.add_argument(
        "--max-daily-trades", type=int, default=8, metavar="N",
        help=(
            "Hard cap: maximum total trades across ALL cities per calendar day. "
            "[default: 8]"
        ),
    )
    g.add_argument(
        "--max-session-trades", type=int, default=4, metavar="N",
        help=(
            "Hard cap: maximum trades within a single session across all cities. "
            "Must be <= --max-daily-trades. [default: 4]"
        ),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument(
        "--model", type=str, default="BayesianRidge",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Temperature distribution model. [default: BayesianRidge]",
    )
    g.add_argument(
        "--lookback", type=int, default=730, metavar="ROWS",
        help=(
            "Number of training rows per city (0 = all available history). "
            "[default: 730]"
        ),
    )
    g.add_argument(
        "--refit-every", type=int, default=1, metavar="DAYS",
        help=(
            "Refit the model every N calendar days. 1 = daily refits (slowest, "
            "freshest). 7 = weekly refits (~7x faster). [default: 1]"
        ),
    )

    # ── sizing ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Bet sizing")
    g.add_argument(
        "--bankroll", type=float, default=1000.0, metavar="$",
        help="Starting bankroll in dollars. [default: 1000]",
    )
    g.add_argument(
        "--kelly", type=float, default=0.33, metavar="F",
        help="Fractional Kelly multiplier (0–1). [default: 0.33]",
    )
    g.add_argument(
        "--max-bet-frac", type=float, default=0.05, metavar="F",
        help=(
            "Hard cap: maximum fraction of current bankroll per bet. "
            "[default: 0.05]"
        ),
    )
    g.add_argument(
        "--fee-rate", type=float, default=0.02, metavar="F",
        help="Kalshi fee as a fraction of winnings on winning trades. [default: 0.02]",
    )
    g.add_argument(
        "--max-bet-dollars", type=float, default=None, metavar="$",
        help="Hard dollar cap per trade (overrides fraction cap when tighter). [default: None]",
    )

    # ── edge & acquisition ────────────────────────────────────────────────────
    g = p.add_argument_group("Edge and acquisition")
    g.add_argument(
        "--min-edge", type=float, default=0.05, metavar="F",
        help=(
            "Minimum expected edge (model_p - market_p) before placing a trade. "
            "[default: 0.05]"
        ),
    )
    g.add_argument(
        "--thompson-draws", type=int, default=100, metavar="N",
        help=(
            "Number of posterior samples per prediction for Thompson Sampling. "
            "[default: 100]"
        ),
    )
    g.add_argument(
        "--spread-alpha", type=float, default=0.3, metavar="F",
        help=(
            "How much ensemble spread widens the synthetic market sigma: "
            "market_sigma = climo_sigma * (1 + spread_alpha * ensemble_spread). "
            "[default: 0.3]"
        ),
    )

    # ── scope ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Scope")
    g.add_argument(
        "--cities", type=str, nargs="+", metavar="CITY", default=None,
        help=(
            'Cities to trade. Default: all 4. '
            'Example: --cities "New York" Chicago'
        ),
    )
    g.add_argument(
        "--min-train-rows", type=int, default=365, metavar="N",
        help=(
            "Minimum rows of training data before the first trade is allowed. "
            "[default: 365]"
        ),
    )

    # ── multi-day trading ─────────────────────────────────────────────────────
    g = p.add_argument_group("Multi-day trading")
    g.add_argument(
        "--no-trade-tomorrow", action="store_true",
        help=(
            "Disable next-day settlement market evaluation. By default, each trade "
            "date evaluates both today's market (settlement=today) and tomorrow's "
            "market (settlement=tomorrow) — matching run_live.py's TARGET_DATE + "
            "TOMORROW_DATE logic. Use this flag to evaluate only same-day markets."
        ),
    )

    # ── output ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument(
        "--output-dir", type=str, default="logs/backtest", metavar="DIR",
        help="Directory for output CSVs and charts. [default: logs/backtest]",
    )
    g.add_argument(
        "--no-validate", action="store_true",
        help="Skip data quality validation (faster, not recommended).",
    )
    g.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation.",
    )
    g.add_argument(
        "--verbose", action="store_true",
        help="Print per-row model fit diagnostics and trade decisions.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = BacktestConfig(
        n_runs=args.runs,
        seed=args.seed,
        start_date=args.start,
        end_date=args.end,
        sessions_per_day=args.sessions_per_day,
        max_daily_trades=args.max_daily_trades,
        max_session_trades=args.max_session_trades,
        model_name=args.model,
        lookback=args.lookback,
        refit_every=args.refit_every,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_bet_fraction=args.max_bet_frac,
        max_bet_dollars=args.max_bet_dollars,
        fee_rate=args.fee_rate,
        min_edge=args.min_edge,
        n_thompson_draws=args.thompson_draws,
        spread_alpha=args.spread_alpha,
        cities=args.cities,
        min_train_rows=args.min_train_rows,
        trade_tomorrow=not args.no_trade_tomorrow,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # ── Load price history (once, reused across all runs) ─────────────────────
    price_store = KalshiPriceStore().load(verbose=args.verbose)
    print(price_store.coverage_summary())

    # ── Load and validate feature data ────────────────────────────────────────
    loader = DataLoader()
    dataset = loader.load(cfg, skip_validation=args.no_validate)

    # ── Run backtest ──────────────────────────────────────────────────────────
    engine = BacktestEngine(dataset, cfg, price_store)
    results = engine.run()

    results.print_report()
    results.save(cfg.output_dir)

    if not args.no_plot:
        results.plot(cfg.output_dir)


if __name__ == "__main__":
    main()
