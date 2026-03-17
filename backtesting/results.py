"""
BacktestResults: metrics, reporting, and persistence for one or more runs.

Hierarchy:
  BacktestResults
    └─ list[RunResult]          one per run
         └─ pd.DataFrame        one row per trade

Metrics computed per RunResult:
  - total_pnl, total_wagered, roi
  - win_rate, avg_edge
  - daily Sharpe ratio (annualised at sqrt(252))
  - max peak-to-trough drawdown (dollar terms)
  - final bankroll, CAGR

When n_runs > 1, BacktestResults also computes across-run percentiles so you
can report e.g. "ROI 95% CI = [x%, y%]".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.config import BacktestConfig
from backtesting.validators import ValidationReport


@dataclass
class RunResult:
    """Outcome of a single backtest run."""
    run_id: int
    seed: int
    trades: pd.DataFrame
    config: BacktestConfig

    # ── per-run metrics ───────────────────────────────────────────────────────

    @property
    def total_pnl(self) -> float:
        return float(self.trades["pnl"].sum()) if not self.trades.empty else 0.0

    @property
    def total_wagered(self) -> float:
        return float(self.trades["size"].sum()) if not self.trades.empty else 0.0

    @property
    def roi(self) -> float:
        w = self.total_wagered
        return self.total_pnl / w if w > 0 else 0.0

    @property
    def win_rate(self) -> float:
        if self.trades.empty:
            return 0.0
        return float((self.trades["pnl"] > 0).mean())

    @property
    def avg_edge(self) -> float:
        if self.trades.empty:
            return 0.0
        return float(self.trades["edge"].mean())

    @property
    def sharpe(self) -> float:
        """
        Annualised daily Sharpe ratio.

        Daily P&L series is computed by grouping trades by settlement date.
        Multiplied by sqrt(252) to annualise (trading-days convention).
        """
        if self.trades.empty:
            return 0.0
        daily = self.trades.groupby("date")["pnl"].sum()
        if len(daily) < 2 or daily.std() == 0:
            return 0.0
        return float(daily.mean() / daily.std() * math.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """
        Maximum peak-to-trough drawdown in dollar terms.

        Computed on the running bankroll series (bankroll_after column),
        which already reflects compounding.
        """
        if self.trades.empty:
            return 0.0
        br = self.trades["bankroll_after"].values
        peak = np.maximum.accumulate(br)
        return float((peak - br).max())

    @property
    def final_bankroll(self) -> float:
        if self.trades.empty:
            return self.config.initial_bankroll
        return float(self.trades["bankroll_after"].iloc[-1])

    @property
    def cagr(self) -> float:
        """
        Compound annual growth rate, computed from start/end bankroll and
        the configured date range (not the actual number of trading days).
        """
        if self.trades.empty:
            return 0.0
        years = (
            pd.Timestamp(self.config.end_date) - pd.Timestamp(self.config.start_date)
        ).days / 365.25
        if years <= 0:
            return 0.0
        ratio = self.final_bankroll / self.config.initial_bankroll
        if ratio <= 0:
            return -1.0
        return float(ratio ** (1.0 / years) - 1.0)

    def summary_dict(self) -> dict:
        return {
            "run_id":        self.run_id,
            "seed":          self.seed,
            "n_trades":      len(self.trades),
            "total_pnl":     round(self.total_pnl, 2),
            "roi":           round(self.roi, 4),
            "win_rate":      round(self.win_rate, 4),
            "avg_edge":      round(self.avg_edge, 4),
            "sharpe":        round(self.sharpe, 3),
            "max_drawdown":  round(self.max_drawdown, 2),
            "final_bankroll": round(self.final_bankroll, 2),
            "cagr":          round(self.cagr, 4),
        }


@dataclass
class BacktestResults:
    """Aggregated results across all runs of a backtest."""
    runs: list[RunResult]
    config: BacktestConfig
    validation_report: ValidationReport

    # ── aggregation ───────────────────────────────────────────────────────────

    def summary_table(self) -> pd.DataFrame:
        """One row per run, plus mean and std rows when n_runs > 1."""
        rows = [r.summary_dict() for r in self.runs]
        df = pd.DataFrame(rows)
        if len(self.runs) > 1:
            numeric = df.select_dtypes(include="number")
            mean_row = numeric.mean().to_dict()
            mean_row.update({"run_id": "mean", "seed": "—"})
            std_row = numeric.std().to_dict()
            std_row.update({"run_id": "std", "seed": "—"})
            df = pd.concat(
                [df, pd.DataFrame([mean_row, std_row])],
                ignore_index=True,
            )
        return df

    def all_trades(self) -> pd.DataFrame:
        """All trades from all runs concatenated into a single DataFrame."""
        frames = [r.trades for r in self.runs if not r.trades.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ── reporting ─────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        cfg = self.config
        sep = "=" * 62

        print(f"\n{sep}")
        print("BACKTEST REPORT")
        print(sep)
        print(f"  Model          : {cfg.model_name}")
        print(f"  Period         : {cfg.start_date} → {cfg.end_date}")
        print(f"  Runs           : {cfg.n_runs}  (base seed: {cfg.seed})")
        print(f"  Sessions/day   : {cfg.sessions_per_day}")
        print(f"  Trade limits   : {cfg.max_daily_trades}/day, "
              f"{cfg.max_session_trades}/session")
        print(f"  Kelly          : {cfg.kelly_fraction}  |  "
              f"max_bet: {cfg.max_bet_fraction:.0%}  |  "
              f"fee: {cfg.fee_rate:.0%}")
        print(f"  Min edge       : {cfg.min_edge}  |  "
              f"lookback: {cfg.lookback} rows  |  "
              f"refit every: {cfg.refit_every}d")
        print(f"  Bankroll start : ${cfg.initial_bankroll:,.2f}")
        print(f"  Cities         : {', '.join(cfg.cities)}")
        print()

        table = self.summary_table()
        print(table.to_string(index=False))

        # Confidence intervals across runs
        if len(self.runs) > 1:
            rois = [r.roi for r in self.runs]
            sharpes = [r.sharpe for r in self.runs]
            print(
                f"\n  ROI    95% CI : "
                f"[{np.percentile(rois, 2.5):+.2%}, {np.percentile(rois, 97.5):+.2%}]"
            )
            print(
                f"  Sharpe 95% CI : "
                f"[{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]"
            )

        # Per-city breakdown
        all_df = self.all_trades()
        if not all_df.empty and "city" in all_df.columns:
            print(f"\n  PnL by city (all {cfg.n_runs} run(s) combined):")
            by_city = all_df.groupby("city").agg(
                n_trades=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                total_wagered=("size", "sum"),
                win_rate=("outcome", "mean"),
                avg_edge=("edge", "mean"),
            )
            by_city["roi"] = by_city["total_pnl"] / by_city["total_wagered"]
            print(by_city.round(3).to_string())

            print(f"\n  PnL by year (run 1):")
            run1 = self.runs[0].trades.copy()
            if not run1.empty:
                run1["year"] = pd.to_datetime(run1["date"]).dt.year
                by_year = run1.groupby("year").agg(
                    n_trades=("pnl", "count"),
                    total_pnl=("pnl", "sum"),
                    total_wagered=("size", "sum"),
                )
                by_year["roi"] = by_year["total_pnl"] / by_year["total_wagered"]
                print(by_year.round(3).to_string())

        print(sep)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Write trades.csv and summary.csv to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_df = self.all_trades()
        if not all_df.empty:
            trades_path = out / "trades.csv"
            all_df.to_csv(trades_path, index=False)
            print(f"  Trades   → {trades_path}  ({len(all_df):,} rows)")

        summary_path = out / "summary.csv"
        self.summary_table().to_csv(summary_path, index=False)
        print(f"  Summary  → {summary_path}")

    def plot(self, output_dir: str) -> None:
        """
        Four-panel chart:
          1. Bankroll over time (all runs)
          2. Cumulative PnL by city (run 1)
          3. Daily trade count vs. limits (run 1)
          4. Mean edge by hour-of-day UTC (run 1) — shows intraday alpha decay
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        run1_df = (
            self.runs[0].trades.copy()
            if self.runs and not self.runs[0].trades.empty
            else pd.DataFrame()
        )
        if not run1_df.empty:
            run1_df["date"] = pd.to_datetime(run1_df["date"])

        # ── panel 1: bankroll over time ───────────────────────────────────────
        ax = axes[0]
        multi = len(self.runs) > 1
        for result in self.runs:
            if result.trades.empty:
                continue
            df = result.trades.sort_values("date").copy()
            df["date"] = pd.to_datetime(df["date"])
            label = f"Run {result.run_id}" if multi else None
            alpha = 0.35 if multi else 1.0
            ax.plot(df["date"], df["bankroll_after"],
                    linewidth=1.2, alpha=alpha, label=label)
        ax.axhline(self.config.initial_bankroll, color="black",
                   linewidth=0.6, linestyle="--", label="Start")
        ax.set_title(
            f"Bankroll — {self.config.model_name}, {self.config.n_runs} run(s)"
        )
        ax.set_ylabel("Dollars ($)")
        ax.grid(True, alpha=0.3)
        if multi:
            ax.legend(fontsize=7, ncol=min(len(self.runs), 5))

        # ── panel 2: cumulative PnL by city (run 1) ───────────────────────────
        ax = axes[1]
        if not run1_df.empty:
            for city in sorted(run1_df["city"].unique()):
                ct = run1_df[run1_df["city"] == city].sort_values("date")
                ax.plot(ct["date"], ct["pnl"].cumsum(), label=city, linewidth=1.3)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title("Cumulative PnL by City (Run 1, real Kalshi prices)")
        ax.set_ylabel("Dollars ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── panel 3: daily trade count vs. limits ─────────────────────────────
        ax = axes[2]
        if not run1_df.empty:
            daily_count = run1_df.groupby("date").size()
            ax.bar(daily_count.index, daily_count.values,
                   width=1, alpha=0.6, color="steelblue", label="Trades/day")
            ax.axhline(self.config.max_daily_trades, color="red",
                       linewidth=0.9, linestyle="--",
                       label=f"max_daily = {self.config.max_daily_trades}")
            ax.axhline(self.config.max_session_trades, color="orange",
                       linewidth=0.9, linestyle=":",
                       label=f"max_session = {self.config.max_session_trades}")
        ax.set_title("Daily Trade Count (Run 1)")
        ax.set_ylabel("# Trades")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── panel 4: mean edge by hour of day UTC ─────────────────────────────
        ax = axes[3]
        if not run1_df.empty and "session_ts" in run1_df.columns:
            ts_nonzero = run1_df[run1_df["session_ts"] > 0].copy()
            if not ts_nonzero.empty:
                ts_nonzero["hour_utc"] = (
                    pd.to_datetime(ts_nonzero["session_ts"], unit="s", utc=True)
                    .dt.hour
                )
                by_hour = ts_nonzero.groupby("hour_utc")["edge"].agg(["mean", "std", "count"])
                ax.plot(by_hour.index, by_hour["mean"],
                        marker="o", linewidth=1.5, label="Mean edge")
                ax.fill_between(
                    by_hour.index,
                    by_hour["mean"] - by_hour["std"],
                    by_hour["mean"] + by_hour["std"],
                    alpha=0.2, label="±1 std",
                )
                ax.axhline(self.config.min_edge, color="red",
                           linewidth=0.9, linestyle="--",
                           label=f"min_edge = {self.config.min_edge}")
        ax.set_title("Mean Edge by Hour of Day UTC (Run 1) — intraday alpha decay?")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Mean Edge")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = out / "backtest.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Chart    → {path}")
