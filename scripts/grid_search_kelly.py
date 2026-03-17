"""
Grid search over MAX_BET_FRACTION and KELLY_FRACTION using existing backtest trades.

Trade selection (Thompson Sampling) is fixed. Only sizing changes, so we can
recompute PnL for each parameter combo without re-running the walk-forward model.

Metrics reported per combo:
  - total_pnl
  - roi
  - sharpe  (daily PnL mean / std, annualized)
  - max_drawdown_pct  (peak-to-trough as % of total wagered)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANKROLL = 500.0
TRADES_PATH = Path("logs/pnl_backtest_trades.csv")

MAX_BET_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
KELLY_FRACTIONS   = [0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00]


def kelly_size(edge: float, max_bet_frac: float, kelly_frac: float) -> float:
    if edge <= 0:
        return 0.0
    return min(BANKROLL * max_bet_frac, BANKROLL * edge * kelly_frac)


def compute_metrics(trades: pd.DataFrame, max_bet_frac: float, kelly_frac: float) -> dict:
    df = trades.copy()
    df["size"] = df["edge"].apply(lambda e: kelly_size(e, max_bet_frac, kelly_frac))

    # pnl = size * (outcome - mkt_p) for yes; size * ((1-outcome) - (1-mkt_p)) for no
    # outcome column is already adjusted (1 - outcome for "no" side), mkt_p is always the yes price
    # Reconstruct: pnl = size * (adjusted_outcome - adjusted_mkt_p)
    # From original code: yes -> size*(outcome - mkt_p), no -> size*((1-outcome)-(1-mkt_p))
    # Both reduce to: size * (adj_outcome - adj_mkt_p) where adj values are stored
    # We stored outcome (already flipped for "no") and mkt_p (always yes price)
    # For "no": pnl = size * ((1-raw_outcome) - (1-mkt_p)) = size * (mkt_p - raw_outcome)
    # But the stored "outcome" column is already 1-raw_outcome for "no" side
    # and stored mkt_p is always the yes price.
    # So: pnl = size * (stored_outcome - stored_mkt_p)  for YES
    #     pnl = size * (stored_outcome - (1 - stored_mkt_p))  for NO ... no that's wrong
    # Let's re-derive from stored columns: stored outcome = outcome if yes else 1-outcome
    # stored mkt_p = raw mkt_p (yes price always)
    # For YES: pnl = size * (outcome - mkt_p)
    # For NO:  pnl = size * ((1-outcome) - (1-mkt_p)) = size * (mkt_p - outcome)
    # So for NO: stored_outcome = 1-outcome, so pnl = size * (stored_outcome - (1-mkt_p))
    #                                                = size * (stored_outcome + mkt_p - 1)
    yes_mask = df["side"] == "yes"
    df.loc[yes_mask,  "pnl"] = df.loc[yes_mask,  "size"] * (df.loc[yes_mask,  "outcome"] - df.loc[yes_mask,  "mkt_p"])
    df.loc[~yes_mask, "pnl"] = df.loc[~yes_mask, "size"] * (df.loc[~yes_mask, "outcome"] - (1 - df.loc[~yes_mask, "mkt_p"]))

    total_pnl    = df["pnl"].sum()
    total_wagered = df["size"].sum()
    roi          = total_pnl / total_wagered if total_wagered > 0 else 0.0

    # Daily PnL for Sharpe
    daily = df.groupby("date")["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    # Max drawdown as % of peak cumulative PnL
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    drawdown = (cum - peak)
    max_dd = drawdown.min()  # most negative value
    max_dd_pct = (max_dd / peak.max() * 100) if peak.max() > 0 else 0.0

    return {
        "max_bet_frac": max_bet_frac,
        "kelly_frac":   kelly_frac,
        "total_pnl":    round(total_pnl, 2),
        "roi":          round(roi, 4),
        "sharpe":       round(sharpe, 3),
        "max_dd_pct":   round(max_dd_pct, 1),
        "total_wagered": round(total_wagered, 2),
    }


def main():
    if not TRADES_PATH.exists():
        print(f"Trades file not found: {TRADES_PATH}")
        print("Run scripts/run_pnl_backtest.py first.")
        return

    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"])
    print(f"Loaded {len(trades)} trades.\n")

    results = []
    for mbf in MAX_BET_FRACTIONS:
        for kf in KELLY_FRACTIONS:
            results.append(compute_metrics(trades, mbf, kf))

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print("=== GRID SEARCH RESULTS (sorted by Sharpe) ===")
    print(df.to_string(index=False))

    # Pivot tables
    print("\n--- Total PnL grid ---")
    pnl_pivot = df.pivot(index="max_bet_frac", columns="kelly_frac", values="total_pnl")
    print(pnl_pivot.to_string())

    print("\n--- Sharpe grid ---")
    sharpe_pivot = df.pivot(index="max_bet_frac", columns="kelly_frac", values="sharpe")
    print(sharpe_pivot.to_string())

    print("\n--- Max Drawdown % grid ---")
    dd_pivot = df.pivot(index="max_bet_frac", columns="kelly_frac", values="max_dd_pct")
    print(dd_pivot.to_string())

    best = df.iloc[0]
    print(f"\n=== BEST (by Sharpe) ===")
    print(f"  MAX_BET_FRACTION = {best['max_bet_frac']}")
    print(f"  KELLY_FRACTION   = {best['kelly_frac']}")
    print(f"  Total PnL:  ${best['total_pnl']:,.2f}")
    print(f"  ROI:        {best['roi']:.2%}")
    print(f"  Sharpe:     {best['sharpe']:.3f}")
    print(f"  Max DD:     {best['max_dd_pct']:.1f}%")

    # Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pivot, title, fmt in [
        (axes[0], pnl_pivot,    "Total PnL ($)",     "{:.0f}"),
        (axes[1], sharpe_pivot, "Sharpe Ratio",      "{:.2f}"),
        (axes[2], dd_pivot,     "Max Drawdown (%)",  "{:.1f}"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn" if "Drawdown" not in title else "RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0%}" for v in pivot.index])
        ax.set_xlabel("Kelly Fraction")
        ax.set_ylabel("Max Bet Fraction")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, fmt.format(pivot.values[i, j]),
                        ha="center", va="center", fontsize=7, color="black")

    plt.suptitle("Kelly / Cap Grid Search", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "logs/grid_search_kelly.png"
    plt.savefig(out, dpi=150)
    print(f"\nHeatmap saved to {out}")

    df.to_csv("logs/grid_search_kelly.csv", index=False)
    print("Full results saved to logs/grid_search_kelly.csv")


if __name__ == "__main__":
    main()
