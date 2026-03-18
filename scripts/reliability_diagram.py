"""
Plot a reliability diagram from calibration_data.csv.

Buckets predicted P(tmax >= T) values into bins and compares the mean
predicted probability to the observed frequency. A perfectly calibrated
model lies on the diagonal.

Usage:
    python scripts/reliability_diagram.py
    python scripts/reliability_diagram.py --bins 20
    python scripts/reliability_diagram.py --by-range "0.0,0.2"  # zoom in on a range

Output:
    logs/reliability_diagram.png  (if matplotlib is installed)
    Printed table always shown.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins",    type=int,   default=10)
    parser.add_argument("--data",    default="logs/calibration_data.csv")
    parser.add_argument("--out",     default="logs/reliability_diagram.png")
    args = parser.parse_args()

    cal_path = Path(args.data)
    if not cal_path.exists():
        print(f"No calibration data at {cal_path}")
        print("Run: python scripts/fit_calibrator.py")
        sys.exit(1)

    df = pd.read_csv(cal_path)
    print(f"Loaded {len(df):,} calibration points\n")

    bins = np.linspace(0, 1, args.bins + 1)
    df["bin"] = pd.cut(df["raw_p"], bins=bins, labels=False, include_lowest=True)

    grouped = (
        df.groupby("bin", observed=True)
        .agg(mean_pred=("raw_p", "mean"), mean_actual=("actual", "mean"), count=("actual", "count"))
        .reset_index()
    )

    print(f"{'Range':<12} {'Pred':>8} {'Actual':>8} {'Count':>8}  {'Error':>7}")
    print("-" * 52)
    errors = []
    for _, row in grouped.iterrows():
        b = int(row["bin"])
        lo, hi = bins[b], bins[b + 1]
        err = row["mean_pred"] - row["mean_actual"]
        errors.append(abs(err))
        flag = " <-- OVERCONFIDENT" if err > 0.05 else (" <-- UNDERCONFIDENT" if err < -0.05 else "")
        print(
            f"{lo:.0%}-{hi:.0%}    {row['mean_pred']:>8.3f}  {row['mean_actual']:>8.3f}"
            f"  {row['count']:>8,.0f}  {err:>+7.3f}{flag}"
        )

    mce = float(np.mean(errors)) if errors else float("nan")
    wmce = float(
        np.average(errors, weights=grouped["count"].values)
    ) if errors else float("nan")
    print(f"\nMean calibration error (MCE):          {mce:.4f}")
    print(f"Weighted mean calibration error (WMCE): {wmce:.4f}")
    print("(0 = perfect, >0.05 = significant miscalibration)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Left: reliability diagram
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        sizes = grouped["count"] / grouped["count"].max() * 400
        ax.scatter(grouped["mean_pred"], grouped["mean_actual"], s=sizes,
                   alpha=0.85, zorder=3, label="Model (size ∝ count)")
        ax.plot(grouped["mean_pred"], grouped["mean_actual"], alpha=0.4)
        ax.fill_between([0, 1], [0, 1], grouped["mean_actual"].reindex(
            range(args.bins), fill_value=np.nan
        ).values if False else [0, 1], alpha=0)  # just for structure
        ax.set_xlabel("Mean predicted probability", fontsize=12)
        ax.set_ylabel("Observed frequency", fontsize=12)
        ax.set_title(f"Reliability Diagram\nMCE={mce:.4f}  WMCE={wmce:.4f}", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        # Right: calibration error per bin
        ax2 = axes[1]
        bin_centers = (grouped["mean_pred"].values)
        bar_errors  = grouped["mean_pred"].values - grouped["mean_actual"].values
        colors = ["#d62728" if e > 0 else "#2ca02c" for e in bar_errors]
        ax2.bar(range(len(bar_errors)), bar_errors, color=colors, alpha=0.8)
        ax2.axhline(0, color="black", lw=1)
        ax2.axhline(0.05,  color="red",  lw=1, ls="--", alpha=0.5, label="±5% threshold")
        ax2.axhline(-0.05, color="red",  lw=1, ls="--", alpha=0.5)
        ax2.set_xticks(range(len(bar_errors)))
        ax2.set_xticklabels(
            [f"{bins[int(r['bin'])]:.0%}" for _, r in grouped.iterrows()],
            rotation=45, ha="right",
        )
        ax2.set_xlabel("Predicted probability bin", fontsize=12)
        ax2.set_ylabel("Pred − Actual (overconfidence > 0)", fontsize=12)
        ax2.set_title("Calibration Error per Bin\n(red = overconfident, green = underconfident)", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = Path(args.out)
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {out_path}")

    except ImportError:
        print("\nmatplotlib not installed — skipping plot. pip install matplotlib")


if __name__ == "__main__":
    main()
