"""
Phase 2: Build feature table and run walk-forward backtest.

Run from project root:
  python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.features import build_feature_table
from src.backtest import run_backtest


def main():
    hist_path = Path("data/historical_tmax.csv")
    if not hist_path.exists():
        print("historical_tmax.csv not found. Run scripts/download_history.py first.")
        return

    hist     = pd.read_csv(hist_path)
    gfs_path = Path("data/forecasts/openmeteo_forecast_history.csv")
    gfs_df   = pd.read_csv(gfs_path) if gfs_path.exists() else None
    indices_path = Path("data/climate_indices.csv")
    indices_df   = pd.read_csv(indices_path) if indices_path.exists() else None
    forecast_df  = pd.DataFrame(columns=["city", "forecast_high", "target_date"])

    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=indices_df)
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    run_backtest(df, threshold=72, min_train_rows=365)


if __name__ == "__main__":
    main()
