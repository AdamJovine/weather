"""
Walk-forward backtest for the temperature model.

Trains on all data before a given date, predicts that date, then moves forward.
Outputs per-day forecast accuracy and calibration metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model import TempDistributionModel, ProbabilityCalibrator
from src.pricing import fair_prob_geq, fair_prob_range
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX


def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "bias": round(bias, 3)}


def calibration_table(
    p_raw: np.ndarray,
    y_event: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin predicted probabilities and compare to empirical event rates.
    Perfect calibration: predicted ≈ empirical in every bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_raw, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low": round(bins[i], 2),
            "bin_high": round(bins[i + 1], 2),
            "n": int(mask.sum()),
            "mean_pred": round(float(p_raw[mask].mean()), 4),
            "empirical_rate": round(float(y_event[mask].mean()), 4),
        })

    return pd.DataFrame(rows)


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_rows: int = 365,
    threshold: int = 72,
) -> pd.DataFrame:
    """
    Walk-forward evaluation over every date in df that has y_tmax observed.

    For each test date:
      - Train on all prior rows with observed y_tmax
      - Predict mean and integer probs for the test row
      - Record point forecast error and event probability for P(tmax >= threshold)

    Returns a DataFrame with one row per evaluated date.
    """
    df = df.sort_values(["city", "date"]).copy()
    cities = df["city"].unique()

    results = []

    for city in cities:
        city_df = df[df["city"] == city].copy().reset_index(drop=True)

        for i in range(min_train_rows, len(city_df)):
            test_row = city_df.iloc[i]

            if pd.isna(test_row["y_tmax"]):
                continue  # no ground truth to evaluate against

            train_df = city_df.iloc[:i]

            model = TempDistributionModel()
            try:
                model.fit(train_df)
            except ValueError:
                continue  # not enough data

            from src.model import FEATURES
            test_df = city_df.iloc[[i]]
            if test_df[FEATURES].isnull().any(axis=1).iloc[0]:
                continue  # skip rows with missing features (early lags, etc.)

            pred_mean = model.predict_mean(test_df)
            probs_df = model.predict_integer_probs(test_df)
            prob_row = probs_df.iloc[0]

            p_geq = fair_prob_geq(prob_row, threshold)

            results.append({
                "city": city,
                "date": test_row["date"],
                "y_tmax": test_row["y_tmax"],
                "pred_mean": round(float(pred_mean[0]), 2),
                "sigma": round(model.sigma_, 2),
                f"p_geq_{threshold}": round(p_geq, 4),
                f"event_geq_{threshold}": int(test_row["y_tmax"] >= threshold),
            })

    return pd.DataFrame(results)


def run_backtest(
    df: pd.DataFrame,
    threshold: int = 72,
    min_train_rows: int = 365,
) -> None:
    """
    Run the full walk-forward backtest and print summary metrics.
    Also saves results to logs/backtest_results.csv.
    """
    print(f"Running walk-forward backtest (threshold={threshold}°F)...")
    results = walk_forward_backtest(df, min_train_rows=min_train_rows, threshold=threshold)

    if results.empty:
        print("No results — check that df has enough rows with observed y_tmax.")
        return

    results.to_csv("logs/backtest_results.csv", index=False)
    print(f"Saved {len(results)} rows to logs/backtest_results.csv")

    # Point forecast metrics
    metrics = evaluate_point_forecast(
        results["y_tmax"].values,
        results["pred_mean"].values,
    )
    print("\nPoint forecast metrics (all cities):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Calibration
    p_col = f"p_geq_{threshold}"
    y_col = f"event_geq_{threshold}"
    cal = calibration_table(results[p_col].values, results[y_col].values)
    print(f"\nCalibration table for P(tmax >= {threshold}°F):")
    print(cal.to_string(index=False))
