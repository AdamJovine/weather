"""
Walk-forward calibration: generate (raw_p, actual) pairs for all cities,
dates, and temperature thresholds. Fits an isotonic regression calibrator
that maps raw model P(tmax >= T) → calibrated probability, and saves it to
logs/calibrator.pkl.

Usage:
    python scripts/fit_calibrator.py
    python scripts/fit_calibrator.py --start 2022-01-01 --end 2025-12-31
    python scripts/fit_calibrator.py --refit-every 7   # more refits = slower but better

Output:
    logs/calibrator.pkl       — fitted ProbabilityCalibrator
    logs/calibration_data.csv — raw (raw_p, actual) pairs for plotting
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.config import BacktestConfig
from backtesting.data_loader import DataLoader
from src.app_config import cfg as _cfg
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX
from src.model import FEATURES, ProbabilityCalibrator

import scripts.tune_hyperparams as _th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=_cfg.live.model_name)
    parser.add_argument("--start",       default="2022-01-01")
    parser.add_argument("--end",         default="2025-12-31")
    parser.add_argument("--refit-every", type=int, default=14,
                        help="Days between model refits (larger = faster)")
    parser.add_argument("--min-train",   type=int, default=365)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    best_path  = Path(f"logs/mega_tune/{model_name}/best.json")
    if not best_path.exists():
        print(f"ERROR: no best.json at {best_path}. Run scripts/mega_tune.py first.")
        sys.exit(1)
    best_params = json.load(open(best_path))["params"]
    lookback    = int(best_params["lookback"])

    print(f"Model: {model_name}  lookback={lookback}  refit_every={args.refit_every}d")
    print(f"Period: {args.start} → {args.end}\n")

    # Load feature data
    print("Loading feature data...")
    cfg = BacktestConfig(
        model_name=model_name,
        start_date=args.start,
        end_date=args.end,
    )
    dataset = DataLoader().load(cfg, skip_validation=args.no_validate)

    thresholds = list(range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1))
    all_raw_p  = []
    all_actual = []

    start_ts = pd.Timestamp(args.start)

    for city, city_df in dataset.city_frames.items():
        print(f"  {city}...", flush=True)
        last_refit  = None
        model       = None
        n_points    = 0

        for i in range(args.min_train, len(city_df)):
            row  = city_df.iloc[i]
            date = pd.Timestamp(row["date"])

            if date < start_ts:
                continue
            if pd.isna(row.get("y_tmax")):
                continue
            if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                continue

            # Refit model if due
            needs_refit = (
                model is None
                or last_refit is None
                or (date - last_refit).days >= args.refit_every
            )
            if needs_refit:
                train_df = city_df.iloc[max(0, i - lookback):i]
                model    = _th.make_model(model_name, best_params)
                model._lookback = lookback
                try:
                    model.fit(train_df)
                except Exception as e:
                    print(f"    fit failed at {date.date()}: {e}")
                    model = None
                    continue
                last_refit = date

            # Predict full distribution
            try:
                probs_df = model.predict_integer_probs(city_df.iloc[[i]])
            except Exception:
                continue

            prob_row  = probs_df.iloc[0]
            y_tmax    = float(row["y_tmax"])
            temp_keys = [f"temp_{k}" for k in thresholds]
            raw_probs = prob_row[temp_keys].values.astype(float)

            # P(tmax >= k) for each threshold k — decreasing array
            cumsum = np.cumsum(raw_probs[::-1])[::-1]

            for T, raw_p in zip(thresholds, cumsum):
                all_raw_p.append(float(raw_p))
                all_actual.append(1 if y_tmax >= T else 0)
            n_points += 1

        print(f"    → {n_points} dates, {n_points * len(thresholds):,} (p, outcome) pairs")

    total = len(all_raw_p)
    print(f"\nTotal calibration pairs: {total:,}")
    if total == 0:
        print("ERROR: no data collected — check date range and data availability.")
        sys.exit(1)

    # Save calibration data
    cal_df   = pd.DataFrame({"raw_p": all_raw_p, "actual": all_actual})
    cal_path = Path("logs/calibration_data.csv")
    cal_df.to_csv(cal_path, index=False)
    print(f"Calibration data → {cal_path}")

    # Fit and save calibrator
    calibrator = ProbabilityCalibrator()
    calibrator.fit(np.array(all_raw_p), np.array(all_actual))

    import pickle
    pkl_path = Path("logs/calibrator.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator       → {pkl_path}")

    # Quick sanity check
    test_ps  = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    cal_ps   = calibrator.predict(test_ps)
    print("\nCalibration check (raw → calibrated):")
    for r, c in zip(test_ps, cal_ps):
        print(f"  {r:.2f} → {c:.3f}")


if __name__ == "__main__":
    main()
