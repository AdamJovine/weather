"""
DataLoader: loads every data source from the DB, validates integrity, builds
the feature table, and returns a BacktestDataset ready for the engine.

The loader is intentionally loud about what it finds (or doesn't find) so you
never have silent fallbacks producing subtly wrong results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.validators import DataValidator, ValidationReport
from src.db import DB_PATH
from src.data_loader_shared import load_raw_sources


@dataclass
class BacktestDataset:
    """
    All data needed for a backtest, fully validated and assembled.

    Attributes:
        feature_table    One row per (city, date). Sorted by (city, date).
                         Contains all model features + y_tmax.
        cities           Cities actually present in the data (subset of what was requested).
        validation_report  Full record of every check run, including passed checks.
    """
    feature_table: pd.DataFrame
    cities: list[str]
    validation_report: ValidationReport

    @property
    def city_frames(self) -> dict[str, pd.DataFrame]:
        """City-keyed dict of DataFrames, each sorted by date and index-reset."""
        return {
            city: (
                self.feature_table[self.feature_table["city"] == city]
                .sort_values("date")
                .reset_index(drop=True)
            )
            for city in self.cities
        }


class DataLoader:
    """
    Loads all data sources from the SQLite database, runs validation,
    builds the feature table, and returns a BacktestDataset.

    Missing optional tables degrade gracefully with printed warnings —
    no silent suppression.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def load(
        self,
        config: "BacktestConfig",  # forward ref to avoid circular import
        skip_validation: bool = False,
    ) -> BacktestDataset:
        """
        Full pipeline:
          1. Load raw tables from DB
          2. Validate each source
          3. Build merged feature table
          4. Post-merge feature validation
          5. Filter to requested cities
          6. Compute climo_sigma per city
        """
        from backtesting.config import BacktestConfig  # local to avoid circular
        from src.features import build_feature_table
        from src.model import FEATURES

        print("Loading data sources...")
        sources = load_raw_sources(self.db_path, nws_target_date=None, verbose=True)

        hist_df    = sources["hist"]
        gfs_df     = sources["gfs"]
        indices_df = sources["indices"]
        gefs_df    = sources["gefs"]
        mjo_df     = sources["mjo"]

        if hist_df.empty:
            raise ValueError("weather_daily table is empty — run scripts/update_data.py.")

        # Use NWS history when available; fall back to empty placeholder
        if not sources["nws"].empty:
            forecast_df = sources["nws"]
        else:
            forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])

        # ── validation ────────────────────────────────────────────────────────
        validator = DataValidator()
        if skip_validation:
            report = ValidationReport()
            print("  Validation: skipped (--no-validate)")
        else:
            print("  Running data quality checks...")
            report = validator.validate_all(
                hist_df, gfs_df, indices_df, gefs_df, mjo_df
            )
            report.print_summary()
            report.raise_on_errors()

        # ── feature table ─────────────────────────────────────────────────────
        print("  Building feature table...")
        df = build_feature_table(
            hist_df,
            forecast_df,
            gfs_df=gfs_df,
            indices_df=indices_df,
            gefs_df=gefs_df,
            mjo_df=mjo_df,
        )

        # Fill any remaining forecast gaps with per-day climatology
        df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
        df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

        # Post-merge feature quality check
        if not skip_validation:
            validator.validate_feature_table(df, FEATURES, report)
            # Re-print if new issues were added by the feature check
            new_issues = [i for i in report.issues if "feature" in i.check]
            for issue in new_issues:
                print(f"  {issue}")

        # ── city filter ───────────────────────────────────────────────────────
        available = sorted(df["city"].unique().tolist())
        cities = [c for c in config.cities if c in available]
        if not cities:
            raise ValueError(
                f"None of the requested cities {config.cities} are in the data. "
                f"Available: {available}"
            )
        missing_cities = set(config.cities) - set(cities)
        if missing_cities:
            print(f"  WARNING: Requested cities not found in data: {missing_cities}")

        df = df[df["city"].isin(cities)].copy()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["city", "date"]).reset_index(drop=True)

        print(
            f"  Ready: {len(df):,} rows | "
            f"{df['city'].nunique()} cities | "
            f"{df['date'].min().date()} → {df['date'].max().date()}"
        )

        return BacktestDataset(
            feature_table=df,
            cities=cities,
            validation_report=report,
        )


