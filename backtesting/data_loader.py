"""
DataLoader: loads every CSV source, validates integrity, builds the feature
table, and returns a BacktestDataset ready for the engine.

The loader is intentionally loud about what it finds (or doesn't find) so you
never have silent fallbacks producing subtly wrong results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from backtesting.validators import DataValidator, ValidationReport


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
    Loads all data sources from the project data/ directory, runs validation,
    builds the feature table, and returns a BacktestDataset.

    All data paths are relative to data_dir (default: "data").
    Missing optional sources degrade gracefully with printed warnings —
    no silent suppression.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def load(
        self,
        config: "BacktestConfig",  # forward ref to avoid circular import
        skip_validation: bool = False,
    ) -> BacktestDataset:
        """
        Full pipeline:
          1. Load raw CSVs
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
        hist_df = self._load_historical()
        gfs_df = self._load_gfs()
        indices_df = self._load_climate_indices()
        gefs_df = self._load_gefs()
        mjo_df = self._load_mjo()

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
        # NWS live forecasts are absent in backtest mode; pass empty placeholder
        forecast_df = pd.DataFrame(columns=["city", "forecast_high", "target_date"])
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

    # ── private loaders ───────────────────────────────────────────────────────

    def _load_historical(self) -> pd.DataFrame:
        path = self.data_dir / "historical_tmax.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Historical TMAX data not found at {path}. "
                "Run scripts/download_history.py first."
            )
        df = pd.read_csv(path)
        print(f"  historical_tmax         : {len(df):>7,} rows")
        return df

    def _load_gfs(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "forecasts" / "openmeteo_forecast_history.csv"
        if not path.exists():
            print("  openmeteo_forecast_history: NOT FOUND — using climatology as forecast")
            return None
        df = pd.read_csv(path)
        print(f"  openmeteo_forecast_history: {len(df):>7,} rows")
        return df

    def _load_climate_indices(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "climate_indices.csv"
        if not path.exists():
            print("  climate_indices         : NOT FOUND — AO/ONI/NAO features will be absent")
            return None
        df = pd.read_csv(path)
        print(f"  climate_indices         : {len(df):>7,} rows")
        return df

    def _load_gefs(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "forecasts" / "gefs_spread.csv"
        if not path.exists():
            print("  gefs_spread             : NOT FOUND — falling back to ensemble_spread")
            return None
        df = pd.read_csv(path)
        print(f"  gefs_spread             : {len(df):>7,} rows")
        return df

    def _load_mjo(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "mjo_indices.csv"
        if not path.exists():
            print("  mjo_indices             : NOT FOUND — MJO features will be zero")
            return None
        df = pd.read_csv(path)
        print(f"  mjo_indices             : {len(df):>7,} rows")
        return df

