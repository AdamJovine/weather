"""
DataValidator: rigorous pre-backtest checks on every data source.

Each check emits a ValidationIssue at one of two severity levels:

  error   – would produce incorrect results; the run is aborted unless
             --no-validate is passed.
  warning – quality is degraded but the run can proceed; issues are printed.

Checks performed:
  Historical TMAX
    ✓ No duplicate (city, date) rows
    ✓ Temperature values in [-60, 140] °F
    ✓ No temporal gaps > MAX_GAP_DAYS per city
    ✓ NaN rate below threshold

  GFS / Forecast archive
    ✓ No duplicate (city, date) rows (warns; last value wins)
    ✓ Forecast temperature values in plausible range

  Climate indices (AO, NAO, ONI, PNA, PDO)
    ✓ Required columns present
    ✓ NaN rate below threshold per index

  GEFS spread
    ✓ No negative spread values
    ✓ No implausibly large values (> 50 °F)

  MJO indices
    ✓ Required columns present
    ✓ No negative amplitude values

  Feature table (post-merge)
    ✓ All model feature columns present
    ✓ NaN rate per feature below threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ── issue / report dataclasses ────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str   # "error" | "warning"
    check: str
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        tag = "ERROR  " if self.severity == "error" else "WARNING"
        return f"[{tag}] {self.check}: {self.message}"


class ValidationError(Exception):
    """Raised when one or more error-level validation issues are found."""
    pass


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, check: str, message: str, **details) -> None:
        self.issues.append(ValidationIssue(severity, check, message, dict(details)))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def raise_on_errors(self) -> None:
        if self.errors:
            lines = "\n  ".join(str(e) for e in self.errors)
            raise ValidationError(f"Data validation failed:\n  {lines}")

    def print_summary(self) -> None:
        if not self.issues:
            print("  Validation: all checks passed.")
            return
        for issue in self.issues:
            print(f"  {issue}")
        status = "PASSED" if self.passed else "FAILED"
        print(
            f"  Validation {status} — "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )


# ── validator ─────────────────────────────────────────────────────────────────

class DataValidator:
    """Validates all data sources before a backtest begins."""

    TMAX_MIN: float = -60.0
    TMAX_MAX: float = 140.0
    MAX_GAP_DAYS: int = 14
    MAX_NAN_RATE: float = 0.30     # warn if any feature column exceeds this
    MAX_OBS_NAN_RATE: float = 0.05  # stricter threshold for observed tmax

    def validate_all(
        self,
        hist_df: pd.DataFrame,
        gfs_df: Optional[pd.DataFrame],
        indices_df: Optional[pd.DataFrame],
        gefs_df: Optional[pd.DataFrame],
        mjo_df: Optional[pd.DataFrame],
    ) -> ValidationReport:
        report = ValidationReport()
        self._check_historical(hist_df, report)
        if gfs_df is not None:
            self._check_forecasts(gfs_df, report)
        if indices_df is not None:
            self._check_climate_indices(indices_df, report)
        if gefs_df is not None:
            self._check_gefs(gefs_df, report)
        if mjo_df is not None:
            self._check_mjo(mjo_df, report)
        return report

    def validate_feature_table(
        self,
        df: pd.DataFrame,
        features: list[str],
        report: ValidationReport,
    ) -> None:
        """Post-merge checks on the assembled feature table."""
        for col in features:
            if col not in df.columns:
                report.add("error", "feature_table_missing_col",
                           f"Feature column '{col}' is absent after merge",
                           column=col)
                continue
            nan_rate = float(df[col].isna().mean())
            if nan_rate > self.MAX_NAN_RATE:
                report.add(
                    "warning", "feature_nan_rate",
                    f"'{col}' has {nan_rate:.1%} NaN rate (threshold: {self.MAX_NAN_RATE:.0%})",
                    column=col, nan_rate=nan_rate,
                )

    # ── private checkers ─────────────────────────────────────────────────────

    def _check_historical(self, df: pd.DataFrame, report: ValidationReport) -> None:
        required = {"date", "city", "tmax"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            report.add("error", "historical_missing_cols",
                       f"Missing required columns: {missing_cols}")
            return  # further checks would fail

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Duplicates
        n_dupes = int(df.duplicated(subset=["city", "date"]).sum())
        if n_dupes:
            report.add("error", "historical_duplicates",
                       f"{n_dupes} duplicate (city, date) rows in historical_tmax.csv",
                       n=n_dupes)

        # Temperature range
        obs = df["tmax"].dropna()
        n_low = int((obs < self.TMAX_MIN).sum())
        n_high = int((obs > self.TMAX_MAX).sum())
        if n_low:
            report.add("error", "tmax_range_low",
                       f"{n_low} observations below {self.TMAX_MIN}°F", n=n_low)
        if n_high:
            report.add("error", "tmax_range_high",
                       f"{n_high} observations above {self.TMAX_MAX}°F", n=n_high)

        # Per-city temporal gaps
        for city, grp in df.groupby("city"):
            grp = grp.sort_values("date")
            gaps = grp["date"].diff().dt.days.dropna()
            big_gaps = gaps[gaps > self.MAX_GAP_DAYS]
            if len(big_gaps):
                worst = int(gaps.max())
                report.add(
                    "warning", "historical_gap",
                    f"{city}: {len(big_gaps)} gap(s) > {self.MAX_GAP_DAYS}d "
                    f"(worst: {worst}d)",
                    city=city, n_gaps=len(big_gaps), max_gap=worst,
                )

        # Overall NaN rate
        nan_rate = float(df["tmax"].isna().mean())
        if nan_rate > self.MAX_OBS_NAN_RATE:
            report.add("warning", "historical_nan",
                       f"historical_tmax has {nan_rate:.1%} NaN rate",
                       nan_rate=nan_rate)

    def _check_forecasts(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if not {"city", "date"}.issubset(df.columns):
            report.add("warning", "forecast_missing_cols",
                       "Forecast DataFrame is missing 'city' or 'date' column")
            return

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        n_dupes = int(df.duplicated(subset=["city", "date"]).sum())
        if n_dupes:
            report.add("warning", "forecast_duplicates",
                       f"{n_dupes} duplicate (city, date) forecast rows — last value wins",
                       n=n_dupes)

        for col in ("forecast_high", "gfs_high", "ecmwf_high"):
            if col not in df.columns:
                continue
            obs = df[col].dropna()
            n_bad = int(((obs < self.TMAX_MIN) | (obs > self.TMAX_MAX)).sum())
            if n_bad:
                report.add("warning", "forecast_range",
                           f"{col}: {n_bad} values outside [{self.TMAX_MIN}, {self.TMAX_MAX}]°F",
                           column=col, n_bad=n_bad)

    def _check_climate_indices(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if not {"year", "month"}.issubset(df.columns):
            report.add("warning", "climate_index_missing_cols",
                       "Climate indices missing 'year' or 'month' column")
            return

        for col in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index"):
            if col not in df.columns:
                report.add("warning", "climate_index_absent",
                           f"Expected climate index column '{col}' is absent")
                continue
            nan_rate = float(df[col].isna().mean())
            if nan_rate > self.MAX_NAN_RATE:
                report.add("warning", "climate_index_nan",
                           f"'{col}' has {nan_rate:.1%} NaN rate",
                           column=col, nan_rate=nan_rate)

    def _check_gefs(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if "gefs_spread" not in df.columns:
            report.add("warning", "gefs_missing_col",
                       "GEFS DataFrame is missing 'gefs_spread' column")
            return

        obs = df["gefs_spread"].dropna()
        n_neg = int((obs < 0).sum())
        if n_neg:
            report.add("error", "gefs_negative",
                       f"{n_neg} GEFS spread values are negative", n=n_neg)

        n_huge = int((obs > 50).sum())
        if n_huge:
            report.add("warning", "gefs_range_high",
                       f"{n_huge} GEFS spread values > 50°F (implausibly large)",
                       n=n_huge)

    def _check_mjo(self, df: pd.DataFrame, report: ValidationReport) -> None:
        required = {"date", "mjo_amplitude"}
        missing = required - set(df.columns)
        if missing:
            report.add("warning", "mjo_missing_cols",
                       f"MJO DataFrame missing columns: {missing}")
            return

        obs = df["mjo_amplitude"].dropna()
        n_neg = int((obs < 0).sum())
        if n_neg:
            report.add("error", "mjo_negative_amplitude",
                       f"{n_neg} MJO amplitude values are negative", n=n_neg)
