"""
src/data_contracts.py

Schema contracts, validation, and atomic CSV writes for weather data.

Typical usage
─────────────
Validate an in-memory DataFrame (tests / ad-hoc):

    result = SCHEMAS["gefs_spread"].validate(df)
    result.raise_if_errors()

Validate and write atomically (used by tests):

    from src.data_contracts import SCHEMAS, atomic_write_csv

    result = atomic_write_csv(df, tmp_path / "out.csv", SCHEMAS["historical_tmax"])

Note: production freshness checks use check_freshness_db() from src.db,
not check_all_freshness() here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


# ─── Validation result ───────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a schema validation or freshness check."""

    source: str
    ok: bool = True
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats:    dict       = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        stat_str = "  ".join(f"{k}={v}" for k, v in self.stats.items())
        header = f"[{status}] {self.source}"
        if stat_str:
            header += f"  ({stat_str})"
        lines = [header]
        for e in self.errors:
            lines.append(f"  ERROR  {e}")
        for w in self.warnings:
            lines.append(f"  WARN   {w}")
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        """Raise DataValidationError if any hard errors were found."""
        if not self.ok:
            raise DataValidationError(self.summary())


class DataValidationError(ValueError):
    pass


# ─── Column specification ─────────────────────────────────────────────────────

@dataclass
class ColumnSpec:
    """
    Constraints for one column.

    max_nan_fraction: fraction of rows allowed to be NaN (0.0 = none)
    min_val / max_val: physically sensible bounds — violations are warnings,
                       not errors, because real outliers do occur.
    """
    max_nan_fraction: float         = 0.0
    min_val:          Optional[float] = None
    max_val:          Optional[float] = None


# ─── Data schema ─────────────────────────────────────────────────────────────

@dataclass
class DataSchema:
    """
    Complete contract for one data source.

    name          – human-readable identifier
    columns       – required columns and their per-column constraints
    primary_key   – columns that must be unique together (dedup check)
    max_age_days  – how many days old the latest row can be before flagged stale
    date_col      – column name containing the date (for freshness + range stats)
    date_fn       – optional callable (df) -> pd.Series[Timestamp] when the date
                    must be derived (e.g., climate_indices has year + month, not date)
    """
    name:         str
    columns:      dict[str, ColumnSpec]
    primary_key:  list[str]
    max_age_days: float
    date_col:     Optional[str]      = None
    date_fn:      Optional[Callable] = None

    # ── Core validation ───────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all schema checks against df.  Returns a ValidationResult —
        does NOT raise; caller decides whether to raise_if_errors().
        """
        result = ValidationResult(source=self.name)
        result.stats["rows"] = len(df)

        # Required columns present
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            result.add_error(f"missing required columns: {missing}")
            return result   # can't check values without the columns

        if len(df) == 0:
            result.add_error("DataFrame is empty")
            return result

        n = len(df)

        # Per-column: NaN budget + value bounds
        for col, spec in self.columns.items():
            series = df[col]
            nan_count = int(series.isna().sum())
            nan_frac  = nan_count / n

            if nan_frac > spec.max_nan_fraction:
                msg = (
                    f"{col}: {nan_count} NaN ({nan_frac:.1%}) "
                    f"exceeds budget {spec.max_nan_fraction:.1%}"
                )
                if spec.max_nan_fraction == 0.0:
                    result.add_error(msg)
                else:
                    result.add_warning(msg)

            valid = series.dropna()
            if len(valid) and pd.api.types.is_numeric_dtype(valid):
                if spec.min_val is not None:
                    n_low = int((valid < spec.min_val).sum())
                    if n_low:
                        result.add_warning(
                            f"{col}: {n_low} values below minimum {spec.min_val}"
                        )
                if spec.max_val is not None:
                    n_high = int((valid > spec.max_val).sum())
                    if n_high:
                        result.add_warning(
                            f"{col}: {n_high} values above maximum {spec.max_val}"
                        )

        # Duplicate primary-key rows
        pk_cols = [c for c in self.primary_key if c in df.columns]
        if pk_cols:
            n_dupes = int(df.duplicated(subset=pk_cols).sum())
            if n_dupes:
                result.add_warning(
                    f"({', '.join(pk_cols)}): {n_dupes} duplicate rows"
                )

        # Date range stats
        dates = self._get_dates(df)
        if dates is not None and len(dates.dropna()):
            result.stats["date_range"] = (
                f"{dates.min().date()} → {dates.max().date()}"
            )

        return result

    # ── Freshness check ───────────────────────────────────────────────────────

    def check_freshness(self, df: pd.DataFrame) -> ValidationResult:
        """
        Return ok=False if the most recent date in df is more than
        max_age_days old relative to today.
        """
        result = ValidationResult(source=f"{self.name} (freshness)")

        dates = self._get_dates(df)
        if dates is None or len(dates.dropna()) == 0:
            result.add_warning("no parseable date column — cannot check freshness")
            return result

        latest    = dates.dropna().max().date()
        age_days  = (date_type.today() - latest).days
        result.stats["latest"] = str(latest)
        result.stats["age"]    = f"{age_days}d"

        if age_days > self.max_age_days:
            result.add_error(
                f"latest row is {age_days} day(s) old "
                f"(threshold: {self.max_age_days:.0f}d).  "
                f"Run the corresponding download script to refresh."
            )

        return result

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_dates(self, df: pd.DataFrame) -> Optional[pd.Series]:
        try:
            if self.date_fn is not None:
                return pd.to_datetime(self.date_fn(df), errors="coerce")
            if self.date_col and self.date_col in df.columns:
                return pd.to_datetime(df[self.date_col], errors="coerce")
        except Exception:
            pass
        return None


# ─── Schema definitions ───────────────────────────────────────────────────────

# Reusable specs
_TEMP   = ColumnSpec(max_nan_fraction=0.03, min_val=-60.0, max_val=140.0)
_SPREAD = ColumnSpec(max_nan_fraction=0.03, min_val=0.0,   max_val=30.0)
_IDX    = ColumnSpec(max_nan_fraction=0.10, min_val=-10.0, max_val=10.0)

SCHEMAS: dict[str, DataSchema] = {

    "historical_tmax": DataSchema(
        name="historical_tmax",
        columns={
            "date": ColumnSpec(),
            "city": ColumnSpec(),
            "tmax": _TEMP,
        },
        primary_key=["date", "city"],
        max_age_days=3,        # NOAA CDO lags ~1 day; allow up to 3
        date_col="date",
    ),

    "openmeteo_forecasts": DataSchema(
        name="openmeteo_forecasts",
        columns={
            "date":                ColumnSpec(),
            "city":                ColumnSpec(),
            "forecast_high_gfs":   _TEMP,
            "forecast_high_ecmwf": _TEMP,
            "ensemble_spread":     _SPREAD,
            "ecmwf_minus_gfs":     ColumnSpec(max_nan_fraction=0.03,
                                              min_val=-25.0, max_val=25.0),
            "precip_forecast":     ColumnSpec(max_nan_fraction=0.03,
                                              min_val=0.0,   max_val=500.0),
        "temp_850hpa":         ColumnSpec(max_nan_fraction=0.05,
                                              min_val=-100.0, max_val=120.0),
        "shortwave_radiation": ColumnSpec(max_nan_fraction=0.05,
                                              min_val=0.0,    max_val=40.0),
        "dew_point_max":       ColumnSpec(max_nan_fraction=0.05,
                                              min_val=-60.0,  max_val=100.0),
        },
        primary_key=["date", "city"],
        max_age_days=2,        # should have yesterday's GFS run at minimum
        date_col="date",
    ),

    "gefs_spread": DataSchema(
        name="gefs_spread",
        columns={
            "date":        ColumnSpec(),
            "city":        ColumnSpec(),
            "gefs_spread": _SPREAD,
        },
        primary_key=["date", "city"],
        max_age_days=2,
        date_col="date",
    ),

    "climate_indices": DataSchema(
        name="climate_indices",
        columns={
            "year":      ColumnSpec(),
            "month":     ColumnSpec(),
            "ao_index":  _IDX,
            "nao_index": _IDX,
            "oni":       _IDX,
            "pna_index": _IDX,
            "pdo_index": _IDX,
        },
        primary_key=["year", "month"],
        max_age_days=50,       # monthly release; ONI/PDO lag up to ~6 weeks
        date_col=None,
        date_fn=lambda df: pd.to_datetime(
            df["year"].astype(str)
            + "-"
            + df["month"].astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        ),
    ),

    "mjo_indices": DataSchema(
        name="mjo_indices",
        columns={
            "date":          ColumnSpec(),
            "mjo_amplitude": ColumnSpec(max_nan_fraction=0.02,
                                        min_val=0.0, max_val=10.0),
            "mjo_phase_sin": ColumnSpec(max_nan_fraction=0.02,
                                        min_val=-1.0, max_val=1.0),
            "mjo_phase_cos": ColumnSpec(max_nan_fraction=0.02,
                                        min_val=-1.0, max_val=1.0),
        },
        primary_key=["date"],
        max_age_days=5,        # BOM archives lag up to ~5 days
        date_col="date",
    ),
}


# ─── Atomic write ─────────────────────────────────────────────────────────────

def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
    schema: DataSchema,
    *,
    warn_only: bool = False,
) -> ValidationResult:
    """
    Validate df against schema, then write atomically to path.

    Steps
    ─────
    1. Validate df.  If hard errors exist and warn_only=False, raise
       DataValidationError — the existing file is untouched.
    2. Write to <path>.tmp on the same filesystem.
    3. Atomically rename .tmp → path (os.replace, POSIX-atomic).

    Returns the ValidationResult so callers can print warnings.
    """
    path = Path(path)
    result = schema.validate(df)

    if not result.ok and not warn_only:
        raise DataValidationError(
            f"Refusing to write {path} — validation failed:\n{result.summary()}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)   # atomic on POSIX and Windows (Python 3.3+)

    result.stats["written_to"] = str(path)
    return result


# ─── Freshness check (all sources) ───────────────────────────────────────────

def check_all_freshness(
    files: dict[str, Path],
    *,
    warn_only: bool = False,
) -> list[ValidationResult]:
    """
    Check the freshness of the provided data files.

    For each source:
      - Missing file        → error
      - File too old        → error
      - Unreadable file     → error

    If warn_only=False (default), raises DataValidationError listing every
    stale source.  Returns all results regardless so callers can log them.
    """

    results: list[ValidationResult] = []

    for source_name, path in files.items():
        schema = SCHEMAS.get(source_name)
        if schema is None:
            continue

        if not path.exists():
            r = ValidationResult(source=f"{source_name} (freshness)")
            r.add_error(f"file not found: {path}")
            results.append(r)
            continue

        try:
            df = pd.read_csv(path)
            r  = schema.check_freshness(df)
        except Exception as exc:
            r = ValidationResult(source=f"{source_name} (freshness)")
            r.add_error(f"could not read {path}: {exc}")

        results.append(r)

    if not warn_only:
        failures = [r for r in results if not r.ok]
        if failures:
            raise DataValidationError(
                "Stale data detected:\n"
                + "\n".join(r.summary() for r in failures)
            )

    return results
