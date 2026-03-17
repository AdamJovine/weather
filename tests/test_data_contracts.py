"""
tests/test_data_contracts.py

Tests for src/data_contracts.py — schema validation, atomic writes,
and staleness monitoring.

Run from project root:
    pytest tests/test_data_contracts.py -v
"""

import sys
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_contracts import (
    ColumnSpec,
    DataSchema,
    DataValidationError,
    ValidationResult,
    SCHEMAS,
    atomic_write_csv,
    check_all_freshness,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fresh_tmax(n: int = 10, age_days: int = 0) -> pd.DataFrame:
    """Minimal valid historical_tmax DataFrame, latest row `age_days` old."""
    latest = date.today() - timedelta(days=age_days)
    dates  = pd.date_range(end=latest, periods=n).date
    return pd.DataFrame({
        "date": dates,
        "city": "TestCity",
        "tmax": np.full(n, 72.0),
    })


def _fresh_gefs(n: int = 10, age_days: int = 0) -> pd.DataFrame:
    latest = date.today() - timedelta(days=age_days)
    dates  = pd.date_range(end=latest, periods=n).date
    return pd.DataFrame({
        "date":        dates,
        "city":        "TestCity",
        "gefs_spread": np.full(n, 2.5),
    })


def _fresh_openmeteo(n: int = 10, age_days: int = 0) -> pd.DataFrame:
    latest = date.today() - timedelta(days=age_days)
    dates  = pd.date_range(end=latest, periods=n).date
    return pd.DataFrame({
        "date":                dates,
        "city":                "TestCity",
        "forecast_high_gfs":   np.full(n, 72.0),
        "forecast_high_ecmwf": np.full(n, 71.0),
        "ensemble_spread":     np.full(n, 2.0),
        "ecmwf_minus_gfs":     np.full(n, -1.0),
        "precip_forecast":     np.full(n, 0.0),
    })


def _fresh_climate_indices(n_months: int = 12, lag_months: int = 0) -> pd.DataFrame:
    """Monthly climate index rows, latest entry `lag_months` months before this month."""
    today = date.today()
    rows = []
    for i in range(n_months):
        m = today.month - lag_months - i
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        rows.append({"year": y, "month": m, "ao_index": 0.5, "nao_index": -0.3,
                     "oni": 0.1, "pna_index": 0.2, "pdo_index": -0.1})
    return pd.DataFrame(rows)


def _fresh_mjo(n: int = 10, age_days: int = 0) -> pd.DataFrame:
    latest = date.today() - timedelta(days=age_days)
    dates  = pd.date_range(end=latest, periods=n)
    return pd.DataFrame({
        "date":          dates,
        "mjo_amplitude": np.full(n, 1.2),
        "mjo_phase_sin": np.full(n, 0.5),
        "mjo_phase_cos": np.full(n, 0.866),
    })


# ─── 1. ValidationResult ─────────────────────────────────────────────────────

class TestValidationResult:

    def test_ok_by_default(self):
        r = ValidationResult(source="test")
        assert r.ok is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_sets_ok_false(self):
        r = ValidationResult(source="test")
        r.add_error("something broke")
        assert r.ok is False
        assert "something broke" in r.errors

    def test_add_warning_does_not_change_ok(self):
        r = ValidationResult(source="test")
        r.add_warning("minor issue")
        assert r.ok is True
        assert "minor issue" in r.warnings

    def test_raise_if_errors_raises_on_failure(self):
        r = ValidationResult(source="test")
        r.add_error("bad data")
        with pytest.raises(DataValidationError, match="bad data"):
            r.raise_if_errors()

    def test_raise_if_errors_does_not_raise_on_pass(self):
        r = ValidationResult(source="test")
        r.add_warning("minor")
        r.raise_if_errors()  # should not raise

    def test_summary_contains_source_and_status(self):
        r = ValidationResult(source="my_source")
        assert "my_source" in r.summary()
        assert "PASS" in r.summary()
        r.add_error("kaboom")
        assert "FAIL" in r.summary()
        assert "kaboom" in r.summary()


# ─── 2. DataSchema.validate ───────────────────────────────────────────────────

class TestValidate:

    def test_valid_dataframe_passes(self):
        df = _fresh_tmax()
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok
        assert result.errors == []

    def test_missing_required_column_is_error(self):
        df = _fresh_tmax().drop(columns=["tmax"])
        result = SCHEMAS["historical_tmax"].validate(df)
        assert not result.ok
        assert any("tmax" in e for e in result.errors)

    def test_empty_dataframe_is_error(self):
        df = pd.DataFrame({"date": [], "city": [], "tmax": []})
        result = SCHEMAS["historical_tmax"].validate(df)
        assert not result.ok

    def test_zero_nan_budget_column_with_nan_is_error(self):
        df = _fresh_tmax()
        df.loc[0, "tmax"] = np.nan
        # tmax allows 3% NaN, so 1/10 = 10% → should be a warning, not error
        # But "date" and "city" have 0% budget → test with those
        df.loc[0, "date"] = None
        result = SCHEMAS["historical_tmax"].validate(df)
        # date column has 0.0 max_nan_fraction → error
        assert not result.ok
        assert any("date" in e for e in result.errors)

    def test_nan_within_budget_is_warning_not_error(self):
        """tmax allows 3% NaN; 1 NaN in 100 rows = 1% — within budget."""
        n = 100
        df = _fresh_tmax(n=n)
        df.loc[0, "tmax"] = np.nan   # 1% NaN, budget is 3%
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok    # no error
        assert result.warnings == []   # within budget → no warning either

    def test_nan_exceeding_budget_is_warning(self):
        """tmax allows 3% NaN; 5 NaN in 10 rows = 50% — exceeds budget → warning."""
        df = _fresh_tmax(n=10)
        df.loc[:4, "tmax"] = np.nan   # 50% NaN
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok    # still ok (not an error, just a warning)
        assert any("tmax" in w for w in result.warnings)

    def test_value_below_min_is_warning(self):
        df = _fresh_tmax()
        df.loc[0, "tmax"] = -100.0   # way below -60°F minimum
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok   # warnings, not errors
        assert any("below minimum" in w for w in result.warnings)

    def test_value_above_max_is_warning(self):
        df = _fresh_tmax()
        df.loc[0, "tmax"] = 999.0    # way above 140°F maximum
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok
        assert any("above maximum" in w for w in result.warnings)

    def test_duplicate_primary_key_is_warning(self):
        df = _fresh_tmax()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate first row
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.ok   # duplicates are a warning
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_stats_include_row_count_and_date_range(self):
        df = _fresh_tmax(n=20)
        result = SCHEMAS["historical_tmax"].validate(df)
        assert result.stats["rows"] == 20
        assert "date_range" in result.stats

    def test_all_schemas_pass_on_valid_data(self):
        """Each registered schema must pass on freshly generated valid data."""
        valid_dfs = {
            "historical_tmax":     _fresh_tmax(),
            "openmeteo_forecasts": _fresh_openmeteo(),
            "gefs_spread":         _fresh_gefs(),
            "climate_indices":     _fresh_climate_indices(),
            "mjo_indices":         _fresh_mjo(),
        }
        for name, df in valid_dfs.items():
            result = SCHEMAS[name].validate(df)
            assert result.ok, (
                f"Schema '{name}' failed on valid data:\n{result.summary()}"
            )


# ─── 3. DataSchema.check_freshness ───────────────────────────────────────────

class TestCheckFreshness:

    def test_fresh_data_passes(self):
        df = _fresh_tmax(age_days=1)
        result = SCHEMAS["historical_tmax"].check_freshness(df)
        assert result.ok

    def test_stale_data_is_error(self):
        """Data that is older than max_age_days must fail freshness check."""
        schema = SCHEMAS["historical_tmax"]  # max_age_days=3
        df = _fresh_tmax(age_days=10)        # 10 days old → stale
        result = schema.check_freshness(df)
        assert not result.ok
        assert any("day" in e for e in result.errors)

    def test_boundary_exactly_at_max_age_passes(self):
        schema = SCHEMAS["historical_tmax"]
        df = _fresh_tmax(age_days=int(schema.max_age_days))
        result = schema.check_freshness(df)
        assert result.ok

    def test_boundary_one_day_over_fails(self):
        schema = SCHEMAS["historical_tmax"]
        df = _fresh_tmax(age_days=int(schema.max_age_days) + 1)
        result = schema.check_freshness(df)
        assert not result.ok

    def test_gefs_spread_freshness(self):
        df = _fresh_gefs(age_days=1)
        result = SCHEMAS["gefs_spread"].check_freshness(df)
        assert result.ok

    def test_climate_indices_derived_date(self):
        """climate_indices uses year+month → date_fn path."""
        df = _fresh_climate_indices(lag_months=1)
        result = SCHEMAS["climate_indices"].check_freshness(df)
        assert result.ok

    def test_climate_indices_stale(self):
        """climate_indices older than max_age_days should fail."""
        df = _fresh_climate_indices(lag_months=3)   # ~90 days old
        result = SCHEMAS["climate_indices"].check_freshness(df)
        assert not result.ok

    def test_missing_date_column_gives_warning_not_error(self):
        """If the date column is absent, issue a warning but don't hard-fail."""
        schema = DataSchema(
            name="no_date",
            columns={"val": ColumnSpec()},
            primary_key=[],
            max_age_days=1,
            date_col="missing_col",
        )
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = schema.check_freshness(df)
        assert result.ok   # warning, not error
        assert result.warnings


# ─── 4. atomic_write_csv ─────────────────────────────────────────────────────

class TestAtomicWriteCsv:

    def test_writes_file_on_valid_data(self, tmp_path):
        df   = _fresh_tmax()
        path = tmp_path / "tmax.csv"
        atomic_write_csv(df, path, SCHEMAS["historical_tmax"])
        assert path.exists()
        reloaded = pd.read_csv(path)
        assert len(reloaded) == len(df)

    def test_raises_on_invalid_data(self, tmp_path):
        df   = _fresh_tmax().drop(columns=["tmax"])   # missing required column
        path = tmp_path / "tmax.csv"
        with pytest.raises(DataValidationError, match="missing required columns"):
            atomic_write_csv(df, path, SCHEMAS["historical_tmax"])

    def test_does_not_overwrite_existing_file_on_error(self, tmp_path):
        """Existing file must be untouched when validation fails."""
        path = tmp_path / "tmax.csv"
        original = _fresh_tmax(n=5)
        original.to_csv(path, index=False)

        bad_df = _fresh_tmax().drop(columns=["tmax"])
        with pytest.raises(DataValidationError):
            atomic_write_csv(bad_df, path, SCHEMAS["historical_tmax"])

        # Original file unchanged
        reloaded = pd.read_csv(path)
        assert len(reloaded) == 5

    def test_no_tmp_file_left_on_success(self, tmp_path):
        df   = _fresh_tmax()
        path = tmp_path / "tmax.csv"
        atomic_write_csv(df, path, SCHEMAS["historical_tmax"])
        assert not path.with_suffix(".tmp").exists()

    def test_no_tmp_file_left_on_failure(self, tmp_path):
        """Even when validation fails, no .tmp file should be left behind."""
        path = tmp_path / "tmax.csv"
        bad_df = pd.DataFrame({"date": [], "city": [], "tmax": []})
        with pytest.raises(DataValidationError):
            atomic_write_csv(bad_df, path, SCHEMAS["historical_tmax"])
        assert not path.with_suffix(".tmp").exists()

    def test_warn_only_writes_despite_errors(self, tmp_path):
        """warn_only=True should write the file even when validation has errors."""
        df   = pd.DataFrame({"date": [], "city": [], "tmax": []})   # empty → error
        path = tmp_path / "tmax.csv"
        result = atomic_write_csv(df, path, SCHEMAS["historical_tmax"], warn_only=True)
        assert path.exists()
        assert not result.ok   # still reports the error

    def test_creates_parent_directories(self, tmp_path):
        df   = _fresh_tmax()
        path = tmp_path / "deep" / "nested" / "tmax.csv"
        atomic_write_csv(df, path, SCHEMAS["historical_tmax"])
        assert path.exists()

    def test_result_stats_include_written_to(self, tmp_path):
        df   = _fresh_tmax()
        path = tmp_path / "tmax.csv"
        result = atomic_write_csv(df, path, SCHEMAS["historical_tmax"])
        assert "written_to" in result.stats

    def test_roundtrip_preserves_data(self, tmp_path):
        df   = _fresh_tmax(n=50)
        path = tmp_path / "tmax.csv"
        atomic_write_csv(df, path, SCHEMAS["historical_tmax"])
        reloaded = pd.read_csv(path)
        assert list(reloaded.columns) == list(df.columns)
        assert len(reloaded) == len(df)


# ─── 5. check_all_freshness ──────────────────────────────────────────────────

class TestCheckAllFreshness:

    def _write_source(self, tmp_path, name, df):
        """Write a named source to tmp_path using the expected relative subpath."""
        dest_map = {
            "historical_tmax":     "data/historical_tmax.csv",
            "openmeteo_forecasts": "data/forecasts/openmeteo_forecast_history.csv",
            "gefs_spread":         "data/forecasts/gefs_spread.csv",
            "climate_indices":     "data/climate_indices.csv",
            "mjo_indices":         "data/mjo_indices.csv",
        }
        path = tmp_path / dest_map[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def _make_file_map(self, tmp_path, overrides: dict = None):
        fresh = {
            "historical_tmax":     _fresh_tmax(),
            "openmeteo_forecasts": _fresh_openmeteo(),
            "gefs_spread":         _fresh_gefs(),
            "climate_indices":     _fresh_climate_indices(),
            "mjo_indices":         _fresh_mjo(),
        }
        if overrides:
            fresh.update(overrides)
        return {
            name: self._write_source(tmp_path, name, df)
            for name, df in fresh.items()
        }

    def test_all_fresh_passes(self, tmp_path):
        files = self._make_file_map(tmp_path)
        results = check_all_freshness(files=files, warn_only=True)
        for r in results:
            assert r.ok, f"{r.source} failed:\n{r.summary()}"

    def test_stale_source_fails(self, tmp_path):
        files = self._make_file_map(
            tmp_path,
            overrides={"gefs_spread": _fresh_gefs(age_days=10)},
        )
        results = check_all_freshness(files=files, warn_only=True)
        gefs_result = next(r for r in results if "gefs_spread" in r.source)
        assert not gefs_result.ok

    def test_missing_file_is_error(self, tmp_path):
        files = self._make_file_map(tmp_path)
        files["historical_tmax"] = tmp_path / "data" / "does_not_exist.csv"
        results = check_all_freshness(files=files, warn_only=True)
        tmax_result = next(r for r in results if "historical_tmax" in r.source)
        assert not tmax_result.ok
        assert any("not found" in e for e in tmax_result.errors)

    def test_warn_only_false_raises_on_stale(self, tmp_path):
        files = self._make_file_map(
            tmp_path,
            overrides={"mjo_indices": _fresh_mjo(age_days=30)},
        )
        with pytest.raises(DataValidationError, match="Stale data"):
            check_all_freshness(files=files, warn_only=False)

    def test_warn_only_true_returns_all_results(self, tmp_path):
        files = self._make_file_map(
            tmp_path,
            overrides={"mjo_indices": _fresh_mjo(age_days=30)},
        )
        results = check_all_freshness(files=files, warn_only=True)
        assert len(results) == len(files)

    def test_returns_one_result_per_source(self, tmp_path):
        files = self._make_file_map(tmp_path)
        results = check_all_freshness(files=files, warn_only=True)
        assert len(results) == len(files)
