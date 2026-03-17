"""
tests/test_features.py

Unit tests for src/features.py — correctness of all feature engineering functions.
Data-leakage properties are covered separately in test_no_data_leakage.py.

Run from project root:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    add_time_features,
    add_climate_indices,
    add_mjo_indices,
    build_feature_table,
)
from src.model import FEATURES


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _df(dates, city="C", tmax=70.0):
    return pd.DataFrame({
        "city": city,
        "date": pd.to_datetime(dates),
        "y_tmax": tmax,
    })


def _minimal_build_inputs(n=400, city="C", start="2019-01-01", tmax=70.0):
    """Return (historical_df, forecast_df, gfs_df) for build_feature_table."""
    dates = pd.date_range(start, periods=n)
    historical_df = pd.DataFrame({"city": city, "date": dates, "tmax": tmax})
    forecast_df = pd.DataFrame({
        "city":          pd.Series([], dtype=str),
        "forecast_high": pd.Series([], dtype=float),
        "target_date":   pd.to_datetime(pd.Series([], dtype=str)),
    })
    gfs_df = pd.DataFrame({
        "city":              city,
        "date":              dates,
        "forecast_high_gfs": tmax,
        "ensemble_spread":   1.0,
        "ecmwf_minus_gfs":   0.0,
        "precip_forecast":   0.0,
    })
    return historical_df, forecast_df, gfs_df


# ─── add_time_features ───────────────────────────────────────────────────────

class TestAddTimeFeatures:
    """Correctness of cyclical time encodings and linear trend."""

    def test_output_contains_all_expected_columns(self):
        result = add_time_features(_df(["2020-06-15"]))
        for col in ("day_of_year", "month", "doy_sin", "doy_cos",
                    "doy_sin2", "doy_cos2", "days_since_2018"):
            assert col in result.columns, f"missing column: {col}"

    def test_day_of_year_jan1_is_1(self):
        result = add_time_features(_df(["2020-01-01"]))
        assert result.iloc[0]["day_of_year"] == 1

    def test_day_of_year_dec31_leap_year(self):
        result = add_time_features(_df(["2020-12-31"]))
        assert result.iloc[0]["day_of_year"] == 366  # 2020 is a leap year

    def test_day_of_year_dec31_non_leap_year(self):
        result = add_time_features(_df(["2021-12-31"]))
        assert result.iloc[0]["day_of_year"] == 365

    def test_month_column(self):
        for month in range(1, 13):
            result = add_time_features(_df([f"2020-{month:02d}-15"]))
            assert result.iloc[0]["month"] == month

    def test_doy_sin_cos_form_unit_vector(self):
        """sin²(x) + cos²(x) == 1 for every date in the year."""
        result = add_time_features(_df(pd.date_range("2020-01-01", periods=366)))
        total = result["doy_sin"] ** 2 + result["doy_cos"] ** 2
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_doy_sin2_cos2_form_unit_vector(self):
        result = add_time_features(_df(pd.date_range("2020-01-01", periods=366)))
        total = result["doy_sin2"] ** 2 + result["doy_cos2"] ** 2
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_days_since_2018_epoch_is_zero(self):
        result = add_time_features(_df(["2018-01-01"]))
        assert result.iloc[0]["days_since_2018"] == 0

    def test_days_since_2018_increments_daily(self):
        result = add_time_features(_df(["2018-01-01", "2018-01-02", "2018-01-08"]))
        assert result.iloc[0]["days_since_2018"] == 0
        assert result.iloc[1]["days_since_2018"] == 1
        assert result.iloc[2]["days_since_2018"] == 7

    def test_days_since_2018_clipped_to_zero_before_epoch(self):
        result = add_time_features(_df(["2017-12-31", "2015-06-01"]))
        assert result.iloc[0]["days_since_2018"] == 0
        assert result.iloc[1]["days_since_2018"] == 0

    def test_doy_cos_near_positive_one_on_jan1(self):
        """Jan 1 (start of cycle) → cos ≈ +1."""
        result = add_time_features(_df(["2020-01-01"]))
        assert result.iloc[0]["doy_cos"] > 0.99

    def test_doy_cos_near_negative_one_on_midsummer(self):
        """Around day 183 (mid-year) → cos ≈ -1 (half cycle)."""
        result = add_time_features(_df(["2020-07-02"]))   # doy≈184 in leap year
        assert result.iloc[0]["doy_cos"] < -0.99

    def test_doy_sin_positive_in_spring_negative_in_autumn(self):
        """sin peaks in late spring (~Apr) and troughs in late autumn (~Oct)."""
        spring = add_time_features(_df(["2020-04-01"])).iloc[0]["doy_sin"]
        autumn = add_time_features(_df(["2020-10-01"])).iloc[0]["doy_sin"]
        assert spring > 0
        assert autumn < 0

    def test_second_harmonic_period_is_half_year(self):
        """doy_sin2/cos2 complete two full cycles per year."""
        # Second harmonic peak at doy≈46 (Feb 15): sin(4π×46/365.25) ≈ +1
        result = add_time_features(_df(["2020-02-15"]))   # doy≈46
        assert result.iloc[0]["doy_sin2"] > 0.99


# ─── add_climate_indices ─────────────────────────────────────────────────────

class TestAddClimateIndices:
    """Monthly climate indices join correctly onto the feature table."""

    def _feat_df(self, dates, city="C"):
        """DataFrame with time features already applied (needed for _year/_month)."""
        return add_time_features(_df(dates, city=city))

    def _indices(self):
        return pd.DataFrame({
            "year":      [2020, 2020, 2020],
            "month":     [1,    2,    3],
            "ao_index":  [0.5, -0.3,  1.2],
            "nao_index": [-0.1, 0.4, -0.8],
            "oni":       [0.2,  0.2,  0.1],
            "pna_index": [0.3, -0.2,  0.4],
            "pdo_index": [-0.5, 0.1, -0.3],
        })

    def test_all_index_columns_added(self):
        df = self._feat_df(["2020-01-15"])
        result = add_climate_indices(df, self._indices())
        for col in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index"):
            assert col in result.columns, f"missing: {col}"

    def test_correct_value_for_matching_month(self):
        df = self._feat_df(["2020-01-15"])
        result = add_climate_indices(df, self._indices())
        assert result.iloc[0]["ao_index"] == pytest.approx(0.5)
        assert result.iloc[0]["nao_index"] == pytest.approx(-0.1)

    def test_different_months_get_different_values(self):
        df = self._feat_df(["2020-01-15", "2020-02-15", "2020-03-15"])
        result = add_climate_indices(df, self._indices())
        assert result.iloc[0]["ao_index"] == pytest.approx(0.5)
        assert result.iloc[1]["ao_index"] == pytest.approx(-0.3)
        assert result.iloc[2]["ao_index"] == pytest.approx(1.2)

    def test_missing_month_gives_nan(self):
        """Dates with no matching (year, month) in indices → NaN."""
        df = self._feat_df(["2019-06-15"])   # 2019 not in indices
        result = add_climate_indices(df, self._indices())
        assert pd.isna(result.iloc[0]["ao_index"])

    def test_multiple_cities_same_month_same_value(self):
        """Indices are global — all cities get the same value for the same month."""
        df = pd.concat([
            self._feat_df(["2020-01-15"], city="NYC"),
            self._feat_df(["2020-01-15"], city="LAX"),
        ])
        result = add_climate_indices(df, self._indices())
        assert result.iloc[0]["ao_index"] == result.iloc[1]["ao_index"]

    def test_original_columns_preserved(self):
        """add_climate_indices must not drop any existing columns."""
        df = self._feat_df(["2020-01-15"])
        original_cols = set(df.columns)
        result = add_climate_indices(df, self._indices())
        assert original_cols.issubset(set(result.columns))


# ─── add_mjo_indices ─────────────────────────────────────────────────────────

class TestAddMjoIndices:
    """Daily MJO RMM indices join correctly onto the feature table."""

    def _mjo(self):
        return pd.DataFrame({
            "date":          pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "mjo_amplitude": [1.5, 0.8, 2.1],
            "mjo_phase_sin": [0.5, -0.3, 0.9],
            "mjo_phase_cos": [0.866, 0.954, 0.436],
        })

    def test_all_mjo_columns_added(self):
        result = add_mjo_indices(_df(["2020-01-01"]), self._mjo())
        for col in ("mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"):
            assert col in result.columns, f"missing: {col}"

    def test_correct_amplitude_joined(self):
        result = add_mjo_indices(_df(["2020-01-01"]), self._mjo())
        assert result.iloc[0]["mjo_amplitude"] == pytest.approx(1.5)

    def test_correct_phase_joined(self):
        result = add_mjo_indices(_df(["2020-01-02"]), self._mjo())
        assert result.iloc[0]["mjo_phase_sin"] == pytest.approx(-0.3)
        assert result.iloc[0]["mjo_phase_cos"] == pytest.approx(0.954)

    def test_missing_date_gives_nan(self):
        """Dates absent from mjo_df get NaN (imputed to 0 by build_feature_table)."""
        result = add_mjo_indices(_df(["2020-01-10"]), self._mjo())
        assert pd.isna(result.iloc[0]["mjo_amplitude"])

    def test_multiple_dates_joined_independently(self):
        df = _df(["2020-01-01", "2020-01-02", "2020-01-03"])
        result = add_mjo_indices(df, self._mjo())
        assert result.iloc[0]["mjo_amplitude"] == pytest.approx(1.5)
        assert result.iloc[1]["mjo_amplitude"] == pytest.approx(0.8)
        assert result.iloc[2]["mjo_amplitude"] == pytest.approx(2.1)

    def test_original_columns_preserved(self):
        df = _df(["2020-01-01"])
        original_cols = set(df.columns)
        result = add_mjo_indices(df, self._mjo())
        assert original_cols.issubset(set(result.columns))


# ─── build_feature_table ─────────────────────────────────────────────────────

class TestBuildFeatureTable:
    """Integration tests for the full feature engineering pipeline."""

    def test_all_features_columns_present(self):
        """Every column in model.FEATURES must appear in the output."""
        hist, fcast, gfs = _minimal_build_inputs()
        result = build_feature_table(hist, fcast, gfs_df=gfs)
        missing = [c for c in FEATURES if c not in result.columns]
        assert missing == [], f"FEATURES columns missing: {missing}"

    def test_no_nan_in_features_for_interior_rows(self):
        """
        Rows with enough lag history (past the 30-day warmup) must have
        zero NaN values in any FEATURES column after imputation.
        """
        hist, fcast, gfs = _minimal_build_inputs(n=400)
        result = build_feature_table(hist, fcast, gfs_df=gfs)
        # Row 0 has NaN forecast_high (GFS shift), first ~30 rows have NaN lags
        interior = result.iloc[35:].copy()
        for col in FEATURES:
            n_nan = int(interior[col].isna().sum())
            assert n_nan == 0, (
                f"FEATURES['{col}'] has {n_nan} NaN in interior rows "
                f"(should be 0 after imputation)"
            )

    def test_ytmax_matches_historical_tmax(self):
        """y_tmax in output must equal the tmax values passed in historical_df."""
        hist, fcast, gfs = _minimal_build_inputs(n=100)
        hist.loc[10, "tmax"] = 99.0   # inject a known value
        result = build_feature_table(hist, fcast, gfs_df=gfs)
        target_date = hist.iloc[10]["date"]
        row = result[pd.to_datetime(result["date"]) == pd.to_datetime(target_date)]
        assert row.iloc[0]["y_tmax"] == pytest.approx(99.0)

    def test_two_cities_lag_features_are_independent(self):
        """lag1_tmax for city A must reflect only city A's history."""
        hist_a = pd.DataFrame({"city": "A",
                                "date": pd.date_range("2019-01-01", periods=60),
                                "tmax": 70.0})
        hist_b = pd.DataFrame({"city": "B",
                                "date": pd.date_range("2019-01-01", periods=60),
                                "tmax": 90.0})
        hist   = pd.concat([hist_a, hist_b])
        fcast  = pd.DataFrame({"city": pd.Series([], dtype=str),
                                "forecast_high": pd.Series([], dtype=float),
                                "target_date": pd.to_datetime(pd.Series([], dtype=str))})
        result = build_feature_table(hist, fcast)

        a_lags = result[result["city"] == "A"]["lag1_tmax"].dropna()
        b_lags = result[result["city"] == "B"]["lag1_tmax"].dropna()
        np.testing.assert_allclose(a_lags.values, 70.0, atol=0.01)
        np.testing.assert_allclose(b_lags.values, 90.0, atol=0.01)

    def test_climate_indices_propagated(self):
        """When indices_df is provided, ao_index etc. must be non-NaN."""
        hist, fcast, gfs = _minimal_build_inputs(n=400)
        indices_df = pd.DataFrame({
            "year":      [2019] * 12 + [2020] * 12,
            "month":     list(range(1, 13)) * 2,
            "ao_index":  [0.5] * 24,
            "nao_index": [0.1] * 24,
            "oni":       [0.2] * 24,
            "pna_index": [0.3] * 24,
            "pdo_index": [-0.1] * 24,
        })
        result = build_feature_table(hist, fcast, gfs_df=gfs, indices_df=indices_df)
        assert result["ao_index"].isna().sum() == 0

    def test_mjo_indices_propagated(self):
        """When mjo_df is provided, mjo_amplitude etc. must be non-NaN."""
        hist, fcast, gfs = _minimal_build_inputs(n=60)
        dates = pd.date_range("2019-01-01", periods=60)
        mjo_df = pd.DataFrame({
            "date":          dates,
            "mjo_amplitude": 1.5,
            "mjo_phase_sin": 0.5,
            "mjo_phase_cos": 0.866,
        })
        result = build_feature_table(hist, fcast, gfs_df=gfs, mjo_df=mjo_df)
        assert result["mjo_amplitude"].isna().sum() == 0

    def test_mjo_missing_dates_imputed_to_zero(self):
        """Dates absent from mjo_df must be imputed to 0 (neutral MJO)."""
        hist, fcast, gfs = _minimal_build_inputs(n=60)
        # Provide no mjo_df → all mjo columns should be 0
        result = build_feature_table(hist, fcast, gfs_df=gfs, mjo_df=None)
        assert (result["mjo_amplitude"] == 0.0).all()
        assert (result["mjo_phase_sin"] == 0.0).all()
        assert (result["mjo_phase_cos"] == 0.0).all()

    def test_missing_climate_indices_default_to_zero(self):
        """When indices_df is None, climate index columns must exist and equal 0."""
        hist, fcast, gfs = _minimal_build_inputs(n=60)
        result = build_feature_table(hist, fcast, gfs_df=gfs, indices_df=None)
        for col in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index"):
            assert col in result.columns
            assert (result[col] == 0.0).all(), (
                f"{col} should default to 0.0 when indices_df is None"
            )

    def test_forecast_minus_climo_computed(self):
        """forecast_minus_climo = forecast_high - climo_mean_doy."""
        hist, fcast, gfs = _minimal_build_inputs(n=400)
        result = build_feature_table(hist, fcast, gfs_df=gfs)
        interior = result.dropna(subset=["forecast_high", "climo_mean_doy"])
        diff = interior["forecast_high"] - interior["climo_mean_doy"]
        np.testing.assert_allclose(
            diff.values,
            interior["forecast_minus_climo"].values,
            atol=1e-6,
        )

    def test_output_sorted_by_city_and_date(self):
        """build_feature_table must return rows sorted by (city, date)."""
        hist, fcast, gfs = _minimal_build_inputs(n=100)
        result = build_feature_table(hist, fcast, gfs_df=gfs)
        dates_sorted = result.groupby("city")["date"].is_monotonic_increasing
        assert dates_sorted.all(), "rows not sorted by date within each city"
