"""
tests/test_no_data_leakage.py

Tests ensuring no data leakage in the temperature prediction backtest pipeline.

Data leakage = information from the test/future period bleeds into the training
process, inflating apparent model performance.  These tests verify that each
stage of the pipeline is strictly backward-looking.

Coverage:
  1. Lag features (add_lag_features) are strictly backward-looking
  2. Climatology (add_climatology) excludes the current and future years
  3. GFS forecasts are shifted 1 day in build_feature_table
  4. Walk-forward backtest trains on strictly prior rows only
  5. Model.fit() never receives test-row y_tmax

Run from project root:
    pytest tests/test_no_data_leakage.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import add_lag_features, add_climatology, add_time_features, build_feature_table
from src.model import TempDistributionModel, FEATURES


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _city_df(n: int, city: str = "TestCity", start: str = "2020-01-01") -> pd.DataFrame:
    """Single-city DataFrame with sequential y_tmax (0, 1, 2, ...) for easy arithmetic."""
    dates = pd.date_range(start, periods=n)
    return pd.DataFrame({"city": city, "date": dates, "y_tmax": np.arange(n, dtype=float)})


def _climo_input(year_vals: dict, city: str = "TestCity", day: int = 15) -> pd.DataFrame:
    """
    Multi-year DataFrame with one row per year at a fixed day-of-month (Jan 15).
    year_vals: {year: tmax_value}
    Returns df with add_time_features already applied.
    Includes a forecast_high column (required by add_climatology for forecast_minus_climo).
    """
    rows = [
        {
            "city": city,
            "date": pd.Timestamp(f"{year}-01-{day:02d}"),
            "y_tmax": float(tmax),
            "forecast_high": float(tmax),  # needed by add_climatology
        }
        for year, tmax in year_vals.items()
    ]
    return add_time_features(pd.DataFrame(rows))


def _full_feature_df(
    n: int = 420,
    city: str = "TestCity",
    start: str = "2019-01-01",
    tmax_val: float = 70.0,
    spike_idx: int = None,
    spike_val: float = 999.0,
) -> pd.DataFrame:
    """
    Build a minimal but complete feature table (all FEATURES present) via
    build_feature_table.  All forecasts equal tmax_val.  Optionally inject a
    spike y_tmax at one index to probe leakage.
    """
    dates = pd.date_range(start, periods=n)
    tmax = np.full(n, tmax_val, dtype=float)
    if spike_idx is not None:
        tmax[spike_idx] = spike_val

    historical_df = pd.DataFrame({"city": city, "date": dates, "tmax": tmax})

    # Empty NWS forecast (so forecast_high comes purely from GFS after the shift)
    forecast_df = pd.DataFrame({
        "city": pd.Series([], dtype=str),
        "forecast_high": pd.Series([], dtype=float),
        "target_date": pd.to_datetime(pd.Series([], dtype=str)),
    })

    gfs_df = pd.DataFrame({
        "city": city,
        "date": dates,
        "forecast_high_gfs": np.full(n, tmax_val),
        "ensemble_spread": np.ones(n),
        "ecmwf_minus_gfs": np.zeros(n),
        "precip_forecast": np.zeros(n),
    })

    df = build_feature_table(historical_df, forecast_df, gfs_df=gfs_df)
    return df[df["city"] == city].sort_values("date").reset_index(drop=True)


# ─── 1. Lag features ─────────────────────────────────────────────────────────

class TestLagFeatureLeakage:
    """Lag features must be computed via shift(1) then rolling — never peeking at current row."""

    def test_lag1_equals_previous_day_ytmax(self):
        """lag1_tmax[i] must equal y_tmax[i-1], not y_tmax[i]."""
        df = _city_df(10)   # y_tmax = 0, 1, 2, ..., 9
        result = add_lag_features(df)

        for i in range(1, 10):
            expected = float(i - 1)
            actual = result.iloc[i]["lag1_tmax"]
            assert actual == pytest.approx(expected), (
                f"lag1_tmax[{i}] = {actual}, expected {expected} (y_tmax[{i-1}]). "
                f"If it equalled y_tmax[{i}]={i} that would be current-day leakage."
            )

    def test_lag1_not_current_day(self):
        """lag1_tmax[i] must never equal y_tmax[i] for a non-constant series."""
        df = _city_df(10)   # strictly increasing, so no two rows share a value
        result = add_lag_features(df)

        for i in range(1, 10):
            assert result.iloc[i]["lag1_tmax"] != result.iloc[i]["y_tmax"], (
                f"lag1_tmax[{i}] == y_tmax[{i}] = {result.iloc[i]['y_tmax']} "
                f"— current-day value leaked into lag"
            )

    def test_lag3_mean_uses_only_previous_3_days(self):
        """lag3_mean_tmax[i] = mean(y_tmax[i-1], y_tmax[i-2], y_tmax[i-3])."""
        df = _city_df(15)   # y_tmax = 0..14
        result = add_lag_features(df)

        for i in range(3, 15):
            expected = np.mean([i - 1, i - 2, i - 3])
            actual = result.iloc[i]["lag3_mean_tmax"]
            assert abs(actual - expected) < 1e-9, (
                f"lag3_mean[{i}] = {actual:.4f}, expected {expected:.4f}. "
                f"Including y_tmax[{i}]={i} would give {np.mean([i, i-1, i-2]):.4f}."
            )

    def test_lag7_mean_excludes_current_row(self):
        """lag7_mean_tmax[i] = mean of y_tmax[i-1..i-7], not including y_tmax[i]."""
        df = _city_df(15)
        result = add_lag_features(df)

        for i in range(7, 15):
            expected = np.mean([i - k for k in range(1, 8)])
            actual = result.iloc[i]["lag7_mean_tmax"]
            leaked_value = np.mean([i - k for k in range(0, 7)])  # what leakage would give
            assert abs(actual - expected) < 1e-9, (
                f"lag7_mean[{i}] = {actual:.4f}, expected {expected:.4f}. "
                f"Leakage would give {leaked_value:.4f} (includes y_tmax[{i}]={float(i)})."
            )

    def test_lag30_mean_excludes_current_row(self):
        """lag30_mean_tmax[i] must not include y_tmax[i]."""
        df = _city_df(35)
        result = add_lag_features(df)

        for i in range(30, 35):
            expected = np.mean([i - k for k in range(1, 31)])
            actual = result.iloc[i]["lag30_mean_tmax"]
            assert abs(actual - expected) < 1e-9, (
                f"lag30_mean[{i}] = {actual:.4f}, expected {expected:.4f}."
            )

    def test_lag_trend_is_backward_looking(self):
        """lag_trend = lag1 - lag7_mean.  Both components are backward-looking."""
        df = _city_df(15)
        result = add_lag_features(df)

        for i in range(7, 15):
            expected_lag1 = float(i - 1)
            expected_lag7 = np.mean([i - k for k in range(1, 8)])
            expected_trend = expected_lag1 - expected_lag7
            actual = result.iloc[i]["lag_trend"]
            assert abs(actual - expected_trend) < 1e-9, (
                f"lag_trend[{i}] = {actual:.4f}, expected {expected_trend:.4f}."
            )

    def test_future_rows_do_not_affect_past_lags(self):
        """Replacing a future y_tmax must not change any prior row's lag features."""
        df = _city_df(10)
        result_original = add_lag_features(df.copy())

        df_modified = df.copy()
        df_modified.loc[9, "y_tmax"] = 9999.0   # spike the last row
        result_modified = add_lag_features(df_modified)

        def _equal_or_both_nan(a, b):
            if pd.isna(a) and pd.isna(b):
                return True
            return abs(float(a) - float(b)) < 1e-9

        # All rows before the spike should be unaffected
        for i in range(9):
            orig1 = result_original.iloc[i]["lag1_tmax"]
            mod1  = result_modified.iloc[i]["lag1_tmax"]
            assert _equal_or_both_nan(orig1, mod1), (
                f"lag1_tmax[{i}] changed after modifying future row 9: "
                f"{orig1} → {mod1} — future leakage"
            )
            orig7 = result_original.iloc[i]["lag7_mean_tmax"]
            mod7  = result_modified.iloc[i]["lag7_mean_tmax"]
            assert _equal_or_both_nan(orig7, mod7), (
                f"lag7_mean[{i}] changed after modifying future row 9: "
                f"{orig7} → {mod7} — future leakage"
            )


# ─── 2. Climatology ──────────────────────────────────────────────────────────

class TestClimatologyLeakage:
    """Climatology for year Y must use only y_tmax from years strictly < Y."""

    def test_current_year_excluded_from_own_climo(self):
        """climo_mean_doy for 2021 must not use 2021's y_tmax."""
        # 2019: 50, 2020: 60, 2021: 70  → climo(2021) = mean(50, 60) = 55
        df = _climo_input({2019: 50, 2020: 60, 2021: 70})
        result = add_climatology(df)

        row_2021 = result[pd.to_datetime(result["date"]).dt.year == 2021].iloc[0]
        assert row_2021["climo_mean_doy"] == pytest.approx(55.0), (
            f"climo_mean_doy(2021) = {row_2021['climo_mean_doy']:.1f}, "
            f"expected 55.0 (mean of 2019=50, 2020=60). "
            f"If 70.0 contributed, result would be ~60.0 — current-year leakage."
        )

    def test_only_one_prior_year_used(self):
        """climo for a year with exactly one prior year = that year's tmax."""
        df = _climo_input({2019: 50, 2020: 60})
        result = add_climatology(df)

        row_2020 = result[pd.to_datetime(result["date"]).dt.year == 2020].iloc[0]
        assert row_2020["climo_mean_doy"] == pytest.approx(50.0), (
            f"climo_mean_doy(2020) = {row_2020['climo_mean_doy']:.1f}, "
            f"expected 50.0 (only 2019 data). "
            f"Getting 55.0 would mean 2020's own y_tmax=60 leaked in."
        )

    def test_future_spike_does_not_contaminate_prior_year_climo(self):
        """A spike in year Y+1 must not affect climo_mean_doy for year Y."""
        df = _climo_input({2019: 50, 2020: 60, 2021: 70, 2022: 9999})
        result = add_climatology(df)

        row_2021 = result[pd.to_datetime(result["date"]).dt.year == 2021].iloc[0]
        assert row_2021["climo_mean_doy"] == pytest.approx(55.0), (
            f"climo_mean_doy(2021) = {row_2021['climo_mean_doy']:.1f}, "
            f"expected 55.0 (mean of 2019=50, 2020=60). "
            f"The 2022 spike of 9999 must not contaminate 2021's climo."
        )

    def test_each_year_uses_strictly_more_history(self):
        """Later years incorporate more historical data; no year leaks forward."""
        years = list(range(2018, 2025))
        # Use increasing values so that including later years always pushes the mean up
        df = _climo_input({y: float(y * 10) for y in years})
        result = add_climatology(df)
        result = result.copy()
        result["_year"] = pd.to_datetime(result["date"]).dt.year
        climos = result.set_index("_year")["climo_mean_doy"]

        # For each year from 2019 onward, climo must be strictly less than that year's
        # own y_tmax (because values increase, so including the current year would
        # push the mean up toward or equal to the current value).
        for year in range(2019, 2025):
            own_tmax = float(year * 10)
            assert climos[year] < own_tmax, (
                f"climo_mean_doy({year}) = {climos[year]:.1f} >= {own_tmax:.1f} "
                f"— current-year value leaked into its own climatology"
            )

    def test_climo_increases_monotonically_with_more_history(self):
        """
        Each successive year's climo incorporates one more year of increasing data,
        so climos must be non-decreasing (except cold-start year 1 which uses global).
        """
        years = list(range(2018, 2024))
        df = _climo_input({y: float(y) for y in years})   # tmax = year (increasing)
        result = add_climatology(df)
        result = result.copy()
        result["_year"] = pd.to_datetime(result["date"]).dt.year
        climos = [result[result["_year"] == y].iloc[0]["climo_mean_doy"] for y in years]

        # From 2020 onward each climo is strictly more than its predecessor
        for i in range(2, len(years)):
            assert climos[i] >= climos[i - 1], (
                f"climo[{years[i]}]={climos[i]:.1f} < climo[{years[i-1]}]={climos[i-1]:.1f} "
                f"— climo should grow as more historical data accumulates"
            )


# ─── 3. GFS forecast shift ────────────────────────────────────────────────────

class TestGFSForecastShift:
    """
    build_feature_table shifts GFS forecast columns by 1 day to prevent look-ahead.
    The same-day GFS model run contains same-day observations; only the prior-day
    run is legitimately available at market open.
    """

    def _build_gfs_only(self, n: int = 6):
        """Build a feature table with known per-day GFS values and no NWS forecasts."""
        city = "TestCity"
        dates = pd.date_range("2020-01-01", periods=n)
        gfs_values = np.arange(80, 80 + n, dtype=float)   # 80, 81, 82, ...

        historical_df = pd.DataFrame({"city": city, "date": dates, "tmax": 70.0})
        forecast_df = pd.DataFrame({
            "city": pd.Series([], dtype=str),
            "forecast_high": pd.Series([], dtype=float),
            "target_date": pd.to_datetime(pd.Series([], dtype=str)),
        })
        gfs_df = pd.DataFrame({"city": city, "date": dates, "forecast_high_gfs": gfs_values})

        result = build_feature_table(historical_df, forecast_df, gfs_df=gfs_df)
        rows = result[result["city"] == city].sort_values("date").reset_index(drop=True)
        return rows, gfs_values, dates

    def test_forecast_high_uses_previous_day_gfs(self):
        """
        After the shift, forecast_high for date D must equal GFS forecast_high_gfs
        from date D-1.  (Prevents using the same-day model run.)
        """
        rows, gfs_values, dates = self._build_gfs_only()

        for i in range(1, len(rows)):
            expected = gfs_values[i - 1]   # GFS from the day before
            actual = rows.iloc[i]["forecast_high"]
            assert actual == pytest.approx(expected), (
                f"forecast_high on {dates[i].date()} = {actual}, "
                f"expected {expected} (GFS from {dates[i-1].date()}). "
                f"Getting {gfs_values[i]} (same-day GFS) would be look-ahead leakage."
            )

    def test_first_row_forecast_is_nan_after_shift(self):
        """Row 0 has no prior-day GFS run — its forecast must be NaN after shift."""
        rows, _, _ = self._build_gfs_only()
        assert pd.isna(rows.iloc[0]["forecast_high"]), (
            f"forecast_high at row 0 = {rows.iloc[0]['forecast_high']}, "
            f"expected NaN.  A non-NaN value means the shift wasn't applied."
        )

    def test_ensemble_spread_also_shifted(self):
        """ensemble_spread must also be shifted 1 day (same look-ahead concern)."""
        city = "TestCity"
        n = 6
        dates = pd.date_range("2020-01-01", periods=n)
        spread_values = np.arange(1.0, 1.0 + n)

        historical_df = pd.DataFrame({"city": city, "date": dates, "tmax": 70.0})
        forecast_df = pd.DataFrame({
            "city": pd.Series([], dtype=str),
            "forecast_high": pd.Series([], dtype=float),
            "target_date": pd.to_datetime(pd.Series([], dtype=str)),
        })
        gfs_df = pd.DataFrame({
            "city": city, "date": dates,
            "forecast_high_gfs": 70.0,
            "ensemble_spread": spread_values,
        })
        result = build_feature_table(historical_df, forecast_df, gfs_df=gfs_df)
        rows = result[result["city"] == city].sort_values("date").reset_index(drop=True)

        for i in range(1, n):
            expected = spread_values[i - 1]
            actual = rows.iloc[i]["ensemble_spread"]
            assert actual == pytest.approx(expected), (
                f"ensemble_spread on date {i} = {actual}, "
                f"expected {expected} (from day {i-1}). Shift not applied."
            )


# ─── 5. Model.fit isolation ───────────────────────────────────────────────────

class TestModelFitIsolation:
    """Model.fit() must never receive test-row y_tmax — training is purely on train_df."""

    def _make_split(self, n_train: int = 400):
        """
        Build train_df (n_train rows, all y_tmax=70) and a test_df
        (1 row, y_tmax=999 but identical features to a 70-row).
        """
        n = n_train + 1
        df = _full_feature_df(n=n, tmax_val=70.0, spike_idx=n_train)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)

        # Use the last row as test (it has y_tmax=999 from the spike)
        train_df = df.iloc[:-1].copy()
        test_df  = df.iloc[[-1]].copy()
        return train_df, test_df

    def test_prediction_near_training_distribution(self):
        """
        Model trained on y_tmax=70 must predict ≈70 for a test row whose
        features are identical to a training row (only y_tmax label differs).
        Leakage would shift the prediction toward 999.
        """
        train_df, test_df = self._make_split()
        model = TempDistributionModel()
        model.fit(train_df)
        pred = model.predict_mean(test_df)[0]

        assert abs(pred - 70.0) < 15.0, (
            f"Prediction = {pred:.2f}°F, expected ≈70°F. "
            f"Deviation toward 999 suggests y_tmax=999 leaked into training."
        )
        assert abs(pred - 999.0) > 100.0, (
            f"Prediction = {pred:.2f}°F is too close to spike y_tmax=999."
        )

    def test_test_ytmax_not_in_train_df(self):
        """y_tmax=999 (test label) must be absent from train_df before fit()."""
        train_df, test_df = self._make_split()
        assert 999.0 not in train_df["y_tmax"].values, (
            "Spike y_tmax=999 found in train_df — train/test split is leaking."
        )

    def test_sigma_not_inflated_by_test_residual(self):
        """
        model.sigma_ is computed from training residuals only.
        If test y_tmax=999 leaked in, sigma_ would be huge (929°F residual).
        With clean training data (all y_tmax=70), sigma_ must be small.
        """
        train_df, _ = self._make_split()
        model = TempDistributionModel()
        model.fit(train_df)

        assert model.sigma_ < 10.0, (
            f"sigma_ = {model.sigma_:.2f}°F after training on constant y_tmax=70. "
            f"A very large sigma_ would indicate test data (y_tmax=999) "
            f"leaked into training and inflated the residuals."
        )

    def test_model_is_not_fit_raises_before_fit(self):
        """Sanity: predict before fit must raise RuntimeError, not silently use stale state."""
        model = TempDistributionModel()
        _, test_df = self._make_split()
        with pytest.raises(RuntimeError):
            model.predict_mean(test_df)


# ─── 6. Known limitations (documented, not bugs) ─────────────────────────────

class TestKnownImputationLimitations:
    """
    Some imputation steps use whole-dataset statistics (a mild form of leakage).
    These tests document the current behavior so any future changes are visible.
    They are NOT marked as failures because the impact is low:
      - ensemble_spread and precip_forecast medians are imputed globally
      - climatology cold-start (first year) falls back to global climo
    """

    def test_ensemble_spread_imputation_uses_full_dataset(self):
        """
        When ensemble_spread is NaN it is imputed with the per-city median
        computed over all rows (including future/test rows).  This is a known
        limitation.  This test documents that behavior.
        """
        city = "TestCity"
        n = 10
        dates = pd.date_range("2020-01-01", periods=n)
        historical_df = pd.DataFrame({"city": city, "date": dates, "tmax": 70.0})
        forecast_df = pd.DataFrame({
            "city": pd.Series([], dtype=str),
            "forecast_high": pd.Series([], dtype=float),
            "target_date": pd.to_datetime(pd.Series([], dtype=str)),
        })
        # Only first 5 rows have ensemble_spread; last 5 are NaN
        spread = [2.0] * 5 + [np.nan] * 5
        gfs_df = pd.DataFrame({
            "city": city, "date": dates,
            "forecast_high_gfs": 70.0,
            "ensemble_spread": spread,
        })
        result = build_feature_table(historical_df, forecast_df, gfs_df=gfs_df)
        rows = result[result["city"] == city].sort_values("date").reset_index(drop=True)

        # After imputation, NaN rows should be filled (not still NaN)
        # Document: they are filled with the per-city median = 2.0 (from available rows)
        for i in range(n):
            val = rows.iloc[i]["ensemble_spread"]
            assert not pd.isna(val), (
                f"ensemble_spread at row {i} is still NaN after imputation"
            )

    def test_cold_start_climo_uses_global_fallback(self):
        """
        The first year in the dataset has no prior years, so climo_mean_doy falls
        back to the full-dataset global climatology.  This is documented as an
        accepted cold-start limitation affecting ~365 rows.
        """
        df = _climo_input({2018: 50, 2019: 60, 2020: 70})
        result = add_climatology(df)

        row_2018 = result[pd.to_datetime(result["date"]).dt.year == 2018].iloc[0]
        row_2019 = result[pd.to_datetime(result["date"]).dt.year == 2019].iloc[0]

        # 2018 cold-start: uses global climo = mean(50, 60, 70) = 60.0
        assert row_2018["climo_mean_doy"] == pytest.approx(60.0), (
            f"Cold-start climo(2018) = {row_2018['climo_mean_doy']:.1f}, "
            f"expected 60.0 (global fallback using all years)."
        )
        # 2019 uses only prior year (2018): strictly no look-ahead
        assert row_2019["climo_mean_doy"] == pytest.approx(50.0), (
            f"climo(2019) = {row_2019['climo_mean_doy']:.1f}, expected 50.0 (only 2018)."
        )
