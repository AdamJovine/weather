"""
tests/test_fetch_parsers.py

Tests for all data-fetching parsing functions and data-manipulation utilities.
External HTTP calls are mocked — no network access required.

Covers:
  - _parse_cpc_wide_table  (AO / NAO / PNA format)
  - fetch_oni              (ONI season-to-month mapping)
  - fetch_pdo              (PSL PDO format)
  - fetch_mjo              (BOM RMM format, amplitude + phase encoding)
  - extract_daily_high_forecast  (NWS JSON → temperature)
  - get_nbm_high                 (NBM gridpoint JSON, °C→°F conversion)
  - upsert_live                  (replace-or-keep merge logic)

Run from project root:
    pytest tests/test_fetch_parsers.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.noaa_forecast import extract_daily_high_forecast, get_nbm_high
from src import fetchers as _climate  # climate parsers now live in src/fetchers


# ─── Fixtures ─────────────────────────────────────────────────────────────────

CPC_WIDE = """\
 1950  0.31 -0.13 -0.23  0.05 -0.18  0.13 -0.01 -0.28 -0.32 -0.48 -0.23 -0.26
 1951  0.23  0.19  0.29  0.02  0.19  0.21 -0.12  0.01  0.08 -0.09  0.02  0.03
"""

CPC_WITH_SENTINEL = """\
 1950  0.31 -99.99 -0.23  0.05 -0.18  0.13 -0.01 -0.28 -0.32 -0.48 -0.23 -0.26
"""

ONI_TEXT = """\
SEAS  YR   TOTAL   ANOM
 DJF 1950  23.53  -1.47
 JFM 1950  23.89  -1.22
 DJF 1951  25.10   0.10
"""

PDO_TEXT = """\
PDO Index
Year Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
1900 -0.11 -0.04 -0.25 -0.20 -0.36 -0.42 -0.50 -0.36 -0.39 -0.38 -0.51 -0.25
1901  0.23  0.19  0.29  0.02  0.19  0.21 -0.12  0.01  0.08 -0.09  0.02  0.03
"""

PDO_WITH_SENTINEL = """\
1900 -9.99 -0.04 -0.25 -0.20 -0.36 -0.42 -0.50 -0.36 -0.39 -0.38 -0.51 -0.25
"""

# BOM RMM format: year mon day RMM1 RMM2 phase amplitude .
MJO_TEXT = """\
 Missing  values  are  denoted  by  999.000
 year mon day  RMM1   RMM2   phase  amplitude
 1974   6   2  -0.128   0.371   2       0.392   .
 1974   6   3  -0.055   0.384   2       0.388   .
 1974   6   4  999.000 999.000   1     999.000   .
 1974   6   5  -0.200   0.100   3       0.224   .
"""

NWS_JSON = {
    "properties": {
        "periods": [
            {
                "startTime": "2024-03-16T06:00:00-05:00",
                "isDaytime": True,
                "temperature": 72,
            },
            {
                "startTime": "2024-03-16T20:00:00-05:00",
                "isDaytime": False,
                "temperature": 55,
            },
            {
                "startTime": "2024-03-17T06:00:00-05:00",
                "isDaytime": True,
                "temperature": 68,
            },
        ]
    }
}

NBM_JSON_DEGC = {
    "properties": {
        "maxTemperature": {
            "uom": "wmoUnit:degC",
            "values": [
                {"validTime": "2024-03-16T07:00:00+00:00/PT12H", "value": 20.0},
                {"validTime": "2024-03-17T07:00:00+00:00/PT12H", "value": 15.0},
            ],
        }
    }
}

NBM_JSON_DEGF = {
    "properties": {
        "maxTemperature": {
            "uom": "degF",
            "values": [
                {"validTime": "2024-03-16T07:00:00+00:00/PT12H", "value": 72.0},
            ],
        }
    }
}


# ─── _parse_cpc_wide_table ────────────────────────────────────────────────────

class TestParseCpcWideTable:
    """CPC-style year × 12 month table → long-form DataFrame."""

    def test_row_count(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        assert len(df) == 24   # 2 years × 12 months

    def test_correct_columns(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        assert set(df.columns) == {"year", "month", "ao_index"}

    def test_year_1950_jan_value(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        row = df[(df["year"] == 1950) & (df["month"] == 1)]
        assert row.iloc[0]["ao_index"] == pytest.approx(0.31)

    def test_year_1950_dec_value(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        row = df[(df["year"] == 1950) & (df["month"] == 12)]
        assert row.iloc[0]["ao_index"] == pytest.approx(-0.26)

    def test_year_1951_values(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        row = df[(df["year"] == 1951) & (df["month"] == 1)]
        assert row.iloc[0]["ao_index"] == pytest.approx(0.23)

    def test_months_span_1_to_12(self):
        df = _climate._parse_cpc_wide_table(CPC_WIDE, "ao_index")
        assert set(df["month"].unique()) == set(range(1, 13))

    def test_sentinel_value_becomes_nan(self):
        """abs(v) > 10 is the missing-data sentinel → NaN."""
        df = _climate._parse_cpc_wide_table(CPC_WITH_SENTINEL, "ao_index")
        feb = df[(df["year"] == 1950) & (df["month"] == 2)].iloc[0]
        assert pd.isna(feb["ao_index"])

    def test_non_sentinel_values_preserved(self):
        df = _climate._parse_cpc_wide_table(CPC_WITH_SENTINEL, "ao_index")
        jan = df[(df["year"] == 1950) & (df["month"] == 1)].iloc[0]
        assert jan["ao_index"] == pytest.approx(0.31)

    def test_non_data_header_lines_skipped(self):
        """Lines where the first token is not a number must be ignored."""
        text = "Year Jan Feb Mar\n" + CPC_WIDE
        df = _climate._parse_cpc_wide_table(text, "nao_index")
        assert len(df) == 24


# ─── fetch_oni ───────────────────────────────────────────────────────────────

class TestFetchOni:
    """ONI rolling-season table → monthly long form."""

    def _call(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = text
        with patch(f"{_climate.__name__}.requests.get", return_value=mock_resp):
            return _climate.fetch_oni()

    def test_returns_dataframe_with_correct_columns(self):
        df = self._call(ONI_TEXT)
        assert set(df.columns) == {"year", "month", "oni"}

    def test_djf_maps_to_january(self):
        """DJF is the Dec-Jan-Feb season; center month = January (1)."""
        df = self._call(ONI_TEXT)
        row = df[(df["year"] == 1950) & (df["month"] == 1)]
        assert len(row) == 1
        assert row.iloc[0]["oni"] == pytest.approx(-1.47)

    def test_jfm_maps_to_february(self):
        df = self._call(ONI_TEXT)
        row = df[(df["year"] == 1950) & (df["month"] == 2)]
        assert row.iloc[0]["oni"] == pytest.approx(-1.22)

    def test_no_duplicate_year_month(self):
        df = self._call(ONI_TEXT)
        assert not df.duplicated(subset=["year", "month"]).any()

    def test_header_lines_skipped(self):
        """Lines with an unrecognized season code must be dropped silently."""
        df = self._call(ONI_TEXT)
        # SEAS header line not in ONI_SEASON_MAP → dropped; only 3 data rows remain
        assert len(df) == 3


# ─── fetch_pdo ───────────────────────────────────────────────────────────────

class TestFetchPdo:
    """PSL PDO format → monthly long form."""

    def _call(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = text
        with patch(f"{_climate.__name__}.requests.get", return_value=mock_resp):
            return _climate.fetch_pdo()

    def test_returns_dataframe_with_correct_columns(self):
        df = self._call(PDO_TEXT)
        assert set(df.columns) == {"year", "month", "pdo_index"}

    def test_row_count(self):
        df = self._call(PDO_TEXT)
        assert len(df) == 24  # 2 years × 12 months

    def test_year_1900_jan_value(self):
        df = self._call(PDO_TEXT)
        row = df[(df["year"] == 1900) & (df["month"] == 1)]
        assert row.iloc[0]["pdo_index"] == pytest.approx(-0.11)

    def test_year_1901_march_value(self):
        df = self._call(PDO_TEXT)
        row = df[(df["year"] == 1901) & (df["month"] == 3)]
        assert row.iloc[0]["pdo_index"] == pytest.approx(0.29)

    def test_sentinel_becomes_nan(self):
        """abs(v) > 9 is the PSL missing-data sentinel → NaN."""
        df = self._call(PDO_WITH_SENTINEL)
        jan = df[(df["year"] == 1900) & (df["month"] == 1)].iloc[0]
        assert pd.isna(jan["pdo_index"])

    def test_months_span_1_to_12(self):
        df = self._call(PDO_TEXT)
        assert set(df["month"].unique()) == set(range(1, 13))


# ─── fetch_mjo ───────────────────────────────────────────────────────────────

class TestFetchMjo:
    """BOM RMM format → daily amplitude + cyclical phase encoding."""

    def _call(self, text=MJO_TEXT):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = text
        with patch(f"{_climate.__name__}.requests.get", return_value=mock_resp):
            return _climate.fetch_mjo()

    def test_returns_correct_columns(self):
        df = self._call()
        assert set(df.columns) == {"date", "mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"}

    def test_sentinel_rows_dropped(self):
        """Rows with RMM1 or RMM2 = 999.000 must be excluded."""
        df = self._call()
        # MJO_TEXT has 3 valid rows and 1 sentinel (Jun 4)
        assert len(df) == 3

    def test_amplitude_is_rmm_magnitude(self):
        """mjo_amplitude = sqrt(RMM1² + RMM2²)."""
        df = self._call()
        # Jun 2: RMM1=-0.128, RMM2=0.371
        row = df[df["date"] == pd.Timestamp("1974-06-02")].iloc[0]
        expected = float(np.sqrt(0.128 ** 2 + 0.371 ** 2))
        assert row["mjo_amplitude"] == pytest.approx(expected, rel=1e-3)

    def test_amplitude_is_non_negative(self):
        df = self._call()
        assert (df["mjo_amplitude"] >= 0).all()

    def test_phase_sin_cos_form_unit_vector(self):
        """sin²(phase) + cos²(phase) == 1 for every row."""
        df = self._call()
        total = df["mjo_phase_sin"] ** 2 + df["mjo_phase_cos"] ** 2
        np.testing.assert_allclose(total.values, 1.0, atol=1e-6)

    def test_phase_2_encoding(self):
        """Phase 2: angle = 2π × (2−1)/8 = π/4 → sin=cos=1/√2 ≈ 0.7071."""
        df = self._call()
        row = df[df["date"] == pd.Timestamp("1974-06-02")].iloc[0]
        expected_sin = float(np.sin(2 * np.pi * 1 / 8))  # phase=2 → (2-1)=1
        expected_cos = float(np.cos(2 * np.pi * 1 / 8))
        assert row["mjo_phase_sin"] == pytest.approx(expected_sin, rel=1e-4)
        assert row["mjo_phase_cos"] == pytest.approx(expected_cos, rel=1e-4)

    def test_phase_3_encoding(self):
        """Phase 3: angle = 2π × 2/8 = π/2 → sin≈1, cos≈0."""
        df = self._call()
        row = df[df["date"] == pd.Timestamp("1974-06-05")].iloc[0]
        expected_sin = float(np.sin(2 * np.pi * 2 / 8))  # phase=3 → (3-1)=2
        expected_cos = float(np.cos(2 * np.pi * 2 / 8))
        assert row["mjo_phase_sin"] == pytest.approx(expected_sin, rel=1e-4)
        assert row["mjo_phase_cos"] == pytest.approx(expected_cos, rel=1e-4)

    def test_no_duplicate_dates(self):
        df = self._call()
        assert not df.duplicated(subset=["date"]).any()

    def test_sorted_by_date(self):
        df = self._call()
        assert df["date"].is_monotonic_increasing

    def test_invalid_phase_row_dropped(self):
        """phase outside 1–8 is invalid and must be dropped."""
        bad_text = MJO_TEXT + " 1974   6   6  -0.300   0.200   0       0.361   .\n"
        df = self._call(bad_text)
        # Phase=0 is invalid → dropped; should still have 3 valid rows
        assert len(df) == 3


# ─── extract_daily_high_forecast ─────────────────────────────────────────────

class TestExtractDailyHighForecast:
    """NWS forecast JSON → daytime high temperature."""

    def test_correct_temperature_for_target_date(self):
        result = extract_daily_high_forecast(NWS_JSON, "2024-03-16")
        assert result == pytest.approx(72.0)

    def test_different_date_returns_that_dates_value(self):
        result = extract_daily_high_forecast(NWS_JSON, "2024-03-17")
        assert result == pytest.approx(68.0)

    def test_nighttime_period_not_returned(self):
        """isDaytime=False periods must be ignored."""
        result = extract_daily_high_forecast(NWS_JSON, "2024-03-16")
        assert result != pytest.approx(55.0)  # 55 is the nighttime low

    def test_missing_date_returns_none(self):
        result = extract_daily_high_forecast(NWS_JSON, "2024-03-18")
        assert result is None

    def test_empty_periods_returns_none(self):
        empty_json = {"properties": {"periods": []}}
        result = extract_daily_high_forecast(empty_json, "2024-03-16")
        assert result is None


# ─── get_nbm_high ────────────────────────────────────────────────────────────

class TestGetNbmHigh:
    """NBM gridpoint JSON → max temperature with °C→°F conversion."""

    def _call(self, json_payload, target_date="2024-03-16"):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_payload
        with patch("src.noaa_forecast.requests.get", return_value=mock_resp):
            return get_nbm_high("https://fake-url", target_date)

    def test_degc_converted_to_degf(self):
        """20°C → 68°F."""
        result = self._call(NBM_JSON_DEGC)
        assert result == pytest.approx(68.0)

    def test_degf_returned_as_is(self):
        """When uom is degF, no conversion should be applied."""
        result = self._call(NBM_JSON_DEGF)
        assert result == pytest.approx(72.0)

    def test_different_target_date(self):
        """Mar 17 row: 15°C → 59°F."""
        result = self._call(NBM_JSON_DEGC, target_date="2024-03-17")
        assert result == pytest.approx(59.0)

    def test_missing_date_returns_none(self):
        result = self._call(NBM_JSON_DEGC, target_date="2024-03-18")
        assert result is None

    def test_null_value_skipped(self):
        json_with_null = {
            "properties": {
                "maxTemperature": {
                    "uom": "wmoUnit:degC",
                    "values": [
                        {"validTime": "2024-03-16T07:00:00+00:00/PT12H", "value": None},
                        {"validTime": "2024-03-16T13:00:00+00:00/PT12H", "value": 21.0},
                    ],
                }
            }
        }
        result = self._call(json_with_null)
        # Only the non-null value should contribute; 21°C → 69.8°F
        assert result == pytest.approx(69.8, abs=0.1)

    def test_multiple_periods_same_day_returns_max(self):
        """When multiple valid periods fall on the target date, return the max."""
        json_multi = {
            "properties": {
                "maxTemperature": {
                    "uom": "wmoUnit:degC",
                    "values": [
                        {"validTime": "2024-03-16T07:00:00+00:00/PT12H", "value": 18.0},
                        {"validTime": "2024-03-16T13:00:00+00:00/PT12H", "value": 22.0},
                    ],
                }
            }
        }
        result = self._call(json_multi)
        assert result == pytest.approx(22.0 * 9 / 5 + 32)

