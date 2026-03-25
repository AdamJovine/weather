"""
Comprehensive time-correctness tests for backtesting data access.

Every test uses an in-memory SQLite database seeded with synthetic data at
known timestamps, then verifies that point-in-time queries return **exactly**
the data that was available at a given moment — no more, no less.

Run:
    pytest test_backtest_time.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector.backtest import (
    DISSEMINATION_LAG,
    _availability_cutoff,
    fetch_forecasts,
    fetch_kalshi,
    fetch_metar,
    get_forecasts_at,
    get_kalshi_history_at,
    get_kalshi_prices_at,
    get_latest_forecast_at,
    get_metar_at,
    get_metar_running_max_at,
    get_snapshot_at,
    store_forecasts,
    store_kalshi,
    store_metar,
)
from collector.db import _SCHEMA


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    """Fresh in-memory database with the collector schema."""
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA)
    c.commit()
    yield c
    c.close()


def _insert_metar(conn, station: str, obs_time: datetime, temp_f: float):
    """Insert a single METAR observation."""
    df = pd.DataFrame([{
        "station": station,
        "obs_time": _ts(obs_time),
        "temp_f": temp_f,
        "dew_point_f": temp_f - 10,
        "wind_speed_kt": 5.0,
        "wind_dir": 180,
        "raw_metar": f"METAR {station} {temp_f}F",
    }])
    store_metar(conn, df)
    conn.commit()


def _insert_forecast(
    conn,
    station: str,
    model: str,
    runtime: datetime,
    ftimes_temps: list[tuple[datetime, float]],
):
    """Insert a forecast run with multiple valid-time/temp pairs."""
    rows = []
    for ftime, tmp in ftimes_temps:
        rows.append({
            "station": station,
            "model": model,
            "runtime": _ts(runtime),
            "ftime": _ts(ftime),
            "tmp": tmp,
        })
    df = pd.DataFrame(rows)
    store_forecasts(conn, df)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
#  1. METAR point-in-time basics
# ═══════════════════════════════════════════════════════════════════════════

class TestMetarPointInTime:
    """Verify that METAR queries respect the as_of boundary."""

    def test_sees_only_past_obs(self, conn):
        """Query at T2 must see T1 and T2 but not T3."""
        t1 = _utc(2026, 3, 25, 10, 0)
        t2 = _utc(2026, 3, 25, 11, 0)
        t3 = _utc(2026, 3, 25, 12, 0)

        _insert_metar(conn, "KJFK", t1, 50.0)
        _insert_metar(conn, "KJFK", t2, 55.0)
        _insert_metar(conn, "KJFK", t3, 60.0)

        df = get_metar_at(conn, "KJFK", t2)
        assert len(df) == 2
        assert set(df["obs_time"]) == {_ts(t1), _ts(t2)}

    def test_sees_nothing_before_first_obs(self, conn):
        """Query before the first observation returns empty."""
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_metar(conn, "KJFK", t1, 50.0)

        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 25, 9, 59))
        assert len(df) == 0

    def test_exact_timestamp_is_included(self, conn):
        """Query at the exact obs_time includes that observation."""
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_metar(conn, "KJFK", t1, 50.0)

        df = get_metar_at(conn, "KJFK", t1)
        assert len(df) == 1

    def test_one_second_before_excluded(self, conn):
        """One second before an obs must not see it."""
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_metar(conn, "KJFK", t1, 50.0)

        df = get_metar_at(conn, "KJFK", t1 - timedelta(seconds=1))
        assert len(df) == 0

    def test_date_filter(self, conn):
        """date_str restricts to a single calendar day (UTC)."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 24, 23, 0), 45.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 15, 0), 60.0)

        # as_of well past all obs, but date filter limits to 3/25
        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 26, 0, 0), date_str="2026-03-25")
        assert len(df) == 2
        assert all(o.startswith("2026-03-25") for o in df["obs_time"])

    def test_station_isolation(self, conn):
        """Obs from other stations must not leak in."""
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_metar(conn, "KJFK", t1, 50.0)
        _insert_metar(conn, "KORD", t1, 40.0)

        df = get_metar_at(conn, "KJFK", t1)
        assert len(df) == 1
        assert df.iloc[0]["station"] == "KJFK"


# ═══════════════════════════════════════════════════════════════════════════
#  2. METAR running max
# ═══════════════════════════════════════════════════════════════════════════

class TestMetarRunningMax:

    def test_running_max_increases(self, conn):
        """Running max should increase as warmer obs arrive."""
        times_temps = [
            (_utc(2026, 3, 25, 8, 0), 45.0),
            (_utc(2026, 3, 25, 10, 0), 55.0),
            (_utc(2026, 3, 25, 12, 0), 60.0),
            (_utc(2026, 3, 25, 14, 0), 65.0),
            (_utc(2026, 3, 25, 16, 0), 58.0),  # cooling
        ]
        for t, temp in times_temps:
            _insert_metar(conn, "KJFK", t, temp)

        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 8, 0)) == 45.0
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 10, 0)) == 55.0
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 14, 0)) == 65.0
        # After cooling, max should still be 65
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 16, 0)) == 65.0

    def test_running_max_none_before_data(self, conn):
        """No obs yet -> None."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 9, 0)) is None

    def test_running_max_ignores_other_days(self, conn):
        """Running max on 3/25 must not include 3/24 obs."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 24, 20, 0), 99.0)  # prev day
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)

        result = get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 23, 0))
        assert result == 55.0

    def test_running_max_ignores_other_stations(self, conn):
        """Running max for KJFK must not include KORD obs."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)
        _insert_metar(conn, "KORD", _utc(2026, 3, 25, 10, 0), 99.0)

        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 23, 0)) == 55.0

    def test_running_max_respects_as_of(self, conn):
        """Future obs (after as_of) must not affect the running max."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 14, 0), 99.0)

        # at 12:00 we should only see the 10:00 obs
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 12, 0)) == 55.0


# ═══════════════════════════════════════════════════════════════════════════
#  3. Forecast point-in-time (no lag)
# ═══════════════════════════════════════════════════════════════════════════

class TestForecastPointInTimeNaive:
    """
    Forecast queries with ``lag_adjusted=False``: data is available as soon
    as its runtime <= as_of.
    """

    def test_sees_only_past_runs(self, conn):
        """Two runs at 06Z and 12Z; query at 10Z should see only the 06Z run."""
        run1 = _utc(2026, 3, 25, 6, 0)
        run2 = _utc(2026, 3, 25, 12, 0)
        target = _utc(2026, 3, 26, 0, 0)

        _insert_forecast(conn, "KJFK", "NBS", run1, [(target, 60.0)])
        _insert_forecast(conn, "KJFK", "NBS", run2, [(target, 62.0)])

        df = get_forecasts_at(conn, "KJFK", "NBS", _utc(2026, 3, 25, 10, 0), lag_adjusted=False)
        assert len(df) == 1
        assert df.iloc[0]["runtime"] == _ts(run1)
        assert df.iloc[0]["tmp"] == 60.0

    def test_both_runs_visible_after_second(self, conn):
        run1 = _utc(2026, 3, 25, 6, 0)
        run2 = _utc(2026, 3, 25, 12, 0)
        target = _utc(2026, 3, 26, 0, 0)

        _insert_forecast(conn, "KJFK", "NBS", run1, [(target, 60.0)])
        _insert_forecast(conn, "KJFK", "NBS", run2, [(target, 62.0)])

        df = get_forecasts_at(conn, "KJFK", "NBS", _utc(2026, 3, 25, 12, 0), lag_adjusted=False)
        assert len(df) == 2

    def test_latest_run_selected(self, conn):
        """get_latest_forecast_at returns only the most recent available run."""
        run1 = _utc(2026, 3, 25, 6, 0)
        run2 = _utc(2026, 3, 25, 12, 0)
        target = _utc(2026, 3, 26, 0, 0)

        _insert_forecast(conn, "KJFK", "NBS", run1, [(target, 60.0)])
        _insert_forecast(conn, "KJFK", "NBS", run2, [(target, 62.0)])

        df = get_latest_forecast_at(conn, "KJFK", "NBS", _utc(2026, 3, 25, 13, 0), lag_adjusted=False)
        assert len(df) == 1
        assert df.iloc[0]["runtime"] == _ts(run2)
        assert df.iloc[0]["tmp"] == 62.0

    def test_no_runs_before_first(self, conn):
        run1 = _utc(2026, 3, 25, 6, 0)
        _insert_forecast(conn, "KJFK", "LAV", run1, [(_utc(2026, 3, 25, 12, 0), 55.0)])

        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 5, 0), lag_adjusted=False)
        assert len(df) == 0

    def test_model_isolation(self, conn):
        """LAMP data must not appear in NBM queries."""
        run = _utc(2026, 3, 25, 6, 0)
        target = _utc(2026, 3, 25, 12, 0)

        _insert_forecast(conn, "KJFK", "LAV", run, [(target, 55.0)])
        _insert_forecast(conn, "KJFK", "NBS", run, [(target, 57.0)])

        df = get_forecasts_at(conn, "KJFK", "LAV", _utc(2026, 3, 26, 0, 0), lag_adjusted=False)
        assert all(df["model"] == "LAV")
        assert len(df) == 1

    def test_station_isolation(self, conn):
        run = _utc(2026, 3, 25, 6, 0)
        target = _utc(2026, 3, 25, 12, 0)

        _insert_forecast(conn, "KJFK", "LAV", run, [(target, 55.0)])
        _insert_forecast(conn, "KORD", "LAV", run, [(target, 40.0)])

        df = get_forecasts_at(conn, "KJFK", "LAV", _utc(2026, 3, 26, 0, 0), lag_adjusted=False)
        assert len(df) == 1
        assert df.iloc[0]["station"] == "KJFK"


# ═══════════════════════════════════════════════════════════════════════════
#  4. Dissemination lag handling
# ═══════════════════════════════════════════════════════════════════════════

class TestDisseminationLag:
    """
    With lag_adjusted=True, a forecast run is NOT available until
    runtime + lag.  E.g. LAMP with runtime=10:00 and lag=30 min is
    unavailable at 10:15 but available at 10:35.
    """

    @pytest.mark.parametrize("model,lag", list(DISSEMINATION_LAG.items()))
    def test_unavailable_before_lag(self, conn, model, lag):
        """Runtime + half the lag should NOT yet see the run."""
        runtime = _utc(2026, 3, 25, 10, 0)
        ftime = _utc(2026, 3, 25, 18, 0)
        _insert_forecast(conn, "KJFK", model, runtime, [(ftime, 60.0)])

        query_time = runtime + lag / 2
        df = get_forecasts_at(conn, "KJFK", model, query_time, lag_adjusted=True)
        assert len(df) == 0, f"{model}: should not be visible {lag/2} after runtime"

    @pytest.mark.parametrize("model,lag", list(DISSEMINATION_LAG.items()))
    def test_available_after_lag(self, conn, model, lag):
        """Runtime + lag + 1 minute should see the run."""
        runtime = _utc(2026, 3, 25, 10, 0)
        ftime = _utc(2026, 3, 25, 18, 0)
        _insert_forecast(conn, "KJFK", model, runtime, [(ftime, 60.0)])

        query_time = runtime + lag + timedelta(minutes=1)
        df = get_forecasts_at(conn, "KJFK", model, query_time, lag_adjusted=True)
        assert len(df) == 1, f"{model}: should be visible {lag + timedelta(minutes=1)} after runtime"

    @pytest.mark.parametrize("model,lag", list(DISSEMINATION_LAG.items()))
    def test_exactly_at_lag_boundary(self, conn, model, lag):
        """At exactly runtime + lag the run should be visible (cutoff = as_of - lag = runtime)."""
        runtime = _utc(2026, 3, 25, 10, 0)
        ftime = _utc(2026, 3, 25, 18, 0)
        _insert_forecast(conn, "KJFK", model, runtime, [(ftime, 60.0)])

        query_time = runtime + lag
        df = get_forecasts_at(conn, "KJFK", model, query_time, lag_adjusted=True)
        assert len(df) == 1, f"{model}: should be visible exactly at runtime + lag"

    def test_lamp_lag_30min(self, conn):
        """Concrete LAMP scenario: runtime 10:00, lag 30 min."""
        runtime = _utc(2026, 3, 25, 10, 0)
        ftime = _utc(2026, 3, 25, 12, 0)
        _insert_forecast(conn, "KJFK", "LAV", runtime, [(ftime, 55.0)])

        # 10:15 — not yet available
        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 10, 15), lag_adjusted=True)
        assert len(df) == 0

        # 10:35 — available
        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 10, 35), lag_adjusted=True)
        assert len(df) == 1
        assert df.iloc[0]["tmp"] == 55.0

    def test_nbm_lag_60min(self, conn):
        """Concrete NBM scenario: runtime 12:00, lag 60 min."""
        runtime = _utc(2026, 3, 25, 12, 0)
        ftime = _utc(2026, 3, 26, 0, 0)
        _insert_forecast(conn, "KJFK", "NBS", runtime, [(ftime, 62.0)])

        # 12:45 — not yet
        df = get_latest_forecast_at(conn, "KJFK", "NBS", _utc(2026, 3, 25, 12, 45), lag_adjusted=True)
        assert len(df) == 0

        # 13:05 — available
        df = get_latest_forecast_at(conn, "KJFK", "NBS", _utc(2026, 3, 25, 13, 5), lag_adjusted=True)
        assert len(df) == 1

    def test_gfs_lag_4hr(self, conn):
        """GFS MOS: runtime 06:00, 4-hr lag means available at 10:00."""
        runtime = _utc(2026, 3, 25, 6, 0)
        ftime = _utc(2026, 3, 25, 18, 0)
        _insert_forecast(conn, "KJFK", "GFS", runtime, [(ftime, 58.0)])

        assert len(get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 9, 0), lag_adjusted=True)) == 0
        assert len(get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 10, 1), lag_adjusted=True)) == 1

    def test_mex_lag_5hr(self, conn):
        """GFS Extended: runtime 06:00, 5-hr lag means available at 11:00."""
        runtime = _utc(2026, 3, 25, 6, 0)
        ftime = _utc(2026, 3, 27, 0, 0)
        _insert_forecast(conn, "KJFK", "MEX", runtime, [(ftime, 70.0)])

        assert len(get_latest_forecast_at(conn, "KJFK", "MEX", _utc(2026, 3, 25, 10, 30), lag_adjusted=True)) == 0
        assert len(get_latest_forecast_at(conn, "KJFK", "MEX", _utc(2026, 3, 25, 11, 1), lag_adjusted=True)) == 1

    def test_lag_adjusted_vs_naive(self, conn):
        """Same query returns different results with/without lag adjustment."""
        runtime = _utc(2026, 3, 25, 10, 0)
        ftime = _utc(2026, 3, 25, 18, 0)
        _insert_forecast(conn, "KJFK", "NBS", runtime, [(ftime, 60.0)])

        # 10:30: naive sees it (runtime <= 10:30), lag-adjusted doesn't (10:30 - 60min = 09:30 < 10:00)
        query_time = _utc(2026, 3, 25, 10, 30)
        naive = get_forecasts_at(conn, "KJFK", "NBS", query_time, lag_adjusted=False)
        adjusted = get_forecasts_at(conn, "KJFK", "NBS", query_time, lag_adjusted=True)
        assert len(naive) == 1
        assert len(adjusted) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  5. No future data leakage
# ═══════════════════════════════════════════════════════════════════════════

class TestNoFutureLeakage:
    """
    At any time T, no query should return data from the future.
    These tests seed dense data and scan many time points.
    """

    def test_metar_no_future_obs(self, conn):
        """Walk through 24 hours and verify no obs_time > as_of at each step."""
        base = _utc(2026, 3, 25, 0, 0)
        # Insert obs every 20 minutes for 24 hours
        obs_times = [base + timedelta(minutes=20 * i) for i in range(72)]
        for t in obs_times:
            _insert_metar(conn, "KJFK", t, 50.0 + t.hour)

        # Check every hour
        for hour in range(25):
            check_time = base + timedelta(hours=hour)
            df = get_metar_at(conn, "KJFK", check_time)
            if not df.empty:
                max_obs = max(df["obs_time"])
                assert max_obs <= _ts(check_time), \
                    f"Future leakage at {check_time}: saw obs at {max_obs}"

    def test_forecast_no_future_runtime_naive(self, conn):
        """No forecast with runtime > as_of should be returned (naive mode)."""
        base = _utc(2026, 3, 25, 0, 0)
        for hour in range(0, 24, 3):
            runtime = base + timedelta(hours=hour)
            ftime = runtime + timedelta(hours=12)
            _insert_forecast(conn, "KJFK", "GFS", runtime, [(ftime, 60.0 + hour)])

        for hour in range(25):
            check_time = base + timedelta(hours=hour)
            df = get_forecasts_at(conn, "KJFK", "GFS", check_time, lag_adjusted=False)
            if not df.empty:
                max_runtime = max(df["runtime"])
                assert max_runtime <= _ts(check_time), \
                    f"Future leakage at {check_time}: runtime {max_runtime}"

    @pytest.mark.parametrize("model", ["LAV", "NBS", "GFS", "MEX"])
    def test_forecast_no_future_with_lag(self, conn, model):
        """With lag adjustment, no forecast should appear before runtime + lag."""
        lag = DISSEMINATION_LAG[model]
        base = _utc(2026, 3, 25, 0, 0)
        runtimes = [base + timedelta(hours=h) for h in range(0, 24, 3)]

        for rt in runtimes:
            _insert_forecast(conn, "KJFK", model, rt, [(rt + timedelta(hours=12), 60.0)])

        # Check every 30 minutes
        for minutes in range(0, 24 * 60, 30):
            check_time = base + timedelta(minutes=minutes)
            df = get_forecasts_at(conn, "KJFK", model, check_time, lag_adjusted=True)
            for _, row in df.iterrows():
                rt = datetime.fromisoformat(row["runtime"].replace("Z", "+00:00"))
                available_at = rt + lag
                assert available_at <= check_time, (
                    f"{model} at {check_time}: runtime {row['runtime']} "
                    f"not available until {available_at}"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Forecast supersession
# ═══════════════════════════════════════════════════════════════════════════

class TestForecastSupersession:
    """When a newer run becomes available, it should be the 'latest'."""

    def test_latest_switches_to_newer_run(self, conn):
        """After the 12Z run becomes available, it supersedes the 06Z run."""
        run_06z = _utc(2026, 3, 25, 6, 0)
        run_12z = _utc(2026, 3, 25, 12, 0)
        target = _utc(2026, 3, 26, 0, 0)

        _insert_forecast(conn, "KJFK", "LAV", run_06z, [(target, 60.0)])
        _insert_forecast(conn, "KJFK", "LAV", run_12z, [(target, 63.0)])

        # At 06:35 (after 06Z lag), latest is 06Z
        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 6, 35), lag_adjusted=True)
        assert df.iloc[0]["runtime"] == _ts(run_06z)
        assert df.iloc[0]["tmp"] == 60.0

        # At 12:35 (after 12Z lag), latest switches to 12Z
        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 12, 35), lag_adjusted=True)
        assert df.iloc[0]["runtime"] == _ts(run_12z)
        assert df.iloc[0]["tmp"] == 63.0

    def test_three_runs_progressive_availability(self, conn):
        """Three successive runs, each superseding the previous."""
        runs = [
            (_utc(2026, 3, 25, 0, 0), 58.0),
            (_utc(2026, 3, 25, 6, 0), 61.0),
            (_utc(2026, 3, 25, 12, 0), 64.0),
        ]
        target = _utc(2026, 3, 26, 0, 0)
        for rt, tmp in runs:
            _insert_forecast(conn, "KJFK", "GFS", rt, [(target, tmp)])

        lag = DISSEMINATION_LAG["GFS"]

        # Before any run is available
        df = get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 3, 0), lag_adjusted=True)
        assert len(df) == 0

        # After 00Z becomes available (00:00 + 4hr = 04:00)
        df = get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 4, 1), lag_adjusted=True)
        assert df.iloc[0]["tmp"] == 58.0

        # After 06Z becomes available (06:00 + 4hr = 10:00)
        df = get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 10, 1), lag_adjusted=True)
        assert df.iloc[0]["tmp"] == 61.0

        # After 12Z becomes available (12:00 + 4hr = 16:00)
        df = get_latest_forecast_at(conn, "KJFK", "GFS", _utc(2026, 3, 25, 16, 1), lag_adjusted=True)
        assert df.iloc[0]["tmp"] == 64.0


# ═══════════════════════════════════════════════════════════════════════════
#  7. Multi-station, multi-model scenarios
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiStationMultiModel:

    STATIONS = ["KJFK", "KORD", "KHOU", "KMDW", "KCQT"]
    MODELS = ["LAV", "NBS", "GFS", "MEX"]

    def test_each_station_sees_own_data(self, conn):
        """Insert data for 5 stations; each query returns only its own."""
        t = _utc(2026, 3, 25, 10, 0)
        for i, stn in enumerate(self.STATIONS):
            _insert_metar(conn, stn, t, 50.0 + i)

        for i, stn in enumerate(self.STATIONS):
            df = get_metar_at(conn, stn, t)
            assert len(df) == 1
            assert df.iloc[0]["temp_f"] == 50.0 + i

    def test_each_model_independent(self, conn):
        """Insert forecasts for all 4 models; each query is isolated."""
        runtime = _utc(2026, 3, 25, 6, 0)
        ftime = _utc(2026, 3, 26, 0, 0)

        for i, model in enumerate(self.MODELS):
            _insert_forecast(conn, "KJFK", model, runtime, [(ftime, 60.0 + i)])

        for i, model in enumerate(self.MODELS):
            df = get_forecasts_at(conn, "KJFK", model, _utc(2026, 3, 26, 0, 0), lag_adjusted=False)
            assert len(df) == 1
            assert df.iloc[0]["tmp"] == 60.0 + i

    def test_snapshot_combines_all_sources(self, conn):
        """get_snapshot_at should return METAR max + latest run for each model."""
        date_str = "2026-03-25"
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 14, 0), 62.0)

        runtime = _utc(2026, 3, 25, 6, 0)
        ftime = _utc(2026, 3, 26, 0, 0)
        for model in self.MODELS:
            _insert_forecast(conn, "KJFK", model, runtime, [(ftime, 60.0)])

        snap = get_snapshot_at(conn, "KJFK", date_str, _utc(2026, 3, 25, 20, 0), lag_adjusted=False)
        assert snap["metar_running_max"] == 62.0
        assert snap["metar_obs_count"] == 2
        assert len(snap["latest_lamp"]) == 1
        assert len(snap["latest_nbm"]) == 1
        assert len(snap["latest_gfs"]) == 1
        assert len(snap["latest_mex"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  8. Cross-midnight / time boundary scenarios
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeBoundaries:

    def test_cross_midnight_metar(self, conn):
        """Obs spanning midnight UTC should be queryable from both sides."""
        pre_midnight = _utc(2026, 3, 25, 23, 50)
        post_midnight = _utc(2026, 3, 26, 0, 10)

        _insert_metar(conn, "KJFK", pre_midnight, 50.0)
        _insert_metar(conn, "KJFK", post_midnight, 48.0)

        # At 23:55 — only pre-midnight
        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 25, 23, 55))
        assert len(df) == 1

        # At 00:15 — both
        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 26, 0, 15))
        assert len(df) == 2

    def test_cross_midnight_running_max(self, conn):
        """Running max for a date should not include next-day obs even if as_of is after midnight."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 22, 0), 60.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 26, 2, 0), 70.0)  # next day

        # Running max for 3/25 at 3/26 03:00 should only see the 60.0 obs
        result = get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 26, 3, 0))
        assert result == 60.0

    def test_forecast_run_at_midnight(self, conn):
        """A run at exactly 00:00Z should be queryable."""
        runtime = _utc(2026, 3, 26, 0, 0)
        ftime = _utc(2026, 3, 26, 12, 0)
        _insert_forecast(conn, "KJFK", "LAV", runtime, [(ftime, 55.0)])

        # At 00:00 — visible naive
        df = get_forecasts_at(conn, "KJFK", "LAV", runtime, lag_adjusted=False)
        assert len(df) == 1

        # At 00:00 — not visible with lag (needs 00:30)
        df = get_forecasts_at(conn, "KJFK", "LAV", runtime, lag_adjusted=True)
        assert len(df) == 0

    def test_early_morning_data_sparse(self, conn):
        """Early morning (00-06Z): few METAR obs, no new GFS runs."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 1, 0), 40.0)
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 3, 0), 38.0)

        # Only GFS run from previous day's 18Z
        gfs_rt = _utc(2026, 3, 24, 18, 0)
        _insert_forecast(conn, "KJFK", "GFS", gfs_rt, [(_utc(2026, 3, 25, 18, 0), 55.0)])

        # At 04:00 UTC
        snap = get_snapshot_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 4, 0), lag_adjusted=False)
        assert snap["metar_running_max"] == 40.0
        assert snap["metar_obs_count"] == 2
        assert len(snap["latest_gfs"]) == 1

    def test_multi_day_metar_query(self, conn):
        """Without date_str filter, metar_at returns obs across days."""
        for day in range(24, 27):
            _insert_metar(conn, "KJFK", _utc(2026, 3, day, 12, 0), 50.0 + day)

        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 27, 0, 0))
        assert len(df) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  9. Dense backtesting simulation (5-minute steps)
# ═══════════════════════════════════════════════════════════════════════════

class TestDenseBacktestSimulation:
    """Simulate a full day of 5-minute backtesting snapshots."""

    def _seed_full_day(self, conn, station: str, date_str: str):
        """Seed a realistic day of data: METAR every 20 min, model runs on schedule."""
        base = datetime.fromisoformat(date_str + "T00:00:00+00:00")

        # METAR: every 20 minutes, temp rises then falls
        for m in range(0, 24 * 60, 20):
            t = base + timedelta(minutes=m)
            hour = t.hour + t.minute / 60
            # Simple diurnal curve: min at 6, max at 15
            temp = 50.0 + 15.0 * max(0, 1 - abs(hour - 15) / 9)
            _insert_metar(conn, station, t, round(temp, 1))

        # LAMP: runs every hour on the hour
        for h in range(0, 24):
            rt = base + timedelta(hours=h)
            ftimes = [(rt + timedelta(hours=fh), 55.0 + h * 0.3) for fh in range(1, 6)]
            _insert_forecast(conn, station, "LAV", rt, ftimes)

        # NBM: runs every hour
        for h in range(0, 24):
            rt = base + timedelta(hours=h)
            ftimes = [(rt + timedelta(hours=fh), 57.0 + h * 0.2) for fh in range(3, 24, 3)]
            _insert_forecast(conn, station, "NBS", rt, ftimes)

        # GFS: runs at 00, 06, 12, 18
        for h in [0, 6, 12, 18]:
            rt = base + timedelta(hours=h)
            ftimes = [(rt + timedelta(hours=fh), 58.0 + h * 0.5) for fh in range(6, 48, 3)]
            _insert_forecast(conn, station, "GFS", rt, ftimes)

        # MEX: runs at 00, 12
        for h in [0, 12]:
            rt = base + timedelta(hours=h)
            ftimes = [(rt + timedelta(hours=fh), 59.0 + h * 0.4) for fh in range(24, 168, 12)]
            _insert_forecast(conn, station, "MEX", rt, ftimes)

    def test_no_leakage_across_day(self, conn):
        """Walk 5-min steps through a full day; no future data at any step."""
        self._seed_full_day(conn, "KJFK", "2026-03-25")
        base = _utc(2026, 3, 25, 0, 0)

        for step_min in range(0, 24 * 60, 5):
            as_of = base + timedelta(minutes=step_min)
            ts = _ts(as_of)

            # METAR: no obs after as_of
            df = get_metar_at(conn, "KJFK", as_of)
            if not df.empty:
                assert max(df["obs_time"]) <= ts

            # Forecasts: all runtimes must be available by as_of
            for model in ["LAV", "NBS", "GFS", "MEX"]:
                df = get_forecasts_at(conn, "KJFK", model, as_of, lag_adjusted=True)
                lag = DISSEMINATION_LAG[model]
                for _, row in df.iterrows():
                    rt = datetime.fromisoformat(row["runtime"].replace("Z", "+00:00"))
                    assert rt + lag <= as_of

    def test_running_max_monotonic(self, conn):
        """Running max can only increase or stay flat over time."""
        self._seed_full_day(conn, "KJFK", "2026-03-25")
        base = _utc(2026, 3, 25, 0, 0)

        prev_max = None
        for step_min in range(0, 24 * 60, 5):
            as_of = base + timedelta(minutes=step_min)
            current_max = get_metar_running_max_at(conn, "KJFK", "2026-03-25", as_of)
            if current_max is not None and prev_max is not None:
                assert current_max >= prev_max, \
                    f"Running max decreased at {as_of}: {prev_max} -> {current_max}"
            if current_max is not None:
                prev_max = current_max

    def test_forecast_count_grows(self, conn):
        """Number of available forecast rows should only increase over time."""
        self._seed_full_day(conn, "KJFK", "2026-03-25")
        base = _utc(2026, 3, 25, 0, 0)

        for model in ["LAV", "NBS", "GFS", "MEX"]:
            prev_count = 0
            for step_min in range(0, 24 * 60, 30):
                as_of = base + timedelta(minutes=step_min)
                df = get_forecasts_at(conn, "KJFK", model, as_of, lag_adjusted=True)
                assert len(df) >= prev_count, \
                    f"{model} count decreased at {as_of}: {prev_count} -> {len(df)}"
                prev_count = len(df)


# ═══════════════════════════════════════════════════════════════════════════
#  10. Store / fetch helpers (simple CRUD)
# ═══════════════════════════════════════════════════════════════════════════

class TestStoreAndFetch:

    def test_store_and_fetch_metar(self, conn):
        """Round-trip: store METAR then fetch."""
        df = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T10:00:00Z", "temp_f": 55.0},
            {"station": "KJFK", "obs_time": "2026-03-25T11:00:00Z", "temp_f": 58.0},
        ])
        n = store_metar(conn, df)
        conn.commit()
        assert n == 2

        result = fetch_metar(conn, "KJFK")
        assert len(result) == 2

        result = fetch_metar(conn, "KJFK", "2026-03-25")
        assert len(result) == 2

        result = fetch_metar(conn, "KJFK", "2026-03-24")
        assert len(result) == 0

    def test_store_and_fetch_forecasts(self, conn):
        df = pd.DataFrame([
            {"station": "KJFK", "model": "LAV", "runtime": "2026-03-25T06:00:00Z",
             "ftime": "2026-03-25T12:00:00Z", "tmp": 55.0},
            {"station": "KJFK", "model": "LAV", "runtime": "2026-03-25T06:00:00Z",
             "ftime": "2026-03-25T13:00:00Z", "tmp": 57.0},
        ])
        n = store_forecasts(conn, df)
        conn.commit()
        assert n == 2

        result = fetch_forecasts(conn, "KJFK")
        assert len(result) == 2

        result = fetch_forecasts(conn, "KJFK", model="LAV")
        assert len(result) == 2

        result = fetch_forecasts(conn, "KJFK", model="NBS")
        assert len(result) == 0

    def test_upsert_updates_existing(self, conn):
        """Upserting with same PK should overwrite the row."""
        df1 = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T10:00:00Z", "temp_f": 55.0},
        ])
        store_metar(conn, df1)
        conn.commit()

        df2 = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T10:00:00Z", "temp_f": 56.0},
        ])
        store_metar(conn, df2)
        conn.commit()

        result = fetch_metar(conn, "KJFK")
        assert len(result) == 1
        assert result.iloc[0]["temp_f"] == 56.0

    def test_empty_df_noop(self, conn):
        """Storing an empty DataFrame should return 0 and not fail."""
        assert store_metar(conn, pd.DataFrame()) == 0
        assert store_forecasts(conn, pd.DataFrame()) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  11. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_null_temp_excluded_from_max(self, conn):
        """Obs with temp_f=NULL should not contribute to running max."""
        df = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T10:00:00Z", "temp_f": None,
             "raw_metar": "missing temp"},
            {"station": "KJFK", "obs_time": "2026-03-25T11:00:00Z", "temp_f": 55.0,
             "raw_metar": "ok"},
        ])
        store_metar(conn, df)
        conn.commit()

        result = get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 12, 0))
        assert result == 55.0

    def test_no_data_returns_empty(self, conn):
        """Querying an empty DB returns empty DataFrames / None."""
        assert get_metar_at(conn, "KJFK", _utc(2026, 3, 25, 12, 0)).empty
        assert get_metar_running_max_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 12, 0)) is None
        assert get_forecasts_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 12, 0)).empty
        assert get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 25, 12, 0)).empty

    def test_string_as_of_works(self, conn):
        """as_of can be passed as an ISO string instead of datetime."""
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 0), 55.0)

        df = get_metar_at(conn, "KJFK", "2026-03-25T10:00:00Z")
        assert len(df) == 1

    def test_forecast_multiple_ftimes_per_run(self, conn):
        """A model run has many ftime steps; all should be returned."""
        runtime = _utc(2026, 3, 25, 6, 0)
        ftimes = [(runtime + timedelta(hours=h), 55.0 + h) for h in range(1, 25)]
        _insert_forecast(conn, "KJFK", "LAV", runtime, ftimes)

        df = get_latest_forecast_at(conn, "KJFK", "LAV", _utc(2026, 3, 26, 0, 0), lag_adjusted=False)
        assert len(df) == 24

    def test_snapshot_empty_db(self, conn):
        """Snapshot on empty DB should return None max and empty forecasts."""
        snap = get_snapshot_at(conn, "KJFK", "2026-03-25", _utc(2026, 3, 25, 12, 0))
        assert snap["metar_running_max"] is None
        assert snap["metar_obs_count"] == 0
        assert snap["latest_lamp"].empty
        assert snap["latest_nbm"].empty
        assert snap["latest_gfs"].empty
        assert snap["latest_mex"].empty


# ═══════════════════════════════════════════════════════════════════════════
#  12. Availability cutoff unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAvailabilityCutoff:
    """Verify the internal _availability_cutoff calculation."""

    def test_lamp_cutoff(self):
        as_of = _utc(2026, 3, 25, 10, 30)
        cutoff = _availability_cutoff(as_of, "LAV")
        assert cutoff == "2026-03-25T10:00:00Z"

    def test_nbm_cutoff(self):
        as_of = _utc(2026, 3, 25, 13, 0)
        cutoff = _availability_cutoff(as_of, "NBS")
        assert cutoff == "2026-03-25T12:00:00Z"

    def test_gfs_cutoff(self):
        as_of = _utc(2026, 3, 25, 10, 0)
        cutoff = _availability_cutoff(as_of, "GFS")
        assert cutoff == "2026-03-25T06:00:00Z"

    def test_mex_cutoff(self):
        as_of = _utc(2026, 3, 25, 11, 0)
        cutoff = _availability_cutoff(as_of, "MEX")
        assert cutoff == "2026-03-25T06:00:00Z"

    def test_string_input(self):
        cutoff = _availability_cutoff("2026-03-25T10:30:00Z", "LAV")
        assert cutoff == "2026-03-25T10:00:00Z"

    def test_unknown_model_no_lag(self):
        """Unknown model has zero lag."""
        as_of = _utc(2026, 3, 25, 10, 0)
        cutoff = _availability_cutoff(as_of, "UNKNOWN")
        assert cutoff == "2026-03-25T10:00:00Z"


# ═══════════════════════════════════════════════════════════════════════════
#  13. Realistic multi-station backtesting day
# ═══════════════════════════════════════════════════════════════════════════

class TestRealisticMultiStation:
    """Simulate backtesting across multiple stations and models for one day."""

    STATIONS = ["KJFK", "KORD", "KHOU"]

    def _seed_stations(self, conn, date_str: str):
        base = datetime.fromisoformat(date_str + "T00:00:00+00:00")
        for stn_i, stn in enumerate(self.STATIONS):
            base_temp = 40.0 + stn_i * 10  # different base temps per city

            # METAR every 30 min
            for m in range(0, 24 * 60, 30):
                t = base + timedelta(minutes=m)
                temp = base_temp + 10 * max(0, 1 - abs(t.hour - 15) / 9)
                _insert_metar(conn, stn, t, round(temp, 1))

            # LAMP hourly
            for h in range(0, 24):
                rt = base + timedelta(hours=h)
                _insert_forecast(conn, stn, "LAV", rt,
                                 [(rt + timedelta(hours=f), base_temp + 5 + h * 0.2) for f in range(1, 6)])

            # NBM hourly
            for h in range(0, 24):
                rt = base + timedelta(hours=h)
                _insert_forecast(conn, stn, "NBS", rt,
                                 [(rt + timedelta(hours=f), base_temp + 7 + h * 0.2) for f in range(3, 18, 3)])

            # GFS 4x daily
            for h in [0, 6, 12, 18]:
                rt = base + timedelta(hours=h)
                _insert_forecast(conn, stn, "GFS", rt,
                                 [(rt + timedelta(hours=f), base_temp + 8 + h * 0.3) for f in range(6, 48, 3)])

    def test_no_cross_station_leakage(self, conn):
        """At every time step, each station's data is independent."""
        self._seed_stations(conn, "2026-03-25")
        base = _utc(2026, 3, 25, 0, 0)

        for step in range(0, 24 * 60, 60):
            as_of = base + timedelta(minutes=step)
            for stn in self.STATIONS:
                df = get_metar_at(conn, stn, as_of, date_str="2026-03-25")
                if not df.empty:
                    assert (df["station"] == stn).all()

                for model in ["LAV", "NBS", "GFS"]:
                    fdf = get_forecasts_at(conn, stn, model, as_of, lag_adjusted=True)
                    if not fdf.empty:
                        assert (fdf["station"] == stn).all()
                        assert (fdf["model"] == model).all()

    def test_snapshots_consistent(self, conn):
        """Snapshots for each station should have mutually exclusive data."""
        self._seed_stations(conn, "2026-03-25")
        as_of = _utc(2026, 3, 25, 18, 0)

        snaps = {}
        for stn in self.STATIONS:
            snaps[stn] = get_snapshot_at(conn, stn, "2026-03-25", as_of)

        # Each station has different METAR data
        maxes = [snaps[s]["metar_running_max"] for s in self.STATIONS]
        assert len(set(maxes)) == len(maxes), "Stations should have different running maxes"

    def test_all_stations_have_data(self, conn):
        """End of day, all stations should have metar and forecast data."""
        self._seed_stations(conn, "2026-03-25")
        as_of = _utc(2026, 3, 25, 23, 0)

        for stn in self.STATIONS:
            snap = get_snapshot_at(conn, stn, "2026-03-25", as_of, lag_adjusted=True)
            assert snap["metar_running_max"] is not None, f"{stn}: no METAR max"
            assert snap["metar_obs_count"] > 0, f"{stn}: no METAR obs"
            assert not snap["latest_lamp"].empty, f"{stn}: no LAMP"
            assert not snap["latest_nbm"].empty, f"{stn}: no NBM"
            assert not snap["latest_gfs"].empty, f"{stn}: no GFS"


# ═══════════════════════════════════════════════════════════════════════════
#  14. Timestamp format consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestTimestampFormat:
    """All timestamps in the DB should be consistent ISO-8601 UTC."""

    def test_metar_timestamps_are_iso_utc(self, conn):
        _insert_metar(conn, "KJFK", _utc(2026, 3, 25, 10, 5, 30), 55.0)
        df = fetch_metar(conn, "KJFK")
        ts = df.iloc[0]["obs_time"]
        assert ts.endswith("Z"), f"obs_time not UTC: {ts}"
        # Verify parseable
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_forecast_timestamps_are_iso_utc(self, conn):
        _insert_forecast(conn, "KJFK", "LAV",
                         _utc(2026, 3, 25, 6, 0),
                         [(_utc(2026, 3, 25, 12, 0), 55.0)])
        df = fetch_forecasts(conn, "KJFK")
        for col in ["runtime", "ftime"]:
            val = df.iloc[0][col]
            assert val.endswith("Z"), f"{col} not UTC: {val}"
            datetime.fromisoformat(val.replace("Z", "+00:00"))

    def test_timestamps_sort_correctly(self, conn):
        """ISO-8601 strings should sort lexicographically = chronologically."""
        times = [
            _utc(2026, 3, 25, 9, 0),
            _utc(2026, 3, 25, 10, 0),
            _utc(2026, 3, 25, 11, 0),
        ]
        for t in reversed(times):
            _insert_metar(conn, "KJFK", t, 55.0)

        df = get_metar_at(conn, "KJFK", _utc(2026, 3, 26, 0, 0))
        assert list(df["obs_time"]) == [_ts(t) for t in times]


# ═══════════════════════════════════════════════════════════════════════════
#  15. Kalshi price helpers
# ═══════════════════════════════════════════════════════════════════════════

def _insert_kalshi(conn, ticker: str, ts: datetime, yes_bid: int, yes_ask: int):
    """Insert a single Kalshi price snapshot."""
    df = pd.DataFrame([{
        "ticker": ticker,
        "ts": _ts(ts),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask,
        "no_ask": 100 - yes_bid,
        "volume": 100,
        "open_interest": 50,
    }])
    store_kalshi(conn, df)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
#  16. Kalshi point-in-time
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiPointInTime:
    """Verify that Kalshi price queries respect the as_of boundary."""

    def test_sees_only_past_snapshots(self, conn):
        """Query at T2 must see T1's snapshot but not T3's."""
        t1 = _utc(2026, 3, 25, 10, 0)
        t2 = _utc(2026, 3, 25, 10, 5)
        t3 = _utc(2026, 3, 25, 10, 10)

        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t1, 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t2, 42, 47)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t3, 38, 43)

        df = get_kalshi_prices_at(conn, t2, ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1
        assert df.iloc[0]["yes_bid"] == 42  # latest at t2 is t2's own snapshot
        assert df.iloc[0]["ts"] == _ts(t2)

    def test_returns_latest_per_contract(self, conn):
        """With multiple contracts, returns the latest snapshot for each."""
        t1 = _utc(2026, 3, 25, 10, 0)
        t2 = _utc(2026, 3, 25, 10, 5)

        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t1, 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t2, 42, 47)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", t1, 20, 25)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", t2, 22, 27)

        df = get_kalshi_prices_at(conn, t2)
        assert len(df) == 2
        tickers = set(df["ticker"])
        assert tickers == {"KXHIGHNY-26MAR25-T55", "KXHIGHNY-26MAR25-T60"}
        # Both should be from t2
        assert all(df["ts"] == _ts(t2))

    def test_sees_nothing_before_first_snapshot(self, conn):
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t1, 40, 45)

        df = get_kalshi_prices_at(conn, _utc(2026, 3, 25, 9, 59))
        assert len(df) == 0

    def test_exact_timestamp_included(self, conn):
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t1, 40, 45)

        df = get_kalshi_prices_at(conn, t1, ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1

    def test_ticker_isolation(self, conn):
        """Filtering by ticker returns only that contract."""
        t1 = _utc(2026, 3, 25, 10, 0)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t1, 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", t1, 20, 25)

        df = get_kalshi_prices_at(conn, t1, ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "KXHIGHNY-26MAR25-T55"

    def test_string_as_of_works(self, conn):
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 40, 45)
        df = get_kalshi_prices_at(conn, "2026-03-25T10:00:00Z", ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  17. Kalshi price history
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiHistory:

    def test_full_history_up_to_as_of(self, conn):
        """get_kalshi_history_at returns all snapshots up to as_of."""
        times = [_utc(2026, 3, 25, 10, m) for m in range(0, 30, 5)]
        for i, t in enumerate(times):
            _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t, 40 + i, 45 + i)

        # Query at t[3] — should see 4 snapshots (t[0] through t[3])
        df = get_kalshi_history_at(conn, "KXHIGHNY-26MAR25-T55", times[3])
        assert len(df) == 4
        assert df.iloc[0]["ts"] == _ts(times[0])
        assert df.iloc[-1]["ts"] == _ts(times[3])

    def test_history_empty_before_data(self, conn):
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 40, 45)
        df = get_kalshi_history_at(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 9, 0))
        assert len(df) == 0

    def test_history_excludes_other_tickers(self, conn):
        t = _utc(2026, 3, 25, 10, 0)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t, 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", t, 20, 25)

        df = get_kalshi_history_at(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 26, 0, 0))
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "KXHIGHNY-26MAR25-T55"


# ═══════════════════════════════════════════════════════════════════════════
#  18. Kalshi store / fetch CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiStoreAndFetch:

    def test_round_trip(self, conn):
        df = pd.DataFrame([
            {"ticker": "KXHIGHNY-26MAR25-T55", "ts": "2026-03-25T10:00:00Z",
             "yes_bid": 40, "yes_ask": 45, "volume": 100},
            {"ticker": "KXHIGHNY-26MAR25-T55", "ts": "2026-03-25T10:05:00Z",
             "yes_bid": 42, "yes_ask": 47, "volume": 110},
        ])
        n = store_kalshi(conn, df)
        conn.commit()
        assert n == 2

        result = fetch_kalshi(conn, ticker="KXHIGHNY-26MAR25-T55")
        assert len(result) == 2

    def test_fetch_by_date(self, conn):
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 26, 10, 0), 42, 47)

        result = fetch_kalshi(conn, date_str="2026-03-25")
        assert len(result) == 1
        assert result.iloc[0]["ts"].startswith("2026-03-25")

    def test_fetch_all(self, conn):
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", _utc(2026, 3, 25, 10, 0), 20, 25)

        result = fetch_kalshi(conn)
        assert len(result) == 2

    def test_upsert_same_pk(self, conn):
        """Same (ticker, ts) should overwrite."""
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 40, 45)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 10, 0), 42, 47)

        result = fetch_kalshi(conn, ticker="KXHIGHNY-26MAR25-T55")
        assert len(result) == 1
        assert result.iloc[0]["yes_bid"] == 42

    def test_empty_noop(self, conn):
        assert store_kalshi(conn, pd.DataFrame()) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  19. Kalshi no-future-leakage
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiNoFutureLeakage:

    def test_no_future_prices(self, conn):
        """Walk through a day of 5-min Kalshi snapshots; no ts > as_of."""
        base = _utc(2026, 3, 25, 9, 30)  # market open ~9:30 ET = 14:30 UTC approx
        tickers = [f"KXHIGHNY-26MAR25-T{t}" for t in [50, 55, 60, 65, 70]]

        # Insert snapshots every 5 min for 8 hours across 5 contracts
        for m in range(0, 8 * 60, 5):
            t = base + timedelta(minutes=m)
            for ticker in tickers:
                _insert_kalshi(conn, ticker, t, 30 + m % 20, 35 + m % 20)

        # Check every 15 minutes
        for m in range(0, 8 * 60, 15):
            check_time = base + timedelta(minutes=m)
            df = get_kalshi_prices_at(conn, check_time)
            if not df.empty:
                assert max(df["ts"]) <= _ts(check_time), \
                    f"Future Kalshi data at {check_time}: {max(df['ts'])}"

    def test_prices_available_instantly(self, conn):
        """Kalshi prices have no dissemination lag — available immediately."""
        t = _utc(2026, 3, 25, 14, 30)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t, 40, 45)

        # Available at exact timestamp (no lag)
        df = get_kalshi_prices_at(conn, t, ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1

    def test_price_evolution_visible_progressively(self, conn):
        """As time advances, we see progressively updated prices."""
        times_prices = [
            (_utc(2026, 3, 25, 14, 0), 30),
            (_utc(2026, 3, 25, 14, 5), 35),
            (_utc(2026, 3, 25, 14, 10), 40),
            (_utc(2026, 3, 25, 14, 15), 45),
        ]
        for t, bid in times_prices:
            _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", t, bid, bid + 5)

        # At 14:07, latest visible is the 14:05 snapshot
        df = get_kalshi_prices_at(conn, _utc(2026, 3, 25, 14, 7), ticker="KXHIGHNY-26MAR25-T55")
        assert len(df) == 1
        assert df.iloc[0]["yes_bid"] == 35

        # At 14:12, latest is the 14:10 snapshot
        df = get_kalshi_prices_at(conn, _utc(2026, 3, 25, 14, 12), ticker="KXHIGHNY-26MAR25-T55")
        assert df.iloc[0]["yes_bid"] == 40


# ═══════════════════════════════════════════════════════════════════════════
#  20. Kalshi in snapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiInSnapshot:
    """get_snapshot_at should include Kalshi prices."""

    def test_snapshot_includes_kalshi(self, conn):
        _insert_metar(conn, "KNYC", _utc(2026, 3, 25, 14, 0), 62.0)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T55", _utc(2026, 3, 25, 14, 0), 85, 90)
        _insert_kalshi(conn, "KXHIGHNY-26MAR25-T60", _utc(2026, 3, 25, 14, 0), 55, 60)

        snap = get_snapshot_at(conn, "KNYC", "2026-03-25", _utc(2026, 3, 25, 14, 5))
        assert "kalshi_prices" in snap
        assert len(snap["kalshi_prices"]) == 2

    def test_snapshot_kalshi_empty_before_data(self, conn):
        snap = get_snapshot_at(conn, "KNYC", "2026-03-25", _utc(2026, 3, 25, 12, 0))
        assert snap["kalshi_prices"].empty
