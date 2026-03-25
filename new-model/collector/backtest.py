"""
Point-in-time data access for backtesting.

Every query takes an ``as_of`` UTC timestamp and returns **only** the data
that would have been available at that moment:

  - METAR:     obs_time <= as_of
  - Forecasts: runtime + dissemination_lag <= as_of
  - Kalshi:    ts <= as_of

The dissemination lag accounts for the delay between a model's initialisation
time (runtime) and when the data is actually downloadable.

Simple wrappers (``store_*`` / ``fetch_*``) make it easy to update the DB
and bulk-read without worrying about SQL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from collector.db import upsert_df

# ── Dissemination lags ──────────────────────────────────────────────────────
# Conservative estimates of the delay between model init (runtime) and the
# moment the data is available for download.

DISSEMINATION_LAG: dict[str, timedelta] = {
    "LAV": timedelta(minutes=30),   # LAMP: ~25-35 min
    "NBS": timedelta(minutes=60),   # NBM: ~45-75 min
    "GFS": timedelta(hours=4),      # GFS MOS: ~3.5-4.5 hr
    "MEX": timedelta(hours=5),      # GFS Ext MOS: ~4-5.5 hr
}


def _ts(dt: datetime | str) -> str:
    """Normalise a datetime or ISO string to ``YYYY-MM-DDTHH:MM:SSZ``."""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _availability_cutoff(as_of: datetime | str, model: str) -> str:
    """
    Return the latest ``runtime`` whose data would be available at *as_of*
    after accounting for the model's dissemination lag.
    """
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    lag = DISSEMINATION_LAG.get(model, timedelta(0))
    cutoff = as_of - lag
    return _ts(cutoff)


# ── Point-in-time queries ──────────────────────────────────────────────────

def get_metar_at(
    conn: sqlite3.Connection,
    station: str,
    as_of: datetime | str,
    *,
    date_str: str | None = None,
) -> pd.DataFrame:
    """
    All METAR observations for *station* available at *as_of*.

    Parameters
    ----------
    station : ICAO code
    as_of   : only obs with ``obs_time <= as_of`` are returned
    date_str: optional ``YYYY-MM-DD`` to restrict to a single day
    """
    ts = _ts(as_of)
    if date_str:
        sql = """
            SELECT * FROM metar_obs
            WHERE station = ?
              AND obs_time >= ? || 'T00:00:00Z'
              AND obs_time <= ?
            ORDER BY obs_time
        """
        return pd.read_sql(sql, conn, params=(station, date_str, ts))
    else:
        sql = """
            SELECT * FROM metar_obs
            WHERE station = ? AND obs_time <= ?
            ORDER BY obs_time
        """
        return pd.read_sql(sql, conn, params=(station, ts))


def get_metar_running_max_at(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    as_of: datetime | str,
) -> Optional[float]:
    """
    Running max temperature on *target_date* using only obs available at
    *as_of*.  Returns ``None`` when there are no observations.
    """
    ts = _ts(as_of)
    row = conn.execute(
        """
        SELECT MAX(temp_f) FROM metar_obs
        WHERE station = ?
          AND obs_time >= ? || 'T00:00:00Z'
          AND obs_time <  ? || 'T23:59:60Z'
          AND obs_time <= ?
          AND temp_f IS NOT NULL
        """,
        (station, target_date, target_date, ts),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def get_forecasts_at(
    conn: sqlite3.Connection,
    station: str,
    model: str,
    as_of: datetime | str,
    *,
    lag_adjusted: bool = True,
) -> pd.DataFrame:
    """
    All forecast rows for *station* / *model* whose runtime is available
    at *as_of* (optionally lag-adjusted).
    """
    cutoff = _availability_cutoff(as_of, model) if lag_adjusted else _ts(as_of)
    sql = """
        SELECT * FROM iem_forecasts
        WHERE station = ? AND model = ? AND runtime <= ?
        ORDER BY runtime, ftime
    """
    return pd.read_sql(sql, conn, params=(station, model, cutoff))


def get_latest_forecast_at(
    conn: sqlite3.Connection,
    station: str,
    model: str,
    as_of: datetime | str,
    *,
    lag_adjusted: bool = True,
) -> pd.DataFrame:
    """
    Rows from the *most recent* model run available at *as_of*.
    """
    cutoff = _availability_cutoff(as_of, model) if lag_adjusted else _ts(as_of)
    sql = """
        SELECT * FROM iem_forecasts
        WHERE station = ? AND model = ?
          AND runtime = (
              SELECT MAX(runtime) FROM iem_forecasts
              WHERE station = ? AND model = ? AND runtime <= ?
          )
        ORDER BY ftime
    """
    return pd.read_sql(sql, conn, params=(station, model, station, model, cutoff))


def get_snapshot_at(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    as_of: datetime | str,
    *,
    lag_adjusted: bool = True,
) -> dict:
    """
    Complete data snapshot at *as_of* for backtesting a single station/date.

    Returns
    -------
    dict with keys:
      as_of, station, target_date,
      metar_running_max, metar_obs_count,
      latest_lamp, latest_nbm, latest_gfs, latest_mex
    (each ``latest_*`` is a DataFrame of forecast rows from the most recent run)
    """
    ts = _ts(as_of)
    running_max = get_metar_running_max_at(conn, station, target_date, ts)

    metar_df = get_metar_at(conn, station, ts, date_str=target_date)
    obs_count = len(metar_df)

    result: dict = {
        "as_of": ts,
        "station": station,
        "target_date": target_date,
        "metar_running_max": running_max,
        "metar_obs_count": obs_count,
    }

    for model, key in [("LAV", "latest_lamp"), ("NBS", "latest_nbm"),
                        ("GFS", "latest_gfs"), ("MEX", "latest_mex")]:
        result[key] = get_latest_forecast_at(
            conn, station, model, ts, lag_adjusted=lag_adjusted,
        )

    result["kalshi_prices"] = get_kalshi_prices_at(conn, ts)

    return result


# ── Kalshi point-in-time queries ────────────────────────────────────────────

def get_kalshi_prices_at(
    conn: sqlite3.Connection,
    as_of: datetime | str,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Most recent Kalshi price snapshot for each contract at *as_of*.

    Parameters
    ----------
    as_of  : only snapshots with ``ts <= as_of`` are returned
    ticker : optional filter to a single contract
    """
    ts = _ts(as_of)
    if ticker:
        sql = """
            SELECT * FROM kalshi_prices
            WHERE ticker = ? AND ts = (
                SELECT MAX(ts) FROM kalshi_prices
                WHERE ticker = ? AND ts <= ?
            )
        """
        return pd.read_sql(sql, conn, params=(ticker, ticker, ts))
    else:
        sql = """
            SELECT k.* FROM kalshi_prices k
            INNER JOIN (
                SELECT ticker, MAX(ts) AS max_ts
                FROM kalshi_prices
                WHERE ts <= ?
                GROUP BY ticker
            ) latest ON k.ticker = latest.ticker AND k.ts = latest.max_ts
            ORDER BY k.ticker
        """
        return pd.read_sql(sql, conn, params=(ts,))


def get_kalshi_history_at(
    conn: sqlite3.Connection,
    ticker: str,
    as_of: datetime | str,
) -> pd.DataFrame:
    """All price snapshots for a single contract up to *as_of*."""
    ts = _ts(as_of)
    sql = """
        SELECT * FROM kalshi_prices
        WHERE ticker = ? AND ts <= ?
        ORDER BY ts
    """
    return pd.read_sql(sql, conn, params=(ticker, ts))


# ── Simple DB helpers ──────────────────────────────────────────────────────

def store_metar(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert METAR observations.  Returns rows written."""
    return upsert_df(conn, "metar_obs", df)


def store_forecasts(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert IEM forecast rows.  Returns rows written."""
    return upsert_df(conn, "iem_forecasts", df)


def fetch_metar(
    conn: sqlite3.Connection,
    station: str,
    date_str: str | None = None,
) -> pd.DataFrame:
    """Read all METAR obs for a station, optionally filtered to one day."""
    if date_str:
        return pd.read_sql(
            "SELECT * FROM metar_obs WHERE station = ? "
            "AND obs_time >= ? || 'T00:00:00Z' AND obs_time < ? || 'T23:59:60Z' "
            "ORDER BY obs_time",
            conn, params=(station, date_str, date_str),
        )
    return pd.read_sql(
        "SELECT * FROM metar_obs WHERE station = ? ORDER BY obs_time",
        conn, params=(station,),
    )


def store_kalshi(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert Kalshi price snapshots.  Returns rows written."""
    return upsert_df(conn, "kalshi_prices", df)


def fetch_kalshi(
    conn: sqlite3.Connection,
    ticker: str | None = None,
    date_str: str | None = None,
) -> pd.DataFrame:
    """Read Kalshi price snapshots, optionally filtered by ticker and/or date."""
    sql = "SELECT * FROM kalshi_prices WHERE 1=1"
    params: list = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker)
    if date_str:
        sql += " AND ts >= ? || 'T00:00:00Z' AND ts < ? || 'T23:59:60Z'"
        params.extend([date_str, date_str])
    sql += " ORDER BY ticker, ts"
    return pd.read_sql(sql, conn, params=params)


def fetch_forecasts(
    conn: sqlite3.Connection,
    station: str,
    model: str | None = None,
    date_str: str | None = None,
) -> pd.DataFrame:
    """
    Read forecast rows for a station, optionally filtered by model and/or
    target date (ftime date).
    """
    sql = "SELECT * FROM iem_forecasts WHERE station = ?"
    params: list = [station]
    if model:
        sql += " AND model = ?"
        params.append(model)
    if date_str:
        sql += " AND ftime >= ? || 'T00:00:00Z' AND ftime < ? || 'T23:59:60Z'"
        params.extend([date_str, date_str])
    sql += " ORDER BY model, runtime, ftime"
    return pd.read_sql(sql, conn, params=params)
