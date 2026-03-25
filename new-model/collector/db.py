"""
SQLite storage for the EMOS data collector.

Tables
──────
  metar_obs        — raw METAR observations (5-min polling)
  iem_forecasts    — LAMP / NBM / GFS MOS forecasts from IEM archive
  kalshi_prices    — Kalshi market snapshots (5-min polling)
  daily_summary    — per-station per-day derived max-temp forecasts + actuals

WAL mode for concurrent reads.  INSERT OR IGNORE on (station, model, runtime,
ftime) so re-fetching the same model run is a no-op.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from collector.config import DB_PATH

_SCHEMA = """
-- Raw METAR observations from aviationweather.gov
CREATE TABLE IF NOT EXISTS metar_obs (
    station     TEXT    NOT NULL,   -- ICAO code (e.g. KJFK)
    obs_time    TEXT    NOT NULL,   -- UTC ISO-8601 timestamp
    temp_f      REAL,              -- temperature in °F
    dew_point_f REAL,              -- dew point in °F
    wind_speed_kt REAL,            -- wind speed in knots
    wind_dir    INTEGER,           -- wind direction in degrees
    raw_metar   TEXT,              -- full METAR string for debugging
    PRIMARY KEY (station, obs_time)
);

CREATE INDEX IF NOT EXISTS idx_metar_station_date
    ON metar_obs (station, substr(obs_time, 1, 10));

-- IEM MOS forecasts (LAMP, NBM, GFS MOS, GFS Extended MOS)
-- Stores the union of fields across all models; unused fields are NULL.
CREATE TABLE IF NOT EXISTS iem_forecasts (
    station     TEXT    NOT NULL,   -- ICAO code
    model       TEXT    NOT NULL,   -- GFS, LAV (LAMP), NBS (NBM), MEX
    runtime     TEXT    NOT NULL,   -- model initialization time (UTC ISO)
    ftime       TEXT    NOT NULL,   -- forecast valid time (UTC ISO)
    tmp         REAL,              -- temperature at ftime (°F)
    dpt         REAL,              -- dew point at ftime (°F)
    n_x         REAL,              -- GFS MOS / MEX: 12-hr min or max (°F)
    txn         REAL,              -- NBM: 18-hr max or min (°F)
    xnd         REAL,              -- NBM: std dev of max/min (°F)
    tsd         REAL,              -- NBM: temperature std dev (°F)
    wsp         REAL,              -- wind speed (kt)
    sky         REAL,              -- sky cover (%)
    p06         REAL,              -- 6-hr precipitation probability (%)
    PRIMARY KEY (station, model, runtime, ftime)
);

CREATE INDEX IF NOT EXISTS idx_iem_station_model
    ON iem_forecasts (station, model);
CREATE INDEX IF NOT EXISTS idx_iem_ftime
    ON iem_forecasts (ftime);

-- Kalshi market price snapshots (polled every 5 minutes)
CREATE TABLE IF NOT EXISTS kalshi_prices (
    ticker      TEXT    NOT NULL,   -- contract ticker (e.g. KXHIGHNY-26MAR25-T55)
    ts          TEXT    NOT NULL,   -- UTC ISO-8601 fetch timestamp
    yes_bid     INTEGER,           -- best yes bid (cents)
    yes_ask     INTEGER,           -- best yes ask (cents)
    no_bid      INTEGER,           -- best no bid (cents)
    no_ask      INTEGER,           -- best no ask (cents)
    volume      INTEGER,           -- cumulative volume
    open_interest INTEGER,         -- open interest
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_ts
    ON kalshi_prices (ts);
CREATE INDEX IF NOT EXISTS idx_kalshi_ticker_ts
    ON kalshi_prices (ticker, ts);

-- Daily summary: derived max-temp forecasts from each source + observed actual.
-- Populated by a post-processing step that reads metar_obs + iem_forecasts.
CREATE TABLE IF NOT EXISTS daily_summary (
    station       TEXT NOT NULL,    -- ICAO code
    target_date   TEXT NOT NULL,    -- YYYY-MM-DD
    f_nbm         REAL,            -- NBM max-temp forecast (°F)
    f_nbm_sd      REAL,            -- NBM max-temp std dev
    f_gfs_mos     REAL,            -- GFS MOS max-temp forecast (°F)
    f_lamp_max    REAL,            -- max of LAMP hourly temps (°F)
    f_ecmwf       REAL,            -- ECMWF forecast (from Open-Meteo)
    ensemble_spread REAL,          -- spread across models
    obs_max_f     REAL,            -- observed daily max from METAR (°F)
    obs_count     INTEGER,         -- number of METAR obs that day
    updated_at    TEXT,            -- last update timestamp
    PRIMARY KEY (station, target_date)
);
"""


@contextmanager
def get_db(path: Path | str = DB_PATH):
    """Open (or create) the collector database as a context manager."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE rows from df into table. Returns count written."""
    if df.empty:
        return 0

    cur = conn.execute(f"PRAGMA table_info({table})")
    table_cols = [row[1] for row in cur.fetchall()]
    write_cols = [c for c in table_cols if c in df.columns]
    if not write_cols:
        return 0

    chunk = df[write_cols].copy()
    for col in chunk.columns:
        if pd.api.types.is_datetime64_any_dtype(chunk[col]):
            chunk[col] = chunk[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif chunk[col].dtype == object:
            sample = chunk[col].dropna()
            if not sample.empty and hasattr(sample.iloc[0], "isoformat"):
                chunk[col] = chunk[col].apply(
                    lambda v: v.isoformat() if hasattr(v, "isoformat") else v
                )

    placeholders = ", ".join("?" * len(write_cols))
    col_names = ", ".join(write_cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
    rows = [tuple(r) for r in chunk.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


def read_df(
    conn: sqlite3.Connection,
    table: str,
    where: str | None = None,
    params: tuple | None = None,
) -> pd.DataFrame:
    """Read all (or filtered) rows from a table into a DataFrame."""
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return pd.read_sql(sql, conn, params=params)
