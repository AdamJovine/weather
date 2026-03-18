"""
src/db.py

SQLite storage for all weather, forecast, climate, and Kalshi price data.

Schema
──────
  weather_daily    (city, date)      — NOAA TMAX observations
  forecasts_daily  (city, date)      — OpenMeteo GFS/ECMWF/ICON forecasts
  gefs_spread      (city, date)      — GEFS 31-member ensemble spread
  climate_monthly  (year, month)     — AO/NAO/ONI/PNA/PDO monthly indices
  mjo_daily        (date)            — BOM MJO RMM amplitude + phase
  kalshi_markets   (ticker)          — settled Kalshi weather markets
  kalshi_candles   (ticker, ts)      — hourly candlestick prices

Guarantees
──────────
  - WAL mode: concurrent reads never block writes (safe for run_live.py +
    download scripts running at the same time)
  - INSERT OR REPLACE: PRIMARY KEY dedup enforced at write time — no
    drop_duplicates needed before writing
  - All writes are transactional: a failed upsert leaves the file untouched

Typical usage
─────────────
Write (download scripts):

    from src.db import get_db, upsert_df
    with get_db() as conn:
        upsert_df(conn, "weather_daily", df)

Read (run_live.py, scripts):

    from src.db import get_db, read_df
    with get_db() as conn:
        hist = read_df(conn, "weather_daily")

One-time migration from existing CSVs:

    python scripts/migrate_to_db.py
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

from src.app_config import cfg as _cfg

DB_PATH = Path(_cfg.paths.db)

# (table, date_sql, max_age_days, label)
_f = _cfg.freshness
_FRESHNESS_CHECKS = [
    ("weather_daily",   "MAX(date)",                           _f.weather_daily,   "NOAA TMAX"),
    ("forecasts_daily", "MAX(date)",                           _f.forecasts_daily, "OpenMeteo GFS"),
    ("gefs_spread",     "MAX(date)",                           _f.gefs_spread,     "GEFS spread"),
    ("nws_forecasts",   "MAX(target_date)",                    _f.nws_forecasts,   "NWS forecasts"),
    ("mjo_daily",       "MAX(date)",                           _f.mjo_daily,       "MJO"),
    ("climate_monthly", "MAX(year || printf('%02d', month))",  _f.climate_monthly, "Climate indices"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_daily (
    city    TEXT NOT NULL,
    date    TEXT NOT NULL,
    station TEXT,
    tmax    REAL,
    PRIMARY KEY (city, date)
);

CREATE TABLE IF NOT EXISTS forecasts_daily (
    city                TEXT NOT NULL,
    date                TEXT NOT NULL,
    forecast_high_gfs   REAL,
    forecast_high_ecmwf REAL,
    ensemble_spread     REAL,
    ecmwf_minus_gfs     REAL,
    precip_forecast     REAL,
    temp_850hpa         REAL,
    shortwave_radiation REAL,
    dew_point_max       REAL,
    PRIMARY KEY (city, date)
);

CREATE TABLE IF NOT EXISTS gefs_spread (
    city        TEXT NOT NULL,
    date        TEXT NOT NULL,
    gefs_spread REAL,
    PRIMARY KEY (city, date)
);

CREATE TABLE IF NOT EXISTS climate_monthly (
    year      INTEGER NOT NULL,
    month     INTEGER NOT NULL,
    ao_index  REAL,
    nao_index REAL,
    oni       REAL,
    pna_index REAL,
    pdo_index REAL,
    PRIMARY KEY (year, month)
);

CREATE TABLE IF NOT EXISTS mjo_daily (
    date          TEXT NOT NULL,
    mjo_amplitude REAL,
    mjo_phase_sin REAL,
    mjo_phase_cos REAL,
    PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS kalshi_markets (
    ticker          TEXT NOT NULL,
    title           TEXT,
    series          TEXT,
    city            TEXT,
    settlement_date TEXT,
    PRIMARY KEY (ticker)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_markets_series ON kalshi_markets (series);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_city   ON kalshi_markets (city);

CREATE TABLE IF NOT EXISTS kalshi_candles (
    ticker        TEXT    NOT NULL,
    ts            INTEGER NOT NULL,
    close_dollars REAL,
    volume        INTEGER,
    high_dollars  REAL,
    low_dollars   REAL,
    PRIMARY KEY (ticker, ts),
    FOREIGN KEY (ticker) REFERENCES kalshi_markets (ticker)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_candles_ticker ON kalshi_candles (ticker);

CREATE TABLE IF NOT EXISTS kalshi_no_candles (
    ticker     TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    PRIMARY KEY (ticker)
);

CREATE TABLE IF NOT EXISTS nws_forecasts (
    city          TEXT NOT NULL,
    target_date   TEXT NOT NULL,
    forecast_high REAL,
    nbm_high      REAL,
    PRIMARY KEY (city, target_date)
);
"""


# ─── Connection ───────────────────────────────────────────────────────────────

@contextmanager
def get_db(path: Path | str = DB_PATH):
    """
    Open (or create) the SQLite database as a context manager.

    On entry: opens WAL-mode connection, initialises schema.
    On success: commits and closes.
    On exception: rolls back and closes.

    Usage:
        with get_db() as conn:
            upsert_df(conn, "weather_daily", df)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add new columns to existing tables without recreating them."""
    new_cols = [
        ("forecasts_daily", "temp_850hpa",         "REAL"),
        ("forecasts_daily", "shortwave_radiation",  "REAL"),
        ("forecasts_daily", "dew_point_max",        "REAL"),
        ("kalshi_candles",  "high_dollars",         "REAL"),
        ("kalshi_candles",  "low_dollars",          "REAL"),
    ]
    for table, col, dtype in new_cols:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass  # column already exists


# ─── Write ────────────────────────────────────────────────────────────────────

def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """
    Insert or replace all rows of df into table.

    - Only columns that exist in both df and the table are written; extras
      in df are silently ignored.
    - INSERT OR REPLACE enforces the PRIMARY KEY — duplicate keys overwrite
      the existing row (identical to drop_duplicates + write, but atomic).
    - Date objects and datetime64 values are normalised to YYYY-MM-DD strings.

    Returns the number of rows written.
    """
    if df.empty:
        return 0

    cur = conn.execute(f"PRAGMA table_info({table})")
    table_cols = [row[1] for row in cur.fetchall()]
    if not table_cols:
        raise ValueError(f"Table '{table}' does not exist in the database.")

    write_cols = [c for c in table_cols if c in df.columns]
    if not write_cols:
        raise ValueError(
            f"DataFrame has no columns matching table '{table}'. "
            f"Table expects: {table_cols}.  DataFrame has: {list(df.columns)}."
        )

    chunk = df[write_cols].copy()

    # Normalise dates → ISO strings so SQLite stores them uniformly
    for col in chunk.columns:
        if pd.api.types.is_datetime64_any_dtype(chunk[col]):
            chunk[col] = chunk[col].dt.strftime("%Y-%m-%d")
        elif chunk[col].dtype == object:
            sample = chunk[col].dropna()
            if not sample.empty and hasattr(sample.iloc[0], "isoformat"):
                chunk[col] = chunk[col].apply(
                    lambda v: v.isoformat() if hasattr(v, "isoformat") else v
                )

    placeholders = ", ".join("?" * len(write_cols))
    col_names    = ", ".join(write_cols)
    sql  = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
    rows = [tuple(r) for r in chunk.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


# ─── Read ─────────────────────────────────────────────────────────────────────

def read_df(
    conn: sqlite3.Connection,
    table: str,
    where: Optional[str] = None,
    params: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Read all (or filtered) rows from a table into a DataFrame.

    Examples:
        hist    = read_df(conn, "weather_daily")
        candles = read_df(conn, "kalshi_candles",
                          "ticker = ?", ("KXHIGHNY-26MAR15-T54",))
    """
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return pd.read_sql(sql, conn, params=params)


# ─── Freshness check ─────────────────────────────────────────────────────────

def check_freshness_db(conn: sqlite3.Connection) -> list:
    """
    Check whether each DB table has been updated recently enough.
    Returns a list of ValidationResult objects (imported lazily to avoid
    circular imports with data_contracts).

    Example (run_live.py):
        with get_db() as conn:
            for r in check_freshness_db(conn):
                print(r.summary())
    """
    from datetime import date as _date
    from src.data_contracts import ValidationResult

    results = []
    for table, date_sql, max_age_days, label in _FRESHNESS_CHECKS:
        r = ValidationResult(source=f"{label} (freshness)")
        try:
            row = conn.execute(
                f"SELECT {date_sql} FROM {table}"
            ).fetchone()
            latest_str = row[0] if row else None

            if not latest_str:
                r.add_error(f"table '{table}' is empty — run scripts/update_data.py")
            else:
                # climate_monthly returns YYYYMM string; convert to date
                if table == "climate_monthly":
                    latest_str = latest_str[:4] + "-" + latest_str[4:6] + "-01"
                latest   = pd.to_datetime(latest_str).date()
                age_days = (_date.today() - latest).days
                r.stats["latest"] = str(latest)
                r.stats["age"]    = f"{age_days}d"
                if age_days > max_age_days:
                    r.add_error(
                        f"latest row is {age_days}d old "
                        f"(threshold: {max_age_days}d).  "
                        f"Run scripts/update_data.py."
                    )
        except Exception as exc:
            r.add_error(f"query failed: {exc}")
        results.append(r)
    return results

