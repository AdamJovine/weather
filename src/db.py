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


DB_PATH = Path("data/weather.db")

# (table, date_sql, max_age_days, label)
_FRESHNESS_CHECKS = [
    ("weather_daily",   "MAX(date)",                           3,  "NOAA TMAX"),
    ("forecasts_daily", "MAX(date)",                           2,  "OpenMeteo GFS"),
    ("gefs_spread",     "MAX(date)",                           2,  "GEFS spread"),
    ("mjo_daily",       "MAX(date)",                           5,  "MJO"),
    ("climate_monthly", "MAX(year || printf('%02d', month))",  50, "Climate indices"),
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
    PRIMARY KEY (ticker, ts),
    FOREIGN KEY (ticker) REFERENCES kalshi_markets (ticker)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_candles_ticker ON kalshi_candles (ticker);
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
    conn.commit()


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


# ─── One-time CSV migration ───────────────────────────────────────────────────

def migrate_from_csvs(conn: sqlite3.Connection, data_dir: Path = Path("data")) -> None:
    """
    Read every existing CSV file and upsert its rows into the corresponding
    table.  Safe to run multiple times — INSERT OR REPLACE deduplicates.
    """
    migrations: list[tuple[Path, str | None]] = [
        (data_dir / "historical_tmax.csv",                      "weather_daily"),
        (data_dir / "forecasts/openmeteo_forecast_history.csv", "forecasts_daily"),
        (data_dir / "forecasts/gefs_spread.csv",                "gefs_spread"),
        (data_dir / "climate_indices.csv",                      "climate_monthly"),
        (data_dir / "mjo_indices.csv",                          "mjo_daily"),
        (data_dir / "kalshi_price_history.csv",                 None),   # split
    ]

    for path, table in migrations:
        if not path.exists():
            print(f"  SKIP  {path}  (not found)")
            continue

        df = pd.read_csv(path)

        if path.name == "kalshi_price_history.csv":
            market_cols = ["ticker", "title", "series", "city", "settlement_date"]
            candle_cols = ["ticker", "ts", "close_dollars", "volume"]
            markets_df  = df[market_cols].drop_duplicates(subset=["ticker"])
            candles_df  = df[candle_cols]
            n_m = upsert_df(conn, "kalshi_markets", markets_df)
            n_c = upsert_df(conn, "kalshi_candles", candles_df)
            print(f"  OK    {path.name}  →  kalshi_markets ({n_m} markets) "
                  f"+ kalshi_candles ({n_c} candles)")
        else:
            n = upsert_df(conn, table, df)
            print(f"  OK    {path.name}  →  {table} ({n} rows)")
