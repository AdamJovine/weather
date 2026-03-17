"""
src/data_loader_shared.py

Shared DB loading helper used by both run_live.py and backtesting/data_loader.py.

The single public function load_raw_sources() opens one DB connection and loads
all 6 relevant tables, ensuring live trading and backtesting read data identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.db import get_db, read_df as db_read


def load_raw_sources(
    db_path,
    nws_target_date: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Load all raw data sources from the SQLite database in a single connection.

    Parameters
    ----------
    db_path : path-like
        Path to the SQLite database file.
    nws_target_date : str or None
        If given as "YYYY-MM-DD", loads NWS forecasts only for that date (live
        mode).  If None, loads the full NWS history (backtest mode).
    verbose : bool
        If True, prints row counts for each table loaded.

    Returns
    -------
    dict with keys:
        "hist"    : pd.DataFrame  — weather_daily (always a DataFrame)
        "gfs"     : pd.DataFrame or None  — forecasts_daily
        "gefs"    : pd.DataFrame or None  — gefs_spread
        "indices" : pd.DataFrame or None  — climate_monthly
        "mjo"     : pd.DataFrame or None  — mjo_daily
        "nws"     : pd.DataFrame          — nws_forecasts (possibly empty)

    Raises
    ------
    FileNotFoundError
        If db_path does not exist.
    ValueError
        If weather_daily is empty.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run scripts/update_data.py first."
        )

    with get_db(db_path) as conn:
        hist     = db_read(conn, "weather_daily")
        gfs_raw  = db_read(conn, "forecasts_daily")
        gefs_raw = db_read(conn, "gefs_spread")
        idx_raw  = db_read(conn, "climate_monthly")
        mjo_raw  = db_read(conn, "mjo_daily")

        if nws_target_date is not None:
            nws = pd.read_sql(
                "SELECT city, forecast_high, nbm_high, target_date "
                "FROM nws_forecasts WHERE target_date = ?",
                conn,
                params=(nws_target_date,),
            )
        else:
            nws = pd.read_sql(
                "SELECT city, forecast_high, nbm_high, target_date "
                "FROM nws_forecasts",
                conn,
            )

    if verbose:
        print(f"  {'weather_daily':<28}: {len(hist):>7,} rows")

        if gfs_raw.empty:
            print(f"  {'forecasts_daily':<28}: {'EMPTY':>7}  — using climatology as forecast")
        else:
            print(f"  {'forecasts_daily':<28}: {len(gfs_raw):>7,} rows")

        if idx_raw.empty:
            print(f"  {'climate_monthly':<28}: {'EMPTY':>7}  — AO/ONI/NAO features will be absent")
        else:
            print(f"  {'climate_monthly':<28}: {len(idx_raw):>7,} rows")

        if gefs_raw.empty:
            print(f"  {'gefs_spread':<28}: {'EMPTY':>7}  — falling back to ensemble_spread")
        else:
            print(f"  {'gefs_spread':<28}: {len(gefs_raw):>7,} rows")

        if mjo_raw.empty:
            print(f"  {'mjo_daily':<28}: {'EMPTY':>7}  — MJO features will be zero")
        else:
            print(f"  {'mjo_daily':<28}: {len(mjo_raw):>7,} rows")

        if nws_target_date is not None:
            label = f"nws_forecasts ({nws_target_date})"
            print(f"  {label:<28}: {len(nws):>7,} rows")
        else:
            label = "nws_forecasts (history)"
            if nws.empty:
                print(
                    f"  {label:<28}: {len(nws):>7,} rows"
                    "  WARNING: no NWS history — GFS is the forecast baseline"
                )
            else:
                print(f"  {label:<28}: {len(nws):>7,} rows")

    if hist.empty:
        raise ValueError("weather_daily table is empty — run scripts/update_data.py.")

    return {
        "hist":    hist,
        "gfs":     gfs_raw  if not gfs_raw.empty  else None,
        "gefs":    gefs_raw if not gefs_raw.empty else None,
        "indices": idx_raw  if not idx_raw.empty  else None,
        "mjo":     mjo_raw  if not mjo_raw.empty  else None,
        "nws":     nws,
    }
