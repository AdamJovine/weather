"""
IEM (Iowa Environmental Mesonet) forecast fetchers.

Fetches LAMP, NBM, and GFS MOS data from IEM's archive APIs.
IEM archives these NWS forecast products going back years, giving us
the actual forecasts that were available at each point in time — solving
the look-ahead bias problem for backtesting.

Two endpoints:
  - /api/1/mos.json       — latest model run (real-time collection)
  - /cgi-bin/request/mos.py — bulk historical download with date ranges

Model codes:
  LAV = LAMP (hourly, 1-25h)
  NBS = NBM  (3-hourly, 6-192h, includes uncertainty)
  GFS = GFS MOS MAV (3-hourly, 6-72h)
  MEX = GFS Extended MOS (12-hourly, 24-192h)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from collector.config import ICAO_LIST

log = logging.getLogger(__name__)

IEM_API_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"
IEM_BULK_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"

# IEM /api/1/mos.json accepts at most 6 stations per request.
_IEM_MAX_STATIONS = 6

# Columns to keep from each model (superset across all models).
# Missing columns for a given model will be NaN.
_KEEP_COLS = [
    "station", "model", "runtime", "ftime",
    "tmp", "dpt", "n_x", "txn", "xnd", "tsd", "wsp", "sky", "p06",
]


def _batches(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _parse_iem_json(data) -> pd.DataFrame:
    """
    Parse IEM JSON response into a DataFrame.

    Handles two formats:
      - /api/1/mos.json: {"schema": {"fields": [...]}, "data": [{...}, ...]}
      - /cgi-bin/request/mos.py?format=json: plain list of dicts [{...}, ...]
    """
    if isinstance(data, list):
        # Bulk CGI format: plain list of dicts
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # FastAPI format: schema + data
        schema = data.get("schema", {})
        fields = [f["name"] for f in schema.get("fields", [])]
        rows = data.get("data", [])
        if not fields or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=fields)
    else:
        return pd.DataFrame()

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower()

    # Prefer the UTC timestamp columns the API provides
    if "runtime_utc" in df.columns:
        df["runtime"] = pd.to_datetime(df["runtime_utc"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    elif "runtime" in df.columns:
        df["runtime"] = pd.to_datetime(df["runtime"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    if "ftime_utc" in df.columns:
        df["ftime"] = pd.to_datetime(df["ftime_utc"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    elif "ftime" in df.columns:
        df["ftime"] = pd.to_datetime(df["ftime"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # Coerce numeric columns (IEM sometimes returns them as strings)
    for col in ("tmp", "dpt", "n_x", "txn", "xnd", "tsd", "wsp", "sky", "p06"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only the columns we care about
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    return df


def fetch_latest(model: str, stations: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch the latest model run from IEM for all stations.

    Batches requests in groups of 6 (IEM API limit) with a short
    delay between batches to be polite.

    Parameters
    ----------
    model : str
        One of: GFS, LAV, NBS, MEX
    stations : list[str] | None
        ICAO codes. Defaults to all configured stations.

    Returns
    -------
    DataFrame with columns from _KEEP_COLS.
    """
    if stations is None:
        stations = ICAO_LIST

    all_dfs = []
    for batch in _batches(stations, _IEM_MAX_STATIONS):
        params = [("model", model)]
        for s in batch:
            params.append(("station", s))

        try:
            resp = requests.get(IEM_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("IEM %s fetch failed for %s: %s", model, batch, e)
            continue

        df = _parse_iem_json(data)
        if not df.empty:
            all_dfs.append(df)

        if len(stations) > _IEM_MAX_STATIONS:
            time.sleep(0.5)  # be polite

    if not all_dfs:
        log.warning("IEM %s: no data returned", model)
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    n_stations = df["station"].nunique()
    runtime = df["runtime"].iloc[0] if "runtime" in df.columns else "?"
    log.info("IEM %s: %d rows, %d stations, run=%s", model, len(df), n_stations, runtime)
    return df


def fetch_history(
    model: str,
    start: str,
    end: str,
    stations: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bulk-download historical model runs from IEM.

    Parameters
    ----------
    model : str
        One of: GFS, LAV, NBS, MEX
    start : str
        Start datetime in ISO format (e.g. "2024-01-01T00:00Z")
    end : str
        End datetime in ISO format
    stations : list[str] | None
        ICAO codes. Defaults to all configured stations.

    Returns
    -------
    DataFrame with columns from _KEEP_COLS.
    """
    if stations is None:
        stations = ICAO_LIST

    # The CGI endpoint only accepts one station per request.
    all_dfs = []
    for i, stn in enumerate(stations):
        params = {
            "model": model,
            "station": stn,
            "sts": start,
            "ets": end,
            "format": "json",
        }

        try:
            resp = requests.get(IEM_BULK_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("IEM %s history fetch failed for %s: %s", model, stn, e)
            continue

        df = _parse_iem_json(data)
        if not df.empty:
            all_dfs.append(df)

        if i < len(stations) - 1:
            time.sleep(0.5)  # be polite

    if not all_dfs:
        log.warning("IEM %s history: empty response for %s to %s", model, start, end)
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "IEM %s history: %d rows, %d stations, %s to %s",
        model, len(df), df["station"].nunique(),
        df["ftime"].min(), df["ftime"].max(),
    )
    return df


# ── Convenience wrappers ─────────────────────────────────────────────────────

def fetch_latest_lamp(stations: list[str] | None = None) -> pd.DataFrame:
    """Fetch latest LAMP (hourly 1-25h ahead) from IEM."""
    return fetch_latest("LAV", stations)


def fetch_latest_nbm(stations: list[str] | None = None) -> pd.DataFrame:
    """Fetch latest NBM (3-hourly, includes uncertainty) from IEM."""
    return fetch_latest("NBS", stations)


def fetch_latest_gfs_mos(stations: list[str] | None = None) -> pd.DataFrame:
    """Fetch latest GFS MOS (3-hourly 6-72h) from IEM."""
    return fetch_latest("GFS", stations)


def fetch_latest_gfs_ext(stations: list[str] | None = None) -> pd.DataFrame:
    """Fetch latest GFS Extended MOS (12-hourly, days 4-8) from IEM."""
    return fetch_latest("MEX", stations)


# ── Max-temp extraction helpers ──────────────────────────────────────────────

def extract_nbm_max_temp(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Extract NBM daily max temperature forecast for a target date.

    NBM convention: txn at the 00Z ftime of the *next* day is the daily max.
    E.g., for target_date=2026-03-25, look at ftime=2026-03-26T00:00:00Z.

    Returns DataFrame: station, f_nbm, f_nbm_sd
    """
    if df.empty:
        return pd.DataFrame(columns=["station", "f_nbm", "f_nbm_sd"])

    target_dt = pd.to_datetime(target_date)
    next_day_00z = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    mask = (df["model"] == "NBS") & (df["ftime"] == next_day_00z)
    subset = df.loc[mask, ["station", "txn", "xnd"]].copy()
    subset = subset.rename(columns={"txn": "f_nbm", "xnd": "f_nbm_sd"})

    # If no txn, fall back to max of hourly tmp values for that day
    if subset["f_nbm"].isna().all() and not df.empty:
        day_mask = (
            (df["model"] == "NBS")
            & (df["ftime"] >= target_date + "T00:00:00Z")
            & (df["ftime"] < next_day_00z)
            & df["tmp"].notna()
        )
        fallback = df.loc[day_mask].groupby("station")["tmp"].max().reset_index()
        fallback = fallback.rename(columns={"tmp": "f_nbm"})
        fallback["f_nbm_sd"] = None
        return fallback

    return subset.dropna(subset=["f_nbm"])


def extract_gfs_mos_max_temp(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Extract GFS MOS daily max temperature forecast for a target date.

    GFS MOS convention: n_x at the 00Z ftime of the *next* day is the daily max.
    E.g., for target_date=2026-03-25, look at ftime=2026-03-26T00:00:00Z.

    Returns DataFrame: station, f_gfs_mos
    """
    if df.empty:
        return pd.DataFrame(columns=["station", "f_gfs_mos"])

    target_dt = pd.to_datetime(target_date)
    next_day_00z = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    mask = (df["model"].isin(["GFS", "MEX"])) & (df["ftime"] == next_day_00z)
    subset = df.loc[mask, ["station", "n_x"]].copy()
    subset = subset.rename(columns={"n_x": "f_gfs_mos"})

    # If no n_x, fall back to max of hourly tmp values
    if subset["f_gfs_mos"].isna().all() and not df.empty:
        day_mask = (
            df["model"].isin(["GFS", "MEX"])
            & (df["ftime"] >= target_date + "T00:00:00Z")
            & (df["ftime"] < next_day_00z)
            & df["tmp"].notna()
        )
        fallback = df.loc[day_mask].groupby("station")["tmp"].max().reset_index()
        fallback = fallback.rename(columns={"tmp": "f_gfs_mos"})
        return fallback

    return subset.dropna(subset=["f_gfs_mos"])


def extract_lamp_max_temp(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Extract LAMP daily max temperature estimate for a target date.

    LAMP has no explicit max/min field — we take the maximum of hourly
    tmp values within the local day window (roughly 12Z-03Z next day
    to capture the afternoon peak).

    Returns DataFrame: station, f_lamp_max
    """
    if df.empty:
        return pd.DataFrame(columns=["station", "f_lamp_max"])

    # Use 12Z to 03Z+1 to capture the afternoon max in US timezones
    start_utc = target_date + "T12:00:00Z"
    end_utc = pd.to_datetime(target_date + "T03:00:00Z") + pd.Timedelta(days=1)
    end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    mask = (
        (df["model"] == "LAV")
        & (df["ftime"] >= start_utc)
        & (df["ftime"] <= end_str)
        & df["tmp"].notna()
    )
    subset = df.loc[mask].groupby("station")["tmp"].max().reset_index()
    subset = subset.rename(columns={"tmp": "f_lamp_max"})
    return subset
