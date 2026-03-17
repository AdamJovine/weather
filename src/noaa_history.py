"""
Download historical daily TMAX from NOAA CDO (GHCN-Daily).

NOAA CDO docs: https://www.ncei.noaa.gov/cdo-web/webservices/v2
Rate limits: 5 req/sec, 10,000 req/day
Units: 'standard' returns Fahrenheit for temperature.
"""

import time
import requests
import pandas as pd

from src.config import NOAA_TOKEN

BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
RATE_LIMIT_SLEEP = 0.25  # 4 req/sec to stay under 5/sec limit


def _fetch_one_year(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch TMAX for a single station within a single calendar year.
    NOAA CDO enforces a 1-year max date range per request.
    """
    # NOAA CDO requires dataset-prefixed station IDs: "GHCND:USW00094728"
    if not station_id.startswith("GHCND:"):
        station_id = f"GHCND:{station_id}"

    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": "TMAX",
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "standard",
    }

    all_rows = []
    offset = 1

    while True:
        params["offset"] = offset
        r = requests.get(f"{BASE}/data", headers=headers, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        rows = payload.get("results", [])
        if not rows:
            break

        all_rows.extend(rows)

        metadata = payload.get("metadata", {})
        resultset = metadata.get("resultset", {})
        count = resultset.get("count", 0)
        limit = resultset.get("limit", 1000)
        offset += limit

        if len(all_rows) >= count:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    return all_rows


def get_daily_tmax(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch GHCN-Daily TMAX for a single station over a multi-year date range.

    NOAA CDO enforces a max 1-year window per request, so this function
    automatically splits the range into per-year chunks and concatenates results.

    Returns a DataFrame with columns: date, station, tmax (°F).
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_rows = []
    chunk_start = start

    while chunk_start <= end:
        # Cap chunk end at Dec 31 of the same year or the overall end date
        chunk_end = min(
            pd.Timestamp(year=chunk_start.year, month=12, day=31),
            end,
        )
        print(f"      chunk {chunk_start.date()} → {chunk_end.date()}", end=" ", flush=True)
        try:
            rows = _fetch_one_year(
                station_id,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            print(f"({len(rows)} rows)")
            all_rows.extend(rows)
        except Exception as e:
            print(f"(ERROR: {e})")

        chunk_start = pd.Timestamp(year=chunk_start.year + 1, month=1, day=1)
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={"value": "tmax"})
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    return df[["date", "station", "tmax"]]


def get_station_info(station_id: str) -> dict:
    """Fetch metadata for a station (name, location, etc.)."""
    headers = {"token": NOAA_TOKEN}
    r = requests.get(f"{BASE}/stations/{station_id}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()
