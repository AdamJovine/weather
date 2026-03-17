"""
Pull daily high temperature forecasts from the NWS API (api.weather.gov).

No auth required. Rate limits are loose but be polite (cache aggressively).
Docs: https://www.weather.gov/documentation/services-web-api

Also fetches the NBM (National Blend of Models) gridded max temperature via
the /gridpoints endpoint.  NBM gives the raw model-blend value before the
human forecaster edits applied in the textual /forecast product.
"""

import requests
import pandas as pd
from datetime import date, timedelta
from typing import Optional


NWS_BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "weather-kalshi-bot (contact@example.com)"}


def get_gridpoint_forecast(lat: float, lon: float) -> dict:
    """
    Fetch the NWS forecast JSON for a lat/lon.

    Two-step:
      1. /points/{lat},{lon} → returns office + gridX + gridY
      2. forecast URL from properties.forecast
    """
    points_resp = requests.get(
        f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
        headers=HEADERS,
        timeout=30,
    )
    points_resp.raise_for_status()
    props = points_resp.json()["properties"]

    forecast_url = props["forecast"]
    forecast_resp = requests.get(forecast_url, headers=HEADERS, timeout=30)
    forecast_resp.raise_for_status()

    return forecast_resp.json()


def extract_daily_high_forecast(forecast_json: dict, target_date: str) -> Optional[float]:
    """
    Pull daytime high temperature for target_date (YYYY-MM-DD) from NWS forecast.

    NWS returns periods; daytime periods named "Today" / day-of-week contain the high.
    Returns None if no matching period found.
    """
    periods = forecast_json["properties"]["periods"]
    target = pd.to_datetime(target_date).date()

    for p in periods:
        start = pd.to_datetime(p["startTime"])
        if start.date() == target and p.get("isDaytime", False):
            return float(p["temperature"])

    return None


def get_nbm_high(grid_data_url: str, target_date: str) -> Optional[float]:
    """
    Fetch the NBM gridded max temperature for target_date from the NWS
    /gridpoints/{wfo}/{x},{y} endpoint.

    Returns temperature in °F, or None if no matching period is found.
    The maxTemperature values are in degC and converted here.
    """
    resp = requests.get(grid_data_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    props = resp.json()["properties"]

    max_temp = props.get("maxTemperature", {})
    uom      = max_temp.get("uom", "")
    values   = max_temp.get("values", [])

    target = pd.to_datetime(target_date).date()

    matching = []
    for entry in values:
        try:
            # validTime format: "2024-03-16T07:00:00+00:00/PT12H"
            dt_str = entry["validTime"].split("/")[0]
            start_date = pd.to_datetime(dt_str, utc=True).date()
            if start_date == target and entry["value"] is not None:
                matching.append(float(entry["value"]))
        except Exception:
            continue

    if not matching:
        return None

    val = max(matching)  # maxTemperature — take max if multiple periods on same day
    if "degC" in uom:
        val = val * 9 / 5 + 32
    return round(val, 1)


def get_forecast_high(lat: float, lon: float, target_date: Optional[str] = None) -> Optional[float]:
    """
    Convenience wrapper: fetch NWS forecast and return high for target_date.
    Defaults to tomorrow if target_date is None.
    """
    if target_date is None:
        target_date = (date.today() + timedelta(days=1)).isoformat()

    forecast_json = get_gridpoint_forecast(lat, lon)
    return extract_daily_high_forecast(forecast_json, target_date)


def get_forecasts_for_stations(stations_df: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch NWS forecast highs and NBM gridded highs for all stations.

    Makes a single /points call per station, then fetches:
      - /forecast          → NWS textual forecast high (forecast_high)
      - /gridpoints/{wfo}/{x},{y} → NBM gridded max temperature (nbm_high)

    stations_df must have columns: city, lat, lon
    Returns DataFrame with columns: city, forecast_high, nbm_high, target_date
    """
    if target_date is None:
        target_date = (date.today() + timedelta(days=1)).isoformat()

    rows = []
    for _, row in stations_df.iterrows():
        forecast_high = None
        nbm_high      = None
        try:
            # Single /points call to get both URLs
            points_resp = requests.get(
                f"{NWS_BASE}/points/{row['lat']:.4f},{row['lon']:.4f}",
                headers=HEADERS,
                timeout=30,
            )
            points_resp.raise_for_status()
            props = points_resp.json()["properties"]

            # NWS textual forecast
            forecast_resp = requests.get(props["forecast"], headers=HEADERS, timeout=30)
            forecast_resp.raise_for_status()
            forecast_high = extract_daily_high_forecast(forecast_resp.json(), target_date)

            # NBM gridded max temperature
            try:
                nbm_high = get_nbm_high(props["forecastGridData"], target_date)
            except Exception as nbm_err:
                print(f"  NBM fetch failed for {row['city']}: {nbm_err}")

        except Exception as e:
            print(f"Warning: could not fetch forecast for {row['city']}: {e}")

        rows.append({
            "city":         row["city"],
            "forecast_high": forecast_high,
            "nbm_high":      nbm_high,
            "target_date":  target_date,
        })

    return pd.DataFrame(rows)
