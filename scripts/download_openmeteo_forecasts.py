"""
Download historical GFS day-ahead max-temperature forecasts from OpenMeteo.

OpenMeteo's historical forecast API stores past model runs so you can
reconstruct what GFS *predicted* for each day — exactly what's needed
for a proper backtest (as opposed to filling with climatology).

No API key required. Saves to data/forecasts/openmeteo_forecast_history.csv.

Run from project root:
  python scripts/download_openmeteo_forecasts.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import openmeteo_requests

import requests_cache
from retry_requests import retry

from src.config import TRAIN_START, TRAIN_END

STATIONS_FILE = Path("data/stations.csv")
OUTPUT_FILE   = Path("data/forecasts/openmeteo_forecast_history.csv")

# OpenMeteo client with disk cache + auto-retry on transient failures
cache_session  = requests_cache.CachedSession(".cache/openmeteo", expire_after=-1)
retry_session  = retry(cache_session, retries=5, backoff_factor=0.3)
openmeteo_client = openmeteo_requests.Client(session=retry_session)

OPENMETEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_city_forecast(city: str, lat: float, lon: float,
                        timezone: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch GFS + ECMWF day-ahead max temperature forecasts, multi-model ensemble
    spread, and GFS precipitation forecast for one city.

    Requests three independent NWP models (GFS, ICON, ECMWF) in a single call.
    ensemble_spread = std(gfs, icon, ecmwf) — when models agree the forecast is
    reliable; when they disagree uncertainty is high.
    ecmwf_minus_gfs  = signed model disagreement (ECMWF warmer/cooler than GFS).
    precip_forecast  = GFS precipitation_sum (mm); dry days → higher tmax.
    """
    MODELS = ["gfs_seamless", "icon_seamless", "ecmwf_ifs"]

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start,
        "end_date":         end,
        "daily":            ["temperature_2m_max", "precipitation_sum"],
        "temperature_unit": "fahrenheit",
        "timezone":         timezone,
        "models":           MODELS,
    }

    responses = openmeteo_client.weather_api(OPENMETEO_URL, params=params)

    # Extract dates from first response (all models share the same grid)
    daily0 = responses[0].Daily()
    dates  = pd.date_range(
        start     = pd.to_datetime(daily0.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(daily0.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=daily0.Interval()),
        inclusive = "left",
    ).tz_localize(None).normalize()

    model_tmax   = []
    model_precip = []
    for r in responses:
        daily = r.Daily()
        model_tmax.append(daily.Variables(0).ValuesAsNumpy())    # temperature_2m_max
        model_precip.append(daily.Variables(1).ValuesAsNumpy())  # precipitation_sum

    stacked         = np.stack(model_tmax)        # (3, n_days)
    tmax_gfs        = stacked[0]                  # GFS primary forecast
    tmax_ecmwf      = stacked[2]                  # ECMWF forecast
    ensemble_spread = np.nanstd(stacked, axis=0)  # std across models
    ecmwf_minus_gfs = tmax_ecmwf - tmax_gfs       # signed model disagreement
    precip_forecast = model_precip[0]             # GFS precipitation (mm)

    df = pd.DataFrame({
        "date":             dates.date,
        "city":             city,
        "forecast_high_gfs":  np.round(tmax_gfs, 1),
        "forecast_high_ecmwf": np.round(tmax_ecmwf, 1),
        "ensemble_spread":    np.round(ensemble_spread, 2),
        "ecmwf_minus_gfs":    np.round(ecmwf_minus_gfs, 2),
        "precip_forecast":    np.round(precip_forecast, 2),
    })

    df = df.dropna(subset=["forecast_high_gfs"])
    return df


def main():
    stations = pd.read_csv(STATIONS_FILE)

    NEW_COLS = {"forecast_high_ecmwf", "ecmwf_minus_gfs", "precip_forecast"}

    # Load existing data to allow incremental updates.
    # If new columns are missing, clear and re-fetch everything.
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        existing["date"] = existing["date"].dt.date
        if not NEW_COLS.issubset(existing.columns):
            print("New columns detected — clearing cache for full re-fetch.")
            existing = pd.DataFrame(columns=["date", "city", "forecast_high_gfs"])
        else:
            print(f"Existing: {len(existing)} rows for {existing['city'].nunique()} cities.")
    else:
        existing = pd.DataFrame(columns=["date", "city", "forecast_high_gfs"])

    frames = [existing]

    for _, row in stations.iterrows():
        city     = row["city"]
        lat      = row["lat"]
        lon      = row["lon"]
        timezone = row["timezone"]

        # Only fetch dates not yet in file
        if len(existing) and city in existing["city"].values:
            city_existing = existing[existing["city"] == city]
            covered_end   = str(city_existing["date"].max())
            if covered_end >= TRAIN_END:
                print(f"  {city}: up to date, skipping.")
                continue
            fetch_start = covered_end   # re-fetch last date to catch gaps
        else:
            fetch_start = TRAIN_START

        print(f"  {city}: fetching {fetch_start} → {TRAIN_END}...")
        for attempt in range(3):
            try:
                df = fetch_city_forecast(city, lat, lon, timezone, fetch_start, TRAIN_END)
                print(f"    {len(df)} rows  ({df['date'].min()} → {df['date'].max()})")
                frames.append(df)
                break
            except Exception as e:
                msg = str(e)
                if "limit exceeded" in msg.lower() and attempt < 2:
                    print(f"    Rate limited — waiting 65s before retry {attempt + 1}/2...")
                    time.sleep(65)
                else:
                    print(f"    ERROR: {e}")
                    break

        time.sleep(4)   # stay well under OpenMeteo's per-minute limit

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "city"]).sort_values(["city", "date"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_FILE}")
    for city, g in combined.groupby("city"):
        print(f"  {city:12s}: {g['date'].min()} → {g['date'].max()}  ({len(g)} rows)")


if __name__ == "__main__":
    main()
