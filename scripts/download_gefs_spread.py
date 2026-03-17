"""
Download GEFS ensemble spread from OpenMeteo's ensemble API.

Fetches daily max-temperature forecasts for all 31 GEFS members (model: gfs05),
then computes gefs_spread = std(member_0, ..., member_30) per day.  This captures
the internal ensemble uncertainty of the GFS model itself — distinct from the
inter-model spread (GFS vs ICON vs ECMWF) in openmeteo_forecast_history.csv.

Archive limit: OpenMeteo's ensemble API keeps a rolling 92-day window.
Dates outside that window fall back to ensemble_spread in features.py.
Run this script daily (or via cron) to accumulate a growing archive.

Saves to data/forecasts/gefs_spread.csv.

Run from project root:
  python scripts/download_gefs_spread.py
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

STATIONS_FILE  = Path("data/stations.csv")
OUTPUT_FILE    = Path("data/forecasts/gefs_spread.csv")
ENSEMBLE_URL   = "https://ensemble-api.open-meteo.com/v1/ensemble"
LOOKBACK_DAYS  = 5    # OpenMeteo free tier archives only ~5 days of GEFS history
FORECAST_DAYS  = 7    # Include upcoming week — spread grows with lead time


def fetch_gefs_spread(city: str, lat: float, lon: float,
                      timezone: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch GEFS ensemble member forecasts for one city and compute daily spread.

    Uses model gfs05 (31 members: control + 30 perturbations).
    Returns DataFrame with columns: [date, city, gefs_spread].
    """
    resp = requests.get(
        ENSEMBLE_URL,
        params={
            "latitude":         lat,
            "longitude":        lon,
            "daily":            "temperature_2m_max",
            "models":           "gfs05",
            "temperature_unit": "fahrenheit",
            "timezone":         timezone,
            "start_date":       start,
            "end_date":         end,
        },
        timeout=60,
    )
    if not resp.ok:
        raise ValueError(resp.json().get("reason", resp.text))

    data  = resp.json()
    daily = data.get("daily", {})
    times = daily.get("time", [])

    member_cols = [k for k in daily.keys() if k.startswith("temperature_2m_max")]
    if not member_cols:
        raise ValueError("No temperature_2m_max columns in response.")

    members = np.array([np.array(daily[col], dtype=float) for col in member_cols])
    spread  = np.nanstd(members, axis=0)

    df = pd.DataFrame({
        "date":        times,
        "city":        city,
        "gefs_spread": np.round(spread, 2),
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.dropna(subset=["gefs_spread"])


def main():
    stations = pd.read_csv(STATIONS_FILE)

    today        = date.today()
    window_start = today - timedelta(days=LOOKBACK_DAYS)

    # Load existing archive
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.date
        existing = existing.dropna(subset=["date"])
        print(f"Existing: {len(existing)} rows for {existing['city'].nunique()} cities.")
    else:
        existing = pd.DataFrame(columns=["date", "city", "gefs_spread"])

    frames = [existing]

    for _, row in stations.iterrows():
        city     = row["city"]
        lat      = row["lat"]
        lon      = row["lon"]
        timezone = row["timezone"]

        # Fetch from the later of: rolling window start OR day after last covered date
        if len(existing) and city in existing["city"].values:
            covered_end  = existing[existing["city"] == city]["date"].max()
            if covered_end >= today + timedelta(days=FORECAST_DAYS):
                print(f"  {city}: up to date, skipping.")
                continue
            fetch_start = max(covered_end + timedelta(days=1), window_start)
        else:
            fetch_start = window_start

        fetch_end = today + timedelta(days=FORECAST_DAYS)
        print(f"  {city}: fetching {fetch_start} → {fetch_end}...")
        try:
            df = fetch_gefs_spread(city, lat, lon, timezone,
                                   str(fetch_start), str(fetch_end))
            n_members = 31  # gfs05 always has 31
            print(f"    {len(df)} rows  ({df['date'].min()} → {df['date'].max()})  "
                  f"{n_members} members  spread_mean={df['gefs_spread'].mean():.2f}°F")
            frames.append(df)
        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(0.4)

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "city"])
        .sort_values(["city", "date"])
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_FILE}")
    for city, g in combined.groupby("city"):
        print(f"  {city:15s}: {g['date'].min()} → {g['date'].max()}  "
              f"({len(g)} rows)  spread_mean={g['gefs_spread'].mean():.2f}°F")


if __name__ == "__main__":
    main()
