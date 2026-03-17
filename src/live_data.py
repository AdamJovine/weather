"""
Live data fetching for intraday trading.

Fetches the most recent NWP model runs from OpenMeteo so run_live.py
always uses up-to-date GFS/ECMWF/ICON forecasts and GEFS ensemble spread
rather than stale CSV snapshots.

Update cadence:
  GFS:   00Z / 06Z / 12Z / 18Z  (~every 6 hours)
  ECMWF: 00Z / 12Z               (~every 12 hours)
  GEFS:  00Z / 06Z / 12Z / 18Z  (31 members, ~every 6 hours)

Key functions:
  fetch_live_openmeteo(stations_df)          → GFS/ECMWF/ICON, today + 2 days
  fetch_live_gefs_spread(stations_df, date)  → 31-member GEFS spread for target date
  upsert_live(base_df, live_df)              → replace stale (city, date) rows in base_df
"""

import time
import numpy as np
import pandas as pd
import requests
from datetime import date

try:
    import openmeteo_requests
    _HAS_OPENMETEO = True
except ImportError:
    _HAS_OPENMETEO = False

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_MODELS = ["gfs_seamless", "icon_seamless", "ecmwf_ifs"]


def fetch_live_openmeteo(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch the latest GFS/ECMWF/ICON forecasts via OpenMeteo's live forecast API.

    Returns a DataFrame matching openmeteo_forecast_history.csv format for
    today + the next 2 days using the most recent available model run.

    Why today's date matters: build_feature_table() shifts GFS columns forward
    by 1 day (look-ahead prevention for training data).  Today's live model run
    therefore lands in tomorrow's (target_date) feature row — exactly right for
    intraday trading.

    Columns: date, city, forecast_high_gfs, forecast_high_ecmwf,
             ensemble_spread, ecmwf_minus_gfs, precip_forecast
    """
    if not _HAS_OPENMETEO:
        raise ImportError(
            "openmeteo_requests not installed. Run: pip install openmeteo-requests"
        )

    client = openmeteo_requests.Client()  # no cache — always use the freshest run
    frames = []

    for _, row in stations_df.iterrows():
        city = row["city"]
        try:
            responses = client.weather_api(FORECAST_URL, params={
                "latitude":         row["lat"],
                "longitude":        row["lon"],
                "daily":            ["temperature_2m_max", "precipitation_sum"],
                "temperature_unit": "fahrenheit",
                "timezone":         row["timezone"],
                "forecast_days":    3,
                "models":           _MODELS,
            })

            daily0 = responses[0].Daily()
            dates = pd.date_range(
                start=pd.to_datetime(daily0.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily0.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily0.Interval()),
                inclusive="left",
            ).tz_localize(None).normalize()

            model_tmax, model_precip = [], []
            for r in responses:
                d = r.Daily()
                model_tmax.append(d.Variables(0).ValuesAsNumpy())    # temperature_2m_max
                model_precip.append(d.Variables(1).ValuesAsNumpy())  # precipitation_sum

            stacked         = np.stack(model_tmax)          # (3, n_days)
            tmax_gfs        = stacked[0]                    # gfs_seamless
            tmax_ecmwf      = stacked[2]                    # ecmwf_ifs
            ensemble_spread = np.nanstd(stacked, axis=0)
            ecmwf_minus_gfs = tmax_ecmwf - tmax_gfs

            df = pd.DataFrame({
                "date":                dates.date,
                "city":                city,
                "forecast_high_gfs":   np.round(tmax_gfs, 1),
                "forecast_high_ecmwf": np.round(tmax_ecmwf, 1),
                "ensemble_spread":     np.round(ensemble_spread, 2),
                "ecmwf_minus_gfs":     np.round(ecmwf_minus_gfs, 2),
                "precip_forecast":     np.round(model_precip[0], 2),
            })
            frames.append(df.dropna(subset=["forecast_high_gfs"]))
            print(f"  {city}: live OpenMeteo OK  "
                  f"(GFS={tmax_gfs[0]:.1f}°F  spread={ensemble_spread[0]:.2f}°F)")
        except Exception as e:
            print(f"  {city}: live OpenMeteo failed — {e}")

        time.sleep(0.2)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_live_gefs_spread(stations_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Fetch the current 31-member GEFS ensemble spread for target_date.

    Unlike the GFS forecast columns, gefs_spread is NOT shifted inside
    build_feature_table() — it merges directly onto the target_date row.

    Columns: date, city, gefs_spread
    """
    today = str(date.today())
    frames = []

    for _, row in stations_df.iterrows():
        city = row["city"]
        try:
            resp = requests.get(
                ENSEMBLE_URL,
                params={
                    "latitude":         row["lat"],
                    "longitude":        row["lon"],
                    "daily":            "temperature_2m_max",
                    "models":           "gfs05",
                    "temperature_unit": "fahrenheit",
                    "timezone":         row["timezone"],
                    "start_date":       today,
                    "end_date":         target_date,
                },
                timeout=60,
            )
            if not resp.ok:
                raise ValueError(resp.json().get("reason", resp.text))

            data  = resp.json()
            daily = data.get("daily", {})
            times = daily.get("time", [])

            member_cols = [k for k in daily if k.startswith("temperature_2m_max")]
            if not member_cols:
                raise ValueError("No temperature_2m_max member columns in response")

            members = np.array([np.array(daily[col], dtype=float) for col in member_cols])
            spread  = np.nanstd(members, axis=0)

            df = pd.DataFrame({
                "date":        times,
                "city":        city,
                "gefs_spread": np.round(spread, 2),
            })
            df["date"] = pd.to_datetime(df["date"]).dt.date
            frames.append(df.dropna(subset=["gefs_spread"]))
            n = len(member_cols)
            target_spread = df.loc[df["date"].astype(str) == target_date, "gefs_spread"]
            spread_val = f"{target_spread.iloc[0]:.2f}" if len(target_spread) else "n/a"
            print(f"  {city}: live GEFS OK  ({n} members  target spread={spread_val}°F)")
        except Exception as e:
            print(f"  {city}: live GEFS spread failed — {e}")

        time.sleep(0.2)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def upsert_live(base_df: pd.DataFrame | None, live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace stale (city, date) rows in base_df with fresh rows from live_df.
    If base_df is None or empty, returns live_df directly.
    """
    if base_df is None or base_df.empty:
        return live_df.copy()

    base = base_df.copy()
    live = live_df.copy()

    base["date"] = pd.to_datetime(base["date"]).dt.date
    live["date"] = pd.to_datetime(live["date"]).dt.date

    live_keys = pd.MultiIndex.from_arrays([live["city"], live["date"]])
    base_keys = pd.MultiIndex.from_arrays([base["city"], base["date"]])
    keep = ~base_keys.isin(live_keys)

    return pd.concat([base[keep], live], ignore_index=True)
