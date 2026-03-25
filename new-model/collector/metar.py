"""
METAR observation poller.

Fetches the latest METAR reports from aviationweather.gov for all stations.
The Aviation Weather Center API is free, requires no key, and updates every
~5-20 minutes — the same ASOS network Kalshi uses for settlement.

Returns a DataFrame ready to upsert into metar_obs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from collector.config import ICAO_LIST, ICAO_TO_STATION

log = logging.getLogger(__name__)

METAR_URL = "https://aviationweather.gov/api/data/metar"


def fetch_metar(hours: int = 3) -> pd.DataFrame:
    """
    Fetch recent METAR observations for all configured stations.

    Parameters
    ----------
    hours : int
        How many hours of history to request (default 3, covers gaps
        in polling if the collector was briefly offline).

    Returns
    -------
    DataFrame with columns: station, obs_time, temp_f, dew_point_f,
                            wind_speed_kt, wind_dir, raw_metar
    """
    try:
        resp = requests.get(
            METAR_URL,
            params={
                "ids": ",".join(ICAO_LIST),
                "format": "json",
                "hours": str(hours),
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs_list = resp.json()
    except Exception as e:
        log.error("METAR fetch failed: %s", e)
        return pd.DataFrame()

    rows = []
    for obs in obs_list:
        icao = (obs.get("icaoId") or obs.get("stationId") or "").upper()
        if icao not in ICAO_TO_STATION:
            continue

        # Parse observation time
        raw_time = obs.get("obsTime") or obs.get("receiptTime")
        if raw_time is None:
            continue
        try:
            if isinstance(raw_time, (int, float)):
                obs_dt = datetime.fromtimestamp(raw_time, tz=timezone.utc)
            else:
                obs_dt = datetime.fromisoformat(
                    str(raw_time).replace("Z", "+00:00")
                )
        except Exception:
            continue

        # Temperature (°C → °F)
        temp_c = obs.get("temp")
        temp_f = None
        if temp_c is not None:
            try:
                temp_f = round(float(temp_c) * 9 / 5 + 32, 1)
            except (TypeError, ValueError):
                pass

        # Dew point (°C → °F)
        dew_c = obs.get("dewp")
        dew_f = None
        if dew_c is not None:
            try:
                dew_f = round(float(dew_c) * 9 / 5 + 32, 1)
            except (TypeError, ValueError):
                pass

        # Wind
        wspd = obs.get("wspd")
        wdir = obs.get("wdir")

        rows.append({
            "station": icao,
            "obs_time": obs_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temp_f": temp_f,
            "dew_point_f": dew_f,
            "wind_speed_kt": float(wspd) if wspd is not None else None,
            "wind_dir": int(wdir) if wdir is not None and str(wdir).isdigit() else None,
            "raw_metar": obs.get("rawOb"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["station", "obs_time"])
        log.info(
            "METAR: %d obs across %d stations",
            len(df),
            df["station"].nunique(),
        )
    else:
        log.warning("METAR: no observations returned")
    return df


def compute_running_max(conn, station: str, date_str: str) -> dict | None:
    """
    Compute the running maximum temperature for a station on a given date
    from stored METAR observations.

    Parameters
    ----------
    conn : sqlite3.Connection
    station : str — ICAO code
    date_str : str — YYYY-MM-DD

    Returns
    -------
    dict with keys: running_max_f, current_temp_f, obs_count, latest_obs_utc
    or None if no observations exist for that day.
    """
    cursor = conn.execute(
        """
        SELECT temp_f, obs_time
        FROM metar_obs
        WHERE station = ?
          AND obs_time >= ? || 'T00:00:00Z'
          AND obs_time <  ? || 'T23:59:59Z'
          AND temp_f IS NOT NULL
        ORDER BY obs_time ASC
        """,
        (station, date_str, date_str),
    )
    rows = cursor.fetchall()
    if not rows:
        return None

    temps = [r[0] for r in rows]
    latest_time = rows[-1][1]

    return {
        "running_max_f": round(max(temps), 1),
        "current_temp_f": round(temps[-1], 1),
        "obs_count": len(temps),
        "latest_obs_utc": latest_time,
    }
