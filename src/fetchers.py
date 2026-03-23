"""
src/fetchers.py

HTTP fetch and parse functions for all external weather data sources.
No side effects — each function returns a DataFrame; callers write to DB.

Sources
───────
  Climate indices  — NOAA CPC AO/NAO/ONI/PNA, NOAA PSL PDO, BOM MJO
  OpenMeteo GFS    — historical GFS/ECMWF/ICON forecast archive
  GEFS spread      — 31-member GEFS ensemble spread
  Kalshi           — settled market listings + hourly candlestick prices
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ─── Climate index URLs ───────────────────────────────────────────────────────

AO_URL  = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
PNA_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table"
PDO_URL = "https://psl.noaa.gov/data/correlation/pdo.data"
MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"

ONI_SEASON_MAP = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


# ─── Climate index parsers ────────────────────────────────────────────────────

def _parse_cpc_wide_table(text: str, col_name: str) -> pd.DataFrame:
    """
    Parse CPC-style year × 12-month wide table into long form.
    Each data line: year val1 val2 ... val12
    Missing sentinel: abs(v) > 10 → NaN
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if not parts[0].lstrip("-").isdigit():
            continue
        year = int(parts[0])
        for month_idx, val in enumerate(parts[1:13], start=1):
            try:
                v = float(val)
            except ValueError:
                v = np.nan
            if abs(v) > 10:
                v = np.nan
            rows.append({"year": year, "month": month_idx, col_name: v})
    return pd.DataFrame(rows)


def fetch_ao() -> pd.DataFrame:
    resp = requests.get(AO_URL, timeout=30)
    resp.raise_for_status()
    df = _parse_cpc_wide_table(resp.text, "ao_index")
    print(f"  AO:  {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_nao() -> pd.DataFrame:
    resp = requests.get(NAO_URL, timeout=30)
    resp.raise_for_status()
    df = _parse_cpc_wide_table(resp.text, "nao_index")
    print(f"  NAO: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_oni() -> pd.DataFrame:
    resp = requests.get(ONI_URL, timeout=30)
    resp.raise_for_status()
    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        seas, yr_str, _total, anom_str = parts[0], parts[1], parts[2], parts[3]
        if seas not in ONI_SEASON_MAP:
            continue
        try:
            year = int(yr_str)
            anom = float(anom_str)
        except ValueError:
            continue
        rows.append({"year": year, "month": ONI_SEASON_MAP[seas], "oni": anom})
    df = pd.DataFrame(rows).drop_duplicates(subset=["year", "month"])
    print(f"  ONI: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_pna() -> pd.DataFrame:
    resp = requests.get(PNA_URL, timeout=30)
    resp.raise_for_status()
    df = _parse_cpc_wide_table(resp.text, "pna_index")
    print(f"  PNA: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_pdo() -> pd.DataFrame:
    resp = requests.get(PDO_URL, timeout=30)
    resp.raise_for_status()
    rows = []
    pending_year = None
    for line in resp.text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
            pending_year = int(parts[0])
            continue
        first = parts[0].lstrip("-")
        if first.isdigit() and len(parts[0].lstrip("-")) == 4 and len(parts) >= 13:
            year = int(parts[0])
            vals = parts[1:13]
            pending_year = None
        elif pending_year is not None and len(parts) >= 12:
            year = pending_year
            vals = parts[:12]
            pending_year = None
        else:
            continue
        for month_idx, val in enumerate(vals, start=1):
            try:
                v = float(val)
            except ValueError:
                v = np.nan
            if abs(v) > 9:
                v = np.nan
            rows.append({"year": year, "month": month_idx, "pdo_index": v})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  PDO: WARNING — no data parsed")
        return df
    print(f"  PDO: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_mjo() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; weather-model/1.0)"}
    resp = requests.get(MJO_URL, timeout=30, headers=headers)
    resp.raise_for_status()
    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        if not (parts[0].isdigit() and len(parts[0]) == 4):
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            rmm1  = float(parts[3])
            rmm2  = float(parts[4])
            phase = int(parts[5])
        except (ValueError, IndexError):
            continue
        if abs(rmm1) > 100 or abs(rmm2) > 100 or phase < 1 or phase > 8:
            continue
        amplitude = np.sqrt(rmm1 ** 2 + rmm2 ** 2)
        angle     = 2 * np.pi * (phase - 1) / 8
        rows.append({
            "date":          f"{year:04d}-{month:02d}-{day:02d}",
            "mjo_amplitude": round(amplitude, 4),
            "mjo_phase_sin": round(float(np.sin(angle)), 6),
            "mjo_phase_cos": round(float(np.cos(angle)), 6),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  MJO: {len(df)} rows  ({df['date'].min().date()}–{df['date'].max().date()})")
    return df


# ─── OpenMeteo GFS / ECMWF ───────────────────────────────────────────────────

OPENMETEO_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Lazy singleton — only constructed when actually needed
_openmeteo_client = None


def _get_openmeteo_client():
    global _openmeteo_client
    if _openmeteo_client is None:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        def _log_retry(r, *args, **kwargs):
            if r.status_code >= 400:
                print(f"\n    [openmeteo] {r.status_code} — retrying...", flush=True)

        cache = requests_cache.CachedSession(".cache/openmeteo", expire_after=-1)
        session = retry(cache, retries=5, backoff_factor=0.5)
        session.hooks["response"].append(_log_retry)
        _openmeteo_client = openmeteo_requests.Client(session=session)
    return _openmeteo_client


def fetch_city_forecast(
    city: str, lat: float, lon: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch GFS + ECMWF day-ahead max-temperature forecasts for one city.
    Returns columns: date, city, forecast_high_gfs, forecast_high_ecmwf,
                     ensemble_spread, ecmwf_minus_gfs, precip_forecast

    Falls back to GFS-only if the multi-model request fails (e.g. icon/ecmwf
    unavailable for the requested date range), filling ensemble columns with NaN.
    """
    client = _get_openmeteo_client()
    base_params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start,
        "end_date":         end,
        "daily":            ["temperature_2m_max", "precipitation_sum",
                             "shortwave_radiation_sum"],
        "temperature_unit": "fahrenheit",
        "timezone":         timezone,
    }

    MODELS = ["icon_seamless", "ecmwf_ifs04", "ecmwf_ifs"]
    multi_model = False
    try:
        responses = client.weather_api(OPENMETEO_FORECAST_URL,
                                       params={**base_params, "models": MODELS})
        multi_model = True
    except Exception as e:
        print(f"\n    flatbuffers/multi-model failed — falling back to GFS JSON", flush=True)

    if multi_model:
        daily0 = responses[0].Daily()
        dates  = pd.date_range(
            start     = pd.to_datetime(daily0.Time(),    unit="s", utc=True),
            end       = pd.to_datetime(daily0.TimeEnd(), unit="s", utc=True),
            freq      = pd.Timedelta(seconds=daily0.Interval()),
            inclusive = "left",
        ).tz_localize(None).normalize()

        model_tmax, model_precip, model_srad = [], [], []
        for r in responses:
            d = r.Daily()
            model_tmax.append(d.Variables(0).ValuesAsNumpy())
            model_precip.append(d.Variables(1).ValuesAsNumpy())
            model_srad.append(d.Variables(2).ValuesAsNumpy())

        stacked         = np.stack(model_tmax)
        tmax_gfs        = stacked[0]   # icon_seamless (stored in gfs column for DB compat)
        tmax_ecmwf      = stacked[1]   # ecmwf_ifs04
        ensemble_spread = np.nanstd(stacked, axis=0)
        ecmwf_minus_gfs = tmax_ecmwf - tmax_gfs
        precip_forecast = model_precip[0]
        srad_forecast   = model_srad[0]
    else:
        # Fallback: plain JSON request for GFS only — bypasses flatbuffers entirely
        import requests as _req
        r = _req.get(OPENMETEO_FORECAST_URL, params={
            **base_params,
            "models":  "icon_seamless",
            "format":  "json",
        }, timeout=30)
        r.raise_for_status()
        daily = r.json()["daily"]
        dates           = pd.to_datetime(daily["time"])
        tmax_gfs        = np.array(daily["temperature_2m_max"],       dtype=float)
        precip_forecast = np.array(daily["precipitation_sum"],        dtype=float)
        srad_forecast   = np.array(daily.get("shortwave_radiation_sum",
                                             [np.nan] * len(tmax_gfs)), dtype=float)
        tmax_ecmwf      = np.full_like(tmax_gfs, np.nan)
        ensemble_spread = np.full_like(tmax_gfs, np.nan)
        ecmwf_minus_gfs = np.full_like(tmax_gfs, np.nan)

    df = pd.DataFrame({
        "date":                dates.date,
        "city":                city,
        "forecast_high_gfs":   np.round(tmax_gfs, 1),
        "forecast_high_ecmwf": np.round(tmax_ecmwf, 1),
        "ensemble_spread":     np.round(ensemble_spread, 2),
        "ecmwf_minus_gfs":     np.round(ecmwf_minus_gfs, 2),
        "precip_forecast":     np.round(precip_forecast, 2),
        "shortwave_radiation": np.round(srad_forecast, 2),
    })
    df = df.dropna(subset=["forecast_high_gfs"])

    # Fetch hourly 850hPa temperature and dew point, aggregate to daily
    extras = _fetch_city_hourly_extras(lat, lon, timezone, start, end)
    if extras is not None:
        extras["date"] = pd.to_datetime(extras["date"]).dt.date
        df = df.merge(extras, on="date", how="left")

    return df


def _fetch_city_hourly_extras(
    lat: float, lon: float, timezone: str, start: str, end: str
) -> "pd.DataFrame | None":
    """
    Fetch hourly GFS 850hPa temperature and 2m dew point for one city,
    aggregate to daily max, and return as a DataFrame with columns:
        date, temp_850hpa, dew_point_max
    Returns None on any failure.
    """
    import requests as _req
    # Try the regular forecast API first (supports 850hPa for recent/future dates),
    # then fall back to the historical forecast API.
    _URLS = [
        "https://api.open-meteo.com/v1/forecast",
        OPENMETEO_FORECAST_URL,
    ]
    hourly = None
    for url in _URLS:
        try:
            r = _req.get(
                url,
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "start_date":       start,
                    "end_date":         end,
                    "hourly":           ["temperature_850hPa", "dew_point_2m"],
                    "temperature_unit": "fahrenheit",
                    "timezone":         timezone,
                    "models":           "gfs_seamless",
                    "format":           "json",
                },
                timeout=60,
            )
            r.raise_for_status()
            hourly = r.json().get("hourly", {})
            if hourly and "time" in hourly:
                break
            hourly = None
        except Exception:
            continue
    if not hourly or "time" not in hourly:
        return None

    df_h = pd.DataFrame({
        "datetime":        pd.to_datetime(hourly["time"]),
        "temperature_850": np.array(hourly.get("temperature_850hPa", hourly.get("temperature_850hpa", [])), dtype=float),
        "dew_point":       np.array(hourly.get("dew_point_2m", []),       dtype=float),
    })
    df_h["date"] = df_h["datetime"].dt.date
    daily = (
        df_h.groupby("date")
        .agg(
            temp_850hpa=("temperature_850", "max"),
            dew_point_max=("dew_point",     "max"),
        )
        .reset_index()
    )
    daily["temp_850hpa"]   = np.round(daily["temp_850hpa"].astype(float),   1)
    daily["dew_point_max"] = np.round(daily["dew_point_max"].astype(float), 1)
    return daily


# ─── GEFS ensemble spread ─────────────────────────────────────────────────────

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def fetch_gefs_spread(
    city: str, lat: float, lon: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch 31-member GEFS ensemble and return daily spread (std) per city.
    Returns columns: date, city, gefs_spread
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
    member_cols = [k for k in daily if k.startswith("temperature_2m_max")]
    if not member_cols:
        raise ValueError("No temperature_2m_max columns in GEFS response.")

    members = np.array([np.array(daily[c], dtype=float) for c in member_cols])
    spread  = np.nanstd(members, axis=0)

    df = pd.DataFrame({"date": times, "city": city,
                       "gefs_spread": np.round(spread, 2)})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.dropna(subset=["gefs_spread"])


# ─── OpenMeteo ERA5 archive (recent actuals backfill) ────────────────────────

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_obs_actuals(
    city: str, lat: float, lon: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch daily TMAX from the OpenMeteo ERA5 reanalysis archive.

    ERA5 is a high-quality reanalysis product that closely tracks ASOS station
    observations and is updated daily with a ~5-day lag.  This function is used
    to fill recent gaps in weather_daily when NOAA GHCN-Daily data is stale.

    Returns columns: date, city, station, tmax (°F).
    Only rows with non-null tmax are returned.
    """
    import requests as _req
    resp = _req.get(
        OPENMETEO_ARCHIVE_URL,
        params={
            "latitude":         lat,
            "longitude":        lon,
            "start_date":       start,
            "end_date":         end,
            "daily":            "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone":         timezone,
        },
        timeout=30,
    )
    if not resp.ok:
        raise ValueError(
            f"OpenMeteo archive error {resp.status_code}: "
            f"{resp.json().get('reason', resp.text)}"
        )

    daily     = resp.json().get("daily", {})
    times     = daily.get("time", [])
    tmax_vals = daily.get("temperature_2m_max", [])

    if not times or not tmax_vals:
        return pd.DataFrame(columns=["date", "city", "station", "tmax"])

    df = pd.DataFrame({
        "date":    pd.to_datetime(times).date,
        "city":    city,
        "station": "openmeteo_era5",
        "tmax":    np.round(np.array(tmax_vals, dtype=float), 1),
    })
    return df.dropna(subset=["tmax"])


def fetch_recent_actuals(
    city: str, lat: float, lon: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch daily TMAX from the OpenMeteo *forecast* API for recent days.

    The forecast API provides analyzed/observed temperatures for the last
    ~5 days via the ``past_days`` mechanism, with only ~1-day lag — much
    fresher than ERA5 (~5-day lag).  This fills the gap between ERA5 and
    yesterday in weather_daily.

    Returns columns: date, city, station, tmax (°F).
    Only rows with non-null tmax are returned.
    """
    import requests as _req
    resp = _req.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":         lat,
            "longitude":        lon,
            "start_date":       start,
            "end_date":         end,
            "daily":            "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone":         timezone,
        },
        timeout=30,
    )
    if not resp.ok:
        raise ValueError(
            f"OpenMeteo forecast error {resp.status_code}: "
            f"{resp.json().get('reason', resp.text)}"
        )

    daily = resp.json().get("daily", {})
    times     = daily.get("time", [])
    tmax_vals = daily.get("temperature_2m_max", [])

    if not times or not tmax_vals:
        return pd.DataFrame(columns=["date", "city", "station", "tmax"])

    df = pd.DataFrame({
        "date":    pd.to_datetime(times).date,
        "city":    city,
        "station": "openmeteo_forecast",
        "tmax":    np.round(np.array(tmax_vals, dtype=float), 1),
    })
    return df.dropna(subset=["tmax"])


# ─── Real-time METAR observations ────────────────────────────────────────────

AVIATIONWX_METAR_URL = "https://aviationweather.gov/api/data/metar"


def fetch_current_obs(stations: "pd.DataFrame") -> dict:
    """
    Fetch the most recent METAR observations for all stations and return
    intraday running-maximum temperatures in °F, keyed by city name.

    Uses the Aviation Weather Center API (free, no key required, ~5-60 min
    update cadence — the same ASOS network Kalshi uses for settlement).

    Returns
    -------
    dict[str, dict]:
        city → {
            "current_temp_f": float,   # most recent observation
            "running_max_f":  float,   # highest temp recorded today so far
            "obs_count":      int,     # number of today's observations used
            "latest_obs_utc": str,     # ISO timestamp of freshest observation
        }
    Only cities with at least one valid observation are included.
    """
    if "icao_id" not in stations.columns:
        return {}

    icao_ids = stations["icao_id"].dropna().tolist()
    if not icao_ids:
        return {}

    try:
        resp = requests.get(
            AVIATIONWX_METAR_URL,
            params={
                "ids":    ",".join(icao_ids),
                "format": "json",
                "hours":  "24",
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs_list = resp.json()
    except Exception as e:
        print(f"  [obs] METAR fetch failed: {e}")
        return {}

    # Build icao → city lookup
    icao_to_city = dict(zip(stations["icao_id"], stations["city"]))

    # Group observations by station; filter to today (UTC date)
    from datetime import date as _date, timezone as _tz
    today_utc = _date.today()
    grouped: dict[str, list] = {}
    for obs in obs_list:
        icao = (obs.get("icaoId") or obs.get("stationId") or "").upper()
        city = icao_to_city.get(icao)
        if not city:
            continue
        temp_c = obs.get("temp")
        if temp_c is None:
            continue
        try:
            temp_f = float(temp_c) * 9 / 5 + 32
        except (TypeError, ValueError):
            continue

        # obs_time can be epoch int or ISO string
        raw_time = obs.get("obsTime") or obs.get("receiptTime") or ""
        try:
            if isinstance(raw_time, (int, float)):
                obs_dt = datetime.fromtimestamp(raw_time, tz=timezone.utc)
            else:
                obs_dt = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        except Exception:
            continue

        if obs_dt.date() != today_utc:
            continue

        grouped.setdefault(city, []).append((obs_dt, temp_f))

    result = {}
    for city, readings in grouped.items():
        readings.sort(key=lambda x: x[0])   # oldest → newest
        temps = [t for _, t in readings]
        latest_dt, latest_temp = readings[-1]
        result[city] = {
            "current_temp_f": round(latest_temp, 1),
            "running_max_f":  round(max(temps), 1),
            "obs_count":      len(readings),
            "latest_obs_utc": latest_dt.strftime("%H:%MZ"),
        }

    return result


# ─── Kalshi helpers ───────────────────────────────────────────────────────────

# Markets settled before this timestamp use the historical API endpoints
HISTORICAL_CUTOFF_TS = int(
    datetime(2025, 3, 16, 0, 0, 0, tzinfo=timezone.utc).timestamp()
)
KALSHI_RATE_SLEEP = 0.10   # seconds between API calls (~10 req/s limit)
KALSHI_PERIOD_MINS = 60    # candlestick interval in minutes


def parse_settlement_date(ticker: str):
    """
    Extract the settlement date from a Kalshi weather ticker.
    Format: SERIES-YYMONDD-Tthreshold   e.g. KXHIGHNYC-26MAR17-T59
    Returns a date object or None if not parseable.
    """
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    try:
        return datetime.strptime(
            f"{m.group(1)}{m.group(2)}{m.group(3)}", "%y%b%d"
        ).date()
    except ValueError:
        return None


def _to_dict(obj):
    return obj if isinstance(obj, dict) else obj.to_dict()


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def fetch_settled_markets_post(client, series: str) -> list[dict]:
    """Paginate settled markets via the regular (post-cutoff) API."""
    markets, cursor = [], None
    while True:
        kwargs = dict(series_ticker=series, status="settled", limit=200)
        if cursor:
            kwargs["cursor"] = cursor
        try:
            resp = client._market_api.get_markets(**kwargs)
        except Exception as e:
            print(f"    post-cutoff market list error: {e}")
            break
        page   = [_to_dict(m) for m in (_get_attr(resp, "markets", []) or [])]
        markets.extend(page)
        cursor = _get_attr(resp, "cursor")
        if not cursor or not page:
            break
        time.sleep(KALSHI_RATE_SLEEP)
    return markets


def fetch_settled_markets_pre(client, series: str) -> list[dict]:
    """Paginate settled markets via the historical (pre-cutoff) API."""
    markets, cursor = [], None
    while True:
        kwargs = dict(event_ticker=series, limit=200)
        if cursor:
            kwargs["cursor"] = cursor
        try:
            resp = client._historical_api.get_historical_markets(**kwargs)
        except Exception as e:
            print(f"    pre-cutoff market list error: {e}")
            break
        page   = [_to_dict(m) for m in (_get_attr(resp, "markets", []) or [])]
        markets.extend(page)
        cursor = _get_attr(resp, "cursor")
        if not cursor or not page:
            break
        time.sleep(KALSHI_RATE_SLEEP)
    return markets


def _parse_candles(resp, is_pre: bool) -> list[dict]:
    """Parse a candlestick response into [{ts, close_dollars, high_dollars, low_dollars, volume}]."""
    rows = []
    for c in (_get_attr(resp, "candlesticks", []) or []):
        ts        = _get_attr(c, "end_period_ts")
        price_obj = _get_attr(c, "price")
        if ts is None or price_obj is None:
            continue
        close_dollars = (
            _get_attr(price_obj, "close_dollars") or _get_attr(price_obj, "close")
        )
        if close_dollars is None:
            continue
        high_raw  = _get_attr(price_obj, "high_dollars")
        low_raw   = _get_attr(price_obj, "low_dollars")
        vol_raw   = _get_attr(c, "volume") if is_pre else _get_attr(c, "volume_fp")
        rows.append({
            "ts":            int(ts),
            "close_dollars": round(float(close_dollars), 4),
            "high_dollars":  round(float(high_raw), 4) if high_raw is not None else None,
            "low_dollars":   round(float(low_raw),  4) if low_raw  is not None else None,
            "volume":        int(float(vol_raw or 0)),
        })
    return rows


def fetch_candlesticks(client, ticker: str, series: str, sdate) -> list[dict]:
    """
    Fetch hourly candlestick data for one market.
    Routes to historical or regular API based on settlement date vs cutoff.
    Window: settlement midnight UTC ± 4 days (captures full trading period).
    """
    settle_midnight = int(
        datetime(sdate.year, sdate.month, sdate.day, tzinfo=timezone.utc).timestamp()
    )
    start_ts = settle_midnight - 4 * 86400
    end_ts   = settle_midnight + 2 * 86400
    is_pre   = settle_midnight < HISTORICAL_CUTOFF_TS

    try:
        if is_pre:
            resp = client._historical_api.get_market_candlesticks_historical(
                ticker=ticker, start_ts=start_ts, end_ts=end_ts,
                period_interval=KALSHI_PERIOD_MINS,
            )
        else:
            resp = client._market_api.get_market_candlesticks(
                series_ticker=series, ticker=ticker,
                start_ts=start_ts, end_ts=end_ts,
                period_interval=KALSHI_PERIOD_MINS,
            )
    except Exception as e:
        raise RuntimeError(f"candlestick fetch failed: {e}") from e

    return _parse_candles(resp, is_pre=is_pre)
