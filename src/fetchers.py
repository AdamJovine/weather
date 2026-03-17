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
        cache = requests_cache.CachedSession(".cache/openmeteo", expire_after=-1)
        _openmeteo_client = openmeteo_requests.Client(
            session=retry(cache, retries=5, backoff_factor=0.3)
        )
    return _openmeteo_client


def fetch_city_forecast(
    city: str, lat: float, lon: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch GFS + ECMWF day-ahead max-temperature forecasts for one city.
    Returns columns: date, city, forecast_high_gfs, forecast_high_ecmwf,
                     ensemble_spread, ecmwf_minus_gfs, precip_forecast
    """
    client = _get_openmeteo_client()
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
    responses = client.weather_api(OPENMETEO_FORECAST_URL, params=params)

    daily0 = responses[0].Daily()
    dates  = pd.date_range(
        start     = pd.to_datetime(daily0.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(daily0.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=daily0.Interval()),
        inclusive = "left",
    ).tz_localize(None).normalize()

    model_tmax, model_precip = [], []
    for r in responses:
        d = r.Daily()
        model_tmax.append(d.Variables(0).ValuesAsNumpy())
        model_precip.append(d.Variables(1).ValuesAsNumpy())

    stacked         = np.stack(model_tmax)
    tmax_gfs        = stacked[0]
    tmax_ecmwf      = stacked[2]
    ensemble_spread = np.nanstd(stacked, axis=0)
    ecmwf_minus_gfs = tmax_ecmwf - tmax_gfs
    precip_forecast = model_precip[0]

    df = pd.DataFrame({
        "date":                dates.date,
        "city":                city,
        "forecast_high_gfs":   np.round(tmax_gfs, 1),
        "forecast_high_ecmwf": np.round(tmax_ecmwf, 1),
        "ensemble_spread":     np.round(ensemble_spread, 2),
        "ecmwf_minus_gfs":     np.round(ecmwf_minus_gfs, 2),
        "precip_forecast":     np.round(precip_forecast, 2),
    })
    return df.dropna(subset=["forecast_high_gfs"])


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
    """Parse a candlestick response into [{ts, close_dollars, volume}]."""
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
        vol_raw = _get_attr(c, "volume") if is_pre else _get_attr(c, "volume_fp")
        rows.append({
            "ts":            int(ts),
            "close_dollars": round(float(close_dollars), 4),
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
