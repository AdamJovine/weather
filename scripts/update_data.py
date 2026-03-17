"""
scripts/update_data.py

Update all data sources in data/weather.db to the most recent available data.

Sources
───────
  noaa     — NOAA GHCN-Daily observed TMAX (incremental per city, by year)
  gfs      — OpenMeteo GFS/ECMWF/ICON historical forecast archive (incremental)
  gefs     — GEFS 31-member ensemble spread (rolling window: 5d back + 7d ahead)
  indices  — Climate indices: AO/NAO/ONI/PNA/PDO (monthly) + MJO (daily)
  kalshi   — Kalshi settled-market hourly candlesticks (incremental by ticker)

Usage
─────
  python scripts/update_data.py                        # all sources
  python scripts/update_data.py --only noaa gfs        # specific sources
  python scripts/update_data.py --skip kalshi          # skip a source

Each source is incremental: only rows not already in the DB are fetched.
Kalshi checkpoints after each city so a partial run keeps completed cities.
"""

import sys
import time
import argparse
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import TRAIN_START, TRAIN_END
from src.db import get_db, upsert_df, read_df
from src.fetchers import (
    fetch_ao, fetch_nao, fetch_oni, fetch_pna, fetch_pdo, fetch_mjo,
    fetch_city_forecast, fetch_gefs_spread, fetch_obs_actuals,
    fetch_settled_markets_post, fetch_settled_markets_pre,
    fetch_candlesticks, parse_settlement_date,
    KALSHI_RATE_SLEEP,
)
from src.noaa_history import get_daily_tmax
from src.noaa_forecast import get_forecasts_for_stations


STATIONS_FILE = Path("data/stations.csv")


# ─── NOAA TMAX ────────────────────────────────────────────────────────────────

def update_noaa(conn) -> None:
    """
    Download missing NOAA GHCN-Daily TMAX and upsert into weather_daily.
    Incremental per city: finds years with fewer than 300 rows and re-fetches them.
    """
    print("\n[noaa] Updating historical TMAX...")
    stations = pd.read_csv(STATIONS_FILE)

    target_start = pd.to_datetime(TRAIN_START).year
    target_end   = pd.to_datetime(TRAIN_END).year

    existing = read_df(conn, "weather_daily")

    total_new = 0
    for _, row in stations.iterrows():
        city       = row["city"]
        station_id = row["station_id"]

        city_data = existing[existing["city"] == city].copy()
        city_data["_year"] = pd.to_datetime(city_data["date"], errors="coerce").dt.year
        rows_per_year = city_data.groupby("_year").size()

        gaps = [
            (f"{yr}-01-01", f"{yr}-12-31")
            for yr in range(target_start, target_end + 1)
            if rows_per_year.get(yr, 0) < 300
        ]

        if not gaps:
            print(f"  {city}: up to date")
            continue

        print(f"  {city}: {len(gaps)} gap year(s)...")
        for gap_start, gap_end in gaps:
            print(f"    {gap_start} → {gap_end}", end=" ", flush=True)
            try:
                df = get_daily_tmax(station_id=station_id,
                                    start_date=gap_start, end_date=gap_end)
                df["city"] = city
                n = upsert_df(conn, "weather_daily", df)
                total_new += n
                print(f"→ {n} rows")
            except Exception as e:
                print(f"→ ERROR: {e}")

    conn.commit()
    print(f"  [noaa] done — {total_new} new rows written")


# ─── OpenMeteo ERA5 actuals backfill ──────────────────────────────────────────

def update_obs(conn) -> None:
    """
    Fill recent gaps in weather_daily using the OpenMeteo ERA5 reanalysis archive.

    NOAA GHCN-Daily can lag by weeks or months; ERA5 fills the gap with high-
    quality reanalysis data updated daily (~5-day lag).  Only dates not already
    present in weather_daily are inserted — NOAA observations take priority and
    are never overwritten.
    """
    print("\n[obs] Filling recent weather_daily gaps via OpenMeteo ERA5 archive...")
    stations = pd.read_csv(STATIONS_FILE)

    today = date.today()
    # ERA5 is published with a ~5-day lag; cap the end date to avoid requesting
    # data that hasn't been processed yet.
    ERA5_LAG_DAYS = 5
    fetch_end = str(today - timedelta(days=ERA5_LAG_DAYS))

    existing = read_df(conn, "weather_daily")

    total_new = 0
    for _, row in stations.iterrows():
        city     = row["city"]
        lat, lon = row["lat"], row["lon"]
        tz       = row["timezone"]

        city_existing = existing[existing["city"] == city] if not existing.empty else pd.DataFrame()
        city_dates = set(pd.to_datetime(city_existing["date"]).dt.date) if not city_existing.empty else set()

        if city_dates:
            fetch_start = str(max(city_dates) + timedelta(days=1))
        else:
            fetch_start = TRAIN_START

        if fetch_start > fetch_end:
            print(f"  {city}: up to date")
            continue

        print(f"  {city}: {fetch_start} → {fetch_end}", end=" ", flush=True)
        try:
            df = fetch_obs_actuals(city, lat, lon, tz, fetch_start, fetch_end)
            # Exclude any dates already covered (belt-and-suspenders guard)
            if not df.empty and city_dates:
                df = df[~df["date"].isin(city_dates)]
            n = upsert_df(conn, "weather_daily", df) if not df.empty else 0
            total_new += n
            print(f"→ {n} rows")
        except Exception as e:
            print(f"→ ERROR: {e}")
        time.sleep(0.4)

    conn.commit()
    print(f"  [obs] done — {total_new} new rows written")


# ─── OpenMeteo GFS / ECMWF ───────────────────────────────────────────────────

def update_gfs(conn) -> None:
    """
    Download missing OpenMeteo GFS/ECMWF historical forecasts and upsert
    into forecasts_daily. Incremental per city from last covered date.

    The target end date extends 2 days past today so that tomorrow's forecast
    row is always in the DB before run_live.py executes.
    """
    print("\n[gfs] Updating OpenMeteo forecast archive...")
    stations = pd.read_csv(STATIONS_FILE)

    # Quick health probe before looping all cities — avoids hammering a down API
    try:
        import requests as _requests
        r = _requests.get(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params={"latitude": 40.71, "longitude": -74.01,
                    "start_date": str(date.today()), "end_date": str(date.today()),
                    "daily": "temperature_2m_max", "timezone": "America/New_York",
                    "models": ["gfs_seamless", "icon_seamless", "ecmwf_ifs"]},
            timeout=10,
        )
        r.raise_for_status()
        print("  OpenMeteo probe: OK")
    except Exception as e:
        print(f"  OpenMeteo probe FAILED — skipping GFS update (check https://status.open-meteo.com): {e}")
        return

    # Always extend through at least today+2 so live trading rows exist
    gfs_end = max(TRAIN_END, str(date.today() + timedelta(days=2)))

    # Columns whose presence in the schema is necessary (structural migration check)
    NEW_SCHEMA_COLS = {"forecast_high_ecmwf", "ecmwf_minus_gfs", "precip_forecast",
                       "temp_850hpa", "shortwave_radiation", "dew_point_max"}
    # Columns added in the most recent feature expansion — must have actual data,
    # not just NULL entries from the ALTER TABLE migration.
    NEW_DATA_COLS = ["temp_850hpa", "shortwave_radiation", "dew_point_max"]

    existing = read_df(conn, "forecasts_daily")

    # If schema columns are missing entirely, treat as empty → full re-fetch all cities
    if not NEW_SCHEMA_COLS.issubset(set(existing.columns)):
        print("  New columns not yet in DB — treating as empty for full re-fetch.")
        existing = pd.DataFrame(columns=["date", "city", "forecast_high_gfs"])

    total_new = 0
    for _, row in stations.iterrows():
        city     = row["city"]
        lat, lon = row["lat"], row["lon"]
        tz       = row["timezone"]

        if len(existing) and city in existing["city"].values:
            city_rows = existing[existing["city"] == city]
            city_max  = city_rows["date"].max()
            covered_end = str(city_max)

            # If new data columns exist in schema but are entirely NULL for this city,
            # the migration ran but no data was fetched yet — force full re-fetch.
            data_cols_present = [c for c in NEW_DATA_COLS if c in city_rows.columns]
            needs_backfill = (
                data_cols_present
                and not city_rows[data_cols_present].notna().any(axis=None)
            )

            if needs_backfill:
                fetch_start = TRAIN_START
                print(f"  {city}: backfilling new columns from {fetch_start}", end=" ", flush=True)
            elif covered_end >= gfs_end:
                print(f"  {city}: up to date")
                continue
            else:
                fetch_start = covered_end
        else:
            fetch_start = TRAIN_START

        print(f"  {city}: {fetch_start} → {gfs_end}", end=" ", flush=True)
        for attempt in range(2):
            try:
                df = fetch_city_forecast(city, lat, lon, tz, fetch_start, gfs_end)
                n  = upsert_df(conn, "forecasts_daily", df)
                total_new += n
                print(f"→ {n} rows")
                break
            except Exception as e:
                if attempt == 0:
                    wait = 65 if "limit exceeded" in str(e).lower() else 15
                    print(f"\n    error — retrying in {wait}s: {e}", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"→ ERROR: {e}")
        time.sleep(4)

    conn.commit()
    print(f"  [gfs] done — {total_new} new rows written")


# ─── GEFS spread ──────────────────────────────────────────────────────────────

def update_gefs(conn) -> None:
    """
    Fetch GEFS ensemble spread for a rolling window (5 days back + 7 days
    ahead) and upsert into gefs_spread. Always re-fetches the window since
    OpenMeteo's free tier only keeps ~5 days of GEFS history.
    """
    print("\n[gefs] Updating GEFS ensemble spread...")
    stations    = pd.read_csv(STATIONS_FILE)
    today       = date.today()
    fetch_start = str(today - timedelta(days=5))
    fetch_end   = str(today + timedelta(days=7))

    total_new = 0
    for _, row in stations.iterrows():
        city     = row["city"]
        lat, lon = row["lat"], row["lon"]
        tz       = row["timezone"]

        print(f"  {city}: {fetch_start} → {fetch_end}", end=" ", flush=True)
        try:
            df = fetch_gefs_spread(city, lat, lon, tz, fetch_start, fetch_end)
            n  = upsert_df(conn, "gefs_spread", df)
            total_new += n
            print(f"→ {n} rows  spread_mean={df['gefs_spread'].mean():.2f}°F")
        except Exception as e:
            print(f"→ ERROR: {e}")
        time.sleep(0.4)

    conn.commit()
    print(f"  [gefs] done — {total_new} rows upserted")


# ─── NWS forecasts ───────────────────────────────────────────────────────────

def update_nws(conn) -> None:
    """
    Fetch NWS textual forecast highs and NBM gridded highs for all stations
    for today through today+7 and upsert into nws_forecasts.

    Run this immediately before run_live.py so the DB always has fresh
    NWS forecasts for the upcoming trading window.
    """
    print("\n[nws] Updating NWS/NBM forecasts...")
    stations = pd.read_csv(STATIONS_FILE)

    total_new = 0
    for days_ahead in range(0, 8):  # include today (day 0) for same-day trading
        target_date = str(date.today() + timedelta(days=days_ahead))
        print(f"  Fetching forecasts for {target_date}...")
        try:
            df = get_forecasts_for_stations(stations, target_date=target_date)
            df = df.rename(columns={"target_date": "target_date"})
            n = upsert_df(conn, "nws_forecasts", df)
            total_new += n
        except Exception as e:
            print(f"    ERROR for {target_date}: {e}")

    conn.commit()
    print(f"  [nws] done — {total_new} rows upserted")


# ─── Climate indices ──────────────────────────────────────────────────────────

def _monthly_is_current(conn) -> bool:
    """True if climate_monthly has data for last month or later."""
    try:
        row = conn.execute(
            "SELECT MAX(year * 100 + month) FROM climate_monthly"
        ).fetchone()
        if not row or not row[0]:
            return False
        yyyymm = int(row[0])
        latest_year, latest_month = yyyymm // 100, yyyymm % 100
        today = date.today()
        # Last month's yyyymm
        last_month = today.month - 1 or 12
        last_month_year = today.year if today.month > 1 else today.year - 1
        return (latest_year > last_month_year) or (
            latest_year == last_month_year and latest_month >= last_month
        )
    except Exception:
        return False


def _mjo_is_current(conn) -> bool:
    """True if mjo_daily has data within the last 5 days."""
    try:
        row = conn.execute("SELECT MAX(date) FROM mjo_daily").fetchone()
        if not row or not row[0]:
            return False
        latest = pd.to_datetime(row[0]).date()
        return (date.today() - latest).days <= 5
    except Exception:
        return False


def update_indices(conn) -> None:
    """
    Conditionally re-fetch climate indices — only when data is stale.

    Monthly indices (AO/NAO/ONI/PNA/PDO): skipped if DB already has data
    through last month (indices are published with ~1 month lag).

    MJO: skipped if last DB row is within 5 days of today (published daily).
    """
    print("\n[indices] Updating climate indices...")
    n_m, n_j = 0, 0

    monthly_current = _monthly_is_current(conn)
    mjo_current     = _mjo_is_current(conn)

    if monthly_current:
        print("  climate_monthly: up to date — skipping")
    else:
        ao_df  = fetch_ao()
        nao_df = fetch_nao()
        oni_df = fetch_oni()
        pna_df = fetch_pna()
        pdo_df = fetch_pdo()

        monthly = ao_df.merge(nao_df, on=["year", "month"], how="outer")
        monthly = monthly.merge(oni_df, on=["year", "month"], how="outer")
        monthly = monthly.merge(pna_df, on=["year", "month"], how="outer")
        if not pdo_df.empty:
            monthly = monthly.merge(pdo_df, on=["year", "month"], how="outer")
        monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)
        n_m = upsert_df(conn, "climate_monthly", monthly)

    if mjo_current:
        print("  mjo_daily: up to date — skipping")
    else:
        mjo_df = fetch_mjo()
        n_j = upsert_df(conn, "mjo_daily", mjo_df)

    conn.commit()
    print(f"  [indices] done — climate_monthly: {n_m} new rows, mjo_daily: {n_j} new rows")


# ─── Kalshi price history ─────────────────────────────────────────────────────

def update_kalshi(conn) -> None:
    """
    Download Kalshi settled-market candlestick history and upsert into
    kalshi_markets + kalshi_candles. Incremental: skips tickers already in DB.
    Checkpoints after each city so interrupted runs keep completed cities.
    """
    from src.kalshi_client import KalshiWeatherClient

    print("\n[kalshi] Updating Kalshi price history...")
    stations = pd.read_csv(STATIONS_FILE)

    print("  Connecting to Kalshi...")
    kalshi = KalshiWeatherClient.from_env()
    client = kalshi._client

    # Load existing tickers from DB
    existing_tickers: set[str] = set(
        pd.read_sql("SELECT ticker FROM kalshi_markets", conn)["ticker"].tolist()
    )
    no_candles_tickers: set[str] = set(
        pd.read_sql("SELECT ticker FROM kalshi_no_candles", conn)["ticker"].tolist()
    )
    print(f"  Existing: {len(existing_tickers)} tickers in DB  |  {len(no_candles_tickers)} known-missing cached")

    for _, station in stations.iterrows():
        city   = station["city"]
        series = station["kalshi_series"]
        print(f"\n  {city}  ({series})")

        # Fetch market listings from both API paths
        post_markets, pre_markets = [], []
        try:
            post_markets = fetch_settled_markets_post(client, series)
        except Exception as e:
            print(f"    post-cutoff listing error: {e}")
        try:
            pre_markets = fetch_settled_markets_pre(client, series)
        except Exception as e:
            print(f"    pre-cutoff listing error: {e}")

        # Deduplicate by ticker (post + pre may overlap around the cutoff)
        seen, markets = set(), []
        for m in post_markets + pre_markets:
            t = m.get("ticker", "")
            if t and t not in seen:
                seen.add(t)
                markets.append(m)

        # Settlement dates already confirmed to have no candle data for this series.
        # If any ticker for a date is known missing, the whole date is missing —
        # Kalshi either didn't publish candles or the market had no trades.
        no_candle_dates: set = {
            parse_settlement_date(t)
            for t in no_candles_tickers
            if t.startswith(series + "-")
        } - {None}

        skip = existing_tickers | no_candles_tickers
        new_markets = [
            m for m in markets
            if m.get("ticker", "") not in skip
            and parse_settlement_date(m.get("ticker", "")) not in no_candle_dates
        ]
        skipped_dates = len(markets) - len(new_markets) - sum(
            1 for m in markets if m.get("ticker", "") in existing_tickers
        )
        print(f"    {len(markets)} settled markets  |  {len(new_markets)} new"
              + (f"  |  {skipped_dates} skipped (known-missing dates)" if skipped_dates > 0 else ""))

        city_market_rows, city_candle_rows = [], []
        for idx, market in enumerate(new_markets):
            ticker = market.get("ticker", "")
            title  = market.get("title", "")
            sdate  = parse_settlement_date(ticker)
            if not ticker or not sdate:
                print(f"    [{idx+1}/{len(new_markets)}] {ticker}: "
                      f"could not parse date — skip")
                continue

            try:
                candles = fetch_candlesticks(client, ticker, series, sdate)
            except Exception as e:
                print(f"    [{idx+1}/{len(new_markets)}] {ticker}: {e}")
                time.sleep(KALSHI_RATE_SLEEP)
                continue

            if not candles:
                print(f"    [{idx+1}/{len(new_markets)}] {ticker}: no candle data")
                conn.execute(
                    "INSERT OR REPLACE INTO kalshi_no_candles (ticker, checked_at) VALUES (?, ?)",
                    (ticker, str(date.today())),
                )
                no_candles_tickers.add(ticker)
                time.sleep(KALSHI_RATE_SLEEP)
                continue

            city_market_rows.append({
                "ticker": ticker, "title": title,
                "series": series, "city": city,
                "settlement_date": str(sdate),
            })
            for c in candles:
                city_candle_rows.append({"ticker": ticker, **c})

            prices = [c["close_dollars"] for c in candles]
            print(f"    [{idx+1}/{len(new_markets)}] {ticker}  "
                  f"sdate={sdate}  candles={len(candles)}  "
                  f"price=[{min(prices):.2f},{max(prices):.2f}]")

            time.sleep(KALSHI_RATE_SLEEP)

        # Checkpoint: write completed city to DB and commit.
        # Always commit — ensures no_candles entries are persisted even when
        # a city returns only missing-candle tickers and no valid rows.
        if city_market_rows:
            upsert_df(conn, "kalshi_markets", pd.DataFrame(city_market_rows))
            upsert_df(conn, "kalshi_candles", pd.DataFrame(city_candle_rows))
            new_tickers = {r["ticker"] for r in city_market_rows}
            existing_tickers |= new_tickers
            print(f"    → checkpoint: {len(new_tickers)} new tickers saved")
        conn.commit()

    print(f"  [kalshi] done")


# ─── Main ─────────────────────────────────────────────────────────────────────

ALL_SOURCES = ["noaa", "obs", "indices", "gfs", "gefs", "nws", "kalshi"]

SOURCE_FNS = {
    "noaa":    update_noaa,
    "obs":     update_obs,
    "gfs":     update_gfs,
    "gefs":    update_gefs,
    "nws":     update_nws,
    "indices": update_indices,
    "kalshi":  update_kalshi,
}


def main():
    parser = argparse.ArgumentParser(
        description="Update data/weather.db with the latest data from all sources."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", nargs="+", choices=ALL_SOURCES, metavar="SOURCE",
        help=f"Only update these sources: {ALL_SOURCES}",
    )
    group.add_argument(
        "--skip", nargs="+", choices=ALL_SOURCES, metavar="SOURCE",
        help="Skip these sources (run all others)",
    )
    args = parser.parse_args()

    if args.only:
        sources = args.only
    elif args.skip:
        sources = [s for s in ALL_SOURCES if s not in args.skip]
    else:
        sources = ALL_SOURCES

    print(f"Updating sources: {sources}")

    with get_db() as conn:
        for source in sources:
            SOURCE_FNS[source](conn)

    print("\nAll done.")


if __name__ == "__main__":
    main()
