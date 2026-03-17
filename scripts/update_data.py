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
    fetch_city_forecast, fetch_gefs_spread,
    fetch_settled_markets_post, fetch_settled_markets_pre,
    fetch_candlesticks, parse_settlement_date,
    KALSHI_RATE_SLEEP,
)
from src.noaa_history import get_daily_tmax


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


# ─── OpenMeteo GFS / ECMWF ───────────────────────────────────────────────────

def update_gfs(conn) -> None:
    """
    Download missing OpenMeteo GFS/ECMWF historical forecasts and upsert
    into forecasts_daily. Incremental per city from last covered date.
    """
    print("\n[gfs] Updating OpenMeteo forecast archive...")
    stations = pd.read_csv(STATIONS_FILE)

    NEW_COLS = {"forecast_high_ecmwf", "ecmwf_minus_gfs", "precip_forecast"}
    existing = read_df(conn, "forecasts_daily")

    # If new columns are missing from DB (e.g., first run), treat as empty
    if not NEW_COLS.issubset(set(existing.columns)):
        print("  New columns not yet in DB — treating as empty for full re-fetch.")
        existing = pd.DataFrame(columns=["date", "city", "forecast_high_gfs"])

    total_new = 0
    for _, row in stations.iterrows():
        city     = row["city"]
        lat, lon = row["lat"], row["lon"]
        tz       = row["timezone"]

        if len(existing) and city in existing["city"].values:
            city_max = existing[existing["city"] == city]["date"].max()
            covered_end = str(city_max)
            if covered_end >= TRAIN_END:
                print(f"  {city}: up to date")
                continue
            fetch_start = covered_end
        else:
            fetch_start = TRAIN_START

        print(f"  {city}: {fetch_start} → {TRAIN_END}", end=" ", flush=True)
        for attempt in range(3):
            try:
                df = fetch_city_forecast(city, lat, lon, tz, fetch_start, TRAIN_END)
                n  = upsert_df(conn, "forecasts_daily", df)
                total_new += n
                print(f"→ {n} rows")
                break
            except Exception as e:
                msg = str(e)
                if "limit exceeded" in msg.lower() and attempt < 2:
                    print(f"\n    rate limited — waiting 65s...", end=" ", flush=True)
                    time.sleep(65)
                else:
                    print(f"→ ERROR: {e}")
                    break
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


# ─── Climate indices ──────────────────────────────────────────────────────────

def update_indices(conn) -> None:
    """
    Full re-fetch of all monthly climate indices (AO/NAO/ONI/PNA/PDO) and
    daily MJO. These sources are small and always published as full history,
    so a full refresh is faster than incremental.
    """
    print("\n[indices] Updating climate indices...")
    ao_df  = fetch_ao()
    nao_df = fetch_nao()
    oni_df = fetch_oni()
    pna_df = fetch_pna()
    pdo_df = fetch_pdo()
    mjo_df = fetch_mjo()

    monthly = ao_df.merge(nao_df, on=["year", "month"], how="outer")
    monthly = monthly.merge(oni_df, on=["year", "month"], how="outer")
    monthly = monthly.merge(pna_df, on=["year", "month"], how="outer")
    if not pdo_df.empty:
        monthly = monthly.merge(pdo_df, on=["year", "month"], how="outer")
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)

    n_m = upsert_df(conn, "climate_monthly", monthly)
    n_j = upsert_df(conn, "mjo_daily", mjo_df)
    conn.commit()
    print(f"  [indices] done — climate_monthly: {n_m} rows, mjo_daily: {n_j} rows")


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
    print(f"  Existing: {len(existing_tickers)} tickers in DB")

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

        new_markets = [m for m in markets if m.get("ticker", "") not in existing_tickers]
        print(f"    {len(markets)} settled markets  |  {len(new_markets)} new")

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

        # Checkpoint: write completed city to DB and commit
        if city_market_rows:
            upsert_df(conn, "kalshi_markets", pd.DataFrame(city_market_rows))
            upsert_df(conn, "kalshi_candles", pd.DataFrame(city_candle_rows))
            conn.commit()
            new_tickers = {r["ticker"] for r in city_market_rows}
            existing_tickers |= new_tickers
            print(f"    → checkpoint: {len(new_tickers)} new tickers saved")

    print(f"  [kalshi] done")


# ─── Main ─────────────────────────────────────────────────────────────────────

ALL_SOURCES = ["noaa", "gfs", "gefs", "indices", "kalshi"]

SOURCE_FNS = {
    "noaa":    update_noaa,
    "gfs":     update_gfs,
    "gefs":    update_gefs,
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
