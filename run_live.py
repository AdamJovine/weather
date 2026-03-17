"""
Main live trading loop.

Phases:
  1. Load config and historical data
  2. Refresh GFS/ECMWF/ICON + GEFS spread with latest model runs (live APIs)
  3. Build features + fit model per city
  4. Fetch today's NWS forecasts
  5. Build prediction rows for target date
  6. Fetch Kalshi weather markets by series ticker
  7. For each market: parse, price, compute edge, log recommendation
  8. (Optional) Place orders if --live flag is set

Usage:
  python run_live.py                     # dry run — logs recommendations only
  python run_live.py --live              # places orders (use demo URL first!)
  python run_live.py --interval 30       # re-run every 30 minutes (dry run)
  python run_live.py --live --interval 30  # live trading, refreshed every 30 min
"""

import sys
import time
import argparse
import requests
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import pickle

from src.config import KALSHI_BASE_URL
from src.features import build_feature_table
from src.model import InteractionBayesModel, FEATURES
from src.kalshi_client import KalshiWeatherClient
from src.strategy import evaluate_market
from src.logger import log_trade, log_forecast, log_market_snapshot
from src.noaa_forecast import get_forecasts_for_stations
from src.ucb import CityUCB
from src.live_data import fetch_live_openmeteo, fetch_live_gefs_spread, upsert_live


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

BANKROLL = 100.0
STATIONS_FILE = Path("data/stations.csv")
HISTORY_FILE = Path("data/historical_tmax.csv")
KALSHI_RATE_SLEEP = 0.5   # seconds between Kalshi API calls


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def kalshi_get_raw(auth, path: str, params: dict = None) -> dict:
    """Direct REST call with proper auth — bypasses SDK deserialization bugs."""
    url = KALSHI_BASE_URL.rstrip("/") + path
    headers = auth.create_auth_headers("GET", path.split("?")[0])
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_weather_markets(auth, stations: pd.DataFrame, target_date: str) -> list[dict]:
    """
    Fetch open weather markets for all cities via their KXHIGH{CITY} series ticker.
    Filters to markets closing on or after target_date.
    """
    all_markets = []
    date_tag = pd.to_datetime(target_date).strftime("%y%b%d").upper()  # e.g. "26MAR17"

    for _, row in stations.iterrows():
        series = row["kalshi_series"]
        try:
            data = kalshi_get_raw(auth, "/markets", {
                "series_ticker": series,
                "limit": 50,
                "status": "open",
            })
            mkts = data.get("markets", [])
            # Keep only markets for the target date (ticker contains the date tag)
            target_mkts = [m for m in mkts if date_tag in m.get("ticker", "")]
            for m in target_mkts:
                m["_city"] = row["city"]
            all_markets.extend(target_mkts)
            print(f"  {series}: {len(target_mkts)} markets for {target_date}")
            time.sleep(KALSHI_RATE_SLEEP)
        except Exception as e:
            print(f"  {series}: ERROR — {e}")

    return all_markets


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(dry_run: bool = True):
    TARGET_DATE = (date.today() + timedelta(days=1)).isoformat()

    print(f"=== Weather Kalshi Bot — {TARGET_DATE} ===")
    print(f"Mode: {'DRY RUN (no orders)' if dry_run else '*** LIVE TRADING ***'}")
    print(f"Bankroll: ${BANKROLL:.2f}\n")

    # 1. Load data
    if not HISTORY_FILE.exists():
        print("ERROR: historical_tmax.csv not found. Run scripts/download_history.py first.")
        sys.exit(1)

    stations = pd.read_csv(STATIONS_FILE)
    hist     = pd.read_csv(HISTORY_FILE)
    gfs_path = Path("data/forecasts/openmeteo_forecast_history.csv")
    gfs_df   = pd.read_csv(gfs_path) if gfs_path.exists() else None
    indices_path = Path("data/climate_indices.csv")
    indices_df   = pd.read_csv(indices_path) if indices_path.exists() else None
    gefs_path = Path("data/forecasts/gefs_spread.csv")
    gefs_df   = pd.read_csv(gefs_path) if gefs_path.exists() else None
    print(f"Loaded {len(hist)} historical rows for {hist['city'].nunique()} cities.")
    if gfs_df is not None:
        print(f"Loaded {len(gfs_df)} GFS forecast rows.")
    if indices_df is not None:
        print(f"Loaded {len(indices_df)} climate index rows.")
    if gefs_df is not None:
        print(f"Loaded {len(gefs_df)} GEFS spread rows.")

    # 2. Refresh with latest model runs from OpenMeteo live APIs
    # GFS/ECMWF/ICON updates 4x/day; GEFS ensemble spread updates 4x/day.
    # Today's live run is inserted as today's date in gfs_df — after the
    # 1-day shift in build_feature_table() it becomes tomorrow's features.
    print("\nFetching live model data...")
    live_gfs = fetch_live_openmeteo(stations)
    if not live_gfs.empty:
        gfs_df = upsert_live(gfs_df, live_gfs)
        print(f"  OpenMeteo: refreshed {len(live_gfs)} rows for {live_gfs['city'].nunique()} cities")
    else:
        print("  OpenMeteo: no live data retrieved, using CSV cache")

    live_gefs = fetch_live_gefs_spread(stations, TARGET_DATE)
    if not live_gefs.empty:
        gefs_df = upsert_live(gefs_df, live_gefs)
        print(f"  GEFS spread: refreshed {len(live_gefs)} rows for {live_gefs['city'].nunique()} cities")
    else:
        print("  GEFS spread: no live data retrieved, using CSV cache")

    # 3. Fetch NWS forecasts
    print(f"\nFetching NWS forecasts for {TARGET_DATE}...")
    forecast_df = get_forecasts_for_stations(stations, target_date=TARGET_DATE)
    print(forecast_df.to_string(index=False))

    # Save to rolling forecast archive
    archive = Path("data/forecasts/forecast_archive.csv")
    if archive.exists():
        existing = pd.read_csv(archive)
        pd.concat([existing, forecast_df]).drop_duplicates(
            subset=["city", "target_date"]
        ).to_csv(archive, index=False)
    else:
        forecast_df.to_csv(archive, index=False)

    # 4. Build features — GFS fills historical forecast_high; NWS/NBM covers today/tomorrow
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=indices_df,
                             gefs_df=gefs_df)
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    # 5. Fit model per city
    print("\nFitting models...")
    models = {}
    for city in stations["city"]:
        city_df = df[df["city"] == city].copy()
        train_df = city_df[city_df["date"] < pd.to_datetime(TARGET_DATE)]
        model = InteractionBayesModel()
        try:
            model.fit(train_df)
            models[city] = model
            print(f"  {city}: sigma={model.sigma_:.2f}°F")
        except Exception as e:
            print(f"  {city}: model fit failed — {e}")

    # Calibrator disabled: was trained only on P(tmax>=72) events and distorts
    # the full temperature distribution when applied across all bins.
    calibrator = None
    print("Using raw model probabilities (calibrator disabled).")

    # 6. Build probability rows for target date (using freshly fetched features)
    print(f"\nBuilding predictions for {TARGET_DATE}...")
    pred_rows = {}
    for city, model in models.items():
        city_df = df[(df["city"] == city) & (df["date"] == pd.to_datetime(TARGET_DATE))]
        if city_df.empty:
            print(f"  {city}: no feature row for {TARGET_DATE}")
            continue
        if city_df[FEATURES].isnull().any(axis=1).iloc[0]:
            print(f"  {city}: NaN in features, skipping")
            continue

        mu_arr, *_ = model.predict_with_uncertainty(city_df)
        pred_mean = float(mu_arr[0])
        probs_df = model.predict_integer_probs(city_df)
        prob_row = probs_df.iloc[0]

        # Apply calibrator: rescale each temp bin's probability via isotonic regression.
        # The calibrator corrects systematic bias in the raw model probabilities.
        if calibrator is not None:
            import numpy as np
            from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX
            temp_keys = [f"temp_{k}" for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)]
            raw_probs = prob_row[temp_keys].values.astype(float)
            # Compute raw cumulative (geq) probability at each bin boundary,
            # calibrate it, then back-derive bin probabilities.
            cumsum = np.cumsum(raw_probs[::-1])[::-1]  # P(tmax >= k) for each k
            cal_cumsum = calibrator.predict(cumsum)
            cal_cumsum = np.clip(cal_cumsum, 0, 1)
            # Reconstruct bin probabilities from calibrated cumulative
            cal_bins = np.diff(np.concatenate([cal_cumsum, [0.0]]), prepend=0.0)
            cal_bins = np.abs(cal_bins)
            cal_bins /= cal_bins.sum()  # renormalize
            for key, val in zip(temp_keys, cal_bins):
                prob_row[key] = val

        pred_rows[city] = prob_row

        fcast_high = city_df.iloc[0].get("forecast_high")
        log_forecast(
            city=city,
            target_date=TARGET_DATE,
            forecast_high=fcast_high if pd.notna(fcast_high) else 0,
            pred_mean=pred_mean,
            sigma=model.sigma_,
        )
        print(f"  {city}: pred_mean={pred_mean:.1f}°F  forecast_high={fcast_high}")

    # 7. Connect to Kalshi + fetch markets
    print("\nConnecting to Kalshi...")
    try:
        kalshi = KalshiWeatherClient.from_env()
        auth = kalshi._client.kalshi_auth
    except Exception as e:
        print(f"ERROR connecting to Kalshi: {e}")
        sys.exit(1)

    print(f"Fetching weather markets for {TARGET_DATE}...")
    weather_markets = fetch_weather_markets(auth, stations, TARGET_DATE)
    print(f"  {len(weather_markets)} total markets to evaluate.\n")

    # Load UCB city allocation state
    city_ucb = CityUCB(cities=stations["city"].tolist())
    city_ucb.load_state()
    backtest_trades_path = Path("logs/pnl_backtest_trades.csv")
    if backtest_trades_path.exists():
        bt_trades = pd.read_csv(backtest_trades_path)
        if {"city", "pnl", "size"}.issubset(bt_trades.columns):
            city_ucb.initialize_from_backtest(bt_trades)
    ucb_summary = city_ucb.summary()
    print("UCB city multipliers:")
    for city_name, info in sorted(ucb_summary.items(), key=lambda x: -x[1]["multiplier"]):
        print(f"  {city_name:20s}  n={info['n']:4d}  mean_reward={info['mean_reward']:+.3f}  mult={info['multiplier']:.2f}x")
    print()

    # 8. Evaluate each market
    all_recommendations = []

    for market in weather_markets:
        city = market.get("_city")
        ticker = market.get("ticker", "")

        if city not in pred_rows:
            continue

        log_market_snapshot(
            market_ticker=ticker,
            title=market.get("title", ""),
            yes_ask=market.get("yes_ask_dollars") or 0,
            no_ask=market.get("no_ask_dollars") or 0,
            volume=market.get("volume_fp") or 0,
            open_interest=market.get("open_interest_fp") or 0,
            close_time=str(market.get("close_time") or ""),
        )

        trades = evaluate_market(
            market=market,
            prob_row=pred_rows[city],
            bankroll=BANKROLL,
            city_map={},
            city_multiplier=city_ucb.kelly_multiplier(city),
        )

        for trade in trades:
            print(
                f"  [{city}] {ticker}\n"
                f"    title:  {market.get('title')}\n"
                f"    fair_p={trade['fair_p']:.3f}  ask={trade['price_dollars']:.2f}"
                f"  edge={trade['edge']:.3f}  side={trade['side']}"
                f"  size=${trade['dollar_size']:.2f} ({trade['contract_count']:.2f} contracts)"
            )

            log_trade(
                market_ticker=ticker,
                city=city,
                market_type=trade.get("market_type", ""),
                contract_desc=market.get("title", ""),
                fair_p=trade["fair_p"],
                yes_ask=market.get("yes_ask_dollars") or 0,
                no_ask=market.get("no_ask_dollars") or 0,
                edge_yes=trade["edge"] if trade["side"] == "yes" else 0,
                edge_no=trade["edge"] if trade["side"] == "no" else 0,
                action="recommend" if dry_run else "place",
                side=trade["side"],
                size_dollars=trade["dollar_size"],
                contract_count=trade["contract_count"],
                status="dry_run" if dry_run else "pending",
            )

            all_recommendations.append(trade)

        if not trades:
            print(f"  [{city}] {ticker}: no edge  (fair={compute_fair_for_log(market, pred_rows[city])})")

    # 8. Place orders (live mode only)
    if not dry_run and all_recommendations:
        print(f"\nPlacing {len(all_recommendations)} orders...")
        for trade in all_recommendations:
            try:
                resp = kalshi.place_order(
                    ticker=trade["ticker"],
                    side=trade["side"],
                    count=max(1, round(trade["contract_count"])),
                    price=int(round(trade["price_dollars"] * 100)),
                )
                print(f"  Order placed: {resp}")
            except Exception as e:
                print(f"  Order failed for {trade['ticker']}: {e}")
    else:
        print(f"\nDry run complete: {len(all_recommendations)} recommendations logged.")

    print("Done.")


def compute_fair_for_log(market: dict, prob_row) -> str:
    """Utility to show fair value in no-edge log lines."""
    from src.strategy import parse_contract
    from src.pricing import compute_fair_prob
    try:
        contract = parse_contract(market.get("title", ""), "")
        return f"{compute_fair_prob(prob_row, contract):.3f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders (default: dry run)")
    parser.add_argument(
        "--interval", type=int, default=0, metavar="MINUTES",
        help="Re-run every N minutes using the latest model data (0 = run once)",
    )
    args = parser.parse_args()

    if args.interval > 0:
        print(f"Loop mode: refreshing every {args.interval} minutes. Ctrl+C to stop.\n")
        while True:
            main(dry_run=not args.live)
            print(f"\nNext run in {args.interval}m — waiting for next model update...\n")
            time.sleep(args.interval * 60)
    else:
        main(dry_run=not args.live)
