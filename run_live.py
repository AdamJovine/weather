"""
Main live trading loop.

Phases:
  1. Load all data from DB (run scripts/update_data.py first to refresh)
  2. Build features + fit model per city
  3. Build prediction rows for target date
  4. Fetch live Kalshi weather markets + current positions
  5. Evaluate edge via PortfolioManager, log recommendations
  6. (Optional) Place orders if --live flag is set

Usage:
  python run_live.py                     # dry run — logs recommendations only
  python run_live.py --live              # places orders (use demo URL first!)
  python run_live.py --interval 30       # re-run every 30 minutes (dry run)
  python run_live.py --live --interval 30  # live trading, refreshed every 30 min

IMPORTANT: Always run scripts/update_data.py before this script to ensure
           all data sources (GFS, GEFS, NWS, NOAA) are fresh in the DB.
"""

import json
import math
import pickle
import subprocess
import sys
import time
import argparse
import requests
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("scripts")))
import tune_hyperparams as _th

from src.app_config import cfg as _cfg
from src.config import KALSHI_BASE_URL
from src.features import build_feature_table
import src.model as _model_module
from src.model import FEATURES
from src.kalshi_client import KalshiWeatherClient
from src.logger import log_trade, log_forecast, log_market_snapshot
from src.ucb import CityUCB
from src.db import DB_PATH, get_db, check_freshness_db
from src.data_loader_shared import load_raw_sources
from src.portfolio_manager import PortfolioManager, OrderIntent
from src.fetchers import fetch_current_obs
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

BANKROLL               = _cfg.live.bankroll
CASH_RESERVE_FRACTION  = _cfg.live.cash_reserve_fraction
ROTATION_MIN_EDGE_GAIN = _cfg.live.rotation_min_edge_gain
MAX_BET_DOLLARS        = _cfg.live.max_bet_dollars
MAX_SESSION_TRADES     = _cfg.live.max_session_trades
MAX_DAILY_TRADES       = _cfg.live.max_daily_trades

OUT_ROOT      = Path(_cfg.paths.tune_output)
STATIONS_FILE = Path(_cfg.paths.stations)
KALSHI_RATE_SLEEP = _cfg.live.kalshi_rate_sleep


# ------------------------------------------------------------------
# Intraday running-max constraint
# ------------------------------------------------------------------

def _apply_running_max_constraint(prob_row: "pd.Series", running_max_f: float) -> "pd.Series":
    """
    Zero out all probability mass for integer temperatures strictly below
    the observed intraday running maximum, then renormalize.

    The daily high is monotonically non-decreasing: once the METAR reports
    88°F, the final high cannot be less than 88°F.  Applying this constraint
    sharpens the model sharply on same-day trades.

    Parameters
    ----------
    prob_row      : Series with keys "temp_{k}" for k in TEMP_GRID_MIN..MAX
    running_max_f : Observed intraday high in °F (float from METAR)

    Returns a copy of prob_row with updated probabilities.
    """
    cutoff = math.floor(running_max_f)   # lowest possible integer tmax
    row = prob_row.copy()
    for k in range(TEMP_GRID_MIN, cutoff):
        key = f"temp_{k}"
        if key in row.index:
            row[key] = 0.0
    # Renormalize remaining mass
    total = sum(float(row.get(f"temp_{k}", 0.0)) for k in range(cutoff, TEMP_GRID_MAX + 1))
    if total > 0:
        for k in range(cutoff, TEMP_GRID_MAX + 1):
            key = f"temp_{k}"
            if key in row.index:
                row[key] = float(row[key]) / total
    return row


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


def fetch_live_market_price(auth, ticker: str) -> dict | None:
    """
    Fetch the current yes_ask_dollars and no_ask_dollars for a single market.
    Returns None if the fetch fails or prices are unavailable.
    This is called immediately before order placement to ensure we trade at live prices.
    """
    try:
        data = kalshi_get_raw(auth, f"/markets/{ticker}")
        market = data.get("market", data)  # some endpoints wrap in "market" key
        yes_ask = market.get("yes_ask_dollars")
        no_ask  = market.get("no_ask_dollars")
        if yes_ask is None or no_ask is None:
            return None
        return {
            "yes_ask_dollars": float(yes_ask),
            "no_ask_dollars":  float(no_ask),
        }
    except Exception as e:
        print(f"  Warning: could not refresh price for {ticker}: {e}")
        return None


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

def main(dry_run: bool = True, model_name: str = _cfg.live.model_name, best_json: str | None = None):
    from datetime import datetime as _dt

    # Reload best params on every cycle so mega_tune improvements are picked up live
    best_path = Path(best_json) if best_json else OUT_ROOT / model_name / "best.json"
    if not best_path.exists():
        print(f"ERROR: no best.json for model '{model_name}' at {best_path}")
        sys.exit(1)
    _best_doc = json.load(open(best_path))
    _BEST = _best_doc["params"]
    _model_module.SPREAD_ALPHA = float(_BEST["spread_alpha"])

    _now = _dt.now()
    # Before 4 pm local time, today's markets are still open — trade same-day.
    # After 4 pm, target tomorrow.
    if _now.hour < _cfg.live.target_date_cutoff_hour:
        TARGET_DATE = date.today().isoformat()
    else:
        TARGET_DATE = (date.today() + timedelta(days=1)).isoformat()

    TOMORROW_DATE = (pd.to_datetime(TARGET_DATE) + timedelta(days=1)).date().isoformat()

    print(f"=== Weather Kalshi Bot — {model_name} — {TARGET_DATE} + {TOMORROW_DATE} ===")
    print(f"Mode: {'DRY RUN (no orders)' if dry_run else '*** LIVE TRADING ***'}")
    print(f"Bankroll: ${BANKROLL:.2f}\n")

    # 1. Refresh all data sources (GFS, GEFS, NWS, NOAA) into the DB
    print("Refreshing data...")
    result = subprocess.run([sys.executable, "scripts/update_data.py"], check=False)
    if result.returncode != 0:
        print("WARNING: update_data.py returned non-zero — proceeding with existing DB data.")
    print()

    # 2. Load all data from DB
    stations = pd.read_csv(STATIONS_FILE)

    if not DB_PATH.exists():
        print("ERROR: database not found. Run scripts/update_data.py first.")
        sys.exit(1)

    print(f"Reading from {DB_PATH}...")
    sources = load_raw_sources(DB_PATH, nws_target_date=TARGET_DATE, verbose=True)
    hist         = sources["hist"]
    gfs_df       = sources["gfs"]
    gefs_df      = sources["gefs"]
    indices_df   = sources["indices"]
    mjo_df       = sources["mjo"]
    forecast_df  = sources["nws"]
    # Load Kalshi series separately (trading-specific)
    with get_db() as conn:
        _ph = pd.read_sql("SELECT DISTINCT series FROM kalshi_markets", conn)
        nws_tmrw = pd.read_sql(
            "SELECT city, forecast_high, nbm_high, target_date "
            "FROM nws_forecasts WHERE target_date = ?",
            conn,
            params=(TOMORROW_DATE,),
        )

    tradeable_series = set(_ph["series"].tolist()) if not _ph.empty else None

    # NWS forecasts for target date — mandatory for live prediction
    if forecast_df.empty:
        print(f"WARNING: no NWS forecasts in DB for {TARGET_DATE}. "
              "Run: python scripts/update_data.py --only nws")
        forecast_df = pd.DataFrame(columns=["city", "forecast_high", "nbm_high", "target_date"])
    else:
        print(f"NWS forecasts loaded for {TARGET_DATE}:")
        print(forecast_df.to_string(index=False))

    # Append tomorrow's NWS forecasts so feature table covers both dates
    if nws_tmrw.empty:
        print(f"WARNING: no NWS forecasts in DB for {TOMORROW_DATE}. "
              "Run: python scripts/update_data.py --only nws")
    else:
        print(f"NWS forecasts loaded for {TOMORROW_DATE}:")
        print(nws_tmrw.to_string(index=False))
        forecast_df = pd.concat([forecast_df, nws_tmrw], ignore_index=True)

    # Restrict trading to cities with Kalshi price history
    if tradeable_series is not None:
        tradeable_stations = stations[
            stations["kalshi_series"].isin(tradeable_series)
        ].copy()
        skipped = set(stations["kalshi_series"]) - tradeable_series
        print(f"Tradeable cities: {len(tradeable_stations)}/{len(stations)} "
              f"(skipping {len(skipped)} without price history)")
    else:
        tradeable_stations = stations.copy()
        print("Warning: no Kalshi price history — trading all cities (no history filter)")

    # Freshness check — abort if any data source is stale
    print("\n--- Data freshness ---")
    with get_db() as conn:
        for r in check_freshness_db(conn):
            print(r.summary())
    print()

    # 2. Build features — GFS fills historical forecast_high; NWS/NBM covers target date
    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=indices_df,
                             gefs_df=gefs_df, mjo_df=mjo_df)
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]

    # 3. Fit model per city
    print("\nFitting models...")
    models = {}
    for city in stations["city"]:
        city_df = df[df["city"] == city].copy()
        train_df = city_df[city_df["date"] < pd.to_datetime(TARGET_DATE)]
        model = _th.make_model(model_name, _BEST)
        model._lookback = int(_BEST["lookback"])
        try:
            model.fit(train_df)
            models[city] = model
            print(f"  {city}: sigma={model.sigma_:.2f}°F")
        except Exception as e:
            print(f"  {city}: model fit failed — {e}")

    # Load pre-fitted probability calibrator if available.
    # Generate with: python scripts/fit_calibrator.py
    _cal_path = Path("logs/calibrator.pkl")
    if _cal_path.exists():
        with open(_cal_path, "rb") as _f:
            calibrator = pickle.load(_f)
        print(f"Loaded probability calibrator from {_cal_path}")
    else:
        calibrator = None
        print("No calibrator found — using raw model probabilities. Run scripts/fit_calibrator.py to generate one.")

    # 4. Build probability rows for target date
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
            temp_keys = [f"temp_{k}" for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)]
            raw_probs = prob_row[temp_keys].values.astype(float)
            # P(tmax >= k) for each k in the grid — decreasing array
            cumsum = np.cumsum(raw_probs[::-1])[::-1]
            cal_cumsum = calibrator.predict(cumsum)
            cal_cumsum = np.clip(cal_cumsum, 0, 1)
            # P(tmax == k) = P(tmax >= k) - P(tmax >= k+1)
            cal_bins = cal_cumsum - np.concatenate([cal_cumsum[1:], [0.0]])
            cal_bins = np.maximum(cal_bins, 0.0)
            if cal_bins.sum() > 0:
                cal_bins /= cal_bins.sum()
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

    # 4b. Same-day only: fetch live METAR obs and apply running-max constraint.
    # Once a station reports e.g. 88°F, the daily high cannot be below 88°F.
    # Zeroing probability mass below the running max sharpens the distribution
    # and produces near-certain fair_p values when the outcome is already decided.
    is_same_day = (TARGET_DATE == date.today().isoformat())
    if is_same_day:
        print("\nFetching live METAR observations (same-day trading)...")
        obs = fetch_current_obs(stations)
        if obs:
            for city, info in sorted(obs.items()):
                running_max = info["running_max_f"]
                current     = info["current_temp_f"]
                print(f"  {city:20s}  current={current:.1f}°F  "
                      f"running_max={running_max:.1f}°F  "
                      f"obs={info['obs_count']}  @{info['latest_obs_utc']}")
                if city in pred_rows:
                    pred_rows[city] = _apply_running_max_constraint(
                        pred_rows[city], running_max
                    )
        else:
            print("  No observations returned — proceeding with model-only probabilities.")
        print()

    # 4c. Build probability rows for tomorrow
    print(f"\nBuilding predictions for {TOMORROW_DATE}...")
    pred_rows_tmrw = {}
    for city, model in models.items():
        city_df = df[(df["city"] == city) & (df["date"] == pd.to_datetime(TOMORROW_DATE))]
        if city_df.empty:
            print(f"  {city}: no feature row for {TOMORROW_DATE}")
            continue
        if city_df[FEATURES].isnull().any(axis=1).iloc[0]:
            print(f"  {city}: NaN in features for {TOMORROW_DATE}, skipping")
            continue

        mu_arr, *_ = model.predict_with_uncertainty(city_df)
        pred_mean = float(mu_arr[0])
        probs_df = model.predict_integer_probs(city_df)
        prob_row = probs_df.iloc[0]

        if calibrator is not None:
            import numpy as np
            temp_keys = [f"temp_{k}" for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)]
            raw_probs = prob_row[temp_keys].values.astype(float)
            cumsum = np.cumsum(raw_probs[::-1])[::-1]
            cal_cumsum = calibrator.predict(cumsum)
            cal_cumsum = np.clip(cal_cumsum, 0, 1)
            cal_bins = cal_cumsum - np.concatenate([cal_cumsum[1:], [0.0]])
            cal_bins = np.maximum(cal_bins, 0.0)
            if cal_bins.sum() > 0:
                cal_bins /= cal_bins.sum()
            for key, val in zip(temp_keys, cal_bins):
                prob_row[key] = val

        pred_rows_tmrw[city] = prob_row

        fcast_high = city_df.iloc[0].get("forecast_high")
        print(f"  {city}: pred_mean={pred_mean:.1f}°F  forecast_high={fcast_high}")

    # 5. Connect to Kalshi + fetch live markets and positions
    print("\nConnecting to Kalshi...")
    try:
        kalshi = KalshiWeatherClient.from_env()
        auth = kalshi._client.kalshi_auth
    except Exception as e:
        print(f"ERROR connecting to Kalshi: {e}")
        sys.exit(1)

    print(f"Fetching weather markets for {TARGET_DATE}...")
    weather_markets = fetch_weather_markets(auth, tradeable_stations, TARGET_DATE)
    print(f"  {len(weather_markets)} markets for {TARGET_DATE}.")

    print(f"Fetching weather markets for {TOMORROW_DATE}...")
    weather_markets_tmrw = fetch_weather_markets(auth, tradeable_stations, TOMORROW_DATE)
    print(f"  {len(weather_markets_tmrw)} markets for {TOMORROW_DATE}.\n")

    # Load UCB city allocation state.
    # Warm-start from the best trial's trades CSV, which lives alongside best.json.
    # best.json records n_evals (the trial number of the best result) so we can
    # reconstruct the exact path: <best_dir>/trial_NNNN_trades.csv
    city_ucb = CityUCB(cities=tradeable_stations["city"].tolist())
    city_ucb.load_state()
    best_n_evals = int(_best_doc.get("n_evals", 0))
    backtest_trades_path = best_path.parent / f"trial_{best_n_evals:04d}_trades.csv"
    if backtest_trades_path.exists():
        bt_trades = pd.read_csv(backtest_trades_path)
        if {"city", "pnl", "size"}.issubset(bt_trades.columns):
            city_ucb.initialize_from_backtest(bt_trades)
            print(f"UCB warm-started from {backtest_trades_path} ({len(bt_trades)} trades)")
    else:
        print(f"UCB: no backtest trades found at {backtest_trades_path} — starting cold")
    ucb_summary = city_ucb.summary()
    print("UCB city multipliers:")
    for city_name, info in sorted(ucb_summary.items(), key=lambda x: -x[1]["multiplier"]):
        print(f"  {city_name:20s}  n={info['n']:4d}  mean_reward={info['mean_reward']:+.3f}  mult={info['multiplier']:.2f}x")
    print()

    # 8. Fetch current positions so we only trade the delta on each loop iteration.
    # Kalshi "position" is net yes contracts: positive = long yes, negative = long no.
    # Without this, looping every 30 min would stack a full new Kelly position each time.
    try:
        raw_positions = kalshi.get_positions()
        positions = {p["ticker"]: p.get("position", 0) for p in raw_positions}
        print(f"Fetched {len(positions)} open positions.")
    except Exception as e:
        print(f"Warning: could not fetch positions ({e}) — assuming no open positions.")
        positions = {}

    # 6. Build PortfolioManager and evaluate all markets (entries + exits)
    pm = PortfolioManager(
        kelly_fraction=float(_BEST["kelly_fraction"]),
        max_bet_fraction=float(_BEST["max_bet_frac"]),
        max_bet_dollars=MAX_BET_DOLLARS,
        min_edge=float(_BEST["min_edge"]),
        min_confidence=float(_BEST["min_confidence"]),
        min_fair_p=float(_BEST.get("min_fair_p", 0.05)),
        max_fair_p=float(_BEST.get("max_fair_p", 0.95)),
        min_mkt_price=float(_BEST.get("min_mkt_price", 0.0)),
        cash_reserve_fraction=CASH_RESERVE_FRACTION,
        rotation_min_edge_gain=ROTATION_MIN_EDGE_GAIN,
        max_session_trades=MAX_SESSION_TRADES,
    )
    city_multipliers = {
        city: city_ucb.kelly_multiplier(city)
        for city in tradeable_stations["city"].tolist()
    }

    # Log market snapshots for every market (pre-pass, regardless of edge)
    for market in weather_markets + weather_markets_tmrw:
        ticker = market.get("ticker", "")
        log_market_snapshot(
            market_ticker=ticker,
            title=market.get("title", ""),
            yes_ask=market.get("yes_ask_dollars") or 0,
            no_ask=market.get("no_ask_dollars") or 0,
            volume=market.get("volume_fp") or 0,
            open_interest=market.get("open_interest_fp") or 0,
            close_time=str(market.get("close_time") or ""),
        )

    # 7. Evaluate all markets via PortfolioManager (exits first, then entries)
    # Run separately per date so each uses the correct prediction distribution.
    # Both the bankroll budget and the per-session trade count are split across
    # the two calls so today + tomorrow together count as one session.
    intents_today = pm.evaluate(weather_markets, pred_rows, positions, BANKROLL, city_multipliers,
                                session_slots=MAX_SESSION_TRADES)
    buys_used_today = sum(1 for i in intents_today if i.action == "buy")
    deployed_today  = sum(i.size for i in intents_today if i.action == "buy")
    bankroll_remaining = max(0.0, BANKROLL - deployed_today)
    slots_remaining    = max(0, MAX_SESSION_TRADES - buys_used_today)
    intents_tmrw = pm.evaluate(weather_markets_tmrw, pred_rows_tmrw, positions, bankroll_remaining,
                               city_multipliers, session_slots=slots_remaining)
    intents = intents_today + intents_tmrw

    sell_intents = [i for i in intents if i.action == "sell"]
    buy_intents  = [i for i in intents if i.action == "buy"]

    # Log and optionally place sell (exit) orders
    print("--- Checking open positions for exits ---")
    if sell_intents:
        for intent in sell_intents:
            print(
                f"  EXIT [{intent.city}] {intent.ticker}\n"
                f"    side={intent.side}  contracts={intent.contracts:.1f}"
                f"  fair_p={intent.fair_p:.3f}  exit_price={intent.price:.2f}"
            )
            log_trade(
                market_ticker=intent.ticker,
                city=intent.city,
                market_type=intent.contract_def.get("market_type", ""),
                contract_desc=intent.title,
                fair_p=intent.fair_p,
                yes_ask=0,
                no_ask=0,
                edge_yes=0,
                edge_no=0,
                action="sell" if not dry_run else "recommend_exit",
                side=intent.side,
                size_dollars=intent.size,
                contract_count=intent.contracts,
                status="dry_run" if dry_run else "pending",
            )

        if not dry_run:
            print(f"\nPlacing {len(sell_intents)} exit order(s)...")
            for intent in sell_intents:
                try:
                    # Re-fetch live price immediately before exit to trade at current market
                    live = fetch_live_market_price(auth, intent.ticker)
                    if live is not None:
                        live_yes = live["yes_ask_dollars"]
                        live_no  = live["no_ask_dollars"]
                        live_bid = max(0.0, 1.0 - live_no) if intent.side == "yes" else max(0.0, 1.0 - live_yes)
                        stale_price = intent.price
                        if abs(live_bid - stale_price) >= 0.01:
                            print(f"  {intent.ticker}: exit price updated {stale_price:.4f} → {live_bid:.4f}")
                        # Re-validate: if the position has recovered and entry edge is
                        # still valid at the live price, hold rather than exit.
                        if intent.side == "yes":
                            live_entry_edge = intent.fair_p - live_yes
                        else:
                            live_entry_edge = (1.0 - intent.fair_p) - live_no
                        if live_entry_edge >= pm.min_edge:
                            print(
                                f"  {intent.ticker}: position recovered at live price "
                                f"(live_edge={live_entry_edge:.4f} >= {pm.min_edge:.4f}) — holding"
                            )
                            continue
                        exit_price = live_bid
                    else:
                        exit_price = intent.price
                    resp = kalshi.place_order(
                        ticker=intent.ticker,
                        side=intent.side,
                        count=max(1, round(intent.contracts)),
                        price=int(round(exit_price * 100)),
                        action="sell",
                    )
                    print(f"  Exit order placed: {resp}")
                except Exception as e:
                    print(f"  Exit order failed for {intent.ticker}: {e}")
        else:
            print(f"Dry run: would exit {len(sell_intents)} position(s).")
    else:
        print("  No positions require exit.")
    print()

    # Log and optionally place buy (entry) orders
    print("--- Entry recommendations ---")
    for intent in buy_intents:
        print(
            f"  [{intent.city}] {intent.ticker}\n"
            f"    fair_p={intent.fair_p:.3f}  ask={intent.price:.2f}"
            f"  edge={intent.edge:.3f}  side={intent.side}  contracts={intent.contracts:.2f}"
        )
        log_trade(
            market_ticker=intent.ticker,
            city=intent.city,
            market_type=intent.contract_def.get("market_type", ""),
            contract_desc=intent.title,
            fair_p=intent.fair_p,
            yes_ask=intent.price if intent.side == "yes" else 0,
            no_ask=intent.price if intent.side == "no" else 0,
            edge_yes=intent.edge if intent.side == "yes" else 0,
            edge_no=intent.edge if intent.side == "no" else 0,
            action="recommend" if dry_run else "place",
            side=intent.side,
            size_dollars=intent.size,
            contract_count=intent.contracts,
            status="dry_run" if dry_run else "pending",
        )

    # 8. Place buy orders (live mode only)
    if not dry_run and buy_intents:
        print(f"Placing {len(buy_intents)} orders...")
        for intent in buy_intents:
            try:
                # Re-fetch the live ask price immediately before entry to catch any market moves
                live = fetch_live_market_price(auth, intent.ticker)
                if live is not None:
                    live_ask = live["yes_ask_dollars"] if intent.side == "yes" else live["no_ask_dollars"]
                    stale_ask = intent.price
                    if abs(live_ask - stale_ask) >= 0.01:
                        print(f"  {intent.ticker}: ask moved {stale_ask:.4f} → {live_ask:.4f}")
                    # Re-validate edge at current live price
                    live_edge = intent.fair_p - live_ask if intent.side == "yes" else (1.0 - intent.fair_p) - live_ask
                    if live_edge < pm.min_edge:
                        print(f"  {intent.ticker}: edge gone at live price ({live_edge:.4f} < {pm.min_edge:.4f}) — skipping")
                        continue
                    order_price = live_ask
                else:
                    order_price = intent.price
                # Recompute contract count at the actual order price so that
                # max_bet_dollars is enforced on what we actually spend.
                order_count = max(1, round(intent.size / order_price))
                resp = kalshi.place_order(
                    ticker=intent.ticker,
                    side=intent.side,
                    count=order_count,
                    price=int(round(order_price * 100)),
                )
                print(f"  Order placed: {resp}")
            except Exception as e:
                raw_body = getattr(e, "body", None)
                print(f"  Order failed for {intent.ticker}: {e}")
                if raw_body and raw_body != str(e):
                    print(f"    Raw body: {raw_body}")
    else:
        print(f"\nDry run complete: {len(buy_intents)} entry recommendation(s).")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders (default: dry run)")
    parser.add_argument(
        "--interval", type=int, default=0, metavar="MINUTES",
        help="Re-run every N minutes using the latest model data (0 = run once)",
    )
    parser.add_argument(
        "--model", type=str, default="BaggingRidge", metavar="MODEL",
        help="Model name matching a subdirectory in logs/mega_tune/ (default: BaggingRidge)",
    )
    parser.add_argument(
        "--best-json", type=str, default=None, metavar="PATH",
        help="Direct path to a best.json file (overrides --model path lookup)",
    )
    args = parser.parse_args()

    if args.interval > 0:
        print(f"Loop mode: refreshing every {args.interval} minutes. Ctrl+C to stop.\n")
        while True:
            main(dry_run=not args.live, model_name=args.model, best_json=args.best_json)
            print(f"\nNext run in {args.interval}m — waiting for next model update...\n")
            time.sleep(args.interval * 60)
    else:
        main(dry_run=not args.live, model_name=args.model, best_json=args.best_json)
