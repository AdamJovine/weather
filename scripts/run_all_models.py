"""
scripts/run_all_models.py

Evaluate every model's current best.json against today's live Kalshi markets.
Data and markets are loaded once; each model is fit and evaluated independently.
Prints a per-model summary of predictions and entry recommendations.

Usage:
    python scripts/run_all_models.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for tune_hyperparams

import numpy as np
import pandas as pd
import requests

import src.model as _model_module
from src.config import KALSHI_BASE_URL, TEMP_GRID_MIN, TEMP_GRID_MAX
from src.features import build_feature_table
from src.model import FEATURES
from src.kalshi_client import KalshiWeatherClient
from src.db import DB_PATH, get_db
from src.data_loader_shared import load_raw_sources
from src.portfolio_manager import PortfolioManager
from src.fetchers import fetch_current_obs
from src.ucb import CityUCB

import tune_hyperparams as th

STATIONS_FILE  = Path("data/stations.csv")
BANKROLL       = 5.0
OUT_ROOT       = Path("logs/mega_tune")
KALSHI_SLEEP   = 0.5

MODEL_NAMES = [
    "ARD", "BaggingRidge", "BayesianRidge", "ExtraTrees",
    "GBResidual", "InteractionBayes", "KernelRidge",
    "NGBoost", "QuantileGB", "RandomForest",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def kalshi_get_raw(auth, path: str, params: dict = None) -> dict:
    url = KALSHI_BASE_URL.rstrip("/") + path
    headers = auth.create_auth_headers("GET", path.split("?")[0])
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_weather_markets(auth, stations: pd.DataFrame, target_date: str) -> list[dict]:
    all_markets = []
    date_tag = pd.to_datetime(target_date).strftime("%y%b%d").upper()
    for _, row in stations.iterrows():
        series = row["kalshi_series"]
        try:
            data = kalshi_get_raw(auth, "/markets", {
                "series_ticker": series, "limit": 50, "status": "open",
            })
            mkts = [m for m in data.get("markets", []) if date_tag in m.get("ticker", "")]
            for m in mkts:
                m["_city"] = row["city"]
            all_markets.extend(mkts)
            time.sleep(KALSHI_SLEEP)
        except Exception as e:
            print(f"  {series}: ERROR — {e}")
    return all_markets


def apply_running_max(prob_row: pd.Series, running_max_f: float) -> pd.Series:
    cutoff = math.floor(running_max_f)
    row = prob_row.copy()
    for k in range(TEMP_GRID_MIN, cutoff):
        key = f"temp_{k}"
        if key in row.index:
            row[key] = 0.0
    total = sum(float(row.get(f"temp_{k}", 0.0)) for k in range(cutoff, TEMP_GRID_MAX + 1))
    if total > 0:
        for k in range(cutoff, TEMP_GRID_MAX + 1):
            key = f"temp_{k}"
            if key in row.index:
                row[key] = float(row[key]) / total
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime as _dt
    _now = _dt.now()
    TARGET_DATE = date.today().isoformat() if _now.hour < 16 else (date.today() + timedelta(days=1)).isoformat()

    print(f"=== All-model live evaluation — {TARGET_DATE} ===")
    print(f"Bankroll: ${BANKROLL:.2f}\n")

    stations = pd.read_csv(STATIONS_FILE)

    # ── Load data once ────────────────────────────────────────────────────────
    print("Loading data...")
    sources = load_raw_sources(DB_PATH, nws_target_date=TARGET_DATE, verbose=False)
    hist        = sources["hist"]
    gfs_df      = sources["gfs"]
    gefs_df     = sources["gefs"]
    indices_df  = sources["indices"]
    mjo_df      = sources["mjo"]
    forecast_df = sources["nws"]

    with get_db() as conn:
        _ph = pd.read_sql("SELECT DISTINCT series FROM kalshi_markets", conn)
    tradeable_series = set(_ph["series"].tolist()) if not _ph.empty else None
    tradeable_stations = (
        stations[stations["kalshi_series"].isin(tradeable_series)].copy()
        if tradeable_series else stations.copy()
    )

    df = build_feature_table(hist, forecast_df, gfs_df=gfs_df, indices_df=indices_df,
                             gefs_df=gefs_df, mjo_df=mjo_df)
    df["forecast_high"] = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]
    print(f"Feature table: {len(df)} rows\n")

    # ── Live METAR running-max observations ───────────────────────────────────
    is_same_day = (TARGET_DATE == date.today().isoformat())
    metar_obs = {}
    if is_same_day:
        print("Fetching METAR observations...")
        metar_obs = fetch_current_obs(stations)
        for city, info in sorted(metar_obs.items()):
            print(f"  {city:20s}  running_max={info['running_max_f']:.1f}°F")
        print()

    # ── UCB city multipliers (shared) ─────────────────────────────────────────
    city_ucb = CityUCB(cities=tradeable_stations["city"].tolist())
    city_ucb.load_state()
    bt_path = Path("logs/pnl_backtest_trades.csv")
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        if {"city", "pnl", "size"}.issubset(bt.columns):
            city_ucb.initialize_from_backtest(bt)
    city_multipliers = {c: city_ucb.kelly_multiplier(c) for c in tradeable_stations["city"]}

    # ── Kalshi: fetch markets and positions once ───────────────────────────────
    print("Connecting to Kalshi...")
    try:
        kalshi = KalshiWeatherClient.from_env()
        auth   = kalshi._client.kalshi_auth
    except Exception as e:
        print(f"ERROR connecting to Kalshi: {e}")
        sys.exit(1)

    print(f"Fetching markets for {TARGET_DATE}...")
    weather_markets = fetch_weather_markets(auth, tradeable_stations, TARGET_DATE)
    print(f"  {len(weather_markets)} markets\n")

    try:
        raw_positions = kalshi.get_positions()
        positions = {p["ticker"]: p.get("position", 0) for p in raw_positions}
    except Exception as e:
        print(f"Warning: could not fetch positions ({e})")
        positions = {}

    # ── Per-model evaluation ──────────────────────────────────────────────────
    results = []

    for model_name in MODEL_NAMES:
        best_path = OUT_ROOT / model_name / "best.json"
        if not best_path.exists():
            print(f"[{model_name}] no best.json — skipping")
            continue

        params = json.load(open(best_path))["params"]

        print(f"{'─'*60}")
        print(f"[{model_name}]  lookback={int(params['lookback'])}  "
              f"min_edge={params['min_edge']:.4f}  "
              f"min_conf={params['min_confidence']:.4f}")

        # Set spread_alpha globally for this model
        _model_module.SPREAD_ALPHA = float(params["spread_alpha"])

        # Fit per city
        models = {}
        for city in stations["city"]:
            city_df  = df[df["city"] == city].copy()
            train_df = city_df[city_df["date"] < pd.to_datetime(TARGET_DATE)]
            try:
                model = th.make_model(model_name, params)
                model._lookback = int(params["lookback"])
                model.fit(train_df)
                models[city] = model
            except Exception as e:
                print(f"  {city}: fit failed — {e}")

        if not models:
            print(f"  No cities fitted — skipping")
            continue

        # Build predictions + apply running max
        pred_rows = {}
        print("  Predictions (tradeable cities):")
        for city, model in models.items():
            city_df = df[(df["city"] == city) & (df["date"] == pd.to_datetime(TARGET_DATE))]
            if city_df.empty or city_df[FEATURES].isnull().any(axis=1).iloc[0]:
                continue
            try:
                mu_arr, *_ = model.predict_with_uncertainty(city_df)
                probs_df   = model.predict_integer_probs(city_df)
                prob_row   = probs_df.iloc[0]
                if is_same_day and city in metar_obs:
                    prob_row = apply_running_max(prob_row, metar_obs[city]["running_max_f"])
                pred_rows[city] = prob_row
                if city in tradeable_stations["city"].values:
                    nws = city_df.iloc[0].get("forecast_high", float("nan"))
                    print(f"    {city:20s}  pred={float(mu_arr[0]):5.1f}°F  NWS={nws:.0f}°F")
            except Exception as e:
                print(f"  {city}: predict failed — {e}")

        # Evaluate via PortfolioManager
        pm = PortfolioManager(
            kelly_fraction   = float(params["kelly_fraction"]),
            max_bet_fraction = float(params["max_bet_frac"]),
            min_edge         = float(params["min_edge"]),
            min_confidence   = float(params["min_confidence"]),
            min_fair_p       = float(params.get("min_fair_p", 0.05)),
            max_fair_p       = float(params.get("max_fair_p", 0.95)),
        )
        intents = pm.evaluate(weather_markets, pred_rows, positions, BANKROLL, city_multipliers)
        buys    = [i for i in intents if i.action == "buy"]

        print(f"  → {len(buys)} entry recommendation(s)")
        for i in buys:
            print(f"     {i.ticker}  side={i.side}  fair_p={i.fair_p:.3f}  "
                  f"ask={i.price:.2f}  edge={i.edge:.3f}  "
                  f"contracts={i.contracts:.1f}  size=${i.size:.2f}")

        results.append({
            "model":   model_name,
            "n_buys":  len(buys),
            "intents": buys,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — entry recommendations by model:")
    results.sort(key=lambda r: -r["n_buys"])
    for r in results:
        flag = "  <<<" if r["n_buys"] > 0 else ""
        print(f"  {r['model']:20s}  {r['n_buys']} recommendation(s){flag}")


if __name__ == "__main__":
    main()
