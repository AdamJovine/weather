"""
Simple live weather mispricing bot.

Purpose:
  - Trade a simple calibrated forecast-vs-price edge model.
  - Keep risk controls strict for real money:
      * $1 target notional per order
      * never exceed $10 spent on a single market ticker

Default mode is dry-run. Use --live to place real orders.
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import subprocess
import requests
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import KALSHI_BASE_URL
from src.fetchers import fetch_current_obs
from src.kalshi_client import KalshiWeatherClient
from src.logger import init_run, log_trade, TRADE_LOG_PATH
from src.run_tracker import start_run, end_run
from src.strategy import parse_contract

from scripts.backtest_simple_mispricing_bot import (
    _apply_probability_safety,
    _build_daily_distribution,
    _compute_yes_prob,
    _compute_yes_prob_empirical,
)


STATIONS_FILE = Path("data/stations.csv")
HARD_STAKE_PER_TRADE = 5.0
HARD_MAX_PER_MARKET = 50.0

# Kalshi occasionally renames city series tickers. Keep a small alias map so
# live fetches still find markets even if stations.csv has an older series id.
SERIES_ALIASES: dict[str, list[str]] = {
    "KXHIGHNO": ["KXHIGHTNOLA"],  # New Orleans
}

DATA_REFRESH_SOURCE_CHOICES = ["noaa", "obs", "indices", "gfs", "gefs", "nws", "kalshi"]
DEFAULT_DATA_REFRESH_SOURCES = ["noaa", "obs", "gfs", "gefs", "nws"]


def _resolve_target_date(args: argparse.Namespace) -> str:
    """
    Resolve the target settlement date for this cycle.

    Priority:
      1) --target-date (explicit)
      2) --trade-today
      3) auto cutoff behavior: before cutoff hour use today, else tomorrow
    """
    if args.target_date:
        return str(args.target_date)
    if bool(args.trade_today):
        return date.today().isoformat()

    cutoff_hour = int(args.target_date_cutoff_hour)
    now_local = datetime.now()
    if now_local.hour < cutoff_hour:
        return date.today().isoformat()
    return (date.today() + timedelta(days=1)).isoformat()


def _resolve_target_dates(args: argparse.Namespace) -> list[str]:
    """
    Resolve one or more settlement dates to trade this cycle.

    Default behavior keeps the existing single-date selection from
    _resolve_target_date(). If --include-tomorrow is enabled, also include
    tomorrow's date relative to local time.
    """
    base_target = _resolve_target_date(args)
    target_dates = [str(base_target)]
    if bool(getattr(args, "include_tomorrow", False)):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        if tomorrow not in target_dates:
            target_dates.append(tomorrow)
    return target_dates


def _price_to_dollars(v: float | int | str | None) -> float:
    """
    Normalize Kalshi price fields to dollar units [0, 1].

    Some payloads use dollars (0.63), others cents (63).
    """
    if v is None:
        return 0.0
    p = float(v)
    return p if p <= 1.0 else p / 100.0


def _to_float(v: float | int | str | None, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_nws_forecast_map(db_path: str, target_date: str) -> dict[str, float]:
    """
    Return city -> blended NWS/NBM forecast high for target_date.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT city, forecast_high, nbm_high
                FROM nws_forecasts
                WHERE target_date = ?
                """,
                conn,
                params=[str(target_date)],
            )
    except Exception:
        return {}
    if df.empty:
        return {}

    out: dict[str, float] = {}
    for _, r in df.iterrows():
        city = str(r.get("city", "") or "")
        if not city:
            continue
        vals: list[float] = []
        for col in ("forecast_high", "nbm_high"):
            v = r.get(col)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if vals:
            out[city] = float(sum(vals) / len(vals))
    return out


def _normal_event_prob(contract: dict, mu: float, sigma: float, running_max_f: float | None = None) -> float:
    """
    Contract YES probability under Normal(mu, sigma), optionally conditioned on
    final daily high being >= running_max_f.
    """
    mu = float(mu)
    sigma = max(float(sigma), 0.5)

    mtype = str(contract.get("market_type", "") or "")
    thr = float(contract.get("threshold", float("nan")))
    low = float(contract.get("low", float("nan")))
    high = float(contract.get("high", float("nan")))

    cdf = lambda x: float(norm.cdf(float(x), loc=mu, scale=sigma))

    def _uncond() -> float:
        if mtype in {"gt", "geq"}:
            return 1.0 - cdf(thr)
        if mtype in {"lt", "leq"}:
            return cdf(thr)
        if mtype == "range":
            return cdf(high) - cdf(low)
        return float("nan")

    if running_max_f is None or not math.isfinite(float(running_max_f)):
        p = _uncond()
        return max(0.0, min(1.0, float(p if math.isfinite(p) else float("nan"))))

    lb = float(running_max_f)
    p_lb = 1.0 - cdf(lb)
    if p_lb <= 1e-9:
        return 0.0

    if mtype in {"gt", "geq"}:
        numer = 1.0 - cdf(max(thr, lb))
    elif mtype in {"lt", "leq"}:
        numer = cdf(thr) - cdf(lb)
    elif mtype == "range":
        lo = max(low, lb)
        hi = high
        numer = (cdf(hi) - cdf(lo)) if hi >= lo else 0.0
    else:
        return float("nan")

    p = numer / p_lb
    return max(0.0, min(1.0, float(p)))


def _empirical_event_prob(
    contract: dict,
    forecast_blend: float,
    errors: np.ndarray,
    running_max_f: float | None = None,
) -> float:
    """
    Contract YES probability using the empirical error distribution,
    optionally conditioned on final daily high >= running_max_f.
    """
    simulated = float(forecast_blend) + errors

    if running_max_f is not None and math.isfinite(float(running_max_f)):
        simulated = simulated[simulated >= float(running_max_f)]
        if len(simulated) == 0:
            return float("nan")

    mtype = str(contract.get("market_type", "") or "")
    thr = float(contract.get("threshold", float("nan")))
    low = float(contract.get("low", float("nan")))
    high = float(contract.get("high", float("nan")))

    if mtype in ("gt", "geq"):
        p = float(np.mean(simulated > thr))
    elif mtype in ("lt", "leq"):
        p = float(np.mean(simulated < thr))
    elif mtype == "range":
        p = float(np.mean((simulated >= low) & (simulated <= high)))
    else:
        return float("nan")

    return max(0.0, min(1.0, p))


def _build_same_day_city_context(
    stations: pd.DataFrame,
    target_date: str,
    db_path: str,
    dist_daily: pd.DataFrame,
    sigma_floor: float,
) -> dict[str, dict]:
    """
    Build same-day fair-value inputs per city:
      - mu from NWS/NBM forecast high (fallback: dist_daily mu)
      - sigma from dist_daily (fallback: sigma_floor)
      - running_max_f from live METAR observations
    """
    out: dict[str, dict] = {}
    nws_mu = _load_nws_forecast_map(db_path=db_path, target_date=target_date)
    obs = fetch_current_obs(stations)

    td = pd.to_datetime(target_date)
    for _, srow in stations.iterrows():
        city = str(srow.get("city", "") or "")
        if not city:
            continue

        d = dist_daily[(dist_daily["city"] == city) & (dist_daily["date"] == td)]
        if d.empty:
            continue
        drow = d.iloc[0]

        mu_base = _to_float(drow.get("mu"), float("nan"))
        sigma = _to_float(drow.get("sigma"), float("nan"))
        n_hist = _to_float(drow.get("n_hist"), 0.0)
        if not math.isfinite(mu_base):
            continue
        if not math.isfinite(sigma):
            sigma = float(sigma_floor)
        sigma = max(float(sigma), float(sigma_floor))

        mu_nws = nws_mu.get(city)
        mu = float(mu_nws) if mu_nws is not None else float(mu_base)

        city_obs = obs.get(city, {}) if isinstance(obs, dict) else {}
        running_max_f = city_obs.get("running_max_f")
        if running_max_f is not None and math.isfinite(float(running_max_f)):
            # The final max cannot be below the observed intraday running max.
            mu = max(float(mu), float(running_max_f))
            running_max_f = float(running_max_f)
        else:
            running_max_f = None

        out[city] = {
            "mu": float(mu),
            "sigma": float(sigma),
            "n_hist": float(n_hist),
            "mu_nws": float(mu_nws) if mu_nws is not None else None,
            "mu_base": float(mu_base),
            "running_max_f": running_max_f,
            "obs_count": int(_to_float(city_obs.get("obs_count"), 0.0)),
            "latest_obs_utc": str(city_obs.get("latest_obs_utc", "") or ""),
        }
    return out


def _load_ticker_cost_from_logs() -> dict[str, float]:
    """
    Sum dollars spent per ticker from recent run logs.

    Uses today's and yesterday's trade logs to enforce per-ticker spend caps.
    """
    import glob

    ticker_cost: dict[str, float] = {}
    today = date.today().strftime("%Y%m%d")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    for day in [today, yesterday]:
        for path in glob.glob(f"logs/trade_log_{day}*.csv"):
            try:
                df = pd.read_csv(path)
                placed = df[df["action"] == "place"]
                for _, row in placed.iterrows():
                    ticker = str(row.get("market_ticker", "") or "")
                    size = float(row.get("size_dollars", 0) or 0)
                    if ticker and size > 0:
                        ticker_cost[ticker] = ticker_cost.get(ticker, 0.0) + size
            except Exception:
                pass
    return ticker_cost


def _load_ticker_cost_from_fills(kalshi: KalshiWeatherClient, lookback_days: int = 3) -> dict[str, float]:
    """
    Approximate spent dollars per ticker from recent executed fills.

    Conservative accounting:
      - Buy fills increase used spend.
      - Sell fills reduce used spend, floored at zero.
    """
    ticker_cost: dict[str, float] = {}
    min_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    try:
        fills = kalshi.get_fills(limit=1000, min_ts=min_ts)
    except Exception as e:
        print(f"Warning: could not fetch fills for cap accounting ({e})")
        return ticker_cost

    for fill in fills:
        ticker = str(fill.get("ticker", "") or "")
        if not ticker:
            continue
        side = str(fill.get("side", "") or "").lower()
        action = str(fill.get("action", "buy") or "buy").lower()
        count = _to_float(fill.get("count_fp"), _to_float(fill.get("count"), 0.0))
        if count <= 0:
            continue

        yes_price = _price_to_dollars(
            fill.get("yes_price_dollars", fill.get("yes_price"))
        )
        no_price = _price_to_dollars(
            fill.get("no_price_dollars", fill.get("no_price"))
        )
        if side == "yes":
            px = yes_price if yes_price > 0 else max(0.0, 1.0 - no_price)
        elif side == "no":
            px = no_price if no_price > 0 else max(0.0, 1.0 - yes_price)
        else:
            continue

        notional = float(count) * float(px)
        if action == "sell":
            ticker_cost[ticker] = max(0.0, ticker_cost.get(ticker, 0.0) - notional)
        else:
            ticker_cost[ticker] = ticker_cost.get(ticker, 0.0) + notional

    return ticker_cost


def _load_ticker_cost_from_positions(kalshi: KalshiWeatherClient) -> dict[str, float]:
    """
    Load current per-ticker exposure from open positions.

    Uses market_exposure_dollars (fallback: market_exposure) and keeps only
    positive exposure values.
    """
    ticker_cost: dict[str, float] = {}
    try:
        positions = kalshi.get_positions(limit=1000)
    except Exception as e:
        print(f"Warning: could not fetch positions for cap accounting ({e})")
        return ticker_cost

    for p in positions:
        ticker = str(p.get("ticker", "") or "")
        if not ticker:
            continue
        exposure = _to_float(
            p.get("market_exposure_dollars"),
            _to_float(p.get("market_exposure"), 0.0),
        )
        if exposure > 0:
            ticker_cost[ticker] = max(float(ticker_cost.get(ticker, 0.0)), float(exposure))
    return ticker_cost


def _kalshi_get_raw(auth, path: str, params: dict | None = None) -> dict:
    url = KALSHI_BASE_URL.rstrip("/") + path
    headers = auth.create_auth_headers("GET", path.split("?")[0])
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_live_market_snapshot(auth, ticker: str) -> dict[str, float] | None:
    """
    Fetch latest top-of-book quote snapshot for a single ticker.
    """
    try:
        data = _kalshi_get_raw(auth, f"/markets/{ticker}")
        market = data.get("market", data)
        yes_ask = market.get("yes_ask_dollars")
        no_ask = market.get("no_ask_dollars")
        if yes_ask is None or no_ask is None:
            return None
        return {
            "yes_bid_dollars": _to_float(market.get("yes_bid_dollars"), 0.0),
            "yes_bid_size_fp": _to_float(market.get("yes_bid_size_fp"), 0.0),
            "yes_ask_dollars": _to_float(yes_ask, 0.0),
            "yes_ask_size_fp": _to_float(market.get("yes_ask_size_fp"), 0.0),
            "no_bid_dollars": _to_float(market.get("no_bid_dollars"), 0.0),
            "no_ask_dollars": _to_float(no_ask, 0.0),
            "no_ask_size_fp": _to_float(market.get("no_ask_size_fp"), 0.0),
        }
    except Exception:
        return None


def _extract_fill_cost_and_fees(order_resp: dict) -> tuple[float | None, float | None]:
    """
    Extract executed fill notional and fees from create_order response.
    """
    order = order_resp.get("order", {}) if isinstance(order_resp, dict) else {}
    if not isinstance(order, dict):
        return None, None

    taker_fill = _price_to_dollars(order.get("taker_fill_cost_dollars"))
    maker_fill = _price_to_dollars(order.get("maker_fill_cost_dollars"))
    taker_fees = _price_to_dollars(order.get("taker_fees_dollars"))
    maker_fees = _price_to_dollars(order.get("maker_fees_dollars"))

    fill_total = taker_fill + maker_fill
    fee_total = taker_fees + maker_fees
    if fill_total <= 0:
        fill_total = None
    if fee_total <= 0:
        fee_total = None
    return fill_total, fee_total


def _extract_fill_count(order_resp: dict) -> float:
    order = order_resp.get("order", {}) if isinstance(order_resp, dict) else {}
    if not isinstance(order, dict):
        return 0.0
    if order.get("fill_count_fp") is not None:
        return _to_float(order.get("fill_count_fp"), 0.0)
    if order.get("fill_count") is not None:
        return _to_float(order.get("fill_count"), 0.0)
    if order.get("initial_count_fp") is not None and str(order.get("status", "")).lower() == "executed":
        return _to_float(order.get("initial_count_fp"), 0.0)
    return 0.0


def _extract_http_status(exc: Exception) -> int | None:
    for attr in ("status", "status_code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return int(v)
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    m = re.search(r"\((\d{3})\)", str(exc))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _refresh_data_sources(sources: list[str]) -> None:
    """
    Refresh selected DB sources by running scripts/update_data.py.
    """
    selected = [s for s in sources if s]
    if not selected:
        print("Data refresh skipped: no sources selected.")
        return

    cmd = [sys.executable, "scripts/update_data.py", "--only", *selected]
    print(f"Refreshing data sources: {', '.join(selected)}")
    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"Warning: data refresh failed to start ({e})")
        return

    if proc.returncode != 0:
        print(f"Warning: data refresh exited non-zero ({proc.returncode}).")
    else:
        print("Data refresh complete.")


def _partial_retry_counts(initial_count: int, max_steps: int) -> list[int]:
    """
    Build a descending list of smaller sizes to retry after a 409 conflict.
    """
    n = max(1, int(initial_count))
    steps = max(0, int(max_steps))
    if n <= 1 or steps <= 0:
        return []

    out: list[int] = []
    for frac in (0.75, 0.5, 0.33, 0.2, 0.1):
        c = int(math.floor(float(n) * float(frac)))
        if 1 <= c < n and c not in out:
            out.append(c)
        if len(out) >= steps:
            return out[:steps]

    if 1 not in out:
        out.append(1)
    return out[:steps]


def _top_ask_size_for_side(snapshot: dict[str, float], side: str) -> float | None:
    """
    Best-effort top-of-book displayed size for a buy on `side`.
    """
    if side == "yes":
        s = _to_float(snapshot.get("yes_ask_size_fp"), 0.0)
        return s if s > 0 else None
    if side == "no":
        # no_ask_size_fp is not always present; fall back to yes_bid_size if
        # no-ask appears implied by the best yes bid.
        s_no = _to_float(snapshot.get("no_ask_size_fp"), 0.0)
        if s_no > 0:
            return s_no
        no_ask = _to_float(snapshot.get("no_ask_dollars"), 0.0)
        yes_bid = _to_float(snapshot.get("yes_bid_dollars"), 0.0)
        if 0.0 < no_ask < 1.0 and 0.0 < yes_bid < 1.0 and abs((1.0 - yes_bid) - no_ask) <= 0.011:
            s_yes_bid = _to_float(snapshot.get("yes_bid_size_fp"), 0.0)
            return s_yes_bid if s_yes_bid > 0 else None
    return None


def _fetch_weather_markets(auth, stations: pd.DataFrame, target_date: str) -> list[dict]:
    """
    Fetch open weather markets matching target_date from each city series.
    """
    out: list[dict] = []
    date_tag = pd.to_datetime(target_date).strftime("%y%b%d").upper()
    for _, row in stations.iterrows():
        series = row["kalshi_series"]
        city = row["city"]

        # Try the configured series first, then known aliases if needed.
        candidates = [series] + SERIES_ALIASES.get(str(series), [])
        seen_tickers: set[str] = set()
        found_for_city = False
        for series_candidate in candidates:
            try:
                data = _kalshi_get_raw(
                    auth,
                    "/markets",
                    {"series_ticker": series_candidate, "limit": 100, "status": "open"},
                )
                mkts = data.get("markets", [])
                mkts = [m for m in mkts if date_tag in str(m.get("ticker", ""))]
                if not mkts:
                    continue
                found_for_city = True
                for m in mkts:
                    ticker = str(m.get("ticker", ""))
                    if ticker in seen_tickers:
                        continue
                    seen_tickers.add(ticker)
                    m["_city"] = city
                    out.append(m)
            except Exception as e:
                print(f"  {series_candidate}: fetch error: {e}")
        time.sleep(0.08)
    return out


def _fair_prob_for_market(
    market: dict,
    city: str,
    target_date: str,
    dist_daily: pd.DataFrame,
    prob_floor: float,
    prob_ceil: float,
    confidence_shrink_k: float,
    same_day_city_ctx: dict[str, dict] | None = None,
    errors_dict: dict | None = None,
) -> tuple[float, float] | None:
    """
    Return (p_yes_used, n_hist) for a market, or None if unavailable.
    """
    title = str(market.get("title", ""))
    try:
        contract = parse_contract(title, "")
    except Exception:
        return None

    d = dist_daily[(dist_daily["city"] == city) & (dist_daily["date"] == pd.to_datetime(target_date))]
    if d.empty:
        return None
    drow = d.iloc[0]

    mu = float(drow["mu"])
    sigma = float(drow["sigma"])
    n_hist = float(drow["n_hist"])

    same_day_ctx = (same_day_city_ctx or {}).get(city)
    if same_day_ctx:
        mu = float(same_day_ctx.get("mu", mu))
        sigma = float(same_day_ctx.get("sigma", sigma))
        n_hist = float(same_day_ctx.get("n_hist", n_hist))
        running_max_f = same_day_ctx.get("running_max_f")

        # Use empirical CDF for same-day trades when available.
        date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d")
        ekey = (city, date_str)
        if errors_dict is not None and ekey in errors_dict:
            p_yes_model = _empirical_event_prob(
                contract=contract,
                forecast_blend=float(drow["forecast_blend"]),
                errors=errors_dict[ekey],
                running_max_f=running_max_f,
            )
        else:
            p_yes_model = _normal_event_prob(
                contract=contract,
                mu=mu,
                sigma=sigma,
                running_max_f=running_max_f,
            )
    else:
        row_tmp = pd.DataFrame(
            [
                {
                    "market_type": contract.get("market_type"),
                    "threshold": float(contract.get("threshold", float("nan"))),
                    "low": float(contract.get("low", float("nan"))),
                    "high": float(contract.get("high", float("nan"))),
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "n_hist": float(n_hist),
                    "forecast_blend": float(drow["forecast_blend"]),
                    "city": city,
                    "date": pd.to_datetime(target_date),
                    "p_yes_model": 0.5,  # placeholder before vectorized call
                }
            ]
        )
        if errors_dict is not None:
            row_tmp["p_yes_model"] = _compute_yes_prob_empirical(row_tmp, errors_dict)
        else:
            row_tmp["p_yes_model"] = _compute_yes_prob(row_tmp)
        p_yes_model = float(row_tmp["p_yes_model"].iloc[0])

    row = pd.DataFrame(
        [
            {
                "market_type": contract.get("market_type"),
                "threshold": float(contract.get("threshold", float("nan"))),
                "low": float(contract.get("low", float("nan"))),
                "high": float(contract.get("high", float("nan"))),
                "mu": float(mu),
                "sigma": float(sigma),
                "n_hist": float(n_hist),
                "p_yes_model": float(p_yes_model),
            }
        ]
    )
    row = _apply_probability_safety(
        row,
        prob_floor=prob_floor,
        prob_ceil=prob_ceil,
        confidence_shrink_k=confidence_shrink_k,
    )
    return float(row["p_yes_used"].iloc[0]), float(row["n_hist"].iloc[0])


def run_once(args: argparse.Namespace, dry_run: bool) -> None:
    target_dates = _resolve_target_dates(args)
    if len(target_dates) == 1:
        target_label = target_dates[0]
        print(f"\n=== Simple Mispricing Bot | target={target_label} | mode={'DRY' if dry_run else 'LIVE'} ===")
    else:
        target_label = ",".join(target_dates)
        print(f"\n=== Simple Mispricing Bot | targets={target_label} | mode={'DRY' if dry_run else 'LIVE'} ===")

    if not STATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing stations file: {STATIONS_FILE}")
    stations = pd.read_csv(STATIONS_FILE)
    if args.cities:
        city_set = {c.strip() for c in args.cities.split(",") if c.strip()}
        stations = stations[stations["city"].isin(city_set)].copy()
    if args.exclude_cities:
        exclude_set = {c.strip() for c in args.exclude_cities.split(",") if c.strip()}
        stations = stations[~stations["city"].isin(exclude_set)].copy()
    if stations.empty:
        print("No stations selected.")
        return

    build_result = _build_daily_distribution(
        db_path=args.db,
        lookback_days=args.lookback_days,
        min_hist_days=args.min_hist_days,
        sigma_floor=args.sigma_floor,
        spread_alpha=args.spread_alpha,
        ecmwf_weight=args.ecmwf_weight,
        nbm_weight=float(args.nbm_weight),
        disagree_alpha=float(args.disagree_alpha),
        t850_sigma_alpha=float(args.t850_sigma_alpha),
        return_errors=bool(args.use_empirical),
    )
    if bool(args.use_empirical):
        dist_daily, errors_dict = build_result
    else:
        dist_daily = build_result
        errors_dict = None
    same_day_city_ctx_by_date: dict[str, dict[str, dict] | None] = {}
    for td in target_dates:
        same_day_city_ctx: dict[str, dict] | None = None
        if str(td) == date.today().isoformat():
            same_day_city_ctx = _build_same_day_city_context(
                stations=stations,
                target_date=td,
                db_path=args.db,
                dist_daily=dist_daily,
                sigma_floor=float(args.sigma_floor),
            )
            if same_day_city_ctx:
                print(
                    "Same-day fair source: NWS/NBM + METAR high-so-far "
                    f"for {len(same_day_city_ctx)} city/cities."
                )
        same_day_city_ctx_by_date[td] = same_day_city_ctx

    kalshi = KalshiWeatherClient.from_env()
    auth = kalshi._client.kalshi_auth
    markets: list[dict] = []
    markets_per_date: dict[str, int] = {}
    for td in target_dates:
        mkts = _fetch_weather_markets(auth, stations, td)
        markets_per_date[td] = len(mkts)
        for m in mkts:
            m["_target_date"] = td
        markets.extend(mkts)

    # In auto mode with a single target, if no open markets, try the adjacent day once.
    if (
        not markets
        and len(target_dates) == 1
        and not args.target_date
        and not args.trade_today
        and not bool(args.include_tomorrow)
    ):
        target_date = target_dates[0]
        fallback_date = (
            (date.today() + timedelta(days=1)).isoformat()
            if target_date == date.today().isoformat()
            else date.today().isoformat()
        )
        print(f"No open markets for target={target_date}; trying fallback target={fallback_date} ...")
        fallback_markets = _fetch_weather_markets(auth, stations, fallback_date)
        if fallback_markets:
            markets = []
            markets_per_date = {fallback_date: len(fallback_markets)}
            for m in fallback_markets:
                m["_target_date"] = fallback_date
            markets.extend(fallback_markets)
            target_dates = [fallback_date]
            print(f"Fallback target selected: {fallback_date}")

    if len(markets_per_date) <= 1:
        print(f"Markets fetched: {len(markets)}")
    else:
        breakdown = ", ".join(f"{td}:{n}" for td, n in markets_per_date.items())
        print(f"Markets fetched: {len(markets)} total ({breakdown})")
    if not markets:
        return

    # Hard risk controls for live safety — CLI can go lower, never higher.
    stake_per_trade = min(float(args.stake_per_trade), float(HARD_STAKE_PER_TRADE))
    max_per_market = min(float(args.max_per_market), float(HARD_MAX_PER_MARKET))
    print(f"Risk controls: stake_per_trade=${stake_per_trade:.2f}, max_per_market=${max_per_market:.2f}")
    if float(args.stake_per_trade) > float(HARD_STAKE_PER_TRADE):
        print(f"  (stake capped at hard limit ${HARD_STAKE_PER_TRADE:.2f})")
    if float(args.max_per_market) > float(HARD_MAX_PER_MARKET):
        print(f"  (max-per-market capped at hard limit ${HARD_MAX_PER_MARKET:.2f})")

    # Exposure accounting for per-ticker cap enforcement.
    # Priority: live open-position exposure, then fill-derived spend, then local logs.
    ticker_cost = _load_ticker_cost_from_positions(kalshi)
    ticker_cost_fills = _load_ticker_cost_from_fills(kalshi, lookback_days=3)
    for ticker, cost in ticker_cost_fills.items():
        ticker_cost[ticker] = max(float(cost), float(ticker_cost.get(ticker, 0.0)))
    ticker_cost_logs = _load_ticker_cost_from_logs()
    for ticker, cost in ticker_cost_logs.items():
        ticker_cost[ticker] = max(float(cost), float(ticker_cost.get(ticker, 0.0)))

    recommendations = []

    for m in markets:
        market_target_date = str(m.get("_target_date", target_dates[0]))
        ticker = str(m.get("ticker", ""))
        city = str(m.get("_city", ""))
        yes_ask = m.get("yes_ask_dollars")
        no_ask = m.get("no_ask_dollars")
        if yes_ask is None or no_ask is None:
            continue
        yes_ask = float(yes_ask)
        no_ask = float(no_ask)
        if not (0.0 < yes_ask < 1.0 and 0.0 < no_ask < 1.0):
            continue

        fair = _fair_prob_for_market(
            market=m,
            city=city,
            target_date=market_target_date,
            dist_daily=dist_daily,
            prob_floor=args.prob_floor,
            prob_ceil=args.prob_ceil,
            confidence_shrink_k=args.confidence_shrink_k,
            same_day_city_ctx=same_day_city_ctx_by_date.get(market_target_date),
            errors_dict=errors_dict,
        )
        if fair is None:
            continue
        p_yes_used, n_hist = fair

        edge_yes = p_yes_used - yes_ask
        edge_no = (1.0 - p_yes_used) - no_ask
        p_std = ((p_yes_used * (1.0 - p_yes_used)) / max(n_hist, 1.0)) ** 0.5
        effective_min_edge = float(args.min_edge) + float(args.uncertainty_z) * p_std

        if args.side_mode == "yes":
            side = "yes"
            edge = edge_yes
            ask = yes_ask
        elif args.side_mode == "no":
            side = "no"
            edge = edge_no
            ask = no_ask
        else:
            if edge_yes >= edge_no:
                side, edge, ask = "yes", edge_yes, yes_ask
            else:
                side, edge, ask = "no", edge_no, no_ask

        if edge < effective_min_edge:
            continue
        if float(args.max_edge) > 0 and edge > float(args.max_edge):
            continue
        if float(args.skip_edge_lo) > 0 and float(args.skip_edge_hi) > 0:
            if edge > float(args.skip_edge_lo) and edge < float(args.skip_edge_hi):
                continue

        # Use floor so expected spend is never above $1 target.
        contracts = max(1, int(stake_per_trade / ask))
        while contracts > 1 and (contracts * ask) > (stake_per_trade + 1e-9):
            contracts -= 1
        est_spend = contracts * ask
        used = float(ticker_cost.get(ticker, 0.0))
        if used + est_spend > max_per_market + 1e-9:
            continue

        recommendations.append(
            {
                "ticker": ticker,
                "city": city,
                "title": str(m.get("title", "")),
                "target_date": market_target_date,
                "side": side,
                "ask": ask,
                "yes_ask_signal": yes_ask,
                "no_ask_signal": no_ask,
                "yes_bid_signal": _to_float(m.get("yes_bid_dollars"), 0.0),
                "no_bid_signal": _to_float(m.get("no_bid_dollars"), 0.0),
                "contracts": contracts,
                "est_spend": est_spend,
                "used": used,
                "stake_target": stake_per_trade,
                "fair_p_yes": p_yes_used,
                "edge_yes": edge_yes,
                "edge_no": edge_no,
                "edge_chosen": edge,
                "effective_min_edge": effective_min_edge,
            }
        )

    recommendations = sorted(recommendations, key=lambda x: x["edge_chosen"], reverse=True)
    print(f"Signals passing threshold: {len(recommendations)}")
    for r in recommendations:
        print(
            f"  {r['ticker']} [{r['city']} {r['target_date']}] side={r['side']} ask={r['ask']:.3f} "
            f"edge={r['edge_chosen']:.3f} min={r['effective_min_edge']:.3f} "
            f"spend~${r['est_spend']:.2f} (target=${r['stake_target']:.2f}) used=${r['used']:.2f}"
        )

    execution_candidates = recommendations
    max_orders_per_cycle = int(args.max_orders_per_cycle)
    if max_orders_per_cycle > 0 and len(execution_candidates) > max_orders_per_cycle:
        skipped = len(execution_candidates) - max_orders_per_cycle
        execution_candidates = execution_candidates[:max_orders_per_cycle]
        print(
            f"Execution throttle: top {len(execution_candidates)} signal(s) this cycle, "
            f"skipping {skipped} lower-edge signal(s)."
        )

    for r in execution_candidates:
        # Log recommendation before attempting order.
        log_trade(
            market_ticker=r["ticker"],
            city=r["city"],
            market_type="simple_mispricing",
            contract_desc=r["title"],
            fair_p=r["fair_p_yes"],
            yes_ask=r["ask"] if r["side"] == "yes" else 0,
            no_ask=r["ask"] if r["side"] == "no" else 0,
            edge_yes=r["edge_yes"],
            edge_no=r["edge_no"],
            action="recommend",
            side=r["side"],
            size_dollars=r["est_spend"],
            contract_count=r["contracts"],
            status="dry_run" if dry_run else "pending",
        )

        if dry_run:
            continue

        try:
            # Re-fetch top-of-book immediately before placement.
            live = _fetch_live_market_snapshot(auth, r["ticker"])
            if live is None:
                print(f"  Skip {r['ticker']}: could not fetch fresh market snapshot.")
                continue

            if bool(args.require_book_stable):
                max_drift = float(args.max_book_drift)
                moved_fields: list[str] = []
                if r["side"] == "yes":
                    signal_ask = float(r["yes_ask_signal"])
                    live_ask_now = float(live["yes_ask_dollars"])
                    ask_label = "yes_ask"
                    signal_bid = float(r.get("yes_bid_signal", 0.0) or 0.0)
                    live_bid = float(live.get("yes_bid_dollars", 0.0) or 0.0)
                    bid_label = "yes_bid"
                else:
                    signal_ask = float(r["no_ask_signal"])
                    live_ask_now = float(live["no_ask_dollars"])
                    ask_label = "no_ask"
                    signal_bid = float(r.get("no_bid_signal", 0.0) or 0.0)
                    live_bid = float(live.get("no_bid_dollars", 0.0) or 0.0)
                    bid_label = "no_bid"

                if abs(live_ask_now - signal_ask) > max_drift + 1e-12:
                    moved_fields.append(f"{ask_label}:{signal_ask:.2f}->{live_ask_now:.2f}")
                if signal_bid > 0 and live_bid > 0 and abs(live_bid - signal_bid) > max_drift + 1e-12:
                    moved_fields.append(f"{bid_label}:{signal_bid:.2f}->{live_bid:.2f}")
                if moved_fields:
                    print(f"  Skip {r['ticker']}: book moved ({'; '.join(moved_fields)}).")
                    continue

            live_ask = float(live["yes_ask_dollars"]) if r["side"] == "yes" else float(live["no_ask_dollars"])
            if not (0.0 < live_ask < 1.0):
                print(f"  Skip {r['ticker']}: invalid live ask for side={r['side']} ({live_ask}).")
                continue

            edge_buffer = max(0.0, float(r["edge_chosen"]) - float(r["effective_min_edge"]))
            allowed_ask_increase = float(args.max_ask_increase) + float(args.confidence_reprice_k) * edge_buffer
            execution_price_buffer = max(0.0, float(args.execution_price_buffer_cents)) / 100.0
            if (live_ask + execution_price_buffer) > float(r["ask"]) + allowed_ask_increase + 1e-12:
                print(
                    f"  Skip {r['ticker']}: live ask moved away "
                    f"({r['ask']:.3f}->{live_ask:.3f}, max_increase={allowed_ask_increase:.3f}, "
                    f"buffer={execution_price_buffer:.3f})."
                )
                continue

            live_ask_exec = min(0.99, max(0.01, live_ask + execution_price_buffer))
            price_cents = int(math.ceil(float(live_ask_exec) * 100.0 - 1e-9))
            live_ask_exec = float(price_cents) / 100.0
            fair_side = float(r["fair_p_yes"]) if r["side"] == "yes" else (1.0 - float(r["fair_p_yes"]))
            live_edge = fair_side - float(live_ask_exec)
            if live_edge < float(r["effective_min_edge"]) - 1e-12:
                print(
                    f"  Skip {r['ticker']}: live edge fell below threshold "
                    f"({live_edge:.3f} < {float(r['effective_min_edge']):.3f})."
                )
                continue

            # Recompute count at executable ask (includes price buffer / cent rounding).
            live_contracts = max(1, int(stake_per_trade / live_ask_exec))
            while live_contracts > 1 and (live_contracts * live_ask_exec) > (stake_per_trade + 1e-9):
                live_contracts -= 1
            live_est_spend = float(live_contracts) * float(live_ask_exec)
            top_size = _top_ask_size_for_side(live, r["side"])
            if bool(args.require_top_size):
                if top_size is None:
                    print(f"  Skip {r['ticker']}: top ask size unavailable for side={r['side']}.")
                    continue
                max_top_contracts = int(math.floor(float(top_size) + 1e-9))
                if max_top_contracts <= 0:
                    print(f"  Skip {r['ticker']}: top ask size too small (have={float(top_size):.2f}).")
                    continue
                if max_top_contracts < int(live_contracts):
                    prev_contracts = int(live_contracts)
                    prev_spend = float(live_est_spend)
                    live_contracts = int(max_top_contracts)
                    live_est_spend = float(live_contracts) * float(live_ask_exec)
                    print(
                        f"  Downsize {r['ticker']}: top ask size limited "
                        f"(need={prev_contracts}, have={float(top_size):.2f}) -> "
                        f"x{live_contracts} (est ${prev_spend:.2f} -> ${live_est_spend:.2f})."
                    )

            # Re-check cap after any top-size downsize.
            live_used = float(ticker_cost.get(r["ticker"], 0.0))
            if live_used + live_est_spend > max_per_market + 1e-9:
                print(
                    f"  Skip {r['ticker']}: live cap block "
                    f"(used=${live_used:.2f} + est=${live_est_spend:.2f} > ${max_per_market:.2f})"
                )
                continue

            buy_cap_cents = int(live_contracts) * int(price_cents) + int(
                max(0, int(args.buy_max_cost_buffer_cents))
            )
            place_kwargs: dict = {}
            if bool(args.fill_or_kill):
                place_kwargs["time_in_force"] = "fill_or_kill"
                place_kwargs["buy_max_cost"] = buy_cap_cents

            resp = kalshi.place_order(
                ticker=r["ticker"],
                side=r["side"],
                count=int(live_contracts),
                price=price_cents,
                **place_kwargs,
            )

            fill_cost, fee_cost = _extract_fill_cost_and_fees(resp if isinstance(resp, dict) else {})
            filled_count = _extract_fill_count(resp if isinstance(resp, dict) else {})
            order_status = str((resp.get("order", {}) if isinstance(resp, dict) else {}).get("status", "")).lower()
            effective_spend = (
                float(fill_cost) if fill_cost is not None else float(filled_count) * float(live_ask_exec)
            )

            if bool(args.fill_or_kill) and filled_count + 1e-9 < float(live_contracts):
                print(
                    f"  No fill {r['ticker']} {r['side']} x{int(live_contracts)} "
                    f"(status={order_status or 'unknown'}; likely book moved or size disappeared)."
                )
                log_trade(
                    market_ticker=r["ticker"],
                    city=r["city"],
                    market_type="simple_mispricing",
                    contract_desc=r["title"],
                    fair_p=r["fair_p_yes"],
                    yes_ask=live_ask_exec if r["side"] == "yes" else 0,
                    no_ask=live_ask_exec if r["side"] == "no" else 0,
                    edge_yes=r["edge_yes"],
                    edge_no=r["edge_no"],
                    action="place",
                    side=r["side"],
                    size_dollars=0.0,
                    contract_count=int(live_contracts),
                    status="no_fill",
                    notes=f"status={order_status or 'unknown'};fok=1;buy_cap_cents={buy_cap_cents}",
                )
                continue

            if effective_spend <= 0 or filled_count <= 0:
                print(
                    f"  No fill {r['ticker']} {r['side']} x{int(live_contracts)} "
                    f"(status={order_status or 'unknown'})."
                )
                log_trade(
                    market_ticker=r["ticker"],
                    city=r["city"],
                    market_type="simple_mispricing",
                    contract_desc=r["title"],
                    fair_p=r["fair_p_yes"],
                    yes_ask=live_ask_exec if r["side"] == "yes" else 0,
                    no_ask=live_ask_exec if r["side"] == "no" else 0,
                    edge_yes=r["edge_yes"],
                    edge_no=r["edge_no"],
                    action="place",
                    side=r["side"],
                    size_dollars=0.0,
                    contract_count=int(live_contracts),
                    status="no_fill",
                    notes=f"status={order_status or 'unknown'};fok={1 if args.fill_or_kill else 0}",
                )
                continue

            ticker_cost[r["ticker"]] = ticker_cost.get(r["ticker"], 0.0) + effective_spend
            fee_msg = f", fees=${fee_cost:.2f}" if fee_cost is not None else ""
            print(
                f"  Order filled: {r['ticker']} {r['side']} "
                f"x{int(round(filled_count))}/{int(live_contracts)} "
                f"limit={live_ask_exec:.3f} est=${live_est_spend:.2f} fill=${effective_spend:.2f}{fee_msg}"
            )
            log_trade(
                market_ticker=r["ticker"],
                city=r["city"],
                market_type="simple_mispricing",
                contract_desc=r["title"],
                fair_p=r["fair_p_yes"],
                yes_ask=live_ask_exec if r["side"] == "yes" else 0,
                no_ask=live_ask_exec if r["side"] == "no" else 0,
                edge_yes=r["edge_yes"],
                edge_no=r["edge_no"],
                action="place",
                side=r["side"],
                size_dollars=effective_spend,
                contract_count=int(round(filled_count)),
                status="placed",
                notes=(
                    f"est_spend={live_est_spend:.4f};"
                    f"fill_spend={effective_spend:.4f};"
                    f"fill_count={filled_count:.2f};"
                    f"status={order_status or 'unknown'};"
                    f"fok={1 if args.fill_or_kill else 0};"
                    f"buy_cap_cents={buy_cap_cents};"
                    f"fees={(fee_cost if fee_cost is not None else 0.0):.4f}"
                ),
            )
        except Exception as e:
            status_code = _extract_http_status(e)
            if bool(args.fill_or_kill) and status_code == 409:
                requested_contracts = int(locals().get("live_contracts", int(r["contracts"])))
                if bool(args.retry_partial_on_409):
                    partial_retry_filled = False
                    for retry_count in _partial_retry_counts(
                        requested_contracts,
                        int(args.partial_retry_steps),
                    ):
                        retry_live = _fetch_live_market_snapshot(auth, r["ticker"])
                        if retry_live is None:
                            continue
                        retry_ask = (
                            float(retry_live["yes_ask_dollars"])
                            if r["side"] == "yes"
                            else float(retry_live["no_ask_dollars"])
                        )
                        if not (0.0 < retry_ask < 1.0):
                            continue

                        retry_edge_buffer = max(
                            0.0,
                            float(r["edge_chosen"]) - float(r["effective_min_edge"]),
                        )
                        retry_allowed_ask_increase = (
                            float(args.max_ask_increase)
                            + float(args.confidence_reprice_k) * retry_edge_buffer
                        )
                        retry_execution_buffer = max(0.0, float(args.execution_price_buffer_cents)) / 100.0
                        if (retry_ask + retry_execution_buffer) > float(r["ask"]) + retry_allowed_ask_increase + 1e-12:
                            continue

                        retry_ask_exec = min(0.99, max(0.01, retry_ask + retry_execution_buffer))
                        retry_price_cents = int(math.ceil(float(retry_ask_exec) * 100.0 - 1e-9))
                        retry_ask_exec = float(retry_price_cents) / 100.0
                        retry_fair_side = (
                            float(r["fair_p_yes"])
                            if r["side"] == "yes"
                            else (1.0 - float(r["fair_p_yes"]))
                        )
                        retry_edge = retry_fair_side - float(retry_ask_exec)
                        if retry_edge < float(r["effective_min_edge"]) - 1e-12:
                            continue

                        retry_top_size = _top_ask_size_for_side(retry_live, r["side"])
                        if bool(args.require_top_size) and retry_top_size is not None:
                            retry_count = min(
                                int(retry_count),
                                int(math.floor(float(retry_top_size) + 1e-9)),
                            )
                            if retry_count <= 0:
                                continue

                        retry_est_spend = float(retry_count) * float(retry_ask_exec)
                        retry_used = float(ticker_cost.get(r["ticker"], 0.0))
                        if retry_used + retry_est_spend > max_per_market + 1e-9:
                            continue

                        retry_buy_cap_cents = int(retry_count) * int(retry_price_cents) + int(
                            max(0, int(args.buy_max_cost_buffer_cents))
                        )
                        try:
                            retry_resp = kalshi.place_order(
                                ticker=r["ticker"],
                                side=r["side"],
                                count=int(retry_count),
                                price=retry_price_cents,
                                time_in_force="fill_or_kill",
                                buy_max_cost=retry_buy_cap_cents,
                            )
                        except Exception as retry_exc:
                            if _extract_http_status(retry_exc) == 409:
                                continue
                            break

                        retry_fill_cost, retry_fee_cost = _extract_fill_cost_and_fees(
                            retry_resp if isinstance(retry_resp, dict) else {}
                        )
                        retry_filled_count = _extract_fill_count(
                            retry_resp if isinstance(retry_resp, dict) else {}
                        )
                        retry_order_status = str(
                            (retry_resp.get("order", {}) if isinstance(retry_resp, dict) else {}).get(
                                "status",
                                "",
                            )
                        ).lower()
                        retry_effective_spend = (
                            float(retry_fill_cost)
                            if retry_fill_cost is not None
                            else float(retry_filled_count) * float(retry_ask_exec)
                        )
                        if retry_effective_spend <= 0 or retry_filled_count <= 0:
                            continue

                        ticker_cost[r["ticker"]] = ticker_cost.get(r["ticker"], 0.0) + retry_effective_spend
                        retry_fee_msg = (
                            f", fees=${retry_fee_cost:.2f}" if retry_fee_cost is not None else ""
                        )
                        print(
                            f"  Partial fill after conflict: {r['ticker']} {r['side']} "
                            f"x{int(round(retry_filled_count))}/{requested_contracts} "
                            f"(retry x{int(retry_count)} @ {retry_ask_exec:.3f}) "
                            f"fill=${retry_effective_spend:.2f}{retry_fee_msg}"
                        )
                        log_trade(
                            market_ticker=r["ticker"],
                            city=r["city"],
                            market_type="simple_mispricing",
                            contract_desc=r["title"],
                            fair_p=r["fair_p_yes"],
                            yes_ask=retry_ask_exec if r["side"] == "yes" else 0,
                            no_ask=retry_ask_exec if r["side"] == "no" else 0,
                            edge_yes=r["edge_yes"],
                            edge_no=r["edge_no"],
                            action="place",
                            side=r["side"],
                            size_dollars=retry_effective_spend,
                            contract_count=int(round(retry_filled_count)),
                            status="placed",
                            notes=(
                                f"partial_after_409=1;"
                                f"requested={requested_contracts};"
                                f"retry_count={int(retry_count)};"
                                f"fill_count={retry_filled_count:.2f};"
                                f"fill_spend={retry_effective_spend:.4f};"
                                f"status={retry_order_status or 'unknown'};"
                                f"fok=1;"
                                f"buy_cap_cents={retry_buy_cap_cents};"
                                f"fees={(retry_fee_cost if retry_fee_cost is not None else 0.0):.4f}"
                            ),
                        )
                        partial_retry_filled = True
                        break
                    if partial_retry_filled:
                        continue

                print(
                    f"  No fill {r['ticker']} {r['side']} x{requested_contracts} "
                    "(409 conflict: top-of-book changed before execution)."
                )
                log_trade(
                    market_ticker=r["ticker"],
                    city=r["city"],
                    market_type="simple_mispricing",
                    contract_desc=r["title"],
                    fair_p=r["fair_p_yes"],
                    yes_ask=r["ask"] if r["side"] == "yes" else 0,
                    no_ask=r["ask"] if r["side"] == "no" else 0,
                    edge_yes=r["edge_yes"],
                    edge_no=r["edge_no"],
                    action="place",
                    side=r["side"],
                    size_dollars=0.0,
                    contract_count=int(requested_contracts),
                    status="no_fill",
                    notes=f"http_status=409;fok=1;error={str(e)[:200]}",
                )
                continue
            print(f"  Order failed for {r['ticker']}: {e}")
            log_trade(
                market_ticker=r["ticker"],
                city=r["city"],
                market_type="simple_mispricing",
                contract_desc=r["title"],
                fair_p=r["fair_p_yes"],
                yes_ask=r["ask"] if r["side"] == "yes" else 0,
                no_ask=r["ask"] if r["side"] == "no" else 0,
                edge_yes=r["edge_yes"],
                edge_no=r["edge_no"],
                action="place",
                side=r["side"],
                size_dollars=r["est_spend"],
                contract_count=int(r["contracts"]),
                status="failed",
                notes=str(e),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders. Default is dry-run.")
    parser.add_argument("--interval", type=int, default=1, help="Minutes between cycles (0 = run once).")
    parser.add_argument("--db", default="data/weather.db")
    parser.add_argument(
        "--target-date",
        default=None,
        help="YYYY-MM-DD explicit target. If omitted, auto mode uses today before cutoff hour, tomorrow after.",
    )
    parser.add_argument("--trade-today", action="store_true", help="Use today's markets by default.")
    parser.add_argument(
        "--include-tomorrow",
        action="store_true",
        help="Also include tomorrow's markets in each cycle (in addition to the primary target date).",
    )
    parser.add_argument(
        "--target-date-cutoff-hour",
        type=int,
        default=16,
        help="Auto mode cutoff hour (local time): before this hour trade today, otherwise tomorrow.",
    )
    parser.add_argument("--cities", default=None, help="Comma list of cities.")
    parser.add_argument("--exclude-cities", default=None, help="Comma list of cities to exclude from trading.")
    parser.add_argument("--side-mode", choices=["best", "yes", "no"], default="best")
    parser.add_argument(
        "--stake-per-trade",
        type=float,
        default=5.0,
        help="Ignored in execution: hard-capped to $5.00 for safety.",
    )
    parser.add_argument(
        "--max-per-market",
        type=float,
        default=15.0,
        help="Ignored in execution: hard-capped to $15.00 per ticker for safety.",
    )
    parser.add_argument(
        "--require-book-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, skip placement unless the live top-of-book matches the signal snapshot within --max-book-drift.",
    )
    parser.add_argument(
        "--max-book-drift",
        type=float,
        default=0.01,
        help="Allowed absolute drift in dollars for each top-of-book field when --require-book-stable is enabled.",
    )
    parser.add_argument(
        "--max-ask-increase",
        type=float,
        default=0.01,
        help="Maximum allowed increase in the chosen side ask between signal and placement.",
    )
    parser.add_argument(
        "--confidence-reprice-k",
        type=float,
        default=0.0,
        help=(
            "Extra allowed ask increase = k * max(0, signal_edge - effective_min_edge). "
            "Uses confidence-adjusted spare edge; 0 disables."
        ),
    )
    parser.add_argument(
        "--execution-price-buffer-cents",
        type=float,
        default=1.0,
        help="Add this many cents to order limit price (both initial and 409 retries) to improve fill probability.",
    )
    parser.add_argument(
        "--buy-max-cost-buffer-cents",
        type=int,
        default=1,
        help="Extra cents added to buy_max_cost on FOK orders to absorb rounding/fee drift.",
    )
    parser.add_argument(
        "--fill-or-kill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, place with time_in_force=fill_or_kill and buy_max_cost so orders fully fill immediately or do nothing.",
    )
    parser.add_argument(
        "--retry-partial-on-409",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After a 409 conflict on FOK, retry smaller sizes so partial size can still be filled.",
    )
    parser.add_argument(
        "--partial-retry-steps",
        type=int,
        default=4,
        help="How many smaller-size retry attempts to make after a 409 conflict.",
    )
    parser.add_argument(
        "--require-top-size",
        action="store_true",
        help="Require displayed top ask size to cover requested contracts before placement.",
    )
    parser.add_argument(
        "--max-orders-per-cycle",
        type=int,
        default=0,
        help="Execute at most this many highest-edge signals per cycle (0 = no cap).",
    )
    parser.add_argument(
        "--data-refresh-minutes",
        type=float,
        default=5.0,
        help="Refresh DB inputs every N minutes via scripts/update_data.py (0 disables).",
    )
    parser.add_argument(
        "--data-refresh-sources",
        nargs="+",
        choices=DATA_REFRESH_SOURCE_CHOICES,
        default=list(DEFAULT_DATA_REFRESH_SOURCES),
        help="Sources passed to update_data --only during periodic refresh.",
    )
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--max-edge", type=float, default=1.0, help="Skip signals with edge above this (0=disabled).")
    parser.add_argument("--skip-edge-lo", type=float, default=0.05, help="Lower bound of edge skip zone (0=disabled).")
    parser.add_argument("--skip-edge-hi", type=float, default=0.20, help="Upper bound of edge skip zone (0=disabled).")
    parser.add_argument("--use-empirical", action=argparse.BooleanOptionalAction, default=True,
                        help="Use empirical error CDF instead of Normal CDF for probabilities.")
    parser.add_argument("--uncertainty-z", type=float, default=1.0)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-hist-days", type=int, default=45)
    parser.add_argument("--sigma-floor", type=float, default=3.0)
    parser.add_argument("--spread-alpha", type=float, default=0.20)
    parser.add_argument("--ecmwf-weight", type=float, default=0.50)
    parser.add_argument("--nbm-weight", type=float, default=0.30, help="Weight for NWS NBM forecast when available (0=disabled).")
    parser.add_argument("--disagree-alpha", type=float, default=0.15, help="Sigma inflation per degree of GFS/ECMWF disagreement.")
    parser.add_argument("--t850-sigma-alpha", type=float, default=0.0, help="Sigma inflation based on 850hPa temperature anomaly.")
    parser.add_argument("--prob-floor", type=float, default=0.03)
    parser.add_argument("--prob-ceil", type=float, default=0.97)
    parser.add_argument(
        "--confidence-shrink-k",
        type=float,
        default=0.0,
        help="Probability shrink strength toward 50%%; w=n_hist/(n_hist+k).",
    )
    args = parser.parse_args()
    if float(args.max_book_drift) < 0:
        raise ValueError("--max-book-drift must be >= 0.")
    if float(args.max_ask_increase) < 0:
        raise ValueError("--max-ask-increase must be >= 0.")
    if int(args.max_orders_per_cycle) < 0:
        raise ValueError("--max-orders-per-cycle must be >= 0.")
    if int(args.partial_retry_steps) < 0:
        raise ValueError("--partial-retry-steps must be >= 0.")
    if float(args.confidence_shrink_k) < 0:
        raise ValueError("--confidence-shrink-k must be >= 0.")
    if float(args.confidence_reprice_k) < 0:
        raise ValueError("--confidence-reprice-k must be >= 0.")
    if float(args.execution_price_buffer_cents) < 0:
        raise ValueError("--execution-price-buffer-cents must be >= 0.")
    if int(args.buy_max_cost_buffer_cents) < 0:
        raise ValueError("--buy-max-cost-buffer-cents must be >= 0.")
    if float(args.data_refresh_minutes) < 0:
        raise ValueError("--data-refresh-minutes must be >= 0.")

    init_run(model_name="SIMPLE")
    if not args.live:
        print("Mode: DRY RUN")
    else:
        print("Mode: LIVE")

    run_id = start_run("run_live_simple.py", vars(args))
    refresh_period_sec = float(args.data_refresh_minutes) * 60.0
    last_refresh_ts_mono: float | None = None

    try:
        while True:
            now_mono = time.monotonic()
            refresh_due = (
                refresh_period_sec > 0
                and (last_refresh_ts_mono is None or (now_mono - last_refresh_ts_mono) >= refresh_period_sec)
            )
            if refresh_due:
                _refresh_data_sources([str(s) for s in args.data_refresh_sources])
                last_refresh_ts_mono = time.monotonic()

            try:
                run_once(args, dry_run=not args.live)
            except Exception as e:
                print(f"Cycle error: {e}")

            if args.interval <= 0:
                break
            next_ts = datetime.now() + timedelta(minutes=int(args.interval))
            print(f"Sleeping {args.interval} min (next run ~{next_ts.strftime('%Y-%m-%d %H:%M:%S')})")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        end_run(run_id, trade_log_path=TRADE_LOG_PATH)


if __name__ == "__main__":
    main()
