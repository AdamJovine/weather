"""
Live empirical weather trader — prices from iem_forecasts error history.

At each new forecast run:
  1. Get the NBS forecast and note the lead time (hours to settlement)
  2. For the past 10 days, find NBS forecasts made at the SAME lead time
  3. Compare each to the actual observed high → error pool
  4. Price every contract by replaying those errors against today's forecast
  5. Trade where empirical fair value diverges from market

No Normal distribution. No hardcoded sigma. Only real forecast-vs-actual data.

Usage:
    cd new-model
    python trader.py                   # paper trade, continuous
    python trader.py --live            # REAL MONEY
    python trader.py --once --dry-run  # see what it would do
    python trader.py --status          # show positions
    python trader.py --reset           # clear state
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from collector.config import STATIONS, DB_PATH
from collector.backtest import DISSEMINATION_LAG

log = logging.getLogger("trader")

# ── Config ───────────────────────────────────────────────────────────────

DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"

SERIES_TO_ICAO = {s.kalshi_series: s.icao for s in STATIONS}
SERIES_TO_CITY = {s.kalshi_series: s.city for s in STATIONS}
ICAO_TO_CITY = {s.icao: s.city for s in STATIONS}

NBS_LAG = DISSEMINATION_LAG["NBS"]

# Edge (cents) = |empirical_fv - market_price|
# Cost = what you pay per contract (ask for YES, 100-bid for NO)
RULES = [
    {"name": "R1", "min_edge": 10, "max_cost": 40},
    {"name": "R2", "min_edge": 15, "max_cost": 30},
    {"name": "R3", "min_edge": 20, "max_cost": 20},
    {"name": "R4", "min_edge": 30, "max_cost": 15},
]

MAX_CAPITAL = 10000  # cents = $100
MAX_POS = 5
ERROR_LOOKBACK_DAYS = 10
MAX_LEAD_MISMATCH_HOURS = 3

LOG_FILE = Path(__file__).resolve().parent / "data" / "trades.log"
STATE_FILE = Path(__file__).resolve().parent / "data" / "trader_state.json"

_TICKER_RE = re.compile(r"^(KXHIGH\w+)-(\d{2}[A-Z]{3}\d{2})-([TB])([\d.]+)$")


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class Contract:
    ticker: str
    series: str
    target_date: str
    ctype: str
    value: float


def parse_ticker(ticker: str) -> Contract | None:
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    series, date_str, ctype, val = m.groups()
    target_date = datetime.strptime(date_str, "%y%b%d").strftime("%Y-%m-%d")
    return Contract(ticker, series, target_date, ctype, float(val))


# ── Empirical error distribution ─────────────────────────────────────────

def get_errors_at_lead(conn, station, target_date, current_runtime):
    """
    Build forecast errors from the past N days at similar lead times.

    If the current NBS runtime is 15:00Z forecasting April 13's high
    (ftime = April 14 00Z, lead = 9 hours), then for each past settled
    day, find an NBS forecast made ~9 hours before that day's ftime,
    and compare to the actual METAR max.

    Returns list of floats: error = observed_max - nbs_forecast.
    """
    td = datetime.strptime(target_date, "%Y-%m-%d")
    target_ftime = (td + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    runtime_dt = datetime.fromisoformat(
        current_runtime.replace("Z", "+00:00"))
    ftime_dt = datetime.fromisoformat(
        target_ftime.replace("Z", "+00:00"))
    lead = ftime_dt - runtime_dt
    lead_seconds = lead.total_seconds()

    errors = []
    for days_back in range(1, ERROR_LOOKBACK_DAYS + 15):
        if len(errors) >= ERROR_LOOKBACK_DAYS:
            break

        past_date = (td - timedelta(days=days_back)).strftime("%Y-%m-%d")
        past_next = (td - timedelta(days=days_back - 1)).strftime("%Y-%m-%d")
        past_ftime = past_next + "T00:00:00Z"

        # Ideal past runtime = past_ftime - lead
        ideal_rt = (
            datetime.fromisoformat(past_ftime.replace("Z", "+00:00"))
            - lead
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Find NBS forecasts for this past date near the ideal runtime
        rows = conn.execute(
            "SELECT runtime, txn FROM iem_forecasts "
            "WHERE model='NBS' AND station=? AND ftime=? "
            "AND txn IS NOT NULL "
            "ORDER BY runtime",
            (station, past_ftime),
        ).fetchall()

        if not rows:
            continue

        # Pick the runtime closest to ideal_rt
        ideal_dt = datetime.fromisoformat(ideal_rt.replace("Z", "+00:00"))
        best_row = None
        best_diff = float("inf")
        for rt, txn in rows:
            rt_dt = datetime.fromisoformat(rt.replace("Z", "+00:00"))
            diff = abs((rt_dt - ideal_dt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_row = (rt, txn)

        if best_row is None:
            continue
        if best_diff > MAX_LEAD_MISMATCH_HOURS * 3600:
            continue

        forecast = float(best_row[1])

        # Observed max from METAR
        obs_row = conn.execute(
            "SELECT MAX(temp_f) FROM metar_obs "
            "WHERE station=? AND obs_time >= ? AND obs_time < ? "
            "AND temp_f IS NOT NULL",
            (station, past_date + "T00:00:00Z", past_next + "T00:00:00Z"),
        ).fetchone()

        if obs_row is None or obs_row[0] is None:
            continue

        observed = float(obs_row[0])
        errors.append(observed - forecast)

    return errors


def price_empirical(c, forecast, errors, t_low):
    """
    P(YES) by replaying each historical error against the forecast.
    Laplace-smoothed: +1 yes, +1 no pseudo-observations.
    Returns fair value in cents (0-100), or None if no errors.
    """
    if not errors:
        return None

    n_yes = 0
    for err in errors:
        actual = forecast + err
        if c.ctype == "T" and c.value == t_low:
            if actual < c.value:
                n_yes += 1
        elif c.ctype == "T":
            if actual > c.value:
                n_yes += 1
        else:
            x = c.value - 0.5
            if x <= actual <= x + 1:
                n_yes += 1

    p = (n_yes + 1) / (len(errors) + 2)
    return round(p * 100, 1)


# ── Kalshi client ────────────────────────────────────────────────────────

def build_kalshi_client(live):
    from kalshi_python_sync import KalshiAuth, KalshiClient, Configuration
    api_key = os.getenv("KALSHI_API_KEY_ID")
    private_key = os.getenv("KALSHI_PRIVATE_KEY")
    if not api_key or not private_key:
        raise RuntimeError("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY not set")
    base_url = PROD_URL if live else DEMO_URL
    config = Configuration(host=base_url)
    client = KalshiClient(configuration=config)
    client.kalshi_auth = KalshiAuth(api_key, private_key)
    return client, base_url


def fetch_live_markets(client):
    series_tickers = sorted({s.kalshi_series for s in STATIONS
                             if s.kalshi_series in SERIES_TO_ICAO})
    markets = {}
    for i, series in enumerate(series_tickers):
        if i > 0:
            time.sleep(0.2)
        try:
            resp = client._market_api.get_markets(
                series_ticker=series, status="open", limit=200)
            mlist = (resp.markets if hasattr(resp, "markets")
                     else resp.get("markets", []))
        except Exception:
            continue
        for m in mlist:
            m = m if isinstance(m, dict) else m.to_dict()
            ticker = m.get("ticker")
            if not ticker:
                continue
            def to_cents(v):
                if v is None: return None
                f = float(v)
                return round(f * 100) if f < 1.01 else round(f)
            markets[ticker] = {
                "yes_bid": to_cents(
                    m.get("yes_bid_dollars") or m.get("yes_bid")),
                "yes_ask": to_cents(
                    m.get("yes_ask_dollars") or m.get("yes_ask")),
            }
    return markets


def place_order(client, ticker, side, count, price_cents, dry_run=False):
    order_id = str(uuid.uuid4())
    if dry_run:
        return {"status": "dry_run", "client_order_id": order_id}
    try:
        kwargs = dict(ticker=ticker, action="buy", side=side, count=count,
                      type="limit", client_order_id=order_id,
                      time_in_force="fill_or_kill")
        if side == "yes":
            kwargs["yes_price"] = price_cents
        else:
            kwargs["no_price"] = price_cents
        resp = client._orders_api.create_order(**kwargs)
        return resp if isinstance(resp, dict) else resp.to_dict()
    except Exception as e:
        log.error("ORDER FAILED %s: %s", ticker, e)
        return None


# ── Forecast detection ───────────────────────────────────────────────────

def detect_new_forecasts(db_path, prev_runtimes):
    conn = sqlite3.connect(str(db_path))
    now = pd.Timestamp.now(tz="UTC")
    new_stations = defaultdict(list)

    for model in ("NBS", "GFS", "LAV"):
        lag = DISSEMINATION_LAG.get(model, timedelta(0))
        cutoff = (now - lag).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = conn.execute(
            "SELECT station, MAX(runtime) FROM iem_forecasts "
            "WHERE model=? AND runtime <= ? GROUP BY station",
            (model, cutoff),
        ).fetchall()
        for stn, rt in rows:
            if rt is None:
                continue
            key = f"{model}:{stn}"
            if key not in prev_runtimes or rt != prev_runtimes[key]:
                prev_runtimes[key] = rt
                new_stations[stn].append(model)

    conn.close()
    return dict(new_stations)


def get_nbs_forecast_and_runtime(db_path, station, target_date):
    """Returns (txn, runtime_str) or (None, None)."""
    conn = sqlite3.connect(str(db_path))
    td = datetime.strptime(target_date, "%Y-%m-%d")
    next_00z = (td + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    now = pd.Timestamp.now(tz="UTC")
    cutoff = (now - NBS_LAG).strftime("%Y-%m-%dT%H:%M:%SZ")

    row = conn.execute(
        "SELECT txn, runtime FROM iem_forecasts "
        "WHERE model='NBS' AND station=? AND ftime=? "
        "AND runtime <= ? AND txn IS NOT NULL "
        "ORDER BY runtime DESC LIMIT 1",
        (station, next_00z, cutoff),
    ).fetchone()
    conn.close()
    if row and row[0] is not None:
        return float(row[0]), row[1]
    return None, None


# ── State ────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"prev_runtimes": {}, "positions": {},
            "capital_used": 0, "trades": []}


def save_state(state):
    # Don't persist transient metadata
    to_save = {k: v for k, v in state.items() if not k.startswith("_")}
    STATE_FILE.write_text(json.dumps(to_save, indent=2, default=str))


def get_run_metadata(live: bool, dry_run: bool) -> dict:
    """Capture everything needed to reproduce this run."""
    git_hash = "unknown"
    git_dirty = False
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_dirty = len(dirty) > 0
    except Exception:
        pass

    return {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "rules": RULES,
        "max_capital_cents": MAX_CAPITAL,
        "max_pos_per_contract": MAX_POS,
        "error_lookback_days": ERROR_LOOKBACK_DAYS,
        "max_lead_mismatch_hours": MAX_LEAD_MISMATCH_HOURS,
        "nbs_lag_minutes": NBS_LAG.total_seconds() / 60,
        "environment": "production" if live else "demo",
        "dry_run": dry_run,
        "db_path": str(DB_PATH),
    }


def log_run_start(metadata: dict):
    """Write a RUN_START header line to the trade log."""
    entry = {
        "type": "RUN_START",
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        **metadata,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_trade(trade: dict, metadata: dict):
    """Append one trade line with run metadata."""
    entry = {
        "type": "TRADE",
        "git_hash": metadata["git_hash"],
        "environment": metadata["environment"],
        "dry_run": metadata["dry_run"],
        **trade,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ── Display ──────────────────────────────────────────────────────────────

def print_trade_table(trades):
    if not trades:
        return
    print(f"  {'Rule':<4} {'Side':<8} {'Ticker':<30} {'City':<12} "
          f"{'Fcst':>5} {'FV':>5} {'Mkt':>5} {'Edge':>5} "
          f"{'Cost':>5} {'Errs':>4}")
    print(f"  {'─' * 105}")
    for t in trades:
        side_str = "BUY YES" if t["side"] == "BUY_YES" else "BUY NO"
        mkt = t["price"] if t["side"] == "BUY_YES" else t["price"]
        print(f"  {t['rule']:<4} {side_str:<8} {t['ticker']:<30} "
              f"{t['city']:<12} "
              f"{t['forecast']:>4.0f}° {t['fv']:>4.0f}c "
              f"{mkt:>4}c {t['edge']:>+4.0f}c "
              f"{t['cost']:>4}c {t['n_errors']:>4}")


def print_error_pool(errors, station, target_date, forecast, runtime):
    if not errors:
        print(f"    {ICAO_TO_CITY.get(station, station)} → {target_date}: "
              f"no past data at this lead time")
        return
    import numpy as np
    errs_str = " ".join(f"{e:+.0f}" for e in sorted(errors))
    print(f"    {ICAO_TO_CITY.get(station, station)} → {target_date}: "
          f"fcst={forecast:.0f}°F  runtime={runtime[11:16]}  "
          f"N={len(errors)}  errors=[{errs_str}]  "
          f"mean={np.mean(errors):+.1f}  std={np.std(errors):.1f}")


def print_summary(state):
    cap = state["capital_used"]
    positions = {k: v for k, v in state["positions"].items() if v != 0}
    n_trades = len(state["trades"])

    print(f"\n  Capital: ${cap/100:.2f} / ${MAX_CAPITAL/100:.2f} used  "
          f"| {len(positions)} open positions | {n_trades} total trades")

    if positions:
        by_rule = defaultdict(list)
        for k, v in sorted(positions.items()):
            rule, ticker = k.split(":", 1)
            by_rule[rule].append((ticker, v))
        for rule in sorted(by_rule):
            tickers = by_rule[rule]
            longs = sum(1 for _, v in tickers if v > 0)
            shorts = sum(1 for _, v in tickers if v < 0)
            total_lots = sum(abs(v) for _, v in tickers)
            print(f"    {rule}: {len(tickers)} contracts "
                  f"({longs} long, {shorts} short, {total_lots} lots)")


# ── Main loop ────────────────────────────────────────────────────────────

def run_once(client, state, dry_run=False):
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    new_stations = detect_new_forecasts(DB_PATH, state["prev_runtimes"])
    if not new_stations:
        return 0

    triggers = []
    for stn in sorted(new_stations):
        city = ICAO_TO_CITY.get(stn, stn)
        models = ",".join(new_stations[stn])
        triggers.append(f"{city}({models})")

    print(f"\n{'━' * 80}")
    print(f"  {now_str}  NEW FORECAST DATA")
    print(f"  {', '.join(triggers)}")
    print(f"{'━' * 80}")

    # Fetch live prices
    try:
        markets = fetch_live_markets(client)
    except Exception as e:
        print(f"  ERROR fetching markets: {e}")
        return 0
    if not markets:
        print("  No open markets.")
        return 0
    print(f"  Fetched {len(markets)} contracts")

    # Parse and group
    groups = defaultdict(list)
    for ticker in markets:
        c = parse_ticker(ticker)
        if c:
            groups[(c.series, c.target_date)].append(c)
    bounds = {}
    for key, grp in groups.items():
        tv = sorted(c.value for c in grp if c.ctype == "T")
        bounds[key] = tv[0] if tv else 0

    # Build error pools and price
    conn = sqlite3.connect(str(DB_PATH))
    signal_trades = []

    print(f"\n  Error pools (past {ERROR_LOOKBACK_DAYS} days "
          f"at matched lead time):")

    for (series, td), grp in sorted(groups.items()):
        icao = SERIES_TO_ICAO.get(series)
        city = SERIES_TO_CITY.get(series, series)
        if not icao or icao not in new_stations:
            continue

        forecast, runtime = get_nbs_forecast_and_runtime(DB_PATH, icao, td)
        if forecast is None:
            continue

        errors = get_errors_at_lead(conn, icao, td, runtime)
        print_error_pool(errors, icao, td, forecast, runtime)

        if not errors:
            continue

        t_low = bounds[(series, td)]

        for c in grp:
            mkt = markets.get(c.ticker)
            if not mkt:
                continue
            bid = mkt.get("yes_bid")
            ask = mkt.get("yes_ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            if bid >= 97 or ask <= 3:
                continue

            fv = price_empirical(c, forecast, errors, t_low)
            if fv is None:
                continue

            mid = (bid + ask) / 2.0
            edge_buy = fv - ask     # positive = YES underpriced
            edge_sell = bid - fv    # positive = YES overpriced

            for rule in RULES:
                rname = rule["name"]
                min_edge = rule["min_edge"]
                max_cost = rule["max_cost"]
                pos_key = f"{rname}:{c.ticker}"
                pos = state["positions"].get(pos_key, 0)

                if state["capital_used"] >= MAX_CAPITAL:
                    continue

                # BUY YES: model says worth more than ask
                if (edge_buy >= min_edge and ask <= max_cost
                        and pos < MAX_POS):
                    cost = ask
                    if state["capital_used"] + cost > MAX_CAPITAL:
                        continue
                    result = place_order(client, c.ticker, "yes", 1,
                                         ask, dry_run=dry_run)
                    if result:
                        state["positions"][pos_key] = pos + 1
                        state["capital_used"] += cost
                        trade = {
                            "time": now_str, "rule": rname,
                            "ticker": c.ticker, "side": "BUY_YES",
                            "price": ask, "cost": cost, "fv": fv,
                            "edge": round(edge_buy, 1),
                            "forecast": forecast,
                            "n_errors": len(errors),
                            "city": city, "target_date": td,
                        }
                        state["trades"].append(trade)
                        signal_trades.append(trade)
                        log_trade(trade, state["_metadata"])

                # BUY NO: model says YES worth less than bid
                elif (edge_sell >= min_edge
                      and (100 - bid) <= max_cost
                      and pos > -MAX_POS):
                    cost = 100 - bid
                    if state["capital_used"] + cost > MAX_CAPITAL:
                        continue
                    result = place_order(client, c.ticker, "no", 1,
                                         cost, dry_run=dry_run)
                    if result:
                        state["positions"][pos_key] = pos - 1
                        state["capital_used"] += cost
                        trade = {
                            "time": now_str, "rule": rname,
                            "ticker": c.ticker, "side": "BUY_NO",
                            "price": bid, "cost": cost, "fv": fv,
                            "edge": round(edge_sell, 1),
                            "forecast": forecast,
                            "n_errors": len(errors),
                            "city": city, "target_date": td,
                        }
                        state["trades"].append(trade)
                        signal_trades.append(trade)
                        log_trade(trade, state["_metadata"])

    conn.close()

    if signal_trades:
        print(f"\n  {len(signal_trades)} ORDERS"
              f"{' (DRY RUN)' if dry_run else ''}:")
        print()
        print_trade_table(signal_trades)
    else:
        print(f"\n  No trades (no contracts met edge/cost criteria "
              f"given the empirical distribution)")

    print_summary(state)
    save_state(state)
    return len(signal_trades)


def main():
    parser = argparse.ArgumentParser(description="Empirical weather trader")
    parser.add_argument("--live", action="store_true",
                        help="PRODUCTION API (real money)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.reset:
        save_state({"prev_runtimes": {}, "positions": {},
                     "capital_used": 0, "trades": []})
        print("State reset.")
        return

    state = load_state()

    if args.status:
        print_summary(state)
        trades = state.get("trades", [])
        if trades:
            print(f"\n  Last 20 trades:")
            print_trade_table(trades[-20:])
        return

    env = "PRODUCTION" if args.live else "DEMO (paper)"
    dry = " [DRY RUN]" if args.dry_run else ""

    print(f"{'━' * 80}")
    print(f"  WEATHER TRADER — {env}{dry}")
    print(f"  Pricing: empirical errors from past {ERROR_LOOKBACK_DAYS} "
          f"days at matched lead time")
    print(f"  Rules: R1(edge>=10,c<=40) R2(edge>=15,c<=30) "
          f"R3(edge>=20,c<=20) R4(edge>=30,c<=15)")
    print(f"  Capital: ${MAX_CAPITAL/100:.0f} max  |  "
          f"Position: {MAX_POS} lots/contract")
    print(f"  Trigger: any new NBS/GFS/LAV run  |  "
          f"Poll: every {args.interval}s")
    print(f"  Log: {LOG_FILE}")
    print(f"{'━' * 80}")

    if args.live and not args.dry_run:
        print("\n  *** REAL MONEY ***")
        resp = input("  Type 'yes' to confirm: ")
        if resp.strip().lower() != "yes":
            print("  Aborted.")
            return

    try:
        client, base_url = build_kalshi_client(live=args.live)
        print(f"  Connected to {base_url}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    # Capture run metadata and log it
    metadata = get_run_metadata(live=args.live, dry_run=args.dry_run)
    state["_metadata"] = metadata
    log_run_start(metadata)
    print(f"  Git: {metadata['git_hash'][:10]}"
          f"{'(dirty)' if metadata['git_dirty'] else ''}")

    print_summary(state)

    if args.once:
        run_once(client, state, dry_run=args.dry_run)
    else:
        print(f"\n  Watching for new forecasts... (Ctrl+C to stop)\n")
        try:
            while True:
                run_once(client, state, dry_run=args.dry_run)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n{'━' * 80}")
            print(f"  STOPPED")
            print_summary(state)
            save_state(state)
            print(f"{'━' * 80}")


if __name__ == "__main__":
    main()
