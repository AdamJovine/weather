"""
Live weather trader — two strategies, run one per terminal.

  buy-yes-dip:  BUY YES when forecast edge >= 10c AND ask <= 8c
                AND price dropped 3c+ in last hour.
                High ROI (400%+), 33% win rate, cheap bets.

  buy-no-drift: BUY NO when forecast edge >= 5c AND mid 20-35c
                AND price rose 2c+ in last 6 hours.
                Steady (22-35% ROI), 89%+ win rate.

Usage:
    # Terminal 1:
    python trader.py buy-yes-dip --live -y

    # Terminal 2:
    python trader.py buy-no-drift --live -y
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
ERROR_LOOKBACK_DAYS = 180
MAX_LEAD_MISMATCH_HOURS = 3
MAX_CAPITAL = 10000  # cents = $100
MAX_POS = 5

STRATEGIES = {
    "buy-yes-dip": {
        "side": "YES",
        "min_edge": 10,
        "max_cost": 8,       # ask <= 8c
        "min_drop": 3,       # price dropped 3c+ in last hour
        "drop_window": 12,   # 12 ticks = 60 min
        "desc": "BUY YES: edge>=10c, ask<=8c, dropped 3c+ in 1h",
    },
    "buy-no-drift": {
        "side": "NO",
        "min_edge": 5,
        "mid_lo": 20,
        "mid_hi": 35,
        "min_rise": 2,       # price rose 2c+ in last 6h
        "rise_window": 72,   # 72 ticks = 6h
        "desc": "BUY NO: edge>=5c, mid 20-35c, rose 2c+ in 6h",
    },
}

_TICKER_RE = re.compile(r"^(KXHIGH\w+)-(\d{2}[A-Z]{3}\d{2})-([TB])([\d.]+)$")


@dataclass
class Contract:
    ticker: str
    series: str
    target_date: str
    ctype: str
    value: float


def parse_ticker(ticker):
    m = _TICKER_RE.match(ticker)
    if not m: return None
    series, date_str, ctype, val = m.groups()
    td = datetime.strptime(date_str, "%y%b%d").strftime("%Y-%m-%d")
    return Contract(ticker, series, td, ctype, float(val))


# ── Empirical pricing ───────────────────────────────────────────────────

def get_errors_at_lead(conn, station, target_date, current_runtime):
    td = datetime.strptime(target_date, "%Y-%m-%d")
    target_ftime = (td + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    runtime_dt = datetime.fromisoformat(current_runtime.replace("Z", "+00:00"))
    ftime_dt = datetime.fromisoformat(target_ftime.replace("Z", "+00:00"))
    lead = ftime_dt - runtime_dt

    errors = []
    for days_back in range(1, ERROR_LOOKBACK_DAYS + 30):
        if len(errors) >= ERROR_LOOKBACK_DAYS:
            break
        past_date = (td - timedelta(days=days_back)).strftime("%Y-%m-%d")
        past_next = (td - timedelta(days=days_back - 1)).strftime("%Y-%m-%d")
        past_ftime = past_next + "T00:00:00Z"
        ideal_rt = (datetime.fromisoformat(
            past_ftime.replace("Z", "+00:00")) - lead)

        best_row = conn.execute(
            "SELECT runtime, txn FROM iem_forecasts "
            "WHERE model='NBS' AND station=? AND ftime=? "
            "AND txn IS NOT NULL ORDER BY runtime DESC",
            (station, past_ftime)).fetchall()

        best_txn = None
        best_diff = float("inf")
        for rt_str, txn in best_row:
            rt_dt = datetime.fromisoformat(rt_str.replace("Z", "+00:00"))
            diff = abs((rt_dt - ideal_rt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_txn = txn

        if best_txn is None or best_diff > MAX_LEAD_MISMATCH_HOURS * 3600:
            continue

        obs_val = None
        try:
            oh = conn.execute(
                "SELECT max_tmpf FROM observed_highs WHERE station=? AND date=?",
                (station, past_date)).fetchone()
            if oh and oh[0] is not None:
                obs_val = float(oh[0])
        except sqlite3.OperationalError:
            pass
        if obs_val is None:
            mr = conn.execute(
                "SELECT MAX(temp_f) FROM metar_obs "
                "WHERE station=? AND obs_time >= ? AND obs_time < ? "
                "AND temp_f IS NOT NULL",
                (station, past_date + "T00:00:00Z",
                 past_next + "T00:00:00Z")).fetchone()
            if mr and mr[0] is not None:
                obs_val = float(mr[0])
        if obs_val is None:
            continue

        errors.append(obs_val - best_txn)
    return errors


def price_empirical(c, forecast, errors, t_low):
    if not errors: return None
    n_yes = 0
    for err in errors:
        actual = forecast + err
        if c.ctype == "T" and c.value == t_low:
            if actual < c.value: n_yes += 1
        elif c.ctype == "T":
            if actual > c.value: n_yes += 1
        else:
            x = c.value - 0.5
            if x <= actual <= x + 1: n_yes += 1
    return round((n_yes + 1) / (len(errors) + 2) * 100, 1)


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
        if i > 0: time.sleep(0.2)
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
            if not ticker: continue
            def to_cents(v):
                if v is None: return None
                f = float(v)
                return round(f * 100) if f < 1.01 else round(f)
            markets[ticker] = {
                "yes_bid": to_cents(m.get("yes_bid_dollars") or m.get("yes_bid")),
                "yes_ask": to_cents(m.get("yes_ask_dollars") or m.get("yes_ask")),
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
        if "409" in str(e):
            log.warning("Order rejected (closed): %s", ticker)
        else:
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
            (model, cutoff)).fetchall()
        for stn, rt in rows:
            if rt is None: continue
            key = f"{model}:{stn}"
            if key not in prev_runtimes or rt != prev_runtimes[key]:
                prev_runtimes[key] = rt
                new_stations[stn].append(model)
    conn.close()
    return dict(new_stations)


def get_nbs_forecast_and_runtime(db_path, station, target_date):
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
        (station, next_00z, cutoff)).fetchone()
    conn.close()
    if row and row[0] is not None:
        return float(row[0]), row[1]
    return None, None


# ── Price history for swing detection ────────────────────────────────────

def get_price_history(db_path, ticker, n_ticks=72):
    """Get last N 5-minute mid prices for a ticker."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT ts, (yes_bid + yes_ask) / 2.0 as mid "
        "FROM kalshi_prices WHERE ticker=? "
        "ORDER BY ts DESC LIMIT ?",
        (ticker, n_ticks + 1)).fetchall()
    conn.close()
    if not rows:
        return None, None
    current_mid = rows[0][1]
    if len(rows) <= 1:
        return current_mid, None
    return current_mid, rows


# ── State + logging ──────────────────────────────────────────────────────

def load_state(state_file):
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"prev_runtimes": {}, "positions": {},
            "capital_used": 0, "trades": []}


def save_state(state, state_file):
    to_save = {k: v for k, v in state.items() if not k.startswith("_")}
    state_file.write_text(json.dumps(to_save, indent=2, default=str))


def get_run_metadata(strategy_name, live, dry_run):
    git_hash = "unknown"
    git_dirty = False
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
        git_dirty = len(dirty) > 0
    except Exception:
        pass

    strat = STRATEGIES[strategy_name]
    return {
        "git_hash": git_hash, "git_dirty": git_dirty,
        "strategy": strategy_name,
        "strategy_params": strat,
        "max_capital_cents": MAX_CAPITAL,
        "max_pos": MAX_POS,
        "error_lookback_days": ERROR_LOOKBACK_DAYS,
        "environment": "production" if live else "demo",
        "dry_run": dry_run,
    }


def log_entry(entry, log_file):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def print_summary(state, strategy_name):
    cap = state["capital_used"]
    positions = {k: v for k, v in state["positions"].items() if v != 0}
    n_trades = len(state["trades"])
    print(f"\n  [{strategy_name}] Capital: ${cap/100:.2f} / "
          f"${MAX_CAPITAL/100:.2f}  |  "
          f"{len(positions)} positions  |  {n_trades} trades")
    if positions:
        for k, v in sorted(positions.items()):
            direction = f"+{v} YES" if v > 0 else f"{v} YES"
            print(f"    {k:<32} {direction}")


# ── Main loop ────────────────────────────────────────────────────────────

def run_once(client, state, strategy_name, log_file, dry_run=False):
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    strat = STRATEGIES[strategy_name]

    new_stations = detect_new_forecasts(DB_PATH, state["prev_runtimes"])
    if not new_stations:
        return 0

    triggers = [f"{ICAO_TO_CITY.get(s, s)}({','.join(m)})"
                for s, m in sorted(new_stations.items())]
    print(f"\n{'━' * 80}")
    print(f"  [{strategy_name}] {now_str}  NEW FORECAST")
    print(f"  {', '.join(triggers)}")
    print(f"{'━' * 80}")

    try:
        markets = fetch_live_markets(client)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0
    if not markets:
        print("  No markets.")
        return 0
    print(f"  {len(markets)} contracts")

    groups = defaultdict(list)
    for ticker in markets:
        c = parse_ticker(ticker)
        if c: groups[(c.series, c.target_date)].append(c)
    bounds = {}
    for key, grp in groups.items():
        tv = sorted(c.value for c in grp if c.ctype == "T")
        bounds[key] = tv[0] if tv else 0

    conn = sqlite3.connect(str(DB_PATH))
    signal_trades = []
    today = now.strftime("%Y-%m-%d")

    for (series, td), grp in sorted(groups.items()):
        if td <= today: continue
        icao = SERIES_TO_ICAO.get(series)
        city = SERIES_TO_CITY.get(series, series)
        if not icao or icao not in new_stations: continue

        forecast, runtime = get_nbs_forecast_and_runtime(DB_PATH, icao, td)
        if forecast is None: continue

        errors = get_errors_at_lead(conn, icao, td, runtime)
        if len(errors) < 10: continue

        t_low = bounds[(series, td)]

        for c in grp:
            mkt = markets.get(c.ticker)
            if not mkt: continue
            bid = mkt.get("yes_bid")
            ask = mkt.get("yes_ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0: continue
            if bid >= 97 or ask <= 3: continue

            fv = price_empirical(c, forecast, errors, t_low)
            if fv is None: continue

            mid = (bid + ask) / 2.0
            pos_key = c.ticker
            pos = state["positions"].get(pos_key, 0)

            if state["capital_used"] >= MAX_CAPITAL:
                continue

            traded = False

            if strat["side"] == "YES":
                edge = fv - ask
                if edge < strat["min_edge"] or ask > strat["max_cost"]:
                    continue
                if pos >= MAX_POS:
                    continue

                # Check price drop
                current_mid, history = get_price_history(
                    DB_PATH, c.ticker, strat["drop_window"])
                if history is None or len(history) <= strat["drop_window"]:
                    continue
                old_mid = history[strat["drop_window"]][1]
                if old_mid is None:
                    continue
                price_change = current_mid - old_mid
                if price_change > -strat["min_drop"]:
                    continue

                cost = ask
                if state["capital_used"] + cost > MAX_CAPITAL:
                    continue

                result = place_order(client, c.ticker, "yes", 1,
                                     ask, dry_run=dry_run)
                if result:
                    state["positions"][pos_key] = pos + 1
                    state["capital_used"] += cost
                    trade = {
                        "time": now_str, "strategy": strategy_name,
                        "ticker": c.ticker, "side": "BUY_YES",
                        "price": ask, "cost": cost, "fv": fv,
                        "edge": round(edge, 1),
                        "price_change_1h": round(price_change, 1),
                        "forecast": forecast, "n_errors": len(errors),
                        "city": city, "target_date": td,
                    }
                    state["trades"].append(trade)
                    signal_trades.append(trade)
                    log_entry({"type": "TRADE",
                               "git_hash": state["_metadata"]["git_hash"],
                               **trade}, log_file)

            elif strat["side"] == "NO":
                edge = bid - fv
                no_cost = 100 - bid
                if edge < strat["min_edge"]:
                    continue
                if mid < strat["mid_lo"] or mid >= strat["mid_hi"]:
                    continue
                if pos <= -MAX_POS:
                    continue

                # Check price rise
                current_mid, history = get_price_history(
                    DB_PATH, c.ticker, strat["rise_window"])
                if history is None or len(history) <= strat["rise_window"]:
                    continue
                old_mid = history[strat["rise_window"]][1]
                if old_mid is None:
                    continue
                price_change = current_mid - old_mid
                if price_change < strat["min_rise"]:
                    continue

                cost = no_cost
                if state["capital_used"] + cost > MAX_CAPITAL:
                    continue

                result = place_order(client, c.ticker, "no", 1,
                                     cost, dry_run=dry_run)
                if result:
                    state["positions"][pos_key] = pos - 1
                    state["capital_used"] += cost
                    trade = {
                        "time": now_str, "strategy": strategy_name,
                        "ticker": c.ticker, "side": "BUY_NO",
                        "price": bid, "cost": cost, "fv": fv,
                        "edge": round(edge, 1),
                        "price_change_6h": round(price_change, 1),
                        "forecast": forecast, "n_errors": len(errors),
                        "city": city, "target_date": td,
                    }
                    state["trades"].append(trade)
                    signal_trades.append(trade)
                    log_entry({"type": "TRADE",
                               "git_hash": state["_metadata"]["git_hash"],
                               **trade}, log_file)

    conn.close()

    if signal_trades:
        print(f"\n  {len(signal_trades)} ORDERS"
              f"{' (DRY RUN)' if dry_run else ''}:")
        for t in signal_trades:
            side = t["side"]
            pc = t.get("price_change_1h", t.get("price_change_6h", 0))
            print(f"    {side:<8} {t['ticker']:<30} {t['city']:<12} "
                  f"fcst={t['forecast']:.0f}° fv={t['fv']:.0f}c "
                  f"edge={t['edge']:+.0f}c cost={t['cost']}c "
                  f"Δprice={pc:+.0f}c")
    else:
        print(f"  No trades matching {strat['desc']}")

    print_summary(state, strategy_name)
    save_state(state, state["_state_file"])
    return len(signal_trades)


def main():
    parser = argparse.ArgumentParser(description="Weather trader")
    parser.add_argument("strategy", choices=list(STRATEGIES.keys()),
                        help="Which strategy to run")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    sname = args.strategy
    strat = STRATEGIES[sname]
    data_dir = Path(__file__).resolve().parent / "data"
    log_file = data_dir / f"trades_{sname}.log"
    state_file = data_dir / f"state_{sname}.json"

    if args.reset:
        save_state({"prev_runtimes": {}, "positions": {},
                     "capital_used": 0, "trades": []}, state_file)
        print(f"[{sname}] State reset.")
        return

    state = load_state(state_file)
    state["_state_file"] = state_file

    if args.status:
        print_summary(state, sname)
        for t in state.get("trades", [])[-10:]:
            pc = t.get("price_change_1h", t.get("price_change_6h", 0))
            print(f"    {t['time']}  {t['side']:<8} {t['ticker']:<28} "
                  f"cost={t['cost']}c  edge={t['edge']:+.0f}c  "
                  f"Δ={pc:+.0f}c")
        return

    env = "PRODUCTION" if args.live else "DEMO (paper)"
    dry = " [DRY RUN]" if args.dry_run else ""

    print(f"{'━' * 80}")
    print(f"  [{sname}] {env}{dry}")
    print(f"  {strat['desc']}")
    print(f"  Capital: ${MAX_CAPITAL/100:.0f}  |  "
          f"Max pos: {MAX_POS}  |  Poll: {args.interval}s")
    print(f"  Log: {log_file}")
    print(f"{'━' * 80}")

    if args.live and not args.dry_run and not args.yes:
        resp = input("  Type 'yes' to confirm: ")
        if resp.strip().lower() != "yes":
            print("  Aborted."); return

    try:
        client, base_url = build_kalshi_client(live=args.live)
        print(f"  Connected to {base_url}")
    except Exception as e:
        print(f"  Connection failed: {e}"); sys.exit(1)

    metadata = get_run_metadata(sname, args.live, args.dry_run)
    state["_metadata"] = metadata
    log_entry({"type": "RUN_START", "time": datetime.now(timezone.utc)
               .strftime("%Y-%m-%d %H:%M:%S UTC"), **metadata}, log_file)
    print(f"  Git: {metadata['git_hash'][:10]}"
          f"{'(dirty)' if metadata['git_dirty'] else ''}")

    print_summary(state, sname)

    if args.once:
        run_once(client, state, sname, log_file, dry_run=args.dry_run)
    else:
        print(f"\n  Watching for forecasts... (Ctrl+C to stop)\n")
        try:
            while True:
                run_once(client, state, sname, log_file,
                         dry_run=args.dry_run)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n  [{sname}] STOPPED")
            print_summary(state, sname)
            save_state(state, state["_state_file"])


if __name__ == "__main__":
    main()
