"""
Backtest: empirical lead-time-matched pricing from iem_forecasts only.

1. Single pass: compute empirical fair value for every contract at every
   NBS signal tick. Store all opportunities.
2. Fast sweep: filter opportunities by (min_edge, max_cost) thresholds.

Usage:
    cd new-model
    python probability.py                        # full sweep
    python probability.py --edge 15 --cost 30    # specific params
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collector.config import STATIONS
from collector.backtest import DISSEMINATION_LAG

DATA_DB = Path(__file__).resolve().parent / "data" / "collector.db"
SERIES_TO_ICAO = {s.kalshi_series: s.icao for s in STATIONS}
SERIES_TO_CITY = {s.kalshi_series: s.city for s in STATIONS}
ICAO_TO_CITY = {s.icao: s.city for s in STATIONS}
NBS_LAG = DISSEMINATION_LAG["NBS"]
_TICKER_RE = re.compile(r"^(KXHIGH\w+)-(\d{2}[A-Z]{3}\d{2})-([TB])([\d.]+)$")

ERROR_LOOKBACK = 10
MAX_LEAD_MISMATCH_HOURS = 3


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


def get_errors_at_lead(nbs_rows, metar_data, station, target_date,
                       current_runtime):
    td = datetime.strptime(target_date, "%Y-%m-%d")
    target_ftime = (td + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    runtime_dt = datetime.fromisoformat(current_runtime.replace("Z", "+00:00"))
    ftime_dt = datetime.fromisoformat(target_ftime.replace("Z", "+00:00"))
    lead = ftime_dt - runtime_dt

    errors = []
    for days_back in range(1, ERROR_LOOKBACK + 15):
        if len(errors) >= ERROR_LOOKBACK:
            break
        past_date = (td - timedelta(days=days_back)).strftime("%Y-%m-%d")
        past_next = (td - timedelta(days=days_back - 1)).strftime("%Y-%m-%d")
        past_ftime = past_next + "T00:00:00Z"
        ideal_rt = (datetime.fromisoformat(
            past_ftime.replace("Z", "+00:00")) - lead)

        best_txn = None
        best_diff = float("inf")
        for rt_str, ft_str, txn in nbs_rows:
            if ft_str != past_ftime or txn is None:
                continue
            rt_dt = datetime.fromisoformat(rt_str.replace("Z", "+00:00"))
            diff = abs((rt_dt - ideal_rt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_txn = txn

        if best_txn is None or best_diff > MAX_LEAD_MISMATCH_HOURS * 3600:
            continue

        obs_temps = metar_data.get(station)
        if not obs_temps: continue
        day_temps = [t for ot, t in obs_temps
                     if ot >= past_date + "T00:00:00Z"
                     and ot < past_next + "T00:00:00Z" and t is not None]
        if not day_temps: continue

        errors.append(max(day_temps) - best_txn)
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


def settle(c, obs_max, t_low):
    if c.ctype == "T" and c.value == t_low:
        return 100 if obs_max < c.value else 0
    elif c.ctype == "T":
        return 100 if obs_max > c.value else 0
    else:
        x = c.value - 0.5
        return 100 if x <= obs_max <= x + 1 else 0


def run(fixed_edge=None, fixed_cost=None, max_pos=5):
    conn = sqlite3.connect(str(DATA_DB))

    # Parse
    all_tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM kalshi_prices").fetchall()]
    contracts = {}; groups = defaultdict(list)
    for t in all_tickers:
        c = parse_ticker(t)
        if c: contracts[t] = c; groups[(c.series, c.target_date)].append(c)
    bounds = {}
    for key, grp in groups.items():
        tv = sorted(c.value for c in grp if c.ctype == "T")
        bounds[key] = tv[0] if tv else 0
    all_td = sorted(set(c.target_date for c in contracts.values()))
    eval_dates = set(all_td[-2:]) if len(all_td) >= 2 else set()
    active = sorted(set(SERIES_TO_ICAO[s] for s, _ in groups
                        if s in SERIES_TO_ICAO))

    # Preload
    print("Loading data...")
    ph = ",".join("?" * len(active))
    nbs_raw = conn.execute(
        f"SELECT station, runtime, ftime, txn FROM iem_forecasts "
        f"WHERE model='NBS' AND station IN ({ph}) "
        f"AND runtime >= '2026-03-20' ORDER BY station, runtime",
        active).fetchall()
    nbs_by_stn = defaultdict(list)
    for stn, rt, ft, txn in nbs_raw:
        nbs_by_stn[stn].append((rt, ft, txn))

    metar_raw = conn.execute(
        f"SELECT station, obs_time, temp_f FROM metar_obs "
        f"WHERE station IN ({ph})", active).fetchall()
    metar_data = defaultdict(list)
    for stn, ot, tf in metar_raw:
        metar_data[stn].append((ot, tf))

    obs_lookup = {}
    for stn in active:
        by_day = defaultdict(list)
        for ot, tf in metar_data.get(stn, []):
            if tf is not None: by_day[ot[:10]].append(tf)
        for d, tl in by_day.items():
            obs_lookup[(stn, d)] = max(tl)

    snapshots = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts FROM kalshi_prices ORDER BY ts").fetchall()]
    price_data = pd.read_sql(
        "SELECT ticker, ts, yes_bid, yes_ask FROM kalshi_prices", conn)
    price_by_ts = {ts: grp.set_index("ticker")
                   for ts, grp in price_data.groupby("ts")}
    conn.close()

    print(f"  {len(snapshots)} snapshots, {len(contracts)} tickers")
    print(f"  {len(all_td)} target dates: {all_td[0]} to {all_td[-1]}")
    print(f"  Eval: {sorted(eval_dates)}")

    # ── SINGLE PASS: compute all opportunities ──────────────────────────
    print(f"\nComputing empirical fair values (single pass)...")
    prev_nbs_rt = {}
    opps = []  # all tradeable opportunities

    for si, ts in enumerate(snapshots):
        as_of = pd.Timestamp(ts)
        cutoff_str = (as_of - NBS_LAG).strftime("%Y-%m-%dT%H:%M:%SZ")

        signal_stations = set()
        for stn in active:
            rows = nbs_by_stn.get(stn)
            if not rows: continue
            latest_rt = None
            for rt_str, _, _ in rows:
                if rt_str <= cutoff_str:
                    if latest_rt is None or rt_str > latest_rt:
                        latest_rt = rt_str
            if latest_rt and (stn not in prev_nbs_rt
                              or latest_rt != prev_nbs_rt[stn]):
                signal_stations.add(stn)
                prev_nbs_rt[stn] = latest_rt

        if not signal_stations:
            continue

        snap = price_by_ts.get(ts)
        if snap is None:
            continue

        for (series, td), grp in groups.items():
            icao = SERIES_TO_ICAO.get(series)
            if not icao or icao not in signal_stations:
                continue

            runtime = prev_nbs_rt[icao]
            td_dt = datetime.strptime(td, "%Y-%m-%d")
            next_00z = (td_dt + timedelta(days=1)).strftime(
                "%Y-%m-%dT00:00:00Z")
            forecast = None
            for rt_str, ft_str, txn in reversed(nbs_by_stn.get(icao, [])):
                if ft_str == next_00z and txn is not None and rt_str <= cutoff_str:
                    forecast = txn
                    runtime = rt_str
                    break
            if forecast is None:
                continue

            errors = get_errors_at_lead(
                nbs_by_stn[icao], metar_data, icao, td, runtime)
            if not errors:
                continue

            t_low = bounds[(series, td)]

            for c in grp:
                if c.ticker not in snap.index: continue
                row = snap.loc[c.ticker]
                bid, ask = row["yes_bid"], row["yes_ask"]
                if pd.isna(bid) or pd.isna(ask): continue
                bid, ask = int(bid), int(ask)
                if bid <= 0 or ask <= 0 or bid >= 97 or ask <= 3:
                    continue

                fv = price_empirical(c, forecast, errors, t_low)
                if fv is None: continue

                obs = obs_lookup.get((icao, c.target_date))
                stl = settle(c, obs, t_low) if obs is not None else None

                opps.append({
                    "ts": ts, "ticker": c.ticker,
                    "city": SERIES_TO_CITY.get(series, series),
                    "target_date": td,
                    "side_buy": "BUY", "side_sell": "SELL",
                    "bid": bid, "ask": ask, "fv": fv,
                    "edge_buy": fv - ask,    # positive = YES underpriced
                    "edge_sell": bid - fv,   # positive = YES overpriced
                    "cost_buy": ask,
                    "cost_sell": 100 - bid,
                    "forecast": forecast, "n_errors": len(errors),
                    "settlement": stl,
                    "is_eval": td in eval_dates,
                })

        if (si + 1) % 500 == 0:
            print(f"  {si+1}/{len(snapshots)} ticks, "
                  f"{len(opps)} opportunities")

    print(f"  Done: {len(opps)} opportunities from "
          f"{len(snapshots)} snapshots\n")

    odf = pd.DataFrame(opps)
    if odf.empty:
        print("No opportunities found.")
        return

    # ── FAST SWEEP ──────────────────────────────────────────────────────
    if fixed_edge is not None and fixed_cost is not None:
        edges = [fixed_edge]
        costs = [fixed_cost]
    else:
        edges = [5, 8, 10, 15, 20, 25, 30, 40]
        costs = [10, 15, 20, 25, 30, 40, 50]

    def sweep_one(min_edge, max_cost):
        pos = defaultdict(int)
        total_pnl = 0; total_cap = 0; n_set = 0; n_wins = 0
        pnl_tr = 0; pnl_ev = 0; cap_tr = 0; cap_ev = 0
        n_tr = 0; n_ev = 0; n_buys = 0; n_sells = 0

        for _, o in odf.iterrows():
            tk = o["ticker"]
            # BUY YES
            if (o["edge_buy"] >= min_edge and o["cost_buy"] <= max_cost
                    and pos[tk] < max_pos):
                pos[tk] += 1
                cost = o["cost_buy"]
                stl = o["settlement"]
                if pd.notna(stl):
                    pnl = stl - o["ask"]
                    total_pnl += pnl; total_cap += cost; n_set += 1
                    if pnl > 0: n_wins += 1
                    if o["is_eval"]:
                        pnl_ev += pnl; cap_ev += cost; n_ev += 1
                    else:
                        pnl_tr += pnl; cap_tr += cost; n_tr += 1
                n_buys += 1
            # BUY NO
            elif (o["edge_sell"] >= min_edge
                  and o["cost_sell"] <= max_cost
                  and pos[tk] > -max_pos):
                pos[tk] -= 1
                cost = o["cost_sell"]
                stl = o["settlement"]
                if pd.notna(stl):
                    pnl = o["bid"] - stl
                    total_pnl += pnl; total_cap += cost; n_set += 1
                    if pnl > 0: n_wins += 1
                    if o["is_eval"]:
                        pnl_ev += pnl; cap_ev += cost; n_ev += 1
                    else:
                        pnl_tr += pnl; cap_tr += cost; n_tr += 1
                n_sells += 1

        fills = n_buys + n_sells
        return {
            "fills": fills, "settled": n_set,
            "pnl": total_pnl, "cap": total_cap,
            "roi": total_pnl / total_cap * 100 if total_cap > 0 else 0,
            "win": n_wins / n_set * 100 if n_set > 0 else 0,
            "pnl_tr": pnl_tr, "cap_tr": cap_tr, "n_tr": n_tr,
            "roi_tr": pnl_tr / cap_tr * 100 if cap_tr > 0 else 0,
            "pnl_ev": pnl_ev, "cap_ev": cap_ev, "n_ev": n_ev,
            "roi_ev": pnl_ev / cap_ev * 100 if cap_ev > 0 else 0,
            "buys": n_buys, "sells": n_sells,
        }

    print(f"{'=' * 130}")
    print(f"  PARAMETER SWEEP  (empirical lead-matched pricing)")
    print(f"{'=' * 130}")

    grid = {}
    best_pnl = -9999; best = (10, 30)

    # PnL grid
    print(f"\n  TOTAL PnL / fills / ROI:")
    print(f"  {'':>10}", end="")
    for c in costs:
        print(f"  {'cost<='+str(c)+'c':>17}", end="")
    print()

    for e in edges:
        print(f"  edge>={e:>2}c", end="")
        for c in costs:
            r = sweep_one(e, c)
            grid[(e, c)] = r
            if r["pnl"] > best_pnl and r["settled"] >= 5:
                best_pnl = r["pnl"]; best = (e, c)
            if r["settled"] > 0:
                print(f"  {r['pnl']:>+5.0f}c {r['settled']:>3}f"
                      f" {r['roi']:>+4.0f}%", end="")
            else:
                print(f"     {'--':>5} {r['settled']:>3}f"
                      f"     ", end="")
        print()

    # Eval grid
    print(f"\n  EVAL-ONLY PnL:")
    print(f"  {'':>10}", end="")
    for c in costs:
        print(f"  {'cost<='+str(c)+'c':>17}", end="")
    print()

    for e in edges:
        print(f"  edge>={e:>2}c", end="")
        for c in costs:
            r = grid[(e, c)]
            if r["n_ev"] > 0:
                print(f"  {r['pnl_ev']:>+5.0f}c {r['n_ev']:>3}f"
                      f" {r['roi_ev']:>+4.0f}%", end="")
            else:
                print(f"     {'--':>5} {r['n_ev']:>3}f"
                      f"     ", end="")
        print()

    # Best detail
    be, bc = best
    r = sweep_one(be, bc)

    print(f"\n{'=' * 130}")
    print(f"  BEST: edge >= {be}c, cost <= {bc}c  "
          f"(max PnL with >= 5 settled)")
    print(f"{'=' * 130}")
    print(f"  Fills:    {r['fills']} ({r['buys']} BUY YES / "
          f"{r['sells']} BUY NO)")
    print(f"  Settled:  {r['settled']}   Win: {r['win']:.0f}%")
    print(f"  PnL:      {r['pnl']:+.0f}c (${r['pnl']/100:+,.2f})  "
          f"Cap: {r['cap']:.0f}c  ROI: {r['roi']:+.1f}%")
    if r["n_tr"]:
        print(f"  TRAIN:    {r['pnl_tr']:+.0f}c  "
              f"ROI: {r['roi_tr']:+.1f}%  ({r['n_tr']} fills)")
    if r["n_ev"]:
        print(f"  EVAL:     {r['pnl_ev']:+.0f}c  "
              f"ROI: {r['roi_ev']:+.1f}%  ({r['n_ev']} fills)")

    # Run best with trade detail
    pos = defaultdict(int)
    trade_detail = []
    for _, o in odf.iterrows():
        tk = o["ticker"]
        if (o["edge_buy"] >= be and o["cost_buy"] <= bc and pos[tk] < max_pos):
            pos[tk] += 1
            if pd.notna(o["settlement"]):
                pnl = o["settlement"] - o["ask"]
                trade_detail.append({**o, "side": "BUY", "cost": o["cost_buy"],
                                     "edge": o["edge_buy"], "pnl": pnl})
        elif (o["edge_sell"] >= be and o["cost_sell"] <= bc and pos[tk] > -max_pos):
            pos[tk] -= 1
            if pd.notna(o["settlement"]):
                pnl = o["bid"] - o["settlement"]
                trade_detail.append({**o, "side": "SELL", "cost": o["cost_sell"],
                                     "edge": o["edge_sell"], "pnl": pnl})

    if trade_detail:
        tpdf = pd.DataFrame(trade_detail)

        for side, lbl in [("BUY", "BUY YES"), ("SELL", "BUY NO")]:
            s = tpdf[tpdf["side"] == side]
            if s.empty: continue
            print(f"\n  {lbl}: {len(s)} fills  "
                  f"PnL={s['pnl'].sum():+.0f}c  "
                  f"ROI={s['pnl'].sum()/s['cost'].sum()*100:+.0f}%  "
                  f"win={(s['pnl']>0).sum()}/{len(s)}")

        city_s = (tpdf.groupby("city").agg(
            f=("pnl","count"), p=("pnl","sum"), c=("cost","sum"))
            .sort_values("p", ascending=False))
        city_s["roi"] = city_s["p"] / city_s["c"] * 100
        print(f"\n  {'City':<15} {'Fills':>6} {'PnL':>8} {'ROI':>8}")
        print(f"  {'-' * 42}")
        for city, row in city_s.iterrows():
            print(f"  {city:<15} {int(row['f']):>6} "
                  f"{row['p']:>+7.0f}c {row['roi']:>+7.0f}%")

        print(f"\n  ALL TRADES ({len(tpdf)}):")
        print(f"  {'Time':<9} {'Ticker':<28} {'Side':<5} "
              f"{'Cost':>5} {'FV':>5} {'Edge':>5} {'Fcst':>5} "
              f"{'Obs':>5} {'PnL':>6} {'N':>3}")
        print(f"  {'-' * 100}")
        for _, t in tpdf.sort_values("ts").iterrows():
            obs_v = t["settlement"]  # actually this is stl value
            obs_str = "YES" if obs_v == 100 else " NO"
            print(f"  {t['ts'][11:19]:<9} {t['ticker']:<28} "
                  f"{t['side']:<5} {t['cost']:>4}c "
                  f"{t['fv']:>4.0f}c {t['edge']:>+4.0f}c "
                  f"{t['forecast']:>5.0f} {obs_str:>5} "
                  f"{t['pnl']:>+5.0f}c {t['n_errors']:>3}")

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--edge", type=int, default=None)
    p.add_argument("--cost", type=int, default=None)
    p.add_argument("--max-pos", type=int, default=5)
    args = p.parse_args()
    import logging; logging.basicConfig(level=logging.WARNING)
    run(fixed_edge=args.edge, fixed_cost=args.cost, max_pos=args.max_pos)
