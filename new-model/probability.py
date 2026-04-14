"""
Backtest: empirical pricing at fixed NBS runtimes (01Z, 07Z, 13Z, 19Z).

Since error distributions are stable across lead times, we pool all
runtimes per station into one error distribution. Trade only when a new
NBS run arrives at one of the 4 bulk runtimes.

NY and PHL errors are correlated (r=0.65) — shared position limit.

Sweeps: min_edge, max_cost, max_pos, and NE_combined_limit.

Usage:
    cd new-model
    python probability.py
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

# Correlated pair — shared position limit
NE_PAIR = {"KJFK", "KPHL"}


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


# ── Build pooled error distributions per station ────────────────────────

def build_station_errors(conn, stations):
    """
    For each station, pool ALL NBS forecast errors across all runtimes.
    Returns {station: [errors]}.
    """
    result = {}
    for stn in stations:
        obs = dict(conn.execute(
            "SELECT date, max_tmpf FROM observed_highs WHERE station=?",
            (stn,)).fetchall())
        if not obs:
            # Fallback to METAR
            rows = conn.execute(
                "SELECT date(obs_time), MAX(temp_f) FROM metar_obs "
                "WHERE station=? AND temp_f IS NOT NULL "
                "GROUP BY date(obs_time)", (stn,)).fetchall()
            obs = {r[0]: r[1] for r in rows if r[1] is not None}

        nbs = conn.execute(
            "SELECT runtime, ftime, txn FROM iem_forecasts "
            "WHERE model='NBS' AND station=? AND txn IS NOT NULL "
            "AND substr(ftime,12,8)='00:00:00'",
            (stn,)).fetchall()

        # Group by target_date (ftime - 1 day)
        by_target = defaultdict(list)
        for rt, ft, txn in nbs:
            target = (datetime.fromisoformat(ft.replace("Z", "+00:00"))
                      - timedelta(days=1)).strftime("%Y-%m-%d")
            by_target[target].append((rt, txn))

        errors = []
        for target, forecasts in by_target.items():
            if target not in obs:
                continue
            actual = obs[target]
            # Use the LATEST forecast for this target date
            latest_rt, latest_txn = max(forecasts, key=lambda x: x[0])
            errors.append(actual - latest_txn)

        result[stn] = errors
    return result


# ── Empirical pricing ───────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────

def run():
    conn = sqlite3.connect(str(DATA_DB))

    # Parse contracts
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

    # Build pooled error distributions
    stn_errors = build_station_errors(conn, active)
    for stn in active:
        n = len(stn_errors.get(stn, []))
        print(f"  {ICAO_TO_CITY.get(stn, stn):>15}: {n} errors")

    # NBS data for signal detection and forecasts
    nbs_raw = conn.execute(
        f"SELECT station, runtime, ftime, txn FROM iem_forecasts "
        f"WHERE model='NBS' AND station IN ({ph}) "
        f"ORDER BY station, runtime", active).fetchall()
    nbs_by_stn = defaultdict(list)
    for stn, rt, ft, txn in nbs_raw:
        nbs_by_stn[stn].append((rt, ft, txn))

    # Index by (station, ftime) for fast forecast lookup
    nbs_idx = defaultdict(dict)  # {(stn, ftime): {runtime: txn}}
    for stn, rt, ft, txn in nbs_raw:
        if txn is not None:
            nbs_idx[(stn, ft)][rt] = txn

    # Settlement data
    obs_lookup = {}
    try:
        for stn, d, t in conn.execute(
                f"SELECT station, date, max_tmpf FROM observed_highs "
                f"WHERE station IN ({ph})", active).fetchall():
            if t is not None: obs_lookup[(stn, d)] = t
    except sqlite3.OperationalError:
        pass
    metar_raw = conn.execute(
        f"SELECT station, obs_time, temp_f FROM metar_obs "
        f"WHERE station IN ({ph})", active).fetchall()
    for stn, ot, tf in metar_raw:
        if tf is not None:
            key = (stn, ot[:10])
            obs_lookup[key] = max(obs_lookup.get(key, -999), tf)

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
    print(f"\nComputing opportunities...")
    prev_nbs_rt = {}
    opps = []

    for si, ts in enumerate(snapshots):
        as_of = pd.Timestamp(ts)
        cutoff_str = (as_of - NBS_LAG).strftime("%Y-%m-%dT%H:%M:%SZ")
        today = ts[:10]

        # Detect new NBS runs
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
            if td <= today:
                continue
            icao = SERIES_TO_ICAO.get(series)
            if not icao or icao not in signal_stations:
                continue

            # Get latest NBS forecast
            td_dt = datetime.strptime(td, "%Y-%m-%d")
            next_00z = (td_dt + timedelta(days=1)).strftime(
                "%Y-%m-%dT00:00:00Z")
            runtimes = nbs_idx.get((icao, next_00z), {})
            valid_rts = {rt: txn for rt, txn in runtimes.items()
                         if rt <= cutoff_str}
            if not valid_rts:
                continue
            best_rt = max(valid_rts.keys())
            forecast = valid_rts[best_rt]

            errors = stn_errors.get(icao, [])
            if len(errors) < 30:
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
                    "station": icao,
                    "target_date": td,
                    "bid": bid, "ask": ask, "fv": fv,
                    "edge_buy": fv - ask,
                    "edge_sell": bid - fv,
                    "cost_buy": ask,
                    "cost_sell": 100 - bid,
                    "forecast": forecast,
                    "settlement": stl,
                    "is_eval": td in eval_dates,
                })

        if (si + 1) % 1000 == 0:
            print(f"  {si+1}/{len(snapshots)} ticks, "
                  f"{len(opps)} opportunities")

    print(f"  Done: {len(opps)} opportunities\n")
    odf = pd.DataFrame(opps)
    if odf.empty:
        print("No opportunities."); return

    # ── SWEEP ────────────────────────────────────────────────────────────
    edges = [5, 8, 10, 15, 20, 25, 30]
    costs = [5, 8, 10, 15, 20, 25, 30]
    positions = [3, 5, 10]
    ne_limits = [3, 5, 8, 10]  # combined NY+PHL limit

    print(f"{'=' * 100}")
    print(f"  HYPERPARAMETER SEARCH")
    print(f"  Params: min_edge x max_cost x max_pos x ne_limit")
    print(f"  NY-PHL correlated (r=0.65) — shared NE position limit")
    print(f"{'=' * 100}")

    best_pnl = -9999
    best_params = {}
    all_results = []

    for max_pos in positions:
        for ne_limit in ne_limits:
            if ne_limit > max_pos * 2:
                continue
            for min_edge in edges:
                for max_cost in costs:
                    pos = defaultdict(int)
                    total_pnl = 0; total_cap = 0
                    n_set = 0; n_wins = 0
                    pnl_tr = 0; pnl_ev = 0
                    cap_tr = 0; cap_ev = 0
                    n_tr = 0; n_ev = 0

                    for _, o in odf.iterrows():
                        tk = o["ticker"]
                        stn = o["station"]
                        stl = o["settlement"]

                        # NE combined limit
                        ne_pos = sum(abs(pos[k]) for k in pos
                                     if any(s in k for s in NE_PAIR))

                        did_trade = False

                        if (o["edge_buy"] >= min_edge
                                and o["cost_buy"] <= max_cost
                                and pos[tk] < max_pos):
                            if stn in NE_PAIR and ne_pos >= ne_limit:
                                pass
                            else:
                                pos[tk] += 1
                                did_trade = True
                                if pd.notna(stl):
                                    pnl = stl - o["ask"]
                                    total_pnl += pnl
                                    total_cap += o["cost_buy"]
                                    n_set += 1
                                    if pnl > 0: n_wins += 1
                                    if o["is_eval"]:
                                        pnl_ev += pnl; cap_ev += o["cost_buy"]; n_ev += 1
                                    else:
                                        pnl_tr += pnl; cap_tr += o["cost_buy"]; n_tr += 1

                        if not did_trade and (o["edge_sell"] >= min_edge
                                and o["cost_sell"] <= max_cost
                                and pos[tk] > -max_pos):
                            if stn in NE_PAIR and ne_pos >= ne_limit:
                                pass
                            else:
                                pos[tk] -= 1
                                if pd.notna(stl):
                                    pnl = o["bid"] - stl
                                    total_pnl += pnl
                                    total_cap += o["cost_sell"]
                                    n_set += 1
                                    if pnl > 0: n_wins += 1
                                    if o["is_eval"]:
                                        pnl_ev += pnl; cap_ev += o["cost_sell"]; n_ev += 1
                                    else:
                                        pnl_tr += pnl; cap_tr += o["cost_sell"]; n_tr += 1

                    if n_set < 5:
                        continue

                    roi = total_pnl / total_cap * 100 if total_cap > 0 else 0
                    roi_ev = pnl_ev / cap_ev * 100 if cap_ev > 0 else 0

                    r = {
                        "edge": min_edge, "cost": max_cost,
                        "max_pos": max_pos, "ne_limit": ne_limit,
                        "pnl": total_pnl, "cap": total_cap,
                        "roi": roi, "settled": n_set,
                        "win": n_wins / n_set * 100 if n_set else 0,
                        "pnl_tr": pnl_tr, "pnl_ev": pnl_ev,
                        "roi_ev": roi_ev, "n_ev": n_ev,
                    }
                    all_results.append(r)

                    if total_pnl > best_pnl:
                        best_pnl = total_pnl
                        best_params = r

    rdf = pd.DataFrame(all_results)

    # Show top 20 by total PnL
    top_pnl = rdf.nlargest(20, "pnl")
    print(f"\n  TOP 20 BY TOTAL PnL:")
    print(f"  {'Edge':>5} {'Cost':>5} {'Pos':>4} {'NE':>3} "
          f"{'Fills':>6} {'PnL':>8} {'ROI':>6} {'Win%':>5} "
          f"{'EvPnL':>7} {'EvROI':>6} {'EvN':>4}")
    print(f"  {'-' * 70}")
    for _, r in top_pnl.iterrows():
        print(f"  {r['edge']:>4.0f}c {r['cost']:>4.0f}c "
              f"{r['max_pos']:>4.0f} {r['ne_limit']:>3.0f} "
              f"{r['settled']:>6.0f} {r['pnl']:>+7.0f}c "
              f"{r['roi']:>+5.0f}% {r['win']:>4.0f}% "
              f"{r['pnl_ev']:>+6.0f}c {r['roi_ev']:>+5.0f}% "
              f"{r['n_ev']:>4.0f}")

    # Show top 20 by eval PnL (out-of-sample)
    eval_results = rdf[rdf["n_ev"] >= 3]
    if not eval_results.empty:
        top_ev = eval_results.nlargest(20, "pnl_ev")
        print(f"\n  TOP 20 BY EVAL PnL (out-of-sample):")
        print(f"  {'Edge':>5} {'Cost':>5} {'Pos':>4} {'NE':>3} "
              f"{'Fills':>6} {'PnL':>8} {'ROI':>6} {'Win%':>5} "
              f"{'EvPnL':>7} {'EvROI':>6} {'EvN':>4}")
        print(f"  {'-' * 70}")
        for _, r in top_ev.iterrows():
            print(f"  {r['edge']:>4.0f}c {r['cost']:>4.0f}c "
                  f"{r['max_pos']:>4.0f} {r['ne_limit']:>3.0f} "
                  f"{r['settled']:>6.0f} {r['pnl']:>+7.0f}c "
                  f"{r['roi']:>+5.0f}% {r['win']:>4.0f}% "
                  f"{r['pnl_ev']:>+6.0f}c {r['roi_ev']:>+5.0f}% "
                  f"{r['n_ev']:>4.0f}")

    # Best balanced: top eval PnL with train PnL > 0
    balanced = rdf[(rdf["pnl_tr"] > 0) & (rdf["n_ev"] >= 3)]
    if not balanced.empty:
        top_bal = balanced.nlargest(10, "pnl_ev")
        print(f"\n  BEST BALANCED (train profitable + best eval):")
        print(f"  {'Edge':>5} {'Cost':>5} {'Pos':>4} {'NE':>3} "
              f"{'Fills':>6} {'TrPnL':>8} {'TrROI':>6} "
              f"{'EvPnL':>7} {'EvROI':>6} {'EvN':>4}")
        print(f"  {'-' * 70}")
        for _, r in top_bal.iterrows():
            tr_roi = r["pnl_tr"] / (r["cap"] - (r.get("cap_ev", 0) or 0)) * 100 if r["cap"] > 0 else 0
            print(f"  {r['edge']:>4.0f}c {r['cost']:>4.0f}c "
                  f"{r['max_pos']:>4.0f} {r['ne_limit']:>3.0f} "
                  f"{r['settled']:>6.0f} {r['pnl_tr']:>+7.0f}c "
                  f"{r['roi']:>+5.0f}% "
                  f"{r['pnl_ev']:>+6.0f}c {r['roi_ev']:>+5.0f}% "
                  f"{r['n_ev']:>4.0f}")

    # Show the best params
    bp = best_params
    print(f"\n{'=' * 100}")
    print(f"  BEST BY TOTAL PnL:")
    print(f"  edge >= {bp['edge']}c, cost <= {bp['cost']}c, "
          f"max_pos = {bp['max_pos']}, ne_limit = {bp['ne_limit']}")
    print(f"  PnL = {bp['pnl']:+.0f}c (${bp['pnl']/100:+,.2f})  "
          f"ROI = {bp['roi']:+.0f}%  Win = {bp['win']:.0f}%  "
          f"Fills = {bp['settled']}")
    print(f"  Eval: {bp['pnl_ev']:+.0f}c  ROI: {bp['roi_ev']:+.0f}%  "
          f"N: {bp['n_ev']}")
    print(f"{'=' * 100}")
    print()


if __name__ == "__main__":
    import logging; logging.basicConfig(level=logging.WARNING)
    run()
