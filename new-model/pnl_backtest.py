"""
PnL backtest using real collected data (Kalshi prices, IEM forecasts, METAR).

For each contract in the kalshi_prices table:
  1. Parse the ticker (series, expiry date, type, threshold)
  2. Get real market bid/ask from the latest snapshot
  3. Build model distribution using forecasts + METAR available at that time
  4. Price the contract with the blend model
  5. Trade when |model_price - market_mid| > min_edge
  6. Settle on observed daily high — or mark OPEN if pending

As the collector runs and more contracts settle, the backtest grows.

Usage:
    cd new-model
    python pnl_backtest.py                    # all contracts in DB
    python pnl_backtest.py --edge 5           # min edge to trade (cents)
    python pnl_backtest.py --csv trades.csv   # export all trades
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from collector.config import DB_PATH, STATIONS
from collector.backtest import (
    get_latest_forecast_at,
    get_metar_running_max_at,
)
from probability import (
    parse_ticker,
    Contract,
    ForecastDist,
    price_group,
    SERIES_TO_ICAO,
    SERIES_TO_CITY,
)
from backtest import ICAO_TO_IEM, fetch_observed_highs

log = logging.getLogger(__name__)


# ── Point-in-time forecast builder ───────────────────────────────────────────


def build_forecast_at(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    as_of: str,
) -> ForecastDist | None:
    """
    Build a blended Normal(mu, sigma) using only data available at *as_of*.

    Same blending logic as probability.py but with point-in-time constraint
    via collector.backtest lag-adjusted queries.
    """
    td = datetime.strptime(target_date, "%Y-%m-%d")
    next_00z = (td + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    day_12z = target_date + "T12:00:00Z"
    next_03z = (td + timedelta(days=1)).strftime("%Y-%m-%dT03:00:00Z")

    sources: dict = {}
    vals: list[float] = []
    wts: list[float] = []
    nbm_sd: float | None = None

    # NBM (NBS)
    nbm_df = get_latest_forecast_at(conn, station, "NBS", as_of)
    if not nbm_df.empty:
        hit = nbm_df[nbm_df["ftime"] == next_00z]
        if not hit.empty and pd.notna(hit.iloc[0].get("txn")):
            nbm_max = float(hit.iloc[0]["txn"])
            sources["nbm"] = nbm_max
            vals.append(nbm_max)
            wts.append(2.0)
            if pd.notna(hit.iloc[0].get("xnd")):
                nbm_sd = float(hit.iloc[0]["xnd"])

    # GFS MOS
    gfs_df = get_latest_forecast_at(conn, station, "GFS", as_of)
    if not gfs_df.empty:
        hit = gfs_df[gfs_df["ftime"] == next_00z]
        if not hit.empty and pd.notna(hit.iloc[0].get("n_x")):
            gfs_max = float(hit.iloc[0]["n_x"])
            sources["gfs"] = gfs_max
            vals.append(gfs_max)
            wts.append(1.0)

    # LAMP (LAV) — max of hourly temps in 12Z–03Z+1 window
    lav_df = get_latest_forecast_at(conn, station, "LAV", as_of)
    if not lav_df.empty:
        win = lav_df[(lav_df["ftime"] >= day_12z) & (lav_df["ftime"] <= next_03z)]
        tmps = win["tmp"].dropna()
        if not tmps.empty:
            lamp_max = float(tmps.max())
            sources["lamp"] = lamp_max
            vals.append(lamp_max)
            wts.append(1.0)

    if not vals:
        return None

    mu = float(np.average(vals, weights=wts))
    sigma = max(nbm_sd, 1.5) if nbm_sd and nbm_sd > 0 else 4.0

    # METAR running max — shift mu up if obs already exceed forecast
    metar_max = get_metar_running_max_at(conn, station, target_date, as_of)
    if metar_max is not None:
        sources["obs"] = metar_max
        mu = max(mu, metar_max)

    return ForecastDist(station, target_date, mu, sigma, sources)


# ── Settlement ───────────────────────────────────────────────────────────────


def settle_contract(
    contract: Contract, obs_max: float, t_low: float, t_high: float
) -> int:
    """Return 100 if YES settled, 0 if NO."""
    if contract.ctype == "T":
        if contract.value == t_low:
            # Under: YES if Tmax < threshold
            return 100 if obs_max < contract.value else 0
        else:
            # Over: YES if Tmax > threshold
            return 100 if obs_max > contract.value else 0
    else:
        # Bracket B(x.5): YES if x <= Tmax <= x+1
        x = contract.value - 0.5
        return 100 if x <= obs_max <= x + 1 else 0


# ── Core backtest ────────────────────────────────────────────────────────────


def run_pnl_backtest(min_edge: int = 5) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))

    # 1. Load all Kalshi price snapshots
    kalshi_df = pd.read_sql(
        "SELECT * FROM kalshi_prices ORDER BY ticker, ts", conn
    )
    if kalshi_df.empty:
        print("No Kalshi price data in DB.")
        conn.close()
        return pd.DataFrame()

    # 2. Parse tickers, group by (series, target_date)
    contracts: dict[str, Contract] = {}
    groups: dict[tuple, list[Contract]] = defaultdict(list)
    for ticker in kalshi_df["ticker"].unique():
        c = parse_ticker(ticker)
        if c:
            contracts[ticker] = c
            groups[(c.series, c.target_date)].append(c)

    # 3. Fetch observed highs for all contract dates (downloads if missing)
    contract_dates = sorted(
        set(td for _, td in groups.keys())
    )
    if contract_dates:
        stations_with_iem = [
            s for s in STATIONS if ICAO_TO_IEM.get(s.icao) is not None
        ]
        print("Checking settlement data (observed highs)...")
        n_fetched = fetch_observed_highs(
            conn, stations_with_iem, contract_dates[0], contract_dates[-1]
        )
        if n_fetched:
            print(f"  Downloaded {n_fetched} observation-days.\n")

    obs_df = pd.read_sql(
        "SELECT station, date, max_tmpf FROM observed_highs", conn
    )
    obs_lookup: dict[tuple[str, str], float] = {
        (r["station"], r["date"]): r["max_tmpf"]
        for _, r in obs_df.iterrows()
    }

    # 4. Process each (series, target_date) group
    trades: list[dict] = []

    for (series, target_date), group_contracts in sorted(groups.items()):
        icao = SERIES_TO_ICAO.get(series)
        city = SERIES_TO_CITY.get(series, series)
        if not icao:
            continue

        # Get the LATEST Kalshi snapshot for these contracts
        tickers = [c.ticker for c in group_contracts]
        group_prices = kalshi_df[kalshi_df["ticker"].isin(tickers)]
        latest_ts = group_prices["ts"].max()

        snap = (
            group_prices[group_prices["ts"] == latest_ts]
            .drop_duplicates(subset="ticker")
            .set_index("ticker")
        )

        # Skip groups where all contracts have extreme prices (already settled)
        mids = []
        for c in group_contracts:
            if c.ticker in snap.index:
                row = snap.loc[c.ticker]
                bid = row.get("yes_bid", 0) or 0
                ask = row.get("yes_ask", 0) or 0
                mids.append((bid + ask) / 2)
        if mids and all(m < 3 or m > 97 for m in mids):
            continue  # already settled, no tradeable prices

        # Build model distribution at snapshot time using real forecasts
        dist = build_forecast_at(conn, icao, target_date, latest_ts)
        if not dist:
            continue

        # Price contracts using the model
        model_rows = price_group(group_contracts, dist)
        model_map = {r["ticker"]: r for r in model_rows}

        # Determine t_low / t_high for settlement
        t_vals = sorted(c.value for c in group_contracts if c.ctype == "T")
        t_low = t_vals[0] if t_vals else None
        t_high = t_vals[-1] if len(t_vals) > 1 else None

        # Official observed high for settlement (only source we trust)
        obs = obs_lookup.get((icao, target_date))

        for c in group_contracts:
            if c.ticker not in model_map or c.ticker not in snap.index:
                continue

            mr = model_map[c.ticker]
            pr = snap.loc[c.ticker]

            bid = int(pr.get("yes_bid", 0) or 0)
            ask = int(pr.get("yes_ask", 0) or 0)
            mid = (bid + ask) / 2.0

            # Skip contracts with extreme prices (already settled or illiquid)
            if mid < 3 or mid > 97:
                continue

            model_cents = round(mr["model_p"] * 100)
            edge = model_cents - round(mid)

            if abs(edge) < min_edge:
                continue

            # Determine trade direction and entry price
            if edge > 0:
                # BUY YES at ask — model thinks more likely than market
                side = "BUY"
                entry = ask
            else:
                # SELL (BUY NO) at no_ask — model thinks less likely
                side = "SELL"
                entry = 100 - bid

            # Settlement — only use official observed highs
            if obs is not None and t_low is not None and t_high is not None:
                stl = settle_contract(c, obs, t_low, t_high)
                if side == "BUY":
                    pnl = stl - entry
                else:
                    pnl = (100 - stl) - entry
                status = "SETTLED"
            else:
                stl = None
                pnl = None
                status = "OPEN"

            trades.append(
                {
                    "date": target_date,
                    "city": city,
                    "station": icao,
                    "ticker": c.ticker,
                    "type": mr["type"],
                    "threshold": c.value,
                    "side": side,
                    "model_c": model_cents,
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid),
                    "edge": edge,
                    "entry": entry,
                    "obs_high": obs,
                    "settlement": stl,
                    "pnl": pnl,
                    "status": status,
                    "mu": round(dist.mu, 1),
                    "sigma": round(dist.sigma, 1),
                    "sources": dist.sources,
                    "snap_time": latest_ts,
                }
            )

    conn.close()
    return pd.DataFrame(trades)


# ── Display ──────────────────────────────────────────────────────────────────


def print_results(df: pd.DataFrame):
    if df.empty:
        print("\nNo trades generated.")
        return

    n = len(df)
    settled = df[df["status"] == "SETTLED"]
    open_pos = df[df["status"] == "OPEN"]

    print(f"\n{'=' * 110}")
    print(f"  PnL BACKTEST — Real Collected Data")
    print(f"  Data window: {df['snap_time'].min()[:16]} to {df['snap_time'].max()[:16]} UTC")
    print(
        f"  {n} trades  |  {len(settled)} settled  |  {len(open_pos)} open"
    )
    print(f"{'=' * 110}")

    # ── Settled trades summary ────────────────────────────────────────────
    if not settled.empty:
        total = settled["pnl"].sum()
        avg = settled["pnl"].mean()
        wins = (settled["pnl"] > 0).sum()
        losses = (settled["pnl"] <= 0).sum()
        wr = wins / len(settled) * 100
        print(f"\n  SETTLED TRADES: {len(settled)}")
        print(
            f"  Total PnL:      {total:>+.0f}c  (${total / 100:>+,.2f} per contract)"
        )
        print(f"  Avg PnL/trade:  {avg:>+.1f}c")
        print(f"  Win Rate:       {wr:.0f}%  ({wins}W / {losses}L)")

    # ── Open positions summary ────────────────────────────────────────────
    if not open_pos.empty:
        buys = open_pos[open_pos["side"] == "BUY"]
        sells = open_pos[open_pos["side"] == "SELL"]
        buy_risk = buys["entry"].sum()
        sell_risk = sells["entry"].sum()
        print(f"\n  OPEN POSITIONS: {len(open_pos)}  (awaiting settlement)")
        print(
            f"  {len(buys)} BUYs (cost {buy_risk:.0f}c)  |"
            f"  {len(sells)} SELLs (cost {sell_risk:.0f}c)"
        )
        print(f"  Total capital at risk: {buy_risk + sell_risk:.0f}c (${(buy_risk + sell_risk) / 100:.2f})")

    # ── Forecast distributions used ───────────────────────────────────────
    seen = set()
    dist_rows = []
    for _, r in df.iterrows():
        key = (r["city"], r["date"])
        if key not in seen:
            seen.add(key)
            dist_rows.append(r)

    print(f"\n  MODEL DISTRIBUTIONS")
    print(
        f"  {'City':<18} {'Date':<12} {'mu':>6} {'sigma':>6}  Sources"
    )
    print(f"  {'-' * 75}")
    for r in sorted(dist_rows, key=lambda x: (x["date"], x["city"])):
        src = r["sources"]
        parts = [f"{k}={v:.0f}" for k, v in src.items()]
        print(
            f"  {r['city']:<18} {r['date']:<12}"
            f" {r['mu']:>5.1f}F {r['sigma']:>5.1f}F  {', '.join(parts)}"
        )

    # ── Full trade detail ─────────────────────────────────────────────────
    print(f"\n  ALL TRADES ({n})")
    print(
        f"  {'Ticker':<36} {'Side':<5} {'Model':>6} {'Bid':>5}"
        f" {'Ask':>5} {'Mid':>5} {'Edge':>5} {'Entry':>6}"
        f" {'Obs':>6} {'Sttl':>5} {'PnL':>7} {'Status':<8}"
    )
    print(f"  {'-' * 110}")

    for _, r in df.sort_values(["date", "city", "threshold"]).iterrows():
        obs_val = r.get("obs_high")
        obs_str = f"{obs_val:.0f}" if obs_val is not None and pd.notna(obs_val) else "   -"
        stl_val = r.get("settlement")
        stl_str = f"{int(stl_val)}" if stl_val is not None and pd.notna(stl_val) else "  -"
        pnl_val = r.get("pnl")
        pnl_str = f"{pnl_val:>+.0f}c" if pnl_val is not None and pd.notna(pnl_val) else "   -"

        print(
            f"  {r['ticker']:<36} {r['side']:<5} {r['model_c']:>5}c"
            f" {r['bid']:>4} {r['ask']:>5} {r['mid']:>4}"
            f" {r['edge']:>+4}c {r['entry']:>5}c"
            f" {obs_str:>6} {stl_str:>5} {pnl_str:>7} {r['status']:<8}"
        )

    # ── By city ───────────────────────────────────────────────────────────
    if not settled.empty:
        print(f"\n  SETTLED PnL BY CITY")
        city_pnl = (
            settled.groupby("city")
            .agg(
                trades=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                win_rate=("pnl", lambda x: (x > 0).mean() * 100),
            )
            .sort_values("total_pnl", ascending=False)
        )
        print(f"  {'City':<18} {'Trades':>7} {'PnL':>8} {'Win%':>6}")
        print(f"  {'-' * 42}")
        for city, r in city_pnl.iterrows():
            print(
                f"  {city:<18} {int(r['trades']):>7}"
                f" {r['total_pnl']:>+7.0f}c {r['win_rate']:>5.0f}%"
            )

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PnL backtest from real collected data"
    )
    parser.add_argument(
        "--edge",
        type=int,
        default=5,
        help="Min |edge| to trade, in cents (default 5)",
    )
    parser.add_argument("--csv", help="Export all trades to CSV")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    results = run_pnl_backtest(min_edge=args.edge)

    if results.empty:
        print(
            "No trades generated. "
            "Run the collector longer to accumulate Kalshi price data."
        )
        sys.exit(1)

    print_results(results)

    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"  Exported {len(results)} trades to {args.csv}")


if __name__ == "__main__":
    main()
