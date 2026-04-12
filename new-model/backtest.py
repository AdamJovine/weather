"""
Backtest the probability engine against historical IEM forecast archives.

Uses point-in-time data access: for each historical date, only forecasts
that were actually available at the decision time are used (with proper
dissemination lags).  Observed daily highs are fetched from IEM ASOS and
forecast history is backfilled from IEM's bulk archive — both cached in the
DB so they only download once.

Usage:
    cd new-model
    python backtest.py                    # full backtest, all stations
    python backtest.py --days 90          # last 90 days only
    python backtest.py --station KMDW     # single station
    python backtest.py --decision-hour 15 # change bet time (UTC, default 15)
"""

from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import sys
import time as _time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from collector.config import DB_PATH, STATIONS, ICAO_TO_STATION
from collector.backtest import DISSEMINATION_LAG
from collector.db import upsert_df
from collector.iem import fetch_history

log = logging.getLogger(__name__)

# ── IEM ASOS network lookup ─────────────────────────────────────────────────
# (IEM 3-letter id, IEM network code) — KCQT has no ASOS feed, skipped.

ICAO_TO_IEM: dict[str, tuple[str, str] | None] = {
    "KATL": ("ATL", "GA_ASOS"),  "KAUS": ("AUS", "TX_ASOS"),
    "KBOS": ("BOS", "MA_ASOS"),  "KDCA": ("DCA", "VA_ASOS"),
    "KDEN": ("DEN", "CO_ASOS"),  "KDFW": ("DFW", "TX_ASOS"),
    "KHOU": ("HOU", "TX_ASOS"),  "KJFK": ("JFK", "NY_ASOS"),
    "KLAS": ("LAS", "NV_ASOS"),  "KMDW": ("MDW", "IL_ASOS"),
    "KMIA": ("MIA", "FL_ASOS"),  "KMSP": ("MSP", "MN_ASOS"),
    "KMSY": ("MSY", "LA_ASOS"),  "KOKC": ("OKC", "OK_ASOS"),
    "KPHL": ("PHL", "PA_ASOS"),  "KPHX": ("PHX", "AZ_ASOS"),
    "KSAT": ("SAT", "TX_ASOS"),  "KSEA": ("SEA", "WA_ASOS"),
    "KSFO": ("SFO", "CA_ASOS"),  "KCQT": None,
}

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"


# ── Ensure forecast data (IEM backfill) ──────────────────────────────────────

def ensure_forecast_data(
    conn: sqlite3.Connection, stations, start: str, end: str
) -> int:
    """Backfill IEM forecast history for any station/model pair missing data."""
    total = 0
    start_iso = start + "T00:00Z"
    end_iso = end + "T23:59Z"
    expected_days = (
        datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")
    ).days

    # Build set of (station, model) that have sufficient daily coverage
    covered = set()
    for row in conn.execute(
        "SELECT station, model, COUNT(DISTINCT date(runtime)) "
        "FROM iem_forecasts "
        "WHERE runtime >= ? AND runtime <= ? GROUP BY station, model",
        (start + "T00:00:00Z", end + "T23:59:00Z"),
    ).fetchall():
        if row[2] >= expected_days * 0.5:
            covered.add((row[0], row[1]))

    for stn in stations:
        if ICAO_TO_IEM.get(stn.icao) is None:
            continue
        for model in ("NBS", "GFS", "LAV"):
            if (stn.icao, model) in covered:
                continue

            print(
                f"  Backfilling {model} for {stn.city} ({stn.icao})...",
                end=" ", flush=True,
            )
            try:
                df = fetch_history(model, start_iso, end_iso, [stn.icao])
            except Exception as e:
                print(f"FAILED ({e})")
                continue

            if df.empty:
                print("no data")
                continue

            n = upsert_df(conn, "iem_forecasts", df)
            conn.commit()
            print(f"{n} rows")
            total += n

    return total


# ── Observed-high cache ──────────────────────────────────────────────────────

def _ensure_obs_table(conn: sqlite3.Connection):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS observed_highs (
               station TEXT, date TEXT, max_tmpf REAL, min_tmpf REAL,
               PRIMARY KEY (station, date))"""
    )
    conn.commit()


def fetch_observed_highs(
    conn: sqlite3.Connection, stations, start: str, end: str
) -> int:
    """Download daily Tmax from IEM ASOS; skip stations already cached."""
    _ensure_obs_table(conn)
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    expected_days = (ed - sd).days
    total = 0

    for stn in stations:
        iem = ICAO_TO_IEM.get(stn.icao)
        if not iem:
            continue
        iem_id, network = iem

        cnt = conn.execute(
            "SELECT COUNT(*) FROM observed_highs WHERE station=? AND date>=? AND date<=?",
            (stn.icao, start, end),
        ).fetchone()[0]
        if cnt >= expected_days * 0.9:
            continue

        print(f"  Fetching obs for {stn.city} ({stn.icao})...", end=" ", flush=True)
        try:
            resp = requests.get(
                IEM_DAILY_URL,
                params=[
                    ("network", network), ("stations", iem_id),
                    ("year1", sd.year), ("month1", sd.month), ("day1", sd.day),
                    ("year2", ed.year), ("month2", ed.month), ("day2", ed.day),
                    ("na", "blank"), ("format", "comma"),
                ],
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        try:
            df = pd.read_csv(io.StringIO(resp.text), comment="#")
        except Exception:
            print("PARSE ERROR")
            continue

        if df.empty or "max_temp_f" not in df.columns:
            print("NO DATA")
            continue

        df = df[["day", "max_temp_f"]].rename(
            columns={"day": "date", "max_temp_f": "max_tmpf"}
        )
        df["station"] = stn.icao
        df = df.dropna(subset=["max_tmpf"])

        for _, r in df.iterrows():
            conn.execute(
                "INSERT OR REPLACE INTO observed_highs VALUES (?,?,?,NULL)",
                (r["station"], r["date"], r["max_tmpf"]),
            )
        conn.commit()
        print(f"{len(df)} days")
        total += len(df)
        _time.sleep(0.5)

    return total


# ── Bulk forecast loader ─────────────────────────────────────────────────────

def load_forecasts(
    conn: sqlite3.Connection, station: str
) -> dict[str, pd.DataFrame]:
    out = {}
    for model in ("NBS", "GFS", "LAV"):
        df = pd.read_sql(
            "SELECT * FROM iem_forecasts WHERE station=? AND model=?",
            conn, params=(station, model),
        )
        if not df.empty:
            df["runtime_dt"] = pd.to_datetime(df["runtime"])
            df["ftime_dt"] = pd.to_datetime(df["ftime"])
        out[model] = df
    return out


# ── Point-in-time extraction ─────────────────────────────────────────────────

def _latest_run_with_value(
    df: pd.DataFrame, cutoff_dt: datetime, ftime_target, value_col: str
) -> pd.Series | None:
    """
    Find the latest model run before *cutoff_dt* that has a non-null
    *value_col* at *ftime_target*.  Falls back through older runs if the
    most recent one has NaN (common for NBM near end-of-day).
    """
    if df.empty:
        return None
    avail = df[df["runtime_dt"] <= cutoff_dt]
    if avail.empty:
        return None

    # Filter to rows matching the target ftime that have a valid value
    candidates = avail[
        (avail["ftime_dt"] == ftime_target) & avail[value_col].notna()
    ]
    if candidates.empty:
        return None

    # Pick the row from the latest runtime
    best = candidates.loc[candidates["runtime_dt"].idxmax()]
    return best


def extract_at(
    forecasts: dict[str, pd.DataFrame],
    target_date: str,
    as_of: datetime,
) -> dict:
    """
    Extract point forecasts available at *as_of* for *target_date*.
    Returns dict: nbm_max, nbm_sd, gfs_max, lamp_max.
    """
    td = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    next_00z = td + timedelta(days=1)
    day_12z = td.replace(hour=12)
    next_03z = (td + timedelta(days=1)).replace(hour=3)

    out: dict = {
        "nbm_max": None, "nbm_sd": None, "gfs_max": None, "lamp_max": None,
    }

    # NBM — find latest run with valid txn (falls back past NaN runs)
    nbs = forecasts.get("NBS", pd.DataFrame())
    if not nbs.empty:
        hit = _latest_run_with_value(
            nbs, as_of - DISSEMINATION_LAG["NBS"], next_00z, "txn"
        )
        if hit is not None:
            out["nbm_max"] = float(hit["txn"])
            out["nbm_sd"] = float(hit["xnd"]) if pd.notna(hit.get("xnd")) else None

    # GFS MOS
    gfs = forecasts.get("GFS", pd.DataFrame())
    if not gfs.empty:
        hit = _latest_run_with_value(
            gfs, as_of - DISSEMINATION_LAG["GFS"], next_00z, "n_x"
        )
        if hit is not None:
            out["gfs_max"] = float(hit["n_x"])

    # LAMP — max of hourly temps in 12Z-03Z+1 window from latest run
    lav = forecasts.get("LAV", pd.DataFrame())
    if not lav.empty:
        cutoff = as_of - DISSEMINATION_LAG["LAV"]
        avail = lav[lav["runtime_dt"] <= cutoff]
        if not avail.empty:
            latest_rt = avail["runtime_dt"].max()
            run = avail[avail["runtime_dt"] == latest_rt]
            win = run[(run["ftime_dt"] >= day_12z) & (run["ftime_dt"] <= next_03z)]
            tmps = win["tmp"].dropna()
            if not tmps.empty:
                out["lamp_max"] = float(tmps.max())

    return out


def blend(fc: dict) -> tuple[float | None, float | None]:
    """Weighted blend -> (mu, sigma).  Same logic as probability.py."""
    vals, wts = [], []
    if fc["nbm_max"] is not None:
        vals.append(fc["nbm_max"]); wts.append(2.0)
    if fc["gfs_max"] is not None:
        vals.append(fc["gfs_max"]); wts.append(1.0)
    if fc["lamp_max"] is not None:
        vals.append(fc["lamp_max"]); wts.append(1.0)
    if not vals:
        return None, None
    mu = float(np.average(vals, weights=wts))
    sigma = max(fc["nbm_sd"], 1.5) if fc["nbm_sd"] and fc["nbm_sd"] > 0 else 4.0
    return mu, sigma


# ── Main backtest loop ───────────────────────────────────────────────────────

def run_backtest(
    stations,
    start: str,
    end: str,
    decision_hour: int = 15,
) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))

    # 1. Ensure forecast data is backfilled from IEM
    print("Checking forecast data coverage...")
    n_fc = ensure_forecast_data(conn, stations, start, end)
    if n_fc:
        print(f"  Backfilled {n_fc} forecast rows.\n")
    else:
        print("  All forecast data present.\n")

    # 2. Ensure observed highs
    print("Checking observed-high cache...")
    n_obs = fetch_observed_highs(conn, stations, start, end)
    if n_obs:
        print(f"  Downloaded {n_obs} observation-days.\n")
    else:
        print("  All observations cached.\n")

    # Load observations into lookup
    obs_df = pd.read_sql(
        "SELECT station, date, max_tmpf FROM observed_highs WHERE date>=? AND date<=?",
        conn, params=(start, end),
    )
    obs_lookup: dict[tuple[str, str], float] = {
        (r["station"], r["date"]): r["max_tmpf"] for _, r in obs_df.iterrows()
    }

    # 3. Build date list
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    dates = [
        (sd + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((ed - sd).days + 1)
    ]

    # 4. Run backtest
    print("Running backtest...")
    rows = []
    for stn in stations:
        if ICAO_TO_IEM.get(stn.icao) is None:
            continue

        print(f"  {stn.city} ({stn.icao})...", end=" ", flush=True)
        forecasts = load_forecasts(conn, stn.icao)
        n_ok = 0

        for d in dates:
            obs = obs_lookup.get((stn.icao, d))
            if obs is None:
                continue

            as_of = datetime.strptime(d, "%Y-%m-%d").replace(
                hour=decision_hour, tzinfo=timezone.utc
            )
            fc = extract_at(forecasts, d, as_of)
            mu, sigma = blend(fc)
            if mu is None:
                continue

            rows.append({
                "city": stn.city,
                "station": stn.icao,
                "date": d,
                "obs_max": obs,
                "nbm_max": fc["nbm_max"],
                "nbm_sd_raw": fc["nbm_sd"],
                "gfs_max": fc["gfs_max"],
                "lamp_max": fc["lamp_max"],
                "blend_mu": mu,
                "blend_sigma": sigma,
            })
            n_ok += 1

        print(f"{n_ok} days")

    conn.close()
    return pd.DataFrame(rows)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame):
    df = df.dropna(subset=["blend_mu", "obs_max"]).copy()
    n = len(df)
    if n == 0:
        print("No valid results to evaluate.")
        return

    print(f"\n{'=' * 80}")
    print(f"  BACKTEST RESULTS")
    print(
        f"  {df['date'].min()} to {df['date'].max()}  |  {n} station-days"
        f"  |  {df['station'].nunique()} stations"
    )
    print(f"{'=' * 80}")

    # ── Point-forecast accuracy ──────────────────────────────────────────
    print(f"\n  Point Forecast (predicting daily Tmax):")
    print(f"  {'Model':<12} {'MAE':>7} {'Bias':>8} {'RMSE':>7} {'N':>7}")
    print(f"  {'-' * 45}")

    for label, col in [
        ("Blend", "blend_mu"),
        ("NBM", "nbm_max"),
        ("GFS MOS", "gfs_max"),
        ("LAMP", "lamp_max"),
    ]:
        v = df.dropna(subset=[col])
        if v.empty:
            continue
        err = v[col] - v["obs_max"]
        print(
            f"  {label:<12} {err.abs().mean():>6.1f}F"
            f" {err.mean():>+7.1f}F"
            f" {np.sqrt((err ** 2).mean()):>6.1f}F"
            f" {len(v):>7}"
        )

    # ── Sigma calibration ────────────────────────────────────────────────
    err = df["blend_mu"] - df["obs_max"]
    sig = df["blend_sigma"]
    w1 = (err.abs() <= sig).mean() * 100
    w2 = (err.abs() <= 2 * sig).mean() * 100

    print(f"\n  Sigma Calibration:")
    print(f"  Within +/- 1 sigma: {w1:5.1f}%  (ideal 68%)")
    print(f"  Within +/- 2 sigma: {w2:5.1f}%  (ideal 95%)")

    # ── Brier score ──────────────────────────────────────────────────────
    thresholds = np.arange(40, 110, 5)
    bs_blend, bs_nbm = [], []
    for T in thresholds:
        o = (df["obs_max"] >= T).astype(float)
        p = 1.0 - norm.cdf(T, df["blend_mu"], df["blend_sigma"])
        bs_blend.append(((p - o) ** 2).mean())

        nv = df.dropna(subset=["nbm_max", "nbm_sd_raw"])
        if not nv.empty:
            ns = nv["nbm_sd_raw"].clip(lower=1.5)
            pn = 1.0 - norm.cdf(T, nv["nbm_max"], ns)
            on = (nv["obs_max"] >= T).astype(float)
            bs_nbm.append(((pn - on) ** 2).mean())

    print(f"\n  Brier Score (lower = better, avg across {len(thresholds)} thresholds):")
    print(f"  Blend:    {np.mean(bs_blend):.4f}")
    if bs_nbm:
        print(f"  NBM-only: {np.mean(bs_nbm):.4f}")

    # ── Reliability diagram ──────────────────────────────────────────────
    all_p, all_o = [], []
    for T in range(30, 115):
        p = 1.0 - norm.cdf(T, df["blend_mu"].values, df["blend_sigma"].values)
        o = (df["obs_max"].values >= T).astype(float)
        all_p.append(p)
        all_o.append(o)
    all_p = np.concatenate(all_p)
    all_o = np.concatenate(all_o)

    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    bins[-1] = (0.9, 1.01)  # include p=1.0

    print(f"\n  Reliability (P(Tmax >= T), all thresholds 30-114F):")
    print(f"  {'Bin':<12} {'Predicted':>10} {'Observed':>10} {'Count':>10}")
    print(f"  {'-' * 46}")

    for lo, hi in bins:
        mask = (all_p >= lo) & (all_p < hi)
        cnt = mask.sum()
        if cnt < 10:
            continue
        print(
            f"  {lo * 100:3.0f}-{hi * 100:3.0f}%"
            f"     {all_p[mask].mean() * 100:>7.1f}%"
            f"  {all_o[mask].mean() * 100:>7.1f}%"
            f"  {cnt:>10}"
        )

    # ── Per-city breakdown ───────────────────────────────────────────────
    print(f"\n  City Breakdown (sorted by MAE):")
    print(f"  {'City':<18} {'MAE':>7} {'Bias':>8} {'Days':>6}")
    print(f"  {'-' * 42}")

    city_stats = (
        df.groupby("city")
        .apply(
            lambda g: pd.Series({
                "mae": (g["blend_mu"] - g["obs_max"]).abs().mean(),
                "bias": (g["blend_mu"] - g["obs_max"]).mean(),
                "n": len(g),
            }),
            include_groups=False,
        )
        .sort_values("mae", ascending=False)
    )
    for city, r in city_stats.iterrows():
        print(f"  {city:<18} {r['mae']:>6.1f}F {r['bias']:>+7.1f}F {int(r['n']):>6}")

    # ── Monthly breakdown ────────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    monthly = (
        df.groupby("month")
        .apply(
            lambda g: pd.Series({
                "mae": (g["blend_mu"] - g["obs_max"]).abs().mean(),
                "bias": (g["blend_mu"] - g["obs_max"]).mean(),
                "n": len(g),
            }),
            include_groups=False,
        )
    )
    print(f"\n  Monthly Trend:")
    print(f"  {'Month':<10} {'MAE':>7} {'Bias':>8} {'Days':>6}")
    print(f"  {'-' * 34}")
    for month, r in monthly.iterrows():
        print(f"  {month:<10} {r['mae']:>6.1f}F {r['bias']:>+7.1f}F {int(r['n']):>6}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest probability engine")
    parser.add_argument("--days", type=int, help="Limit to last N days (default: all)")
    parser.add_argument("--station", help="Single ICAO station (e.g. KMDW)")
    parser.add_argument(
        "--decision-hour", type=int, default=15,
        help="UTC hour at which forecasts are evaluated (default 15 = ~10am ET)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Default date range: last 365 days (or --days override)
    end_dt = datetime.now(timezone.utc) - timedelta(days=1)  # yesterday
    end_date = end_dt.strftime("%Y-%m-%d")
    days = args.days or 365
    start_date = (end_dt - timedelta(days=days)).strftime("%Y-%m-%d")

    # Stations
    if args.station:
        stn = ICAO_TO_STATION.get(args.station.upper())
        if not stn:
            print(f"Unknown station: {args.station}")
            sys.exit(1)
        stations = [stn]
    else:
        stations = [s for s in STATIONS if ICAO_TO_IEM.get(s.icao) is not None]

    print(
        f"Backtest: {start_date} to {end_date} ({days} days),"
        f" {len(stations)} stations,"
        f" decision hour {args.decision_hour}:00 UTC\n"
    )

    results = run_backtest(stations, start_date, end_date, args.decision_hour)

    if results.empty:
        print("No results — forecast data may still be downloading on first run.")
        sys.exit(1)

    evaluate(results)


if __name__ == "__main__":
    main()
