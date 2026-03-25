#!/usr/bin/env python3
"""
Collector integration tests — verifies API fetches, DB storage, and
point-in-time reconstruction for backtesting integrity.

The critical property: at any simulated time T, queries must return ONLY
data with runtime <= T (forecasts) or obs_time <= T (METAR). No future leakage.

Usage:
    python -m test_collector              # run all tests
    python -m test_collector --live       # include live API fetch tests (hits real APIs)
    python -m test_collector --audit KJFK 2026-03-25  # audit one station/date
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Make sure collector package is importable ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector.config import DB_PATH, ICAO_LIST, ICAO_TO_STATION
from collector.db import get_db, upsert_df, _SCHEMA
from collector.iem import (
    _parse_iem_json,
    extract_gfs_mos_max_temp,
    extract_lamp_max_temp,
    extract_nbm_max_temp,
    fetch_latest_gfs_ext,
    fetch_latest_gfs_mos,
    fetch_latest_lamp,
    fetch_latest_nbm,
)
from collector.metar import fetch_metar

# ── Known dissemination lags (conservative estimates) ────────────────────────
# These represent the delay between model init time (runtime) and when the
# data is actually available for download. If we use runtime directly as
# "available at", we introduce look-ahead bias by this many minutes.
DISSEMINATION_LAG = {
    "LAV": timedelta(minutes=30),   # LAMP: ~25-35 min after init
    "NBS": timedelta(minutes=60),   # NBM: ~45-75 min after init
    "GFS": timedelta(hours=4),      # GFS MOS: ~3.5-4.5 hr after init
    "MEX": timedelta(hours=5),      # GFS Ext MOS: ~4-5.5 hr after init
}

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
INFO = "\033[94mℹ INFO\033[0m"
HEADER = "\033[1;96m"
RESET = "\033[0m"

n_pass = 0
n_fail = 0
n_warn = 0


def check(condition: bool, msg: str, warn_only: bool = False) -> bool:
    global n_pass, n_fail, n_warn
    if condition:
        print(f"  {PASS}  {msg}")
        n_pass += 1
    elif warn_only:
        print(f"  {WARN}  {msg}")
        n_warn += 1
    else:
        print(f"  {FAIL}  {msg}")
        n_fail += 1
    return condition


def section(title: str):
    print(f"\n{HEADER}{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: Live API fetch tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_live_api_fetches():
    section("TEST 1: Live API Fetches")

    test_stations = ["KJFK", "KORD", "KPHX"]  # small set for speed

    # ── METAR ──
    print("  Fetching METAR (3 stations)...")
    df = fetch_metar(hours=2)
    check(not df.empty, f"METAR returned data: {len(df)} rows")
    if not df.empty:
        check("station" in df.columns and "obs_time" in df.columns and "temp_f" in df.columns,
              f"METAR has required columns: {list(df.columns)}")
        check(df["temp_f"].notna().sum() > 0, "METAR has non-null temperatures")

        # Verify obs_time is in the past
        latest = pd.to_datetime(df["obs_time"].max())
        now = datetime.now(timezone.utc)
        check(latest <= now.replace(tzinfo=None) + timedelta(minutes=5),
              f"METAR latest obs_time in past: {df['obs_time'].max()}")

        # Show sample
        print(f"\n  {INFO}  METAR sample:")
        for _, row in df.head(3).iterrows():
            print(f"         {row['station']}  {row['obs_time']}  {row['temp_f']}°F")
    print()

    # ── LAMP ──
    print("  Fetching LAMP (3 stations)...")
    df = fetch_latest_lamp(stations=test_stations)
    check(not df.empty, f"LAMP returned data: {len(df)} rows")
    if not df.empty:
        check("runtime" in df.columns and "ftime" in df.columns,
              "LAMP has runtime and ftime columns")
        check(df["tmp"].notna().sum() > 0, "LAMP has non-null tmp values")

        # runtime should be in the recent past (within last 2 hours)
        rt = pd.to_datetime(df["runtime"].iloc[0])
        age_hr = (datetime.now(timezone.utc) - rt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
        check(age_hr < 3, f"LAMP runtime is recent: {df['runtime'].iloc[0]} ({age_hr:.1f}h ago)")

        print(f"\n  {INFO}  LAMP runtime={df['runtime'].iloc[0]}, "
              f"ftimes: {df['ftime'].min()} → {df['ftime'].max()}")
    print()

    # ── NBM ──
    print("  Fetching NBM (3 stations)...")
    df = fetch_latest_nbm(stations=test_stations)
    check(not df.empty, f"NBM returned data: {len(df)} rows")
    if not df.empty:
        check("runtime" in df.columns and "ftime" in df.columns,
              "NBM has runtime and ftime columns")
        has_txn = df["txn"].notna().sum() > 0 if "txn" in df.columns else False
        has_xnd = df["xnd"].notna().sum() > 0 if "xnd" in df.columns else False
        check(has_txn, f"NBM has txn (max/min temp): {df['txn'].notna().sum()} non-null",
              warn_only=True)
        check(has_xnd, f"NBM has xnd (std dev): {df['xnd'].notna().sum()} non-null",
              warn_only=True)

        print(f"\n  {INFO}  NBM runtime={df['runtime'].iloc[0]}, "
              f"ftimes: {df['ftime'].min()} → {df['ftime'].max()}")
    print()

    # ── GFS MOS ──
    print("  Fetching GFS MOS (3 stations)...")
    df = fetch_latest_gfs_mos(stations=test_stations)
    check(not df.empty, f"GFS MOS returned data: {len(df)} rows")
    if not df.empty:
        has_nx = df["n_x"].notna().sum() > 0 if "n_x" in df.columns else False
        check(has_nx, f"GFS MOS has n_x (max/min): {df['n_x'].notna().sum()} non-null",
              warn_only=True)
        print(f"\n  {INFO}  GFS MOS runtime={df['runtime'].iloc[0]}, "
              f"ftimes: {df['ftime'].min()} → {df['ftime'].max()}")
    print()

    # ── GFS Extended ──
    print("  Fetching GFS Extended MOS (3 stations)...")
    df = fetch_latest_gfs_ext(stations=test_stations)
    check(not df.empty, f"GFS Ext returned data: {len(df)} rows")
    if not df.empty:
        print(f"\n  {INFO}  GFS Ext runtime={df['runtime'].iloc[0]}, "
              f"ftimes: {df['ftime'].min()} → {df['ftime'].max()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: DB round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def test_db_roundtrip():
    section("TEST 2: DB Round-Trip (temp database)")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        # ── Insert synthetic METAR ──
        metar_df = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T12:00:00Z", "temp_f": 42.0,
             "dew_point_f": 30.0, "wind_speed_kt": 10.0, "wind_dir": 180, "raw_metar": "TEST"},
            {"station": "KJFK", "obs_time": "2026-03-25T12:05:00Z", "temp_f": 42.5,
             "dew_point_f": 30.0, "wind_speed_kt": 10.0, "wind_dir": 180, "raw_metar": "TEST2"},
            {"station": "KJFK", "obs_time": "2026-03-25T13:00:00Z", "temp_f": 45.0,
             "dew_point_f": 31.0, "wind_speed_kt": 12.0, "wind_dir": 200, "raw_metar": "TEST3"},
        ])

        with get_db(tmp_db) as conn:
            n = upsert_df(conn, "metar_obs", metar_df)
        check(n == 3, f"Inserted {n} METAR rows")

        # Read back
        with get_db(tmp_db) as conn:
            df = pd.read_sql("SELECT * FROM metar_obs ORDER BY obs_time", conn)
        check(len(df) == 3, f"Read back {len(df)} METAR rows")
        check(df["temp_f"].tolist() == [42.0, 42.5, 45.0], "Temperatures preserved exactly")

        # ── Test upsert (INSERT OR REPLACE) ──
        update_df = pd.DataFrame([
            {"station": "KJFK", "obs_time": "2026-03-25T12:00:00Z", "temp_f": 42.1,
             "dew_point_f": 30.0, "wind_speed_kt": 10.0, "wind_dir": 180, "raw_metar": "UPDATED"},
        ])
        with get_db(tmp_db) as conn:
            n = upsert_df(conn, "metar_obs", update_df)
        with get_db(tmp_db) as conn:
            df = pd.read_sql("SELECT * FROM metar_obs ORDER BY obs_time", conn)
        check(len(df) == 3, "Upsert did not create duplicate (still 3 rows)")
        check(df.iloc[0]["temp_f"] == 42.1, f"Upsert updated value: 42.0 → {df.iloc[0]['temp_f']}")
        check(df.iloc[0]["raw_metar"] == "UPDATED", "Upsert updated raw_metar field")
        print()

        # ── Insert synthetic IEM forecasts (two model runs) ──
        iem_df = pd.DataFrame([
            # Run 1: 06Z GFS MOS
            {"station": "KJFK", "model": "GFS", "runtime": "2026-03-25T06:00:00Z",
             "ftime": "2026-03-26T00:00:00Z", "tmp": 45.0, "n_x": 52.0},
            {"station": "KJFK", "model": "GFS", "runtime": "2026-03-25T06:00:00Z",
             "ftime": "2026-03-26T12:00:00Z", "tmp": 48.0, "n_x": 42.0},
            # Run 2: 12Z GFS MOS (updated forecast)
            {"station": "KJFK", "model": "GFS", "runtime": "2026-03-25T12:00:00Z",
             "ftime": "2026-03-26T00:00:00Z", "tmp": 44.0, "n_x": 50.0},
            {"station": "KJFK", "model": "GFS", "runtime": "2026-03-25T12:00:00Z",
             "ftime": "2026-03-26T12:00:00Z", "tmp": 47.0, "n_x": 43.0},
            # NBM with uncertainty
            {"station": "KJFK", "model": "NBS", "runtime": "2026-03-25T06:00:00Z",
             "ftime": "2026-03-26T00:00:00Z", "tmp": 44.0, "txn": 51.0, "xnd": 2.5},
            {"station": "KJFK", "model": "NBS", "runtime": "2026-03-25T12:00:00Z",
             "ftime": "2026-03-26T00:00:00Z", "tmp": 43.0, "txn": 49.0, "xnd": 2.0},
        ])

        with get_db(tmp_db) as conn:
            n = upsert_df(conn, "iem_forecasts", iem_df)
        check(n == 6, f"Inserted {n} IEM forecast rows")

        # Both model runs should be stored (different PK due to different runtime)
        with get_db(tmp_db) as conn:
            df = pd.read_sql(
                "SELECT * FROM iem_forecasts WHERE station='KJFK' AND model='GFS' "
                "AND ftime='2026-03-26T00:00:00Z' ORDER BY runtime", conn)
        check(len(df) == 2, f"Both GFS runs stored for same target: {len(df)} rows")
        check(df.iloc[0]["n_x"] == 52.0 and df.iloc[1]["n_x"] == 50.0,
              f"GFS n_x values preserved: {df['n_x'].tolist()}")
        print()

        # ── Point-in-time query: before 12Z, only 06Z run visible ──
        with get_db(tmp_db) as conn:
            df = pd.read_sql(
                "SELECT * FROM iem_forecasts WHERE station='KJFK' AND model='GFS' "
                "AND ftime='2026-03-26T00:00:00Z' AND runtime <= '2026-03-25T10:00:00Z'",
                conn)
        check(len(df) == 1, "Point-in-time @10Z: only 06Z run visible")
        check(df.iloc[0]["n_x"] == 52.0, f"Point-in-time @10Z: n_x={df.iloc[0]['n_x']} (06Z value)")

        # ── Point-in-time query: after 12Z, both runs visible, latest wins ──
        with get_db(tmp_db) as conn:
            df = pd.read_sql(
                "SELECT * FROM iem_forecasts WHERE station='KJFK' AND model='GFS' "
                "AND ftime='2026-03-26T00:00:00Z' AND runtime <= '2026-03-25T14:00:00Z' "
                "ORDER BY runtime DESC LIMIT 1", conn)
        check(len(df) == 1 and df.iloc[0]["n_x"] == 50.0,
              f"Point-in-time @14Z: latest run n_x={df.iloc[0]['n_x']} (12Z value)")
        print()

        # ── METAR running max at different times ──
        with get_db(tmp_db) as conn:
            cur = conn.execute(
                "SELECT MAX(temp_f) FROM metar_obs "
                "WHERE station='KJFK' AND obs_time <= '2026-03-25T12:05:00Z'")
            running_max_1205 = cur.fetchone()[0]
            cur = conn.execute(
                "SELECT MAX(temp_f) FROM metar_obs "
                "WHERE station='KJFK' AND obs_time <= '2026-03-25T13:00:00Z'")
            running_max_1300 = cur.fetchone()[0]

        check(running_max_1205 == 42.5,
              f"Running max @12:05Z = {running_max_1205} (should be 42.5)")
        check(running_max_1300 == 45.0,
              f"Running max @13:00Z = {running_max_1300} (should be 45.0, new obs included)")

    finally:
        Path(tmp_db).unlink(missing_ok=True)
        Path(tmp_db + "-wal").unlink(missing_ok=True)
        Path(tmp_db + "-shm").unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Point-in-time reconstruction on real data
# ═══════════════════════════════════════════════════════════════════════════════

def test_point_in_time_real():
    section("TEST 3: Point-in-Time Reconstruction (real DB)")

    if not DB_PATH.exists():
        print(f"  {WARN}  No real database at {DB_PATH} — skipping")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pick a station and date that has data
        cur = conn.execute(
            "SELECT station, substr(obs_time,1,10) as d, COUNT(*) "
            "FROM metar_obs GROUP BY station, d ORDER BY COUNT(*) DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print(f"  {WARN}  No METAR data in DB — skipping")
            return

        station, date_str = row[0], row[1]
        print(f"  Testing station={station}, date={date_str}\n")

        # ── 3a: METAR running max at every observation point ──
        print(f"  {HEADER}3a: METAR Running Max Timeline{RESET}")
        print(f"  {'Time UTC':<22} {'Temp °F':>8} {'Running Max':>12} {'Obs Count':>10}")
        print(f"  {'─' * 55}")

        cur = conn.execute(
            "SELECT obs_time, temp_f FROM metar_obs "
            "WHERE station=? AND obs_time >= ? || 'T00:00:00Z' "
            "AND obs_time < ? || 'T23:59:59Z' AND temp_f IS NOT NULL "
            "ORDER BY obs_time",
            (station, date_str, date_str))
        obs_rows = cur.fetchall()

        running_max = float("-inf")
        for i, (obs_time, temp_f) in enumerate(obs_rows):
            running_max = max(running_max, temp_f)
            print(f"  {obs_time:<22} {temp_f:>8.1f} {running_max:>12.1f} {i + 1:>10}")

        check(len(obs_rows) > 0, f"METAR: {len(obs_rows)} observations for {station} on {date_str}")

        # Verify monotonicity of running max
        maxes = []
        rm = float("-inf")
        for _, t in obs_rows:
            rm = max(rm, t)
            maxes.append(rm)
        is_monotonic = all(maxes[i] <= maxes[i + 1] for i in range(len(maxes) - 1))
        check(is_monotonic, "Running max is monotonically non-decreasing")
        print()

        # ── 3b: Forecast evolution across model runs ──
        print(f"  {HEADER}3b: Forecast Evolution Across Model Runs{RESET}")

        # Find target ftime (next day 00Z for max temp)
        target_dt = pd.to_datetime(date_str)
        target_ftime = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        for model, field, label in [
            ("NBS", "txn", "NBM txn"),
            ("GFS", "n_x", "GFS n_x"),
        ]:
            cur = conn.execute(
                f"SELECT runtime, {field}, tmp FROM iem_forecasts "
                f"WHERE station=? AND model=? AND ftime=? "
                f"ORDER BY runtime",
                (station, model, target_ftime))
            frows = cur.fetchall()

            if frows:
                print(f"\n  {label} for {station}, target ftime={target_ftime}:")
                print(f"  {'Runtime':<24} {field:>8} {'tmp':>8}  {'Δ vs prev':>10}")
                print(f"  {'─' * 55}")
                prev_val = None
                for runtime, val, tmp in frows:
                    delta = ""
                    if val is not None and prev_val is not None:
                        d = val - prev_val
                        delta = f"{d:+.1f}" if d != 0 else "  —"
                    prev_val = val if val is not None else prev_val
                    val_str = f"{val:.1f}" if val is not None else "NULL"
                    tmp_str = f"{tmp:.1f}" if tmp is not None else "NULL"
                    print(f"  {runtime:<24} {val_str:>8} {tmp_str:>8}  {delta:>10}")

                non_null = [r[1] for r in frows if r[1] is not None]
                check(len(non_null) > 0,
                      f"{label}: {len(non_null)}/{len(frows)} runs have non-null {field}")
            else:
                check(False, f"{label}: no data for ftime={target_ftime}", warn_only=True)
        print()

        # ── 3c: Simulated 5-minute backtesting snapshots ──
        print(f"  {HEADER}3c: Simulated 5-Minute Backtesting Snapshots{RESET}")
        print(f"  Simulating what data is available at each 5-min interval\n")

        # Generate 5-minute time points from first obs to last obs
        if not obs_rows:
            return

        first_obs = pd.to_datetime(obs_rows[0][0]).tz_localize(None)
        last_obs = pd.to_datetime(obs_rows[-1][0]).tz_localize(None)

        # Wider range: start from 06Z (before most US markets open)
        sim_start = pd.Timestamp(f"{date_str}T06:00:00")
        sim_end = min(last_obs + pd.Timedelta(hours=1),
                      pd.Timestamp(f"{date_str}T23:55:00"))

        print(f"  {'Sim Time UTC':<20} {'METAR':>6} {'Run Max':>8} "
              f"{'NBM rt':>14} {'NBM txn':>8} "
              f"{'GFS rt':>14} {'GFS n_x':>8} "
              f"{'LAMP rt':>14} {'LAMP mx':>8}")
        print(f"  {'─' * 120}")

        t = sim_start
        prev_nbm_rt = prev_gfs_rt = prev_lamp_rt = None
        snapshot_count = 0

        while t <= sim_end:
            t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")

            # METAR: latest temp and running max as of time t
            cur = conn.execute(
                "SELECT temp_f, obs_time FROM metar_obs "
                "WHERE station=? AND obs_time <= ? AND temp_f IS NOT NULL "
                "AND obs_time >= ? || 'T00:00:00Z' "
                "ORDER BY obs_time DESC LIMIT 1",
                (station, t_str, date_str))
            metar_row = cur.fetchone()

            cur = conn.execute(
                "SELECT MAX(temp_f) FROM metar_obs "
                "WHERE station=? AND obs_time <= ? AND temp_f IS NOT NULL "
                "AND obs_time >= ? || 'T00:00:00Z'",
                (station, t_str, date_str))
            max_row = cur.fetchone()

            metar_str = f"{metar_row[0]:.0f}" if metar_row else "—"
            rmax_str = f"{max_row[0]:.0f}" if max_row and max_row[0] else "—"

            # NBM: latest runtime available as of time t
            cur = conn.execute(
                "SELECT runtime, txn FROM iem_forecasts "
                "WHERE station=? AND model='NBS' AND ftime=? AND runtime <= ? "
                "ORDER BY runtime DESC LIMIT 1",
                (station, target_ftime, t_str))
            nbm_row = cur.fetchone()
            nbm_rt = nbm_row[0][-9:-1] if nbm_row else "—"
            nbm_txn = f"{nbm_row[1]:.0f}" if nbm_row and nbm_row[1] is not None else "NULL"

            # GFS MOS: latest runtime available as of time t
            cur = conn.execute(
                "SELECT runtime, n_x FROM iem_forecasts "
                "WHERE station=? AND model IN ('GFS','MEX') AND ftime=? AND runtime <= ? "
                "ORDER BY runtime DESC LIMIT 1",
                (station, target_ftime, t_str))
            gfs_row = cur.fetchone()
            gfs_rt = gfs_row[0][-9:-1] if gfs_row else "—"
            gfs_nx = f"{gfs_row[1]:.0f}" if gfs_row and gfs_row[1] is not None else "NULL"

            # LAMP: latest runtime + max hourly tmp for target date
            cur = conn.execute(
                "SELECT runtime, MAX(tmp) FROM iem_forecasts "
                "WHERE station=? AND model='LAV' AND runtime <= ? "
                "AND ftime >= ? || 'T12:00:00Z' "
                "AND ftime <= ? || 'T03:00:00Z' "
                "AND tmp IS NOT NULL "
                "GROUP BY runtime ORDER BY runtime DESC LIMIT 1",
                (station, t_str, date_str,
                 (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
            lamp_row = cur.fetchone()
            lamp_rt = lamp_row[0][-9:-1] if lamp_row else "—"
            lamp_mx = f"{lamp_row[1]:.0f}" if lamp_row and lamp_row[1] is not None else "NULL"

            # Only print if something changed (or every 30 min regardless)
            nbm_rt_full = nbm_row[0] if nbm_row else None
            gfs_rt_full = gfs_row[0] if gfs_row else None
            lamp_rt_full = lamp_row[0] if lamp_row else None

            something_changed = (
                nbm_rt_full != prev_nbm_rt
                or gfs_rt_full != prev_gfs_rt
                or lamp_rt_full != prev_lamp_rt
                or t.minute == 0  # always show on the hour
            )

            if something_changed:
                marker = ""
                changes = []
                if nbm_rt_full != prev_nbm_rt and prev_nbm_rt is not None:
                    changes.append("NBM↑")
                if gfs_rt_full != prev_gfs_rt and prev_gfs_rt is not None:
                    changes.append("GFS↑")
                if lamp_rt_full != prev_lamp_rt and prev_lamp_rt is not None:
                    changes.append("LAMP↑")
                if changes:
                    marker = f"  ← {', '.join(changes)}"

                print(f"  {t_str:<20} {metar_str:>6} {rmax_str:>8} "
                      f"{nbm_rt:>14} {nbm_txn:>8} "
                      f"{gfs_rt:>14} {gfs_nx:>8} "
                      f"{lamp_rt:>14} {lamp_mx:>8}{marker}")

                prev_nbm_rt = nbm_rt_full
                prev_gfs_rt = gfs_rt_full
                prev_lamp_rt = lamp_rt_full
                snapshot_count += 1

            t += pd.Timedelta(minutes=5)

        print()
        check(snapshot_count > 0, f"Generated {snapshot_count} backtesting snapshots")

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: No future data leakage
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_future_leakage():
    section("TEST 4: No Future Data Leakage")

    if not DB_PATH.exists():
        print(f"  {WARN}  No real database — skipping")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # ── 4a: Every runtime should be <= every ftime in that row ──
        cur = conn.execute(
            "SELECT COUNT(*) FROM iem_forecasts WHERE runtime > ftime")
        bad = cur.fetchone()[0]
        check(bad == 0,
              f"No forecast rows where runtime > ftime (found {bad})")

        # ── 4b: METAR obs_time should be in the past ──
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = conn.execute(
            "SELECT COUNT(*) FROM metar_obs WHERE obs_time > ?", (now_str,))
        bad = cur.fetchone()[0]
        check(bad == 0, f"No METAR obs from the future (found {bad})")

        # ── 4c: For each model, runtime should be <= now ──
        cur = conn.execute(
            "SELECT model, MAX(runtime) FROM iem_forecasts GROUP BY model")
        for model, max_rt in cur.fetchall():
            check(max_rt <= now_str,
                  f"{model} latest runtime {max_rt} is in the past")

        # ── 4d: Verify runtimes match expected model cadence ──
        print()
        cur = conn.execute(
            "SELECT model, runtime, COUNT(*) FROM iem_forecasts "
            "GROUP BY model, runtime ORDER BY model, runtime")
        model_runs = {}
        for model, rt, cnt in cur.fetchall():
            model_runs.setdefault(model, []).append((rt, cnt))

        for model, runs in model_runs.items():
            runtimes = [pd.to_datetime(r[0]) for r in runs]
            if len(runtimes) >= 2:
                gaps = [(runtimes[i + 1] - runtimes[i]).total_seconds() / 3600
                        for i in range(len(runtimes) - 1)]
                min_gap = min(gaps)
                max_gap = max(gaps)
                median_gap = sorted(gaps)[len(gaps) // 2]
                print(f"  {INFO}  {model}: {len(runs)} runs, "
                      f"gap min={min_gap:.1f}h median={median_gap:.1f}h max={max_gap:.1f}h")
            else:
                print(f"  {INFO}  {model}: {len(runs)} run(s)")

        # LAMP should have ~1h gaps, NBM ~1h, GFS ~6h, MEX ~12h
        if "LAV" in model_runs and len(model_runs["LAV"]) >= 3:
            rts = [pd.to_datetime(r[0]) for r in model_runs["LAV"]]
            gaps = [(rts[i + 1] - rts[i]).total_seconds() / 3600
                    for i in range(len(rts) - 1)]
            check(sorted(gaps)[len(gaps) // 2] <= 2.0,
                  f"LAMP median gap ≤ 2h: {sorted(gaps)[len(gaps) // 2]:.1f}h (expect ~1h)")

        if "GFS" in model_runs and len(model_runs["GFS"]) >= 2:
            rts = [pd.to_datetime(r[0]) for r in model_runs["GFS"]]
            gaps = [(rts[i + 1] - rts[i]).total_seconds() / 3600
                    for i in range(len(rts) - 1)]
            check(sorted(gaps)[len(gaps) // 2] <= 8.0,
                  f"GFS median gap ≤ 8h: {sorted(gaps)[len(gaps) // 2]:.1f}h (expect ~6h)")

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: Dissemination lag awareness
# ═══════════════════════════════════════════════════════════════════════════════

def test_dissemination_lag():
    section("TEST 5: Dissemination Lag Analysis")

    print("  NWS models have a delay between init time (runtime) and when data")
    print("  is actually downloadable. Using runtime as 'available at' creates")
    print("  look-ahead bias by the dissemination lag.\n")

    print(f"  {'Model':<8} {'Code':<6} {'Diss. Lag':>10} {'Impact on Backtesting'}")
    print(f"  {'─' * 70}")
    for code, lag in DISSEMINATION_LAG.items():
        names = {"LAV": "LAMP", "NBS": "NBM", "GFS": "GFS MOS", "MEX": "GFS Ext"}
        mins = lag.total_seconds() / 60
        impact = "LOW" if mins <= 30 else "MEDIUM" if mins <= 120 else "HIGH"
        print(f"  {names[code]:<8} {code:<6} {mins:>7.0f}min   "
              f"{'A ' + code + ' 12Z run is not available until ~' + (datetime(2026,3,25,12) + lag).strftime('%H:%MZ')}")

    print(f"\n  {HEADER}Recommended: use runtime + lag as 'available_at' in backtest queries{RESET}")
    print(f"  Example for NBM: WHERE runtime + 60min <= sim_time")
    print()

    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Show concrete example: for each model run, what time was it really available?
        print(f"  {HEADER}Concrete Example — Adjusted Availability Times{RESET}\n")
        print(f"  {'Model':<6} {'Runtime (init)':<24} {'Available At (est.)':<24} {'Rows'}")
        print(f"  {'─' * 75}")

        cur = conn.execute(
            "SELECT model, runtime, COUNT(*) FROM iem_forecasts "
            "GROUP BY model, runtime ORDER BY model, runtime")
        for model, rt, cnt in cur.fetchall():
            lag = DISSEMINATION_LAG.get(model, timedelta(0))
            avail = (pd.to_datetime(rt) + lag).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"  {model:<6} {rt:<24} {avail:<24} {cnt}")

        # ── Compare naive vs lag-adjusted snapshots for one station ──
        print(f"\n  {HEADER}Naive vs Lag-Adjusted: KJFK NBM Max Temp Forecast{RESET}\n")

        # Find the target ftime
        cur = conn.execute(
            "SELECT DISTINCT substr(obs_time,1,10) FROM metar_obs "
            "WHERE station='KJFK' ORDER BY obs_time DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return
        date_str = row[0]
        target_dt = pd.to_datetime(date_str)
        target_ftime = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        print(f"  Target: KJFK max temp for {date_str} (ftime={target_ftime})")
        print(f"  {'Sim Time':<20} {'Naive NBM':>14} {'Lag-Adj NBM':>14} {'Difference'}")
        print(f"  {'─' * 65}")

        nbm_lag = DISSEMINATION_LAG["NBS"]

        for hour in range(0, 24, 2):
            sim_t = pd.Timestamp(f"{date_str}T{hour:02d}:00:00")
            sim_str = sim_t.strftime("%Y-%m-%dT%H:%M:%SZ")
            adj_str = (sim_t - nbm_lag).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Naive: runtime <= sim_time
            cur = conn.execute(
                "SELECT txn FROM iem_forecasts "
                "WHERE station='KJFK' AND model='NBS' AND ftime=? "
                "AND runtime <= ? ORDER BY runtime DESC LIMIT 1",
                (target_ftime, sim_str))
            naive = cur.fetchone()
            naive_str = f"{naive[0]:.0f}" if naive and naive[0] is not None else "—"

            # Lag-adjusted: runtime <= sim_time - lag
            cur = conn.execute(
                "SELECT txn FROM iem_forecasts "
                "WHERE station='KJFK' AND model='NBS' AND ftime=? "
                "AND runtime <= ? ORDER BY runtime DESC LIMIT 1",
                (target_ftime, adj_str))
            adj = cur.fetchone()
            adj_str_val = f"{adj[0]:.0f}" if adj and adj[0] is not None else "—"

            diff = ""
            if naive and adj and naive[0] is not None and adj[0] is not None:
                d = naive[0] - adj[0]
                diff = f"{d:+.0f}" if d != 0 else "same"
            elif naive_str != adj_str_val:
                diff = "differs"

            print(f"  {sim_str:<20} {naive_str:>14} {adj_str_val:>14} {diff:>10}")

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: Max-temp extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_max_temp_extraction():
    section("TEST 6: Max-Temp Extraction Helpers")

    if not DB_PATH.exists():
        print(f"  {WARN}  No real database — skipping")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Get date with data
        cur = conn.execute(
            "SELECT DISTINCT substr(obs_time,1,10) FROM metar_obs "
            "ORDER BY substr(obs_time,1,10) DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return
        date_str = row[0]
        print(f"  Testing max-temp extraction for {date_str}\n")

        # Load forecast data — latest run only per station/model for clean comparison
        all_df = pd.read_sql("SELECT * FROM iem_forecasts", conn)

        # For cross-model comparison, use only the latest run per station/model
        # This is what a backtester would actually use at query time
        latest_df = (
            all_df.sort_values("runtime")
            .drop_duplicates(subset=["station", "model", "ftime"], keep="last")
        )

        # ── NBM ──
        nbm_max = extract_nbm_max_temp(latest_df, date_str)
        if not nbm_max.empty:
            # Deduplicate to one row per station (latest)
            nbm_max = nbm_max.drop_duplicates(subset=["station"], keep="last")
            print(f"  NBM max temp forecasts (latest run):")
            for _, r in nbm_max.head(5).iterrows():
                sd = f" +/- {r['f_nbm_sd']:.1f}" if pd.notna(r.get("f_nbm_sd")) else ""
                print(f"    {r['station']}: {r['f_nbm']:.0f}F{sd}")
            check(True, f"NBM: {nbm_max['station'].nunique()} stations")
        else:
            check(False, "NBM max temp extraction returned empty", warn_only=True)

        # ── GFS MOS ──
        gfs_max = extract_gfs_mos_max_temp(latest_df, date_str)
        if not gfs_max.empty:
            gfs_max = gfs_max.drop_duplicates(subset=["station"], keep="last")
            print(f"\n  GFS MOS max temp forecasts (latest run):")
            for _, r in gfs_max.head(5).iterrows():
                print(f"    {r['station']}: {r['f_gfs_mos']:.0f}F")
            check(True, f"GFS MOS: {gfs_max['station'].nunique()} stations")
        else:
            check(False, "GFS MOS max temp extraction returned empty", warn_only=True)

        # ── LAMP ──
        lamp_max = extract_lamp_max_temp(latest_df, date_str)
        if not lamp_max.empty:
            lamp_max = lamp_max.drop_duplicates(subset=["station"], keep="last")
            print(f"\n  LAMP max temp forecasts (latest run):")
            for _, r in lamp_max.head(5).iterrows():
                print(f"    {r['station']}: {r['f_lamp_max']:.0f}F")
            check(True, f"LAMP: {lamp_max['station'].nunique()} stations")
        else:
            check(False, "LAMP max temp extraction returned empty", warn_only=True)

        # ── Cross-model comparison (one row per station) ──
        if not nbm_max.empty and not gfs_max.empty:
            merged = nbm_max.merge(gfs_max, on="station", how="inner")
            if not lamp_max.empty:
                merged = merged.merge(lamp_max, on="station", how="left")
            if not merged.empty:
                print(f"\n  {HEADER}Cross-Model Comparison (latest runs){RESET}")
                print(f"  {'Station':<8} {'NBM':>6} {'GFS':>6} {'LAMP':>6} {'Spread':>8}")
                print(f"  {'─' * 40}")
                for _, r in merged.iterrows():
                    vals = [r.get("f_nbm"), r.get("f_gfs_mos"), r.get("f_lamp_max")]
                    vals = [v for v in vals if pd.notna(v)]
                    spread = max(vals) - min(vals) if len(vals) >= 2 else None
                    nbm_s = f"{r['f_nbm']:.0f}" if pd.notna(r.get("f_nbm")) else "—"
                    gfs_s = f"{r['f_gfs_mos']:.0f}" if pd.notna(r.get("f_gfs_mos")) else "—"
                    lamp_s = f"{r['f_lamp_max']:.0f}" if pd.notna(r.get("f_lamp_max")) else "—"
                    spr_s = f"{spread:.0f}" if spread is not None else "—"
                    print(f"  {r['station']:<8} {nbm_s:>6} {gfs_s:>6} {lamp_s:>6} {spr_s:>8}")

                check(True, f"Cross-model comparison: {len(merged)} stations")

        # ── Compare forecast vs METAR actual ──
        cur = conn.execute(
            "SELECT station, MAX(temp_f) as obs_max FROM metar_obs "
            "WHERE substr(obs_time,1,10) = ? AND temp_f IS NOT NULL "
            "GROUP BY station", (date_str,))
        actuals = {r[0]: r[1] for r in cur.fetchall()}

        if actuals and not nbm_max.empty:
            print(f"\n  {HEADER}Forecast vs Observed (running max as of latest METAR){RESET}")
            print(f"  {'Station':<8} {'NBM':>6} {'GFS':>6} {'METAR Max':>10} {'NBM Err':>8}")
            print(f"  {'─' * 45}")
            for _, r in merged.iterrows() if not merged.empty else []:
                stn = r["station"]
                if stn in actuals:
                    obs = actuals[stn]
                    nbm_v = r["f_nbm"] if pd.notna(r.get("f_nbm")) else None
                    gfs_v = r["f_gfs_mos"] if pd.notna(r.get("f_gfs_mos")) else None
                    nbm_s = f"{nbm_v:.0f}" if nbm_v else "—"
                    gfs_s = f"{gfs_v:.0f}" if gfs_v else "—"
                    err_s = f"{nbm_v - obs:+.1f}" if nbm_v else "—"
                    print(f"  {stn:<8} {nbm_s:>6} {gfs_s:>6} {obs:>10.1f} {err_s:>8}")

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: Full station audit
# ═══════════════════════════════════════════════════════════════════════════════

def audit_station(station: str, date_str: str):
    """Deep audit of one station on one date — shows every piece of data available."""
    section(f"AUDIT: {station} on {date_str}")

    if not DB_PATH.exists():
        print(f"  {FAIL}  No database at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        target_dt = pd.to_datetime(date_str)
        target_ftime = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        # ── METAR timeline ──
        print(f"  {HEADER}METAR Observations{RESET}")
        cur = conn.execute(
            "SELECT obs_time, temp_f, dew_point_f, wind_speed_kt, wind_dir "
            "FROM metar_obs WHERE station=? AND obs_time >= ? || 'T00:00:00Z' "
            "AND obs_time < ? || 'T23:59:59Z' ORDER BY obs_time",
            (station, date_str, date_str))
        rows = cur.fetchall()
        rm = float("-inf")
        print(f"  {'Time UTC':<22} {'Temp':>6} {'Dew':>6} {'Wind':>6} {'Dir':>5} {'RunMax':>7}")
        print(f"  {'─' * 55}")
        for obs_time, temp, dew, wspd, wdir in rows:
            if temp is not None:
                rm = max(rm, temp)
            t_s = f"{temp:.1f}" if temp else "—"
            d_s = f"{dew:.1f}" if dew else "—"
            w_s = f"{wspd:.0f}" if wspd else "—"
            wd_s = f"{wdir}" if wdir else "—"
            rm_s = f"{rm:.1f}" if rm > float("-inf") else "—"
            print(f"  {obs_time:<22} {t_s:>6} {d_s:>6} {w_s:>6} {wd_s:>5} {rm_s:>7}")
        print(f"  Total: {len(rows)} observations\n")

        # ── IEM forecast runs ──
        for model, label in [("NBS", "NBM"), ("GFS", "GFS MOS"), ("LAV", "LAMP"), ("MEX", "GFS Ext")]:
            print(f"  {HEADER}{label} Forecast Runs{RESET}")
            cur = conn.execute(
                "SELECT DISTINCT runtime FROM iem_forecasts "
                "WHERE station=? AND model=? AND runtime >= ? || 'T00:00:00Z' "
                "AND runtime <= ? || 'T23:59:59Z' ORDER BY runtime",
                (station, model, date_str, date_str))
            runtimes = [r[0] for r in cur.fetchall()]

            if not runtimes:
                print(f"  No runs on {date_str}\n")
                continue

            for rt in runtimes:
                lag = DISSEMINATION_LAG.get(model, timedelta(0))
                avail = (pd.to_datetime(rt) + lag).strftime("%H:%MZ")

                cur = conn.execute(
                    "SELECT ftime, tmp, n_x, txn, xnd, tsd "
                    "FROM iem_forecasts WHERE station=? AND model=? AND runtime=? "
                    "ORDER BY ftime",
                    (station, model, rt))
                frows = cur.fetchall()
                print(f"\n  Run {rt} (est. available ~{avail}) — {len(frows)} forecast hours:")
                print(f"    {'ftime':<24} {'tmp':>6} {'n_x':>6} {'txn':>6} {'xnd':>6} {'tsd':>6}")
                print(f"    {'─' * 58}")

                for ftime, tmp, n_x, txn, xnd, tsd in frows[:10]:
                    vals = [
                        f"{tmp:.0f}" if tmp is not None else "—",
                        f"{n_x:.0f}" if n_x is not None else "—",
                        f"{txn:.0f}" if txn is not None else "—",
                        f"{xnd:.1f}" if xnd is not None else "—",
                        f"{tsd:.1f}" if tsd is not None else "—",
                    ]
                    marker = " ←MAX" if ftime == target_ftime else ""
                    print(f"    {ftime:<24} {vals[0]:>6} {vals[1]:>6} "
                          f"{vals[2]:>6} {vals[3]:>6} {vals[4]:>6}{marker}")
                if len(frows) > 10:
                    print(f"    ... ({len(frows) - 10} more rows)")
            print()

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collector integration tests and backtesting audit")
    parser.add_argument("--live", action="store_true",
                        help="Include live API fetch tests (hits real endpoints)")
    parser.add_argument("--audit", nargs=2, metavar=("STATION", "DATE"),
                        help="Deep audit one station on one date (e.g. KJFK 2026-03-25)")
    args = parser.parse_args()

    print(f"\n{HEADER}{'═' * 70}")
    print(f"  EMOS Collector Test Suite")
    print(f"  DB: {DB_PATH}")
    print(f"  Stations: {len(ICAO_LIST)} configured")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"{'═' * 70}{RESET}")

    if args.audit:
        audit_station(args.audit[0], args.audit[1])
        return

    # Always run these
    test_db_roundtrip()
    test_no_future_leakage()
    test_point_in_time_real()
    test_dissemination_lag()
    test_max_temp_extraction()

    # Optional: live API tests
    if args.live:
        test_live_api_fetches()
    else:
        print(f"\n  {INFO}  Skipping live API tests (use --live to include)")

    # Summary
    section("SUMMARY")
    total = n_pass + n_fail
    print(f"  Passed: {n_pass}/{total}")
    if n_warn:
        print(f"  Warnings: {n_warn}")
    if n_fail:
        print(f"  Failed: {n_fail}")
        print(f"\n  {FAIL}  Some tests failed!")
        sys.exit(1)
    else:
        print(f"\n  {PASS}  All tests passed!")


if __name__ == "__main__":
    main()
