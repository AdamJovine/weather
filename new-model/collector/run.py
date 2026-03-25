#!/usr/bin/env python3
"""
Main orchestrator for the EMOS data collector.

Runs a continuous loop that polls METAR and fetches LAMP/NBM/GFS MOS from IEM
on staggered schedules. Each source is fetched independently so a failure in
one doesn't block the others.

Usage:
    python -m collector.run              # continuous collection
    python -m collector.run --once       # single pass (for cron)
    python -m collector.run --backfill 7 # backfill last N days from IEM
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from collector.config import (
    DB_PATH,
    GFS_MOS_INTERVAL,
    ICAO_LIST,
    LAMP_INTERVAL,
    METAR_INTERVAL,
    NBM_INTERVAL,
)
from collector.db import get_db, upsert_df
from collector.iem import (
    fetch_history,
    fetch_latest_gfs_ext,
    fetch_latest_gfs_mos,
    fetch_latest_lamp,
    fetch_latest_nbm,
)
from collector.kalshi import fetch_kalshi_prices
from collector.metar import fetch_metar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collector")


def collect_metar() -> int:
    """Fetch and store METAR observations. Returns rows written."""
    df = fetch_metar(hours=3)
    if df.empty:
        return 0
    with get_db() as conn:
        return upsert_df(conn, "metar_obs", df)


def collect_kalshi() -> int:
    """Fetch and store Kalshi price snapshots. Returns rows written."""
    df = fetch_kalshi_prices()
    if df.empty:
        return 0
    with get_db() as conn:
        return upsert_df(conn, "kalshi_prices", df)


def collect_lamp() -> int:
    """Fetch and store latest LAMP forecast. Returns rows written."""
    df = fetch_latest_lamp()
    if df.empty:
        return 0
    with get_db() as conn:
        return upsert_df(conn, "iem_forecasts", df)


def collect_nbm() -> int:
    """Fetch and store latest NBM forecast. Returns rows written."""
    df = fetch_latest_nbm()
    if df.empty:
        return 0
    with get_db() as conn:
        return upsert_df(conn, "iem_forecasts", df)


def collect_gfs_mos() -> int:
    """Fetch and store latest GFS MOS + Extended. Returns rows written."""
    total = 0
    for fetch_fn in (fetch_latest_gfs_mos, fetch_latest_gfs_ext):
        df = fetch_fn()
        if not df.empty:
            with get_db() as conn:
                total += upsert_df(conn, "iem_forecasts", df)
    return total


def backfill(days: int) -> None:
    """
    Backfill IEM forecast data for the last N days.
    Useful for bootstrapping the database or filling gaps after downtime.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end.strftime("%Y-%m-%dT%H:%MZ")

    log.info("Backfilling %d days: %s to %s", days, start_str, end_str)

    for model, label in [("LAV", "LAMP"), ("NBS", "NBM"), ("GFS", "GFS MOS"), ("MEX", "GFS Ext")]:
        log.info("  Fetching %s history...", label)
        df = fetch_history(model, start_str, end_str)
        if not df.empty:
            with get_db() as conn:
                n = upsert_df(conn, "iem_forecasts", df)
            log.info("  %s: %d rows written", label, n)
        else:
            log.warning("  %s: no data returned", label)
        time.sleep(2)  # be polite to IEM

    # Also backfill METAR (only gets last 24h from the API, so limited)
    log.info("  Fetching METAR (last 24h)...")
    df = fetch_metar(hours=24)
    if not df.empty:
        with get_db() as conn:
            n = upsert_df(conn, "metar_obs", df)
        log.info("  METAR: %d rows written", n)


def run_once() -> None:
    """Single collection pass — fetch everything once."""
    log.info("=== Single collection pass ===")
    log.info("DB: %s", DB_PATH)

    for name, fn in [
        ("METAR", collect_metar),
        ("KALSHI", collect_kalshi),
        ("LAMP", collect_lamp),
        ("NBM", collect_nbm),
        ("GFS MOS", collect_gfs_mos),
    ]:
        try:
            n = fn()
            log.info("%s: %d rows written", name, n)
        except Exception as e:
            log.error("%s failed: %s", name, e, exc_info=True)

    log.info("=== Pass complete ===")


def collect_metar_and_kalshi() -> None:
    """Fetch METAR then immediately fetch Kalshi prices (synced snapshots)."""
    try:
        n = collect_metar()
        log.info("METAR: %d rows", n)
    except Exception as e:
        log.error("METAR failed: %s", e, exc_info=True)

    try:
        n = collect_kalshi()
        log.info("KALSHI: %d rows", n)
    except Exception as e:
        log.error("KALSHI failed: %s", e, exc_info=True)


def run_loop() -> None:
    """
    Continuous collection loop with staggered schedules.

    METAR and Kalshi are fetched together so price snapshots are synced with
    the latest observations.  Forecast models run on their own timers.
    """
    log.info("=== Starting continuous collection ===")
    log.info("DB: %s", DB_PATH)
    log.info("Stations: %s", ", ".join(ICAO_LIST))
    log.info(
        "Intervals: METAR+KALSHI=%ds  LAMP=%ds  NBM=%ds  GFS_MOS=%ds",
        METAR_INTERVAL, LAMP_INTERVAL, NBM_INTERVAL, GFS_MOS_INTERVAL,
    )

    # Schedule: [name, collect_fn, interval_sec, last_run_ts]
    schedule = [
        ["METAR+KALSHI", collect_metar_and_kalshi, METAR_INTERVAL,    0.0],
        ["LAMP",         collect_lamp,             LAMP_INTERVAL,     0.0],
        ["NBM",          collect_nbm,              NBM_INTERVAL,      0.0],
        ["GFS MOS",      collect_gfs_mos,          GFS_MOS_INTERVAL,  0.0],
    ]

    while True:
        now = time.time()

        for entry in schedule:
            name, fn, interval, last_run = entry
            if now - last_run >= interval:
                try:
                    fn()
                except Exception as e:
                    log.error("%s failed: %s", name, e, exc_info=True)
                entry[3] = now  # update last_run

        # Sleep until the next source is due
        next_due = min(entry[3] + entry[2] for entry in schedule)
        sleep_sec = max(1.0, next_due - time.time())
        time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description="EMOS data collector")
    parser.add_argument(
        "--once", action="store_true",
        help="Single collection pass then exit",
    )
    parser.add_argument(
        "--backfill", type=int, metavar="DAYS",
        help="Backfill IEM data for the last N days",
    )
    args = parser.parse_args()

    if args.backfill:
        backfill(args.backfill)
    elif args.once:
        run_once()
    else:
        try:
            run_loop()
        except KeyboardInterrupt:
            log.info("Shutting down.")


if __name__ == "__main__":
    main()
