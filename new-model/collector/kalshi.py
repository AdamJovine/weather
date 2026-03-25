"""
Kalshi price fetcher for NYC high-temperature markets.

Polls all open KXHIGHNY contracts and records bid/ask/volume snapshots.
Auth uses RSA key-pair via env vars KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KALSHI_BASE_URL = os.getenv(
    "KALSHI_BASE_URL",
    "https://api.elections.kalshi.com/trade-api/v2",
)
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# NYC high-temp series on Kalshi
SERIES_TICKER = "KXHIGHNY"


def _get_client():
    """Build an authenticated Kalshi client. Raises on missing creds."""
    from kalshi_python_sync import KalshiAuth, KalshiClient, Configuration

    if not KALSHI_API_KEY_ID:
        raise RuntimeError("KALSHI_API_KEY_ID not set")
    key_path = Path(KALSHI_PRIVATE_KEY_PATH or "")
    if not key_path.exists():
        raise RuntimeError(f"Kalshi private key not found: {key_path}")

    config = Configuration(host=KALSHI_BASE_URL)
    client = KalshiClient(configuration=config)
    client.kalshi_auth = KalshiAuth(KALSHI_API_KEY_ID, key_path.read_text())
    return client


def _to_cents(val) -> int | None:
    """Convert a dollar string like '0.4600' or a cent int to integer cents."""
    if val is None:
        return None
    try:
        f = float(val)
        # If it looks like dollars (< 1.01), convert to cents
        if f < 1.01:
            return round(f * 100)
        return round(f)
    except (TypeError, ValueError):
        return None


def _to_int(val) -> int | None:
    """Convert a numeric string or float to int."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def fetch_kalshi_prices() -> pd.DataFrame:
    """
    Fetch current bid/ask/volume for all open KXHIGHNY contracts.

    Returns a DataFrame ready to upsert into ``kalshi_prices``:
      ticker, ts, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest
    """
    try:
        client = _get_client()
    except Exception as e:
        log.error("Kalshi auth failed: %s", e)
        return pd.DataFrame()

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = client._market_api.get_markets(
            series_ticker=SERIES_TICKER, status="open", limit=200,
        )
        markets = resp.markets if hasattr(resp, "markets") else resp.get("markets", [])
    except Exception as e:
        log.error("Kalshi market fetch failed: %s", e)
        return pd.DataFrame()

    rows = []
    for m in markets:
        m = m if isinstance(m, dict) else m.to_dict()
        ticker = m.get("ticker")
        if not ticker:
            continue

        # SDK returns prices as dollar strings (e.g. "0.4600") — convert to cents.
        # Also handle older SDK versions that may use plain cent integers.
        yes_bid = _to_cents(m.get("yes_bid_dollars") or m.get("yes_bid"))
        yes_ask = _to_cents(m.get("yes_ask_dollars") or m.get("yes_ask"))
        no_bid = _to_cents(m.get("no_bid_dollars") or m.get("no_bid"))
        no_ask = _to_cents(m.get("no_ask_dollars") or m.get("no_ask"))
        volume = _to_int(m.get("volume_fp") or m.get("volume"))
        open_interest = _to_int(m.get("open_interest_fp") or m.get("open_interest"))

        rows.append({
            "ticker": ticker,
            "ts": now_ts,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "volume": volume,
            "open_interest": open_interest,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info("Kalshi: %d contracts snapshotted", len(df))
    else:
        log.warning("Kalshi: no open %s markets found", SERIES_TICKER)
    return df
