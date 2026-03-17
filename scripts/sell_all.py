"""
Liquidate all open Kalshi positions.

Usage:
  python scripts/sell_all.py            # dry run — shows what would be sold
  python scripts/sell_all.py --live     # places real sell orders
"""

import argparse
import sys
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import KALSHI_BASE_URL
from src.kalshi_client import KalshiWeatherClient

RATE_SLEEP = 0.5  # seconds between API calls


def kalshi_get_raw(auth, path: str, params: dict = None) -> dict:
    url = KALSHI_BASE_URL.rstrip("/") + path
    headers = auth.create_auth_headers("GET", path.split("?")[0])
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_best_prices(auth, ticker: str) -> dict | None:
    """Return yes_ask_dollars and no_ask_dollars for a market, or None on failure."""
    try:
        data = kalshi_get_raw(auth, f"/markets/{ticker}")
        market = data.get("market", data)
        yes_ask = market.get("yes_ask_dollars")
        no_ask = market.get("no_ask_dollars")
        if yes_ask is None or no_ask is None:
            return None
        return {"yes_ask": float(yes_ask), "no_ask": float(no_ask)}
    except Exception as e:
        print(f"  Warning: could not fetch prices for {ticker}: {e}")
        return None


def main(dry_run: bool = True):
    print(f"=== Kalshi Sell-All {'(DRY RUN)' if dry_run else '*** LIVE ***'} ===\n")

    kalshi = KalshiWeatherClient.from_env()
    auth = kalshi._client.kalshi_auth

    # Fetch all open positions
    raw_positions = kalshi.get_positions(limit=500)
    if not raw_positions:
        print("No open positions found.")
        return

    print(f"Found {len(raw_positions)} open position(s):\n")

    orders = []
    for pos in raw_positions:
        ticker = pos["ticker"]
        # SDK returns position_fp (float string); positive = long yes, negative = long no
        net = float(pos.get("position_fp") or pos.get("position") or 0)

        if net == 0:
            continue

        side = "yes" if net > 0 else "no"
        count = abs(net)

        prices = fetch_best_prices(auth, ticker)
        time.sleep(RATE_SLEEP)

        if prices is None:
            print(f"  {ticker}: skipping — could not fetch prices")
            continue

        # Bid price = complement of the other side's ask
        if side == "yes":
            bid = max(0.0, 1.0 - prices["no_ask"])
        else:
            bid = max(0.0, 1.0 - prices["yes_ask"])

        bid_cents = int(round(bid * 100))

        print(
            f"  {ticker}\n"
            f"    side={side}  contracts={count}"
            f"  bid=${bid:.2f}  value≈${count * bid:.2f}"
        )
        orders.append((ticker, side, count, bid_cents))

    if not orders:
        print("\nNo positions to close.")
        return

    print(f"\nTotal positions to close: {len(orders)}")

    if dry_run:
        print("\nDry run — no orders placed. Run with --live to execute.")
        return

    print(f"\nPlacing {len(orders)} sell order(s)...")
    for ticker, side, count, price_cents in orders:
        try:
            resp = kalshi.place_order(
                ticker=ticker,
                side=side,
                count=count,
                price=price_cents,
                action="sell",
            )
            print(f"  Sold {count} {side} @ {price_cents}¢ [{ticker}]: {resp}")
            time.sleep(RATE_SLEEP)
        except Exception as e:
            print(f"  FAILED {ticker}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders (default: dry run)")
    args = parser.parse_args()
    main(dry_run=not args.live)
