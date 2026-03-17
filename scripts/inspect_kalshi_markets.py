"""
Phase 3: Connect read-only to Kalshi and inspect available weather markets.

This is a discovery script — run it to find the series tickers and contract
formats for cities you want to trade. Log everything so you can build the
market_definitions.json.

Run from project root:
  python scripts/inspect_kalshi_markets.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kalshi_client import KalshiWeatherClient


def main():
    print("Connecting to Kalshi...")
    client = KalshiWeatherClient.from_env()

    print("Fetching open markets (no series filter)...")
    # Adjust series_ticker to known weather series once you find them
    # Common weather series tickers on Kalshi include things like "HIGHNY", "HIGHCHI"
    # Inspect the full list first:
    markets = client.get_weather_markets(status="open", limit=200)

    print(f"Total open markets returned: {len(markets)}")

    # Filter to weather markets by title keywords
    weather_markets = [
        m for m in markets
        if any(kw in (m.get("title") or "").lower()
               for kw in ["temperature", "high temp", "low temp", "weather", "°f", "degrees"])
        or any(kw in (m.get("event_ticker") or "").upper()
               for kw in ["HIGH", "LOW", "TEMP", "WEATHER"])
    ]
    print(f"Weather-related markets found: {len(weather_markets)}")
    if not weather_markets:
        print("(No weather markets in this page — showing first 5 markets for schema reference)")
        weather_markets = markets[:5]

    print("\nSample weather markets:")
    for m in weather_markets[:10]:
        print(json.dumps({
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "title": m.get("title"),
            "yes_sub_title": m.get("yes_sub_title"),
            "yes_ask_dollars": m.get("yes_ask_dollars"),
            "no_ask_dollars": m.get("no_ask_dollars"),
            "close_time": str(m.get("close_time") or m.get("expiration_time")),
        }, indent=2))

    # Save full snapshot for inspection
    out_path = Path("data/kalshi_markets/market_snapshot.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(markets, f, indent=2, default=str)

    print(f"\nFull snapshot saved to {out_path}")
    print("Review this file to identify series tickers and contract formats.")
    print("Then update data/market_definitions.json accordingly.")


if __name__ == "__main__":
    main()
