"""
Live integration test for Kalshi portfolio API methods.

Tests get_fills(), get_settlements(), get_positions() and validates
that the returned data has the expected structure.

Requires valid credentials in .env:
  KALSHI_API_KEY_ID
  KALSHI_PRIVATE_KEY_PATH

Run from project root:
  python scripts/test_kalshi_portfolio.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kalshi_client import KalshiWeatherClient

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[34mINFO\033[0m"

_failures = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if not condition:
        if detail:
            print(f"         {detail}")
        _failures.append(label)


def section(title: str):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

section("Connect to Kalshi")
try:
    client = KalshiWeatherClient.from_env()
    print(f"  [{PASS}] Client constructed")
except Exception as e:
    print(f"  [{FAIL}] Could not build client: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# get_fills
# ---------------------------------------------------------------------------

section("get_fills()")
try:
    fills = client.get_fills(limit=50)
    check("Returns a list", isinstance(fills, list))
    print(f"  [{INFO}] {len(fills)} fill(s) returned")

    if fills:
        f0 = fills[0]
        print(f"  [{INFO}] First fill keys: {list(f0.keys())}")
        check("Fill has 'ticker'",        "ticker" in f0)
        check("Fill has 'side'",          "side" in f0)
        check("Fill has 'count'",         "count" in f0)
        check("Fill has 'created_time'",  "created_time" in f0)

        print(f"\n  Most recent fill:")
        print(json.dumps({
            k: str(v) for k, v in f0.items()
            if k in ("ticker", "side", "count", "yes_price", "no_price", "created_time", "trade_id")
        }, indent=4))
    else:
        print(f"  [{INFO}] No fills yet — skipping field checks")

except Exception as e:
    check("get_fills() raised no exception", False, str(e))


# ---------------------------------------------------------------------------
# get_fills — filter by ticker (if we have at least one fill)
# ---------------------------------------------------------------------------

section("get_fills(ticker=...) — filtered")
try:
    all_fills = client.get_fills(limit=200)
    if all_fills:
        ticker = all_fills[0]["ticker"]
        filtered = client.get_fills(ticker=ticker, limit=10)
        check("Returns a list", isinstance(filtered, list))
        if filtered:
            check(
                f"All returned fills have ticker={ticker!r}",
                all(f.get("ticker") == ticker for f in filtered),
                f"tickers found: {list(set(f.get('ticker') for f in filtered))}"
            )
        print(f"  [{INFO}] {len(filtered)} fill(s) for ticker {ticker!r}")
    else:
        print(f"  [{INFO}] No fills to filter on — skipping")
except Exception as e:
    check("get_fills(ticker) raised no exception", False, str(e))


# ---------------------------------------------------------------------------
# get_settlements
# ---------------------------------------------------------------------------

section("get_settlements()")
try:
    settlements = client.get_settlements(limit=50)
    check("Returns a list", isinstance(settlements, list))
    print(f"  [{INFO}] {len(settlements)} settlement(s) returned")

    if settlements:
        s0 = settlements[0]
        print(f"  [{INFO}] First settlement keys: {list(s0.keys())}")
        check("Settlement has 'ticker'",        "ticker" in s0)
        check("Settlement has 'revenue'",        "revenue" in s0)
        check("Settlement has 'settled_time'",   "settled_time" in s0)

        # Compute P&L summary across all settlements
        total_revenue = 0.0
        for s in settlements:
            rev = s.get("revenue") or s.get("profit") or 0
            try:
                total_revenue += float(rev)
            except (TypeError, ValueError):
                pass

        print(f"\n  Most recent settlement:")
        print(json.dumps({
            k: str(v) for k, v in s0.items()
            if k in ("ticker", "revenue", "profit", "no_count", "yes_count",
                      "market_result", "settled_time")
        }, indent=4))

        print(f"\n  [{INFO}] Total revenue across {len(settlements)} settlements: ${total_revenue:.2f}")

        # Per-ticker P&L summary
        by_ticker: dict[str, float] = {}
        for s in settlements:
            t = s.get("ticker", "unknown")
            rev = float(s.get("revenue") or s.get("profit") or 0)
            by_ticker[t] = by_ticker.get(t, 0.0) + rev

        if by_ticker:
            print(f"\n  P&L by ticker (last {len(settlements)} settlements):")
            for t, pnl in sorted(by_ticker.items(), key=lambda x: -abs(x[1])):
                sign = "+" if pnl >= 0 else ""
                print(f"    {t:40s}  {sign}{pnl:.2f}")
    else:
        print(f"  [{INFO}] No settlements yet — skipping field checks")

except Exception as e:
    check("get_settlements() raised no exception", False, str(e))


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------

section("get_positions()")
try:
    positions = client.get_positions(limit=100)
    check("Returns a list", isinstance(positions, list))
    print(f"  [{INFO}] {len(positions)} position(s)")

    if positions:
        p0 = positions[0]
        print(f"  [{INFO}] First position keys: {list(p0.keys())}")
        check("Position has 'ticker'",   "ticker" in p0)
        check("Position has 'position'", "position" in p0)

        total_rpnl = sum(float(p.get("realized_pnl") or 0) for p in positions)
        total_fees = sum(float(p.get("fees_paid") or 0) for p in positions)
        print(f"  [{INFO}] Realized P&L: ${total_rpnl:.2f}  Fees: ${total_fees:.2f}  Net: ${total_rpnl - total_fees:.2f}")

        print(f"\n  Positions:")
        for p in positions:
            t    = p.get("ticker", "?")
            pos  = p.get("position", 0)
            exp  = p.get("market_exposure", 0)
            rpnl = p.get("realized_pnl", 0)
            fees = p.get("fees_paid", 0)
            print(f"    {t:40s}  pos={pos:+d}  exposure=${float(exp or 0):.2f}  "
                  f"realized_pnl=${float(rpnl or 0):.2f}  fees=${float(fees or 0):.2f}")
    else:
        print(f"  [{INFO}] No positions")

except Exception as e:
    check("get_positions() raised no exception", False, str(e))


# ---------------------------------------------------------------------------
# Cross-check: fills vs settlements
# ---------------------------------------------------------------------------

section("Cross-check fills vs settlements")
try:
    all_fills        = client.get_fills(limit=200)
    all_settlements  = client.get_settlements(limit=200)

    fill_tickers       = {f.get("ticker") for f in all_fills}
    settlement_tickers = {s.get("ticker") for s in all_settlements}
    overlap            = fill_tickers & settlement_tickers

    print(f"  [{INFO}] Tickers with fills:       {len(fill_tickers)}")
    print(f"  [{INFO}] Tickers with settlements: {len(settlement_tickers)}")
    print(f"  [{INFO}] Overlap (traded + settled): {len(overlap)}")

    if overlap:
        print(f"\n  Sample settled tickers from your fills:")
        for t in sorted(overlap)[:10]:
            print(f"    {t}")

except Exception as e:
    print(f"  [{INFO}] Cross-check skipped: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if _failures:
    print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[32mAll checks passed.\033[0m")
