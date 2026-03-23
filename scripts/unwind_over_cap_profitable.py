"""
Unwind oversized Kalshi positions, but only at a profit.

Policy:
  - Only consider positions with exposure > --max-exposure (default: $50).
  - Compute average entry price from open-position exposure:
      avg_entry = market_exposure_dollars / abs(position_fp)
  - Sell only enough contracts to bring exposure back toward the cap.
  - Only place a sell order when current bid > avg_entry + min-profit buffer.

Usage:
  python scripts/unwind_over_cap_profitable.py
  python scripts/unwind_over_cap_profitable.py --live
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import KALSHI_BASE_URL
from src.kalshi_client import KalshiWeatherClient


RATE_SLEEP = 0.25


def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _extract_http_status(exc: Exception) -> int | None:
    for attr in ("status", "status_code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return int(v)
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    m = re.search(r"\((\d{3})\)", str(exc))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def kalshi_get_raw(auth, path: str, params: dict | None = None) -> dict:
    url = KALSHI_BASE_URL.rstrip("/") + path
    headers = auth.create_auth_headers("GET", path.split("?")[0])
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_market_snapshot(auth, ticker: str) -> dict | None:
    try:
        data = kalshi_get_raw(auth, f"/markets/{ticker}")
        m = data.get("market", data)
        return {
            "yes_bid": _to_float(m.get("yes_bid_dollars"), 0.0),
            "yes_ask": _to_float(m.get("yes_ask_dollars"), 0.0),
            "no_bid": _to_float(m.get("no_bid_dollars"), 0.0),
            "no_ask": _to_float(m.get("no_ask_dollars"), 0.0),
        }
    except Exception as e:
        print(f"  {ticker}: quote fetch failed ({e})")
        return None


def _best_bid_for_side(snapshot: dict, side: str) -> float:
    if side == "yes":
        bid = _to_float(snapshot.get("yes_bid"), 0.0)
        if bid > 0:
            return bid
        no_ask = _to_float(snapshot.get("no_ask"), 0.0)
        return max(0.0, 1.0 - no_ask) if 0.0 < no_ask < 1.0 else 0.0

    bid = _to_float(snapshot.get("no_bid"), 0.0)
    if bid > 0:
        return bid
    yes_ask = _to_float(snapshot.get("yes_ask"), 0.0)
    return max(0.0, 1.0 - yes_ask) if 0.0 < yes_ask < 1.0 else 0.0


def _planned_sell_count(
    exposure: float,
    avg_entry: float,
    held_count: int,
    max_exposure: float,
) -> int:
    if exposure <= max_exposure:
        return 0
    excess = exposure - max_exposure
    if avg_entry <= 0:
        return 0
    need = int(math.ceil(excess / avg_entry))
    return max(1, min(int(held_count), int(need)))


def _try_place_unwind_sell(
    kalshi: KalshiWeatherClient,
    ticker: str,
    side: str,
    sell_count: int,
    bid_cents: int,
    fok: bool,
) -> tuple[bool, str]:
    """
    Try placing an unwind sell. If a 400 occurs, retry with reduced flags.
    """
    attempts: list[tuple[str, dict]] = []
    if bool(fok):
        attempts.append(("reduce_only+fok", {"reduce_only": True, "time_in_force": "fill_or_kill"}))
    attempts.append(("reduce_only", {"reduce_only": True}))
    attempts.append(("plain", {}))

    last_err = "unknown_error"
    for label, kwargs in attempts:
        try:
            kalshi.place_order(
                ticker=ticker,
                side=side,
                count=int(sell_count),
                price=int(bid_cents),
                action="sell",
                **kwargs,
            )
            return True, label
        except Exception as e:
            last_err = str(e)
            status = _extract_http_status(e)
            if status == 400:
                continue
            return False, f"{label}:{last_err}"
    return False, last_err


def run_once(
    dry_run: bool,
    max_exposure: float,
    min_profit: float,
    min_profit_pct: float,
    fok: bool,
) -> None:
    print(
        "=== Unwind Oversized Positions "
        f"{'(DRY RUN)' if dry_run else '*** LIVE ***'} ==="
    )
    print(
        f"Policy: exposure cap=${max_exposure:.2f}, "
        f"min_profit={min_profit:.3f} per contract, "
        f"min_profit_pct={100.0 * float(min_profit_pct):.2f}%\n"
    )

    kalshi = KalshiWeatherClient.from_env()
    auth = kalshi._client.kalshi_auth
    positions = kalshi.get_positions(limit=1000)
    if not positions:
        print("No positions returned.")
        return

    candidates: list[dict] = []
    for p in positions:
        ticker = str(p.get("ticker", "") or "")
        if not ticker:
            continue

        net = _to_float(p.get("position_fp"), _to_float(p.get("position"), 0.0))
        held_count = int(math.floor(abs(net) + 1e-9))
        if held_count <= 0:
            continue

        side = "yes" if net > 0 else "no"
        exposure = _to_float(
            p.get("market_exposure_dollars"),
            _to_float(p.get("market_exposure"), 0.0),
        )
        if exposure <= float(max_exposure) + 1e-9:
            continue

        avg_entry = exposure / float(held_count) if held_count > 0 else 0.0
        if avg_entry <= 0:
            continue

        snapshot = fetch_market_snapshot(auth, ticker)
        time.sleep(RATE_SLEEP)
        if snapshot is None:
            continue
        bid = _best_bid_for_side(snapshot, side)
        if not (0.0 < bid < 1.0):
            print(f"  {ticker}: no valid bid for side={side}.")
            continue

        sell_count = _planned_sell_count(
            exposure=exposure,
            avg_entry=avg_entry,
            held_count=held_count,
            max_exposure=float(max_exposure),
        )
        if sell_count <= 0:
            continue

        per_contract_profit = bid - avg_entry
        required_bid = max(
            float(avg_entry) + float(min_profit),
            float(avg_entry) * (1.0 + float(min_profit_pct)),
        )
        if bid <= required_bid + 1e-12:
            print(
                f"  {ticker}: over cap (${exposure:.2f}) but not profitable to unwind "
                f"(bid={bid:.3f} <= required={required_bid:.3f})."
            )
            continue

        candidates.append(
            {
                "ticker": ticker,
                "side": side,
                "held_count": held_count,
                "sell_count": int(sell_count),
                "bid": float(bid),
                "bid_cents": int(round(float(bid) * 100)),
                "avg_entry": float(avg_entry),
                "exposure": float(exposure),
                "projected_profit": float(per_contract_profit) * float(sell_count),
            }
        )

    if not candidates:
        print("No profitable over-cap unwind opportunities found.")
        return

    candidates = sorted(candidates, key=lambda x: x["projected_profit"], reverse=True)
    print(f"Planned unwind orders: {len(candidates)}\n")
    for c in candidates:
        print(
            f"  {c['ticker']} side={c['side']} held={c['held_count']} sell={c['sell_count']} "
            f"exp=${c['exposure']:.2f} avg={c['avg_entry']:.3f} bid={c['bid']:.3f} "
            f"proj_profit=${c['projected_profit']:.2f}"
        )

    if dry_run:
        print("\nDry run only. Re-run with --live to place orders.")
        return

    print("\nPlacing orders...")
    for c in candidates:
        try:
            ok, mode = _try_place_unwind_sell(
                kalshi=kalshi,
                ticker=str(c["ticker"]),
                side=str(c["side"]),
                sell_count=int(c["sell_count"]),
                bid_cents=int(c["bid_cents"]),
                fok=bool(fok),
            )
            if not ok:
                print(f"  FAILED {c['ticker']} ({mode})")
                time.sleep(RATE_SLEEP)
                continue
            print(
                f"  SOLD {c['ticker']} {c['side']} x{c['sell_count']} @ {c['bid_cents']}¢ "
                f"(proj_profit=${c['projected_profit']:.2f}, mode={mode})"
            )
        except Exception as e:
            print(f"  FAILED {c['ticker']} ({e})")
        time.sleep(RATE_SLEEP)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real unwind orders (default: dry run).")
    parser.add_argument("--max-exposure", type=float, default=50.0, help="Per-ticker exposure cap in dollars.")
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.0,
        help="Required per-contract bid premium over average entry (dollars).",
    )
    parser.add_argument(
        "--min-profit-pct",
        type=float,
        default=0.04,
        help="Required profit percentage over average entry (e.g., 0.04 = 4%%).",
    )
    parser.add_argument(
        "--fill-or-kill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place unwind sells as FOK (default false; can reduce 400 conflicts).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=30,
        help="Run repeatedly every N seconds (0 = run once).",
    )
    args = parser.parse_args()
    if float(args.max_exposure) < 0:
        raise ValueError("--max-exposure must be >= 0.")
    if float(args.min_profit) < 0:
        raise ValueError("--min-profit must be >= 0.")
    if float(args.min_profit_pct) < 0:
        raise ValueError("--min-profit-pct must be >= 0.")
    if int(args.interval_seconds) < 0:
        raise ValueError("--interval-seconds must be >= 0.")

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unwind cycle")
        run_once(
            dry_run=not args.live,
            max_exposure=float(args.max_exposure),
            min_profit=float(args.min_profit),
            min_profit_pct=float(args.min_profit_pct),
            fok=bool(args.fill_or_kill),
        )
        if int(args.interval_seconds) <= 0:
            break
        next_ts = datetime.now() + timedelta(seconds=int(args.interval_seconds))
        print(
            f"Sleeping {int(args.interval_seconds)}s "
            f"(next run ~{next_ts.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        time.sleep(int(args.interval_seconds))
