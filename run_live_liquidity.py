"""
Live liquidity quoting bot for Kalshi's Liquidity Incentive Program.

Goal:
  - Place POST-ONLY resting buy orders (YES and NO bids) near fair value.
  - Refresh quotes on a schedule.
  - Keep inventory bounded per market.

This script is dry-run by default. Add --live to send orders.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
import glob
from zoneinfo import ZoneInfo

import pandas as pd

from src.kalshi_client import KalshiWeatherClient
from src.logger import init_run, log_trade, TRADE_LOG_PATH
from src.run_tracker import start_run, end_run

from run_live_simple import (
    STATIONS_FILE,
    _build_daily_distribution,
    _fair_prob_for_market,
    _fetch_weather_markets,
    _resolve_target_date,
)


def _to_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _round_down_to_cent(x: float) -> float:
    return math.floor(float(x) * 100.0 + 1e-9) / 100.0


def _market_thinness_key(m: dict) -> tuple[float, float, float]:
    """
    Lower key = thinner market (potentially easier share of displayed liquidity).
    """
    liq = _to_float(m.get("liquidity_dollars"), 1e9)
    vol = _to_float(m.get("volume_24h_fp", m.get("volume_fp")), 1e9)
    oi = _to_float(m.get("open_interest_fp"), 1e9)
    return (liq, vol, oi)


def _resting_orders_all(kalshi: KalshiWeatherClient, limit: int = 200) -> list[dict]:
    """
    Fetch all resting orders using cursor pagination.
    """
    out: list[dict] = []
    cursor = None
    for _ in range(50):
        resp = kalshi.get_orders(status="resting", limit=limit, cursor=cursor)
        rows = resp.get("orders", []) or []
        out.extend(rows)
        cursor = resp.get("cursor")
        if not cursor or not rows:
            break
    return out


def _extract_order_price(order: dict) -> float:
    yp = order.get("yes_price_dollars")
    np_ = order.get("no_price_dollars")
    if yp is not None:
        return _to_float(yp, 0.0)
    if np_ is not None:
        return _to_float(np_, 0.0)
    return 0.0


def _extract_remaining_count(order: dict) -> float:
    if order.get("remaining_count_fp") is not None:
        return _to_float(order.get("remaining_count_fp"), 0.0)
    if order.get("remaining_count") is not None:
        return _to_float(order.get("remaining_count"), 0.0)
    return 0.0


def _top_bid_from_orderbook_fp(orderbook: dict, side: str) -> tuple[float | None, float | None]:
    """
    Parse top bid price/size for `side` from Kalshi orderbook_fp payload.

    Expected shape:
      {"orderbook_fp": {"yes_dollars": [["0.03","12.00"], ...], "no_dollars": ...}}
    """
    fp = (orderbook or {}).get("orderbook_fp", {}) or {}
    levels = fp.get(f"{side}_dollars", []) or []
    if not levels:
        return None, None
    try:
        best = levels[-1]  # levels are sorted ascending by price.
        return float(best[0]), float(best[1])
    except Exception:
        return None, None


def _desired_bid_price(fair_p: float, ask: float, edge_buffer: float, tick: float, pmin: float, pmax: float) -> float | None:
    """
    Highest bid that is:
      1) at least edge_buffer below fair probability,
      2) at least one tick below current ask (stays non-crossing),
      3) within [pmin, pmax].
    """
    if not (0.0 < ask < 1.0):
        return None
    upper_non_cross = ask - tick
    upper_by_edge = fair_p - edge_buffer
    raw = min(upper_non_cross, upper_by_edge)
    px = _round_down_to_cent(raw)
    if px < pmin or px > pmax:
        return None
    if px <= 0.0:
        return None
    return px


def _positions_by_ticker(kalshi: KalshiWeatherClient) -> dict[str, float]:
    try:
        rows = kalshi.get_positions(limit=2000)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for r in rows:
        t = str(r.get("ticker", "") or "")
        if not t:
            continue
        out[t] = _to_float(r.get("position"), 0.0)
    return out


def _load_lip_exposure_from_logs() -> dict:
    """
    Estimate cumulative placed exposure for LIP bot from local trade logs.

    We use `action=place` and `status=placed` rows from *_LIP logs.
    Since this quoting bot only sends buy orders, cumulative filled contracts
    are a practical conservative proxy for gross inventory growth.
    """
    by_ticker_contracts: dict[str, int] = {}
    by_ticker_side_contracts: dict[tuple[str, str], int] = {}
    by_ticker_dollars: dict[str, float] = {}
    total_contracts = 0
    total_dollars = 0.0

    for p in sorted(glob.glob("logs/trade_log_*_LIP.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        placed = df[(df["action"] == "place") & (df["status"] == "placed")].copy()
        if placed.empty:
            continue
        for _, r in placed.iterrows():
            ticker = str(r.get("market_ticker", "") or "")
            side = str(r.get("side", "") or "").lower()
            if not ticker or side not in {"yes", "no"}:
                continue
            c = int(float(r.get("contract_count", 0) or 0))
            d = float(r.get("size_dollars", 0) or 0.0)
            if c <= 0:
                continue
            by_ticker_contracts[ticker] = by_ticker_contracts.get(ticker, 0) + c
            by_ticker_side_contracts[(ticker, side)] = by_ticker_side_contracts.get((ticker, side), 0) + c
            by_ticker_dollars[ticker] = by_ticker_dollars.get(ticker, 0.0) + max(d, 0.0)
            total_contracts += c
            total_dollars += max(d, 0.0)

    return {
        "by_ticker_contracts": by_ticker_contracts,
        "by_ticker_side_contracts": by_ticker_side_contracts,
        "by_ticker_dollars": by_ticker_dollars,
        "total_contracts": total_contracts,
        "total_dollars": total_dollars,
    }


def run_once(args: argparse.Namespace, dry_run: bool) -> None:
    target_date = _resolve_target_date(args)
    print(f"\n=== LIP Quote Bot | target={target_date} | mode={'DRY' if dry_run else 'LIVE'} ===")
    now_et = datetime.now(ZoneInfo("America/New_York"))
    flatten_mode = int(args.flatten_only_after_hour_et) >= 0 and now_et.hour >= int(args.flatten_only_after_hour_et)
    if flatten_mode:
        print(
            "Flatten mode: ON "
            f"(now_et={now_et.strftime('%Y-%m-%d %H:%M:%S')} "
            f"cutoff_hour={int(args.flatten_only_after_hour_et)})"
        )

    if not STATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing stations file: {STATIONS_FILE}")
    stations = pd.read_csv(STATIONS_FILE)
    if args.cities:
        city_set = {c.strip() for c in args.cities.split(",") if c.strip()}
        stations = stations[stations["city"].isin(city_set)].copy()
    if stations.empty:
        print("No stations selected.")
        return

    dist_daily = _build_daily_distribution(
        db_path=args.db,
        lookback_days=args.lookback_days,
        min_hist_days=args.min_hist_days,
        sigma_floor=args.sigma_floor,
        spread_alpha=args.spread_alpha,
        ecmwf_weight=args.ecmwf_weight,
    )

    kalshi = KalshiWeatherClient.from_env()
    auth = kalshi._client.kalshi_auth
    markets = _fetch_weather_markets(auth, stations, target_date)
    if args.tickers:
        tickers = {x.strip() for x in str(args.tickers).split(",") if x.strip()}
        markets = [m for m in markets if str(m.get("ticker", "")) in tickers]
    elif args.auto_pick_smallest:
        markets = sorted(markets, key=_market_thinness_key)
        if markets:
            top = markets[0]
            print(
                "Auto-picked smallest market: "
                f"{top.get('ticker')} "
                f"(liq=${_to_float(top.get('liquidity_dollars'), 0.0):.2f}, "
                f"vol24h={_to_float(top.get('volume_24h_fp', top.get('volume_fp')), 0.0):.2f}, "
                f"oi={_to_float(top.get('open_interest_fp'), 0.0):.2f})"
            )
            markets = [top]
    if int(args.max_markets) > 0:
        markets = markets[: int(args.max_markets)]
    print(f"Markets fetched: {len(markets)}")
    if not markets:
        return

    positions = _positions_by_ticker(kalshi)
    resting = _resting_orders_all(kalshi, limit=200)
    exposure = _load_lip_exposure_from_logs()

    print(
        "Exposure from logs: "
        f"placed_contracts_total={exposure['total_contracts']} "
        f"placed_dollars_total=${exposure['total_dollars']:.2f}"
    )
    print(
        "Risk caps: "
        f"max_filled_per_market={args.max_filled_contracts_per_market} "
        f"max_filled_per_side={args.max_filled_contracts_per_market_side} "
        f"max_filled_dollars_per_market=${args.max_filled_dollars_per_market:.2f} "
        f"max_filled_total={args.max_filled_contracts_total}"
    )

    # Keep only our resting BUY orders, bucketed by ticker+side.
    resting_by_key: dict[tuple[str, str], list[dict]] = {}
    for o in resting:
        if str(o.get("action", "")).lower() != "buy":
            continue
        ticker = str(o.get("ticker", "") or "")
        side = str(o.get("side", "") or "").lower()
        if not ticker or side not in {"yes", "no"}:
            continue
        resting_by_key.setdefault((ticker, side), []).append(o)

    actions = {"placed": 0, "canceled": 0, "kept": 0, "skipped": 0}
    tick = float(args.tick_size)
    quote_count = int(args.quote_count)
    placed_this_cycle = 0

    for m in markets:
        if int(args.max_new_orders_per_cycle) > 0 and placed_this_cycle >= int(args.max_new_orders_per_cycle):
            break
        ticker = str(m.get("ticker", ""))
        city = str(m.get("_city", ""))
        yes_ask = _to_float(m.get("yes_ask_dollars"), 0.0)
        no_ask = _to_float(m.get("no_ask_dollars"), 0.0)
        if not (0.0 < yes_ask < 1.0 and 0.0 < no_ask < 1.0):
            actions["skipped"] += 1
            continue
        best_yes_bid = _to_float(m.get("yes_bid_dollars"), 0.0)
        best_no_bid = _to_float(m.get("no_bid_dollars"), 0.0)
        if bool(args.use_orderbook_competitive) or bool(args.print_orderbook_summary):
            try:
                ob = kalshi.get_orderbook(ticker=ticker, depth=int(args.orderbook_depth))
                yb, _ = _top_bid_from_orderbook_fp(ob, "yes")
                nb, _ = _top_bid_from_orderbook_fp(ob, "no")
                if yb is not None:
                    best_yes_bid = float(yb)
                if nb is not None:
                    best_no_bid = float(nb)
            except Exception as e:
                print(f"  Orderbook fetch failed {ticker}: {e}")
        if bool(args.print_orderbook_summary):
            pair_ask = yes_ask + no_ask
            pair_bid = (best_yes_bid + best_no_bid) if (best_yes_bid > 0 and best_no_bid > 0) else None
            if pair_bid is None:
                print(f"  Book {ticker}: yes_bid={best_yes_bid:.2f} no_bid={best_no_bid:.2f} pair_ask={pair_ask:.2f}")
            else:
                print(
                    f"  Book {ticker}: yes_bid={best_yes_bid:.2f} no_bid={best_no_bid:.2f} "
                    f"pair_bid={pair_bid:.2f} pair_ask={pair_ask:.2f}"
                )

        fair = _fair_prob_for_market(
            market=m,
            city=city,
            target_date=target_date,
            dist_daily=dist_daily,
            prob_floor=args.prob_floor,
            prob_ceil=args.prob_ceil,
            confidence_shrink_k=args.confidence_shrink_k,
        )
        if fair is None:
            actions["skipped"] += 1
            continue
        fair_yes, _ = fair
        fair_no = 1.0 - fair_yes

        desired_yes = _desired_bid_price(
            fair_p=fair_yes,
            ask=yes_ask,
            edge_buffer=float(args.edge_buffer),
            tick=tick,
            pmin=float(args.min_price),
            pmax=float(args.max_price),
        )
        desired_no = _desired_bid_price(
            fair_p=fair_no,
            ask=no_ask,
            edge_buffer=float(args.edge_buffer),
            tick=tick,
            pmin=float(args.min_price),
            pmax=float(args.max_price),
        )
        desired_yes_raw = desired_yes
        desired_no_raw = desired_no
        if bool(args.use_orderbook_competitive):
            max_gap = float(args.max_gap_to_best_bid)
            if desired_yes is not None and best_yes_bid > 0 and max_gap >= 0 and desired_yes < (best_yes_bid - max_gap):
                desired_yes = None
            if desired_no is not None and best_no_bid > 0 and max_gap >= 0 and desired_no < (best_no_bid - max_gap):
                desired_no = None
        if bool(args.print_orderbook_summary):
            print(
                f"  Target {ticker}: fair_yes={fair_yes:.3f} fair_no={fair_no:.3f} "
                f"desired_raw=(yes:{desired_yes_raw}, no:{desired_no_raw}) "
                f"desired_after_filters=(yes:{desired_yes}, no:{desired_no})"
            )

        net_yes = _to_float(positions.get(ticker), 0.0)
        max_net = float(args.max_net_contracts_per_market)

        if bool(args.require_two_sided) and (desired_yes is None or desired_no is None):
            if bool(args.print_orderbook_summary):
                print(
                    "  Skip reason: require-two-sided enabled but one side was filtered "
                    f"(yes={desired_yes is not None}, no={desired_no is not None})."
                )
            actions["skipped"] += 1
            continue
        if desired_yes is not None and desired_no is not None:
            pair_cost = float(desired_yes) + float(desired_no)
            max_pair_cost = float(args.max_pair_cost)
            if max_pair_cost > 0 and pair_cost > max_pair_cost:
                if bool(args.print_orderbook_summary):
                    print(
                        "  Skip reason: pair-cost cap violated "
                        f"(pair_cost={pair_cost:.2f} > max_pair_cost={max_pair_cost:.2f})."
                    )
                actions["skipped"] += 1
                continue

        allowed_sides = {"yes", "no"}
        if flatten_mode:
            if net_yes > 0:
                allowed_sides = {"no"}
            elif net_yes < 0:
                allowed_sides = {"yes"}
            else:
                # Don't open fresh exposure during flatten-only window.
                allowed_sides = set()

        for side, desired_px in [("yes", desired_yes), ("no", desired_no)]:
            if int(args.max_new_orders_per_cycle) > 0 and placed_this_cycle >= int(args.max_new_orders_per_cycle):
                break
            if side not in allowed_sides:
                continue
            if desired_px is None:
                continue

            # Inventory guard.
            if side == "yes" and net_yes >= max_net:
                continue
            if side == "no" and (-net_yes) >= max_net:
                continue

            # Hard exposure caps from historical placed-order logs.
            filled_total = int(exposure["total_contracts"])
            if int(args.max_filled_contracts_total) > 0 and filled_total + quote_count > int(args.max_filled_contracts_total):
                continue
            filled_mkt = int(exposure["by_ticker_contracts"].get(ticker, 0))
            if int(args.max_filled_contracts_per_market) > 0 and filled_mkt + quote_count > int(args.max_filled_contracts_per_market):
                continue
            filled_side = int(exposure["by_ticker_side_contracts"].get((ticker, side), 0))
            if int(args.max_filled_contracts_per_market_side) > 0 and filled_side + quote_count > int(args.max_filled_contracts_per_market_side):
                continue
            filled_dol = float(exposure["by_ticker_dollars"].get(ticker, 0.0))
            est_dol = quote_count * desired_px
            if float(args.max_filled_dollars_per_market) > 0 and filled_dol + est_dol > float(args.max_filled_dollars_per_market):
                continue

            key = (ticker, side)
            existing = resting_by_key.get(key, [])
            # Keep one correctly-priced order; cancel stale/excess orders.
            keep_order = None
            for o in existing:
                px = _extract_order_price(o)
                rem = _extract_remaining_count(o)
                if abs(px - desired_px) < 1e-9 and rem >= quote_count:
                    keep_order = o
                    break

            if keep_order is not None:
                actions["kept"] += 1
                # cancel extras at this key
                for o in existing:
                    if o is keep_order:
                        continue
                    oid = str(o.get("order_id", "") or "")
                    if not oid:
                        continue
                    if dry_run:
                        print(f"  DRY cancel extra {ticker} {side} order_id={oid}")
                    else:
                        try:
                            kalshi.cancel_order(oid)
                            actions["canceled"] += 1
                        except Exception as e:
                            print(f"  Cancel failed {ticker} {side} {oid}: {e}")
                continue

            # Cancel all stale orders for this ticker/side before posting fresh quote.
            for o in existing:
                oid = str(o.get("order_id", "") or "")
                if not oid:
                    continue
                if dry_run:
                    print(f"  DRY cancel stale {ticker} {side} order_id={oid}")
                else:
                    try:
                        kalshi.cancel_order(oid)
                        actions["canceled"] += 1
                    except Exception as e:
                        print(f"  Cancel failed {ticker} {side} {oid}: {e}")

            if dry_run:
                print(
                    f"  DRY quote {ticker} [{city}] {side} x{quote_count} @ {desired_px:.2f} "
                    f"(fair_yes={fair_yes:.3f} fair_no={fair_no:.3f})"
                )
                actions["placed"] += 1
                placed_this_cycle += 1
                continue

            try:
                resp = kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    count=quote_count,
                    price=int(round(desired_px * 100)),
                    post_only=True,
                )
                print(f"  Quote placed {ticker} {side} x{quote_count} @ {desired_px:.2f}: {resp}")
                log_trade(
                    market_ticker=ticker,
                    city=city,
                    market_type="liquidity_quote",
                    contract_desc=str(m.get("title", "")),
                    fair_p=fair_yes,
                    yes_ask=desired_px if side == "yes" else 0,
                    no_ask=desired_px if side == "no" else 0,
                    edge_yes=fair_yes - yes_ask,
                    edge_no=fair_no - no_ask,
                    action="place",
                    side=side,
                    size_dollars=quote_count * desired_px,
                    contract_count=quote_count,
                    status="placed",
                    notes="post_only",
                )
                actions["placed"] += 1
                placed_this_cycle += 1
                exposure["total_contracts"] = int(exposure["total_contracts"]) + quote_count
                exposure["total_dollars"] = float(exposure["total_dollars"]) + (quote_count * desired_px)
                exposure["by_ticker_contracts"][ticker] = int(exposure["by_ticker_contracts"].get(ticker, 0)) + quote_count
                exposure["by_ticker_side_contracts"][(ticker, side)] = int(
                    exposure["by_ticker_side_contracts"].get((ticker, side), 0)
                ) + quote_count
                exposure["by_ticker_dollars"][ticker] = float(exposure["by_ticker_dollars"].get(ticker, 0.0)) + (
                    quote_count * desired_px
                )
            except Exception as e:
                print(f"  Quote failed {ticker} {side} @ {desired_px:.2f}: {e}")
                log_trade(
                    market_ticker=ticker,
                    city=city,
                    market_type="liquidity_quote",
                    contract_desc=str(m.get("title", "")),
                    fair_p=fair_yes,
                    yes_ask=desired_px if side == "yes" else 0,
                    no_ask=desired_px if side == "no" else 0,
                    edge_yes=fair_yes - yes_ask,
                    edge_no=fair_no - no_ask,
                    action="place",
                    side=side,
                    size_dollars=quote_count * desired_px,
                    contract_count=quote_count,
                    status="failed",
                    notes=str(e),
                )

    print(
        "Cycle summary: "
        f"placed={actions['placed']} kept={actions['kept']} "
        f"canceled={actions['canceled']} skipped={actions['skipped']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Place real orders. Default dry-run.")
    ap.add_argument("--interval", type=int, default=60, help="Minutes between quote refresh cycles.")
    ap.add_argument("--db", default="data/weather.db")
    ap.add_argument("--target-date", default=None, help="YYYY-MM-DD explicit target.")
    ap.add_argument("--trade-today", action="store_true", help="Force today's settlement markets.")
    ap.add_argument("--target-date-cutoff-hour", type=int, default=16)
    ap.add_argument("--cities", default=None, help="Comma list of cities.")
    ap.add_argument("--tickers", default=None, help="Optional comma list of exact tickers to quote.")
    ap.add_argument("--auto-pick-smallest", action="store_true", help="Pick the thinnest market by liquidity/volume/OI.")
    ap.add_argument("--max-markets", type=int, default=0, help="0 = no cap, otherwise max markets per cycle.")

    # Quote controls
    ap.add_argument("--quote-count", type=int, default=1, help="Contracts per posted quote.")
    ap.add_argument("--edge-buffer", type=float, default=0.02, help="Require fair_p - quote >= edge_buffer.")
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--min-price", type=float, default=0.03)
    ap.add_argument("--max-price", type=float, default=0.97)
    ap.add_argument(
        "--use-orderbook-competitive",
        action="store_true",
        help="Only quote when desired bid is within --max-gap-to-best-bid of live best bid (per side).",
    )
    ap.add_argument(
        "--max-gap-to-best-bid",
        type=float,
        default=0.03,
        help="With --use-orderbook-competitive, drop side if desired_bid < best_bid - gap.",
    )
    ap.add_argument("--orderbook-depth", type=int, default=20)
    ap.add_argument(
        "--print-orderbook-summary",
        action="store_true",
        help="Print top bid/ask pair diagnostics each cycle.",
    )
    ap.add_argument(
        "--require-two-sided",
        action="store_true",
        help="Only quote markets where both YES and NO quotes are available this cycle.",
    )
    ap.add_argument(
        "--max-pair-cost",
        type=float,
        default=0.0,
        help="If >0, require (yes_quote + no_quote) <= this cap before quoting.",
    )
    ap.add_argument(
        "--flatten-only-after-hour-et",
        type=int,
        default=-1,
        help="If >=0, after this ET hour only quote the side that reduces net inventory; flat markets get no new quotes.",
    )
    ap.add_argument("--max-net-contracts-per-market", type=float, default=10.0)
    ap.add_argument("--max-filled-contracts-per-market", type=int, default=10)
    ap.add_argument("--max-filled-contracts-per-market-side", type=int, default=6)
    ap.add_argument("--max-filled-dollars-per-market", type=float, default=10.0)
    ap.add_argument("--max-filled-contracts-total", type=int, default=100)
    ap.add_argument("--max-new-orders-per-cycle", type=int, default=2)

    # Forecast calibration controls
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--min-hist-days", type=int, default=45)
    ap.add_argument("--sigma-floor", type=float, default=3.0)
    ap.add_argument("--spread-alpha", type=float, default=0.20)
    ap.add_argument("--ecmwf-weight", type=float, default=0.50)
    ap.add_argument("--prob-floor", type=float, default=0.03)
    ap.add_argument("--prob-ceil", type=float, default=0.97)
    ap.add_argument("--confidence-shrink-k", type=float, default=60.0)
    args = ap.parse_args()
    if float(args.max_pair_cost) > 0 and not (0.0 < float(args.max_pair_cost) <= 1.0):
        raise ValueError("--max-pair-cost must be in (0, 1] when provided.")
    if int(args.flatten_only_after_hour_et) < -1 or int(args.flatten_only_after_hour_et) > 23:
        raise ValueError("--flatten-only-after-hour-et must be -1 or an ET hour in [0, 23].")
    if int(args.orderbook_depth) <= 0:
        raise ValueError("--orderbook-depth must be >= 1.")
    if float(args.max_gap_to_best_bid) < 0 and bool(args.use_orderbook_competitive):
        raise ValueError("--max-gap-to-best-bid must be >= 0 when --use-orderbook-competitive is enabled.")

    init_run(model_name="LIP")
    print("Mode:", "LIVE" if args.live else "DRY RUN")

    run_id = start_run("run_live_liquidity.py", vars(args))
    try:
        while True:
            try:
                run_once(args, dry_run=not args.live)
            except Exception as e:
                print(f"Cycle error: {e}")
            if args.interval <= 0:
                break
            next_ts = datetime.now() + timedelta(minutes=int(args.interval))
            print(f"Sleeping {args.interval} min (next run ~{next_ts.strftime('%Y-%m-%d %H:%M:%S')})")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        end_run(run_id, trade_log_path=TRADE_LOG_PATH)


if __name__ == "__main__":
    main()
