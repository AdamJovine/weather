"""
Trade decision and sizing logic.

The decision function is intentionally conservative.
Log recommendations before placing any real orders.

Kalshi weather market price format:
  - yes_ask_dollars / no_ask_dollars: float in [0.0, 1.0]
  - Each contract pays $1 if YES settles true
  - Fractional contracts are supported (fractional_trading_enabled=True)

Observed Kalshi weather title formats (as of March 2026):
  ">30°"    → geq, threshold=31  (strictly greater than 30)
  "<23°"    → leq, threshold=22  (strictly less than 23)
  "29-30°"  → range, low=29, high=30
"""

import re
import pandas as pd

from src.config import TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER, MAX_BET_FRACTION, KELLY_FRACTION
from src.pricing import compute_fair_prob, dollars_to_prob


# ------------------------------------------------------------------
# Contract parsing
# ------------------------------------------------------------------

def parse_contract(title: str, subtitle: str = "") -> dict:
    """
    Map a Kalshi weather market title to a structured contract definition.

    Handles the live Kalshi formats observed:
      "Will the high temp in Chicago be >30° on Mar 17, 2026?"
      "Will the high temp in Chicago be <23° on Mar 17, 2026?"
      "Will the high temp in Chicago be 29-30° on Mar 17, 2026?"

    For > and < contracts, the threshold is the integer after the symbol,
    and the contract settles on strict inequality:
      >30 → tmax >= 31 → geq(31)
      <23 → tmax <= 22 → leq(22)

    Returns dict with keys: market_type, and type-specific fields.
    Raises ValueError if no pattern matches.
    """
    text = f"{title} {subtitle}".lower()

    # ">30°" or "> 30°" — strictly greater than N → geq(N+1)
    m = re.search(r"be\s*>\s*(\d+)\s*°", text)
    if m:
        return {"market_type": "geq", "threshold": int(m.group(1)) + 1}

    # "<23°" — strictly less than N → leq(N-1)
    m = re.search(r"be\s*<\s*(\d+)\s*°", text)
    if m:
        return {"market_type": "leq", "threshold": int(m.group(1)) - 1}

    # "29-30°" — range with hyphen
    m = re.search(r"be\s*(\d+)-(\d+)\s*°", text)
    if m:
        return {"market_type": "range", "low": int(m.group(1)), "high": int(m.group(2))}

    # "59° to 60°" — range with "to" (actual Kalshi format)
    m = re.search(r"(\d+)\s*°?\s*to\s*(\d+)\s*°", text)
    if m:
        return {"market_type": "range", "low": int(m.group(1)), "high": int(m.group(2))}

    # Fallback generic patterns
    m = re.search(r"at least\s*(\d+)", text)
    if m:
        return {"market_type": "geq", "threshold": int(m.group(1))}

    m = re.search(r"(\d+)\s*(?:°f?\s*)?or (?:higher|above)", text)
    if m:
        return {"market_type": "geq", "threshold": int(m.group(1))}

    m = re.search(r"at most\s*(\d+)", text)
    if m:
        return {"market_type": "leq", "threshold": int(m.group(1))}

    m = re.search(r"(\d+)\s*(?:°f?\s*)?or (?:lower|below)", text)
    if m:
        return {"market_type": "leq", "threshold": int(m.group(1))}

    raise ValueError(f"Could not parse contract text: {text!r}")


# ------------------------------------------------------------------
# Decision functions — prices in dollars [0.0, 1.0]
# ------------------------------------------------------------------

def should_buy_yes(
    fair_p: float,
    yes_ask_dollars: float,
    fee_buffer: float = TRADE_FEE_BUFFER,
    model_buffer: float = TRADE_MODEL_BUFFER,
) -> bool:
    return (fair_p - yes_ask_dollars) > (fee_buffer + model_buffer)


def should_buy_no(
    fair_p: float,
    no_ask_dollars: float,
    fee_buffer: float = TRADE_FEE_BUFFER,
    model_buffer: float = TRADE_MODEL_BUFFER,
) -> bool:
    fair_no = 1.0 - fair_p
    return (fair_no - no_ask_dollars) > (fee_buffer + model_buffer)


# ------------------------------------------------------------------
# Sizing
# ------------------------------------------------------------------

def simple_bet_size(
    edge: float,
    bankroll: float,
    max_fraction: float = MAX_BET_FRACTION,
    kelly_fraction: float = KELLY_FRACTION,
    city_multiplier: float = 1.0,
) -> float:
    """
    Return dollar bet size, optionally scaled by a UCB city multiplier.

    city_multiplier > 1 → allocate more to high-performing cities.
    city_multiplier < 1 → allocate less to cities in a drawdown.
    The hard cap scales with the multiplier but is capped at 2× max_fraction.
    """
    if edge <= 0:
        return 0.0
    kelly_size = bankroll * edge * kelly_fraction * city_multiplier
    hard_cap = bankroll * max_fraction * min(city_multiplier, 2.0)
    return min(hard_cap, kelly_size)


def dollars_to_contracts(dollar_size: float, price_dollars: float) -> float:
    """
    Convert a dollar size to a number of Kalshi contracts.

    Kalshi supports fractional contracts when fractional_trading_enabled=True.
    Each contract pays $1 if it wins. Cost = price_dollars per contract.
    Returns a float; the API accepts fractional sizes.
    """
    if price_dollars <= 0:
        return 0.0
    return dollar_size / price_dollars


# ------------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------------

def evaluate_market(
    market: dict,
    prob_row: pd.Series,
    bankroll: float,
    city_map: dict,
    city_multiplier: float = 1.0,
) -> list[dict]:
    """
    Given a Kalshi weather market dict and a model probability row,
    return a list of trade recommendations (may be empty if no edge).

    market:    dict from the Kalshi REST API (has yes_ask_dollars, no_ask_dollars)
    prob_row:  one row from TempDistributionModel.predict_integer_probs()
    bankroll:  available bankroll in dollars
    city_map:  dict mapping series_ticker prefix → city name (unused here, for logging)
    """
    title = market.get("title", "")
    subtitle = market.get("yes_sub_title") or market.get("subtitle", "")
    ticker = market.get("ticker", "")

    try:
        contract = parse_contract(title, subtitle)
    except ValueError as e:
        print(f"  Skipping {ticker}: {e}")
        return []

    fair_p = compute_fair_prob(prob_row, contract)

    yes_ask = market.get("yes_ask_dollars")
    no_ask = market.get("no_ask_dollars")

    if yes_ask is None or no_ask is None:
        print(f"  Skipping {ticker}: missing price fields")
        return []

    yes_ask = float(yes_ask)
    no_ask = float(no_ask)

    trades = []

    if should_buy_yes(fair_p, yes_ask):
        edge = fair_p - yes_ask
        dollar_size = simple_bet_size(edge, bankroll, city_multiplier=city_multiplier)
        n_contracts = dollars_to_contracts(dollar_size, yes_ask)
        if n_contracts > 0:
            trades.append({
                "ticker": ticker,
                "side": "yes",
                "fair_p": round(fair_p, 4),
                "price_dollars": yes_ask,
                "edge": round(edge, 4),
                "dollar_size": round(dollar_size, 2),
                "contract_count": round(n_contracts, 4),
            })

    if should_buy_no(fair_p, no_ask):
        edge = (1.0 - fair_p) - no_ask
        dollar_size = simple_bet_size(edge, bankroll, city_multiplier=city_multiplier)
        n_contracts = dollars_to_contracts(dollar_size, no_ask)
        if n_contracts > 0:
            trades.append({
                "ticker": ticker,
                "side": "no",
                "fair_p": round(fair_p, 4),
                "price_dollars": no_ask,
                "edge": round(edge, 4),
                "dollar_size": round(dollar_size, 2),
                "contract_count": round(n_contracts, 4),
            })

    return trades
