"""
Contract parsing and settlement logic for Kalshi weather markets.

Observed Kalshi weather title formats (as of March 2026):
  ">30°"    → gt, threshold=30  (strictly greater than 30)
  "<23°"    → lt, threshold=23  (strictly less than 23)
  "29-30°"  → range, low=29, high=30 (inclusive between)
"""

import re


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

    For > and < contracts, the threshold is the integer after the symbol and
    the contract settles on strict inequality against the raw expiration value:
      >30 → tmax > 30   → gt(30)
      <23 → tmax < 23   → lt(23)

    Returns dict with keys: market_type, and type-specific fields.
    Raises ValueError if no pattern matches.
    """
    text = f"{title} {subtitle}".lower()

    # ">30°" or "> 30°" — strictly greater than N
    m = re.search(r"be\s*>\s*(\d+)\s*°", text)
    if m:
        return {"market_type": "gt", "threshold": int(m.group(1))}

    # "<23°" — strictly less than N
    m = re.search(r"be\s*<\s*(\d+)\s*°", text)
    if m:
        return {"market_type": "lt", "threshold": int(m.group(1))}

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


def contract_yes_outcome(contract: dict, observed_tmax: float) -> int:
    """
    Resolve YES(=1)/NO(=0) for a parsed contract definition.

    Payout criterion semantics:
      - range/between: lower <= expiration_value <= upper
      - gt:            expiration_value > threshold
      - lt:            expiration_value < threshold
      - geq/leq:       inclusive comparisons used by "or above"/"or below"
                       style markets
    """
    tmax = float(observed_tmax)
    mtype = contract["market_type"]

    if mtype == "gt":
        return int(tmax > float(contract["threshold"]))
    if mtype == "lt":
        return int(tmax < float(contract["threshold"]))
    if mtype == "geq":
        return int(tmax >= float(contract["threshold"]))
    if mtype == "leq":
        return int(tmax <= float(contract["threshold"]))
    if mtype == "range":
        return int(float(contract["low"]) <= tmax <= float(contract["high"]))
    raise ValueError(f"Unsupported market_type: {mtype}")
