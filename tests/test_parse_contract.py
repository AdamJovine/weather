"""
Unit tests for parse_contract and compute_fair_prob.

No API credentials required — runs entirely offline.

Run from project root:
  python scripts/test_parse_contract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy import parse_contract
from src.pricing import compute_fair_prob, fair_prob_geq, fair_prob_leq, fair_prob_range
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX

import pandas as pd

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures = []


def check(label: str, got, expected):
    ok = got == expected
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         got:      {got!r}")
        print(f"         expected: {expected!r}")
        _failures.append(label)


def check_raises(label: str, fn, exc_type=ValueError):
    try:
        fn()
        print(f"  [{FAIL}] {label}  (expected {exc_type.__name__}, got no exception)")
        _failures.append(label)
    except exc_type:
        print(f"  [{PASS}] {label}")
    except Exception as e:
        print(f"  [{FAIL}] {label}  (unexpected exception: {e})")
        _failures.append(label)


# ---------------------------------------------------------------------------
# Helper: build a flat prob_row with uniform probability across a range
# ---------------------------------------------------------------------------

def uniform_prob_row(low: int, high: int) -> pd.Series:
    """
    Returns a prob_row where P(tmax=k) = 1/(high-low+1) for k in [low, high], else 0.
    """
    n = high - low + 1
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    for k in range(low, high + 1):
        data[f"temp_{k}"] = 1.0 / n
    return pd.Series(data)


def spike_prob_row(temp: int) -> pd.Series:
    """Returns a prob_row with 100% probability at one temperature."""
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    data[f"temp_{temp}"] = 1.0
    return pd.Series(data)


# ===========================================================================
# SECTION 1: parse_contract — title formats
# ===========================================================================

print("\n=== parse_contract: Kalshi '> N°' format ===")
check(">30° (be >30°)",
      parse_contract("Will the high temp be >30° on Mar 17?"),
      {"market_type": "geq", "threshold": 31})

check("> 72° with space",
      parse_contract("high temp be > 72°"),
      {"market_type": "geq", "threshold": 73})


print("\n=== parse_contract: Kalshi '< N°' format ===")
check("<23° (be <23°)",
      parse_contract("Will the high temp be <23° on Mar 17?"),
      {"market_type": "leq", "threshold": 22})

check("< 50° with space",
      parse_contract("high temp be < 50°"),
      {"market_type": "leq", "threshold": 49})


print("\n=== parse_contract: range with hyphen '29-30°' ===")
check("29-30° range",
      parse_contract("Will the high temp be 29-30° on Mar 17?"),
      {"market_type": "range", "low": 29, "high": 30})

check("61-62° range",
      parse_contract("high temp be 61-62°"),
      {"market_type": "range", "low": 61, "high": 62})


print("\n=== parse_contract: range with 'to' — actual Kalshi format ===")
check("59° to 60°",
      parse_contract("59° to 60°"),
      {"market_type": "range", "low": 59, "high": 60})

check("63° to 64° (full sentence)",
      parse_contract("Will the high temp be 63° to 64° on Mar 17?"),
      {"market_type": "range", "low": 63, "high": 64})

check("65 to 66° (no leading °)",
      parse_contract("65 to 66°"),
      {"market_type": "range", "low": 65, "high": 66})


print("\n=== parse_contract: 'or above' — actual Kalshi format ===")
check("67° or above",
      parse_contract("67° or above"),
      {"market_type": "geq", "threshold": 67})

check("70° or above (full sentence)",
      parse_contract("Will the high be 70° or above on Mar 17?"),
      {"market_type": "geq", "threshold": 70})


print("\n=== parse_contract: 'or below' / 'or lower' ===")
check("58° or below",
      parse_contract("58° or below"),
      {"market_type": "leq", "threshold": 58})

check("55° or lower",
      parse_contract("55° or lower"),
      {"market_type": "leq", "threshold": 55})


print("\n=== parse_contract: 'or higher' ===")
check("80° or higher",
      parse_contract("80° or higher"),
      {"market_type": "geq", "threshold": 80})


print("\n=== parse_contract: 'at least' / 'at most' ===")
check("at least 75",
      parse_contract("will the high be at least 75"),
      {"market_type": "geq", "threshold": 75})

check("at most 40",
      parse_contract("will the high be at most 40"),
      {"market_type": "leq", "threshold": 40})


print("\n=== parse_contract: unrecognized → ValueError ===")
check_raises("gibberish title",
             lambda: parse_contract("something completely unrelated"))


# ===========================================================================
# SECTION 2: compute_fair_prob — probability math
# ===========================================================================

print("\n=== compute_fair_prob: spike distribution ===")

row_65 = spike_prob_row(65)

# geq: P(tmax >= 65) with all mass at 65 → 1.0
p = compute_fair_prob(row_65, {"market_type": "geq", "threshold": 65})
check("geq(65) | spike@65 → 1.0", round(p, 6), 1.0)

# geq: P(tmax >= 66) with all mass at 65 → 0.0
p = compute_fair_prob(row_65, {"market_type": "geq", "threshold": 66})
check("geq(66) | spike@65 → 0.0", round(p, 6), 0.0)

# leq: P(tmax <= 65) with all mass at 65 → 1.0
p = compute_fair_prob(row_65, {"market_type": "leq", "threshold": 65})
check("leq(65) | spike@65 → 1.0", round(p, 6), 1.0)

# leq: P(tmax <= 64) with all mass at 65 → 0.0
p = compute_fair_prob(row_65, {"market_type": "leq", "threshold": 64})
check("leq(64) | spike@65 → 0.0", round(p, 6), 0.0)

# range: P(63 <= tmax <= 67) with all mass at 65 → 1.0
p = compute_fair_prob(row_65, {"market_type": "range", "low": 63, "high": 67})
check("range(63,67) | spike@65 → 1.0", round(p, 6), 1.0)

# range: P(66 <= tmax <= 68) with all mass at 65 → 0.0
p = compute_fair_prob(row_65, {"market_type": "range", "low": 66, "high": 68})
check("range(66,68) | spike@65 → 0.0", round(p, 6), 0.0)


print("\n=== compute_fair_prob: uniform distribution [60, 69] ===")

row_uniform = uniform_prob_row(60, 69)  # 10 bins, each 0.1

# geq(65): bins 65–69 → 5 * 0.1 = 0.5
p = compute_fair_prob(row_uniform, {"market_type": "geq", "threshold": 65})
check("geq(65) | uniform[60,69] → 0.5", round(p, 6), 0.5)

# leq(64): bins 60–64 → 5 * 0.1 = 0.5
p = compute_fair_prob(row_uniform, {"market_type": "leq", "threshold": 64})
check("leq(64) | uniform[60,69] → 0.5", round(p, 6), 0.5)

# range(63, 66): 4 bins → 0.4
p = compute_fair_prob(row_uniform, {"market_type": "range", "low": 63, "high": 66})
check("range(63,66) | uniform[60,69] → 0.4", round(p, 6), 0.4)

# geq + leq complement: should sum to 1.0 at any split
p_geq = compute_fair_prob(row_uniform, {"market_type": "geq", "threshold": 63})
p_leq = compute_fair_prob(row_uniform, {"market_type": "leq", "threshold": 62})
check("geq(63) + leq(62) = 1.0 | uniform[60,69]", round(p_geq + p_leq, 6), 1.0)


# ===========================================================================
# SECTION 3: End-to-end parse → price round-trip
# ===========================================================================

print("\n=== End-to-end: parse title → compute fair prob ===")

row = spike_prob_row(67)

title_above = "67° or above"
contract = parse_contract(title_above)
p = compute_fair_prob(row, contract)
check(f'"{title_above}" | spike@67 → 1.0', round(p, 6), 1.0)

title_below = "58° or below"
contract = parse_contract(title_below)
p = compute_fair_prob(spike_prob_row(58), contract)
check(f'"{title_below}" | spike@58 → 1.0', round(p, 6), 1.0)

title_range = "63° to 64°"
contract = parse_contract(title_range)
p = compute_fair_prob(spike_prob_row(63), contract)
check(f'"{title_range}" | spike@63 → 1.0', round(p, 6), 1.0)

p = compute_fair_prob(spike_prob_row(65), contract)
check(f'"{title_range}" | spike@65 → 0.0', round(p, 6), 0.0)


# ===========================================================================
# Summary
# ===========================================================================

print()
if _failures:
    print(f"\033[31m{len(_failures)} test(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[32mAll tests passed.\033[0m")
