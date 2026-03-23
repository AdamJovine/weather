"""
Convert model probability distributions to fair values for Kalshi contract types.

Contract types:
  exact:  P(tmax == k)
  range:  P(low <= tmax <= high)
  gt:     P(tmax > threshold)
  lt:     P(tmax < threshold)
  geq:    P(tmax >= threshold)
  leq:    P(tmax <= threshold)
"""

import pandas as pd

from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX


def fair_prob_exact(prob_row: pd.Series, temp_value: int) -> float:
    key = f"temp_{temp_value}"
    if key not in prob_row:
        raise KeyError(f"temp_{temp_value} not in prob_row. Check TEMP_GRID range.")
    return float(prob_row[key])


def fair_prob_range(prob_row: pd.Series, low: int, high: int) -> float:
    return sum(
        float(prob_row.get(f"temp_{k}", 0.0))
        for k in range(low, high + 1)
    )


def fair_prob_geq(
    prob_row: pd.Series,
    threshold: int,
    max_temp: int = TEMP_GRID_MAX,
) -> float:
    return sum(
        float(prob_row.get(f"temp_{k}", 0.0))
        for k in range(threshold, max_temp + 1)
    )


def fair_prob_gt(
    prob_row: pd.Series,
    threshold: int,
    max_temp: int = TEMP_GRID_MAX,
) -> float:
    return sum(
        float(prob_row.get(f"temp_{k}", 0.0))
        for k in range(threshold + 1, max_temp + 1)
    )


def fair_prob_leq(
    prob_row: pd.Series,
    threshold: int,
    min_temp: int = TEMP_GRID_MIN,
) -> float:
    return sum(
        float(prob_row.get(f"temp_{k}", 0.0))
        for k in range(min_temp, threshold + 1)
    )


def fair_prob_lt(
    prob_row: pd.Series,
    threshold: int,
    min_temp: int = TEMP_GRID_MIN,
) -> float:
    return sum(
        float(prob_row.get(f"temp_{k}", 0.0))
        for k in range(min_temp, threshold)
    )


def compute_fair_prob(prob_row: pd.Series, contract: dict) -> float:
    """
    Dispatch to the correct pricing function based on contract type.

    contract dict keys:
      market_type: "exact" | "range" | "gt" | "lt" | "geq" | "leq"
      temp_value:  int           (exact)
      low, high:   int, int      (range)
      threshold:   int           (gt / lt / geq / leq)
    """
    mtype = contract["market_type"]
    if mtype == "exact":
        return fair_prob_exact(prob_row, contract["temp_value"])
    elif mtype == "range":
        return fair_prob_range(prob_row, contract["low"], contract["high"])
    elif mtype == "gt":
        return fair_prob_gt(prob_row, contract["threshold"])
    elif mtype == "lt":
        return fair_prob_lt(prob_row, contract["threshold"])
    elif mtype == "geq":
        return fair_prob_geq(prob_row, contract["threshold"])
    elif mtype == "leq":
        return fair_prob_leq(prob_row, contract["threshold"])
    else:
        raise ValueError(f"Unknown market_type: {mtype}")


def dollars_to_prob(dollars) -> float:
    """
    Kalshi weather market prices are quoted as dollars (0.0–1.0).
    This is a passthrough for clarity and to keep call sites explicit.
    """
    return float(dollars)
