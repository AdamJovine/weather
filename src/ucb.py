"""
UCB-Tuned city allocation for the Kalshi weather trading bot.

Each city is treated as a bandit arm.  After each trading session we update
running statistics (n, sum_reward, sum_sq_reward) for that city and persist
them to data/ucb_state.json.

The UCB multiplier scales the Kelly bet size UP for cities performing well
and DOWN for cities in a drawdown, within a [MIN_MULT, MAX_MULT] band.

Formula (Auer et al. 2002 UCB-Tuned):
    x_bar = sum_reward / n
    s2    = sum_sq_reward / n - x_bar**2          # sample variance
    V     = s2 + sqrt(2 * ln(t) / n)              # variance bound
    ucb   = x_bar + sqrt(ln(t) / n * min(V, CAP))

    multiplier = clip(ucb / mean_ucb, MIN_MULT, MAX_MULT)

Reward signal:  pnl / size  (ROI per trade, ≈ [-1, +1])

Usage:
    from src.ucb import CityUCB
    ucb = CityUCB(cities=stations["city"].tolist())
    ucb.load_state()                            # load persisted history
    ucb.initialize_from_backtest(trades_df)     # warm-start from backtest log

    mult = ucb.kelly_multiplier("Chicago")      # call before sizing each trade
    ucb.update("Chicago", reward=pnl/size)      # call after outcome is known
    ucb.save_state()
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_STATE_PATH = "data/ucb_state.json"

# Multiplier bounds — UCB can raise or lower Kelly by at most this factor
MIN_MULT = 0.5
MAX_MULT = 2.0

# Cap on estimated reward variance (reward normalised to approx [-1, +1])
VARIANCE_CAP = 0.5

# Optimistic initialisation value for unseen cities (encourages trying them)
OPTIMISTIC_UCB = 1.0


class CityUCB:
    """
    UCB-Tuned multi-armed bandit over cities.

    State per city:
        n              – total trades observed
        sum_reward     – running sum of normalised ROI per trade
        sum_sq_reward  – running sum of squares (for variance estimation)
    """

    def __init__(
        self,
        cities: list[str],
        state_path: str = DEFAULT_STATE_PATH,
        min_mult: float = MIN_MULT,
        max_mult: float = MAX_MULT,
        variance_cap: float = VARIANCE_CAP,
    ):
        self.cities = cities
        self.state_path = Path(state_path)
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.variance_cap = variance_cap

        # Initialise all cities with zero observations
        self._states: dict[str, dict] = {
            c: {"n": 0, "sum_reward": 0.0, "sum_sq_reward": 0.0}
            for c in cities
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_state(self) -> "CityUCB":
        """Load persisted UCB state from JSON (no-op if file does not exist)."""
        if not self.state_path.exists():
            return self
        with open(self.state_path) as f:
            saved = json.load(f)
        for city, state in saved.items():
            if city in self._states:
                self._states[city].update(state)
        return self

    def save_state(self) -> "CityUCB":
        """Persist UCB state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._states, f, indent=2)
        return self

    # ------------------------------------------------------------------
    # Warm-start from backtest trades
    # ------------------------------------------------------------------

    def initialize_from_backtest(self, trades_df) -> "CityUCB":
        """
        Pre-load per-city statistics from a backtest trade log.

        trades_df must have columns: city, pnl, size
        Reward = pnl / size (normalised ROI per trade).
        Existing persisted state is merged (backtest fills gaps only).
        """
        import pandas as pd  # local import to keep module lightweight

        if trades_df is None or trades_df.empty:
            return self

        for city, group in trades_df.groupby("city"):
            if city not in self._states:
                continue
            rewards = (group["pnl"] / group["size"].replace(0, float("nan"))).dropna()
            if rewards.empty:
                continue
            existing = self._states[city]
            # Only overwrite if backtest has more data than persisted state
            if len(rewards) > existing["n"]:
                self._states[city] = {
                    "n": len(rewards),
                    "sum_reward": float(rewards.sum()),
                    "sum_sq_reward": float((rewards**2).sum()),
                }
        return self

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, city: str, reward: float) -> "CityUCB":
        """
        Record one trade outcome for a city.

        reward = pnl / size  (positive = profitable, negative = loss)
        Call this after each trade result is confirmed.
        """
        if city not in self._states:
            self._states[city] = {"n": 0, "sum_reward": 0.0, "sum_sq_reward": 0.0}
        s = self._states[city]
        s["n"] += 1
        s["sum_reward"] += reward
        s["sum_sq_reward"] += reward**2
        return self

    # ------------------------------------------------------------------
    # UCB scoring
    # ------------------------------------------------------------------

    def _total_n(self) -> int:
        return max(sum(s["n"] for s in self._states.values()), 1)

    def _ucb_score(self, city: str, total_n: int) -> float:
        s = self._states.get(city, {"n": 0, "sum_reward": 0.0, "sum_sq_reward": 0.0})
        n = s["n"]
        if n == 0:
            return OPTIMISTIC_UCB
        x_bar = s["sum_reward"] / n
        s2 = max(s["sum_sq_reward"] / n - x_bar**2, 0.0)
        ln_t = math.log(max(total_n, 2))
        V = s2 + math.sqrt(2 * ln_t / n)
        bonus = math.sqrt(ln_t / n * min(V, self.variance_cap))
        return x_bar + bonus

    def kelly_multiplier(self, city: str) -> float:
        """
        Return a Kelly scaling multiplier for this city in [MIN_MULT, MAX_MULT].

        A city with above-average UCB score gets multiplier > 1 (bet more).
        A city in a drawdown gets multiplier < 1 (bet less).
        """
        total_n = self._total_n()
        scores = {c: self._ucb_score(c, total_n) for c in self._states}

        # Avoid division by zero if all scores are zero
        mean_score = sum(scores.values()) / max(len(scores), 1)
        if mean_score == 0:
            return 1.0

        raw_mult = scores.get(city, OPTIMISTIC_UCB) / mean_score
        return max(self.min_mult, min(self.max_mult, raw_mult))

    def summary(self) -> dict:
        """Return a dict of {city: {n, mean_reward, ucb_score, multiplier}} for logging."""
        total_n = self._total_n()
        out = {}
        for city in self._states:
            s = self._states[city]
            n = s["n"]
            mean_r = s["sum_reward"] / n if n > 0 else 0.0
            out[city] = {
                "n": n,
                "mean_reward": round(mean_r, 4),
                "ucb_score": round(self._ucb_score(city, total_n), 4),
                "multiplier": round(self.kelly_multiplier(city), 3),
            }
        return out
