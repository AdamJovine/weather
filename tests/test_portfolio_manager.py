"""
Tests for PortfolioManager: bet sizing and money accounting.

Focuses on verifying that dollars are tracked correctly at every step:
  - Kelly formula produces the right dollar amount
  - Hard caps (fraction and dollar) bind at the right time
  - contracts × price == size (dollar consistency for every buy intent)
  - Exit intents carry the right contract count and dollar value
  - No doubling-up on existing positions
  - City multipliers scale money correctly without exceeding the cap
  - allow_exits=False prevents any sell intents regardless of price movement

Run from project root:
    pytest tests/test_portfolio_manager.py -v
"""
import pytest
import pandas as pd

from src.portfolio_manager import PortfolioManager, OrderIntent
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX


# ── Test helpers ──────────────────────────────────────────────────────────────

def spike_row(temp: int) -> pd.Series:
    """Probability distribution with 100% mass at one temperature."""
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    data[f"temp_{temp}"] = 1.0
    return pd.Series(data)


def uniform_row(low: int, high: int) -> pd.Series:
    """Uniform probability over [low, high]; each bin = 1/(high-low+1)."""
    n = high - low + 1
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    for k in range(low, high + 1):
        data[f"temp_{k}"] = 1.0 / n
    return pd.Series(data)


def mkt(ticker, title, yes_ask, no_ask, city="Chicago"):
    """Minimal market dict."""
    return {
        "ticker": ticker,
        "title": title,
        "_city": city,
        "yes_ask_dollars": yes_ask,
        "no_ask_dollars": no_ask,
    }


def pm_simple(**kwargs) -> PortfolioManager:
    """
    PortfolioManager with clean defaults for arithmetic clarity:
      kelly_fraction = 0.5
      max_bet_fraction = 0.10
      min_edge = 0.05
    Override any field via kwargs.
    """
    defaults = dict(
        kelly_fraction=0.5,
        max_bet_fraction=0.10,
        min_edge=0.05,
        max_bet_dollars=None,
        allow_exits=True,
        exit_edge_threshold=0.0,
        city_multiplier_cap=2.0,
    )
    defaults.update(kwargs)
    return PortfolioManager(**defaults)


# Shared prob_row: uniform over [60, 69]
# P(tmax >= 65) = 5/10 = 0.50  (title ">64°" → geq(65))
# P(tmax >= 63) = 7/10 = 0.70  (title ">62°" → geq(63))
# P(tmax <= 65) = 6/10 = 0.60  (title "<66°" → leq(65))
PROB_ROW = uniform_row(60, 69)
PRED = {"Chicago": PROB_ROW}

# Titles that give known fair_p with PROB_ROW
T_GEQ65 = "Will the high temp in Chicago be >64° on Mar 17, 2026?"  # fair_p=0.50
T_GEQ63 = "Will the high temp in Chicago be >62° on Mar 17, 2026?"  # fair_p=0.70
T_LEQ65 = "Will the high temp in Chicago be <66° on Mar 17, 2026?"  # fair_p=0.60


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  kelly_size — pure math
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellySize:
    """kelly_size() should produce the correct dollar amount for every scenario."""

    def test_basic_formula_below_cap(self):
        # bankroll=1000, edge=0.10, kelly_frac=0.5
        # kelly raw = 1000 × 0.10 × 0.5 = 50
        # cap      = 1000 × 0.10 × 1.0 = 100
        # size = min(50, 100) = 50
        assert pm_simple().kelly_size(1000, 0.10) == pytest.approx(50.0)

    def test_fraction_cap_binds(self):
        # edge=0.30 → raw = 1000 × 0.30 × 0.5 = 150  > cap = 100
        assert pm_simple().kelly_size(1000, 0.30) == pytest.approx(100.0)

    def test_exact_boundary(self):
        # edge = 0.20 → raw = 1000 × 0.20 × 0.5 = 100 = cap exactly
        assert pm_simple().kelly_size(1000, 0.20) == pytest.approx(100.0)

    def test_dollar_cap_tighter_than_fraction_cap(self):
        # raw=50, fraction_cap=100, dollar_cap=20 → 20 wins
        assert pm_simple(max_bet_dollars=20).kelly_size(1000, 0.10) == pytest.approx(20.0)

    def test_dollar_cap_looser_than_kelly(self):
        # raw=50, fraction_cap=100, dollar_cap=200 → kelly wins = 50
        assert pm_simple(max_bet_dollars=200).kelly_size(1000, 0.10) == pytest.approx(50.0)

    def test_zero_edge_returns_zero(self):
        assert pm_simple().kelly_size(1000, 0.0) == 0.0

    def test_negative_edge_returns_zero(self):
        assert pm_simple().kelly_size(1000, -0.10) == 0.0

    def test_zero_bankroll_returns_zero(self):
        assert pm_simple().kelly_size(0, 0.20) == 0.0

    def test_city_multiplier_scales_both_kelly_and_cap(self):
        # mult=2: raw = 1000×0.10×0.5×2 = 100, cap = 1000×0.10×2 = 200
        # min(100, 200) = 100
        assert pm_simple().kelly_size(1000, 0.10, city_multiplier=2.0) == pytest.approx(100.0)

    def test_city_multiplier_cap_limits_fraction_cap(self):
        # mult=3 but city_multiplier_cap=2 → cap uses min(3, 2)=2
        # raw = 1000×0.10×0.5×3 = 150, cap = 1000×0.10×2 = 200
        # min(150, 200) = 150
        assert pm_simple().kelly_size(1000, 0.10, city_multiplier=3.0) == pytest.approx(150.0)

    def test_city_multiplier_cap_fraction_binds(self):
        # mult=3, edge=0.50 → raw = 1000×0.50×0.5×3 = 750, cap = 200 → 200
        assert pm_simple().kelly_size(1000, 0.50, city_multiplier=3.0) == pytest.approx(200.0)

    def test_bankroll_proportionality(self):
        # Doubling bankroll should double the size
        s1 = pm_simple().kelly_size(1000, 0.10)
        s2 = pm_simple().kelly_size(2000, 0.10)
        assert s2 == pytest.approx(s1 * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Entry sizing — dollars in, contracts out
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntryMoney:
    """Buy intents must satisfy: contracts × price == size (dollar consistency)."""

    def test_contracts_times_price_equals_size_yes(self):
        # fair_p=0.50, yes_ask=0.40 → yes_edge=0.10 ≥ 0.05 → BUY YES
        # size = kelly_size(1000, 0.10) = 50
        # contracts = 50 / 0.40 = 125
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy"]
        assert len(buys) == 1
        i = buys[0]
        assert i.side == "yes"
        assert i.size == pytest.approx(50.0)
        assert i.contracts == pytest.approx(125.0)
        assert i.contracts * i.price == pytest.approx(i.size, rel=1e-9)

    def test_contracts_times_price_equals_size_no(self):
        # fair_p=0.70, no_ask=0.20 → no_edge=(1-0.70)-0.20=0.10 ≥ 0.05 → BUY NO
        markets = [mkt("T1", T_GEQ63, yes_ask=0.80, no_ask=0.20)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy" and i.side == "no"]
        assert len(buys) == 1
        i = buys[0]
        assert i.contracts * i.price == pytest.approx(i.size, rel=1e-9)

    def test_size_equals_kelly_formula(self):
        # fair_p=0.50, yes_ask=0.40, edge=0.10, bankroll=2000
        # kelly_size: raw=2000×0.10×0.5=100, cap=2000×0.10=200 → size=100
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=2000)
        buys = [i for i in intents if i.action == "buy"]
        assert buys[0].size == pytest.approx(100.0)

    def test_fraction_cap_applied_to_entry(self):
        # edge=0.40 → raw=1000×0.40×0.5=200 > cap=100 → size=100
        # fair_p=0.70, yes_ask=0.30, edge=0.40
        markets = [mkt("T1", T_GEQ63, yes_ask=0.30, no_ask=0.35)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert buys[0].size == pytest.approx(100.0)  # capped at 10% of 1000

    def test_dollar_cap_applied_to_entry(self):
        # max_bet_dollars=30, edge=0.10 → raw=50, fraction_cap=100 → 30
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_simple(max_bet_dollars=30).evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy"]
        assert buys[0].size == pytest.approx(30.0)
        assert buys[0].contracts * buys[0].price == pytest.approx(30.0, rel=1e-9)

    def test_edge_below_min_no_entry(self):
        # edge=0.04 < min_edge=0.05 → no entry
        # fair_p=0.50, yes_ask=0.46 → edge=0.04
        markets = [mkt("T1", T_GEQ65, yes_ask=0.46, no_ask=0.56)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        assert all(i.action != "buy" for i in intents)

    def test_edge_at_min_triggers_entry(self):
        # fair_p=0.50, yes_ask=0.44 → edge≈0.06 > min_edge=0.05 → should enter
        # (0.50 - 0.45 = 0.04999... in float, so use 0.44 to avoid boundary FP issue)
        markets = [mkt("T1", T_GEQ65, yes_ask=0.44, no_ask=0.57)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy"]
        assert len(buys) >= 1

    def test_entries_sorted_by_edge_descending(self):
        # Two markets, higher-edge one should come first in buys
        # T_GEQ63: fair_p=0.70, yes_ask=0.40 → edge=0.30 (higher)
        # T_GEQ65: fair_p=0.50, yes_ask=0.40 → edge=0.10 (lower)
        markets = [
            mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.65),
            mkt("T2", T_GEQ63, yes_ask=0.40, no_ask=0.65),
        ]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy"]
        assert len(buys) == 2
        assert buys[0].edge >= buys[1].edge

    def test_bankroll_change_proportionally_scales_size(self):
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        s1 = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        s2 = pm_simple().evaluate(markets, PRED, {}, bankroll=500)
        b1 = next(i for i in s1 if i.action == "buy")
        b2 = next(i for i in s2 if i.action == "buy")
        assert b1.size == pytest.approx(b2.size * 2)

    def test_city_multiplier_scales_entry_size(self):
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        # mult=2 → raw=1000×0.10×0.5×2=100, cap=1000×0.10×2=200 → 100
        intents = pm_simple().evaluate(
            markets, PRED, {}, bankroll=1000,
            city_multipliers={"Chicago": 2.0}
        )
        buys = [i for i in intents if i.action == "buy"]
        assert buys[0].size == pytest.approx(100.0)
        assert buys[0].contracts * buys[0].price == pytest.approx(100.0, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  No doubling-up on existing positions
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoDoubleUp:
    """Already-held positions must not generate a duplicate buy intent."""

    def test_long_yes_suppresses_yes_entry(self):
        # Holding +10 YES → no additional YES buy
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        yes_buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert len(yes_buys) == 0

    def test_long_no_suppresses_no_entry(self):
        # Holding -10 (long NO) → no additional NO buy
        # Use T_GEQ63: fair_p=0.70, no_ask=0.20 → no_edge=0.10 → would normally enter NO
        markets = [mkt("T1", T_GEQ63, yes_ask=0.85, no_ask=0.20)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": -10.0}, bankroll=1000
        )
        no_buys = [i for i in intents if i.action == "buy" and i.side == "no"]
        assert len(no_buys) == 0

    def test_flat_position_allows_both_sides(self):
        # no position → both sides with edge should generate intents
        # fair_p=0.70, yes_ask=0.50 → yes_edge=0.20, no_ask=0.20 → no_edge=0.10
        markets = [mkt("T1", T_GEQ63, yes_ask=0.50, no_ask=0.20)]
        intents = pm_simple().evaluate(markets, PRED, {}, bankroll=1000)
        buys = [i for i in intents if i.action == "buy"]
        sides = {i.side for i in buys}
        assert "yes" in sides
        assert "no" in sides

    def test_long_yes_allows_no_entry_for_same_ticker(self):
        # Holding YES doesn't block NO entry on the same ticker
        # (market has moved: now NO also has edge)
        # fair_p=0.70, yes_ask=0.50 → yes_edge=0.20, no_ask=0.20 → no_edge=0.10
        markets = [mkt("T1", T_GEQ63, yes_ask=0.50, no_ask=0.20)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        no_buys = [i for i in intents if i.action == "buy" and i.side == "no"]
        assert len(no_buys) == 1

    def test_unrelated_tickers_dont_interfere(self):
        # Position on T2 should not suppress entry on T1
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T2": 10.0}, bankroll=1000
        )
        yes_buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert len(yes_buys) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Exit (sell) intents — dollar and contract accounting
# ═══════════════════════════════════════════════════════════════════════════════

class TestExitMoney:
    """Sell intents must carry the correct contract count and dollar value."""

    def test_yes_exit_triggered_when_bid_at_fair(self):
        # fair_p=0.50, yes_bid=1-no_ask=1-0.50=0.50 ≥ 0.50 → EXIT
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 20.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 1
        assert sells[0].side == "yes"

    def test_yes_exit_carries_full_held_contracts(self):
        # We hold 25 contracts; exit should sell all 25
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 25.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert sells[0].contracts == pytest.approx(25.0)

    def test_yes_exit_size_equals_contracts_times_exit_price(self):
        # exit_price = yes_bid = 1 - no_ask
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.44)]
        held = 15.0
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": held}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert sells[0].size == pytest.approx(held * sells[0].price, rel=1e-9)

    def test_no_exit_triggered_when_no_bid_at_fair(self):
        # fair_p=0.50, no_bid = 1 - yes_ask
        # Long NO, so no_bid needs to >= (1 - fair_p) = 0.50
        # yes_ask=0.47 → no_bid=0.53 ≥ 0.50 → EXIT
        markets = [mkt("T1", T_GEQ65, yes_ask=0.47, no_ask=0.55)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": -18.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 1
        assert sells[0].side == "no"
        assert sells[0].contracts == pytest.approx(18.0)

    def test_no_exit_when_yes_still_has_edge(self):
        # fair_p=0.50, yes_bid = 1 - no_ask = 1 - 0.60 = 0.40 < 0.50 → HOLD
        markets = [mkt("T1", T_GEQ65, yes_ask=0.42, no_ask=0.60)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 0

    def test_exit_edge_threshold_exits_earlier(self):
        # exit_edge_threshold=0.05: exit when yes_bid >= fair_p - 0.05
        # fair_p=0.50, yes_bid=0.46 → 0.46 >= 0.45 → EXIT
        markets = [mkt("T1", T_GEQ65, yes_ask=0.56, no_ask=0.54)]
        # yes_bid = 1 - no_ask = 1 - 0.54 = 0.46
        intents = pm_simple(exit_edge_threshold=0.05).evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 1

    def test_exit_edge_threshold_holds_when_bid_just_below(self):
        # exit_edge_threshold=0.05: exit when yes_bid >= 0.45
        # yes_ask=0.58, no_ask=0.56 → yes_bid = 1 - 0.56 = 0.44 < 0.45 → HOLD
        markets = [mkt("T1", T_GEQ65, yes_ask=0.58, no_ask=0.56)]
        intents = pm_simple(exit_edge_threshold=0.05).evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 0

    def test_allow_exits_false_suppresses_all_sells(self):
        # Even with clear exit condition, allow_exits=False → no sell intents
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50)]
        intents = pm_simple(allow_exits=False).evaluate(
            markets, PRED, positions={"T1": 20.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 0

    def test_zero_position_never_exits(self):
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50)]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 0.0}, bankroll=1000
        )
        sells = [i for i in intents if i.action == "sell"]
        assert len(sells) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Ordering and statelesness
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrdering:
    """Sells must precede buys; evaluate() must be stateless across calls."""

    def test_sells_precede_buys(self):
        # Position T1 needs exit; T2 has entry edge
        markets = [
            mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50),  # exit: yes_bid=0.50 >= fair_p=0.50
            mkt("T2", T_GEQ63, yes_ask=0.40, no_ask=0.65),  # entry: yes_edge=0.30
        ]
        intents = pm_simple().evaluate(
            markets, PRED, positions={"T1": 5.0}, bankroll=1000
        )
        actions = [i.action for i in intents]
        # All sells must appear before any buy
        last_sell = max((j for j, a in enumerate(actions) if a == "sell"), default=-1)
        first_buy = min((j for j, a in enumerate(actions) if a == "buy"), default=len(actions))
        assert last_sell < first_buy, f"sells={last_sell}, first_buy={first_buy}, actions={actions}"

    def test_stateless_same_result_on_repeat_call(self):
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        p = pm_simple()
        r1 = p.evaluate(markets, PRED, {}, bankroll=1000)
        r2 = p.evaluate(markets, PRED, {}, bankroll=1000)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.size == pytest.approx(b.size)
            assert a.contracts == pytest.approx(b.contracts)

    def test_stateless_different_bankrolls_give_different_sizes(self):
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        p = pm_simple()
        r1 = p.evaluate(markets, PRED, {}, bankroll=1000)
        r2 = p.evaluate(markets, PRED, {}, bankroll=500)
        b1 = next(i for i in r1 if i.action == "buy")
        b2 = next(i for i in r2 if i.action == "buy")
        assert b1.size != pytest.approx(b2.size)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  from_config wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromConfig:
    """from_config() must wire all relevant fields correctly."""

    def test_from_config_uses_config_kelly(self):
        class Cfg:
            kelly_fraction = 0.25
            max_bet_fraction = 0.08
            max_bet_dollars = None
            min_edge = 0.06
            min_confidence = 0.0
            allow_exits = True
            exit_edge_threshold = 0.01

        pm_cfg = PortfolioManager.from_config(Cfg())
        assert pm_cfg.kelly_fraction == 0.25
        assert pm_cfg.max_bet_fraction == 0.08
        assert pm_cfg.min_edge == 0.06
        assert pm_cfg.allow_exits is True
        assert pm_cfg.exit_edge_threshold == 0.01

    def test_from_config_allow_exits_false(self):
        class Cfg:
            kelly_fraction = 0.33
            max_bet_fraction = 0.05
            max_bet_dollars = None
            min_edge = 0.05
            min_confidence = 0.0
            allow_exits = False
            exit_edge_threshold = 0.0

        pm_cfg = PortfolioManager.from_config(Cfg())
        markets = [mkt("T1", T_GEQ65, yes_ask=0.52, no_ask=0.50)]
        intents = pm_cfg.evaluate(
            markets, PRED, positions={"T1": 10.0}, bankroll=1000
        )
        assert all(i.action != "sell" for i in intents)

    def test_from_config_min_edge_respected(self):
        # min_edge=0.12: edge=0.10 should produce no entry
        class Cfg:
            kelly_fraction = 0.5
            max_bet_fraction = 0.10
            max_bet_dollars = None
            min_edge = 0.12
            min_confidence = 0.0
            allow_exits = True
            exit_edge_threshold = 0.0

        pm_cfg = PortfolioManager.from_config(Cfg())
        # fair_p=0.50, yes_ask=0.40 → edge=0.10 < 0.12
        markets = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm_cfg.evaluate(markets, PRED, {}, bankroll=1000)
        assert all(i.action != "buy" for i in intents)
