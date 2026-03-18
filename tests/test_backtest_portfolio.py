"""
Thorough tests for backtesting/portfolio.py (Portfolio) and the per-ticker cap
logic in src/portfolio_manager.py (PortfolioManager).

Key invariants verified
───────────────────────
1. Total dollars committed to any ticker (portfolio._ticker_cost) never
   exceeds max_bet_dollars across multiple trades / sessions / days.
2. PortfolioManager clamps intent size to remaining room under max_bet_dollars
   when ticker_cost is passed.
3. Bankroll updates exactly as prev + pnl, floored at $1.
4. _ticker_cost accumulates correctly across record_trade / open_position.
5. Daily and session trade limits are enforced and reset at the right times.
6. count_toward_limits=False (exit/settle path) does NOT consume slots or
   add to ticker_cost.

Run from project root:
    pytest tests/test_backtest_portfolio.py -v
"""
import pytest
import pandas as pd

from backtesting.portfolio import Portfolio, Trade, OpenPosition
from src.portfolio_manager import PortfolioManager, OrderIntent
from src.config import TEMP_GRID_MIN, TEMP_GRID_MAX


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trade(
    ticker: str = "T1",
    size: float = 10.0,
    pnl: float = 2.0,
    city: str = "Chicago",
    date: str = "2026-01-01",
    session: int = 0,
) -> Trade:
    return Trade(
        run_id=1, date=date, city=city, session=session, session_ts=0,
        ticker=ticker, title="Test contract", contract_type="geq", threshold=65,
        side="yes", mkt_p=0.40, model_p=0.50, edge=0.10,
        size=size, pnl=pnl, outcome=1,
        bankroll_after=100.0 + pnl,
    )


def make_position(
    ticker: str = "T1",
    size: float = 10.0,
    city: str = "Chicago",
    entry_mkt_p: float = 0.40,
) -> OpenPosition:
    return OpenPosition(
        city=city, ticker=ticker, title="Test contract",
        contract_type="geq", threshold=65,
        contract_def={"market_type": "geq", "threshold": 65},
        side="yes",
        entry_mkt_p=entry_mkt_p, entry_model_p=0.50, entry_edge=0.10,
        size=size,
        entry_session=0, entry_session_ts=0,
        entry_date="2026-01-01",
    )


def make_portfolio(
    initial_bankroll: float = 100.0,
    max_bet_dollars: float = 10.0,
    kelly_fraction: float = 0.5,
    max_bet_fraction: float = 0.10,
    max_daily_trades: int = 20,
    max_session_trades: int = 10,
) -> Portfolio:
    return Portfolio(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        max_bet_dollars=max_bet_dollars,
        max_daily_trades=max_daily_trades,
        max_session_trades=max_session_trades,
    )


def uniform_row(low: int, high: int) -> pd.Series:
    n = high - low + 1
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    for k in range(low, high + 1):
        data[f"temp_{k}"] = 1.0 / n
    return pd.Series(data)


def spike_row(temp: int) -> pd.Series:
    data = {f"temp_{k}": 0.0 for k in range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)}
    data[f"temp_{temp}"] = 1.0
    return pd.Series(data)


def mkt(ticker, title, yes_ask, no_ask, city="Chicago"):
    return {
        "ticker": ticker, "title": title, "_city": city,
        "yes_ask_dollars": yes_ask, "no_ask_dollars": no_ask,
    }


def pm(**kwargs) -> PortfolioManager:
    defaults = dict(
        kelly_fraction=0.5,
        max_bet_fraction=0.10,
        min_edge=0.05,
        max_bet_dollars=None,
        allow_exits=False,
        exit_edge_threshold=0.0,
        city_multiplier_cap=2.0,
        min_bet_dollars=0.01,
    )
    defaults.update(kwargs)
    return PortfolioManager(**defaults)


# Shared probability rows (uniform over [60, 69]):
#   fair_p for geq(65)  = P(tmax >= 65) = 5/10 = 0.50
#   fair_p for geq(63)  = P(tmax >= 63) = 7/10 = 0.70
PROB_ROW = uniform_row(60, 69)
PRED = {"Chicago": PROB_ROW}

# Titles that parse to known contracts with PROB_ROW
T_GEQ65 = "Will the high temp in Chicago be >64° on Mar 17, 2026?"  # fair_p=0.50
T_GEQ63 = "Will the high temp in Chicago be >62° on Mar 17, 2026?"  # fair_p=0.70


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Portfolio — bankroll accounting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioBankrollAccounting:
    """Bankroll must update exactly as prev + pnl, floored at $1."""

    def test_initial_bankroll(self):
        p = make_portfolio(initial_bankroll=500.0)
        assert p.bankroll == pytest.approx(500.0)

    def test_winning_trade_increases_bankroll(self):
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=5.0))
        assert p.bankroll == pytest.approx(105.0)

    def test_losing_trade_decreases_bankroll(self):
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=-8.0))
        assert p.bankroll == pytest.approx(92.0)

    def test_bankroll_floored_at_one_on_large_loss(self):
        p = make_portfolio(initial_bankroll=10.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=-100.0))
        assert p.bankroll == pytest.approx(1.0)

    def test_bankroll_floored_at_one_on_exact_wipeout(self):
        p = make_portfolio(initial_bankroll=50.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=-50.0))
        assert p.bankroll == pytest.approx(1.0)

    def test_bankroll_floor_one_not_less_than_one(self):
        p = make_portfolio(initial_bankroll=5.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        for _ in range(10):
            p.record_trade(make_trade(pnl=-10.0))
        assert p.bankroll >= 1.0

    def test_multiple_trades_compound_correctly(self):
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=10.0))
        p.record_trade(make_trade(pnl=-3.0))
        p.record_trade(make_trade(pnl=7.0))
        assert p.bankroll == pytest.approx(114.0)

    def test_zero_pnl_trade_leaves_bankroll_unchanged(self):
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=0.0))
        assert p.bankroll == pytest.approx(100.0)

    def test_record_trade_count_false_still_updates_bankroll(self):
        """Exit/settle path: bankroll must update even with count_toward_limits=False."""
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.record_trade(make_trade(pnl=5.0), count_toward_limits=False)
        assert p.bankroll == pytest.approx(105.0)

    def test_bankroll_recovers_after_floor(self):
        p = make_portfolio(initial_bankroll=10.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(pnl=-100.0))  # floor to 1.0
        p.record_trade(make_trade(pnl=4.0))
        assert p.bankroll == pytest.approx(5.0)

    def test_bankroll_never_below_one_regardless_of_sequence(self):
        p = make_portfolio(initial_bankroll=50.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        pnls = [20.0, -80.0, 5.0, -3.0, -2.0, 10.0]
        for pnl in pnls:
            p.record_trade(make_trade(pnl=pnl))
            assert p.bankroll >= 1.0, f"bankroll dropped below $1 after pnl={pnl}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Portfolio — holdings (ticker_cost) accounting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioHoldingsAccounting:
    """_ticker_cost must accumulate exactly the entry sizes, never decremented on exit."""

    def test_no_cost_before_any_trade(self):
        p = make_portfolio()
        assert p.ticker_cost == {}

    def test_record_trade_accumulates_ticker_cost(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=8.0, pnl=2.0))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(8.0)

    def test_multiple_trades_same_ticker_accumulate(self):
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=3.0, pnl=0.0))
        p.record_trade(make_trade(ticker="T1", size=4.0, pnl=0.0))
        p.record_trade(make_trade(ticker="T1", size=2.0, pnl=0.0))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(9.0)

    def test_different_tickers_tracked_independently(self):
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=5.0, pnl=0.0))
        p.record_trade(make_trade(ticker="T2", size=7.0, pnl=0.0))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(5.0)
        assert p.ticker_cost.get("T2", 0.0) == pytest.approx(7.0)

    def test_exit_trade_does_not_add_to_ticker_cost(self):
        """count_toward_limits=False must NOT add to _ticker_cost."""
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=8.0, pnl=2.0))
        cost_before = p.ticker_cost.get("T1", 0.0)
        # Simulate exit/settle — count_toward_limits=False
        p.record_trade(make_trade(ticker="T1", size=5.0, pnl=3.0), count_toward_limits=False)
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(cost_before)

    def test_open_position_adds_to_ticker_cost(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=6.0))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(6.0)

    def test_open_position_accumulates_across_calls(self):
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=4.0))
        p.open_position(make_position(ticker="T1", size=3.0))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(7.0)

    def test_ticker_cost_persists_across_days(self):
        """Cost from day 1 must still be present on day 2 (lifetime tracker)."""
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=5.0, pnl=0.0))
        p.begin_day("2026-01-02")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=3.0, pnl=0.0, date="2026-01-02"))
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(8.0)

    def test_ticker_cost_is_a_dict(self):
        assert isinstance(make_portfolio().ticker_cost, dict)

    def test_close_position_does_not_affect_ticker_cost(self):
        """Removing an open position must not decrement ticker_cost."""
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=8.0))
        cost_after_open = p.ticker_cost.get("T1", 0.0)
        p.close_position("T1")
        assert p.ticker_cost.get("T1", 0.0) == pytest.approx(cost_after_open)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Portfolio — trade limit enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioTradeLimits:
    """Daily and session limits must be enforced and reset correctly."""

    def test_can_trade_initially_true(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        assert p.can_trade() is True

    def test_can_trade_false_after_daily_limit_hit(self):
        p = make_portfolio(max_daily_trades=2, max_session_trades=10)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.can_trade() is False

    def test_can_trade_false_after_session_limit_hit(self):
        p = make_portfolio(max_daily_trades=10, max_session_trades=2)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.can_trade() is False

    def test_begin_day_resets_daily_limit(self):
        p = make_portfolio(max_daily_trades=2, max_session_trades=10)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.can_trade() is False
        p.begin_day("2026-01-02")
        p.begin_session(0)
        assert p.can_trade() is True

    def test_begin_session_new_index_resets_session_counter(self):
        p = make_portfolio(max_daily_trades=10, max_session_trades=2)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.can_trade() is False
        p.begin_session(1)   # new session index
        assert p.can_trade() is True

    def test_begin_session_same_index_does_not_reset_counter(self):
        p = make_portfolio(max_daily_trades=10, max_session_trades=2)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        p.begin_session(0)   # same index — must NOT reset
        assert p.can_trade() is False

    def test_exit_trade_count_false_does_not_consume_slot(self):
        p = make_portfolio(max_daily_trades=2, max_session_trades=2)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.can_trade() is False
        # Exit trade — should not consume extra slot
        p.record_trade(make_trade(), count_toward_limits=False)
        assert p.can_trade() is False  # still exhausted, not changed

    def test_remaining_today_decrements_with_each_trade(self):
        p = make_portfolio(max_daily_trades=5, max_session_trades=10)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        assert p.remaining_today() == 5
        p.record_trade(make_trade())
        assert p.remaining_today() == 4
        p.record_trade(make_trade())
        assert p.remaining_today() == 3

    def test_remaining_today_never_negative(self):
        p = make_portfolio(max_daily_trades=2, max_session_trades=10)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        p.record_trade(make_trade())
        assert p.remaining_today() == 0

    def test_remaining_this_session_never_negative(self):
        p = make_portfolio(max_daily_trades=10, max_session_trades=1)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        assert p.remaining_this_session() == 0

    def test_open_position_consumes_trade_slot(self):
        p = make_portfolio(max_daily_trades=2, max_session_trades=2)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1"))
        p.open_position(make_position(ticker="T2"))
        assert p.can_trade() is False

    def test_exit_trade_does_not_consume_slot_after_position_open(self):
        """open_position uses a slot; subsequent exit must not use another."""
        p = make_portfolio(max_daily_trades=1, max_session_trades=1)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1"))
        assert p.can_trade() is False
        p.close_position("T1")
        p.record_trade(make_trade(ticker="T1"), count_toward_limits=False)
        assert p.can_trade() is False  # still at daily limit


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Portfolio — Kelly sizing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioKellySize:
    """Portfolio.kelly_size must respect fraction and hard dollar caps."""

    def test_basic_kelly_formula(self):
        # edge=0.10, bankroll=1000, kelly_frac=0.5
        # raw=1000×0.10×0.5=50, fraction_cap=1000×0.10=100 → 50
        p = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=None,
                           kelly_fraction=0.5, max_bet_fraction=0.10)
        assert p.kelly_size(0.10) == pytest.approx(50.0)

    def test_fraction_cap_binds(self):
        # edge=0.30 → raw=150 > fraction_cap=100 → 100
        p = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=None,
                           kelly_fraction=0.5, max_bet_fraction=0.10)
        assert p.kelly_size(0.30) == pytest.approx(100.0)

    def test_dollar_cap_tighter_than_fraction_cap(self):
        # raw=50, fraction_cap=100, dollar_cap=10 → 10
        p = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=10.0,
                           kelly_fraction=0.5, max_bet_fraction=0.10)
        assert p.kelly_size(0.10) == pytest.approx(10.0)

    def test_dollar_cap_tighter_on_large_edge(self):
        # edge=0.60, raw=300, fraction_cap=500, dollar_cap=7 → 7
        p = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=7.0,
                           kelly_fraction=0.5, max_bet_fraction=0.50)
        assert p.kelly_size(0.60) == pytest.approx(7.0)

    def test_dollar_cap_looser_than_kelly(self):
        # raw=50, fraction_cap=100, dollar_cap=200 → 50
        p = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=200.0,
                           kelly_fraction=0.5, max_bet_fraction=0.10)
        assert p.kelly_size(0.10) == pytest.approx(50.0)

    def test_zero_edge_returns_zero(self):
        p = make_portfolio()
        assert p.kelly_size(0.0) == 0.0

    def test_negative_edge_returns_zero(self):
        p = make_portfolio()
        assert p.kelly_size(-0.10) == 0.0

    def test_kelly_size_scales_with_bankroll(self):
        p1 = make_portfolio(initial_bankroll=1000.0, max_bet_dollars=None)
        p2 = make_portfolio(initial_bankroll=2000.0, max_bet_dollars=None)
        assert p2.kelly_size(0.10) == pytest.approx(p1.kelly_size(0.10) * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Portfolio — open position tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioOpenPositions:
    """open_position / close_position / has_position / open_positions_for_city."""

    def test_open_position_is_stored(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1"))
        assert p.has_position("T1") is True

    def test_close_position_returns_position(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        pos = make_position(ticker="T1")
        p.open_position(pos)
        ret = p.close_position("T1")
        assert ret is pos

    def test_close_position_removes_from_open(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1"))
        p.close_position("T1")
        assert p.has_position("T1") is False

    def test_close_nonexistent_returns_none(self):
        p = make_portfolio()
        assert p.close_position("NONEXISTENT") is None

    def test_open_positions_for_city_filters_by_city(self):
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", city="Chicago"))
        p.open_position(make_position(ticker="T2", city="NewYork"))
        chicago = p.open_positions_for_city("Chicago")
        assert "T1" in chicago
        assert "T2" not in chicago

    def test_open_positions_for_city_empty_initially(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        assert p.open_positions_for_city("Chicago") == {}

    def test_multiple_positions_same_city(self):
        p = make_portfolio(max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", city="Chicago"))
        p.open_position(make_position(ticker="T2", city="Chicago"))
        chicago = p.open_positions_for_city("Chicago")
        assert len(chicago) == 2

    def test_has_position_false_after_close(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1"))
        p.close_position("T1")
        assert p.has_position("T1") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Portfolio — constructor validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioValidation:
    def _base_kwargs(self, **overrides):
        kw = dict(initial_bankroll=100, kelly_fraction=0.5, max_bet_fraction=0.1,
                  max_daily_trades=5, max_session_trades=5)
        kw.update(overrides)
        return kw

    def test_zero_bankroll_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(initial_bankroll=0))

    def test_negative_bankroll_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(initial_bankroll=-10))

    def test_kelly_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(kelly_fraction=1.5))

    def test_kelly_fraction_zero_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(kelly_fraction=0.0))

    def test_max_bet_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(max_bet_fraction=1.5))

    def test_max_daily_trades_zero_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(max_daily_trades=0))

    def test_max_session_trades_zero_raises(self):
        with pytest.raises(ValueError):
            Portfolio(**self._base_kwargs(max_session_trades=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Portfolio — to_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioDataframe:
    def test_empty_portfolio_returns_empty_df(self):
        df = make_portfolio().to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_one_row_per_trade(self):
        p = make_portfolio(max_daily_trades=10, max_session_trades=10)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        for ticker in ["T1", "T2", "T3"]:
            p.record_trade(make_trade(ticker=ticker))
        df = p.to_dataframe()
        assert len(df) == 3

    def test_key_columns_present(self):
        p = make_portfolio()
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade())
        df = p.to_dataframe()
        for col in ["size", "pnl", "bankroll_after", "ticker", "side", "date"]:
            assert col in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PortfolioManager — per-ticker dollar cap (core invariant)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPMPerTickerCap:
    """
    max_bet_dollars is the TOTAL dollars ever committed to a ticker.
    When ticker_cost is passed, PM must:
      - Skip entry entirely if total_committed >= max_bet_dollars
      - Clamp size to remaining room if total_committed < max_bet_dollars
      - Never produce an intent where size + existing_cost > cap
    """

    # fair_p=0.50, yes_ask=0.40 → yes_edge=0.10 ≥ min_edge=0.05 → BUY YES
    # kelly: 1000×0.10×0.5=50, fraction_cap=100 → min(50,100,cap)
    MARKET = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]

    def test_no_cap_produces_full_kelly(self):
        intents = pm(max_bet_dollars=None).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={}
        )
        buys = [i for i in intents if i.action == "buy"]
        assert buys[0].size == pytest.approx(50.0)

    def test_at_full_cap_blocks_entry(self):
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": 10.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 0

    def test_above_cap_blocks_entry(self):
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": 15.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 0

    def test_zero_committed_caps_at_max_bet_dollars(self):
        # kelly→50, cap=10 → clamped to 10
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": 0.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1
        assert buys[0].size == pytest.approx(10.0)

    def test_absent_ticker_in_cost_dict_treated_as_zero(self):
        # T1 not in ticker_cost → same as committed=0 → size=10
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1
        assert buys[0].size == pytest.approx(10.0)

    def test_partial_commitment_clamps_to_remaining(self):
        # Committed $7, cap=$10 → remaining=$3; kelly→50 → clamped to 3
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": 7.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1
        assert buys[0].size == pytest.approx(3.0)

    def test_no_side_also_respects_cap(self):
        # fair_p=0.70, no_ask=0.20 → no_edge=0.10; committed $8 → remaining=$2
        market = [mkt("T1", T_GEQ63, yes_ask=0.85, no_ask=0.20)]
        intents = pm(max_bet_dollars=10.0).evaluate(
            market, PRED, {}, bankroll=1000, ticker_cost={"T1": 8.0}
        )
        no_buys = [i for i in intents if i.action == "buy" and i.ticker == "T1" and i.side == "no"]
        assert len(no_buys) == 1
        assert no_buys[0].size == pytest.approx(2.0)

    def test_unrelated_ticker_not_affected_by_cap(self):
        # T1 at cap → blocked; T2 has no cost → allowed
        markets = [
            mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62),
            mkt("T2", T_GEQ63, yes_ask=0.40, no_ask=0.65),
        ]
        intents = pm(max_bet_dollars=10.0).evaluate(
            markets, PRED, {}, bankroll=1000, ticker_cost={"T1": 10.0, "T2": 0.0}
        )
        t1_buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        t2_buys = [i for i in intents if i.action == "buy" and i.ticker == "T2"]
        assert len(t1_buys) == 0
        assert len(t2_buys) > 0

    @pytest.mark.parametrize("committed", [0, 1, 2, 4, 6, 8, 9])
    def test_size_plus_committed_never_exceeds_cap(self, committed):
        cap = 10.0
        intents = pm(max_bet_dollars=cap).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": float(committed)}
        )
        for b in intents:
            if b.action == "buy" and b.ticker == "T1":
                total = committed + b.size
                assert total <= cap + 1e-9, (
                    f"committed={committed}, size={b.size:.4f}, cap={cap}: total={total:.4f} > cap"
                )

    def test_contracts_times_price_equals_size_when_capped(self):
        # Clamped to remaining=$7; contract count × price must still equal size
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, {}, bankroll=1000, ticker_cost={"T1": 3.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1
        b = buys[0]
        assert b.contracts * b.price == pytest.approx(b.size, rel=1e-9)

    def test_ticker_cost_none_flat_position_no_restriction(self):
        # ticker_cost=None, net_pos=0 → total_committed=0 → size capped only by max_bet_dollars
        intents = pm(max_bet_dollars=10.0).evaluate(
            self.MARKET, PRED, positions={"T1": 0.0}, bankroll=1000, ticker_cost=None
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1
        assert buys[0].size == pytest.approx(10.0)

    def test_ticker_cost_none_long_no_exposure_used_as_committed(self):
        # ticker_cost=None, net_pos=-25 (long NO at no_ask=0.20)
        # exposure = abs(-25) × 0.20 = 5.0 → remaining = 10 - 5 = 5
        # NO side: net_pos=-25 ≤ 0 → eligible; kelly→50 → clamped to 5
        market = [mkt("T1", T_GEQ63, yes_ask=0.85, no_ask=0.20)]
        intents = pm(max_bet_dollars=10.0).evaluate(
            market, PRED, positions={"T1": -25.0}, bankroll=1000, ticker_cost=None
        )
        no_buys = [i for i in intents if i.action == "buy" and i.side == "no" and i.ticker == "T1"]
        assert len(no_buys) == 1
        assert no_buys[0].size == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Portfolio + PortfolioManager — integrated per-ticker cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegratedPerTickerCap:
    """
    Simulate evaluate() + record_trade() cycles and verify that:
      - portfolio.ticker_cost[ticker] never exceeds max_bet_dollars
      - PortfolioManager blocks further buys once the cap is reached
      - Bankroll and ticker_cost update independently and correctly
    """

    CAP = 10.0
    MARKET = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]

    def _setup(self):
        portfolio = make_portfolio(
            initial_bankroll=1000.0,
            max_bet_dollars=self.CAP,
            max_daily_trades=50,
            max_session_trades=50,
        )
        manager = pm(max_bet_dollars=self.CAP, allow_exits=False)
        portfolio.begin_day("2026-01-01")
        portfolio.begin_session(0)
        return portfolio, manager

    def test_first_buy_is_capped_at_max_bet_dollars(self):
        portfolio, manager = self._setup()
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        buys = [i for i in intents if i.action == "buy"]
        assert len(buys) == 1
        assert buys[0].size == pytest.approx(self.CAP)

    def test_after_full_spend_no_more_t1_intents(self):
        portfolio, manager = self._setup()
        # Round 1: buy fills the cap
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        size = next(i.size for i in intents if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=0.0))
        # Round 2: no room left
        intents2 = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        t1_buys = [i for i in intents2 if i.action == "buy" and i.ticker == "T1"]
        assert len(t1_buys) == 0

    def test_cumulative_spend_never_exceeds_cap(self):
        """Multi-round loop: sum of all T1 buy sizes must not exceed cap."""
        portfolio, manager = self._setup()
        total_spent = 0.0
        for _ in range(10):
            intents = manager.evaluate(
                self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
                ticker_cost=portfolio.ticker_cost
            )
            t1_buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
            if not t1_buys:
                break
            size = t1_buys[0].size
            total_spent += size
            portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=0.0))

        assert total_spent <= self.CAP + 1e-9, (
            f"Total T1 spend ${total_spent:.4f} exceeded cap ${self.CAP}"
        )

    def test_ticker_cost_matches_sum_of_recorded_sizes(self):
        portfolio, manager = self._setup()
        recorded_sizes = []
        for _ in range(5):
            intents = manager.evaluate(
                self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
                ticker_cost=portfolio.ticker_cost
            )
            t1_buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
            if not t1_buys:
                break
            size = t1_buys[0].size
            recorded_sizes.append(size)
            portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=0.0))

        assert portfolio.ticker_cost.get("T1", 0.0) == pytest.approx(sum(recorded_sizes))

    def test_bankroll_updated_correctly_alongside_ticker_cost(self):
        portfolio, manager = self._setup()
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        size = next(i.size for i in intents if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=3.0))
        assert portfolio.bankroll == pytest.approx(1000.0 + 3.0)
        # Winning doesn't retroactively change the entry cost
        assert portfolio.ticker_cost.get("T1", 0.0) == pytest.approx(size)

    def test_bankroll_and_ticker_cost_independent_on_loss(self):
        portfolio, manager = self._setup()
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        size = next(i.size for i in intents if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=-size))
        # Bankroll decreases by size
        assert portfolio.bankroll == pytest.approx(1000.0 - size)
        # ticker_cost is NOT affected by the loss
        assert portfolio.ticker_cost.get("T1", 0.0) == pytest.approx(size)

    def test_ticker_cost_persists_across_new_day(self):
        """Cap is a lifetime tracker: spending on day 1 blocks buys on day 2."""
        portfolio, manager = self._setup()
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        size = next(i.size for i in intents if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=0.0))

        # New day
        portfolio.begin_day("2026-01-02")
        portfolio.begin_session(0)

        intents_day2 = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        t1_day2 = [i for i in intents_day2 if i.action == "buy" and i.ticker == "T1"]

        if size >= self.CAP:
            # Full cap used on day 1 → no buys on day 2
            assert len(t1_day2) == 0, "Lifetime cap must block T1 buys on day 2"
        else:
            # Partial spend → remaining must be respected
            for b in t1_day2:
                total = portfolio.ticker_cost.get("T1", 0.0) + b.size
                assert total <= self.CAP + 1e-9

    def test_exit_trade_does_not_re_open_cap_space(self):
        """Exiting a position must NOT reduce ticker_cost, so cap stays closed."""
        portfolio, manager = self._setup()
        intents = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        size = next(i.size for i in intents if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=0.0))
        # Simulate exit (count_toward_limits=False)
        portfolio.record_trade(make_trade(ticker="T1", size=size, pnl=1.0), count_toward_limits=False)
        # Cap space should NOT be re-opened
        intents_after = manager.evaluate(
            self.MARKET, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        )
        t1_buys = [i for i in intents_after if i.action == "buy" and i.ticker == "T1"]
        if size >= self.CAP:
            assert len(t1_buys) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  PortfolioManager — min_bet_dollars filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPMMinBetFilter:
    """Intents below min_bet_dollars must be suppressed."""

    def test_intent_below_min_bet_suppressed(self):
        # remaining cap = $0.40, min_bet = $1.00 → suppressed
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(max_bet_dollars=10.0, min_bet_dollars=1.0).evaluate(
            market, PRED, {}, bankroll=1000, ticker_cost={"T1": 9.60}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 0

    def test_intent_above_min_bet_allowed(self):
        # remaining cap = $5.00, min_bet = $1.00 → allowed
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(max_bet_dollars=10.0, min_bet_dollars=1.0).evaluate(
            market, PRED, {}, bankroll=1000, ticker_cost={"T1": 5.0}
        )
        buys = [i for i in intents if i.action == "buy" and i.ticker == "T1"]
        assert len(buys) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  PortfolioManager — min/max fair_p filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPMFairPFilter:
    """Near-certain outcomes must be filtered regardless of dollar edge."""

    def test_near_certain_yes_skipped(self):
        # P(tmax >= 65) ≈ 1.0 from spike at 80 → fair_p > max_fair_p=0.95 → skip
        pred = {"Chicago": spike_row(80)}
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(min_fair_p=0.05, max_fair_p=0.95).evaluate(
            market, pred, {}, bankroll=1000
        )
        assert all(i.action != "buy" for i in intents)

    def test_near_certain_no_skipped(self):
        # P(tmax >= 65) = 0 from spike at 50 → fair_p < min_fair_p=0.05 → skip
        pred = {"Chicago": spike_row(50)}
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(min_fair_p=0.05, max_fair_p=0.95).evaluate(
            market, pred, {}, bankroll=1000
        )
        assert all(i.action != "buy" for i in intents)

    def test_mid_range_fair_p_passes_filter(self):
        # fair_p=0.50 → within [0.05, 0.95] → not filtered
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(min_fair_p=0.05, max_fair_p=0.95, min_edge=0.05).evaluate(
            market, PRED, {}, bankroll=1000
        )
        buys = [i for i in intents if i.action == "buy"]
        assert len(buys) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  PortfolioManager — min_confidence filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPMMinConfidence:
    """min_confidence must suppress entries the model isn't confident enough about."""

    def test_low_fair_p_filtered_by_confidence(self):
        # fair_p=0.50, min_confidence=0.30 → need >= 0.80 for YES → skip
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(min_confidence=0.30).evaluate(market, PRED, {}, bankroll=1000)
        yes_buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert len(yes_buys) == 0

    def test_high_fair_p_passes_confidence(self):
        # fair_p=0.70, min_confidence=0.15 → need >= 0.65 → qualifies
        market = [mkt("T1", T_GEQ63, yes_ask=0.40, no_ask=0.65)]
        intents = pm(min_confidence=0.15).evaluate(market, PRED, {}, bankroll=1000)
        yes_buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert len(yes_buys) == 1

    def test_zero_confidence_threshold_allows_all_edge_entries(self):
        market = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
        intents = pm(min_confidence=0.0).evaluate(market, PRED, {}, bankroll=1000)
        yes_buys = [i for i in intents if i.action == "buy" and i.side == "yes"]
        assert len(yes_buys) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  Portfolio — available_cash (intraday budget enforcement)
#
# The bug this section guards against:
#   In the original implementation, record_trade() updated bankroll immediately.
#   When city A settled with a win during the day, portfolio.bankroll rose —
#   and the engine passed bankroll (not available_cash) to PM, so city B got
#   sized off the inflated figure. A $100 → $219 intraday bankroll was observed,
#   letting the model deploy more than the opening balance — impossible since
#   Kalshi contracts settle at midnight, not when placed.
#
# The fix: Portfolio tracks _day_opening_bankroll and _day_deployed.
#   available_cash = max(0, opening_bankroll - deployed_today)
#   This resets each morning via begin_day() to the settled bankroll,
#   but never increases intraday regardless of winning trades.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioAvailableCash:
    """available_cash must reflect opening balance minus deployed, not live bankroll."""

    def test_available_cash_equals_opening_bankroll_at_day_start(self):
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        assert p.available_cash == pytest.approx(100.0)

    def test_available_cash_decreases_by_deployed_amount(self):
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(size=15.0, pnl=0.0))
        assert p.available_cash == pytest.approx(85.0)

    def test_winning_trade_does_not_increase_available_cash_intraday(self):
        """
        THE BUG REGRESSION: city A wins big → bankroll rises, but
        available_cash must NOT rise. Only deployed amount matters intraday.
        """
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        # Deploy $10, win $50 → bankroll goes to $150
        p.record_trade(make_trade(ticker="T1", size=10.0, pnl=50.0))
        assert p.bankroll == pytest.approx(150.0)         # inflated
        assert p.available_cash == pytest.approx(90.0)    # 100 - 10 deployed, NOT 150

    def test_losing_trade_does_not_double_penalise_available_cash(self):
        """Losing reduces bankroll but available_cash only tracks deployed, not losses."""
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(size=10.0, pnl=-10.0))
        assert p.bankroll == pytest.approx(90.0)
        # available_cash = 100 - 10 deployed = 90 (coincidentally equals bankroll here)
        assert p.available_cash == pytest.approx(90.0)

    def test_multiple_wins_do_not_inflate_available_cash(self):
        """Sequential winning trades must not compound into more available_cash."""
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        # Three winning trades, $10 each deployed, $30 won each
        p.record_trade(make_trade(ticker="T1", size=10.0, pnl=30.0))
        p.record_trade(make_trade(ticker="T2", size=10.0, pnl=30.0))
        p.record_trade(make_trade(ticker="T3", size=10.0, pnl=30.0))
        assert p.bankroll == pytest.approx(190.0)      # 100 + 90 won
        assert p.available_cash == pytest.approx(70.0) # 100 - 30 deployed

    def test_multiple_trades_deplete_available_cash_cumulatively(self):
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(ticker="T1", size=10.0, pnl=5.0))
        p.record_trade(make_trade(ticker="T2", size=20.0, pnl=0.0))
        p.record_trade(make_trade(ticker="T3", size=15.0, pnl=-8.0))
        # Deployed = 10 + 20 + 15 = 45; available = 100 - 45 = 55
        assert p.available_cash == pytest.approx(55.0)

    def test_available_cash_floors_at_zero_not_negative(self):
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        # Deploy more than the opening bankroll
        for _ in range(15):
            p.record_trade(make_trade(size=10.0, pnl=0.0))
        assert p.available_cash == pytest.approx(0.0)

    def test_available_cash_resets_to_settled_bankroll_on_begin_day(self):
        """Day 2 opening must use yesterday's settled bankroll (compounding works)."""
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(size=30.0, pnl=50.0))  # bankroll → $150
        p.begin_day("2026-01-02")
        assert p.available_cash == pytest.approx(150.0)

    def test_available_cash_resets_to_floored_bankroll_on_begin_day(self):
        """If day ended with ruin (floored to $1), day 2 starts at $1."""
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.record_trade(make_trade(size=10.0, pnl=-200.0))  # bankroll floors to $1
        p.begin_day("2026-01-02")
        assert p.available_cash == pytest.approx(1.0)

    def test_open_position_reduces_available_cash(self):
        """allow_exits=True: open_position must deplete available_cash immediately."""
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=12.0))
        assert p.available_cash == pytest.approx(88.0)

    def test_exit_settle_trade_does_not_reduce_available_cash(self):
        """count_toward_limits=False (exit/settle) must NOT reduce available_cash."""
        p = make_portfolio(initial_bankroll=100.0)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=12.0))
        cash_after_open = p.available_cash
        p.record_trade(make_trade(ticker="T1", size=12.0, pnl=5.0), count_toward_limits=False)
        assert p.available_cash == pytest.approx(cash_after_open)

    def test_settlement_win_does_not_inflate_available_cash_intraday(self):
        """
        allow_exits=True path: settling a winning position mid-day must not
        increase available_cash so that remaining cities can't use the profit.
        """
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)
        p.begin_day("2026-01-01")
        p.begin_session(0)
        p.open_position(make_position(ticker="T1", size=20.0))   # cash: 80
        p.open_position(make_position(ticker="T2", size=10.0))   # cash: 70
        cash_before_settle = p.available_cash
        # City T1 settles with a huge win
        p.record_trade(make_trade(ticker="T1", size=20.0, pnl=100.0), count_toward_limits=False)
        assert p.bankroll == pytest.approx(200.0)                      # inflated
        assert p.available_cash == pytest.approx(cash_before_settle)   # unchanged

    def test_available_cash_compounding_correct_day_over_day(self):
        """
        Each day starts with the full settled bankroll — compounding works
        correctly, just not intraday.
        """
        p = make_portfolio(initial_bankroll=100.0, max_daily_trades=50, max_session_trades=50)

        p.begin_day("2026-01-01")
        p.begin_session(0)
        assert p.available_cash == pytest.approx(100.0)
        p.record_trade(make_trade(size=10.0, pnl=10.0))   # bankroll → 110

        p.begin_day("2026-01-02")
        assert p.available_cash == pytest.approx(110.0)
        p.begin_session(0)
        p.record_trade(make_trade(size=10.0, pnl=15.0))   # bankroll → 125

        p.begin_day("2026-01-03")
        assert p.available_cash == pytest.approx(125.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  Portfolio + PortfolioManager — intraday budget integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntradayBudget:
    """
    Simulate city-by-city evaluation exactly as the engine does.
    PM must receive available_cash (not bankroll) to prevent intraday recycling.
    """

    MARKET_A = [mkt("T1", T_GEQ65, yes_ask=0.40, no_ask=0.62)]
    MARKET_B = [mkt("T2", T_GEQ63, yes_ask=0.40, no_ask=0.65)]

    def test_city_b_sized_off_available_cash_not_inflated_bankroll(self):
        """
        Core regression test.

        $100 opening, no dollar cap, kelly=0.5, max_fraction=10%:
          City A: deploy $10, win $50 → bankroll=150, available_cash=90
          City B sizing (correct):  kelly_size(90,  edge) → smaller
          City B sizing (bug):      kelly_size(150, edge) → larger

        The test verifies that using available_cash gives a strictly smaller
        bet than using the inflated bankroll — i.e., intraday PnL was not recycled.
        """
        portfolio = make_portfolio(
            initial_bankroll=100.0, max_bet_dollars=None,
            max_daily_trades=50, max_session_trades=50,
            kelly_fraction=0.5, max_bet_fraction=0.10,
        )
        manager = pm(max_bet_dollars=None, kelly_fraction=0.5, max_bet_fraction=0.10)
        portfolio.begin_day("2026-01-01")
        portfolio.begin_session(0)

        # City A: evaluate using available_cash, record a big win
        intents_a = manager.evaluate(
            self.MARKET_A, PRED, {}, bankroll=portfolio.available_cash,
            ticker_cost=portfolio.ticker_cost
        )
        size_a = next(i.size for i in intents_a if i.action == "buy")
        portfolio.record_trade(make_trade(ticker="T1", size=size_a, pnl=50.0))

        assert portfolio.bankroll > 100.0                       # inflated
        assert portfolio.available_cash < portfolio.bankroll    # budget not inflated

        # City B: compare correct (available_cash) vs buggy (bankroll) sizing
        buys_correct = [i for i in manager.evaluate(
            self.MARKET_B, PRED, {}, bankroll=portfolio.available_cash,
            ticker_cost=portfolio.ticker_cost
        ) if i.action == "buy"]
        buys_buggy = [i for i in manager.evaluate(
            self.MARKET_B, PRED, {}, bankroll=portfolio.bankroll,
            ticker_cost=portfolio.ticker_cost
        ) if i.action == "buy"]

        if buys_correct and buys_buggy:
            assert buys_correct[0].size < buys_buggy[0].size, (
                f"available_cash sizing ({buys_correct[0].size:.2f}) should be strictly "
                f"less than bankroll sizing ({buys_buggy[0].size:.2f})"
            )

    def test_total_deployed_bounded_by_opening_bankroll(self):
        """
        Even with big intraday wins, total deployed in one day cannot exceed
        the opening bankroll when sizing is based on available_cash.
        """
        portfolio = make_portfolio(
            initial_bankroll=100.0, max_bet_dollars=None,
            max_daily_trades=50, max_session_trades=50,
            kelly_fraction=0.5, max_bet_fraction=0.10,
        )
        manager = pm(max_bet_dollars=None, kelly_fraction=0.5, max_bet_fraction=0.10)
        portfolio.begin_day("2026-01-01")
        portfolio.begin_session(0)

        total_deployed = 0.0
        for i in range(30):
            intents = manager.evaluate(
                [mkt(f"T{i}", T_GEQ65, yes_ask=0.40, no_ask=0.62)],
                PRED, {}, bankroll=portfolio.available_cash,
                ticker_cost=portfolio.ticker_cost,
            )
            buys = [x for x in intents if x.action == "buy"]
            if not buys or portfolio.available_cash <= 0:
                break
            size = buys[0].size
            total_deployed += size
            portfolio.record_trade(make_trade(ticker=f"T{i}", size=size, pnl=size * 3))

        assert total_deployed <= 100.0 + 1e-9, (
            f"Deployed ${total_deployed:.2f} from $100 opening — intraday PnL recycled"
        )

    def test_available_cash_hits_zero_after_full_budget_deployed(self):
        """
        When fixed-size bets exhaust the opening bankroll, available_cash hits 0
        even though bankroll has grown from intraday wins.
        """
        portfolio = make_portfolio(
            initial_bankroll=100.0, max_bet_dollars=10.0,
            max_daily_trades=50, max_session_trades=50,
        )
        portfolio.begin_day("2026-01-01")
        portfolio.begin_session(0)

        # Deploy 10 × $10 = $100 (full opening bankroll), each with a big win
        for i in range(10):
            portfolio.record_trade(make_trade(ticker=f"T{i}", size=10.0, pnl=50.0))

        # bankroll has ballooned, but available_cash is exhausted
        assert portfolio.bankroll > 100.0
        assert portfolio.available_cash == pytest.approx(0.0)
        assert portfolio._day_deployed == pytest.approx(100.0)
