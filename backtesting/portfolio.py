"""
Portfolio: bankroll management and trade-rate enforcement.

Responsibilities:
  - Track the running bankroll (with a $1 floor to prevent total ruin)
  - Enforce max_daily_trades and max_session_trades limits
  - Compute fractional Kelly bet sizes
  - Record every trade for downstream analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class OpenPosition:
    """
    A position that has been opened but not yet exited or settled.

    Stored in Portfolio._open_positions keyed by ticker.
    Created when allow_exits=True and a trade passes edge/size checks.
    Removed when the position is exited (edge gone) or settled at contract expiry.
    """
    city: str
    ticker: str
    title: str
    contract_type: str      # "geq" | "leq" | "range"
    threshold: int          # geq/leq threshold; low bound for range
    contract_def: dict      # raw dict from parse_contract()
    side: str               # "yes" | "no"
    entry_mkt_p: float      # YES close price at entry
    entry_model_p: float    # model's fair_p at entry
    entry_edge: float       # edge at entry
    size: float             # abstract bet size (same units as Trade.size)
    entry_session: int
    entry_session_ts: int
    entry_date: str         # ISO date string


@dataclass
class Trade:
    """
    One recorded trade from the backtest.

    All fields are plain Python types (no numpy scalars) so the object
    serialises trivially to CSV via pd.DataFrame([vars(t) ...]).
    """
    run_id: int
    date: str               # ISO settlement date (YYYY-MM-DD)
    city: str
    session: int            # session index within the day (0-based)
    session_ts: int         # real candle timestamp (Unix s UTC); 0 if unavailable
    ticker: str             # Kalshi market ticker, e.g. KXHIGHNYC-26MAR17-T59
    title: str              # human-readable market title
    contract_type: str      # "geq" | "leq" | "range"
    threshold: int          # threshold for geq/leq; low bound for range
    side: str               # "yes" | "no"
    mkt_p: float            # real Kalshi close_dollars at entry (or exit price for exit trades)
    model_p: float          # model's fair probability for the YES side
    edge: float             # |model_p - mkt_p| in our favour
    size: float             # dollar amount wagered
    pnl: float              # realised profit / loss
    outcome: int            # 1 = our side won, 0 = lost
    bankroll_after: float   # bankroll immediately after this trade settles
    # Fields added for exit tracking (have defaults for backward compatibility)
    trade_type: str = "settle"   # "settle" (held to expiry) | "exit" (early close)
    entry_mkt_p: float = 0.0     # YES close price when position was opened


class Portfolio:
    """
    Manages bankroll and enforces per-day / per-session trading limits.

    The limits are enforced at the point of calling can_trade(), which the
    engine checks before placing each trade. This makes the logic explicit
    and testable rather than scattered through the engine.

    Trade-limit semantics:
      max_daily_trades   – total fills across ALL cities in a calendar day
      max_session_trades – total fills within ONE session (evaluation pass)

    Bankroll:
      - Starts at initial_bankroll.
      - Updated after each trade: bankroll += pnl.
      - Floored at $1.00 to prevent total ruin from crashing compounding.
    """

    def __init__(
        self,
        initial_bankroll: float,
        kelly_fraction: float,
        max_bet_fraction: float,
        max_daily_trades: int,
        max_session_trades: int,
        max_bet_dollars: Optional[float] = None,
    ) -> None:
        if initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive")
        if not (0 < kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not (0 < max_bet_fraction <= 1):
            raise ValueError("max_bet_fraction must be in (0, 1]")
        if max_daily_trades < 1:
            raise ValueError("max_daily_trades must be >= 1")
        if max_session_trades < 1:
            raise ValueError("max_session_trades must be >= 1")

        self._bankroll = float(initial_bankroll)
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.max_bet_dollars = max_bet_dollars
        self.max_daily_trades = max_daily_trades
        self.max_session_trades = max_session_trades

        self._current_day: Optional[str] = None
        self._day_trade_count: int = 0

        self._current_session: int = -1
        self._session_trade_count: int = 0

        self._trades: list[Trade] = []

        # Open positions (populated when allow_exits=True)
        # Keyed by ticker; removed on exit or settlement.
        self._open_positions: dict[str, "OpenPosition"] = {}

    # ── day / session state transitions ──────────────────────────────────────

    def begin_day(self, date: str) -> None:
        """
        Must be called once at the start of each new calendar day.
        Resets daily and session counters.
        """
        self._current_day = date
        self._day_trade_count = 0
        self._current_session = -1
        self._session_trade_count = 0

    def begin_session(self, session: int) -> None:
        """
        Must be called at the start of each new session within a day.
        Resets the per-session counter only when the session index changes.
        """
        if session != self._current_session:
            self._current_session = session
            self._session_trade_count = 0

    # ── capacity queries ─────────────────────────────────────────────────────

    def can_trade(self) -> bool:
        """True iff both daily and session limits permit one more trade."""
        return (
            self._day_trade_count < self.max_daily_trades
            and self._session_trade_count < self.max_session_trades
        )

    def remaining_today(self) -> int:
        return max(0, self.max_daily_trades - self._day_trade_count)

    def remaining_this_session(self) -> int:
        return max(0, self.max_session_trades - self._session_trade_count)

    # ── open position tracking ────────────────────────────────────────────────

    def open_position(self, pos: "OpenPosition") -> None:
        """
        Register an open position and count it against trade limits.

        Called when allow_exits=True and a trade passes Kelly sizing.
        The actual PnL is recorded later via record_trade() at exit or settlement.
        """
        self._open_positions[pos.ticker] = pos
        self._day_trade_count += 1
        self._session_trade_count += 1

    def close_position(self, ticker: str) -> Optional["OpenPosition"]:
        """Remove and return the open position for ticker, or None if not found."""
        return self._open_positions.pop(ticker, None)

    def has_position(self, ticker: str) -> bool:
        return ticker in self._open_positions

    def open_positions_for_city(self, city: str) -> dict[str, "OpenPosition"]:
        return {t: p for t, p in self._open_positions.items() if p.city == city}

    # ── sizing ───────────────────────────────────────────────────────────────

    def kelly_size(self, edge: float) -> float:
        """
        Fractional Kelly bet size, capped at max_bet_fraction of current bankroll.

          kelly_amount = bankroll * edge * kelly_fraction
          cap          = bankroll * max_bet_fraction
          size         = min(kelly_amount, cap)
        """
        if edge <= 0:
            return 0.0
        kelly = self._bankroll * edge * self.kelly_fraction
        size = min(kelly, self._bankroll * self.max_bet_fraction)
        if self.max_bet_dollars is not None:
            size = min(size, self.max_bet_dollars)
        return float(size)

    # ── recording ────────────────────────────────────────────────────────────

    def record_trade(self, trade: Trade, count_toward_limits: bool = True) -> None:
        """
        Update bankroll and (optionally) increment trade-rate counters.

        count_toward_limits=True  → legacy path (allow_exits=False); each
                                    record_trade() call uses a trade slot.
        count_toward_limits=False → allow_exits path; the slot was already
                                    consumed when open_position() was called,
                                    so exit/settle trades must NOT count again.
        """
        self._bankroll = max(self._bankroll + trade.pnl, 1.0)
        if count_toward_limits:
            self._day_trade_count += 1
            self._session_trade_count += 1
        self._trades.append(trade)

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def bankroll(self) -> float:
        return self._bankroll

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._trades])
