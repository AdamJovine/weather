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
    mkt_p: float            # real Kalshi close_dollars at entry
    model_p: float          # model's fair probability for the YES side
    edge: float             # |model_p - mkt_p| in our favour
    size: float             # dollar amount wagered
    pnl: float              # realised profit / loss
    outcome: int            # 1 = our side won, 0 = lost
    bankroll_after: float   # bankroll immediately after this trade settles


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

    def record_trade(self, trade: Trade) -> None:
        """
        Update bankroll and increment counters.
        The bankroll floor of $1 prevents log(0) in downstream CAGR calcs.
        """
        self._bankroll = max(self._bankroll + trade.pnl, 1.0)
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
