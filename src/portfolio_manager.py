"""
PortfolioManager: unified trade decision engine for live trading and backtesting.

Both run_live.py and the backtesting engine instantiate one PortfolioManager and
call evaluate() each cycle.  The manager returns OrderIntent objects; execution
(Kalshi API calls or simulated fills) is the caller's responsibility.

Shared logic
────────────
  kelly_size     Fractional Kelly bet sizing with hard caps
  evaluate_exits   Exit when market price reaches fair value (edge gone)
  evaluate_entries Enter when edge ≥ min_edge, Kelly-sized

Position format expected by evaluate()
───────────────────────────────────────
  positions: dict[str, float]
    ticker → net YES contracts
    positive = long yes, negative = long no, 0/absent = flat

Markets format expected by evaluate()
──────────────────────────────────────
  list of dicts, each with:
    ticker          str
    title           str
    _city           str   (must match a key in pred_rows)
    yes_ask_dollars float [0, 1]
    no_ask_dollars  float [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.config import (
    KELLY_FRACTION, MAX_BET_FRACTION, TRADE_FEE_BUFFER, TRADE_MODEL_BUFFER,
    MIN_BET_DOLLARS, CITY_MULTIPLIER_CAP,
)
from src.pricing import compute_fair_prob
from src.strategy import parse_contract


@dataclass
class OrderIntent:
    """
    A desired trade action returned by PortfolioManager.evaluate().

    action="buy"  → open a new position at price (limit buy)
    action="sell" → close an existing position at price (limit sell)
    """
    ticker: str
    title: str
    city: str
    side: str         # "yes" | "no"
    action: str       # "buy" | "sell"
    fair_p: float     # model's fair prob for YES side
    price: float      # limit price in dollars [0, 1]
    edge: float       # expected edge (> 0 for entries; 0.0 for exits)
    size: float       # Kelly bet in dollars
    contracts: float  # size / price (for buys) or contracts held (for sells)
    contract_def: dict


class PortfolioManager:
    """
    Stateless buy/sell decision engine shared by live trading and backtesting.

    Parameters
    ----------
    kelly_fraction       Fraction of full Kelly (default ⅓)
    max_bet_fraction     Hard cap as fraction of bankroll per trade
    max_bet_dollars      Hard dollar cap per trade (None = disabled)
    min_edge             Minimum required edge to enter.  Includes all buffers.
                         Default = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER = 0.05
    min_confidence       Only enter when the model is at least 0.5 + min_confidence
                         certain the outcome goes our way.
                         0.0 = no filter (any fair_p with positive edge qualifies).
                         0.15 = only trade when model is ≥65% confident.
                         Directly controls win rate: expected win rate ≈ fair_p,
                         so higher min_confidence → higher win rate, fewer trades.
    allow_exits          If False, never emit sell intents (hold to settlement)
    exit_edge_threshold  Exit when edge drops below this value.
                         0.0 = exit exactly when market reaches fair value.
    city_multiplier_cap  UCB multiplier upper bound for Kelly scaling.
    """

    def __init__(
        self,
        kelly_fraction: float = KELLY_FRACTION,
        max_bet_fraction: float = MAX_BET_FRACTION,
        max_bet_dollars: Optional[float] = None,
        min_bet_dollars: float = MIN_BET_DOLLARS,
        min_edge: float = TRADE_FEE_BUFFER + TRADE_MODEL_BUFFER,
        min_confidence: float = 0.0,
        allow_exits: bool = True,
        exit_edge_threshold: float = 0.0,
        city_multiplier_cap: float = CITY_MULTIPLIER_CAP,
        min_fair_p: float = 0.05,
        max_fair_p: float = 0.95,
        min_contract_volume: float = 0.0,
        cash_reserve_fraction: float = 0.0,
        rotation_min_edge_gain: float = 0.0,
        max_session_trades: Optional[int] = None,
        min_mkt_price: float = 0.0,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.max_bet_dollars = max_bet_dollars
        self.min_bet_dollars = min_bet_dollars
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.allow_exits = allow_exits
        self.exit_edge_threshold = exit_edge_threshold
        self.min_fair_p = min_fair_p
        self.max_fair_p = max_fair_p
        self.city_multiplier_cap = city_multiplier_cap
        self.min_contract_volume = min_contract_volume
        self.cash_reserve_fraction = cash_reserve_fraction
        self.rotation_min_edge_gain = rotation_min_edge_gain
        self.max_session_trades = max_session_trades
        self.min_mkt_price = min_mkt_price

    @classmethod
    def from_config(cls, cfg) -> "PortfolioManager":
        """Construct from a BacktestConfig (or any object with the same fields)."""
        return cls(
            kelly_fraction=cfg.kelly_fraction,
            max_bet_fraction=cfg.max_bet_fraction,
            max_bet_dollars=cfg.max_bet_dollars,
            min_edge=cfg.min_edge,
            min_confidence=cfg.min_confidence,
            allow_exits=cfg.allow_exits,
            exit_edge_threshold=cfg.exit_edge_threshold,
            min_fair_p=getattr(cfg, "min_fair_p", 0.05),
            max_fair_p=getattr(cfg, "max_fair_p", 0.95),
            min_contract_volume=getattr(cfg, "min_contract_volume", 0.0),
            min_mkt_price=getattr(cfg, "min_mkt_price", 0.0),
            cash_reserve_fraction=getattr(cfg, "cash_reserve_fraction", 0.0),
            rotation_min_edge_gain=getattr(cfg, "rotation_min_edge_gain", 0.0),
        )

    # ── sizing ────────────────────────────────────────────────────────────────

    def kelly_size(
        self,
        bankroll: float,
        edge: float,
        city_multiplier: float = 1.0,
    ) -> float:
        """
        Fractional Kelly bet size in dollars.

          kelly = bankroll × edge × kelly_fraction × city_multiplier
          cap   = bankroll × max_bet_fraction × min(city_multiplier, cap)
          size  = min(kelly, cap, max_bet_dollars)
        """
        if edge <= 0 or bankroll <= 0:
            return 0.0
        kelly = bankroll * edge * self.kelly_fraction * city_multiplier
        cap = bankroll * self.max_bet_fraction * min(city_multiplier, self.city_multiplier_cap)
        size = min(kelly, cap)
        if self.max_bet_dollars is not None:
            size = min(size, self.max_bet_dollars)
        return float(size)

    # ── main evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        markets: list[dict],
        pred_rows: dict[str, "pd.Series"],
        positions: dict[str, float],
        bankroll: float,
        city_multipliers: dict[str, float] | None = None,
        ticker_cost: dict[str, float] | None = None,
        session_slots: Optional[int] = None,
    ) -> list[OrderIntent]:
        """
        Evaluate all markets for this cycle and return trade intents.

        Returns sell intents first (reduce risk before adding exposure),
        then buy intents sorted descending by edge.

        Parameters
        ----------
        markets         Market dicts with ticker, title, _city,
                        yes_ask_dollars, no_ask_dollars
        pred_rows       city → probability distribution Series
        positions       ticker → net YES contracts (+yes, -no); absent = flat
        bankroll        Current available capital
        city_multipliers  Optional UCB multipliers keyed by city name
        session_slots   Max buy intents to return this call.  Overrides
                        self.max_session_trades; use when splitting a session
                        budget across multiple evaluate() calls (e.g. today +
                        tomorrow).  None = use self.max_session_trades.
        """
        if city_multipliers is None:
            city_multipliers = {}

        sells = self._evaluate_exits(markets, pred_rows, positions) if self.allow_exits else []
        buys  = self._evaluate_entries(markets, pred_rows, positions, bankroll, city_multipliers, ticker_cost)
        if self.cash_reserve_fraction > 0 or self.rotation_min_edge_gain > 0:
            sells, buys = self._apply_capital_management(sells, buys, markets, pred_rows, positions, bankroll)

        # Enforce per-session trade limit (buys already sorted by edge desc).
        limit = session_slots if session_slots is not None else self.max_session_trades
        if limit is not None:
            buys = buys[:limit]

        return sells + buys

    # ── exit logic ────────────────────────────────────────────────────────────

    def _evaluate_exits(
        self,
        markets: list[dict],
        pred_rows: dict[str, "pd.Series"],
        positions: dict[str, float],
    ) -> list[OrderIntent]:
        """
        Return sell intents for positions that have lost their edge.

        Exit condition (per side):
          long YES: yes_bid  ≥ fair_p − exit_edge_threshold
          long NO:  no_bid   ≥ (1−fair_p) − exit_edge_threshold

        Bid prices are approximated via the binary-market complement:
          yes_bid ≈ 1 − no_ask
          no_bid  ≈ 1 − yes_ask
        """
        markets_by_ticker = {m.get("ticker", ""): m for m in markets}
        exits = []

        for ticker, net_pos in positions.items():
            if net_pos == 0:
                continue

            market = markets_by_ticker.get(ticker)
            if market is None:
                continue

            city = market.get("_city")
            if not city or city not in pred_rows:
                continue

            try:
                contract = parse_contract(market.get("title", ""), "")
                fair_p   = float(compute_fair_prob(pred_rows[city], contract))
            except Exception:
                continue

            yes_ask = float(market.get("yes_ask_dollars") or 0)
            no_ask  = float(market.get("no_ask_dollars")  or 0)
            yes_bid = max(0.0, 1.0 - no_ask)
            no_bid  = max(0.0, 1.0 - yes_ask)

            side = "yes" if net_pos > 0 else "no"
            held = abs(net_pos)

            if side == "yes":
                should_exit = yes_bid >= fair_p - self.exit_edge_threshold
                exit_price  = yes_bid
            else:
                should_exit = no_bid >= (1.0 - fair_p) - self.exit_edge_threshold
                exit_price  = no_bid

            if not should_exit:
                continue

            exits.append(OrderIntent(
                ticker=ticker,
                title=market.get("title", ""),
                city=city,
                side=side,
                action="sell",
                fair_p=fair_p,
                price=exit_price,
                edge=0.0,
                size=held * exit_price,
                contracts=held,
                contract_def=contract,
            ))

        return exits

    # ── capital management ────────────────────────────────────────────────────

    def _apply_capital_management(self, sells, buys, markets, pred_rows, positions, bankroll):
        """
        Apply cash reserve and capital rotation constraints to the proposed orders.

        Returns (sells, buys) where sells may include rotation exits and buys
        is filtered to only those that fit within the available capital.
        """
        markets_by_ticker = {m.get("ticker", ""): m for m in markets}
        already_exiting = {s.ticker for s in sells}

        held = {}
        for ticker, net_pos in positions.items():
            if net_pos == 0 or ticker in already_exiting:
                continue
            market = markets_by_ticker.get(ticker)
            if market is None:
                continue
            city = market.get("_city")
            if not city or city not in pred_rows:
                continue
            try:
                contract = parse_contract(market.get("title", ""), "")
                fair_p = float(compute_fair_prob(pred_rows[city], contract))
            except Exception:
                continue
            side = "yes" if net_pos > 0 else "no"
            contracts = abs(net_pos)
            yes_ask = float(market.get("yes_ask_dollars") or 0)
            no_ask = float(market.get("no_ask_dollars") or 0)
            yes_bid = max(0.0, 1.0 - no_ask)
            no_bid = max(0.0, 1.0 - yes_ask)
            ask = yes_ask if side == "yes" else no_ask
            exit_price = yes_bid if side == "yes" else no_bid
            current_edge = (fair_p - yes_bid) if side == "yes" else ((1.0 - fair_p) - no_bid)
            held[ticker] = {
                "edge": current_edge,
                "contracts": contracts,
                "exposure": contracts * ask,
                "exit_price": exit_price,
                "side": side,
                "fair_p": fair_p,
                "market": market,
                "contract": contract,
                "city": city,
            }

        total_exposure = sum(h["exposure"] for h in held.values())
        pending_proceeds = sum(s.size for s in sells)
        max_deployable = bankroll * (1.0 - self.cash_reserve_fraction)
        free_capital = max(0.0, max_deployable - total_exposure + pending_proceeds)

        rotation_sells = []
        filtered_buys = []

        for buy in buys:
            if free_capital >= buy.size:
                filtered_buys.append(buy)
                free_capital -= buy.size
                continue

            if self.rotation_min_edge_gain > 0 and held:
                candidates = [
                    (t, h) for t, h in held.items()
                    if h["edge"] < buy.edge - self.rotation_min_edge_gain
                ]
                if candidates:
                    ticker_out, h = min(candidates, key=lambda x: x[1]["edge"])
                    rotation_sells.append(OrderIntent(
                        ticker=ticker_out,
                        title=h["market"].get("title", ""),
                        city=h["city"],
                        side=h["side"],
                        action="sell",
                        fair_p=h["fair_p"],
                        price=h["exit_price"],
                        edge=0.0,
                        size=h["contracts"] * h["exit_price"],
                        contracts=h["contracts"],
                        contract_def=h["contract"],
                    ))
                    free_capital += h["exposure"]
                    del held[ticker_out]
                    already_exiting.add(ticker_out)
                    if free_capital >= buy.size:
                        filtered_buys.append(buy)
                        free_capital -= buy.size

        return sells + rotation_sells, filtered_buys

    # ── entry logic ───────────────────────────────────────────────────────────

    def _evaluate_entries(
        self,
        markets: list[dict],
        pred_rows: dict[str, "pd.Series"],
        positions: dict[str, float],
        bankroll: float,
        city_multipliers: dict[str, float],
        ticker_cost: dict[str, float] | None = None,
    ) -> list[OrderIntent]:
        """
        Return buy intents for markets with edge ≥ min_edge.

        Skips tickers that already have a position in the same direction
        (no doubling-up on open positions).
        """
        entries = []

        for market in markets:
            ticker = market.get("ticker", "")
            city   = market.get("_city")
            if not city or city not in pred_rows:
                continue

            try:
                contract = parse_contract(market.get("title", ""), "")
                fair_p   = float(compute_fair_prob(pred_rows[city], contract))
            except Exception:
                continue

            # Skip near-certain outcomes: outcome is essentially decided and
            # the remaining edge is too small to justify the model-error risk.
            if fair_p < self.min_fair_p or fair_p > self.max_fair_p:
                continue

            yes_ask = float(market.get("yes_ask_dollars") or 0)
            no_ask  = float(market.get("no_ask_dollars")  or 0)
            if yes_ask <= 0 or no_ask <= 0:
                continue

            # Skip contracts where the market price of the side we'd buy is too low.
            # A 1-2¢ ask means the market is highly skeptical; our model is unlikely
            # to be right when it disagrees this strongly with the consensus.
            if self.min_mkt_price > 0:
                if fair_p > 0.5 and yes_ask < self.min_mkt_price:
                    continue
                if fair_p <= 0.5 and no_ask < self.min_mkt_price:
                    continue

            # Skip illiquid contracts — use candle volume (backtest) or
            # volume_fp (live Kalshi API); absent = pass through (no filter)
            vol = market.get("volume") or market.get("volume_fp")
            if vol is not None and self.min_contract_volume > 0 and float(vol) < self.min_contract_volume:
                continue

            net_pos = positions.get(ticker, 0.0)
            mult    = city_multipliers.get(city, 1.0)

            # Total dollars already committed to this ticker across the entire run.
            # When ticker_cost is provided it includes both open and already-settled
            # entries, so the $10 cap is enforced across multiple trading days
            # (prevents buying the same contract as "tomorrow" on Day D and again
            # as "today" on Day D+1).  When ticker_cost is None (e.g. called from
            # live trading without history), fall back to open-position exposure.
            if ticker_cost is not None:
                total_committed = ticker_cost.get(ticker, 0.0)
            elif net_pos > 0:
                total_committed = net_pos * yes_ask
            elif net_pos < 0:
                total_committed = abs(net_pos) * no_ask
            else:
                total_committed = 0.0

            # Skip if already at or above the per-ticker dollar cap.
            if self.max_bet_dollars is not None and total_committed >= self.max_bet_dollars:
                continue

            # How much room is left up to the cap for this ticker.
            remaining_cap = (
                self.max_bet_dollars - total_committed
                if self.max_bet_dollars is not None
                else None
            )

            # YES side — only if not already long no and model is confident enough
            yes_edge = fair_p - yes_ask
            yes_conf_ok = self.min_confidence <= 0.0 or fair_p >= 0.5 + self.min_confidence
            if yes_edge >= self.min_edge and yes_conf_ok and net_pos >= 0:
                size      = self.kelly_size(bankroll, yes_edge, mult)
                if remaining_cap is not None:
                    size = min(size, remaining_cap)
                contracts = size / yes_ask if size > 0 else 0.0
                if contracts > 0 and size >= self.min_bet_dollars:
                    entries.append(OrderIntent(
                        ticker=ticker,
                        title=market.get("title", ""),
                        city=city,
                        side="yes",
                        action="buy",
                        fair_p=fair_p,
                        price=yes_ask,
                        edge=yes_edge,
                        size=size,
                        contracts=contracts,
                        contract_def=contract,
                    ))

            # NO side — only if not already long yes and model is confident enough
            no_edge = (1.0 - fair_p) - no_ask
            no_conf_ok = self.min_confidence <= 0.0 or (1.0 - fair_p) >= 0.5 + self.min_confidence
            if no_edge >= self.min_edge and no_conf_ok and net_pos <= 0:
                size      = self.kelly_size(bankroll, no_edge, mult)
                if remaining_cap is not None:
                    size = min(size, remaining_cap)
                contracts = size / no_ask if size > 0 else 0.0
                if contracts > 0 and size >= self.min_bet_dollars:
                    entries.append(OrderIntent(
                        ticker=ticker,
                        title=market.get("title", ""),
                        city=city,
                        side="no",
                        action="buy",
                        fair_p=fair_p,
                        price=no_ask,
                        edge=no_edge,
                        size=size,
                        contracts=contracts,
                        contract_def=contract,
                    ))

        entries.sort(key=lambda x: -x.edge)
        return entries
