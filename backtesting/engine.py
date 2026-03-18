"""
BacktestEngine: walk-forward backtesting orchestrator.

Uses real Kalshi historical prices from KalshiPriceStore — no synthetic prices.

Flow per run
────────────
For each calendar day in [start_date, end_date]:
  portfolio.begin_day(date)

  For each city (shuffled for fairness each day):
    Skip if:
      - No price history for (city, date)
      - Insufficient training data
      - Any model feature is NaN for this row

    Fit model on lookback window (or reuse if refit not due)

    Compute full temperature probability distribution ONCE via
    predict_integer_probs() — this is reused across all sessions that day.

    Select sessions from price store, capped at sessions_per_day.

    For each session:
      portfolio.begin_session(session_index)
      If trade limits are exhausted → skip remaining sessions for this city
      Evaluate each available contract via compute_fair_prob()
      Record fills (up to remaining trade capacity)

Sessions
────────
sessions_per_day controls how many of the real hourly candles to use per day:
  - If real candle count <= sessions_per_day: use all
  - If real candle count >  sessions_per_day: subsample evenly
  - sessions_per_day = 0: use all (no cap)

This lets you study how trading frequency affects outcomes:
  sessions_per_day=1 → trade at most once per city per day (first session only)
  sessions_per_day=3 → use up to 3 evenly-spaced snapshots during the day
  sessions_per_day=0 → use every available candle (~12–20 per city per day)

Contract pricing
────────────────
predict_integer_probs() gives P(tmax == k) for k in [-20, 129].
compute_fair_prob() sums the relevant slice of this distribution.
Edge = |fair_p - close_dollars|. We trade YES if fair_p > close_dollars,
NO if close_dollars > fair_p — whichever side has edge ≥ min_edge.

Model caching
─────────────
Refitting is expensive. The engine caches one model per city and only
refits when (current_date - last_refit_date).days >= refit_every.
"""
from __future__ import annotations

from datetime import date as dt_date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtesting.config import BacktestConfig
from backtesting.data_loader import BacktestDataset
from backtesting.market_data import KalshiPriceStore, Session
from backtesting.portfolio import Portfolio, Trade, OpenPosition
from src.portfolio_manager import PortfolioManager, OrderIntent

# A model factory is any callable that takes no arguments and returns a
# fitted-model-ready instance. Used by the hyperparameter tuner to inject
# models with trial-specific hyperparameters.
ModelFactory = Optional[Callable[[], object]]


class BacktestEngine:
    """Orchestrates N independent walk-forward backtest runs."""

    def __init__(
        self,
        dataset: BacktestDataset,
        config: BacktestConfig,
        price_store: KalshiPriceStore,
        model_factory: ModelFactory = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.price_store = price_store
        # When set, overrides the default MODEL_REGISTRY instantiation.
        # The factory is called once per model refit: () -> fresh model instance.
        self._model_factory = model_factory

    def run(self) -> "BacktestResults":  # noqa: F821
        from backtesting.results import BacktestResults, RunResult

        run_results: list[RunResult] = []
        for k in range(self.config.n_runs):
            seed = self.config.seed + k
            if self.config.n_runs > 1 or self.config.verbose:
                print(f"\n── Run {k + 1}/{self.config.n_runs}  (seed={seed}) ──")
            result = self._run_single(run_id=k + 1, seed=seed)
            run_results.append(result)
            _print_run_summary(result)

        return BacktestResults(
            runs=run_results,
            config=self.config,
            validation_report=self.dataset.validation_report,
        )

    # ── single run ────────────────────────────────────────────────────────────

    def _run_single(self, run_id: int, seed: int) -> "RunResult":  # noqa: F821
        from backtesting.results import RunResult

        np.random.seed(seed)
        cfg = self.config

        portfolio = Portfolio(
            initial_bankroll=cfg.initial_bankroll,
            kelly_fraction=cfg.kelly_fraction,
            max_bet_fraction=cfg.max_bet_fraction,
            max_bet_dollars=cfg.max_bet_dollars,
            max_daily_trades=cfg.max_daily_trades,
            max_session_trades=cfg.max_session_trades,
        )

        city_frames = self.dataset.city_frames
        # model_state: city → (fitted_model, last_refit_date)
        model_state: dict[str, tuple] = {}

        day_groups = self._build_day_groups(city_frames, cfg)

        pm = PortfolioManager.from_config(cfg)

        for date, city_records in day_groups:
            portfolio.begin_day(date.isoformat()[:10])

            # Shuffle city order each day so no city systematically trades first
            shuffled = list(city_records)
            np.random.shuffle(shuffled)

            for city, row_i, city_df, settlement, trade_date_str in shuffled:
                if portfolio.remaining_today() == 0:
                    break

                sessions = self._pick_sessions(city, settlement, cfg)
                if not sessions:
                    continue

                model = self._get_model(city, city_df, row_i, date, model_state, cfg)
                if model is None:
                    continue

                # ── compute probability distribution ONCE per (city, settlement) ──
                try:
                    prob_df = model.predict_integer_probs(city_df.iloc[[row_i]])
                    prob_row = prob_df.iloc[0]
                except Exception as exc:
                    if cfg.verbose:
                        print(f"    WARN: predict_integer_probs failed ({city}): {exc}")
                    continue

                test_row = city_df.iloc[row_i]

                for s_idx, session in enumerate(sessions):
                    portfolio.begin_session(s_idx)

                    if cfg.allow_exits:
                        markets = _session_to_markets(session, city, cfg.half_spread, cfg.pessimistic_pricing)
                        bt_positions = _backtest_positions(portfolio, city)
                        intents = pm.evaluate(markets, {city: prob_row}, bt_positions, portfolio.available_cash,
                                              ticker_cost=portfolio.ticker_cost)

                        for intent in intents:
                            if intent.action == "sell":
                                pos = portfolio.close_position(intent.ticker)
                                if pos is None:
                                    continue
                                exit_trade = _build_exit_trade(
                                    run_id=run_id, city=city, s_idx=s_idx,
                                    session_ts=session.ts,
                                    pos=pos, exit_price=intent.price, fair_p=intent.fair_p,
                                    portfolio=portfolio, fee_rate=cfg.fee_rate,
                                )
                                portfolio.record_trade(exit_trade, count_toward_limits=False)

                            elif intent.action == "buy":
                                if not portfolio.can_trade():
                                    break
                                cash_at_entry = portfolio.available_cash
                                portfolio.open_position(OpenPosition(
                                    city=city,
                                    ticker=intent.ticker,
                                    title=intent.title,
                                    contract_type=intent.contract_def["market_type"],
                                    threshold=_threshold_from_def(intent.contract_def),
                                    contract_def=intent.contract_def,
                                    side=intent.side,
                                    entry_mkt_p=intent.price,
                                    entry_model_p=intent.fair_p,
                                    entry_edge=intent.edge,
                                    size=intent.size,
                                    entry_session=s_idx,
                                    entry_session_ts=session.ts,
                                    entry_date=trade_date_str,
                                    entry_available_cash=cash_at_entry,
                                ))
                    else:
                        if not portfolio.can_trade():
                            break
                        markets = _session_to_markets(session, city, cfg.half_spread, cfg.pessimistic_pricing)
                        intents = pm.evaluate(markets, {city: prob_row}, {}, portfolio.available_cash,
                                              ticker_cost=portfolio.ticker_cost)
                        for intent in intents:
                            if intent.action != "buy":
                                continue
                            if not portfolio.can_trade():
                                break
                            cash_at_entry = portfolio.available_cash
                            trade = _build_trade_from_intent(
                                run_id=run_id, city=city, session=s_idx,
                                session_ts=session.ts, trade_date_str=trade_date_str,
                                test_row=test_row,
                                intent=intent, portfolio=portfolio,
                                fee_rate=cfg.fee_rate,
                                available_cash=cash_at_entry,
                            )
                            if trade is not None:
                                portfolio.record_trade(trade)

                # Settle any positions still open for this city (not exited)
                if cfg.allow_exits:
                    for ticker, pos in list(portfolio.open_positions_for_city(city).items()):
                        portfolio.close_position(ticker)
                        settle = _build_settlement_trade(
                            run_id=run_id,
                            city=city,
                            test_row=test_row,
                            pos=pos,
                            portfolio=portfolio,
                            fee_rate=cfg.fee_rate,
                        )
                        portfolio.record_trade(settle, count_toward_limits=False)

        return RunResult(
            run_id=run_id,
            seed=seed,
            trades=portfolio.to_dataframe(),
            config=cfg,
        )

    # ── day-group builder ─────────────────────────────────────────────────────

    def _build_day_groups(
        self,
        city_frames: dict[str, pd.DataFrame],
        cfg: BacktestConfig,
    ) -> list[tuple[pd.Timestamp, list[tuple[str, int, pd.DataFrame, "dt_date", str]]]]:
        """
        Build an ordered list of (trade_date, [(city, row_idx, city_df, settlement_date, trade_date_str), ...]).

        Each entry represents one (city, settlement_date) pair to evaluate on trade_date.
        When cfg.trade_tomorrow=True, two entries are produced per city per day — mirroring
        run_live.py's TARGET_DATE + TOMORROW_DATE logic:
          - today's market:    settlement_date == trade_date
          - tomorrow's market: settlement_date == trade_date + 1

        Only includes rows where:
          - date is in [start_date, end_date]
          - y_tmax is observed
          - at least min_train_rows of prior data exist
          - no model feature is NaN
          - price history exists for (city, settlement_date)
        """
        from src.model import FEATURES

        start = pd.Timestamp(cfg.start_date)
        end = pd.Timestamp(cfg.end_date)

        day_map: dict[pd.Timestamp, list] = {}

        for city, city_df in city_frames.items():
            for i in range(cfg.min_train_rows, len(city_df)):
                row = city_df.iloc[i]
                date = pd.Timestamp(row["date"])

                if date < start or date > end:
                    continue
                if pd.isna(row.get("y_tmax")):
                    continue
                if city_df.iloc[[i]][FEATURES].isnull().any(axis=1).iloc[0]:
                    continue

                trade_date_str = date.date().isoformat()

                # Today's market: snapshot taken today, settles today
                # Mirrors TARGET_DATE in run_live.py
                if self.price_store.has_data(city, date.date()):
                    day_map.setdefault(date, []).append(
                        (city, i, city_df, date.date(), trade_date_str)
                    )

                # Tomorrow's market: snapshot taken today, settles tomorrow.
                # Mirrors TOMORROW_DATE in run_live.py.
                # Uses the D+1 feature row so the model predicts tomorrow's temp,
                # but records the trade as occurring on today (trade_date_str).
                if cfg.trade_tomorrow and i + 1 < len(city_df):
                    next_row = city_df.iloc[i + 1]
                    next_date = pd.Timestamp(next_row["date"])
                    tomorrow = date + pd.Timedelta(days=1)
                    if (
                        next_date == tomorrow
                        and tomorrow <= end
                        and not pd.isna(next_row.get("y_tmax"))
                        and not city_df.iloc[[i + 1]][FEATURES].isnull().any(axis=1).iloc[0]
                        and self.price_store.has_data(city, tomorrow.date())
                    ):
                        day_map.setdefault(date, []).append(
                            (city, i + 1, city_df, tomorrow.date(), trade_date_str)
                        )

        return sorted(day_map.items())

    def _pick_sessions(
        self, city: str, settlement: dt_date, cfg: BacktestConfig
    ) -> list[Session]:
        """
        Return sessions for (city, settlement_date), capped at sessions_per_day.

        Cap = 0 means unlimited (use all available candles).
        When the real candle count exceeds the cap, sessions are subsampled
        by uniform spacing so we spread evaluation across the trading day.
        """
        sessions = self.price_store.sessions_for(city, settlement)
        cap = cfg.sessions_per_day
        if cap <= 0 or len(sessions) <= cap:
            return sessions
        if cap == 1:
            return [sessions[0]]
        # Uniform subsample: pick `cap` evenly-spaced indices
        indices = [int(round(i * (len(sessions) - 1) / (cap - 1))) for i in range(cap)]
        return [sessions[i] for i in sorted(set(indices))]

    # ── model management ──────────────────────────────────────────────────────

    def _get_model(
        self,
        city: str,
        city_df: pd.DataFrame,
        row_i: int,
        current_date: pd.Timestamp,
        model_state: dict,
        cfg: BacktestConfig,
    ) -> Optional[object]:
        """Return a fitted model, refitting only when refit_every days have elapsed."""
        if city in model_state:
            cached_model, last_refit_date = model_state[city]
            if (current_date - last_refit_date).days < cfg.effective_refit_every:
                return cached_model

        lb = cfg.lookback if cfg.lookback > 0 else row_i
        train_df = city_df.iloc[max(0, row_i - lb):row_i]

        if self._model_factory is not None:
            model = self._model_factory()
        else:
            model = _instantiate_model(cfg.model_name)
        try:
            model.fit(train_df)
        except (ValueError, RuntimeError) as exc:
            if cfg.verbose:
                print(f"    WARN: model fit failed ({city}, row {row_i}): {exc}")
            return None

        model_state[city] = (model, current_date)
        return model


def _session_to_markets(
    session: Session,
    city: str,
    half_spread: float = 0.0,
    pessimistic: bool = False,
) -> list[dict]:
    """Convert a backtest Session to PortfolioManager-compatible market dicts.

    half_spread adds a simulated bid-ask spread so yes_ask + no_ask > 1.0,
    matching real Kalshi execution costs.  With half_spread=0.02 the total
    round-trip spread is ~4 cents (actual observed median in contested markets
    is 6-12 cents round-trip; 4 cents is a conservative baseline).

    When pessimistic=True, uses the candle's intra-period high/low instead of
    the close, giving worst-case fills within the hourly window:
      YES buys → yes_ask = high_dollars  (paid highest YES during candle)
      NO  buys → no_ask  = 1 - low_dollars (lowest NO price = highest YES)
    half_spread is NOT added in pessimistic mode — high/low already represent
    the worst observed fill, adding spread on top would double-count.
    Falls back to close + half_spread when high/low are unavailable.
    """
    markets = []
    for pc in session.contracts:
        if pessimistic and pc.high_dollars is not None and pc.low_dollars is not None:
            # Use actual intra-candle extremes as the fill price — no extra spread
            yes_ask = pc.high_dollars
            no_ask  = 1.0 - pc.low_dollars
            spread_to_add = 0.0
        else:
            yes_ask = pc.close_dollars
            no_ask  = 1.0 - pc.close_dollars
            spread_to_add = half_spread
        markets.append({
            "ticker": pc.ticker,
            "title": pc.title,
            "_city": city,
            "yes_ask_dollars": min(0.99, round(yes_ask + spread_to_add, 6)),
            "no_ask_dollars":  min(0.99, round(no_ask  + spread_to_add, 6)),
            "volume": pc.volume,
        })
    return markets


def _backtest_positions(portfolio: Portfolio, city: str) -> dict[str, float]:
    """Net YES contracts for all open positions in this city.
    Positive = long yes, negative = long no (matches PortfolioManager convention)."""
    return {
        ticker: (pos.size if pos.side == "yes" else -pos.size)
        for ticker, pos in portfolio.open_positions_for_city(city).items()
    }


def _threshold_from_def(contract_def: dict) -> int:
    """Extract the display threshold integer from a parsed contract definition."""
    if "threshold" in contract_def:
        return int(contract_def["threshold"])
    return int(contract_def.get("low", 0))


def _build_trade_from_intent(
    run_id: int,
    city: str,
    session: int,
    session_ts: int,
    trade_date_str: str,
    test_row: pd.Series,
    intent: "OrderIntent",
    portfolio: Portfolio,
    fee_rate: float = 0.02,
    available_cash: float = 0.0,
) -> Optional[Trade]:
    """
    Settle a PortfolioManager buy intent against the observed y_tmax.

    Used by the allow_exits=False path: PM sizes and selects the trade;
    we immediately compute final PnL from the day's outcome.
    """
    y_tmax = float(test_row["y_tmax"])
    cd = intent.contract_def

    if cd["market_type"] == "geq":
        outcome_yes = int(y_tmax >= cd["threshold"])
    elif cd["market_type"] == "leq":
        outcome_yes = int(y_tmax <= cd["threshold"])
    elif cd["market_type"] == "range":
        outcome_yes = int(cd["low"] <= y_tmax <= cd["high"])
    else:
        return None

    side = intent.side
    mkt_p = intent.price
    outcome = outcome_yes if side == "yes" else (1 - outcome_yes)

    # intent.size is the dollar cost; contracts = size / price (ask paid)
    contracts = intent.size / mkt_p
    if side == "yes":
        gross_pnl = contracts * (outcome_yes - mkt_p)
    else:
        gross_pnl = contracts * ((1 - outcome_yes) - mkt_p)
    pnl = gross_pnl - fee_rate * max(gross_pnl, 0.0)

    return Trade(
        run_id=run_id,
        date=trade_date_str,
        city=city,
        session=session,
        session_ts=session_ts,
        ticker=intent.ticker,
        title=intent.title,
        contract_type=cd["market_type"],
        threshold=_threshold_from_def(cd),
        side=side,
        mkt_p=mkt_p,
        model_p=intent.fair_p,
        edge=intent.edge,
        size=intent.size,
        pnl=pnl,
        outcome=outcome,
        bankroll_after=portfolio.bankroll + pnl,
        available_cash=available_cash,
    )


def _build_exit_trade(
    run_id: int,
    city: str,
    s_idx: int,
    session_ts: int,
    pos: "OpenPosition",
    exit_price: float,
    fair_p: float,
    portfolio: Portfolio,
    fee_rate: float,
) -> Trade:
    """
    Build a Trade representing an early exit from a position.

    PnL is realised mark-to-market: (exit_price - entry_price) per contract.
    exit_price is the bid received when closing (yes_bid for YES, no_bid for NO).
    entry_mkt_p is the ask paid when opening (yes_ask for YES, no_ask for NO).
    Same formula for both sides: profit = (sell_price - buy_price) × contracts.
    """
    contracts = pos.size / pos.entry_mkt_p
    gross_pnl = contracts * (exit_price - pos.entry_mkt_p)

    pnl = gross_pnl - fee_rate * max(gross_pnl, 0.0)

    return Trade(
        run_id=run_id,
        date=pos.entry_date,
        city=city,
        session=s_idx,
        session_ts=session_ts,
        ticker=pos.ticker,
        title=pos.title,
        contract_type=pos.contract_type,
        threshold=pos.threshold,
        side=pos.side,
        mkt_p=exit_price,
        model_p=fair_p,
        edge=0.0,
        size=pos.size,
        pnl=pnl,
        outcome=1 if pnl >= 0 else 0,
        bankroll_after=portfolio.bankroll + pnl,
        trade_type="exit",
        entry_mkt_p=pos.entry_mkt_p,
        available_cash=pos.entry_available_cash,
    )


def _build_settlement_trade(
    run_id: int,
    city: str,
    test_row: pd.Series,
    pos: "OpenPosition",
    portfolio: Portfolio,
    fee_rate: float,
) -> Trade:
    """
    Settle an open position using the observed y_tmax (end-of-day outcome).

    This mirrors the PnL logic in _build_trade_from_intent but operates on a
    stored OpenPosition rather than an OrderIntent.

    Raises ValueError for unknown contract_type — callers must not swallow this,
    as the position has already been removed from the portfolio before this is called.
    """
    y_tmax = float(test_row["y_tmax"])
    cd = pos.contract_def

    if pos.contract_type == "geq":
        outcome_yes = int(y_tmax >= cd["threshold"])
    elif pos.contract_type == "leq":
        outcome_yes = int(y_tmax <= cd["threshold"])
    elif pos.contract_type == "range":
        outcome_yes = int(cd["low"] <= y_tmax <= cd["high"])
    else:
        raise ValueError(
            f"_build_settlement_trade: unknown contract_type {pos.contract_type!r} "
            f"for ticker {pos.ticker!r} (city={city}, entry_date={pos.entry_date}). "
            f"Position was already closed; bankroll would be corrupted by silently "
            f"skipping settlement. Fix the contract_type stored on this position."
        )

    outcome = outcome_yes if pos.side == "yes" else (1 - outcome_yes)
    mkt_p = pos.entry_mkt_p

    # pos.size is the dollar cost; contracts = size / price (ask paid at entry)
    contracts = pos.size / mkt_p
    if pos.side == "yes":
        gross_pnl = contracts * (outcome_yes - mkt_p)
    else:
        gross_pnl = contracts * ((1 - outcome_yes) - mkt_p)

    pnl = gross_pnl - fee_rate * max(gross_pnl, 0.0)

    return Trade(
        run_id=run_id,
        date=pos.entry_date,
        city=city,
        session=pos.entry_session,
        session_ts=pos.entry_session_ts,
        ticker=pos.ticker,
        title=pos.title,
        contract_type=pos.contract_type,
        threshold=pos.threshold,
        side=pos.side,
        mkt_p=mkt_p,
        model_p=pos.entry_model_p,
        edge=pos.entry_edge,
        size=pos.size,
        pnl=pnl,
        outcome=outcome,
        bankroll_after=portfolio.bankroll + pnl,
        trade_type="settle",
        entry_mkt_p=mkt_p,
        available_cash=pos.entry_available_cash,
    )


def _instantiate_model(model_name: str) -> object:
    from src import model as model_module
    from backtesting.config import MODEL_REGISTRY
    return getattr(model_module, MODEL_REGISTRY[model_name])()


def _print_run_summary(result: "RunResult") -> None:  # noqa: F821
    df = result.trades
    if df.empty:
        print("  No trades — check that price history covers the backtest date range.")
        return

    total_pnl     = df["pnl"].sum()
    total_wagered = df["size"].sum()
    roi           = total_pnl / total_wagered if total_wagered > 0 else 0.0
    win_rate      = (df["pnl"] > 0).mean()
    final_br      = df["bankroll_after"].iloc[-1]
    start_br      = result.config.initial_bankroll

    daily = df.groupby("date")["pnl"].sum()
    d_mean = daily.mean()
    d_std  = daily.std()
    d_min  = daily.min()
    d_max  = daily.max()

    print(
        f"  trades={len(df):,}  wagered=${total_wagered:,.2f}  ROI={roi:+.2%}  win={win_rate:.1%}\n"
        f"  PnL=${total_pnl:+,.2f}  bankroll=${start_br:,.2f} → ${final_br:,.2f}\n"
        f"  daily PnL — mean=${d_mean:+.2f}  std=${d_std:.2f}  min=${d_min:+.2f}  max=${d_max:+.2f}"
    )
