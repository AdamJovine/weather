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
from backtesting.portfolio import Portfolio, Trade

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

        for date, city_records in day_groups:
            portfolio.begin_day(date.isoformat()[:10])

            # Shuffle city order each day so no city systematically trades first
            shuffled = list(city_records)
            np.random.shuffle(shuffled)

            for city, row_i, city_df in shuffled:
                if portfolio.remaining_today() == 0:
                    break

                settlement = date.date()  # pd.Timestamp → datetime.date
                sessions = self._pick_sessions(city, settlement, cfg)
                if not sessions:
                    continue

                model = self._get_model(city, city_df, row_i, date, model_state, cfg)
                if model is None:
                    continue

                # ── compute probability distribution ONCE per (city, date) ──
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

                    if not portfolio.can_trade():
                        break

                    candidates = _select_candidates(prob_row, session, cfg)

                    for candidate in candidates:
                        if not portfolio.can_trade():
                            break
                        trade = _build_trade(
                            run_id=run_id,
                            city=city,
                            session=s_idx,
                            session_ts=session.ts,
                            test_row=test_row,
                            candidate=candidate,
                            portfolio=portfolio,
                            fee_rate=cfg.fee_rate,
                        )
                        if trade is not None:
                            portfolio.record_trade(trade)

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
    ) -> list[tuple[pd.Timestamp, list[tuple[str, int, pd.DataFrame]]]]:
        """
        Build an ordered list of (date, [(city, row_idx, city_df), ...]).

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
                # Only include dates where real price data exists
                if not self.price_store.has_data(city, date.date()):
                    continue

                day_map.setdefault(date, []).append((city, i, city_df))

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
            if (current_date - last_refit_date).days < cfg.refit_every:
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


# ── contract selection (module-level for testability) ─────────────────────────

def _select_candidates(
    prob_row: pd.Series,
    session: Session,
    cfg: BacktestConfig,
) -> list[dict]:
    """
    Evaluate every contract in this session against the model's probability
    distribution. Return candidates with edge >= min_edge, sorted by edge.

    For each ParsedContract:
      fair_p = compute_fair_prob(prob_row, contract_def)   ← full discrete distribution
      yes_edge = fair_p - close_dollars
      no_edge  = close_dollars - fair_p   (= 1 - fair_p) - (1 - close_dollars)

    We take the better side. No edge → skipped.
    """
    from src.pricing import compute_fair_prob

    candidates = []
    for pc in session.contracts:
        try:
            fair_p = float(compute_fair_prob(prob_row, pc.contract_def))
        except Exception:
            continue

        mkt_p = pc.close_dollars
        yes_edge = fair_p - mkt_p
        no_edge = mkt_p - fair_p   # equivalent to (1-fair_p) - (1-mkt_p)

        if yes_edge >= no_edge and yes_edge >= cfg.min_edge:
            side, edge = "yes", yes_edge
        elif no_edge > yes_edge and no_edge >= cfg.min_edge:
            side, edge = "no", no_edge
        else:
            continue

        candidates.append({
            "ticker":        pc.ticker,
            "title":         pc.title,
            "contract_def":  pc.contract_def,
            "contract_type": pc.contract_type,
            "threshold":     pc.display_threshold,
            "side":          side,
            "mkt_p":         mkt_p,
            "model_p":       fair_p,
            "edge":          edge,
        })

    candidates.sort(key=lambda x: -x["edge"])
    return candidates


def _build_trade(
    run_id: int,
    city: str,
    session: int,
    session_ts: int,
    test_row: pd.Series,
    candidate: dict,
    portfolio: Portfolio,
    fee_rate: float = 0.02,
) -> Optional[Trade]:
    """Convert a candidate contract into a Trade with Kelly sizing and realised PnL."""
    size = portfolio.kelly_size(candidate["edge"])
    if size <= 0:
        return None

    contract_def = candidate["contract_def"]
    contract_type = candidate["contract_type"]
    side = candidate["side"]
    mkt_p = candidate["mkt_p"]
    y_tmax = float(test_row["y_tmax"])

    # Settle the contract (YES-side outcome)
    if contract_type == "geq":
        outcome_yes = int(y_tmax >= contract_def["threshold"])
    elif contract_type == "leq":
        outcome_yes = int(y_tmax <= contract_def["threshold"])
    elif contract_type == "range":
        outcome_yes = int(contract_def["low"] <= y_tmax <= contract_def["high"])
    else:
        return None

    outcome = outcome_yes if side == "yes" else (1 - outcome_yes)

    if side == "yes":
        gross_pnl = size * (outcome_yes - mkt_p)
    else:
        gross_pnl = size * ((1 - outcome_yes) - (1 - mkt_p))
    pnl = gross_pnl - fee_rate * max(gross_pnl, 0.0)

    return Trade(
        run_id=run_id,
        date=str(test_row["date"])[:10],
        city=city,
        session=session,
        session_ts=session_ts,
        ticker=candidate["ticker"],
        title=candidate["title"],
        contract_type=contract_type,
        threshold=candidate["threshold"],
        side=side,
        mkt_p=mkt_p,
        model_p=candidate["model_p"],
        edge=candidate["edge"],
        size=size,
        pnl=pnl,
        outcome=outcome,
        bankroll_after=portfolio.bankroll + pnl,
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
    total_pnl = df["pnl"].sum()
    total_wagered = df["size"].sum()
    roi = total_pnl / total_wagered if total_wagered > 0 else 0.0
    win_rate = (df["pnl"] > 0).mean()
    final_br = df["bankroll_after"].iloc[-1]
    print(
        f"  trades={len(df):,}  ROI={roi:+.2%}  "
        f"win={win_rate:.1%}  PnL=${total_pnl:,.2f}  "
        f"bankroll=${final_br:,.2f}"
    )
