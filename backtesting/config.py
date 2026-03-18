"""
BacktestConfig: all parameters that control a backtest run.

Every CLI flag maps 1-to-1 to a field here so that configs are
fully serialisable, reproducible, and easy to diff between runs.

Defaults are loaded from config.yaml (project root) via src.app_config.cfg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.app_config import cfg as _cfg

_bc = _cfg.backtest

# Registry: CLI name -> src.model class name
MODEL_REGISTRY: dict[str, str] = {
    "BayesianRidge":    "BayesianTempModel",
    "ARD":              "ARDTempModel",
    "InteractionBayes": "InteractionBayesModel",
    "KernelRidge":      "KernelRidgeModel",
    "NGBoost":          "NGBoostTempModel",
    "RandomForest":     "RandomForestModel",
    "ExtraTrees":       "ExtraTreesModel",
    "QuantileGB":       "QuantileGBModel",
    "BaggingRidge":     "BaggingRidgeModel",
    "GBResidual":       "GBResidualModel",
}

# Minimum refit cadence per model (days) — loaded from config.yaml.
# Linear/Bayesian models are fast enough for daily refits; tree/ensemble models
# don't meaningfully improve with sub-weekly refitting on this data size and
# have expensive fit costs. The effective cadence = max(user refit_every, this floor).
MODEL_REFIT_EVERY: dict[str, int] = {
    k: int(v) for k, v in vars(_bc.model_refit_floor).items()
}

DEFAULT_CITIES: list[str] = list(_bc.cities)
DEFAULT_THRESHOLD_OFFSETS: list[int] = list(_bc.threshold_offsets)


@dataclass
class BacktestConfig:
    # ── Run control ──────────────────────────────────────────────────────────
    n_runs: int = _bc.n_runs
    """Number of independent runs. Run k uses seed+k so outcomes vary by
    Thompson Sampling noise. Use n_runs > 1 to measure variance / build CIs."""

    seed: int = _bc.seed
    """Base random seed. Run k uses seed+k."""

    # ── Date range ───────────────────────────────────────────────────────────
    start_date: str = _bc.start_date
    end_date: str = _bc.end_date

    # ── Trading frequency & limits ───────────────────────────────────────────
    sessions_per_day: int = _bc.sessions_per_day
    """Maximum real hourly price snapshots to evaluate per city per day.
    0 = use all available candles (no cap).
    N > 0 = subsample N evenly-spaced snapshots from the trading window."""

    max_daily_trades: int = _bc.max_daily_trades
    """Hard cap: total trades across ALL cities per calendar day."""

    max_session_trades: int = _bc.max_session_trades
    """Hard cap: trades within a single session (one evaluation pass across
    all cities). Must be <= max_daily_trades."""

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = _bc.model_name
    """Temperature model to use. Must be a key in MODEL_REGISTRY."""

    lookback: int = _bc.lookback
    """Rows of historical data per city to train on (0 = unlimited)."""

    refit_every: int = _bc.refit_every
    """Refit the model every N calendar days. refit_every=1 = daily refits
    (slowest, freshest). refit_every=7 = weekly refits (7x faster)."""

    # ── Sizing ───────────────────────────────────────────────────────────────
    initial_bankroll: float = _bc.initial_bankroll
    kelly_fraction: float = _bc.kelly_fraction
    max_bet_fraction: float = _bc.max_bet_fraction
    max_bet_dollars: Optional[float] = None
    """Hard dollar cap per trade (applied after Kelly + fraction cap). None = no cap."""
    fee_rate: float = _bc.fee_rate
    """Kalshi fee as a fraction of winnings (applied only on winning trades)."""

    # ── Edge & acquisition ───────────────────────────────────────────────────
    min_edge: float = _bc.min_edge
    """Minimum expected edge before placing a trade."""

    min_confidence: float = _bc.min_confidence
    """Only enter when fair_p >= 0.5 + min_confidence (YES) or <= 0.5 - min_confidence (NO).
    0.0 = no filter. 0.15 = model must be ≥65% confident. Controls win rate directly."""

    n_thompson_draws: int = _bc.n_thompson_draws
    """Posterior samples drawn per prediction for Thompson Sampling."""

    spread_alpha: float = _bc.spread_alpha
    """How much ensemble spread widens the market's sigma:
    market_sigma = climo_sigma * (1 + spread_alpha * ensemble_spread)."""

    # ── Scope ────────────────────────────────────────────────────────────────
    min_train_rows: int = _bc.min_train_rows
    """Minimum rows of training data required before the first trade."""

    cities: Optional[List[str]] = None
    """Cities to trade. None = all cities found in data."""

    threshold_offsets: Optional[List[int]] = None
    """Temperature thresholds as offsets from climo mean. None = default grid."""

    # ── Exit logic ───────────────────────────────────────────────────────────
    allow_exits: bool = _bc.allow_exits
    """When True, positions are exited early if the market price has moved to
    (or beyond) the model's fair value — i.e. no edge remains.
    When False, all positions are held to settlement (original behaviour)."""

    exit_edge_threshold: float = _bc.exit_edge_threshold
    """Exit a held YES position when current_price >= fair_p - exit_edge_threshold.
    0.0 = exit exactly at fair value; positive values exit slightly before fair
    value is reached (e.g. 0.02 exits when 2 cents of edge remain)."""

    # ── Probability band filter ───────────────────────────────────────────────
    min_fair_p: float = _bc.min_fair_p
    """Skip entries where fair_p < min_fair_p (near-certain NO outcome)."""

    max_fair_p: float = _bc.max_fair_p
    """Skip entries where fair_p > max_fair_p (near-certain YES outcome)."""

    # ── Execution realism ─────────────────────────────────────────────────────
    half_spread: float = _bc.half_spread
    """Half the bid-ask spread in dollars, added to both yes_ask and no_ask.
    With half_spread=0.02: yes_ask + no_ask = 1.04 (4-cent total spread).
    Observed data: median round-trip ~3¢ overall, but 6-12¢ in the contested
    35-65¢ range where most trades occur. 4¢ is a conservative baseline.
    Set to 0.0 to disable spread simulation."""

    pessimistic_pricing: bool = _bc.pessimistic_pricing
    """When True, assume worst-case fills across all candles for the settlement day.
    For YES buys: use max(close_dollars) across all sessions (highest ask all day).
    For NO  buys: use 1 - min(close_dollars) across all sessions (lowest NO price).
    This gives a lower-bound on strategy performance — if it's profitable here,
    it's robust to bad timing within the day."""

    min_contract_volume: float = _bc.min_contract_volume
    """Minimum contracts traded in the candle to be eligible for entry.
    0.0 = no filter (beyond the global MIN_VOLUME=10 at load time).
    E.g. 50.0 = only trade contracts that cleared at least 50 contracts
    in that hourly snapshot — avoids stale/thin prices."""

    min_mkt_price: float = _bc.min_mkt_price
    """Minimum market ask price for the side we'd buy (YES ask for YES trades,
    NO ask for NO trades).  Contracts priced below this are skipped — the market
    is too skeptical to trust our model.  E.g. 0.10 blocks all 1-9¢ contracts."""

    # ── Capital management ────────────────────────────────────────────────────
    cash_reserve_fraction: float = _bc.cash_reserve_fraction
    """Fraction of bankroll to keep free at all times. 0.0 = no reserve."""

    rotation_min_edge_gain: float = _bc.rotation_min_edge_gain
    """Exit the lowest-edge held position when a new entry has at least this
much more edge, freeing capital. 0.0 = no rotation."""

    # ── Multi-day trading ────────────────────────────────────────────────────
    trade_tomorrow: bool = _bc.trade_tomorrow
    """When True, each trade date also evaluates the next-day settlement market
    using tomorrow's feature row — mirroring run_live.py's TOMORROW_DATE logic.
    Set to False to trade only same-day settling contracts."""

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = _cfg.paths.backtest_output
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.cities is None:
            self.cities = list(DEFAULT_CITIES)
        if self.threshold_offsets is None:
            self.threshold_offsets = list(DEFAULT_THRESHOLD_OFFSETS)
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Valid choices: {sorted(MODEL_REGISTRY.keys())}"
            )
        if self.sessions_per_day < 0:
            raise ValueError("sessions_per_day must be >= 0 (0 = use all available sessions)")
        if self.max_session_trades > self.max_daily_trades:
            raise ValueError(
                f"max_session_trades ({self.max_session_trades}) cannot exceed "
                f"max_daily_trades ({self.max_daily_trades})"
            )
        if self.n_runs < 1:
            raise ValueError("n_runs must be >= 1")

    @property
    def model_class_name(self) -> str:
        return MODEL_REGISTRY[self.model_name]

    @property
    def effective_refit_every(self) -> int:
        """Refit cadence with per-model floor applied.

        Slow tree/ensemble models are floored at MODEL_REFIT_EVERY[model] so
        they don't refit every day by default. The user's --refit-every only
        takes effect if it's *higher* than the floor.
        """
        floor = MODEL_REFIT_EVERY.get(self.model_name, 1)
        return max(self.refit_every, floor)
