"""
Temperature distribution models.

All models share the _BaseTempModel interface:
    model.fit(train_df)
    mu, epistemic_std, aleatoric = model.predict_with_uncertainty(predict_df)
    probs_df = model.predict_integer_probs(predict_df)

predict_with_uncertainty returns a 3-tuple:
  mu            – point prediction (°F)
  epistemic_std – model's uncertainty in its own mean (used for Thompson Sampling draws)
  aleatoric     – residual noise per row, scaled by ensemble_spread

Total predictive sigma = sqrt(epistemic² + aleatoric²)

Available models (all drop-in replaceable):
  BayesianTempModel      – Bayesian Ridge, analytic weight posterior
  ARDTempModel           – ARD regression, per-feature alpha (auto-prunes weak features)
  InteractionBayesModel  – BayesianRidge on degree-2 interaction features
  KernelRidgeModel       – RBF Kernel Ridge + bootstrap epistemic (nonlinear, dual form)
  NGBoostTempModel       – NGBoost Normal dist; per-row heteroscedastic sigma
  RandomForestModel      – RF, std of tree predictions as epistemic
  ExtraTreesModel        – Extra-Trees, std of tree predictions
  QuantileGBModel        – HistGradientBoosting at q16/q84, range as epistemic
  BaggingRidgeModel      – Bagging(Ridge), std of bootstrap predictions
  GBResidualModel        – GradientBoosting mean + separate GBR for |residual|
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm

from src.app_config import cfg as _cfg
from src.config import RIDGE_ALPHA, TEMP_GRID_MIN, TEMP_GRID_MAX

FEATURES = [
    "forecast_high",
    "climo_mean_doy",
    "forecast_minus_climo",   # NWS forecast minus historical climo — main directional signal
    "lag1_tmax",
    "lag3_mean_tmax",
    "lag7_mean_tmax",         # 7-day rolling mean — week-scale regime persistence
    "lag14_mean_tmax",        # 14-day rolling mean — synoptic-scale regime
    "lag21_mean_tmax",        # 21-day rolling mean — blocking pattern persistence
    "lag30_mean_tmax",        # 30-day rolling mean — monthly regime persistence
    "lag_trend",              # lag1 - lag7_mean: recent warming (+) or cooling (-) trend
    "forecast_minus_lag1",    # NWS forecast minus yesterday's obs — warm/cold-front signal
    "nbm_high",               # NBM gridded max temp (NWS /gridpoints); imputed from forecast_high when missing
    "temp_850hpa",            # GFS 850hPa temperature daily max (°F) — upper-air thermal regime
    "shortwave_radiation",    # GFS shortwave radiation sum (MJ/m²/day) — inversely correlated with cloud cover → tmax
    "dew_point_max",          # GFS 2m dew point daily max (°F) — moisture cap on afternoon heating
    "nws_bias_14d",           # trailing 14-day mean signed NWS error (°F); positive = NWS running warm
]

SPREAD_ALPHA = _cfg.model.spread_alpha  # how much ensemble_spread widens the aleatoric sigma


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseTempModel:
    """
    Shared interface for all temperature uncertainty models.

    Subclasses must implement fit() and _epistemic_std(X, df) only.
    predict_with_uncertainty, predict_integer_probs, and _aleatoric are shared.
    """

    sigma_: float = None   # residual std on training data
    is_fit: bool = False
    _sigma_floor: float = 4.0  # minimum predictive sigma — prevents overconfidence on tail events

    def fit(self, df: pd.DataFrame) -> "_BaseTempModel":
        raise NotImplementedError

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _epistemic_std(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Return per-row epistemic uncertainty in the mean prediction."""
        raise NotImplementedError

    def _aleatoric(self, df: pd.DataFrame) -> np.ndarray:
        """Aleatoric noise sigma, uniformly scaled by ensemble_spread."""
        base = max(self.sigma_, 1.0)
        spread = (
            df["ensemble_spread"].fillna(0).values
            if "ensemble_spread" in df.columns
            else np.zeros(len(df))
        )
        return np.maximum(base * (1.0 + SPREAD_ALPHA * spread), 1.0)

    def predict_with_uncertainty(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mu, epistemic_std, aleatoric).
        Total predictive sigma = sqrt(epistemic² + aleatoric²).
        """
        if not self.is_fit:
            raise RuntimeError("Model not fit. Call fit() first.")
        X = np.ascontiguousarray(df[FEATURES].values, dtype=np.float64)
        mu          = self._predict_mu(X)
        epistemic   = self._epistemic_std(X, df)
        aleatoric   = self._aleatoric(df)
        return mu, epistemic, aleatoric

    def predict_integer_probs(
        self,
        df: pd.DataFrame,
        temp_grid: range = None,
    ) -> pd.DataFrame:
        if temp_grid is None:
            temp_grid = range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)

        mu, _epistemic, aleatoric = self.predict_with_uncertainty(df)
        # Use only the aleatoric sigma (empirically calibrated from training
        # residuals + ensemble spread) to set the distribution width.
        # Epistemic (weight posterior uncertainty) inflates badly when
        # prediction features are out-of-distribution, producing a flat
        # useless distribution.  Epistemic is reserved for Thompson Sampling.
        total_sigma = aleatoric

        rows = []
        for i, (m, s) in enumerate(zip(mu, total_sigma)):
            s = max(s, self._sigma_floor)
            probs = [
                norm.cdf(k + 0.5, loc=m, scale=s) - norm.cdf(k - 0.5, loc=m, scale=s)
                for k in temp_grid
            ]
            t = sum(probs)
            row = {"row_id": i}
            row.update({f"temp_{k}": p / t for k, p in zip(temp_grid, probs)})
            rows.append(row)

        return pd.DataFrame(rows)

    def _fit_core(self, df: pd.DataFrame):
        """
        Shared boilerplate: drop NaNs, apply optional lookback and exponential
        decay weighting, return (X, y, sample_weight).

        Set instance attributes before calling fit() to activate:
          model._lookback       = int, rows of history to keep (0 = all)
          model._decay_halflife = float, half-life in days for exp decay (0 = uniform)
        """
        train = df.dropna(subset=FEATURES + ["y_tmax"]).copy()
        if train.empty:
            raise ValueError("No complete rows to train on after dropping NaNs.")

        lookback = int(getattr(self, "_lookback", 0) or 0)
        if lookback > 0:
            train = train.iloc[-lookback:]

        X  = np.ascontiguousarray(train[FEATURES].values, dtype=np.float64)
        y  = train["y_tmax"].values

        halflife = float(getattr(self, "_decay_halflife", 0) or 0)
        if halflife > 0:
            ages = np.arange(len(y) - 1, -1, -1, dtype=np.float64)
            sw   = np.exp(-np.log(2) / halflife * ages)
        else:
            sw = None

        return X, y, sw

    def _fit_and_oos_sigma(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sw,
        fit_fn,
        val_frac: float = 0.2,
        min_val: int = 30,
        min_train: int = 50,
    ) -> float:
        """
        Estimate out-of-sample sigma via a single time-series train/val split:
          1. Fit on the first (1 - val_frac) rows.
          2. Measure residuals on the held-out tail.
          3. Refit on all rows so the deployed model uses full history.
        Falls back to in-sample sigma when the dataset is too small to split.
        fit_fn signature: (X, y, sw) -> None  (fits model in place)
        """
        n = len(y)
        n_val = max(min_val, int(n * val_frac))
        n_train = n - n_val
        if n_train < min_train:
            fit_fn(X, y, sw)
            return max(float((y - self._predict_mu(X)).std(ddof=1)), 1.0)
        sw_tr = sw[:n_train] if sw is not None else None
        fit_fn(X[:n_train], y[:n_train], sw_tr)
        oos_resid = y[n_train:] - self._predict_mu(X[n_train:])
        oos_sigma = float(oos_resid.std(ddof=1))
        fit_fn(X, y, sw)
        return max(oos_sigma, 1.0)


# ---------------------------------------------------------------------------
# 1. Bayesian Ridge  (analytic posterior → epistemic from weight uncertainty)
# ---------------------------------------------------------------------------

class BayesianTempModel(_BaseTempModel):
    """BayesianRidge — analytic posterior over weights."""

    def __init__(self):
        self._model = BayesianRidge()

    def fit(self, df: pd.DataFrame) -> "BayesianTempModel":
        X, y, _ = self._fit_core(df)
        self.sigma_ = self._fit_and_oos_sigma(
            X, y, None, lambda Xt, yt, _w: self._model.fit(Xt, yt),
        )
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        mu, _ = self._model.predict(X, return_std=True)
        return mu

    def _epistemic_std(self, X, df):
        _, epistemic = self._model.predict(X, return_std=True)
        return epistemic


# ---------------------------------------------------------------------------
# 2. ARD Regression  (per-feature alpha — automatic feature pruning)
# ---------------------------------------------------------------------------

class ARDTempModel(_BaseTempModel):
    """
    Automatic Relevance Determination regression.

    Identical to BayesianRidge except each feature gets its own alpha prior,
    so irrelevant features are automatically shrunk toward zero.  With 17
    features of varying importance (forecast_high matters much more than oni),
    ARD often improves calibration over a single shared alpha.

    epistemic_std: analytic posterior std from the weight covariance.
    """

    def __init__(self):
        self._model = ARDRegression()

    def fit(self, df: pd.DataFrame) -> "ARDTempModel":
        X, y, _ = self._fit_core(df)
        self.sigma_ = self._fit_and_oos_sigma(
            X, y, None, lambda Xt, yt, _w: self._model.fit(Xt, yt),
        )
        self.is_fit = True
        return self

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        mu, _ = self._model.predict(X, return_std=True)
        return mu

    def _epistemic_std(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        _, epistemic = self._model.predict(X, return_std=True)
        return epistemic


# ---------------------------------------------------------------------------
# 3. BayesianRidge + degree-2 interactions  (nonlinear via feature expansion)
# ---------------------------------------------------------------------------

class InteractionBayesModel(_BaseTempModel):
    """
    BayesianRidge on degree-2 interaction features.

    PolynomialFeatures(interaction_only=True) creates all pairwise products
    (forecast_high × lag1, season × ao_index, etc.) without squaring individual
    features.  With 17 base features this adds ~136 interaction terms.
    BayesianRidge's L2 prior handles the expanded space without overfitting.
    StandardScaler normalises interactions that would otherwise span very
    different magnitudes.

    epistemic_std: analytic posterior from the expanded-space weight covariance.
    """

    def __init__(self):
        self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self._scaler = StandardScaler()
        self._model = BayesianRidge()

    def fit(self, df: pd.DataFrame) -> "InteractionBayesModel":
        X, y, _ = self._fit_core(df)
        def _fn(Xt, yt, _w):
            Xp = self._poly.fit_transform(Xt)
            Xs = self._scaler.fit_transform(Xp)
            self._model.fit(Xs, yt)
        self.sigma_ = self._fit_and_oos_sigma(X, y, None, _fn)
        self.is_fit = True
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(self._poly.transform(X))

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        mu, _ = self._model.predict(self._transform(X), return_std=True)
        return mu

    def _epistemic_std(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        _, epistemic = self._model.predict(self._transform(X), return_std=True)
        return epistemic


# ---------------------------------------------------------------------------
# 4. Kernel Ridge + bootstrap epistemic  (RBF nonlinearity, dual form)
# ---------------------------------------------------------------------------

class KernelRidgeModel(_BaseTempModel):
    """
    RBF Kernel Ridge Regression with bootstrap epistemic uncertainty.

    KernelRidge in dual form solves the same problem as Ridge in the infinite-
    dimensional RBF feature space — capturing nonlinear interactions without
    explicit feature expansion.  O(n²) train, O(n) predict per row.

    epistemic_std: std of predictions across B bootstrap KernelRidge models.
    This gives a frequentist uncertainty estimate without Bayesian priors.
    """

    N_BOOT = 15   # bootstrap replicates — kept small so fit stays fast

    def __init__(self, alpha: float = 0.5, gamma: float = None):
        self._alpha = alpha
        self._gamma = gamma   # None → sklearn default (1/n_features)
        self._scaler = StandardScaler()
        self._models: list = []

    def fit(self, df: pd.DataFrame) -> "KernelRidgeModel":
        X, y, _ = self._fit_core(df)
        def _fn(Xt, yt, _w):
            Xs = self._scaler.fit_transform(Xt)
            n = len(yt)
            rng = np.random.default_rng(42)
            self._models = []
            for _ in range(self.N_BOOT):
                idx = rng.integers(0, n, size=n)
                m = KernelRidge(kernel="rbf", alpha=self._alpha, gamma=self._gamma)
                m.fit(Xs[idx], yt[idx])
                self._models.append(m)
            m0 = KernelRidge(kernel="rbf", alpha=self._alpha, gamma=self._gamma)
            m0.fit(Xs, yt)
            self._base_model = m0
        self.sigma_ = self._fit_and_oos_sigma(X, y, None, _fn)
        self.is_fit = True
        return self

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        return self._base_model.predict(self._scaler.transform(X))

    def _epistemic_std(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        Xs = self._scaler.transform(X)
        preds = np.array([m.predict(Xs) for m in self._models])
        return preds.std(axis=0)


# ---------------------------------------------------------------------------
# 5. NGBoost  (Natural Gradient Boosting — per-row heteroscedastic sigma)
# ---------------------------------------------------------------------------

class NGBoostTempModel(_BaseTempModel):
    """
    NGBoost fits a Normal(mu, sigma) per row using natural gradient descent.

    Key advantage over BayesianRidge: sigma is heteroscedastic — it varies
    per row based on features (e.g. higher sigma for Denver in spring, low
    sigma for Phoenix in summer).  This directly improves probability pricing
    for high-variance city/season combinations.

    epistemic_std = dist.scale * EPISTEMIC_FRAC  (fraction of total uncertainty)
    aleatoric     = dist.scale * (1 - EPISTEMIC_FRAC), scaled by gefs_spread
    """

    EPISTEMIC_FRAC = 0.25   # fraction of NGBoost total sigma assigned to epistemic

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05):
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._model = None

    def fit(self, df: pd.DataFrame) -> "NGBoostTempModel":
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
        except ImportError as e:
            raise ImportError("ngboost not installed. Run: pip install ngboost") from e

        X, y, sw = self._fit_core(df)
        self._model = NGBRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            natural_gradient=True,
            verbose=False,
            random_state=42,
        )
        self._model.fit(X, y, sample_weight=sw)
        dist = self._model.pred_dist(X)
        self.sigma_ = float(dist.scale.mean())
        self.is_fit = True
        return self

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _epistemic_std(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        dist = self._model.pred_dist(X)
        return dist.scale * self.EPISTEMIC_FRAC

    def _aleatoric(self, df: pd.DataFrame) -> np.ndarray:
        """Override: use NGBoost's per-row predicted sigma instead of a fixed scalar."""
        X = df[FEATURES].values
        dist = self._model.pred_dist(X)
        aleatoric_base = dist.scale * (1.0 - self.EPISTEMIC_FRAC)
        spread = (
            df["gefs_spread"].fillna(0).values
            if "gefs_spread" in df.columns
            else df["ensemble_spread"].fillna(0).values
            if "ensemble_spread" in df.columns
            else np.zeros(len(df))
        )
        return np.maximum(aleatoric_base * (1.0 + SPREAD_ALPHA * spread), 1.0)


# ---------------------------------------------------------------------------
# 3. Random Forest  (tree variance → epistemic from disagreement of trees)
# ---------------------------------------------------------------------------

class RandomForestModel(_BaseTempModel):
    """Random Forest — epistemic = std of individual tree predictions."""

    def __init__(self, n_estimators: int = 40, max_depth: int = 8):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> "RandomForestModel":
        X, y, sw = self._fit_core(df)
        self.sigma_ = self._fit_and_oos_sigma(
            X, y, sw, lambda Xt, yt, wt: self._model.fit(Xt, yt, sample_weight=wt),
        )
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        return self._model.predict(X)

    def _epistemic_std(self, X, df):
        tree_preds = np.array([t.predict(X) for t in self._model.estimators_])
        return tree_preds.std(axis=0)


# ---------------------------------------------------------------------------
# 3. Extra-Trees  (higher tree diversity → different epistemic profile)
# ---------------------------------------------------------------------------

class ExtraTreesModel(_BaseTempModel):
    """Extra-Trees — epistemic = std of tree predictions (more variance than RF)."""

    def __init__(self, n_estimators: int = 40, max_depth: int = 8):
        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> "ExtraTreesModel":
        X, y, sw = self._fit_core(df)
        self.sigma_ = self._fit_and_oos_sigma(
            X, y, sw, lambda Xt, yt, wt: self._model.fit(Xt, yt, sample_weight=wt),
        )
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        return self._model.predict(X)

    def _epistemic_std(self, X, df):
        tree_preds = np.array([t.predict(X) for t in self._model.estimators_])
        return tree_preds.std(axis=0)


# ---------------------------------------------------------------------------
# 4. Quantile Gradient Boosting  (HistGB at q16/q84 → interval as epistemic)
# ---------------------------------------------------------------------------

class QuantileGBModel(_BaseTempModel):
    """
    HistGradientBoosting at q16, q50, q84.
    epistemic_std = (q84 - q16) / 4  — half the empirical 1σ interval.
    aleatoric is computed from training residuals as usual.
    """

    def __init__(self, max_iter: int = 30, max_depth: int = 3):
        self._m_lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.16, max_iter=max_iter, max_depth=max_depth)
        self._m_mu = HistGradientBoostingRegressor(loss="squared_error", max_iter=max_iter, max_depth=max_depth)
        self._m_hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.84, max_iter=max_iter, max_depth=max_depth)

    def fit(self, df: pd.DataFrame) -> "QuantileGBModel":
        X, y, sw = self._fit_core(df)
        def _fn(Xt, yt, wt):
            self._m_mu.fit(Xt, yt, sample_weight=wt)
            self._m_lo.fit(Xt, yt, sample_weight=wt)
            self._m_hi.fit(Xt, yt, sample_weight=wt)
        self.sigma_ = self._fit_and_oos_sigma(X, y, sw, _fn)
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        return self._m_mu.predict(X)

    def _epistemic_std(self, X, df):
        interval = self._m_hi.predict(X) - self._m_lo.predict(X)
        # half the 68% interval = ~1σ; divide by 2 to get epistemic component
        return np.maximum(interval / 4.0, 0.1)


# ---------------------------------------------------------------------------
# 5. Bagging Ridge  (bootstrap ensemble of Ridge → std of predictions)
# ---------------------------------------------------------------------------

class BaggingRidgeModel(_BaseTempModel):
    """Bagging over Ridge regressors — epistemic = std across bootstrap predictions."""

    def __init__(self, n_estimators: int = 30):
        self._model = BaggingRegressor(
            estimator=Ridge(alpha=RIDGE_ALPHA),
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame) -> "BaggingRidgeModel":
        X, y, sw = self._fit_core(df)
        self.sigma_ = self._fit_and_oos_sigma(
            X, y, sw, lambda Xt, yt, wt: self._model.fit(Xt, yt, sample_weight=wt),
        )
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        return self._model.predict(X)

    def _epistemic_std(self, X, df):
        est_preds = np.array([e.predict(X) for e in self._model.estimators_])
        return est_preds.std(axis=0)


# ---------------------------------------------------------------------------
# 6. Gradient Boosting + Residual Model  (heteroscedastic variance prediction)
# ---------------------------------------------------------------------------

class GBResidualModel(_BaseTempModel):
    """
    Two-stage Histogram Gradient Boosting:
      Stage 1: predict temperature mean (mu).
      Stage 2: predict |residual| from stage 1 — gives per-row noise scale.

    Uses HistGradientBoostingRegressor (histogram-based, 10-50× faster than
    the sequential GradientBoostingRegressor on typical dataset sizes here).

    epistemic_std = 0.5 × predicted_abs_residual  (uncertainty in the mean estimate)
    aleatoric     = predicted_abs_residual × spread_scale
    """

    def __init__(self, max_iter: int = 30, max_depth: int = 3):
        self._model_mu    = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, random_state=42)
        self._model_sigma = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, random_state=42)

    def fit(self, df: pd.DataFrame) -> "GBResidualModel":
        X, y, sw = self._fit_core(df)
        self._model_mu.fit(X, y, sample_weight=sw)
        abs_resid = np.abs(y - self._model_mu.predict(X))
        self._model_sigma.fit(X, abs_resid, sample_weight=sw)
        self.sigma_ = float(abs_resid.mean())   # mean abs residual ≈ σ√(2/π)
        self.is_fit = True
        return self

    def _predict_mu(self, X):
        return self._model_mu.predict(X)

    def _epistemic_std(self, X, df):
        pred_scale = np.maximum(self._model_sigma.predict(X), 0.5)
        return pred_scale * 0.5   # half of predicted noise scale = epistemic component

    def _aleatoric(self, df: pd.DataFrame) -> np.ndarray:
        """Override: use the predicted noise scale instead of a fixed base sigma."""
        X = df[FEATURES].values
        pred_scale = np.maximum(self._model_sigma.predict(X), 0.5)
        spread = (
            df["ensemble_spread"].fillna(0).values
            if "ensemble_spread" in df.columns
            else np.zeros(len(df))
        )
        return np.maximum(pred_scale * (1.0 + SPREAD_ALPHA * spread), 1.0)


# ---------------------------------------------------------------------------
# Legacy: TempDistributionModel (Ridge, no epistemic)
# ---------------------------------------------------------------------------

class TempDistributionModel:
    """
    Ridge regression point forecast + Gaussian residual uncertainty.
    Kept for backward compatibility. Prefer BayesianTempModel for new code.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.model = Ridge(alpha=alpha)
        self.sigma_: float = None
        self.is_fit: bool = False

    def fit(self, df: pd.DataFrame) -> "TempDistributionModel":
        train = df.dropna(subset=FEATURES + ["y_tmax"]).copy()
        if train.empty:
            raise ValueError("No complete rows to train on after dropping NaNs.")
        X = train[FEATURES].values
        y = train["y_tmax"].values
        self.model.fit(X, y)
        self.sigma_ = float((y - self.model.predict(X)).std(ddof=1))
        self.is_fit = True
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fit. Call fit() first.")
        return self.model.predict(df[FEATURES].values)

    def predict_integer_probs(self, df, temp_grid=None):
        if temp_grid is None:
            temp_grid = range(TEMP_GRID_MIN, TEMP_GRID_MAX + 1)
        mu = self.predict_mean(df)
        base_sigma = max(self.sigma_, 1.0)
        spread_vals = df["ensemble_spread"].fillna(0).values if "ensemble_spread" in df.columns else np.zeros(len(mu))
        rows = []
        for i, m in enumerate(mu):
            sigma = max(base_sigma * (1.0 + SPREAD_ALPHA * float(spread_vals[i])), 1.0)
            probs = [norm.cdf(k + 0.5, loc=m, scale=sigma) - norm.cdf(k - 0.5, loc=m, scale=sigma) for k in temp_grid]
            t = sum(probs)
            row = {"row_id": i}
            row.update({f"temp_{k}": p / t for k, p in zip(temp_grid, probs)})
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Isotonic regression calibrator for event probabilities.

    Fit on a held-out validation set:
        calibrator.fit(raw_probs_array, binary_outcomes_array)

    Then apply before pricing:
        calibrated = calibrator.predict(raw_probs_array)
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.is_fit: bool = False

    def fit(self, p_raw: np.ndarray, y_event: np.ndarray) -> "ProbabilityCalibrator":
        self.iso.fit(p_raw, y_event)
        self.is_fit = True
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Calibrator not fit. Call fit() first.")
        return self.iso.predict(p_raw)
