"""
Feature engineering for the temperature prediction model.

Build one modeling table: one row per (city, date) with:
  - y_tmax: realized TMAX (target, NaN for future rows)
  - forecast_high: NWS forecast
  - seasonal/cyclical features
  - lag features
  - climatology features
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "Downcasting object dtype arrays", FutureWarning)


_EPOCH = pd.Timestamp("2018-01-01")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_sin2"] = np.sin(4 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos2"] = np.cos(4 * np.pi * df["day_of_year"] / 365.25)
    df["days_since_2018"] = (df["date"] - _EPOCH).dt.days.clip(lower=0)
    return df


_MAX_LAG_GAP_DAYS = 7   # gaps larger than this mean lags are from a stale regime


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Days since the previous row (within city) — used to detect stale lags
    df["_date_gap"] = df.groupby("city")["date"].diff().dt.days

    shifted = df.groupby("city")["y_tmax"].shift(1)
    df["lag1_tmax"] = shifted
    df["lag3_mean_tmax"] = (
        shifted.rolling(3).mean().reset_index(level=0, drop=True)
    )
    df["lag7_mean_tmax"] = (
        shifted.rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["lag14_mean_tmax"] = (
        shifted.rolling(14).mean().reset_index(level=0, drop=True)
    )
    df["lag21_mean_tmax"] = (
        shifted.rolling(21).mean().reset_index(level=0, drop=True)
    )
    df["lag30_mean_tmax"] = (
        shifted.rolling(30).mean().reset_index(level=0, drop=True)
    )
    # Rising/falling trend: yesterday vs the recent 7-day mean.
    # Positive = warming into the bet date; negative = cooling.
    df["lag_trend"] = df["lag1_tmax"] - df["lag7_mean_tmax"]

    # Nullify lag features for any row whose predecessor is more than
    # _MAX_LAG_GAP_DAYS away.  Without this, live prediction rows (where
    # recent NOAA data hasn't been ingested yet) inherit stale lags from
    # the last available historical row — potentially months old and from
    # a completely different seasonal regime.  The build_feature_table()
    # caller then imputes NaN lags with climo_mean_doy, which is correct.
    stale = df["_date_gap"].isna() | (df["_date_gap"] > _MAX_LAG_GAP_DAYS)
    lag_cols = ["lag1_tmax", "lag3_mean_tmax", "lag7_mean_tmax",
                "lag14_mean_tmax", "lag21_mean_tmax", "lag30_mean_tmax",
                "lag_trend"]
    for col in lag_cols:
        df.loc[stale, col] = float("nan")

    df = df.drop(columns=["_date_gap"])
    return df


def add_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-city, per-day-of-year mean TMAX using only PRIOR years' data.

    For a row in year Y, climo_mean_doy is the mean of y_tmax for that
    (city, day_of_year) across all years < Y.  This eliminates look-ahead
    bias: the model never sees future years' temperatures baked into its
    seasonal baseline.

    Cold-start: the very first year in the dataset has no prior years, so it
    falls back to the full-dataset climo (unavoidable; affects only ~365 rows).
    """
    df = df.copy()
    df["_year"] = pd.to_datetime(df["date"]).dt.year

    observed = df.dropna(subset=["y_tmax"])[["city", "day_of_year", "_year", "y_tmax"]]
    all_years = sorted(df["_year"].unique())

    # Full-dataset climo used only as cold-start fallback for the first year
    global_climo = (
        observed.groupby(["city", "day_of_year"])["y_tmax"]
        .mean()
        .rename("climo_mean_doy")
        .reset_index()
    )

    climo_list = []
    for year in all_years:
        prior = observed[observed["_year"] < year]
        if prior.empty:
            year_climo = global_climo.copy()
        else:
            year_climo = (
                prior.groupby(["city", "day_of_year"])["y_tmax"]
                .mean()
                .rename("climo_mean_doy")
                .reset_index()
            )
        year_climo["_year"] = year
        climo_list.append(year_climo)

    climo_df = pd.concat(climo_list, ignore_index=True)
    df = df.merge(climo_df, on=["city", "day_of_year", "_year"], how="left")
    df = df.drop(columns=["_year"])
    df["forecast_minus_climo"] = df["forecast_high"] - df["climo_mean_doy"]
    return df


def add_climate_indices(df: pd.DataFrame, indices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join monthly climate regime indices onto the feature table.

    indices_df: columns [year, month, ao_index, nao_index, oni, pna_index, pdo_index]
    Rows in df without a matching (year, month) get NaN, which the model
    imputes with global medians.
    """
    df = df.copy()
    df["_year"]  = pd.to_datetime(df["date"]).dt.year
    df["_month"] = pd.to_datetime(df["date"]).dt.month
    idx_cols = ["year", "month"] + [
        c for c in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index")
        if c in indices_df.columns
    ]
    idx = indices_df[idx_cols].rename(columns={"year": "_year", "month": "_month"})
    df = df.merge(idx, on=["_year", "_month"], how="left")
    df = df.drop(columns=["_year", "_month"])
    return df


def add_mjo_indices(df: pd.DataFrame, mjo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join daily MJO RMM amplitude and phase onto the feature table.

    mjo_df: columns [date, mjo_amplitude, mjo_phase_sin, mjo_phase_cos]
      mjo_amplitude > 1  → active MJO; < 1 → weak/neutral
      mjo_phase_sin/cos  → cyclical encoding of MJO phase (1–8)

    Dates before the BOM archive starts (Jun 1974) or any gaps get NaN,
    which build_feature_table() imputes with zeros (neutral/inactive MJO).
    """
    df = df.copy()
    mjo = mjo_df[["date", "mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"]].copy()
    mjo["date"] = pd.to_datetime(mjo["date"])
    df = df.merge(mjo, on="date", how="left")
    return df


def build_feature_table(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    gfs_df: pd.DataFrame = None,
    indices_df: pd.DataFrame = None,
    gefs_df: pd.DataFrame = None,
    mjo_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge historical TMAX with forecast highs, then add all features.

    historical_df: columns [date, city, tmax]
    forecast_df:   columns [city, forecast_high, target_date]  (NWS live forecast)
                   may also contain [nbm_high] if fetched alongside NWS
    gfs_df:        columns [date, city, forecast_high_gfs]     (OpenMeteo historical GFS)
    indices_df:    columns [year, month, ao_index, oni, ...]   (monthly climate regime indices)
    gefs_df:       columns [date, city, gefs_spread]           (GEFS ensemble member spread)
    mjo_df:        columns [date, mjo_amplitude, mjo_phase_sin, mjo_phase_cos]  (daily MJO)

    forecast_high is set to (in priority order):
      1. NWS live forecast (forecast_df)  — used for today/tomorrow in live mode
      2. GFS historical forecast (gfs_df) — used for historical dates in backtest
      3. climo_mean_doy fallback          — filled after add_climatology()

    nbm_high: NBM gridded max temperature from NWS /gridpoints endpoint.
      - Taken from forecast_df if present (live mode)
      - Missing values imputed with forecast_high

    gefs_spread: std of GEFS ensemble members from OpenMeteo ensemble API.
      - Available from ~Dec 2022 onward
      - Missing values imputed with ensemble_spread (3-model inter-model spread)

    Returns a combined DataFrame ready for model.fit() and model.predict().
    """
    hist = historical_df.rename(columns={"tmax": "y_tmax"}).copy()
    hist["date"] = pd.to_datetime(hist["date"])

    # Merge in NWS live forecast (today/tomorrow); extract nbm_high if present
    fcast = forecast_df.rename(columns={"target_date": "date"}).copy()
    fcast["date"] = pd.to_datetime(fcast["date"])
    fcast_cols = ["city", "date", "forecast_high"]
    if "nbm_high" in fcast.columns:
        fcast_cols.append("nbm_high")

    df = hist.merge(fcast[fcast_cols], on=["city", "date"], how="outer")

    # Merge in GFS historical forecasts, ECMWF, ensemble spread, and precip
    if gfs_df is not None:
        gfs = gfs_df.copy()
        gfs["date"] = pd.to_datetime(gfs["date"])
        merge_cols = ["city", "date", "forecast_high_gfs"]
        for optional in ("ensemble_spread", "forecast_high_ecmwf",
                         "ecmwf_minus_gfs", "precip_forecast",
                         "temp_850hpa", "shortwave_radiation", "dew_point_max"):
            if optional in gfs.columns:
                merge_cols.append(optional)
        df = df.merge(gfs[merge_cols], on=["city", "date"], how="left")
        missing = df["forecast_high"].isna()
        df.loc[missing, "forecast_high"] = df.loc[missing, "forecast_high_gfs"]
        df = df.drop(columns=["forecast_high_gfs"])

    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # ── Issue 2 fix: day-ahead forecast alignment ──────────────────
    # OpenMeteo's historical forecast API returns the same-day model run
    # (00Z–18Z on date D), which incorporates observations from D itself
    # and is far more accurate than what a trader has at market open.
    # Shifting by 1 day ensures every forecast feature reflects what was
    # available the *previous* day — a genuine day-ahead signal.
    _gfs_cols = [c for c in ("forecast_high", "ensemble_spread",
                              "forecast_high_ecmwf", "ecmwf_minus_gfs",
                              "precip_forecast", "temp_850hpa",
                              "shortwave_radiation", "dew_point_max") if c in df.columns]
    for col in _gfs_cols:
        df[col] = df.groupby("city")[col].shift(1)

    # Re-apply NWS live forecasts after the shift.
    # The shift above is correct for GFS historical data (look-ahead prevention),
    # but NWS forecasts in forecast_df are already for the target date and must
    # not be displaced. Restore them here.
    _fcast_restore = fcast[fcast_cols].copy()
    _fcast_restore["date"] = pd.to_datetime(_fcast_restore["date"])
    for _, frow in _fcast_restore.iterrows():
        if pd.notna(frow.get("forecast_high")):
            mask = (df["city"] == frow["city"]) & (df["date"] == frow["date"])
            df.loc[mask, "forecast_high"] = frow["forecast_high"]
        if "nbm_high" in frow and pd.notna(frow.get("nbm_high")) and "nbm_high" in df.columns:
            mask = (df["city"] == frow["city"]) & (df["date"] == frow["date"])
            df.loc[mask, "nbm_high"] = frow["nbm_high"]

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_climatology(df)

    # Impute NaN lag features with climo_mean_doy so predictions never receive NaN.
    # This only affects the very first rows of each city where no lag exists yet.
    for col in ("lag1_tmax", "lag3_mean_tmax", "lag7_mean_tmax",
                "lag14_mean_tmax", "lag21_mean_tmax", "lag30_mean_tmax"):
        mask = df[col].isna() & df["climo_mean_doy"].notna()
        df.loc[mask, col] = df.loc[mask, "climo_mean_doy"]
    # lag_trend imputed to 0 (no trend signal) for warm-up rows
    if "lag_trend" in df.columns:
        df["lag_trend"] = df["lag_trend"].fillna(0.0)

    # How much the NWS forecast departs from the most recent observed day.
    # A large positive value means a warm front is moving in; large negative = cold.
    # Use filled forecast_high so this is never NaN (0 = forecast tracks recent obs).
    _fcast_filled = df["forecast_high"].fillna(df["climo_mean_doy"])
    df["forecast_minus_lag1"] = _fcast_filled - df["lag1_tmax"]
    df["forecast_minus_lag1"] = df["forecast_minus_lag1"].fillna(0.0)

    # Impute missing ensemble_spread with its per-city median (handles dates
    # before model archive starts or NWS-only live rows).
    if "ensemble_spread" in df.columns:
        city_medians = df.groupby("city")["ensemble_spread"].transform("median")
        df["ensemble_spread"] = df["ensemble_spread"].fillna(city_medians)

    # Merge monthly AO / NAO / ONI climate regime indices.
    # These are the same for all cities in a given month, so we join on (year, month).
    if indices_df is not None:
        df = add_climate_indices(df, indices_df)
        # Impute with global median — indices don't vary by city
        for col in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index"):
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    # Ensure monthly indices exist even if indices_df was None
    for col in ("ao_index", "nao_index", "oni", "pna_index", "pdo_index"):
        if col not in df.columns:
            df[col] = 0.0

    # NBM gridded max temperature.
    # Available from forecast_df (live mode); impute historical with forecast_high.
    if "nbm_high" not in df.columns:
        df["nbm_high"] = np.nan
    df["nbm_high"] = df["nbm_high"].fillna(df["forecast_high"])

    # ECMWF signed disagreement: positive = ECMWF warmer than GFS.
    # Impute with 0 (no disagreement) where ECMWF data is unavailable.
    if "ecmwf_minus_gfs" not in df.columns:
        df["ecmwf_minus_gfs"] = 0.0
    df["ecmwf_minus_gfs"] = df["ecmwf_minus_gfs"].fillna(0.0)

    # GFS precipitation forecast (mm). Dry days tend toward higher tmax.
    # Impute with per-city-month median where missing.
    if "precip_forecast" not in df.columns:
        df["precip_forecast"] = np.nan
    city_month_precip = df.groupby(["city", "month"])["precip_forecast"].transform("median")
    df["precip_forecast"] = df["precip_forecast"].fillna(city_month_precip).fillna(0.0)

    # 850hPa temperature (°F daily max). Impute with per-city-month median.
    if "temp_850hpa" not in df.columns:
        df["temp_850hpa"] = np.nan
    city_month_850 = df.groupby(["city", "month"])["temp_850hpa"].transform("median")
    df["temp_850hpa"] = (
        df["temp_850hpa"]
        .fillna(city_month_850)
        .fillna(df["climo_mean_doy"] - 20.0)
        .fillna(df["forecast_high"] - 20.0)
        .fillna(30.0)
    )

    # Shortwave radiation sum (MJ/m²/day). Impute with per-city-month median.
    if "shortwave_radiation" not in df.columns:
        df["shortwave_radiation"] = np.nan
    city_month_sw = df.groupby(["city", "month"])["shortwave_radiation"].transform("median")
    df["shortwave_radiation"] = df["shortwave_radiation"].fillna(city_month_sw).fillna(0.0)

    # Dew point daily max (°F). Impute with per-city-month median.
    if "dew_point_max" not in df.columns:
        df["dew_point_max"] = np.nan
    city_month_dp = df.groupby(["city", "month"])["dew_point_max"].transform("median")
    df["dew_point_max"] = (
        df["dew_point_max"]
        .fillna(city_month_dp)
        .fillna(df["climo_mean_doy"] - 25.0)
        .fillna(df["forecast_high"] - 25.0)
        .fillna(20.0)
    )

    # GEFS ensemble spread (std of 31 GEFS members).
    # Available from Dec 2022 onward; impute older dates with 3-model ensemble_spread.
    if gefs_df is not None:
        gefs = gefs_df.copy()
        gefs["date"] = pd.to_datetime(gefs["date"])
        df = df.merge(gefs[["city", "date", "gefs_spread"]], on=["city", "date"], how="left")
    else:
        df["gefs_spread"] = np.nan

    # Fallback: use 3-model ensemble_spread where GEFS data is missing
    if "ensemble_spread" in df.columns:
        df["gefs_spread"] = df["gefs_spread"].fillna(df["ensemble_spread"])
    else:
        city_gefs_medians = df.groupby("city")["gefs_spread"].transform("median")
        df["gefs_spread"] = df["gefs_spread"].fillna(city_gefs_medians)

    # MJO daily indices (amplitude + cyclical phase encoding).
    # Impute with 0 = neutral/inactive MJO (physically correct for missing dates).
    if mjo_df is not None:
        df = add_mjo_indices(df, mjo_df)
    for col in ("mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"):
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Trailing 14-day mean NWS signed forecast error per city.
    # At date T: mean(forecast_high[D] - y_tmax[D]) for D in [T-14, T-1].
    # forecast_high is already day-ahead aligned (shifted above), so
    # forecast_high[D] - y_tmax[D] is the GFS/NWS error for day D.
    # shift(1) inside the rolling ensures day T only sees errors through T-1.
    # Rows where y_tmax is NaN (future dates) contribute NaN errors, which
    # rolling() ignores; min_periods=3 keeps the feature defined early in history.
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    _nws_error = df["forecast_high"] - df["y_tmax"]
    df["nws_bias_14d"] = (
        df.assign(_e=_nws_error)
        .groupby("city")["_e"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
        .fillna(0.0)
    )

    return df
