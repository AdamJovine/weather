#!/usr/bin/env python3
"""
Backtest a simple weather-vs-price mispricing bot.

Design goals:
  - Simple and interpretable.
  - Uses weather forecast data + market prices together.
  - Avoids overfit-heavy ML stacks.

Core logic:
  1) Build a daily fair-value temperature distribution per city/date:
       mu = forecast_blend + rolling_bias(error)
       sigma = rolling_std(error) * (1 + spread_alpha * ensemble_spread)
  2) Convert each contract to model YES probability via Normal CDF.
  3) Every check interval (default 5 min; backtest data is hourly candles),
     detect if market implied probability is far from model probability.
  4) Enter once per ticker when edge exceeds threshold + uncertainty buffer.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy import contract_yes_outcome, parse_contract


def _safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return float("nan")
    return float(num) / float(den)


def _forecast_blend(gfs: pd.Series, ecmwf: pd.Series, w_ecmwf: float) -> pd.Series:
    g = gfs.astype(float)
    e = ecmwf.astype(float)
    out = (1.0 - float(w_ecmwf)) * g + float(w_ecmwf) * e
    out = out.where(g.notna() | e.notna(), np.nan)
    out = out.where(g.notna() & e.notna(), g.fillna(e))
    return out


def _build_daily_distribution(
    db_path: str,
    lookback_days: int,
    min_hist_days: int,
    sigma_floor: float,
    spread_alpha: float,
    ecmwf_weight: float,
    nbm_weight: float = 0.0,
    disagree_alpha: float = 0.0,
    t850_sigma_alpha: float = 0.0,
    return_errors: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    with sqlite3.connect(db_path) as conn:
        daily = pd.read_sql_query(
            """
            SELECT f.city, f.date,
                   f.forecast_high_gfs, f.forecast_high_ecmwf, f.ensemble_spread,
                   f.ecmwf_minus_gfs, f.temp_850hpa,
                   n.nbm_high,
                   w.tmax
            FROM forecasts_daily f
            LEFT JOIN weather_daily w
              ON w.city = f.city AND w.date = f.date
            LEFT JOIN nws_forecasts n
              ON n.city = f.city AND n.target_date = f.date
            """,
            conn,
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)
    daily["forecast_blend"] = _forecast_blend(
        daily["forecast_high_gfs"],
        daily["forecast_high_ecmwf"],
        w_ecmwf=float(ecmwf_weight),
    )

    # Blend in NWS NBM forecast when available.
    if float(nbm_weight) > 0:
        nbm = daily["nbm_high"].astype(float)
        has_nbm = nbm.notna()
        daily["forecast_blend"] = daily["forecast_blend"].where(
            ~has_nbm,
            (1.0 - float(nbm_weight)) * daily["forecast_blend"] + float(nbm_weight) * nbm,
        )

    daily = daily[daily["forecast_blend"].notna()].copy()
    daily["error"] = daily["tmax"] - daily["forecast_blend"]

    # Compute ecmwf_minus_gfs from columns when the dedicated column is missing.
    if daily["ecmwf_minus_gfs"].notna().sum() == 0:
        gfs = daily["forecast_high_gfs"].astype(float)
        ecmwf = daily["forecast_high_ecmwf"].astype(float)
        daily["ecmwf_minus_gfs"] = ecmwf - gfs

    outputs: list[pd.DataFrame] = []
    errors_dict: dict[tuple[str, str], np.ndarray] = {}
    for city, g in daily.groupby("city", sort=False):
        g = g.sort_values("date").copy()
        err_shift = g["error"].shift(1)

        if return_errors:
            dates_arr = g["date"].values
            errors_vals = err_shift.values
            for i in range(len(g)):
                start = max(0, i - lookback_days + 1) if lookback_days > 0 else 0
                window = errors_vals[start:i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) >= min_hist_days:
                    dt_str = pd.Timestamp(dates_arr[i]).strftime("%Y-%m-%d")
                    errors_dict[(city, dt_str)] = valid

        if lookback_days > 0:
            bias = err_shift.rolling(lookback_days, min_periods=min_hist_days).mean()
            sigma = err_shift.rolling(lookback_days, min_periods=min_hist_days).std()
            n_hist = err_shift.rolling(lookback_days, min_periods=1).count()
        else:
            bias = err_shift.expanding(min_periods=min_hist_days).mean()
            sigma = err_shift.expanding(min_periods=min_hist_days).std()
            n_hist = err_shift.expanding(min_periods=1).count()

        # Fallback when history is short.
        city_sigma = float(err_shift.std(ddof=1)) if err_shift.notna().sum() >= 2 else float("nan")
        if not np.isfinite(city_sigma):
            city_sigma = float(sigma_floor)

        g["bias"] = bias.fillna(0.0)
        g["sigma_err"] = sigma.fillna(city_sigma)
        g["n_hist"] = n_hist.fillna(0.0)
        g["mu"] = g["forecast_blend"] + g["bias"]
        g["sigma"] = g["sigma_err"] * (1.0 + float(spread_alpha) * g["ensemble_spread"].fillna(0.0).clip(lower=0.0))

        # Inflate sigma when GFS and ECMWF disagree.
        if float(disagree_alpha) > 0:
            disagree = g["ecmwf_minus_gfs"].fillna(0.0).abs()
            g["sigma"] *= (1.0 + float(disagree_alpha) * disagree / g["sigma_err"].clip(lower=1.0))

        # Inflate sigma when 850hPa temperature is anomalous.
        if float(t850_sigma_alpha) > 0:
            t850 = g["temp_850hpa"].astype(float)
            if t850.notna().sum() >= max(2, min_hist_days):
                t850_mean = t850.shift(1).rolling(lookback_days, min_periods=min_hist_days).mean()
                t850_std = t850.shift(1).rolling(lookback_days, min_periods=min_hist_days).std().clip(lower=1.0)
                t850_z = ((t850 - t850_mean) / t850_std).fillna(0.0).abs()
                g["sigma"] *= (1.0 + float(t850_sigma_alpha) * t850_z)

        g["sigma"] = g["sigma"].clip(lower=float(sigma_floor))
        outputs.append(g[["city", "date", "mu", "sigma", "n_hist", "forecast_blend", "ensemble_spread"]])

    result = pd.concat(outputs, ignore_index=True)
    if return_errors:
        return result, errors_dict
    return result


def _load_market_rows(
    db_path: str,
    min_volume: float,
    min_price: float,
    max_price: float,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT c.ticker, c.ts, c.close_dollars, c.volume,
                   m.city, m.settlement_date, m.title,
                   w.tmax
            FROM kalshi_candles c
            JOIN kalshi_markets m ON m.ticker = c.ticker
            LEFT JOIN weather_daily w
              ON w.city = m.city AND w.date = m.settlement_date
            """,
            conn,
        )

    df = df[df["volume"] >= float(min_volume)].copy()
    df = df[df["close_dollars"].between(float(min_price), float(max_price))].copy()
    df = df[df["tmax"].notna()].copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["year"] = df["settlement_date"].dt.year.astype(int)
    df["ts_et"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["hour_et"] = df["ts_et"].dt.hour.astype(int)

    parsed = []
    for title in df["title"].astype(str):
        try:
            parsed.append(parse_contract(title, ""))
        except Exception:
            parsed.append(None)
    df["contract"] = parsed
    df = df[df["contract"].notna()].copy()

    mtypes = []
    thresh = []
    low = []
    high = []
    for c in df["contract"]:
        mt = str(c["market_type"])
        mtypes.append(mt)
        thresh.append(float(c.get("threshold", np.nan)))
        low.append(float(c.get("low", np.nan)))
        high.append(float(c.get("high", np.nan)))
    df["market_type"] = mtypes
    df["threshold"] = thresh
    df["low"] = low
    df["high"] = high

    df["yes_outcome"] = [
        contract_yes_outcome(c, float(t))
        for c, t in zip(df["contract"], df["tmax"])
    ]
    df["no_outcome"] = 1 - df["yes_outcome"]
    return df


def _compute_yes_prob(df: pd.DataFrame) -> pd.Series:
    sigma = df["sigma"].astype(float).clip(lower=0.5)
    mu = df["mu"].astype(float)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    mt = df["market_type"]

    m = mt == "gt"
    out.loc[m] = 1.0 - norm.cdf(df.loc[m, "threshold"], loc=mu.loc[m], scale=sigma.loc[m])

    m = mt == "lt"
    out.loc[m] = norm.cdf(df.loc[m, "threshold"], loc=mu.loc[m], scale=sigma.loc[m])

    m = mt == "geq"
    out.loc[m] = 1.0 - norm.cdf(df.loc[m, "threshold"], loc=mu.loc[m], scale=sigma.loc[m])

    m = mt == "leq"
    out.loc[m] = norm.cdf(df.loc[m, "threshold"], loc=mu.loc[m], scale=sigma.loc[m])

    m = mt == "range"
    out.loc[m] = (
        norm.cdf(df.loc[m, "high"], loc=mu.loc[m], scale=sigma.loc[m])
        - norm.cdf(df.loc[m, "low"], loc=mu.loc[m], scale=sigma.loc[m])
    )
    return out.clip(lower=0.0, upper=1.0)


def _compute_yes_prob_empirical(
    df: pd.DataFrame,
    errors_dict: dict[tuple[str, str], np.ndarray],
) -> pd.Series:
    """Compute P(YES) using the empirical forecast error distribution.

    Instead of assuming errors ~ Normal(mu, sigma), use the actual historical
    error samples:  simulated_tmax = forecast_blend + each_error.
    This captures fat tails, skewness, and city-specific shapes that the Normal
    CDF misses.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    # Use the 'date' column that comes from merging with dist_daily
    date_col = df["date"] if "date" in df.columns else df["settlement_date"]
    date_strs = pd.to_datetime(date_col).dt.strftime("%Y-%m-%d")

    for (city, date_str), group_idx in df.groupby([df["city"], date_strs]).groups.items():
        key = (city, date_str)
        if key not in errors_dict:
            continue

        sub = df.loc[group_idx]
        errors = errors_dict[key]
        forecast = float(sub["forecast_blend"].iloc[0])
        simulated = np.sort(forecast + errors)
        n = len(simulated)

        for mt_val in sub["market_type"].unique():
            mt_sub = sub[sub["market_type"] == mt_val]

            if mt_val == "gt":
                thresholds = mt_sub["threshold"].astype(float).values
                probs = 1.0 - np.searchsorted(simulated, thresholds, side="right") / n
                out.loc[mt_sub.index] = probs
            elif mt_val == "geq":
                thresholds = mt_sub["threshold"].astype(float).values
                probs = 1.0 - np.searchsorted(simulated, thresholds, side="left") / n
                out.loc[mt_sub.index] = probs
            elif mt_val == "lt":
                thresholds = mt_sub["threshold"].astype(float).values
                probs = np.searchsorted(simulated, thresholds, side="left") / n
                out.loc[mt_sub.index] = probs
            elif mt_val == "leq":
                thresholds = mt_sub["threshold"].astype(float).values
                probs = np.searchsorted(simulated, thresholds, side="right") / n
                out.loc[mt_sub.index] = probs
            elif mt_val == "range":
                lows = mt_sub["low"].astype(float).values
                highs = mt_sub["high"].astype(float).values
                counts_le_high = np.searchsorted(simulated, highs, side="right")
                counts_lt_low = np.searchsorted(simulated, lows, side="left")
                probs = (counts_le_high - counts_lt_low) / n
                out.loc[mt_sub.index] = probs

    # Fall back to Normal CDF for rows without enough empirical history
    missing = out.isna()
    if missing.any():
        out.loc[missing] = _compute_yes_prob(df.loc[missing])

    return out.clip(lower=0.0, upper=1.0)


def _apply_probability_safety(
    rows: pd.DataFrame,
    prob_floor: float,
    prob_ceil: float,
    confidence_shrink_k: float,
) -> pd.DataFrame:
    out = rows.copy()
    p = out["p_yes_model"].astype(float).clip(lower=float(prob_floor), upper=float(prob_ceil))
    if float(confidence_shrink_k) > 0:
        # Shrink extreme probabilities toward 50% when historical calibration depth is low.
        n = out["n_hist"].astype(float).clip(lower=0.0)
        w = n / (n + float(confidence_shrink_k))
        p = 0.5 + w * (p - 0.5)
    out["p_yes_used"] = p.clip(lower=float(prob_floor), upper=float(prob_ceil))
    return out


def _pick_entries(
    rows: pd.DataFrame,
    check_minutes: int,
    side_mode: str,
    min_edge: float,
    uncertainty_z: float,
    half_spread: float,
    min_minutes_between_ticker_trades: int,
    max_trades_per_ticker_day: int,
    max_daily_trades: int,
    max_city_daily_trades: int,
) -> pd.DataFrame:
    rows = rows.copy()
    slot = int(max(1, check_minutes)) * 60
    rows["check_slot"] = (rows["ts"] // slot).astype(np.int64)

    # "Check every N minutes": choose first candle seen in each slot.
    rows = (
        rows.sort_values(["ticker", "ts"])
        .groupby(["ticker", "check_slot"], as_index=False)
        .first()
    )

    rows["yes_ask"] = (rows["close_dollars"] + float(half_spread)).clip(lower=0.01, upper=0.99)
    rows["no_ask"] = (1.0 - rows["close_dollars"] + float(half_spread)).clip(lower=0.01, upper=0.99)
    rows["edge_yes"] = rows["p_yes_used"] - rows["yes_ask"]
    rows["edge_no"] = (1.0 - rows["p_yes_used"]) - rows["no_ask"]

    n_eff = rows["n_hist"].clip(lower=1.0)
    p = rows["p_yes_used"].clip(0.0, 1.0)
    p_std = np.sqrt((p * (1.0 - p)) / n_eff)
    rows["effective_min_edge"] = float(min_edge) + float(uncertainty_z) * p_std

    if side_mode == "yes":
        rows["side"] = "yes"
        rows["edge_chosen"] = rows["edge_yes"]
        rows["price_chosen"] = rows["yes_ask"]
        rows["outcome_chosen"] = rows["yes_outcome"]
    elif side_mode == "no":
        rows["side"] = "no"
        rows["edge_chosen"] = rows["edge_no"]
        rows["price_chosen"] = rows["no_ask"]
        rows["outcome_chosen"] = rows["no_outcome"]
    else:
        choose_yes = rows["edge_yes"] >= rows["edge_no"]
        rows["side"] = np.where(choose_yes, "yes", "no")
        rows["edge_chosen"] = np.where(choose_yes, rows["edge_yes"], rows["edge_no"])
        rows["price_chosen"] = np.where(choose_yes, rows["yes_ask"], rows["no_ask"])
        rows["outcome_chosen"] = np.where(choose_yes, rows["yes_outcome"], rows["no_outcome"])

    rows["trade_signal"] = rows["edge_chosen"] >= rows["effective_min_edge"]
    entries = rows[rows["trade_signal"]].sort_values(["ts", "ticker"]).copy()

    if entries.empty:
        return entries

    entries["trade_date_et"] = pd.to_datetime(entries["ts"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.date

    # Optional cooldown to avoid repeatedly hammering one market ticker.
    if int(min_minutes_between_ticker_trades) > 0:
        cooldown_sec = int(min_minutes_between_ticker_trades) * 60
        entries["prev_ts_ticker"] = entries.groupby("ticker")["ts"].shift(1)
        entries["ticker_gap_sec"] = entries["ts"] - entries["prev_ts_ticker"]
        entries = entries[
            entries["prev_ts_ticker"].isna() | (entries["ticker_gap_sec"] >= cooldown_sec)
        ].copy()

    # Optional per-ticker-per-day cap to diversify across markets.
    if int(max_trades_per_ticker_day) > 0:
        entries["ticker_daily_rank"] = entries.groupby(["trade_date_et", "ticker"]).cumcount()
        entries = entries[entries["ticker_daily_rank"] < int(max_trades_per_ticker_day)].copy()

    if int(max_city_daily_trades) > 0:
        entries["city_daily_rank"] = entries.groupby(["trade_date_et", "city"]).cumcount()
        entries = entries[entries["city_daily_rank"] < int(max_city_daily_trades)].copy()

    if int(max_daily_trades) > 0:
        entries["daily_rank"] = entries.groupby(["trade_date_et"]).cumcount()
        entries = entries[entries["daily_rank"] < int(max_daily_trades)].copy()

    return entries


def _settle(entries: pd.DataFrame, stake: float, fee_rate: float) -> pd.DataFrame:
    if entries.empty:
        return entries
    t = entries.copy()
    t["stake"] = float(stake)
    t["contracts"] = t["stake"] / t["price_chosen"]
    gross = t["contracts"] * (t["outcome_chosen"] - t["price_chosen"])
    t["pnl"] = gross - float(fee_rate) * gross.clip(lower=0.0)
    return t


def _summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=by + ["trades", "stake_total", "pnl_total", "roi", "win_rate"])
    if not by:
        trades = int(len(df))
        stake_total = float(df["stake"].sum())
        pnl_total = float(df["pnl"].sum())
        win_rate = float(df["outcome_chosen"].mean())
        return pd.DataFrame(
            [
                {
                    "trades": trades,
                    "stake_total": stake_total,
                    "pnl_total": pnl_total,
                    "roi": _safe_div(pnl_total, stake_total),
                    "win_rate": win_rate,
                }
            ]
        )
    out = (
        df.groupby(by)
        .agg(
            trades=("pnl", "size"),
            stake_total=("stake", "sum"),
            pnl_total=("pnl", "sum"),
            win_rate=("outcome_chosen", "mean"),
        )
        .reset_index()
    )
    out["roi"] = out["pnl_total"] / out["stake_total"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/weather.db")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--city", default=None)
    ap.add_argument("--cities", default=None, help="Comma list")
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.02)
    ap.add_argument("--half-spread", type=float, default=0.02)
    ap.add_argument("--min-volume", type=float, default=10.0)
    ap.add_argument("--min-price", type=float, default=0.05)
    ap.add_argument("--max-price", type=float, default=0.95)
    ap.add_argument("--check-minutes", type=int, default=5)
    ap.add_argument(
        "--allow-settlement-day",
        action="store_true",
        help="If set, allow entries on settlement date (default: only pre-settlement entries).",
    )
    ap.add_argument(
        "--prob-floor",
        type=float,
        default=0.03,
        help="Clip model YES probabilities to [prob_floor, prob_ceil].",
    )
    ap.add_argument(
        "--prob-ceil",
        type=float,
        default=0.97,
        help="Clip model YES probabilities to [prob_floor, prob_ceil].",
    )
    ap.add_argument(
        "--confidence-shrink-k",
        type=float,
        default=60.0,
        help="Probability shrink strength toward 50%%: w=n_hist/(n_hist+k).",
    )
    ap.add_argument(
        "--min-minutes-between-ticker-trades",
        type=int,
        default=0,
        help="Cooldown between trades on same ticker (0 = off).",
    )
    ap.add_argument(
        "--max-trades-per-ticker-day",
        type=int,
        default=3,
        help="Per-ticker/day cap for diversification (0 = no cap).",
    )
    ap.add_argument("--max-daily-trades", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-city-daily-trades", type=int, default=0, help="0 = no cap")
    ap.add_argument("--side-mode", choices=["best", "yes", "no"], default="best")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--uncertainty-z", type=float, default=1.0)
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--min-hist-days", type=int, default=45)
    ap.add_argument("--sigma-floor", type=float, default=3.0)
    ap.add_argument("--spread-alpha", type=float, default=0.20)
    ap.add_argument("--ecmwf-weight", type=float, default=0.50)
    ap.add_argument("--nbm-weight", type=float, default=0.0,
                    help="Blend weight for NWS NBM forecast (0 = off).")
    ap.add_argument("--disagree-alpha", type=float, default=0.0,
                    help="Sigma inflation when GFS/ECMWF disagree (0 = off).")
    ap.add_argument("--t850-sigma-alpha", type=float, default=0.0,
                    help="Sigma inflation from 850hPa anomaly (0 = off).")
    ap.add_argument("--use-empirical", action="store_true",
                    help="Use empirical error CDF instead of Normal CDF.")
    ap.add_argument("--csv-out", default=None, help="Optional trade-level CSV output")
    args = ap.parse_args()

    build_kw = dict(
        db_path=args.db,
        lookback_days=args.lookback_days,
        min_hist_days=args.min_hist_days,
        sigma_floor=args.sigma_floor,
        spread_alpha=args.spread_alpha,
        ecmwf_weight=args.ecmwf_weight,
        nbm_weight=args.nbm_weight,
        disagree_alpha=args.disagree_alpha,
        t850_sigma_alpha=args.t850_sigma_alpha,
        return_errors=args.use_empirical,
    )
    errors_dict: dict = {}
    result = _build_daily_distribution(**build_kw)
    if args.use_empirical:
        dist_daily, errors_dict = result
    else:
        dist_daily = result
    mkt = _load_market_rows(
        db_path=args.db,
        min_volume=args.min_volume,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    mkt = mkt.merge(
        dist_daily,
        left_on=["city", "settlement_date"],
        right_on=["city", "date"],
        how="left",
    )
    mkt = mkt[mkt["mu"].notna() & mkt["sigma"].notna()].copy()

    city_set: set[str] = set()
    if args.city:
        city_set.add(args.city.strip())
    if args.cities:
        city_set.update({x.strip() for x in args.cities.split(",") if x.strip()})
    if city_set:
        mkt = mkt[mkt["city"].isin(city_set)].copy()
    if args.start:
        mkt = mkt[mkt["settlement_date"] >= pd.to_datetime(args.start)].copy()
    if args.end:
        mkt = mkt[mkt["settlement_date"] <= pd.to_datetime(args.end)].copy()
    mkt["trade_date_et"] = mkt["ts_et"].dt.date
    mkt["settle_date"] = mkt["settlement_date"].dt.date
    if not args.allow_settlement_day:
        mkt = mkt[mkt["trade_date_et"] < mkt["settle_date"]].copy()
    if mkt.empty:
        raise RuntimeError("No market rows remain after filters.")

    if args.use_empirical:
        mkt["p_yes_model"] = _compute_yes_prob_empirical(mkt, errors_dict)
    else:
        mkt["p_yes_model"] = _compute_yes_prob(mkt)
    mkt = mkt[mkt["p_yes_model"].notna()].copy()
    mkt = _apply_probability_safety(
        mkt,
        prob_floor=args.prob_floor,
        prob_ceil=args.prob_ceil,
        confidence_shrink_k=args.confidence_shrink_k,
    )

    entries = _pick_entries(
        mkt,
        check_minutes=args.check_minutes,
        side_mode=args.side_mode,
        min_edge=args.min_edge,
        uncertainty_z=args.uncertainty_z,
        half_spread=args.half_spread,
        min_minutes_between_ticker_trades=args.min_minutes_between_ticker_trades,
        max_trades_per_ticker_day=args.max_trades_per_ticker_day,
        max_daily_trades=args.max_daily_trades,
        max_city_daily_trades=args.max_city_daily_trades,
    )
    trades = _settle(entries, stake=args.stake, fee_rate=args.fee_rate)
    if trades.empty:
        raise RuntimeError("No trades triggered under current settings.")

    overall = _summary(trades, by=[])
    by_year = _summary(trades, by=["year"])
    by_city = _summary(trades, by=["city"])
    by_side = _summary(trades, by=["side"])
    by_hour = _summary(trades, by=["hour_et"])

    print("Simple Mispricing Bot Backtest")
    print(f"Date range: {trades['settlement_date'].min().date()} to {trades['settlement_date'].max().date()}")
    print(f"Trades: {len(trades)}")
    print(f"Tickers covered: {trades['ticker'].nunique()}")
    print(f"Side mode: {args.side_mode}")
    print(f"Check interval: {args.check_minutes} min (hourly candles in backtest data)")
    print(f"Min edge: {args.min_edge:.4f} | Uncertainty z: {args.uncertainty_z:.2f}")
    print(
        "Prob safety: "
        f"[{args.prob_floor:.2f}, {args.prob_ceil:.2f}] "
        f"| shrink_k={args.confidence_shrink_k:.1f}"
    )
    print(
        "Ticker controls: "
        f"cooldown={args.min_minutes_between_ticker_trades}m "
        f"| max_ticker_day={args.max_trades_per_ticker_day}"
    )
    print(f"Max daily trades: {args.max_daily_trades} | Max city/day: {args.max_city_daily_trades}")
    print(f"Lookback days: {args.lookback_days} | Min hist days: {args.min_hist_days}")
    print()
    print("Overall:")
    print(overall.to_string(index=False))
    print("\nBy year:")
    print(by_year.to_string(index=False))
    print("\nBy city:")
    print(by_city.sort_values("roi", ascending=False).to_string(index=False))
    print("\nBy side:")
    print(by_side.to_string(index=False))
    print("\nBy hour:")
    print(by_hour.sort_values("roi", ascending=False).to_string(index=False))

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out, index=False)
        print(f"\nTrade CSV: {out}")


if __name__ == "__main__":
    main()
