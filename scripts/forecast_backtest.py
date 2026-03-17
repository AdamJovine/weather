"""
Forecast accuracy backtest — measures how well the model predicts daily high temperature.

Unlike the trading backtest, this ignores market prices entirely.
It evaluates raw model predictions against observed y_tmax using the same
walk-forward training setup as run_live.py (no lookahead).

Metrics reported (overall + per city + per month):
  MAE        – mean absolute error (°F)
  RMSE       – root mean squared error (°F)
  Bias       – mean signed error; positive = model runs warm
  Within N°F – % of predictions within 1/2/3/5°F of actual
  Cov68      – % of actuals inside predicted 68% interval (mu ± 1σ)
  Cov90      – % of actuals inside predicted 90% interval (mu ± 1.645σ)
  Sharpness  – average predicted σ; lower = sharper / more decisive

Baselines compared:
  NWS/GFS    – raw forecast_high from the database (same signal the model uses)
  Climo      – per-city day-of-year historical mean

Usage:
    python scripts/forecast_backtest.py
    python scripts/forecast_backtest.py --model ARD --start 2024-01-01
    python scripts/forecast_backtest.py --model BayesianRidge --no-validate
    python scripts/forecast_backtest.py --all-models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.config import BacktestConfig, MODEL_REGISTRY, MODEL_REFIT_EVERY
from backtesting.data_loader import DataLoader
from src import model as model_module

OUT_DIR = Path("logs/forecast_backtest")

PRUNE_THRESHOLD = 0.01   # °F — permutation ΔMae below this → prune candidate


# ── walk-forward evaluation ────────────────────────────────────────────────────

def _run_model(model_name: str, city_df: pd.DataFrame,
               lookback: int, refit_every: int, min_train_rows: int,
               start_ts: pd.Timestamp, end_ts: pd.Timestamp,
               features: list[str] = None) -> list[dict]:
    """
    Walk forward through city_df, training on rows before index i and
    predicting row i.  Returns a list of result dicts.

    features: if provided, temporarily overrides model_module.FEATURES so the
              model trains and predicts on this subset. Restored on exit.
    """
    orig_features = model_module.FEATURES
    if features is not None:
        model_module.FEATURES = features
    try:
        active_features = model_module.FEATURES
        model = None
        last_refit_date = None
        results = []

        for i in range(min_train_rows, len(city_df)):
            row = city_df.iloc[i]
            date = pd.Timestamp(row["date"])

            if date < start_ts or date > end_ts:
                continue
            if pd.isna(row.get("y_tmax")):
                continue
            if city_df.iloc[[i]][active_features].isnull().any(axis=1).iloc[0]:
                continue

            # Refit when needed — apply per-model minimum cadence floor
            effective_cadence = max(refit_every, MODEL_REFIT_EVERY.get(model_name, 1))
            needs_refit = (
                model is None
                or last_refit_date is None
                or (date - last_refit_date).days >= effective_cadence
            )
            if needs_refit:
                lb = lookback if lookback > 0 else i
                train_df = city_df.iloc[max(0, i - lb):i]
                m = getattr(model_module, MODEL_REGISTRY[model_name])()
                try:
                    m.fit(train_df)
                except (ValueError, RuntimeError):
                    continue
                model = m
                last_refit_date = date

            # Predict
            try:
                mu, _epi, aleatoric = model.predict_with_uncertainty(city_df.iloc[[i]])
            except Exception:
                continue

            pred_mean = float(mu[0])
            pred_sigma = float(aleatoric[0])
            y_tmax = float(row["y_tmax"])
            forecast_high = float(row["forecast_high"]) if pd.notna(row.get("forecast_high")) else np.nan
            climo = float(row["climo_mean_doy"]) if pd.notna(row.get("climo_mean_doy")) else np.nan

            results.append({
                "date": date.date(),
                "city": row["city"],
                "y_tmax": y_tmax,
                "pred_mean": pred_mean,
                "pred_sigma": pred_sigma,
                "forecast_high": forecast_high,
                "climo_mean_doy": climo,
                "month": date.month,
            })

        return results
    finally:
        model_module.FEATURES = orig_features


# ── metrics helpers ────────────────────────────────────────────────────────────

def _metrics(df: pd.DataFrame, pred_col: str, actual_col: str = "y_tmax",
             sigma_col: str = None) -> dict:
    err = df[pred_col] - df[actual_col]
    abs_err = err.abs()
    n = len(df)
    m = {
        "n": n,
        "mae": round(abs_err.mean(), 3),
        "rmse": round(float(np.sqrt((err**2).mean())), 3),
        "bias": round(err.mean(), 3),
        "within_1": round((abs_err <= 1).mean() * 100, 1),
        "within_2": round((abs_err <= 2).mean() * 100, 1),
        "within_3": round((abs_err <= 3).mean() * 100, 1),
        "within_5": round((abs_err <= 5).mean() * 100, 1),
    }
    if sigma_col and sigma_col in df.columns:
        sigma = df[sigma_col].clip(lower=0.1)
        z = abs_err / sigma
        m["cov68"] = round((z <= 1.0).mean() * 100, 1)
        m["cov90"] = round((z <= 1.645).mean() * 100, 1)
        m["sharpness"] = round(sigma.mean(), 3)
    return m


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    hdr = f"  {'Label':<22}  {'n':>5}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>6}  "
    hdr += f"{'≤1°':>5}  {'≤2°':>5}  {'≤3°':>5}  {'≤5°':>5}"
    if any("cov68" in r for r in rows):
        hdr += f"  {'Cov68':>6}  {'Cov90':>6}  {'σ̄':>5}"
    print(hdr)
    print("  " + "─" * 68)
    for r in rows:
        line = (
            f"  {r['label']:<22}  {r['n']:>5}  {r['mae']:>6.2f}  {r['rmse']:>6.2f}"
            f"  {r['bias']:>+6.2f}  {r['within_1']:>4.0f}%  {r['within_2']:>4.0f}%"
            f"  {r['within_3']:>4.0f}%  {r['within_5']:>4.0f}%"
        )
        if "cov68" in r:
            line += f"  {r['cov68']:>5.0f}%  {r['cov90']:>5.0f}%  {r['sharpness']:>5.1f}"
        print(line)


# ── feature importance ────────────────────────────────────────────────────────

def _compute_permutation_importance(
    model_name: str,
    city_frames: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    lookback: int,
    min_train_rows: int,
    n_repeats: int = 5,
) -> tuple:
    """
    Train one model per city on pre-eval data, then measure how much each
    feature's permutation raises MAE on the eval window.

    Returns
    -------
    summary : pd.DataFrame   columns: feature, imp_mean, imp_std, imp_min, imp_max
    city_data : dict         city → {"model": m, "eval_df": df, "train_df": df}
                             for re-use in pruned comparison
    """
    features = model_module.FEATURES
    rng = np.random.default_rng(42)
    records = []
    city_data = {}

    live_path = OUT_DIR / f"importance_{model_name}_live.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    live_path.unlink(missing_ok=True)   # start fresh each run

    for city, city_df in city_frames.items():
        eval_mask = (
            (city_df["date"] >= start_ts)
            & (city_df["date"] <= end_ts)
            & city_df["y_tmax"].notna()
        )
        eval_df = city_df[eval_mask].dropna(subset=features + ["y_tmax"]).copy()
        if len(eval_df) < 30:
            continue

        # Train on data before eval window (up to lookback rows)
        pre_mask = city_df["date"] < start_ts
        pre_df = city_df[pre_mask].dropna(subset=features + ["y_tmax"])
        if lookback > 0:
            pre_df = pre_df.iloc[-lookback:]
        if len(pre_df) < min_train_rows:
            continue

        m = getattr(model_module, MODEL_REGISTRY[model_name])()
        try:
            m.fit(pre_df)
        except Exception:
            continue

        mu_base, _, _ = m.predict_with_uncertainty(eval_df)
        base_mae = float(np.abs(mu_base - eval_df["y_tmax"].values).mean())

        city_data[city] = {"model": m, "eval_df": eval_df, "train_df": pre_df,
                           "base_mae": base_mae}

        city_records = []
        for feat in features:
            deltas = []
            for _ in range(n_repeats):
                perm = eval_df.copy()
                perm[feat] = rng.permutation(perm[feat].values)
                try:
                    mu_p, _, _ = m.predict_with_uncertainty(perm)
                    perm_mae = float(np.abs(mu_p - eval_df["y_tmax"].values).mean())
                    deltas.append(perm_mae - base_mae)
                except Exception:
                    pass
            if deltas:
                city_records.append({
                    "city":    city,
                    "feature": feat,
                    "delta":   float(np.mean(deltas)),
                    "std":     float(np.std(deltas)),
                    "min":     float(np.min(deltas)),
                    "max":     float(np.max(deltas)),
                })

        if city_records:
            records.extend(city_records)
            city_df_out = pd.DataFrame(city_records).sort_values("delta", ascending=False)
            write_header = not live_path.exists()
            city_df_out.to_csv(live_path, mode="a", header=write_header, index=False)
            print(f"  [{city}] base_mae={base_mae:.3f}°F  "
                  f"top feature: {city_df_out.iloc[0]['feature']} "
                  f"(+{city_df_out.iloc[0]['delta']:.4f}°F)  → {live_path}")

    if not records:
        return pd.DataFrame(), city_data

    df = pd.DataFrame(records)
    summary = (
        df.groupby("feature")
        .agg(
            imp_mean=("delta", "mean"),
            imp_std=("delta", "std"),
            imp_min=("delta", "min"),
            imp_max=("delta", "max"),
            n_cities=("city", "count"),
        )
        .reset_index()
        .sort_values("imp_mean", ascending=False)
        .reset_index(drop=True)
    )
    # Preserve FEATURES ordering as a secondary sort for ties
    feat_order = {f: i for i, f in enumerate(features)}
    summary["_ord"] = summary["feature"].map(feat_order)
    summary = summary.sort_values(["imp_mean", "_ord"], ascending=[False, True]).drop(columns=["_ord"])

    summary_path = OUT_DIR / f"importance_{model_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved → {summary_path}")

    return summary, city_data


def _print_importance_table(imp_df: pd.DataFrame, model_name: str) -> list[str]:
    """
    Print ranked feature importance table. Returns list of prune-candidate feature names.
    """
    print(f"\n{'─'*72}")
    print(f"  {model_name} — Feature Importance  "
          f"(permutation ΔMae, averaged over cities, n_repeats=5)")
    print(f"{'─'*72}")
    print(f"  {'Rank':>4}  {'Feature':<26}  {'ΔMae(°F)':>9}  {'±std':>6}  {'min':>6}  {'max':>6}  {'Action'}")
    print("  " + "─" * 70)

    prune_candidates = []
    for rank, row in imp_df.iterrows():
        action = "PRUNE?" if row["imp_mean"] < PRUNE_THRESHOLD else "keep"
        if row["imp_mean"] < PRUNE_THRESHOLD:
            prune_candidates.append(row["feature"])
        print(
            f"  {rank+1:>4}  {row['feature']:<26}  {row['imp_mean']:>+9.4f}"
            f"  {row['imp_std']:>6.4f}  {row['imp_min']:>+6.3f}  {row['imp_max']:>+6.3f}"
            f"  {action}"
        )

    if prune_candidates:
        print(f"\n  Prune candidates (ΔMae < {PRUNE_THRESHOLD}°F):")
        print(f"    {', '.join(prune_candidates)}")
        pruned = [f for f in model_module.FEATURES if f not in prune_candidates]
        print(f"\n  Pruned FEATURES ({len(model_module.FEATURES)} → {len(pruned)} features):")
        print(f"    {pruned}")
    else:
        print("\n  No prune candidates — all features contribute meaningfully.")

    return prune_candidates


def _run_pruned_comparison(
    model_name: str,
    city_data: dict,
    prune_candidates: list[str],
    city_frames: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    lookback: int,
    min_train_rows: int,
    refit_every: int,
) -> None:
    """
    Re-run walk-forward with pruned FEATURES and compare MAE to original.
    Uses the full walk-forward (not one-shot) so the comparison is apples-to-apples.
    """
    pruned_features = [f for f in model_module.FEATURES if f not in prune_candidates]

    print(f"\n{'─'*72}")
    print(f"  Pruned model comparison  ({len(model_module.FEATURES)} → {len(pruned_features)} features)")
    print(f"{'─'*72}")

    orig_records, pruned_records = [], []
    for city, city_df in city_frames.items():
        orig_records.extend(_run_model(
            model_name, city_df, lookback, refit_every, min_train_rows,
            start_ts, end_ts, features=None,
        ))
        pruned_records.extend(_run_model(
            model_name, city_df, lookback, refit_every, min_train_rows,
            start_ts, end_ts, features=pruned_features,
        ))

    if not orig_records or not pruned_records:
        print("  Insufficient data for comparison.")
        return

    orig_df   = pd.DataFrame(orig_records)
    pruned_df = pd.DataFrame(pruned_records)

    orig_mae   = (orig_df["pred_mean"]   - orig_df["y_tmax"]).abs().mean()
    pruned_mae = (pruned_df["pred_mean"] - pruned_df["y_tmax"]).abs().mean()
    delta = pruned_mae - orig_mae
    verdict = "BETTER" if delta < 0 else ("WORSE" if delta > 0 else "SAME")

    print(f"  Original  MAE: {orig_mae:.4f}°F  ({len(orig_df)} preds)")
    print(f"  Pruned    MAE: {pruned_mae:.4f}°F  ({len(pruned_df)} preds)")
    print(f"  Delta:    {delta:+.4f}°F  → pruned is {verdict}")

    # Per-city breakdown
    print(f"\n  {'City':<20}  {'Original':>10}  {'Pruned':>10}  {'Delta':>8}")
    print("  " + "─" * 52)
    for city in sorted(set(orig_df["city"])):
        og = orig_df[orig_df["city"] == city]
        pr = pruned_df[pruned_df["city"] == city]
        if og.empty or pr.empty:
            continue
        om = (og["pred_mean"] - og["y_tmax"]).abs().mean()
        pm = (pr["pred_mean"] - pr["y_tmax"]).abs().mean()
        d  = pm - om
        sign = "▼" if d < -0.01 else ("▲" if d > 0.01 else "≈")
        print(f"  {city:<20}  {om:>9.3f}°  {pm:>9.3f}°  {d:>+7.3f}° {sign}")

    if delta < 0:
        print(f"\n  ✓ Recommend pruning — update FEATURES in src/model.py")
        print(f"  Dropped: {prune_candidates}")
    elif delta > 0.02:
        print(f"\n  ✗ Pruning hurts — keep all features")
    else:
        print(f"\n  ~ Negligible difference — pruning is safe if you want fewer features")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="python scripts/forecast_backtest.py",
        description="Walk-forward forecast accuracy evaluation (no market prices).",
    )
    p.add_argument("--model", default="BayesianRidge", choices=sorted(MODEL_REGISTRY.keys()),
                   help="Model to evaluate. [BayesianRidge]")
    p.add_argument("--all-models", action="store_true",
                   help="Evaluate every model in MODEL_REGISTRY.")
    p.add_argument("--start", default="2023-01-01", metavar="YYYY-MM-DD",
                   help="Evaluation window start. [2023-01-01]")
    p.add_argument("--end", default="2025-12-31", metavar="YYYY-MM-DD",
                   help="Evaluation window end.   [2025-12-31]")
    p.add_argument("--lookback", type=int, default=730,
                   help="Training rows per city (0 = all). [730]")
    p.add_argument("--refit-every", type=int, default=1, metavar="DAYS",
                   help="Refit cadence in calendar days. [1]")
    p.add_argument("--min-train-rows", type=int, default=365,
                   help="Minimum training rows before first prediction. [365]")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip data quality validation.")
    p.add_argument("--save", action="store_true",
                   help="Save full predictions CSV to logs/forecast_backtest/.")
    p.add_argument("--feature-importance", action="store_true",
                   help="Compute permutation feature importance after the walk-forward.")
    p.add_argument("--prune", action="store_true",
                   help="Compute importance, then re-run with low-impact features removed "
                        "and compare MAE. Implies --feature-importance.")
    p.add_argument("--importance-repeats", type=int, default=5, metavar="N",
                   help="Permutation repeats per feature (higher = more stable). [5]")
    args = p.parse_args()

    if args.prune:
        args.feature_importance = True

    models_to_run = sorted(MODEL_REGISTRY.keys()) if args.all_models else [args.model]

    # ── load data ──────────────────────────────────────────────────────────────
    cfg = BacktestConfig(
        model_name=models_to_run[0],
        start_date=args.start,
        end_date=args.end,
        lookback=args.lookback,
        refit_every=args.refit_every,
    )
    dataset = DataLoader().load(cfg, skip_validation=args.no_validate)
    city_frames = dataset.city_frames

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── walk-forward per model ─────────────────────────────────────────────────
    all_rows: dict[str, pd.DataFrame] = {}   # model_name → predictions df

    for model_name in models_to_run:
        print(f"\nEvaluating {model_name} ({args.start} → {args.end})...")
        records = []
        for city, city_df in city_frames.items():
            city_records = _run_model(
                model_name, city_df,
                lookback=args.lookback,
                refit_every=args.refit_every,
                min_train_rows=args.min_train_rows,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            records.extend(city_records)
            n = len(city_records)
            if n > 0:
                m = _metrics(pd.DataFrame(city_records), "pred_mean", sigma_col="pred_sigma")
                print(f"  {city:<18}  n={n:4d}  MAE={m['mae']:.2f}°F  "
                      f"bias={m['bias']:+.2f}°F  cov68={m.get('cov68', '—')}%")

        if not records:
            print(f"  No predictions generated — check date range and data coverage.")
            continue

        df = pd.DataFrame(records)
        all_rows[model_name] = df

        # ── per-model summary (printed + saved immediately) ────────────────────
        overall_rows = []
        m_model = _metrics(df, "pred_mean", sigma_col="pred_sigma")
        m_model["label"] = f"{model_name} (model)"
        overall_rows.append(m_model)
        nws_df = df.dropna(subset=["forecast_high"])
        if not nws_df.empty:
            m_nws = _metrics(nws_df, "forecast_high")
            m_nws["label"] = "NWS/GFS forecast"
            overall_rows.append(m_nws)
        climo_df = df.dropna(subset=["climo_mean_doy"])
        if not climo_df.empty:
            m_climo = _metrics(climo_df, "climo_mean_doy")
            m_climo["label"] = "Climatology"
            overall_rows.append(m_climo)
        _print_table(overall_rows, f"{model_name} — Overall accuracy vs baselines")

        city_summary = []
        for city, g in df.groupby("city"):
            m = _metrics(g, "pred_mean", sigma_col="pred_sigma")
            m["label"] = city
            city_summary.append(m)
        city_summary.sort(key=lambda r: r["mae"])
        _print_table(city_summary, f"{model_name} — Per-city (sorted by MAE)")

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_rows = []
        for month, g in df.groupby("month"):
            m = _metrics(g, "pred_mean", sigma_col="pred_sigma")
            m["label"] = month_names[int(month) - 1]
            month_rows.append(m)
        _print_table(month_rows, f"{model_name} — Per-month")

        if args.save:
            out_path = OUT_DIR / f"{model_name}_{args.start}_{args.end}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Predictions saved → {out_path}")

        # ── feature importance immediately after this model's walk-forward ─────
        if getattr(args, "feature_importance", False):
            print(f"\nComputing permutation importance for {model_name} "
                  f"(n_repeats={args.importance_repeats})...")
            imp_df, city_data = _compute_permutation_importance(
                model_name=model_name,
                city_frames=city_frames,
                start_ts=start_ts,
                end_ts=end_ts,
                lookback=args.lookback,
                min_train_rows=args.min_train_rows,
                n_repeats=args.importance_repeats,
            )
            if imp_df.empty:
                print("  Could not compute importance — insufficient data.")
            else:
                prune_candidates = _print_importance_table(imp_df, model_name)
                if args.prune and prune_candidates:
                    _run_pruned_comparison(
                        model_name=model_name,
                        city_data=city_data,
                        prune_candidates=prune_candidates,
                        city_frames=city_frames,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        lookback=args.lookback,
                        min_train_rows=args.min_train_rows,
                        refit_every=args.refit_every,
                    )
                elif args.prune:
                    print("\n  No prune candidates — nothing to compare.")

    if not all_rows:
        print("\nNo results to summarise.")
        return

    # ── cross-model leaderboard (if multiple models ran) ──────────────────────
    if len(all_rows) > 1:
        leaderboard = []
        for model_name, df in all_rows.items():
            m = _metrics(df, "pred_mean", sigma_col="pred_sigma")
            m["label"] = model_name
            leaderboard.append(m)
        leaderboard.sort(key=lambda r: r["mae"])
        _print_table(leaderboard, "Cross-model leaderboard (sorted by MAE)")

    print()


if __name__ == "__main__":
    main()
