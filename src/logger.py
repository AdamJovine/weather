"""
Structured logging for forecasts, market snapshots, trade decisions, and outcomes.
All logs are appended to CSV files in logs/.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

TRADE_LOG_PATH = LOG_DIR / "trade_log.csv"
FORECAST_LOG_PATH = LOG_DIR / "forecast_log.csv"
MARKET_SNAPSHOT_PATH = LOG_DIR / "market_snapshots.csv"


def init_run(model_name: str = "") -> None:
    """Set log paths to a unique timestamped name for this run.

    Call once at process startup (before any logging) so each invocation
    of run_live.py writes to its own files instead of appending to a shared log.

    Example output:
        logs/trade_log_20260318_143000_ARD.csv
        logs/forecast_log_20260318_143000_ARD.csv
        logs/market_snapshots_20260318_143000_ARD.csv
    """
    global TRADE_LOG_PATH, FORECAST_LOG_PATH, MARKET_SNAPSHOT_PATH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ts}_{model_name}" if model_name else f"_{ts}"
    TRADE_LOG_PATH = LOG_DIR / f"trade_log{suffix}.csv"
    FORECAST_LOG_PATH = LOG_DIR / f"forecast_log{suffix}.csv"
    MARKET_SNAPSHOT_PATH = LOG_DIR / f"market_snapshots{suffix}.csv"
    print(f"Run logs: {TRADE_LOG_PATH.name}")

TRADE_LOG_FIELDS = [
    "timestamp", "market_ticker", "city", "market_type", "contract_desc",
    "fair_p", "yes_ask", "no_ask", "edge_yes", "edge_no",
    "action", "side", "size_dollars", "contract_count", "status", "notes",
]

FORECAST_LOG_FIELDS = [
    "timestamp", "city", "target_date", "forecast_high",
    "pred_mean", "sigma", "source",
]

MARKET_SNAPSHOT_FIELDS = [
    "timestamp", "market_ticker", "title", "yes_ask", "no_ask",
    "volume", "open_interest", "close_time",
]


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_trade(
    market_ticker: str,
    city: str,
    market_type: str,
    contract_desc: str,
    fair_p: float,
    yes_ask: int,
    no_ask: int,
    edge_yes: float,
    edge_no: float,
    action: str,          # "recommend" | "place" | "skip"
    side: str = "",
    size_dollars: float = 0.0,
    contract_count: int = 0,
    status: str = "pending",
    notes: str = "",
) -> None:
    _append_row(TRADE_LOG_PATH, TRADE_LOG_FIELDS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_ticker": market_ticker,
        "city": city,
        "market_type": market_type,
        "contract_desc": contract_desc,
        "fair_p": round(fair_p, 4),
        "yes_ask": yes_ask,
        "no_ask": no_ask,
        "edge_yes": round(edge_yes, 4),
        "edge_no": round(edge_no, 4),
        "action": action,
        "side": side,
        "size_dollars": round(size_dollars, 2),
        "contract_count": contract_count,
        "status": status,
        "notes": notes,
    })


def log_forecast(
    city: str,
    target_date: str,
    forecast_high: float,
    pred_mean: float,
    sigma: float,
    source: str = "nws",
) -> None:
    _append_row(FORECAST_LOG_PATH, FORECAST_LOG_FIELDS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "target_date": target_date,
        "forecast_high": round(forecast_high, 1) if forecast_high is not None else "",
        "pred_mean": round(pred_mean, 2),
        "sigma": round(sigma, 2),
        "source": source,
    })


def log_market_snapshot(
    market_ticker: str,
    title: str,
    yes_ask: int,
    no_ask: int,
    volume: int = 0,
    open_interest: int = 0,
    close_time: str = "",
) -> None:
    _append_row(MARKET_SNAPSHOT_PATH, MARKET_SNAPSHOT_FIELDS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_ticker": market_ticker,
        "title": title,
        "yes_ask": yes_ask,
        "no_ask": no_ask,
        "volume": volume,
        "open_interest": open_interest,
        "close_time": close_time,
    })
