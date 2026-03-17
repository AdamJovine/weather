"""
KalshiPriceStore: load, validate, and index historical Kalshi price data.

The store is built once from data/kalshi_price_history.csv (produced by
scripts/download_kalshi_history.py) and provides O(1) lookup of the
pre-settlement hourly sessions available for any (city, settlement_date).

Why load once?
  Parsing and indexing ~200k rows takes ~2–3 seconds. The engine runs
  thousands of (city, date) lookups — indexing up front avoids repeating
  that work on every lookup.

Data quality enforced at load time:
  - Required columns present
  - No duplicate (ticker, ts) rows — deduped if found
  - Prices filtered to (0.01, 0.99) — endpoints are stale/illiquid
  - Pre-settlement filter: snapshot_date < settlement_date
  - Contracts whose titles fail to parse are counted and skipped

Usage:
    store = KalshiPriceStore().load()
    sessions = store.sessions_for("New York", date(2024, 3, 17))
    for session in sessions:
        for pc in session.contracts:
            print(pc.ticker, pc.close_dollars)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ParsedContract:
    """
    One tradeable contract at a specific price snapshot.

    contract_def mirrors what parse_contract() returns:
      geq   → {"market_type": "geq", "threshold": int}
      leq   → {"market_type": "leq", "threshold": int}
      range → {"market_type": "range", "low": int, "high": int}
    """
    ticker: str
    title: str
    contract_def: dict          # immutable after creation (use dict but treat as read-only)
    close_dollars: float        # real last-traded price in (0.01, 0.99)

    @property
    def contract_type(self) -> str:
        return self.contract_def["market_type"]

    @property
    def display_threshold(self) -> int:
        """Best single integer to represent this contract for logging."""
        d = self.contract_def
        if "threshold" in d:
            return d["threshold"]
        return d.get("low", 0)


@dataclass
class Session:
    """
    One hourly price snapshot: all contracts available at timestamp ts.

    ts is the candle's end_period_ts (Unix seconds UTC) — exactly as stored
    in kalshi_price_history.csv.
    """
    ts: int
    contracts: list[ParsedContract]

    @property
    def hour_utc(self) -> int:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).hour

    @property
    def n_contracts(self) -> int:
        return len(self.contracts)


class KalshiPriceStore:
    """
    Loaded-once index of historical Kalshi price data.

    Lifecycle:
        store = KalshiPriceStore(data_dir="data").load()
        # then pass store to BacktestEngine

    Lookup:
        sessions = store.sessions_for("Chicago", date(2024, 6, 15))
        # → list[Session] ordered by ts ascending, pre-settlement only
        # → empty list if no data for that (city, date)
    """

    PRICE_FILE = "kalshi_price_history.csv"
    PRICE_MIN = 0.01
    PRICE_MAX = 0.99

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = Path(data_dir)
        self._index: dict[tuple[str, date], list[Session]] = {}
        self._loaded = False

        # diagnostics (populated by load)
        self._n_raw_rows: int = 0
        self._n_tickers: int = 0
        self._n_indexed_pairs: int = 0
        self._n_skipped_parse: int = 0
        self._n_skipped_price: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def load(self, verbose: bool = False) -> "KalshiPriceStore":
        """
        Load kalshi_price_history.csv, validate it, and build the index.
        Returns self for chaining.
        """
        from src.strategy import parse_contract

        path = self._data_dir / self.PRICE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "  Download it first:  python scripts/download_kalshi_history.py"
            )

        print(f"  Loading Kalshi price history: {path}")
        df = pd.read_csv(path)
        self._n_raw_rows = len(df)

        self._validate_schema(df)
        df = self._clean(df)
        self._build_index(df, parse_contract, verbose=verbose)

        self._loaded = True
        self._print_summary()
        return self

    def sessions_for(self, city: str, settlement_date: date) -> list[Session]:
        """
        Pre-settlement hourly sessions for (city, settlement_date),
        sorted by timestamp ascending.

        Returns an empty list if no price history exists for this pair.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before using the store.")
        return self._index.get((city, settlement_date), [])

    def has_data(self, city: str, settlement_date: date) -> bool:
        return (city, settlement_date) in self._index

    def cities(self) -> list[str]:
        return sorted({k[0] for k in self._index})

    def date_range(self, city: str) -> Optional[tuple[date, date]]:
        dates = [k[1] for k in self._index if k[0] == city]
        return (min(dates), max(dates)) if dates else None

    def coverage_summary(self) -> str:
        lines = [f"  Price store: {self._n_indexed_pairs} (city, date) pairs"]
        for city in self.cities():
            dr = self.date_range(city)
            n = sum(1 for k in self._index if k[0] == city)
            lines.append(f"    {city:<12}: {n} dates  {dr[0]} → {dr[1]}")
        return "\n".join(lines)

    # ── private ───────────────────────────────────────────────────────────────

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = {"ticker", "title", "city", "settlement_date", "ts", "close_dollars"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"kalshi_price_history.csv is missing columns: {missing}\n"
                "  Re-run:  python scripts/download_kalshi_history.py"
            )

        n_dupes = int(df.duplicated(subset=["ticker", "ts"]).sum())
        if n_dupes:
            print(f"  WARNING: {n_dupes} duplicate (ticker, ts) rows — keeping last")

        n_out = int((~df["close_dollars"].between(self.PRICE_MIN, self.PRICE_MAX)).sum())
        if n_out:
            self._n_skipped_price = n_out
            print(
                f"  WARNING: {n_out} price rows outside "
                f"[{self.PRICE_MIN}, {self.PRICE_MAX}] — filtered out"
            )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df[df["close_dollars"].between(self.PRICE_MIN, self.PRICE_MAX)]
            .drop_duplicates(subset=["ticker", "ts"], keep="last")
            .copy()
        )
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

        # Snapshot date: convert Unix ts (UTC) → date (tz-naive)
        df["snapshot_date"] = (
            pd.to_datetime(df["ts"], unit="s", utc=True)
            .dt.tz_convert(None)
            .dt.date
        )

        # Keep only pre-settlement candles (the actual trading window)
        df = df[df["snapshot_date"] < df["settlement_date"]].copy()
        self._n_tickers = df["ticker"].nunique()
        return df

    def _build_index(self, df: pd.DataFrame, parse_contract, verbose: bool) -> None:
        """
        Parse every contract title once, then group into sessions.

        Contract titles that fail to parse are silently skipped and counted.
        One (ticker, ts) pair = one ParsedContract in a session.
        """
        n_skipped = 0

        for (city, sdate), group in df.groupby(["city", "settlement_date"]):
            # ts → list of ParsedContracts for that timestamp
            sessions_map: dict[int, list[ParsedContract]] = {}

            for _, row in group.iterrows():
                try:
                    contract_def = parse_contract(str(row["title"]), "")
                except (ValueError, KeyError):
                    n_skipped += 1
                    if verbose:
                        print(f"    skipped (parse fail): {row['title']!r}")
                    continue

                pc = ParsedContract(
                    ticker=str(row["ticker"]),
                    title=str(row["title"]),
                    contract_def=contract_def,
                    close_dollars=float(row["close_dollars"]),
                )
                sessions_map.setdefault(int(row["ts"]), []).append(pc)

            if sessions_map:
                self._index[(city, sdate)] = [
                    Session(ts=ts, contracts=contracts)
                    for ts, contracts in sorted(sessions_map.items())
                ]

        self._n_skipped_parse = n_skipped
        self._n_indexed_pairs = len(self._index)

    def _print_summary(self) -> None:
        n_cities = len(self.cities())
        print(
            f"  Price store ready: {self._n_raw_rows:,} rows | "
            f"{self._n_tickers} tickers | "
            f"{self._n_indexed_pairs} (city, date) pairs | "
            f"{n_cities} cities | "
            f"{self._n_skipped_parse} contracts skipped (unparseable title)"
        )
