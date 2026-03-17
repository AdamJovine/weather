"""
KalshiPriceStore: load, validate, and index historical Kalshi price data.

The store is built once from the DB tables kalshi_candles + kalshi_markets
and provides O(1) lookup of the pre-settlement hourly sessions available
for any (city, settlement_date).

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

from src.db import DB_PATH, get_db


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
    volume: float = 0.0         # contracts traded in this candle period

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

    ts is the candle's end_period_ts (Unix seconds UTC) — as stored
    in the kalshi_candles DB table.
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
        store = KalshiPriceStore().load()
        # then pass store to BacktestEngine

    Lookup:
        sessions = store.sessions_for("Chicago", date(2024, 6, 15))
        # → list[Session] ordered by ts ascending, pre-settlement only
        # → empty list if no data for that (city, date)
    """

    PRICE_MIN = 0.01
    PRICE_MAX = 0.99
    MIN_VOLUME = 10  # skip candles with fewer than this many contracts traded

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._index: dict[tuple[str, date], list[Session]] = {}
        self._loaded = False

        # diagnostics (populated by load)
        self._n_raw_rows: int = 0
        self._n_tickers: int = 0
        self._n_indexed_pairs: int = 0
        self._n_skipped_parse: int = 0
        self._n_skipped_price: int = 0
        self._n_skipped_volume: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def load(self, verbose: bool = False) -> "KalshiPriceStore":
        """
        Load Kalshi price history from the DB (kalshi_candles + kalshi_markets),
        validate it, and build the index. Returns self for chaining.
        """
        from src.strategy import parse_contract

        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._db_path}.\n"
                "  Run scripts/update_data.py first."
            )

        print(f"  Loading Kalshi price history from {self._db_path}...")
        with get_db(self._db_path) as conn:
            df = pd.read_sql(
                """
                SELECT c.ticker, m.title, m.city, m.settlement_date,
                       c.ts, c.close_dollars, c.volume
                FROM kalshi_candles c
                JOIN kalshi_markets m ON c.ticker = m.ticker
                """,
                conn,
            )
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
        required = {"ticker", "title", "city", "settlement_date", "ts", "close_dollars", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Kalshi price data is missing columns: {missing}\n"
                "  Re-run:  python scripts/update_data.py --only kalshi"
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

        # Volume filter — thin candles have unreliable close prices
        n_low_vol = int((df["volume"] < self.MIN_VOLUME).sum())
        if n_low_vol:
            self._n_skipped_volume = n_low_vol
            df = df[df["volume"] >= self.MIN_VOLUME].copy()

        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

        # Snapshot date: convert UTC timestamp → local city date so that evening
        # pre-settlement candles (e.g. 7 PM CT on Dec 31 = 1 AM UTC Jan 1) are
        # not incorrectly excluded by a UTC-date comparison.
        city_tz = self._load_city_timezones()
        df["_tz"] = df["city"].map(city_tz).fillna("UTC")
        snapshot_local: pd.Series = pd.Series(index=df.index, dtype="object")
        for tz_str, grp in df.groupby("_tz"):
            snapshot_local[grp.index] = (
                pd.to_datetime(grp["ts"], unit="s", utc=True)
                .dt.tz_convert(tz_str)
                .dt.date
            )
        df["snapshot_date"] = snapshot_local
        df = df.drop(columns=["_tz"])

        # Keep only pre-settlement candles (the actual trading window)
        df = df[df["snapshot_date"] < df["settlement_date"]].copy()
        self._n_tickers = df["ticker"].nunique()
        return df

    def _load_city_timezones(self) -> dict[str, str]:
        """Load city→timezone from stations.csv; fall back to hardcoded map."""
        stations_path = Path(__file__).resolve().parent.parent / "data" / "stations.csv"
        if stations_path.exists():
            try:
                stations = pd.read_csv(stations_path)
                if {"city", "timezone"}.issubset(stations.columns):
                    return stations.set_index("city")["timezone"].to_dict()
            except Exception:
                pass
        return {
            "New York":      "America/New_York",
            "Chicago":       "America/Chicago",
            "Phoenix":       "America/Phoenix",
            "Miami":         "America/New_York",
            "Denver":        "America/Denver",
            "Atlanta":       "America/New_York",
            "Los Angeles":   "America/Los_Angeles",
            "Houston":       "America/Chicago",
            "Austin":        "America/Chicago",
            "Philadelphia":  "America/New_York",
            "Washington DC": "America/New_York",
            "Las Vegas":     "America/Los_Angeles",
            "Oklahoma City": "America/Chicago",
            "San Francisco": "America/Los_Angeles",
            "Seattle":       "America/Los_Angeles",
            "Dallas":        "America/Chicago",
            "New Orleans":   "America/Chicago",
            "Boston":        "America/New_York",
            "Minneapolis":   "America/Chicago",
            "San Antonio":   "America/Chicago",
        }

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
                    volume=float(row["volume"]),
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
            f"{self._n_skipped_parse} skipped (unparseable title) | "
            f"{self._n_skipped_volume} skipped (volume < {self.MIN_VOLUME})"
        )
