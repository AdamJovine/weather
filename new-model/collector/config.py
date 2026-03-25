"""
Station configuration and collection schedule settings.

Stations are defined here (not CSV) so the collector is self-contained.
ICAO codes match the Kalshi settlement stations from contract PDFs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Station:
    city: str
    icao: str           # METAR station for live obs + IEM MOS queries
    lat: float
    lon: float
    timezone: str
    kalshi_series: str


# Load stations from the canonical CSV (old/data/stations.csv)
_STATIONS_CSV = Path(__file__).resolve().parents[2] / "old" / "data" / "stations.csv"


def _load_stations() -> list[Station]:
    stations = []
    with open(_STATIONS_CSV) as f:
        for row in csv.DictReader(f):
            stations.append(Station(
                city=row["city"],
                icao=row["icao_id"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                timezone=row["timezone"],
                kalshi_series=row["kalshi_series"],
            ))
    return stations


STATIONS: list[Station] = _load_stations()
ICAO_LIST: list[str] = [s.icao for s in STATIONS]
ICAO_TO_STATION: dict[str, Station] = {s.icao: s for s in STATIONS}

# ── Collection schedule (seconds) ────────────────────────────────────────────

METAR_INTERVAL = 5 * 60       # poll METAR every 5 minutes
KALSHI_INTERVAL = 5 * 60      # poll Kalshi prices every 5 minutes
LAMP_INTERVAL = 30 * 60       # fetch LAMP every 30 min (new runs hourly)
NBM_INTERVAL = 30 * 60        # fetch NBM every 30 min (new runs ~hourly)
GFS_MOS_INTERVAL = 3 * 3600   # fetch GFS MOS every 3 hours (runs 4x/day)

# ── Database ─────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "collector.db"
