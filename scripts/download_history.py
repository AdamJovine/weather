"""
Phase 1: Download NOAA GHCN-Daily TMAX for all stations in data/stations.csv.
Saves to data/historical_tmax.csv.

Incremental: only downloads date ranges not yet present for each city.
Safe to re-run — existing data is preserved, gaps are filled.

Run from project root:
  python scripts/download_history.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.noaa_history import get_daily_tmax
from src.config import TRAIN_START, TRAIN_END

STATIONS_FILE = Path("data/stations.csv")
OUTPUT_FILE = Path("data/historical_tmax.csv")


def missing_years(city: str, existing: pd.DataFrame, start: str, end: str) -> list[tuple]:
    """
    Return list of (year_start, year_end) tuples that have fewer than
    300 rows in existing data (treats them as missing or incomplete).
    """
    target_start = pd.to_datetime(start).year
    target_end = pd.to_datetime(end).year

    city_data = existing[existing["city"] == city].copy()
    city_data["date"] = pd.to_datetime(city_data["date"])
    city_data["year"] = city_data["date"].dt.year

    rows_per_year = city_data.groupby("year").size()

    gaps = []
    for year in range(target_start, target_end + 1):
        count = rows_per_year.get(year, 0)
        if count < 300:   # less than ~10 months of data → treat as gap
            gaps.append((f"{year}-01-01", f"{year}-12-31"))
    return gaps


def main():
    stations = pd.read_csv(STATIONS_FILE)

    # Load existing data (if any)
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        print(f"Existing data: {len(existing)} rows for {existing['city'].nunique()} cities.")
    else:
        existing = pd.DataFrame(columns=["date", "station", "tmax", "city"])

    new_frames = []

    for _, row in stations.iterrows():
        city = row["city"]
        station_id = row["station_id"]
        gaps = missing_years(city, existing, TRAIN_START, TRAIN_END)

        if not gaps:
            print(f"  {city}: up to date, skipping.")
            continue

        print(f"  {city}: downloading {len(gaps)} missing year(s)...")
        for gap_start, gap_end in gaps:
            print(f"    {gap_start} → {gap_end}", end=" ", flush=True)
            try:
                df = get_daily_tmax(
                    station_id=station_id,
                    start_date=gap_start,
                    end_date=gap_end,
                )
                df["city"] = city
                new_frames.append(df)
                print(f"→ {len(df)} rows")
            except Exception as e:
                print(f"→ ERROR: {e}")

    if not new_frames:
        print("Nothing new to download.")
        return

    new_data = pd.concat(new_frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "city"]).sort_values(["city", "date"])
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(combined)} rows to {OUTPUT_FILE}")
    for city, g in combined.groupby("city"):
        g["date"] = pd.to_datetime(g["date"])
        print(f"  {city:12s}: {g.date.min().date()} → {g.date.max().date()}  ({len(g)} rows)")


if __name__ == "__main__":
    main()
