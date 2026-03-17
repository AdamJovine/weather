"""
Download monthly AO and ENSO (ONI) climate regime indices from NOAA CPC.

AO  (Arctic Oscillation): positive = mild/zonal flow; negative = cold outbreaks.
ONI (Oceanic Niño Index):  positive = El Niño (warm); negative = La Niña (cool).

Both are monthly, so they are joined to the feature table via (year, month).

Run from project root:
  python scripts/download_climate_indices.py

Output: data/climate_indices.csv  — columns: year, month, ao_index, oni
"""

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import numpy as np
import pandas as pd

AO_URL  = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
OUTPUT  = Path("data/climate_indices.csv")

# ONI 3-month season → center month index (1=Jan … 12=Dec)
ONI_SEASON_MAP = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def fetch_ao() -> pd.DataFrame:
    """
    Parse the NOAA CPC AO wide table (year × 12 months) into long form.
    Returns DataFrame with columns: year, month, ao_index.
    """
    resp = requests.get(AO_URL, timeout=30)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if l.strip()]

    rows = []
    for line in lines:
        parts = line.split()
        # Header row starts with a non-numeric token
        if not parts[0].lstrip("-").isdigit():
            continue
        year = int(parts[0])
        for month_idx, val in enumerate(parts[1:13], start=1):
            try:
                v = float(val)
            except ValueError:
                v = np.nan
            # NOAA uses 99.9 or similar sentinels for missing
            if abs(v) > 10:
                v = np.nan
            rows.append({"year": year, "month": month_idx, "ao_index": v})

    df = pd.DataFrame(rows)
    print(f"AO: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_nao() -> pd.DataFrame:
    """
    Parse the NOAA CPC NAO wide table (year × 12 months) into long form.
    Same format as AO. Returns DataFrame with columns: year, month, nao_index.
    """
    resp = requests.get(NAO_URL, timeout=30)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if l.strip()]

    rows = []
    for line in lines:
        parts = line.split()
        if not parts[0].lstrip("-").isdigit():
            continue
        year = int(parts[0])
        for month_idx, val in enumerate(parts[1:13], start=1):
            try:
                v = float(val)
            except ValueError:
                v = np.nan
            if abs(v) > 10:
                v = np.nan
            rows.append({"year": year, "month": month_idx, "nao_index": v})

    df = pd.DataFrame(rows)
    print(f"NAO: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def fetch_oni() -> pd.DataFrame:
    """
    Parse the NOAA CPC ONI rolling-season table into monthly long form.
    Each season's ANOM is assigned to the center month.
    Returns DataFrame with columns: year, month, oni.
    """
    resp = requests.get(ONI_URL, timeout=30)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if l.strip()]

    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        seas, yr_str, _total, anom_str = parts[0], parts[1], parts[2], parts[3]
        if seas not in ONI_SEASON_MAP:
            continue
        try:
            year  = int(yr_str)
            anom  = float(anom_str)
        except ValueError:
            continue
        month = ONI_SEASON_MAP[seas]
        rows.append({"year": year, "month": month, "oni": anom})

    df = pd.DataFrame(rows).drop_duplicates(subset=["year", "month"])
    print(f"ONI: {len(df)} rows  ({df['year'].min()}–{df['year'].max()})")
    return df


def main():
    print("Fetching AO index...")
    ao_df = fetch_ao()

    print("Fetching NAO index...")
    nao_df = fetch_nao()

    print("Fetching ONI (ENSO) index...")
    oni_df = fetch_oni()

    combined = ao_df.merge(nao_df, on=["year", "month"], how="outer")
    combined = combined.merge(oni_df, on=["year", "month"], how="outer")
    combined = combined.sort_values(["year", "month"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT}")
    print(combined.tail(14).to_string(index=False))


if __name__ == "__main__":
    main()
