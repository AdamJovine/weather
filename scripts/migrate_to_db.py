"""
One-time migration: copy all existing CSV files into data/weather.db.

Safe to re-run — INSERT OR REPLACE deduplicates on primary key.

Run from project root:
  python scripts/migrate_to_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_db, migrate_from_csvs, DB_PATH


def main():
    print(f"Migrating CSV data → {DB_PATH}")
    with get_db() as conn:
        migrate_from_csvs(conn)
    print(f"\nDone. Database: {DB_PATH}  ({DB_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
