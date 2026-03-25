"""
Data collector for the EMOS weather forecasting system.

Polls METAR observations and fetches LAMP/NBM/GFS MOS forecasts from IEM
on configurable schedules, storing everything in a local SQLite database.
"""

from pathlib import Path

from dotenv import load_dotenv

# Walk up from collector/ to find .env at the repo root
_env_file = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_file)
