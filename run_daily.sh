#!/bin/bash
# Daily live trading run.
# Cron schedule (7:00am ET = 12:00 UTC, adjust for your timezone):
#   0 12 * * * /Users/adamjovine/Documents/weather/run_daily.sh >> /Users/adamjovine/Documents/weather/logs/cron.log 2>&1

set -e
cd /Users/adamjovine/Documents/weather

PYTHON="$(which python3)"
LOG_DIR="logs"
DATE="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "========================================"
echo "  Weather bot run: $DATE"
echo "========================================"

# --- Refresh NOAA history (NOAA data lags ~2 days; run daily to stay current) ---
echo "[1/3] Refreshing NOAA historical data..."
$PYTHON scripts/download_history.py || echo "  WARNING: history download failed, using cached data"

# --- Run live bot ---
echo "[2/3] Running live bot (--live)..."
$PYTHON run_live.py --live

echo "[3/3] Done."
