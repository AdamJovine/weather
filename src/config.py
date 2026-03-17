from dotenv import load_dotenv
import os

load_dotenv()

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

# Trading parameters
TRADE_FEE_BUFFER = 0.02       # estimated fee cost per trade
TRADE_MODEL_BUFFER = 0.03     # minimum model edge above fee buffer before entering
MAX_BET_FRACTION = 0.05       # max fraction of bankroll per bet
KELLY_FRACTION = 0.33         # fraction of Kelly criterion to use

# Backtest / model parameters
TRAIN_START = "2018-01-01"
TRAIN_END = "2025-12-31"
RIDGE_ALPHA = 1.0

# Temperature grid for discrete probability model
TEMP_GRID_MIN = -20
TEMP_GRID_MAX = 129
