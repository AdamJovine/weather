from dotenv import load_dotenv
import os

from src.app_config import cfg as _cfg

load_dotenv()

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

# Trading parameters
TRADE_FEE_BUFFER = _cfg.portfolio.trade_fee_buffer
TRADE_MODEL_BUFFER = _cfg.portfolio.trade_model_buffer
MAX_BET_FRACTION = _cfg.portfolio.max_bet_fraction
KELLY_FRACTION = _cfg.portfolio.kelly_fraction
MIN_BET_DOLLARS = _cfg.portfolio.min_bet_dollars
CITY_MULTIPLIER_CAP = _cfg.portfolio.city_multiplier_cap

# Backtest / model parameters
TRAIN_START = _cfg.model.train_start
TRAIN_END = _cfg.model.train_end
RIDGE_ALPHA = _cfg.model.ridge_alpha

# Temperature grid for discrete probability model
TEMP_GRID_MIN = _cfg.model.temp_grid_min
TEMP_GRID_MAX = _cfg.model.temp_grid_max
