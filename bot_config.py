"""Bot-specific configuration for the live trading module."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Operating mode (overridden by CLI --mode flag)
DEFAULT_MODE = "dry"

# API keys (loaded from .env, NEVER hardcoded)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Exchange settings
FUTURES_MODE = True                   # Use Binance Futures (USDT-M)
DEFAULT_MARGIN_TYPE = "cross"         # "cross" or "isolated"

# Risk limits
EQUITY_FLOOR = 100.0                  # Stop trading if equity drops below this ($)
MAX_SINGLE_POSITION = 2.0            # Max weight per asset (200%)
MAX_TOTAL_LEVERAGE = 5.0             # Max sum of absolute weights (500%)
MAX_SINGLE_ORDER_PCT = 0.50          # Max single order as fraction of equity
MIN_ORDER_VALUE = 10.0               # Minimum order value in USDT

# Execution
ORDER_TYPE = "market"                 # "market" or "limit"
MIN_REBALANCE_DELTA = 0.01           # Ignore weight deltas smaller than 1%
SELL_BEFORE_BUY = True               # Process sells first to free margin

# Scheduling
EXECUTION_HOUR_UTC = 0               # Hour of day to run (0 = midnight UTC)
EXECUTION_MINUTE_UTC = 5             # Minute (5 = 00:05 UTC)

# Universe
CANDIDATE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "SOL/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
    "LINK/USDT", "AVAX/USDT", "UNI/USDT", "ATOM/USDT", "XLM/USDT",
    "ETC/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT", "TRX/USDT",
    "NEAR/USDT", "FTM/USDT", "AAVE/USDT", "SHIB/USDT", "PEPE/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "APT/USDT", "INJ/USDT",
]

# Logging
LOG_DIR = "logs"

# Notifications (optional, set in .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
