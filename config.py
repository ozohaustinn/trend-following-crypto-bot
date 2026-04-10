# Donchian lookback periods (days)
LOOKBACK_PERIODS = [5, 10, 20, 30, 60, 90, 150, 250, 360]

# Volatility targeting
VOL_TARGET = 0.25              # 25% annualized target volatility
VOL_WINDOW = 90                # 90-day rolling window for volatility estimation
MAX_LEVERAGE = 2.0             # Cap position weight at 200%
REBALANCE_THRESHOLD = 0.20     # Only rebalance vol-sizing if delta > 20%

# Universe selection
UNIVERSE_SIZE = 20             # Number of assets in portfolio
MIN_LISTING_DAYS = 365         # Minimum days since listing
MIN_MEDIAN_VOLUME = 2_000_000  # $2M median daily volume over 30 days
EXIT_MIN_VOLUME = 1_000_000    # $1M exit threshold
EXIT_MIN_PRICE_CHANGE = 0.005  # 0.5% minimum median daily price change

# Transaction costs
TRANSACTION_COST_BPS = 10      # 10 basis points per trade (one way)

# Backtest defaults
DEFAULT_CAPITAL = 100_000
DEFAULT_START = "2017-01-01"
DEFAULT_END = "2025-01-01"
DEFAULT_EXCHANGE = "binance"
