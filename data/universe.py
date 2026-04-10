"""Monthly universe selection — top N cryptocurrencies by median daily volume."""

from __future__ import annotations

import pandas as pd

import config

EXCLUDED_SYMBOLS = [
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FRAX", "LUSD",
    "WBTC", "WETH", "STETH", "WBNB", "CBETH", "RETH",
]


def _base_symbol(symbol: str) -> str:
    """Return the base currency from a trading pair like ``"BTC/USDT"``."""
    return symbol.split("/")[0].upper()


def select_universe(
    all_data: dict[str, pd.DataFrame],
    as_of_date: str,
    n_assets: int = config.UNIVERSE_SIZE,
) -> list[str]:
    """Return the top *n_assets* symbols by trailing 30-day median volume.

    Parameters
    ----------
    all_data:
        Mapping ``{symbol: ohlcv_df}`` for all candidate assets.
    as_of_date:
        End-of-month date used as the look-back reference (``"YYYY-MM-DD"``).
    n_assets:
        How many assets to include (default from ``config.UNIVERSE_SIZE``).

    Returns
    -------
    list[str]
        Selected symbol list, sorted by descending median volume.
    """
    cutoff = pd.Timestamp(as_of_date)
    lookback_start = cutoff - pd.Timedelta(days=30)

    scores: dict[str, float] = {}
    for symbol, df in all_data.items():
        if _base_symbol(symbol) in EXCLUDED_SYMBOLS:
            continue
        if df.empty:
            continue

        # Minimum listing age filter
        listing_days = (cutoff - df.index.min()).days
        if listing_days < config.MIN_LISTING_DAYS:
            continue

        window = df.loc[lookback_start:cutoff, "volume"]
        if window.empty:
            continue
        median_vol = window.median()
        if median_vol < config.MIN_MEDIAN_VOLUME:
            continue

        scores[symbol] = median_vol

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    return ranked[:n_assets]


def check_exit_conditions(df: pd.DataFrame, as_of_date: str) -> bool:
    """Return *True* if the asset should be removed mid-month.

    Removal conditions (either one is sufficient):
    - Median daily volume over the past 30 days falls below $1 M.
    - Median daily |price change| over the past 30 days is less than 0.5 %.

    Parameters
    ----------
    df:
        OHLCV DataFrame with DatetimeIndex for a single asset.
    as_of_date:
        Reference date for the 30-day look-back window.
    """
    cutoff = pd.Timestamp(as_of_date)
    lookback_start = cutoff - pd.Timedelta(days=30)
    window = df.loc[lookback_start:cutoff]

    if window.empty:
        return True

    median_vol = window["volume"].median()
    if median_vol < config.EXIT_MIN_VOLUME:
        return True

    price_changes = window["close"].pct_change().abs()
    if price_changes.median() < config.EXIT_MIN_PRICE_CHANGE:
        return True

    return False
