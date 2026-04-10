"""Donchian Channel computation."""

from __future__ import annotations

import pandas as pd


def donchian_channel(close: pd.Series, period: int) -> pd.DataFrame:
    """Return a DataFrame with Donchian Channel bands.

    Parameters
    ----------
    close:
        Daily close price series with a DatetimeIndex.
    period:
        Lookback window in days.

    Returns
    -------
    pd.DataFrame
        Columns: ``upper``, ``lower``, ``mid``.

        - ``upper`` = rolling max of *close* over *period* bars.
        - ``lower`` = rolling min of *close* over *period* bars.
        - ``mid``   = ``(upper + lower) / 2``.
    """
    upper = close.rolling(window=period, min_periods=period).max()
    lower = close.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid}, index=close.index)
