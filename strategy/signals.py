"""Entry/exit signal generation per asset for a single Donchian lookback period."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.donchian import donchian_channel


def generate_signals(
    close: pd.Series,
    period: int,
    mode: str = "long_only",
) -> pd.DataFrame:
    """Generate entry/exit signals for a single asset and lookback period.

    The signal state machine is computed bar-by-bar (sequential, not vectorised)
    because the trailing stop depends on the current position state.

    Parameters
    ----------
    close:
        Daily close price series.
    period:
        Donchian lookback in days.
    mode:
        ``"long_only"`` (default) or ``"long_short"``.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``position``      : 1 (long), 0 (flat), or -1 (short).
        - ``trailing_stop`` : trailing stop level (NaN when flat).
        - ``entry_price``   : price at which the current position was opened
          (NaN when flat).
    """
    if mode not in ("long_only", "long_short"):
        raise ValueError(f"mode must be 'long_only' or 'long_short', got {mode!r}")

    dc = donchian_channel(close, period)
    upper = dc["upper"].values
    lower = dc["lower"].values
    mid = dc["mid"].values
    close_arr = close.values
    n = len(close_arr)

    position = np.zeros(n, dtype=np.int8)
    trailing_stop = np.full(n, np.nan)
    entry_price = np.full(n, np.nan)

    cur_pos: int = 0
    cur_stop: float = np.nan
    cur_entry: float = np.nan

    for i in range(n):
        c = close_arr[i]
        u = upper[i]
        lo = lower[i]
        m = mid[i]

        if np.isnan(u):
            # Not enough history yet — stay flat
            position[i] = 0
            trailing_stop[i] = np.nan
            entry_price[i] = np.nan
            continue

        if cur_pos == 0:
            # --- Flat: check for entry ---
            if c == u:
                cur_pos = 1
                cur_stop = m
                cur_entry = c
            elif mode == "long_short" and c == lo:
                cur_pos = -1
                cur_stop = m
                cur_entry = c
        elif cur_pos == 1:
            # --- Long: update trailing stop, check for exit ---
            cur_stop = max(cur_stop, m)
            if c <= cur_stop:
                cur_pos = 0
                cur_stop = np.nan
                cur_entry = np.nan
        elif cur_pos == -1:
            # --- Short: update trailing stop (moves down only), check for exit ---
            cur_stop = min(cur_stop, m)
            if c >= cur_stop:
                cur_pos = 0
                cur_stop = np.nan
                cur_entry = np.nan

        position[i] = cur_pos
        trailing_stop[i] = cur_stop
        entry_price[i] = cur_entry

    return pd.DataFrame(
        {
            "position": position,
            "trailing_stop": trailing_stop,
            "entry_price": entry_price,
        },
        index=close.index,
    )
