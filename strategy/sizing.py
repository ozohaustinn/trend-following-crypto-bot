"""Volatility-targeted position sizing with rebalance threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def compute_weight(
    close: pd.Series,
    position: pd.Series,
    vol_target: float = config.VOL_TARGET,
    vol_window: int = config.VOL_WINDOW,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.Series:
    """Compute daily position weight using volatility targeting.

    .. math::

        w_t = \\min\\!\\left(\\frac{\\text{vol\\_target}}{\\sigma_t},\\,
        \\text{max\\_leverage}\\right) \\cdot p_t

    where :math:`\\sigma_t` is the *vol_window*-day rolling annualized
    standard deviation of daily log-returns (annualized with ``sqrt(365)``).

    Parameters
    ----------
    close:
        Daily close price series.
    position:
        Signal position series (1, 0, -1).
    vol_target:
        Annualized volatility target (default 0.25).
    vol_window:
        Rolling window for volatility estimation (default 90 days).
    max_leverage:
        Maximum absolute position weight (default 2.0).

    Returns
    -------
    pd.Series
        Position weight series (same index as *close*).
    """
    log_returns = np.log(close / close.shift(1))
    sigma = log_returns.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(365)

    raw_weight = np.where(
        sigma > 0,
        np.minimum(vol_target / sigma, max_leverage),
        0.0,
    )
    weight = pd.Series(raw_weight * position.values, index=close.index)
    # When position is 0, weight is 0 regardless of vol
    weight = weight.where(position != 0, other=0.0)
    return weight


def apply_rebalance_threshold(
    target_weight: pd.Series,
    threshold: float = config.REBALANCE_THRESHOLD,
) -> pd.Series:
    """Apply a rebalance threshold to suppress small volatility-driven changes.

    Signal-driven changes (position flips: 0→non-zero, non-zero→0, or sign
    change) always execute immediately.  Only *within* a continuous position,
    volatility-driven weight adjustments are subject to the threshold.

    Parameters
    ----------
    target_weight:
        Raw target weight series from :func:`compute_weight`.
    threshold:
        Fractional threshold relative to the current actual weight.
        A change is executed only when
        ``|target - actual| / |actual| > threshold`` (or when
        ``actual == 0`` and a new position is opened).

    Returns
    -------
    pd.Series
        Actual (realised) weight series after applying the threshold.
    """
    actual = np.zeros(len(target_weight))
    target_arr = target_weight.values

    for i in range(len(target_arr)):
        t = target_arr[i]
        prev = actual[i - 1] if i > 0 else 0.0

        # Detect signal-driven change: crossing zero (entry or exit)
        if (prev == 0.0 and t != 0.0) or (prev != 0.0 and t == 0.0):
            actual[i] = t
        elif np.sign(t) != np.sign(prev) and t != 0.0:
            # Direction flip (long_short mode)
            actual[i] = t
        elif prev == 0.0 and t == 0.0:
            actual[i] = 0.0
        else:
            # Both non-zero, same sign — apply threshold
            delta = abs(t - prev)
            if delta / abs(prev) > threshold:
                actual[i] = t
            else:
                actual[i] = prev

    return pd.Series(actual, index=target_weight.index)
