"""Multi-horizon ensemble: combine Donchian models across lookback periods."""

from __future__ import annotations

import pandas as pd

import config
from strategy.signals import generate_signals
from strategy.sizing import apply_rebalance_threshold, compute_weight


def ensemble_weights(
    close: pd.Series,
    lookback_periods: list[int] = config.LOOKBACK_PERIODS,
    mode: str = "long_only",
    vol_target: float = config.VOL_TARGET,
    vol_window: int = config.VOL_WINDOW,
    max_leverage: float = config.MAX_LEVERAGE,
    rebalance_threshold: float = config.REBALANCE_THRESHOLD,
) -> pd.Series:
    """Return the equal-weighted average of all single-period weights.

    For each period in *lookback_periods*:

    1. Generate signals (entry/exit state machine).
    2. Compute volatility-targeted weight.
    3. Apply the rebalance threshold.

    The final combo weight is the mean of all individual weights.

    Parameters
    ----------
    close:
        Daily close price series.
    lookback_periods:
        List of Donchian lookback windows (days).
    mode:
        ``"long_only"`` or ``"long_short"``.
    vol_target, vol_window, max_leverage, rebalance_threshold:
        Passed through to :func:`~strategy.sizing.compute_weight` and
        :func:`~strategy.sizing.apply_rebalance_threshold`.

    Returns
    -------
    pd.Series
        Combined (combo) weight series, same index as *close*.
    """
    all_weights: list[pd.Series] = []

    for period in lookback_periods:
        signals = generate_signals(close, period, mode=mode)
        raw_weight = compute_weight(
            close,
            signals["position"],
            vol_target=vol_target,
            vol_window=vol_window,
            max_leverage=max_leverage,
        )
        actual_weight = apply_rebalance_threshold(raw_weight, threshold=rebalance_threshold)
        all_weights.append(actual_weight)

    combo = pd.concat(all_weights, axis=1).mean(axis=1)
    combo.name = "combo_weight"
    return combo
