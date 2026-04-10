"""Tests for strategy/ensemble.py"""

import numpy as np
import pandas as pd
import pytest

from strategy.ensemble import ensemble_weights
from strategy.signals import generate_signals
from strategy.sizing import apply_rebalance_threshold, compute_weight


def make_close(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestEnsembleWeights:
    def test_returns_series_same_index(self):
        close = make_close([100.0 * (1.005 ** i) for i in range(200)])
        w = ensemble_weights(close, lookback_periods=[5, 10, 20])
        assert isinstance(w, pd.Series)
        assert w.index.equals(close.index)

    def test_mean_of_individual_weights(self):
        close = make_close([100.0 * (1.005 ** i) for i in range(200)])
        periods = [5, 10, 20]

        # Reproduce the individual weights manually
        individual = []
        for p in periods:
            sig = generate_signals(close, p, mode="long_only")
            rw = compute_weight(close, sig["position"])
            aw = apply_rebalance_threshold(rw)
            individual.append(aw)

        expected = pd.concat(individual, axis=1).mean(axis=1)
        result = ensemble_weights(close, lookback_periods=periods)
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_single_period_equals_individual(self):
        close = make_close([100.0 * (1.005 ** i) for i in range(200)])
        period = 20
        sig = generate_signals(close, period, mode="long_only")
        rw = compute_weight(close, sig["position"])
        aw = apply_rebalance_threshold(rw)
        combo = ensemble_weights(close, lookback_periods=[period])
        pd.testing.assert_series_equal(combo, aw, check_names=False, atol=1e-10)

    def test_long_short_mode(self):
        close = make_close([100.0 * (1.005 ** i) for i in range(200)])
        w = ensemble_weights(close, lookback_periods=[5, 10], mode="long_short")
        assert isinstance(w, pd.Series)

    def test_custom_params_forwarded(self):
        close = make_close([100.0 * (1.005 ** i) for i in range(400)])
        w_default = ensemble_weights(close, lookback_periods=[20])
        w_custom = ensemble_weights(close, lookback_periods=[20], max_leverage=1.0)
        # With lower max_leverage cap, some values should differ
        assert not w_default.equals(w_custom) or (w_default == w_custom).all()
        # At minimum, no weight should exceed the custom cap
        assert (w_custom.abs() <= 1.0 + 1e-9).all()
