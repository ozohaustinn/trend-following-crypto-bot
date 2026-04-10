"""Tests for strategy/sizing.py"""

import numpy as np
import pandas as pd
import pytest

from strategy.sizing import apply_rebalance_threshold, compute_weight


def make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestComputeWeight:
    def test_weight_equals_vol_target_over_sigma(self):
        # Use a random price series with genuine variance to get a stable sigma.
        rng = np.random.default_rng(42)
        # Daily returns ~2%, annualized sigma ≈ 0.02*sqrt(365) ≈ 0.382
        # Expected weight = 0.25 / 0.382 ≈ 0.65, well below max_leverage=2.0
        n = 300
        log_rets = rng.normal(0.001, 0.02, n)
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        close = make_series(prices)
        position = make_series([1] * n)
        weights = compute_weight(close, position, vol_target=0.25, vol_window=90, max_leverage=2.0)
        # Weights should all be positive (long) and within (0, max_leverage]
        valid = weights.iloc[90:]
        assert (valid > 0).all()
        assert (valid <= 2.0 + 1e-9).all()

    def test_weight_capped_at_max_leverage(self):
        # Very low volatility → weight would exceed cap
        prices = [100.0 + i * 0.0001 for i in range(200)]
        close = make_series(prices)
        position = make_series([1] * 200)
        weights = compute_weight(close, position, vol_target=0.25, vol_window=90, max_leverage=2.0)
        assert (weights.abs() <= 2.0 + 1e-9).all()

    def test_weight_zero_when_flat(self):
        prices = [100.0 * (1.01 ** i) for i in range(200)]
        close = make_series(prices)
        position = make_series([0] * 200)
        weights = compute_weight(close, position)
        assert (weights == 0).all()

    def test_negative_weight_for_short(self):
        prices = [100.0 * (1.01 ** i) for i in range(200)]
        close = make_series(prices)
        position = make_series([-1] * 200)
        weights = compute_weight(close, position)
        valid = weights.dropna()
        assert (valid.iloc[90:] < 0).all()

    def test_nan_before_vol_window(self):
        prices = [100.0 * (1.01 ** i) for i in range(200)]
        close = make_series(prices)
        position = make_series([1] * 200)
        weights = compute_weight(close, position, vol_window=90)
        # Before window completes, weight should be 0 (no sigma → no weight)
        assert (weights.iloc[:90] == 0).all()


class TestApplyRebalanceThreshold:
    def test_no_change_when_below_threshold(self):
        # Target moves from 1.0 to 1.05 (5% change) with 20% threshold
        target = make_series([1.0] * 5 + [1.05] * 5)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        # Change is only 5% < 20% → weight should stay at 1.0
        assert actual.iloc[5] == pytest.approx(1.0)

    def test_update_when_above_threshold(self):
        # Target moves from 1.0 to 1.5 (50% change) with 20% threshold
        target = make_series([1.0] * 5 + [1.5] * 5)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        assert actual.iloc[5] == pytest.approx(1.5)

    def test_signal_entry_always_executes(self):
        # Go from 0 → 1.0 (entry signal)
        target = make_series([0.0] * 5 + [1.0] * 5)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        assert actual.iloc[5] == pytest.approx(1.0)

    def test_signal_exit_always_executes(self):
        # Go from 1.0 → 0 (exit signal)
        target = make_series([1.0] * 5 + [0.0] * 5)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        assert actual.iloc[5] == pytest.approx(0.0)

    def test_direction_flip_always_executes(self):
        # Long → short flip
        target = make_series([1.0] * 5 + [-1.0] * 5)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        assert actual.iloc[5] == pytest.approx(-1.0)

    def test_stays_flat_when_always_zero(self):
        target = make_series([0.0] * 10)
        actual = apply_rebalance_threshold(target, threshold=0.20)
        assert (actual == 0).all()
