"""Tests for backtest/engine.py"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, Trade


def make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestBacktestEngine:
    def test_zero_weight_no_return(self):
        close = make_series([100, 110, 120, 130])
        weights = make_series([0.0, 0.0, 0.0, 0.0])
        engine = BacktestEngine(initial_capital=100_000, cost_bps=0)
        result = engine.run_single_asset(close, weights)
        assert result.returns.sum() == pytest.approx(0.0)

    def test_pnl_equals_weight_times_return(self):
        # Weight = 1.0 (fully invested), returns should equal asset returns
        close = make_series([100, 110, 121, 133.1])
        weights = make_series([1.0, 1.0, 1.0, 1.0])
        engine = BacktestEngine(initial_capital=100_000, cost_bps=0)
        result = engine.run_single_asset(close, weights)
        # Daily return at index 1: weight[0] * (110/100 - 1) = 1.0 * 0.10
        assert result.returns.iloc[1] == pytest.approx(0.10, abs=1e-6)
        assert result.returns.iloc[2] == pytest.approx(0.10, abs=1e-6)

    def test_transaction_costs_deducted(self):
        # Engine uses weight[t-1] for the return at bar t.
        # At bar 1: w_prev=0, w_curr=1 → gross=0*0.10=0, tc=1*0.001=0.001 → return=-0.001
        # At bar 2: w_prev=1, w_curr=1 → gross=1*0.10=0.10, tc=0 → return=0.10
        close = make_series([100, 110, 121])
        weights = make_series([0.0, 1.0, 1.0])
        engine = BacktestEngine(initial_capital=100_000, cost_bps=10)
        result = engine.run_single_asset(close, weights)
        assert result.returns.iloc[1] == pytest.approx(-0.001, abs=1e-6)
        assert result.returns.iloc[2] == pytest.approx(0.10, abs=1e-6)

    def test_equity_curve_is_cumulative_product(self):
        close = make_series([100, 110, 121])
        weights = make_series([1.0, 1.0, 1.0])
        capital = 100_000
        engine = BacktestEngine(initial_capital=capital, cost_bps=0)
        result = engine.run_single_asset(close, weights)
        expected = (1 + result.returns).cumprod() * capital
        pd.testing.assert_series_equal(result.equity, expected)

    def test_trade_list_populated(self):
        # A simple buy-hold-sell scenario via weights
        close = make_series([100, 110, 120, 130, 140, 100])
        weights = make_series([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        engine = BacktestEngine(initial_capital=100_000, cost_bps=0)
        result = engine.run_single_asset(close, weights)
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.direction == 1
        assert trade.entry_price == pytest.approx(110.0)
        assert trade.exit_price == pytest.approx(100.0)

    def test_no_cost_without_weight_change(self):
        # Engine loop starts at i=1 and compares weight[i] vs weight[i-1].
        # With constant weights=[0.5,...], no consecutive change → no tc after bar 1.
        # At bar 2: w_prev=0.5, w_curr=0.5, r=0.10 → no tc → return=0.05
        close = make_series([100, 110, 121, 133.1])
        weights = make_series([0.5, 0.5, 0.5, 0.5])
        engine = BacktestEngine(initial_capital=100_000, cost_bps=100)
        result = engine.run_single_asset(close, weights)
        # Bar 2 and 3: no weight change, so no transaction cost
        assert result.returns.iloc[2] == pytest.approx(0.05, abs=1e-6)
        assert result.returns.iloc[3] == pytest.approx(0.05, abs=1e-4)

    def test_initial_equity_equals_capital(self):
        close = make_series([100, 110])
        weights = make_series([0.0, 0.0])
        engine = BacktestEngine(initial_capital=50_000, cost_bps=0)
        result = engine.run_single_asset(close, weights)
        assert result.equity.iloc[0] == pytest.approx(50_000.0)
