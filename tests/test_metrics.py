"""Tests for backtest/metrics.py"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import Trade
from backtest.metrics import compute_metrics, print_report, trade_statistics


def make_returns(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestComputeMetrics:
    def test_sharpe_formula(self):
        # Known returns → known Sharpe
        rng = np.random.default_rng(0)
        rets = make_returns(rng.normal(0.001, 0.02, 365))
        m = compute_metrics(rets)
        expected_sharpe = rets.mean() / rets.std() * np.sqrt(365)
        assert m["sharpe"] == pytest.approx(expected_sharpe, rel=1e-3)

    def test_total_return(self):
        rets = make_returns([0.01, 0.02, -0.01, 0.05])
        m = compute_metrics(rets)
        expected = (1.01 * 1.02 * 0.99 * 1.05) - 1
        assert m["total_return"] == pytest.approx(expected, rel=1e-6)

    def test_max_drawdown_negative(self):
        # Returns that go up then crash
        rets = make_returns([0.10, 0.10, -0.50, 0.10])
        m = compute_metrics(rets)
        assert m["max_drawdown"] < 0

    def test_max_drawdown_zero_for_always_positive(self):
        rets = make_returns([0.01] * 100)
        m = compute_metrics(rets)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_volatility_annualized(self):
        rng = np.random.default_rng(1)
        rets = make_returns(rng.normal(0, 0.02, 365))
        m = compute_metrics(rets)
        expected_vol = rets.std() * np.sqrt(365)
        assert m["volatility"] == pytest.approx(expected_vol, rel=1e-6)

    def test_alpha_beta_with_benchmark(self):
        rng = np.random.default_rng(2)
        bench = make_returns(rng.normal(0.001, 0.02, 365))
        strat = bench * 0.5 + make_returns(rng.normal(0.0005, 0.01, 365))
        m = compute_metrics(strat, benchmark_returns=bench)
        assert "alpha" in m
        assert "beta" in m
        assert m["beta"] == pytest.approx(0.5, abs=0.3)  # approximate

    def test_empty_returns(self):
        m = compute_metrics(make_returns([]))
        assert m == {}

    def test_sortino_ratio_positive_for_positive_drift(self):
        rng = np.random.default_rng(3)
        rets = make_returns(rng.normal(0.002, 0.015, 365))
        m = compute_metrics(rets)
        assert m["sortino"] > 0

    def test_mar_equals_cagr_over_mdd(self):
        rng = np.random.default_rng(4)
        rets = make_returns(rng.normal(0.001, 0.02, 365))
        m = compute_metrics(rets)
        if m["max_drawdown"] != 0:
            expected_mar = m["cagr"] / abs(m["max_drawdown"])
            assert m["mar"] == pytest.approx(expected_mar, rel=1e-6)


class TestTradeStatistics:
    def _make_trades(self, pnls):
        trades = []
        for p in pnls:
            trades.append(Trade(
                entry_date=pd.Timestamp("2020-01-01"),
                exit_date=pd.Timestamp("2020-01-10"),
                entry_price=100.0,
                exit_price=100.0 * (1 + p),
                direction=1,
                pnl_pct=p,
                holding_days=9,
            ))
        return trades

    def test_empty_trades(self):
        stats = trade_statistics([])
        assert stats["n_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_win_rate(self):
        trades = self._make_trades([0.1, 0.2, -0.05, 0.15, -0.10])
        stats = trade_statistics(trades)
        assert stats["n_trades"] == 5
        assert stats["win_rate"] == pytest.approx(3 / 5)

    def test_avg_return(self):
        pnls = [0.10, 0.20, -0.05]
        trades = self._make_trades(pnls)
        stats = trade_statistics(trades)
        assert stats["avg_return"] == pytest.approx(np.mean(pnls), rel=1e-6)

    def test_max_min_return(self):
        pnls = [0.10, 0.20, -0.05]
        trades = self._make_trades(pnls)
        stats = trade_statistics(trades)
        assert stats["max_return"] == pytest.approx(0.20)
        assert stats["min_return"] == pytest.approx(-0.05)

    def test_gain_to_loss_ratio(self):
        trades = self._make_trades([0.10, 0.20, -0.05, -0.10])
        stats = trade_statistics(trades)
        avg_win = np.mean([0.10, 0.20])
        avg_loss = np.mean([-0.05, -0.10])
        expected = abs(avg_win / avg_loss)
        assert stats["gain_to_loss_ratio"] == pytest.approx(expected, rel=1e-6)


class TestPrintReport:
    def test_returns_string(self):
        rets = make_returns([0.01, -0.005, 0.02])
        m = compute_metrics(rets)
        report = print_report(m)
        assert isinstance(report, str)
        assert "Sharpe" in report

    def test_with_trade_stats(self):
        rets = make_returns([0.01, -0.005, 0.02])
        m = compute_metrics(rets)
        stats = trade_statistics([])
        report = print_report(m, stats)
        assert "TRADE STATISTICS" in report
