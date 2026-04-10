"""Integration tests for bot/runner.py (dry-run mode only)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.runner import BotRunner


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_close(n: int = 500, start: str = "2024-01-01") -> pd.Series:
    """Generate a rising price series long enough for all lookbacks."""
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    prices = 100.0 * np.cumprod(1 + np.random.default_rng(42).normal(0.001, 0.02, n))
    return pd.Series(prices, index=idx)


def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    close = _make_close(n)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(n, 5_000_000.0),
        },
        index=close.index,
    )


def _make_all_data(symbols: list[str], n: int = 500) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    result = {}
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    for sym in symbols:
        close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        result[sym] = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(n, 5_000_000.0),
            },
            index=idx,
        )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBotRunnerDryMode:
    """End-to-end dry-run tests with mocked data fetching."""

    def _make_runner(self, tmp_path, n_assets: int = 3) -> BotRunner:
        runner = BotRunner(mode="dry", n_assets=n_assets, starting_equity=10_000.0)
        # Redirect logs to temp directory
        from bot.logger import TradeLogger
        runner.logger = TradeLogger(log_dir=str(tmp_path))
        return runner

    def test_run_returns_dict_with_expected_keys(self, tmp_path):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        all_data = _make_all_data(symbols)

        runner = self._make_runner(tmp_path)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()

        assert isinstance(result, dict)
        for key in ("timestamp", "mode", "account_equity", "universe", "signals",
                    "orders_placed", "risk_checks", "errors"):
            assert key in result

    def test_run_mode_is_dry(self, tmp_path):
        all_data = _make_all_data(["BTC/USDT"])
        runner = self._make_runner(tmp_path)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()
        assert result["mode"] == "dry"

    def test_run_does_not_place_real_orders(self, tmp_path):
        symbols = ["BTC/USDT", "ETH/USDT"]
        all_data = _make_all_data(symbols)
        runner = self._make_runner(tmp_path, n_assets=2)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()
        # In dry mode all fills must be simulated
        for order in result.get("orders_placed", []):
            assert order["status"] == "simulated"

    def test_run_writes_log_file(self, tmp_path):
        all_data = _make_all_data(["BTC/USDT"])
        runner = self._make_runner(tmp_path)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            runner.run()
        log_files = list(tmp_path.glob("*.json"))
        assert len(log_files) >= 1

    def test_run_survives_empty_data(self, tmp_path):
        runner = self._make_runner(tmp_path)
        with patch.object(runner, "_fetch_ohlcv", return_value={}):
            result = runner.run()
        # Should complete without raising; universe will be empty
        assert isinstance(result, dict)
        assert result["universe"] == []

    def test_signals_include_all_universe_assets(self, tmp_path):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        all_data = _make_all_data(symbols)
        runner = self._make_runner(tmp_path, n_assets=3)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()
        for sym in result.get("universe", []):
            assert sym in result["signals"]

    def test_account_equity_from_exchange(self, tmp_path):
        all_data = _make_all_data(["BTC/USDT"])
        runner = self._make_runner(tmp_path)
        runner.exchange._starting_equity = 7_777.0
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()
        assert result["account_equity"] == pytest.approx(7_777.0)

    def test_risk_checks_structure(self, tmp_path):
        all_data = _make_all_data(["BTC/USDT"])
        runner = self._make_runner(tmp_path)
        with patch.object(runner, "_fetch_ohlcv", return_value=all_data):
            result = runner.run()
        rc = result["risk_checks"]
        assert "max_position_exceeded" in rc
        assert "max_leverage_exceeded" in rc
        assert "equity_floor_breach" in rc
