"""Tests for bot/exchange.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot.exchange import ExchangeClient


class TestDryMode:
    def test_dry_mode_no_exchange_connection(self):
        client = ExchangeClient(mode="dry")
        assert client._exchange is None

    def test_get_account_equity_returns_starting_equity(self):
        client = ExchangeClient(mode="dry", starting_equity=5_000.0)
        assert client.get_account_equity() == 5_000.0

    def test_get_positions_returns_empty_dict(self):
        client = ExchangeClient(mode="dry")
        assert client.get_positions() == {}

    def test_get_current_price_returns_placeholder(self):
        client = ExchangeClient(mode="dry")
        assert client.get_current_price("BTC/USDT") == 1.0

    def test_place_market_order_returns_simulated(self):
        client = ExchangeClient(mode="dry")
        fill = client.place_market_order("ETH/USDT", "buy", 0.5)
        assert fill["symbol"] == "ETH/USDT"
        assert fill["side"] == "buy"
        assert fill["amount"] == 0.5
        assert fill["status"] == "simulated"
        assert fill["order_id"] is None

    def test_set_leverage_noop(self):
        client = ExchangeClient(mode="dry")
        # Should not raise
        client.set_leverage("BTC/USDT", 2)

    def test_set_margin_type_noop(self):
        client = ExchangeClient(mode="dry")
        # Should not raise
        client.set_margin_type("BTC/USDT", "cross")


class TestLiveModeConstruction:
    """Verify exchange is constructed with the correct mode flags."""

    def _make_mock_exchange(self):
        mock_ex = MagicMock()
        mock_ex.rateLimit = 0
        return mock_ex

    @patch("ccxt.binanceusdm")
    def test_testnet_sets_sandbox_mode(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_binanceusdm.return_value = mock_ex
        ExchangeClient(mode="testnet")
        mock_ex.set_sandbox_mode.assert_called_once_with(True)

    @patch("ccxt.binanceusdm")
    def test_live_does_not_set_sandbox_mode(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_binanceusdm.return_value = mock_ex
        ExchangeClient(mode="live")
        mock_ex.set_sandbox_mode.assert_not_called()

    @patch("ccxt.binanceusdm")
    def test_get_account_equity_reads_usdt_balance(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_ex.fetch_balance.return_value = {"USDT": {"total": 1234.56}}
        mock_binanceusdm.return_value = mock_ex
        client = ExchangeClient(mode="live")
        assert client.get_account_equity() == pytest.approx(1234.56)

    @patch("ccxt.binanceusdm")
    def test_get_positions_computes_weights(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_ex.fetch_balance.return_value = {"USDT": {"total": 1000.0}}
        mock_ex.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "notional": 200.0},
            {"symbol": "ETH/USDT", "notional": -100.0},
        ]
        mock_binanceusdm.return_value = mock_ex
        client = ExchangeClient(mode="live")
        positions = client.get_positions()
        assert positions["BTC/USDT"] == pytest.approx(0.2)
        assert positions["ETH/USDT"] == pytest.approx(-0.1)

    @patch("ccxt.binanceusdm")
    def test_place_market_order_returns_fill(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_ex.create_market_order.return_value = {
            "id": "ord123",
            "status": "closed",
            "average": 50_000.0,
        }
        mock_binanceusdm.return_value = mock_ex
        client = ExchangeClient(mode="live")
        fill = client.place_market_order("BTC/USDT", "buy", 0.001)
        assert fill["fill_price"] == pytest.approx(50_000.0)
        assert fill["status"] == "closed"
        assert fill["order_id"] == "ord123"

    @patch("ccxt.binanceusdm")
    def test_set_margin_type_ignores_already_set_error(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_ex.set_margin_mode.side_effect = Exception("already set")
        mock_binanceusdm.return_value = mock_ex
        client = ExchangeClient(mode="live")
        # Should not raise
        client.set_margin_type("BTC/USDT", "cross")

    @patch("ccxt.binanceusdm")
    def test_set_margin_type_raises_on_other_errors(self, mock_binanceusdm):
        mock_ex = self._make_mock_exchange()
        mock_ex.set_margin_mode.side_effect = Exception("network timeout")
        mock_binanceusdm.return_value = mock_ex
        client = ExchangeClient(mode="live")
        with pytest.raises(Exception, match="network timeout"):
            client.set_margin_type("BTC/USDT", "cross")
