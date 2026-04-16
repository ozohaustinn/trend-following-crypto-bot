"""Binance Futures API wrapper (ccxt-based) with dry/testnet/live mode support."""

from __future__ import annotations

import bot_config


class ExchangeClient:
    """Thin ccxt wrapper for Binance Futures.

    Parameters
    ----------
    mode:
        ``"dry"``     – no exchange connection; all methods return mock data.
        ``"testnet"`` – connects to ``testnet.binancefuture.com`` with fake USDT.
        ``"live"``    – connects to ``fapi.binance.com`` with real funds.
    starting_equity:
        Starting equity used for dry-run equity simulation (USDT).
    """

    def __init__(self, mode: str = "dry", starting_equity: float = 10_000.0) -> None:
        self.mode = mode
        self._starting_equity = starting_equity
        self._exchange = None

        if mode != "dry":
            self._exchange = self._build_exchange(mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_exchange(self, mode: str):
        import ccxt  # lazy import — not needed for dry runs

        if mode == "testnet":
            api_key = bot_config.BINANCE_TESTNET_API_KEY
            api_secret = bot_config.BINANCE_TESTNET_API_SECRET
        else:  # live
            api_key = bot_config.BINANCE_API_KEY
            api_secret = bot_config.BINANCE_API_SECRET

        exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "future"},
            }
        )

        if mode == "testnet":
            exchange.set_sandbox_mode(True)

        return exchange

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_account_equity(self) -> float:
        """Return total account equity in USDT."""
        if self.mode == "dry":
            return self._starting_equity

        balance = self._exchange.fetch_balance()
        # USDT-M futures: total equity is in the 'USDT' wallet
        usdt = balance.get("USDT", {})
        # 'total' includes unrealised PnL; fall back to 'free' if absent
        equity = usdt.get("total") or usdt.get("free", 0.0)
        return float(equity)

    def get_positions(self) -> dict[str, float]:
        """Return ``{symbol: current_weight}`` for all open positions.

        Weight is computed as ``position_notional / account_equity``.
        Positive weight = long, negative = short.
        """
        if self.mode == "dry":
            return {}

        equity = self.get_account_equity()
        if equity == 0:
            return {}

        raw_positions = self._exchange.fetch_positions()
        result: dict[str, float] = {}
        for pos in raw_positions:
            notional = float(pos.get("notional") or 0.0)
            if notional == 0.0:
                continue
            symbol = pos["symbol"]
            # ccxt returns notional as signed (positive=long, negative=short)
            result[symbol] = notional / equity

        return result

    def get_current_price(self, symbol: str) -> float:
        """Return the last traded price for *symbol*."""
        if self.mode == "dry":
            # Return a placeholder; the runner uses OHLCV close prices instead
            return 1.0

        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def place_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Place a market order and return fill information.

        Parameters
        ----------
        symbol:
            e.g. ``"BTC/USDT"``
        side:
            ``"buy"`` or ``"sell"``
        amount:
            Quantity in base currency units (e.g. 0.001 BTC).

        Returns
        -------
        dict
            Keys: ``symbol``, ``side``, ``amount``, ``fill_price``,
            ``status``, ``order_id``.
        """
        if self.mode == "dry":
            return {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "fill_price": self.get_current_price(symbol),
                "status": "simulated",
                "order_id": None,
            }

        order = self._exchange.create_market_order(symbol, side, amount)
        fill_price = float(order.get("average") or order.get("price") or 0.0)
        return {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "fill_price": fill_price,
            "status": order.get("status", "unknown"),
            "order_id": order.get("id"),
        }

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for *symbol*. No-op in dry mode."""
        if self.mode == "dry":
            return
        self._exchange.set_leverage(leverage, symbol)

    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        """Set margin type (``"cross"`` or ``"isolated"``) for *symbol*.

        No-op in dry mode. Silently ignores errors when the margin type
        is already set to the requested value.
        """
        if self.mode == "dry":
            return
        try:
            self._exchange.set_margin_mode(margin_type, symbol)
        except Exception as exc:  # noqa: BLE001
            # Binance returns an error if the margin type is already set
            if "already" not in str(exc).lower():
                raise
