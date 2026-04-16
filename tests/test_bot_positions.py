"""Tests for bot/positions.py"""

from __future__ import annotations

import pytest

from bot.positions import compute_order_deltas


class TestComputeOrderDeltas:
    def test_buy_when_target_exceeds_current(self):
        orders = compute_order_deltas({"BTC/USDT": 0.10}, {"BTC/USDT": 0.05})
        assert len(orders) == 1
        assert orders[0]["side"] == "buy"
        assert orders[0]["delta_weight"] == pytest.approx(0.05)

    def test_sell_when_target_below_current(self):
        orders = compute_order_deltas({"SOL/USDT": 0.02}, {"SOL/USDT": 0.08})
        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["delta_weight"] == pytest.approx(-0.06)

    def test_hold_when_delta_below_min(self):
        orders = compute_order_deltas(
            {"BTC/USDT": 0.10},
            {"BTC/USDT": 0.095},
            min_delta=0.01,
        )
        assert orders == []

    def test_full_close_for_asset_removed_from_universe(self):
        orders = compute_order_deltas(
            target_weights={},
            current_weights={"XRP/USDT": 0.05},
        )
        assert len(orders) == 1
        assert orders[0]["symbol"] == "XRP/USDT"
        assert orders[0]["side"] == "sell"
        assert orders[0]["delta_weight"] == pytest.approx(-0.05)

    def test_sells_ordered_before_buys(self):
        orders = compute_order_deltas(
            target_weights={"BTC/USDT": 0.20, "SOL/USDT": 0.0},
            current_weights={"BTC/USDT": 0.05, "SOL/USDT": 0.10},
        )
        sides = [o["side"] for o in orders]
        # All sells should appear before any buy
        last_sell = max((i for i, s in enumerate(sides) if s == "sell"), default=-1)
        first_buy = min((i for i, s in enumerate(sides) if s == "buy"), default=len(sides))
        assert last_sell < first_buy

    def test_new_asset_not_in_current(self):
        orders = compute_order_deltas(
            target_weights={"ETH/USDT": 0.05},
            current_weights={},
        )
        assert len(orders) == 1
        assert orders[0]["side"] == "buy"
        assert orders[0]["symbol"] == "ETH/USDT"

    def test_exact_threshold_excluded(self):
        """Delta exactly equal to min_delta should be excluded (<=)."""
        orders = compute_order_deltas(
            {"BTC/USDT": 0.11},
            {"BTC/USDT": 0.10},
            min_delta=0.01,
        )
        assert orders == []

    def test_multiple_assets_correct_deltas(self):
        target = {"A/USDT": 0.10, "B/USDT": 0.05, "C/USDT": 0.0}
        current = {"A/USDT": 0.05, "B/USDT": 0.08, "C/USDT": 0.03}
        orders = compute_order_deltas(target, current, min_delta=0.01)
        by_symbol = {o["symbol"]: o for o in orders}
        assert by_symbol["A/USDT"]["side"] == "buy"
        assert by_symbol["B/USDT"]["side"] == "sell"
        assert by_symbol["C/USDT"]["side"] == "sell"

    def test_empty_inputs(self):
        orders = compute_order_deltas({}, {})
        assert orders == []
