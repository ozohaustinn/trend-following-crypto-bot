"""Tests for bot/risk.py"""

from __future__ import annotations

import pytest

from bot.risk import (
    check_equity_floor,
    check_min_order,
    check_order_size,
    check_single_position,
    check_total_leverage,
    run_all_checks,
)


class TestCheckSinglePosition:
    def test_approves_within_limit(self):
        ok, msg = check_single_position("BTC/USDT", 1.5, max_weight=2.0)
        assert ok is True
        assert msg == "OK"

    def test_rejects_when_exceeds_limit(self):
        ok, msg = check_single_position("BTC/USDT", 2.1, max_weight=2.0)
        assert ok is False
        assert "BTC/USDT" in msg

    def test_approves_at_exact_limit(self):
        ok, _ = check_single_position("ETH/USDT", 2.0, max_weight=2.0)
        assert ok is True

    def test_checks_absolute_value_for_shorts(self):
        ok, msg = check_single_position("SOL/USDT", -2.1, max_weight=2.0)
        assert ok is False
        assert "SOL/USDT" in msg


class TestCheckTotalLeverage:
    def test_approves_within_limit(self):
        weights = {"A": 1.0, "B": 1.5, "C": 0.5}
        ok, msg = check_total_leverage(weights, max_leverage=5.0)
        assert ok is True

    def test_rejects_when_exceeds_limit(self):
        weights = {"A": 2.0, "B": 2.0, "C": 2.0}
        ok, msg = check_total_leverage(weights, max_leverage=5.0)
        assert ok is False
        assert "leverage" in msg.lower()

    def test_uses_absolute_values_for_shorts(self):
        weights = {"A": -2.0, "B": -2.0, "C": -2.0}
        ok, _ = check_total_leverage(weights, max_leverage=5.0)
        assert ok is False

    def test_empty_weights_approved(self):
        ok, _ = check_total_leverage({}, max_leverage=5.0)
        assert ok is True


class TestCheckOrderSize:
    def test_approves_within_limit(self):
        ok, _ = check_order_size(400.0, 1000.0, max_pct=0.50)
        assert ok is True

    def test_rejects_when_exceeds_limit(self):
        ok, msg = check_order_size(600.0, 1000.0, max_pct=0.50)
        assert ok is False
        assert "$600.00" in msg

    def test_approves_at_exact_limit(self):
        ok, _ = check_order_size(500.0, 1000.0, max_pct=0.50)
        assert ok is True


class TestCheckMinOrder:
    def test_approves_above_minimum(self):
        ok, _ = check_min_order(15.0, min_value=10.0)
        assert ok is True

    def test_rejects_below_minimum(self):
        ok, msg = check_min_order(5.0, min_value=10.0)
        assert ok is False
        assert "$5.00" in msg

    def test_approves_at_exactly_minimum(self):
        """Equal to min_value is not below minimum — should be approved."""
        ok, _ = check_min_order(10.0, min_value=10.0)
        assert ok is True

    def test_approves_just_above_minimum(self):
        ok, _ = check_min_order(10.01, min_value=10.0)
        assert ok is True


class TestCheckEquityFloor:
    def test_approves_above_floor(self):
        ok, _ = check_equity_floor(500.0, floor=100.0)
        assert ok is True

    def test_rejects_below_floor(self):
        ok, msg = check_equity_floor(50.0, floor=100.0)
        assert ok is False
        assert "floor" in msg.lower()

    def test_approves_at_floor(self):
        """Equity exactly at floor is not below it — should be approved."""
        ok, _ = check_equity_floor(100.0, floor=100.0)
        assert ok is True


class TestRunAllChecks:
    def _default_kwargs(self, **overrides):
        kwargs = dict(
            symbol="BTC/USDT",
            new_weight=0.5,
            order_value_usd=100.0,
            all_new_weights={"BTC/USDT": 0.5},
            account_equity=1000.0,
            max_single_position=2.0,
            max_total_leverage=5.0,
            max_order_pct=0.50,
            min_order_value=10.0,
            equity_floor=100.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_all_passing_returns_ok(self):
        ok, reason = run_all_checks(**self._default_kwargs())
        assert ok is True
        assert reason == "OK"

    def test_equity_floor_blocks_all(self):
        ok, reason = run_all_checks(**self._default_kwargs(account_equity=50.0))
        assert ok is False
        assert "floor" in reason.lower()

    def test_single_position_check_fails(self):
        ok, reason = run_all_checks(**self._default_kwargs(new_weight=3.0))
        assert ok is False
        assert "exceed" in reason.lower()

    def test_total_leverage_check_fails(self):
        weights = {f"A{i}": 1.5 for i in range(5)}  # total = 7.5
        ok, reason = run_all_checks(**self._default_kwargs(all_new_weights=weights))
        assert ok is False
        assert "leverage" in reason.lower()

    def test_order_too_large_fails(self):
        ok, reason = run_all_checks(**self._default_kwargs(order_value_usd=600.0))
        assert ok is False

    def test_min_order_fails(self):
        ok, reason = run_all_checks(**self._default_kwargs(order_value_usd=5.0))
        assert ok is False
        assert "minimum" in reason.lower()

    def test_stops_at_first_failure(self):
        # Equity floor should fail first, before other checks
        ok, reason = run_all_checks(
            **self._default_kwargs(account_equity=50.0, new_weight=3.0)
        )
        assert ok is False
        assert "floor" in reason.lower()
