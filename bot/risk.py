"""Pre-trade risk checks. Every public function returns (approved, reason)."""

from __future__ import annotations


def check_single_position(
    symbol: str,
    new_weight: float,
    max_weight: float = 2.0,
) -> tuple[bool, str]:
    """Reject if a single position would exceed *max_weight* (absolute)."""
    if abs(new_weight) > max_weight:
        return False, (
            f"{symbol}: position weight {new_weight:.1%} would exceed "
            f"max {max_weight:.0%}"
        )
    return True, "OK"


def check_total_leverage(
    all_new_weights: dict[str, float],
    max_leverage: float = 5.0,
) -> tuple[bool, str]:
    """Reject if the total portfolio leverage would exceed *max_leverage*."""
    total = sum(abs(w) for w in all_new_weights.values())
    if total > max_leverage:
        return False, (
            f"Total leverage {total:.1%} would exceed max {max_leverage:.0%}"
        )
    return True, "OK"


def check_order_size(
    order_value_usd: float,
    account_equity: float,
    max_pct: float = 0.50,
) -> tuple[bool, str]:
    """Reject if a single order exceeds *max_pct* of equity."""
    limit = max_pct * account_equity
    if order_value_usd > limit:
        return False, (
            f"Order value ${order_value_usd:.2f} exceeds "
            f"{max_pct:.0%} of equity (${limit:.2f})"
        )
    return True, "OK"


def check_min_order(
    order_value_usd: float,
    min_value: float = 10.0,
) -> tuple[bool, str]:
    """Reject if the order is below the Binance minimum notional."""
    if order_value_usd < min_value:
        return False, (
            f"Order value ${order_value_usd:.2f} is below minimum ${min_value:.2f}"
        )
    return True, "OK"


def check_equity_floor(
    account_equity: float,
    floor: float = 100.0,
) -> tuple[bool, str]:
    """Reject ALL orders when account equity is below the safety floor."""
    if account_equity < floor:
        return False, (
            f"Account equity ${account_equity:.2f} is below floor ${floor:.2f}"
        )
    return True, "OK"


def run_all_checks(
    symbol: str,
    new_weight: float,
    order_value_usd: float,
    all_new_weights: dict[str, float],
    account_equity: float,
    max_single_position: float = 2.0,
    max_total_leverage: float = 5.0,
    max_order_pct: float = 0.50,
    min_order_value: float = 10.0,
    equity_floor: float = 100.0,
) -> tuple[bool, str]:
    """Run all risk checks in sequence; return the first failure or ``(True, 'OK')``."""
    checks = [
        check_equity_floor(account_equity, equity_floor),
        check_single_position(symbol, new_weight, max_single_position),
        check_total_leverage(all_new_weights, max_total_leverage),
        check_order_size(order_value_usd, account_equity, max_order_pct),
        check_min_order(order_value_usd, min_order_value),
    ]
    for approved, reason in checks:
        if not approved:
            return False, reason
    return True, "OK"
