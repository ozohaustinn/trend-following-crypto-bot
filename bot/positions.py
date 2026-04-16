"""Compute required order deltas between target and current positions."""

from __future__ import annotations


def compute_order_deltas(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    min_delta: float = 0.01,
) -> list[dict]:
    """Return the list of required orders sorted sells-first.

    Parameters
    ----------
    target_weights:
        Mapping of ``{symbol: target_weight}``.  Weights are fractions of
        total equity (e.g. ``0.05`` = 5 % of equity long).
    current_weights:
        Mapping of ``{symbol: current_weight}`` from the exchange.
        Assets not in this dict are treated as having weight 0.
    min_delta:
        Minimum absolute weight change required to generate an order.
        Changes smaller than this are treated as HOLD to avoid dust orders.

    Returns
    -------
    list[dict]
        Each element has keys:

        * ``symbol``       – trading pair string
        * ``side``         – ``"buy"`` or ``"sell"``
        * ``delta_weight`` – signed weight change (positive = buy)

        SELL orders appear before BUY orders so margin is freed first.
        Assets present in *current_weights* but absent from *target_weights*
        receive a full close order (``delta = -current_weight``).
    """
    all_symbols = set(target_weights) | set(current_weights)
    orders: list[dict] = []

    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        delta = target - current

        if abs(delta) <= min_delta:
            continue

        side = "buy" if delta > 0 else "sell"
        orders.append({"symbol": symbol, "side": side, "delta_weight": delta})

    # Sells before buys to free margin first
    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, o["symbol"]))
    return orders
