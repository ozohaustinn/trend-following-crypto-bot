"""Optional Telegram and email notifications. Fails silently if not configured."""

from __future__ import annotations

import urllib.request
import urllib.parse

import bot_config


def send_telegram(message: str) -> bool:
    """Send *message* via Telegram Bot API.

    Returns ``True`` if the message was sent successfully, ``False`` if
    the bot is not configured or the request fails.
    """
    token = bot_config.TELEGRAM_BOT_TOKEN
    chat_id = bot_config.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()

    try:
        with urllib.request.urlopen(url, data=payload, timeout=10) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def format_daily_summary(run_data: dict) -> str:
    """Format *run_data* into a short human-readable Telegram message.

    Example output::

        [TREND BOT] 2026-04-10 00:05 UTC
        Mode: testnet | Equity: $487.32
        Universe: 14 assets
        Signals: 3 buy, 1 sell, 10 hold
        Orders: 4 placed, 4 filled
        P&L today: N/A
    """
    timestamp = run_data.get("timestamp", "unknown")
    mode = run_data.get("mode", "unknown")
    equity = run_data.get("account_equity", 0.0)
    universe = run_data.get("universe", [])
    signals = run_data.get("signals", {})
    orders_placed = run_data.get("orders_placed", [])

    n_buy = sum(1 for o in orders_placed if o.get("side") == "buy")
    n_sell = sum(1 for o in orders_placed if o.get("side") == "sell")
    n_filled = sum(1 for o in orders_placed if o.get("status") == "filled")

    sig_buys = sum(1 for s in signals.values() if s.get("action") == "BUY")
    sig_sells = sum(1 for s in signals.values() if s.get("action") == "SELL")
    sig_holds = sum(1 for s in signals.values() if s.get("action") == "HOLD")

    lines = [
        f"[TREND BOT] {timestamp}",
        f"Mode: {mode} | Equity: ${equity:.2f}",
        f"Universe: {len(universe)} assets",
        f"Signals: {sig_buys} buy, {sig_sells} sell, {sig_holds} hold",
        f"Orders: {n_buy + n_sell} placed, {n_filled} filled",
    ]

    errors = run_data.get("errors", [])
    if errors:
        lines.append(f"Errors: {len(errors)}")

    return "\n".join(lines)
