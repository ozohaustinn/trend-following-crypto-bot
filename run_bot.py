#!/usr/bin/env python
"""CLI entry point for the daily trend-following crypto bot.

Usage examples
--------------
# Phase 1: Dry run (no exchange connection)
python run_bot.py --mode dry

# Phase 2: Paper trading on Binance testnet
python run_bot.py --mode testnet

# Phase 3: Live trading
python run_bot.py --mode live

# Options
python run_bot.py --mode dry --n-assets 10
python run_bot.py --mode dry --strategy long_short
python run_bot.py --mode dry --verbose
"""

from __future__ import annotations

import argparse
import json
import sys

import bot_config
from bot.runner import BotRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily trend-following crypto bot (Donchian Channel ensemble)",
    )
    parser.add_argument(
        "--mode",
        choices=["dry", "testnet", "live"],
        default=bot_config.DEFAULT_MODE,
        help="Operating mode (default: %(default)s)",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=20,
        help="Number of assets in portfolio (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        choices=["long_only", "long_short"],
        default="long_only",
        help="Strategy mode (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output to stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        print(f"[run_bot] Starting in {args.mode!r} mode …")

    runner = BotRunner(
        mode=args.mode,
        strategy_mode=args.strategy,
        n_assets=args.n_assets,
    )

    run_data = runner.run()

    if args.verbose:
        print(json.dumps(run_data, indent=2, default=str))
    else:
        n_orders = len(run_data.get("orders_placed", []))
        equity = run_data.get("account_equity", 0.0)
        errors = run_data.get("errors", [])
        print(
            f"[run_bot] Done. Equity: ${equity:.2f} | "
            f"Orders: {n_orders} | Errors: {len(errors)}"
        )
        if errors:
            for err in errors:
                print(f"  [ERROR] {err}", file=sys.stderr)

    return 0 if not run_data.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
