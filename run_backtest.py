#!/usr/bin/env python
"""Full multi-asset portfolio backtest CLI."""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from backtest.metrics import compute_metrics, print_report, trade_statistics
from backtest.portfolio import PortfolioManager
from data.loader import fetch_multiple


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-asset rotational trend-following portfolio backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exchange", default=config.DEFAULT_EXCHANGE, help="ccxt exchange id")
    parser.add_argument("--start", default=config.DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=config.DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--n-assets", type=int, default=config.UNIVERSE_SIZE,
                        dest="n_assets", help="Number of assets in portfolio")
    parser.add_argument(
        "--mode",
        choices=["long_only", "long_short"],
        default="long_only",
        help="Signal mode",
    )
    parser.add_argument("--cost-bps", type=float, default=config.TRANSACTION_COST_BPS,
                        dest="cost_bps", help="One-way transaction cost in basis points")
    parser.add_argument("--capital", type=float, default=config.DEFAULT_CAPITAL,
                        help="Starting capital")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit list of symbols to use (skips universe selection from exchange)",
    )
    parser.add_argument("--output", default="portfolio_equity.png",
                        help="Path to save equity curve PNG")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.symbols:
        symbols = args.symbols
    else:
        # Default: a representative set of majors (extend as needed)
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "SOL/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
            "LINK/USDT", "AVAX/USDT", "UNI/USDT", "ATOM/USDT", "XLM/USDT",
            "ETC/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT", "TRX/USDT",
        ]

    print(f"Fetching data for {len(symbols)} symbols ({args.start} to {args.end})…")
    all_data = fetch_multiple(symbols, args.start, args.end, exchange=args.exchange)

    if not all_data:
        print("ERROR: No data fetched. Check symbol list / date range.", file=sys.stderr)
        sys.exit(1)

    pm = PortfolioManager(n_assets=args.n_assets, mode=args.mode)
    print("Running portfolio backtest…")
    result_df = pm.run(all_data, args.start, args.end)

    returns = result_df["return"]
    metrics = compute_metrics(returns)
    print_report(metrics)

    # Equity curve
    equity = (1 + returns).cumprod() * args.capital
    fig, ax = plt.subplots(figsize=(12, 5))
    equity.plot(ax=ax, color="steelblue", linewidth=1.5)
    ax.set_title(f"Portfolio — Trend-Following Combo ({args.mode}, top {args.n_assets})")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nEquity curve saved to: {args.output}")

    # Monthly universe log
    print("\nMonthly universe composition (last 6 months):")
    last_rows = result_df.tail(180)
    monthly = last_rows.resample("ME")["universe"].last()
    for dt, uni in monthly.items():
        print(f"  {dt.date()}: {uni}")


if __name__ == "__main__":
    main()
