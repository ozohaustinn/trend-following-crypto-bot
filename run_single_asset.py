#!/usr/bin/env python
"""Single-asset backtest CLI — simplified BTC (or any symbol) backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_report, trade_statistics
from data.loader import fetch_ohlcv
from strategy.ensemble import ensemble_weights


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-asset trend-following backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (ccxt format)")
    parser.add_argument("--exchange", default=config.DEFAULT_EXCHANGE, help="ccxt exchange id")
    parser.add_argument("--start", default=config.DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=config.DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--mode",
        choices=["long_only", "long_short"],
        default="long_only",
        help="Signal mode",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=config.LOOKBACK_PERIODS,
        help="Donchian lookback periods (days)",
    )
    parser.add_argument("--cost-bps", type=float, default=config.TRANSACTION_COST_BPS,
                        dest="cost_bps", help="One-way transaction cost in basis points")
    parser.add_argument("--capital", type=float, default=config.DEFAULT_CAPITAL,
                        help="Starting capital")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local parquet cache")
    parser.add_argument("--output", default=None,
                        help="Path to save the equity curve PNG (default: <symbol>_equity.png)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Fetching {args.symbol} from {args.exchange} ({args.start} to {args.end})…")
    df = fetch_ohlcv(
        args.symbol,
        args.start,
        args.end,
        exchange=args.exchange,
        use_cache=not args.no_cache,
    )
    if df.empty:
        print("ERROR: No data returned. Check symbol / date range.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing ensemble weights (lookbacks={args.lookbacks}, mode={args.mode})…")
    weights = ensemble_weights(
        df["close"],
        lookback_periods=args.lookbacks,
        mode=args.mode,
    )

    engine = BacktestEngine(initial_capital=args.capital, cost_bps=args.cost_bps)
    result = engine.run_single_asset(df["close"], weights)

    metrics = compute_metrics(result.returns)
    tstats = trade_statistics(result.trades)
    print_report(metrics, tstats)

    # Equity curve plot
    out_path = args.output or f"{args.symbol.replace('/', '_')}_equity.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    result.equity.plot(ax=ax, color="steelblue", linewidth=1.5)
    ax.set_title(f"{args.symbol} — Trend-Following Combo ({args.mode})")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nEquity curve saved to: {out_path}")


if __name__ == "__main__":
    main()
