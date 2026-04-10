"""Performance metrics for the trend-following backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import Trade


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute standard performance statistics from a daily return series.

    Parameters
    ----------
    returns:
        Daily portfolio returns (not log-returns).
    benchmark_returns:
        Optional daily benchmark returns (e.g. BTC buy-and-hold).
    risk_free_rate:
        Annualized risk-free rate (default 0.0).

    Returns
    -------
    dict
        Keys: ``total_return``, ``cagr``, ``volatility``, ``sharpe``,
        ``sortino``, ``max_drawdown``, ``mar``, ``calmar``,
        ``alpha`` (if benchmark provided), ``beta`` (if benchmark provided).
    """
    returns = returns.dropna()
    if returns.empty:
        return {}

    n_days = len(returns)
    ann_factor = 365.0

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)

    # CAGR
    years = n_days / ann_factor
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    # Volatility (annualized)
    volatility = float(returns.std() * np.sqrt(ann_factor))

    # Sharpe
    daily_rf = (1 + risk_free_rate) ** (1 / ann_factor) - 1
    excess = returns - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor)) if excess.std() > 0 else 0.0

    # Sortino
    downside = excess[excess < 0]
    downside_std = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() * ann_factor / downside_std) if downside_std > 0 else 0.0

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # MAR / Calmar
    mar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    result = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "mar": mar,
        "calmar": mar,
    }

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).dropna()
        common = returns.reindex(bench.index).dropna()
        bench = bench.reindex(common.index)
        if len(common) > 1:
            cov = np.cov(common, bench)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
            bench_ann = float(bench.mean() * ann_factor)
            strategy_ann = float(common.mean() * ann_factor)
            alpha = float(strategy_ann - risk_free_rate - beta * (bench_ann - risk_free_rate))
            result["alpha"] = alpha
            result["beta"] = beta

    return result


def trade_statistics(trades: list[Trade]) -> dict:
    """Compute trade-level statistics.

    Parameters
    ----------
    trades:
        List of completed :class:`~backtest.engine.Trade` objects.

    Returns
    -------
    dict
        Keys: ``n_trades``, ``win_rate``, ``avg_return``, ``max_return``,
        ``min_return``, ``gain_to_loss_ratio``, ``avg_holding_days``.
    """
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
            "gain_to_loss_ratio": 0.0,
            "avg_holding_days": 0.0,
        }

    pnls = [t.pnl_pct for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / len(pnls)
    avg_win = float(np.mean(winners)) if winners else 0.0
    avg_loss = float(np.mean(losers)) if losers else 0.0
    gain_to_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_return": float(np.mean(pnls)),
        "max_return": float(np.max(pnls)),
        "min_return": float(np.min(pnls)),
        "gain_to_loss_ratio": gain_to_loss,
        "avg_holding_days": float(np.mean([t.holding_days for t in trades])),
    }


def print_report(metrics: dict, trade_stats: dict | None = None) -> str:
    """Format and print a performance report.

    Parameters
    ----------
    metrics:
        Output of :func:`compute_metrics`.
    trade_stats:
        Optional output of :func:`trade_statistics`.

    Returns
    -------
    str
        The formatted report string (also printed to stdout).
    """
    lines = [
        "=" * 52,
        "  PERFORMANCE REPORT",
        "=" * 52,
        f"  Total Return     : {metrics.get('total_return', 0):.2%}",
        f"  CAGR             : {metrics.get('cagr', 0):.2%}",
        f"  Volatility       : {metrics.get('volatility', 0):.2%}",
        f"  Sharpe Ratio     : {metrics.get('sharpe', 0):.2f}",
        f"  Sortino Ratio    : {metrics.get('sortino', 0):.2f}",
        f"  Max Drawdown     : {metrics.get('max_drawdown', 0):.2%}",
        f"  MAR / Calmar     : {metrics.get('mar', 0):.2f}",
    ]
    if "alpha" in metrics:
        lines.append(f"  Alpha (ann.)     : {metrics['alpha']:.2%}")
        lines.append(f"  Beta             : {metrics['beta']:.2f}")

    if trade_stats:
        lines += [
            "-" * 52,
            "  TRADE STATISTICS",
            "-" * 52,
            f"  # Trades         : {trade_stats.get('n_trades', 0)}",
            f"  Win Rate         : {trade_stats.get('win_rate', 0):.2%}",
            f"  Avg Return/Trade : {trade_stats.get('avg_return', 0):.2%}",
            f"  Best Trade       : {trade_stats.get('max_return', 0):.2%}",
            f"  Worst Trade      : {trade_stats.get('min_return', 0):.2%}",
            f"  Gain/Loss Ratio  : {trade_stats.get('gain_to_loss_ratio', 0):.2f}",
            f"  Avg Holding Days : {trade_stats.get('avg_holding_days', 0):.1f}",
        ]

    lines.append("=" * 52)
    report = "\n".join(lines)
    print(report)
    return report
