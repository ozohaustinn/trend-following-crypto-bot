"""Bar-by-bar backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

import config


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    direction: int          # 1 for long, -1 for short
    pnl_pct: float          # percentage return on the trade
    holding_days: int


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[Trade]
    weights: pd.Series


class BacktestEngine:
    """Bar-by-bar backtest engine.

    Parameters
    ----------
    initial_capital:
        Starting portfolio value.
    cost_bps:
        One-way transaction cost in basis points (default 10 bps).
    """

    def __init__(
        self,
        initial_capital: float = config.DEFAULT_CAPITAL,
        cost_bps: float = config.TRANSACTION_COST_BPS,
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_bps = cost_bps

    # ------------------------------------------------------------------
    # Single-asset backtest
    # ------------------------------------------------------------------

    def run_single_asset(
        self,
        close: pd.Series,
        weights: pd.Series,
    ) -> BacktestResult:
        """Run a backtest for one asset given precomputed weight series.

        Daily portfolio return:
        ``portfolio_return[t] = weight[t-1] * asset_return[t] - transaction_cost``

        Transaction cost is incurred whenever the weight changes:
        ``cost = |weight[t] - weight[t-1]| * cost_bps / 10_000``

        Parameters
        ----------
        close:
            Daily close price series.
        weights:
            Precomputed position weight series (from ensemble or single model).

        Returns
        -------
        BacktestResult
        """
        # Align series
        close, weights = close.align(weights, join="inner")
        asset_returns = close.pct_change()

        port_returns = pd.Series(0.0, index=close.index)
        weight_vals = weights.values
        ret_vals = asset_returns.values
        close_vals = close.values
        cost_factor = self.cost_bps / 10_000

        for i in range(1, len(close)):
            w_prev = weight_vals[i - 1]
            w_curr = weight_vals[i]
            r = ret_vals[i]
            if np.isnan(r):
                continue
            gross = w_prev * r
            tc = abs(w_curr - w_prev) * cost_factor
            port_returns.iloc[i] = gross - tc

        equity = (1 + port_returns).cumprod() * self.initial_capital

        trades = self._extract_trades(close, weights)
        return BacktestResult(
            equity=equity,
            returns=port_returns,
            trades=trades,
            weights=weights,
        )

    # ------------------------------------------------------------------
    # Multi-asset portfolio backtest
    # ------------------------------------------------------------------

    def run_portfolio(
        self,
        all_data: dict[str, pd.DataFrame],
        universe_schedule: dict[str, list[str]],
        ensemble_fn,  # callable(close) -> pd.Series of weights
        n_assets: int = config.UNIVERSE_SIZE,
    ) -> BacktestResult:
        """Run the full multi-asset rotational backtest.

        Parameters
        ----------
        all_data:
            Mapping ``{symbol: ohlcv_df}``.
        universe_schedule:
            Mapping ``{month_end_date_str: [symbol, ...]}``.  Each entry
            specifies which assets are active for the following month.
        ensemble_fn:
            Callable that accepts a close price ``pd.Series`` and returns a
            weight ``pd.Series``.
        n_assets:
            Expected portfolio size (used for equal-weight normalisation).

        Returns
        -------
        BacktestResult
        """
        # Build a sorted list of rebalance dates
        rebalance_dates = sorted(universe_schedule.keys())

        # Determine the full date range from the data
        all_dates: set[pd.Timestamp] = set()
        for df in all_data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)
        if not dates:
            raise ValueError("all_data is empty")

        port_returns = pd.Series(0.0, index=pd.DatetimeIndex(dates))
        all_trades: list[Trade] = []
        combined_weights: dict[pd.Timestamp, float] = {}

        current_symbols: list[str] = []
        reb_idx = 0
        asset_weights: dict[str, pd.Series] = {}

        for dt in dates:
            dt_str = str(dt.date())

            # Check if we should rebalance
            if reb_idx < len(rebalance_dates) and dt_str >= rebalance_dates[reb_idx]:
                current_symbols = universe_schedule[rebalance_dates[reb_idx]]
                reb_idx += 1
                # (Re)compute weights for all active symbols up to now
                for sym in current_symbols:
                    if sym in all_data and not all_data[sym].empty:
                        asset_weights[sym] = ensemble_fn(all_data[sym]["close"])

            if not current_symbols:
                continue

            day_return = 0.0
            for sym in current_symbols:
                if sym not in asset_weights:
                    continue
                w_series = asset_weights[sym]
                if dt not in w_series.index:
                    continue
                close_series = all_data[sym]["close"]
                if dt not in close_series.index:
                    continue
                loc = w_series.index.get_loc(dt)
                if loc == 0:
                    continue
                prev_dt = w_series.index[loc - 1]
                w_prev = w_series.iloc[loc - 1]
                w_curr = w_series.iloc[loc]
                c_prev = close_series.get(prev_dt, np.nan)
                c_curr = close_series.get(dt, np.nan)
                if np.isnan(c_prev) or np.isnan(c_curr) or c_prev == 0:
                    continue
                asset_ret = (c_curr - c_prev) / c_prev
                gross = (w_prev / n_assets) * asset_ret
                tc = abs(w_curr - w_prev) / n_assets * (self.cost_bps / 10_000)
                day_return += gross - tc

            port_returns[dt] = day_return

        equity = (1 + port_returns).cumprod() * self.initial_capital
        return BacktestResult(
            equity=equity,
            returns=port_returns,
            trades=all_trades,
            weights=pd.Series(combined_weights),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_trades(self, close: pd.Series, weights: pd.Series) -> list[Trade]:
        """Extract completed trades from a weight series."""
        trades: list[Trade] = []
        in_trade = False
        entry_date: pd.Timestamp | None = None
        entry_price: float = np.nan
        direction: int = 0

        for i in range(len(weights)):
            w = weights.iloc[i]
            prev_w = weights.iloc[i - 1] if i > 0 else 0.0
            c = close.iloc[i]
            dt = weights.index[i]

            was_flat = prev_w == 0.0
            is_flat = w == 0.0

            if was_flat and not is_flat:
                # New trade opened
                in_trade = True
                entry_date = dt
                entry_price = c
                direction = 1 if w > 0 else -1

            elif in_trade and is_flat:
                # Trade closed
                pnl = (c - entry_price) / entry_price * direction
                holding = (dt - entry_date).days  # type: ignore[operator]
                trades.append(
                    Trade(
                        entry_date=entry_date,  # type: ignore[arg-type]
                        exit_date=dt,
                        entry_price=entry_price,
                        exit_price=c,
                        direction=direction,
                        pnl_pct=pnl,
                        holding_days=holding,
                    )
                )
                in_trade = False

        return trades
