"""Multi-asset portfolio manager with monthly rotation."""

from __future__ import annotations

import pandas as pd

import config
from data.universe import check_exit_conditions, select_universe
from strategy.ensemble import ensemble_weights


class PortfolioManager:
    """Manage a rotating multi-asset portfolio.

    Parameters
    ----------
    n_assets:
        Number of assets to hold each month (default from config).
    rebalance_freq:
        Pandas offset alias for rebalancing frequency (default ``"ME"``
        which is month-end).
    mode:
        Signal mode: ``"long_only"`` or ``"long_short"``.
    lookback_periods:
        Donchian lookback periods for the ensemble.
    vol_target, vol_window, max_leverage, rebalance_threshold:
        Sizing parameters forwarded to the ensemble.
    """

    def __init__(
        self,
        n_assets: int = config.UNIVERSE_SIZE,
        rebalance_freq: str = "ME",
        mode: str = "long_only",
        lookback_periods: list[int] = config.LOOKBACK_PERIODS,
        vol_target: float = config.VOL_TARGET,
        vol_window: int = config.VOL_WINDOW,
        max_leverage: float = config.MAX_LEVERAGE,
        rebalance_threshold: float = config.REBALANCE_THRESHOLD,
    ) -> None:
        self.n_assets = n_assets
        self.rebalance_freq = rebalance_freq
        self.mode = mode
        self.lookback_periods = lookback_periods
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.rebalance_threshold = rebalance_threshold

    def _ensemble_fn(self, close: pd.Series) -> pd.Series:
        return ensemble_weights(
            close,
            lookback_periods=self.lookback_periods,
            mode=self.mode,
            vol_target=self.vol_target,
            vol_window=self.vol_window,
            max_leverage=self.max_leverage,
            rebalance_threshold=self.rebalance_threshold,
        )

    def run(
        self,
        all_data: dict[str, pd.DataFrame],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Execute the full rotational trend-following programme.

        For each month:

        1. Select universe (top *n_assets* by trailing 30-day volume).
        2. Allocate 1/N capital to each asset.
        3. Apply ensemble trend strategy independently per asset.
        4. Aggregate daily returns across all assets.
        5. Check mid-month exit conditions (remove stale/illiquid assets).

        Parameters
        ----------
        all_data:
            Mapping ``{symbol: ohlcv_df}`` covering at least *start* to *end*.
        start, end:
            Date range for the backtest.

        Returns
        -------
        pd.DataFrame
            Daily portfolio-level returns with columns ``["return"]`` and an
            additional ``"universe"`` column (list of active symbols) for
            inspection.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Generate month-end rebalance dates within the range
        rebalance_dates = pd.date_range(start_ts, end_ts, freq=self.rebalance_freq)

        # Build universe schedule and precompute ensemble weights per asset
        universe_schedule: dict[str, list[str]] = {}
        asset_weights: dict[str, pd.Series] = {}

        for reb_date in rebalance_dates:
            reb_str = str(reb_date.date())
            selected = select_universe(all_data, reb_str, n_assets=self.n_assets)
            universe_schedule[reb_str] = selected
            for sym in selected:
                if sym not in asset_weights and sym in all_data:
                    asset_weights[sym] = self._ensemble_fn(all_data[sym]["close"])

        # Build a full daily date range
        all_dates: set[pd.Timestamp] = set()
        for df in all_data.values():
            subset = df.loc[start_ts:end_ts]
            all_dates.update(subset.index)
        dates = sorted(all_dates)

        daily_returns: list[float] = []
        daily_universe: list[list[str]] = []

        rebalance_list = sorted(universe_schedule.keys())
        reb_idx = 0
        current_symbols: list[str] = []

        for dt in dates:
            dt_str = str(dt.date())

            # Advance the universe when a new rebalance date is reached
            while reb_idx < len(rebalance_list) and dt_str >= rebalance_list[reb_idx]:
                current_symbols = universe_schedule[rebalance_list[reb_idx]]
                reb_idx += 1

            # Mid-month exit filter
            active_symbols = [
                sym
                for sym in current_symbols
                if sym in all_data
                and not check_exit_conditions(all_data[sym], dt_str)
            ]

            day_return = 0.0
            n_active = len(active_symbols) if active_symbols else 1

            for sym in active_symbols:
                if sym not in asset_weights:
                    continue
                w_series = asset_weights[sym]
                close_series = all_data[sym]["close"]

                if dt not in w_series.index or dt not in close_series.index:
                    continue
                loc = w_series.index.get_loc(dt)
                if loc == 0:
                    continue

                w_prev = w_series.iloc[loc - 1]
                w_curr = w_series.iloc[loc]
                prev_dt = w_series.index[loc - 1]
                c_prev = close_series.get(prev_dt)
                c_curr = close_series.get(dt)

                if c_prev is None or c_curr is None or c_prev == 0:
                    continue

                asset_ret = (c_curr - c_prev) / c_prev
                gross = (w_prev / n_active) * asset_ret
                tc = abs(w_curr - w_prev) / n_active * (config.TRANSACTION_COST_BPS / 10_000)
                day_return += gross - tc

            daily_returns.append(day_return)
            daily_universe.append(active_symbols)

        idx = pd.DatetimeIndex(dates)
        result = pd.DataFrame(
            {"return": daily_returns, "universe": daily_universe},
            index=idx,
        )
        return result
