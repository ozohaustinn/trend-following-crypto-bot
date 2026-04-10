"""Tests for strategy/donchian.py"""

import numpy as np
import pandas as pd
import pytest

from strategy.donchian import donchian_channel


def make_close(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestDonchianChannel:
    def test_known_values(self):
        close = make_close([1, 2, 3, 4, 5, 4, 3, 2, 1])
        dc = donchian_channel(close, period=3)
        # Window [3,4,5]: upper=5, lower=3, mid=4
        assert dc["upper"].iloc[4] == 5.0
        assert dc["lower"].iloc[4] == 3.0
        assert dc["mid"].iloc[4] == 4.0

    def test_initial_nans(self):
        close = make_close([1, 2, 3, 4, 5])
        dc = donchian_channel(close, period=3)
        # First (period-1) rows should be NaN
        assert dc["upper"].iloc[0] != dc["upper"].iloc[0]  # NaN check
        assert dc["upper"].iloc[1] != dc["upper"].iloc[1]
        assert not np.isnan(dc["upper"].iloc[2])

    def test_all_equal_prices(self):
        close = make_close([5.0] * 10)
        dc = donchian_channel(close, period=5)
        assert dc["upper"].iloc[4] == 5.0
        assert dc["lower"].iloc[4] == 5.0
        assert dc["mid"].iloc[4] == 5.0

    def test_single_bar_period1(self):
        close = make_close([10.0, 20.0, 30.0])
        dc = donchian_channel(close, period=1)
        assert dc["upper"].iloc[0] == 10.0
        assert dc["lower"].iloc[0] == 10.0
        assert dc["mid"].iloc[0] == 10.0

    def test_upper_lower_relationship(self):
        rng = np.random.default_rng(42)
        close = make_close(rng.uniform(100, 200, 100))
        dc = donchian_channel(close, period=20)
        valid = dc.dropna()
        assert (valid["upper"] >= valid["lower"]).all()
        assert ((valid["upper"] + valid["lower"]) / 2 - valid["mid"]).abs().max() < 1e-10

    def test_nan_in_input_propagates(self):
        close = make_close([1, 2, np.nan, 4, 5])
        dc = donchian_channel(close, period=3)
        # NaN in window should make upper/lower NaN
        assert np.isnan(dc["upper"].iloc[2])

    def test_returns_dataframe_with_correct_columns(self):
        close = make_close([1, 2, 3])
        dc = donchian_channel(close, period=2)
        assert set(dc.columns) == {"upper", "lower", "mid"}
        assert dc.index.equals(close.index)

    def test_period_equals_length(self):
        close = make_close([3, 1, 4, 1, 5])
        dc = donchian_channel(close, period=5)
        assert np.isnan(dc["upper"].iloc[3])
        assert dc["upper"].iloc[4] == 5.0
        assert dc["lower"].iloc[4] == 1.0
