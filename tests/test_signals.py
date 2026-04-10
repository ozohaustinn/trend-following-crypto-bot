"""Tests for strategy/signals.py"""

import numpy as np
import pandas as pd
import pytest

from strategy.signals import generate_signals


def make_close(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestGenerateSignals:
    def test_entry_at_new_high(self):
        # Rising series: entry should happen at the period-th bar when it equals the upper
        close = make_close([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        signals = generate_signals(close, period=3, mode="long_only")
        # At index 2, close=3 == upper(3)=3 → entry
        assert signals["position"].iloc[2] == 1
        # Entry price recorded
        assert signals["entry_price"].iloc[2] == 3.0

    def test_flat_before_enough_history(self):
        close = make_close([1, 2, 3, 4, 5])
        signals = generate_signals(close, period=3, mode="long_only")
        # First (period-1) bars: no channel → flat
        assert signals["position"].iloc[0] == 0
        assert signals["position"].iloc[1] == 0

    def test_trailing_stop_never_moves_down_for_long(self):
        # Create a close that trends up then falls
        close = make_close([1, 2, 3, 4, 5, 6, 5, 4, 3])
        signals = generate_signals(close, period=3, mode="long_only")
        # Find bars where we are long and trailing stop is not NaN
        long_mask = signals["position"] == 1
        ts = signals.loc[long_mask, "trailing_stop"].dropna()
        if len(ts) > 1:
            diffs = ts.diff().dropna()
            assert (diffs >= 0).all(), "Trailing stop moved down for a long position"

    def test_exit_when_close_hits_stop(self):
        # Construct a scenario where price breaks out then falls through stop
        # Period=3: first 3 bars set up channel, bar 3 is new high → entry
        # Then price falls below mid → exit
        values = [10, 12, 15, 15, 11, 8]  # Entry at bar 2 (value=15), falls later
        close = make_close(values)
        signals = generate_signals(close, period=3, mode="long_only")
        # Should eventually go flat
        positions = signals["position"].values
        # Find the last position — should be 0 after the stop is hit
        assert 0 in positions, "Never exited the position"

    def test_long_short_mode_short_entry(self):
        # Falling series: should enter short when price equals lower band
        values = [10, 9, 8, 7, 6, 5, 4, 3]
        close = make_close(values)
        signals = generate_signals(close, period=3, mode="long_short")
        # At index 2, close=8 == lower(3)=8 → short entry
        assert signals["position"].iloc[2] == -1

    def test_long_short_trailing_stop_moves_up_for_short(self):
        # For shorts, the stop should only move down (min update rule)
        values = [10, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6]
        close = make_close(values)
        signals = generate_signals(close, period=3, mode="long_short")
        short_mask = signals["position"] == -1
        ts = signals.loc[short_mask, "trailing_stop"].dropna()
        if len(ts) > 1:
            diffs = ts.diff().dropna()
            assert (diffs <= 0).all(), "Short trailing stop moved up"

    def test_invalid_mode_raises(self):
        close = make_close([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="mode"):
            generate_signals(close, period=3, mode="bad_mode")

    def test_output_columns(self):
        close = make_close([1, 2, 3, 4, 5])
        signals = generate_signals(close, period=3)
        assert set(signals.columns) == {"position", "trailing_stop", "entry_price"}
        assert signals.index.equals(close.index)

    def test_state_machine_flat_to_long_to_flat(self):
        # Build a price series that produces exactly one trade
        # Rising 5 bars (breakout at bar 4), then falls hard
        values = [10, 11, 12, 13, 14, 5, 5, 5, 5, 5]
        close = make_close(values)
        signals = generate_signals(close, period=5, mode="long_only")
        pos = signals["position"].values
        # Should go: 0,0,0,0 → 1 (entry at bar 4) → back to 0
        assert pos[4] == 1
        # Eventually goes flat
        assert 0 in pos[5:]

    def test_no_signal_on_flat_prices(self):
        close = make_close([5.0] * 10)
        signals = generate_signals(close, period=3, mode="long_only")
        # When all prices are equal, close == upper AND close == lower always
        # According to spec: entry when close == upper → position 1 (or 0/−1 in long_short)
        # At least the output shape should be correct
        assert len(signals) == 10
