"""Tests for bot/logger.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.logger import TradeLogger


@pytest.fixture
def tmp_logger(tmp_path):
    return TradeLogger(log_dir=str(tmp_path))


class TestTradeLogger:
    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        assert not log_dir.exists()
        TradeLogger(log_dir=str(log_dir))
        assert log_dir.exists()

    def test_log_run_creates_dated_file(self, tmp_logger, tmp_path):
        run_data = {"timestamp": "2026-04-10T00:05:00Z", "mode": "dry"}
        path = tmp_logger.log_run(run_data)
        assert path.exists()
        assert path.suffix == ".json"

    def test_log_run_file_is_valid_json(self, tmp_logger):
        run_data = {"timestamp": "2026-04-10T00:05:00Z", "account_equity": 500.0}
        path = tmp_logger.log_run(run_data)
        loaded = json.loads(path.read_text())
        assert loaded["account_equity"] == 500.0

    def test_second_run_does_not_overwrite(self, tmp_logger, tmp_path):
        run1 = {"timestamp": "first", "mode": "dry"}
        run2 = {"timestamp": "second", "mode": "dry"}
        path1 = tmp_logger.log_run(run1)
        path2 = tmp_logger.log_run(run2)
        # Both files must exist and be distinct
        assert path1.exists()
        assert path2.exists()
        assert path1 != path2

    def test_second_run_uses_timestamp_suffix(self, tmp_logger, tmp_path):
        tmp_logger.log_run({"x": 1})
        path2 = tmp_logger.log_run({"x": 2})
        # The second file should have a longer name (timestamp suffix)
        assert len(path2.stem) > 10  # more than just "YYYY-MM-DD"

    def test_get_last_positions_empty_when_no_logs(self, tmp_logger):
        positions = tmp_logger.get_last_positions()
        assert positions == {}

    def test_get_last_positions_reads_target_weights(self, tmp_logger):
        run_data = {
            "timestamp": "2026-04-10T00:05:00Z",
            "signals": {
                "BTC/USDT": {"target_weight": 0.05, "current_weight": 0.0},
                "ETH/USDT": {"target_weight": 0.0, "current_weight": 0.0},
            },
        }
        tmp_logger.log_run(run_data)
        positions = tmp_logger.get_last_positions()
        assert "BTC/USDT" in positions
        assert positions["BTC/USDT"] == pytest.approx(0.05)
        # Zero-weight assets should be excluded
        assert "ETH/USDT" not in positions

    def test_get_last_positions_handles_corrupt_log(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        bad_file = log_dir / "2026-04-09.json"
        bad_file.write_text("not valid json {{{{")
        good_file = log_dir / "2026-04-10.json"
        good_file.write_text(
            json.dumps(
                {"signals": {"SOL/USDT": {"target_weight": 0.03}}}
            )
        )
        logger = TradeLogger(log_dir=str(log_dir))
        positions = logger.get_last_positions()
        # Should recover the good file
        assert "SOL/USDT" in positions
