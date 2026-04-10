"""Structured daily JSON trade logging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class TradeLogger:
    """Write one structured JSON log file per trading day.

    Parameters
    ----------
    log_dir:
        Directory where log files are stored.  Created automatically if it
        does not exist.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(self, run_data: dict) -> Path:
        """Write *run_data* to a dated JSON file.

        If the file for today already exists (e.g. bot ran twice), a
        timestamp suffix is appended to avoid overwriting the earlier run.

        Returns
        -------
        Path
            Path of the file that was written.
        """
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        base_path = self.log_dir / f"{today}.json"

        if base_path.exists():
            ts_suffix = datetime.now(tz=timezone.utc).strftime("%H%M%S")
            log_path = self.log_dir / f"{today}_{ts_suffix}.json"
        else:
            log_path = base_path

        log_path.write_text(json.dumps(run_data, indent=2, default=str))
        return log_path

    def get_last_positions(self) -> dict[str, float]:
        """Return target weights from the most recent log file.

        Used by dry-run mode to simulate position continuity across days.
        Returns an empty dict if no previous log is found.
        """
        log_files = sorted(self.log_dir.glob("*.json"))
        if not log_files:
            return {}

        # Prefer the base dated file (no timestamp suffix) for the most
        # recent date, then fall back to any timestamped variant.
        for log_file in reversed(log_files):
            try:
                data = json.loads(log_file.read_text())
                signals = data.get("signals", {})
                positions: dict[str, float] = {}
                for symbol, info in signals.items():
                    weight = info.get("target_weight", 0.0)
                    if weight != 0.0:
                        positions[symbol] = weight
                return positions
            except Exception:  # noqa: BLE001
                continue

        return {}
