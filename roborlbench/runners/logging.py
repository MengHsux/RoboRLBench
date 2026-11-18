"""Minimal logging: a CSV writer plus an optional TensorBoard writer."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

__all__ = ["RunLogger"]


class RunLogger:
    """Writes ``metrics.csv``, ``config.json`` and (optionally) TensorBoard."""

    def __init__(self, run_dir: str, use_tensorboard: bool = False) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._fields: Optional[list] = None
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=[])
        self._tb = None
        if use_tensorboard:
            try:                                    # pragma: no cover - optional
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=run_dir)
            except Exception:
                self._tb = None

    # ------------------------------------------------------------------ #
    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": int(step)}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
        if self._fields is None:
            self._fields = list(row)
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            self._writer.writeheader()
        else:
            missing = [k for k in row if k not in self._fields]
            if missing:
                # widen the header once; keep earlier rows readable
                self._fields.extend(missing)
                self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            row = {k: row.get(k, "") for k in self._fields}
        self._writer.writerow(row)
        self._file.flush()
        if self._tb is not None:
            for key, value in row.items():
                if key != "step":
                    self._tb.add_scalar(key, value, step)

    # ------------------------------------------------------------------ #
    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._tb is not None:                    # pragma: no cover
            self._tb.close()
            self._tb = None
