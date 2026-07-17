"""Execution-free append-only forecast and settlement ledgers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


class ProspectiveLedger:
    """Separate hash-chained forecast and settlement streams."""

    def __init__(self, root: Path, *, opened_through: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.forecast_path = root / "forecast_ledger.jsonl"
        self.settlement_path = root / "settlement_ledger.jsonl"
        self.opened_through = pd.Timestamp(opened_through).date()

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        existing = ProspectiveLedger._records(path)
        previous_hash = existing[-1]["record_hash"] if existing else None
        payload = {**record, "previous_hash": previous_hash}
        digest_input = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["record_hash"] = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_forecast(self, record: dict[str, Any]) -> None:
        if record.get("research_only") is not True or record.get("execution_enabled") is not False:
            raise ValueError("forecast must be research-only with execution disabled")
        session = pd.Timestamp(str(record["session"])).date()
        if session <= self.opened_through:
            raise ValueError("prospective mode rejects opened historical snapshots")
        opportunity_id = str(record["opportunity_id"])
        if any(
            str(row["opportunity_id"]) == opportunity_id
            for row in self._records(self.forecast_path)
        ):
            raise FileExistsError(f"duplicate prospective opportunity_id: {opportunity_id}")
        self._append(self.forecast_path, record)

    def append_settlement(self, record: dict[str, Any]) -> None:
        opportunity_id = str(record["opportunity_id"])
        forecasts = self._records(self.forecast_path)
        if not any(str(row["opportunity_id"]) == opportunity_id for row in forecasts):
            raise ValueError("settlement requires an existing immutable forecast")
        if any(
            str(row["opportunity_id"]) == opportunity_id
            for row in self._records(self.settlement_path)
        ):
            raise FileExistsError(f"duplicate settlement opportunity_id: {opportunity_id}")
        self._append(self.settlement_path, record)
