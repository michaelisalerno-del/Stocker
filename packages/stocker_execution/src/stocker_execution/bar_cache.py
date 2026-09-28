"""Compact completed Saxo bars; no raw quotes or synthetic volume in this cache."""

import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from stocker_execution.config import RecorderConfig
from stocker_execution.contracts import key
from stocker_execution.rules import Bar


class BarCache:
    def __init__(self, directory: Path, config: RecorderConfig):
        self.directory, self.config = directory, config
        self.problem = ""
        self.size = 0

    def save(self, identity: dict[str, Any], bars: list[Bar], version: Any) -> None:
        if not self.config.persistent_capture:
            self.problem = "RECORDING_PERMISSION_NOT_VERIFIED"
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for temporary in self.directory.glob("*.tmp"):
            temporary.unlink()
        self.size = sum(p.stat().st_size for p in self.directory.glob("*.gz"))
        days = sorted({b.at.date().isoformat() for b in bars})
        for day in days:
            name = hashlib.sha256((key(identity) + day).encode()).hexdigest()
            path = self.directory / (name + ".json.gz")
            previous = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
            # A DataVersion change requires replacement, never merging revised versions.
            rows = previous.get("bars", {}) if previous.get("data_version") == version else {}
            rows.update(
                {
                    b.at.isoformat(): {
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                    if b.at.date().isoformat() == day
                }
            )
            value = {
                "identity": identity,
                "day": day,
                "data_version": version,
                "source": "SAXO_CHART_COMPLETED_1M",
                "bars": rows,
            }
            blob = gzip.compress(
                json.dumps(value, separators=(",", ":"), allow_nan=False).encode(),
                compresslevel=6,
                mtime=0,
            )
            before = path.stat().st_size if path.exists() else 0
            if (
                self.size + len(blob) > self.config.bar_max_bytes
                or shutil.disk_usage(self.directory).free - len(blob)
                < self.config.disk_reserve_bytes
            ):
                self.problem = "BAR_STORAGE_LIMIT_REACHED"
                return
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as out:
                out.write(blob)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, path)
            self.size += len(blob) - before
        self.problem = ""
