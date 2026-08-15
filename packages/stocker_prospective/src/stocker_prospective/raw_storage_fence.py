"""Cross-process exclusion for raw-file writes and retention finalization."""

from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class RawStorageFence:
    """Serialize a raw partition write+manifest commit with session retirement."""

    def __init__(self, raw_root: str | Path, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("raw storage fence timeout must be positive")
        self.raw_root = Path(raw_root)
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        self.raw_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.raw_root / ".raw-storage-retention.lock"
        with lock_path.open("a+b") as handle:
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("RAW_STORAGE_FENCE_TIMEOUT") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
