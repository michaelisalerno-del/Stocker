#!/usr/bin/env python3
"""Run the fixed Stocker V2 market-open replay without external broker access."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from stocker_runtime.opening_burst import opening_burst_failures, run_opening_burst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, choices=(10, 60), default=60)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="stocker-opening-burst-") as temporary:
        result = run_opening_burst(Path(temporary) / "replay.sqlite3", seconds=arguments.seconds)
    failures = opening_burst_failures(result, include_performance=arguments.seconds == 60)
    result["acceptance_failures"] = list(failures)
    result["status"] = "ok" if not failures else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
