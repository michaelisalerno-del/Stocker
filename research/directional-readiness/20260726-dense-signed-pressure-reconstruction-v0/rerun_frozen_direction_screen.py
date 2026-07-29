#!/usr/bin/env python3
"""Gate the frozen quiet-accumulation rerun on the exact Phase 1 decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[2]
PACKAGE_SOURCE = ROOT / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.dense_signed_pressure_v0 import (  # noqa: E402
    assert_phase2_authorized,
)

DEFAULT_PHASE1_DECISION = EXPERIMENT / "artifacts/primary/phase1_decision.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-decision",
        type=Path,
        default=DEFAULT_PHASE1_DECISION,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    decision = json.loads(arguments.phase1_decision.read_text(encoding="utf-8"))
    primary = str(decision["primary_decision"])
    try:
        assert_phase2_authorized(primary)
    except RuntimeError:
        print(
            json.dumps(
                {
                    "phase1_decision": primary,
                    "phase2_authorized": False,
                    "phase2_executed": False,
                    "reason": "frozen directional rerun prohibited by Phase 1 gate",
                },
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError(
        "Phase 1 unexpectedly passed; this run must use the unchanged frozen predecessor "
        "screen with the validated dense surface"
    )


if __name__ == "__main__":
    raise SystemExit(main())
