#!/usr/bin/env python3
"""Run one explicit Observable Event Ranking V1 stage."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_sources() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    root = Path(__file__).resolve().parents[4]
    for relative in (
        "packages/stocker_research/src",
        "packages/stocker_execution/src",
        "packages/stocker_data/src",
        "packages/stocker_core/src",
    ):
        source = str(root / relative)
        if source not in sys.path:
            sys.path.insert(0, source)


if __name__ == "__main__":
    _ensure_sources()
    from stocker_research.observable_event_ranking_v1.cli import main

    raise SystemExit(main())
