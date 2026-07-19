#!/usr/bin/env python3
"""Thin executable for the bounded Regime Model Validity V2 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from regime_validity_pipeline_v2 import (  # noqa: E402
    BASELINE_SHA,
    EXPECTED_CONTRACT_HASH,
    PART_A_REQUIRED_ARTIFACTS,
    SEEDS,
    PartAGateEvidence,
    _causal_filter_summary_compiled,
    _markdown_table,
    decide_part_a,
    main,
    safety_flags,
)

__all__ = (
    "BASELINE_SHA",
    "EXPECTED_CONTRACT_HASH",
    "PART_A_REQUIRED_ARTIFACTS",
    "SEEDS",
    "PartAGateEvidence",
    "_causal_filter_summary_compiled",
    "_markdown_table",
    "decide_part_a",
    "main",
    "safety_flags",
)


if __name__ == "__main__":
    main()
