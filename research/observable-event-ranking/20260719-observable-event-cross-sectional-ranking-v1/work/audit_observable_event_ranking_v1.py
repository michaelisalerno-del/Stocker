#!/usr/bin/env python3
"""Run the independent V1 auditor without importing the runner or candidate helpers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_sources() -> Path:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    root = Path(__file__).resolve().parents[4]
    source = str(root / "packages/stocker_research/src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


if __name__ == "__main__":
    repository_root = _ensure_sources()
    from stocker_research.observable_event_ranking_v1.audit import run_independent_audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--exact-rerun", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    arguments = parser.parse_args()
    result = run_independent_audit(
        primary_dir=arguments.primary,
        exact_dir=arguments.exact_rerun,
        repository_root=repository_root,
    )
    if arguments.output_json is not None:
        arguments.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if arguments.output_markdown is not None:
        arguments.output_markdown.write_text(
            "# Independent audit\n\n"
            f"- Passed: `{str(result['audit_passed']).lower()}`\n"
            f"- Decision: `{result['decision_audited']}`\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["audit_passed"] else 1)
