#!/usr/bin/env python3
"""Run the frozen M1C quiet-state concentration audit V0."""

from __future__ import annotations

import argparse
import json

from audit_v0 import build_audit, retrospective_determinism_check, write_audit
from independent_audit_v0 import audit_primary_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="write the preregistered retrospective artifacts",
    )
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")

    result = build_audit()
    determinism = retrospective_determinism_check(result)
    if not determinism["passed"]:
        raise RuntimeError("retrospective reconstruction is not deterministic")
    write_audit(
        result,
        independent_audit={"passed": False, "status": "audit_pending"},
        determinism=determinism,
    )
    independent = audit_primary_artifacts()
    if not independent["passed"]:
        raise RuntimeError("independent retrospective audit failed closed")
    write_audit(
        result,
        independent_audit=independent,
        determinism=determinism,
    )
    summary = {
        "failed_stress_month": result.payloads["stress_month_concentration_explanation.json"][
            "failed_stress_month"
        ],
        "failed_month_share": result.payloads["stress_month_concentration_explanation.json"][
            "exact_failed_share"
        ],
        "month_concentration_explanation": result.payloads[
            "stress_month_concentration_explanation.json"
        ]["month_concentration_explanation"],
        "surprise_concentration_explanation": result.payloads[
            "surprise_concentration_explanation.json"
        ]["surprise_concentration_explanation"],
        "original_decision": result.payloads["decision.json"]["original_overall_decision"],
        "independent_audit_passed": independent["passed"],
        "determinism_passed": determinism["passed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
