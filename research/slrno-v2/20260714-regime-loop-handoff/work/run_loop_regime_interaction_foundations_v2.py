#!/usr/bin/env python3
"""Create only the blocked Part B contract surface authorized by Part A.

No source population is opened and no interaction statistic or model is
computed.  The runner exits successfully only after proving the Part A gate is
closed and writing the allowed schema-only scaffold and blocker report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stocker_research.loop_regime_interaction_v2 import (  # noqa: E402
    PartAGateState,
    PartBGateClosedError,
    assert_part_b_scoring_authorized,
    population_scaffold,
)
from stocker_research.regime_validity_v2 import safety_flags  # noqa: E402

PART_A_PRIMARY = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "primary"
CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-loop-regime-interaction-foundations-v2.json"
EXPECTED_CONTRACT_HASH = "f8c063f225af9543c76ab4a662b761ca5ad785303a2b6c2c2a15168715ac624c"
DEFAULT_OUTPUT = (
    WORK_DIR / "artifacts" / "20260718-loop-regime-interaction-foundations-v2" / "blocked"
)
ALLOWED_OUTPUT_FILES = frozenset({"part_b_population_scaffold.json", "part_b_blocker_report.json"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n")


def load_gate_state() -> PartAGateState:
    """Load and cross-check the independently audited Part A identities."""

    if _sha256_file(CONTRACT_PATH) != EXPECTED_CONTRACT_HASH:
        raise RuntimeError("proposed Part B contract identity changed")
    contract = _read_json(CONTRACT_PATH)
    bound = contract["frozen_part_a_binding"]
    decision_path = PART_A_PRIMARY / "part_a_decision.json"
    audit_path = PART_A_PRIMARY / "independent_audit.json"
    exact_path = PART_A_PRIMARY / "exact_rerun_manifest.json"
    decision = _read_json(decision_path)
    audit = _read_json(audit_path)
    exact = _read_json(exact_path)
    actual_hashes = {
        "part_a_decision_file_hash": _sha256_file(decision_path),
        "part_a_independent_audit_file_hash": _sha256_file(audit_path),
        "part_a_exact_rerun_manifest_file_hash": _sha256_file(exact_path),
    }
    if any(bound[key] != value for key, value in actual_hashes.items()):
        raise RuntimeError("proposed Part B contract does not match frozen Part A files")
    if decision["decision"] != bound["part_a_decision"]:
        raise RuntimeError("Part A decision identity changed after contract proposal")
    if decision["binding"]["binding_hash"] != bound["part_a_binding_hash"]:
        raise RuntimeError("Part A frozen binding identity changed")
    if audit["status"] != bound["part_a_independent_audit_status"]:
        raise RuntimeError("Part A independent audit status changed")
    if exact["byte_identical"] is not bound["part_a_exact_rerun_byte_identical"]:
        raise RuntimeError("Part A exact-rerun status changed")
    return PartAGateState(
        decision=str(decision["decision"]),
        decision_file_hash=actual_hashes["part_a_decision_file_hash"],
        binding_hash=str(decision["binding"]["binding_hash"]),
        state_model_hash=str(decision["binding"]["state_model_hash"]),
        state_alignment_hash=str(decision["binding"]["state_alignment_hash"]),
        independent_audit_status=str(audit["status"]),
        independent_audit_file_hash=actual_hashes["part_a_independent_audit_file_hash"],
        exact_rerun_byte_identical=bool(exact["byte_identical"]),
        exact_rerun_manifest_file_hash=actual_hashes["part_a_exact_rerun_manifest_file_hash"],
    )


def run(output: Path) -> dict[str, Any]:
    gate = load_gate_state()
    gate_error: str | None = None
    try:
        assert_part_b_scoring_authorized(gate)
    except PartBGateClosedError as error:
        gate_error = str(error)
    if gate_error is None:
        raise RuntimeError(
            "this bounded runner is for a closed gate and must not access authorized Part B data"
        )

    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir() if path.is_file()}
    unexpected = existing.difference(ALLOWED_OUTPUT_FILES)
    if unexpected:
        raise RuntimeError(f"blocked Part B output contains forbidden files: {sorted(unexpected)}")

    contract_hash = _sha256_file(CONTRACT_PATH)
    scaffold = population_scaffold(gate, proposed_contract_hash=contract_hash)
    scaffold_path = output / "part_b_population_scaffold.json"
    _write_json(scaffold_path, scaffold)
    blocker: dict[str, Any] = {
        "blocker_report_version": "loop_regime_interaction_foundations_v2",
        "decision": "loop_regime_interaction_experiment_blocked",
        "blocking_part_a_decision": gate.decision,
        "blocker": gate_error,
        "part_b_authorized": False,
        "part_b_scoring_accessed": False,
        "population_rows_read": 0,
        "interaction_results_inspected": False,
        "interaction_models_fit": 0,
        "part_b_report_created": False,
        "semantic_dictionary_may_proceed": False,
        "next_loop_forecast_justified": False,
        "proposed_contract_hash": contract_hash,
        "part_a_decision_file_hash": gate.decision_file_hash,
        "part_a_binding_hash": gate.binding_hash,
        "part_a_independent_audit_file_hash": gate.independent_audit_file_hash,
        "part_a_exact_rerun_manifest_file_hash": gate.exact_rerun_manifest_file_hash,
        "population_scaffold_hash": _sha256_file(scaffold_path),
        "required_repair": (
            "right-censored state-duration refit with a fully archived deterministic "
            "panel builder and row order, followed by a complete Part A rerun"
        ),
        **safety_flags(),
    }
    blocker["blocker_report_hash"] = _sha256_payload(blocker)
    _write_json(output / "part_b_blocker_report.json", blocker)
    final_files = {path.name for path in output.iterdir() if path.is_file()}
    if final_files != ALLOWED_OUTPUT_FILES:
        raise AssertionError("blocked Part B runner emitted an unauthorized artifact family")
    return blocker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
