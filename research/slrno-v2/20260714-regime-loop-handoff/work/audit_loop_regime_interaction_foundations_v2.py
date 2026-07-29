#!/usr/bin/env python3
"""Independently audit the fail-closed Part B scaffold without scoring data."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PART_A_PRIMARY = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "primary"
PART_B_BLOCKED = (
    WORK_DIR / "artifacts" / "20260718-loop-regime-interaction-foundations-v2" / "blocked"
)
CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-loop-regime-interaction-foundations-v2.json"
PART_B_REPORT = WORK_DIR / "reports" / "20260718-loop-regime-interaction-foundations-v2.md"
AUTHORIZED_DECISIONS = {
    "regime_representation_validated_for_loop_dictionary",
    "regime_representation_valid_with_required_sensitivity",
    "hierarchical_market_stock_regime_representation_preferred",
}
ALLOWED_FILES = {"part_b_population_scaffold.json", "part_b_blocker_report.json"}
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}


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


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def audit() -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    scaffold = _read_json(PART_B_BLOCKED / "part_b_population_scaffold.json")
    blocker = _read_json(PART_B_BLOCKED / "part_b_blocker_report.json")
    decision_path = PART_A_PRIMARY / "part_a_decision.json"
    audit_path = PART_A_PRIMARY / "independent_audit.json"
    exact_path = PART_A_PRIMARY / "exact_rerun_manifest.json"
    decision = _read_json(decision_path)
    part_a_audit = _read_json(audit_path)
    exact = _read_json(exact_path)
    binding = contract["frozen_part_a_binding"]

    checks: dict[str, bool] = {}
    checks["only_allowed_blocked_outputs"] = {
        path.name for path in PART_B_BLOCKED.iterdir() if path.is_file()
    } == ALLOWED_FILES
    checks["part_a_file_hashes_bound"] = (
        binding["part_a_decision_file_hash"] == _sha256_file(decision_path)
        and binding["part_a_independent_audit_file_hash"] == _sha256_file(audit_path)
        and binding["part_a_exact_rerun_manifest_file_hash"] == _sha256_file(exact_path)
    )
    checks["part_a_identity_bound"] = (
        decision["decision"] == binding["part_a_decision"]
        and decision["binding"]["binding_hash"] == binding["part_a_binding_hash"]
        and decision["binding"]["state_model_hash"] == binding["state_model_hash"]
        and decision["binding"]["state_alignment_hash"] == binding["state_alignment_hash"]
    )
    checks["part_a_audit_and_rerun_pass"] = (
        part_a_audit["status"] == "pass"
        and part_a_audit["independent_audit_reproducible"] is True
        and exact["byte_identical"] is True
    )
    checks["gate_is_closed"] = (
        decision["decision"] not in AUTHORIZED_DECISIONS
        and decision["part_b_authorized"] is False
        and contract["access_gate"]["current_decision_authorizes_scoring"] is False
    )
    contract_hash = _sha256_file(CONTRACT_PATH)
    checks["contract_hash_propagated"] = (
        scaffold["proposed_contract_hash"] == contract_hash
        and blocker["proposed_contract_hash"] == contract_hash
    )
    checks["population_remained_unopened"] = (
        scaffold["population_rows_read"] == 0
        and scaffold["interaction_results_inspected"] is False
        and scaffold["interaction_models_fit"] == 0
        and blocker["population_rows_read"] == 0
        and blocker["part_b_scoring_accessed"] is False
        and blocker["interaction_results_inspected"] is False
        and blocker["interaction_models_fit"] == 0
    )
    checks["blocker_decision"] = (
        blocker["decision"] == "loop_regime_interaction_experiment_blocked"
        and blocker["blocking_part_a_decision"] == decision["decision"]
        and blocker["part_b_report_created"] is False
        and blocker["semantic_dictionary_may_proceed"] is False
        and blocker["next_loop_forecast_justified"] is False
    )
    checks["scaffold_is_schema_only"] = scaffold[
        "status"
    ] == "schema_only_part_b_scoring_not_authorized" and set(scaffold["schema_groups"]) == {
        "identity_and_provenance",
        "loop_structure",
        "current_regime",
        "regime_history",
        "market_context",
        "first_event_target",
    }
    stored_blocker_hash = str(blocker["blocker_report_hash"])
    blocker_without_hash = dict(blocker)
    del blocker_without_hash["blocker_report_hash"]
    checks["blocker_hash"] = stored_blocker_hash == _sha256_payload(blocker_without_hash)
    checks["scaffold_hash"] = blocker["population_scaffold_hash"] == _sha256_file(
        PART_B_BLOCKED / "part_b_population_scaffold.json"
    )
    checks["safety_flags"] = all(
        all(payload.get(key) == expected for key, expected in SAFETY_FLAGS.items())
        for payload in (contract, scaffold, blocker)
    )
    checks["part_b_report_absent"] = not PART_B_REPORT.exists()

    source_paths = (
        REPO_ROOT
        / "packages"
        / "stocker_research"
        / "src"
        / "stocker_research"
        / "loop_orientation_v2.py",
        REPO_ROOT
        / "packages"
        / "stocker_research"
        / "src"
        / "stocker_research"
        / "loop_regime_interaction_v2.py",
        WORK_DIR / "run_loop_regime_interaction_foundations_v2.py",
    )
    imports = set().union(*(_import_names(path) for path in source_paths))
    forbidden_import_tokens = ("broker", "execution", "order", "position", "runtime")
    checks["no_trading_runtime_imports"] = not any(
        token in imported.lower() for imported in imports for token in forbidden_import_tokens
    )

    status = "pass" if all(checks.values()) else "fail"
    payload: dict[str, Any] = {
        "audit_version": "loop_regime_interaction_foundations_v2_blocked_audit",
        "status": status,
        "checks": checks,
        "part_a_decision": decision["decision"],
        "part_b_scoring_accessed": False,
        "population_rows_read": 0,
        "proposed_contract_hash": contract_hash,
        **SAFETY_FLAGS,
    }
    print(json.dumps(payload, sort_keys=True))
    if status != "pass":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise SystemExit(f"blocked Part B audit failed: {failed}")
    return payload


if __name__ == "__main__":
    audit()
