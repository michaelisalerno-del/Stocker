"""Research-lineage contract and orchestration tests for excursion events V1."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
CONTRACT = WORK_DIR / "contracts" / "20260719-cluster-invariant-excursion-events-v1.json"
PRIMARY = WORK_DIR / "artifacts" / "20260719-cluster-invariant-excursion-events-v1" / "primary"
RUNNER = WORK_DIR / "run_cluster_invariant_excursion_events_v1.py"
AUDITOR = WORK_DIR / "audit_cluster_invariant_excursion_events_v1.py"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safety_flags_are_exact() -> None:
    contract = _contract()
    assert contract["research_only"] is True
    assert contract["execution_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["broker_connected"] is False
    assert contract["economic_outcomes_used"] is False
    assert contract["payoff_selection_used"] is False
    assert contract["production_runtime_modified"] is False
    assert contract["strategy_promotion"] is False


def test_source_freeze_hashes_match() -> None:
    source = _contract()["source_identity"]
    assert isinstance(source, dict)
    for name in ("pre_run_source_identity", "pre_run_tree_manifest"):
        path = REPO_ROOT / str(source[name])
        assert path.is_file()
        assert _sha256(path) == source[f"{name}_hash"]


def test_pre_run_artifacts_precede_implementation() -> None:
    source = json.loads((PRIMARY / "pre_run_source_identity.json").read_text())
    tree = json.loads((PRIMARY / "pre_run_tree_manifest.json").read_text())
    assert source["generated_before_new_contract_or_implementation_code"] is True
    assert tree["generated_before_new_contract_or_implementation_code"] is True


def test_candidate_grid_is_bounded_and_declared() -> None:
    candidates = _contract()["candidate_registry"]
    assert isinstance(candidates, list)
    assert len(candidates) == 12
    assert {candidate["representation"] for candidate in candidates} == {"E", "P", "H"}
    assert {candidate["threshold_quantile"] for candidate in candidates} <= {
        0.80,
        0.90,
        0.95,
    }


def test_validation_is_unchanged_and_protected_2026_is_forbidden() -> None:
    chronology = _contract()["chronology"]
    assert chronology["development_and_definition_selection"] == "2024"
    assert chronology["unchanged_retrospective_validation"] == "2025"
    assert chronology["protected_2026"] == "forbidden"
    assert chronology["validation_may_change_definition_or_gate"] is False


def test_precedence_is_scientific_not_lexical() -> None:
    precedence = _contract()["resolution"]["precedence"]
    assert precedence[:5] == [
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
        "RETURN_TO_ORIGIN",
        "ROTATE_TO_NEW_REGION",
        "CONTINUE_AWAY",
    ]


def test_part_b_is_hard_gated() -> None:
    gate = _contract()["part_b_gate"]
    assert gate["part_a_decision_must_be_final"] is True
    assert gate["part_a_binding_hash_required"] is True
    assert gate["part_a_exact_rerun_required"] is True
    assert gate["part_a_independent_audit_required"] is True
    assert gate["final_part_b_metrics_before_gate"] == "forbidden"


def test_runner_does_not_import_economic_or_runtime_modules() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(
        token in module
        for module in modules
        for token in ("execution", "broker", "position_policy", "holding_policy")
    )


def test_auditor_does_not_import_primary_runner() -> None:
    tree = ast.parse(AUDITOR.read_text(encoding="utf-8"))
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "run_cluster_invariant_excursion_events_v1" not in modules


def test_required_part_a_artifacts_are_named_by_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = {
        "trajectory_feature_manifest.json",
        "emission_trajectory_ledger.parquet",
        "posterior_trajectory_ledger.parquet",
        "hybrid_trajectory_ledger.parquet",
        "unique_excursion_events.parquet",
        "event_definition_selection.json",
        "trajectory_null_results.parquet",
        "part_a_decision.json",
        "artifact_manifest.json",
    }
    assert all(name in source for name in required)
