from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from stocker_research.loop_regime_interaction_v2 import PartBGateClosedError

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
RUNNER = WORK_DIR / "run_loop_regime_interaction_foundations_v2.py"
AUDITOR = WORK_DIR / "audit_loop_regime_interaction_foundations_v2.py"
CONTRACT = WORK_DIR / "contracts" / "20260718-loop-regime-interaction-foundations-v2.json"
PART_A = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "primary"
PART_B_REPORT = WORK_DIR / "reports" / "20260718-loop-regime-interaction-foundations-v2.md"
SAFETY_FLAGS = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}


def _load_runner() -> object:
    package_root = REPO_ROOT / "packages" / "stocker_research" / "src"
    for path in (package_root, WORK_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("loop_regime_blocked_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER_MODULE = _load_runner()


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_proposed_contract_is_hash_frozen_and_structural_only() -> None:
    contract = _read(CONTRACT)
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        RUNNER_MODULE.EXPECTED_CONTRACT_HASH
    )
    assert contract["contract_status"] == "proposed_blocked_scoring_not_open"
    for key, expected in SAFETY_FLAGS.items():
        assert contract[key] == expected


def test_contract_hash_binds_independently_audited_part_a() -> None:
    contract = _read(CONTRACT)
    binding = contract["frozen_part_a_binding"]
    paths = {
        "part_a_decision_file_hash": PART_A / "part_a_decision.json",
        "part_a_independent_audit_file_hash": PART_A / "independent_audit.json",
        "part_a_exact_rerun_manifest_file_hash": PART_A / "exact_rerun_manifest.json",
    }
    for field, path in paths.items():
        assert binding[field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["part_a_decision"] == "regime_representation_requires_targeted_repair"
    assert binding["part_a_independent_audit_status"] == "pass"
    assert binding["part_a_exact_rerun_byte_identical"] is True


def test_part_a_gate_closes_before_any_part_b_source_access() -> None:
    gate = RUNNER_MODULE.load_gate_state()
    assert gate.scoring_authorized is False
    with pytest.raises(PartBGateClosedError):
        RUNNER_MODULE.assert_part_b_scoring_authorized(gate)


def test_blocked_runner_emits_only_scaffold_and_blocker(tmp_path: Path) -> None:
    blocker = RUNNER_MODULE.run(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "part_b_population_scaffold.json",
        "part_b_blocker_report.json",
    }
    scaffold = _read(tmp_path / "part_b_population_scaffold.json")
    assert scaffold["population_rows_read"] == 0
    assert scaffold["interaction_results_inspected"] is False
    assert scaffold["interaction_models_fit"] == 0
    assert blocker["decision"] == "loop_regime_interaction_experiment_blocked"
    assert blocker["part_b_scoring_accessed"] is False
    assert blocker["part_b_report_created"] is False


def test_blocked_artifacts_have_every_safety_flag(tmp_path: Path) -> None:
    RUNNER_MODULE.run(tmp_path)
    for filename in ("part_b_population_scaffold.json", "part_b_blocker_report.json"):
        payload = _read(tmp_path / filename)
        for key, expected in SAFETY_FLAGS.items():
            assert payload[key] == expected


def test_population_schema_retains_corrected_target_and_distinct_age_fields(
    tmp_path: Path,
) -> None:
    RUNNER_MODULE.run(tmp_path)
    schema = _read(tmp_path / "part_b_population_scaffold.json")["schema_groups"]
    assert set(schema["first_event_target"]) >= {
        "OTHER_PRIMITIVE_LOOP",
        "NO_LOOP_WITHIN_HORIZON",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }
    assert "hard_state_age" in schema["current_regime"]
    assert "expected_state_age" in schema["current_regime"]
    assert "previous_completed_state_4" in schema["regime_history"]


def test_contract_freezes_chronology_support_and_no_future_redefinition() -> None:
    contract = _read(CONTRACT)
    assert contract["chronology"]["development_year"] == 2024
    assert contract["chronology"]["unchanged_retrospective_assessment_year"] == 2025
    assert contract["chronology"]["protected_2026_enabled"] is False
    assert contract["chronology"]["preprocessing_fit_on_training_fold_only"] is True
    assert contract["chronology"]["validation_may_change_features_or_thresholds"] is False
    assert contract["support_gate"] == {
        "minimum_decision_rows": 200,
        "minimum_completion_events": 40,
        "minimum_stocks": 10,
        "minimum_sessions": 30,
        "minimum_months": 4,
        "maximum_single_stock_share": 0.25,
        "fail_closed": True,
    }


def test_contract_requires_prefix_position_and_causal_market_state() -> None:
    contract = _read(CONTRACT)
    assert contract["orientation"]["prefix_position_required_even_when_state_repeats"] is True
    assert contract["orientation"]["current_state_alone_is_orientation"] is False
    assert contract["causal_population_schema"]["completed_bar_availability_required"] is True
    assert contract["causal_population_schema"]["session_history_reset_required"] is True
    assert "market_regime_posterior" in contract["causal_population_schema"]["market_context"]


def test_contract_declares_identical_populations_and_interpretable_models_only() -> None:
    contract = _read(CONTRACT)
    constraints = contract["model_constraints"]
    assert constraints["same_decision_rows_for_every_paired_comparison"] is True
    assert constraints["large_boosted_models_allowed"] is False
    assert constraints["neural_networks_allowed"] is False
    assert constraints["economic_outcomes_allowed"] is False
    assert len(contract["attribution_models"]) == 11


def test_runner_and_modules_import_no_trading_runtime() -> None:
    paths = (
        RUNNER,
        AUDITOR,
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
    )
    names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.add(node.module)
    assert not any(
        token in name.lower()
        for name in names
        for token in ("broker", "execution", "orders", "positions", "deployment")
    )


def test_no_part_b_scientific_report_exists_when_gate_is_closed() -> None:
    assert not PART_B_REPORT.exists()
