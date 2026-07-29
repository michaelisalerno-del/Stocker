from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stocker_research.regime_validity_v2 import causal_filter_summary

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
CONTRACT = WORK_DIR / "contracts" / "20260718-regime-model-validity-v2.json"
RUNNER = WORK_DIR / "run_regime_model_validity_v2.py"
PRIMARY = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "primary"


def _load_runner() -> object:
    package_root = REPO_ROOT / "packages" / "stocker_research" / "src"
    for path in (package_root, WORK_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("regime_validity_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER_MODULE = _load_runner()


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_is_hash_frozen() -> None:
    digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert digest == RUNNER_MODULE.EXPECTED_CONTRACT_HASH


def test_contract_has_all_safety_flags() -> None:
    contract = _contract()
    for key, value in RUNNER_MODULE.safety_flags().items():
        assert contract[key] == value


def test_contract_closes_economic_and_protected_future_data() -> None:
    contract = _contract()
    assert contract["economic_outcomes_used"] is False
    assert contract["payoff_selection_used"] is False
    assert contract["periods"]["protected_future"]["read_enabled"] is False


def test_contract_preregisters_k_seed_surface() -> None:
    contract = _contract()
    assert contract["k_seed_sensitivity"]["state_counts"] == [6, 8, 10, 12]
    assert len(contract["k_seed_sensitivity"]["seeds"]) == 5
    assert tuple(contract["k_seed_sensitivity"]["seeds"]) == RUNNER_MODULE.SEEDS


def test_contract_preregisters_cleaning_variants() -> None:
    variants = _contract()["cleaning_variants"]
    assert set(variants) >= {"CLEANING_0", "CLEANING_1", "CLEANING_CAUSAL"}
    assert variants["winner_selection_allowed"] is False


def test_contract_requires_gate_freeze_before_part_b() -> None:
    contract = _contract()
    assert contract["part_b_access_gate"]["decision_hash_frozen_before_access"] is True
    assert contract["part_b_access_gate"]["independent_audit_passed_before_access"] is True


def test_required_artifact_registry_matches_preregistered_family() -> None:
    required = set(RUNNER_MODULE.PART_A_REQUIRED_ARTIFACTS)
    assert len(required) == 50
    assert {
        "regime_implementation_census.csv",
        "implementation_source_manifest.json",
        "current_state_parameters.npz",
        "state_transition_confidence.parquet",
        "hierarchical_state_mapping.parquet",
        "part_a_decision.json",
        "independent_audit.json",
        "exact_rerun_manifest.json",
    } <= required


def test_implementation_census_has_required_schema() -> None:
    census = pd.read_csv(PRIMARY / "regime_implementation_census.csv")
    assert {
        "file",
        "function_or_class",
        "active_or_frozen",
        "fit_period",
        "score_period",
        "future_information_possible",
        "existing_auditor",
        "proposed_audit",
        "risk",
    } <= set(census.columns)
    assert len(census) >= 30


def test_source_identity_precedes_implementation() -> None:
    source = json.loads((PRIMARY / "source_identity_manifest.json").read_text(encoding="utf-8"))
    assert source["git_sha"] == RUNNER_MODULE.BASELINE_SHA
    assert source["branch"] == "agent/slrno-research-handoff"
    assert source["source_identity_status"] == ("pre_run_frozen_pending_independent_reconstruction")


def test_pre_run_tree_manifest_binds_frozen_lineage() -> None:
    manifest = json.loads((PRIMARY / "pre_run_tree_manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_sha"] == RUNNER_MODULE.BASELINE_SHA
    assert manifest["frozen_tree_hash"]
    assert manifest["protected_lineage_file_count"] > 1000
    assert manifest["generated_before_python_source_edits"] is True


def test_runner_has_no_execution_or_broker_import() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        token in name
        for name in imported
        for token in ("stocker_execution", "broker", "orders", "positions")
    )


def test_part_a_gate_rejects_duration_failure() -> None:
    evidence = RUNNER_MODULE.PartAGateEvidence(
        source_available=True,
        exact_reconstruction_pass=True,
        independent_audit_reproducible=True,
        mathematical_audit_pass=True,
        posterior_duration_pass=False,
        critical_future_leakage=False,
        hysteretic_same_primitive_fraction=1.0,
        k8_selected_loop_seed_gate_pass=True,
        minimum_state_occupancy=0.1,
        maximum_single_stock_share=0.1,
        semantic_drift_pass=True,
        training_sample_dictionary_coverage_ratio=1.0,
        combined_stability_deficit=0.0,
        representation_sensitive=False,
        usable_with_sensitivity=True,
        recoverable_local_defect=True,
        hierarchical_materially_more_stable=False,
        hierarchical_reproducible=False,
    )
    assert RUNNER_MODULE.decide_part_a(evidence).value == (
        "regime_representation_requires_targeted_repair"
    )


def test_compiled_filter_summary_matches_reference_recursion() -> None:
    emissions = np.asarray(
        [[-0.1, -1.2], [-0.2, -0.8], [-1.0, -0.1], [-0.6, -0.4], [-0.3, -0.9], [-0.8, -0.2]]
    )
    groups = (np.arange(0, 3, dtype=int), np.arange(3, 6, dtype=int))
    model = {
        "duration_hazard": np.asarray([[0.2, 0.4, 1.0], [0.1, 0.5, 1.0]]),
        "transitions": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "initial": np.asarray([0.6, 0.4]),
        "occupancy": np.asarray([0.55, 0.45]),
    }
    reference = causal_filter_summary(emissions, groups=groups, model=model)
    compiled = RUNNER_MODULE._causal_filter_summary_compiled(emissions, groups=groups, model=model)
    np.testing.assert_allclose(compiled.state_probabilities, reference.state_probabilities)
    np.testing.assert_array_equal(compiled.hard_states, reference.hard_states)
    np.testing.assert_allclose(compiled.expected_age, reference.expected_age)
    np.testing.assert_allclose(compiled.departure_probability, reference.departure_probability)
    np.testing.assert_allclose(compiled.posterior_entropy, reference.posterior_entropy)
    np.testing.assert_allclose(compiled.log_likelihood, reference.log_likelihood)
    np.testing.assert_allclose(compiled.iid_log_likelihood, reference.iid_log_likelihood)


def test_report_table_has_no_optional_dependency() -> None:
    rendered = RUNNER_MODULE._markdown_table(pd.DataFrame([{"model": "combined", "score": 1.5}]))
    assert rendered == "| model | score |\n| --- | --- |\n| combined | 1.5 |"
