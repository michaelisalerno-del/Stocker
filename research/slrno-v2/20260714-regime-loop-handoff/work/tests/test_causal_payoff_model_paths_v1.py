from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/20260714-causal-payoff-model-paths-v1.json"
RUNNER_PATH = ROOT / "run_causal_payoff_model_paths_v1.py"
SPEC = importlib.util.spec_from_file_location("causal_payoff_model_paths_v1", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_contract_preserves_research_only_boundary() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    seal = contract["sealed_data_status"]
    assert safety["research_only"] is True
    assert safety["live_ordering_enabled"] is False
    assert safety["order_placement"] == "disabled"
    assert safety["broker_connection_enabled"] is False
    assert safety["paper_or_demo_execution_enabled"] is False
    assert safety["deployment_enabled"] is False
    assert safety["position_or_order_functionality_allowed"] is False
    assert safety["application_code_modification_allowed"] is False
    assert safety["repository_write_allowed"] is False
    assert seal["genuinely_unseen_sessions_available"] is False
    assert seal["validation_claim_allowed"] is False
    assert seal["diversion_specific_hypothesis_test_allowed"] is False
    assert contract["evaluation"]["promotion_allowed"] is False


def test_hypotheses_are_frozen_and_diversion_is_deferred() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert [item["id"] for item in contract["hypotheses"]] == [
        "H1_direct_admission_payoff_state",
        "H2_route_branch_forecast",
        "H3_predicted_route_adds_to_direct_payoff",
        "H4_sequential_route_plus_price_path",
        "H5_diversion_specific_payoff",
    ]
    assert contract["models"]["uncertainty_class"]["point_mean_policy"].endswith("cannot qualify")
    assert contract["population"]["warmup_completed_sessions"] == 60
    assert contract["population"]["score_completed_sessions"] == 68


def test_topology_and_causal_route_status_are_exhaustive() -> None:
    classify = RUNNER.topology_from_transitions
    assert classify([], 4, 2) == "no_transition"
    assert classify([(2, 4)], 4, 2) == "expected_leg_partial"
    assert classify([(2, 4), (4, 9)], 4, 2) == "exact_parent_completion"
    assert classify([(7, 4)], 4, 2) == "incompatible_first_transition"
    assert classify([(2, 4), (7, 9)], 4, 2) == "expected_leg_then_diversion"
    transitions = [(2, 4), (4, 9)]
    assert RUNNER.causal_route_status(transitions, 3, 4, 2) == "orientation_intact"
    assert RUNNER.causal_route_status(transitions, 4, 4, 2) == "expected_leg_active"
    assert RUNNER.causal_route_status(transitions, 9, 4, 2) == "exact_parent_completion_detected"


def test_pre_entry_clock_excludes_entry_bar_transition() -> None:
    transitions = [(2, 4), (4, 9)]
    assert RUNNER.pre_entry_status(transitions, 4, 4, 2) == "orientation_intact"
    assert RUNNER.pre_entry_status(transitions, 9, 4, 2) == "expected_leg_active"
    assert RUNNER.pre_entry_status(transitions, 10, 4, 2) == "completed_before_entry"


def test_checkpoints_are_fixed_holding_window_quartiles() -> None:
    assert RUNNER.checkpoint_offsets(0) == []
    assert RUNNER.checkpoint_offsets(1) == []
    assert RUNNER.checkpoint_offsets(2) == [(0.25, 1)]
    assert RUNNER.checkpoint_offsets(20) == [(0.25, 5), (0.5, 10), (0.75, 15)]
    assert RUNNER.checkpoint_offsets(3) == [(0.25, 1), (0.5, 2)]


def test_uncertainty_classes_require_interval_clearance() -> None:
    mean = np.array([100.0, -100.0, 1.0])
    std = np.array([10.0, 10.0, 10.0])
    assert RUNNER.interval_class(mean, std, "admission").tolist() == [
        "positive",
        "negative",
        "unknown_abstain",
    ]
    assert RUNNER.interval_class(mean, std, "sequential").tolist() == [
        "positive_hold",
        "negative_exit",
        "unknown_abstain",
    ]


def test_body_and_wick_fractions_are_bounded() -> None:
    body, upper, lower = RUNNER.body_wicks(100.0, 110.0, 90.0, 105.0)
    assert np.isclose(body, 0.25)
    assert np.isclose(upper, 0.25)
    assert np.isclose(lower, 0.5)
    assert RUNNER.body_wicks(100.0, 100.0, 100.0, 100.0) == (0.0, 0.0, 0.0)


def test_holm_requires_positive_interval() -> None:
    frame = pd.DataFrame(
        {
            "p_one_sided": [0.001, 0.01, 0.2],
            "ci_lower": [1.0, -1.0, 1.0],
        }
    )
    adjusted = RUNNER.holm_adjust(frame)
    assert bool(adjusted.iloc[0]["passes_holm_0_05"])
    assert not bool(adjusted.iloc[1]["passes_holm_0_05"])
    assert not bool(adjusted.iloc[2]["passes_holm_0_05"])


def test_feature_allowlist_excludes_outcomes() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    allowed = set(contract["admission_features"]["numeric"]) | set(
        contract["admission_features"]["categorical"]
    )
    forbidden_tokens = ("gross", "net_return", "future", "mfe", "mae", "child", "morph")
    assert not [
        name for name in allowed if any(token in name.lower() for token in forbidden_tokens)
    ]
    assert tuple(contract["admission_features"]["numeric"]) == RUNNER.ADMISSION_NUMERIC
    assert tuple(contract["admission_features"]["categorical"]) == RUNNER.ADMISSION_CATEGORICAL
