from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/20260714-causal-loop-state-path-v1.json"
RUNNER_PATH = ROOT / "run_causal_loop_state_path_v1.py"
SPEC = importlib.util.spec_from_file_location("causal_loop_state_path_v1", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_contract_is_research_only_and_cannot_promote() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    assert safety["research_only"] is True
    assert safety["live_ordering_enabled"] is False
    assert safety["order_placement"] == "disabled"
    assert safety["broker_connection_enabled"] is False
    assert safety["paper_or_demo_execution_enabled"] is False
    assert safety["deployment_enabled"] is False
    assert safety["position_or_order_functionality_allowed"] is False
    assert safety["application_code_modification_allowed"] is False
    assert safety["repository_write_allowed"] is False
    assert contract["sealed_data_status"]["2023_or_2025_validation_claim_allowed"] is False
    assert contract["evaluation"]["promotion_allowed"] is False


def test_frozen_hypotheses_cover_mechanism_and_control() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert [item["id"] for item in contract["hypotheses"]] == [
        "H1_path_topology_separates_payoff_anatomy",
        "H2_terminal_route_event_preserves_payoff",
        "H3_completion_and_invalidation_have_distinct_roles",
        "H4_route_structure_beats_generic_transition_timing",
        "H5_child_or_morph_followup_requires_causal_forecast",
    ]
    assert contract["causal_clock"]["same_bar_execution_forbidden"] is True
    assert contract["outcome_only_path_topology"]["use_as_admission_feature"] is False


def test_path_topology_is_exhaustive_for_first_two_transitions() -> None:
    classify = RUNNER.topology_from_transitions
    assert classify([], 4, 2)[0] == "no_transition"
    assert classify([(2, 11)], 4, 2)[0] == "expected_leg_partial"
    exact = classify([(2, 11), (4, 14), (7, 18)], 4, 2)
    assert exact == ("exact_parent_completion", 14, None, "completion", 14)
    incompatible = classify([(6, 11), (4, 14)], 4, 2)
    assert incompatible == ("incompatible_first_transition", None, 11, "invalidation", 11)
    diversion = classify([(2, 11), (7, 14), (4, 18)], 4, 2)
    assert diversion == ("expected_leg_then_diversion", None, 14, "invalidation", 14)


def test_pre_entry_clock_excludes_event_on_entry_bar() -> None:
    transitions = [(2, 11), (4, 14)]
    assert RUNNER.pre_entry_status(transitions, 11, 4, 2) == "orientation_intact"
    assert RUNNER.pre_entry_status(transitions, 14, 4, 2) == "expected_leg_active"
    assert RUNNER.pre_entry_status(transitions, 15, 4, 2) == "completed_before_entry"


def test_event_policy_detects_at_close_and_exits_next_open() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 14:30", periods=30, freq="5min", tz="UTC"),
            "open": [100.0 + index for index in range(30)],
        }
    )
    result = RUNNER.event_policy(
        event_position=106,
        entry_state_position=103,
        anchor_state_position=100,
        tape_anchor_ordinal=2,
        entry_ordinal=5,
        frozen_exit_ordinal=26,
        bars=bars,
        direction=1,
        entry_price=103.0,
        fixed_exit_price=130.0,
    )
    assert result["detection_ordinal"] == 8
    assert result["next_open_ordinal"] == 9
    assert result["exit_ordinal"] == 9
    assert result["exit_price"] == 109.0
    assert result["actionable"] is True


def test_pre_entry_or_too_late_event_falls_back_to_frozen_close() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 14:30", periods=30, freq="5min", tz="UTC"),
            "open": [100.0 + index for index in range(30)],
        }
    )
    pre_entry = RUNNER.event_policy(102, 103, 100, 2, 5, 26, bars, 1, 103.0, 130.0)
    too_late = RUNNER.event_policy(124, 103, 100, 2, 5, 26, bars, 1, 103.0, 130.0)
    for result in (pre_entry, too_late):
        assert result["actionable"] is False
        assert result["exit_ordinal"] == 26
        assert result["exit_price"] == 130.0


def test_cost_is_five_bps_per_side() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["population"]["primary_cost_bps_per_side"] == 5
    assert contract["population"]["round_trip_cost_bps"] == 10
    assert RUNNER.PRIMARY_COST_PER_SIDE == 5
    assert RUNNER.ROUND_TRIP_COST == 10
