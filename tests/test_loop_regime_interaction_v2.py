from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.loop_orientation_v2 import (
    build_orientation_registry,
    orientation_for_prefix,
)
from stocker_research.loop_regime_interaction_v2 import (
    NON_LOOP_FIRST_EVENT_CLASSES,
    CounterfactualSupport,
    PartAGateState,
    PartBGateClosedError,
    assert_identical_decision_populations,
    assert_part_b_scoring_authorized,
    benjamini_hochberg,
    completed_state_history,
    evaluate_counterfactual_support,
    expanding_period_folds,
    population_scaffold,
    stock_deletion_populations,
    validate_first_event_classes,
    validate_interaction_columns,
)
from stocker_research.state_representation_sensitivity_v2 import (
    compare_representation_events,
)


def _gate(decision: str = "regime_representation_requires_targeted_repair") -> PartAGateState:
    return PartAGateState(
        decision=decision,
        decision_file_hash="a" * 64,
        binding_hash="b" * 64,
        state_model_hash="c" * 64,
        state_alignment_hash="d" * 64,
        independent_audit_status="pass",
        independent_audit_file_hash="e" * 64,
        exact_rerun_byte_identical=True,
        exact_rerun_manifest_file_hash="f" * 64,
    )


def test_orientation_includes_route_and_prefix_position() -> None:
    orientation = orientation_for_prefix(
        primitive_loop_id="loop_p_2-4-2",
        oriented_path=(2, 4, 2),
        active_prefix=(2, 4),
    )
    assert orientation.orientation_id == ("loop_p_2-4-2::route_2-4-2::position_1_at_4_waiting_2")
    assert orientation.transitions_completed == 1
    assert orientation.transitions_remaining == 1
    assert orientation.prefix_progress == 0.5


def test_repeated_internal_states_have_unambiguous_orientation_ids() -> None:
    registry = build_orientation_registry({"loop_p_synthetic": ((1, 2, 1, 3, 1),)})
    repeated = registry.loc[registry["current_state"].eq(1)]
    assert len(repeated) == 2
    assert repeated["orientation_id"].nunique() == 2
    assert set(repeated["prefix_position"]) == {0, 2}


def test_prefix_progress_rejects_unobserved_or_completed_suffix() -> None:
    with pytest.raises(ValueError, match="observed prefix"):
        orientation_for_prefix(
            primitive_loop_id="loop_p_2-4-2",
            oriented_path=(2, 4, 2),
            active_prefix=(2, 2),
        )
    with pytest.raises(ValueError, match="incomplete"):
        orientation_for_prefix(
            primitive_loop_id="loop_p_2-4-2",
            oriented_path=(2, 4, 2),
            active_prefix=(2, 4, 2),
        )


def test_completed_state_history_is_past_only_and_resets_by_session() -> None:
    history = completed_state_history(
        np.array([1, 1, 2, 2, 3, 7, 7, 6]),
        session_groups=(np.arange(5), np.arange(5, 8)),
        depth=3,
    )
    assert history.previous_states[2].tolist() == [1, -1, -1]
    assert history.previous_durations[2].tolist() == [2, -1, -1]
    assert history.previous_states[4].tolist() == [2, 1, -1]
    assert history.previous_states[5].tolist() == [-1, -1, -1]
    assert history.history_tokens[5] == "START"


def test_chronological_folds_never_train_on_validation_or_later_periods() -> None:
    periods = ["2024Q1", "2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    folds = expanding_period_folds(periods, minimum_training_periods=2)
    assert [fold.validation_period for fold in folds] == ["2024Q3", "2024Q4"]
    for fold in folds:
        assert max(fold.train_periods) < fold.validation_period
        assert not set(fold.train_indices).intersection(fold.validation_indices)


def test_all_model_comparisons_require_identical_ordered_decisions() -> None:
    signature = assert_identical_decision_populations(
        {"M1": ("d1", "d2"), "M4": ("d1", "d2"), "M9": ("d1", "d2")}
    )
    assert len(signature) == 64
    with pytest.raises(ValueError, match="identical ordered"):
        assert_identical_decision_populations({"M1": ("d1", "d2"), "M4": ("d2", "d1")})


def test_economic_and_future_outcome_columns_fail_closed() -> None:
    validate_interaction_columns(("current_state", "prefix_progress", "primary_class"))
    with pytest.raises(ValueError, match="forbidden"):
        validate_interaction_columns(("current_state", "future_return_5m"))
    with pytest.raises(ValueError, match="forbidden"):
        validate_interaction_columns(("prefix_progress", "pnl"))


def test_corrected_first_event_classes_are_retained_and_legacy_labels_rejected() -> None:
    classes = sorted(NON_LOOP_FIRST_EVENT_CLASSES) + ["loop_p_2-4-2"]
    validate_first_event_classes(classes, selected_primitive_ids=frozenset({"loop_p_2-4-2"}))
    with pytest.raises(ValueError, match="legacy"):
        validate_first_event_classes(
            classes + ["legacy_cycle_17"],
            selected_primitive_ids=frozenset({"loop_p_2-4-2"}),
        )


def test_counterfactual_support_gates_fail_closed() -> None:
    supported = evaluate_counterfactual_support(CounterfactualSupport(200, 40, 10, 30, 4, 0.25))
    assert supported.supported
    failed = evaluate_counterfactual_support(CounterfactualSupport(199, 39, 9, 29, 3, 0.251))
    assert not failed.supported
    assert set(failed.failures) == {
        "decision_rows",
        "completion_events",
        "stocks",
        "sessions",
        "months",
        "single_stock_concentration",
    }


def test_bh_families_are_deterministic_under_input_reordering() -> None:
    left = benjamini_hochberg({"c": 0.04, "a": 0.01, "b": 0.03})
    right = benjamini_hochberg({"b": 0.03, "c": 0.04, "a": 0.01})
    assert left == right == {"a": 0.03, "b": 0.04, "c": 0.04}


def test_stock_deletions_are_recomputed_from_the_full_population() -> None:
    deletions = stock_deletion_populations(("A", "A", "B", "C"), ("d0", "d1", "d2", "d3"))
    by_symbol = {item.omitted_symbol: item for item in deletions}
    assert by_symbol["A"].retained_indices.tolist() == [2, 3]
    assert by_symbol["B"].retained_indices.tolist() == [0, 1, 3]
    assert by_symbol["C"].retained_indices.tolist() == [0, 1, 2]
    assert len({item.decision_signature for item in deletions}) == 3


def test_hard_hysteretic_event_alignment_classifies_exact_shift_and_mismatch() -> None:
    reference = pd.DataFrame(
        {
            "decision_id": ["a", "b", "c"],
            "primary_label": ["loop_p_2-4-2", "loop_p_2-4-2", "loop_p_2-4-2"],
            "bars_until_completion": [2, 3, 4],
        }
    )
    candidate = pd.DataFrame(
        {
            "decision_id": ["a", "b", "c"],
            "primary_label": ["loop_p_2-4-2", "loop_p_2-4-2", "loop_p_4-6-4"],
            "bars_until_completion": [2, 4, 4],
        }
    )
    compared, metrics = compare_representation_events(reference, candidate, allowed_shift_bars=1)
    assert compared["agreement_class"].tolist() == [
        "EXACT_EVENT_AGREEMENT",
        "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
        "PRIMITIVE_MISMATCH",
    ]
    assert metrics.same_primitive_bounded_shift_fraction == pytest.approx(2 / 3)


def test_part_b_gate_closes_before_population_access() -> None:
    with pytest.raises(PartBGateClosedError, match="targeted_repair"):
        assert_part_b_scoring_authorized(_gate())
    authorized = _gate("regime_representation_valid_with_required_sensitivity")
    assert_part_b_scoring_authorized(authorized)


def test_population_scaffold_distinguishes_hard_and_expected_age_and_has_safety() -> None:
    scaffold = population_scaffold(_gate(), proposed_contract_hash="1" * 64)
    regime_fields = scaffold["schema_groups"]["current_regime"]
    assert "hard_state_age" in regime_fields
    assert "expected_state_age" in regime_fields
    assert scaffold["population_rows_read"] == 0
    assert scaffold["interaction_results_inspected"] is False
    assert scaffold["research_only"] is True
    assert scaffold["execution_enabled"] is False
    assert scaffold["economic_outcomes_used"] is False
