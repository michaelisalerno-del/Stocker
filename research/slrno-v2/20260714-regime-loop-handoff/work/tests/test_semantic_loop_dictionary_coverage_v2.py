from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from run_semantic_loop_dictionary_coverage_v2 import (
    ArtifactIdentity,
    PreflightContext,
    RunContext,
    _dictionary_replication,
    _family_outputs,
    _filter_generated_git_status,
    _write_mixed_period_frame,
)

from stocker_research.first_event_target_v2 import (
    build_first_event_target,
    build_loop_family_mapping,
    decide_target_tractability,
)
from stocker_research.loop_nulls_v2 import SemiMarkovNull, SessionRunSequence, SimulatedSession
from stocker_research.loop_structural_nulls_v2 import (
    first_event_candidate_counts,
    information_increment_from_chronological_folds,
    qualify_structural_candidates,
    reject_economic_columns,
    simulate_first_event_family_null_counts_fast,
    simulate_first_event_null_counts_by_group_fast,
    simulate_first_event_null_counts_fast,
)
from stocker_research.loop_tie_resolution_v2 import (
    TieClass,
    resolve_registered_ties,
)
from stocker_research.prefix_features_v2 import build_compressed_prefix_features
from stocker_research.semantic_loop_dictionary_v2 import (
    PRIMARY_PRIMITIVE_TRANSITION_LENGTHS,
    SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS,
    CandidateSupportGates,
    SemanticMotifType,
    build_candidate_universe,
    decompose_semantic_path,
    deterministic_dictionary_hash,
    select_primary_dictionary,
)
from stocker_research.unregistered_loop_census_v2 import reconstruct_first_events


def _decision_frame(
    states: list[int],
    *,
    session: str = "2024-01-02",
    source_available: bool = True,
    structurally_eligible: bool = True,
) -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 14:30:00", tz="UTC")
    rows = []
    for ordinal, state in enumerate(states):
        bar_start = start + timedelta(minutes=5 * ordinal)
        rows.append(
            {
                "decision_id": f"{session}:{ordinal}",
                "symbol": "SYN",
                "session": session,
                "bar_ordinal": ordinal,
                "bar_start_timestamp": bar_start,
                "bar_complete_timestamp": bar_start + timedelta(minutes=5),
                "decision_timestamp": bar_start + timedelta(minutes=5),
                "hard_state_legacy": state,
                "structural_event_eligibility": structurally_eligible,
                "source_available": source_available,
                "source_sequence_complete": structurally_eligible,
                "source_sequence_missing_reason": (
                    None if structurally_eligible else "in_session_structural_gap"
                ),
                "clock_phase": "opening",
                "run_id": ordinal,
                "git_sha": "g",
                "contract_hash": "c",
                "data_snapshot_hash": "d",
                "dictionary_version": "input_v2",
                "dictionary_hash": "old",
                "state_model_version": "state_v2",
                "source_artifact_hash": "source",
            }
        )
    return pd.DataFrame(rows)


def _old_tied_outcome(decision_id: str = "d") -> pd.DataFrame:
    return pd.DataFrame(
        [{"decision_id": decision_id, "primary_label": "TIED_REGISTERED_COMPLETION"}]
    )


def test_semantic_id_is_independent_of_rotation_and_rank() -> None:
    left = decompose_semantic_path((2, 4, 6, 2))
    right = decompose_semantic_path((4, 6, 2, 4))
    assert left.primitive_loop_id == right.primitive_loop_id == "loop_p_2-4-6-2"
    rows = [
        {
            "semantic_loop_id": left.primitive_loop_id,
            "canonical_primitive_core": [2, 4, 6],
            "selection_rank": 1,
        }
    ]
    reranked = [{**rows[0], "selection_rank": 31}]
    assert deterministic_dictionary_hash(rows) == deterministic_dictionary_hash(reranked)


def test_reverse_traversal_is_distinct_orientation_metadata() -> None:
    forward = decompose_semantic_path((0, 1, 2, 0))
    reverse = decompose_semantic_path((0, 2, 1, 0))
    assert forward.primitive_loop_id != reverse.primitive_loop_id
    assert forward.reverse_path_id == reverse.primitive_loop_id
    assert forward.orientation == (0, 1, 2, 0)


def test_repeated_traversal_is_auxiliary_to_one_primitive() -> None:
    event = decompose_semantic_path((0, 1, 0, 1, 0))
    assert event.motif_type is SemanticMotifType.REPEAT
    assert event.primitive_loop_id == "loop_p_0-1-0"
    assert event.repeat_depth == 2
    assert event.semantic_motif_id == "loop_r2_0-1-0"
    assert event.primary_class_eligible is False


def test_nested_composite_has_ordered_components_and_final_primitive() -> None:
    event = decompose_semantic_path((0, 1, 2, 1, 0))
    assert event.motif_type is SemanticMotifType.COMPOSITE
    assert event.component_primitive_ids == ("loop_p_1-2-1", "loop_p_0-1-0")
    assert event.primitive_loop_id == "loop_p_0-1-0"
    assert event.component_boundaries == ((1, 3), (0, 4))


def test_composite_tie_final_root_uses_event_order_not_canonical_rotation() -> None:
    outcomes = _old_tied_outcome()
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_c_nested",
                "primitive_loop_id": None,
                "motif_type": "composite",
                "repeat_depth": 1,
                "full_path": np.asarray([1, 0, 1, 2, 1, 0, 1]),
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "full_path": [1, 0, 1],
                "is_primary_completion": True,
            },
        ]
    )
    resolved = resolve_registered_ties(outcomes, completions).classification.iloc[0]
    assert resolved["tie_class"] == TieClass.NESTED_SAME_PRIMITIVE_TIE
    assert resolved["tied_primitive_ids"] == ["loop_p_0-1-0"]
    assert resolved["rewritten_primary_label"] == "loop_p_0-1-0"


def test_allowed_lengths_reuse_the_audited_primary_range() -> None:
    assert frozenset({2, 3, 4, 5}) == PRIMARY_PRIMITIVE_TRANSITION_LENGTHS
    assert frozenset({6, 7, 8}) == SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS


def test_primitive_plus_repeat_resolves_to_one_primary_event() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_r2_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "repeat",
                "repeat_depth": 2,
                "is_primary_completion": True,
            },
        ]
    )
    result = resolve_registered_ties(_old_tied_outcome(), completions)
    row = result.classification.iloc[0]
    assert row.tie_class == TieClass.NESTED_SAME_PRIMITIVE_TIE
    assert row.rewritten_primary_label == "loop_p_0-1-0"
    assert row.maximum_repeat_depth == 2


def test_primitive_plus_composite_ending_at_same_root_is_not_genuine_tie() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_c_x",
                "primitive_loop_id": None,
                "motif_type": "composite",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
        ]
    )
    result = resolve_registered_ties(
        _old_tied_outcome(),
        completions,
        composite_components={"loop_c_x": ("loop_p_1-2-1", "loop_p_0-1-0")},
    )
    row = result.classification.iloc[0]
    assert row.tie_class == TieClass.NESTED_SAME_PRIMITIVE_TIE
    assert row.rewritten_primary_label == "loop_p_0-1-0"


def test_two_distinct_primitive_roots_remain_a_genuine_tie() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "loop_p_0-2-0",
                "primitive_loop_id": "loop_p_0-2-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
        ]
    )
    result = resolve_registered_ties(_old_tied_outcome(), completions)
    row = result.classification.iloc[0]
    assert row.tie_class == TieClass.DISTINCT_PRIMITIVE_TIE
    assert row.rewritten_primary_label == "DISTINCT_PRIMITIVE_TIE"
    assert row.tied_primitive_ids == ["loop_p_0-1-0", "loop_p_0-2-0"]


def test_legacy_aliases_do_not_create_a_semantic_tie() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "legacy_a",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "legacy_b",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
        ]
    )
    result = resolve_registered_ties(
        _old_tied_outcome(),
        completions,
        legacy_aliases={"legacy_a": "loop_p_0-1-0", "legacy_b": "loop_p_0-1-0"},
    )
    row = result.classification.iloc[0]
    assert row.tie_class == TieClass.MIGRATION_OR_IDENTITY_TIE
    assert row.rewritten_primary_label == "loop_p_0-1-0"


def test_unknown_tie_identity_fails_closed() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "unknown",
                "primitive_loop_id": None,
                "motif_type": None,
                "repeat_depth": None,
                "is_primary_completion": True,
            }
        ]
    )
    result = resolve_registered_ties(_old_tied_outcome(), completions)
    row = result.classification.iloc[0]
    assert row.tie_class == TieClass.UNKNOWN_TIE
    assert row.rewritten_primary_label == "UNAVAILABLE_STRUCTURAL_GAP"


def test_tie_metadata_is_sorted_but_never_selects_a_winner() -> None:
    completions = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "semantic_loop_id": "z",
                "primitive_loop_id": "loop_p_0-2-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
            {
                "decision_id": "d",
                "semantic_loop_id": "a",
                "primitive_loop_id": "loop_p_0-1-0",
                "motif_type": "primitive",
                "repeat_depth": 1,
                "is_primary_completion": True,
            },
        ]
    )
    row = resolve_registered_ties(_old_tied_outcome(), completions).classification.iloc[0]
    assert row.tied_semantic_ids == ["a", "z"]
    assert row.rewritten_primary_label == "DISTINCT_PRIMITIVE_TIE"


def test_minimal_closed_suffix_and_earliest_event_are_selected() -> None:
    decisions = _decision_frame([0, 1, 2, 1, 0])
    ledger = reconstruct_first_events(decisions, horizon_bars=24)
    first = ledger.loc[ledger.decision_id.eq("2024-01-02:0")].iloc[0]
    assert first.full_closed_path == [1, 2, 1]
    assert first.primitive_loop_id == "loop_p_1-2-1"
    assert first.event_bar_ordinal == 3


def test_nested_past_loop_does_not_hide_final_primitive_root() -> None:
    decisions = _decision_frame([0, 1, 2, 1, 0])
    ledger = reconstruct_first_events(decisions, horizon_bars=24)
    row = ledger.loc[ledger.decision_id.eq("2024-01-02:3")].iloc[0]
    assert row.full_closed_path == [0, 1, 2, 1, 0]
    assert row.motif_type == "composite"
    assert row.primitive_loop_id == "loop_p_0-1-0"
    assert row.component_primitive_ids == ["loop_p_1-2-1", "loop_p_0-1-0"]


def test_session_boundary_is_never_crossed() -> None:
    left = _decision_frame([0, 1], session="2024-01-02")
    right = _decision_frame([0], session="2024-01-03")
    ledger = reconstruct_first_events(pd.concat([left, right], ignore_index=True), horizon_bars=24)
    assert not ledger.primary_event.str.startswith("loop_p_").any()
    assert set(ledger.primary_event) == {"SESSION_END"}


def test_structural_gap_prevents_false_closure() -> None:
    decisions = _decision_frame([0, 1, 0], structurally_eligible=False)
    ledger = reconstruct_first_events(decisions, horizon_bars=24)
    assert ledger.primary_event.eq("UNAVAILABLE_STRUCTURAL_GAP").all()
    assert ledger.full_closed_path.map(len).eq(0).all()


def test_unavailable_source_has_precedence_over_structural_gap() -> None:
    decisions = _decision_frame([0, 1, 0], source_available=False, structurally_eligible=False)
    ledger = reconstruct_first_events(decisions, horizon_bars=24)
    assert ledger.primary_event.eq("UNAVAILABLE_SOURCE").all()


def test_completion_exactly_at_horizon_is_included_and_after_is_excluded() -> None:
    states = [0] * 1 + [1] * 24 + [0]
    decisions = _decision_frame(states)
    at_horizon = reconstruct_first_events(decisions, horizon_bars=25).iloc[0]
    after_horizon = reconstruct_first_events(decisions, horizon_bars=24).iloc[0]
    assert at_horizon.primary_event == "loop_p_0-1-0"
    assert at_horizon.bars_until_completion == 25
    assert after_horizon.primary_event == "NO_LOOP_WITHIN_HORIZON"


def test_unregistered_event_extraction_is_deterministic() -> None:
    decisions = _decision_frame([0, 1, 2, 1, 0, 3, 0])
    left = reconstruct_first_events(decisions, horizon_bars=24)
    right = reconstruct_first_events(decisions.sample(frac=1.0, random_state=7), horizon_bars=24)
    columns = ["decision_id", "primary_event", "full_closed_path", "primitive_loop_id"]
    pd.testing.assert_frame_equal(
        left[columns].sort_values("decision_id").reset_index(drop=True),
        right[columns].sort_values("decision_id").reset_index(drop=True),
    )


def _supported_first_events(
    *,
    primitive_id: str = "loop_p_0-1-0",
    occurrences: int = 120,
    stocks: int = 12,
    months: int = 6,
) -> pd.DataFrame:
    rows = []
    for index in range(occurrences):
        month = index % months + 1
        symbol = f"S{index % stocks:02d}"
        timestamp = pd.Timestamp(f"2024-{month:02d}-{index % 20 + 1:02d} 15:00", tz="UTC")
        rows.append(
            {
                "decision_id": f"d{index}",
                "symbol": symbol,
                "session": timestamp.date().isoformat(),
                "decision_timestamp": timestamp,
                "clock_phase": ("opening", "middle", "late")[index % 3],
                "hard_state_legacy": index % 8,
                "primary_event": primitive_id,
                "primitive_loop_id": primitive_id,
                "primitive_transition_length": len(primitive_id.removeprefix("loop_p_").split("-"))
                - 1,
                "motif_type": "primitive",
                "source_completeness": True,
                "event_key": f"event-{index}",
            }
        )
    return pd.DataFrame(rows)


def test_candidate_support_gates_require_breadth_not_frequency_alone() -> None:
    events = _supported_first_events(occurrences=120, stocks=12, months=6)
    bundle = build_candidate_universe(events, gates=CandidateSupportGates())
    assert len(bundle.universe) == 1
    assert bool(bundle.support.iloc[0].support_pass)
    concentrated = events.copy()
    concentrated["symbol"] = "ONE"
    rejected = build_candidate_universe(concentrated, gates=CandidateSupportGates())
    assert not bool(rejected.support.iloc[0].support_pass)
    assert "below_minimum_stocks" in set(rejected.rejections.rejection_reason)
    assert "top_stock_share_above_maximum" in set(rejected.rejections.rejection_reason)


def test_sensitivity_length_cannot_enter_primary_candidate_universe() -> None:
    events = _supported_first_events(primitive_id="loop_p_0-1-2-3-4-5-6-0")
    events["primitive_transition_length"] = 7
    bundle = build_candidate_universe(events, gates=CandidateSupportGates())
    assert not bool(bundle.support.iloc[0].primary_length_eligible)
    assert "sensitivity_only_transition_length" in set(bundle.rejections.rejection_reason)


def _selection_candidates(count: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "semantic_loop_id": f"loop_p_{index}-{index + 1}-{index}",
                "primitive_loop_id": f"loop_p_{index}-{index + 1}-{index}",
                "motif_type": "primitive",
                "transition_length": 2,
                "development_count": 200 - index * 10,
                "support_pass": True,
                "structurally_qualified": True,
                "information_qualified": True,
                "oof_log_loss_increment": 0.04 - index * 0.005,
                "semi_markov_rate_ratio": 2.0 - index * 0.1,
                "stock_breadth": 12,
                "month_breadth": 8,
                "selection_period": "development",
            }
        )
    return pd.DataFrame(rows)


def test_unsupported_or_nonprimitive_candidates_cannot_be_selected() -> None:
    candidates = _selection_candidates()
    candidates.loc[0, "support_pass"] = False
    candidates.loc[1, "motif_type"] = "repeat"
    selected = select_primary_dictionary(
        candidates, total_valid_primitive_events=1_000, maximum_entries=32
    )
    assert set(selected.dictionary.semantic_loop_id) == set(candidates.semantic_loop_id[2:])


def test_structural_and_information_failures_cannot_be_selected() -> None:
    candidates = _selection_candidates()
    candidates.loc[0, "structurally_qualified"] = False
    candidates.loc[1, "information_qualified"] = False
    selected = select_primary_dictionary(
        candidates, total_valid_primitive_events=1_000, maximum_entries=32
    )
    assert set(selected.dictionary.semantic_loop_id) == set(candidates.semantic_loop_id[2:])


def test_maximum_32_and_marginal_coverage_stop_are_enforced() -> None:
    candidates = pd.concat([_selection_candidates(4)] * 10, ignore_index=True)
    candidates["semantic_loop_id"] = [f"loop_p_{index}-7-{index}" for index in range(40)]
    candidates["primitive_loop_id"] = candidates["semantic_loop_id"]
    candidates["development_count"] = 100
    maximum = select_primary_dictionary(
        candidates, total_valid_primitive_events=1_000, maximum_entries=32
    )
    assert len(maximum.dictionary) == 32
    marginal = _selection_candidates()
    marginal.loc[2:, "development_count"] = 4
    stopped = select_primary_dictionary(
        marginal,
        total_valid_primitive_events=1_000,
        maximum_entries=32,
        minimum_marginal_coverage=0.005,
    )
    assert len(stopped.dictionary) == 2
    assert stopped.selection_path.iloc[-1].selection_action == "STOP_BELOW_MARGINAL_COVERAGE"


def test_selection_tie_breaking_is_deterministic_and_validation_is_rejected() -> None:
    candidates = _selection_candidates()
    for field in ("oof_log_loss_increment", "semi_markov_rate_ratio", "development_count"):
        candidates[field] = candidates[field].iloc[0]
    left = select_primary_dictionary(
        candidates.sample(frac=1.0, random_state=1), total_valid_primitive_events=1_000
    )
    right = select_primary_dictionary(
        candidates.sample(frac=1.0, random_state=2), total_valid_primitive_events=1_000
    )
    assert left.dictionary.semantic_loop_id.tolist() == right.dictionary.semantic_loop_id.tolist()
    candidates["selection_period"] = "validation"
    try:
        select_primary_dictionary(candidates, total_valid_primitive_events=1_000)
    except ValueError as error:
        assert "development" in str(error)
    else:
        raise AssertionError("validation candidates altered dictionary membership")


def _complete_target_auxiliary_evidence(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    defaults: dict[str, object] = {
        "event_timestamp": None,
        "state_events_until_completion": None,
        "active_prefix_length_at_decision": 0,
        "initiated_before_decision": False,
        "initiated_after_decision": False,
        "previous_same_primitive_completion_timestamp": None,
        "bars_since_previous_same_primitive": None,
        "transitions_since_previous_same_primitive": None,
        "is_consecutive_repeat": False,
        "is_same_as_previous_primitive": None,
        "earliest_composite_completion": None,
        "first_component_completion": None,
        "final_component_completion": None,
        "earlier_primitive_completion_already_occurred": False,
        "composite_adds_information_beyond_primitive_sequence": False,
    }
    for field, value in defaults.items():
        output[field] = value
    output["semantic_loop_id"] = output["primitive_loop_id"]
    output["component_primitive_ids"] = [[] for _ in range(len(output))]
    output["component_boundaries"] = [[] for _ in range(len(output))]
    output["component_completion_timestamps"] = [[] for _ in range(len(output))]
    output["legacy_overlapping_positive_labels"] = [[] for _ in range(len(output))]
    return output


def test_frozen_target_uses_selected_exact_other_and_distinct_nonloop_classes() -> None:
    events = pd.DataFrame(
        [
            {
                "decision_id": "selected",
                "primary_event": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "bars_until_completion": 1,
                "repeat_depth": 1,
                "current_repeat_depth": 1,
                "motif_type": "primitive",
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            },
            {
                "decision_id": "other",
                "primary_event": "loop_p_0-2-0",
                "primitive_loop_id": "loop_p_0-2-0",
                "bars_until_completion": 2,
                "repeat_depth": 1,
                "current_repeat_depth": 1,
                "motif_type": "primitive",
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            },
            {
                "decision_id": "no",
                "primary_event": "NO_LOOP_WITHIN_HORIZON",
                "primitive_loop_id": None,
                "bars_until_completion": None,
                "repeat_depth": None,
                "current_repeat_depth": None,
                "motif_type": None,
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            },
            {
                "decision_id": "end",
                "primary_event": "SESSION_END",
                "primitive_loop_id": None,
                "bars_until_completion": None,
                "repeat_depth": None,
                "current_repeat_depth": None,
                "motif_type": None,
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            },
            {
                "decision_id": "gap",
                "primary_event": "UNAVAILABLE_STRUCTURAL_GAP",
                "primitive_loop_id": None,
                "bars_until_completion": None,
                "repeat_depth": None,
                "current_repeat_depth": None,
                "motif_type": None,
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            },
        ]
    )
    events = _complete_target_auxiliary_evidence(events)
    bundle = build_first_event_target(
        events, selected_primitive_ids={"loop_p_0-1-0"}, horizon_bars=24
    )
    labels = bundle.outcomes.set_index("decision_id").primary_class.to_dict()
    assert labels == {
        "selected": "loop_p_0-1-0",
        "other": "OTHER_PRIMITIVE_LOOP",
        "no": "NO_LOOP_WITHIN_HORIZON",
        "end": "SESSION_END",
        "gap": "UNAVAILABLE_STRUCTURAL_GAP",
    }
    assert bundle.outcomes.groupby("decision_id").size().eq(1).all()


def test_repeat_and_composite_are_auxiliary_not_primary() -> None:
    events = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "primary_event": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "bars_until_completion": 4,
                "repeat_depth": 2,
                "current_repeat_depth": 3,
                "motif_type": "composite",
                "nested_repeat_ids": ["loop_r2_0-1-0"],
                "nested_composite_ids": ["loop_c_x"],
            }
        ]
    )
    events = _complete_target_auxiliary_evidence(events)
    bundle = build_first_event_target(
        events, selected_primitive_ids={"loop_p_0-1-0"}, horizon_bars=24
    )
    assert bundle.outcomes.iloc[0].primary_class == "loop_p_0-1-0"
    auxiliary = bundle.auxiliary.iloc[0]
    assert auxiliary.repeat_depth == 2
    assert auxiliary.nested_composite_ids == ["loop_c_x"]


def test_target_horizon_fails_closed_for_late_completion() -> None:
    events = pd.DataFrame(
        [
            {
                "decision_id": "d",
                "primary_event": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "bars_until_completion": 25,
                "repeat_depth": 1,
                "current_repeat_depth": 1,
                "motif_type": "primitive",
                "nested_repeat_ids": [],
                "nested_composite_ids": [],
            }
        ]
    )
    try:
        build_first_event_target(events, selected_primitive_ids={"loop_p_0-1-0"}, horizon_bars=24)
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("event after the frozen horizon entered the target")


def test_first_event_null_counts_decisions_not_overlapping_raw_paths() -> None:
    session = SimulatedSession(
        states=(0, 1, 0, 1, 0),
        durations=(2, 2, 2, 2, 2),
        terminal_right_censored=True,
        phase_labels=("opening",) * 10,
    )
    counts = first_event_candidate_counts(
        session,
        candidate_ids=("loop_p_0-1-0",),
        horizon_bars=4,
        decision_ordinals=tuple(range(10)),
    )
    assert counts.tolist() == [8]


def test_first_event_null_never_crosses_session_boundary_or_horizon() -> None:
    session = SimulatedSession(
        states=(0, 1, 0),
        durations=(4, 4, 4),
        terminal_right_censored=True,
        phase_labels=("opening",) * 12,
    )
    short = first_event_candidate_counts(
        session,
        candidate_ids=("loop_p_0-1-0",),
        horizon_bars=3,
        decision_ordinals=tuple(range(12)),
    )
    assert short.tolist() == [3]


def test_group_resolved_null_recomputes_exact_deletion_counts_without_scaling() -> None:
    sessions = (
        SessionRunSequence("A", "s1", (0, 1, 0), (2, 2, 2), True),
        SessionRunSequence("B", "s2", (0, 1, 0), (3, 2, 1), True),
    )
    model = SemiMarkovNull.fit(sessions, state_count=8, maximum_duration=78)
    labels, grouped = simulate_first_event_null_counts_by_group_fast(
        model,
        session_lengths=(6, 6),
        session_groups=("A", "B"),
        candidate_ids=("loop_p_0-1-0",),
        horizon_bars=4,
        draws=16,
        seed=17,
    )
    aggregate = simulate_first_event_null_counts_fast(
        model,
        session_lengths=(6, 6),
        candidate_ids=("loop_p_0-1-0",),
        horizon_bars=4,
        draws=16,
        seed=17,
    )
    assert labels == ("A", "B")
    np.testing.assert_array_equal(grouped.sum(axis=1), aggregate)
    np.testing.assert_array_equal(aggregate - grouped[:, 0, :], grouped[:, 1, :])


def test_family_null_counts_every_first_primitive_by_frozen_length_family() -> None:
    sessions = (
        SessionRunSequence("A", "s1", (0, 1, 2, 0), (2, 2, 2, 2), True),
        SessionRunSequence("B", "s2", (3, 4, 3), (3, 2, 3), True),
    )
    model = SemiMarkovNull.fit(sessions, state_count=8, maximum_duration=78)
    left = simulate_first_event_family_null_counts_fast(
        model,
        session_lengths=(8, 8),
        horizon_bars=4,
        draws=16,
        seed=19,
    )
    right = simulate_first_event_family_null_counts_fast(
        model,
        session_lengths=(8, 8),
        horizon_bars=4,
        draws=16,
        seed=19,
    )
    assert left.shape == (16, 5)
    assert (left >= 0).all()
    np.testing.assert_array_equal(left, right)


def test_structural_qualification_enforces_fdr_ratio_quarters_and_deletions() -> None:
    candidates = pd.DataFrame(
        [
            {
                "semantic_loop_id": "pass",
                "support_pass": True,
                "semi_markov_q": 0.05,
                "semi_markov_rate_ratio": 1.4,
                "excess_count": 10,
                "positive_excess_quarters": 3,
                "leave_one_stock_out_minimum_rate_ratio": 1.01,
                "clock_null_q": 0.08,
            },
            {
                "semantic_loop_id": "frequency_only",
                "support_pass": True,
                "semi_markov_q": 0.20,
                "semi_markov_rate_ratio": 2.0,
                "excess_count": 100,
                "positive_excess_quarters": 4,
                "leave_one_stock_out_minimum_rate_ratio": 1.5,
                "clock_null_q": 0.01,
            },
        ]
    )
    result = qualify_structural_candidates(candidates)
    assert bool(result.loc[result.semantic_loop_id.eq("pass"), "structurally_qualified"].iloc[0])
    assert not bool(
        result.loc[result.semantic_loop_id.eq("frequency_only"), "structurally_qualified"].iloc[0]
    )
    assert (
        result.loc[result.semantic_loop_id.eq("pass"), "clock_null_status"].iloc[0]
        == "GLOBALLY_RECURRENT"
    )


def test_economic_columns_are_rejected_before_information_attribution() -> None:
    safe = pd.DataFrame({"decision_id": ["d"], "hard_state_legacy": [0]})
    reject_economic_columns(safe)
    try:
        reject_economic_columns(safe.assign(future_return=[0.1]))
    except ValueError as error:
        assert "economic" in str(error)
    else:
        raise AssertionError("economic outcome entered structural attribution")


def test_chronological_information_increment_is_deterministic_and_oof() -> None:
    rows = []
    for month in range(1, 9):
        for index in range(40):
            is_candidate = index % 2 == 0
            state = 0
            rows.append(
                {
                    "decision_id": f"{month}-{index}",
                    "decision_timestamp": pd.Timestamp(
                        f"2024-{month:02d}-{index % 20 + 1:02d}", tz="UTC"
                    ),
                    "symbol": f"S{index % 10}",
                    "hard_state_legacy": state,
                    "previous_completed_state_1": 1,
                    "previous_completed_state_2": 2,
                    "previous_completed_state_3": 3,
                    "hard_run_age": 1,
                    "recent_state_events": (1, 0) if is_candidate else (2, 0),
                    "previous_completed_primitive_loop": (
                        "loop_p_0-1-0" if is_candidate else "loop_p_2-3-2"
                    ),
                    "same_primitive_repeat_depth": 2 if is_candidate else 1,
                    "bars_since_previous_primitive_completion": 2,
                    "primitive_loop_id": "loop_p_0-1-0" if is_candidate else None,
                }
            )
    frame = pd.DataFrame(rows)
    left = information_increment_from_chronological_folds(frame, candidate_ids=("loop_p_0-1-0",))
    right = information_increment_from_chronological_folds(
        frame.sample(frac=1.0, random_state=9),
        candidate_ids=("loop_p_0-1-0",),
    )
    pd.testing.assert_frame_equal(left, right)
    assert left.iloc[0].scored_fold_count >= 3
    assert left.iloc[0].oof_log_loss_increment > 0


def _prefix_decisions(states: list[int], *, session: str = "2024-01-02") -> pd.DataFrame:
    frame = _decision_frame(states, session=session)
    frame["hard_state_hysteretic"] = frame["hard_state_legacy"]
    frame["posterior_entropy"] = 0.4
    frame["top_second_margin"] = 0.7
    frame["expected_state_age"] = frame.groupby(["symbol", "session"]).cumcount() + 1.0
    frame["hard_run_age"] = 1
    frame["transition_probability_next_bar"] = 0.2
    frame["bars_remaining_in_session"] = len(frame) - frame["bar_ordinal"] - 1
    frame["previous_completed_primitive_loop"] = None
    frame["previous_two_completed_primitive_loops"] = [[] for _ in range(len(frame))]
    frame["same_primitive_repeat_depth"] = 0
    frame["bars_since_previous_primitive_completion"] = None
    frame["state_events_since_previous_primitive_completion"] = None
    return frame


def _prefix_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "semantic_loop_id": "loop_p_0-1-0",
                "primitive_loop_id": "loop_p_0-1-0",
                "closed_path": [0, 1, 0],
                "allowed_orientations": [[0, 1, 0], [1, 0, 1]],
                "motif_type": "primitive",
                "development_count": 200,
                "semi_markov_rate_ratio": 2.0,
            }
        ]
    )


def test_prefix_compression_reconciles_counts_and_required_next_entropy() -> None:
    bundle = build_compressed_prefix_features(
        _prefix_decisions([0, 1, 0]), primary_dictionary=_prefix_dictionary()
    )
    row = bundle.features.loc[bundle.features.decision_id.eq("2024-01-02:1")].iloc[0]
    assert row.active_prefix_count == 2
    assert row.active_prefix_count == len(
        bundle.full_prefixes.loc[bundle.full_prefixes.decision_id.eq("2024-01-02:1")]
    )
    assert row.prefixes_one_transition_away == 1
    assert row.dominant_required_next_state == 0
    assert row.required_next_state_entropy == 0.0
    assert row.fraction_of_prefixes_agreeing_on_dominant_next_state == 1.0


def test_prefix_bar_estimate_uses_duration_hazard_not_transition_count() -> None:
    hazard = np.full((8, 78), 0.5, dtype=float)
    bundle = build_compressed_prefix_features(
        _prefix_decisions([0, 1]),
        primary_dictionary=_prefix_dictionary(),
        duration_hazard=hazard,
    )
    row = bundle.features.loc[bundle.features.decision_id.eq("2024-01-02:1")].iloc[0]
    assert row.minimum_transitions_remaining == 1
    assert row.minimum_bars_remaining_estimate > 1.5


def test_prefix_features_never_export_future_completion_fields() -> None:
    bundle = build_compressed_prefix_features(
        _prefix_decisions([0, 1, 0]), primary_dictionary=_prefix_dictionary()
    )
    forbidden = {
        "future_loop",
        "future_state",
        "bars_until_completion",
        "route_outcome",
        "mfe",
        "mae",
    }
    assert forbidden.isdisjoint(bundle.features.columns)
    assert forbidden.isdisjoint(bundle.manifest["feature_name"])
    assert bundle.features["event_timestamp"].isna().all()


def test_prefix_history_resets_at_session_boundary_and_bars_remaining_is_preserved() -> None:
    decisions = pd.concat(
        [
            _prefix_decisions([0, 1], session="2024-01-02"),
            _prefix_decisions([0], session="2024-01-03"),
        ],
        ignore_index=True,
    )
    bundle = build_compressed_prefix_features(decisions, primary_dictionary=_prefix_dictionary())
    first_new_session = bundle.features.loc[bundle.features.decision_id.eq("2024-01-03:0")].iloc[0]
    assert first_new_session.longest_active_prefix == 1
    assert first_new_session.bars_remaining_in_session == 0
    assert first_new_session.previous_completed_primitive_loop is None


def test_prefix_feature_availability_never_follows_decision() -> None:
    bundle = build_compressed_prefix_features(
        _prefix_decisions([0, 1, 0]), primary_dictionary=_prefix_dictionary()
    )
    assert (
        pd.to_datetime(bundle.features.feature_available_timestamp, utc=True)
        <= pd.to_datetime(bundle.features.decision_timestamp, utc=True)
    ).all()
    assert bundle.manifest["causal_only"].all()


def test_prefix_features_emit_only_eligible_rows_and_reset_after_gap() -> None:
    decisions = _prefix_decisions([0, 1, 0, 1])
    decisions["structural_event_eligibility"] = [True, True, False, True]
    bundle = build_compressed_prefix_features(decisions, primary_dictionary=_prefix_dictionary())
    assert bundle.features["decision_id"].tolist() == [
        "2024-01-02:0",
        "2024-01-02:1",
        "2024-01-02:3",
    ]
    after_gap = bundle.features.iloc[-1]
    assert after_gap.longest_active_prefix == 1
    assert after_gap.active_prefix_count == 1


def test_family_mapping_is_topology_only_and_repeat_status_is_auxiliary() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "decision_id": "exact",
                "primary_class": "loop_p_0-1-2-0",
                "primitive_loop_id": "loop_p_0-1-2-0",
                "primitive_transition_length": 3,
                "current_repeat_depth": 2,
            },
            {
                "decision_id": "other",
                "primary_class": "OTHER_PRIMITIVE_LOOP",
                "primitive_loop_id": "loop_p_0-1-2-3-4-0",
                "primitive_transition_length": 5,
                "current_repeat_depth": 1,
            },
            {
                "decision_id": "no",
                "primary_class": "NO_LOOP_WITHIN_HORIZON",
                "primitive_loop_id": None,
                "primitive_transition_length": None,
                "current_repeat_depth": None,
            },
        ]
    )
    mapping = build_loop_family_mapping(outcomes).set_index("decision_id")
    assert mapping.loc["exact", "loop_family"] == "THREE_STATE_CYCLE"
    assert mapping.loc["exact", "repeat_status"] == "SAME_PRIMITIVE_REPEAT"
    assert mapping.loc["other", "loop_family"] == "FIVE_TO_SIX_STATE_CYCLE"
    assert mapping.loc["no", "loop_family"] == "NO_LOOP"
    assert mapping.loc["no", "repeat_status"] == "NOT_APPLICABLE"


def test_family_outputs_score_complete_primitive_vocabulary_against_nulls() -> None:
    lengths = [2, 3, 4, 5, 7]
    rows = [
        {
            "decision_id": f"d{index}",
            "primary_class": "OTHER_PRIMITIVE_LOOP",
            "primitive_loop_id": f"loop_p_{'-'.join(map(str, range(length)))}-0",
            "primitive_transition_length": length,
            "current_repeat_depth": 1,
        }
        for index, length in enumerate(lengths)
    ]
    development = pd.DataFrame(rows)
    validation = pd.DataFrame(
        [{**row, "decision_id": f"v{index}"} for index, row in enumerate(rows)]
    )
    primary_draws = np.ones((8, 5), dtype=int)
    clock_draws = np.full((4, 5), 2, dtype=int)
    _, _, _, stability = _family_outputs(
        development,
        validation,
        primary_draws,
        clock_draws,
        primary_draws,
        clock_draws,
    )
    assert len(stability) == 10
    assert stability["observed_count"].eq(1).all()
    assert (
        stability[["semi_markov_null_mean", "clock_null_mean", "semi_markov_q", "clock_null_q"]]
        .notna()
        .all()
        .all()
    )


def test_mixed_period_artifact_pins_each_row_to_its_source_snapshot(tmp_path: Path) -> None:
    identity = ArtifactIdentity("run", "sha", "contract", "dev", "dict-v", "dict", "state")
    preflight = PreflightContext(
        output_dir=tmp_path,
        primary_dir=tmp_path,
        contract={},
        contract_hash="contract",
        preflight={},
        git_sha="sha",
        branch="branch",
        development_snapshot_hash="dev",
        validation_snapshot_hash="val",
    )
    context = RunContext(preflight, identity, identity.for_snapshot("val"))
    path = tmp_path / "mixed.csv"
    _write_mixed_period_frame(
        path,
        pd.DataFrame(
            [
                {"period": "development_2024", "value": 1},
                {"period": "unchanged_retrospective_validation_2025", "value": 2},
            ]
        ),
        source_artifact="synthetic",
        context=context,
    )
    written = pd.read_csv(path).set_index("period")
    assert written.loc["development_2024", "data_snapshot_hash"] == "dev"
    assert written.loc["unchanged_retrospective_validation_2025", "data_snapshot_hash"] == "val"


def _exact_gate_metrics() -> dict[str, object]:
    return {
        "dictionary_size": 8,
        "development_coverage": 0.55,
        "validation_coverage": 0.48,
        "entries_rate_ratio_above_one_share": 0.80,
        "entries_threshold_retained_share": 0.60,
        "top_stock_share": 0.15,
        "genuine_tie_rate": 0.01,
        "other_dominated_by_obvious_candidate": False,
        "semantic_ids_stable": True,
        "exact_dictionary_stable_and_informative": True,
        "other_is_diffuse": True,
        "family_reduces_residual_entropy": True,
        "family_coverage_stable_and_higher": True,
        "exact_excess_consistent": True,
        "coverage_collapsed": False,
        "structural_excess_reversed": False,
        "blocked": False,
    }


def test_scientific_decision_metrics_fail_closed_when_a_gate_is_missing() -> None:
    incomplete = _exact_gate_metrics()
    incomplete.pop("other_is_diffuse")
    try:
        decide_target_tractability(incomplete)
    except ValueError as error:
        assert "other_is_diffuse" in str(error)
    else:
        raise AssertionError("missing scientific gate input was silently defaulted")


def test_exact_decision_gate_requires_every_preregistered_condition() -> None:
    decision = decide_target_tractability(_exact_gate_metrics())
    assert (
        decision["decision_label"]
        == "exact_next_loop_identity_tractable_for_preregistered_forecast"
    )
    failed = _exact_gate_metrics()
    failed["validation_coverage"] = 0.44
    assert (
        decide_target_tractability(failed)["decision_label"]
        == "hybrid_exact_dictionary_plus_other_ready_for_forecast"
    )


def test_family_and_instability_decision_gates_are_deterministic() -> None:
    family = _exact_gate_metrics()
    family.update(
        {
            "validation_coverage": 0.20,
            "exact_dictionary_stable_and_informative": False,
            "exact_excess_consistent": False,
        }
    )
    assert (
        decide_target_tractability(family)["decision_label"]
        == "topological_loop_family_target_preferred"
    )
    collapsed = _exact_gate_metrics()
    collapsed["coverage_collapsed"] = True
    assert (
        decide_target_tractability(collapsed)["decision_label"]
        == "semantic_loop_dictionary_not_stable"
    )


def test_uncovered_stable_low_coverage_decision_region_fails_closed() -> None:
    uncovered = _exact_gate_metrics()
    uncovered.update(
        {
            "dictionary_size": 2,
            "development_coverage": 0.12,
            "validation_coverage": 0.11,
            "exact_excess_consistent": True,
        }
    )
    decision = decide_target_tractability(uncovered)
    assert decision["decision_label"] == "semantic_dictionary_experiment_blocked"
    assert decision["next_loop_predictor_justified"] is False
    assert decision["decision_rule_gap"] is True


def test_period_replication_columns_are_explicit_and_complete() -> None:
    dictionary = pd.DataFrame(
        [
            {
                "semantic_loop_id": "loop_p_0-1-0",
                "development_count": 120,
                "semi_markov_rate_ratio": 1.5,
                "semi_markov_q": 0.05,
                "selection_rank": 1,
            }
        ]
    )
    development_null = pd.DataFrame([{"semantic_loop_id": "loop_p_0-1-0", "excess_count": 40.0}])
    validation_null = pd.DataFrame(
        [
            {
                "semantic_loop_id": "loop_p_0-1-0",
                "observed_count": 95,
                "semi_markov_rate_ratio": 1.3,
                "semi_markov_p": 0.01,
                "semi_markov_q": 0.04,
                "excess_count": 22.0,
            }
        ]
    )
    replication = _dictionary_replication(dictionary, development_null, validation_null)
    assert replication.loc[0, "observed_count_validation"] == 95
    assert replication.loc[0, "excess_count_development"] == 40.0
    assert replication.loc[0, "excess_count_validation"] == 22.0
    assert replication.loc[0, "semi_markov_rate_ratio_development"] == 1.5
    assert replication.loc[0, "semi_markov_rate_ratio_validation"] == 1.3


def test_exact_rerun_status_excludes_only_self_generated_outputs() -> None:
    status = "\n".join(
        [
            "?? packages/stocker_research/src/stocker_research/new_module.py",
            "?? research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/"
            "20260718-semantic-loop-dictionary-coverage-v2/",
            "?? research/slrno-v2/20260714-regime-loop-handoff/work/reports/"
            "20260718-semantic-loop-dictionary-coverage-v2.md",
        ]
    )
    assert _filter_generated_git_status(status) == (
        "?? packages/stocker_research/src/stocker_research/new_module.py"
    )
