from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel
from stocker_research.loop_ledger_v2 import (
    adapt_legacy_overlapping_target_panel,
    adapt_legacy_run_ledger,
    build_loop_event_ledgers,
    compare_legacy_targets_to_v2_outcomes,
)

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _dictionary(*paths: tuple[int, ...]) -> LoopDictionary:
    return LoopDictionary.from_definitions(
        (decompose_closed_path(path) for path in paths),
        version="semantic_loop_dictionary_v2_test",
    )


def _decisions(
    states: list[int],
    *,
    session: str = "2024-01-02",
    minute_offsets: list[int] | None = None,
) -> pd.DataFrame:
    offsets = minute_offsets or [5 * index for index in range(len(states))]
    rows = []
    for index, (state, offset) in enumerate(zip(states, offsets, strict=True)):
        start = BASE + timedelta(minutes=offset)
        rows.append(
            {
                "decision_id": f"d{session}-{index}",
                "run_id": 0,
                "git_sha": "a" * 40,
                "contract_hash": "b" * 64,
                "data_snapshot_hash": "c" * 64,
                "dictionary_version": "semantic_loop_dictionary_v2_test",
                "state_model_version": "state_v2_test",
                "symbol": "AAA",
                "session": session,
                "bar_ordinal": index,
                "bar_start_timestamp": start,
                "bar_complete_timestamp": start + timedelta(minutes=5),
                "decision_timestamp": start + timedelta(minutes=5),
                "hard_state_legacy": state,
                "hard_state_hysteretic": state,
                "posterior_state_probabilities": [
                    1.0 if item == state else 0.0 for item in range(8)
                ],
                "posterior_entropy": 0.0,
                "top_second_margin": 1.0,
                "hard_run_age": 1,
                "expected_state_age": 1.0,
                "transition_probability_next_bar": 0.5,
                "bars_remaining_in_session": len(states) - index - 1,
                "is_run_entry": index == 0 or state != states[index - 1],
                "research_only": True,
                "execution_enabled": False,
                "order_placement": "disabled",
                "broker_connected": False,
                "strategy_promotion": False,
            }
        )
    return pd.DataFrame(rows)


def test_every_completed_bar_gets_one_prefix_aware_outcome() -> None:
    decisions = _decisions([2, 4, 4, 2, 3])
    bundle = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((2, 4, 2)),
        horizon_bars=3,
        allowed_states=frozenset(range(8)),
    )

    assert len(bundle.decisions) == len(decisions)
    assert len(bundle.outcomes) == len(decisions)
    inside_run = bundle.outcomes.set_index("decision_id").loc["d2024-01-02-2"]
    assert inside_run["primary_label"] == "loop_p_2-4-2"
    assert inside_run["bars_until_completion"] == 1
    assert bool(inside_run["first_event_was_open_prefix"])
    assert bundle.decisions.loc[2, "active_prefix_count"] > 0


def test_legacy_rotation_differs_from_prefix_event_on_per_bar_surface() -> None:
    decisions = _decisions([2, 4, 4, 2])
    bundle = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((2, 4, 2)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )
    comparison = bundle.target_comparison.set_index("decision_id").loc["d2024-01-02-2"]

    assert comparison["legacy_positive_count"] == 0
    assert comparison["v2_first_event"] == "loop_p_2-4-2"
    assert bool(comparison["semantics_differ"])


def test_missing_bar_gap_fails_closed_without_dropping_decisions() -> None:
    decisions = _decisions([1, 2, 1], minute_offsets=[0, 5, 15])
    bundle = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((1, 2, 1)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )

    assert len(bundle.outcomes) == 3
    assert set(bundle.outcomes["primary_label"]) == {PrimaryOutcomeLabel.UNAVAILABLE}
    assert not bundle.decisions["structural_event_eligibility"].any()
    assert bundle.prefixes.empty
    assert not bundle.target_comparison["comparison_available"].any()
    assert not bundle.target_comparison["semantics_differ"].any()


def test_session_boundary_prevents_prefix_bridge() -> None:
    first = _decisions([2, 4], session="2024-01-02")
    second = _decisions([2], session="2024-01-03")
    decisions = pd.concat([first, second], ignore_index=True)
    bundle = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((2, 4, 2)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )

    next_session = bundle.outcomes.loc[bundle.outcomes["session"].eq("2024-01-03"), "primary_label"]
    assert next_session.tolist() == [PrimaryOutcomeLabel.SESSION_END]
    assert bundle.completions.empty


def test_decision_feature_surface_contains_history_not_outcomes() -> None:
    bundle = build_loop_event_ledgers(
        _decisions([1, 2, 1, 3, 1]),
        dictionary=_dictionary((1, 2, 1), (1, 2, 1, 3, 1)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )

    expected = {
        "previous_completed_state_1",
        "previous_completed_state_4",
        "previous_primitive_loop_1",
        "bars_since_previous_loop",
        "same_loop_repeat_depth",
        "active_primitive_prefixes",
        "active_composite_prefixes",
        "shortest_transitions_remaining",
        "highest_soft_prefix_probability",
        "soft_completion_probabilities",
    }
    assert expected <= set(bundle.decisions)
    forbidden = ("primary_label", "future", "payoff", "mfe", "mae", "route_completion")
    assert not any(token in column.lower() for column in bundle.decisions for token in forbidden)
    assert bundle.decisions["research_only"].all()
    assert not bundle.decisions["execution_enabled"].any()


def test_same_event_ties_remain_one_primary_label_and_several_event_rows() -> None:
    dictionary = _dictionary((1, 3, 1), (1, 3, 1, 3, 1))
    bundle = build_loop_event_ledgers(
        _decisions([1, 3, 1, 3, 1]),
        dictionary=dictionary,
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )
    outcome = bundle.outcomes.set_index("decision_id").loc["d2024-01-02-3"]
    tied_rows = bundle.completions.loc[bundle.completions["decision_id"].eq("d2024-01-02-3")]

    assert outcome["primary_label"] == PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION
    assert len(outcome["tied_semantic_loop_ids"]) == 2
    assert tied_rows["tied_completion"].all()
    assert len(tied_rows) == 2


def test_completion_ledger_retains_later_registered_event_after_unregistered_primary() -> None:
    bundle = build_loop_event_ledgers(
        _decisions([1, 3, 1, 2, 1]),
        dictionary=_dictionary((1, 2, 1)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )
    outcome = bundle.outcomes.set_index("decision_id").loc["d2024-01-02-0"]
    later = bundle.completions.loc[bundle.completions["decision_id"].eq("d2024-01-02-0")]

    assert outcome["primary_label"] == PrimaryOutcomeLabel.UNREGISTERED_LOOP
    assert not bool(outcome["first_completion_same_as_previous_primitive_loop"])
    assert later["semantic_loop_id"].tolist() == ["loop_p_1-2-1"]
    assert later["bars_until_completion"].tolist() == [4]


def test_cross_dictionary_comparison_uses_semantic_v2_primary_outcome() -> None:
    decisions = _decisions([1, 3, 1, 2, 1])
    legacy = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((1, 3, 1)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )
    semantic = build_loop_event_ledgers(
        decisions,
        dictionary=_dictionary((1, 2, 1)),
        horizon_bars=4,
        allowed_states=frozenset(range(8)),
    )
    comparison = compare_legacy_targets_to_v2_outcomes(
        legacy.legacy_targets,
        semantic.outcomes,
        semantic.decisions,
    ).set_index("decision_id")
    first = comparison.loc["d2024-01-02-0"]

    assert first["legacy_positive_labels"] == ["loop_p_1-3-1"]
    assert first["v2_first_event"] == PrimaryOutcomeLabel.UNREGISTERED_LOOP
    assert first["v2_only_events"] == [PrimaryOutcomeLabel.UNREGISTERED_LOOP]
    assert bool(first["registered_event_set_differs"])
    assert bool(first["semantics_differ"])


def test_legacy_run_and_overlapping_target_adapters_fail_closed() -> None:
    runs = adapt_legacy_run_ledger(
        pd.DataFrame(
            {
                "run_id": [4],
                "symbol": ["AAA"],
                "session": ["2024-01-02"],
                "state": [2],
                "duration": [3],
                "start_timestamp": [BASE],
                "end_timestamp": [BASE + timedelta(minutes=10)],
            }
        )
    )
    dictionary = LoopDictionary.from_legacy_table(
        pd.DataFrame(
            {
                "legacy_cycle_id": ["cycle_01"],
                "cycle": ["2->4->2"],
                "discovery_rank": [1],
            }
        ),
        version="legacy_dictionary_v1",
    )
    targets = adapt_legacy_overlapping_target_panel(
        pd.DataFrame(
            {
                "decision_id": ["d1", "d1"],
                "legacy_cycle_id": ["cycle_01", "cycle_unknown"],
                "target": [1, 1],
            }
        ),
        dictionary=dictionary,
    )

    assert runs.loc[0, "legacy_run_id"] == 4
    assert runs.loc[0, "migration_status"] == "compatible_read_only"
    assert targets.loc[0, "semantic_loop_id"] == "loop_p_2-4-2"
    assert targets.loc[1, "migration_status"] == "unavailable"
    assert targets.loc[1, "ambiguity_reason"] == "unknown_legacy_cycle_id"
