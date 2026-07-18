from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel
from stocker_research.loop_prefix_automaton_v2 import (
    FirstNextLoopEventEngine,
    LoopEventAutomaton,
    UnknownStateError,
    legacy_compatible_cycle_labels,
)

BASE = datetime(2024, 1, 2, 14, 35, tzinfo=UTC)


def _dictionary(*paths: tuple[int, ...]) -> LoopDictionary:
    return LoopDictionary.from_definitions(
        (decompose_closed_path(path) for path in paths),
        version="semantic_loop_dictionary_v2_test",
    )


def _trace(
    dictionary: LoopDictionary,
    states: tuple[int, ...],
    *,
    bars: tuple[int, ...] | None = None,
):
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    ordinals = bars or tuple(range(len(states)))
    timestamps = tuple(BASE + timedelta(minutes=5 * value) for value in ordinals)
    return engine, engine.scan_state_events(
        states,
        bar_ordinals=ordinals,
        event_timestamps=timestamps,
        available_timestamps=timestamps,
    )


def test_one_transition_away_active_prefix_completes_on_next_state_event() -> None:
    dictionary = _dictionary((2, 4, 2))
    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))

    automaton.feed(2, event_timestamp=BASE, available_timestamp=BASE, bar_ordinal=0)
    automaton.feed(
        4,
        event_timestamp=BASE + timedelta(minutes=5),
        available_timestamp=BASE + timedelta(minutes=5),
        bar_ordinal=1,
    )
    active = automaton.active_prefixes()
    assert any(
        prefix.prefix_path == (2, 4) and prefix.transitions_remaining == 1 for prefix in active
    )

    completed = automaton.feed(
        2,
        event_timestamp=BASE + timedelta(minutes=10),
        available_timestamp=BASE + timedelta(minutes=10),
        bar_ordinal=2,
    )
    assert [event.semantic_loop_id for event in completed] == ["loop_p_2-4-2"]
    assert completed[0].state_events_until_completion == 2


def test_two_transition_away_prefix_progresses_without_restart() -> None:
    dictionary = _dictionary((2, 4, 6, 2))
    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))
    for index, state in enumerate((2, 4)):
        automaton.feed(
            state,
            event_timestamp=BASE + timedelta(minutes=5 * index),
            available_timestamp=BASE + timedelta(minutes=5 * index),
            bar_ordinal=index,
        )

    active = automaton.active_prefixes()
    assert any(
        prefix.prefix_path == (2, 4) and prefix.transitions_remaining == 2 for prefix in active
    )
    assert (
        automaton.feed(
            6,
            event_timestamp=BASE + timedelta(minutes=10),
            available_timestamp=BASE + timedelta(minutes=10),
            bar_ordinal=2,
        )
        == ()
    )
    completed = automaton.feed(
        2,
        event_timestamp=BASE + timedelta(minutes=15),
        available_timestamp=BASE + timedelta(minutes=15),
        bar_ordinal=3,
    )
    assert completed[0].semantic_loop_id == "loop_p_2-4-6-2"


def test_shared_prefix_and_multiple_active_prefixes_are_retained() -> None:
    dictionary = _dictionary((1, 2, 1), (1, 2, 3, 1))
    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))
    for index, state in enumerate((1, 2)):
        automaton.feed(
            state,
            event_timestamp=BASE + timedelta(minutes=5 * index),
            available_timestamp=BASE + timedelta(minutes=5 * index),
            bar_ordinal=index,
        )

    active = automaton.active_prefixes()
    matching = {
        (prefix.semantic_loop_id, prefix.prefix_path, prefix.transitions_remaining)
        for prefix in active
        if prefix.prefix_path == (1, 2)
    }
    assert matching == {
        ("loop_p_1-2-1", (1, 2), 1),
        ("loop_p_1-2-3-1", (1, 2), 2),
    }


def test_earlier_primitive_completion_precedes_later_repeat_completion() -> None:
    dictionary = _dictionary((1, 3, 1), (1, 3, 1, 3, 1))
    engine, trace = _trace(dictionary, (1, 3, 1, 3, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="d0",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert outcome.primary_label == "loop_p_1-3-1"
    assert outcome.bars_until_completion == 2
    assert "loop_r2_1-3-1" in outcome.every_registered_completion_within_horizon
    assert {
        (event.semantic_loop_id, event.completion_event_index)
        for event in outcome.every_registered_completion_event
    } >= {
        ("loop_p_1-3-1", 2),
        ("loop_r2_1-3-1", 4),
    }


def test_same_event_primitive_and_repeat_completion_remains_a_tie() -> None:
    dictionary = _dictionary((1, 3, 1), (1, 3, 1, 3, 1))
    engine, trace = _trace(dictionary, (1, 3, 1, 3, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="d3",
        decision_event_index=3,
        decision_bar_ordinal=3,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert outcome.primary_label == PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION
    assert outcome.tied_semantic_loop_ids == ("loop_p_1-3-1", "loop_r2_1-3-1")
    assert all(event.tied_completion for event in outcome.earliest_registered_events)
    assert any(event.nested_completion for event in outcome.earliest_registered_events)


def test_composite_completion_does_not_erase_earlier_primitive_event() -> None:
    dictionary = _dictionary((1, 2, 1), (1, 2, 1, 3, 1))
    engine, trace = _trace(dictionary, (1, 2, 1, 3, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="d0",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert outcome.primary_label == "loop_p_1-2-1"
    assert any(
        value.startswith("loop_c_") for value in outcome.every_registered_completion_within_horizon
    )


def test_session_boundary_and_missing_event_clear_prefix_state() -> None:
    dictionary = _dictionary((2, 4, 2))
    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))
    for index, state in enumerate((2, 4)):
        automaton.feed(
            state,
            event_timestamp=BASE + timedelta(minutes=5 * index),
            available_timestamp=BASE + timedelta(minutes=5 * index),
            bar_ordinal=index,
        )
    assert any(prefix.prefix_path == (2, 4) for prefix in automaton.active_prefixes())

    automaton.mark_missing()
    assert automaton.active_prefixes() == ()
    assert (
        automaton.feed(
            2,
            event_timestamp=BASE + timedelta(minutes=10),
            available_timestamp=BASE + timedelta(minutes=10),
            bar_ordinal=2,
        )
        == ()
    )
    automaton.reset_session()
    assert automaton.active_prefixes() == ()


def test_unknown_state_fails_closed_and_clears_state() -> None:
    dictionary = _dictionary((2, 4, 2))
    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))
    automaton.feed(2, event_timestamp=BASE, available_timestamp=BASE, bar_ordinal=0)

    with pytest.raises(UnknownStateError):
        automaton.feed(
            99,
            event_timestamp=BASE + timedelta(minutes=5),
            available_timestamp=BASE + timedelta(minutes=5),
            bar_ordinal=1,
        )
    assert automaton.active_prefixes() == ()


def test_event_trace_rejects_adjacent_duplicate_state_events() -> None:
    dictionary = _dictionary((2, 4, 2))
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    timestamps = (BASE, BASE + timedelta(minutes=5), BASE + timedelta(minutes=10))

    with pytest.raises(ValueError, match="adjacent duplicate"):
        engine.scan_state_events(
            (2, 2, 4),
            bar_ordinals=(0, 1, 2),
            event_timestamps=timestamps,
            available_timestamps=timestamps,
        )


def test_no_registered_completion_and_session_end_have_distinct_labels() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (1, 3, 4), bars=(0, 1, 5))
    no_loop = engine.outcome_for_decision(
        trace,
        decision_id="no_loop",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=2,
        session_end_bar_ordinal=20,
    )
    assert no_loop.primary_label == PrimaryOutcomeLabel.NO_REGISTERED_LOOP_WITHIN_HORIZON

    _, terminal_trace = _trace(dictionary, (1,))
    terminal = engine.outcome_for_decision(
        terminal_trace,
        decision_id="terminal",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=24,
        session_end_bar_ordinal=10,
    )
    assert terminal.primary_label == PrimaryOutcomeLabel.SESSION_END


def test_unregistered_closed_loop_can_be_the_first_structural_event() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (1, 3, 1, 2, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="unregistered",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert outcome.primary_label == PrimaryOutcomeLabel.UNREGISTERED_LOOP
    assert outcome.bars_until_completion == 2
    assert [event.semantic_loop_id for event in outcome.every_registered_completion_event] == [
        "loop_p_1-2-1"
    ]


def test_loop_initiated_after_decision_is_identified_as_such() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (5, 1, 2, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="later",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert outcome.primary_label == "loop_p_1-2-1"
    assert outcome.first_event_began_after_decision is True
    assert outcome.first_event_was_open_prefix is False


def test_one_state_prefix_at_decision_is_explicitly_open() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (1, 2, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="at_decision",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )

    event = outcome.earliest_registered_events[0]
    assert outcome.first_event_was_open_prefix is True
    assert event.initiated_at_decision is True
    assert event.initiated_before_decision is False
    assert event.initiated_after_decision is False


def test_per_bar_decision_time_is_not_replaced_by_state_event_time() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (1, 2, 1), bars=(0, 4, 8))
    completed_bar_decision = BASE + timedelta(minutes=15)

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="inside_run",
        decision_event_index=0,
        decision_bar_ordinal=2,
        decision_timestamp=completed_bar_decision,
        decision_available_timestamp=completed_bar_decision,
        horizon_bars=10,
        session_end_bar_ordinal=20,
    )

    assert outcome.primary_label == "loop_p_1-2-1"
    event = outcome.earliest_registered_events[0]
    assert event.decision_timestamp == completed_bar_decision
    assert event.decision_available_timestamp == completed_bar_decision
    assert event.initiated_before_decision is True
    assert event.bars_until_completion == 6


def test_legacy_rotation_differs_from_active_prefix_first_event() -> None:
    dictionary = _dictionary((2, 4, 2))
    engine, trace = _trace(dictionary, (2, 4, 2))

    legacy = legacy_compatible_cycle_labels(trace, decision_event_index=1, dictionary=dictionary)
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="prefix",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert legacy == ()
    assert outcome.primary_label == "loop_p_2-4-2"
    assert outcome.first_event_was_open_prefix is True
    assert outcome.transitions_remaining_at_decision == 1


def test_legacy_multiple_positives_remain_secondary_only() -> None:
    dictionary = _dictionary((1, 2, 1), (1, 2, 1, 3, 1))
    engine, trace = _trace(dictionary, (1, 2, 1, 3, 1))

    legacy = legacy_compatible_cycle_labels(trace, decision_event_index=0, dictionary=dictionary)
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="overlap",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
    )
    assert len(legacy) == 2
    assert outcome.primary_label == "loop_p_1-2-1"
    assert isinstance(outcome.primary_label, str)


def test_unavailable_source_has_frozen_precedence() -> None:
    dictionary = _dictionary((1, 2, 1))
    engine, trace = _trace(dictionary, (1, 2, 1))

    outcome = engine.outcome_for_decision(
        trace,
        decision_id="bad_source",
        decision_event_index=0,
        decision_bar_ordinal=0,
        horizon_bars=10,
        session_end_bar_ordinal=10,
        source_available=False,
    )
    assert outcome.primary_label == PrimaryOutcomeLabel.UNAVAILABLE
    assert outcome.earliest_registered_events == ()
