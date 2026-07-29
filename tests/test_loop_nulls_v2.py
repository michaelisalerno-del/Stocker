from __future__ import annotations

from dataclasses import fields

import numpy as np

from stocker_research.loop_dictionary_v2 import (
    DictionaryCandidateMetrics,
    candidate_selection_score,
    decompose_closed_path,
    loop_complexity_penalty,
    select_dictionary_candidates,
)
from stocker_research.loop_nulls_v2 import (
    CLOCK_PHASE_BOUNDARIES,
    ClockConditionedSemiMarkovNull,
    SemiMarkovNull,
    SessionRunSequence,
    benjamini_hochberg,
    circular_session_control,
    count_candidate_paths,
    empirical_p_values,
    first_order_expected_counts,
    session_phase,
    simulate_null_counts,
)


def _alternating_higher_order_sessions(count: int = 40) -> tuple[SessionRunSequence, ...]:
    states = tuple([0, 1, 0, 2] * 10 + [0])
    durations = (1,) * len(states)
    return tuple(
        SessionRunSequence(
            symbol=f"S{index:02d}",
            session=f"2024-01-{index + 1:02d}",
            states=states,
            durations=durations,
            terminal_right_censored=False,
        )
        for index in range(count)
    )


def test_semimarkov_simulation_preserves_session_length_and_valid_states() -> None:
    sessions = _alternating_higher_order_sessions(4)
    model = SemiMarkovNull.fit(sessions, state_count=3, maximum_duration=78)
    rng = np.random.default_rng(20260718)

    simulated = model.simulate_session(41, rng=rng)

    assert sum(simulated.durations) == 41
    assert set(simulated.states) <= {0, 1, 2}
    assert len(simulated.states) == len(simulated.durations)
    assert simulated.terminal_right_censored is True


def test_clock_conditioned_null_uses_frozen_broad_phase_definitions() -> None:
    assert CLOCK_PHASE_BOUNDARIES == (0, 12, 60, 78)
    assert session_phase(0) == "opening"
    assert session_phase(11) == "opening"
    assert session_phase(12) == "middle"
    assert session_phase(59) == "middle"
    assert session_phase(60) == "late"
    assert session_phase(77) == "late"

    model = ClockConditionedSemiMarkovNull.fit(
        _alternating_higher_order_sessions(4), state_count=3, maximum_duration=78
    )
    assert model.phase_boundaries == CLOCK_PHASE_BOUNDARIES
    simulated = model.simulate_session(78, rng=np.random.default_rng(19))
    assert sum(simulated.durations) == 78
    assert simulated.phase_labels[0] == "opening"
    assert simulated.phase_labels[-1] == "late"


def test_structural_null_input_has_no_economic_outcome_surface() -> None:
    field_names = {item.name.lower() for item in fields(SessionRunSequence)}
    forbidden = {"return", "payoff", "pnl", "mfe", "mae", "profit"}

    assert field_names.isdisjoint(forbidden)


def test_whole_session_circular_control_preserves_state_duration_pairs() -> None:
    source = SessionRunSequence(
        symbol="AAA",
        session="2024-01-02",
        states=(0, 1, 2, 3),
        durations=(2, 3, 4, 5),
        terminal_right_censored=True,
    )
    rotated = circular_session_control(source, offset=2)

    assert rotated.states == (2, 3, 0, 1)
    assert rotated.durations == (4, 5, 2, 3)
    assert sorted(zip(rotated.states, rotated.durations, strict=True)) == sorted(
        zip(source.states, source.durations, strict=True)
    )
    assert sum(rotated.durations) == sum(source.durations)


def test_circular_control_merges_equal_states_at_rotation_seam() -> None:
    source = SessionRunSequence(
        symbol="AAA",
        session="2024-01-02",
        states=(0, 1, 0),
        durations=(2, 3, 4),
        terminal_right_censored=True,
    )

    rotated = circular_session_control(source, offset=1)

    assert rotated.states == (1, 0)
    assert rotated.durations == (3, 6)
    assert sum(rotated.durations) == sum(source.durations)


def test_first_order_expected_counts_match_supported_path_formula() -> None:
    sessions = _alternating_higher_order_sessions(2)
    model = SemiMarkovNull.fit(sessions, state_count=3, maximum_duration=78)
    candidates = ((0, 1, 0), (0, 1, 0, 2, 0))
    expected = first_order_expected_counts(sessions, model, candidates)

    assert expected.shape == (2,)
    assert expected[0] > expected[1] > 0.0


def test_first_order_cycle_is_not_false_positive_but_higher_order_motif_is() -> None:
    sessions = _alternating_higher_order_sessions(40)
    candidates = ((0, 1, 0), (0, 1, 0, 2, 0))
    model = SemiMarkovNull.fit(sessions, state_count=3, maximum_duration=78)
    observed = count_candidate_paths(sessions, candidates)
    draws = simulate_null_counts(
        model,
        session_lengths=[sum(session.durations) for session in sessions],
        candidates=candidates,
        draws=499,
        seed=20260718,
    )
    p_values = empirical_p_values(observed, draws)

    assert p_values[0] > 0.05
    assert p_values[1] < 0.01


def test_empirical_p_values_and_fdr_are_deterministic() -> None:
    observed = np.asarray([5, 10, 3])
    draws = np.asarray(
        [
            [1, 11, 3],
            [2, 9, 4],
            [6, 8, 1],
            [4, 7, 2],
        ]
    )
    first_p = empirical_p_values(observed, draws)
    second_p = empirical_p_values(observed, draws.copy())
    first_q = benjamini_hochberg(first_p)
    second_q = benjamini_hochberg(second_p)

    assert np.array_equal(first_p, second_p)
    assert np.array_equal(first_q, second_q)
    assert np.all(first_q >= first_p)


def test_dictionary_score_separates_support_from_information() -> None:
    frequent = DictionaryCandidateMetrics(
        definition=decompose_closed_path((0, 1, 0)),
        eligible_anchor_count=10_000,
        observed_completions=1_000,
        expected_completions=990.0,
        empirical_p_value=0.4,
        fdr_q_value=0.4,
        conditional_information_gain=0.0001,
        increment_beyond_current_state=0.0001,
        increment_beyond_previous_state_history=0.0001,
        stock_breadth=20,
        month_breadth=12,
        clock_breadth=3,
        period_consistency=0.6,
    )
    informative = DictionaryCandidateMetrics(
        definition=decompose_closed_path((0, 2, 0)),
        eligible_anchor_count=2_000,
        observed_completions=200,
        expected_completions=50.0,
        empirical_p_value=0.001,
        fdr_q_value=0.002,
        conditional_information_gain=0.08,
        increment_beyond_current_state=0.06,
        increment_beyond_previous_state_history=0.04,
        stock_breadth=12,
        month_breadth=10,
        clock_breadth=3,
        period_consistency=0.9,
    )

    assert frequent.observed_completions > informative.observed_completions
    assert candidate_selection_score(informative) > candidate_selection_score(frequent)
    selected = select_dictionary_candidates((frequent, informative), maximum_entries=2)
    assert selected[0].definition.semantic_loop_id == "loop_p_0-2-0"


def test_complexity_penalty_is_deterministic_and_prefers_simple_primitive() -> None:
    primitive = decompose_closed_path((1, 2, 1))
    repeated = decompose_closed_path((1, 2, 1, 2, 1))
    composite = decompose_closed_path((1, 2, 1, 3, 1))

    assert loop_complexity_penalty(primitive) == loop_complexity_penalty(primitive)
    assert loop_complexity_penalty(primitive) < loop_complexity_penalty(repeated)
    assert loop_complexity_penalty(primitive) < loop_complexity_penalty(composite)


def test_dictionary_selection_keeps_primitive_dependencies_before_larger_motifs() -> None:
    primitive_definition = decompose_closed_path((1, 2, 1))
    repeat_definition = decompose_closed_path((1, 2, 1, 2, 1))

    def metrics(definition, information: float) -> DictionaryCandidateMetrics:
        return DictionaryCandidateMetrics(
            definition=definition,
            eligible_anchor_count=1_000,
            observed_completions=400,
            expected_completions=50.0,
            empirical_p_value=0.001,
            fdr_q_value=0.001,
            conditional_information_gain=information,
            increment_beyond_current_state=information,
            increment_beyond_previous_state_history=information,
            stock_breadth=20,
            month_breadth=12,
            clock_breadth=3,
            period_consistency=1.0,
        )

    selected = select_dictionary_candidates(
        (metrics(repeat_definition, 1.0), metrics(primitive_definition, 0.01)),
        maximum_entries=2,
    )

    assert [item.definition.semantic_loop_id for item in selected] == [
        "loop_p_1-2-1",
        "loop_r2_1-2-1",
    ]
