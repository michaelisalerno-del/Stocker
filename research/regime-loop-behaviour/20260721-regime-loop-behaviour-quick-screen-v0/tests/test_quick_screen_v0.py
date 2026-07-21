from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stocker_research.causal_state_export_v2 import causal_semimarkov_filter_v2
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.regime_loop_behaviour_quick_v0 import (
    BEHAVIOURAL_DIMENSIONS,
    INTERACTION_FEATURES,
    active_prefix_records,
    apply_interaction_clipping,
    assert_decision_time_causality,
    assert_no_protected_rows,
    assign_candidate_weights,
    causal_checkpoint_filter,
    compute_interactions,
    decide_screen,
    fit_candidate_logistic,
    fit_interaction_clipping,
    manual_logistic_probabilities,
    normalize_first_event_outcome,
    permute_behavioural_bundle_within_slates,
    session_block_bootstrap_draws,
    target_candidate_rows,
    verify_behavioural_ledger_reconstruction,
)


def _timestamp(minutes: int) -> datetime:
    return datetime(2025, 2, 3, 14, 30, tzinfo=UTC) + timedelta(minutes=minutes)


def _engine(*paths: tuple[int, ...]) -> FirstNextLoopEventEngine:
    dictionary = LoopDictionary.from_definitions(
        [decompose_closed_path(path) for path in paths],
        version="test_dictionary",
    )
    return FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))


def _trace(
    engine: FirstNextLoopEventEngine,
    states: list[int],
    bars: list[int] | None = None,
):
    ordinals = bars if bars is not None else list(range(len(states)))
    timestamps = [_timestamp(5 * value) for value in ordinals]
    return engine.scan_state_events(
        states,
        bar_ordinals=ordinals,
        event_timestamps=timestamps,
        available_timestamps=[value + timedelta(minutes=5) for value in timestamps],
    )


def _behavioural_rows() -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "symbol": ["AAL", "AAOI"],
        "session": ["2025-02-03", "2025-02-03"],
        "decision_ordinal": [6, 6],
        "decision_timestamp": [pd.Timestamp(_timestamp(30)), pd.Timestamp(_timestamp(30))],
    }
    for index, column in enumerate(BEHAVIOURAL_DIMENSIONS):
        values[column] = [float(index), float(index + 1)]
    return pd.DataFrame(values)


def test_behavioural_ledger_join_requires_exact_natural_keys_timestamps_and_values() -> None:
    primary = _behavioural_rows()
    rerun = primary.copy()
    result = verify_behavioural_ledger_reconstruction(primary, rerun)
    assert result["passed"] is True
    assert result["maximum_absolute_error"] == 0.0

    rerun.loc[1, "tension"] += 2e-12
    with pytest.raises(ValueError, match="not reconstructable"):
        verify_behavioural_ledger_reconstruction(primary, rerun)


def test_decision_features_must_be_available_by_completed_checkpoint_bar() -> None:
    frame = pd.DataFrame(
        {
            "source_timestamp": [pd.Timestamp(_timestamp(25))],
            "available_timestamp": [pd.Timestamp(_timestamp(30))],
            "decision_timestamp": [pd.Timestamp(_timestamp(30))],
        }
    )
    assert_decision_time_causality(frame)
    frame.loc[0, "available_timestamp"] = pd.Timestamp(_timestamp(35))
    with pytest.raises(ValueError, match="after decision"):
        assert_decision_time_causality(frame)


def test_active_prefix_reconstruction_requires_a_causal_transition() -> None:
    engine = _engine((0, 1, 0), (0, 1, 2, 0))
    trace = _trace(engine, [0, 1])
    rows = active_prefix_records(trace, decision_event_index=1, decision_bar_ordinal=1)
    assert {row["semantic_loop_id"] for row in rows} == {
        "loop_p_0-1-0",
        "loop_p_0-1-2-0",
    }
    assert {row["prefix_matched_length"] for row in rows} == {1}
    assert all(row["prefix_completion_fraction"] > 0.0 for row in rows)

    initial_trace = _trace(engine, [0])
    assert (
        active_prefix_records(
            initial_trace,
            decision_event_index=0,
            decision_bar_ordinal=0,
        )
        == []
    )


def test_memory_bounded_checkpoint_filter_matches_v2_forward_recursion() -> None:
    emissions = np.array([[0.0, -1.0], [-0.2, -0.1], [-1.0, 0.0]])
    model = {
        "duration_hazard": np.array([[0.2, 1.0], [0.3, 1.0]]),
        "transitions": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "initial": np.array([0.6, 0.4]),
    }
    groups = [np.array([0, 1, 2])]
    starts = [_timestamp(index * 5) for index in range(3)]
    expected = causal_semimarkov_filter_v2(
        emissions,
        session_groups=groups,
        model=model,
        bar_start_timestamps=starts,
        bar_duration=timedelta(minutes=5),
    )
    actual = causal_checkpoint_filter(emissions, groups=groups, model=model)
    np.testing.assert_allclose(actual.state_probabilities, expected.state_probabilities)
    np.testing.assert_allclose(
        actual.next_state_probabilities,
        expected.next_state_probabilities,
    )
    np.testing.assert_allclose(actual.expected_state_age, expected.expected_state_age)
    np.testing.assert_allclose(
        actual.current_transition_probability,
        expected.probability_state_transitions_next_bar,
    )


def test_candidate_weighting_equalises_candidates_then_stocks_within_slate() -> None:
    candidates = pd.DataFrame(
        {
            "slate_id": ["s1", "s1", "s1", "s1"],
            "decision_id": ["A", "A", "A", "B"],
            "symbol": ["AAL", "AAL", "AAL", "AAOI"],
        }
    )
    weighted = assign_candidate_weights(candidates)
    np.testing.assert_allclose(weighted["candidate_weight"], [1 / 3, 1 / 3, 1 / 3, 1.0])
    np.testing.assert_allclose(weighted["slate_weight"], [0.5, 0.5, 0.5, 0.5])
    assert weighted.groupby("decision_id")["row_weight"].sum().to_dict() == pytest.approx(
        {"A": 0.5, "B": 0.5}
    )
    assert weighted.groupby("slate_id")["row_weight"].sum().iloc[0] == pytest.approx(1.0)


def test_six_bar_horizon_uses_first_registered_completion() -> None:
    engine = _engine((0, 1, 0), (1, 2, 1))
    trace = _trace(engine, [0, 1, 2, 1, 0], [0, 1, 2, 3, 7])
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="d",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    assert outcome.primary_label == "loop_p_1-2-1"
    assert outcome.bars_until_completion == 2


def test_exact_oriented_first_completion_is_the_only_positive_candidate() -> None:
    engine = _engine((0, 1, 0), (0, 1, 2, 0))
    trace = _trace(engine, [0, 1, 0])
    candidates = pd.DataFrame(
        active_prefix_records(trace, decision_event_index=1, decision_bar_ordinal=1)
    )
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="d",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    targeted, tied = target_candidate_rows(candidates, outcome)
    assert tied is False
    positives = targeted.loc[targeted["candidate_completes_first_within_6_bars"].eq(1)]
    assert positives["semantic_loop_id"].tolist() == ["loop_p_0-1-0"]


def test_diversion_and_no_completion_are_negative() -> None:
    engine = _engine((0, 1, 0), (0, 1, 2, 0), (1, 2, 1))
    diversion_trace = _trace(engine, [0, 1, 2, 1])
    candidates = pd.DataFrame(
        active_prefix_records(diversion_trace, decision_event_index=1, decision_bar_ordinal=1)
    )
    diverted = engine.outcome_for_decision(
        diversion_trace,
        decision_id="diverted",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    targeted, _ = target_candidate_rows(candidates, diverted)
    assert targeted["candidate_completes_first_within_6_bars"].sum() == 0

    no_completion_trace = _trace(engine, [0, 1, 2])
    no_completion = engine.outcome_for_decision(
        no_completion_trace,
        decision_id="none",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    targeted, _ = target_candidate_rows(candidates, no_completion)
    assert targeted["candidate_completes_first_within_6_bars"].sum() == 0


def test_no_future_transition_before_horizon_is_no_completion_not_session_end() -> None:
    engine = _engine((0, 1, 0))
    trace = _trace(engine, [0, 1])
    raw = engine.outcome_for_decision(
        trace,
        decision_id="no-transition",
        decision_event_index=1,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    assert raw.primary_label == PrimaryOutcomeLabel.SESSION_END
    normalized = normalize_first_event_outcome(
        raw,
        decision_bar_ordinal=1,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    assert normalized.primary_label == PrimaryOutcomeLabel.NO_REGISTERED_LOOP_WITHIN_HORIZON


def test_tied_registered_completion_is_excluded_without_lexical_break() -> None:
    primitive = decompose_closed_path((0, 1, 0))
    repeat = decompose_closed_path((0, 1, 0, 1, 0))
    dictionary = LoopDictionary.from_definitions([primitive, repeat], version="ties")
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    trace = _trace(engine, [0, 1, 0, 1, 0])
    candidates = pd.DataFrame(
        active_prefix_records(trace, decision_event_index=3, decision_bar_ordinal=3)
    )
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="tie",
        decision_event_index=3,
        decision_bar_ordinal=3,
        horizon_bars=6,
        session_end_bar_ordinal=77,
    )
    assert outcome.primary_label == PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION
    targeted, tied = target_candidate_rows(candidates, outcome)
    assert tied is True
    assert targeted.empty


def test_exact_five_preregistered_interactions() -> None:
    frame = pd.DataFrame(
        {
            "candidate_orientation_sign": [-1.0],
            "signed_pressure": [0.4],
            "prefix_completion_fraction": [0.5],
            "conviction": [0.8],
            "current_transition_probability": [0.25],
            "arousal": [0.6],
            "repeat_depth": [3.0],
            "tension": [0.7],
            "probability_of_next_required_state": [0.2],
            "signed_exhaustion": [-0.5],
        }
    )
    result = compute_interactions(frame)
    assert tuple(result.columns) == INTERACTION_FEATURES
    np.testing.assert_allclose(result.iloc[0], [-0.4, 0.4, 0.15, 2.1, 0.1])


def test_interaction_clipping_is_fit_on_development_only() -> None:
    development = pd.DataFrame({name: np.arange(100, dtype=float) for name in INTERACTION_FEATURES})
    assessment = pd.DataFrame({name: [1_000.0] for name in INTERACTION_FEATURES})
    bounds = fit_interaction_clipping(development)
    clipped = apply_interaction_clipping(assessment, bounds)
    for name in INTERACTION_FEATURES:
        assert clipped.loc[0, name] == pytest.approx(bounds[name][1])


def test_manual_logistic_reconstruction_matches_fitted_model() -> None:
    frame = pd.DataFrame(
        {
            "x": [-2.0, -1.0, 1.0, 2.0, -1.5, 1.5],
            "semantic_loop_id": ["a", "a", "a", "b", "b", "b"],
            "decision_id": [f"d{index}" for index in range(6)],
            "row_weight": [1.0] * 6,
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    model = fit_candidate_logistic(
        frame,
        target_column="target",
        numeric_features=("x",),
        categorical_features=("semantic_loop_id",),
        model_id="M0",
    )
    np.testing.assert_allclose(
        model.predict(frame),
        manual_logistic_probabilities(model.as_dict(), frame),
        rtol=0.0,
        atol=1e-15,
    )


def test_session_block_bootstrap_resamples_whole_sessions_deterministically() -> None:
    draws = session_block_bootstrap_draws(["2025-01-02", "2025-01-03"], draws=100, seed=7)
    assert len(draws) == 100
    assert all(len(draw.sampled_sessions) == 2 for draw in draws)
    assert draws == session_block_bootstrap_draws(["2025-01-03", "2025-01-02"], draws=100, seed=7)
    with pytest.raises(ValueError, match="exactly 100"):
        session_block_bootstrap_draws(["2025-01-02"], draws=99, seed=7)


def test_behavioural_null_permutes_full_bundle_by_stock_within_slate() -> None:
    rows: list[dict[str, object]] = []
    for symbol, base in (("AAL", 0.0), ("AAOI", 100.0), ("APLD", 200.0)):
        for candidate in ("a", "b"):
            row: dict[str, object] = {
                "slate_id": "2025-02-03|06",
                "symbol": symbol,
                "semantic_loop_id": candidate,
            }
            row.update({name: base + index for index, name in enumerate(BEHAVIOURAL_DIMENSIONS)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    permuted = permute_behavioural_bundle_within_slates(frame, seed=13, draw=0)
    for _, stock in permuted.groupby("symbol", sort=True):
        assert stock[list(BEHAVIOURAL_DIMENSIONS)].drop_duplicates().shape[0] == 1
    original_bundles = {
        tuple(row) for row in frame[list(BEHAVIOURAL_DIMENSIONS)].drop_duplicates().to_numpy()
    }
    permuted_bundles = {
        tuple(row) for row in permuted[list(BEHAVIOURAL_DIMENSIONS)].drop_duplicates().to_numpy()
    }
    assert permuted_bundles == original_bundles
    pd.testing.assert_frame_equal(
        frame[["slate_id", "symbol", "semantic_loop_id"]],
        permuted[["slate_id", "symbol", "semantic_loop_id"]],
    )


def test_protected_date_rejection() -> None:
    assert_no_protected_rows(pd.Series([pd.Timestamp("2025-08-22T20:00:00Z")]))
    with pytest.raises(ValueError, match="protected"):
        assert_no_protected_rows(pd.Series([pd.Timestamp("2025-08-23T00:00:00Z")]))


@pytest.mark.parametrize(
    ("m1", "m2", "m2_adverse", "expected"),
    [
        (True, True, False, "structural_behavioural_interactions_improve_loop_completion"),
        (True, False, False, "behavioural_context_improves_loop_completion"),
        (True, False, True, "behavioural_main_effects_only"),
        (False, False, False, "no_behavioural_context_increment"),
    ],
)
def test_decision_logic(m1: bool, m2: bool, m2_adverse: bool, expected: str) -> None:
    assert (
        decide_screen(
            {
                "integrity_blocker": None,
                "m1_passes": m1,
                "m2_passes": m2,
                "m2_materially_adverse": m2_adverse,
            }
        )
        == expected
    )


def test_decision_logic_preserves_valid_blocker() -> None:
    assert (
        decide_screen(
            {
                "integrity_blocker": "blocked_protected_boundary_failure",
                "m1_passes": False,
                "m2_passes": False,
                "m2_materially_adverse": False,
            }
        )
        == "blocked_protected_boundary_failure"
    )
