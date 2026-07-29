from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stocker_research.causal_state_export_v2 import (
    HysteresisConfig,
    SoftLoopPrefixTracker,
    causal_semimarkov_filter_v2,
    hysteretic_states,
)
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.regime_validity_v2 import (
    CleaningVariant,
    EmissionPreprocessing,
    PartADecision,
    PartAGateEvidence,
    PartBBlockedError,
    apply_cleaning_variant,
    audit_feature_availability,
    authorize_part_b,
    build_run_ledger,
    build_training_sample,
    causal_filter_summary,
    cleaning_policy_metadata,
    decide_part_a,
    deterministic_model_registry,
    emission_feature_provenance,
    estimate_semimarkov_parameters,
    fit_clustered_semimarkov,
    fit_emission_preprocessing,
    freeze_part_a_binding,
    frozen_emission_partition,
    gaussian_log_emissions,
    reject_outcome_columns,
    safety_flags,
    semantic_remap_by_activity_direction,
    transform_emissions,
)
from stocker_research.state_alignment_v2 import (
    AlignmentWeights,
    align_states,
    apply_state_mapping,
)
from stocker_research.state_representation_sensitivity_v2 import (
    build_loop_ledgers_by_representation,
    classify_soft_support,
    cleaning_run_changes,
    compare_representation_events,
    hierarchical_state_ids,
    hysteretic_states_by_session,
    reconstruct_first_event_outcomes,
    transition_confidence,
)

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _posterior_fixture() -> tuple[object, dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    model = {
        "duration_hazard": np.asarray([[0.25, 0.5, 0.5], [0.4, 0.5, 0.5]]),
        "transitions": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "initial": np.asarray([0.75, 0.25]),
    }
    emissions = np.log(
        np.asarray(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.9, 0.1],
            ]
        )
    )
    groups = (np.asarray([0, 1]), np.asarray([2, 3]))
    timestamps = tuple(BASE + timedelta(minutes=5 * index) for index in range(4))
    exported = causal_semimarkov_filter_v2(
        emissions,
        session_groups=groups,
        model=model,
        bar_start_timestamps=timestamps,
        bar_duration=timedelta(minutes=5),
    )
    return exported, model, groups


def _dictionary() -> LoopDictionary:
    return LoopDictionary.from_definitions(
        [decompose_closed_path((0, 1, 0))],
        version="regime_validity_v2_test",
    )


def _decisions(hard: list[int], hysteretic: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (hard_state, hysteretic_state) in enumerate(zip(hard, hysteretic, strict=True)):
        start = BASE + timedelta(minutes=5 * index)
        probability = [0.05, 0.05]
        probability[hard_state] = 0.95
        rows.append(
            {
                "decision_id": f"d{index}",
                "run_id": "run-test",
                "git_sha": "a" * 40,
                "contract_hash": "b" * 64,
                "data_snapshot_hash": "c" * 64,
                "dictionary_version": "regime_validity_v2_test",
                "state_model_version": "state-test",
                "symbol": "AAA",
                "session": "2024-01-02",
                "bar_ordinal": index,
                "bar_start_timestamp": start,
                "bar_complete_timestamp": start + timedelta(minutes=5),
                "decision_timestamp": start + timedelta(minutes=5),
                "hard_state_legacy": hard_state,
                "hard_state_hysteretic": hysteretic_state,
                "posterior_state_probabilities": probability,
                "posterior_entropy": 0.2,
                "top_second_margin": 0.9,
                "hard_run_age": 1,
                "expected_state_age": 1.0,
                "transition_probability_next_bar": 0.2,
                "bars_remaining_in_session": len(hard) - index - 1,
                "is_run_entry": index == 0 or hard_state != hard[index - 1],
                "research_only": True,
                "execution_enabled": False,
                "order_placement": "disabled",
                "broker_connected": False,
                "strategy_promotion": False,
            }
        )
    return pd.DataFrame(rows)


def test_state_posterior_normalizes() -> None:
    exported, _, _ = _posterior_fixture()
    assert np.allclose(exported.state_probabilities.sum(axis=1), 1.0)


def test_state_age_posterior_normalizes() -> None:
    exported, _, _ = _posterior_fixture()
    assert np.allclose(exported.state_age_probabilities.sum(axis=(1, 2)), 1.0)


def test_session_reset_clears_prior_age_mass() -> None:
    exported, _, groups = _posterior_fixture()
    for positions in groups:
        first = int(positions[0])
        assert np.isclose(exported.state_age_probabilities[first, :, 1:].sum(), 0.0)


def test_no_future_bar_enters_emission_features() -> None:
    frame = pd.DataFrame(
        {
            "decision_timestamp": [BASE + timedelta(minutes=5), BASE + timedelta(minutes=10)],
            "feature__source_timestamp": [BASE, BASE + timedelta(minutes=5)],
            "feature__available_timestamp": [
                BASE + timedelta(minutes=5),
                BASE + timedelta(minutes=10),
            ],
        }
    )
    audit = audit_feature_availability(frame, feature_names=("feature",))
    assert audit["causal_pass"].all()
    frame.loc[1, "feature__available_timestamp"] = BASE + timedelta(minutes=15)
    assert not audit_feature_availability(frame, feature_names=("feature",)).loc[1, "causal_pass"]


def test_frozen_feature_provenance_declares_no_positive_bar_offsets() -> None:
    provenance = emission_feature_provenance()
    assert set(provenance) == set(frozen_emission_partition().combined)
    assert all(item.latest_bar_offset == 0 for item in provenance.values())
    assert all(not item.future_rows_required for item in provenance.values())
    assert all(item.availability_rule == "completed_bar" for item in provenance.values())


def test_duration_and_age_support_are_correct() -> None:
    exported, model, _ = _posterior_fixture()
    assert exported.state_age_probabilities.shape[2] == model["duration_hazard"].shape[1]
    assert exported.hard_map_run_age.max() <= model["duration_hazard"].shape[1]


def test_hard_map_state_equals_posterior_argmax() -> None:
    exported, _, _ = _posterior_fixture()
    assert np.array_equal(exported.hard_map_state, exported.state_probabilities.argmax(axis=1))


def test_expected_age_is_probability_weighted_one_based_age() -> None:
    exported, _, _ = _posterior_fixture()
    ages = np.arange(1, exported.state_age_probabilities.shape[2] + 1)
    expected = np.sum(exported.state_age_probabilities * ages[None, None, :], axis=(1, 2))
    assert np.allclose(exported.expected_state_age, expected)


def test_departure_probability_is_posterior_weighted_hazard() -> None:
    exported, model, _ = _posterior_fixture()
    expected = np.sum(
        exported.state_age_probabilities * model["duration_hazard"][None, :, :],
        axis=(1, 2),
    )
    assert np.allclose(exported.probability_state_transitions_next_bar, expected)


def test_memory_bounded_filter_summary_matches_full_posterior_export() -> None:
    exported, model, groups = _posterior_fixture()
    emissions = np.log(np.asarray([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]]))
    summary = causal_filter_summary(emissions, groups=groups, model=model)
    assert np.allclose(summary.state_probabilities, exported.state_probabilities)
    assert np.array_equal(summary.hard_states, exported.hard_map_state)
    assert np.allclose(summary.expected_age, exported.expected_state_age)
    assert np.allclose(
        summary.departure_probability,
        exported.probability_state_transitions_next_bar,
    )


def test_gaussian_emission_likelihood_matches_closed_form_example() -> None:
    parameters = estimate_semimarkov_parameters(
        np.asarray([[0.0], [0.0], [3.0], [3.0]]),
        np.asarray([0, 0, 1, 1]),
        groups=(np.arange(4),),
        state_count=2,
        maximum_duration=4,
    )
    likelihood = gaussian_log_emissions(np.asarray([[0.0]]), parameters)
    expected = -0.5 * np.log(2.0 * np.pi * 0.05)
    assert likelihood[0, 0] == pytest.approx(expected)


def test_semimarkov_parameter_estimation_normalizes_and_applies_variance_floor() -> None:
    parameters = estimate_semimarkov_parameters(
        np.asarray([[0.0], [0.0], [2.0], [2.0]]),
        np.asarray([0, 0, 1, 1]),
        groups=(np.arange(4),),
        state_count=2,
        maximum_duration=4,
    )
    assert parameters.variances.tolist() == [[0.05], [0.05]]
    assert np.allclose(parameters.transitions.sum(axis=1), 1.0)
    assert np.isclose(parameters.initial.sum(), 1.0)
    assert np.isclose(parameters.occupancy.sum(), 1.0)
    assert np.all((parameters.duration_hazard >= 0.0) & (parameters.duration_hazard <= 1.0))


def test_terminal_censoring_changes_hazard_without_changing_run_identity() -> None:
    labels = np.asarray([0, 0, 1, 1])
    groups = (np.arange(4),)
    runs = build_run_ledger(labels, groups=groups)
    legacy = estimate_semimarkov_parameters(
        np.asarray([[0.0], [0.0], [2.0], [2.0]]),
        labels,
        groups=groups,
        state_count=2,
        maximum_duration=4,
        censor_terminal_runs=False,
        force_terminal_exit=False,
    )
    censored = estimate_semimarkov_parameters(
        np.asarray([[0.0], [0.0], [2.0], [2.0]]),
        labels,
        groups=groups,
        state_count=2,
        maximum_duration=4,
        censor_terminal_runs=True,
        force_terminal_exit=False,
    )
    assert runs["right_censored"].tolist() == [False, True]
    assert censored.duration_hazard[1, 1] < legacy.duration_hazard[1, 1]


def test_semantic_remap_uses_activity_then_direction_not_numeric_label() -> None:
    labels = np.asarray([7, 7, 3, 3])
    features = pd.DataFrame(
        {
            "activity": [2.0, 2.0, 1.0, 1.0],
            "direction": [-0.5, -0.5, 0.5, 0.5],
        }
    )
    remapped, mapping = semantic_remap_by_activity_direction(
        labels,
        features,
        activity_column="activity",
        direction_column="direction",
    )
    assert mapping == {3: 0, 7: 1}
    assert remapped.tolist() == [1, 1, 0, 0]


def test_emission_preprocessing_fits_training_medians_and_robust_scale_only() -> None:
    raw = pd.DataFrame({"a": [0.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0]})
    preprocessing = fit_emission_preprocessing(raw, feature_names=("a", "b"))
    transformed = transform_emissions(raw, preprocessing)
    assert preprocessing.feature_names == ("a", "b")
    assert preprocessing.medians.tolist() == [2.0, 20.0]
    assert transformed[1].tolist() == pytest.approx([0.0, 0.0])


def test_frozen_preprocessing_transform_preserves_declared_feature_order() -> None:
    preprocessing = EmissionPreprocessing(
        feature_names=("a", "b"),
        medians=np.asarray([1.0, 2.0]),
        centers=np.asarray([1.0, 2.0]),
        scales=np.asarray([2.0, 4.0]),
    )
    transformed = transform_emissions(pd.DataFrame({"b": [6.0], "a": [3.0]}), preprocessing)
    assert transformed.tolist() == [[1.0, 1.0]]


def test_clustered_semimarkov_fit_is_deterministic_for_fixed_seed() -> None:
    raw = pd.DataFrame(
        {
            "activity": [0.0, 0.1, 0.2, 4.0, 4.1, 4.2],
            "direction": [-1.0, -0.9, -0.8, 0.8, 0.9, 1.0],
        }
    )
    scaled = raw.to_numpy(dtype=float)
    arguments = {
        "scaled": scaled,
        "fit_feature_names": ("activity", "direction"),
        "semantic_features": raw,
        "groups": (np.arange(6),),
        "sample_indices": np.arange(6),
        "state_count": 2,
        "seed": 42,
        "cleaning_variant": CleaningVariant.CLEANING_0,
        "activity_column": "activity",
        "direction_column": "direction",
        "maximum_duration": 4,
        "batch_size": 6,
        "n_init": 3,
        "max_iter": 30,
    }
    first = fit_clustered_semimarkov(**arguments)
    second = fit_clustered_semimarkov(**arguments)
    assert np.array_equal(first.semantic_labels, second.semantic_labels)
    assert np.allclose(first.parameters.means, second.parameters.means)
    assert first.training_objective == pytest.approx(second.training_objective)


def test_offline_cleanup_reproduces_neighbor_aware_historical_algorithm() -> None:
    labels = np.asarray([0, 0, 1, 2, 2], dtype=np.int16)
    scaled = np.asarray([[0.0], [0.0], [0.1], [1.0], [1.0]])
    centroids = np.asarray([[0.0], [10.0], [1.0]])
    cleaned = apply_cleaning_variant(
        labels,
        scaled=scaled,
        groups=(np.arange(5),),
        centroids=centroids,
        variant=CleaningVariant.CLEANING_1,
    )
    assert cleaned.tolist() == [0, 0, 0, 2, 2]


def test_no_cleaning_and_causal_cleaning_remain_separate() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int16)
    scaled = labels[:, None].astype(float)
    centroids = np.asarray([[0.0], [1.0]])
    raw = apply_cleaning_variant(
        labels,
        scaled=scaled,
        groups=(np.arange(4),),
        centroids=centroids,
        variant=CleaningVariant.CLEANING_0,
    )
    causal = apply_cleaning_variant(
        labels,
        scaled=scaled,
        groups=(np.arange(4),),
        centroids=centroids,
        variant=CleaningVariant.CLEANING_CAUSAL,
    )
    assert raw.tolist() == [0, 1, 1, 0]
    assert causal.tolist() == [0, 0, 1, 1]


def test_future_neighbor_cleanup_is_not_called_causal() -> None:
    legacy = cleaning_policy_metadata(CleaningVariant.CLEANING_1)
    causal = cleaning_policy_metadata(CleaningVariant.CLEANING_CAUSAL)
    assert legacy.uses_future_neighbor and not legacy.causal
    assert causal.causal and not causal.uses_future_neighbor


def test_state_alignment_is_independent_of_numeric_ids() -> None:
    reference = np.asarray([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    candidate = reference[[2, 0, 1]]
    transition = np.eye(3)
    duration = np.asarray([[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
    alignment = align_states(
        reference,
        candidate,
        reference_transition=transition,
        candidate_transition=transition[[2, 0, 1]][:, [2, 0, 1]],
        reference_duration=duration,
        candidate_duration=duration[[2, 0, 1]],
        weights=AlignmentWeights(centroid=1.0, transition=0.0, duration=0.0),
    )
    assert alignment.candidate_to_reference == {0: 2, 1: 0, 2: 1}


def test_hungarian_assignment_reproduces_synthetic_state_mapping() -> None:
    reference = np.asarray([[0.0], [2.0]])
    candidate = np.asarray([[2.01], [0.01]])
    transition = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    duration = np.asarray([[0.3, 0.7], [0.4, 0.6]])
    alignment = align_states(
        reference,
        candidate,
        reference_transition=transition,
        candidate_transition=transition,
        reference_duration=duration,
        candidate_duration=duration[[1, 0]],
    )
    mapped = apply_state_mapping(np.asarray([0, 1, 0]), alignment.candidate_to_reference)
    assert mapped.tolist() == [1, 0, 1]


def test_k_seed_registry_is_deterministic() -> None:
    first = deterministic_model_registry((6, 8), (11, 12))
    second = deterministic_model_registry((6, 8), (11, 12))
    pd.testing.assert_frame_equal(first, second)
    assert first[["state_count", "seed"]].to_records(index=False).tolist() == [
        (6, 11),
        (6, 12),
        (8, 11),
        (8, 12),
    ]


def test_training_samples_preserve_declared_strata() -> None:
    rows = []
    for symbol in ("A", "B"):
        for month in ("2024-01", "2024-02"):
            for phase in ("opening", "middle"):
                for ordinal in range(3):
                    rows.append(
                        {
                            "symbol": symbol,
                            "month": month,
                            "clock_phase": phase,
                            "bar_ordinal": ordinal,
                        }
                    )
    frame = pd.DataFrame(rows)
    sample = build_training_sample(
        frame,
        variant="SAMPLE_D",
        maximum_rows=16,
        seed=20260718,
    )
    selected = frame.iloc[sample]
    assert len(selected) == 16
    assert selected.groupby(["symbol", "month", "clock_phase"]).ngroups == 8


def test_no_outcome_column_enters_any_state_fit() -> None:
    reject_outcome_columns(["signed_efficiency_12", "regime_log_activity_3"])
    with pytest.raises(ValueError, match="economic or future outcome"):
        reject_outcome_columns(["signed_efficiency_12", "future_return_12"])
    for forbidden in ("forward_5m_return", "next_close", "training_target"):
        with pytest.raises(ValueError, match="economic or future outcome"):
            reject_outcome_columns([forbidden])


def test_cluster_fit_rejects_outcome_bearing_feature_schema() -> None:
    raw = pd.DataFrame({"activity": [0.0, 0.1, 3.0, 3.1], "direction": [-1.0, -0.8, 0.8, 1.0]})
    with pytest.raises(ValueError, match="economic or future outcome"):
        fit_clustered_semimarkov(
            scaled=raw.to_numpy(dtype=float),
            fit_feature_names=("activity", "future_return_5m"),
            semantic_features=raw,
            groups=(np.arange(4),),
            sample_indices=np.arange(4),
            state_count=2,
            seed=42,
            cleaning_variant=CleaningVariant.CLEANING_0,
            activity_column="activity",
            direction_column="direction",
            maximum_duration=4,
            batch_size=4,
            n_init=2,
            max_iter=20,
        )


def test_low_margin_transitions_are_identified() -> None:
    probabilities = np.asarray([[0.60, 0.40], [0.49, 0.51], [0.10, 0.90]], dtype=float)
    hard = np.asarray([0, 1, 1])
    transitions = transition_confidence(
        probabilities,
        hard_states=hard,
        hysteretic_states=np.asarray([0, 0, 1]),
        session_groups=(np.arange(3),),
    )
    first = transitions.iloc[0]
    assert bool(first["margin_lt_0_05"])
    assert bool(first["new_state_probability_lt_0_50"]) is False


def test_one_bar_and_two_bar_reversals_are_session_aware() -> None:
    probabilities = np.eye(2)[np.asarray([0, 1, 0, 1, 1])]
    transitions = transition_confidence(
        probabilities,
        hard_states=np.asarray([0, 1, 0, 1, 1]),
        hysteretic_states=np.asarray([0, 1, 0, 1, 1]),
        session_groups=(np.asarray([0, 1, 2]), np.asarray([3, 4])),
    )
    assert transitions["one_bar_reversal"].tolist() == [True, False]
    assert transitions["two_bar_reversal"].tolist() == [True, False]


def test_transition_duration_is_departing_run_age() -> None:
    probabilities = np.eye(2)[np.asarray([0, 0, 0, 1])]
    transitions = transition_confidence(
        probabilities,
        hard_states=np.asarray([0, 0, 0, 1]),
        hysteretic_states=np.asarray([0, 0, 0, 1]),
        session_groups=(np.arange(4),),
    )
    assert transitions["departing_state_duration"].tolist() == [3]
    assert transitions["new_state_age"].tolist() == [1]


def test_hysteretic_state_resets_at_each_session_boundary() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    result = hysteretic_states_by_session(
        probabilities,
        session_groups=(np.asarray([0, 1]), np.asarray([2, 3])),
        config=HysteresisConfig(switch_probability=0.95, switch_margin=0.90),
    )
    assert result.tolist() == [0, 0, 1, 1]


def test_cleaning_run_changes_identifies_source_runs_not_remaining_short_runs() -> None:
    raw = np.asarray([0, 0, 1, 2, 2, 1, 1])
    cleaned = np.asarray([0, 0, 0, 2, 2, 1, 1])
    audit = cleaning_run_changes(raw, cleaned, session_groups=(np.arange(7),))
    assert audit["changed"].tolist() == [False, True, False, False]
    assert int(audit["changed"].sum()) == 1
    unchanged = cleaning_run_changes(raw, raw, session_groups=(np.arange(7),))
    assert not unchanged["changed"].any()


def test_hysteretic_state_uses_only_past_and_current_posterior() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.48, 0.52], [0.2, 0.8], [0.9, 0.1]], dtype=float)
    config = HysteresisConfig(switch_probability=0.55, switch_margin=0.10)
    original = hysteretic_states(probabilities, config=config)
    changed_future = probabilities.copy()
    changed_future[3] = [0.01, 0.99]
    replay = hysteretic_states(changed_future, config=config)
    assert np.array_equal(original[:3], replay[:3])


def test_soft_posterior_does_not_create_a_hard_event() -> None:
    tracker = SoftLoopPrefixTracker(_dictionary(), state_count=2)
    tracker.update(np.asarray([0.9, 0.1]))
    tracker.update(np.asarray([0.1, 0.9]))
    snapshot = tracker.update(np.asarray([0.9, 0.1]))
    assert snapshot.highest_completion_probability > 0.0
    assert not snapshot.hard_completion
    assert classify_soft_support(False, 0.99) == "NO_HARD_EVENT"


def test_market_and_stock_feature_partitions_are_disjoint() -> None:
    partition = frozen_emission_partition()
    assert partition.stock.isdisjoint(partition.market)
    assert partition.stock.isdisjoint(partition.relative)
    assert partition.market.isdisjoint(partition.relative)
    assert partition.combined == partition.stock | partition.market | partition.relative


def test_hierarchical_state_identities_are_deterministic() -> None:
    market = np.asarray([0, 1, 1, 2])
    stock = np.asarray([3, 0, 3, 1])
    first = hierarchical_state_ids(market, stock, stock_state_count=4)
    second = hierarchical_state_ids(market, stock, stock_state_count=4)
    assert first.numeric.tolist() == [3, 4, 7, 9]
    assert first.tokens == second.tokens


def test_loop_events_are_reconstructed_under_each_hard_representation() -> None:
    ledgers = build_loop_ledgers_by_representation(
        _decisions([0, 1, 0], [0, 0, 1]),
        dictionary=_dictionary(),
        horizon_bars=2,
        allowed_states=frozenset({0, 1}),
    )
    assert set(ledgers) == {"legacy_hard_map", "causal_hysteretic"}
    assert len(ledgers["legacy_hard_map"].outcomes) == 3
    assert len(ledgers["causal_hysteretic"].outcomes) == 3


def test_fast_first_event_outcomes_match_full_ledger() -> None:
    decisions = _decisions([0, 1, 0, 1], [0, 0, 1, 0])
    expected = build_loop_ledgers_by_representation(
        decisions,
        dictionary=_dictionary(),
        horizon_bars=2,
        allowed_states=frozenset({0, 1}),
    )["legacy_hard_map"].outcomes
    actual = reconstruct_first_event_outcomes(
        decisions,
        decisions["hard_state_legacy"].to_numpy(dtype=int),
        dictionary=_dictionary(),
        horizon_bars=2,
        allowed_states=frozenset({0, 1}),
    )
    pd.testing.assert_frame_equal(
        actual,
        expected[["decision_id", "primary_label", "bars_until_completion"]],
        check_dtype=False,
    )


def test_event_agreement_metrics_distinguish_exact_shift_and_mismatch() -> None:
    reference = pd.DataFrame(
        {
            "decision_id": ["a", "b", "c"],
            "primary_label": ["loop_p_0-1-0", "loop_p_0-1-0", "loop_p_0-1-0"],
            "bars_until_completion": [1, 2, 2],
        }
    )
    candidate = pd.DataFrame(
        {
            "decision_id": ["a", "b", "c"],
            "primary_label": ["loop_p_0-1-0", "loop_p_0-1-0", "NO_LOOP_WITHIN_HORIZON"],
            "bars_until_completion": [1, 3, np.nan],
        }
    )
    comparison, metrics = compare_representation_events(reference, candidate, allowed_shift_bars=1)
    assert comparison.columns[:5].tolist() == [
        "decision_id",
        "primary_label_reference",
        "bars_until_completion_reference",
        "primary_label_candidate",
        "bars_until_completion_candidate",
    ]
    assert comparison["agreement_class"].tolist() == [
        "EXACT_EVENT_AGREEMENT",
        "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
        "PRIMITIVE_MISMATCH",
    ]
    assert metrics.exact_fraction == pytest.approx(1 / 3)
    assert metrics.same_primitive_bounded_shift_fraction == pytest.approx(2 / 3)


def test_part_a_gate_binding_is_hash_frozen_before_part_b_access() -> None:
    evidence = PartAGateEvidence.passing()
    decision = decide_part_a(evidence)
    binding = freeze_part_a_binding(
        decision,
        state_model_hash="a" * 64,
        state_count=8,
        state_representation="legacy_hard_map_with_required_hysteretic_sensitivity",
        hysteresis_policy={"switch_probability": 0.55, "switch_margin": 0.10},
        posterior_support_fields=("posterior_entropy", "top_second_margin"),
        state_alignment_hash="b" * 64,
    )
    assert decision is PartADecision.VALIDATED
    assert len(binding.binding_hash) == 64
    assert authorize_part_b(binding, independently_audited=True) is binding


def test_primary_part_a_decision_precedes_independent_audit() -> None:
    evidence = PartAGateEvidence.passing().with_updates(
        independent_audit_reproducible=False,
        posterior_duration_pass=False,
    )
    assert decide_part_a(evidence) is PartADecision.REQUIRES_TARGETED_REPAIR


def test_required_sensitivity_cannot_bypass_loop_language_gates() -> None:
    evidence = PartAGateEvidence.passing().with_updates(
        training_sample_dictionary_coverage_ratio=0.20,
        representation_sensitive=True,
        usable_with_sensitivity=True,
    )
    assert decide_part_a(evidence) is PartADecision.UNSTABLE


def test_failed_part_a_prevents_part_b_scoring() -> None:
    evidence = PartAGateEvidence.passing().with_updates(critical_future_leakage=True)
    decision = decide_part_a(evidence)
    binding = freeze_part_a_binding(
        decision,
        state_model_hash="a" * 64,
        state_count=8,
        state_representation="blocked",
        hysteresis_policy={"switch_probability": 0.55, "switch_margin": 0.10},
        posterior_support_fields=(),
        state_alignment_hash="b" * 64,
    )
    assert decision is PartADecision.REQUIRES_TARGETED_REPAIR
    with pytest.raises(PartBBlockedError):
        authorize_part_b(binding, independently_audited=True)


def test_safety_flags_include_every_required_boundary() -> None:
    assert safety_flags() == {
        "research_only": True,
        "execution_enabled": False,
        "order_placement": "disabled",
        "broker_connected": False,
        "economic_outcomes_used": False,
        "payoff_selection_used": False,
        "production_runtime_modified": False,
        "strategy_promotion": False,
    }
