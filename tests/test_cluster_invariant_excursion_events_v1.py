from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stocker_research.continuous_trajectory_v1 import (
    SAFETY_FLAGS,
    diagonal_distance,
    fit_robust_geometry,
    fit_shrinkage_metric,
    jensen_shannon_distance,
    trajectory_features,
)
from stocker_research.excursion_alignment_v1 import (
    align_event_ledgers,
    event_alignment_summary,
)
from stocker_research.excursion_events_v1 import (
    DistanceCalibration,
    EventFamily,
    ExcursionConfig,
    PartAGateMetrics,
    decide_part_a,
    detect_excursions,
    event_definition_hash,
)
from stocker_research.excursion_nulls_v1 import (
    benjamini_hochberg,
    circular_increment_control,
    phase_conditioned_increment_block_null,
)
from stocker_research.excursion_origin_v1 import (
    locally_stable_origins,
    trailing_robust_origins,
)


def _frame(row_count: int = 12, *, segment_break: int | None = None) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    segment = ["TEST::2024-01-02::segment_00"] * row_count
    if segment_break is not None:
        segment[segment_break:] = ["TEST::2024-01-02::segment_01"] * (row_count - segment_break)
    return pd.DataFrame(
        {
            "decision_id": [f"decision_{index:03d}" for index in range(row_count)],
            "symbol": "TEST",
            "session": "2024-01-02",
            "segment_id": segment,
            "bar_ordinal": np.arange(row_count),
            "bar_start_timestamp": [
                start + pd.Timedelta(minutes=5 * index) for index in range(row_count)
            ],
            "bar_complete_timestamp": [
                start + pd.Timedelta(minutes=5 * (index + 1)) for index in range(row_count)
            ],
            "decision_timestamp": [
                start + pd.Timedelta(minutes=5 * (index + 1)) for index in range(row_count)
            ],
            "segment_end_reason": ["continued"] * (row_count - 1) + ["scheduled_session_end"],
            "session_source_complete": True,
        }
    )


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(
        group.index.to_numpy(dtype=int) for _, group in frame.groupby("segment_id", sort=False)
    )


def _config(**updates: object) -> ExcursionConfig:
    config = ExcursionConfig(
        candidate_id="TEST_E",
        representation="E",
        distance_metric="DIAGONAL",
        departure_threshold=1.0,
        confirmation_bars=2,
        velocity_condition=False,
        minimum_departure_velocity=0.0,
        return_ratio=0.5,
        rotation_persistence=3,
        rotation_separation_ratio=1.0,
        rotation_maximum_velocity=0.08,
        continuation_ratio=1.5,
        partial_retracement_fraction=0.35,
        horizon_bars=6,
        lockout_bars=1,
    )
    return replace(config, **updates)


def _calibration(dimension: int = 2) -> DistanceCalibration:
    return DistanceCalibration(
        emission_scale=np.ones(dimension, dtype=float),
        emission_q90=1.0,
        posterior_q90=0.25,
        mahalanobis_precision=np.eye(dimension, dtype=float),
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("research_only", True),
        ("execution_enabled", False),
        ("order_placement", "disabled"),
        ("broker_connected", False),
        ("economic_outcomes_used", False),
        ("payoff_selection_used", False),
        ("production_runtime_modified", False),
        ("strategy_promotion", False),
    ],
)
def test_safety_flags_are_complete(key: str, expected: object) -> None:
    assert SAFETY_FLAGS[key] == expected


def test_robust_geometry_fits_only_declared_development_rows() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 4.0], [100.0, 100.0]])
    geometry = fit_robust_geometry(values, fit_mask=np.asarray([True, True, False]))
    np.testing.assert_allclose(geometry.centers, [1.0, 2.0])
    assert geometry.fit_row_count == 2


def test_robust_geometry_preserves_missingness_when_median_scales_to_zero() -> None:
    values = np.asarray([[0.0, np.nan], [2.0, 4.0], [4.0, 8.0]])
    geometry = fit_robust_geometry(values, fit_mask=np.asarray([True, True, True]))
    transformed, missing = geometry.transform(values)
    assert missing[0, 1]
    assert np.isfinite(transformed).all()
    assert transformed[0, 1] == 0.0


def test_trajectory_features_are_prefix_invariant_and_gap_local() -> None:
    frame = _frame(8, segment_break=4)
    values = np.c_[np.arange(8, dtype=float), np.zeros(8)]
    groups = _groups(frame)
    whole = trajectory_features(values, groups=groups, window=3)
    prefix = trajectory_features(values[:4], groups=(np.arange(4),), window=3)
    np.testing.assert_allclose(whole.velocity[:4], prefix.velocity)
    assert np.isnan(whole.first_difference[4]).all()
    assert whole.local_path_length[4] == 0.0


def test_diagonal_distance_is_nonnegative() -> None:
    assert diagonal_distance(np.asarray([1.0, 2.0]), np.zeros(2), np.ones(2)) >= 0.0


def test_shrinkage_metric_is_positive_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(100, 3))
    first = fit_shrinkage_metric(values)
    second = fit_shrinkage_metric(values)
    np.testing.assert_array_equal(first.precision, second.precision)
    assert np.linalg.eigvalsh(first.precision).min() > 0.0


def test_jensen_shannon_distance_is_symmetric_bounded_and_zero_on_identity() -> None:
    left = np.asarray([0.8, 0.2])
    right = np.asarray([0.1, 0.9])
    assert jensen_shannon_distance(left, left) == pytest.approx(0.0)
    assert jensen_shannon_distance(left, right) == pytest.approx(
        jensen_shannon_distance(right, left)
    )
    assert 0.0 <= jensen_shannon_distance(left, right) <= np.sqrt(np.log(2.0))


def test_jensen_shannon_distance_handles_subnormal_mass() -> None:
    left = np.asarray([1.0, np.nextafter(0.0, 1.0)])
    right = np.asarray([1.0, 0.0])
    distance = jensen_shannon_distance(left, right)
    assert np.isfinite(distance)
    assert distance >= 0.0


def test_trailing_origin_uses_strictly_previous_completed_rows() -> None:
    values = np.asarray([[0.0], [2.0], [100.0], [4.0]])
    origins = trailing_robust_origins(values, groups=(np.arange(4),), window=2)
    assert origins.eligible.tolist() == [False, False, True, True]
    assert origins.centers[2, 0] == pytest.approx(1.0)
    assert origins.centers[3, 0] == pytest.approx(51.0)


def test_origin_never_crosses_a_segment_boundary() -> None:
    values = np.arange(8, dtype=float)[:, None]
    origins = trailing_robust_origins(
        values,
        groups=(np.arange(4), np.arange(4, 8)),
        window=2,
    )
    assert not origins.eligible[4]
    assert not origins.eligible[5]
    assert origins.centers[6, 0] == pytest.approx(4.5)


def test_locally_stable_origin_requires_frozen_stability_condition() -> None:
    values = np.asarray([[0.0], [0.01], [0.02], [3.0], [6.0], [9.0]])
    origins = locally_stable_origins(
        values,
        groups=(np.arange(6),),
        window=3,
        maximum_path_length=0.1,
        maximum_velocity=0.05,
    )
    assert origins.eligible[3]
    assert not origins.eligible[5]


def test_two_bar_confirmation_and_origin_freeze_return_to_origin() -> None:
    frame = _frame(10)
    values = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.2, 0.0],
            [1.4, 0.0],
            [0.4, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(),
        config=_config(),
    )
    event = result.events.iloc[0]
    assert event["event_family"] == EventFamily.RETURN_TO_ORIGIN.value
    assert int(event["first_detectable_bar_ordinal"]) == 4
    assert int(event["confirmation_bar_ordinal"]) == 5
    assert int(event["resolution_bar_ordinal"]) == 6
    assert event["frozen_origin_vector"] == "[0.0,0.0]"


def test_one_bar_and_two_bar_confirmation_differ() -> None:
    frame = _frame(8)
    values = np.asarray([[0.0], [0.0], [0.0], [1.2], [0.2], [0.0], [0.0], [0.0]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    one = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(confirmation_bars=1),
    )
    two = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(confirmation_bars=2),
    )
    assert len(one.events) == 1
    assert two.events.empty


def test_rotation_requires_stable_new_neighbourhood() -> None:
    frame = _frame(12)
    moving = np.asarray(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.1],
            [1.2],
            [1.3],
            [1.4],
            [1.5],
            [1.6],
            [1.7],
            [1.8],
            [1.9],
        ]
    )
    stable = moving.copy()
    stable[5:9, 0] = 1.25
    moving_origins = trailing_robust_origins(moving, groups=_groups(frame), window=3)
    stable_origins = trailing_robust_origins(stable, groups=_groups(frame), window=3)
    moving_result = detect_excursions(
        frame,
        emission_vectors=moving,
        emission_origins=moving_origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(continuation_ratio=5.0),
    )
    stable_result = detect_excursions(
        frame,
        emission_vectors=stable,
        emission_origins=stable_origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(continuation_ratio=5.0),
    )
    assert EventFamily.ROTATE_TO_NEW_REGION.value not in set(moving_result.events["event_family"])
    assert stable_result.events.iloc[0]["event_family"] == EventFamily.ROTATE_TO_NEW_REGION.value


def test_continuation_uses_geometry_only() -> None:
    frame = _frame(10)
    values = np.asarray([[0.0], [0.0], [0.0], [1.1], [1.2], [1.7], [1.8], [1.9], [2.0], [2.1]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(),
    )
    assert result.events.iloc[0]["event_family"] == EventFamily.CONTINUE_AWAY.value


def test_partial_return_is_not_assigned_before_horizon() -> None:
    frame = _frame(12)
    values = np.asarray(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.1],
            [1.2],
            [1.1],
            [0.9],
            [0.8],
            [0.75],
            [0.75],
            [0.75],
            [0.75],
        ]
    )
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(horizon_bars=4, continuation_ratio=5.0),
    )
    event = result.events.iloc[0]
    assert event["event_family"] == EventFamily.PARTIAL_RETURN.value
    assert int(event["bars_from_confirmation_to_resolution"]) == 4


def test_event_precedence_and_coincident_metadata_are_deterministic() -> None:
    frame = _frame(9)
    values = np.asarray([[0.0], [0.0], [0.0], [1.1], [1.2], [0.4], [0.3], [0.2], [0.1]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(return_ratio=0.5, rotation_separation_ratio=0.1, rotation_persistence=1),
    )
    assert result.events.iloc[0]["event_family"] == EventFamily.RETURN_TO_ORIGIN.value
    assert "RETURN_TO_ORIGIN" in result.coincident_conditions.iloc[-1]["conditions_json"]


def test_session_end_is_distinct_from_unresolved_horizon() -> None:
    frame = _frame(7)
    values = np.asarray([[0.0], [0.0], [0.0], [1.1], [1.2], [1.1], [1.1]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(horizon_bars=20, continuation_ratio=5.0, partial_retracement_fraction=0.9),
    )
    assert result.events.iloc[0]["event_family"] == EventFamily.SESSION_END.value


def test_structural_gap_fails_closed() -> None:
    frame = _frame(9, segment_break=6)
    frame.loc[5, "segment_end_reason"] = "source_gap"
    values = np.asarray([[0.0], [0.0], [0.0], [1.1], [1.2], [1.2], [0.0], [0.0], [0.0]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    result = detect_excursions(
        frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(horizon_bars=20, continuation_ratio=5.0),
    )
    assert result.events.iloc[0]["event_family"] == EventFamily.UNAVAILABLE_STRUCTURAL_GAP.value


def test_event_ids_and_deduplication_are_deterministic() -> None:
    frame = _frame(10)
    values = np.asarray([[0.0], [0.0], [0.0], [1.1], [1.2], [0.4], [0.0], [0.0], [0.0], [0.0]])
    origins = trailing_robust_origins(values, groups=_groups(frame), window=3)
    kwargs = dict(
        frame=frame,
        emission_vectors=values,
        emission_origins=origins,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=_calibration(1),
        config=_config(),
    )
    first = detect_excursions(**kwargs)
    second = detect_excursions(**kwargs)
    assert first.events["event_id"].tolist() == second.events["event_id"].tolist()
    assert first.events["event_id"].is_unique
    assert first.decision_mapping["event_id"].nunique() == len(first.events)


def _events(families: list[str], onsets: list[int], resolutions: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"event_{index}" for index in range(len(families))],
            "symbol": "TEST",
            "session": "2024-01-02",
            "segment_id": "segment",
            "event_family": families,
            "onset_bar_ordinal": onsets,
            "resolution_bar_ordinal": resolutions,
        }
    )


def test_same_family_within_two_bars_matches_without_numeric_state_ids() -> None:
    reference = _events(["RETURN_TO_ORIGIN"], [5], [9])
    candidate = _events(["RETURN_TO_ORIGIN"], [7], [11])
    alignment = align_event_ledgers(reference, candidate, tolerance_bars=2)
    assert alignment.iloc[0]["alignment_class"] == "SAME_FAMILY_BOUNDED_SHIFT"
    assert "state" not in alignment.columns


def test_different_family_does_not_match() -> None:
    reference = _events(["RETURN_TO_ORIGIN"], [5], [9])
    candidate = _events(["CONTINUE_AWAY"], [5], [9])
    alignment = align_event_ledgers(reference, candidate, tolerance_bars=2)
    assert alignment.iloc[0]["alignment_class"] == "DIFFERENT_FAMILY"


def test_split_and_merged_events_are_identified() -> None:
    reference = _events(["RETURN_TO_ORIGIN"], [5], [12])
    candidate = _events(["RETURN_TO_ORIGIN", "RETURN_TO_ORIGIN"], [5, 9], [8, 12])
    alignment = align_event_ledgers(reference, candidate, tolerance_bars=2)
    assert "SPLIT_EVENT" in set(alignment["alignment_class"])
    reverse = align_event_ledgers(candidate, reference, tolerance_bars=2)
    assert "MERGED_EVENT" in set(reverse["alignment_class"])


def test_alignment_summary_counts_family_and_time() -> None:
    alignment = align_event_ledgers(
        _events(["RETURN_TO_ORIGIN"], [5], [9]),
        _events(["RETURN_TO_ORIGIN"], [5], [9]),
        tolerance_bars=2,
    )
    summary = event_alignment_summary(alignment)
    assert summary["same_family_fraction"] == 1.0
    assert summary["median_timing_disagreement_bars"] == 0.0


def test_increment_block_null_preserves_length_phase_and_is_deterministic() -> None:
    increments = np.arange(18, dtype=float).reshape(9, 2)
    phases = np.asarray(["OPENING"] * 3 + ["MIDDLE"] * 3 + ["LATE"] * 3)
    first = phase_conditioned_increment_block_null(
        increments,
        phases=phases,
        block_length=2,
        seed=11,
    )
    second = phase_conditioned_increment_block_null(
        increments,
        phases=phases,
        block_length=2,
        seed=11,
    )
    np.testing.assert_array_equal(first.increments, second.increments)
    np.testing.assert_array_equal(first.source_phases, phases)
    assert len(first.increments) == len(increments)


def test_circular_increment_control_is_deterministic_and_length_preserving() -> None:
    increments = np.arange(12, dtype=float).reshape(6, 2)
    first = circular_increment_control(increments, offset=2, block_length=2)
    second = circular_increment_control(increments, offset=2, block_length=2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == increments.shape


def test_bh_correction_is_deterministic_and_monotone_in_rank() -> None:
    values = np.asarray([0.04, 0.001, 0.02, 0.5])
    first = benjamini_hochberg(values)
    second = benjamini_hochberg(values)
    np.testing.assert_array_equal(first, second)
    order = np.argsort(values)
    assert np.all(np.diff(first[order]) >= -1e-15)


def test_event_definition_hash_changes_with_threshold_and_not_object_identity() -> None:
    first = _config()
    assert event_definition_hash(first) == event_definition_hash(_config())
    assert event_definition_hash(first) != event_definition_hash(_config(departure_threshold=1.1))


def _passing_metrics(**updates: object) -> PartAGateMetrics:
    metrics = PartAGateMetrics(
        representation="E",
        cross_lineage_agreement=0.8,
        cross_seed_agreement=0.7,
        cross_sample_agreement=0.65,
        cross_k_agreement=0.6,
        maximum_validation_share_shift_pp=5.0,
        unique_development_events=2500,
        unique_validation_events=1500,
        stock_count=20,
        month_count=12,
        maximum_stock_share=0.1,
        maximum_month_share=0.15,
        median_timing_disagreement_bars=1.0,
        posterior_hybrid_validated=False,
        secondary_gate_narrow_failure=False,
        source_blocked=False,
        exact_rerun_pass=True,
        independent_audit_pass=True,
    )
    return replace(metrics, **updates)


def test_part_a_decision_hierarchy_is_deterministic() -> None:
    assert decide_part_a(_passing_metrics()) == "emission_space_excursion_events_validated"
    assert decide_part_a(_passing_metrics(posterior_hybrid_validated=True)) == (
        "cluster_invariant_excursion_events_validated"
    )
    assert decide_part_a(_passing_metrics(source_blocked=True)) == (
        "cluster_invariant_event_experiment_blocked"
    )
    assert decide_part_a(_passing_metrics(unique_validation_events=999)) == (
        "cluster_invariant_event_population_too_sparse"
    )


def test_part_a_failure_prevents_part_b_authorization() -> None:
    failed = _passing_metrics(cross_lineage_agreement=0.4)
    assert decide_part_a(failed) == "cluster_invariant_excursion_events_not_stable"
