from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from stocker_prospective.m1c_opening_reversal_analysis_v1 import (
    EVENT_BOOTSTRAP_SEED_V1,
    PRIMARY_NULL_REPLICATIONS_V1,
    PRIMARY_NULL_SEED_V1,
    SESSION_BOOTSTRAP_SEED_V1,
    OpeningReversalAnalysisEpisodeV1,
    OpeningReversalOptionEpisodeV1,
    analyze_direction_cohort_v1,
    analyze_option_economics_v1,
    build_opening_reversal_direction_decision_receipt_v1,
    evaluate_support_v1,
)


def _episode(
    *,
    ordinal: int,
    phase: str = "prospective_development",
    aligned_return: float = 0.02,
    experiment_version: str = "1",
) -> OpeningReversalAnalysisEpisodeV1:
    event = ordinal // 3
    transition_sign = 1 if event % 2 == 0 else -1
    prediction_sign = -transition_sign
    signed_return = prediction_sign * aligned_return
    threshold = 0.01
    outcome_direction = 1 if signed_return > threshold else -1 if signed_return < -threshold else 0
    return OpeningReversalAnalysisEpisodeV1(
        experiment_version=experiment_version,
        prediction_receipt_hash_v1=f"{ordinal + 1:064x}",
        outcome_receipt_hash_v1=f"{ordinal + 10_001:064x}",
        cohort_phase=phase,
        session=date(2026, 10, 1) + timedelta(days=event),
        stock=f"S{(event * 3 + ordinal % 3) % 15:02d}",
        opening_transition_event_id_v1=f"event-{event:03d}",
        opening_transition_sign_v1=transition_sign,
        prediction_sign_v1=prediction_sign,
        prediction_v1="CALL" if prediction_sign == 1 else "PUT",
        r_15m=signed_return,
        threshold_15m=threshold,
        outcome_direction_v1=outcome_direction,
        material_direction_correct_v1=outcome_direction == prediction_sign,
        opening_reversal_aligned_return_v1=aligned_return,
        promoted=ordinal % 3 == 0,
        primary_option_evidence_complete=False,
        baseline_prediction_signs={
            "follow_vti_severe_opening_sign": transition_sign,
            "oppose_vti_severe_opening_sign": prediction_sign,
            "always_call": 1,
            "always_put": -1,
        },
        baseline_unavailable_reasons={
            "most_recent_completed_five_minute_stock_momentum": (
                "test_fixture_causal_baseline_unavailable"
            ),
            "complete_stock_opening_window_momentum": ("test_fixture_causal_baseline_unavailable"),
            "existing_clean_market_direction_baseline": (
                "test_fixture_causal_baseline_unavailable"
            ),
            "frozen_a1": "test_fixture_causal_baseline_unavailable",
            "frozen_historical_asymmetric_downside_score": (
                "test_fixture_causal_baseline_unavailable"
            ),
            "independently_frozen_microstructure_rule": (
                "test_fixture_causal_baseline_unavailable"
            ),
        },
    )


def test_analysis_cannot_mix_v1_and_v1_1_episodes() -> None:
    with pytest.raises(ValueError, match="mix experiment versions"):
        analyze_direction_cohort_v1(
            (
                _episode(ordinal=0, experiment_version="1"),
                _episode(ordinal=1, experiment_version="1.1"),
            ),
            phase="prospective_development",
        )


def _supported(
    *,
    phase: str = "prospective_development",
) -> tuple[OpeningReversalAnalysisEpisodeV1, ...]:
    return tuple(_episode(ordinal=index, phase=phase) for index in range(150))


def test_support_gate_uses_events_not_stock_episode_labels() -> None:
    support = evaluate_support_v1(_supported())

    assert support.passes
    assert support.complete_eligible_stock_episodes == 150
    assert support.unique_severe_opening_events == 50
    assert support.positive_transition_events == 25
    assert support.negative_transition_events == 25
    assert support.represented_stocks == 15
    assert support.sessions == 50
    assert support.maximum_stock_episode_fraction == pytest.approx(1 / 15)
    assert support.maximum_event_episode_fraction == pytest.approx(0.02)


def test_development_analysis_runs_frozen_cluster_null_and_placebo_contract() -> None:
    result = analyze_direction_cohort_v1(
        _supported(),
        phase="prospective_development",
    )

    assert result.decision == ("prospective_opening_reversal_development_supported")
    assert result.summary.mean_aligned_return == pytest.approx(0.02)
    assert result.summary.material_direction_accuracy == 1.0
    assert (
        result.summary.material_up_count
        + result.summary.material_down_count
        + result.summary.no_material_move_count
        == result.summary.episode_count
    )
    assert result.summary.difference_versus_follow_vti == pytest.approx(0.04)
    assert result.summary.positive_transition_mean_reversal_return == pytest.approx(0.02)
    assert result.summary.negative_transition_mean_reversal_return == pytest.approx(0.02)
    assert len(result.session_bootstrap_draws) == 2_000
    assert len(result.event_bootstrap_draws) == 2_000
    assert len(result.primary_null_draws) == PRIMARY_NULL_REPLICATIONS_V1
    assert {draw.seed for draw in result.primary_null_draws} == {PRIMARY_NULL_SEED_V1}
    assert result.summary.primary_null_p_value is not None
    assert result.summary.primary_null_p_value < 0.05
    assert result.summary.temporal_placebo_mean is not None
    assert result.summary.temporal_placebo_mean < 0.0
    baseline_means = dict(result.baseline_means)
    assert baseline_means["oppose_vti_severe_opening_sign"] == pytest.approx(0.02)
    assert baseline_means["follow_vti_severe_opening_sign"] == pytest.approx(-0.02)
    populations = {
        (row.population, row.baseline): row for row in result.baseline_population_results
    }
    promoted = populations[("promoted_episodes", "oppose_vti_severe_opening_sign")]
    assert promoted.population_episode_count == 50
    assert promoted.available_episode_count == 50
    assert promoted.reversal_minus_baseline_mean == pytest.approx(0.0)
    unavailable = populations[
        (
            "complete_primary_option_evidence",
            "frozen_historical_asymmetric_downside_score",
        )
    ]
    assert unavailable.population_episode_count == 0
    assert unavailable.available_episode_count == 0
    assert SESSION_BOOTSTRAP_SEED_V1 == 2026072902
    assert EVENT_BOOTSTRAP_SEED_V1 == 2026072903


def test_confirmation_requires_both_cluster_lower_bounds_and_null() -> None:
    result = analyze_direction_cohort_v1(
        _supported(phase="untouched_confirmation"),
        phase="untouched_confirmation",
    )

    assert result.decision == ("prospective_opening_reversal_direction_supported")
    assert result.summary.session_cluster_interval.lower_95 == pytest.approx(0.02)
    assert result.summary.event_cluster_interval.lower_95 == pytest.approx(0.02)
    assert result.summary.winsorised_one_percent_mean == pytest.approx(0.02)
    assert result.summary.leave_one_stock_out_minimum_mean == pytest.approx(0.02)
    assert result.summary.leave_one_session_out_minimum_mean == pytest.approx(0.02)
    assert result.summary.leave_one_event_out_minimum_mean == pytest.approx(0.02)


def test_insufficient_support_blocks_without_relaxing_gates() -> None:
    result = analyze_direction_cohort_v1(
        (_episode(ordinal=0),),
        phase="prospective_development",
    )

    assert result.decision == ("blocked_insufficient_prospective_development_support")
    assert "complete_eligible_stock_episodes_below_150" in result.decision_reasons
    assert "unique_severe_opening_events_below_40" in result.decision_reasons


def test_direction_decision_receipt_recomputes_full_frozen_analysis() -> None:
    rows = _supported()
    result = analyze_direction_cohort_v1(
        rows,
        phase="prospective_development",
    )
    tampered = result.model_copy(
        update={"decision": "prospective_opening_reversal_development_not_supported"}
    )

    with pytest.raises(ValueError, match="statistics differ"):
        build_opening_reversal_direction_decision_receipt_v1(
            result=tampered,
            episodes=rows,
            boundary_timestamp_utc=datetime(2027, 1, 1, tzinfo=UTC),
        )


def test_analysis_rejects_cross_cohort_and_non_reversal_rows() -> None:
    with pytest.raises(ValueError, match="cannot cross"):
        analyze_direction_cohort_v1(
            (_episode(ordinal=0, phase="untouched_confirmation"),),
            phase="prospective_development",
        )
    invalid = _episode(ordinal=0).model_dump(mode="python")
    invalid["prediction_sign_v1"] = 1
    with pytest.raises(ValidationError, match="differs from reversal rule"):
        OpeningReversalAnalysisEpisodeV1.model_validate(invalid)


def test_analysis_blocks_one_event_id_with_multiple_sessions_or_signs() -> None:
    first = _episode(ordinal=0)
    contradictory = _episode(ordinal=3).model_copy(
        update={"opening_transition_event_id_v1": (first.opening_transition_event_id_v1)}
    )

    with pytest.raises(ValueError, match="multiple sessions or signs"):
        analyze_direction_cohort_v1(
            (first, contradictory),
            phase="prospective_development",
        )


def test_every_frozen_baseline_is_present_or_explicitly_unavailable() -> None:
    payload = _episode(ordinal=0).model_dump(mode="python")
    payload["baseline_unavailable_reasons"] = ()

    with pytest.raises(ValidationError, match="every frozen baseline"):
        OpeningReversalAnalysisEpisodeV1.model_validate(payload)


def test_analysis_result_maps_are_deeply_immutable() -> None:
    result = analyze_direction_cohort_v1(
        (_episode(ordinal=0),),
        phase="prospective_development",
    )

    with pytest.raises(TypeError):
        result.baseline_means[0][1] = 1.0  # type: ignore[index]


def test_option_economics_remains_a_separate_actual_bid_ask_decision() -> None:
    rows = tuple(
        OpeningReversalOptionEpisodeV1(
            prediction_receipt_hash_v1=f"{index + 1:064x}",
            predicted_leg_outcome_hash_v1=f"{index + 10_001:064x}",
            opposite_leg_outcome_hash_v1=f"{index + 20_001:064x}",
            session=date(2027, 1, 4) + timedelta(days=index // 2),
            stock=f"S{index % 20:02d}",
            opening_transition_event_id_v1=f"event-{index // 2:03d}",
            prediction_v1="CALL" if index % 2 == 0 else "PUT",
            expiry=date(2027, 1, 5) + timedelta(days=index // 2),
            predicted_leg_conservative_return_v1=0.10,
            opposite_leg_conservative_return_v1=-0.05,
            actual_bid_ask_evidence=True,
            quote_quality_passed=True,
            staleness_passed=True,
            continuously_or_adequately_quoted=True,
        )
        for index in range(100)
    )

    result = analyze_option_economics_v1(
        rows,
        underlying_direction_supported=True,
    )

    assert result.support_passes
    assert result.episode_count == 100
    assert result.call_episode_count == 50
    assert result.put_episode_count == 50
    assert result.unique_event_count == 50
    assert result.mean_predicted_leg_conservative_return == pytest.approx(0.10)
    assert result.mean_opposite_leg_conservative_return == pytest.approx(-0.05)
    assert result.event_cluster_interval.lower_95 == pytest.approx(0.10)
    assert result.decision == ("prospective_opening_reversal_option_economics_supported")

    with pytest.raises(ValueError, match="duplicate prediction receipts"):
        analyze_option_economics_v1(
            (*rows, rows[0]),
            underlying_direction_supported=True,
        )
