from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.registered_loop_precursor_hidden_veto_v0 import (
    assign_hidden_risk,
    benjamini_hochberg,
    candidate_threshold,
    choose_primary_decision,
    deduplicate_registered_completions,
    exact_precursor_identity_eligible,
    freeze_hidden_risk_thresholds,
    hidden_event_target,
    nearest_precursor_label,
    opening_panel_differences,
    permute_hidden_probability_within_slates,
    precursor_window_features,
    registered_completion_target,
    reject_protected_dates,
    sample_matched_pseudo_completions,
    session_block_bootstrap_indices,
    veto_feature_frame,
)


def test_opening_panel_reconstruction_compares_shared_fields_and_probabilities() -> None:
    archived = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "session": ["2025-01-02", "2025-01-02"],
            "decision_ordinal": [6, 6],
            "state_p_0": [0.2, 0.4],
            "B0_probability": [0.1, 0.3],
            "U1_probability": [0.2, 0.5],
        }
    )
    reconstructed = archived.copy()
    reconstructed.loc[1, "state_p_0"] += 5e-13

    result = opening_panel_differences(
        archived,
        reconstructed,
        shared_fields=("state_p_0",),
        probability_fields=("B0_probability", "U1_probability"),
    )

    assert result["rows"] == 2
    assert result["maximum_shared_field_difference"] == pytest.approx(5e-13)
    assert result["maximum_probability_difference"] == 0.0
    assert result["passed"] is True


def test_registered_and_hidden_targets_are_strict_and_horizon_frozen() -> None:
    completions = pd.DataFrame({"completion_bar_ordinal": [5, 6, 17, 18]})
    hidden = pd.DataFrame({"completion_bar_ordinal": [5, 6, 11, 12]})

    assert registered_completion_target(5, completions) == 1
    assert registered_completion_target(17, completions) == 1
    assert registered_completion_target(18, completions) == 0
    assert hidden_event_target(5, hidden) == 1
    assert hidden_event_target(11, hidden) == 1
    assert hidden_event_target(12, hidden) == 0


def test_registered_event_deduplication_attaches_latest_eligible_checkpoint() -> None:
    completions = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "session": ["2025-01-02"] * 3,
            "completion_timestamp_utc": [pd.Timestamp("2025-01-02T15:35:00Z")] * 3,
            "completion_bar_ordinal": [13] * 3,
            "semantic_loop_id": ["loop_p_1-2-1"] * 3,
            "motif_type": ["primitive"] * 3,
            "orientation_id": ["one", "two", "three"],
        }
    )
    decisions = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "session": ["2025-01-02", "2025-01-02"],
            "decision_ordinal": [6, 12],
            "repo_bar_start_ordinal": [5, 11],
            "feature_available_timestamp_utc": [
                pd.Timestamp("2025-01-02T15:00:00Z"),
                pd.Timestamp("2025-01-02T15:30:00Z"),
            ],
        }
    )

    result = deduplicate_registered_completions(completions, decisions)

    assert len(result) == 1
    assert int(result.iloc[0]["decision_ordinal"]) == 12
    assert result.iloc[0]["orientation_ids_json"] == '["one", "three", "two"]'


def _precursor_fixture() -> tuple[
    pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    completion = pd.Series(
        {
            "completion_bar_ordinal": 20,
            "semantic_loop_id": "loop_p_1-2-1",
            "motif_type": "primitive",
        }
    )
    registered = pd.DataFrame(
        {
            "completion_bar_ordinal": [16, 17],
            "semantic_loop_id": ["loop_p_4-5-4", "loop_p_1-2-1"],
            "motif_type": ["primitive", "primitive"],
        }
    )
    hidden = pd.DataFrame(
        {
            "completion_bar_ordinal": [19],
            "hidden_family_class": ["unregistered_primitive_like__5-6-5"],
        }
    )
    prefixes = pd.DataFrame(
        {
            "bar_ordinal": [18, 19],
            "semantic_loop_id": ["loop_p_4-5-4", "loop_p_1-2-1"],
            "orientation_id": ["other", "matching"],
            "progress_states": [2, 3],
        }
    )
    states = pd.DataFrame(
        {
            "bar_ordinal": list(range(8, 20)),
            "causal_hard_state": [0] * 10 + [1, 1],
            "transition_probability": np.linspace(0.1, 0.21, 12),
            "posterior_entropy": np.linspace(0.5, 0.61, 12),
            "top_state_probability": np.linspace(0.9, 0.79, 12),
            "expected_state_age": np.arange(1.0, 13.0),
        }
    )
    return completion, registered, hidden, prefixes, states


def test_three_six_and_twelve_bar_precursor_windows_are_strictly_pre_completion() -> None:
    completion, registered, hidden, prefixes, states = _precursor_fixture()

    three = precursor_window_features(
        completion, registered, hidden, prefixes, states, lookback_bars=3
    )
    six = precursor_window_features(
        completion, registered, hidden, prefixes, states, lookback_bars=6
    )
    twelve = precursor_window_features(
        completion, registered, hidden, prefixes, states, lookback_bars=12
    )

    assert three["same_registered_identity"] is True
    assert three["same_registered_broad_family_different_identity"] is False
    assert six["same_registered_broad_family_different_identity"] is True
    assert twelve["complete_prior_history"] is True
    assert all(result["window_end_bar_ordinal"] == 19 for result in (three, six, twelve))


def test_matching_other_hidden_and_regime_precursors_are_all_retained() -> None:
    completion, registered, hidden, prefixes, states = _precursor_fixture()

    result = precursor_window_features(
        completion, registered, hidden, prefixes, states, lookback_bars=3
    )

    assert result["active_prefix_immediately_before_completion"] is True
    assert result["matching_prefix_any"] is True
    assert result["other_prefix_any"] is True
    assert result["prefix_candidate_count"] == 2
    assert result["maximum_prefix_depth"] == 3
    assert result["hidden_5_6_5"] is True
    assert result["any_regime_transition"] is True
    assert result["regime_transition_count"] == 1
    assert result["no_identified_structural_precursor"] is False


def test_regime_transition_into_first_lookback_bar_is_retained() -> None:
    completion, registered, hidden, prefixes, states = _precursor_fixture()
    states.loc[states["bar_ordinal"].eq(17), "causal_hard_state"] = 1

    result = precursor_window_features(
        completion,
        registered.iloc[0:0],
        hidden.iloc[0:0],
        prefixes.iloc[0:0],
        states,
        lookback_bars=3,
    )

    assert result["state_ordinals_json"] == "[17, 18, 19]"
    assert result["any_regime_transition"] is True
    assert result["regime_transition_count"] == 1
    assert result["nearest_precursor_label"] == "REGIME_TRANSITION"
    assert result["no_identified_structural_precursor"] is False


def test_pooled_other_hidden_family_is_not_an_exact_precursor_identity() -> None:
    assert exact_precursor_identity_eligible("registered", "loop_p_1-2-1") is True
    assert exact_precursor_identity_eligible("hidden", "unregistered_primitive_like__5-6-5") is True
    assert exact_precursor_identity_eligible("hidden", "OTHER_UNREGISTERED_FAMILY") is False


def test_nearest_precursor_priority_prefers_completed_then_matching_then_other() -> None:
    assert (
        nearest_precursor_label(
            completed_loop=True,
            matching_active_prefix=True,
            other_active_prefix=True,
            regime_transition=True,
        )
        == "NEAREST_COMPLETED_LOOP_EVENT"
    )
    assert (
        nearest_precursor_label(
            completed_loop=False,
            matching_active_prefix=True,
            other_active_prefix=True,
            regime_transition=True,
        )
        == "ACTIVE_MATCHING_PREFIX"
    )
    assert (
        nearest_precursor_label(
            completed_loop=False,
            matching_active_prefix=False,
            other_active_prefix=True,
            regime_transition=True,
        )
        == "OTHER_ACTIVE_PREFIX"
    )


def test_matched_pseudo_completion_sampling_preserves_stock_month_and_clock() -> None:
    observed = pd.DataFrame(
        {
            "event_id": ["one", "two"],
            "symbol": ["A", "B"],
            "session": ["2025-01-02", "2025-01-03"],
            "year_month": ["2025-01", "2025-01"],
            "clock_bin": ["10:30", "10:30"],
            "completion_bar_ordinal": [15, 16],
            "completion_timestamp_utc": pd.to_datetime(
                ["2025-01-02T15:45:00Z", "2025-01-03T15:50:00Z"]
            ),
        }
    )
    eligible = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "session": ["2025-01-06", "2025-02-03", "2025-01-07", "2025-01-08"],
            "year_month": ["2025-01", "2025-02", "2025-01", "2025-01"],
            "clock_bin": ["10:30", "10:30", "10:30", "10:30"],
            "completion_bar_ordinal": [14, 14, 14, 15],
            "completion_timestamp_utc": pd.to_datetime(
                [
                    "2025-01-06T15:40:00Z",
                    "2025-02-03T15:40:00Z",
                    "2025-01-07T15:40:00Z",
                    "2025-01-08T15:45:00Z",
                ]
            ),
            "full_prior_history": [True, True, True, False],
            "registered_completion_at_timestamp": [False, False, False, False],
        }
    )

    sampled = sample_matched_pseudo_completions(observed, eligible, seed=17)

    assert sampled["symbol"].tolist() == ["A", "B"]
    assert sampled["year_month"].tolist() == ["2025-01", "2025-01"]
    assert sampled["clock_bin"].tolist() == ["10:30", "10:30"]
    assert sampled["session"].tolist() == ["2025-01-06", "2025-01-07"]
    assert sampled["source_event_id"].tolist() == ["one", "two"]


def test_bh_correction_and_protected_date_rejection() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.20]) == [
        0.04,
        0.05333333333333334,
        0.05333333333333334,
        0.20,
    ]
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def test_candidate_and_hidden_risk_thresholds_are_frozen_from_development() -> None:
    threshold = candidate_threshold(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))
    hidden = freeze_hidden_risk_thresholds(pd.Series(np.arange(1.0, 11.0) / 10.0))

    assert threshold == pytest.approx(0.42)
    assert hidden["low_maximum"] == pytest.approx(0.325)
    assert hidden["high_minimum"] == pytest.approx(0.775)
    assert hidden["quintile_boundaries"] == pytest.approx([0.28, 0.46, 0.64, 0.82])
    assigned = assign_hidden_risk(pd.Series([0.325, 0.5, 0.775]), hidden)
    assert assigned["hidden_risk_group"].tolist() == ["low", "middle", "high"]
    assert assigned["hidden_risk_quintile"].tolist() == [2, 3, 4]


def test_v0_v1_features_clip_logits_and_exclude_actual_hidden_event() -> None:
    frame = pd.DataFrame(
        {
            "B0_probability": [0.0, 0.8],
            "U1_probability": [1.0, 0.2],
            "decision_ordinal": [6, 12],
            "actual_hidden_event_within_6_bars": [1, 0],
        }
    )

    v0 = veto_feature_frame(frame, include_hidden_risk=False)
    v1 = veto_feature_frame(frame, include_hidden_risk=True)

    assert v0.columns.tolist() == ["logit_B0_probability", "checkpoint_12"]
    assert v1.columns.tolist() == [
        "logit_B0_probability",
        "checkpoint_12",
        "logit_U1_probability",
    ]
    assert np.isfinite(v1.to_numpy()).all()
    assert "actual_hidden_event_within_6_bars" not in v1


def test_session_bootstrap_resamples_whole_sessions() -> None:
    frame = pd.DataFrame({"session": ["A", "A", "B", "C", "C", "C"]})

    draws = session_block_bootstrap_indices(frame, draws=4, seed=23)

    assert len(draws) == 4
    for indices in draws:
        sampled = frame.iloc[indices]
        for session, count in sampled["session"].value_counts().items():
            assert count % int(frame["session"].eq(session).sum()) == 0


def test_hidden_probability_permutation_stays_inside_session_checkpoint_slate() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["A|06", "A|06", "A|12", "A|12"],
            "U1_probability": [0.1, 0.2, 0.7, 0.8],
            "B0_probability": [0.3, 0.4, 0.5, 0.6],
            "registered_completion_within_12_bars": [0, 1, 0, 1],
            "decision_ordinal": [6, 6, 12, 12],
        }
    )

    permuted = permute_hidden_probability_within_slates(frame, seed=31)

    assert permuted["B0_probability"].tolist() == frame["B0_probability"].tolist()
    assert (
        permuted["registered_completion_within_12_bars"].tolist()
        == frame["registered_completion_within_12_bars"].tolist()
    )
    for slate in frame["slate_id"].unique():
        before = sorted(frame.loc[frame["slate_id"].eq(slate), "U1_probability"])
        after = sorted(permuted.loc[permuted["slate_id"].eq(slate), "U1_probability"])
        assert before == after


def test_decision_logic_keeps_precursor_and_veto_support_independent() -> None:
    assert (
        choose_primary_decision(
            precursor_status="supported",
            predictive_veto_status="supported",
            realised_diversion_status="supported",
        )
        == "hidden_diversion_veto_and_registered_precursor_supported"
    )
    assert (
        choose_primary_decision(
            precursor_status="insufficient_support",
            predictive_veto_status="supported",
            realised_diversion_status="descriptive_only",
        )
        == "hidden_diversion_veto_supported_only"
    )
    assert (
        choose_primary_decision(
            precursor_status="supported",
            predictive_veto_status="not_supported",
            realised_diversion_status="not_supported",
        )
        == "registered_precursor_structure_supported_only"
    )
    assert (
        choose_primary_decision(
            precursor_status="descriptive_only",
            predictive_veto_status="not_supported",
            realised_diversion_status="descriptive_only",
        )
        == "descriptive_precursor_or_veto_structure_only"
    )
