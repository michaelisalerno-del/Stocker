from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stocker_research.causal_state_export_v2 import (
    HysteresisConfig,
    SoftLoopPrefixTracker,
    audit_legacy_run_context,
    build_completed_bar_decisions,
    build_hard_state_runs_v2,
    causal_semimarkov_filter_v2,
    churn_diagnostics,
    expand_duration_hazard_v2,
    hysteretic_states,
)
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_events_v2 import FeatureProvenance

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _model() -> dict[str, np.ndarray]:
    return {
        "duration_hazard": np.asarray([[0.25, 0.5, 0.7], [0.2, 0.4, 0.6]]),
        "transitions": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "initial": np.asarray([0.6, 0.4]),
    }


def test_completed_bar_feature_is_unavailable_before_bar_completion() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        FeatureProvenance(
            source_timestamp=BASE,
            source_bar_ordinal=0,
            available_timestamp=BASE + timedelta(minutes=5),
            decision_timestamp=BASE + timedelta(minutes=4, seconds=59),
            causal_valid=True,
            missing_reason=None,
            source_field="close",
            source_artifact_hash="a" * 64,
        )


def test_feature_provenance_accepts_exact_completed_bar_boundary() -> None:
    provenance = FeatureProvenance(
        source_timestamp=BASE,
        source_bar_ordinal=0,
        available_timestamp=BASE + timedelta(minutes=5),
        decision_timestamp=BASE + timedelta(minutes=5),
        causal_valid=True,
        missing_reason=None,
        source_field="close",
        source_artifact_hash="a" * 64,
    )

    assert provenance.source_timestamp <= provenance.decision_timestamp
    assert provenance.available_timestamp <= provenance.decision_timestamp


def test_feature_provenance_rejects_availability_before_source() -> None:
    with pytest.raises(ValueError, match="precedes its source"):
        FeatureProvenance(
            source_timestamp=BASE,
            source_bar_ordinal=0,
            available_timestamp=BASE - timedelta(seconds=1),
            decision_timestamp=BASE + timedelta(minutes=5),
            causal_valid=True,
            missing_reason=None,
            source_field="close",
            source_artifact_hash="a" * 64,
        )


def test_invalid_feature_provenance_requires_missing_reason() -> None:
    with pytest.raises(ValueError, match="missing reason"):
        FeatureProvenance(
            source_timestamp=BASE,
            source_bar_ordinal=0,
            available_timestamp=BASE,
            decision_timestamp=BASE,
            causal_valid=False,
            missing_reason=None,
            source_field="close",
            source_artifact_hash="a" * 64,
        )


def test_run_start_feature_uses_first_source_bar_and_exposes_legacy_mismatch() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session": ["2024-01-02"] * 4,
            "bar_ordinal": [0, 1, 2, 3],
            "bar_start_timestamp": [BASE + timedelta(minutes=5 * i) for i in range(4)],
            "bar_complete_timestamp": [BASE + timedelta(minutes=5 * (i + 1)) for i in range(4)],
            "b0_state_numeric": [-1.0, 0.0, 1.0, 1.0],
        }
    )
    labels = np.asarray([2, 2, 2, 4])

    runs = build_hard_state_runs_v2(bars, labels, context_fields=("b0_state_numeric",))
    audit = audit_legacy_run_context(bars, labels, context_fields=("b0_state_numeric",))

    assert runs.loc[0, "b0_state_numeric"] == -1.0
    assert runs.loc[0, "b0_state_numeric__source_bar_ordinal"] == 0
    assert audit.loc[0, "start_value"] == -1.0
    assert audit.loc[0, "end_value"] == 1.0
    assert audit.loc[0, "stored_legacy_value"] == 1.0
    assert bool(audit.loc[0, "start_end_differ"])


def test_missing_run_start_context_stays_missing() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "session": ["2024-01-02", "2024-01-02"],
            "bar_ordinal": [0, 1],
            "bar_start_timestamp": [BASE, BASE + timedelta(minutes=5)],
            "bar_complete_timestamp": [
                BASE + timedelta(minutes=5),
                BASE + timedelta(minutes=10),
            ],
            "b0_state_numeric": [np.nan, 1.0],
        }
    )
    runs = build_hard_state_runs_v2(bars, np.asarray([2, 2]), context_fields=("b0_state_numeric",))

    assert pd.isna(runs.loc[0, "b0_state_numeric"])
    assert runs.loc[0, "b0_state_numeric__missing_reason"] == "source_value_missing"
    assert not bool(runs.loc[0, "b0_state_numeric__causal_valid"])


def test_run_builder_uses_positions_when_dataframe_index_is_not_range_index() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "session": ["2024-01-02", "2024-01-02"],
            "bar_ordinal": [0, 1],
            "bar_start_timestamp": [BASE, BASE + timedelta(minutes=5)],
            "bar_complete_timestamp": [
                BASE + timedelta(minutes=5),
                BASE + timedelta(minutes=10),
            ],
            "b0_state_numeric": [-1.0, 1.0],
        },
        index=[10, 20],
    )

    runs = build_hard_state_runs_v2(bars, np.asarray([2, 2]), context_fields=("b0_state_numeric",))

    assert runs.loc[0, "start_row"] == 0
    assert runs.loc[0, "end_row"] == 1
    assert runs.loc[0, "b0_state_numeric"] == -1.0


def test_full_causal_posterior_normalizes_and_exports_top_two_metrics() -> None:
    log_emissions = np.log(
        np.asarray(
            [
                [0.8, 0.2],
                [0.55, 0.45],
                [0.2, 0.8],
            ]
        )
    )
    starts = tuple(BASE + timedelta(minutes=5 * i) for i in range(3))

    exported = causal_semimarkov_filter_v2(
        log_emissions,
        session_groups=(np.asarray([0, 1, 2]),),
        model=_model(),
        bar_start_timestamps=starts,
        bar_duration=timedelta(minutes=5),
    )

    assert np.allclose(exported.state_probabilities.sum(axis=1), 1.0)
    assert np.allclose(exported.state_age_probabilities.sum(axis=(1, 2)), 1.0)
    assert np.allclose(exported.next_state_probabilities.sum(axis=1), 1.0)
    expected_entropy = -np.sum(
        exported.state_probabilities[0] * np.log(exported.state_probabilities[0])
    )
    assert exported.posterior_entropy[0] == pytest.approx(expected_entropy)
    assert exported.top_second_margin[0] == pytest.approx(
        exported.state_probabilities[0, exported.top_state[0]]
        - exported.state_probabilities[0, exported.second_state[0]]
    )
    assert exported.hard_map_state.tolist() == exported.top_state.tolist()
    assert exported.available_timestamps[0] == starts[0] + timedelta(minutes=5)


def test_v2_duration_hazard_extends_age_24_without_forcing_exit() -> None:
    frozen = _model()
    frozen["duration_hazard"] = np.asarray([[0.10] * 23 + [1.0], [0.20] * 23 + [1.0]])

    expanded = expand_duration_hazard_v2(frozen, maximum_age=78, tail_window=6)

    assert expanded["duration_hazard"].shape == (2, 78)
    assert np.allclose(expanded["duration_hazard"][:, :23], frozen["duration_hazard"][:, :23])
    assert expanded["duration_hazard"][0, 23] == pytest.approx(0.10)
    assert expanded["duration_hazard"][1, 23] == pytest.approx(0.20)
    assert np.all(expanded["duration_hazard"][:, 23:] < 1.0)


def test_causal_filter_rejects_nonchronological_session_positions() -> None:
    starts = tuple(BASE + timedelta(minutes=5 * i) for i in range(2))

    with pytest.raises(ValueError, match="strictly increasing"):
        causal_semimarkov_filter_v2(
            np.log(np.asarray([[0.8, 0.2], [0.7, 0.3]])),
            session_groups=(np.asarray([1, 0]),),
            model=_model(),
            bar_start_timestamps=starts,
            bar_duration=timedelta(minutes=5),
        )


def test_hard_map_reproduces_argmax_and_expected_age_is_not_hard_run_age() -> None:
    log_emissions = np.log(np.asarray([[0.51, 0.49], [0.51, 0.49], [0.51, 0.49]]))
    starts = tuple(BASE + timedelta(minutes=5 * i) for i in range(3))
    exported = causal_semimarkov_filter_v2(
        log_emissions,
        session_groups=(np.asarray([0, 1, 2]),),
        model=_model(),
        bar_start_timestamps=starts,
        bar_duration=timedelta(minutes=5),
    )

    assert np.array_equal(exported.hard_map_state, np.argmax(exported.state_probabilities, axis=1))
    assert not np.allclose(exported.expected_state_age, exported.hard_map_run_age)


def test_hysteretic_state_uses_only_present_and_prior_causal_rows() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.30],
            [0.48, 0.52],
            [0.20, 0.80],
            [0.60, 0.40],
        ]
    )
    config = HysteresisConfig(switch_probability=0.60, switch_margin=0.10)
    original = hysteretic_states(probabilities, config=config)
    changed_future = probabilities.copy()
    changed_future[3] = [0.01, 0.99]
    replay = hysteretic_states(changed_future, config=config)

    assert original.tolist() == [0, 0, 1, 0]
    assert np.array_equal(original[:3], replay[:3])


def test_soft_prefix_mass_never_creates_a_hard_completion() -> None:
    dictionary = LoopDictionary.from_definitions(
        (decompose_closed_path((0, 1, 0)),), version="soft_test"
    )
    tracker = SoftLoopPrefixTracker(dictionary, state_count=2)

    first = tracker.update(np.asarray([0.8, 0.2]))
    second = tracker.update(np.asarray([0.3, 0.7]))
    third = tracker.update(np.asarray([0.9, 0.1]))

    assert first.hard_completion is False
    assert second.hard_completion is False
    assert third.hard_completion is False
    assert 0.0 <= third.highest_completion_probability <= 1.0
    assert third.approximation == "marginal_forward_diagnostic_not_a_hard_event"


def test_low_margin_transition_and_one_bar_reversal_are_reported() -> None:
    states = np.asarray([0, 1, 0, 0, 1, 1])
    margins = np.asarray([0.8, 0.02, 0.03, 0.7, 0.4, 0.5])
    entropy = np.asarray([0.1, 0.68, 0.67, 0.2, 0.3, 0.2])

    diagnostic = churn_diagnostics(
        states,
        margins=margins,
        entropy=entropy,
        low_margin_threshold=0.10,
    )

    assert diagnostic["hard_transitions"] == 3
    assert diagnostic["low_margin_hard_transitions"] == 2
    assert diagnostic["one_bar_reversals"] == 1
    assert diagnostic["two_bar_reversals"] >= 1


def test_every_eligible_completed_bar_gets_one_deterministic_decision() -> None:
    starts = [BASE + timedelta(minutes=5 * i) for i in range(4)]
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session": ["2024-01-02"] * 4,
            "bar_ordinal": [0, 1, 2, 3],
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": [value + timedelta(minutes=5) for value in starts],
            "bar_is_complete": [True, True, True, False],
        }
    )
    log_emissions = np.log(np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]]))
    exported = causal_semimarkov_filter_v2(
        log_emissions,
        session_groups=(np.asarray([0, 1, 2, 3]),),
        model=_model(),
        bar_start_timestamps=tuple(starts),
        bar_duration=timedelta(minutes=5),
    )
    decisions = build_completed_bar_decisions(
        bars,
        exported,
        git_sha="a" * 40,
        contract_hash="b" * 64,
        data_snapshot_hash="c" * 64,
        dictionary_version="dictionary_v2",
        state_model_version="state_v2",
    )
    replay = build_completed_bar_decisions(
        bars,
        exported,
        git_sha="a" * 40,
        contract_hash="b" * 64,
        data_snapshot_hash="c" * 64,
        dictionary_version="dictionary_v2",
        state_model_version="state_v2",
    )

    assert len(decisions) == 3
    assert decisions["decision_id"].is_unique
    assert decisions["decision_id"].tolist() == replay["decision_id"].tolist()
    assert decisions["bar_complete_timestamp"].le(decisions["decision_timestamp"]).all()
    assert decisions["bars_remaining_in_session"].tolist() == [2, 1, 0]
    assert decisions["is_session_start"].tolist() == [True, False, False]
    assert decisions["is_session_end"].tolist() == [False, False, True]
    assert decisions["is_run_entry"].sum() < len(decisions)


def test_decision_export_keeps_legacy_map_separate_and_resets_hysteresis() -> None:
    starts = [BASE + timedelta(minutes=5 * i) for i in range(4)]
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "bar_ordinal": [0, 1, 0, 1],
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": [value + timedelta(minutes=5) for value in starts],
            "bar_is_complete": [True] * 4,
        }
    )
    exported = causal_semimarkov_filter_v2(
        np.log(np.asarray([[0.9, 0.1], [0.9, 0.1], [0.01, 0.99], [0.01, 0.99]])),
        session_groups=(np.asarray([0, 1]), np.asarray([2, 3])),
        model=_model(),
        bar_start_timestamps=tuple(starts),
        bar_duration=timedelta(minutes=5),
    )
    legacy = np.asarray([1, 1, 0, 0])

    decisions = build_completed_bar_decisions(
        bars,
        exported,
        legacy_hard_states=legacy,
        git_sha="a" * 40,
        contract_hash="b" * 64,
        data_snapshot_hash="c" * 64,
        dictionary_version="dictionary_v2",
        state_model_version="state_v2",
        hysteresis_config=HysteresisConfig(switch_probability=0.55, switch_margin=0.10),
    )

    assert decisions["hard_state_legacy"].tolist() == legacy.tolist()
    assert decisions["hard_state_posterior_map"].tolist() == [0, 0, 1, 1]
    assert decisions["hard_state_hysteretic"].tolist() == [0, 0, 1, 1]
    assert decisions["hard_run_age"].tolist() == [1, 2, 1, 2]


def test_feature_ledger_forbidden_names_are_absent_from_decisions() -> None:
    starts = [BASE]
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "session": ["2024-01-02"],
            "bar_ordinal": [0],
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": [BASE + timedelta(minutes=5)],
            "bar_is_complete": [True],
        }
    )
    exported = causal_semimarkov_filter_v2(
        np.log(np.asarray([[0.9, 0.1]])),
        session_groups=(np.asarray([0]),),
        model=_model(),
        bar_start_timestamps=tuple(starts),
        bar_duration=timedelta(minutes=5),
    )
    decisions = build_completed_bar_decisions(
        bars,
        exported,
        git_sha="a" * 40,
        contract_hash="b" * 64,
        data_snapshot_hash="c" * 64,
        dictionary_version="dictionary_v2",
        state_model_version="state_v2",
    )

    forbidden = ("future", "payoff", "mfe", "mae", "route_completion")
    assert not any(token in column.lower() for column in decisions for token in forbidden)
