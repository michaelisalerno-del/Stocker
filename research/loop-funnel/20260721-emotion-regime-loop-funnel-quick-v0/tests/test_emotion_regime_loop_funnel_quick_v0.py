from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.emotion_regime_loop_funnel_quick_v0 import (
    BEHAVIOURAL_DIMENSIONS,
    BlockedScreen,
    build_interactions,
    decide_funnel,
    join_behavioural_ledger,
    join_v2_posteriors,
    multiclass_brier,
    permute_behavioural_bundle_within_slates,
    pool_target_class,
    prediction_entropy,
    reject_protected_dates,
    resolve_first_loop_target,
    select_exact_loop_classes,
    session_block_bootstrap_draws,
    validate_checkpoint_timing,
)
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine


def _behavioural_row() -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": "AAA",
        "session": "2024-02-01",
        "decision_ordinal": 6,
        "feature_available_timestamp_utc": pd.Timestamp("2024-02-01T15:00:00Z"),
        "z_activity_effort": 1.0,
        "z_range_effort": 2.0,
        "z_travel_effort": 3.0,
        "z_absolute_efficiency": 0.5,
        "z_close_retention": 1.0,
        "z_directional_persistence": 1.5,
        "z_extreme_rejection": 4.0,
        "z_absolute_progress": 0.25,
        "z_compression": 0.75,
        "z_signed_progress": -1.0,
        "z_signed_efficiency": -2.0,
        "z_mean_close_location": -3.0,
        "z_boundary_slope": -4.0,
        "z_effort_acceleration": 2.0,
        "z_aligned_progress_acceleration": 0.5,
        "z_directional_rejection": 0.25,
        "z_return_gap": -1.0,
        "z_activity_gap": 2.0,
        "z_range_gap": -3.0,
        "return_gap": -0.1,
    }
    signed_pressure = -2.5
    exhaustion = 1.75
    row.update(
        {
            "arousal": 2.0,
            "conviction": 1.0,
            "frustration": (1.0 + 3.0 + 4.0) / 3.0 - (0.25 + 0.5) / 2.0,
            "tension": (1.0 + 0.75 + 4.0) / 3.0 - 0.25,
            "signed_pressure": signed_pressure,
            "pressure_magnitude": abs(signed_pressure),
            "exhaustion_magnitude": exhaustion,
            "signed_exhaustion": -exhaustion,
            "independence": 2.0,
            "signed_independence": -2.0,
        }
    )
    return row


def test_behavioural_ledger_join_is_exact_and_reconstructs_all_dimensions() -> None:
    ledger = pd.DataFrame([_behavioural_row()])
    decisions = ledger.loc[
        :, ["symbol", "session", "decision_ordinal", "feature_available_timestamp_utc"]
    ].copy()

    joined, audit = join_behavioural_ledger(decisions, ledger)

    assert len(joined) == 1
    assert audit["maximum_absolute_reconstruction_error"] <= 1e-12
    assert set(BEHAVIOURAL_DIMENSIONS).issubset(joined.columns)


def test_behavioural_ledger_join_fails_closed_on_value_drift() -> None:
    ledger = pd.DataFrame([_behavioural_row()])
    decisions = ledger.loc[
        :, ["symbol", "session", "decision_ordinal", "feature_available_timestamp_utc"]
    ].copy()
    ledger.loc[0, "arousal"] = float(ledger.loc[0, "arousal"]) + 1e-10

    with pytest.raises(BlockedScreen, match="blocked_behavioural_ledger_not_reconstructable"):
        join_behavioural_ledger(decisions, ledger)


def test_checkpoint_timing_requires_exact_new_york_clocks() -> None:
    valid = pd.DataFrame(
        {
            "decision_ordinal": [6, 12],
            "feature_available_timestamp_utc": pd.to_datetime(
                ["2024-02-01T15:00:00Z", "2024-02-01T15:30:00Z"], utc=True
            ),
        }
    )
    assert validate_checkpoint_timing(valid)["passed"] is True

    invalid = valid.copy()
    invalid.loc[1, "feature_available_timestamp_utc"] += pd.Timedelta(minutes=5)
    with pytest.raises(BlockedScreen, match="blocked_chronology_or_leakage_failure"):
        validate_checkpoint_timing(invalid)


def test_protected_date_rejection_occurs_before_scoring() -> None:
    safe = pd.DataFrame({"session": ["2025-08-22"]})
    reject_protected_dates(safe)

    protected = pd.DataFrame({"session": ["2025-08-23"]})
    with pytest.raises(BlockedScreen, match="blocked_protected_boundary_failure"):
        reject_protected_dates(protected)


def test_dimension_values_are_finite() -> None:
    ledger = pd.DataFrame([_behavioural_row()])
    values = ledger.loc[:, list(BEHAVIOURAL_DIMENSIONS)].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_v2_posterior_join_is_exact() -> None:
    timestamp = pd.Timestamp("2024-02-01T15:00:00Z")
    archived = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "session": ["2024-02-01"],
            "decision_ordinal": [6],
            "feature_available_timestamp_utc": [timestamp],
            **{f"posterior_state_{state}": [1.0 / 8.0] for state in range(8)},
            "posterior_entropy": [np.log(8.0)],
            "maximum_posterior_probability": [1.0 / 8.0],
            "current_state": [0],
        }
    )
    reproduced = archived.loc[
        :, ["symbol", "session", "decision_ordinal", "feature_available_timestamp_utc"]
    ].copy()
    for state in range(8):
        reproduced[f"state_p_{state}"] = 1.0 / 8.0
    reproduced["posterior_entropy_reproduced"] = np.log(8.0)
    reproduced["top_state_probability"] = 1.0 / 8.0
    reproduced["hard_top_state"] = 0
    reproduced["top_second_margin"] = 0.0
    reproduced["expected_state_age"] = 1.0
    reproduced["transition_probability"] = 0.25
    reproduced["persistence_probability"] = 0.75

    joined, audit = join_v2_posteriors(archived, reproduced)

    assert len(joined) == 1
    assert audit["maximum_posterior_absolute_error"] <= 1e-12


def _event_engine() -> FirstNextLoopEventEngine:
    primitive = decompose_closed_path((0, 2, 0))
    composite = decompose_closed_path((0, 1, 0, 2, 0))
    dictionary = LoopDictionary(
        {
            primitive.semantic_loop_id: primitive,
            composite.semantic_loop_id: composite,
        },
        (),
        version="test-v2",
    )
    return FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))


def test_six_bar_first_event_target_preserves_orientation() -> None:
    engine = _event_engine()
    timestamps = pd.date_range("2024-02-01T14:50:00Z", periods=3, freq="5min").to_pydatetime()
    trace = engine.scan_state_events(
        [0, 2, 0],
        bar_ordinals=[4, 6, 7],
        event_timestamps=timestamps,
        available_timestamps=timestamps,
    )

    target = resolve_first_loop_target(
        engine,
        trace,
        decision_id="AAA|2024-02-01|6",
        decision_event_index=0,
        decision_bar_ordinal=5,
        decision_timestamp=pd.Timestamp("2024-02-01T15:00:00Z").to_pydatetime(),
        session_end_bar_ordinal=77,
    )

    assert target["raw_outcome"] == "REGISTERED_PRIMITIVE_COMPLETION"
    assert target["semantic_loop_id"].startswith("loop_p_")
    assert target["orientation"].endswith("o_0-2-0")
    assert target["bars_until_completion"] == 2


def test_tied_registered_completion_is_not_broken_lexically() -> None:
    engine = _event_engine()
    timestamps = pd.date_range("2024-02-01T14:45:00Z", periods=5, freq="5min").to_pydatetime()
    trace = engine.scan_state_events(
        [0, 1, 0, 2, 0],
        bar_ordinals=[2, 3, 5, 6, 7],
        event_timestamps=timestamps,
        available_timestamps=timestamps,
    )
    target = resolve_first_loop_target(
        engine,
        trace,
        decision_id="AAA|2024-02-01|6",
        decision_event_index=2,
        decision_bar_ordinal=5,
        decision_timestamp=pd.Timestamp("2024-02-01T15:00:00Z").to_pydatetime(),
        session_end_bar_ordinal=77,
    )
    assert target["raw_outcome"] == "TIED_REGISTERED_COMPLETION"
    assert target["target_excluded"] is True
    assert len(target["tied_semantic_loop_ids"]) == 2


def test_no_transition_within_horizon_is_not_mislabeled_session_end() -> None:
    engine = _event_engine()
    timestamp = pd.Timestamp("2024-02-01T14:55:00Z").to_pydatetime()
    trace = engine.scan_state_events(
        [0],
        bar_ordinals=[5],
        event_timestamps=[timestamp],
        available_timestamps=[timestamp],
    )
    target = resolve_first_loop_target(
        engine,
        trace,
        decision_id="AAA|2024-02-01|6",
        decision_event_index=0,
        decision_bar_ordinal=5,
        decision_timestamp=pd.Timestamp("2024-02-01T15:00:00Z").to_pydatetime(),
        session_end_bar_ordinal=77,
    )
    assert target["raw_outcome"] == "NO_REGISTERED_COMPLETION"


def _selection_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    supports = {"loop_a|o_a": 60, "loop_b|o_b": 55, "loop_c|o_c": 50, "loop_d|o_d": 50}
    months = ["01", "02", "03", "04"]
    for loop_key, support in supports.items():
        for index in range(support):
            rows.append(
                {
                    "year": 2024,
                    "session": f"2024-{months[index % 4]}-{index % 20 + 1:02d}",
                    "symbol": f"S{index % 10:02d}",
                    "year_month": f"2024-{months[index % 4]}",
                    "raw_outcome": "REGISTERED_PRIMITIVE_COMPLETION",
                    "oriented_loop_key": loop_key,
                }
            )
    rows.extend(
        {
            "year": 2025,
            "session": "2025-01-02",
            "symbol": "S00",
            "year_month": "2025-01",
            "raw_outcome": "REGISTERED_PRIMITIVE_COMPLETION",
            "oriented_loop_key": "future_only|o_future",
        }
        for _ in range(100)
    )
    return pd.DataFrame(rows)


def test_exact_loop_selection_uses_development_only_and_stable_ties() -> None:
    mapping, support = select_exact_loop_classes(_selection_panel())
    assert mapping == {
        "loop_a|o_a": "LOOP_1",
        "loop_b|o_b": "LOOP_2",
        "loop_c|o_c": "LOOP_3",
        "loop_d|o_d": "LOOP_4",
    }
    assert "future_only|o_future" not in support["eligible_oriented_loops"]


@pytest.mark.parametrize(
    ("raw_outcome", "key", "expected"),
    [
        ("REGISTERED_PRIMITIVE_COMPLETION", "loop_a|o_a", "LOOP_1"),
        ("REGISTERED_REPEAT_COMPLETION", "other|o_other", "OTHER_REGISTERED_LOOP"),
        ("UNREGISTERED_LOOP", None, "UNREGISTERED_LOOP"),
        ("SESSION_END", None, "NO_REGISTERED_COMPLETION"),
        ("NO_REGISTERED_COMPLETION", None, "NO_REGISTERED_COMPLETION"),
        ("TIED_REGISTERED_COMPLETION", None, None),
    ],
)
def test_target_pooling(raw_outcome: str, key: str | None, expected: str | None) -> None:
    assert pool_target_class(raw_outcome, key, {"loop_a|o_a": "LOOP_1"}) == expected


def test_all_four_preregistered_interaction_groups_are_exact() -> None:
    frame = pd.DataFrame(
        {
            **{
                f"state_p_{state}": [0.1 + state / 100.0, 0.2 + state / 100.0] for state in range(8)
            },
            "signed_pressure": [2.0, -3.0],
            "signed_exhaustion": [-4.0, 5.0],
            "posterior_entropy": [0.5, 1.0],
            "frustration": [1.5, -2.0],
            "tension": [0.25, 3.0],
            "transition_probability": [0.2, 0.4],
            "arousal": [2.5, -1.0],
            "top_second_margin": [0.3, 0.6],
            "conviction": [0.75, -0.5],
        }
    )
    interactions, no_bounds = build_interactions(frame)
    assert interactions.shape == (2, 20)
    assert interactions.loc[0, "state_p_0_x_signed_pressure"] == pytest.approx(0.2)
    assert interactions.loc[1, "state_p_7_x_signed_exhaustion"] == pytest.approx(1.35)
    assert interactions.loc[0, "posterior_entropy_x_frustration"] == pytest.approx(0.75)
    assert interactions.loc[1, "posterior_entropy_x_tension"] == pytest.approx(3.0)
    assert interactions.loc[0, "transition_probability_x_arousal"] == pytest.approx(0.5)
    assert interactions.loc[1, "top_second_margin_x_conviction"] == pytest.approx(-0.3)
    assert no_bounds == {}
    clipped, bounds = build_interactions(frame, fit_bounds=True)
    assert len(bounds) == 20
    assert clipped.loc[0, "state_p_0_x_signed_pressure"] == pytest.approx(
        bounds["state_p_0_x_signed_pressure"][1]
    )


def test_multiclass_brier_is_mean_sum_of_classwise_squared_errors() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=float)
    targets = np.array([0, 1], dtype=int)
    expected = np.mean(
        [
            (0.7 - 1.0) ** 2 + 0.2**2 + 0.1**2,
            0.1**2 + (0.3 - 1.0) ** 2 + 0.6**2,
        ]
    )
    assert multiclass_brier(targets, probabilities) == pytest.approx(expected)


def test_prediction_entropy_uses_natural_log_and_zero_safe_terms() -> None:
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=float)
    entropy = prediction_entropy(probabilities)
    assert entropy.tolist() == pytest.approx([0.0, np.log(2.0)])


def test_session_block_bootstrap_preserves_whole_session_slates() -> None:
    frame = pd.DataFrame(
        {
            "session": np.repeat(["2025-01-02", "2025-01-03", "2025-01-06"], 4),
            "decision_ordinal": [6, 6, 12, 12] * 3,
            "symbol": ["AAA", "BBB", "AAA", "BBB"] * 3,
        }
    )
    draws = session_block_bootstrap_draws(frame, draws=5, seed=17)
    assert len(draws) == 5
    for indices in draws:
        sampled = frame.iloc[indices]
        assert len(sampled) == len(frame)
        for session in sampled["session"].unique():
            count = int(sampled["session"].eq(session).sum())
            assert count % int(frame["session"].eq(session).sum()) == 0


def test_within_slate_permutation_moves_complete_behavioural_bundle() -> None:
    rows = []
    for slate in ("2024-01-02|06", "2024-01-02|12"):
        for index, symbol in enumerate(("AAA", "BBB", "CCC")):
            row: dict[str, object] = {
                "slate_id": slate,
                "symbol": symbol,
                "state_p_0": index / 10.0,
            }
            row.update(
                {
                    feature: index * 10.0 + offset
                    for offset, feature in enumerate(
                        (
                            "arousal",
                            "conviction",
                            "frustration",
                            "tension",
                            "signed_pressure",
                            "signed_exhaustion",
                        )
                    )
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    shuffled = permute_behavioural_bundle_within_slates(frame, seed=23)
    assert shuffled[["slate_id", "symbol", "state_p_0"]].equals(
        frame[["slate_id", "symbol", "state_p_0"]]
    )
    for slate, original in frame.groupby("slate_id", sort=True):
        permuted = shuffled.loc[shuffled["slate_id"].eq(slate)]
        original_bundles = sorted(
            map(
                tuple,
                original.loc[
                    :,
                    [
                        "arousal",
                        "conviction",
                        "frustration",
                        "tension",
                        "signed_pressure",
                        "signed_exhaustion",
                    ],
                ].to_numpy(),
            )
        )
        permuted_bundles = sorted(
            map(
                tuple,
                permuted.loc[
                    :,
                    [
                        "arousal",
                        "conviction",
                        "frustration",
                        "tension",
                        "signed_pressure",
                        "signed_exhaustion",
                    ],
                ].to_numpy(),
            )
        )
        assert original_bundles == permuted_bundles


@pytest.mark.parametrize(
    ("m1_pass", "m2_pass", "descriptive", "expected"),
    [
        (True, True, True, "regime_mix_filters_behaviour_into_loop_distribution"),
        (True, False, True, "behaviour_main_effects_only"),
        (False, False, True, "descriptive_funnel_only_no_predictive_increment"),
        (False, False, False, "no_behaviour_regime_loop_funnel_increment"),
    ],
)
def test_decision_logic(m1_pass: bool, m2_pass: bool, descriptive: bool, expected: str) -> None:
    assert (
        decide_funnel(
            m1_pass=m1_pass,
            m2_pass=m2_pass,
            descriptive_change=descriptive,
        )
        == expected
    )
