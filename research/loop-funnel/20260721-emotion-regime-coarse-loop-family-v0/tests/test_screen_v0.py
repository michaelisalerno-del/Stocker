from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.emotion_regime_coarse_loop_funnel_v0 import (
    BEHAVIOURAL_DIMENSIONS,
    BlockedScreen,
    build_interactions,
    decide_funnel,
    effective_candidate_count,
    freeze_coarse_target_mapping,
    join_behavioural_ledger,
    join_v2_posteriors,
    multiclass_brier,
    permute_behavioural_bundle_within_slates,
    pool_coarse_target,
    prediction_entropy,
    registered_family_from_motif,
    reject_protected_dates,
    resolve_first_loop_target,
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
    row.update(
        {
            "arousal": 2.0,
            "conviction": 1.0,
            "frustration": (1.0 + 3.0 + 4.0) / 3.0 - (0.25 + 0.5) / 2.0,
            "tension": (1.0 + 0.75 + 4.0) / 3.0 - 0.25,
            "signed_pressure": -2.5,
            "pressure_magnitude": 2.5,
            "exhaustion_magnitude": 1.75,
            "signed_exhaustion": -1.75,
            "independence": 2.0,
            "signed_independence": -2.0,
        }
    )
    return row


def test_behavioural_ledger_reconstruction_is_exact() -> None:
    ledger = pd.DataFrame([_behavioural_row()])
    keys = ledger.loc[
        :, ["symbol", "session", "decision_ordinal", "feature_available_timestamp_utc"]
    ]

    joined, audit = join_behavioural_ledger(keys, ledger)

    assert len(joined) == 1
    assert audit["maximum_absolute_reconstruction_error"] <= 1e-12
    assert set(BEHAVIOURAL_DIMENSIONS).issubset(joined.columns)


def test_v2_posterior_reconstruction_is_exact() -> None:
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


def test_checkpoint_timing_and_protected_boundary_are_fail_closed() -> None:
    valid = pd.DataFrame(
        {
            "decision_ordinal": [6, 12],
            "feature_available_timestamp_utc": pd.to_datetime(
                ["2024-02-01T15:00:00Z", "2024-02-01T15:30:00Z"], utc=True
            ),
        }
    )
    assert validate_checkpoint_timing(valid)["passed"] is True
    reject_protected_dates(pd.DataFrame({"session": ["2025-08-22"]}))

    invalid = valid.copy()
    invalid.loc[1, "feature_available_timestamp_utc"] += pd.Timedelta(minutes=5)
    with pytest.raises(BlockedScreen, match="blocked_chronology_or_leakage_failure"):
        validate_checkpoint_timing(invalid)
    with pytest.raises(BlockedScreen, match="blocked_protected_boundary_failure"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def _event_engine(*paths: tuple[int, ...]) -> FirstNextLoopEventEngine:
    definitions = [decompose_closed_path(path) for path in paths]
    dictionary = LoopDictionary(
        {definition.semantic_loop_id: definition for definition in definitions},
        (),
        version="test-v2",
    )
    return FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))


def _resolved_target(
    engine: FirstNextLoopEventEngine,
    states: list[int],
    ordinals: list[int],
    *,
    decision_event_index: int = 0,
    decision_bar_ordinal: int = 2,
    session_end_bar_ordinal: int = 77,
) -> dict[str, object]:
    timestamps = pd.date_range("2024-02-01T14:30:00Z", periods=len(states), freq="5min")
    trace = engine.scan_state_events(
        states,
        bar_ordinals=ordinals,
        event_timestamps=timestamps.to_pydatetime(),
        available_timestamps=timestamps.to_pydatetime(),
    )
    return resolve_first_loop_target(
        engine,
        trace,
        decision_id="AAA|2024-02-01|6",
        decision_event_index=decision_event_index,
        decision_bar_ordinal=decision_bar_ordinal,
        decision_timestamp=pd.Timestamp("2024-02-01T15:00:00Z").to_pydatetime(),
        session_end_bar_ordinal=session_end_bar_ordinal,
    )


def test_six_bar_first_event_is_registered_primitive() -> None:
    target = _resolved_target(_event_engine((0, 2, 0)), [0, 2, 0], [2, 4, 6])

    assert target["raw_outcome"] == "REGISTERED_PRIMITIVE"
    assert target["bars_until_completion"] <= 6


@pytest.mark.parametrize(
    ("motif_type", "expected"),
    [
        ("primitive", "REGISTERED_PRIMITIVE"),
        ("repeat", "REGISTERED_REPEAT"),
        ("composite", "REGISTERED_COMPOSITE"),
    ],
)
def test_registered_motif_family_classification(motif_type: str, expected: str) -> None:
    assert registered_family_from_motif(motif_type) == expected


def test_unregistered_and_no_completion_first_event_outcomes() -> None:
    engine = _event_engine((0, 2, 0))

    unregistered = _resolved_target(engine, [0, 1, 0], [2, 4, 6])
    no_completion = _resolved_target(engine, [0, 2], [2, 6])

    assert unregistered["raw_outcome"] == "UNREGISTERED_LOOP"
    assert no_completion["raw_outcome"] == "NO_REGISTERED_COMPLETION"


def test_tied_registered_completion_is_excluded_without_lexical_break() -> None:
    engine = _event_engine((0, 2, 0), (0, 1, 0, 2, 0))
    target = _resolved_target(
        engine,
        [0, 1, 0, 2, 0],
        [1, 2, 3, 5, 7],
        decision_event_index=2,
        decision_bar_ordinal=3,
    )

    assert target["raw_outcome"] == "TIED_REGISTERED_COMPLETION"
    assert target["target_excluded"] is True
    assert len(target["tied_semantic_loop_ids"]) == 2


def _supported_rows(raw_outcome: str, count: int, *, year: int = 2024) -> list[dict[str, object]]:
    months = ("01", "02", "03", "04")
    stocks = tuple(f"S{index:02d}" for index in range(10))
    return [
        {
            "raw_outcome": raw_outcome,
            "year": year,
            "session": (f"{year}-{months[(index % 40) // 10]}-{(index % 40) % 10 + 1:02d}"),
            "year_month": f"{year}-{months[(index % 40) // 10]}",
            "symbol": stocks[index % len(stocks)],
        }
        for index in range(count)
    ]


def test_development_only_subtype_pooling_can_freeze_four_classes() -> None:
    rows = [
        *_supported_rows("REGISTERED_PRIMITIVE", 80),
        *_supported_rows("REGISTERED_REPEAT", 40),
        *_supported_rows("REGISTERED_COMPOSITE", 40),
        *_supported_rows("UNREGISTERED_LOOP", 80),
        *_supported_rows("NO_REGISTERED_COMPLETION", 80),
    ]
    mapping, manifest = freeze_coarse_target_mapping(pd.DataFrame(rows))

    assert manifest["target_variant"] == "four_classes"
    assert mapping["REGISTERED_PRIMITIVE"] == "REGISTERED_PRIMITIVE"
    assert mapping["REGISTERED_REPEAT"] == "OTHER_REGISTERED_COMPLETION"
    assert mapping["REGISTERED_COMPOSITE"] == "OTHER_REGISTERED_COMPLETION"


def test_three_class_fallback_is_preregistered_not_a_support_failure() -> None:
    rows = [
        *_supported_rows("REGISTERED_PRIMITIVE", 80),
        *_supported_rows("UNREGISTERED_LOOP", 80),
        *_supported_rows("NO_REGISTERED_COMPLETION", 80),
    ]
    mapping, manifest = freeze_coarse_target_mapping(pd.DataFrame(rows))

    assert manifest["target_variant"] == "three_class_fallback"
    assert manifest["development_support_passed"] is True
    assert mapping["REGISTERED_PRIMITIVE"] == "REGISTERED_COMPLETION"
    assert mapping["REGISTERED_REPEAT"] == "REGISTERED_COMPLETION"
    assert mapping["REGISTERED_COMPOSITE"] == "REGISTERED_COMPLETION"


def test_assessment_rows_cannot_change_the_frozen_mapping() -> None:
    development = [
        *_supported_rows("REGISTERED_PRIMITIVE", 80),
        *_supported_rows("UNREGISTERED_LOOP", 80),
        *_supported_rows("NO_REGISTERED_COMPLETION", 80),
    ]
    assessment = _supported_rows("REGISTERED_REPEAT", 500, year=2025)
    mapping, _ = freeze_coarse_target_mapping(pd.DataFrame([*development, *assessment]))

    assert pool_coarse_target("REGISTERED_REPEAT", mapping) == "REGISTERED_COMPLETION"
    assert pool_coarse_target("TIED_REGISTERED_COMPLETION", mapping) is None
    assert pool_coarse_target("SOURCE_UNAVAILABLE", mapping) is None


def test_every_preregistered_interaction_group_is_exact() -> None:
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


def test_multiclass_brier_entropy_and_effective_candidate_count() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=float)
    targets = np.array([0, 1], dtype=int)
    expected_brier = np.mean(
        [
            (0.7 - 1.0) ** 2 + 0.2**2 + 0.1**2,
            0.1**2 + (0.3 - 1.0) ** 2 + 0.6**2,
        ]
    )
    deterministic = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=float)

    assert multiclass_brier(targets, probabilities) == pytest.approx(expected_brier)
    assert prediction_entropy(deterministic).tolist() == pytest.approx([0.0, np.log(2.0)])
    assert effective_candidate_count(deterministic).tolist() == pytest.approx([1.0, 2.0])


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
        for session in sampled["session"].unique():
            count = int(sampled["session"].eq(session).sum())
            assert count % int(frame["session"].eq(session).sum()) == 0


def test_within_slate_permutation_moves_the_complete_bundle_only() -> None:
    features = (
        "arousal",
        "conviction",
        "frustration",
        "tension",
        "signed_pressure",
        "signed_exhaustion",
    )
    rows: list[dict[str, object]] = []
    for slate in ("2024-01-02|06", "2024-01-02|12"):
        for index, symbol in enumerate(("AAA", "BBB", "CCC")):
            row: dict[str, object] = {
                "slate_id": slate,
                "symbol": symbol,
                "state_p_0": index / 10.0,
            }
            row.update({feature: index * 10.0 + offset for offset, feature in enumerate(features)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    shuffled = permute_behavioural_bundle_within_slates(frame, seed=23)

    assert shuffled[["slate_id", "symbol", "state_p_0"]].equals(
        frame[["slate_id", "symbol", "state_p_0"]]
    )
    for slate, original in frame.groupby("slate_id", sort=True):
        permuted = shuffled.loc[shuffled["slate_id"].eq(slate)]
        assert sorted(map(tuple, original.loc[:, list(features)].to_numpy())) == sorted(
            map(tuple, permuted.loc[:, list(features)].to_numpy())
        )


@pytest.mark.parametrize(
    ("m1_pass", "m2_pass", "descriptive", "expected"),
    [
        (True, True, True, "regime_mix_filters_behaviour_into_coarse_loop_family"),
        (True, False, True, "behaviour_main_effects_only"),
        (False, False, True, "descriptive_coarse_funnel_only"),
        (False, False, False, "no_behaviour_regime_family_increment"),
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
