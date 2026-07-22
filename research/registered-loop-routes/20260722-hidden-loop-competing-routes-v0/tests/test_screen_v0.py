from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.hidden_loop_competing_routes_v0 import (
    FROZEN_HIDDEN_FAMILIES,
    HIDDEN_A,
    HIDDEN_B,
    HIDDEN_C,
    HIDDEN_D,
    HIDDEN_OTHER,
    NO_REGISTERED_COMPLETION,
    OTHER_REGISTERED_COMPLETION,
    PREREGISTERED_TARGETS,
    TARGET_A,
    TARGET_B,
    TARGET_C,
    benjamini_hochberg,
    candidate_normalised_weights,
    candidate_threshold,
    choose_primary_decision,
    counterfactual_probability_difference,
    deduplicate_registered_completions,
    fit_multinomial,
    freeze_target_class_mapping,
    hidden_history_features,
    lookback_is_complete,
    map_registered_route,
    matched_control_relations,
    model_feature_sets,
    next_registered_route,
    permute_hidden_bundle,
    precursor_present,
    predict_multinomial,
    registered_history_features,
    reject_protected_dates,
    sample_matched_pseudo_completions,
    sequential_update_ordinals,
    session_bootstrap_multiplicities,
    target_prefix_snapshot,
    transition_hypothesis_manifest,
)


def _completion_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session": ["2024-01-02"] * 3,
            "completion_timestamp_utc": pd.to_datetime(
                ["2024-01-02T15:00:00Z"] * 2 + ["2024-01-02T15:05:00Z"]
            ),
            "completion_available_timestamp_utc": pd.to_datetime(
                ["2024-01-02T15:05:00Z"] * 2 + ["2024-01-02T15:10:00Z"]
            ),
            "completion_bar_ordinal": [6, 6, 7],
            "semantic_loop_id": [TARGET_A, TARGET_A, TARGET_C],
            "orientation_id": ["a", "a", "c"],
            "motif_type": ["primitive"] * 3,
        }
    )


def test_frozen_target_identities() -> None:
    assert PREREGISTERED_TARGETS == (TARGET_A, TARGET_B, TARGET_C)
    assert PREREGISTERED_TARGETS == (
        "loop_p_2-5-6-2",
        "loop_p_2-6-2",
        "loop_p_4-6-4",
    )


def test_frozen_hidden_family_identities() -> None:
    assert FROZEN_HIDDEN_FAMILIES == (HIDDEN_A, HIDDEN_B, HIDDEN_C, HIDDEN_D, HIDDEN_OTHER)


def test_development_only_target_support() -> None:
    rows = []
    for target in PREREGISTERED_TARGETS:
        for index in range(60):
            rows.append(
                {
                    "semantic_loop_id": target,
                    "session": f"2024-{index % 6 + 1:02d}-{index % 28 + 1:02d}",
                    "symbol": f"S{index % 10}",
                    "year_month": f"2024-{index % 6 + 1:02d}",
                }
            )
    mapping = freeze_target_class_mapping(pd.DataFrame(rows))
    assert mapping["retained_exact_targets"] == list(PREREGISTERED_TARGETS)
    assert mapping["final_target_classes"][:2] == [
        NO_REGISTERED_COMPLETION,
        OTHER_REGISTERED_COMPLETION,
    ]


def test_registered_event_deduplication_retains_orientation() -> None:
    result = deduplicate_registered_completions(_completion_rows())
    assert len(result) == 2
    assert result["event_id"].is_unique


def test_six_bar_lookback_eligibility() -> None:
    assert lookback_is_complete(range(20), 10, 6)
    assert not lookback_is_complete([5, 6, 7, 8, 10], 10, 6)


def test_twelve_bar_lookback_eligibility() -> None:
    assert lookback_is_complete(range(20), 15, 12)
    assert not lookback_is_complete(range(4, 15), 15, 12)


def test_every_fixed_transition_hypothesis() -> None:
    manifest = transition_hypothesis_manifest()
    assert [row["hypothesis_id"] for row in manifest] == ["H1", "H2", "H3", "H4"]
    assert manifest[2]["lookbacks"] == [6, 12]
    assert manifest[3]["expected_sign"] == "negative"


def test_transition_precursor_predicates() -> None:
    registered = pd.DataFrame({"completion_bar_ordinal": [4], "semantic_loop_id": [TARGET_C]})
    hidden = pd.DataFrame({"completion_bar_ordinal": [5], "hidden_family_class": [HIDDEN_A]})
    assert precursor_present(
        completion_bar_ordinal=10,
        lookback_bars=6,
        precursor_kind="registered",
        precursor_identity=TARGET_C,
        registered_events=registered,
        hidden_events=hidden,
    )
    assert precursor_present(
        completion_bar_ordinal=10,
        lookback_bars=6,
        precursor_kind="hidden",
        precursor_identity=HIDDEN_A,
        registered_events=registered,
        hidden_events=hidden,
    )


def test_matched_pseudo_completion_excludes_same_target() -> None:
    observed = deduplicate_registered_completions(_completion_rows()).iloc[[0]]
    eligible = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "session": ["2024-01-03", "2024-01-04"],
            "year_month": ["2024-01", "2024-01"],
            "clock_bin": [observed.iloc[0]["clock_bin"]] * 2,
            "completion_bar_ordinal": [6, 6],
            "completion_timestamp_utc": pd.to_datetime(
                ["2024-01-03T15:00:00Z", "2024-01-04T15:00:00Z"]
            ),
            "full_prior_history": [True, True],
            "semantic_loop_ids_at_timestamp": [[TARGET_A], [TARGET_B]],
        }
    )
    sampled = sample_matched_pseudo_completions(observed, eligible, seed=4)
    assert sampled.iloc[0]["session"] == "2024-01-04"


def test_bh_correction() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.002]) == pytest.approx([0.02, 0.04, 0.04, 0.008])


def test_frozen_candidate_threshold_reconstruction() -> None:
    values = pd.Series(np.arange(10, dtype=float) / 10)
    assert candidate_threshold(values) == pytest.approx(0.72)


def test_sequential_update_rows_and_stop_after_completion() -> None:
    rows = sequential_update_ordinals(
        opening_ordinal=5,
        first_completion_ordinal=8,
        available_ordinals=range(20),
    )
    assert rows == (5, 6, 7, 8)


def test_original_horizon_target() -> None:
    events = pd.DataFrame(
        {
            "completion_bar_ordinal": [9, 20],
            "semantic_loop_id": [TARGET_A, TARGET_B],
            "orientation_id": ["a", "b"],
        }
    )
    target, identity, ordinal = next_registered_route(
        events, update_ordinal=8, horizon_end_ordinal=17, retained_targets=[TARGET_A]
    )
    assert (target, identity, ordinal) == (TARGET_A, TARGET_A, 9)
    assert map_registered_route(TARGET_B, [TARGET_A]) == OTHER_REGISTERED_COMPLETION


def test_candidate_normalised_weighting() -> None:
    panel = pd.DataFrame(
        {"candidate_id": ["a", "a", "b"], "candidate_total_weight": [0.2, 0.2, 0.5]}
    )
    panel["weight"] = candidate_normalised_weights(panel)
    assert panel.groupby("candidate_id")["weight"].sum().to_dict() == pytest.approx(
        {"a": 0.2, "b": 0.5}
    )


def test_target_specific_active_prefix_depth() -> None:
    prefix = pd.DataFrame(
        {
            "bar_ordinal": [3, 4, 4],
            "semantic_loop_id": [TARGET_A, TARGET_A, TARGET_B],
            "orientation_id": ["canon", "canon", "other"],
            "progress_states": [1, 2, 1],
        }
    )
    snapshot = target_prefix_snapshot(
        prefix,
        current_ordinal=4,
        target_identity=TARGET_A,
        canonical_orientation_id="canon",
        transition_length=3,
    )
    assert snapshot["depth"] == 2
    assert snapshot["bars_since_first_active"] == 1
    assert snapshot["conflicting_prefix_active"] == 1


def test_prior_exact_registered_loop_indicators() -> None:
    events = pd.DataFrame(
        {"completion_bar_ordinal": [6, 9], "semantic_loop_id": [TARGET_A, TARGET_C]}
    )
    features = registered_history_features(events, opening_ordinal=5, current_ordinal=10)
    assert features["prior_target_a_within_6"] == 1
    assert features["prior_target_c_within_6"] == 1
    assert features["bars_since_latest_registered_completion"] == 1


def test_hidden_family_history_indicators() -> None:
    events = pd.DataFrame(
        {"completion_bar_ordinal": [7, 9], "hidden_family_class": [HIDDEN_A, HIDDEN_B]}
    )
    features = hidden_history_features(events, opening_ordinal=5, current_ordinal=10)
    assert features["hidden_5_6_5_seen_since_opening"] == 1
    assert features["hidden_2_3_2_seen_since_opening"] == 1
    assert features["most_recent_hidden_family__hidden_2_3_2"] == 1


def test_c0_c1_c2_feature_construction_is_nested() -> None:
    features = model_feature_sets(PREREGISTERED_TARGETS)
    assert set(features["C0"]) < set(features["C1"]) < set(features["C2"])
    assert "hidden_5_6_5_seen_since_opening" in features["C2"]


def test_counterfactual_probability_contrast() -> None:
    frame = pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, -0.5, 0.5, 1.5],
            "h": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "next_registered_route": ["a", "b", "c", "a", "b", "c"],
            "sequential_row_weight": [1.0] * 6,
        }
    )
    model = fit_multinomial("C2", frame, feature_names=["x", "h"])
    treated = frame.loc[frame["h"].eq(1)]
    contrast = counterfactual_probability_difference(
        model, treated, zero_features=["h"], target_classes=["c"]
    )
    assert len(contrast) == 3
    assert predict_multinomial(model, frame).shape == (6, 3)


def test_matched_candidate_panel_controls() -> None:
    panel = pd.DataFrame(
        {
            "sequential_row_id": [f"r{i}" for i in range(7)],
            "treated": [1, 0, 0, 0, 0, 0, 0],
            "seen": [1, 0, 0, 0, 0, 0, 0],
            "stage": ["x"] * 7,
        }
    )
    relations = matched_control_relations(
        panel,
        treated_column="treated",
        untreated_history_column="seen",
        stratum_columns=["stage"],
    )
    assert len(relations) == 6
    assert relations["control_weight_within_treated"].sum() == pytest.approx(1.0)


def test_session_block_bootstrap() -> None:
    first = session_bootstrap_multiplicities(["a", "b", "c"], draws=25, seed=8)
    second = session_bootstrap_multiplicities(["a", "b", "c"], draws=25, seed=8)
    assert first == second
    assert all(sum(draw.values()) == 3 for draw in first)


def test_hidden_history_bundle_permutation() -> None:
    panel = pd.DataFrame(
        {
            "period": ["development"] * 3,
            "session": ["2024-01-02"] * 3,
            "checkpoint": [6] * 3,
            "elapsed": [1] * 3,
            "symbol": ["a", "b", "c"],
            "h1": [1, 2, 3],
            "h2": [10, 20, 30],
            "target": [0, 1, 0],
        }
    )
    permuted = permute_hidden_bundle(
        panel,
        bundle_columns=["h1", "h2"],
        group_columns=["period", "session", "checkpoint", "elapsed"],
        seed=2,
    )
    assert sorted(zip(permuted["h1"], permuted["h2"], strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]
    assert permuted["target"].tolist() == panel["target"].tolist()


def test_protected_date_rejection() -> None:
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def test_decision_logic() -> None:
    decision = choose_primary_decision(
        blocker=None,
        target_a_status="supported",
        target_b_status="not_supported",
        target_c_status="supported",
        diversion_status="not_supported",
        hidden_increment_status="descriptive_only",
    )
    assert decision == "target_specific_hidden_routes_and_registered_recurrence_supported"
    blocked = choose_primary_decision(
        blocker="blocked_sequential_model_support_failure",
        target_a_status="insufficient_support",
        target_b_status="insufficient_support",
        target_c_status="insufficient_support",
        diversion_status="insufficient_support",
        hidden_increment_status="insufficient_support",
    )
    assert blocked == "blocked_sequential_model_support_failure"
