from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.behavioural_trajectory_late_loops_v01 import (
    build_trajectory_regime_interactions,
    causal_anchor_prefix,
    decide_quick_screen,
    late_loop_subgroup,
    map_six_bar_structural_target,
    permute_trajectory_bundle_within_slates,
    phase_label,
    reject_protected_dates,
    session_block_bootstrap_draws,
    structural_history_controls,
    trajectory_anchors,
    trajectory_feature_values,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (6, (2, 4, 6)),
        (12, (4, 8, 12)),
        (24, (8, 16, 24)),
        (36, (12, 24, 36)),
    ],
)
def test_each_checkpoint_uses_the_preregistered_even_anchor_triplet(
    checkpoint: int,
    expected: tuple[int, int, int],
) -> None:
    assert trajectory_anchors(checkpoint) == expected


def test_even_anchor_enforcement_rejects_an_odd_completed_bar_count() -> None:
    starts = pd.date_range("2025-01-02 14:30:00+00:00", periods=12, freq="5min")
    bars = pd.DataFrame(
        {
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
        }
    )

    with pytest.raises(ValueError, match="even"):
        causal_anchor_prefix(
            bars,
            completed_bar_count=9,
            decision_available_timestamp=pd.Timestamp("2025-01-02 15:30:00+00:00"),
        )


def test_earlier_anchor_contains_no_bar_completed_after_its_availability() -> None:
    starts = pd.date_range("2025-01-02 14:30:00+00:00", periods=12, freq="5min")
    bars = pd.DataFrame(
        {
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "value": range(12),
        }
    )

    prefix = causal_anchor_prefix(
        bars,
        completed_bar_count=4,
        decision_available_timestamp=pd.Timestamp("2025-01-02 15:30:00+00:00"),
    )

    assert prefix["value"].tolist() == [0, 1, 2, 3]
    assert prefix["bar_complete_timestamp"].max() == pd.Timestamp("2025-01-02 14:50:00+00:00")


def test_missing_anchor_is_rejected_instead_of_using_a_substitute() -> None:
    starts = pd.date_range("2025-01-02 14:30:00+00:00", periods=3, freq="5min")
    bars = pd.DataFrame(
        {
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
        }
    )

    with pytest.raises(ValueError, match="fewer bars"):
        causal_anchor_prefix(
            bars,
            completed_bar_count=4,
            decision_available_timestamp=pd.Timestamp("2025-01-02 15:30:00+00:00"),
        )


def test_emotion_change_is_final_minus_earliest() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["change"] == 1.0


def test_emotion_acceleration_is_the_difference_between_consecutive_changes() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["acceleration"] == -3.0


def test_emotion_reversal_requires_opposite_nonzero_changes() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["reversal"] == 1
    assert trajectory_feature_values(1.0, 1.0, 2.0)["reversal"] == 0
    assert trajectory_feature_values(1.0, 2.0, 3.0)["reversal"] == 0


def test_each_preregistered_trajectory_regime_interaction_is_exact() -> None:
    frame = pd.DataFrame(
        {
            "transition_probability": [0.2],
            "posterior_entropy": [0.4],
            "top_second_margin": [0.5],
            "top_state_probability": [0.6],
            "arousal_change": [2.0],
            "frustration_change": [-2.0],
            "conviction_change": [3.0],
            "signed_pressure_acceleration": [-3.0],
            "tension_acceleration": [4.0],
            "signed_exhaustion_change": [-4.0],
        }
    )

    interactions, bounds = build_trajectory_regime_interactions(frame)

    assert bounds == {}
    assert interactions.iloc[0].to_dict() == pytest.approx(
        {
            "transition_probability_x_arousal_change": 0.4,
            "posterior_entropy_x_frustration_change": -0.8,
            "top_second_margin_x_conviction_change": 1.5,
            "transition_probability_x_signed_pressure_acceleration": -0.6,
            "posterior_entropy_x_tension_acceleration": 1.6,
            "top_state_probability_x_signed_exhaustion_change": -2.4,
        }
    )


def test_prior_registered_completion_controls_use_only_known_completion_bars() -> None:
    controls = structural_history_controls(
        registered_completion_bar_ordinals=(3, 9, 14),
        decision_bar_ordinal=11,
        active_registered_prefix_count=2,
    )

    assert controls == {
        "registered_completion_count_before_decision": 2,
        "bars_since_last_registered_completion": 2.0,
        "bars_since_last_registered_completion_missing": 0,
        "active_registered_prefix_count_at_decision": 2,
    }


def test_prior_completion_missing_indicator_is_explicit() -> None:
    controls = structural_history_controls(
        registered_completion_bar_ordinals=(),
        decision_bar_ordinal=23,
        active_registered_prefix_count=0,
    )

    assert controls["registered_completion_count_before_decision"] == 0
    assert controls["bars_since_last_registered_completion"] == 0.0
    assert controls["bars_since_last_registered_completion_missing"] == 1


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [(6, "OPENING_PHASE"), (12, "OPENING_PHASE"), (24, "LATER_PHASE"), (36, "LATER_PHASE")],
)
def test_opening_and_later_phase_assignment(checkpoint: int, expected: str) -> None:
    assert phase_label(checkpoint) == expected


def test_late_no_open_loop_assignment_is_preregistered() -> None:
    assert late_loop_subgroup(24, opening_registered_completion_count=0) == (
        "LATE_NO_OPEN_REGISTERED_LOOP"
    )
    assert late_loop_subgroup(36, opening_registered_completion_count=1) == (
        "LATE_AFTER_OPEN_REGISTERED_LOOP"
    )
    assert late_loop_subgroup(12, opening_registered_completion_count=0) is None


@pytest.mark.parametrize(
    ("raw_outcome", "expected"),
    [
        ("REGISTERED_PRIMITIVE", "REGISTERED_COMPLETION"),
        ("REGISTERED_REPEAT", "REGISTERED_COMPLETION"),
        ("REGISTERED_COMPOSITE", "REGISTERED_COMPLETION"),
        ("UNREGISTERED_LOOP", "UNREGISTERED_LOOP"),
        ("NO_REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"),
        ("TIED_REGISTERED_COMPLETION", None),
        ("SOURCE_UNAVAILABLE", None),
    ],
)
def test_six_bar_target_uses_the_frozen_three_class_mapping(
    raw_outcome: str,
    expected: str | None,
) -> None:
    assert map_six_bar_structural_target(raw_outcome, horizon_bars=6) == expected


def test_structural_target_rejects_a_non_six_bar_horizon() -> None:
    with pytest.raises(ValueError, match="six"):
        map_six_bar_structural_target("UNREGISTERED_LOOP", horizon_bars=5)


def test_session_block_bootstrap_preserves_every_checkpoint_and_stock() -> None:
    frame = pd.DataFrame(
        {
            "session": ["A"] * 4 + ["B"] * 4,
            "decision_ordinal": [6, 12, 24, 36] * 2,
            "symbol": ["X"] * 8,
        }
    )

    draws = session_block_bootstrap_draws(frame, draws=3, seed=7)
    repeated = session_block_bootstrap_draws(frame, draws=3, seed=7)

    assert len(draws) == 3
    for draw, repeated_draw in zip(draws, repeated, strict=True):
        assert draw.sampled_sessions == repeated_draw.sampled_sessions
        assert np.array_equal(draw.row_indices, repeated_draw.row_indices)
        for offset, session in enumerate(draw.sampled_sessions):
            selected = frame.iloc[draw.row_indices[offset * 4 : (offset + 1) * 4]]
            assert selected["session"].eq(session).all()
            assert selected["decision_ordinal"].tolist() == [6, 12, 24, 36]


def test_within_slate_permutation_keeps_each_trajectory_bundle_intact() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["A", "A", "A", "B", "B", "B"],
            "symbol": ["X", "Y", "Z", "X", "Y", "Z"],
            "level": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "trajectory_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "trajectory_b": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
        }
    )

    permuted = permute_trajectory_bundle_within_slates(
        frame,
        features=("trajectory_a", "trajectory_b"),
        seed=17,
    )

    assert permuted[["slate_id", "symbol", "level"]].equals(frame[["slate_id", "symbol", "level"]])
    for slate_id in ("A", "B"):
        before = frame.loc[frame["slate_id"].eq(slate_id), ["trajectory_a", "trajectory_b"]]
        after = permuted.loc[permuted["slate_id"].eq(slate_id), ["trajectory_a", "trajectory_b"]]
        assert sorted(map(tuple, before.to_numpy())) == sorted(map(tuple, after.to_numpy()))


def test_protected_date_rejection_fails_closed() -> None:
    reject_protected_dates(pd.DataFrame({"session": ["2025-08-22"]}))
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def test_decision_logic_uses_preregistered_scope_and_main_effect_precedence() -> None:
    empty = {"pooled": False, "opening": False, "later": False, "late_no_open": False}
    assert (
        decide_quick_screen(
            t1_positive={**empty, "pooled": True, "later": True},
            t2_positive=empty,
            point_estimate_improves=True,
        )
        == "trajectory_main_effects_only"
    )
    assert (
        decide_quick_screen(
            t1_positive=empty,
            t2_positive={**empty, "pooled": True, "late_no_open": True},
            point_estimate_improves=True,
        )
        == "trajectory_signal_feasible_pooled_and_late"
    )
    assert (
        decide_quick_screen(
            t1_positive=empty,
            t2_positive={**empty, "opening": True},
            point_estimate_improves=True,
        )
        == "trajectory_signal_feasible_opening_only"
    )
    assert (
        decide_quick_screen(
            t1_positive=empty,
            t2_positive=empty,
            point_estimate_improves=True,
        )
        == "descriptive_trajectory_structure_only"
    )
    assert (
        decide_quick_screen(
            t1_positive=empty,
            t2_positive=empty,
            point_estimate_improves=False,
        )
        == "no_behavioural_trajectory_increment"
    )


def test_ordinal_6_and_12_final_values_reproduce_the_frozen_ledger() -> None:
    manifest = json.loads((PRIMARY / "checkpoint_anchor_manifest.json").read_text(encoding="utf-8"))
    reproduction = manifest["final_anchor_reproduction"]

    assert reproduction["rows_compared"] == 15_549
    assert reproduction["maximum_behavioural_level_difference"] <= 1e-12
    assert reproduction["maximum_scaling_parameter_difference"] <= 1e-12
    assert reproduction["passed"] is True


@pytest.mark.parametrize(
    "scale_group",
    ["checkpoint_24_anchor_24", "checkpoint_36_anchor_36"],
)
def test_new_later_final_anchors_have_development_only_scaling(scale_group: str) -> None:
    scaling = json.loads((PRIMARY / "trajectory_anchor_scaling.json").read_text(encoding="utf-8"))
    ledger = pd.read_parquet(PRIMARY / "trajectory_ledger.parquet")
    checkpoint, anchor = (int(value) for value in scale_group.split("_")[1::2])
    development = ledger.loc[
        ledger["year"].eq(2024) & ledger["scale_group"].eq(checkpoint * 100 + anchor)
    ]

    assert scaling["fit_interval"] == "2024-01-01_through_2024-12-31_only"
    assert scale_group in scaling["base_components"]
    assert scale_group in scaling["pressure_aligned_components"]
    assert not development.empty
    for family in ("base_components", "pressure_aligned_components"):
        for component, parameters in scaling[family][scale_group].items():
            values = development[component]
            expected_center = float(values.median())
            expected_scale = float(values.quantile(0.75) - values.quantile(0.25))
            if expected_scale <= 1e-12:
                expected_scale = 1.0
            expected_z = np.clip((values - expected_center) / expected_scale, -5.0, 5.0)

            assert float(parameters["center"]) == pytest.approx(expected_center, abs=1e-12)
            assert float(parameters["scale"]) == pytest.approx(expected_scale, abs=1e-12)
            np.testing.assert_allclose(
                development[f"z_{component}"], expected_z, rtol=0.0, atol=1e-12
            )
