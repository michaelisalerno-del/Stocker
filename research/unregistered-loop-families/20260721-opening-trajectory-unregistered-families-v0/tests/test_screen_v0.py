from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.opening_trajectory_unregistered_families_v0 import (
    binary_brier,
    binary_log_loss,
    binary_targets,
    canonical_unregistered_path,
    decide_screen,
    first_unregistered_path,
    hidden_family_census,
    multiclass_brier,
    multiclass_log_loss,
    opening_anchor_triplet,
    opening_population,
    permute_group_within_slates,
    permute_trajectory_bundle_within_slates,
    pool_hidden_family,
    reject_protected_dates,
    select_hidden_families,
    session_block_bootstrap_indices,
    trajectory_feature_names,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR = (
    EXPERIMENT_DIR.parents[1]
    / "behavioural-trajectory"
    / "20260721-behavioural-trajectory-late-loops-v01"
    / "artifacts"
    / "primary"
)


@pytest.mark.parametrize(("checkpoint", "expected"), [(6, (2, 4, 6)), (12, (4, 8, 12))])
def test_corrected_opening_trajectory_anchors(
    checkpoint: int, expected: tuple[int, int, int]
) -> None:
    assert opening_anchor_triplet(checkpoint) == expected


def test_frozen_trajectory_surface_contains_exactly_eighteen_fields() -> None:
    features = trajectory_feature_names()

    assert len(features) == 18
    assert features[:3] == ("arousal_change", "arousal_acceleration", "arousal_reversal")
    assert features[-3:] == (
        "signed_exhaustion_change",
        "signed_exhaustion_acceleration",
        "signed_exhaustion_reversal",
    )


def test_unregistered_and_registered_binary_targets_are_disjoint() -> None:
    raw = pd.Series(
        [
            "UNREGISTERED_LOOP",
            "REGISTERED_COMPLETION",
            "NO_REGISTERED_COMPLETION",
            "TIED_REGISTERED_COMPLETION",
            "SOURCE_UNAVAILABLE",
        ]
    )

    targets = binary_targets(raw)

    np.testing.assert_allclose(
        targets["unregistered_event"], [1.0, 0.0, 0.0, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        targets["registered_completion"], [np.nan, 1.0, 0.0, np.nan, np.nan], equal_nan=True
    )


def test_six_bar_first_event_path_is_the_earliest_unregistered_completion() -> None:
    event = first_unregistered_path(
        bar_states=(0, 0, 1, 1, 2, 0, 3, 3),
        bar_ordinals=tuple(range(8)),
        decision_bar_ordinal=2,
        decision_event_index=1,
        registered_paths=frozenset({(0, 1, 0)}),
        horizon_bars=6,
    )

    assert event is not None
    assert event.full_path == (0, 1, 2, 0)
    assert event.start_event_index == 0
    assert event.completion_event_index == 3
    assert event.completion_bar_ordinal == 5


def test_six_bar_path_does_not_use_a_completion_after_the_horizon() -> None:
    event = first_unregistered_path(
        bar_states=(0, 1, 2, 3, 4, 5, 6, 0),
        bar_ordinals=tuple(range(8)),
        decision_bar_ordinal=0,
        decision_event_index=0,
        registered_paths=frozenset(),
        horizon_bars=6,
    )

    assert event is None


def test_opening_population_reconstructs_frozen_predecessor_keys() -> None:
    predecessor = pd.read_parquet(PREDECESSOR / "decision_panel.parquet")

    population = opening_population(predecessor)

    assert len(population) == 15_549
    assert population["session"].min() == "2024-01-17"
    assert population["session"].max() == "2025-08-22"
    assert population.groupby("decision_ordinal").size().to_dict() == {6: 7_775, 12: 7_774}
    assert not population.duplicated(["symbol", "session", "decision_ordinal"]).any()


def test_canonical_identity_is_forward_rotation_invariant() -> None:
    first = canonical_unregistered_path((2, 3, 4, 2))
    rotated = canonical_unregistered_path((3, 4, 2, 3))

    assert first.family_id == rotated.family_id
    assert first.canonical_path == (2, 3, 4, 2)
    assert first.oriented_path != rotated.oriented_path
    assert first.orientation_id != rotated.orientation_id


def test_reverse_orientation_remains_a_separate_family() -> None:
    forward = canonical_unregistered_path((2, 3, 4, 2))
    reverse = canonical_unregistered_path((2, 4, 3, 2))

    assert forward.family_id != reverse.family_id
    assert forward.reverse_family_id == reverse.family_id
    assert reverse.reverse_family_id == forward.family_id
    assert forward.reverse_orientation_equivalent is False


def test_repeat_depth_is_part_of_canonical_identity() -> None:
    primitive = canonical_unregistered_path((0, 1, 0))
    repeated = canonical_unregistered_path((0, 1, 0, 1, 0))

    assert primitive.motif_type == "primitive_like"
    assert primitive.repeat_depth == 1
    assert repeated.motif_type == "repeat_like"
    assert repeated.repeat_depth == 2
    assert primitive.family_id != repeated.family_id


def _family_rows(
    family_id: str, outcomes: int, *, stocks: int = 10, sessions: int = 25, months: int = 5
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family_id": family_id,
            "session": [
                f"2024-{index % months + 1:02d}-{index % 27 + 1:02d}" for index in range(outcomes)
            ],
            "symbol": [f"S{index % stocks:02d}" for index in range(outcomes)],
            "year_month": [f"2024-{index % months + 1:02d}" for index in range(outcomes)],
            "session_bucket": [index % sessions for index in range(outcomes)],
        }
    ).assign(
        session=lambda frame: frame["session_bucket"].map(
            lambda value: f"2024-{value % months + 1:02d}-{value + 1:02d}"
        )
    )


def test_development_only_hidden_family_selection_uses_support_then_stable_ties() -> None:
    development = pd.concat(
        [
            _family_rows("family_b", 40),
            _family_rows("family_a", 40),
            _family_rows("family_c", 35),
            _family_rows("under_supported", 29),
        ],
        ignore_index=True,
    )

    census = hidden_family_census(development)
    selected = select_hidden_families(census, maximum=2)

    assert selected == ("family_a", "family_b")
    assert not bool(census.set_index("family_id").loc["under_supported", "eligible"])


def test_unselected_family_is_pooled_into_other() -> None:
    assert pool_hidden_family("family_a", ("family_a", "family_b")) == "family_a"
    assert pool_hidden_family("family_c", ("family_a", "family_b")) == "OTHER_UNREGISTERED_FAMILY"


def test_grouped_trajectory_permutation_moves_three_fields_as_one_bundle() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a"] * 4 + ["b"] * 4,
            "year": [2024] * 4 + [2025] * 4,
            "change": np.arange(8),
            "acceleration": np.arange(8) + 100,
            "reversal": np.arange(8) + 200,
            "untouched": np.arange(8) + 300,
        }
    )

    permuted = permute_group_within_slates(frame, ("change", "acceleration", "reversal"), seed=19)

    assert permuted["untouched"].equals(frame["untouched"])
    assert np.array_equal(permuted["acceleration"] - permuted["change"], np.full(8, 100))
    assert np.array_equal(permuted["reversal"] - permuted["change"], np.full(8, 200))
    for slate in ("a", "b"):
        original = set(frame.loc[frame["slate_id"].eq(slate), "change"])
        actual = set(permuted.loc[permuted["slate_id"].eq(slate), "change"])
        assert actual == original


def test_trajectory_null_permutation_is_separate_by_year_and_slate() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["same"] * 6,
            "year": [2024] * 3 + [2025] * 3,
            "first": [1, 2, 3, 11, 12, 13],
            "second": [101, 102, 103, 111, 112, 113],
        }
    )

    permuted = permute_trajectory_bundle_within_slates(frame, ("first", "second"), seed=7)

    assert set(permuted.loc[permuted["year"].eq(2024), "first"]) == {1, 2, 3}
    assert set(permuted.loc[permuted["year"].eq(2025), "first"]) == {11, 12, 13}
    assert np.array_equal(permuted["second"] - permuted["first"], np.full(6, 100))


def test_session_block_bootstrap_keeps_each_sampled_session_whole() -> None:
    frame = pd.DataFrame({"session": ["a", "a", "b", "b", "b"], "value": range(5)})

    draws = session_block_bootstrap_indices(frame, draws=4, seed=11)

    assert len(draws) == 4
    for indices in draws:
        counts = pd.Series(frame.iloc[indices]["session"]).value_counts()
        assert counts.get("a", 0) % 2 == 0
        assert counts.get("b", 0) % 3 == 0


def test_binary_brier_and_log_loss_match_worked_values() -> None:
    targets = np.asarray([1, 0])
    probabilities = np.asarray([0.8, 0.25])

    assert binary_brier(targets, probabilities) == pytest.approx((0.2**2 + 0.25**2) / 2)
    assert binary_log_loss(targets, probabilities) == pytest.approx(
        (-np.log(0.8) - np.log(0.75)) / 2
    )


def test_multiclass_family_brier_and_log_loss_match_worked_values() -> None:
    targets = np.asarray([0, 2])
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    expected_brier = ((0.3**2 + 0.2**2 + 0.1**2) + (0.1**2 + 0.3**2 + 0.4**2)) / 2
    assert multiclass_brier(targets, probabilities) == pytest.approx(expected_brier)
    assert multiclass_log_loss(targets, probabilities) == pytest.approx(
        (-np.log(0.7) - np.log(0.6)) / 2
    )


def test_protected_date_rejection_fails_closed() -> None:
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


@pytest.mark.parametrize(
    ("stage_a", "stage_b", "point", "expected"),
    [
        (True, True, True, "opening_trajectories_predict_unregistered_events_and_families"),
        (True, False, True, "opening_trajectories_predict_unregistered_events_only"),
        (False, True, True, "opening_trajectories_predict_hidden_families_only"),
        (False, False, True, "opening_trajectory_signal_descriptive_only"),
        (False, False, False, "no_opening_trajectory_unregistered_increment"),
    ],
)
def test_primary_decision_logic(stage_a: bool, stage_b: bool, point: bool, expected: str) -> None:
    assert (
        decide_screen(
            stage_a_passes=stage_a,
            stage_b_passes=stage_b,
            point_estimate_improves=point,
        )
        == expected
    )


def test_materialised_opening_and_trajectory_reconstruction_pass_exactly() -> None:
    population = json.loads(
        (PRIMARY / "opening_population_reconstruction.json").read_text(encoding="utf-8")
    )
    trajectories = json.loads(
        (PRIMARY / "trajectory_feature_reconstruction.json").read_text(encoding="utf-8")
    )

    assert population["opening_rows"] == 15_549
    assert population["protected_rows_materialised"] == 0
    assert population["maximum_row_weight_difference"] <= 1e-15
    assert trajectories["anchors"] == {"6": [2, 4, 6], "12": [4, 8, 12]}
    assert trajectories["maximum_feature_difference"] <= 1e-12


def test_materialised_structural_census_preserves_tie_exclusions() -> None:
    census = pd.read_csv(PRIMARY / "structural_outcome_census.csv")
    tied = census.loc[census["raw_outcome"].eq("TIED_REGISTERED_COMPLETION")]

    assert set(census["raw_outcome"]) == {
        "REGISTERED_COMPLETION",
        "UNREGISTERED_LOOP",
        "NO_REGISTERED_COMPLETION",
        "TIED_REGISTERED_COMPLETION",
        "SOURCE_UNAVAILABLE",
    }
    assert tied["rows"].sum() == 4
    assert tied["excluded_from_primary"].all()
    unavailable = census.loc[census["raw_outcome"].eq("SOURCE_UNAVAILABLE")]
    assert len(unavailable) == 4
    assert unavailable["rows"].sum() == 0
    assert unavailable["excluded_from_primary"].all()


def test_materialised_hidden_family_mapping_was_frozen_on_development() -> None:
    mapping = json.loads((PRIMARY / "hidden_family_mapping.json").read_text(encoding="utf-8"))

    assert mapping["fit_period"] == "2024_only"
    assert mapping["frozen_before_assessment_family_support"] is True
    assert 2 <= len(mapping["selected_families"]) <= 4


def test_materialised_determinism_and_independent_audit_pass() -> None:
    determinism = json.loads((PRIMARY / "determinism_check.json").read_text(encoding="utf-8"))
    audit = json.loads((PRIMARY / "lightweight_audit.json").read_text(encoding="utf-8"))

    assert determinism["passed"] is True
    assert determinism["maximum_probability_difference"] <= 1e-12
    assert audit["passed"] is True
    assert audit["independent_from_runner_helpers"] is True
