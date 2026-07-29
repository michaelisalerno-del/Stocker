from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.route_competition_fixed_lead_v01 import (
    choose_fixed_lead_decision,
    earliest_completion_lead,
    fixed_lead_increment_passes,
    fixed_lead_labels,
    predecessor_surface_differences,
    prefix_proximity,
    remaining_required_transitions,
    theoretical_assessment_support,
)
from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES,
    H1_FEATURES,
    ROUTE_FEATURES,
    fit_hazard_model,
    permute_route_bundle,
    reject_protected_dates,
    session_bootstrap_multiplicities,
)


@pytest.mark.parametrize(
    ("completion_ordinals", "expected"),
    [
        ([10, 11, 13, 14], 1),
        ([12, 13], 2),
        ([13], 3),
        ([10, 14], 0),
        ([11, 11, 12], 1),
    ],
    ids=("lead-one", "lead-two", "lead-three", "lead-zero", "same-bar-tie"),
)
def test_earliest_completion_lead(completion_ordinals: list[int], expected: int) -> None:
    assert earliest_completion_lead(10, completion_ordinals) == expected


@pytest.mark.parametrize(
    ("motif_type", "path", "progress", "declared", "expected"),
    [
        ("primitive", [0, 1, 0], 2, 1, 1),
        ("repeat", [0, 1, 0, 1, 0], 3, 2, 2),
        ("composite", [1, 0, 1, 0, 1, 2, 1], 4, 3, 3),
    ],
)
def test_canonical_prefix_remaining_transitions(
    motif_type: str,
    path: list[int],
    progress: int,
    declared: int,
    expected: int,
) -> None:
    assert (
        remaining_required_transitions(
            progress_states=progress,
            canonical_oriented_path=path,
            motif_type=motif_type,
            declared_transitions_remaining=declared,
        )
        == expected
    )


def test_canonical_prefix_rejects_incorrect_declared_remainder() -> None:
    with pytest.raises(ValueError, match="differs"):
        remaining_required_transitions(
            progress_states=2,
            canonical_oriented_path=[0, 1, 0],
            motif_type="primitive",
            declared_transitions_remaining=2,
        )


def test_one_transition_away_prefix_detection() -> None:
    prefixes = pd.DataFrame(
        {
            "bar_ordinal": [10, 10, 10, 11],
            "semantic_loop_id": ["P", "R", "C", "FUTURE"],
            "orientation_id": ["p", "r", "c", "future"],
            "motif_type": ["primitive", "repeat", "composite", "primitive"],
            "progress_states": [3, 4, 5, 2],
            "transitions_remaining": [1, 3, 1, 1],
        }
    )
    paths = {
        ("P", "p"): [0, 1, 2, 0],
        ("R", "r"): [0, 1, 0, 1, 0, 1, 0],
        ("C", "c"): [1, 0, 1, 2, 0, 1],
    }
    assert prefix_proximity(
        prefixes,
        checkpoint=10,
        canonical_oriented_paths=paths,
    ) == {
        "any_prefix_one_transition_from_completion": 1,
        "minimum_remaining_transitions": 1.0,
        "number_of_one_transition_away_prefixes": 2,
    }


@pytest.mark.parametrize(
    ("lead", "near_complete", "expected_eligible", "expected_advance_target"),
    [
        (1, 0, 0, 0),
        (2, 1, 0, 1),
        (2, 0, 1, 1),
        (3, 0, 1, 1),
        (0, 0, 1, 0),
    ],
)
def test_fixed_lead_model_targets_and_advance_eligibility(
    lead: int,
    near_complete: int,
    expected_eligible: int,
    expected_advance_target: int,
) -> None:
    labels = fixed_lead_labels(
        first_completion_lead=lead,
        any_prefix_one_transition_from_completion=near_complete,
    )
    assert labels["completion_next_1_bar"] == int(lead == 1)
    assert labels["advance_eligible"] == expected_eligible
    assert labels["completion_in_bars_2_or_3"] == expected_advance_target


def test_predecessor_panel_and_frozen_feature_equivalence() -> None:
    reference = pd.DataFrame(
        {
            "row_id": ["A|2025-01-02|6", "B|2025-01-02|6"],
            "checkpoint_timestamp_utc": pd.to_datetime(
                ["2025-01-02T14:55:00Z", "2025-01-02T14:55:00Z"]
            ),
            "period": ["assessment", "assessment"],
            "row_weight": [0.5, 0.5],
            "registered_completion_next_3_bars": [0, 1],
            "H0_probability": [0.1, 0.4],
            "H1_probability": [0.2, 0.6],
            "f0": [1.0, 2.0],
            "r0": [3.0, 4.0],
        }
    )
    exact = predecessor_surface_differences(
        reference,
        reference.copy(),
        feature_columns=("f0", "r0"),
    )
    assert exact == {
        "row_identity_mismatches": 0,
        "checkpoint_timestamp_mismatches": 0,
        "split_mismatches": 0,
        "target_mismatches": 0,
        "maximum_weight_difference": 0.0,
        "maximum_feature_difference": 0.0,
        "maximum_probability_difference": 0.0,
    }

    changed = reference.copy()
    changed.loc[1, "r0"] += 0.25
    assert predecessor_surface_differences(
        reference,
        changed,
        feature_columns=("f0", "r0"),
    )["maximum_feature_difference"] == pytest.approx(0.25)


def test_corrected_theoretical_assessment_support() -> None:
    support = theoretical_assessment_support(
        sessions=160,
        stocks=20,
        checkpoints=8,
        retained_rows=25_518,
    )
    assert support["theoretical_eligible_rows"] == 25_600
    assert support["retained_rows"] == 25_518
    assert support["retention"] == pytest.approx(0.996796875)


def test_development_only_scaling_and_frozen_feature_order() -> None:
    rows = 8
    development = pd.DataFrame(
        {
            feature: np.linspace(index, index + 1, rows)
            for index, feature in enumerate(BASELINE_FEATURES)
        }
    )
    development["registered_completion_next_3_bars"] = [0, 1] * 4
    development["row_weight"] = 1.0
    model = fit_hazard_model(development, features=BASELINE_FEATURES)
    shifted_assessment = development.copy()
    shifted_assessment.loc[:, list(BASELINE_FEATURES)] += 1000.0
    model.predict_probability(shifted_assessment)
    assert tuple(model.features) == BASELINE_FEATURES
    assert tuple((*BASELINE_FEATURES, *ROUTE_FEATURES)) == H1_FEATURES
    assert np.allclose(model.scaler.mean_, development[list(BASELINE_FEATURES)].mean())


def test_session_bootstrap_and_route_bundle_permutation_preserve_slates() -> None:
    sessions = pd.Series(["s1", "s1", "s2", "s2"])
    draws = session_bootstrap_multiplicities(sessions, draws=15, seed=31)
    assert len(draws) == 15
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)

    frame = pd.DataFrame(
        {
            "period": ["development"] * 4,
            "session": ["s1"] * 4,
            "checkpoint": [6] * 4,
            "symbol": ["A", "B", "C", "D"],
            "route_a": [1, 2, 3, 4],
            "route_b": [10, 20, 30, 40],
            "baseline": [100, 200, 300, 400],
        }
    )
    permuted = permute_route_bundle(
        frame,
        route_features=("route_a", "route_b"),
        strata=("period", "session", "checkpoint"),
        seed=9,
    )
    assert permuted["baseline"].tolist() == frame["baseline"].tolist()
    assert sorted(zip(permuted.route_a, permuted.route_b, strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
    ]


def test_protected_date_rejection() -> None:
    reject_protected_dates(pd.DataFrame({"session": ["2025-08-22"]}))
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def _passing_gates() -> dict[str, object]:
    return {
        "log_loss_improvement": 0.01,
        "brier_improvement": 0.001,
        "auc_improvement": 0.0,
        "average_precision_improvement": 0.002,
        "bootstrap_80_log_loss_lower": 0.0,
        "bootstrap_80_brier_lower": 0.0,
        "bootstrap_80_average_precision_lower": 0.0,
        "positive_months": 5,
        "materially_adverse_checkpoints": 0,
        "real_exceeds_all_nulls": True,
        "support_and_concentration_passed": True,
    }


def test_decision_logic_distinguishes_immediate_from_advance_warning() -> None:
    gates = _passing_gates()
    assert fixed_lead_increment_passes(gates, require_average_precision=False)
    assert fixed_lead_increment_passes(gates, require_average_precision=True)
    assert (
        choose_fixed_lead_decision(
            blocker=None,
            immediate_passed=True,
            advance_passed=True,
            descriptive_lead_structure=True,
            baseline_meaningful=True,
        )
        == "route_competition_adds_immediate_and_advance_warning"
    )
    assert (
        choose_fixed_lead_decision(
            blocker=None,
            immediate_passed=True,
            advance_passed=False,
            descriptive_lead_structure=True,
            baseline_meaningful=True,
        )
        == "route_competition_is_imminent_confirmation_only"
    )
    assert (
        choose_fixed_lead_decision(
            blocker="blocked_insufficient_advance_positive_support",
            immediate_passed=True,
            advance_passed=True,
            descriptive_lead_structure=True,
            baseline_meaningful=True,
        )
        == "blocked_insufficient_advance_positive_support"
    )
