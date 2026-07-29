from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.route_competition_hazard_v0 import (
    CHECKPOINTS,
    ROUTE_FEATURES,
    assign_frozen_quartile,
    assign_route_resolution_state,
    choose_primary_decision,
    completion_targets,
    freeze_route_thresholds,
    permute_route_bundle,
    reject_protected_dates,
    route_competition_features_from_ledger,
    route_increment_passes,
    session_bootstrap_multiplicities,
)


def test_checkpoint_construction_is_frozen() -> None:
    assert CHECKPOINTS == (6, 10, 14, 18, 22, 26, 30, 34)
    assert all(checkpoint % 2 == 0 for checkpoint in CHECKPOINTS)


def test_three_bar_completion_target_is_strictly_after_checkpoint() -> None:
    targets = completion_targets(checkpoint=10, completion_ordinals=[10, 11, 13, 14])
    assert targets == {
        "registered_completion_next_3_bars": 1,
        "registered_completion_next_1_bar": 1,
    }

    outside = completion_targets(checkpoint=10, completion_ordinals=[10, 14])
    assert outside == {
        "registered_completion_next_3_bars": 0,
        "registered_completion_next_1_bar": 0,
    }


def test_protected_date_rejection() -> None:
    reject_protected_dates(pd.DataFrame({"session": ["2025-08-22"]}))
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def _prefix_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bar_ordinal": 7,
                "semantic_loop_id": "A",
                "motif_type": "primitive",
                "orientation_id": "a",
                "progress_states": 2,
                "transitions_remaining": 2,
            },
            {
                "bar_ordinal": 7,
                "semantic_loop_id": "B",
                "motif_type": "primitive",
                "orientation_id": "b",
                "progress_states": 3,
                "transitions_remaining": 1,
            },
            {
                "bar_ordinal": 7,
                "semantic_loop_id": "C",
                "motif_type": "repeat",
                "orientation_id": "c",
                "progress_states": 2,
                "transitions_remaining": 1,
            },
            {
                "bar_ordinal": 9,
                "semantic_loop_id": "A",
                "motif_type": "primitive",
                "orientation_id": "a",
                "progress_states": 2,
                "transitions_remaining": 2,
            },
            {
                "bar_ordinal": 9,
                "semantic_loop_id": "D",
                "motif_type": "composite",
                "orientation_id": "d",
                "progress_states": 4,
                "transitions_remaining": 1,
            },
            {
                "bar_ordinal": 10,
                "semantic_loop_id": "A",
                "motif_type": "primitive",
                "orientation_id": "a",
                "progress_states": 2,
                "transitions_remaining": 2,
            },
            {
                "bar_ordinal": 10,
                "semantic_loop_id": "B",
                "motif_type": "primitive",
                "orientation_id": "b",
                "progress_states": 3,
                "transitions_remaining": 1,
            },
            {
                "bar_ordinal": 10,
                "semantic_loop_id": "C",
                "motif_type": "repeat",
                "orientation_id": "c",
                "progress_states": 2,
                "transitions_remaining": 1,
            },
            {
                "bar_ordinal": 11,
                "semantic_loop_id": "FUTURE",
                "motif_type": "primitive",
                "orientation_id": "future",
                "progress_states": 3,
                "transitions_remaining": 1,
            },
        ]
    )


def test_complete_route_competition_bundle() -> None:
    completions = pd.DataFrame(
        {
            "completion_bar_ordinal": [5, 0, 11],
            "semantic_loop_id": ["A", "C", "FUTURE"],
        }
    )
    features = route_competition_features_from_ledger(_prefix_ledger(), completions, checkpoint=10)

    assert tuple(features) == ROUTE_FEATURES
    assert features["active_prefix_count"] == 3.0
    assert features["active_prefix_family_count"] == 2.0
    assert features["top_prefix_depth_fraction"] == pytest.approx(2 / 3)
    assert features["second_prefix_depth_fraction"] == pytest.approx(1 / 2)
    assert features["top_minus_second_prefix_depth"] == pytest.approx(1 / 6)
    assert features["prefix_family_entropy"] == pytest.approx(0.6365141683)
    assert features["orientation_disagreement_fraction"] == pytest.approx(2 / 3)
    assert features["new_prefixes_last_1_bar"] == 2.0
    assert features["invalidated_prefixes_last_1_bar"] == 1.0
    assert features["active_prefix_count_change_last_1_bar"] == 1.0
    assert features["active_prefix_count_change_last_3_bars"] == 0.0
    assert features["top_prefix_depth_change_last_1_bar"] == pytest.approx(-1 / 12)
    assert features["top_prefix_depth_change_last_3_bars"] == pytest.approx(0.0)
    assert features["matching_recent_loop_prefix_count"] == 1.0
    assert features["recent_loop_memory_weighted_top_depth"] == pytest.approx(3 / 4)


def test_future_prefix_transition_and_completion_are_not_visible() -> None:
    completions = pd.DataFrame({"completion_bar_ordinal": [11], "semantic_loop_id": ["FUTURE"]})
    features = route_competition_features_from_ledger(_prefix_ledger(), completions, checkpoint=10)
    assert features["active_prefix_count"] == 3.0
    assert features["matching_recent_loop_prefix_count"] == 0.0


def test_orientation_disagreement_uses_cross_route_anchor_state() -> None:
    prefixes = pd.DataFrame(
        {
            "bar_ordinal": [10, 10, 10],
            "semantic_loop_id": ["A", "B", "C"],
            "motif_type": ["primitive", "repeat", "composite"],
            "orientation_id": ["A__o_2-5-2", "B__o_2-6-2", "C__o_6-4-6"],
            "progress_states": [2, 2, 2],
            "transitions_remaining": [1, 1, 1],
        }
    )
    completions = pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id"])
    features = route_competition_features_from_ledger(prefixes, completions, checkpoint=10)
    assert features["orientation_disagreement_fraction"] == pytest.approx(1 / 3)


def test_development_frozen_bins_and_route_resolution_labels() -> None:
    development = pd.DataFrame(
        {
            "top_prefix_depth_fraction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "top_minus_second_prefix_depth": [0.01, 0.02, 0.03, 0.04, 0.05],
            "prefix_family_entropy": [0.0, 0.2, 0.4, 0.6, 0.8],
        }
    )
    thresholds = freeze_route_thresholds(development)
    assert thresholds["top_prefix_depth_fraction"] == pytest.approx((0.2, 0.3, 0.4))
    assert assign_frozen_quartile(
        pd.Series([0.05, 0.25, 0.35, 0.9]),
        thresholds["top_prefix_depth_fraction"],
    ).tolist() == ["Q1", "Q2", "Q3", "Q4"]

    assessment = pd.DataFrame(
        {
            "active_prefix_count": [8, 7, 9, 2, 5],
            "active_prefix_count_change_last_3_bars": [0, -1, 0, 0, 1],
            "top_prefix_depth_fraction": [0.2, 0.3, 0.5, 0.1, 0.3],
            "top_minus_second_prefix_depth": [0.01, 0.03, 0.05, 0.02, 0.03],
            "prefix_family_entropy": [0.8, 0.3, 0.2, 0.0, 0.3],
            "depth_margin_change_last_3_bars": [0.0, 0.01, 0.0, 0.0, 0.0],
        }
    )
    labels = assign_route_resolution_state(assessment, thresholds)
    assert labels.tolist() == [
        "BROAD_CONFLICT",
        "NARROWING",
        "DOMINANT_ROUTE",
        "LOW_ROUTE_SUPPORT",
        "OTHER",
    ]


def test_session_bootstrap_and_route_bundle_permutation_are_group_safe() -> None:
    sessions = pd.Series(["s1", "s1", "s2", "s2"])
    draws = session_bootstrap_multiplicities(sessions, draws=3, seed=17)
    assert len(draws) == 3
    assert all(len(draw) == 4 for draw in draws)
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)
    assert any(draw.sum() == 4 for draw in draws)

    frame = pd.DataFrame(
        {
            "period": ["development"] * 4,
            "session": ["s1"] * 4,
            "checkpoint": [6] * 4,
            "symbol": ["A", "B", "C", "D"],
            "r1": [1, 2, 3, 4],
            "r2": [10, 20, 30, 40],
            "baseline": [100, 200, 300, 400],
        }
    )
    permuted = permute_route_bundle(
        frame,
        route_features=("r1", "r2"),
        strata=("period", "session", "checkpoint"),
        seed=4,
    )
    assert permuted["baseline"].tolist() == frame["baseline"].tolist()
    assert sorted(zip(permuted["r1"], permuted["r2"], strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
    ]


def test_decision_logic_is_fail_closed_and_requires_every_increment_gate() -> None:
    gates = {
        "log_loss_improvement": 0.01,
        "brier_improvement": 0.002,
        "auc_improvement": 0.0,
        "bootstrap_80_log_loss_lower": 0.0,
        "bootstrap_80_brier_lower": 0.0,
        "positive_months": 5,
        "materially_adverse_checkpoints": 0,
        "real_exceeds_all_nulls": True,
        "concentration_passed": True,
    }
    assert route_increment_passes(gates)
    assert (
        choose_primary_decision(
            blocker=None,
            h1_passed=True,
            route_narrowing_ordered=False,
            h0_meaningful=False,
        )
        == "route_competition_improves_completion_hazard"
    )

    adverse = dict(gates, brier_improvement=-0.001)
    assert not route_increment_passes(adverse)
    assert (
        choose_primary_decision(
            blocker=None,
            h1_passed=False,
            route_narrowing_ordered=True,
            h0_meaningful=True,
        )
        == "descriptive_route_narrowing_only"
    )
    assert (
        choose_primary_decision(
            blocker="blocked_insufficient_support",
            h1_passed=True,
            route_narrowing_ordered=True,
            h0_meaningful=True,
        )
        == "blocked_insufficient_support"
    )
