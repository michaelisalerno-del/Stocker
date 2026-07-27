from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from stocker_research.m1c_quiet_state_concentration_v0 import (
    BOTTOM_10_THRESHOLD,
    ORIGINAL_DECISION,
    analysis_weights,
    audit_claims,
    classify_month_concentration,
    classify_surprise_concentration,
    cluster_quiet_state_runs,
    cluster_surprise_events,
    fresh_quiet_episodes,
    reconstruct_frozen_tail,
    small_count_feasibility,
)


def _row(
    row_id: str,
    *,
    checkpoint: int,
    minute: int,
    probability: float,
    stock: str = "AAL",
    session: str = "2025-10-01",
    excursion: float = 0.5,
) -> dict[str, object]:
    timestamp = datetime(2025, 10, 1, 14, minute, tzinfo=UTC)
    return {
        "row_id": row_id,
        "stock": stock,
        "session": session,
        "month": session[:7],
        "checkpoint": checkpoint,
        "entry_timestamp": timestamp,
        "M1C_probability": probability,
        "m1c_bottom_10_percent": probability <= BOTTOM_10_THRESHOLD,
        "excursion_sigma_ratio_15m": excursion,
        "maximum_absolute_excursion_15m": excursion / 100.0,
        "row_weight": 1.0,
    }


def test_original_decision_and_claims_are_immutable() -> None:
    claims = audit_claims()

    assert claims["original_decision"] == ORIGINAL_DECISION
    assert claims["original_low_movement_decision_preserved"] is True
    assert claims["retrospective_gate_relaxation_allowed"] is False
    assert claims["m1c_bottom_10_threshold"] == pytest.approx(
        0.135896965695626,
        abs=0.0,
    )
    assert claims["paper_orders_allowed"] is False
    assert claims["live_orders_allowed"] is False
    assert claims["broker_order_methods_allowed"] is False


def test_frozen_tail_reconstruction_requires_exact_identity_probability_and_membership() -> None:
    predictions = pd.DataFrame(
        [
            _row("tail", checkpoint=6, minute=0, probability=0.1),
            _row("not-tail", checkpoint=8, minute=10, probability=0.2),
        ]
    )
    original = predictions.loc[
        predictions["m1c_bottom_10_percent"],
        ["row_id", "M1C_probability"],
    ]

    audit = reconstruct_frozen_tail(predictions, original)

    assert audit == {
        "row_identity_mismatches": 0,
        "maximum_m1c_probability_difference": 0.0,
        "tail_membership_mismatches": 0,
        "passed": True,
    }


def test_quiet_state_runs_continue_only_across_eligible_tail_gaps_at_most_15_minutes() -> None:
    rows = pd.DataFrame(
        [
            _row("a", checkpoint=6, minute=0, probability=0.10),
            _row("b", checkpoint=8, minute=10, probability=0.11),
            _row("break", checkpoint=10, minute=20, probability=0.20),
            _row("c", checkpoint=12, minute=30, probability=0.12),
            _row("d", checkpoint=14, minute=50, probability=0.09),
        ]
    )

    runs = cluster_quiet_state_runs(rows)

    assert runs["trigger_row_id"].tolist() == ["a", "c", "d"]
    assert runs["member_count"].tolist() == [2, 1, 1]
    assert runs["member_row_ids"].tolist() == [("a", "b"), ("c",), ("d",)]


def test_fresh_episode_identity_uses_downward_crossing_and_thirty_minute_spacing() -> None:
    rows = pd.DataFrame(
        [
            _row("first", checkpoint=6, minute=0, probability=0.10),
            _row("above", checkpoint=8, minute=10, probability=0.20),
            _row("too-soon", checkpoint=10, minute=20, probability=0.11),
            _row("above-again", checkpoint=12, minute=30, probability=0.19),
            _row("second", checkpoint=14, minute=40, probability=0.12),
        ]
    )

    episodes = fresh_quiet_episodes(rows)

    assert episodes["row_id"].tolist() == ["first", "second"]
    assert episodes["episode_number"].tolist() == [1, 2]
    assert pd.isna(episodes.iloc[0]["minutes_since_previous_quiet_episode"])
    assert episodes.iloc[1]["minutes_since_previous_quiet_episode"] == 40.0


def test_surprise_rows_cluster_only_when_trigger_windows_overlap() -> None:
    rows = pd.DataFrame(
        [
            _row("a", checkpoint=6, minute=0, probability=0.10, excursion=1.6),
            _row("b", checkpoint=8, minute=10, probability=0.11, excursion=1.8),
            _row("c", checkpoint=12, minute=30, probability=0.12, excursion=2.1),
        ]
    )

    events = cluster_surprise_events(rows, sigma_threshold=1.5)

    assert events["member_count"].tolist() == [2, 1]
    assert events["member_row_ids"].tolist() == [("a", "b"), ("c",)]
    assert events["extreme_surprise_mover"].tolist() == [False, True]


def test_small_count_concentration_reports_theoretical_minima_and_fragility() -> None:
    result = small_count_feasibility(
        month_counts={"2025-09": 2, "2025-10": 7, "2025-12": 2},
        stock_counts={"AAL": 3, "AAOI": 2, "APLD": 1, "IONQ": 1, "MRNA": 1, "MSTR": 1, "SOFI": 2},
        month_limit=0.60,
        stock_limit=0.50,
    )

    assert result["event_count"] == 11
    assert result["observed_maximum_month_share"] == pytest.approx(7 / 11)
    assert result["minimum_theoretical_maximum_month_share"] == pytest.approx(3 / 11)
    assert result["minimum_theoretical_maximum_stock_share"] == pytest.approx(1 / 11)
    assert result["one_event_share"] == pytest.approx(1 / 11)
    assert result["events_over_month_limit"] == 1
    assert result["small_count_concentration_fragile"] is True


@pytest.mark.parametrize(
    ("scheme", "groups"),
    [
        ("equal_month", ("month",)),
        ("equal_stock", ("stock",)),
        ("equal_stock_month", ("stock", "month")),
    ],
)
def test_equal_exposure_weights_assign_equal_total_mass_to_each_group(
    scheme: str,
    groups: tuple[str, ...],
) -> None:
    frame = pd.DataFrame(
        [
            {**_row("a", checkpoint=6, minute=0, probability=0.10), "row_weight": 0.25},
            {**_row("b", checkpoint=8, minute=10, probability=0.10), "row_weight": 0.75},
            {
                **_row(
                    "c",
                    checkpoint=6,
                    minute=0,
                    probability=0.10,
                    stock="AAOI",
                    session="2025-11-03",
                ),
                "row_weight": 1.0,
            },
        ]
    )

    weights = analysis_weights(frame, scheme=scheme)
    grouped = frame.assign(weight=weights).groupby(list(groups), sort=True)["weight"].sum()

    assert grouped.max() == pytest.approx(grouped.min())
    assert weights.sum() == pytest.approx(1.0)


def test_month_explanation_recognises_exposure_incidence_and_persistence_mixture() -> None:
    explanation = classify_month_concentration(
        maximum_composition_share=529 / 1426,
        source_exposure_share=5043 / 18136,
        fresh_episode_share=179 / 541,
        frozen_limit=0.35,
    )

    assert explanation == "month_concentration_has_multiple_causes"


def test_surprise_explanation_prioritises_small_count_fragility_when_one_event_failed_gate() -> (
    None
):
    explanation = classify_surprise_concentration(
        clustered_maximum_stock_share=3 / 11,
        clustered_maximum_month_share=7 / 11,
        clustered_maximum_stock_month_share=2 / 11,
        stock_limit=0.50,
        month_limit=0.60,
        small_count_fragile=True,
    )

    assert explanation == "surprise_concentration_is_small_count_fragile"


def test_run_gap_uses_elapsed_time_not_checkpoint_number() -> None:
    first = _row("a", checkpoint=6, minute=0, probability=0.10)
    second = _row("b", checkpoint=34, minute=10, probability=0.10)
    second["entry_timestamp"] = first["entry_timestamp"] + timedelta(minutes=10)

    runs = cluster_quiet_state_runs(pd.DataFrame([first, second]))

    assert len(runs) == 1
    assert runs.iloc[0]["member_count"] == 2
