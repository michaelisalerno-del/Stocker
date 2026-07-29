from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from stocker_research.pressure_onset_screen_v0 import (
    annotate_economic_selection_semantics,
    apply_support_contract_repair,
    assert_allowed_feature_names,
    assert_safe_timestamps,
    concentration_aware_decision,
    decide_pressure_screen,
    fit_fixed_logistic,
    largest_admitted_stock,
    manual_logistic_prediction,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
V0_ROOT = EXPERIMENT_ROOT.parent / "20260720-high-movement-pressure-onset-screen-v0"
V0_PRIMARY = V0_ROOT / "artifacts" / "primary"
V0_1_PRIMARY = EXPERIMENT_ROOT / "artifacts" / "primary"
STATIC_CONFIRMATION_FEATURES = (
    "change_cohort_relative_return_bps",
    "change_relative_strength_acceleration",
    "change_activity_shock",
    "change_range_acceleration",
    "change_signed_efficiency_3",
    "change_close_location",
    "new_high_at_t_plus_1",
    "new_low_at_t_plus_1",
    "opening_range_acceptance_persisted",
)


def _slate(*, size: int, admitted: int, slate_id: str = "2025-01-02|06") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slate_id": [slate_id] * size,
            "session": [slate_id.split("|")[0]] * size,
            "decision_ordinal": [int(slate_id.split("|")[1])] * size,
            "symbol": [f"S{index:02d}" for index in range(size)],
            "high_movement_admitted": [index < admitted for index in range(size)],
        }
    )


def test_parent_slate_with_twenty_valid_stocks_and_one_admission_is_valid() -> None:
    repaired = apply_support_contract_repair(_slate(size=20, admitted=1))

    assert len(repaired.primary_rows) == 1
    assert repaired.parent_slates.iloc[0]["parent_valid_stock_count"] == 20
    assert repaired.admitted_slates.iloc[0]["admitted_stock_count"] == 1
    assert repaired.admitted_slates.iloc[0]["support_status"] == "valid_singleton_admission"


def test_parent_slate_with_fourteen_valid_stocks_and_ten_admissions_is_invalid() -> None:
    repaired = apply_support_contract_repair(_slate(size=14, admitted=10))

    assert repaired.primary_rows.empty
    assert (
        repaired.parent_slates.iloc[0]["support_status"] == "parent_slate_insufficient_valid_stocks"
    )


def test_parent_slate_with_fifteen_valid_stocks_and_zero_admissions_has_no_primary_rows() -> None:
    repaired = apply_support_contract_repair(_slate(size=15, admitted=0))

    assert repaired.primary_rows.empty
    assert repaired.admitted_slates.iloc[0]["support_status"] == "no_high_movement_admission"


def test_admitted_slate_weights_use_admitted_count_and_sum_to_one() -> None:
    frame = pd.concat(
        [
            _slate(size=20, admitted=1, slate_id="2025-01-02|06"),
            _slate(size=20, admitted=2, slate_id="2025-01-02|12"),
            _slate(size=20, admitted=5, slate_id="2025-01-03|06"),
        ],
        ignore_index=True,
    )
    repaired = apply_support_contract_repair(frame)

    weights = repaired.primary_rows.groupby("parent_slate_id", sort=True)["row_weight"]
    assert weights.get_group("2025-01-02|06").tolist() == [1.0]
    assert weights.get_group("2025-01-02|12").tolist() == [0.5, 0.5]
    assert weights.get_group("2025-01-03|06").tolist() == [0.2] * 5
    assert weights.sum().eq(1.0).all()


def test_singleton_is_retained_in_occurrence_and_conditional_direction_matrices() -> None:
    frame = _slate(size=20, admitted=1)
    frame["directional_onset"] = [1, *([0] * 19)]
    repaired = apply_support_contract_repair(frame)

    occurrence = repaired.primary_rows
    direction = occurrence.loc[occurrence["directional_onset"].eq(1)]
    assert len(occurrence) == 1
    assert len(direction) == 1


def test_fixed_logistic_uses_precomputed_admitted_slate_weights() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a", "a", "b", "b"],
            "x": [-2.0, -1.0, 1.0, 2.0],
            "row_weight": [1.0, 0.25, 0.25, 1.0],
        }
    )
    labels = pd.Series([0, 1, 0, 1])

    model = fit_fixed_logistic(
        frame,
        labels,
        features=["x"],
        slate_column="slate_id",
        model_id="weighted_worked_example",
        sample_weight_column="row_weight",
    )
    values = frame[["x"]].to_numpy(dtype=float)
    standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    expected = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    ).fit(standardized, labels, sample_weight=frame["row_weight"])

    np.testing.assert_allclose(model.coefficients, expected.coef_[0], atol=1e-12)
    assert model.intercept == expected.intercept_[0]


def test_singleton_economic_candidate_is_retained_and_labelled_degenerate() -> None:
    selections = pd.DataFrame(
        {
            "candidate": ["pressure"],
            "slate_id": ["2025-01-02|06"],
            "symbol": ["ONLY"],
        }
    )

    annotated = annotate_economic_selection_semantics(selections, {"2025-01-02|06": 1})

    assert len(annotated) == 1
    assert annotated.iloc[0]["admitted_slate_type"] == "singleton"
    assert annotated.iloc[0]["within_admitted_comparison_status"] == "degenerate_singleton"


def test_largest_stock_concentration_does_not_remove_primary_rows() -> None:
    frame = pd.concat(
        [_slate(size=15, admitted=1, slate_id=f"2025-01-{day:02d}|06") for day in range(2, 14)],
        ignore_index=True,
    )
    frame.loc[frame["high_movement_admitted"], "symbol"] = [
        "HEAVY" if day < 11 else f"OTHER{day}" for day in range(2, 14)
    ]

    repaired = apply_support_contract_repair(frame)

    assert len(repaired.primary_rows) == 12
    assert largest_admitted_stock(repaired.primary_rows).share > 0.10


def test_concentration_can_prevent_an_otherwise_positive_decision() -> None:
    decision = concentration_aware_decision(
        "pressure_onset_and_direction_increment_observed",
        maximum_admitted_row_share=0.109,
        deletion_same_signed_conclusions=False,
        principal_increments_non_negative=True,
        no_material_adversity=True,
        economic_not_dominated=True,
    )

    assert decision == "pressure_signal_observed_but_concentration_gate_failed"


def test_concentration_stress_can_retain_the_frozen_positive_category() -> None:
    decision = concentration_aware_decision(
        "pressure_onset_occurrence_only",
        maximum_admitted_row_share=0.109,
        deletion_same_signed_conclusions=True,
        principal_increments_non_negative=True,
        no_material_adversity=True,
        economic_not_dominated=True,
    )

    assert decision == "pressure_onset_occurrence_only"


def test_largest_stock_deletion_is_selected_only_by_admitted_row_share() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["B", "A", "B", "C", "B", "A"],
            "outcome": [0, 1, 1, 0, 0, 1],
        }
    )

    leader = largest_admitted_stock(frame)

    assert leader.symbol == "B"
    assert leader.rows == 3
    assert leader.share == 0.5


def test_frozen_thresholds_and_barriers_are_byte_equivalent_to_v0() -> None:
    for name in ("movement_admission_thresholds.json", "onset_barriers.json"):
        v0 = json.loads((V0_PRIMARY / name).read_text(encoding="utf-8"))
        repaired = json.loads((V0_1_PRIMARY / name).read_text(encoding="utf-8"))
        assert repaired == v0
    thresholds = json.loads(
        (V0_1_PRIMARY / "movement_admission_thresholds.json").read_text(encoding="utf-8")
    )["thresholds"]
    barriers = json.loads((V0_1_PRIMARY / "onset_barriers.json").read_text(encoding="utf-8"))[
        "barriers_bps"
    ]
    assert thresholds == {"6": 0.30288693685048057, "12": 0.3003493391782974}
    assert barriers == {"6": 137.38384537217985, "12": 114.06303345896207}


def test_v0_and_v0_1_admissions_and_outcomes_are_identical() -> None:
    keys = ["session", "decision_ordinal", "symbol"]
    frozen = pd.read_parquet(V0_PRIMARY / "compact_decision_panel.parquet").sort_values(keys)
    repaired = pd.read_parquet(V0_1_PRIMARY / "compact_decision_panel.parquet").sort_values(keys)
    frozen = frozen.reset_index(drop=True)
    repaired = repaired.reset_index(drop=True)

    pd.testing.assert_frame_equal(
        repaired.loc[:, [*keys, "high_movement_admitted", "onset_label"]],
        frozen.loc[:, [*keys, "high_movement_admitted", "onset_label"]],
        check_exact=True,
    )


def test_nonadmitted_development_static_confirmation_values_remain_frozen() -> None:
    keys = ["session", "decision_ordinal", "symbol"]
    columns = [*keys, "year", "high_movement_admitted", *STATIC_CONFIRMATION_FEATURES]
    frozen = pd.read_parquet(V0_PRIMARY / "compact_decision_panel.parquet", columns=columns)
    repaired = pd.read_parquet(V0_1_PRIMARY / "compact_decision_panel.parquet", columns=columns)
    frozen = frozen.loc[frozen["year"].eq(2024) & ~frozen["high_movement_admitted"]]
    repaired = repaired.loc[repaired["year"].eq(2024) & ~repaired["high_movement_admitted"]]
    frozen = frozen.sort_values(keys, kind="mergesort").reset_index(drop=True)
    repaired = repaired.sort_values(keys, kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(repaired, frozen, check_exact=True, check_dtype=True)


def test_failed_assessment_parent_slates_are_reconstructed_as_thirteen_stocks() -> None:
    parent = pd.read_csv(V0_1_PRIMARY / "parent_slate_accounting.csv")
    failed = parent.loc[parent["year"].eq(2025) & ~parent["parent_slate_eligible"].astype(bool)]

    assert failed["parent_valid_stock_count"].tolist() == [13, 13]
    assert failed["history_complete_stock_count"].tolist() == [13, 13]
    assert failed["support_status"].eq("parent_slate_insufficient_valid_stocks").all()


def test_protected_dates_and_structural_features_fail_closed() -> None:
    with pytest.raises(ValueError, match="protected market row"):
        assert_safe_timestamps(["2025-08-23T13:30:00Z"])
    with pytest.raises(ValueError, match="forbidden feature"):
        assert_allowed_feature_names(["p_large_remaining_move", "regime_posterior"])


def test_repaired_primary_timestamps_remain_below_protected_boundary() -> None:
    panel = pd.read_parquet(
        V0_1_PRIMARY / "compact_decision_panel.parquet",
        columns=["decision_bar_start_timestamp_utc"],
    )
    assert_safe_timestamps(panel["decision_bar_start_timestamp_utc"])


def test_manual_logistic_prediction_reconstructs_serialized_equation() -> None:
    model = {
        "feature_names": ["x", "y"],
        "means": [1.0, -1.0],
        "scales": [2.0, 4.0],
        "coefficients": [0.5, -0.25],
        "intercept": 0.1,
    }
    frame = pd.DataFrame({"x": [1.0, 5.0], "y": [-1.0, 3.0]})

    actual = manual_logistic_prediction(model, frame)
    linear = np.array([0.1, 0.1 + 0.5 * 2.0 - 0.25 * 1.0])
    expected = 1.0 / (1.0 + np.exp(-linear))

    np.testing.assert_allclose(actual, expected, atol=1e-15)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            {"occurrence_passes": True, "direction_passes": True},
            "pressure_onset_and_direction_increment_observed",
        ),
        ({"occurrence_passes": True}, "pressure_onset_occurrence_only"),
        ({"direction_passes": True}, "directional_pressure_only"),
        ({"confirmation_occurrence_passes": True}, "one_bar_confirmation_required"),
        ({"readiness_useful": True}, "movement_readiness_but_direction_unresolved"),
        ({}, "no_pressure_onset_increment"),
        (
            {"integrity_blocker": "blocked_parent_slate_support_failure"},
            "blocked_parent_slate_support_failure",
        ),
    ],
)
def test_exact_frozen_decision_precedence(evidence: dict[str, bool | str], expected: str) -> None:
    assert decide_pressure_screen(evidence) == expected
