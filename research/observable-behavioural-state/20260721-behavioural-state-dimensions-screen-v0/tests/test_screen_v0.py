from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.behavioural_state_dimensions_v0 import (
    CONJUNCTION_FEATURES,
    DESCRIPTIVE_LABELS,
    apply_component_scaling,
    apply_conjunction_bounds,
    assert_allowed_model_features,
    assert_safe_timestamps,
    assign_descriptive_labels,
    bar_component_frame,
    decide_behavioural_screen,
    derive_behavioural_dimensions,
    derive_conjunctions,
    derive_exhaustion_inputs,
    equal_slate_weights,
    fit_component_scaling,
    fit_conjunction_bounds,
    fit_fixed_logistic,
    fit_label_thresholds,
    manual_logistic_prediction,
    opening_raw_components,
    permute_bundle_within_slates,
    session_block_bootstrap_draws,
)


def test_frozen_predecessor_probabilities_reconstruct_exactly() -> None:
    repository = Path(__file__).resolve().parents[4]
    artifacts = (
        repository
        / "research"
        / "opening-regime-path"
        / "20260720-opening-regime-path-direction-screen-v0"
        / "artifacts"
        / "primary"
    )
    panel = pd.read_parquet(artifacts / "opening_decision_panel.parquet")
    assessment = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    archived = pd.read_parquet(artifacts / "assessment_predictions.parquet").sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    models = json.loads((artifacts / "model_coefficients.json").read_text(encoding="utf-8"))[
        "models"
    ]
    for target in ("large_remaining_move", "up_given_large_move"):
        actual = manual_logistic_prediction(models[target]["M1"], assessment)
        np.testing.assert_allclose(
            actual,
            archived[f"p__{target}__M1"],
            rtol=0.0,
            atol=1e-12,
        )


def test_opening_components_translate_price_and_activity_without_future_rows() -> None:
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "historical_relative_activity": [1.0, 3.0],
        }
    )

    calculated = bar_component_frame(bars)
    expected_returns = np.asarray([100.0, 10_000.0 * (102.0 / 101.0 - 1.0)])
    expected_ranges = np.asarray([300.0, 10_000.0 * 3.0 / 101.0])
    np.testing.assert_allclose(calculated["return_bps"], expected_returns)
    np.testing.assert_allclose(calculated["true_range_bps"], expected_ranges)
    np.testing.assert_allclose(calculated["close_location"], [2.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(calculated["upper_wick_fraction"], [1.0 / 3.0] * 2)
    np.testing.assert_allclose(calculated["lower_wick_fraction"], [1.0 / 3.0] * 2)

    components = opening_raw_components(
        calculated,
        trailing_opening_range_median_bps=200.0,
        signed_progress_bps=125.0,
        signed_progress_acceleration_bps=20.0,
        return_gap_bps=125.0,
    )
    assert components["activity_effort"] == pytest.approx(math.log1p(2.0))
    assert components["range_effort"] == pytest.approx(math.log1p(expected_ranges.sum()))
    assert components["travel_effort"] == pytest.approx(math.log1p(np.abs(expected_returns).sum()))
    assert components["signed_progress"] == 125.0
    assert components["absolute_progress"] == 125.0
    assert components["signed_efficiency"] == pytest.approx(1.0)
    assert components["absolute_efficiency"] == pytest.approx(1.0)
    assert components["close_retention"] == pytest.approx(1.0 / 3.0)
    assert components["directional_persistence"] == 1.0
    assert components["new_high_fraction"] == 1.0
    assert components["new_low_fraction"] == 0.5
    assert components["up_extreme_rejection"] == pytest.approx(1.0 / 3.0)
    assert components["down_extreme_rejection"] == pytest.approx(1.0 / 3.0)
    assert components["extreme_rejection"] == pytest.approx(1.0 / 3.0)
    assert components["compression"] == pytest.approx(-math.log(2.0))
    assert components["normalised_high_slope"] == pytest.approx(0.25)
    assert components["normalised_low_slope"] == pytest.approx(0.25)
    assert components["boundary_slope"] == pytest.approx(0.25)
    assert components["activity_acceleration"] == pytest.approx(math.log(4.0) - math.log(2.0))
    assert components["range_acceleration"] == pytest.approx(
        expected_ranges[1] - expected_ranges[0]
    )
    assert components["effort_acceleration"] == pytest.approx(
        0.5 * (math.log(4.0) - math.log(2.0) + expected_ranges[1] - expected_ranges[0])
    )
    assert components["signed_progress_acceleration"] == 20.0
    assert components["return_gap"] == 125.0
    assert components["mean_close_location"] == pytest.approx(2.0 / 3.0)


def test_development_scaling_and_fixed_dimensions_follow_preregistered_equations() -> None:
    development = pd.DataFrame(
        {"decision_ordinal": [6, 6, 6, 6], "component": [0.0, 2.0, 4.0, 6.0]}
    )
    scaling = fit_component_scaling(development, components=("component",))
    assert scaling[6]["component"].center == 3.0
    assert scaling[6]["component"].scale == 3.0
    transformed = apply_component_scaling(
        pd.DataFrame({"decision_ordinal": [6, 6, 6], "component": [-100.0, 3.0, 100.0]}),
        scaling,
        components=("component",),
    )
    np.testing.assert_allclose(transformed["z_component"], [-5.0, 0.0, 5.0])

    raw = pd.DataFrame(
        {
            "signed_pressure": [1.0, -2.0, 0.0],
            "signed_progress_acceleration": [3.0, 4.0, 5.0],
            "up_extreme_rejection": [0.1, 0.2, 0.3],
            "down_extreme_rejection": [0.4, 0.5, 0.6],
        }
    )
    exhaustion_inputs = derive_exhaustion_inputs(raw)
    np.testing.assert_allclose(exhaustion_inputs["aligned_progress_acceleration"], [3.0, -4.0, 0.0])
    np.testing.assert_allclose(exhaustion_inputs["directional_rejection"], [0.1, 0.5, 0.45])

    standardised = pd.DataFrame(
        {
            "z_activity_effort": [3.0],
            "z_range_effort": [0.0],
            "z_travel_effort": [0.0],
            "z_absolute_efficiency": [3.0],
            "z_close_retention": [0.0],
            "z_directional_persistence": [0.0],
            "z_extreme_rejection": [0.0],
            "z_absolute_progress": [0.0],
            "z_compression": [0.0],
            "z_signed_progress": [4.0],
            "z_signed_efficiency": [0.0],
            "z_mean_close_location": [0.0],
            "z_boundary_slope": [0.0],
            "z_effort_acceleration": [2.0],
            "z_aligned_progress_acceleration": [0.5],
            "z_directional_rejection": [0.25],
            "z_return_gap": [-3.0],
            "z_activity_gap": [0.0],
            "z_range_gap": [0.0],
            "return_gap": [-10.0],
        }
    )
    dimensions = derive_behavioural_dimensions(standardised)
    assert dimensions.loc[0, "arousal"] == 1.0
    assert dimensions.loc[0, "conviction"] == 1.0
    assert dimensions.loc[0, "frustration"] == -0.5
    assert dimensions.loc[0, "tension"] == 1.0
    assert dimensions.loc[0, "signed_pressure"] == 1.0
    assert dimensions.loc[0, "pressure_magnitude"] == 1.0
    assert dimensions.loc[0, "exhaustion_magnitude"] == 1.75
    assert dimensions.loc[0, "signed_exhaustion"] == 1.75
    assert dimensions.loc[0, "independence"] == 1.0
    assert dimensions.loc[0, "signed_independence"] == -1.0

    conjunctions = derive_conjunctions(dimensions)
    assert conjunctions.loc[0, "active_conviction"] == 1.0
    assert conjunctions.loc[0, "active_frustration"] == -0.5
    assert conjunctions.loc[0, "pressurised_tension"] == 1.0
    assert conjunctions.loc[0, "pressurised_exhaustion"] == 1.75
    assert conjunctions.loc[0, "independent_pressure"] == 1.0


def test_conjunction_clipping_and_descriptive_labels_are_development_frozen() -> None:
    development = pd.DataFrame(
        {
            "decision_ordinal": [6] * 10,
            **{feature: np.arange(10, dtype=float) for feature in CONJUNCTION_FEATURES},
            "arousal": np.arange(10, dtype=float),
            "conviction": np.arange(10, dtype=float),
            "frustration": np.arange(10, dtype=float),
            "tension": np.arange(10, dtype=float),
            "signed_pressure": np.arange(10, dtype=float),
            "exhaustion_magnitude": np.arange(10, dtype=float),
            "independence": np.arange(10, dtype=float),
        }
    )
    bounds = fit_conjunction_bounds(development)
    assert bounds[6]["active_conviction"] == pytest.approx((0.09, 8.91))
    clipped = apply_conjunction_bounds(
        pd.DataFrame(
            {
                "decision_ordinal": [6, 6],
                **{feature: [-100.0, 100.0] for feature in CONJUNCTION_FEATURES},
            }
        ),
        bounds,
    )
    np.testing.assert_allclose(clipped["active_conviction"], [0.09, 8.91])

    thresholds = fit_label_thresholds(development)
    assessment = pd.DataFrame(
        {
            "decision_ordinal": [6, 6],
            "arousal": [2.0, 8.0],
            "conviction": [6.0, 6.0],
            "frustration": [7.0, 1.0],
            "tension": [7.0, 1.0],
            "signed_pressure": [-2.0, 7.0],
            "exhaustion_magnitude": [7.0, 7.0],
            "independence": [7.0, 1.0],
        }
    )
    labelled = assign_descriptive_labels(assessment, thresholds)
    assert labelled.loc[0, "label__CALM"]
    assert labelled.loc[0, "label__TENSE"]
    assert labelled.loc[0, "label__CONFLICTED"]
    assert labelled.loc[0, "label__BEARISH_PRESSURE"]
    assert labelled.loc[0, "label__DOWNWARD_PRESSURE_EXHAUSTING"]
    assert labelled.loc[0, "label__INDEPENDENT"]
    assert labelled.loc[1, "label__BULLISH_PRESSURE"]
    assert labelled.loc[1, "label__UPWARD_PRESSURE_EXHAUSTING"]
    assert set(DESCRIPTIVE_LABELS).issubset(
        {column.removeprefix("label__") for column in labelled if column.startswith("label__")}
    )


def test_weighting_prediction_bootstrap_null_safety_and_decision_utilities() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a", "a", "b", "b", "b"],
            "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "target": [0, 0, 0, 1, 1],
        }
    )
    weights = equal_slate_weights(frame["slate_id"])
    np.testing.assert_allclose(weights, [0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    model = fit_fixed_logistic(
        frame,
        frame["target"],
        features=("x",),
        slate_column="slate_id",
        model_id="worked_example",
    )
    np.testing.assert_allclose(
        model.predict(frame), manual_logistic_prediction(model.as_dict(), frame)
    )

    sessions = pd.Series(["s1", "s1", "s2", "s2"])
    draws = session_block_bootstrap_draws(sessions, draws=5, seed=7)
    assert len(draws) == 5
    assert all(len(draw.row_indices) == 4 for draw in draws)
    assert all(sum(index in {0, 1} for index in draw.row_indices) in {0, 2, 4} for draw in draws)

    bundled = pd.DataFrame(
        {
            "slate_id": ["a", "a", "a", "b", "b", "b"],
            "dimension": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "paired": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "outcome": [0, 1, 0, 1, 0, 1],
        }
    )
    permuted = permute_bundle_within_slates(
        bundled,
        features=("dimension", "paired"),
        slate_column="slate_id",
        rng=np.random.default_rng(11),
    )
    assert permuted["outcome"].equals(bundled["outcome"])
    assert np.allclose(permuted["paired"], 10.0 * permuted["dimension"])
    for slate in ("a", "b"):
        before = bundled.loc[bundled["slate_id"].eq(slate), "dimension"].sort_values().tolist()
        after = permuted.loc[permuted["slate_id"].eq(slate), "dimension"].sort_values().tolist()
        assert before == after

    assert_safe_timestamps(pd.Series(pd.to_datetime(["2025-08-22T23:59:59Z"], utc=True)))
    with pytest.raises(ValueError, match="protected"):
        assert_safe_timestamps(pd.Series(pd.to_datetime(["2025-08-23T00:00:00Z"], utc=True)))
    assert_allowed_model_features(("arousal", "historical_activity_proxy_shock"))
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_model_features(("future_return",))
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_model_features(("behavioural_label_calm",))

    assert (
        decide_behavioural_screen(
            movement_passes=True,
            direction_passes=False,
            conjunction_passes=False,
            descriptive_differences=False,
        )
        == "behavioural_dimensions_add_movement_only"
    )
    assert (
        decide_behavioural_screen(
            movement_passes=False,
            direction_passes=False,
            conjunction_passes=False,
            descriptive_differences=True,
        )
        == "behavioural_descriptions_only_no_predictive_increment"
    )
