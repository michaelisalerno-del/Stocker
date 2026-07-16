from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.directed_economic_rotation import (
    activation_metrics,
    calibration_table,
    paired_model_comparison,
    shrink_pair_probability,
    system_activation_metrics,
)


def _forecasts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, target in enumerate([False, False, True, True]):
        session = f"2025-01-0{index + 2}"
        for model, probability in (
            ("M1_destination_own_history", [0.1, 0.2, 0.6, 0.7][index]),
            ("M3_directed_family_rotation", [0.05, 0.1, 0.8, 0.9][index]),
        ):
            rows.append(
                {
                    "forecast_id": f"{model}-{index}",
                    "period": 2025,
                    "forecast_session": session,
                    "destination_family": "family-a",
                    "target_window_sessions": 3,
                    "model_name": model,
                    "predicted_activation_probability": probability,
                    "prediction_state": "nominated" if probability >= 0.5 else "abstain",
                    "target_available": True,
                    "activation_target": target,
                    "probability_no_activation": 1.0 - probability,
                    "probability_multiple_activation": probability * 0.1,
                    "no_activation_flag": not target,
                    "multiple_activation_flag": False,
                }
            )
    return pd.DataFrame(rows)


def test_activation_metrics_match_known_brier_and_preserve_abstention() -> None:
    forecasts = _forecasts()
    result = activation_metrics(
        forecasts.loc[forecasts["model_name"].eq("M1_destination_own_history")]
    )

    expected = np.mean(np.square(np.array([0.1, 0.2, 0.6, 0.7]) - [0, 0, 1, 1]))
    assert result["brier_score"] == pytest.approx(expected)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["abstention"] == pytest.approx(0.5)
    assert result["precision"] == 1.0


def test_paired_comparison_is_control_minus_treatment_on_identical_rows() -> None:
    comparison = paired_model_comparison(
        _forecasts(),
        treatment="M3_directed_family_rotation",
        control="M1_destination_own_history",
        bootstrap_resamples=100,
        seed=7,
    )

    assert comparison["paired_rows"] == 4
    assert comparison["brier_improvement"] > 0.0
    assert comparison["log_loss_improvement"] > 0.0
    assert comparison["brier_interval_lower"] <= comparison["brier_improvement"]
    assert comparison["brier_interval_upper"] >= comparison["brier_improvement"]


def test_paired_comparison_fails_when_model_populations_differ() -> None:
    frame = _forecasts().iloc[:-1].copy()

    with pytest.raises(ValueError, match="paired population"):
        paired_model_comparison(
            frame,
            treatment="M3_directed_family_rotation",
            control="M1_destination_own_history",
            bootstrap_resamples=0,
        )


def test_calibration_table_does_not_turn_missing_target_into_zero() -> None:
    frame = _forecasts()
    missing = frame.iloc[[0]].copy()
    missing["target_available"] = False
    missing["activation_target"] = pd.NA
    frame = pd.concat([frame, missing], ignore_index=True)

    table = calibration_table(
        frame.loc[frame["model_name"].eq("M1_destination_own_history")], bins=5
    )

    assert table["forecasts"].sum() == 4


def test_system_metrics_keep_no_and_multiple_activation_separate() -> None:
    frame = _forecasts().loc[_forecasts()["model_name"].eq("M3_directed_family_rotation")].copy()
    frame.loc[frame.index[-1], "multiple_activation_flag"] = True
    result = system_activation_metrics(frame)

    assert "no_activation_brier" in result
    assert "multiple_activation_brier" in result
    assert result["multiple_activation_observations"] == 1


def test_pair_probability_is_bounded_and_shrinks_to_family_when_sparse() -> None:
    sparse = shrink_pair_probability(
        pair_activations=1,
        pair_support=2,
        family_probability=0.2,
        pooling_strength=20.0,
    )
    supported = shrink_pair_probability(
        pair_activations=80,
        pair_support=100,
        family_probability=0.2,
        pooling_strength=20.0,
    )

    assert 0.2 < sparse < 0.5
    assert supported > sparse
    assert supported < 0.8
