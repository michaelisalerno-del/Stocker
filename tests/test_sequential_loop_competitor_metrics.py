from __future__ import annotations

import math

import pandas as pd

from stocker_research.sequential_loop_competitor_veto import paired_predictive_metrics


def test_synthetic_leading_elimination_improves_paired_prediction() -> None:
    frame = pd.DataFrame(
        {
            "session_date": [f"2025-01-{day:02d}" for day in range(1, 21) for _ in range(2)],
            "target_positive": [0, 1] * 20,
            "target_remaining_net_bps": [-20.0, 30.0] * 20,
            "anchor_probability": [0.5] * 40,
            "sequential_probability": [0.2, 0.8] * 20,
        }
    )

    result = paired_predictive_metrics(frame, bootstrap_resamples=200, seed=7)

    assert result["paired_rows"] == 40
    assert result["brier_improvement"] > 0.0
    assert result["log_loss_improvement"] > 0.0
    assert result["brier_interval_lower"] > 0.0


def test_null_elimination_does_not_manufacture_incremental_value() -> None:
    frame = pd.DataFrame(
        {
            "session_date": ["2025-01-01", "2025-01-02"] * 10,
            "target_positive": [0, 1] * 10,
            "target_remaining_net_bps": [-10.0, 10.0] * 10,
            "anchor_probability": [0.5] * 20,
            "sequential_probability": [0.5] * 20,
        }
    )

    result = paired_predictive_metrics(frame, bootstrap_resamples=100, seed=9)

    assert math.isclose(result["brier_improvement"], 0.0)
    assert math.isclose(result["log_loss_improvement"], 0.0)
    assert math.isclose(result["paired_economic_increment_bps"], 0.0)
