from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.sequential_loop_competitor_veto import (
    PayoffClassConfig,
    classify_payoff_families,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "period": 2025,
        "score_session": "2025-04-03",
        "decision_timestamp": "2025-04-03T13:30:00Z",
        "prediction_frozen_at": "2025-04-03T13:30:00Z",
        "training_latest_availability_timestamp": "2025-04-02T20:00:00Z",
        "loop_id": "cycle_04",
        "orientation": "state_4",
        "horizon": 24,
        "model_name": "hierarchical_payoff_history_change_point",
        "posterior_mean_net_bps": 20.0,
        "posterior_std_net_bps": 5.0,
        "posterior_lower_bound_net_bps": 11.8,
        "effective_sessions": 12.0,
        "independent_stocks": 8,
        "raw_fills": 100,
        "effective_sample_size": 20.0,
    }
    row.update(updates)
    return row


def test_good_bad_and_unknown_use_only_frozen_prior_support() -> None:
    frame = pd.DataFrame(
        [
            _row(loop_id="cycle_04"),
            _row(
                loop_id="cycle_07",
                posterior_mean_net_bps=-20.0,
                posterior_lower_bound_net_bps=-28.2,
            ),
            _row(loop_id="cycle_10", independent_stocks=1, raw_fills=10_000),
        ]
    )

    result = classify_payoff_families(frame, PayoffClassConfig())

    assert dict(zip(result["loop_id"], result["payoff_class"], strict=True)) == {
        "cycle_04": "good",
        "cycle_07": "bad",
        "cycle_10": "unknown",
    }


def test_appending_future_forecasts_does_not_change_frozen_classification() -> None:
    current = pd.DataFrame([_row()])
    future = pd.DataFrame([_row(score_session="2025-04-04", posterior_mean_net_bps=-999.0)])

    expected = classify_payoff_families(current, PayoffClassConfig())
    appended = classify_payoff_families(
        pd.concat([current, future], ignore_index=True),
        PayoffClassConfig(),
        score_session="2025-04-03",
    )

    pd.testing.assert_frame_equal(expected.reset_index(drop=True), appended.reset_index(drop=True))


def test_current_or_future_outcome_columns_are_rejected_from_classification() -> None:
    frame = pd.DataFrame([_row(target_robust_net_bps=100.0)])

    with pytest.raises(ValueError, match="outcome column"):
        classify_payoff_families(frame, PayoffClassConfig())


def test_training_information_must_precede_the_scoring_decision() -> None:
    frame = pd.DataFrame([_row(training_latest_availability_timestamp="2025-04-03T13:31:00Z")])

    with pytest.raises(ValueError, match="training availability"):
        classify_payoff_families(frame, PayoffClassConfig())
