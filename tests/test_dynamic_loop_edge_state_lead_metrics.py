from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state_lead_lag.metrics import (
    build_feature_contribution_bins,
    build_paired_prediction_table,
    paired_lead_metrics,
    validate_paired_training_identity,
)


def _synthetic_pairs(*, related: bool) -> pd.DataFrame:
    sessions = pd.date_range("2025-01-02", periods=80, freq="B").strftime("%Y-%m-%d")
    signal = np.tile([0, 1], 40)
    target = np.roll(signal, 1)
    target[0] = 0
    rows: list[dict[str, object]] = []
    for model in ("hierarchical_payoff_history_change_point", "hierarchical_change_point"):
        for index, session in enumerate(sessions):
            for lead in (0, 1):
                target_index = index + lead
                if target_index >= len(sessions):
                    continue
                probability = 0.5
                if model == "hierarchical_change_point" and related:
                    probability = 0.9 if signal[index] else 0.1
                rows.append(
                    {
                        "forecast_id": f"{model}-{index}",
                        "model_name": model,
                        "period": 2025,
                        "score_session": session,
                        "loop_id": "cycle_01",
                        "orientation": "state_1",
                        "horizon": 24,
                        "target_lead_sessions": lead,
                        "target_session": sessions[target_index],
                        "target_outcome_id": f"outcome-{target_index}",
                        "target_status": "payoff_settled",
                        "target_payoff_available": True,
                        "target_payoff_positive": bool(target[target_index]),
                        "target_robust_net_bps": 20.0 if target[target_index] else -20.0,
                        "p_next_payoff_positive": probability,
                        "p_edge_positive": 0.5,
                        "p_edge_active": probability,
                        "p_on_next": probability,
                        "p_off_next": 1.0 - probability,
                        "p_survive_horizon": probability,
                        "posterior_mean_net_bps": 0.0,
                        "posterior_lower_bound_net_bps": -1.0,
                        "edge_state": "active" if probability >= 0.8 else "unknown",
                        "target_independent_stocks": 3,
                        "target_effective_sample_size": 3.0,
                        "target_episode_state": "positive" if target[target_index] else "neutral",
                        "target_episode_onset_within_lead": False,
                        "target_episode_survival": False,
                    }
                )
    return pd.DataFrame(rows)


def test_synthetic_leading_feature_is_detected_at_lead_one_not_lead_zero() -> None:
    paired = build_paired_prediction_table(_synthetic_pairs(related=True))
    metrics = paired_lead_metrics(paired, bootstrap_resamples=100, seed=7)

    lead_zero = metrics.loc[metrics["target_lead_sessions"].eq(0)].iloc[0]
    lead_one = metrics.loc[metrics["target_lead_sessions"].eq(1)].iloc[0]
    assert lead_one["paired_brier_improvement"] > 0.1
    assert lead_zero["paired_brier_improvement"] < 0.0
    assert lead_one["paired_economic_increment_bps"] > 0.0


def test_null_case_does_not_manufacture_feature_improvement() -> None:
    paired = build_paired_prediction_table(_synthetic_pairs(related=False))
    metrics = paired_lead_metrics(paired, bootstrap_resamples=100, seed=7)

    assert metrics["paired_brier_improvement"].eq(0.0).all()
    assert metrics["paired_log_loss_improvement"].eq(0.0).all()
    assert metrics["paired_economic_increment_bps"].eq(0.0).all()


def test_contribution_bins_use_predictor_only_and_not_target() -> None:
    paired = build_paired_prediction_table(_synthetic_pairs(related=True))
    original = build_feature_contribution_bins(paired, bins=5)
    shuffled = paired.copy()
    shuffled["target_robust_net_bps"] = (
        shuffled["target_robust_net_bps"].sample(frac=1.0, random_state=4).to_numpy()
    )
    rebuilt = build_feature_contribution_bins(shuffled, bins=5)

    assert original["contribution_bin"].equals(rebuilt["contribution_bin"])


def test_full_and_control_must_have_identical_training_state() -> None:
    forecasts = pd.DataFrame(
        {
            "period": [2025, 2025],
            "score_session": ["2025-01-03"] * 2,
            "loop_id": ["cycle_01"] * 2,
            "orientation": ["state_1"] * 2,
            "horizon": [24, 24],
            "model_name": [
                "hierarchical_payoff_history_change_point",
                "hierarchical_change_point",
            ],
            "posterior_mean_net_bps": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="training-state mismatch"):
        validate_paired_training_identity(forecasts, ["posterior_mean_net_bps"])
