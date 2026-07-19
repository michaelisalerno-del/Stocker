from __future__ import annotations

import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import PRIMARY_FEATURES
from stocker_research.observable_event_ranking_v1.development import (
    fit_final_frozen_components,
    run_development_oof,
)


def _development_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(
        pd.date_range("2024-01-01", periods=12, freq="MS", tz="UTC")
    ):
        for candidate in range(8):
            target = candidate / 7
            row: dict[str, object] = {
                "event_id": f"event-{month_number}-{candidate}",
                "slate_id": f"slate-{month_number}",
                "symbol": f"S{candidate}",
                "sector": "Technology" if candidate < 4 else "Industrials",
                "session": month,
                "assigned_decision_time": month + pd.Timedelta(hours=15),
                "future_return_60m": (candidate - 3.5) / 1_000,
                "target_rank_60m": target,
                "slate_evaluable": True,
            }
            for feature_number, feature in enumerate(PRIMARY_FEATURES):
                row[feature] = target + feature_number / 1_000
            rows.append(row)
    return pd.DataFrame(rows)


def test_development_oof_freezes_strongest_deterministic_baseline_and_one_linear_model() -> None:
    result = run_development_oof(_development_rows(), bootstrap_draws=100)

    assert result.strongest_baseline != "B0_RANDOM_ELIGIBLE"
    assert result.strongest_baseline == "B1_EVENT_STRENGTH"
    assert set(result.candidate_predictions["fold_id"]) == {"fold_01", "fold_02"}
    assert result.model_effective_configuration["model_id"] == "M1_POOLED_LINEAR_RANKER"
    assert result.model_effective_configuration["alpha"] == 1.0
    assert result.model_effective_configuration["hyperparameter_search"] is False
    assert result.slate_metrics["candidate_ic"].mean() > 0.99
    assert result.ic_bootstrap.draws == 100
    assert len(result.turnover_results) == len(result.slate_metrics)
    assert result.turnover_results["unique_selections_that_day"].eq(2).all()
    assert result.leave_one_stock_out["full_pipeline_refitted"].all()
    assert result.leave_one_stock_out["remaining_rows"].eq(84).all()

    final_model, final_baseline = fit_final_frozen_components(
        _development_rows(),
        result.strongest_baseline,
    )
    assert final_model["training_rows"] == 96
    assert final_model["feature_names"] == list(PRIMARY_FEATURES)
    assert final_baseline == {
        "baseline_id": "B1_EVENT_STRENGTH",
        "kind": "direct_observable_feature",
        "source_feature": "event_strength",
        "fit_uses_outcomes": False,
    }


def test_evaluation_outcome_mutation_does_not_change_same_fold_training_preprocessor() -> None:
    frame = _development_rows()
    original = run_development_oof(frame, bootstrap_draws=20)
    mutated = frame.copy()
    evaluation_mask = mutated["session"].between(
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-09-01", tz="UTC"),
    )
    mutated.loc[evaluation_mask, "target_rank_60m"] = (
        1.0 - mutated.loc[evaluation_mask, "target_rank_60m"]
    )
    rerun = run_development_oof(mutated, bootstrap_draws=20)

    assert original.model_parameters["fold_01"] == rerun.model_parameters["fold_01"]
