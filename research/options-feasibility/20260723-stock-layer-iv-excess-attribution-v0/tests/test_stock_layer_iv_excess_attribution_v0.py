from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.stock_layer_iv_excess_attribution_v0 import (
    GROUP_D,
    GROUP_I,
    GROUP_M,
    GROUP_O,
    GROUP_R,
    MODEL_FEATURES,
    SAFETY_FLAGS,
    adjacent_increment_metrics,
    apply_tail_memberships,
    calculate_iv_excess_outcomes,
    choose_overall_decision,
    development_prediction_thresholds,
    fit_model_ladder,
    group_null_refits,
    grouped_permutation_attribution,
    incremental_tail_capture,
    permute_group_within_slates,
    session_bootstrap_multiplicities,
    shared_session_bootstrap,
    tail_metrics,
    tail_overlap,
    validate_feature_groups,
    validate_protected_boundary,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"


def test_contract_has_every_binding_safety_flag() -> None:
    contract = pd.read_json(EXPERIMENT_DIR / "contract.json", typ="series").to_dict()

    assert all(contract[key] == value for key, value in SAFETY_FLAGS.items())
    assert contract["limits"]["primary_logistic_models"] == 5
    assert contract["limits"]["session_bootstrap_draws"] == 10
    assert contract["limits"]["group_null_refits"] == 12
    assert contract["limits"]["assessment_permutations_per_group"] == 5


def test_feature_groups_are_exact_and_disjoint() -> None:
    validate_feature_groups()

    assert len(GROUP_O) == 31
    assert len(GROUP_D) == 14
    assert len(GROUP_I) == 26
    assert len(GROUP_R) == 15
    assert len(GROUP_M) == 5
    assert GROUP_D[:8] == (
        "daily_compression",
        "daily_directional_efficiency",
        "daily_trend_persistence",
        "daily_extension",
        "daily_rejection",
        "daily_volatility_acceleration",
        "daily_relative_strength",
        "daily_activity_acceleration",
    )
    assert GROUP_R == (
        "active_prefix_count",
        "active_prefix_family_count",
        "top_prefix_depth_fraction",
        "second_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
        "orientation_disagreement_fraction",
        "new_prefixes_last_1_bar",
        "invalidated_prefixes_last_1_bar",
        "active_prefix_count_change_last_1_bar",
        "active_prefix_count_change_last_3_bars",
        "top_prefix_depth_change_last_1_bar",
        "top_prefix_depth_change_last_3_bars",
        "matching_recent_loop_prefix_count",
        "recent_loop_memory_weighted_top_depth",
    )
    assert GROUP_M == (
        "mismatch_compression_vs_front_iv",
        "mismatch_daily_volatility_vs_front_iv",
        "mismatch_route_vs_front_premium",
        "mismatch_direction_agreement",
        "mismatch_complacent_broad_conflict",
    )


def test_g0_through_g4_are_a_strict_frozen_ladder() -> None:
    assert MODEL_FEATURES["G0"] == GROUP_O
    assert MODEL_FEATURES["G1"] == (*GROUP_O, *GROUP_D)
    assert MODEL_FEATURES["G2"] == (*GROUP_O, *GROUP_D, *GROUP_I)
    assert MODEL_FEATURES["G3"] == (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R)
    assert MODEL_FEATURES["G4"] == (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R, *GROUP_M)


def test_iv_excess_target_and_continuous_residual_are_frozen() -> None:
    result = calculate_iv_excess_outcomes(
        entry_price=[100.0, 100.0],
        close_15m=[101.0, 100.1],
        atm_iv=[0.40, 0.40],
    )
    expected = 0.40 * math.sqrt(15.0 / (252.0 * 390.0)) * math.sqrt(2.0 / math.pi)

    assert result.loc[0, "absolute_log_return_15m"] == pytest.approx(abs(math.log(1.01)))
    assert result.loc[0, "iv_expected_absolute_15m"] == pytest.approx(expected)
    assert result.loc[0, "iv_absolute_residual_15m"] == pytest.approx(
        abs(math.log(1.01)) - expected
    )
    assert result["movement_exceeds_prior_close_iv_15m"].tolist() == [1, 0]
    assert "option_pnl" not in result


def test_prediction_quantiles_are_development_frozen() -> None:
    probabilities = np.linspace(0.01, 1.0, 100)

    thresholds = development_prediction_thresholds(probabilities)

    assert thresholds == {
        "top_decile": pytest.approx(np.quantile(probabilities, 0.90)),
        "top_quintile": pytest.approx(np.quantile(probabilities, 0.80)),
        "top_5pct": pytest.approx(np.quantile(probabilities, 0.95)),
        "top_2pct": pytest.approx(np.quantile(probabilities, 0.98)),
    }


def _tail_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "symbol": ["A", "A", "B", "B"],
            "session": ["2025-01-02", "2025-01-02", "2025-02-03", "2025-02-03"],
            "row_weight": [0.25, 0.25, 0.25, 0.25],
            "G0_probability": [0.95, 0.90, 0.20, 0.10],
            "G1_probability": [0.20, 0.90, 0.95, 0.10],
            "G4_probability": [0.20, 0.90, 0.95, 0.10],
            "absolute_log_return_15m": [0.03, 0.02, 0.04, 0.001],
            "iv_expected_absolute_15m": [0.01, 0.01, 0.01, 0.01],
            "iv_sigma_15m": [0.0125, 0.0125, 0.0125, 0.0125],
            "iv_absolute_residual_15m": [0.02, 0.01, 0.03, -0.009],
            "movement_exceeds_prior_close_iv_15m": [1, 1, 1, 0],
        }
    )


def test_g0_and_g4_tail_membership_and_overlap_use_separate_thresholds() -> None:
    frame = apply_tail_memberships(
        _tail_frame(),
        {
            "G0": {
                "top_decile": 0.90,
                "top_quintile": 0.80,
                "top_5pct": 0.94,
                "top_2pct": 0.96,
            },
            "G1": {
                "top_decile": 0.90,
                "top_quintile": 0.80,
                "top_5pct": 0.94,
                "top_2pct": 0.96,
            },
            "G4": {
                "top_decile": 0.90,
                "top_quintile": 0.80,
                "top_5pct": 0.94,
                "top_2pct": 0.96,
            },
        },
    )

    assert frame["G0_top_decile"].tolist() == [True, True, False, False]
    assert frame["G4_top_decile"].tolist() == [False, True, True, False]
    overlap = tail_overlap(frame, "G0_top_decile", "G4_top_decile")
    assert overlap == {
        "intersection_rows": 1,
        "union_rows": 3,
        "jaccard_overlap": pytest.approx(1.0 / 3.0),
        "G4_only_rows": 1,
        "G0_only_rows": 1,
    }


def test_tail_metrics_and_incremental_capture_use_iv_residuals_not_pnl() -> None:
    frame = apply_tail_memberships(
        _tail_frame(),
        {
            model: {
                "top_decile": 0.90,
                "top_quintile": 0.80,
                "top_5pct": 0.94,
                "top_2pct": 0.96,
            }
            for model in ("G0", "G1", "G4")
        },
    )

    metrics = tail_metrics(frame.loc[frame["G4_top_decile"]], model="G4", tail="top_decile")
    capture = incremental_tail_capture(frame, earlier_model="G0", later_model="G1")

    assert metrics["rows"] == 2
    assert metrics["mean_iv_residual"] == pytest.approx(0.02)
    assert metrics["median_iv_residual"] == pytest.approx(0.02)
    assert metrics["exceed_iv_rate"] == pytest.approx(1.0)
    assert capture["new_positive_targets_entering_top_decile"] == 1
    assert capture["positive_targets_leaving_top_decile"] == 1
    assert capture["net_change_captured_positive_targets"] == 0
    assert capture["mean_iv_residual_entering_top_decile"] == pytest.approx(0.03)
    assert capture["mean_iv_residual_leaving_top_decile"] == pytest.approx(0.02)
    assert "option_pnl" not in metrics


def test_grouped_assessment_permutation_keeps_each_bundle_inside_its_slate() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-02"] * 3,
            "checkpoint": [6] * 3,
            "symbol": ["A", "B", "C"],
            "outcome": [0, 1, 0],
            "d1": [1.0, 2.0, 3.0],
            "d2": [10.0, 20.0, 30.0],
            "untouched": [100.0, 200.0, 300.0],
        }
    )

    permuted = permute_group_within_slates(
        frame,
        columns=("d1", "d2"),
        slate_columns=("session", "checkpoint"),
        seed=20260723,
    )

    assert set(permuted[["d1", "d2"]].itertuples(index=False, name=None)) == set(
        frame[["d1", "d2"]].itertuples(index=False, name=None)
    )
    assert permuted["outcome"].equals(frame["outcome"])
    assert permuted["untouched"].equals(frame["untouched"])


def test_session_bootstrap_has_exactly_ten_shared_whole_session_draws() -> None:
    sessions = pd.Series(["a", "a", "b", "b", "c"])

    draws = session_bootstrap_multiplicities(sessions, draws=10, seed=20260723)

    assert len(draws) == 10
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)
    with pytest.raises(ValueError, match="exactly 10"):
        session_bootstrap_multiplicities(sessions, draws=9, seed=20260723)


def test_protected_date_rejection_applies_to_market_and_option_observations() -> None:
    safe = pd.DataFrame(
        {
            "session": ["2025-08-22"],
            "stock_information_date": ["2025-08-21"],
            "options_observation_date": ["2025-08-21"],
        }
    )
    audit = validate_protected_boundary(safe)

    assert audit["protected_market_rows_materialised"] == 0
    assert audit["protected_option_observations_materialised"] == 0
    with pytest.raises(ValueError, match="blocked_chronology_or_leakage_failure"):
        validate_protected_boundary(safe.assign(options_observation_date="2025-08-23"))


@pytest.mark.parametrize(
    ("statuses", "tail_status", "expected"),
    [
        (
            {"D": "supported", "I": "supported", "R": "not_supported", "M": "not_supported"},
            "supported",
            "multiple_stock_layers_contribute_to_iv_excess",
        ),
        (
            {"D": "not_supported", "I": "not_supported", "R": "supported", "M": "not_supported"},
            "supported",
            "route_competition_drives_iv_excess_increment",
        ),
        (
            {"D": "not_supported", "I": "not_supported", "R": "not_supported", "M": "supported"},
            "supported",
            "cross_market_mismatch_adds_iv_excess_increment",
        ),
        (
            {"D": "supported", "I": "not_supported", "R": "not_supported", "M": "not_supported"},
            "not_supported",
            "stock_layers_improve_ranking_but_not_positive_iv_tail",
        ),
        (
            {
                "D": "not_supported",
                "I": "not_supported",
                "R": "not_supported",
                "M": "not_supported",
            },
            "supported",
            "positive_iv_excess_tail_without_localised_group",
        ),
    ],
)
def test_decision_logic(
    statuses: dict[str, str],
    tail_status: str,
    expected: str,
) -> None:
    assert (
        choose_overall_decision(
            group_statuses=statuses,
            final_tail_status=tail_status,
            full_bundle_increment_reproduced=True,
        )
        == expected
    )


def _synthetic_model_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260723)
    rows = 80
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{value}" for value in range(rows)],
            "period": ["development"] * 60 + ["assessment"] * 20,
            "session": [
                *(f"2024-01-{value % 20 + 2:02d}" for value in range(60)),
                *(f"2025-01-{value % 10 + 2:02d}" for value in range(20)),
            ],
            "symbol": ["A" if value % 2 == 0 else "B" for value in range(rows)],
            "checkpoint": [6 + 2 * (value % 15) for value in range(rows)],
            "route_resolution_state": [
                ("BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "NARROWING", "OTHER")[value % 4]
                for value in range(rows)
            ],
            "row_weight": [0.5] * rows,
            "movement_exceeds_prior_close_iv_15m": [value % 2 for value in range(rows)],
            "absolute_log_return_15m": [0.02 if value % 2 else 0.005 for value in range(rows)],
            "iv_expected_absolute_15m": [0.01] * rows,
            "iv_sigma_15m": [0.0125] * rows,
            "iv_absolute_residual_15m": [0.01 if value % 2 else -0.005 for value in range(rows)],
        }
    )
    for feature in dict.fromkeys((*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R, *GROUP_M)):
        frame[feature] = rng.normal(size=rows)
    for checkpoint in range(6, 35, 2):
        frame[f"checkpoint_{checkpoint}"] = frame["checkpoint"].eq(checkpoint).astype(float)
    frame.loc[frame["period"].eq("assessment"), "daily_compression"] += 0.5
    return frame


def test_g0_g4_model_ladder_and_development_only_scaling() -> None:
    panel = _synthetic_model_panel()

    result = fit_model_ladder(panel)

    assert list(result.models) == ["G0", "G1", "G2", "G3", "G4"]
    assert len(result.metrics) == 5
    assert set(result.thresholds) == {"G0", "G1", "G2", "G3", "G4"}
    daily_index = result.models["G1"].numeric_features.index("daily_compression")
    expected_development_mean = float(
        panel.loc[panel["period"].eq("development"), "daily_compression"].mean()
    )
    assert result.models["G1"].numeric_means[daily_index] == pytest.approx(
        expected_development_mean
    )
    assert result.models["G1"].numeric_means[daily_index] != pytest.approx(
        float(panel["daily_compression"].mean())
    )
    assert np.isfinite(
        result.assessment[[f"G{index}_probability" for index in range(5)]].to_numpy(float)
    ).all()


def test_adjacent_increment_metrics_use_positive_as_improvement() -> None:
    result = fit_model_ladder(_synthetic_model_panel())

    increments = adjacent_increment_metrics(result.metrics)

    assert increments["comparison"].tolist() == ["G1-G0", "G2-G1", "G3-G2", "G4-G3"]
    first = increments.iloc[0]
    g0 = result.metrics.set_index("model").loc["G0"]
    g1 = result.metrics.set_index("model").loc["G1"]
    assert first["log_loss_improvement"] == pytest.approx(g0["log_loss"] - g1["log_loss"])
    assert first["brier_improvement"] == pytest.approx(g0["brier_score"] - g1["brier_score"])
    assert first["auc_improvement"] == pytest.approx(g1["auc"] - g0["auc"])
    assert first["average_precision_improvement"] == pytest.approx(
        g1["average_precision"] - g0["average_precision"]
    )


def test_grouped_final_model_permutation_has_five_draws_per_stock_group() -> None:
    result = fit_model_ladder(_synthetic_model_panel())

    attribution = grouped_permutation_attribution(result)

    assert len(attribution) == 20
    assert attribution.groupby("group").size().to_dict() == {"D": 5, "I": 5, "M": 5, "R": 5}
    assert attribution["refit"].eq(False).all()
    assert attribution["within_slate"].eq("session_x_checkpoint").all()


def test_every_group_specific_null_has_exactly_three_refits() -> None:
    panel = _synthetic_model_panel()
    result = fit_model_ladder(panel)

    nulls = group_null_refits(panel, result)

    assert len(nulls.metrics) == 12
    assert len(nulls.models) == 12
    assert nulls.metrics.groupby("group").size().to_dict() == {"D": 3, "I": 3, "M": 3, "R": 3}
    assert nulls.metrics["null_refit"].nunique() == 3
    assert nulls.metrics["within_slate"].eq("period_x_session_x_checkpoint").all()


def test_shared_session_bootstrap_has_required_adjacent_and_tail_intervals() -> None:
    result = fit_model_ladder(_synthetic_model_panel())
    test_thresholds = {
        model: development_prediction_thresholds(result.assessment[f"{model}_probability"])
        for model in ("G0", "G1", "G2", "G3", "G4")
    }
    result = type(result)(
        development=result.development,
        assessment=result.assessment,
        models=result.models,
        thresholds=test_thresholds,
        metrics=result.metrics,
    )
    assessment = apply_tail_memberships(result.assessment, test_thresholds)

    intervals = shared_session_bootstrap(result, assessment)

    assert set(intervals["confidence"]) == {0.80, 0.90, 0.95}
    assert intervals["draws"].eq(10).all()
    assert intervals["fixed_prediction"].eq(True).all()
    assert intervals["whole_session_resampling"].eq(True).all()
    assert intervals["statistic"].eq("G1_minus_G0_log_loss_improvement").any()
    assert intervals["statistic"].eq("G4_top_decile_mean_iv_residual").any()
    assert intervals["statistic"].eq("G4_minus_G0_top_decile_mean_iv_residual_difference").any()


def _read_artifact_json(name: str) -> dict[str, object]:
    return json.loads((PRIMARY / name).read_text(encoding="utf-8"))


def test_frozen_branch_c_reconstruction_artifact_passes_exactly() -> None:
    reconstruction = _read_artifact_json("frozen_panel_reconstruction.json")

    assert reconstruction["passed"] is True
    assert reconstruction["row_identity_mismatches"] == 0
    assert reconstruction["selected_contract_mismatches"] == 0
    assert float(reconstruction["maximum_feature_difference"]) <= 1e-12
    assert float(reconstruction["maximum_outcome_difference"]) <= 1e-12


def test_g0_and_g4_are_exact_predecessor_endpoint_reconstructions() -> None:
    reconstruction = _read_artifact_json("predecessor_model_reconstruction.json")

    assert reconstruction["passed"] is True
    assert float(reconstruction["G0"]["maximum_probability_difference"]) <= 1e-12
    assert float(reconstruction["G0"]["maximum_metric_difference"]) <= 1e-12
    assert float(reconstruction["G4"]["maximum_probability_difference"]) <= 1e-12
    assert float(reconstruction["G4"]["maximum_metric_difference"]) <= 1e-12


def test_run_artifacts_record_exact_fit_and_resampling_limits() -> None:
    configuration = _read_artifact_json("model_configurations.json")
    permutation = pd.read_csv(PRIMARY / "grouped_permutation_attribution.csv")
    nulls = pd.read_csv(PRIMARY / "group_null_metrics.csv")
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")

    assert configuration["primary_logistic_models_fitted"] == 5
    assert set(configuration["models"]) == {"G0", "G1", "G2", "G3", "G4"}
    assert len(permutation) == 20
    assert len(nulls) == 12
    assert nulls.groupby("group").size().eq(3).all()
    assert bootstrap["draws"].eq(10).all()


def test_determinism_and_independent_audit_pass() -> None:
    determinism = _read_artifact_json("determinism_check.json")
    audit = _read_artifact_json("independent_audit.json")

    assert determinism["passed"] is True
    assert determinism["joined_row_mismatches"] == 0
    assert float(determinism["maximum_model_probability_difference"]) <= 1e-12
    assert determinism["tail_membership_mismatches"] == 0
    assert audit["passed"] is True
    assert audit["checks_failed"] == 0
    assert audit["models_independently_refitted_for_coefficient_audit"] == 5
    checks = {item["check"]: item for item in audit["checks"]}
    coefficient_check = checks["all_model_coefficients_and_manual_probability_reconstruction"]
    assert coefficient_check["passed"] is True
    assert coefficient_check["evidence"]["models_independently_refitted"] == 5
    decision_check = checks["decision_logic"]
    assert decision_check["passed"] is True
    assert decision_check["evidence"]["independent_reconstruction"]["overall_decision"] == (
        "stock_layers_improve_ranking_but_not_positive_iv_tail"
    )
