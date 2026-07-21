#!/usr/bin/env python3
"""Independently audit Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "behavioural_trajectory_test": True,
    "soft_regime_mixture": True,
    "coarse_loop_family_target": True,
    "structural_outcomes_only": True,
    "economic_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
ANCHORS_BY_DECISION = {6: (2, 4, 6), 12: (6, 9, 12)}
TRAJECTORY_INTERACTION_SPECS = (
    ("transition_probability_x_arousal_change", "transition_probability", "arousal_change"),
    ("posterior_entropy_x_frustration_change", "posterior_entropy", "frustration_change"),
    ("top_second_margin_x_conviction_change", "top_second_margin", "conviction_change"),
    (
        "transition_probability_x_signed_pressure_acceleration",
        "transition_probability",
        "signed_pressure_acceleration",
    ),
    (
        "posterior_entropy_x_tension_acceleration",
        "posterior_entropy",
        "tension_acceleration",
    ),
    (
        "top_state_probability_x_signed_exhaustion_change",
        "top_state_probability",
        "signed_exhaustion_change",
    ),
)
TRAJECTORY_INTERACTION_FEATURES = tuple(row[0] for row in TRAJECTORY_INTERACTION_SPECS)

DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "artifacts"
    / "primary"
)
DEFAULT_COMPONENT_LEDGER = (
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
    / "artifacts"
    / "primary"
    / "behavioural_component_ledger.parquet"
)
EXPECTED_PANEL_HASH = "8e6919c03e207e3eee2f05a47271dbff0129c02a0cb0f04c0b6eb1410a65dca5"
EXPECTED_COMPONENT_HASH = "96fad90eeeb6edf8b2f82ba809fb6939ad3ce5b7d5bf8193abacd90f471a91cb"
TARGET_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
)
PRIMARY_BEHAVIOURS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "predecessor_population_reconstruction.json",
    "frozen_m2_reconstruction.json",
    "trajectory_anchor_manifest.json",
    "trajectory_feature_manifest.json",
    "trajectory_missingness.csv",
    "interaction_manifest.json",
    "decision_panel.parquet",
    "trajectory_ledger.parquet",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "pooled_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "class_metrics.csv",
    "trajectory_diagnostics.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
    "report.md",
)


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), **details})


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def anchor_formula_availability(completed_bar_count: int) -> tuple[bool, str | None]:
    count = int(completed_bar_count)
    if count < 2:
        return False, "frozen_opening_raw_components_requires_at_least_two_completed_bars"
    if count % 2:
        return False, "frozen_opening_raw_components_requires_even_completed_bar_count"
    return True, None


def trajectory_anchors(decision_ordinal: int) -> tuple[int, int, int]:
    try:
        return ANCHORS_BY_DECISION[int(decision_ordinal)]
    except KeyError as error:
        raise ValueError(f"unexpected decision ordinal: {decision_ordinal}") from error


def trajectory_feature_values(
    earliest: float, middle: float, final: float
) -> dict[str, float | int]:
    first = middle - earliest
    recent = final - middle
    persistence = 1 if earliest < middle < final else -1 if earliest > middle > final else 0
    reversal = int(first != 0.0 and recent != 0.0 and ((first > 0.0) != (recent > 0.0)))
    return {
        "change": final - earliest,
        "recent_change": recent,
        "acceleration": recent - first,
        "persistence": persistence,
        "reversal": reversal,
        "peak_displacement": final - max(earliest, middle, final),
    }


def manual_multinomial_probabilities(
    frame: pd.DataFrame,
    model: dict[str, Any],
) -> np.ndarray:
    features = [str(value) for value in model["features"]]
    values = frame.loc[:, features].to_numpy(dtype=float)
    means = np.asarray(model["scaler_mean"], dtype=float)
    scales = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficient"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    logits = ((values - means) / scales) @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return np.asarray(exponentiated / exponentiated.sum(axis=1, keepdims=True), dtype=float)


def multiclass_brier(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return weighted_mean(np.square(probabilities - one_hot).sum(axis=1), weights)


def prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=float)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return np.asarray(-terms.sum(axis=1), dtype=float)


def calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, 11)
    total = float(weights.sum())
    result = 0.0
    for index in range(10):
        upper = confidence <= edges[index + 1] if index == 9 else confidence < edges[index + 1]
        mask = (confidence >= edges[index]) & upper
        if mask.any():
            result += (
                float(weights[mask].sum())
                / total
                * abs(
                    weighted_mean(correct[mask].astype(float), weights[mask])
                    - weighted_mean(confidence[mask], weights[mask])
                )
            )
    return result


def independent_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    class_order: tuple[str, ...],
) -> dict[str, float]:
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(order[index] == target)[0]) + 1 for index, target in enumerate(targets)]
    )
    realised = probabilities[np.arange(len(targets)), targets]
    entropy = prediction_entropy(probabilities)
    mean_entropy = weighted_mean(entropy, weights)
    return {
        "multiclass_log_loss": -weighted_mean(np.log(realised), weights),
        "multiclass_brier": multiclass_brier(targets, probabilities, weights),
        "top_one_accuracy": weighted_mean((ranks <= 1).astype(float), weights),
        "top_two_accuracy": weighted_mean((ranks <= 2).astype(float), weights),
        "top_three_accuracy": weighted_mean((ranks <= 3).astype(float), weights),
        "mean_reciprocal_rank": weighted_mean(1.0 / ranks, weights),
        "mean_probability_realised_class": weighted_mean(realised, weights),
        "macro_ovr_auc": float(
            roc_auc_score(
                targets,
                probabilities,
                labels=np.arange(len(class_order)),
                multi_class="ovr",
                average="macro",
                sample_weight=weights,
            )
        ),
        "expected_calibration_error": calibration_error(targets, probabilities, weights),
        "prediction_entropy": mean_entropy,
        "effective_candidate_count": math.exp(mean_entropy),
    }


def maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return math.inf
    return float(np.abs(left.astype(float) - right.astype(float)).max(initial=0.0))


def audit_artifacts(
    artifacts: Path,
    *,
    predecessor_primary: Path = DEFAULT_PREDECESSOR,
    component_ledger_path: Path = DEFAULT_COMPONENT_LEDGER,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifacts / name).is_file()]
    check(checks, "required_artifacts", not missing, missing=missing)
    if missing:
        result = {
            **SAFETY_FLAGS,
            "audit": "independent_lightweight_audit_v0",
            "checks": checks,
            "passed": False,
        }
        (artifacts / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
        return result

    contract = cast(dict[str, Any], json.loads((artifacts / "contract.json").read_text()))
    decision = cast(dict[str, Any], json.loads((artifacts / "decision.json").read_text()))
    source = cast(dict[str, Any], json.loads((artifacts / "source_manifest.json").read_text()))
    boundary = cast(
        dict[str, Any], json.loads((artifacts / "protected_boundary_audit.json").read_text())
    )
    population = cast(
        dict[str, Any],
        json.loads((artifacts / "predecessor_population_reconstruction.json").read_text()),
    )
    frozen_m2 = cast(
        dict[str, Any], json.loads((artifacts / "frozen_m2_reconstruction.json").read_text())
    )
    anchor_manifest = cast(
        dict[str, Any], json.loads((artifacts / "trajectory_anchor_manifest.json").read_text())
    )
    feature_manifest = cast(
        dict[str, Any], json.loads((artifacts / "trajectory_feature_manifest.json").read_text())
    )
    interaction_manifest = cast(
        dict[str, Any], json.loads((artifacts / "interaction_manifest.json").read_text())
    )
    configurations = cast(
        dict[str, Any], json.loads((artifacts / "model_configurations.json").read_text())
    )
    coefficients = cast(
        dict[str, Any], json.loads((artifacts / "model_coefficients.json").read_text())
    )
    determinism = cast(
        dict[str, Any], json.loads((artifacts / "determinism_check.json").read_text())
    )
    safety_passed = all(
        contract.get(key) == value
        and contract.get("safety", {}).get(key) == value
        and decision.get(key) == value
        for key, value in SAFETY_FLAGS.items()
    )
    check(checks, "safety_flags", safety_passed)
    check(
        checks,
        "dates_and_protected_boundary",
        bool(
            source["dates_read"]
            == {
                "start": "2024-01-01",
                "end_inclusive": "2025-08-22",
                "protected_start": "2025-08-23",
            }
            and boundary["protected_rows_materialised"] == 0
            and source["protected_rows_materialised"] == 0
        ),
        protected_rows_materialised=boundary["protected_rows_materialised"],
    )

    panel_path = artifacts / "decision_panel.parquet"
    predecessor_panel_path = predecessor_primary / "decision_panel.parquet"
    panel_hash = sha256_file(panel_path)
    predecessor_hash = sha256_file(predecessor_panel_path)
    panel = pd.read_parquet(panel_path)
    predecessor_panel = pd.read_parquet(predecessor_panel_path)
    exact_panel = bool(
        panel_hash == EXPECTED_PANEL_HASH
        and predecessor_hash == EXPECTED_PANEL_HASH
        and panel.equals(predecessor_panel)
        and len(panel) == 15_549
        and population["passed"]
    )
    check(
        checks,
        "exact_predecessor_population",
        exact_panel,
        rows=len(panel),
        panel_sha256=panel_hash,
    )
    sessions = pd.to_datetime(panel["session"], utc=True, errors="raise")
    check(
        checks,
        "2024_fit_2025_assessment_and_protected_dates",
        bool(
            sessions.min() >= pd.Timestamp("2024-01-01T00:00:00Z")
            and sessions.max() <= pd.Timestamp("2025-08-22T00:00:00Z")
            and set(panel["year"].unique()) == {2024, 2025}
        ),
        minimum_session=str(sessions.min().date()),
        maximum_session=str(sessions.max().date()),
    )

    archived_coefficients = cast(
        dict[str, Any],
        json.loads((predecessor_primary / "model_coefficients.json").read_text()),
    )["models"]["M2"]
    t0_payload = coefficients["models"]["T0"]
    coefficient_error = max(
        maximum_difference(
            np.asarray(t0_payload["coefficient"]),
            np.asarray(archived_coefficients["coefficient"]),
        ),
        maximum_difference(
            np.asarray(t0_payload["intercept"]),
            np.asarray(archived_coefficients["intercept"]),
        ),
        maximum_difference(
            np.asarray(t0_payload["scaler_mean"]),
            np.asarray(archived_coefficients["scaler_mean"]),
        ),
        maximum_difference(
            np.asarray(t0_payload["scaler_scale"]),
            np.asarray(archived_coefficients["scaler_scale"]),
        ),
    )
    check(
        checks,
        "development_only_preprocessing_and_model_coefficients",
        bool(
            coefficient_error <= 1e-12
            and configurations["preprocessor"] == "StandardScaler_fit_on_2024_only"
            and configurations["actual_primary_fitted_model_count"] == 1
        ),
        maximum_error=coefficient_error,
    )

    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)].reset_index(
        drop=True
    )
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    probabilities = manual_multinomial_probabilities(assessment, t0_payload)
    stored_probabilities = predictions.loc[
        :, [f"probability__T0__{label}" for label in TARGET_CLASSES]
    ].to_numpy(dtype=float)
    archived_predictions = pd.read_parquet(predecessor_primary / "assessment_predictions.parquet")
    archived_probabilities = archived_predictions.loc[
        :, [f"probability__M2__{label}" for label in TARGET_CLASSES]
    ].to_numpy(dtype=float)
    probability_error = maximum_difference(probabilities, stored_probabilities)
    archived_probability_error = maximum_difference(probabilities, archived_probabilities)
    sample_rows = min(100, len(assessment))
    sample_error = maximum_difference(
        probabilities[:sample_rows], stored_probabilities[:sample_rows]
    )
    check(
        checks,
        "manual_probability_reconstruction_at_least_100_rows",
        bool(
            sample_rows >= 100
            and probability_error <= 1e-12
            and archived_probability_error <= 1e-12
            and frozen_m2["passed"]
            and float(frozen_m2["maximum_probability_difference"]) <= 1e-12
        ),
        rows_checked=sample_rows,
        maximum_probability_error=probability_error,
        maximum_archived_probability_error=archived_probability_error,
        sampled_probability_error=sample_error,
    )
    actual_metrics = independent_metrics(assessment, probabilities, TARGET_CLASSES)
    pooled = pd.read_csv(artifacts / "pooled_metrics.csv").iloc[0]
    metric_errors = {
        metric: abs(actual - float(pooled[metric])) for metric, actual in actual_metrics.items()
    }
    archived_metric = (
        pd.read_csv(predecessor_primary / "pooled_metrics.csv")
        .loc[lambda frame: frame["model"].eq("M2")]
        .iloc[0]
    )
    predecessor_metrics = dict(actual_metrics)
    predecessor_metrics["effective_candidate_count"] = weighted_mean(
        np.exp(prediction_entropy(probabilities)),
        assessment["row_weight"].to_numpy(dtype=float),
    )
    archived_metric_errors = {
        metric: abs(actual - float(archived_metric[metric]))
        for metric, actual in predecessor_metrics.items()
    }
    screen_metric_errors = {
        metric: abs(actual - float(frozen_m2["screen_metrics"][metric]))
        for metric, actual in actual_metrics.items()
    }
    maximum_metric_error = max(
        *metric_errors.values(),
        *archived_metric_errors.values(),
        *screen_metric_errors.values(),
    )
    check(
        checks,
        "multiclass_proper_and_ranking_metrics",
        bool(
            maximum_metric_error <= 1e-12
            and float(frozen_m2["maximum_metric_difference"]) <= 1e-12
            and frozen_m2["effective_candidate_count_definition"]
            == "exp(pooled_prediction_entropy)"
        ),
        maximum_metric_error=maximum_metric_error,
        expected_effective_candidate_count=math.exp(actual_metrics["prediction_entropy"]),
    )

    trajectory = pd.read_parquet(artifacts / "trajectory_ledger.parquet")
    components_hash = sha256_file(component_ledger_path)
    components = (
        pd.read_parquet(
            component_ledger_path,
            columns=[
                "symbol",
                "session",
                "decision_ordinal",
                "bar_start_timestamps_utc",
                "bar_open",
                "bar_high",
                "bar_low",
                "bar_close",
                "historical_relative_activity",
            ],
        )
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    anchor_errors = 0
    future_anchor_rows = 0
    formula_errors = 0
    for trajectory_row, component_row in zip(
        trajectory.itertuples(index=False), components.itertuples(index=False), strict=True
    ):
        anchors = trajectory_anchors(int(trajectory_row.decision_ordinal))
        starts = pd.to_datetime(list(component_row.bar_start_timestamps_utc), utc=True)
        lengths = {
            len(starts),
            len(component_row.bar_open),
            len(component_row.bar_high),
            len(component_row.bar_low),
            len(component_row.bar_close),
            len(component_row.historical_relative_activity),
        }
        if lengths != {int(trajectory_row.decision_ordinal)}:
            anchor_errors += 1
            continue
        final_available = pd.Timestamp(trajectory_row.feature_available_timestamp_utc)
        expected_complete = True
        for role, anchor in zip(("e0", "e1", "e2"), anchors, strict=True):
            available, reason = anchor_formula_availability(anchor)
            actual_available = bool(getattr(trajectory_row, f"anchor_{role}_formula_available"))
            actual_reason = getattr(trajectory_row, f"anchor_{role}_unavailable_reason")
            complete = pd.Timestamp(starts[anchor - 1]) + pd.Timedelta(minutes=5)
            stored_complete = pd.Timestamp(
                getattr(trajectory_row, f"anchor_{role}_available_timestamp_utc")
            )
            if complete > final_available:
                future_anchor_rows += 1
            if complete != stored_complete:
                anchor_errors += 1
            if available != actual_available or (
                reason is not None and str(actual_reason) != str(reason)
            ):
                formula_errors += 1
            expected_complete = expected_complete and available
        if expected_complete != bool(trajectory_row.complete_trajectory_available):
            formula_errors += 1
    check(
        checks,
        "all_three_anchors_and_completed_bar_causality",
        bool(
            components_hash == EXPECTED_COMPONENT_HASH
            and anchor_errors == 0
            and future_anchor_rows == 0
            and formula_errors == 0
            and anchor_manifest["passed_causality"]
        ),
        rows_checked=len(trajectory),
        anchor_errors=anchor_errors,
        future_anchor_rows=future_anchor_rows,
        formula_errors=formula_errors,
    )
    check(
        checks,
        "bar_9_frozen_formula_unavailability",
        bool(
            not anchor_manifest["bar_9_formula_available"]
            and anchor_manifest["substitution_or_alternative_split_used"] is False
            and trajectory.loc[
                trajectory["decision_ordinal"].eq(12), "complete_trajectory_available"
            ]
            .eq(False)
            .all()
        ),
        unavailable_ordinal_12_rows=int(
            trajectory.loc[trajectory["decision_ordinal"].eq(12), "complete_trajectory_available"]
            .eq(False)
            .sum()
        ),
    )
    level_error = maximum_difference(
        trajectory[[f"{emotion}_current_level" for emotion in PRIMARY_BEHAVIOURS]].to_numpy(),
        panel[list(PRIMARY_BEHAVIOURS)].to_numpy(),
    )
    check(
        checks,
        "frozen_final_emotion_levels",
        level_error <= 1e-12,
        maximum_error=level_error,
    )
    formula_examples = trajectory_feature_values(1.0, 3.0, 2.0) == {
        "change": 1.0,
        "recent_change": -1.0,
        "acceleration": -3.0,
        "persistence": 0,
        "reversal": 1,
        "peak_displacement": -1.0,
    }
    check(
        checks,
        "trajectory_feature_formulas_and_fail_closed_non_materialisation",
        bool(
            formula_examples
            and feature_manifest["materialisation_status"]
            == "stopped_before_feature_calculation_due_to_anchor_support_preflight"
            and not any(
                feature in trajectory for feature in feature_manifest["all_trajectory_features"]
            )
        ),
    )
    expected_interactions = [specification[0] for specification in TRAJECTORY_INTERACTION_SPECS]
    check(
        checks,
        "six_trajectory_regime_interactions_and_development_clipping_contract",
        bool(
            expected_interactions == list(TRAJECTORY_INTERACTION_FEATURES)
            and [row["feature"] for row in interaction_manifest["interactions"]]
            == expected_interactions
            and interaction_manifest["development_only_clip_quantiles"] == [0.01, 0.99]
            and interaction_manifest["clip_bounds"] is None
            and interaction_manifest["status"] == "not_fitted_due_to_pre_model_support_blocker"
        ),
    )

    missingness = pd.read_csv(artifacts / "trajectory_missingness.csv")
    bar9 = missingness.loc[missingness["anchor_completed_bars"].eq(9)]
    other = missingness.loc[~missingness["anchor_completed_bars"].eq(9)]
    check(
        checks,
        "missingness_by_emotion_anchor_stock_month_checkpoint",
        bool(
            set(missingness["breakdown_type"]) == {"emotion_anchor", "stock", "month", "checkpoint"}
            and set(missingness["emotion"]) == set(PRIMARY_BEHAVIOURS)
            and bar9["missing_percent"].eq(100.0).all()
            and other["missing_percent"].eq(0.0).all()
        ),
        rows=len(missingness),
    )
    retention = float(trajectory["complete_trajectory_available"].mean())
    scoring_assessment = trajectory.loc[
        trajectory["scoring_eligible"] & trajectory["year"].eq(2025)
    ]
    complete_assessment = scoring_assessment.loc[
        scoring_assessment["complete_trajectory_available"]
    ]
    complete_development = trajectory.loc[
        trajectory["scoring_eligible"]
        & trajectory["year"].eq(2024)
        & trajectory["complete_trajectory_available"]
    ]
    assessment_retention = float(scoring_assessment["complete_trajectory_available"].mean())
    final_class_support = (
        complete_assessment["target_class"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    )
    final_stock_share = float(complete_assessment["symbol"].value_counts(normalize=True).max())
    final_class_share = float(
        complete_assessment["target_class"].value_counts(normalize=True).max()
    )
    independently_computed_gates = {
        "assessment_rows_at_least_5500": len(complete_assessment) >= 5_500,
        "assessment_sessions_at_least_140": complete_assessment["session"].nunique() >= 140,
        "all_20_stocks": complete_assessment["symbol"].nunique() == 20,
        "eight_assessment_months": complete_assessment["year_month"].nunique() == 8,
        "complete_trajectory_retention_at_least_95_percent": assessment_retention >= 0.95,
        "every_assessment_target_class_at_least_50": bool(final_class_support.ge(50).all()),
        "maximum_stock_share_at_most_10_percent": final_stock_share <= 0.10,
        "maximum_target_class_share_at_most_75_percent": final_class_share <= 0.75,
    }
    support = decision["trajectory_support"]
    check(
        checks,
        "support_and_concentration",
        bool(
            abs(retention - float(support["complete_trajectory_retention"])) <= 1e-12
            and abs(
                assessment_retention - float(support["assessment_complete_trajectory_retention"])
            )
            <= 1e-12
            and assessment_retention < 0.95
            and int(support["assessment_predecessor_rows"]) == len(scoring_assessment)
            and int(support["assessment_rows"]) == len(complete_assessment)
            and int(support["assessment_complete_trajectory_rows"]) == len(complete_assessment)
            and int(support["development_complete_trajectory_rows"]) == len(complete_development)
            and support["assessment_class_support"]
            == {str(key): int(value) for key, value in final_class_support.items()}
            and abs(float(support["maximum_assessment_stock_share"]) - final_stock_share) <= 1e-12
            and abs(float(support["maximum_assessment_class_share"]) - final_class_share) <= 1e-12
            and support["gates"] == independently_computed_gates
            and set(support["failed_gates"])
            == {
                "assessment_rows_at_least_5500",
                "complete_trajectory_retention_at_least_95_percent",
            }
        ),
        complete_trajectory_retention=retention,
        assessment_complete_trajectory_retention=assessment_retention,
        final_assessment_rows=len(complete_assessment),
        maximum_final_stock_share=final_stock_share,
        maximum_final_class_share=final_class_share,
    )

    bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    null = pd.read_csv(artifacts / "null_metrics.csv")
    check(
        checks,
        "session_bootstrap_and_trajectory_null_fail_closed",
        bool(
            len(bootstrap) == 9
            and bootstrap["planned_draw_count"].eq(50).all()
            and bootstrap["draw_count"].eq(0).all()
            and len(null) == 5
            and null["planned_draw_count"].eq(10).all()
            and null["draw_count"].eq(0).all()
            and configurations["bootstrap_draws_run"] == 0
            and configurations["trajectory_null_draws_run"] == 0
        ),
    )
    failed_support_gates = {name for name, passed in support["gates"].items() if not bool(passed)}
    expected_decision = (
        "blocked_insufficient_trajectory_support"
        if failed_support_gates
        else "blocked_reproducibility_or_audit_failure"
    )
    check(
        checks,
        "decision_logic",
        bool(
            decision["decision"] == expected_decision
            and decision["T1"]["status"] == "not_fitted"
            and decision["T2"]["status"] == "not_fitted"
            and configurations["actual_primary_fitted_model_count"] == 1
        ),
        actual=decision["decision"],
        expected=expected_decision,
    )
    check(
        checks,
        "fast_determinism_check",
        bool(
            determinism["passed"]
            and determinism["maximum_probability_difference"] <= 1e-12
            and determinism["final_decision_equal"]
        ),
    )
    passed = all(record["passed"] for record in checks)
    result = {
        **SAFETY_FLAGS,
        "audit": "independent_lightweight_audit_v0",
        "check_count": len(checks),
        "checks": checks,
        "passed": passed,
    }
    (artifacts / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--predecessor-primary", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument(
        "--behavioural-component-ledger",
        type=Path,
        default=DEFAULT_COMPONENT_LEDGER,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_artifacts(
        args.artifacts.expanduser().resolve(),
        predecessor_primary=args.predecessor_primary.expanduser().resolve(),
        component_ledger_path=args.behavioural_component_ledger.expanduser().resolve(),
    )
    print(canonical_json(result), end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
