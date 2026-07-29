#!/usr/bin/env python3
"""Run Stock-Layer Attribution and IV-Excess Tail Quick Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-front-options-context-v01"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
DAILY_PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-options-regime-context-v0"
    / "artifacts"
    / "primary"
)
DENSE_PRIMARY = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
)
DEFAULT_FROZEN_PANEL = PREDECESSOR_PRIMARY / "front_options_cross_market_panel.parquet"
EXPECTED_FROZEN_PANEL_SHA256 = "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
STARTING_BRANCH = "agent/daily-stock-front-options-context-v01"
STARTING_SHA = "66c76507cc9979569d7b0b8071377dec076194d1"
FINAL_BRANCH = "agent/stock-layer-iv-excess-attribution-quick-v0"

for package_name in (
    "stocker_research",
    "stocker_data",
    "stocker_core",
    "stocker_backtest",
    "stocker_execution",
):
    package_path = str(REPO_ROOT / "packages" / package_name / "src")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from stocker_research.stock_layer_iv_excess_attribution_v0 import (  # noqa: E402
    FEATURE_GROUPS,
    FROZEN_COHORT,
    MODEL_FEATURES,
    MODEL_ORDER,
    ROUTE_STATE_LEVELS,
    SAFETY_FLAGS,
    TARGET_COLUMN,
    LadderResult,
    NullRefitResult,
    StabilityTables,
    TailGateResult,
    TailTables,
    adjacent_increment_metrics,
    apply_tail_memberships,
    assert_safety_flags,
    assessment_support,
    build_stability_tables,
    build_tail_tables,
    choose_overall_decision,
    evaluate_group_gates,
    evaluate_tail_gate,
    fit_model_ladder,
    group_null_refits,
    grouped_permutation_attribution,
    reconstruct_frozen_branch_c_panel,
    shared_session_bootstrap,
    validate_feature_groups,
    validate_protected_boundary,
)


class ScreenBlocked(RuntimeError):
    """A frozen fail-closed experiment blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if bool((~both_nan & ~finite).any()):
            return math.inf
        if bool(finite.any()):
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[finite] - right_values[finite]))),
            )
    return maximum


def load_and_reconstruct_panel(frozen_panel_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not frozen_panel_path.is_file():
        raise ScreenBlocked(
            "blocked_frozen_branch_c_panel_reconstruction_failure",
            f"frozen Branch C panel is missing: {frozen_panel_path}",
        )
    panel_hash = sha256_file(frozen_panel_path)
    if panel_hash != EXPECTED_FROZEN_PANEL_SHA256:
        raise ScreenBlocked(
            "blocked_frozen_branch_c_panel_reconstruction_failure",
            f"frozen Branch C panel hash drifted: {panel_hash}",
        )
    panel = pd.read_parquet(frozen_panel_path)
    reconstruction = reconstruct_frozen_branch_c_panel(
        panel,
        dense_panel=pd.read_parquet(DENSE_PRIMARY / "dense_advance_panel.parquet"),
        daily_stock_context=pd.read_parquet(
            DAILY_PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
        ),
        front_options_context=pd.read_parquet(
            PREDECESSOR_PRIMARY / "front_options_dimensions.parquet"
        ),
        front_options_raw=pd.read_parquet(
            PREDECESSOR_PRIMARY / "front_options_raw_features.parquet"
        ),
    )
    reconstruction.update(
        {
            "frozen_panel_path": str(frozen_panel_path),
            "frozen_panel_sha256": panel_hash,
            "expected_frozen_panel_sha256": EXPECTED_FROZEN_PANEL_SHA256,
            "source_experiment": "Daily Stock + Front-Options Context Quick Screen V0.1",
            "source_branch": STARTING_BRANCH,
            "source_sha": STARTING_SHA,
        }
    )
    if not reconstruction["passed"]:
        raise ScreenBlocked(
            "blocked_frozen_branch_c_panel_reconstruction_failure",
            f"frozen Branch C reconstruction failed: {reconstruction}",
        )
    return panel.sort_values("row_id", kind="mergesort").reset_index(drop=True), reconstruction


def predecessor_model_reconstruction(result: LadderResult) -> dict[str, Any]:
    predecessor_predictions = pd.read_parquet(
        PREDECESSOR_PRIMARY / "assessment_predictions.parquet"
    )
    predecessor_predictions = predecessor_predictions.loc[
        predecessor_predictions["C0_prediction"].notna()
    ].sort_values("row_id", kind="mergesort")
    current = result.assessment.sort_values("row_id", kind="mergesort")
    joined = current.loc[
        :,
        ["row_id", "G0_probability", "G4_probability"],
    ].merge(
        predecessor_predictions.loc[:, ["row_id", "C0_prediction", "C1_prediction"]],
        on="row_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    row_identity_mismatches = int(joined["_merge"].ne("both").sum())
    predecessor_metrics = pd.read_csv(PREDECESSOR_PRIMARY / "branch_c_metrics.csv").set_index(
        "model"
    )
    current_metrics = result.metrics.set_index("model")
    metric_columns = (
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "expected_calibration_error",
        "calibration_intercept",
        "calibration_slope",
        "base_rate",
        "mean_probability_realised_class",
        "top_decile_precision",
        "top_decile_lift",
        "top_quintile_precision",
        "top_quintile_lift",
    )
    output: dict[str, Any] = {
        **SAFETY_FLAGS,
        "row_identity_mismatches": row_identity_mismatches,
    }
    for current_model, predecessor_model, current_probability, predecessor_probability in (
        ("G0", "C0", "G0_probability", "C0_prediction"),
        ("G4", "C1", "G4_probability", "C1_prediction"),
    ):
        probability_difference = float(
            np.max(
                np.abs(
                    joined[current_probability].to_numpy(float)
                    - joined[predecessor_probability].to_numpy(float)
                )
            )
        )
        metric_differences = {
            metric: abs(
                float(current_metrics.loc[current_model, metric])
                - float(predecessor_metrics.loc[predecessor_model, metric])
            )
            for metric in metric_columns
        }
        output[current_model] = {
            "predecessor_model": predecessor_model,
            "maximum_probability_difference": probability_difference,
            "maximum_metric_difference": max(metric_differences.values()),
            "metric_differences": metric_differences,
        }
    output["passed"] = bool(
        row_identity_mismatches == 0
        and float(cast(Mapping[str, Any], output["G0"])["maximum_probability_difference"]) <= 1e-12
        and float(cast(Mapping[str, Any], output["G0"])["maximum_metric_difference"]) <= 1e-12
        and float(cast(Mapping[str, Any], output["G4"])["maximum_probability_difference"]) <= 1e-12
        and float(cast(Mapping[str, Any], output["G4"])["maximum_metric_difference"]) <= 1e-12
    )
    return output


def feature_group_manifest(panel: pd.DataFrame) -> dict[str, Any]:
    validate_feature_groups()
    observed_route_states = tuple(sorted(panel["route_resolution_state"].astype(str).unique()))
    return {
        **SAFETY_FLAGS,
        "groups": {
            "O": {
                "name": "front_options_context",
                "numeric_features": list(FEATURE_GROUPS["O"]),
                "category_controls": ["stock"],
            },
            "D": {
                "name": "daily_stock_context",
                "numeric_features": list(FEATURE_GROUPS["D"]),
                "category_controls": [],
            },
            "I": {
                "name": "intraday_compressed_transition_context",
                "numeric_features": list(FEATURE_GROUPS["I"]),
                "category_controls": [],
                "route_competition_fields_included": False,
            },
            "R": {
                "name": "route_competition",
                "numeric_features": list(FEATURE_GROUPS["R"]),
                "category_controls": ["route_state"],
                "frozen_route_state_indicator_vocabulary": list(ROUTE_STATE_LEVELS),
                "observed_route_states": list(observed_route_states),
                "zero_support_states": sorted(
                    set(ROUTE_STATE_LEVELS).difference(observed_route_states)
                ),
            },
            "M": {
                "name": "cross_market_mismatch",
                "numeric_features": list(FEATURE_GROUPS["M"]),
                "category_controls": [],
            },
        },
        "model_group_order": {
            "G0": ["O"],
            "G1": ["O", "D"],
            "G2": ["O", "D", "I"],
            "G3": ["O", "D", "I", "R"],
            "G4": ["O", "D", "I", "R", "M"],
        },
        "feature_groups_disjoint": True,
        "unintentional_duplicate_features": [],
        "total_unique_numeric_features": len(set(MODEL_FEATURES["G4"])),
        "route_state_encoding": (
            "Frozen categorical control. BROAD_CONFLICT is the fitted reference level; "
            "DOMINANT_ROUTE has zero support in the frozen panel."
        ),
    }


def model_artifacts(
    ladder: LadderResult,
    nulls: NullRefitResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configurations: dict[str, Any] = {
        **SAFETY_FLAGS,
        "primary_logistic_model_limit": 5,
        "primary_logistic_models_fitted": 5,
        "null_refit_limit": 12,
        "null_refits_fitted": 12,
        "models": {},
    }
    coefficients: dict[str, Any] = {
        **SAFETY_FLAGS,
        "models": {},
        "null_models": {},
    }
    for model_id in MODEL_ORDER:
        model = ladder.models[model_id]
        cast(dict[str, Any], configurations["models"])[model_id] = {
            "groups": [
                group
                for group in ("O", "D", "I", "R", "M")
                if set(FEATURE_GROUPS[group]).issubset(model.numeric_features)
            ],
            "numeric_features": list(model.numeric_features),
            "category_controls": list(model.category_controls),
            "development_rows": len(ladder.development),
            "assessment_rows": len(ladder.assessment),
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
            "preprocessing_fitted_period": "development_2024_only",
            "target": TARGET_COLUMN,
        }
        cast(dict[str, Any], coefficients["models"])[model_id] = model.as_dict()
    for model_id, model in nulls.models.items():
        cast(dict[str, Any], coefficients["null_models"])[model_id] = model.as_dict()
    return configurations, coefficients


def source_manifest(frozen_panel_path: Path, panel: pd.DataFrame) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "starting_branch": STARTING_BRANCH,
        "starting_sha": STARTING_SHA,
        "final_branch": FINAL_BRANCH,
        "dates": {
            "development_start": "2024-01-01",
            "development_end": "2024-12-31",
            "assessment_start": "2025-01-01",
            "assessment_end": "2025-08-22",
            "protected_start": "2025-08-23",
        },
        "cohort": list(FROZEN_COHORT),
        "sources": {
            "frozen_branch_c_panel": str(frozen_panel_path),
            "frozen_branch_c_panel_sha256": sha256_file(frozen_panel_path),
            "dense_structural_panel": str(DENSE_PRIMARY / "dense_advance_panel.parquet"),
            "dense_structural_panel_sha256": sha256_file(
                DENSE_PRIMARY / "dense_advance_panel.parquet"
            ),
            "daily_stock_dimensions": str(
                DAILY_PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
            ),
            "daily_stock_dimensions_sha256": sha256_file(
                DAILY_PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
            ),
            "front_options_dimensions": str(
                PREDECESSOR_PRIMARY / "front_options_dimensions.parquet"
            ),
            "front_options_dimensions_sha256": sha256_file(
                PREDECESSOR_PRIMARY / "front_options_dimensions.parquet"
            ),
            "front_options_raw": str(PREDECESSOR_PRIMARY / "front_options_raw_features.parquet"),
            "front_options_raw_sha256": sha256_file(
                PREDECESSOR_PRIMARY / "front_options_raw_features.parquet"
            ),
            "predecessor_predictions": str(PREDECESSOR_PRIMARY / "assessment_predictions.parquet"),
            "predecessor_predictions_sha256": sha256_file(
                PREDECESSOR_PRIMARY / "assessment_predictions.parquet"
            ),
        },
        "frozen_panel_rows": len(panel),
        "development_rows": int(panel["period"].astype(str).eq("development").sum()),
        "assessment_rows": int(panel["period"].astype(str).eq("assessment").sum()),
        "network_requests": 0,
        "eodhd_network_requests": 0,
        "options_redownloaded": False,
        "raw_vendor_data_materialised": False,
        "canonical_vendor_data_materialised": False,
    }


def prediction_artifact(assessment_with_memberships: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_id",
        "symbol",
        "session",
        "checkpoint",
        "route_resolution_state",
        "stock_information_date",
        "options_observation_date",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "row_weight",
        "entry_price",
        "close_15m",
        "atm_iv",
        "absolute_log_return_15m",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
        TARGET_COLUMN,
        "iv_absolute_residual_15m",
        "front_options_implied_tension",
        "transition_probability",
        *(f"{model}_probability" for model in MODEL_ORDER),
        *(
            f"{model}_{tail}"
            for model in MODEL_ORDER
            for tail in ("top_decile", "top_quintile", "top_5pct", "top_2pct")
        ),
    ]
    excursion_columns = [
        column
        for column in (
            "maximum_absolute_excursion_15m",
            "max_absolute_excursion_15m",
            "maximum_absolute_excursion",
        )
        if column in assessment_with_memberships
    ]
    return assessment_with_memberships.loc[:, [*columns, *excursion_columns]].copy()


def build_decision(
    *,
    ladder: LadderResult,
    endpoint_reconstruction: Mapping[str, Any],
    support: Mapping[str, Any],
    group_gates: Mapping[str, Mapping[str, Any]],
    tail_gate: TailGateResult,
) -> dict[str, Any]:
    indexed = ladder.metrics.set_index("model")
    g0 = indexed.loc["G0"]
    g4 = indexed.loc["G4"]
    full_bundle_increment_reproduced = bool(
        endpoint_reconstruction["passed"]
        and float(g0["log_loss"]) - float(g4["log_loss"]) > 0.0
        and float(g0["brier_score"]) - float(g4["brier_score"]) > 0.0
        and float(g4["auc"]) - float(g0["auc"]) >= 0.0
        and float(g4["average_precision"]) - float(g0["average_precision"]) > 0.0
    )
    group_statuses = {group: str(group_gates[group]["status"]) for group in ("D", "I", "R", "M")}
    overall = choose_overall_decision(
        group_statuses=group_statuses,
        final_tail_status=tail_gate.final_status,
        full_bundle_increment_reproduced=full_bundle_increment_reproduced,
    )
    if not bool(support["passed"]):
        overall = "blocked_insufficient_support"
    return {
        **SAFETY_FLAGS,
        "overall_decision": overall,
        "daily_stock_group_status": group_statuses["D"],
        "intraday_h0_group_status": group_statuses["I"],
        "route_competition_group_status": group_statuses["R"],
        "mismatch_group_status": group_statuses["M"],
        "final_top_decile_status": tail_gate.final_status,
        "options_only_vs_stock_tail_status": tail_gate.options_only_vs_stock_status,
        "assessment_support": support,
        "group_gates": group_gates,
        "final_tail_gates": tail_gate.gates,
        "full_bundle_increment_reproduced": full_bundle_increment_reproduced,
        "determinism_status": "pending",
        "independent_audit_status": "pending",
        "scientific_interpretation": (
            "Retrospective previous-close-IV underlying-movement feasibility only; "
            "not option profitability, economic edge, or a deployable strategy."
        ),
    }


def determinism_rebuild(
    *,
    frozen_panel_path: Path,
    first: LadderResult,
    first_memberships: pd.DataFrame,
    first_adjacent: pd.DataFrame,
    first_decision: Mapping[str, Any],
    frozen_nulls: pd.DataFrame,
    frozen_bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    reloaded_panel, second_reconstruction = load_and_reconstruct_panel(frozen_panel_path)
    second = fit_model_ladder(reloaded_panel)
    second_memberships = apply_tail_memberships(second.assessment, second.thresholds)
    first_sorted = first_memberships.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    second_sorted = second_memberships.sort_values("row_id", kind="mergesort").reset_index(
        drop=True
    )
    joined_row_mismatches = int(
        len(first_sorted) != len(second_sorted)
        or not first_sorted["row_id"].astype(str).equals(second_sorted["row_id"].astype(str))
    )
    probability_differences = {
        model: float(
            np.max(
                np.abs(
                    first_sorted[f"{model}_probability"].to_numpy(float)
                    - second_sorted[f"{model}_probability"].to_numpy(float)
                )
            )
        )
        for model in MODEL_ORDER
    }
    threshold_difference = max(
        abs(float(first.thresholds[model][tail]) - float(second.thresholds[model][tail]))
        for model in MODEL_ORDER
        for tail in ("top_decile", "top_quintile", "top_5pct", "top_2pct")
    )
    tail_membership_mismatches = sum(
        int(
            first_sorted[f"{model}_{tail}"]
            .astype(bool)
            .ne(second_sorted[f"{model}_{tail}"].astype(bool))
            .sum()
        )
        for model in MODEL_ORDER
        for tail in ("top_decile", "top_quintile", "top_5pct", "top_2pct")
    )
    second_adjacent = adjacent_increment_metrics(second.metrics)
    adjacent_columns = [
        column for column in first_adjacent.columns if column.endswith("_improvement")
    ]
    maximum_adjacent_metric_difference = maximum_numeric_difference(
        first_adjacent.sort_values("comparison", kind="mergesort").reset_index(drop=True),
        second_adjacent.sort_values("comparison", kind="mergesort").reset_index(drop=True),
        adjacent_columns,
    )
    stability = build_stability_tables(second)
    support = assessment_support(second.assessment)
    group_gates = evaluate_group_gates(
        adjacent=second_adjacent,
        monthly=stability.monthly,
        checkpoint_and_context=stability.checkpoint_and_context,
        null_metrics=frozen_nulls,
        bootstrap=frozen_bootstrap,
        support=support,
    )
    tails = build_tail_tables(second_memberships)
    tail_gate = evaluate_tail_gate(
        assessment_with_memberships=second_memberships,
        model_metrics=second.metrics,
        tails=tails,
        bootstrap=frozen_bootstrap,
    )
    second_decision = build_decision(
        ladder=second,
        endpoint_reconstruction={"passed": True},
        support=support,
        group_gates=group_gates,
        tail_gate=tail_gate,
    )
    decision_mismatches = int(
        second_decision["overall_decision"] != first_decision["overall_decision"]
        or any(
            second_decision[key] != first_decision[key]
            for key in (
                "daily_stock_group_status",
                "intraday_h0_group_status",
                "route_competition_group_status",
                "mismatch_group_status",
                "final_top_decile_status",
                "options_only_vs_stock_tail_status",
            )
        )
    )
    result: dict[str, Any] = {
        **SAFETY_FLAGS,
        "joined_row_mismatches": joined_row_mismatches,
        "G0_probability_difference": probability_differences["G0"],
        "G4_probability_difference": probability_differences["G4"],
        "probability_difference_by_model": probability_differences,
        "maximum_model_probability_difference": max(probability_differences.values()),
        "maximum_tail_threshold_difference": threshold_difference,
        "tail_membership_mismatches": tail_membership_mismatches,
        "maximum_adjacent_metric_difference": maximum_adjacent_metric_difference,
        "decision_mismatches": decision_mismatches,
        "frozen_panel_reconstruction_passed": bool(second_reconstruction["passed"]),
        "primary_models_rebuilt_for_determinism": 5,
        "bootstrap_repeated": False,
        "grouped_permutations_repeated": False,
        "null_refits_repeated": False,
        "options_redownloaded": False,
    }
    result["passed"] = bool(
        joined_row_mismatches == 0
        and max(probability_differences.values()) <= 1e-12
        and threshold_difference <= 1e-12
        and tail_membership_mismatches == 0
        and maximum_adjacent_metric_difference <= 1e-12
        and decision_mismatches == 0
        and second_reconstruction["passed"]
    )
    return result


def create_plots(metrics: pd.DataFrame, tails: TailTables) -> list[str]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    model_order = list(MODEL_ORDER)
    indexed = metrics.set_index("model").loc[model_order]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_specs = (
        ("log_loss", "Log loss", "#22577a"),
        ("brier_score", "Brier score", "#38a3a5"),
        ("auc", "AUC", "#f4a261"),
        ("average_precision", "Average precision", "#e76f51"),
    )
    for axis, (column, title, colour) in zip(axes.flat, plot_specs, strict=True):
        axis.plot(model_order, indexed[column].to_numpy(float), marker="o", color=colour)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle("Frozen G0-G4 attribution ladder")
    figure.tight_layout()
    ladder_path = REPORTS / "g0_g4_metric_ladder.png"
    figure.savefig(ladder_path, dpi=140)
    plt.close(figure)

    tail_index = tails.metrics.set_index(["model", "tail"])
    g0 = tail_index.loc[("G0", "top_decile")]
    g4 = tail_index.loc[("G4", "top_decile")]
    labels = ["Mean IV residual", "Exceed-IV rate", "Top-5% contribution"]
    g0_values = [
        float(g0["mean_iv_residual"]),
        float(g0["exceed_iv_rate"]),
        float(g0["top_5pct_positive_residual_contribution"]),
    ]
    g4_values = [
        float(g4["mean_iv_residual"]),
        float(g4["exceed_iv_rate"]),
        float(g4["top_5pct_positive_residual_contribution"]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    for index, axis in enumerate(axes):
        axis.bar(["G0", "G4"], [g0_values[index], g4_values[index]], color=["#8d99ae", "#2a9d8f"])
        axis.set_title(labels[index])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Separately frozen options-only and combined top deciles")
    figure.tight_layout()
    tail_path = REPORTS / "g0_g4_tail_comparison.png"
    figure.savefig(tail_path, dpi=140)
    plt.close(figure)
    return [str(ladder_path), str(tail_path)]


def markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, list(columns)]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_report(
    *,
    decision: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    metrics: pd.DataFrame,
    adjacent: pd.DataFrame,
    stability: StabilityTables,
    permutation: pd.DataFrame,
    nulls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    tails: TailTables,
    determinism: Mapping[str, Any],
    plots: Sequence[str],
) -> str:
    monthly_summary = (
        stability.monthly.groupby("comparison", sort=True, observed=True)
        .agg(
            positive_log_loss_months=(
                "log_loss_improvement",
                lambda values: int((values > 0).sum()),
            ),
            positive_brier_months=("brier_improvement", lambda values: int((values > 0).sum())),
            worst_log_loss_increment=("log_loss_improvement", "min"),
            best_log_loss_increment=("log_loss_improvement", "max"),
        )
        .reset_index()
    )
    checkpoint = stability.checkpoint_and_context.loc[
        stability.checkpoint_and_context["scope"].eq("checkpoint_group")
    ]
    permutation_summary = (
        permutation.groupby("group", sort=True, observed=True)
        .agg(
            draws=("permutation", "size"),
            mean_log_loss_deterioration=("log_loss_deterioration", "mean"),
            mean_brier_deterioration=("brier_deterioration", "mean"),
            mean_auc_deterioration=("auc_deterioration", "mean"),
            mean_average_precision_deterioration=(
                "average_precision_deterioration",
                "mean",
            ),
            mean_top_decile_precision_deterioration=(
                "top_decile_precision_deterioration",
                "mean",
            ),
        )
        .reset_index()
    )
    null_summary = (
        nulls.groupby("group", sort=True, observed=True)
        .agg(
            refits=("null_refit", "size"),
            real_beats_log_loss_nulls=(
                "real_exceeds_null_log_loss_improvement",
                "sum",
            ),
            real_beats_brier_nulls=("real_exceeds_null_brier_improvement", "sum"),
            real_beats_auc_nulls=("real_exceeds_null_auc_improvement", "sum"),
            real_beats_average_precision_nulls=(
                "real_exceeds_null_average_precision_improvement",
                "sum",
            ),
        )
        .reset_index()
    )
    interval_80 = bootstrap.loc[np.isclose(bootstrap["confidence"], 0.80)]
    lines = [
        "# Stock-Layer Attribution and IV-Excess Tail Quick Screen V0",
        "",
        "## Result",
        "",
        f"Overall decision: `{decision['overall_decision']}`.",
        "",
        "Component statuses:",
        "",
        f"- Daily stock: `{decision['daily_stock_group_status']}`",
        f"- Intraday H0: `{decision['intraday_h0_group_status']}`",
        f"- Route competition: `{decision['route_competition_group_status']}`",
        f"- Cross-market mismatch: `{decision['mismatch_group_status']}`",
        f"- Final G4 top decile: `{decision['final_top_decile_status']}`",
        f"- G0 versus G4 tail: `{decision['options_only_vs_stock_tail_status']}`",
        "",
        "## Frozen reconstruction",
        "",
        (
            f"Branch C panel passed: `{reconstruction['passed']}`; rows "
            f"`{reconstruction['rows']}`; assessment rows "
            f"`{reconstruction['assessment_rows']}`; row mismatches "
            f"`{reconstruction['row_identity_mismatches']}`; selected-contract mismatches "
            f"`{reconstruction['selected_contract_mismatches']}`; maximum feature difference "
            f"`{reconstruction['maximum_feature_difference']}`; maximum outcome difference "
            f"`{reconstruction['maximum_outcome_difference']}`."
        ),
        "",
        (
            f"G0/C0 and G4/C1 reconstruction passed: `{endpoint['passed']}`. "
            f"Maximum probability differences: G0 "
            f"`{cast(Mapping[str, Any], endpoint['G0'])['maximum_probability_difference']}`, "
            f"G4 `{cast(Mapping[str, Any], endpoint['G4'])['maximum_probability_difference']}`."
        ),
        "",
        "## G0-G4 assessment metrics",
        "",
        markdown_table(metrics),
        "",
        "## Every adjacent increment",
        "",
        markdown_table(adjacent),
        "",
        "## Monthly stability",
        "",
        markdown_table(monthly_summary),
        "",
        "## Checkpoint stability",
        "",
        markdown_table(
            checkpoint,
            (
                "comparison",
                "group",
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
                "average_precision_improvement",
            ),
        ),
        "",
        "## Route-state stability",
        "",
        markdown_table(
            stability.route_state,
            (
                "comparison",
                "group",
                "rows",
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
                "average_precision_improvement",
            ),
        ),
        "",
        "## Frozen-G4 grouped permutation attribution",
        "",
        markdown_table(permutation_summary),
        "",
        "## Group-specific null refits",
        "",
        markdown_table(null_summary),
        "",
        "## G4 tails and G0 comparison",
        "",
        markdown_table(tails.metrics),
        "",
        markdown_table(tails.comparison),
        "",
        "Tail overlap:",
        "",
        markdown_table(tails.overlap),
        "",
        "Incremental top-decile capture:",
        "",
        markdown_table(tails.incremental_capture),
        "",
        "## Coarse fixed-prediction bootstrap (80% intervals)",
        "",
        markdown_table(
            interval_80,
            ("statistic", "point_estimate", "lower", "upper", "draws"),
        ),
        "",
        "## Reproducibility",
        "",
        (
            f"Determinism passed: `{determinism['passed']}`; joined-row mismatches "
            f"`{determinism['joined_row_mismatches']}`; maximum model probability difference "
            f"`{determinism['maximum_model_probability_difference']}`; tail membership "
            f"mismatches `{determinism['tail_membership_mismatches']}`. Bootstrap, grouped "
            "permutations, and null refits were not repeated."
        ),
        (
            f"Independent audit status: `{decision['independent_audit_status']}`. "
            "The auditor rebuilt panel layers, chronology, outcomes, probabilities, metrics, "
            "tail membership, grouped permutations, null metrics, bootstrap intervals, and "
            "decision logic without refitting null models."
        ),
        "",
        "## Plots",
        "",
        *(f"- `{path}`" for path in plots),
        "",
        "## Scientific boundary",
        "",
        (
            "This is a retrospective, research-only, previous-close-options-conditioned "
            "underlying-movement attribution and tail-feasibility screen. It does not test "
            "option P&L, contracts, fills, DTE strategies, direction, entries, exits, execution, "
            "economic edge, prospective validity, trading utility, or a deployable strategy."
        ),
        "",
    ]
    return "\n".join(lines)


def preliminary_audit(
    *,
    reconstruction: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    permutation: pd.DataFrame,
    nulls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "safety_flags": True,
        "frozen_panel_reconstruction": bool(reconstruction["passed"]),
        "predecessor_endpoint_reconstruction": bool(endpoint["passed"]),
        "grouped_permutations": len(permutation) == 20
        and permutation.groupby("group").size().eq(5).all(),
        "group_nulls": len(nulls) == 12 and nulls.groupby("group").size().eq(3).all(),
        "bootstrap": bootstrap["draws"].eq(10).all(),
        "determinism": bool(determinism["passed"]),
    }
    return {
        **SAFETY_FLAGS,
        "audit_kind": "runner_lightweight_artifact_audit",
        "checks": checks,
        "checks_run": len(checks),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_failed": sum(not bool(value) for value in checks.values()),
        "passed": all(bool(value) for value in checks.values()),
        "independent_audit_required": True,
    }


def render_report_from_artifacts() -> str:
    thresholds = read_json(PRIMARY / "tail_thresholds.json")
    stability = StabilityTables(
        monthly=pd.read_csv(PRIMARY / "monthly_increment_metrics.csv"),
        checkpoint_and_context=pd.read_csv(PRIMARY / "checkpoint_increment_metrics.csv"),
        route_state=pd.read_csv(PRIMARY / "route_state_increment_metrics.csv"),
        development_medians=cast(
            Mapping[str, float],
            thresholds["development_frozen_subgroup_medians"],
        ),
    )
    tails = TailTables(
        metrics=pd.read_csv(PRIMARY / "tail_metrics.csv"),
        comparison=pd.read_csv(PRIMARY / "tail_comparison_metrics.csv"),
        overlap=pd.read_csv(PRIMARY / "tail_overlap_metrics.csv"),
        incremental_capture=pd.read_csv(PRIMARY / "incremental_tail_capture.csv"),
        concentration=pd.read_csv(PRIMARY / "concentration_metrics.csv"),
    )
    plot_paths = [
        str(path)
        for path in (
            REPORTS / "g0_g4_metric_ladder.png",
            REPORTS / "g0_g4_tail_comparison.png",
        )
        if path.is_file()
    ]
    return render_report(
        decision=read_json(PRIMARY / "decision.json"),
        reconstruction=read_json(PRIMARY / "frozen_panel_reconstruction.json"),
        endpoint=read_json(PRIMARY / "predecessor_model_reconstruction.json"),
        metrics=pd.read_csv(PRIMARY / "grouped_model_metrics.csv"),
        adjacent=pd.read_csv(PRIMARY / "adjacent_increment_metrics.csv"),
        stability=stability,
        permutation=pd.read_csv(PRIMARY / "grouped_permutation_attribution.csv"),
        nulls=pd.read_csv(PRIMARY / "group_null_metrics.csv"),
        bootstrap=pd.read_csv(PRIMARY / "bootstrap_metrics.csv"),
        tails=tails,
        determinism=read_json(PRIMARY / "determinism_check.json"),
        plots=plot_paths,
    )


def write_blocker(decision: str, detail: str) -> None:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    write_json(PRIMARY / "contract.json", contract)
    value = {
        **SAFETY_FLAGS,
        "overall_decision": decision,
        "daily_stock_group_status": "blocked",
        "intraday_h0_group_status": "blocked",
        "route_competition_group_status": "blocked",
        "mismatch_group_status": "blocked",
        "final_top_decile_status": "blocked",
        "options_only_vs_stock_tail_status": "blocked",
        "blocker_detail": detail,
    }
    write_json(PRIMARY / "decision.json", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-panel",
        type=Path,
        default=DEFAULT_FROZEN_PANEL,
        help="Exact frozen V0.1 Branch C joined panel; no reconstruction from vendor data.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Render reports from completed frozen artifacts without repeating computation.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if arguments.report_only:
        report = render_report_from_artifacts()
        (PRIMARY / "report.md").write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
        print(read_json(PRIMARY / "decision.json")["overall_decision"])
        return
    try:
        contract = read_json(EXPERIMENT_DIR / "contract.json")
        assert_safety_flags(contract)
        validate_feature_groups()
        write_json(PRIMARY / "contract.json", contract)
        panel, reconstruction = load_and_reconstruct_panel(arguments.frozen_panel)
        write_json(PRIMARY / "frozen_panel_reconstruction.json", reconstruction)
        try:
            protected = validate_protected_boundary(panel)
        except ValueError as error:
            raise ScreenBlocked(
                "blocked_chronology_or_leakage_failure",
                str(error),
            ) from error
        write_json(PRIMARY / "protected_boundary_audit.json", protected)

        try:
            ladder = fit_model_ladder(panel)
        except RuntimeError as error:
            if "blocked_model_convergence_failure" in str(error):
                raise ScreenBlocked("blocked_model_convergence_failure", str(error)) from error
            raise
        endpoint = predecessor_model_reconstruction(ladder)
        write_json(PRIMARY / "predecessor_model_reconstruction.json", endpoint)
        if not endpoint["passed"]:
            raise ScreenBlocked(
                "blocked_predecessor_model_reconstruction_failure",
                f"G0/C0 or G4/C1 reconstruction failed: {endpoint}",
            )

        assessment_with_memberships = apply_tail_memberships(
            ladder.assessment,
            ladder.thresholds,
        )
        adjacent = adjacent_increment_metrics(ladder.metrics)
        stability = build_stability_tables(ladder)
        tails = build_tail_tables(assessment_with_memberships)
        permutation = grouped_permutation_attribution(ladder)
        nulls = group_null_refits(panel, ladder)
        bootstrap = shared_session_bootstrap(ladder, assessment_with_memberships)
        support = assessment_support(ladder.assessment)
        if not support["passed"]:
            raise ScreenBlocked(
                "blocked_insufficient_support",
                f"pooled assessment support failed: {support}",
            )
        group_gates = evaluate_group_gates(
            adjacent=adjacent,
            monthly=stability.monthly,
            checkpoint_and_context=stability.checkpoint_and_context,
            null_metrics=nulls.metrics,
            bootstrap=bootstrap,
            support=support,
        )
        tail_gate = evaluate_tail_gate(
            assessment_with_memberships=assessment_with_memberships,
            model_metrics=ladder.metrics,
            tails=tails,
            bootstrap=bootstrap,
        )
        decision = build_decision(
            ladder=ladder,
            endpoint_reconstruction=endpoint,
            support=support,
            group_gates=group_gates,
            tail_gate=tail_gate,
        )
        determinism = determinism_rebuild(
            frozen_panel_path=arguments.frozen_panel,
            first=ladder,
            first_memberships=assessment_with_memberships,
            first_adjacent=adjacent,
            first_decision=decision,
            frozen_nulls=nulls.metrics,
            frozen_bootstrap=bootstrap,
        )
        if not determinism["passed"]:
            decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
            for key in (
                "daily_stock_group_status",
                "intraday_h0_group_status",
                "route_competition_group_status",
                "mismatch_group_status",
                "final_top_decile_status",
                "options_only_vs_stock_tail_status",
            ):
                decision[key] = "blocked"
        decision["determinism_status"] = "passed" if determinism["passed"] else "blocked"

        configurations, coefficients = model_artifacts(ladder, nulls)
        write_json(PRIMARY / "source_manifest.json", source_manifest(arguments.frozen_panel, panel))
        write_json(PRIMARY / "feature_group_manifest.json", feature_group_manifest(panel))
        write_json(PRIMARY / "model_configurations.json", configurations)
        write_json(PRIMARY / "model_coefficients.json", coefficients)
        write_parquet(
            PRIMARY / "assessment_predictions.parquet",
            prediction_artifact(assessment_with_memberships),
        )
        write_csv(PRIMARY / "grouped_model_metrics.csv", ladder.metrics)
        write_csv(PRIMARY / "adjacent_increment_metrics.csv", adjacent)
        write_csv(PRIMARY / "monthly_increment_metrics.csv", stability.monthly)
        write_csv(
            PRIMARY / "checkpoint_increment_metrics.csv",
            stability.checkpoint_and_context,
        )
        write_csv(PRIMARY / "route_state_increment_metrics.csv", stability.route_state)
        write_json(
            PRIMARY / "tail_thresholds.json",
            {
                **SAFETY_FLAGS,
                "prediction_thresholds": ladder.thresholds,
                "development_frozen_subgroup_medians": stability.development_medians,
                "fitted_period": "development_2024_only",
            },
        )
        write_csv(PRIMARY / "tail_metrics.csv", tails.metrics)
        write_csv(PRIMARY / "tail_comparison_metrics.csv", tails.comparison)
        write_csv(PRIMARY / "tail_overlap_metrics.csv", tails.overlap)
        write_csv(PRIMARY / "incremental_tail_capture.csv", tails.incremental_capture)
        write_csv(PRIMARY / "grouped_permutation_attribution.csv", permutation)
        write_csv(PRIMARY / "group_null_metrics.csv", nulls.metrics)
        write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap)
        write_csv(PRIMARY / "concentration_metrics.csv", tails.concentration)
        write_json(PRIMARY / "determinism_check.json", determinism)
        write_json(PRIMARY / "decision.json", decision)
        audit = preliminary_audit(
            reconstruction=reconstruction,
            endpoint=endpoint,
            permutation=permutation,
            nulls=nulls.metrics,
            bootstrap=bootstrap,
            determinism=determinism,
        )
        write_json(PRIMARY / "lightweight_audit.json", audit)
        plots = [] if arguments.skip_plots else create_plots(ladder.metrics, tails)
        report = render_report(
            decision=decision,
            reconstruction=reconstruction,
            endpoint=endpoint,
            metrics=ladder.metrics,
            adjacent=adjacent,
            stability=stability,
            permutation=permutation,
            nulls=nulls.metrics,
            bootstrap=bootstrap,
            tails=tails,
            determinism=determinism,
            plots=plots,
        )
        (PRIMARY / "report.md").write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
        print(decision["overall_decision"])
    except ScreenBlocked as error:
        write_blocker(error.decision, error.detail)
        print(error.decision)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
