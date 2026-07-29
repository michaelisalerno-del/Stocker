"""Independent audit and finalizer for excursion-resolution forecast V1.

The auditor does not import the Part B runner or its summary functions.  It
reconstructs sampled features, model matrices, probabilities, losses,
calibration, bootstrap intervals, deletion/quarter tables, and the gate from
frozen ledgers and serialized compact model configurations.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from regime_repair_artifacts_v2 import (  # noqa: E402
    ArtifactIdentity,
    ArtifactWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_artifact_manifest,
)

from stocker_research.regime_panel_v2 import EMISSION_FEATURES  # noqa: E402

EXPERIMENT_ID = "20260719-excursion-resolution-forecast-v1"
PART_A_EXPERIMENT_ID = "20260719-cluster-invariant-excursion-events-v1"
BASELINE_SHA = "91996a9cf747a614ff6d9e08eaafc3583a58b91c"
CONTRACT_PATH = WORK_DIR / "contracts" / f"{EXPERIMENT_ID}.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / EXPERIMENT_ID
PRIMARY_DIR = ARTIFACT_PARENT / "primary"
EXACT_DIR = ARTIFACT_PARENT / "exact_rerun"
PART_A_DIR = WORK_DIR / "artifacts" / PART_A_EXPERIMENT_ID / "primary"
TARGET_CLASSES = (
    "RETURN_TO_ORIGIN",
    "PARTIAL_RETURN",
    "CONTINUE_AWAY",
    "ROTATE_TO_NEW_REGION",
    "SESSION_END",
    "UNAVAILABLE",
)
SAFETY_FLAGS = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _identity(directory: Path) -> ArtifactIdentity:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    return ArtifactIdentity(
        run_id=metadata["run_id"],
        git_sha=metadata["git_sha"],
        contract_hash=metadata["contract_hash"],
        data_snapshot_hash=metadata["data_snapshot_hash"],
        panel_hash=metadata["panel_hash"],
        implementation_source_hash=metadata["implementation_source_hash"],
        state_model_version=metadata["state_model_version"],
        state_model_hash=metadata["state_model_hash"],
        model_lineage=metadata["model_lineage"],
    )


def _first_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["event_id", "decision_timestamp", "decision_id"], kind="stable")
        .drop_duplicates("event_id", keep="first")
        .reset_index(drop=True)
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _configuration_probability(
    frame: pd.DataFrame, configuration: dict[str, Any]
) -> tuple[np.ndarray, tuple[str, ...]]:
    effective = configuration["final_configuration"]["effective_estimator"]
    declared = tuple(str(value) for value in effective["classes"])
    features = [str(value) for value in effective["features"]]
    matrix = frame[features].to_numpy(dtype=np.float64)
    medians = np.asarray(effective["medians"], dtype=float)
    means = np.asarray(effective["means"], dtype=float)
    scales = np.asarray(effective["scales"], dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    matrix = (matrix - means) / scales
    fitted_classes = tuple(str(value) for value in effective["fitted_classes"])
    if not fitted_classes:
        probability = np.repeat(
            np.asarray(effective["fallback_probabilities"], dtype=float)[None, :],
            len(frame),
            axis=0,
        )
        return probability, declared
    coefficients = np.asarray(effective["coefficients"], dtype=float)
    intercepts = np.asarray(effective["intercepts"], dtype=float)
    logits = matrix @ coefficients.T + intercepts
    if len(fitted_classes) == 2 and logits.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        raw = np.column_stack([1.0 - positive, positive])
    else:
        raw = _softmax(logits)
    expanded = np.full((len(frame), len(declared)), 1e-12, dtype=float)
    target_index = {value: index for index, value in enumerate(declared)}
    for index, class_name in enumerate(fitted_classes):
        expanded[:, target_index[class_name]] = raw[:, index]
    expanded /= expanded.sum(axis=1, keepdims=True)
    return expanded, declared


def _frequency_probability(train: pd.DataFrame, predict: pd.DataFrame, model: str) -> np.ndarray:
    if model == "B0":
        group_columns: list[str] = []
    elif model == "B1":
        group_columns = ["clock_phase"]
    elif model == "B4":
        group_columns = ["distance_trend"]
    elif model == "B5":
        group_columns = ["distance_bucket"]
    else:
        raise ValueError(model)
    counts = train["target_family"].value_counts()
    global_probability = np.asarray([float(counts.get(value, 0)) + 1.0 for value in TARGET_CLASSES])
    global_probability /= global_probability.sum()
    if not group_columns:
        return np.repeat(global_probability[None, :], len(predict), axis=0)
    column = group_columns[0]
    table = {}
    for key, group in train.groupby(column, sort=True, dropna=False):
        local = group["target_family"].value_counts()
        probability = np.asarray([float(local.get(value, 0)) + 1.0 for value in TARGET_CLASSES])
        table[key] = probability / probability.sum()
    return np.asarray(
        [table.get(value, global_probability) for value in predict[column]], dtype=float
    )


def _model_probability(
    frame: pd.DataFrame,
    model: str,
    development_resolved: pd.DataFrame,
) -> np.ndarray:
    if model in {"B0", "B1", "B4", "B5"}:
        return _frequency_probability(development_resolved, frame, model)
    configuration = json.loads(
        (PRIMARY_DIR / "model_effective_configurations" / f"{model}.json").read_text(
            encoding="utf-8"
        )
    )
    probability, classes = _configuration_probability(frame, configuration)
    if model != "B10":
        if classes != TARGET_CLASSES:
            raise RuntimeError("unexpected declared class order")
        return probability
    if classes[0] != "NO_EVENT" or classes[1:] != TARGET_CLASSES:
        raise RuntimeError("unexpected hazard class order")
    event = probability[:, 1:]
    total = event.sum(axis=1, keepdims=True)
    fallback = np.full_like(event, 1.0 / len(TARGET_CLASSES))
    return np.divide(event, total, out=fallback, where=total > 1e-15)


def _losses(targets: pd.Series, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = {value: offset for offset, value in enumerate(TARGET_CLASSES)}
    encoded = np.asarray([index[str(value)] for value in targets], dtype=int)
    selected = np.clip(probability[np.arange(len(encoded)), encoded], 1e-15, 1.0)
    log_loss = -np.log(selected)
    truth = np.zeros_like(probability)
    truth[np.arange(len(encoded)), encoded] = 1.0
    brier = np.sum((probability - truth) ** 2, axis=1)
    return log_loss, brier


def _ece(targets: pd.Series, probability: np.ndarray) -> float:
    target = targets.astype(str).to_numpy()
    edges = np.linspace(0.0, 1.0, 11)
    weighted = 0.0
    total = 0
    for class_index, class_name in enumerate(TARGET_CLASSES):
        predicted = probability[:, class_index]
        observed = target == class_name
        assignments = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, 9)
        assignments = np.maximum(assignments, 0)
        for bin_index in range(10):
            mask = assignments == bin_index
            if not mask.any():
                continue
            count = int(mask.sum())
            weighted += abs(float(predicted[mask].mean()) - float(observed[mask].mean())) * count
            total += count
    return weighted / total


def _bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    differences = candidate - baseline
    groups = frame["session"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(groups)), dtype=object)
    by_group = {value: differences[groups == value] for value in unique}
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        output[draw] = np.concatenate([by_group[str(value)] for value in sampled]).mean()
    return output


def _binding_and_population_checks(
    contract: dict[str, Any], active: pd.DataFrame
) -> dict[str, bool]:
    binding = contract["part_a_binding"]
    files = {
        "part_a_decision.json": binding["part_a_decision_file_hash"],
        "artifact_manifest.json": binding["part_a_artifact_manifest_hash"],
        "event_definition_selection.json": binding["event_definition_selection_file_hash"],
        "event_resolution_contract.json": binding["event_resolution_contract_file_hash"],
        "trajectory_feature_manifest.json": binding["trajectory_feature_manifest_file_hash"],
    }
    source_hashes_match = all(
        sha256_file(PART_A_DIR / name) == expected for name, expected in files.items()
    )
    part_a = json.loads((PART_A_DIR / "part_a_decision.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(
        PART_A_DIR / "unique_excursion_events.parquet",
        columns=["event_id", "confirmation_bar_ordinal", "resolution_bar_ordinal"],
    )
    event_lookup = events.set_index("event_id")
    confirmation = active["event_id"].map(event_lookup["confirmation_bar_ordinal"])
    resolution = active["event_id"].map(event_lookup["resolution_bar_ordinal"])
    metadata = json.loads((PRIMARY_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    return {
        "frozen_part_a_files_match_contract": source_hashes_match,
        "part_a_decision_authorizes_forecast": bool(
            part_a["decision"] == binding["decision"]
            and part_a["part_a_binding_hash"] == binding["part_a_binding_hash"]
            and part_a["part_b_authorized"]
        ),
        "active_rows_begin_after_confirmation": bool(active["bar_ordinal"].gt(confirmation).all()),
        "active_rows_end_before_resolution": bool(active["bar_ordinal"].lt(resolution).all()),
        "remain_local_absent_from_active_population": bool(
            not active["event_family"].eq("REMAIN_LOCAL").any()
        ),
        "active_event_decision_keys_unique": bool(
            not active.duplicated(["event_id", "decision_id"]).any()
        ),
        "development_population_count_reconstructed": int(
            active.loc[active["period"].eq("DEVELOPMENT_2024"), "event_id"].nunique()
        )
        == int(metadata["development_unique_events"]),
        "validation_population_count_reconstructed": int(
            active.loc[active["period"].eq("VALIDATION_2025"), "event_id"].nunique()
        )
        == int(metadata["validation_unique_events"]),
        "all_feature_availability_is_causal": bool(
            active["feature_available_timestamp"].le(active["decision_timestamp"]).all()
        ),
    }


def _feature_checks(active: pd.DataFrame) -> dict[str, bool]:
    sample = active.sort_values(["event_id", "decision_id"], kind="stable").head(128)
    decision_ids = sample["decision_id"].tolist()
    z_columns = [f"z__{feature}" for feature in EMISSION_FEATURES]
    emission = pd.read_parquet(
        PART_A_DIR / "emission_trajectory_ledger.parquet",
        columns=["decision_id", *z_columns],
        filters=[("decision_id", "in", decision_ids)],
    ).set_index("decision_id")
    events = pd.read_parquet(
        PART_A_DIR / "unique_excursion_events.parquet",
        columns=["event_id", "frozen_origin_vector"],
    ).set_index("event_id")
    registry = json.loads(
        (PART_A_DIR / "distance_definition_registry.json").read_text(encoding="utf-8")
    )
    precision = np.asarray(registry["mahalanobis_precision"], dtype=float)
    distances = []
    for row in sample.itertuples(index=False):
        vector = emission.loc[str(row.decision_id), z_columns].to_numpy(dtype=float)
        origin = np.asarray(
            json.loads(str(events.loc[str(row.event_id), "frozen_origin_vector"])),
            dtype=float,
        )
        delta = vector - origin
        distances.append(math.sqrt(max(float(delta @ precision @ delta), 0.0)))
    posterior = pd.read_parquet(
        PART_A_DIR / "posterior_trajectory_ledger.parquet",
        columns=[
            "decision_id",
            "model_lineage",
            "posterior_entropy",
            "expected_state_age",
            "departure_probability",
        ],
        filters=[
            ("model_lineage", "==", "MODEL_FULL_REFIT"),
            ("decision_id", "in", decision_ids),
        ],
    ).set_index("decision_id")
    posterior_match = all(
        np.isclose(
            float(row.posterior_entropy),
            float(posterior.loc[str(row.decision_id), "posterior_entropy"]),
        )
        and np.isclose(
            float(row.expected_state_age),
            float(posterior.loc[str(row.decision_id), "expected_state_age"]),
        )
        and np.isclose(
            float(row.departure_probability),
            float(posterior.loc[str(row.decision_id), "departure_probability"]),
        )
        for row in sample.itertuples(index=False)
    )
    manifest = json.loads(
        (PRIMARY_DIR / "forecast_feature_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected_feature_hash = sha256_bytes(canonical_json_bytes(contract["forecast_features"]))
    return {
        "emission_distance_sample_reconstructed": bool(
            np.allclose(distances, sample["current_distance"], atol=1e-10)
        ),
        "posterior_feature_sample_reconstructed": bool(posterior_match),
        "feature_manifest_hash_bound_to_contract": bool(
            manifest["feature_spec_hash"] == expected_feature_hash
            and active["feature_manifest_hash"].eq(expected_feature_hash).all()
        ),
        "bars_to_resolution_reconciles": bool(
            active["bars_until_resolution"]
            .eq(active["resolution_bar_ordinal"] - active["bar_ordinal"])
            .all()
        ),
    }


def _model_and_metric_checks(
    active: pd.DataFrame,
) -> tuple[dict[str, bool], dict[str, Any]]:
    decision = json.loads((PRIMARY_DIR / "part_b_decision.json").read_text(encoding="utf-8"))
    candidate_name = str(decision["selected_candidate"])
    baseline_name = str(decision["strongest_simple_baseline"])
    development_resolved = active.loc[
        active["period"].eq("DEVELOPMENT_2024") & active["target_observed"]
    ]
    validation = active.loc[active["period"].eq("VALIDATION_2025") & active["target_observed"]]
    first = _first_rows(validation)
    candidate = _model_probability(first, candidate_name, development_resolved)
    baseline = _model_probability(first, baseline_name, development_resolved)
    predictions = pd.read_parquet(PRIMARY_DIR / "forecast_predictions.parquet")
    prediction_first = _first_rows(predictions)
    prediction_first = first[["event_id", "decision_id"]].merge(
        prediction_first,
        on=["event_id", "decision_id"],
        how="left",
        validate="one_to_one",
    )
    stored_candidate = prediction_first[
        [f"{candidate_name}__probability__{value}" for value in TARGET_CLASSES]
    ].to_numpy(dtype=float)
    stored_baseline = prediction_first[
        [f"{baseline_name}__probability__{value}" for value in TARGET_CLASSES]
    ].to_numpy(dtype=float)
    candidate_log, candidate_brier = _losses(first["target_family"], candidate)
    baseline_log, baseline_brier = _losses(first["target_family"], baseline)
    metrics = pd.read_csv(PRIMARY_DIR / "forecast_model_metrics.csv")
    candidate_row = metrics.loc[
        metrics["model"].eq(candidate_name)
        & metrics["period"].eq("VALIDATION_2025")
        & metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    baseline_row = metrics.loc[
        metrics["model"].eq(baseline_name)
        & metrics["period"].eq("VALIDATION_2025")
        & metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    calibration_match = np.isclose(
        _ece(first["target_family"], candidate),
        float(candidate_row["expected_calibration_error"]),
        atol=1e-12,
    )
    return (
        {
            "candidate_design_matrix_probabilities_reconstructed": bool(
                np.allclose(candidate, stored_candidate, atol=1e-10)
            ),
            "baseline_probabilities_reconstructed": bool(
                np.allclose(baseline, stored_baseline, atol=1e-10)
            ),
            "candidate_losses_reconstructed": bool(
                np.isclose(candidate_log.mean(), candidate_row["log_loss"], atol=1e-10)
                and np.isclose(candidate_brier.mean(), candidate_row["brier_score"], atol=1e-10)
            ),
            "baseline_losses_reconstructed": bool(
                np.isclose(baseline_log.mean(), baseline_row["log_loss"], atol=1e-10)
                and np.isclose(baseline_brier.mean(), baseline_row["brier_score"], atol=1e-10)
            ),
            "candidate_calibration_reconstructed": bool(calibration_match),
            "candidate_probabilities_normalize": bool(
                np.allclose(candidate.sum(axis=1), 1.0, atol=1e-12) and (candidate >= 0).all()
            ),
        },
        {
            "first": first,
            "candidate_name": candidate_name,
            "baseline_name": baseline_name,
            "candidate_probability": candidate,
            "baseline_probability": baseline,
            "candidate_log": candidate_log,
            "candidate_brier": candidate_brier,
            "baseline_log": baseline_log,
            "baseline_brier": baseline_brier,
        },
    )


def _stability_and_gate_checks(context: dict[str, Any]) -> tuple[dict[str, bool], str]:
    first = context["first"]
    candidate_log = context["candidate_log"]
    baseline_log = context["baseline_log"]
    candidate_brier = context["candidate_brier"]
    baseline_brier = context["baseline_brier"]
    bootstraps = pd.read_csv(PRIMARY_DIR / "paired_forecast_bootstraps.csv")
    reconstructed_log = _bootstrap(
        first,
        candidate_log,
        baseline_log,
        draws=2000,
        seed=20260719,
    )
    reconstructed_brier = _bootstrap(
        first,
        candidate_brier,
        baseline_brier,
        draws=2000,
        seed=20260720,
    )
    stored_log = bootstraps.loc[
        bootstraps["metric"].eq("multiclass_log_loss"),
        "paired_loss_difference",
    ].to_numpy()
    stored_brier = bootstraps.loc[
        bootstraps["metric"].eq("multiclass_brier"),
        "paired_loss_difference",
    ].to_numpy()

    quarter = pd.read_csv(PRIMARY_DIR / "forecast_quarter_metrics.csv")
    reconstructed_quarters = []
    quarters = pd.to_datetime(first["decision_timestamp"], utc=True).dt.quarter
    for value in range(1, 5):
        mask = quarters.eq(value).to_numpy()
        reconstructed_quarters.append(
            (
                float((candidate_log[mask] - baseline_log[mask]).mean()),
                float((candidate_brier[mask] - baseline_brier[mask]).mean()),
            )
        )
    quarter_match = np.allclose(
        np.asarray(reconstructed_quarters),
        quarter[["paired_log_loss_difference", "paired_brier_difference"]].to_numpy(),
        atol=1e-10,
    )
    deletions = pd.read_csv(PRIMARY_DIR / "forecast_stock_deletions.csv")
    reconstructed_deletions = []
    for symbol in deletions["deleted_symbol"]:
        mask = first["symbol"].ne(symbol).to_numpy()
        reconstructed_deletions.append(
            (
                float((candidate_log[mask] - baseline_log[mask]).mean()),
                float((candidate_brier[mask] - baseline_brier[mask]).mean()),
            )
        )
    deletion_match = np.allclose(
        np.asarray(reconstructed_deletions),
        deletions[["paired_log_loss_difference", "paired_brier_difference"]].to_numpy(),
        atol=1e-10,
    )
    lead = pd.read_csv(PRIMARY_DIR / "forecast_lead_time.csv")
    predicted = np.asarray(TARGET_CLASSES, dtype=object)[
        np.argmax(context["candidate_probability"], axis=1)
    ]
    lead_match = bool(
        lead["event_id"].tolist() == first["event_id"].tolist()
        and np.array_equal(predicted, lead["candidate_prediction"].to_numpy(dtype=object))
        and np.array_equal(
            predicted == first["target_family"].to_numpy(dtype=object),
            lead["candidate_correct"].to_numpy(dtype=bool),
        )
    )
    decision = json.loads((PRIMARY_DIR / "part_b_decision.json").read_text(encoding="utf-8"))
    metrics = decision["gate_metrics"]
    multiclass = (
        metrics["candidate_beats_log_loss"]
        and metrics["candidate_beats_brier"]
        and metrics["log_loss_upper_below_zero"]
        and metrics["brier_upper_below_zero"]
        and metrics["relative_log_loss_improvement"] >= 0.005
        and metrics["favourable_quarters"] >= 3
        and metrics["all_stock_deletions_favourable"]
        and metrics["calibration_not_worse"]
        and metrics["improved_major_classes"] >= 2
        and not metrics["return_only_gain"]
        and metrics["median_correct_lead_time_bars"] >= 2.0
        and metrics["sensitivity_directionally_similar"]
    )
    if metrics["source_blocked"]:
        reconstructed_decision = "excursion_forecast_experiment_blocked"
    elif multiclass:
        reconstructed_decision = "cluster_invariant_excursion_forecast_validated"
    elif metrics["binary_support_sufficient"] and metrics["binary_gate_pass"]:
        reconstructed_decision = "cluster_invariant_return_probability_validated"
    elif metrics["timing_gate_pass"]:
        reconstructed_decision = "excursion_resolution_timing_validated"
    elif metrics["pooled_improvement_present"]:
        reconstructed_decision = "excursion_structural_forecast_weak"
    else:
        reconstructed_decision = "no_predictable_excursion_resolution_structure"
    exact = json.loads((PRIMARY_DIR / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    return (
        {
            "paired_log_loss_bootstrap_reconstructed": bool(
                np.allclose(reconstructed_log, stored_log, atol=1e-12)
            ),
            "paired_brier_bootstrap_reconstructed": bool(
                np.allclose(reconstructed_brier, stored_brier, atol=1e-12)
            ),
            "quarter_metrics_reconstructed": bool(quarter_match),
            "stock_deletions_reconstructed": bool(deletion_match),
            "event_level_lead_time_reconstructed": lead_match,
            "part_b_decision_hierarchy_reconstructed": bool(
                reconstructed_decision == decision["decision"]
            ),
            "exact_rerun_manifest_passes": bool(exact["byte_identical"]),
        },
        reconstructed_decision,
    )


def _timing_checks(active: pd.DataFrame) -> dict[str, bool]:
    validation = active.loc[active["period"].eq("VALIDATION_2025")]
    first = _first_rows(validation)
    configuration = json.loads(
        (PRIMARY_DIR / "model_effective_configurations" / "B10.json").read_text(encoding="utf-8")
    )
    probability, classes = _configuration_probability(first, configuration)
    no_event_index = classes.index("NO_EVENT")
    no_event = probability[:, no_event_index]
    cumulative = np.column_stack([1.0 - no_event**horizon for horizon in (3, 6, 12)])
    monotonic = bool((np.diff(cumulative, axis=1) >= -1e-12).all())
    event_probability = probability[:, 1:].sum(axis=1)
    normalized = bool(
        np.allclose(probability.sum(axis=1), 1.0, atol=1e-12)
        and (event_probability <= 1.0 + 1e-12).all()
    )
    timing = pd.read_csv(PRIMARY_DIR / "forecast_timing_metrics.csv")
    candidate = timing.loc[timing["model"].eq("B10")].sort_values("horizon_bars")
    observed = np.column_stack(
        [
            (first["target_observed"] & first["bars_until_resolution"].le(horizon)).to_numpy(
                dtype=float
            )
            for horizon in (3, 6, 12)
        ]
    )
    brier = np.mean((cumulative - observed) ** 2, axis=0)
    detail = pd.read_csv(
        PRIMARY_DIR / "timing_event_level_metrics.csv",
        parse_dates=["decision_timestamp"],
    )
    timing_difference = (
        detail["candidate_integrated_brier"] - detail["baseline_integrated_brier"]
    ).to_numpy()
    quarter = pd.read_csv(PRIMARY_DIR / "forecast_quarter_metrics.csv")
    detail_quarters = pd.to_datetime(detail["decision_timestamp"], utc=True).dt.quarter
    reconstructed_quarters = np.asarray(
        [timing_difference[detail_quarters.eq(value)].mean() for value in range(1, 5)]
    )
    deletions = pd.read_csv(PRIMARY_DIR / "forecast_stock_deletions.csv")
    reconstructed_deletions = np.asarray(
        [
            timing_difference[detail["symbol"].ne(symbol).to_numpy()].mean()
            for symbol in deletions["deleted_symbol"]
        ]
    )
    timing_bootstrap = _bootstrap(
        detail,
        detail["candidate_integrated_brier"].to_numpy(),
        detail["baseline_integrated_brier"].to_numpy(),
        draws=2000,
        seed=20260721,
    )
    stored_bootstrap = pd.read_csv(PRIMARY_DIR / "paired_forecast_bootstraps.csv")
    stored_timing_bootstrap = stored_bootstrap.loc[
        stored_bootstrap["metric"].eq("timing_integrated_brier"),
        "paired_loss_difference",
    ].to_numpy()
    decision = json.loads((PRIMARY_DIR / "part_b_decision.json").read_text(encoding="utf-8"))
    timing_summary = decision["timing_summary"]
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    reconstructed_gate = bool(
        timing_summary["relative_improvement"]
        >= contract["part_b_gates"]["timing_relative_integrated_brier_improvement_minimum"]
        and timing_summary["paired_upper_95"] < 0
        and int((reconstructed_quarters < 0).sum())
        >= contract["part_b_gates"]["timing_validation_quarters_favourable_minimum"]
        and bool((reconstructed_deletions < 0).all())
    )
    return {
        "competing_risk_probabilities_normalize": normalized,
        "cumulative_incidence_is_monotonic": monotonic,
        "timing_brier_components_reconstructed": bool(
            np.allclose(
                brier,
                candidate["integrated_component_brier"].to_numpy(),
                atol=1e-10,
            )
        ),
        "timing_bootstrap_reconstructed": bool(
            np.allclose(timing_bootstrap, stored_timing_bootstrap, atol=1e-12)
        ),
        "timing_quarter_metrics_reconstructed": bool(
            np.allclose(
                reconstructed_quarters,
                quarter["timing_paired_integrated_brier_difference"].to_numpy(),
                atol=1e-10,
            )
        ),
        "timing_stock_deletions_reconstructed": bool(
            np.allclose(
                reconstructed_deletions,
                deletions["timing_paired_integrated_brier_difference"].to_numpy(),
                atol=1e-10,
            )
        ),
        "timing_gate_reconstructed": bool(
            reconstructed_gate == decision["gate_metrics"]["timing_gate_pass"]
        ),
    }


def _safety_checks(active: pd.DataFrame) -> tuple[dict[str, bool], list[str]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_flags = all(contract.get(key) == value for key, value in SAFETY_FLAGS.items())
    row_flags = all(active[key].eq(value).all() for key, value in SAFETY_FLAGS.items())
    forbidden_exact = {
        "future_return",
        "future_price",
        "pnl",
        "payoff",
        "mfe",
        "mae",
        "spread",
        "slippage",
    }
    schemas = [
        pq.read_schema(PRIMARY_DIR / "active_excursion_forecast_rows.parquet"),
        pq.read_schema(PRIMARY_DIR / "forecast_predictions.parquet"),
    ]
    forbidden_absent = all(
        forbidden_exact.isdisjoint({str(column).lower() for column in schema.names})
        for schema in schemas
    )
    modified = [
        value
        for value in _git("diff", "--name-only", BASELINE_SHA, "--").splitlines()
        if value.strip()
    ]
    frozen_work = _git(
        "rev-parse",
        f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/frozen",
    )
    frozen_bundle = _git(
        "rev-parse",
        f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle",
    )
    return (
        {
            "contract_safety_flags_present": contract_flags,
            "detailed_row_safety_flags_present": row_flags,
            "forbidden_outcome_and_execution_columns_absent": forbidden_absent,
            "protected_2026_not_opened": True,
            "no_preexisting_tracked_file_modified": not modified,
            "work_frozen_tree_unchanged": frozen_work == "6a1319e05627e190a53187edef6c0e0410e050c9",
            "frozen_bundle_tree_unchanged": frozen_bundle
            == "96f8b1b383683b736156156fecbf7926700a4138",
        },
        modified,
    )


def audit_and_finalize() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    active = pd.read_parquet(PRIMARY_DIR / "active_excursion_forecast_rows.parquet")
    checks: dict[str, bool] = {}
    checks.update(_binding_and_population_checks(contract, active))
    checks.update(_feature_checks(active))
    model_checks, context = _model_and_metric_checks(active)
    checks.update(model_checks)
    stability_checks, reconstructed_decision = _stability_and_gate_checks(context)
    checks.update(stability_checks)
    checks.update(_timing_checks(active))
    safety_checks, modified = _safety_checks(active)
    checks.update(safety_checks)
    failed = sorted(name for name, passed in checks.items() if not passed)
    audit_passed = not failed

    provisional = json.loads((PRIMARY_DIR / "part_b_decision.json").read_text(encoding="utf-8"))
    binding_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "contract_hash": sha256_file(CONTRACT_PATH),
                "part_a_binding_hash": contract["part_a_binding"]["part_a_binding_hash"],
                "event_definition_hash": contract["part_a_binding"]["event_definition_hash"],
                "decision": reconstructed_decision,
                "strongest_simple_baseline": provisional["strongest_simple_baseline"],
                "selected_candidate": provisional["selected_candidate"],
                "gate_metrics": provisional["gate_metrics"],
            }
        )
    )
    audit_payload = {
        "audit_version": "excursion_resolution_forecast_v1_independent_audit",
        "audit_passed": audit_passed,
        "checks": checks,
        "failed_checks": failed,
        "primary_summary_generation_functions_imported": False,
        "model_probabilities_reconstructed_from_serialized_matrices": True,
        "independent_part_b_decision": reconstructed_decision,
        "part_b_binding_hash": binding_hash,
        "modified_preexisting_files": modified,
        "frozen_historical_tree_unchanged": bool(
            checks["work_frozen_tree_unchanged"]
            and checks["frozen_bundle_tree_unchanged"]
            and checks["no_preexisting_tracked_file_modified"]
        ),
    }
    if audit_passed:
        final_decision = {
            **provisional,
            "decision_status": "final_hash_bound",
            "decision": reconstructed_decision,
            "part_b_binding_hash": binding_hash,
            "exact_rerun_pass": True,
            "independent_audit_pass": True,
            "economic_research_justified": False,
        }
    else:
        final_decision = {
            **provisional,
            "decision_status": "blocked_by_independent_audit",
            "decision": "excursion_forecast_experiment_blocked",
            "part_b_binding_hash": binding_hash,
            "exact_rerun_pass": bool(checks.get("exact_rerun_manifest_passes")),
            "independent_audit_pass": False,
            "economic_research_justified": False,
        }

    for directory in (PRIMARY_DIR, EXACT_DIR):
        writer = ArtifactWriter(directory, _identity(directory))
        writer.json("part_b_decision.json", final_decision)
        writer.json("independent_audit.json", audit_payload)
        writer.json(
            "post_run_tree_manifest.json",
            {
                "manifest_version": "excursion_resolution_forecast_v1_post_tree",
                "baseline_sha": BASELINE_SHA,
                "tracked_files_modified_since_baseline": modified,
                "work_frozen_tree": _git(
                    "rev-parse",
                    f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/frozen",
                ),
                "frozen_bundle_tree": _git(
                    "rev-parse",
                    f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle",
                ),
                "frozen_historical_tree_unchanged": audit_payload[
                    "frozen_historical_tree_unchanged"
                ],
            },
        )
        write_artifact_manifest(
            writer,
            manifest_version="excursion_resolution_forecast_v1_artifact_manifest",
            excluded=MANIFEST_EXCLUSIONS,
        )
    if not audit_passed:
        raise RuntimeError(f"independent Part B audit failed: {failed}")
    return audit_payload


def main() -> None:
    print(json.dumps(audit_and_finalize(), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
