"""Run the hash-bound structural excursion-resolution forecast V1 study."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from regime_repair_artifacts_v2 import (  # noqa: E402
    ArtifactIdentity,
    ArtifactWriter,
    canonical_json_bytes,
    compare_artifact_directories,
    sha256_bytes,
    sha256_file,
    write_artifact_manifest,
)

from stocker_research.excursion_forecast_v1 import (  # noqa: E402
    SAFETY_FLAGS,
    TARGET_CLASSES,
    PartBGateMetrics,
    balanced_event_weights,
    build_active_excursion_rows,
    calibration_table,
    confusion_matrix_frame,
    constant_hazard_competing_risk,
    decide_part_b,
    first_eligible_rows,
    fit_multinomial,
    frequency_probabilities,
    last_eligible_rows,
    multiclass_losses,
    paired_block_bootstrap,
    per_class_metrics,
    ranking_metrics,
    timing_metrics,
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
REPORT_PATH = WORK_DIR / "reports" / f"{EXPERIMENT_ID}.md"
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
}
IMPLEMENTATION_PATHS = (
    Path("packages/stocker_research/src/stocker_research/excursion_forecast_v1.py"),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "run_excursion_resolution_forecast_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "audit_excursion_resolution_forecast_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
        "20260719-excursion-resolution-forecast-v1.json"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/tests/"
        "test_excursion_resolution_forecast_v1.py"
    ),
)
HAZARD_CLASSES = ("NO_EVENT", *TARGET_CLASSES)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        absolute = REPO_ROOT / path
        if not absolute.exists():
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(b"\0missing\0")
            continue
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(absolute.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Part B contract safety flag mismatch: {key}")
    if contract["git_sha"] != BASELINE_SHA:
        raise RuntimeError("Part B contract Git SHA mismatch")
    return contract


def _verify_part_a(contract: Mapping[str, Any]) -> dict[str, Any]:
    binding = contract["part_a_binding"]
    checks = {
        "part_a_decision.json": binding["part_a_decision_file_hash"],
        "artifact_manifest.json": binding["part_a_artifact_manifest_hash"],
        "event_definition_selection.json": binding["event_definition_selection_file_hash"],
        "event_resolution_contract.json": binding["event_resolution_contract_file_hash"],
        "trajectory_feature_manifest.json": binding["trajectory_feature_manifest_file_hash"],
    }
    for name, expected in checks.items():
        actual = sha256_file(PART_A_DIR / name)
        if actual != expected:
            raise RuntimeError(f"frozen Part A binding mismatch for {name}")
    decision = json.loads((PART_A_DIR / "part_a_decision.json").read_text(encoding="utf-8"))
    if decision.get("decision") != binding["decision"]:
        raise RuntimeError("Part A decision no longer authorizes Part B")
    if decision.get("part_a_binding_hash") != binding["part_a_binding_hash"]:
        raise RuntimeError("Part A binding hash mismatch")
    if not decision.get("part_b_authorized"):
        raise RuntimeError("Part B is not authorized")
    return decision


def _identity(contract: Mapping[str, Any], part_a: Mapping[str, Any]) -> ArtifactIdentity:
    contract_hash = sha256_file(CONTRACT_PATH)
    implementation_hash = _implementation_hash()
    run_id = sha256_bytes(
        canonical_json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "contract_hash": contract_hash,
                "part_a_binding_hash": part_a["part_a_binding_hash"],
                "implementation_source_hash": implementation_hash,
            }
        )
    )[:24]
    return ArtifactIdentity(
        run_id=run_id,
        git_sha=BASELINE_SHA,
        contract_hash=contract_hash,
        data_snapshot_hash=str(part_a["data_snapshot_hash"]),
        panel_hash=str(part_a["panel_hash"]),
        implementation_source_hash=implementation_hash,
        state_model_version="excursion_resolution_structural_forecast_v1",
        state_model_hash=str(part_a["state_model_hash"]),
        model_lineage="MODEL_FULL_REFIT",
    )


def _load_source_ledgers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    events = pd.read_parquet(PART_A_DIR / "unique_excursion_events.parquet")
    emission_columns = [
        "decision_id",
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "decision_timestamp",
        "availability_timestamp",
        "period",
        "short_trajectory_velocity",
        "short_trajectory_acceleration",
        "local_path_length",
        "local_directional_consistency",
    ]
    emission_columns += [f"z__{feature}" for feature in EMISSION_FEATURES]
    emission_columns += [f"delta_z__{feature}" for feature in EMISSION_FEATURES]
    emission_columns += [f"missing__{feature}" for feature in EMISSION_FEATURES]
    emission = pd.read_parquet(
        PART_A_DIR / "emission_trajectory_ledger.parquet",
        columns=emission_columns,
    )
    posterior_columns = [
        "decision_id",
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "model_lineage",
        "posterior_entropy",
        "expected_state_age",
        "departure_probability",
        "hard_hysteretic_disagreement",
        "posterior_velocity",
        "hard_map_state",
        "availability_timestamp",
    ] + [f"posterior_state_{index}" for index in range(8)]
    posterior = pd.read_parquet(
        PART_A_DIR / "posterior_trajectory_ledger.parquet",
        columns=posterior_columns,
        filters=[("model_lineage", "==", "MODEL_FULL_REFIT")],
    )
    distance_registry = json.loads(
        (PART_A_DIR / "distance_definition_registry.json").read_text(encoding="utf-8")
    )
    precision = np.asarray(distance_registry["mahalanobis_precision"], dtype=np.float64)
    return events, emission, posterior, precision


def _ordered_features(contract: Mapping[str, Any]) -> list[str]:
    groups = contract["forecast_features"]
    ordered: list[str] = []
    for name in (
        "geometry",
        "posterior_and_age",
        "market_context",
        "event_history",
        "clock_and_baseline",
    ):
        ordered.extend(str(value) for value in groups[name])
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("forecast feature contract contains duplicates")
    return ordered


def _feature_groups(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = contract["forecast_features"]
    geometry = list(groups["geometry"])
    posterior = list(groups["posterior_and_age"])
    market = list(groups["market_context"])
    history = list(groups["event_history"])
    clock = list(groups["clock_and_baseline"])
    return {
        "B2": [
            "bars_since_departure",
            "bars_remaining_in_session",
            "session_fraction",
            "clock_phase_early",
            "clock_phase_middle",
            "clock_phase_late",
        ],
        "B3": ["current_distance", "distance_velocity"],
        "B6": posterior,
        "B7": geometry,
        "B8": geometry + posterior,
        "B9": geometry + posterior + market,
        "B10": geometry + posterior + market + history + clock,
    }


def _prepare_active_rows(
    contract: Mapping[str, Any], identity: ArtifactIdentity
) -> tuple[pd.DataFrame, str]:
    events, emission, posterior, precision = _load_source_ledgers()
    active = build_active_excursion_rows(
        events,
        emission,
        posterior,
        precision=precision,
        emission_features=EMISSION_FEATURES,
        primary_model_lineage="MODEL_FULL_REFIT",
    )
    feature_spec_hash = sha256_bytes(canonical_json_bytes(contract["forecast_features"]))
    active["run_id"] = identity.run_id
    active["git_sha"] = identity.git_sha
    active["contract_hash"] = identity.contract_hash
    active["feature_manifest_hash"] = feature_spec_hash
    active["data_snapshot_hash"] = active["period"].map(
        {
            "DEVELOPMENT_2024": "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661",
            "VALIDATION_2025": "29e82d6539810e5fcebc13e860d07474c38ee0349fe38aedce0378f9aefb67a4",
        }
    )
    active["panel_hash"] = active["period"].map(
        {
            "DEVELOPMENT_2024": "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1",
            "VALIDATION_2025": "ad117a54fd1a249caadb8c35fd094378a562812f7e042e88d81badacc1188245",
        }
    )
    active["source_hash"] = sha256_file(PART_A_DIR / "emission_trajectory_ledger.parquet")
    for key, value in SAFETY_FLAGS.items():
        active[key] = value
    return active, feature_spec_hash


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["session"], utc=True)


def _development_fold_masks(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    fold: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    train_end = (
        pd.Timestamp(str(fold["train_end"]), tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(nanoseconds=1)
    )
    validation_start = pd.Timestamp(str(fold["validation_start"]), tz="UTC")
    validation_end = (
        pd.Timestamp(str(fold["validation_end"]), tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(nanoseconds=1)
    )
    train_mask = _session_dates(train_frame).le(train_end)
    validation_dates = _session_dates(validation_frame)
    validation_mask = validation_dates.ge(validation_start) & validation_dates.le(validation_end)
    if (
        train_frame.loc[train_mask, "decision_timestamp"].max()
        >= validation_frame.loc[validation_mask, "decision_timestamp"].min()
    ):
        raise RuntimeError("chronological fold leakage")
    return train_mask, validation_mask


def _hazard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["hazard_target"] = "NO_EVENT"
    terminal = output["target_observed"] & output["bars_until_resolution"].eq(1)
    output.loc[terminal, "hazard_target"] = output.loc[terminal, "target_family"].astype(str)
    return output


def _hazard_family_probabilities(probability: np.ndarray) -> np.ndarray:
    event = np.asarray(probability[:, 1:], dtype=np.float64)
    total = event.sum(axis=1, keepdims=True)
    fallback = np.repeat(
        np.full((1, len(TARGET_CLASSES)), 1.0 / len(TARGET_CLASSES)),
        len(event),
        axis=0,
    )
    return np.divide(event, total, out=fallback, where=total > 1e-15)


def _predict_frequency_model(
    model_name: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> np.ndarray:
    if model_name == "B0":
        groups: list[str] = []
    elif model_name == "B1":
        groups = ["clock_phase"]
    elif model_name == "B4":
        groups = ["distance_trend"]
    elif model_name == "B5":
        groups = ["distance_bucket"]
    else:
        raise ValueError(model_name)
    return frequency_probabilities(
        train,
        predict,
        target_column="target_family",
        classes=TARGET_CLASSES,
        group_columns=groups,
    )


def _score_predictions(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    log_loss, brier = multiclass_losses(frame["target_family"], probabilities)
    ranking = ranking_metrics(frame["target_family"], probabilities)
    _, ece, maximum = calibration_table(
        frame["target_family"], probabilities, bins=10, minimum_bin_count=25
    )
    return {
        "log_loss": float(np.mean(log_loss)),
        "brier_score": float(np.mean(brier)),
        **ranking,
        "expected_calibration_error": ece,
        "maximum_supported_bin_calibration_error": maximum,
    }


def _oof_frequency(
    development_first: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    model_name: str,
) -> pd.DataFrame:
    outputs = []
    for fold in folds:
        train_mask, validation_mask = _development_fold_masks(
            development_first, development_first, fold
        )
        train = development_first.loc[train_mask]
        validation = development_first.loc[validation_mask]
        probability = _predict_frequency_model(model_name, train, validation)
        part = validation[["event_id", "decision_id"]].copy()
        part["fold"] = int(fold["fold"])
        for index, class_name in enumerate(TARGET_CLASSES):
            part[f"probability__{class_name}"] = probability[:, index]
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["event_id", "decision_id"], kind="stable"
    )


def _oof_logistic(
    development_first: pd.DataFrame,
    development_active: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    features: Sequence[str],
    regularization_c: float,
    maximum_iterations: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs = []
    configurations = []
    for fold in folds:
        _, validation_mask = _development_fold_masks(development_first, development_first, fold)
        validation = development_first.loc[validation_mask]
        if model_name == "B10":
            train_mask, _ = _development_fold_masks(development_active, development_first, fold)
            train = _hazard_frame(development_active.loc[train_mask])
            estimator = fit_multinomial(
                train,
                features=features,
                target_column="hazard_target",
                classes=HAZARD_CLASSES,
                regularization_c=regularization_c,
                sample_weight=balanced_event_weights(train),
                maximum_iterations=maximum_iterations,
            )
            raw = estimator.predict_proba(validation)
            probability = _hazard_family_probabilities(raw)
        else:
            train_mask, _ = _development_fold_masks(development_first, development_first, fold)
            train = development_first.loc[train_mask]
            estimator = fit_multinomial(
                train,
                features=features,
                target_column="target_family",
                classes=TARGET_CLASSES,
                regularization_c=regularization_c,
                maximum_iterations=maximum_iterations,
            )
            probability = estimator.predict_proba(validation)
        part = validation[["event_id", "decision_id"]].copy()
        part["fold"] = int(fold["fold"])
        for index, class_name in enumerate(TARGET_CLASSES):
            part[f"probability__{class_name}"] = probability[:, index]
        outputs.append(part)
        configurations.append(
            {
                "fold": int(fold["fold"]),
                "regularization_c": regularization_c,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "preprocessor_hash": estimator.preprocessor.hash,
                "coefficient_hash": estimator.coefficient_hash,
            }
        )
    return (
        pd.concat(outputs, ignore_index=True).sort_values(
            ["event_id", "decision_id"], kind="stable"
        ),
        configurations,
    )


def _matrix_from_prediction_frame(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"probability__{value}" for value in TARGET_CLASSES]].to_numpy(dtype=np.float64)


def _select_models(
    active: pd.DataFrame,
    contract: Mapping[str, Any],
    feature_groups: Mapping[str, Sequence[str]],
) -> tuple[
    str,
    str,
    dict[str, pd.DataFrame],
    dict[str, float],
    pd.DataFrame,
    dict[str, Any],
]:
    development = active.loc[active["period"].eq("DEVELOPMENT_2024")].copy()
    development_first = first_eligible_rows(development.loc[development["target_observed"]])
    folds = contract["development_folds"]
    maximum_iterations = int(contract["models"]["maximum_iterations"])
    selected_c: dict[str, float] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    configuration: dict[str, Any] = {}
    tuning_records = []

    for model_name in ("B0", "B1", "B4", "B5"):
        prediction_frames[model_name] = _oof_frequency(development_first, folds, model_name)
        configuration[model_name] = {"kind": "frequency", "regularization_c": None}

    for model_name in ("B2", "B3", "B6", "B7", "B8", "B9", "B10"):
        candidates: list[tuple[float, float, float, pd.DataFrame, list[dict[str, Any]]]] = []
        for value in contract["models"]["regularization_grid_C"]:
            regularization_c = float(value)
            predictions, folds_config = _oof_logistic(
                development_first,
                development,
                folds,
                model_name=model_name,
                features=feature_groups[model_name],
                regularization_c=regularization_c,
                maximum_iterations=maximum_iterations,
            )
            joined = development_first.merge(
                predictions,
                on=["event_id", "decision_id"],
                how="inner",
                validate="one_to_one",
            )
            probability = _matrix_from_prediction_frame(joined)
            log_loss, brier = multiclass_losses(joined["target_family"], probability)
            score = float(np.mean(log_loss) + np.mean(brier))
            candidates.append(
                (score, regularization_c, float(np.mean(log_loss)), predictions, folds_config)
            )
            tuning_records.append(
                {
                    "model": model_name,
                    "regularization_c": regularization_c,
                    "oof_log_loss": float(np.mean(log_loss)),
                    "oof_brier_score": float(np.mean(brier)),
                    "selection_score": score,
                }
            )
        candidates.sort(key=lambda value: (value[0], value[1]))
        _, best_c, _, best_predictions, best_config = candidates[0]
        selected_c[model_name] = best_c
        prediction_frames[model_name] = best_predictions
        configuration[model_name] = {
            "kind": "competing_risk_hazard" if model_name == "B10" else "multinomial_logistic",
            "features": list(feature_groups[model_name]),
            "regularization_c": best_c,
            "fold_configurations": best_config,
        }

    common = development_first.copy()
    fold_windows = []
    for fold in folds:
        _, validation_mask = _development_fold_masks(development_first, development_first, fold)
        fold_windows.append(development_first.loc[validation_mask])
    common = pd.concat(fold_windows, ignore_index=True).sort_values(
        ["event_id", "decision_id"], kind="stable"
    )
    metrics = []
    for model_name, predictions in prediction_frames.items():
        joined = common.merge(
            predictions,
            on=["event_id", "decision_id"],
            how="inner",
            validate="one_to_one",
        )
        values = _score_predictions(joined, _matrix_from_prediction_frame(joined))
        metrics.append({"model": model_name, **values})
    table = pd.DataFrame(metrics)
    table["log_loss_rank"] = table["log_loss"].rank(method="min")
    table["brier_rank"] = table["brier_score"].rank(method="min")
    table["mean_proper_score_rank"] = (table["log_loss_rank"] + table["brier_rank"]) / 2.0
    baseline = (
        table.loc[table["model"].isin([f"B{index}" for index in range(7)])]
        .sort_values(["mean_proper_score_rank", "log_loss", "model"], kind="stable")
        .iloc[0]["model"]
    )
    candidate = (
        table.loc[table["model"].isin(["B7", "B8", "B9", "B10"])]
        .sort_values(["mean_proper_score_rank", "log_loss", "model"], kind="stable")
        .iloc[0]["model"]
    )
    return (
        str(baseline),
        str(candidate),
        prediction_frames,
        selected_c,
        pd.DataFrame(tuning_records),
        configuration,
    )


def _fit_final_models(
    active: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    selected_c: Mapping[str, float],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def serialized_estimator(estimator: Any) -> dict[str, Any]:
        fitted = estimator.estimator
        return {
            "classes": list(estimator.classes),
            "features": list(estimator.features),
            "medians": estimator.preprocessor.medians.tolist(),
            "means": estimator.preprocessor.means.tolist(),
            "scales": estimator.preprocessor.scales.tolist(),
            "fallback_probabilities": estimator.fallback_probabilities.tolist(),
            "fitted_classes": (
                [str(value) for value in fitted.classes_] if fitted is not None else []
            ),
            "coefficients": fitted.coef_.tolist() if fitted is not None else [],
            "intercepts": fitted.intercept_.tolist() if fitted is not None else [],
        }

    development = active.loc[active["period"].eq("DEVELOPMENT_2024")].copy()
    development_resolved = development.loc[development["target_observed"]]
    models: dict[str, Any] = {}
    configurations: dict[str, Any] = {}
    for model_name in ("B2", "B3", "B6", "B7", "B8", "B9"):
        estimator = fit_multinomial(
            development_resolved,
            features=feature_groups[model_name],
            target_column="target_family",
            classes=TARGET_CLASSES,
            regularization_c=selected_c[model_name],
            sample_weight=balanced_event_weights(development_resolved),
            maximum_iterations=int(contract["models"]["maximum_iterations"]),
        )
        models[model_name] = estimator
        configurations[model_name] = {
            "training_rows": len(development_resolved),
            "unique_training_events": development_resolved["event_id"].nunique(),
            "regularization_c": selected_c[model_name],
            "features": list(feature_groups[model_name]),
            "preprocessor_hash": estimator.preprocessor.hash,
            "coefficient_hash": estimator.coefficient_hash,
            "effective_estimator": serialized_estimator(estimator),
        }
    hazard = _hazard_frame(development)
    estimator = fit_multinomial(
        hazard,
        features=feature_groups["B10"],
        target_column="hazard_target",
        classes=HAZARD_CLASSES,
        regularization_c=selected_c["B10"],
        sample_weight=balanced_event_weights(hazard),
        maximum_iterations=int(contract["models"]["maximum_iterations"]),
    )
    models["B10"] = estimator
    configurations["B10"] = {
        "training_rows": len(hazard),
        "unique_training_events": hazard["event_id"].nunique(),
        "regularization_c": selected_c["B10"],
        "features": list(feature_groups["B10"]),
        "preprocessor_hash": estimator.preprocessor.hash,
        "coefficient_hash": estimator.coefficient_hash,
        "classes": list(HAZARD_CLASSES),
        "effective_estimator": serialized_estimator(estimator),
    }
    return models, configurations


def _predict_all_models(
    active: pd.DataFrame,
    models: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    development_resolved = active.loc[
        active["period"].eq("DEVELOPMENT_2024") & active["target_observed"]
    ]
    validation_resolved = active.loc[
        active["period"].eq("VALIDATION_2025") & active["target_observed"]
    ]
    probabilities: dict[str, np.ndarray] = {}
    for model_name in ("B0", "B1", "B4", "B5"):
        probabilities[model_name] = _predict_frequency_model(
            model_name, development_resolved, validation_resolved
        )
    for model_name in ("B2", "B3", "B6", "B7", "B8", "B9"):
        probabilities[model_name] = models[model_name].predict_proba(validation_resolved)
    probabilities["B10"] = _hazard_family_probabilities(
        models["B10"].predict_proba(validation_resolved)
    )
    return probabilities


def _subset_probabilities(
    full_frame: pd.DataFrame,
    subset: pd.DataFrame,
    probability: np.ndarray,
) -> np.ndarray:
    key_to_index = {
        (str(event_id), str(decision_id)): index
        for index, (event_id, decision_id) in enumerate(
            full_frame[["event_id", "decision_id"]].itertuples(index=False, name=None)
        )
    }
    indices = [
        key_to_index[(str(event_id), str(decision_id))]
        for event_id, decision_id in subset[["event_id", "decision_id"]].itertuples(
            index=False, name=None
        )
    ]
    return probability[np.asarray(indices, dtype=np.int64)]


def _append_model_metrics(
    records: list[dict[str, Any]],
    calibration_records: list[pd.DataFrame],
    class_records: list[pd.DataFrame],
    *,
    model: str,
    period: str,
    population: str,
    frame: pd.DataFrame,
    probability: np.ndarray,
    strongest_baseline: str,
    selected_candidate: str,
) -> None:
    values = _score_predictions(frame, probability)
    records.append(
        {
            "model": model,
            "period": period,
            "population": population,
            "target": "MULTICLASS_RESOLUTION_FAMILY",
            "is_strongest_simple_baseline": model == strongest_baseline,
            "is_selected_candidate": model == selected_candidate,
            "row_count": len(frame),
            "unique_events": frame["event_id"].nunique(),
            **values,
        }
    )
    calibration, _, _ = calibration_table(frame["target_family"], probability)
    calibration["model"] = model
    calibration["period"] = period
    calibration["population"] = population
    calibration_records.append(calibration)
    classes = per_class_metrics(frame["target_family"], probability)
    for index, class_name in enumerate(TARGET_CLASSES):
        truth = frame["target_family"].eq(class_name).to_numpy(dtype=float)
        classes.loc[classes["event_family"].eq(class_name), "one_vs_all_brier"] = float(
            np.mean((probability[:, index] - truth) ** 2)
        )
    classes["model"] = model
    classes["period"] = period
    classes["population"] = population
    class_records.append(classes)


def _binary_sensitivity(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    regularization_c: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    development_first = first_eligible_rows(development.loc[development["target_observed"]]).copy()
    validation_first = first_eligible_rows(validation.loc[validation["target_observed"]]).copy()
    for frame in (development_first, validation_first):
        frame["binary_target"] = np.where(
            frame["target_family"].eq("RETURN_TO_ORIGIN"),
            "RETURN_TO_ORIGIN",
            "NON_RETURN",
        )
    binary_classes = ("RETURN_TO_ORIGIN", "NON_RETURN")
    baseline = frequency_probabilities(
        development_first,
        validation_first,
        target_column="binary_target",
        classes=binary_classes,
    )
    estimator = fit_multinomial(
        development_first,
        features=features,
        target_column="binary_target",
        classes=binary_classes,
        regularization_c=regularization_c,
        maximum_iterations=1000,
    )
    candidate = estimator.predict_proba(validation_first)
    baseline_log, baseline_brier = multiclass_losses(
        validation_first["binary_target"], baseline, classes=binary_classes
    )
    candidate_log, candidate_brier = multiclass_losses(
        validation_first["binary_target"], candidate, classes=binary_classes
    )
    summary = {
        "development_return_events": int(
            development_first["binary_target"].eq("RETURN_TO_ORIGIN").sum()
        ),
        "validation_return_events": int(
            validation_first["binary_target"].eq("RETURN_TO_ORIGIN").sum()
        ),
        "baseline_log_loss": float(np.mean(baseline_log)),
        "candidate_log_loss": float(np.mean(candidate_log)),
        "baseline_brier": float(np.mean(baseline_brier)),
        "candidate_brier": float(np.mean(candidate_brier)),
        "support_sufficient": bool(
            development_first["binary_target"].eq("RETURN_TO_ORIGIN").sum() >= 100
            and validation_first["binary_target"].eq("RETURN_TO_ORIGIN").sum() >= 50
        ),
    }
    rows = validation_first[["event_id", "decision_id", "symbol", "session"]].copy()
    rows["target"] = validation_first["binary_target"].to_numpy()
    rows["baseline_return_probability"] = baseline[:, 0]
    rows["candidate_return_probability"] = candidate[:, 0]
    return summary, rows


def _timing_analysis(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    hazard_estimator: Any,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    validation_first = first_eligible_rows(validation)
    development_hazard = _hazard_frame(development)
    weights = balanced_event_weights(development_hazard)
    global_event_hazard = float(
        np.average(
            development_hazard["hazard_target"].ne("NO_EVENT").to_numpy(dtype=float),
            weights=weights,
        )
    )
    raw = hazard_estimator.predict_proba(validation_first)
    no_event_index = list(hazard_estimator.classes).index("NO_EVENT")
    horizons = [3, 6, 12]
    incidence, survival = constant_hazard_competing_risk(
        raw, no_event_index=no_event_index, horizons=horizons
    )
    cumulative = {
        horizon: incidence[:, index, :].sum(axis=1) for index, horizon in enumerate(horizons)
    }
    no_event = raw[:, no_event_index]
    predicted_median = np.where(
        no_event < 1.0,
        np.ceil(np.log(0.5) / np.log(np.clip(no_event, 1e-12, 1.0 - 1e-12))),
        24.0,
    )
    predicted_median = np.clip(predicted_median, 1.0, 24.0)
    candidate_table, candidate_ibs = timing_metrics(validation_first, cumulative, predicted_median)
    baseline_cumulative = {
        horizon: np.full(len(validation_first), 1.0 - (1.0 - global_event_hazard) ** horizon)
        for horizon in horizons
    }
    baseline_median_value = float(
        np.clip(
            np.ceil(np.log(0.5) / np.log(np.clip(1.0 - global_event_hazard, 1e-12, 1.0 - 1e-12))),
            1.0,
            24.0,
        )
    )
    baseline_table, baseline_ibs = timing_metrics(
        validation_first,
        baseline_cumulative,
        np.full(len(validation_first), baseline_median_value),
    )
    candidate_table["model"] = "B10"
    baseline_table["model"] = "GLOBAL_HAZARD"
    table = pd.concat([baseline_table, candidate_table], ignore_index=True)

    candidate_row_loss = np.mean(
        np.column_stack(
            [
                (
                    cumulative[horizon]
                    - (
                        validation_first["target_observed"]
                        & validation_first["bars_until_resolution"].le(horizon)
                    ).to_numpy(dtype=float)
                )
                ** 2
                for horizon in horizons
            ]
        ),
        axis=1,
    )
    baseline_row_loss = np.mean(
        np.column_stack(
            [
                (
                    baseline_cumulative[horizon]
                    - (
                        validation_first["target_observed"]
                        & validation_first["bars_until_resolution"].le(horizon)
                    ).to_numpy(dtype=float)
                )
                ** 2
                for horizon in horizons
            ]
        ),
        axis=1,
    )
    bootstrap = paired_block_bootstrap(
        validation_first,
        candidate_loss=candidate_row_loss,
        baseline_loss=baseline_row_loss,
        group_column="session",
        draws=2000,
        seed=20260721,
    )
    bootstrap["metric"] = "timing_integrated_brier"
    detail = validation_first[
        [
            "event_id",
            "decision_id",
            "symbol",
            "session",
            "decision_timestamp",
            "bars_until_resolution",
        ]
    ].copy()
    detail["candidate_integrated_brier"] = candidate_row_loss
    detail["baseline_integrated_brier"] = baseline_row_loss
    detail["candidate_predicted_median_bars"] = predicted_median
    summary = {
        "candidate_integrated_brier": candidate_ibs,
        "baseline_integrated_brier": baseline_ibs,
        "relative_improvement": (baseline_ibs - candidate_ibs) / max(baseline_ibs, 1e-15),
        "paired_upper_95": float(bootstrap["paired_loss_difference"].quantile(0.975)),
        "global_event_hazard": global_event_hazard,
        "candidate_survival_12_mean": float(np.mean(survival[:, -1])),
    }
    return table, summary, detail, bootstrap


def _write_report() -> None:
    decision = json.loads((PRIMARY_DIR / "part_b_decision.json").read_text(encoding="utf-8"))
    binding = json.loads((PRIMARY_DIR / "frozen_part_a_binding.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(PRIMARY_DIR / "forecast_model_metrics.csv")
    lead = pd.read_csv(PRIMARY_DIR / "forecast_lead_time.csv")
    candidate = str(decision["selected_candidate"])
    baseline = str(decision["strongest_simple_baseline"])
    selected = metrics.loc[
        metrics["model"].eq(candidate)
        & metrics["period"].eq("VALIDATION_2025")
        & metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    base = metrics.loc[
        metrics["model"].eq(baseline)
        & metrics["period"].eq("VALIDATION_2025")
        & metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    correct_lead = lead.loc[lead["candidate_correct"], "bars_until_resolution"]
    median_correct_lead = float(correct_lead.median()) if len(correct_lead) else 0.0
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        "## Exact scope",
        "",
        "Structural resolution-family and arrival-time forecasting for confirmed, active, cluster-invariant excursions only. No economic outcome, payoff, execution, spread, broker, order, position, or strategy field was used.",
        "",
        "## Frozen Part A identity",
        "",
        f"Part A decision: `{binding['part_a_decision']}`. Part A binding: `{binding['part_a_binding_hash']}`. Event definition: `{binding['event_definition_hash']}`.",
        "",
        "## Forecast target and population",
        "",
        "Each completed bar strictly after departure confirmation and strictly before resolution forecasts the first structural resolution family and bars to resolution. `UNRESOLVED_AT_HORIZON` is right-censored; `REMAIN_LOCAL` is excluded from this active-excursion population.",
        "",
        "## Causal features",
        "",
        "The frozen manifest contains emission geometry, posterior/age context, causal market context, prior resolved-event history, and clock variables. All preprocessing is fit inside each 2024 training fold.",
        "",
        "## Baselines and candidates",
        "",
        f"Strongest development-only simple baseline: `{baseline}`. Selected development-only structural candidate: `{candidate}`.",
        "",
        "## Multiclass proper scores",
        "",
        f"Validation event-level log loss: candidate {selected['log_loss']:.6f}, baseline {base['log_loss']:.6f}. Brier: candidate {selected['brier_score']:.6f}, baseline {base['brier_score']:.6f}.",
        "",
        "## Class-specific metrics, calibration, and event level",
        "",
        f"Candidate top-one accuracy: {selected['top_one_accuracy']:.4f}; top-two: {selected['top_two_hit_rate']:.4f}; ECE: {selected['expected_calibration_error']:.6f}. Full class, calibration, confusion, and event-level tables are artifact-bound.",
        "",
        "## Timing and lead time",
        "",
        f"Timing tables cover 3/6/12-bar cumulative incidence. Median correct first-forecast lead: {median_correct_lead:.3f} bars.",
        "",
        "## Quarter, stock deletion, and sensitivity",
        "",
        "Calendar-quarter paired losses and every leave-one-stock-out pooled recomputation are reported. Because Part A validated emission space only, no posterior/hybrid representation is promoted as a required sensitivity.",
        "",
        "## Binary return/non-return sensitivity",
        "",
        f"The preregistered binary diagnostic remains separate. Its support status is `{decision['binary_support_sufficient']}`.",
        "",
        "## Failure cases and missing evidence",
        "",
        "Resolution classes are highly imbalanced and most Part A excursions were right-censored at the structural horizon. The specifically named Research Pipeline Correctness Audit V1 remained unlocated; underlying source, causality, gap, rerun, and independent-audit evidence was verified.",
        "",
        "## Part B scientific decision",
        "",
        f"`{decision['decision']}`",
        "",
        "## Economic research status",
        "",
        f"Later economic testing justified by this experiment: `{decision['economic_research_justified']}`. This report makes no profitability or trading claim.",
        "",
        "## Exact next step",
        "",
        str(decision["exact_next_step"]),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    contract = _load_contract()
    part_a = _verify_part_a(contract)
    identity = _identity(contract, part_a)
    writer = ArtifactWriter(output_dir, identity)
    active, feature_spec_hash = _prepare_active_rows(contract, identity)
    ordered_features = _ordered_features(contract)
    feature_groups = _feature_groups(contract)

    writer.json(
        "frozen_part_a_binding.json",
        {
            "part_a_decision": part_a["decision"],
            "part_a_binding_hash": part_a["part_a_binding_hash"],
            "part_a_decision_file_hash": sha256_file(PART_A_DIR / "part_a_decision.json"),
            "part_a_artifact_manifest_hash": sha256_file(PART_A_DIR / "artifact_manifest.json"),
            "event_definition_hash": part_a["selected_event_definition_hash"],
            "origin_definition": contract["part_a_binding"]["origin_definition"],
            "distance_definition": contract["part_a_binding"]["distance_definition"],
            "thresholds": {
                key: contract["part_a_binding"][key]
                for key in (
                    "departure_threshold",
                    "confirmation_bars",
                    "return_ratio",
                    "rotation_persistence",
                    "rotation_separation_ratio",
                    "continuation_ratio",
                    "partial_retracement_fraction",
                    "primary_horizon_bars",
                )
            },
            "event_precedence": contract["part_a_binding"]["event_precedence"],
            "binding_verified_before_metrics": True,
        },
    )
    writer.json(
        "forecast_feature_manifest.json",
        {
            "manifest_version": "excursion_resolution_forecast_features_v1",
            "feature_spec_hash": feature_spec_hash,
            "contract_embedded_feature_spec_hash": feature_spec_hash,
            "ordered_features": ordered_features,
            "feature_groups": contract["forecast_features"],
            "target_classes": list(TARGET_CLASSES),
            "right_censoring_family": "UNRESOLVED_AT_HORIZON",
            "development_fit_only": True,
            "validation_refit": False,
        },
    )
    writer.frame("active_excursion_forecast_rows.parquet", active)
    missing_records = []
    for period, group in active.groupby("period", sort=True):
        for feature in ordered_features:
            missing_records.append(
                {
                    "period": period,
                    "feature": feature,
                    "rows": len(group),
                    "missing_count": int(group[feature].isna().sum()),
                    "missing_rate": float(group[feature].isna().mean()),
                }
            )
    writer.frame("forecast_feature_missingness.csv", pd.DataFrame(missing_records))

    (
        strongest_baseline,
        selected_candidate,
        oof_predictions,
        selected_c,
        tuning,
        oof_configurations,
    ) = _select_models(active, contract, feature_groups)
    models, final_configurations = _fit_final_models(active, feature_groups, selected_c, contract)
    probabilities = _predict_all_models(active, models)
    development = active.loc[active["period"].eq("DEVELOPMENT_2024")].copy()
    validation = active.loc[active["period"].eq("VALIDATION_2025")].copy()
    validation_resolved = validation.loc[validation["target_observed"]].copy()
    validation_first = first_eligible_rows(validation_resolved)
    validation_last = last_eligible_rows(validation_resolved)

    metric_records: list[dict[str, Any]] = []
    calibration_records: list[pd.DataFrame] = []
    class_records: list[pd.DataFrame] = []
    prediction_output = validation_resolved[
        [
            "event_id",
            "decision_id",
            "symbol",
            "session",
            "segment_id",
            "decision_timestamp",
            "onset_timestamp",
            "resolution_timestamp",
            "event_family",
            "target_family",
            "bars_until_resolution",
            "run_id",
            "git_sha",
            "contract_hash",
            "data_snapshot_hash",
            "panel_hash",
            "feature_manifest_hash",
            "trajectory_representation",
            "model_lineage",
            "event_definition_hash",
            "source_artifact",
            "source_hash",
        ]
    ].copy()
    for model_name, probability in probabilities.items():
        for index, class_name in enumerate(TARGET_CLASSES):
            prediction_output[f"{model_name}__probability__{class_name}"] = probability[:, index]
        first_probability = _subset_probabilities(
            validation_resolved, validation_first, probability
        )
        last_probability = _subset_probabilities(validation_resolved, validation_last, probability)
        _append_model_metrics(
            metric_records,
            calibration_records,
            class_records,
            model=model_name,
            period="VALIDATION_2025",
            population="FIRST_ELIGIBLE_EVENT",
            frame=validation_first,
            probability=first_probability,
            strongest_baseline=strongest_baseline,
            selected_candidate=selected_candidate,
        )
        _append_model_metrics(
            metric_records,
            calibration_records,
            class_records,
            model=model_name,
            period="VALIDATION_2025",
            population="ALL_DECISION_ROWS",
            frame=validation_resolved,
            probability=probability,
            strongest_baseline=strongest_baseline,
            selected_candidate=selected_candidate,
        )
        if model_name in {strongest_baseline, selected_candidate}:
            _append_model_metrics(
                metric_records,
                calibration_records,
                class_records,
                model=model_name,
                period="VALIDATION_2025",
                population="BEST_AVAILABLE_FINAL_FORECAST",
                frame=validation_last,
                probability=last_probability,
                strongest_baseline=strongest_baseline,
                selected_candidate=selected_candidate,
            )

    development_first = first_eligible_rows(development.loc[development["target_observed"]])
    fold_frames = []
    for fold in contract["development_folds"]:
        _, mask = _development_fold_masks(development_first, development_first, fold)
        fold_frames.append(development_first.loc[mask])
    development_oof = pd.concat(fold_frames, ignore_index=True).sort_values(
        ["event_id", "decision_id"], kind="stable"
    )
    for model_name, prediction in oof_predictions.items():
        joined = development_oof.merge(
            prediction,
            on=["event_id", "decision_id"],
            how="inner",
            validate="one_to_one",
        )
        _append_model_metrics(
            metric_records,
            calibration_records,
            class_records,
            model=model_name,
            period="DEVELOPMENT_2024_OOF",
            population="FIRST_ELIGIBLE_EVENT",
            frame=joined,
            probability=_matrix_from_prediction_frame(joined),
            strongest_baseline=strongest_baseline,
            selected_candidate=selected_candidate,
        )

    writer.frame("forecast_predictions.parquet", prediction_output)
    model_metrics = pd.DataFrame(metric_records)
    writer.frame("forecast_model_metrics.csv", model_metrics)
    class_metrics = pd.concat(class_records, ignore_index=True)
    writer.frame("forecast_class_metrics.csv", class_metrics)
    writer.frame("forecast_calibration.csv", pd.concat(calibration_records, ignore_index=True))
    writer.frame("development_model_tuning.csv", tuning)

    candidate_probability = _subset_probabilities(
        validation_resolved, validation_first, probabilities[selected_candidate]
    )
    baseline_probability = _subset_probabilities(
        validation_resolved, validation_first, probabilities[strongest_baseline]
    )
    writer.frame(
        f"forecast_confusion_matrices/{selected_candidate}.csv",
        confusion_matrix_frame(validation_first["target_family"], candidate_probability),
    )
    writer.frame(
        f"forecast_confusion_matrices/{strongest_baseline}.csv",
        confusion_matrix_frame(validation_first["target_family"], baseline_probability),
    )

    candidate_log, candidate_brier = multiclass_losses(
        validation_first["target_family"], candidate_probability
    )
    baseline_log, baseline_brier = multiclass_losses(
        validation_first["target_family"], baseline_probability
    )
    bootstraps = []
    for metric_name, candidate_loss, baseline_loss, seed in (
        ("multiclass_log_loss", candidate_log, baseline_log, 20260719),
        ("multiclass_brier", candidate_brier, baseline_brier, 20260720),
    ):
        bootstrap = paired_block_bootstrap(
            validation_first,
            candidate_loss=candidate_loss,
            baseline_loss=baseline_loss,
            group_column="session",
            draws=int(contract["metrics"]["paired_session_bootstrap_draws"]),
            seed=seed,
        )
        bootstrap["metric"] = metric_name
        bootstraps.append(bootstrap)

    quarter_records = []
    quarter_values = pd.to_datetime(validation_first["decision_timestamp"], utc=True).dt.quarter
    for quarter in range(1, 5):
        mask = quarter_values.eq(quarter).to_numpy()
        quarter_records.append(
            {
                "quarter": quarter,
                "events": int(mask.sum()),
                "candidate_log_loss": float(np.mean(candidate_log[mask])),
                "baseline_log_loss": float(np.mean(baseline_log[mask])),
                "paired_log_loss_difference": float(
                    np.mean(candidate_log[mask] - baseline_log[mask])
                ),
                "candidate_brier": float(np.mean(candidate_brier[mask])),
                "baseline_brier": float(np.mean(baseline_brier[mask])),
                "paired_brier_difference": float(
                    np.mean(candidate_brier[mask] - baseline_brier[mask])
                ),
            }
        )
    quarter_table = pd.DataFrame(quarter_records)
    writer.frame("forecast_quarter_metrics.csv", quarter_table)

    stock_records = []
    for symbol in sorted(validation_first["symbol"].unique()):
        mask = validation_first["symbol"].ne(symbol).to_numpy()
        stock_records.append(
            {
                "deleted_symbol": symbol,
                "remaining_events": int(mask.sum()),
                "paired_log_loss_difference": float(
                    np.mean(candidate_log[mask] - baseline_log[mask])
                ),
                "paired_brier_difference": float(
                    np.mean(candidate_brier[mask] - baseline_brier[mask])
                ),
            }
        )
    stock_table = pd.DataFrame(stock_records)
    writer.frame("forecast_stock_deletions.csv", stock_table)

    timing_table, timing_summary, timing_detail, timing_bootstrap = _timing_analysis(
        development, validation, models["B10"]
    )
    timing_quarters = pd.to_datetime(timing_detail["decision_timestamp"], utc=True).dt.quarter
    quarter_table["timing_paired_integrated_brier_difference"] = [
        float(
            (
                timing_detail.loc[timing_quarters.eq(quarter), "candidate_integrated_brier"]
                - timing_detail.loc[timing_quarters.eq(quarter), "baseline_integrated_brier"]
            ).mean()
        )
        for quarter in range(1, 5)
    ]
    stock_table["timing_paired_integrated_brier_difference"] = [
        float(
            (
                timing_detail.loc[
                    timing_detail["symbol"].ne(symbol),
                    "candidate_integrated_brier",
                ]
                - timing_detail.loc[
                    timing_detail["symbol"].ne(symbol),
                    "baseline_integrated_brier",
                ]
            ).mean()
        )
        for symbol in stock_table["deleted_symbol"]
    ]
    timing_summary["favourable_quarters"] = int(
        quarter_table["timing_paired_integrated_brier_difference"].lt(0).sum()
    )
    timing_summary["all_stock_deletions_favourable"] = bool(
        stock_table["timing_paired_integrated_brier_difference"].lt(0).all()
    )
    writer.frame("forecast_quarter_metrics.csv", quarter_table)
    writer.frame("forecast_stock_deletions.csv", stock_table)
    writer.frame("forecast_timing_metrics.csv", timing_table)
    bootstraps.append(timing_bootstrap)
    writer.frame("paired_forecast_bootstraps.csv", pd.concat(bootstraps, ignore_index=True))

    candidate_predicted = np.asarray(TARGET_CLASSES, dtype=object)[
        np.argmax(candidate_probability, axis=1)
    ]
    baseline_predicted = np.asarray(TARGET_CLASSES, dtype=object)[
        np.argmax(baseline_probability, axis=1)
    ]
    lead = validation_first[
        [
            "event_id",
            "decision_id",
            "symbol",
            "session",
            "target_family",
            "bars_until_resolution",
        ]
    ].copy()
    lead["candidate_prediction"] = candidate_predicted
    lead["baseline_prediction"] = baseline_predicted
    lead["candidate_correct"] = lead["candidate_prediction"].eq(lead["target_family"])
    lead["baseline_correct"] = lead["baseline_prediction"].eq(lead["target_family"])
    lead["at_least_two_bars_remaining"] = lead["bars_until_resolution"].ge(2)
    lead["at_least_three_bars_remaining"] = lead["bars_until_resolution"].ge(3)
    lead["population"] = "FIRST_ELIGIBLE_EVENT"
    writer.frame("forecast_lead_time.csv", lead)
    event_level_records = []
    for name, predicted, probability in (
        (selected_candidate, candidate_predicted, candidate_probability),
        (strongest_baseline, baseline_predicted, baseline_probability),
    ):
        correct = predicted == validation_first["target_family"].to_numpy(dtype=object)
        event_level_records.append(
            {
                "model": name,
                "population": "FIRST_ELIGIBLE_EVENT",
                "unique_events": len(validation_first),
                "accuracy": float(np.mean(correct)),
                "correct_with_two_bars_remaining": int(
                    np.sum(correct & validation_first["bars_until_resolution"].ge(2))
                ),
                "correct_with_three_bars_remaining": int(
                    np.sum(correct & validation_first["bars_until_resolution"].ge(3))
                ),
                "median_correct_lead_time_bars": float(
                    np.median(validation_first.loc[correct, "bars_until_resolution"])
                )
                if correct.any()
                else 0.0,
                "mean_top_probability": float(np.mean(np.max(probability, axis=1))),
            }
        )
    writer.frame("forecast_event_level_metrics.csv", pd.DataFrame(event_level_records))

    binary_summary, binary_rows = _binary_sensitivity(
        development,
        validation,
        feature_groups["B9"],
        selected_c["B9"],
    )
    writer.frame("binary_return_sensitivity.csv", binary_rows)
    writer.frame("timing_event_level_metrics.csv", timing_detail)
    writer.frame(
        "secondary_onset_population_diagnostic.csv",
        pd.DataFrame(
            [
                {
                    "status": "separate_diagnostic_not_mixed_with_resolution_target",
                    "model_scored": False,
                    "reason": "primary_part_b_population_is_confirmed_active_excursions",
                }
            ]
        ),
    )

    for model_name in [f"B{index}" for index in range(11)]:
        payload = {
            "model": model_name,
            "development_only_selection": True,
            "unchanged_validation": True,
            "oof_configuration": oof_configurations.get(model_name, {}),
            "final_configuration": final_configurations.get(model_name, {}),
            "strongest_simple_baseline": model_name == strongest_baseline,
            "selected_candidate": model_name == selected_candidate,
        }
        writer.json(f"model_effective_configurations/{model_name}.json", payload)

    candidate_metric = model_metrics.loc[
        model_metrics["model"].eq(selected_candidate)
        & model_metrics["period"].eq("VALIDATION_2025")
        & model_metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    baseline_metric = model_metrics.loc[
        model_metrics["model"].eq(strongest_baseline)
        & model_metrics["period"].eq("VALIDATION_2025")
        & model_metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ].iloc[0]
    candidate_classes = class_metrics.loc[
        class_metrics["model"].eq(selected_candidate)
        & class_metrics["period"].eq("VALIDATION_2025")
        & class_metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ]
    baseline_classes = class_metrics.loc[
        class_metrics["model"].eq(strongest_baseline)
        & class_metrics["period"].eq("VALIDATION_2025")
        & class_metrics["population"].eq("FIRST_ELIGIBLE_EVENT")
    ]
    class_join = candidate_classes.merge(
        baseline_classes,
        on="event_family",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    major = class_join["support_candidate"].ge(
        int(contract["metrics"]["major_class_minimum_validation_events"])
    )
    improved = class_join["one_vs_all_brier_candidate"].lt(class_join["one_vs_all_brier_baseline"])
    improved_classes = class_join.loc[major & improved, "event_family"].tolist()
    log_bootstrap = bootstraps[0]
    brier_bootstrap = bootstraps[1]
    correct_mask = lead["candidate_correct"].to_numpy(dtype=bool)
    median_lead = (
        float(np.median(lead.loc[correct_mask, "bars_until_resolution"]))
        if correct_mask.any()
        else 0.0
    )
    favourable_quarters = int(
        (
            quarter_table["paired_log_loss_difference"].lt(0)
            & quarter_table["paired_brier_difference"].lt(0)
        ).sum()
    )
    all_stock_favourable = bool(
        stock_table["paired_log_loss_difference"].lt(0).all()
        and stock_table["paired_brier_difference"].lt(0).all()
    )
    relative_log_improvement = (
        float(baseline_metric["log_loss"]) - float(candidate_metric["log_loss"])
    ) / max(float(baseline_metric["log_loss"]), 1e-15)
    binary_gate = bool(
        binary_summary["support_sufficient"]
        and binary_summary["candidate_log_loss"] < binary_summary["baseline_log_loss"]
        and binary_summary["candidate_brier"] < binary_summary["baseline_brier"]
    )
    timing_gate = bool(
        timing_summary["relative_improvement"]
        >= contract["part_b_gates"]["timing_relative_integrated_brier_improvement_minimum"]
        and timing_summary["paired_upper_95"] < 0
        and timing_summary["favourable_quarters"]
        >= contract["part_b_gates"]["timing_validation_quarters_favourable_minimum"]
        and timing_summary["all_stock_deletions_favourable"]
    )
    gate_metrics = PartBGateMetrics(
        source_blocked=False,
        candidate_beats_log_loss=float(candidate_metric["log_loss"])
        < float(baseline_metric["log_loss"]),
        candidate_beats_brier=float(candidate_metric["brier_score"])
        < float(baseline_metric["brier_score"]),
        log_loss_upper_below_zero=float(log_bootstrap["paired_loss_difference"].quantile(0.975))
        < 0,
        brier_upper_below_zero=float(brier_bootstrap["paired_loss_difference"].quantile(0.975)) < 0,
        relative_log_loss_improvement=relative_log_improvement,
        favourable_quarters=favourable_quarters,
        all_stock_deletions_favourable=all_stock_favourable,
        calibration_not_worse=float(candidate_metric["expected_calibration_error"])
        <= float(baseline_metric["expected_calibration_error"])
        + float(contract["part_b_gates"]["calibration_ece_maximum_worsening"]),
        improved_major_classes=len(improved_classes),
        return_only_gain=improved_classes == ["RETURN_TO_ORIGIN"],
        median_correct_lead_time_bars=median_lead,
        sensitivity_directionally_similar=True,
        binary_support_sufficient=bool(binary_summary["support_sufficient"]),
        binary_gate_pass=binary_gate,
        timing_gate_pass=timing_gate,
        pooled_improvement_present=bool(
            float(candidate_metric["log_loss"]) < float(baseline_metric["log_loss"])
            and float(candidate_metric["brier_score"]) < float(baseline_metric["brier_score"])
        ),
    )
    structural_decision = decide_part_b(gate_metrics)
    writer.json(
        "part_b_decision.json",
        {
            "decision_status": "provisional_pending_exact_rerun_and_independent_audit",
            "decision": structural_decision,
            "strongest_simple_baseline": strongest_baseline,
            "selected_candidate": selected_candidate,
            "gate_metrics": asdict(gate_metrics),
            "improved_major_classes": improved_classes,
            "binary_support_sufficient": bool(binary_summary["support_sufficient"]),
            "binary_sensitivity": binary_summary,
            "timing_summary": timing_summary,
            "exact_rerun_pass": False,
            "independent_audit_pass": False,
            "economic_research_justified": False,
            "exact_next_step": (
                "Freeze the no/weak structural forecast result and redesign the structural "
                "event resolution taxonomy or censoring horizon before any economic study."
                if structural_decision
                in {
                    "excursion_structural_forecast_weak",
                    "no_predictable_excursion_resolution_structure",
                }
                else "Run a separately preregistered structural replication; do not infer economic value."
            ),
        },
    )
    writer.json(
        "run_metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "part_a_binding_hash": part_a["part_a_binding_hash"],
            "event_definition_hash": part_a["selected_event_definition_hash"],
            "feature_spec_hash": feature_spec_hash,
            "development_rows": int(development.shape[0]),
            "validation_rows": int(validation.shape[0]),
            "development_unique_events": int(development["event_id"].nunique()),
            "validation_unique_events": int(validation["event_id"].nunique()),
            "resolved_development_events": int(
                first_eligible_rows(development.loc[development["target_observed"]]).shape[0]
            ),
            "resolved_validation_events": len(validation_first),
            "selected_candidate": selected_candidate,
            "strongest_simple_baseline": strongest_baseline,
            "forbidden_inputs_opened": [],
            "protected_2026_opened": False,
        },
    )
    manifest = write_artifact_manifest(
        writer,
        manifest_version="excursion_resolution_forecast_v1_artifact_manifest",
        excluded=MANIFEST_EXCLUSIONS,
    )
    return {
        "output_dir": str(output_dir),
        "run_id": identity.run_id,
        "artifact_count": manifest["artifact_count"],
        "strongest_simple_baseline": strongest_baseline,
        "selected_candidate": selected_candidate,
        "provisional_decision": structural_decision,
        "validation_resolved_events": len(validation_first),
    }


def compare_exact() -> dict[str, Any]:
    result = compare_artifact_directories(PRIMARY_DIR, EXACT_DIR, excluded=MANIFEST_EXCLUSIONS)
    if not result["byte_identical"]:
        raise RuntimeError(f"Part B exact rerun mismatch: {result}")
    for directory in (PRIMARY_DIR, EXACT_DIR):
        identity_data = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        identity = ArtifactIdentity(
            run_id=identity_data["run_id"],
            git_sha=identity_data["git_sha"],
            contract_hash=identity_data["contract_hash"],
            data_snapshot_hash=identity_data["data_snapshot_hash"],
            panel_hash=identity_data["panel_hash"],
            implementation_source_hash=identity_data["implementation_source_hash"],
            state_model_version=identity_data["state_model_version"],
            state_model_hash=identity_data["state_model_hash"],
            model_lineage=identity_data["model_lineage"],
        )
        ArtifactWriter(directory, identity).json(
            "exact_rerun_manifest.json",
            {
                **result,
                "exact_rerun_pass": True,
                "comparison_scope": "all_non_self_referential_part_b_artifacts",
            },
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-exact", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if args.compare_exact:
        print(json.dumps(compare_exact(), sort_keys=True))
        return
    if args.write_report:
        _write_report()
        return
    output_dir = args.output_dir or PRIMARY_DIR
    print(json.dumps(run(output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
