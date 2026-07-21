#!/usr/bin/env python3
"""Run Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-behavioural-trajectory-v0-mpl")

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_SRC = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from stocker_research.behavioural_trajectory_regime_funnel_v0 import (
    SAFETY_FLAGS,
    TRAJECTORY_INTERACTION_FEATURES,
    TRAJECTORY_INTERACTION_SPECS,
    BlockedScreen,
    anchor_formula_availability,
    decide_trajectory_screen,
    manual_multinomial_probabilities,
    multiclass_brier,
    prediction_entropy,
    reject_protected_dates,
    trajectory_anchors,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT / "research" / "loop-funnel" / "20260721-emotion-regime-coarse-loop-family-v0"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_PANEL = PREDECESSOR_PRIMARY / "decision_panel.parquet"
PREDECESSOR_PREDICTIONS = PREDECESSOR_PRIMARY / "assessment_predictions.parquet"
PREDECESSOR_COEFFICIENTS = PREDECESSOR_PRIMARY / "model_coefficients.json"
PREDECESSOR_CONFIGURATIONS = PREDECESSOR_PRIMARY / "model_configurations.json"
PREDECESSOR_METRICS = PREDECESSOR_PRIMARY / "pooled_metrics.csv"
PREDECESSOR_DECISION = PREDECESSOR_PRIMARY / "decision.json"
PREDECESSOR_AUDIT = PREDECESSOR_PRIMARY / "lightweight_audit.json"
PREDECESSOR_DETERMINISM = PREDECESSOR_PRIMARY / "determinism_check.json"
PREDECESSOR_BOUNDARY = PREDECESSOR_PRIMARY / "protected_boundary_audit.json"
PREDECESSOR_SOURCE = PREDECESSOR_PRIMARY / "source_manifest.json"
DEFAULT_COMPONENT_LEDGER = (
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
    / "artifacts"
    / "primary"
    / "behavioural_component_ledger.parquet"
)

EXPECTED_PREDECESSOR_COMMIT = "29c61bbe74f45d0a00c5f2a12cdcd5d996131a23"
EXPECTED_HASHES = {
    "decision_panel.parquet": "8e6919c03e207e3eee2f05a47271dbff0129c02a0cb0f04c0b6eb1410a65dca5",
    "assessment_predictions.parquet": (
        "7336e25c4ad619e77672030d8203293028a9355c6d727ac6f07e4d1378d498f1"
    ),
    "model_coefficients.json": "0daeaa4c4f261f8b0a24fa5c3aaefea3b262a9ac9480c4951056ce53542f77e9",
    "model_configurations.json": "312436ece283e6121161b6c4fbb2485827fc99fca1a9310fd01974922fa6e2ea",
    "pooled_metrics.csv": "2516ef382cee52d460d2b95cf886b82b0ba6923955184a24d820267c359f9398",
    "decision.json": "71c97bd13f8893e4ccc17ffd6574592b616b005b41e1ded354d6c9b1bc085b24",
    "lightweight_audit.json": "cf7a2084819a71528d3e90627d3cd9a261bb266bf5e2e88d728b8316c465271b",
    "determinism_check.json": "b47563eb81740abebc1d9882203c93634ec8422cbea67946f1ee459450df497c",
    "protected_boundary_audit.json": (
        "d2d4387c8cdbc891913199e6d3e210a8d050111f66495f434ecf1558f9ccb2e8"
    ),
    "source_manifest.json": "747809c81edd82d837b85f318d67a125ebb036fddccb3b699e69b14180373a5c",
}
EXPECTED_COMPONENT_LEDGER_HASH = "96fad90eeeb6edf8b2f82ba809fb6939ad3ce5b7d5bf8193abacd90f471a91cb"
MODEL_SEED = 20260721
PRIMARY_BEHAVIOURS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
PRIMARY_FORMS = ("change", "acceleration", "reversal")
DESCRIPTIVE_FORMS = ("recent_change", "persistence", "peak_displacement")
PRIMARY_TRAJECTORY_FEATURES = tuple(
    f"{emotion}_{form}" for emotion in PRIMARY_BEHAVIOURS for form in PRIMARY_FORMS
)
ALL_TRAJECTORY_FEATURES = tuple(
    f"{emotion}_{form}"
    for emotion in PRIMARY_BEHAVIOURS
    for form in (*PRIMARY_FORMS, *DESCRIPTIVE_FORMS)
)
TARGET_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
)
EXPECTED_LOCAL_CLOCKS = {2: "09:40", 4: "09:50", 6: "10:00", 9: "10:15", 12: "10:30"}
MODEL_NUMERIC_METRICS = (
    "multiclass_log_loss",
    "multiclass_brier",
    "top_one_accuracy",
    "top_two_accuracy",
    "top_three_accuracy",
    "mean_reciprocal_rank",
    "mean_probability_realised_class",
    "macro_ovr_auc",
    "expected_calibration_error",
    "prediction_entropy",
    "effective_candidate_count",
    "rows",
    "sessions",
    "stocks",
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    for key, value in SAFETY_FLAGS.items():
        if contract.get(key) != value or contract.get("safety", {}).get(key) != value:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if tuple(contract["target_classes"]) != TARGET_CLASSES:
        raise RuntimeError("contract target classes differ")
    return contract


@dataclass(slots=True)
class FittedMultinomial:
    name: str
    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression
    class_order: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        probabilities = self.estimator.predict_proba(self.scaler.transform(matrix))
        if not np.array_equal(self.estimator.classes_, np.arange(len(self.class_order))):
            raise BlockedScreen(
                "blocked_reproducibility_or_audit_failure",
                "multinomial class ordering differs",
            )
        return np.asarray(probabilities, dtype=float)

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_model": "M2",
            "features": list(self.features),
            "class_order": list(self.class_order),
            "estimator_classes": self.estimator.classes_.astype(int).tolist(),
            "coefficient": self.estimator.coef_.tolist(),
            "intercept": self.estimator.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_variance": self.scaler.var_.tolist(),
            "n_iter": self.estimator.n_iter_.astype(int).tolist(),
        }


def fit_multinomial(
    name: str,
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    class_order: tuple[str, ...],
) -> FittedMultinomial:
    names = tuple(str(feature) for feature in features)
    class_index = {label: index for index, label in enumerate(class_order)}
    target = frame["target_class"].map(class_index)
    matrix = frame.loc[:, list(names)].to_numpy(dtype=float)
    weights = frame["row_weight"].to_numpy(dtype=float)
    if target.isna().any() or not np.isfinite(matrix).all() or not np.isfinite(weights).all():
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "frozen model input is incomplete",
        )
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="lbfgs",
        max_iter=300,
        class_weight=None,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(scaled, target.to_numpy(dtype=int), sample_weight=weights)
    except ConvergenceWarning as error:
        raise BlockedScreen(
            "blocked_model_convergence_failure",
            f"{name} emitted a convergence warning",
        ) from error
    if not np.array_equal(estimator.classes_, np.arange(len(class_order))) or bool(
        np.any(estimator.n_iter_ >= 300)
    ):
        raise BlockedScreen(
            "blocked_model_convergence_failure",
            f"{name} did not converge with the frozen class ordering",
        )
    return FittedMultinomial(name, names, scaler, estimator, class_order)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float))
    )


def probability_diagnostics(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(target_indices), dtype=int)
    for index, target in enumerate(target_indices):
        ranks[index] = int(np.flatnonzero(order[index] == target)[0]) + 1
    realised = probabilities[np.arange(len(target_indices)), target_indices]
    return ranks, realised, prediction_entropy(np.asarray(probabilities, dtype=float))


def expected_calibration_error(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == target_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        upper = (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        mask = (confidence >= edges[index]) & upper
        if mask.any():
            bin_weight = float(weights[mask].sum())
            value += (
                bin_weight
                / total
                * abs(
                    weighted_mean(correct[mask].astype(float), weights[mask])
                    - weighted_mean(confidence[mask], weights[mask])
                )
            )
    return float(value)


def metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    class_order: tuple[str, ...],
) -> dict[str, Any]:
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    ranks, realised, entropy = probability_diagnostics(targets, probabilities)
    mean_entropy = weighted_mean(entropy, weights)
    auc = math.nan
    if set(targets) == set(range(len(class_order))):
        auc = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=np.arange(len(class_order)),
                multi_class="ovr",
                average="macro",
                sample_weight=weights,
            )
        )
    support = frame["target_class"].value_counts().reindex(class_order, fill_value=0)
    predicted_indices = probabilities.argmax(axis=1)
    return {
        "model": model,
        "multiclass_log_loss": float(
            log_loss(
                targets,
                probabilities,
                labels=np.arange(len(class_order)),
                sample_weight=weights,
            )
        ),
        "multiclass_brier": multiclass_brier(targets, probabilities, weights),
        "top_one_accuracy": weighted_mean((ranks <= 1).astype(float), weights),
        "top_two_accuracy": weighted_mean((ranks <= 2).astype(float), weights),
        "top_three_accuracy": weighted_mean((ranks <= 3).astype(float), weights),
        "mean_reciprocal_rank": weighted_mean(1.0 / ranks, weights),
        "mean_probability_realised_class": weighted_mean(realised, weights),
        "macro_ovr_auc": auc,
        "expected_calibration_error": expected_calibration_error(targets, probabilities, weights),
        "prediction_entropy": mean_entropy,
        "effective_candidate_count": math.exp(mean_entropy),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "class_support": json.dumps(
            {str(key): int(value) for key, value in support.items()}, sort_keys=True
        ),
        "predicted_class_distribution": json.dumps(
            {
                label: weighted_mean((predicted_indices == index).astype(float), weights)
                for index, label in enumerate(class_order)
            },
            sort_keys=True,
        ),
        "mean_predicted_probability_distribution": json.dumps(
            {
                label: weighted_mean(probabilities[:, index], weights)
                for index, label in enumerate(class_order)
            },
            sort_keys=True,
        ),
    }


def predecessor_paths() -> tuple[Path, ...]:
    return (
        PREDECESSOR_PANEL,
        PREDECESSOR_PREDICTIONS,
        PREDECESSOR_COEFFICIENTS,
        PREDECESSOR_CONFIGURATIONS,
        PREDECESSOR_METRICS,
        PREDECESSOR_DECISION,
        PREDECESSOR_AUDIT,
        PREDECESSOR_DETERMINISM,
        PREDECESSOR_BOUNDARY,
        PREDECESSOR_SOURCE,
    )


def verify_predecessor_hashes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in predecessor_paths():
        if not path.is_file():
            raise BlockedScreen(
                "blocked_predecessor_population_not_reconstructable",
                f"frozen predecessor artifact is missing: {path.name}",
            )
        digest = sha256_file(path)
        if digest != EXPECTED_HASHES[path.name]:
            raise BlockedScreen(
                "blocked_predecessor_population_not_reconstructable",
                f"frozen predecessor artifact hash differs: {path.name}",
            )
        records.append(
            {
                "repository_relative_path": str(path.relative_to(REPO_ROOT)),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def reconstruct_predecessor_population() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    panel = (
        pd.read_parquet(PREDECESSOR_PANEL)
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    reject_protected_dates(panel)
    archived = (
        pd.read_parquet(PREDECESSOR_PREDICTIONS)
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    configurations = cast(
        dict[str, Any], json.loads(PREDECESSOR_CONFIGURATIONS.read_text(encoding="utf-8"))
    )
    coefficients = cast(
        dict[str, Any], json.loads(PREDECESSOR_COEFFICIENTS.read_text(encoding="utf-8"))
    )
    predecessor_decision = cast(
        dict[str, Any], json.loads(PREDECESSOR_DECISION.read_text(encoding="utf-8"))
    )
    predecessor_audit = cast(
        dict[str, Any], json.loads(PREDECESSOR_AUDIT.read_text(encoding="utf-8"))
    )
    predecessor_determinism = cast(
        dict[str, Any], json.loads(PREDECESSOR_DETERMINISM.read_text(encoding="utf-8"))
    )
    predecessor_boundary = cast(
        dict[str, Any], json.loads(PREDECESSOR_BOUNDARY.read_text(encoding="utf-8"))
    )
    if predecessor_decision.get("decision") != "descriptive_coarse_funnel_only":
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor decision category differs",
        )
    if not predecessor_audit.get("passed") or not predecessor_determinism.get("passed"):
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor audit or determinism result did not pass",
        )
    if predecessor_boundary.get("protected_rows_materialised") != 0:
        raise BlockedScreen(
            "blocked_protected_boundary_failure",
            "predecessor protected row count differs",
        )
    if len(panel) != 15_549 or len(panel) > 20_000:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            f"predecessor decision row count differs: {len(panel)}",
        )
    if set(panel["decision_ordinal"].astype(int).unique()) != {6, 12}:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor checkpoint set differs",
        )
    local_clocks = (
        pd.to_datetime(panel["feature_available_timestamp_utc"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    expected_clocks = panel["decision_ordinal"].astype(int).map({6: "10:00", 12: "10:30"})
    if not bool(local_clocks.eq(expected_clocks).all()):
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure",
            "predecessor checkpoint timestamp convention differs",
        )
    scoring = panel.loc[panel["scoring_eligible"]].copy()
    development = scoring.loc[scoring["year"].eq(2024)]
    assessment = scoring.loc[scoring["year"].eq(2025)].reset_index(drop=True)
    keys = ["symbol", "session", "year_month", "decision_ordinal", "slate_id", "target_class"]
    if not assessment.loc[:, keys].equals(archived.loc[:, keys]):
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor assessment keys, assignments, or targets differ",
        )
    numeric_columns = ["row_weight", "posterior_entropy", "transition_probability"]
    maximum_archived_error = float(
        np.abs(
            assessment.loc[:, numeric_columns].to_numpy(dtype=float)
            - archived.loc[:, numeric_columns].to_numpy(dtype=float)
        ).max(initial=0.0)
    )
    if maximum_archived_error > 1e-12:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor assessment weights or contexts differ",
        )
    features = cast(dict[str, list[str]], configurations["features"])
    expected_counts = {"M0": 16, "M1": 22, "M2": 42}
    if set(features) != set(expected_counts) or any(
        len(features[name]) != count for name, count in expected_counts.items()
    ):
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "predecessor M0/M1/M2 feature ladder differs",
        )
    for feature_names in features.values():
        if not np.isfinite(scoring.loc[:, feature_names].to_numpy(dtype=float)).all():
            raise BlockedScreen(
                "blocked_predecessor_population_not_reconstructable",
                "predecessor feature value is not finite",
            )
    eligible_weights = scoring.groupby("slate_id", sort=True)["row_weight"].sum()
    maximum_slate_weight_error = float(np.abs(eligible_weights.to_numpy(dtype=float) - 1.0).max())
    if maximum_slate_weight_error > 1e-12:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "frozen slate weights do not sum to one",
        )
    class_order = tuple(coefficients["models"]["M2"]["class_order"])
    if class_order != TARGET_CLASSES:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "frozen target class order differs",
        )
    support = predecessor_decision["support"]
    record = {
        **SAFETY_FLAGS,
        "source_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "source_commit": EXPECTED_PREDECESSOR_COMMIT,
        "decision_panel_sha256": EXPECTED_HASHES["decision_panel.parquet"],
        "assessment_predictions_sha256": EXPECTED_HASHES["assessment_predictions.parquet"],
        "rows": len(panel),
        "scoring_rows": len(scoring),
        "development_rows": len(development),
        "development_sessions": int(development["session"].nunique()),
        "development_stocks": int(development["symbol"].nunique()),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "development_assignment": "2024-01-01_through_2024-12-31",
        "assessment_assignment": "2025-01-01_through_2025-08-22",
        "decision_ordinals": [6, 12],
        "local_clocks": ["10:00", "10:30"],
        "target_classes": list(class_order),
        "target_horizon_completed_five_minute_bars": 6,
        "feature_counts": expected_counts,
        "row_weight": "1/eligible_stocks_in_session_checkpoint",
        "maximum_slate_weight_error": maximum_slate_weight_error,
        "maximum_assessment_archive_difference": maximum_archived_error,
        "tied_exclusions": int(panel["raw_outcome"].eq("TIED_REGISTERED_COMPLETION").sum()),
        "source_unavailable_exclusions": int(panel["raw_outcome"].eq("SOURCE_UNAVAILABLE").sum()),
        "assessment_class_support": support["assessment_class_support"],
        "development_class_support": support["development_class_support"],
        "exact_population_reconstructed": True,
        "passed": True,
    }
    context = {
        "configurations": configurations,
        "coefficients": coefficients,
        "predecessor_decision": predecessor_decision,
        "predecessor_audit": predecessor_audit,
        "predecessor_determinism": predecessor_determinism,
        "predecessor_boundary": predecessor_boundary,
    }
    return panel, archived, record, context, cast(dict[str, Any], support)


def reconstruct_frozen_m2(
    panel: pd.DataFrame,
    archived: pd.DataFrame,
    context: Mapping[str, Any],
) -> tuple[FittedMultinomial, np.ndarray, dict[str, Any], dict[str, Any]]:
    scoring = panel.loc[panel["scoring_eligible"]].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy().reset_index(drop=True)
    payload = cast(dict[str, Any], context["coefficients"])["models"]["M2"]
    features = tuple(str(feature) for feature in payload["features"])
    class_order = tuple(str(label) for label in payload["class_order"])
    model = fit_multinomial("T0", development, features=features, class_order=class_order)
    probabilities = model.predict(assessment)
    archived_probabilities = archived.loc[
        :, [f"probability__M2__{label}" for label in class_order]
    ].to_numpy(dtype=float)
    manual_probabilities = manual_multinomial_probabilities(assessment, payload)
    maximum_probability_difference = float(
        np.abs(probabilities - archived_probabilities).max(initial=0.0)
    )
    maximum_manual_probability_difference = float(
        np.abs(manual_probabilities - archived_probabilities).max(initial=0.0)
    )
    actual_metric = metric_row(assessment, probabilities, model="T0", class_order=class_order)
    # The predecessor averaged row-wise effective counts. Preserve that legacy
    # aggregation only for the exact reconstruction comparison; this screen's
    # reported metric follows exp(pooled prediction entropy), as preregistered.
    predecessor_metric = dict(actual_metric)
    predecessor_metric["effective_candidate_count"] = weighted_mean(
        np.exp(prediction_entropy(probabilities)),
        assessment["row_weight"].to_numpy(dtype=float),
    )
    archived_metrics = pd.read_csv(PREDECESSOR_METRICS)
    expected_metric = archived_metrics.loc[archived_metrics["model"].eq("M2")].iloc[0]
    metric_differences = {
        metric: abs(float(predecessor_metric[metric]) - float(expected_metric[metric]))
        for metric in MODEL_NUMERIC_METRICS
    }
    maximum_metric_difference = max(metric_differences.values())
    serialized = model.serialize()
    coefficient_difference = max(
        float(
            np.abs(
                np.asarray(serialized["coefficient"], dtype=float)
                - np.asarray(payload["coefficient"], dtype=float)
            ).max(initial=0.0)
        ),
        float(
            np.abs(
                np.asarray(serialized["intercept"], dtype=float)
                - np.asarray(payload["intercept"], dtype=float)
            ).max(initial=0.0)
        ),
    )
    scaler_difference = max(
        float(
            np.abs(
                np.asarray(serialized["scaler_mean"], dtype=float)
                - np.asarray(payload["scaler_mean"], dtype=float)
            ).max(initial=0.0)
        ),
        float(
            np.abs(
                np.asarray(serialized["scaler_scale"], dtype=float)
                - np.asarray(payload["scaler_scale"], dtype=float)
            ).max(initial=0.0)
        ),
    )
    passed = bool(
        maximum_probability_difference <= 1e-12
        and maximum_manual_probability_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and scaler_difference <= 1e-12
    )
    record = {
        **SAFETY_FLAGS,
        "source_model": "M2",
        "binding_baseline": "T0",
        "class_order": list(class_order),
        "features": list(features),
        "development_rows": len(development),
        "assessment_rows": len(assessment),
        "maximum_permitted_probability_difference": 1e-12,
        "maximum_permitted_metric_difference": 1e-12,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_manual_probability_difference": maximum_manual_probability_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_scaler_difference": scaler_difference,
        "actual_metrics": {metric: predecessor_metric[metric] for metric in MODEL_NUMERIC_METRICS},
        "screen_metrics": {metric: actual_metric[metric] for metric in MODEL_NUMERIC_METRICS},
        "effective_candidate_count_definition": "exp(pooled_prediction_entropy)",
        "predecessor_legacy_effective_candidate_count_definition": (
            "weighted_mean(exp(row_prediction_entropy))"
        ),
        "archived_metrics": {
            metric: float(expected_metric[metric]) for metric in MODEL_NUMERIC_METRICS
        },
        "metric_differences": metric_differences,
        "passed": passed,
    }
    if not passed:
        raise BlockedScreen(
            "blocked_frozen_m2_not_reconstructable",
            "frozen M2 probability, metric, coefficient, or scaler reconstruction differs",
        )
    return model, probabilities, actual_metric, record


def _materialized_parquet(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"PAR1"


def build_anchor_preflight(
    panel: pd.DataFrame,
    *,
    component_ledger_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not _materialized_parquet(component_ledger_path):
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "the frozen behavioural component ledger is absent or is only a Git LFS pointer",
        )
    component_hash = sha256_file(component_ledger_path)
    if component_hash != EXPECTED_COMPONENT_LEDGER_HASH:
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "the frozen behavioural component ledger hash differs",
        )
    ledger_columns = (
        "symbol",
        "session",
        "decision_ordinal",
        "slate_id",
        "feature_available_timestamp_utc",
        "bar_count",
        "bar_start_timestamps_utc",
        "bar_open",
        "bar_high",
        "bar_low",
        "bar_close",
        "historical_relative_activity",
        "activity_normalisation",
    )
    components = (
        pd.read_parquet(component_ledger_path, columns=list(ledger_columns))
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    keys = ["symbol", "session", "decision_ordinal", "slate_id", "feature_available_timestamp_utc"]
    left = panel.loc[:, keys].copy()
    right = components.loc[:, keys].copy()
    for candidate in (left, right):
        candidate["feature_available_timestamp_utc"] = pd.to_datetime(
            candidate["feature_available_timestamp_utc"], utc=True, errors="raise"
        )
    if len(components) != len(panel) or not left.equals(right):
        raise BlockedScreen(
            "blocked_predecessor_population_not_reconstructable",
            "frozen behavioural component keys differ from the predecessor population",
        )
    joined = panel.loc[
        :,
        [
            "symbol",
            "session",
            "year",
            "year_month",
            "decision_ordinal",
            "slate_id",
            "feature_available_timestamp_utc",
            "target_class",
            "scoring_eligible",
            *PRIMARY_BEHAVIOURS,
        ],
    ].copy()
    anchor_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    maximum_after_decision_seconds = -math.inf
    for panel_row, component_row in zip(
        joined.itertuples(index=False), components.itertuples(index=False), strict=True
    ):
        checkpoint = int(panel_row.decision_ordinal)
        anchors = trajectory_anchors(checkpoint)
        starts = pd.to_datetime(
            list(component_row.bar_start_timestamps_utc), utc=True, errors="raise"
        )
        path_lengths = {
            "bar_start_timestamps_utc": len(starts),
            "bar_open": len(component_row.bar_open),
            "bar_high": len(component_row.bar_high),
            "bar_low": len(component_row.bar_low),
            "bar_close": len(component_row.bar_close),
            "historical_relative_activity": len(component_row.historical_relative_activity),
        }
        if set(path_lengths.values()) != {checkpoint} or int(component_row.bar_count) != checkpoint:
            raise BlockedScreen(
                "blocked_chronology_or_leakage_failure",
                "frozen completed-bar path length differs for "
                f"{panel_row.symbol}/{panel_row.session}",
            )
        final_available = pd.Timestamp(panel_row.feature_available_timestamp_utc)
        row_record: dict[str, Any] = {
            "symbol": str(panel_row.symbol),
            "session": str(panel_row.session),
            "year": int(panel_row.year),
            "year_month": str(panel_row.year_month),
            "decision_ordinal": checkpoint,
            "slate_id": str(panel_row.slate_id),
            "feature_available_timestamp_utc": final_available,
            "target_class": panel_row.target_class,
            "scoring_eligible": bool(panel_row.scoring_eligible),
            "frozen_path_bar_count": checkpoint,
            "activity_normalisation": str(component_row.activity_normalisation),
            "anchor_levels_materialised": False,
        }
        row_available = True
        for role, anchor in zip(("e0", "e1", "e2"), anchors, strict=True):
            formula_available, formula_reason = anchor_formula_availability(anchor)
            anchor_complete = pd.Timestamp(starts[anchor - 1]) + pd.Timedelta(minutes=5)
            local_clock = anchor_complete.tz_convert("America/New_York").strftime("%H:%M")
            causal = bool(anchor_complete <= final_available)
            seconds_after = float((anchor_complete - final_available).total_seconds())
            maximum_after_decision_seconds = max(maximum_after_decision_seconds, seconds_after)
            if local_clock != EXPECTED_LOCAL_CLOCKS[anchor] or not causal:
                raise BlockedScreen(
                    "blocked_chronology_or_leakage_failure",
                    f"anchor clock or completed-bar causality differs at bar {anchor}",
                )
            available = formula_available and causal
            row_available = row_available and available
            row_record[f"anchor_{role}_completed_bars"] = anchor
            row_record[f"anchor_{role}_available_timestamp_utc"] = anchor_complete
            row_record[f"anchor_{role}_local_clock"] = local_clock
            row_record[f"anchor_{role}_causal"] = causal
            row_record[f"anchor_{role}_formula_available"] = formula_available
            row_record[f"anchor_{role}_unavailable_reason"] = formula_reason
            anchor_records.append(
                {
                    "symbol": str(panel_row.symbol),
                    "session": str(panel_row.session),
                    "year": int(panel_row.year),
                    "year_month": str(panel_row.year_month),
                    "decision_ordinal": checkpoint,
                    "anchor_role": role.upper(),
                    "anchor_completed_bars": anchor,
                    "anchor_available_timestamp_utc": anchor_complete,
                    "anchor_local_clock": local_clock,
                    "causal": causal,
                    "formula_available": formula_available,
                    "available": available,
                    "unavailable_reason": formula_reason,
                }
            )
        row_record["complete_trajectory_available"] = row_available
        for emotion in PRIMARY_BEHAVIOURS:
            row_record[f"{emotion}_current_level"] = float(getattr(panel_row, emotion))
        trajectory_rows.append(row_record)
    anchor_long = pd.DataFrame(anchor_records)
    trajectory_ledger = (
        pd.DataFrame(trajectory_rows)
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    missingness = trajectory_missingness(anchor_long)
    complete = trajectory_ledger["complete_trajectory_available"]
    scoring_mask = trajectory_ledger["scoring_eligible"]
    development_mask = scoring_mask & trajectory_ledger["year"].eq(2024)
    assessment_mask = scoring_mask & trajectory_ledger["year"].eq(2025)
    complete_development = trajectory_ledger.loc[complete & development_mask].copy()
    complete_assessment = trajectory_ledger.loc[complete & assessment_mask].copy()
    predecessor_assessment_rows = int(assessment_mask.sum())
    assessment_stock_share = complete_assessment["symbol"].value_counts(normalize=True)
    assessment_class_share = complete_assessment["target_class"].value_counts(normalize=True)
    assessment_support = (
        complete_assessment["target_class"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    )
    overall_retention = float(complete.mean())
    scoring_retention = float(complete.loc[scoring_mask].mean())
    assessment_retention = float(complete.loc[assessment_mask].mean())
    support_gates = {
        "assessment_rows_at_least_5500": len(complete_assessment) >= 5_500,
        "assessment_sessions_at_least_140": complete_assessment["session"].nunique() >= 140,
        "all_20_stocks": complete_assessment["symbol"].nunique() == 20,
        "eight_assessment_months": complete_assessment["year_month"].nunique() == 8,
        "complete_trajectory_retention_at_least_95_percent": assessment_retention >= 0.95,
        "every_assessment_target_class_at_least_50": bool(assessment_support.ge(50).all()),
        "maximum_stock_share_at_most_10_percent": float(assessment_stock_share.max()) <= 0.10,
        "maximum_target_class_share_at_most_75_percent": float(assessment_class_share.max())
        <= 0.75,
    }
    support = {
        **SAFETY_FLAGS,
        "predecessor_rows": len(trajectory_ledger),
        "complete_trajectory_rows": int(complete.sum()),
        "complete_trajectory_retention": overall_retention,
        "scoring_complete_trajectory_rows": int((complete & scoring_mask).sum()),
        "scoring_complete_trajectory_retention": scoring_retention,
        "development_predecessor_rows": int(development_mask.sum()),
        "development_complete_trajectory_rows": len(complete_development),
        "assessment_predecessor_rows": predecessor_assessment_rows,
        "assessment_rows": len(complete_assessment),
        "assessment_complete_trajectory_rows": int((complete & assessment_mask).sum()),
        "assessment_complete_trajectory_retention": assessment_retention,
        "assessment_sessions": int(complete_assessment["session"].nunique()),
        "assessment_stocks": int(complete_assessment["symbol"].nunique()),
        "assessment_months": int(complete_assessment["year_month"].nunique()),
        "assessment_class_support": {
            str(key): int(value) for key, value in assessment_support.items()
        },
        "maximum_assessment_stock_share": float(assessment_stock_share.max()),
        "maximum_assessment_class_share": float(assessment_class_share.max()),
        "gates": support_gates,
        "failed_gates": [name for name, passed in support_gates.items() if not passed],
        "passed": all(support_gates.values()),
    }
    manifest = {
        **SAFETY_FLAGS,
        "completed_bar_convention": (
            "bar_ordinal_is_zero_based_start; anchor_count_includes "
            "bars_0_through_anchor_minus_1; features_available_at_last_bar_start_plus_5_minutes"
        ),
        "anchors": {
            "6": {
                "completed_bar_counts": [2, 4, 6],
                "local_clocks": ["09:40", "09:50", "10:00"],
            },
            "12": {
                "completed_bar_counts": [6, 9, 12],
                "local_clocks": ["10:00", "10:15", "10:30"],
            },
        },
        "frozen_formula_precondition": (
            "opening_raw_components requires count >= 2 and count % 2 == 0"
        ),
        "bar_9_formula_available": False,
        "bar_9_unavailable_reason": (
            "frozen_opening_raw_components_requires_even_completed_bar_count"
        ),
        "substitution_or_alternative_split_used": False,
        "earlier_anchor_future_bars_used": False,
        "maximum_anchor_timestamp_after_final_decision_seconds": maximum_after_decision_seconds,
        "component_ledger_sha256": component_hash,
        "component_ledger_rows": len(components),
        "activity_normalisation": sorted(components["activity_normalisation"].astype(str).unique()),
        "support": support,
        "passed_causality": True,
    }
    return trajectory_ledger, missingness, manifest, support


def trajectory_missingness(anchor_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    breakdowns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("emotion_anchor", ("decision_ordinal", "anchor_role", "anchor_completed_bars")),
        ("stock", ("symbol", "decision_ordinal", "anchor_role", "anchor_completed_bars")),
        ("month", ("year_month", "decision_ordinal", "anchor_role", "anchor_completed_bars")),
        ("checkpoint", ("decision_ordinal", "anchor_role", "anchor_completed_bars")),
    )
    for emotion in PRIMARY_BEHAVIOURS:
        for breakdown_type, group_columns in breakdowns:
            for group_values, subset in anchor_long.groupby(list(group_columns), sort=True):
                values = group_values if isinstance(group_values, tuple) else (group_values,)
                record: dict[str, Any] = {
                    "breakdown_type": breakdown_type,
                    "emotion": emotion,
                    "symbol": None,
                    "year_month": None,
                    "decision_ordinal": None,
                    "anchor_role": None,
                    "anchor_completed_bars": None,
                }
                record.update(dict(zip(group_columns, values, strict=True)))
                missing = ~subset["available"]
                reasons = sorted(
                    value
                    for value in subset.loc[missing, "unavailable_reason"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                record.update(
                    {
                        "rows": len(subset),
                        "missing_rows": int(missing.sum()),
                        "missing_percent": 100.0 * float(missing.mean()),
                        "unavailable_reason": "|".join(reasons),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def assessment_with_t0_predictions(
    panel: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    class_order: tuple[str, ...],
) -> pd.DataFrame:
    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)].copy()
    assessment = assessment.reset_index(drop=True)
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = assessment["target_class"].map(class_index).to_numpy(dtype=int)
    ranks, realised, entropy = probability_diagnostics(targets, probabilities)
    output = assessment.loc[
        :,
        [
            "symbol",
            "session",
            "year_month",
            "decision_ordinal",
            "slate_id",
            "target_class",
            "row_weight",
            "posterior_entropy",
            "transition_probability",
        ],
    ].copy()
    for index, label in enumerate(class_order):
        output[f"probability__T0__{label}"] = probabilities[:, index]
    output["realised_probability__T0"] = realised
    output["realised_rank__T0"] = ranks
    output["prediction_entropy__T0"] = entropy
    output["effective_candidate_count__T0"] = np.exp(entropy)
    output["trajectory_support_status"] = "not_fitted_blocked_insufficient_trajectory_support"
    return output


def sliced_t0_metrics(
    assessment: pd.DataFrame,
    model: FittedMultinomial,
    *,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, subset in assessment.groupby(group_column, sort=True):
        row = metric_row(
            subset,
            model.predict(subset),
            model="T0",
            class_order=model.class_order,
        )
        row[group_column] = group
        rows.append(row)
    return pd.DataFrame(rows)


def t0_class_metrics(
    assessment: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    class_order: tuple[str, ...],
) -> pd.DataFrame:
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = assessment["target_class"].map(class_index).to_numpy(dtype=int)
    weights = assessment["row_weight"].to_numpy(dtype=float)
    ranks, _, _ = probability_diagnostics(targets, probabilities)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(class_order):
        mask = targets == index
        binary = mask.astype(int)
        rows.append(
            {
                "model": "T0",
                "target_class": label,
                "support": int(mask.sum()),
                "mean_realised_probability": weighted_mean(
                    probabilities[mask, index], weights[mask]
                ),
                "top_one_accuracy": weighted_mean((ranks[mask] <= 1).astype(float), weights[mask]),
                "top_two_accuracy": weighted_mean((ranks[mask] <= 2).astype(float), weights[mask]),
                "top_three_accuracy": weighted_mean(
                    (ranks[mask] <= 3).astype(float), weights[mask]
                ),
                "ovr_auc": float(
                    roc_auc_score(binary, probabilities[:, index], sample_weight=weights)
                ),
                "ovr_brier": weighted_mean(np.square(probabilities[:, index] - binary), weights),
                "trajectory_comparison_available": False,
            }
        )
    return pd.DataFrame(rows)


def t0_concentration_metrics(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    model: FittedMultinomial,
    support: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    entropy_median = float(development["posterior_entropy"].median())
    transition_median = float(development["transition_probability"].median())
    grouped = assessment.copy()
    grouped["group"] = np.where(grouped["posterior_entropy"] <= entropy_median, "low", "high")
    rows: list[dict[str, Any]] = []
    for group, subset in grouped.groupby("group", sort=True):
        row = metric_row(
            subset,
            model.predict(subset),
            model="T0",
            class_order=model.class_order,
        )
        row.update(
            {
                "breakdown_type": "development_frozen_posterior_entropy",
                "group": group,
                "value": math.nan,
                "trajectory_comparison_available": False,
            }
        )
        rows.append(row)
    grouped["group"] = np.where(
        grouped["transition_probability"] <= transition_median, "low", "high"
    )
    for group, subset in grouped.groupby("group", sort=True):
        row = metric_row(
            subset,
            model.predict(subset),
            model="T0",
            class_order=model.class_order,
        )
        row.update(
            {
                "breakdown_type": "development_frozen_transition_probability",
                "group": group,
                "value": math.nan,
                "trajectory_comparison_available": False,
            }
        )
        rows.append(row)
    for name in (
        "maximum_assessment_stock_share",
        "maximum_assessment_class_share",
        "assessment_complete_trajectory_retention",
    ):
        rows.append(
            {
                "breakdown_type": "support_concentration",
                "group": name,
                "model": "ALL",
                "value": float(support[name]),
                "trajectory_comparison_available": False,
            }
        )
    return pd.DataFrame(rows), {
        "posterior_entropy_development_median": entropy_median,
        "transition_probability_development_median": transition_median,
    }


def blocked_trajectory_diagnostics(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for emotion in PRIMARY_BEHAVIOURS:
        rows.append(
            {
                "emotion": emotion,
                "scope": "pooled_assessment",
                "level_mean": float(assessment[emotion].mean()),
                "level_standard_deviation": float(assessment[emotion].std(ddof=1)),
                "change_mean": math.nan,
                "change_standard_deviation": math.nan,
                "acceleration_mean": math.nan,
                "acceleration_standard_deviation": math.nan,
                "reversal_frequency": math.nan,
                "persistence_frequency": math.nan,
                "peak_displacement_mean": math.nan,
                "correlation_with_current_level": math.nan,
                "correlation_with_each_other_trajectory": None,
                "development_target_class_rate_by_frozen_quintile": None,
                "assessment_target_class_rate_by_frozen_quintile": None,
                "level_and_change_opposite_sign_percent": math.nan,
                "reverses_before_decision_percent": math.nan,
                "at_local_peak_at_decision_percent": math.nan,
                "status": "not_calculated_blocked_insufficient_trajectory_support",
            }
        )
    return pd.DataFrame(rows)


def blocked_bootstrap_metrics() -> pd.DataFrame:
    metrics = (
        "t1_minus_t0_log_loss_improvement",
        "t1_minus_t0_brier_improvement",
        "t1_minus_t0_top_two_improvement",
        "t1_minus_t0_realised_probability_improvement",
        "t2_minus_t1_log_loss_improvement",
        "t2_minus_t1_brier_improvement",
        "t2_minus_t1_top_two_improvement",
        "t2_minus_t1_realised_probability_improvement",
        "t2_minus_t1_prediction_entropy_reduction",
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "real_value": math.nan,
                "draw_mean": math.nan,
                "interval_90_lower": math.nan,
                "interval_90_upper": math.nan,
                "interval_95_lower": math.nan,
                "interval_95_upper": math.nan,
                "draw_count": 0,
                "planned_draw_count": 50,
                "seed": 20260722,
                "status": "not_run_pre_model_support_blocker",
            }
            for metric in metrics
        ]
    )


def blocked_null_metrics() -> pd.DataFrame:
    metrics = (
        "t1_minus_t0_log_loss_improvement",
        "t1_minus_t0_brier_improvement",
        "t2_minus_t1_log_loss_improvement",
        "t2_minus_t1_brier_improvement",
        "t2_minus_t1_top_two_improvement",
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "real_value": math.nan,
                "null_mean": math.nan,
                "null_q90": math.nan,
                "real_percentile": math.nan,
                "draw_count": 0,
                "planned_draw_count": 10,
                "seed_base": 20260723,
                "status": "not_run_pre_model_support_blocker",
            }
            for metric in metrics
        ]
    )


def render_report(
    *,
    decision: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    frozen_m2: Mapping[str, Any],
    t0_metrics: Mapping[str, Any],
) -> str:
    support = cast(Mapping[str, Any], decision["trajectory_support"])
    lines = [
        "# Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0",
        "",
        "Retrospective, research-only, observable structural feasibility screen. Economic "
        "outcomes were not opened; execution, broker integration, and strategy promotion remained "
        "disabled.",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "## Binding preflight result",
        "",
        "The frozen behavioural constructor requires an even number of completed bars. The "
        "preregistered ordinal-12 middle anchor is bar 9, so it cannot be evaluated with the "
        "frozen formula. No 4/5 split, alternate anchor, or revised formula was substituted.",
        "",
        f"- Complete-trajectory retention: {support['complete_trajectory_rows']}/"
        f"{support['predecessor_rows']} ({100.0 * support['complete_trajectory_retention']:.4f}%).",
        f"- Assessment retention: {support['assessment_complete_trajectory_rows']}/"
        f"{support['assessment_predecessor_rows']} "
        f"({100.0 * support['assessment_complete_trajectory_retention']:.4f}%).",
        f"- Final complete assessment rows: {support['assessment_rows']} (required: 5,500).",
        "- Required retention: at least 95%.",
        "- T1 and T2 were not fitted; bootstrap and null draws were not run after this binding "
        "support failure.",
        "",
        "## Frozen predecessor and T0 reconstruction",
        "",
        f"- Predecessor rows: {predecessor['rows']}; development: "
        f"{predecessor['development_rows']}; assessment: {predecessor['assessment_rows']}.",
        f"- Frozen M2 maximum probability difference: "
        f"{frozen_m2['maximum_probability_difference']:.3g}.",
        f"- Frozen M2 maximum metric difference: {frozen_m2['maximum_metric_difference']:.3g}.",
        "",
        "```text",
        pd.DataFrame([t0_metrics])[
            [
                "model",
                "multiclass_log_loss",
                "multiclass_brier",
                "top_one_accuracy",
                "top_two_accuracy",
                "mean_probability_realised_class",
                "prediction_entropy",
                "effective_candidate_count",
            ]
        ].to_string(index=False),
        "```",
        "",
        "This blocked feasibility result is not prospective validation and provides no evidence "
        "of economic value, price-direction edge, trading utility, or achieved P&L.",
        "",
    ]
    return "\n".join(lines)


def determinism_check(
    output: Path,
    *,
    original_model: FittedMultinomial,
    original_probabilities: np.ndarray,
    original_metric: Mapping[str, Any],
    original_decision: Mapping[str, Any],
) -> dict[str, Any]:
    panel = pd.read_parquet(output / "decision_panel.parquet")
    scoring = panel.loc[panel["scoring_eligible"]].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy().reset_index(drop=True)
    refit = fit_multinomial(
        "T0",
        development,
        features=original_model.features,
        class_order=original_model.class_order,
    )
    refit_probabilities = refit.predict(assessment)
    refit_metric = metric_row(
        assessment,
        refit_probabilities,
        model="T0",
        class_order=refit.class_order,
    )
    probability_difference = float(
        np.abs(refit_probabilities - original_probabilities).max(initial=0.0)
    )
    coefficient_difference = max(
        float(np.abs(refit.estimator.coef_ - original_model.estimator.coef_).max(initial=0.0)),
        float(
            np.abs(refit.estimator.intercept_ - original_model.estimator.intercept_).max(
                initial=0.0
            )
        ),
    )
    scaler_difference = max(
        float(np.abs(refit.scaler.mean_ - original_model.scaler.mean_).max(initial=0.0)),
        float(np.abs(refit.scaler.scale_ - original_model.scaler.scale_).max(initial=0.0)),
    )
    metric_difference = max(
        abs(float(refit_metric[metric]) - float(original_metric[metric]))
        for metric in MODEL_NUMERIC_METRICS
    )
    availability = (
        panel["decision_ordinal"]
        .astype(int)
        .map(
            lambda ordinal: all(
                anchor_formula_availability(anchor)[0] for anchor in trajectory_anchors(ordinal)
            )
        )
    )
    retention = float(availability.mean())
    expected_retention = float(
        original_decision["trajectory_support"]["complete_trajectory_retention"]
    )
    final_decision = decide_trajectory_screen(
        t1_pass=False,
        t2_pass=False,
        descriptive_structure=False,
        blocker=(
            "blocked_insufficient_trajectory_support"
            if retention < 0.95
            else "blocked_reproducibility_or_audit_failure"
        ),
    )
    class_order_equal = refit.class_order == original_model.class_order and np.array_equal(
        refit.estimator.classes_, original_model.estimator.classes_
    )
    passed = bool(
        probability_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and scaler_difference <= 1e-12
        and metric_difference <= 1e-12
        and abs(retention - expected_retention) <= 1e-12
        and class_order_equal
        and final_decision == original_decision["decision"]
    )
    result = {
        **SAFETY_FLAGS,
        "method": "reload_frozen_panel_refit_T0_and_reproduce_binding_support_blocker",
        "requested_three_model_refit_not_applicable": True,
        "not_fitted_models": ["T1", "T2"],
        "not_fitted_reason": "blocked_insufficient_trajectory_support_before_model_fitting",
        "maximum_permitted_probability_difference": 1e-12,
        "maximum_probability_difference": probability_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_scaler_difference": scaler_difference,
        "maximum_pooled_metric_difference": metric_difference,
        "trajectory_retention_difference": abs(retention - expected_retention),
        "class_order_equal": class_order_equal,
        "final_decision_equal": final_decision == original_decision["decision"],
        "passed": passed,
    }
    write_json(output / "determinism_check.json", result)
    if not passed:
        raise BlockedScreen(
            "blocked_reproducibility_or_audit_failure",
            "fast blocked-path determinism check failed",
        )
    return result


def write_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    input_hashes: Sequence[Mapping[str, Any]],
    component_ledger_path: Path,
    panel: pd.DataFrame,
    predecessor_record: Mapping[str, Any],
    context: Mapping[str, Any],
    model: FittedMultinomial,
    probabilities: np.ndarray,
    t0_metric: Mapping[str, Any],
    frozen_m2: Mapping[str, Any],
    trajectory_ledger: pd.DataFrame,
    missingness: pd.DataFrame,
    anchor_manifest: Mapping[str, Any],
    trajectory_support: Mapping[str, Any],
    predecessor_support: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    source_manifest = cast(
        dict[str, Any], json.loads(PREDECESSOR_SOURCE.read_text(encoding="utf-8"))
    )
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "dates_read": {
                "start": "2024-01-01",
                "end_inclusive": "2025-08-22",
                "protected_start": "2025-08-23",
            },
            "raw_market_data_read": False,
            "raw_data_downloaded": False,
            "predecessor": {
                "experiment": "Emotion × Regime-Mix Coarse Loop-Family Funnel V0",
                "commit": EXPECTED_PREDECESSOR_COMMIT,
                "decision": "descriptive_coarse_funnel_only",
                "artifacts": list(input_hashes),
                "audit_passed": bool(context["predecessor_audit"]["passed"]),
                "determinism_passed": bool(context["predecessor_determinism"]["passed"]),
            },
            "behavioural_component_ledger": {
                "logical_repository_path": (
                    "research/observable-behavioural-state/"
                    "20260721-behavioural-state-dimensions-screen-v0/artifacts/primary/"
                    "behavioural_component_ledger.parquet"
                ),
                "sha256": sha256_file(component_ledger_path),
                "size_bytes": component_ledger_path.stat().st_size,
            },
            "predecessor_market_source": source_manifest["market_sources"],
            "protected_rows_materialised": 0,
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "read_start": "2024-01-01",
            "read_end_inclusive": "2025-08-22",
            "protected_start": "2025-08-23",
            "maximum_timestamp_materialised": source_manifest["market_sources"][
                "maximum_timestamp_read"
            ],
            "protected_rows_materialised": 0,
            "date_predicate_inherited_from_frozen_predecessor": True,
            "passed": True,
        },
    )
    write_json(output / "predecessor_population_reconstruction.json", predecessor_record)
    write_json(output / "frozen_m2_reconstruction.json", frozen_m2)
    write_json(output / "trajectory_anchor_manifest.json", anchor_manifest)
    write_json(
        output / "trajectory_feature_manifest.json",
        {
            **SAFETY_FLAGS,
            "emotions": list(PRIMARY_BEHAVIOURS),
            "anchor_notation": {"E0": "earliest", "E1": "middle", "E2": "final"},
            "formulas": {
                "change": "E2-E0",
                "recent_change": "E2-E1",
                "acceleration": "(E2-E1)-(E1-E0)",
                "persistence": "1_if_E0<E1<E2;-1_if_E0>E1>E2;0_otherwise",
                "reversal": "1_if_nonzero_consecutive_changes_have_different_signs_else_0",
                "peak_displacement": "E2-max(E0,E1,E2)",
            },
            "primary_forms": list(PRIMARY_FORMS),
            "primary_trajectory_features": list(PRIMARY_TRAJECTORY_FEATURES),
            "primary_trajectory_feature_count": len(PRIMARY_TRAJECTORY_FEATURES),
            "reporting_only_forms": list(DESCRIPTIVE_FORMS),
            "all_trajectory_features": list(ALL_TRAJECTORY_FEATURES),
            "signed_dimension_final_sign_retained_in_T0": True,
            "feature_search_performed": False,
            "trajectory_window_search_performed": False,
            "materialisation_status": (
                "stopped_before_feature_calculation_due_to_anchor_support_preflight"
            ),
        },
    )
    write_csv(output / "trajectory_missingness.csv", missingness)
    write_json(
        output / "interaction_manifest.json",
        {
            **SAFETY_FLAGS,
            "interaction_count": len(TRAJECTORY_INTERACTION_FEATURES),
            "interactions": [
                {
                    "feature": feature,
                    "regime_term": regime,
                    "trajectory_term": trajectory,
                }
                for feature, regime, trajectory in TRAJECTORY_INTERACTION_SPECS
            ],
            "development_only_clip_quantiles": [0.01, 0.99],
            "clip_bounds": None,
            "status": "not_fitted_due_to_pre_model_support_blocker",
            "additional_interactions_created": False,
        },
    )
    shutil.copyfile(PREDECESSOR_PANEL, output / "decision_panel.parquet")
    write_parquet(output / "trajectory_ledger.parquet", trajectory_ledger)
    t0_features = tuple(model.features)
    t1_features = (*t0_features, *PRIMARY_TRAJECTORY_FEATURES)
    t2_features = (*t1_features, *TRAJECTORY_INTERACTION_FEATURES)
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "requested_configuration": {
                "penalty": "l2",
                "C": 0.25,
                "solver": "lbfgs",
                "max_iter": 300,
                "multi_class": "multinomial",
                "class_weight": None,
                "random_state": MODEL_SEED,
                "n_jobs": 1,
            },
            "effective_multiclass_handling": (
                "scikit-learn_lbfgs_automatic_multinomial; multi_class keyword removed in "
                "sklearn 1.9"
            ),
            "features": {
                "T0": list(t0_features),
                "T1": list(t1_features),
                "T2": list(t2_features),
            },
            "planned_primary_model_count": 3,
            "actual_primary_fitted_model_count": 1,
            "fitted_models": ["T0"],
            "not_fitted_models": ["T1", "T2"],
            "not_fitted_reason": "blocked_insufficient_trajectory_support",
            "row_weight": "frozen_predecessor_slate_weight",
            "preprocessor": "StandardScaler_fit_on_2024_only",
            "bootstrap_draws_planned": 50,
            "bootstrap_draws_run": 0,
            "trajectory_null_draws_planned": 10,
            "trajectory_null_draws_run": 0,
        },
    )
    write_json(
        output / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "models": {"T0": model.serialize()},
            "T1": {"status": "not_fitted", "reason": "blocked_insufficient_trajectory_support"},
            "T2": {"status": "not_fitted", "reason": "blocked_insufficient_trajectory_support"},
        },
    )
    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)].copy()
    development = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2024)].copy()
    assessment_predictions = assessment_with_t0_predictions(
        panel, probabilities, class_order=model.class_order
    )
    write_parquet(output / "assessment_predictions.parquet", assessment_predictions)
    pooled = pd.DataFrame([{**t0_metric, "trajectory_comparison_available": False}])
    monthly = sliced_t0_metrics(assessment, model, group_column="year_month")
    monthly["trajectory_comparison_available"] = False
    checkpoint = sliced_t0_metrics(assessment, model, group_column="decision_ordinal")
    checkpoint["trajectory_comparison_available"] = False
    per_class = t0_class_metrics(assessment, probabilities, class_order=model.class_order)
    diagnostics = blocked_trajectory_diagnostics(assessment)
    bootstrap = blocked_bootstrap_metrics()
    null = blocked_null_metrics()
    concentration, frozen_medians = t0_concentration_metrics(
        development, assessment, model, trajectory_support
    )
    write_csv(output / "pooled_metrics.csv", pooled)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "class_metrics.csv", per_class)
    write_csv(output / "trajectory_diagnostics.csv", diagnostics)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", null)
    write_csv(output / "concentration_metrics.csv", concentration)
    decision = {
        **SAFETY_FLAGS,
        "decision": decide_trajectory_screen(
            t1_pass=False,
            t2_pass=False,
            descriptive_structure=False,
            blocker="blocked_insufficient_trajectory_support",
        ),
        "binding_question": contract["binding_question"],
        "blocker_detail": (
            "The frozen behavioural formula rejects the required nine-completed-bar middle anchor "
            "for ordinal 12; complete assessment trajectory retention is below 95%."
        ),
        "predecessor_population_reconstructed": True,
        "frozen_m2_reconstructed": True,
        "trajectory_support": dict(trajectory_support),
        "predecessor_support": dict(predecessor_support),
        "frozen_split_medians": frozen_medians,
        "T0": {"status": "reconstructed", "metrics": dict(t0_metric)},
        "T1": {"status": "not_fitted", "reason": "pre_model_support_gate_failed"},
        "T2": {"status": "not_fitted", "reason": "pre_model_support_gate_failed"},
        "bootstrap_draws_run": 0,
        "trajectory_null_draws_run": 0,
        "alternative_anchor_or_formula_used": False,
        "determinism_check_passed": False,
        "lightweight_audit_passed": False,
    }
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        predecessor=predecessor_record,
        frozen_m2=frozen_m2,
        t0_metrics=t0_metric,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return decision, pooled, monthly, checkpoint, per_class


def execute_screen(output: Path, *, component_ledger_path: Path) -> dict[str, Any]:
    contract = load_contract()
    input_hashes = verify_predecessor_hashes()
    panel, archived, predecessor_record, context, predecessor_support = (
        reconstruct_predecessor_population()
    )
    model, probabilities, t0_metric, frozen_m2 = reconstruct_frozen_m2(panel, archived, context)
    trajectory_ledger, missingness, anchor_manifest, trajectory_support = build_anchor_preflight(
        panel, component_ledger_path=component_ledger_path
    )
    if bool(trajectory_support["passed"]):
        raise BlockedScreen(
            "blocked_quick_trajectory_resource_limit",
            "the frozen input unexpectedly passed the preregistered support preflight; this "
            "blocked-path implementation will not broaden into an unreviewed experiment",
        )
    decision, _, _, _, _ = write_artifacts(
        output,
        contract=contract,
        input_hashes=input_hashes,
        component_ledger_path=component_ledger_path,
        panel=panel,
        predecessor_record=predecessor_record,
        context=context,
        model=model,
        probabilities=probabilities,
        t0_metric=t0_metric,
        frozen_m2=frozen_m2,
        trajectory_ledger=trajectory_ledger,
        missingness=missingness,
        anchor_manifest=anchor_manifest,
        trajectory_support=trajectory_support,
        predecessor_support=predecessor_support,
    )
    determinism = determinism_check(
        output,
        original_model=model,
        original_probabilities=probabilities,
        original_metric=t0_metric,
        original_decision=decision,
    )
    decision["determinism_check_passed"] = bool(determinism["passed"])
    write_json(output / "decision.json", decision)
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    from audit_screen_v0 import audit_artifacts

    audit = audit_artifacts(
        output,
        predecessor_primary=PREDECESSOR_PRIMARY,
        component_ledger_path=component_ledger_path,
    )
    if not audit.get("passed"):
        raise BlockedScreen(
            "blocked_reproducibility_or_audit_failure",
            "lightweight independent audit failed",
        )
    decision["lightweight_audit_passed"] = True
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        predecessor=predecessor_record,
        frozen_m2=frozen_m2,
        t0_metrics=t0_metric,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_unexpected_blocker(output: Path, blocker: BlockedScreen) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "decision": blocker.code,
        "blocker_detail": blocker.detail,
        "determinism_check_passed": False,
        "lightweight_audit_passed": False,
    }
    write_json(output / "decision.json", decision)
    write_json(output / "contract.json", load_contract())
    report = (
        "# Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0\n\n"
        f"Decision: `{blocker.code}`\n\n"
        f"Blocked fail-closed: {blocker.detail}\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--behavioural-component-ledger",
        type=Path,
        default=Path(
            os.environ.get("STOCKER_BEHAVIOURAL_COMPONENT_LEDGER", DEFAULT_COMPONENT_LEDGER)
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    component_ledger_path = args.behavioural_component_ledger.expanduser().resolve()
    try:
        decision = execute_screen(output, component_ledger_path=component_ledger_path)
        print(canonical_json(decision), end="")
        return 0
    except BlockedScreen as blocker:
        write_unexpected_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
