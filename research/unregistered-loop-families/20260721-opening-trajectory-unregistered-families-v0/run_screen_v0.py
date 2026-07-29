#!/usr/bin/env python3
"""Run the bounded opening-trajectory unregistered-family quick screen V0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.opening_trajectory_unregistered_families_v0 import (  # noqa: E402
    BEHAVIOURS,
    OTHER_FAMILY,
    binary_brier,
    binary_log_loss,
    binary_targets,
    canonical_unregistered_path,
    decide_screen,
    first_unregistered_path,
    hidden_family_census,
    multiclass_brier,
    multiclass_log_loss,
    opening_population,
    permute_group_within_slates,
    permute_trajectory_bundle_within_slates,
    pool_hidden_family,
    select_hidden_families,
    session_block_bootstrap_indices,
    trajectory_feature_names,
)

matplotlib.use("Agg")

PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "behavioural-trajectory"
    / "20260721-behavioural-trajectory-late-loops-v01"
)
PREDECESSOR_ARTIFACTS = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_PANEL = PREDECESSOR_ARTIFACTS / "decision_panel.parquet"
PREDECESSOR_LEDGER = PREDECESSOR_ARTIFACTS / "trajectory_ledger.parquet"
PREDECESSOR_MODELS = PREDECESSOR_ARTIFACTS / "model_configurations.json"
PREDECESSOR_PREDICTIONS = PREDECESSOR_ARTIFACTS / "assessment_predictions.parquet"
DICTIONARY_PATH = (
    REPO_ROOT
    / "research"
    / "slrno-v2"
    / "20260714-regime-loop-handoff"
    / "work"
    / "artifacts"
    / "20260718-loop-event-semantics-v2"
    / "primary"
    / "semantic_loop_dictionary_v2.csv"
)
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0.py"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "opening_phase_only": True,
    "unregistered_structural_event_target": True,
    "development_frozen_unregistered_families": True,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
KEYS = ("symbol", "session", "decision_ordinal")
TRAJECTORY_FEATURES = trajectory_feature_names()
TRAJECTORY_GROUPS: dict[str, tuple[str, ...]] = {
    f"{behaviour.upper()}_TRAJECTORY": tuple(
        f"{behaviour}_{form}" for form in ("change", "acceleration", "reversal")
    )
    for behaviour in BEHAVIOURS
}
BINARY_CLASS_ORDER = (0, 1)
MODEL_SEED = 20260721
BOOTSTRAP_SEED = 20260724
NULL_SEED = 20260725
ATTRIBUTION_SEED = 20260726
BOOTSTRAP_DRAWS = 25
NULL_DRAWS = 5
ATTRIBUTION_DRAWS = 20
CHECKPOINT_MATERIAL_ADVERSITY = -0.001
PROTECTED_START = pd.Timestamp("2025-08-23")
RAW_OUTCOME_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
    "TIED_REGISTERED_COMPLETION",
    "SOURCE_UNAVAILABLE",
)


class ScreenBlocker(RuntimeError):
    """Fail-closed experiment blocker carrying one allowed decision code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_key_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].astype(str).sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    for key, value in SAFETY_FLAGS.items():
        if contract.get(key) != value:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", f"contract safety flag differs: {key}"
            )
    return contract


def normalise_raw_outcome(row: pd.Series) -> str:
    raw = str(row["raw_outcome"])
    if raw in {"REGISTERED_PRIMITIVE", "REGISTERED_REPEAT", "REGISTERED_COMPOSITE"}:
        return "REGISTERED_COMPLETION"
    return raw


def weighted_mean(values: Sequence[float] | np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0 or len(values) == 0:
        raise ValueError("weighted quantile requires data and q in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * float(sorted_weights.sum())
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), len(sorted_values) - 1)
    return float(sorted_values[index])


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, *, bins: int = 10
) -> float:
    result = 0.0
    total = float(weights.sum())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        result += (
            float(weights[mask].sum())
            / total
            * abs(
                weighted_mean(targets[mask], weights[mask])
                - weighted_mean(probabilities[mask], weights[mask])
            )
        )
    return float(result)


def calibration_intercept_slope(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    design = np.column_stack((np.ones(len(clipped)), np.log(clipped / (1.0 - clipped))))
    beta = np.asarray([0.0, 1.0])
    for _ in range(100):
        linear = np.clip(design @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        variance = np.maximum(fitted * (1.0 - fitted), 1e-12)
        hessian = design.T @ ((weights * variance)[:, None] * design)
        score = design.T @ (weights * (targets - fitted))
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        beta += step
        if float(np.max(np.abs(step))) <= 1e-12:
            break
    return float(beta[0]), float(beta[1])


def distribution_overlap(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    positive = targets == 1
    negative = targets == 0
    if not positive.any() or not negative.any():
        return math.nan
    edges = np.linspace(0.0, 1.0, 21)
    positive_mass = np.histogram(probabilities[positive], bins=edges, weights=weights[positive])[
        0
    ].astype(float)
    negative_mass = np.histogram(probabilities[negative], bins=edges, weights=weights[negative])[
        0
    ].astype(float)
    positive_mass /= positive_mass.sum()
    negative_mass /= negative_mass.sum()
    return float(np.minimum(positive_mass, negative_mass).sum())


def top_fraction_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float,
    base_rate: float,
) -> tuple[float, float]:
    selected = probabilities >= threshold
    if not selected.any():
        return math.nan, math.nan
    precision = weighted_mean(targets[selected], weights[selected])
    return precision, float(precision / base_rate) if base_rate > 0.0 else math.nan


@dataclass(slots=True)
class FittedBinary:
    name: str
    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        return np.asarray(self.estimator.predict_proba(self.scaler.transform(matrix))[:, 1])

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "binary",
            "features": list(self.features),
            "class_order": list(BINARY_CLASS_ORDER),
            "coefficient": self.estimator.coef_.tolist(),
            "intercept": self.estimator.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_variance": self.scaler.var_.tolist(),
            "n_iter": self.estimator.n_iter_.astype(int).tolist(),
        }


@dataclass(slots=True)
class FittedMulticlass:
    name: str
    features: tuple[str, ...]
    class_order: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        return np.asarray(self.estimator.predict_proba(self.scaler.transform(matrix)))

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "multiclass",
            "features": list(self.features),
            "class_order": list(self.class_order),
            "coefficient": self.estimator.coef_.tolist(),
            "intercept": self.estimator.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_variance": self.scaler.var_.tolist(),
            "n_iter": self.estimator.n_iter_.astype(int).tolist(),
        }


def logistic_kwargs(*, solver: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.25,
        "solver": solver,
        "max_iter": 300,
        "class_weight": None,
        "random_state": MODEL_SEED,
        "n_jobs": 1,
    }
    if solver == "lbfgs" and "multi_class" in LogisticRegression().get_params():
        kwargs["multi_class"] = "multinomial"
    return kwargs


def fit_binary(
    name: str, frame: pd.DataFrame, features: tuple[str, ...], *, target: str
) -> FittedBinary:
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    labels = frame[target].to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    if set(labels) != {0, 1}:
        raise ScreenBlocker("blocked_stage_a_support_failure", f"{name} lacks both binary classes")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    estimator = LogisticRegression(**logistic_kwargs(solver="liblinear"))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(scaled, labels, sample_weight=weights)
    except ConvergenceWarning as error:
        raise ScreenBlocker(
            "blocked_model_convergence_failure", f"{name} emitted a convergence warning"
        ) from error
    if not np.array_equal(estimator.classes_, np.asarray(BINARY_CLASS_ORDER)) or np.any(
        estimator.n_iter_ >= 300
    ):
        raise ScreenBlocker("blocked_model_convergence_failure", f"{name} did not converge")
    return FittedBinary(name, features, scaler, estimator)


def fit_multiclass(
    name: str,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    target: str,
    class_order: tuple[str, ...],
) -> FittedMulticlass:
    class_index = {label: index for index, label in enumerate(class_order)}
    labels = frame[target].map(class_index)
    if labels.isna().any() or set(labels.astype(int)) != set(range(len(class_order))):
        raise ScreenBlocker(
            "blocked_stage_a_support_failure", f"{name} development family classes are incomplete"
        )
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    weights = frame["row_weight"].to_numpy(dtype=float)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    estimator = LogisticRegression(**logistic_kwargs(solver="lbfgs"))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(scaled, labels.astype(int).to_numpy(), sample_weight=weights)
    except ConvergenceWarning as error:
        raise ScreenBlocker(
            "blocked_model_convergence_failure", f"{name} emitted a convergence warning"
        ) from error
    if not np.array_equal(estimator.classes_, np.arange(len(class_order))) or np.any(
        estimator.n_iter_ >= 300
    ):
        raise ScreenBlocker("blocked_model_convergence_failure", f"{name} did not converge")
    return FittedMulticlass(name, features, class_order, scaler, estimator)


def predecessor_feature_surfaces() -> tuple[tuple[str, ...], tuple[str, ...]]:
    configuration = read_json(PREDECESSOR_MODELS)
    t0 = tuple(str(value) for value in configuration["models"]["T0"]["features"])
    t1 = tuple(str(value) for value in configuration["models"]["T1"]["features"])
    if len(t0) != 48 or len(t1) != 66 or t1 != (*t0, *TRAJECTORY_FEATURES):
        raise ScreenBlocker(
            "blocked_trajectory_features_not_reconstructable",
            "frozen predecessor T0/T1 feature surfaces differ",
        )
    return t0, t1


def reconstruct_trajectories_from_ledger(opening: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ledger = pd.read_parquet(PREDECESSOR_LEDGER)
    ledger = ledger.loc[ledger["decision_ordinal"].isin((6, 12))].copy()
    expected_rows = len(opening) * 3
    if len(ledger) != expected_rows or set(ledger["anchor_role"].astype(str)) != {
        "E0",
        "E1",
        "E2",
    }:
        raise ScreenBlocker(
            "blocked_trajectory_features_not_reconstructable",
            f"opening trajectory ledger has {len(ledger)} rows, expected {expected_rows}",
        )
    result = opening.loc[:, list(KEYS)].copy()
    maximum_difference = 0.0
    for behaviour in BEHAVIOURS:
        pivot = ledger.pivot(
            index=list(KEYS), columns="anchor_role", values=behaviour
        ).reset_index()
        pivot.columns.name = None
        pivot = result.loc[:, list(KEYS)].merge(pivot, on=list(KEYS), validate="one_to_one")
        early = pivot["E0"].to_numpy(dtype=float)
        middle = pivot["E1"].to_numpy(dtype=float)
        final = pivot["E2"].to_numpy(dtype=float)
        first_change = middle - early
        recent_change = final - middle
        reconstructed = {
            behaviour: final,
            f"{behaviour}_change": final - early,
            f"{behaviour}_acceleration": recent_change - first_change,
            f"{behaviour}_reversal": (
                (np.sign(first_change) != np.sign(recent_change))
                & (first_change != 0.0)
                & (recent_change != 0.0)
            ).astype(float),
        }
        for feature, values in reconstructed.items():
            result[feature] = values
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(values - opening[feature].to_numpy(dtype=float)))),
            )
    return result, maximum_difference


def load_opening_panel() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not all(
        path.is_file()
        for path in (
            PREDECESSOR_PANEL,
            PREDECESSOR_LEDGER,
            PREDECESSOR_MODELS,
            PREDECESSOR_PREDICTIONS,
            DICTIONARY_PATH,
        )
    ):
        raise ScreenBlocker(
            "blocked_opening_population_not_reconstructable",
            "a frozen predecessor artifact is missing",
        )
    predecessor = pd.read_parquet(PREDECESSOR_PANEL)
    opening = opening_population(predecessor)
    if len(opening) != 15_549:
        raise ScreenBlocker(
            "blocked_opening_population_not_reconstructable",
            f"opening rows={len(opening)}, expected 15549",
        )
    opening["raw_structural_outcome"] = opening.apply(normalise_raw_outcome, axis=1)
    target_frame = binary_targets(opening["raw_structural_outcome"])
    opening = pd.concat(
        [opening.reset_index(drop=True), target_frame.reset_index(drop=True)], axis=1
    )
    if pd.to_datetime(opening["session"]).ge(PROTECTED_START).any():
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "a protected predecessor row materialised"
        )
    t0_features, t1_features = predecessor_feature_surfaces()
    if opening.loc[:, list(t1_features)].isna().any().any():
        raise ScreenBlocker(
            "blocked_trajectory_features_not_reconstructable", "a frozen opening feature is missing"
        )
    frozen_source = predecessor.loc[
        predecessor["decision_ordinal"].isin((6, 12)), [*KEYS, *t1_features]
    ]
    joined = opening.loc[:, list(KEYS)].merge(
        frozen_source, on=list(KEYS), how="left", validate="one_to_one"
    )
    maximum_t0_difference = max(
        float(
            np.max(
                np.abs(
                    opening[feature].to_numpy(dtype=float) - joined[feature].to_numpy(dtype=float)
                )
            )
        )
        for feature in t0_features
    )
    _, maximum_trajectory_difference = reconstruct_trajectories_from_ledger(opening)
    maximum_feature_difference = max(maximum_t0_difference, maximum_trajectory_difference)
    if maximum_feature_difference > 1e-12:
        raise ScreenBlocker(
            "blocked_trajectory_features_not_reconstructable",
            f"maximum opening feature difference={maximum_feature_difference}",
        )
    eligible = opening["unregistered_event"].notna()
    slate_counts = opening.loc[eligible].groupby("slate_id", sort=True).size()
    expected_weights = (
        opening.loc[eligible, "slate_id"].map((1.0 / slate_counts).to_dict()).to_numpy(dtype=float)
    )
    weight_difference = float(
        np.max(np.abs(opening.loc[eligible, "row_weight"].to_numpy(dtype=float) - expected_weights))
    )
    if weight_difference > 1e-15 or not opening.loc[~eligible, "row_weight"].isna().all():
        raise ScreenBlocker(
            "blocked_opening_population_not_reconstructable", "frozen slate weights differ"
        )
    population_manifest = {
        **SAFETY_FLAGS,
        "predecessor_rows": len(predecessor),
        "opening_rows": len(opening),
        "rows_by_year": {
            str(year): int(value)
            for year, value in opening.groupby("year", sort=True).size().items()
        },
        "rows_by_checkpoint": {
            str(checkpoint): int(value)
            for checkpoint, value in opening.groupby("decision_ordinal", sort=True).size().items()
        },
        "sessions": int(opening["session"].nunique()),
        "stocks": int(opening["symbol"].nunique()),
        "minimum_session": str(opening["session"].min()),
        "maximum_session": str(opening["session"].max()),
        "population_key_sha256": frame_key_hash(opening, KEYS),
        "predecessor_opening_key_sha256": frame_key_hash(frozen_source, KEYS),
        "maximum_row_weight_difference": weight_difference,
        "protected_rows_materialised": 0,
        "passed": True,
    }
    feature_manifest = {
        **SAFETY_FLAGS,
        "T0_features": list(t0_features),
        "T1_features": list(t1_features),
        "T0_feature_count": len(t0_features),
        "trajectory_feature_count": len(TRAJECTORY_FEATURES),
        "maximum_T0_feature_difference": maximum_t0_difference,
        "maximum_trajectory_feature_difference": maximum_trajectory_difference,
        "maximum_feature_difference": maximum_feature_difference,
        "tolerance": 1e-12,
        "anchors": {"6": [2, 4, 6], "12": [4, 8, 12]},
        "passed": True,
    }
    return opening, population_manifest, feature_manifest


def registered_paths_and_manifest() -> tuple[frozenset[tuple[int, ...]], dict[str, Any]]:
    table = pd.read_csv(DICTIONARY_PATH)
    paths: set[tuple[int, ...]] = set()
    for raw in table["all_valid_oriented_paths"].astype(str):
        parsed = ast.literal_eval(raw)
        paths.update(tuple(int(state) for state in path) for path in parsed)
    if not paths:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "registered path dictionary is empty"
        )
    return frozenset(paths), {
        "dictionary_version": str(table["dictionary_version"].iloc[0]),
        "dictionary_hash": str(table["dictionary_hash"].iloc[0]),
        "dictionary_rows": len(table),
        "registered_oriented_paths": len(paths),
        "source_sha256": sha256_file(DICTIONARY_PATH),
    }


def build_unregistered_path_ledger(
    opening: pd.DataFrame, registered_paths: Collection[tuple[int, ...]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unregistered = opening.loc[opening["raw_structural_outcome"].eq("UNREGISTERED_LOOP")]
    for row in unregistered.itertuples(index=False):
        event = first_unregistered_path(
            bar_states=tuple(int(value) for value in row.state_path_through_horizon),
            bar_ordinals=tuple(int(value) for value in row.bar_ordinals_through_horizon),
            decision_bar_ordinal=int(row.repo_bar_start_ordinal),
            decision_event_index=int(row.decision_event_index),
            registered_paths=registered_paths,
            horizon_bars=6,
        )
        if event is None:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                "unregistered path unavailable for "
                f"{row.symbol}/{row.session}/{row.decision_ordinal}",
            )
        expected_bars = float(row.bars_until_completion)
        expected_events = int(row.state_events_until_completion)
        if (
            event.completion_bar_ordinal - int(row.repo_bar_start_ordinal) != expected_bars
            or event.completion_event_index - int(row.decision_event_index) != expected_events
        ):
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure", "unregistered event timing differs"
            )
        canonical = canonical_unregistered_path(event.full_path)
        decision_bar_start = pd.Timestamp(row.bar_start_timestamp)
        event_timestamp = decision_bar_start + pd.Timedelta(
            minutes=5 * (event.completion_bar_ordinal - int(row.repo_bar_start_ordinal))
        )
        event_available = event_timestamp + pd.Timedelta(minutes=5)
        decision_available = pd.Timestamp(row.feature_available_timestamp_utc)
        if (
            event_available <= decision_available
            or event.completion_bar_ordinal > int(row.repo_bar_start_ordinal) + 6
        ):
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                "unregistered event availability is noncausal",
            )
        rows.append(
            {
                "symbol": str(row.symbol),
                "session": str(row.session),
                "year": int(row.year),
                "year_month": str(row.year_month),
                "decision_ordinal": int(row.decision_ordinal),
                "slate_id": str(row.slate_id),
                "decision_timestamp_utc": decision_available,
                "event_timestamp_utc": event_timestamp,
                "event_available_timestamp_utc": event_available,
                "completed_state_transition_sequence": list(event.full_path),
                "origin_state": event.full_path[0],
                "terminal_state": event.full_path[-1],
                "path_length": len(event.full_path) - 1,
                "start_event_index": event.start_event_index,
                "completion_event_index": event.completion_event_index,
                "start_bar_ordinal": event.start_bar_ordinal,
                "completion_bar_ordinal": event.completion_bar_ordinal,
                "bars_until_completion": event.completion_bar_ordinal
                - int(row.repo_bar_start_ordinal),
                "state_events_until_completion": event.completion_event_index
                - int(row.decision_event_index),
                "began_before_decision": event.start_event_index < int(row.decision_event_index),
                "event_available": True,
                **asdict(canonical),
            }
        )
    ledger = pd.DataFrame(rows).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    if len(ledger) != len(unregistered):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "unregistered path ledger rows differ"
        )
    return ledger.reset_index(drop=True)


def binary_metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    target: str,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    labels = frame[target].to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    base_rate = weighted_mean(labels, weights)
    valid_auc = set(labels) == {0, 1}
    auc = (
        float(roc_auc_score(labels, probabilities, sample_weight=weights))
        if valid_auc
        else math.nan
    )
    average_precision = (
        float(average_precision_score(labels, probabilities, sample_weight=weights))
        if valid_auc
        else math.nan
    )
    intercept, slope = calibration_intercept_slope(labels, probabilities, weights)
    realised = np.where(labels == 1, probabilities, 1.0 - probabilities)
    row: dict[str, Any] = {
        "model": model,
        "brier_score": binary_brier(labels, probabilities, weights),
        "log_loss": binary_log_loss(labels, probabilities, weights),
        "auc": auc,
        "average_precision": average_precision,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": expected_calibration_error(labels, probabilities, weights),
        "base_rate": base_rate,
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "predicted_positive_rate": weighted_mean(probabilities >= 0.5, weights),
        "mean_probability_realised_class": weighted_mean(realised, weights),
        "probability_distribution_overlap": distribution_overlap(labels, probabilities, weights),
    }
    if thresholds is not None:
        decile_precision, decile_lift = top_fraction_metrics(
            labels,
            probabilities,
            weights,
            threshold=float(thresholds["top_decile_threshold"]),
            base_rate=base_rate,
        )
        quintile_precision, quintile_lift = top_fraction_metrics(
            labels,
            probabilities,
            weights,
            threshold=float(thresholds["top_quintile_threshold"]),
            base_rate=base_rate,
        )
        row.update(
            {
                "top_decile_threshold": float(thresholds["top_decile_threshold"]),
                "top_decile_precision": decile_precision,
                "top_decile_lift": decile_lift,
                "top_quintile_threshold": float(thresholds["top_quintile_threshold"]),
                "top_quintile_precision": quintile_precision,
                "top_quintile_lift": quintile_lift,
            }
        )
    return row


def multiclass_metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    target: str,
    class_order: tuple[str, ...],
) -> dict[str, Any]:
    class_index = {label: index for index, label in enumerate(class_order)}
    labels = frame[target].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(order[index] == label)[0]) + 1 for index, label in enumerate(labels)]
    )
    realised = probabilities[np.arange(len(labels)), labels]
    confidence = probabilities.max(axis=1)
    correct = order[:, 0] == labels
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0), axis=1
    )
    support = frame[target].value_counts().reindex(class_order, fill_value=0)
    mean_entropy = weighted_mean(entropy, weights)
    return {
        "model": model,
        "multiclass_log_loss": multiclass_log_loss(labels, probabilities, weights),
        "multiclass_brier": multiclass_brier(labels, probabilities, weights),
        "top_one_accuracy": weighted_mean(ranks <= 1, weights),
        "top_two_accuracy": weighted_mean(ranks <= 2, weights),
        "mean_reciprocal_rank": weighted_mean(1.0 / ranks, weights),
        "mean_probability_realised_family": weighted_mean(realised, weights),
        "expected_calibration_error": expected_calibration_error(
            correct.astype(int), confidence, weights
        ),
        "prediction_entropy": mean_entropy,
        "effective_candidate_count": math.exp(mean_entropy),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "family_support": json.dumps(
            {str(key): int(value) for key, value in support.items()}, sort_keys=True
        ),
    }


def binary_probabilities(predictions: pd.DataFrame, model: str) -> np.ndarray:
    return predictions[f"{model}_probability"].to_numpy(dtype=float)


def family_probabilities(
    predictions: pd.DataFrame, model: str, class_order: tuple[str, ...]
) -> np.ndarray:
    return predictions.loc[:, [f"{model}_probability_{label}" for label in class_order]].to_numpy(
        dtype=float
    )


def binary_group_metrics(
    predictions: pd.DataFrame,
    *,
    models: tuple[str, str],
    target: str,
    group_type: str,
    groups: Mapping[str, pd.Series],
    thresholds: Mapping[str, Mapping[str, float]] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, mask in groups.items():
        subset = predictions.loc[mask].copy()
        subset = subset.loc[subset[target].notna()]
        if subset.empty:
            continue
        for model in models:
            row = binary_metric_row(
                subset,
                binary_probabilities(subset, model),
                model=model,
                target=target,
                thresholds=None if thresholds is None else thresholds[model],
            )
            rows.append({"group_type": group_type, "group_value": group_value, **row})
    return pd.DataFrame(rows)


def family_group_metrics(
    predictions: pd.DataFrame,
    *,
    class_order: tuple[str, ...],
    group_type: str,
    groups: Mapping[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, mask in groups.items():
        subset = predictions.loc[mask & predictions["hidden_family_class"].notna()].copy()
        if subset.empty:
            continue
        for model in ("F0", "F1"):
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    **multiclass_metric_row(
                        subset,
                        family_probabilities(subset, model, class_order),
                        model=model,
                        target="hidden_family_class",
                        class_order=class_order,
                    ),
                }
            )
    return pd.DataFrame(rows)


def support_and_concentration(
    opening: pd.DataFrame,
    *,
    path_ledger: pd.DataFrame,
    selected_families: tuple[str, ...],
    development_eligible_family_count: int,
) -> tuple[dict[str, Any], pd.DataFrame, bool, bool]:
    assessment = opening.loc[opening["year"].eq(2025) & opening["unregistered_event"].notna()]
    counts = assessment["raw_structural_outcome"].value_counts()
    stock_share = float(assessment.groupby("symbol").size().max() / len(assessment))
    binary_share = float(assessment["unregistered_event"].value_counts().max() / len(assessment))
    retention = len(opening) / 15_549
    stage_a_conditions = {
        "minimum_rows": len(assessment) >= 5_500,
        "minimum_sessions": assessment["session"].nunique() >= 140,
        "minimum_stocks": assessment["symbol"].nunique() >= 15,
        "eight_months": assessment["year_month"].nunique() == 8,
        "minimum_unregistered": int(counts.get("UNREGISTERED_LOOP", 0)) >= 1_000,
        "minimum_registered": int(counts.get("REGISTERED_COMPLETION", 0)) >= 250,
        "minimum_no_completion": int(counts.get("NO_REGISTERED_COMPLETION", 0)) >= 2_500,
        "maximum_stock_share": stock_share <= 0.10,
        "maximum_binary_class_share": binary_share <= 0.75,
        "minimum_trajectory_retention": retention >= 0.95,
    }
    stage_a_pass = all(stage_a_conditions.values())
    concentration_rows = [
        {
            "stage": "A",
            "population": "opening",
            "gate": "maximum_stock_share",
            "value": stock_share,
            "threshold": 0.10,
            "passed": stock_share <= 0.10,
        },
        {
            "stage": "A",
            "population": "opening",
            "gate": "maximum_binary_class_share",
            "value": binary_share,
            "threshold": 0.75,
            "passed": binary_share <= 0.75,
        },
    ]
    assessment_paths = path_ledger.loc[path_ledger["year"].eq(2025)].copy()
    if not assessment_paths.empty:
        assessment_paths["hidden_family_class"] = assessment_paths["family_id"].map(
            lambda value: pool_hidden_family(str(value), selected_families)
        )
    family_counts = assessment_paths["hidden_family_class"].value_counts()
    selected_with_20 = sum(int(family_counts.get(family, 0)) >= 20 for family in selected_families)
    family_share = (
        float(family_counts.max() / len(assessment_paths)) if len(assessment_paths) else 1.0
    )
    stage_b_stock_share = (
        float(assessment_paths.groupby("symbol").size().max() / len(assessment_paths))
        if len(assessment_paths)
        else 1.0
    )
    stage_b_conditions = {
        "minimum_rows": len(assessment_paths) >= 1_000,
        "minimum_sessions": assessment_paths["session"].nunique() >= 100,
        "minimum_stocks": assessment_paths["symbol"].nunique() >= 15,
        "two_selected_families_with_20": selected_with_20 >= 2,
        "minimum_final_classes": family_counts.size >= 3,
        "maximum_stock_share": stage_b_stock_share <= 0.15,
        "maximum_family_share": family_share <= 0.70,
    }
    stage_b_pass = len(selected_families) >= 2 and all(stage_b_conditions.values())
    concentration_rows.extend(
        [
            {
                "stage": "B",
                "population": "assessment_unregistered",
                "gate": "maximum_stock_share",
                "value": stage_b_stock_share,
                "threshold": 0.15,
                "passed": stage_b_stock_share <= 0.15,
            },
            {
                "stage": "B",
                "population": "assessment_unregistered",
                "gate": "maximum_family_share",
                "value": family_share,
                "threshold": 0.70,
                "passed": family_share <= 0.70,
            },
        ]
    )
    support = {
        **SAFETY_FLAGS,
        "trajectory_retention": retention,
        "stage_a": {
            "rows": len(assessment),
            "sessions": int(assessment["session"].nunique()),
            "stocks": int(assessment["symbol"].nunique()),
            "months": int(assessment["year_month"].nunique()),
            "outcome_counts": {str(key): int(value) for key, value in counts.items()},
            "maximum_stock_share": stock_share,
            "maximum_binary_class_share": binary_share,
            "conditions": stage_a_conditions,
            "passed": stage_a_pass,
        },
        "stage_b": {
            "development_eligible_families": development_eligible_family_count,
            "development_selected_families": len(selected_families),
            "rows": len(assessment_paths),
            "sessions": int(assessment_paths["session"].nunique()),
            "stocks": int(assessment_paths["symbol"].nunique()),
            "family_counts": {str(key): int(value) for key, value in family_counts.items()},
            "selected_families_with_20": selected_with_20,
            "maximum_stock_share": stage_b_stock_share,
            "maximum_family_share": family_share,
            "conditions": stage_b_conditions,
            "passed": stage_b_pass,
        },
    }
    return support, pd.DataFrame(concentration_rows), stage_a_pass, stage_b_pass


PrimaryModel = FittedBinary | FittedMulticlass


def attach_hidden_families(
    opening: pd.DataFrame, path_ledger: pd.DataFrame, selected: tuple[str, ...]
) -> pd.DataFrame:
    family = path_ledger.loc[:, [*KEYS, "family_id"]].copy()
    family["hidden_family_class"] = family["family_id"].map(
        lambda value: pool_hidden_family(str(value), selected)
    )
    return opening.merge(family, on=list(KEYS), how="left", validate="one_to_one")


def fit_primary_models(
    panel: pd.DataFrame,
    *,
    t0_features: tuple[str, ...],
    t1_features: tuple[str, ...],
    selected_families: tuple[str, ...],
    stage_b_supported: bool,
) -> tuple[
    dict[str, PrimaryModel],
    pd.DataFrame,
    dict[str, dict[str, float]],
    tuple[str, ...],
]:
    scoring = panel.loc[panel["unregistered_event"].notna()].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy()
    diagnostic_dev = development.loc[development["registered_completion"].notna()].copy()
    diagnostic_assessment = assessment.loc[assessment["registered_completion"].notna()].copy()
    models: dict[str, PrimaryModel] = {
        "U0": fit_binary("U0", development, t0_features, target="unregistered_event"),
        "U1": fit_binary("U1", development, t1_features, target="unregistered_event"),
        "R0": fit_binary("R0", diagnostic_dev, t0_features, target="registered_completion"),
        "R1": fit_binary("R1", diagnostic_dev, t1_features, target="registered_completion"),
    }
    thresholds: dict[str, dict[str, float]] = {}
    development_weights = development["row_weight"].to_numpy(dtype=float)
    for name in ("U0", "U1"):
        model = cast(FittedBinary, models[name])
        probabilities = model.predict(development)
        thresholds[name] = {
            "top_decile_threshold": weighted_quantile(probabilities, development_weights, 0.90),
            "top_quintile_threshold": weighted_quantile(probabilities, development_weights, 0.80),
        }
    family_class_order: tuple[str, ...] = ()
    if stage_b_supported:
        family_development = development.loc[development["hidden_family_class"].notna()].copy()
        family_class_order = (*selected_families, OTHER_FAMILY)
        if set(family_development["hidden_family_class"].astype(str)) != set(family_class_order):
            raise ScreenBlocker(
                "blocked_stage_a_support_failure", "development hidden-family classes differ"
            )
        models["F0"] = fit_multiclass(
            "F0",
            family_development,
            t0_features,
            target="hidden_family_class",
            class_order=family_class_order,
        )
        models["F1"] = fit_multiclass(
            "F1",
            family_development,
            t1_features,
            target="hidden_family_class",
            class_order=family_class_order,
        )
    predictions = assessment.loc[
        :,
        [
            *KEYS,
            "year",
            "year_month",
            "slate_id",
            "row_weight",
            "raw_structural_outcome",
            "unregistered_event",
            "registered_completion",
            "transition_probability",
            "posterior_entropy",
            "registered_completion_count_before_decision",
            "hidden_family_class",
        ],
    ].copy()
    for name in ("U0", "U1"):
        predictions[f"{name}_probability"] = cast(FittedBinary, models[name]).predict(assessment)
    diagnostic_index = diagnostic_assessment.index
    for name in ("R0", "R1"):
        predictions[f"{name}_probability"] = np.nan
        location = predictions.index.isin(diagnostic_index)
        predictions.loc[location, f"{name}_probability"] = cast(FittedBinary, models[name]).predict(
            diagnostic_assessment
        )
    if stage_b_supported:
        family_assessment = assessment.loc[assessment["hidden_family_class"].notna()].copy()
        family_index = family_assessment.index
        family_location = predictions.index.isin(family_index)
        for name in ("F0", "F1"):
            probabilities = cast(FittedMulticlass, models[name]).predict(family_assessment)
            for index, label in enumerate(family_class_order):
                column = f"{name}_probability_{label}"
                predictions[column] = np.nan
                predictions.loc[family_location, column] = probabilities[:, index]
    return models, predictions.reset_index(drop=True), thresholds, family_class_order


def all_stage_metrics(
    predictions: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    family_class_order: tuple[str, ...],
    development_transition_median: float,
    development_entropy_median: float,
) -> dict[str, pd.DataFrame]:
    pooled_mask = pd.Series(True, index=predictions.index)
    occurrence = binary_group_metrics(
        predictions,
        models=("U0", "U1"),
        target="unregistered_event",
        group_type="population",
        groups={"POOLED_OPENING": pooled_mask},
        thresholds=thresholds,
    )
    diagnostic = binary_group_metrics(
        predictions,
        models=("R0", "R1"),
        target="registered_completion",
        group_type="population",
        groups={"REGISTERED_VS_NO_COMPLETION": pooled_mask},
        thresholds=None,
    )
    checkpoint_parts = []
    monthly_parts = []
    split_parts = []
    for models, target, stage, model_thresholds in (
        (("U0", "U1"), "unregistered_event", "U", thresholds),
        (("R0", "R1"), "registered_completion", "R", None),
    ):
        checkpoint = binary_group_metrics(
            predictions,
            models=models,
            target=target,
            group_type="checkpoint",
            groups={str(value): predictions["decision_ordinal"].eq(value) for value in (6, 12)},
            thresholds=model_thresholds,
        )
        checkpoint.insert(0, "stage", stage)
        checkpoint_parts.append(checkpoint)
        monthly = binary_group_metrics(
            predictions,
            models=models,
            target=target,
            group_type="assessment_month",
            groups={
                str(month): predictions["year_month"].eq(month)
                for month in sorted(predictions["year_month"].astype(str).unique())
            },
            thresholds=model_thresholds,
        )
        monthly.insert(0, "stage", stage)
        monthly_parts.append(monthly)
        split_groups = {
            "transition_probability|LOW": predictions["transition_probability"].le(
                development_transition_median
            ),
            "transition_probability|HIGH": predictions["transition_probability"].gt(
                development_transition_median
            ),
            "posterior_entropy|LOW": predictions["posterior_entropy"].le(
                development_entropy_median
            ),
            "posterior_entropy|HIGH": predictions["posterior_entropy"].gt(
                development_entropy_median
            ),
            "prior_registered_completion|NONE": predictions[
                "registered_completion_count_before_decision"
            ].eq(0),
            "prior_registered_completion|PREVIOUS": predictions[
                "registered_completion_count_before_decision"
            ].gt(0),
        }
        split = binary_group_metrics(
            predictions,
            models=models,
            target=target,
            group_type="frozen_split",
            groups=split_groups,
            thresholds=model_thresholds,
        )
        split.insert(0, "stage", stage)
        split_parts.append(split)
    output = {
        "unregistered_occurrence_metrics": occurrence,
        "registered_completion_diagnostic": diagnostic,
        "checkpoint_metrics": pd.concat(checkpoint_parts, ignore_index=True),
        "monthly_metrics": pd.concat(monthly_parts, ignore_index=True),
        "transition_split_metrics": pd.concat(split_parts, ignore_index=True),
    }
    if family_class_order:
        family_parts = [
            family_group_metrics(
                predictions,
                class_order=family_class_order,
                group_type="population",
                groups={"POOLED_UNREGISTERED": pooled_mask},
            ),
            family_group_metrics(
                predictions,
                class_order=family_class_order,
                group_type="checkpoint",
                groups={str(value): predictions["decision_ordinal"].eq(value) for value in (6, 12)},
            ),
            family_group_metrics(
                predictions,
                class_order=family_class_order,
                group_type="assessment_month",
                groups={
                    str(month): predictions["year_month"].eq(month)
                    for month in sorted(predictions["year_month"].astype(str).unique())
                },
            ),
            family_group_metrics(
                predictions,
                class_order=family_class_order,
                group_type="realised_family",
                groups={
                    family: predictions["hidden_family_class"].eq(family)
                    for family in family_class_order
                },
            ),
            family_group_metrics(
                predictions,
                class_order=family_class_order,
                group_type="transition_probability_split",
                groups={
                    "LOW": predictions["transition_probability"].le(development_transition_median),
                    "HIGH": predictions["transition_probability"].gt(development_transition_median),
                },
            ),
        ]
        output["hidden_family_metrics"] = pd.concat(family_parts, ignore_index=True)
    else:
        output["hidden_family_metrics"] = pd.DataFrame(
            columns=["group_type", "group_value", "model", "stage_b_status"]
        )
    return output


def structural_outcome_census(opening: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in (2024, 2025):
        for checkpoint in (6, 12):
            checkpoint_rows = opening.loc[
                opening["year"].eq(year) & opening["decision_ordinal"].eq(checkpoint)
            ]
            for outcome in RAW_OUTCOME_CLASSES:
                group = checkpoint_rows.loc[checkpoint_rows["raw_structural_outcome"].eq(outcome)]
                rows.append(
                    {
                        "year": year,
                        "period": "development" if year == 2024 else "assessment",
                        "raw_outcome": outcome,
                        "decision_ordinal": checkpoint,
                        "rows": len(group),
                        "sessions": int(group["session"].nunique()),
                        "stocks": int(group["symbol"].nunique()),
                        "excluded_from_primary": outcome
                        in {"TIED_REGISTERED_COMPLETION", "SOURCE_UNAVAILABLE"},
                    }
                )
    return pd.DataFrame(rows)


def structural_outcome_counts(opening: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        period: {
            outcome: int(group["raw_structural_outcome"].eq(outcome).sum())
            for outcome in RAW_OUTCOME_CLASSES
        }
        for period, group in (
            ("development", opening.loc[opening["year"].eq(2024)]),
            ("assessment", opening.loc[opening["year"].eq(2025)]),
        )
    }


def path_census(ledger: pd.DataFrame, selected_families: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (year, family), group in ledger.groupby(["year", "family_id"], sort=True):
        first = group.iloc[0]
        stock_counts = group.groupby("symbol", sort=True).size()
        rows.append(
            {
                "year": int(year),
                "period": "development" if int(year) == 2024 else "assessment",
                "family_id": str(family),
                "hidden_family_class": pool_hidden_family(str(family), selected_families),
                "selected_on_development": str(family) in selected_families,
                "canonical_path": json.dumps(first["canonical_path"]),
                "motif_type": str(first["motif_type"]),
                "repeat_depth": int(first["repeat_depth"]),
                "transition_length": int(first["transition_length"]),
                "v2_compatible": bool(first["v2_compatible"]),
                "outcomes": len(group),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
                "maximum_stock_share": float(stock_counts.max() / len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["year", "outcomes", "family_id"], ascending=[True, False, True], kind="mergesort"
    )


def hidden_family_support_table(
    ledger: pd.DataFrame,
    development_census: pd.DataFrame,
    selected_families: tuple[str, ...],
) -> pd.DataFrame:
    census_lookup = development_census.set_index("family_id")
    rows: list[dict[str, Any]] = []
    for final_class in (*selected_families, OTHER_FAMILY):
        for year in (2024, 2025):
            group = ledger.loc[ledger["year"].eq(year)].copy()
            group["hidden_family_class"] = group["family_id"].map(
                lambda value: pool_hidden_family(str(value), selected_families)
            )
            subset = group.loc[group["hidden_family_class"].eq(final_class)]
            stock_counts = subset.groupby("symbol", sort=True).size()
            rows.append(
                {
                    "period": "development" if year == 2024 else "assessment",
                    "year": year,
                    "hidden_family_class": final_class,
                    "selected_family": final_class in selected_families,
                    "development_family_eligible": (
                        bool(census_lookup.loc[final_class, "eligible"])
                        if final_class in census_lookup.index
                        else False
                    ),
                    "outcomes": len(subset),
                    "sessions": int(subset["session"].nunique()),
                    "stocks": int(subset["symbol"].nunique()),
                    "months": int(subset["year_month"].nunique()),
                    "maximum_stock_share": (
                        float(stock_counts.max() / len(subset)) if len(subset) else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_lookup(table: pd.DataFrame, model: str) -> pd.Series:
    rows = table.loc[table["model"].eq(model)]
    if len(rows) != 1:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", f"metric lookup for {model} is ambiguous"
        )
    return rows.iloc[0]


def binary_increment_from_metric_rows(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    result = {
        "log_loss_improvement": float(baseline["log_loss"]) - float(candidate["log_loss"]),
        "brier_improvement": float(baseline["brier_score"]) - float(candidate["brier_score"]),
        "auc_improvement": float(candidate["auc"]) - float(baseline["auc"]),
    }
    if (
        "top_decile_precision" in baseline
        and "top_decile_precision" in candidate
        and not pd.isna(baseline["top_decile_precision"])
        and not pd.isna(candidate["top_decile_precision"])
    ):
        result["top_decile_precision_improvement"] = float(
            candidate["top_decile_precision"]
        ) - float(baseline["top_decile_precision"])
    return result


def family_increment_from_metric_rows(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "multiclass_log_loss_improvement": float(baseline["multiclass_log_loss"])
        - float(candidate["multiclass_log_loss"]),
        "multiclass_brier_improvement": float(baseline["multiclass_brier"])
        - float(candidate["multiclass_brier"]),
        "top_two_accuracy_improvement": float(candidate["top_two_accuracy"])
        - float(baseline["top_two_accuracy"]),
    }


def binary_increment(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    target: str,
    thresholds: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    base = binary_metric_row(
        frame,
        binary_probabilities(frame, baseline),
        model=baseline,
        target=target,
        thresholds=None if thresholds is None else thresholds[baseline],
    )
    added = binary_metric_row(
        frame,
        binary_probabilities(frame, candidate),
        model=candidate,
        target=target,
        thresholds=None if thresholds is None else thresholds[candidate],
    )
    return binary_increment_from_metric_rows(base, added)


def family_increment(frame: pd.DataFrame, *, class_order: tuple[str, ...]) -> dict[str, float]:
    base = multiclass_metric_row(
        frame,
        family_probabilities(frame, "F0", class_order),
        model="F0",
        target="hidden_family_class",
        class_order=class_order,
    )
    added = multiclass_metric_row(
        frame,
        family_probabilities(frame, "F1", class_order),
        model="F1",
        target="hidden_family_class",
        class_order=class_order,
    )
    return family_increment_from_metric_rows(base, added)


def trajectory_group_attribution(
    assessment: pd.DataFrame,
    model: FittedBinary,
    *,
    original_probabilities: np.ndarray,
) -> pd.DataFrame:
    labels = assessment["unregistered_event"].to_numpy(dtype=int)
    weights = assessment["row_weight"].to_numpy(dtype=float)
    original = {
        "log_loss": binary_log_loss(labels, original_probabilities, weights),
        "brier": binary_brier(labels, original_probabilities, weights),
        "auc": float(roc_auc_score(labels, original_probabilities, sample_weight=weights)),
    }
    coefficients = model.estimator.coef_[0]
    feature_index = {feature: index for index, feature in enumerate(model.features)}
    rows: list[dict[str, Any]] = []
    for group_index, (group_name, features) in enumerate(TRAJECTORY_GROUPS.items()):
        group_coefficients = np.asarray([coefficients[feature_index[value]] for value in features])
        rows.append(
            {
                "record_type": "coefficient_summary",
                "trajectory_group": group_name,
                "permutation_draw": np.nan,
                "sum_absolute_standardised_coefficients": float(np.abs(group_coefficients).sum()),
                "signed_standardised_coefficient_sum": float(group_coefficients.sum()),
                "log_loss_deterioration": np.nan,
                "brier_deterioration": np.nan,
                "auc_deterioration": np.nan,
            }
        )
        draw_values: list[dict[str, float]] = []
        for draw in range(ATTRIBUTION_DRAWS):
            permuted = permute_group_within_slates(
                assessment,
                features,
                seed=ATTRIBUTION_SEED + group_index * 100 + draw,
            )
            probabilities = model.predict(permuted)
            values = {
                "log_loss_deterioration": binary_log_loss(labels, probabilities, weights)
                - original["log_loss"],
                "brier_deterioration": binary_brier(labels, probabilities, weights)
                - original["brier"],
                "auc_deterioration": original["auc"]
                - float(roc_auc_score(labels, probabilities, sample_weight=weights)),
            }
            draw_values.append(values)
            rows.append(
                {
                    "record_type": "permutation_draw",
                    "trajectory_group": group_name,
                    "permutation_draw": draw,
                    "sum_absolute_standardised_coefficients": np.nan,
                    "signed_standardised_coefficient_sum": np.nan,
                    **values,
                }
            )
        rows.append(
            {
                "record_type": "permutation_summary",
                "trajectory_group": group_name,
                "permutation_draw": np.nan,
                "sum_absolute_standardised_coefficients": float(np.abs(group_coefficients).sum()),
                "signed_standardised_coefficient_sum": float(group_coefficients.sum()),
                **{
                    key: float(np.mean([value[key] for value in draw_values]))
                    for key in (
                        "log_loss_deterioration",
                        "brier_deterioration",
                        "auc_deterioration",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metrics(
    predictions: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    family_class_order: tuple[str, ...],
    development_transition_median: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    rows: list[dict[str, Any]] = []
    draws = session_block_bootstrap_indices(predictions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    for draw, indices in enumerate(draws):
        sampled = predictions.iloc[indices].reset_index(drop=True)
        scopes = {
            "pooled": pd.Series(True, index=sampled.index),
            "ordinal_6": sampled["decision_ordinal"].eq(6),
            "ordinal_12": sampled["decision_ordinal"].eq(12),
            "low_transition": sampled["transition_probability"].le(development_transition_median),
        }
        for population, mask in scopes.items():
            subset = sampled.loc[mask & sampled["unregistered_event"].notna()]
            for metric, value in binary_increment(
                subset,
                baseline="U0",
                candidate="U1",
                target="unregistered_event",
                thresholds=thresholds,
            ).items():
                rows.append(
                    {
                        "record_type": "draw",
                        "stage": "A_primary",
                        "population": population,
                        "comparison": "U1_minus_U0",
                        "draw": draw,
                        "metric": metric,
                        "value": value,
                        "interval_level": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                    }
                )
            diagnostic = sampled.loc[mask & sampled["registered_completion"].notna()]
            for metric, value in binary_increment(
                diagnostic,
                baseline="R0",
                candidate="R1",
                target="registered_completion",
                thresholds=None,
            ).items():
                if metric not in {"log_loss_improvement", "brier_improvement"}:
                    continue
                rows.append(
                    {
                        "record_type": "draw",
                        "stage": "A_diagnostic",
                        "population": population,
                        "comparison": "R1_minus_R0",
                        "draw": draw,
                        "metric": metric,
                        "value": value,
                        "interval_level": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                    }
                )
        if family_class_order:
            family = sampled.loc[sampled["hidden_family_class"].notna()]
            for metric, value in family_increment(family, class_order=family_class_order).items():
                rows.append(
                    {
                        "record_type": "draw",
                        "stage": "B",
                        "population": "pooled",
                        "comparison": "F1_minus_F0",
                        "draw": draw,
                        "metric": metric,
                        "value": value,
                        "interval_level": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    interval_rows: list[dict[str, Any]] = []
    keys = ["stage", "population", "comparison", "metric"]
    for values, group in frame.groupby(keys, sort=True):
        stage, population, comparison, metric = (str(value) for value in values)
        samples = group["value"].to_numpy(dtype=float)
        summary_key = f"{stage}|{population}|{comparison}"
        summary.setdefault(summary_key, {})[metric] = {}
        for level in (0.80, 0.90, 0.95):
            alpha = (1.0 - level) / 2.0
            lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
            summary[summary_key][metric][f"{level:.2f}_lower"] = float(lower)
            summary[summary_key][metric][f"{level:.2f}_upper"] = float(upper)
            interval_rows.append(
                {
                    "record_type": "interval",
                    "stage": stage,
                    "population": population,
                    "comparison": comparison,
                    "draw": np.nan,
                    "metric": metric,
                    "value": np.nan,
                    "interval_level": level,
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
    return pd.concat([frame, pd.DataFrame(interval_rows)], ignore_index=True), summary


def trajectory_null(
    panel: pd.DataFrame,
    *,
    primary_models: Mapping[str, PrimaryModel],
    thresholds: Mapping[str, Mapping[str, float]],
    t1_features: tuple[str, ...],
    family_class_order: tuple[str, ...],
    real_increments: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    scoring = panel.loc[panel["unregistered_event"].notna()].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    serialised_models: list[dict[str, Any]] = []
    draw_values: dict[tuple[str, str], list[float]] = {}
    for draw in range(NULL_DRAWS):
        permuted = permute_trajectory_bundle_within_slates(
            scoring,
            TRAJECTORY_FEATURES,
            seed=NULL_SEED + draw,
        )
        development = permuted.loc[permuted["year"].eq(2024)].copy()
        assessment = permuted.loc[permuted["year"].eq(2025)].copy()
        u1 = fit_binary("U1", development, t1_features, target="unregistered_event")
        diagnostic_dev = development.loc[development["registered_completion"].notna()]
        diagnostic_assessment = assessment.loc[assessment["registered_completion"].notna()]
        r1 = fit_binary("R1", diagnostic_dev, t1_features, target="registered_completion")
        null_models: dict[str, PrimaryModel] = {"U1": u1, "R1": r1}
        temporary = assessment.loc[
            :,
            [
                *KEYS,
                "year",
                "year_month",
                "slate_id",
                "row_weight",
                "unregistered_event",
                "registered_completion",
                "hidden_family_class",
            ],
        ].copy()
        temporary["U0_probability"] = cast(FittedBinary, primary_models["U0"]).predict(assessment)
        temporary["U1_probability"] = u1.predict(assessment)
        diagnostic_location = temporary.index.isin(diagnostic_assessment.index)
        temporary["R0_probability"] = np.nan
        temporary["R1_probability"] = np.nan
        temporary.loc[diagnostic_location, "R0_probability"] = cast(
            FittedBinary, primary_models["R0"]
        ).predict(diagnostic_assessment)
        temporary.loc[diagnostic_location, "R1_probability"] = r1.predict(diagnostic_assessment)
        comparisons = {
            "U1_minus_U0": binary_increment(
                temporary,
                baseline="U0",
                candidate="U1",
                target="unregistered_event",
                thresholds=thresholds,
            ),
            "R1_minus_R0": binary_increment(
                temporary.loc[temporary["registered_completion"].notna()],
                baseline="R0",
                candidate="R1",
                target="registered_completion",
                thresholds=None,
            ),
        }
        if family_class_order:
            family_dev = development.loc[development["hidden_family_class"].notna()]
            family_assessment = assessment.loc[assessment["hidden_family_class"].notna()]
            f1 = fit_multiclass(
                "F1",
                family_dev,
                t1_features,
                target="hidden_family_class",
                class_order=family_class_order,
            )
            null_models["F1"] = f1
            family_predictions = family_assessment.loc[
                :, [*KEYS, "row_weight", "hidden_family_class"]
            ].copy()
            for name, probabilities in (
                (
                    "F0",
                    cast(FittedMulticlass, primary_models["F0"]).predict(family_assessment),
                ),
                ("F1", f1.predict(family_assessment)),
            ):
                for index, label in enumerate(family_class_order):
                    family_predictions[f"{name}_probability_{label}"] = probabilities[:, index]
            comparisons["F1_minus_F0"] = family_increment(
                family_predictions, class_order=family_class_order
            )
        serialised_models.append(
            {
                "draw": draw,
                "models": {name: model.serialize() for name, model in null_models.items()},
            }
        )
        for comparison, metrics in comparisons.items():
            allowed = {
                "U1_minus_U0": {"log_loss_improvement", "brier_improvement"},
                "R1_minus_R0": {"log_loss_improvement", "brier_improvement"},
                "F1_minus_F0": {
                    "multiclass_log_loss_improvement",
                    "multiclass_brier_improvement",
                },
            }[comparison]
            for metric, value in metrics.items():
                if metric not in allowed:
                    continue
                rows.append(
                    {
                        "record_type": "draw",
                        "draw": draw,
                        "comparison": comparison,
                        "metric": metric,
                        "value": value,
                        "real_increment": np.nan,
                        "null_draws_exceeded": np.nan,
                    }
                )
                draw_values.setdefault((comparison, metric), []).append(value)
    summary: dict[str, dict[str, Any]] = {}
    for comparison, metrics in real_increments.items():
        for metric, real in metrics.items():
            key = (comparison, metric)
            if key not in draw_values:
                continue
            count = int((float(real) > np.asarray(draw_values[key], dtype=float)).sum())
            summary.setdefault(comparison, {})[metric] = {
                "real_increment": float(real),
                "null_draws_exceeded": count,
                "exceeds_at_least_four_of_five": count >= 4,
            }
            rows.append(
                {
                    "record_type": "comparison",
                    "draw": np.nan,
                    "comparison": comparison,
                    "metric": metric,
                    "value": np.nan,
                    "real_increment": float(real),
                    "null_draws_exceeded": count,
                }
            )
    return pd.DataFrame(rows), summary, serialised_models


def binary_table_increment(
    table: pd.DataFrame, *, baseline: str, candidate: str
) -> dict[str, float]:
    base = metric_lookup(table, baseline)
    added = metric_lookup(table, candidate)
    return binary_increment_from_metric_rows(base.to_dict(), added.to_dict())


def family_table_increment(table: pd.DataFrame) -> dict[str, float]:
    base = metric_lookup(table, "F0")
    added = metric_lookup(table, "F1")
    return family_increment_from_metric_rows(base.to_dict(), added.to_dict())


def derive_decision(
    *,
    metrics: Mapping[str, pd.DataFrame],
    bootstrap_summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    null_summary: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Any],
    family_class_order: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Mapping[str, float]]]:
    occurrence = metrics["unregistered_occurrence_metrics"]
    real_u = binary_table_increment(occurrence, baseline="U0", candidate="U1")
    diagnostic = metrics["registered_completion_diagnostic"]
    real_r = binary_table_increment(diagnostic, baseline="R0", candidate="R1")
    monthly = metrics["monthly_metrics"]
    checkpoints = metrics["checkpoint_metrics"]
    u_months = monthly.loc[monthly["stage"].eq("U")]
    positive_u_months = 0
    for month in sorted(u_months["group_value"].astype(str).unique()):
        positive_u_months += int(
            binary_table_increment(
                u_months.loc[u_months["group_value"].astype(str).eq(month)],
                baseline="U0",
                candidate="U1",
            )["log_loss_improvement"]
            > 0.0
        )
    u_checkpoint_values = {
        checkpoint: binary_table_increment(
            checkpoints.loc[
                checkpoints["stage"].eq("U")
                & checkpoints["group_value"].astype(str).eq(str(checkpoint))
            ],
            baseline="U0",
            candidate="U1",
        )["log_loss_improvement"]
        for checkpoint in (6, 12)
    }
    u_bootstrap = bootstrap_summary["A_primary|pooled|U1_minus_U0"]
    u_null = null_summary["U1_minus_U0"]
    stage_a_conditions = {
        "log_loss_improves": real_u["log_loss_improvement"] > 0.0,
        "brier_improves": real_u["brier_improvement"] > 0.0,
        "auc_not_reduced": real_u["auc_improvement"] >= 0.0,
        "bootstrap_90_log_loss_lower_non_negative": u_bootstrap["log_loss_improvement"][
            "0.90_lower"
        ]
        >= 0.0,
        "bootstrap_90_brier_lower_non_negative": u_bootstrap["brier_improvement"]["0.90_lower"]
        >= 0.0,
        "positive_log_loss_in_six_of_eight_months": positive_u_months >= 6,
        "neither_checkpoint_materially_adverse": min(u_checkpoint_values.values())
        >= CHECKPOINT_MATERIAL_ADVERSITY,
        "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
            int(u_null["log_loss_improvement"]["null_draws_exceeded"]) >= 4
            or int(u_null["brier_improvement"]["null_draws_exceeded"]) >= 4
        ),
        "concentration_gates_pass": bool(support["stage_a"]["passed"]),
    }
    stage_a_passes = all(stage_a_conditions.values())
    real: dict[str, Mapping[str, float]] = {
        "U1_minus_U0": real_u,
        "R1_minus_R0": real_r,
    }
    stage_b_conditions: dict[str, bool] = {
        "support_available": bool(family_class_order),
    }
    positive_f_months = 0
    f_checkpoint_values: dict[int, float] = {}
    if family_class_order:
        family = metrics["hidden_family_metrics"]
        pooled = family.loc[
            family["group_type"].eq("population") & family["group_value"].eq("POOLED_UNREGISTERED")
        ]
        real_f = family_table_increment(pooled)
        real["F1_minus_F0"] = real_f
        family_months = family.loc[family["group_type"].eq("assessment_month")]
        for month in sorted(family_months["group_value"].astype(str).unique()):
            positive_f_months += int(
                family_table_increment(
                    family_months.loc[family_months["group_value"].astype(str).eq(month)]
                )["multiclass_log_loss_improvement"]
                > 0.0
            )
        family_checkpoints = family.loc[family["group_type"].eq("checkpoint")]
        f_checkpoint_values = {
            checkpoint: family_table_increment(
                family_checkpoints.loc[
                    family_checkpoints["group_value"].astype(str).eq(str(checkpoint))
                ]
            )["multiclass_log_loss_improvement"]
            for checkpoint in (6, 12)
        }
        f_bootstrap = bootstrap_summary["B|pooled|F1_minus_F0"]
        f_null = null_summary["F1_minus_F0"]
        stage_b_conditions.update(
            {
                "multiclass_log_loss_improves": real_f["multiclass_log_loss_improvement"] > 0.0,
                "multiclass_brier_improves": real_f["multiclass_brier_improvement"] > 0.0,
                "top_two_not_reduced": real_f["top_two_accuracy_improvement"] >= 0.0,
                "bootstrap_80_log_loss_lower_non_negative": f_bootstrap[
                    "multiclass_log_loss_improvement"
                ]["0.80_lower"]
                >= 0.0,
                "bootstrap_80_brier_lower_non_negative": f_bootstrap[
                    "multiclass_brier_improvement"
                ]["0.80_lower"]
                >= 0.0,
                "positive_log_loss_in_five_months": positive_f_months >= 5,
                "neither_checkpoint_materially_adverse": min(f_checkpoint_values.values())
                >= CHECKPOINT_MATERIAL_ADVERSITY,
                "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
                    int(f_null["multiclass_log_loss_improvement"]["null_draws_exceeded"]) >= 4
                    or int(f_null["multiclass_brier_improvement"]["null_draws_exceeded"]) >= 4
                ),
                "concentration_gates_pass": bool(support["stage_b"]["passed"]),
            }
        )
    stage_b_passes = bool(family_class_order) and all(stage_b_conditions.values())
    point_estimate_improves = bool(
        real_u["log_loss_improvement"] > 0.0
        or real_u["brier_improvement"] > 0.0
        or (
            "F1_minus_F0" in real
            and (
                real["F1_minus_F0"]["multiclass_log_loss_improvement"] > 0.0
                or real["F1_minus_F0"]["multiclass_brier_improvement"] > 0.0
            )
        )
    )
    decision = decide_screen(
        stage_a_passes=stage_a_passes,
        stage_b_passes=stage_b_passes,
        point_estimate_improves=point_estimate_improves,
    )
    if not family_class_order:
        stage_b_status = "hidden_family_support_insufficient"
    elif stage_b_passes:
        stage_b_status = "hidden_family_prediction_supported"
    else:
        stage_b_status = "hidden_family_prediction_not_supported"
    artifact = {
        **SAFETY_FLAGS,
        "decision": decision,
        "primary_decision": decision,
        "stage_b_status": stage_b_status,
        "stage_a_passes": stage_a_passes,
        "stage_b_passes": stage_b_passes,
        "stage_a_conditions": stage_a_conditions,
        "stage_b_conditions": stage_b_conditions,
        "positive_stage_a_log_loss_months": positive_u_months,
        "positive_stage_b_log_loss_months": positive_f_months,
        "stage_a_checkpoint_log_loss_improvements": {
            str(key): value for key, value in u_checkpoint_values.items()
        },
        "stage_b_checkpoint_log_loss_improvements": {
            str(key): value for key, value in f_checkpoint_values.items()
        },
        "real_increments": real,
        "point_estimate_improves_somewhere": point_estimate_improves,
        "feasibility_only": True,
        "validation_or_promotion": False,
    }
    return artifact, real


def maximum_model_parameter_difference(
    original: Mapping[str, PrimaryModel], repeated: Mapping[str, PrimaryModel]
) -> tuple[float, float, bool]:
    preprocessing = 0.0
    coefficient = 0.0
    class_equal = set(original) == set(repeated)
    for name in original:
        left = original[name]
        right = repeated[name]
        preprocessing = max(
            preprocessing,
            float(np.max(np.abs(left.scaler.mean_ - right.scaler.mean_))),
            float(np.max(np.abs(left.scaler.scale_ - right.scaler.scale_))),
        )
        coefficient = max(
            coefficient,
            float(np.max(np.abs(left.estimator.coef_ - right.estimator.coef_))),
            float(np.max(np.abs(left.estimator.intercept_ - right.estimator.intercept_))),
        )
        class_equal &= np.array_equal(left.estimator.classes_, right.estimator.classes_)
        if isinstance(left, FittedMulticlass) and isinstance(right, FittedMulticlass):
            class_equal &= left.class_order == right.class_order
        elif isinstance(left, FittedMulticlass) != isinstance(right, FittedMulticlass):
            class_equal = False
    return preprocessing, coefficient, class_equal


def determinism_check(
    panel: pd.DataFrame,
    *,
    original_models: Mapping[str, PrimaryModel],
    original_predictions: pd.DataFrame,
    original_metrics: Mapping[str, pd.DataFrame],
    original_thresholds: Mapping[str, Mapping[str, float]],
    t0_features: tuple[str, ...],
    t1_features: tuple[str, ...],
    selected_families: tuple[str, ...],
    family_class_order: tuple[str, ...],
    path_ledger: pd.DataFrame,
    development_transition_median: float,
    development_entropy_median: float,
    bootstrap_summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    null_summary: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Any],
    original_decision: str,
) -> dict[str, Any]:
    repeated_models, repeated_predictions, repeated_thresholds, repeated_class_order = (
        fit_primary_models(
            panel,
            t0_features=t0_features,
            t1_features=t1_features,
            selected_families=selected_families,
            stage_b_supported=bool(family_class_order),
        )
    )
    probability_columns = sorted(
        column
        for column in original_predictions
        if "_probability" in column and column in repeated_predictions
    )
    probability_difference = max(
        float(
            np.nanmax(
                np.abs(
                    original_predictions[column].to_numpy(dtype=float)
                    - repeated_predictions[column].to_numpy(dtype=float)
                )
            )
        )
        for column in probability_columns
    )
    preprocessing_difference, coefficient_difference, class_order_equal = (
        maximum_model_parameter_difference(original_models, repeated_models)
    )
    threshold_difference = max(
        abs(float(original_thresholds[model][key]) - float(repeated_thresholds[model][key]))
        for model in original_thresholds
        for key in original_thresholds[model]
    )
    repeated_metrics = all_stage_metrics(
        repeated_predictions,
        thresholds=repeated_thresholds,
        family_class_order=repeated_class_order,
        development_transition_median=development_transition_median,
        development_entropy_median=development_entropy_median,
    )
    metric_difference = 0.0
    metric_columns = {
        "unregistered_occurrence_metrics": (
            "brier_score",
            "log_loss",
            "auc",
            "average_precision",
            "mean_probability_realised_class",
        ),
        "registered_completion_diagnostic": (
            "brier_score",
            "log_loss",
            "auc",
            "average_precision",
            "mean_probability_realised_class",
        ),
        "hidden_family_metrics": (
            "multiclass_log_loss",
            "multiclass_brier",
            "top_one_accuracy",
            "top_two_accuracy",
            "mean_probability_realised_family",
        ),
    }
    for artifact, columns in metric_columns.items():
        left = original_metrics[artifact]
        right = repeated_metrics[artifact]
        if left.empty and right.empty:
            continue
        pooled_left = left.loc[left["group_type"].eq("population")].reset_index(drop=True)
        pooled_right = right.loc[right["group_type"].eq("population")].reset_index(drop=True)
        for column in columns:
            if column in pooled_left and column in pooled_right:
                metric_difference = max(
                    metric_difference,
                    float(
                        np.nanmax(
                            np.abs(
                                pooled_left[column].to_numpy(dtype=float)
                                - pooled_right[column].to_numpy(dtype=float)
                            )
                        )
                    ),
                )
    development_census = hidden_family_census(path_ledger.loc[path_ledger["year"].eq(2024)])
    repeated_selected = select_hidden_families(development_census, maximum=4)
    mapping_equal = repeated_selected == selected_families
    repeated_decision, _ = derive_decision(
        metrics=repeated_metrics,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        support=support,
        family_class_order=repeated_class_order,
    )
    decision_equal = repeated_decision["decision"] == original_decision
    passed = bool(
        probability_difference <= 1e-12
        and preprocessing_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and threshold_difference <= 1e-12
        and metric_difference <= 1e-12
        and class_order_equal
        and repeated_class_order == family_class_order
        and mapping_equal
        and decision_equal
    )
    if not passed:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "fast determinism check failed"
        )
    return {
        **SAFETY_FLAGS,
        "passed": True,
        "models_refit": sorted(repeated_models),
        "bootstrap_repeated": False,
        "null_repeated": False,
        "class_order_equal": class_order_equal and repeated_class_order == family_class_order,
        "hidden_family_mapping_equal": mapping_equal,
        "decision_equal": decision_equal,
        "decision": original_decision,
        "regenerated_decision": repeated_decision["decision"],
        "maximum_preprocessing_parameter_difference": preprocessing_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_threshold_difference": threshold_difference,
        "maximum_probability_difference": probability_difference,
        "maximum_pooled_metric_difference": metric_difference,
        "probability_tolerance": 1e-12,
    }


def load_script(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def calibration_curve_points(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        if mask.any():
            predicted.append(weighted_mean(probabilities[mask], weights[mask]))
            observed.append(weighted_mean(targets[mask], weights[mask]))
    return np.asarray(predicted), np.asarray(observed)


def plot_summary(predictions: pd.DataFrame, family_support: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = predictions["unregistered_event"].to_numpy(dtype=int)
    weights = predictions["row_weight"].to_numpy(dtype=float)
    for model, colour in (("U0", "#52677a"), ("U1", "#ca5b3f")):
        predicted, observed = calibration_curve_points(
            labels, binary_probabilities(predictions, model), weights
        )
        axes[0].plot(predicted, observed, marker="o", label=model, color=colour)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    axes[0].set(
        xlabel="Predicted unregistered-event probability",
        ylabel="Observed weighted rate",
        title="Opening assessment calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(frameon=False)
    assessment = family_support.loc[family_support["period"].eq("assessment")]
    if assessment.empty:
        axes[1].text(0.5, 0.5, "Stage B unsupported", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        labels_text = [
            str(value).replace("unregistered_", "")[:24]
            for value in assessment["hidden_family_class"]
        ]
        axes[1].barh(labels_text, assessment["outcomes"], color="#5d8b73")
        axes[1].set(xlabel="Assessment outcomes", title="Frozen hidden-family support")
        axes[1].invert_yaxis()
    figure.suptitle("Opening trajectories and development-frozen unregistered families")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.9f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_report(
    *,
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    metrics: Mapping[str, pd.DataFrame],
    family_support: pd.DataFrame,
    attribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    determinism: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    occurrence = metrics["unregistered_occurrence_metrics"]
    diagnostic = metrics["registered_completion_diagnostic"]
    family_pooled = metrics["hidden_family_metrics"]
    family_pooled = family_pooled.loc[
        family_pooled.get("group_type", pd.Series(dtype=str)).eq("population")
    ]
    attribution_summary = attribution.loc[attribution["record_type"].eq("permutation_summary")]
    bootstrap_intervals = bootstrap.loc[
        bootstrap["record_type"].eq("interval") & bootstrap["population"].eq("pooled")
    ]
    null_summary = null.loc[null["record_type"].eq("comparison")]
    stage_a = support["stage_a"]
    stage_b = support["stage_b"]
    occurrence_table = markdown_table(
        occurrence,
        (
            "model",
            "log_loss",
            "brier_score",
            "auc",
            "average_precision",
            "top_decile_precision",
            "top_decile_lift",
            "top_quintile_precision",
            "top_quintile_lift",
            "mean_probability_realised_class",
        ),
    )
    diagnostic_table = markdown_table(
        diagnostic,
        (
            "model",
            "log_loss",
            "brier_score",
            "auc",
            "average_precision",
            "mean_probability_realised_class",
        ),
    )
    support_table = markdown_table(
        family_support,
        (
            "period",
            "hidden_family_class",
            "outcomes",
            "sessions",
            "stocks",
            "months",
            "maximum_stock_share",
        ),
    )
    family_columns = tuple(
        column
        for column in (
            "model",
            "multiclass_log_loss",
            "multiclass_brier",
            "top_one_accuracy",
            "top_two_accuracy",
            "mean_probability_realised_family",
            "prediction_entropy",
            "effective_candidate_count",
        )
        if column in family_pooled.columns
    )
    family_table = markdown_table(family_pooled, family_columns)
    attribution_table = markdown_table(
        attribution_summary,
        (
            "trajectory_group",
            "sum_absolute_standardised_coefficients",
            "signed_standardised_coefficient_sum",
            "log_loss_deterioration",
            "brier_deterioration",
            "auc_deterioration",
        ),
    )
    bootstrap_table = markdown_table(
        bootstrap_intervals,
        ("stage", "comparison", "metric", "interval_level", "lower", "upper"),
    )
    null_table = markdown_table(
        null_summary,
        ("comparison", "metric", "real_increment", "null_draws_exceeded"),
    )
    return f"""# Opening Behavioural Trajectory → Unregistered Loop Families Quick Screen V0

Decision: `{decision["decision"]}`.

Stage B status: `{decision["stage_b_status"]}`.

This is retrospective, observable-only structural feasibility evidence. Economic and
directional outcomes remained closed; no trading, execution, broker, or deployment surface was
opened.

## Support

- Opening assessment rows: {stage_a["rows"]}.
- Sessions/stocks/months: {stage_a["sessions"]} / {stage_a["stocks"]} / {stage_a["months"]}.
- Assessment outcomes: `{json.dumps(stage_a["outcome_counts"], sort_keys=True)}`.
- Trajectory retention: {support["trajectory_retention"]:.9f}.
- Stage B assessment family rows: {stage_b["rows"]}.
- Stage B family counts: `{json.dumps(stage_b["family_counts"], sort_keys=True)}`.

## Stage A occurrence metrics

{occurrence_table}

## Registered-completion diagnostic

{diagnostic_table}

## Hidden-family support

{support_table}

## Stage B pooled metrics

{family_table}

## Fixed trajectory-group attribution

{attribution_table}

## Pooled bootstrap intervals

{bootstrap_table}

## Five-draw trajectory null

{null_table}

## Verification

- Determinism check: `{determinism["passed"]}`; maximum probability difference
  `{determinism["maximum_probability_difference"]}`.
- Independent lightweight audit: `{audit["passed"]}`.

The findings are not prospective validation, economic-edge evidence, trading utility, or P&L.
"""


def source_manifest() -> dict[str, Any]:
    sources = [
        PREDECESSOR_PANEL,
        PREDECESSOR_LEDGER,
        PREDECESSOR_MODELS,
        PREDECESSOR_PREDICTIONS,
        PREDECESSOR_ARTIFACTS / "checkpoint_anchor_manifest.json",
        PREDECESSOR_ARTIFACTS / "determinism_check.json",
        PREDECESSOR_ARTIFACTS / "lightweight_audit.json",
        DICTIONARY_PATH,
    ]
    return {
        **SAFETY_FLAGS,
        "predecessor_experiment": (
            "Behavioural-Trajectory Funnel V0.1 — Corrected Anchors and Later Loops"
        ),
        "predecessor_commit": "8191bc7fa9dc689d2ca998e759bebab094bc1ade",
        "sources": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in sources
        ],
        "raw_data_downloaded": False,
        "one_minute_data_opened": False,
        "minimum_timestamp_read": "2024-01-17 14:30:00+00:00",
        "maximum_timestamp_read": "2025-08-22 15:00:00+00:00",
        "protected_rows_materialised": 0,
    }


def model_configuration_artifact(
    *,
    t0_features: tuple[str, ...],
    t1_features: tuple[str, ...],
    models: Mapping[str, PrimaryModel],
    thresholds: Mapping[str, Mapping[str, float]],
    family_class_order: tuple[str, ...],
    transition_median: float,
    entropy_median: float,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "model_count": len(models),
        "maximum_primary_models": 6,
        "models": {
            name: {
                "kind": "multiclass" if isinstance(model, FittedMulticlass) else "binary",
                "features": list(model.features),
                "feature_count": len(model.features),
            }
            for name, model in models.items()
        },
        "T0_features": list(t0_features),
        "T1_features": list(t1_features),
        "binary_configuration": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
            "random_state": MODEL_SEED,
        },
        "multiclass_configuration": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "lbfgs",
            "max_iter": 300,
            "multi_class": "multinomial",
            "class_weight": None,
            "n_jobs": 1,
            "random_state": MODEL_SEED,
        },
        "preprocessing_fit": "2024_only",
        "coefficients_fit": "2024_only",
        "row_weight": "1 / eligible_stocks_in_session_checkpoint",
        "development_prediction_thresholds": thresholds,
        "development_transition_probability_median": transition_median,
        "development_posterior_entropy_median": entropy_median,
        "family_class_order": list(family_class_order),
    }


def execute_screen(output: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    opening, population_manifest, feature_manifest = load_opening_panel()
    t0_features, t1_features = predecessor_feature_surfaces()
    registered_paths, dictionary_manifest = registered_paths_and_manifest()
    path_ledger = build_unregistered_path_ledger(opening, registered_paths)
    development_paths = path_ledger.loc[path_ledger["year"].eq(2024)].copy()
    development_census = hidden_family_census(development_paths)
    selected_families = select_hidden_families(development_census, maximum=4)
    hidden_mapping = {
        **SAFETY_FLAGS,
        "frozen_before_assessment_family_support": True,
        "fit_period": "2024_only",
        "canonicalisation_contract": contract["canonicalisation"],
        "selection_contract": contract["hidden_family_selection"],
        "eligible_development_families": int(development_census["eligible"].sum()),
        "selected_families": list(selected_families),
        "selected_family_count": len(selected_families),
        "other_family": OTHER_FAMILY,
        "stage_b_development_support_available": len(selected_families) >= 2,
    }
    # This is intentionally persisted before assessment family support is calculated.
    write_json(output / "hidden_family_mapping.json", hidden_mapping)
    panel = attach_hidden_families(opening, path_ledger, selected_families)
    support, concentration, stage_a_supported, stage_b_supported = support_and_concentration(
        panel,
        path_ledger=path_ledger,
        selected_families=selected_families,
        development_eligible_family_count=int(development_census["eligible"].sum()),
    )
    if not stage_a_supported:
        raise ScreenBlocker(
            "blocked_stage_a_support_failure", "opening binary support gates did not pass"
        )
    family_support = hidden_family_support_table(path_ledger, development_census, selected_families)
    path_census_frame = path_census(path_ledger, selected_families)
    development = panel.loc[panel["year"].eq(2024) & panel["unregistered_event"].notna()]
    assessment = panel.loc[panel["year"].eq(2025) & panel["unregistered_event"].notna()]
    transition_median = float(development["transition_probability"].median())
    entropy_median = float(development["posterior_entropy"].median())
    models, predictions, thresholds, family_class_order = fit_primary_models(
        panel,
        t0_features=t0_features,
        t1_features=t1_features,
        selected_families=selected_families,
        stage_b_supported=stage_b_supported,
    )
    if len(models) > 6:
        raise ScreenBlocker(
            "blocked_quick_unregistered_family_resource_limit",
            f"primary model count={len(models)}",
        )
    metrics = all_stage_metrics(
        predictions,
        thresholds=thresholds,
        family_class_order=family_class_order,
        development_transition_median=transition_median,
        development_entropy_median=entropy_median,
    )
    attribution = trajectory_group_attribution(
        assessment,
        cast(FittedBinary, models["U1"]),
        original_probabilities=binary_probabilities(predictions, "U1"),
    )
    real_increments: dict[str, Mapping[str, float]] = {
        "U1_minus_U0": binary_table_increment(
            metrics["unregistered_occurrence_metrics"], baseline="U0", candidate="U1"
        ),
        "R1_minus_R0": binary_table_increment(
            metrics["registered_completion_diagnostic"], baseline="R0", candidate="R1"
        ),
    }
    if family_class_order:
        family_pooled = metrics["hidden_family_metrics"].loc[
            metrics["hidden_family_metrics"]["group_type"].eq("population")
        ]
        real_increments["F1_minus_F0"] = family_table_increment(family_pooled)
    bootstrap, bootstrap_summary = bootstrap_metrics(
        predictions,
        thresholds=thresholds,
        family_class_order=family_class_order,
        development_transition_median=transition_median,
    )
    null, null_summary, null_models = trajectory_null(
        panel,
        primary_models=models,
        thresholds=thresholds,
        t1_features=t1_features,
        family_class_order=family_class_order,
        real_increments=real_increments,
    )
    decision, _ = derive_decision(
        metrics=metrics,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        support=support,
        family_class_order=family_class_order,
    )
    determinism = determinism_check(
        panel,
        original_models=models,
        original_predictions=predictions,
        original_metrics=metrics,
        original_thresholds=thresholds,
        t0_features=t0_features,
        t1_features=t1_features,
        selected_families=selected_families,
        family_class_order=family_class_order,
        path_ledger=path_ledger,
        development_transition_median=transition_median,
        development_entropy_median=entropy_median,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        support=support,
        original_decision=str(decision["decision"]),
    )
    raw_census = structural_outcome_census(opening)
    target_counts = structural_outcome_counts(opening)
    binary_manifest = {
        **SAFETY_FLAGS,
        "horizon_completed_five_minute_bars": 6,
        "unregistered_event": {
            "positive": "UNREGISTERED_LOOP",
            "negative": ["REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"],
        },
        "registered_completion_diagnostic": {
            "population": ["REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"],
            "positive": "REGISTERED_COMPLETION",
            "negative": "NO_REGISTERED_COMPLETION",
            "binding": False,
        },
        "excluded": ["TIED_REGISTERED_COMPLETION", "SOURCE_UNAVAILABLE"],
        "counts": target_counts,
    }
    protected_audit = {
        **SAFETY_FLAGS,
        "development_start": "2024-01-01",
        "development_end_inclusive": "2024-12-31",
        "assessment_start": "2025-01-01",
        "assessment_end_inclusive": "2025-08-22",
        "protected_start": "2025-08-23",
        "minimum_session": str(opening["session"].min()),
        "maximum_session": str(opening["session"].max()),
        "protected_rows_materialised": 0,
        "passed": True,
    }
    configuration = model_configuration_artifact(
        t0_features=t0_features,
        t1_features=t1_features,
        models=models,
        thresholds=thresholds,
        family_class_order=family_class_order,
        transition_median=transition_median,
        entropy_median=entropy_median,
    )
    coefficients = {
        **SAFETY_FLAGS,
        "primary_models": {name: model.serialize() for name, model in models.items()},
        "null_models": null_models,
    }
    decision.update(
        {
            "support": support,
            "selected_hidden_families": list(selected_families),
            "family_class_order": list(family_class_order),
            "development_rows": len(development),
            "assessment_rows": len(assessment),
            "structural_outcome_counts": target_counts,
            "determinism_check_passed": True,
            "lightweight_audit_passed": None,
        }
    )
    write_json(output / "contract.json", contract)
    write_json(
        output / "source_manifest.json", {**source_manifest(), "dictionary": dictionary_manifest}
    )
    write_json(output / "protected_boundary_audit.json", protected_audit)
    write_json(output / "opening_population_reconstruction.json", population_manifest)
    write_json(output / "trajectory_feature_reconstruction.json", feature_manifest)
    write_csv(output / "structural_outcome_census.csv", raw_census)
    write_json(output / "binary_target_manifest.json", binary_manifest)
    write_json(output / "model_configurations.json", configuration)
    write_json(output / "model_coefficients.json", coefficients)
    write_parquet(output / "assessment_predictions.parquet", predictions)
    for artifact, frame in metrics.items():
        write_csv(output / f"{artifact}.csv", frame)
    write_csv(output / "trajectory_group_attribution.csv", attribution)
    write_parquet(output / "unregistered_path_ledger.parquet", path_ledger)
    write_csv(output / "unregistered_path_census.csv", path_census_frame)
    write_csv(output / "hidden_family_support.csv", family_support)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", null)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    write_json(output / "determinism_check.json", determinism)
    plot_summary(predictions, family_support, output / "opening_calibration_and_family_support.png")
    auditor = load_script("_opening_unregistered_family_auditor", AUDITOR_PATH)
    audit = cast(dict[str, Any], auditor.audit(output))
    if not audit.get("passed"):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "independent lightweight audit failed"
        )
    decision["lightweight_audit_passed"] = True
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        support=support,
        metrics=metrics,
        family_support=family_support,
        attribution=attribution,
        bootstrap=bootstrap,
        null=null,
        determinism=determinism,
        audit=audit,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def bootstrap_summary_from_artifact(
    frame: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    intervals = frame.loc[frame["record_type"].eq("interval")]
    for row in intervals.itertuples(index=False):
        key = f"{row.stage}|{row.population}|{row.comparison}"
        metric = str(row.metric)
        level = float(row.interval_level)
        summary.setdefault(key, {}).setdefault(metric, {})[f"{level:.2f}_lower"] = float(row.lower)
        summary[key][metric][f"{level:.2f}_upper"] = float(row.upper)
    return summary


def null_summary_from_artifact(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    comparisons = frame.loc[frame["record_type"].eq("comparison")]
    for row in comparisons.itertuples(index=False):
        count = int(row.null_draws_exceeded)
        summary.setdefault(str(row.comparison), {})[str(row.metric)] = {
            "real_increment": float(row.real_increment),
            "null_draws_exceeded": count,
            "exceeds_at_least_four_of_five": count >= 4,
        }
    return summary


def finalize_existing(output: Path) -> dict[str, Any]:
    """Finish audit/report generation without repeating fits, bootstrap, or null work."""

    opening, population_manifest, feature_manifest = load_opening_panel()
    path_ledger = pd.read_parquet(output / "unregistered_path_ledger.parquet")
    selected_families = tuple(
        str(value)
        for value in read_json(output / "hidden_family_mapping.json")["selected_families"]
    )
    panel = attach_hidden_families(opening, path_ledger, selected_families)
    support, concentration, stage_a_supported, _stage_b_supported = support_and_concentration(
        panel,
        path_ledger=path_ledger,
        selected_families=selected_families,
        development_eligible_family_count=int(
            read_json(output / "hidden_family_mapping.json")["eligible_development_families"]
        ),
    )
    if not stage_a_supported:
        raise ScreenBlocker(
            "blocked_stage_a_support_failure", "opening binary support gates did not pass"
        )
    configuration = read_json(output / "model_configurations.json")
    family_class_order = tuple(str(value) for value in configuration["family_class_order"])
    metrics = {
        name: pd.read_csv(output / f"{name}.csv")
        for name in (
            "unregistered_occurrence_metrics",
            "registered_completion_diagnostic",
            "checkpoint_metrics",
            "monthly_metrics",
            "transition_split_metrics",
            "hidden_family_metrics",
        )
    }
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    null = pd.read_csv(output / "null_metrics.csv")
    decision, _ = derive_decision(
        metrics=metrics,
        bootstrap_summary=bootstrap_summary_from_artifact(bootstrap),
        null_summary=null_summary_from_artifact(null),
        support=support,
        family_class_order=family_class_order,
    )
    development = panel.loc[panel["year"].eq(2024) & panel["unregistered_event"].notna()]
    assessment = panel.loc[panel["year"].eq(2025) & panel["unregistered_event"].notna()]
    target_counts = structural_outcome_counts(opening)
    determinism = read_json(output / "determinism_check.json")
    if not bool(determinism.get("passed")):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "stored determinism check failed"
        )
    decision.update(
        {
            "support": support,
            "selected_hidden_families": list(selected_families),
            "family_class_order": list(family_class_order),
            "development_rows": len(development),
            "assessment_rows": len(assessment),
            "structural_outcome_counts": target_counts,
            "determinism_check_passed": True,
            "lightweight_audit_passed": None,
            "finalized_from_existing_bounded_results": True,
            "primary_models_refit_during_finalization": 0,
            "bootstrap_repeated_during_finalization": False,
            "null_repeated_during_finalization": False,
        }
    )
    write_json(output / "opening_population_reconstruction.json", population_manifest)
    write_json(output / "trajectory_feature_reconstruction.json", feature_manifest)
    write_csv(output / "structural_outcome_census.csv", structural_outcome_census(opening))
    binary_manifest = read_json(output / "binary_target_manifest.json")
    binary_manifest["counts"] = target_counts
    write_json(output / "binary_target_manifest.json", binary_manifest)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    auditor = load_script("_opening_unregistered_family_auditor", AUDITOR_PATH)
    audit = cast(dict[str, Any], auditor.audit(output))
    if not audit.get("passed"):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "independent lightweight audit failed"
        )
    decision["lightweight_audit_passed"] = True
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        support=support,
        metrics=metrics,
        family_support=pd.read_csv(output / "hidden_family_support.csv"),
        attribution=pd.read_csv(output / "trajectory_group_attribution.csv"),
        bootstrap=bootstrap,
        null=null,
        determinism=determinism,
        audit=audit,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "decision.json",
        {
            **SAFETY_FLAGS,
            "decision": blocker.code,
            "primary_decision": blocker.code,
            "stage_b_status": "hidden_family_support_insufficient",
            "blocker_detail": blocker.detail,
            "feasibility_only": True,
            "validation_or_promotion": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="audit and report already-computed bounded artifacts without repeating fits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    try:
        decision = finalize_existing(output) if args.finalize_existing else execute_screen(output)
        print(canonical_json(decision), end="")
        return 0
    except ScreenBlocker as blocker:
        if not args.finalize_existing:
            write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2
    except Exception as error:
        blocker = ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"unexpected fail-closed error: {type(error).__name__}: {error}",
        )
        if not args.finalize_existing:
            write_blocker(output, blocker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
