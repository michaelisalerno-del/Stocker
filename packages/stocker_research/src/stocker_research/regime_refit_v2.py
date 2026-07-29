"""Deterministic duration-only and complete right-censored regime refits V2."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups
from stocker_research.regime_panel_v2 import NATURAL_KEY, canonical_frame_hash
from stocker_research.regime_validity_v2 import (
    CleaningVariant,
    EmissionPreprocessing,
    SemiMarkovParameters,
    apply_cleaning_variant,
    fit_emission_preprocessing,
    reject_outcome_columns,
    semantic_remap_by_activity_direction,
    transform_emissions,
)
from stocker_research.right_censored_duration_v2 import (
    DurationFitConfig,
    RightCensoredDurationFit,
    classify_training_run_endings,
    estimate_right_censored_durations,
)

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False
PART_B_INTERACTION_SCORING_ENABLED = False
SEMANTIC_DICTIONARY_PROMOTION_ENABLED = False

DURATION_ONLY_MODEL_ID = "regime_model_v2_duration_only_repair"
FULL_REFIT_MODEL_ID = "regime_model_v2_full_right_censored_refit"


@dataclass(frozen=True, slots=True)
class RefitConfig:
    """Frozen clustering and duration configuration for one complete refit."""

    state_count: int = 8
    seed: int = 20260710
    nominal_maximum_rows: int = 200_000
    maximum_age: int = 78
    cleaning_variant: CleaningVariant = CleaningVariant.CLEANING_1
    batch_size: int = 4096
    n_init: int = 10
    max_iter: int = 300
    variance_floor: float = 0.05
    transition_pseudocount: float = 0.5
    initial_pseudocount: float = 0.5
    duration_alpha: float = 0.5
    duration_beta: float = 0.5
    minimum_state_at_risk: int = 5
    tail_prior_hazard: float = 0.05
    activity_column: str = "regime_log_activity_12"
    direction_column: str = "signed_efficiency_12"

    def __post_init__(self) -> None:
        if self.state_count <= 1:
            raise ValueError("state_count must exceed one")
        if self.nominal_maximum_rows < self.state_count:
            raise ValueError("nominal training rows must cover every state")
        if self.maximum_age <= 0 or self.maximum_age > 78:
            raise ValueError("maximum_age must be in [1, 78]")
        if self.batch_size <= 0 or self.n_init <= 0 or self.max_iter <= 0:
            raise ValueError("KMeans settings must be positive")
        if self.variance_floor <= 0.0:
            raise ValueError("variance floor must be positive")
        if self.transition_pseudocount <= 0.0 or self.initial_pseudocount <= 0.0:
            raise ValueError("state-count pseudocounts must be positive")


@dataclass(frozen=True, slots=True)
class DurationOnlyRepairResult:
    model_id: str
    parameters: SemiMarkovParameters
    run_ledger: pd.DataFrame
    duration_fit: RightCensoredDurationFit
    parameter_hash: str


@dataclass(frozen=True, slots=True)
class FullRefitResult:
    model_id: str
    preprocessing: EmissionPreprocessing
    scaled: np.ndarray
    training_indices: np.ndarray
    training_row_hash: str
    raw_labels: np.ndarray
    cleaned_labels: np.ndarray
    semantic_labels: np.ndarray
    semantic_mapping: dict[int, int]
    raw_cluster_centers: np.ndarray
    semantic_cluster_centers: np.ndarray
    parameters: SemiMarkovParameters
    run_ledger: pd.DataFrame
    duration_fit: RightCensoredDurationFit
    training_objective: float
    kmeans_iterations: int
    kmeans_steps: int
    kmeans_converged: bool
    parameter_hash: str
    preprocessing_hash: str
    model_hash: str
    effective_configuration: dict[str, Any]


def deterministic_stride_indices(row_count: int, *, nominal_maximum_rows: int) -> np.ndarray:
    """Reproduce the historical floor-step stride policy exactly."""

    if row_count <= 0 or nominal_maximum_rows <= 0:
        raise ValueError("row_count and nominal maximum must be positive")
    step = max(1, row_count // nominal_maximum_rows)
    return np.arange(0, row_count, step, dtype=np.int64)


def _hash_array(digest: Any, name: str, array: np.ndarray) -> None:
    values = np.ascontiguousarray(np.asarray(array))
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(values.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))


def deterministic_parameter_hash(parameters: SemiMarkovParameters) -> str:
    """Hash exact parameter dtypes, shapes, names, and bytes."""

    parameters.validate()
    digest = hashlib.sha256()
    for name, values in sorted(parameters.as_dict().items()):
        _hash_array(digest, name, values)
    return digest.hexdigest()


def deterministic_preprocessing_hash(preprocessing: EmissionPreprocessing) -> str:
    """Hash the fitted feature order and every preprocessing float."""

    preprocessing.validate()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            list(preprocessing.feature_names),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    for name, values in (
        ("medians", preprocessing.medians),
        ("centers", preprocessing.centers),
        ("scales", preprocessing.scales),
    ):
        _hash_array(digest, name, values)
    return digest.hexdigest()


def deterministic_model_hash(
    *,
    parameters: SemiMarkovParameters,
    preprocessing: EmissionPreprocessing,
    semantic_mapping: Mapping[int, int],
    semantic_cluster_centers: np.ndarray,
    configuration: Mapping[str, Any],
) -> str:
    """Bind all fitted state semantics, not only semi-Markov arrays."""

    digest = hashlib.sha256()
    digest.update(deterministic_parameter_hash(parameters).encode("ascii"))
    digest.update(deterministic_preprocessing_hash(preprocessing).encode("ascii"))
    digest.update(
        json.dumps(
            {str(key): int(value) for key, value in sorted(semantic_mapping.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _hash_array(digest, "semantic_cluster_centers", semantic_cluster_centers)
    digest.update(
        json.dumps(
            dict(configuration),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with sorted members and fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _duration_config(
    *,
    maximum_age: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    minimum_state_at_risk: int = 5,
    tail_prior_hazard: float = 0.05,
) -> DurationFitConfig:
    return DurationFitConfig(
        maximum_age=maximum_age,
        alpha=alpha,
        beta=beta,
        minimum_state_at_risk=minimum_state_at_risk,
        tail_prior_hazard=tail_prior_hazard,
    )


def build_duration_only_repair(
    *,
    frozen_parameters: SemiMarkovParameters,
    panel: pd.DataFrame,
    frozen_training_labels: np.ndarray,
    maximum_age: int = 78,
    duration_alpha: float = 0.5,
    duration_beta: float = 0.5,
    minimum_state_at_risk: int = 5,
    tail_prior_hazard: float = 0.05,
) -> DurationOnlyRepairResult:
    """Replace only duration and gap semantics in the frozen parameter family."""

    frozen_parameters.validate()
    labels = np.asarray(frozen_training_labels, dtype=int)
    if labels.shape != (len(panel),):
        raise ValueError("frozen training labels differ from panel rows")
    bars = panel.copy().reset_index(drop=True)
    bars["state"] = labels
    run_ledger = classify_training_run_endings(bars)
    state_count = frozen_parameters.means.shape[0]
    duration_fit = estimate_right_censored_durations(
        run_ledger,
        state_count=state_count,
        config=_duration_config(
            maximum_age=maximum_age,
            alpha=duration_alpha,
            beta=duration_beta,
            minimum_state_at_risk=minimum_state_at_risk,
            tail_prior_hazard=tail_prior_hazard,
        ),
    )
    repaired = SemiMarkovParameters(
        means=frozen_parameters.means.copy(),
        variances=frozen_parameters.variances.copy(),
        duration_hazard=duration_fit.hazard.copy(),
        transitions=frozen_parameters.transitions.copy(),
        initial=frozen_parameters.initial.copy(),
        occupancy=frozen_parameters.occupancy.copy(),
    )
    repaired.validate()
    return DurationOnlyRepairResult(
        model_id=DURATION_ONLY_MODEL_ID,
        parameters=repaired,
        run_ledger=run_ledger,
        duration_fit=duration_fit,
        parameter_hash=deterministic_parameter_hash(repaired),
    )


def _fit_non_duration_parameters(
    *,
    scaled: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[np.ndarray],
    state_count: int,
    variance_floor: float,
    transition_pseudocount: float,
    initial_pseudocount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scaled, dtype=float)
    states = np.asarray(labels, dtype=int)
    means = np.zeros((state_count, values.shape[1]), dtype=float)
    variances = np.zeros_like(means)
    occupancy_count = np.zeros(state_count, dtype=float)
    for state in range(state_count):
        state_values = values[states == state]
        if len(state_values) == 0:
            raise ValueError(f"empty fitted state {state}")
        means[state] = state_values.mean(axis=0)
        variances[state] = np.maximum(state_values.var(axis=0), variance_floor)
        occupancy_count[state] = len(state_values)
    occupancy = (occupancy_count + 0.5) / (occupancy_count.sum() + 0.5 * state_count)

    transitions = np.full((state_count, state_count), transition_pseudocount, dtype=float)
    np.fill_diagonal(transitions, 0.0)
    initial = np.full(state_count, initial_pseudocount, dtype=float)
    for raw_positions in groups:
        positions = np.asarray(raw_positions, dtype=int)
        local = states[positions]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        run_states = local[starts]
        if len(run_states) and int(positions[0]) >= 0:
            initial[int(run_states[0])] += 1.0
        for origin, destination in zip(run_states[:-1], run_states[1:], strict=True):
            if int(origin) != int(destination):
                transitions[int(origin), int(destination)] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)
    initial /= initial.sum()
    return means, variances, transitions, initial, occupancy


def _training_row_hash(panel: pd.DataFrame, sample: np.ndarray) -> str:
    training = panel.iloc[sample].copy()
    key_hash = canonical_frame_hash(training, columns=NATURAL_KEY)
    digest = hashlib.sha256()
    digest.update(key_hash.encode("ascii"))
    _hash_array(digest, "training_indices", sample.astype(np.int64))
    return digest.hexdigest()


def fit_full_right_censored_refit(
    panel: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    config: RefitConfig,
    training_indices: np.ndarray | None = None,
) -> FullRefitResult:
    """Fit the complete combined K-state lineage with corrected censoring."""

    names = tuple(str(name) for name in feature_names)
    reject_outcome_columns(names)
    required = {
        *NATURAL_KEY,
        "segment_id",
        "segment_index",
        "bar_complete_timestamp",
        "session_source_complete",
        "expected_session_bars",
        config.activity_column,
        config.direction_column,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"refit panel lacks required columns: {missing}")
    ordered = panel.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    if ordered[list(NATURAL_KEY)].duplicated().any():
        raise ValueError("refit panel natural keys are not unique")
    sample = (
        deterministic_stride_indices(
            len(ordered),
            nominal_maximum_rows=config.nominal_maximum_rows,
        )
        if training_indices is None
        else np.asarray(training_indices, dtype=np.int64)
    )
    if len(sample) < config.state_count:
        raise ValueError("training sample is smaller than state count")
    if (
        sample.ndim != 1
        or sample.min() < 0
        or sample.max() >= len(ordered)
        or len(np.unique(sample)) != len(sample)
        or np.any(np.diff(sample) <= 0)
    ):
        raise ValueError("training indices must be unique, increasing, and inside the panel")
    preprocessing = fit_emission_preprocessing(ordered.iloc[sample], feature_names=names)
    scaled = transform_emissions(ordered, preprocessing)
    groups = causal_segment_groups(ordered)
    clusterer = MiniBatchKMeans(
        n_clusters=config.state_count,
        batch_size=config.batch_size,
        n_init=config.n_init,
        max_iter=config.max_iter,
        random_state=config.seed,
    )
    clusterer.fit(scaled[sample])
    raw_labels = clusterer.predict(scaled).astype(np.int16)
    raw_centers = np.asarray(clusterer.cluster_centers_, dtype=float)
    cleaned = apply_cleaning_variant(
        raw_labels,
        scaled=scaled,
        groups=groups,
        centroids=raw_centers,
        variant=config.cleaning_variant,
    )
    semantic_labels, mapping = semantic_remap_by_activity_direction(
        cleaned,
        ordered,
        activity_column=config.activity_column,
        direction_column=config.direction_column,
    )
    if len(mapping) != config.state_count:
        raise ValueError("semantic mapping lost a fitted state")
    semantic_centers = np.empty_like(raw_centers)
    for old_state, new_state in mapping.items():
        semantic_centers[new_state] = raw_centers[old_state]

    run_input = ordered.copy()
    run_input["state"] = semantic_labels
    run_ledger = classify_training_run_endings(run_input)
    duration_fit = estimate_right_censored_durations(
        run_ledger,
        state_count=config.state_count,
        config=_duration_config(
            maximum_age=config.maximum_age,
            alpha=config.duration_alpha,
            beta=config.duration_beta,
            minimum_state_at_risk=config.minimum_state_at_risk,
            tail_prior_hazard=config.tail_prior_hazard,
        ),
    )
    means, variances, transitions, initial, occupancy = _fit_non_duration_parameters(
        scaled=scaled,
        labels=semantic_labels,
        groups=groups,
        state_count=config.state_count,
        variance_floor=config.variance_floor,
        transition_pseudocount=config.transition_pseudocount,
        initial_pseudocount=config.initial_pseudocount,
    )
    parameters = SemiMarkovParameters(
        means=means,
        variances=variances,
        duration_hazard=duration_fit.hazard.copy(),
        transitions=transitions,
        initial=initial,
        occupancy=occupancy,
    )
    parameters.validate()
    configuration: dict[str, Any] = {
        "model_id": FULL_REFIT_MODEL_ID,
        "feature_names": list(names),
        "state_count": config.state_count,
        "seed": config.seed,
        "sampling_policy": (
            "historical_deterministic_floor_stride"
            if training_indices is None
            else "caller_bound_unchanged_gate_sample"
        ),
        "nominal_maximum_rows": config.nominal_maximum_rows,
        "actual_training_rows": len(sample),
        "stride": int(sample[1] - sample[0]) if len(sample) > 1 else 1,
        "cleaning_variant": config.cleaning_variant.value,
        "cleanup_uses_future_neighbor": (config.cleaning_variant is CleaningVariant.CLEANING_1),
        "batch_size": config.batch_size,
        "n_init": config.n_init,
        "max_iter": config.max_iter,
        "maximum_age": config.maximum_age,
        "duration_alpha": config.duration_alpha,
        "duration_beta": config.duration_beta,
        "minimum_state_at_risk": config.minimum_state_at_risk,
        "tail_prior_hazard": config.tail_prior_hazard,
        "variance_floor": config.variance_floor,
        "transition_pseudocount": config.transition_pseudocount,
        "initial_pseudocount": config.initial_pseudocount,
        "semantic_remap": "ascending_mean_activity_then_direction",
    }
    parameter_hash = deterministic_parameter_hash(parameters)
    preprocessing_hash = deterministic_preprocessing_hash(preprocessing)
    model_hash = deterministic_model_hash(
        parameters=parameters,
        preprocessing=preprocessing,
        semantic_mapping=mapping,
        semantic_cluster_centers=semantic_centers,
        configuration=configuration,
    )
    iterations = int(clusterer.n_iter_)
    steps = int(clusterer.n_steps_)
    return FullRefitResult(
        model_id=FULL_REFIT_MODEL_ID,
        preprocessing=preprocessing,
        scaled=scaled,
        training_indices=sample,
        training_row_hash=_training_row_hash(ordered, sample),
        raw_labels=raw_labels,
        cleaned_labels=cleaned,
        semantic_labels=semantic_labels,
        semantic_mapping=mapping,
        raw_cluster_centers=raw_centers,
        semantic_cluster_centers=semantic_centers,
        parameters=parameters,
        run_ledger=run_ledger,
        duration_fit=duration_fit,
        training_objective=float(clusterer.inertia_),
        kmeans_iterations=iterations,
        kmeans_steps=steps,
        kmeans_converged=iterations <= config.max_iter,
        parameter_hash=parameter_hash,
        preprocessing_hash=preprocessing_hash,
        model_hash=model_hash,
        effective_configuration=configuration,
    )


__all__ = [
    "DURATION_ONLY_MODEL_ID",
    "FULL_REFIT_MODEL_ID",
    "DurationOnlyRepairResult",
    "FullRefitResult",
    "RefitConfig",
    "build_duration_only_repair",
    "deterministic_model_hash",
    "deterministic_parameter_hash",
    "deterministic_preprocessing_hash",
    "deterministic_stride_indices",
    "fit_full_right_censored_refit",
    "write_deterministic_npz",
]
