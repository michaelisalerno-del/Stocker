"""Structural-only contracts for causal regime validity V2.

This module contains no price-outcome target, execution surface, or strategy
promotion path.  It exposes deterministic cleaning, sampling, provenance, and
scientific-gate seams used by the versioned research runners.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False

FROZEN_EMISSION_FEATURES = (
    "regime_log_activity_3",
    "regime_log_activity_12",
    "regime_activity_acceleration",
    "signed_efficiency_6",
    "signed_efficiency_12",
    "regime_log_bar_range",
    "close_location_value",
    "regime_wick_balance",
    "log_relative_historical_volume",
    "log_relative_cumulative_historical_volume",
    "regime_log_market_dispersion",
    "regime_stock_minus_market_scaled",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
)


@dataclass(frozen=True, slots=True)
class EmissionFeatureProvenance:
    feature: str
    source_family: str
    latest_bar_offset: int
    history_requirement: str
    availability_rule: str
    future_rows_required: bool


def emission_feature_provenance() -> dict[str, EmissionFeatureProvenance]:
    """Declare the latest raw input and completed-bar availability of each emission."""

    market_features = {
        "regime_log_market_dispersion": "stock_cross_section_current_and_past",
        "regime_stock_minus_market_scaled": "stock_and_market_current_and_past",
        "regime_market_breadth_centered": "stock_cross_section_current_and_past",
        "vti__signed_efficiency_12": "benchmark_current_and_past",
    }
    historical_features = {
        "log_relative_historical_volume",
        "log_relative_cumulative_historical_volume",
    }
    result: dict[str, EmissionFeatureProvenance] = {}
    for feature in FROZEN_EMISSION_FEATURES:
        source_family = market_features.get(feature, "stock_current_and_past")
        history_requirement = (
            "prior_sessions_plus_current_bar"
            if feature in historical_features
            else "rolling_or_current_bar_history"
        )
        result[feature] = EmissionFeatureProvenance(
            feature=feature,
            source_family=source_family,
            latest_bar_offset=0,
            history_requirement=history_requirement,
            availability_rule="completed_bar",
            future_rows_required=False,
        )
    return result


@dataclass(frozen=True, slots=True)
class EmissionPreprocessing:
    """Training-only median imputation and robust-scaling parameters."""

    feature_names: tuple[str, ...]
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    def validate(self) -> None:
        width = len(self.feature_names)
        if len(set(self.feature_names)) != width:
            raise ValueError("emission feature names must be unique")
        if any(values.shape != (width,) for values in (self.medians, self.centers, self.scales)):
            raise ValueError("preprocessing arrays differ from the feature width")
        if not all(
            np.isfinite(values).all() for values in (self.medians, self.centers, self.scales)
        ):
            raise ValueError("preprocessing arrays contain non-finite values")
        if np.any(self.scales <= 0.0):
            raise ValueError("robust scales must be positive")


def _numeric_feature_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    names = tuple(str(value) for value in feature_names)
    reject_outcome_columns(names)
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"emission frame lacks declared features: {missing}")
    result = (
        frame.loc[:, list(names)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return cast(pd.DataFrame, result)


def fit_emission_preprocessing(
    frame: pd.DataFrame, *, feature_names: Sequence[str]
) -> EmissionPreprocessing:
    """Fit the frozen preprocessing family on development rows only."""

    names = tuple(str(value) for value in feature_names)
    raw = _numeric_feature_frame(frame, names)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    imputed = imputer.fit_transform(raw)
    scaler.fit(imputed)
    medians = np.asarray(imputer.statistics_, dtype=float)
    centers = np.asarray(scaler.center_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    preprocessing = EmissionPreprocessing(
        feature_names=names,
        medians=medians,
        centers=centers,
        scales=scales,
    )
    preprocessing.validate()
    return preprocessing


def transform_emissions(frame: pd.DataFrame, preprocessing: EmissionPreprocessing) -> np.ndarray:
    """Apply frozen preprocessing in its declared feature order."""

    preprocessing.validate()
    raw = _numeric_feature_frame(frame, preprocessing.feature_names)
    values = raw.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(preprocessing.medians, np.nonzero(missing)[1])
    scaled = ((values - preprocessing.centers) / preprocessing.scales).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("transformed emissions contain non-finite values")
    return np.asarray(scaled, dtype=np.float32)


def safety_flags() -> dict[str, object]:
    """Return the complete mandatory research safety boundary."""

    return {
        "research_only": RESEARCH_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "order_placement": ORDER_PLACEMENT,
        "broker_connected": BROKER_CONNECTED,
        "economic_outcomes_used": ECONOMIC_OUTCOMES_USED,
        "payoff_selection_used": PAYOFF_SELECTION_USED,
        "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
        "strategy_promotion": STRATEGY_PROMOTION,
    }


class CleaningVariant(StrEnum):
    """Preregistered offline-label variants."""

    CLEANING_0 = "CLEANING_0"
    CLEANING_1 = "CLEANING_1"
    CLEANING_CAUSAL = "CLEANING_CAUSAL"


@dataclass(frozen=True, slots=True)
class SemiMarkovParameters:
    """Diagonal-Gaussian semi-Markov parameters with explicit duration support."""

    means: np.ndarray
    variances: np.ndarray
    duration_hazard: np.ndarray
    transitions: np.ndarray
    initial: np.ndarray
    occupancy: np.ndarray

    def validate(self) -> None:
        state_count, feature_count = self.means.shape
        if self.variances.shape != (state_count, feature_count):
            raise ValueError("mean and variance dimensions differ")
        if self.duration_hazard.ndim != 2 or len(self.duration_hazard) != state_count:
            raise ValueError("duration hazard dimensions differ from state count")
        if self.transitions.shape != (state_count, state_count):
            raise ValueError("transition dimensions differ from state count")
        if self.initial.shape != (state_count,) or self.occupancy.shape != (state_count,):
            raise ValueError("initial or occupancy dimensions differ from state count")
        arrays = (
            self.means,
            self.variances,
            self.duration_hazard,
            self.transitions,
            self.initial,
            self.occupancy,
        )
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("semi-Markov parameters contain non-finite values")
        if np.any(self.variances <= 0.0):
            raise ValueError("emission variances must be positive")
        if np.any((self.duration_hazard < 0.0) | (self.duration_hazard > 1.0)):
            raise ValueError("duration hazards must be probabilities")
        if not np.allclose(self.transitions.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError("transition rows do not normalize")
        if not np.isclose(self.initial.sum(), 1.0, atol=1e-12):
            raise ValueError("initial probabilities do not normalize")
        if not np.isclose(self.occupancy.sum(), 1.0, atol=1e-12):
            raise ValueError("occupancy probabilities do not normalize")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means,
            "variances": self.variances,
            "duration_hazard": self.duration_hazard,
            "transitions": self.transitions,
            "initial": self.initial,
            "occupancy": self.occupancy,
        }


@dataclass(frozen=True, slots=True)
class CausalFilterSummary:
    """Memory-bounded causal posterior summaries for sensitivity fits."""

    state_probabilities: np.ndarray
    hard_states: np.ndarray
    expected_age: np.ndarray
    departure_probability: np.ndarray
    posterior_entropy: np.ndarray
    log_likelihood: np.ndarray
    iid_log_likelihood: np.ndarray


def _propagate_state_age(
    posterior: np.ndarray, hazard: np.ndarray, transitions: np.ndarray
) -> np.ndarray:
    stay = posterior * (1.0 - hazard)
    predicted = np.zeros_like(posterior)
    predicted[:, 1:] += stay[:, :-1]
    predicted[:, -1] += stay[:, -1]
    exit_mass = np.sum(posterior * hazard, axis=1)
    predicted[:, 0] += exit_mass @ transitions
    total = float(predicted.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("semi-Markov propagation lost probability mass")
    return predicted / total


def causal_filter_summary(
    log_emissions: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    model: Mapping[str, np.ndarray],
) -> CausalFilterSummary:
    """Run the same forward recursion without retaining every state-age cell."""

    emissions = np.asarray(log_emissions, dtype=float)
    hazard = np.asarray(model["duration_hazard"], dtype=float)
    transitions = np.asarray(model["transitions"], dtype=float)
    initial = np.asarray(model["initial"], dtype=float)
    if emissions.ndim != 2 or hazard.ndim != 2:
        raise ValueError("emissions and hazard must be matrices")
    row_count, state_count = emissions.shape
    if hazard.shape[0] != state_count:
        raise ValueError("hazard state count differs from emissions")
    if transitions.shape != (state_count, state_count) or initial.shape != (state_count,):
        raise ValueError("transition or initial dimensions differ from state count")
    normalized_groups = _validated_groups(groups, row_count)
    probabilities = np.zeros((row_count, state_count), dtype=float)
    hard = np.full(row_count, -1, dtype=np.int16)
    expected_age = np.zeros(row_count, dtype=float)
    departure = np.zeros(row_count, dtype=float)
    entropy = np.zeros(row_count, dtype=float)
    likelihood = np.zeros(row_count, dtype=float)
    iid_likelihood = np.full(row_count, np.nan, dtype=float)
    ages = np.arange(1, hazard.shape[1] + 1, dtype=float)[None, :]
    occupancy_value = model.get("occupancy")
    log_iid = (
        np.log(np.clip(np.asarray(occupancy_value, dtype=float), 1e-300, 1.0))
        if occupancy_value is not None
        else None
    )
    for group in normalized_groups:
        alpha: np.ndarray | None = None
        for position in group:
            if alpha is None:
                prior = np.zeros_like(hazard)
                prior[:, 0] = initial / initial.sum()
            else:
                prior = _propagate_state_age(alpha, hazard, transitions)
            emission = emissions[int(position)]
            state_prior = prior.sum(axis=1)
            likelihood[int(position)] = float(
                logsumexp(np.log(np.clip(state_prior, 1e-300, 1.0)) + emission)
            )
            if log_iid is not None:
                iid_likelihood[int(position)] = float(logsumexp(log_iid + emission))
            relative_likelihood = np.exp(emission - np.max(emission))
            posterior = prior * relative_likelihood[:, None]
            total = float(posterior.sum())
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("semi-Markov posterior underflow")
            alpha = posterior / total
            state_probability = alpha.sum(axis=1)
            probabilities[int(position)] = state_probability
            hard[int(position)] = int(np.argmax(state_probability))
            expected_age[int(position)] = float(np.sum(alpha * ages))
            departure[int(position)] = float(np.sum(alpha * hazard))
            entropy[int(position)] = float(
                -np.sum(state_probability * np.log(np.clip(state_probability, 1e-300, 1.0)))
            )
    if np.any(hard < 0):
        raise ValueError("causal filter left a row unassigned")
    return CausalFilterSummary(
        state_probabilities=probabilities,
        hard_states=hard,
        expected_age=expected_age,
        departure_probability=departure,
        posterior_entropy=entropy,
        log_likelihood=likelihood,
        iid_log_likelihood=iid_likelihood,
    )


def build_run_ledger(labels: np.ndarray, *, groups: Sequence[np.ndarray]) -> pd.DataFrame:
    """Compress labels into session-bounded runs and mark terminal censoring."""

    states = np.asarray(labels, dtype=int)
    normalized_groups = _validated_groups(groups, len(states))
    rows: list[dict[str, object]] = []
    run_id = 0
    for group_index, positions in enumerate(normalized_groups):
        local = states[positions]
        local_runs = _runs(local)
        for run_index, (start, end, state) in enumerate(local_runs):
            rows.append(
                {
                    "run_id": run_id,
                    "group_index": group_index,
                    "state": state,
                    "duration": end - start,
                    "start_position": int(positions[start]),
                    "end_position": int(positions[end - 1]),
                    "right_censored": run_index == len(local_runs) - 1,
                }
            )
            run_id += 1
    return pd.DataFrame(rows)


def estimate_semimarkov_parameters(
    scaled: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    state_count: int,
    maximum_duration: int,
    variance_floor: float = 0.05,
    censor_terminal_runs: bool = False,
    force_terminal_exit: bool = True,
) -> SemiMarkovParameters:
    """Estimate the frozen model family with an explicit censoring switch."""

    values = np.asarray(scaled, dtype=float)
    states = np.asarray(labels, dtype=int)
    if values.ndim != 2 or states.shape != (len(values),):
        raise ValueError("scaled emissions and labels have incompatible shapes")
    if state_count <= 1 or maximum_duration <= 0 or variance_floor <= 0.0:
        raise ValueError("invalid state count, duration support, or variance floor")
    if np.any((states < 0) | (states >= state_count)):
        raise ValueError("labels exceed the declared state support")
    reject_outcome_columns(())
    normalized_groups = _validated_groups(groups, len(states))
    runs = build_run_ledger(states, groups=normalized_groups)
    means = np.zeros((state_count, values.shape[1]), dtype=float)
    variances = np.zeros_like(means)
    occupancy_count = np.zeros(state_count, dtype=float)
    for state in range(state_count):
        state_values = values[states == state]
        if len(state_values) == 0:
            raise ValueError(f"empty training state {state}")
        means[state] = state_values.mean(axis=0)
        variances[state] = np.maximum(state_values.var(axis=0), variance_floor)
        occupancy_count[state] = len(state_values)
    occupancy = (occupancy_count + 0.5) / (occupancy_count.sum() + 0.5 * state_count)

    duration_hazard = np.zeros((state_count, maximum_duration), dtype=float)
    for state in range(state_count):
        state_runs = runs.loc[runs["state"].eq(state)]
        durations = state_runs["duration"].to_numpy(dtype=int)
        censored = state_runs["right_censored"].to_numpy(dtype=bool)
        for age in range(1, maximum_duration + 1):
            at_risk = int(np.sum(durations >= age))
            exact_exit = durations == age
            if censor_terminal_runs:
                exact_exit &= ~censored
            exits = int(np.sum(exact_exit))
            if force_terminal_exit and age == maximum_duration:
                exits = at_risk
            duration_hazard[state, age - 1] = np.clip((exits + 0.5) / (at_risk + 1.0), 0.01, 1.0)
        if force_terminal_exit:
            duration_hazard[state, -1] = 1.0

    transition_counts = np.full((state_count, state_count), 0.5, dtype=float)
    np.fill_diagonal(transition_counts, 0.0)
    initial_counts = np.full(state_count, 0.5, dtype=float)
    for _, group in runs.groupby("group_index", sort=False):
        run_states = group["state"].to_numpy(dtype=int)
        if len(run_states):
            initial_counts[run_states[0]] += 1.0
        for origin, destination in zip(run_states[:-1], run_states[1:], strict=True):
            if origin != destination:
                transition_counts[origin, destination] += 1.0
    transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial = initial_counts / initial_counts.sum()
    parameters = SemiMarkovParameters(
        means=means,
        variances=variances,
        duration_hazard=duration_hazard,
        transitions=transitions,
        initial=initial,
        occupancy=occupancy,
    )
    parameters.validate()
    return parameters


def gaussian_log_emissions(scaled: np.ndarray, parameters: SemiMarkovParameters) -> np.ndarray:
    """Evaluate the independently auditable diagonal-Gaussian likelihood."""

    values = np.asarray(scaled, dtype=float)
    parameters.validate()
    if values.ndim != 2 or values.shape[1] != parameters.means.shape[1]:
        raise ValueError("scaled emissions differ from fitted feature width")
    output = np.empty((len(values), len(parameters.means)), dtype=float)
    constant = np.log(2.0 * np.pi * parameters.variances)
    for state in range(len(parameters.means)):
        output[:, state] = -0.5 * np.sum(
            constant[state]
            + np.square(values - parameters.means[state]) / parameters.variances[state],
            axis=1,
        )
    return output


def semantic_remap_by_activity_direction(
    labels: np.ndarray,
    raw_features: pd.DataFrame,
    *,
    activity_column: str,
    direction_column: str,
) -> tuple[np.ndarray, dict[int, int]]:
    """Reproduce the frozen fit-time semantic ordering without trusting label IDs."""

    states = np.asarray(labels, dtype=int)
    if len(states) != len(raw_features):
        raise ValueError("labels and semantic feature rows differ")
    missing = [
        column for column in (activity_column, direction_column) if column not in raw_features
    ]
    if missing:
        raise ValueError(f"semantic remap lacks required columns: {missing}")
    summary = (
        pd.DataFrame(
            {
                "state": states,
                "activity": pd.to_numeric(raw_features[activity_column], errors="coerce"),
                "direction": pd.to_numeric(raw_features[direction_column], errors="coerce"),
            }
        )
        .groupby("state", sort=True)
        .mean()
    )
    if summary.isna().any().any():
        raise ValueError("semantic remap features contain an all-missing state")
    order = summary.sort_values(["activity", "direction"], kind="mergesort").index
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    remapped = np.asarray([mapping[int(state)] for state in states], dtype=np.int16)
    return remapped, mapping


@dataclass(frozen=True, slots=True)
class ClusteredSemiMarkovFit:
    """One deterministic audit-only clustering and semi-Markov fit."""

    state_count: int
    seed: int
    cleaning_variant: CleaningVariant
    sample_row_count: int
    training_objective: float
    raw_cluster_labels: np.ndarray
    raw_cluster_centers: np.ndarray
    cleaned_cluster_labels: np.ndarray
    semantic_labels: np.ndarray
    semantic_mapping: dict[int, int]
    semantic_cluster_centers: np.ndarray
    parameters: SemiMarkovParameters


def fit_clustered_semimarkov(
    *,
    scaled: np.ndarray,
    fit_feature_names: Sequence[str],
    semantic_features: pd.DataFrame,
    groups: Sequence[np.ndarray],
    sample_indices: np.ndarray,
    state_count: int,
    seed: int,
    cleaning_variant: CleaningVariant,
    activity_column: str,
    direction_column: str,
    maximum_duration: int,
    batch_size: int = 4096,
    n_init: int = 10,
    max_iter: int = 300,
) -> ClusteredSemiMarkovFit:
    """Fit one preregistered structural model without reading an outcome."""

    values = np.asarray(scaled, dtype=np.float32)
    feature_names = tuple(str(name) for name in fit_feature_names)
    reject_outcome_columns(feature_names)
    sample = np.asarray(sample_indices, dtype=int)
    if values.ndim != 2 or len(values) != len(semantic_features):
        raise ValueError("scaled and semantic feature rows differ")
    if len(feature_names) != values.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("fit feature schema differs from the scaled matrix width")
    normalized_groups = _validated_groups(groups, len(values))
    if sample.ndim != 1 or len(sample) < state_count:
        raise ValueError("cluster sample is too small for the declared state count")
    if sample.min() < 0 or sample.max() >= len(values) or len(np.unique(sample)) != len(sample):
        raise ValueError("cluster sample contains duplicate or invalid rows")
    if batch_size <= 0 or n_init <= 0 or max_iter <= 0:
        raise ValueError("cluster fitting parameters must be positive")
    clusterer = MiniBatchKMeans(
        n_clusters=state_count,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    clusterer.fit(values[sample])
    raw_labels = clusterer.predict(values).astype(np.int16)
    centers = np.asarray(clusterer.cluster_centers_, dtype=float)
    cleaned = apply_cleaning_variant(
        raw_labels,
        scaled=values,
        groups=normalized_groups,
        centroids=centers,
        variant=cleaning_variant,
    )
    semantic_labels, mapping = semantic_remap_by_activity_direction(
        cleaned,
        semantic_features,
        activity_column=activity_column,
        direction_column=direction_column,
    )
    if len(mapping) != state_count:
        raise ValueError("semantic remap lost a fitted state")
    semantic_centers = np.empty_like(centers)
    for old_state, new_state in mapping.items():
        semantic_centers[new_state] = centers[old_state]
    parameters = estimate_semimarkov_parameters(
        values,
        semantic_labels,
        groups=normalized_groups,
        state_count=state_count,
        maximum_duration=maximum_duration,
        variance_floor=0.05,
        censor_terminal_runs=False,
        force_terminal_exit=True,
    )
    return ClusteredSemiMarkovFit(
        state_count=state_count,
        seed=seed,
        cleaning_variant=cleaning_variant,
        sample_row_count=len(sample),
        training_objective=float(clusterer.inertia_),
        raw_cluster_labels=raw_labels,
        raw_cluster_centers=centers,
        cleaned_cluster_labels=cleaned,
        semantic_labels=semantic_labels,
        semantic_mapping=mapping,
        semantic_cluster_centers=semantic_centers,
        parameters=parameters,
    )


@dataclass(frozen=True, slots=True)
class CleaningPolicyMetadata:
    variant: CleaningVariant
    causal: bool
    uses_future_neighbor: bool
    description: str


def cleaning_policy_metadata(variant: CleaningVariant) -> CleaningPolicyMetadata:
    """Describe timing semantics without relabelling an offline rule as causal."""

    policies = {
        CleaningVariant.CLEANING_0: CleaningPolicyMetadata(
            variant=CleaningVariant.CLEANING_0,
            causal=True,
            uses_future_neighbor=False,
            description="raw cluster labels; no cleanup",
        ),
        CleaningVariant.CLEANING_1: CleaningPolicyMetadata(
            variant=CleaningVariant.CLEANING_1,
            causal=False,
            uses_future_neighbor=True,
            description="frozen two-pass neighboring-run cleanup",
        ),
        CleaningVariant.CLEANING_CAUSAL: CleaningPolicyMetadata(
            variant=CleaningVariant.CLEANING_CAUSAL,
            causal=True,
            uses_future_neighbor=False,
            description="past/current confirmation rule",
        ),
    }
    return policies[variant]


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [
        (int(start), int(end), int(labels[start])) for start, end in zip(starts, ends, strict=True)
    ]


def _validated_groups(groups: Sequence[np.ndarray], row_count: int) -> tuple[np.ndarray, ...]:
    normalized = tuple(np.asarray(group, dtype=int) for group in groups)
    assigned = np.zeros(row_count, dtype=bool)
    for group in normalized:
        if len(group) == 0:
            continue
        if np.any(np.diff(group) <= 0):
            raise ValueError("cleaning groups must be strictly increasing")
        if group.min() < 0 or group.max() >= row_count or assigned[group].any():
            raise ValueError("cleaning groups overlap or reference an invalid row")
        assigned[group] = True
    if not assigned.all():
        raise ValueError("cleaning groups do not cover every row")
    return normalized


def _legacy_clean_short_runs(
    labels: np.ndarray,
    *,
    scaled: np.ndarray,
    groups: Sequence[np.ndarray],
    centroids: np.ndarray,
    minimum_run: int,
    passes: int,
) -> np.ndarray:
    output = np.asarray(labels, dtype=np.int16).copy()
    for _ in range(passes):
        changes = 0
        for positions in groups:
            local = output[positions].copy()
            runs = _runs(local)
            for run_index, (start, end, label) in enumerate(runs):
                if end - start >= minimum_run:
                    continue
                candidates: list[int] = []
                if run_index > 0:
                    candidates.append(runs[run_index - 1][2])
                if run_index + 1 < len(runs):
                    candidates.append(runs[run_index + 1][2])
                eligible = sorted({candidate for candidate in candidates if candidate != label})
                if not eligible:
                    continue
                values = scaled[positions[start:end]]
                best = min(
                    eligible,
                    key=lambda candidate: float(np.mean(np.square(values - centroids[candidate]))),
                )
                local[start:end] = best
                changes += 1
            output[positions] = local
        if changes == 0:
            break
    return output


def _causal_confirm_states(
    labels: np.ndarray, *, groups: Sequence[np.ndarray], confirmations: int
) -> np.ndarray:
    output = np.asarray(labels, dtype=np.int16).copy()
    for positions in groups:
        raw = labels[positions]
        if len(raw) == 0:
            continue
        current = int(raw[0])
        pending = -1
        pending_count = 0
        output[int(positions[0])] = current
        for position, candidate_value in zip(positions[1:], raw[1:], strict=True):
            candidate = int(candidate_value)
            if candidate == current:
                pending = -1
                pending_count = 0
            elif candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1
            if pending_count >= confirmations:
                current = pending
                pending = -1
                pending_count = 0
            output[int(position)] = current
    return output


def apply_cleaning_variant(
    labels: np.ndarray,
    *,
    scaled: np.ndarray,
    groups: Sequence[np.ndarray],
    centroids: np.ndarray,
    variant: CleaningVariant,
    minimum_run: int = 2,
    passes: int = 2,
) -> np.ndarray:
    """Apply exactly one declared cleanup policy."""

    states = np.asarray(labels, dtype=np.int16)
    values = np.asarray(scaled, dtype=float)
    centers = np.asarray(centroids, dtype=float)
    if states.ndim != 1 or values.ndim != 2 or len(states) != len(values):
        raise ValueError("labels and scaled emissions have incompatible shapes")
    if centers.ndim != 2 or centers.shape[1] != values.shape[1]:
        raise ValueError("centroids and scaled emissions have incompatible shapes")
    if minimum_run <= 0 or passes <= 0:
        raise ValueError("minimum_run and passes must be positive")
    normalized_groups = _validated_groups(groups, len(states))
    if variant is CleaningVariant.CLEANING_0:
        return states.copy()
    if variant is CleaningVariant.CLEANING_1:
        return _legacy_clean_short_runs(
            states,
            scaled=values,
            groups=normalized_groups,
            centroids=centers,
            minimum_run=minimum_run,
            passes=passes,
        )
    if variant is CleaningVariant.CLEANING_CAUSAL:
        return _causal_confirm_states(
            states,
            groups=normalized_groups,
            confirmations=minimum_run,
        )
    raise ValueError(f"unsupported cleaning variant: {variant}")


def audit_feature_availability(
    frame: pd.DataFrame, *, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Verify every declared feature is available by its completed-bar decision."""

    if "decision_timestamp" not in frame:
        raise ValueError("decision_timestamp is required")
    result = pd.DataFrame(index=frame.index)
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
    valid = decision.notna()
    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    for feature in feature_names:
        source_column = f"{feature}__source_timestamp"
        available_column = f"{feature}__available_timestamp"
        missing = [name for name in (source_column, available_column) if name not in frame]
        if missing:
            raise ValueError(f"missing feature provenance columns: {missing}")
        source = pd.to_datetime(frame[source_column], utc=True, errors="coerce")
        available = pd.to_datetime(frame[available_column], utc=True, errors="coerce")
        feature_valid = (
            source.notna() & available.notna() & source.le(available) & available.le(decision)
        )
        valid &= feature_valid
        for ordinal, passed in enumerate(feature_valid.to_numpy(dtype=bool)):
            if not passed:
                reasons[ordinal].append(feature)
    result["causal_pass"] = valid.astype(bool)
    result["failing_features"] = [tuple(values) for values in reasons]
    return result


_FORBIDDEN_OUTCOME_TOKENS = (
    "forward",
    "future",
    "next_",
    "lead_",
    "target",
    "outcome",
    "future_return",
    "future_price",
    "payoff",
    "pnl",
    "profit",
    "mfe",
    "mae",
    "execution_outcome",
    "economic_outcome",
    "hindsight_positive",
)


def reject_outcome_columns(columns: Sequence[str]) -> None:
    """Fail closed if a state fit is offered an economic or future outcome."""

    rejected = sorted(
        name
        for name in columns
        if any(token in name.lower() for token in _FORBIDDEN_OUTCOME_TOKENS)
    )
    if rejected:
        raise ValueError(f"economic or future outcome columns are forbidden: {rejected}")


@dataclass(frozen=True, slots=True)
class EmissionFeaturePartition:
    stock: frozenset[str]
    market: frozenset[str]
    relative: frozenset[str]

    @property
    def combined(self) -> frozenset[str]:
        return self.stock | self.market | self.relative


def frozen_emission_partition() -> EmissionFeaturePartition:
    """Return disjoint feature roles for the frozen 14-feature representation."""

    market = frozenset(
        {
            "regime_log_market_dispersion",
            "vti__signed_efficiency_12",
            "regime_market_breadth_centered",
        }
    )
    relative = frozenset({"regime_stock_minus_market_scaled"})
    stock = frozenset(FROZEN_EMISSION_FEATURES).difference(market | relative)
    partition = EmissionFeaturePartition(stock=stock, market=market, relative=relative)
    if partition.combined != frozenset(FROZEN_EMISSION_FEATURES):
        raise AssertionError("frozen emission partition is incomplete")
    return partition


def deterministic_model_registry(state_counts: Sequence[int], seeds: Sequence[int]) -> pd.DataFrame:
    """Enumerate a stable K/seed registry without fitting or ranking models."""

    pairs = [(int(state_count), int(seed)) for state_count in state_counts for seed in seeds]
    rows = [
        {
            "model_id": f"regime_k{state_count}_seed{seed}",
            "state_count": state_count,
            "seed": seed,
            "registry_ordinal": ordinal,
        }
        for ordinal, (state_count, seed) in enumerate(pairs)
    ]
    return pd.DataFrame(rows)


def _sampling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "symbol" not in result and "symbol_norm" in result:
        result["symbol"] = result["symbol_norm"]
    if "month" not in result:
        if "session" in result:
            result["month"] = result["session"].astype(str).str[:7]
        elif "session_date" in result:
            result["month"] = result["session_date"].astype(str).str[:7]
        else:
            raise ValueError("month or session is required for stratified sampling")
    if "symbol" not in result:
        raise ValueError("symbol is required for stratified sampling")
    return result


def build_training_sample(
    frame: pd.DataFrame,
    *,
    variant: str,
    maximum_rows: int,
    seed: int,
) -> np.ndarray:
    """Build one bounded deterministic training sample using declared strata."""

    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    row_count = len(frame)
    target = min(row_count, maximum_rows)
    if target == 0:
        return np.asarray([], dtype=int)
    if variant == "SAMPLE_A":
        step = max(1, row_count // maximum_rows)
        return np.arange(0, row_count, step, dtype=int)[:target]
    columns_by_variant = {
        "SAMPLE_B": ("symbol",),
        "SAMPLE_C": ("symbol", "month"),
        "SAMPLE_D": ("symbol", "month", "clock_phase"),
    }
    if variant not in columns_by_variant:
        raise ValueError(f"unsupported training sample variant: {variant}")
    working = _sampling_columns(frame)
    strata = columns_by_variant[variant]
    missing = [column for column in strata if column not in working]
    if missing:
        raise ValueError(f"missing sampling strata: {missing}")
    grouped = [
        np.asarray(group.index, dtype=int)
        for _, group in working.groupby(list(strata), sort=True, dropna=False)
    ]
    if target < len(grouped):
        raise ValueError("maximum_rows is smaller than the declared stratum count")
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(indices) for indices in grouped]
    base, remainder = divmod(target, len(shuffled))
    chosen: list[int] = []
    leftovers: list[int] = []
    for ordinal, indices in enumerate(shuffled):
        quota = base + (1 if ordinal < remainder else 0)
        take = min(quota, len(indices))
        chosen.extend(int(value) for value in indices[:take])
        leftovers.extend(int(value) for value in indices[take:])
    if len(chosen) < target:
        refill = rng.permutation(np.asarray(leftovers, dtype=int))
        chosen.extend(int(value) for value in refill[: target - len(chosen)])
    if len(chosen) != target:
        raise AssertionError("stratified sample could not reach its bounded row count")
    return np.asarray(sorted(chosen), dtype=int)


class PartADecision(StrEnum):
    VALIDATED = "regime_representation_validated_for_loop_dictionary"
    VALID_WITH_SENSITIVITY = "regime_representation_valid_with_required_sensitivity"
    REQUIRES_TARGETED_REPAIR = "regime_representation_requires_targeted_repair"
    UNSTABLE = "regime_representation_unstable_loop_dictionary_must_pause"
    HIERARCHICAL_PREFERRED = "hierarchical_market_stock_regime_representation_preferred"
    BLOCKED = "regime_validity_audit_blocked"


@dataclass(frozen=True, slots=True)
class PartAGateEvidence:
    source_available: bool
    exact_reconstruction_pass: bool
    independent_audit_reproducible: bool
    mathematical_audit_pass: bool
    posterior_duration_pass: bool
    critical_future_leakage: bool
    hysteretic_same_primitive_fraction: float
    k8_selected_loop_seed_gate_pass: bool
    minimum_state_occupancy: float
    maximum_single_stock_share: float
    semantic_drift_pass: bool
    training_sample_dictionary_coverage_ratio: float
    combined_stability_deficit: float
    representation_sensitive: bool
    usable_with_sensitivity: bool
    recoverable_local_defect: bool
    hierarchical_materially_more_stable: bool
    hierarchical_reproducible: bool

    @classmethod
    def passing(cls) -> PartAGateEvidence:
        return cls(
            source_available=True,
            exact_reconstruction_pass=True,
            independent_audit_reproducible=True,
            mathematical_audit_pass=True,
            posterior_duration_pass=True,
            critical_future_leakage=False,
            hysteretic_same_primitive_fraction=0.90,
            k8_selected_loop_seed_gate_pass=True,
            minimum_state_occupancy=0.05,
            maximum_single_stock_share=0.10,
            semantic_drift_pass=True,
            training_sample_dictionary_coverage_ratio=0.90,
            combined_stability_deficit=0.0,
            representation_sensitive=False,
            usable_with_sensitivity=True,
            recoverable_local_defect=True,
            hierarchical_materially_more_stable=False,
            hierarchical_reproducible=False,
        )

    def with_updates(self, **changes: Any) -> PartAGateEvidence:
        return replace(self, **changes)


def decide_part_a(evidence: PartAGateEvidence) -> PartADecision:
    """Apply preregistered structural gates before the independent audit runs.

    Independent reproducibility is intentionally enforced later by
    :func:`authorize_part_b`; requiring it here would make the primary decision
    circular because the auditor must reconstruct an already frozen decision.
    """

    if not (evidence.source_available and evidence.exact_reconstruction_pass):
        return PartADecision.BLOCKED
    causal_core_pass = (
        evidence.mathematical_audit_pass
        and evidence.posterior_duration_pass
        and not evidence.critical_future_leakage
    )
    if not causal_core_pass:
        return (
            PartADecision.REQUIRES_TARGETED_REPAIR
            if evidence.recoverable_local_defect
            else PartADecision.UNSTABLE
        )
    state_language_pass = (
        evidence.minimum_state_occupancy >= 0.01
        and evidence.maximum_single_stock_share <= 0.25
        and evidence.semantic_drift_pass
        and evidence.combined_stability_deficit <= 0.10
    )
    if not state_language_pass:
        return PartADecision.UNSTABLE
    loop_language_pass = (
        evidence.hysteretic_same_primitive_fraction >= 0.75
        and evidence.k8_selected_loop_seed_gate_pass
        and evidence.training_sample_dictionary_coverage_ratio >= 0.75
    )
    if evidence.hierarchical_materially_more_stable and evidence.hierarchical_reproducible:
        return PartADecision.HIERARCHICAL_PREFERRED
    if not loop_language_pass:
        return PartADecision.UNSTABLE
    if loop_language_pass and not evidence.representation_sensitive:
        return PartADecision.VALIDATED
    if evidence.usable_with_sensitivity:
        return PartADecision.VALID_WITH_SENSITIVITY
    return PartADecision.UNSTABLE


@dataclass(frozen=True, slots=True)
class PartABinding:
    decision: PartADecision
    state_model_hash: str
    state_count: int
    state_representation: str
    hysteresis_policy: tuple[tuple[str, float], ...]
    posterior_support_fields: tuple[str, ...]
    state_alignment_hash: str
    binding_hash: str


class PartBBlockedError(RuntimeError):
    """Raised when interaction scoring is attempted without an audited Part A gate."""


def _binding_payload(
    decision: PartADecision,
    *,
    state_model_hash: str,
    state_count: int,
    state_representation: str,
    hysteresis_policy: tuple[tuple[str, float], ...],
    posterior_support_fields: tuple[str, ...],
    state_alignment_hash: str,
) -> dict[str, Any]:
    return {
        "decision": decision.value,
        "state_model_hash": state_model_hash,
        "state_count": state_count,
        "state_representation": state_representation,
        "hysteresis_policy": list(hysteresis_policy),
        "posterior_support_fields": list(posterior_support_fields),
        "state_alignment_hash": state_alignment_hash,
        **safety_flags(),
    }


def freeze_part_a_binding(
    decision: PartADecision,
    *,
    state_model_hash: str,
    state_count: int,
    state_representation: str,
    hysteresis_policy: Mapping[str, float],
    posterior_support_fields: Sequence[str],
    state_alignment_hash: str,
) -> PartABinding:
    """Create the immutable identity Part B must bind before reading scores."""

    if len(state_model_hash) != 64 or len(state_alignment_hash) != 64:
        raise ValueError("state model and alignment hashes must be SHA-256 identities")
    if state_count <= 1:
        raise ValueError("state_count must exceed one")
    policy = tuple(sorted((str(key), float(value)) for key, value in hysteresis_policy.items()))
    fields = tuple(str(value) for value in posterior_support_fields)
    payload = _binding_payload(
        decision,
        state_model_hash=state_model_hash,
        state_count=state_count,
        state_representation=state_representation,
        hysteresis_policy=policy,
        posterior_support_fields=fields,
        state_alignment_hash=state_alignment_hash,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PartABinding(
        decision=decision,
        state_model_hash=state_model_hash,
        state_count=state_count,
        state_representation=state_representation,
        hysteresis_policy=policy,
        posterior_support_fields=fields,
        state_alignment_hash=state_alignment_hash,
        binding_hash=digest,
    )


def authorize_part_b(binding: PartABinding, *, independently_audited: bool) -> PartABinding:
    """Fail closed unless the frozen decision explicitly authorizes Part B."""

    authorized = {
        PartADecision.VALIDATED,
        PartADecision.VALID_WITH_SENSITIVITY,
        PartADecision.HIERARCHICAL_PREFERRED,
    }
    if not independently_audited or binding.decision not in authorized:
        raise PartBBlockedError(
            f"Part B blocked by Part A decision {binding.decision.value!r} "
            f"and independently_audited={independently_audited}"
        )
    return binding


__all__ = [
    "CausalFilterSummary",
    "ClusteredSemiMarkovFit",
    "CleaningPolicyMetadata",
    "CleaningVariant",
    "EmissionFeaturePartition",
    "EmissionFeatureProvenance",
    "EmissionPreprocessing",
    "FROZEN_EMISSION_FEATURES",
    "PartABinding",
    "PartADecision",
    "PartAGateEvidence",
    "PartBBlockedError",
    "SemiMarkovParameters",
    "apply_cleaning_variant",
    "audit_feature_availability",
    "authorize_part_b",
    "build_run_ledger",
    "build_training_sample",
    "causal_filter_summary",
    "cleaning_policy_metadata",
    "decide_part_a",
    "deterministic_model_registry",
    "emission_feature_provenance",
    "estimate_semimarkov_parameters",
    "fit_clustered_semimarkov",
    "fit_emission_preprocessing",
    "freeze_part_a_binding",
    "frozen_emission_partition",
    "gaussian_log_emissions",
    "reject_outcome_columns",
    "safety_flags",
    "semantic_remap_by_activity_direction",
    "transform_emissions",
]
