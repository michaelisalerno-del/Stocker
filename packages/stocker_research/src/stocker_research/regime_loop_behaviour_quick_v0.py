"""Reusable seams for the research-only regime/loop/behaviour quick screen V0.

The module is deliberately structural.  It contains candidate construction,
weighting, fixed interactions, deterministic modelling, and resampling helpers;
it has no execution surface and opens no economic outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from stocker_research.causal_state_export_v2 import propagate_state_age_posterior
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel, StructuralOutcomeRow
from stocker_research.loop_prefix_automaton_v2 import EventTrace

type FloatArray = NDArray[np.float64]

RESEARCH_ONLY = True
QUICK_FEASIBILITY_SCREEN = True
STRUCTURAL_PREDICTION_ONLY = True
PRE_COMPLETION_CONTEXT_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_INTEGRATION_REQUIRED = False
STRATEGY_PROMOTION = False
ECONOMIC_OUTCOMES_OPENED = False
PRODUCTION_RUNTIME_MODIFIED = False

SAFETY_FLAGS: dict[str, object] = {
    "research_only": RESEARCH_ONLY,
    "quick_feasibility_screen": QUICK_FEASIBILITY_SCREEN,
    "structural_prediction_only": STRUCTURAL_PREDICTION_ONLY,
    "pre_completion_context_only": PRE_COMPLETION_CONTEXT_ONLY,
    "execution_enabled": EXECUTION_ENABLED,
    "order_placement": ORDER_PLACEMENT,
    "broker_integration_required": BROKER_INTEGRATION_REQUIRED,
    "strategy_promotion": STRATEGY_PROMOTION,
    "economic_outcomes_opened": ECONOMIC_OUTCOMES_OPENED,
    "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
}

BEHAVIOURAL_DIMENSIONS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)

INTERACTION_FEATURES = (
    "orientation_pressure_alignment",
    "prefix_conviction",
    "transition_arousal",
    "repeat_tension",
    "next_leg_exhaustion_alignment",
)

DECISIONS = frozenset(
    {
        "behavioural_context_improves_loop_completion",
        "structural_behavioural_interactions_improve_loop_completion",
        "behavioural_main_effects_only",
        "no_behavioural_context_increment",
        "blocked_behavioural_ledger_not_reconstructable",
        "blocked_v2_prefix_population_not_reconstructable",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_insufficient_active_prefix_support",
        "blocked_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)

BEHAVIOURAL_NATURAL_KEY = (
    "symbol",
    "session",
    "decision_ordinal",
    "decision_timestamp",
)


@dataclass(frozen=True, slots=True)
class CausalCheckpointFilter:
    """Memory-bounded V2 posterior quantities needed by checkpoint candidates."""

    state_probabilities: FloatArray
    next_state_probabilities: FloatArray
    hard_states: NDArray[np.int16]
    posterior_entropy: FloatArray
    top_state_probability: FloatArray
    top_versus_second_margin: FloatArray
    expected_state_age: FloatArray
    current_persistence_probability: FloatArray
    current_transition_probability: FloatArray


def causal_checkpoint_filter(
    log_emissions: NDArray[np.floating[Any]],
    *,
    groups: Sequence[NDArray[np.integer[Any]]],
    model: Mapping[str, NDArray[np.floating[Any]]],
) -> CausalCheckpointFilter:
    """Run the frozen V2 recursion without retaining every state-age posterior."""

    emissions = np.asarray(log_emissions, dtype=np.float64)
    hazard = np.asarray(model["duration_hazard"], dtype=np.float64)
    initial = np.asarray(model["initial"], dtype=np.float64)
    transitions = np.asarray(model["transitions"], dtype=np.float64)
    if emissions.ndim != 2 or hazard.ndim != 2:
        raise ValueError("emissions and duration hazard must be matrices")
    row_count, state_count = emissions.shape
    if (
        hazard.shape[0] != state_count
        or initial.shape != (state_count,)
        or transitions.shape != (state_count, state_count)
    ):
        raise ValueError("causal checkpoint model dimensions differ")
    probabilities = np.zeros((row_count, state_count), dtype=np.float64)
    next_probabilities = np.zeros_like(probabilities)
    hard = np.full(row_count, -1, dtype=np.int16)
    entropy = np.zeros(row_count, dtype=np.float64)
    top_probability = np.zeros(row_count, dtype=np.float64)
    margin = np.zeros(row_count, dtype=np.float64)
    expected_age = np.zeros(row_count, dtype=np.float64)
    persistence = np.zeros(row_count, dtype=np.float64)
    transition_probability = np.zeros(row_count, dtype=np.float64)
    assigned = np.zeros(row_count, dtype=bool)
    ages = np.arange(1, hazard.shape[1] + 1, dtype=np.float64)[None, :]
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=np.int64)
        if len(group) == 0:
            continue
        if np.any(np.diff(group) <= 0):
            raise ValueError("causal checkpoint groups must be strictly increasing")
        alpha: FloatArray | None = None
        for raw_position in group:
            position = int(raw_position)
            if position < 0 or position >= row_count or assigned[position]:
                raise ValueError("causal checkpoint groups overlap or reference an invalid row")
            if alpha is None:
                prior = np.zeros_like(hazard)
                prior[:, 0] = initial / initial.sum()
            else:
                prior = propagate_state_age_posterior(alpha, model)
            likelihood = np.exp(emissions[position] - np.max(emissions[position]))
            posterior = prior * likelihood[:, None]
            total = float(posterior.sum())
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("causal checkpoint posterior underflow")
            alpha = np.asarray(posterior / total, dtype=np.float64)
            state_probability = alpha.sum(axis=1)
            predicted = propagate_state_age_posterior(alpha, model)
            order = np.argsort(-state_probability, kind="stable")
            departure = float(np.sum(alpha * hazard))
            probabilities[position] = state_probability
            next_probabilities[position] = predicted.sum(axis=1)
            hard[position] = int(order[0])
            entropy[position] = float(
                -np.sum(state_probability * np.log(np.clip(state_probability, 1e-300, 1.0)))
            )
            top_probability[position] = float(state_probability[order[0]])
            margin[position] = float(state_probability[order[0]] - state_probability[order[1]])
            expected_age[position] = float(np.sum(alpha * ages))
            transition_probability[position] = departure
            persistence[position] = 1.0 - departure
            assigned[position] = True
    if not assigned.all() or np.any(hard < 0):
        raise ValueError("causal checkpoint filter left an input row unassigned")
    return CausalCheckpointFilter(
        state_probabilities=probabilities,
        next_state_probabilities=next_probabilities,
        hard_states=hard,
        posterior_entropy=entropy,
        top_state_probability=top_probability,
        top_versus_second_margin=margin,
        expected_state_age=expected_age,
        current_persistence_probability=persistence,
        current_transition_probability=transition_probability,
    )


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def verify_behavioural_ledger_reconstruction(
    primary: pd.DataFrame,
    exact_rerun: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Require identical natural keys and frozen ten-dimensional values."""

    required = set(BEHAVIOURAL_NATURAL_KEY).union(BEHAVIOURAL_DIMENSIONS)
    for label, frame in (("primary", primary), ("exact_rerun", exact_rerun)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"behavioural ledger not reconstructable: {label} lacks {missing}")
        if frame.duplicated(list(BEHAVIOURAL_NATURAL_KEY)).any():
            raise ValueError(
                f"behavioural ledger not reconstructable: {label} natural keys are not unique"
            )
    left = primary.loc[:, [*BEHAVIOURAL_NATURAL_KEY, *BEHAVIOURAL_DIMENSIONS]].copy()
    right = exact_rerun.loc[:, [*BEHAVIOURAL_NATURAL_KEY, *BEHAVIOURAL_DIMENSIONS]].copy()
    left["decision_timestamp"] = _utc_series(left["decision_timestamp"])
    right["decision_timestamp"] = _utc_series(right["decision_timestamp"])
    left = left.sort_values(list(BEHAVIOURAL_NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    right = right.sort_values(list(BEHAVIOURAL_NATURAL_KEY), kind="mergesort").reset_index(
        drop=True
    )
    if len(left) != len(right) or not left[list(BEHAVIOURAL_NATURAL_KEY)].equals(
        right[list(BEHAVIOURAL_NATURAL_KEY)]
    ):
        raise ValueError("behavioural ledger not reconstructable: natural keys differ")
    left_values = left[list(BEHAVIOURAL_DIMENSIONS)].to_numpy(dtype=np.float64)
    right_values = right[list(BEHAVIOURAL_DIMENSIONS)].to_numpy(dtype=np.float64)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("behavioural ledger not reconstructable: non-finite frozen value")
    maximum_error = float(np.max(np.abs(left_values - right_values), initial=0.0))
    if maximum_error > tolerance:
        raise ValueError(
            "behavioural ledger not reconstructable: "
            f"maximum absolute error {maximum_error:.17g} exceeds {tolerance:.1e}"
        )
    return {
        **SAFETY_FLAGS,
        "rows": len(left),
        "natural_key_columns": list(BEHAVIOURAL_NATURAL_KEY),
        "dimension_columns": list(BEHAVIOURAL_DIMENSIONS),
        "maximum_absolute_error": maximum_error,
        "required_maximum_absolute_error": tolerance,
        "passed": True,
    }


def assert_decision_time_causality(
    frame: pd.DataFrame,
    *,
    source_column: str = "source_timestamp",
    available_column: str = "available_timestamp",
    decision_column: str = "decision_timestamp",
) -> None:
    """Reject a predictor whose source or availability is after its decision."""

    required = {source_column, available_column, decision_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"causality audit lacks timestamp columns: {missing}")
    source = _utc_series(frame[source_column])
    available = _utc_series(frame[available_column])
    decision = _utc_series(frame[decision_column])
    if source.isna().any() or available.isna().any() or decision.isna().any():
        raise ValueError("causality audit contains a missing timestamp")
    if (available < source).any():
        raise ValueError("feature availability precedes its source timestamp")
    if (source > decision).any() or (available > decision).any():
        raise ValueError("feature source or availability is after decision")


def _orientation_path(orientation_id: str) -> tuple[int, ...]:
    marker = "__o_"
    if marker not in orientation_id:
        raise ValueError(f"orientation ID lacks route: {orientation_id}")
    try:
        path = tuple(int(value) for value in orientation_id.split(marker, maxsplit=1)[1].split("-"))
    except ValueError as error:
        raise ValueError(f"orientation ID has a nonnumeric route: {orientation_id}") from error
    if len(path) < 3 or path[0] != path[-1]:
        raise ValueError(f"orientation ID is not a closed path: {orientation_id}")
    return path


def active_prefix_records(
    trace: EventTrace,
    *,
    decision_event_index: int,
    decision_bar_ordinal: int,
) -> list[dict[str, object]]:
    """Reconstruct every active proper prefix with at least one state transition."""

    if decision_event_index < 0 or decision_event_index >= len(trace.state_events):
        raise ValueError("decision event index is outside the trace")
    decision_event = trace.state_events[decision_event_index]
    if decision_event.bar_ordinal > decision_bar_ordinal:
        raise ValueError("decision state event occurs after the completed decision bar")
    records: list[dict[str, object]] = []
    for prefix in trace.prefixes_after_event[decision_event_index]:
        if prefix.progress_states < 2:
            continue
        full_path = _orientation_path(prefix.orientation_id)
        if tuple(full_path[: prefix.progress_states]) != prefix.prefix_path:
            raise ValueError("active prefix does not match its registered oriented path")
        if prefix.progress_states >= len(full_path):
            raise ValueError("completed loop entered the active-prefix population")
        full_transitions = len(full_path) - 1
        matched_transitions = prefix.progress_states - 1
        start = trace.state_events[prefix.start_event_index]
        records.append(
            {
                "semantic_loop_id": prefix.semantic_loop_id,
                "primitive_loop_id": prefix.primitive_loop_id,
                "candidate_orientation": prefix.orientation_id,
                "candidate_class": prefix.motif_type.value,
                "candidate_path_length": full_transitions,
                "repeat_depth": (
                    int(prefix.repeat_depth) if prefix.motif_type.value == "repeat" else 0
                ),
                "prefix_path": "->".join(str(value) for value in prefix.prefix_path),
                "prefix_matched_length": matched_transitions,
                "prefix_completion_fraction": matched_transitions / full_transitions,
                "prefix_age": int(decision_bar_ordinal - start.bar_ordinal),
                "next_required_state": int(full_path[prefix.progress_states]),
                "prefix_start_event_index": int(prefix.start_event_index),
                "prefix_start_bar_ordinal": int(start.bar_ordinal),
                "prefix_start_timestamp": pd.Timestamp(prefix.start_prefix_timestamp),
                "prefix_available_timestamp": pd.Timestamp(prefix.start_prefix_available_timestamp),
            }
        )
    records.sort(
        key=lambda row: (
            str(row["semantic_loop_id"]),
            str(row["candidate_orientation"]),
            int(cast(int, row["prefix_matched_length"])),
        )
    )
    return records


def target_candidate_rows(
    candidates: pd.DataFrame,
    outcome: StructuralOutcomeRow,
) -> tuple[pd.DataFrame, bool]:
    """Apply the exact V2 first-event target, excluding registered ties."""

    target = "candidate_completes_first_within_6_bars"
    if outcome.primary_label == PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION:
        empty = candidates.iloc[0:0].copy()
        empty[target] = pd.Series(dtype=np.int8)
        return empty, True
    output = candidates.copy()
    output[target] = np.int8(0)
    if len(outcome.earliest_registered_events) == 1:
        event = outcome.earliest_registered_events[0]
        matches = output["semantic_loop_id"].astype(str).eq(event.semantic_loop_id) & output[
            "candidate_orientation"
        ].astype(str).eq(event.orientation_id)
        output.loc[matches, target] = np.int8(1)
    return output, False


def normalize_first_event_outcome(
    outcome: StructuralOutcomeRow,
    *,
    decision_bar_ordinal: int,
    horizon_bars: int,
    session_end_bar_ordinal: int,
) -> StructuralOutcomeRow:
    """Distinguish no horizon event from a session that truly ends in-horizon."""

    if horizon_bars <= 0:
        raise ValueError("first-event horizon must be positive")
    horizon_end = int(decision_bar_ordinal) + int(horizon_bars)
    if (
        outcome.primary_label == PrimaryOutcomeLabel.SESSION_END
        and int(session_end_bar_ordinal) > horizon_end
    ):
        return replace(
            outcome,
            primary_label=str(PrimaryOutcomeLabel.NO_REGISTERED_LOOP_WITHIN_HORIZON),
        )
    return outcome


def assign_candidate_weights(candidates: pd.DataFrame) -> pd.DataFrame:
    """Equalise candidates within stock decisions and eligible stocks within slates."""

    required = {"slate_id", "decision_id", "symbol"}
    missing = sorted(required.difference(candidates.columns))
    if missing or candidates.empty:
        raise ValueError(f"candidate weighting lacks a nonempty population or fields: {missing}")
    frame = candidates.copy()
    decisions_per_stock = frame.groupby(["slate_id", "symbol"], sort=True)["decision_id"].nunique()
    if not decisions_per_stock.eq(1).all():
        raise ValueError("a stock has multiple decision identities in one slate")
    candidate_counts = frame.groupby("decision_id", sort=True)["decision_id"].transform("size")
    eligible_counts = frame.groupby("slate_id", sort=True)["decision_id"].transform("nunique")
    frame["active_candidate_count_for_stock_session_checkpoint"] = candidate_counts.astype(int)
    frame["eligible_stocks_in_session_checkpoint"] = eligible_counts.astype(int)
    frame["candidate_weight"] = 1.0 / candidate_counts.to_numpy(dtype=np.float64)
    frame["slate_weight"] = 1.0 / eligible_counts.to_numpy(dtype=np.float64)
    frame["row_weight"] = frame["candidate_weight"] * frame["slate_weight"]
    decision_totals = frame.groupby("decision_id", sort=True)["row_weight"].sum()
    expected = frame.groupby("decision_id", sort=True)["slate_weight"].first()
    if not np.allclose(decision_totals, expected, atol=1e-12, rtol=0.0):
        raise AssertionError("candidate row weights do not equal one stock share")
    slate_totals = frame.groupby("slate_id", sort=True)["row_weight"].sum()
    if not np.allclose(slate_totals, 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("candidate row weights do not sum to one per slate")
    return frame


def compute_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct exactly the five preregistered structural/behavioural products."""

    required = {
        "candidate_orientation_sign",
        "signed_pressure",
        "prefix_completion_fraction",
        "conviction",
        "current_transition_probability",
        "arousal",
        "repeat_depth",
        "tension",
        "probability_of_next_required_state",
        "signed_exhaustion",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"interaction inputs are incomplete: {missing}")
    numeric = frame.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("interaction inputs contain a non-finite value")
    output = pd.DataFrame(index=frame.index)
    output["orientation_pressure_alignment"] = frame["candidate_orientation_sign"].astype(
        float
    ) * frame["signed_pressure"].astype(float)
    output["prefix_conviction"] = frame["prefix_completion_fraction"].astype(float) * frame[
        "conviction"
    ].astype(float)
    output["transition_arousal"] = frame["current_transition_probability"].astype(float) * frame[
        "arousal"
    ].astype(float)
    output["repeat_tension"] = frame["repeat_depth"].astype(float) * frame["tension"].astype(float)
    output["next_leg_exhaustion_alignment"] = (
        frame["probability_of_next_required_state"].astype(float)
        * frame["candidate_orientation_sign"].astype(float)
        * frame["signed_exhaustion"].astype(float)
    )
    return output.loc[:, list(INTERACTION_FEATURES)]


def fit_interaction_clipping(
    development: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Fit deterministic 1st/99th percentile bounds on development rows only."""

    missing = sorted(set(INTERACTION_FEATURES).difference(development.columns))
    if missing or development.empty:
        raise ValueError(f"interaction clipping lacks development fields or rows: {missing}")
    output: dict[str, tuple[float, float]] = {}
    for feature in INTERACTION_FEATURES:
        values = pd.to_numeric(development[feature], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"interaction {feature} contains a non-finite development value")
        lower, upper = np.quantile(values, [0.01, 0.99], method="linear")
        output[feature] = (float(lower), float(upper))
    return output


def apply_interaction_clipping(
    interactions: pd.DataFrame,
    bounds: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Apply frozen interaction bounds without changing other values."""

    output = interactions.copy()
    if set(bounds) != set(INTERACTION_FEATURES):
        raise ValueError("interaction clipping bounds differ from the registered five")
    for feature in INTERACTION_FEATURES:
        pair = tuple(float(value) for value in bounds[feature])
        if len(pair) != 2 or not math.isfinite(pair[0]) or not math.isfinite(pair[1]):
            raise ValueError(f"invalid clipping bounds for {feature}")
        if pair[0] > pair[1]:
            raise ValueError(f"reversed clipping bounds for {feature}")
        output[feature] = pd.to_numeric(output[feature], errors="raise").clip(*pair)
    return output


def _weighted_location_scale(
    values: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("model row weights must have positive finite mass")
    means = np.sum(values * weights[:, None], axis=0) / total
    variance = np.sum(np.square(values - means) * weights[:, None], axis=0) / total
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    return np.asarray(means, dtype=np.float64), np.asarray(scales, dtype=np.float64)


def _categorical_design(
    frame: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[FloatArray, tuple[str, ...]]:
    blocks: list[FloatArray] = []
    names: list[str] = []
    for column, raw_levels in categories.items():
        if column not in frame:
            raise ValueError(f"categorical feature is absent: {column}")
        values = frame[column].astype(str).to_numpy()
        for level in raw_levels:
            name = str(level)
            blocks.append(np.asarray(values == name, dtype=np.float64)[:, None])
            names.append(f"{column}={name}")
    if not blocks:
        return np.empty((len(frame), 0), dtype=np.float64), ()
    return np.concatenate(blocks, axis=1), tuple(names)


@dataclass(frozen=True, slots=True)
class FrozenCandidateLogistic:
    """JSON-serializable development-fitted deterministic candidate model."""

    model_id: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    numeric_means: FloatArray
    numeric_scales: FloatArray
    design_feature_names: tuple[str, ...]
    coefficients: FloatArray
    intercept: float
    training_rows: int
    training_decisions: int
    iterations: int
    converged: bool

    def design(self, frame: pd.DataFrame) -> FloatArray:
        numeric = frame.loc[:, list(self.numeric_features)].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{self.model_id} prediction features contain a non-finite value")
        standardized = (numeric - self.numeric_means) / self.numeric_scales
        categorical, categorical_names = _categorical_design(frame, self.categories)
        names = (*self.numeric_features, *categorical_names)
        if names != self.design_feature_names:
            raise ValueError(f"{self.model_id} serialized design feature order differs")
        return np.concatenate([standardized, categorical], axis=1)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        linear = self.intercept + self.design(frame) @ self.coefficients
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))

    def as_dict(self) -> dict[str, object]:
        return {
            **SAFETY_FLAGS,
            "model_id": self.model_id,
            "kind": "weighted_l2_logistic_regression",
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "categories": {key: list(values) for key, values in self.categories.items()},
            "numeric_means": self.numeric_means.tolist(),
            "numeric_scales": self.numeric_scales.tolist(),
            "design_feature_names": list(self.design_feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "training_rows": self.training_rows,
            "training_decisions": self.training_decisions,
            "iterations": self.iterations,
            "converged": self.converged,
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        }


def fit_candidate_logistic(
    development: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_id: str,
    weight_column: str = "row_weight",
    decision_column: str = "decision_id",
) -> FrozenCandidateLogistic:
    """Fit the frozen weighted L2/liblinear candidate model on development only."""

    numeric_names = tuple(str(value) for value in numeric_features)
    categorical_names = tuple(str(value) for value in categorical_features)
    required = {
        target_column,
        weight_column,
        decision_column,
        *numeric_names,
        *categorical_names,
    }
    missing = sorted(required.difference(development.columns))
    if missing or development.empty:
        raise ValueError(f"{model_id} training population lacks fields or rows: {missing}")
    numeric = development.loc[:, list(numeric_names)].to_numpy(dtype=np.float64)
    labels = pd.to_numeric(development[target_column], errors="raise").to_numpy(dtype=np.int64)
    weights = pd.to_numeric(development[weight_column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(f"{model_id} training values or weights are invalid")
    if labels.shape != (len(development),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model_id} requires aligned examples from both binary classes")
    means, scales = _weighted_location_scale(numeric, weights)
    standardized = (numeric - means) / scales
    categories = {
        name: tuple(sorted(development[name].astype(str).unique().tolist()))
        for name in categorical_names
    }
    categorical, encoded_names = _categorical_design(development, categories)
    design = np.concatenate([standardized, categorical], axis=1)
    design_names = (*numeric_names, *encoded_names)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260721,
        n_jobs=1,
    )
    estimator.fit(design, labels, sample_weight=weights)
    iterations = int(np.max(estimator.n_iter_))
    if iterations >= 250:
        raise RuntimeError(f"{model_id} failed to converge")
    return FrozenCandidateLogistic(
        model_id=model_id,
        numeric_features=numeric_names,
        categorical_features=categorical_names,
        categories=categories,
        numeric_means=means,
        numeric_scales=scales,
        design_feature_names=design_names,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        training_rows=len(development),
        training_decisions=int(development[decision_column].astype(str).nunique()),
        iterations=iterations,
        converged=True,
    )


def manual_logistic_probabilities(
    serialized: Mapping[str, Any],
    frame: pd.DataFrame,
) -> FloatArray:
    """Independently reconstruct probabilities from serialized model coefficients."""

    numeric_names = tuple(str(value) for value in serialized["numeric_features"])
    categories = {
        str(key): tuple(str(value) for value in values)
        for key, values in cast(Mapping[str, Sequence[object]], serialized["categories"]).items()
    }
    numeric = frame.loc[:, list(numeric_names)].to_numpy(dtype=np.float64)
    means = np.asarray(serialized["numeric_means"], dtype=np.float64)
    scales = np.asarray(serialized["numeric_scales"], dtype=np.float64)
    standardized = (numeric - means) / scales
    categorical, encoded_names = _categorical_design(frame, categories)
    actual_names = (*numeric_names, *encoded_names)
    expected_names = tuple(str(value) for value in serialized["design_feature_names"])
    if actual_names != expected_names:
        raise ValueError("manual reconstruction design order differs")
    design = np.concatenate([standardized, categorical], axis=1)
    coefficients = np.asarray(serialized["coefficients"], dtype=np.float64)
    if coefficients.shape != (design.shape[1],):
        raise ValueError("manual reconstruction coefficient width differs")
    linear = float(serialized["intercept"]) + design @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


@dataclass(frozen=True, slots=True)
class SessionBootstrapDraw:
    """One fixed-seed whole-session bootstrap selection."""

    draw: int
    sampled_sessions: tuple[str, ...]


def session_block_bootstrap_draws(
    sessions: Sequence[str],
    *,
    draws: int = 100,
    seed: int = 20260721,
) -> list[SessionBootstrapDraw]:
    """Resample whole sessions while retaining every nested paired row."""

    if draws != 100:
        raise ValueError("quick screen requires exactly 100 session-block bootstrap draws")
    unique = tuple(sorted(set(str(value) for value in sessions)))
    if not unique:
        raise ValueError("session-block bootstrap requires at least one session")
    rng = np.random.default_rng(seed)
    return [
        SessionBootstrapDraw(
            draw=draw,
            sampled_sessions=tuple(
                str(value) for value in rng.choice(unique, size=len(unique), replace=True)
            ),
        )
        for draw in range(draws)
    ]


def permute_behavioural_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    seed: int,
    draw: int,
    slate_column: str = "slate_id",
    stock_column: str = "symbol",
) -> pd.DataFrame:
    """Permute all ten dimensions as one stock bundle inside each market slate."""

    required = {slate_column, stock_column, *BEHAVIOURAL_DIMENSIONS}
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"behavioural null lacks fields or rows: {missing}")
    output = frame.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    for _, slate in frame.groupby(slate_column, sort=True):
        stocks = tuple(sorted(slate[stock_column].astype(str).unique().tolist()))
        source_bundles: dict[str, FloatArray] = {}
        for stock in stocks:
            values = slate.loc[
                slate[stock_column].astype(str).eq(stock), list(BEHAVIOURAL_DIMENSIONS)
            ].to_numpy(dtype=np.float64)
            if len(values) == 0 or not np.allclose(values, values[0], atol=0.0, rtol=0.0):
                raise ValueError("one stock does not carry one behavioural bundle per slate")
            source_bundles[stock] = values[0]
        permuted_sources = [stocks[int(index)] for index in rng.permutation(len(stocks))]
        for target_stock, source_stock in zip(stocks, permuted_sources, strict=True):
            indices = slate.index[slate[stock_column].astype(str).eq(target_stock)]
            output.loc[indices, list(BEHAVIOURAL_DIMENSIONS)] = source_bundles[source_stock]
    return output


def assert_no_protected_rows(
    timestamps: pd.Series,
    *,
    protected_start: str | pd.Timestamp = "2025-08-23T00:00:00Z",
) -> None:
    """Fail closed if any materialised row reaches the protected date."""

    values = _utc_series(timestamps)
    boundary = pd.Timestamp(protected_start)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    else:
        boundary = boundary.tz_convert("UTC")
    if values.isna().any() or values.ge(boundary).any():
        raise ValueError("protected row materialised")


def decide_screen(evidence: Mapping[str, object]) -> str:
    """Apply the single registered screen decision with blocker precedence."""

    blocker = evidence.get("integrity_blocker")
    if blocker is not None:
        decision = str(blocker)
        if decision not in DECISIONS or not decision.startswith("blocked_"):
            raise ValueError(f"invalid quick-screen blocker: {decision}")
        return decision
    m1 = bool(evidence["m1_passes"])
    m2 = bool(evidence["m2_passes"])
    m2_adverse = bool(evidence["m2_materially_adverse"])
    if m2:
        return "structural_behavioural_interactions_improve_loop_completion"
    if m1 and m2_adverse:
        return "behavioural_main_effects_only"
    if m1:
        return "behavioural_context_improves_loop_completion"
    return "no_behavioural_context_increment"


__all__ = [
    "BEHAVIOURAL_DIMENSIONS",
    "BEHAVIOURAL_NATURAL_KEY",
    "CausalCheckpointFilter",
    "DECISIONS",
    "FrozenCandidateLogistic",
    "INTERACTION_FEATURES",
    "SAFETY_FLAGS",
    "SessionBootstrapDraw",
    "active_prefix_records",
    "apply_interaction_clipping",
    "assert_decision_time_causality",
    "assert_no_protected_rows",
    "assign_candidate_weights",
    "compute_interactions",
    "causal_checkpoint_filter",
    "decide_screen",
    "fit_candidate_logistic",
    "fit_interaction_clipping",
    "manual_logistic_probabilities",
    "normalize_first_event_outcome",
    "permute_behavioural_bundle_within_slates",
    "session_block_bootstrap_draws",
    "target_candidate_rows",
    "verify_behavioural_ledger_reconstruction",
]
