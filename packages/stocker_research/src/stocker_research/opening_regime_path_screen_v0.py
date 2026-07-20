"""Primitives for Opening Regime-Path Direction Screen V0.

The module is intentionally research-only.  It contains deterministic time,
outcome, topology, fixed-linear-model, resampling, permutation, and decision
helpers; it has no broker, order, account, position, sizing, or runtime surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression

RESEARCH_ONLY = True
FEASIBILITY_SCREEN = True
REPRESENTATION_SPECIFIC = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_INTEGRATION_REQUIRED = False
STRATEGY_PROMOTION = False
PRODUCTION_RUNTIME_MODIFIED = False

SAFETY_FLAGS: dict[str, object] = {
    "research_only": RESEARCH_ONLY,
    "feasibility_screen": FEASIBILITY_SCREEN,
    "representation_specific": REPRESENTATION_SPECIFIC,
    "execution_enabled": EXECUTION_ENABLED,
    "order_placement": ORDER_PLACEMENT,
    "broker_integration_required": BROKER_INTEGRATION_REQUIRED,
    "strategy_promotion": STRATEGY_PROMOTION,
    "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
}

DECISION_ORDINALS = (6, 12)
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")

FORBIDDEN_FEATURE_EXACT = {
    "future_state",
    "future_run_duration",
    "future_closure",
    "future_loop",
    "payoff_history",
    "profitable_loop_label",
    "exact_loop_id",
    "outcome_selected_state_identity",
}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "future_",
    "payoff",
    "profitable_loop",
    "exact_loop",
    "economic_history",
    "outcome_selected",
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DECISIONS = (
    "opening_regime_path_adds_movement_and_direction",
    "opening_regime_path_adds_movement_only",
    "opening_regime_path_adds_direction_only",
    "opening_loop_regime_interaction_only",
    "opening_structure_no_increment_over_price",
    "blocked_missing_frozen_regime_inputs",
    "blocked_protected_boundary_failure",
    "blocked_chronology_or_leakage_failure",
    "blocked_insufficient_opening_path_support",
    "blocked_model_convergence_failure",
    "blocked_quick_screen_resource_limit",
    "blocked_reproducibility_or_audit_failure",
)


def decision_bar_start_ordinal(decision_ordinal: int) -> int:
    """Map completed-bar count to the repository's zero-based bar-start ordinal."""

    if decision_ordinal not in DECISION_ORDINALS:
        raise ValueError("V0 permits exactly decision ordinals 6 and 12")
    return decision_ordinal - 1


def decision_time_local(decision_ordinal: int) -> time:
    """Return the exact New York decision time after the bar has completed."""

    return {6: time(10, 0), 12: time(10, 30)}[decision_ordinal]


@dataclass(frozen=True, slots=True)
class DecisionHistory:
    """Validated causal opening history ending at the completed decision bar."""

    bar_start_ordinal: int
    decision_timestamp: pd.Timestamp
    feature_available_timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class OutcomeAnchor:
    """The fixed delayed entry and regular-session terminal."""

    entry_bar_ordinal: int
    delayed_entry_open: float
    terminal_bar_ordinal: int
    terminal_close: float


def _utc_timestamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(str(value))
    if result.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return result.tz_convert("UTC")


def reject_invalid_decision_history(
    session_frame: pd.DataFrame,
    *,
    decision_ordinal: int,
) -> DecisionHistory:
    """Validate the exact causal opening tape and completed decision-bar timing."""

    required = {
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "close",
        "segment_id",
        "session",
        "session_source_complete",
        "source_data_error_in_session",
    }
    missing = sorted(required.difference(session_frame.columns))
    if missing:
        raise ValueError(f"decision history missing columns: {missing}")
    origin_ordinal = decision_bar_start_ordinal(decision_ordinal)
    frame = session_frame.copy()
    frame["bar_ordinal"] = pd.to_numeric(frame["bar_ordinal"], errors="raise").astype(int)
    opening = frame.loc[frame["bar_ordinal"].between(0, origin_ordinal)].sort_values(
        "bar_ordinal", kind="mergesort"
    )
    if opening["session"].astype(str).nunique() != 1:
        raise ValueError("session boundary in opening history")
    if opening["bar_ordinal"].tolist() != list(range(origin_ordinal + 1)):
        raise ValueError("source gap in opening history")
    if opening["segment_id"].astype(str).nunique() != 1:
        raise ValueError("source gap in opening history")
    if not opening["session_source_complete"].astype(bool).all():
        raise ValueError("source gap or incomplete session")
    if opening["source_data_error_in_session"].astype(bool).any():
        raise ValueError("quarantined source QA failure")
    prices = opening[["open", "close"]].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("opening history contains invalid prices")

    starts = pd.to_datetime(opening["bar_start_timestamp"], utc=True, errors="raise")
    completions = pd.to_datetime(opening["bar_complete_timestamp"], utc=True, errors="raise")
    if starts.ge(PROTECTED_START).any() or completions.ge(PROTECTED_START).any():
        raise ValueError("protected timestamp materialised")
    expected_starts = pd.date_range(
        starts.iloc[0], periods=origin_ordinal + 1, freq="5min", tz="UTC"
    )
    if not starts.reset_index(drop=True).equals(pd.Series(expected_starts)):
        raise ValueError("source gap or non-grid opening history")
    if not completions.reset_index(drop=True).equals(
        (starts + pd.Timedelta(minutes=5)).reset_index(drop=True)
    ):
        raise ValueError("decision bar is not completed")

    origin = opening.iloc[-1]
    start = _utc_timestamp(origin["bar_start_timestamp"])
    available = _utc_timestamp(origin["bar_complete_timestamp"])
    local_available = available.tz_convert("America/New_York")
    if local_available.time().replace(tzinfo=None) != decision_time_local(decision_ordinal):
        raise ValueError("decision checkpoint time does not match completed bar")
    return DecisionHistory(origin_ordinal, start, available)


def delayed_entry_and_terminal(
    session_frame: pd.DataFrame,
    *,
    decision_ordinal: int,
) -> OutcomeAnchor:
    """Extract open(t+2) and the exact final regular-session close."""

    reject_invalid_decision_history(session_frame, decision_ordinal=decision_ordinal)
    frame = session_frame.copy()
    frame["bar_ordinal"] = pd.to_numeric(frame["bar_ordinal"], errors="raise").astype(int)
    if frame["session"].astype(str).nunique() != 1:
        raise ValueError("session boundary in outcome tape")
    expected_values = pd.to_numeric(frame["expected_session_bars"], errors="raise").unique()
    if len(expected_values) != 1:
        raise ValueError("ambiguous session terminal")
    terminal_ordinal = int(expected_values[0]) - 1
    entry_ordinal = decision_bar_start_ordinal(decision_ordinal) + 2
    by_ordinal = frame.set_index("bar_ordinal")
    if not by_ordinal.index.is_unique:
        raise ValueError("outcome tape contains duplicate bar ordinals")
    if entry_ordinal not in by_ordinal.index or terminal_ordinal not in by_ordinal.index:
        raise ValueError("incomplete remaining outcome")
    entry = float(cast(Any, by_ordinal.loc[entry_ordinal, "open"]))
    terminal = float(cast(Any, by_ordinal.loc[terminal_ordinal, "close"]))
    if not np.isfinite([entry, terminal]).all() or entry <= 0.0 or terminal <= 0.0:
        raise ValueError("outcome anchor contains invalid prices")
    return OutcomeAnchor(entry_ordinal, entry, terminal_ordinal, terminal)


def cohort_relative_returns_bps(raw_return_bps: Sequence[float]) -> tuple[FloatArray, FloatArray]:
    """Return leave-one-stock-out cohort medians and residual returns in bps."""

    values = np.asarray(raw_return_bps, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("cohort-relative returns need at least two finite rows")
    medians = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        medians[index] = float(np.median(np.delete(values, index)))
    return np.asarray(values - medians, dtype=np.float64), medians


def assert_allowed_feature_names(feature_names: Sequence[str]) -> None:
    """Reject future, payoff-history, selected-loop, and outcome-selected fields."""

    invalid = []
    for feature in feature_names:
        lowered = feature.lower()
        if lowered in FORBIDDEN_FEATURE_EXACT or any(
            fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS
        ):
            invalid.append(feature)
    if invalid:
        raise ValueError(f"forbidden opening-screen feature(s): {sorted(invalid)}")


def _state_runs(states: Sequence[int]) -> tuple[list[int], list[int]]:
    values = [int(value) for value in states]
    if not values:
        raise ValueError("opening path must contain at least one completed bar")
    run_states = [values[0]]
    run_durations = [1]
    for value in values[1:]:
        if value == run_states[-1]:
            run_durations[-1] += 1
        else:
            run_states.append(value)
            run_durations.append(1)
    return run_states, run_durations


def opening_path_features(states: Sequence[int]) -> dict[str, float]:
    """Reconstruct preregistered opening topology from completed causal states."""

    values = [int(value) for value in states]
    run_states, run_durations = _state_runs(values)
    transitions = len(run_states) - 1
    revisits = sum(run_states[index] in run_states[:index] for index in range(1, len(run_states)))
    returns_to_open = sum(value == run_states[0] for value in run_states[1:])
    two_state_closures = sum(
        run_states[index] == run_states[index + 2] and run_states[index] != run_states[index + 1]
        for index in range(max(0, len(run_states) - 2))
    )
    three_state_closures = sum(
        run_states[index] == run_states[index + 3] and len(set(run_states[index : index + 3])) == 3
        for index in range(max(0, len(run_states) - 3))
    )
    recent_two = bool(
        len(run_states) >= 3
        and run_states[-3] == run_states[-1]
        and run_states[-3] != run_states[-2]
    )
    recent_three = bool(
        len(run_states) >= 4
        and run_states[-4] == run_states[-1]
        and len(set(run_states[-4:-1])) == 3
    )
    alternation_denominator = max(1, len(run_states) - 2)
    alternations = sum(
        run_states[index] == run_states[index + 2] for index in range(max(0, len(run_states) - 2))
    )
    completed_durations = run_durations[:-1]
    if completed_durations:
        mean_completed = float(np.mean(completed_durations))
        maximum_completed = float(max(completed_durations))
        minimum_completed = float(min(completed_durations))
        recent_completed = float(completed_durations[-1])
    else:
        mean_completed = maximum_completed = minimum_completed = recent_completed = 0.0
    _, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    occupancy = counts.astype(np.float64) / float(len(values))
    entropy = float(-np.sum(occupancy * np.log(occupancy)))
    current_age = float(run_durations[-1])
    return {
        "opening_transition_count": float(transitions),
        "opening_unique_state_count": float(len(set(run_states))),
        "opening_state_revisit_count": float(revisits),
        "opening_return_to_origin_count": float(returns_to_open),
        "opening_two_state_closure_count": float(two_state_closures),
        "opening_three_state_closure_count": float(three_state_closures),
        "opening_any_short_closure": float(two_state_closures + three_state_closures > 0),
        "opening_most_recent_path_was_closure": float(recent_two or recent_three),
        "opening_alternation_ratio": float(alternations / alternation_denominator),
        "opening_transition_rate": float(transitions / len(values)),
        "opening_mean_completed_run_duration": mean_completed,
        "opening_maximum_completed_run_duration": maximum_completed,
        "opening_minimum_completed_run_duration": minimum_completed,
        "opening_most_recent_completed_run_duration": recent_completed,
        "opening_time_since_latest_transition": current_age - 1.0
        if transitions
        else float(len(values)),
        "opening_state_occupancy_entropy": entropy,
        "opening_largest_state_occupancy_fraction": float(occupancy.max()),
        "current_state_age": current_age,
    }


def current_regime_features(
    states: Sequence[int],
    posterior: Sequence[float],
    *,
    state_count: int = 8,
) -> dict[str, float]:
    """Build hard-state one-hots and the current causal posterior summary."""

    run_states, run_durations = _state_runs(states)
    current = run_states[-1]
    previous = run_states[-2] if len(run_states) >= 2 else None
    if current < 0 or current >= state_count or (previous is not None and previous >= state_count):
        raise ValueError("state is outside the frozen representation")
    probabilities = np.asarray(posterior, dtype=np.float64)
    if probabilities.shape != (state_count,) or not np.isfinite(probabilities).all():
        raise ValueError("posterior must be one complete finite frozen-state vector")
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise ValueError("posterior is not normalized")
    safe = probabilities[probabilities > 0.0]
    output: dict[str, float] = {}
    for state in range(state_count):
        output[f"current_state_{state}"] = float(current == state)
        output[f"previous_completed_state_{state}"] = float(previous == state)
        output[f"posterior_state_{state}"] = float(probabilities[state])
    output.update(
        {
            "maximum_posterior_probability": float(probabilities.max()),
            "posterior_entropy": float(-np.sum(safe * np.log(safe))),
            "current_state_age": float(run_durations[-1]),
            "opening_state_equals_current": float(run_states[0] == current),
        }
    )
    return output


def interaction_features(
    current_state: int,
    topology: dict[str, float],
    *,
    state_count: int = 8,
) -> dict[str, float]:
    """Construct exactly the four registered current-state interaction families."""

    if current_state < 0 or current_state >= state_count:
        raise ValueError("current state is outside the frozen representation")
    sources = {
        "any_short_closure": topology["opening_any_short_closure"],
        "opening_return_to_origin_count": topology["opening_return_to_origin_count"],
        "transition_rate": topology["opening_transition_rate"],
        "current_state_age": topology["current_state_age"],
    }
    output: dict[str, float] = {}
    for state in range(state_count):
        active = float(state == current_state)
        for label, value in sources.items():
            output[f"current_state_{state}_x_{label}"] = active * float(value)
    return output


def development_movement_thresholds(
    frame: pd.DataFrame,
    *,
    target_column: str = "residual_remaining_return_bps",
) -> dict[int, float]:
    """Freeze checkpoint-specific absolute-movement q75 values on 2024 only."""

    development = frame.loc[pd.to_numeric(frame["year"], errors="raise").eq(2024)]
    if development.empty:
        raise ValueError("movement thresholds require 2024 development rows")
    output: dict[int, float] = {}
    for ordinal in DECISION_ORDINALS:
        values = (
            pd.to_numeric(
                development.loc[development["decision_ordinal"].eq(ordinal), target_column],
                errors="coerce",
            )
            .abs()
            .dropna()
        )
        if values.empty:
            raise ValueError(f"no development movement rows at checkpoint {ordinal}")
        output[ordinal] = float(values.quantile(0.75, interpolation="linear"))
    return output


def equal_slate_weights(slate_ids: pd.Series) -> FloatArray:
    """Give every represented simultaneous slate total model weight one."""

    values = slate_ids.astype(str).reset_index(drop=True)
    if values.empty or values.isna().any():
        raise ValueError("slate weights require complete slate identifiers")
    sizes = values.groupby(values, sort=True).transform("size").to_numpy(dtype=np.float64)
    weights = np.asarray(1.0 / sizes, dtype=np.float64)
    totals = pd.Series(weights).groupby(values, sort=True).sum().to_numpy(dtype=np.float64)
    if not np.allclose(totals, 1.0, atol=1e-12):
        raise AssertionError("slate weights do not sum to one")
    return weights


@dataclass(frozen=True, slots=True)
class FrozenLogisticModel:
    """JSON-serializable standardized fixed L2 logistic model."""

    model_id: str
    feature_names: tuple[str, ...]
    means: FloatArray
    scales: FloatArray
    coefficients: FloatArray
    intercept: float
    training_rows: int
    training_slates: int
    iterations: int
    converged: bool

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        values = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.model_id} prediction features are not finite")
        linear = self.intercept + ((values - self.means) / self.scales) @ self.coefficients
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": "logistic",
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "training_rows": self.training_rows,
            "training_slates": self.training_slates,
            "iterations": self.iterations,
            "converged": self.converged,
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        }


def fit_fixed_logistic(
    frame: pd.DataFrame,
    target: Sequence[int] | pd.Series,
    *,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
) -> FrozenLogisticModel:
    """Fit the deterministic C=1 L2 liblinear model with equal-slate weights."""

    names = tuple(features)
    assert_allowed_feature_names(names)
    values = frame.loc[:, list(names)].to_numpy(dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_id} training features are not finite")
    if labels.shape != (len(frame),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model_id} requires both aligned binary classes")
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    design = (values - means) / scales
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    )
    estimator.fit(
        design,
        labels,
        sample_weight=equal_slate_weights(frame[slate_column]),
    )
    iterations = int(np.max(estimator.n_iter_))
    if iterations >= 250:
        raise RuntimeError(f"{model_id} failed to converge")
    return FrozenLogisticModel(
        model_id=model_id,
        feature_names=names,
        means=means,
        scales=scales,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        training_rows=len(frame),
        training_slates=int(frame[slate_column].astype(str).nunique()),
        iterations=iterations,
        converged=True,
    )


def manual_logistic_prediction(model: Mapping[str, Any], frame: pd.DataFrame) -> FloatArray:
    """Reconstruct fixed logistic probabilities from serialized parameters."""

    names = [str(value) for value in model["feature_names"]]
    values = frame.loc[:, names].to_numpy(dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (
        values.shape[1] != len(means)
        or means.shape != scales.shape
        or means.shape != coefficients.shape
    ):
        raise ValueError("serialized logistic dimensions do not align")
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


@dataclass(frozen=True, slots=True)
class SessionBootstrapDraw:
    """One whole-session bootstrap selection."""

    draw: int
    sampled_sessions: tuple[str, ...]


def session_block_bootstrap_draws(
    sessions: Sequence[str],
    *,
    draws: int = 300,
    seed: int = 20260720,
) -> list[SessionBootstrapDraw]:
    """Sample entire sessions, retaining both clocks, all stocks, and all models."""

    unique = tuple(sorted(set(str(value) for value in sessions)))
    if not unique:
        raise ValueError("session bootstrap needs at least one session")
    if draws != 300:
        raise ValueError("V0 requires exactly 300 bootstrap draws")
    rng = np.random.default_rng(seed)
    return [
        SessionBootstrapDraw(
            draw=draw,
            sampled_sessions=tuple(
                str(value) for value in rng.choice(unique, len(unique), replace=True)
            ),
        )
        for draw in range(draws)
    ]


def permute_structural_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    structural_columns: Sequence[str],
    seed: int,
    draw: int,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute one complete structural bundle among stocks inside every slate."""

    columns = tuple(structural_columns)
    if not columns or slate_column not in frame:
        raise ValueError("structural permutation needs bundle columns and slate identifiers")
    assert_allowed_feature_names(columns)
    output = frame.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    for indices in frame.groupby(slate_column, sort=True).groups.values():
        positions = np.asarray(list(indices), dtype=np.int64)
        source = frame.loc[positions, list(columns)].to_numpy(copy=True)
        output.loc[positions, list(columns)] = source[rng.permutation(len(positions))]
    return output


def decide_screen(evidence: Mapping[str, Any]) -> str:
    """Apply the registered final-classification precedence exactly once."""

    blocker = evidence.get("integrity_blocker")
    if blocker is not None:
        label = str(blocker)
        if label not in DECISIONS or not label.startswith("blocked_"):
            raise ValueError(f"invalid opening-screen blocker: {label}")
        return label
    movement = bool(evidence["movement_increment_passes"])
    direction = bool(evidence["direction_increment_passes"])
    interaction = bool(evidence["interaction_increment_passes"])
    if movement and direction:
        return "opening_regime_path_adds_movement_and_direction"
    if movement:
        return "opening_regime_path_adds_movement_only"
    if direction:
        return "opening_regime_path_adds_direction_only"
    if interaction:
        return "opening_loop_regime_interaction_only"
    return "opening_structure_no_increment_over_price"


__all__ = [
    "BROKER_INTEGRATION_REQUIRED",
    "DECISION_ORDINALS",
    "DECISIONS",
    "DecisionHistory",
    "EXECUTION_ENABLED",
    "FEASIBILITY_SCREEN",
    "ORDER_PLACEMENT",
    "OutcomeAnchor",
    "PRODUCTION_RUNTIME_MODIFIED",
    "REPRESENTATION_SPECIFIC",
    "RESEARCH_ONLY",
    "SAFETY_FLAGS",
    "SessionBootstrapDraw",
    "STRATEGY_PROMOTION",
    "assert_allowed_feature_names",
    "cohort_relative_returns_bps",
    "current_regime_features",
    "decide_screen",
    "decision_bar_start_ordinal",
    "decision_time_local",
    "delayed_entry_and_terminal",
    "development_movement_thresholds",
    "equal_slate_weights",
    "fit_fixed_logistic",
    "interaction_features",
    "manual_logistic_prediction",
    "opening_path_features",
    "permute_structural_bundle_within_slates",
    "reject_invalid_decision_history",
    "session_block_bootstrap_draws",
]
