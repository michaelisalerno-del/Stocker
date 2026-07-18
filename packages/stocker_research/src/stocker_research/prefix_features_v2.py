"""Causal compression of semantic loop-prefix state for later research.

The ledger contains only information available at each completed-bar decision.
It deliberately excludes the future loop, future states, completion timing, and
all economic outcomes.  Session boundaries reset the suffix automaton.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.semantic_loop_dictionary_v2 import safety_flags


@dataclass(frozen=True, slots=True)
class _PrefixBinding:
    semantic_loop_id: str
    primitive_loop_id: str | None
    motif_type: str
    repeat_depth: int
    orientation: tuple[int, ...]
    prefix_path: tuple[int, ...]
    progress_states: int
    transitions_remaining: int
    required_next_state: int
    total_transitions: int
    development_count: int
    structural_rate_ratio: float


@dataclass(frozen=True, slots=True)
class CompressedPrefixFeatureBundle:
    """Compact features, reconciliation ledger, and field-level provenance."""

    features: pd.DataFrame
    full_prefixes: pd.DataFrame
    manifest: pd.DataFrame


_PREFIX_FEATURES = (
    "active_prefix_count",
    "active_primitive_prefix_count",
    "active_repeat_prefix_count",
    "active_composite_prefix_count",
    "minimum_transitions_remaining",
    "minimum_bars_remaining_estimate",
    "prefixes_one_transition_away",
    "prefixes_two_transitions_away",
    "prefixes_three_or_more_transitions_away",
    "distinct_required_next_state_count",
    "required_next_state_entropy",
    "dominant_required_next_state",
    "fraction_of_prefixes_agreeing_on_dominant_next_state",
    "highest_support_active_primitive",
    "highest_structural_rate_ratio_active_primitive",
    "highest_prefix_progress",
    "longest_active_prefix",
)

_HISTORY_FEATURES = (
    "previous_completed_primitive_loop",
    "previous_two_completed_primitive_loops",
    "same_primitive_repeat_depth",
    "bars_since_previous_primitive_completion",
    "state_events_since_previous_primitive_completion",
)

_STATE_FEATURE_MAP = {
    "hard_state_legacy": "hard_state_legacy",
    "hard_state_hysteretic": "hard_state_hysteretic",
    "posterior_entropy": "posterior_entropy",
    "top_second_state_margin": "top_second_margin",
    "expected_state_age": "expected_state_age",
    "probability_state_transition_next_bar": "transition_probability_next_bar",
    "bars_remaining_in_session": "bars_remaining_in_session",
    "clock_phase": "clock_phase",
}


def _normalise_paths(value: Any, fallback: Any) -> list[tuple[int, ...]]:
    raw_paths = value if isinstance(value, list | tuple) and value else [fallback]
    if raw_paths and isinstance(raw_paths[0], int):
        raw_paths = [raw_paths]
    output: list[tuple[int, ...]] = []
    for raw_path in raw_paths:
        path = tuple(int(state) for state in raw_path)
        if len(path) < 3 or path[0] != path[-1]:
            raise ValueError(f"prefix registry path is not closed: {path}")
        output.append(path)
    return output


def _row_value(row: pd.Series, field: str, default: Any) -> Any:
    if field not in row:
        return default
    value = row[field]
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _build_lookup(
    primary_dictionary: pd.DataFrame,
    auxiliary_registry: pd.DataFrame | None,
) -> tuple[dict[tuple[int, ...], tuple[_PrefixBinding, ...]], int]:
    if primary_dictionary.empty:
        raise ValueError("selected primary dictionary cannot be empty")
    required = {"semantic_loop_id", "primitive_loop_id", "closed_path", "motif_type"}
    missing = required.difference(primary_dictionary.columns)
    if missing:
        raise ValueError(f"primary dictionary missing prefix fields: {sorted(missing)}")
    frames = [(primary_dictionary, "semantic_loop_id")]
    if auxiliary_registry is not None and not auxiliary_registry.empty:
        frames.append((auxiliary_registry, "semantic_motif_id"))
    lookup: dict[tuple[int, ...], list[_PrefixBinding]] = defaultdict(list)
    maximum_progress = 0
    for frame, id_field in frames:
        for _, row in frame.iterrows():
            semantic_id = str(row[id_field])
            primitive_raw = _row_value(row, "primitive_loop_id", None)
            primitive_id = str(primitive_raw) if primitive_raw is not None else None
            paths = _normalise_paths(
                _row_value(row, "allowed_orientations", []),
                _row_value(row, "closed_path", _row_value(row, "full_path", [])),
            )
            for orientation in paths:
                total_transitions = len(orientation) - 1
                for progress in range(1, len(orientation)):
                    prefix = orientation[:progress]
                    binding = _PrefixBinding(
                        semantic_loop_id=semantic_id,
                        primitive_loop_id=primitive_id,
                        motif_type=str(_row_value(row, "motif_type", "primitive")),
                        repeat_depth=int(_row_value(row, "repeat_depth", 1)),
                        orientation=orientation,
                        prefix_path=prefix,
                        progress_states=progress,
                        transitions_remaining=len(orientation) - progress,
                        required_next_state=orientation[progress],
                        total_transitions=total_transitions,
                        development_count=int(_row_value(row, "development_count", 0)),
                        structural_rate_ratio=float(_row_value(row, "semi_markov_rate_ratio", 0.0)),
                    )
                    lookup[prefix].append(binding)
                    maximum_progress = max(maximum_progress, progress)
    frozen_lookup = {
        prefix: tuple(
            sorted(
                bindings,
                key=lambda item: (
                    item.semantic_loop_id,
                    item.orientation,
                    item.progress_states,
                ),
            )
        )
        for prefix, bindings in lookup.items()
    }
    return frozen_lookup, maximum_progress


def _active_bindings(
    state_events: list[int],
    lookup: dict[tuple[int, ...], tuple[_PrefixBinding, ...]],
    maximum_progress: int,
) -> tuple[_PrefixBinding, ...]:
    active: list[_PrefixBinding] = []
    for width in range(1, min(len(state_events), maximum_progress) + 1):
        active.extend(lookup.get(tuple(state_events[-width:]), ()))
    return tuple(active)


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counter.values()))


def _best_primitive(active: tuple[_PrefixBinding, ...], *, metric: str) -> str | None:
    primitive = [binding for binding in active if binding.motif_type == "primitive"]
    if not primitive:
        return None
    if metric == "support":
        selected = max(
            primitive,
            key=lambda item: (item.development_count, item.semantic_loop_id),
        )
    else:
        selected = max(
            primitive,
            key=lambda item: (item.structural_rate_ratio, item.semantic_loop_id),
        )
    return selected.primitive_loop_id


def _duration_expectations(
    duration_hazard: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if duration_hazard is None:
        return None
    hazard = np.asarray(duration_hazard, dtype=float)
    if hazard.ndim != 2 or hazard.shape[0] < 8 or hazard.shape[1] < 1:
        raise ValueError("duration hazard must contain the frozen state-by-duration surface")
    if not np.isfinite(hazard).all() or (hazard < 0.0).any() or (hazard > 1.0).any():
        raise ValueError("duration hazard contains invalid probabilities")
    mean_duration = np.zeros(hazard.shape[0], dtype=float)
    for state in range(hazard.shape[0]):
        survival = 1.0
        for probability in hazard[state]:
            mean_duration[state] += survival
            survival *= 1.0 - probability
    return hazard, mean_duration


def _binding_bar_estimate(
    binding: _PrefixBinding,
    *,
    current_state: int,
    current_age: int,
    duration_expectations: tuple[np.ndarray, np.ndarray] | None,
) -> float | None:
    if duration_expectations is None:
        return None
    hazard, mean_duration = duration_expectations
    if current_state < 0 or current_state >= hazard.shape[0] or current_age < 1:
        return None
    next_duration_index = current_age
    if next_duration_index >= hazard.shape[1]:
        return None
    survival = 1.0
    expected_current_wait = 0.0
    for probability in hazard[current_state, next_duration_index:]:
        expected_current_wait += survival
        survival *= 1.0 - probability
    intermediate_states = binding.orientation[binding.progress_states : -1]
    return float(
        expected_current_wait + sum(float(mean_duration[state]) for state in intermediate_states)
    )


def _feature_manifest() -> pd.DataFrame:
    records = []
    for feature in _PREFIX_FEATURES:
        duration_estimate = feature == "minimum_bars_remaining_estimate"
        records.append(
            {
                "feature_name": feature,
                "source_artifact": (
                    "corrected duration_model_v2.npz plus completed-bar hard-state history"
                    if duration_estimate
                    else "semantic dictionary plus completed-bar hard-state history"
                ),
                "source_field": (
                    "state-by-duration hazard, hard_run_age, and active prefix route"
                    if duration_estimate
                    else "causal state-event suffix and frozen motif registry"
                ),
                "availability_rule": "computed no later than decision_timestamp",
                "causal_only": True,
            }
        )
    for feature in _HISTORY_FEATURES:
        records.append(
            {
                "feature_name": feature,
                "source_artifact": "uncapped first-event history ledger",
                "source_field": feature,
                "availability_rule": "prior completion available no later than decision_timestamp",
                "causal_only": True,
            }
        )
    for feature, source in _STATE_FEATURE_MAP.items():
        records.append(
            {
                "feature_name": feature,
                "source_artifact": "causal completed-bar decision/state-posterior ledger",
                "source_field": source,
                "availability_rule": "source availability no later than decision_timestamp",
                "causal_only": True,
            }
        )
    return pd.DataFrame.from_records(records)


def build_compressed_prefix_features(
    decisions: pd.DataFrame,
    *,
    primary_dictionary: pd.DataFrame,
    auxiliary_registry: pd.DataFrame | None = None,
    duration_hazard: np.ndarray | None = None,
    include_full_prefixes: bool = True,
) -> CompressedPrefixFeatureBundle:
    """Build causal prefix features and the low-level reconciliation rows."""

    required = {
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "decision_timestamp",
        "hard_state_legacy",
        "hard_run_age",
        *set(_STATE_FEATURE_MAP.values()),
        *set(_HISTORY_FEATURES),
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"completed decisions missing prefix fields: {sorted(missing)}")
    if decisions.empty or decisions["decision_id"].duplicated().any():
        raise ValueError("decisions must be nonempty and unique")
    lookup, maximum_progress = _build_lookup(primary_dictionary, auxiliary_registry)
    duration_expectations = _duration_expectations(duration_hazard)
    ordered = decisions.sort_values(
        ["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    feature_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    eligibility_declared = "structural_event_eligibility" in ordered.columns
    expected_feature_rows = (
        int(ordered["structural_event_eligibility"].fillna(False).astype(bool).sum())
        if eligibility_declared
        else len(ordered)
    )
    for (symbol, session), group in ordered.groupby(["symbol", "session"], sort=False):
        state_events: list[int] = []
        previous_state: int | None = None
        for _, decision in group.iterrows():
            if eligibility_declared and not bool(
                _row_value(decision, "structural_event_eligibility", False)
            ):
                state_events = []
                previous_state = None
                continue
            state = int(decision["hard_state_legacy"])
            if previous_state != state:
                state_events.append(state)
                previous_state = state
            active = _active_bindings(state_events, lookup, maximum_progress)
            next_states = Counter(binding.required_next_state for binding in active)
            dominant = (
                min(
                    (
                        state
                        for state, count in next_states.items()
                        if count == max(next_states.values())
                    ),
                    default=None,
                )
                if next_states
                else None
            )
            minimum_remaining = min(
                (binding.transitions_remaining for binding in active), default=None
            )
            bar_estimates = [
                estimate
                for binding in active
                if (
                    estimate := _binding_bar_estimate(
                        binding,
                        current_state=state,
                        current_age=int(decision["hard_run_age"]),
                        duration_expectations=duration_expectations,
                    )
                )
                is not None
            ]
            minimum_bar_estimate = min(bar_estimates, default=None)
            decision_id = str(decision["decision_id"])
            if include_full_prefixes:
                prefix_rows.extend(
                    {
                        "decision_id": decision_id,
                        "symbol": str(symbol),
                        "session": str(session),
                        "decision_timestamp": pd.Timestamp(decision["decision_timestamp"]),
                        "semantic_loop_id": binding.semantic_loop_id,
                        "primitive_loop_id": binding.primitive_loop_id,
                        "motif_type": binding.motif_type,
                        "repeat_depth": binding.repeat_depth,
                        "orientation": list(binding.orientation),
                        "prefix_path": list(binding.prefix_path),
                        "progress_states": binding.progress_states,
                        "transitions_remaining": binding.transitions_remaining,
                        "required_next_state": binding.required_next_state,
                    }
                    for binding in active
                )
            primitive_count = sum(binding.motif_type == "primitive" for binding in active)
            repeat_count = sum(binding.motif_type == "repeat" for binding in active)
            composite_count = sum(binding.motif_type == "composite" for binding in active)
            common = {
                "decision_id": decision_id,
                "symbol": str(symbol),
                "session": str(session),
                "decision_timestamp": pd.Timestamp(decision["decision_timestamp"]),
                "bar_ordinal": int(decision["bar_ordinal"]),
                "feature_available_timestamp": pd.Timestamp(decision["decision_timestamp"]),
                "active_prefix_count": len(active),
                "active_primitive_prefix_count": primitive_count,
                "active_repeat_prefix_count": repeat_count,
                "active_composite_prefix_count": composite_count,
                "minimum_transitions_remaining": minimum_remaining,
                "minimum_bars_remaining_estimate": minimum_bar_estimate,
                "prefixes_one_transition_away": sum(
                    binding.transitions_remaining == 1 for binding in active
                ),
                "prefixes_two_transitions_away": sum(
                    binding.transitions_remaining == 2 for binding in active
                ),
                "prefixes_three_or_more_transitions_away": sum(
                    binding.transitions_remaining >= 3 for binding in active
                ),
                "distinct_required_next_state_count": len(next_states),
                "required_next_state_entropy": _entropy(next_states),
                "dominant_required_next_state": dominant,
                "fraction_of_prefixes_agreeing_on_dominant_next_state": (
                    next_states[dominant] / len(active) if dominant is not None and active else 0.0
                ),
                "highest_support_active_primitive": _best_primitive(active, metric="support"),
                "highest_structural_rate_ratio_active_primitive": _best_primitive(
                    active, metric="rate_ratio"
                ),
                "highest_prefix_progress": max(
                    (
                        (binding.progress_states - 1) / binding.total_transitions
                        for binding in active
                    ),
                    default=0.0,
                ),
                "longest_active_prefix": max(
                    (binding.progress_states for binding in active), default=0
                ),
                **{feature: _row_value(decision, feature, None) for feature in _HISTORY_FEATURES},
                **{
                    feature: _row_value(decision, source, None)
                    for feature, source in _STATE_FEATURE_MAP.items()
                },
                "run_id": _row_value(decision, "run_id_v2", _row_value(decision, "run_id", None)),
                "git_sha": _row_value(decision, "git_sha", None),
                "contract_hash": _row_value(decision, "contract_hash", None),
                "data_snapshot_hash": _row_value(decision, "data_snapshot_hash", None),
                "dictionary_version": _row_value(decision, "dictionary_version", None),
                "dictionary_hash": _row_value(decision, "dictionary_hash", None),
                "state_model_version": _row_value(decision, "state_model_version", None),
                "semantic_loop_id": None,
                "primitive_loop_id": None,
                "event_timestamp": None,
                "source_artifact": "causal_completed_bar_decisions.parquet",
                "source_hash": _row_value(decision, "source_artifact_hash", None),
                **safety_flags(),
            }
            feature_rows.append(common)
    features = pd.DataFrame.from_records(feature_rows)
    full_prefixes = pd.DataFrame.from_records(
        prefix_rows,
        columns=[
            "decision_id",
            "symbol",
            "session",
            "decision_timestamp",
            "semantic_loop_id",
            "primitive_loop_id",
            "motif_type",
            "repeat_depth",
            "orientation",
            "prefix_path",
            "progress_states",
            "transitions_remaining",
            "required_next_state",
        ],
    )
    if len(features) != expected_feature_rows:
        raise AssertionError("prefix compression is not one-to-one with eligible decisions")
    if include_full_prefixes:
        reconciled = (
            full_prefixes.groupby("decision_id").size()
            if not full_prefixes.empty
            else pd.Series(dtype=int)
        )
        expected = features.set_index("decision_id")["active_prefix_count"]
        if not expected.eq(reconciled.reindex(expected.index, fill_value=0)).all():
            raise AssertionError("compressed and full prefix counts differ")
    return CompressedPrefixFeatureBundle(
        features=features,
        full_prefixes=full_prefixes,
        manifest=_feature_manifest(),
    )


__all__ = ["CompressedPrefixFeatureBundle", "build_compressed_prefix_features"]
