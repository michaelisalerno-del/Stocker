"""Causal state-posterior, provenance, and completed-bar decision export V2.

The filter is forward-only.  It exports the complete state-by-age posterior
after each completed source bar and never invokes retrospective smoothing or
Viterbi decoding.  Provider timestamps are treated as bar starts; completion
availability is supplied explicitly as ``bar_start + bar_duration``.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.loop_dictionary_v2 import LoopDictionary
from stocker_research.loop_events_v2 import safety_flags

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


@dataclass(frozen=True, slots=True)
class HysteresisConfig:
    """Frozen causal switch rule; it uses no future posterior row."""

    switch_probability: float = 0.55
    switch_margin: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.switch_probability <= 1.0:
            raise ValueError("switch_probability must be in [0, 1]")
        if not 0.0 <= self.switch_margin <= 1.0:
            raise ValueError("switch_margin must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CausalStateExport:
    """All causal posterior surfaces aligned one-to-one with input bars."""

    state_probabilities: np.ndarray
    state_age_probabilities: np.ndarray
    posterior_entropy: np.ndarray
    top_state: np.ndarray
    second_state: np.ndarray
    top_second_margin: np.ndarray
    expected_state_age: np.ndarray
    probability_state_persists_next_bar: np.ndarray
    probability_state_transitions_next_bar: np.ndarray
    next_state_probabilities: np.ndarray
    hard_map_state: np.ndarray
    hard_map_run_age: np.ndarray
    source_timestamps: tuple[datetime, ...]
    available_timestamps: tuple[datetime, ...]

    def validate(self) -> None:
        rows = len(self.state_probabilities)
        if self.state_age_probabilities.shape[0] != rows:
            raise AssertionError("state and state-age posterior rows differ")
        if not np.allclose(self.state_probabilities.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("state probabilities do not normalize")
        if not np.allclose(self.state_age_probabilities.sum(axis=(1, 2)), 1.0, atol=1e-12):
            raise AssertionError("state-age probabilities do not normalize")
        if not np.allclose(self.next_state_probabilities.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("next-state probabilities do not normalize")
        if any(
            available < source
            for source, available in zip(
                self.source_timestamps, self.available_timestamps, strict=True
            )
        ):
            raise AssertionError("posterior availability precedes its source bar")


@dataclass(frozen=True, slots=True)
class SoftPrefixSnapshot:
    """Diagnostic probability mass; it is never a hard structural completion."""

    highest_prefix_probability: float
    highest_completion_probability: float
    completion_probabilities: tuple[tuple[str, str, float], ...]
    hard_completion: bool = False
    approximation: str = "marginal_forward_diagnostic_not_a_hard_event"


def expand_duration_hazard_v2(
    model: Mapping[str, np.ndarray],
    *,
    maximum_age: int,
    tail_window: int = 6,
) -> dict[str, np.ndarray]:
    """Replace a forced terminal hazard with an explicit regular-session tail.

    Historical state models used a final age bucket whose hazard was forced to
    one.  V2 preserves every nonterminal fitted hazard, estimates one frozen
    geometric tail hazard from the immediately preceding ages, and extends it
    through ``maximum_age``.  The final cell remains a survival tail bucket;
    it is deliberately not forced to exit.
    """

    hazard = np.asarray(model["duration_hazard"], dtype=float)
    if hazard.ndim != 2 or hazard.shape[1] < 2:
        raise ValueError("duration_hazard must contain at least two ages")
    if maximum_age < hazard.shape[1]:
        raise ValueError("maximum_age cannot truncate the frozen duration support")
    if tail_window <= 0:
        raise ValueError("tail_window must be positive")
    if not np.isfinite(hazard).all() or np.any((hazard < 0.0) | (hazard > 1.0)):
        raise ValueError("duration hazards must be finite probabilities")

    terminal_index = hazard.shape[1] - 1
    window_start = max(0, terminal_index - tail_window)
    tail = np.mean(hazard[:, window_start:terminal_index], axis=1)
    tail = np.clip(tail, 1e-6, 1.0 - 1e-6)
    expanded = np.empty((hazard.shape[0], maximum_age), dtype=float)
    expanded[:, :terminal_index] = hazard[:, :terminal_index]
    expanded[:, terminal_index:] = tail[:, None]
    output = {key: np.asarray(value).copy() for key, value in model.items()}
    output["duration_hazard"] = expanded
    return output


def propagate_state_age_posterior(
    posterior: np.ndarray, model: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Advance one bar while conserving state-by-age probability mass."""

    alpha = np.asarray(posterior, dtype=float)
    hazard = np.asarray(model["duration_hazard"], dtype=float)
    transitions = np.asarray(model["transitions"], dtype=float)
    if alpha.shape != hazard.shape:
        raise ValueError("posterior and duration hazard shapes differ")
    if transitions.shape != (alpha.shape[0], alpha.shape[0]):
        raise ValueError("transition matrix shape differs from state count")
    stay = alpha * (1.0 - hazard)
    predicted = np.zeros_like(alpha)
    predicted[:, 1:] += stay[:, :-1]
    predicted[:, -1] += stay[:, -1]
    exit_mass = np.sum(alpha * hazard, axis=1)
    predicted[:, 0] += exit_mass @ transitions
    total = float(predicted.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise AssertionError("semi-Markov propagation lost probability mass")
    return predicted / total


def causal_semimarkov_filter_v2(
    log_emissions: np.ndarray,
    *,
    session_groups: Sequence[np.ndarray],
    model: Mapping[str, np.ndarray],
    bar_start_timestamps: Sequence[datetime],
    bar_duration: timedelta,
) -> CausalStateExport:
    """Run the frozen forward recursion and retain every posterior coordinate."""

    emissions = np.asarray(log_emissions, dtype=float)
    hazard = np.asarray(model["duration_hazard"], dtype=float)
    initial = np.asarray(model["initial"], dtype=float)
    if emissions.ndim != 2 or hazard.ndim != 2:
        raise ValueError("emissions and duration_hazard must be two-dimensional")
    rows, state_count = emissions.shape
    if hazard.shape[0] != state_count or initial.shape != (state_count,):
        raise ValueError("state model dimensions differ")
    if len(bar_start_timestamps) != rows:
        raise ValueError("bar timestamp count differs from emission rows")
    if bar_duration <= timedelta(0):
        raise ValueError("bar_duration must be positive")
    if not np.isfinite(emissions).all():
        raise ValueError("log emissions contain a nonfinite value")

    maximum_age = hazard.shape[1]
    state_age = np.zeros((rows, state_count, maximum_age), dtype=float)
    state_probability = np.zeros((rows, state_count), dtype=float)
    next_probability = np.zeros((rows, state_count), dtype=float)
    entropy = np.zeros(rows, dtype=float)
    top = np.zeros(rows, dtype=np.int16)
    second = np.zeros(rows, dtype=np.int16)
    margin = np.zeros(rows, dtype=float)
    expected_age = np.zeros(rows, dtype=float)
    persists = np.zeros(rows, dtype=float)
    transitions_next = np.zeros(rows, dtype=float)
    hard_age = np.zeros(rows, dtype=np.int16)
    assigned = np.zeros(rows, dtype=bool)
    age_values = np.arange(1, maximum_age + 1, dtype=float)[None, :]

    for raw_positions in session_groups:
        positions = np.asarray(raw_positions, dtype=int)
        if len(positions) == 0:
            continue
        if np.any(np.diff(positions) <= 0):
            raise ValueError("session positions must be strictly increasing")
        position_timestamps = [bar_start_timestamps[int(position)] for position in positions]
        if any(
            later <= earlier
            for earlier, later in zip(
                position_timestamps[:-1], position_timestamps[1:], strict=True
            )
        ):
            raise ValueError("session source timestamps must be strictly increasing")
        alpha: np.ndarray | None = None
        previous_state = -1
        current_hard_age = 0
        for position in positions:
            if position < 0 or position >= rows or assigned[position]:
                raise ValueError("session groups overlap or reference an invalid row")
            if alpha is None:
                prior = np.zeros_like(hazard)
                prior[:, 0] = initial / initial.sum()
            else:
                prior = propagate_state_age_posterior(alpha, model)
            emission = emissions[position]
            likelihood = np.exp(emission - np.max(emission))
            posterior = prior * likelihood[:, None]
            total = float(posterior.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise AssertionError("semi-Markov posterior underflow")
            alpha = posterior / total
            probabilities = alpha.sum(axis=1)
            order = np.argsort(-probabilities, kind="stable")
            hard_state = int(order[0])
            current_hard_age = current_hard_age + 1 if hard_state == previous_state else 1
            previous_state = hard_state

            predicted = propagate_state_age_posterior(alpha, model)
            exit_probability = float(np.sum(alpha * hazard))
            state_age[position] = alpha
            state_probability[position] = probabilities
            next_probability[position] = predicted.sum(axis=1)
            entropy[position] = float(
                -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)))
            )
            top[position] = hard_state
            second[position] = int(order[1])
            margin[position] = float(probabilities[order[0]] - probabilities[order[1]])
            expected_age[position] = float(np.sum(alpha * age_values))
            transitions_next[position] = exit_probability
            persists[position] = 1.0 - exit_probability
            hard_age[position] = min(current_hard_age, maximum_age)
            assigned[position] = True
    if not assigned.all():
        raise AssertionError("causal filter left an input row unassigned")

    source_timestamps = tuple(bar_start_timestamps)
    available_timestamps = tuple(timestamp + bar_duration for timestamp in source_timestamps)
    result = CausalStateExport(
        state_probabilities=state_probability,
        state_age_probabilities=state_age,
        posterior_entropy=entropy,
        top_state=top,
        second_state=second,
        top_second_margin=margin,
        expected_state_age=expected_age,
        probability_state_persists_next_bar=persists,
        probability_state_transitions_next_bar=transitions_next,
        next_state_probabilities=next_probability,
        hard_map_state=top.copy(),
        hard_map_run_age=hard_age,
        source_timestamps=source_timestamps,
        available_timestamps=available_timestamps,
    )
    result.validate()
    return result


def hysteretic_states(state_probabilities: np.ndarray, *, config: HysteresisConfig) -> np.ndarray:
    """Create a causal stabilized state using only current and prior rows."""

    probabilities = np.asarray(state_probabilities, dtype=float)
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("state_probabilities must be a nonempty matrix")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("state probabilities do not normalize")
    output = np.zeros(len(probabilities), dtype=np.int16)
    current = int(np.argmax(probabilities[0]))
    output[0] = current
    for index in range(1, len(probabilities)):
        candidate = int(np.argmax(probabilities[index]))
        if candidate != current:
            candidate_probability = float(probabilities[index, candidate])
            advantage = candidate_probability - float(probabilities[index, current])
            if (
                candidate_probability >= config.switch_probability
                and advantage >= config.switch_margin
            ):
                current = candidate
        output[index] = current
    return output


class SoftLoopPrefixTracker:
    """Propagate approximate prefix mass from posterior marginals only."""

    def __init__(self, dictionary: LoopDictionary, *, state_count: int) -> None:
        if state_count <= 1:
            raise ValueError("state_count must exceed one")
        self._state_count = state_count
        self._paths = tuple(
            (
                definition.semantic_loop_id,
                definition.orientation_id_for(path),
                path,
            )
            for definition in dictionary.definitions.values()
            for path in definition.oriented_paths
        )
        self._mass = {
            (semantic_id, orientation_id): np.zeros(len(path), dtype=float)
            for semantic_id, orientation_id, path in self._paths
        }

    def reset_session(self) -> None:
        for values in self._mass.values():
            values.fill(0.0)

    def update(self, state_probabilities: np.ndarray) -> SoftPrefixSnapshot:
        probabilities = np.asarray(state_probabilities, dtype=float)
        if probabilities.shape != (self._state_count,):
            raise ValueError("posterior width differs from the declared state count")
        if (probabilities < 0.0).any() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("posterior probabilities must be nonnegative and normalize")
        completions: list[tuple[str, str, float]] = []
        highest_prefix = 0.0
        for semantic_id, orientation_id, path in self._paths:
            key = (semantic_id, orientation_id)
            previous = self._mass[key]
            updated = np.zeros_like(previous)
            updated[0] = max(
                float(probabilities[path[0]]),
                float(previous[0] * probabilities[path[0]]),
            )
            for progress in range(1, len(path)):
                advance = previous[progress - 1] * probabilities[path[progress]]
                stay = (
                    previous[progress] * probabilities[path[progress]]
                    if progress < len(path) - 1
                    else 0.0
                )
                updated[progress] = min(1.0, float(advance + stay))
            self._mass[key] = updated
            if len(updated) > 1:
                highest_prefix = max(highest_prefix, float(updated[:-1].max()))
            completions.append((semantic_id, orientation_id, float(updated[-1])))
        highest_completion = max((row[2] for row in completions), default=0.0)
        return SoftPrefixSnapshot(
            highest_prefix_probability=highest_prefix,
            highest_completion_probability=highest_completion,
            completion_probabilities=tuple(sorted(completions)),
        )


def build_hard_state_runs_v2(
    bars: pd.DataFrame,
    labels: np.ndarray,
    *,
    context_fields: Sequence[str],
) -> pd.DataFrame:
    """Build hard runs with every entry feature sourced from the first bar."""

    frame = bars.reset_index(drop=True)
    required = {
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        *context_fields,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"bar frame lacks required fields: {missing}")
    states = np.asarray(labels, dtype=int)
    if len(states) != len(frame):
        raise ValueError("hard-state count differs from bar rows")
    rows: list[dict[str, Any]] = []
    run_number = 0
    for (_, _), group in frame.groupby(["symbol", "session"], sort=False):
        positions = group.index.to_numpy(dtype=int)
        local = states[positions]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        ends = np.r_[starts[1:], len(local)]
        for start, end in zip(starts, ends, strict=True):
            first = int(positions[start])
            last = int(positions[end - 1])
            row: dict[str, Any] = {
                "run_id": run_number,
                "symbol": str(frame.at[first, "symbol"]),
                "session": str(frame.at[first, "session"]),
                "state": int(states[first]),
                "duration": int(end - start),
                "start_row": first,
                "end_row": last,
                "start_bar_ordinal": int(cast(Any, frame.at[first, "bar_ordinal"])),
                "end_bar_ordinal": int(cast(Any, frame.at[last, "bar_ordinal"])),
                "start_timestamp": frame.at[first, "bar_start_timestamp"],
                "end_timestamp": frame.at[last, "bar_start_timestamp"],
            }
            for field_name in context_fields:
                value = frame.at[first, field_name]
                valid = bool(pd.notna(value))
                row[field_name] = value
                row[f"{field_name}__source_timestamp"] = frame.at[first, "bar_start_timestamp"]
                row[f"{field_name}__source_bar_ordinal"] = int(
                    cast(Any, frame.at[first, "bar_ordinal"])
                )
                row[f"{field_name}__available_timestamp"] = frame.at[
                    first, "bar_complete_timestamp"
                ]
                row[f"{field_name}__causal_valid"] = valid
                row[f"{field_name}__missing_reason"] = None if valid else "source_value_missing"
                row[f"{field_name}__source_field"] = field_name
            rows.append(row)
            run_number += 1
    return pd.DataFrame(rows)


def audit_legacy_run_context(
    bars: pd.DataFrame,
    labels: np.ndarray,
    *,
    context_fields: Sequence[str],
) -> pd.DataFrame:
    """Compare first-row V2 values with the historical last-row assignment."""

    frame = bars.reset_index(drop=True)
    runs = build_hard_state_runs_v2(frame, labels, context_fields=context_fields)
    rows: list[dict[str, Any]] = []
    for run in runs.itertuples(index=False):
        for field_name in context_fields:
            first = int(cast(Any, run.start_row))
            last = int(cast(Any, run.end_row))
            start_value = frame.at[first, field_name]
            end_value = frame.at[last, field_name]
            both_missing = pd.isna(start_value) and pd.isna(end_value)
            equal = both_missing or (
                pd.notna(start_value) and pd.notna(end_value) and bool(start_value == end_value)
            )
            rows.append(
                {
                    "run_id": int(cast(Any, run.run_id)),
                    "symbol": str(run.symbol),
                    "session": str(run.session),
                    "state": int(cast(Any, run.state)),
                    "field": field_name,
                    "start_timestamp": run.start_timestamp,
                    "end_timestamp": run.end_timestamp,
                    "start_bar_ordinal": int(cast(Any, run.start_bar_ordinal)),
                    "end_bar_ordinal": int(cast(Any, run.end_bar_ordinal)),
                    "start_value": start_value,
                    "end_value": end_value,
                    "stored_legacy_value": end_value,
                    "v2_entry_value": start_value,
                    "start_end_differ": not equal,
                }
            )
    return pd.DataFrame(rows)


def churn_diagnostics(
    hard_states: np.ndarray,
    *,
    margins: np.ndarray,
    entropy: np.ndarray,
    low_margin_threshold: float,
) -> dict[str, Any]:
    """Describe hard-state transitions that depend on uncertain posterior rows."""

    states = np.asarray(hard_states, dtype=int)
    margin_values = np.asarray(margins, dtype=float)
    entropy_values = np.asarray(entropy, dtype=float)
    if not (len(states) == len(margin_values) == len(entropy_values)):
        raise ValueError("churn inputs have different lengths")
    transition = np.r_[False, states[1:] != states[:-1]]
    one_bar = sum(
        states[index] == states[index - 2] and states[index] != states[index - 1]
        for index in range(2, len(states))
    )
    two_bar = sum(
        states[index] == states[index - 2]
        or (
            index >= 3 and states[index] == states[index - 3] and states[index] != states[index - 1]
        )
        for index in range(2, len(states))
    )
    quantiles = np.quantile(entropy_values, [0.25, 0.5, 0.75]) if len(states) else []
    entropy_bin = np.searchsorted(quantiles, entropy_values, side="right")
    rate_by_entropy = {
        str(index): float(transition[entropy_bin == index].mean())
        if np.any(entropy_bin == index)
        else None
        for index in range(4)
    }
    return {
        "rows": len(states),
        "hard_transitions": int(transition.sum()),
        "low_margin_hard_transitions": int(
            (transition & (margin_values < low_margin_threshold)).sum()
        ),
        "one_bar_reversals": int(one_bar),
        "two_bar_reversals": int(two_bar),
        "transition_rate_by_entropy_quartile": rate_by_entropy,
    }


def build_completed_bar_decisions(
    bars: pd.DataFrame,
    state_export: CausalStateExport,
    *,
    legacy_hard_states: np.ndarray | None = None,
    git_sha: str,
    contract_hash: str,
    data_snapshot_hash: str,
    dictionary_version: str,
    state_model_version: str,
    hysteresis_config: HysteresisConfig = HysteresisConfig(),
    include_state_age_posterior: bool = True,
) -> pd.DataFrame:
    """Create exactly one deterministic decision for each completed source bar."""

    required = {
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "bar_is_complete",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"bar frame lacks required decision fields: {missing}")
    if len(bars) != len(state_export.state_probabilities):
        raise ValueError("bar and posterior populations differ")
    state_export.validate()
    eligible_positions = np.flatnonzero(bars["bar_is_complete"].to_numpy(bool))
    frame = bars.iloc[eligible_positions].copy().reset_index(drop=True)
    probabilities = state_export.state_probabilities[eligible_positions]
    hysteretic = np.zeros(len(frame), dtype=np.int16)
    for _, group_frame in frame.groupby(["symbol", "session"], sort=False):
        positions = group_frame.index.to_numpy(dtype=int)
        hysteretic[positions] = hysteretic_states(
            probabilities[positions], config=hysteresis_config
        )
    posterior_map = state_export.hard_map_state[eligible_positions]
    if legacy_hard_states is None:
        legacy = posterior_map.copy()
    else:
        supplied = np.asarray(legacy_hard_states, dtype=int)
        if supplied.shape != (len(bars),):
            raise ValueError("legacy hard-state count differs from bar rows")
        legacy = supplied[eligible_positions]
    frame["hard_state_legacy"] = legacy
    frame["hard_state_posterior_map"] = posterior_map
    frame["hard_state_hysteretic"] = hysteretic
    frame["posterior_state_probabilities"] = [row.tolist() for row in probabilities]
    if include_state_age_posterior:
        frame["state_age_posterior"] = [
            row.reshape(-1).tolist()
            for row in state_export.state_age_probabilities[eligible_positions]
        ]
    frame["posterior_entropy"] = state_export.posterior_entropy[eligible_positions]
    frame["top_state"] = state_export.top_state[eligible_positions]
    frame["second_state"] = state_export.second_state[eligible_positions]
    frame["top_second_margin"] = state_export.top_second_margin[eligible_positions]
    frame["posterior_map_run_age"] = state_export.hard_map_run_age[eligible_positions]
    frame["expected_state_age"] = state_export.expected_state_age[eligible_positions]
    frame["probability_state_persists_next_bar"] = state_export.probability_state_persists_next_bar[
        eligible_positions
    ]
    frame["transition_probability_next_bar"] = state_export.probability_state_transitions_next_bar[
        eligible_positions
    ]
    frame["next_state_probabilities"] = [
        row.tolist() for row in state_export.next_state_probabilities[eligible_positions]
    ]
    frame["posterior_source_timestamp"] = [
        state_export.source_timestamps[position] for position in eligible_positions
    ]
    frame["posterior_available_timestamp"] = [
        state_export.available_timestamps[position] for position in eligible_positions
    ]
    frame["decision_timestamp"] = frame["bar_complete_timestamp"]
    if not frame["bar_complete_timestamp"].le(frame["decision_timestamp"]).all():
        raise AssertionError("a decision precedes source-bar completion")

    group = frame.groupby(["symbol", "session"], sort=False)
    frame["is_session_start"] = group.cumcount().eq(0)
    reverse_count = group.cumcount(ascending=False)
    frame["is_session_end"] = reverse_count.eq(0)
    frame["bars_remaining_in_session"] = reverse_count.astype(int)
    prior_hard = group["hard_state_legacy"].shift(1)
    frame["is_run_entry"] = prior_hard.isna() | frame["hard_state_legacy"].ne(prior_hard)
    frame["run_id"] = (
        frame.groupby(["symbol", "session"], sort=False)["is_run_entry"].cumsum() - 1
    ).astype(int)
    frame["hard_run_age"] = (
        frame.groupby(["symbol", "session", "run_id"], sort=False).cumcount().add(1).astype(int)
    )

    frame["git_sha"] = git_sha
    frame["contract_hash"] = contract_hash
    frame["data_snapshot_hash"] = data_snapshot_hash
    frame["dictionary_version"] = dictionary_version
    frame["state_model_version"] = state_model_version
    frame["decision_id"] = [
        hashlib.sha256(
            "|".join(
                (
                    str(row.symbol),
                    str(row.session),
                    str(int(cast(Any, row.bar_ordinal))),
                    pd.Timestamp(cast(Any, row.bar_start_timestamp)).isoformat(),
                    state_model_version,
                    dictionary_version,
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        for row in frame.itertuples(index=False)
    ]
    flags = safety_flags()
    for key, value in flags.items():
        frame[key] = cast(Any, value)
    if frame["decision_id"].duplicated().any():
        raise AssertionError("decision IDs are not unique")
    return frame


__all__ = [
    "CausalStateExport",
    "HysteresisConfig",
    "SoftLoopPrefixTracker",
    "SoftPrefixSnapshot",
    "audit_legacy_run_context",
    "build_completed_bar_decisions",
    "build_hard_state_runs_v2",
    "causal_semimarkov_filter_v2",
    "churn_diagnostics",
    "hysteretic_states",
    "propagate_state_age_posterior",
]
