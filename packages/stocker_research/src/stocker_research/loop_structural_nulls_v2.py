"""Primitive-first structural nulls and outcome-free information attribution.

This compatibility layer reuses the corrected duration and semi-Markov models
without altering the frozen Loop Event Semantics V2 implementation.  Counts are
mutually exclusive *first future primitive-event decisions*, not overlapping
raw path matches.  Economic outcomes are rejected at the public boundary.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from numba import njit

from stocker_research.loop_nulls_v2 import (
    ClockConditionedSemiMarkovNull,
    SemiMarkovNull,
    SimulatedSession,
    benjamini_hochberg,
    empirical_p_values,
)
from stocker_research.semantic_loop_dictionary_v2 import semantic_primitive_id

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False

FIRST_EVENT_FAMILY_ORDER = (
    "TWO_STATE_OSCILLATION",
    "THREE_STATE_CYCLE",
    "FOUR_STATE_CYCLE",
    "FIVE_TO_SIX_STATE_CYCLE",
    "LONG_PRIMITIVE_CYCLE",
)

_ECONOMIC_COLUMN_FRAGMENTS = (
    "future_return",
    "return_after",
    "current_bar_log_return",
    "session_return",
    "payoff",
    "profit",
    "pnl",
    "mfe",
    "mae",
    "execution_price",
    "route_outcome",
)


class StructuralNullSimulator(Protocol):
    """Minimal simulation interface shared by both corrected V2 nulls."""

    def simulate_session(
        self, session_length: int, *, rng: np.random.Generator
    ) -> SimulatedSession: ...


def _candidate_lookup(candidate_ids: Sequence[str]) -> np.ndarray:
    lookup = np.full(196_608, -1, dtype=np.int64)
    for candidate_index, semantic_id in enumerate(candidate_ids):
        if not semantic_id.startswith("loop_p_"):
            raise ValueError("fast null candidates must be primitive semantic IDs")
        path = tuple(int(value) for value in semantic_id.removeprefix("loop_p_").split("-"))
        if len(path) < 3 or path[0] != path[-1]:
            raise ValueError(f"invalid primitive semantic ID: {semantic_id}")
        core = path[:-1]
        if len(core) > 5:
            raise ValueError("fast primary null supports frozen primitive lengths through five")
        code = len(core)
        for state in core:
            if state < 0 or state >= 8:
                raise ValueError("fast primary null requires the frozen eight-state surface")
            code = code * 8 + state
        if lookup[code] >= 0:
            raise ValueError("candidate semantic IDs contain a duplicate primitive identity")
        lookup[code] = candidate_index
    return lookup


@njit(cache=True)  # type: ignore[untyped-decorator]
def _simulate_fast_kernel(
    initial_probabilities: np.ndarray,
    transition_probabilities: np.ndarray,
    phase_transition_probabilities: np.ndarray,
    duration_cumulative_probabilities: np.ndarray,
    session_lengths: np.ndarray,
    session_groups: np.ndarray,
    group_count: int,
    candidate_lookup: np.ndarray,
    candidate_count: int,
    horizon_bars: int,
    draws: int,
    seed: int,
    clock_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((draws, group_count, candidate_count), dtype=np.int64)
    family_output = np.zeros((draws, group_count, len(FIRST_EVENT_FAMILY_ORDER)), dtype=np.int64)
    for draw in range(draws):
        np.random.seed(seed + draw)
        for session_index in range(len(session_lengths)):
            session_length = session_lengths[session_index]
            session_group = session_groups[session_index]
            states = np.empty(78, dtype=np.int64)
            durations = np.empty(78, dtype=np.int64)
            starts = np.empty(78, dtype=np.int64)
            uniform = np.random.random()
            cumulative = 0.0
            state = 0
            for state_index in range(initial_probabilities.shape[0]):
                cumulative += initial_probabilities[state_index]
                if uniform <= cumulative:
                    state = state_index
                    break
            elapsed = 0
            run_count = 0
            while elapsed < session_length:
                states[run_count] = state
                starts[run_count] = elapsed
                duration_uniform = np.random.random()
                selected_duration = duration_cumulative_probabilities.shape[1]
                for duration_index in range(duration_cumulative_probabilities.shape[1]):
                    if duration_uniform <= duration_cumulative_probabilities[state, duration_index]:
                        selected_duration = duration_index + 1
                        break
                remaining = session_length - elapsed
                duration = min(selected_duration, remaining)
                durations[run_count] = duration
                elapsed += duration
                run_count += 1
                if elapsed >= session_length:
                    break
                transition_uniform = np.random.random()
                cumulative = 0.0
                next_state = 0
                phase_index = 0 if elapsed < 12 else (1 if elapsed < 60 else 2)
                for destination in range(initial_probabilities.shape[0]):
                    probability = (
                        phase_transition_probabilities[phase_index, state, destination]
                        if clock_conditioned
                        else transition_probabilities[state, destination]
                    )
                    cumulative += probability
                    if transition_uniform <= cumulative:
                        next_state = destination
                        break
                state = next_state

            stack = np.empty(8, dtype=np.int64)
            stack_size = 0
            closure_events = np.empty(78, dtype=np.int64)
            closure_candidates = np.empty(78, dtype=np.int64)
            closure_families = np.empty(78, dtype=np.int64)
            closure_count = 0
            for event_index in range(run_count):
                current = states[event_index]
                stack_index = -1
                for index in range(stack_size):
                    if stack[index] == current:
                        stack_index = index
                        break
                if stack_index < 0:
                    stack[stack_size] = current
                    stack_size += 1
                    continue
                core_length = stack_size - stack_index
                candidate_index = -1
                if core_length == 2:
                    family_index = 0
                elif core_length == 3:
                    family_index = 1
                elif core_length == 4:
                    family_index = 2
                elif core_length <= 6:
                    family_index = 3
                else:
                    family_index = 4
                if 2 <= core_length <= 5:
                    minimum_position = stack_index
                    for index in range(stack_index + 1, stack_size):
                        if stack[index] < stack[minimum_position]:
                            minimum_position = index
                    code = core_length
                    for offset in range(core_length):
                        source = stack_index + (
                            (minimum_position - stack_index + offset) % core_length
                        )
                        code = code * 8 + stack[source]
                    candidate_index = candidate_lookup[code]
                closure_events[closure_count] = event_index
                closure_candidates[closure_count] = candidate_index
                closure_families[closure_count] = family_index
                closure_count += 1
                stack[stack_index] = current
                stack_size = stack_index + 1

            closure_pointer = 0
            for event_index in range(run_count):
                while (
                    closure_pointer < closure_count
                    and closure_events[closure_pointer] <= event_index
                ):
                    closure_pointer += 1
                if closure_pointer >= closure_count:
                    break
                candidate_index = closure_candidates[closure_pointer]
                event_bar = starts[closure_events[closure_pointer]]
                run_start = starts[event_index]
                run_end = run_start + durations[event_index] - 1
                lower = max(run_start, event_bar - horizon_bars)
                upper = min(run_end, event_bar - 1)
                if upper >= lower:
                    count = upper - lower + 1
                    family_output[draw, session_group, closure_families[closure_pointer]] += count
                    if candidate_index >= 0:
                        output[draw, session_group, candidate_index] += count
    return output, family_output


def reject_economic_columns(frame: pd.DataFrame) -> None:
    """Fail closed if economic or execution outcomes enter structural scoring."""

    prohibited = sorted(
        column
        for column in frame.columns
        if any(fragment in str(column).lower() for fragment in _ECONOMIC_COLUMN_FRAGMENTS)
    )
    if prohibited:
        raise ValueError(f"economic outcome columns are prohibited: {prohibited}")


def _closure_schedule(states: Sequence[int]) -> tuple[list[int], list[str]]:
    stack: list[int] = []
    event_indices: list[int] = []
    primitive_ids: list[str] = []
    for event_index, raw_state in enumerate(states):
        state = int(raw_state)
        if state < 0:
            raise ValueError("null session contains an unknown state")
        if state not in stack:
            stack.append(state)
            continue
        stack_index = stack.index(state)
        primitive_core = tuple(stack[stack_index:])
        if len(primitive_core) < 2:
            raise ValueError("compressed null session contains a self transition")
        event_indices.append(event_index)
        primitive_ids.append(semantic_primitive_id(primitive_core))
        stack = stack[:stack_index] + [state]
    return event_indices, primitive_ids


def first_event_candidate_counts(
    session: SimulatedSession,
    *,
    candidate_ids: Sequence[str],
    horizon_bars: int,
    decision_ordinals: Sequence[int] | None = None,
) -> np.ndarray:
    """Count candidate identities under exact future-first-event precedence."""

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    if not session.states or len(session.states) != len(session.durations):
        raise ValueError("null states and durations must be nonempty and aligned")
    if any(duration <= 0 for duration in session.durations):
        raise ValueError("null duration must be positive")
    session_length = int(sum(session.durations))
    decisions = (
        tuple(range(session_length))
        if decision_ordinals is None
        else tuple(int(value) for value in decision_ordinals)
    )
    if any(value < 0 or value >= session_length for value in decisions):
        raise ValueError("decision grid lies outside the simulated session")
    candidate_index = {str(value): index for index, value in enumerate(candidate_ids)}
    if len(candidate_index) != len(candidate_ids):
        raise ValueError("candidate semantic IDs must be unique")
    if any(not value.startswith("loop_p_") for value in candidate_index):
        raise ValueError("null candidates must be primitive semantic IDs")

    run_starts = np.r_[0, np.cumsum(np.asarray(session.durations, dtype=int))[:-1]]
    closure_events, closure_ids = _closure_schedule(session.states)
    closure_bars = [int(run_starts[event]) for event in closure_events]
    output = np.zeros(len(candidate_ids), dtype=np.int64)
    for decision in decisions:
        current_event = int(np.searchsorted(run_starts, decision, side="right") - 1)
        closure_position = bisect.bisect_right(closure_events, current_event)
        if closure_position >= len(closure_events):
            continue
        event_bar = closure_bars[closure_position]
        bars_until = event_bar - decision
        if bars_until <= 0:
            raise AssertionError("null first event is not strictly future causal")
        candidate_position = candidate_index.get(closure_ids[closure_position])
        if bars_until <= horizon_bars and candidate_position is not None:
            output[candidate_position] += 1
    return output


def simulate_first_event_null_counts(
    model: StructuralNullSimulator,
    *,
    session_lengths: Sequence[int],
    candidate_ids: Sequence[str],
    horizon_bars: int,
    draws: int,
    seed: int,
    decision_grids: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Simulate deterministic decision-level first-event counts for every session."""

    if draws < 1:
        raise ValueError("draws must be positive")
    lengths = tuple(int(value) for value in session_lengths)
    if not lengths or any(value < 1 for value in lengths):
        raise ValueError("session lengths must be positive")
    if decision_grids is not None and len(decision_grids) != len(lengths):
        raise ValueError("decision grids must align with session lengths")
    grids = decision_grids or tuple(tuple(range(length)) for length in lengths)
    rng = np.random.default_rng(seed)
    output = np.zeros((draws, len(candidate_ids)), dtype=np.int64)
    for draw in range(draws):
        counts = np.zeros(len(candidate_ids), dtype=np.int64)
        for session_index, session_length in enumerate(lengths):
            simulated = model.simulate_session(session_length, rng=rng)
            if sum(simulated.durations) != session_length:
                raise AssertionError("semi-Markov null failed to preserve session length")
            counts += first_event_candidate_counts(
                simulated,
                candidate_ids=candidate_ids,
                horizon_bars=horizon_bars,
                decision_ordinals=grids[session_index],
            )
        output[draw] = counts
    return output


def simulate_first_event_null_counts_fast(
    model: SemiMarkovNull | ClockConditionedSemiMarkovNull,
    *,
    session_lengths: Sequence[int],
    candidate_ids: Sequence[str],
    horizon_bars: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Numba-accelerated exact counterpart for the frozen eight-state primary census."""

    if draws < 1 or horizon_bars < 1:
        raise ValueError("draws and horizon_bars must be positive")
    lengths = np.asarray(session_lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 1).any() or (lengths > 78).any():
        raise ValueError("fast null requires original session lengths within one through 78")
    lookup = _candidate_lookup(candidate_ids)
    if isinstance(model, ClockConditionedSemiMarkovNull):
        base = model.base
        phase_probabilities = np.asarray(model.phase_transition_probabilities, dtype=float)
        clock_conditioned = True
    else:
        base = model
        phase_probabilities = np.broadcast_to(
            base.transition_probabilities[None, :, :],
            (3, base.state_count, base.state_count),
        ).copy()
        clock_conditioned = False
    if base.state_count != 8:
        raise ValueError("fast primary null requires the frozen eight-state model")
    grouped, _ = cast(
        tuple[np.ndarray, np.ndarray],
        _simulate_fast_kernel(
            np.asarray(base.initial_probabilities, dtype=float),
            np.asarray(base.transition_probabilities, dtype=float),
            phase_probabilities,
            np.asarray(base.duration_cumulative_probabilities, dtype=float),
            lengths,
            np.zeros(len(lengths), dtype=np.int64),
            1,
            lookup,
            len(candidate_ids),
            horizon_bars,
            draws,
            seed,
            clock_conditioned,
        ),
    )
    return grouped[:, 0, :]


def simulate_first_event_null_counts_by_group_fast(
    model: SemiMarkovNull | ClockConditionedSemiMarkovNull,
    *,
    session_lengths: Sequence[int],
    session_groups: Sequence[str],
    candidate_ids: Sequence[str],
    horizon_bars: int,
    draws: int,
    seed: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Simulate once while retaining exact per-group counts for deletion metrics."""

    if len(session_lengths) != len(session_groups):
        raise ValueError("session group IDs must align with session lengths")
    labels = tuple(sorted({str(value) for value in session_groups}))
    if not labels:
        raise ValueError("at least one session group is required")
    label_index = {label: index for index, label in enumerate(labels)}
    group_codes = np.asarray([label_index[str(value)] for value in session_groups], dtype=np.int64)
    lengths = np.asarray(session_lengths, dtype=np.int64)
    if draws < 1 or horizon_bars < 1:
        raise ValueError("draws and horizon_bars must be positive")
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 1).any() or (lengths > 78).any():
        raise ValueError(
            "fast grouped null requires original session lengths within one through 78"
        )
    lookup = _candidate_lookup(candidate_ids)
    if isinstance(model, ClockConditionedSemiMarkovNull):
        base = model.base
        phase_probabilities = np.asarray(model.phase_transition_probabilities, dtype=float)
        clock_conditioned = True
    else:
        base = model
        phase_probabilities = np.broadcast_to(
            base.transition_probabilities[None, :, :],
            (3, base.state_count, base.state_count),
        ).copy()
        clock_conditioned = False
    if base.state_count != 8:
        raise ValueError("fast grouped null requires the frozen eight-state model")
    grouped, _ = cast(
        tuple[np.ndarray, np.ndarray],
        _simulate_fast_kernel(
            np.asarray(base.initial_probabilities, dtype=float),
            np.asarray(base.transition_probabilities, dtype=float),
            phase_probabilities,
            np.asarray(base.duration_cumulative_probabilities, dtype=float),
            lengths,
            group_codes,
            len(labels),
            lookup,
            len(candidate_ids),
            horizon_bars,
            draws,
            seed,
            clock_conditioned,
        ),
    )
    return labels, grouped


def simulate_first_event_family_null_counts_fast(
    model: SemiMarkovNull | ClockConditionedSemiMarkovNull,
    *,
    session_lengths: Sequence[int],
    horizon_bars: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Count every first primitive event by the frozen topology-only family."""

    if draws < 1 or horizon_bars < 1:
        raise ValueError("draws and horizon_bars must be positive")
    lengths = np.asarray(session_lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 1).any() or (lengths > 78).any():
        raise ValueError("fast family null requires original session lengths within one through 78")
    if isinstance(model, ClockConditionedSemiMarkovNull):
        base = model.base
        phase_probabilities = np.asarray(model.phase_transition_probabilities, dtype=float)
        clock_conditioned = True
    else:
        base = model
        phase_probabilities = np.broadcast_to(
            base.transition_probabilities[None, :, :],
            (3, base.state_count, base.state_count),
        ).copy()
        clock_conditioned = False
    if base.state_count != 8:
        raise ValueError("fast family null requires the frozen eight-state model")
    _, grouped_families = cast(
        tuple[np.ndarray, np.ndarray],
        _simulate_fast_kernel(
            np.asarray(base.initial_probabilities, dtype=float),
            np.asarray(base.transition_probabilities, dtype=float),
            phase_probabilities,
            np.asarray(base.duration_cumulative_probabilities, dtype=float),
            lengths,
            np.zeros(len(lengths), dtype=np.int64),
            1,
            _candidate_lookup(()),
            0,
            horizon_bars,
            draws,
            seed,
            clock_conditioned,
        ),
    )
    return grouped_families[:, 0, :]


def qualify_structural_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the preregistered support, null, chronology, and deletion gates."""

    required = {
        "semantic_loop_id",
        "support_pass",
        "semi_markov_q",
        "semi_markov_rate_ratio",
        "excess_count",
        "positive_excess_quarters",
        "leave_one_stock_out_minimum_rate_ratio",
        "clock_null_q",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"structural qualification missing fields: {sorted(missing)}")
    output = candidates.copy()
    output["structurally_qualified"] = (
        output["support_pass"].astype(bool)
        & output["semi_markov_q"].le(0.10)
        & output["semi_markov_rate_ratio"].ge(1.20)
        & output["excess_count"].gt(0.0)
        & output["positive_excess_quarters"].ge(3)
        & output["leave_one_stock_out_minimum_rate_ratio"].gt(1.0)
    )
    output["clock_null_status"] = np.where(
        output["clock_null_q"].le(0.10),
        "GLOBALLY_RECURRENT",
        "CLOCK_DEPENDENT_STRUCTURAL_CANDIDATE",
    )
    reasons: list[str] = []
    for row in output.itertuples(index=False):
        failed = []
        if not bool(row.support_pass):
            failed.append("support")
        if float(cast(Any, row.semi_markov_q)) > 0.10:
            failed.append("semi_markov_fdr")
        if float(cast(Any, row.semi_markov_rate_ratio)) < 1.20:
            failed.append("rate_ratio")
        if float(cast(Any, row.excess_count)) <= 0:
            failed.append("nonpositive_excess")
        if int(cast(Any, row.positive_excess_quarters)) < 3:
            failed.append("quarter_consistency")
        if float(cast(Any, row.leave_one_stock_out_minimum_rate_ratio)) <= 1.0:
            failed.append("stock_deletion")
        reasons.append(";".join(failed))
    output["structural_rejection_reason"] = reasons
    return output


def summarize_null_draws(
    observed_counts: Sequence[int],
    primary_draws: np.ndarray,
    clock_draws: np.ndarray,
) -> pd.DataFrame:
    """Return deterministic empirical statistics and independent FDR columns."""

    observed = np.asarray(observed_counts, dtype=np.int64)
    primary = np.asarray(primary_draws, dtype=np.int64)
    clock = np.asarray(clock_draws, dtype=np.int64)
    if primary.ndim != 2 or clock.ndim != 2:
        raise ValueError("null draws must be matrices")
    if primary.shape[1] != len(observed) or clock.shape[1] != len(observed):
        raise ValueError("candidate count differs across observed and null matrices")
    primary_p = empirical_p_values(observed, primary)
    clock_p = empirical_p_values(observed, clock)
    primary_mean = primary.mean(axis=0)
    clock_mean = clock.mean(axis=0)
    return pd.DataFrame(
        {
            "observed_count": observed,
            "semi_markov_null_mean": primary_mean,
            "semi_markov_null_lower": np.quantile(primary, 0.025, axis=0),
            "semi_markov_null_upper": np.quantile(primary, 0.975, axis=0),
            "semi_markov_rate_ratio": (observed + 0.5) / (primary_mean + 0.5),
            "excess_count": observed - primary_mean,
            "semi_markov_p": primary_p,
            "semi_markov_q": benjamini_hochberg(primary_p),
            "clock_null_mean": clock_mean,
            "clock_null_lower": np.quantile(clock, 0.025, axis=0),
            "clock_null_upper": np.quantile(clock, 0.975, axis=0),
            "clock_null_rate_ratio": (observed + 0.5) / (clock_mean + 0.5),
            "clock_null_p": clock_p,
            "clock_null_q": benjamini_hochberg(clock_p),
        }
    )


def _conditional_probabilities(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    target: str,
    keys: Sequence[str],
) -> np.ndarray:
    positives = float(train[target].sum())
    global_probability = (positives + 1.0) / (len(train) + 2.0)
    if not keys:
        return np.full(len(score), global_probability, dtype=float)
    grouped = train.groupby(list(keys), dropna=False)[target].agg(["sum", "count"])
    lookup = ((grouped["sum"] + 1.0) / (grouped["count"] + 2.0)).to_dict()
    score_keys: list[object]
    if len(keys) == 1:
        score_keys = score[keys[0]].tolist()
    else:
        score_keys = list(score[list(keys)].itertuples(index=False, name=None))
    return np.asarray([float(lookup.get(key, global_probability)) for key in score_keys])


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def _brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(y - probability)))


def _calibration_error(y: np.ndarray, probability: np.ndarray) -> float:
    bins = np.minimum((probability * 10).astype(int), 9)
    error = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(y[selected].mean()) - float(probability[selected].mean())
            )
    return error


def _candidate_prefix_codes(candidate_id: str) -> dict[int, np.ndarray]:
    """Encode every proper oriented prefix of one rank-independent primitive ID."""

    if not candidate_id.startswith("loop_p_"):
        raise ValueError("information candidates must use primitive semantic IDs")
    closed_path = tuple(int(value) for value in candidate_id.removeprefix("loop_p_").split("-"))
    if len(closed_path) < 3 or closed_path[0] != closed_path[-1]:
        raise ValueError(f"invalid primitive semantic ID: {candidate_id}")
    core = closed_path[:-1]
    encoded: dict[int, set[int]] = {width: set() for width in range(1, len(core) + 1)}
    for rotation_index in range(len(core)):
        rotation = core[rotation_index:] + core[:rotation_index]
        oriented = rotation + (rotation[0],)
        for width in range(1, len(oriented)):
            code = width
            for state in oriented[:width]:
                if state < 0 or state >= 8:
                    raise ValueError("information prefix encoding requires the frozen states 0-7")
                code = code * 8 + state
            encoded[width].add(code)
    return {width: np.asarray(sorted(values), dtype=np.int64) for width, values in encoded.items()}


def _history_suffix_codes(histories: pd.Series, maximum_width: int) -> dict[int, np.ndarray]:
    """Encode causal state-event suffixes once for all candidate attribution models."""

    normalised = [tuple(int(state) for state in value) for value in histories]
    output: dict[int, np.ndarray] = {}
    for width in range(1, maximum_width + 1):
        codes = np.full(len(normalised), -1, dtype=np.int64)
        for index, history in enumerate(normalised):
            if len(history) < width:
                continue
            code = width
            for state in history[-width:]:
                if state < 0 or state >= 8:
                    raise ValueError("information state history requires the frozen states 0-7")
                code = code * 8 + state
            codes[index] = code
        output[width] = codes
    return output


def information_increment_from_chronological_folds(
    decisions_and_events: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
) -> pd.DataFrame:
    """Attribute binary candidate information with expanding chronological folds."""

    reject_economic_columns(decisions_and_events)
    required = {
        "decision_id",
        "decision_timestamp",
        "symbol",
        "hard_state_legacy",
        "previous_completed_state_1",
        "previous_completed_state_2",
        "previous_completed_state_3",
        "hard_run_age",
        "recent_state_events",
        "previous_completed_primitive_loop",
        "same_primitive_repeat_depth",
        "bars_since_previous_primitive_completion",
        "primitive_loop_id",
    }
    missing = required.difference(decisions_and_events.columns)
    if missing:
        raise ValueError(f"information attribution missing fields: {sorted(missing)}")
    frame = decisions_and_events.copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    frame["_period"] = frame["decision_timestamp"].dt.strftime("%Y-%m")
    frame["_quarter"] = (
        frame["decision_timestamp"].dt.year.astype(str)
        + "Q"
        + (((frame["decision_timestamp"].dt.month - 1) // 3) + 1).astype(str)
    )
    frame["_age_bucket"] = pd.cut(
        pd.to_numeric(frame["hard_run_age"], errors="coerce"),
        bins=[-math.inf, 1, 2, 4, 8, 16, math.inf],
        labels=False,
    )
    frame = frame.sort_values(["decision_timestamp", "decision_id"], kind="mergesort")
    maximum_candidate_width = max(
        len(str(candidate_id).removeprefix("loop_p_").split("-")) - 1
        for candidate_id in candidate_ids
    )
    suffix_codes = _history_suffix_codes(frame["recent_state_events"], maximum_candidate_width)
    periods = sorted(frame["_period"].unique())
    if len(periods) < 4:
        raise ValueError("at least four chronological periods are required")
    baselines: dict[str, tuple[str, ...]] = {
        "B0": (),
        "B1": ("hard_state_legacy",),
        "B2": ("hard_state_legacy", "previous_completed_state_1"),
        "B3": (
            "hard_state_legacy",
            "previous_completed_state_1",
            "previous_completed_state_2",
        ),
        "B4": (
            "hard_state_legacy",
            "previous_completed_state_1",
            "previous_completed_state_2",
            "previous_completed_state_3",
        ),
        "B5": ("hard_state_legacy", "_age_bucket"),
    }
    records: list[dict[str, object]] = []
    for candidate_id in sorted(str(value) for value in candidate_ids):
        candidate = frame.copy()
        candidate["_target"] = candidate["primitive_loop_id"].eq(candidate_id).astype(int)
        truth_parts: list[np.ndarray] = []
        symbol_parts: list[np.ndarray] = []
        quarter_parts: list[np.ndarray] = []
        prefix_progress = np.zeros(len(candidate), dtype=np.int64)
        for width, codes in _candidate_prefix_codes(candidate_id).items():
            prefix_progress[np.isin(suffix_codes[width], codes)] = width
        candidate["_candidate_prefix_progress"] = prefix_progress
        candidate["_candidate_was_previous"] = candidate["previous_completed_primitive_loop"].eq(
            candidate_id
        )
        repeat_depth = pd.to_numeric(
            candidate["same_primitive_repeat_depth"], errors="coerce"
        ).fillna(0)
        candidate["_candidate_repeat_bucket"] = np.where(
            candidate["_candidate_was_previous"], np.minimum(repeat_depth, 3), 0
        ).astype(int)
        bars_since = pd.to_numeric(
            candidate["bars_since_previous_primitive_completion"], errors="coerce"
        )
        candidate["_candidate_recency_bucket"] = np.where(
            candidate["_candidate_was_previous"],
            pd.cut(
                bars_since,
                bins=[-math.inf, 1, 2, 4, 8, 16, math.inf],
                labels=False,
            ).fillna(-1),
            -1,
        ).astype(int)
        candidate_aware_keys = {
            "C4": baselines["B4"]
            + (
                "_candidate_prefix_progress",
                "_candidate_was_previous",
                "_candidate_repeat_bucket",
                "_candidate_recency_bucket",
            ),
            "C5": baselines["B5"]
            + (
                "_candidate_prefix_progress",
                "_candidate_was_previous",
                "_candidate_repeat_bucket",
                "_candidate_recency_bucket",
            ),
        }
        prediction_parts: dict[str, list[np.ndarray]] = {
            key: [] for key in (*baselines, *candidate_aware_keys)
        }
        for score_period in periods[2:]:
            train = candidate.loc[candidate["_period"].lt(score_period)]
            score = candidate.loc[candidate["_period"].eq(score_period)]
            if train.empty or score.empty:
                continue
            truth_parts.append(score["_target"].to_numpy(dtype=float))
            symbol_parts.append(score["symbol"].astype(str).to_numpy())
            quarter_parts.append(score["_quarter"].astype(str).to_numpy())
            for baseline, keys in baselines.items():
                prediction_parts[baseline].append(
                    _conditional_probabilities(train, score, target="_target", keys=keys)
                )
            for model, keys in candidate_aware_keys.items():
                prediction_parts[model].append(
                    _conditional_probabilities(train, score, target="_target", keys=keys)
                )
        if not truth_parts:
            raise ValueError("chronological folds produced no scored observations")
        truth = np.concatenate(truth_parts)
        symbols = np.concatenate(symbol_parts)
        quarters = np.concatenate(quarter_parts)
        predictions = {
            baseline: np.concatenate(parts) for baseline, parts in prediction_parts.items()
        }
        losses = {baseline: _log_loss(truth, values) for baseline, values in predictions.items()}
        briers = {baseline: _brier(truth, values) for baseline, values in predictions.items()}
        strongest = min(("B4", "B5"), key=lambda baseline: (losses[baseline], baseline))
        candidate_model = f"C{strongest.removeprefix('B')}"
        increment = losses[strongest] - losses[candidate_model]
        quarter_directions = []
        for quarter in sorted(set(quarters)):
            mask = quarters == quarter
            quarter_directions.append(
                _log_loss(truth[mask], predictions[strongest][mask])
                > _log_loss(truth[mask], predictions[candidate_model][mask])
            )
        deletion_directions = []
        for symbol in sorted(set(symbols)):
            mask = symbols != symbol
            if mask.any():
                deletion_directions.append(
                    _log_loss(truth[mask], predictions[strongest][mask])
                    > _log_loss(truth[mask], predictions[candidate_model][mask])
                )
        records.append(
            {
                "semantic_loop_id": candidate_id,
                "primitive_loop_id": candidate_id,
                "candidate_event_prevalence": float(truth.mean()),
                **{f"{baseline.lower()}_log_loss": losses[baseline] for baseline in baselines},
                "strongest_retained_baseline": strongest,
                "candidate_aware_model": candidate_model,
                "baseline_log_loss": losses[strongest],
                "candidate_aware_log_loss": losses[candidate_model],
                "oof_log_loss_increment": increment,
                "relative_log_loss_improvement": increment / max(losses[strongest], 1e-12),
                "brier_improvement": briers[strongest] - briers[candidate_model],
                "conditional_information_gain": max(0.0, increment) / math.log(2.0),
                "calibration_error": _calibration_error(truth, predictions[candidate_model]),
                "quarter_consistency": float(np.mean(quarter_directions)),
                "stock_deletion_consistency": float(np.mean(deletion_directions)),
                "scored_fold_count": len(periods) - 2,
                "scored_decisions": len(truth),
                "information_qualified": increment > 0.0
                and briers[strongest] > briers[candidate_model],
            }
        )
    return pd.DataFrame.from_records(records).sort_values("semantic_loop_id").reset_index(drop=True)


__all__ = [
    "FIRST_EVENT_FAMILY_ORDER",
    "StructuralNullSimulator",
    "first_event_candidate_counts",
    "information_increment_from_chronological_folds",
    "qualify_structural_candidates",
    "reject_economic_columns",
    "simulate_first_event_null_counts",
    "simulate_first_event_family_null_counts_fast",
    "simulate_first_event_null_counts_by_group_fast",
    "simulate_first_event_null_counts_fast",
    "summarize_null_draws",
]
