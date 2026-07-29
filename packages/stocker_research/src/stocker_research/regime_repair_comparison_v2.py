"""Aligned structural comparisons for repaired regime model lineages V2."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from stocker_research.semantic_loop_dictionary_v2 import semantic_primitive_id
from stocker_research.state_alignment_v2 import apply_state_mapping

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


def _validated_labels(labels: np.ndarray, row_count: int) -> np.ndarray:
    states = np.asarray(labels, dtype=int)
    if states.shape != (row_count,):
        raise ValueError("state labels differ from panel rows")
    if np.any(states < 0):
        raise ValueError("state labels must be nonnegative")
    return states


def run_boundary_ledger(panel: pd.DataFrame, labels: np.ndarray, *, lineage: str) -> pd.DataFrame:
    """Compress labels without allowing a run to cross a causal segment."""

    required = {
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"comparison panel lacks run columns: {missing}")
    states = _validated_labels(labels, len(panel))
    rows: list[dict[str, Any]] = []
    run_index = 0
    for segment_id, group in panel.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        local = states[positions]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        ends = np.r_[starts[1:], len(local)]
        for start, end in zip(starts, ends, strict=True):
            first = int(positions[int(start)])
            last = int(positions[int(end) - 1])
            rows.append(
                {
                    "lineage": lineage,
                    "run_id": f"{lineage}::run_{run_index:08d}",
                    "symbol": str(panel.at[first, "symbol"]),
                    "session": str(panel.at[first, "session"]),
                    "segment_id": str(segment_id),
                    "state": int(states[first]),
                    "duration": int(end - start),
                    "start_position": first,
                    "end_position": last,
                    "start_bar_ordinal": int(cast(Any, panel.at[first, "bar_ordinal"])),
                    "end_bar_ordinal": int(cast(Any, panel.at[last, "bar_ordinal"])),
                    "start_timestamp": panel.at[first, "bar_start_timestamp"],
                    "end_timestamp": panel.at[last, "bar_start_timestamp"],
                }
            )
            run_index += 1
    return pd.DataFrame(rows)


def primitive_loop_events(
    panel: pd.DataFrame,
    labels: np.ndarray,
    *,
    minimum_transitions: int = 2,
    maximum_transitions: int = 5,
) -> pd.DataFrame:
    """Enumerate closed primitive state paths inside individual segments."""

    if minimum_transitions <= 0 or maximum_transitions < minimum_transitions:
        raise ValueError("invalid primitive transition bounds")
    required = {
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"comparison panel lacks loop columns: {missing}")
    states = _validated_labels(labels, len(panel))
    rows: list[dict[str, Any]] = []
    for segment_id, group in panel.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        local = states[positions]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        run_states = local[starts]
        for transition_length in range(minimum_transitions, maximum_transitions + 1):
            width = transition_length + 1
            for start in range(0, len(run_states) - width + 1):
                path = tuple(int(value) for value in run_states[start : start + width])
                if path[0] != path[-1]:
                    continue
                event_local = int(starts[start + transition_length])
                event_position = int(positions[event_local])
                rows.append(
                    {
                        "symbol": str(panel.at[event_position, "symbol"]),
                        "session": str(panel.at[event_position, "session"]),
                        "segment_id": str(segment_id),
                        "event_position": event_position,
                        "event_bar_ordinal": int(
                            cast(Any, panel.at[event_position, "bar_ordinal"])
                        ),
                        "event_timestamp": panel.at[event_position, "bar_complete_timestamp"],
                        "primitive_loop_id": semantic_primitive_id(path[:-1]),
                        "orientation_id": "->".join(str(value) for value in path),
                        "prefix_progress": transition_length,
                        "transition_length": transition_length,
                    }
                )
    columns = [
        "symbol",
        "session",
        "segment_id",
        "event_position",
        "event_bar_ordinal",
        "event_timestamp",
        "primitive_loop_id",
        "orientation_id",
        "prefix_progress",
        "transition_length",
    ]
    return pd.DataFrame(rows, columns=columns)


def aligned_assignment_metrics(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    candidate_to_reference: Mapping[int, int],
) -> dict[str, float]:
    """Compute label-invariant agreement after a declared alignment."""

    reference = np.asarray(reference_labels, dtype=int)
    candidate = np.asarray(candidate_labels, dtype=int)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("assignment vectors must be aligned one-dimensional rows")
    aligned = apply_state_mapping(candidate, candidate_to_reference)
    comparable = aligned >= 0
    if not comparable.any():
        return {
            "bar_level_aligned_agreement": math.nan,
            "adjusted_rand_index": math.nan,
            "normalized_mutual_information": math.nan,
            "comparable_fraction": 0.0,
        }
    return {
        "bar_level_aligned_agreement": float(np.mean(reference[comparable] == aligned[comparable])),
        "adjusted_rand_index": float(
            adjusted_rand_score(reference[comparable], aligned[comparable])
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(reference[comparable], aligned[comparable])
        ),
        "comparable_fraction": float(np.mean(comparable)),
    }


def compare_loop_events(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    allowed_shift_bars: int,
) -> dict[str, float | int]:
    """Compare candidate events against one fixed reference-event population."""

    if allowed_shift_bars < 0:
        raise ValueError("allowed shift must be nonnegative")
    required = {
        "symbol",
        "session",
        "primitive_loop_id",
        "event_bar_ordinal",
    }
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} events lack columns: {missing}")
    reference_rows = [
        (str(symbol), str(session), str(loop_id), int(bar))
        for symbol, session, loop_id, bar in reference[
            ["symbol", "session", "primitive_loop_id", "event_bar_ordinal"]
        ].itertuples(index=False, name=None)
    ]
    candidate_rows = {
        (str(symbol), str(session), str(loop_id), int(bar))
        for symbol, session, loop_id, bar in candidate[
            ["symbol", "session", "primitive_loop_id", "event_bar_ordinal"]
        ].itertuples(index=False, name=None)
    }
    exact = sum(row in candidate_rows for row in reference_rows)
    by_identity: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for symbol, session, loop_id, bar in candidate_rows:
        by_identity[(symbol, session, loop_id)].add(bar)
    bounded = sum(
        any(
            abs(candidate_bar - bar) <= allowed_shift_bars
            for candidate_bar in by_identity[(symbol, session, loop_id)]
        )
        for symbol, session, loop_id, bar in reference_rows
    )
    denominator = len(reference_rows)
    return {
        "reference_event_count": denominator,
        "candidate_event_count": len(candidate_rows),
        "exact_event_agreement": exact / denominator if denominator else math.nan,
        "same_primitive_bounded_shift_fraction": (
            bounded / denominator if denominator else math.nan
        ),
        "unmatched_reference_events": denominator - bounded,
    }


def state_occupancy(labels: np.ndarray, *, state_count: int) -> np.ndarray:
    states = np.asarray(labels, dtype=int)
    if np.any((states < 0) | (states >= state_count)):
        raise ValueError("occupancy labels exceed state support")
    return np.bincount(states, minlength=state_count).astype(float) / len(states)


def transition_matrix(
    labels: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    state_count: int,
    pseudocount: float = 0.5,
) -> np.ndarray:
    """Estimate run-transition rows without crossing supplied groups."""

    if pseudocount <= 0.0:
        raise ValueError("transition pseudocount must be positive")
    states = np.asarray(labels, dtype=int)
    counts = np.full((state_count, state_count), pseudocount, dtype=float)
    np.fill_diagonal(counts, 0.0)
    for raw_positions in groups:
        positions = np.asarray(raw_positions, dtype=int)
        local = states[positions]
        compressed = local[np.r_[True, local[1:] != local[:-1]]]
        for origin, destination in zip(compressed[:-1], compressed[1:], strict=True):
            if int(origin) != int(destination):
                counts[int(origin), int(destination)] += 1.0
    return np.asarray(counts / counts.sum(axis=1, keepdims=True), dtype=float)


def reversal_rates(labels: np.ndarray, groups: Sequence[np.ndarray]) -> tuple[float, float]:
    """Return one-bar and two-bar hard-state reversal rates."""

    states = np.asarray(labels, dtype=int)
    transitions = 0
    one_bar = 0
    two_bar = 0
    for raw_positions in groups:
        positions = np.asarray(raw_positions, dtype=int)
        local = states[positions]
        transitions += int(np.sum(local[1:] != local[:-1]))
        for index in range(1, len(local) - 1):
            if local[index] != local[index - 1] and local[index + 1] == local[index - 1]:
                one_bar += 1
        for index in range(1, len(local) - 2):
            if (
                local[index] != local[index - 1]
                and local[index + 1] == local[index]
                and local[index + 2] == local[index - 1]
            ):
                two_bar += 1
    if transitions == 0:
        return math.nan, math.nan
    return one_bar / transitions, two_bar / transitions


def compare_posteriors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Summarize row-wise posterior displacement on identical rows."""

    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("posterior matrices must share shape")
    if not np.allclose(left.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("reference posterior does not normalize")
    if not np.allclose(right.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("candidate posterior does not normalize")
    absolute = np.abs(left - right)
    return {
        "mean_l1_distance": float(np.mean(np.sum(absolute, axis=1))),
        "median_l1_distance": float(np.median(np.sum(absolute, axis=1))),
        "maximum_absolute_probability_change": float(np.max(absolute)),
        "argmax_agreement": float(np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1))),
    }


__all__ = [
    "aligned_assignment_metrics",
    "compare_loop_events",
    "compare_posteriors",
    "primitive_loop_events",
    "reversal_rates",
    "run_boundary_ledger",
    "state_occupancy",
    "transition_matrix",
]
