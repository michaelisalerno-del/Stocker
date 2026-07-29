"""Deterministic label-free state alignment for regime validity V2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class AlignmentWeights:
    centroid: float = 0.60
    transition: float = 0.25
    duration: float = 0.15

    def __post_init__(self) -> None:
        values = (self.centroid, self.transition, self.duration)
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("alignment weights must be finite and nonnegative")
        if sum(values) <= 0.0:
            raise ValueError("at least one alignment weight must be positive")


@dataclass(frozen=True, slots=True)
class AlignedStatePair:
    candidate_state: int
    reference_state: int
    centroid_distance: float
    transition_distance: float
    duration_distance: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class StateAlignment:
    candidate_to_reference: dict[int, int]
    pairs: tuple[AlignedStatePair, ...]
    unmatched_reference: tuple[int, ...]
    unmatched_candidate: tuple[int, ...]
    total_cost: float


def _row_distance(left: np.ndarray, right: np.ndarray) -> float:
    width = max(len(left), len(right))
    padded_left = np.zeros(width, dtype=float)
    padded_right = np.zeros(width, dtype=float)
    padded_left[: len(left)] = left
    padded_right[: len(right)] = right
    return float(np.linalg.norm(padded_left - padded_right) / math.sqrt(max(width, 1)))


def _normalized_profile(values: np.ndarray) -> np.ndarray:
    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1 or not np.isfinite(profile).all():
        raise ValueError("state profile must be a finite vector")
    total = float(profile.sum())
    return profile / total if total > 0.0 else profile


def align_states(
    reference_centroids: np.ndarray,
    candidate_centroids: np.ndarray,
    *,
    reference_transition: np.ndarray,
    candidate_transition: np.ndarray,
    reference_duration: np.ndarray,
    candidate_duration: np.ndarray,
    weights: AlignmentWeights = AlignmentWeights(),
) -> StateAlignment:
    """Align arbitrary candidate labels by emission, transition, and duration shape."""

    reference = np.asarray(reference_centroids, dtype=float)
    candidate = np.asarray(candidate_centroids, dtype=float)
    ref_transition = np.asarray(reference_transition, dtype=float)
    cand_transition = np.asarray(candidate_transition, dtype=float)
    ref_duration = np.asarray(reference_duration, dtype=float)
    cand_duration = np.asarray(candidate_duration, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape[1] != candidate.shape[1]:
        raise ValueError("reference and candidate centroids require the same feature width")
    if ref_transition.shape != (len(reference), len(reference)):
        raise ValueError("reference transition dimensions differ from reference states")
    if cand_transition.shape != (len(candidate), len(candidate)):
        raise ValueError("candidate transition dimensions differ from candidate states")
    if ref_duration.ndim != 2 or len(ref_duration) != len(reference):
        raise ValueError("reference duration profiles differ from reference states")
    if cand_duration.ndim != 2 or len(cand_duration) != len(candidate):
        raise ValueError("candidate duration profiles differ from candidate states")
    finite_inputs = (
        reference,
        candidate,
        ref_transition,
        cand_transition,
        ref_duration,
        cand_duration,
    )
    if not all(np.isfinite(values).all() for values in finite_inputs):
        raise ValueError("alignment inputs must be finite")

    weight_sum = weights.centroid + weights.transition + weights.duration
    costs = np.zeros((len(candidate), len(reference)), dtype=float)
    components: dict[tuple[int, int], tuple[float, float, float]] = {}
    for candidate_state in range(len(candidate)):
        candidate_transition_profile = np.sort(
            _normalized_profile(cand_transition[candidate_state])
        )
        candidate_duration_profile = _normalized_profile(cand_duration[candidate_state])
        for reference_state in range(len(reference)):
            centroid_distance = _row_distance(
                candidate[candidate_state], reference[reference_state]
            )
            transition_distance = _row_distance(
                candidate_transition_profile,
                np.sort(_normalized_profile(ref_transition[reference_state])),
            )
            duration_distance = _row_distance(
                candidate_duration_profile,
                _normalized_profile(ref_duration[reference_state]),
            )
            components[(candidate_state, reference_state)] = (
                centroid_distance,
                transition_distance,
                duration_distance,
            )
            costs[candidate_state, reference_state] = (
                weights.centroid * centroid_distance
                + weights.transition * transition_distance
                + weights.duration * duration_distance
            ) / weight_sum
    candidate_rows, reference_rows = linear_sum_assignment(costs)
    pairs = tuple(
        AlignedStatePair(
            candidate_state=int(candidate_state),
            reference_state=int(reference_state),
            centroid_distance=components[(int(candidate_state), int(reference_state))][0],
            transition_distance=components[(int(candidate_state), int(reference_state))][1],
            duration_distance=components[(int(candidate_state), int(reference_state))][2],
            total_cost=float(costs[candidate_state, reference_state]),
        )
        for candidate_state, reference_state in sorted(
            zip(candidate_rows, reference_rows, strict=True), key=lambda item: int(item[0])
        )
    )
    mapping = {pair.candidate_state: pair.reference_state for pair in pairs}
    matched_candidate = set(mapping)
    matched_reference = set(mapping.values())
    return StateAlignment(
        candidate_to_reference=mapping,
        pairs=pairs,
        unmatched_reference=tuple(sorted(set(range(len(reference))) - matched_reference)),
        unmatched_candidate=tuple(sorted(set(range(len(candidate))) - matched_candidate)),
        total_cost=float(sum(pair.total_cost for pair in pairs)),
    )


def apply_state_mapping(
    labels: np.ndarray, mapping: Mapping[int, int], *, unmatched_value: int = -1
) -> np.ndarray:
    """Translate labels through an explicit alignment; never assume numeric identity."""

    states = np.asarray(labels, dtype=int)
    return np.asarray(
        [mapping.get(int(state), unmatched_value) for state in states], dtype=np.int16
    )


__all__ = [
    "AlignedStatePair",
    "AlignmentWeights",
    "StateAlignment",
    "align_states",
    "apply_state_mapping",
]
