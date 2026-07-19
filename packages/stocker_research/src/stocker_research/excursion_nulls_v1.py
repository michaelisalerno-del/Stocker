"""Continuous-trajectory structural nulls for excursion research V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class IncrementNullResult:
    increments: FloatArray
    source_indices: IntArray
    source_phases: StringArray


def _increments(values: FloatArray) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("null increments must be a finite feature matrix")
    return matrix


def phase_conditioned_increment_block_null(
    increments: FloatArray,
    *,
    phases: StringArray,
    block_length: int,
    seed: int,
) -> IncrementNullResult:
    """Resample contiguous increment blocks without crossing clock-phase strata."""

    matrix = _increments(increments)
    phase_values = np.asarray(phases, dtype=str)
    if phase_values.shape != (len(matrix),):
        raise ValueError("clock phases differ from increment rows")
    if block_length <= 0:
        raise ValueError("increment block length must be positive")
    rng = np.random.default_rng(seed)
    output = np.empty_like(matrix)
    source_indices = np.full(len(matrix), -1, dtype=np.int64)
    ordered_phases = tuple(dict.fromkeys(phase_values.tolist()))
    for phase in ordered_phases:
        targets = np.flatnonzero(phase_values == phase)
        if len(targets) == 0:
            continue
        blocks = [
            targets[start : start + block_length] for start in range(0, len(targets), block_length)
        ]
        chosen: list[int] = []
        while len(chosen) < len(targets):
            block = blocks[int(rng.integers(0, len(blocks)))]
            chosen.extend(int(value) for value in block)
        selected = np.asarray(chosen[: len(targets)], dtype=np.int64)
        output[targets] = matrix[selected]
        source_indices[targets] = selected
    if np.any(source_indices < 0):
        raise AssertionError("phase-conditioned null left increments unassigned")
    source_phases = phase_values[source_indices]
    if not np.array_equal(source_phases, phase_values):
        raise AssertionError("phase-conditioned null crossed a clock-phase stratum")
    return IncrementNullResult(
        increments=output,
        source_indices=source_indices,
        source_phases=source_phases,
    )


def circular_increment_control(
    increments: FloatArray,
    *,
    offset: int,
    block_length: int,
) -> FloatArray:
    """Circularly rotate complete ordered blocks and preserve session length."""

    matrix = _increments(increments)
    if block_length <= 0:
        raise ValueError("circular block length must be positive")
    blocks = [matrix[start : start + block_length] for start in range(0, len(matrix), block_length)]
    if not blocks:
        return matrix.copy()
    rotation = offset % len(blocks)
    rotated = blocks[rotation:] + blocks[:rotation]
    return np.concatenate(rotated, axis=0)[: len(matrix)].copy()


def reconstruct_trajectory(start: FloatArray, increments: FloatArray) -> FloatArray:
    """Reconstruct a complete null trajectory from its observed session start."""

    origin = np.asarray(start, dtype=np.float64)
    matrix = _increments(increments)
    if origin.shape != (matrix.shape[1],) or not np.isfinite(origin).all():
        raise ValueError("null trajectory start differs from increment dimensions")
    if len(matrix) == 0:
        return origin[None, :]
    return np.vstack([origin, origin + np.cumsum(matrix, axis=0)])


def clock_phase_labels(bar_ordinals: IntArray) -> StringArray:
    """Return frozen opening middle and late regular-session strata."""

    values = np.asarray(bar_ordinals, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values > 77):
        raise ValueError("bar ordinals must lie inside regular-session support")
    return np.asarray(
        np.where(values < 18, "OPENING", np.where(values < 60, "MIDDLE", "LATE")),
        dtype=str,
    )


@dataclass(frozen=True, slots=True)
class PhaseVAR1Model:
    phases: tuple[str, ...]
    intercepts: dict[str, FloatArray]
    coefficients: dict[str, FloatArray]
    residuals: dict[str, FloatArray]
    ridge: float


def fit_phase_conditioned_var1(
    increments: FloatArray,
    *,
    phases: StringArray,
    ridge: float = 1e-3,
) -> PhaseVAR1Model:
    """Fit one compact ridge VAR(1) increment transition per clock phase."""

    matrix = _increments(increments)
    phase_values = np.asarray(phases, dtype=str)
    if phase_values.shape != (len(matrix),) or ridge <= 0.0:
        raise ValueError("VAR phases or ridge are invalid")
    intercepts: dict[str, FloatArray] = {}
    coefficients: dict[str, FloatArray] = {}
    residuals: dict[str, FloatArray] = {}
    phase_order = tuple(dict.fromkeys(phase_values.tolist()))
    dimension = matrix.shape[1]
    for phase in phase_order:
        phase_positions = np.flatnonzero(phase_values == phase)
        consecutive = phase_positions[1:][np.diff(phase_positions) == 1]
        previous = consecutive - 1
        if len(consecutive) < dimension + 2:
            selected = matrix[phase_positions]
            intercepts[phase] = selected.mean(axis=0)
            coefficients[phase] = np.zeros((dimension, dimension), dtype=float)
            residuals[phase] = selected - intercepts[phase]
            continue
        design = np.c_[np.ones(len(previous)), matrix[previous]]
        target = matrix[consecutive]
        penalty = np.eye(dimension + 1, dtype=float) * ridge
        penalty[0, 0] = 0.0
        fitted = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        intercepts[phase] = np.asarray(fitted[0], dtype=float)
        coefficients[phase] = np.asarray(fitted[1:].T, dtype=float)
        residuals[phase] = np.asarray(target - design @ fitted, dtype=float)
    return PhaseVAR1Model(
        phases=phase_order,
        intercepts=intercepts,
        coefficients=coefficients,
        residuals=residuals,
        ridge=ridge,
    )


def simulate_phase_conditioned_var1(
    model: PhaseVAR1Model,
    *,
    phases: StringArray,
    initial_increment: FloatArray,
    seed: int,
) -> FloatArray:
    """Simulate increments with phase-local residual resampling."""

    phase_values = np.asarray(phases, dtype=str)
    previous = np.asarray(initial_increment, dtype=np.float64)
    if previous.ndim != 1:
        raise ValueError("initial VAR increment must be one-dimensional")
    output = np.empty((len(phase_values), len(previous)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index, phase in enumerate(phase_values):
        if phase not in model.intercepts:
            raise ValueError(f"VAR model lacks phase {phase}")
        candidates = model.residuals[phase]
        residual = candidates[int(rng.integers(0, len(candidates)))]
        current = model.intercepts[phase] + model.coefficients[phase] @ previous + residual
        output[index] = current
        previous = current
    return output


def benjamini_hochberg(p_values: FloatArray) -> FloatArray:
    """Return deterministic Benjamini-Hochberg adjusted q-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite probabilities")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


__all__ = [
    "IncrementNullResult",
    "PhaseVAR1Model",
    "benjamini_hochberg",
    "circular_increment_control",
    "clock_phase_labels",
    "fit_phase_conditioned_var1",
    "phase_conditioned_increment_block_null",
    "reconstruct_trajectory",
    "simulate_phase_conditioned_var1",
]
