"""Causal continuous-trajectory geometry for structural excursion research V1.

The module contains no price outcome, payoff, broker, order, position, or runtime
surface.  It transforms already-declared structural emissions and posteriors and
keeps missingness explicit for callers that must fail closed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _matrix(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("trajectory values must be a two-dimensional nonempty feature matrix")
    return result


def _validated_groups(groups: Sequence[NDArray[Any]], rows: int) -> tuple[NDArray[np.int64], ...]:
    normalized: list[NDArray[np.int64]] = []
    assigned = np.zeros(rows, dtype=bool)
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=np.int64)
        if group.ndim != 1 or len(group) == 0:
            continue
        if group.min() < 0 or group.max() >= rows or np.any(np.diff(group) <= 0):
            raise ValueError("trajectory groups must be increasing and inside the input")
        if assigned[group].any():
            raise ValueError("trajectory groups overlap")
        assigned[group] = True
        normalized.append(group)
    if rows and not assigned.all():
        raise ValueError("trajectory groups do not cover every row")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class RobustGeometry:
    """Development-fitted median and IQR transformation with missingness flags."""

    medians: FloatArray
    centers: FloatArray
    scales: FloatArray
    fit_row_count: int

    def __post_init__(self) -> None:
        feature_count = len(self.medians)
        if (
            feature_count == 0
            or self.centers.shape != (feature_count,)
            or self.scales.shape != (feature_count,)
        ):
            raise ValueError("robust geometry arrays have inconsistent dimensions")
        if self.fit_row_count <= 0:
            raise ValueError("robust geometry requires development rows")
        if not np.isfinite(self.medians).all() or not np.isfinite(self.centers).all():
            raise ValueError("robust geometry locations must be finite")
        if not np.isfinite(self.scales).all() or np.any(self.scales <= 0.0):
            raise ValueError("robust geometry scales must be finite and positive")

    def transform(
        self,
        values: ArrayLike,
    ) -> tuple[FloatArray, BoolArray]:
        """Transform rows and return the original cell-level missingness mask."""

        matrix = _matrix(values).copy()
        if matrix.shape[1] != len(self.medians):
            raise ValueError("trajectory feature count differs from fitted geometry")
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(self.medians, np.nonzero(missing)[1])
        transformed = (matrix - self.centers) / self.scales
        if not np.isfinite(transformed).all():
            raise AssertionError("robust trajectory transform produced nonfinite values")
        return transformed, missing


def fit_robust_geometry(
    values: ArrayLike,
    *,
    fit_mask: BoolArray | None = None,
) -> RobustGeometry:
    """Fit median imputation and diagonal robust scaling on declared rows only."""

    matrix = _matrix(values)
    mask = (
        np.ones(len(matrix), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    )
    if mask.shape != (len(matrix),) or not mask.any():
        raise ValueError("fit_mask must select at least one trajectory row")
    fitted = matrix[mask].copy()
    finite_counts = np.isfinite(fitted).sum(axis=0)
    if np.any(finite_counts == 0):
        raise ValueError("a trajectory feature is wholly unavailable in development")
    medians = np.nanmedian(fitted, axis=0)
    missing = ~np.isfinite(fitted)
    if missing.any():
        fitted[missing] = np.take(medians, np.nonzero(missing)[1])
    centers = np.median(fitted, axis=0)
    lower = np.quantile(fitted, 0.25, axis=0)
    upper = np.quantile(fitted, 0.75, axis=0)
    scales = upper - lower
    fallback = np.median(np.abs(fitted - centers), axis=0) * 1.4826
    scales = np.where(scales > 1e-12, scales, fallback)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return RobustGeometry(
        medians=np.asarray(medians, dtype=np.float64),
        centers=np.asarray(centers, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        fit_row_count=int(mask.sum()),
    )


@dataclass(frozen=True, slots=True)
class TrajectoryFeatures:
    """Gap-local causal trajectory summaries."""

    first_difference: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    local_path_length: FloatArray
    directional_consistency: FloatArray


def trajectory_features(
    values: ArrayLike,
    *,
    groups: Sequence[NDArray[Any]],
    window: int = 3,
    valid: BoolArray | None = None,
) -> TrajectoryFeatures:
    """Compute completed-prefix features without crossing a supplied causal group."""

    matrix = _matrix(values)
    if window <= 0:
        raise ValueError("trajectory window must be positive")
    row_valid: BoolArray = np.asarray(np.isfinite(matrix).all(axis=1), dtype=np.bool_)
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != (len(matrix),):
            raise ValueError("valid mask differs from trajectory rows")
        row_valid &= supplied
    normalized_groups = _validated_groups(groups, len(matrix))
    differences = np.full_like(matrix, np.nan, dtype=np.float64)
    velocity = np.zeros(len(matrix), dtype=np.float64)
    acceleration = np.zeros(len(matrix), dtype=np.float64)
    path_length = np.zeros(len(matrix), dtype=np.float64)
    consistency = np.zeros(len(matrix), dtype=np.float64)

    for group in normalized_groups:
        local_steps: list[FloatArray] = []
        previous_velocity = 0.0
        previous_position: int | None = None
        for raw_position in group:
            position = int(raw_position)
            if (
                previous_position is None
                or not row_valid[position]
                or not row_valid[previous_position]
            ):
                local_steps = []
                previous_velocity = 0.0
                previous_position = position
                continue
            step = matrix[position] - matrix[previous_position]
            differences[position] = step
            local_steps.append(np.asarray(step, dtype=np.float64))
            local_steps = local_steps[-window:]
            norms = np.asarray([np.linalg.norm(value) for value in local_steps], dtype=float)
            velocity[position] = float(norms.mean()) if len(norms) else 0.0
            acceleration[position] = velocity[position] - previous_velocity
            path_length[position] = float(norms.sum())
            denominator = float(norms.sum())
            consistency[position] = (
                float(np.linalg.norm(np.sum(local_steps, axis=0)) / denominator)
                if denominator > 0.0
                else 0.0
            )
            previous_velocity = velocity[position]
            previous_position = position
    return TrajectoryFeatures(
        first_difference=differences,
        velocity=velocity,
        acceleration=acceleration,
        local_path_length=path_length,
        directional_consistency=np.clip(consistency, 0.0, 1.0),
    )


def diagonal_distance(current: FloatArray, origin: FloatArray, scale: FloatArray) -> float:
    """Return robust diagonally scaled Euclidean distance."""

    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(origin, dtype=np.float64)
    divisor = np.asarray(scale, dtype=np.float64)
    if left.shape != right.shape or left.shape != divisor.shape or left.ndim != 1:
        raise ValueError("diagonal distance arrays must share one-dimensional shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan
    if not np.isfinite(divisor).all() or np.any(divisor <= 0.0):
        raise ValueError("diagonal distance scales must be finite and positive")
    return float(np.linalg.norm((left - right) / divisor))


@dataclass(frozen=True, slots=True)
class ShrinkageMetric:
    """Development-fitted positive-definite Mahalanobis metric."""

    location: FloatArray
    precision: FloatArray
    shrinkage: float
    fit_row_count: int


def fit_shrinkage_metric(
    values: ArrayLike,
) -> ShrinkageMetric:
    """Fit deterministic Ledoit-Wolf covariance on complete development rows."""

    matrix = _matrix(values)
    complete = matrix[np.isfinite(matrix).all(axis=1)]
    if len(complete) < max(3, matrix.shape[1] + 1):
        raise ValueError("shrinkage metric has insufficient complete development rows")
    estimator = LedoitWolf(assume_centered=False).fit(complete)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    precision = (precision + precision.T) * 0.5
    if np.linalg.eigvalsh(precision).min() <= 0.0:
        raise AssertionError("shrinkage precision is not positive definite")
    return ShrinkageMetric(
        location=np.asarray(estimator.location_, dtype=np.float64),
        precision=precision,
        shrinkage=float(estimator.shrinkage_),
        fit_row_count=len(complete),
    )


def mahalanobis_distance(current: FloatArray, origin: FloatArray, precision: FloatArray) -> float:
    """Return nonnegative shrinkage-Mahalanobis distance."""

    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(origin, dtype=np.float64)
    matrix = np.asarray(precision, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or matrix.shape != (len(left), len(left)):
        raise ValueError("Mahalanobis arrays have inconsistent shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan
    squared = float((left - right) @ matrix @ (left - right))
    return math.sqrt(max(0.0, squared))


def _probability_vector(values: FloatArray) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise ValueError("posterior must be a finite probability vector")
    vector = np.clip(vector, 0.0, np.inf)
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("posterior has no probability mass")
    return vector / total


def jensen_shannon_distance(left: FloatArray, right: FloatArray) -> float:
    """Return the symmetric square-root Jensen-Shannon metric in natural-log units."""

    first = _probability_vector(left)
    second = _probability_vector(right)
    if first.shape != second.shape:
        raise ValueError("posterior vectors have different dimensions")
    with np.errstate(divide="ignore"):
        log_first = np.log(first)
        log_second = np.log(second)
    log_midpoint = np.logaddexp(log_first, log_second) - math.log(2.0)

    def divergence(source: FloatArray, log_source: FloatArray) -> float:
        positive = source > 0.0
        return float(np.sum(source[positive] * (log_source[positive] - log_midpoint[positive])))

    value = max(
        0.0,
        0.5 * divergence(first, log_first) + 0.5 * divergence(second, log_second),
    )
    return math.sqrt(value)


def posterior_entropy(probabilities: FloatArray) -> FloatArray:
    """Compute row-wise entropy after strict probability validation."""

    matrix = _matrix(probabilities)
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("posterior rows must normalize")
    clipped = np.clip(matrix, 1e-300, 1.0)
    return np.asarray(-np.sum(matrix * np.log(clipped), axis=1), dtype=np.float64)


def posterior_velocity(
    probabilities: FloatArray,
    *,
    groups: Sequence[NDArray[Any]],
) -> FloatArray:
    """Compute gap-local posterior Jensen-Shannon velocity."""

    matrix = _matrix(probabilities)
    normalized_groups = _validated_groups(groups, len(matrix))
    output = np.zeros(len(matrix), dtype=np.float64)
    for group in normalized_groups:
        for previous, current in zip(group[:-1], group[1:], strict=True):
            output[int(current)] = jensen_shannon_distance(
                matrix[int(previous)], matrix[int(current)]
            )
    return output


__all__ = [
    "SAFETY_FLAGS",
    "RobustGeometry",
    "ShrinkageMetric",
    "TrajectoryFeatures",
    "diagonal_distance",
    "fit_robust_geometry",
    "fit_shrinkage_metric",
    "jensen_shannon_distance",
    "mahalanobis_distance",
    "posterior_entropy",
    "posterior_velocity",
    "trajectory_features",
]
