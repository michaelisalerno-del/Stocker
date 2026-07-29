"""Causal origin-neighbourhood definitions for excursion research V1."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class OriginSurface:
    """One strictly trailing origin candidate per decision row."""

    centers: FloatArray
    eligible: BoolArray
    origin_ids: tuple[str, ...]
    window_bars: int
    definition_id: str

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.eligible.shape != (len(self.centers),):
            raise ValueError("origin surface arrays have inconsistent shapes")
        if len(self.origin_ids) != len(self.centers):
            raise ValueError("origin ID count differs from rows")
        if self.window_bars <= 0 or not self.definition_id:
            raise ValueError("origin surface metadata is invalid")
        if not bool(np.isfinite(self.centers[self.eligible]).all()):
            raise ValueError("eligible origins must be finite")


def _matrix(values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("origin values must be a nonempty feature matrix")
    return matrix


def _valid_mask(matrix: FloatArray, valid: BoolArray | None) -> BoolArray:
    result = np.isfinite(matrix).all(axis=1)
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != (len(matrix),):
            raise ValueError("origin valid mask differs from rows")
        result &= supplied
    return np.asarray(result, dtype=bool)


def _origin_id(
    *,
    definition_id: str,
    group_ordinal: int,
    position: int,
    window: int,
    center: FloatArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(definition_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{group_ordinal}:{position}:{window}".encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(center, dtype=np.float64).tobytes())
    return f"origin_{digest.hexdigest()[:24]}"


def _empty_surface(
    matrix: FloatArray,
    *,
    window: int,
    definition_id: str,
) -> tuple[FloatArray, BoolArray, list[str]]:
    if window <= 0:
        raise ValueError("origin window must be positive")
    if not definition_id:
        raise ValueError("origin definition ID cannot be empty")
    centers = np.full_like(matrix, np.nan, dtype=np.float64)
    eligible = np.zeros(len(matrix), dtype=bool)
    identifiers = [""] * len(matrix)
    return centers, eligible, identifiers


def trailing_robust_origins(
    values: ArrayLike,
    *,
    groups: Sequence[NDArray[Any]],
    window: int,
    valid: BoolArray | None = None,
    definition_id: str | None = None,
) -> OriginSurface:
    """Use the coordinatewise median of strictly previous completed rows."""

    matrix = _matrix(values)
    row_valid = _valid_mask(matrix, valid)
    identifier = definition_id or f"ORIGIN_A_W{window}"
    centers, eligible, identifiers = _empty_surface(
        matrix,
        window=window,
        definition_id=identifier,
    )
    assigned = np.zeros(len(matrix), dtype=bool)
    for group_ordinal, raw_group in enumerate(groups):
        group = np.asarray(raw_group, dtype=np.int64)
        if len(group) == 0:
            continue
        if group.min() < 0 or group.max() >= len(matrix) or np.any(np.diff(group) <= 0):
            raise ValueError("origin groups must be increasing and inside the input")
        if assigned[group].any():
            raise ValueError("origin groups overlap")
        assigned[group] = True
        for local_position in range(window, len(group)):
            position = int(group[local_position])
            trailing = group[local_position - window : local_position]
            if not row_valid[position] or not row_valid[trailing].all():
                continue
            center = np.median(matrix[trailing], axis=0)
            centers[position] = center
            eligible[position] = True
            identifiers[position] = _origin_id(
                definition_id=identifier,
                group_ordinal=group_ordinal,
                position=position,
                window=window,
                center=np.asarray(center, dtype=np.float64),
            )
    if len(matrix) and not assigned.all():
        raise ValueError("origin groups do not cover every row")
    return OriginSurface(
        centers=centers,
        eligible=eligible,
        origin_ids=tuple(identifiers),
        window_bars=window,
        definition_id=identifier,
    )


def locally_stable_origins(
    values: ArrayLike,
    *,
    groups: Sequence[NDArray[Any]],
    window: int,
    maximum_path_length: float,
    maximum_velocity: float,
    valid: BoolArray | None = None,
    definition_id: str = "ORIGIN_B_STABLE_W6",
) -> OriginSurface:
    """Admit a trailing median only when its completed path was locally stable."""

    if maximum_path_length < 0.0 or maximum_velocity < 0.0:
        raise ValueError("local-stability thresholds cannot be negative")
    matrix = _matrix(values)
    row_valid = _valid_mask(matrix, valid)
    centers, eligible, identifiers = _empty_surface(
        matrix,
        window=window,
        definition_id=definition_id,
    )
    assigned = np.zeros(len(matrix), dtype=bool)
    for group_ordinal, raw_group in enumerate(groups):
        group = np.asarray(raw_group, dtype=np.int64)
        if len(group) == 0:
            continue
        if group.min() < 0 or group.max() >= len(matrix) or np.any(np.diff(group) <= 0):
            raise ValueError("origin groups must be increasing and inside the input")
        if assigned[group].any():
            raise ValueError("origin groups overlap")
        assigned[group] = True
        for local_position in range(window, len(group)):
            position = int(group[local_position])
            trailing = group[local_position - window : local_position]
            if not row_valid[position] or not row_valid[trailing].all():
                continue
            local = matrix[trailing]
            steps = np.linalg.norm(np.diff(local, axis=0), axis=1)
            path_length = float(steps.sum())
            velocity = float(steps.mean()) if len(steps) else 0.0
            if path_length > maximum_path_length or velocity > maximum_velocity:
                continue
            center = np.median(local, axis=0)
            centers[position] = center
            eligible[position] = True
            identifiers[position] = _origin_id(
                definition_id=definition_id,
                group_ordinal=group_ordinal,
                position=position,
                window=window,
                center=np.asarray(center, dtype=np.float64),
            )
    if len(matrix) and not assigned.all():
        raise ValueError("origin groups do not cover every row")
    return OriginSurface(
        centers=centers,
        eligible=eligible,
        origin_ids=tuple(identifiers),
        window_bars=window,
        definition_id=definition_id,
    )


__all__ = ["OriginSurface", "locally_stable_origins", "trailing_robust_origins"]
