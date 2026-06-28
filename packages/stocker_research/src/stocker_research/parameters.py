"""Parameter grid generation with hard sweep guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class ParameterSet:
    """One deterministic parameter set."""

    parameter_set_id: str
    params: dict[str, Any]


def generate_parameter_grid(
    parameter_space: dict[str, list[Any]],
    *,
    max_size: int = 100,
) -> list[ParameterSet]:
    """Generate a deterministic grid with a maximum-size guardrail."""

    if not parameter_space:
        raise ValueError("parameter_space must not be empty")
    keys = sorted(parameter_space)
    total = 1
    for key in keys:
        choices = parameter_space[key]
        if not choices:
            raise ValueError(f"parameter_space entry is empty: {key}")
        total *= len(choices)
    if total > max_size:
        raise ValueError(f"parameter grid size {total} exceeds max_size {max_size}")

    grid: list[ParameterSet] = []
    for index, values in enumerate(product(*(parameter_space[key] for key in keys)), start=1):
        grid.append(
            ParameterSet(
                parameter_set_id=f"ps_{index:04d}",
                params=dict(zip(keys, values, strict=True)),
            )
        )
    return grid
