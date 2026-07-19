"""Deterministic causal orientation identities for primitive loop prefixes.

The identity always includes the registered route and prefix position.  It is
therefore unambiguous when a state appears more than once in a path.  This is
research-only structural metadata; it does not score future events or expose
an execution surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from stocker_research.regime_validity_v2 import safety_flags


@dataclass(frozen=True, slots=True)
class PrefixOrientation:
    """One causal position in one registered oriented primitive traversal."""

    primitive_loop_id: str
    oriented_path: tuple[int, ...]
    active_prefix: tuple[int, ...]
    prefix_position: int
    current_state: int
    required_next_state: int
    transitions_completed: int
    transitions_remaining: int
    prefix_progress: float
    traversal_direction: str
    orientation_id: str


def _validated_closed_path(path: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(state) for state in path)
    if len(result) < 3 or result[0] != result[-1]:
        raise ValueError("an oriented primitive path must be closed")
    if any(state < 0 for state in result):
        raise ValueError("orientation paths cannot contain unknown states")
    if any(left == right for left, right in zip(result[:-1], result[1:], strict=True)):
        raise ValueError("compressed orientation paths cannot contain self transitions")
    return result


def _route_token(path: Sequence[int]) -> str:
    return "-".join(str(int(state)) for state in path)


def orientation_for_prefix(
    *,
    primitive_loop_id: str,
    oriented_path: Sequence[int],
    active_prefix: Sequence[int],
) -> PrefixOrientation:
    """Describe an active prefix using only its observed states.

    A full closed traversal is a completion rather than an active prefix and
    is rejected.  Requiring the observed prefix to equal the registered route
    prefix prevents a future suffix from being smuggled into prefix progress.
    """

    if not primitive_loop_id.startswith("loop_p_"):
        raise ValueError("orientation requires a primitive semantic loop ID")
    route = _validated_closed_path(oriented_path)
    prefix = tuple(int(state) for state in active_prefix)
    if not prefix or len(prefix) >= len(route):
        raise ValueError("an active prefix must be nonempty and incomplete")
    if prefix != route[: len(prefix)]:
        raise ValueError("active prefix is not an observed prefix of the registered route")
    position = len(prefix) - 1
    transition_count = len(route) - 1
    current_state = route[position]
    required_next_state = route[position + 1]
    route_token = _route_token(route)
    orientation_id = (
        f"{primitive_loop_id}::route_{route_token}::position_{position}"
        f"_at_{current_state}_waiting_{required_next_state}"
    )
    return PrefixOrientation(
        primitive_loop_id=primitive_loop_id,
        oriented_path=route,
        active_prefix=prefix,
        prefix_position=position,
        current_state=current_state,
        required_next_state=required_next_state,
        transitions_completed=position,
        transitions_remaining=transition_count - position,
        prefix_progress=float(position / transition_count),
        traversal_direction=f"registered_route_{route_token}",
        orientation_id=orientation_id,
    )


def build_orientation_registry(
    oriented_paths_by_primitive: Mapping[str, Sequence[Sequence[int]]],
) -> pd.DataFrame:
    """Enumerate every causal prefix position without inspecting event outcomes."""

    rows: list[dict[str, object]] = []
    identities: set[str] = set()
    for primitive_loop_id in sorted(oriented_paths_by_primitive):
        routes = sorted(
            {
                _validated_closed_path(path)
                for path in oriented_paths_by_primitive[primitive_loop_id]
            }
        )
        for route in routes:
            for prefix_length in range(1, len(route)):
                orientation = orientation_for_prefix(
                    primitive_loop_id=primitive_loop_id,
                    oriented_path=route,
                    active_prefix=route[:prefix_length],
                )
                if orientation.orientation_id in identities:
                    raise AssertionError("orientation registry produced a duplicate identity")
                identities.add(orientation.orientation_id)
                rows.append(
                    {
                        "primitive_loop_id": orientation.primitive_loop_id,
                        "orientation_id": orientation.orientation_id,
                        "oriented_path": "->".join(map(str, orientation.oriented_path)),
                        "active_prefix": "->".join(map(str, orientation.active_prefix)),
                        "prefix_position": orientation.prefix_position,
                        "current_state": orientation.current_state,
                        "required_next_state": orientation.required_next_state,
                        "transitions_completed": orientation.transitions_completed,
                        "transitions_remaining": orientation.transitions_remaining,
                        "prefix_progress": orientation.prefix_progress,
                        "traversal_direction": orientation.traversal_direction,
                        **safety_flags(),
                    }
                )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "PrefixOrientation",
    "build_orientation_registry",
    "orientation_for_prefix",
]
