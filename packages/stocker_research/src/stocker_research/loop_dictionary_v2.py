"""Versioned semantic identities for research-only structural loop events.

The V2 identity is derived from the closed state path, never from discovery
rank.  Rotations share a semantic identity while oriented paths remain
explicit metadata.  Reverse traversal is intentionally not declared
equivalent.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

ALLOWED_PRIMITIVE_TRANSITION_LENGTHS = frozenset({2, 3, 4, 5})
ALLOWED_COMPOSITE_TRANSITION_LENGTHS = frozenset({4, 5, 6, 7, 8})
MAX_EVENT_TRANSITIONS = 8

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


class MotifType(StrEnum):
    """Mutually exclusive structural type of a registered closed path."""

    PRIMITIVE = "primitive"
    REPEAT = "repeat"
    COMPOSITE = "composite"


class UnsupportedLoopError(ValueError):
    """Raised when a legacy or discovered path cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class LegacyCycleRecord:
    """One immutable legacy identity and its closed state path."""

    legacy_cycle_id: str
    closed_path: tuple[int, ...]
    discovery_rank: int


@dataclass(frozen=True, slots=True)
class LoopDefinition:
    """Semantic representation of one registered closed structural path."""

    semantic_loop_id: str
    primitive_loop_id: str | None
    motif_type: MotifType
    primitive_core: tuple[int, ...]
    primitive_transition_length: int
    repeat_depth: int
    full_core: tuple[int, ...]
    full_transition_length: int
    canonical_orientation: tuple[int, ...]
    oriented_paths: tuple[tuple[int, ...], ...]
    component_primitive_ids: tuple[str, ...]
    component_boundaries: tuple[tuple[int, int], ...]

    def orientation_id_for(self, closed_path: Sequence[int]) -> str:
        """Return a stable orientation ID without changing semantic identity."""

        candidate = tuple(int(state) for state in closed_path)
        if candidate not in self.oriented_paths:
            raise UnsupportedLoopError(
                f"{candidate!r} is not an orientation of {self.semantic_loop_id}"
            )
        route = "-".join(str(state) for state in candidate)
        return f"{self.semantic_loop_id}__o_{route}"


@dataclass(frozen=True, slots=True)
class DictionaryCandidateMetrics:
    """Structural support and information kept separate for V2 selection."""

    definition: LoopDefinition
    eligible_anchor_count: int
    observed_completions: int
    expected_completions: float
    empirical_p_value: float
    fdr_q_value: float
    conditional_information_gain: float
    increment_beyond_current_state: float
    increment_beyond_previous_state_history: float
    stock_breadth: int
    month_breadth: int
    clock_breadth: int
    period_consistency: float

    def __post_init__(self) -> None:
        if self.eligible_anchor_count <= 0:
            raise ValueError("eligible_anchor_count must be positive")
        if self.observed_completions < 0 or self.expected_completions < 0.0:
            raise ValueError("completion support cannot be negative")
        if not 0.0 <= self.empirical_p_value <= 1.0:
            raise ValueError("empirical_p_value must be in [0, 1]")
        if not 0.0 <= self.fdr_q_value <= 1.0:
            raise ValueError("fdr_q_value must be in [0, 1]")
        if not 0.0 <= self.period_consistency <= 1.0:
            raise ValueError("period_consistency must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PrimitiveLoop:
    """Typed primitive view used by consumers that reject larger motifs."""

    definition: LoopDefinition

    def __post_init__(self) -> None:
        if self.definition.motif_type is not MotifType.PRIMITIVE:
            raise TypeError("PrimitiveLoop requires a primitive definition")


@dataclass(frozen=True, slots=True)
class CompositeMotif:
    """Typed non-periodic composite view with explicit components."""

    definition: LoopDefinition

    def __post_init__(self) -> None:
        if self.definition.motif_type is not MotifType.COMPOSITE:
            raise TypeError("CompositeMotif requires a composite definition")


def _closed_path(core: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(state) for state in core)
    return values + (values[0],)


def _canonical_core(core: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(state) for state in core)
    return min(values[index:] + values[:index] for index in range(len(values)))


def _oriented_paths(core: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(state) for state in core)
    paths = {_closed_path(values[index:] + values[:index]) for index in range(len(values))}
    return tuple(sorted(paths))


def _validate_closed_path(closed_path: Sequence[int]) -> tuple[int, ...]:
    path = tuple(int(state) for state in closed_path)
    if len(path) < 3 or path[0] != path[-1]:
        raise UnsupportedLoopError("a loop path must be closed and contain two transitions")
    if any(state < 0 for state in path):
        raise UnsupportedLoopError("negative or unknown states cannot enter a dictionary")
    core = path[:-1]
    if len(core) > MAX_EVENT_TRANSITIONS:
        raise UnsupportedLoopError(f"transition length {len(core)} exceeds {MAX_EVENT_TRANSITIONS}")
    if any(left == right for left, right in zip(path[:-1], path[1:], strict=True)):
        raise UnsupportedLoopError("compressed loop paths cannot contain self transitions")
    return core


def _primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width:
            continue
        candidate = core[:width]
        depth = len(core) // width
        if candidate * depth == core:
            return candidate, depth
    return core, 1


def _component_decomposition(
    core: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Find a rotation that splits into sequential simple primitive loops.

    A lexicographically minimal rotation is not necessarily a valid component
    anchor.  We therefore inspect every rotation and deterministically choose
    the smallest rotation whose first-return segments are all primitive.  A
    repeated state with no such representation is ambiguous and fails closed
    in :func:`decompose_closed_path`.
    """

    rotations = sorted({core[index:] + core[:index] for index in range(len(core))})
    for oriented_core in rotations:
        anchor = oriented_core[0]
        boundaries = [0]
        boundaries.extend(
            index for index, state in enumerate(oriented_core[1:], start=1) if state == anchor
        )
        boundaries.append(len(oriented_core))
        if len(boundaries) <= 2:
            continue

        component_ids: list[str] = []
        component_bounds: list[tuple[int, int]] = []
        valid = True
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment_core = oriented_core[left:right]
            if len(segment_core) not in ALLOWED_PRIMITIVE_TRANSITION_LENGTHS or len(
                set(segment_core)
            ) != len(segment_core):
                valid = False
                break
            canonical_segment = _canonical_core(segment_core)
            component_ids.append(f"loop_p_{_path_text(_closed_path(canonical_segment))}")
            component_bounds.append((left, right))
        if valid and len(component_ids) >= 2:
            return oriented_core, tuple(component_ids), tuple(component_bounds)
    return (), (), ()


def _path_text(path: Sequence[int]) -> str:
    return "-".join(str(int(state)) for state in path)


def decompose_closed_path(closed_path: Sequence[int]) -> LoopDefinition:
    """Return the deterministic primitive/repeat/composite representation."""

    raw_core = _validate_closed_path(closed_path)
    canonical_full_core = _canonical_core(raw_core)
    root, repeat_depth = _primitive_root(canonical_full_core)
    canonical_root = _canonical_core(root)
    full_length = len(canonical_full_core)

    if repeat_depth > 1:
        if len(set(canonical_root)) != len(canonical_root):
            raise UnsupportedLoopError(
                "a periodic path whose root is composite is structurally ambiguous"
            )
        if len(canonical_root) not in ALLOWED_PRIMITIVE_TRANSITION_LENGTHS:
            raise UnsupportedLoopError("repeated path has an unsupported primitive root")
        primitive_path = _closed_path(canonical_root)
        primitive_id = f"loop_p_{_path_text(primitive_path)}"
        semantic_id = f"loop_r{repeat_depth}_{_path_text(primitive_path)}"
        motif_type = MotifType.REPEAT
        components: tuple[str, ...] = (primitive_id,) * repeat_depth
        component_boundaries = tuple(
            (index * len(canonical_root), (index + 1) * len(canonical_root))
            for index in range(repeat_depth)
        )
    else:
        composite_core, components, component_boundaries = _component_decomposition(
            canonical_full_core
        )
        if components:
            if full_length not in ALLOWED_COMPOSITE_TRANSITION_LENGTHS:
                raise UnsupportedLoopError("composite path has an unsupported length")
            canonical_full_core = composite_core
            canonical_path = _closed_path(canonical_full_core)
            digest = hashlib.sha256(_path_text(canonical_path).encode("ascii")).hexdigest()[:8]
            semantic_id = f"loop_c_{digest}"
            primitive_id = None
            canonical_root = ()
            motif_type = MotifType.COMPOSITE
        else:
            if len(set(canonical_full_core)) != len(canonical_full_core):
                raise UnsupportedLoopError(
                    "non-periodic composite has no unambiguous sequential primitive decomposition"
                )
            if full_length not in ALLOWED_PRIMITIVE_TRANSITION_LENGTHS:
                raise UnsupportedLoopError("primitive path has an unsupported length")
            primitive_path = _closed_path(canonical_full_core)
            primitive_id = f"loop_p_{_path_text(primitive_path)}"
            semantic_id = primitive_id
            motif_type = MotifType.PRIMITIVE

    canonical_orientation = _closed_path(canonical_full_core)
    return LoopDefinition(
        semantic_loop_id=semantic_id,
        primitive_loop_id=primitive_id,
        motif_type=motif_type,
        primitive_core=canonical_root,
        primitive_transition_length=len(canonical_root),
        repeat_depth=repeat_depth,
        full_core=canonical_full_core,
        full_transition_length=full_length,
        canonical_orientation=canonical_orientation,
        oriented_paths=_oriented_paths(canonical_full_core),
        component_primitive_ids=components,
        component_boundaries=component_boundaries,
    )


def loop_complexity_penalty(definition: LoopDefinition) -> float:
    """Deterministic description-length proxy with a primitive simplicity prior."""

    penalty = 0.05 * definition.full_transition_length
    if definition.motif_type is MotifType.REPEAT:
        penalty += 0.25 * (definition.repeat_depth - 1)
    elif definition.motif_type is MotifType.COMPOSITE:
        penalty += 0.40 + 0.10 * len(definition.component_primitive_ids)
    return float(penalty)


def candidate_selection_score(metrics: DictionaryCandidateMetrics) -> float:
    """Rank stable structural excess and information, not raw frequency."""

    excess = max(0.0, metrics.observed_completions - metrics.expected_completions)
    rate_ratio = (metrics.observed_completions + 0.5) / (metrics.expected_completions + 0.5)
    information = (
        metrics.conditional_information_gain
        + metrics.increment_beyond_current_state
        + metrics.increment_beyond_previous_state_history
    )
    breadth = (
        min(metrics.stock_breadth / 20.0, 1.0)
        + min(metrics.month_breadth / 12.0, 1.0)
        + min(metrics.clock_breadth / 3.0, 1.0)
    ) / 3.0
    significance = max(0.0, -math.log10(max(metrics.fdr_q_value, 1e-12)))
    return float(
        math.log1p(excess)
        + math.log(max(rate_ratio, 1e-12))
        + 10.0 * information
        + breadth
        + metrics.period_consistency
        + 0.10 * significance
        - loop_complexity_penalty(metrics.definition)
    )


def select_dictionary_candidates(
    candidates: Iterable[DictionaryCandidateMetrics], *, maximum_entries: int
) -> tuple[DictionaryCandidateMetrics, ...]:
    """Return a deterministic, primitive-closed structural score order.

    A repeat or composite cannot enter the event dictionary unless every
    component primitive is available.  Dependencies are inserted before the
    larger motif so its earlier primitive completion remains observable.
    """

    if maximum_entries <= 0:
        raise ValueError("maximum_entries must be positive")
    records = tuple(candidates)
    by_id = {item.definition.semantic_loop_id: item for item in records}
    ordered = sorted(
        records,
        key=lambda item: (
            -candidate_selection_score(item),
            item.definition.semantic_loop_id,
        ),
    )
    selected: list[DictionaryCandidateMetrics] = []
    selected_ids: set[str] = set()
    for item in ordered:
        semantic_id = item.definition.semantic_loop_id
        if semantic_id in selected_ids:
            continue
        dependency_ids = tuple(dict.fromkeys(item.definition.component_primitive_ids))
        if any(dependency not in by_id for dependency in dependency_ids):
            continue
        additions = [
            by_id[dependency] for dependency in dependency_ids if dependency not in selected_ids
        ]
        additions.append(item)
        additions = [
            addition
            for addition in additions
            if addition.definition.semantic_loop_id not in selected_ids
        ]
        if len(selected) + len(additions) > maximum_entries:
            continue
        for addition in additions:
            selected.append(addition)
            selected_ids.add(addition.definition.semantic_loop_id)
        if len(selected) == maximum_entries:
            break
    return tuple(selected)


class LoopDictionary:
    """Immutable semantic dictionary plus a separate legacy migration ledger."""

    def __init__(
        self,
        definitions: Mapping[str, LoopDefinition],
        migrations: Sequence[tuple[LegacyCycleRecord, LoopDefinition]],
        *,
        version: str,
    ) -> None:
        ordered = dict(sorted(definitions.items()))
        self._definitions: Mapping[str, LoopDefinition] = MappingProxyType(ordered)
        self._migrations = tuple(migrations)
        self.version = str(version)
        payload = [
            {
                "semantic_loop_id": definition.semantic_loop_id,
                "canonical_orientation": definition.canonical_orientation,
                "motif_type": definition.motif_type.value,
                "primitive_loop_id": definition.primitive_loop_id,
                "repeat_depth": definition.repeat_depth,
                "components": definition.component_primitive_ids,
            }
            for definition in ordered.values()
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.dictionary_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_legacy(cls, records: Iterable[LegacyCycleRecord], *, version: str) -> LoopDictionary:
        definitions: dict[str, LoopDefinition] = {}
        migrations: list[tuple[LegacyCycleRecord, LoopDefinition]] = []
        for record in records:
            definition = decompose_closed_path(record.closed_path)
            definitions.setdefault(definition.semantic_loop_id, definition)
            migrations.append((record, definition))
        if not definitions:
            raise UnsupportedLoopError("a loop dictionary cannot be empty")
        return cls(definitions, migrations, version=version)

    @classmethod
    def from_legacy_table(
        cls,
        table: Any,
        *,
        version: str,
        legacy_id_column: str = "legacy_cycle_id",
        path_column: str = "cycle",
        rank_column: str = "discovery_rank",
    ) -> LoopDictionary:
        """Read a legacy cycle table without overwriting its ranked identity."""

        columns = set(getattr(table, "columns", ()))
        required = {legacy_id_column, path_column, rank_column}
        missing = sorted(required.difference(columns))
        if missing:
            raise UnsupportedLoopError(f"legacy cycle table lacks columns: {missing}")
        records: list[LegacyCycleRecord] = []
        for row in table.itertuples(index=False):
            values = row._asdict()
            raw_path = values[path_column]
            if isinstance(raw_path, str):
                separator = "->" if "->" in raw_path else "-"
                try:
                    path = tuple(int(value.strip()) for value in raw_path.split(separator))
                except ValueError as error:
                    raise UnsupportedLoopError("legacy cycle path is not numeric") from error
            elif isinstance(raw_path, Sequence):
                path = tuple(int(value) for value in raw_path)
            else:
                raise UnsupportedLoopError("legacy cycle path has an unsupported representation")
            records.append(
                LegacyCycleRecord(
                    legacy_cycle_id=str(values[legacy_id_column]),
                    closed_path=path,
                    discovery_rank=int(values[rank_column]),
                )
            )
        return cls.from_legacy(records, version=version)

    @classmethod
    def from_definitions(
        cls, definitions: Iterable[LoopDefinition], *, version: str
    ) -> LoopDictionary:
        by_id = {definition.semantic_loop_id: definition for definition in definitions}
        if not by_id:
            raise UnsupportedLoopError("a loop dictionary cannot be empty")
        return cls(by_id, (), version=version)

    @property
    def definitions(self) -> Mapping[str, LoopDefinition]:
        return self._definitions

    @property
    def semantic_ids(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def migration_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for record, definition in sorted(
            self._migrations, key=lambda pair: pair[0].legacy_cycle_id
        ):
            rows.append(
                {
                    "legacy_cycle_id": record.legacy_cycle_id,
                    "legacy_cycle_path": "->".join(map(str, record.closed_path)),
                    "discovery_rank": record.discovery_rank,
                    "dictionary_version": self.version,
                    "semantic_loop_id": definition.semantic_loop_id,
                    "primitive_loop_id": definition.primitive_loop_id,
                    "motif_type": definition.motif_type.value,
                    "repeat_depth": definition.repeat_depth,
                    "component_loops": list(definition.component_primitive_ids),
                    "supported": True,
                    "migration_status": "migrated",
                    "ambiguity_reason": None,
                }
            )
        return rows


__all__ = [
    "ALLOWED_COMPOSITE_TRANSITION_LENGTHS",
    "ALLOWED_PRIMITIVE_TRANSITION_LENGTHS",
    "MAX_EVENT_TRANSITIONS",
    "CompositeMotif",
    "DictionaryCandidateMetrics",
    "LegacyCycleRecord",
    "LoopDefinition",
    "LoopDictionary",
    "MotifType",
    "PrimitiveLoop",
    "UnsupportedLoopError",
    "candidate_selection_score",
    "decompose_closed_path",
    "loop_complexity_penalty",
    "select_dictionary_candidates",
]
