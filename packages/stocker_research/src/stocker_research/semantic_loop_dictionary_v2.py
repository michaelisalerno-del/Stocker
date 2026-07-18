"""Primitive-first semantic identities for structural loop research V2.

This module is an isolated compatibility layer over the frozen Loop Event
Semantics V2 lineage.  It never changes the historical dictionary.  Closed
paths are reduced into deterministic primitive components; repeats and
composites remain auxiliary motifs rather than primary forecast classes.

Safety boundary: research only.  Economic outcomes are not accepted, execution
is disabled, order placement is disabled, and no broker is connected.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import pandas as pd

PRIMARY_PRIMITIVE_TRANSITION_LENGTHS = frozenset({2, 3, 4, 5})
SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS = frozenset({6, 7, 8})
ALL_SUPPORTED_PRIMITIVE_TRANSITION_LENGTHS = (
    PRIMARY_PRIMITIVE_TRANSITION_LENGTHS | SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS
)

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False


class SemanticMotifType(StrEnum):
    """Mutually exclusive description of one observed closed path."""

    PRIMITIVE = "primitive"
    REPEAT = "repeat"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class SemanticPathIdentity:
    """Primitive-first identity plus complete auxiliary motif metadata."""

    full_closed_path: tuple[int, ...]
    open_core: tuple[int, ...]
    transition_length: int
    canonical_primitive_core: tuple[int, ...]
    primitive_transition_length: int
    primitive_loop_id: str
    semantic_motif_id: str
    motif_type: SemanticMotifType
    repeat_depth: int
    component_primitive_ids: tuple[str, ...]
    component_boundaries: tuple[tuple[int, int], ...]
    orientation: tuple[int, ...]
    allowed_orientations: tuple[tuple[int, ...], ...]
    reverse_path_id: str
    primary_class_eligible: bool

    @property
    def semantic_loop_id(self) -> str:
        """Return the primitive identity used by the mutually exclusive target."""

        return self.primitive_loop_id


@dataclass(frozen=True, slots=True)
class CandidateSupportGates:
    """Development-only breadth and concentration gates frozen before scoring."""

    minimum_occurrences: int = 100
    minimum_sessions: int = 50
    minimum_stocks: int = 10
    minimum_months: int = 6
    minimum_clock_phases: int = 3
    maximum_top_stock_share: float = 0.20
    maximum_top_month_share: float = 0.30


@dataclass(frozen=True, slots=True)
class CandidateUniverseBundle:
    """Candidate identities, support statistics, and explicit rejection reasons."""

    universe: pd.DataFrame
    support: pd.DataFrame
    rejections: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DictionarySelectionBundle:
    """Frozen selected dictionary and its deterministic forward-selection path."""

    dictionary: pd.DataFrame
    selection_path: pd.DataFrame


def safety_flags() -> dict[str, object]:
    """Return the complete mandatory research boundary."""

    return {
        "research_only": RESEARCH_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "order_placement": ORDER_PLACEMENT,
        "broker_connected": BROKER_CONNECTED,
        "economic_outcomes_used": ECONOMIC_OUTCOMES_USED,
        "payoff_selection_used": PAYOFF_SELECTION_USED,
        "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
        "strategy_promotion": STRATEGY_PROMOTION,
    }


def canonical_rotation(core: Sequence[int]) -> tuple[int, ...]:
    """Return the lexicographically smallest rotation without reversing direction."""

    values = tuple(int(state) for state in core)
    if len(values) < 2:
        raise ValueError("a primitive core requires at least two states")
    return min(values[index:] + values[:index] for index in range(len(values)))


def _closed(core: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(state) for state in core)
    return values + (values[0],)


def _path_text(path: Sequence[int]) -> str:
    return "-".join(str(int(state)) for state in path)


def semantic_primitive_id(core: Sequence[int]) -> str:
    """Create a readable rank-independent primitive ID."""

    canonical = canonical_rotation(core)
    return f"loop_p_{_path_text(_closed(canonical))}"


def _primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width:
            continue
        candidate = core[:width]
        depth = len(core) // width
        if candidate * depth == core:
            return candidate, depth
    return core, 1


def _oriented_paths(core: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(state) for state in core)
    return tuple(sorted({_closed(values[index:] + values[:index]) for index in range(len(values))}))


def _validate_closed_path(path: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(state) for state in path)
    if len(values) < 3 or values[0] != values[-1]:
        raise ValueError("a structural loop must be closed and contain at least two transitions")
    if any(state < 0 for state in values):
        raise ValueError("unknown or negative states cannot enter semantic identity")
    if any(left == right for left, right in zip(values[:-1], values[1:], strict=True)):
        raise ValueError("compressed structural paths cannot contain a self transition")
    return values


def _stack_components(
    path: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """Decompose nested or sequential cycles using an independent unique-state stack."""

    stack_states: list[int] = []
    stack_positions: list[int] = []
    components: list[tuple[int, ...]] = []
    boundaries: list[tuple[int, int]] = []
    for position, state in enumerate(path):
        if state not in stack_states:
            stack_states.append(state)
            stack_positions.append(position)
            continue
        stack_index = stack_states.index(state)
        component = tuple(stack_states[stack_index:] + [state])
        if len(component) < 3:
            raise AssertionError("adjacent duplicate state escaped path validation")
        components.append(component)
        boundaries.append((stack_positions[stack_index], position))
        stack_states = stack_states[:stack_index] + [state]
        stack_positions = stack_positions[:stack_index] + [position]
    if not components:
        raise AssertionError("a validated closed path produced no primitive component")
    return tuple(components), tuple(boundaries)


def decompose_semantic_path(path: Sequence[int]) -> SemanticPathIdentity:
    """Reduce a closed route to one final primitive plus auxiliary motif metadata.

    The exact full route remains available.  A unique-state stack independently
    extracts nested components in causal order.  The component that closes at
    the final event supplies the primary primitive identity; earlier components
    are metadata.  Exact periodic traversal sets repeat depth, while any other
    path with more than one component is a composite.
    """

    closed_path = _validate_closed_path(path)
    core = closed_path[:-1]
    periodic_root, periodic_depth = _primitive_root(core)
    component_paths, boundaries = _stack_components(closed_path)
    component_ids = tuple(semantic_primitive_id(component[:-1]) for component in component_paths)
    final_component = component_paths[-1]
    final_core = canonical_rotation(final_component[:-1])
    primitive_id = semantic_primitive_id(final_core)

    exact_repeat = periodic_depth > 1 and len(set(periodic_root)) == len(periodic_root)
    if exact_repeat:
        motif_type = SemanticMotifType.REPEAT
        repeat_depth = periodic_depth
        motif_id = f"loop_r{repeat_depth}_{_path_text(_closed(canonical_rotation(periodic_root)))}"
    elif len(component_paths) == 1 and len(set(core)) == len(core):
        motif_type = SemanticMotifType.PRIMITIVE
        repeat_depth = 1
        motif_id = primitive_id
    else:
        motif_type = SemanticMotifType.COMPOSITE
        repeat_depth = 1
        digest = hashlib.sha256(_path_text(closed_path).encode("ascii")).hexdigest()[:12]
        motif_id = f"loop_c_{digest}"

    reverse_core = canonical_rotation(tuple(reversed(final_core)))
    return SemanticPathIdentity(
        full_closed_path=closed_path,
        open_core=core,
        transition_length=len(core),
        canonical_primitive_core=final_core,
        primitive_transition_length=len(final_core),
        primitive_loop_id=primitive_id,
        semantic_motif_id=motif_id,
        motif_type=motif_type,
        repeat_depth=repeat_depth,
        component_primitive_ids=component_ids,
        component_boundaries=boundaries,
        orientation=final_component,
        allowed_orientations=_oriented_paths(final_core),
        reverse_path_id=semantic_primitive_id(reverse_core),
        primary_class_eligible=motif_type is SemanticMotifType.PRIMITIVE,
    )


def _primitive_path_from_id(primitive_loop_id: str) -> tuple[int, ...]:
    prefix = "loop_p_"
    if not primitive_loop_id.startswith(prefix):
        raise ValueError(f"not a primitive semantic ID: {primitive_loop_id}")
    try:
        path = tuple(int(value) for value in primitive_loop_id.removeprefix(prefix).split("-"))
    except ValueError as error:
        raise ValueError(f"invalid primitive semantic ID: {primitive_loop_id}") from error
    _validate_closed_path(path)
    return path


def _top_share(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.value_counts(dropna=False).iloc[0] / len(values))


def _candidate_reasons(row: Mapping[str, Any], gates: CandidateSupportGates) -> list[str]:
    reasons: list[str] = []
    comparisons = (
        (int(row["development_count"]) < gates.minimum_occurrences, "below_minimum_occurrences"),
        (int(row["session_breadth"]) < gates.minimum_sessions, "below_minimum_sessions"),
        (int(row["stock_breadth"]) < gates.minimum_stocks, "below_minimum_stocks"),
        (int(row["month_breadth"]) < gates.minimum_months, "below_minimum_months"),
        (
            int(row["clock_breadth"]) < gates.minimum_clock_phases,
            "below_minimum_clock_phases",
        ),
        (
            float(row["top_stock_share"]) > gates.maximum_top_stock_share,
            "top_stock_share_above_maximum",
        ),
        (
            float(row["top_month_share"]) > gates.maximum_top_month_share,
            "top_month_share_above_maximum",
        ),
        (not bool(row["complete_semantic_identity"]), "incomplete_semantic_identity"),
        (not bool(row["source_gap_free"]), "source_gap_ambiguity"),
    )
    reasons.extend(reason for rejected, reason in comparisons if rejected)
    if not bool(row["primary_length_eligible"]):
        length = int(row["transition_length"])
        reasons.append(
            "sensitivity_only_transition_length"
            if length in SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS
            else "unsupported_transition_length"
        )
    return reasons


def build_candidate_universe(
    first_events: pd.DataFrame,
    *,
    gates: CandidateSupportGates,
) -> CandidateUniverseBundle:
    """Build the development-only primitive candidate census without outcome data."""

    required = {
        "decision_id",
        "symbol",
        "session",
        "decision_timestamp",
        "clock_phase",
        "primitive_loop_id",
        "primitive_transition_length",
        "motif_type",
        "source_completeness",
    }
    missing = required.difference(first_events.columns)
    if missing:
        raise ValueError(f"candidate census missing fields: {sorted(missing)}")
    events = first_events.loc[first_events["primitive_loop_id"].notna(), sorted(required)].copy()
    if events.empty:
        columns = [
            "semantic_loop_id",
            "primitive_loop_id",
            "transition_length",
            "development_count",
            "support_pass",
        ]
        empty = pd.DataFrame(columns=columns)
        return CandidateUniverseBundle(
            universe=empty.copy(),
            support=empty.copy(),
            rejections=pd.DataFrame(columns=["primitive_loop_id", "rejection_reason"]),
        )
    timestamps = pd.to_datetime(events["decision_timestamp"], utc=True)
    events["_month"] = timestamps.dt.strftime("%Y-%m")
    events["_quarter"] = (
        timestamps.dt.year.astype(str) + "Q" + (((timestamps.dt.month - 1) // 3) + 1).astype(str)
    )
    records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    for primitive_id, group in events.groupby("primitive_loop_id", sort=True, dropna=False):
        primitive_loop_id = str(primitive_id)
        parsed_path = _primitive_path_from_id(primitive_loop_id)
        lengths = set(int(value) for value in group["primitive_transition_length"].dropna())
        identity_complete = len(lengths) == 1 and len(parsed_path) - 1 in lengths
        transition_length = next(iter(lengths)) if len(lengths) == 1 else len(parsed_path) - 1
        stock_session = group["symbol"].astype(str) + "|" + group["session"].astype(str)
        source_gap_free = bool(group["source_completeness"].fillna(False).all())
        row: dict[str, Any] = {
            "semantic_loop_id": primitive_loop_id,
            "primitive_loop_id": primitive_loop_id,
            "canonical_primitive_core": list(parsed_path[:-1]),
            "closed_path": list(parsed_path),
            "transition_length": transition_length,
            "allowed_orientations": [list(path) for path in _oriented_paths(parsed_path[:-1])],
            "reverse_path_id": semantic_primitive_id(tuple(reversed(parsed_path[:-1]))),
            "motif_type": SemanticMotifType.PRIMITIVE.value,
            "development_count": int(group["decision_id"].nunique()),
            "stock_breadth": int(group["symbol"].nunique()),
            "session_breadth": int(stock_session.nunique()),
            "month_breadth": int(group["_month"].nunique()),
            "quarter_breadth": int(group["_quarter"].nunique()),
            "clock_breadth": int(group["clock_phase"].nunique()),
            "top_stock_share": _top_share(group["symbol"]),
            "top_month_share": _top_share(group["_month"]),
            "complete_semantic_identity": identity_complete,
            "source_gap_free": source_gap_free,
            "primary_length_eligible": transition_length in PRIMARY_PRIMITIVE_TRANSITION_LENGTHS,
            "sensitivity_length_eligible": transition_length
            in SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS,
            "selection_period": "development",
        }
        reasons = _candidate_reasons(row, gates)
        row["support_pass"] = not reasons
        records.append(row)
        rejection_records.extend(
            {
                "primitive_loop_id": primitive_loop_id,
                "semantic_loop_id": primitive_loop_id,
                "rejection_reason": reason,
            }
            for reason in reasons
        )
    support = (
        pd.DataFrame.from_records(records).sort_values("semantic_loop_id").reset_index(drop=True)
    )
    rejections = pd.DataFrame.from_records(
        rejection_records,
        columns=["primitive_loop_id", "semantic_loop_id", "rejection_reason"],
    )
    return CandidateUniverseBundle(universe=support.copy(), support=support, rejections=rejections)


def select_primary_dictionary(
    candidates: pd.DataFrame,
    *,
    total_valid_primitive_events: int,
    maximum_entries: int = 32,
    minimum_marginal_coverage: float = 0.005,
) -> DictionarySelectionBundle:
    """Apply the frozen deterministic development-only forward selection."""

    if total_valid_primitive_events <= 0:
        raise ValueError("total_valid_primitive_events must be positive")
    if maximum_entries < 1 or maximum_entries > 32:
        raise ValueError("maximum_entries must be between one and the frozen maximum 32")
    if not 0.0 <= minimum_marginal_coverage <= 1.0:
        raise ValueError("minimum_marginal_coverage must be within [0, 1]")
    required = {
        "semantic_loop_id",
        "primitive_loop_id",
        "motif_type",
        "transition_length",
        "development_count",
        "support_pass",
        "structurally_qualified",
        "information_qualified",
        "oof_log_loss_increment",
        "semi_markov_rate_ratio",
        "stock_breadth",
        "month_breadth",
        "selection_period",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"dictionary selection missing fields: {sorted(missing)}")
    if not candidates["selection_period"].eq("development").all():
        raise ValueError("dictionary selection may use development candidates only")
    eligible = candidates.loc[
        candidates["support_pass"].astype(bool)
        & candidates["structurally_qualified"].astype(bool)
        & candidates["information_qualified"].astype(bool)
        & candidates["motif_type"].eq(SemanticMotifType.PRIMITIVE.value)
    ].copy()
    eligible["marginal_coverage"] = (
        eligible["development_count"].astype(float) / total_valid_primitive_events
    )
    eligible = eligible.sort_values(
        [
            "oof_log_loss_increment",
            "semi_markov_rate_ratio",
            "marginal_coverage",
            "stock_breadth",
            "month_breadth",
            "transition_length",
            "semantic_loop_id",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_indices: list[int] = []
    path_records: list[dict[str, Any]] = []
    cumulative_coverage = 0.0
    for index, row in eligible.iterrows():
        marginal = float(row["marginal_coverage"])
        if len(selected_indices) >= maximum_entries:
            path_records.append(
                {
                    "selection_step": len(selected_indices) + 1,
                    "semantic_loop_id": row["semantic_loop_id"],
                    "marginal_coverage": marginal,
                    "cumulative_coverage": cumulative_coverage,
                    "selection_action": "STOP_MAXIMUM_DICTIONARY_SIZE",
                }
            )
            break
        if marginal < minimum_marginal_coverage:
            path_records.append(
                {
                    "selection_step": len(selected_indices) + 1,
                    "semantic_loop_id": row["semantic_loop_id"],
                    "marginal_coverage": marginal,
                    "cumulative_coverage": cumulative_coverage,
                    "selection_action": "STOP_BELOW_MARGINAL_COVERAGE",
                }
            )
            break
        selected_indices.append(int(cast(Any, index)))
        cumulative_coverage += marginal
        path_records.append(
            {
                "selection_step": len(selected_indices),
                "semantic_loop_id": row["semantic_loop_id"],
                "marginal_coverage": marginal,
                "cumulative_coverage": cumulative_coverage,
                "selection_action": "SELECT",
            }
        )
    selected = eligible.loc[selected_indices].copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["discovery_rank"] = selected["selection_rank"]
    selected["selection_reason"] = "passed_support_null_information_and_forward_selection"
    selected["selected"] = True
    selected["rejection_reason"] = ""
    return DictionarySelectionBundle(
        dictionary=selected.reset_index(drop=True),
        selection_path=pd.DataFrame.from_records(
            path_records,
            columns=[
                "selection_step",
                "semantic_loop_id",
                "marginal_coverage",
                "cumulative_coverage",
                "selection_action",
            ],
        ),
    )


def _normalise_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalise_identity_value(item) for item in value]
    if hasattr(value, "item"):
        return _normalise_identity_value(value.item())
    return str(value)


def deterministic_dictionary_hash(
    entries: Iterable[Mapping[str, Any]] | Any,
    *,
    dictionary_version: str = "semantic_loop_dictionary_first_event_v2",
) -> str:
    """Hash semantic membership and paths while deliberately excluding rank/metrics."""

    records = entries.to_dict(orient="records") if hasattr(entries, "to_dict") else list(entries)
    identity_fields = (
        "semantic_loop_id",
        "primitive_loop_id",
        "canonical_primitive_core",
        "canonical_primitive_path",
        "closed_path",
        "transition_length",
        "allowed_orientations",
        "reverse_path_id",
    )
    payload = []
    for raw in records:
        row = dict(raw)
        identity = {
            field: _normalise_identity_value(row[field])
            for field in identity_fields
            if field in row
        }
        if "semantic_loop_id" not in identity:
            raise ValueError("dictionary entries require semantic_loop_id")
        payload.append(identity)
    payload.sort(key=lambda row: str(row["semantic_loop_id"]))
    encoded = json.dumps(
        {"dictionary_version": dictionary_version, "entries": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ALL_SUPPORTED_PRIMITIVE_TRANSITION_LENGTHS",
    "CandidateSupportGates",
    "CandidateUniverseBundle",
    "DictionarySelectionBundle",
    "PRIMARY_PRIMITIVE_TRANSITION_LENGTHS",
    "SENSITIVITY_PRIMITIVE_TRANSITION_LENGTHS",
    "SemanticMotifType",
    "SemanticPathIdentity",
    "build_candidate_universe",
    "canonical_rotation",
    "decompose_semantic_path",
    "deterministic_dictionary_hash",
    "safety_flags",
    "select_primary_dictionary",
    "semantic_primitive_id",
]
