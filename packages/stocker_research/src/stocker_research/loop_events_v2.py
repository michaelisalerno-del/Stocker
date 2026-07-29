"""Typed contracts for causal first-next-loop event infrastructure V2.

All timestamps represent explicit source or availability times.  A provider
bar-start timestamp must never be treated as a completed-bar decision time;
callers supply the completed-bar availability timestamp separately.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from stocker_research.loop_dictionary_v2 import MotifType

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


class PrimaryOutcomeLabel(StrEnum):
    """Non-loop primary classes; a unique registered event uses its semantic ID."""

    UNAVAILABLE = "UNAVAILABLE"
    SESSION_END = "SESSION_END"
    TIED_REGISTERED_COMPLETION = "TIED_REGISTERED_COMPLETION"
    UNREGISTERED_LOOP = "UNREGISTERED_LOOP"
    NO_REGISTERED_LOOP_WITHIN_HORIZON = "NO_REGISTERED_LOOP_WITHIN_HORIZON"


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """Causal availability record for one feature value at one decision."""

    source_timestamp: datetime
    source_bar_ordinal: int | None
    available_timestamp: datetime
    decision_timestamp: datetime
    causal_valid: bool
    missing_reason: str | None
    source_field: str
    source_artifact_hash: str

    def __post_init__(self) -> None:
        if self.source_timestamp > self.decision_timestamp:
            raise ValueError("feature source_timestamp exceeds decision_timestamp")
        if self.available_timestamp < self.source_timestamp:
            raise ValueError("feature availability precedes its source timestamp")
        if self.available_timestamp > self.decision_timestamp:
            raise ValueError("feature is unavailable at the decision timestamp")
        if self.causal_valid and self.missing_reason is not None:
            raise ValueError("a causally valid feature cannot have a missing reason")
        if not self.causal_valid and self.missing_reason is None:
            raise ValueError("a causally invalid feature requires a missing reason")


@dataclass(frozen=True, slots=True)
class LoopPrefixState:
    """One registered loop prefix that is a suffix of observed state events."""

    semantic_loop_id: str
    primitive_loop_id: str | None
    orientation_id: str
    motif_type: MotifType
    repeat_depth: int
    prefix_path: tuple[int, ...]
    progress_states: int
    transitions_remaining: int
    start_event_index: int
    start_prefix_timestamp: datetime
    start_prefix_available_timestamp: datetime


@dataclass(frozen=True, slots=True)
class LoopCompletionEvent:
    """A detected completion, optionally enriched for a particular decision."""

    semantic_loop_id: str
    primitive_loop_id: str | None
    orientation_id: str
    motif_type: MotifType
    repeat_depth: int
    full_path: tuple[int, ...]
    start_event_index: int
    completion_event_index: int
    start_prefix_timestamp: datetime
    start_prefix_available_timestamp: datetime
    completion_state_event_timestamp: datetime
    completion_available_timestamp: datetime
    start_bar_ordinal: int
    completion_bar_ordinal: int
    state_events_until_completion: int
    decision_id: str | None = None
    symbol: str | None = None
    session: str | None = None
    decision_timestamp: datetime | None = None
    decision_available_timestamp: datetime | None = None
    transitions_remaining_at_decision: int | None = None
    bars_until_completion: int | None = None
    active_prefix_at_decision: tuple[int, ...] = ()
    initiated_before_decision: bool = False
    initiated_at_decision: bool = False
    initiated_after_decision: bool = False
    tied_completion: bool = False
    nested_completion: bool = False
    session_terminal: bool = False
    source_hashes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuralDecisionRow:
    """Core identity and causal timing for one completed-bar decision."""

    decision_id: str
    symbol: str
    session: str
    bar_ordinal: int
    bar_start_timestamp: datetime
    bar_complete_timestamp: datetime
    decision_timestamp: datetime
    state_model_version: str
    dictionary_version: str
    source_available: bool

    def __post_init__(self) -> None:
        if self.bar_complete_timestamp > self.decision_timestamp:
            raise ValueError("decision occurs before the bar completes")


@dataclass(frozen=True, slots=True)
class StructuralOutcomeRow:
    """Mutually exclusive primary result plus explicitly secondary labels."""

    decision_id: str
    primary_label: str
    tied_semantic_loop_ids: tuple[str, ...]
    earliest_registered_events: tuple[LoopCompletionEvent, ...]
    every_registered_completion_event: tuple[LoopCompletionEvent, ...]
    every_registered_completion_within_horizon: tuple[str, ...]
    earliest_primitive_completion: str | None
    earliest_repeated_completion: str | None
    earliest_composite_completion: str | None
    bars_until_completion: int | None
    state_events_until_completion: int | None
    transitions_remaining_at_decision: int | None
    first_event_was_open_prefix: bool
    first_event_began_after_decision: bool
    repeat_depth: int | None
    source_available: bool
    missing_reason: str | None = None


def safety_flags() -> Mapping[str, object]:
    """Return the mandatory safety payload used by every V2 artifact."""

    return {
        "research_only": RESEARCH_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "order_placement": ORDER_PLACEMENT,
        "broker_connected": BROKER_CONNECTED,
        "strategy_promotion": STRATEGY_PROMOTION,
    }


__all__ = [
    "FeatureProvenance",
    "LoopCompletionEvent",
    "LoopPrefixState",
    "PrimaryOutcomeLabel",
    "StructuralDecisionRow",
    "StructuralOutcomeRow",
    "safety_flags",
]
