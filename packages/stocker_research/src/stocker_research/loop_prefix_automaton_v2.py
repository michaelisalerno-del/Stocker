"""Deterministic prefix automaton and first-next-loop event resolver V2.

The automaton consumes causal *state events*, not bars.  Repeated bars in the
same hard state do not advance it.  Session boundaries and missing state
events reset it, so no structural prefix can cross an unobserved gap.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from stocker_research.loop_dictionary_v2 import (
    MAX_EVENT_TRANSITIONS,
    LoopDefinition,
    LoopDictionary,
    MotifType,
)
from stocker_research.loop_events_v2 import (
    LoopCompletionEvent,
    LoopPrefixState,
    PrimaryOutcomeLabel,
    StructuralOutcomeRow,
)

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


class UnknownStateError(ValueError):
    """Raised after clearing the automaton when a state is outside its contract."""


@dataclass(frozen=True, slots=True)
class StateEventRecord:
    event_index: int
    state: int
    bar_ordinal: int
    event_timestamp: datetime
    available_timestamp: datetime


@dataclass(frozen=True, slots=True)
class UnregisteredLoopCompletion:
    full_path: tuple[int, ...]
    start_event_index: int
    completion_event_index: int
    start_timestamp: datetime
    completion_timestamp: datetime
    start_bar_ordinal: int
    completion_bar_ordinal: int


@dataclass(frozen=True, slots=True)
class EventTrace:
    state_events: tuple[StateEventRecord, ...]
    registered_completions: tuple[LoopCompletionEvent, ...]
    unregistered_completions: tuple[UnregisteredLoopCompletion, ...]
    prefixes_after_event: tuple[tuple[LoopPrefixState, ...], ...]


@dataclass(frozen=True, slots=True)
class _PatternBinding:
    definition: LoopDefinition
    full_path: tuple[int, ...]
    orientation_id: str


@dataclass(slots=True)
class _TrieNode:
    sequence: tuple[int, ...]
    transitions: dict[int, int] = field(default_factory=dict)
    failure: int = 0
    prefix_bindings: list[_PatternBinding] = field(default_factory=list)
    outputs: list[_PatternBinding] = field(default_factory=list)


class LoopEventAutomaton:
    """Aho-Corasick-style matcher over every registered oriented loop path."""

    def __init__(self, dictionary: LoopDictionary, *, allowed_states: frozenset[int]) -> None:
        if not allowed_states:
            raise ValueError("allowed_states cannot be empty")
        self.dictionary = dictionary
        self.allowed_states = allowed_states
        self._nodes = [_TrieNode(())]
        self._install_patterns(dictionary.definitions.values())
        self._build_failures()
        self._node_index = 0
        self._events: list[StateEventRecord] = []

    def _install_patterns(self, definitions: Iterable[LoopDefinition]) -> None:
        for definition in sorted(definitions, key=lambda item: item.semantic_loop_id):
            for path in definition.oriented_paths:
                binding = _PatternBinding(
                    definition=definition,
                    full_path=path,
                    orientation_id=definition.orientation_id_for(path),
                )
                node_index = 0
                for state in path:
                    node = self._nodes[node_index]
                    if state not in node.transitions:
                        node.transitions[state] = len(self._nodes)
                        self._nodes.append(_TrieNode(node.sequence + (state,)))
                    node_index = node.transitions[state]
                    self._nodes[node_index].prefix_bindings.append(binding)
                self._nodes[node_index].outputs.append(binding)

    def _build_failures(self) -> None:
        queue: deque[int] = deque()
        for child in self._nodes[0].transitions.values():
            self._nodes[child].failure = 0
            queue.append(child)
        while queue:
            parent_index = queue.popleft()
            for state, child_index in self._nodes[parent_index].transitions.items():
                queue.append(child_index)
                fallback = self._nodes[parent_index].failure
                while fallback and state not in self._nodes[fallback].transitions:
                    fallback = self._nodes[fallback].failure
                self._nodes[child_index].failure = self._nodes[fallback].transitions.get(state, 0)
                inherited = self._nodes[self._nodes[child_index].failure].outputs
                for binding in inherited:
                    if binding not in self._nodes[child_index].outputs:
                        self._nodes[child_index].outputs.append(binding)

    def _clear(self) -> None:
        self._node_index = 0
        self._events.clear()

    def reset_session(self) -> None:
        """Clear every prefix at an explicit regular-session boundary."""

        self._clear()

    def mark_missing(self) -> None:
        """Clear every prefix when a state event is incomplete or ambiguous."""

        self._clear()

    def feed(
        self,
        state: int,
        *,
        event_timestamp: datetime,
        available_timestamp: datetime,
        bar_ordinal: int,
    ) -> tuple[LoopCompletionEvent, ...]:
        """Advance on one causal state event and return all same-event completions."""

        value = int(state)
        if value not in self.allowed_states:
            self._clear()
            raise UnknownStateError(f"state {value} is outside the frozen state set")
        if available_timestamp < event_timestamp:
            raise ValueError("state event availability precedes its source timestamp")
        if self._events and self._events[-1].state == value:
            return ()

        while self._node_index and value not in self._nodes[self._node_index].transitions:
            self._node_index = self._nodes[self._node_index].failure
        self._node_index = self._nodes[self._node_index].transitions.get(value, 0)
        event = StateEventRecord(
            event_index=len(self._events),
            state=value,
            bar_ordinal=int(bar_ordinal),
            event_timestamp=event_timestamp,
            available_timestamp=available_timestamp,
        )
        self._events.append(event)

        detected: dict[tuple[str, str], LoopCompletionEvent] = {}
        for binding in self._nodes[self._node_index].outputs:
            width = len(binding.full_path)
            start_index = event.event_index - width + 1
            if start_index < 0:
                continue
            start = self._events[start_index]
            key = (binding.definition.semantic_loop_id, binding.orientation_id)
            detected[key] = LoopCompletionEvent(
                semantic_loop_id=binding.definition.semantic_loop_id,
                primitive_loop_id=binding.definition.primitive_loop_id,
                orientation_id=binding.orientation_id,
                motif_type=binding.definition.motif_type,
                repeat_depth=binding.definition.repeat_depth,
                full_path=binding.full_path,
                start_event_index=start_index,
                completion_event_index=event.event_index,
                start_prefix_timestamp=start.event_timestamp,
                start_prefix_available_timestamp=start.available_timestamp,
                completion_state_event_timestamp=event.event_timestamp,
                completion_available_timestamp=event.available_timestamp,
                start_bar_ordinal=start.bar_ordinal,
                completion_bar_ordinal=event.bar_ordinal,
                state_events_until_completion=width - 1,
            )
        events = sorted(
            detected.values(), key=lambda item: (len(item.full_path), item.semantic_loop_id)
        )
        tied = len({item.semantic_loop_id for item in events}) > 1
        enriched: list[LoopCompletionEvent] = []
        for item in events:
            nested = any(
                len(other.full_path) < len(item.full_path)
                and item.full_path[-len(other.full_path) :] == other.full_path
                for other in events
            )
            enriched.append(replace(item, tied_completion=tied, nested_completion=nested))
        return tuple(enriched)

    def active_prefixes(self) -> tuple[LoopPrefixState, ...]:
        """Return every registered proper prefix matching an observed suffix."""

        if not self._events:
            return ()
        node_indices: list[int] = []
        cursor = self._node_index
        while cursor:
            node_indices.append(cursor)
            cursor = self._nodes[cursor].failure
        prefixes: dict[tuple[str, str, tuple[int, ...]], LoopPrefixState] = {}
        for node_index in node_indices:
            node = self._nodes[node_index]
            progress = len(node.sequence)
            if progress == 0:
                continue
            start_index = len(self._events) - progress
            if start_index < 0:
                continue
            start = self._events[start_index]
            for binding in node.prefix_bindings:
                if progress >= len(binding.full_path):
                    continue
                key = (
                    binding.definition.semantic_loop_id,
                    binding.orientation_id,
                    node.sequence,
                )
                prefixes[key] = LoopPrefixState(
                    semantic_loop_id=binding.definition.semantic_loop_id,
                    primitive_loop_id=binding.definition.primitive_loop_id,
                    orientation_id=binding.orientation_id,
                    motif_type=binding.definition.motif_type,
                    repeat_depth=binding.definition.repeat_depth,
                    prefix_path=node.sequence,
                    progress_states=progress,
                    transitions_remaining=len(binding.full_path) - progress,
                    start_event_index=start_index,
                    start_prefix_timestamp=start.event_timestamp,
                    start_prefix_available_timestamp=start.available_timestamp,
                )
        return tuple(
            sorted(
                prefixes.values(),
                key=lambda item: (
                    item.semantic_loop_id,
                    item.orientation_id,
                    item.progress_states,
                ),
            )
        )


class FirstNextLoopEventEngine:
    """Resolve mutually exclusive first structural outcomes from one event trace."""

    def __init__(self, dictionary: LoopDictionary, *, allowed_states: frozenset[int]) -> None:
        self.dictionary = dictionary
        self.allowed_states = allowed_states

    def scan_state_events(
        self,
        states: Sequence[int],
        *,
        bar_ordinals: Sequence[int],
        event_timestamps: Sequence[datetime],
        available_timestamps: Sequence[datetime],
    ) -> EventTrace:
        lengths = {
            len(states),
            len(bar_ordinals),
            len(event_timestamps),
            len(available_timestamps),
        }
        if len(lengths) != 1:
            raise ValueError("state-event inputs have different lengths")
        if any(left == right for left, right in zip(states[:-1], states[1:], strict=True)):
            raise ValueError("state-event input contains an adjacent duplicate state")
        automaton = LoopEventAutomaton(self.dictionary, allowed_states=self.allowed_states)
        registered: list[LoopCompletionEvent] = []
        prefixes: list[tuple[LoopPrefixState, ...]] = []
        raw_events: list[StateEventRecord] = []
        unregistered: list[UnregisteredLoopCompletion] = []
        registered_paths = {
            path
            for definition in self.dictionary.definitions.values()
            for path in definition.oriented_paths
        }
        for state, bar, timestamp, available in zip(
            states,
            bar_ordinals,
            event_timestamps,
            available_timestamps,
            strict=True,
        ):
            completed = automaton.feed(
                int(state),
                event_timestamp=timestamp,
                available_timestamp=available,
                bar_ordinal=int(bar),
            )
            registered.extend(completed)
            current = StateEventRecord(
                event_index=len(raw_events),
                state=int(state),
                bar_ordinal=int(bar),
                event_timestamp=timestamp,
                available_timestamp=available,
            )
            raw_events.append(current)
            prefixes.append(automaton.active_prefixes())
            lower = max(0, current.event_index - MAX_EVENT_TRANSITIONS)
            for start_index in range(current.event_index - 2, lower - 1, -1):
                if raw_events[start_index].state != current.state:
                    continue
                path = tuple(
                    event.state for event in raw_events[start_index : current.event_index + 1]
                )
                if path in registered_paths:
                    continue
                start = raw_events[start_index]
                unregistered.append(
                    UnregisteredLoopCompletion(
                        full_path=path,
                        start_event_index=start_index,
                        completion_event_index=current.event_index,
                        start_timestamp=start.event_timestamp,
                        completion_timestamp=current.event_timestamp,
                        start_bar_ordinal=start.bar_ordinal,
                        completion_bar_ordinal=current.bar_ordinal,
                    )
                )
        return EventTrace(
            state_events=tuple(raw_events),
            registered_completions=tuple(registered),
            unregistered_completions=tuple(unregistered),
            prefixes_after_event=tuple(prefixes),
        )

    def outcome_for_decision(
        self,
        trace: EventTrace,
        *,
        decision_id: str,
        decision_event_index: int,
        decision_bar_ordinal: int,
        decision_timestamp: datetime | None = None,
        decision_available_timestamp: datetime | None = None,
        horizon_bars: int,
        session_end_bar_ordinal: int,
        source_available: bool = True,
        symbol: str | None = None,
        session: str | None = None,
        source_hashes: Sequence[tuple[str, str]] = (),
    ) -> StructuralOutcomeRow:
        if not source_available:
            return _empty_outcome(
                decision_id,
                PrimaryOutcomeLabel.UNAVAILABLE,
                source_available=False,
                missing_reason="source_sequence_incomplete_or_ambiguous",
            )
        if decision_event_index < 0 or decision_event_index >= len(trace.state_events):
            return _empty_outcome(
                decision_id,
                PrimaryOutcomeLabel.UNAVAILABLE,
                source_available=False,
                missing_reason="decision_has_no_causal_state_event",
            )
        if horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        decision = trace.state_events[decision_event_index]
        actual_decision_timestamp = decision_timestamp or decision.event_timestamp
        actual_decision_available = decision_available_timestamp or decision.available_timestamp
        if actual_decision_available < actual_decision_timestamp:
            raise ValueError("decision availability precedes its timestamp")
        if actual_decision_available < decision.available_timestamp:
            raise ValueError("decision precedes causal state-event availability")
        horizon_end = decision_bar_ordinal + horizon_bars
        registered = sorted(
            (
                event
                for event in trace.registered_completions
                if event.completion_event_index > decision_event_index
                and event.completion_bar_ordinal > decision_bar_ordinal
                and event.completion_bar_ordinal <= horizon_end
            ),
            key=lambda event: (
                event.completion_bar_ordinal,
                event.completion_event_index,
                event.semantic_loop_id,
            ),
        )
        unregistered = sorted(
            (
                event
                for event in trace.unregistered_completions
                if event.completion_event_index > decision_event_index
                and event.completion_bar_ordinal > decision_bar_ordinal
                and event.completion_bar_ordinal <= horizon_end
            ),
            key=lambda event: (
                event.completion_bar_ordinal,
                event.completion_event_index,
                event.full_path,
            ),
        )
        all_registered_ids = _ordered_unique(event.semantic_loop_id for event in registered)
        earliest_registered_bar = registered[0].completion_bar_ordinal if registered else None
        earliest_unregistered_bar = unregistered[0].completion_bar_ordinal if unregistered else None
        active = trace.prefixes_after_event[decision_event_index]
        all_enriched = tuple(
            self._enrich_event(
                event,
                decision_id=decision_id,
                decision=decision,
                decision_event_index=decision_event_index,
                decision_bar_ordinal=decision_bar_ordinal,
                decision_timestamp=actual_decision_timestamp,
                decision_available_timestamp=actual_decision_available,
                active=active,
                tied=(
                    len(
                        {
                            other.semantic_loop_id
                            for other in registered
                            if other.completion_event_index == event.completion_event_index
                        }
                    )
                    > 1
                ),
                symbol=symbol,
                session=session,
                source_hashes=source_hashes,
            )
            for event in registered
        )

        if registered and (
            earliest_unregistered_bar is None
            or earliest_registered_bar is not None
            and earliest_registered_bar <= earliest_unregistered_bar
        ):
            earliest_index = registered[0].completion_event_index
            enriched = tuple(
                event for event in all_enriched if event.completion_event_index == earliest_index
            )
            semantic_ids = tuple(sorted({event.semantic_loop_id for event in enriched}))
            tied = len(semantic_ids) > 1
            primary = PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION if tied else semantic_ids[0]
            motif_earliest = {motif: _first_id_for_motif(registered, motif) for motif in MotifType}
            transitions_remaining = min(
                event.transitions_remaining_at_decision
                for event in enriched
                if event.transitions_remaining_at_decision is not None
            )
            repeat_depth = enriched[0].repeat_depth if len(enriched) == 1 else None
            return StructuralOutcomeRow(
                decision_id=decision_id,
                primary_label=str(primary),
                tied_semantic_loop_ids=semantic_ids if tied else (),
                earliest_registered_events=enriched,
                every_registered_completion_event=all_enriched,
                every_registered_completion_within_horizon=all_registered_ids,
                earliest_primitive_completion=motif_earliest[MotifType.PRIMITIVE],
                earliest_repeated_completion=motif_earliest[MotifType.REPEAT],
                earliest_composite_completion=motif_earliest[MotifType.COMPOSITE],
                bars_until_completion=enriched[0].bars_until_completion,
                state_events_until_completion=enriched[0].state_events_until_completion,
                transitions_remaining_at_decision=transitions_remaining,
                first_event_was_open_prefix=any(
                    bool(event.active_prefix_at_decision) for event in enriched
                ),
                first_event_began_after_decision=all(
                    event.initiated_after_decision for event in enriched
                ),
                repeat_depth=repeat_depth,
                source_available=True,
            )

        if unregistered:
            event = unregistered[0]
            return StructuralOutcomeRow(
                decision_id=decision_id,
                primary_label=PrimaryOutcomeLabel.UNREGISTERED_LOOP,
                tied_semantic_loop_ids=(),
                earliest_registered_events=(),
                every_registered_completion_event=all_enriched,
                every_registered_completion_within_horizon=all_registered_ids,
                earliest_primitive_completion=_first_id_for_motif(registered, MotifType.PRIMITIVE),
                earliest_repeated_completion=_first_id_for_motif(registered, MotifType.REPEAT),
                earliest_composite_completion=_first_id_for_motif(registered, MotifType.COMPOSITE),
                bars_until_completion=event.completion_bar_ordinal - decision_bar_ordinal,
                state_events_until_completion=event.completion_event_index - decision_event_index,
                transitions_remaining_at_decision=None,
                first_event_was_open_prefix=event.start_event_index < decision_event_index,
                first_event_began_after_decision=event.start_event_index > decision_event_index,
                repeat_depth=None,
                source_available=True,
            )

        future_state_event_exists = any(
            event.event_index > decision_event_index for event in trace.state_events
        )
        if not future_state_event_exists or session_end_bar_ordinal <= horizon_end:
            return _empty_outcome(
                decision_id,
                PrimaryOutcomeLabel.SESSION_END,
                source_available=True,
            )
        return _empty_outcome(
            decision_id,
            PrimaryOutcomeLabel.NO_REGISTERED_LOOP_WITHIN_HORIZON,
            source_available=True,
        )

    @staticmethod
    def _enrich_event(
        event: LoopCompletionEvent,
        *,
        decision_id: str,
        decision: StateEventRecord,
        decision_event_index: int,
        decision_bar_ordinal: int,
        decision_timestamp: datetime,
        decision_available_timestamp: datetime,
        active: Sequence[LoopPrefixState],
        tied: bool,
        symbol: str | None,
        session: str | None,
        source_hashes: Sequence[tuple[str, str]],
    ) -> LoopCompletionEvent:
        candidates = [
            prefix
            for prefix in active
            if prefix.semantic_loop_id == event.semantic_loop_id
            and prefix.orientation_id == event.orientation_id
            and prefix.start_event_index == event.start_event_index
        ]
        prefix = max(candidates, key=lambda item: item.progress_states, default=None)
        transitions_remaining = (
            prefix.transitions_remaining if prefix is not None else len(event.full_path) - 1
        )
        return replace(
            event,
            decision_id=decision_id,
            symbol=symbol,
            session=session,
            decision_timestamp=decision_timestamp,
            decision_available_timestamp=decision_available_timestamp,
            transitions_remaining_at_decision=transitions_remaining,
            state_events_until_completion=event.completion_event_index - decision_event_index,
            bars_until_completion=event.completion_bar_ordinal - decision_bar_ordinal,
            active_prefix_at_decision=prefix.prefix_path if prefix is not None else (),
            initiated_before_decision=event.start_bar_ordinal < decision_bar_ordinal,
            initiated_at_decision=event.start_bar_ordinal == decision_bar_ordinal,
            initiated_after_decision=event.start_bar_ordinal > decision_bar_ordinal,
            tied_completion=tied,
            source_hashes=tuple(source_hashes),
        )


def legacy_compatible_cycle_labels(
    trace: EventTrace,
    *,
    decision_event_index: int,
    dictionary: LoopDictionary,
) -> tuple[str, ...]:
    """Reproduce the legacy rotated whole-cycle labels as a diagnostic only."""

    if decision_event_index < 0 or decision_event_index >= len(trace.state_events):
        return ()
    current = trace.state_events[decision_event_index].state
    future = tuple(event.state for event in trace.state_events[decision_event_index + 1 :])
    positives: list[str] = []
    for definition in dictionary.definitions.values():
        matched = any(
            path[0] == current
            and len(future) >= len(path) - 1
            and future[: len(path) - 1] == path[1:]
            for path in definition.oriented_paths
        )
        if matched:
            positives.append(definition.semantic_loop_id)
    return tuple(sorted(positives))


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _first_id_for_motif(events: Sequence[LoopCompletionEvent], motif: MotifType) -> str | None:
    return next(
        (event.semantic_loop_id for event in events if event.motif_type is motif),
        None,
    )


def _empty_outcome(
    decision_id: str,
    label: PrimaryOutcomeLabel,
    *,
    source_available: bool,
    missing_reason: str | None = None,
) -> StructuralOutcomeRow:
    return StructuralOutcomeRow(
        decision_id=decision_id,
        primary_label=label,
        tied_semantic_loop_ids=(),
        earliest_registered_events=(),
        every_registered_completion_event=(),
        every_registered_completion_within_horizon=(),
        earliest_primitive_completion=None,
        earliest_repeated_completion=None,
        earliest_composite_completion=None,
        bars_until_completion=None,
        state_events_until_completion=None,
        transitions_remaining_at_decision=None,
        first_event_was_open_prefix=False,
        first_event_began_after_decision=False,
        repeat_depth=None,
        source_available=source_available,
        missing_reason=missing_reason,
    )


__all__ = [
    "EventTrace",
    "FirstNextLoopEventEngine",
    "LoopEventAutomaton",
    "StateEventRecord",
    "UnknownStateError",
    "UnregisteredLoopCompletion",
    "legacy_compatible_cycle_labels",
]
