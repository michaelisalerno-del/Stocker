"""Broker-isolated deterministic raw-event replay for recorder V0."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from stocker_prospective.events import RawEvent


class ReplayMode(StrEnum):
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"
    STEP = "step_by_step"
    EPISODE_ONLY = "episode_only"


@dataclass(frozen=True)
class DeterministicReplayResult:
    mode: ReplayMode
    event_ids: tuple[str, ...]
    digest: str
    event_count: int
    ibkr_connections_attempted: int
    maximum_floating_difference: float


def ordered_events(
    events: Iterable[RawEvent],
    *,
    mode: ReplayMode,
    episode_id: str | None = None,
) -> tuple[RawEvent, ...]:
    selected = tuple(events)
    if mode is ReplayMode.EPISODE_ONLY:
        if not episode_id:
            raise ValueError("episode-only replay requires an episode_id")
        selected = tuple(
            event for event in selected if getattr(event, "episode_id", None) == episode_id
        )
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.ordering_timestamp,
                item.received_monotonic_ns,
                item.source_sequence,
                item.event_id,
            ),
        )
    )


def step_replay(
    events: Iterable[RawEvent],
    *,
    episode_id: str | None = None,
) -> Iterator[RawEvent]:
    yield from ordered_events(
        events,
        mode=ReplayMode.EPISODE_ONLY if episode_id else ReplayMode.STEP,
        episode_id=episode_id,
    )


def deterministic_replay(
    events: Iterable[RawEvent],
    *,
    mode: ReplayMode,
    acceleration: float = 100.0,
    episode_id: str | None = None,
    on_event: Callable[[RawEvent], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> DeterministicReplayResult:
    """Replay stored events without accepting or constructing any IBKR adapter."""

    if acceleration <= 0.0:
        raise ValueError("replay acceleration must be positive")
    ordered = ordered_events(events, mode=mode, episode_id=episode_id)
    previous = None
    canonical: list[dict[str, object]] = []
    for event in ordered:
        if mode is ReplayMode.REAL_TIME and previous is not None:
            delay = (event.ordering_timestamp - previous.ordering_timestamp).total_seconds()
            if delay > 0.0:
                sleeper(delay)
        elif mode is ReplayMode.ACCELERATED and previous is not None:
            delay = (event.ordering_timestamp - previous.ordering_timestamp).total_seconds()
            if delay > 0.0 and acceleration < 100.0:
                sleeper(delay / acceleration)
        if on_event is not None:
            on_event(event)
        canonical.append(event.model_dump(mode="json"))
        previous = event
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return DeterministicReplayResult(
        mode=mode,
        event_ids=tuple(item.event_id for item in ordered),
        digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        event_count=len(ordered),
        ibkr_connections_attempted=0,
        maximum_floating_difference=0.0,
    )
