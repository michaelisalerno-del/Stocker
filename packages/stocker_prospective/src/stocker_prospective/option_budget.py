"""Deterministic budget-aware option episode queue and recording state machine."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from stocker_prospective.options import DteBucket
from stocker_prospective.subscriptions import (
    BudgetState,
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
)


class EpisodeState(StrEnum):
    IDLE = "IDLE"
    UNIVERSE_MONITORING = "UNIVERSE_MONITORING"
    EPISODE_QUEUED = "EPISODE_QUEUED"
    CONTRACT_DISCOVERY = "CONTRACT_DISCOVERY"
    PRIMARY_LEGS_STREAMING = "PRIMARY_LEGS_STREAMING"
    COMPARISON_LEGS_STREAMING = "COMPARISON_LEGS_STREAMING"
    HORIZON_FINALISING = "HORIZON_FINALISING"
    CANCELLING_SUBSCRIPTIONS = "CANCELLING_SUBSCRIPTIONS"
    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class EpisodeKind(StrEnum):
    QUIET = "quiet"
    HIGH_TAIL = "high_tail"
    NEUTRAL_CONTROL = "neutral_control"


@dataclass(frozen=True)
class DteAllocation:
    primary: DteBucket | None
    secondary: tuple[DteBucket, ...]
    skipped: dict[str, str]


class DteAllocator:
    """Freeze 1DTE-first quiet allocation and outcome-independent controls."""

    _priority = (
        DteBucket.ONE_DTE,
        DteBucket.ZERO_DTE,
        DteBucket.THREE_TO_FIVE_DTE,
    )

    def allocate(
        self,
        *,
        episode_id: str,
        kind: EpisodeKind,
        available: tuple[DteBucket, ...],
        allow_secondary: bool,
        neutral_control_ordinal: int | None = None,
    ) -> DteAllocation:
        available_set = set(available)
        skipped: dict[str, str] = {}
        if kind is EpisodeKind.QUIET:
            if DteBucket.ONE_DTE not in available_set:
                skipped[DteBucket.ONE_DTE.value] = "no_1dte_expiry"
                for bucket in available_set:
                    skipped[bucket.value] = "not_substituted_for_missing_primary_1dte"
                return DteAllocation(None, (), skipped)
            secondary: tuple[DteBucket, ...] = ()
            if allow_secondary:
                secondary = next(
                    (
                        (bucket,)
                        for bucket in (
                            DteBucket.ZERO_DTE,
                            DteBucket.THREE_TO_FIVE_DTE,
                        )
                        if bucket in available_set
                    ),
                    (),
                )
            for bucket in self._priority:
                if bucket not in available_set:
                    skipped.setdefault(bucket.value, "expiry_unavailable")
                elif bucket is not DteBucket.ONE_DTE and bucket not in secondary:
                    skipped.setdefault(bucket.value, "optional_dte_capacity_not_allocated")
            return DteAllocation(DteBucket.ONE_DTE, secondary, skipped)
        if not available_set:
            return DteAllocation(
                None,
                (),
                {bucket.value: "expiry_unavailable" for bucket in self._priority},
            )
        ordered_available = tuple(bucket for bucket in self._priority if bucket in available_set)
        if kind is EpisodeKind.NEUTRAL_CONTROL:
            if neutral_control_ordinal is None or neutral_control_ordinal < 0:
                raise ValueError("neutral controls require a nonnegative frozen ordinal")
            primary = ordered_available[neutral_control_ordinal % len(ordered_available)]
        else:
            digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
            primary = ordered_available[int.from_bytes(digest[:8], "big") % len(ordered_available)]
        for bucket in self._priority:
            if bucket not in available_set:
                skipped[bucket.value] = "expiry_unavailable"
            elif bucket is not primary:
                skipped[bucket.value] = "frozen_single_dte_schedule"
        return DteAllocation(primary, (), skipped)


class SnapshotConcurrencyGate:
    """Idempotent bounded ownership for sequential discovery snapshots."""

    def __init__(self, *, max_concurrent: int) -> None:
        if max_concurrent <= 0:
            raise ValueError("snapshot concurrency must be positive")
        self.max_concurrent = max_concurrent
        self._active: set[str] = set()
        self.maximum_observed = 0

    def reserve(self, snapshot_id: str) -> bool:
        if not snapshot_id:
            raise ValueError("snapshot id is required")
        if snapshot_id in self._active:
            return True
        if len(self._active) >= self.max_concurrent:
            return False
        self._active.add(snapshot_id)
        self.maximum_observed = max(self.maximum_observed, len(self._active))
        return True

    def release(self, snapshot_id: str) -> bool:
        if snapshot_id not in self._active:
            return False
        self._active.remove(snapshot_id)
        return True

    @property
    def active_count(self) -> int:
        return len(self._active)


@dataclass(frozen=True)
class OptionSubscriptionIntent:
    key: str
    con_id: int
    role: str
    subscription_class: SubscriptionClass
    required: bool
    dte_bucket: DteBucket

    def __post_init__(self) -> None:
        if self.con_id <= 0 or self.key != f"OPTION_LEVEL1|{self.con_id}":
            raise ValueError("option intent must use its canonical conId key")
        if not self.role:
            raise ValueError("option intent role is required")
        if "naked" in self.role.lower():
            raise ValueError("naked option structures are forbidden")


@dataclass(frozen=True)
class OptionEpisodeTask:
    episode_id: str
    symbol: str
    kind: EpisodeKind
    probability: float
    triggered_at_utc: datetime
    useful_until_utc: datetime
    requested_subscriptions: tuple[OptionSubscriptionIntent, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or not self.symbol:
            raise ValueError("episode identity and symbol are required")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("episode probability must lie in [0, 1]")
        if (
            self.triggered_at_utc.tzinfo is None
            or self.triggered_at_utc.utcoffset() is None
            or self.useful_until_utc.tzinfo is None
            or self.useful_until_utc.utcoffset() is None
        ):
            raise ValueError("episode timestamps must be timezone-aware")
        if self.useful_until_utc <= self.triggered_at_utc:
            raise ValueError("option episode usefulness window is invalid")
        keys = [intent.key for intent in self.requested_subscriptions]
        if len(keys) != len(set(keys)):
            raise ValueError("option episode contains duplicate canonical streams")


@dataclass(frozen=True)
class EpisodeAllocationRecord:
    episode_id: str
    symbol: str
    kind: EpisodeKind
    state: EpisodeState
    requested_subscriptions: tuple[str, ...]
    approved_subscriptions: tuple[str, ...]
    queued_subscriptions: tuple[str, ...]
    denied_subscriptions: tuple[str, ...]
    degradation_reason: str | None
    capacity_before: dict[str, object]
    capacity_after: dict[str, object]
    updated_at_utc: datetime
    cohort_phase: str = "engineering_transfer"
    scientific_option_evidence: bool = False


PersistenceSink = Callable[[EpisodeAllocationRecord], object]
DisplacementSink = Callable[[str, str, datetime], object]
EvictionSink = Callable[[str, str, datetime], bool]
PhaseResolver = Callable[[OptionEpisodeTask], tuple[str, bool]]


class BudgetAwareEpisodeStateMachine:
    """Reserve before request, queue overlap, and release every episode line."""

    _active_states = frozenset(
        {
            EpisodeState.CONTRACT_DISCOVERY,
            EpisodeState.PRIMARY_LEGS_STREAMING,
            EpisodeState.COMPARISON_LEGS_STREAMING,
            EpisodeState.HORIZON_FINALISING,
            EpisodeState.CANCELLING_SUBSCRIPTIONS,
            EpisodeState.DEGRADED,
        }
    )

    def __init__(
        self,
        *,
        budget: SubscriptionBudgetManager,
        max_active_episodes: int = 1,
        max_option_lines_per_episode: int = 8,
        max_concurrent_snapshots: int = 2,
        maximum_recording_duration: timedelta = timedelta(minutes=65),
        persistence_sink: PersistenceSink | None = None,
        displacement_sink: DisplacementSink | None = None,
        eviction_sink: EvictionSink | None = None,
        phase_resolver: PhaseResolver | None = None,
    ) -> None:
        if max_active_episodes not in {1, 2}:
            raise ValueError("active option episodes must be one or two")
        if max_option_lines_per_episode < 4:
            raise ValueError("option episode line limit must secure four primary legs")
        if maximum_recording_duration < timedelta(minutes=30):
            raise ValueError("maximum option recording duration is too short")
        self.budget = budget
        self.max_active_episodes = max_active_episodes
        self.max_option_lines_per_episode = max_option_lines_per_episode
        self.maximum_recording_duration = maximum_recording_duration
        self.snapshots = SnapshotConcurrencyGate(max_concurrent=max_concurrent_snapshots)
        self.persistence_sink = persistence_sink
        self.displacement_sink = displacement_sink
        self.eviction_sink = eviction_sink
        self.phase_resolver = phase_resolver
        self._tasks: dict[str, OptionEpisodeTask] = {}
        self._records: dict[str, EpisodeAllocationRecord] = {}

    @staticmethod
    def _priority(task: OptionEpisodeTask) -> tuple[int, float, datetime, str, str]:
        kind_rank = {
            EpisodeKind.QUIET: 0,
            EpisodeKind.HIGH_TAIL: 1,
            EpisodeKind.NEUTRAL_CONTROL: 2,
        }[task.kind]
        probability_rank = task.probability if task.kind is EpisodeKind.QUIET else -task.probability
        return (
            kind_rank,
            probability_rank,
            task.triggered_at_utc.astimezone(UTC),
            task.symbol,
            task.episode_id,
        )

    def _persist(self, record: EpisodeAllocationRecord) -> EpisodeAllocationRecord:
        self._records[record.episode_id] = record
        if self.persistence_sink is not None:
            self.persistence_sink(record)
        return record

    def _new_record(
        self,
        task: OptionEpisodeTask,
        *,
        state: EpisodeState,
        now: datetime,
        before: dict[str, object],
        approved: tuple[str, ...] = (),
        queued: tuple[str, ...] = (),
        denied: tuple[str, ...] = (),
        reason: str | None = None,
    ) -> EpisodeAllocationRecord:
        cohort_phase, scientific_option_evidence = (
            ("engineering_transfer", False)
            if self.phase_resolver is None
            else self.phase_resolver(task)
        )
        if cohort_phase == "engineering_transfer" and scientific_option_evidence:
            raise ValueError("engineering-transfer option records cannot be scientific evidence")
        return EpisodeAllocationRecord(
            episode_id=task.episode_id,
            symbol=task.symbol,
            kind=task.kind,
            state=state,
            requested_subscriptions=tuple(intent.key for intent in task.requested_subscriptions),
            approved_subscriptions=approved,
            queued_subscriptions=queued,
            denied_subscriptions=denied,
            degradation_reason=reason,
            capacity_before=before,
            capacity_after=self.budget.snapshot(),
            updated_at_utc=now.astimezone(UTC),
            cohort_phase=cohort_phase,
            scientific_option_evidence=scientific_option_evidence,
        )

    @property
    def active_episode_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                episode_id
                for episode_id, record in self._records.items()
                if record.state in self._active_states
            )
        )

    def record(self, episode_id: str) -> EpisodeAllocationRecord:
        return self._records[episode_id]

    def _queue(
        self,
        task: OptionEpisodeTask,
        *,
        now: datetime,
        reason: str,
    ) -> EpisodeAllocationRecord:
        before = self.budget.snapshot()
        keys = tuple(intent.key for intent in task.requested_subscriptions)
        return self._persist(
            self._new_record(
                task,
                state=EpisodeState.EPISODE_QUEUED,
                now=now,
                before=before,
                queued=keys,
                reason=reason,
            )
        )

    def submit(
        self,
        task: OptionEpisodeTask,
        *,
        now: datetime,
    ) -> EpisodeAllocationRecord:
        observed = now.astimezone(UTC)
        existing_task = self._tasks.get(task.episode_id)
        if existing_task is not None:
            if existing_task != task:
                raise ValueError("option episode task is immutable")
            return self._records[task.episode_id]
        self._tasks[task.episode_id] = task
        if len(self.active_episode_ids) >= self.max_active_episodes:
            worst_active_id = max(
                self.active_episode_ids,
                key=lambda episode_id: self._priority(self._tasks[episode_id]),
            )
            worst_task = self._tasks[worst_active_id]
            if self._priority(task) < self._priority(worst_task):
                self._displace(
                    worst_active_id,
                    replacement_episode_id=task.episode_id,
                    now=observed,
                )
            else:
                return self._queue(
                    task,
                    now=observed,
                    reason="active_option_episode_limit",
                )
        return self._start(task, now=observed)

    def _start(
        self,
        task: OptionEpisodeTask,
        *,
        now: datetime,
    ) -> EpisodeAllocationRecord:
        before = self.budget.snapshot()
        self._persist(
            self._new_record(
                task,
                state=EpisodeState.CONTRACT_DISCOVERY,
                now=now,
                before=before,
            )
        )
        indexed_intents = tuple(enumerate(task.requested_subscriptions))
        ordered = sorted(
            indexed_intents,
            key=lambda item: (
                int(item[1].subscription_class),
                0 if item[1].required else 1,
                item[0],
            ),
        )
        approved_by_index: dict[int, str] = {}
        denied_by_index: dict[int, tuple[str, str]] = {}
        owner_id = f"episode:{task.episode_id}"
        for allocation_rank, (source_index, intent) in enumerate(ordered):
            if allocation_rank >= self.max_option_lines_per_episode:
                denied_by_index[source_index] = (
                    intent.key,
                    "per_episode_option_line_limit",
                )
                continue
            decision = self.budget.allocate(
                key=intent.key,
                kind=SubscriptionKind.OPTION,
                symbol=task.symbol,
                con_id=intent.con_id,
                request_id=-1,
                priority=SubscriptionPriority.from_class(intent.subscription_class),
                subscription_class=intent.subscription_class,
                owner_episode=task.episode_id,
                owner_id=owner_id,
                protected=intent.required,
                now_monotonic=now.timestamp() + allocation_rank / 1_000_000.0,
                now_utc=now,
            )
            if decision.accepted:
                eviction_results = (
                    ()
                    if self.eviction_sink is None
                    else tuple(
                        self.eviction_sink(evicted_key, intent.key, now)
                        for evicted_key in decision.evicted_keys
                    )
                )
                eviction_cancelled = all(eviction_results)
                if not eviction_cancelled:
                    self.budget.release(
                        intent.key,
                        owner_id=owner_id,
                        reason="evicted_subscription_cancellation_failed",
                        now_utc=now,
                    )
                    denied_by_index[source_index] = (
                        intent.key,
                        "evicted_subscription_cancellation_failed",
                    )
                    continue
                approved_by_index[source_index] = intent.key
            else:
                denied_by_index[source_index] = (intent.key, decision.reason)
        required_denied = [
            index
            for index, intent in indexed_intents
            if intent.required and index in denied_by_index
        ]
        if required_denied:
            for key in approved_by_index.values():
                self.budget.release(
                    key,
                    owner_id=owner_id,
                    reason="primary_option_legs_incomplete",
                    now_utc=now,
                )
            return self._queue(
                task,
                now=now,
                reason="primary_option_legs_incomplete",
            )
        approved = tuple(key for index, key in sorted(approved_by_index.items()))
        denied = tuple(value[0] for _index, value in sorted(denied_by_index.items()))
        if denied:
            state = EpisodeState.DEGRADED
            reasons = ",".join(sorted({value[1] for value in denied_by_index.values()}))
        else:
            has_comparison = any(
                not intent.required and index in approved_by_index
                for index, intent in indexed_intents
            )
            state = (
                EpisodeState.COMPARISON_LEGS_STREAMING
                if has_comparison
                else EpisodeState.PRIMARY_LEGS_STREAMING
            )
            reasons = None
        return self._persist(
            self._new_record(
                task,
                state=state,
                now=now,
                before=before,
                approved=approved,
                denied=denied,
                reason=reasons,
            )
        )

    def _release_episode(self, episode_id: str, *, now: datetime, reason: str) -> None:
        record = self._records[episode_id]
        owner_id = f"episode:{episode_id}"
        for key in record.approved_subscriptions:
            self.budget.release(
                key,
                owner_id=owner_id,
                reason=reason,
                now_utc=now,
            )

    def _displace(
        self,
        episode_id: str,
        *,
        replacement_episode_id: str,
        now: datetime,
    ) -> None:
        task = self._tasks[episode_id]
        previous = self._records[episode_id]
        before = self.budget.snapshot()
        if self.displacement_sink is not None:
            self.displacement_sink(
                episode_id,
                replacement_episode_id,
                now.astimezone(UTC),
            )
        self._release_episode(
            episode_id,
            now=now,
            reason=f"displaced_by:{replacement_episode_id}",
        )
        displaced = self._persist(
            self._new_record(
                task,
                state=EpisodeState.DEGRADED,
                now=now,
                before=before,
                denied=previous.approved_subscriptions,
                reason=f"displaced_by_higher_priority_episode:{replacement_episode_id}",
            )
        )
        if now < task.useful_until_utc:
            self._persist(
                replace(
                    displaced,
                    state=EpisodeState.EPISODE_QUEUED,
                    queued_subscriptions=tuple(
                        intent.key for intent in task.requested_subscriptions
                    ),
                    denied_subscriptions=(),
                    capacity_before=self.budget.snapshot(),
                    capacity_after=self.budget.snapshot(),
                )
            )

    def complete(self, episode_id: str, *, now: datetime) -> EpisodeAllocationRecord:
        observed = now.astimezone(UTC)
        task = self._tasks[episode_id]
        current = self._records[episode_id]
        before = self.budget.snapshot()
        finalising = self._persist(
            replace(
                current,
                state=EpisodeState.HORIZON_FINALISING,
                capacity_before=before,
                capacity_after=before,
                updated_at_utc=observed,
            )
        )
        cancelling = self._persist(
            replace(
                finalising,
                state=EpisodeState.CANCELLING_SUBSCRIPTIONS,
                updated_at_utc=observed,
            )
        )
        self._release_episode(
            episode_id,
            now=observed,
            reason="episode_recording_horizon_complete",
        )
        return self._persist(
            self._new_record(
                task,
                state=EpisodeState.COMPLETE,
                now=observed,
                before=cancelling.capacity_before,
                approved=current.approved_subscriptions,
                denied=current.denied_subscriptions,
                reason=current.degradation_reason,
            )
        )

    def fail_request(
        self,
        episode_id: str,
        *,
        key: str,
        now: datetime,
        reason: str,
    ) -> EpisodeAllocationRecord:
        current = self._records[episode_id]
        if key not in current.approved_subscriptions:
            return current
        owner_id = f"episode:{episode_id}"
        before = self.budget.snapshot()
        self.budget.release(
            key,
            owner_id=owner_id,
            reason=reason,
            now_utc=now,
        )
        approved = tuple(item for item in current.approved_subscriptions if item != key)
        denied = (*current.denied_subscriptions, key)
        return self._persist(
            replace(
                current,
                state=EpisodeState.DEGRADED,
                approved_subscriptions=approved,
                denied_subscriptions=denied,
                degradation_reason=reason,
                capacity_before=before,
                capacity_after=self.budget.snapshot(),
                updated_at_utc=now.astimezone(UTC),
            )
        )

    def poll(self, *, now: datetime) -> tuple[EpisodeAllocationRecord, ...]:
        observed = now.astimezone(UTC)
        changed: list[EpisodeAllocationRecord] = []
        for episode_id in tuple(self.active_episode_ids):
            task = self._tasks[episode_id]
            deadline = min(
                task.useful_until_utc.astimezone(UTC),
                task.triggered_at_utc.astimezone(UTC) + self.maximum_recording_duration,
            )
            if observed >= deadline:
                changed.append(self.complete(episode_id, now=observed))
        queued = sorted(
            (
                self._tasks[episode_id]
                for episode_id, record in self._records.items()
                if record.state is EpisodeState.EPISODE_QUEUED
            ),
            key=self._priority,
        )
        for task in queued:
            if len(self.active_episode_ids) >= self.max_active_episodes:
                break
            if observed >= task.useful_until_utc.astimezone(UTC):
                previous = self._records[task.episode_id]
                failed = replace(
                    previous,
                    state=EpisodeState.FAILED,
                    queued_subscriptions=(),
                    denied_subscriptions=previous.queued_subscriptions,
                    degradation_reason="queued_episode_expired",
                    capacity_before=self.budget.snapshot(),
                    capacity_after=self.budget.snapshot(),
                    updated_at_utc=observed,
                )
                changed.append(self._persist(failed))
                continue
            changed.append(self._start(task, now=observed))
        return tuple(changed)

    def snapshot(self) -> dict[str, object]:
        queued = any(
            record.state is EpisodeState.EPISODE_QUEUED for record in self._records.values()
        )
        partially_recorded = any(
            record.state is EpisodeState.DEGRADED
            and bool(
                set(record.denied_subscriptions).intersection(
                    intent.key
                    for intent in self._tasks[episode_id].requested_subscriptions
                    if intent.required
                )
            )
            for episode_id, record in self._records.items()
        )
        optional_degraded = any(
            record.state is EpisodeState.DEGRADED for record in self._records.values()
        )
        return {
            "active_episode_ids": self.active_episode_ids,
            "queued_episodes": sum(
                record.state is EpisodeState.EPISODE_QUEUED for record in self._records.values()
            ),
            "degraded_episodes": sum(
                record.state is EpisodeState.DEGRADED for record in self._records.values()
            ),
            "states": {
                episode_id: record.state.value
                for episode_id, record in sorted(self._records.items())
            },
            "snapshot_concurrency": {
                "active": self.snapshots.active_count,
                "limit": self.snapshots.max_concurrent,
                "maximum_observed": self.snapshots.maximum_observed,
            },
            "budget": self.budget.snapshot(),
            "budget_state": (
                BudgetState.OPTION_EPISODE_PARTIALLY_RECORDED.value
                if partially_recorded
                else BudgetState.OPTION_EPISODE_QUEUED.value
                if queued
                else BudgetState.OPTIONAL_FEEDS_DEGRADED.value
                if optional_degraded
                else self.budget.snapshot()["budget_state"]
            ),
        }


__all__ = [
    "BudgetAwareEpisodeStateMachine",
    "DteAllocation",
    "DteAllocator",
    "EpisodeAllocationRecord",
    "EpisodeKind",
    "EpisodeState",
    "OptionEpisodeTask",
    "OptionSubscriptionIntent",
    "SnapshotConcurrencyGate",
]
