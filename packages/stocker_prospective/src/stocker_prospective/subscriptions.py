"""Budget-aware subscription ownership, shedding, and promotion scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Literal


class SubscriptionKind(StrEnum):
    LEVEL1 = "level1"
    TICK_BY_TICK = "tick_by_tick"
    DEPTH = "depth"
    OPTION = "option"
    BAR = "bar"
    MARKET_PROXY = "market_proxy"


class SubscriptionClass(IntEnum):
    """Binding IBKR subscription classes; smaller values are more important."""

    CRITICAL_SYSTEM = 0
    FROZEN_UNIVERSE_SIGNAL = 1
    ACTIVE_EPISODE = 2
    EPISODE_ENGINEERING = 3
    MICROSTRUCTURE_ENHANCEMENT = 4
    OPTIONAL_RESEARCH = 5


class SubscriptionPriority(IntEnum):
    """Compatibility priority where larger values are more important."""

    OPTIONAL_RESEARCH = 10
    MICROSTRUCTURE_ENHANCEMENT = 20
    ARMED_CANDIDATE = 20
    EPISODE_ENGINEERING = 30
    ACTIVE_EPISODE = 40
    ACTIVE_OPTION = 40
    FROZEN_UNIVERSE_SIGNAL = 50
    UNIVERSE_LEVEL1 = 50
    CRITICAL_SYSTEM = 60
    MARKET_PROXY = 60

    @classmethod
    def from_class(cls, subscription_class: SubscriptionClass) -> SubscriptionPriority:
        return {
            SubscriptionClass.CRITICAL_SYSTEM: cls.CRITICAL_SYSTEM,
            SubscriptionClass.FROZEN_UNIVERSE_SIGNAL: cls.FROZEN_UNIVERSE_SIGNAL,
            SubscriptionClass.ACTIVE_EPISODE: cls.ACTIVE_EPISODE,
            SubscriptionClass.EPISODE_ENGINEERING: cls.EPISODE_ENGINEERING,
            SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT: cls.MICROSTRUCTURE_ENHANCEMENT,
            SubscriptionClass.OPTIONAL_RESEARCH: cls.OPTIONAL_RESEARCH,
        }[subscription_class]


class BudgetState(StrEnum):
    BUDGET_HEALTHY = "budget_healthy"
    BUDGET_CONSTRAINED = "budget_constrained"
    OPTIONAL_FEEDS_DEGRADED = "optional_feeds_degraded"
    OPTION_EPISODE_QUEUED = "option_episode_queued"
    OPTION_EPISODE_PARTIALLY_RECORDED = "option_episode_partially_recorded"
    CRITICAL_BUDGET_UNAVAILABLE = "critical_budget_unavailable"


class SubscriptionStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"


def canonical_subscription_key(
    kind: SubscriptionKind,
    *,
    con_id: int,
    bar_size: str = "5m",
    use_rth: bool = True,
    tick_type: Literal["BidAsk", "Last"] | None = None,
    depth_rows: int = 5,
    smart_depth: bool = True,
) -> str:
    """Return the sole broker-stream identity used across recorder services."""

    if con_id <= 0:
        raise ValueError("canonical subscription key requires a positive conId")
    if kind is SubscriptionKind.LEVEL1:
        return f"LEVEL1|{con_id}"
    if kind is SubscriptionKind.BAR:
        if not bar_size:
            raise ValueError("bar size is required")
        return f"BAR|{con_id}|{bar_size}|{'RTH' if use_rth else 'ALL'}"
    if kind is SubscriptionKind.TICK_BY_TICK:
        if tick_type not in {"BidAsk", "Last"}:
            raise ValueError("tick-by-tick key requires BidAsk or Last")
        suffix = "BIDASK" if tick_type == "BidAsk" else "LAST"
        return f"TBT_{suffix}|{con_id}"
    if kind is SubscriptionKind.DEPTH:
        if depth_rows <= 0:
            raise ValueError("depth rows must be positive")
        return f"DEPTH|{con_id}|{depth_rows}|{int(smart_depth)}"
    if kind is SubscriptionKind.OPTION:
        return f"OPTION_LEVEL1|{con_id}"
    if kind is SubscriptionKind.MARKET_PROXY:
        return f"LEVEL1|{con_id}"
    raise ValueError(f"unsupported subscription kind: {kind}")


@dataclass
class SubscriptionRecord:
    key: str
    kind: SubscriptionKind
    symbol: str
    con_id: int
    request_id: int
    priority: SubscriptionPriority
    owner_episode: str | None
    protected: bool
    started_at_utc: datetime
    started_monotonic: float
    cancelled_at_utc: datetime | None = None
    cancellation_reason: str | None = None
    ibkr_error_codes: tuple[int, ...] = ()
    capacity_denied: bool = False
    subscription_class: SubscriptionClass = SubscriptionClass.OPTIONAL_RESEARCH
    line_cost: int = 1
    status: SubscriptionStatus = SubscriptionStatus.PENDING
    owners: dict[str, SubscriptionClass] = field(default_factory=dict)
    owner_priorities: dict[str, SubscriptionPriority] = field(default_factory=dict)
    protected_owners: set[str] = field(default_factory=set)
    last_callback_at_utc: datetime | None = None
    generation: int = 0

    @property
    def owner_count(self) -> int:
        return len(self.owners)

    @property
    def active(self) -> bool:
        return self.cancelled_at_utc is None and self.status not in {
            SubscriptionStatus.CANCELLED,
            SubscriptionStatus.FAILED,
        }


@dataclass(frozen=True)
class AllocationDecision:
    accepted: bool
    key: str
    reason: str
    evicted_key: str | None
    evicted_keys: tuple[str, ...] = ()
    budget_state: BudgetState = BudgetState.BUDGET_HEALTHY
    owner_count: int = 0


@dataclass(frozen=True)
class ReleaseDecision:
    key: str
    owner_removed: bool
    cancel_upstream: bool
    remaining_owner_count: int


@dataclass(frozen=True)
class ReconciliationWarning:
    code: str
    key: str | None
    request_id: int | None
    repaired: bool


@dataclass(frozen=True)
class ReconciliationResult:
    warnings: tuple[ReconciliationWarning, ...]
    released_keys: tuple[str, ...]
    orphan_request_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromotionDecision:
    promotion_time: datetime
    symbol: str
    m1c_probability: float
    rank: int
    capacity_available: int
    subscription_type: str
    reason: str


def _class_for_priority(priority: SubscriptionPriority) -> SubscriptionClass:
    if priority >= SubscriptionPriority.CRITICAL_SYSTEM:
        return SubscriptionClass.CRITICAL_SYSTEM
    if priority >= SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL:
        return SubscriptionClass.FROZEN_UNIVERSE_SIGNAL
    if priority >= SubscriptionPriority.ACTIVE_EPISODE:
        return SubscriptionClass.ACTIVE_EPISODE
    if priority >= SubscriptionPriority.EPISODE_ENGINEERING:
        return SubscriptionClass.EPISODE_ENGINEERING
    if priority >= SubscriptionPriority.MICROSTRUCTURE_ENHANCEMENT:
        return SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT
    return SubscriptionClass.OPTIONAL_RESEARCH


class SubscriptionBudgetManager:
    """Own each canonical stream once and preserve critical/reserved capacity."""

    def __init__(
        self,
        *,
        limits: dict[SubscriptionKind, int],
        request_rate_limit: int,
        request_rate_window_seconds: float = 1.0,
        total_line_limit: int | None = None,
        externally_reserved_lines: int = 0,
        future_trading_reserve_lines: int = 0,
        safety_margin_lines: int = 0,
    ) -> None:
        if request_rate_limit <= 0 or request_rate_window_seconds <= 0:
            raise ValueError("request rate bounds must be positive")
        if any(limit < 0 for limit in limits.values()):
            raise ValueError("subscription limits must be nonnegative")
        resolved_total = sum(limits.values()) if total_line_limit is None else total_line_limit
        if resolved_total < 0:
            raise ValueError("total subscription line limit must be nonnegative")
        reserved = externally_reserved_lines + future_trading_reserve_lines + safety_margin_lines
        if (
            min(
                externally_reserved_lines,
                future_trading_reserve_lines,
                safety_margin_lines,
            )
            < 0
        ):
            raise ValueError("reserved subscription lines must be nonnegative")
        if reserved > 0 and reserved >= resolved_total:
            raise ValueError("reserved lines must leave at least one research line")
        self.limits = dict(limits)
        self.request_rate_limit = request_rate_limit
        self.request_rate_window_seconds = request_rate_window_seconds
        self.total_line_limit = resolved_total
        self.externally_reserved_lines = externally_reserved_lines
        self.future_trading_reserve_lines = future_trading_reserve_lines
        self.safety_margin_lines = safety_margin_lines
        self.records: dict[str, SubscriptionRecord] = {}
        self.lifecycle: list[SubscriptionRecord] = []
        self.capacity_denials: list[dict[str, object]] = []
        self.reconciliation_warnings: list[ReconciliationWarning] = []
        self._request_times: deque[float] = deque()

    @property
    def usable_research_lines(self) -> int:
        return (
            self.total_line_limit
            - self.externally_reserved_lines
            - self.future_trading_reserve_lines
            - self.safety_margin_lines
        )

    def _trim_rate(self, now: float) -> None:
        while self._request_times and self._request_times[0] <= (
            now - self.request_rate_window_seconds
        ):
            self._request_times.popleft()

    def _active_records(self) -> list[SubscriptionRecord]:
        return [record for record in self.records.values() if record.active]

    def _active_for_kind(self, kind: SubscriptionKind) -> list[SubscriptionRecord]:
        return [record for record in self._active_records() if record.kind is kind]

    def _current_usage(self) -> int:
        return sum(record.line_cost for record in self._active_records())

    @staticmethod
    def _owner_id(
        *,
        key: str,
        owner_id: str | None,
        owner_episode: str | None,
    ) -> str:
        if owner_id:
            return owner_id
        if owner_episode:
            return f"episode:{owner_episode}"
        return f"anonymous:{key}"

    @staticmethod
    def _denial_state(subscription_class: SubscriptionClass) -> BudgetState:
        if subscription_class <= SubscriptionClass.FROZEN_UNIVERSE_SIGNAL:
            return BudgetState.CRITICAL_BUDGET_UNAVAILABLE
        if subscription_class is SubscriptionClass.ACTIVE_EPISODE:
            return BudgetState.OPTION_EPISODE_QUEUED
        return BudgetState.OPTIONAL_FEEDS_DEGRADED

    @staticmethod
    def _denial_reason(subscription_class: SubscriptionClass, cause: str) -> str:
        if subscription_class <= SubscriptionClass.FROZEN_UNIVERSE_SIGNAL:
            return f"critical_budget_unavailable:{cause}"
        if subscription_class is SubscriptionClass.ACTIVE_EPISODE:
            return f"option_episode_queued:{cause}"
        return f"optional_capacity_unavailable:{cause}"

    def _shedding_candidates(
        self,
        *,
        requested_class: SubscriptionClass,
        kind: SubscriptionKind | None,
    ) -> list[SubscriptionRecord]:
        return sorted(
            (
                record
                for record in self._active_records()
                if not record.protected
                and record.subscription_class > requested_class
                and (kind is None or record.kind is kind)
            ),
            key=lambda record: (
                -int(record.subscription_class),
                int(record.priority),
                record.started_monotonic,
                record.symbol,
                record.key,
            ),
        )

    def allocate(
        self,
        *,
        key: str,
        kind: SubscriptionKind,
        symbol: str,
        con_id: int,
        request_id: int,
        priority: SubscriptionPriority,
        owner_episode: str | None = None,
        owner_id: str | None = None,
        subscription_class: SubscriptionClass | None = None,
        protected: bool = False,
        line_cost: int = 1,
        now_monotonic: float,
        now_utc: datetime | None = None,
    ) -> AllocationDecision:
        if line_cost <= 0:
            raise ValueError("subscription line cost must be positive")
        resolved_class = (
            _class_for_priority(priority) if subscription_class is None else subscription_class
        )
        resolved_owner = self._owner_id(
            key=key,
            owner_id=owner_id,
            owner_episode=owner_episode,
        )
        existing = self.get(key)
        if existing is not None:
            owner_added = resolved_owner not in existing.owners
            existing.owners[resolved_owner] = resolved_class
            existing.owner_priorities[resolved_owner] = priority
            if protected:
                existing.protected_owners.add(resolved_owner)
            existing.subscription_class = min(existing.owners.values())
            existing.priority = max(existing.owner_priorities.values())
            existing.protected = bool(existing.protected_owners)
            if owner_episode is not None and existing.owner_episode is None:
                existing.owner_episode = owner_episode
            return AllocationDecision(
                True,
                key,
                "already_active_owner_added" if owner_added else "already_active",
                None,
                budget_state=self._state(),
                owner_count=existing.owner_count,
            )
        self._trim_rate(now_monotonic)
        if len(self._request_times) >= self.request_rate_limit:
            return self._deny(
                key,
                kind,
                symbol,
                resolved_class,
                cause="request_rate",
            )
        limit = self.limits.get(kind, 0)
        evicted: list[str] = []
        while len(self._active_for_kind(kind)) >= limit:
            candidates = self._shedding_candidates(
                requested_class=resolved_class,
                kind=kind,
            )
            if not candidates:
                return self._deny(
                    key,
                    kind,
                    symbol,
                    resolved_class,
                    cause=f"{kind.value}_lines",
                )
            victim = candidates[0]
            self.cancel(
                victim.key,
                reason=f"preempted_by:{key}",
                now_utc=now_utc,
            )
            evicted.append(victim.key)
        while self._current_usage() + line_cost > self.usable_research_lines:
            candidates = self._shedding_candidates(
                requested_class=resolved_class,
                kind=None,
            )
            if not candidates:
                return self._deny(
                    key,
                    kind,
                    symbol,
                    resolved_class,
                    cause="ordinary_level1_lines",
                )
            victim = candidates[0]
            self.cancel(
                victim.key,
                reason=f"preempted_by:{key}",
                now_utc=now_utc,
            )
            evicted.append(victim.key)
        observed_at = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
        record = SubscriptionRecord(
            key=key,
            kind=kind,
            symbol=symbol,
            con_id=con_id,
            request_id=request_id,
            priority=priority,
            owner_episode=owner_episode,
            protected=protected,
            started_at_utc=observed_at,
            started_monotonic=now_monotonic,
            subscription_class=resolved_class,
            line_cost=line_cost,
            owners={resolved_owner: resolved_class},
            owner_priorities={resolved_owner: priority},
            protected_owners={resolved_owner} if protected else set(),
        )
        self.records[key] = record
        self.lifecycle.append(record)
        self._request_times.append(now_monotonic)
        unique_evictions = tuple(dict.fromkeys(evicted))
        return AllocationDecision(
            True,
            key,
            "allocated",
            unique_evictions[0] if unique_evictions else None,
            evicted_keys=unique_evictions,
            budget_state=(
                BudgetState.OPTIONAL_FEEDS_DEGRADED if unique_evictions else self._state()
            ),
            owner_count=1,
        )

    def _deny(
        self,
        key: str,
        kind: SubscriptionKind,
        symbol: str,
        subscription_class: SubscriptionClass,
        *,
        cause: str,
    ) -> AllocationDecision:
        state = self._denial_state(subscription_class)
        reason = self._denial_reason(subscription_class, cause)
        self.capacity_denials.append(
            {
                "key": key,
                "kind": kind.value,
                "symbol": symbol,
                "subscription_class": int(subscription_class),
                "reason": reason,
                "budget_state": state.value,
            }
        )
        return AllocationDecision(
            False,
            key,
            reason,
            None,
            budget_state=state,
        )

    def _state(self) -> BudgetState:
        if self._current_usage() >= self.usable_research_lines:
            return BudgetState.BUDGET_CONSTRAINED
        return BudgetState.BUDGET_HEALTHY

    def mark_active(self, key: str, *, request_id: int) -> bool:
        record = self.get(key)
        if record is None:
            raise KeyError(key)
        duplicate = next(
            (
                candidate
                for candidate in self._active_records()
                if candidate.key != key
                and candidate.request_id == request_id
                and candidate.status is SubscriptionStatus.ACTIVE
            ),
            None,
        )
        if duplicate is not None:
            record.status = SubscriptionStatus.FAILED
            record.cancelled_at_utc = datetime.now(UTC)
            record.cancellation_reason = f"duplicate_request_id:{duplicate.key}"
            self.reconciliation_warnings.append(
                ReconciliationWarning(
                    code="duplicate_request_id",
                    key=key,
                    request_id=request_id,
                    repaired=True,
                )
            )
            return False
        record.request_id = request_id
        record.status = SubscriptionStatus.ACTIVE
        return True

    def note_callback(self, key: str, *, observed_at: datetime) -> None:
        record = self.get(key)
        if record is None:
            return
        record.last_callback_at_utc = observed_at.astimezone(UTC)
        if record.status is SubscriptionStatus.PENDING:
            record.status = SubscriptionStatus.ACTIVE

    def request_cancellation(self, key: str) -> bool:
        record = self.get(key)
        if record is None:
            return False
        record.status = SubscriptionStatus.CANCELLATION_REQUESTED
        return True

    def release(
        self,
        key: str,
        *,
        owner_id: str,
        reason: str,
        now_utc: datetime | None = None,
    ) -> ReleaseDecision:
        record = self.get(key)
        if record is None:
            return ReleaseDecision(key, False, False, 0)
        removed = owner_id in record.owners
        record.owners.pop(owner_id, None)
        record.owner_priorities.pop(owner_id, None)
        record.protected_owners.discard(owner_id)
        if record.owners:
            record.subscription_class = min(record.owners.values())
            record.priority = max(record.owner_priorities.values())
            record.protected = bool(record.protected_owners)
            return ReleaseDecision(key, removed, False, record.owner_count)
        self.cancel(key, reason=reason, now_utc=now_utc)
        return ReleaseDecision(key, removed, True, 0)

    def cancel(
        self,
        key: str,
        *,
        reason: str,
        now_utc: datetime | None = None,
    ) -> bool:
        record = self.get(key)
        if record is None:
            return False
        record.cancelled_at_utc = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
        record.cancellation_reason = reason
        record.status = SubscriptionStatus.CANCELLED
        record.owners.clear()
        record.owner_priorities.clear()
        record.protected_owners.clear()
        return True

    def note_ibkr_error(self, key: str, code: int) -> None:
        record = self.records.get(key)
        if record is None:
            self.capacity_denials.append(
                {
                    "key": key,
                    "kind": None,
                    "symbol": None,
                    "reason": f"ibkr_error_{code}",
                }
            )
            return
        record.ibkr_error_codes = (*record.ibkr_error_codes, code)
        if code in {100, 101, 354, 10089, 10090, 10186, 10197}:
            record.capacity_denied = True

    def get(self, key: str) -> SubscriptionRecord | None:
        record = self.records.get(key)
        return None if record is None or not record.active else record

    def upgrade(
        self,
        key: str,
        *,
        priority: SubscriptionPriority,
        owner_episode: str | None,
        protected: bool,
    ) -> SubscriptionRecord:
        record = self.get(key)
        if record is None:
            raise ValueError("cannot upgrade an inactive subscription")
        if priority < record.priority:
            raise ValueError("subscription priority cannot be reduced through upgrade")
        owner_id = (
            f"episode:{owner_episode}"
            if owner_episode is not None
            else next(iter(record.owners), f"anonymous:{key}")
        )
        resolved_class = _class_for_priority(priority)
        record.owners[owner_id] = resolved_class
        record.owner_priorities[owner_id] = priority
        if protected:
            record.protected_owners.add(owner_id)
        record.priority = max(record.owner_priorities.values())
        record.subscription_class = min(record.owners.values())
        record.owner_episode = owner_episode or record.owner_episode
        record.protected = bool(record.protected_owners)
        return record

    def reconcile(
        self,
        *,
        actual_request_ids: set[int],
        now_monotonic: float,
        pending_timeout_seconds: float,
    ) -> ReconciliationResult:
        if pending_timeout_seconds <= 0:
            raise ValueError("pending timeout must be positive")
        warnings: list[ReconciliationWarning] = []
        released: list[str] = []
        for record in sorted(self._active_records(), key=lambda item: item.key):
            if (
                record.status is SubscriptionStatus.PENDING
                and now_monotonic - record.started_monotonic > pending_timeout_seconds
            ):
                warnings.append(
                    ReconciliationWarning(
                        "stale_pending_reservation",
                        record.key,
                        None if record.request_id < 0 else record.request_id,
                        True,
                    )
                )
                self.cancel(record.key, reason="reconciled_stale_pending")
                released.append(record.key)
                continue
            if (
                record.status is SubscriptionStatus.ACTIVE
                and record.request_id not in actual_request_ids
            ):
                warnings.append(
                    ReconciliationWarning(
                        "active_internal_request_missing_upstream",
                        record.key,
                        record.request_id,
                        True,
                    )
                )
                self.cancel(record.key, reason="reconciled_missing_upstream")
                released.append(record.key)
        known = {record.request_id for record in self._active_records() if record.request_id >= 0}
        orphans = tuple(sorted(actual_request_ids.difference(known)))
        warnings.extend(
            ReconciliationWarning(
                "actual_request_without_internal_owner",
                None,
                request_id,
                False,
            )
            for request_id in orphans
        )
        ordered_warnings = tuple(
            sorted(
                warnings,
                key=lambda item: (
                    item.code,
                    item.key or "",
                    -1 if item.request_id is None else item.request_id,
                ),
            )
        )
        self.reconciliation_warnings.extend(ordered_warnings)
        return ReconciliationResult(
            warnings=ordered_warnings,
            released_keys=tuple(released),
            orphan_request_ids=orphans,
        )

    def restore_order(
        self,
        *,
        active_episode_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Return reconnect restoration order without expired episode streams."""

        active_ids = active_episode_ids
        candidates = []
        for record in self._active_records():
            if (
                active_ids is not None
                and record.subscription_class >= SubscriptionClass.ACTIVE_EPISODE
                and record.owner_episode is not None
                and record.owner_episode not in active_ids
            ):
                continue
            candidates.append(record)
        return tuple(
            record.key
            for record in sorted(
                candidates,
                key=lambda item: (
                    int(item.subscription_class),
                    -int(item.priority),
                    item.key,
                ),
            )
        )

    def snapshot(self) -> dict[str, object]:
        active = self._active_records()
        counts = {
            kind.value: sum(record.kind is kind for record in active) for kind in SubscriptionKind
        }
        class_counts = {
            f"class_{int(subscription_class)}": sum(
                record.subscription_class is subscription_class for record in active
            )
            for subscription_class in SubscriptionClass
        }
        symbols: dict[str, int] = {}
        episodes: dict[str, int] = {}
        for record in active:
            symbols[record.symbol] = symbols.get(record.symbol, 0) + record.line_cost
            if record.owner_episode is not None:
                episodes[record.owner_episode] = (
                    episodes.get(record.owner_episode, 0) + record.line_cost
                )
        oldest_optional = min(
            (
                record.started_at_utc
                for record in active
                if record.subscription_class >= SubscriptionClass.EPISODE_ENGINEERING
            ),
            default=None,
        )
        usage = self._current_usage()
        return {
            "budget_state": self._state().value,
            "active": counts,
            "limits": {kind.value: value for kind, value in self.limits.items()},
            "total_line_limit": self.total_line_limit,
            "externally_reserved_lines": self.externally_reserved_lines,
            "reserved_future_trading_lines": self.future_trading_reserve_lines,
            "safety_margin_lines": self.safety_margin_lines,
            "usable_research_lines": self.usable_research_lines,
            "current_internal_usage": usage,
            "available_research_lines": max(0, self.usable_research_lines - usage),
            "request_rate": len(self._request_times),
            "pending_requests": sum(
                record.status is SubscriptionStatus.PENDING for record in active
            ),
            "awaiting_cancellation": sum(
                record.status is SubscriptionStatus.CANCELLATION_REQUESTED for record in active
            ),
            "capacity_denials": len(self.capacity_denials),
            "subscriptions_by_priority_class": class_counts,
            "subscriptions_by_symbol": dict(sorted(symbols.items())),
            "subscriptions_by_episode": dict(sorted(episodes.items())),
            "oldest_active_optional_subscription": (
                None if oldest_optional is None else oldest_optional.isoformat()
            ),
            "reconciliation_warnings": len(self.reconciliation_warnings),
        }


class PromotionScheduler:
    """Operational scheduler only; promotion creates no scientific claim."""

    def __init__(
        self,
        *,
        max_tick_by_tick: int,
        max_depth: int,
        max_level1: int = 0,
        quiet_arming_threshold: float = 0.167095528962669,
        high_arming_threshold: float = 0.40,
    ) -> None:
        if min(max_tick_by_tick, max_depth, max_level1) < 0:
            raise ValueError("promotion capacities must be nonnegative")
        if not (0.0 <= quiet_arming_threshold <= high_arming_threshold <= 1.0):
            raise ValueError("promotion arming thresholds are invalid")
        self.max_tick_by_tick = max_tick_by_tick
        self.max_depth = max_depth
        self.max_level1 = max_level1
        self.quiet_arming_threshold = quiet_arming_threshold
        self.high_arming_threshold = high_arming_threshold

    def rank_checkpoint(
        self,
        *,
        checkpoint_time: datetime,
        probabilities: dict[str, float],
        eligible_symbols: set[str],
        active_episode_symbols: set[str],
    ) -> tuple[PromotionDecision, ...]:
        candidates = sorted(
            (
                (symbol, probability)
                for symbol, probability in probabilities.items()
                if symbol in eligible_symbols and symbol not in active_episode_symbols
            ),
            key=lambda item: (-item[1], item[0]),
        )
        arming_candidates = sorted(
            (
                (symbol, probability)
                for symbol, probability in candidates
                if probability <= self.quiet_arming_threshold
                or probability >= self.high_arming_threshold
            ),
            key=lambda item: (
                0 if item[1] <= self.quiet_arming_threshold else 1,
                item[1] if item[1] <= self.quiet_arming_threshold else -item[1],
                item[0],
            ),
        )
        decisions: list[PromotionDecision] = []
        families: tuple[tuple[str, int, list[tuple[str, float]]], ...] = (
            ("level1", self.max_level1, arming_candidates),
            ("tick_by_tick", self.max_tick_by_tick, candidates),
            ("depth", self.max_depth, candidates),
        )
        for subscription_type, capacity, family_candidates in families:
            for rank, (symbol, probability) in enumerate(
                family_candidates[:capacity],
                start=1,
            ):
                decisions.append(
                    PromotionDecision(
                        promotion_time=checkpoint_time,
                        symbol=symbol,
                        m1c_probability=probability,
                        rank=rank,
                        capacity_available=capacity,
                        subscription_type=subscription_type,
                        reason="checkpoint_ranked_armed_candidate",
                    )
                )
        return tuple(decisions)


__all__ = [
    "AllocationDecision",
    "BudgetState",
    "PromotionDecision",
    "PromotionScheduler",
    "ReconciliationResult",
    "ReconciliationWarning",
    "ReleaseDecision",
    "SubscriptionBudgetManager",
    "SubscriptionClass",
    "SubscriptionKind",
    "SubscriptionPriority",
    "SubscriptionRecord",
    "SubscriptionStatus",
    "canonical_subscription_key",
]
