"""One bounded market-data budget and deterministic promotion scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum


class SubscriptionKind(StrEnum):
    LEVEL1 = "level1"
    TICK_BY_TICK = "tick_by_tick"
    DEPTH = "depth"
    OPTION = "option"
    BAR = "bar"
    MARKET_PROXY = "market_proxy"


class SubscriptionPriority(IntEnum):
    UNIVERSE_LEVEL1 = 10
    MARKET_PROXY = 20
    ARMED_CANDIDATE = 30
    ACTIVE_OPTION = 40
    ACTIVE_EPISODE = 50


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


@dataclass(frozen=True)
class AllocationDecision:
    accepted: bool
    key: str
    reason: str
    evicted_key: str | None


@dataclass(frozen=True)
class PromotionDecision:
    promotion_time: datetime
    symbol: str
    m1c_probability: float
    rank: int
    capacity_available: int
    subscription_type: str
    reason: str


class SubscriptionBudgetManager:
    """Track all data subscriptions; never evict protected universe Level I."""

    def __init__(
        self,
        *,
        limits: dict[SubscriptionKind, int],
        request_rate_limit: int,
        request_rate_window_seconds: float = 1.0,
    ) -> None:
        if request_rate_limit <= 0 or request_rate_window_seconds <= 0:
            raise ValueError("request rate bounds must be positive")
        if any(limit < 0 for limit in limits.values()):
            raise ValueError("subscription limits must be nonnegative")
        self.limits = dict(limits)
        self.request_rate_limit = request_rate_limit
        self.request_rate_window_seconds = request_rate_window_seconds
        self.records: dict[str, SubscriptionRecord] = {}
        self.lifecycle: list[SubscriptionRecord] = []
        self.capacity_denials: list[dict[str, object]] = []
        self._request_times: deque[float] = deque()

    def _trim_rate(self, now: float) -> None:
        while self._request_times and self._request_times[0] <= (
            now - self.request_rate_window_seconds
        ):
            self._request_times.popleft()

    def _active_for_kind(self, kind: SubscriptionKind) -> list[SubscriptionRecord]:
        return [
            record
            for record in self.records.values()
            if record.kind is kind and record.cancelled_at_utc is None
        ]

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
        protected: bool = False,
        now_monotonic: float,
        now_utc: datetime | None = None,
    ) -> AllocationDecision:
        existing = self.records.get(key)
        if existing is not None and existing.cancelled_at_utc is None:
            return AllocationDecision(True, key, "already_active", None)
        self._trim_rate(now_monotonic)
        if len(self._request_times) >= self.request_rate_limit:
            return self._deny(key, kind, symbol, "request_rate_capacity_denied")
        limit = self.limits.get(kind, 0)
        active = self._active_for_kind(kind)
        evicted: SubscriptionRecord | None = None
        if len(active) >= limit:
            candidates = [
                record for record in active if not record.protected and record.priority < priority
            ]
            if candidates:
                evicted = min(
                    candidates,
                    key=lambda item: (
                        int(item.priority),
                        item.started_monotonic,
                        item.symbol,
                        item.key,
                    ),
                )
                self.cancel(
                    evicted.key,
                    reason=f"preempted_by:{key}",
                    now_utc=now_utc,
                )
            else:
                return self._deny(key, kind, symbol, "subscription_capacity_denied")
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
        )
        self.records[key] = record
        self.lifecycle.append(record)
        self._request_times.append(now_monotonic)
        return AllocationDecision(
            True,
            key,
            "allocated",
            None if evicted is None else evicted.key,
        )

    def _deny(
        self,
        key: str,
        kind: SubscriptionKind,
        symbol: str,
        reason: str,
    ) -> AllocationDecision:
        self.capacity_denials.append(
            {
                "key": key,
                "kind": kind.value,
                "symbol": symbol,
                "reason": reason,
            }
        )
        return AllocationDecision(False, key, reason, None)

    def cancel(
        self,
        key: str,
        *,
        reason: str,
        now_utc: datetime | None = None,
    ) -> bool:
        record = self.records.get(key)
        if record is None or record.cancelled_at_utc is not None:
            return False
        record.cancelled_at_utc = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
        record.cancellation_reason = reason
        return True

    def note_ibkr_error(self, key: str, code: int) -> None:
        record = self.records.get(key)
        if record is None:
            self.capacity_denials.append(
                {"key": key, "kind": None, "symbol": None, "reason": f"ibkr_error_{code}"}
            )
            return
        record.ibkr_error_codes = (*record.ibkr_error_codes, code)
        if code in {100, 101, 354, 10089, 10090, 10186, 10197}:
            record.capacity_denied = True

    def get(self, key: str) -> SubscriptionRecord | None:
        record = self.records.get(key)
        return None if record is None or record.cancelled_at_utc is not None else record

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
        record.priority = priority
        record.owner_episode = owner_episode
        record.protected = protected
        return record

    def snapshot(self) -> dict[str, object]:
        counts = {kind.value: len(self._active_for_kind(kind)) for kind in SubscriptionKind}
        return {
            "active": counts,
            "limits": {kind.value: value for kind, value in self.limits.items()},
            "request_rate": len(self._request_times),
            "capacity_denials": len(self.capacity_denials),
        }


class PromotionScheduler:
    """Operational scheduler only; promotion creates no scientific claim."""

    def __init__(self, *, max_tick_by_tick: int, max_depth: int) -> None:
        if max_tick_by_tick < 0 or max_depth < 0:
            raise ValueError("promotion capacities must be nonnegative")
        self.max_tick_by_tick = max_tick_by_tick
        self.max_depth = max_depth

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
        decisions: list[PromotionDecision] = []
        for subscription_type, capacity in (
            ("tick_by_tick", self.max_tick_by_tick),
            ("depth", self.max_depth),
        ):
            for rank, (symbol, probability) in enumerate(candidates[:capacity], start=1):
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
