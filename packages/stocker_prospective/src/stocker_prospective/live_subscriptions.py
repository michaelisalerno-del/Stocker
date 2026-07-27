"""One read-only controller for bounded underlying market-data subscriptions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
    StreamKind,
    StreamOwner,
)
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.subscriptions import (
    PromotionDecision,
    SubscriptionBudgetManager,
    SubscriptionKind,
    SubscriptionPriority,
    SubscriptionRecord,
)

TICK_STREAMS: tuple[
    tuple[
        str,
        StreamKind,
        Literal["BidAsk", "Last"],
    ],
    ...,
] = (
    ("bidask", StreamKind.UNDERLYING_TICK_BIDASK, "BidAsk"),
    ("last", StreamKind.UNDERLYING_TICK_LAST, "Last"),
)


@dataclass(frozen=True)
class QualifiedUnderlying:
    symbol: str
    con_id: int
    upstream_contract: Any
    exchange: str
    market_proxy: bool = False


@dataclass(frozen=True)
class _OwnedStream:
    key: str
    symbol: str
    request_id: int
    stream_kind: StreamKind
    budget_kind: SubscriptionKind
    contract: QualifiedUnderlying
    tick_type: Literal["BidAsk", "Last"] | None
    depth_rows: int | None


class LiveSubscriptionController:
    """Start, promote, cancel, and rebuild only market-data subscriptions."""

    def __init__(
        self,
        *,
        adapter: IBKRMarketDataAdapter,
        budget: SubscriptionBudgetManager,
        normalizer: IBKRCallbackNormalizer,
        repository: FrozenRecorderRepository,
        depth_rows: int,
        enable_depth: bool,
        stream_registration_sink: Callable[[StreamOwner], None] | None = None,
        request_pacer: Callable[[], object] | None = None,
    ) -> None:
        if depth_rows <= 0:
            raise ValueError("depth rows must be positive")
        self.adapter = adapter
        self.budget = budget
        self.normalizer = normalizer
        self.repository = repository
        self.depth_rows = depth_rows
        self.enable_depth = enable_depth
        self.stream_registration_sink = (
            normalizer.register if stream_registration_sink is None else stream_registration_sink
        )
        self.request_pacer = request_pacer
        self._owned: dict[str, _OwnedStream] = {}
        self._contracts: dict[str, QualifiedUnderlying] = {}

    def _allocate(
        self,
        metadata: EvidenceMetadata,
        *,
        key: str,
        contract: QualifiedUnderlying,
        budget_kind: SubscriptionKind,
        stream_kind: StreamKind,
        priority: SubscriptionPriority,
        owner_episode: str | None,
        protected: bool,
        tick_type: Literal["BidAsk", "Last"] | None = None,
        depth_rows: int | None = None,
    ) -> SubscriptionRecord | None:
        existing = self.budget.get(key)
        if existing is not None:
            if priority > existing.priority or owner_episode is not None:
                self.budget.upgrade(
                    key,
                    priority=priority,
                    owner_episode=owner_episode,
                    protected=protected,
                )
                self.repository.record_subscription(metadata, existing)
            return existing
        decision = self.budget.allocate(
            key=key,
            kind=budget_kind,
            symbol=contract.symbol,
            con_id=contract.con_id,
            request_id=-1,
            priority=priority,
            owner_episode=owner_episode,
            protected=protected,
            now_monotonic=time.monotonic(),
            now_utc=metadata.recorded_at_utc,
        )
        if not decision.accepted:
            return None
        if decision.evicted_key is not None:
            self._cancel_upstream(
                metadata,
                decision.evicted_key,
                reason=f"preempted_by:{key}",
                budget_already_cancelled=True,
            )
        try:
            if stream_kind in {
                StreamKind.UNDERLYING_LEVEL1,
                StreamKind.OPTION_LEVEL1,
            }:
                request_id = self.adapter.request_market_data(
                    contract.upstream_contract,
                    subscription_key=key,
                )
            elif stream_kind is StreamKind.UNDERLYING_BAR:
                request_id = self.adapter.request_historical_five_minute_updates(
                    contract.upstream_contract,
                    subscription_key=key,
                )
            elif stream_kind in {
                StreamKind.UNDERLYING_TICK_BIDASK,
                StreamKind.UNDERLYING_TICK_LAST,
            }:
                if tick_type is None:
                    raise ValueError("tick-by-tick subscription type is absent")
                request_id = self.adapter.request_tick_by_tick(
                    contract.upstream_contract,
                    subscription_key=key,
                    tick_type=tick_type,
                )
            elif stream_kind is StreamKind.UNDERLYING_DEPTH:
                request_id = self.adapter.request_market_depth(
                    contract.upstream_contract,
                    subscription_key=key,
                    rows=depth_rows or self.depth_rows,
                    smart_depth=True,
                )
            else:
                raise ValueError("unsupported live stream kind")
            if self.request_pacer is not None:
                self.request_pacer()
        except Exception:
            self.budget.cancel(
                key,
                reason="upstream_subscription_failed",
                now_utc=metadata.recorded_at_utc,
            )
            raise
        record = self.budget.get(key)
        assert record is not None
        record.request_id = request_id
        owned = _OwnedStream(
            key=key,
            symbol=contract.symbol,
            request_id=request_id,
            stream_kind=stream_kind,
            budget_kind=budget_kind,
            contract=contract,
            tick_type=tick_type,
            depth_rows=depth_rows,
        )
        self._owned[key] = owned
        self.stream_registration_sink(
            StreamOwner(
                request_id=request_id,
                kind=stream_kind,
                symbol=contract.symbol,
                con_id=contract.con_id,
                exchange=contract.exchange,
            )
        )
        self.repository.record_subscription(metadata, record)
        return record

    def start_always_on(
        self,
        metadata: EvidenceMetadata,
        contracts: tuple[QualifiedUnderlying, ...],
    ) -> None:
        if len({item.symbol for item in contracts}) != len(contracts):
            raise ValueError("qualified underlying symbols are not unique")
        self._contracts.update({item.symbol: item for item in contracts})
        failed_level1: list[str] = []
        failed_bars: list[str] = []
        for contract in sorted(contracts, key=lambda item: item.symbol):
            priority = (
                SubscriptionPriority.MARKET_PROXY
                if contract.market_proxy
                else SubscriptionPriority.UNIVERSE_LEVEL1
            )
            level1 = self._allocate(
                metadata,
                key=f"underlying:{contract.symbol}:level1",
                contract=contract,
                budget_kind=SubscriptionKind.LEVEL1,
                stream_kind=StreamKind.UNDERLYING_LEVEL1,
                priority=priority,
                owner_episode=None,
                protected=True,
            )
            if level1 is None:
                failed_level1.append(contract.symbol)
                continue
            bar = self._allocate(
                metadata,
                key=f"underlying:{contract.symbol}:bar",
                contract=contract,
                budget_kind=SubscriptionKind.BAR,
                stream_kind=StreamKind.UNDERLYING_BAR,
                priority=priority,
                owner_episode=None,
                protected=True,
            )
            if bar is None:
                failed_bars.append(contract.symbol)
        if failed_level1:
            raise RuntimeError(
                "blocked_universe_level1_capacity:" + ",".join(sorted(failed_level1))
            )
        if failed_bars:
            raise RuntimeError(
                "blocked_required_five_minute_bar_capacity:" + ",".join(sorted(failed_bars))
            )

    def apply_checkpoint_promotions(
        self,
        metadata: EvidenceMetadata,
        decisions: tuple[PromotionDecision, ...],
    ) -> None:
        for decision in decisions:
            self.repository.record_promotion_decision(metadata, decision)
        desired = {(decision.symbol, decision.subscription_type) for decision in decisions}
        for key, stream in tuple(self._owned.items()):
            record = self.budget.get(key)
            if (
                record is not None
                and record.priority is SubscriptionPriority.ARMED_CANDIDATE
                and stream.stream_kind
                in {
                    StreamKind.UNDERLYING_TICK_BIDASK,
                    StreamKind.UNDERLYING_TICK_LAST,
                    StreamKind.UNDERLYING_DEPTH,
                }
            ):
                family = (
                    "depth" if stream.stream_kind is StreamKind.UNDERLYING_DEPTH else "tick_by_tick"
                )
                if (stream.symbol, family) not in desired:
                    self.cancel(
                        metadata,
                        key,
                        reason="next_checkpoint_rank_replaced",
                    )
        for decision in decisions:
            contract = self._contracts.get(decision.symbol)
            if contract is None:
                continue
            if decision.subscription_type == "tick_by_tick":
                started: list[str] = []
                for suffix, stream_kind, tick_type in TICK_STREAMS:
                    key = f"underlying:{contract.symbol}:tbt:{suffix}"
                    record = self._allocate(
                        metadata,
                        key=key,
                        contract=contract,
                        budget_kind=SubscriptionKind.TICK_BY_TICK,
                        stream_kind=stream_kind,
                        priority=SubscriptionPriority.ARMED_CANDIDATE,
                        owner_episode=None,
                        protected=False,
                        tick_type=tick_type,
                    )
                    if record is None:
                        for started_key in started:
                            self.cancel(
                                metadata,
                                started_key,
                                reason="paired_tick_capacity_denied",
                            )
                        break
                    started.append(key)
            elif decision.subscription_type == "depth" and self.enable_depth:
                self._allocate(
                    metadata,
                    key=f"underlying:{contract.symbol}:depth",
                    contract=contract,
                    budget_kind=SubscriptionKind.DEPTH,
                    stream_kind=StreamKind.UNDERLYING_DEPTH,
                    priority=SubscriptionPriority.ARMED_CANDIDATE,
                    owner_episode=None,
                    protected=False,
                    depth_rows=self.depth_rows,
                )

    def promote_active_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
        episode_id: str,
    ) -> None:
        contract = self._contracts[symbol]
        started: list[str] = []
        for suffix, stream_kind, tick_type in TICK_STREAMS:
            key = f"underlying:{symbol}:tbt:{suffix}"
            record = self._allocate(
                metadata,
                key=key,
                contract=contract,
                budget_kind=SubscriptionKind.TICK_BY_TICK,
                stream_kind=stream_kind,
                priority=SubscriptionPriority.ACTIVE_EPISODE,
                owner_episode=episode_id,
                protected=True,
                tick_type=tick_type,
            )
            if record is None:
                for started_key in started:
                    self.cancel(
                        metadata,
                        started_key,
                        reason="active_episode_paired_tick_capacity_denied",
                    )
                break
            started.append(key)
        if self.enable_depth:
            self._allocate(
                metadata,
                key=f"underlying:{symbol}:depth",
                contract=contract,
                budget_kind=SubscriptionKind.DEPTH,
                stream_kind=StreamKind.UNDERLYING_DEPTH,
                priority=SubscriptionPriority.ACTIVE_EPISODE,
                owner_episode=episode_id,
                protected=True,
                depth_rows=self.depth_rows,
            )

    def resubscribe_depth(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
    ) -> bool:
        """Discard an invalid book stream and request a fresh deterministic book."""

        key = f"underlying:{symbol}:depth"
        owned = self._owned.get(key)
        record = self.budget.get(key)
        if owned is None or record is None:
            return False
        priority = record.priority
        owner_episode = record.owner_episode
        protected = record.protected
        contract = owned.contract
        self.cancel(metadata, key, reason="depth_reset_resubscribe")
        return (
            self._allocate(
                metadata,
                key=key,
                contract=contract,
                budget_kind=SubscriptionKind.DEPTH,
                stream_kind=StreamKind.UNDERLYING_DEPTH,
                priority=priority,
                owner_episode=owner_episode,
                protected=protected,
                depth_rows=self.depth_rows,
            )
            is not None
        )

    def record_ibkr_error(
        self,
        metadata: EvidenceMetadata,
        *,
        request_id: int,
        code: int,
    ) -> None:
        """Attach an IBKR error to its exact subscription lifecycle record."""

        record = next(
            (
                item
                for item in self.budget.records.values()
                if item.request_id == request_id and item.cancelled_at_utc is None
            ),
            None,
        )
        if record is None:
            return
        self.budget.note_ibkr_error(record.key, code)
        self.repository.record_subscription(metadata, record)

    def cancel(
        self,
        metadata: EvidenceMetadata,
        key: str,
        *,
        reason: str,
    ) -> None:
        self._cancel_upstream(
            metadata,
            key,
            reason=reason,
            budget_already_cancelled=False,
        )

    def _cancel_upstream(
        self,
        metadata: EvidenceMetadata,
        key: str,
        *,
        reason: str,
        budget_already_cancelled: bool,
    ) -> None:
        owned = self._owned.pop(key, None)
        record = self.budget.records.get(key)
        if owned is None or record is None:
            return
        if owned.stream_kind is StreamKind.UNDERLYING_LEVEL1:
            self.adapter.cancel_market_data(
                owned.request_id,
                subscription_key=key,
            )
        elif owned.stream_kind is StreamKind.UNDERLYING_BAR:
            self.adapter.cancel_historical_updates(
                owned.request_id,
                subscription_key=key,
            )
        elif owned.stream_kind in {
            StreamKind.UNDERLYING_TICK_BIDASK,
            StreamKind.UNDERLYING_TICK_LAST,
        }:
            self.adapter.cancel_tick_by_tick(
                owned.request_id,
                subscription_key=key,
            )
        elif owned.stream_kind is StreamKind.UNDERLYING_DEPTH:
            self.adapter.cancel_market_depth(
                owned.request_id,
                subscription_key=key,
            )
        self.normalizer.unregister(owned.request_id)
        if not budget_already_cancelled:
            self.budget.cancel(
                key,
                reason=reason,
                now_utc=metadata.recorded_at_utc,
            )
        self.repository.record_subscription(metadata, record)

    def end_active_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
    ) -> None:
        for key, record in tuple(self.budget.records.items()):
            if record.cancelled_at_utc is None and record.owner_episode == episode_id:
                self.cancel(
                    metadata,
                    key,
                    reason="active_episode_T_plus_30_complete",
                )

    def rebuild_after_data_loss(
        self,
        metadata: EvidenceMetadata,
    ) -> None:
        specifications = tuple(
            (
                owned,
                self.budget.records[owned.key].priority,
                self.budget.records[owned.key].owner_episode,
                self.budget.records[owned.key].protected,
            )
            for owned in self._owned.values()
        )
        for owned, _, _, _ in specifications:
            self.normalizer.unregister(owned.request_id)
            self.budget.cancel(
                owned.key,
                reason="data_lost_reconnect",
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.budget.records[owned.key],
            )
            self._owned.pop(owned.key, None)
        for owned, priority, owner_episode, protected in specifications:
            self._allocate(
                metadata,
                key=owned.key,
                contract=owned.contract,
                budget_kind=owned.budget_kind,
                stream_kind=owned.stream_kind,
                priority=priority,
                owner_episode=owner_episode,
                protected=protected,
                tick_type=owned.tick_type,
                depth_rows=owned.depth_rows,
            )

    def shutdown(self, metadata: EvidenceMetadata) -> None:
        for key in tuple(self._owned):
            self.cancel(metadata, key, reason="recorder_shutdown")


__all__ = ["LiveSubscriptionController", "QualifiedUnderlying"]
