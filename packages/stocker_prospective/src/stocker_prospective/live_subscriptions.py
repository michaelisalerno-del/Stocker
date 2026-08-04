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
    BudgetState,
    PromotionDecision,
    ReconciliationResult,
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
    SubscriptionRecord,
    canonical_subscription_key,
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


@dataclass(frozen=True)
class UnderlyingPromotionResult:
    symbol: str
    episode_id: str
    level1_started: bool
    approved_keys: tuple[str, ...]
    denied_keys: tuple[str, ...]
    budget_state: BudgetState


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
        depth_phase_permitted: Callable[[EvidenceMetadata], bool] | None = None,
        stream_registration_sink: Callable[[StreamOwner], None] | None = None,
        request_pacer: Callable[[], object] | None = None,
        historical_request_pacer: Callable[[], object] | None = None,
    ) -> None:
        if depth_rows <= 0:
            raise ValueError("depth rows must be positive")
        self.adapter = adapter
        self.budget = budget
        self.normalizer = normalizer
        self.repository = repository
        self.depth_rows = depth_rows
        self.enable_depth = enable_depth
        self.depth_phase_permitted = depth_phase_permitted
        self.stream_registration_sink = (
            normalizer.register if stream_registration_sink is None else stream_registration_sink
        )
        self.request_pacer = request_pacer
        self.historical_request_pacer = historical_request_pacer
        self._owned: dict[str, _OwnedStream] = {}
        self._contracts: dict[str, QualifiedUnderlying] = {}

    def _depth_permitted(self, metadata: EvidenceMetadata) -> bool:
        if not self.enable_depth:
            return False
        return True if self.depth_phase_permitted is None else self.depth_phase_permitted(metadata)

    def _allocate(
        self,
        metadata: EvidenceMetadata,
        *,
        key: str,
        contract: QualifiedUnderlying,
        budget_kind: SubscriptionKind,
        stream_kind: StreamKind,
        priority: SubscriptionPriority,
        subscription_class: SubscriptionClass,
        owner_id: str,
        owner_episode: str | None,
        protected: bool,
        tick_type: Literal["BidAsk", "Last"] | None = None,
        depth_rows: int | None = None,
    ) -> SubscriptionRecord | None:
        existing = self.budget.get(key)
        decision = self.budget.allocate(
            key=key,
            kind=budget_kind,
            symbol=contract.symbol,
            con_id=contract.con_id,
            request_id=-1,
            priority=priority,
            owner_episode=owner_episode,
            owner_id=owner_id,
            subscription_class=subscription_class,
            protected=protected,
            now_monotonic=time.monotonic(),
            now_utc=metadata.recorded_at_utc,
        )
        if not decision.accepted:
            return None
        if existing is not None:
            self.repository.record_subscription(metadata, existing)
            return existing
        for evicted_key in decision.evicted_keys:
            self._cancel_upstream(
                metadata,
                evicted_key,
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
                if self.historical_request_pacer is not None:
                    self.historical_request_pacer()
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
        except Exception as exc:
            self.budget.cancel(
                key,
                reason="upstream_subscription_failed",
                now_utc=metadata.recorded_at_utc,
            )
            capacity_failure = (
                "market_data_budget" in str(exc)
                or "capacity" in str(exc)
                or "market_data_lines" in str(exc)
            )
            if capacity_failure and subscription_class >= SubscriptionClass.ACTIVE_EPISODE:
                return None
            if capacity_failure:
                raise RuntimeError(f"critical_budget_unavailable:upstream:{key}") from exc
            raise
        record = self.budget.get(key)
        assert record is not None
        if not self.budget.mark_active(key, request_id=request_id):
            self._cancel_request(stream_kind, request_id, key)
            return None
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

    def _cancel_request(
        self,
        stream_kind: StreamKind,
        request_id: int,
        key: str,
    ) -> None:
        if stream_kind in {StreamKind.UNDERLYING_LEVEL1, StreamKind.OPTION_LEVEL1}:
            self.adapter.cancel_market_data(request_id, subscription_key=key)
        elif stream_kind is StreamKind.UNDERLYING_BAR:
            self.adapter.cancel_historical_updates(request_id, subscription_key=key)
        elif stream_kind in {
            StreamKind.UNDERLYING_TICK_BIDASK,
            StreamKind.UNDERLYING_TICK_LAST,
        }:
            self.adapter.cancel_tick_by_tick(request_id, subscription_key=key)
        elif stream_kind is StreamKind.UNDERLYING_DEPTH:
            self.adapter.cancel_market_depth(request_id, subscription_key=key)

    def start_always_on(
        self,
        metadata: EvidenceMetadata,
        contracts: tuple[QualifiedUnderlying, ...],
    ) -> None:
        if len({item.symbol for item in contracts}) != len(contracts):
            raise ValueError("qualified underlying symbols are not unique")
        self._contracts.update({item.symbol: item for item in contracts})
        failed_bars: list[str] = []
        for contract in sorted(contracts, key=lambda item: item.symbol):
            priority = (
                SubscriptionPriority.MARKET_PROXY
                if contract.market_proxy
                else SubscriptionPriority.UNIVERSE_LEVEL1
            )
            subscription_class = (
                SubscriptionClass.CRITICAL_SYSTEM
                if contract.market_proxy
                else SubscriptionClass.FROZEN_UNIVERSE_SIGNAL
            )
            bar = self._allocate(
                metadata,
                key=canonical_subscription_key(
                    SubscriptionKind.BAR,
                    con_id=contract.con_id,
                    bar_size="5m",
                    use_rth=True,
                ),
                contract=contract,
                budget_kind=SubscriptionKind.BAR,
                stream_kind=StreamKind.UNDERLYING_BAR,
                priority=priority,
                subscription_class=subscription_class,
                owner_id=(
                    f"system:market_proxy:{contract.symbol}"
                    if contract.market_proxy
                    else f"universe:{contract.symbol}"
                ),
                owner_episode=None,
                protected=True,
            )
            if bar is None:
                failed_bars.append(contract.symbol)
        if failed_bars:
            raise RuntimeError(
                "critical_budget_unavailable:required_five_minute_bars:"
                + ",".join(sorted(failed_bars))
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
                and f"arming:{stream.symbol}" in record.owners
                and stream.stream_kind
                in {
                    StreamKind.UNDERLYING_LEVEL1,
                    StreamKind.UNDERLYING_TICK_BIDASK,
                    StreamKind.UNDERLYING_TICK_LAST,
                    StreamKind.UNDERLYING_DEPTH,
                }
            ):
                if stream.stream_kind is StreamKind.UNDERLYING_LEVEL1:
                    family = "level1"
                elif stream.stream_kind is StreamKind.UNDERLYING_DEPTH:
                    family = "depth"
                else:
                    family = "tick_by_tick"
                if (stream.symbol, family) not in desired:
                    self._release_owner(
                        metadata,
                        key,
                        owner_id=f"arming:{stream.symbol}",
                        reason="next_checkpoint_rank_replaced",
                    )
        for decision in decisions:
            contract = self._contracts.get(decision.symbol)
            if contract is None:
                continue
            if decision.subscription_type == "level1":
                self._allocate(
                    metadata,
                    key=canonical_subscription_key(
                        SubscriptionKind.LEVEL1,
                        con_id=contract.con_id,
                    ),
                    contract=contract,
                    budget_kind=SubscriptionKind.LEVEL1,
                    stream_kind=StreamKind.UNDERLYING_LEVEL1,
                    priority=SubscriptionPriority.EPISODE_ENGINEERING,
                    subscription_class=SubscriptionClass.EPISODE_ENGINEERING,
                    owner_id=f"arming:{contract.symbol}",
                    owner_episode=None,
                    protected=False,
                )
            elif decision.subscription_type == "tick_by_tick":
                started: list[str] = []
                for _suffix, stream_kind, tick_type in TICK_STREAMS:
                    key = canonical_subscription_key(
                        SubscriptionKind.TICK_BY_TICK,
                        con_id=contract.con_id,
                        tick_type=tick_type,
                    )
                    record = self._allocate(
                        metadata,
                        key=key,
                        contract=contract,
                        budget_kind=SubscriptionKind.TICK_BY_TICK,
                        stream_kind=stream_kind,
                        priority=SubscriptionPriority.ARMED_CANDIDATE,
                        subscription_class=SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT,
                        owner_id=f"arming:{contract.symbol}",
                        owner_episode=None,
                        protected=False,
                        tick_type=tick_type,
                    )
                    if record is None:
                        for started_key in started:
                            self._release_owner(
                                metadata,
                                started_key,
                                owner_id=f"arming:{contract.symbol}",
                                reason="paired_tick_capacity_denied",
                            )
                        break
                    started.append(key)
            elif decision.subscription_type == "depth" and self._depth_permitted(metadata):
                self._allocate(
                    metadata,
                    key=canonical_subscription_key(
                        SubscriptionKind.DEPTH,
                        con_id=contract.con_id,
                        depth_rows=self.depth_rows,
                        smart_depth=True,
                    ),
                    contract=contract,
                    budget_kind=SubscriptionKind.DEPTH,
                    stream_kind=StreamKind.UNDERLYING_DEPTH,
                    priority=SubscriptionPriority.ARMED_CANDIDATE,
                    subscription_class=SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT,
                    owner_id=f"arming:{contract.symbol}",
                    owner_episode=None,
                    protected=False,
                    depth_rows=self.depth_rows,
                )

    def promote_opening_leader_underlying(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
        selection_id: str,
    ) -> UnderlyingPromotionResult:
        """Keep one replaceable, record-only L1 stream for the selected leader."""

        contract = self._contracts[symbol]
        owner_id = "research:opening-leader-continuation-v0"
        level1_key = canonical_subscription_key(
            SubscriptionKind.LEVEL1,
            con_id=contract.con_id,
        )
        for key, record in tuple(self.budget.records.items()):
            if record.active and owner_id in record.owners and key != level1_key:
                self._release_owner(
                    metadata,
                    key,
                    owner_id=owner_id,
                    reason=f"opening_leader_replaced_by:{selection_id}",
                )
        level1 = self._allocate(
            metadata,
            key=level1_key,
            contract=contract,
            budget_kind=SubscriptionKind.LEVEL1,
            stream_kind=StreamKind.UNDERLYING_LEVEL1,
            priority=SubscriptionPriority.ACTIVE_EPISODE,
            subscription_class=SubscriptionClass.ACTIVE_EPISODE,
            owner_id=owner_id,
            owner_episode=selection_id,
            protected=True,
        )
        if level1 is None:
            return UnderlyingPromotionResult(
                symbol=symbol,
                episode_id=selection_id,
                level1_started=False,
                approved_keys=(),
                denied_keys=(level1_key,),
                budget_state=BudgetState.OPTION_EPISODE_QUEUED,
            )
        return UnderlyingPromotionResult(
            symbol=symbol,
            episode_id=selection_id,
            level1_started=True,
            approved_keys=(level1_key,),
            denied_keys=(),
            budget_state=BudgetState.BUDGET_HEALTHY,
        )

    def promote_active_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
        episode_id: str,
    ) -> UnderlyingPromotionResult:
        contract = self._contracts[symbol]
        owner_id = f"episode:{episode_id}"
        approved: list[str] = []
        denied: list[str] = []
        level1_key = canonical_subscription_key(
            SubscriptionKind.LEVEL1,
            con_id=contract.con_id,
        )
        level1 = self._allocate(
            metadata,
            key=level1_key,
            contract=contract,
            budget_kind=SubscriptionKind.LEVEL1,
            stream_kind=StreamKind.UNDERLYING_LEVEL1,
            priority=SubscriptionPriority.ACTIVE_EPISODE,
            subscription_class=SubscriptionClass.ACTIVE_EPISODE,
            owner_id=owner_id,
            owner_episode=episode_id,
            protected=True,
        )
        if level1 is None:
            record_skipped = getattr(
                self.repository,
                "record_skipped_recording",
                None,
            )
            if callable(record_skipped):
                record_skipped(
                    metadata,
                    session=metadata.recorded_at_utc.date(),
                    episode_id=episode_id,
                    symbol=symbol,
                    recording_kind="underlying_level1_promotion",
                    reason="option_episode_queued:underlying_level1_capacity",
                    requested_payload={"requested_subscriptions": [level1_key]},
                )
            return UnderlyingPromotionResult(
                symbol=symbol,
                episode_id=episode_id,
                level1_started=False,
                approved_keys=(),
                denied_keys=(level1_key,),
                budget_state=BudgetState.OPTION_EPISODE_QUEUED,
            )
        approved.append(level1_key)
        state = BudgetState.BUDGET_HEALTHY
        started: list[str] = []
        for _suffix, stream_kind, tick_type in TICK_STREAMS:
            key = canonical_subscription_key(
                SubscriptionKind.TICK_BY_TICK,
                con_id=contract.con_id,
                tick_type=tick_type,
            )
            record = self._allocate(
                metadata,
                key=key,
                contract=contract,
                budget_kind=SubscriptionKind.TICK_BY_TICK,
                stream_kind=stream_kind,
                priority=SubscriptionPriority.MICROSTRUCTURE_ENHANCEMENT,
                subscription_class=SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT,
                owner_id=owner_id,
                owner_episode=episode_id,
                protected=False,
                tick_type=tick_type,
            )
            if record is None:
                denied.append(key)
                state = BudgetState.OPTIONAL_FEEDS_DEGRADED
                for started_key in started:
                    self._release_owner(
                        metadata,
                        started_key,
                        owner_id=owner_id,
                        reason="active_episode_paired_tick_capacity_denied",
                    )
                    if started_key in approved:
                        approved.remove(started_key)
                break
            started.append(key)
            approved.append(key)
        if self.enable_depth:
            depth_key = canonical_subscription_key(
                SubscriptionKind.DEPTH,
                con_id=contract.con_id,
                depth_rows=self.depth_rows,
                smart_depth=True,
            )
            if not self._depth_permitted(metadata):
                denied.append(depth_key)
                state = BudgetState.OPTIONAL_FEEDS_DEGRADED
            else:
                depth = self._allocate(
                    metadata,
                    key=depth_key,
                    contract=contract,
                    budget_kind=SubscriptionKind.DEPTH,
                    stream_kind=StreamKind.UNDERLYING_DEPTH,
                    priority=SubscriptionPriority.MICROSTRUCTURE_ENHANCEMENT,
                    subscription_class=SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT,
                    owner_id=owner_id,
                    owner_episode=episode_id,
                    protected=False,
                    depth_rows=self.depth_rows,
                )
                if depth is None:
                    denied.append(depth_key)
                    state = BudgetState.OPTIONAL_FEEDS_DEGRADED
                else:
                    approved.append(depth_key)
        result = UnderlyingPromotionResult(
            symbol=symbol,
            episode_id=episode_id,
            level1_started=True,
            approved_keys=tuple(approved),
            denied_keys=tuple(denied),
            budget_state=state,
        )
        if denied:
            record_skipped = getattr(
                self.repository,
                "record_skipped_recording",
                None,
            )
            if callable(record_skipped):
                record_skipped(
                    metadata,
                    session=metadata.recorded_at_utc.date(),
                    episode_id=episode_id,
                    symbol=symbol,
                    recording_kind="optional_microstructure_promotion",
                    reason="optional_feeds_degraded",
                    requested_payload={
                        "approved_subscriptions": approved,
                        "denied_subscriptions": denied,
                    },
                )
        return result

    def resubscribe_depth(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
    ) -> bool:
        """Discard an invalid book stream and request a fresh deterministic book."""

        contract = self._contracts.get(symbol)
        if contract is None:
            return False
        key = canonical_subscription_key(
            SubscriptionKind.DEPTH,
            con_id=contract.con_id,
            depth_rows=self.depth_rows,
            smart_depth=True,
        )
        owned = self._owned.get(key)
        record = self.budget.get(key)
        if owned is None or record is None:
            return False
        owners = tuple(
            (
                owner_id,
                owner_class,
                record.owner_priorities[owner_id],
                owner_id in record.protected_owners,
            )
            for owner_id, owner_class in sorted(record.owners.items())
        )
        owner_episode = record.owner_episode
        self.cancel(metadata, key, reason="depth_reset_resubscribe")
        restored = False
        for owner_id, owner_class, priority, protected in owners:
            restored_record = self._allocate(
                metadata,
                key=key,
                contract=owned.contract,
                budget_kind=SubscriptionKind.DEPTH,
                stream_kind=StreamKind.UNDERLYING_DEPTH,
                priority=priority,
                subscription_class=owner_class,
                owner_id=owner_id,
                owner_episode=owner_episode,
                protected=protected,
                depth_rows=self.depth_rows,
            )
            if restored_record is None:
                return False
            restored = True
        return restored

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

    def _release_owner(
        self,
        metadata: EvidenceMetadata,
        key: str,
        *,
        owner_id: str,
        reason: str,
    ) -> None:
        decision = self.budget.release(
            key,
            owner_id=owner_id,
            reason=reason,
            now_utc=metadata.recorded_at_utc,
        )
        record = self.budget.records.get(key)
        if decision.cancel_upstream:
            self._cancel_upstream(
                metadata,
                key,
                reason=reason,
                budget_already_cancelled=True,
            )
        elif record is not None:
            self.repository.record_subscription(metadata, record)

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
        self._cancel_request(owned.stream_kind, owned.request_id, key)
        self.normalizer.unregister(owned.request_id)
        if not budget_already_cancelled:
            self.budget.cancel(
                key,
                reason=reason,
                now_utc=metadata.recorded_at_utc,
            )
        self.repository.record_subscription(metadata, record)

    def cancel_evicted_subscription(
        self,
        metadata: EvidenceMetadata,
        *,
        key: str,
        replacement_key: str,
    ) -> bool:
        """Cancel a broker stream already shed by the shared budget registry."""

        if key not in self._owned:
            return False
        self._cancel_upstream(
            metadata,
            key,
            reason=f"preempted_by:{replacement_key}",
            budget_already_cancelled=True,
        )
        return True

    def end_active_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
    ) -> None:
        owner_id = f"episode:{episode_id}"
        for key, record in tuple(self.budget.records.items()):
            if record.active and owner_id in record.owners:
                self._release_owner(
                    metadata,
                    key,
                    owner_id=owner_id,
                    reason="active_episode_T_plus_30_complete",
                )

    def rebuild_after_data_loss(
        self,
        metadata: EvidenceMetadata,
    ) -> None:
        specifications = []
        for owned in self._owned.values():
            record = self.budget.records[owned.key]
            specifications.append(
                (
                    owned,
                    tuple(
                        (
                            owner_id,
                            owner_class,
                            record.owner_priorities[owner_id],
                            owner_id in record.protected_owners,
                        )
                        for owner_id, owner_class in sorted(record.owners.items())
                    ),
                    record.owner_episode,
                    record.subscription_class,
                )
            )
        specifications.sort(key=lambda item: (int(item[3]), item[0].key))
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
        for owned, owners, owner_episode, _ in specifications:
            for owner_id, owner_class, priority, protected in owners:
                if (
                    self._allocate(
                        metadata,
                        key=owned.key,
                        contract=owned.contract,
                        budget_kind=owned.budget_kind,
                        stream_kind=owned.stream_kind,
                        priority=priority,
                        subscription_class=owner_class,
                        owner_id=owner_id,
                        owner_episode=owner_episode,
                        protected=protected,
                        tick_type=owned.tick_type,
                        depth_rows=owned.depth_rows,
                    )
                    is None
                ):
                    break

    def reconcile(
        self,
        metadata: EvidenceMetadata,
        *,
        actual_request_ids: set[int],
        pending_timeout_seconds: float,
    ) -> ReconciliationResult:
        """Repair stale reservations and cancel adapter-local orphan requests."""

        result = self.budget.reconcile(
            actual_request_ids=actual_request_ids,
            now_monotonic=time.monotonic(),
            pending_timeout_seconds=pending_timeout_seconds,
        )
        for key in result.released_keys:
            if key in self._owned:
                self._cancel_upstream(
                    metadata,
                    key,
                    reason="periodic_reconciliation_repair",
                    budget_already_cancelled=True,
                )
        cancel_orphan = getattr(
            self.adapter,
            "cancel_orphaned_market_data_request",
            None,
        )
        if callable(cancel_orphan):
            for request_id in result.orphan_request_ids:
                cancel_orphan(request_id)
                self.normalizer.unregister(request_id)
        for warning in result.warnings:
            record_skipped = getattr(
                self.repository,
                "record_skipped_recording",
                None,
            )
            if callable(record_skipped):
                record_skipped(
                    metadata,
                    session=metadata.recorded_at_utc.date(),
                    recording_kind="subscription_reconciliation_warning",
                    reason=warning.code,
                    requested_payload={
                        "subscription_key": warning.key,
                        "request_id": warning.request_id,
                        "repaired": warning.repaired,
                    },
                )
        return result

    def shutdown(self, metadata: EvidenceMetadata) -> None:
        for key in tuple(self._owned):
            self.cancel(metadata, key, reason="recorder_shutdown")


__all__ = [
    "LiveSubscriptionController",
    "QualifiedUnderlying",
    "UnderlyingPromotionResult",
]
