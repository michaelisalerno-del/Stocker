"""Bounded post-episode option top-of-book recording coordinator."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.event_ingest import StreamKind, StreamOwner
from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import (
    DirectionalShadowSelection,
    OptionContract,
    OptionContractPlan,
    ShadowOptionOutcome,
    build_shadow_outcomes,
    map_directional_shadow,
    quote_quality_flags,
    retrospective_oracle,
    straddle_outcome,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.safety import (
    OptionOutcomeSafetyInputs,
    evaluate_option_outcome_safety,
)
from stocker_prospective.short_premium_shadow import (
    CreditShadowOutcome,
    DefinedRiskStructure,
    calculate_credit_shadow,
    select_delta_iron_condor,
    select_fixed_width_credit_spread,
    select_iron_butterfly,
)
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
    canonical_subscription_key,
)


@dataclass(frozen=True)
class ResolvedOptionContract:
    contract: OptionContract
    upstream_contract: Any


@dataclass
class _ActiveOption:
    database_id: int
    episode_id: str
    symbol: str
    contract: OptionContract
    upstream_contract: Any
    request_id: int
    subscription_key: str
    subscription_class: SubscriptionClass
    recording_ends_at_utc: datetime
    quiet_state: bool


def _resolved_contract_match(
    planned: OptionContract | None,
    resolved_contracts: tuple[OptionContract, ...],
) -> OptionContract | None:
    if planned is None:
        return None
    return next(
        (
            contract
            for contract in resolved_contracts
            if contract.expiry == planned.expiry
            and contract.strike == planned.strike
            and contract.right == planned.right
        ),
        None,
    )


def _planned_atm_contract(
    planned_contracts: tuple[OptionContract, ...],
    *,
    selection_roles: Mapping[str, tuple[str, ...]],
    right: Literal["C", "P"],
) -> OptionContract | None:
    marker = "atm_call" if right == "C" else "atm_put"
    return next(
        (
            contract
            for contract in planned_contracts
            if contract.right == right
            and any(marker in role for role in selection_roles.get(contract.con_id_key, ()))
        ),
        None,
    )


def _required_quote_matrix_completion(
    *,
    planned_contracts: tuple[OptionContract, ...],
    requested_contract_count: int,
    plan_capacity_reduced: bool,
    outcomes: tuple[ShadowOptionOutcome, ...],
    horizon_minutes: tuple[int, ...],
    subscription_gap_spans_horizon: bool,
) -> tuple[int, int, bool]:
    required = {
        (contract.expiry, contract.strike, contract.right, minutes)
        for contract in planned_contracts
        for minutes in horizon_minutes
    }
    complete = {
        (outcome.expiry, outcome.strike, outcome.right, outcome.horizon_minutes)
        for outcome in outcomes
        if outcome.entry_ask is not None and outcome.exit_bid is not None
    }
    complete_required = required.intersection(complete)
    return (
        requested_contract_count * len(horizon_minutes),
        len(complete_required),
        bool(required)
        and len(planned_contracts) == requested_contract_count
        and not plan_capacity_reduced
        and complete_required == required
        and not subscription_gap_spans_horizon,
    )


def _quiet_virtual_leg_evidence(
    *,
    structure: DefinedRiskStructure,
    entry_surface: tuple[OptionQuoteEvent, ...],
    exit_surface: tuple[OptionQuoteEvent, ...],
) -> list[dict[str, object]]:
    """Freeze the exact bid/ask inputs behind one quiet structure outcome."""

    entry_by_con_id = {quote.con_id: quote for quote in entry_surface}
    exit_by_con_id = {quote.con_id: quote for quote in exit_surface}
    evidence: list[dict[str, object]] = []
    for leg in structure.legs:
        con_id = int(leg.contract.con_id or 0)
        entry_quote = entry_by_con_id.get(con_id)
        exit_quote = exit_by_con_id.get(con_id)
        entry_fill = (
            None
            if entry_quote is None
            else entry_quote.bid
            if leg.side == "short"
            else entry_quote.ask
        )
        exit_fill = (
            None
            if exit_quote is None
            else exit_quote.ask
            if leg.side == "short"
            else exit_quote.bid
        )
        evidence.append(
            {
                "side": leg.side,
                "con_id": leg.contract.con_id,
                "expiry": leg.contract.expiry.isoformat(),
                "dte": leg.contract.dte,
                "dte_bucket": leg.contract.dte_bucket.value,
                "strike": leg.contract.strike,
                "right": leg.contract.right,
                "multiplier": leg.contract.multiplier,
                "target_delta": leg.target_delta,
                "entry_quote_timestamp_utc": (
                    None if entry_quote is None else entry_quote.ordering_timestamp.isoformat()
                ),
                "entry_bid": None if entry_quote is None else entry_quote.bid,
                "entry_ask": None if entry_quote is None else entry_quote.ask,
                "entry_fill_price": entry_fill,
                "exit_quote_timestamp_utc": (
                    None if exit_quote is None else exit_quote.ordering_timestamp.isoformat()
                ),
                "exit_bid": None if exit_quote is None else exit_quote.bid,
                "exit_ask": None if exit_quote is None else exit_quote.ask,
                "exit_fill_price": exit_fill,
            }
        )
    return evidence


class OptionEpisodeFinalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    raw_contract_outcomes: tuple[ShadowOptionOutcome, ...]
    directional_selections: tuple[DirectionalShadowSelection, ...]
    straddles: tuple[dict[str, object], ...]
    oracle_diagnostics: tuple[DirectionalShadowSelection, ...]
    defined_risk_outcomes: tuple[CreditShadowOutcome, ...] = ()
    planned_contract_count: int = 0
    requested_contract_count: int = 0
    option_plan_capacity_reduced: bool = False
    option_plan_missing_buckets: tuple[str, ...] = ()
    required_contract_horizon_count: int = 0
    complete_contract_horizon_count: int = 0
    required_option_quote_windows_finalised: bool = False
    oracle_live_panel_visible: bool = False
    orders_placed: int = 0


class BoundedOptionRecorder:
    """Resolve, stream, persist, and freeze quote outcomes without execution."""

    def __init__(
        self,
        *,
        adapter: IBKRMarketDataAdapter,
        subscriptions: SubscriptionBudgetManager,
        repository: FrozenRecorderRepository,
        raw_store: PartitionedEventStore,
        maximum_quote_age: timedelta,
        recording_duration: timedelta = timedelta(minutes=30),
        stream_registration_sink: Callable[[StreamOwner], None] | None = None,
        stream_unregistration_sink: Callable[[int], None] | None = None,
        request_pacer: Callable[[], object] | None = None,
        underlying_path_provider: (
            Callable[[str, datetime, datetime], tuple[float, ...]] | None
        ) = None,
        underlying_halt_provider: Callable[[str, datetime, datetime], bool] | None = None,
        eviction_sink: Callable[[str, str, datetime], bool] | None = None,
    ) -> None:
        if maximum_quote_age <= timedelta(0):
            raise ValueError("maximum_quote_age must be positive")
        if recording_duration < timedelta(minutes=30):
            raise ValueError("option recording must continue for at least thirty minutes")
        self.adapter = adapter
        self.subscriptions = subscriptions
        self.repository = repository
        self.raw_store = raw_store
        self.maximum_quote_age = maximum_quote_age
        self.recording_duration = recording_duration
        self.stream_registration_sink = stream_registration_sink
        self.stream_unregistration_sink = stream_unregistration_sink
        self.request_pacer = request_pacer
        self.underlying_path_provider = underlying_path_provider
        self.underlying_halt_provider = underlying_halt_provider
        self.eviction_sink = eviction_sink
        self._active: dict[tuple[str, int], _ActiveOption] = {}
        self._planned_contracts_by_episode: dict[str, tuple[OptionContract, ...]] = {}
        self._selection_roles_by_episode: dict[
            str,
            dict[str, tuple[str, ...]],
        ] = {}
        self._requested_contract_count_by_episode: dict[str, int] = {}
        self._plan_capacity_reduced_by_episode: dict[str, bool] = {}
        self._plan_missing_buckets_by_episode: dict[str, tuple[str, ...]] = {}
        self._recording_ends_by_episode: dict[str, datetime] = {}
        self._contracts_by_episode: dict[str, list[OptionContract]] = {}
        self._quotes_by_episode: dict[str, list[OptionQuoteEvent]] = {}
        self._raw_flushed_event_ids: set[str] = set()
        self._gap_episodes: set[str] = set()
        self._quiet_observations: set[str] = set()
        self._quiet_defined_risk_outcomes: dict[
            str,
            list[CreditShadowOutcome],
        ] = {}
        self._live_option_quote_seen = False
        self._option_computation_seen = False

    def start_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        symbol: str,
        entry_timestamp: datetime,
        plan: OptionContractPlan,
        resolver: Callable[[OptionContract], ResolvedOptionContract | None],
        quiet_state: bool = False,
        recording_duration: timedelta | None = None,
    ) -> tuple[OptionContract, ...]:
        """Qualify only the bounded plan and start exact top-of-book streams."""

        if entry_timestamp.tzinfo is None or entry_timestamp.utcoffset() is None:
            raise ValueError("option entry timestamp must be timezone-aware")
        started_at = metadata.recorded_at_utc.astimezone(UTC)
        duration = self.recording_duration if recording_duration is None else recording_duration
        if duration < timedelta(minutes=30):
            raise ValueError("option recording must continue for at least thirty minutes")
        if quiet_state and duration < timedelta(minutes=60):
            raise ValueError("quiet option recording must continue for sixty minutes")
        recording_ends = entry_timestamp.astimezone(UTC) + duration
        self._recording_ends_by_episode[episode_id] = recording_ends
        if quiet_state:
            self.repository.record_quiet_option_plan(
                metadata,
                observation_id=episode_id,
                plan=plan,
            )
            self._quiet_observations.add(episode_id)
        self._planned_contracts_by_episode[episode_id] = plan.contracts
        self._selection_roles_by_episode[episode_id] = dict(plan.selection_roles)
        self._requested_contract_count_by_episode[episode_id] = plan.requested_contract_count
        self._plan_capacity_reduced_by_episode[episode_id] = plan.capacity_reduced
        self._plan_missing_buckets_by_episode[episode_id] = plan.missing_buckets
        selected: list[OptionContract] = []
        for rank, unresolved in enumerate(plan.contracts, start=1):
            resolved = resolver(unresolved)
            if resolved is None or resolved.contract.con_id is None:
                if quiet_state:
                    self.repository.record_quiet_option_contract(
                        metadata,
                        observation_id=episode_id,
                        contract=unresolved,
                        selection_rank=rank,
                        selection_roles=plan.selection_roles.get(
                            unresolved.con_id_key,
                            (),
                        ),
                        resolution_status="contract_not_resolved",
                        rejection_reason="contract_not_resolved",
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                else:
                    self.repository.record_option_contract(
                        metadata,
                        episode_id=episode_id,
                        contract=unresolved,
                        selection_rank=rank,
                        resolution_status="contract_not_resolved",
                        rejection_reason="contract_not_resolved",
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                continue
            contract = resolved.contract
            con_id = contract.con_id
            assert con_id is not None
            subscription_key = canonical_subscription_key(
                SubscriptionKind.OPTION,
                con_id=con_id,
            )
            owner_id = f"episode:{episode_id}"
            selection_roles = plan.selection_roles.get(
                unresolved.con_id_key,
                (),
            )
            if selection_roles and all(role.startswith("neutral_") for role in selection_roles):
                subscription_class = SubscriptionClass.OPTIONAL_RESEARCH
            elif any(role.startswith("primary_") for role in selection_roles):
                subscription_class = SubscriptionClass.ACTIVE_EPISODE
            else:
                subscription_class = SubscriptionClass.EPISODE_ENGINEERING
            decision = self.subscriptions.allocate(
                key=subscription_key,
                kind=SubscriptionKind.OPTION,
                symbol=metadata.run_id + ":" + episode_id,
                con_id=con_id,
                request_id=-1,
                priority=SubscriptionPriority.from_class(subscription_class),
                owner_episode=episode_id,
                owner_id=owner_id,
                subscription_class=subscription_class,
                protected=subscription_class is SubscriptionClass.ACTIVE_EPISODE,
                now_monotonic=time.monotonic(),
                now_utc=started_at,
            )
            eviction_results = (
                ()
                if self.eviction_sink is None
                else tuple(
                    self.eviction_sink(evicted_key, subscription_key, started_at)
                    for evicted_key in decision.evicted_keys
                )
            )
            allocation_denial_reason = (
                decision.reason
                if not decision.accepted
                else (
                    "evicted_subscription_cancellation_failed"
                    if not all(eviction_results)
                    else None
                )
            )
            if allocation_denial_reason is not None:
                if decision.accepted:
                    self.subscriptions.release(
                        subscription_key,
                        owner_id=owner_id,
                        reason=allocation_denial_reason,
                        now_utc=started_at,
                    )
                if quiet_state:
                    self.repository.record_quiet_option_contract(
                        metadata,
                        observation_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        selection_roles=selection_roles,
                        resolution_status="subscription_capacity_denied",
                        rejection_reason=allocation_denial_reason,
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                else:
                    self.repository.record_option_contract(
                        metadata,
                        episode_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        resolution_status="subscription_capacity_denied",
                        rejection_reason=allocation_denial_reason,
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                continue
            shared_active = next(
                (
                    active
                    for active in self._active.values()
                    if active.subscription_key == subscription_key
                    and active.episode_id != episode_id
                ),
                None,
            )
            if shared_active is not None:
                self.subscriptions.release(
                    subscription_key,
                    owner_id=owner_id,
                    reason="canonical_option_stream_owned_by_active_episode",
                    now_utc=started_at,
                )
                if quiet_state:
                    self.repository.record_quiet_option_contract(
                        metadata,
                        observation_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        selection_roles=selection_roles,
                        resolution_status="subscription_capacity_denied",
                        rejection_reason="canonical_stream_already_owned_by_active_episode",
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                else:
                    self.repository.record_option_contract(
                        metadata,
                        episode_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        resolution_status="subscription_capacity_denied",
                        rejection_reason="canonical_stream_already_owned_by_active_episode",
                        recording_started_at_utc=None,
                        recording_ends_at_utc=recording_ends,
                    )
                continue
            request_id: int | None = None
            try:
                request_id = self.adapter.request_market_data(
                    resolved.upstream_contract,
                    subscription_key=subscription_key,
                    generic_ticks="100,101,106",
                )
                if self.request_pacer is not None:
                    self.request_pacer()
                record = self.subscriptions.get(subscription_key)
                assert record is not None
                if not self.subscriptions.mark_active(
                    subscription_key,
                    request_id=request_id,
                ):
                    raise RuntimeError("duplicate option request ID")
                if self.stream_registration_sink is not None:
                    self.stream_registration_sink(
                        StreamOwner(
                            request_id=request_id,
                            kind=StreamKind.OPTION_LEVEL1,
                            symbol=symbol,
                            con_id=con_id,
                            exchange=contract.exchange,
                            episode_id=episode_id,
                            option_contract=contract,
                        )
                    )
                if quiet_state:
                    database_id = self.repository.record_quiet_option_contract(
                        metadata,
                        observation_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        selection_roles=selection_roles,
                        resolution_status="recording",
                        rejection_reason=None,
                        recording_started_at_utc=started_at,
                        recording_ends_at_utc=recording_ends,
                    )
                else:
                    database_id = self.repository.record_option_contract(
                        metadata,
                        episode_id=episode_id,
                        contract=contract,
                        selection_rank=rank,
                        resolution_status="recording",
                        rejection_reason=None,
                        recording_started_at_utc=started_at,
                        recording_ends_at_utc=recording_ends,
                    )
                self.repository.record_subscription(metadata, record)
                self._active[(episode_id, con_id)] = _ActiveOption(
                    database_id=database_id,
                    episode_id=episode_id,
                    symbol=symbol,
                    contract=contract,
                    upstream_contract=resolved.upstream_contract,
                    request_id=request_id,
                    subscription_key=subscription_key,
                    subscription_class=subscription_class,
                    recording_ends_at_utc=recording_ends,
                    quiet_state=quiet_state,
                )
            except Exception:
                if request_id is not None:
                    self.adapter.cancel_market_data(
                        request_id,
                        subscription_key=subscription_key,
                    )
                    if self.stream_unregistration_sink is not None:
                        self.stream_unregistration_sink(request_id)
                self.subscriptions.release(
                    subscription_key,
                    owner_id=owner_id,
                    reason="upstream_subscription_failed",
                    now_utc=started_at,
                )
                failed = self.subscriptions.records.get(subscription_key)
                if failed is not None:
                    self.repository.record_subscription(metadata, failed)
                self._rollback_episode_start(
                    metadata,
                    episode_id=episode_id,
                    reason="episode_start_rolled_back",
                )
                raise
            selected.append(contract)
        self._contracts_by_episode[episode_id] = selected
        self._quotes_by_episode.setdefault(episode_id, [])
        return tuple(selected)

    def _rollback_episode_start(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        reason: str,
    ) -> None:
        """Cancel every stream opened by a start attempt before propagating failure."""

        for active_key, active in tuple(self._active.items()):
            if active.episode_id != episode_id:
                continue
            self.adapter.cancel_market_data(
                active.request_id,
                subscription_key=active.subscription_key,
            )
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self.subscriptions.release(
                active.subscription_key,
                owner_id=f"episode:{episode_id}",
                reason=reason,
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            self._active.pop(active_key)
        self._contracts_by_episode.pop(episode_id, None)
        self._planned_contracts_by_episode.pop(episode_id, None)
        self._selection_roles_by_episode.pop(episode_id, None)
        self._requested_contract_count_by_episode.pop(episode_id, None)
        self._plan_capacity_reduced_by_episode.pop(episode_id, None)
        self._plan_missing_buckets_by_episode.pop(episode_id, None)
        self._recording_ends_by_episode.pop(episode_id, None)
        self._quotes_by_episode.pop(episode_id, None)
        self._quiet_observations.discard(episode_id)

    def cancel_evicted_subscription(
        self,
        metadata: EvidenceMetadata,
        *,
        key: str,
        replacement_key: str,
    ) -> bool:
        """Cancel an optional option stream shed by the shared budget registry."""

        matching = tuple(
            (active_key, active)
            for active_key, active in self._active.items()
            if active.subscription_key == key
        )
        if not matching:
            return False
        for active_key, active in matching:
            self.adapter.cancel_market_data(
                active.request_id,
                subscription_key=active.subscription_key,
            )
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            record = self.subscriptions.records.get(active.subscription_key)
            if record is not None:
                self.repository.record_subscription(metadata, record)
            self.repository.record_skipped_recording(
                metadata,
                session=metadata.recorded_at_utc.date(),
                episode_id=active.episode_id,
                symbol=active.symbol,
                recording_kind="optional_option_stream_evicted",
                reason=f"preempted_by:{replacement_key}",
                requested_payload={
                    "subscription_key": active.subscription_key,
                    "request_id": active.request_id,
                },
            )
            self._active.pop(active_key)
        return True

    def record_quote(
        self,
        metadata: EvidenceMetadata,
        event: OptionQuoteEvent,
    ) -> None:
        active = self._active.get((event.episode_id, event.con_id))
        if active is None:
            raise ValueError("option quote does not belong to an active bounded contract")
        self._quotes_by_episode[event.episode_id].append(event)
        self._live_option_quote_seen = (
            self._live_option_quote_seen or event.market_data_type is MarketDataType.LIVE
        )
        self._option_computation_seen = self._option_computation_seen or any(
            value is not None
            for value in (
                event.option_model_price,
                event.implied_volatility,
                event.delta,
                event.gamma,
                event.theta,
                event.vega,
            )
        )
        flags = quote_quality_flags(
            event,
            reference_timestamp=event.received_timestamp_utc,
            maximum_quote_age=self.maximum_quote_age,
        )
        names = tuple(
            name for name, value in flags.model_dump().items() if isinstance(value, bool) and value
        )
        if active.quiet_state:
            self.repository.update_quiet_option_quote_projection(
                option_contract_id=active.database_id,
                event=event,
                recording_status="recording",
                quote_quality_flags=names,
            )
        else:
            self.repository.update_option_quote_projection(
                option_contract_id=active.database_id,
                event=event,
                recording_status="recording",
                quote_quality_flags=names,
            )

    def flush_raw(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        complete: bool,
        gap_count: int = 0,
    ) -> tuple[str, ...]:
        quotes = tuple(
            event
            for event in self._quotes_by_episode.get(episode_id, ())
            if event.event_id not in self._raw_flushed_event_ids
        )
        if not quotes:
            return ()
        partitions = self.raw_store.write_grouped(
            data_source="ibkr",
            events=quotes,
            complete=complete,
            gap_count=gap_count,
        )
        for partition in partitions:
            first = next(
                event
                for event in quotes
                if (event.received_timestamp_utc == partition.minimum_timestamp_utc)
            )
            self.repository.record_partition(
                metadata,
                data_source="ibkr",
                session_date=first.session,
                symbol=first.symbol,
                event_type="option_quote_event",
                partition=partition,
            )
        self._raw_flushed_event_ids.update(event.event_id for event in quotes)
        return tuple(item.content_hash for item in partitions)

    def mark_data_gap(self) -> None:
        """Mark every currently active option episode as spanning a subscription gap."""

        self._gap_episodes.update(active.episode_id for active in self._active.values())

    def flush_pending(self, metadata: EvidenceMetadata) -> tuple[str, ...]:
        """Atomically seal every unflushed raw option update seen by this poll."""

        hashes: list[str] = []
        for episode_id in sorted(self._quotes_by_episode):
            hashes.extend(
                self.flush_raw(
                    metadata,
                    episode_id=episode_id,
                    complete=True,
                    gap_count=int(episode_id in self._gap_episodes),
                )
            )
        return tuple(hashes)

    def rebuild_after_data_loss(self, metadata: EvidenceMetadata) -> None:
        """Recreate active option streams while preserving an explicit invalidating gap."""

        self.mark_data_gap()
        for active_key, active in tuple(self._active.items()):
            if metadata.recorded_at_utc >= active.recording_ends_at_utc:
                self.subscriptions.release(
                    active.subscription_key,
                    owner_id=f"episode:{active.episode_id}",
                    reason="expired_episode_not_restored",
                    now_utc=metadata.recorded_at_utc,
                )
                self.repository.record_subscription(
                    metadata,
                    self.subscriptions.records[active.subscription_key],
                )
                self._active.pop(active_key)
                continue
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self.subscriptions.cancel(
                active.subscription_key,
                reason="data_lost_reconnect",
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            decision = self.subscriptions.allocate(
                key=active.subscription_key,
                kind=SubscriptionKind.OPTION,
                symbol=metadata.run_id + ":" + active.episode_id,
                con_id=active.contract.con_id or 0,
                request_id=-1,
                priority=SubscriptionPriority.from_class(active.subscription_class),
                owner_episode=active.episode_id,
                owner_id=f"episode:{active.episode_id}",
                subscription_class=active.subscription_class,
                protected=(active.subscription_class is SubscriptionClass.ACTIVE_EPISODE),
                now_monotonic=time.monotonic(),
                now_utc=metadata.recorded_at_utc,
            )
            if not decision.accepted:
                self._active.pop(active_key)
                continue
            try:
                request_id = self.adapter.request_market_data(
                    active.upstream_contract,
                    subscription_key=active.subscription_key,
                    generic_ticks="100,101,106",
                )
                if self.request_pacer is not None:
                    self.request_pacer()
            except Exception:
                self.subscriptions.cancel(
                    active.subscription_key,
                    reason="reconnect_subscription_failed",
                    now_utc=metadata.recorded_at_utc,
                )
                self.repository.record_subscription(
                    metadata,
                    self.subscriptions.records[active.subscription_key],
                )
                self._active.pop(active_key)
                continue
            record = self.subscriptions.get(active.subscription_key)
            assert record is not None
            if not self.subscriptions.mark_active(
                active.subscription_key,
                request_id=request_id,
            ):
                self._active.pop(active_key)
                continue
            active.request_id = request_id
            if self.stream_registration_sink is not None:
                self.stream_registration_sink(
                    StreamOwner(
                        request_id=request_id,
                        kind=StreamKind.OPTION_LEVEL1,
                        symbol=active.symbol,
                        con_id=active.contract.con_id or 0,
                        exchange=active.contract.exchange,
                        episode_id=active.episode_id,
                        option_contract=active.contract,
                    )
                )
            self.repository.record_subscription(metadata, record)

    def cancel_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        reason: str,
        reset_for_restart: bool = False,
    ) -> None:
        """Cancel one displaced or timed-out episode without touching other streams."""

        self.flush_raw(
            metadata,
            episode_id=episode_id,
            complete=False,
            gap_count=1,
        )
        for active_key, active in tuple(self._active.items()):
            if active.episode_id != episode_id:
                continue
            self.adapter.cancel_market_data(
                active.request_id,
                subscription_key=active.subscription_key,
            )
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self.subscriptions.release(
                active.subscription_key,
                owner_id=f"episode:{episode_id}",
                reason=reason,
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            self._active.pop(active_key)
        self._gap_episodes.add(episode_id)
        if reset_for_restart:
            self._planned_contracts_by_episode.pop(episode_id, None)
            self._selection_roles_by_episode.pop(episode_id, None)
            self._requested_contract_count_by_episode.pop(episode_id, None)
            self._plan_capacity_reduced_by_episode.pop(episode_id, None)
            self._plan_missing_buckets_by_episode.pop(episode_id, None)
            self._recording_ends_by_episode.pop(episode_id, None)
            self._contracts_by_episode.pop(episode_id, None)
            self._quotes_by_episode.pop(episode_id, None)
            self._quiet_observations.discard(episode_id)

    def reconcile(
        self,
        metadata: EvidenceMetadata,
        *,
        actual_request_ids: set[int],
    ) -> tuple[str, ...]:
        """Drop option stream projections whose broker request no longer exists."""

        repaired: list[str] = []
        for active_key, active in tuple(self._active.items()):
            record = self.subscriptions.get(active.subscription_key)
            if record is not None and active.request_id in actual_request_ids:
                continue
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self._active.pop(active_key)
            self._gap_episodes.add(active.episode_id)
            repaired.append(active.subscription_key)
            self.repository.record_skipped_recording(
                metadata,
                session=metadata.recorded_at_utc.date(),
                episode_id=active.episode_id,
                symbol=active.symbol,
                recording_kind="orphaned_option_line",
                reason="reconciled_missing_upstream_option_request",
                requested_payload={
                    "subscription_key": active.subscription_key,
                    "request_id": active.request_id,
                },
            )
        return tuple(sorted(repaired))

    @property
    def live_option_quote_seen(self) -> bool:
        return self._live_option_quote_seen

    @property
    def option_computation_seen(self) -> bool:
        return self._option_computation_seen

    @property
    def active_episode_ids(self) -> frozenset[str]:
        return frozenset(active.episode_id for active in self._active.values())

    def shutdown(self, metadata: EvidenceMetadata) -> None:
        self.flush_pending(metadata)
        for active_key, active in tuple(self._active.items()):
            self.adapter.cancel_market_data(
                active.request_id,
                subscription_key=active.subscription_key,
            )
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self.subscriptions.release(
                active.subscription_key,
                owner_id=f"episode:{active.episode_id}",
                reason="recorder_shutdown",
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            self._active.pop(active_key)

    def _surface_at(
        self,
        quotes: tuple[OptionQuoteEvent, ...],
        *,
        target: datetime,
        entry: bool,
    ) -> tuple[OptionQuoteEvent, ...]:
        by_contract: dict[int, list[OptionQuoteEvent]] = {}
        for quote in quotes:
            by_contract.setdefault(quote.con_id, []).append(quote)
        surface: list[OptionQuoteEvent] = []
        for con_id in sorted(by_contract):
            ordered = sorted(
                by_contract[con_id],
                key=lambda item: (
                    item.ordering_timestamp,
                    item.received_monotonic_ns,
                    item.source_sequence,
                    item.event_id,
                ),
            )
            if entry:
                candidate = next(
                    (
                        quote
                        for quote in ordered
                        if target <= quote.ordering_timestamp <= target + self.maximum_quote_age
                    ),
                    None,
                )
            else:
                candidate = next(
                    (
                        quote
                        for quote in reversed(ordered)
                        if target - self.maximum_quote_age <= quote.ordering_timestamp <= target
                    ),
                    None,
                )
            if candidate is not None:
                surface.append(candidate)
        return tuple(surface)

    def _record_quiet_long_outcome(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        structure_type: Literal["LONG_CALL", "LONG_PUT"],
        outcome: ShadowOptionOutcome,
        subscription_gap_spans_horizon: bool,
        horizon_label: str | None = None,
        additional_quality_flags: tuple[str, ...] = (),
    ) -> None:
        flags = set((*outcome.quote_quality_flags, *additional_quality_flags))
        if subscription_gap_spans_horizon:
            flags.add("subscription_gap_spans_horizon")
        complete = outcome.entry_ask is not None and outcome.exit_bid is not None
        strict = complete and not flags
        self.repository.record_quiet_shadow_structure(
            metadata,
            observation_id=observation_id,
            structure_type=structure_type,
            dte_bucket=outcome.dte_bucket.value,
            horizon_label=horizon_label or f"{outcome.horizon_minutes}m",
            horizon_minutes=outcome.horizon_minutes,
            payload=outcome.model_dump(mode="json"),
            opening_credit_or_debit=outcome.premium_at_risk,
            maximum_defined_risk=outcome.premium_at_risk,
            conservative_pnl=outcome.dollar_pnl_per_contract,
            return_on_maximum_risk=outcome.ask_to_bid_return,
            short_strike_touched=None,
            protective_wing_touched=None,
            attempted=True,
            complete_quote_quality=complete,
            strict_quote_quality=strict,
            quality_status=(
                "strict_quality"
                if strict
                else "complete_quote_quality"
                if complete
                else "incomplete"
            ),
            quality_flags=tuple(sorted(flags)),
        )

    def _record_missing_quiet_long_attempt(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        structure_type: Literal["LONG_CALL", "LONG_PUT"],
        bucket: DteBucket,
        horizon_minutes: int,
        subscription_gap_spans_horizon: bool,
        horizon_label: str | None = None,
        additional_quality_flags: tuple[str, ...] = (),
    ) -> None:
        flags = {"missing_leg_quote", *additional_quality_flags}
        if subscription_gap_spans_horizon:
            flags.add("subscription_gap_spans_horizon")
        self.repository.record_quiet_shadow_structure(
            metadata,
            observation_id=observation_id,
            structure_type=structure_type,
            dte_bucket=bucket.value,
            horizon_label=horizon_label or f"{horizon_minutes}m",
            horizon_minutes=horizon_minutes,
            payload={
                "attempted": True,
                "dte_bucket": bucket.value,
                "horizon_minutes": horizon_minutes,
                "reason": "atm_contract_or_quote_outcome_unavailable",
            },
            opening_credit_or_debit=None,
            maximum_defined_risk=None,
            conservative_pnl=None,
            return_on_maximum_risk=None,
            short_strike_touched=None,
            protective_wing_touched=None,
            attempted=True,
            complete_quote_quality=False,
            strict_quote_quality=False,
            quality_status="incomplete",
            quality_flags=tuple(sorted(flags)),
        )

    def _record_quiet_bucket_outcomes(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        symbol: str,
        entry_timestamp: datetime,
        bucket: DteBucket,
        planned_bucket_contracts: tuple[OptionContract, ...],
        bucket_contracts: tuple[OptionContract, ...],
        atm_call: OptionContract | None,
        atm_put: OptionContract | None,
        quotes: tuple[OptionQuoteEvent, ...],
        outcomes: tuple[ShadowOptionOutcome, ...],
        horizon_minutes: tuple[int, ...],
        subscription_gap_spans_horizon: bool,
        plan_capacity_reduced: bool,
        horizon_labels: Mapping[int, str] | None = None,
    ) -> None:
        entry_surface = self._surface_at(quotes, target=entry_timestamp, entry=True)
        underlying_entry_path = (
            ()
            if self.underlying_path_provider is None
            else self.underlying_path_provider(
                symbol,
                entry_timestamp,
                entry_timestamp + self.maximum_quote_age,
            )
        )
        if underlying_entry_path:
            underlying_entry = underlying_entry_path[0]
        else:
            references = [
                quote.underlying_reference_price
                for quote in entry_surface
                if quote.underlying_reference_price is not None
                and math.isfinite(quote.underlying_reference_price)
                and quote.underlying_reference_price > 0.0
            ]
            underlying_entry = references[0] if references else math.nan
        first_quote_by_contract: dict[int, datetime] = {}
        for quote in sorted(
            quotes,
            key=lambda item: (
                item.ordering_timestamp,
                item.received_monotonic_ns,
                item.source_sequence,
                item.event_id,
            ),
        ):
            if quote.ordering_timestamp >= entry_timestamp:
                first_quote_by_contract.setdefault(
                    quote.con_id,
                    quote.ordering_timestamp,
                )
        subscription_started_late = any(
            first_quote_by_contract.get(int(contract.con_id or 0)) is not None
            and (first_quote_by_contract[int(contract.con_id or 0)] - entry_timestamp)
            > self.maximum_quote_age
            for contract in bucket_contracts
        )
        planned_contract_missing = len(bucket_contracts) < len(planned_bucket_contracts)
        structures = (
            select_iron_butterfly(
                contracts=bucket_contracts,
                underlying_entry_price=underlying_entry,
            ),
            select_delta_iron_condor(
                contracts=bucket_contracts,
                entry_quotes=entry_surface,
            ),
            select_fixed_width_credit_spread(
                contracts=bucket_contracts,
                underlying_entry_price=underlying_entry,
                right="C",
            ),
            select_fixed_width_credit_spread(
                contracts=bucket_contracts,
                underlying_entry_price=underlying_entry,
                right="P",
            ),
        )
        defined_risk = self._quiet_defined_risk_outcomes.setdefault(
            observation_id,
            [],
        )
        for minutes in horizon_minutes:
            horizon_label = (
                f"{minutes}m"
                if horizon_labels is None
                else horizon_labels.get(minutes, f"{minutes}m")
            )
            target = entry_timestamp + timedelta(minutes=minutes)
            path = (
                ()
                if self.underlying_path_provider is None
                else self.underlying_path_provider(symbol, entry_timestamp, target)
            )
            underlying_halted = (
                False
                if self.underlying_halt_provider is None
                else self.underlying_halt_provider(symbol, entry_timestamp, target)
            )
            observation_quality_flags = tuple(
                flag
                for flag, present in (
                    ("underlying_path_unavailable", not path),
                    ("underlying_halted", underlying_halted),
                    ("option_plan_capacity_reduced", plan_capacity_reduced),
                )
                if present
            )
            call = (
                None
                if atm_call is None
                else next(
                    (
                        item
                        for item in outcomes
                        if item.con_id == atm_call.con_id and item.horizon_minutes == minutes
                    ),
                    None,
                )
            )
            put = (
                None
                if atm_put is None
                else next(
                    (
                        item
                        for item in outcomes
                        if item.con_id == atm_put.con_id and item.horizon_minutes == minutes
                    ),
                    None,
                )
            )
            if call is not None:
                self._record_quiet_long_outcome(
                    metadata,
                    observation_id=observation_id,
                    structure_type="LONG_CALL",
                    outcome=call,
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                    horizon_label=horizon_label,
                    additional_quality_flags=observation_quality_flags,
                )
            else:
                self._record_missing_quiet_long_attempt(
                    metadata,
                    observation_id=observation_id,
                    structure_type="LONG_CALL",
                    bucket=bucket,
                    horizon_minutes=minutes,
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                    horizon_label=horizon_label,
                    additional_quality_flags=observation_quality_flags,
                )
            if put is not None:
                self._record_quiet_long_outcome(
                    metadata,
                    observation_id=observation_id,
                    structure_type="LONG_PUT",
                    outcome=put,
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                    horizon_label=horizon_label,
                    additional_quality_flags=observation_quality_flags,
                )
            else:
                self._record_missing_quiet_long_attempt(
                    metadata,
                    observation_id=observation_id,
                    structure_type="LONG_PUT",
                    bucket=bucket,
                    horizon_minutes=minutes,
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                    horizon_label=horizon_label,
                    additional_quality_flags=observation_quality_flags,
                )
            if call is not None and put is not None:
                assert atm_call is not None
                straddle = straddle_outcome(call, put)
                entry_value = straddle["entry_call_ask_plus_put_ask"]
                exit_value = straddle["exit_call_bid_plus_put_bid"]
                flags = set(
                    (
                        *call.quote_quality_flags,
                        *put.quote_quality_flags,
                        *observation_quality_flags,
                    )
                )
                if subscription_gap_spans_horizon:
                    flags.add("subscription_gap_spans_horizon")
                complete = entry_value is not None and exit_value is not None
                strict = complete and not flags
                entry_debit = (
                    None if entry_value is None else float(entry_value) * atm_call.multiplier
                )
                pnl = (
                    None
                    if entry_value is None or exit_value is None
                    else (float(exit_value) - float(entry_value)) * atm_call.multiplier
                )
                self.repository.record_quiet_shadow_structure(
                    metadata,
                    observation_id=observation_id,
                    structure_type="ATM_STRADDLE",
                    dte_bucket=bucket.value,
                    horizon_label=horizon_label,
                    horizon_minutes=minutes,
                    payload={
                        "dte_bucket": bucket.value,
                        "horizon_minutes": minutes,
                        **straddle,
                    },
                    opening_credit_or_debit=entry_debit,
                    maximum_defined_risk=entry_debit,
                    conservative_pnl=pnl,
                    return_on_maximum_risk=(
                        None
                        if entry_debit is None or pnl is None or entry_debit <= 0.0
                        else pnl / entry_debit
                    ),
                    short_strike_touched=None,
                    protective_wing_touched=None,
                    attempted=True,
                    complete_quote_quality=complete,
                    strict_quote_quality=strict,
                    quality_status=(
                        "strict_quality"
                        if strict
                        else "complete_quote_quality"
                        if complete
                        else "incomplete"
                    ),
                    quality_flags=tuple(sorted(flags)),
                )
            else:
                flags = {"missing_leg_quote", *observation_quality_flags}
                if subscription_gap_spans_horizon:
                    flags.add("subscription_gap_spans_horizon")
                self.repository.record_quiet_shadow_structure(
                    metadata,
                    observation_id=observation_id,
                    structure_type="ATM_STRADDLE",
                    dte_bucket=bucket.value,
                    horizon_label=horizon_label,
                    horizon_minutes=minutes,
                    payload={
                        "attempted": True,
                        "dte_bucket": bucket.value,
                        "horizon_minutes": minutes,
                        "reason": "atm_call_or_put_outcome_unavailable",
                    },
                    opening_credit_or_debit=None,
                    maximum_defined_risk=None,
                    conservative_pnl=None,
                    return_on_maximum_risk=None,
                    short_strike_touched=None,
                    protective_wing_touched=None,
                    attempted=True,
                    complete_quote_quality=False,
                    strict_quote_quality=False,
                    quality_status="incomplete",
                    quality_flags=tuple(sorted(flags)),
                )
            exit_surface = self._surface_at(quotes, target=target, entry=False)
            mark_surfaces = tuple(
                self._surface_at(quotes, target=mark_timestamp, entry=False)
                for mark_timestamp in sorted(
                    {
                        quote.ordering_timestamp
                        for quote in quotes
                        if entry_timestamp <= quote.ordering_timestamp <= target
                    }
                )
            )
            for structure in structures:
                outcome = calculate_credit_shadow(
                    structure=structure,
                    entry_quotes=entry_surface,
                    exit_quotes=exit_surface,
                    entry_timestamp=entry_timestamp,
                    exit_timestamp=target,
                    underlying_path=path,
                    mark_quote_surfaces=mark_surfaces,
                    additional_quality_flags=tuple(
                        flag
                        for flag, present in (
                            (
                                "subscription_gap_spans_horizon",
                                subscription_gap_spans_horizon,
                            ),
                            (
                                "subscription_started_late",
                                subscription_started_late,
                            ),
                            (
                                "missing_leg_quote",
                                planned_contract_missing,
                            ),
                            (
                                "underlying_path_unavailable",
                                not path,
                            ),
                            (
                                "underlying_halted",
                                underlying_halted,
                            ),
                            (
                                "option_plan_capacity_reduced",
                                plan_capacity_reduced,
                            ),
                        )
                        if present
                    ),
                )
                defined_risk.append(outcome)
                flags = set(outcome.quote_quality_flags)
                if subscription_gap_spans_horizon:
                    flags.add("subscription_gap_spans_horizon")
                payload: dict[str, object] = {
                    **outcome.model_dump(mode="json"),
                    "legs": _quiet_virtual_leg_evidence(
                        structure=structure,
                        entry_surface=entry_surface,
                        exit_surface=exit_surface,
                    ),
                }
                self.repository.record_quiet_shadow_structure(
                    metadata,
                    observation_id=observation_id,
                    structure_type=structure.structure_type.value,
                    dte_bucket=bucket.value,
                    horizon_label=horizon_label,
                    horizon_minutes=minutes,
                    payload=payload,
                    opening_credit_or_debit=outcome.opening_net_credit,
                    maximum_defined_risk=outcome.maximum_defined_risk,
                    conservative_pnl=outcome.commission_free_pnl,
                    return_on_maximum_risk=outcome.return_on_maximum_risk,
                    short_strike_touched=outcome.short_strike_touched,
                    protective_wing_touched=outcome.protective_wing_touched,
                    attempted=outcome.attempted,
                    complete_quote_quality=outcome.complete_quote_quality,
                    strict_quote_quality=(
                        outcome.strict_quote_quality and not subscription_gap_spans_horizon
                    ),
                    quality_status=(
                        "strict_quality"
                        if outcome.strict_quote_quality and not subscription_gap_spans_horizon
                        else outcome.quote_quality_status
                    ),
                    quality_flags=tuple(sorted(flags)),
                )

    def finalise_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        symbol: str,
        entry_timestamp: datetime,
        directional_actions: Mapping[str, str],
        subscription_gap_spans_horizon: bool | None = None,
    ) -> OptionEpisodeFinalization:
        if subscription_gap_spans_horizon is None:
            subscription_gap_spans_horizon = episode_id in self._gap_episodes
        self.flush_raw(
            metadata,
            episode_id=episode_id,
            complete=True,
            gap_count=int(subscription_gap_spans_horizon),
        )
        planned_contracts = self._planned_contracts_by_episode.get(episode_id, ())
        requested_contract_count = self._requested_contract_count_by_episode.get(
            episode_id,
            len(planned_contracts),
        )
        plan_capacity_reduced = self._plan_capacity_reduced_by_episode.get(
            episode_id,
            False,
        )
        plan_missing_buckets = self._plan_missing_buckets_by_episode.get(episode_id, ())
        selection_roles = self._selection_roles_by_episode.get(episode_id, {})
        contracts = tuple(self._contracts_by_episode.get(episode_id, ()))
        quotes = tuple(self._quotes_by_episode.get(episode_id, ()))
        quiet_state = episode_id in self._quiet_observations
        fixed_horizons = (5, 10, 15, 30, 60)
        horizon_labels = {minutes: f"{minutes}m" for minutes in fixed_horizons}
        recording_ends = self._recording_ends_by_episode.get(
            episode_id,
            entry_timestamp + timedelta(minutes=60),
        )
        try:
            _market_open, market_close = xnys_session_bounds(entry_timestamp.date())
        except ValueError:
            market_close = None
        session_end_minutes: int | None = None
        if market_close is not None and entry_timestamp < market_close <= recording_ends:
            session_end_minutes = int((market_close - entry_timestamp).total_seconds() // 60)
            if session_end_minutes > 0 and session_end_minutes not in horizon_labels:
                horizon_labels[session_end_minutes] = "session_end"
        else:
            skip_sink = getattr(self.repository, "record_skipped_recording", None)
            if callable(skip_sink):
                skip_sink(
                    metadata,
                    session=entry_timestamp.date(),
                    episode_id=episode_id,
                    symbol=symbol,
                    recording_kind="session_end_shadow_outcome",
                    reason=(
                        "session_end_outside_bounded_recording_horizon"
                        if market_close is not None
                        else "session_end_calendar_unavailable"
                    ),
                    requested_payload={
                        "entry_timestamp_utc": (entry_timestamp.astimezone(UTC).isoformat()),
                        "recording_ends_at_utc": (recording_ends.astimezone(UTC).isoformat()),
                        "session_end_utc": (
                            None
                            if market_close is None
                            else market_close.astimezone(UTC).isoformat()
                        ),
                    },
                )
        horizon_minutes = tuple(sorted(horizon_labels))
        outcomes = build_shadow_outcomes(
            episode_id=episode_id,
            symbol=symbol,
            entry_timestamp=entry_timestamp,
            contracts=contracts,
            quotes=quotes,
            horizons=tuple(timedelta(minutes=value) for value in horizon_minutes),
            maximum_quote_age=self.maximum_quote_age,
        )
        (
            required_contract_horizon_count,
            complete_contract_horizon_count,
            required_option_quote_windows_finalised,
        ) = _required_quote_matrix_completion(
            planned_contracts=planned_contracts,
            requested_contract_count=requested_contract_count,
            plan_capacity_reduced=plan_capacity_reduced,
            outcomes=outcomes,
            horizon_minutes=horizon_minutes,
            subscription_gap_spans_horizon=subscription_gap_spans_horizon,
        )
        outcome_validity: dict[tuple[int | None, int], bool] = {}
        for outcome in outcomes:
            valid = evaluate_option_outcome_safety(
                OptionOutcomeSafetyInputs(
                    contract_resolved=outcome.con_id is not None,
                    market_data_type=(
                        next(
                            (
                                quote.market_data_type
                                for quote in quotes
                                if quote.con_id == outcome.con_id
                            ),
                            None,
                        )
                        or MarketDataType.DELAYED
                    ),
                    valid_entry_ask=outcome.entry_ask is not None,
                    valid_exit_bid=outcome.exit_bid is not None,
                    quote_freshness_recorded=(
                        outcome.entry_quote_age_seconds is not None
                        and outcome.exit_quote_age_seconds is not None
                    ),
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                )
            ).scientific_recording_valid
            outcome_validity[(outcome.con_id, outcome.horizon_minutes)] = valid
            if not quiet_state or directional_actions:
                self.repository.record_shadow_outcome(
                    metadata,
                    archetype="raw_contract",
                    direction="CALL" if outcome.right == "C" else "PUT",
                    outcome=outcome,
                    valid=valid,
                )
        selections: list[DirectionalShadowSelection] = []
        straddles: list[dict[str, object]] = []
        oracles: list[DirectionalShadowSelection] = []
        bucket_source = (
            (
                DteBucket.ZERO_DTE,
                DteBucket.ONE_DTE,
                DteBucket.THREE_TO_FIVE_DTE,
            )
            if quiet_state
            else tuple(sorted({contract.dte_bucket for contract in contracts}, key=str))
        )
        for bucket in bucket_source:
            planned_bucket_contracts = tuple(
                contract for contract in planned_contracts if contract.dte_bucket is bucket
            )
            bucket_contracts = [contract for contract in contracts if contract.dte_bucket is bucket]
            if quiet_state:
                resolved_bucket_contracts = tuple(bucket_contracts)
                atm_call = _resolved_contract_match(
                    _planned_atm_contract(
                        planned_bucket_contracts,
                        selection_roles=selection_roles,
                        right="C",
                    ),
                    resolved_bucket_contracts,
                )
                atm_put = _resolved_contract_match(
                    _planned_atm_contract(
                        planned_bucket_contracts,
                        selection_roles=selection_roles,
                        right="P",
                    ),
                    resolved_bucket_contracts,
                )
                self._record_quiet_bucket_outcomes(
                    metadata,
                    observation_id=episode_id,
                    symbol=symbol,
                    entry_timestamp=entry_timestamp,
                    bucket=bucket,
                    planned_bucket_contracts=planned_bucket_contracts,
                    bucket_contracts=tuple(bucket_contracts),
                    atm_call=atm_call,
                    atm_put=atm_put,
                    quotes=quotes,
                    outcomes=outcomes,
                    horizon_minutes=horizon_minutes,
                    horizon_labels=horizon_labels,
                    subscription_gap_spans_horizon=subscription_gap_spans_horizon,
                    plan_capacity_reduced=plan_capacity_reduced,
                )
                if session_end_minutes is not None and session_end_minutes in fixed_horizons:
                    self._record_quiet_bucket_outcomes(
                        metadata,
                        observation_id=episode_id,
                        symbol=symbol,
                        entry_timestamp=entry_timestamp,
                        bucket=bucket,
                        planned_bucket_contracts=planned_bucket_contracts,
                        bucket_contracts=tuple(bucket_contracts),
                        atm_call=atm_call,
                        atm_put=atm_put,
                        quotes=quotes,
                        outcomes=outcomes,
                        horizon_minutes=(session_end_minutes,),
                        horizon_labels={session_end_minutes: "session_end"},
                        subscription_gap_spans_horizon=(subscription_gap_spans_horizon),
                        plan_capacity_reduced=plan_capacity_reduced,
                    )
                for minutes in horizon_minutes:
                    call = (
                        None
                        if atm_call is None
                        else next(
                            (
                                item
                                for item in outcomes
                                if item.con_id == atm_call.con_id
                                and item.horizon_minutes == minutes
                            ),
                            None,
                        )
                    )
                    put = (
                        None
                        if atm_put is None
                        else next(
                            (
                                item
                                for item in outcomes
                                if item.con_id == atm_put.con_id and item.horizon_minutes == minutes
                            ),
                            None,
                        )
                    )
                    if call is not None and put is not None and not directional_actions:
                        straddles.append(
                            {
                                "dte_bucket": bucket.value,
                                "horizon_minutes": minutes,
                                **straddle_outcome(call, put),
                            }
                        )
                if not directional_actions:
                    continue
            calls = [item for item in bucket_contracts if item.right == "C"]
            puts = [item for item in bucket_contracts if item.right == "P"]
            if not calls or not puts:
                continue
            atm_call = calls[0]
            atm_put = puts[0]
            archetypes: tuple[Literal["A1", "C1", "R1"], ...] = (
                "A1",
                "C1",
                "R1",
            )
            for archetype in archetypes:
                raw_action = directional_actions.get(archetype, "ABSTAIN")
                if raw_action not in {"CALL", "PUT", "ABSTAIN"}:
                    raise ValueError("directional shadow action is invalid")
                selection = map_directional_shadow(
                    archetype=archetype,
                    action=cast(
                        Literal["CALL", "PUT", "ABSTAIN"],
                        raw_action,
                    ),
                    atm_call=atm_call,
                    atm_put=atm_put,
                )
                selections.append(selection)
                selected_contract = (
                    atm_call if raw_action == "CALL" else atm_put if raw_action == "PUT" else None
                )
                if selected_contract is not None:
                    for outcome in outcomes:
                        if outcome.con_id == selected_contract.con_id:
                            self.repository.record_shadow_outcome(
                                metadata,
                                archetype=archetype,
                                direction=raw_action,
                                outcome=outcome,
                                valid=(
                                    outcome_validity.get(
                                        (outcome.con_id, outcome.horizon_minutes),
                                        False,
                                    )
                                    and not outcome.quote_quality_flags
                                ),
                            )
            for minutes in horizon_minutes:
                call = next(
                    (
                        item
                        for item in outcomes
                        if item.con_id == atm_call.con_id and item.horizon_minutes == minutes
                    ),
                    None,
                )
                put = next(
                    (
                        item
                        for item in outcomes
                        if item.con_id == atm_put.con_id and item.horizon_minutes == minutes
                    ),
                    None,
                )
                if call is not None and put is not None:
                    straddle: dict[str, object] = {
                        "dte_bucket": bucket.value,
                        "horizon_minutes": minutes,
                        "horizon_label": horizon_labels[minutes],
                        **straddle_outcome(call, put),
                    }
                    straddles.append(straddle)
                    self.repository.record_shadow_structure(
                        metadata,
                        episode_id=episode_id,
                        structure_type="ATM_STRADDLE",
                        dte_bucket=bucket.value,
                        horizon_minutes=minutes,
                        payload=straddle,
                        valid=(
                            straddle["entry_call_ask_plus_put_ask"] is not None
                            and straddle["exit_call_bid_plus_put_bid"] is not None
                            and not call.quote_quality_flags
                            and not put.quote_quality_flags
                            and outcome_validity.get(
                                (call.con_id, call.horizon_minutes),
                                False,
                            )
                            and outcome_validity.get(
                                (put.con_id, put.horizon_minutes),
                                False,
                            )
                        ),
                    )
                    oracle = retrospective_oracle(call, put)
                    oracles.append(oracle)
                    self.repository.record_shadow_structure(
                        metadata,
                        episode_id=episode_id,
                        structure_type="RETROSPECTIVE_ORACLE",
                        dte_bucket=bucket.value,
                        horizon_minutes=minutes,
                        payload={
                            **oracle.model_dump(mode="json"),
                            "horizon_label": horizon_labels[minutes],
                            "call_ask_to_bid_return": call.ask_to_bid_return,
                            "put_ask_to_bid_return": put.ask_to_bid_return,
                        },
                        valid=(
                            oracle.selected_contract_key is not None
                            and outcome_validity.get(
                                (call.con_id, call.horizon_minutes),
                                False,
                            )
                            and outcome_validity.get(
                                (put.con_id, put.horizon_minutes),
                                False,
                            )
                        ),
                    )
        for active_key, active in tuple(self._active.items()):
            if active.episode_id != episode_id:
                continue
            self.adapter.cancel_market_data(
                active.request_id,
                subscription_key=active.subscription_key,
            )
            if self.stream_unregistration_sink is not None:
                self.stream_unregistration_sink(active.request_id)
            self.subscriptions.release(
                active.subscription_key,
                owner_id=f"episode:{episode_id}",
                reason="episode_recording_horizon_complete",
                now_utc=metadata.recorded_at_utc,
            )
            record = self.subscriptions.records[active.subscription_key]
            self.repository.record_subscription(metadata, record)
            self._active.pop(active_key)
        self._gap_episodes.discard(episode_id)
        self._quiet_observations.discard(episode_id)
        self._planned_contracts_by_episode.pop(episode_id, None)
        self._selection_roles_by_episode.pop(episode_id, None)
        self._requested_contract_count_by_episode.pop(episode_id, None)
        self._plan_capacity_reduced_by_episode.pop(episode_id, None)
        self._plan_missing_buckets_by_episode.pop(episode_id, None)
        self._contracts_by_episode.pop(episode_id, None)
        self._quotes_by_episode.pop(episode_id, None)
        return OptionEpisodeFinalization(
            episode_id=episode_id,
            raw_contract_outcomes=outcomes,
            directional_selections=tuple(selections),
            straddles=tuple(straddles),
            oracle_diagnostics=tuple(oracles),
            defined_risk_outcomes=tuple(self._quiet_defined_risk_outcomes.pop(episode_id, ())),
            planned_contract_count=len(planned_contracts),
            requested_contract_count=requested_contract_count,
            option_plan_capacity_reduced=plan_capacity_reduced,
            option_plan_missing_buckets=plan_missing_buckets,
            required_contract_horizon_count=required_contract_horizon_count,
            complete_contract_horizon_count=complete_contract_horizon_count,
            required_option_quote_windows_finalised=(required_option_quote_windows_finalised),
        )


__all__ = [
    "BoundedOptionRecorder",
    "OptionEpisodeFinalization",
    "ResolvedOptionContract",
]
