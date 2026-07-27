"""Bounded post-episode option top-of-book recording coordinator."""

from __future__ import annotations

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
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.safety import (
    OptionOutcomeSafetyInputs,
    evaluate_option_outcome_safety,
)
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionKind,
    SubscriptionPriority,
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
    recording_ends_at_utc: datetime


class OptionEpisodeFinalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    raw_contract_outcomes: tuple[ShadowOptionOutcome, ...]
    directional_selections: tuple[DirectionalShadowSelection, ...]
    straddles: tuple[dict[str, object], ...]
    oracle_diagnostics: tuple[DirectionalShadowSelection, ...]
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
        self._active: dict[tuple[str, int], _ActiveOption] = {}
        self._contracts_by_episode: dict[str, list[OptionContract]] = {}
        self._quotes_by_episode: dict[str, list[OptionQuoteEvent]] = {}
        self._raw_flushed_event_ids: set[str] = set()
        self._gap_episodes: set[str] = set()
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
    ) -> tuple[OptionContract, ...]:
        """Qualify only the bounded plan and start exact top-of-book streams."""

        if entry_timestamp.tzinfo is None or entry_timestamp.utcoffset() is None:
            raise ValueError("option entry timestamp must be timezone-aware")
        started_at = metadata.recorded_at_utc.astimezone(UTC)
        recording_ends = entry_timestamp.astimezone(UTC) + self.recording_duration
        selected: list[OptionContract] = []
        for rank, unresolved in enumerate(plan.contracts, start=1):
            resolved = resolver(unresolved)
            if resolved is None or resolved.contract.con_id is None:
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
            subscription_key = f"option:{episode_id}:{contract.con_id_key}"
            decision = self.subscriptions.allocate(
                key=subscription_key,
                kind=SubscriptionKind.OPTION,
                symbol=metadata.run_id + ":" + episode_id,
                con_id=con_id,
                request_id=-1,
                priority=SubscriptionPriority.ACTIVE_OPTION,
                owner_episode=episode_id,
                protected=False,
                now_monotonic=time.monotonic(),
                now_utc=started_at,
            )
            if not decision.accepted:
                self.repository.record_option_contract(
                    metadata,
                    episode_id=episode_id,
                    contract=contract,
                    selection_rank=rank,
                    resolution_status="subscription_capacity_denied",
                    rejection_reason=decision.reason,
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
                record.request_id = request_id
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
                    recording_ends_at_utc=recording_ends,
                )
            except Exception:
                if request_id is not None:
                    self.adapter.cancel_market_data(
                        request_id,
                        subscription_key=subscription_key,
                    )
                    if self.stream_unregistration_sink is not None:
                        self.stream_unregistration_sink(request_id)
                self.subscriptions.cancel(
                    subscription_key,
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
            self.subscriptions.cancel(
                active.subscription_key,
                reason=reason,
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            self._active.pop(active_key)
        self._contracts_by_episode.pop(episode_id, None)
        self._quotes_by_episode.pop(episode_id, None)

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
                priority=SubscriptionPriority.ACTIVE_OPTION,
                owner_episode=active.episode_id,
                protected=False,
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
            record.request_id = request_id
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
            self.subscriptions.cancel(
                active.subscription_key,
                reason="recorder_shutdown",
                now_utc=metadata.recorded_at_utc,
            )
            self.repository.record_subscription(
                metadata,
                self.subscriptions.records[active.subscription_key],
            )
            self._active.pop(active_key)

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
        contracts = tuple(self._contracts_by_episode.get(episode_id, ()))
        quotes = tuple(self._quotes_by_episode.get(episode_id, ()))
        outcomes = build_shadow_outcomes(
            episode_id=episode_id,
            symbol=symbol,
            entry_timestamp=entry_timestamp,
            contracts=contracts,
            quotes=quotes,
            horizons=tuple(timedelta(minutes=value) for value in (5, 10, 15, 30)),
            maximum_quote_age=self.maximum_quote_age,
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
        for bucket in sorted({contract.dte_bucket for contract in contracts}, key=str):
            bucket_contracts = [contract for contract in contracts if contract.dte_bucket is bucket]
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
            for minutes in (5, 10, 15, 30):
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
            self.subscriptions.cancel(
                active.subscription_key,
                reason="episode_recording_horizon_complete",
                now_utc=metadata.recorded_at_utc,
            )
            record = self.subscriptions.records[active.subscription_key]
            self.repository.record_subscription(metadata, record)
            self._active.pop(active_key)
        self._gap_episodes.discard(episode_id)
        return OptionEpisodeFinalization(
            episode_id=episode_id,
            raw_contract_outcomes=outcomes,
            directional_selections=tuple(selections),
            straddles=tuple(straddles),
            oracle_diagnostics=tuple(oracles),
        )


__all__ = [
    "BoundedOptionRecorder",
    "OptionEpisodeFinalization",
    "ResolvedOptionContract",
]
