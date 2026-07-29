"""Bounded IBKR option discovery and post-episode recording lifecycle."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Any, Literal, cast

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionReceiptV1,
    OptionContractCandidateV1,
    OptionTopOfBookV1,
    PrimaryOptionPairSelectionV1,
    build_primary_option_bid_ask_outcome_v1,
    select_primary_option_pair_v1,
)
from stocker_prospective.option_budget import (
    BudgetAwareEpisodeStateMachine,
    DteAllocator,
    EpisodeKind,
    EpisodeState,
    OptionEpisodeTask,
    OptionSubscriptionIntent,
)
from stocker_prospective.option_ledger import (
    OptionContract,
    OptionContractPlan,
    build_contract_plan,
)
from stocker_prospective.option_recorder import (
    BoundedOptionRecorder,
    OptionEpisodeFinalization,
    ResolvedOptionContract,
)
from stocker_prospective.options import DteBucket, select_expiries
from stocker_prospective.recorder_v0 import RecorderCheckpointResult
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    canonical_subscription_key,
)

OptionContractFactory = Callable[
    [str, date, float, Literal["C", "P"], int, str, str],
    Any,
]
MetadataFactory = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]
ReferenceQuoteProvider = Callable[
    [str, datetime],
    UnderlyingLevel1QuoteEvent | None,
]
QUIET_OPTION_STRIKE_STEPS = 4
QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION = 8
QUIET_WING_TARGET_FRACTIONS = (0.01, 0.03, 0.06, 0.10)
MAXIMUM_DISCOVERY_SNAPSHOTS = 20
SHORT_DELTA_TOLERANCE = 0.12
LONG_DELTA_TOLERANCE = 0.08


def merge_snapshot_items(items: tuple[object, ...]) -> dict[str, object]:
    """Merge official field/value callbacks and direct fake-adapter snapshots."""

    merged: dict[str, object] = {}
    for item in items:
        values = (
            item
            if isinstance(item, dict)
            else {
                name: getattr(item, name)
                for name in (
                    "bid",
                    "ask",
                    "delta",
                    "market_data_type",
                )
                if hasattr(item, name)
            }
        )
        callback_field = values.get("field")
        if isinstance(callback_field, str) and "value" in values:
            callback_value = values.get("value")
            if callback_value is not None:
                merged[callback_field] = callback_value
        for name, value in values.items():
            if name in {"field", "value"} or value is None:
                continue
            merged[str(name)] = value
    return merged


def _attribute(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _parse_expiry(value: object) -> date | None:
    raw = str(value)
    if len(raw) < 8:
        return None
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


@dataclass
class _PendingEpisode:
    episode_id: str
    symbol: str
    session: date
    entry_timestamp: datetime
    underlying: QualifiedUnderlying
    directional_actions: dict[str, str]
    checkpoint_id: int = -1
    episode_kind: EpisodeKind = EpisodeKind.HIGH_TAIL
    probability: float = 0.5
    quiet_state: bool = False
    recording_duration: timedelta = timedelta(minutes=30)
    strike_steps: int = 2
    maximum_contracts: int = 30
    started: bool = False
    finalised: bool = False
    plan: OptionContractPlan | None = None


class BoundedOptionDiscoveryService:
    """Discover metadata once, qualify exact contracts, and freeze outcomes."""

    def __init__(
        self,
        *,
        adapter: IBKRMarketDataAdapter,
        option_recorder: BoundedOptionRecorder,
        budget: SubscriptionBudgetManager,
        underlying_contracts: Mapping[str, QualifiedUnderlying],
        contract_factory: OptionContractFactory,
        metadata_factory: MetadataFactory,
        reference_quote_provider: ReferenceQuoteProvider,
        run_id: str | None = None,
        strike_steps: int = 2,
        maximum_contracts_per_episode: int = 30,
        maximum_entry_delay: timedelta = timedelta(minutes=2),
        sensitivity_wait: timedelta = timedelta(seconds=5),
        common_strike_fallback_attempts: int = 3,
        heartbeat: Callable[[], object] | None = None,
        episode_state_machine: BudgetAwareEpisodeStateMachine | None = None,
        dte_allocator: DteAllocator | None = None,
        maximum_continuous_lines: int = 8,
        maximum_discovery_snapshots: int = MAXIMUM_DISCOVERY_SNAPSHOTS,
    ) -> None:
        if strike_steps < 0 or maximum_contracts_per_episode < 0:
            raise ValueError("option discovery bounds must be nonnegative")
        if common_strike_fallback_attempts < 0:
            raise ValueError("common-strike fallback attempts must be nonnegative")
        if maximum_continuous_lines < 4:
            raise ValueError("continuous option line limit must secure four primary legs")
        if maximum_discovery_snapshots < maximum_continuous_lines:
            raise ValueError("discovery snapshot bound cannot be below continuous lines")
        self.adapter = adapter
        self.option_recorder = option_recorder
        self.budget = budget
        self.underlying_contracts = dict(underlying_contracts)
        self.contract_factory = contract_factory
        self.metadata_factory = metadata_factory
        self.run_id = run_id
        self.reference_quote_provider = reference_quote_provider
        self.strike_steps = strike_steps
        self.maximum_contracts_per_episode = maximum_contracts_per_episode
        self.maximum_entry_delay = maximum_entry_delay
        self.sensitivity_wait = sensitivity_wait
        self.common_strike_fallback_attempts = common_strike_fallback_attempts
        self.heartbeat = heartbeat
        self.episode_state_machine = episode_state_machine
        self.dte_allocator = dte_allocator or DteAllocator()
        self.maximum_continuous_lines = maximum_continuous_lines
        self.maximum_discovery_snapshots = maximum_discovery_snapshots
        self._neutral_control_ordinal = 0
        self._pending: dict[str, _PendingEpisode] = {}
        self._finalizations: dict[str, OptionEpisodeFinalization] = {}
        self._rejections: dict[str, str] = {}
        self._resolved_contracts: dict[str, ResolvedOptionContract] = {}
        self._chain_metadata_cache: dict[
            tuple[str, date],
            object,
        ] = {}
        self._opening_reversal_audits: dict[
            str,
            PrimaryOptionPairSelectionV1,
        ] = {}
        self._opening_reversal_receipts: dict[
            str,
            OpeningReversalPredictionReceiptV1,
        ] = {}
        self._opening_reversal_subscription_started_at: dict[
            str,
            datetime,
        ] = {}
        self._opening_reversal_discovery_context: dict[
            str,
            tuple[str, bool, int],
        ] = {}
        if run_id is not None:
            self._restore_schedules()

    def _restore_schedules(self) -> None:
        assert self.run_id is not None
        now = datetime.now(UTC)
        for row in self.option_recorder.repository.restorable_option_episode_schedules(
            run_id=self.run_id
        ):
            episode_id = str(row["episode_id"])
            symbol = str(row["symbol"])
            entry = datetime.fromisoformat(str(row["entry_timestamp_utc"])).astimezone(UTC)
            duration = timedelta(seconds=int(row["recording_duration_seconds"]))
            metadata = self.metadata_factory(now, (entry,))
            if str(row["status"]) == "streaming":
                reason = "process_restart_quote_continuity_lost"
                self.option_recorder.repository.update_option_episode_schedule_status(
                    metadata,
                    episode_id=episode_id,
                    status="rejected",
                    degradation_reason=reason,
                )
                self.option_recorder.repository.record_skipped_recording(
                    metadata,
                    session=date.fromisoformat(str(row["session_date"])),
                    episode_id=episode_id,
                    symbol=symbol,
                    recording_kind="option_episode",
                    reason=reason,
                    requested_payload={"restored_status": "streaming"},
                )
                continue
            if entry + duration + self.sensitivity_wait <= now:
                self.option_recorder.repository.update_option_episode_schedule_status(
                    metadata,
                    episode_id=episode_id,
                    status="expired",
                    degradation_reason="expired_before_process_restore",
                )
                continue
            underlying = self.underlying_contracts.get(symbol)
            if underlying is None:
                self.option_recorder.repository.update_option_episode_schedule_status(
                    metadata,
                    episode_id=episode_id,
                    status="rejected",
                    degradation_reason="underlying_contract_not_resolved_after_restart",
                )
                continue
            self._pending[episode_id] = _PendingEpisode(
                episode_id=episode_id,
                checkpoint_id=int(row["checkpoint_id"]),
                symbol=symbol,
                session=date.fromisoformat(str(row["session_date"])),
                entry_timestamp=entry,
                underlying=underlying,
                directional_actions=cast(
                    dict[str, str],
                    json.loads(str(row["directional_actions_json"])),
                ),
                episode_kind=EpisodeKind(str(row["episode_kind"])),
                probability=float(row["probability"]),
                quiet_state=bool(row["quiet_state"]),
                recording_duration=duration,
                strike_steps=int(row["strike_steps"]),
                maximum_contracts=int(row["maximum_contracts"]),
            )

    def persist_checkpoint_schedules(
        self,
        result: RecorderCheckpointResult,
        *,
        episode_ids: Iterable[str] | None = None,
    ) -> None:
        """Durably admit selected in-memory tasks before checkpoint completion."""

        identities = (
            {
                value
                for value in (
                    result.episode_decision.episode_id,
                    result.quiet_observation_id,
                    result.neutral_control_id,
                    result.high_tail_control_id,
                )
                if value is not None
            }
            if episode_ids is None
            else set(episode_ids)
        )
        metadata = self.metadata_factory(
            result.episode_decision.trigger_bar_end,
            (result.episode_decision.trigger_bar_end,),
        )
        for episode_id in sorted(identities):
            episode = self._pending.get(episode_id)
            if episode is None:
                if episode_id not in self._rejections:
                    raise RuntimeError("option episode was neither scheduled nor rejected")
                continue
            self.option_recorder.repository.record_option_episode_schedule(
                metadata,
                episode_id=episode.episode_id,
                checkpoint_id=episode.checkpoint_id,
                symbol=episode.symbol,
                session=episode.session,
                entry_timestamp=episode.entry_timestamp,
                episode_kind=episode.episode_kind.value,
                probability=episode.probability,
                quiet_state=episode.quiet_state,
                directional_actions=episode.directional_actions,
                recording_duration=episode.recording_duration,
                strike_steps=episode.strike_steps,
                maximum_contracts=episode.maximum_contracts,
            )

    def schedule(self, result: RecorderCheckpointResult) -> None:
        decision = result.episode_decision
        if not decision.fresh_episode or decision.episode_id is None:
            return
        underlying = self.underlying_contracts.get(decision.symbol)
        if underlying is None:
            self._rejections[decision.episode_id] = "underlying_contract_not_resolved"
            self._record_unscheduled_rejection(
                result,
                episode_id=decision.episode_id,
                episode_kind=EpisodeKind.HIGH_TAIL,
                quiet_state=False,
                recording_duration=timedelta(minutes=60),
                strike_steps=self.strike_steps,
                maximum_contracts=self.maximum_contracts_per_episode,
                reason="underlying_contract_not_resolved",
            )
            return
        actions: dict[str, str] = {
            model_id: classification.action
            for model_id, classification in result.directional_classifications.items()
        }
        self._pending.setdefault(
            decision.episode_id,
            _PendingEpisode(
                episode_id=decision.episode_id,
                checkpoint_id=result.checkpoint_id,
                symbol=decision.symbol,
                session=decision.session,
                entry_timestamp=decision.prospective_entry_timestamp,
                underlying=underlying,
                directional_actions=actions,
                episode_kind=EpisodeKind.HIGH_TAIL,
                probability=result.score.probability,
                quiet_state=False,
                recording_duration=timedelta(minutes=60),
                strike_steps=self.strike_steps,
                maximum_contracts=self.maximum_contracts_per_episode,
            ),
        )

    def schedule_opening_reversal(
        self,
        result: RecorderCheckpointResult,
        receipt: OpeningReversalPredictionReceiptV1,
    ) -> None:
        """Schedule only the promoted V1 stock and its primary 1DTE pair."""

        decision = result.episode_decision
        if (
            not receipt.eligibility_v1
            or receipt.prediction_v1 == "ABSTAIN"
            or not decision.fresh_episode
            or decision.episode_id is None
            or decision.episode_id != receipt.fresh_episode_id
        ):
            return
        self._opening_reversal_receipts[decision.episode_id] = receipt
        underlying = self.underlying_contracts.get(decision.symbol)
        if underlying is None:
            self._rejections[decision.episode_id] = "underlying_contract_not_resolved"
            self._record_unscheduled_rejection(
                result,
                episode_id=decision.episode_id,
                episode_kind=EpisodeKind.OPENING_REVERSAL,
                quiet_state=False,
                recording_duration=timedelta(minutes=30),
                strike_steps=self.strike_steps,
                maximum_contracts=self.maximum_contracts_per_episode,
                reason="underlying_contract_not_resolved",
            )
            return
        self._pending.setdefault(
            decision.episode_id,
            _PendingEpisode(
                episode_id=decision.episode_id,
                checkpoint_id=result.checkpoint_id,
                symbol=decision.symbol,
                session=decision.session,
                entry_timestamp=decision.prospective_entry_timestamp,
                underlying=underlying,
                directional_actions={
                    (
                        "OPENING_REVERSAL_V1"
                        if receipt.experiment_version == "1"
                        else "OPENING_REVERSAL_V1_1"
                    ): receipt.prediction_v1
                },
                episode_kind=EpisodeKind.OPENING_REVERSAL,
                probability=result.score.probability,
                quiet_state=False,
                recording_duration=timedelta(minutes=30),
                strike_steps=self.strike_steps,
                maximum_contracts=self.maximum_contracts_per_episode,
            ),
        )

    def schedule_quiet_state(self, result: RecorderCheckpointResult) -> None:
        """Schedule the same bounded 60-minute panel for all comparison cohorts."""

        decision = result.quiet_episode_decision
        observations = tuple(
            (value, kind)
            for value, kind in (
                (result.quiet_observation_id, EpisodeKind.QUIET),
                (result.neutral_control_id, EpisodeKind.NEUTRAL_CONTROL),
                (result.high_tail_control_id, EpisodeKind.HIGH_TAIL),
            )
            if value is not None
        )
        underlying = self.underlying_contracts.get(decision.symbol)
        for observation_id, episode_kind in observations:
            if underlying is None:
                self._rejections[observation_id] = "underlying_contract_not_resolved"
                self._record_unscheduled_rejection(
                    result,
                    episode_id=observation_id,
                    episode_kind=episode_kind,
                    quiet_state=True,
                    recording_duration=timedelta(minutes=60),
                    strike_steps=QUIET_OPTION_STRIKE_STEPS,
                    maximum_contracts=QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION,
                    reason="underlying_contract_not_resolved",
                )
                continue
            existing = self._pending.get(observation_id)
            if existing is not None:
                if existing.started:
                    raise RuntimeError("quiet comparison upgrade occurred after option start")
                existing.quiet_state = True
                existing.episode_kind = episode_kind
                existing.probability = result.score.probability
                existing.recording_duration = timedelta(minutes=60)
                existing.strike_steps = QUIET_OPTION_STRIKE_STEPS
                existing.maximum_contracts = QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION
                continue
            self._pending[observation_id] = _PendingEpisode(
                episode_id=observation_id,
                checkpoint_id=result.checkpoint_id,
                symbol=decision.symbol,
                session=decision.session,
                entry_timestamp=decision.prospective_entry_timestamp,
                underlying=underlying,
                directional_actions={},
                episode_kind=episode_kind,
                probability=result.score.probability,
                quiet_state=True,
                recording_duration=timedelta(minutes=60),
                strike_steps=QUIET_OPTION_STRIKE_STEPS,
                maximum_contracts=QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION,
            )

    def poll(self, *, now: datetime) -> None:
        for episode in tuple(self._pending.values()):
            if episode.finalised:
                continue
            if not episode.started and now >= episode.entry_timestamp:
                if episode.plan is None:
                    quote = self.reference_quote_provider(
                        episode.symbol,
                        episode.entry_timestamp,
                    )
                    if quote is None:
                        if now - episode.entry_timestamp > self.maximum_entry_delay:
                            self._reject(
                                episode,
                                now=now,
                                reason="underlying_entry_reference_quote_unavailable",
                            )
                        continue
                    reference = (
                        (quote.bid + quote.ask) / 2.0
                        if quote.bid is not None and quote.ask is not None
                        else quote.last
                    )
                    if reference is None or reference <= 0.0:
                        continue
                    try:
                        episode.plan = self._discover_plan(
                            episode,
                            reference,
                            discovered_at=now,
                        )
                    except (RuntimeError, ValueError) as exc:
                        self._reject(episode, now=now, reason=str(exc))
                        continue
                metadata = self.metadata_factory(now, (episode.entry_timestamp,))
                plan = episode.plan
                assert plan is not None
                if self.episode_state_machine is not None:
                    try:
                        allocation = self.episode_state_machine.submit(
                            self._episode_task(episode, plan),
                            now=now,
                        )
                    except (RuntimeError, ValueError) as exc:
                        self._reject(episode, now=now, reason=str(exc))
                        continue
                    if allocation.state is EpisodeState.EPISODE_QUEUED:
                        continue
                    if allocation.state in {EpisodeState.FAILED, EpisodeState.COMPLETE}:
                        self._reject(
                            episode,
                            now=now,
                            reason=allocation.degradation_reason or "option_episode_not_streamable",
                        )
                        continue
                    plan = self._approved_plan(plan, allocation.approved_subscriptions)
                try:
                    selected = self.option_recorder.start_episode(
                        metadata,
                        episode_id=episode.episode_id,
                        symbol=episode.symbol,
                        entry_timestamp=episode.entry_timestamp,
                        plan=plan,
                        resolver=partial(self._resolve, episode.symbol),
                        quiet_state=episode.quiet_state,
                        recording_duration=episode.recording_duration,
                    )
                except (RuntimeError, ValueError) as exc:
                    self._reject(episode, now=now, reason=str(exc))
                    continue
                if not selected:
                    self._reject(
                        episode,
                        now=now,
                        reason="option_episode_no_contract_stream_started",
                    )
                    continue
                episode.started = True
                self.option_recorder.repository.update_option_episode_schedule_status(
                    metadata,
                    episode_id=episode.episode_id,
                    status="streaming",
                )
                if episode.episode_kind is EpisodeKind.OPENING_REVERSAL:
                    self._opening_reversal_subscription_started_at[episode.episode_id] = now
            finish_at = episode.entry_timestamp + episode.recording_duration + self.sensitivity_wait
            if episode.started and now >= finish_at:
                metadata = self.metadata_factory(now, (finish_at,))
                finalization = self.option_recorder.finalise_episode(
                    metadata,
                    episode_id=episode.episode_id,
                    symbol=episode.symbol,
                    entry_timestamp=episode.entry_timestamp,
                    directional_actions=episode.directional_actions,
                )
                self._finalizations[episode.episode_id] = finalization
                if episode.episode_kind is EpisodeKind.OPENING_REVERSAL:
                    self._record_opening_reversal_option_outcomes(
                        episode=episode,
                        finalization=finalization,
                        metadata=metadata,
                        finalised_at=now,
                    )
                if self.episode_state_machine is not None:
                    self.episode_state_machine.complete(
                        episode.episode_id,
                        now=now,
                    )
                episode.finalised = True
                self.option_recorder.repository.update_option_episode_schedule_status(
                    metadata,
                    episode_id=episode.episode_id,
                    status="complete",
                )
        if self.episode_state_machine is not None:
            self.episode_state_machine.poll(now=now)
        self.option_recorder.flush_pending(self.metadata_factory(now, (now,)))

    def _record_opening_reversal_option_outcomes(
        self,
        *,
        episode: _PendingEpisode,
        finalization: OptionEpisodeFinalization,
        metadata: EvidenceMetadata,
        finalised_at: datetime,
    ) -> None:
        receipt = self._opening_reversal_receipts.get(episode.episode_id)
        audit = self._opening_reversal_audits.get(episode.episode_id)
        if receipt is None or audit is None or not receipt.scientific_outcome_eligible_v1:
            return
        by_con_id = {
            int(outcome.con_id): outcome
            for outcome in finalization.raw_contract_outcomes
            if outcome.horizon_minutes == 15 and outcome.con_id is not None
        }
        predicted_right = "C" if receipt.prediction_v1 == "CALL" else "P"
        started_at = self._opening_reversal_subscription_started_at.get(
            episode.episode_id,
            episode.entry_timestamp,
        )
        for contract in (audit.call, audit.put):
            raw = by_con_id.get(contract.con_id)
            entry_age = None if raw is None else raw.entry_quote_age_seconds
            entry_timestamp = (
                episode.entry_timestamp
                if entry_age is None
                else episode.entry_timestamp + timedelta(seconds=entry_age)
            )
            entry_missing = (
                "entry_ask_invalid"
                if raw is None or raw.entry_ask is None
                else (
                    "entry_quote_quality_incomplete"
                    if raw.entry_bid is None or raw.entry_quality is None
                    else None
                )
            )
            entry_quality = None if raw is None else raw.entry_quality
            entry_quote = OptionTopOfBookV1(
                timestamp_utc=entry_timestamp,
                bid=None if raw is None else raw.entry_bid,
                ask=None if raw is None else raw.entry_ask,
                quote_age_seconds=entry_age,
                locked_or_crossed=(
                    False
                    if entry_quality is None
                    else entry_quality.locked_quote or entry_quality.crossed_quote
                ),
                stale=False if entry_quality is None else entry_quality.stale_quote,
                missing_reason=entry_missing,
            )
            horizon = episode.entry_timestamp + timedelta(minutes=15)
            exit_timestamp = (
                horizon
                if raw is None or raw.first_bid_after_horizon_timestamp is None
                else raw.first_bid_after_horizon_timestamp
            )
            exit_bid = None if raw is None else raw.first_bid_after_horizon
            exit_ask = None if raw is None else raw.first_ask_after_horizon
            exit_age = (
                None
                if raw is None or raw.first_bid_after_horizon_timestamp is None
                else (raw.first_bid_after_horizon_timestamp - horizon).total_seconds()
            )
            exit_quality_missing = exit_bid is None or exit_ask is None or exit_age is None
            exit_quote = OptionTopOfBookV1(
                timestamp_utc=exit_timestamp,
                bid=exit_bid,
                ask=exit_ask,
                quote_age_seconds=exit_age,
                locked_or_crossed=(
                    exit_bid is not None and exit_ask is not None and exit_bid >= exit_ask
                ),
                stale=(
                    exit_age is None
                    or exit_age > self.option_recorder.maximum_quote_age.total_seconds()
                ),
                missing_reason=(
                    "exit_bid_invalid"
                    if exit_bid is None
                    else ("exit_quote_quality_incomplete" if exit_quality_missing else None)
                ),
            )
            result = build_primary_option_bid_ask_outcome_v1(
                prediction_receipt_hash_v1=receipt.receipt_hash_v1,
                contract=contract,
                role=("predicted_leg" if contract.right == predicted_right else "opposite_leg"),
                entry_timestamp_utc=episode.entry_timestamp,
                subscription_start_utc=started_at,
                subscription_end_utc=finalised_at,
                capacity_line_owner=(f"opening_reversal_primary_pair:{episode.episode_id}"),
                entry_quote=entry_quote,
                exit_quote=exit_quote,
            )
            self.option_recorder.repository.record_opening_reversal_primary_option_outcome_v1(
                metadata,
                result,
            )

    def _reject(
        self,
        episode: _PendingEpisode,
        *,
        now: datetime,
        reason: str,
    ) -> None:
        self._rejections[episode.episode_id] = reason
        metadata = self.metadata_factory(now, (episode.entry_timestamp,))
        self.option_recorder.repository.update_option_episode_schedule_status(
            metadata,
            episode_id=episode.episode_id,
            status="rejected",
            degradation_reason=reason,
        )
        self.option_recorder.repository.record_skipped_recording(
            metadata,
            session=episode.session,
            episode_id=episode.episode_id,
            symbol=episode.symbol,
            recording_kind="option_episode",
            reason=reason,
            requested_payload={
                "episode_kind": episode.episode_kind.value,
                "entry_timestamp": episode.entry_timestamp.isoformat(),
                "requested_contract_count": (
                    0 if episode.plan is None else episode.plan.requested_contract_count
                ),
            },
        )
        discovery_context = self._opening_reversal_discovery_context.get(episode.episode_id)
        if (
            episode.episode_kind is EpisodeKind.OPENING_REVERSAL
            and episode.episode_id not in self._opening_reversal_audits
            and discovery_context is not None
        ):
            source, cache_hit, candidates_inspected = discovery_context
            self.option_recorder.repository.record_opening_reversal_contract_discovery_failure_v1(
                metadata,
                episode_id=episode.episode_id,
                discovery_timestamp_utc=now,
                contract_source=source,
                cache_hit=cache_hit,
                candidates_inspected=candidates_inspected,
                missing_reason=reason,
            )
        episode.finalised = True

    def _record_unscheduled_rejection(
        self,
        result: RecorderCheckpointResult,
        *,
        episode_id: str,
        episode_kind: EpisodeKind,
        quiet_state: bool,
        recording_duration: timedelta,
        strike_steps: int,
        maximum_contracts: int,
        reason: str,
    ) -> None:
        metadata = self.metadata_factory(
            result.episode_decision.trigger_bar_end,
            (result.episode_decision.trigger_bar_end,),
        )
        self.option_recorder.repository.record_option_episode_schedule(
            metadata,
            episode_id=episode_id,
            checkpoint_id=result.checkpoint_id,
            symbol=result.episode_decision.symbol,
            session=result.episode_decision.session,
            entry_timestamp=result.episode_decision.prospective_entry_timestamp,
            episode_kind=episode_kind.value,
            probability=result.score.probability,
            quiet_state=quiet_state,
            directional_actions={},
            recording_duration=recording_duration,
            strike_steps=strike_steps,
            maximum_contracts=maximum_contracts,
        )
        self.option_recorder.repository.update_option_episode_schedule_status(
            metadata,
            episode_id=episode_id,
            status="rejected",
            degradation_reason=reason,
        )
        self.option_recorder.repository.record_skipped_recording(
            metadata,
            session=result.episode_decision.session,
            episode_id=episode_id,
            symbol=result.episode_decision.symbol,
            recording_kind="option_episode",
            reason=reason,
            requested_payload={"durable_rejection": True},
        )

    def _episode_task(
        self,
        episode: _PendingEpisode,
        plan: OptionContractPlan,
    ) -> OptionEpisodeTask:
        intents: list[OptionSubscriptionIntent] = []
        for contract in plan.contracts:
            resolved = self._resolve(episode.symbol, contract)
            if resolved is None or resolved.contract.con_id is None:
                continue
            con_id = resolved.contract.con_id
            roles = plan.selection_roles.get(contract.con_id_key, ("comparison",))
            required = episode.episode_kind in {
                EpisodeKind.HIGH_TAIL,
                EpisodeKind.OPENING_REVERSAL,
            } or any(role.startswith("primary_") for role in roles)
            if episode.episode_kind is EpisodeKind.NEUTRAL_CONTROL:
                subscription_class = SubscriptionClass.OPTIONAL_RESEARCH
                required = False
            elif required:
                subscription_class = SubscriptionClass.ACTIVE_EPISODE
            else:
                subscription_class = SubscriptionClass.EPISODE_ENGINEERING
            intents.append(
                OptionSubscriptionIntent(
                    key=canonical_subscription_key(
                        SubscriptionKind.OPTION,
                        con_id=con_id,
                    ),
                    con_id=con_id,
                    role="+".join(roles),
                    subscription_class=subscription_class,
                    required=required,
                    dte_bucket=contract.dte_bucket,
                    drop_order=(
                        None
                        if required
                        else 1
                        if episode.episode_kind is EpisodeKind.NEUTRAL_CONTROL
                        else 3
                        if contract.dte_bucket is DteBucket.THREE_TO_FIVE_DTE
                        else 4
                        if contract.dte_bucket is DteBucket.ZERO_DTE
                        else 2
                    ),
                )
            )
        if (
            episode.episode_kind is EpisodeKind.QUIET
            and sum(intent.required for intent in intents) != 4
        ):
            raise ValueError("primary_1dte_iron_condor_not_resolved")
        if episode.episode_kind is EpisodeKind.HIGH_TAIL and len(intents) < 2:
            raise ValueError("high_tail_atm_call_put_not_resolved")
        if episode.episode_kind is EpisodeKind.OPENING_REVERSAL and len(intents) != 2:
            raise ValueError("opening_reversal_primary_1dte_pair_not_resolved")
        return OptionEpisodeTask(
            episode_id=episode.episode_id,
            symbol=episode.symbol,
            kind=episode.episode_kind,
            probability=episode.probability,
            triggered_at_utc=episode.entry_timestamp,
            useful_until_utc=(
                episode.entry_timestamp + episode.recording_duration + self.sensitivity_wait
            ),
            requested_subscriptions=tuple(intents),
        )

    def _approved_plan(
        self,
        plan: OptionContractPlan,
        approved_keys: tuple[str, ...],
    ) -> OptionContractPlan:
        approved = set(approved_keys)
        selected = tuple(
            contract
            for contract in plan.contracts
            if (
                (resolved := self._resolved_contracts.get(contract.con_id_key)) is not None
                and resolved.contract.con_id is not None
                and canonical_subscription_key(
                    SubscriptionKind.OPTION,
                    con_id=resolved.contract.con_id,
                )
                in approved
            )
        )
        return OptionContractPlan(
            contracts=selected,
            requested_contract_count=plan.requested_contract_count,
            maximum_contracts=plan.maximum_contracts,
            capacity_reduced=len(selected) < plan.requested_contract_count,
            missing_buckets=plan.missing_buckets,
            selection_rule=plan.selection_rule,
            selection_roles={
                contract.con_id_key: plan.selection_roles.get(contract.con_id_key, ())
                for contract in selected
            },
        )

    def mark_data_gap(self) -> None:
        self.option_recorder.mark_data_gap()

    def handle_displacement(
        self,
        episode_id: str,
        replacement_episode_id: str,
        observed_at: datetime,
    ) -> None:
        """Cancel physical streams before a frozen higher-priority replacement."""

        episode = self._pending.get(episode_id)
        if episode is None:
            return
        metadata = self.metadata_factory(observed_at, (observed_at,))
        self.option_recorder.cancel_episode(
            metadata,
            episode_id=episode_id,
            reason=f"displaced_by:{replacement_episode_id}",
            reset_for_restart=True,
        )
        episode.started = False

    def rebuild_after_data_loss(self, metadata: EvidenceMetadata) -> None:
        self.option_recorder.rebuild_after_data_loss(metadata)

    def reconcile(
        self,
        metadata: EvidenceMetadata,
        *,
        actual_request_ids: set[int],
    ) -> tuple[str, ...]:
        return self.option_recorder.reconcile(
            metadata,
            actual_request_ids=actual_request_ids,
        )

    def shutdown(self, metadata: EvidenceMetadata) -> None:
        self.option_recorder.shutdown(metadata)

    def _discover_plan(
        self,
        episode: _PendingEpisode,
        underlying_reference: float,
        *,
        discovered_at: datetime,
    ) -> OptionContractPlan:
        cache_key = (episode.symbol, episode.session)
        if episode.episode_kind is EpisodeKind.OPENING_REVERSAL:
            self._opening_reversal_discovery_context.setdefault(
                episode.episode_id,
                ("ibkr_secdef_metadata", False, 0),
            )
        selected = self._chain_metadata_cache.get(cache_key)
        cache_hit = selected is not None
        if selected is None:
            result = self.adapter.request_option_chain_metadata(
                underlying_symbol=episode.symbol,
                exchange="",
                underlying_security_type="STK",
                underlying_contract_id=episode.underlying.con_id,
            )
            if self.heartbeat is not None:
                self.heartbeat()
            candidates = [
                item
                for item in result.items
                if int(
                    _attribute(
                        item,
                        "underlying_contract_id",
                        "underlyingConId",
                    )
                    or 0
                )
                == episode.underlying.con_id
            ]
            if not candidates:
                raise ValueError("option_chain_metadata_unavailable")
            selected = min(
                candidates,
                key=lambda item: (
                    0 if str(_attribute(item, "exchange") or "") == "SMART" else 1,
                    str(_attribute(item, "exchange") or ""),
                    str(_attribute(item, "trading_class", "tradingClass") or ""),
                ),
            )
            self._chain_metadata_cache[cache_key] = selected
        if episode.episode_kind is EpisodeKind.OPENING_REVERSAL:
            self._opening_reversal_discovery_context[episode.episode_id] = (
                ("cached_ibkr_secdef_metadata" if cache_hit else "ibkr_secdef_metadata"),
                cache_hit,
                0,
            )
        expiries = tuple(
            parsed
            for value in cast(list[object], _attribute(selected, "expirations") or [])
            for parsed in (_parse_expiry(value),)
            if parsed is not None
        )
        selections = select_expiries(episode.session, expiries)
        available_buckets = tuple(
            bucket for bucket, selection in selections.items() if selection.expiry is not None
        )
        neutral_ordinal: int | None = None
        if episode.episode_kind is EpisodeKind.NEUTRAL_CONTROL:
            neutral_ordinal = self._neutral_control_ordinal
            self._neutral_control_ordinal += 1
        allocation = self.dte_allocator.allocate(
            episode_id=episode.episode_id,
            kind=episode.episode_kind,
            available=available_buckets,
            allow_secondary=(
                episode.episode_kind is EpisodeKind.QUIET and self.maximum_continuous_lines >= 8
            ),
            neutral_control_ordinal=neutral_ordinal,
        )
        if allocation.primary is None:
            reason = (
                "no_1dte_expiry"
                if episode.episode_kind
                in {
                    EpisodeKind.QUIET,
                    EpisodeKind.OPENING_REVERSAL,
                }
                else "scheduled_dte_expiry_unavailable"
            )
            raise ValueError(reason)
        allocated_buckets = (allocation.primary, *allocation.secondary)
        chosen_expiries = {
            bucket: (selections[bucket].expiry if bucket in allocated_buckets else None)
            for bucket in DteBucket
        }
        strikes = tuple(
            float(cast(Any, value))
            for value in cast(list[object], _attribute(selected, "strikes") or [])
            if float(cast(Any, value)) > 0.0
        )
        exchange = str(_attribute(selected, "exchange") or "SMART")
        trading_class = str(_attribute(selected, "trading_class", "tradingClass") or episode.symbol)
        if episode.episode_kind is EpisodeKind.OPENING_REVERSAL:
            primary_expiry = selections[DteBucket.ONE_DTE].expiry
            if (
                allocation.primary is not DteBucket.ONE_DTE
                or primary_expiry is None
                or allocation.secondary
            ):
                raise ValueError("opening_reversal_requires_only_primary_1dte")
            return self._discover_opening_reversal_plan(
                episode=episode,
                underlying_reference=underlying_reference,
                expiry=primary_expiry,
                metadata_strikes=strikes,
                exchange=exchange,
                trading_class=trading_class,
                discovered_at=discovered_at,
                cache_hit=cache_hit,
                skipped=allocation.skipped,
            )
        strikes_by_expiry_right: dict[tuple[date, str], tuple[float, ...]] = {}
        if self.maximum_discovery_snapshots > 0:
            for expiry in dict.fromkeys(chosen_expiries.values()):
                if expiry is None:
                    continue
                valid_common = self._qualify_valid_common_strikes(
                    episode=episode,
                    expiry=expiry,
                    candidate_strikes=strikes,
                    underlying_reference=underlying_reference,
                    exchange=exchange,
                    trading_class=trading_class,
                )
                strikes_by_expiry_right[(expiry, "C")] = valid_common
                strikes_by_expiry_right[(expiry, "P")] = valid_common
        candidate_plan = build_contract_plan(
            underlying_con_id=episode.underlying.con_id,
            session_date=episode.session,
            underlying_reference=underlying_reference,
            expiries=chosen_expiries,
            strikes_by_expiry_right=strikes_by_expiry_right,
            strike_steps=episode.strike_steps,
            maximum_contracts=10_000,
            exchange=exchange,
            trading_class=trading_class,
        )
        plan = self._select_snapshot_plan(
            episode=episode,
            candidates=candidate_plan,
            underlying_reference=underlying_reference,
            primary_bucket=allocation.primary,
            secondary_buckets=allocation.secondary,
            skipped=allocation.skipped,
        )
        return plan

    def _discover_opening_reversal_plan(
        self,
        *,
        episode: _PendingEpisode,
        underlying_reference: float,
        expiry: date,
        metadata_strikes: tuple[float, ...],
        exchange: str,
        trading_class: str,
        discovered_at: datetime,
        cache_hit: bool,
        skipped: Mapping[str, str],
    ) -> OptionContractPlan:
        """Select from metadata first; qualify and later stream only one pair."""

        ordered_strikes = sorted(
            {strike for strike in metadata_strikes if math.isfinite(strike) and strike > 0.0},
            key=lambda strike: (
                abs(strike - underlying_reference),
                strike,
            ),
        )
        attempt_limit = min(
            len(ordered_strikes),
            1 + self.common_strike_fallback_attempts,
        )
        candidates_inspected = 0
        for strike in ordered_strikes[:attempt_limit]:
            rights: tuple[Literal["C", "P"], ...] = ("C", "P")
            unresolved_pair = tuple(
                OptionContract(
                    underlying_con_id=episode.underlying.con_id,
                    con_id=None,
                    expiry=expiry,
                    dte=(expiry - episode.session).days,
                    dte_bucket=DteBucket.ONE_DTE,
                    strike=strike,
                    right=right,
                    multiplier=100,
                    exchange=exchange,
                    trading_class=trading_class,
                )
                for right in rights
            )
            resolved_pair: list[ResolvedOptionContract] = []
            for contract in unresolved_pair:
                candidates_inspected += 1
                source, context_cache_hit, _ = self._opening_reversal_discovery_context[
                    episode.episode_id
                ]
                self._opening_reversal_discovery_context[episode.episode_id] = (
                    source,
                    context_cache_hit,
                    candidates_inspected,
                )
                resolved = self._resolve_uncached(
                    episode.symbol,
                    contract,
                )
                if resolved is not None:
                    resolved_pair.append(resolved)
            if len(resolved_pair) != 2:
                continue
            for unresolved, resolved in zip(
                unresolved_pair,
                resolved_pair,
                strict=True,
            ):
                self._resolved_contracts[unresolved.con_id_key] = resolved
                self._resolved_contracts[resolved.contract.con_id_key] = resolved
            resolved_candidates = tuple(
                OptionContractCandidateV1(
                    con_id=cast(int, resolved.contract.con_id),
                    underlying=episode.symbol,
                    expiry=resolved.contract.expiry,
                    strike=resolved.contract.strike,
                    right=resolved.contract.right,
                    multiplier=resolved.contract.multiplier,
                    exchange=resolved.contract.exchange,
                    trading_class=resolved.contract.trading_class,
                )
                for resolved in resolved_pair
            )
            audit = select_primary_option_pair_v1(
                session=episode.session,
                underlying_reference=underlying_reference,
                candidates=resolved_candidates,
                discovery_timestamp_utc=discovered_at,
                contract_source=(
                    "cached_ibkr_secdef_metadata" if cache_hit else "ibkr_secdef_metadata"
                ),
                cache_hit=cache_hit,
                candidates_inspected=candidates_inspected,
            )
            self._opening_reversal_audits[episode.episode_id] = audit
            self.option_recorder.repository.record_opening_reversal_contract_discovery_v1(
                self.metadata_factory(discovered_at, (discovered_at,)),
                episode_id=episode.episode_id,
                selection=audit,
            )
            return OptionContractPlan(
                contracts=unresolved_pair,
                requested_contract_count=2,
                maximum_contracts=2,
                capacity_reduced=False,
                missing_buckets=tuple(
                    f"{bucket}:{reason}" for bucket, reason in sorted(skipped.items())
                ),
                selection_rule=("opening_reversal_metadata_only_primary_1dte_atm_pair_v1"),
                selection_roles={
                    unresolved_pair[0].con_id_key: ("primary_atm_call",),
                    unresolved_pair[1].con_id_key: ("primary_atm_put",),
                },
            )
        raise ValueError("primary_1dte_option_pair_unavailable")

    def _select_snapshot_plan(
        self,
        *,
        episode: _PendingEpisode,
        candidates: OptionContractPlan,
        underlying_reference: float,
        primary_bucket: DteBucket,
        secondary_buckets: tuple[DteBucket, ...],
        skipped: Mapping[str, str],
    ) -> OptionContractPlan:
        primary = primary_bucket
        secondary = secondary_buckets
        ordered = sorted(
            candidates.contracts,
            key=lambda contract: (
                0
                if contract.dte_bucket == primary
                else 1
                if contract.dte_bucket in secondary
                else 2,
                abs(contract.strike - underlying_reference),
                contract.strike,
                contract.right,
            ),
        )[: self.maximum_discovery_snapshots]
        snapshots: dict[str, dict[str, object]] = {}
        for contract in ordered:
            resolved = self._resolve(episode.symbol, contract)
            if resolved is None:
                continue
            self._resolved_contracts[contract.con_id_key] = resolved
            if resolved.contract.con_id is not None:
                self._resolved_contracts[resolved.contract.con_id_key] = resolved
            payload = self._capture_snapshot(
                episode_id=episode.episode_id,
                contract=resolved,
            )
            if payload is not None:
                snapshots[contract.con_id_key] = payload
        primary_contracts = tuple(
            contract
            for contract in ordered
            if contract.dte_bucket == primary and contract.con_id_key in snapshots
        )
        selected: list[OptionContract] = []
        roles: dict[str, tuple[str, ...]] = {}
        if episode.episode_kind in {
            EpisodeKind.HIGH_TAIL,
            EpisodeKind.OPENING_REVERSAL,
        }:
            atm_pair = self._atm_pair(
                primary_contracts,
                snapshots=snapshots,
                underlying_reference=underlying_reference,
            )
            if atm_pair is None:
                raise ValueError("high_tail_atm_call_put_live_quote_unavailable")
            for contract, role in zip(
                atm_pair,
                ("primary_atm_call", "primary_atm_put"),
                strict=True,
            ):
                selected.append(contract)
                roles[contract.con_id_key] = (role,)
        else:
            condor = self._delta_condor(
                primary_contracts,
                snapshots=snapshots,
            )
            if condor is None:
                raise ValueError("primary_delta_iron_condor_not_available")
            condor_roles = (
                "primary_short_call_025_delta",
                "primary_short_put_minus_025_delta",
                "primary_long_call_010_delta",
                "primary_long_put_minus_010_delta",
            )
            for contract, role in zip(condor, condor_roles, strict=True):
                selected.append(contract)
                roles[contract.con_id_key] = (role,)
            atm_pair = self._atm_pair(
                primary_contracts,
                snapshots=snapshots,
                underlying_reference=underlying_reference,
            )
            if atm_pair is not None:
                for contract, role in zip(
                    atm_pair,
                    ("comparison_atm_call", "comparison_atm_put"),
                    strict=True,
                ):
                    if contract not in selected and len(selected) < self.maximum_continuous_lines:
                        selected.append(contract)
                    roles[contract.con_id_key] = (
                        *roles.get(contract.con_id_key, ()),
                        role,
                    )
            for bucket in secondary:
                if len(selected) + 2 > self.maximum_continuous_lines:
                    break
                pair = self._atm_pair(
                    tuple(
                        contract
                        for contract in ordered
                        if contract.dte_bucket == bucket and contract.con_id_key in snapshots
                    ),
                    snapshots=snapshots,
                    underlying_reference=underlying_reference,
                )
                if pair is None:
                    continue
                for contract, right_name in zip(
                    pair,
                    ("call", "put"),
                    strict=True,
                ):
                    if contract not in selected:
                        selected.append(contract)
                    roles[contract.con_id_key] = (f"alternate_dte_atm_{right_name}",)
        if episode.episode_kind is EpisodeKind.NEUTRAL_CONTROL:
            roles = {
                key: tuple(f"neutral_{role}" for role in value) for key, value in roles.items()
            }
        selected = selected[: self.maximum_continuous_lines]
        missing = tuple(f"{bucket}:{reason}" for bucket, reason in sorted(skipped.items()))
        return OptionContractPlan(
            contracts=tuple(selected),
            requested_contract_count=len(selected),
            maximum_contracts=self.maximum_continuous_lines,
            capacity_reduced=False,
            missing_buckets=missing,
            selection_rule=(
                "bounded_live_snapshot_delta_condor_and_atm_comparisons_v0"
                if episode.episode_kind
                not in {
                    EpisodeKind.HIGH_TAIL,
                    EpisodeKind.OPENING_REVERSAL,
                }
                else (
                    "opening_reversal_primary_1dte_atm_call_put_v1"
                    if episode.episode_kind is EpisodeKind.OPENING_REVERSAL
                    else "bounded_live_snapshot_atm_call_put_v0"
                )
            ),
            selection_roles={
                contract.con_id_key: roles[contract.con_id_key] for contract in selected
            },
        )

    def _capture_snapshot(
        self,
        *,
        episode_id: str,
        contract: ResolvedOptionContract,
    ) -> dict[str, object] | None:
        capture = getattr(self.adapter, "capture_temporary_quote", None)
        if not callable(capture):
            raise ValueError("bounded_option_snapshot_capability_unavailable")
        assert contract.contract.con_id is not None
        snapshot_id = f"{episode_id}:{contract.contract.con_id}"
        state = self.episode_state_machine
        if state is not None and not state.snapshots.reserve(snapshot_id):
            return None
        try:
            result = capture(
                contract=contract.upstream_contract,
                generic_ticks="",
            )
            if self.heartbeat is not None:
                self.heartbeat()
        except (RuntimeError, TimeoutError):
            return None
        finally:
            if state is not None:
                state.snapshots.release(snapshot_id)
        merged = merge_snapshot_items(result.items)
        market_data_type = str(merged.get("market_data_type", "")).lower()
        bid = _attribute(merged, "bid")
        ask = _attribute(merged, "ask")
        if (
            market_data_type != "live"
            or bid is None
            or ask is None
            or float(cast(Any, bid)) < 0.0
            or float(cast(Any, ask)) <= 0.0
            or float(cast(Any, bid)) > float(cast(Any, ask))
        ):
            return None
        return merged

    @staticmethod
    def _atm_pair(
        contracts: tuple[OptionContract, ...],
        *,
        snapshots: Mapping[str, Mapping[str, object]],
        underlying_reference: float,
    ) -> tuple[OptionContract, OptionContract] | None:
        pair: list[OptionContract] = []
        for right in ("C", "P"):
            candidate = min(
                (
                    contract
                    for contract in contracts
                    if (contract.right == right and contract.con_id_key in snapshots)
                ),
                key=lambda contract: (
                    abs(contract.strike - underlying_reference),
                    contract.strike,
                ),
                default=None,
            )
            if candidate is None:
                return None
            pair.append(candidate)
        return pair[0], pair[1]

    @staticmethod
    def _delta_contract(
        contracts: tuple[OptionContract, ...],
        *,
        snapshots: Mapping[str, Mapping[str, object]],
        right: str,
        target: float,
        tolerance: float,
        excluded: set[str],
    ) -> OptionContract | None:
        candidates: list[tuple[float, float, OptionContract]] = []
        for contract in contracts:
            if contract.right != right or contract.con_id_key in excluded:
                continue
            raw_delta = snapshots.get(contract.con_id_key, {}).get("delta")
            if raw_delta is None:
                continue
            delta = float(cast(Any, raw_delta))
            distance = abs(delta - target)
            if math.isfinite(delta) and distance <= tolerance:
                candidates.append((distance, contract.strike, contract))
        return None if not candidates else min(candidates, key=lambda item: item[:2])[2]

    def _delta_condor(
        self,
        contracts: tuple[OptionContract, ...],
        *,
        snapshots: Mapping[str, Mapping[str, object]],
    ) -> tuple[OptionContract, OptionContract, OptionContract, OptionContract] | None:
        excluded: set[str] = set()
        short_call = self._delta_contract(
            contracts,
            snapshots=snapshots,
            right="C",
            target=0.25,
            tolerance=SHORT_DELTA_TOLERANCE,
            excluded=excluded,
        )
        if short_call is None:
            return None
        excluded.add(short_call.con_id_key)
        short_put = self._delta_contract(
            contracts,
            snapshots=snapshots,
            right="P",
            target=-0.25,
            tolerance=SHORT_DELTA_TOLERANCE,
            excluded=excluded,
        )
        if short_put is None:
            return None
        excluded.add(short_put.con_id_key)
        long_call = self._delta_contract(
            contracts,
            snapshots=snapshots,
            right="C",
            target=0.10,
            tolerance=LONG_DELTA_TOLERANCE,
            excluded=excluded,
        )
        if long_call is None:
            return None
        excluded.add(long_call.con_id_key)
        long_put = self._delta_contract(
            contracts,
            snapshots=snapshots,
            right="P",
            target=-0.10,
            tolerance=LONG_DELTA_TOLERANCE,
            excluded=excluded,
        )
        if long_put is None:
            return None
        if not (long_put.strike < short_put.strike < short_call.strike < long_call.strike):
            return None
        opening_credit = (
            float(cast(Any, snapshots[short_call.con_id_key]["bid"]))
            + float(cast(Any, snapshots[short_put.con_id_key]["bid"]))
            - float(cast(Any, snapshots[long_call.con_id_key]["ask"]))
            - float(cast(Any, snapshots[long_put.con_id_key]["ask"]))
        )
        if opening_credit <= 0.0:
            return None
        return short_call, short_put, long_call, long_put

    def _qualify_valid_common_strikes(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        candidate_strikes: tuple[float, ...],
        underlying_reference: float,
        exchange: str,
        trading_class: str,
    ) -> tuple[float, ...]:
        """Boundedly find exact strikes that resolve for both call and put."""

        available = sorted(set(candidate_strikes))
        if not available:
            return ()
        nearest_order = sorted(
            available,
            key=lambda strike: (abs(strike - underlying_reference), strike),
        )
        atm: float | None = None
        search_limit = min(
            len(nearest_order),
            1 + self.common_strike_fallback_attempts,
        )
        for strike in nearest_order[:search_limit]:
            if self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=strike,
                exchange=exchange,
                trading_class=trading_class,
            ):
                atm = strike
                break
        if atm is None:
            return ()

        if episode.quiet_state:
            return self._qualify_quiet_symmetric_strikes(
                episode=episode,
                expiry=expiry,
                available=tuple(available),
                atm=atm,
                underlying_reference=underlying_reference,
                exchange=exchange,
                trading_class=trading_class,
            )

        selected = [atm]
        side_limit = episode.strike_steps + self.common_strike_fallback_attempts
        for side_candidates in (
            sorted(strike for strike in available if strike > atm),
            sorted((strike for strike in available if strike < atm), reverse=True),
        ):
            found = 0
            for strike in side_candidates[:side_limit]:
                if self._qualify_pair(
                    episode=episode,
                    expiry=expiry,
                    strike=strike,
                    exchange=exchange,
                    trading_class=trading_class,
                ):
                    selected.append(strike)
                    found += 1
                    if found >= episode.strike_steps:
                        break
        return tuple(sorted(selected))

    def _qualify_quiet_symmetric_strikes(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        available: tuple[float, ...],
        atm: float,
        underlying_reference: float,
        exchange: str,
        trading_class: str,
    ) -> tuple[float, ...]:
        """Select frozen broad offsets while preserving exact symmetric fly wings."""

        upper = {round(strike - atm, 10): strike for strike in available if strike > atm}
        lower = {round(atm - strike, 10): strike for strike in available if strike < atm}
        remaining = set(upper).intersection(lower)
        ordered_distances: list[float] = []
        minimum_distance = QUIET_WING_TARGET_FRACTIONS[0] * underlying_reference
        eligible_wings = {
            distance for distance in remaining if distance + 1e-12 >= minimum_distance
        }
        if eligible_wings:
            nearest_valid_wing = min(eligible_wings)
            ordered_distances.append(nearest_valid_wing)
            remaining.remove(nearest_valid_wing)
        for fraction in QUIET_WING_TARGET_FRACTIONS[1:]:
            if not remaining:
                break
            target = fraction * underlying_reference
            eligible_remaining = {
                distance for distance in remaining if distance + 1e-12 >= minimum_distance
            }
            if not eligible_remaining:
                break
            distance = min(
                eligible_remaining,
                key=lambda value: (abs(value - target), value),
            )
            ordered_distances.append(distance)
            remaining.remove(distance)
        ordered_distances.extend(
            sorted(distance for distance in remaining if distance + 1e-12 >= minimum_distance)
        )
        attempt_limit = episode.strike_steps + self.common_strike_fallback_attempts
        selected = [atm]
        selected_pairs = 0
        for distance in ordered_distances[:attempt_limit]:
            upper_strike = upper[distance]
            lower_strike = lower[distance]
            upper_valid = self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=upper_strike,
                exchange=exchange,
                trading_class=trading_class,
            )
            lower_valid = self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=lower_strike,
                exchange=exchange,
                trading_class=trading_class,
            )
            if not upper_valid or not lower_valid:
                continue
            selected.extend((lower_strike, upper_strike))
            selected_pairs += 1
            if selected_pairs >= episode.strike_steps:
                break
        return tuple(sorted(selected))

    def _qualify_pair(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        strike: float,
        exchange: str,
        trading_class: str,
    ) -> bool:
        pair: list[tuple[OptionContract, ResolvedOptionContract]] = []
        for right in ("C", "P"):
            unresolved = OptionContract(
                underlying_con_id=episode.underlying.con_id,
                con_id=None,
                expiry=expiry,
                dte=(expiry - episode.session).days,
                dte_bucket=next(
                    bucket
                    for bucket, selection in select_expiries(
                        episode.session,
                        (expiry,),
                    ).items()
                    if selection.expiry == expiry
                ),
                strike=strike,
                right=right,
                multiplier=100,
                exchange=exchange,
                trading_class=trading_class,
            )
            resolved = self._resolve_uncached(episode.symbol, unresolved)
            if resolved is None:
                return False
            pair.append((unresolved, resolved))
        for unresolved, resolved in pair:
            self._resolved_contracts[unresolved.con_id_key] = resolved
        return True

    def _resolve(
        self,
        symbol: str,
        contract: OptionContract,
    ) -> ResolvedOptionContract | None:
        cached = self._resolved_contracts.get(contract.con_id_key)
        if cached is not None:
            return cached
        return self._resolve_uncached(symbol, contract)

    def _resolve_uncached(
        self,
        symbol: str,
        contract: OptionContract,
    ) -> ResolvedOptionContract | None:
        upstream = self.contract_factory(
            symbol,
            contract.expiry,
            contract.strike,
            contract.right,
            contract.multiplier,
            contract.exchange,
            contract.trading_class,
        )
        result = self.adapter.qualify_exact_contract(upstream)
        if self.heartbeat is not None:
            self.heartbeat()
        matching = []
        for detail in result.items:
            qualified = _attribute(detail, "contract") or detail
            expiry = _parse_expiry(
                _attribute(
                    qualified,
                    "lastTradeDateOrContractMonth",
                    "expiry",
                )
            )
            if (
                str(_attribute(qualified, "symbol") or "") == symbol
                and str(_attribute(qualified, "secType", "sec_type") or "") == "OPT"
                and expiry == contract.expiry
                and float(_attribute(qualified, "strike") or 0.0) == contract.strike
                and str(_attribute(qualified, "right") or "") == contract.right
                and int(_attribute(qualified, "conId", "con_id") or 0) > 0
            ):
                matching.append(qualified)
        if len(matching) != 1:
            return None
        qualified = matching[0]
        con_id = int(_attribute(qualified, "conId", "con_id"))
        return ResolvedOptionContract(
            contract=OptionContract(
                **{
                    **contract.__dict__,
                    "con_id": con_id,
                    "exchange": str(_attribute(qualified, "exchange") or contract.exchange),
                    "trading_class": str(
                        _attribute(
                            qualified,
                            "tradingClass",
                            "trading_class",
                        )
                        or contract.trading_class
                    ),
                }
            ),
            upstream_contract=qualified,
        )

    @property
    def rejections(self) -> dict[str, str]:
        return dict(self._rejections)

    def pending_symbol(self, episode_id: str) -> str | None:
        episode = self._pending.get(episode_id)
        return None if episode is None else episode.symbol

    @property
    def finalizations(self) -> dict[str, OptionEpisodeFinalization]:
        return dict(self._finalizations)

    @property
    def live_option_quote_seen(self) -> bool:
        return self.option_recorder.live_option_quote_seen

    @property
    def option_computation_seen(self) -> bool:
        return self.option_recorder.option_computation_seen


__all__ = ["BoundedOptionDiscoveryService"]
