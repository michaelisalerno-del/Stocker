"""Activation-bounded live coordinator for frozen M1C and raw market evidence."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from stocker_prospective.context import previous_xnys_session
from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.direction_features import DirectionFeatureBar
from stocker_prospective.durable_inbox import (
    CallbackInboxError,
    CallbackInboxEvent,
    DurableCallbackInbox,
)
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
    NormalizedCallback,
    StreamKind,
    StreamOwner,
)
from stocker_prospective.events import (
    FiveMinuteBarEvent,
    OptionQuoteEvent,
    RawEvent,
    UnderlyingDepthEvent,
    UnderlyingDepthSnapshot,
    UnderlyingDepthSnapshotEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickBidAskEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.group_o import FrozenGroupOContext
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.live_bars import (
    AuditedFiveMinuteBarAdapter,
    AuditedLiveBar,
    HistoricalBarUpdate,
    KeepUpToDateBarFinalizer,
    xnys_session_bounds,
)
from stocker_prospective.m1c_features import (
    FROZEN_CHECKPOINTS,
    HistoricalActivityBaseline,
    LiveFeatureBar,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionReceiptV1,
    PostEntryBarV1,
    build_incomplete_opening_reversal_outcome_v1,
    build_opening_reversal_outcome_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    OpeningReversalActivationReceiptV1_1,
    OpeningReversalCausalBarrierAuditV1_1,
    OpeningReversalDecisionDataGateV1_1,
    build_causal_barrier_audit_v1_1,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.microstructure import (
    STANDARD_WINDOWS,
    compare_frozen_archetypes,
    episode_relative_windows,
    standard_window_summaries,
    summarise_microstructure_window,
)
from stocker_prospective.opening_market_transition_v1 import (
    calculate_opening_preentry_window_v1,
    classify_opening_market_transition_v1,
)
from stocker_prospective.operational_state import (
    GapIncident,
    OperationalThresholds,
    RecorderOperationalRepository,
    stable_gap_id,
)
from stocker_prospective.order_book import DepthBook
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
    RecorderCheckpointResult,
)
from stocker_prospective.signed_market_shock_v1 import MarketShockBarV1

MetadataFactory = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]
GroupOProvider = Callable[[str, date], FrozenGroupOContext]
OptionQuoteSink = Callable[[EvidenceMetadata, OptionQuoteEvent], None]
EpisodeCallback = Callable[[RecorderCheckpointResult], None]
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ScientificReadiness:
    m1c_parity_passed: bool
    direction_parity_passed: bool
    bar_compatibility_passed: bool
    clock_drift_within_tolerance: bool
    historical_activity_baseline_available: bool = True
    capability_preflight_passed: bool = False


class LivePollResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    callback_count: int
    raw_event_count: int
    finalised_bar_count: int
    checkpoint_count: int
    fresh_episode_count: int
    partition_hashes: tuple[str, ...]
    raw_event_ids: tuple[str, ...] = ()
    blocked_checkpoints: dict[str, str]
    checkpoint_results: tuple[RecorderCheckpointResult, ...]
    opening_reversal_prediction_receipts: tuple[
        OpeningReversalPredictionReceiptV1,
        ...,
    ] = ()
    opening_reversal_causal_barrier_audits_v1_1: tuple[
        OpeningReversalCausalBarrierAuditV1_1,
        ...,
    ] = ()
    depth_reset_symbols: tuple[str, ...] = ()
    ibkr_errors: tuple[tuple[int, int], ...] = ()
    broker_mutations: int = 0
    durable_inbox_event_ids: tuple[str, ...] = ()
    durable_lease_batch_id: str | None = None
    raw_materialization_reused: bool = False


@dataclass(frozen=True)
class _DeferredDecisionEventV1_1:
    event: RawEvent
    completed_bar: AuditedLiveBar | None = None


class CallbackNormalizationFatal(RuntimeError):
    """A leased callback could not be deterministically normalised."""

    def __init__(self, callback_index: int) -> None:
        super().__init__("CALLBACK_NORMALIZATION_FAILED")
        self.callback_index = callback_index


def _bar_event(
    bar: AuditedLiveBar,
    *,
    request_id: int,
    source_sequence: int,
    received_monotonic_ns: int,
) -> FiveMinuteBarEvent:
    identity = (
        f"{bar.symbol}|{bar.session.isoformat()}|{bar.checkpoint}|"
        f"{bar.bar_end_utc.isoformat()}|{source_sequence}"
    )
    return FiveMinuteBarEvent(
        event_id=hashlib.sha256(identity.encode()).hexdigest(),
        received_timestamp_utc=bar.received_timestamp_utc,
        received_monotonic_ns=received_monotonic_ns,
        provider_timestamp_utc=bar.provider_timestamp_utc,
        source_sequence=source_sequence,
        session=bar.session,
        symbol=bar.symbol,
        con_id=0,
        request_id=request_id,
        bar_start_utc=bar.bar_start_utc,
        bar_end_utc=bar.bar_end_utc,
        checkpoint=bar.checkpoint,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume_or_activity_field=bar.volume_or_activity_field,
        wap_where_available=bar.wap_where_available,
        trade_count_where_available=bar.trade_count_where_available,
        source=bar.source,
        source_completeness=bar.source_completeness,
        finalised=bar.finalised,
    )


def _event_type_name(event: RawEvent) -> str:
    return {
        "UnderlyingLevel1QuoteEvent": "underlying_level1_quote_event",
        "UnderlyingTickBidAskEvent": "underlying_tick_bidask_event",
        "UnderlyingTickTradeEvent": "underlying_tick_trade_event",
        "UnderlyingDepthEvent": "underlying_depth_event",
        "UnderlyingDepthSnapshotEvent": "underlying_depth_snapshot",
        "OptionQuoteEvent": "option_quote_event",
        "FiveMinuteBarEvent": "five_minute_bar_event",
    }[type(event).__name__]


class FrozenM1CLiveRecorder:
    """Drain callbacks, persist raw evidence, and score only completed causal bars."""

    def __init__(
        self,
        *,
        adapter: IBKRMarketDataAdapter,
        normalizer: IBKRCallbackNormalizer,
        raw_store: PartitionedEventStore,
        repository: FrozenRecorderRepository,
        engine: FrozenM1CRecorderEngine,
        activity_baseline: HistoricalActivityBaseline,
        group_o_provider: GroupOProvider,
        metadata_factory: MetadataFactory,
        run_id: str,
        universe_symbols: tuple[str, ...],
        market_proxy_symbol: str,
        sector_proxy_by_symbol: Mapping[str, str] | None = None,
        readiness: ScientificReadiness,
        maximum_quote_age: timedelta,
        maximum_clock_drift_seconds: float = 2.0,
        minimum_trade_classification_valid_fraction: float = 0.5,
        depth_rows: int = 5,
        depth_snapshot_interval: timedelta = timedelta(seconds=1),
        option_quote_sink: OptionQuoteSink | None = None,
        episode_callback: EpisodeCallback | None = None,
        durable_inbox: DurableCallbackInbox | None = None,
        recorder_generation: int | None = None,
        lease_owner: str | None = None,
        inbox_lease_timeout: timedelta = timedelta(seconds=30),
        inbox_batch_limit: int = 2_048,
        failure_injector: Callable[[str], None] | None = None,
        operational_repository: RecorderOperationalRepository | None = None,
        operational_thresholds: OperationalThresholds | None = None,
    ) -> None:
        if len(universe_symbols) != 20 or len(set(universe_symbols)) != 20:
            raise ValueError("frozen M1C live recorder requires the exact 20-stock cohort")
        if market_proxy_symbol in universe_symbols:
            raise ValueError("market proxy must remain outside the stock cohort")
        sector_proxies = dict(sector_proxy_by_symbol or {})
        if sector_proxies and set(sector_proxies) != set(universe_symbols):
            raise ValueError("sector proxy map must cover the exact stock cohort")
        if set(sector_proxies.values()).intersection((*universe_symbols, market_proxy_symbol)):
            raise ValueError("sector proxies must remain outside stocks and the market proxy")
        if maximum_quote_age <= timedelta(0):
            raise ValueError("maximum quote age must be positive")
        if maximum_clock_drift_seconds <= 0.0:
            raise ValueError("maximum clock drift must be positive")
        if not 0.0 <= minimum_trade_classification_valid_fraction <= 1.0:
            raise ValueError("trade-classification quality threshold is invalid")
        if depth_rows <= 0:
            raise ValueError("depth rows must be positive")
        if depth_snapshot_interval <= timedelta(0):
            raise ValueError("depth snapshot interval must be positive")
        if durable_inbox is not None and (
            recorder_generation is None or recorder_generation <= 0 or not lease_owner
        ):
            raise ValueError("durable callback lease identity is required")
        if inbox_lease_timeout <= timedelta(0) or inbox_batch_limit <= 0:
            raise ValueError("durable callback lease bounds must be positive")
        self.adapter = adapter
        self.normalizer = normalizer
        self.raw_store = raw_store
        self.repository = repository
        self.engine = engine
        self.activity_baseline = activity_baseline
        self.group_o_provider = group_o_provider
        self.metadata_factory = metadata_factory
        self.run_id = run_id
        self.universe_symbols = universe_symbols
        self.market_proxy_symbol = market_proxy_symbol
        self.sector_proxy_by_symbol = sector_proxies
        self.context_proxy_symbols = frozenset((market_proxy_symbol, *sector_proxies.values()))
        self.readiness = readiness
        self.maximum_quote_age = maximum_quote_age
        self.maximum_clock_drift_seconds = maximum_clock_drift_seconds
        self.minimum_trade_classification_valid_fraction = (
            minimum_trade_classification_valid_fraction
        )
        self.depth_rows = depth_rows
        self.depth_snapshot_interval = depth_snapshot_interval
        self.option_quote_sink = option_quote_sink
        self.episode_callback = episode_callback
        self.durable_inbox = durable_inbox
        self.recorder_generation = recorder_generation
        self.lease_owner = lease_owner
        self.inbox_lease_timeout = inbox_lease_timeout
        self.inbox_batch_limit = inbox_batch_limit
        self.failure_injector = failure_injector
        self.operational_repository = operational_repository
        self.operational_thresholds = operational_thresholds or OperationalThresholds()
        self._inflight_durable_events: tuple[CallbackInboxEvent, ...] = ()
        self._finalizer = KeepUpToDateBarFinalizer(
            prospective_collection_start=normalizer.prospective_collection_start
        )
        self._bar_adapter = AuditedFiveMinuteBarAdapter()
        self._bars: dict[tuple[str, date], dict[int, AuditedLiveBar]] = {}
        self._processed = repository.recorded_checkpoint_identities(run_id=run_id)
        self._blocked: dict[tuple[str, date, int], str] = {}
        self._latest_quotes: dict[str, UnderlyingLevel1QuoteEvent] = {}
        self._quotes: dict[str, deque[UnderlyingLevel1QuoteEvent]] = {}
        self._trades: dict[str, deque[UnderlyingTickTradeEvent]] = {}
        self._books: dict[str, DepthBook] = {}
        self._last_depth_snapshot_at: dict[str, datetime] = {}
        self._last_depth_validity: dict[str, bool] = {}
        self._bar_order: dict[tuple[int, datetime], tuple[int, int]] = {}
        self._episode_windows: dict[tuple[str, str], tuple[str, datetime, datetime]] = {}
        self._episode_actions: dict[str, dict[str, str]] = {}
        self._opening_reversal_outcome_inputs: dict[
            str,
            tuple[OpeningReversalPredictionReceiptV1, float | None],
        ] = {}
        self._quiet_observation_ids: set[str] = set()
        self._completed_episode_windows: set[tuple[str, str]] = set()
        self._gap_symbols: set[str] = set()
        self._gap_intervals: dict[
            str,
            list[tuple[datetime, datetime | None]],
        ] = {}
        self._active_gap_ids: dict[
            tuple[str, str, int | None, Literal["optional", "degraded", "scientific"]],
            list[tuple[str, int]],
        ] = {}
        if self.operational_repository is not None and self.recorder_generation is not None:
            restored_scientific_starts: dict[str, datetime] = {}
            for gap in self.operational_repository.active_gaps(run_id=self.run_id):
                key = (gap.symbol, gap.stream_kind, gap.request_id, gap.severity)
                self._active_gap_ids.setdefault(key, []).append(
                    (gap.gap_id, gap.recorder_generation)
                )
                if gap.severity == "scientific":
                    self._gap_symbols.add(gap.symbol)
                    prior = restored_scientific_starts.get(gap.symbol)
                    restored_scientific_starts[gap.symbol] = (
                        gap.start_timestamp_utc
                        if prior is None
                        else min(prior, gap.start_timestamp_utc)
                    )
            for symbol, started_at in restored_scientific_starts.items():
                self._gap_intervals[symbol] = [(started_at, None)]
        self._capability_preflight_passed = readiness.capability_preflight_passed
        self._session_context_ready = True
        self._scientific_prerequisites_passed = (
            readiness.m1c_parity_passed
            and readiness.direction_parity_passed
            and readiness.bar_compatibility_passed
            and readiness.historical_activity_baseline_available
            and readiness.clock_drift_within_tolerance
        )
        self._scientific_scoring_enabled = (
            readiness.capability_preflight_passed
            and self._scientific_prerequisites_passed
            and self._session_context_ready
        )
        self._clock_drift_seconds: float | None = None
        self._depth_exchanges: tuple[str, ...] = ()
        self._opening_reversal_decision_gate_v1_1 = (
            None
            if getattr(
                engine,
                "opening_reversal_activation_v1_1",
                None,
            )
            is None
            else OpeningReversalDecisionDataGateV1_1(
                protected_symbols=frozenset(
                    (
                        *universe_symbols,
                        *self.context_proxy_symbols,
                    )
                )
            )
        )
        self._deferred_decision_events_v1_1: dict[
            date,
            list[_DeferredDecisionEventV1_1],
        ] = {}
        self._opening_reversal_gate_sessions_restored_v1_1: set[date] = set()

    def register_stream(self, owner: StreamOwner) -> None:
        self.normalizer.register(owner)
        if owner.kind is StreamKind.UNDERLYING_BAR:
            self._finalizer.register(
                owner.request_id,
                symbol=owner.symbol,
                con_id=owner.con_id,
            )
        elif owner.kind is StreamKind.UNDERLYING_DEPTH:
            self._books[owner.symbol] = DepthBook(
                symbol=owner.symbol,
                con_id=owner.con_id,
                rows_per_side=self.depth_rows,
            )
            self._last_depth_snapshot_at.pop(owner.symbol, None)
            self._last_depth_validity.pop(owner.symbol, None)

    def mark_gap(
        self,
        symbol: str,
        *,
        started_at: datetime,
        cause_code: str = "REQUIRED_STREAM_INTERRUPTION",
        request_id: int | None = None,
        stream_kind: str = "required_market_stream",
        recoverability: Literal["recoverable", "unrecoverable", "unknown"] = "unknown",
        severity: Literal["optional", "degraded", "scientific"] = "scientific",
    ) -> None:
        if symbol in self.universe_symbols or symbol in self.context_proxy_symbols:
            observed = started_at.astimezone(UTC)
            gap_key = (symbol, stream_kind, request_id, severity)
            if gap_key in self._active_gap_ids:
                return
            if severity == "scientific":
                self._gap_symbols.add(symbol)
                intervals = self._gap_intervals.setdefault(symbol, [])
                if not intervals or intervals[-1][1] is not None:
                    intervals.append((observed, None))
            if self.operational_repository is not None and self.recorder_generation is not None:
                gap_id = stable_gap_id(
                    run_id=self.run_id,
                    recorder_generation=self.recorder_generation,
                    symbol=symbol,
                    stream_kind=stream_kind,
                    request_id=request_id,
                    connection_generation=self.adapter.connection_generation,
                    start_timestamp_utc=observed,
                    cause_code=cause_code,
                )
                self.operational_repository.record_gap(
                    GapIncident(
                        gap_id=gap_id,
                        run_id=self.run_id,
                        recorder_generation=self.recorder_generation,
                        symbol=symbol,
                        stream_kind=stream_kind,
                        request_id=request_id,
                        connection_generation=self.adapter.connection_generation,
                        start_timestamp_utc=observed,
                        detection_timestamp_utc=observed,
                        cause_code=cause_code,
                        severity=severity,
                        recoverability=recoverability,
                    )
                )
                self._active_gap_ids[gap_key] = [(gap_id, self.recorder_generation)]

    def _resolve_active_gaps(
        self,
        *,
        symbol: str,
        completed_at: datetime,
        resolution_evidence: str,
        stream_kind: str | None = None,
        request_id: int | None = None,
        scientific_only: bool = False,
    ) -> None:
        completed = completed_at.astimezone(UTC)
        for gap_key, active_incidents in tuple(self._active_gap_ids.items()):
            gap_symbol, gap_stream_kind, gap_request_id, severity = gap_key
            if (
                gap_symbol != symbol
                or (stream_kind is not None and gap_stream_kind != stream_kind)
                or (request_id is not None and gap_request_id != request_id)
                or (scientific_only and severity != "scientific")
            ):
                continue
            if self.operational_repository is not None and self.recorder_generation is not None:
                for gap_id, owning_generation in active_incidents:
                    self.operational_repository.resolve_gap(
                        gap_id=gap_id,
                        run_id=self.run_id,
                        recorder_generation=owning_generation,
                        resolved_at=completed,
                        resolution_evidence=resolution_evidence,
                        end_timestamp_utc=completed,
                    )
            del self._active_gap_ids[gap_key]

    def clear_gap_after_complete_bar(self, symbol: str, *, completed_at: datetime) -> None:
        completed = completed_at.astimezone(UTC)
        self._resolve_active_gaps(
            symbol=symbol,
            completed_at=completed,
            resolution_evidence="complete_required_bar_observed_after_gap",
            stream_kind=StreamKind.UNDERLYING_BAR.value,
            scientific_only=True,
        )
        scientific_gap_remains = any(
            gap_symbol == symbol and severity == "scientific"
            for gap_symbol, _, _, severity in self._active_gap_ids
        )
        if not scientific_gap_remains:
            intervals = self._gap_intervals.get(symbol)
            if intervals and intervals[-1][1] is None:
                started, _ = intervals[-1]
                intervals[-1] = (started, max(started, completed))
            self._gap_symbols.discard(symbol)

    def gap_overlaps(
        self,
        symbol: str,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> bool:
        start = window_start.astimezone(UTC)
        end = window_end.astimezone(UTC)
        return any(
            gap_start <= end and (gap_end is None or gap_end >= start)
            for gap_start, gap_end in self._gap_intervals.get(symbol, ())
        )

    def episode_has_gap(self, episode_id: str) -> bool:
        return any(
            self.gap_overlaps(
                symbol,
                window_start=start,
                window_end=end,
            )
            for (candidate, _), (symbol, start, end) in self._episode_windows.items()
            if candidate == episode_id
        )

    def set_capability_preflight(self, *, passed: bool) -> None:
        """Enable scoring only after the live capability manifest passes."""

        self._capability_preflight_passed = passed
        self._scientific_scoring_enabled = (
            passed and self._scientific_prerequisites_passed and self._session_context_ready
        )

    def set_session_context_ready(self, *, passed: bool) -> None:
        """Keep missing D-1 context retryable without consuming checkpoints."""

        self._session_context_ready = passed
        self._scientific_scoring_enabled = (
            self._capability_preflight_passed
            and self._scientific_prerequisites_passed
            and self._session_context_ready
        )

    def acknowledge_checkpoint(self, result: RecorderCheckpointResult) -> None:
        """Suppress replay only after the application commits all side effects."""

        decision = result.episode_decision
        self._processed.add((decision.symbol, decision.session, decision.checkpoint))

    def first_valid_quote_at_or_after(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> UnderlyingLevel1QuoteEvent | None:
        candidates = (
            event
            for event in self._quotes.get(symbol, ())
            if event.ordering_timestamp >= timestamp and event.quote_valid
        )
        return min(
            candidates,
            key=lambda item: (
                item.ordering_timestamp,
                item.received_monotonic_ns,
                item.source_sequence,
                item.event_id,
            ),
            default=None,
        )

    def episode_window_completed(self, episode_id: str, window_name: str) -> bool:
        return (episode_id, window_name) in self._completed_episode_windows

    def underlying_price_path(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> tuple[float, ...]:
        """Return the retained Level-I/trade path in deterministic event order."""

        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or end.tzinfo is None
            or end.utcoffset() is None
        ):
            raise ValueError("underlying path timestamps must be timezone-aware")
        if end < start:
            raise ValueError("underlying path cannot end before it starts")
        points: list[tuple[datetime, int, str, float]] = []
        for quote in self._quotes.get(symbol, ()):
            if not start <= quote.ordering_timestamp <= end:
                continue
            price: float | None = None
            if (
                quote.quote_valid
                and quote.bid is not None
                and quote.ask is not None
                and math.isfinite(quote.bid)
                and math.isfinite(quote.ask)
                and 0.0 < quote.bid <= quote.ask
            ):
                price = (quote.bid + quote.ask) / 2.0
            elif quote.last is not None and math.isfinite(quote.last) and quote.last > 0.0:
                price = quote.last
            if price is not None:
                points.append(
                    (
                        quote.ordering_timestamp,
                        quote.source_sequence,
                        "level_i",
                        float(price),
                    )
                )
        for trade in self._trades.get(symbol, ()):
            if (
                start <= trade.ordering_timestamp <= end
                and math.isfinite(trade.price)
                and trade.price > 0.0
            ):
                points.append(
                    (
                        trade.ordering_timestamp,
                        trade.source_sequence,
                        "tick_trade",
                        float(trade.price),
                    )
                )
        points.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return tuple(point[3] for point in points)

    def underlying_halted_in_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Return whether retained Level-I or trade evidence reports a halt."""

        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or end.tzinfo is None
            or end.utcoffset() is None
        ):
            raise ValueError("underlying halt timestamps must be timezone-aware")
        if end < start:
            raise ValueError("underlying halt window cannot end before it starts")
        return any(
            start <= event.ordering_timestamp <= end and event.halted is True
            for event in (
                *self._quotes.get(symbol, ()),
                *self._trades.get(symbol, ()),
            )
        )

    @property
    def latest_quotes(self) -> dict[str, UnderlyingLevel1QuoteEvent]:
        return dict(self._latest_quotes)

    @property
    def scientific_scoring_enabled(self) -> bool:
        return self._scientific_scoring_enabled

    @property
    def clock_drift_seconds(self) -> float | None:
        return self._clock_drift_seconds

    @property
    def depth_exchanges(self) -> tuple[str, ...]:
        return self._depth_exchanges

    def _persist_raw(
        self,
        metadata: EvidenceMetadata,
        events: tuple[RawEvent, ...],
        *,
        committed_at: datetime,
    ) -> tuple[str, ...]:
        if not events:
            return ()
        partitions = self.raw_store.write_grouped(
            data_source="ibkr",
            events=events,
            complete=True,
            # Retain the legacy manifest column without copying a batch-wide
            # count into every partition. GapIncident is the canonical tally.
            gap_count=0,
        )
        for partition in partitions:
            path_parts = {
                key: value
                for item in partition.data_path.parts
                if "=" in item
                for key, value in (item.split("=", maxsplit=1),)
            }
            self.repository.record_partition(
                metadata,
                data_source="ibkr",
                session_date=date.fromisoformat(path_parts["session_date"]),
                symbol=path_parts["symbol"],
                event_type=path_parts["event_type"],
                partition=partition,
            )
        if (
            self.operational_repository is not None
            and self.recorder_generation is not None
            and self.lease_owner is not None
        ):
            self.operational_repository.touch(
                run_id=self.run_id,
                recorder_generation=self.recorder_generation,
                owner_id=self.lease_owner,
                now=committed_at,
                latest_raw_partition_committed_at_utc=committed_at,
            )
        return tuple(item.content_hash for item in partitions)

    @staticmethod
    def _nominal_opening_reversal_entry_v1_1(session: date) -> datetime:
        session_open, _ = xnys_session_bounds(session)
        return session_open + timedelta(minutes=30)

    def _opening_reversal_gate_applies_v1_1(
        self,
        *,
        session: date,
    ) -> bool:
        activation = getattr(
            self.engine,
            "opening_reversal_activation_v1_1",
            None,
        )
        if (
            not isinstance(
                activation,
                OpeningReversalActivationReceiptV1_1,
            )
            or self._opening_reversal_decision_gate_v1_1 is None
        ):
            return False
        try:
            nominal_entry = self._nominal_opening_reversal_entry_v1_1(session)
        except ValueError:
            return False
        return nominal_entry > activation.activation_timestamp_utc

    def _restore_opening_reversal_gate_session_v1_1(
        self,
        *,
        session: date,
        observed_at_utc: datetime,
    ) -> None:
        gate = self._opening_reversal_decision_gate_v1_1
        if gate is None or session in self._opening_reversal_gate_sessions_restored_v1_1:
            return
        metadata = self.metadata_factory(
            observed_at_utc,
            (observed_at_utc,),
        )
        audit = self.repository.load_opening_reversal_causal_barrier_audit_v1_1(
            run_id=metadata.run_id,
            session=session,
        )
        if audit is not None:
            if audit.barrier_status == "passed":
                gate.authorize_release_after_durable_audit(
                    session=session,
                    audit_hash_v1_1=audit.audit_hash_v1_1,
                )
            else:
                assert audit.failure_reason is not None
                gate.fail_closed_for_science_and_continue_core(
                    session=session,
                    reason=audit.failure_reason,
                )
        self._opening_reversal_gate_sessions_restored_v1_1.add(session)

    @staticmethod
    def _decision_event_ordering_timestamp_v1_1(
        event: RawEvent,
    ) -> datetime:
        # A bar belongs to its bar-start interval.  The 09:55 bar is the
        # sixth frozen predictor even though it completes at the 10:00
        # boundary; the 10:00 bar is the first protected entry bar.
        if isinstance(event, FiveMinuteBarEvent):
            return event.bar_start_utc
        return event.ordering_timestamp

    def _route_decision_event_v1_1(
        self,
        event: RawEvent,
        *,
        completed_bar: AuditedLiveBar | None = None,
    ) -> Literal["admit", "buffer"]:
        gate = self._opening_reversal_decision_gate_v1_1
        if gate is None or not self._opening_reversal_gate_applies_v1_1(session=event.session):
            return "admit"
        self._restore_opening_reversal_gate_session_v1_1(
            session=event.session,
            observed_at_utc=event.received_timestamp_utc,
        )
        disposition = gate.observe(
            session=event.session,
            symbol=event.symbol,
            nominal_entry_timestamp_utc=(self._nominal_opening_reversal_entry_v1_1(event.session)),
            event_ordering_timestamp_utc=(self._decision_event_ordering_timestamp_v1_1(event)),
            event_received_timestamp_utc=event.received_timestamp_utc,
            event_id=event.event_id,
        )
        if disposition == "buffer":
            deferred = self._deferred_decision_events_v1_1.setdefault(
                event.session,
                [],
            )
            if not any(item.event.event_id == event.event_id for item in deferred):
                deferred.append(
                    _DeferredDecisionEventV1_1(
                        event=event,
                        completed_bar=completed_bar,
                    )
                )
        return disposition

    def _admit_decision_event_v1_1(
        self,
        event: RawEvent,
        *,
        completed_bar: AuditedLiveBar | None = None,
    ) -> tuple[RawEvent, ...]:
        admitted: list[RawEvent] = [event]
        if completed_bar is not None:
            self._bars.setdefault(
                (completed_bar.symbol, completed_bar.session),
                {},
            )[completed_bar.checkpoint] = completed_bar
            self.clear_gap_after_complete_bar(
                completed_bar.symbol,
                completed_at=completed_bar.bar_end_utc,
            )
        else:
            derived = self._retain(event)
            if derived is not None:
                admitted.append(derived)
            if isinstance(event, UnderlyingDepthEvent) and event.reset:
                self.mark_gap(
                    event.symbol,
                    started_at=event.received_timestamp_utc,
                    cause_code="OPTIONAL_DEPTH_RESET",
                    request_id=event.request_id,
                    stream_kind="underlying_depth",
                    recoverability="recoverable",
                    severity="optional",
                )
        return tuple(admitted)

    def _project_decision_event(
        self,
        metadata: EvidenceMetadata,
        event: RawEvent,
    ) -> None:
        if isinstance(event, UnderlyingLevel1QuoteEvent):
            self.repository.update_underlying_live_projection(
                metadata,
                event,
                tick_by_tick_status=(
                    "active"
                    if any(
                        owner.symbol == event.symbol
                        and owner.kind
                        in {
                            StreamKind.UNDERLYING_TICK_BIDASK,
                            StreamKind.UNDERLYING_TICK_LAST,
                        }
                        for owner in self.normalizer.owners
                    )
                    else "inactive"
                ),
                depth_status=("active" if event.symbol in self._books else "inactive"),
            )
        elif isinstance(event, FiveMinuteBarEvent):
            self.repository.update_completed_bar_projection(metadata, event)
        elif isinstance(event, OptionQuoteEvent) and self.option_quote_sink is not None:
            self.option_quote_sink(metadata, event)

    def _release_deferred_decision_events_v1_1(
        self,
        *,
        session: date,
    ) -> tuple[tuple[RawEvent, ...], tuple[RawEvent, ...]]:
        deferred = self._deferred_decision_events_v1_1.pop(session, [])
        admitted: list[RawEvent] = []
        newly_derived_raw: list[RawEvent] = []
        for item in deferred:
            released = self._admit_decision_event_v1_1(
                item.event,
                completed_bar=item.completed_bar,
            )
            admitted.extend(released)
            if len(released) > 1:
                newly_derived_raw.extend(released[1:])
        return tuple(admitted), tuple(newly_derived_raw)

    @staticmethod
    def _depth_snapshot_event(
        event: UnderlyingDepthEvent,
        snapshot: UnderlyingDepthSnapshot,
    ) -> UnderlyingDepthSnapshotEvent:
        return UnderlyingDepthSnapshotEvent(
            event_id=hashlib.sha256(f"{event.event_id}|depth_snapshot".encode()).hexdigest(),
            received_timestamp_utc=event.received_timestamp_utc,
            received_monotonic_ns=event.received_monotonic_ns,
            provider_timestamp_utc=event.provider_timestamp_utc,
            source_sequence=event.source_sequence,
            session=event.session,
            symbol=event.symbol,
            con_id=event.con_id,
            request_id=event.request_id,
            trigger_event_id=event.event_id,
            snapshot=snapshot,
        )

    def _retain(self, event: RawEvent) -> UnderlyingDepthSnapshotEvent | None:
        if isinstance(event, UnderlyingLevel1QuoteEvent):
            self._latest_quotes[event.symbol] = event
            self._quotes.setdefault(event.symbol, deque()).append(event)
        elif isinstance(event, UnderlyingTickBidAskEvent):
            synthetic = UnderlyingLevel1QuoteEvent(
                **event.model_dump(
                    exclude={
                        "bid_past_low",
                        "ask_past_high",
                        "market_data_type",
                    }
                ),
                bid=event.bid,
                bid_size=event.bid_size,
                ask=event.ask,
                ask_size=event.ask_size,
                last=None,
                last_size=None,
                market_data_type=event.market_data_type,
                source="official_ibkr_tick_by_tick_bidask",
                quote_valid=event.ask >= event.bid > 0.0,
                staleness_ms=None,
                tick_type="BidAsk",
                exchange=event.exchange,
                quote_attributes={
                    "bid_past_low": event.bid_past_low,
                    "ask_past_high": event.ask_past_high,
                },
                halted=None,
            )
            self._quotes.setdefault(event.symbol, deque()).append(synthetic)
        elif isinstance(event, UnderlyingTickTradeEvent):
            self._trades.setdefault(event.symbol, deque()).append(event)
        elif isinstance(event, UnderlyingDepthEvent):
            book = self._books.get(event.symbol)
            if book is not None:
                try:
                    book.apply(event)
                except ValueError:
                    book.reset(
                        event.received_timestamp_utc,
                        reason="depth_sequence_gap",
                    )
                    self.mark_gap(
                        event.symbol,
                        started_at=event.received_timestamp_utc,
                        cause_code="OPTIONAL_DEPTH_SEQUENCE_GAP",
                        request_id=event.request_id,
                        stream_kind="underlying_depth",
                        recoverability="recoverable",
                        severity="optional",
                    )
                    snapshot = book.snapshot(event.received_timestamp_utc)
                    self._last_depth_validity[event.symbol] = False
                    self._last_depth_snapshot_at[event.symbol] = event.received_timestamp_utc
                    return self._depth_snapshot_event(event, snapshot)
                snapshot = book.snapshot(event.received_timestamp_utc)
                if (
                    not snapshot.book_valid
                    and len(snapshot.bid_rows) >= self.depth_rows
                    and len(snapshot.ask_rows) >= self.depth_rows
                ):
                    book.mark_complete()
                    snapshot = book.snapshot(event.received_timestamp_utc)
                    self._resolve_active_gaps(
                        symbol=event.symbol,
                        completed_at=event.received_timestamp_utc,
                        resolution_evidence="complete_depth_book_observed_after_gap",
                        stream_kind="underlying_depth",
                        request_id=event.request_id,
                    )
                previous_validity = self._last_depth_validity.get(event.symbol)
                previous_timestamp = self._last_depth_snapshot_at.get(event.symbol)
                due = (
                    previous_timestamp is None
                    or event.received_timestamp_utc - previous_timestamp
                    >= self.depth_snapshot_interval
                    or previous_validity != snapshot.book_valid
                    or event.reset
                )
                self._last_depth_validity[event.symbol] = snapshot.book_valid
                if due:
                    self._last_depth_snapshot_at[event.symbol] = event.received_timestamp_utc
                    snapshot = book.snapshot(
                        event.received_timestamp_utc,
                        advance_centroid_baseline=True,
                    )
                    return self._depth_snapshot_event(event, snapshot)
        return None

    def _trim_history(self, as_of: datetime) -> None:
        cutoff = as_of - timedelta(minutes=60)
        for collection in (self._quotes, self._trades):
            for events in collection.values():
                while events and events[0].ordering_timestamp < cutoff:
                    events.popleft()

    def _consume_bar(
        self,
        update_request_id: int,
        bar_start: datetime,
        source_sequence: int,
        received_monotonic_ns: int,
        raw_events: list[RawEvent],
        admitted_decision_events: list[RawEvent],
        finalised_bars: list[AuditedLiveBar],
        *,
        update: HistoricalBarUpdate,
    ) -> None:
        self._bar_order[(update_request_id, bar_start)] = (
            source_sequence,
            received_monotonic_ns,
        )
        completed_updates = self._finalizer.add(
            request_id=update_request_id,
            bar_start_utc=update.bar_start_utc,
            provider_timestamp_utc=update.provider_timestamp_utc,
            received_timestamp_utc=update.received_timestamp_utc,
            open=update.open,
            high=update.high,
            low=update.low,
            close=update.close,
            volume=update.volume,
            wap=update.wap,
            trade_count=update.trade_count,
        )
        for completed in completed_updates:
            for bar in self._bar_adapter.add(completed):
                sequence, monotonic = self._bar_order[
                    (completed.request_id, completed.bar_start_utc)
                ]
                event = _bar_event(
                    bar,
                    request_id=completed.request_id,
                    source_sequence=sequence,
                    received_monotonic_ns=monotonic,
                ).model_copy(update={"con_id": completed.con_id})
                raw_events.append(event)
                finalised_bars.append(bar)
                if (
                    self._route_decision_event_v1_1(
                        event,
                        completed_bar=bar,
                    )
                    == "admit"
                ):
                    admitted_decision_events.extend(
                        self._admit_decision_event_v1_1(
                            event,
                            completed_bar=bar,
                        )
                    )

    def _close_opening_reversal_barrier_v1_1(
        self,
        *,
        observed_now: datetime,
    ) -> tuple[
        tuple[OpeningReversalCausalBarrierAuditV1_1, ...],
        tuple[RawEvent, ...],
        tuple[RawEvent, ...],
    ]:
        gate = self._opening_reversal_decision_gate_v1_1
        activation = getattr(
            self.engine,
            "opening_reversal_activation_v1_1",
            None,
        )
        if gate is None or activation is None:
            return (), (), ()
        session = observed_now.astimezone(NEW_YORK).date()
        if not self._opening_reversal_gate_applies_v1_1(session=session):
            return (), (), ()
        nominal_entry = self._nominal_opening_reversal_entry_v1_1(session)
        if observed_now < nominal_entry:
            return (), (), ()
        self._restore_opening_reversal_gate_session_v1_1(
            session=session,
            observed_at_utc=observed_now,
        )
        if gate.released(session):
            return (), (), ()
        metadata = self.metadata_factory(
            observed_now,
            (nominal_entry,),
        )
        receipts = tuple(
            receipt
            for symbol in self.universe_symbols
            if (
                receipt := (
                    self.repository.load_opening_reversal_prediction_v1(
                        run_id=metadata.run_id,
                        session=session,
                        stock=symbol,
                        experiment_version="1.1",
                    )
                )
            )
            is not None
        )
        deferred = tuple(
            item.event.received_timestamp_utc
            for item in self._deferred_decision_events_v1_1.get(
                session,
                (),
            )
        )
        audit = build_causal_barrier_audit_v1_1(
            activation_receipt_hash_v1_1=(activation.activation_receipt_hash_v1_1),
            session=session,
            nominal_entry_timestamp_utc=nominal_entry,
            prediction_receipts=receipts,
            deferred_event_received_timestamps=deferred,
            entry_or_post_entry_data_admitted_before_receipts=False,
            release_authorized_at_utc=observed_now,
        )
        try:
            self.repository.record_opening_reversal_causal_barrier_audit_v1_1(
                metadata,
                audit,
            )
        except Exception as error:
            failure_reason = f"causal_barrier_audit_not_durable:{type(error).__name__}"
            audit = build_causal_barrier_audit_v1_1(
                activation_receipt_hash_v1_1=(activation.activation_receipt_hash_v1_1),
                session=session,
                nominal_entry_timestamp_utc=nominal_entry,
                prediction_receipts=receipts,
                deferred_event_received_timestamps=deferred,
                entry_or_post_entry_data_admitted_before_receipts=False,
                release_authorized_at_utc=observed_now,
                operational_failure_reason=failure_reason,
            )
            gate.fail_closed_for_science_and_continue_core(
                session=session,
                reason=failure_reason,
            )
        else:
            if audit.barrier_status == "passed":
                gate.authorize_release_after_durable_audit(
                    session=session,
                    audit_hash_v1_1=audit.audit_hash_v1_1,
                )
            else:
                assert audit.failure_reason is not None
                gate.fail_closed_for_science_and_continue_core(
                    session=session,
                    reason=audit.failure_reason,
                )
        admitted, newly_derived_raw = self._release_deferred_decision_events_v1_1(
            session=session,
        )
        return (audit,), admitted, newly_derived_raw

    def _activate_v1_1_checkpoint_results_after_barrier(
        self,
        results: tuple[RecorderCheckpointResult, ...],
        *,
        barrier_passed_sessions: frozenset[date],
    ) -> None:
        for result in results:
            receipt = result.opening_reversal_prediction_v1
            if receipt is None or receipt.experiment_version != "1.1":
                continue
            if not result.episode_decision.fresh_episode:
                continue
            self._arm_episode_windows(result)
            episode_id = result.episode_decision.episode_id
            if (
                receipt.session in barrier_passed_sessions
                and receipt.scientific_outcome_eligible_v1
                and receipt.eligibility_v1
                and episode_id is not None
            ):
                self._opening_reversal_outcome_inputs[episode_id] = (
                    receipt,
                    result.movement_consumed_state_v1.movement_consumed_numerator_v1,
                )
            if self.episode_callback is not None:
                self.episode_callback(result)

    def _failure_checkpoint(self, phase: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(phase)

    def poll(self, *, now: datetime) -> LivePollResult:
        """Lease callbacks and durably materialise raw evidence.

        The owning application must call :meth:`finalize_durable_poll` only
        after every application-side effect and checkpoint marker is durable.
        """

        observed_now = now.astimezone(UTC)
        inbox = self.durable_inbox
        if inbox is None:
            return self._poll_callbacks(
                now=observed_now,
                callbacks=self.adapter.drain_stream_events(),
            )
        assert self.recorder_generation is not None
        assert self.lease_owner is not None
        if self._inflight_durable_events:
            raise RuntimeError("CALLBACK_DURABLE_POLL_NOT_FINALIZED")
        self.adapter.flush_pending_callback_failure()
        interrupted = inbox.quarantine_interrupted_provider_envelopes(
            current_recorder_generation=self.recorder_generation,
            observed_at=observed_now,
        )
        if interrupted:
            first = interrupted[0]
            for event in interrupted:
                inbox.record_incident(
                    stable_error_code="CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED",
                    component="official_ibkr_callback",
                    severity="fatal",
                    occurred_at=observed_now,
                    error_class="InterruptedProviderCallback",
                    evidence_loss_possible=True,
                    callback_kind=event.callback_kind,
                    request_id=event.request_id,
                    source_sequence=event.source_sequence,
                    connection_generation=event.connection_generation,
                    subscription_owner=event.subscription_owner,
                    symbol=event.symbol,
                )
            inbox.latch_fatal(
                latch_kind="ingestion",
                stable_error_code="CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED",
                occurred_at=observed_now,
                error_class="InterruptedProviderCallback",
                evidence_loss_possible=True,
                first_possibly_lost_source_sequence=first.source_sequence,
                callback_kind=first.callback_kind,
                request_id=first.request_id,
                connection_generation=first.connection_generation,
            )
        if self.adapter.fatal_callback_code is not None or inbox.has_active_fatal():
            self._scientific_scoring_enabled = False
        leased = inbox.lease(
            lease_owner=self.lease_owner,
            lease_generation=self.recorder_generation,
            now=observed_now,
            lease_timeout=self.inbox_lease_timeout,
            limit=self.inbox_batch_limit,
        )
        self._failure_checkpoint("after_callback_lease")
        pending: list[CallbackInboxEvent] = []
        for event in leased:
            committed_hashes = inbox.processing_commit(event.inbox_event_id)
            if committed_hashes is None:
                pending.append(event)
                continue
            inbox.acknowledge(
                (event,),
                lease_owner=self.lease_owner,
                lease_generation=self.recorder_generation,
                raw_partition_hashes=committed_hashes,
                acknowledged_at=observed_now,
            )
        pending_events = tuple(pending)
        callbacks = tuple(event.normalizer_payload() for event in pending_events)
        try:
            materialization = inbox.raw_materialization(pending_events)
            result = self._poll_callbacks(
                now=observed_now,
                callbacks=callbacks,
                precommitted_partition_hashes=(
                    None if materialization is None else materialization.partition_hashes
                ),
                expected_raw_event_ids=(
                    None if materialization is None else materialization.raw_event_ids
                ),
            )
            self._failure_checkpoint("before_callback_raw_materialization")
            inbox.commit_raw_materialization(
                pending_events,
                run_id=self.run_id,
                recorder_generation=self.recorder_generation,
                raw_partition_hashes=result.partition_hashes,
                raw_event_ids=result.raw_event_ids,
                materialized_at=observed_now,
            )
            self._failure_checkpoint("after_callback_raw_materialization")
            batch_ids = {event.lease_batch_id for event in pending_events}
            batch_id = None if not batch_ids else next(iter(batch_ids))
            self._inflight_durable_events = pending_events
            return result.model_copy(
                update={
                    "durable_inbox_event_ids": tuple(
                        event.inbox_event_id for event in pending_events
                    ),
                    "durable_lease_batch_id": batch_id,
                    "raw_materialization_reused": materialization is not None,
                }
            )
        except CallbackNormalizationFatal as exc:
            failed = pending_events[exc.callback_index]
            inbox.quarantine(
                failed,
                failure_classification="CALLBACK_NORMALIZATION_FAILED",
                lease_owner=self.lease_owner,
                lease_generation=self.recorder_generation,
                now=observed_now,
            )
            inbox.release(
                (
                    event
                    for index, event in enumerate(pending_events)
                    if index != exc.callback_index
                ),
                lease_owner=self.lease_owner,
                lease_generation=self.recorder_generation,
                now=observed_now,
            )
            inbox.record_incident(
                stable_error_code="CALLBACK_NORMALIZATION_FAILED",
                component="prospective_live_recorder",
                severity="fatal",
                occurred_at=observed_now,
                error_class=type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
                evidence_loss_possible=True,
                callback_kind=failed.callback_kind,
                request_id=failed.request_id,
                source_sequence=failed.source_sequence,
                connection_generation=failed.connection_generation,
                subscription_owner=failed.subscription_owner,
                symbol=failed.symbol,
            )
            inbox.latch_fatal(
                latch_kind="ingestion",
                stable_error_code="CALLBACK_NORMALIZATION_FAILED",
                occurred_at=observed_now,
                error_class=type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
                evidence_loss_possible=True,
                first_possibly_lost_source_sequence=failed.source_sequence,
                callback_kind=failed.callback_kind,
                request_id=failed.request_id,
                connection_generation=failed.connection_generation,
            )
            self._scientific_scoring_enabled = False
            raise
        except Exception as exc:
            inbox.release(
                pending_events,
                lease_owner=self.lease_owner,
                lease_generation=self.recorder_generation,
                now=observed_now,
            )
            storage_failure = isinstance(exc, (OSError, sqlite3.Error)) or any(
                token in str(exc).lower()
                for token in (
                    "partition",
                    "parquet",
                    "pyarrow",
                    "disk",
                    "storage",
                    "materialization",
                )
            )
            stable_code = (
                "RAW_STORAGE_COMMIT_FAILED" if storage_failure else "RECORDER_PROCESSING_FAILED"
            )
            latch_kind = "storage" if storage_failure else "ingestion"
            first_sequence = (
                None
                if not pending_events
                else min(event.source_sequence for event in pending_events)
            )
            inbox.record_incident(
                stable_error_code=stable_code,
                component="prospective_live_recorder",
                severity="fatal",
                occurred_at=observed_now,
                error_class=type(exc).__name__,
                evidence_loss_possible=True,
                source_sequence=first_sequence,
                connection_generation=self.adapter.connection_generation,
            )
            inbox.latch_fatal(
                latch_kind=latch_kind,
                stable_error_code=stable_code,
                occurred_at=observed_now,
                error_class=type(exc).__name__,
                evidence_loss_possible=True,
                first_possibly_lost_source_sequence=first_sequence,
                connection_generation=self.adapter.connection_generation,
            )
            self._scientific_scoring_enabled = False
            raise

    def finalize_durable_poll(
        self,
        result: LivePollResult,
        *,
        acknowledged_at: datetime,
    ) -> None:
        """Commit outer application completion, then generation-fenced ack."""

        inbox = self.durable_inbox
        if inbox is None:
            return
        assert self.recorder_generation is not None
        assert self.lease_owner is not None
        events = self._inflight_durable_events
        if tuple(event.inbox_event_id for event in events) != result.durable_inbox_event_ids:
            raise CallbackInboxError("CALLBACK_DURABLE_POLL_IDENTITY_CHANGED")
        observed = acknowledged_at.astimezone(UTC)
        self._failure_checkpoint("before_callback_processing_commit")
        inbox.commit_processing(
            events,
            run_id=self.run_id,
            recorder_generation=self.recorder_generation,
            raw_partition_hashes=result.partition_hashes,
            committed_at=observed,
        )
        self._failure_checkpoint("after_callback_processing_commit")
        inbox.acknowledge(
            events,
            lease_owner=self.lease_owner,
            lease_generation=self.recorder_generation,
            raw_partition_hashes=result.partition_hashes,
            acknowledged_at=observed,
        )
        self._failure_checkpoint("after_callback_acknowledgement")
        self._inflight_durable_events = ()

    def fail_inflight_durable_poll(
        self,
        error: Exception,
        *,
        occurred_at: datetime,
    ) -> None:
        """Persist a fatal outer-application failure before process exit."""

        inbox = self.durable_inbox
        events = self._inflight_durable_events
        if inbox is None:
            return
        assert self.recorder_generation is not None
        assert self.lease_owner is not None
        observed = occurred_at.astimezone(UTC)
        first_sequence = (
            min(event.source_sequence for event in events)
            if events
            else inbox.latest_source_sequence()
        )
        processing_committed = bool(events) and all(
            inbox.processing_commit(event.inbox_event_id) is not None for event in events
        )
        if processing_committed:
            # SQLite already proves every recorder/application side effect in
            # this lease is complete. A failed ack is a recoverable cursor
            # transition; the expired generation-fenced lease will be
            # reclaimed and acknowledged without re-projecting evidence.
            try:
                inbox.record_incident(
                    stable_error_code="CALLBACK_ACK_DEFERRED",
                    component="frozen_prospective_application",
                    severity="degraded",
                    occurred_at=observed,
                    error_class=type(error).__name__,
                    evidence_loss_possible=False,
                    source_sequence=first_sequence,
                    connection_generation=self.adapter.connection_generation,
                    details={
                        "processing_commit_present": True,
                        "callbacks_remain_unacknowledged": True,
                    },
                )
            except Exception:
                return
            return
        self._scientific_scoring_enabled = False
        storage_failure = isinstance(error, (OSError, sqlite3.Error)) or any(
            token in str(error).lower()
            for token in ("database", "partition", "parquet", "storage", "disk")
        )
        stable_code = (
            "RECORDER_APPLICATION_STORAGE_COMMIT_FAILED"
            if storage_failure
            else "RECORDER_APPLICATION_COMMIT_FAILED"
        )
        try:
            inbox.record_incident(
                stable_error_code=stable_code,
                component="frozen_prospective_application",
                severity="fatal",
                occurred_at=observed,
                error_class=type(error).__name__,
                evidence_loss_possible=True,
                source_sequence=first_sequence,
                connection_generation=self.adapter.connection_generation,
                details={
                    "callbacks_remain_unacknowledged": bool(events),
                    "failure_before_callback_lease": not events,
                },
            )
            inbox.latch_fatal(
                latch_kind="storage" if storage_failure else "ingestion",
                stable_error_code=stable_code,
                occurred_at=observed,
                error_class=type(error).__name__,
                evidence_loss_possible=True,
                first_possibly_lost_source_sequence=first_sequence,
                connection_generation=self.adapter.connection_generation,
            )
            if events:
                inbox.release(
                    events,
                    lease_owner=self.lease_owner,
                    lease_generation=self.recorder_generation,
                    now=observed,
                )
        except Exception:
            # This is the process-level failure boundary. Leaving the durable
            # lease untouched is safer than obscuring the original application
            # error or pretending the batch was released.
            return
        self._inflight_durable_events = ()

    def _poll_callbacks(
        self,
        *,
        now: datetime,
        callbacks: tuple[dict[str, Any], ...],
        precommitted_partition_hashes: tuple[str, ...] | None = None,
        expected_raw_event_ids: tuple[str, ...] | None = None,
    ) -> LivePollResult:
        observed_now = now.astimezone(UTC)
        raw_events: list[RawEvent] = []
        admitted_decision_events: list[RawEvent] = []
        finalised_bars: list[AuditedLiveBar] = []
        depth_reset_symbols: set[str] = set()
        ibkr_errors: list[tuple[int, int]] = []
        deferred_control_gaps: list[tuple[str, datetime]] = []
        normalized_callbacks: list[tuple[dict[str, Any], NormalizedCallback | None]] = []
        for callback_index, payload in enumerate(callbacks):
            try:
                normalized = self.normalizer.normalize(payload)
            except Exception as exc:
                raise CallbackNormalizationFatal(callback_index) from exc
            normalized_callbacks.append((payload, normalized))
        # No raw/derived side effect begins until the entire lease is known to
        # be normalisable. A poison callback therefore cannot partially apply
        # an earlier callback from the same batch.
        for payload, normalized in normalized_callbacks:
            if normalized is None:
                continue
            raw_disposition: Literal["admit", "buffer"] = "admit"
            if normalized.raw_event is not None:
                raw_events.append(normalized.raw_event)
                raw_disposition = self._route_decision_event_v1_1(normalized.raw_event)
                if raw_disposition == "admit":
                    admitted = self._admit_decision_event_v1_1(normalized.raw_event)
                    admitted_decision_events.extend(admitted)
                    if len(admitted) > 1:
                        raw_events.extend(admitted[1:])
            if normalized.historical_bar is not None:
                source_sequence = int(payload["source_sequence"])
                received_monotonic_ns = int(payload["received_monotonic_ns"])
                self._consume_bar(
                    normalized.historical_bar.request_id,
                    normalized.historical_bar.bar_start_utc,
                    source_sequence,
                    received_monotonic_ns,
                    raw_events,
                    admitted_decision_events,
                    finalised_bars,
                    update=normalized.historical_bar,
                )
            if normalized.control_kind == "depth_reset":
                control = normalized.control_payload or {}
                symbol = str(control.get("symbol", ""))
                started_at = (
                    normalized.raw_event.received_timestamp_utc
                    if normalized.raw_event is not None
                    else observed_now
                )
                if symbol:
                    depth_reset_symbols.add(symbol)
            elif normalized.control_kind == "current_time":
                control = normalized.control_payload or {}
                provider = control.get("provider_timestamp_utc")
                if isinstance(provider, str):
                    try:
                        parsed = datetime.fromisoformat(provider.replace("Z", "+00:00"))
                        received = datetime.fromisoformat(
                            str(control["received_timestamp_utc"]).replace("Z", "+00:00")
                        )
                    except (KeyError, ValueError):
                        self._clock_drift_seconds = None
                    else:
                        if (
                            parsed.tzinfo is None
                            or parsed.utcoffset() is None
                            or received.tzinfo is None
                            or received.utcoffset() is None
                        ):
                            self._clock_drift_seconds = None
                        else:
                            self._clock_drift_seconds = (
                                received.astimezone(UTC) - parsed.astimezone(UTC)
                            ).total_seconds()
            elif normalized.control_kind == "depth_exchanges":
                control = normalized.control_payload or {}
                exchanges = control.get("exchanges", ())
                self._depth_exchanges = tuple(
                    sorted(
                        {
                            str(item.get("exchange"))
                            for item in exchanges
                            if isinstance(item, dict) and item.get("exchange")
                        }
                    )
                )
            elif normalized.control_kind == "ibkr_error":
                request_id = int(payload["request_id"])
                raw_code = payload.get("error_code")
                try:
                    code = -1 if raw_code is None else int(raw_code)
                except (TypeError, ValueError):
                    code = -1
                ibkr_errors.append((request_id, code))
                owner = self.normalizer.owner(request_id)
                if owner is not None:
                    session = observed_now.astimezone(NEW_YORK).date()
                    gate = self._opening_reversal_decision_gate_v1_1
                    if (
                        gate is not None
                        and self._opening_reversal_gate_applies_v1_1(session=session)
                        and observed_now >= self._nominal_opening_reversal_entry_v1_1(session)
                        and not gate.released(session)
                    ):
                        deferred_control_gaps.append((owner.symbol, observed_now))
                    else:
                        optional_stream = owner.kind in {
                            StreamKind.UNDERLYING_DEPTH,
                            StreamKind.UNDERLYING_TICK_BIDASK,
                            StreamKind.UNDERLYING_TICK_LAST,
                        }
                        self.mark_gap(
                            owner.symbol,
                            started_at=observed_now,
                            cause_code=(
                                "OPTIONAL_STREAM_IBKR_ERROR"
                                if optional_stream
                                else "REQUIRED_STREAM_IBKR_ERROR"
                            ),
                            request_id=request_id,
                            stream_kind=owner.kind.value,
                            recoverability="unknown",
                            severity="optional" if optional_stream else "scientific",
                        )
        source_times = tuple(event.ordering_timestamp for event in raw_events)
        metadata = self.metadata_factory(
            max(
                (event.received_timestamp_utc for event in raw_events),
                default=observed_now,
            ),
            source_times or (observed_now,),
        )
        partition_hashes = (
            precommitted_partition_hashes
            if precommitted_partition_hashes is not None
            else self._persist_raw(
                metadata,
                tuple(raw_events),
                committed_at=observed_now,
            )
        )
        results = self._score_ready(observed_now=observed_now)
        complete_opening_receipts = tuple(
            receipt
            for result in results
            if (receipt := result.opening_reversal_prediction_v1) is not None
        )
        deadline_opening_receipts = self._emit_opening_reversal_deadline_receipts_v1(
            observed_now,
        )
        opening_receipts_by_hash = {
            receipt.receipt_hash_v1: receipt
            for receipt in (
                *complete_opening_receipts,
                *deadline_opening_receipts,
            )
        }
        (
            barrier_audits_v1_1,
            released_decision_events,
            newly_derived_raw_events,
        ) = self._close_opening_reversal_barrier_v1_1(
            observed_now=observed_now,
        )
        if newly_derived_raw_events and precommitted_partition_hashes is None:
            raw_events.extend(newly_derived_raw_events)
            partition_hashes = (
                *partition_hashes,
                *self._persist_raw(
                    metadata,
                    newly_derived_raw_events,
                    committed_at=observed_now,
                ),
            )
        elif newly_derived_raw_events:
            raw_events.extend(newly_derived_raw_events)
        raw_event_ids = tuple(sorted(event.event_id for event in raw_events))
        if expected_raw_event_ids is not None and raw_event_ids != tuple(
            sorted(expected_raw_event_ids)
        ):
            raise CallbackInboxError("CALLBACK_RAW_EVENT_IDENTITY_DIFFERS")
        for event in (
            *admitted_decision_events,
            *released_decision_events,
        ):
            self._project_decision_event(metadata, event)
        for symbol, started_at in deferred_control_gaps:
            self.mark_gap(symbol, started_at=started_at)
        barrier_passed_sessions = {
            audit.session for audit in barrier_audits_v1_1 if audit.barrier_status == "passed"
        }
        v1_1_result_sessions = {
            receipt.session
            for result in results
            if (receipt := result.opening_reversal_prediction_v1) is not None
            and receipt.experiment_version == "1.1"
        }
        for session in v1_1_result_sessions - barrier_passed_sessions:
            persisted = self.repository.load_opening_reversal_causal_barrier_audit_v1_1(
                run_id=metadata.run_id,
                session=session,
            )
            if persisted is not None and persisted.barrier_status == "passed":
                barrier_passed_sessions.add(session)
        self._activate_v1_1_checkpoint_results_after_barrier(
            results,
            barrier_passed_sessions=frozenset(barrier_passed_sessions),
        )
        self._record_due_episode_windows(observed_now)
        self._trim_history(observed_now)
        if (
            self.operational_repository is not None
            and self.recorder_generation is not None
            and self.lease_owner is not None
        ):
            try:
                session_open, session_close = xnys_session_bounds(
                    observed_now.astimezone(NEW_YORK).date()
                )
                market_session_open = session_open <= observed_now <= session_close
            except ValueError:
                market_session_open = False
            health = self.adapter.connection.health()
            completed_bar_times = tuple(
                event.bar_end_utc
                for event in raw_events
                if isinstance(event, FiveMinuteBarEvent) and event.finalised
            )
            self.operational_repository.touch(
                run_id=self.run_id,
                recorder_generation=self.recorder_generation,
                owner_id=self.lease_owner,
                now=observed_now,
                market_session_open=market_session_open,
                callbacks_expected=market_session_open and bool(self.normalizer.owners),
                ibkr_connection_state=health.state.value,
                observed_market_data_mode=(
                    None if health.market_data_type is None else health.market_data_type.value
                ),
                scientific_prerequisites_valid=(
                    self._scientific_prerequisites_passed
                    and self._capability_preflight_passed
                    and self._session_context_ready
                    and self.adapter.scientific_recording_valid
                ),
                latest_completed_five_minute_bar_at_utc=(
                    None if not completed_bar_times else max(completed_bar_times)
                ),
                latest_successful_checkpoint_at_utc=(None if not results else observed_now),
                broker_state_mutation_count=0,
            )
            self.operational_repository.refresh_projection(
                run_id=self.run_id,
                recorder_generation=self.recorder_generation,
                owner_id=self.lease_owner,
                now=observed_now,
                prospective_start_utc=self.raw_store.prospective_collection_start,
                thresholds=self.operational_thresholds,
            )
        return LivePollResult(
            callback_count=len(callbacks),
            raw_event_count=len(raw_events),
            finalised_bar_count=len(finalised_bars),
            checkpoint_count=len(results),
            fresh_episode_count=sum(result.episode_decision.fresh_episode for result in results),
            partition_hashes=partition_hashes,
            raw_event_ids=raw_event_ids,
            blocked_checkpoints={
                f"{symbol}:{session.isoformat()}:{checkpoint}": reason
                for (symbol, session, checkpoint), reason in sorted(self._blocked.items())
            },
            checkpoint_results=results,
            opening_reversal_prediction_receipts=tuple(opening_receipts_by_hash.values()),
            opening_reversal_causal_barrier_audits_v1_1=(barrier_audits_v1_1),
            depth_reset_symbols=tuple(sorted(depth_reset_symbols)),
            ibkr_errors=tuple(ibkr_errors),
        )

    def _emit_opening_reversal_deadline_receipts_v1(
        self,
        observed_now: datetime,
    ) -> tuple[OpeningReversalPredictionReceiptV1, ...]:
        """Freeze one ABSTAIN receipt per missing checkpoint-6 stock at 10:00."""

        if (
            not hasattr(self.engine, "opening_reversal_activation_v1")
            or self.engine.opening_reversal_activation_v1 is None
        ):
            return ()
        session = observed_now.astimezone(NEW_YORK).date()
        try:
            session_open, _ = xnys_session_bounds(session)
        except ValueError:
            return ()
        signal_timestamp = session_open + timedelta(minutes=30)
        if observed_now < signal_timestamp:
            return ()
        addendum = getattr(
            self.engine,
            "opening_reversal_activation_v1_1",
            None,
        )
        if addendum is not None and signal_timestamp <= addendum.activation_timestamp_utc:
            return ()
        market_previous_session, market_prior_close = self._market_opening_reference_v1(
            session=session
        )
        opening_window = calculate_opening_preentry_window_v1(
            market_proxy=self.market_proxy_symbol,
            session=session,
            previous_session=(
                session if market_previous_session is None else market_previous_session
            ),
            session_open_timestamp=session_open,
            signal_timestamp=signal_timestamp,
            entry_timestamp=signal_timestamp,
            completed_bars=self._market_shock_bars_v1(
                session=session,
                checkpoint=6,
            ),
            prior_regular_session_close=market_prior_close,
        )
        opening_state = classify_opening_market_transition_v1(
            window=opening_window,
            thresholds=self.engine.opening_transition_thresholds_v1,
        )
        receipts: list[OpeningReversalPredictionReceiptV1] = []
        run_id = self.metadata_factory(
            observed_now,
            (signal_timestamp,),
        ).run_id
        gate = self._opening_reversal_decision_gate_v1_1
        for symbol in self.universe_symbols:
            existing = self.repository.load_opening_reversal_prediction_v1(
                run_id=run_id,
                session=session,
                stock=symbol,
                experiment_version=("1" if addendum is None else "1.1"),
            )
            if existing is not None:
                continue
            stock_bars = self._bars.get((symbol, session), {})
            missing_stock = tuple(ordinal for ordinal in range(1, 7) if ordinal not in stock_bars)
            # A complete result would already have been created by _score_ready.
            # Any remaining row failed one of the causal checkpoint requirements.
            reason = (
                "checkpoint_6_missing_stock_bars:" + ",".join(str(value) for value in missing_stock)
                if missing_stock
                else self._blocked.get(
                    (symbol, session, 6),
                    "checkpoint_6_causal_inputs_incomplete",
                )
            )
            metadata = self.metadata_factory(
                observed_now,
                (signal_timestamp,),
            )
            try:
                group_o_context = self.group_o_provider(symbol, session)
            except (KeyError, RuntimeError, ValueError):
                group_o_context = None
            try:
                receipt = self.engine.build_incomplete_opening_reversal_prediction_v1(
                    metadata=metadata,
                    session=session,
                    stock=symbol,
                    signal_timestamp=signal_timestamp,
                    opening_window_v1=opening_window,
                    opening_transition_state_v1=opening_state,
                    group_o_context=group_o_context,
                    missing_reason=reason,
                    receipt_created_at_utc_v1_1=(observed_now if addendum is not None else None),
                    first_buffered_event_received_at_utc_v1_1=(
                        None if gate is None else gate.first_deferred_event_received_at(session)
                    ),
                    entry_data_admitted_before_receipt_v1_1=(
                        bool(gate is not None and gate.scientific_barrier_compromised(session))
                    ),
                )
            except Exception as error:
                self._blocked[(symbol, session, 6)] = (
                    f"opening_reversal_v1_1_receipt_failure:{type(error).__name__}"
                )
                continue
            receipts.append(receipt)
        return tuple(receipts)

    def _feature_bars(
        self,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
    ) -> tuple[LiveFeatureBar, ...]:
        bars = self._bars[(symbol, session)]
        output: list[LiveFeatureBar] = []
        for expected in range(1, checkpoint + 1):
            bar = bars[expected]
            if bar.volume_or_activity_field is None:
                raise ValueError("completed IBKR bar volume is unavailable")
            activity = self.activity_baseline.relative_activity(
                symbol=symbol,
                session=session,
                bar_ordinal=expected - 1,
                volume=bar.volume_or_activity_field,
            )
            if activity is None:
                raise ValueError("historical activity baseline is unavailable")
            output.append(
                LiveFeatureBar(
                    symbol=symbol,
                    session=session,
                    bar_ordinal=expected - 1,
                    bar_start_timestamp=bar.bar_start_utc,
                    bar_complete_timestamp=bar.bar_end_utc,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume_or_activity_field,
                    historical_relative_activity=activity,
                    finalised=True,
                    source=bar.source,
                )
            )
        return tuple(output)

    def _direction_bars(
        self,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
        feature_bars: tuple[LiveFeatureBar, ...],
    ) -> tuple[DirectionFeatureBar, ...]:
        market = self._bars[(self.market_proxy_symbol, session)]
        output: list[DirectionFeatureBar] = []
        previous_stock_close: float | None = None
        previous_market_close: float | None = None
        for expected, stock in enumerate(feature_bars, start=1):
            market_bar = market[expected]
            if stock.historical_relative_activity is None:
                raise ValueError("direction historical activity is unavailable")
            stock_denominator = stock.open if previous_stock_close is None else previous_stock_close
            market_denominator = (
                market_bar.open if previous_market_close is None else previous_market_close
            )
            if stock_denominator <= 0.0 or market_denominator <= 0.0:
                raise ValueError("direction bar return denominator is invalid")
            output.append(
                DirectionFeatureBar(
                    **stock.model_dump(
                        exclude={
                            "historical_relative_activity",
                            "source",
                        }
                    ),
                    historical_relative_activity=stock.historical_relative_activity,
                    stock_log_return=math.log(stock.close / stock_denominator),
                    market_log_return=math.log(market_bar.close / market_denominator),
                )
            )
            previous_stock_close = stock.close
            previous_market_close = market_bar.close
        if len(output) != checkpoint:
            raise ValueError("direction bar width differs from checkpoint")
        return tuple(output)

    def _market_shock_bars_v1(
        self,
        *,
        session: date,
        checkpoint: int,
    ) -> tuple[MarketShockBarV1, ...]:
        """Map the already-subscribed canonical proxy into causal logging bars."""

        bars = self._bars.get((self.market_proxy_symbol, session), {})
        return tuple(
            MarketShockBarV1(
                symbol=self.market_proxy_symbol,
                session=session,
                bar_ordinal=expected - 1,
                bar_start_timestamp=bar.bar_start_utc,
                bar_complete_timestamp=bar.bar_end_utc,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                finalised=bar.finalised,
            )
            for expected in range(1, checkpoint + 1)
            if (bar := bars.get(expected)) is not None
        )

    def _market_opening_reference_v1(
        self,
        *,
        session: date,
    ) -> tuple[date | None, float | None]:
        """Use only an already-retained final prior-session VTI close."""

        try:
            previous_session = previous_xnys_session(session)
            _, previous_close_timestamp = xnys_session_bounds(previous_session)
        except (RuntimeError, ValueError):
            return None, None
        bars = self._bars.get((self.market_proxy_symbol, previous_session), {})
        completed = [bar for bar in bars.values() if bar.finalised and bar.close > 0.0]
        if not completed:
            return previous_session, None
        final = max(completed, key=lambda bar: bar.bar_end_utc)
        if final.bar_end_utc != previous_close_timestamp:
            return previous_session, None
        return previous_session, float(final.close)

    def _score_ready(
        self,
        *,
        observed_now: datetime,
    ) -> tuple[RecorderCheckpointResult, ...]:
        if not self._scientific_scoring_enabled:
            return ()
        results: list[RecorderCheckpointResult] = []
        for symbol in self.universe_symbols:
            sessions = sorted(session for candidate, session in self._bars if candidate == symbol)
            for session in sessions:
                stock_bars = self._bars[(symbol, session)]
                market_bars = self._bars.get((self.market_proxy_symbol, session), {})
                for checkpoint in FROZEN_CHECKPOINTS:
                    key = (symbol, session, checkpoint)
                    if key in self._processed:
                        continue
                    required = set(range(1, checkpoint + 1))
                    if not required.issubset(stock_bars) or not required.issubset(market_bars):
                        continue
                    try:
                        feature_bars = self._feature_bars(
                            symbol=symbol,
                            session=session,
                            checkpoint=checkpoint,
                        )
                        direction_bars = self._direction_bars(
                            symbol=symbol,
                            session=session,
                            checkpoint=checkpoint,
                            feature_bars=feature_bars,
                        )
                        context = self.group_o_provider(symbol, session)
                        latest = self._latest_quotes.get(symbol)
                        trigger_end = feature_bars[-1].bar_complete_timestamp
                        quote_fresh = (
                            latest is not None
                            and latest.quote_valid
                            and latest.market_data_type.primary_eligible
                            and abs((latest.ordering_timestamp - trigger_end).total_seconds())
                            <= self.maximum_quote_age.total_seconds()
                        )
                        v1_1_checkpoint = (
                            checkpoint == 6
                            and self._opening_reversal_gate_applies_v1_1(session=session)
                        )
                        metadata = self.metadata_factory(
                            (
                                observed_now
                                if v1_1_checkpoint
                                else feature_bars[-1].bar_complete_timestamp
                            ),
                            (
                                feature_bars[-1].bar_complete_timestamp,
                                latest.ordering_timestamp
                                if latest is not None
                                else feature_bars[-1].bar_complete_timestamp,
                            ),
                        )
                        self.repository.record_group_o_context(metadata, context)
                        (
                            market_previous_session_v1,
                            market_prior_regular_session_close_v1,
                        ) = self._market_opening_reference_v1(session=session)
                        result = self.engine.process_checkpoint(
                            RecorderCheckpointInput(
                                metadata=metadata,
                                symbol=symbol,
                                session=session,
                                completed_m1c_bars=feature_bars,
                                completed_direction_bars=direction_bars,
                                group_o_context=context,
                                market_data_type=(
                                    latest.market_data_type
                                    if latest is not None
                                    else (
                                        self.adapter.connection.health().market_data_type
                                        or MarketDataType.UNKNOWN
                                    )
                                ),
                                capability_preflight_passed=(
                                    self._capability_preflight_passed
                                    and self._scientific_scoring_enabled
                                ),
                                m1c_parity_passed=(
                                    self.readiness.m1c_parity_passed
                                    and self.readiness.bar_compatibility_passed
                                ),
                                direction_parity_passed=self.readiness.direction_parity_passed,
                                clock_drift_within_tolerance=(
                                    self.readiness.clock_drift_within_tolerance
                                    and self._clock_drift_seconds is not None
                                    and abs(self._clock_drift_seconds)
                                    <= self.maximum_clock_drift_seconds
                                ),
                                underlying_quote_fresh=quote_fresh,
                                unresolved_bar_gap=symbol in self._gap_symbols,
                                raw_event_storage_writable=True,
                                completed_market_shock_bars_v1=(
                                    self._market_shock_bars_v1(
                                        session=session,
                                        checkpoint=checkpoint,
                                    )
                                ),
                                market_previous_session_v1=(market_previous_session_v1),
                                market_prior_regular_session_close_v1=(
                                    market_prior_regular_session_close_v1
                                ),
                                opening_reversal_receipt_created_at_utc_v1_1=(
                                    observed_now if v1_1_checkpoint else None
                                ),
                                opening_reversal_first_buffered_event_received_at_utc_v1_1=(
                                    None
                                    if (
                                        not v1_1_checkpoint
                                        or self._opening_reversal_decision_gate_v1_1 is None
                                    )
                                    else (
                                        self._opening_reversal_decision_gate_v1_1.first_deferred_event_received_at(
                                            session
                                        )
                                    )
                                ),
                                opening_reversal_entry_data_admitted_before_receipt_v1_1=(
                                    bool(
                                        v1_1_checkpoint
                                        and self._opening_reversal_decision_gate_v1_1 is not None
                                        and (
                                            self._opening_reversal_decision_gate_v1_1.scientific_barrier_compromised(
                                                session
                                            )
                                        )
                                    )
                                ),
                            )
                        )
                    except (KeyError, ValueError) as exc:
                        self._blocked[key] = str(exc)
                        self._processed.add(key)
                        continue
                    self._blocked.pop(key, None)
                    results.append(result)
                    self._record_standard_windows(
                        symbol=symbol,
                        as_of=feature_bars[-1].bar_complete_timestamp,
                        metadata=metadata,
                    )
                    v1_1_receipt = result.opening_reversal_prediction_v1 is not None and (
                        result.opening_reversal_prediction_v1.experiment_version == "1.1"
                    )
                    if result.episode_decision.fresh_episode and not v1_1_receipt:
                        self._arm_episode_windows(result)
                        receipt = result.opening_reversal_prediction_v1
                        if (
                            receipt is not None
                            and receipt.scientific_outcome_eligible_v1
                            and receipt.eligibility_v1
                            and result.episode_decision.episode_id is not None
                        ):
                            self._opening_reversal_outcome_inputs[
                                result.episode_decision.episode_id
                            ] = (
                                receipt,
                                result.movement_consumed_state_v1.movement_consumed_numerator_v1,
                            )
                        if self.episode_callback is not None:
                            self.episode_callback(result)
                    if not v1_1_receipt and (
                        result.quiet_observation_id is not None
                        or result.neutral_control_id is not None
                        or result.high_tail_control_id is not None
                    ):
                        self._arm_quiet_windows(result)
        return tuple(results)

    def _record_standard_windows(
        self,
        *,
        symbol: str,
        as_of: datetime,
        metadata: EvidenceMetadata,
    ) -> None:
        summaries = standard_window_summaries(
            symbol=symbol,
            as_of=as_of,
            quotes=tuple(self._quotes.get(symbol, ())),
            trades=tuple(self._trades.get(symbol, ())),
            maximum_quote_age=self.maximum_quote_age,
            minimum_classification_valid_fraction=(
                self.minimum_trade_classification_valid_fraction
            ),
        )
        names = ("1s", "5s", "15s", "30s", "60s", "5m")
        for name, summary in zip(names, summaries, strict=True):
            self.repository.record_microstructure_summary(
                metadata,
                episode_id=None,
                window_name=name,
                summary=summary,
                level1_valid=bool(self._quotes.get(symbol)),
                tick_valid=summary.trade_flow.classification_valid_fraction
                >= self.minimum_trade_classification_valid_fraction,
                depth_valid=(
                    symbol in self._books and self._books[symbol].snapshot(as_of).book_valid
                ),
                quality_flags=(
                    ("data_gap",)
                    if self.gap_overlaps(
                        symbol,
                        window_start=as_of - STANDARD_WINDOWS[-1],
                        window_end=as_of,
                    )
                    else ()
                ),
            )

    def _arm_episode_windows(self, result: RecorderCheckpointResult) -> None:
        decision = result.episode_decision
        if decision.episode_id is None or decision.trigger_bar_end is None:
            return
        self._episode_actions[decision.episode_id] = {
            model_id: classification.action
            for model_id, classification in result.directional_classifications.items()
        }
        for name, (start, end) in episode_relative_windows(
            trigger_timestamp=decision.trigger_bar_end,
            entry_timestamp=decision.prospective_entry_timestamp,
        ).items():
            self._episode_windows[(decision.episode_id, name)] = (
                decision.symbol,
                start,
                end,
            )

    def _arm_quiet_windows(self, result: RecorderCheckpointResult) -> None:
        decision = result.quiet_episode_decision
        identities = tuple(
            value
            for value in (
                result.quiet_observation_id,
                result.neutral_control_id,
                result.high_tail_control_id,
            )
            if value is not None
        )
        windows = {
            **episode_relative_windows(
                trigger_timestamp=decision.trigger_timestamp,
                entry_timestamp=decision.prospective_entry_timestamp,
            ),
            "entry_to_+60m": (
                decision.prospective_entry_timestamp,
                decision.prospective_entry_timestamp + timedelta(minutes=60),
            ),
        }
        for observation_id in identities:
            self._quiet_observation_ids.add(observation_id)
            for name, (start, end) in windows.items():
                self._episode_windows[(observation_id, name)] = (
                    decision.symbol,
                    start,
                    end,
                )

    @staticmethod
    def _proxy_path_projection(
        symbol: str,
        prices: tuple[float, ...],
    ) -> dict[str, object]:
        entry = prices[0] if prices else None
        terminal = prices[-1] if prices else None
        minimum = min(prices) if prices else None
        maximum = max(prices) if prices else None
        return {
            "symbol": symbol,
            "entry_reference_price": entry,
            "terminal_reference_price": terminal,
            "minimum_reference_price": minimum,
            "maximum_reference_price": maximum,
            "maximum_absolute_excursion": (
                None if entry is None else max(abs(price - entry) for price in prices)
            ),
            "maximum_up_return": (
                None if entry is None or maximum is None else maximum / entry - 1.0
            ),
            "maximum_down_return": (
                None if entry is None or minimum is None else minimum / entry - 1.0
            ),
            "terminal_return": (
                None if entry is None or terminal is None else terminal / entry - 1.0
            ),
            "path_point_count": len(prices),
        }

    def _quiet_underlying_path(
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        quality_flags: tuple[str, ...],
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Build a directionless path projection from retained Level I and trades."""

        quotes = tuple(
            quote
            for quote in self._quotes.get(symbol, ())
            if start <= quote.ordering_timestamp <= end
        )
        trades = tuple(
            trade
            for trade in self._trades.get(symbol, ())
            if start <= trade.ordering_timestamp <= end
        )
        points: list[tuple[datetime, int, str, float]] = []
        for quote in quotes:
            price: float | None = None
            if (
                quote.quote_valid
                and quote.bid is not None
                and quote.ask is not None
                and math.isfinite(quote.bid)
                and math.isfinite(quote.ask)
                and 0.0 < quote.bid <= quote.ask
            ):
                price = (quote.bid + quote.ask) / 2.0
            elif quote.last is not None and math.isfinite(quote.last) and quote.last > 0.0:
                price = quote.last
            if price is not None:
                points.append(
                    (
                        quote.ordering_timestamp,
                        quote.source_sequence,
                        "level_i",
                        float(price),
                    )
                )
        for trade in trades:
            if math.isfinite(trade.price) and trade.price > 0.0:
                points.append(
                    (
                        trade.ordering_timestamp,
                        trade.source_sequence,
                        "tick_trade",
                        float(trade.price),
                    )
                )
        points.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        flags = set(quality_flags)
        if not quotes:
            flags.add("underlying_quote_unavailable")
        if not points:
            flags.add("underlying_path_unavailable")
        if any(quote.market_data_type is not MarketDataType.LIVE for quote in quotes) or any(
            trade.market_data_type is not MarketDataType.LIVE for trade in trades
        ):
            flags.add("market_data_not_live")
        halted = any(quote.halted is True for quote in quotes) or any(
            trade.halted is True for trade in trades
        )
        if halted:
            flags.add("underlying_halted")
        prices = tuple(point[3] for point in points)
        entry = prices[0] if prices else None
        terminal = prices[-1] if prices else None
        minimum = min(prices) if prices else None
        maximum = max(prices) if prices else None
        payload: dict[str, object] = {
            "source": "retained_underlying_level_i_and_tick_trades",
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "entry_reference_price": entry,
            "terminal_reference_price": terminal,
            "minimum_reference_price": minimum,
            "maximum_reference_price": maximum,
            "maximum_absolute_excursion": (
                None if entry is None else max(abs(price - entry) for price in prices)
            ),
            "maximum_up_return": (
                None if entry is None or maximum is None else maximum / entry - 1.0
            ),
            "maximum_down_return": (
                None if entry is None or minimum is None else minimum / entry - 1.0
            ),
            "terminal_return": (
                None if entry is None or terminal is None else terminal / entry - 1.0
            ),
            "quote_observation_count": len(quotes),
            "trade_observation_count": len(trades),
            "path_point_count": len(points),
            "first_observation_timestamp_utc": (None if not points else points[0][0].isoformat()),
            "last_observation_timestamp_utc": (None if not points else points[-1][0].isoformat()),
            "underlying_halted": halted,
        }
        market_prices = self.underlying_price_path(
            self.market_proxy_symbol,
            start,
            end,
        )
        sector_proxy_symbol = self.sector_proxy_by_symbol.get(symbol)
        sector_prices = (
            ()
            if sector_proxy_symbol is None
            else self.underlying_price_path(sector_proxy_symbol, start, end)
        )
        payload["market_proxy_path"] = self._proxy_path_projection(
            self.market_proxy_symbol,
            market_prices,
        )
        payload["sector_proxy_path"] = (
            None
            if sector_proxy_symbol is None
            else self._proxy_path_projection(sector_proxy_symbol, sector_prices)
        )
        if not market_prices:
            flags.add("market_proxy_path_unavailable")
        if sector_proxy_symbol is None or not sector_prices:
            flags.add("sector_proxy_path_unavailable")
        if self.gap_overlaps(
            self.market_proxy_symbol,
            window_start=start,
            window_end=end,
        ):
            flags.add("market_proxy_data_gap")
        if sector_proxy_symbol is not None and self.gap_overlaps(
            sector_proxy_symbol,
            window_start=start,
            window_end=end,
        ):
            flags.add("sector_proxy_data_gap")
        return payload, tuple(sorted(flags))

    def _record_opening_reversal_outcome_v1(
        self,
        *,
        episode_id: str,
        symbol: str,
        now: datetime,
    ) -> None:
        inputs = self._opening_reversal_outcome_inputs.get(episode_id)
        if inputs is None:
            return
        receipt, pre_trigger_range = inputs
        metadata = self.metadata_factory(
            now,
            (receipt.entry_timestamp_utc + timedelta(minutes=15),),
        )
        missing_reason: str | None = None
        if self.gap_overlaps(
            symbol,
            window_start=receipt.entry_timestamp_utc,
            window_end=receipt.entry_timestamp_utc + timedelta(minutes=15),
        ):
            missing_reason = "post_entry_data_gap"
        session_bars = self._bars.get((symbol, receipt.session), {})
        audited = tuple(session_bars.get(checkpoint) for checkpoint in (7, 8, 9))
        if missing_reason is None and any(bar is None for bar in audited):
            missing_reason = "post_entry_bar_missing"
        if missing_reason is None and any(bar is not None and not bar.finalised for bar in audited):
            missing_reason = "post_entry_bar_partial"
        if missing_reason is not None:
            outcome = build_incomplete_opening_reversal_outcome_v1(
                prediction_receipt=receipt,
                missing_reason_v1=missing_reason,
                outcome_created_at_utc=now,
            )
        else:
            complete_bars = tuple(
                PostEntryBarV1(
                    ordinal=index,
                    bar_start_timestamp_utc=bar.bar_start_utc,
                    bar_complete_timestamp_utc=bar.bar_end_utc,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    finalised=bar.finalised,
                )
                for index, candidate in enumerate(audited)
                if (bar := candidate) is not None
            )
            post_ten_minute_range = math.log(
                max(bar.high for bar in complete_bars[:2])
                / min(bar.low for bar in complete_bars[:2])
            )
            range_share = (
                post_ten_minute_range / (pre_trigger_range + post_ten_minute_range)
                if pre_trigger_range is not None
                and math.isfinite(pre_trigger_range)
                and pre_trigger_range >= 0.0
                and post_ten_minute_range >= 0.0
                and pre_trigger_range + post_ten_minute_range > 0.0
                else None
            )
            assert receipt.previous_close_atm_iv_scale_15m is not None
            outcome = build_opening_reversal_outcome_v1(
                prediction_receipt=receipt,
                completed_post_entry_bars=complete_bars,
                threshold_15m=receipt.previous_close_atm_iv_scale_15m,
                outcome_created_at_utc=now,
                canonical_post_entry_local_range_share_v1=range_share,
            )
        self.repository.record_opening_reversal_underlying_outcome_v1(
            metadata,
            outcome,
        )

    def _record_due_episode_windows(self, now: datetime) -> None:
        for identity, (symbol, start, end) in sorted(self._episode_windows.items()):
            if identity in self._completed_episode_windows or end > now:
                continue
            episode_id, name = identity
            metadata = self.metadata_factory(now, (end,))
            summary = summarise_microstructure_window(
                symbol=symbol,
                window_start=start,
                window_end=end,
                quotes=tuple(self._quotes.get(symbol, ())),
                trades=tuple(self._trades.get(symbol, ())),
                maximum_quote_age=self.maximum_quote_age,
                minimum_classification_valid_fraction=(
                    self.minimum_trade_classification_valid_fraction
                ),
            )
            level1_valid = bool(self._quotes.get(symbol))
            tick_valid = (
                summary.trade_flow.classification_valid_fraction
                >= self.minimum_trade_classification_valid_fraction
            )
            depth_valid = symbol in self._books and self._books[symbol].snapshot(end).book_valid
            quality_flags = (
                ("data_gap",)
                if self.gap_overlaps(
                    symbol,
                    window_start=start,
                    window_end=end,
                )
                else ()
            )
            if episode_id in self._quiet_observation_ids:
                self.repository.record_quiet_microstructure_summary(
                    metadata,
                    observation_id=episode_id,
                    window_name=name,
                    summary=summary,
                    level1_valid=level1_valid,
                    tick_valid=tick_valid,
                    depth_valid=depth_valid,
                    quality_flags=quality_flags,
                )
                if name.startswith("entry_to_+"):
                    path_payload, path_quality = self._quiet_underlying_path(
                        symbol=symbol,
                        start=start,
                        end=end,
                        quality_flags=quality_flags,
                    )
                    self.repository.record_quiet_underlying_path(
                        metadata,
                        observation_id=episode_id,
                        horizon_label=name.removeprefix("entry_to_+"),
                        target_timestamp_utc=end,
                        payload=path_payload,
                        quality_flags=path_quality,
                    )
            else:
                self.repository.record_microstructure_summary(
                    metadata,
                    episode_id=episode_id,
                    window_name=name,
                    summary=summary,
                    level1_valid=level1_valid,
                    tick_valid=tick_valid,
                    depth_valid=depth_valid,
                    quality_flags=quality_flags,
                    archetype_relationships=compare_frozen_archetypes(
                        actions=self._episode_actions.get(episode_id, {}),
                        summary=summary,
                    ),
                )
                if name == "entry_to_+15m":
                    self._record_opening_reversal_outcome_v1(
                        episode_id=episode_id,
                        symbol=symbol,
                        now=now,
                    )
            self._completed_episode_windows.add(identity)


__all__ = [
    "FrozenM1CLiveRecorder",
    "LivePollResult",
    "ScientificReadiness",
]
