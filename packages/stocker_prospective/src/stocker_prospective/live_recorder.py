"""Activation-bounded live coordinator for frozen M1C and raw market evidence."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.direction_features import DirectionFeatureBar
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
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
)
from stocker_prospective.m1c_features import (
    FROZEN_CHECKPOINTS,
    HistoricalActivityBaseline,
    LiveFeatureBar,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.microstructure import (
    STANDARD_WINDOWS,
    compare_frozen_archetypes,
    episode_relative_windows,
    standard_window_summaries,
    summarise_microstructure_window,
)
from stocker_prospective.order_book import DepthBook
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
    RecorderCheckpointResult,
)

MetadataFactory = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]
GroupOProvider = Callable[[str, date], FrozenGroupOContext]
OptionQuoteSink = Callable[[EvidenceMetadata, OptionQuoteEvent], None]
EpisodeCallback = Callable[[RecorderCheckpointResult], None]


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
    blocked_checkpoints: dict[str, str]
    checkpoint_results: tuple[RecorderCheckpointResult, ...]
    depth_reset_symbols: tuple[str, ...] = ()
    ibkr_errors: tuple[tuple[int, int], ...] = ()
    broker_mutations: int = 0


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
        self.adapter = adapter
        self.normalizer = normalizer
        self.raw_store = raw_store
        self.repository = repository
        self.engine = engine
        self.activity_baseline = activity_baseline
        self.group_o_provider = group_o_provider
        self.metadata_factory = metadata_factory
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
        self._finalizer = KeepUpToDateBarFinalizer(
            prospective_collection_start=normalizer.prospective_collection_start
        )
        self._bar_adapter = AuditedFiveMinuteBarAdapter()
        self._bars: dict[tuple[str, date], dict[int, AuditedLiveBar]] = {}
        self._processed: set[tuple[str, date, int]] = set()
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
        self._quiet_observation_ids: set[str] = set()
        self._completed_episode_windows: set[tuple[str, str]] = set()
        self._gap_symbols: set[str] = set()
        self._gap_intervals: dict[
            str,
            list[tuple[datetime, datetime | None]],
        ] = {}
        self._capability_preflight_passed = readiness.capability_preflight_passed
        self._scientific_prerequisites_passed = (
            readiness.m1c_parity_passed
            and readiness.direction_parity_passed
            and readiness.bar_compatibility_passed
            and readiness.historical_activity_baseline_available
            and readiness.clock_drift_within_tolerance
        )
        self._scientific_scoring_enabled = (
            readiness.capability_preflight_passed and self._scientific_prerequisites_passed
        )
        self._clock_drift_seconds: float | None = None
        self._depth_exchanges: tuple[str, ...] = ()

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

    def mark_gap(self, symbol: str, *, started_at: datetime) -> None:
        if symbol in self.universe_symbols or symbol in self.context_proxy_symbols:
            observed = started_at.astimezone(UTC)
            self._gap_symbols.add(symbol)
            intervals = self._gap_intervals.setdefault(symbol, [])
            if not intervals or intervals[-1][1] is not None:
                intervals.append((observed, None))

    def clear_gap_after_complete_bar(self, symbol: str, *, completed_at: datetime) -> None:
        intervals = self._gap_intervals.get(symbol)
        if intervals and intervals[-1][1] is None:
            started, _ = intervals[-1]
            intervals[-1] = (started, max(started, completed_at.astimezone(UTC)))
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
        self._scientific_scoring_enabled = passed and self._scientific_prerequisites_passed

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
    def clock_drift_seconds(self) -> float | None:
        return self._clock_drift_seconds

    @property
    def depth_exchanges(self) -> tuple[str, ...]:
        return self._depth_exchanges

    def _persist_raw(
        self,
        metadata: EvidenceMetadata,
        events: tuple[RawEvent, ...],
    ) -> tuple[str, ...]:
        if not events:
            return ()
        partitions = self.raw_store.write_grouped(
            data_source="ibkr",
            events=events,
            complete=True,
            gap_count=len({event.symbol for event in events if event.symbol in self._gap_symbols}),
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
        return tuple(item.content_hash for item in partitions)

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
                self._bars.setdefault((bar.symbol, bar.session), {})[bar.checkpoint] = bar
                self.clear_gap_after_complete_bar(
                    bar.symbol,
                    completed_at=bar.bar_end_utc,
                )

    def poll(self, *, now: datetime) -> LivePollResult:
        observed_now = now.astimezone(UTC)
        raw_events: list[RawEvent] = []
        finalised_bars: list[AuditedLiveBar] = []
        depth_reset_symbols: set[str] = set()
        ibkr_errors: list[tuple[int, int]] = []
        callbacks = self.adapter.drain_stream_events()
        for payload in callbacks:
            normalized = self.normalizer.normalize(payload)
            if normalized is None:
                continue
            if normalized.raw_event is not None:
                raw_events.append(normalized.raw_event)
                derived = self._retain(normalized.raw_event)
                if derived is not None:
                    raw_events.append(derived)
            if normalized.historical_bar is not None:
                source_sequence = int(payload["source_sequence"])
                received_monotonic_ns = int(payload["received_monotonic_ns"])
                self._consume_bar(
                    normalized.historical_bar.request_id,
                    normalized.historical_bar.bar_start_utc,
                    source_sequence,
                    received_monotonic_ns,
                    raw_events,
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
                self.mark_gap(symbol, started_at=started_at)
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
                code = int(payload.get("error_code", -1))
                ibkr_errors.append((request_id, code))
                owner = self.normalizer.owner(request_id)
                if owner is not None:
                    self.mark_gap(owner.symbol, started_at=observed_now)
        source_times = tuple(event.ordering_timestamp for event in raw_events)
        metadata = self.metadata_factory(
            max(
                (event.received_timestamp_utc for event in raw_events),
                default=observed_now,
            ),
            source_times or (observed_now,),
        )
        partition_hashes = self._persist_raw(metadata, tuple(raw_events))
        for event in raw_events:
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
        results = self._score_ready()
        self._record_due_episode_windows(observed_now)
        self._trim_history(observed_now)
        return LivePollResult(
            callback_count=len(callbacks),
            raw_event_count=len(raw_events),
            finalised_bar_count=len(finalised_bars),
            checkpoint_count=len(results),
            fresh_episode_count=sum(result.episode_decision.fresh_episode for result in results),
            partition_hashes=partition_hashes,
            blocked_checkpoints={
                f"{symbol}:{session.isoformat()}:{checkpoint}": reason
                for (symbol, session, checkpoint), reason in sorted(self._blocked.items())
            },
            checkpoint_results=results,
            depth_reset_symbols=tuple(sorted(depth_reset_symbols)),
            ibkr_errors=tuple(ibkr_errors),
        )

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

    def _score_ready(self) -> tuple[RecorderCheckpointResult, ...]:
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
                        metadata = self.metadata_factory(
                            feature_bars[-1].bar_complete_timestamp,
                            (
                                feature_bars[-1].bar_complete_timestamp,
                                latest.ordering_timestamp
                                if latest is not None
                                else feature_bars[-1].bar_complete_timestamp,
                            ),
                        )
                        self.repository.record_group_o_context(metadata, context)
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
                            )
                        )
                    except (KeyError, ValueError) as exc:
                        self._blocked[key] = str(exc)
                        self._processed.add(key)
                        continue
                    self._processed.add(key)
                    self._blocked.pop(key, None)
                    results.append(result)
                    self._record_standard_windows(
                        symbol=symbol,
                        as_of=feature_bars[-1].bar_complete_timestamp,
                        metadata=metadata,
                    )
                    if result.episode_decision.fresh_episode:
                        self._arm_episode_windows(result)
                        if self.episode_callback is not None:
                            self.episode_callback(result)
                    if (
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
            self._completed_episode_windows.add(identity)


__all__ = [
    "FrozenM1CLiveRecorder",
    "LivePollResult",
    "ScientificReadiness",
]
