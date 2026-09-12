"""Opt-in, bounded evidence recorder. Never supplies production TradeEvents.

TBT streams belong to the caller: this observer never cancels them. The standalone
operator owns its dedicated read-only connection and disconnects it after export.
Ordinary subscriptions use the adapter's existing reference-counted resource seam.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from time import monotonic_ns
from typing import Any

from stocker_core.runs import Environment
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument, _to_ib_contract

REFERENCE = "REFERENCE_TBT"
ORDINARY = "ORDINARY_TRADE_STREAM"
GENERIC_TICKS = "375"


class DualFeedConnection(IbkrConnection):
    """Diagnostic-only request error routing; all production feed methods inherited.

    The ordinary request's rejection must not invalidate a separately owned TBT
    prefix. Preserve normal resource/error counters, omitting conId only for our
    known ordinary requests. All TBT and non-diagnostic errors retain base behavior.
    IDs remain known until the connection epoch changes, including after release,
    because IBKR can deliver a late rejection for an already cancelled request.
    """

    _diagnostic_ordinary_requests: set[tuple[int, int]]

    def _record_resource_error(
        self, request_id: int, code: int, message: str, contract: object, *extra: object
    ) -> None:
        ordinary = (self.connection_epoch, request_id) in getattr(
            self, "_diagnostic_ordinary_requests", ()
        )
        super()._record_resource_error(
            request_id,
            code,
            message,
            None if ordinary else contract,
            *extra,
        )


@dataclass(frozen=True)
class TapeEvent:
    con_id: int
    symbol: str
    feed: str
    event_at: datetime | None
    received_at: datetime
    price: float | None
    size: float | None
    sequence: int
    received_monotonic_ns: int
    market_data_type: int | None
    connection_epoch: int
    subscription_created_at: datetime
    raw: dict[str, Any]
    invalid_reason: str = ""
    broker_at: datetime | None = None


@dataclass
class StreamEvidence:
    con_id: int
    symbol: str
    feed: str
    requested_at: datetime
    subscribed_at: datetime | None = None
    recording_started_at: datetime | None = None
    released_at: datetime | None = None
    rejection: str = ""
    request_id: int | None = None
    connection_epoch: int = 0
    errors: list[str] = field(default_factory=list)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (ValueError, TypeError):
        return None


def ordinary_fields(value: str) -> tuple[datetime | None, float | None, float | None, str]:
    """Fixed conversion: each raw 77 callback is one observation, never expanded/deduped."""
    fields = value.split(";")
    if len(fields) != 6:
        return None, None, None, "MALFORMED_77_PAYLOAD"
    price, size = _number(fields[0]), _number(fields[1])
    timestamp = None
    try:
        if fields[2] and int(fields[2]) > 0:
            timestamp = datetime.fromtimestamp(int(fields[2]) / 1000, UTC)
    except (ValueError, OverflowError, OSError):
        pass
    reason = ""
    if price is None or price <= 0:
        reason = "NO_VALID_PRICE"
    elif timestamp is None:
        reason = "NO_BROKER_TIMESTAMP"
    elif fields[1] and (size is None or size < 0):
        reason = "INVALID_SIZE"
    return timestamp, price, size, reason


class DualFeedRecorder:
    """One recorder per read-only PAPER connection; all callbacks restored on close.

    Attach reference streams immediately after prepare_trade_events, before T0.
    No backfill of earlier in-memory TBT: local receipt provenance is unavailable.
    """

    def __init__(
        self,
        broker: DualFeedConnection,
        *,
        max_events: int = 250_000,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not isinstance(broker, DualFeedConnection):
            raise ValueError("Recorder requires its dedicated diagnostic connection")
        if broker.environment is not Environment.PAPER or broker._execution_enabled:
            raise ValueError("Diagnostic requires read-only PAPER; execution must be disabled")
        broker._require_connected()
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.broker = broker
        self.clock = clock
        self.epoch = broker.connection_epoch
        broker._diagnostic_ordinary_requests = {
            key
            for key in getattr(broker, "_diagnostic_ordinary_requests", ())
            if key[0] == self.epoch
        }
        self.max_events = max_events
        self.events: list[TapeEvent] = []
        self.streams: dict[tuple[int, str], StreamEvidence] = {}
        self.dropped_events = 0
        self.closed = False
        self.integrity_errors: list[str] = []
        self._leases: list[tuple[int, str, str, str, int]] = []
        self._listeners: list[tuple[Any, Any]] = []
        self._wrapper: Any = getattr(broker._client, "wrapper", None)
        if self._wrapper is None or not callable(getattr(self._wrapper, "tickString", None)):
            raise ValueError("Explicit ib_async tickString callback unavailable")
        if getattr(self._wrapper, "_stocker_dual_feed", False):
            raise ValueError("Only one diagnostic recorder per connection")
        self._previous_string = self._wrapper.tickString
        self._previous_tbt = self._wrapper.tickByTickAllLast
        self._tbt_broker_times: dict[int, int] = {}
        self._tbt_hook = self._receive_tbt
        self._wrapper.tickByTickAllLast = self._tbt_hook
        self._string_hook = self._receive_string
        self._wrapper.tickString = self._string_hook
        self._wrapper._stocker_dual_feed = True
        self._error_event = getattr(broker._client, "errorEvent", None)
        if self._error_event is not None:
            self._error_event += self._receive_error

    def _check(self) -> None:
        if self.closed:
            raise ValueError("Recorder closed")
        if not self.broker.is_connected or self.broker.connection_epoch != self.epoch:
            self.integrity_errors.append("CONNECTION_EPOCH_CHANGED_OR_DISCONNECTED")
            raise IbkrError(self.integrity_errors[-1])

    def attach_reference(self, instrument: QualifiedInstrument) -> StreamEvidence:
        self._check()
        key = instrument.con_id, REFERENCE
        if key in self.streams:
            return self.streams[key]
        evidence = StreamEvidence(instrument.con_id, instrument.symbol, REFERENCE, self.clock())
        self.streams[key] = evidence
        stream = getattr(self.broker, "_causal_trade_streams", {}).get(instrument.con_id)
        if stream is None or stream[0] != self.epoch:
            evidence.rejection = "REFERENCE_NOT_PREPARED"
            return evidence
        evidence.subscribed_at = stream[1]
        evidence.recording_started_at = self.clock()
        evidence.connection_epoch = self.epoch
        ticker = stream[4]
        evidence.request_id = self._wrapper.ticker2ReqId["Last"].get(ticker)

        def receive(updated: Any) -> None:
            if self.closed:
                return
            for tick in updated.tickByTicks:
                self._append(
                    evidence,
                    tick.time,
                    _number(tick.price),
                    _number(getattr(tick, "size", None)),
                    getattr(updated, "marketDataType", None),
                    {
                        "tick_type": getattr(tick, "tickType", None),
                        "exchange": getattr(tick, "exchange", None),
                        "special_conditions": getattr(tick, "specialConditions", None),
                        "unreported": getattr(
                            getattr(tick, "tickAttribLast", None), "unreported", None
                        ),
                        "past_limit": getattr(
                            getattr(tick, "tickAttribLast", None), "pastLimit", None
                        ),
                        "event_timestamp_semantics": "ib_async_local_packet_receive",
                        "broker_epoch_seconds": self._tbt_broker_times.pop(id(tick), None),
                    },
                )

        ticker.updateEvent += receive
        self._listeners.append((ticker, receive))
        return evidence

    def _receive_tbt(self, *args: Any) -> Any:
        result = self._previous_tbt(*args)
        ticker = self._wrapper.reqId2Ticker.get(args[0])
        if (
            ticker is not None
            and ticker.tickByTicks
            and any(s.feed == REFERENCE and s.request_id == args[0] for s in self.streams.values())
        ):
            self._tbt_broker_times[id(ticker.tickByTicks[-1])] = args[2]
        return result

    def acquire_ordinary(self, instrument: QualifiedInstrument) -> StreamEvidence:
        self._check()
        key = instrument.con_id, ORDINARY
        if key in self.streams:
            return self.streams[key]
        evidence = StreamEvidence(instrument.con_id, instrument.symbol, ORDINARY, self.clock())
        self.streams[key] = evidence
        # ib_async shares a Ticker by contract and one cancelMktData mapping. A
        # second, incompatible generic-tick request could steal another owner.
        # Do not upgrade/restart it or pretend a 236/quote request supplies 375.
        desired = (
            instrument.con_id,
            instrument.security_type.upper(),
            instrument.exchange.upper(),
            GENERIC_TICKS,
            1,
        )
        for active_key in self.broker._active_market_data:
            if active_key[0] == instrument.con_id and active_key != desired:
                evidence.rejection = "INCOMPATIBLE_EXISTING_MARKET_DATA_SUBSCRIPTION"
                return evidence
        try:
            lease, ticker = self.broker._acquire_market_data_stream(
                _to_ib_contract(instrument),
                generic_tick_list=GENERIC_TICKS,
                market_data_type=1,
                purpose="DUAL_FEED_RESEARCH_ONLY",
            )
        except IbkrError as exc:
            evidence.rejection = str(exc)
            return evidence
        self._leases.append(lease)
        active = self.broker._active_market_data[lease]
        evidence.subscribed_at = active.created_at
        evidence.recording_started_at = self.clock()
        evidence.connection_epoch = self.epoch
        # Resolve by subscription type, not first reqId2Ticker match: TBT and
        # reqMktData share the same ticker in the real SDK.
        evidence.request_id = self._wrapper.ticker2ReqId["mktData"].get(ticker)
        if evidence.request_id is None:
            evidence.errors.append("ORDINARY_REQUEST_ID_UNAVAILABLE")
        else:
            self.broker._diagnostic_ordinary_requests.add((self.epoch, evidence.request_id))
        return evidence

    def _receive_string(self, request_id: int, tick_type: int, value: str) -> Any:
        if tick_type == 77 and not self.closed:
            for evidence in self.streams.values():
                if evidence.feed == ORDINARY and evidence.request_id == request_id:
                    timestamp, price, size, reason = ordinary_fields(value)
                    ticker = self._wrapper.reqId2Ticker.get(request_id)
                    self._append(
                        evidence,
                        timestamp,
                        price,
                        size,
                        getattr(ticker, "marketDataType", None),
                        {
                            "tick_type": 77,
                            "generic_ticks": GENERIC_TICKS,
                            "payload": value,
                            "timestamp_resolution": "milliseconds",
                        },
                        reason,
                    )
                    break
        return self._previous_string(request_id, tick_type, value)

    def _append(
        self,
        evidence: StreamEvidence,
        timestamp: datetime | None,
        price: float | None,
        size: float | None,
        data_type: int | None,
        raw: dict[str, Any],
        reason: str = "",
    ) -> None:
        if self.broker.connection_epoch != self.epoch or not self.broker.is_connected:
            if "CONNECTION_EPOCH_CHANGED_OR_DISCONNECTED" not in self.integrity_errors:
                self.integrity_errors.append("CONNECTION_EPOCH_CHANGED_OR_DISCONNECTED")
            return
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        assert evidence.subscribed_at is not None
        # ib_async.lastTime is local packet receipt time for both callbacks.
        received = getattr(self._wrapper, "lastTime", None) or self.clock()
        if timestamp is None or timestamp.tzinfo is None or price is None or price <= 0:
            reason = reason or "INVALID_EVENT"
        self.events.append(
            TapeEvent(
                evidence.con_id,
                evidence.symbol,
                evidence.feed,
                timestamp,
                received,
                price,
                size,
                len(self.events) + 1,
                monotonic_ns(),
                data_type,
                self.epoch,
                evidence.subscribed_at,
                raw,
                reason,
                datetime.fromtimestamp(raw["broker_epoch_seconds"], UTC)
                if raw.get("broker_epoch_seconds") is not None
                else timestamp
                if evidence.feed == ORDINARY
                else None,
            )
        )

    def _receive_error(
        self, request_id: int, code: int, message: str, contract: Any, *extra: Any
    ) -> None:
        # Preserve code/request identity, never account-bearing free text.
        for evidence in self.streams.values():
            if evidence.request_id == request_id or (
                getattr(contract, "conId", None) == evidence.con_id
                and (code == 10190 or request_id < 0)
            ):
                evidence.errors.append(f"IBKR_ERROR:{code}:request={request_id}")
        if code in {1100, 1101, 1102, 1300}:
            self.integrity_errors.append(f"BROKER_CONNECTIVITY:{code}")

    def close(self) -> None:
        if self.closed:
            return
        if self.broker.connection_epoch != self.epoch or not self.broker.is_connected:
            self.integrity_errors.append("CONNECTION_EPOCH_CHANGED_OR_DISCONNECTED")
        self.closed = True
        for ticker, callback in self._listeners:
            ticker.updateEvent -= callback
        if self._wrapper.tickString is self._string_hook:
            self._wrapper.tickString = self._previous_string
        else:
            self.integrity_errors.append("CALLBACK_REPLACED_DURING_RECORDING")
        if self._wrapper.tickByTickAllLast is self._tbt_hook:
            self._wrapper.tickByTickAllLast = self._previous_tbt
        else:
            self.integrity_errors.append("TBT_CALLBACK_REPLACED_DURING_RECORDING")
        self._wrapper._stocker_dual_feed = False
        if self._error_event is not None:
            self._error_event -= self._receive_error
        for lease in self._leases:
            # A reconnected stream with the same key belongs to a new epoch.
            if self.broker.connection_epoch == self.epoch:
                try:
                    self.broker._release_market_data_stream(lease)
                except Exception as exc:
                    self.integrity_errors.append(f"RELEASE_FAILED:{type(exc).__name__}")
        for evidence in self.streams.values():
            if evidence.subscribed_at is not None:
                evidence.released_at = self.clock()
