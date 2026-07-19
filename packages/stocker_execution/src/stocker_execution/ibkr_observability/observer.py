"""High-level local-only contract and top-of-book observer."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from stocker_execution.ibkr_observability.config import IBKRObserverConfig
from stocker_execution.ibkr_observability.models import (
    ContractIdentity,
    ContractRequest,
    MarketDataType,
    ObservationClassification,
    ObservationPlanItem,
    QuoteObservationRecord,
    QuoteSnapshot,
    ReferenceQuoteUncertainty,
)
from stocker_execution.ibkr_observability.protocol import IBKRObservabilityClient

COLLECTOR_VERSION = "observable_event_ranking_v1_ibkr_observer"
COLLECTOR_HASH = hashlib.sha256(COLLECTOR_VERSION.encode()).hexdigest()


def classify_snapshot(
    snapshot: QuoteSnapshot, *, maximum_delay_seconds: float = 10.0
) -> ObservationClassification:
    """Classify quote observability without making any execution or fill claim."""

    if snapshot.error_code is not None or snapshot.error_message:
        return ObservationClassification.ERROR
    if snapshot.market_data_type is MarketDataType.FROZEN:
        return ObservationClassification.FROZEN_NON_CURRENT
    if snapshot.market_data_type in {MarketDataType.DELAYED, MarketDataType.DELAYED_FROZEN}:
        return ObservationClassification.DELAYED_NON_EXECUTABLE
    if snapshot.first_response_timestamp is None:
        return ObservationClassification.UNAVAILABLE
    response_delay = (
        snapshot.first_response_timestamp - snapshot.requested_timestamp
    ).total_seconds()
    if response_delay < 0.0 or response_delay > maximum_delay_seconds:
        return ObservationClassification.STALE
    if snapshot.market_data_type is not MarketDataType.LIVE:
        return ObservationClassification.UNAVAILABLE
    if not snapshot.snapshot_complete or snapshot.snapshot_completion_timestamp is None:
        return ObservationClassification.LIVE_PARTIAL_QUOTE
    completion_delay = (
        snapshot.snapshot_completion_timestamp - snapshot.requested_timestamp
    ).total_seconds()
    if completion_delay < response_delay or completion_delay > maximum_delay_seconds:
        return ObservationClassification.STALE
    if snapshot.bid is None or snapshot.ask is None:
        return ObservationClassification.LIVE_PARTIAL_QUOTE
    if snapshot.bid <= 0.0 or snapshot.ask <= 0.0 or snapshot.ask < snapshot.bid:
        return ObservationClassification.ERROR
    return ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED


class IBKRObserver:
    """Read-only observer composed with, and never subclassing, an order-capable broker."""

    def __init__(
        self,
        *,
        client: IBKRObservabilityClient,
        config: IBKRObserverConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._in_flight = asyncio.Semaphore(config.maximum_in_flight_requests)
        self._pacing_lock = asyncio.Lock()
        self._last_request_started: float | None = None

    async def _wait_for_request_slot(self) -> None:
        """Apply the versioned conservative request-rate limit."""

        minimum_interval = 1.0 / self._config.maximum_requests_per_second
        async with self._pacing_lock:
            now = time.monotonic()
            if self._last_request_started is not None:
                delay = minimum_interval - (now - self._last_request_started)
                if delay > 0.0:
                    await asyncio.sleep(delay)
            self._last_request_started = time.monotonic()

    async def _cancel_observer_subscription(self, request_id: int) -> None:
        async with self._in_flight:
            await self._wait_for_request_slot()
            await self._client.cancel_market_data(request_id)

    async def connect(self) -> None:
        """Connect only when explicitly enabled and configured for localhost."""

        if not self._config.enabled:
            raise RuntimeError("IBKR observability is disabled; explicit invocation is required")
        if not self._config.require_tws_read_only_api_mode:
            raise RuntimeError("TWS/IB Gateway read-only API mode is required")
        await self._client.connect(
            host=self._config.host,
            port=self._config.port,
            client_id=self._config.client_id,
        )

    async def disconnect(self) -> None:
        """Disconnect the observation-only client."""

        await self._client.disconnect()

    async def request_server_time(self) -> datetime:
        """Request IBKR server time without touching account state."""

        async with self._in_flight:
            await self._wait_for_request_slot()
            return await self._client.request_server_time()

    async def resolve_contract(self, request: ContractRequest) -> ContractIdentity:
        """Resolve current execution-feasibility identity only."""

        async with self._in_flight:
            await self._wait_for_request_slot()
            return await self._client.resolve_stock_contract(request)

    async def capture_quote(
        self,
        *,
        plan: ObservationPlanItem,
        contract: ContractIdentity,
        request_id: int,
    ) -> QuoteObservationRecord:
        """Capture and always cancel one bounded snapshot subscription."""

        planned = plan.planned_timestamp
        if planned.tzinfo is None:
            raise ValueError("planned observation timestamp must be timezone-aware")
        deadline = planned + timedelta(seconds=plan.maximum_collection_delay_seconds)
        invoked_at = self._clock()
        if invoked_at.tzinfo is None:
            raise ValueError("observer clock must return a timezone-aware timestamp")
        snapshot: QuoteSnapshot | None = None
        errors: list[str] = []
        server_time: datetime | None = None
        subscription_attempted = False
        cancellation_failed = False
        if invoked_at < planned or invoked_at > deadline:
            snapshot = QuoteSnapshot(
                request_id=request_id,
                requested_timestamp=planned,
                server_time_observation=None,
                local_send_timestamp=invoked_at,
                first_response_timestamp=None,
                snapshot_completion_timestamp=None,
                bid=None,
                ask=None,
                bid_size=None,
                ask_size=None,
                last=None,
                last_size=None,
                market_data_type=None,
                snapshot_complete=False,
                error_code=None,
                error_message=None,
                connection_status="not_requested_outside_frozen_window",
            )
        else:
            remaining = (deadline - self._clock()).total_seconds()
            try:
                server_time = await asyncio.wait_for(
                    self.request_server_time(),
                    timeout=max(0.001, remaining),
                )
            except Exception as exc:
                errors.append(f"server_time:{type(exc).__name__}:{exc}")
            for attempt in range(self._config.bounded_retries + 1):
                remaining = (deadline - self._clock()).total_seconds()
                if server_time is None or remaining <= 0.0:
                    break
                timeout = min(self._config.request_timeout_seconds, remaining)
                try:
                    async with self._in_flight:
                        await self._wait_for_request_slot()
                        subscription_attempted = True
                        snapshot = await asyncio.wait_for(
                            self._client.capture_top_of_book_snapshot(
                                request_id=request_id,
                                contract=contract,
                                timeout_seconds=timeout,
                            ),
                            timeout=timeout,
                        )
                    break
                except Exception as exc:
                    errors.append(f"attempt_{attempt + 1}:{type(exc).__name__}:{exc}")
                finally:
                    if subscription_attempted:
                        try:
                            await self._cancel_observer_subscription(request_id)
                        except Exception as exc:
                            cancellation_failed = True
                            errors.append(f"cancel:{type(exc).__name__}:{exc}")
        if snapshot is None:
            now = self._clock()
            snapshot = QuoteSnapshot(
                request_id=request_id,
                requested_timestamp=plan.planned_timestamp,
                server_time_observation=server_time,
                local_send_timestamp=now,
                first_response_timestamp=None,
                snapshot_completion_timestamp=now,
                bid=None,
                ask=None,
                bid_size=None,
                ask_size=None,
                last=None,
                last_size=None,
                market_data_type=None,
                snapshot_complete=False,
                error_code=None,
                error_message="|".join(errors),
                connection_status="error",
            )
        classification = classify_snapshot(
            snapshot,
            maximum_delay_seconds=plan.maximum_collection_delay_seconds,
        )
        if (
            classification is ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED
            and snapshot.snapshot_completion_timestamp is not None
        ):
            plan_delay = (snapshot.snapshot_completion_timestamp - planned).total_seconds()
            send_delay = (snapshot.local_send_timestamp - planned).total_seconds()
            if (
                plan_delay < 0.0
                or plan_delay > plan.maximum_collection_delay_seconds
                or send_delay < 0.0
                or send_delay > plan.maximum_collection_delay_seconds
            ):
                classification = ObservationClassification.UNAVAILABLE
        if cancellation_failed:
            classification = ObservationClassification.ERROR
        timing_uncertainty = (
            None
            if snapshot.snapshot_completion_timestamp is None
            else (snapshot.snapshot_completion_timestamp - plan.planned_timestamp).total_seconds()
        )
        reference_uncertainty = (
            ReferenceQuoteUncertainty.EXACT_REFERENCE_QUOTE_OBSERVED
            if classification is ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED
            else ReferenceQuoteUncertainty.UNAVAILABLE
        )
        if not subscription_attempted:
            subscription_status = snapshot.connection_status
        elif cancellation_failed:
            subscription_status = "snapshot_cancellation_failed"
        elif classification is ObservationClassification.ERROR:
            subscription_status = "snapshot_cancelled_after_error"
        else:
            subscription_status = "snapshot_cancelled_after_capture"
        error_message = (
            "|".join(message for message in (snapshot.error_message, *errors) if message) or None
        )
        return QuoteObservationRecord(
            observation_id=plan.observation_id,
            event_id=plan.event_id,
            decision_id=plan.decision_id,
            request_id=request_id,
            requested_timestamp=plan.planned_timestamp,
            ibkr_server_time_observation=snapshot.server_time_observation,
            local_send_timestamp_utc=snapshot.local_send_timestamp,
            first_response_timestamp_utc=snapshot.first_response_timestamp,
            snapshot_completion_timestamp_utc=snapshot.snapshot_completion_timestamp,
            symbol=plan.symbol,
            con_id=plan.con_id,
            exchange=contract.routing_exchange,
            primary_exchange=contract.primary_exchange,
            bid=snapshot.bid,
            ask=snapshot.ask,
            bid_size=snapshot.bid_size,
            ask_size=snapshot.ask_size,
            last=snapshot.last,
            last_size=snapshot.last_size,
            market_data_type=snapshot.market_data_type,
            classification=classification,
            quote_age_or_timing_uncertainty_seconds=timing_uncertainty,
            subscription_status=subscription_status,
            snapshot_complete=snapshot.snapshot_complete,
            error_code=snapshot.error_code,
            error_message=error_message,
            connection_status=snapshot.connection_status,
            api_tws_version=self._client.api_tws_version,
            source_identifier="IBKR_TWS_TOP_OF_BOOK",
            collector_version=COLLECTOR_VERSION,
            collector_hash=COLLECTOR_HASH,
            reference_uncertainty=reference_uncertainty,
        )
