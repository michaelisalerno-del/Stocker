"""Bounded IBKR diagnostic recorder for the frozen prospective universe.

This service records market evidence only. It deliberately marks IBKR-built
bars ineligible for frozen M1 because their volume semantics have not been
shown equivalent to the historical EODHD activity proxy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from stocker_prospective.bars import (
    CompletedBar,
    DiagnosticFiveMinuteBarAggregator,
    assess_bar_for_features,
)
from stocker_prospective.bundle import ANCHOR_COHORT, BundleVerification
from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    UnderlyingContractInput,
    UnderlyingQuoteInput,
)
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.market_data import (
    CallbackRequestError,
    ConnectionState,
    MarketDataBudgetError,
    MarketDataBudgetSnapshot,
    MarketDataType,
)
from stocker_prospective.universe import RegisteredUniverse

FEATURE_SEMANTICS_BLOCKER = "blocked_feature_source_semantics_mismatch"


@dataclass(frozen=True)
class RecorderDeploymentIdentity:
    """Verified universe identity used even when frozen scoring is blocked."""

    model_artifact_id: str
    universe_id: str
    universe_hash: str
    symbols: tuple[str, ...]
    bundle_verified: bool

    @classmethod
    def from_bundle(cls, bundle: BundleVerification) -> RecorderDeploymentIdentity:
        if not bundle.verified:
            raise ValueError("blocked_frozen_artifact_hash_mismatch")
        universe = bundle.manifest.universe
        return cls(
            model_artifact_id=bundle.manifest.bundle_id,
            universe_id=universe.universe_id,
            universe_hash=universe.universe_hash,
            symbols=tuple(universe.symbols),
            bundle_verified=True,
        )

    @classmethod
    def from_registered_universe(
        cls,
        universe: RegisteredUniverse,
    ) -> RecorderDeploymentIdentity:
        return cls(
            model_artifact_id="blocked_missing_verified_frozen_bundle",
            universe_id=universe.universe_id,
            universe_hash=universe.universe_hash,
            symbols=universe.symbols,
            bundle_verified=False,
        )


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


def _qualified_contract(detail: Any) -> Any:
    contract = _attribute(detail, "contract")
    return detail if contract is None else contract


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    return None


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


class IBKRDiagnosticRecorder:
    """Coordinate exact contracts, bounded subscriptions, and evidence writes."""

    def __init__(
        self,
        *,
        config: ProspectiveConfig,
        repository: ProspectiveRepository,
        adapter: IBKRMarketDataAdapter,
        identity: RecorderDeploymentIdentity,
        contract_factory: Callable[[str], Any],
        sleep: Callable[[float], None] = time.sleep,
        heartbeat: Callable[[], object] | None = None,
    ) -> None:
        if len(identity.symbols) != 20 or len(set(identity.symbols)) != 20:
            raise ValueError("blocked_frozen_universe_mismatch")
        self.config = config
        self.repository = repository
        self.adapter = adapter
        self.identity = identity
        self.contract_factory = contract_factory
        self._sleep = sleep
        self._heartbeat = heartbeat
        self._request_interval = 1.1 / config.ibkr.request_rate_per_second
        self._aggregator = DiagnosticFiveMinuteBarAggregator()
        self._contracts: dict[str, Any] = {}
        self._quote_requests: dict[str, int] = {}
        self._bar_requests: dict[str, int] = {}
        self._last_budget_snapshot: MarketDataBudgetSnapshot | None = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _metadata(
        self,
        recorded_at: datetime,
        *,
        source_timestamps: tuple[datetime, ...],
    ) -> EvidenceMetadata:
        return EvidenceMetadata(
            run_id=self.config.runtime.run_id or "",
            prospective_start_utc=self.config.runtime.prospective_start_utc,
            app_version=self.config.runtime.app_version,
            git_commit=self.config.runtime.git_commit,
            model_artifact_id=self.identity.model_artifact_id,
            universe_id=self.identity.universe_id,
            cohort=ANCHOR_COHORT,
            source_timestamps=[value.astimezone(UTC).isoformat() for value in source_timestamps],
            recorded_at_utc=max(
                recorded_at.astimezone(UTC),
                self.config.runtime.prospective_start_utc.astimezone(UTC),
            ),
        )

    def _pace(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat()
        self._sleep(self._request_interval)

    def initialize(self, *, now: datetime) -> bool:
        """Start one exact bounded subscription set after the prospective gate."""

        now = now.astimezone(UTC)
        if self._initialized:
            return True
        if now < self.config.runtime.prospective_start_utc.astimezone(UTC):
            return False
        metadata = self._metadata(now, source_timestamps=(now,))
        self.repository.create_run(metadata, mode=self.config.runtime.mode)
        statuses: dict[str, tuple[str, str | None]] = {}
        for symbol in self.identity.symbols:
            if self._heartbeat is not None:
                self._heartbeat()
            contract, reason = self._qualify_symbol(symbol, metadata)
            if contract is None:
                statuses[symbol] = ("rejected_contract_qualification", reason)
                continue
            self._contracts[symbol] = contract
            statuses[symbol] = self._subscribe_symbol(symbol, contract, metadata)
        symbols = self.identity.symbols
        self.repository.register_universe_membership(
            metadata,
            symbols=symbols,
            operational_status_by_symbol=statuses,
        )
        self.repository.record_data_health_event(
            metadata,
            severity="blocker",
            blocker_code=FEATURE_SEMANTICS_BLOCKER,
            component="feature_parity",
            message=("IBKR diagnostic bars are recorded but are not frozen-M1 feature inputs"),
            details={
                "historical_activity_semantics": "EODHD historical activity proxy",
                "runtime_activity_semantics": (
                    "IBKR realtime-bar trade volume; equivalence not established"
                ),
                "scoring_allowed": False,
            },
        )
        if not self.identity.bundle_verified:
            self.repository.record_data_health_event(
                metadata,
                severity="blocker",
                blocker_code="blocked_missing_verified_frozen_bundle",
                component="frozen_scoring",
                message="record-only diagnostics active; frozen scoring is unavailable",
                details={
                    "bundle_verified": False,
                    "universe_hash": self.identity.universe_hash,
                    "scoring_allowed": False,
                },
            )
        self.repository.record_audit_event(
            metadata,
            event_type="ibkr_diagnostic_recorder_initialized",
            actor=self.config.runtime.instance_id,
            message="bounded market-data-only subscriptions initialized",
            payload={
                "anchor_symbol_count": len(symbols),
                "qualified_symbol_count": len(self._contracts),
                "quote_subscription_count": len(self._quote_requests),
                "bar_subscription_count": len(self._bar_requests),
                "order_path": "absent",
                "bundle_verified": self.identity.bundle_verified,
                "universe_hash": self.identity.universe_hash,
            },
        )
        self._initialized = True
        self._persist_budget(now)
        self._persist_connection_events(now)
        return True

    def _qualify_symbol(
        self,
        symbol: str,
        metadata: EvidenceMetadata,
    ) -> tuple[Any | None, str | None]:
        requested = self.contract_factory(symbol)
        try:
            result = self.adapter.qualify_exact_contract(requested)
            self._pace()
        except (CallbackRequestError, MarketDataBudgetError, RuntimeError) as exc:
            reason = str(exc) or "blocked_ibkr_market_data_subscription"
            self.repository.record_underlying_contract(
                UnderlyingContractInput(
                    metadata=metadata,
                    symbol=symbol,
                    con_id=None,
                    exchange="SMART",
                    currency="USD",
                    local_symbol=None,
                    qualification_status="rejected",
                    rejection_reason=reason,
                )
            )
            self._record_symbol_health(metadata, symbol=symbol, reason=reason)
            return None, reason
        matching: list[Any] = []
        for detail in result.items:
            contract = _qualified_contract(detail)
            if (
                _attribute(contract, "symbol") == symbol
                and _attribute(contract, "secType", "sec_type") == "STK"
                and _attribute(contract, "currency") == "USD"
                and int(_attribute(contract, "conId", "con_id") or 0) > 0
            ):
                matching.append(contract)
        if len(matching) != 1:
            reason = (
                "missing_exact_underlying_contract"
                if not matching
                else "ambiguous_exact_underlying_contract"
            )
            self.repository.record_underlying_contract(
                UnderlyingContractInput(
                    metadata=metadata,
                    symbol=symbol,
                    con_id=None,
                    exchange="SMART",
                    currency="USD",
                    local_symbol=None,
                    qualification_status="rejected",
                    rejection_reason=reason,
                )
            )
            self._record_symbol_health(metadata, symbol=symbol, reason=reason)
            return None, reason
        contract = matching[0]
        self.repository.record_underlying_contract(
            UnderlyingContractInput(
                metadata=metadata,
                symbol=symbol,
                con_id=int(_attribute(contract, "conId", "con_id")),
                exchange=str(_attribute(contract, "exchange") or "SMART"),
                currency=str(_attribute(contract, "currency") or "USD"),
                local_symbol=(
                    None
                    if _attribute(contract, "localSymbol", "local_symbol") is None
                    else str(_attribute(contract, "localSymbol", "local_symbol"))
                ),
                qualification_status="qualified_exact",
                rejection_reason=None,
            )
        )
        return contract, None

    def _subscribe_symbol(
        self,
        symbol: str,
        contract: Any,
        metadata: EvidenceMetadata,
    ) -> tuple[str, str | None]:
        failures: list[str] = []
        try:
            quote_request = self.adapter.request_market_data(
                contract,
                subscription_key=f"underlying:{symbol}:quote",
            )
            self._quote_requests[symbol] = quote_request
            self._pace()
        except (CallbackRequestError, MarketDataBudgetError, RuntimeError) as exc:
            failures.append(str(exc) or "blocked_ibkr_market_data_subscription")
        try:
            bar_request = self.adapter.request_realtime_bars(
                contract,
                subscription_key=f"underlying:{symbol}:bars",
            )
            self._bar_requests[symbol] = bar_request
            self._aggregator.register(
                bar_request,
                symbol=symbol,
                permanent_contract_id=int(_attribute(contract, "conId", "con_id")),
            )
            self._pace()
        except (CallbackRequestError, MarketDataBudgetError, RuntimeError) as exc:
            failures.append(str(exc) or "blocked_ibkr_market_data_subscription")
        if failures:
            reason = ";".join(dict.fromkeys(failures))
            self._record_symbol_health(metadata, symbol=symbol, reason=reason)
            status = (
                "partial_subscription"
                if symbol in self._quote_requests or symbol in self._bar_requests
                else "rejected_subscription"
            )
            return status, reason
        return "recording_diagnostics", None

    def _record_symbol_health(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
        reason: str,
    ) -> None:
        blocker = (
            "blocked_market_data_budget_exhausted"
            if "blocked_market_data_budget_exhausted" in reason
            else "blocked_ibkr_market_data_subscription"
        )
        self.repository.record_data_health_event(
            metadata,
            severity="warning",
            blocker_code=blocker,
            component="ibkr_underlying_subscription",
            message=f"{symbol}: {reason}",
            details={"symbol": symbol, "reason": reason},
        )

    def poll(self, *, now: datetime) -> None:
        """Drain bounded callbacks, persist completed bars, and rebuild if required."""

        now = now.astimezone(UTC)
        if not self.initialize(now=now):
            return
        self._persist_connection_events(now)
        health = self.adapter.connection.health()
        if health.state is ConnectionState.PORT_RESET:
            raise RuntimeError("blocked_ibkr_connection: socket_port_mismatch_or_reset")
        if health.state is ConnectionState.DISCONNECTED:
            self._recover_socket(now)
            health = self.adapter.connection.health()
        if health.subscriptions_require_rebuild and health.state is ConnectionState.CONNECTED:
            self._rebuild_subscriptions(now)
        for update in self.adapter.realtime_bars.drain():
            if update.source_timestamp_utc < self.config.runtime.prospective_start_utc:
                continue
            try:
                completed = self._aggregator.add(update)
            except CallbackRequestError as exc:
                metadata = self._metadata(now, source_timestamps=(update.source_timestamp_utc,))
                self.repository.record_data_health_event(
                    metadata,
                    severity="warning",
                    blocker_code="blocked_ibkr_market_data_subscription",
                    component="ibkr_realtime_bar",
                    message=str(exc),
                    details={"request_id": update.request_id},
                )
                continue
            for bar in completed:
                self._record_bar_and_quote(bar, now=now)
        self._persist_budget(now)

    def _recover_socket(self, now: datetime) -> None:
        attempts = self.config.ibkr.reconnect_max_attempts
        last_error = "blocked_ibkr_connection"
        for attempt in range(1, attempts + 1):
            if self._heartbeat is not None:
                self._heartbeat()
            if attempt > 1:
                self._sleep(self.config.ibkr.reconnect_backoff_seconds * (2 ** (attempt - 2)))
            try:
                self.adapter.reconnect()
            except RuntimeError as exc:
                last_error = str(exc) or last_error
                continue
            self._persist_connection_events(now)
            self._rebuild_subscriptions(now)
            metadata = self._metadata(now, source_timestamps=(now,))
            self.repository.record_audit_event(
                metadata,
                event_type="ibkr_socket_reconnected",
                actor=self.config.runtime.instance_id,
                message="official socket reconnected and subscriptions rebuilt",
                payload={"attempt": attempt, "data_maintained": False},
            )
            return
        raise RuntimeError(f"blocked_ibkr_connection: reconnect_exhausted:{last_error}")

    def _record_bar_and_quote(self, bar: CompletedBar, *, now: datetime) -> None:
        recorded_at = max(now, bar.receive_timestamp_utc, bar.source_timestamp_utc)
        metadata = self._metadata(
            recorded_at,
            source_timestamps=(bar.source_timestamp_utc, bar.receive_timestamp_utc),
        )
        if bar.bar_start_utc < self.config.runtime.prospective_start_utc:
            eligibility = False
            reason = "bar_started_before_prospective_start"
        else:
            assessment = assess_bar_for_features(
                bar,
                maximum_feature_age=timedelta(seconds=15),
                source_semantics_allowed=False,
            )
            eligibility = assessment.eligible
            reason = assessment.rejection_reason or FEATURE_SEMANTICS_BLOCKER
        self.repository.record_underlying_bar(
            metadata,
            bar,
            eligibility=eligibility,
            rejection_reason=reason,
        )
        if bar.complete:
            self._record_underlying_quote(bar, metadata=metadata)

    def _record_underlying_quote(
        self,
        bar: CompletedBar,
        *,
        metadata: EvidenceMetadata,
    ) -> None:
        request_id = self._quote_requests.get(bar.symbol)
        payloads: tuple[dict[str, Any], ...] = ()
        missing_reason: str | None = None
        if request_id is None:
            missing_reason = "missing_underlying_quote_subscription"
        else:
            try:
                payloads = self.adapter.stream_quotes.snapshot(request_id)
            except CallbackRequestError:
                missing_reason = "missing_underlying_quote_subscription"
        fields = {
            str(payload.get("field")): payload
            for payload in payloads
            if payload.get("field") is not None
        }

        def field(name: str) -> float | None:
            payload = fields.get(name)
            return None if payload is None else _number(payload.get("value"))

        receive_times = tuple(
            timestamp
            for timestamp in (
                _timestamp(payload.get("receive_timestamp_utc")) for payload in payloads
            )
            if timestamp is not None
        )
        actual = max(receive_times) if receive_times else None
        lag = None if actual is None else (actual - bar.bar_end_utc).total_seconds()
        executable_receive_times = tuple(
            timestamp
            for name in ("bid", "ask")
            for timestamp in (_timestamp((fields.get(name) or {}).get("receive_timestamp_utc")),)
            if timestamp is not None
        )
        market_data_type = next(
            (
                str(payload["market_data_type"])
                for payload in reversed(payloads)
                if payload.get("market_data_type") is not None
            ),
            None,
        )
        market_data_payload = fields.get("market_data_type")
        if market_data_payload is not None and market_data_payload.get("value") is not None:
            market_data_type = str(market_data_payload["value"])
        allowed_market_data_types = {
            item.value for item in self.adapter.config.allowed_market_data_types
        }
        bid = field("bid")
        ask = field("ask")
        midpoint = None if bid is None or ask is None else (bid + ask) / 2
        spread = None if bid is None or ask is None else ask - bid
        complete = (
            bid is not None
            and ask is not None
            and bid > 0
            and ask > 0
            and spread is not None
            and spread >= 0
        )
        freshness = (
            "missing"
            if len(executable_receive_times) != 2
            else (
                "fresh"
                if all(
                    abs((timestamp - bar.bar_end_utc).total_seconds())
                    <= self.config.ibkr.quote_capture_timeout_seconds
                    for timestamp in executable_receive_times
                )
                else "stale"
            )
        )
        if missing_reason is None and not complete:
            missing_reason = "incomplete_underlying_quote"
        if complete and market_data_type is None:
            missing_reason = "unconfirmed_market_data_type"
        elif complete and market_data_type not in allowed_market_data_types:
            missing_reason = "market_data_type_not_allowed"
        elif complete and market_data_type != MarketDataType.LIVE.value:
            missing_reason = "blocked_non_live_market_data"
        if complete and freshness == "stale":
            missing_reason = "stale_underlying_quote"
        self.repository.record_underlying_quote(
            UnderlyingQuoteInput(
                metadata=metadata,
                symbol=bar.symbol,
                con_id=bar.permanent_contract_id,
                target_timestamp_utc=bar.bar_end_utc,
                actual_quote_timestamp_utc=actual,
                capture_lag_seconds=lag,
                bid=bid,
                ask=ask,
                bid_size=field("bid_size"),
                ask_size=field("ask_size"),
                last=field("last"),
                last_size=field("last_size"),
                midpoint=midpoint,
                spread=spread,
                provider_timestamp_utc=None,
                receive_timestamp_utc=actual,
                market_data_type=market_data_type,
                freshness=freshness,
                completeness="complete" if complete else "partial",
                capture_status=(
                    "recorded_live"
                    if complete
                    and market_data_type == MarketDataType.LIVE.value
                    and market_data_type in allowed_market_data_types
                    else "recorded_diagnostic"
                    if complete
                    else "missed"
                ),
                missing_quote_reason=missing_reason,
            )
        )

    def _persist_connection_events(self, now: datetime) -> None:
        for event in self.adapter.connection.drain_events():
            if event.recorded_at < self.config.runtime.prospective_start_utc:
                continue
            metadata = self._metadata(
                max(now, event.recorded_at),
                source_timestamps=(event.recorded_at,),
            )
            self.repository.record_ibkr_connection_event(
                metadata,
                state=event.state.value,
                error_code=event.code,
                message=event.message,
                data_maintained=event.data_maintained,
                reconnect_attempt=None,
                details={
                    "source": "official_ibkr_callback",
                    "event_kind": event.event_kind.value,
                },
            )

    def _persist_budget(self, now: datetime) -> None:
        snapshot = self.adapter.budget.snapshot()
        if snapshot == self._last_budget_snapshot:
            return
        metadata = self._metadata(now, source_timestamps=(now,))
        self.repository.record_market_data_budget_event(metadata, snapshot)
        self._last_budget_snapshot = snapshot

    def _rebuild_subscriptions(self, now: datetime) -> None:
        for bar in self._aggregator.flush(completed_through_utc=now):
            self._record_bar_and_quote(bar, now=now)
        discarded = self.adapter.realtime_bars.drain()
        self._aggregator = DiagnosticFiveMinuteBarAggregator()
        self._quote_requests.clear()
        self._bar_requests.clear()
        metadata = self._metadata(now, source_timestamps=(now,))
        if discarded:
            self.repository.record_data_health_event(
                metadata,
                severity="warning",
                blocker_code="blocked_ibkr_market_data_subscription",
                component="ibkr_reconnect",
                message="buffered callbacks discarded after confirmed data-loss reconnect",
                details={"discarded_realtime_bar_callbacks": len(discarded)},
            )
        for symbol, contract in self._contracts.items():
            self._subscribe_symbol(symbol, contract, metadata)
        self.adapter.connection.subscriptions_rebuilt()
        self.repository.record_audit_event(
            metadata,
            event_type="ibkr_subscriptions_rebuilt",
            actor=self.config.runtime.instance_id,
            message="lost-data reconnect rebuilt bounded subscriptions",
            payload={
                "quote_subscription_count": len(self._quote_requests),
                "bar_subscription_count": len(self._bar_requests),
            },
        )

    def shutdown(self, *, now: datetime) -> None:
        """Persist incomplete bars as missed and cancel only owned subscriptions."""

        if not self._initialized:
            return
        for bar in self._aggregator.flush(completed_through_utc=now):
            self._record_bar_and_quote(bar, now=now.astimezone(UTC))
        for symbol, request_id in tuple(self._quote_requests.items()):
            self.adapter.cancel_market_data(
                request_id,
                subscription_key=f"underlying:{symbol}:quote",
            )
        for symbol, request_id in tuple(self._bar_requests.items()):
            self.adapter.cancel_realtime_bars(
                request_id,
                subscription_key=f"underlying:{symbol}:bars",
            )
        self._quote_requests.clear()
        self._bar_requests.clear()
        self._persist_budget(now.astimezone(UTC))
        timestamp = now.astimezone(UTC)
        metadata = self._metadata(timestamp, source_timestamps=(timestamp,))
        self.repository.record_audit_event(
            metadata,
            event_type="ibkr_diagnostic_recorder_stopped",
            actor=self.config.runtime.instance_id,
            message="market-data subscriptions cancelled",
            payload={"order_path": "absent"},
        )
