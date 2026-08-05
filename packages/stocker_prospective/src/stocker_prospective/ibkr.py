"""Optional official IBKR market-data adapter.

This module deliberately exposes no trading or account methods. The official
``ibapi`` client is imported only when an IBKR recorder is configured, keeping
CI and replay independent from the optional dependency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from stocker_prospective.config import RuntimeSafetyError
from stocker_prospective.durable_inbox import (
    CallbackClassification,
    CallbackIdentityCollision,
    CallbackInboxOverflow,
    DurableCallbackInbox,
)
from stocker_prospective.ibkr_api import (
    OfficialIBKRApiProvenanceError,
    load_official_ibkr_api_provenance,
    load_official_ibkr_api_update_status,
    python_package_tree_sha256,
)
from stocker_prospective.market_data import (
    BoundedCallbackRegistry,
    BoundedRealtimeBarQueue,
    BoundedStreamQuoteCache,
    CallbackResult,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataType,
    RealtimeBarUpdate,
    RequestIdAllocator,
    SubscriptionRegistry,
    classify_ibkr_error,
)

IBKR_DEPENDENCY_BLOCKER = "blocked_official_ibkr_api_not_installed"
IBKR_PROVENANCE_BLOCKER = "blocked_unverified_official_ibkr_api"
IBKR_API_UPDATE_MAX_AGE = timedelta(days=14)
IBKR_INFORMATIONAL_NOTIFICATION_CODES = frozenset({2104, 2106, 2107, 2108, 2119, 2158})
FORBIDDEN_BROKER_SURFACE = frozenset(
    {
        "placeOrder",
        "cancelOrder",
        "exerciseOptions",
        "reqGlobalCancel",
        "reqOpenOrders",
        "reqAllOpenOrders",
        "reqAutoOpenOrders",
        "reqAccountSummary",
        "reqAccountUpdates",
        "reqAccountUpdatesMulti",
        "reqPositions",
        "reqPositionsMulti",
        "reqExecutions",
        "reqCompletedOrders",
        "reqPnL",
        "reqPnLSingle",
        "place_order",
        "cancel_order",
        "exercise_options",
        "request_global_cancel",
        "request_account",
        "request_positions",
        "request_executions",
    }
)


@dataclass(frozen=True)
class _ClockProbeRequest:
    requested_at_utc: datetime
    requested_monotonic_ns: int
    connection_generation: int


def _serialisable_provider_value(
    value: object,
    *,
    _seen: set[int] | None = None,
) -> object:
    """Losslessly traverse one finite official callback envelope.

    A cycle or unsupported provider object is a callback-boundary failure. It
    is never replaced with a truncation marker that could pretend the original
    provider delivery was durably replayable.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__non_finite_float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__provider_bytes_hex__": value.hex()}
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        raise TypeError("provider callback payload contains a cycle")
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): _serialisable_provider_value(item, _seen=seen)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list, set, frozenset)):
            items = (
                sorted(value, key=lambda item: repr(item))
                if isinstance(value, (set, frozenset))
                else value
            )
            return [_serialisable_provider_value(item, _seen=seen) for item in items]
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return {
                "__provider_type__": type(value).__name__,
                "attributes": {
                    str(key): _serialisable_provider_value(item, _seen=seen)
                    for key, item in sorted(
                        attributes.items(),
                        key=lambda pair: str(pair[0]),
                    )
                    if not str(key).startswith("_") and not callable(item)
                },
            }
        return {
            "__provider_type__": type(value).__name__,
            "value": str(value),
        }
    finally:
        seen.remove(identity)


class OfficialIBKRDependencyError(RuntimeError):
    """The operator has not installed the official IBKR Python client."""


def _decode_proc_net_address(value: str, *, ipv6: bool) -> str:
    raw = bytes.fromhex(value)
    raw = b"".join(raw[index : index + 4][::-1] for index in range(0, 16, 4)) if ipv6 else raw[::-1]
    return str(ipaddress.ip_address(raw))


def require_ibkr_socket_loopback_only(
    host: str,
    port: int,
    *,
    proc_net_root: Path = Path("/proc/net"),
) -> tuple[str, ...]:
    """Fail closed unless the exact configured Linux listener is loopback-only."""

    try:
        destination = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeSafetyError(
            "blocked_unsafe_runtime_configuration: "
            f"ibkr host must be a literal loopback address: {host}"
        ) from exc
    if not destination.is_loopback:
        raise RuntimeSafetyError(
            "blocked_unsafe_runtime_configuration: "
            f"ibkr host must be a literal loopback address: {host}"
        )
    listeners: list[str] = []
    for filename, ipv6 in (("tcp", False), ("tcp6", True)):
        table = proc_net_root / filename
        if not table.is_file():
            continue
        for row in table.read_text(encoding="ascii").splitlines()[1:]:
            fields = row.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            try:
                address_hex, port_hex = fields[1].split(":", maxsplit=1)
                listener_port = int(port_hex, 16)
                address = _decode_proc_net_address(address_hex, ipv6=ipv6)
            except (ValueError, OSError):
                continue
            if listener_port == port and address not in listeners:
                listeners.append(address)
    if not listeners:
        raise RuntimeError(f"blocked_ibkr_connection: configured_socket_not_listening:{port}")
    unsafe = tuple(
        address for address in listeners if not ipaddress.ip_address(address).is_loopback
    )
    if unsafe:
        raise RuntimeSafetyError(
            "blocked_unsafe_runtime_configuration: "
            f"ibkr socket is not loopback-only: port={port}, addresses={','.join(unsafe)}"
        )
    return tuple(listeners)


@dataclass(frozen=True)
class IBKRConnectionConfig:
    host: str
    port: int
    client_id: int
    expected_environment: str
    connect_timeout_seconds: float
    request_timeout_seconds: float
    quote_capture_timeout_seconds: float
    allowed_market_data_types: tuple[MarketDataType, ...]

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("IBKR host is required")
        try:
            host_address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("IBKR host must be a literal loopback address") from exc
        if not host_address.is_loopback:
            raise ValueError("IBKR host must be a literal loopback address")
        if not 1 <= self.port <= 65535:
            raise ValueError("IBKR port must be explicitly configured")
        if self.client_id < 0:
            raise ValueError("IBKR client_id must be nonnegative")
        if self.expected_environment not in {"read_only", "paper", "live_read_only"}:
            raise ValueError("expected_environment must be read_only, paper, or live_read_only")
        if not self.allowed_market_data_types:
            raise ValueError("at least one market-data type must be allowed")


def official_ibkr_api_available() -> bool:
    return importlib.util.find_spec("ibapi") is not None


def require_official_ibkr_api(provenance_path: str | Path | None = None) -> ModuleType:
    if not official_ibkr_api_available():
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: install the official TWS API Python client "
            "from the IBKR Latest Mac/Unix distribution"
        )
    configured_path = provenance_path or os.environ.get("STOCKER_IBKR_API_PROVENANCE")
    if configured_path is None or not Path(configured_path).is_file():
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: official archive provenance is absent"
        )
    try:
        provenance = load_official_ibkr_api_provenance(configured_path)
    except OfficialIBKRApiProvenanceError as exc:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: official archive provenance is invalid"
        ) from exc
    module = __import__("ibapi")
    if not isinstance(module, ModuleType):
        raise OfficialIBKRDependencyError(f"{IBKR_DEPENDENCY_BLOCKER}: invalid ibapi module")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, (str, os.PathLike)) or not Path(module_file).is_file():
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi module path is invalid"
        )
    if getattr(module, "__version__", None) != provenance.api_version:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi version mismatch"
        )
    try:
        installed_tree_sha256 = python_package_tree_sha256(Path(module_file).parent)
    except OfficialIBKRApiProvenanceError as exc:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi tree is invalid"
        ) from exc
    if installed_tree_sha256 != provenance.installed_tree_sha256:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi tree hash mismatch"
        )
    return module


def official_ibkr_api_projection() -> dict[str, Any]:
    """Return secret-free installed-client and update health for the web process."""

    provenance_path = os.environ.get("STOCKER_IBKR_API_PROVENANCE")
    update_status_path = os.environ.get("STOCKER_IBKR_API_UPDATE_STATUS")
    projection: dict[str, Any] = {
        "installed": official_ibkr_api_available(),
        "verified": False,
        "api_version": None,
        "release_channel": None,
        "release_date": None,
        "update_checked_at_utc": None,
        "latest_api_version": None,
        "update_available": None,
        "update_status_fresh": False,
        "automatic_installation": False,
        "blocker": None,
    }
    try:
        require_official_ibkr_api(provenance_path)
        provenance = load_official_ibkr_api_provenance(str(provenance_path))
    except OfficialIBKRDependencyError as exc:
        projection["blocker"] = str(exc).split(":", 1)[0]
        return projection
    projection.update(
        {
            "verified": True,
            "api_version": provenance.api_version,
            "release_channel": provenance.release_channel,
            "release_date": provenance.release_date.isoformat(),
        }
    )
    if update_status_path is None:
        projection["blocker"] = "blocked_ibkr_api_update_status_missing"
        return projection
    if not Path(update_status_path).is_file():
        projection["blocker"] = "blocked_ibkr_api_update_status_missing"
        return projection
    try:
        status = load_official_ibkr_api_update_status(update_status_path)
    except OfficialIBKRApiProvenanceError:
        projection["blocker"] = "blocked_ibkr_api_update_status_invalid"
        return projection
    if (
        status.installed_api_version != provenance.api_version
        or status.installed_source_url != provenance.source_url
    ):
        projection["blocker"] = "blocked_ibkr_api_update_status_invalid"
        return projection
    projection.update(
        {
            "update_checked_at_utc": status.checked_at_utc.isoformat(),
            "latest_api_version": status.latest_api_version,
        }
    )
    update_age = datetime.now(UTC) - status.checked_at_utc.astimezone(UTC)
    if update_age > IBKR_API_UPDATE_MAX_AGE or update_age < timedelta(0):
        projection["blocker"] = "blocked_ibkr_api_update_check_stale"
        return projection
    projection.update(
        {
            "update_available": status.update_available,
            "update_status_fresh": True,
            "blocker": ("blocked_outdated_official_ibkr_api" if status.update_available else None),
        }
    )
    return projection


class IBKRMarketDataAdapter:
    """Narrow event-loop owner for exact market-data requests only."""

    def __init__(
        self,
        *,
        config: IBKRConnectionConfig,
        budget: MarketDataBudget,
        socket_preflight: Callable[[str, int], tuple[str, ...]] = (
            require_ibkr_socket_loopback_only
        ),
        max_stream_events: int = 65_536,
        durable_inbox: DurableCallbackInbox | None = None,
        require_durable_inbox_on_start: bool = False,
    ) -> None:
        if max_stream_events <= 0:
            raise ValueError("max_stream_events must be positive")
        self.config = config
        self.budget = budget
        self.request_ids = RequestIdAllocator()
        self.connection = ConnectionTracker()
        self.callbacks = BoundedCallbackRegistry(
            max_pending_requests=256,
            max_items_per_request=256,
        )
        self.stream_quotes = BoundedStreamQuoteCache(
            max_subscriptions=256,
            max_fields_per_subscription=64,
        )
        self.realtime_bars = BoundedRealtimeBarQueue(max_items=4096)
        self.subscriptions = SubscriptionRegistry()
        self._subscription_kinds: dict[int, str] = {}
        self._depth_smart: dict[int, bool] = {}
        self._stream_events: deque[dict[str, Any]] = deque()
        self._stream_event_limit = max_stream_events
        self._stream_event_sequence = 0
        self._stream_event_lock = threading.RLock()
        self._durable_inbox = durable_inbox
        self._require_durable_inbox_on_start = require_durable_inbox_on_start
        self._connection_generation = 0
        self._request_generations: dict[int, int] = {}
        self._request_owners: dict[int, str] = {}
        self._request_stream_owners: dict[int, dict[str, object]] = {}
        self._fatal_callback_code: str | None = None
        self._fatal_callback_sequence: int | None = None
        self._latest_durably_admitted_sequence: int | None = None
        self._pending_callback_failure: dict[str, Any] | None = None
        self._callback_failure_lock = threading.RLock()
        self._clock_probe_requests: deque[_ClockProbeRequest] = deque()
        self._clock_probe_lock = threading.RLock()
        self._official_callback_context = threading.local()
        self._loop_thread: threading.Thread | None = None
        self._client: Any | None = None
        self._stopping = threading.Event()
        self._client_loop_exit_expected = threading.Event()
        self._connected = threading.Event()
        self._socket_preflight = socket_preflight
        if self._durable_inbox is not None:
            self._restore_persisted_ingestion_latch()

    @property
    def connection_generation(self) -> int:
        return self._connection_generation

    @property
    def fatal_callback_code(self) -> str | None:
        return self._fatal_callback_code

    @property
    def scientific_recording_valid(self) -> bool:
        return self._fatal_callback_code is None

    def attach_durable_inbox(self, inbox: DurableCallbackInbox) -> None:
        """Attach the WAL-backed callback spool before the socket starts."""

        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("cannot replace durable inbox while callback loop is active")
        self._durable_inbox = inbox
        self._restore_persisted_ingestion_latch()

    def _restore_persisted_ingestion_latch(self) -> None:
        assert self._durable_inbox is not None
        active = self._durable_inbox.active_fatal("ingestion")
        if active is None:
            return
        self._fatal_callback_code, self._fatal_callback_sequence = active

    @property
    def dependency_blocker(self) -> str | None:
        return None if official_ibkr_api_available() else IBKR_DEPENDENCY_BLOCKER

    def attach_official_client(self, client: Any) -> None:
        """Attach a configured official EClient wrapper before ``start``.

        The wrapper is injected because the official API uses callback
        inheritance. Stocker never adds order callbacks or account routing.
        """

        require_official_ibkr_api()
        required = ("connect", "disconnect", "run", "reqMktData", "cancelMktData")
        missing = [name for name in required if not callable(getattr(client, name, None))]
        if missing:
            raise TypeError(f"official client missing market-data methods: {missing}")
        forbidden = sorted(
            name for name in FORBIDDEN_BROKER_SURFACE if callable(getattr(client, name, None))
        )
        if forbidden:
            raise TypeError(
                "order-capable or account/position/execution wrapper cannot be "
                "attached to the recorder: " + ", ".join(forbidden)
            )
        self._client = client

    def start(self) -> None:
        require_official_ibkr_api()
        if self._client is None:
            raise RuntimeError("official IBKR callback client has not been attached")
        if self._require_durable_inbox_on_start and self._durable_inbox is None:
            raise RuntimeError("blocked_ibkr_connection: durable_callback_inbox_required")
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._connection_generation += 1
        self._socket_preflight(self.config.host, self.config.port)
        self.connection.connecting()
        self._connected.clear()
        connection_result = self._client.connect(
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        if connection_result is False:
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection",
            )
            raise RuntimeError("blocked_ibkr_connection")
        self._stopping.clear()
        self._client_loop_exit_expected.clear()
        self._loop_thread = threading.Thread(
            target=self._run_client_loop,
            name="stocker-ibkr-callback-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._connected.wait(self.config.connect_timeout_seconds):
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection: handshake_timeout",
            )
            self.stop()
            raise RuntimeError("blocked_ibkr_connection: handshake_timeout")

    def stop(self) -> None:
        self.connection.shutting_down()
        self._stopping.set()
        self._client_loop_exit_expected.set()
        self.callbacks.shutdown()
        self.stream_quotes.clear()
        for key in self.budget.shutdown():
            if key.isdigit():
                self._tombstone_request(int(key), "adapter_shutdown")
                self._cancel_upstream(int(key))
        self._subscription_kinds.clear()
        if self._client is not None:
            self._client.disconnect()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=self.config.connect_timeout_seconds)
        self._loop_thread = None
        self._connected.clear()

    def reconnect(self) -> None:
        """Reconnect the socket after confirmed loss; subscriptions rebuild above."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        self._socket_preflight(self.config.host, self.config.port)
        self._connection_generation += 1
        self._connected.clear()
        self._client_loop_exit_expected.set()
        try:
            self._client.disconnect()
        finally:
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=self.config.connect_timeout_seconds)
            self._loop_thread = None
        self.connection.connecting()
        connection_result = self._client.connect(
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        if connection_result is False:
            self.connection.degraded(code=-1, message="blocked_ibkr_connection")
            raise RuntimeError("blocked_ibkr_connection")
        self._client_loop_exit_expected.clear()
        self._loop_thread = threading.Thread(
            target=self._run_client_loop,
            name="stocker-ibkr-callback-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._connected.wait(self.config.connect_timeout_seconds):
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection: reconnect_handshake_timeout",
            )
            raise RuntimeError("blocked_ibkr_connection: reconnect_handshake_timeout")

    def _run_client_loop(self) -> None:
        """Contain the official client's external network-loop boundary."""

        assert self._client is not None
        try:
            self._client.run()
        except Exception as exc:
            if self._stopping.is_set() or self._client_loop_exit_expected.is_set():
                return
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection: client_loop_failure",
            )
            try:
                self._latch_callback_failure(
                    callback_kind="client_run_loop",
                    request_id=-1,
                    error=exc,
                    source_sequence=self._latest_durably_admitted_sequence,
                    stable_error_code="IBKR_CLIENT_LOOP_FAILURE",
                    component="official_ibkr_client_loop",
                )
            except Exception:
                # This is an external thread boundary. Preserve a bounded
                # in-process fatal signal even if durable classification fails.
                with self._callback_failure_lock:
                    if self._fatal_callback_code is None:
                        self._fatal_callback_code = "IBKR_CLIENT_LOOP_FAILURE"
                        self._fatal_callback_sequence = self._latest_durably_admitted_sequence

    def request_market_data(
        self,
        contract: Any,
        *,
        subscription_key: str,
        generic_ticks: str = "",
    ) -> int:
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        request_id = self.request_ids.next()
        self._track_request(request_id, subscription_key)
        key = str(request_id)
        self.budget.reserve(key)
        self.stream_quotes.register(request_id)
        if not self.subscriptions.register(subscription_key, request_id):
            self.stream_quotes.remove(request_id)
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
                self._forget_request(request_id)
                return existing
        try:
            self._subscription_kinds[request_id] = "market_data"
            self._client.reqMktData(
                request_id,
                contract,
                generic_ticks,
                False,
                False,
                [],
            )
            self.budget.mark_active(key)
        except Exception:
            self.budget.confirm_cancellation(key)
            self.stream_quotes.remove(request_id)
            self.subscriptions.remove(subscription_key)
            self._subscription_kinds.pop(request_id, None)
            self._tombstone_request(request_id, "market_data_request_failed")
            raise
        return request_id

    def cancel_market_data(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        key = str(request_id)
        if not self.budget.request_cancellation(key):
            return
        self._tombstone_request(request_id, "market_data_cancelled")
        self._cancel_upstream(request_id)
        self.budget.confirm_cancellation(key)
        self.stream_quotes.remove(request_id)
        self._subscription_kinds.pop(request_id, None)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def request_realtime_bars(
        self,
        contract: Any,
        *,
        subscription_key: str,
    ) -> int:
        """Request bounded 5-second RTH trade bars for diagnostic aggregation."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqRealTimeBars", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        self._track_request(request_id, subscription_key)
        key = str(request_id)
        self.budget.reserve(key)
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
                self._forget_request(request_id)
                return existing
        try:
            self._subscription_kinds[request_id] = "realtime_bars"
            method(
                request_id,
                contract,
                5,
                "TRADES",
                True,
                [],
            )
            self.budget.mark_active(key)
        except Exception:
            self.budget.confirm_cancellation(key)
            self.subscriptions.remove(subscription_key)
            self._subscription_kinds.pop(request_id, None)
            self._tombstone_request(request_id, "realtime_bar_request_failed")
            raise
        return request_id

    def cancel_realtime_bars(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        key = str(request_id)
        if not self.budget.request_cancellation(key):
            return
        self._tombstone_request(request_id, "realtime_bar_cancelled")
        self._cancel_upstream(request_id)
        self.budget.confirm_cancellation(key)
        self._subscription_kinds.pop(request_id, None)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def request_tick_by_tick(
        self,
        contract: Any,
        *,
        subscription_key: str,
        tick_type: Literal["BidAsk", "Last"],
    ) -> int:
        """Request one bounded official BidAsk or Last stream."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqTickByTickData", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        if tick_type not in {"BidAsk", "Last"}:
            raise ValueError("tick_type must be BidAsk or Last")
        request_id, existing = self._begin_stream_subscription(
            subscription_key=subscription_key,
            kind="tick_by_tick",
        )
        if existing is not None:
            return existing
        try:
            method(request_id, contract, tick_type, 0, False)
            self.budget.mark_active(str(request_id))
        except Exception:
            self._abort_stream_subscription(request_id, subscription_key)
            raise
        return request_id

    def cancel_tick_by_tick(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        self._cancel_subscription(request_id, subscription_key=subscription_key)

    def request_market_depth(
        self,
        contract: Any,
        *,
        subscription_key: str,
        rows: int,
        smart_depth: bool = True,
    ) -> int:
        """Request a bounded order book; Level II remains optional."""

        if rows <= 0:
            raise ValueError("depth rows must be positive")
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqMktDepth", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id, existing = self._begin_stream_subscription(
            subscription_key=subscription_key,
            kind="market_depth",
        )
        if existing is not None:
            return existing
        self._depth_smart[request_id] = smart_depth
        try:
            method(request_id, contract, rows, smart_depth, [])
            self.budget.mark_active(str(request_id))
        except Exception:
            self._depth_smart.pop(request_id, None)
            self._abort_stream_subscription(request_id, subscription_key)
            raise
        return request_id

    def cancel_market_depth(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        self._cancel_subscription(request_id, subscription_key=subscription_key)

    def request_historical_five_minute_updates(
        self,
        contract: Any,
        *,
        subscription_key: str,
    ) -> int:
        """Request causal completed IBKR five-minute RTH trade bars."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqHistoricalData", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id, existing = self._begin_stream_subscription(
            subscription_key=subscription_key,
            kind="historical_5m",
        )
        if existing is not None:
            return existing
        try:
            method(
                request_id,
                contract,
                "",
                "1 D",
                "5 mins",
                "TRADES",
                1,
                2,
                True,
                [],
            )
            self.budget.mark_active(str(request_id))
        except Exception:
            self._abort_stream_subscription(request_id, subscription_key)
            raise
        return request_id

    def cancel_historical_updates(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        self._cancel_subscription(request_id, subscription_key=subscription_key)

    def request_current_time(self) -> None:
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqCurrentTime", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_capability_preflight")
        request = _ClockProbeRequest(
            requested_at_utc=datetime.now(UTC),
            requested_monotonic_ns=time.monotonic_ns(),
            connection_generation=self._connection_generation,
        )
        with self._clock_probe_lock:
            while (
                self._clock_probe_requests
                and self._clock_probe_requests[0].connection_generation
                < self._connection_generation
            ):
                self._clock_probe_requests.popleft()
            if self._clock_probe_requests:
                outstanding = self._clock_probe_requests[-1]
                elapsed_ns = request.requested_monotonic_ns - outstanding.requested_monotonic_ns
                timeout_ns = int(self.config.request_timeout_seconds * 1_000_000_000)
                if (
                    outstanding.connection_generation == self._connection_generation
                    and 0 <= elapsed_ns <= timeout_ns
                ):
                    return
                # ``currentTime`` has no request identifier. Keep one request
                # in flight, but permit exactly one bounded retry when IBKR
                # loses a response.  Preserve the original request boundary:
                # callbacks are ordered but untagged, so discarding it lets a
                # late first response steal the retry's timestamp.
                if len(self._clock_probe_requests) >= 2:
                    return
            self._clock_probe_requests.append(request)
        try:
            method()
        except Exception:
            with self._clock_probe_lock:
                if request in self._clock_probe_requests:
                    self._clock_probe_requests.remove(request)
            raise

    def request_depth_exchanges(self) -> None:
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqMktDepthExchanges", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_capability_preflight")
        method()

    def require_live_market_data(self) -> None:
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqMarketDataType", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_capability_preflight")
        method(1)

    def server_version(self) -> int | None:
        if self._client is None:
            return None
        method = getattr(self._client, "serverVersion", None)
        if not callable(method):
            return None
        value = method()
        return None if value is None else int(value)

    def _begin_stream_subscription(
        self,
        *,
        subscription_key: str,
        kind: str,
    ) -> tuple[int, int | None]:
        request_id = self.request_ids.next()
        self._track_request(request_id, subscription_key)
        key = str(request_id)
        self.budget.reserve(key)
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is None:
                raise RuntimeError("subscription registry changed during allocation")
            self.subscriptions.register(subscription_key, existing)
            self._forget_request(request_id)
            return request_id, existing
        self._subscription_kinds[request_id] = kind
        return request_id, None

    def _track_request(self, request_id: int, subscription_owner: str) -> None:
        self._request_generations[request_id] = self._connection_generation
        self._request_owners[request_id] = subscription_owner

    def register_stream_owner(
        self,
        request_id: int,
        stream_owner: Mapping[str, object],
    ) -> None:
        """Attach typed ownership after a request ID is allocated.

        The durable inbox backfill closes the small official-client race where
        a synchronous callback arrives before the request method returns.
        """

        if self._request_generations.get(request_id) != self._connection_generation:
            raise ValueError("stream owner request generation differs")
        encoded = json.dumps(
            dict(stream_owner),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or int(payload.get("request_id", -1)) != request_id:
            raise ValueError("stream owner request identity differs")
        existing = self._request_stream_owners.get(request_id)
        if existing is not None and existing != payload:
            raise ValueError("stream owner identity differs")
        self._request_stream_owners[request_id] = payload
        if self._durable_inbox is not None:
            self._durable_inbox.attach_stream_owner(
                request_id=request_id,
                connection_generation=self._connection_generation,
                stream_owner=payload,
                attached_at=datetime.now(UTC),
            )

    def _forget_request(self, request_id: int) -> None:
        self._request_generations.pop(request_id, None)
        self._request_owners.pop(request_id, None)
        self._request_stream_owners.pop(request_id, None)

    @staticmethod
    def _symbol_from_owner(owner: str | None) -> str | None:
        if owner is None:
            return None
        candidate = owner.split(":", 1)[0].strip().upper()
        return candidate if candidate.isalpha() and 1 <= len(candidate) <= 8 else None

    def _tombstone_request(self, request_id: int, reason: str) -> None:
        generation = self._request_generations.get(
            request_id,
            self._connection_generation,
        )
        owner = self._request_owners.get(request_id)
        if self._durable_inbox is not None:
            self._durable_inbox.record_tombstone(
                request_id=request_id,
                connection_generation=generation,
                subscription_owner=owner,
                symbol=self._symbol_from_owner(owner),
                cancellation_reason=reason,
                cancelled_at=datetime.now(UTC),
                ttl=timedelta(minutes=15),
            )
        self._forget_request(request_id)

    def _classify_callback(
        self,
        request_id: int,
        *,
        now: datetime,
    ) -> CallbackClassification:
        if self._fatal_callback_code is not None:
            return CallbackClassification.AFTER_DATA_LOSS_LATCH
        if request_id < 0:
            return CallbackClassification.CONTROL
        request_generation = self._request_generations.get(request_id)
        if request_generation is not None and request_generation < self._connection_generation:
            return CallbackClassification.PREVIOUS_CONNECTION
        if request_id in self._subscription_kinds or self.callbacks.is_pending(request_id):
            return CallbackClassification.ACCEPTED_ACTIVE
        if self._durable_inbox is not None:
            tombstone = self._durable_inbox.tombstone(request_id=request_id, now=now)
            if tombstone is not None:
                if tombstone.connection_generation < self._connection_generation:
                    return CallbackClassification.PREVIOUS_CONNECTION
                return CallbackClassification.EXPECTED_LATE
        return CallbackClassification.UNKNOWN

    def contain_official_callback(
        self,
        callback_kind: str,
        request_id: int,
        callback: Callable[[], None],
        *,
        provider_arguments: tuple[object, ...] = (),
        provider_keywords: Mapping[str, object] | None = None,
    ) -> None:
        """True external boundary: classify every failure and never re-raise."""

        sequence_before = self._latest_durably_admitted_sequence
        provider_event_id: str | None = None
        try:
            self.flush_pending_callback_failure()
            received_at = datetime.now(UTC)
            received_monotonic_ns = time.monotonic_ns()
            classification = (
                CallbackClassification.CONTROL
                if callback_kind
                in {
                    "error",
                    "connection_closed",
                    "current_time",
                    "market_depth_exchanges",
                }
                else self._classify_callback(request_id, now=received_at)
            )
            bounded_recovery_callback = (
                classification is CallbackClassification.AFTER_DATA_LOSS_LATCH
                and request_id >= 0
                and self.callbacks.is_pending(request_id)
            )
            provider_payload = {
                "provider_arguments": _serialisable_provider_value(provider_arguments),
                "provider_keywords": _serialisable_provider_value(dict(provider_keywords or {})),
            }
            provider_event_id = hashlib.sha256(
                "|".join(
                    (
                        "official_provider_callback",
                        callback_kind,
                        str(request_id),
                        str(self._durable_inbox.run_id if self._durable_inbox else None),
                        str(self._connection_generation),
                        received_at.isoformat(),
                        str(received_monotonic_ns),
                    )
                ).encode()
            ).hexdigest()
            single_row_hot_path = (
                classification is CallbackClassification.ACCEPTED_ACTIVE
                and callback_kind
                in {
                    "tick_price",
                    "tick_size",
                    "historical_data",
                    "historical_data_update",
                }
            )
            if self._durable_inbox is not None:
                owner = self._request_owners.get(request_id)
                admission = self._durable_inbox.admit(
                    callback_kind=f"official_provider_{callback_kind}",
                    request_id=request_id,
                    payload=provider_payload,
                    connection_generation=self._connection_generation,
                    classification=classification,
                    received_utc=received_at,
                    received_monotonic_ns=received_monotonic_ns,
                    inbox_event_id=provider_event_id,
                    subscription_owner=owner,
                    symbol=self._symbol_from_owner(owner),
                    stream_owner=self._request_stream_owners.get(request_id),
                    provider_envelope=True,
                    allow_after_data_loss_provider_processing=(bounded_recovery_callback),
                )
                self._latest_durably_admitted_sequence = admission.event.source_sequence
                if admission.duplicate:
                    raise CallbackIdentityCollision("CALLBACK_PROVIDER_DELIVERY_ID_COLLISION")
            if classification in {
                CallbackClassification.EXPECTED_LATE,
                CallbackClassification.PREVIOUS_CONNECTION,
            }:
                return
            if classification is CallbackClassification.UNKNOWN or (
                classification is CallbackClassification.AFTER_DATA_LOSS_LATCH
                and not bounded_recovery_callback
            ):
                self._latch_callback_failure(
                    callback_kind=callback_kind,
                    request_id=request_id,
                    error=RuntimeError(
                        "unknown_callback_request_id"
                        if classification is CallbackClassification.UNKNOWN
                        else "callback_after_data_loss_latch"
                    ),
                    source_sequence=self._latest_durably_admitted_sequence,
                )
                return
            # The provider envelope and its canonical scientific event describe
            # one delivery. Preserve the true EWrapper boundary timestamps so
            # scheduler delay cannot masquerade as provider clock drift.
            self._official_callback_context.provider_event_id = provider_event_id
            self._official_callback_context.received_at_utc = received_at
            self._official_callback_context.received_monotonic_ns = received_monotonic_ns
            self._official_callback_context.capture_single_row = single_row_hot_path
            self._official_callback_context.captured_stream_event = None
            callback()
            captured = self._official_callback_context.captured_stream_event
            if self._durable_inbox is not None and single_row_hot_path and captured is not None:
                captured_kind, captured_request_id, captured_payload = captured
                provider_json = json.dumps(
                    provider_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                )
                durable_payload = {
                    **captured_payload,
                    "original_provider_callback_kind": callback_kind,
                    "original_provider_callback": provider_payload,
                    "original_provider_callback_sha256": hashlib.sha256(
                        provider_json.encode()
                    ).hexdigest(),
                }
                materialized = self._durable_inbox.materialize_provider_envelope(
                    provider_envelope_event_id=provider_event_id,
                    callback_kind=captured_kind,
                    request_id=captured_request_id,
                    payload=durable_payload,
                    materialized_at=datetime.now(UTC),
                )
                self._latest_durably_admitted_sequence = materialized.source_sequence
                if captured_kind == "level1_quote_update":
                    self.stream_quotes.add(captured_request_id, captured_payload)
            elif self._durable_inbox is not None:
                self._durable_inbox.complete_provider_envelope(
                    provider_envelope_event_id=provider_event_id,
                    completed_at=datetime.now(UTC),
                )
        except Exception as exc:
            admitted_sequence = (
                self._latest_durably_admitted_sequence
                if self._latest_durably_admitted_sequence != sequence_before
                else None
            )
            try:
                if self._durable_inbox is not None and provider_event_id is not None:
                    self._durable_inbox.quarantine_provider_envelope(
                        provider_envelope_event_id=provider_event_id,
                        failure_classification=self._stable_callback_error_code(exc),
                        quarantined_at=datetime.now(UTC),
                    )
                self._latch_callback_failure(
                    callback_kind=callback_kind,
                    request_id=request_id,
                    error=exc,
                    source_sequence=admitted_sequence,
                )
            except Exception:
                # This is the last-resort in-process latch. Durable recording
                # is already best-effort inside _latch_callback_failure, but
                # even an unexpected classifier bug must not escape EWrapper.
                with self._callback_failure_lock:
                    if self._fatal_callback_code is None:
                        self._fatal_callback_code = "CALLBACK_BOUNDARY_FAILURE"
                        self._fatal_callback_sequence = admitted_sequence
        finally:
            self._official_callback_context.provider_event_id = None
            self._official_callback_context.received_at_utc = None
            self._official_callback_context.received_monotonic_ns = None
            self._official_callback_context.capture_single_row = False
            self._official_callback_context.captured_stream_event = None

    @staticmethod
    def _stable_callback_error_code(error: Exception) -> str:
        if isinstance(error, CallbackInboxOverflow) or "queue_exhausted" in str(error):
            return "CALLBACK_OVERFLOW"
        if isinstance(error, CallbackIdentityCollision):
            return "CALLBACK_IDENTITY_COLLISION"
        if isinstance(error, sqlite3.Error):
            return "CALLBACK_DURABLE_ADMISSION_FAILED"
        if "unknown_callback_request_id" in str(error):
            return "CALLBACK_UNKNOWN_REQUEST_ID"
        if "cache_failed" in str(error):
            return "CALLBACK_CACHE_FAILURE"
        if isinstance(error, (TypeError, ValueError, OverflowError)):
            return "CALLBACK_MALFORMED_VALUE"
        return "CALLBACK_BOUNDARY_FAILURE"

    def _latch_callback_failure(
        self,
        *,
        callback_kind: str,
        request_id: int,
        error: Exception,
        source_sequence: int | None,
        stable_error_code: str | None = None,
        component: str = "official_ibkr_callback",
    ) -> None:
        occurred_at = datetime.now(UTC)
        code = stable_error_code or self._stable_callback_error_code(error)
        with self._callback_failure_lock:
            if self._fatal_callback_code is None:
                self._fatal_callback_code = code
                self._fatal_callback_sequence = source_sequence
            payload: dict[str, Any] = {
                "stable_error_code": code,
                "callback_kind": callback_kind,
                "request_id": request_id,
                "error_class": type(error).__name__,
                "occurred_at": occurred_at,
                "source_sequence": source_sequence,
                "connection_generation": self._connection_generation,
                "subscription_owner": self._request_owners.get(request_id),
                "symbol": self._symbol_from_owner(self._request_owners.get(request_id)),
                "component": component,
                "failure_count": 1,
            }
            if not self._persist_callback_failure(payload):
                if self._pending_callback_failure is None:
                    self._pending_callback_failure = payload
                else:
                    self._pending_callback_failure["failure_count"] = (
                        int(self._pending_callback_failure.get("failure_count", 1)) + 1
                    )

    def _persist_callback_failure(self, payload: dict[str, Any]) -> bool:
        if self._durable_inbox is None:
            return False
        occurred_at = payload["occurred_at"]
        assert isinstance(occurred_at, datetime)
        try:
            self._durable_inbox.record_incident(
                stable_error_code=str(payload["stable_error_code"]),
                component=str(payload.get("component", "official_ibkr_callback")),
                severity="fatal",
                occurred_at=occurred_at,
                error_class=str(payload["error_class"]),
                evidence_loss_possible=True,
                callback_kind=str(payload["callback_kind"]),
                request_id=int(payload["request_id"]),
                source_sequence=(
                    None if payload["source_sequence"] is None else int(payload["source_sequence"])
                ),
                connection_generation=int(payload["connection_generation"]),
                subscription_owner=(
                    None
                    if payload["subscription_owner"] is None
                    else str(payload["subscription_owner"])
                ),
                symbol=None if payload["symbol"] is None else str(payload["symbol"]),
                details={"failure_count": int(payload.get("failure_count", 1))},
            )
            self._durable_inbox.latch_fatal(
                latch_kind="ingestion",
                stable_error_code=str(payload["stable_error_code"]),
                occurred_at=occurred_at,
                error_class=str(payload["error_class"]),
                evidence_loss_possible=True,
                first_possibly_lost_source_sequence=(
                    None if payload["source_sequence"] is None else int(payload["source_sequence"])
                ),
                callback_kind=str(payload["callback_kind"]),
                request_id=int(payload["request_id"]),
                connection_generation=int(payload["connection_generation"]),
            )
        except Exception:
            return False
        return True

    def flush_pending_callback_failure(self) -> None:
        """Retry the single bounded aggregate when SQLite becomes writable."""

        with self._callback_failure_lock:
            pending = self._pending_callback_failure
            if pending is not None and self._persist_callback_failure(pending):
                self._pending_callback_failure = None

    def _abort_stream_subscription(self, request_id: int, subscription_key: str) -> None:
        self._tombstone_request(request_id, "stream_subscription_failed")
        self.budget.confirm_cancellation(str(request_id))
        self.subscriptions.remove(subscription_key)
        self._subscription_kinds.pop(request_id, None)

    def _cancel_subscription(
        self,
        request_id: int,
        *,
        subscription_key: str | None,
    ) -> None:
        key = str(request_id)
        if not self.budget.request_cancellation(key):
            return
        self._tombstone_request(request_id, "stream_subscription_cancelled")
        self._cancel_upstream(request_id)
        self.budget.confirm_cancellation(key)
        self._subscription_kinds.pop(request_id, None)
        self._depth_smart.pop(request_id, None)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def _cancel_upstream(self, request_id: int) -> None:
        if self._client is None:
            return
        kind = self._subscription_kinds.get(request_id)
        if kind == "realtime_bars":
            method = getattr(self._client, "cancelRealTimeBars", None)
            if callable(method):
                method(request_id)
            return
        if kind == "tick_by_tick":
            method = getattr(self._client, "cancelTickByTickData", None)
            if callable(method):
                method(request_id)
            return
        if kind == "market_depth":
            method = getattr(self._client, "cancelMktDepth", None)
            if callable(method):
                method(request_id, self._depth_smart.get(request_id, True))
            return
        if kind == "historical_5m":
            method = getattr(self._client, "cancelHistoricalData", None)
            if callable(method):
                method(request_id)
            return
        self._client.cancelMktData(request_id)

    def capture_temporary_quote(
        self,
        *,
        contract: Any,
        timeout_seconds: float | None = None,
        generic_ticks: str = "",
    ) -> CallbackResult:
        """Capture one official snapshot and always cancel local capacity.

        IBKR snapshots complete through ``tickSnapshotEnd``. Generic ticks are
        deliberately unavailable in snapshot mode, so volume/open-interest
        remain missing unless present in the standard callbacks.
        """

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        if generic_ticks:
            raise ValueError("generic ticks are unavailable for bounded snapshot captures")
        request_id = self.request_ids.next()
        self._track_request(request_id, f"temporary_quote:{request_id}")
        key = str(request_id)
        self.budget.reserve(key)
        self.callbacks.begin(request_id, kind="temporary_quote")
        broker_snapshot_complete = False
        try:
            try:
                self._client.reqMktData(
                    request_id,
                    contract,
                    generic_ticks,
                    True,
                    False,
                    [],
                )
            except Exception:
                self.callbacks.abort(request_id, "upstream_quote_request_failed")
                raise
            self.budget.mark_active(key)
            result = self.callbacks.wait(
                request_id,
                timeout_seconds=(
                    self.config.quote_capture_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            )
            broker_snapshot_complete = result.complete
            return result
        finally:
            self._tombstone_request(request_id, "temporary_quote_complete")
            self.budget.request_cancellation(key)
            if not broker_snapshot_complete:
                self._client.cancelMktData(request_id)
            self.budget.confirm_cancellation(key)

    def actual_subscription_request_ids(self) -> set[int]:
        """Expose only this adapter's request IDs for registry reconciliation."""

        return {request_id for _key, request_id in self.subscriptions.active_items()}

    def cancel_orphaned_market_data_request(self, request_id: int) -> None:
        """Repair a request that has no higher-level internal owner."""

        self._tombstone_request(request_id, "orphaned_market_data_request")
        self._cancel_upstream(request_id)
        self.budget.request_cancellation(str(request_id))
        self.budget.confirm_cancellation(str(request_id))
        self._subscription_kinds.pop(request_id, None)
        self._depth_smart.pop(request_id, None)
        for key, candidate in self.subscriptions.active_items():
            if candidate == request_id:
                self.subscriptions.remove(key)

    def request_option_chain_metadata(
        self,
        *,
        underlying_symbol: str,
        exchange: str,
        underlying_security_type: str,
        underlying_contract_id: int,
    ) -> CallbackResult:
        """Call official ``reqSecDefOptParams``; never stream every contract."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqSecDefOptParams", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        self._track_request(request_id, f"option_chain_metadata:{underlying_symbol}")
        budget_key = f"metadata:{request_id}"
        self.budget.reserve(budget_key, lines=0)
        try:
            self.callbacks.begin(request_id, kind="option_chain_metadata")
            try:
                method(
                    request_id,
                    underlying_symbol,
                    exchange,
                    underlying_security_type,
                    underlying_contract_id,
                )
            except Exception:
                self.callbacks.abort(request_id, "upstream_metadata_request_failed")
                raise
            return self.callbacks.wait(
                request_id,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        finally:
            self._tombstone_request(request_id, "option_chain_metadata_complete")
            self.budget.confirm_cancellation(budget_key)

    def qualify_exact_contract(self, contract: Any) -> CallbackResult:
        """Qualify one exact bounded contract through ``reqContractDetails``."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqContractDetails", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        self._track_request(request_id, f"exact_contract_qualification:{request_id}")
        budget_key = f"qualification:{request_id}"
        self.budget.reserve(budget_key, lines=0)
        try:
            self.callbacks.begin(request_id, kind="exact_contract_qualification")
            try:
                method(request_id, contract)
            except Exception:
                self.callbacks.abort(request_id, "upstream_contract_request_failed")
                raise
            return self.callbacks.wait(
                request_id,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        finally:
            self._tombstone_request(request_id, "contract_qualification_complete")
            self.budget.confirm_cancellation(budget_key)

    def on_option_parameter(self, request_id: int, payload: Any) -> None:
        self.callbacks.add(request_id, payload)

    def on_option_parameter_end(self, request_id: int) -> None:
        self.callbacks.complete(request_id)

    def on_contract_details(self, request_id: int, payload: Any) -> None:
        self.callbacks.add(request_id, payload)

    def on_contract_details_end(self, request_id: int) -> None:
        self.callbacks.complete(request_id)

    def on_quote_update(
        self,
        request_id: int,
        payload: dict[str, Any],
        *,
        complete: bool = False,
    ) -> bool:
        """Receive a quote update without converting missing fields to zero."""

        if self.callbacks.is_pending(request_id):
            self.callbacks.add(request_id, payload)
            if complete:
                self.callbacks.complete(request_id)
            return True
        admitted = self._append_stream_event("level1_quote_update", request_id, payload)
        if admitted:
            self.stream_quotes.add(request_id, payload)
        return admitted

    def on_connected(self, market_data_type: MarketDataType | None) -> None:
        self.connection.connected(market_data_type)
        self._connected.set()

    def on_market_data_type(
        self,
        request_id: int,
        market_data_type: MarketDataType,
    ) -> None:
        admitted = self.on_quote_update(
            request_id,
            {
                "field": "market_data_type",
                "value": market_data_type.value,
                "market_data_type": market_data_type.value,
            },
        )
        if admitted:
            self.connection.market_data_type_observed(market_data_type)

    def on_realtime_bar(self, update: RealtimeBarUpdate) -> None:
        payload = asdict(update)
        if self._append_stream_event("realtime_bar", update.request_id, payload):
            self.realtime_bars.add(update)

    def on_tick_by_tick_bidask(self, request_id: int, payload: dict[str, Any]) -> None:
        self._append_stream_event("tick_by_tick_bidask", request_id, payload)

    def on_tick_by_tick_trade(self, request_id: int, payload: dict[str, Any]) -> None:
        self._append_stream_event("tick_by_tick_trade", request_id, payload)

    def on_depth_update(self, request_id: int, payload: dict[str, Any]) -> None:
        self._append_stream_event("depth", request_id, payload)

    def on_depth_reset(self, request_id: int, reason: str) -> None:
        self._append_stream_event(
            "depth_reset",
            request_id,
            {"reason": reason, "book_valid": False},
        )

    def on_historical_bar(
        self,
        request_id: int,
        payload: dict[str, Any],
        *,
        update: bool,
    ) -> None:
        self._append_stream_event(
            "historical_bar_update" if update else "historical_bar",
            request_id,
            payload,
        )

    def on_historical_bar_end(
        self,
        request_id: int,
        *,
        start: str,
        end: str,
    ) -> None:
        self._append_stream_event(
            "historical_backfill_end",
            request_id,
            {"start": start, "end": end},
        )

    def on_current_time(self, provider_timestamp_utc: datetime) -> None:
        request: _ClockProbeRequest | None = None
        with self._clock_probe_lock:
            while (
                self._clock_probe_requests
                and self._clock_probe_requests[0].connection_generation
                < self._connection_generation
            ):
                self._clock_probe_requests.popleft()
            if (
                self._clock_probe_requests
                and self._clock_probe_requests[0].connection_generation
                == self._connection_generation
            ):
                request = self._clock_probe_requests.popleft()
        payload: dict[str, Any] = {
            "provider_timestamp_utc": provider_timestamp_utc.astimezone(UTC).isoformat()
        }
        if request is not None:
            payload.update(
                {
                    "clock_probe_requested_at_utc": request.requested_at_utc.isoformat(),
                    "clock_probe_requested_monotonic_ns": (request.requested_monotonic_ns),
                }
            )
        self._append_stream_event(
            "current_time",
            -1,
            payload,
        )

    def on_depth_exchanges(self, exchanges: tuple[dict[str, Any], ...]) -> None:
        self._append_stream_event("depth_exchanges", -1, {"exchanges": exchanges})

    def _append_stream_event(
        self,
        kind: str,
        request_id: int,
        payload: dict[str, Any],
    ) -> bool:
        boundary_received_at = getattr(
            self._official_callback_context,
            "received_at_utc",
            None,
        )
        boundary_received_monotonic_ns = getattr(
            self._official_callback_context,
            "received_monotonic_ns",
            None,
        )
        received_at = datetime.now(UTC) if boundary_received_at is None else boundary_received_at
        received_monotonic_ns = (
            time.monotonic_ns()
            if boundary_received_monotonic_ns is None
            else boundary_received_monotonic_ns
        )
        if self._durable_inbox is not None and bool(
            getattr(self._official_callback_context, "capture_single_row", False)
        ):
            if getattr(self._official_callback_context, "captured_stream_event", None) is not None:
                raise RuntimeError("official callback emitted multiple canonical stream events")
            self._official_callback_context.captured_stream_event = (
                kind,
                request_id,
                dict(payload),
            )
            return False
        if self._durable_inbox is not None:
            classification = self._classify_callback(request_id, now=received_at)
            identity_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"receive_timestamp_utc", "received_timestamp_utc"}
            }
            identity_json = json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
                allow_nan=True,
            )
            provider_event_id = getattr(
                self._official_callback_context,
                "provider_event_id",
                None,
            )
            if provider_event_id is None:
                provider_event_id = identity_payload.get("provider_event_id")
            delivery_identity = (
                f"provider:{provider_event_id}:{kind}"
                if provider_event_id is not None
                else f"callback:{received_at.isoformat()}:{received_monotonic_ns}"
            )
            event_id = hashlib.sha256(
                "|".join(
                    (
                        kind,
                        str(request_id),
                        str(self._durable_inbox.run_id),
                        str(self._connection_generation),
                        delivery_identity,
                        "" if provider_event_id is not None else identity_json,
                    )
                ).encode()
            ).hexdigest()
            owner = self._request_owners.get(request_id)
            result = self._durable_inbox.admit(
                callback_kind=kind,
                request_id=request_id,
                payload=payload,
                connection_generation=self._connection_generation,
                classification=classification,
                received_utc=received_at,
                received_monotonic_ns=received_monotonic_ns,
                inbox_event_id=event_id,
                subscription_owner=owner,
                symbol=self._symbol_from_owner(owner),
                stream_owner=self._request_stream_owners.get(request_id),
                provider_envelope_event_id=(
                    None if provider_event_id is None else str(provider_event_id)
                ),
            )
            self._latest_durably_admitted_sequence = result.event.source_sequence
            if result.duplicate:
                self._durable_inbox.record_incident(
                    stable_error_code="DUPLICATE_CALLBACK",
                    component="official_ibkr_callback",
                    severity="diagnostic",
                    occurred_at=received_at,
                    error_class="DuplicateCallback",
                    evidence_loss_possible=False,
                    callback_kind=kind,
                    request_id=request_id,
                    source_sequence=result.event.source_sequence,
                    connection_generation=self._connection_generation,
                    subscription_owner=owner,
                    symbol=self._symbol_from_owner(owner),
                )
                return False
            if classification is CallbackClassification.UNKNOWN:
                self._latch_callback_failure(
                    callback_kind=kind,
                    request_id=request_id,
                    error=RuntimeError("unknown_callback_request_id"),
                    source_sequence=result.event.source_sequence,
                )
                return False
            return classification in {
                CallbackClassification.ACCEPTED_ACTIVE,
                CallbackClassification.CONTROL,
            }
        with self._stream_event_lock:
            if len(self._stream_events) >= self._stream_event_limit:
                raise RuntimeError("bounded_ibkr_stream_event_queue_exhausted")
            self._stream_event_sequence += 1
            event = {
                **payload,
                "kind": kind,
                "request_id": request_id,
                "received_timestamp_utc": payload.get(
                    "receive_timestamp_utc",
                    received_at.isoformat(),
                ),
                "received_monotonic_ns": received_monotonic_ns,
                "source_sequence": self._stream_event_sequence,
            }
            event.pop("receive_timestamp_utc", None)
            self._stream_events.append(event)
        return True

    def drain_stream_events(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        if limit is not None and limit <= 0:
            raise ValueError("drain limit must be positive")
        with self._stream_event_lock:
            count = (
                len(self._stream_events) if limit is None else min(limit, len(self._stream_events))
            )
            return tuple(self._stream_events.popleft() for _ in range(count))

    def _clear_lost_subscriptions(self) -> None:
        self.callbacks.shutdown()
        for _, request_id in self.subscriptions.active_items():
            self._tombstone_request(request_id, "connection_generation_replaced")
            self.budget.confirm_cancellation(str(request_id))
            self.stream_quotes.remove(request_id)
            self._subscription_kinds.pop(request_id, None)
            self._depth_smart.pop(request_id, None)
        self.subscriptions.after_reconnect(data_maintained=False)

    def on_connection_closed(self) -> None:
        self._connected.clear()
        self.connection.connection_lost(
            code=1100,
            message="official_socket_connection_closed",
        )
        self._clear_lost_subscriptions()

    def on_error(self, request_id: int, code: int, message: str) -> None:
        if code in IBKR_INFORMATIONAL_NOTIFICATION_CODES:
            self.connection.notification(code=code, message=message)
            return
        if code == 1100:
            self._connected.clear()
            self.connection.connection_lost(code=code, message=message)
            return
        if code == 1101:
            self.connection.connection_restored(data_maintained=False, code=code)
            self._clear_lost_subscriptions()
            self._connected.set()
            return
        if code == 1102:
            self.connection.connection_restored(data_maintained=True, code=code)
            self._connected.set()
            return
        if code == 1300:
            self._connected.clear()
            self.connection.socket_port_reset(self.config.port)
            self._clear_lost_subscriptions()
            return
        if code == 504:
            self._connected.clear()
            self.connection.connection_lost(
                code=code,
                message=f"{classify_ibkr_error(code)}:{message}",
            )
            self._clear_lost_subscriptions()
            return
        if code == 317:
            self.on_depth_reset(request_id, "ibkr_market_depth_reset")
            return
        reason = classify_ibkr_error(code)
        self._append_stream_event(
            "ibkr_error",
            request_id,
            {"error_code": code, "reason": reason, "message": message},
        )
        self.connection.degraded(code=code, message=f"{reason}:{message}")
        try:
            self.callbacks.fail(request_id, reason)
        except Exception:
            return
