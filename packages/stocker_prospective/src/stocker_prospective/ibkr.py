"""Optional official IBKR market-data adapter.

This module deliberately exposes no trading or account methods. The official
``ibapi`` client is imported only when an IBKR recorder is configured, keeping
CI and replay independent from the optional dependency.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from stocker_prospective.config import RuntimeSafetyError
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
        self._loop_thread: threading.Thread | None = None
        self._client: Any | None = None
        self._stopping = threading.Event()
        self._connected = threading.Event()
        self._socket_preflight = socket_preflight

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
        forbidden = ("placeOrder", "cancelOrder", "reqOpenOrders", "reqGlobalCancel")
        if any(callable(getattr(client, name, None)) for name in forbidden):
            raise TypeError("order-capable wrapper cannot be attached to the recorder")
        self._client = client

    def start(self) -> None:
        require_official_ibkr_api()
        if self._client is None:
            raise RuntimeError("official IBKR callback client has not been attached")
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
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
        self._loop_thread = threading.Thread(
            target=self._client.run,
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
        self.callbacks.shutdown()
        self.stream_quotes.clear()
        for key in self.budget.shutdown():
            if key.isdigit():
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
        self._connected.clear()
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
        self._loop_thread = threading.Thread(
            target=self._client.run,
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
        key = str(request_id)
        self.budget.reserve(key)
        self.stream_quotes.register(request_id)
        if not self.subscriptions.register(subscription_key, request_id):
            self.stream_quotes.remove(request_id)
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
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
        key = str(request_id)
        self.budget.reserve(key)
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
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
        method()

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
        return None if not callable(method) else int(method())

    def _begin_stream_subscription(
        self,
        *,
        subscription_key: str,
        kind: str,
    ) -> tuple[int, int | None]:
        request_id = self.request_ids.next()
        key = str(request_id)
        self.budget.reserve(key)
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is None:
                raise RuntimeError("subscription registry changed during allocation")
            self.subscriptions.register(subscription_key, existing)
            return request_id, existing
        self._subscription_kinds[request_id] = kind
        return request_id, None

    def _abort_stream_subscription(self, request_id: int, subscription_key: str) -> None:
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
            self.budget.request_cancellation(key)
            if not broker_snapshot_complete:
                self._client.cancelMktData(request_id)
            self.budget.confirm_cancellation(key)

    def actual_subscription_request_ids(self) -> set[int]:
        """Expose only this adapter's request IDs for registry reconciliation."""

        return {request_id for _key, request_id in self.subscriptions.active_items()}

    def cancel_orphaned_market_data_request(self, request_id: int) -> None:
        """Repair a request that has no higher-level internal owner."""

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
            self.budget.confirm_cancellation(budget_key)

    def qualify_exact_contract(self, contract: Any) -> CallbackResult:
        """Qualify one exact bounded contract through ``reqContractDetails``."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqContractDetails", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
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
    ) -> None:
        """Receive a quote update without converting missing fields to zero."""

        if self.callbacks.is_pending(request_id):
            self.callbacks.add(request_id, payload)
            if complete:
                self.callbacks.complete(request_id)
            return
        self.stream_quotes.add(request_id, payload)
        self._append_stream_event("level1_quote_update", request_id, payload)

    def on_connected(self, market_data_type: MarketDataType | None) -> None:
        self.connection.connected(market_data_type)
        self._connected.set()

    def on_market_data_type(
        self,
        request_id: int,
        market_data_type: MarketDataType,
    ) -> None:
        self.connection.market_data_type_observed(market_data_type)
        self.on_quote_update(
            request_id,
            {
                "field": "market_data_type",
                "value": market_data_type.value,
                "market_data_type": market_data_type.value,
            },
        )

    def on_realtime_bar(self, update: RealtimeBarUpdate) -> None:
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
        self._append_stream_event(
            "current_time",
            -1,
            {"provider_timestamp_utc": provider_timestamp_utc.astimezone(UTC).isoformat()},
        )

    def on_depth_exchanges(self, exchanges: tuple[dict[str, Any], ...]) -> None:
        self._append_stream_event("depth_exchanges", -1, {"exchanges": exchanges})

    def _append_stream_event(
        self,
        kind: str,
        request_id: int,
        payload: dict[str, Any],
    ) -> None:
        with self._stream_event_lock:
            if len(self._stream_events) >= self._stream_event_limit:
                raise RuntimeError("bounded_ibkr_stream_event_queue_exhausted")
            self._stream_event_sequence += 1
            received_at = datetime.now(UTC)
            event = {
                **payload,
                "kind": kind,
                "request_id": request_id,
                "received_timestamp_utc": payload.get(
                    "receive_timestamp_utc",
                    received_at.isoformat(),
                ),
                "received_monotonic_ns": time.monotonic_ns(),
                "source_sequence": self._stream_event_sequence,
            }
            event.pop("receive_timestamp_utc", None)
            self._stream_events.append(event)

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
