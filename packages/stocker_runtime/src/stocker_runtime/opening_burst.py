"""Deterministic fake-adapter replay for the accepted XNYS opening burst."""

from __future__ import annotations

import math
import resource
import sqlite3
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from stocker_runtime.ingestion import (
    AdmissionResult,
    CallbackFence,
    InstrumentSpec,
    MarketDataCallback,
    MarketDataStatus,
    Recorder,
    RecorderConfig,
    SubscriptionSpec,
)
from stocker_runtime.ingestion.ibkr_market_data import IBKRSubscription
from stocker_runtime.market_session import xnys_session_window_us
from stocker_runtime.storage import connect_v2, initialize_database
from stocker_runtime.web import WebConfig
from stocker_runtime.web.queries import ReadModel

OPENING_BURST_SESSION = date(2026, 8, 10)
OPENING_BURST_FEEDS = 100
OPENING_BURST_SECONDS = 60
CI_OPENING_BURST_SECONDS = 10


class ReplayMarketData:
    """Minimal market-data-only adapter that records active logical requests."""

    capabilities = frozenset({"market_data"})

    def __init__(self) -> None:
        self.callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult] | None = None
        self.disconnect_callback: Callable[[int], None] | None = None
        self.status_callback: Callable[[MarketDataStatus], None] | None = None
        self.connected = False
        self.active_request_ids: set[int] = set()
        self.configured: tuple[IBKRSubscription, ...] = ()

    def set_callback(
        self,
        callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult],
    ) -> None:
        self.callback = callback

    def set_disconnect_callback(self, callback: Callable[[int], None]) -> None:
        self.disconnect_callback = callback

    def set_status_callback(self, callback: Callable[[MarketDataStatus], None]) -> None:
        self.status_callback = callback

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        self.configured = subscriptions

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.active_request_ids.clear()

    def subscribe(self, fence: CallbackFence) -> None:
        assert fence.request_id is not None
        if fence.request_id in self.active_request_ids:
            raise RuntimeError("duplicate active replay request")
        self.active_request_ids.add(fence.request_id)

    def retry_subscription(self, fence: CallbackFence) -> None:
        assert fence.request_id is not None
        self.active_request_ids.discard(fence.request_id)
        self.subscribe(fence)

    def cancel(self, request_id: int) -> None:
        self.active_request_ids.discard(request_id)


def _scenario(
    seconds: int,
) -> tuple[
    tuple[InstrumentSpec, ...],
    tuple[SubscriptionSpec, ...],
    tuple[tuple[str, int, int, int, MarketDataCallback], ...],
]:
    if seconds < 1 or seconds > OPENING_BURST_SECONDS:
        raise ValueError("opening replay seconds must be between 1 and 60")
    window = xnys_session_window_us(OPENING_BURST_SESSION)
    if window is None:
        raise RuntimeError("accepted XNYS replay session is unavailable")
    opened_at_us, _closed_at_us = window
    definitions = (("bars", 21, 1), ("quotes", 40, 3), ("trades", 39, 2))
    instruments: list[InstrumentSpec] = []
    subscriptions: list[SubscriptionSpec] = []
    callbacks: list[tuple[str, int, int, int, MarketDataCallback]] = []
    request_id = 1
    for feed_kind, feed_count, callbacks_per_second in definitions:
        for feed_index in range(feed_count):
            identity = f"{feed_kind}-{feed_index:02d}"
            instrument_id = f"OPEN-{identity}"
            instruments.append(
                InstrumentSpec(
                    instrument_id,
                    request_id,
                    "stock",
                    f"O{request_id:03d}",
                    "SMART",
                    "USD",
                )
            )
            subscriptions.append(
                SubscriptionSpec(
                    identity,
                    instrument_id,
                    feed_kind,
                    request_id,
                    True,
                    False,
                    15_000_000,
                )
            )
            for second in range(seconds):
                if feed_kind == "bars" and second % 5:
                    continue
                for ordinal in range(callbacks_per_second):
                    offset_us = ordinal * (1_000_000 // callbacks_per_second)
                    received_at_us = opened_at_us + second * 1_000_000 + offset_us
                    price = 100.0 + request_id / 1000 + second / 10000
                    if feed_kind == "bars":
                        callback_kind = "bar"
                        payload: dict[str, int | float] = {
                            "event_at_us": received_at_us,
                            "open": price,
                            "high": price + 0.02,
                            "low": price - 0.02,
                            "close": price + 0.01,
                            "volume": 1000 + second,
                        }
                    elif feed_kind == "quotes":
                        callback_kind = "quote"
                        payload = {
                            "event_at_us": received_at_us,
                            "bid": price,
                            "ask": price + 0.01,
                            "bid_size": 10 + ordinal,
                            "ask_size": 11 + ordinal,
                        }
                    else:
                        callback_kind = "trade"
                        payload = {
                            "event_at_us": received_at_us,
                            "last": price,
                            "size": 100 + ordinal,
                        }
                    callbacks.append(
                        (
                            identity,
                            ordinal,
                            second,
                            request_id,
                            MarketDataCallback(
                                callback_kind,
                                received_at_us,
                                received_at_us,
                                payload,
                            ),
                        )
                    )
            request_id += 1
    callbacks.sort(key=lambda item: (item[4].received_at_us, item[0], item[1]))
    return tuple(instruments), tuple(subscriptions), tuple(callbacks)


def opening_burst_callback_count(seconds: int) -> int:
    """Return the fixed scenario size without constructing callback payloads."""

    return 21 * math.ceil(seconds / 5) + 40 * 3 * seconds + 39 * 2 * seconds


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def run_opening_burst(database: Path, *, seconds: int) -> dict[str, Any]:
    """Run the bounded replay and return measurements before clean shutdown."""

    initialize_database(database)
    instruments, subscriptions, callbacks = _scenario(seconds)
    adapter = ReplayMarketData()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="opening-burst-replay",
            owner_id="opening-burst-simulator",
            mode="prospective_record",
            host="127.0.0.1",
            port=4001,
            client_id=71,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            market_data_line_limit=OPENING_BURST_FEEDS,
        ),
        adapter,
    )
    opened_at_us = callbacks[0][4].received_at_us
    state = recorder.start(
        now_us=opened_at_us,
        instruments=instruments,
        subscriptions=subscriptions,
    )
    fences = {fence.request_id: fence for fence in state.fences}
    latencies_ms: list[float] = []
    expected_order: list[tuple[int, int, str]] = []
    admitted = 0
    projected = 0
    max_backlog = 0
    drain_seconds = 0.0
    rss_before = _peak_rss_bytes()
    replay_started = time.perf_counter()
    current_second = callbacks[0][2]
    try:
        for _identity, _ordinal, second, request_id, callback in callbacks:
            if second != current_second:
                while recorder.inbox.nonterminal_count():
                    drain_started = time.perf_counter()
                    projected += recorder.drain(
                        now_us=callback.received_at_us,
                        limit=256,
                        defer_downstream_when_full=False,
                    )
                    drain_seconds += time.perf_counter() - drain_started
                current_second = second
            admission_started = time.perf_counter()
            result = recorder.receive(fences[request_id], callback)
            latencies_ms.append((time.perf_counter() - admission_started) * 1000)
            admitted += int(result.inserted)
            expected_order.append((request_id, callback.received_at_us, callback.callback_kind))
            max_backlog = max(max_backlog, recorder.inbox.nonterminal_count())
        admission_window_seconds = time.perf_counter() - replay_started
        drain_after_burst_started = time.perf_counter()
        backlog_before_final_drain = recorder.inbox.nonterminal_count()
        backlog_to_256_seconds: float | None = 0.0 if backlog_before_final_drain <= 256 else None
        while recorder.inbox.nonterminal_count():
            pass_started = time.perf_counter()
            projected += recorder.drain(
                now_us=callbacks[-1][4].received_at_us,
                limit=256,
                defer_downstream_when_full=False,
            )
            drain_seconds += time.perf_counter() - pass_started
            if backlog_to_256_seconds is None and recorder.inbox.nonterminal_count() <= 256:
                backlog_to_256_seconds = time.perf_counter() - drain_after_burst_started
        final_drain_seconds = time.perf_counter() - drain_after_burst_started
        final_backlog = recorder.inbox.nonterminal_count()
        last_received_at_us = callbacks[-1][4].received_at_us
        readiness_started = time.perf_counter()
        readiness = ReadModel(
            WebConfig(
                database=database,
                production=True,
                git_commit="deadbee",
                config_hash="b" * 64,
                allowed_hosts=["localhost"],
            )
        ).ready(now_us=last_received_at_us)
        readiness_query_ms = (time.perf_counter() - readiness_started) * 1000
        with connect_v2(database) as connection:
            rows = tuple(
                connection.execute(
                    "SELECT source_sequence, request_id, received_at_us, callback_kind, "
                    "event_uid, run_id, recorder_generation, connection_generation "
                    "FROM callback_inbox ORDER BY source_sequence"
                )
            )
            durable_count = len(rows)
            distinct_count = int(
                connection.execute(
                    "SELECT count(DISTINCT event_uid) FROM callback_inbox"
                ).fetchone()[0]
            )
            projected_count = int(
                connection.execute("SELECT count(*) FROM market_events").fetchone()[0]
            )
            provenance_violations = int(
                connection.execute(
                    "SELECT count(*) FROM callback_inbox callback LEFT JOIN subscriptions sub "
                    "ON sub.run_id=callback.run_id "
                    "AND sub.recorder_generation=callback.recorder_generation "
                    "AND sub.connection_generation=callback.connection_generation "
                    "AND sub.request_id=callback.request_id LEFT JOIN market_events event "
                    "ON event.run_id=callback.run_id "
                    "AND event.source_sequence=callback.source_sequence "
                    "WHERE sub.subscription_id IS NULL OR event.event_id IS NULL "
                    "OR callback.run_id!='opening-burst-replay' OR callback.recorder_generation!=? "
                    "OR callback.connection_generation!=? "
                    "OR event.instrument_id!=sub.instrument_id "
                    "OR event.feed_kind!=sub.feed_kind",
                    (state.recorder_generation, state.connection_generation),
                ).fetchone()[0]
            )
            fresh_required = int(
                connection.execute(
                    "SELECT count(*) FROM (SELECT sub.subscription_id FROM subscriptions sub "
                    "JOIN callback_inbox callback ON callback.run_id=sub.run_id "
                    "AND callback.recorder_generation=sub.recorder_generation "
                    "AND callback.connection_generation=sub.connection_generation "
                    "AND callback.request_id=sub.request_id "
                    "WHERE sub.optional=0 AND sub.lifecycle='active' "
                    "GROUP BY sub.subscription_id HAVING max(callback.received_at_us)>=?)",
                    (last_received_at_us - 15_000_000,),
                ).fetchone()[0]
            )
            heartbeat = int(
                connection.execute(
                    "SELECT process_heartbeat_at_us FROM runtime_state "
                    "WHERE run_id='opening-burst-replay'"
                ).fetchone()[0]
            )
        ordering_violations = sum(
            1
            for index, (row, expected) in enumerate(zip(rows, expected_order, strict=True), start=1)
            if int(row["source_sequence"]) != index
            or (int(row["request_id"]), int(row["received_at_us"]), str(row["callback_kind"]))
            != expected
        )
        rss_growth = max(0, _peak_rss_bytes() - rss_before)
        return {
            "seconds": seconds,
            "presented": len(callbacks),
            "admitted": admitted,
            "durable_callbacks": durable_count,
            "projected": projected_count,
            "missing": len(callbacks) - durable_count,
            "duplicate_durable_callbacks": durable_count - distinct_count,
            "ordering_violations": ordering_violations,
            "provenance_violations": provenance_violations,
            "escaped_sqlite_busy_locked_errors": 0,
            "admission_p50_ms": _percentile(latencies_ms, 0.50),
            "admission_p95_ms": _percentile(latencies_ms, 0.95),
            "admission_p99_ms": _percentile(latencies_ms, 0.99),
            "admission_throughput_per_second": len(callbacks) / admission_window_seconds,
            "projection_throughput_per_second": projected / max(drain_seconds, 1e-9),
            "maximum_durable_inbox_backlog": max_backlog,
            "final_durable_inbox_backlog": final_backlog,
            "backlog_to_256_seconds": (
                final_drain_seconds if backlog_to_256_seconds is None else backlog_to_256_seconds
            ),
            "backlog_to_zero_seconds": final_drain_seconds,
            "recorder_heartbeat_delay_seconds": max(
                0.0, (last_received_at_us - heartbeat) / 1_000_000
            ),
            "required_feeds_fresh": fresh_required,
            "required_feeds_expected": OPENING_BURST_FEEDS,
            "healthy_feeds_active": len(adapter.active_request_ids),
            "readiness_ready": bool(readiness["ready"]),
            "readiness_query_ms": readiness_query_ms,
            "rss_growth_bytes": rss_growth,
        }
    except sqlite3.OperationalError as error:
        if "busy" in str(error).lower() or "locked" in str(error).lower():
            raise RuntimeError("SQLite busy/locked escaped opening replay") from error
        raise
    finally:
        if recorder.state is not None:
            recorder.stop(now_us=callbacks[-1][4].received_at_us + 1)


def opening_burst_failures(result: dict[str, Any], *, include_performance: bool) -> tuple[str, ...]:
    """Evaluate the frozen pre-baseline acceptance thresholds."""

    failures: list[str] = []
    expected = opening_burst_callback_count(int(result["seconds"]))
    exact_zero = (
        "missing",
        "duplicate_durable_callbacks",
        "ordering_violations",
        "provenance_violations",
        "escaped_sqlite_busy_locked_errors",
        "final_durable_inbox_backlog",
    )
    for field in exact_zero:
        if int(result[field]) != 0:
            failures.append(f"{field}={result[field]}")
    for field in ("presented", "admitted", "durable_callbacks", "projected"):
        if int(result[field]) != expected:
            failures.append(f"{field}={result[field]} expected={expected}")
    if int(result["maximum_durable_inbox_backlog"]) > 5_000:
        failures.append("maximum_durable_inbox_backlog>5000")
    if float(result["backlog_to_256_seconds"]) > 15:
        failures.append("backlog_to_256_seconds>15")
    if float(result["backlog_to_zero_seconds"]) > 30:
        failures.append("backlog_to_zero_seconds>30")
    if float(result["recorder_heartbeat_delay_seconds"]) > 5:
        failures.append("recorder_heartbeat_delay_seconds>5")
    for field in ("required_feeds_fresh", "healthy_feeds_active"):
        if int(result[field]) != OPENING_BURST_FEEDS:
            failures.append(f"{field}={result[field]} expected={OPENING_BURST_FEEDS}")
    if not bool(result["readiness_ready"]):
        failures.append("readiness_ready=false")
    if include_performance:
        for field, threshold in (
            ("admission_p50_ms", 5),
            ("admission_p95_ms", 15),
            ("admission_p99_ms", 50),
        ):
            if float(result[field]) > threshold:
                failures.append(f"{field}>{threshold}")
        if float(result["admission_throughput_per_second"]) < 200:
            failures.append("admission_throughput_per_second<200")
        if float(result["projection_throughput_per_second"]) < 500:
            failures.append("projection_throughput_per_second<500")
        if int(result["rss_growth_bytes"]) > 64 * 1024**2:
            failures.append("rss_growth_bytes>67108864")
    return tuple(failures)
