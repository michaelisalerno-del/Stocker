from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import stocker_prospective.ibkr as ibkr_module
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter
from stocker_prospective.market_data import MarketDataBudget, MarketDataType

NOW = datetime(2026, 7, 29, 15, 0, tzinfo=UTC)
RUN_ID = "run-callback-containment"


def database(tmp_path: Path) -> Path:
    path = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(path)
    repository.migrate()
    repository.create_run(
        EvidenceMetadata(
            run_id=RUN_ID,
            prospective_start_utc=NOW - timedelta(days=1),
            app_version="test",
            git_commit="a" * 40,
            model_artifact_id="frozen-m1c",
            universe_id="anchor-frozen-20",
            cohort="anchor_frozen_20",
            source_timestamps=[NOW.isoformat()],
            recorded_at_utc=NOW,
        )
    )
    return path


def adapter(
    tmp_path: Path,
    *,
    max_unacknowledged: int = 100,
    busy_timeout_ms: int = 5_000,
) -> tuple[IBKRMarketDataAdapter, DurableCallbackInbox]:
    inbox = DurableCallbackInbox(
        database(tmp_path),
        max_unacknowledged=max_unacknowledged,
        busy_timeout_ms=busy_timeout_ms,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )
    value = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=91,
            expected_environment="read_only",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=20,
            reserved_headroom=1,
            request_rate_limit=100,
        ),
        durable_inbox=inbox,
    )
    value._connection_generation = 1
    return value, inbox


def activate(value: IBKRMarketDataAdapter, request_id: int = 7) -> None:
    value._track_request(request_id, "AAL:level1")
    value._subscription_kinds[request_id] = "market_data"
    value.stream_quotes.register(request_id)
    value.register_stream_owner(
        request_id,
        {
            "request_id": request_id,
            "kind": "underlying_level1",
            "symbol": "AAL",
            "con_id": 123,
            "exchange": "SMART",
            "episode_id": None,
            "option_contract": None,
        },
    )


def classifications(inbox: DurableCallbackInbox) -> tuple[str, ...]:
    with inbox._connect() as connection:
        rows = connection.execute(
            """
            SELECT callback_classification
            FROM callback_inbox_v1
            ORDER BY source_sequence
            """
        ).fetchall()
    return tuple(str(row["callback_classification"]) for row in rows)


def scientific_callback_count(inbox: DurableCallbackInbox) -> int:
    with inbox._connect() as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM callback_inbox_v1
                WHERE callback_kind NOT LIKE 'official_provider_%'
                """
            ).fetchone()[0]
        )


def test_canonical_event_preserves_external_callback_boundary_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, inbox = adapter(tmp_path)
    boundary_received_at = NOW
    delayed_materialisation_at = NOW + timedelta(seconds=200)
    wall_clock_values = iter(
        (
            boundary_received_at,
            delayed_materialisation_at,
            delayed_materialisation_at + timedelta(seconds=1),
        )
    )
    monotonic_values = iter((1_000_000_000, 201_000_000_000))

    class SequencedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            value = next(wall_clock_values)
            return value if tz is not None else value.replace(tzinfo=None)

    class SequencedTime:
        @staticmethod
        def monotonic_ns() -> int:
            return next(monotonic_values)

    monkeypatch.setattr(ibkr_module, "datetime", SequencedDateTime)
    monkeypatch.setattr(ibkr_module, "time", SequencedTime)

    value.contain_official_callback(
        "current_time",
        -1,
        lambda: value.on_current_time(boundary_received_at),
        provider_arguments=(int(boundary_received_at.timestamp()),),
    )

    with inbox._connect() as connection:
        rows = connection.execute(
            """
            SELECT callback_kind, received_utc, received_monotonic_ns
            FROM callback_inbox_v1
            WHERE callback_kind IN ('official_provider_current_time', 'current_time')
            ORDER BY source_sequence
            """
        ).fetchall()

    assert len(rows) == 2
    assert tuple(str(row["received_utc"]) for row in rows) == (
        boundary_received_at.isoformat(),
        boundary_received_at.isoformat(),
    )
    assert tuple(int(row["received_monotonic_ns"]) for row in rows) == (
        1_000_000_000,
        1_000_000_000,
    )


def test_current_time_request_boundary_is_durably_admitted(tmp_path: Path) -> None:
    value, inbox = adapter(tmp_path)

    class ClockClient:
        def reqCurrentTime(self) -> None:  # noqa: N802
            return None

    value._client = ClockClient()
    value.request_current_time()
    provider_at = datetime.now(UTC).replace(microsecond=0)
    value.contain_official_callback(
        "current_time",
        -1,
        lambda: value.on_current_time(provider_at),
        provider_arguments=(int(provider_at.timestamp()),),
    )

    with inbox._connect() as connection:
        row = connection.execute(
            """
            SELECT original_payload_json, received_utc
            FROM callback_inbox_v1
            WHERE callback_kind = 'current_time'
            """
        ).fetchone()

    assert row is not None
    payload = json.loads(str(row["original_payload_json"]))
    assert payload["provider_timestamp_utc"] == provider_at.isoformat()
    assert datetime.fromisoformat(payload["clock_probe_requested_at_utc"]) <= (
        datetime.fromisoformat(str(row["received_utc"]))
    )
    assert payload["clock_probe_requested_monotonic_ns"] > 0


def test_hot_callback_is_durable_before_normalization_or_cache_update(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)

    def normalize_after_durable_admission() -> None:
        with inbox._connect() as connection:
            row = connection.execute(
                """
                SELECT callback_kind, status
                FROM callback_inbox_v1
                ORDER BY source_sequence DESC
                LIMIT 1
                """
            ).fetchone()
        assert row is not None
        assert row["callback_kind"] == "official_provider_tick_price"
        assert row["status"] == "provider_pending"
        assert value.stream_quotes.snapshot(7) == ()
        value.on_quote_update(7, {"field": "bid", "value": 10.0})

    value.contain_official_callback(
        "tick_price",
        7,
        normalize_after_durable_admission,
        provider_arguments=(7, 1, 10.0, {}),
    )

    assert value.fatal_callback_code is None
    assert scientific_callback_count(inbox) == 1
    assert value.stream_quotes.snapshot(7) == ({"field": "bid", "value": 10.0},)


def test_unknown_request_is_quarantined_and_never_escapes_boundary(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)

    value.contain_official_callback(
        "tick_price",
        999,
        lambda: value.on_quote_update(999, {"field": "bid", "value": 10.0}),
    )

    assert value.fatal_callback_code == "CALLBACK_UNKNOWN_REQUEST_ID"
    assert inbox.has_active_fatal("ingestion")
    assert classifications(inbox) == ("unknown_callback",)
    assert inbox.accounting().quarantined == 1


def test_late_callback_after_cancellation_is_diagnostic_not_active(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    value._tombstone_request(7, "replaced")
    value._subscription_kinds.pop(7)

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    assert value.scientific_recording_valid
    assert classifications(inbox) == ("expected_late_callback_after_cancellation",)
    assert inbox.accounting().diagnostic == 1
    assert value.stream_quotes.snapshot(7) == ()


def test_callback_from_previous_connection_generation_is_diagnostic(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    value._connection_generation = 2

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    assert value.scientific_recording_valid
    assert classifications(inbox) == ("callback_from_previous_connection_generation",)
    assert inbox.accounting().diagnostic == 1


def test_informational_error_callback_is_control_not_unknown_request(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)

    value.contain_official_callback(
        "error",
        999,
        lambda: value.on_error(999, 2104, "market data farm is connected"),
        provider_arguments=(999, 0, 2104, "market data farm is connected"),
    )

    assert value.fatal_callback_code is None
    assert classifications(inbox) == ("control_callback",)
    assert inbox.accounting().diagnostic == 1


def test_durable_inbox_exhaustion_latches_fatal_without_silent_drop(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path, max_unacknowledged=1)
    activate(value)
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    value.contain_official_callback(
        "tick_size",
        7,
        lambda: value.on_quote_update(7, {"field": "bid_size", "value": 100.0}),
    )

    assert value.fatal_callback_code == "CALLBACK_OVERFLOW"
    assert inbox.has_active_fatal("ingestion")
    accounting = inbox.accounting()
    assert accounting.admitted == 1
    assert accounting.pending == 1


def test_transient_sqlite_writer_contention_does_not_latch_callback_loss(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path, busy_timeout_ms=10)
    activate(value)
    lock_acquired = threading.Event()

    def hold_writer_lock() -> None:
        with sqlite3.connect(inbox.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            lock_acquired.set()
            time.sleep(0.05)

    lock_thread = threading.Thread(target=hold_writer_lock)
    lock_thread.start()
    assert lock_acquired.wait(timeout=1)

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )
    lock_thread.join(timeout=1)

    assert not lock_thread.is_alive()
    assert value.fatal_callback_code is None
    assert value.scientific_recording_valid
    accounting = inbox.accounting()
    assert accounting.admitted == 1
    assert accounting.pending == 1
    assert accounting.diagnostic == 0
    assert not inbox.has_active_fatal("ingestion")


def test_callback_reader_reuses_sqlite_connection_and_keeps_up_with_stream_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_connect = sqlite3.connect
    connection_count = 0

    def slow_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal connection_count
        connection_count += 1
        time.sleep(0.075)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", slow_connect)
    value, inbox = adapter(tmp_path)
    setup_connection_count = connection_count
    request_ids = tuple(range(100, 108))
    for request_id in request_ids:
        activate(value, request_id)
    payload = {
        "date": "20260803 14:00:00",
        "open": 10.0,
        "high": 10.1,
        "low": 9.9,
        "close": 10.05,
        "volume": 1_000,
    }

    started = time.perf_counter()
    for request_id in request_ids:
        value.contain_official_callback(
            "historical_data_update",
            request_id,
            lambda request_id=request_id: value.on_historical_bar(
                request_id,
                payload,
                update=True,
            ),
            provider_arguments=(request_id, payload),
        )
    elapsed_seconds = time.perf_counter() - started

    callback_rate_hz = len(request_ids) / elapsed_seconds
    required_stream_rate_hz = 28 / 5
    assert callback_rate_hz >= required_stream_rate_hz
    assert connection_count - setup_connection_count <= 1
    assert value.fatal_callback_code is None
    assert scientific_callback_count(inbox) == len(request_ids)


def test_cached_inbox_connection_is_owned_by_callback_thread(tmp_path: Path) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    payload = {
        "date": "20260803 14:00:00",
        "open": 10.0,
        "high": 10.1,
        "low": 9.9,
        "close": 10.05,
        "volume": 1_000,
    }

    callback_thread = threading.Thread(
        target=lambda: value.contain_official_callback(
            "historical_data_update",
            7,
            lambda: value.on_historical_bar(7, payload, update=True),
            provider_arguments=(7, payload),
        )
    )
    callback_thread.start()
    callback_thread.join(timeout=1)

    assert not callback_thread.is_alive()
    assert value.fatal_callback_code is None
    assert scientific_callback_count(inbox) == 1
    accounting = inbox.accounting()
    assert accounting.admitted == 1
    assert accounting.pending == 1
    assert accounting.diagnostic == 0


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("not-a-market-number"), "CALLBACK_MALFORMED_VALUE"),
        (sqlite3.OperationalError("database is locked"), "CALLBACK_DURABLE_ADMISSION_FAILED"),
    ],
)
def test_malformed_or_admission_failure_never_escapes(
    tmp_path: Path,
    error: Exception,
    expected_code: str,
) -> None:
    value, _ = adapter(tmp_path)
    activate(value)

    value.contain_official_callback(
        "tick_size",
        7,
        lambda: (_ for _ in ()).throw(error),
    )

    assert value.fatal_callback_code == expected_code
    assert not value.scientific_recording_valid


def test_cache_failure_occurs_after_durable_admission_and_latches_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)

    def broken_cache(_: int, __: dict[str, Any]) -> None:
        raise RuntimeError("cache_failed")

    monkeypatch.setattr(value.stream_quotes, "add", broken_cache)
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    assert value.fatal_callback_code == "CALLBACK_CACHE_FAILURE"
    assert inbox.accounting().pending == 1
    assert inbox.accounting().acknowledged == 0
    assert inbox.has_active_fatal("ingestion")


def test_unexpected_failure_inside_failure_classifier_still_cannot_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, _ = adapter(tmp_path)

    def broken_classifier(**_keywords: Any) -> None:
        raise RuntimeError("classifier_failed")

    monkeypatch.setattr(value, "_latch_callback_failure", broken_classifier)

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: (_ for _ in ()).throw(RuntimeError("callback_failed")),
    )

    assert value.fatal_callback_code == "CALLBACK_BOUNDARY_FAILURE"
    assert not value.scientific_recording_valid


def test_identical_payloads_without_provider_delivery_id_are_distinct(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    payload = {
        "field": "bid",
        "value": 10.0,
        "provider_timestamp_utc": NOW.isoformat(),
        "provider_event_id": "synthetic-provider-delivery-1",
    }

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, payload),
    )
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, payload),
    )

    assert scientific_callback_count(inbox) == 2
    # The bounded convenience cache may coalesce identical values, but the
    # durable scientific inbox must retain both provider deliveries.
    assert len(value.stream_quotes.snapshot(7)) == 1
    with inbox._connect() as connection:
        scientific_sequences = connection.execute(
            """
                SELECT source_sequence
                FROM callback_inbox_v1
                WHERE callback_kind NOT LIKE 'official_provider_%'
                ORDER BY source_sequence
            """
        ).fetchall()
    assert len({int(row[0]) for row in scientific_sequences}) == 2


def test_identical_official_provider_deliveries_each_receive_a_sequence(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    payload = {"field": "bid", "value": 10.0}

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, payload),
    )
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, payload),
    )

    assert scientific_callback_count(inbox) == 2
    assert value.stream_quotes.snapshot(7) == (payload,)


def test_nonfinite_provider_value_is_durably_preserved_for_poison_handling(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": float("nan")}),
    )

    leased = inbox.lease(
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=30),
        limit=1,
    )
    assert leased[0].original_payload["value"] == {"__non_finite_float__": "nan"}
    assert value.fatal_callback_code is None


def test_payload_provider_identity_cannot_collapse_official_deliveries(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(
            7,
            {"field": "bid", "value": 10.0, "provider_event_id": "provider-1"},
        ),
        provider_arguments=(7, 1, 10.0),
    )
    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(
            7,
            {"field": "bid", "value": 11.0, "provider_event_id": "provider-1"},
        ),
        provider_arguments=(7, 1, 11.0),
    )

    assert scientific_callback_count(inbox) == 2
    assert value.fatal_callback_code is None
    assert not inbox.has_active_fatal("ingestion")


def test_official_provider_envelope_is_not_depth_or_collection_truncated(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    provider_argument = {
        "nested": {"one": {"two": {"three": {"four": {"five": "complete"}}}}},
        "items": list(range(300)),
    }

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
        provider_arguments=(7, provider_argument),
    )

    with inbox._connect() as connection:
        row = connection.execute(
            """
            SELECT original_payload_json
            FROM callback_inbox_v1
            WHERE callback_kind = 'level1_quote_update'
            """
        ).fetchone()
    assert row is not None
    payload = json.loads(str(row["original_payload_json"]))
    encoded = json.dumps(
        payload["original_provider_callback"],
        separators=(",", ":"),
    )
    assert "__truncated_type__" not in encoded
    assert '"five":"complete"' in encoded
    assert '"items":[0,1,2' in encoded
    assert ",299]" in encoded


def test_unserialisable_provider_cycle_is_contained_and_latched(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    cyclic: list[object] = []
    cyclic.append(cyclic)

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
        provider_arguments=(7, cyclic),
    )

    assert value.fatal_callback_code == "CALLBACK_MALFORMED_VALUE"
    assert inbox.has_active_fatal("ingestion")
    assert scientific_callback_count(inbox) == 0


def test_callback_after_data_loss_latch_is_quarantined_and_auditable(
    tmp_path: Path,
) -> None:
    value, inbox = adapter(tmp_path)
    activate(value)
    value.contain_official_callback(
        "tick_price",
        999,
        lambda: value.on_quote_update(999, {"field": "bid", "value": 10.0}),
    )

    value.contain_official_callback(
        "tick_price",
        7,
        lambda: value.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    assert classifications(inbox) == (
        "unknown_callback",
        "callback_after_data_loss_latch",
    )
    assert inbox.accounting().quarantined == 2
    assert value.stream_quotes.snapshot(7) == ()
    assert not value.scientific_recording_valid


def test_restarted_adapter_restores_persisted_ingestion_latch(
    tmp_path: Path,
) -> None:
    first, inbox = adapter(tmp_path)
    activate(first)
    first.contain_official_callback(
        "tick_price",
        999,
        lambda: first.on_quote_update(999, {"field": "bid", "value": 10.0}),
    )
    assert inbox.has_active_fatal("ingestion")

    restarted, restarted_inbox = adapter(tmp_path)
    activate(restarted)
    restarted.contain_official_callback(
        "tick_price",
        7,
        lambda: restarted.on_quote_update(7, {"field": "bid", "value": 10.0}),
    )

    assert restarted.fatal_callback_code == "CALLBACK_UNKNOWN_REQUEST_ID"
    assert classifications(restarted_inbox)[-1] == "callback_after_data_loss_latch"


def test_restarted_adapter_completes_bounded_bootstrap_request_under_latch(
    tmp_path: Path,
) -> None:
    first, inbox = adapter(tmp_path)
    activate(first)
    first.contain_official_callback(
        "tick_price",
        999,
        lambda: first.on_quote_update(999, {"field": "bid", "value": 10.0}),
    )
    assert inbox.has_active_fatal("ingestion")

    restarted, restarted_inbox = adapter(tmp_path)
    restarted._track_request(7, "exact_contract_qualification:7")
    restarted.callbacks.begin(7, kind="exact_contract_qualification")
    restarted.contain_official_callback(
        "contract_details",
        7,
        lambda: restarted.on_contract_details(7, {"symbol": "AAL", "con_id": 1}),
    )
    restarted.contain_official_callback(
        "contract_details_end",
        7,
        lambda: restarted.on_contract_details_end(7),
    )

    result = restarted.callbacks.wait(7, timeout_seconds=0.01)

    assert result.complete is True
    assert result.items == ({"symbol": "AAL", "con_id": 1},)
    assert classifications(restarted_inbox)[-2:] == (
        "callback_after_data_loss_latch",
        "callback_after_data_loss_latch",
    )
    assert restarted_inbox.accounting().quarantined == 1
    assert restarted.fatal_callback_code == "CALLBACK_UNKNOWN_REQUEST_ID"


def test_restarted_adapter_quarantines_failed_bounded_bootstrap_callback(
    tmp_path: Path,
) -> None:
    first, inbox = adapter(tmp_path)
    activate(first)
    first.contain_official_callback(
        "tick_price",
        999,
        lambda: first.on_quote_update(999, {"field": "bid", "value": 10.0}),
    )
    assert inbox.has_active_fatal("ingestion")

    restarted, restarted_inbox = adapter(tmp_path)
    restarted._track_request(7, "exact_contract_qualification:7")
    restarted.callbacks.begin(7, kind="exact_contract_qualification")

    def fail_bootstrap_callback() -> None:
        raise ValueError("malformed contract details")

    restarted.contain_official_callback(
        "contract_details",
        7,
        fail_bootstrap_callback,
    )

    assert restarted_inbox.accounting().quarantined == 2
    assert restarted.fatal_callback_code == "CALLBACK_UNKNOWN_REQUEST_ID"
    with restarted_inbox._connect() as connection:
        failure = connection.execute(
            """
            SELECT failure_classification
            FROM callback_inbox_v1
            WHERE callback_kind = 'official_provider_contract_details'
            ORDER BY source_sequence DESC LIMIT 1
            """
        ).fetchone()
    assert failure["failure_classification"] == "CALLBACK_MALFORMED_VALUE"
