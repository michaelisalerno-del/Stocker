from __future__ import annotations

import sqlite3
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
) -> tuple[IBKRMarketDataAdapter, DurableCallbackInbox]:
    inbox = DurableCallbackInbox(
        database(tmp_path),
        max_unacknowledged=max_unacknowledged,
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
    assert accounting.admitted == 2
    assert accounting.pending == 1


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
            WHERE callback_kind = 'official_provider_tick_price'
            """
        ).fetchone()
    encoded = str(row["original_payload_json"])
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
