from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import stocker_runtime.ingestion.lifecycle as lifecycle_module
from stocker_runtime.cli import app
from stocker_runtime.ingestion import (
    AdmissionResult,
    AuthoritativeLeaseLost,
    CallbackFence,
    CallbackIdentityCollision,
    CallbackInbox,
    DuplicateWriterError,
    InboxAdmissionError,
    InboxFullError,
    InstrumentSpec,
    MarketDataCallback,
    MarketDataStatus,
    NormalizationError,
    Recorder,
    RecorderConfig,
    RecorderFatalError,
    SubscriptionSpec,
    WriterAuthority,
    market_data_input_hash,
)
from stocker_runtime.ingestion.dynamic_market_data import OptionDiscoveryBackend
from stocker_runtime.ingestion.inbox import (
    LeasedCallback,
    ProjectionBatchResult,
    transport_incident_id,
)
from stocker_runtime.ingestion.lifecycle import (
    LocalWriterLock,
    LocalWriterLockError,
    PayloadDrainIncompleteError,
    drain_callback_payloads,
    recover_fatal_generation,
)
from stocker_runtime.storage import (
    MaintenanceDeadlineExceeded,
    RetentionPolicy,
    RetentionResult,
    StorageCapState,
    connect_v2,
    initialize_database,
    migrate_database,
    migration_plan,
)


def _config(database: Path, **changes: object) -> RecorderConfig:
    values: dict[str, object] = {
        "database": database,
        "run_id": "run-1",
        "owner_id": "owner-1",
        "mode": "prospective_record",
        "host": "127.0.0.1",
        "port": 4001,
        "client_id": 71,
        "read_only": True,
        "external_read_only_verified": True,
        "config_hash": "a" * 64,
        "git_commit": "deadbee",
    }
    values.update(changes)
    return RecorderConfig.model_validate(values)


def _seed_generation(database: Path) -> None:
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, git_commit, "
            "data_class, status) VALUES ('run-1', 'prospective_record', 'ibkr', 1, ?, "
            "'deadbee', 'prospective_protected', 'running')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 1, 'owner-1', 1)"
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, "
            "process_heartbeat_at_us, connection_generation) "
            "VALUES ('run-1', 1, 'running', 1, 2)"
        )


def _fence() -> CallbackFence:
    return CallbackFence(
        run_id="run-1",
        recorder_generation=1,
        connection_generation=2,
        request_id=3,
        subscription_id=None,
    )


def _authority() -> WriterAuthority:
    return WriterAuthority("run-1", 1, "owner-1")


def _seed_subscription(database: Path) -> CallbackFence:
    fence = CallbackFence("run-1", 1, 2, 3, "subscription-1")
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES ('instrument-1', ?, 123, 'stock', 'AAPL', 'SMART', 'USD')",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('subscription-1', 'run-1', 1, 2, 'instrument-1', 'quotes', 3, 'active', ?, 1)",
            ("c" * 64,),
        )
    return fence


def test_recorder_config_rejects_unsafe_mode_host_and_read_only_state(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    for changes in (
        {"mode": "paper"},
        {"mode": "live"},
        {"host": "192.0.2.1"},
        {"read_only": False},
        {"external_read_only_verified": False},
        {"writer_lease_stale_us": 14_999_999},
    ):
        with pytest.raises(ValidationError):
            _config(database, **changes)


def test_callback_is_durable_before_admission_returns_and_exact_retry_is_stable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)
    callback = MarketDataCallback(
        callback_kind="quote",
        received_at_us=10,
        provider_at_us=9,
        payload={"event_at_us": 9, "bid": 100.0, "ask": 100.5},
    )

    first = inbox.admit(_fence(), callback)
    second = inbox.admit(_fence(), callback)

    assert first.inserted is True
    assert second == first.__class__(
        source_sequence=first.source_sequence,
        event_uid=first.event_uid,
        inserted=False,
    )
    with connect_v2(database) as connection:
        row = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence = ?",
            (first.source_sequence,),
        ).fetchone()
    assert row["lifecycle"] == "pending"
    assert json.loads(row["payload_json"])["bid"] == 100.0


def test_inbox_hard_boundary_refuses_the_50001st_nonterminal_callback(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database, max_nonterminal_rows=2)
    inbox.admit(_fence(), MarketDataCallback("quote", 10, None, {"event_at_us": 10}))
    inbox.admit(_fence(), MarketDataCallback("quote", 11, None, {"event_at_us": 11}))

    with pytest.raises(InboxFullError):
        inbox.admit(_fence(), MarketDataCallback("quote", 12, None, {"event_at_us": 12}))

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending', 'leased')"
            ).fetchone()[0]
            == 2
        )


def test_default_inbox_hard_boundary_is_exactly_50000_rows(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, 'run-1', 1, 2, 'quote', ?, '{}', ?, "
            "'pending')",
            ((f"event-{sequence}", sequence, f"{sequence:064x}") for sequence in range(1, 50_001)),
        )

    with pytest.raises(InboxFullError):
        CallbackInbox(database).admit(
            _fence(),
            MarketDataCallback("quote", 50_001, None, {"event_at_us": 50_001}),
        )


def test_exact_duplicate_at_50000_rows_is_a_nonaction_not_an_overflow(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)
    callback = MarketDataCallback("quote", 1, None, {"event_at_us": 1, "bid": 100.0})
    first = inbox.admit(_fence(), callback)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, 'run-1', 1, 2, 'quote', ?, '{}', ?, "
            "'pending')",
            (
                (f"capacity-{sequence}", sequence, f"{sequence:064x}")
                for sequence in range(2, 50_001)
            ),
        )

    duplicate = inbox.admit(_fence(), callback)

    assert duplicate == replace(first, inserted=False)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_inbox").fetchone()[0] == 50_000
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
        assert connection.execute("SELECT count(*) FROM incidents").fetchone()[0] == 0


def test_concurrent_admission_serializes_the_exact_50000_row_boundary(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, 'run-1', 1, 2, 'quote', ?, '{}', ?, "
            "'pending')",
            (
                (f"capacity-{sequence}", sequence, f"{sequence:064x}")
                for sequence in range(1, 50_000)
            ),
        )
    inbox = CallbackInbox(database)
    start = threading.Barrier(2)

    def admit(value: float) -> object:
        start.wait(timeout=5)
        try:
            return inbox.admit(
                _fence(),
                MarketDataCallback("quote", 50_000, None, {"event_at_us": 50_000, "bid": value}),
            )
        except InboxFullError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(admit, (100.0, 101.0)))

    assert sum(isinstance(result, InboxFullError) for result in results) == 1
    assert sum(isinstance(result, AdmissionResult) for result in results) == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_inbox").fetchone()[0] == 50_000
        assert (
            connection.execute("SELECT max(source_sequence) FROM callback_inbox").fetchone()[0]
            == 50_000
        )
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == ("fatal", "INBOX_FULL", "disconnected")


def test_public_ingestion_surface_has_no_broker_authority() -> None:
    from stocker_runtime.ingestion.ibkr_market_data import IBKRMarketData

    public_names = {
        name.lower()
        for name, _ in inspect.getmembers(IBKRMarketData, predicate=callable)
        if not name.startswith("_")
    }
    forbidden = ("order", "account", "position", "execution", "pnl", "fill")
    assert not any(term in name for name in public_names for term in forbidden)


def test_validate_config_cli_is_offline_and_machine_readable(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    inputs_path = tmp_path / "market-data.json"
    config_path.write_text(
        json.dumps(_config(tmp_path / "v2.sqlite3").model_dump(mode="json")),
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "instrument_id": "instrument-1",
                        "ibkr_con_id": 123,
                        "kind": "stock",
                        "symbol": "AAPL",
                        "exchange": "SMART",
                        "currency": "USD",
                    }
                ],
                "subscriptions": [
                    {
                        "name": "required-quotes",
                        "instrument_id": "instrument-1",
                        "feed_kind": "quotes",
                        "request_id": 3,
                        "continuity_required": True,
                        "optional": False,
                        "stale_after_us": 15_000_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["validate-recorder", str(config_path), "--inputs", str(inputs_path)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "host": "127.0.0.1",
        "mode": "prospective_record",
        "read_only": True,
        "required_subscriptions": 1,
        "status": "ok",
        "subscriptions": 1,
    }


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("zero_instruments", "zero instruments"),
        ("zero_subscriptions", "zero subscriptions"),
        ("zero_required", "zero required subscriptions"),
        ("duplicate_instrument", "duplicate instrument identities"),
        ("duplicate_request", "duplicate request identities"),
        ("duplicate_subscription", "duplicate subscription identities"),
        ("missing_instrument", "reference missing instruments"),
        ("contradictory_feed", "contradictory feed definitions"),
        ("malformed_identity", "request_id must be a nonnegative integer"),
        ("base_snapshot", "cannot configure base snapshot subscriptions"),
    ),
)
def test_production_preflight_rejects_invalid_market_data_input_before_connect(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    config_path = tmp_path / "runtime.json"
    inputs_path = tmp_path / "market-data.json"
    config_path.write_text(
        json.dumps(_config(tmp_path / "v2.sqlite3").model_dump(mode="json")),
        encoding="utf-8",
    )
    instrument = {
        "instrument_id": "instrument-1",
        "ibkr_con_id": 123,
        "kind": "stock",
        "symbol": "AAPL",
        "exchange": "SMART",
        "currency": "USD",
    }
    subscription = {
        "name": "required-quotes",
        "instrument_id": "instrument-1",
        "feed_kind": "quotes",
        "request_id": 3,
        "continuity_required": True,
        "optional": False,
        "stale_after_us": 15_000_000,
    }
    instruments = [instrument]
    subscriptions = [subscription]
    if mutation == "zero_instruments":
        instruments = []
    elif mutation == "zero_subscriptions":
        subscriptions = []
    elif mutation == "zero_required":
        subscriptions = [{**subscription, "optional": True}]
    elif mutation == "duplicate_instrument":
        instruments = [instrument, dict(instrument)]
    elif mutation == "duplicate_request":
        subscriptions = [
            subscription,
            {**subscription, "name": "required-trades", "feed_kind": "trades"},
        ]
    elif mutation == "duplicate_subscription":
        subscriptions = [
            subscription,
            {**subscription, "request_id": 4, "feed_kind": "trades"},
        ]
    elif mutation == "missing_instrument":
        subscriptions = [{**subscription, "instrument_id": "missing"}]
    elif mutation == "contradictory_feed":
        subscriptions = [subscription, {**subscription, "name": "duplicate", "request_id": 4}]
    elif mutation == "malformed_identity":
        subscriptions = [{**subscription, "request_id": -1}]
    elif mutation == "base_snapshot":
        subscriptions = [{**subscription, "snapshot": True}]
    inputs_path.write_text(
        json.dumps({"instruments": instruments, "subscriptions": subscriptions}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["validate-recorder", str(config_path), "--inputs", str(inputs_path)],
    )

    assert result.exit_code == 78
    payload = json.loads(result.stdout)
    assert str(inputs_path) in payload["message"]
    assert reason in payload["message"]


def test_empty_recorder_input_fails_before_adapter_connect(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    adapter = FakeMarketData()

    with pytest.raises(RecorderFatalError, match="zero instruments"):
        Recorder(_config(database), adapter).start(
            now_us=100,
            instruments=(),
            subscriptions=(),
        )

    assert adapter.connect_calls == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_production_preflight_rejects_input_above_configured_line_cap(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    inputs_path = tmp_path / "market-data.json"
    config_path.write_text(
        json.dumps(
            _config(tmp_path / "v2.sqlite3", market_data_line_limit=1).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    input_payload = _replay_fixture_without_callbacks()
    input_payload.pop("callbacks")
    inputs_path.write_text(json.dumps(input_payload), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["validate-recorder", str(config_path), "--inputs", str(inputs_path)],
    )

    assert result.exit_code == 78
    payload = json.loads(result.stdout)
    assert str(inputs_path) in payload["message"]
    assert "market_data_line_limit" in payload["message"]


def test_lease_projects_in_source_order_and_acknowledges_only_after_projection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = [
        inbox.admit(
            fence,
            MarketDataCallback(
                "quote",
                received_at_us=received,
                provider_at_us=received - 1,
                payload={"event_at_us": received - 1, "bid": float(received)},
            ),
        )
        for received in (10, 11)
    ]

    leased = inbox.lease_pending(
        "worker-1", now_us=20, lease_us=10, limit=10, authority=_authority()
    )
    first_event = inbox.project(leased[0], authority=_authority())
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence = ?",
                (admitted[0].source_sequence,),
            ).fetchone()[0]
            == "leased"
        )
    inbox.acknowledge(
        leased[0], first_event.event_id, acknowledged_at_us=21, authority=_authority()
    )
    second_event = inbox.project(leased[1], authority=_authority())
    inbox.acknowledge(
        leased[1], second_event.event_id, acknowledged_at_us=22, authority=_authority()
    )

    assert [item.source_sequence for item in leased] == [item.source_sequence for item in admitted]
    with connect_v2(database) as connection:
        rows = tuple(connection.execute("SELECT * FROM market_events ORDER BY source_sequence"))
        latest = connection.execute("SELECT * FROM market_latest").fetchone()
    assert len(rows) == 2
    assert latest["event_id"] == second_event.event_id


def test_crash_after_admission_and_after_projection_recovers_idempotently(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = inbox.admit(
        fence,
        MarketDataCallback("trade", 10, 9, {"event_at_us": 9, "last": 100.25, "size": 2}),
    )

    first_lease = inbox.lease_pending(
        "dead-worker", now_us=10, lease_us=5, limit=1, authority=_authority()
    )[0]
    projected = inbox.project(first_lease, authority=_authority())
    assert inbox.reclaim_expired_leases(now_us=14, authority=_authority()) == 0
    assert inbox.reclaim_expired_leases(now_us=15, authority=_authority()) == 1
    retry_lease = inbox.lease_pending(
        "new-worker", now_us=15, lease_us=5, limit=1, authority=_authority()
    )[0]
    retried = inbox.project(retry_lease, authority=_authority())
    inbox.acknowledge(retry_lease, retried.event_id, acknowledged_at_us=16, authority=_authority())

    assert retried.event_id == projected.event_id
    assert projected.inserted is True
    assert retried.inserted is False
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 1
        row = connection.execute(
            "SELECT lifecycle, attempts FROM callback_inbox WHERE source_sequence = ?",
            (admitted.source_sequence,),
        ).fetchone()
    assert tuple(row) == ("acknowledged", 2)


def test_acknowledgement_write_failure_preserves_leased_callback_and_projection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = inbox.admit(
        fence,
        MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 100.0}),
    )
    leased = inbox.lease_pending("worker", now_us=11, lease_us=10, limit=1, authority=_authority())[
        0
    ]
    projected = inbox.project(leased, authority=_authority())
    with connect_v2(database) as connection:
        connection.execute(
            "CREATE TRIGGER fail_acknowledgement BEFORE UPDATE OF lifecycle ON callback_inbox "
            "WHEN NEW.lifecycle='acknowledged' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        inbox.acknowledge(
            leased,
            projected.event_id,
            acknowledged_at_us=12,
            authority=_authority(),
        )

    with connect_v2(database, verify_schema=False) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json, normalized_event_id FROM callback_inbox "
            "WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_id=?", (projected.event_id,)
            ).fetchone()[0]
            == 1
        )
        connection.execute("DROP TRIGGER fail_acknowledgement")
    assert callback[0] == "leased"
    assert callback[1] is not None
    assert callback[2] is None

    inbox.acknowledge(
        leased,
        projected.event_id,
        acknowledged_at_us=13,
        authority=_authority(),
    )
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT lifecycle, normalized_event_id FROM callback_inbox WHERE source_sequence=?",
                (admitted.source_sequence,),
            ).fetchone()
        ) == ("acknowledged", projected.event_id)


def test_poison_callback_is_failed_without_blocking_the_next_callback(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    inbox.admit(fence, MarketDataCallback("quote", 10, None, {"bid": "bad"}))
    good = inbox.admit(
        fence, MarketDataCallback("quote", 11, None, {"event_at_us": 11, "bid": 1.0})
    )
    leased = inbox.lease_pending("worker", now_us=20, lease_us=10, limit=10, authority=_authority())

    with pytest.raises(NormalizationError):
        inbox.project(leased[0], authority=_authority())
    inbox.fail(leased[0], "MALFORMED_CALLBACK", failed_at_us=20, authority=_authority())
    event = inbox.project(leased[1], authority=_authority())
    inbox.acknowledge(leased[1], event.event_id, acknowledged_at_us=21, authority=_authority())

    with connect_v2(database) as connection:
        states = tuple(
            connection.execute(
                "SELECT lifecycle, failure_code FROM callback_inbox ORDER BY source_sequence"
            )
        )
    assert tuple(states[0]) == ("failed", "MALFORMED_CALLBACK")
    assert good.source_sequence == 2
    assert tuple(states[1]) == ("acknowledged", None)


def test_projection_batch_isolates_poison_and_acknowledges_later_callback(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    inbox.admit(fence, MarketDataCallback("quote", 10, None, {"bid": "bad"}))
    good = inbox.admit(
        fence, MarketDataCallback("quote", 11, None, {"event_at_us": 11, "bid": 1.0})
    )
    leased = inbox.lease_pending("worker", now_us=20, lease_us=10, limit=10, authority=_authority())

    result = inbox.project_batch(leased, now_us=20, authority=_authority())

    assert result.processed == 1
    assert result.causal_now_us == 20
    with connect_v2(database) as connection:
        states = tuple(
            connection.execute(
                "SELECT lifecycle, failure_code, normalized_event_id FROM callback_inbox "
                "ORDER BY source_sequence"
            )
        )
    assert tuple(states[0]) == ("failed", "MALFORMED_CALLBACK", None)
    assert tuple(states[1]) == ("acknowledged", None, good.event_uid)


def test_projection_batch_acknowledgement_failure_preserves_durable_projection_for_recovery(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = inbox.admit(
        fence,
        MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 100.0}),
    )
    leased = inbox.lease_pending("worker", now_us=11, lease_us=10, limit=1, authority=_authority())
    with connect_v2(database) as connection:
        connection.execute(
            "CREATE TRIGGER fail_batch_acknowledgement BEFORE UPDATE OF lifecycle "
            "ON callback_inbox WHEN NEW.lifecycle='acknowledged' "
            "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        inbox.project_batch(leased, now_us=12, authority=_authority())

    with connect_v2(database, verify_schema=False) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json, normalized_event_id FROM callback_inbox "
            "WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 1
        connection.execute("DROP TRIGGER fail_batch_acknowledgement")
    assert tuple(callback) == ("leased", callback[1], None)
    assert callback[1] is not None

    result = inbox.project_batch(leased, now_us=13, authority=_authority())
    assert result.processed == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT lifecycle, normalized_event_id FROM callback_inbox WHERE source_sequence=?",
                (admitted.source_sequence,),
            ).fetchone()
        ) == ("acknowledged", admitted.event_uid)


def test_waiting_callback_admission_precedes_the_next_projection_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    for received_at_us in range(10, 74):
        inbox.admit(
            fence,
            MarketDataCallback(
                "quote",
                received_at_us,
                None,
                {"event_at_us": received_at_us, "bid": 100.0},
            ),
        )
    leased = inbox.lease_pending("worker", now_us=80, lease_us=10, limit=64, authority=_authority())
    first_projection_entered = threading.Event()
    release_first_projection = threading.Event()
    original_project = inbox.project
    original_acquire_projection_writer = inbox._acquire_projection_writer
    projection_transactions = 0

    def slow_first_projection(*args: object, **kwargs: object) -> object:
        if not first_projection_entered.is_set():
            first_projection_entered.set()
            assert release_first_projection.wait(timeout=5)
        return original_project(*args, **kwargs)  # type: ignore[arg-type]

    def assert_admission_precedes_next_projection_transaction() -> None:
        nonlocal projection_transactions
        original_acquire_projection_writer()
        projection_transactions += 1
        if projection_transactions == 2:
            with connect_v2(database) as connection:
                assert (
                    connection.execute(
                        "SELECT count(*) FROM callback_inbox WHERE received_at_us=1000"
                    ).fetchone()[0]
                    == 1
                )
                assert (
                    connection.execute(
                        "SELECT count(*) FROM callback_inbox WHERE received_at_us=1001"
                    ).fetchone()[0]
                    == 0
                )

    monkeypatch.setattr(inbox, "project", slow_first_projection)
    monkeypatch.setattr(
        inbox,
        "_acquire_projection_writer",
        assert_admission_precedes_next_projection_transaction,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        drain = executor.submit(
            inbox.project_batch,
            leased,
            now_us=80,
            authority=_authority(),
        )
        assert first_projection_entered.wait(timeout=5)

        def admit_continuously() -> tuple[AdmissionResult, AdmissionResult]:
            first = inbox.admit(
                fence,
                MarketDataCallback("quote", 1000, None, {"event_at_us": 1000, "bid": 101.0}),
                authority=_authority(),
            )
            second = inbox.admit(
                fence,
                MarketDataCallback("quote", 1001, None, {"event_at_us": 1001, "bid": 102.0}),
                authority=_authority(),
            )
            return first, second

        admission = executor.submit(admit_continuously)
        with inbox._writer_condition:
            assert inbox._writer_condition.wait_for(
                lambda: inbox._next_writer_ticket - inbox._serving_writer_ticket >= 2,
                timeout=5,
            )
        release_first_projection.set()
        admitted, admitted_after_projection = admission.result(timeout=5)
        result = drain.result(timeout=5)

    assert result.processed == 64
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
                (admitted.source_sequence,),
            ).fetchone()[0]
            == "pending"
        )
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
                (admitted_after_projection.source_sequence,),
            ).fetchone()[0]
            == "pending"
        )


def test_stale_request_generation_is_terminal_evidence_not_active_state(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    _seed_subscription(database)
    inbox = CallbackInbox(database)

    stale = inbox.admit(
        CallbackFence("run-1", 1, 1, 3, "subscription-1"),
        MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 1.0}),
    )

    with connect_v2(database) as connection:
        row = connection.execute(
            "SELECT lifecycle, failure_code FROM callback_inbox WHERE source_sequence = ?",
            (stale.source_sequence,),
        ).fetchone()
        assert connection.execute("SELECT count(*) FROM market_latest").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gaps").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1
    assert tuple(row) == ("failed", "STALE_REQUEST_GENERATION")


def test_receipt_chain_matches_phase2_contract_and_authorizes_compaction(tmp_path: Path) -> None:
    from stocker_runtime.storage import RetentionManager, RetentionPolicy

    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    for received in (10, 11):
        inbox.admit(
            fence,
            MarketDataCallback("quote", received, None, {"event_at_us": received, "bid": 1.0}),
        )
    for leased in inbox.lease_pending(
        "worker", now_us=12, lease_us=10, limit=10, authority=_authority()
    ):
        event = inbox.project(leased, authority=_authority())
        inbox.acknowledge(leased, event.event_id, acknowledged_at_us=12, authority=_authority())

    receipt = inbox.create_receipt("run-1", created_at_us=13, limit=10, authority=_authority())
    compacted = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1, receipt_us=1_000, tombstone_us=1_000),
    ).run(now_us=14, measured_database_bytes=1, measured_wal_bytes=0)

    assert receipt is not None
    assert receipt.callback_count == 2
    assert compacted.payloads_compacted == 2


def test_receipt_write_failure_preserves_terminal_payload_and_unassigned_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = inbox.admit(
        fence,
        MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 100.0}),
    )
    leased = inbox.lease_pending("worker", now_us=11, lease_us=10, limit=1, authority=_authority())[
        0
    ]
    projected = inbox.project(leased, authority=_authority())
    inbox.acknowledge(
        leased,
        projected.event_id,
        acknowledged_at_us=12,
        authority=_authority(),
    )
    with connect_v2(database) as connection:
        connection.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON callback_receipts "
            "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="disk full"):
        inbox.create_receipt("run-1", created_at_us=13, authority=_authority())

    with connect_v2(database, verify_schema=False) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json, receipt_batch_id FROM callback_inbox "
            "WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 0
        connection.execute("DROP TRIGGER fail_receipt")
    assert tuple(callback) == ("acknowledged", '{"bid":100.0,"event_at_us":10}', None)

    receipt = inbox.create_receipt("run-1", created_at_us=14, authority=_authority())
    assert receipt is not None and receipt.callback_count == 1


def test_receipt_creation_rejects_batches_larger_than_the_recorder_drain(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)

    with pytest.raises(ValueError, match="between 1 and 256"):
        inbox.create_receipt(
            "run-1",
            created_at_us=13,
            limit=257,
            authority=_authority(),
        )


class FakeMarketData:
    capabilities = frozenset({"market_data"})

    def __init__(
        self, *, fail_connect: bool = False, fail_subscribe: set[int] | None = None
    ) -> None:
        self.callback = None
        self.disconnect_callback = None
        self.status_callback = None
        self.connected = False
        self.subscriptions: list[CallbackFence] = []
        self.subscribe_attempts: list[int] = []
        self.active_request_ids: set[int] = set()
        self.retry_calls: list[int] = []
        self.cancelled: list[int] = []
        self.fail_connect = fail_connect
        self.fail_subscribe = set() if fail_subscribe is None else fail_subscribe
        self.connect_calls = 0
        self.disconnect_calls = 0

    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def set_disconnect_callback(self, callback: object) -> None:
        self.disconnect_callback = callback

    def set_status_callback(self, callback: object) -> None:
        self.status_callback = callback

    def connect(self) -> None:
        self.connect_calls += 1
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False
        self.active_request_ids.clear()

    def subscribe(self, fence: CallbackFence) -> None:
        assert fence.request_id is not None
        self.subscribe_attempts.append(fence.request_id)
        if fence.request_id in self.fail_subscribe:
            raise RuntimeError("subscribe failed")
        if fence.request_id in self.active_request_ids:
            raise RuntimeError("duplicate active request")
        self.subscriptions.append(fence)
        self.active_request_ids.add(fence.request_id)

    def retry_subscription(self, fence: CallbackFence) -> None:
        assert fence.request_id is not None
        self.retry_calls.append(fence.request_id)
        self.active_request_ids.discard(fence.request_id)
        self.subscribe(fence)

    def cancel(self, request_id: int) -> None:
        self.cancelled.append(request_id)
        self.active_request_ids.discard(request_id)

    def emit(self, fence: CallbackFence, callback: MarketDataCallback) -> object:
        assert callable(self.callback)
        return self.callback(fence, callback)


def _specs() -> tuple[InstrumentSpec, tuple[SubscriptionSpec, ...]]:
    instrument = InstrumentSpec(
        instrument_id="instrument-1",
        ibkr_con_id=123,
        kind="stock",
        symbol="AAPL",
        exchange="SMART",
        currency="USD",
    )
    subscriptions = (
        SubscriptionSpec(
            name="required-quotes",
            instrument_id="instrument-1",
            feed_kind="quotes",
            request_id=3,
            continuity_required=True,
            optional=False,
            stale_after_us=10,
        ),
        SubscriptionSpec(
            name="optional-bars",
            instrument_id="instrument-1",
            feed_kind="bars",
            request_id=4,
            continuity_required=False,
            optional=True,
            stale_after_us=20,
        ),
    )
    return instrument, subscriptions


def _replay_fixture_without_callbacks() -> dict[str, object]:
    instrument, subscriptions = _specs()
    return {
        "instruments": [
            {key: value for key, value in instrument.__dict__.items() if value is not None}
        ],
        "subscriptions": [item.__dict__ for item in subscriptions],
        "callbacks": [],
    }


def _three_required_specs() -> tuple[InstrumentSpec, tuple[SubscriptionSpec, ...]]:
    instrument, _ = _specs()
    return instrument, tuple(
        SubscriptionSpec(
            name=f"required-{feed_kind}",
            instrument_id=instrument.instrument_id,
            feed_kind=feed_kind,
            request_id=request_id,
            continuity_required=True,
            optional=False,
            stale_after_us=10,
        )
        for request_id, feed_kind in ((3, "quotes"), (4, "trades"), (5, "bars"))
    )


def test_recorder_admits_exact_logical_alias_for_imported_ibkr_contract(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES "
            "('legacy-instrument-aal', ?, 123, 'stock', 'AAL', 'SMART', 'USD')",
            ("b" * 64,),
        )

    instrument = InstrumentSpec(
        instrument_id="AAL",
        ibkr_con_id=123,
        kind="stock",
        symbol="AAL",
        exchange="SMART",
        currency="USD",
    )
    subscription = SubscriptionSpec(
        name="active-aal-quotes",
        instrument_id="AAL",
        feed_kind="quotes",
        request_id=3,
        continuity_required=True,
        optional=False,
        stale_after_us=15_000_000,
    )
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)

    state = recorder.start(
        now_us=100,
        instruments=(instrument,),
        subscriptions=(subscription,),
    )
    adapter.emit(
        state.fences[0],
        MarketDataCallback(
            "quote",
            101,
            101,
            {"event_at_us": 101, "bid": 14.9, "ask": 15.1},
        ),
    )
    assert recorder.drain(now_us=102, run_downstream_when_idle=False) == 1
    recorder.stop(now_us=103)

    assert len(state.fences) == 1
    assert len(adapter.subscriptions) == 1
    with connect_v2(database) as connection:
        aliases = tuple(
            connection.execute(
                "SELECT instrument_id FROM instruments WHERE ibkr_con_id=123 ORDER BY instrument_id"
            )
        )
        subscription_instrument = connection.execute(
            "SELECT instrument_id FROM subscriptions WHERE run_id='run-1'"
        ).fetchone()[0]
        event_instrument = connection.execute(
            "SELECT instrument_id FROM market_events WHERE run_id='run-1'"
        ).fetchone()[0]
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
    assert [row["instrument_id"] for row in aliases] == ["AAL", "legacy-instrument-aal"]
    assert subscription_instrument == "AAL"
    assert event_instrument == "AAL"
    assert foreign_keys == ()


def test_logical_alias_is_idempotent_across_unclean_recorder_restart(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES "
            "('legacy-instrument-aal', ?, 123, 'stock', 'AAL', 'SMART', 'USD')",
            ("b" * 64,),
        )
    instrument = InstrumentSpec("AAL", 123, "stock", "AAL", "SMART", "USD")
    subscription = SubscriptionSpec(
        name="active-aal-quotes",
        instrument_id="AAL",
        feed_kind="quotes",
        request_id=3,
        continuity_required=True,
        optional=False,
        stale_after_us=15_000_000,
    )
    first = Recorder(_config(database), FakeMarketData())
    first_state = first.start(
        now_us=100,
        instruments=(instrument,),
        subscriptions=(subscription,),
    )
    first.abandon_unclean()

    second = Recorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=15_000_000),
        FakeMarketData(),
    )
    second_state = second.start(
        now_us=15_000_101,
        instruments=(instrument,),
        subscriptions=(subscription,),
    )
    second.stop(now_us=15_000_102)

    with connect_v2(database) as connection:
        aliases = connection.execute(
            "SELECT count(*) FROM instruments WHERE ibkr_con_id=123"
        ).fetchone()[0]
        subscription_instruments = {
            row["instrument_id"]
            for row in connection.execute(
                "SELECT instrument_id FROM subscriptions WHERE run_id='run-1'"
            )
        }
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
    assert first_state.recorder_generation == 1
    assert second_state.recorder_generation == 2
    assert aliases == 2
    assert subscription_instruments == {"AAL"}
    assert foreign_keys == ()


@pytest.mark.parametrize(
    "changes",
    (
        {
            "kind": "stock",
            "option_expiry": None,
            "option_strike": None,
            "option_right": None,
            "option_multiplier": None,
        },
        {"symbol": "AAOI"},
        {"exchange": "CBOE"},
        {"currency": "EUR"},
        {"option_expiry": "20260815"},
        {"option_strike": "16"},
        {"option_right": "put"},
        {"option_multiplier": "10"},
    ),
    ids=("kind", "symbol", "exchange", "currency", "expiry", "strike", "right", "multiplier"),
)
def test_recorder_rejects_conflicting_physical_identity_for_logical_alias(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, "
            "option_multiplier) VALUES "
            "('legacy-option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
            "'20260814', '15.0', 'call', '100')",
            ("b" * 64,),
        )

    values: dict[str, Any] = {
        "instrument_id": "ibkr-option-321",
        "ibkr_con_id": 321,
        "kind": "option",
        "symbol": "AAL",
        "exchange": "SMART",
        "currency": "USD",
        "option_expiry": "20260814",
        "option_strike": "15",
        "option_right": "call",
        "option_multiplier": "100",
    }
    values.update(changes)
    instrument = InstrumentSpec(**values)
    subscription = SubscriptionSpec(
        name="active-aal-option",
        instrument_id=instrument.instrument_id,
        feed_kind="quotes",
        request_id=3,
        continuity_required=True,
        optional=False,
        stale_after_us=15_000_000,
    )

    with pytest.raises(
        RecorderFatalError,
        match="IBKR contract identity conflicts with an existing logical instrument",
    ):
        Recorder(_config(database), FakeMarketData()).start(
            now_us=100,
            instruments=(instrument,),
            subscriptions=(subscription,),
        )

    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM recorder_generations").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM subscriptions").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM instruments").fetchone()[0] == 1


def test_recorder_rejects_concurrent_writer_and_recovers_immediately_after_crash(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    started = first.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with pytest.raises(DuplicateWriterError):
        Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
            now_us=101, instruments=(instrument,), subscriptions=specs
        )

    first.abandon_unclean()
    restarted = Recorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=15_000_000),
        FakeMarketData(),
    ).start(now_us=102, instruments=(instrument,), subscriptions=specs)

    assert started.recorder_generation == 1
    assert restarted.recorder_generation == 2
    assert restarted.connection_generation == 2
    with connect_v2(database) as connection:
        old = connection.execute(
            "SELECT ended_at_us, clean_stop FROM recorder_generations "
            "WHERE run_id = 'run-1' AND generation = 1"
        ).fetchone()
        assert (
            connection.execute(
                "SELECT count(*) FROM gaps WHERE reason = 'UNCLEAN_RECORDER_RESTART'"
            ).fetchone()[0]
            == 2
        )
    assert tuple(old) == (102, 0)


def test_clean_stop_restarts_same_run_with_generation_and_callback_continuity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first_adapter = FakeMarketData()
    first = Recorder(_config(database), first_adapter)
    first_state = first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    first_result = first_adapter.emit(
        first_state.fences[0],
        MarketDataCallback("quote", 101, 101, {"event_at_us": 101, "bid": 100.0}),
    )
    first.drain(now_us=102)
    first.stop(now_us=103)

    second_adapter = FakeMarketData()
    second = Recorder(_config(database, owner_id="owner-2"), second_adapter)
    second_state = second.start(now_us=104, instruments=(instrument,), subscriptions=specs)
    second_result = second_adapter.emit(
        second_state.fences[0],
        MarketDataCallback("quote", 105, 105, {"event_at_us": 105, "bid": 101.0}),
    )
    second.drain(now_us=106)

    assert first_state.recorder_generation == 1
    assert second_state.recorder_generation == 2
    assert second_state.connection_generation == 2
    assert first_result.source_sequence < second_result.source_sequence
    with connect_v2(database) as connection:
        generations = tuple(
            connection.execute(
                "SELECT generation, clean_stop, termination_code, ownership_protocol, git_commit "
                "FROM recorder_generations WHERE run_id='run-1' ORDER BY generation"
            )
        )
        provenance = tuple(
            connection.execute(
                "SELECT source_sequence, recorder_generation, connection_generation "
                "FROM callback_inbox ORDER BY source_sequence"
            )
        )
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-1'"
        ).fetchone()
    assert [tuple(row) for row in generations] == [
        (1, 1, "CLEAN_STOP", "local_flock_v1", "deadbee"),
        (2, 0, None, "local_flock_v1", "deadbee"),
    ]
    assert [tuple(row) for row in provenance] == [
        (first_result.source_sequence, 1, 1),
        (second_result.source_sequence, 2, 2),
    ]
    assert tuple(run) == ("running", None)


def test_clean_restart_rejects_incompatible_frozen_configuration(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    first.stop(now_us=101)

    with pytest.raises(RecorderFatalError, match="frozen configuration changed"):
        Recorder(_config(database, config_hash="b" * 64), FakeMarketData()).start(
            now_us=102,
            instruments=(instrument,),
            subscriptions=specs,
        )

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM recorder_generations WHERE run_id='run-1'"
            ).fetchone()[0]
            == 1
        )


def test_clean_restart_rejects_changed_market_data_input(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    first.stop(now_us=101)

    changed = (replace(specs[0], stale_after_us=11), specs[1])
    with pytest.raises(RecorderFatalError, match="frozen market-data input changed"):
        Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
            now_us=102,
            instruments=(instrument,),
            subscriptions=changed,
        )

    with connect_v2(database) as connection:
        generation = connection.execute(
            "SELECT input_hash FROM recorder_generations WHERE run_id='run-1'"
        ).fetchone()
    assert generation["input_hash"] == market_data_input_hash((instrument,), specs)


def test_schema_15_clean_stop_can_bind_inputs_and_restart_same_run(tmp_path: Path) -> None:
    database = tmp_path / "schema-15.sqlite3"
    migration_root = tmp_path / "schema-15-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:15]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, ended_at_us, config_hash, "
            "git_commit, data_class, status) VALUES "
            "('run-1', 'prospective_record', 'ibkr', 1, 2, ?, 'old-code', "
            "'prospective_protected', 'stopped')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us, "
            "ended_at_us, clean_stop, termination_code) VALUES "
            "('run-1', 1, 'old-owner', 1, 2, 1, 'CLEAN_STOP')"
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, reason, "
            "process_heartbeat_at_us, connection_state, connection_generation) VALUES "
            "('run-1', 1, 'stopped', NULL, 2, 'disconnected', 1)"
        )
    assert migrate_database(database, applied_at_us=3).applied_versions == (16, 17, 18, 19)
    instrument, specs = _specs()

    restarted = Recorder(_config(database, owner_id="new-owner"), FakeMarketData()).start(
        now_us=4,
        instruments=(instrument,),
        subscriptions=specs,
    )

    assert restarted.recorder_generation == 2
    with connect_v2(database) as connection:
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-1'"
        ).fetchone()
        generations = tuple(
            connection.execute(
                "SELECT generation, input_hash FROM recorder_generations "
                "WHERE run_id='run-1' ORDER BY generation"
            )
        )
    assert tuple(run) == ("running", None)
    assert [tuple(row) for row in generations] == [
        (1, None),
        (2, market_data_input_hash((instrument,), specs)),
    ]


def test_database_symlink_alias_cannot_bypass_local_writer_lock(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    alias = tmp_path / "database-alias.sqlite3"
    initialize_database(database)
    alias.symlink_to(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with pytest.raises(DuplicateWriterError, match="writer lock is held"):
        Recorder(_config(alias, owner_id="owner-2"), FakeMarketData()).start(
            now_us=101,
            instruments=(instrument,),
            subscriptions=specs,
        )


def test_subprocess_writer_is_rejected_and_kill_releases_lock_immediately(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    child_script = """
import sys
from pathlib import Path
from stocker_runtime.ingestion import InstrumentSpec, Recorder, RecorderConfig, SubscriptionSpec

class Adapter:
    capabilities = frozenset({"market_data"})
    connected = False
    def set_callback(self, callback): self.callback = callback
    def set_disconnect_callback(self, callback): self.disconnect_callback = callback
    def set_status_callback(self, callback): self.status_callback = callback
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def subscribe(self, fence): pass
    def retry_subscription(self, fence): pass
    def cancel(self, request_id): pass

database = Path(sys.argv[1])
config = RecorderConfig(
    database=database, run_id="run-1", owner_id="child-owner",
    mode="prospective_record", host="127.0.0.1", port=4001, client_id=71,
    read_only=True, external_read_only_verified=True, config_hash="a" * 64,
    git_commit="deadbee",
)
instrument = InstrumentSpec("instrument-1", 123, "stock", "AAPL", "SMART", "USD")
subscription = SubscriptionSpec("required-quotes", "instrument-1", "quotes", 3, True, False, 10)
Recorder(config, Adapter()).start(
    now_us=100, instruments=(instrument,), subscriptions=(subscription,)
)
print("READY", flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_script, str(database)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        instrument = InstrumentSpec("instrument-1", 123, "stock", "AAPL", "SMART", "USD")
        subscription = SubscriptionSpec(
            "required-quotes", "instrument-1", "quotes", 3, True, False, 10
        )
        with pytest.raises(DuplicateWriterError, match="writer lock is held"):
            Recorder(_config(database, owner_id="parent-owner"), FakeMarketData()).start(
                now_us=101,
                instruments=(instrument,),
                subscriptions=(subscription,),
            )

        process.kill()
        process.wait(timeout=5)
        replacement = Recorder(_config(database, owner_id="parent-owner"), FakeMarketData()).start(
            now_us=102,
            instruments=(instrument,),
            subscriptions=(subscription,),
        )
        assert replacement.recorder_generation == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_legacy_database_lease_requires_expiry_without_local_lock_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    instrument, specs = _specs()

    with pytest.raises(DuplicateWriterError, match="legacy authoritative writer lease"):
        Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
            now_us=2,
            instruments=(instrument,),
            subscriptions=specs,
        )

    restarted = Recorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=15_000_000),
        FakeMarketData(),
    ).start(
        now_us=15_000_002,
        instruments=(instrument,),
        subscriptions=specs,
    )
    assert restarted.recorder_generation == 2


def test_recorder_disconnect_reconnect_and_staleness_are_scoped(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    required, optional = state.fences

    recorder.mark_stale(now_us=111)
    recorder.disconnected(now_us=112)
    reconnected = recorder.reconnect(now_us=113)
    adapter.emit(
        reconnected.fences[0],
        MarketDataCallback("quote", 114, 114, {"event_at_us": 114, "bid": 1.0}),
    )
    assert recorder.drain(now_us=115) == 1

    with connect_v2(database) as connection:
        gaps = tuple(
            connection.execute(
                "SELECT gap.subscription_id, gap.reason, gap.continuity_required, "
                "gap.resolved_at_us, subscription.feed_kind FROM gaps gap "
                "JOIN subscriptions subscription USING(subscription_id) "
                "ORDER BY gap.started_at_us, gap.gap_id"
            )
        )
    assert required.connection_generation == optional.connection_generation == 1
    assert reconnected.connection_generation == 2
    assert any(row["reason"] == "STREAM_STALE" and row["continuity_required"] == 1 for row in gaps)
    assert any(row["reason"] == "IBKR_DISCONNECT" for row in gaps)
    assert any(row["resolved_at_us"] == 115 for row in gaps)
    quote_gaps = tuple(row for row in gaps if row["feed_kind"] == "quotes")
    assert {row["reason"] for row in quote_gaps} == {
        "RECONNECT_UNCERTAINTY",
        "STREAM_STALE",
    }
    assert all(row["resolved_at_us"] is not None for row in quote_gaps)
    assert (
        next(
            row["resolved_at_us"] for row in quote_gaps if row["reason"] == "RECONNECT_UNCERTAINTY"
        )
        == 115
    )


def test_recorder_recovery_is_automatic_backed_off_and_bounded(tmp_path: Path) -> None:
    database = tmp_path / "automatic-recovery.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_connect=True)
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert adapter.connect_calls == 1

    assert recorder.recover_connection(now_us=101) is False
    assert adapter.connect_calls == 2
    assert recorder.recover_connection(now_us=1_000_100) is False
    assert adapter.connect_calls == 2

    adapter.fail_connect = False
    assert recorder.recover_connection(now_us=1_000_101) is True
    assert adapter.connect_calls == 3
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT connection_generation, connection_state, lifecycle FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == (3, "connected", "running")
    recorder.stop(now_us=1_000_102)


def test_recorder_recovery_backoff_caps_at_sixty_seconds(tmp_path: Path) -> None:
    database = tmp_path / "bounded-recovery.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_connect=True)
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    attempt_at_us = 101

    for delay_seconds in (1, 2, 4, 8, 16, 32, 60, 60):
        assert recorder.recover_connection(now_us=attempt_at_us) is False
        calls_after_attempt = adapter.connect_calls
        with connect_v2(database) as connection:
            unresolved_during_outage = connection.execute(
                "SELECT count(*) FROM gaps WHERE resolved_at_us IS NULL"
            ).fetchone()[0]
            unresolved_incidents_during_outage = connection.execute(
                "SELECT count(*) FROM incidents WHERE code='IBKR_CONNECT_FAILED' "
                "AND resolved_at_us IS NULL"
            ).fetchone()[0]
        assert unresolved_during_outage <= 2 * len(specs)
        assert unresolved_incidents_during_outage == 1
        attempt_at_us += delay_seconds * 1_000_000
        assert recorder.recover_connection(now_us=attempt_at_us - 1) is False
        assert adapter.connect_calls == calls_after_attempt

    assert adapter.connect_calls == 9
    adapter.fail_connect = False
    assert recorder.recover_connection(now_us=attempt_at_us) is True
    current = recorder.state
    assert current is not None
    adapter.emit(
        current.fences[0],
        MarketDataCallback(
            "quote",
            attempt_at_us + 1,
            attempt_at_us + 1,
            {"event_at_us": attempt_at_us + 1, "bid": 1.0},
        ),
    )
    adapter.emit(
        current.fences[1],
        MarketDataCallback(
            "bar",
            attempt_at_us + 1,
            attempt_at_us + 1,
            {
                "event_at_us": attempt_at_us + 1,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            },
        ),
    )
    assert recorder.drain(now_us=attempt_at_us + 2) == 2
    with connect_v2(database) as connection:
        unresolved = connection.execute(
            "SELECT count(*) FROM gaps WHERE resolved_at_us IS NULL AND reason IN "
            "('IBKR_CONNECT_FAILED','IBKR_DISCONNECT','IBKR_SUBSCRIBE_FAILED',"
            "'RECONNECT_UNCERTAINTY','STREAM_STALE')"
        ).fetchone()[0]
        unresolved_incidents = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_CONNECT_FAILED' "
            "AND resolved_at_us IS NULL"
        ).fetchone()[0]
        connect_incidents = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_CONNECT_FAILED'"
        ).fetchone()[0]
    assert unresolved == 0
    assert unresolved_incidents == 0
    assert connect_incidents == 1
    recorder.stop(now_us=attempt_at_us + 3)


def test_required_subscribe_retries_keep_bounded_incidents_and_resolve_on_success(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bounded-subscribe-incidents.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_subscribe={3})
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert recorder.recover_connection(now_us=101) is False
    with connect_v2(database) as connection:
        during_outage = tuple(
            connection.execute(
                "SELECT subscription_id, resolved_at_us FROM incidents "
                "WHERE code='IBKR_SUBSCRIBE_FAILED' ORDER BY subscription_id"
            )
        )
    assert len(during_outage) == 1
    assert all(row["resolved_at_us"] is None for row in during_outage)

    adapter.fail_subscribe.clear()
    assert recorder.recover_subscriptions(now_us=1_000_101) == 1
    with connect_v2(database) as connection:
        after_recovery = tuple(
            connection.execute(
                "SELECT resolved_at_us FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED'"
            )
        )
    assert len(after_recovery) == 1
    assert after_recovery[0]["resolved_at_us"] is None
    state = recorder.state
    assert state is not None
    adapter.emit(
        state.fences[0],
        MarketDataCallback("quote", 1_000_102, None, {"event_at_us": 1_000_102, "bid": 1.0}),
    )
    assert recorder.drain(now_us=1_000_103) == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT resolved_at_us FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED'"
            ).fetchone()[0]
            == 1_000_103
        )
    recorder.stop(now_us=1_000_104)


def test_reconnect_does_not_resolve_future_transport_gap(tmp_path: Path) -> None:
    database = tmp_path / "future-transport-gap.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.disconnected(now_us=300)

    reconnected = recorder.reconnect(now_us=200)

    with connect_v2(database) as connection:
        future_gaps = tuple(
            connection.execute(
                "SELECT gap.resolved_at_us FROM gaps gap JOIN subscriptions subscription "
                "USING(subscription_id) WHERE gap.reason='IBKR_DISCONNECT' "
                "AND subscription.connection_generation=1"
            )
        )
        fresh_uncertainty = connection.execute(
            "SELECT started_at_us, resolved_at_us FROM gaps gap "
            "JOIN subscriptions subscription USING(subscription_id) "
            "WHERE gap.reason='RECONNECT_UNCERTAINTY' "
            "AND subscription.connection_generation=? ORDER BY gap.started_at_us LIMIT 1",
            (reconnected.connection_generation,),
        ).fetchone()
    assert future_gaps
    assert all(row["resolved_at_us"] is None for row in future_gaps)
    assert tuple(fresh_uncertainty) == (200, None)
    recorder.stop(now_us=301)


def test_callback_after_clock_catchup_resolves_future_transport_incident(
    tmp_path: Path,
) -> None:
    database = tmp_path / "future-transport-incident.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_connect=True)
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=300, instruments=(instrument,), subscriptions=specs)
    adapter.fail_connect = False
    reconnected = recorder.reconnect(now_us=200)

    with connect_v2(database) as connection:
        before = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents WHERE code='IBKR_CONNECT_FAILED'"
        ).fetchone()
    assert tuple(before) == (300, None)

    recorder.receive(
        reconnected.fences[0],
        MarketDataCallback("quote", 301, 301, {"event_at_us": 301, "bid": 1.0}),
    )
    assert recorder.drain(now_us=302) == 1
    with connect_v2(database) as connection:
        after = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents WHERE code='IBKR_CONNECT_FAILED'"
        ).fetchone()
    assert tuple(after) == (300, 302)
    recorder.stop(now_us=303)


def test_high_rate_callback_path_uses_single_authoritative_admission_and_projects_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "callback-load.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    from stocker_runtime.ingestion import recorder as recorder_module

    real_connect = recorder_module.connect_v2
    redundant_owner_connections: list[bool] = []
    real_admission_connect = recorder.inbox._connect
    admission_connections = 0

    def connect_after_start(path: str | Path, *, verify_schema: bool = True) -> sqlite3.Connection:
        redundant_owner_connections.append(verify_schema)
        return real_connect(path, verify_schema=False)

    def connect_for_admission() -> sqlite3.Connection:
        nonlocal admission_connections
        admission_connections += 1
        return real_admission_connect()

    quote_fence = next(fence for fence in state.fences if fence.request_id == 3)
    bar_fence = next(fence for fence in state.fences if fence.request_id == 4)
    callbacks = [
        (
            quote_fence,
            MarketDataCallback(
                "quote",
                101 + offset,
                None,
                {"event_at_us": 101 + offset, "bid": 100.0 + offset},
            ),
        )
        for offset in range(255)
    ]
    callbacks.append(
        (
            bar_fence,
            MarketDataCallback(
                "bar",
                357,
                None,
                {
                    "event_at_us": 357,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                },
            ),
        )
    )
    with monkeypatch.context() as callback_context:
        callback_context.setattr(recorder_module, "connect_v2", connect_after_start)
        callback_context.setattr(recorder.inbox, "_connect", connect_for_admission)
        for fence, callback in callbacks:
            recorder.receive(fence, callback)
    assert admission_connections == 1

    from stocker_runtime.ingestion import inbox as inbox_module

    real_inbox_connect = inbox_module.connect_v2
    drain_connections = 0
    drain_transactions: list[str] = []
    drain_nonterminal_counts = 0

    def trace_transactions(connection: sqlite3.Connection) -> sqlite3.Connection:
        def trace(statement: str) -> None:
            nonlocal drain_nonterminal_counts
            normalized = statement.strip().upper()
            if normalized == "BEGIN IMMEDIATE" or normalized == "COMMIT":
                drain_transactions.append(normalized)
            if normalized.startswith("SELECT COUNT(*) FROM CALLBACK_INBOX") and (
                "LIFECYCLE IN ('PENDING', 'LEASED')" in normalized
            ):
                drain_nonterminal_counts += 1

        connection.set_trace_callback(trace)
        return connection

    def connect_during_drain(path: str | Path, *, verify_schema: bool = True) -> sqlite3.Connection:
        nonlocal drain_connections
        drain_connections += 1
        return trace_transactions(real_inbox_connect(path, verify_schema=verify_schema))

    def connect_recorder_during_drain(
        path: str | Path, *, verify_schema: bool = True
    ) -> sqlite3.Connection:
        return trace_transactions(real_connect(path, verify_schema=verify_schema))

    from stocker_runtime.ingestion import snapshot_projection

    downstream_calls: list[str] = []
    assert recorder._idea_runner is not None
    with monkeypatch.context() as drain_context:
        drain_context.setattr(inbox_module, "connect_v2", connect_during_drain)
        drain_context.setattr(recorder_module, "connect_v2", connect_recorder_during_drain)
        drain_context.setattr(
            snapshot_projection,
            "project_option_snapshot_captures",
            lambda *_args, **_kwargs: downstream_calls.append("snapshot"),
        )
        drain_context.setattr(
            recorder,
            "_fulfill_snapshot_interests_from_streams",
            lambda *, now_us: downstream_calls.append("interests"),
        )
        drain_context.setattr(
            recorder._idea_runner,
            "run_once",
            lambda *, now_us: downstream_calls.append("ideas"),
        )
        drain_context.setattr(
            recorder,
            "_reconcile_dynamic_market_data",
            lambda *, now_us: downstream_calls.append("dynamic"),
        )
        drain_context.setattr(
            recorder,
            "_shadow_engine",
            types.SimpleNamespace(run_once=lambda *, now_us: downstream_calls.append("shadow")),
        )
        assert recorder.drain(now_us=356) == 256
        assert downstream_calls == []
        connections_after_full_batch = drain_connections
        transactions_after_full_batch = len(drain_transactions)
        assert recorder.drain(now_us=358, run_downstream_when_idle=False) == 0
        assert downstream_calls == []
        assert drain_connections == connections_after_full_batch + 1
        assert len(drain_transactions) <= transactions_after_full_batch + 2
        assert recorder.drain(now_us=359, run_downstream_when_idle=True) == 0
    assert downstream_calls == ["snapshot", "interests", "ideas", "dynamic", "shadow"]
    assert redundant_owner_connections == []
    assert drain_connections <= 8
    assert drain_transactions.count("BEGIN IMMEDIATE") <= 24
    assert drain_transactions.count("COMMIT") <= 24
    assert drain_nonterminal_counts <= 8
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE lifecycle='acknowledged'"
            ).fetchone()[0]
            == 256
        )
        assert tuple(
            connection.execute(
                "SELECT count(*), count(DISTINCT receipt_batch_id) FROM callback_inbox "
                "WHERE run_id='run-1' AND receipt_batch_id IS NOT NULL"
            ).fetchone()
        ) == (256, 1)
        assert (
            connection.execute(
                "SELECT callback_count FROM callback_receipts WHERE run_id='run-1'"
            ).fetchone()[0]
            == 256
        )
        assert tuple(
            connection.execute(
                "SELECT count(*), sum(callback_count) FROM callback_receipts"
            ).fetchone()
        ) == (1, 256)
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_kind='bar'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT acknowledged_at_us FROM callback_inbox WHERE callback_kind='bar'"
            ).fetchone()[0]
            == 357
        )
        assert (
            connection.execute(
                "SELECT process_heartbeat_at_us FROM runtime_state WHERE run_id='run-1'"
            ).fetchone()[0]
            == 359
        )


def test_durable_callback_admission_wakes_the_idle_drain_without_polling(tmp_path: Path) -> None:
    database = tmp_path / "callback-wakeup.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    fence = next(item for item in state.fences if item.request_id == 3)

    assert recorder.wait_for_pending_callbacks(timeout=0) is False
    recorder.receive(
        fence,
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )
    assert recorder.wait_for_pending_callbacks(timeout=0) is True

    recorder.prepare_pending_callback_drain()
    assert recorder.wait_for_pending_callbacks(timeout=0) is False
    assert recorder.drain(now_us=102, run_downstream_when_idle=False) == 1
    recorder.stop(now_us=103)


def test_full_drain_batch_rearms_wakeup_for_preexisting_backlog(tmp_path: Path) -> None:
    database = tmp_path / "callback-backlog-wakeup.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    fence = next(item for item in state.fences if item.request_id == 3)
    for offset in range(257):
        recorder.receive(
            fence,
            MarketDataCallback(
                "quote",
                101 + offset,
                None,
                {"event_at_us": 101 + offset, "bid": 100.0 + offset},
            ),
        )

    recorder.prepare_pending_callback_drain()
    assert recorder.drain(now_us=358, run_downstream_when_idle=False) == 256
    assert recorder.wait_for_pending_callbacks(timeout=0) is True

    recorder.prepare_pending_callback_drain()
    assert recorder.drain(now_us=359, run_downstream_when_idle=False) == 1
    recorder.stop(now_us=360)


def test_full_batch_defers_only_when_each_leased_callback_is_receipted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "mixed-run-receipts.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    leased = tuple(
        LeasedCallback(
            source_sequence=sequence,
            event_uid=f"event-{sequence}",
            run_id=run_id,
            recorder_generation=1,
            connection_generation=1,
            request_id=None,
            callback_kind="quote",
            received_at_us=100 + sequence,
            provider_at_us=None,
            payload={},
            payload_sha256=f"{sequence:064x}",
            lease_owner="owner-1",
        )
        for sequence, run_id in ((10, "prior-run"), (11, "run-1"), (12, "prior-run"))
    )
    current_receipt = types.SimpleNamespace(
        run_id="run-1",
        first_source_sequence=11,
        last_source_sequence=11,
        created_at_us=111,
    )
    prior_receipts = (
        types.SimpleNamespace(
            run_id="prior-run",
            first_source_sequence=1,
            last_source_sequence=10,
            created_at_us=111,
        ),
        types.SimpleNamespace(
            run_id="prior-run",
            first_source_sequence=12,
            last_source_sequence=12,
            created_at_us=111,
        ),
    )
    receipt_sets = [(current_receipt, *prior_receipts), (current_receipt,)]
    downstream_calls: list[str] = []

    monkeypatch.setattr(recorder.inbox, "lease_pending", lambda *_args, **_kwargs: leased)
    monkeypatch.setattr(
        recorder.inbox,
        "project_batch",
        lambda *_args, **_kwargs: ProjectionBatchResult(processed=3, causal_now_us=111),
    )
    monkeypatch.setattr(
        recorder.inbox,
        "create_pending_receipts",
        lambda **_kwargs: receipt_sets.pop(0),
    )
    monkeypatch.setattr(recorder, "_heartbeat", lambda _now_us: None)
    from stocker_runtime.ingestion import snapshot_projection

    monkeypatch.setattr(
        snapshot_projection,
        "project_option_snapshot_captures",
        lambda *_args, **_kwargs: downstream_calls.append("snapshot"),
    )
    monkeypatch.setattr(
        recorder,
        "_fulfill_snapshot_interests_from_streams",
        lambda *, now_us: downstream_calls.append("interests"),
    )
    assert recorder._idea_runner is not None
    monkeypatch.setattr(
        recorder._idea_runner,
        "run_once",
        lambda *, now_us: downstream_calls.append("ideas"),
    )
    monkeypatch.setattr(
        recorder,
        "_reconcile_dynamic_market_data",
        lambda *, now_us: downstream_calls.append("dynamic"),
    )
    monkeypatch.setattr(
        recorder,
        "_shadow_engine",
        types.SimpleNamespace(run_once=lambda *, now_us: downstream_calls.append("shadow")),
    )

    assert recorder.drain(now_us=110, limit=3) == 3
    assert downstream_calls == []
    assert recorder.drain(now_us=111, limit=3) == 3
    assert downstream_calls == ["snapshot", "interests", "ideas", "dynamic", "shadow"]
    recorder.stop(now_us=112)


def test_admission_connection_closes_on_callback_thread_exit_and_abnormal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "callback-thread-exit.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    real_connect = inbox._connect
    closed = threading.Event()

    class TrackingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

        def close(self) -> None:
            self.connection.close()
            closed.set()

    with monkeypatch.context() as successful:
        successful.setattr(inbox, "_connect", lambda: TrackingConnection(real_connect()))
        thread = threading.Thread(
            target=lambda: inbox.admit(
                fence,
                MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 1.0}),
                authority=_authority(),
            )
        )
        thread.start()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert closed.wait(timeout=1)

    abnormal = CallbackInbox(database)
    exited = threading.Event()

    def abort_after_begin(connection: sqlite3.Connection) -> sqlite3.Row:
        assert connection.in_transaction
        raise SystemExit

    def admit_then_exit() -> None:
        try:
            abnormal.admit(
                fence,
                MarketDataCallback("quote", 11, None, {"event_at_us": 11, "bid": 2.0}),
                authority=_authority(),
            )
        except SystemExit:
            exited.set()

    monkeypatch.setattr(abnormal, "_authoritative_admission", abort_after_begin)
    thread = threading.Thread(target=admit_then_exit)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert exited.is_set()
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()


def test_staleness_reference_is_clamped_to_current_expected_session(tmp_path: Path) -> None:
    database = tmp_path / "session-clamped-staleness.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert recorder.mark_stale(now_us=1_000, expected_since_us=995) == 0
    assert recorder.mark_stale(now_us=1_006, expected_since_us=995) == 1
    with connect_v2(database) as connection:
        gap = connection.execute(
            "SELECT started_at_us, reason FROM gaps WHERE reason='STREAM_STALE'"
        ).fetchone()
    assert tuple(gap) == (1_005, "STREAM_STALE")
    recorder.stop(now_us=1_007)


def test_stale_feed_recovers_independently_only_when_regular_session_expects_data(
    tmp_path: Path,
) -> None:
    database = tmp_path / "session-stale-recovery.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert recorder.mark_stale(now_us=10_000, market_data_expected=False) == 0
    assert recorder.recover_subscriptions(now_us=10_000) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM gaps").fetchone()[0] == 0
        assert dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions")) == {
            3: "active",
            4: "active",
        }

    assert recorder.mark_stale(now_us=10_001, expected_since_us=100) == 2
    with connect_v2(database) as connection:
        assert dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions")) == {
            3: "disconnected",
            4: "disconnected",
        }
    assert recorder.recover_subscriptions(now_us=1_010_001) == 2
    assert set(adapter.retry_calls) == {3, 4}
    quote_fence = next(item for item in state.fences if item.request_id == 3)
    adapter.emit(
        quote_fence,
        MarketDataCallback("quote", 1_010_002, None, {"event_at_us": 1_010_002, "bid": 1.0}),
    )
    assert recorder.drain(now_us=1_010_003) == 1
    with connect_v2(database) as connection:
        assert dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions")) == {
            3: "active",
            4: "connecting",
        }


def test_admission_database_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)

    def locked(_path: Path) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("stocker_runtime.ingestion.inbox.connect_v2", locked)
    with pytest.raises(Exception, match="locked"):
        inbox.admit(_fence(), MarketDataCallback("quote", 10, None, {"event_at_us": 10}))


def test_unsafe_adapter_capability_is_rejected_before_connection(tmp_path: Path) -> None:
    class UnsafeMarketData(FakeMarketData):
        capabilities = frozenset({"market_data", "orders"})

        def place_order(self) -> None:
            raise AssertionError("must never be called")

    with pytest.raises(Exception, match="exactly market-data-only capability"):
        Recorder(_config(tmp_path / "v2.sqlite3"), UnsafeMarketData())


def test_replay_cli_runs_one_offline_recorder_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    instrument, subscriptions = _specs()
    fixture_path.write_text(
        json.dumps(
            {
                "instruments": [instrument.__dict__],
                "subscriptions": [item.__dict__ for item in subscriptions],
                "callbacks": [
                    {
                        "request_id": 3,
                        "callback_kind": "quote",
                        "received_at_us": 101,
                        "provider_at_us": 100,
                        "payload": {"event_at_us": 100, "bid": 10.0, "ask": 10.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    real_drain = Recorder.drain
    drain_results: list[tuple[int, bool]] = []

    def observe_drain(
        self: Recorder,
        *,
        now_us: int,
        limit: int = 256,
        defer_downstream_when_full: bool = True,
    ) -> int:
        processed = real_drain(
            self,
            now_us=now_us,
            limit=limit,
            defer_downstream_when_full=defer_downstream_when_full,
        )
        drain_results.append((processed, defer_downstream_when_full))
        return processed

    monkeypatch.setattr(Recorder, "drain", observe_drain)
    result = CliRunner().invoke(
        app,
        ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "admitted": 1,
        "mode": "prospective_record",
        "projected": 1,
        "status": "ok",
    }
    assert drain_results[-1] == (1, False)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 1


def test_official_bridge_import_is_lazy_and_safety_flags_fail_before_dependency_load() -> None:
    from stocker_runtime.ingestion import IBKRMarketData

    with pytest.raises(Exception, match="Read-Only"):
        IBKRMarketData.official(
            host="127.0.0.1",
            port=4001,
            client_id=71,
            read_only=False,
            external_read_only_verified=True,
            subscriptions=(),
        )


def test_recorder_consumes_optional_and_fatal_storage_cap_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    degraded = RetentionResult(
        cap_state=StorageCapState.DEGRADED,
        database_bytes=95,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=False,
        required_action="PAUSE_OPTIONAL_FEEDS",
        checkpoint_attempted=False,
        incremental_vacuum_attempted=False,
    )
    fatal = RetentionResult(
        cap_state=StorageCapState.FATAL,
        database_bytes=100,
        wal_bytes=64,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=False,
        optional_feeds_allowed=False,
        required_action="WAL_CAP_FATAL",
        checkpoint_attempted=True,
        incremental_vacuum_attempted=False,
    )

    class FakeRetention:
        result = degraded

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert now_us >= 101
            assert callable(precondition)
            return self.result

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", FakeRetention)
    assert recorder.maintain(now_us=101) is StorageCapState.DEGRADED
    assert adapter.cancelled == [4]
    FakeRetention.result = fatal

    with pytest.raises(Exception, match="WAL_CAP_FATAL"):
        recorder.maintain(now_us=102)

    assert adapter.connected is False
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, database_bytes, wal_bytes FROM runtime_state"
        ).fetchone()
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "fatal"
    assert tuple(runtime) == ("fatal", "WAL_CAP_FATAL", 100, 64)


def test_maintenance_database_failure_preserves_admitted_evidence_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )

    class FailingRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            raise sqlite3.OperationalError("disk I/O error after callback admission")

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", FailingRetention)

    with pytest.raises(RecorderFatalError, match="retention invariant failed"):
        recorder.maintain(now_us=102)

    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert callback[0] == "pending" and callback[1] is not None
    assert tuple(runtime) == ("fatal", "RETENTION_INVARIANT_FAILED", "disconnected")
    assert adapter.connected is False
    assert set(adapter.cancelled) == {3, 4}


def test_regular_session_maintenance_measures_caps_without_heavy_work_or_false_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class CapMeasurementOnly:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.NORMAL, 123, 4, None

        def run(self, **_kwargs: object) -> RetentionResult:
            raise AssertionError("regular-session maintenance must not run heavy retention")

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", CapMeasurementOnly)

    assert recorder.maintain(now_us=101, retention_work_expected=False) is StorageCapState.NORMAL
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )
    assert recorder.drain(now_us=101) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT database_bytes, wal_bytes, process_heartbeat_at_us FROM runtime_state"
            ).fetchone()
        ) == (123, 4, 101)
        assert (
            connection.execute(
                "SELECT count(*) FROM incidents WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
                (admitted.source_sequence,),
            ).fetchone()[0]
            == "acknowledged"
        )

    assert recorder._component_failure(
        "retention_maintenance",
        now_us=102,
        error_name="MaintenanceDeadlineExceeded",
    )
    assert recorder.maintain(now_us=103, retention_work_expected=False) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
        runtime = connection.execute(
            "SELECT lifecycle, reason, database_bytes, wal_bytes, process_heartbeat_at_us "
            "FROM runtime_state"
        ).fetchone()
    assert incident["resolved_at_us"] is None
    assert tuple(runtime) == (
        "degraded",
        "COMPONENT_RETENTION_MAINTENANCE_FAILED",
        123,
        4,
        103,
    )

    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=124,
        wal_bytes=5,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class SuccessfulOffSessionRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert callable(precondition)
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        SuccessfulOffSessionRetention,
    )
    assert recorder.maintain(now_us=1_000_103) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT resolved_at_us FROM incidents "
                "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
            ).fetchone()[0]
            == 1_000_103
        )


def test_regular_session_start_propagates_scheduled_retention_deferral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    maintenance_calls: list[tuple[int, bool]] = []
    schedule_calls: list[int] = []

    def observe_maintenance(
        *, now_us: int, retention_work_expected: bool = True
    ) -> StorageCapState:
        maintenance_calls.append((now_us, retention_work_expected))
        return StorageCapState.NORMAL

    monkeypatch.setattr(recorder, "maintain", observe_maintenance)
    monkeypatch.setattr("stocker_runtime.ingestion.recorder.time_ns", lambda: 150_000_000)

    def schedule(at_us: int) -> bool:
        schedule_calls.append(at_us)
        return False

    recorder.start(
        now_us=100,
        instruments=(instrument,),
        subscriptions=specs,
        retention_schedule=schedule,
    )

    assert schedule_calls == [150_000]
    assert maintenance_calls == [(150_000, False)]


def test_regular_session_cap_measurement_contention_degrades_without_stopping_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class ContendedCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            raise sqlite3.OperationalError("database is locked")

        def run(self, **_kwargs: object) -> RetentionResult:
            raise AssertionError("contention must not start heavy retention")

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager", ContendedCapMeasurement
    )

    assert recorder.maintain(now_us=101, retention_work_expected=False) is StorageCapState.NORMAL

    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        incident = connection.execute(
            "SELECT details_json FROM incidents WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert tuple(runtime) == (
        "degraded",
        "COMPONENT_RETENTION_MAINTENANCE_FAILED",
        "connected",
    )
    assert json.loads(incident["details_json"])["error"] == "OperationalError"
    assert adapter.connected is True
    assert adapter.cancelled == []


def test_regular_session_cap_measurement_still_fails_closed_at_hard_wal_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class FatalCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.FATAL, 100, 64, "WAL_CAP_FATAL"

        def run(self, **_kwargs: object) -> RetentionResult:
            raise AssertionError("hard-cap check must fail before heavy retention")

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", FatalCapMeasurement)

    with pytest.raises(RecorderFatalError, match="WAL_CAP_FATAL"):
        recorder.maintain(now_us=101, retention_work_expected=False)

    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == ("fatal", "WAL_CAP_FATAL", "disconnected")
    assert adapter.connected is False


def test_regular_session_degraded_cap_pauses_optional_feed_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class DegradedCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.DEGRADED, 95, 1, None

        def run(self, **_kwargs: object) -> RetentionResult:
            raise AssertionError("degraded cap must not run heavy retention in-session")

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager", DegradedCapMeasurement
    )

    assert recorder.maintain(now_us=101, retention_work_expected=False) is StorageCapState.DEGRADED
    assert recorder.maintain(now_us=102, retention_work_expected=False) is StorageCapState.DEGRADED

    assert adapter.cancelled == [4]
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, database_bytes, wal_bytes FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS", 95, 1)


def test_regular_session_soft_cap_publishes_without_pausing_feeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class SoftCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.SOFT, 85, 1, None

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", SoftCapMeasurement)

    assert recorder.maintain(now_us=101, retention_work_expected=False) is StorageCapState.SOFT
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, database_bytes, wal_bytes FROM runtime_state"
        ).fetchone()
        incidents = connection.execute("SELECT count(*) FROM incidents").fetchone()[0]
    assert tuple(runtime) == ("running", 85, 1)
    assert incidents == 0
    assert adapter.cancelled == []


def test_degraded_cap_still_pauses_optional_feed_after_publication_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    class DegradedCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def checkpoint_and_measure_cap_state(
            self,
        ) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.DEGRADED, 95, 1, None

    def contended_publication(**_kwargs: object) -> bool:
        recorder._component_failure(
            "retention_maintenance",
            now_us=101,
            error_name="OperationalError",
        )
        return False

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager", DegradedCapMeasurement
    )
    monkeypatch.setattr(recorder, "_publish_storage_measurement", contended_publication)

    assert recorder.maintain(now_us=101, retention_work_expected=False) is StorageCapState.DEGRADED

    assert adapter.cancelled == [4]
    with connect_v2(database) as connection:
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        incident = connection.execute(
            "SELECT resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS")
    assert incident["resolved_at_us"] is None


@pytest.mark.parametrize(
    "failure",
    (
        MaintenanceDeadlineExceeded("retention transaction exceeded its configured deadline"),
        sqlite3.OperationalError("database is locked"),
    ),
)
def test_bounded_retention_contention_degrades_then_recovers_without_stopping_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )
    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class DeferredThenSuccessfulRetention:
        calls = 0

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert callable(precondition)
            self.__class__.calls += 1
            if self.calls == 1:
                raise failure
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        DeferredThenSuccessfulRetention,
    )

    assert recorder.maintain(now_us=102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        deferred_runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
    assert tuple(deferred_runtime) == (
        "degraded",
        "COMPONENT_RETENTION_MAINTENANCE_FAILED",
        "connected",
    )
    assert adapter.connected is True
    assert adapter.cancelled == []

    assert recorder.drain(now_us=103) == 1
    with connect_v2(database) as connection:
        published_runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        first_callback = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
    assert tuple(published_runtime) == (
        "degraded",
        "COMPONENT_RETENTION_MAINTENANCE_FAILED",
        "connected",
    )
    assert first_callback[0] == "acknowledged" and first_callback[1] is not None

    assert recorder.maintain(now_us=1_000_102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        recovered_runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state, database_bytes, wal_bytes "
            "FROM runtime_state"
        ).fetchone()
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
    assert tuple(recovered_runtime) == ("running", None, "connected", 100, 1)
    assert adapter.connected is True


def test_retention_transaction_two_failure_is_auditable_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failure = MaintenanceDeadlineExceeded(
        "retention receipt_transaction_2 exceeded its configured deadline"
    )
    failure.__dict__.update(
        retention_phase="receipt_transaction_2",
        receipt_transactions_committed=1,
        receipt_rows_rolled_committed=17,
        raw_callback_payload={"must_not_persist": "private evidence"},
    )
    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class TransactionTwoThenSuccessfulRetention:
        calls = 0

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert callable(precondition)
            self.__class__.calls += 1
            if self.calls == 1:
                raise failure
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        TransactionTwoThenSuccessfulRetention,
    )

    assert recorder.maintain(now_us=102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert json.loads(incident["details_json"]) == {
        "component": "retention_maintenance",
        "consecutive_failures": 1,
        "error": "MaintenanceDeadlineExceeded",
        "next_retry_at_us": 1_000_102,
        "receipt_rows_rolled_committed": 17,
        "receipt_transactions_committed": 1,
        "retention_phase": "receipt_transaction_2",
    }
    assert incident["resolved_at_us"] is None

    assert recorder.maintain(now_us=1_000_102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        recovered = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state WHERE run_id='run-1'"
        ).fetchone()
    assert json.loads(recovered["details_json"])["retention_phase"] == "receipt_transaction_2"
    assert recovered["resolved_at_us"] == 1_000_102
    assert tuple(runtime) == ("running", None, "connected")


def test_retention_split_phase_diagnostics_are_bounded_and_sanitized() -> None:
    failure = MaintenanceDeadlineExceeded("pruning deadline")
    failure.__dict__.update(
        retention_phase="expired_evidence_pruning",
        receipt_transactions_committed=2,
        receipt_rows_rolled_committed=233,
        terminalizations_committed=3,
        payloads_compacted_committed=1_997,
        expired_rows_deleted_committed=0,
        arbitrary_private_value={"must_not_persist": True},
    )

    assert Recorder._retention_incident_details(failure) == {
        "expired_rows_deleted_committed": 0,
        "payloads_compacted_committed": 1_997,
        "receipt_rows_rolled_committed": 233,
        "receipt_transactions_committed": 2,
        "retention_phase": "expired_evidence_pruning",
        "terminalizations_committed": 3,
    }


def test_generic_pruning_failure_reports_committed_compaction_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    pruning_failure = RuntimeError("pruning dependency failed")
    pruning_failure.__dict__.update(
        retention_phase="expired_evidence_pruning",
        receipt_transactions_committed=2,
        receipt_rows_rolled_committed=233,
        terminalizations_committed=3,
        payloads_compacted_committed=1_997,
        expired_rows_deleted_committed=0,
    )
    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class FailedPruningThenSuccessfulRetention:
        calls = 0

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert callable(precondition)
            self.__class__.calls += 1
            if self.calls == 1:
                raise pruning_failure
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        FailedPruningThenSuccessfulRetention,
    )

    assert recorder.maintain(now_us=102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    details = json.loads(incident["details_json"])
    assert details["error"] == "RuntimeError"
    assert details["retention_phase"] == "expired_evidence_pruning"
    assert details["terminalizations_committed"] == 3
    assert details["payloads_compacted_committed"] == 1_997
    assert details["expired_rows_deleted_committed"] == 0
    assert incident["resolved_at_us"] is None

    assert recorder.maintain(now_us=1_000_102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        recovered = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert json.loads(recovered["details_json"])["payloads_compacted_committed"] == 1_997
    assert recovered["resolved_at_us"] == 1_000_102


def test_unpublished_retention_progress_survives_a_later_generic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    transaction_two = MaintenanceDeadlineExceeded(
        "retention receipt_transaction_2 exceeded its configured deadline"
    )
    transaction_two.__dict__.update(
        retention_phase="receipt_transaction_2",
        receipt_transactions_committed=1,
        receipt_rows_rolled_committed=17,
    )
    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class DiagnosticThenGenericThenSuccessfulRetention:
        calls = 0

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert callable(precondition)
            self.__class__.calls += 1
            if self.calls == 1:
                raise transaction_two
            if self.calls == 2:
                raise sqlite3.OperationalError("database is locked")
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        DiagnosticThenGenericThenSuccessfulRetention,
    )
    persist_incident = recorder._persist_component_incident
    publication_attempts = 0

    def contend_once(component: str, *, now_us: int) -> bool:
        nonlocal publication_attempts
        publication_attempts += 1
        if publication_attempts == 1:
            return False
        return persist_incident(component, now_us=now_us)

    monkeypatch.setattr(recorder, "_persist_component_incident", contend_once)

    assert recorder.maintain(now_us=102) is StorageCapState.NORMAL
    assert recorder.maintain(now_us=103) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert json.loads(incident["details_json"]) == {
        "component": "retention_maintenance",
        "consecutive_failures": 2,
        "error": "OperationalError",
        "next_retry_at_us": 2_000_103,
        "receipt_rows_rolled_committed": 17,
        "receipt_transactions_committed": 1,
        "retention_phase": "receipt_transaction_2",
    }
    assert incident["resolved_at_us"] is None

    assert recorder.maintain(now_us=2_000_103) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        recovered = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED'"
        ).fetchone()
    assert json.loads(recovered["details_json"])["retention_phase"] == ("receipt_transaction_2")
    assert recovered["resolved_at_us"] == 2_000_103


@pytest.mark.parametrize(
    ("contention_stage", "result_state"),
    (
        ("retention", StorageCapState.NORMAL),
        ("result_publication", StorageCapState.NORMAL),
        ("result_publication", StorageCapState.DEGRADED),
    ),
)
def test_retention_contention_with_held_writer_never_uses_a_second_writer_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contention_stage: str,
    result_state: StorageCapState,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    normal = RetentionResult(
        cap_state=result_state,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=result_state is StorageCapState.NORMAL,
        required_action=(
            None if result_state is StorageCapState.NORMAL else "PAUSE_OPTIONAL_FEEDS"
        ),
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class HeldWriterRetention:
        connection: sqlite3.Connection | None = None

        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            self.__class__.connection = connect_v2(database)
            self.connection.execute("BEGIN IMMEDIATE")
            if contention_stage == "retention":
                raise sqlite3.OperationalError("database is locked")
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        HeldWriterRetention,
    )

    try:
        assert recorder.maintain(now_us=102) is result_state
        with connect_v2(database) as observer:
            runtime = observer.execute(
                "SELECT lifecycle, reason, connection_state FROM runtime_state"
            ).fetchone()
            assert observer.execute("SELECT status FROM runs").fetchone()[0] == "running"
        assert tuple(runtime) == ("running", None, "connected")
        assert adapter.connected is True
        assert adapter.cancelled == []
    finally:
        if HeldWriterRetention.connection is not None:
            HeldWriterRetention.connection.rollback()
            HeldWriterRetention.connection.close()

    recorder._heartbeat(103)
    with connect_v2(database) as connection:
        published = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(published) == (
        "degraded",
        "RETENTION_MAINTENANCE_DEFERRED",
        "connected",
    )


def test_retention_deferral_waits_for_an_existing_degradation_to_clear(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runtime_state SET lifecycle='degraded', reason='OTHER_DEGRADED'")
    recorder._retention_maintenance_deferred = True

    recorder._heartbeat(101)
    assert recorder._retention_maintenance_deferred is True
    with connect_v2(database) as connection:
        blocked = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        connection.execute("UPDATE runtime_state SET lifecycle='running', reason=NULL")
    assert tuple(blocked) == ("degraded", "OTHER_DEGRADED")

    recorder._heartbeat(102)
    assert recorder._retention_maintenance_deferred is False
    with connect_v2(database) as connection:
        published = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
    assert tuple(published) == ("degraded", "RETENTION_MAINTENANCE_DEFERRED")


def test_backup_status_failure_is_degraded_and_recovers_without_stopping_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    normal = RetentionResult(
        cap_state=StorageCapState.NORMAL,
        database_bytes=100,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=True,
        required_action=None,
        checkpoint_attempted=True,
        incremental_vacuum_attempted=True,
    )

    class SuccessfulRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            return normal

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        SuccessfulRetention,
    )
    monkeypatch.setattr(
        recorder,
        "_sync_backup_status",
        lambda *, now_us: (_ for _ in ()).throw(sqlite3.IntegrityError("invariant collision")),
    )

    assert recorder.maintain(now_us=102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        run_status = connection.execute("SELECT status FROM runs").fetchone()[0]
        incident = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='COMPONENT_BACKUP_MAINTENANCE_FAILED'"
        ).fetchone()
    assert tuple(runtime) == (
        "degraded",
        "COMPONENT_BACKUP_MAINTENANCE_FAILED",
        "connected",
    )
    assert run_status == "running"
    assert incident[0] is None
    assert adapter.connected is True

    monkeypatch.setattr(recorder, "_sync_backup_status", lambda *, now_us: None)
    assert recorder.maintain(now_us=1_000_102) is StorageCapState.NORMAL
    with connect_v2(database) as connection:
        recovered = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        resolved = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='COMPONENT_BACKUP_MAINTENANCE_FAILED'"
        ).fetchone()[0]
    assert tuple(recovered) == ("running", None)
    assert resolved == 1_000_102


def test_sqlite_error_code_controls_retention_contention_classification() -> None:
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
    assert Recorder._is_sqlite_contention(error) is False


def test_backward_callback_receive_order_is_a_persisted_global_fatal(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)
    inbox.admit(_fence(), MarketDataCallback("quote", 11, None, {"event_at_us": 11}))

    with pytest.raises(Exception, match="backwards"):
        inbox.admit(_fence(), MarketDataCallback("quote", 10, None, {"event_at_us": 10}))

    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "fatal"


def test_runtime_ingestion_does_not_import_legacy_or_cross_vendor_modules() -> None:
    import ast

    root = Path("packages/stocker_runtime/src/stocker_runtime/ingestion")
    imported: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(
        name.startswith("stocker_prospective") or "eodhd" in name.lower() for name in imported
    )


def test_stale_writer_from_another_run_is_closed_before_takeover(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    first.abandon_unclean()

    second = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=15_000_000,
        ),
        FakeMarketData(),
    ).start(now_us=15_000_101, instruments=(instrument,), subscriptions=specs)

    assert second.run_id == "run-2"
    with connect_v2(database) as connection:
        runs = dict(connection.execute("SELECT run_id, status FROM runs"))
        states = dict(connection.execute("SELECT run_id, lifecycle FROM runtime_state"))
        old_subscriptions = tuple(
            connection.execute(
                "SELECT lifecycle, closed_at_us FROM subscriptions WHERE run_id='run-1'"
            )
        )
        replacement_subscriptions = tuple(
            connection.execute(
                "SELECT lifecycle, closed_at_us FROM subscriptions WHERE run_id='run-2'"
            )
        )
    assert runs == {"run-1": "stopped", "run-2": "running"}
    assert states == {"run-1": "stopped", "run-2": "running"}
    assert all(tuple(row) == ("closed", 15_000_101) for row in old_subscriptions)
    assert all(tuple(row) == ("active", None) for row in replacement_subscriptions)


@pytest.mark.parametrize(
    "stale_lifecycle", ("active", "connecting", "disconnected", "degraded", "paused")
)
def test_stale_takeover_closes_every_owned_subscription_and_scopes_uncertainty(
    tmp_path: Path, stale_lifecycle: str
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        if stale_lifecycle == "paused":
            connection.execute("UPDATE subscriptions SET lifecycle='paused' WHERE request_id=4")
        else:
            connection.execute(
                "UPDATE subscriptions SET lifecycle=? WHERE recorder_generation=1",
                (stale_lifecycle,),
            )
        required_id = connection.execute(
            "SELECT subscription_id FROM subscriptions WHERE request_id=3"
        ).fetchone()[0]
        Recorder._open_gap_for_run(
            connection,
            "run-1",
            str(required_id),
            99,
            "EXISTING_SCIENTIFIC_GAP",
            True,
        )
    first.abandon_unclean()

    replacement = Recorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=15_000_000),
        FakeMarketData(),
    )
    replacement.start(now_us=15_000_101, instruments=(instrument,), subscriptions=specs)

    with connect_v2(database) as connection:
        old = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE recorder_generation=1"
            )
        )
        old_closed = {
            int(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT request_id, closed_at_us FROM subscriptions WHERE recorder_generation=1"
            )
        }
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE recorder_generation=2"
            )
        )
        current_closed = tuple(
            connection.execute("SELECT closed_at_us FROM subscriptions WHERE recorder_generation=2")
        )
        uncertainty = {
            int(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT subscription.request_id, gap.continuity_required FROM gaps gap "
                "JOIN subscriptions subscription "
                "ON subscription.subscription_id=gap.subscription_id "
                "WHERE subscription.recorder_generation=1 "
                "AND gap.reason='UNCLEAN_RECORDER_RESTART'"
            )
        }
        existing = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='EXISTING_SCIENTIFIC_GAP'"
        ).fetchone()
    assert old == {3: "closed", 4: "closed"}
    assert old_closed == {3: 15_000_101, 4: 15_000_101}
    assert current == {3: "active", 4: "active"}
    assert all(row[0] is None for row in current_closed)
    assert uncertainty == ({3: 1} if stale_lifecycle == "paused" else {3: 1, 4: 0})
    assert existing is not None and existing[0] is None


def test_stale_takeover_closure_is_idempotent_and_old_rows_are_retention_prunable(
    tmp_path: Path,
) -> None:
    from stocker_runtime.storage import RetentionManager, RetentionPolicy

    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute("UPDATE subscriptions SET lifecycle='disconnected'")
    first.abandon_unclean()

    class RepeatedCloseRecorder(Recorder):
        def _close_stale_writer(self, *args: object, **kwargs: object) -> None:
            super()._close_stale_writer(*args, **kwargs)  # type: ignore[arg-type]
            super()._close_stale_writer(*args, **kwargs)  # type: ignore[arg-type]

    RepeatedCloseRecorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=15_000_000),
        FakeMarketData(),
    ).start(now_us=15_000_101, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM gaps WHERE reason='UNCLEAN_RECORDER_RESTART'"
            ).fetchone()[0]
            == 2
        )

    RetentionManager(
        database,
        RetentionPolicy(closed_subscription_us=1),
    ).run(now_us=15_000_103, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        generations = dict(
            connection.execute(
                "SELECT recorder_generation, count(*) FROM subscriptions "
                "GROUP BY recorder_generation"
            )
        )
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE recorder_generation=2"
            )
        )
    assert generations == {2: 2}
    assert current == {3: "active", 4: "active"}


def test_late_prior_run_callback_is_failed_and_receipted_by_current_recorder(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    prior_recorder = Recorder(_config(database), FakeMarketData())
    prior = prior_recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    prior_recorder.abandon_unclean()
    current_recorder = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=15_000_000,
        ),
        FakeMarketData(),
    )
    current_recorder.start(
        now_us=15_000_101,
        instruments=(instrument,),
        subscriptions=specs,
    )

    callback = MarketDataCallback("quote", 15_000_102, None, {"event_at_us": 15_000_102})
    first = current_recorder.receive(
        prior.fences[0],
        callback,
    )
    retry = current_recorder.receive(prior.fences[0], callback)
    assert current_recorder.drain(now_us=15_000_103) == 0

    with connect_v2(database) as connection:
        durable = connection.execute(
            "SELECT lifecycle, receipt_batch_id FROM callback_inbox WHERE run_id='run-1'"
        ).fetchone()
        connection.execute(
            "UPDATE callback_inbox SET payload_json=NULL WHERE event_uid=?",
            (first.event_uid,),
        )
    compacted_retry = current_recorder.receive(prior.fences[0], callback)
    assert durable["lifecycle"] == "failed"
    assert durable["receipt_batch_id"] is not None
    assert first.inserted is True
    assert retry.event_uid == first.event_uid
    assert retry.inserted is False
    assert compacted_retry == retry


def test_deterministic_callback_identity_collision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    inbox = CallbackInbox(database)
    monkeypatch.setattr("stocker_runtime.ingestion.inbox._event_uid", lambda *_: "f" * 64)
    inbox.admit(_fence(), MarketDataCallback("quote", 10, None, {"event_at_us": 10}))

    with pytest.raises(Exception, match="different content"):
        inbox.admit(_fence(), MarketDataCallback("quote", 11, None, {"event_at_us": 11}))

    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_inbox").fetchone()[0] == 1


def test_official_callback_bridge_defines_no_authority_callbacks() -> None:
    import ast

    path = Path("packages/stocker_runtime/src/stocker_runtime/ingestion/official_bridge.py")
    names = {
        node.name.lower()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    forbidden = ("order", "account", "position", "execution", "pnl", "fill", "portfolio")
    assert not any(term in name for name in names for term in forbidden)


def test_official_tick_semantics_do_not_mix_quotes_and_trades() -> None:
    from stocker_runtime.ingestion.official_bridge import (
        _price_tick_projection,
        _size_tick_projection,
    )

    assert _price_tick_projection("quotes", 1) == ("quote", "bid")
    assert _price_tick_projection("quotes", 2) == ("quote", "ask")
    assert _price_tick_projection("quotes", 9) == ("quote", "close")
    assert _price_tick_projection("quotes", 4) is None
    assert _price_tick_projection("trades", 4) == ("trade", "last")
    assert _size_tick_projection("quotes", 3) == ("quote", "ask_size")
    assert _size_tick_projection("quotes", 27) == ("quote", "call_open_interest")
    assert _size_tick_projection("quotes", 28) == ("quote", "put_open_interest")
    assert _size_tick_projection("quotes", 29) == ("quote", "call_option_volume")
    assert _size_tick_projection("quotes", 30) == ("quote", "put_option_volume")
    assert _size_tick_projection("trades", 3) is None
    assert _size_tick_projection("trades", 5) == ("trade", "size")


def test_official_wrapper_translates_realistic_market_data_sequence_without_broker_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion import IBKRSubscription
    from stocker_runtime.ingestion.official_bridge import create_official_bridge

    clients: list[object] = []
    release_reader = threading.Event()

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper = wrapper
            self.requests: list[tuple[str, int]] = []
            self.generic_ticks: dict[int, str] = {}
            self.cancelled: list[tuple[str, int]] = []
            self.disconnected = False
            self.fail_next_cancel = False
            clients.append(self)

        def connect(self, host: str, port: int, client_id: int) -> bool:
            assert (host, port, client_id) == ("127.0.0.1", 4001, 71)
            return True

        def run(self) -> None:
            self.wrapper.nextValidId(1)
            assert release_reader.wait(timeout=5)

        def disconnect(self) -> None:
            self.disconnected = True
            release_reader.set()

        def reqMktData(  # noqa: N802
            self,
            request_id: int,
            _contract: object,
            generic_ticks: str,
            *_args: object,
        ) -> None:
            self.requests.append(("market", request_id))
            self.generic_ticks[request_id] = generic_ticks

        def reqRealTimeBars(self, request_id: int, *_args: object) -> None:  # noqa: N802
            self.requests.append(("bars", request_id))

        def cancelMktData(self, request_id: int) -> None:  # noqa: N802
            self.cancelled.append(("market", request_id))
            self.wrapper.tickPrice(request_id, 1, 998.0, object())
            self.wrapper.error(request_id, 420, "synchronous pacing during cancel")
            if self.fail_next_cancel:
                self.fail_next_cancel = False
                raise RuntimeError("injected cancel failure")

        def cancelRealTimeBars(self, request_id: int) -> None:  # noqa: N802
            self.cancelled.append(("bars", request_id))

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
        subscriptions=(
            IBKRSubscription(3, 123, "AAPL", "STK", "SMART", "USD", "quotes"),
            IBKRSubscription(4, 123, "AAPL", "STK", "SMART", "USD", "trades"),
            IBKRSubscription(5, 123, "AAPL", "STK", "SMART", "USD", "bars"),
            IBKRSubscription(6, 123, "AAPL", "OPT", "SMART", "USD", "quotes", snapshot=True),
        ),
    )
    callbacks: list[tuple[CallbackFence, MarketDataCallback]] = []
    statuses: list[MarketDataStatus] = []
    disconnects: list[int] = []

    def capture(fence: CallbackFence, callback: MarketDataCallback) -> AdmissionResult:
        callbacks.append((fence, callback))
        return AdmissionResult(len(callbacks), f"event-{len(callbacks)}", True)

    bridge.set_callback(capture)
    bridge.set_status_callback(statuses.append)
    bridge.set_disconnect_callback(disconnects.append)
    bridge.connect()
    for request_id in (3, 4, 5, 6):
        bridge.subscribe(CallbackFence("run-1", 1, 1, request_id, f"sub-{request_id}"))

    client = clients[0]
    wrapper = client.wrapper  # type: ignore[attr-defined]
    private_bridge = cast(Any, bridge)
    private_bridge._callback_context.connection_epoch = private_bridge._active_connection_epoch
    wrapper.tickPrice(3, 1, 100.0, object())
    wrapper.tickPrice(3, 2, 101.0, object())
    wrapper.tickSize(3, 0, 10)
    wrapper.tickSize(3, 27, 999)
    wrapper.tickOptionComputation(3, 13, object(), 0.9, 0.1, 9.0, 0.0, 0.1, 0.1, 0.1, 100.0)
    wrapper.tickPrice(3, 4, 999.0, object())
    wrapper.tickPrice(4, 4, 100.25, object())
    wrapper.tickSize(4, 5, 3)
    wrapper.tickPrice(6, 1, 2.0, object())
    wrapper.tickPrice(6, 2, 2.2, object())
    wrapper.tickPrice(6, 9, 100.0, object())
    wrapper.tickSize(6, 27, 150)
    wrapper.tickSize(6, 29, 25)
    wrapper.tickOptionComputation(6, 13, object(), 0.4, 0.52, 2.1, 0.0, 0.01, 0.1, -0.02, 100.0)
    wrapper.tickSnapshotEnd(6)
    wrapper.realtimeBar(5, 1_700_000_000, 99.0, 101.0, 98.0, 100.0, 50, 0, 2)
    wrapper.error(3, 420, "pacing")
    wrapper.error(-1, 2103, "market farm disconnected")
    wrapper.error(3, 9999, "ignored")
    callbacks_before_retry_evidence = len(callbacks)
    statuses_before_retry_evidence = len(statuses)
    client.fail_next_cancel = True  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="injected cancel failure"):
        bridge.retry_subscription(CallbackFence("run-1", 1, 1, 3, "sub-3"))
    assert len(callbacks) == callbacks_before_retry_evidence
    assert len(statuses) == statuses_before_retry_evidence
    bridge.retry_subscription(CallbackFence("run-1", 1, 1, 3, "sub-3"))
    assert len(callbacks) == callbacks_before_retry_evidence
    assert len(statuses) == statuses_before_retry_evidence
    wrapper.tickPrice(3, 1, 999.0, object())
    wrapper.error(3, 420, "late pacing from canceled request")
    assert len(callbacks) == callbacks_before_retry_evidence
    assert len(statuses) == statuses_before_retry_evidence
    wrapper.tickPrice(1_499_999_999, 1, 102.0, object())
    replacement_callback = callbacks.pop()
    assert replacement_callback[0] == CallbackFence("run-1", 1, 1, 3, "sub-3")
    assert replacement_callback[1].payload["bid"] == 102.0
    wrapper.connectionClosed()
    del private_bridge._callback_context.connection_epoch

    assert [(item.callback_kind, item.payload) for _, item in callbacks] == [
        ("quote", {"event_at_us": callbacks[0][1].received_at_us, "bid": 100.0}),
        ("quote", {"event_at_us": callbacks[1][1].received_at_us, "ask": 101.0}),
        ("quote", {"event_at_us": callbacks[2][1].received_at_us, "bid_size": 10.0}),
        ("trade", {"event_at_us": callbacks[3][1].received_at_us, "last": 100.25}),
        ("trade", {"event_at_us": callbacks[4][1].received_at_us, "size": 3.0}),
        ("quote", {"event_at_us": callbacks[5][1].received_at_us, "bid": 2.0}),
        ("quote", {"event_at_us": callbacks[6][1].received_at_us, "ask": 2.2}),
        ("quote", {"event_at_us": callbacks[7][1].received_at_us, "close": 100.0}),
        ("quote", {"event_at_us": callbacks[8][1].received_at_us, "call_open_interest": 150.0}),
        ("quote", {"event_at_us": callbacks[9][1].received_at_us, "call_option_volume": 25.0}),
        (
            "option_computation",
            {
                "event_at_us": callbacks[10][1].received_at_us,
                "tick_type": 13,
                "implied_volatility": 0.4,
                "delta": 0.52,
                "option_price": 2.1,
                "present_value_dividend": 0.0,
                "gamma": 0.01,
                "vega": 0.1,
                "theta": -0.02,
                "underlying_price": 100.0,
            },
        ),
        (
            "option_snapshot_end",
            {"event_at_us": callbacks[11][1].received_at_us, "complete": True},
        ),
        (
            "bar",
            {
                "event_at_us": 1_700_000_000_000_000,
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 50.0,
            },
        ),
    ]
    assert [(status.kind, status.code, status.request_id) for status in statuses] == [
        ("snapshot_end", 0, 6),
        ("pacing", 420, 3),
        ("farm_degraded", 2103, None),
    ]
    assert len(disconnects) == 1
    assert client.requests == [  # type: ignore[attr-defined]
        ("market", 3),
        ("market", 4),
        ("bars", 5),
        ("market", 6),
        ("market", 1_499_999_999),
    ]
    assert client.cancelled == [("market", 3), ("market", 3)]  # type: ignore[attr-defined]
    assert client.requests.count(("market", 4)) == 1  # type: ignore[attr-defined]
    assert client.generic_ticks == {  # type: ignore[attr-defined]
        3: "",
        4: "",
        6: "100,101",
        1_499_999_999: "",
    }
    bridge.disconnect()
    wrapper.tickSnapshotEnd(6)
    assert [(status.kind, status.code, status.request_id) for status in statuses] == [
        ("snapshot_end", 0, 6),
        ("pacing", 420, 3),
        ("farm_degraded", 2103, None),
    ]
    assert private_bridge._fences == {}
    assert private_bridge._configured == {}
    assert private_bridge._contracts == {}
    assert client.disconnected is True  # type: ignore[attr-defined]


def test_official_bridge_drains_intentional_close_and_fences_old_socket_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion import IBKRSubscription
    from stocker_runtime.ingestion.official_bridge import create_official_bridge

    clients: list[object] = []
    run_started = (threading.Event(), threading.Event())
    release_run = (threading.Event(), threading.Event())
    run_finished = (threading.Event(), threading.Event())
    auxiliary_started = threading.Event()
    release_auxiliary = threading.Event()
    auxiliary_finished = threading.Event()

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper: Any = wrapper
            self.run_count = 0
            self.active_run = -1
            clients.append(self)

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            index = self.run_count
            self.run_count += 1
            self.active_run = index
            self.wrapper.nextValidId(1)
            run_started[index].set()
            if index == 0:

                def delayed_auxiliary_callback() -> None:
                    auxiliary_started.set()
                    assert release_auxiliary.wait(timeout=5)
                    self.wrapper.tickPrice(3, 1, 99.0, object())
                    self.wrapper.error(-1, 2103, "untagged retired farm status")
                    self.wrapper.connectionClosed()
                    auxiliary_finished.set()

                threading.Thread(target=delayed_auxiliary_callback, daemon=True).start()
            assert release_run[index].wait(timeout=5)
            if index == 0:
                self.wrapper.error(-1, 2103, "late status from intentionally closed socket")
            else:
                self.wrapper.tickPrice(3, 1, 101.0, object())
            self.wrapper.connectionClosed()
            run_finished[index].set()

        def disconnect(self) -> None:
            if self.active_run >= 0:
                release_run[self.active_run].set()

        def reqMktData(self, _request_id: int, *_args: object) -> None:  # noqa: N802
            return None

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    subscription = IBKRSubscription(3, 123, "AAPL", "STK", "SMART", "USD", "quotes")
    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
        subscriptions=(subscription,),
    )
    callbacks: list[tuple[CallbackFence, MarketDataCallback]] = []
    statuses: list[MarketDataStatus] = []
    disconnects: list[int] = []
    natural_disconnect_seen = threading.Event()
    bridge.set_callback(
        lambda fence, callback: (
            callbacks.append((fence, callback))
            or AdmissionResult(len(callbacks), f"event-{len(callbacks)}", True)
        )
    )
    bridge.set_status_callback(statuses.append)

    def disconnected(at_us: int) -> None:
        disconnects.append(at_us)
        natural_disconnect_seen.set()

    bridge.set_disconnect_callback(disconnected)
    bridge.connect()
    assert run_started[0].wait(timeout=5)
    assert auxiliary_started.wait(timeout=5)
    bridge.subscribe(CallbackFence("run", 1, 1, 3, "subscription-1"))

    bridge.disconnect()

    assert run_finished[0].is_set()
    assert statuses == []
    assert disconnects == []

    bridge.configure_subscriptions((subscription,))
    bridge.connect()
    assert run_started[1].wait(timeout=5)
    bridge.subscribe(CallbackFence("run", 1, 2, 3, "subscription-2"))
    release_auxiliary.set()
    assert auxiliary_finished.wait(timeout=5)
    assert callbacks == []
    assert statuses == []
    assert disconnects == []
    release_run[1].set()
    assert natural_disconnect_seen.wait(timeout=5)
    assert run_finished[1].wait(timeout=5)
    assert len(callbacks) == 1
    assert callbacks[0][0] == CallbackFence("run", 1, 2, 3, "subscription-2")
    assert callbacks[0][1].payload["bid"] == 101.0
    assert len(disconnects) == 1
    bridge.disconnect()


def test_official_bridge_waits_for_session_ready_and_reports_reader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion.official_bridge import create_official_bridge

    run_started = threading.Event()
    release_ready = threading.Event()
    release_reader = threading.Event()

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper: Any = wrapper

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            run_started.set()
            assert release_ready.wait(timeout=5)
            self.wrapper.nextValidId(1)
            assert release_reader.wait(timeout=5)

        def disconnect(self) -> None:
            release_reader.set()

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
    )
    connect_finished = threading.Event()
    connect_errors: list[BaseException] = []
    disconnected = threading.Event()
    bridge.set_disconnect_callback(lambda _at_us: disconnected.set())

    def connect() -> None:
        try:
            bridge.connect()
        except BaseException as error:
            connect_errors.append(error)
        finally:
            connect_finished.set()

    connecting = threading.Thread(target=connect)
    connecting.start()
    assert run_started.wait(timeout=5)
    returned_before_ready = connect_finished.wait(timeout=0.25)
    release_ready.set()
    connecting.join(timeout=5)
    assert not connecting.is_alive()
    assert returned_before_ready is False
    assert connect_errors == []

    release_reader.set()
    assert disconnected.wait(timeout=5)
    private_bridge = cast(Any, bridge)
    assert private_bridge._active_connection_epoch is None
    assert private_bridge._thread is None
    bridge.disconnect()


def test_official_bridge_session_readiness_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion.official_bridge import (
        OfficialBridgeUnavailable,
        create_official_bridge,
    )

    release_reader = threading.Event()

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, _wrapper: object) -> None:
            return None

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            release_reader.wait(timeout=5)

        def disconnect(self) -> None:
            release_reader.set()

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
    )
    private_bridge = cast(Any, bridge)
    private_bridge._SESSION_READY_TIMEOUT_SECONDS = 0.01

    with pytest.raises(OfficialBridgeUnavailable, match="session readiness timed out"):
        bridge.connect()
    assert private_bridge._active_connection_epoch is None
    assert private_bridge._thread is None


def test_official_bridge_unwinds_reader_thread_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion.official_bridge import create_official_bridge

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, _wrapper: object) -> None:
            self.disconnected = False

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            return None

        def disconnect(self) -> None:
            self.disconnected = True

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )
    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
    )
    original_start = threading.Thread.start

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("synthetic reader thread start failure")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="synthetic reader thread start failure"):
        bridge.connect()

    private_bridge = cast(Any, bridge)
    assert private_bridge._active_connection_epoch is None
    assert private_bridge._thread is None
    bridge.disconnect()
    monkeypatch.setattr(threading.Thread, "start", original_start)


def test_official_bridge_discovers_one_exact_option_before_dynamic_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion import IBKRSubscription
    from stocker_runtime.ingestion.official_bridge import create_official_bridge

    clients: list[object] = []
    release_reader = threading.Event()

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper: Any = wrapper
            self.metadata_requests: list[tuple[str, int]] = []
            self.market_requests: list[tuple[int, bool]] = []
            self.cancelled: list[int] = []
            clients.append(self)

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            self.wrapper.nextValidId(1)
            assert release_reader.wait(timeout=5)

        def disconnect(self) -> None:
            release_reader.set()

        def reqSecDefOptParams(  # noqa: N802
            self, request_id: int, symbol: str, *_args: object
        ) -> None:
            self.metadata_requests.append(("parameters", request_id))
            self.wrapper.securityDefinitionOptionParameter(
                request_id,
                "SMART",
                265598,
                symbol,
                "100",
                {"20260811"},
                {99.0, 100.0, 101.0},
            )
            self.wrapper.securityDefinitionOptionParameterEnd(request_id)

        def reqContractDetails(self, request_id: int, contract: Any) -> None:  # noqa: N802
            self.metadata_requests.append(("contract", request_id))
            contract.conId = 9001
            self.wrapper.contractDetails(request_id, types.SimpleNamespace(contract=contract))
            self.wrapper.contractDetailsEnd(request_id)

        def reqMktData(  # noqa: N802
            self,
            request_id: int,
            _contract: object,
            _generic_ticks: str,
            snapshot: bool,
            *_args: object,
        ) -> None:
            self.market_requests.append((request_id, snapshot))

        def cancelMktData(self, request_id: int) -> None:  # noqa: N802
            self.cancelled.append(request_id)

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )
    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
    )
    option_bridge = cast(OptionDiscoveryBackend, bridge)
    client = clients[0]
    bridge.connect()
    private_bridge = cast(Any, bridge)
    private_bridge._callback_context.connection_epoch = private_bridge._active_connection_epoch

    parameters = option_bridge.option_parameters(underlying_con_id=265598, symbol="AAPL")
    contracts = option_bridge.option_contracts(
        symbol="AAPL",
        expiry="20260811",
        strike=100.0,
        right="C",
        multiplier="100",
        trading_class="AAPL",
    )

    assert parameters[0].expirations == ("20260811",)
    assert contracts[0].con_id == 9001
    assert client.market_requests == []  # type: ignore[attr-defined]
    subscription = IBKRSubscription(
        2_000_001, 9001, "AAPL", "OPT", "SMART", "USD", "quotes", snapshot=True
    )
    statuses: list[MarketDataStatus] = []
    bridge.set_status_callback(statuses.append)
    bridge.configure_subscriptions((subscription,))
    fence = CallbackFence("run", 1, 1, subscription.request_id, "dynamic")
    bridge.subscribe(fence)
    client.wrapper.tickSnapshotEnd(subscription.request_id)  # type: ignore[attr-defined]
    del private_bridge._callback_context.connection_epoch
    bridge.cancel(subscription.request_id)
    assert client.market_requests == [(2_000_001, True)]  # type: ignore[attr-defined]
    bridge.disconnect()
    assert client.cancelled == []  # type: ignore[attr-defined]
    assert [status.kind for status in statuses] == ["snapshot_end"]
    assert private_bridge._configured == {}
    assert private_bridge._contracts == {}
    stream = IBKRSubscription(
        2_000_002, 9001, "AAPL", "OPT", "SMART", "USD", "quotes", snapshot=False
    )
    bridge.configure_subscriptions((stream,))
    bridge.subscribe(CallbackFence("run", 1, 1, stream.request_id, "dynamic-stream"))
    bridge.cancel(stream.request_id)
    assert client.cancelled == [2_000_002]  # type: ignore[attr-defined]
    assert private_bridge._configured == {}
    assert private_bridge._contracts == {}
    assert client.metadata_requests == [  # type: ignore[attr-defined]
        ("parameters", 1_500_000_000),
        ("contract", 1_500_000_001),
    ]


def test_official_bridge_option_metadata_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_runtime.ingestion.official_bridge import (
        OfficialBridgeUnavailable,
        create_official_bridge,
    )

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper = wrapper

        def reqSecDefOptParams(self, *_args: object) -> None:  # noqa: N802
            return None

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    def bounded_wait(_self: threading.Event, timeout: float | None = None) -> bool:
        assert timeout == 5.0
        return False

    monkeypatch.setattr(threading.Event, "wait", bounded_wait)

    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4001,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
    )
    option_bridge = cast(OptionDiscoveryBackend, bridge)

    with pytest.raises(OfficialBridgeUnavailable, match="metadata request timed out"):
        option_bridge.option_parameters(underlying_con_id=265598, symbol="AAPL")


def test_stale_recorder_object_cannot_mutate_or_touch_adapter_after_takeover(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 2, 'owner-2', 101)"
        )
        connection.execute(
            "UPDATE runtime_state SET recorder_generation=2, lifecycle='running', "
            "connection_state='connected', process_heartbeat_at_us=101 WHERE run_id='run-1'"
        )
    disconnects = adapter.disconnect_calls
    for action in (
        lambda: recorder.disconnected(now_us=102),
        lambda: recorder.reconnect(now_us=102),
        lambda: recorder.stop(now_us=102),
        lambda: recorder._fatal("STALE_MUST_NOT_WIN", 102),
    ):
        with pytest.raises(AuthoritativeLeaseLost):
            action()
    assert adapter.disconnect_calls == disconnects
    assert adapter.cancelled == []
    with connect_v2(database) as connection:
        state = connection.execute(
            "SELECT recorder_generation, lifecycle FROM runtime_state WHERE run_id='run-1'"
        ).fetchone()
    assert tuple(state) == (2, "running")


def test_persisted_fatal_blocks_same_run_but_not_an_unrelated_new_run(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder._fatal("TEST_FATAL", 101)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.reconnect(now_us=102)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.receive(
            state.fences[0],
            MarketDataCallback("quote", 102, None, {"event_at_us": 102}),
        )
    recorder.abandon_unclean()
    with pytest.raises(RecorderFatalError, match="operator action"):
        Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
            now_us=103, instruments=(instrument,), subscriptions=specs
        )
    unrelated = Recorder(
        _config(database, run_id="run-2", owner_id="owner-2"), FakeMarketData()
    ).start(now_us=104, instruments=(instrument,), subscriptions=specs)
    assert unrelated.run_id == "run-2"
    with connect_v2(database) as connection:
        runs = dict(connection.execute("SELECT run_id, status FROM runs ORDER BY run_id"))
        fatal_incident = connection.execute(
            "SELECT code, resolved_at_us FROM incidents WHERE run_id='run-1' AND severity='fatal'"
        ).fetchone()
    assert runs == {"run-1": "fatal", "run-2": "running"}
    assert tuple(fatal_incident) == ("TEST_FATAL", None)


def test_recoverable_fatal_generation_restart_is_explicit_audited_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal("POST_ADMISSION_PRESERVATION_FAILED", 101)
    with pytest.raises(LocalWriterLockError, match="writer lock is held"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash=market_data_input_hash((instrument,), specs),
            fatal_code="POST_ADMISSION_PRESERVATION_FAILED",
            operator="operator@example.invalid",
            reason="must not override a live local process",
            authorized_at_us=102,
        )
    failed.abandon_unclean()

    with pytest.raises(LocalWriterLockError, match="market-data input is incompatible"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash="b" * 64,
            fatal_code="POST_ADMISSION_PRESERVATION_FAILED",
            operator="operator@example.invalid",
            reason="must not change the frozen input",
            authorized_at_us=102,
        )
    recover_fatal_generation(
        database=database,
        run_id="run-1",
        generation=1,
        mode="prospective_record",
        config_hash="a" * 64,
        input_hash=market_data_input_hash((instrument,), specs),
        fatal_code="POST_ADMISSION_PRESERVATION_FAILED",
        operator="operator@example.invalid",
        reason="verified durable evidence and restored maintenance dependency",
        authorized_at_us=102,
    )
    restarted = Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
        now_us=103,
        instruments=(instrument,),
        subscriptions=specs,
    )

    assert restarted.recorder_generation == 2
    with connect_v2(database) as connection:
        failed_generation = connection.execute(
            "SELECT termination_code, fatal_recovery_authorized_at_us, "
            "fatal_recovery_operator, recovered_fatal_code FROM recorder_generations "
            "WHERE run_id='run-1' AND generation=1"
        ).fetchone()
        incidents = tuple(
            connection.execute(
                "SELECT severity, code, resolved_at_us FROM incidents WHERE run_id='run-1' "
                "ORDER BY opened_at_us, code"
            )
        )
    assert tuple(failed_generation) == (
        "POST_ADMISSION_PRESERVATION_FAILED",
        102,
        "operator@example.invalid",
        "POST_ADMISSION_PRESERVATION_FAILED",
    )
    assert [tuple(row) for row in incidents] == [
        ("fatal", "POST_ADMISSION_PRESERVATION_FAILED", None),
        ("info", "FATAL_GENERATION_RECOVERY_AUTHORIZED", None),
    ]


def test_wal_cap_fatal_generation_recovery_is_explicit_and_audited(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal("WAL_CAP_FATAL", 101)
    failed.abandon_unclean()

    recover_fatal_generation(
        database=database,
        run_id="run-1",
        generation=1,
        mode="prospective_record",
        config_hash="a" * 64,
        input_hash=market_data_input_hash((instrument,), specs),
        fatal_code="WAL_CAP_FATAL",
        operator="operator@example.invalid",
        reason="passive WAL control installed and integrity verified",
        authorized_at_us=102,
    )
    restarted = Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
        now_us=103,
        instruments=(instrument,),
        subscriptions=specs,
    )

    assert restarted.recorder_generation == 2
    with connect_v2(database) as connection:
        generation = connection.execute(
            "SELECT termination_code, fatal_recovery_authorized_at_us, "
            "recovered_fatal_code FROM recorder_generations "
            "WHERE run_id='run-1' AND generation=1"
        ).fetchone()
        fatal_incident = connection.execute(
            "SELECT code, resolved_at_us FROM incidents "
            "WHERE run_id='run-1' AND code='WAL_CAP_FATAL'"
        ).fetchone()
    assert tuple(generation) == ("WAL_CAP_FATAL", 102, "WAL_CAP_FATAL")
    assert tuple(fatal_incident) == ("WAL_CAP_FATAL", None)


def test_wal_cap_fatal_recovery_rejects_wal_still_at_hard_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal("WAL_CAP_FATAL", 101)
    failed.abandon_unclean()
    monkeypatch.setattr(
        lifecycle_module,
        "RetentionPolicy",
        lambda: types.SimpleNamespace(database_cap_bytes=2**63, wal_cap_bytes=0),
    )

    with pytest.raises(LocalWriterLockError, match="WAL hard cap"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash=market_data_input_hash((instrument,), specs),
            fatal_code="WAL_CAP_FATAL",
            operator="operator@example.invalid",
            reason="must remain closed while WAL is capped",
            authorized_at_us=102,
        )
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "fatal"


def test_offline_payload_drain_uses_one_lock_fixed_time_and_two_zero_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    calls: list[int] = []
    counts = iter((2_000, 0, 0))

    class FakePolicy:
        maintenance_batch_rows = 2_000

    class FakeManager:
        policy = FakePolicy()

        def __init__(self, _database: Path) -> None:
            pass

        def compact_payloads_only(
            self,
            *,
            now_us: int,
            precondition: object,
        ) -> int:
            cast(Any, precondition)(None)
            calls.append(now_us)
            return next(counts)

    monkeypatch.setattr(lifecycle_module, "RetentionManager", FakeManager)
    result = drain_callback_payloads(
        database=database,
        now_us=123,
        max_passes=10,
        max_wall_seconds=10,
        monotonic=lambda: 0.0,
    )

    assert calls == [123, 123, 123]
    assert result.passes == 3
    assert result.payloads_compacted == 2_000
    assert result.consecutive_zero_passes == 2
    assert result.elapsed_ms == 0
    with LocalWriterLock.for_database(database):
        pass


def test_offline_payload_drain_pass_and_wall_limits_are_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)

    class FakePolicy:
        maintenance_batch_rows = 2_000

    class FakeManager:
        policy = FakePolicy()

        def __init__(self, _database: Path) -> None:
            pass

        def compact_payloads_only(
            self,
            *,
            now_us: int,
            precondition: object,
        ) -> int:
            cast(Any, precondition)(None)
            return 1

    monkeypatch.setattr(lifecycle_module, "RetentionManager", FakeManager)
    with pytest.raises(PayloadDrainIncompleteError, match="2 passes and 2 committed"):
        drain_callback_payloads(
            database=database,
            now_us=123,
            max_passes=2,
            max_wall_seconds=10,
            monotonic=lambda: 0.0,
        )

    clock = iter((0.0, 0.0, 2.0))
    with pytest.raises(PayloadDrainIncompleteError, match="1 passes and 1 committed"):
        drain_callback_payloads(
            database=database,
            now_us=123,
            max_passes=10,
            max_wall_seconds=1,
            monotonic=lambda: next(clock),
        )


def test_fatal_generation_recovery_respects_storage_hard_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal("POST_ADMISSION_PRESERVATION_FAILED", 101)
    failed.abandon_unclean()
    monkeypatch.setattr(
        lifecycle_module,
        "RetentionPolicy",
        lambda: RetentionPolicy(database_cap_bytes=1, wal_cap_bytes=1),
    )

    with pytest.raises(LocalWriterLockError, match="database hard cap"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash=market_data_input_hash((instrument,), specs),
            fatal_code="POST_ADMISSION_PRESERVATION_FAILED",
            operator="operator@example.invalid",
            reason="capacity must remain fail closed",
            authorized_at_us=102,
        )
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
            == "fatal"
        )


def test_callback_identity_corruption_is_fatal_and_never_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, 101, {"event_at_us": 101, "bid": 100.0}),
    )

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise CallbackIdentityCollision("durable callback payload hash mismatch")

    monkeypatch.setattr(recorder.inbox, "project_batch", fail_projection)
    with pytest.raises(RecorderFatalError, match="callback provenance corruption"):
        recorder.drain(now_us=102)
    recorder.abandon_unclean()

    with pytest.raises(LocalWriterLockError, match="not recoverable"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash=market_data_input_hash((instrument,), specs),
            fatal_code="CALLBACK_PROVENANCE_CORRUPTION",
            operator="operator@example.invalid",
            reason="semantic evidence corruption must stay closed",
            authorized_at_us=103,
        )
    with connect_v2(database) as connection:
        generation = connection.execute(
            "SELECT termination_code FROM recorder_generations "
            "WHERE run_id='run-1' AND generation=1"
        ).fetchone()
        incident = connection.execute(
            "SELECT code FROM incidents WHERE run_id='run-1' AND severity='fatal'"
        ).fetchone()
    assert generation["termination_code"] == "CALLBACK_PROVENANCE_CORRUPTION"
    assert incident["code"] == "CALLBACK_PROVENANCE_CORRUPTION"


@pytest.mark.parametrize("fatal_code", ("INBOX_FULL", "RETENTION_INVARIANT_FAILED"))
def test_hard_or_ambiguous_fatal_generation_cannot_be_recovered(
    tmp_path: Path,
    fatal_code: str,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal(fatal_code, 101)
    failed.abandon_unclean()

    with pytest.raises(LocalWriterLockError, match="not recoverable"):
        recover_fatal_generation(
            database=database,
            run_id="run-1",
            generation=1,
            mode="prospective_record",
            config_hash="a" * 64,
            input_hash=market_data_input_hash((instrument,), specs),
            fatal_code=fatal_code,
            operator="operator@example.invalid",
            reason="must remain blocked",
            authorized_at_us=102,
        )
    with pytest.raises(RecorderFatalError, match="operator action"):
        Recorder(_config(database, owner_id="owner-2"), FakeMarketData()).start(
            now_us=103,
            instruments=(instrument,),
            subscriptions=specs,
        )


def test_fatal_generation_recovery_cli_is_machine_readable(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    config_path = tmp_path / "recorder.json"
    inputs_path = tmp_path / "market-data.json"
    initialize_database(database)
    config_path.write_text(
        json.dumps(_config(database).model_dump(mode="json")),
        encoding="utf-8",
    )
    instrument, specs = _specs()
    inputs_path.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        key: value
                        for key, value in instrument.__dict__.items()
                        if key
                        in {
                            "instrument_id",
                            "ibkr_con_id",
                            "kind",
                            "symbol",
                            "exchange",
                            "currency",
                        }
                    }
                ],
                "subscriptions": [
                    {key: value for key, value in item.__dict__.items() if key != "snapshot"}
                    for item in specs
                ],
            }
        ),
        encoding="utf-8",
    )
    failed = Recorder(_config(database), FakeMarketData())
    failed.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    failed._fatal("POST_ADMISSION_PRESERVATION_FAILED", 101)
    failed.abandon_unclean()

    result = CliRunner().invoke(
        app,
        [
            "recorder",
            "recover-fatal-generation",
            "--config",
            str(config_path),
            "--inputs",
            str(inputs_path),
            "--generation",
            "1",
            "--fatal-code",
            "POST_ADMISSION_PRESERVATION_FAILED",
            "--operator",
            "on-call",
            "--reason",
            "maintenance dependency restored",
            "--authorized-at-us",
            "102",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "authorized_at_us": 102,
        "fatal_code": "POST_ADMISSION_PRESERVATION_FAILED",
        "generation": 1,
        "run_id": "run-1",
        "status": "ok",
    }


@pytest.mark.parametrize(
    ("failure_kind", "cleanup_failure", "expected_reason"),
    (
        ("inbox_full", None, "INBOX_FULL"),
        ("ordering", "cancel", "CALLBACK_ORDERING_LOSS"),
        ("database", "disconnect", "CALLBACK_ADMISSION_FAILED"),
    ),
)
def test_recorder_admission_fatal_is_terminal_and_cleans_private_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    cleanup_failure: str | None,
    expected_reason: str,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class CleanupFailureMarketData(FakeMarketData):
        def cancel(self, request_id: int) -> None:
            super().cancel(request_id)
            if cleanup_failure == "cancel":
                raise RuntimeError("cancel cleanup failed")

        def disconnect(self) -> None:
            super().disconnect()
            if cleanup_failure == "disconnect":
                raise RuntimeError("disconnect cleanup failed")

    adapter = CleanupFailureMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    if failure_kind == "inbox_full":
        recorder.inbox = CallbackInbox(database, max_nonterminal_rows=1)
        recorder.receive(
            state.fences[0],
            MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 1.0}),
        )
        failing = MarketDataCallback("quote", 102, None, {"event_at_us": 102, "bid": 2.0})
    elif failure_kind == "ordering":
        recorder.receive(
            state.fences[0],
            MarketDataCallback("quote", 102, None, {"event_at_us": 102, "bid": 2.0}),
        )
        failing = MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 1.0})
    else:
        real_admit = recorder.inbox.admit
        failed_once = False

        def fail_database_admission(
            _fence: CallbackFence,
            _callback: MarketDataCallback,
            *,
            authority: WriterAuthority | None = None,
        ) -> AdmissionResult:
            nonlocal failed_once
            assert authority is not None
            if not failed_once:
                failed_once = True
                raise InboxAdmissionError("callback durable admission failed: disk full")
            return real_admit(_fence, _callback, authority=authority)

        monkeypatch.setattr(recorder.inbox, "admit", fail_database_admission)
        failing = MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 1.0})

    with pytest.raises(InboxAdmissionError):
        recorder.receive(state.fences[0], failing)

    assert adapter.connected is False
    assert set(adapter.cancelled) == {3, 4}
    with connect_v2(database, verify_schema=False) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        subscriptions = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        generation = connection.execute(
            "SELECT ended_at_us, clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='run-1' AND generation=1"
        ).fetchone()
        status = connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
    assert tuple(runtime) == ("fatal", expected_reason, "disconnected")
    assert subscriptions == {3: "disconnected", 4: "disconnected"}
    assert tuple(generation) == (failing.received_at_us, 0, expected_reason)
    assert status == "fatal"

    disconnects = adapter.disconnect_calls
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.receive(state.fences[0], failing)
    assert adapter.disconnect_calls > disconnects
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        ) == ("fatal", expected_reason)


def test_stale_receive_cleans_only_private_adapter_and_preserves_replacement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    _force_recorder_takeover(database)

    with pytest.raises(AuthoritativeLeaseLost):
        recorder.receive(
            state.fences[0],
            MarketDataCallback("quote", 102, None, {"event_at_us": 102, "bid": 1.0}),
        )

    assert adapter.connected is False
    assert set(adapter.cancelled) == {3, 4}
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        generation = connection.execute(
            "SELECT ended_at_us, termination_code FROM recorder_generations "
            "WHERE run_id='run-1' AND generation=2"
        ).fetchone()
    assert tuple(replacement) == (2, "running", None, "connected")
    assert tuple(generation) == (None, None)


def test_pre_gap_callback_cannot_clear_startup_storage_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stocker_runtime.storage import RetentionManager, RetentionPolicy

    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.receive(
        state.fences[1],
        MarketDataCallback("bar", 101, None, {"event_at_us": 101, "close": 10.0}),
    )
    degraded = RetentionResult(
        cap_state=StorageCapState.DEGRADED,
        database_bytes=95,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=False,
        required_action="PAUSE_OPTIONAL_FEEDS",
        checkpoint_attempted=False,
        incremental_vacuum_attempted=False,
    )

    class PauseRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert now_us == 102
            assert callable(precondition)
            return degraded

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", PauseRetention)
    recorder.maintain(now_us=102)
    assert recorder.drain(now_us=103) == 1

    with connect_v2(database) as connection:
        gap = connection.execute(
            "SELECT ended_at_us, resolved_at_us FROM gaps "
            "WHERE reason='STORAGE_DEGRADED_OPTIONAL_PAUSED'"
        ).fetchone()
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        optional = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=4"
        ).fetchone()[0]
    assert tuple(gap) == (None, None)
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS")
    assert optional == "paused"

    retained = RetentionManager(
        database,
        RetentionPolicy(
            callback_payload_us=1,
            receipt_us=1_000,
            tombstone_us=1_000,
            resolved_diagnostic_us=1,
        ),
    ).run(now_us=105, measured_database_bytes=1, measured_wal_bytes=0)
    assert retained.payloads_compacted == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT count(*) FROM gaps WHERE reason='STORAGE_DEGRADED_OPTIONAL_PAUSED'"
            ).fetchone()[0]
            == 1
        )


def test_acknowledgement_resolves_only_causal_allowlisted_gap_for_exact_subscription(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, 90, {"event_at_us": 90, "bid": 9.0}),
    )
    with connect_v2(database) as connection:
        required = str(
            connection.execute(
                "SELECT subscription_id FROM subscriptions WHERE request_id=3"
            ).fetchone()[0]
        )
        optional = str(
            connection.execute(
                "SELECT subscription_id FROM subscriptions WHERE request_id=4"
            ).fetchone()[0]
        )
        for reason, started_at_us in (
            ("STREAM_STALE", 102),
            ("RECONNECT_UNCERTAINTY", 103),
            ("STREAM_STALE", 200),
            ("STORAGE_DEGRADED_OPTIONAL_PAUSED", 102),
            ("IBKR_FARM_2103_DEGRADED", 102),
            ("IBKR_STATUS_420_PACING", 102),
            ("IBKR_DISCONNECT", 102),
            ("UNCLEAN_RECORDER_RESTART", 102),
            ("IBKR_SUBSCRIBE_FAILED", 102),
        ):
            Recorder._open_gap_for_run(
                connection,
                "run-1",
                required,
                started_at_us,
                reason,
                True,
            )
        Recorder._open_gap_for_run(connection, "run-1", optional, 102, "STREAM_STALE", False)

    assert recorder.drain(now_us=104) == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM gaps WHERE resolved_at_us IS NOT NULL"
            ).fetchone()[0]
            == 0
        )

    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 2, 'future-owner', 120)"
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "continuity_required, optional, requirements_hash, opened_at_us) "
            "SELECT 'future-subscription', run_id, 2, 99, instrument_id, feed_kind, request_id, "
            "'active', continuity_required, optional, requirements_hash, 120 "
            "FROM subscriptions WHERE subscription_id=?",
            (required,),
        )
        Recorder._open_gap_for_run(
            connection, "run-1", "future-subscription", 102, "STREAM_STALE", True
        )

    callback = MarketDataCallback("quote", 150, 140, {"event_at_us": 140, "bid": 10.0})
    first = recorder.receive(state.fences[0], callback)
    assert recorder.drain(now_us=153) == 1
    retry = recorder.receive(state.fences[0], callback)
    assert retry == replace(first, inserted=False)
    assert recorder.drain(now_us=154) == 0

    with connect_v2(database) as connection:
        rows = tuple(
            connection.execute(
                "SELECT subscription_id, reason, started_at_us, ended_at_us, resolved_at_us "
                "FROM gaps ORDER BY subscription_id, reason, started_at_us"
            )
        )
    resolved = {(str(row[0]), str(row[1]), int(row[2])): (row[3], row[4]) for row in rows}
    assert resolved[(required, "STREAM_STALE", 102)] == (150, 153)
    assert resolved[(required, "RECONNECT_UNCERTAINTY", 103)] == (150, 153)
    assert resolved[(required, "IBKR_DISCONNECT", 102)] == (150, 153)
    assert resolved[(required, "IBKR_SUBSCRIBE_FAILED", 102)] == (150, 153)
    assert resolved[(required, "STREAM_STALE", 200)] == (None, None)
    assert resolved[(optional, "STREAM_STALE", 102)] == (None, None)
    assert resolved[("future-subscription", "STREAM_STALE", 102)] == (None, None)
    for reason in (
        "STORAGE_DEGRADED_OPTIONAL_PAUSED",
        "IBKR_FARM_2103_DEGRADED",
        "IBKR_STATUS_420_PACING",
        "UNCLEAN_RECORDER_RESTART",
    ):
        assert resolved[(required, reason, 102)] == (None, None)


def test_callback_recovery_does_not_cross_snapshot_and_stream_cadence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "callback-cadence.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    stream_subscription_id = "prior-stream"
    snapshot_subscription_id = "current-snapshot"
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES ('instrument-1', ?, 123, 'option', 'AAPL', "
            "'SMART', 'USD')",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us, closed_at_us, snapshot) VALUES "
            "(?, 'run-1', 1, 1, 'instrument-1', 'quotes', 3, 'closed', ?, 1, 120, 0)",
            (stream_subscription_id, "c" * 64),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us, snapshot) VALUES "
            "(?, 'run-1', 1, 2, 'instrument-1', 'quotes', 4, 'active', ?, 130, 1)",
            (snapshot_subscription_id, "d" * 64),
        )
        for subscription_id, cadence in (
            (stream_subscription_id, "stream"),
            (snapshot_subscription_id, "snapshot"),
        ):
            Recorder._open_gap_for_run(
                connection,
                "run-1",
                subscription_id,
                140,
                "IBKR_SUBSCRIBE_FAILED",
                True,
            )
            incident_id = transport_incident_id(
                "run-1",
                "IBKR_SUBSCRIBE_FAILED",
                instrument_id="instrument-1",
                feed_kind="quotes",
                snapshot=cadence == "snapshot",
            )
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "subscription_id, opened_at_us, details_json) VALUES (?, 'run-1', "
                "'market_data', 'degraded', 'IBKR_SUBSCRIBE_FAILED', ?, 140, '{}')",
                (incident_id, subscription_id),
            )

    fence = CallbackFence("run-1", 1, 2, 4, snapshot_subscription_id)
    inbox = CallbackInbox(database)
    admitted = inbox.admit(
        fence,
        MarketDataCallback("quote", 150, 150, {"event_at_us": 150, "bid": 1.0}),
    )
    leased = inbox.lease_pending(
        "owner-1", now_us=151, lease_us=10, limit=1, authority=_authority()
    )[0]
    event = inbox.project(leased, authority=_authority())
    inbox.acknowledge(leased, event.event_id, acknowledged_at_us=152, authority=_authority())

    with connect_v2(database) as connection:
        gaps = dict(
            connection.execute(
                "SELECT subscription_id, resolved_at_us FROM gaps "
                "WHERE reason='IBKR_SUBSCRIBE_FAILED'"
            )
        )
        incidents = dict(
            connection.execute(
                "SELECT subscription_id, resolved_at_us FROM incidents "
                "WHERE code='IBKR_SUBSCRIBE_FAILED'"
            )
        )
        callback = connection.execute(
            "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()[0]
    assert callback == "acknowledged"
    assert gaps == {stream_subscription_id: None, snapshot_subscription_id: 152}
    assert incidents == {stream_subscription_id: None, snapshot_subscription_id: 152}


@pytest.mark.parametrize("with_gap", (False, True))
def test_callback_receipt_time_advances_drain_causal_clock(tmp_path: Path, with_gap: bool) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 150, 140, {"event_at_us": 140, "bid": 10.0}),
    )
    if with_gap:
        with connect_v2(database) as connection:
            Recorder._open_gap_for_run(
                connection,
                "run-1",
                str(state.fences[0].subscription_id),
                102,
                "STREAM_STALE",
                True,
            )

    assert recorder.drain(now_us=104) == 1

    assert adapter.connected is True
    assert adapter.cancelled == []
    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json, normalized_event_id, acknowledged_at_us "
            "FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        event_count = connection.execute(
            "SELECT count(*) FROM market_events WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()[0]
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state, process_heartbeat_at_us FROM runtime_state"
        ).fetchone()
        unresolved_stale = connection.execute(
            "SELECT count(*) FROM gaps WHERE reason='STREAM_STALE' AND resolved_at_us IS NULL"
        ).fetchone()[0]
        receipt = connection.execute(
            "SELECT created_at_us, last_received_at_us FROM callback_receipts"
        ).fetchone()
    assert callback[0] == "acknowledged"
    assert callback[1] == '{"bid":10.0,"event_at_us":140}'
    assert callback[2] is not None
    assert callback[3] == 150
    assert event_count == 1
    assert tuple(runtime) == ("running", None, "connected", 150)
    assert tuple(receipt) == (150, 150)
    assert unresolved_stale == 0
    assert recorder.inbox.nonterminal_count() == 0


def test_direct_terminal_callback_advances_receipt_and_drain_causal_clock(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    stale_fence = replace(
        state.fences[0],
        connection_generation=state.fences[0].connection_generation - 1,
    )

    admitted = recorder.receive(
        stale_fence,
        MarketDataCallback("quote", 143, 140, {"event_at_us": 140, "bid": 10.0}),
    )
    assert recorder.drain(now_us=142) == 0

    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, received_at_us, failure_code FROM callback_inbox "
            "WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        receipt = connection.execute(
            "SELECT created_at_us, last_received_at_us FROM callback_receipts"
        ).fetchone()
        heartbeat = connection.execute(
            "SELECT process_heartbeat_at_us FROM runtime_state"
        ).fetchone()[0]
    assert tuple(callback) == ("failed", 143, "STALE_REQUEST_GENERATION")
    assert tuple(receipt) == (143, 143)
    assert heartbeat == 143


def test_callback_evidence_equal_to_acknowledgement_time_can_resolve_gap(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        Recorder._open_gap_for_run(
            connection,
            "run-1",
            str(state.fences[0].subscription_id),
            150,
            "STREAM_STALE",
            True,
        )
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 150, 140, {"event_at_us": 140, "bid": 10.0}),
    )

    assert recorder.drain(now_us=150) == 1

    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, acknowledged_at_us FROM callback_inbox"
        ).fetchone()
        gap = connection.execute(
            "SELECT ended_at_us, resolved_at_us FROM gaps WHERE reason='STREAM_STALE'"
        ).fetchone()
    assert tuple(callback) == ("acknowledged", 150)
    assert tuple(gap) == (150, 150)


def test_canonical_projection_failure_preserves_raw_callback_and_degrades_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 150, None, {"event_at_us": 150, "bid": 1.0}),
    )
    real_project = recorder.inbox.project
    monkeypatch.setattr(
        recorder.inbox,
        "project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("injected projection failure")
        ),
    )
    assert recorder.drain(now_us=102) == 0
    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        status = connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        incident = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_CANONICAL_PROJECTION_FAILED'"
        ).fetchone()
    assert callback[0] == "leased"
    assert callback[1] is not None
    assert status == "running"
    assert tuple(runtime) == (
        "degraded",
        "COMPONENT_CANONICAL_PROJECTION_FAILED",
        "connected",
    )
    assert tuple(incident) == (150, None)

    monkeypatch.setattr(recorder.inbox, "project", real_project)
    recovered_callback = recorder.receive(
        state.fences[0],
        MarketDataCallback(
            "quote",
            30_000_150,
            None,
            {"event_at_us": 30_000_150, "bid": 2.0},
        ),
    )
    assert recorder.drain(now_us=30_000_151) == 2
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
                (recovered_callback.source_sequence,),
            ).fetchone()[0]
            == "acknowledged"
        )
        recovered_runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        resolved_at_us = connection.execute(
            "SELECT resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_CANONICAL_PROJECTION_FAILED'"
        ).fetchone()[0]
    assert tuple(recovered_runtime) == ("running", None, "connected")
    assert resolved_at_us == 30_000_151


def test_optional_downstream_components_fail_independently_and_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "downstream-boundaries.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert recorder._idea_runner is not None
    failing = True
    successful_calls: list[str] = []

    def component(name: str) -> None:
        if failing:
            raise RuntimeError(f"injected {name} failure")
        successful_calls.append(name)

    from stocker_runtime.ingestion import snapshot_projection

    monkeypatch.setattr(
        snapshot_projection,
        "project_option_snapshot_captures",
        lambda *_args, **_kwargs: component("option_projection"),
    )
    monkeypatch.setattr(
        recorder._idea_runner,
        "run_once",
        lambda *, now_us: component("idea_runner"),
    )
    monkeypatch.setattr(
        recorder,
        "_reconcile_dynamic_market_data",
        lambda *, now_us: component("option_discovery"),
    )
    recorder._shadow_engine = types.SimpleNamespace(
        run_once=lambda *, now_us: component("shadow_evaluation")
    )

    first = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 150, None, {"event_at_us": 150, "bid": 1.0}),
    )
    assert recorder.drain(now_us=150) == 1
    second = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 151, None, {"event_at_us": 151, "bid": 2.0}),
    )
    assert recorder.drain(now_us=151) == 1
    with connect_v2(database) as connection:
        callbacks = dict(
            connection.execute(
                "SELECT source_sequence, lifecycle FROM callback_inbox "
                "WHERE source_sequence IN (?, ?)",
                (first.source_sequence, second.source_sequence),
            )
        )
        incidents = {
            str(row[0])
            for row in connection.execute(
                "SELECT code FROM incidents WHERE scope='component' AND resolved_at_us IS NULL"
            )
        }
        runtime = connection.execute(
            "SELECT lifecycle, connection_state FROM runtime_state"
        ).fetchone()
    assert callbacks == {
        first.source_sequence: "acknowledged",
        second.source_sequence: "acknowledged",
    }
    assert incidents == {
        "COMPONENT_OPTION_PROJECTION_FAILED",
        "COMPONENT_IDEA_RUNNER_FAILED",
        "COMPONENT_OPTION_DISCOVERY_FAILED",
        "COMPONENT_SHADOW_EVALUATION_FAILED",
    }
    assert tuple(runtime) == ("degraded", "connected")
    assert adapter.connected is True

    failing = False
    assert recorder.drain(now_us=1_000_150) == 0
    with connect_v2(database) as connection:
        unresolved = connection.execute(
            "SELECT count(*) FROM incidents WHERE scope='component' AND resolved_at_us IS NULL"
        ).fetchone()[0]
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert unresolved == 0
    assert tuple(runtime) == ("running", None, "connected")
    assert set(successful_calls) == {
        "option_projection",
        "idea_runner",
        "option_discovery",
        "shadow_evaluation",
    }


def test_component_boundary_never_downgrades_recorder_fatal_error(tmp_path: Path) -> None:
    database = tmp_path / "component-fatal.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    def fatal_identity_invariant() -> None:
        raise RecorderFatalError("dynamic subscription identity changed")

    with pytest.raises(RecorderFatalError, match="identity changed"):
        recorder._run_component(
            "option_discovery",
            now_us=101,
            operation=fatal_identity_invariant,
        )
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM incidents WHERE scope='component'").fetchone()[
                0
            ]
            == 0
        )


def test_component_incident_backoff_starts_only_after_durable_publication(
    tmp_path: Path,
) -> None:
    database = tmp_path / "component-incident-contention.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    blocker = connect_v2(database)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        assert (
            recorder._component_failure(
                "backup_maintenance",
                now_us=101,
                error_name="OperationalError",
            )
            is False
        )
        assert recorder._component_failures["backup_maintenance"] == (1, 101)
    finally:
        blocker.rollback()
        blocker.close()

    assert (
        recorder._component_failure(
            "backup_maintenance",
            now_us=102,
            error_name="OperationalError",
        )
        is True
    )
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_BACKUP_MAINTENANCE_FAILED'"
        ).fetchone()
    assert tuple(incident) == (101, None)


def test_component_recovery_contention_remains_degraded_and_raw_admission_resumes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "component-recovery-contention.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert (
        recorder._component_failure(
            "backup_maintenance",
            now_us=101,
            error_name="OSError",
        )
        is True
    )
    blocker = connect_v2(database)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        assert (
            recorder._run_component(
                "backup_maintenance",
                now_us=1_000_101,
                operation=lambda: None,
            )
            is True
        )
        assert "backup_maintenance" in recorder._component_failures
    finally:
        blocker.rollback()
        blocker.close()

    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 2_000_101, None, {"event_at_us": 2_000_101, "bid": 1.0}),
    )
    assert recorder.drain(now_us=2_000_101) == 1
    assert recorder._component_recovered("backup_maintenance", now_us=2_000_102) is True
    with connect_v2(database) as connection:
        callback_lifecycle = connection.execute(
            "SELECT lifecycle FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()[0]
        run_status = connection.execute("SELECT status FROM runs").fetchone()[0]
        incident = connection.execute(
            "SELECT details_json, resolved_at_us FROM incidents "
            "WHERE code='COMPONENT_BACKUP_MAINTENANCE_FAILED'"
        ).fetchone()
    assert callback_lifecycle == "acknowledged"
    assert run_status == "running"
    assert json.loads(incident[0])["error"] == "OSError"
    assert incident[1] == 2_000_102


def test_canonical_observation_recovery_retries_after_incident_write_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "canonical-observation-recovery.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"bid": 1.0}),
    )
    assert recorder.drain(now_us=101) == 0
    assert "canonical_callback" in recorder._component_observations

    real_recovered = recorder._component_recovered

    def recover_while_contended(component: str, *, now_us: int) -> bool:
        blocker = connect_v2(database)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            return real_recovered(component, now_us=now_us)
        finally:
            blocker.rollback()
            blocker.close()

    monkeypatch.setattr(recorder, "_component_recovered", recover_while_contended)
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 1_000_101, None, {"event_at_us": 1_000_101, "bid": 2.0}),
    )
    assert recorder.drain(now_us=1_000_101) == 1
    assert "canonical_callback" in recorder._component_observations
    with connect_v2(database) as connection:
        unresolved = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='COMPONENT_CANONICAL_CALLBACK_FAILED'"
        ).fetchone()[0]
    assert unresolved is None

    monkeypatch.setattr(recorder, "_component_recovered", real_recovered)
    recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 2_000_101, None, {"event_at_us": 2_000_101, "bid": 3.0}),
    )
    assert recorder.drain(now_us=2_000_101) == 1
    assert "canonical_callback" not in recorder._component_observations
    with connect_v2(database) as connection:
        resolved = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='COMPONENT_CANONICAL_CALLBACK_FAILED'"
        ).fetchone()[0]
    assert resolved == 2_000_101


def test_component_recovery_preserves_prior_episode_history(tmp_path: Path) -> None:
    database = tmp_path / "component-episodes.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    recorder._run_component(
        "backup_maintenance",
        now_us=101,
        operation=lambda: (_ for _ in ()).throw(RuntimeError("first")),
    )
    recorder._run_component(
        "backup_maintenance",
        now_us=1_000_101,
        operation=lambda: None,
    )
    recorder._run_component(
        "backup_maintenance",
        now_us=2_000_101,
        operation=lambda: (_ for _ in ()).throw(RuntimeError("second")),
    )

    with connect_v2(database) as connection:
        episodes = tuple(
            connection.execute(
                "SELECT opened_at_us, resolved_at_us FROM incidents "
                "WHERE code='COMPONENT_BACKUP_MAINTENANCE_FAILED' ORDER BY opened_at_us"
            )
        )
    assert [tuple(row) for row in episodes] == [
        (101, 1_000_101),
        (2_000_101, None),
    ]


def test_component_recovery_is_scoped_to_current_recorder_generation(tmp_path: Path) -> None:
    database = tmp_path / "component-generation-scope.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert first._component_failure(
        "retention_maintenance",
        now_us=101,
        error_name="MaintenanceDeadlineExceeded",
    )
    first.stop(now_us=102)

    second = Recorder(_config(database, owner_id="owner-2"), FakeMarketData())
    second.start(now_us=200, instruments=(instrument,), subscriptions=specs)
    assert second._component_failure(
        "retention_maintenance",
        now_us=201,
        error_name="MaintenanceDeadlineExceeded",
    )
    assert second._component_recovered("retention_maintenance", now_us=1_000_201)

    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        incidents = tuple(
            connection.execute(
                "SELECT recorder_generation, opened_at_us, resolved_at_us FROM incidents "
                "WHERE code='COMPONENT_RETENTION_MAINTENANCE_FAILED' "
                "ORDER BY opened_at_us"
            )
        )
    assert tuple(runtime) == (2, "running", None, "connected")
    assert [tuple(row) for row in incidents] == [
        (1, 101, None),
        (2, 201, 1_000_201),
    ]


def test_legacy_component_incident_without_generation_cannot_poison_recovery(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-component-generation.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
            "details_json, recorder_generation) VALUES ('legacy-component', 'run-1', "
            "'component', 'degraded', 'COMPONENT_RETENTION_MAINTENANCE_FAILED', 50, '{}', NULL)"
        )

    assert recorder._component_failure(
        "backup_maintenance",
        now_us=101,
        error_name="OSError",
    )
    assert recorder._component_recovered("backup_maintenance", now_us=1_000_101)

    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        legacy = connection.execute(
            "SELECT recorder_generation, resolved_at_us FROM incidents "
            "WHERE incident_id='legacy-component'"
        ).fetchone()
    assert tuple(runtime) == ("running", None, "connected")
    assert tuple(legacy) == (None, None)


def test_current_generation_stays_degraded_until_every_component_recovers(
    tmp_path: Path,
) -> None:
    database = tmp_path / "current-component-generation.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert recorder._component_failure(
        "retention_maintenance",
        now_us=101,
        error_name="MaintenanceDeadlineExceeded",
    )
    assert recorder._component_failure(
        "backup_maintenance",
        now_us=102,
        error_name="OSError",
    )

    assert recorder._component_recovered("retention_maintenance", now_us=1_000_101)
    with connect_v2(database) as connection:
        after_first = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        unresolved = tuple(
            connection.execute(
                "SELECT code, recorder_generation FROM incidents "
                "WHERE resolved_at_us IS NULL ORDER BY code"
            )
        )
    assert tuple(after_first) == (
        "degraded",
        "COMPONENT_RETENTION_MAINTENANCE_FAILED",
        "connected",
    )
    assert [tuple(row) for row in unresolved] == [("COMPONENT_BACKUP_MAINTENANCE_FAILED", 1)]

    assert recorder._component_recovered("backup_maintenance", now_us=1_000_102)
    with connect_v2(database) as connection:
        after_last = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(after_last) == ("running", None, "connected")


def test_retention_backoff_still_fails_closed_at_storage_hard_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "retention-backoff-cap.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder._component_failures["retention_maintenance"] = (1, 2_000_000)

    class FatalCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def measure_cap_state(self) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.FATAL, 9_000_000_000, 0, "STORAGE_CAP_FATAL"

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        FatalCapMeasurement,
    )

    with pytest.raises(RecorderFatalError, match="STORAGE_CAP_FATAL"):
        recorder.maintain(now_us=101)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "fatal"


def test_retention_backoff_still_pauses_optional_feeds_at_degraded_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "retention-backoff-degraded.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder._component_failures["retention_maintenance"] = (1, 2_000_000)

    class DegradedCapMeasurement:
        def __init__(self, _database: Path) -> None:
            pass

        def measure_cap_state(self) -> tuple[StorageCapState, int, int, str | None]:
            return StorageCapState.DEGRADED, 7_700_000_000, 0, None

    monkeypatch.setattr(
        "stocker_runtime.ingestion.recorder.RetentionManager",
        DegradedCapMeasurement,
    )

    assert recorder.maintain(now_us=101) is StorageCapState.DEGRADED
    assert recorder.maintain(now_us=102) is StorageCapState.DEGRADED
    with connect_v2(database) as connection:
        optional_lifecycle = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=4"
        ).fetchone()[0]
        gap_count = connection.execute(
            "SELECT count(*) FROM gaps WHERE reason='STORAGE_DEGRADED_OPTIONAL_PAUSED' "
            "AND resolved_at_us IS NULL"
        ).fetchone()[0]
    assert optional_lifecycle == "paused"
    assert adapter.cancelled == [4]
    assert gap_count == 1


def test_production_base_snapshot_is_rejected_before_broker_connect(tmp_path: Path) -> None:
    database = tmp_path / "base-snapshot.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    snapshot = replace(specs[0], snapshot=True)
    adapter = FakeMarketData()

    with pytest.raises(RecorderFatalError, match="cannot configure base snapshot"):
        Recorder(_config(database), adapter).start(
            now_us=100,
            instruments=(instrument,),
            subscriptions=(snapshot,),
        )

    assert adapter.connected is False
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_projection_uses_durable_payload_and_rejects_altered_lease_token(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    inbox.admit(
        fence,
        MarketDataCallback("quote", 10, None, {"event_at_us": 10, "bid": 1.0}),
    )
    leased = inbox.lease_pending("worker", now_us=11, lease_us=10, limit=1, authority=_authority())[
        0
    ]
    with pytest.raises(InboxAdmissionError, match="lease token"):
        inbox.project(replace(leased, event_uid="0" * 64), authority=_authority())
    assert isinstance(leased.payload, dict)
    leased.payload["bid"] = 999.0
    projected = inbox.project(leased, authority=_authority())
    with connect_v2(database) as connection:
        bid = connection.execute(
            "SELECT bid_value FROM market_events WHERE event_id=?", (projected.event_id,)
        ).fetchone()[0]
    assert bid == 1.0


@pytest.mark.parametrize(
    ("failed_request", "runtime_lifecycle", "required_lifecycle", "optional_lifecycle"),
    ((3, "degraded", "disconnected", "active"), (4, "running", "active", "disconnected")),
)
def test_subscription_state_is_truthful_when_subscribe_fails(
    tmp_path: Path,
    failed_request: int,
    runtime_lifecycle: str,
    required_lifecycle: str,
    optional_lifecycle: str,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    Recorder(_config(database), FakeMarketData(fail_subscribe={failed_request})).start(
        now_us=100, instruments=(instrument,), subscriptions=specs
    )
    with connect_v2(database) as connection:
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
    assert runtime == runtime_lifecycle
    assert states == {3: required_lifecycle, 4: optional_lifecycle}


@pytest.mark.parametrize("failed_request", (3, 4, 5))
def test_startup_attempts_every_required_subscription_and_recovers_only_failure(
    tmp_path: Path,
    failed_request: int,
) -> None:
    database = tmp_path / f"startup-failure-{failed_request}.sqlite3"
    initialize_database(database)
    instrument, specs = _three_required_specs()
    adapter = FakeMarketData(fail_subscribe={failed_request})
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert adapter.subscribe_attempts == [3, 4, 5]
    assert adapter.active_request_ids == {3, 4, 5} - {failed_request}
    assert adapter.connected is True
    with connect_v2(database) as connection:
        lifecycles = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
    assert lifecycles[failed_request] == "disconnected"
    assert all(
        lifecycle == "active"
        for request_id, lifecycle in lifecycles.items()
        if request_id != failed_request
    )

    adapter.fail_subscribe.clear()
    assert recorder.recover_subscriptions(now_us=1_000_100) == 1
    assert adapter.retry_calls == [failed_request]
    assert adapter.active_request_ids == {3, 4, 5}
    fence = next(item for item in state.fences if item.request_id == failed_request)
    callback_kind = {3: "quote", 4: "trade", 5: "bar"}[failed_request]
    payloads = {
        3: {"event_at_us": 1_000_101, "bid": 1.0},
        4: {"event_at_us": 1_000_101, "last": 1.0},
        5: {
            "event_at_us": 1_000_101,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
    }
    adapter.emit(
        fence,
        MarketDataCallback(callback_kind, 1_000_101, None, payloads[failed_request]),
    )
    assert recorder.drain(now_us=1_000_102) == 1
    with connect_v2(database) as connection:
        lifecycles = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute(
            "SELECT lifecycle, connection_state FROM runtime_state"
        ).fetchone()
    assert lifecycles == {3: "active", 4: "active", 5: "active"}
    assert tuple(runtime) == ("running", "connected")


def test_request_pacing_uses_bounded_backoff_and_callback_evidence_for_recovery(
    tmp_path: Path,
) -> None:
    database = tmp_path / "request-pacing.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    recorder.market_data_status(MarketDataStatus("pacing", 420, 3, "paced", 101))
    recorder.market_data_status(MarketDataStatus("pacing", 420, 3, "paced again", 102))
    with connect_v2(database) as connection:
        retry = connection.execute(
            "SELECT lifecycle, retry_count, next_retry_at_us, permanent_failure "
            "FROM subscriptions WHERE request_id=3"
        ).fetchone()
        healthy = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=4"
        ).fetchone()[0]
    assert tuple(retry) == ("disconnected", 2, 10_000_102, 0)
    assert healthy == "active"
    assert recorder.recover_subscriptions(now_us=10_000_101) == 0
    assert recorder.recover_subscriptions(now_us=10_000_102) == 1
    assert adapter.retry_calls == [3]
    assert adapter.active_request_ids == {3, 4}

    recorder.market_data_status(MarketDataStatus("recovered", 1102, 3, "request accepted", 103))
    with connect_v2(database) as connection:
        before_callback = connection.execute(
            "SELECT lifecycle, last_error_code FROM subscriptions WHERE request_id=3"
        ).fetchone()
        unresolved = connection.execute(
            "SELECT count(*) FROM incidents WHERE subscription_id=? AND resolved_at_us IS NULL",
            (state.fences[0].subscription_id,),
        ).fetchone()[0]
    assert tuple(before_callback) == ("connecting", "IBKR_STATUS_420_PACING")
    assert unresolved >= 1

    adapter.emit(
        state.fences[0],
        MarketDataCallback("quote", 10_000_103, None, {"event_at_us": 10_000_103, "bid": 1.0}),
    )
    assert recorder.drain(now_us=10_000_104) == 1
    with connect_v2(database) as connection:
        recovered = connection.execute(
            "SELECT lifecycle, retry_count, next_retry_at_us, last_error_code "
            "FROM subscriptions WHERE request_id=3"
        ).fetchone()
        pacing_incidents = connection.execute(
            "SELECT count(*) FROM incidents WHERE subscription_id=? "
            "AND code='IBKR_STATUS_420_PACING' AND resolved_at_us IS NULL",
            (state.fences[0].subscription_id,),
        ).fetchone()[0]
    assert tuple(recovered) == ("active", 0, None, None)
    assert pacing_incidents == 0


@pytest.mark.parametrize(
    ("request_id", "expected_runtime"),
    ((3, "degraded"), (4, "running")),
)
def test_permanent_rejection_stays_visible_without_tight_loop(
    tmp_path: Path,
    request_id: int,
    expected_runtime: str,
) -> None:
    database = tmp_path / f"permanent-{request_id}.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    recorder.market_data_status(
        MarketDataStatus("request_rejected", 200, request_id, "contract rejected", 101)
    )
    with connect_v2(database) as connection:
        rejected = connection.execute(
            "SELECT lifecycle, next_retry_at_us, permanent_failure FROM subscriptions "
            "WHERE request_id=?",
            (request_id,),
        ).fetchone()
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
    assert tuple(rejected) == ("degraded", None, 1)
    assert runtime == expected_runtime
    assert recorder.recover_subscriptions(now_us=100_000_000) == 0
    assert adapter.retry_calls == []


def test_subscription_recovery_resolves_only_the_feed_with_callback_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "scoped-request-recovery.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(MarketDataStatus("pacing", 420, 3, "quotes paced", 101))
    recorder.market_data_status(
        MarketDataStatus("temporary_disconnect", 2110, 4, "bars disconnected", 101)
    )

    assert recorder.recover_subscriptions(now_us=1_000_101) == 1
    assert adapter.retry_calls == [4]
    optional_fence = next(item for item in state.fences if item.request_id == 4)
    adapter.emit(
        optional_fence,
        MarketDataCallback(
            "bar",
            1_000_102,
            None,
            {
                "event_at_us": 1_000_102,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            },
        ),
    )
    assert recorder.drain(now_us=1_000_103) == 1
    with connect_v2(database) as connection:
        states = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions ORDER BY request_id"
            )
        )
        open_codes = {
            str(row[0])
            for row in connection.execute("SELECT code FROM incidents WHERE resolved_at_us IS NULL")
        }
    assert states == {3: "disconnected", 4: "active"}
    assert "IBKR_STATUS_420_PACING" in open_codes
    assert "IBKR_STATUS_2110_TEMPORARY_DISCONNECT" not in open_codes


def test_malformed_callback_does_not_claim_subscription_recovery(tmp_path: Path) -> None:
    database = tmp_path / "malformed-recovery-evidence.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(MarketDataStatus("pacing", 420, 3, "paced", 101))
    assert recorder.recover_subscriptions(now_us=5_000_101) == 1

    adapter.emit(
        state.fences[0],
        MarketDataCallback(
            "quote",
            5_000_102,
            None,
            {"event_at_us": 5_000_102, "bid": "not-a-number"},
        ),
    )
    assert recorder.drain(now_us=5_000_103) == 0
    with connect_v2(database) as connection:
        subscription = connection.execute(
            "SELECT lifecycle, last_error_code FROM subscriptions WHERE request_id=3"
        ).fetchone()
        callback = connection.execute(
            "SELECT lifecycle, failure_code FROM callback_inbox "
            "ORDER BY source_sequence DESC LIMIT 1"
        ).fetchone()
        unresolved = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_STATUS_420_PACING' "
            "AND resolved_at_us IS NULL"
        ).fetchone()[0]
    assert tuple(subscription) == ("connecting", "IBKR_STATUS_420_PACING")
    assert tuple(callback) == ("failed", "MALFORMED_CALLBACK")
    assert unresolved == 1


def test_farm_recovery_cannot_override_permanent_request_rejection(tmp_path: Path) -> None:
    database = tmp_path / "farm-request-ordering.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(
        MarketDataStatus("request_rejected", 200, 3, "contract rejected", 101)
    )
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "quotes farm lost", 102, ("quotes",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "quotes farm restored", 103, ("quotes",))
    )

    with connect_v2(database) as connection:
        rejected = connection.execute(
            "SELECT lifecycle, last_error_code, permanent_failure FROM subscriptions "
            "WHERE request_id=3"
        ).fetchone()
        rejection_incident = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='IBKR_STATUS_200_REQUEST_REJECTED'"
        ).fetchone()[0]
    assert tuple(rejected) == ("degraded", "IBKR_STATUS_200_REQUEST_REJECTED", 1)
    assert rejection_incident is None


def test_partial_quote_callbacks_merge_into_latest_projection(tmp_path: Path) -> None:
    from stocker_runtime.storage import RetentionManager, RetentionPolicy

    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    fence = _seed_subscription(database)
    inbox = CallbackInbox(database)
    admitted = []
    for received_at_us, payload in (
        (10, {"event_at_us": 10, "bid": 100.0, "bid_size": 2.0}),
        (11, {"event_at_us": 11, "ask": 101.0, "ask_size": 3.0}),
    ):
        admitted.append(
            inbox.admit(fence, MarketDataCallback("quote", received_at_us, None, payload))
        )
    for leased in inbox.lease_pending(
        "worker", now_us=12, lease_us=10, limit=10, authority=_authority()
    ):
        event = inbox.project(leased, authority=_authority())
        inbox.acknowledge(leased, event.event_id, acknowledged_at_us=12, authority=_authority())
    with connect_v2(database) as connection:
        latest = connection.execute(
            "SELECT bid_value, ask_value, bid_size_value, ask_size_value, "
            "bid_source_event_id, ask_source_event_id, bid_size_source_event_id, "
            "ask_size_source_event_id FROM market_latest"
        ).fetchone()
    assert tuple(latest) == (
        100.0,
        101.0,
        2.0,
        3.0,
        admitted[0].event_uid,
        admitted[1].event_uid,
        admitted[0].event_uid,
        admitted[1].event_uid,
    )
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status='stopped', ended_at_us=20 WHERE run_id='run-1'")
    RetentionManager(database, RetentionPolicy(raw_market_event_us=1)).run(
        now_us=100, measured_database_bytes=1, measured_wal_bytes=0
    )
    with connect_v2(database) as connection:
        retained_sources = connection.execute(
            "SELECT count(*) FROM market_events WHERE event_id IN (?, ?)",
            (admitted[0].event_uid, admitted[1].event_uid),
        ).fetchone()[0]
    assert retained_sources == 2


def test_typed_optional_status_does_not_stop_required_feed(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(MarketDataStatus("pacing", 420, 4, "paced", 101))
    with connect_v2(database) as connection:
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
        optional = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=4"
        ).fetchone()[0]
        gap = connection.execute(
            "SELECT continuity_required FROM gaps WHERE reason LIKE 'IBKR_STATUS_420_%'"
        ).fetchone()[0]
    assert (runtime, optional, gap) == ("running", "disconnected", 0)


def test_replay_rejects_oversized_fixture_before_starting_run(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    fixture_path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    result = CliRunner().invoke(
        app, ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"]
    )
    assert result.exit_code == 1
    assert "8 MiB" in result.stdout
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_replay_rejects_more_than_50000_callbacks_before_starting_run(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    fixture_path.write_text(
        json.dumps(
            {
                "instruments": [],
                "subscriptions": [],
                "callbacks": [None] * 50_001,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app, ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"]
    )

    assert result.exit_code == 1
    assert "at most 50,000" in result.stdout
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_replay_rejects_oversized_callback_and_cleans_started_writer(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    instrument, subscriptions = _specs()
    fixture_path.write_text(
        json.dumps(
            {
                "instruments": [instrument.__dict__],
                "subscriptions": [item.__dict__ for item in subscriptions],
                "callbacks": [
                    {
                        "request_id": 3,
                        "callback_kind": "quote",
                        "received_at_us": 101,
                        "provider_at_us": None,
                        "payload": {"event_at_us": 101, "blob": "x" * 65_536},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app, ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"]
    )

    assert result.exit_code == 1
    assert "payload exceeds 64 KiB" in result.stdout
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_inbox").fetchone()[0] == 0
        assert tuple(
            connection.execute("SELECT lifecycle, connection_state FROM runtime_state").fetchone()
        ) == ("stopped", "disconnected")
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"


def test_receipts_accept_interleaved_global_sequences_without_skipping_same_run(
    tmp_path: Path,
) -> None:
    from stocker_runtime.storage import RetentionManager, RetentionPolicy

    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, git_commit, "
            "data_class, status, ended_at_us) VALUES "
            "('run-2', 'prospective_record', 'ibkr', 1, ?, 'deadbee', "
            "'prospective_protected', 'stopped', 9)",
            ("d" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us, "
            "ended_at_us, clean_stop, termination_code) VALUES "
            "('run-2', 1, 'owner-2', 1, 9, 1, 'CLEAN_STOP')"
        )
        for sequence, run_id in ((1, "run-1"), (2, "run-2"), (3, "run-1"), (4, "run-2")):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code) "
                "VALUES (?, ?, ?, 1, 1, 'quote', ?, '{}', ?, 'failed', 'fixture')",
                (sequence, f"event-{sequence}", run_id, sequence, f"{sequence:064x}"),
            )
    inbox = CallbackInbox(database)
    first = inbox.create_receipt("run-1", created_at_us=10, limit=10, authority=_authority())
    second = inbox.create_receipt("run-2", created_at_us=10, limit=10, authority=_authority())
    assert first is not None and (
        first.first_source_sequence,
        first.last_source_sequence,
        first.callback_count,
    ) == (1, 3, 2)
    assert second is not None and (
        second.first_source_sequence,
        second.last_source_sequence,
        second.callback_count,
    ) == (2, 4, 2)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status='stopped', ended_at_us=11 WHERE run_id='run-1'")
    manager = RetentionManager(
        database, RetentionPolicy(callback_payload_us=1, receipt_us=1_000, tombstone_us=1_000)
    )
    first_result = manager.run(now_us=12, measured_database_bytes=1, measured_wal_bytes=0)
    second_result = manager.run(now_us=12, measured_database_bytes=1, measured_wal_bytes=0)
    assert first_result.payloads_compacted + second_result.payloads_compacted == 4


def _force_recorder_takeover(database: Path) -> None:
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 2, 'replacement', 101)"
        )
        connection.execute(
            "UPDATE runtime_state SET recorder_generation=2, lifecycle='running', "
            "connection_state='connected', process_heartbeat_at_us=101 WHERE run_id='run-1'"
        )


def test_authority_takeover_waits_for_atomic_drain_chunk_and_preserves_replacement_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )
    entered = threading.Event()
    release = threading.Event()
    takeover_done = threading.Event()
    original_project = recorder.inbox.project
    original_receipts = recorder.inbox.create_pending_receipts

    def slow_project(*args: object, **kwargs: object) -> object:
        entered.set()
        assert release.wait(timeout=5)
        return original_project(*args, **kwargs)  # type: ignore[arg-type]

    def wait_for_takeover(*args: object, **kwargs: object) -> object:
        assert takeover_done.wait(timeout=5)
        return original_receipts(*args, **kwargs)  # type: ignore[arg-type]

    def take_over() -> None:
        _force_recorder_takeover(database)
        takeover_done.set()

    monkeypatch.setattr(recorder.inbox, "project", slow_project)
    monkeypatch.setattr(recorder.inbox, "create_pending_receipts", wait_for_takeover)
    disconnects = adapter.disconnect_calls
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(recorder.drain, now_us=102)
        assert entered.wait(timeout=5)
        takeover = executor.submit(take_over)
        release.set()
        takeover.result(timeout=5)
        with pytest.raises(AuthoritativeLeaseLost):
            future.result(timeout=5)

    assert adapter.disconnect_calls == disconnects
    assert adapter.cancelled == []
    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert callback[0] == "acknowledged" and callback[1] is not None
    assert tuple(replacement) == (2, "running", None, "connected")


def test_authority_takeover_during_slow_maintenance_cannot_publish_old_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        prior_measurements = tuple(
            connection.execute("SELECT database_bytes, wal_bytes FROM runtime_state").fetchone()
        )
    entered = threading.Event()
    release = threading.Event()

    class SlowRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            entered.set()
            assert release.wait(timeout=5)
            assert callable(precondition)
            with connect_v2(database) as connection:
                precondition(connection)
            raise AssertionError("lost authority must stop retention")

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", SlowRetention)
    disconnects = adapter.disconnect_calls
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(recorder.maintain, now_us=101)
        assert entered.wait(timeout=5)
        _force_recorder_takeover(database)
        release.set()
        with pytest.raises(AuthoritativeLeaseLost):
            future.result(timeout=5)

    assert adapter.disconnect_calls == disconnects
    assert adapter.cancelled == []
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state, "
            "database_bytes, wal_bytes FROM runtime_state"
        ).fetchone()
    assert tuple(replacement[:4]) == (2, "running", None, "connected")
    assert tuple(replacement[4:]) == prior_measurements


@pytest.mark.parametrize("failure_surface", ("connect", "subscribe"))
def test_external_failure_after_takeover_disconnects_only_stale_adapter(
    tmp_path: Path, failure_surface: str
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class TakeoverFailure(FakeMarketData):
        def connect(self) -> None:
            if failure_surface == "connect":
                self.connect_calls += 1
                _force_recorder_takeover(database)
                raise RuntimeError("connect failed after takeover")
            super().connect()

        def subscribe(self, fence: CallbackFence) -> None:
            if failure_surface == "subscribe":
                _force_recorder_takeover(database)
                raise RuntimeError("subscribe failed after takeover")
            super().subscribe(fence)

    adapter = TakeoverFailure()
    with pytest.raises(AuthoritativeLeaseLost):
        Recorder(_config(database), adapter).start(
            now_us=100, instruments=(instrument,), subscriptions=specs
        )
    assert adapter.disconnect_calls >= 1
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(replacement) == (2, "running", None, "connected")


def test_required_subscribe_failure_preserves_earlier_optional_success(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    reordered = (specs[1], specs[0])
    adapter = FakeMarketData(fail_subscribe={3})
    Recorder(_config(database), adapter).start(
        now_us=100, instruments=(instrument,), subscriptions=reordered
    )
    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
    assert states == {3: "disconnected", 4: "active"}
    assert runtime == "degraded"
    assert adapter.connected is True


def test_market_latest_resets_across_runs_and_tracks_each_field_source(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    first = Recorder(_config(database), FakeMarketData())
    first_state = first.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    bid = first.receive(
        first_state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 100.0}),
    )
    first.drain(now_us=102)
    first.abandon_unclean()
    second = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=15_000_000,
        ),
        FakeMarketData(),
    )
    second_state = second.start(now_us=15_000_103, instruments=(instrument,), subscriptions=specs)
    ask = second.receive(
        second_state.fences[0],
        MarketDataCallback("quote", 15_000_104, None, {"event_at_us": 15_000_104, "ask": 101.0}),
    )
    second.drain(now_us=15_000_105)
    with connect_v2(database) as connection:
        latest = connection.execute(
            "SELECT run_id, bid_value, bid_source_event_id, ask_value, ask_source_event_id "
            "FROM market_latest WHERE instrument_id='instrument-1' AND feed_kind='quotes'"
        ).fetchone()
    assert bid.event_uid != ask.event_uid
    assert tuple(latest) == ("run-2", None, None, 101.0, ask.event_uid)


def test_farm_warning_is_scoped_and_recovery_resolves_only_affected_feed(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    disconnects = adapter.disconnect_calls
    recorder.market_data_status(
        MarketDataStatus(
            "farm_degraded", 2103, None, "market data farm disconnected", 101, ("quotes",)
        )
    )
    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()
    assert states == {3: "degraded", 4: "active"}
    assert gap[0] is None
    assert adapter.disconnect_calls == disconnects
    assert adapter.connected is True
    recorder.market_data_status(
        MarketDataStatus(
            "farm_recovered", 2104, None, "market data farm restored", 102, ("quotes",)
        )
    )
    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
        resolved = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()[0]
    assert states == {3: "active", 4: "active"}
    assert (runtime, resolved) == ("running", 102)
    recorder.market_data_status(
        MarketDataStatus(
            "farm_degraded", 2105, None, "historical farm disconnected", 103, ("bars",)
        )
    )
    with connect_v2(database) as connection:
        optional_scope = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
    assert optional_scope == {3: "active", 4: "degraded"}
    assert adapter.disconnect_calls == disconnects
    assert adapter.connected is True


@pytest.mark.parametrize(
    ("recovered_at_us", "expected_resolution", "expected_lifecycle"),
    ((100, None, "degraded"), (101, 101, "running")),
)
def test_farm_recovery_cannot_resolve_incident_or_gap_before_they_open(
    tmp_path: Path,
    recovered_at_us: int,
    expected_resolution: int | None,
    expected_lifecycle: str,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "quotes lost", 101, ("quotes",))
    )

    recorder.market_data_status(
        MarketDataStatus(
            "farm_recovered",
            2104,
            None,
            "quotes restored",
            recovered_at_us,
            ("quotes",),
        )
    )

    with connect_v2(database) as connection:
        gap_resolution = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()[0]
        incident_resolution = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()[0]
        runtime = connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0]
        subscription = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=3"
        ).fetchone()[0]
    assert gap_resolution == expected_resolution
    assert incident_resolution == expected_resolution
    assert runtime == expected_lifecycle
    assert subscription == ("active" if expected_resolution is not None else "degraded")


def test_replay_malformed_callback_after_start_cleans_up_writer(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    instrument, subscriptions = _specs()
    fixture_path.write_text(
        json.dumps(
            {
                "instruments": [instrument.__dict__],
                "subscriptions": [item.__dict__ for item in subscriptions],
                "callbacks": [{"invalid": True}],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"]
    )
    assert result.exit_code == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
        assert connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0] == "stopped"


def test_replay_foreign_unexpired_lease_errors_bounded_and_stops_current_writer(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle, lease_owner, lease_expires_at_us) "
            "VALUES ('foreign-lease', 'run-1', 1, 2, 'quote', 2, '{}', ?, "
            "'leased', 'dead-worker', 100000000)",
            ("f" * 64,),
        )
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(
        json.dumps(_config(database, writer_lease_stale_us=15_000_000).model_dump(mode="json")),
        encoding="utf-8",
    )
    fixture_path.write_text(json.dumps(_replay_fixture_without_callbacks()), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "15000002"],
    )
    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["error"] == "ReplayBlockedError"
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
        assert connection.execute("SELECT lifecycle FROM runtime_state").fetchone()[0] == "stopped"


def test_replay_budget_uses_preexisting_backlog_not_fixture_count(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_generation(database)
    empty_payload_hash = hashlib.sha256(b"{}").hexdigest()
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, 'run-1', 1, 2, 'quote', ?, '{}', ?, "
            "'pending')",
            ((f"backlog-{index}", index + 2, empty_payload_hash) for index in range(300)),
        )
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(
        json.dumps(_config(database, writer_lease_stale_us=15_000_000).model_dump(mode="json")),
        encoding="utf-8",
    )
    fixture_path.write_text(json.dumps(_replay_fixture_without_callbacks()), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "15000002"],
    )
    assert result.exit_code == 0, result.stdout
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased')"
            ).fetchone()[0]
            == 0
        )


def test_replay_cleanup_does_not_mutate_replacement_after_authority_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "runtime.json"
    fixture_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(_config(database).model_dump(mode="json")), encoding="utf-8")
    instrument, subscriptions = _specs()
    fixture_path.write_text(
        json.dumps(
            {
                "instruments": [instrument.__dict__],
                "subscriptions": [item.__dict__ for item in subscriptions],
                "callbacks": [{"invalid": True}],
            }
        ),
        encoding="utf-8",
    )

    def lose_authority(*_args: object, **_kwargs: object) -> object:
        _force_recorder_takeover(database)
        raise ValueError("fixture lost authority")

    monkeypatch.setattr("stocker_runtime.cli._validated_replay_callback", lose_authority)
    result = CliRunner().invoke(
        app, ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "100"]
    )
    assert result.exit_code == 1
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        status = connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
    assert tuple(replacement) == (2, "running", None, "connected")
    assert status == "running"


@pytest.mark.parametrize("persistence_failure", ("takeover", "database"))
def test_subscription_failure_disconnects_when_failure_persistence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistence_failure: str
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_subscribe={3})
    recorder = Recorder(_config(database), adapter)

    def fail_persistence(*_args: object, **_kwargs: object) -> None:
        if persistence_failure == "takeover":
            _force_recorder_takeover(database)
            with connect_v2(database) as connection:
                connection.execute(
                    "UPDATE runtime_state SET reason='REPLACEMENT_OWNS_STATE' "
                    "WHERE run_id='run-1' AND recorder_generation=2"
                )
            raise AuthoritativeLeaseLost("replacement took authority")
        raise sqlite3.OperationalError("injected second persistence failure")

    monkeypatch.setattr(recorder, "_persist_subscription_failure", fail_persistence)
    with pytest.raises((AuthoritativeLeaseLost, sqlite3.OperationalError)):
        recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    assert adapter.connected is False
    assert adapter.disconnect_calls >= 1
    with connect_v2(database) as connection:
        scoped = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED' "
            "AND subscription_id IS NOT NULL"
        ).fetchone()[0]
        fabricated_global = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED' "
            "AND subscription_id IS NULL"
        ).fetchone()[0]
        runtime = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        gap_requests = {
            int(row[0])
            for row in connection.execute(
                "SELECT subscription.request_id FROM gaps gap JOIN subscriptions subscription "
                "ON subscription.subscription_id=gap.subscription_id "
                "WHERE gap.reason='IBKR_SUBSCRIBE_FAILED'"
            )
        }
    assert scoped == 0
    assert fabricated_global == 0
    assert gap_requests == set()
    if persistence_failure == "takeover":
        assert tuple(runtime) == (2, "running", "REPLACEMENT_OWNS_STATE", "connected")
    else:
        assert tuple(runtime) == (1, "connecting", None, "connecting")
        assert states == {3: "connecting", 4: "connecting"}


def test_optional_subscribe_failure_remains_isolated_and_connected(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_subscribe={4})
    Recorder(_config(database), adapter).start(
        now_us=100, instruments=(instrument,), subscriptions=specs
    )

    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        scoped = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED' "
            "AND subscription_id IS NOT NULL"
        ).fetchone()[0]
        global_incident = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='IBKR_SUBSCRIBE_FAILED' "
            "AND subscription_id IS NULL"
        ).fetchone()[0]
    assert states == {3: "active", 4: "disconnected"}
    assert tuple(runtime) == ("running", None, "connected")
    assert (scoped, global_incident) == (1, 0)
    assert adapter.connected is True
    assert adapter.disconnect_calls == 0


def test_farm_recovery_does_not_clear_unrelated_degraded_or_paused_state(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute("UPDATE subscriptions SET lifecycle='paused' WHERE request_id=4")
        connection.execute(
            "UPDATE runtime_state SET lifecycle='degraded', reason='PAUSE_OPTIONAL_FEEDS'"
        )
    recorder.market_data_status(MarketDataStatus("farm_recovered", 2158, None, "ok", 101, ()))
    with connect_v2(database) as connection:
        paused = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=4"
        ).fetchone()[0]
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
    assert paused == "paused"
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS")
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE runtime_state SET lifecycle='degraded', reason='UNRELATED_DIAGNOSTIC'"
        )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "farm ok", 102, ("quotes",))
    )
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        ) == ("degraded", "UNRELATED_DIAGNOSTIC")


def test_farm_recovery_resolves_only_matching_outage_and_fatal_remains_absorbing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "farm one", 101, ("quotes",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2105, None, "farm two", 102, ("quotes",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "farm one ok", 103, ("quotes",))
    )
    with connect_v2(database) as connection:
        gaps = dict(
            connection.execute(
                "SELECT reason, resolved_at_us FROM gaps WHERE reason LIKE 'IBKR_FARM_%'"
            )
        )
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
    assert gaps["IBKR_FARM_2103_DEGRADED"] == 103
    assert gaps["IBKR_FARM_2105_DEGRADED"] is None
    assert tuple(runtime) == ("degraded", "IBKR_FARM_2105_DEGRADED")
    recorder._fatal("TEST_FATAL", 104)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.market_data_status(
            MarketDataStatus("farm_recovered", 2106, None, "farm two ok", 105, ("quotes",))
        )


def test_disconnect_fences_degraded_farm_before_recovery_and_preserves_paused_optional(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    with connect_v2(database) as connection:
        connection.execute("UPDATE subscriptions SET lifecycle='paused' WHERE request_id=4")

    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "quote farm lost", 101, ("quotes",))
    )
    recorder.disconnected(now_us=102)
    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        connection_state = connection.execute(
            "SELECT connection_state FROM runtime_state"
        ).fetchone()[0]
    assert states == {3: "disconnected", 4: "paused"}
    assert connection_state == "disconnected"

    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "quote farm ok", 103, ("quotes",))
    )
    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        recovery = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()[0]
    assert states == {3: "disconnected", 4: "paused"}
    assert tuple(runtime) == ("degraded", "IBKR_DISCONNECT", "disconnected")
    assert recovery == 103

    recorder.reconnect(now_us=104)
    with connect_v2(database) as connection:
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
        connection_state = connection.execute(
            "SELECT connection_state FROM runtime_state"
        ).fetchone()[0]
    assert current == {3: "active", 4: "paused"}
    assert connection_state == "connected"
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "new quote farm lost", 105, ("quotes",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "new quote farm ok", 106, ("quotes",))
    )
    with connect_v2(database) as connection:
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
    assert current == {3: "active", 4: "paused"}


def test_late_farm_recovery_across_reconnect_is_scoped_and_stale_authority_cannot_mutate(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2103, None, "old quote farm lost", 101, ("quotes",))
    )
    recorder.disconnected(now_us=102)
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2105, None, "late bar farm loss", 103, ("bars",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2106, None, "late bar farm ok", 104, ("bars",))
    )
    with connect_v2(database) as connection:
        disconnected = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=1"
            )
        )
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
    assert disconnected == {3: "disconnected", 4: "disconnected"}
    assert tuple(runtime) == ("degraded", "IBKR_DISCONNECT")

    recorder.reconnect(now_us=105)
    recorder.market_data_status(
        MarketDataStatus("farm_degraded", 2105, None, "bar farm lost", 106, ("bars",))
    )
    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2104, None, "old quote farm ok", 107, ("quotes",))
    )
    with connect_v2(database) as connection:
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
    assert current == {3: "active", 4: "degraded"}
    assert tuple(runtime) == ("degraded", "IBKR_FARM_2105_DEGRADED")

    recorder.market_data_status(
        MarketDataStatus("farm_recovered", 2106, None, "current bar farm ok", 108, ("bars",))
    )
    with connect_v2(database) as connection:
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        old_quote_gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED' "
            "AND subscription_id IN "
            "(SELECT subscription_id FROM subscriptions WHERE connection_generation=1)"
        ).fetchone()[0]
    assert current == {3: "active", 4: "active"}
    assert tuple(runtime) == ("running", None)
    assert old_quote_gap is None

    _force_recorder_takeover(database)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.market_data_status(
            MarketDataStatus("farm_recovered", 2104, None, "quote farm ok", 109, ("quotes",))
        )
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason FROM runtime_state"
        ).fetchone()
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
    assert tuple(replacement) == (2, "running", None)
    assert current == {3: "active", 4: "active"}


@pytest.mark.parametrize(
    ("emitted", "expected_states", "expected_runtime"),
    (
        (
            (MarketDataStatus("farm_degraded", 2103, None, "quotes lost", 101, ("quotes",)),),
            {3: "degraded", 4: "active"},
            ("degraded", "IBKR_FARM_2103_DEGRADED"),
        ),
        (
            (MarketDataStatus("farm_degraded", 2105, None, "bars lost", 101, ("bars",)),),
            {3: "active", 4: "degraded"},
            ("running", None),
        ),
        (
            (
                MarketDataStatus("farm_degraded", 2103, None, "quotes lost", 101, ("quotes",)),
                MarketDataStatus("farm_recovered", 2104, None, "quotes ok", 102, ("quotes",)),
            ),
            {3: "active", 4: "active"},
            ("running", None),
        ),
    ),
)
def test_connect_time_farm_callbacks_are_reconciled_before_running(
    tmp_path: Path,
    emitted: tuple[MarketDataStatus, ...],
    expected_states: dict[int, str],
    expected_runtime: tuple[str, str | None],
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class ConnectStatusMarketData(FakeMarketData):
        def connect(self) -> None:
            super().connect()
            assert callable(self.status_callback)
            for status in emitted:
                self.status_callback(status)

    recorder = Recorder(_config(database), ConnectStatusMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        unresolved = tuple(
            connection.execute(
                "SELECT reason FROM gaps WHERE reason LIKE 'IBKR_FARM_%' "
                "AND resolved_at_us IS NULL ORDER BY reason"
            )
        )
    assert states == expected_states
    assert tuple(runtime) == expected_runtime
    assert tuple(row[0] for row in unresolved) == (
        () if len(emitted) == 2 else (f"IBKR_FARM_{emitted[0].code}_DEGRADED",)
    )


def test_subscribe_time_required_farm_outage_prevents_false_active_and_running(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class SubscribeStatusMarketData(FakeMarketData):
        def subscribe(self, fence: CallbackFence) -> None:
            super().subscribe(fence)
            if fence.request_id == 3:
                assert callable(self.status_callback)
                self.status_callback(
                    MarketDataStatus("farm_degraded", 2103, None, "quotes lost", 101, ("quotes",))
                )

    recorder = Recorder(_config(database), SubscribeStatusMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute("SELECT lifecycle, reason FROM runtime_state").fetchone()
        gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()
    assert states == {3: "degraded", 4: "active"}
    assert tuple(runtime) == ("degraded", "IBKR_FARM_2103_DEGRADED")
    assert gap is not None and gap[0] is None


def test_connect_time_farm_callback_cannot_mutate_after_authority_takeover(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class TakeoverStatusMarketData(FakeMarketData):
        def connect(self) -> None:
            super().connect()
            assert callable(self.status_callback)
            self.status_callback(
                MarketDataStatus("farm_degraded", 2103, None, "quotes lost", 101, ("quotes",))
            )
            _force_recorder_takeover(database)
            with connect_v2(database) as connection:
                connection.execute(
                    "UPDATE runtime_state SET reason='REPLACEMENT_OWNS_STARTUP' "
                    "WHERE recorder_generation=2"
                )
            self.status_callback(
                MarketDataStatus("farm_recovered", 2104, None, "quotes ok", 102, ("quotes",))
            )

    adapter = TakeoverStatusMarketData()
    recorder = Recorder(_config(database), adapter)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert adapter.connected is False
    with connect_v2(database) as connection:
        replacement = connection.execute(
            "SELECT recorder_generation, lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
        gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_FARM_2103_DEGRADED'"
        ).fetchone()
    assert tuple(replacement) == (2, "running", "REPLACEMENT_OWNS_STARTUP", "connected")
    assert gap is not None and gap[0] is None


def test_startup_storage_pause_survives_connection_and_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    degraded = RetentionResult(
        cap_state=StorageCapState.DEGRADED,
        database_bytes=95,
        wal_bytes=1,
        payloads_compacted=0,
        receipts_rolled=0,
        expired_rows_deleted=0,
        admission_allowed=True,
        optional_feeds_allowed=False,
        required_action="PAUSE_OPTIONAL_FEEDS",
        checkpoint_attempted=False,
        incremental_vacuum_attempted=False,
    )

    class StartupRetention:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int, precondition: object = None) -> RetentionResult:
            assert now_us == 100
            assert callable(precondition)
            return degraded

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.RetentionManager", StartupRetention)
    adapter = FakeMarketData()
    recorder = Recorder(_config(database), adapter)
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with connect_v2(database) as connection:
        first = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=1"
            )
        )
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state, connection_generation FROM runtime_state"
        ).fetchone()
        storage_gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='STORAGE_DEGRADED_OPTIONAL_PAUSED'"
        ).fetchone()
    assert first == {3: "active", 4: "paused"}
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS", "connected", 1)
    assert storage_gap is not None and storage_gap[0] is None
    assert adapter.cancelled == [4]
    assert [fence.request_id for fence in adapter.subscriptions] == [3]

    recorder.reconnect(now_us=101)
    with connect_v2(database) as connection:
        current = dict(
            connection.execute(
                "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2"
            )
        )
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state, connection_generation FROM runtime_state"
        ).fetchone()
        storage_gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='STORAGE_DEGRADED_OPTIONAL_PAUSED'"
        ).fetchone()
    assert current == {3: "active", 4: "paused"}
    assert tuple(runtime) == ("degraded", "PAUSE_OPTIONAL_FEEDS", "connected", 2)
    assert storage_gap is not None and storage_gap[0] is None


def test_connect_finalization_preserves_unrelated_degraded_reason(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class PreconnectDegradedMarketData(FakeMarketData):
        def connect(self) -> None:
            super().connect()
            with connect_v2(database) as connection:
                connection.execute(
                    "UPDATE runtime_state SET lifecycle='degraded', reason='PRECONNECT_DIAGNOSTIC'"
                )

    recorder = Recorder(_config(database), PreconnectDegradedMarketData())
    recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    with connect_v2(database) as connection:
        states = dict(connection.execute("SELECT request_id, lifecycle FROM subscriptions"))
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert states == {3: "active", 4: "active"}
    assert tuple(runtime) == ("degraded", "PRECONNECT_DIAGNOSTIC", "connected")


def test_fatal_during_connect_is_absorbing_and_marks_connection_disconnected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()

    class FatalConnectMarketData(FakeMarketData):
        def connect(self) -> None:
            super().connect()
            with connect_v2(database) as connection:
                CallbackInbox._record_fatal(
                    connection,
                    "run-1",
                    101,
                    "STARTUP_FATAL",
                    authority=WriterAuthority("run-1", 1, "owner-1"),
                )

    adapter = FatalConnectMarketData()
    recorder = Recorder(_config(database), adapter)
    with pytest.raises(AuthoritativeLeaseLost):
        recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)

    assert adapter.connected is False
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, reason, connection_state FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == ("fatal", "STARTUP_FATAL", "disconnected")
