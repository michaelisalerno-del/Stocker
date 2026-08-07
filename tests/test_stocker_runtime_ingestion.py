from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import (
    AuthoritativeLeaseLost,
    CallbackFence,
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
    SubscriptionSpec,
    WriterAuthority,
)
from stocker_runtime.storage import (
    RetentionResult,
    StorageCapState,
    connect_v2,
    initialize_database,
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
        {"mode": "shadow"},
        {"mode": "paper"},
        {"mode": "live"},
        {"host": "192.0.2.1"},
        {"read_only": False},
        {"external_read_only_verified": False},
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
    config_path.write_text(
        json.dumps(_config(tmp_path / "v2.sqlite3").model_dump(mode="json")),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["validate-recorder", str(config_path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "host": "127.0.0.1",
        "mode": "prospective_record",
        "read_only": True,
        "status": "ok",
    }


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


class FakeMarketData:
    def __init__(
        self, *, fail_connect: bool = False, fail_subscribe: set[int] | None = None
    ) -> None:
        self.callback = None
        self.disconnect_callback = None
        self.status_callback = None
        self.connected = False
        self.subscriptions: list[CallbackFence] = []
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

    def subscribe(self, fence: CallbackFence) -> None:
        if fence.request_id in self.fail_subscribe:
            raise RuntimeError("subscribe failed")
        self.subscriptions.append(fence)

    def cancel(self, request_id: int) -> None:
        self.cancelled.append(request_id)

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


def test_recorder_owns_one_writer_and_restart_reclaims_only_after_stale_lease(
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

    restarted = Recorder(
        _config(database, owner_id="owner-2", writer_lease_stale_us=5_000_000),
        FakeMarketData(),
    ).start(now_us=5_000_101, instruments=(instrument,), subscriptions=specs)

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
    assert tuple(old) == (5_000_101, 0)


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
                "SELECT subscription_id, reason, continuity_required, resolved_at_us "
                "FROM gaps ORDER BY started_at_us, gap_id"
            )
        )
    assert required.connection_generation == optional.connection_generation == 1
    assert reconnected.connection_generation == 2
    assert any(row["reason"] == "STREAM_STALE" and row["continuity_required"] == 1 for row in gaps)
    assert any(row["reason"] == "IBKR_DISCONNECT" for row in gaps)
    assert any(row["resolved_at_us"] == 115 for row in gaps)


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
        def place_order(self) -> None:
            raise AssertionError("must never be called")

    with pytest.raises(Exception, match="unsafe broker capability"):
        Recorder(_config(tmp_path / "v2.sqlite3"), UnsafeMarketData())


def test_replay_cli_runs_one_offline_recorder_lifecycle(tmp_path: Path) -> None:
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
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "stopped"
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
    Recorder(_config(database), FakeMarketData()).start(
        now_us=100, instruments=(instrument,), subscriptions=specs
    )

    second = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=5_000_000,
        ),
        FakeMarketData(),
    ).start(now_us=5_000_101, instruments=(instrument,), subscriptions=specs)

    assert second.run_id == "run-2"
    with connect_v2(database) as connection:
        runs = dict(connection.execute("SELECT run_id, status FROM runs"))
        states = dict(connection.execute("SELECT run_id, lifecycle FROM runtime_state"))
    assert runs == {"run-1": "stopped", "run-2": "running"}
    assert states == {"run-1": "stopped", "run-2": "running"}


def test_late_prior_run_callback_is_failed_and_receipted_by_current_recorder(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    prior = Recorder(_config(database), FakeMarketData()).start(
        now_us=100, instruments=(instrument,), subscriptions=specs
    )
    current_recorder = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=5_000_000,
        ),
        FakeMarketData(),
    )
    current_recorder.start(
        now_us=5_000_101,
        instruments=(instrument,),
        subscriptions=specs,
    )

    callback = MarketDataCallback("quote", 5_000_102, None, {"event_at_us": 5_000_102})
    first = current_recorder.receive(
        prior.fences[0],
        callback,
    )
    retry = current_recorder.receive(prior.fences[0], callback)
    assert current_recorder.drain(now_us=5_000_103) == 0

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
    assert _price_tick_projection("quotes", 4) is None
    assert _price_tick_projection("trades", 4) == ("trade", "last")
    assert _size_tick_projection("quotes", 3) == ("quote", "ask_size")
    assert _size_tick_projection("trades", 3) is None
    assert _size_tick_projection("trades", 5) == ("trade", "size")


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
            "process_heartbeat_at_us=101 WHERE run_id='run-1'"
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


def test_persisted_fatal_is_absorbing_for_same_object_and_future_start(tmp_path: Path) -> None:
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
    with pytest.raises(Exception, match="fatal"):
        Recorder(_config(database, run_id="run-2"), FakeMarketData()).start(
            now_us=103, instruments=(instrument,), subscriptions=specs
        )


def test_post_admission_projection_failure_preserves_callback_and_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    recorder = Recorder(_config(database), FakeMarketData())
    state = recorder.start(now_us=100, instruments=(instrument,), subscriptions=specs)
    admitted = recorder.receive(
        state.fences[0],
        MarketDataCallback("quote", 101, None, {"event_at_us": 101, "bid": 1.0}),
    )
    monkeypatch.setattr(
        recorder.inbox,
        "project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("injected projection failure")
        ),
    )
    with pytest.raises(Exception, match="post-admission"):
        recorder.drain(now_us=102)
    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, payload_json FROM callback_inbox WHERE source_sequence=?",
            (admitted.source_sequence,),
        ).fetchone()
        status = connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
    assert callback[0] == "leased"
    assert callback[1] is not None
    assert status == "fatal"


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
    ((3, "degraded", "disconnected", "disconnected"), (4, "running", "active", "paused")),
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
    assert (runtime, optional, gap) == ("running", "paused", 0)


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
    result = RetentionManager(
        database, RetentionPolicy(callback_payload_us=1, receipt_us=1_000, tombstone_us=1_000)
    ).run(now_us=12, measured_database_bytes=1, measured_wal_bytes=0)
    assert result.payloads_compacted == 4


def _force_recorder_takeover(database: Path) -> None:
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 2, 'replacement', 101)"
        )
        connection.execute(
            "UPDATE runtime_state SET recorder_generation=2, lifecycle='running', "
            "process_heartbeat_at_us=101 WHERE run_id='run-1'"
        )


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
            "SELECT recorder_generation, lifecycle, reason FROM runtime_state"
        ).fetchone()
    assert tuple(replacement) == (2, "running", None)


def test_required_subscribe_failure_disconnects_earlier_optional_success(tmp_path: Path) -> None:
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
    assert states == {3: "disconnected", 4: "disconnected"}
    assert runtime == "degraded"
    assert adapter.connected is False


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
    second = Recorder(
        _config(
            database,
            run_id="run-2",
            owner_id="owner-2",
            writer_lease_stale_us=5_000_000,
        ),
        FakeMarketData(),
    )
    second_state = second.start(now_us=5_000_103, instruments=(instrument,), subscriptions=specs)
    ask = second.receive(
        second_state.fences[0],
        MarketDataCallback("quote", 5_000_104, None, {"event_at_us": 5_000_104, "ask": 101.0}),
    )
    second.drain(now_us=5_000_105)
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
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "stopped"
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
        json.dumps(_config(database, writer_lease_stale_us=5_000_000).model_dump(mode="json")),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({"instruments": [], "subscriptions": [], "callbacks": []}), encoding="utf-8"
    )
    result = CliRunner().invoke(
        app,
        ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "5000002"],
    )
    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["error"] == "ReplayBlockedError"
    with connect_v2(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "stopped"
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
        json.dumps(_config(database, writer_lease_stale_us=5_000_000).model_dump(mode="json")),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({"instruments": [], "subscriptions": [], "callbacks": []}), encoding="utf-8"
    )
    result = CliRunner().invoke(
        app,
        ["replay-recorder", str(config_path), str(fixture_path), "--now-us", "5000002"],
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
            "SELECT recorder_generation, lifecycle, reason FROM runtime_state"
        ).fetchone()
        status = connection.execute("SELECT status FROM runs WHERE run_id='run-1'").fetchone()[0]
    assert tuple(replacement) == (2, "running", None)
    assert status == "running"


@pytest.mark.parametrize("second_failure", ("takeover", "database"))
def test_required_abort_disconnects_when_second_failure_persistence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_failure: str
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    instrument, specs = _specs()
    adapter = FakeMarketData(fail_subscribe={3})
    recorder = Recorder(_config(database), adapter)

    def fail_second_persistence(**_kwargs: object) -> None:
        if second_failure == "takeover":
            _force_recorder_takeover(database)
            with connect_v2(database) as connection:
                connection.execute(
                    "UPDATE runtime_state SET reason='REPLACEMENT_OWNS_STATE' "
                    "WHERE run_id='run-1' AND recorder_generation=2"
                )
            raise AuthoritativeLeaseLost("replacement took authority")
        raise sqlite3.OperationalError("injected second persistence failure")

    monkeypatch.setattr(recorder, "_persist_connection_failure", fail_second_persistence)
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
            "SELECT recorder_generation, lifecycle, reason FROM runtime_state"
        ).fetchone()
    assert scoped == 1
    assert fabricated_global == 0
    if second_failure == "takeover":
        assert tuple(runtime) == (2, "running", "REPLACEMENT_OWNS_STATE")
    else:
        assert tuple(runtime) == (1, "degraded", "IBKR_SUBSCRIBE_FAILED")


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
