from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import (
    CallbackFence,
    CallbackInbox,
    DuplicateWriterError,
    InboxFullError,
    InstrumentSpec,
    MarketDataCallback,
    NormalizationError,
    Recorder,
    RecorderConfig,
    SubscriptionSpec,
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


def _fence() -> CallbackFence:
    return CallbackFence(
        run_id="run-1",
        recorder_generation=1,
        connection_generation=2,
        request_id=3,
        subscription_id=None,
    )


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

    leased = inbox.lease_pending("worker-1", now_us=20, lease_us=10, limit=10)
    first_event = inbox.project(leased[0])
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM callback_inbox WHERE source_sequence = ?",
                (admitted[0].source_sequence,),
            ).fetchone()[0]
            == "leased"
        )
    inbox.acknowledge(leased[0], first_event.event_id, acknowledged_at_us=21)
    second_event = inbox.project(leased[1])
    inbox.acknowledge(leased[1], second_event.event_id, acknowledged_at_us=22)

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

    first_lease = inbox.lease_pending("dead-worker", now_us=10, lease_us=5, limit=1)[0]
    projected = inbox.project(first_lease)
    assert inbox.reclaim_expired_leases(now_us=14) == 0
    assert inbox.reclaim_expired_leases(now_us=15) == 1
    retry_lease = inbox.lease_pending("new-worker", now_us=15, lease_us=5, limit=1)[0]
    retried = inbox.project(retry_lease)
    inbox.acknowledge(retry_lease, retried.event_id, acknowledged_at_us=16)

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
    leased = inbox.lease_pending("worker", now_us=20, lease_us=10, limit=10)

    with pytest.raises(NormalizationError):
        inbox.project(leased[0])
    inbox.fail(leased[0], "MALFORMED_CALLBACK", failed_at_us=20)
    event = inbox.project(leased[1])
    inbox.acknowledge(leased[1], event.event_id, acknowledged_at_us=21)

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
    for leased in inbox.lease_pending("worker", now_us=12, lease_us=10, limit=10):
        event = inbox.project(leased)
        inbox.acknowledge(leased, event.event_id, acknowledged_at_us=12)

    receipt = inbox.create_receipt("run-1", created_at_us=13, limit=10)
    compacted = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1, receipt_us=1_000, tombstone_us=1_000),
    ).run(now_us=14, measured_database_bytes=1, measured_wal_bytes=0)

    assert receipt is not None
    assert receipt.callback_count == 2
    assert compacted.payloads_compacted == 2


class FakeMarketData:
    def __init__(self) -> None:
        self.callback = None
        self.disconnect_callback = None
        self.connected = False
        self.subscriptions: list[CallbackFence] = []
        self.cancelled: list[int] = []

    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def set_disconnect_callback(self, callback: object) -> None:
        self.disconnect_callback = callback

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def subscribe(self, fence: CallbackFence) -> None:
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

        def run(self, *, now_us: int) -> RetentionResult:
            assert now_us >= 101
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

    current_recorder.receive(
        prior.fences[0],
        MarketDataCallback("quote", 5_000_102, None, {"event_at_us": 5_000_102}),
    )
    assert current_recorder.drain(now_us=5_000_103) == 0

    with connect_v2(database) as connection:
        callback = connection.execute(
            "SELECT lifecycle, receipt_batch_id FROM callback_inbox WHERE run_id='run-1'"
        ).fetchone()
    assert callback["lifecycle"] == "failed"
    assert callback["receipt_batch_id"] is not None


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
