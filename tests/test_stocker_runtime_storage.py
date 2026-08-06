from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocker_runtime.cli import app as runtime_app
from stocker_runtime.storage import (
    EXPECTED_TABLES,
    IdeaOutputRecord,
    IdentityCollisionError,
    JsonAdmissionError,
    OperationalRepository,
    RetentionManager,
    RetentionPolicy,
    SchemaError,
    StorageCapState,
    canonical_json_text,
    connect_v2,
    deterministic_output_id,
    initialize_database,
    migrate_database,
    migration_plan,
)


def test_initialize_database_creates_exact_immediate_schema_and_writer_pragmas(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"

    result = initialize_database(database, applied_at_us=1_700_000_000_000_000)

    assert result.applied_versions == (1,)
    with connect_v2(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == EXPECTED_TABLES
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1000
        assert connection.execute("PRAGMA journal_size_limit").fetchone()[0] == 67_108_864
        assert connection.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_initialize_database_refuses_existing_or_legacy_files(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE prospective_run(run_id TEXT PRIMARY KEY)")

    with pytest.raises(SchemaError, match="new empty path"):
        initialize_database(database)

    assert sqlite3.connect(database).execute(
        "SELECT name FROM sqlite_schema WHERE name = 'prospective_run'"
    ).fetchone() == ("prospective_run",)
    original_bytes = database.read_bytes()
    with pytest.raises(SchemaError, match="not a Stocker V2"):
        migrate_database(database)
    assert database.read_bytes() == original_bytes
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()


def test_migration_verification_fails_closed_for_future_or_tampered_history(
    tmp_path: Path,
) -> None:
    future = tmp_path / "future.sqlite3"
    initialize_database(future, applied_at_us=1)
    with sqlite3.connect(future) as connection:
        connection.execute(
            "INSERT INTO schema_migrations(version, name, sha256, applied_at_us) "
            "VALUES (2, '0002_future.sql', ?, 2)",
            ("f" * 64,),
        )
    with pytest.raises(SchemaError, match="newer"):
        migrate_database(future)

    tampered = tmp_path / "tampered.sqlite3"
    initialize_database(tampered, applied_at_us=1)
    with sqlite3.connect(tampered) as connection:
        connection.execute("UPDATE schema_migrations SET sha256 = ?", ("0" * 64,))
    with pytest.raises(SchemaError, match="checksum"):
        connect_v2(tampered)


def test_migration_plan_requires_order_and_failed_migration_is_atomic(tmp_path: Path) -> None:
    migration_root = tmp_path / "migrations"
    migration_root.mkdir()
    first = migration_plan()[0]
    (migration_root / first.name).write_text(first.sql, encoding="utf-8")
    database = tmp_path / "v2.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    (migration_root / "0002_broken.sql").write_text(
        "CREATE TABLE should_rollback(value TEXT) STRICT;\nTHIS IS NOT SQL;\n",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.OperationalError):
        migrate_database(database, migration_root=migration_root, applied_at_us=2)

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE name = 'should_rollback'"
            ).fetchone()
            is None
        )
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone() == (1,)

    (migration_root / "0002_broken.sql").unlink()
    (migration_root / "0003_gap.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(SchemaError, match="contiguous"):
        migration_plan(migration_root)


def test_schema_has_no_future_trading_tables_or_columns(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)

    with connect_v2(database) as connection:
        schema = "\n".join(
            str(row[0]).lower()
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY name"
            )
        )
    for forbidden_table in (
        "portfolio_requests",
        "risk_decisions",
        "order_intents",
        "broker_orders",
        "fills",
        "accounts",
        "cash_balances",
        "reconciled_positions",
    ):
        assert f"create table {forbidden_table}" not in schema

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX idea_outputs_run_kind_time_idx")
    with pytest.raises(SchemaError, match="schema structure"):
        connect_v2(database)


def test_canonical_json_admission_is_bounded_and_deterministic() -> None:
    assert canonical_json_text({"z": 1, "a": [True, None]}, max_bytes=64) == (
        '{"a":[true,null],"z":1}'
    )
    with pytest.raises(JsonAdmissionError, match="exceeds 8 bytes"):
        canonical_json_text({"payload": "large"}, max_bytes=8)


def _seed_output_dependencies(database: Path) -> None:
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
            ("run-1", "shadow", "ibkr", 10, "a" * 64, "deadbee", "shadow_protected", "created"),
        )
        connection.execute(
            "INSERT INTO recorder_generations VALUES (?, ?, ?, ?, NULL, 0, NULL)",
            ("run-1", 1, "fixture", 10),
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, "
            "exchange, currency) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("instrument-1", "b" * 64, "stock", "XYZ", "SMART", "USD"),
        )
        connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("callback-1", "run-1", 1, 1, "tick", 20, "{}", "c" * 64, "pending"),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("event-1", "run-1", 1, "instrument-1", "trades", "tick", 20, 20, 1, "{}", "d" * 64),
        )
        connection.execute(
            "INSERT INTO idea_plugins VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("idea", "1", 1, "Idea", "Fixture", "e" * 64, "f" * 64, "{}", 21),
        )
        connection.execute(
            "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
            "parameters_json, parameters_hash, activated_at_us, health, data_class) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "instance-1",
                "idea",
                "1",
                "run-1",
                "shadow",
                "{}",
                "0" * 64,
                22,
                "healthy",
                "shadow_protected",
            ),
        )


def _output(*, emitted_at_us: int = 30, instance_id: str = "instance-1") -> IdeaOutputRecord:
    return IdeaOutputRecord(
        run_id="run-1",
        instance_id=instance_id,
        output_kind="signal",
        subject_instrument_id="instrument-1",
        emitted_at_us=emitted_at_us,
        as_of_at_us=25,
        valid_until_at_us=None,
        direction="long",
        strength=0.5,
        confidence=0.8,
        horizon_us=60_000_000,
        first_input_event_id="event-1",
        last_input_event_id="event-1",
        output_ordinal=0,
        payload={"reason": "fixture"},
        data_class="shadow_protected",
        authority_status="recorded",
    )


def test_repository_output_identity_is_deterministic_and_collision_checked(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    repository = OperationalRepository(database)

    first = repository.put_idea_output(_output())
    retry = repository.put_idea_output(_output())

    assert first.output_id == retry.output_id
    assert first.inserted is True
    assert retry.inserted is False
    assert first.output_id == deterministic_output_id(_output())
    with pytest.raises(IdentityCollisionError, match="different content"):
        repository.put_idea_output(_output(emitted_at_us=31))


def test_repository_foreign_keys_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    repository = OperationalRepository(database)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        repository.put_idea_output(_output(instance_id="missing"))


def test_repository_cannot_promote_a_proposal(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    repository = OperationalRepository(database)
    signal = _output()
    proposal = IdeaOutputRecord(
        **{
            **signal.__dict__,
            "output_kind": "proposed_trade",
            "authority_status": "recorded",
        }
    )

    with pytest.raises(ValueError, match="requires authority_status=unapproved"):
        repository.put_idea_output(proposal)


def _seed_callback_for_retention(
    database: Path,
    *,
    uid: str,
    received_at_us: int,
    lifecycle: str = "acknowledged",
    normalized_event_id: str | None = "event-1",
    acknowledged_at_us: int | None = 1,
    receipt_batch_id: str | None = "receipt-1",
) -> int:
    with connect_v2(database) as connection:
        cursor = connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle, normalized_event_id, acknowledged_at_us, "
            "receipt_batch_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uid,
                "run-1",
                1,
                1,
                "tick",
                received_at_us,
                '{"private":"evidence"}',
                "a" * 64,
                lifecycle,
                normalized_event_id,
                acknowledged_at_us,
                receipt_batch_id,
            ),
        )
        return int(cursor.lastrowid)


def test_retention_compacts_only_durably_projected_acknowledged_receipted_payloads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("receipt-1", "run-1", 2, 5, 4, 1, 4, "{}", "{}", "1" * 64, None, None, 4),
        )
    eligible = _seed_callback_for_retention(database, uid="eligible", received_at_us=1)
    pending = _seed_callback_for_retention(
        database,
        uid="pending",
        received_at_us=1,
        lifecycle="pending",
        normalized_event_id=None,
        acknowledged_at_us=None,
        receipt_batch_id=None,
    )
    unreceipted = _seed_callback_for_retention(
        database, uid="unreceipted", received_at_us=1, receipt_batch_id=None
    )
    recent = _seed_callback_for_retention(
        database, uid="recent", received_at_us=99, acknowledged_at_us=99
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=10, receipt_us=1_000, tombstone_us=1_000),
    )

    result = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert result.payloads_compacted == 1
    with connect_v2(database) as connection:
        payloads = {
            int(row["source_sequence"]): row["payload_json"]
            for row in connection.execute(
                "SELECT source_sequence, payload_json FROM callback_inbox"
            )
        }
    assert payloads[eligible] is None
    assert payloads[pending] is not None
    assert payloads[unreceipted] is not None
    assert payloads[recent] is not None


def test_receipt_rotation_rolls_permanent_watermark_before_deletion(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("r1", "run-1", 1, 2, 2, 10, 20, "{}", "{}", "1" * 64, None, None, 20),
                ("r2", "run-1", 3, 4, 2, 30, 40, "{}", "{}", "2" * 64, None, None, 40),
                ("r3", "run-1", 5, 6, 2, 50, 60, "{}", "{}", "3" * 64, None, None, 99),
            ],
        )
    manager = RetentionManager(
        database,
        RetentionPolicy(receipt_us=50, max_receipts_per_run=2, callback_payload_us=1_000),
    )

    result = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert result.receipts_rolled == 2
    with connect_v2(database) as connection:
        assert [
            str(row["batch_id"])
            for row in connection.execute(
                "SELECT batch_id FROM callback_receipts ORDER BY first_source_sequence"
            )
        ] == ["r3"]
        watermark = connection.execute(
            "SELECT compacted_through_sequence, cumulative_callback_count, "
            "rolled_receipt_chain_hash FROM callback_compaction_watermarks WHERE run_id = ?",
            ("run-1",),
        ).fetchone()
    assert tuple(watermark) == (4, 4, "2" * 64)


def test_retention_cap_states_fail_stop_without_pruning_unexpired_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    repository = OperationalRepository(database)
    stored = repository.put_idea_output(_output())
    manager = RetentionManager(database, RetentionPolicy(database_cap_bytes=100))

    soft = manager.run(now_us=100, measured_database_bytes=85, measured_wal_bytes=0)
    degraded = manager.run(now_us=100, measured_database_bytes=95, measured_wal_bytes=0)
    fatal = manager.run(now_us=100, measured_database_bytes=100, measured_wal_bytes=0)

    assert soft.cap_state is StorageCapState.SOFT
    assert soft.admission_allowed is True
    assert degraded.cap_state is StorageCapState.DEGRADED
    assert degraded.optional_feeds_allowed is False
    assert fatal.cap_state is StorageCapState.FATAL
    assert fatal.admission_allowed is False
    assert fatal.required_action == "STORAGE_CAP_FATAL"
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT output_id FROM idea_outputs WHERE output_id = ?", (stored.output_id,)
            ).fetchone()
            is not None
        )


def test_retention_rejects_maintenance_batches_above_hard_bound() -> None:
    with pytest.raises(ValueError, match="10,000"):
        RetentionPolicy(maintenance_batch_rows=10_001)


def test_retention_uses_one_shared_batch_budget_and_expires_dependencies_in_order(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    repository = OperationalRepository(database)
    stored = repository.put_idea_output(_output())
    manager = RetentionManager(
        database,
        RetentionPolicy(
            idea_shadow_us=10,
            raw_market_event_us=10,
            maintenance_batch_rows=1,
        ),
    )

    first = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    second = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert first.payloads_compacted + first.receipts_rolled + first.expired_rows_deleted == 1
    assert second.payloads_compacted + second.receipts_rolled + second.expired_rows_deleted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM idea_outputs WHERE output_id = ?", (stored.output_id,)
            ).fetchone()
            is None
        )
        assert (
            connection.execute("SELECT 1 FROM market_events WHERE event_id = 'event-1'").fetchone()
            is None
        )


def test_planned_ui_projection_queries_use_declared_indexes(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    queries = {
        "market_latest_event_time_idx": (
            "SELECT instrument_id FROM market_latest WHERE event_at_us <= ? "
            "ORDER BY event_at_us DESC, instrument_id LIMIT ?",
            (100, 250),
        ),
        "idea_outputs_instance_kind_time_idx": (
            "SELECT output_id FROM idea_outputs WHERE instance_id = ? AND output_kind = ? "
            "ORDER BY as_of_at_us DESC, output_id LIMIT ?",
            ("instance", "signal", 200),
        ),
        "shadow_positions_run_lifecycle_idx": (
            "SELECT position_id FROM shadow_positions WHERE run_id = ? AND lifecycle = ? "
            "ORDER BY opened_at_us DESC, position_id LIMIT ?",
            ("run", "open", 200),
        ),
        "incidents_unresolved_idx": (
            "SELECT incident_id FROM incidents WHERE run_id = ? AND resolved_at_us IS NULL "
            "ORDER BY severity, opened_at_us LIMIT ?",
            ("run", 200),
        ),
    }
    with connect_v2(database) as connection:
        for expected_index, (query, parameters) in queries.items():
            plan = " ".join(
                str(row["detail"])
                for row in connection.execute(f"EXPLAIN QUERY PLAN {query}", parameters)
            )
            assert expected_index in plan


def test_database_cli_is_machine_readable_and_never_returns_payloads(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    runner = CliRunner()

    initialized = runner.invoke(runtime_app, ["init", str(database)])
    migrated = runner.invoke(runtime_app, ["migrate", str(database)])
    retained = runner.invoke(runtime_app, ["retain", str(database), "--now-us", "100"])

    assert initialized.exit_code == migrated.exit_code == retained.exit_code == 0
    init_payload = json.loads(initialized.stdout)
    migration_payload = json.loads(migrated.stdout)
    retention_payload = json.loads(retained.stdout)
    assert init_payload == {"applied_versions": [1], "current_version": 1, "status": "ok"}
    assert migration_payload == {"applied_versions": [], "current_version": 1, "status": "ok"}
    assert retention_payload["status"] == "ok"
    assert retention_payload["cap_state"] in {"normal", "soft_cap", "degraded"}
    assert "payload_json" not in retained.stdout
    assert "evidence" not in retained.stdout
