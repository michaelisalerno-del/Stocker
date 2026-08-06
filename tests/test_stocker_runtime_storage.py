from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocker_runtime.cli import app as runtime_app
from stocker_runtime.storage import (
    EXPECTED_TABLES,
    CallbackReceiptRecord,
    IdeaOutputRecord,
    IdentityCollisionError,
    JsonAdmissionError,
    MaintenanceDeadlineExceeded,
    OperationalRepository,
    ProvenanceError,
    RetentionInvariantError,
    RetentionManager,
    RetentionPolicy,
    SchemaError,
    StorageCapState,
    callback_rows_hash,
    canonical_json_text,
    connect_v2,
    deterministic_output_id,
    initialize_database,
    migrate_database,
    migration_plan,
    receipt_chain_hash,
    verify_database,
)
from stocker_runtime.storage.retention import (
    ACK_PAYLOAD_CANDIDATES_SQL,
    ACK_TOMBSTONE_CANDIDATES_SQL,
    FAILED_PAYLOAD_CANDIDATES_SQL,
    FAILED_TOMBSTONE_CANDIDATES_SQL,
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


def test_all_pending_migrations_roll_back_when_a_later_migration_fails(tmp_path: Path) -> None:
    migration_root = tmp_path / "migrations"
    migration_root.mkdir()
    first = migration_plan()[0]
    (migration_root / first.name).write_text(first.sql, encoding="utf-8")
    database = tmp_path / "v2.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    (migration_root / "0002_valid.sql").write_text(
        "CREATE INDEX temporary_pending_idx ON runs(started_at_us);",
        encoding="utf-8",
    )
    (migration_root / "0003_broken.sql").write_text("THIS IS NOT SQL;", encoding="utf-8")

    with pytest.raises(sqlite3.OperationalError):
        migrate_database(database, migration_root=migration_root, applied_at_us=2)

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE name = 'temporary_pending_idx'"
            ).fetchone()
            is None
        )
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone() == (1,)


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


def test_schema_rejects_invalid_json_and_mode_data_class_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
                (
                    "bad-run",
                    "shadow",
                    "ibkr",
                    1,
                    "a" * 64,
                    "deadbee",
                    "prospective_protected",
                    "created",
                ),
            )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
            (
                "run-json",
                "shadow",
                "ibkr",
                1,
                "a" * 64,
                "deadbee",
                "shadow_protected",
                "created",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "opened_at_us, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("bad-json", "run-json", "runtime", "info", "bad", 1, "{not-json"),
            )


def test_schema_rejects_valid_but_noncanonical_json(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        statements = (
            "UPDATE callback_inbox SET payload_json = '{ \"a\": 1 }' WHERE source_sequence = 1",
            "UPDATE market_events SET payload_json = '{\"z\":1,\"a\":2}' "
            "WHERE event_id = 'event-1'",
            "UPDATE idea_plugins SET manifest_json = '{ \"a\": 1 }' WHERE idea_id = 'idea'",
            "UPDATE idea_instances SET parameters_json = '{\"z\":1,\"a\":2}' "
            "WHERE instance_id = 'instance-1'",
        )
        for statement in statements:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(statement)


def test_acknowledgement_and_decoupled_event_references_enforce_provenance(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        cursor = connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES ('ack-2', 'run-1', 1, 1, 'tick', 21, "
            "'{}', ?, 'pending')",
            ("a" * 64,),
        )
        sequence = int(cursor.lastrowid)
        with pytest.raises(sqlite3.IntegrityError, match="acknowledgement_event_mismatch"):
            connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'acknowledged', "
                "normalized_event_id = 'event-1' WHERE source_sequence = ?",
                (sequence,),
            )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES ('event-2', 'run-1', ?, 'instrument-1', "
            "'trades', 'tick', 21, 21, 1, '{}', ?)",
            (sequence, "b" * 64),
        )
        connection.execute(
            "UPDATE callback_inbox SET lifecycle = 'acknowledged', "
            "normalized_event_id = 'event-2', acknowledged_at_us = 21 WHERE source_sequence = ?",
            (sequence,),
        )
        connection.execute(
            "INSERT INTO idea_checkpoints VALUES "
            "('instance-1', 'event-1', '{}', ?, 20, 20, 0)",
            ("c" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="checkpoint_event_provenance"):
            connection.execute(
                "UPDATE idea_checkpoints SET last_market_event_id = 'missing' "
                "WHERE instance_id = 'instance-1'"
            )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us, latest_event_id) VALUES "
            "('sub-1', 'run-1', 1, 1, 'instrument-1', 'trades', 10, 'open', ?, 20, 'event-1')",
            ("d" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="subscription_latest_event_provenance"):
            connection.execute(
                "UPDATE subscriptions SET feed_kind = 'quotes' WHERE subscription_id = 'sub-1'"
            )


def test_shadow_leg_event_references_match_position_run_and_instrument(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    proposal = IdeaOutputRecord(
        **{
            **_output().__dict__,
            "output_kind": "proposed_trade",
            "authority_status": "unapproved",
        }
    )
    stored = OperationalRepository(database).put_idea_output(proposal)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, lifecycle, cost_model_id, fill_model_id, currency, data_class) "
            "VALUES ('position-1', ?, 'run-1', 'instance-1', 'open', 'cost', 'fill', "
            "'USD', 'shadow_protected')",
            (stored.output_id,),
        )
        connection.execute(
            "INSERT INTO shadow_legs(position_id, leg_number, instrument_id, side, quantity, "
            "entry_market_event_id) VALUES "
            "('position-1', 0, 'instrument-1', 'buy', 1, 'event-1')"
        )
        with pytest.raises(sqlite3.IntegrityError, match="entry_event_provenance"):
            connection.execute(
                "UPDATE shadow_legs SET entry_market_event_id = 'missing' "
                "WHERE position_id = 'position-1' AND leg_number = 0"
            )


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


class _ChangingPayload(Mapping[str, object]):
    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        return dict(self.items())[key]

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self.items()))

    def __len__(self) -> int:
        return 1

    def items(self) -> object:  # type: ignore[override]
        self.reads += 1
        if self.reads == 1:
            return {"reason": "stable"}.items()
        return {"brokerOrderId": "smuggled"}.items()


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


def test_repository_snapshots_a_stateful_payload_exactly_once(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    payload = _ChangingPayload()
    base = _output()
    proposal = IdeaOutputRecord(
        **{
            **base.__dict__,
            "output_kind": "proposed_trade",
            "payload": payload,
            "authority_status": "unapproved",
        }
    )

    stored = OperationalRepository(database).put_idea_output(proposal)

    assert payload.reads == 1
    with connect_v2(database) as connection:
        row = connection.execute(
            "SELECT payload_json, payload_hash FROM idea_outputs WHERE output_id = ?",
            (stored.output_id,),
        ).fetchone()
    assert row["payload_json"] == '{"reason":"stable"}'
    assert "broker" not in str(row["payload_json"]).lower()


def test_repository_foreign_keys_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    repository = OperationalRepository(database)

    with pytest.raises(ProvenanceError, match="provenance"):
        repository.put_idea_output(_output(instance_id="missing"))


def test_repository_and_schema_reject_cross_run_provenance(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
            (
                "run-2",
                "shadow",
                "ibkr",
                10,
                "1" * 64,
                "deadbee",
                "shadow_protected",
                "created",
            ),
        )
        connection.execute(
            "INSERT INTO recorder_generations VALUES (?, ?, ?, ?, NULL, 0, NULL)",
            ("run-2", 1, "fixture", 10),
        )
        connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("callback-2", "run-2", 1, 1, "tick", 20, "{}", "2" * 64, "pending"),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("event-2", "run-2", 2, "instrument-1", "trades", "tick", 20, 20, 1, "{}", "3" * 64),
        )
        with pytest.raises(sqlite3.IntegrityError, match="provenance_mismatch"):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("cross", "run-1", 2, "instrument-1", "trades", "tick", 20, 20, 1, "{}", "4" * 64),
            )

    cross_event = IdeaOutputRecord(
        **{
            **_output().__dict__,
            "first_input_event_id": "event-2",
            "last_input_event_id": "event-2",
        }
    )
    with pytest.raises(ProvenanceError, match="provenance"):
        OperationalRepository(database).put_idea_output(cross_event)

    valid = OperationalRepository(database).put_idea_output(_output())
    with (
        connect_v2(database) as connection,
        pytest.raises(sqlite3.IntegrityError, match="proposal_provenance_mismatch"),
    ):
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, "
            "run_id, instance_id, lifecycle, cost_model_id, fill_model_id, currency, "
            "data_class) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "not-a-trade",
                valid.output_id,
                "run-1",
                "instance-1",
                "pending",
                "cost-v1",
                "fill-v1",
                "USD",
                "shadow_protected",
            ),
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET run_id = 'run-2' WHERE instance_id = 'instance-1'"
        )
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert valid.output_id
    assert any(row[0] in {"idea_instances", "idea_outputs"} for row in violations)

    with connect_v2(database):
        pass
    with pytest.raises(SchemaError, match="foreign-key violations"):
        verify_database(database)


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
            "payload_sha256, lifecycle, receipt_batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uid,
                "run-1",
                1,
                1,
                "tick",
                received_at_us,
                '{"private":"evidence"}',
                "a" * 64,
                "pending" if lifecycle == "acknowledged" else lifecycle,
                receipt_batch_id,
            ),
        )
        sequence = int(cursor.lastrowid)
        if lifecycle == "acknowledged":
            event_id = f"retention-event-{uid}"
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    "run-1",
                    sequence,
                    "instrument-1",
                    "trades",
                    "tick",
                    received_at_us,
                    received_at_us,
                    1,
                    "{}",
                    "b" * 64,
                ),
            )
            connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'acknowledged', "
                "normalized_event_id = ?, acknowledged_at_us = ? WHERE source_sequence = ?",
                (event_id, acknowledged_at_us, sequence),
            )
        return sequence


def _insert_receipt(
    database: Path,
    *,
    batch_id: str,
    first_sequence: int,
    last_sequence: int,
    created_at_us: int,
    prior_chain_hash: str = "0" * 64,
    run_id: str = "run-1",
) -> str:
    with connect_v2(database) as connection:
        rows = tuple(
            connection.execute(
                "SELECT event_uid, payload_sha256, run_id, source_sequence, callback_kind, "
                "lifecycle, received_at_us, provider_at_us, normalized_event_id, "
                "acknowledged_at_us, failure_code FROM callback_inbox "
                "WHERE run_id = ? AND source_sequence BETWEEN ? AND ? ORDER BY source_sequence",
                (run_id, first_sequence, last_sequence),
            )
        )
    assert len(rows) == last_sequence - first_sequence + 1
    kind_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        kind = str(row["callback_kind"])
        status = str(row["lifecycle"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    count = len(rows)
    record = CallbackReceiptRecord(
        batch_id=batch_id,
        run_id=run_id,
        first_source_sequence=first_sequence,
        last_source_sequence=last_sequence,
        callback_count=count,
        first_received_at_us=int(rows[0]["received_at_us"]),
        last_received_at_us=int(rows[-1]["received_at_us"]),
        kind_counts=kind_counts,
        status_counts=status_counts,
        callback_rows_hash=callback_rows_hash(tuple(dict(row) for row in rows)),
        first_normalized_event_id=rows[0]["normalized_event_id"],
        last_normalized_event_id=rows[-1]["normalized_event_id"],
        created_at_us=created_at_us,
        prior_chain_hash=prior_chain_hash,
    )
    chain_hash = receipt_chain_hash(record)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_receipts(batch_id, run_id, first_source_sequence, "
            "last_source_sequence, callback_count, first_received_at_us, last_received_at_us, "
            "kind_counts_json, status_counts_json, callback_rows_hash, prior_chain_hash, "
            "chained_payload_hash, "
            "first_normalized_event_id, last_normalized_event_id, created_at_us) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.batch_id,
                record.run_id,
                record.first_source_sequence,
                record.last_source_sequence,
                record.callback_count,
                record.first_received_at_us,
                record.last_received_at_us,
                canonical_json_text(record.kind_counts, max_bytes=16_384),
                canonical_json_text(record.status_counts, max_bytes=16_384),
                record.callback_rows_hash,
                record.prior_chain_hash,
                chain_hash,
                record.first_normalized_event_id,
                record.last_normalized_event_id,
                record.created_at_us,
            ),
        )
    return chain_hash


def test_retention_compacts_only_durably_projected_acknowledged_receipted_payloads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
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
    _insert_receipt(
        database,
        batch_id="receipt-1",
        first_sequence=eligible,
        last_sequence=recent,
        created_at_us=4,
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


def test_compaction_rejects_a_corrupt_recent_receipt_before_payload_deletion(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    sequence = _seed_callback_for_retention(
        database, uid="corrupt-proof", received_at_us=1, receipt_batch_id="recent-corrupt"
    )
    _insert_receipt(
        database,
        batch_id="recent-corrupt",
        first_sequence=sequence,
        last_sequence=sequence,
        created_at_us=99,
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_receipts SET chained_payload_hash = ? WHERE batch_id = ?",
            ("f" * 64, "recent-corrupt"),
        )

    with pytest.raises(RetentionInvariantError, match="content hash"):
        RetentionManager(
            database,
            RetentionPolicy(callback_payload_us=10, receipt_us=1_000),
        ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT payload_json FROM callback_inbox WHERE source_sequence = ?", (sequence,)
            ).fetchone()["payload_json"]
            is not None
        )


def test_compaction_rejects_a_self_consistent_receipt_forged_away_from_callback_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    sequence = _seed_callback_for_retention(
        database, uid="forged-proof", received_at_us=1, receipt_batch_id="forged"
    )
    _insert_receipt(
        database,
        batch_id="forged",
        first_sequence=sequence,
        last_sequence=sequence,
        created_at_us=99,
    )
    forged = CallbackReceiptRecord(
        batch_id="forged",
        run_id="run-1",
        first_source_sequence=sequence,
        last_source_sequence=sequence,
        callback_count=1,
        first_received_at_us=1,
        last_received_at_us=1,
        kind_counts={"forged": 1},
        status_counts={"acknowledged": 1},
        callback_rows_hash="f" * 64,
        first_normalized_event_id="retention-event-forged-proof",
        last_normalized_event_id="retention-event-forged-proof",
        created_at_us=99,
        prior_chain_hash="0" * 64,
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_receipts SET kind_counts_json = ?, callback_rows_hash = ?, "
            "chained_payload_hash = ? WHERE batch_id = 'forged'",
            (
                canonical_json_text(forged.kind_counts, max_bytes=16_384),
                forged.callback_rows_hash,
                receipt_chain_hash(forged),
            ),
        )

    with pytest.raises(RetentionInvariantError, match="authoritative callback rows"):
        RetentionManager(database, RetentionPolicy(callback_payload_us=10, receipt_us=1_000)).run(
            now_us=100, measured_database_bytes=1, measured_wal_bytes=0
        )

    with connect_v2(database) as connection:
        assert connection.execute(
            "SELECT payload_json FROM callback_inbox WHERE source_sequence = ?", (sequence,)
        ).fetchone()[0] is not None


def test_receipt_rotation_rolls_permanent_watermark_before_deletion(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'run-1'")
        for sequence in range(101, 107):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'run-1', 1, 1, 'tick', ?, NULL, ?, 'failed', 'fixture', ?)",
                (
                    sequence,
                    f"rotation-{sequence}",
                    sequence,
                    "a" * 64,
                    f"r{1 + (sequence - 101) // 2}",
                ),
            )
    first_hash = _insert_receipt(
        database, batch_id="r1", first_sequence=101, last_sequence=102, created_at_us=20
    )
    second_hash = _insert_receipt(
        database,
        batch_id="r2",
        first_sequence=103,
        last_sequence=104,
        created_at_us=40,
        prior_chain_hash=first_hash,
    )
    _insert_receipt(
        database,
        batch_id="r3",
        first_sequence=105,
        last_sequence=106,
        created_at_us=99,
        prior_chain_hash=second_hash,
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
            "rolled_receipt_chain_hash, last_receipt_chain_hash "
            "FROM callback_compaction_watermarks WHERE run_id = ?",
            ("run-1",),
        ).fetchone()
    assert tuple(watermark)[:2] == (104, 4)
    assert watermark["last_receipt_chain_hash"] == second_hash
    assert watermark["rolled_receipt_chain_hash"] != second_hash


def test_malformed_receipt_rolls_back_without_deleting_any_proof(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'run-1'")
        for sequence in (51, 52):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'run-1', 1, 1, 'tick', ?, NULL, ?, 'failed', 'fixture', 'bad')",
                (sequence, f"bad-{sequence}", sequence, "a" * 64),
            )
    _insert_receipt(database, batch_id="bad", first_sequence=51, last_sequence=52, created_at_us=1)
    with connect_v2(database) as connection:
        connection.execute("UPDATE callback_receipts SET callback_count = 3 WHERE batch_id = 'bad'")

    with pytest.raises(RetentionInvariantError, match="sequence/count"):
        RetentionManager(database, RetentionPolicy(receipt_us=10)).run(
            now_us=100, measured_database_bytes=1, measured_wal_bytes=0
        )

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT batch_id FROM callback_receipts WHERE batch_id = 'bad'"
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM callback_compaction_watermarks WHERE run_id = 'run-1'"
            ).fetchone()
            is None
        )


def test_closed_failed_callback_compacts_and_expires_only_with_receipt_proof(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    failed_sequence = _seed_callback_for_retention(
        database,
        uid="failed",
        received_at_us=1,
        lifecycle="failed",
        normalized_event_id=None,
        acknowledged_at_us=None,
        receipt_batch_id="failed-receipt",
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_inbox SET failure_code = 'malformed' WHERE source_sequence = ?",
            (failed_sequence,),
        )
    _insert_receipt(
        database,
        batch_id="failed-receipt",
        first_sequence=failed_sequence,
        last_sequence=failed_sequence,
        created_at_us=99,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=10, tombstone_us=10, receipt_us=1_000),
    )

    active = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'run-1'")
    closed = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert active.payloads_compacted == 0
    assert closed.payloads_compacted == 1
    assert closed.expired_rows_deleted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM callback_inbox WHERE source_sequence = ?", (failed_sequence,)
            ).fetchone()
            is None
        )


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


def test_wal_cap_is_fatal_after_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)

    result = RetentionManager(database, RetentionPolicy(wal_cap_bytes=100)).run(
        now_us=100,
        measured_database_bytes=1,
        measured_wal_bytes=100,
    )

    assert result.cap_state is StorageCapState.FATAL
    assert result.admission_allowed is False
    assert result.required_action == "WAL_CAP_FATAL"


def test_retention_deadline_rolls_back_safely(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    clock_values = iter((0.0, 1.0))
    manager = RetentionManager(
        database,
        RetentionPolicy(maintenance_transaction_ms=100),
        monotonic=lambda: next(clock_values, 1.0),
    )

    with pytest.raises(MaintenanceDeadlineExceeded, match="deadline"):
        manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_default_100ms_retention_pass_makes_progress_on_expired_rows(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
            "resolved_at_us, details_json) VALUES (?, 'run-1', 'fixture', 'info', "
            "'resolved', 1, 2, '{}')",
            ((f"expired-{index}",) for index in range(500)),
        )

    result = RetentionManager(
        database,
        RetentionPolicy(resolved_diagnostic_us=10),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert result.expired_rows_deleted == 500


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


def test_short_evidence_expiry_preserves_long_provenance_ids_and_current_projection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    stored = OperationalRepository(database).put_idea_output(_output())
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO idea_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("instance-1", "event-1", "{}", "0" * 64, 30, 30, 0),
        )
        connection.execute(
            "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
            "event_at_us, received_at_us, event_kind, quality_bits) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-1", "instrument-1", "trades", "event-1", 20, 20, "tick", 0),
        )
    manager = RetentionManager(
        database,
        RetentionPolicy(raw_market_event_us=10, idea_shadow_us=1_000),
    )

    manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT 1 FROM market_events WHERE event_id = 'event-1'").fetchone()
            is not None
        )
        connection.execute("DELETE FROM market_latest WHERE event_id = 'event-1'")
    manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT 1 FROM market_events WHERE event_id = 'event-1'").fetchone()
            is None
        )
        output = connection.execute(
            "SELECT first_input_event_id FROM idea_outputs WHERE output_id = ?",
            (stored.output_id,),
        ).fetchone()
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id = 'instance-1'"
        ).fetchone()
    assert output["first_input_event_id"] == "event-1"
    assert checkpoint["last_market_event_id"] == "event-1"


def test_inbox_tombstone_expires_while_its_market_event_remains(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_inbox SET lifecycle = 'acknowledged', payload_json = NULL, "
            "normalized_event_id = 'event-1', acknowledged_at_us = 1, "
            "receipt_batch_id = 'tombstone-receipt' WHERE source_sequence = 1"
        )
    _insert_receipt(
        database,
        batch_id="tombstone-receipt",
        first_sequence=1,
        last_sequence=1,
        created_at_us=99,
    )

    result = RetentionManager(
        database,
        RetentionPolicy(tombstone_us=10, raw_market_event_us=1_000, receipt_us=1_000),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert result.expired_rows_deleted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT 1 FROM callback_inbox WHERE source_sequence = 1").fetchone()
            is None
        )
        assert (
            connection.execute("SELECT 1 FROM market_events WHERE event_id = 'event-1'").fetchone()
            is not None
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
        "market_events_retention_idx": (
            "SELECT event_id FROM market_events WHERE event_at_us <= ? "
            "ORDER BY event_at_us, event_kind, event_id LIMIT ?",
            (100, 10_000),
        ),
        "idea_outputs_retention_idx": (
            "SELECT output_id FROM idea_outputs WHERE emitted_at_us <= ? "
            "ORDER BY emitted_at_us, output_id LIMIT ?",
            (100, 10_000),
        ),
        "subscriptions_retention_idx": (
            "SELECT subscription_id FROM subscriptions WHERE closed_at_us IS NOT NULL "
            "AND closed_at_us <= ? ORDER BY closed_at_us, subscription_id LIMIT ?",
            (100, 10_000),
        ),
        "incidents_retention_idx": (
            "SELECT incident_id FROM incidents WHERE resolved_at_us IS NOT NULL "
            "AND resolved_at_us <= ? ORDER BY resolved_at_us, incident_id LIMIT ?",
            (100, 10_000),
        ),
        "gaps_retention_idx": (
            "SELECT gap_id FROM gaps WHERE resolved_at_us IS NOT NULL "
            "AND resolved_at_us <= ? ORDER BY resolved_at_us, gap_id LIMIT ?",
            (100, 10_000),
        ),
        "callback_inbox_terminal_idx": (
            ACK_PAYLOAD_CANDIDATES_SQL,
            (100, 10_000),
        ),
        "callback_inbox_ack_tombstone_idx": (
            ACK_TOMBSTONE_CANDIDATES_SQL,
            (100, 10_000),
        ),
        "callback_inbox_failed_payload_idx": (
            FAILED_PAYLOAD_CANDIDATES_SQL,
            (100, 10_000),
        ),
        "callback_inbox_failed_tombstone_idx": (
            FAILED_TOMBSTONE_CANDIDATES_SQL,
            (100, 10_000),
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


def test_init_cli_reports_sqlite_failures_as_machine_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_database: Path) -> None:
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("stocker_runtime.cli.initialize_database", fail)

    result = CliRunner().invoke(runtime_app, ["init", str(tmp_path / "v2.sqlite3")])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "error": "OperationalError",
        "message": "simulated database failure",
        "status": "error",
    }
