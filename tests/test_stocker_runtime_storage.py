from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocker_runtime import ProposedTradeLeg
from stocker_runtime.cli import app as runtime_app
from stocker_runtime.ingestion.inbox import (
    CALLBACK_GAP_RECOVERY_SQL,
    CALLBACK_INCIDENT_RECOVERY_SQL,
)
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
    IDEA_OUTPUT_RETENTION_CANDIDATES_SQL,
    MAX_IDEA_OUTPUT_CASCADE_ROWS,
    MAX_IDEA_OUTPUT_PARENTS_PER_PASS,
    MAX_SHADOW_POSITION_CASCADE_ROWS,
    MAX_STORED_IDEA_OUTPUT_LEGS,
    SHADOW_POSITION_RETENTION_CANDIDATES_SQL,
)


def _proposal_legs() -> tuple[ProposedTradeLeg, ...]:
    return (
        ProposedTradeLeg(
            instrument_id="instrument-1",
            action="buy",
            target="long",
            quantity_value=1.0,
            currency="USD",
        ),
    )


def test_initialize_database_creates_exact_immediate_schema_and_writer_pragmas(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"

    result = initialize_database(database, applied_at_us=1_700_000_000_000_000)

    assert result.applied_versions == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
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


def test_instrument_alias_migration_preserves_evidence_and_rejects_physical_mismatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "instrument-alias.sqlite3"
    migration_root = tmp_path / "schema-13-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:13]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, git_commit, "
            "data_class, status) VALUES "
            "('legacy-run', 'prospective_record', 'ibkr', 1, ?, 'legacy', "
            "'prospective_protected', 'stopped')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us, "
            "ended_at_us, clean_stop) VALUES ('legacy-run', 0, 'legacy-import', 1, 2, 1)"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES "
            "('legacy-instrument-aal', ?, 123, 'stock', 'AAL', 'SMART', 'USD')",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us, closed_at_us) VALUES "
            "('legacy-subscription', 'legacy-run', 0, 1, 'legacy-instrument-aal', "
            "'bars', 3, 'closed', ?, 1, 2)",
            ("c" * 64,),
        )
        legacy_instrument_before = tuple(
            connection.execute(
                "SELECT * FROM instruments WHERE instrument_id='legacy-instrument-aal'"
            ).fetchone()
        )
        legacy_subscription_before = tuple(
            connection.execute(
                "SELECT * FROM subscriptions WHERE subscription_id='legacy-subscription'"
            ).fetchone()
        )

    result = migrate_database(database, applied_at_us=2)

    assert result.applied_versions == (14,)
    with connect_v2(database) as connection:
        index = next(
            row
            for row in connection.execute("PRAGMA index_list(instruments)")
            if row["name"] == "instruments_ibkr_con_id_idx"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES ('AAL', ?, 123, 'stock', 'AAL', 'SMART', 'USD')",
            ("d" * 64,),
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="instrument_ibkr_physical_identity_mismatch",
        ):
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, "
                "symbol, exchange, currency) VALUES "
                "('wrong-aal', ?, 123, 'stock', 'WRONG', 'SMART', 'USD')",
                ("e" * 64,),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="instrument_ibkr_physical_identity_mismatch",
        ):
            connection.execute("UPDATE instruments SET currency='EUR' WHERE instrument_id='AAL'")
        aliases = tuple(
            connection.execute(
                "SELECT instrument_id FROM instruments WHERE ibkr_con_id=123 ORDER BY instrument_id"
            )
        )
        legacy_reference = connection.execute(
            "SELECT instrument_id FROM subscriptions WHERE subscription_id='legacy-subscription'"
        ).fetchone()[0]
        legacy_instrument_after = tuple(
            connection.execute(
                "SELECT * FROM instruments WHERE instrument_id='legacy-instrument-aal'"
            ).fetchone()
        )
        legacy_subscription_after = tuple(
            connection.execute(
                "SELECT * FROM subscriptions WHERE subscription_id='legacy-subscription'"
            ).fetchone()
        )
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    assert index["unique"] == 0
    assert [row["instrument_id"] for row in aliases] == ["AAL", "legacy-instrument-aal"]
    assert legacy_reference == "legacy-instrument-aal"
    assert legacy_instrument_after == legacy_instrument_before
    assert legacy_subscription_after == legacy_subscription_before
    assert foreign_keys == ()
    assert quick_check == "ok"


def test_instrument_alias_migration_compares_option_strikes_numerically(
    tmp_path: Path,
) -> None:
    database = tmp_path / "option-alias.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, "
            "option_multiplier) VALUES "
            "('legacy-option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
            "'20260814', '15.0', 'call', '100')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, "
            "option_multiplier) VALUES "
            "('option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
            "'20260814', '15', 'call', '100')",
            ("b" * 64,),
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="instrument_ibkr_physical_identity_mismatch",
        ):
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, "
                "symbol, exchange, currency, option_expiry, option_strike, option_right, "
                "option_multiplier) VALUES "
                "('wrong-option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
                "'20260814', '15.5', 'call', '100')",
                ("c" * 64,),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="instrument_ibkr_physical_identity_mismatch",
        ):
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, "
                "symbol, exchange, currency, option_expiry, option_strike, option_right, "
                "option_multiplier) VALUES "
                "('invalid-option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
                "'20260814', '15oops', 'call', '100')",
                ("d" * 64,),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="instrument_ibkr_physical_identity_mismatch",
        ):
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, "
                "symbol, exchange, currency, option_expiry, option_strike, option_right, "
                "option_multiplier) VALUES "
                "('precision-option-aal', ?, 321, 'option', 'AAL', 'SMART', 'USD', "
                "'20260814', '15.0000000000000000001', 'call', '100')",
                ("e" * 64,),
            )

        aliases = tuple(
            connection.execute(
                "SELECT instrument_id, option_strike FROM instruments WHERE ibkr_con_id=321 "
                "ORDER BY instrument_id"
            )
        )

    assert [tuple(row) for row in aliases] == [
        ("legacy-option-aal", "15.0"),
        ("option-aal", "15"),
    ]


def test_shadow_runtime_keeps_only_authoritative_schedule_and_quote_indexes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    initialize_database(database)

    with connect_v2(database) as connection:
        indexes = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='index'")
        }
        schedule_columns = tuple(
            str(row[2])
            for row in connection.execute("PRAGMA index_info(shadow_schedule_run_count_idx)")
        )
        quote_columns = tuple(
            str(row[2])
            for row in connection.execute("PRAGMA index_info(market_events_shadow_raw_idx)")
        )

    assert "shadow_progress_schedule_idx" not in indexes
    assert "market_events_shadow_scan_idx" not in indexes
    assert schedule_columns == ("run_id", "schedule_count", "output_id")
    assert quote_columns == (
        "run_id",
        "instrument_id",
        "event_kind",
        "source_sequence",
        "event_id",
    )


def test_phase2_migration_preserves_dynamic_rows_and_admits_only_causal_derived_receipts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "phase2-derived-events.sqlite3"
    migration_root = tmp_path / "phase1-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:12]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        _insert_dynamic_interest(connection)

    result = migrate_database(database, applied_at_us=2)

    assert result.applied_versions == (13, 14)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT interest_key FROM market_data_interests WHERE interest_id='interest-1'"
            ).fetchone()[0]
            == "key-interest-1"
        )
        connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES "
            "('callback-2', 'run-1', 1, 1, 'bar', 20, '{}', ?, 'pending')",
            ("3" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('bar-1', 'run-1', 2, 'instrument-1', 'bars', 'bar', 10, 20, 1, '{}', ?)",
            ("4" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, payload_json, "
                "payload_sha256) VALUES ('unauthorised-derived', 'run-1', NULL, 2, "
                "'instrument-1', 'bars', 'idea_specific_receipt', 20, 20, 1, '{}', ?)",
                ("0" * 64,),
            )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
            "event_at_us, received_at_us, connection_generation, payload_json, "
            "payload_sha256) VALUES ('bar-5m-1', 'run-1', NULL, 2, 'instrument-1', "
            "'bars', 'bar_5m', 20, 20, 1, '{}', ?)",
            ("5" * 64,),
        )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) "
            "VALUES ('bar-5m-1', 'bar-1', 0, 'constituent', 20)"
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
            "event_at_us, received_at_us, connection_generation, payload_json, "
            "payload_sha256) VALUES ('prefix-1', 'run-1', NULL, 2, 'instrument-1', "
            "'bars', 'bar_5m_session_prefix', 20, 20, 1, '{}', ?)",
            ("1" * 64,),
        )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) "
            "VALUES ('prefix-1', 'bar-5m-1', 0, 'constituent', 20)"
        )
        with pytest.raises(sqlite3.IntegrityError, match="callback_provenance"):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "instrument_id, feed_kind, event_kind, event_at_us, received_at_us, "
                "connection_generation, payload_json, payload_sha256) VALUES "
                "('source-without-callback', 'run-1', 999, 'instrument-1', 'quotes', "
                "'quote', 20, 20, 1, '{}', ?)",
                ("2" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="derivation_provenance"):
            connection.execute(
                "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
                "input_ordinal, input_role, created_at_us) "
                "VALUES ('prefix-1', 'event-1', 1, 'completion', 19)"
            )


def test_runtime_connection_state_is_constrained_and_defaults_disconnected(tmp_path: Path) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    initialize_database(database)
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
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle) "
            "VALUES ('run-1', 1, 'recovering')"
        )
        assert (
            connection.execute("SELECT connection_state FROM runtime_state").fetchone()[0]
            == "disconnected"
        )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "UPDATE runtime_state SET connection_state='unknown' WHERE run_id='run-1'"
            )


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
            "VALUES (15, '0015_future.sql', ?, 2)",
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


def test_shadow_policy_migration_backfills_one_binding_and_rejects_conflicting_history(
    tmp_path: Path,
) -> None:
    def initialize_v6(database: Path, migration_root: Path) -> None:
        migration_root.mkdir()
        for migration in migration_plan()[:6]:
            (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
        initialize_database(database, migration_root=migration_root, applied_at_us=1)
        _seed_output_dependencies(database, verify_schema=False)

    def insert_position(
        database: Path,
        *,
        suffix: str,
        ordinal: int,
        policy_json: str,
        policy_hash: str,
    ) -> None:
        with connect_v2(database, verify_schema=False) as connection:
            output_id = f"proposal-{suffix}"
            position_id = f"position-{suffix}"
            connection.execute(
                "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
                "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
                "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
                "payload_json, payload_hash, content_hash, data_class, authority_status) "
                "VALUES (?, 'run-1', 'instance-1', 'proposed_trade', 'instrument-1', 30, 25, "
                "'event-1', 'event-1', 'event-1', ?, ?, '{}', ?, ?, "
                "'shadow_protected', 'unapproved')",
                (output_id, "3" * 64, ordinal, "4" * 64, suffix * 64),
            )
            connection.execute(
                "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
                "VALUES (?, 'event-1', 0)",
                (output_id,),
            )
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, quantity_value, currency) VALUES "
                "(?, 0, 'instrument-1', 'buy', 'long', 1, 'USD')",
                (output_id,),
            )
            connection.execute(
                "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
                "instance_id, lifecycle, cost_model_id, fill_model_id, currency, data_class, "
                "policy_json, policy_hash) VALUES (?, ?, 'run-1', 'instance-1', 'pending', "
                "'cost', 'fill', 'USD', 'shadow_protected', ?, ?)",
                (position_id, output_id, policy_json, policy_hash),
            )
            connection.execute(
                "INSERT INTO shadow_progress(position_id, entry_after_source_sequence, "
                "next_source_sequence, next_horizon_index, updated_at_us, "
                "pending_retention_deadline_us) VALUES (?, 1, 2, 0, 30, 100)",
                (position_id,),
            )

    backfill_database = tmp_path / "backfill-v6.sqlite3"
    initialize_v6(backfill_database, tmp_path / "backfill-migrations")
    insert_position(
        backfill_database,
        suffix="a",
        ordinal=0,
        policy_json='{"id":"a"}',
        policy_hash="a" * 64,
    )

    result = migrate_database(backfill_database, applied_at_us=2)

    assert result.applied_versions == (7, 8, 9, 10, 11, 12, 13, 14)
    with connect_v2(backfill_database) as connection:
        assert tuple(
            connection.execute(
                "SELECT run_id, policy_hash, policy_json FROM shadow_run_policies"
            ).fetchone()
        ) == ("run-1", "a" * 64, '{"id":"a"}')
        assert (
            connection.execute("SELECT pending_evidence_drained FROM shadow_progress").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT pending_expiry_active FROM shadow_progress").fetchone()[0]
            == 1
        )

    conflict_database = tmp_path / "conflict-v6.sqlite3"
    initialize_v6(conflict_database, tmp_path / "conflict-migrations")
    insert_position(
        conflict_database,
        suffix="a",
        ordinal=0,
        policy_json='{"id":"a"}',
        policy_hash="a" * 64,
    )
    insert_position(
        conflict_database,
        suffix="b",
        ordinal=1,
        policy_json='{"id":"b"}',
        policy_hash="b" * 64,
    )

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        migrate_database(conflict_database, applied_at_us=2)
    with connect_v2(conflict_database, verify_schema=False) as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 6
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='shadow_run_policies'"
            ).fetchone()
            is None
        )
        progress_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(shadow_progress)")
        }
        assert "pending_evidence_drained" not in progress_columns


def test_expiry_projection_migration_deactivates_terminal_history(tmp_path: Path) -> None:
    migration_root = tmp_path / "v7-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:7]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    database = tmp_path / "terminal-v7.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('terminal-proposal', 'run-1', 'instance-1', 'proposed_trade', "
            "'instrument-1', 30, 25, 'event-1', 'event-1', 'event-1', ?, 0, '{}', ?, ?, "
            "'shadow_protected', 'unapproved')",
            ("3" * 64, "4" * 64, "5" * 64),
        )
        connection.execute(
            "INSERT INTO idea_output_inputs VALUES ('terminal-proposal', 'event-1', 0)"
        )
        connection.execute(
            "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, target, "
            "quantity_value, currency) VALUES "
            "('terminal-proposal', 0, 'instrument-1', 'buy', 'long', 1, 'USD')"
        )
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, closed_at_us, lifecycle, cost_model_id, fill_model_id, currency, "
            "invalid_reason, data_class, policy_json, policy_hash) VALUES "
            "('terminal-position', 'terminal-proposal', 'run-1', 'instance-1', 40, 'invalid', "
            "'cost', 'fill', 'USD', 'entry_evidence_expired', 'shadow_protected', '{}', ?)",
            ("6" * 64,),
        )
        connection.execute(
            "INSERT INTO shadow_progress(position_id, entry_after_source_sequence, "
            "next_source_sequence, next_horizon_index, updated_at_us, "
            "pending_retention_deadline_us, pending_evidence_drained) "
            "VALUES ('terminal-position', 1, 2, 0, 40, 30, 1)"
        )

    result = migrate_database(database, applied_at_us=2)

    assert result.applied_versions == (8, 9, 10, 11, 12, 13, 14)
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT pending_evidence_drained, pending_expiry_active FROM shadow_progress"
            ).fetchone()
        ) == (1, 0)
        assert (
            connection.execute(
                "SELECT committed_after_source_sequence FROM idea_output_commit_boundaries"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM shadow_progress "
                "INDEXED BY shadow_progress_pending_expiry_idx "
                "WHERE pending_evidence_drained=1 AND pending_expiry_active=1"
            ).fetchone()[0]
            == 0
        )
        with pytest.raises(sqlite3.IntegrityError, match="shadow_pending_expiry_active_monotonic"):
            connection.execute(
                "UPDATE shadow_progress SET pending_expiry_active=1 "
                "WHERE position_id='terminal-position'"
            )


@pytest.mark.parametrize("lifecycle", ["open", "closed"])
def test_commit_boundary_migration_rejects_completed_shadow_artifacts_atomically(
    tmp_path: Path, lifecycle: str
) -> None:
    migration_root = tmp_path / f"v8-{lifecycle}-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:8]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    database = tmp_path / f"v8-{lifecycle}.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('historical-proposal', 'run-1', 'instance-1', 'proposed_trade', "
            "'instrument-1', 30, 25, 'event-1', 'event-1', 'event-1', ?, 0, '{}', ?, ?, "
            "'shadow_protected', 'unapproved')",
            ("3" * 64, "4" * 64, "5" * 64),
        )
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, opened_at_us, closed_at_us, lifecycle, cost_model_id, fill_model_id, "
            "currency, data_class, policy_json, policy_hash) VALUES "
            "('historical-position', 'historical-proposal', 'run-1', 'instance-1', 31, ?, ?, "
            "'cost', 'fill', 'USD', 'shadow_protected', '{}', ?)",
            (40 if lifecycle == "closed" else None, lifecycle, "6" * 64),
        )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="idea_output_commit_boundary_requires_shadow_recreation",
    ):
        migrate_database(database, applied_at_us=2)
    with connect_v2(database, verify_schema=False) as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 8
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='idea_output_commit_boundaries'"
            ).fetchone()
            is None
        )


def test_commit_boundary_migration_rewinds_pending_shadow_to_conservative_run_max(
    tmp_path: Path,
) -> None:
    migration_root = tmp_path / "v8-pending-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:8]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    database = tmp_path / "v8-pending.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
            "VALUES (2, 'precommit-quote', 'run-1', 1, 1, 'quote', 21, ?, 'pending')",
            ("7" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "bid_value, ask_value, payload_json, payload_sha256) VALUES "
            "('precommit-quote', 'run-1', 2, 'instrument-1', 'quotes', 'quote', 21, 21, 1, "
            "100, 101, '{}', ?)",
            ("8" * 64,),
        )
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('pending-proposal', 'run-1', 'instance-1', 'proposed_trade', "
            "'instrument-1', 30, 20, 'event-1', 'event-1', 'event-1', ?, 0, '{}', ?, ?, "
            "'shadow_protected', 'unapproved')",
            ("3" * 64, "4" * 64, "5" * 64),
        )
        connection.execute(
            "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, target, "
            "quantity_value, currency) VALUES "
            "('pending-proposal', 0, 'instrument-1', 'buy', 'long', 1, 'USD')"
        )
        connection.execute(
            "INSERT INTO idea_output_inputs VALUES ('pending-proposal', 'event-1', 0)"
        )
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, lifecycle, cost_model_id, fill_model_id, currency, data_class, "
            "policy_json, policy_hash) VALUES ('pending-position', 'pending-proposal', 'run-1', "
            "'instance-1', 'pending', 'cost', 'fill', 'USD', 'shadow_protected', '{}', ?)",
            ("6" * 64,),
        )
        connection.execute(
            "INSERT INTO shadow_progress(position_id, entry_after_source_sequence, "
            "next_source_sequence, next_horizon_index, updated_at_us, "
            "pending_retention_deadline_us, pending_evidence_drained, pending_expiry_active) "
            "VALUES ('pending-position', 1, 3, 0, 30, 100, 0, 1)"
        )
        connection.execute(
            "INSERT INTO shadow_quote_state(position_id, leg_number, instrument_id, "
            "bid_event_id, bid_source_sequence, bid_at_us, bid_value, ask_event_id, "
            "ask_source_sequence, ask_at_us, ask_value) VALUES "
            "('pending-position', 0, 'instrument-1', 'precommit-quote', 2, 21, 100, "
            "'precommit-quote', 2, 21, 101)"
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
            "VALUES (3, 'durable-unprojected', 'run-1', 1, 1, 'quote', 22, ?, 'pending')",
            ("9" * 64,),
        )

    assert migrate_database(database, applied_at_us=2).applied_versions == (
        9,
        10,
        11,
        12,
        13,
        14,
    )
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT committed_after_source_sequence FROM idea_output_commit_boundaries "
                "WHERE output_id='pending-proposal'"
            ).fetchone()[0]
            == 3
        )
        assert tuple(
            connection.execute(
                "SELECT entry_after_source_sequence, next_source_sequence "
                "FROM shadow_progress WHERE position_id='pending-position'"
            ).fetchone()
        ) == (3, 4)
        assert tuple(
            connection.execute(
                "SELECT bid_event_id, bid_source_sequence, bid_value, "
                "ask_event_id, ask_source_sequence, ask_value FROM shadow_quote_state "
                "WHERE position_id='pending-position'"
            ).fetchone()
        ) == (None, None, None, None, None, None)


def test_output_seal_migration_rejects_incomplete_existing_output_atomically(
    tmp_path: Path,
) -> None:
    migration_root = tmp_path / "v9-incomplete-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:9]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    database = tmp_path / "v9-incomplete.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('incomplete-output', 'run-1', 'instance-1', 'signal', 'instrument-1', "
            "30, 25, 'event-1', 'event-1', 'event-1', ?, 0, '{}', ?, ?, "
            "'shadow_protected', 'recorded')",
            ("3" * 64, "4" * 64, "5" * 64),
        )

    with pytest.raises(sqlite3.IntegrityError, match="idea_output_seal_incomplete"):
        migrate_database(database, applied_at_us=2)
    with connect_v2(database, verify_schema=False) as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 9
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='idea_output_seals'"
            ).fetchone()
            is None
        )


def test_output_seal_migration_rejects_present_cross_run_child_atomically(
    tmp_path: Path,
) -> None:
    migration_root = tmp_path / "v9-cross-run-migrations"
    migration_root.mkdir()
    for migration in migration_plan()[:9]:
        (migration_root / migration.name).write_text(migration.sql, encoding="utf-8")
    database = tmp_path / "v9-cross-run.sqlite3"
    initialize_database(database, migration_root=migration_root, applied_at_us=1)
    _seed_output_dependencies(database, verify_schema=False)
    with connect_v2(database, verify_schema=False) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES (?, ?, ?, 1, 1, 'tick', 21, ?, 'pending')",
            (
                (2, "cross-run-event", "retention-run", "6" * 64),
                (3, "event-3", "run-1", "7" * 64),
                (4, "event-4", "run-1", "8" * 64),
            ),
        )
        connection.executemany(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "(?, ?, ?, 'instrument-1', 'trades', 'tick', 21, 21, 1, '{}', ?)",
            (
                ("cross-run-event", "retention-run", 2, "6" * 64),
                ("event-3", "run-1", 3, "7" * 64),
                ("event-4", "run-1", 4, "8" * 64),
            ),
        )
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('cross-run-output', 'run-1', 'instance-1', 'signal', 'instrument-1', "
            "30, 25, 'event-1', 'event-3', 'event-3', ?, 0, '{}', ?, ?, "
            "'shadow_protected', 'recorded')",
            ("3" * 64, "4" * 64, "5" * 64),
        )
        connection.executemany(
            "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
            "VALUES ('cross-run-output', ?, ?)",
            (("event-1", 0), ("event-4", 1), ("event-3", 2)),
        )
        connection.execute(
            "UPDATE idea_output_inputs SET event_id='cross-run-event' "
            "WHERE output_id='cross-run-output' AND input_ordinal=1"
        )

    with pytest.raises(sqlite3.IntegrityError, match="idea_output_seal_incomplete"):
        migrate_database(database, applied_at_us=2)
    with connect_v2(database, verify_schema=False) as connection:
        assert connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 9
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='idea_output_seals'"
            ).fetchone()
            is None
        )


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


def test_failed_initialization_removes_partial_database_and_sidecars(tmp_path: Path) -> None:
    migration_root = tmp_path / "migrations"
    migration_root.mkdir()
    first = migration_plan()[0]
    (migration_root / first.name).write_text(first.sql, encoding="utf-8")
    (migration_root / "0002_broken.sql").write_text(
        "CREATE TABLE should_rollback(value TEXT) STRICT;\nTHIS IS NOT SQL;\n",
        encoding="utf-8",
    )
    database = tmp_path / "v2.sqlite3"

    with pytest.raises(sqlite3.OperationalError):
        initialize_database(database, migration_root=migration_root, applied_at_us=1)

    assert not database.exists()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()


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
    stored = OperationalRepository(database).put_idea_output(_output())
    proposal = OperationalRepository(database).put_idea_output(
        IdeaOutputRecord(
            **{
                **_output().__dict__,
                "output_kind": "proposed_trade",
                "authority_status": "unapproved",
                "legs": _proposal_legs(),
            }
        )
    )
    _insert_receipt(
        database,
        batch_id="json-receipt",
        first_sequence=1,
        last_sequence=1,
        created_at_us=30,
        run_id="run-1",
    )
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
            "details_json) VALUES ('json-incident', 'run-1', 'fixture', 'info', "
            "'json', 30, '{}')"
        )
        connection.execute(
            "INSERT INTO idea_checkpoints VALUES "
            "('instance-1', 'event-1', 1, '{}', ?, '[]', 30, 30, 0)",
            ("7" * 64,),
        )
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, lifecycle, cost_model_id, fill_model_id, currency, data_class) "
            "VALUES ('json-position', ?, 'run-1', 'instance-1', 'closed', 'cost', 'fill', "
            "'USD', 'shadow_protected')",
            (proposal.output_id,),
        )
        connection.execute(
            "INSERT INTO shadow_marks(position_id, marked_at_us, payload_json) "
            "VALUES ('json-position', 31, '{}')"
        )
        connection.execute(
            "INSERT INTO shadow_outcomes(position_id, outcome_at_us, reason, completeness, "
            "payload_json) VALUES ('json-position', 32, 'fixture', 'complete', '{}')"
        )
        statements = (
            "UPDATE incidents SET details_json = '{ \"a\": 1 }' "
            "WHERE incident_id = 'json-incident'",
            "UPDATE callback_inbox SET payload_json = '{ \"a\": 1 }' WHERE source_sequence = 1",
            "UPDATE callback_receipts SET kind_counts_json = '{ \"tick\": 1 }' "
            "WHERE batch_id = 'json-receipt'",
            "UPDATE callback_receipts SET status_counts_json = '{ \"pending\": 1 }' "
            "WHERE batch_id = 'json-receipt'",
            "UPDATE idea_plugins SET manifest_json = '{ \"a\": 1 }' WHERE idea_id = 'idea'",
            'UPDATE idea_instances SET parameters_json = \'{"z":1,"a":2}\' '
            "WHERE instance_id = 'instance-1'",
            "UPDATE idea_checkpoints SET state_json = '{ \"a\": 1 }' "
            "WHERE instance_id = 'instance-1'",
            "UPDATE shadow_marks SET payload_json = '{ \"a\": 1 }' "
            "WHERE position_id = 'json-position'",
            "UPDATE shadow_outcomes SET payload_json = '{ \"a\": 1 }' "
            "WHERE position_id = 'json-position'",
        )
        for statement in statements:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(statement)
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_immutable"):
            connection.execute(
                "UPDATE idea_outputs SET payload_json = '{\"a\":1}' WHERE output_id=?",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="market_event_immutable"):
            connection.execute(
                'UPDATE market_events SET payload_json = \'{"z":1,"a":2}\' '
                "WHERE event_id = 'event-1'"
            )


def test_schema_rejects_impossible_gap_and_callback_terminal_times(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO gaps(gap_id, run_id, started_at_us, ended_at_us, reason, "
                "data_loss_possible, continuity_required) "
                "VALUES ('backward-end', 'run-1', 100, 99, 'fixture', 1, 1)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO gaps(gap_id, run_id, started_at_us, reason, "
                "data_loss_possible, continuity_required, resolved_at_us) "
                "VALUES ('missing-end', 'run-1', 100, 'fixture', 1, 1, 101)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO gaps(gap_id, run_id, started_at_us, ended_at_us, reason, "
                "data_loss_possible, continuity_required, resolved_at_us) "
                "VALUES ('backward-resolution', 'run-1', 80, 90, 'fixture', 1, 1, 89)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "UPDATE callback_inbox SET lifecycle='acknowledged', "
                "normalized_event_id='event-1', acknowledged_at_us=19 "
                "WHERE source_sequence=1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "opened_at_us, resolved_at_us, details_json) VALUES "
                "('backward-incident', 'run-1', 'fixture', 'degraded', 'fixture', "
                "30, 29, '{}')"
            )


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
            "('instance-1', 'event-1', 1, '{}', ?, '[]', 20, 20, 0)",
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
            "legs": _proposal_legs(),
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
        connection.execute("DELETE FROM market_events WHERE event_id = 'event-1'")
        cursor = connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES "
            "('exit-callback', 'run-1', 1, 1, 'tick', 40, '{}', ?, 'pending')",
            ("e" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('exit-event', 'run-1', ?, 'instrument-1', 'trades', 'tick', 40, 40, 1, '{}', ?)",
            (int(cursor.lastrowid), "f" * 64),
        )
        with pytest.raises(sqlite3.IntegrityError, match="exit_event_provenance"):
            connection.execute(
                "UPDATE shadow_legs SET exit_market_event_id = 'missing', exit_price = 11 "
                "WHERE position_id = 'position-1' AND leg_number = 0"
            )
        connection.execute(
            "UPDATE shadow_legs SET exit_market_event_id = 'exit-event', exit_price = 11 "
            "WHERE position_id = 'position-1' AND leg_number = 0"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) VALUES ('instrument-2', ?, 'stock', 'ABC', 'SMART', 'USD')",
            ("9" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="event_provenance"):
            connection.execute(
                "UPDATE shadow_legs SET instrument_id = 'instrument-2' "
                "WHERE position_id = 'position-1' AND leg_number = 0"
            )
        with pytest.raises(sqlite3.IntegrityError, match="event_provenance"):
            connection.execute(
                "UPDATE shadow_legs SET position_id = 'missing-position' "
                "WHERE position_id = 'position-1' AND leg_number = 0"
            )


def test_canonical_json_admission_is_bounded_and_deterministic() -> None:
    assert canonical_json_text({"z": 1, "a": [True, None]}, max_bytes=64) == (
        '{"a":[true,null],"z":1}'
    )
    with pytest.raises(JsonAdmissionError, match="exceeds 8 bytes"):
        canonical_json_text({"payload": "large"}, max_bytes=8)


def _seed_output_dependencies(database: Path, *, verify_schema: bool = True) -> None:
    with connect_v2(database, verify_schema=verify_schema) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
            ("run-1", "shadow", "ibkr", 10, "a" * 64, "deadbee", "shadow_protected", "created"),
        )
        connection.execute(
            "INSERT INTO recorder_generations VALUES (?, ?, ?, ?, NULL, 0, NULL)",
            ("run-1", 1, "fixture", 10),
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)",
            (
                "retention-run",
                "shadow",
                "ibkr",
                10,
                "9" * 64,
                "deadbee",
                "shadow_protected",
                "created",
            ),
        )
        connection.execute(
            "INSERT INTO recorder_generations VALUES (?, ?, ?, ?, NULL, 0, NULL)",
            ("retention-run", 1, "retention-fixture", 10),
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
            "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, universe_json, "
            "universe_hash, requirements_json, requirements_hash, activated_after_source_sequence, "
            "activated_at_us, health, data_class) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "instance-1",
                "idea",
                "1",
                "run-1",
                "shadow",
                "{}",
                "0" * 64,
                "f" * 64,
                "e" * 64,
                '["instrument-1"]',
                "1" * 64,
                "[]",
                "2" * 64,
                0,
                22,
                "healthy",
                "shadow_protected",
            ),
        )


def _insert_dynamic_interest(
    connection: sqlite3.Connection,
    *,
    interest_id: str = "interest-1",
    lifecycle: str = "pending",
    updated_at_us: int = 20,
    underlying_instrument_id: str = "instrument-1",
    interest_key: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO market_data_interests(interest_id, run_id, instance_id, interest_key, "
        "underlying_instrument_id, asset_kind, minimum_days_to_expiry, "
        "maximum_days_to_expiry, option_right, strike_offset, reference_price, feed_kind, "
        "cadence, as_of_at_us, expires_at_us, required, priority, maximum_contracts, "
        "input_event_id, content_hash, lifecycle, next_attempt_at_us, created_at_us, "
        "updated_at_us) VALUES (?, 'run-1', 'instance-1', ?, ?, 'option', "
        "1, 1, 'call', 0, 100.0, 'quotes', 'snapshot', 20, 100, 1, 100, 1, "
        "'event-1', ?, ?, 20, 20, ?)",
        (
            interest_id,
            interest_key or f"key-{interest_id}",
            underlying_instrument_id,
            hashlib.sha256(interest_id.encode()).hexdigest(),
            lifecycle,
            updated_at_us,
        ),
    )


def test_dynamic_interest_schema_enforces_scope_identity_and_exact_receipt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dynamic-schema.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        _insert_dynamic_interest(connection)
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, option_multiplier) "
            "VALUES ('option-1', ?, 9001, 'option', 'XYZ', 'SMART', 'USD', '20260811', "
            "'100', 'call', '100')",
            ("8" * 64,),
        )
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, instrument_id, expiry, strike, option_right, multiplier, "
            "candidates_inspected, completed_at_us) VALUES ('receipt-1', 'interest-1', "
            "'run-1', 'instance-1', 'resolved', 'option-1', '20260811', 100, 'call', "
            "'100', 1, 21)"
        )
        connection.execute(
            "UPDATE market_data_interests SET lifecycle='resolved', updated_at_us=21 "
            "WHERE interest_id='interest-1'"
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, "
            "connection_state, connection_generation) VALUES "
            "('run-1', 1, 'running', 'connected', 1)"
        )
        assert (
            connection.execute(
                "SELECT dynamic_request_high_water FROM runtime_state WHERE run_id='run-1'"
            ).fetchone()[0]
            == 1_999_999
        )
        connection.execute(
            "UPDATE runtime_state SET dynamic_request_high_water=2000000 WHERE run_id='run-1'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="high_water_monotonic"):
            connection.execute(
                "UPDATE runtime_state SET dynamic_request_high_water=1999999 WHERE run_id='run-1'"
            )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES ('option-subscription', 'run-1', 1, 1, "
            "'option-1', 'quotes', 20, 'connecting', ?, 21)",
            ("6" * 64,),
        )
        connection.execute(
            "UPDATE market_data_interests SET bound_subscription_id='option-subscription' "
            "WHERE interest_id='interest-1'"
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES ('wrong-subscription', 'run-1', 1, 1, "
            "'instrument-1', 'quotes', 21, 'connecting', ?, 21)",
            ("7" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="subscription_binding"):
            connection.execute(
                "UPDATE market_data_interests SET bound_subscription_id='wrong-subscription' "
                "WHERE interest_id='interest-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE instrument_discovery_receipts SET strike=101 WHERE receipt_id='receipt-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="lifecycle"):
            connection.execute(
                "UPDATE market_data_interests SET lifecycle='pending' "
                "WHERE interest_id='interest-1'"
            )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) VALUES ('outside', ?, 'stock', 'OUT', 'SMART', 'USD')",
            ("7" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="provenance"):
            _insert_dynamic_interest(
                connection,
                interest_id="interest-outside",
                underlying_instrument_id="outside",
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_dynamic_interest(
                connection,
                interest_id="interest-long-key",
                interest_key="x" * 129,
            )
        _insert_dynamic_interest(connection, interest_id="interest-long-reason")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
                "instance_id, status, reason_code, candidates_inspected, completed_at_us) "
                "VALUES ('receipt-long-reason', 'interest-long-reason', 'run-1', "
                "'instance-1', 'denied', ?, 0, 21)",
                ("x" * 129,),
            )


def test_dynamic_interest_query_indexes_are_exercised_by_bounded_lifecycle_plans(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dynamic-query-plans.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        _insert_dynamic_interest(connection)
        pending_plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM market_data_interests WHERE run_id=? "
                "AND lifecycle='pending' AND expires_at_us>? AND next_attempt_at_us<=? "
                "ORDER BY required DESC, priority DESC, interest_id LIMIT 4",
                ("run-1", 10, 20),
            )
        )
        receipt_plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM instrument_discovery_receipts "
                "WHERE instance_id=? ORDER BY completed_at_us DESC, receipt_id DESC LIMIT 64",
                ("instance-1",),
            )
        )
        detail_plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM market_data_interests "
                "WHERE instance_id=? ORDER BY updated_at_us DESC, interest_id LIMIT 64",
                ("instance-1",),
            )
        )
        retention_reference_plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM market_data_interests "
                "WHERE input_event_id=? LIMIT 1",
                ("event-1",),
            )
        )
    assert "market_data_interests_run_lifecycle_idx" in pending_plan
    assert "instrument_discovery_receipts_instance_time_idx" in receipt_plan
    assert "market_data_interests_instance_updated_idx" in detail_plan
    assert "market_data_interests_input_event_idx" in retention_reference_plan
    assert "SCAN market_data_interests" not in retention_reference_plan


def test_retention_counts_interest_receipt_cascade_before_releasing_input_event(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dynamic-retention.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        _insert_dynamic_interest(connection, lifecycle="denied", updated_at_us=20)
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, reason_code, candidates_inspected, completed_at_us) "
            "VALUES ('receipt-1', 'interest-1', 'run-1', 'instance-1', 'denied', "
            "'NO_MATCH', 0, 21)"
        )
    policy = RetentionPolicy(
        idea_shadow_us=50,
        raw_market_event_us=1,
        maintenance_batch_rows=2,
    )

    first = RetentionManager(database, policy).run(now_us=100)

    assert first.expired_rows_deleted == 2
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM market_data_interests").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM instrument_discovery_receipts").fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_id='event-1'"
            ).fetchone()[0]
            == 1
        )

    second = RetentionManager(database, policy).run(now_us=100)
    assert second.expired_rows_deleted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_id='event-1'"
            ).fetchone()[0]
            == 0
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


def test_repository_commit_boundary_is_first_write_stable_and_plugin_uncontrolled(
    tmp_path: Path,
) -> None:
    database = tmp_path / "repository-boundary.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        cursor = connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
            "VALUES ('callback-2', 'run-1', 1, 1, 'quote', 21, ?, 'pending')",
            ("8" * 64,),
        )
        assert cursor.lastrowid == 2
    base = _output()
    record = IdeaOutputRecord(
        **{
            **base.__dict__,
            "output_kind": "proposed_trade",
            "payload": {"committed_after_source_sequence": 0},
            "authority_status": "unapproved",
            "legs": _proposal_legs(),
        }
    )
    repository = OperationalRepository(database)

    first = repository.put_idea_output(record)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT committed_after_source_sequence FROM idea_output_commit_boundaries "
                "WHERE output_id=?",
                (first.output_id,),
            ).fetchone()[0]
            == 2
        )
        cursor = connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
            "VALUES ('callback-3', 'run-1', 1, 1, 'quote', 22, ?, 'pending')",
            ("9" * 64,),
        )
        assert cursor.lastrowid == 3

    retry = repository.put_idea_output(record)
    assert retry.inserted is False
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT committed_after_source_sequence FROM idea_output_commit_boundaries "
                "WHERE output_id=?",
                (first.output_id,),
            ).fetchone()[0]
            == 2
        )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_commit_boundary_immutable"):
            connection.execute(
                "UPDATE idea_output_commit_boundaries "
                "SET committed_after_source_sequence=0 WHERE output_id=?",
                (first.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_commit_boundary_immutable"):
            connection.execute(
                "DELETE FROM idea_output_commit_boundaries WHERE output_id=?",
                (first.output_id,),
            )
        connection.execute("DELETE FROM idea_outputs WHERE output_id=?", (first.output_id,))
        assert (
            connection.execute(
                "SELECT count(*) FROM idea_output_commit_boundaries WHERE output_id=?",
                (first.output_id,),
            ).fetchone()[0]
            == 0
        )


def test_idea_output_and_frozen_input_provenance_are_append_only(tmp_path: Path) -> None:
    database = tmp_path / "append-only-output.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(2, 'callback-2', 'run-1', 1, 1, 'tick', 21, ?, 'pending')",
            ("2" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('event-2', 'run-1', 2, 'instrument-1', 'trades', 'tick', 21, 21, 1, '{}', ?)",
            ("3" * 64,),
        )
    base = _output()
    record = IdeaOutputRecord(
        **{
            **base.__dict__,
            "output_kind": "proposed_trade",
            "last_input_event_id": "event-2",
            "input_event_ids": ("event-1", "event-2"),
            "authority_status": "unapproved",
            "legs": _proposal_legs(),
        }
    )
    stored = OperationalRepository(database).put_idea_output(record)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(3, 'callback-3', 'run-1', 1, 1, 'tick', 22, ?, 'pending')",
            ("4" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('event-3', 'run-1', 3, 'instrument-1', 'trades', 'tick', 22, 22, 1, '{}', ?)",
            ("5" * 64,),
        )
        original = tuple(
            connection.execute(
                "SELECT * FROM idea_outputs WHERE output_id=?", (stored.output_id,)
            ).fetchone()
        )
        mutations = (
            ("last_input_event_id=?", ("event-3",)),
            ("last_input_event_id=?", ("event-1",)),
            ("first_input_event_id=?", ("event-3",)),
            ("run_id=?", ("retention-run",)),
            ("input_watermark=?", ("event-3",)),
            ("output_kind=?", ("observation",)),
            ("output_ordinal=?", (1,)),
            ("payload_json=?", ('{"changed":true}',)),
            ("content_hash=?", ("6" * 64,)),
        )
        for assignment, parameters in mutations:
            with pytest.raises(sqlite3.IntegrityError, match="idea_output_immutable"):
                connection.execute(
                    f"UPDATE idea_outputs SET {assignment} WHERE output_id=?",  # noqa: S608
                    (*parameters, stored.output_id),
                )
            assert (
                tuple(
                    connection.execute(
                        "SELECT * FROM idea_outputs WHERE output_id=?", (stored.output_id,)
                    ).fetchone()
                )
                == original
            )

        with pytest.raises(sqlite3.IntegrityError, match="idea_output_input_immutable"):
            connection.execute(
                "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
                "VALUES (?, 'event-3', 2)",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_input_immutable"):
            connection.execute(
                "UPDATE idea_output_inputs SET event_id='event-3' "
                "WHERE output_id=? AND event_id='event-1'",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_input_immutable"):
            connection.execute(
                "DELETE FROM idea_output_inputs WHERE output_id=? AND event_id='event-1'",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_leg_immutable"):
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, quantity_value, currency) VALUES "
                "(?, 1, 'instrument-1', 'sell', 'short', 1, 'USD')",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_leg_immutable"):
            connection.execute(
                "UPDATE idea_output_legs SET quantity_value=2 WHERE output_id=? AND leg_number=0",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_leg_immutable"):
            connection.execute(
                "DELETE FROM idea_output_legs WHERE output_id=? AND leg_number=0",
                (stored.output_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="idea_output_seal_immutable"):
            connection.execute(
                "DELETE FROM idea_output_seals WHERE output_id=?",
                (stored.output_id,),
            )

        assert OperationalRepository(database).put_idea_output(record).inserted is False

        connection.execute("DELETE FROM idea_outputs WHERE output_id=?", (stored.output_id,))
        for table in (
            "idea_output_inputs",
            "idea_output_legs",
            "idea_output_seals",
            "idea_output_commit_boundaries",
            "shadow_schedule",
        ):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE output_id=?",  # noqa: S608
                    (stored.output_id,),
                ).fetchone()[0]
                == 0
            )


def test_sealed_repository_retry_survives_real_lineage_compaction(tmp_path: Path) -> None:
    database = tmp_path / "compacted-retry.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    repository = OperationalRepository(database)
    record = _output()
    first = repository.put_idea_output(record)

    RetentionManager(
        database,
        RetentionPolicy(raw_market_event_us=1, idea_shadow_us=10**18),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT 1 FROM market_events WHERE event_id='event-1'").fetchone()
            is None
        )

    retry = repository.put_idea_output(record)

    assert retry == type(first)(output_id=first.output_id, inserted=False)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM idea_outputs WHERE output_id=?", (first.output_id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM idea_output_seals WHERE output_id=?", (first.output_id,)
            ).fetchone()[0]
            == 1
        )


def test_commit_boundary_source_max_is_complete_and_uses_bounded_indexes(tmp_path: Path) -> None:
    database = tmp_path / "boundary-source-max.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, event_at_us, "
            "received_at_us, connection_generation, payload_json, payload_sha256) VALUES "
            "('derived-5', 'run-1', NULL, 5, 'instrument-1', 'bars', 'bar_5m', 25, 25, 1, "
            "'{}', ?)",
            ("5" * 64,),
        )
    repository = OperationalRepository(database)
    derived = repository.put_idea_output(
        IdeaOutputRecord(**{**_output().__dict__, "payload": {"stage": "derived"}})
    )
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_receipts(batch_id, run_id, first_source_sequence, "
            "last_source_sequence, callback_count, first_received_at_us, last_received_at_us, "
            "kind_counts_json, status_counts_json, callback_rows_hash, prior_chain_hash, "
            "chained_payload_hash, created_at_us) VALUES "
            "('receipt-7', 'run-1', 6, 7, 2, 26, 27, '{}', '{}', ?, ?, ?, 27)",
            ("6" * 64, "7" * 64, "8" * 64),
        )
    receipt = repository.put_idea_output(
        IdeaOutputRecord(
            **{
                **_output().__dict__,
                "output_ordinal": 1,
                "payload": {"stage": "receipt"},
            }
        )
    )
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_compaction_watermarks(run_id, compacted_through_sequence, "
            "cumulative_callback_count, first_received_at_us, last_received_at_us, "
            "rolled_receipt_chain_hash, last_receipt_chain_hash, updated_at_us) "
            "VALUES ('run-1', 9, 9, 20, 29, ?, ?, 29)",
            ("9" * 64, "a" * 64),
        )
    compacted = repository.put_idea_output(
        IdeaOutputRecord(
            **{
                **_output().__dict__,
                "output_ordinal": 2,
                "payload": {"stage": "compacted"},
            }
        )
    )

    watermark_sql = (
        "SELECT coalesce(max(sequence), 0) FROM ("
        "SELECT max(callback.source_sequence) AS sequence FROM callback_inbox callback "
        "INDEXED BY callback_inbox_run_sequence_idx WHERE callback.run_id=? UNION ALL "
        "SELECT max(coalesce(event.source_sequence, event.derived_after_source_sequence)) "
        "FROM market_events event INDEXED BY market_events_causal_sequence_idx "
        "WHERE event.run_id=? UNION ALL "
        "SELECT max(receipt.last_source_sequence) FROM callback_receipts receipt "
        "INDEXED BY callback_receipts_run_sequence_idx WHERE receipt.run_id=? UNION ALL "
        "SELECT max(watermark.compacted_through_sequence) "
        "FROM callback_compaction_watermarks watermark WHERE watermark.run_id=?)"
    )
    with connect_v2(database) as connection:
        assert tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT boundary.committed_after_source_sequence FROM idea_outputs output "
                "JOIN idea_output_commit_boundaries boundary USING(output_id) "
                "WHERE output.output_id IN (?, ?, ?) ORDER BY output.output_ordinal",
                (derived.output_id, receipt.output_id, compacted.output_id),
            )
        ) == (5, 7, 9)
        plan = tuple(
            str(row[3])
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN {watermark_sql}",
                ("run-1", "run-1", "run-1", "run-1"),
            )
        )

    assert any("callback_inbox_run_sequence_idx" in row for row in plan)
    assert any("market_events_causal_sequence_idx" in row for row in plan)
    assert any("callback_receipts_run_sequence_idx" in row for row in plan)
    assert any("sqlite_autoindex_callback_compaction_watermarks_1" in row for row in plan)
    assert all(
        "USE TEMP B-TREE" not in row
        and "SCAN callback" not in row
        and "SCAN event" not in row
        and "SCAN receipt" not in row
        and "SCAN watermark" not in row
        for row in plan
    )


def test_repository_persists_every_ordered_input_and_typed_trade_leg(tmp_path: Path) -> None:
    database = tmp_path / "full-provenance.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(event_uid, run_id, recorder_generation, "
            "connection_generation, callback_kind, received_at_us, payload_json, "
            "payload_sha256, lifecycle) VALUES "
            "('callback-2', 'run-1', 1, 1, 'tick', 21, '{}', ?, 'pending')",
            ("2" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('event-2', 'run-1', 2, 'instrument-1', 'trades', 'tick', 21, 21, 1, '{}', ?)",
            ("3" * 64,),
        )
    record = IdeaOutputRecord(
        **{
            **_output().__dict__,
            "output_kind": "proposed_trade",
            "authority_status": "unapproved",
            "last_input_event_id": "event-2",
            "input_event_ids": ("event-1", "event-2"),
            "legs": _proposal_legs(),
        }
    )
    stored = OperationalRepository(database).put_idea_output(record)
    with connect_v2(database) as connection:
        inputs = tuple(
            connection.execute(
                "SELECT event_id FROM idea_output_inputs WHERE output_id=? ORDER BY input_ordinal",
                (stored.output_id,),
            )
        )
        leg = connection.execute(
            "SELECT instrument_id, action, target, quantity_value, currency "
            "FROM idea_output_legs WHERE output_id=?",
            (stored.output_id,),
        ).fetchone()
    assert tuple(row[0] for row in inputs) == ("event-1", "event-2")
    assert tuple(leg) == ("instrument-1", "buy", "long", 1.0, "USD")
    changed_leg = ProposedTradeLeg(
        instrument_id="instrument-1",
        action="buy",
        target="long",
        quantity_value=2.0,
        currency="USD",
    )
    with pytest.raises(IdentityCollisionError, match="logical output identity"):
        OperationalRepository(database).put_idea_output(
            IdeaOutputRecord(**{**record.__dict__, "legs": (changed_leg,)})
        )


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
            "legs": _proposal_legs(),
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
            "legs": _proposal_legs(),
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
                "retention-run",
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
                    "retention-run",
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
    run_id: str = "retention-run",
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
        last_sequence=eligible,
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
        run_id="retention-run",
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
        assert (
            connection.execute(
                "SELECT payload_json FROM callback_inbox WHERE source_sequence = ?", (sequence,)
            ).fetchone()[0]
            is not None
        )


def test_receipt_rotation_rolls_permanent_watermark_before_deletion(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'retention-run'")
        for sequence in range(101, 107):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'retention-run', 1, 1, 'tick', ?, NULL, ?, "
                "'failed', 'fixture', ?)",
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
            ("retention-run",),
        ).fetchone()
    assert tuple(watermark)[:2] == (104, 4)
    assert watermark["last_receipt_chain_hash"] == second_hash
    assert watermark["rolled_receipt_chain_hash"] != second_hash


def test_receipt_rotation_continues_from_the_permanent_watermark(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'retention-run'")
        for sequence in range(101, 105):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'retention-run', 1, 1, 'tick', ?, NULL, ?, "
                "'failed', 'fixture', ?)",
                (
                    sequence,
                    f"continuation-{sequence}",
                    sequence,
                    "a" * 64,
                    "first" if sequence <= 102 else "second",
                ),
            )
    first_hash = _insert_receipt(
        database,
        batch_id="first",
        first_sequence=101,
        last_sequence=102,
        created_at_us=1,
    )
    second_hash = _insert_receipt(
        database,
        batch_id="second",
        first_sequence=103,
        last_sequence=104,
        created_at_us=99,
        prior_chain_hash=first_hash,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(receipt_us=50, max_receipts_per_run=1, tombstone_us=1_000),
    )

    first_pass = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        first_rollup_hash = str(
            connection.execute(
                "SELECT rolled_receipt_chain_hash FROM callback_compaction_watermarks "
                "WHERE run_id = 'retention-run'"
            ).fetchone()[0]
        )
    second_pass = manager.run(now_us=200, measured_database_bytes=1, measured_wal_bytes=0)

    assert first_pass.receipts_rolled == 1
    assert second_pass.receipts_rolled == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 0
        watermark = connection.execute(
            "SELECT compacted_through_sequence, cumulative_callback_count, "
            "rolled_receipt_chain_hash, last_receipt_chain_hash "
            "FROM callback_compaction_watermarks WHERE run_id = 'retention-run'"
        ).fetchone()
    assert tuple(watermark)[:2] == (104, 4)
    assert watermark["last_receipt_chain_hash"] == second_hash
    assert watermark["rolled_receipt_chain_hash"] != first_rollup_hash


def test_rolled_watermark_authorizes_later_payload_compaction(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    sequence = _seed_callback_for_retention(
        database,
        uid="watermark-payload",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="watermark-proof",
    )
    _insert_receipt(
        database,
        batch_id="watermark-proof",
        first_sequence=sequence,
        last_sequence=sequence,
        created_at_us=1,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1_000, receipt_us=10, tombstone_us=1_000),
    )

    rolled = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    compacting_manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=10, receipt_us=10, tombstone_us=1_000),
    )
    compacted = compacting_manager.run(
        now_us=100,
        measured_database_bytes=1,
        measured_wal_bytes=0,
    )

    assert rolled.receipts_rolled == 1
    assert compacted.payloads_compacted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM callback_receipts WHERE batch_id = 'watermark-proof'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT payload_json FROM callback_inbox WHERE source_sequence = ?", (sequence,)
            ).fetchone()[0]
            is None
        )


def test_malformed_receipt_rolls_back_without_deleting_any_proof(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'retention-run'")
        for sequence in (51, 52):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'retention-run', 1, 1, 'tick', ?, NULL, ?, "
                "'failed', 'fixture', 'bad')",
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
                "SELECT 1 FROM callback_compaction_watermarks WHERE run_id = 'retention-run'"
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
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'retention-run'")
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


def test_tombstone_retention_rolls_receipt_before_deleting_covered_callbacks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    first_batch_sequences = tuple(
        _seed_callback_for_retention(
            database,
            uid=f"first-tombstone-batch-{index}",
            received_at_us=1,
            acknowledged_at_us=1,
            receipt_batch_id="first-tombstone-receipt",
        )
        for index in range(256)
    )
    second_sequence = _seed_callback_for_retention(
        database,
        uid="second-tombstone-batch",
        received_at_us=95,
        acknowledged_at_us=95,
        receipt_batch_id="second-tombstone-receipt",
    )
    first_hash = _insert_receipt(
        database,
        batch_id="first-tombstone-receipt",
        first_sequence=first_batch_sequences[0],
        last_sequence=first_batch_sequences[-1],
        created_at_us=99,
    )
    _insert_receipt(
        database,
        batch_id="second-tombstone-receipt",
        first_sequence=second_sequence,
        last_sequence=second_sequence,
        created_at_us=99,
        prior_chain_hash=first_hash,
    )
    with connect_v2(database) as connection:
        connection.execute("UPDATE callback_inbox SET payload_json = NULL")
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1_000, receipt_us=1_000, tombstone_us=10),
    )

    first_pass = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE receipt_batch_id = ?",
                ("first-tombstone-receipt",),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT 1 FROM callback_inbox WHERE source_sequence = ?", (second_sequence,)
            ).fetchone()
            is not None
        )
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 2
    second_pass = manager.run(now_us=200, measured_database_bytes=1, measured_wal_bytes=0)
    rotation_pass = manager.run(now_us=1_100, measured_database_bytes=1, measured_wal_bytes=0)

    assert first_pass.receipts_rolled == 0
    assert first_pass.expired_rows_deleted == 256
    assert second_pass.receipts_rolled == 0
    assert second_pass.expired_rows_deleted == 1
    assert rotation_pass.receipts_rolled == 2
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE source_sequence BETWEEN ? AND ?",
                (first_batch_sequences[0], second_sequence),
            ).fetchone()[0]
            == 0
        )
        watermark = connection.execute(
            "SELECT compacted_through_sequence, cumulative_callback_count, "
            "last_receipt_chain_hash FROM callback_compaction_watermarks "
            "WHERE run_id = 'retention-run'"
        ).fetchone()
    assert tuple(watermark)[:2] == (second_sequence, 257)


def test_tombstone_retention_checkpoints_only_a_complete_receipt_batch(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    first_sequence = _seed_callback_for_retention(
        database,
        uid="old-in-mixed-batch",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="mixed-tombstone-receipt",
    )
    second_sequence = _seed_callback_for_retention(
        database,
        uid="recent-in-mixed-batch",
        received_at_us=95,
        acknowledged_at_us=95,
        receipt_batch_id="mixed-tombstone-receipt",
    )
    _insert_receipt(
        database,
        batch_id="mixed-tombstone-receipt",
        first_sequence=first_sequence,
        last_sequence=second_sequence,
        created_at_us=99,
    )
    with connect_v2(database) as connection:
        connection.execute("UPDATE callback_inbox SET payload_json = NULL")
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1_000, receipt_us=1_000, tombstone_us=10),
    )

    waiting = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE source_sequence IN (?, ?)",
                (first_sequence, second_sequence),
            ).fetchone()[0]
            == 2
        )
    completed = manager.run(now_us=200, measured_database_bytes=1, measured_wal_bytes=0)
    rotated = manager.run(now_us=1_100, measured_database_bytes=1, measured_wal_bytes=0)

    assert waiting.receipts_rolled == 0
    assert waiting.expired_rows_deleted == 0
    assert completed.receipts_rolled == 0
    assert completed.expired_rows_deleted == 2
    assert rotated.receipts_rolled == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE source_sequence IN (?, ?)",
                (first_sequence, second_sequence),
            ).fetchone()[0]
            == 0
        )


def test_tombstone_retention_waits_for_failed_batch_run_to_be_terminal(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    acknowledged_sequence = _seed_callback_for_retention(
        database,
        uid="acknowledged-in-mixed-status-batch",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="mixed-status-receipt",
    )
    failed_sequence = _seed_callback_for_retention(
        database,
        uid="failed-in-mixed-status-batch",
        received_at_us=1,
        lifecycle="failed",
        normalized_event_id=None,
        acknowledged_at_us=None,
        receipt_batch_id="mixed-status-receipt",
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_inbox SET failure_code = 'fixture', payload_json = NULL "
            "WHERE source_sequence = ?",
            (failed_sequence,),
        )
        connection.execute(
            "UPDATE callback_inbox SET payload_json = NULL WHERE source_sequence = ?",
            (acknowledged_sequence,),
        )
    _insert_receipt(
        database,
        batch_id="mixed-status-receipt",
        first_sequence=acknowledged_sequence,
        last_sequence=failed_sequence,
        created_at_us=99,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1_000, receipt_us=1_000, tombstone_us=10),
    )

    active = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM callback_inbox WHERE source_sequence = ?", (failed_sequence,)
            ).fetchone()
            is not None
        )
        connection.execute("UPDATE runs SET status = 'stopped' WHERE run_id = 'retention-run'")
    terminal = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    rotated = manager.run(now_us=1_100, measured_database_bytes=1, measured_wal_bytes=0)

    assert active.receipts_rolled == 0
    assert active.expired_rows_deleted == 0
    assert terminal.receipts_rolled == 0
    assert terminal.expired_rows_deleted == 2
    assert rotated.receipts_rolled == 1


def test_retention_rejects_a_receipt_straddling_the_verified_watermark(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    first_sequence = _seed_callback_for_retention(
        database,
        uid="straddled-first",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="straddled-receipt",
    )
    second_sequence = _seed_callback_for_retention(
        database,
        uid="straddled-second",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="straddled-receipt",
    )
    _insert_receipt(
        database,
        batch_id="straddled-receipt",
        first_sequence=first_sequence,
        last_sequence=second_sequence,
        created_at_us=99,
    )
    with connect_v2(database) as connection:
        connection.execute("UPDATE callback_inbox SET payload_json = NULL")
        connection.execute(
            "INSERT INTO callback_compaction_watermarks(run_id, compacted_through_sequence, "
            "cumulative_callback_count, first_received_at_us, last_received_at_us, "
            "rolled_receipt_chain_hash, last_receipt_chain_hash, updated_at_us) "
            "VALUES ('retention-run', ?, 1, 1, 1, ?, ?, 1)",
            (first_sequence, "1" * 64, "2" * 64),
        )

    with pytest.raises(RetentionInvariantError, match="straddles verified watermark"):
        RetentionManager(
            database,
            RetentionPolicy(callback_payload_us=1_000, receipt_us=1_000, tombstone_us=10),
        ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE source_sequence IN (?, ?)",
                (first_sequence, second_sequence),
            ).fetchone()[0]
            == 2
        )
        assert connection.execute("SELECT count(*) FROM callback_receipts").fetchone()[0] == 1


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


def test_retention_deadline_rolls_back_payload_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    sequence = _seed_callback_for_retention(
        database,
        uid="deadline",
        received_at_us=1,
        receipt_batch_id="deadline-receipt",
    )
    _insert_receipt(
        database,
        batch_id="deadline-receipt",
        first_sequence=sequence,
        last_sequence=sequence,
        created_at_us=99,
    )
    compacted = False

    def monotonic() -> float:
        return 1.0 if compacted else 0.0

    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=10, receipt_us=1_000),
        monotonic=monotonic,
    )
    original_compact = manager._compact_payloads

    def compact_then_expire(connection: sqlite3.Connection, cutoff_us: int, limit: int) -> int:
        nonlocal compacted
        result = original_compact(connection, cutoff_us, limit)
        compacted = True
        return result

    monkeypatch.setattr(manager, "_compact_payloads", compact_then_expire)

    with pytest.raises(MaintenanceDeadlineExceeded, match="deadline"):
        manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        payload = connection.execute(
            "SELECT payload_json FROM callback_inbox WHERE source_sequence = ?", (sequence,)
        ).fetchone()[0]
    assert compacted is True
    assert payload is not None


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


def test_retention_prunes_one_maximum_sealed_output_by_parent_cascade(tmp_path: Path) -> None:
    database = tmp_path / "maximum-sealed-output.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.executemany(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES (?, ?, 'run-1', 1, 1, 'tick', 20, ?, 'pending')",
            ((sequence, f"callback-{sequence}", "7" * 64) for sequence in range(2, 257)),
        )
        connection.executemany(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "(?, 'run-1', ?, 'instrument-1', 'trades', 'tick', 20, 20, 1, '{}', ?)",
            ((f"event-{sequence}", sequence, "8" * 64) for sequence in range(2, 257)),
        )
    base = _output()
    input_ids = tuple(f"event-{sequence}" for sequence in range(1, 257))
    legs = tuple(
        ProposedTradeLeg(
            instrument_id="instrument-1",
            action="buy",
            target="long",
            quantity_value=float(leg_number + 1),
            currency="USD",
        )
        for leg_number in range(MAX_STORED_IDEA_OUTPUT_LEGS)
    )
    output = OperationalRepository(database).put_idea_output(
        IdeaOutputRecord(
            **{
                **base.__dict__,
                "output_kind": "proposed_trade",
                "last_input_event_id": input_ids[-1],
                "input_event_ids": input_ids,
                "authority_status": "unapproved",
                "legs": legs,
            }
        )
    )

    first = RetentionManager(
        database,
        RetentionPolicy(
            idea_shadow_us=10,
            raw_market_event_us=10**18,
            maintenance_batch_rows=MAX_IDEA_OUTPUT_CASCADE_ROWS,
        ),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    second = RetentionManager(
        database,
        RetentionPolicy(
            idea_shadow_us=10,
            raw_market_event_us=10**18,
            maintenance_batch_rows=MAX_IDEA_OUTPUT_CASCADE_ROWS,
        ),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert first.expired_rows_deleted == MAX_IDEA_OUTPUT_CASCADE_ROWS
    assert second.expired_rows_deleted == 0
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM idea_outputs WHERE output_id=?", (output.output_id,)
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 256


def _seed_closed_shadow_retention_case(database: Path, *, leg_count: int) -> None:
    initialize_database(database)
    _seed_output_dependencies(database)
    base = _output()
    proposal = OperationalRepository(database).put_idea_output(
        IdeaOutputRecord(
            **{
                **base.__dict__,
                "output_kind": "proposed_trade",
                "authority_status": "unapproved",
                "legs": tuple(
                    ProposedTradeLeg(
                        instrument_id="instrument-1",
                        action="buy",
                        target="long",
                        quantity_value=float(leg_number + 1),
                        currency="USD",
                    )
                    for leg_number in range(leg_count)
                ),
            }
        )
    )
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, closed_at_us, lifecycle, cost_model_id, fill_model_id, currency, "
            "data_class) VALUES ('closed-position', ?, 'run-1', 'instance-1', 40, 'closed', "
            "'cost', 'fill', 'USD', 'shadow_protected')",
            (proposal.output_id,),
        )
        connection.execute(
            "INSERT INTO shadow_progress(position_id, entry_after_source_sequence, "
            "next_source_sequence, next_horizon_index, updated_at_us) "
            "VALUES ('closed-position', 1, 2, 0, 40)"
        )
        connection.executemany(
            "INSERT INTO shadow_quote_state(position_id, leg_number, instrument_id) "
            "VALUES ('closed-position', ?, 'instrument-1')",
            ((leg_number,) for leg_number in range(leg_count)),
        )


def test_shadow_position_retention_budget_one_makes_physical_progress_without_cascade(
    tmp_path: Path,
) -> None:
    database = tmp_path / "one-row-shadow-retention.sqlite3"
    _seed_closed_shadow_retention_case(database, leg_count=1)
    manager = RetentionManager(
        database,
        RetentionPolicy(idea_shadow_us=10, maintenance_batch_rows=1),
    )

    states: list[tuple[int, int, int, int]] = []
    for _ in range(3):
        result = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
        with connect_v2(database) as connection:
            states.append(
                (
                    result.expired_rows_deleted,
                    connection.execute("SELECT count(*) FROM shadow_positions").fetchone()[0],
                    connection.execute("SELECT count(*) FROM shadow_progress").fetchone()[0],
                    connection.execute("SELECT count(*) FROM shadow_quote_state").fetchone()[0],
                )
            )

    assert states == [(1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 0, 0)]


def test_shadow_position_retention_counts_maximum_quote_state_cascade_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "maximum-shadow-retention.sqlite3"
    _seed_closed_shadow_retention_case(database, leg_count=8)

    result = RetentionManager(
        database,
        RetentionPolicy(
            idea_shadow_us=10,
            maintenance_batch_rows=MAX_SHADOW_POSITION_CASCADE_ROWS,
        ),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert result.expired_rows_deleted == MAX_SHADOW_POSITION_CASCADE_ROWS
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM shadow_positions").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM shadow_progress").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM shadow_quote_state").fetchone()[0] == 0


def test_derived_event_retention_waits_for_explicit_mapping_prune_at_batch_cut(
    tmp_path: Path,
) -> None:
    database = tmp_path / "derived-event-cascade.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_json, payload_sha256, lifecycle) VALUES "
            "(2, 'bar-callback', 'run-1', 1, 1, 'bar', 10, '{}', ?, 'pending')",
            ("1" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('input-bar', 'run-1', 2, 'instrument-1', 'bars', 'bar', 10, 10, 1, '{}', ?)",
            ("2" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, event_at_us, "
            "received_at_us, connection_generation, payload_json, payload_sha256) VALUES "
            "('derived-bar', 'run-1', NULL, 2, 'instrument-1', 'bars', 'bar_5m', 20, 20, 1, "
            "'{}', ?)",
            ("3" * 64,),
        )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) VALUES "
            "('derived-bar', 'input-bar', 0, 'constituent', 99)"
        )
    manager = RetentionManager(
        database,
        RetentionPolicy(
            raw_market_event_us=10**18,
            completed_bar_us=10,
            derivation_mapping_us=1_000,
            idea_shadow_us=10**18,
            maintenance_batch_rows=1,
        ),
    )

    before_mapping_expiry = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        first_counts = tuple(
            int(value)
            for value in connection.execute(
                "SELECT "
                "(SELECT count(*) FROM market_events WHERE event_id='derived-bar'), "
                "(SELECT count(*) FROM market_event_derivations "
                "WHERE derived_event_id='derived-bar')"
            ).fetchone()
        )
    mapping_pass = manager.run(now_us=2_000, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        second_counts = tuple(
            int(value)
            for value in connection.execute(
                "SELECT "
                "(SELECT count(*) FROM market_events WHERE event_id='derived-bar'), "
                "(SELECT count(*) FROM market_event_derivations "
                "WHERE derived_event_id='derived-bar')"
            ).fetchone()
        )
    event_pass = manager.run(now_us=2_000, measured_database_bytes=1, measured_wal_bytes=0)

    assert before_mapping_expiry.expired_rows_deleted == 0
    assert first_counts == (1, 1)
    assert mapping_pass.expired_rows_deleted == 1
    assert second_counts == (1, 0)
    assert event_pass.expired_rows_deleted == 1
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_id='derived-bar'"
            ).fetchone()[0]
            == 0
        )


def test_phase2_receipt_mappings_follow_the_bounded_receipt_retention_tier(
    tmp_path: Path,
) -> None:
    database = tmp_path / "phase2-receipt-retention.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_json, payload_sha256, lifecycle) VALUES "
            "(2, 'bar-callback', 'run-1', 1, 1, 'bar', 10, '{}', ?, 'pending')",
            ("1" * 64,),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "payload_json, payload_sha256) VALUES "
            "('input-bar', 'run-1', 2, 'instrument-1', 'bars', 'bar', 10, 10, 1, '{}', ?)",
            ("2" * 64,),
        )
        for event_id, event_kind in (
            ("derived-bar", "bar_5m"),
            ("session-prefix", "bar_5m_session_prefix"),
        ):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, payload_json, "
                "payload_sha256) VALUES (?, 'run-1', NULL, 2, 'instrument-1', 'bars', ?, "
                "20, 20, 1, '{}', ?)",
                (event_id, event_kind, hashlib.sha256(event_id.encode()).hexdigest()),
            )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) VALUES "
            "('derived-bar', 'input-bar', 0, 'constituent', 20)"
        )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) VALUES "
            "('session-prefix', 'derived-bar', 0, 'constituent', 20)"
        )
    manager = RetentionManager(
        database,
        RetentionPolicy(
            raw_market_event_us=10**18,
            completed_bar_us=1_000,
            derivation_mapping_us=10,
            idea_shadow_us=10**18,
            maintenance_batch_rows=100,
        ),
    )

    manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    with connect_v2(database) as connection:
        mappings = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT derived_event_id, input_event_id FROM market_event_derivations "
                "ORDER BY derived_event_id"
            )
        )
    assert mappings == (("session-prefix", "derived-bar"),)


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
            maintenance_batch_rows=4,
        ),
    )

    first = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    second = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)

    assert first.payloads_compacted + first.receipts_rolled + first.expired_rows_deleted == 4
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
            "INSERT INTO idea_checkpoints VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?)",
            ("instance-1", "event-1", 1, "{}", "0" * 64, 30, 30, 0),
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
            "normalized_event_id = 'event-1', acknowledged_at_us = 20, "
            "receipt_batch_id = 'tombstone-receipt' WHERE source_sequence = 1"
        )
    _insert_receipt(
        database,
        batch_id="tombstone-receipt",
        first_sequence=1,
        last_sequence=1,
        created_at_us=99,
        run_id="run-1",
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


def test_granular_receipt_tombstones_expire_as_a_complete_batch(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    first = _seed_callback_for_retention(
        database,
        uid="batch-first",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="batch-proof",
    )
    second = _seed_callback_for_retention(
        database,
        uid="batch-second",
        received_at_us=99,
        acknowledged_at_us=99,
        receipt_batch_id="batch-proof",
    )
    _insert_receipt(
        database,
        batch_id="batch-proof",
        first_sequence=first,
        last_sequence=second,
        created_at_us=99,
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE callback_inbox SET payload_json = NULL WHERE source_sequence BETWEEN ? AND ?",
            (first, second),
        )
    manager = RetentionManager(
        database,
        RetentionPolicy(tombstone_us=10, receipt_us=1_000, raw_market_event_us=1_000),
    )

    partial = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    complete = manager.run(now_us=200, measured_database_bytes=1, measured_wal_bytes=0)

    assert partial.expired_rows_deleted == 0
    assert complete.expired_rows_deleted == 2
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE source_sequence BETWEEN ? AND ?",
                (first, second),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE source_sequence BETWEEN ? AND ?",
                (first, second),
            ).fetchone()[0]
            == 2
        )


def test_receipt_cannot_skip_unproven_same_run_predecessor_at_genesis(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status='stopped' WHERE run_id='retention-run'")
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_json, payload_sha256, lifecycle, failure_code) VALUES "
            "(101, 'skipped-retention', 'retention-run', 1, 1, 'quote', 1, '{}', ?, "
            "'failed', 'fixture')",
            ("1" * 64,),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_json, payload_sha256, lifecycle, failure_code) VALUES "
            "(102, 'interleaved-other-run', 'run-1', 1, 1, 'quote', 1, '{}', ?, "
            "'failed', 'fixture')",
            ("2" * 64,),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) VALUES "
            "(103, 'forged-retention', 'retention-run', 1, 1, 'quote', 2, '{}', ?, "
            "'failed', 'fixture', 'forged-skip')",
            ("3" * 64,),
        )
    _insert_receipt(
        database,
        batch_id="forged-skip",
        first_sequence=103,
        last_sequence=103,
        created_at_us=2,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1, receipt_us=1, tombstone_us=1),
    )
    with pytest.raises(RetentionInvariantError, match="skipped"):
        manager.run(now_us=10, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT payload_json FROM callback_inbox WHERE source_sequence=103"
            ).fetchone()[0]
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM callback_compaction_watermarks WHERE run_id='retention-run'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_receipts WHERE batch_id='forged-skip'"
            ).fetchone()[0]
            == 1
        )


def test_receipt_cannot_skip_same_run_callback_after_permanent_watermark(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status='stopped' WHERE run_id='retention-run'")
        for sequence, batch in ((101, "first-proof"), (103, None), (105, "later-proof")):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_json, payload_sha256, lifecycle, failure_code, receipt_batch_id) "
                "VALUES (?, ?, 'retention-run', 1, 1, 'quote', ?, '{}', ?, "
                "'failed', 'fixture', ?)",
                (sequence, f"watermark-{sequence}", sequence, f"{sequence:064x}", batch),
            )
    first_hash = _insert_receipt(
        database,
        batch_id="first-proof",
        first_sequence=101,
        last_sequence=101,
        created_at_us=1,
    )
    _insert_receipt(
        database,
        batch_id="later-proof",
        first_sequence=105,
        last_sequence=105,
        created_at_us=99,
        prior_chain_hash=first_hash,
    )
    manager = RetentionManager(
        database,
        RetentionPolicy(callback_payload_us=1_000, receipt_us=50, tombstone_us=1_000),
    )
    first_pass = manager.run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    assert first_pass.receipts_rolled == 1
    with pytest.raises(RetentionInvariantError, match="skipped"):
        manager.run(now_us=200, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        watermark = connection.execute(
            "SELECT compacted_through_sequence, cumulative_callback_count "
            "FROM callback_compaction_watermarks WHERE run_id='retention-run'"
        ).fetchone()
        assert tuple(watermark) == (101, 1)
        assert (
            connection.execute(
                "SELECT count(*) FROM callback_receipts WHERE batch_id='later-proof'"
            ).fetchone()[0]
            == 1
        )


def test_actual_wal_size_remains_fatal_when_a_reader_blocks_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    reader = sqlite3.connect(database)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM runs").fetchone()
        with connect_v2(database) as writer:
            writer.executemany(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "opened_at_us, details_json) VALUES (?, 'run-1', 'fixture', 'info', "
                "'reader-blocked', 100, '{}')",
                ((f"wal-{index}",) for index in range(100)),
            )

        result = RetentionManager(database, RetentionPolicy(wal_cap_bytes=1)).run(now_us=100)
    finally:
        reader.close()

    assert result.wal_bytes >= 1
    assert result.cap_state is StorageCapState.FATAL
    assert result.admission_allowed is False
    assert result.required_action == "WAL_CAP_FATAL"


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
            IDEA_OUTPUT_RETENTION_CANDIDATES_SQL,
            (100, MAX_IDEA_OUTPUT_PARENTS_PER_PASS),
        ),
        "shadow_positions_retention_idx": (
            SHADOW_POSITION_RETENTION_CANDIDATES_SQL,
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
            if expected_index in {
                "idea_outputs_retention_idx",
                "shadow_positions_retention_idx",
            }:
                assert "USE TEMP B-TREE" not in plan
            if expected_index == "shadow_positions_retention_idx":
                assert "sqlite_autoindex_shadow_progress_1" in plan
                assert "sqlite_autoindex_shadow_quote_state_1" in plan


def test_callback_recovery_uses_bounded_unresolved_gap_plan(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN {CALLBACK_GAP_RECOVERY_SQL}",
                (
                    150,
                    152,
                    "run",
                    "run",
                    "instrument",
                    "quotes",
                    0,
                    1,
                    100,
                    "IBKR_CONNECT_FAILED",
                    "IBKR_DISCONNECT",
                    "IBKR_SUBSCRIBE_FAILED",
                    "RECONNECT_UNCERTAINTY",
                    "STREAM_STALE",
                    150,
                    150,
                    152,
                ),
            )
        )
    assert "gaps_unresolved_idx" in plan
    assert "sqlite_autoindex_subscriptions_1" in plan


def test_callback_incident_recovery_uses_primary_key_plan(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    with connect_v2(database) as connection:
        plan = " ".join(
            str(row["detail"])
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN {CALLBACK_INCIDENT_RECOVERY_SQL}",
                (152, "a", "b", "c", "d", 150),
            )
        )
    assert "sqlite_autoindex_incidents_1" in plan
    assert "incidents_run_opened_idx" not in plan


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
    assert init_payload == {
        "applied_versions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "current_version": 14,
        "status": "ok",
    }
    assert migration_payload == {"applied_versions": [], "current_version": 14, "status": "ok"}
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


def test_migrate_and_retain_cli_errors_are_machine_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v2.sqlite3"

    def fail_migration(_database: Path) -> None:
        raise SchemaError("simulated schema mismatch")

    class FailingRetentionManager:
        def __init__(self, _database: Path) -> None:
            pass

        def run(self, *, now_us: int) -> None:
            raise RetentionInvariantError(f"simulated receipt failure at {now_us}")

    monkeypatch.setattr("stocker_runtime.cli.migrate_database", fail_migration)
    migrated = CliRunner().invoke(runtime_app, ["migrate", str(database)])
    monkeypatch.setattr("stocker_runtime.cli.RetentionManager", FailingRetentionManager)
    retained = CliRunner().invoke(runtime_app, ["retain", str(database), "--now-us", "100"])

    assert migrated.exit_code == retained.exit_code == 1
    assert json.loads(migrated.stdout) == {
        "error": "SchemaError",
        "message": "simulated schema mismatch",
        "status": "error",
    }
    assert json.loads(retained.stdout) == {
        "error": "RetentionInvariantError",
        "message": "simulated receipt failure at 100",
        "status": "error",
    }


def test_market_events_are_update_immutable_but_retention_can_delete(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    _seed_output_dependencies(database)
    sequence = _seed_callback_for_retention(
        database,
        uid="immutable-event",
        received_at_us=1,
        acknowledged_at_us=1,
        receipt_batch_id="immutable-proof",
    )
    _insert_receipt(
        database,
        batch_id="immutable-proof",
        first_sequence=sequence,
        last_sequence=sequence,
        created_at_us=1,
    )
    with connect_v2(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="market_event_immutable"):
            connection.execute(
                "UPDATE market_events SET event_at_us=2 WHERE source_sequence=?", (sequence,)
            )
        with pytest.raises(sqlite3.IntegrityError, match="market_event_immutable"):
            connection.execute(
                "UPDATE market_events SET payload_sha256=? WHERE source_sequence=?",
                ("f" * 64, sequence),
            )
        connection.execute("UPDATE runs SET status='stopped' WHERE run_id='retention-run'")
    RetentionManager(
        database,
        RetentionPolicy(raw_market_event_us=1, callback_payload_us=1_000, receipt_us=1_000),
    ).run(now_us=100, measured_database_bytes=1, measured_wal_bytes=0)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM market_events WHERE source_sequence=?", (sequence,)
            ).fetchone()
            is None
        )
