from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.migration_order import (
    MigrationOrderError,
    migration_plan,
)

MIGRATION_ROOT = (
    Path(__file__).parents[1]
    / "packages"
    / "stocker_prospective"
    / "src"
    / "stocker_prospective"
    / "migrations"
)
OLDER_FIXTURE_MANIFEST = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "prospective_migrations_through_0021.sha256.json"
)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _normalise_sql(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


def _schema_snapshot(database_path: Path) -> dict[str, object]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        table_rows = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_schema
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: dict[str, object] = {}
        for table_row in table_rows:
            table = str(table_row["name"])
            quoted = _quote_identifier(table)
            columns = tuple(
                tuple(row)
                for row in connection.execute(f"PRAGMA table_xinfo({quoted})")  # noqa: S608
            )
            foreign_keys = tuple(
                tuple(row)
                for row in connection.execute(  # noqa: S608
                    f"PRAGMA foreign_key_list({quoted})"
                )
            )
            indexes: list[object] = []
            for index_row in connection.execute(  # noqa: S608
                f"PRAGMA index_list({quoted})"
            ):
                index_name = str(index_row["name"])
                index_sql_row = connection.execute(
                    "SELECT sql FROM sqlite_schema WHERE type = 'index' AND name = ?",
                    (index_name,),
                ).fetchone()
                indexes.append(
                    {
                        "metadata": tuple(index_row),
                        "columns": tuple(
                            tuple(row)
                            for row in connection.execute(  # noqa: S608
                                f"PRAGMA index_xinfo({_quote_identifier(index_name)})"
                            )
                        ),
                        "sql": (
                            None if index_sql_row is None else _normalise_sql(index_sql_row["sql"])
                        ),
                    }
                )
            tables[table] = {
                "sql": _normalise_sql(table_row["sql"]),
                "columns": columns,
                "foreign_keys": foreign_keys,
                "indexes": tuple(indexes),
            }
        foreign_key_check = tuple(connection.execute("PRAGMA foreign_key_check"))
    return {"tables": tables, "foreign_key_check": foreign_key_check}


def _apply_older_fixture(database_path: Path, *, through_sequence: int) -> None:
    plan = migration_plan(MIGRATION_ROOT)
    frozen_hashes = json.loads(OLDER_FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at_utc TEXT NOT NULL
            )
            """
        )
        for migration in plan:
            if migration.sequence > through_sequence:
                break
            expected_hash = frozen_hashes[migration.path.name]
            actual_hash = hashlib.sha256(migration.path.read_bytes()).hexdigest()
            assert actual_hash == expected_hash, (
                f"deployed migration changed: {migration.path.name}"
            )
            version = migration.path.name.replace("'", "''")
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                f"{migration.path.read_text(encoding='utf-8')}\n"
                "INSERT INTO schema_migrations(version, applied_at_utc) "
                f"VALUES ('{version}', '2026-01-01T00:00:00+00:00');\n"
                "COMMIT;\n"
            )


def _migration_ledger(database_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(database_path) as connection:
        return tuple(
            str(row[0])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY rowid")
        )


def test_migration_plan_preserves_deployed_duplicate_filename_order() -> None:
    names = tuple(item.path.name for item in migration_plan(MIGRATION_ROOT))

    assert names.index("0011_m1c_checkpoint_completion_v0.sql") < names.index(
        "0011_m1c_tail_phase_v1.sql"
    )
    assert names.index("0012_m1c_signed_market_shock_v1.sql") < names.index(
        "0012_option_schedule_degradation_v0.sql"
    )
    assert len(names) == len(set(names))


def test_migration_plan_rejects_any_new_duplicate_prefix(tmp_path: Path) -> None:
    migration_root = tmp_path / "migrations"
    migration_root.mkdir()
    for source in MIGRATION_ROOT.glob("*.sql"):
        (migration_root / source.name).write_bytes(source.read_bytes())
    (migration_root / "0016_conflicting_new_migration.sql").write_text(
        "SELECT 1;\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MigrationOrderError,
        match="duplicate migration sequence 0016",
    ):
        migration_plan(migration_root)


def test_fast_summary_queries_have_growth_independent_indexes(tmp_path: Path) -> None:
    database = tmp_path / "indexed.sqlite3"
    ProspectiveRepository(database).migrate()
    expected_indexes = {
        "idx_underlying_bar_web_latest",
        "idx_model_score_web_latest",
        "idx_previous_session_context_web_latest",
        "idx_ibkr_connection_event_web_latest",
        "idx_option_surface_capture_web_latest",
        "idx_market_data_budget_event_run",
        "idx_source_capture_completion_web_latest",
        "idx_signal_episode_time",
        "idx_data_health_event_web_active",
        "idx_web_active_runtime_blocker",
        "idx_subscription_lifecycle_web_active",
        "idx_web_latest_subscription_active_v0",
        "idx_m1c_checkpoint_web_latest",
        "idx_m1c_checkpoint_web_latest_global",
        "idx_m1c_episode_web_latest_exact",
        "idx_ibkr_connection_event_web_any_latest",
    }
    with sqlite3.connect(database) as connection:
        index_rows = connection.execute(
            "SELECT name, sql FROM sqlite_schema WHERE type = 'index'"
        ).fetchall()
        indexes = {str(name): sql for name, sql in index_rows}
        active_subscription_plan = " ".join(
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT subscription_kind, COUNT(*) AS used
                FROM web_latest_subscription_state_v0
                WHERE run_id = ?
                  AND status IN ('pending', 'active', 'cancellation_requested')
                GROUP BY subscription_kind
                """,
                ("synthetic",),
            )
        )
        latest_checkpoint_plan = " ".join(
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM m1c_checkpoint_v0
                WHERE run_id = ? ORDER BY bar_end_utc DESC, id DESC LIMIT 1
                """,
                ("synthetic",),
            )
        )
        latest_episode_plan = " ".join(
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM m1c_episode_v0
                WHERE run_id = ? ORDER BY trigger_bar_end_utc DESC LIMIT 1
                """,
                ("synthetic",),
            )
        )
        latest_connection_plan = " ".join(
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM ibkr_connection_event
                WHERE run_id = ? ORDER BY id DESC LIMIT 1
                """,
                ("synthetic",),
            )
        )

    assert expected_indexes <= indexes.keys()
    assert "WHERE cancelled_at_utc IS NULL" in str(indexes["idx_subscription_lifecycle_web_active"])
    assert "idx_web_latest_subscription_active_v0" in active_subscription_plan
    assert "idx_m1c_checkpoint_web_latest_global" in latest_checkpoint_plan
    assert "idx_m1c_episode_live" in latest_episode_plan
    assert "USE TEMP B-TREE" not in latest_episode_plan
    assert "idx_ibkr_connection_event_web_any_latest" in latest_connection_plan


def test_latest_subscription_projection_backfills_existing_restart_events(
    tmp_path: Path,
) -> None:
    database = tmp_path / "subscription-backfill.sqlite3"
    repository = ProspectiveRepository(database)
    repository.migrate()
    migration = "0028_web_latest_subscription_state_v0.sql"

    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER web_latest_subscription_state_insert_v0")
        connection.execute("DROP TABLE web_latest_subscription_state_v0")
        connection.execute("DELETE FROM schema_migrations WHERE version = ?", (migration,))
        connection.execute(
            """
            INSERT INTO prospective_run(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, created_at_utc,
                mode, status, scientific_classification
            ) VALUES (
                'restart-run', '2026-08-01T00:00:00+00:00', 'test', 'deadbeef',
                'model', 'universe', 'cohort', '2026-08-01T00:00:00+00:00',
                'record_only', 'active', 'test'
            )
            """
        )
        envelope_id = int(
            connection.execute(
                """
                INSERT INTO evidence_envelope(
                    run_id, prospective_start_utc, app_version, git_commit,
                    model_artifact_id, universe_id, cohort,
                    source_timestamps_json, recorded_at_utc
                ) VALUES (
                    'restart-run', '2026-08-01T00:00:00+00:00', 'test',
                    'deadbeef', 'model', 'universe', 'cohort', '[]',
                    '2026-08-01T00:00:00+00:00'
                )
                """
            ).lastrowid
        )
        for request_id, status, generation in (
            (7, "active", 1),
            (7, "cancelled", 1),
            (8, "active", 2),
        ):
            connection.execute(
                """
                INSERT INTO subscription_lifecycle_event_v0(
                    envelope_id, run_id, occurred_at_utc, subscription_key,
                    request_id, subscription_kind, subscription_class, symbol,
                    con_id, status, owner_ids_json, owner_count, generation,
                    reason, payload_json, claims_json
                ) VALUES (?, 'restart-run', '2026-08-01T00:00:00+00:00',
                          'BAR|123|5m|RTH', ?, 'bar', 0, 'AAL', 123, ?,
                          '["system:AAL"]', 1, ?, NULL, '{}', '{}')
                """,
                (envelope_id, request_id, status, generation),
            )

    repository.migrate()

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT event_id, subscription_kind, status
            FROM web_latest_subscription_state_v0
            WHERE run_id = 'restart-run' AND subscription_key = 'BAR|123|5m|RTH'
            """
        ).fetchone()
        latest_event_id = connection.execute(
            """
            SELECT MAX(id)
            FROM subscription_lifecycle_event_v0
            WHERE run_id = 'restart-run'
            """
        ).fetchone()[0]

    assert row == (latest_event_id, "bar", "active")


def test_quiet_quote_audit_migration_preserves_original_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "quiet-quote-defect-upgrade.sqlite3"
    _apply_older_fixture(database, through_sequence=21)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for migration in migration_plan(MIGRATION_ROOT):
            if not 22 <= migration.sequence <= 29:
                continue
            connection.executescript(migration.path.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                (migration.path.name, "2026-08-04T00:00:00+00:00"),
            )
        connection.execute(
            """
            INSERT INTO prospective_run(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, created_at_utc,
                mode, status, scientific_classification
            ) VALUES (
                'legacy-quiet-run', '2026-08-03T13:30:00+00:00', 'test',
                'deadbeef', 'M1C', 'frozen-20', 'anchor_frozen_20',
                '2026-08-03T13:30:00+00:00', 'record_only', 'active', 'research_only'
            )
            """
        )
        envelope_id = int(
            connection.execute(
                """
                INSERT INTO evidence_envelope(
                    run_id, prospective_start_utc, app_version, git_commit,
                    model_artifact_id, universe_id, cohort,
                    source_timestamps_json, recorded_at_utc
                ) VALUES (
                    'legacy-quiet-run', '2026-08-03T13:30:00+00:00', 'test',
                    'deadbeef', 'M1C', 'frozen-20', 'anchor_frozen_20', '[]',
                    '2026-08-03T14:00:10+00:00'
                )
                """
            ).lastrowid
        )
        checkpoint_id = int(
            connection.execute(
                """
                INSERT INTO m1c_checkpoint_v0(
                    envelope_id, run_id, symbol, session_date, checkpoint,
                    bar_start_utc, bar_end_utc, feature_as_of_utc, model_id,
                    model_version, model_hash, feature_hash,
                    session_context_hash, feature_values_json, probability,
                    threshold, threshold_passed, eligible, feature_freshness,
                    missing_feature_count, rejection_reasons_json, claims_json
                ) VALUES (
                    ?, 'legacy-quiet-run', 'HIMS', '2026-08-03', 6,
                    '2026-08-03T13:55:00+00:00', '2026-08-03T14:00:00+00:00',
                    '2026-08-03T14:00:00+00:00', 'M1C', 'frozen-m1c-v0',
                    ?, ?, ?, '{}', 0.13, 0.488333710794033, 0, 0,
                    'fresh', 0, '["underlying_quote_stale"]', '{}'
                )
                """,
                (envelope_id, "a" * 64, "b" * 64, "c" * 64),
            ).lastrowid
        )
        quiet_checkpoint_id = int(
            connection.execute(
                """
                INSERT INTO quiet_state_checkpoint_v0(
                    envelope_id, checkpoint_id, run_id, symbol, session_date,
                    checkpoint, m1c_probability, previous_m1c_probability,
                    bottom_5, bottom_10, bottom_20, high_tail,
                    distance_from_bottom_10, model_hash, feature_hash, eligible,
                    data_quality_status, data_quality_flags_json, claims_json
                ) VALUES (
                    ?, ?, 'legacy-quiet-run', 'HIMS', '2026-08-03', 6,
                    0.13, 0.20, 0, 1, 1, 0, -0.005896965695626,
                    ?, ?, 0, 'invalid', '["underlying_quote_stale"]', '{}'
                )
                """,
                (envelope_id, checkpoint_id, "a" * 64, "b" * 64),
            ).lastrowid
        )
        original = connection.execute(
            """
            SELECT checkpoint_id, run_id, symbol, session_date, checkpoint,
                   m1c_probability, eligible, data_quality_status,
                   data_quality_flags_json
            FROM quiet_state_checkpoint_v0 WHERE id = ?
            """,
            (quiet_checkpoint_id,),
        ).fetchone()

    ProspectiveRepository(database).migrate()

    with sqlite3.connect(database) as connection:
        preserved = connection.execute(
            """
            SELECT checkpoint_id, run_id, symbol, session_date, checkpoint,
                   m1c_probability, eligible, data_quality_status,
                   data_quality_flags_json
            FROM quiet_state_checkpoint_v0 WHERE id = ?
            """,
            (quiet_checkpoint_id,),
        ).fetchone()
        quote_audit = connection.execute(
            """
            SELECT selected_underlying_quote_event_id,
                   selected_underlying_quote_timestamp_utc,
                   selected_underlying_quote_age_seconds,
                   underlying_quote_selection_policy
            FROM quiet_state_checkpoint_v0 WHERE id = ?
            """,
            (quiet_checkpoint_id,),
        ).fetchone()
        defect = connection.execute(
            """
            SELECT affected_checkpoint_count, dataset_scope,
                   original_evidence_modified, recomputation_authorized,
                   may_create_quiet_observation
            FROM quiet_quote_instrumentation_defect_v0
            """
        ).fetchone()
        affected = connection.execute(
            """
            SELECT quiet_checkpoint_id, original_eligible,
                   original_data_quality_flags_json
            FROM quiet_quote_instrumentation_defect_checkpoint_v0
            """
        ).fetchone()

    assert preserved == original
    assert quote_audit == (None, None, None, None)
    assert defect == (
        1,
        "non_prospective_derived_instrumentation_audit",
        0,
        0,
        0,
    )
    assert affected == (quiet_checkpoint_id, 0, '["underlying_quote_stale"]')


def test_fresh_and_upgrade_migrations_produce_equivalent_schema(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.sqlite3"
    upgraded = tmp_path / "upgraded.sqlite3"
    _apply_older_fixture(upgraded, through_sequence=21)
    deployed_ledger = _migration_ledger(upgraded)
    assert "0011_m1c_checkpoint_completion_v0.sql" in deployed_ledger
    assert "0011_m1c_tail_phase_v1.sql" in deployed_ledger
    assert "0012_m1c_signed_market_shock_v1.sql" in deployed_ledger
    assert "0012_option_schedule_degradation_v0.sql" in deployed_ledger
    assert "0016_prospective_recorder_hardening_v1.sql" in deployed_ledger
    assert "0021_opening_reversal_activation_run_binding_v1.sql" in deployed_ledger

    ProspectiveRepository(fresh).migrate()
    ProspectiveRepository(upgraded).migrate()

    fresh_schema = _schema_snapshot(fresh)
    upgraded_schema = _schema_snapshot(upgraded)
    assert fresh_schema["foreign_key_check"] == upgraded_schema["foreign_key_check"]
    fresh_tables = fresh_schema["tables"]
    upgraded_tables = upgraded_schema["tables"]
    assert isinstance(fresh_tables, dict)
    assert isinstance(upgraded_tables, dict)
    assert fresh_tables.keys() == upgraded_tables.keys()
    for table in fresh_tables:
        assert fresh_tables[table] == upgraded_tables[table], table
    expected_ledger = tuple(item.path.name for item in migration_plan(MIGRATION_ROOT))
    assert _migration_ledger(fresh) == expected_ledger
    assert _migration_ledger(upgraded) == expected_ledger
    assert all(version.endswith(".sql") for version in expected_ledger)
