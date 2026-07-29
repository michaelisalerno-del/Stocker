from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.backup import backup_database
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    SchemaVersionTooNew,
)
from stocker_prospective.durable_inbox import (
    CallbackClassification,
    DurableCallbackInbox,
)
from stocker_prospective.operational_state import (
    RecorderOperationalRepository,
    RuntimeArtifactVerification,
)

MIGRATION_ROOT = (
    Path(__file__).parents[1] / "packages/stocker_prospective/src/stocker_prospective/migrations"
)
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def create_pre_hardening_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at_utc TEXT NOT NULL
            )
            """
        )
        for migration in sorted(MIGRATION_ROOT.glob("*.sql")):
            if migration.name.startswith(("0016_", "0017_", "0018_")):
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                """
                INSERT INTO schema_migrations(version, applied_at_utc)
                VALUES (?, ?)
                """,
                (migration.name, NOW.isoformat()),
            )
        connection.execute(
            """
            INSERT INTO prospective_run(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, created_at_utc,
                mode, status, scientific_classification
            ) VALUES ('legacy-run', ?, 'legacy', ?, 'legacy-model',
                      'anchor-frozen-20-v1', 'anchor_frozen_20', ?,
                      'record_only', 'active', 'research_only')
            """,
            (
                (NOW - timedelta(days=1)).isoformat(),
                "a" * 40,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES ('legacy-run', 'ibkr', '2026-07-29', 'AAL',
                      'five_minute_bar_event', '/legacy/immutable.parquet',
                      1, ?, ?, 'raw-event-v1', ?, 1, 17, 'legacy',
                      'legacy', ?, '{}')
            """,
            (
                NOW.isoformat(),
                NOW.isoformat(),
                "b" * 64,
                NOW.isoformat(),
            ),
        )


def metadata(run_id: str) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=NOW - timedelta(days=1),
        app_version="test",
        git_commit="c" * 40,
        model_artifact_id="synthetic",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=[NOW.isoformat()],
        recorded_at_utc=NOW,
    )


def test_pre_hardening_database_migrates_forward_without_deleting_legacy_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pre-hardening.sqlite3"
    create_pre_hardening_database(path)
    repository = ProspectiveRepository(path)

    repository.migrate()
    repository.migrate()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM prospective_run WHERE run_id = 'legacy-run'"
        ).fetchone() == (1,)
        assert connection.execute("SELECT gap_count FROM raw_partition_manifest_v0").fetchone() == (
            17,
        )
        assert connection.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE version = '0016_prospective_recorder_hardening_v1.sql'
            """
        ).fetchone() == (1,)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE version = '0017_callback_raw_only_recovery_v1.sql'
            """
        ).fetchone() == (1,)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM schema_migrations
            WHERE version = '0018_virtual_position_ledgers_v1.sql'
            """
        ).fetchone() == (1,)
        for table in (
            "callback_inbox_v1",
            "callback_raw_materialization_v1",
            "callback_processing_commit_v1",
            "recorder_generation_v1",
            "recorder_operational_state_v1",
            "recorder_fatal_latch_v1",
            "gap_incident_v1",
            "runtime_artifact_verification_v1",
            "operational_incident_v1",
        ):
            assert connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table,),
            ).fetchone() == (1,)
        for view in (
            "opening_reversal_virtual_position_v1",
            "quiet_state_virtual_position_v1",
        ):
            assert connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'view' AND name = ?
                """,
                (view,),
            ).fetchone() == (1,)
        inbox_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(callback_inbox_v1)")
        }
        processing_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(callback_processing_commit_v1)")
        }
        assert "stream_owner_json" in inbox_columns
        assert {
            "processing_disposition",
            "scientific_projection_complete",
        }.issubset(processing_columns)


def test_database_with_0016_already_applied_receives_0017_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "already-on-0016.sqlite3"
    create_pre_hardening_database(path)
    migration_0016 = MIGRATION_ROOT / "0016_prospective_recorder_hardening_v1.sql"
    with sqlite3.connect(path) as connection:
        connection.executescript(migration_0016.read_text(encoding="utf-8"))
        connection.execute(
            """
            INSERT INTO schema_migrations(version, applied_at_utc)
            VALUES (?, ?)
            """,
            (migration_0016.name, NOW.isoformat()),
        )

    ProspectiveRepository(path).migrate()

    with sqlite3.connect(path) as connection:
        versions = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT version FROM schema_migrations
                WHERE version LIKE '0016_%'
                   OR version LIKE '0017_%'
                   OR version LIKE '0018_%'
                """
            )
        }
        inbox_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(callback_inbox_v1)")
        }
        processing_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(callback_processing_commit_v1)")
        }
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'trigger' AND name LIKE 'trg_opening_reversal%'
                """
            )
        }
        legacy_gap_count = connection.execute(
            "SELECT gap_count FROM raw_partition_manifest_v0"
        ).fetchone()
    assert versions == {
        "0016_prospective_recorder_hardening_v1.sql",
        "0017_callback_raw_only_recovery_v1.sql",
        "0018_virtual_position_ledgers_v1.sql",
    }
    assert "stream_owner_json" in inbox_columns
    assert {
        "processing_disposition",
        "scientific_projection_complete",
    }.issubset(processing_columns)
    assert {
        "trg_opening_reversal_discovery_immutable",
        "trg_opening_reversal_option_outcome_immutable",
    }.issubset(trigger_names)
    assert {
        "trg_opening_reversal_v1_1_discovery_immutable",
        "trg_opening_reversal_v1_1_outcome_immutable",
    }.isdisjoint(trigger_names)
    assert legacy_gap_count == (17,)


def test_migration_makes_legacy_discovery_identity_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-discovery.sqlite3"
    create_pre_hardening_database(path)
    with sqlite3.connect(path) as connection:
        envelope = connection.execute(
            """
            INSERT INTO evidence_envelope(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort,
                source_timestamps_json, recorded_at_utc
            ) VALUES ('legacy-run', ?, 'legacy', ?, 'legacy-model',
                      'anchor-frozen-20-v1', 'anchor_frozen_20', '[]', ?)
            """,
            (
                (NOW - timedelta(days=1)).isoformat(),
                "a" * 40,
                NOW.isoformat(),
            ),
        ).lastrowid
        connection.execute(
            """
            INSERT INTO opening_reversal_contract_discovery_v1(
                envelope_id, run_id, episode_id, discovery_timestamp_utc,
                contract_source, cache_hit, candidates_inspected,
                call_con_id, put_con_id, expiry, strike, tie_break_rule,
                live_market_data_lines_consumed,
                planned_live_market_data_lines, metadata_request_ended,
                full_chain_live_subscription_created, status,
                missing_reason, audit_hash_v1, audit_json
            ) VALUES (?, 'legacy-run', 'legacy-generic-episode', ?,
                      'legacy', 0, 0, NULL, NULL, NULL, NULL, 'legacy',
                      0, 0, 1, 0, 'failed', 'legacy_unavailable', ?, '{}')
            """,
            (envelope, NOW.isoformat(), "9" * 64),
        )

    repository = ProspectiveRepository(path)
    repository.migrate()

    with repository._connect() as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="blocked_opening_reversal_contract_discovery_is_immutable",
        ):
            connection.execute(
                """
                UPDATE opening_reversal_contract_discovery_v1
                SET episode_id = 'fresh-aal'
                WHERE run_id = 'legacy-run'
                """
            )
        stored = connection.execute(
            """
            SELECT episode_id
            FROM opening_reversal_contract_discovery_v1
            WHERE run_id = 'legacy-run'
            """
        ).fetchone()
        assert tuple(stored) == ("legacy-generic-episode",)


def test_schema_newer_than_application_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    repository = ProspectiveRepository(path)
    repository.migrate()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO schema_migrations(version, applied_at_utc)
            VALUES ('9999_future_schema.sql', ?)
            """,
            (NOW.isoformat(),),
        )

    with pytest.raises(SchemaVersionTooNew, match="blocked_schema_newer_than_supported"):
        repository.migrate()


def test_online_backup_includes_durable_inbox_and_runtime_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(path)
    repository.migrate()
    run_metadata = metadata("hardening-backup")
    repository.create_run(run_metadata)
    lease = repository.acquire_recorder_lease(
        run_id=run_metadata.run_id,
        owner_id="backup-recorder",
        now=NOW,
        stale_after=timedelta(seconds=30),
    )
    operational = RecorderOperationalRepository(path)
    operational.start_generation(
        run_id=run_metadata.run_id,
        recorder_generation=lease.generation,
        owner_id=lease.owner_id,
        started_at=NOW,
        required_market_data_mode="LIVE",
        expected_artifact_count=1,
    )
    inbox = DurableCallbackInbox(
        path,
        run_id=run_metadata.run_id,
        recorder_generation=lease.generation,
        owner_id=lease.owner_id,
    )
    inbox.admit(
        inbox_event_id="backup-event",
        callback_kind="realtime_bar",
        request_id=101,
        received_utc=NOW,
        received_monotonic_ns=123,
        payload={"close": None},
        connection_generation=1,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        subscription_owner="core-bars",
        symbol="AAL",
    )
    leased = inbox.lease(
        lease_owner=lease.owner_id,
        lease_generation=lease.generation,
        now=NOW,
        lease_timeout=timedelta(seconds=30),
        limit=1,
    )
    inbox.commit_raw_materialization(
        leased,
        run_id=run_metadata.run_id,
        recorder_generation=lease.generation,
        raw_partition_hashes=("e" * 64,),
        raw_event_ids=("raw-backup-event",),
        materialized_at=NOW,
    )
    inbox.release(
        leased,
        lease_owner=lease.owner_id,
        lease_generation=lease.generation,
        now=NOW,
    )
    operational.record_artifact_verification(
        RuntimeArtifactVerification(
            verification_id="backup-verification",
            run_id=run_metadata.run_id,
            recorder_generation=lease.generation,
            artifact_bundle_id="bundle",
            artifact_name="model.json",
            expected_hash="d" * 64,
            observed_hash="d" * 64,
            feature_contract_version="contract-v1",
            activation_receipt_identity="activation",
            found=True,
            loaded=True,
            schema_validated=True,
            hash_verified=True,
            contract_compatible=True,
            used_by_active_generation=True,
            load_timestamp_utc=NOW,
            verification_result="verified",
            blocker=None,
        )
    )

    backup = backup_database(path, tmp_path / "backups", now=NOW)

    with sqlite3.connect(backup.backup_file) as connection:
        assert connection.execute(
            "SELECT inbox_event_id, status FROM callback_inbox_v1"
        ).fetchone() == ("backup-event", "pending")
        assert connection.execute(
            """
            SELECT verification_result
            FROM runtime_artifact_verification_v1
            """
        ).fetchone() == ("verified",)
        assert connection.execute(
            """
            SELECT raw_partition_hashes_json
            FROM callback_raw_materialization_v1
            """
        ).fetchone() == ('["eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"]',)
