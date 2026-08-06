from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from stocker_prospective.database import (
    RECORDER_SQLITE_BUSY_TIMEOUT_MS,
    EvidenceMetadata,
    ProspectiveRepository,
)
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.operational_state import (
    GapIncident,
    OperationalStateProjection,
    OperationalThresholds,
    RecorderOperationalRepository,
    RecorderOperationalState,
    stable_gap_id,
)
from stocker_prospective.read_store import ProspectiveReadStore

NOW = datetime(2026, 7, 29, 15, 45, tzinfo=UTC)
RUN_ID = "run-operational-state"
THRESHOLDS = OperationalThresholds(
    lease_stale_after=timedelta(seconds=30),
    process_heartbeat_stale_after=timedelta(seconds=30),
    callback_heartbeat_stale_after=timedelta(seconds=30),
    raw_storage_heartbeat_stale_after=timedelta(seconds=30),
    acknowledgement_stale_after=timedelta(seconds=30),
    maximum_inbox_backlog=10,
    maximum_oldest_unacknowledged_age=timedelta(seconds=30),
)


def metadata() -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=RUN_ID,
        prospective_start_utc=NOW - timedelta(days=1),
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[NOW.isoformat()],
        recorded_at_utc=NOW,
    )


def repository(tmp_path: Path) -> ProspectiveRepository:
    value = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    value.migrate()
    value.create_run(metadata())
    return value


def test_recorder_writer_connections_use_extended_busy_timeout(tmp_path: Path) -> None:
    database_path = tmp_path / "writer-contention.sqlite3"
    repository = ProspectiveRepository(database_path)
    operational = RecorderOperationalRepository(database_path)

    with repository._connect() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == (
            RECORDER_SQLITE_BUSY_TIMEOUT_MS
        )
    with operational._connect() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == (
            RECORDER_SQLITE_BUSY_TIMEOUT_MS
        )


@pytest.mark.parametrize(
    "factory",
    (ProspectiveRepository, RecorderOperationalRepository),
)
def test_recorder_writer_busy_timeout_must_be_positive(
    tmp_path: Path,
    factory: type[ProspectiveRepository] | type[RecorderOperationalRepository],
) -> None:
    with pytest.raises(ValueError, match="busy timeout must be positive"):
        factory(tmp_path / "invalid.sqlite3", busy_timeout_ms=0)


def active_repository(
    tmp_path: Path,
) -> tuple[ProspectiveRepository, RecorderOperationalRepository]:
    value = repository(tmp_path)
    lease = value.acquire_recorder_lease(
        run_id=RUN_ID,
        owner_id="recorder",
        now=NOW,
        stale_after=timedelta(seconds=30),
    )
    operational = RecorderOperationalRepository(value.database_path)
    operational.start_generation(
        run_id=RUN_ID,
        recorder_generation=lease.generation,
        owner_id="recorder",
        started_at=NOW,
        required_market_data_mode="live",
        expected_artifact_count=1,
    )
    return value, operational


def set_signals(
    value: ProspectiveRepository,
    *,
    process_at: datetime = NOW,
    callback_at: datetime | None = NOW,
    raw_at: datetime | None = NOW,
    ack_at: datetime | None = NOW,
    completed_bar_at: datetime | None = NOW,
    market_open: bool = True,
    callbacks_expected: bool = True,
) -> None:
    with value._connect() as connection:
        connection.execute(
            """
            UPDATE recorder_operational_state_v1
            SET process_heartbeat_at_utc = ?,
                latest_callback_received_at_utc = ?,
                latest_callback_durably_admitted_at_utc = ?,
                latest_raw_partition_committed_at_utc = ?,
                latest_inbox_acknowledgement_at_utc = ?,
                latest_completed_five_minute_bar_at_utc = ?,
                market_session_open = ?,
                callbacks_expected = ?,
                ibkr_connection_state = 'connected',
                observed_market_data_mode = 'live',
                scientific_prerequisites_valid = 1,
                frozen_artifacts_verified = 1,
                inbox_backlog = 0,
                oldest_unacknowledged_at_utc = NULL,
                updated_at_utc = ?
            WHERE run_id = ?
            """,
            (
                process_at.isoformat(),
                None if callback_at is None else callback_at.isoformat(),
                None if callback_at is None else callback_at.isoformat(),
                None if raw_at is None else raw_at.isoformat(),
                None if ack_at is None else ack_at.isoformat(),
                None if completed_bar_at is None else completed_bar_at.isoformat(),
                int(market_open),
                int(callbacks_expected),
                NOW.isoformat(),
                RUN_ID,
            ),
        )


def projection(value: ProspectiveRepository) -> OperationalStateProjection:
    return ProspectiveReadStore(
        value.database_path,
        run_id=RUN_ID,
    ).recorder_operational_state(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )


def test_old_historical_run_without_operational_generation_is_inactive(
    tmp_path: Path,
) -> None:
    value = repository(tmp_path)

    observed = projection(value)

    assert observed.state is RecorderOperationalState.INACTIVE
    assert not observed.healthy


def test_no_run_is_inactive(tmp_path: Path) -> None:
    value = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    value.migrate()

    observed = ProspectiveReadStore(
        value.database_path,
    ).recorder_operational_state(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )

    assert observed.state is RecorderOperationalState.INACTIVE
    assert observed.conditions["fresh_recorder_lease"] is False


def test_fresh_lease_and_heartbeats_are_recording_healthy(tmp_path: Path) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)

    observed = projection(value)

    assert observed.state is RecorderOperationalState.RECORDING_HEALTHY
    assert observed.healthy
    assert observed.conditions["fresh_recorder_lease"] is True


def test_fresh_inbox_is_degraded_when_provider_bars_are_two_hours_behind(
    tmp_path: Path,
) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)
    with value._connect() as connection:
        connection.execute(
            """
            UPDATE recorder_operational_state_v1
            SET latest_completed_five_minute_bar_at_utc = ?
            WHERE run_id = ?
            """,
            ((NOW - timedelta(hours=2)).isoformat(), RUN_ID),
        )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.RECORDING_DEGRADED
    assert observed.reason_code == "PROVIDER_BAR_PROGRESS_STALE"
    assert observed.conditions["provider_bar_progress_fresh"] is False


@pytest.mark.parametrize(
    "heartbeat",
    [
        "not-a-timestamp",
        NOW.replace(tzinfo=None).isoformat(),
    ],
)
def test_malformed_or_naive_lease_heartbeat_is_stale(
    tmp_path: Path,
    heartbeat: str,
) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)
    with value._connect() as connection:
        connection.execute(
            "UPDATE recorder_lease SET heartbeat_at_utc = ?",
            (heartbeat,),
        )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.STALE_HEARTBEAT
    assert observed.conditions["fresh_recorder_lease"] is False


def test_future_prospective_start_waits_despite_fresh_heartbeats(
    tmp_path: Path,
) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)

    observed = ProspectiveReadStore(
        value.database_path,
        run_id=RUN_ID,
    ).recorder_operational_state(
        now=NOW,
        prospective_start_utc=NOW + timedelta(hours=1),
        thresholds=THRESHOLDS,
    )

    assert observed.state is RecorderOperationalState.WAITING_FOR_PROSPECTIVE_START
    assert observed.scientific_recording_valid


def test_active_scientific_blocker_prevents_recording_healthy(
    tmp_path: Path,
) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)
    with value._connect() as connection:
        connection.execute(
            """
            UPDATE recorder_operational_state_v1
            SET scientific_prerequisites_valid = 0
            WHERE run_id = ?
            """,
            (RUN_ID,),
        )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.SCIENTIFICALLY_BLOCKED
    assert observed.reason_code == "SCIENTIFIC_PREREQUISITES_INVALID"
    assert not observed.healthy


def test_startup_fatal_is_visible_even_before_operational_generation_exists(
    tmp_path: Path,
) -> None:
    value = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    value.migrate()
    inbox = DurableCallbackInbox(
        value.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )
    inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="CALLBACK_DURABLE_ADMISSION_FAILED",
        occurred_at=NOW,
        error_class="OperationalError",
        evidence_loss_possible=True,
    )

    observed = ProspectiveReadStore(
        value.database_path,
        run_id=RUN_ID,
    ).recorder_operational_state(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )

    assert observed.state is RecorderOperationalState.INGESTION_FATAL
    assert observed.reason_code == "CALLBACK_DURABLE_ADMISSION_FAILED"
    assert observed.run_id == RUN_ID


def test_stale_lease_is_stale_not_recording(tmp_path: Path) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value)
    with value._connect() as connection:
        connection.execute(
            "UPDATE recorder_lease SET heartbeat_at_utc = ?",
            ((NOW - timedelta(minutes=2)).isoformat(),),
        )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.STALE_HEARTBEAT
    assert not observed.healthy


def test_clean_stop_cannot_be_overridden_by_a_still_fresh_lease(
    tmp_path: Path,
) -> None:
    value, operational = active_repository(tmp_path)
    set_signals(value)
    operational.set_stopping(
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
        now=NOW,
    )
    operational.set_stopped_cleanly(
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
        now=NOW,
        termination_reason="operator_shutdown",
    )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.STOPPED_CLEANLY
    assert not observed.healthy


def test_fresh_process_but_stale_callback_is_degraded_when_market_open(
    tmp_path: Path,
) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value, callback_at=NOW - timedelta(minutes=2))

    observed = projection(value)

    assert observed.state is RecorderOperationalState.RECORDING_DEGRADED
    assert observed.reason_code == "CALLBACK_HEARTBEAT_STALE"


def test_closed_market_does_not_require_callback_heartbeat(tmp_path: Path) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(
        value,
        callback_at=None,
        raw_at=None,
        ack_at=None,
        market_open=False,
        callbacks_expected=False,
    )

    observed = projection(value)

    assert observed.state is RecorderOperationalState.MARKET_CLOSED
    assert observed.scientific_recording_valid


def test_stale_storage_heartbeat_is_degraded(tmp_path: Path) -> None:
    value, _ = active_repository(tmp_path)
    set_signals(value, raw_at=NOW - timedelta(minutes=2))

    observed = projection(value)

    assert observed.state is RecorderOperationalState.RECORDING_DEGRADED
    assert observed.reason_code == "RAW_STORAGE_HEARTBEAT_STALE"


def gap(
    *,
    symbol: str,
    start: datetime,
    severity: Literal["optional", "degraded", "scientific"] = "scientific",
) -> GapIncident:
    gap_id = stable_gap_id(
        run_id=RUN_ID,
        recorder_generation=1,
        symbol=symbol,
        stream_kind="required_market_stream",
        request_id=7,
        connection_generation=1,
        start_timestamp_utc=start,
        cause_code="CONNECTION_INTERRUPTION",
    )
    return GapIncident(
        gap_id=gap_id,
        run_id=RUN_ID,
        recorder_generation=1,
        symbol=symbol,
        stream_kind="required_market_stream",
        request_id=7,
        connection_generation=1,
        start_timestamp_utc=start,
        detection_timestamp_utc=start,
        cause_code="CONNECTION_INTERRUPTION",
        severity=severity,
        recoverability="recoverable",
    )


def test_one_gap_is_counted_once_and_resolution_moves_the_count(
    tmp_path: Path,
) -> None:
    value, operational = active_repository(tmp_path)
    set_signals(value)
    first = gap(symbol="AAL", start=NOW - timedelta(minutes=1))
    operational.record_gap(first)
    operational.record_gap(first)
    read = ProspectiveReadStore(value.database_path, run_id=RUN_ID)

    active = read.recorder_status_v0(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )
    assert active["gaps"]["active_gaps"] == 1
    assert active["gaps"]["unresolved_scientific_gaps"] == 1
    assert projection(value).state is RecorderOperationalState.SCIENTIFICALLY_BLOCKED

    operational.resolve_gap(
        gap_id=first.gap_id,
        run_id=RUN_ID,
        recorder_generation=1,
        resolved_at=NOW,
        resolution_evidence="complete bar observed",
    )
    resolved = read.recorder_status_v0(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )
    assert resolved["gaps"]["active_gaps"] == 0
    assert resolved["gaps"]["resolved_recoverable_gaps"] == 1


def test_two_distinct_gaps_count_as_two_and_legacy_partition_counts_do_not_multiply(
    tmp_path: Path,
) -> None:
    value, operational = active_repository(tmp_path)
    set_signals(value)
    operational.record_gap(gap(symbol="AAL", start=NOW - timedelta(minutes=2)))
    operational.record_gap(gap(symbol="AAPL", start=NOW - timedelta(minutes=1)))
    with value._connect() as connection:
        for index in range(3):
            connection.execute(
                """
                INSERT INTO raw_partition_manifest_v0(
                    run_id, data_source, session_date, symbol, event_type,
                    file_path, row_count, minimum_timestamp_utc,
                    maximum_timestamp_utc, schema_version, content_hash,
                    complete, gap_count, recorder_version, contract_version,
                    recorded_at_utc, claims_json
                ) VALUES (?, 'ibkr', '2026-07-29', 'AAL', 'test', ?, 1, ?, ?,
                          'test', ?, 1, 99, 'test', 'test', ?, '{}')
                """,
                (
                    RUN_ID,
                    f"/synthetic/partition-{index}",
                    NOW.isoformat(),
                    NOW.isoformat(),
                    f"{index:064x}",
                    NOW.isoformat(),
                ),
            )

    status = ProspectiveReadStore(
        value.database_path,
        run_id=RUN_ID,
    ).recorder_status_v0(
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )

    assert status["data_gaps"] == 2
    assert status["gaps"]["active_gaps"] == 2
    assert status["gaps"]["connection_interruptions"] == 2


def test_authoritative_projection_is_persisted_by_current_generation(
    tmp_path: Path,
) -> None:
    value, operational = active_repository(tmp_path)
    set_signals(value)

    projected = operational.refresh_projection(
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
        now=NOW,
        prospective_start_utc=NOW - timedelta(days=1),
        thresholds=THRESHOLDS,
    )

    assert projected.state is RecorderOperationalState.RECORDING_HEALTHY
    with value._connect() as connection:
        row = connection.execute(
            """
            SELECT state, state_reason_code, scientific_recording_valid
            FROM recorder_operational_state_v1
            WHERE run_id = ?
            """,
            (RUN_ID,),
        ).fetchone()
    assert row is not None
    assert row["state"] == "RECORDING_HEALTHY"
    assert row["state_reason_code"] == "ALL_RECORDING_HEALTH_CONDITIONS_MET"
    assert row["scientific_recording_valid"] == 1
