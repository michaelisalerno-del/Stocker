from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import stocker_prospective.read_store as read_store_module
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.operational_logging import (
    begin_request_metrics,
    reset_request_metrics,
)
from stocker_prospective.read_store import ProspectiveReadStore

RUN_ID = "synthetic-freshness-run"
CURRENT_SESSION = "2026-07-30"
PREVIOUS_SESSION = "2026-07-29"


def insert_envelope(connection: sqlite3.Connection, *, recorded_at: str) -> int:
    cursor = connection.execute(
        """
        INSERT INTO evidence_envelope(
            run_id, prospective_start_utc, app_version, git_commit,
            model_artifact_id, universe_id, cohort, source_timestamps_json,
            recorded_at_utc
        ) VALUES (?, ?, 'test', 'deadbeef', 'synthetic-model',
                  'synthetic-universe', 'anchor_frozen_20', '[]', ?)
        """,
        (RUN_ID, "2026-07-01T00:00:00+00:00", recorded_at),
    )
    assert cursor.lastrowid is not None
    return int(cursor.lastrowid)


def insert_checkpoint(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    session: str,
    checkpoint: int,
    complete: bool = True,
) -> int:
    bar_end = f"{session}T14:{checkpoint * 5:02d}:00+00:00"
    envelope_id = insert_envelope(connection, recorded_at=bar_end)
    cursor = connection.execute(
        """
        INSERT INTO m1c_checkpoint_v0(
            envelope_id, run_id, symbol, session_date, checkpoint,
            bar_start_utc, bar_end_utc, feature_as_of_utc, model_id,
            model_version, model_hash, feature_hash, session_context_hash,
            feature_values_json, probability, threshold, threshold_passed,
            eligible, feature_freshness, missing_feature_count,
            rejection_reasons_json, claims_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'M1C', 'synthetic',
                  'model-hash', 'feature-hash', 'context-hash', '{}',
                  0.60, 0.50, 1, 1, 'fresh', 0, '[]', '{}')
        """,
        (
            envelope_id,
            RUN_ID,
            symbol,
            session,
            checkpoint,
            f"{session}T14:{checkpoint * 5 - 5:02d}:00+00:00",
            bar_end,
            f"{session}T14:{checkpoint * 5 - 5:02d}:00+00:00",
        ),
    )
    assert cursor.lastrowid is not None
    checkpoint_id = int(cursor.lastrowid)
    if complete:
        completion_envelope = insert_envelope(connection, recorded_at=bar_end)
        connection.execute(
            """
            INSERT INTO m1c_checkpoint_completion_v0(
                checkpoint_id, envelope_id, run_id, symbol, session_date,
                checkpoint, completed_at_utc, claims_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                checkpoint_id,
                completion_envelope,
                RUN_ID,
                symbol,
                session,
                checkpoint,
                bar_end,
            ),
        )
    return checkpoint_id


def insert_episode(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    session: str,
    checkpoint: int,
    checkpoint_id: int,
    schedule_status: str,
) -> str:
    episode_id = f"{symbol}-{session}-{checkpoint}"
    bar_end = f"{session}T14:{checkpoint * 5:02d}:00+00:00"
    envelope_id = insert_envelope(connection, recorded_at=bar_end)
    connection.execute(
        """
        INSERT INTO m1c_episode_v0(
            episode_id, envelope_id, checkpoint_id, run_id, symbol,
            session_date, trigger_checkpoint, trigger_bar_end_utc,
            prospective_entry_timestamp_utc, m1c_probability,
            previous_m1c_probability, episode_number,
            minutes_since_previous_episode, scientific_recording_valid,
            rejection_reasons_json, phase, completion_status,
            completed_at_utc, claims_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.60, 0.40, 1, NULL, 1,
                  '[]', 'synthetic', ?, ?, '{}')
        """,
        (
            episode_id,
            envelope_id,
            checkpoint_id,
            RUN_ID,
            symbol,
            session,
            checkpoint,
            bar_end,
            bar_end,
            "complete" if schedule_status == "complete" else "active",
            bar_end if schedule_status == "complete" else None,
        ),
    )
    schedule_envelope = insert_envelope(connection, recorded_at=bar_end)
    connection.execute(
        """
        INSERT INTO option_episode_schedule_v0(
            episode_id, checkpoint_id, envelope_id, run_id, symbol,
            session_date, entry_timestamp_utc, episode_kind, probability,
            quiet_state, directional_actions_json, recording_duration_seconds,
            strike_steps, maximum_contracts, status, updated_at_utc, claims_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'high_tail', 0.60, 0, '{}',
                  1800, 2, 8, ?, ?, '{}')
        """,
        (
            episode_id,
            checkpoint_id,
            schedule_envelope,
            RUN_ID,
            symbol,
            session,
            bar_end,
            schedule_status,
            bar_end,
        ),
    )
    return episode_id


def seed_freshness_database(database: Path) -> None:
    ProspectiveRepository(database).migrate()
    symbols = (
        "PREVIOUS",
        "LATEST",
        "OLDER",
        "COMPLETED",
        "REJECTED",
        "EXPIRED",
        "NONE",
        "MULTIPLE",
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO prospective_run(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, created_at_utc,
                mode, status, scientific_classification
            ) VALUES (?, '2026-07-01T00:00:00+00:00', 'test', 'deadbeef',
                      'synthetic-model', 'synthetic-universe',
                      'anchor_frozen_20', '2026-07-01T00:00:00+00:00',
                      'record_only', 'active', 'synthetic_test_only')
            """,
            (RUN_ID,),
        )
        connection.execute(
            """
            INSERT INTO runtime_session(
                run_id, session_date, opened_at_utc, status
            ) VALUES (?, ?, '2026-07-30T13:00:00+00:00', 'recording')
            """,
            (RUN_ID, CURRENT_SESSION),
        )
        connection.executemany(
            """
            INSERT INTO universe_membership(
                run_id, universe_id, cohort, symbol, operational_status,
                rejection_reason, recorded_at_utc
            ) VALUES (?, 'synthetic-universe', 'anchor_frozen_20', ?,
                      'active', NULL, '2026-07-30T13:00:00+00:00')
            """,
            ((RUN_ID, symbol) for symbol in symbols),
        )

        previous_checkpoint = insert_checkpoint(
            connection,
            symbol="PREVIOUS",
            session=PREVIOUS_SESSION,
            checkpoint=1,
        )
        insert_episode(
            connection,
            symbol="PREVIOUS",
            session=PREVIOUS_SESSION,
            checkpoint=1,
            checkpoint_id=previous_checkpoint,
            schedule_status="complete",
        )
        insert_checkpoint(
            connection,
            symbol="PREVIOUS",
            session=CURRENT_SESSION,
            checkpoint=1,
        )

        latest_checkpoint = insert_checkpoint(
            connection,
            symbol="LATEST",
            session=CURRENT_SESSION,
            checkpoint=1,
        )
        insert_episode(
            connection,
            symbol="LATEST",
            session=CURRENT_SESSION,
            checkpoint=1,
            checkpoint_id=latest_checkpoint,
            schedule_status="streaming",
        )

        older_checkpoint = insert_checkpoint(
            connection,
            symbol="OLDER",
            session=CURRENT_SESSION,
            checkpoint=1,
        )
        insert_episode(
            connection,
            symbol="OLDER",
            session=CURRENT_SESSION,
            checkpoint=1,
            checkpoint_id=older_checkpoint,
            schedule_status="complete",
        )
        insert_checkpoint(
            connection,
            symbol="OLDER",
            session=CURRENT_SESSION,
            checkpoint=2,
        )
        insert_checkpoint(
            connection,
            symbol="OLDER",
            session=CURRENT_SESSION,
            checkpoint=3,
            complete=False,
        )

        for symbol, schedule_status in (
            ("COMPLETED", "complete"),
            ("REJECTED", "rejected"),
            ("EXPIRED", "expired"),
        ):
            checkpoint_id = insert_checkpoint(
                connection,
                symbol=symbol,
                session=CURRENT_SESSION,
                checkpoint=1,
            )
            insert_episode(
                connection,
                symbol=symbol,
                session=CURRENT_SESSION,
                checkpoint=1,
                checkpoint_id=checkpoint_id,
                schedule_status=schedule_status,
            )

        insert_checkpoint(
            connection,
            symbol="NONE",
            session=CURRENT_SESSION,
            checkpoint=1,
        )

        first_multiple = insert_checkpoint(
            connection,
            symbol="MULTIPLE",
            session=CURRENT_SESSION,
            checkpoint=1,
        )
        insert_episode(
            connection,
            symbol="MULTIPLE",
            session=CURRENT_SESSION,
            checkpoint=1,
            checkpoint_id=first_multiple,
            schedule_status="complete",
        )
        latest_multiple = insert_checkpoint(
            connection,
            symbol="MULTIPLE",
            session=CURRENT_SESSION,
            checkpoint=2,
        )
        insert_episode(
            connection,
            symbol="MULTIPLE",
            session=CURRENT_SESSION,
            checkpoint=2,
            checkpoint_id=latest_multiple,
            schedule_status="complete",
        )


def test_fresh_episode_requires_current_session_latest_completed_checkpoint_and_valid_status(
    tmp_path: Path,
) -> None:
    database = tmp_path / "synthetic.sqlite3"
    seed_freshness_database(database)

    items = {
        item["symbol"]: item
        for item in ProspectiveReadStore(database, run_id=RUN_ID).universe_live_v0()
    }

    assert items["PREVIOUS"]["has_historical_episode"] is True
    assert items["PREVIOUS"]["fresh_episode"] is False
    assert items["PREVIOUS"]["latest_episode_session_date"] == PREVIOUS_SESSION

    assert items["LATEST"]["fresh_episode"] is True
    assert items["LATEST"]["latest_episode_status"] == "streaming"

    assert items["OLDER"]["has_historical_episode"] is True
    assert items["OLDER"]["fresh_episode"] is False

    assert items["COMPLETED"]["fresh_episode"] is True
    assert items["COMPLETED"]["latest_episode_status"] == "complete"

    assert items["REJECTED"]["fresh_episode"] is False
    assert items["REJECTED"]["latest_episode_status"] == "rejected"
    assert items["EXPIRED"]["fresh_episode"] is False
    assert items["EXPIRED"]["latest_episode_status"] == "expired"

    assert items["NONE"]["has_historical_episode"] is False
    assert items["NONE"]["fresh_episode"] is False
    assert items["NONE"]["latest_episode_id"] is None

    assert items["MULTIPLE"]["fresh_episode"] is True
    assert items["MULTIPLE"]["latest_episode_id"] == (f"MULTIPLE-{CURRENT_SESSION}-2")


def test_recorder_latest_event_summary_work_is_independent_of_manifest_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "synthetic.sqlite3"
    seed_freshness_database(database)
    rows = [
        (
            RUN_ID,
            CURRENT_SESSION,
            "LATEST",
            f"/synthetic/not-opened/{index}.parquet",
            f"manifest-{index:05d}",
            (datetime(2026, 7, 30, 14, 0, tzinfo=UTC) + timedelta(seconds=index)).isoformat(),
            index % 3,
        )
        for index in range(5_000)
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'synthetic', ?, ?,
                      'underlying_level1_quote_event', ?, 1, ?, ?,
                      'test', ?, 1, ?, 'test', 'test', ?, '{}')
            """,
            [
                (
                    run_id,
                    session,
                    symbol,
                    file_path,
                    observed_at,
                    observed_at,
                    content_hash,
                    gap_count,
                    observed_at,
                )
                for (
                    run_id,
                    session,
                    symbol,
                    file_path,
                    content_hash,
                    observed_at,
                    gap_count,
                ) in rows
            ],
        )

    projected = ProspectiveReadStore(
        database,
        run_id=RUN_ID,
    ).recorder_status_v0()

    assert projected["last_event_timestamp"] == rows[-1][5]
    assert projected["data_gaps"] == sum(row[6] for row in rows)
    progress_steps = 0

    def count_progress() -> int:
        nonlocal progress_steps
        progress_steps += 1
        return 0

    with sqlite3.connect(database) as connection:
        connection.execute(
            "SELECT 1 FROM web_run_event_summary_v0 WHERE run_id = ?",
            (RUN_ID,),
        ).fetchone()
        connection.set_progress_handler(count_progress, 1)
        summary = connection.execute(
            """
            SELECT last_event_timestamp, data_gaps
            FROM web_run_event_summary_v0
            WHERE run_id = ?
            """,
            (RUN_ID,),
        ).fetchone()
        connection.set_progress_handler(None, 0)
    assert summary == (rows[-1][5], sum(row[6] for row in rows))
    assert progress_steps < 50


def test_quote_chart_projection_is_windowed_column_bounded_and_preserves_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "synthetic.sqlite3"
    seed_freshness_database(database)
    partition = tmp_path / "enlarged-quotes.parquet"
    episode_start = datetime(2026, 7, 30, 14, 5, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for group in range(9):
        for index in range(100):
            observed = episode_start - timedelta(days=20 - group, seconds=index)
            rows.append(
                {
                    "event_id": f"old-{group}-{index}",
                    "provider_timestamp_utc": observed.isoformat(),
                    "received_timestamp_utc": observed.isoformat(),
                    "bid": 90.0,
                    "ask": 90.1,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                    "ignored_payload": "x" * 2048,
                }
            )
    for index in range(100):
        observed = episode_start + timedelta(seconds=index)
        rows.append(
            {
                "event_id": f"window-{index:03d}",
                "provider_timestamp_utc": observed.isoformat(),
                "received_timestamp_utc": observed.isoformat(),
                "bid": 100.0 + index / 1000,
                "ask": 100.1 + index / 1000,
                "bid_size": 10.0,
                "ask_size": 12.0,
                "ignored_payload": "x" * 2048,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows),
        partition,
        row_group_size=100,
        compression=None,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'synthetic', ?, 'LATEST',
                      'underlying_level1_quote_event', ?, ?, ?, ?,
                      'test', 'quote-window-partition', 1, 0, 'test', 'test',
                      ?, '{}')
            """,
            (
                RUN_ID,
                CURRENT_SESSION,
                str(partition),
                len(rows),
                str(rows[0]["received_timestamp_utc"]),
                str(rows[-1]["received_timestamp_utc"]),
                str(rows[-1]["received_timestamp_utc"]),
            ),
        )

    observed_metrics = []
    original_read = read_store_module.read_parquet_window

    def tracking_read(*args: object, **kwargs: object) -> object:
        projection = original_read(*args, **kwargs)
        observed_metrics.append(projection.metrics)
        return projection

    monkeypatch.setattr(read_store_module, "read_parquet_window", tracking_read)
    request_metrics, metrics_token = begin_request_metrics()
    try:
        projected = ProspectiveReadStore(
            database,
            run_id=RUN_ID,
        ).episode_quote_series_v0(
            f"LATEST-{CURRENT_SESSION}-1",
            maximum_points=10,
            maximum_input_rows=500,
        )
    finally:
        reset_request_metrics(metrics_token)

    assert len(projected) == 10
    assert projected[0]["event_id"] == "window-000"
    assert projected[-1]["event_id"] == "window-099"
    assert len(observed_metrics) == 1
    assert observed_metrics[0].row_groups_examined == 10
    assert observed_metrics[0].row_groups_read == 1
    assert "ignored_payload" not in observed_metrics[0].columns_read
    assert request_metrics.parquet_files_examined == 1
    assert request_metrics.parquet_row_groups_examined == 10
    assert request_metrics.parquet_row_groups_read == 1
    assert request_metrics.parquet_input_rows == 100
    assert request_metrics.parquet_output_rows == 100
