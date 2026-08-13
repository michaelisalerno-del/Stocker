from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.microstructure_direction_v0 import (
    RESEARCH_LABEL_V0,
    DepthStatusV0,
    DirectionActionV0,
    DirectionMethodV0,
    MicrostructureDirectionResultV0,
    TickByTickStatusV0,
)
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository

NOW = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)


def _metadata() -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id="run-direction-v0",
        prospective_start_utc=NOW,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="frozen-m1c",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[NOW.isoformat()],
        recorded_at_utc=NOW,
    )


def _result() -> MicrostructureDirectionResultV0:
    return MicrostructureDirectionResultV0(
        episode_id="episode-direction-v0",
        run_id="run-direction-v0",
        symbol="AAL",
        trigger_timestamp_utc=NOW,
        decision_timestamp_utc=NOW,
        information_cutoff_utc=NOW,
        confirmation_delay_seconds=0.0,
        window_name="T-5s_to_T0",
        direction_method=DirectionMethodV0.M01,
        action=DirectionActionV0.UP,
        signed_score=0.5,
        component_values={"trade_imbalance": 0.5},
        component_validity={"trade_imbalance": True},
        quote_count=5,
        trade_count=4,
        classified_trade_count=3,
        trade_classification_valid_fraction=0.75,
        stale_quote_fraction=0.0,
        unknown_trade_volume_fraction=0.1,
        probable_buyer_initiated_volume=75.0,
        probable_seller_initiated_volume=25.0,
        tick_by_tick_status=TickByTickStatusV0.BIDASK_AND_LAST_PRESENT,
        depth_status=DepthStatusV0.ABSENT,
        market_data_type="live",
        data_quality_flags=(),
        causal_valid=True,
    )


def _seed_parents(database: ProspectiveRepository) -> int:
    metadata = _metadata()
    database.create_run(metadata)
    with sqlite3.connect(database.database_path) as connection:
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
            ) VALUES (?, 999, 999, ?, 'AAL', '2026-08-06', 6, ?, ?, 0.6,
                      0.4, 1, NULL, 1, '[]', 'prospective_recording',
                      'active', NULL, '{}')
            """,
            (
                "episode-direction-v0",
                metadata.run_id,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        cursor = connection.execute(
            """
            INSERT INTO microstructure_summary_v0(
                envelope_id, run_id, episode_id, symbol, window_name,
                window_start_utc, window_end_utc, calculated_at_utc,
                level1_valid, tick_valid, depth_valid, summary_json,
                component_json, archetype_relationship_json,
                quality_flags_json, claims_json
            ) VALUES (999, ?, ?, 'AAL', 'T-5s_to_T0', ?, ?, ?,
                      1, 1, 0, '{}', '{}', '{}', '[]', '{}')
            """,
            (
                metadata.run_id,
                "episode-direction-v0",
                (NOW.replace(microsecond=0).timestamp() - 5),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)


def test_additive_direction_rows_are_immutable_and_readable(tmp_path) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    summary_id = _seed_parents(database)
    recorder = FrozenRecorderRepository(database)

    first = recorder.record_microstructure_direction_v0(
        _metadata(),
        microstructure_summary_id=summary_id,
        results=(_result(),),
    )
    repeated = recorder.record_microstructure_direction_v0(
        _metadata(),
        microstructure_summary_id=summary_id,
        results=(_result(),),
    )

    assert first == repeated
    projected = ProspectiveReadStore(
        database.database_path,
        run_id="run-direction-v0",
    ).episode_microstructure_direction_v0("episode-direction-v0")
    assert len(projected) == 1
    assert projected[0]["direction_method"] == DirectionMethodV0.M01.value
    assert projected[0]["action"] == "UP"
    assert projected[0]["component_values"] == {"trade_imbalance": 0.5}
    assert projected[0]["research_label"] == RESEARCH_LABEL_V0


def test_direction_migration_keeps_research_rows_separate(tmp_path) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()

    with database._connect() as connection:
        tables = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")
        }
        methods = {
            str(row["sql"])
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name = 'microstructure_direction_v0'"
            )
        }

    assert "microstructure_direction_v0" in tables
    assert "direction_classification_v0" in tables
    assert methods
    assert all("A1" not in sql and "C1" not in sql and "R1" not in sql for sql in methods)
