from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.database import ProspectiveRepository, RecorderLeaseHeld
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
from stocker_prospective.universe import UniverseError, load_registered_universe

ROOT = Path(__file__).parents[1]
UNIVERSE = ROOT / "configs/prospective/anchor-frozen-20.json"


def test_registered_anchor_universe_is_loaded_from_provenance_and_is_immutable() -> None:
    universe = load_registered_universe(UNIVERSE)

    assert universe.cohort == "anchor_frozen_20"
    assert len(universe.symbols) == 20
    assert "AAL" in universe.symbols
    assert "SMCI" in universe.symbols
    assert universe.source_artifact_sha256 == (
        "49bd22e47b20274b1fe058ae15d899fb4a3a5e18feb418f990ef5139528161b9"
    )


def test_anchor_member_cannot_be_silently_removed(tmp_path: Path) -> None:
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    payload["symbols"].pop()
    broken = tmp_path / "universe.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UniverseError, match="blocked_frozen_universe_mismatch"):
        load_registered_universe(broken)


def replay_settings(tmp_path: Path) -> ReplaySettings:
    return ReplaySettings(
        database_path=tmp_path / "shared/data/prospective.sqlite3",
        run_id="replay-run-001",
        prospective_start_utc=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
        app_version="0.1.0-test",
        git_commit="deadbeef",
        universe_path=UNIVERSE,
        owner_id="test-recorder",
        recorder_lease_stale_seconds=60,
    )


def test_deterministic_replay_is_complete_and_restart_idempotent(tmp_path: Path) -> None:
    settings = replay_settings(tmp_path)
    repository = ProspectiveRepository(settings.database_path)

    first = run_deterministic_replay(settings)
    second = run_deterministic_replay(settings)

    assert first == second
    assert first.score_label == "synthetic_replay_not_frozen_m1"
    assert first.signal_episode_count == 2
    assert first.capture_horizons_minutes == (0, 5, 10, 15, 30)
    assert first.shadow_structure_count == 10
    assert first.shadow_horizon_count == 40
    assert first.blockers == (
        "blocked_missing_verified_frozen_bundle",
        "blocked_feature_source_semantics_mismatch",
        "blocked_official_ibkr_api_not_installed",
    )
    assert repository.count("signal_episode") == 2

    with sqlite3.connect(settings.database_path) as connection:
        score_labels = {
            row[0] for row in connection.execute("SELECT DISTINCT score_label FROM model_score")
        }
        capture_statuses = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT capture_status FROM option_surface_capture"
            )
        }
        market_data_types = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT market_data_type FROM option_surface_capture "
                "WHERE market_data_type IS NOT NULL"
            )
        }
        missed = connection.execute(
            """
            SELECT actual_quote_timestamp_utc, missing_contract_reason
            FROM option_surface_capture WHERE capture_status = 'missed'
            ORDER BY id LIMIT 1
            """
        ).fetchone()
        computation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(option_quote_computation)")
        }

    assert score_labels == {"synthetic_replay_not_frozen_m1"}
    assert {"captured", "missed", "diagnostic_only"} <= capture_statuses
    assert {"live", "delayed"} <= market_data_types
    assert missed == (None, "no_expiry_in_bucket")
    assert {
        "computation_source",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
    } <= computation_columns


def test_replay_result_counts_only_the_requested_run(tmp_path: Path) -> None:
    first_settings = replay_settings(tmp_path)
    second_settings = first_settings.model_copy(
        update={
            "run_id": "replay-run-002",
            "owner_id": "second-test-recorder",
        }
    )

    first = run_deterministic_replay(first_settings)
    ProspectiveRepository(first_settings.database_path).release_recorder_lease(
        run_id=first_settings.run_id,
        owner_id=first_settings.owner_id,
    )
    second = run_deterministic_replay(second_settings)

    assert first.signal_episode_count == 2
    assert second.signal_episode_count == 2
    assert first.shadow_structure_count == 10
    assert second.shadow_structure_count == 10
    assert first.shadow_horizon_count == 40
    assert second.shadow_horizon_count == 40


def test_replay_honours_configured_lease_staleness(tmp_path: Path) -> None:
    settings = replay_settings(tmp_path)
    run_deterministic_replay(settings)
    heartbeat = datetime.now(UTC) - timedelta(seconds=45)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "UPDATE recorder_lease SET heartbeat_at_utc = ?",
            (heartbeat.isoformat(),),
        )

    competing = settings.model_copy(update={"owner_id": "competing-recorder"})
    with pytest.raises(RecorderLeaseHeld, match="blocked_recorder_lease_held"):
        run_deterministic_replay(competing)


def test_replay_refuses_records_before_explicit_prospective_start(tmp_path: Path) -> None:
    settings = replay_settings(tmp_path).model_copy(
        update={
            "prospective_start_utc": datetime(
                2026,
                7,
                24,
                15,
                0,
                tzinfo=UTC,
            )
        }
    )

    with pytest.raises(ValueError, match="prospective_start_utc"):
        run_deterministic_replay(settings)
