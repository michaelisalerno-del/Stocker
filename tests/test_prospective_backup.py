from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from stocker_prospective.backup import backup_database
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay

ROOT = Path(__file__).parents[1]


def test_online_backup_is_checked_and_preserves_prospective_records(tmp_path: Path) -> None:
    database = tmp_path / "shared/prospective.sqlite3"
    run_deterministic_replay(
        ReplaySettings(
            database_path=database,
            run_id="backup-test",
            prospective_start_utc=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
            app_version="test",
            git_commit="deadbeef",
            universe_path=ROOT / "configs/prospective/anchor-frozen-20.json",
            owner_id="backup-test-owner",
            recorder_lease_stale_seconds=60,
        )
    )

    backup = backup_database(
        database,
        tmp_path / "backups",
        now=datetime(2026, 7, 25, 0, 0, tzinfo=UTC),
    )

    assert backup.quick_check == "ok"
    assert backup.backup_file.is_file()
    assert backup.manifest_file.is_file()
    with sqlite3.connect(backup.backup_file) as connection:
        count = connection.execute("SELECT COUNT(*) FROM signal_episode").fetchone()
    assert count == (2,)
