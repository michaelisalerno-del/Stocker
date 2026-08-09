from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

import stocker_runtime.storage.backup as backup_module
from stocker_runtime.cli import app
from stocker_runtime.storage import (
    BackupCapacityError,
    BackupError,
    BackupIntegrityError,
    BackupPolicy,
    connect_v2,
    create_backup,
    initialize_database,
    read_backup_manifests,
    restore_backup,
)


def _seed_database(path: Path, *, incidents: int = 0) -> None:
    initialize_database(path)
    with connect_v2(path) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, git_commit, "
            "data_class, status) VALUES "
            "('run-backup', 'prospective_record', 'ibkr', 1, ?, '0000000', "
            "'prospective_protected', 'running')",
            ("a" * 64,),
        )
        for number in range(incidents):
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
                "details_json) VALUES (?, 'run-backup', 'storage', 'degraded', 'TEST', ?, ?)",
                (
                    f"incident-{number}",
                    number + 1,
                    json.dumps({"padding": "x" * 512}, separators=(",", ":"), sort_keys=True),
                ),
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_online_backup_is_checked_deterministic_compressed_and_restorable_under_writes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database, incidents=4_000)

    stop_writes = threading.Event()
    writes_started = threading.Event()
    write_errors: list[BaseException] = []

    def write_during_backup() -> None:
        try:
            connection = connect_v2(database)
            number = 10_000
            try:
                while not stop_writes.is_set() and number < 11_000:
                    connection.execute(
                        "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                        "opened_at_us, details_json) VALUES (?, 'run-backup', 'storage', "
                        "'degraded', 'CONCURRENT', ?, '{}')",
                        (f"concurrent-{number}", number),
                    )
                    number += 1
                    writes_started.set()
                    stop_writes.wait(0.001)
            finally:
                connection.close()
        except BaseException as error:  # pragma: no cover - asserted below
            write_errors.append(error)

    writer = threading.Thread(target=write_during_backup)
    writer.start()
    assert writes_started.wait(timeout=5)
    artifact = create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=1_786_118_400_123_456,
    )
    stop_writes.set()
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert write_errors == []
    assert artifact.archive_path.name.endswith(".sqlite3.gz")
    assert artifact.manifest_path.name.endswith(".manifest.json")
    assert not tuple(backups.glob("*.sqlite3"))
    assert artifact.manifest.quick_check == "ok"
    assert artifact.manifest.compressed_sha256 == _sha256(artifact.archive_path)
    assert stat.S_IMODE(artifact.archive_path.stat().st_mode) == 0o640
    assert stat.S_IMODE(artifact.manifest_path.stat().st_mode) == 0o640
    assert stat.S_IMODE((backups / "backup-status.json").stat().st_mode) == 0o640
    with artifact.archive_path.open("rb") as handle:
        assert handle.read(4) == b"\x1f\x8b\x08\x00"
        assert int.from_bytes(handle.read(4), "little") == 0

    deterministic_one = create_backup(
        database,
        tmp_path / "deterministic-one",
        tier="daily",
        created_at_us=1_786_118_400_123_456,
    )
    deterministic_two = create_backup(
        database,
        tmp_path / "deterministic-two",
        tier="daily",
        created_at_us=1_786_118_400_123_456,
    )
    assert (
        deterministic_one.archive_path.read_bytes() == deterministic_two.archive_path.read_bytes()
    )
    assert (
        deterministic_one.manifest.uncompressed_sha256
        == deterministic_two.manifest.uncompressed_sha256
    )

    restored = tmp_path / "restored.sqlite3"
    result = restore_backup(artifact.manifest_path, restored)
    assert result.destination == restored
    assert result.uncompressed_sha256 == artifact.manifest.uncompressed_sha256
    assert stat.S_IMODE(restored.stat().st_mode) == 0o640
    with connect_v2(restored) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] >= 4_000

    with pytest.raises(FileExistsError):
        restore_backup(artifact.manifest_path, restored)


def test_online_backup_fails_cleanly_when_its_time_budget_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    monkeypatch.setattr(backup_module, "ONLINE_BACKUP_TIMEOUT_SECONDS", 0)

    with pytest.raises(BackupError, match="30-minute time budget"):
        create_backup(database, backups, tier="daily", created_at_us=1)

    assert not tuple(backups.glob("*.gz"))
    assert not tuple(backups.glob("*.manifest.json"))


def test_restore_verifies_both_hashes_and_never_publishes_a_partial_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    artifact = create_backup(database, backups, tier="weekly", created_at_us=2_000_000)

    compressed = bytearray(artifact.archive_path.read_bytes())
    compressed[-1] ^= 1
    artifact.archive_path.write_bytes(compressed)
    os.utime(
        artifact.archive_path,
        ns=(artifact.manifest.archive_mtime_ns, artifact.manifest.archive_mtime_ns),
    )
    destination = tmp_path / "tampered-restore.sqlite3"

    with pytest.raises(BackupIntegrityError, match="compressed hash"):
        restore_backup(artifact.manifest_path, destination)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".stocker-v2-restore-*"))

    second = create_backup(
        database,
        tmp_path / "uncompressed-hash",
        tier="weekly",
        created_at_us=3_000_000,
    )
    payload = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    payload["uncompressed_sha256"] = "0" * 64
    second.manifest_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    uncompressed_destination = tmp_path / "uncompressed-tamper.sqlite3"
    with pytest.raises(BackupIntegrityError, match="uncompressed hash"):
        restore_backup(second.manifest_path, uncompressed_destination)
    assert not uncompressed_destination.exists()


def test_rotation_enforces_daily_weekly_counts_and_preserves_floor_at_byte_cap(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    count_policy = BackupPolicy(
        daily_retention=3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=64 * 1024 * 1024,
    )

    for created_at_us in range(1, 5):
        create_backup(
            database,
            backups,
            tier="daily",
            created_at_us=created_at_us,
            policy=count_policy,
        )
    for created_at_us in range(10, 13):
        create_backup(
            database,
            backups,
            tier="weekly",
            created_at_us=created_at_us,
            policy=count_policy,
        )

    projection = read_backup_manifests(backups, limit=200)
    assert [item.tier for item in projection.items].count("daily") == 3
    assert [item.tier for item in projection.items].count("weekly") == 2
    assert {item.created_at_us for item in projection.items if item.tier == "daily"} == {2, 3, 4}
    assert {item.created_at_us for item in projection.items if item.tier == "weekly"} == {11, 12}
    assert sum(item.stat().st_size for item in backups.iterdir()) <= count_policy.byte_cap

    ordering_directory = tmp_path / "ordering"
    newer_daily = create_backup(
        database,
        ordering_directory,
        tier="daily",
        created_at_us=100,
    )
    create_backup(
        database,
        ordering_directory,
        tier="weekly",
        created_at_us=50,
    )
    ordered = read_backup_manifests(ordering_directory, limit=1)
    assert ordered.truncated is True
    assert ordered.items[0].archive_filename == newer_daily.archive_path.name

    corrupt_directory = tmp_path / "corrupt-floor"
    corrupt_one = create_backup(
        database,
        corrupt_directory,
        tier="daily",
        created_at_us=200,
        policy=count_policy,
    )
    create_backup(
        database,
        corrupt_directory,
        tier="daily",
        created_at_us=201,
        policy=count_policy,
    )
    corrupted = bytearray(corrupt_one.archive_path.read_bytes())
    corrupted[-1] ^= 1
    corrupt_one.archive_path.write_bytes(corrupted)
    create_backup(
        database,
        corrupt_directory,
        tier="daily",
        created_at_us=202,
        policy=BackupPolicy(
            daily_retention=2,
            weekly_retention=2,
            minimum_per_tier=2,
            byte_cap=64 * 1024 * 1024,
        ),
    )
    assert corrupt_one.archive_path.exists()
    assert corrupt_one.manifest_path.exists()

    floor_directory = tmp_path / "floor"
    floor_policy = BackupPolicy(
        daily_retention=14,
        weekly_retention=12,
        minimum_per_tier=2,
        byte_cap=64 * 1024 * 1024,
    )
    for tier, timestamps in (("daily", (20, 21)), ("weekly", (30, 31))):
        for created_at_us in timestamps:
            create_backup(
                database,
                floor_directory,
                tier=tier,
                created_at_us=created_at_us,
                policy=floor_policy,
            )
    names_before = {item.name for item in floor_directory.iterdir()}
    used_bytes = sum(item.stat().st_size for item in floor_directory.iterdir())
    tight_policy = BackupPolicy(
        daily_retention=14,
        weekly_retention=12,
        minimum_per_tier=2,
        byte_cap=used_bytes - 1_000,
    )

    with pytest.raises(BackupCapacityError, match="preserving two valid archives"):
        create_backup(
            database,
            floor_directory,
            tier="daily",
            created_at_us=40,
            policy=tight_policy,
        )
    assert names_before.issubset({item.name for item in floor_directory.iterdir()})
    assert read_backup_manifests(floor_directory, limit=200).status.state == "degraded"


def test_daily_and_weekly_backup_creation_share_one_destination_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    real_online_copy = backup_module._online_copy
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    errors: list[BaseException] = []

    def observed_online_copy(source: Path, destination: Path) -> None:
        nonlocal calls
        with call_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        real_online_copy(source, destination)

    monkeypatch.setattr(backup_module, "_online_copy", observed_online_copy)

    def run(tier: str, created_at_us: int) -> None:
        try:
            create_backup(
                database,
                backups,
                tier=tier,  # type: ignore[arg-type]
                created_at_us=created_at_us,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    daily = threading.Thread(target=run, args=("daily", 100))
    weekly = threading.Thread(target=run, args=("weekly", 101))
    daily.start()
    assert first_entered.wait(timeout=5)
    weekly.start()
    overlapped = second_entered.wait(timeout=0.25)
    release_first.set()
    daily.join(timeout=5)
    weekly.join(timeout=5)

    assert overlapped is False
    assert not daily.is_alive()
    assert not weekly.is_alive()
    assert errors == []
    assert len(read_backup_manifests(backups, limit=200).items) == 2


def test_backup_working_copies_stay_outside_the_capped_archive_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    real_online_copy = backup_module._online_copy
    observed_directories: list[Path] = []

    def observed_online_copy(source: Path, destination: Path) -> None:
        observed_directories.append(destination.parent)
        real_online_copy(source, destination)

    monkeypatch.setattr(backup_module, "_online_copy", observed_online_copy)
    create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=200,
        working_directory=working,
    )

    assert observed_directories == [working.resolve()]
    assert not tuple(backups.glob(".stocker-v2-database-*"))
    assert not tuple(backups.glob(".stocker-v2-archive-*"))
    assert not tuple(working.iterdir())


def test_backup_health_requires_fresh_valid_tier_floors(tmp_path: Path) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    day_us = 24 * 60 * 60 * 1_000_000
    base = 20 * day_us
    for tier, offsets in (("daily", (-day_us, 0)), ("weekly", (-7 * day_us, 0))):
        for offset in offsets:
            create_backup(
                database,
                backups,
                tier=tier,  # type: ignore[arg-type]
                created_at_us=base + offset,
            )

    healthy = read_backup_manifests(backups, limit=200, now_us=base)
    assert healthy.status.state == "healthy"
    assert healthy.status.code is None

    weekly = next(item for item in healthy.items if item.tier == "weekly")
    archive = backups / weekly.archive_filename
    corrupted = bytearray(archive.read_bytes())
    corrupted[-1] ^= 1
    archive.write_bytes(corrupted)
    invalid = read_backup_manifests(backups, limit=200, now_us=base)
    assert invalid.status.state == "degraded"
    assert invalid.status.code == "BACKUP_MANIFEST_INVALID"

    stale_backups = tmp_path / "stale"
    for tier, offsets in (("daily", (-day_us, 0)), ("weekly", (-7 * day_us, 0))):
        for offset in offsets:
            create_backup(
                database,
                stale_backups,
                tier=tier,  # type: ignore[arg-type]
                created_at_us=base + offset,
            )
    stale = read_backup_manifests(stale_backups, limit=200, now_us=base + 9 * day_us)
    assert stale.status.state == "degraded"
    assert stale.status.code == "BACKUP_DAILY_STALE"


def test_manifest_is_atomic_and_diagnostics_ignore_arbitrary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    real_replace = backup_module.os.replace

    def fail_manifest_publish(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        if str(destination).endswith(".manifest.json"):
            raise OSError("injected manifest publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(backup_module.os, "replace", fail_manifest_publish)
    with pytest.raises(OSError, match="injected manifest"):
        create_backup(database, backups, tier="daily", created_at_us=50)

    assert not tuple(backups.glob("*.sqlite3.gz"))
    assert not tuple(backups.glob("*.manifest.json"))
    assert {path.name for path in backups.glob(".stocker-v2-*")} == {".stocker-v2-backup.lock"}

    monkeypatch.setattr(backup_module.os, "replace", real_replace)
    valid = create_backup(database, backups, tier="daily", created_at_us=51)
    (backups / "callback-payload.json").write_text('{"secret":"must-not-render"}')
    (backups / "oversized.manifest.json").write_bytes(b"x" * 70_000)
    (backups / "linked.manifest.json").symlink_to(valid.manifest_path)

    projection = read_backup_manifests(backups, limit=200)
    assert [item.manifest_filename for item in projection.items] == [valid.manifest_path.name]
    assert "secret" not in json.dumps(projection.to_dict())


def test_backup_cli_emits_machine_readable_create_and_restore_results(tmp_path: Path) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    restored = tmp_path / "restored.sqlite3"
    _seed_database(database)
    runner = CliRunner()

    created = runner.invoke(
        app,
        [
            "backup",
            "create",
            "--database",
            str(database),
            "--destination",
            str(backups),
            "--tier",
            "daily",
            "--created-at-us",
            "60",
        ],
    )
    assert created.exit_code == 0, created.output
    created_payload = json.loads(created.stdout)
    assert created_payload["status"] == "ok"
    assert created_payload["tier"] == "daily"

    restored_result = runner.invoke(
        app,
        [
            "backup",
            "restore",
            "--manifest",
            str(backups / created_payload["manifest_filename"]),
            "--destination",
            str(restored),
        ],
    )
    assert restored_result.exit_code == 0, restored_result.output
    assert json.loads(restored_result.stdout)["status"] == "ok"
    assert restored.is_file()


def test_backup_cli_records_degraded_status_when_creation_fails(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    runner = CliRunner()

    failed = runner.invoke(
        app,
        [
            "backup",
            "create",
            "--database",
            str(tmp_path / "missing.sqlite3"),
            "--destination",
            str(backups),
            "--tier",
            "daily",
            "--created-at-us",
            "70",
        ],
    )

    assert failed.exit_code == 1
    assert json.loads(failed.stdout)["status"] == "error"
    status = json.loads((backups / "backup-status.json").read_text(encoding="utf-8"))
    assert status["state"] == "degraded"
    assert status["code"] == "BackupError"
    assert status["checked_at_us"] == 70
