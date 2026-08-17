from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
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
    create_quiescent_backup,
    finalize_quiescent_backup,
    initialize_database,
    read_backup_manifests,
    record_backup_failure,
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


def _regular_directory_bytes(directory: Path) -> int:
    return sum(
        path.stat(follow_symlinks=False).st_size
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    )


def test_quiescent_managed_backup_is_byte_identical_and_restore_checked(tmp_path: Path) -> None:
    database = tmp_path / "operational.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    backups.mkdir()
    working.mkdir()
    _seed_database(database, incidents=3)
    with connect_v2(database) as connection:
        before_incidents = tuple(connection.execute("SELECT * FROM incidents ORDER BY incident_id"))
    precondition_calls = 0

    def verify_lock() -> None:
        nonlocal precondition_calls
        precondition_calls += 1

    artifact = create_quiescent_backup(
        database,
        backups,
        tier="daily",
        precondition=verify_lock,
        created_at_us=1,
        working_directory=working,
    )
    restored = tmp_path / "restored.sqlite3"
    result = restore_backup(artifact.manifest_path, restored)

    assert precondition_calls >= 4
    source_hash = _sha256(database)
    assert result.uncompressed_sha256 == source_hash
    assert _sha256(restored) == source_hash
    with connect_v2(restored) as connection:
        assert tuple(connection.execute("SELECT * FROM incidents ORDER BY incident_id")) == (
            before_incidents
        )
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert tuple(connection.execute("PRAGMA foreign_key_check")) == ()


def test_quiescent_managed_backup_rejects_source_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    backups.mkdir()
    working.mkdir()
    _seed_database(database)
    real_copy = backup_module.shutil.copyfile

    def mutate_after_copy(source: Path, destination: Path) -> None:
        real_copy(source, destination)
        with source.open("ab") as active_database:
            active_database.write(b"injected mutation")

    monkeypatch.setattr(backup_module.shutil, "copyfile", mutate_after_copy)
    with pytest.raises(BackupError, match="changed during the quiescent copy"):
        create_quiescent_backup(
            database,
            backups,
            tier="daily",
            precondition=lambda: None,
            created_at_us=1,
            working_directory=working,
        )

    assert tuple(backups.glob("*.sqlite3.gz")) == ()


def test_quiescent_managed_backup_stays_degraded_until_restore_proof_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    backups.mkdir()
    working.mkdir()
    _seed_database(database)
    observed_status: dict[str, object] = {}
    opened: list[sqlite3.Connection] = []
    real_connect = backup_module.sqlite3.connect

    def observed_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(backup_module.sqlite3, "connect", observed_connect)

    def fail_restore_proof(_artifact: object) -> None:
        _assert_only_work_lock(working)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[-1].execute("SELECT 1")
        observed_status.update(
            json.loads((backups / backup_module.BACKUP_STATUS_FILENAME).read_text())
        )
        raise RuntimeError("injected restore proof failure")

    with pytest.raises(RuntimeError, match="injected restore proof failure"):
        create_quiescent_backup(
            database,
            backups,
            tier="daily",
            precondition=lambda: None,
            created_at_us=1,
            working_directory=working,
            post_publish_verify=fail_restore_proof,
        )

    assert observed_status["state"] == "degraded"
    assert observed_status["code"] == "BACKUP_IN_PROGRESS"
    final_status = json.loads((backups / backup_module.BACKUP_STATUS_FILENAME).read_text())
    assert final_status["state"] == "degraded"
    assert final_status["code"] == "RuntimeError"


def test_deferred_quiescent_backup_becomes_healthy_only_after_finalization(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    backups.mkdir()
    working.mkdir()
    _seed_database(database)
    artifact = create_quiescent_backup(
        database,
        backups,
        tier="daily",
        precondition=lambda: None,
        created_at_us=1,
        working_directory=working,
        post_publish_verify=lambda _artifact: None,
        defer_healthy_status=True,
    )

    pending = json.loads((backups / backup_module.BACKUP_STATUS_FILENAME).read_text())
    assert pending["state"] == "degraded"
    assert pending["code"] == "BACKUP_IN_PROGRESS"
    finalize_quiescent_backup(
        artifact,
        backups,
        working_directory=working,
    )
    healthy = json.loads((backups / backup_module.BACKUP_STATUS_FILENAME).read_text())
    assert healthy["state"] == "healthy"
    assert healthy["code"] is None


def _assert_only_work_lock(working: Path) -> None:
    entries = tuple(working.iterdir())
    assert [path.name for path in entries] == [backup_module.BACKUP_WORK_LOCK_FILENAME]
    lock = entries[0]
    assert lock.is_file()
    assert not lock.is_symlink()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o640


def _seed_rotation_set(database: Path, backups: Path, working: Path) -> BackupPolicy:
    policy = BackupPolicy(
        daily_retention=3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=64 * 1024 * 1024,
    )
    for tier, timestamps in (("daily", (1, 2, 3)), ("weekly", (10, 11))):
        for created_at_us in timestamps:
            create_backup(
                database,
                backups,
                tier=tier,  # type: ignore[arg-type]
                created_at_us=created_at_us,
                policy=policy,
                working_directory=working,
            )
    return policy


def _seed_floor_rotation_set(
    database: Path,
    backups: Path,
    working: Path,
    *,
    post_publication_victim: bool,
) -> BackupPolicy:
    daily_timestamps = (1, 2) if post_publication_victim else (1, 2, 3)
    policy = BackupPolicy(
        daily_retention=2 if post_publication_victim else 3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=64 * 1024 * 1024,
    )
    for tier, timestamps in (("daily", daily_timestamps), ("weekly", (10, 11))):
        for created_at_us in timestamps:
            create_backup(
                database,
                backups,
                tier=tier,  # type: ignore[arg-type]
                created_at_us=created_at_us,
                policy=policy,
                working_directory=working,
            )
    return policy


def _managed_one_sided_names(directory: Path) -> set[str]:
    names = {path.name for path in directory.iterdir()}
    one_sided: set[str] = set()
    manifest_suffix = ".manifest.json"
    for name in names:
        if backup_module._is_managed_archive_filename(name):
            if f"{name}{manifest_suffix}" not in names:
                one_sided.add(name)
            continue
        if name.endswith(manifest_suffix):
            archive_name = name[: -len(manifest_suffix)]
            if (
                backup_module._is_managed_archive_filename(archive_name)
                and archive_name not in names
            ):
                one_sided.add(name)
    return one_sided


def _inject_victim_archive_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
    victim_archive: Path,
    *,
    persistent: bool,
) -> tuple[list[int], dict[str, bool]]:
    real_unlink = Path.unlink
    attempts: list[int] = []
    state = {"enabled": True}

    def failing_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == victim_archive and state["enabled"]:
            attempts.append(len(attempts) + 1)
            if persistent or len(attempts) == 1:
                raise OSError(errno.EIO, os.strerror(errno.EIO), str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    return attempts, state


def _tight_rotation_cap(
    database: Path,
    backups: Path,
    probe: Path,
    probe_working: Path,
) -> tuple[int, str, str]:
    candidate = create_backup(
        database,
        probe,
        tier="daily",
        created_at_us=4,
        working_directory=probe_working,
    )
    projection = read_backup_manifests(backups, limit=200)
    oldest_daily = min(
        (item for item in projection.items if item.tier == "daily"),
        key=lambda item: item.created_at_us,
    )
    victim_bytes = (
        backups.joinpath(oldest_daily.archive_filename).stat().st_size
        + backups.joinpath(oldest_daily.manifest_filename).stat().st_size
    )
    status_path = backups / backup_module.BACKUP_STATUS_FILENAME
    current_status_bytes = status_path.stat().st_size
    persisted_status = json.loads(status_path.read_text(encoding="utf-8"))
    previous_latest = persisted_status["latest_manifest_filename"]
    in_progress_bytes = len(
        backup_module._status_payload(
            state="degraded",
            checked_at_us=4,
            code="BACKUP_IN_PROGRESS",
            latest_manifest_filename=previous_latest,
        )
    )
    healthy_status = backup_module._status_payload(
        state="healthy",
        checked_at_us=4,
        code=None,
        latest_manifest_filename=candidate.manifest.manifest_filename,
    )
    status_reserve = backup_module._status_publication_reserve(
        candidate=candidate.manifest,
        previous_status=backup_module._read_status(backups),
        current_status_bytes=in_progress_bytes,
        healthy_status_bytes=len(healthy_status),
    )
    cap = (
        _regular_directory_bytes(backups)
        - current_status_bytes
        - victim_bytes
        + in_progress_bytes
        + candidate.archive_path.stat().st_size
        + candidate.manifest_path.stat().st_size
        + status_reserve
    )
    return cap, candidate.manifest.archive_filename, oldest_daily.archive_filename


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
    _assert_only_work_lock(working)


def test_backup_publication_uses_a_destination_local_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    real_replace = backup_module.os.replace
    replacements: list[tuple[Path, Path]] = []

    def reject_cross_device_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        replacements.append((source_path, destination_path))
        if source_path.parent == working.resolve() and destination_path.parent == backups.resolve():
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), str(source_path))
        real_replace(source, destination)

    monkeypatch.setattr(backup_module.os, "replace", reject_cross_device_replace)
    artifact = create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=200,
        working_directory=working,
    )

    archive_publications = [
        (source, destination)
        for source, destination in replacements
        if destination == artifact.archive_path.resolve()
    ]
    assert len(archive_publications) == 1
    assert archive_publications[0][0].parent == backups.resolve()
    assert artifact.archive_path.is_file()
    assert artifact.manifest_path.is_file()
    assert not tuple(backups.glob(".stocker-v2-atomic-*"))
    _assert_only_work_lock(working)


def test_cross_destination_backups_lock_shared_work_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    missing_database = tmp_path / "missing.sqlite3"
    first_backups = tmp_path / "first-backups"
    second_backups = tmp_path / "second-backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    real_online_copy = backup_module._online_copy
    first_copy_ready = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []
    first_completed = threading.Event()

    def paused_online_copy(source: Path, destination: Path) -> None:
        real_online_copy(source, destination)
        first_copy_ready.set()
        assert release_first.wait(timeout=5)

    monkeypatch.setattr(backup_module, "_online_copy", paused_online_copy)

    def run_first() -> None:
        try:
            create_backup(
                database,
                first_backups,
                tier="daily",
                created_at_us=200,
                working_directory=working,
            )
            first_completed.set()
        except BaseException as error:  # pragma: no cover - asserted below
            first_errors.append(error)

    def run_invalid_second() -> None:
        try:
            create_backup(
                missing_database,
                second_backups,
                tier="daily",
                created_at_us=201,
                working_directory=working,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            second_errors.append(error)
        finally:
            second_finished.set()

    first = threading.Thread(target=run_first)
    second = threading.Thread(target=run_invalid_second)
    first.start()
    assert first_copy_ready.wait(timeout=5)
    second.start()
    try:
        assert not second_finished.wait(timeout=0.25)
    finally:
        release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert first_errors == []
    assert first_completed.is_set()
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], BackupError)


def test_rotation_never_exceeds_the_physical_destination_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    probe = tmp_path / "probe"
    probe_working = tmp_path / "probe-working"
    working.mkdir()
    probe_working.mkdir()
    _seed_database(database)
    _seed_rotation_set(database, backups, working)
    cap, candidate_name, victim_name = _tight_rotation_cap(
        database,
        backups,
        probe,
        probe_working,
    )
    policy = BackupPolicy(
        daily_retention=3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=cap,
    )
    real_replace = backup_module.os.replace
    observed_bytes: list[int] = []

    def observe_destination_peak(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if source_path.parent == backups.resolve() or destination_path.parent == backups.resolve():
            observed_bytes.append(_regular_directory_bytes(backups))
        real_replace(source, destination)
        if destination_path.parent == backups.resolve():
            observed_bytes.append(_regular_directory_bytes(backups))

    monkeypatch.setattr(backup_module.os, "replace", observe_destination_peak)
    artifact = create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=4,
        policy=policy,
        working_directory=working,
    )

    assert artifact.archive_path.name == candidate_name
    assert observed_bytes
    assert max(observed_bytes) <= cap
    assert _regular_directory_bytes(backups) <= cap
    assert not (backups / victim_name).exists()
    projection = read_backup_manifests(backups, limit=200)
    assert [item.tier for item in projection.items].count("daily") == 3
    assert [item.tier for item in projection.items].count("weekly") == 2


def test_failed_publication_after_planned_rotation_preserves_tier_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    probe = tmp_path / "probe"
    probe_working = tmp_path / "probe-working"
    working.mkdir()
    probe_working.mkdir()
    _seed_database(database)
    _seed_rotation_set(database, backups, working)
    cap, candidate_name, victim_name = _tight_rotation_cap(
        database,
        backups,
        probe,
        probe_working,
    )
    policy = BackupPolicy(
        daily_retention=3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=cap,
    )
    real_replace = backup_module.os.replace

    def fail_archive_publish(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        if Path(destination).resolve() == (backups / candidate_name).resolve():
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(destination))
        real_replace(source, destination)

    monkeypatch.setattr(backup_module.os, "replace", fail_archive_publish)
    with pytest.raises(OSError, match="No space left"):
        create_backup(
            database,
            backups,
            tier="daily",
            created_at_us=4,
            policy=policy,
            working_directory=working,
        )

    projection = read_backup_manifests(backups, limit=200)
    assert [item.tier for item in projection.items].count("daily") == 2
    assert [item.tier for item in projection.items].count("weekly") == 2
    assert not (backups / victim_name).exists()
    assert not (backups / candidate_name).exists()
    assert not (backups / f"{candidate_name}.manifest.json").exists()
    assert not tuple(backups.glob(".stocker-v2-atomic-*"))
    assert not tuple(
        path for path in working.iterdir() if backup_module._is_backup_work_file(path.name)
    )
    assert projection.status.state == "degraded"


@pytest.mark.parametrize(
    "post_publication_victim",
    (False, True),
    ids=("pre-publication-victim", "post-publication-victim"),
)
def test_partial_rotation_delete_is_recovered_before_terminal_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_publication_victim: bool,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    policy = _seed_floor_rotation_set(
        database,
        backups,
        working,
        post_publication_victim=post_publication_victim,
    )
    previous_status = json.loads(
        (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
    )
    victim_name = backup_module._archive_filename("daily", 1)
    victim_archive = backups / victim_name
    victim_manifest = backups / f"{victim_name}.manifest.json"
    candidate_name = backup_module._archive_filename("daily", 4)
    candidate_archive = backups / candidate_name
    candidate_manifest = backups / f"{candidate_name}.manifest.json"
    attempts, _state = _inject_victim_archive_unlink_failure(
        monkeypatch,
        victim_archive,
        persistent=False,
    )

    with pytest.raises(OSError, match="Input/output"):
        create_backup(
            database,
            backups,
            tier="daily",
            created_at_us=4,
            policy=policy,
            working_directory=working,
        )

    assert attempts == [1, 2]
    assert not victim_archive.exists()
    assert not victim_manifest.exists()
    assert _managed_one_sided_names(backups) == set()
    assert candidate_archive.exists() is post_publication_victim
    assert candidate_manifest.exists() is post_publication_victim
    failed_status = json.loads(
        (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
    )
    assert failed_status["code"] == "OSError"
    assert failed_status["latest_manifest_filename"] == (
        candidate_manifest.name
        if post_publication_victim
        else previous_status["latest_manifest_filename"]
    )
    failed_projection = read_backup_manifests(backups, limit=200, now_us=11)
    assert [item.tier for item in failed_projection.items].count("daily") == 2
    assert [item.tier for item in failed_projection.items].count("weekly") == 2
    assert _regular_directory_bytes(backups) <= policy.byte_cap

    artifact = create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=5,
        policy=policy,
        working_directory=working,
    )

    assert _managed_one_sided_names(backups) == set()
    recovered = read_backup_manifests(backups, limit=200, now_us=11)
    assert [item.tier for item in recovered.items].count("daily") == policy.daily_retention
    assert [item.tier for item in recovered.items].count("weekly") == 2
    assert _regular_directory_bytes(backups) <= policy.byte_cap
    assert artifact.manifest.manifest_filename == recovered.status.latest_manifest_filename
    assert recovered.status.state == "healthy"


@pytest.mark.parametrize(
    "post_publication_victim",
    (False, True),
    ids=("pre-publication-victim", "post-publication-victim"),
)
def test_persistent_rotation_cleanup_failure_keeps_recovery_gate_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_publication_victim: bool,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    policy = _seed_floor_rotation_set(
        database,
        backups,
        working,
        post_publication_victim=post_publication_victim,
    )
    previous_status = json.loads(
        (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
    )
    victim_name = backup_module._archive_filename("daily", 1)
    victim_archive = backups / victim_name
    candidate_manifest_name = f"{backup_module._archive_filename('daily', 4)}.manifest.json"
    attempts, failure_state = _inject_victim_archive_unlink_failure(
        monkeypatch,
        victim_archive,
        persistent=True,
    )

    with pytest.raises(OSError, match="Input/output"):
        create_backup(
            database,
            backups,
            tier="daily",
            created_at_us=4,
            policy=policy,
            working_directory=working,
        )

    assert attempts == [1, 2]
    assert victim_archive.exists()
    assert _managed_one_sided_names(backups) == {victim_name}
    interrupted_status = json.loads(
        (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
    )
    assert interrupted_status["code"] == "BACKUP_IN_PROGRESS"
    assert interrupted_status["latest_manifest_filename"] == (
        candidate_manifest_name
        if post_publication_victim
        else previous_status["latest_manifest_filename"]
    )

    with pytest.raises(OSError, match="Input/output"):
        record_backup_failure(backups, code="BackupError", checked_at_us=5)

    assert attempts == [1, 2, 3]
    assert (
        json.loads((backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8"))
        == interrupted_status
    )
    assert _regular_directory_bytes(backups) <= policy.byte_cap

    failure_state["enabled"] = False
    artifact = create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=5,
        policy=policy,
        working_directory=working,
    )

    assert not victim_archive.exists()
    assert _managed_one_sided_names(backups) == set()
    recovered = read_backup_manifests(backups, limit=200, now_us=11)
    assert [item.tier for item in recovered.items].count("daily") == policy.daily_retention
    assert [item.tier for item in recovered.items].count("weekly") == 2
    assert _regular_directory_bytes(backups) <= policy.byte_cap
    assert artifact.manifest.manifest_filename == recovered.status.latest_manifest_filename
    assert recovered.status.state == "healthy"


def test_interruption_after_manifest_leaves_a_bounded_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedSigkill(BaseException):
        pass

    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    probe = tmp_path / "probe"
    probe_working = tmp_path / "probe-working"
    working.mkdir()
    probe_working.mkdir()
    _seed_database(database)
    _seed_rotation_set(database, backups, working)
    cap, candidate_name, victim_name = _tight_rotation_cap(
        database,
        backups,
        probe,
        probe_working,
    )
    policy = BackupPolicy(
        daily_retention=3,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=cap,
    )
    real_write_atomic = backup_module._write_atomic

    def interrupt_after_manifest(
        path: Path,
        payload: bytes,
        *,
        mode: int = 0o640,
    ) -> None:
        real_write_atomic(path, payload, mode=mode)
        if path.name == f"{candidate_name}.manifest.json":
            raise SimulatedSigkill

    monkeypatch.setattr(backup_module, "_write_atomic", interrupt_after_manifest)
    with pytest.raises(SimulatedSigkill):
        create_backup(
            database,
            backups,
            tier="daily",
            created_at_us=4,
            policy=policy,
            working_directory=working,
        )

    assert _regular_directory_bytes(backups) <= cap
    assert (backups / candidate_name).is_file()
    assert (backups / f"{candidate_name}.manifest.json").is_file()
    assert not (backups / victim_name).exists()
    assert not tuple(backups.glob(".stocker-v2-atomic-*"))
    assert read_backup_manifests(backups, limit=200).status.code == "BACKUP_IN_PROGRESS"


def test_backup_retry_cleans_all_temporary_namespaces_before_source_validation(
    tmp_path: Path,
) -> None:
    missing_database = tmp_path / "missing.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    backups.mkdir()
    working.mkdir()
    destination_orphan = backups / ".stocker-v2-atomic-abandoned"
    work_orphans = (
        working / ".stocker-v2-database-abandoned.sqlite3",
        working / ".stocker-v2-archive-abandoned.gz",
    )
    destination_orphan.write_bytes(b"abandoned-by-sigkill")
    for orphan in work_orphans:
        orphan.write_bytes(b"abandoned-by-sigkill")

    with pytest.raises(BackupError, match="backup source"):
        create_backup(
            missing_database,
            backups,
            tier="daily",
            created_at_us=201,
            working_directory=working,
        )

    assert not destination_orphan.exists()
    assert all(not orphan.exists() for orphan in work_orphans)


def test_backup_recovers_terminated_work_and_marks_the_attempt_in_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    working = tmp_path / "working"
    working.mkdir()
    _seed_database(database)
    stale_names = (
        ".stocker-v2-database-abandoned.sqlite3",
        ".stocker-v2-database-abandoned.sqlite3-wal",
        ".stocker-v2-database-abandoned.sqlite3-shm",
        ".stocker-v2-database-abandoned.sqlite3-journal",
        ".stocker-v2-archive-abandoned.gz",
    )
    for name in stale_names:
        (working / name).write_bytes(b"abandoned-by-sigkill")

    real_online_copy = backup_module._online_copy

    def observed_online_copy(source: Path, destination: Path) -> None:
        assert all(not (working / name).exists() for name in stale_names)
        status = json.loads(
            (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
        )
        assert status["state"] == "degraded"
        assert status["code"] == "BACKUP_IN_PROGRESS"
        real_online_copy(source, destination)

    monkeypatch.setattr(backup_module, "_online_copy", observed_online_copy)
    create_backup(
        database,
        backups,
        tier="daily",
        created_at_us=201,
        working_directory=working,
    )

    _assert_only_work_lock(working)
    status = json.loads(
        (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
    )
    assert status["state"] == "healthy"
    assert status["code"] is None


def test_backup_status_stays_degraded_until_rotation_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    policy = BackupPolicy(
        daily_retention=2,
        weekly_retention=2,
        minimum_per_tier=2,
        byte_cap=64 * 1024 * 1024,
    )
    create_backup(database, backups, tier="daily", created_at_us=1, policy=policy)
    create_backup(database, backups, tier="daily", created_at_us=2, policy=policy)
    real_unlink = Path.unlink
    states_during_rotation: list[dict[str, object]] = []

    def observed_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == backups.resolve() and path.name.startswith("stocker-v2-daily-"):
            states_during_rotation.append(
                json.loads(
                    (backups / backup_module.BACKUP_STATUS_FILENAME).read_text(encoding="utf-8")
                )
            )
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observed_unlink)
    create_backup(database, backups, tier="daily", created_at_us=3, policy=policy)

    assert states_during_rotation
    assert all(item["state"] == "degraded" for item in states_during_rotation)
    assert all(item["code"] == "BACKUP_IN_PROGRESS" for item in states_during_rotation)


def test_backup_recovers_orphaned_publication_and_rotation_files(tmp_path: Path) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backups = tmp_path / "backups"
    _seed_database(database)
    record_backup_failure(backups, code="BACKUP_IN_PROGRESS", checked_at_us=1)
    orphan_archive_name = backup_module._archive_filename("daily", 1)
    orphan_manifest_archive_name = backup_module._archive_filename("weekly", 2)
    orphan_archive = backups / orphan_archive_name
    orphan_manifest = backups / f"{orphan_manifest_archive_name}.manifest.json"
    orphan_archive.write_bytes(b"published-before-sigkill")
    orphan_manifest.write_text("{}\n", encoding="utf-8")

    artifact = create_backup(database, backups, tier="daily", created_at_us=3)

    assert not orphan_archive.exists()
    assert not orphan_manifest.exists()
    assert artifact.archive_path.exists()
    assert artifact.manifest_path.exists()


def test_backup_recovers_orphan_before_a_failed_retry_overwrites_status(tmp_path: Path) -> None:
    missing_database = tmp_path / "missing.sqlite3"
    backups = tmp_path / "backups"
    record_backup_failure(backups, code="BACKUP_IN_PROGRESS", checked_at_us=1)
    orphan_name = backup_module._archive_filename("daily", 1)
    orphan = backups / orphan_name
    orphan.write_bytes(b"published-before-sigkill")

    with pytest.raises(BackupError, match="backup source"):
        create_backup(missing_database, backups, tier="daily", created_at_us=2)
    record_backup_failure(backups, code="BackupError", checked_at_us=2)

    assert not orphan.exists()
    assert read_backup_manifests(backups, now_us=2).status.code == "BackupError"


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
