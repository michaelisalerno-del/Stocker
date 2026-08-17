from __future__ import annotations

import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from stocker_runtime.ingestion.lifecycle import LocalWriterLock, LocalWriterLockError
from stocker_runtime.storage import connect_v2, initialize_database


def _script_module() -> ModuleType:
    path = Path("scripts/quiescent_v2_snapshot.py").resolve()
    specification = importlib.util.spec_from_file_location("quiescent_v2_snapshot", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _arguments(database: Path, destination: Path) -> list[str]:
    return [
        "quiescent_v2_snapshot.py",
        "--database",
        str(database),
        "--destination-directory",
        str(destination),
        "--runtime-executable",
        sys.executable,
        "--expected-schema",
        "19",
        "--run-id",
        "run-1",
        "--generation",
        "1",
        "--expected-max-source-sequence",
        "0",
        "--expected-nonterminal",
        "0",
        "--release-commit",
        "deadbee",
        "--operator",
        "test-operator",
    ]


def _inactive_services_and_no_descriptors(
    command: list[str],
    **_kwargs: object,
) -> subprocess.CompletedProcess[str]:
    if command[0] == "systemctl":
        return subprocess.CompletedProcess(command, 3, "", "")
    if command[0] == "lsof":
        return subprocess.CompletedProcess(command, 1, "", "")
    raise AssertionError(f"unexpected subprocess: {command}")


def test_quiescent_snapshot_rejects_competing_writer_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    sqlite3.connect(database).close()
    destination = tmp_path / "snapshots"
    destination.mkdir()
    module = _script_module()
    monkeypatch.setattr(module.subprocess, "run", _inactive_services_and_no_descriptors)
    monkeypatch.setattr(sys, "argv", _arguments(database, destination))

    with (
        LocalWriterLock.for_database(database),
        pytest.raises(LocalWriterLockError, match="writer lock is held"),
    ):
        module.main()

    assert tuple(destination.iterdir()) == ()


def test_quiescent_snapshot_lock_identity_loss_removes_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
    destination = tmp_path / "snapshots"
    destination.mkdir()
    module = _script_module()
    monkeypatch.setattr(module.subprocess, "run", _inactive_services_and_no_descriptors)
    monkeypatch.setattr(sys, "argv", _arguments(database, destination))
    evidence: dict[str, Any] = {"schema": 19, "quick_check": "ok"}

    def lose_lock(*_args: object, **_kwargs: object) -> dict[str, Any]:
        Path(f"{database}.writer.lock").unlink()
        return evidence

    monkeypatch.setattr(module, "_verify", lose_lock)

    with pytest.raises(LocalWriterLockError, match="lock identity changed"):
        module.main()

    assert tuple(destination.iterdir()) == ()


def test_quiescent_snapshot_keeps_only_two_compressed_restore_checked_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
    destination = tmp_path / "snapshots"
    destination.mkdir()
    for index in (1, 2):
        archive = destination / f"stocker-v2-old-{index}.sqlite3.gz"
        archive.write_bytes(b"old")
        Path(f"{str(archive)[:-3]}.manifest.json").write_text("{}\n", encoding="utf-8")
        os.utime(archive, ns=(index, index))
    module = _script_module()
    monkeypatch.setattr(module.subprocess, "run", _inactive_services_and_no_descriptors)
    monkeypatch.setattr(sys, "argv", _arguments(database, destination))
    monkeypatch.setattr(
        module,
        "_verify",
        lambda *_args, **_kwargs: {"schema": 19, "quick_check": "ok"},
    )

    module.main()

    archives = tuple(sorted(destination.glob("*.sqlite3.gz")))
    manifests = tuple(sorted(destination.glob("*.manifest.json")))
    uncompressed = tuple(destination.glob("*.sqlite3"))
    assert len(archives) == len(manifests) == 2
    assert all(archive.name != "stocker-v2-old-1.sqlite3.gz" for archive in archives)
    assert uncompressed == ()


def test_quiescent_snapshot_rejects_incomplete_wal_checkpoint_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    sqlite3.connect(database).close()
    destination = tmp_path / "snapshots"
    destination.mkdir()
    module = _script_module()
    monkeypatch.setattr(module.subprocess, "run", _inactive_services_and_no_descriptors)
    monkeypatch.setattr(sys, "argv", _arguments(database, destination))

    class IncompleteCheckpoint:
        def __enter__(self) -> IncompleteCheckpoint:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def execute(_sql: str) -> IncompleteCheckpoint:
            return IncompleteCheckpoint()

        @staticmethod
        def fetchone() -> tuple[int, int, int]:
            return (1, 7, 2)

    monkeypatch.setattr(module.sqlite3, "connect", lambda *_args, **_kwargs: IncompleteCheckpoint())

    with pytest.raises(RuntimeError, match="WAL checkpoint is incomplete"):
        module.main()

    assert tuple(destination.iterdir()) == ()


def test_quiescent_snapshot_rejects_mismatched_schema_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "snapshot.sqlite3"
    sqlite3.connect(database).close()
    module = _script_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            '{"applied_versions":[],"current_version":20,"status":"ok"}\n',
            "",
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected migration verification"):
        module._verify(
            database,
            runtime=Path(sys.executable),
            expected_schema=19,
            run_id="run-1",
            generation=1,
            expected_termination_code="CLEAN_STOP",
            expected_max_source_sequence=0,
            expected_nonterminal=0,
        )


def test_quiescent_snapshot_verifies_actual_generation_identity_with_matching_runtime(
    tmp_path: Path,
) -> None:
    database = tmp_path / "snapshot.sqlite3"
    initialize_database(database, applied_at_us=1)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, ended_at_us, config_hash, "
            "git_commit, data_class, status) VALUES ('run-1', 'prospective_record', 'ibkr', "
            "1, 2, ?, 'deadbee', 'prospective_protected', 'stopped')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us, "
            "ended_at_us, clean_stop, termination_code, input_hash) VALUES "
            "('run-1', 1, 'owner-1', 1, 2, 1, 'CLEAN_STOP', ?)",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(1, 'callback-1', 'run-1', 1, 1, 'quote', 1, ?, 'pending')",
            ("c" * 64,),
        )
    module = _script_module()

    evidence = module._verify(
        database,
        runtime=Path(".venv/bin/stocker-runtime").resolve(),
        expected_schema=20,
        run_id="run-1",
        generation=1,
        expected_termination_code="CLEAN_STOP",
        expected_max_source_sequence=1,
        expected_nonterminal=1,
    )

    assert evidence["config_hash"] == "a" * 64
    assert evidence["input_hash"] == "b" * 64
    assert evidence["generation_termination_code"] == "CLEAN_STOP"


def test_quiescent_snapshot_restore_hash_failure_removes_all_new_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "operational.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
    destination = tmp_path / "snapshots"
    destination.mkdir()
    module = _script_module()
    monkeypatch.setattr(module.subprocess, "run", _inactive_services_and_no_descriptors)
    monkeypatch.setattr(sys, "argv", _arguments(database, destination))
    monkeypatch.setattr(
        module,
        "_verify",
        lambda *_args, **_kwargs: {"schema": 19, "quick_check": "ok"},
    )
    real_sha256 = module._sha256

    def reject_restore(path: Path) -> str:
        if path.name.endswith(".restore.sqlite3"):
            return "0" * 64
        return cast(str, real_sha256(path))

    monkeypatch.setattr(module, "_sha256", reject_restore)

    with pytest.raises(RuntimeError, match="decompressed restore is not byte-identical"):
        module.main()

    assert tuple(destination.iterdir()) == ()
