from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from stocker_runtime.storage import connect_v2, initialize_database


def _script_module() -> ModuleType:
    path = Path("deploy/scripts/run-v2-quiescent-managed-backup.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "run_v2_quiescent_managed_backup",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000)


def _seed_owned_recorder(database: Path, *, heartbeat_at_us: int) -> None:
    initialize_database(database, applied_at_us=1)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, git_commit, "
            "data_class, status) VALUES ('run-1', 'shadow', 'ibkr', 1, ?, 'deadbee', "
            "'shadow_protected', 'running')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us, "
            "ownership_protocol, git_commit, input_hash) VALUES "
            "('run-1', 1, 'owner-1', 1, 'local_flock_v1', 'deadbee', ?)",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, "
            "process_heartbeat_at_us, connection_state, connection_generation) VALUES "
            "('run-1', 1, 'running', ?, 'connected', 1)",
            (heartbeat_at_us,),
        )


def test_restart_requires_fresh_owned_generation_heartbeat(tmp_path: Path) -> None:
    module = _script_module()
    database = tmp_path / "operational.sqlite3"
    _seed_owned_recorder(database, heartbeat_at_us=module.time.time_ns() // 1_000)

    module._require_fresh_owned_recorder(database)


def test_restart_rejects_stale_owned_generation_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    database = tmp_path / "operational.sqlite3"
    _seed_owned_recorder(database, heartbeat_at_us=1)
    monotonic = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "RECORDER_RESTART_TIMEOUT_SECONDS", 0.5)

    with pytest.raises(RuntimeError, match="fresh owned-generation heartbeat"):
        module._require_fresh_owned_recorder(database)


def test_quiescent_backup_window_uses_exact_xnys_regular_session() -> None:
    module = _script_module()

    with pytest.raises(RuntimeError, match="prohibited during XNYS regular session"):
        module.require_quiescent_backup_window(_timestamp("2026-08-17T14:00:00"))
    with pytest.raises(RuntimeError, match="insufficient time"):
        module.require_quiescent_backup_window(_timestamp("2026-08-17T13:00:00"))

    thanksgiving = _timestamp("2026-11-26T17:00:00")
    early_close = _timestamp("2026-11-27T18:00:00")
    winter_close = _timestamp("2026-12-01T21:00:00")
    assert module.require_quiescent_backup_window(thanksgiving) > thanksgiving
    assert module.require_quiescent_backup_window(early_close) > early_close
    assert module.require_quiescent_backup_window(winter_close) > winter_close


def _service_controller(module: ModuleType) -> tuple[list[tuple[str, str]], dict[str, bool]]:
    calls: list[tuple[str, str]] = []
    active = {module.RECORDER_UNIT: True, module.WEB_UNIT: True}

    def systemctl(action: str, unit: str) -> int:
        calls.append((action, unit))
        if action == "is-active":
            return 0 if active[unit] else 3
        if action == "stop":
            active[unit] = False
            return 0
        if action == "start":
            active[unit] = True
            return 0
        raise AssertionError(action)

    module._systemctl = systemctl
    module._require_fresh_owned_recorder = lambda _database: None
    module._prepare_web_sqlite_boundary = lambda: None
    return calls, active


class _FakeLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def acquire(self) -> None:
        self.events.append("lock:acquire")

    def verify_held(self) -> None:
        self.events.append("lock:verify")

    def release(self) -> None:
        self.events.append("lock:release")


def test_quiescent_backup_stops_once_holds_lock_and_restarts_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    calls, active = _service_controller(module)
    events: list[str] = []
    lock = _FakeLock(events)
    monkeypatch.setattr(module.LocalWriterLock, "for_database", lambda _database: lock)
    monkeypatch.setattr(module, "_require_no_database_descriptors", lambda _database: None)
    monkeypatch.setattr(module, "record_backup_failure", lambda *_args, **_kwargs: None)
    artifact = SimpleNamespace(
        archive_path=Path("daily.sqlite3.gz"),
        manifest_path=Path("daily.sqlite3.gz.manifest.json"),
        manifest=SimpleNamespace(compressed_bytes=123),
    )

    def create(*_args: object, **kwargs: Any) -> object:
        events.append("backup:create")
        kwargs["precondition"]()
        kwargs["post_publish_verify"](artifact)
        return artifact

    monkeypatch.setattr(module, "create_quiescent_backup", create)
    monkeypatch.setattr(
        module,
        "finalize_quiescent_backup",
        lambda *_args, **_kwargs: events.append("backup:finalize"),
    )
    monkeypatch.setattr(
        module,
        "_restore_check",
        lambda *_args: events.append("backup:restore-check"),
    )
    result = module.run_quiescent_managed_backup(
        tier="daily",
        now_us=_timestamp("2026-08-17T22:00:00"),
        database=tmp_path / "db.sqlite3",
        destination=tmp_path / "backups",
        working_directory=tmp_path,
        restart_marker=tmp_path / "restart-required",
        operation_lock=tmp_path / "operation.lock",
    )

    assert result["status"] == "ok"
    assert active == {module.RECORDER_UNIT: True, module.WEB_UNIT: True}
    assert calls == [
        ("is-active", module.RECORDER_UNIT),
        ("is-active", module.WEB_UNIT),
        ("stop", module.WEB_UNIT),
        ("stop", module.RECORDER_UNIT),
        ("is-active", module.WEB_UNIT),
        ("is-active", module.RECORDER_UNIT),
        ("start", module.RECORDER_UNIT),
        ("is-active", module.RECORDER_UNIT),
        ("start", module.WEB_UNIT),
    ]
    assert events == [
        "lock:acquire",
        "lock:verify",
        "backup:create",
        "lock:verify",
        "backup:restore-check",
        "lock:verify",
        "lock:release",
        "backup:finalize",
    ]


def test_quiescent_backup_failure_is_degraded_and_always_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    calls, active = _service_controller(module)
    lock = _FakeLock([])
    monkeypatch.setattr(module.LocalWriterLock, "for_database", lambda _database: lock)
    monkeypatch.setattr(module, "_require_no_database_descriptors", lambda _database: None)
    monkeypatch.setattr(
        module,
        "create_quiescent_backup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected backup failure")),
    )
    failures: list[str] = []
    monkeypatch.setattr(
        module,
        "record_backup_failure",
        lambda *_args, code, **_kwargs: failures.append(code),
    )

    with pytest.raises(RuntimeError, match="injected backup failure"):
        module.run_quiescent_managed_backup(
            tier="weekly",
            now_us=_timestamp("2026-08-23T06:00:00"),
            database=tmp_path / "db.sqlite3",
            destination=tmp_path / "backups",
            working_directory=tmp_path,
            restart_marker=tmp_path / "restart-required",
            operation_lock=tmp_path / "operation.lock",
        )

    assert failures == ["BACKUP_IN_PROGRESS", "RuntimeError"]
    assert active == {module.RECORDER_UNIT: True, module.WEB_UNIT: True}
    assert calls[-2:] == [
        ("is-active", module.RECORDER_UNIT),
        ("start", module.WEB_UNIT),
    ]


@pytest.mark.parametrize("failure_boundary", ["lock", "descriptor"])
def test_quiescent_backup_ownership_boundaries_fail_before_copy_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    module = _script_module()
    calls, active = _service_controller(module)
    lock = _FakeLock([])
    if failure_boundary == "lock":
        monkeypatch.setattr(
            lock,
            "acquire",
            lambda: (_ for _ in ()).throw(RuntimeError("writer lock is held")),
        )
    monkeypatch.setattr(module.LocalWriterLock, "for_database", lambda _database: lock)
    if failure_boundary == "descriptor":
        monkeypatch.setattr(
            module,
            "_require_no_database_descriptors",
            lambda _database: (_ for _ in ()).throw(RuntimeError("descriptor remains")),
        )
    else:
        monkeypatch.setattr(module, "_require_no_database_descriptors", lambda _database: None)
    copied = False

    def reject_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal copied
        copied = True

    monkeypatch.setattr(module, "create_quiescent_backup", reject_copy)
    monkeypatch.setattr(module, "record_backup_failure", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError):
        module.run_quiescent_managed_backup(
            tier="daily",
            now_us=_timestamp("2026-08-17T22:00:00"),
            database=tmp_path / "db.sqlite3",
            destination=tmp_path / "backups",
            working_directory=tmp_path,
            restart_marker=tmp_path / "restart-required",
            operation_lock=tmp_path / "operation.lock",
        )

    assert copied is False
    assert active == {module.RECORDER_UNIT: True, module.WEB_UNIT: True}
    assert calls[-3:] == [
        ("start", module.RECORDER_UNIT),
        ("is-active", module.RECORDER_UNIT),
        ("start", module.WEB_UNIT),
    ]


def test_regular_session_rejection_does_not_touch_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    touched = False

    def systemctl(*_args: object) -> int:
        nonlocal touched
        touched = True
        return 0

    monkeypatch.setattr(module, "_systemctl", systemctl)
    with pytest.raises(RuntimeError, match="prohibited during XNYS regular session"):
        module.run_quiescent_managed_backup(
            tier="daily",
            now_us=_timestamp("2026-08-17T14:00:00"),
            operation_lock=tmp_path / "operation.lock",
        )
    assert touched is False


def test_interruption_restart_is_marker_gated_and_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    calls, active = _service_controller(module)
    failures: list[tuple[str, int]] = []
    monkeypatch.setattr(
        module,
        "record_backup_failure",
        lambda *_args, code, checked_at_us: failures.append((code, checked_at_us)),
    )
    active[module.RECORDER_UNIT] = False
    active[module.WEB_UNIT] = False
    marker = tmp_path / "restart-required"

    module.restart_after_interruption(tier="daily", restart_marker=marker)
    assert calls == []
    assert active == {module.RECORDER_UNIT: False, module.WEB_UNIT: False}

    marker.write_text('{"created_at_us":1}\n', encoding="utf-8")
    module.restart_after_interruption(tier="daily", restart_marker=marker)
    assert calls == [
        ("start", module.RECORDER_UNIT),
        ("is-active", module.RECORDER_UNIT),
        ("start", module.WEB_UNIT),
    ]
    assert active == {module.RECORDER_UNIT: True, module.WEB_UNIT: True}
    assert not marker.exists()
    assert failures == [("BackupInterrupted", 1)]


def test_interruption_restart_skips_web_but_attempts_status_after_recorder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    calls: list[tuple[str, str]] = []

    def systemctl(action: str, unit: str) -> int:
        calls.append((action, unit))
        if unit == module.RECORDER_UNIT:
            raise RuntimeError("injected recorder restart failure")
        return 0

    failures: list[str] = []
    monkeypatch.setattr(module, "_systemctl", systemctl)
    monkeypatch.setattr(module, "_require_fresh_owned_recorder", lambda _database: None)
    monkeypatch.setattr(module, "_prepare_web_sqlite_boundary", lambda: None)
    monkeypatch.setattr(
        module,
        "record_backup_failure",
        lambda *_args, code, **_kwargs: failures.append(code),
    )
    marker = tmp_path / "restart-required"
    marker.write_text('{"created_at_us":1}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="interrupted backup recovery failed"):
        module.restart_after_interruption(tier="daily", restart_marker=marker)

    assert calls == [
        ("start", module.RECORDER_UNIT),
    ]
    assert failures == ["BackupInterrupted"]
    assert marker.is_file()


def test_restart_prepares_sqlite_boundary_after_recorder_and_before_web(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "_systemctl",
        lambda action, unit: events.append(f"{action}:{unit}") or 0,
    )
    monkeypatch.setattr(
        module,
        "_prepare_web_sqlite_boundary",
        lambda: events.append("sqlite-boundary"),
    )
    monkeypatch.setattr(
        module,
        "_require_fresh_owned_recorder",
        lambda _database: events.append("recorder-heartbeat"),
    )

    assert module._restart_services(Path("database.sqlite3")) == []
    assert events == [
        f"start:{module.RECORDER_UNIT}",
        f"is-active:{module.RECORDER_UNIT}",
        "sqlite-boundary",
        "recorder-heartbeat",
        f"start:{module.WEB_UNIT}",
    ]


def test_restart_attempts_web_and_reports_boundary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "_systemctl",
        lambda action, unit: events.append(f"{action}:{unit}") or 0,
    )
    monkeypatch.setattr(
        module,
        "_prepare_web_sqlite_boundary",
        lambda: (_ for _ in ()).throw(RuntimeError("injected boundary failure")),
    )
    monkeypatch.setattr(module, "_require_fresh_owned_recorder", lambda _database: None)

    assert module._restart_services(Path("database.sqlite3")) == ["sqlite-boundary:RuntimeError"]
    assert events == [
        f"start:{module.RECORDER_UNIT}",
        f"is-active:{module.RECORDER_UNIT}",
    ]


def test_restart_skips_web_when_heartbeat_fails_after_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "_systemctl",
        lambda action, unit: events.append(f"{action}:{unit}") or 0,
    )
    monkeypatch.setattr(
        module,
        "_prepare_web_sqlite_boundary",
        lambda: events.append("sqlite-boundary"),
    )
    monkeypatch.setattr(
        module,
        "_require_fresh_owned_recorder",
        lambda _database: (_ for _ in ()).throw(RuntimeError("injected stale heartbeat")),
    )

    assert module._restart_services(Path("database.sqlite3")) == ["recorder-heartbeat:RuntimeError"]
    assert events == [
        f"start:{module.RECORDER_UNIT}",
        f"is-active:{module.RECORDER_UNIT}",
        "sqlite-boundary",
    ]


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            "blocked_unsafe_runtime_configuration:v2_sqlite_boundary:shm_race_exhausted\n",
            "shm_race_exhausted",
        ),
        (
            "blocked_unsafe_runtime_configuration:v2_sqlite_boundary:"
            "persistent_root_mode_update_failed\n",
            "persistent_root_mode_update_failed",
        ),
        (
            "blocked_unsafe_runtime_configuration:v2_sqlite_boundary:wrong_owner\n"
            "attacker-controlled detail\n",
            "unknown_failure",
        ),
    ],
)
def test_boundary_failure_exposes_only_fixed_sanitized_reason(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    expected: str,
) -> None:
    module = _script_module()
    completed = SimpleNamespace(returncode=78, stdout="", stderr=stderr)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(module.SQLiteBoundaryError) as raised:
        module._prepare_web_sqlite_boundary()

    assert raised.value.reason == expected
    assert "attacker-controlled" not in str(raised.value)


def test_restart_skips_boundary_and_web_when_recorder_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    events: list[str] = []

    def systemctl(action: str, unit: str) -> int:
        events.append(f"{action}:{unit}")
        raise RuntimeError("injected recorder failure")

    monkeypatch.setattr(module, "_systemctl", systemctl)
    monkeypatch.setattr(
        module,
        "_require_fresh_owned_recorder",
        lambda _database: events.append("recorder-heartbeat"),
    )
    monkeypatch.setattr(
        module,
        "_prepare_web_sqlite_boundary",
        lambda: events.append("sqlite-boundary"),
    )

    assert module._restart_services(Path("database.sqlite3")) == [
        f"{module.RECORDER_UNIT}:RuntimeError"
    ]
    assert events == [f"start:{module.RECORDER_UNIT}"]


def test_concurrent_daily_and_weekly_backup_cannot_touch_winner_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    calls, active = _service_controller(module)
    daily_marker = tmp_path / "daily-restart-required"
    weekly_marker = tmp_path / "weekly-restart-required"
    daily_marker.write_text('{"created_at_us":1}\n', encoding="utf-8")
    operation_lock = tmp_path / "operation.lock"
    status_changed = False

    def record_failure(*_args: object, **_kwargs: object) -> None:
        nonlocal status_changed
        status_changed = True

    monkeypatch.setattr(module, "record_backup_failure", record_failure)
    with (
        module._operation_lock(operation_lock),
        pytest.raises(RuntimeError, match="another quiescent managed backup is active"),
    ):
        module.run_quiescent_managed_backup(
            tier="weekly",
            now_us=_timestamp("2026-08-23T06:00:00"),
            database=tmp_path / "db.sqlite3",
            destination=tmp_path / "backups",
            working_directory=tmp_path,
            restart_marker=weekly_marker,
            operation_lock=operation_lock,
        )

    assert calls == []
    assert status_changed is False
    assert daily_marker.is_file()
    assert not weekly_marker.exists()

    active[module.RECORDER_UNIT] = False
    active[module.WEB_UNIT] = False
    module.restart_after_interruption(tier="daily", restart_marker=daily_marker)
    assert calls[-3:] == [
        ("start", module.RECORDER_UNIT),
        ("is-active", module.RECORDER_UNIT),
        ("start", module.WEB_UNIT),
    ]
    assert active == {module.RECORDER_UNIT: True, module.WEB_UNIT: True}
    assert not daily_marker.exists()
