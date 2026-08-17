"""Create one bounded managed backup while the V2 database is quiescent."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from stocker_runtime.ingestion.lifecycle import LocalWriterLock
from stocker_runtime.market_session import market_data_expected_since_us, xnys_session_window_us
from stocker_runtime.storage import (
    BackupArtifact,
    BackupPolicy,
    create_quiescent_backup,
    finalize_quiescent_backup,
    record_backup_failure,
    restore_backup,
    verify_database,
)

DATABASE = Path("/var/lib/stocker/v2/stocker-v2.sqlite3")
DESTINATION = Path("/var/lib/stocker/backups-v2")
WORKING_DIRECTORY = Path("/var/cache/stocker-v2-backup-work")
OPERATION_LOCK = Path("/run/lock/stocker-v2-quiescent-managed-backup.lock")
RESTART_MARKERS = {
    "daily": Path("/run/stocker-v2-backup-daily/restart-required"),
    "weekly": Path("/run/stocker-v2-backup-weekly/restart-required"),
}
RECORDER_UNIT = "stocker-v2-recorder.service"
WEB_UNIT = "stocker-v2-web.service"
SQLITE_BOUNDARY = Path("/usr/local/libexec/stocker-prepare-v2-sqlite-boundary")
MINIMUM_OFF_SESSION_US = 60 * 60 * 1_000_000
MAX_SESSION_LOOKAHEAD_DAYS = 10
RECORDER_RESTART_TIMEOUT_SECONDS = 30.0
RECORDER_HEARTBEAT_FRESH_US = 5_000_000
_NEW_YORK = ZoneInfo("America/New_York")


def _next_xnys_open_us(now_us: int) -> int:
    local_date = datetime.fromtimestamp(now_us / 1_000_000, UTC).astimezone(_NEW_YORK).date()
    for offset in range(MAX_SESSION_LOOKAHEAD_DAYS + 1):
        window = xnys_session_window_us(local_date + timedelta(days=offset))
        if window is not None and window[0] > now_us:
            return window[0]
    raise RuntimeError("next XNYS regular session is outside the bounded lookahead")


def require_quiescent_backup_window(now_us: int) -> int:
    """Return the next XNYS open after proving a bounded off-session window."""

    if market_data_expected_since_us(now_us) is not None:
        raise RuntimeError("quiescent backup is prohibited during XNYS regular session")
    next_open_us = _next_xnys_open_us(now_us)
    if next_open_us - now_us < MINIMUM_OFF_SESSION_US:
        raise RuntimeError("quiescent backup has insufficient time before the next XNYS open")
    return next_open_us


def _systemctl(action: Literal["start", "stop", "is-active"], unit: str) -> int:
    completed = subprocess.run(
        ["/usr/bin/systemctl", action, unit],
        check=False,
        capture_output=True,
        text=True,
    )
    if action in {"start", "stop"} and completed.returncode != 0:
        raise RuntimeError(f"systemctl {action} failed for {unit}")
    return completed.returncode


def _require_no_database_descriptors(database: Path) -> None:
    completed = subprocess.run(
        ["/usr/bin/lsof", str(database)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        raise RuntimeError("operational database still has an open descriptor")
    if completed.returncode != 1:
        raise RuntimeError("cannot prove the operational database has no open descriptor")


def _prepare_web_sqlite_boundary() -> None:
    """Recreate the verified WAL/SHM boundary before systemd mounts the web sandbox."""

    completed = subprocess.run(
        [str(SQLITE_BOUNDARY)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("SQLite boundary preparation failed before web restart")


def _require_fresh_owned_recorder(database: Path) -> None:
    deadline = time.monotonic() + RECORDER_RESTART_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        now_us = time.time_ns() // 1_000
        try:
            with closing(
                sqlite3.connect(
                    f"{database.resolve().as_uri()}?mode=ro",
                    uri=True,
                    timeout=0.3,
                    isolation_level=None,
                )
            ) as connection:
                connection.execute("PRAGMA query_only = ON")
                row = connection.execute(
                    "SELECT state.process_heartbeat_at_us "
                    "FROM runtime_state state "
                    "JOIN runs run ON run.run_id=state.run_id "
                    "JOIN recorder_generations generation "
                    "ON generation.run_id=state.run_id "
                    "AND generation.generation=state.recorder_generation "
                    "WHERE run.status='running' AND generation.ended_at_us IS NULL "
                    "AND generation.ownership_protocol='local_flock_v1' "
                    "AND state.lifecycle NOT IN ('stopped','fatal') "
                    "ORDER BY state.process_heartbeat_at_us DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None and row[0] is not None:
            age_us = now_us - int(row[0])
            if 0 <= age_us <= RECORDER_HEARTBEAT_FRESH_US:
                return
        time.sleep(0.25)
    raise RuntimeError("recorder did not publish a fresh owned-generation heartbeat")


def _restart_services(database: Path) -> list[str]:
    errors: list[str] = []
    try:
        _systemctl("start", RECORDER_UNIT)
    except Exception as error:
        errors.append(f"{RECORDER_UNIT}:{type(error).__name__}")
        return errors
    if _systemctl("is-active", RECORDER_UNIT) != 0:
        errors.append(f"{RECORDER_UNIT}:inactive")
        return errors
    try:
        _require_fresh_owned_recorder(database)
    except Exception as error:
        errors.append(f"recorder-heartbeat:{type(error).__name__}")
        return errors
    try:
        _prepare_web_sqlite_boundary()
    except Exception as error:
        errors.append(f"sqlite-boundary:{type(error).__name__}")
        return errors
    try:
        _systemctl("start", WEB_UNIT)
    except Exception as error:
        errors.append(f"{WEB_UNIT}:{type(error).__name__}")
    return errors


def _restore_check(manifest: Path, working_directory: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stocker-v2-managed-restore-",
        suffix=".sqlite3",
        dir=working_directory,
    )
    os.close(descriptor)
    restored = Path(temporary_name)
    restored.unlink()
    try:
        restore_backup(manifest, restored)
        # This verifier registers Stocker's deterministic SQLite functions
        # before running schema, foreign-key, and quick integrity checks.
        verify_database(restored)
    finally:
        restored.unlink(missing_ok=True)
        Path(f"{restored}-wal").unlink(missing_ok=True)
        Path(f"{restored}-shm").unlink(missing_ok=True)


def _write_restart_marker(path: Path, *, now_us: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps({"created_at_us": now_us}, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@contextmanager
def _operation_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("backup operation lock is not a single regular file")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another quiescent managed backup is active") from error
        yield
    finally:
        os.close(descriptor)


def restart_after_interruption(
    *,
    tier: Literal["daily", "weekly"],
    restart_marker: Path | None = None,
) -> None:
    """Restart only when a killed backup left its durable restart marker."""

    restart_marker = RESTART_MARKERS[tier] if restart_marker is None else restart_marker
    if not restart_marker.is_file() or restart_marker.is_symlink():
        return
    if restart_marker.stat().st_size > 4_096:
        raise RuntimeError("backup restart marker exceeds its size bound")
    marker = json.loads(restart_marker.read_text(encoding="utf-8"))
    if set(marker) != {"created_at_us"} or not isinstance(marker["created_at_us"], int):
        raise RuntimeError("backup restart marker is invalid")
    errors = _restart_services(DATABASE)
    try:
        record_backup_failure(
            DESTINATION,
            code="BackupInterrupted",
            checked_at_us=marker["created_at_us"],
        )
    except Exception as error:
        errors.append(f"backup-status:{type(error).__name__}")
    if errors:
        raise RuntimeError("interrupted backup recovery failed: " + ",".join(errors))
    restart_marker.unlink()


def _run_quiescent_managed_backup(
    *,
    tier: Literal["daily", "weekly"],
    now_us: int,
    database: Path = DATABASE,
    destination: Path = DESTINATION,
    working_directory: Path = WORKING_DIRECTORY,
    restart_marker: Path,
) -> dict[str, object]:
    next_open_us = require_quiescent_backup_window(now_us)
    if _systemctl("is-active", RECORDER_UNIT) != 0:
        raise RuntimeError("recorder must be active before managed backup")
    if _systemctl("is-active", WEB_UNIT) != 0:
        raise RuntimeError("web must be active before managed backup")

    restart_required = False
    operation_error: Exception | None = None
    failure_status_error: Exception | None = None
    artifact: BackupArtifact | None = None
    payload: dict[str, object] | None = None
    try:
        _write_restart_marker(restart_marker, now_us=now_us)
        restart_required = True
        record_backup_failure(
            destination,
            code="BACKUP_IN_PROGRESS",
            checked_at_us=now_us,
        )
        _systemctl("stop", WEB_UNIT)
        _systemctl("stop", RECORDER_UNIT)
        if _systemctl("is-active", WEB_UNIT) != 3:
            raise RuntimeError("web did not become inactive")
        if _systemctl("is-active", RECORDER_UNIT) != 3:
            raise RuntimeError("recorder did not become inactive")

        lock = LocalWriterLock.for_database(database)
        try:
            lock.acquire()
            lock.verify_held()
            _require_no_database_descriptors(database)
            artifact = create_quiescent_backup(
                database,
                destination,
                tier=tier,
                precondition=lock.verify_held,
                created_at_us=now_us,
                policy=BackupPolicy(),
                working_directory=working_directory,
                post_publish_verify=lambda artifact: _restore_check(
                    artifact.manifest_path,
                    working_directory,
                ),
                defer_healthy_status=True,
            )
            lock.verify_held()
            _require_no_database_descriptors(database)
            payload = {
                "archive_filename": artifact.archive_path.name,
                "compressed_bytes": artifact.manifest.compressed_bytes,
                "manifest_filename": artifact.manifest_path.name,
                "next_xnys_open_us": next_open_us,
                "status": "ok",
                "tier": tier,
            }
        finally:
            lock.release()
    except Exception as error:
        operation_error = error
        if restart_required:
            try:
                record_backup_failure(
                    destination,
                    code=type(error).__name__[:96],
                    checked_at_us=now_us,
                )
            except Exception as status_error:
                failure_status_error = status_error
    finally:
        restart_errors: list[str] = []
        if restart_required:
            restart_errors = _restart_services(database)
            if not restart_errors:
                restart_marker.unlink(missing_ok=True)
        if restart_errors:
            raise RuntimeError(
                "service restart failed: " + ",".join(restart_errors)
            ) from operation_error
    if operation_error is not None:
        if failure_status_error is not None:
            raise RuntimeError("backup failure status publication failed") from failure_status_error
        raise operation_error
    assert artifact is not None
    finalize_quiescent_backup(
        artifact,
        destination,
        policy=BackupPolicy(),
        working_directory=working_directory,
    )
    assert payload is not None
    return payload


def run_quiescent_managed_backup(
    *,
    tier: Literal["daily", "weekly"],
    now_us: int,
    database: Path = DATABASE,
    destination: Path = DESTINATION,
    working_directory: Path = WORKING_DIRECTORY,
    restart_marker: Path | None = None,
    operation_lock: Path = OPERATION_LOCK,
) -> dict[str, object]:
    marker = RESTART_MARKERS[tier] if restart_marker is None else restart_marker
    with _operation_lock(operation_lock):
        return _run_quiescent_managed_backup(
            tier=tier,
            now_us=now_us,
            database=database,
            destination=destination,
            working_directory=working_directory,
            restart_marker=marker,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("daily", "weekly"))
    parser.add_argument("--restart-after-interruption", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.restart_after_interruption:
        if arguments.tier is None:
            raise SystemExit("--tier is required")
        restart_after_interruption(tier=arguments.tier)
        print(json.dumps({"status": "ok"}, sort_keys=True))
        return
    if arguments.tier is None:
        raise SystemExit("--tier is required")
    now_us = time.time_ns() // 1_000
    try:
        payload = run_quiescent_managed_backup(tier=arguments.tier, now_us=now_us)
    except Exception as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error), "status": "error"},
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
