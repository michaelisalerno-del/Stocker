"""Local single-writer ownership and audited fatal-generation recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from stocker_runtime.storage import RetentionPolicy, connect_v2, verify_database

OWNERSHIP_PROTOCOL = "local_flock_v1"
RECOVERABLE_FATAL_CODES = frozenset(
    {
        # This legacy boundary occurs only after durable raw admission. Phase 3 replaces
        # it with narrow derived-component incidents; retained callbacks can be replayed.
        "POST_ADMISSION_PRESERVATION_FAILED",
    }
)


class LocalWriterLockError(RuntimeError):
    """The local operating-system writer lock cannot be acquired safely."""


@dataclass
class LocalWriterLock:
    """A nonblocking process-lifetime kernel lock beside the operational database."""

    path: Path
    _descriptor: int | None = None

    @classmethod
    def for_database(cls, database: str | Path) -> LocalWriterLock:
        path = Path(database).resolve(strict=False)
        return cls(path.with_name(f"{path.name}.writer.lock"))

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise LocalWriterLockError("local writer lock is already held by this recorder")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o640)
        except OSError as error:
            raise LocalWriterLockError(
                f"local writer lock cannot be opened: {self.path}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LocalWriterLockError("local writer lock must be one regular file")
            os.fchmod(descriptor, 0o640)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LocalWriterLockError("local authoritative writer lock is held") from error
            self._descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def verify_held(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            raise LocalWriterLockError("local authoritative writer lock is not held")
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise LocalWriterLockError("local authoritative writer lock was lost") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LocalWriterLockError("local authoritative writer lock identity changed")

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    def __enter__(self) -> LocalWriterLock:
        self.acquire()
        return self

    def __exit__(self, *_error: object) -> None:
        self.release()


def recover_fatal_generation(
    *,
    database: Path,
    run_id: str,
    generation: int,
    mode: Literal["prospective_record", "shadow"],
    config_hash: str,
    input_hash: str,
    fatal_code: str,
    operator: str,
    reason: str,
    authorized_at_us: int,
) -> None:
    """Authorize one eligible fatal generation for same-lineage restart."""

    if generation < 1 or authorized_at_us < 0:
        raise ValueError("fatal recovery generation must be positive and time nonnegative")
    if len(input_hash) != 64 or any(
        character not in "0123456789abcdef" for character in input_hash
    ):
        raise ValueError("fatal recovery input hash must be 64 lowercase hexadecimal characters")
    if not operator.strip() or len(operator) > 256:
        raise ValueError("fatal recovery operator must be bounded non-empty text")
    if not reason.strip() or len(reason) > 1_024:
        raise ValueError("fatal recovery reason must be bounded non-empty text")
    if fatal_code not in RECOVERABLE_FATAL_CODES:
        raise LocalWriterLockError(f"fatal code is not recoverable: {fatal_code}")

    lock = LocalWriterLock.for_database(database)
    lock.acquire()
    try:
        verify_database(database)
        with connect_v2(database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            wal_path = Path(f"{Path(database).resolve(strict=False)}-wal")
            wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
            policy = RetentionPolicy()
            if page_count * page_size >= policy.database_cap_bytes:
                raise LocalWriterLockError("fatal recovery blocked by the database hard cap")
            if wal_bytes >= policy.wal_cap_bytes:
                raise LocalWriterLockError("fatal recovery blocked by the WAL hard cap")
            row = connection.execute(
                "SELECT run.mode, run.config_hash, run.status, state.recorder_generation, "
                "state.lifecycle, state.reason, generation.termination_code, "
                "generation.ended_at_us, generation.input_hash, "
                "generation.fatal_recovery_authorized_at_us "
                "FROM runs run JOIN runtime_state state ON state.run_id=run.run_id "
                "JOIN recorder_generations generation ON generation.run_id=state.run_id "
                "AND generation.generation=state.recorder_generation "
                "WHERE run.run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise LocalWriterLockError("fatal recovery run is unavailable")
            if str(row["mode"]) != mode or str(row["config_hash"]) != config_hash:
                raise LocalWriterLockError("fatal recovery run identity is incompatible")
            if row["input_hash"] is None:
                raise LocalWriterLockError(
                    "fatal generation predates frozen input identity and cannot be recovered"
                )
            if str(row["input_hash"]) != input_hash:
                raise LocalWriterLockError("fatal recovery market-data input is incompatible")
            if int(row["recorder_generation"]) != generation:
                raise LocalWriterLockError("fatal recovery generation is not current")
            if (
                str(row["status"]) != "fatal"
                or str(row["lifecycle"]) != "fatal"
                or str(row["reason"]) != fatal_code
                or str(row["termination_code"]) != fatal_code
                or row["ended_at_us"] is None
            ):
                raise LocalWriterLockError("fatal recovery evidence does not match exactly")
            if row["fatal_recovery_authorized_at_us"] is not None:
                raise LocalWriterLockError("fatal generation recovery is already authorized")
            connection.execute(
                "UPDATE recorder_generations SET fatal_recovery_authorized_at_us=?, "
                "fatal_recovery_operator=?, fatal_recovery_reason=?, recovered_fatal_code=? "
                "WHERE run_id=? AND generation=?",
                (
                    authorized_at_us,
                    operator.strip(),
                    reason.strip(),
                    fatal_code,
                    run_id,
                    generation,
                ),
            )
            details = json.dumps(
                {
                    "fatal_code": fatal_code,
                    "generation": generation,
                    "operator": operator.strip(),
                    "reason": reason.strip(),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            incident_id = hashlib.sha256(
                f"{run_id}|{generation}|FATAL_GENERATION_RECOVERY_AUTHORIZED".encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "opened_at_us, details_json) VALUES (?, ?, 'recorder', 'info', "
                "'FATAL_GENERATION_RECOVERY_AUTHORIZED', ?, ?)",
                (incident_id, run_id, authorized_at_us, details),
            )
            connection.execute(
                "UPDATE runs SET status='running', ended_at_us=NULL WHERE run_id=?",
                (run_id,),
            )
            connection.execute(
                "UPDATE runtime_state SET lifecycle='stopped', "
                "reason='FATAL_GENERATION_RECOVERY_AUTHORIZED', "
                "connection_state='disconnected', process_heartbeat_at_us=? WHERE run_id=? "
                "AND recorder_generation=?",
                (authorized_at_us, run_id, generation),
            )
            connection.commit()
    finally:
        lock.release()
