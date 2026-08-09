"""Checked, compressed, rotating online backups for Stocker V2."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.storage.connection import SchemaError, verify_database

BackupTier = Literal["daily", "weekly"]
BackupState = Literal["healthy", "degraded", "unavailable"]

BACKUP_FORMAT_VERSION = 1
BACKUP_STATUS_FILENAME = "backup-status.json"
BACKUP_LOCK_FILENAME = ".stocker-v2-backup.lock"
DEFAULT_BACKUP_BYTE_CAP = 8 * 1024**3
DEFAULT_DAILY_RETENTION = 14
DEFAULT_WEEKLY_RETENTION = 12
DEFAULT_MINIMUM_PER_TIER = 2
DAILY_MAX_AGE_US = 36 * 60 * 60 * 1_000_000
WEEKLY_MAX_AGE_US = 8 * 24 * 60 * 60 * 1_000_000
MAX_BACKUP_MANIFEST_BYTES = 64 * 1024
MAX_BACKUP_STATUS_BYTES = 4 * 1024
MAX_MANIFEST_SCAN = 200
MAX_DIRECTORY_ENTRIES = 4_096
MAX_TIMESTAMP_US = 253_402_300_799_999_999
ONLINE_BACKUP_TIMEOUT_SECONDS = 30 * 60
_HASH_CHUNK_BYTES = 1024 * 1024
_MANIFEST_FIELDS = frozenset(
    {
        "archive_filename",
        "archive_mtime_ns",
        "compressed_bytes",
        "compressed_sha256",
        "created_at_us",
        "database_schema_version",
        "format_version",
        "quick_check",
        "source_database_filename",
        "tier",
        "uncompressed_bytes",
        "uncompressed_sha256",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "checked_at_us",
        "code",
        "format_version",
        "latest_manifest_filename",
        "state",
    }
)
_HEX_64 = frozenset("0123456789abcdef")


class BackupError(RuntimeError):
    """A backup operation failed without mutating the active database."""


class BackupIntegrityError(BackupError):
    """A backup or restore artifact failed structural or content verification."""


class BackupCapacityError(BackupError):
    """A new backup cannot fit without violating a retention floor."""


@dataclass(frozen=True)
class BackupPolicy:
    """Frozen tier counts and total backup-directory byte cap."""

    daily_retention: int = DEFAULT_DAILY_RETENTION
    weekly_retention: int = DEFAULT_WEEKLY_RETENTION
    byte_cap: int = DEFAULT_BACKUP_BYTE_CAP
    minimum_per_tier: int = DEFAULT_MINIMUM_PER_TIER

    def __post_init__(self) -> None:
        if self.minimum_per_tier < 1:
            raise ValueError("minimum_per_tier must be positive")
        if self.daily_retention < self.minimum_per_tier:
            raise ValueError("daily_retention is below the tier floor")
        if self.weekly_retention < self.minimum_per_tier:
            raise ValueError("weekly_retention is below the tier floor")
        if self.byte_cap < 1:
            raise ValueError("byte_cap must be positive")


@dataclass(frozen=True)
class BackupManifest:
    """Content-addressed recovery metadata committed after the archive."""

    format_version: int
    tier: BackupTier
    created_at_us: int
    source_database_filename: str
    archive_filename: str
    archive_mtime_ns: int
    database_schema_version: int
    quick_check: Literal["ok"]
    uncompressed_bytes: int
    compressed_bytes: int
    uncompressed_sha256: str
    compressed_sha256: str

    @property
    def manifest_filename(self) -> str:
        return f"{self.archive_filename}.manifest.json"

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "archive_filename": self.archive_filename,
            "archive_mtime_ns": self.archive_mtime_ns,
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
            "created_at_us": self.created_at_us,
            "database_schema_version": self.database_schema_version,
            "format_version": self.format_version,
            "quick_check": self.quick_check,
            "source_database_filename": self.source_database_filename,
            "tier": self.tier,
            "uncompressed_bytes": self.uncompressed_bytes,
            "uncompressed_sha256": self.uncompressed_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> BackupManifest:
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
            raise BackupIntegrityError("backup manifest has invalid fields")
        format_version = _strict_nonnegative_int(payload["format_version"], "format version")
        created_at_us = _strict_nonnegative_int(payload["created_at_us"], "creation time")
        database_schema_version = _strict_nonnegative_int(
            payload["database_schema_version"], "database schema version"
        )
        uncompressed_bytes = _strict_nonnegative_int(
            payload["uncompressed_bytes"], "uncompressed size"
        )
        compressed_bytes = _strict_nonnegative_int(payload["compressed_bytes"], "compressed size")
        archive_mtime_ns = _strict_nonnegative_int(payload["archive_mtime_ns"], "archive mtime")
        tier = payload["tier"]
        source_name = payload["source_database_filename"]
        archive_name = payload["archive_filename"]
        quick_check = payload["quick_check"]
        uncompressed_sha256 = payload["uncompressed_sha256"]
        compressed_sha256 = payload["compressed_sha256"]
        if format_version != BACKUP_FORMAT_VERSION:
            raise BackupIntegrityError("backup manifest format is unsupported")
        if created_at_us > MAX_TIMESTAMP_US:
            raise BackupIntegrityError("backup manifest creation time is out of range")
        if tier not in {"daily", "weekly"}:
            raise BackupIntegrityError("backup manifest tier is invalid")
        if not _safe_filename(source_name):
            raise BackupIntegrityError("backup source filename is invalid")
        if archive_name != _archive_filename(cast(BackupTier, tier), created_at_us):
            raise BackupIntegrityError("backup archive filename does not match its manifest")
        if quick_check != "ok":
            raise BackupIntegrityError("backup quick_check did not pass")
        if uncompressed_bytes < 1 or compressed_bytes < 1:
            raise BackupIntegrityError("backup sizes must be positive")
        if not _sha256_text(uncompressed_sha256) or not _sha256_text(compressed_sha256):
            raise BackupIntegrityError("backup manifest hash is invalid")
        return cls(
            format_version=format_version,
            tier=cast(BackupTier, tier),
            created_at_us=created_at_us,
            source_database_filename=cast(str, source_name),
            archive_filename=cast(str, archive_name),
            archive_mtime_ns=archive_mtime_ns,
            database_schema_version=database_schema_version,
            quick_check="ok",
            uncompressed_bytes=uncompressed_bytes,
            compressed_bytes=compressed_bytes,
            uncompressed_sha256=cast(str, uncompressed_sha256),
            compressed_sha256=cast(str, compressed_sha256),
        )


@dataclass(frozen=True)
class BackupArtifact:
    manifest: BackupManifest
    archive_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RestoreResult:
    destination: Path
    uncompressed_bytes: int
    uncompressed_sha256: str


@dataclass(frozen=True)
class BackupStatus:
    state: BackupState
    checked_at_us: int | None
    code: str | None
    latest_manifest_filename: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "checked_at_us": self.checked_at_us,
            "code": self.code,
            "format_version": BACKUP_FORMAT_VERSION,
            "latest_manifest_filename": self.latest_manifest_filename,
            "state": self.state,
        }


@dataclass(frozen=True)
class BackupManifestEntry:
    manifest_filename: str
    archive_filename: str
    tier: BackupTier
    created_at_us: int
    compressed_bytes: int
    uncompressed_bytes: int
    quick_check: Literal["ok"]

    @classmethod
    def from_manifest(cls, manifest: BackupManifest) -> BackupManifestEntry:
        return cls(
            manifest_filename=manifest.manifest_filename,
            archive_filename=manifest.archive_filename,
            tier=manifest.tier,
            created_at_us=manifest.created_at_us,
            compressed_bytes=manifest.compressed_bytes,
            uncompressed_bytes=manifest.uncompressed_bytes,
            quick_check=manifest.quick_check,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "archive_filename": self.archive_filename,
            "compressed_bytes": self.compressed_bytes,
            "created_at_us": self.created_at_us,
            "manifest_filename": self.manifest_filename,
            "quick_check": self.quick_check,
            "tier": self.tier,
            "uncompressed_bytes": self.uncompressed_bytes,
        }


@dataclass(frozen=True)
class BackupProjection:
    available: bool
    items: tuple[BackupManifestEntry, ...]
    truncated: bool
    status: BackupStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "items": [item.to_dict() for item in self.items],
            "status": self.status.to_dict(),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class _ManagedBackup:
    manifest: BackupManifest
    archive_path: Path
    manifest_path: Path

    @property
    def bytes(self) -> int:
        return self.archive_path.stat().st_size + self.manifest_path.stat().st_size


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackupIntegrityError(f"backup manifest {label} is invalid")
    return value


def _safe_filename(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 255
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
        and value.isascii()
        and all(character.isalnum() or character in "._-" for character in value)
    )


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_64


def _archive_filename(tier: BackupTier, created_at_us: int) -> str:
    instant = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=created_at_us)
    timestamp = instant.strftime("%Y%m%dT%H%M%S") + f".{created_at_us % 1_000_000:06d}Z"
    return f"stocker-v2-{tier}-{timestamp}.sqlite3.gz"


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BackupIntegrityError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BackupIntegrityError(f"{label} must be one regular unlinked file")
    return metadata


def _read_bounded_json(path: Path, *, maximum_bytes: int, label: str) -> object:
    metadata = _regular_file(path, label=label)
    if metadata.st_size < 2 or metadata.st_size > maximum_bytes:
        raise BackupIntegrityError(f"{label} size is invalid")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackupIntegrityError(f"{label} is invalid JSON") from error


def load_backup_manifest(
    path: str | Path,
    *,
    verify_compressed_hash: bool = False,
) -> BackupManifest:
    """Load one strict manifest and verify its local archive identity and size."""

    manifest_path = Path(path)
    manifest = BackupManifest.from_dict(
        _read_bounded_json(
            manifest_path,
            maximum_bytes=MAX_BACKUP_MANIFEST_BYTES,
            label="backup manifest",
        )
    )
    if manifest_path.name != manifest.manifest_filename:
        raise BackupIntegrityError("backup manifest filename does not match its content")
    archive = manifest_path.parent / manifest.archive_filename
    metadata = _regular_file(archive, label="backup archive")
    if metadata.st_size != manifest.compressed_bytes:
        raise BackupIntegrityError("backup archive size does not match its manifest")
    if metadata.st_mtime_ns != manifest.archive_mtime_ns:
        raise BackupIntegrityError("backup archive modification time does not match its manifest")
    if verify_compressed_hash:
        compressed_sha256, compressed_bytes = _hash_file(archive)
        if (
            compressed_bytes != manifest.compressed_bytes
            or compressed_sha256 != manifest.compressed_sha256
        ):
            raise BackupIntegrityError("backup archive hash does not match its manifest")
    return manifest


def _read_status(directory: Path) -> BackupStatus:
    path = directory / BACKUP_STATUS_FILENAME
    if not path.exists():
        return BackupStatus("unavailable", None, None, None)
    try:
        payload = _read_bounded_json(
            path,
            maximum_bytes=MAX_BACKUP_STATUS_BYTES,
            label="backup status",
        )
        if not isinstance(payload, dict) or set(payload) != _STATUS_FIELDS:
            raise BackupIntegrityError("backup status has invalid fields")
        if payload["format_version"] != BACKUP_FORMAT_VERSION:
            raise BackupIntegrityError("backup status format is unsupported")
        state = payload["state"]
        checked_at_us = payload["checked_at_us"]
        code = payload["code"]
        latest = payload["latest_manifest_filename"]
        if state not in {"healthy", "degraded"}:
            raise BackupIntegrityError("backup status state is invalid")
        checked = _strict_nonnegative_int(checked_at_us, "status time")
        if code is not None and (
            not isinstance(code, str) or not code or len(code) > 96 or not code.isascii()
        ):
            raise BackupIntegrityError("backup status code is invalid")
        if latest is not None and not _safe_filename(latest):
            raise BackupIntegrityError("backup status latest manifest is invalid")
        return BackupStatus(
            cast(BackupState, state),
            checked,
            code,
            latest,
        )
    except BackupIntegrityError:
        return BackupStatus("degraded", None, "BACKUP_STATUS_INVALID", None)


def read_backup_manifests(
    directory: str | Path,
    *,
    limit: int = 200,
    now_us: int | None = None,
) -> BackupProjection:
    """Read only strict bounded manifests for diagnostics; ignore all other files."""

    if not 1 <= limit <= 200:
        raise ValueError("backup manifest limit must be between 1 and 200")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        return BackupProjection(False, (), False, BackupStatus("unavailable", None, None, None))
    candidates: list[Path] = []
    truncated = False
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.endswith(".manifest.json"):
                    continue
                if len(candidates) >= MAX_MANIFEST_SCAN:
                    truncated = True
                    break
                candidates.append(root / entry.name)
    except OSError:
        return BackupProjection(False, (), False, BackupStatus("unavailable", None, None, None))
    candidates.sort(key=lambda path: path.name, reverse=True)
    items: list[BackupManifestEntry] = []
    invalid_manifests = 0
    for path in candidates:
        try:
            manifest = load_backup_manifest(path)
        except BackupIntegrityError:
            invalid_manifests += 1
            continue
        items.append(BackupManifestEntry.from_manifest(manifest))
    items.sort(key=lambda item: (item.created_at_us, item.archive_filename), reverse=True)
    truncated = truncated or len(items) > limit
    checked_at_us = time.time_ns() // 1_000 if now_us is None else now_us
    if isinstance(checked_at_us, bool) or not isinstance(checked_at_us, int) or checked_at_us < 0:
        raise ValueError("backup health time must be a nonnegative integer")
    status = _project_backup_health(
        persisted=_read_status(root),
        items=tuple(items),
        invalid_manifests=invalid_manifests,
        now_us=checked_at_us,
    )
    return BackupProjection(True, tuple(items[:limit]), truncated, status)


def _project_backup_health(
    *,
    persisted: BackupStatus,
    items: tuple[BackupManifestEntry, ...],
    invalid_manifests: int,
    now_us: int,
) -> BackupStatus:
    def degraded(code: str) -> BackupStatus:
        return BackupStatus(
            "degraded",
            persisted.checked_at_us,
            code,
            persisted.latest_manifest_filename,
        )

    if persisted.state == "unavailable":
        return degraded("BACKUP_STATUS_UNAVAILABLE") if items else persisted
    if persisted.state == "degraded":
        return persisted
    if invalid_manifests:
        return degraded("BACKUP_MANIFEST_INVALID")
    manifest_names = {item.manifest_filename for item in items}
    if persisted.latest_manifest_filename not in manifest_names:
        return degraded("BACKUP_LATEST_INVALID")
    by_tier = {
        tier: tuple(item for item in items if item.tier == tier) for tier in ("daily", "weekly")
    }
    if any(len(tier_items) < DEFAULT_MINIMUM_PER_TIER for tier_items in by_tier.values()):
        return degraded("BACKUP_TIER_FLOOR_UNMET")
    latest_daily = max(item.created_at_us for item in by_tier["daily"])
    latest_weekly = max(item.created_at_us for item in by_tier["weekly"])
    if latest_daily > now_us or latest_weekly > now_us:
        return degraded("BACKUP_CLOCK_INVALID")
    if now_us - latest_daily > DAILY_MAX_AGE_US:
        return degraded("BACKUP_DAILY_STALE")
    if now_us - latest_weekly > WEEKLY_MAX_AGE_US:
        return degraded("BACKUP_WEEKLY_STALE")
    return persisted


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o640) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".stocker-v2-atomic-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _status_payload(
    *,
    state: Literal["healthy", "degraded"],
    checked_at_us: int,
    code: str | None,
    latest_manifest_filename: str | None,
) -> bytes:
    status = BackupStatus(state, checked_at_us, code, latest_manifest_filename)
    return canonical_json_bytes(cast(JsonValue, status.to_dict())) + b"\n"


def _write_status(
    directory: Path,
    *,
    state: Literal["healthy", "degraded"],
    checked_at_us: int,
    code: str | None,
    latest_manifest_filename: str | None,
) -> None:
    _write_atomic(
        directory / BACKUP_STATUS_FILENAME,
        _status_payload(
            state=state,
            checked_at_us=checked_at_us,
            code=code,
            latest_manifest_filename=latest_manifest_filename,
        ),
    )


def _prepare_backup_directory(path: Path) -> Path:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise BackupError("backup destination must be a real directory")
    else:
        path.mkdir(parents=True, mode=0o750)
    return path.resolve()


def _prepare_working_directory(path: Path, *, backup_directory: Path) -> Path:
    working = _prepare_backup_directory(path)
    if working == backup_directory or backup_directory in working.parents:
        raise BackupError("backup working directory must be outside the archive directory")
    if working.stat().st_dev != backup_directory.stat().st_dev:
        raise BackupError("backup working and archive directories must share a filesystem")
    return working


def _remove_stale_regular_files(
    directory: Path,
    *,
    matches: Callable[[str], bool],
    label: str,
) -> None:
    removed = False
    scanned = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            scanned += 1
            if scanned > MAX_DIRECTORY_ENTRIES:
                raise BackupCapacityError(f"{label} directory entry bound exceeded")
            if not matches(entry.name):
                continue
            metadata = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise BackupIntegrityError(f"{label} contains an unsafe managed path")
            (directory / entry.name).unlink()
            removed = True
    if removed:
        _fsync_directory(directory)


def _is_backup_work_file(name: str) -> bool:
    database_prefix = ".stocker-v2-database-"
    database_suffixes = (
        ".sqlite3",
        ".sqlite3-journal",
        ".sqlite3-shm",
        ".sqlite3-wal",
    )
    return (name.startswith(database_prefix) and name.endswith(database_suffixes)) or (
        name.startswith(".stocker-v2-archive-") and name.endswith(".gz")
    )


def _remove_stale_backup_work(working_directory: Path) -> None:
    _remove_stale_regular_files(
        working_directory,
        matches=_is_backup_work_file,
        label="backup working",
    )


def _remove_stale_atomic_metadata(destination: Path) -> None:
    _remove_stale_regular_files(
        destination,
        matches=lambda name: name.startswith(".stocker-v2-atomic-"),
        label="backup destination",
    )


def _is_managed_archive_filename(name: str) -> bool:
    for tier in ("daily", "weekly"):
        prefix = f"stocker-v2-{tier}-"
        suffix = ".sqlite3.gz"
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        timestamp = name[len(prefix) : -len(suffix)]
        try:
            instant = datetime.strptime(timestamp, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC)
        except ValueError:
            return False
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        elapsed = instant - epoch
        created_at_us = (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
        if not 0 <= created_at_us <= MAX_TIMESTAMP_US:
            return False
        return _archive_filename(tier, created_at_us) == name
    return False


def _remove_interrupted_publication_orphans(
    destination: Path,
    *,
    previous_status: BackupStatus,
) -> None:
    if previous_status.state != "degraded" or previous_status.code != "BACKUP_IN_PROGRESS":
        return
    names: set[str] = set()
    with os.scandir(destination) as entries:
        for scanned, entry in enumerate(entries, start=1):
            if scanned > MAX_DIRECTORY_ENTRIES:
                raise BackupCapacityError("backup destination directory entry bound exceeded")
            names.add(entry.name)
    orphan_names: set[str] = set()
    for name in names:
        if _is_managed_archive_filename(name):
            if f"{name}.manifest.json" not in names:
                orphan_names.add(name)
            continue
        manifest_suffix = ".manifest.json"
        if name.endswith(manifest_suffix):
            archive_name = name[: -len(manifest_suffix)]
            if _is_managed_archive_filename(archive_name) and archive_name not in names:
                orphan_names.add(name)
    if orphan_names:
        _remove_stale_regular_files(
            destination,
            matches=orphan_names.__contains__,
            label="interrupted backup publication",
        )


def record_backup_failure(
    destination: str | Path,
    *,
    code: str,
    checked_at_us: int | None = None,
) -> None:
    """Publish a stable degraded status for a failed backup service invocation."""

    if not code or len(code) > 96 or not code.isascii():
        raise ValueError("backup failure code must be bounded ASCII")
    timestamp = time.time_ns() // 1_000 if checked_at_us is None else checked_at_us
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("backup failure time must be a nonnegative integer")
    root = _prepare_backup_directory(Path(destination))
    with _destination_lock(root):
        previous = _read_status(root)
        _write_status(
            root,
            state="degraded",
            checked_at_us=timestamp,
            code=code,
            latest_manifest_filename=previous.latest_manifest_filename,
        )


@contextmanager
def _destination_lock(directory: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / BACKUP_LOCK_FILENAME, flags, 0o640)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BackupIntegrityError("backup destination lock must be one regular file")
        os.fchmod(descriptor, 0o640)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _online_copy(source: Path, destination: Path) -> None:
    uri = source.resolve().as_uri() + "?mode=ro"
    reader = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
    writer = sqlite3.connect(destination, isolation_level=None, timeout=5.0)
    deadline = time.monotonic() + ONLINE_BACKUP_TIMEOUT_SECONDS

    def enforce_deadline(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise BackupError("online backup exceeded its 30-minute time budget")

    try:
        reader.execute("PRAGMA query_only = ON")
        reader.execute("PRAGMA busy_timeout = 5000")
        writer.execute("PRAGMA synchronous = FULL")
        reader.backup(writer, pages=256, progress=enforce_deadline, sleep=0.01)
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("PRAGMA journal_mode = DELETE")
    finally:
        writer.close()
        reader.close()
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def _compress(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as raw_destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_destination,
            mtime=0,
        ) as compressed:
            shutil.copyfileobj(source_handle, compressed, length=_HASH_CHUNK_BYTES)
        raw_destination.flush()
        os.fsync(raw_destination.fileno())


def _managed_backups(directory: Path) -> tuple[_ManagedBackup, ...]:
    candidates: list[Path] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(".manifest.json"):
                continue
            if len(candidates) >= MAX_MANIFEST_SCAN:
                raise BackupCapacityError("backup manifest scan bound exceeded")
            candidates.append(directory / entry.name)
    managed: list[_ManagedBackup] = []
    for path in candidates:
        try:
            manifest = load_backup_manifest(path, verify_compressed_hash=True)
        except BackupIntegrityError:
            continue
        managed.append(_ManagedBackup(manifest, directory / manifest.archive_filename, path))
    return tuple(managed)


def _directory_bytes(directory: Path, *, excluded: frozenset[Path]) -> int:
    total = 0
    scanned = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            scanned += 1
            if scanned > MAX_DIRECTORY_ENTRIES:
                raise BackupCapacityError("backup directory entry bound exceeded")
            path = directory / entry.name
            if path in excluded or entry.is_symlink():
                continue
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def _rotation_victims(
    managed: tuple[_ManagedBackup, ...],
    *,
    candidate: BackupManifest,
    candidate_manifest_bytes: int,
    candidate_status_bytes: int,
    directory_bytes: int,
    current_status_bytes: int,
    policy: BackupPolicy,
) -> tuple[_ManagedBackup, ...]:
    marker = object()
    items: list[tuple[int, str, _ManagedBackup | object]] = [
        (item.manifest.created_at_us, item.manifest.archive_filename, item) for item in managed
    ]
    items.append((candidate.created_at_us, candidate.archive_filename, marker))
    victims: list[_ManagedBackup] = []
    victim_ids: set[str] = set()
    counts: dict[BackupTier, int] = {"daily": 0, "weekly": 0}
    tier_retentions: tuple[tuple[BackupTier, int], ...] = (
        ("daily", policy.daily_retention),
        ("weekly", policy.weekly_retention),
    )
    for tier, retention in tier_retentions:
        tier_items = sorted(
            (
                item
                for item in items
                if (
                    candidate.tier
                    if item[2] is marker
                    else cast(_ManagedBackup, item[2]).manifest.tier
                )
                == tier
            ),
            key=lambda item: (item[0], item[1]),
        )
        counts[tier] = len(tier_items)
        for _, _, item in tier_items[: max(0, len(tier_items) - retention)]:
            if item is marker:
                raise BackupCapacityError("new backup falls outside its retained tier window")
            managed_item = cast(_ManagedBackup, item)
            victims.append(managed_item)
            victim_ids.add(managed_item.manifest.archive_filename)
            counts[tier] -= 1

    projected = (
        directory_bytes
        - current_status_bytes
        + candidate.compressed_bytes
        + candidate_manifest_bytes
        + candidate_status_bytes
        - sum(item.bytes for item in victims)
    )
    if projected <= policy.byte_cap:
        return tuple(victims)
    tiers: tuple[BackupTier, ...] = ("daily", "weekly")
    for tier in tiers:
        candidates = sorted(
            (
                item
                for item in managed
                if item.manifest.tier == tier and item.manifest.archive_filename not in victim_ids
            ),
            key=lambda item: (item.manifest.created_at_us, item.manifest.archive_filename),
        )
        for item in candidates:
            if projected <= policy.byte_cap or counts[tier] <= policy.minimum_per_tier:
                break
            victims.append(item)
            victim_ids.add(item.manifest.archive_filename)
            counts[tier] -= 1
            projected -= item.bytes
    if projected > policy.byte_cap:
        floor = policy.minimum_per_tier
        floor_word = "two" if floor == 2 else str(floor)
        raise BackupCapacityError(
            f"new backup exceeds the byte cap while preserving {floor_word} valid archives per tier"
        )
    return tuple(victims)


def create_backup(
    database: str | Path,
    destination: str | Path,
    *,
    tier: BackupTier,
    created_at_us: int | None = None,
    policy: BackupPolicy | None = None,
    working_directory: str | Path | None = None,
) -> BackupArtifact:
    """Serialize one checked online backup for a destination."""

    root = _prepare_backup_directory(Path(destination))
    with _destination_lock(root):
        _remove_stale_atomic_metadata(root)
        _remove_interrupted_publication_orphans(root, previous_status=_read_status(root))
        working = _prepare_working_directory(
            Path(tempfile.gettempdir()) if working_directory is None else Path(working_directory),
            backup_directory=root,
        )
        return _create_backup_locked(
            database,
            root,
            tier=tier,
            created_at_us=created_at_us,
            policy=policy,
            working_directory=working,
        )


def _create_backup_locked(
    database: str | Path,
    destination: str | Path,
    *,
    tier: BackupTier,
    created_at_us: int | None = None,
    policy: BackupPolicy | None = None,
    working_directory: Path,
) -> BackupArtifact:
    """Create one checked online backup and atomically commit its strict manifest."""

    if tier not in {"daily", "weekly"}:
        raise ValueError("backup tier must be daily or weekly")
    source = Path(database)
    if source.is_symlink() or not source.is_file():
        raise BackupError("backup source must be an existing regular database")
    root = _prepare_backup_directory(Path(destination))
    if source.resolve().parent == root:
        raise BackupError("active database may not reside in the backup directory")
    frozen_policy = policy or BackupPolicy()
    timestamp = time.time_ns() // 1_000 if created_at_us is None else created_at_us
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or not 0 <= timestamp <= MAX_TIMESTAMP_US
    ):
        raise ValueError("created_at_us must be a supported nonnegative timestamp")
    archive_name = _archive_filename(tier, timestamp)
    archive_path = root / archive_name
    manifest_path = root / f"{archive_name}.manifest.json"
    if archive_path.exists() or manifest_path.exists():
        raise BackupError("backup identity already exists")

    previous_status = _read_status(root)
    _write_status(
        root,
        state="degraded",
        checked_at_us=timestamp,
        code="BACKUP_IN_PROGRESS",
        latest_manifest_filename=previous_status.latest_manifest_filename,
    )
    _remove_stale_backup_work(working_directory)
    temporary_database: Path | None = None
    temporary_archive: Path | None = None
    published_archive = False
    published_manifest = False
    committed = False
    try:
        database_descriptor, database_temporary_name = tempfile.mkstemp(
            prefix=".stocker-v2-database-", suffix=".sqlite3", dir=working_directory
        )
        os.close(database_descriptor)
        temporary_database = Path(database_temporary_name)
        archive_descriptor, archive_temporary_name = tempfile.mkstemp(
            prefix=".stocker-v2-archive-", suffix=".gz", dir=working_directory
        )
        os.close(archive_descriptor)
        temporary_archive = Path(archive_temporary_name)
        temporary_archive.unlink()
        _online_copy(source, temporary_database)
        try:
            verify_database(temporary_database)
        except (OSError, SchemaError, sqlite3.Error) as error:
            raise BackupIntegrityError("online backup database verification failed") from error
        with sqlite3.connect(
            f"{temporary_database.resolve().as_uri()}?mode=ro", uri=True
        ) as verified:
            quick_check = str(verified.execute("PRAGMA quick_check").fetchone()[0])
            schema_version = int(
                verified.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                ).fetchone()[0]
            )
        if quick_check != "ok":
            raise BackupIntegrityError("online backup quick_check failed")
        uncompressed_sha256, uncompressed_bytes = _hash_file(temporary_database)
        _compress(temporary_database, temporary_archive)
        compressed_sha256, compressed_bytes = _hash_file(temporary_archive)
        manifest = BackupManifest(
            format_version=BACKUP_FORMAT_VERSION,
            tier=tier,
            created_at_us=timestamp,
            source_database_filename=source.name,
            archive_filename=archive_name,
            archive_mtime_ns=temporary_archive.stat().st_mtime_ns,
            database_schema_version=schema_version,
            quick_check="ok",
            uncompressed_bytes=uncompressed_bytes,
            compressed_bytes=compressed_bytes,
            uncompressed_sha256=uncompressed_sha256,
            compressed_sha256=compressed_sha256,
        )
        manifest_payload = canonical_json_bytes(cast(JsonValue, manifest.to_dict())) + b"\n"
        status_payload = _status_payload(
            state="healthy",
            checked_at_us=timestamp,
            code=None,
            latest_manifest_filename=manifest.manifest_filename,
        )
        status_path = root / BACKUP_STATUS_FILENAME
        current_status_bytes = (
            status_path.lstat().st_size
            if status_path.exists() and stat.S_ISREG(status_path.lstat().st_mode)
            else 0
        )
        managed = _managed_backups(root)
        victims = _rotation_victims(
            managed,
            candidate=manifest,
            candidate_manifest_bytes=len(manifest_payload),
            candidate_status_bytes=len(status_payload),
            directory_bytes=_directory_bytes(
                root,
                excluded=frozenset({temporary_database, temporary_archive}),
            ),
            current_status_bytes=current_status_bytes,
            policy=frozen_policy,
        )
        os.chmod(temporary_archive, 0o640)
        os.replace(temporary_archive, archive_path)
        published_archive = True
        _fsync_directory(root)
        _write_atomic(manifest_path, manifest_payload)
        published_manifest = True
        committed = True
        for victim in victims:
            victim.manifest_path.unlink()
            victim.archive_path.unlink()
        _fsync_directory(root)
        _write_atomic(status_path, status_payload)
        return BackupArtifact(manifest, archive_path, manifest_path)
    except Exception as error:
        if not committed:
            if published_manifest:
                manifest_path.unlink(missing_ok=True)
            if published_archive:
                archive_path.unlink(missing_ok=True)
        with suppress(Exception):
            _write_status(
                root,
                state="degraded",
                checked_at_us=timestamp,
                code=type(error).__name__[:96],
                latest_manifest_filename=(
                    manifest_path.name if committed else previous_status.latest_manifest_filename
                ),
            )
        raise
    finally:
        if temporary_database is not None:
            temporary_database.unlink(missing_ok=True)
            Path(f"{temporary_database}-journal").unlink(missing_ok=True)
            Path(f"{temporary_database}-wal").unlink(missing_ok=True)
            Path(f"{temporary_database}-shm").unlink(missing_ok=True)
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)


def restore_backup(manifest: str | Path, destination: str | Path) -> RestoreResult:
    """Verify and restore one archive into a new path without overwriting any file."""

    manifest_path = Path(manifest)
    loaded = load_backup_manifest(manifest_path)
    archive_path = manifest_path.parent / loaded.archive_filename
    compressed_sha256, compressed_bytes = _hash_file(archive_path)
    if compressed_bytes != loaded.compressed_bytes:
        raise BackupIntegrityError("compressed size changed during restore")
    if compressed_sha256 != loaded.compressed_sha256:
        raise BackupIntegrityError("compressed hash does not match the manifest")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"restore destination already exists: {target}")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise BackupError("restore destination parent must be a real directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stocker-v2-restore-", suffix=".sqlite3", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    restored_bytes = 0
    published_target = False
    try:
        with os.fdopen(descriptor, "wb") as output, gzip.open(archive_path, "rb") as compressed:
            while chunk := compressed.read(_HASH_CHUNK_BYTES):
                restored_bytes += len(chunk)
                if restored_bytes > loaded.uncompressed_bytes:
                    raise BackupIntegrityError("uncompressed backup exceeds its manifest size")
                output.write(chunk)
                digest.update(chunk)
            os.fchmod(output.fileno(), 0o640)
            output.flush()
            os.fsync(output.fileno())
        if restored_bytes != loaded.uncompressed_bytes:
            raise BackupIntegrityError("uncompressed size does not match the manifest")
        uncompressed_sha256 = digest.hexdigest()
        if uncompressed_sha256 != loaded.uncompressed_sha256:
            raise BackupIntegrityError("uncompressed hash does not match the manifest")
        try:
            verify_database(temporary)
        except (OSError, SchemaError, sqlite3.Error) as error:
            raise BackupIntegrityError("restored database verification failed") from error
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise
        published_target = True
        _fsync_directory(target.parent)
        return RestoreResult(target, restored_bytes, uncompressed_sha256)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        if published_target:
            target.unlink(missing_ok=True)
            with suppress(OSError):
                _fsync_directory(target.parent)
        if isinstance(error, FileExistsError):
            raise
        raise BackupIntegrityError("compressed backup could not be restored") from error
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)
