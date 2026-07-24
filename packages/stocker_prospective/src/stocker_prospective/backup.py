"""Consistent online SQLite backups for append-oriented prospective evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DatabaseBackup(BaseModel):
    model_config = ConfigDict(frozen=True)

    backup_file: Path
    manifest_file: Path
    created_at_utc: datetime
    sha256: str
    quick_check: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_database(
    database: str | Path,
    destination: str | Path,
    *,
    now: datetime | None = None,
) -> DatabaseBackup:
    """Use SQLite's online-backup API without copying WAL files by hand."""

    source_path = Path(database)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"prospective-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    backup_path = root / f"{stem}.sqlite3"
    manifest_path = root / f"{stem}.json"
    if backup_path.exists() or manifest_path.exists():
        raise FileExistsError(f"backup already exists for {timestamp.isoformat()}")
    with sqlite3.connect(source_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
        row = target.execute("PRAGMA quick_check").fetchone()
    quick_check = "unavailable" if row is None else str(row[0])
    if quick_check != "ok":
        backup_path.unlink(missing_ok=True)
        raise RuntimeError(f"backup quick_check failed: {quick_check}")
    digest = _sha256(backup_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "created_at_utc": timestamp.isoformat(),
                "database_filename": source_path.name,
                "backup_filename": backup_path.name,
                "sha256": digest,
                "quick_check": quick_check,
                "retention_policy": "immutable_until_explicit_operator_removal",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(backup_path, 0o600)
    os.chmod(manifest_path, 0o600)
    return DatabaseBackup(
        backup_file=backup_path,
        manifest_file=manifest_path,
        created_at_utc=timestamp,
        sha256=digest,
        quick_check=quick_check,
    )
