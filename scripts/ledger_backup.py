"""Consistent, offline-verifiable SLRNO ledger and configuration backup. No broker access.

Credentials, OAuth tokens and event archives are deliberately excluded; back those up with
their own protected procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER = "ledger.sqlite3"
TABLES = ("signals", "reservations", "orders", "fills", "positions", "lifecycle")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(connection: sqlite3.Connection) -> dict[str, Any]:
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise ValueError("LEDGER_INTEGRITY_FAILURE")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("LEDGER_FOREIGN_KEY_FAILURE")
    provenance = connection.execute(
        "SELECT value FROM futures_meta WHERE key='provenance'"
    ).fetchone()
    return {
        "provenance": json.loads(provenance[0]) if provenance else None,
        "rows": {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES},
    }


def backup(database: Path, config: Path, output: Path) -> dict[str, Any]:
    output.mkdir(mode=0o700)  # a new destination only; never overwrite an earlier bundle
    before = digest(config)
    shutil.copyfile(config, output / config.name)
    # The online backup API copies one consistent snapshot, including committed WAL frames.
    with (
        closing(sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(output / LEDGER)) as copy,
    ):
        source.backup(copy)
        summary = inspect(copy)
    if digest(config) != before:
        raise RuntimeError("CONFIGURATION_CHANGED_DURING_BACKUP_DISCARD_BUNDLE")
    manifest = {
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_database": str(database.resolve()),
        "configuration": config.name,
        **summary,
        "files": {name: digest(output / name) for name in (config.name, LEDGER)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()) or digest(path) != checksum:
            raise ValueError("BUNDLE_CHECKSUM_MISMATCH")
    with closing(sqlite3.connect(f"file:{(bundle / LEDGER).resolve()}?mode=ro", uri=True)) as db:
        if inspect(db) != {k: manifest[k] for k in ("provenance", "rows")}:
            raise ValueError("BUNDLE_LEDGER_SUMMARY_MISMATCH")
    return dict(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an existing bundle and exit")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif args.database and args.config and args.output:
        backup(args.database, args.config, args.output)
        result = verify(args.output)
    else:
        parser.error("use --database, --config and --output, or --verify BUNDLE")
    print(json.dumps({k: result[k] for k in ("provenance", "rows")}, indent=2))
