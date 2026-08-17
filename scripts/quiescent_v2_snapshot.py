#!/usr/bin/env python3
"""Create a byte-identical, restore-checked V2 snapshot while the recorder is stopped."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from stocker_runtime.ingestion.lifecycle import LocalWriterLock

MAX_RECOVERY_SNAPSHOTS = 2
MAX_RECOVERY_ARCHIVE_BYTES = 9 * 1024 * 1024 * 1024
RECOVERY_WORKING_HEADROOM_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_for_archive(archive: Path) -> Path:
    return Path(f"{str(archive)[:-3]}.manifest.json")


def _recovery_archives(destination: Path) -> list[Path]:
    archives = sorted(
        destination.glob("stocker-v2-*.sqlite3.gz"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for archive in archives:
        if archive.stat().st_size > MAX_RECOVERY_ARCHIVE_BYTES:
            raise RuntimeError(f"recovery archive exceeds its size limit: {archive.name}")
        if not _manifest_for_archive(archive).is_file():
            raise RuntimeError(f"recovery archive has no matching manifest: {archive.name}")
    return archives


def _rotate_recovery_archives(destination: Path) -> None:
    archives = _recovery_archives(destination)
    while len(archives) > MAX_RECOVERY_SNAPSHOTS:
        expired = archives.pop(0)
        manifest = _manifest_for_archive(expired)
        expired.unlink()
        manifest.unlink()
    _fsync(destination)


def _verify(
    path: Path,
    *,
    runtime: Path,
    expected_schema: int,
    run_id: str,
    generation: int,
    expected_termination_code: str,
    expected_max_source_sequence: int,
    expected_nonterminal: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [str(runtime), "migrate", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    migration = json.loads(completed.stdout)
    expected_migration = {
        "applied_versions": [],
        "current_version": expected_schema,
        "status": "ok",
    }
    if migration != expected_migration:
        raise RuntimeError(f"unexpected migration verification: {migration}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
        maximum, nonterminal = connection.execute(
            "SELECT MAX(source_sequence), "
            "SUM(CASE WHEN lifecycle IN ('pending','leased') THEN 1 ELSE 0 END) "
            "FROM callback_inbox"
        ).fetchone()
        generation_row = connection.execute(
            "SELECT ended_at_us, clean_stop, termination_code, input_hash "
            "FROM recorder_generations "
            "WHERE run_id=? AND generation=?",
            (run_id, generation),
        ).fetchone()
        run = connection.execute(
            "SELECT config_hash FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if quick_check != "ok" or foreign_keys:
        raise RuntimeError("snapshot integrity verification failed")
    if maximum != expected_max_source_sequence or nonterminal != expected_nonterminal:
        raise RuntimeError(
            "snapshot callback evidence differs from the recorded pre-snapshot values"
        )
    if generation_row is None or run is None:
        raise RuntimeError("snapshot run or generation evidence is absent")
    if (
        generation_row[0] is None
        or generation_row[1] != 1
        or generation_row[2] != expected_termination_code
    ):
        raise RuntimeError(f"snapshot generation is not the expected clean stop: {generation_row}")
    return {
        "schema": expected_schema,
        "quick_check": quick_check,
        "foreign_key_violations": len(foreign_keys),
        "max_source_sequence": maximum,
        "nonterminal_callbacks": nonterminal,
        "generation_ended_at_us": generation_row[0],
        "generation_clean_stop": generation_row[1],
        "generation_termination_code": generation_row[2],
        "config_hash": run[0],
        "input_hash": generation_row[3],
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--destination-directory", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--expected-schema", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--expected-termination-code", default="CLEAN_STOP")
    parser.add_argument("--expected-max-source-sequence", type=int, required=True)
    parser.add_argument("--expected-nonterminal", type=int, required=True)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--recorder-unit", default="stocker-v2-recorder.service")
    parser.add_argument("--web-unit", default="stocker-v2-web.service")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    database = arguments.database.resolve(strict=True)
    destination = arguments.destination_directory.resolve(strict=True)
    runtime = arguments.runtime_executable.resolve(strict=True)
    if not database.is_file() or not destination.is_dir() or not runtime.is_file():
        raise RuntimeError("database, destination, and matching runtime must exist")
    if arguments.expected_schema < 1 or arguments.generation < 0:
        raise RuntimeError("expected schema and generation are invalid")
    _recovery_archives(destination)
    source_bytes = database.stat().st_size
    required_free_bytes = 3 * source_bytes + RECOVERY_WORKING_HEADROOM_BYTES
    if shutil.disk_usage(destination).free < required_free_bytes:
        raise RuntimeError(
            "recovery snapshot lacks space for copy, compressed archive, restore proof, "
            "and working headroom"
        )
    for unit in (arguments.recorder_unit, arguments.web_unit):
        service = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            capture_output=True,
            text=True,
        )
        if service.returncode == 0:
            raise RuntimeError(f"service must be inactive: {unit}")
        if service.returncode != 3:
            raise RuntimeError(f"cannot verify inactive service state: {unit}")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"stocker-v2-pre-migration-gen{arguments.generation}-{stamp}.sqlite3"
    snapshot = destination / base
    archive = Path(f"{snapshot}.gz")
    restore = Path(f"{snapshot}.restore.sqlite3")
    manifest = Path(f"{snapshot}.manifest.json")
    generated = (
        snapshot,
        archive,
        restore,
        manifest,
        Path(f"{snapshot}-wal"),
        Path(f"{snapshot}-shm"),
        Path(f"{restore}-wal"),
        Path(f"{restore}-shm"),
    )
    if any(path.exists() for path in generated):
        raise RuntimeError("a generated snapshot target already exists")

    lock = LocalWriterLock.for_database(database)
    succeeded = False
    try:
        lock.acquire()
        lock.verify_held()
        lsof = subprocess.run(["lsof", str(database)], capture_output=True, text=True)
        if lsof.returncode == 0 and lsof.stdout.strip():
            raise RuntimeError(f"database has an open descriptor: {lsof.stdout.strip()}")
        if lsof.returncode != 1:
            raise RuntimeError("cannot prove that the database has no open descriptor")

        lock.verify_held()
        with sqlite3.connect(database, isolation_level=None) as connection:
            checkpoint = tuple(connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
        lock.verify_held()
        if checkpoint[0] != 0 or checkpoint[2] < checkpoint[1]:
            raise RuntimeError(f"source WAL checkpoint is incomplete: {checkpoint}")
        wal = Path(f"{database}-wal")
        if wal.exists() and wal.stat().st_size != 0:
            raise RuntimeError("source WAL is nonzero after the truncate checkpoint")

        before = database.stat()
        lock.verify_held()
        shutil.copyfile(database, snapshot)
        os.chmod(snapshot, 0o640)
        os.chown(snapshot, before.st_uid, before.st_gid)
        _fsync(snapshot)
        _fsync(destination)
        lock.verify_held()
        after = database.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("source database changed during the snapshot")
        source_hash = _sha256(database)
        snapshot_hash = _sha256(snapshot)
        if before.st_size != snapshot.stat().st_size or source_hash != snapshot_hash:
            raise RuntimeError("source and snapshot are not byte-identical")
        evidence = _verify(
            snapshot,
            runtime=runtime,
            expected_schema=arguments.expected_schema,
            run_id=arguments.run_id,
            generation=arguments.generation,
            expected_termination_code=arguments.expected_termination_code,
            expected_max_source_sequence=arguments.expected_max_source_sequence,
            expected_nonterminal=arguments.expected_nonterminal,
        )
        lock.verify_held()

        with snapshot.open("rb") as source, archive.open("xb") as raw_archive:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_archive, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=8 * 1024 * 1024)
            raw_archive.flush()
            os.fsync(raw_archive.fileno())
        os.chmod(archive, 0o640)
        os.chown(archive, before.st_uid, before.st_gid)
        if archive.stat().st_size > MAX_RECOVERY_ARCHIVE_BYTES:
            raise RuntimeError("recovery archive exceeds its size limit")
        archive_hash = _sha256(archive)
        lock.verify_held()

        with gzip.open(archive, "rb") as compressed, restore.open("xb") as restored:
            shutil.copyfileobj(compressed, restored, length=8 * 1024 * 1024)
            restored.flush()
            os.fsync(restored.fileno())
        if restore.stat().st_size != before.st_size or _sha256(restore) != source_hash:
            raise RuntimeError("decompressed restore is not byte-identical")
        restored_evidence = _verify(
            restore,
            runtime=runtime,
            expected_schema=arguments.expected_schema,
            run_id=arguments.run_id,
            generation=arguments.generation,
            expected_termination_code=arguments.expected_termination_code,
            expected_max_source_sequence=arguments.expected_max_source_sequence,
            expected_nonterminal=arguments.expected_nonterminal,
        )
        if restored_evidence != evidence:
            raise RuntimeError("decompressed restore evidence differs from the snapshot")
        lock.verify_held()

        payload = {
            "status": "ok",
            "format": "stocker_quiescent_v2_snapshot_v1",
            "created_at_us": time.time_ns() // 1_000,
            "operator": arguments.operator,
            "run_id": arguments.run_id,
            "generation": arguments.generation,
            "source_release": arguments.release_commit,
            "snapshot_filename": snapshot.name,
            "snapshot_bytes": snapshot.stat().st_size,
            "snapshot_sha256": snapshot_hash,
            "uncompressed_retained": False,
            "archive_filename": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_hash,
            "restore_verified": True,
            **evidence,
        }
        with manifest.open("x", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(manifest, 0o640)
        os.chown(manifest, before.st_uid, before.st_gid)
        _fsync(destination)
        lock.verify_held()
        snapshot.unlink()
        Path(f"{snapshot}-wal").unlink(missing_ok=True)
        Path(f"{snapshot}-shm").unlink(missing_ok=True)
        _rotate_recovery_archives(destination)
        lock.verify_held()
        print(json.dumps(payload, sort_keys=True))
        succeeded = True
    finally:
        for temporary in (restore, Path(f"{restore}-wal"), Path(f"{restore}-shm")):
            temporary.unlink(missing_ok=True)
        if not succeeded:
            for partial in generated:
                partial.unlink(missing_ok=True)
        _fsync(destination)
        lock.release()


if __name__ == "__main__":
    main()
