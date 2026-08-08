"""Fail-closed SQLite connection and migration management for Stocker V2."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

MIGRATION_PATTERN = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z0-9_]+)\.sql$")


class SchemaError(RuntimeError):
    """The database is not a compatible, untampered Stocker V2 database."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sha256: str
    sql: str


@dataclass(frozen=True)
class MigrationResult:
    applied_versions: tuple[int, ...]
    current_version: int


EXPECTED_TABLES = frozenset(
    {
        "schema_migrations",
        "runs",
        "recorder_generations",
        "runtime_state",
        "incidents",
        "gaps",
        "instruments",
        "subscriptions",
        "callback_inbox",
        "callback_receipts",
        "callback_compaction_watermarks",
        "market_events",
        "market_event_derivations",
        "market_latest",
        "idea_plugins",
        "idea_instances",
        "idea_checkpoints",
        "idea_outputs",
        "idea_output_inputs",
        "idea_output_legs",
        "shadow_positions",
        "shadow_progress",
        "shadow_quote_state",
        "shadow_schedule",
        "shadow_legs",
        "shadow_marks",
        "shadow_outcomes",
        "migration_manifests",
    }
)


def migration_plan(root: Path | None = None) -> tuple[Migration, ...]:
    """Load a contiguous, uniquely numbered migration plan."""

    migration_root = root or Path(__file__).with_name("migrations")
    migrations: list[Migration] = []
    for path in sorted(migration_root.glob("*.sql")):
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if match is None:
            raise SchemaError(f"invalid migration filename: {path.name}")
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=path.name,
                sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                sql=sql,
            )
        )
    if not migrations:
        raise SchemaError("no V2 migrations found")
    versions = tuple(item.version for item in migrations)
    if versions != tuple(range(1, len(migrations) + 1)):
        raise SchemaError(f"migration versions must be contiguous from 1: {versions}")
    if len({item.name for item in migrations}) != len(migrations):
        raise SchemaError("duplicate migration names")
    return tuple(migrations)


def _apply_pragmas(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")
    connection.execute("PRAGMA journal_size_limit = 67108864")


def _is_canonical_json(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    return int(canonical == value)


def _register_functions(connection: sqlite3.Connection) -> None:
    connection.create_function(
        "stocker_canonical_json",
        1,
        _is_canonical_json,
        deterministic=True,
    )


def _raw_connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=5.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    _register_functions(connection)
    _apply_pragmas(connection)
    return connection


def _schema_tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )


def _schema_digest(connection: sqlite3.Connection) -> str:
    definitions = [
        tuple(str(value) for value in row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    ]
    encoded = json.dumps(definitions, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@cache
def _expected_schema_digest(migration_fingerprints: tuple[tuple[str, str], ...]) -> str:
    plan = migration_plan()
    expected_prefix = tuple(
        (item.name, item.sha256) for item in plan[: len(migration_fingerprints)]
    )
    if migration_fingerprints != expected_prefix:
        raise SchemaError("runtime migration plan changed while verifying schema")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        _register_functions(connection)
        for migration in plan[: len(migration_fingerprints)]:
            connection.executescript(migration.sql)
        return _schema_digest(connection)
    finally:
        connection.close()


def _verify_schema_structure(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
    applied: set[int],
) -> None:
    if (
        migrations == migration_plan()
        and applied == {item.version for item in migrations}
        and _schema_tables(connection) != EXPECTED_TABLES
    ):
        raise SchemaError("database table set is incompatible with immediate V2")
    default_plan = migration_plan()
    if migrations == default_plan:
        prefix = tuple(item for item in migrations if item.version in applied)
        fingerprints = tuple((item.name, item.sha256) for item in prefix)
        if _schema_digest(connection) != _expected_schema_digest(fingerprints):
            raise SchemaError("database schema structure does not match the migration checksums")


def _read_only_connect(database_path: Path) -> sqlite3.Connection:
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    _register_functions(connection)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _probe_v2(
    database_path: Path,
    migrations: tuple[Migration, ...],
    *,
    verify_integrity: bool,
) -> set[int]:
    with _read_only_connect(database_path) as connection:
        applied = _verify_applied_migrations(connection, migrations)
        _verify_schema_structure(connection, migrations, applied)
        if verify_integrity and tuple(connection.execute("PRAGMA foreign_key_check")):
            raise SchemaError("database contains foreign-key violations")
        return applied


def _verify_applied_migrations(
    connection: sqlite3.Connection, migrations: tuple[Migration, ...]
) -> set[int]:
    tables = _schema_tables(connection)
    if "schema_migrations" not in tables:
        raise SchemaError("not a Stocker V2 database: schema_migrations is absent")
    rows = tuple(
        connection.execute("SELECT version, name, sha256 FROM schema_migrations ORDER BY version")
    )
    known = {migration.version: migration for migration in migrations}
    applied: set[int] = set()
    for row in rows:
        version = int(row["version"])
        migration = known.get(version)
        if migration is None:
            raise SchemaError(f"database schema version {version} is newer than this runtime")
        if str(row["name"]) != migration.name or str(row["sha256"]) != migration.sha256:
            raise SchemaError(f"migration checksum/name mismatch at version {version}")
        applied.add(version)
    if applied and applied != set(range(1, max(applied) + 1)):
        raise SchemaError("database migration history is not contiguous")
    return applied


def _apply_migrations(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...],
    applied: set[int],
    applied_at_us: int,
) -> tuple[int, ...]:
    pending = tuple(migration for migration in migrations if migration.version not in applied)
    if not pending:
        _verify_schema_structure(connection, migrations, applied)
        if tuple(connection.execute("PRAGMA foreign_key_check")):
            raise SchemaError("database contains foreign-key violations")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SchemaError("database quick_check failed")
        return ()
    statements = ["BEGIN IMMEDIATE;"]
    for migration in pending:
        name = migration.name.replace("'", "''")
        checksum = migration.sha256.replace("'", "''")
        statements.extend(
            (
                migration.sql,
                "INSERT INTO schema_migrations(version, name, sha256, applied_at_us) "
                f"VALUES ({migration.version}, '{name}', '{checksum}', {applied_at_us});",
            )
        )
    try:
        connection.executescript("\n".join(statements))
        final_applied = applied | {migration.version for migration in pending}
        _verify_schema_structure(connection, migrations, final_applied)
        violations = tuple(connection.execute("PRAGMA foreign_key_check"))
        if violations:
            raise SchemaError("database contains foreign-key violations")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SchemaError("database quick_check failed")
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    return tuple(migration.version for migration in pending)


def initialize_database(
    database_path: str | Path,
    *,
    applied_at_us: int | None = None,
    migration_root: Path | None = None,
) -> MigrationResult:
    """Create a new Stocker V2 database; never adopt or mutate an existing file."""

    path = Path(database_path)
    if path.exists():
        raise SchemaError("V2 initialization requires a new empty path")
    path.parent.mkdir(parents=True, exist_ok=True)
    migrations = migration_plan(migration_root)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        _register_functions(connection)
        connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
        _apply_pragmas(connection)
        applied = _apply_migrations(
            connection,
            migrations,
            set(),
            applied_at_us if applied_at_us is not None else time.time_ns() // 1_000,
        )
        return MigrationResult(applied_versions=applied, current_version=migrations[-1].version)
    except Exception:
        if connection is not None:
            connection.close()
        path.unlink(missing_ok=True)
        Path(f"{path}-wal").unlink(missing_ok=True)
        Path(f"{path}-shm").unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def migrate_database(
    database_path: str | Path,
    *,
    applied_at_us: int | None = None,
    migration_root: Path | None = None,
) -> MigrationResult:
    """Verify and migrate an existing Stocker V2 database atomically."""

    path = Path(database_path)
    if not path.is_file():
        raise SchemaError("V2 migration requires an existing database")
    migrations = migration_plan(migration_root)
    _probe_v2(path, migrations, verify_integrity=True)
    with _raw_connect(path) as connection:
        applied = _verify_applied_migrations(connection, migrations)
        newly_applied = _apply_migrations(
            connection,
            migrations,
            applied,
            applied_at_us if applied_at_us is not None else time.time_ns() // 1_000,
        )
    return MigrationResult(applied_versions=newly_applied, current_version=migrations[-1].version)


def connect_v2(database_path: str | Path, *, verify_schema: bool = True) -> sqlite3.Connection:
    """Open a configured writer connection to an existing compatible V2 database."""

    path = Path(database_path)
    if not path.is_file():
        raise SchemaError("V2 database does not exist")
    migrations = migration_plan()
    applied = _probe_v2(path, migrations, verify_integrity=False) if verify_schema else set()
    if verify_schema and applied != {item.version for item in migrations}:
        raise SchemaError("database schema is older than this runtime; run migrate")
    connection = _raw_connect(path)
    try:
        if verify_schema:
            applied = _verify_applied_migrations(connection, migrations)
            if applied != {item.version for item in migrations}:
                raise SchemaError("database schema is older than this runtime; run migrate")
            _verify_schema_structure(connection, migrations, applied)
        return connection
    except Exception:
        connection.close()
        raise


def verify_database(database_path: str | Path) -> None:
    """Run explicit structural, foreign-key, and quick integrity verification."""

    path = Path(database_path)
    if not path.is_file():
        raise SchemaError("V2 database does not exist")
    migrations = migration_plan()
    applied = _probe_v2(path, migrations, verify_integrity=True)
    if applied != {item.version for item in migrations}:
        raise SchemaError("database schema is older than this runtime; run migrate")
    with _read_only_connect(path) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SchemaError("database quick_check failed")
