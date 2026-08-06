"""Machine-readable database maintenance CLI for Stocker V2."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from stocker_runtime.storage import (
    RetentionManager,
    SchemaError,
    initialize_database,
    migrate_database,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Initialize, verify, migrate, and retain an isolated Stocker V2 database.",
)


def _emit(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _migration_payload(
    current_version: int, applied_versions: tuple[int, ...]
) -> dict[str, object]:
    return {
        "applied_versions": list(applied_versions),
        "current_version": current_version,
        "status": "ok",
    }


@app.command("init")
def init_command(database: Annotated[Path, typer.Argument()]) -> None:
    """Create a new V2 database at a path that does not already exist."""

    try:
        result = initialize_database(database)
    except (OSError, SchemaError, ValueError, sqlite3.Error) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(_migration_payload(result.current_version, result.applied_versions))


@app.command("migrate")
def migrate_command(database: Annotated[Path, typer.Argument()]) -> None:
    """Verify checksums and atomically migrate an existing V2 database."""

    try:
        result = migrate_database(database)
    except (OSError, SchemaError, sqlite3.Error) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(_migration_payload(result.current_version, result.applied_versions))


@app.command("retain")
def retain_command(
    database: Annotated[Path, typer.Argument()],
    now_us: Annotated[int | None, typer.Option("--now-us", min=0)] = None,
) -> None:
    """Run one bounded retention/checkpoint pass and report storage-cap state."""

    try:
        result = RetentionManager(database).run(
            now_us=now_us if now_us is not None else time.time_ns() // 1_000
        )
    except (OSError, SchemaError, ValueError, RuntimeError, sqlite3.Error) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    payload = asdict(result)
    payload["cap_state"] = result.cap_state.value
    payload["status"] = "ok"
    _emit(payload)


if __name__ == "__main__":
    app()
