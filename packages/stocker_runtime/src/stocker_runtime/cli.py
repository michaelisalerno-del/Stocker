"""Machine-readable database maintenance CLI for Stocker V2."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import typer

from stocker_runtime.domain import JsonValue
from stocker_runtime.ingestion import (
    AdmissionResult,
    CallbackFence,
    InstrumentSpec,
    MarketDataCallback,
    Recorder,
    SubscriptionSpec,
    load_recorder_config,
)
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


@app.command("validate-recorder")
def validate_recorder_command(config: Annotated[Path, typer.Argument()]) -> None:
    """Validate recorder safety configuration without connecting to IBKR."""

    try:
        loaded = load_recorder_config(config)
    except (OSError, ValueError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "host": loaded.host,
            "mode": loaded.mode,
            "read_only": loaded.read_only,
            "status": "ok",
        }
    )


class _ReplayMarketData:
    """Offline adapter used only by the explicit replay CLI."""

    def __init__(self) -> None:
        self.callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult] | None = None

    def set_callback(
        self, callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult]
    ) -> None:
        self.callback = callback

    def set_disconnect_callback(self, _callback: Callable[[int], None]) -> None:
        return None

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def subscribe(self, _fence: CallbackFence) -> None:
        return None

    def cancel(self, _request_id: int) -> None:
        return None


@app.command("replay-recorder")
def replay_recorder_command(
    config: Annotated[Path, typer.Argument()],
    fixture: Annotated[Path, typer.Argument()],
    now_us: Annotated[int, typer.Option("--now-us", min=0)],
) -> None:
    """Run one fully offline recorder lifecycle from a bounded JSON fixture."""

    try:
        loaded = load_recorder_config(config)
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("replay fixture must be an object")
        instruments = tuple(InstrumentSpec(**item) for item in raw.get("instruments", ()))
        subscriptions = tuple(SubscriptionSpec(**item) for item in raw.get("subscriptions", ()))
        callbacks = raw.get("callbacks", ())
        if not isinstance(callbacks, list) or len(callbacks) > 50_000:
            raise ValueError("replay callbacks must be a list of at most 50,000 items")
        adapter = _ReplayMarketData()
        recorder = Recorder(loaded, adapter)
        state = recorder.start(
            now_us=now_us,
            instruments=instruments,
            subscriptions=subscriptions,
        )
        fences = {fence.request_id: fence for fence in state.fences}
        for item in callbacks:
            if not isinstance(item, dict):
                raise ValueError("replay callback must be an object")
            request_id = item.get("request_id")
            fence = fences.get(request_id)
            if fence is None:
                raise ValueError("replay callback request_id is not configured")
            received_at_us = item.get("received_at_us")
            if isinstance(received_at_us, bool) or not isinstance(received_at_us, int):
                raise ValueError("replay callback received_at_us must be an integer")
            provider_at_us = item.get("provider_at_us")
            if provider_at_us is not None and (
                isinstance(provider_at_us, bool) or not isinstance(provider_at_us, int)
            ):
                raise ValueError("replay callback provider_at_us must be an integer or null")
            payload = cast(JsonValue, item.get("payload"))
            recorder.receive(
                fence,
                MarketDataCallback(
                    callback_kind=str(item.get("callback_kind", "")),
                    received_at_us=received_at_us,
                    provider_at_us=provider_at_us,
                    payload=payload,
                ),
            )
        projected = 0
        while recorder.inbox.nonterminal_count() > 0:
            processed = recorder.drain(now_us=now_us + 1, limit=min(10_000, max(1, len(callbacks))))
            projected += processed
        recorder.stop(now_us=now_us + 2)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, TypeError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "admitted": len(callbacks),
            "mode": loaded.mode,
            "projected": projected,
            "status": "ok",
        }
    )


if __name__ == "__main__":
    app()
