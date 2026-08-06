"""Machine-readable database maintenance CLI for Stocker V2."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import typer

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ingestion import (
    AdmissionResult,
    CallbackFence,
    InstrumentSpec,
    MarketDataCallback,
    MarketDataStatus,
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

MAX_REPLAY_FILE_BYTES = 8 * 1024 * 1024
MAX_REPLAY_INSTRUMENTS = 10_000
MAX_REPLAY_SUBSCRIPTIONS = 10_000
MAX_REPLAY_CALLBACKS = 50_000
MAX_REPLAY_CALLBACK_BYTES = 65_536


class ReplayBlockedError(RuntimeError):
    """A bounded replay pass could not make durable progress."""


def _validated_replay_callback(
    item: object, configured_request_ids: set[int]
) -> tuple[int, int, int | None, str, JsonValue]:
    callback_keys = {
        "request_id",
        "callback_kind",
        "received_at_us",
        "provider_at_us",
        "payload",
    }
    if not isinstance(item, dict) or set(item) != callback_keys:
        raise ValueError("replay callback has invalid fields")
    request_id = item["request_id"]
    received_at_us = item["received_at_us"]
    provider_at_us = item["provider_at_us"]
    callback_kind = item["callback_kind"]
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ValueError("replay callback request_id must be an integer")
    if request_id not in configured_request_ids:
        raise ValueError("replay callback request_id is not configured")
    if isinstance(received_at_us, bool) or not isinstance(received_at_us, int):
        raise ValueError("replay callback received_at_us must be an integer")
    if provider_at_us is not None and (
        isinstance(provider_at_us, bool) or not isinstance(provider_at_us, int)
    ):
        raise ValueError("replay callback provider_at_us must be an integer or null")
    if not isinstance(callback_kind, str) or not callback_kind:
        raise ValueError("replay callback callback_kind must be a non-empty string")
    payload = cast(JsonValue, item["payload"])
    if len(canonical_json_bytes(payload)) > MAX_REPLAY_CALLBACK_BYTES:
        raise ValueError("replay callback payload exceeds 64 KiB")
    return request_id, received_at_us, provider_at_us, callback_kind, payload


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

    def set_status_callback(self, _callback: Callable[[MarketDataStatus], None]) -> None:
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

    recorder: Recorder | None = None
    adapter: _ReplayMarketData | None = None
    try:
        loaded = load_recorder_config(config)
        if fixture.stat().st_size > MAX_REPLAY_FILE_BYTES:
            raise ValueError("replay fixture exceeds the 8 MiB input limit")
        raw = json.loads(fixture.read_bytes().decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("replay fixture must be an object")
        if set(raw) != {"instruments", "subscriptions", "callbacks"}:
            raise ValueError(
                "replay fixture requires exactly instruments, subscriptions, callbacks"
            )
        raw_instruments = raw["instruments"]
        raw_subscriptions = raw["subscriptions"]
        callbacks = raw.get("callbacks", ())
        if not isinstance(raw_instruments, list) or len(raw_instruments) > MAX_REPLAY_INSTRUMENTS:
            raise ValueError("replay instruments must be a bounded list")
        if (
            not isinstance(raw_subscriptions, list)
            or len(raw_subscriptions) > MAX_REPLAY_SUBSCRIPTIONS
        ):
            raise ValueError("replay subscriptions must be a bounded list")
        if not all(isinstance(item, dict) for item in raw_instruments):
            raise ValueError("every replay instrument must be an object")
        if not all(isinstance(item, dict) for item in raw_subscriptions):
            raise ValueError("every replay subscription must be an object")
        instruments = tuple(InstrumentSpec(**item) for item in raw_instruments)
        subscriptions = tuple(SubscriptionSpec(**item) for item in raw_subscriptions)
        if not isinstance(callbacks, list) or len(callbacks) > MAX_REPLAY_CALLBACKS:
            raise ValueError("replay callbacks must be a list of at most 50,000 items")
        configured_request_ids = {spec.request_id for spec in subscriptions}
        adapter = _ReplayMarketData()
        recorder = Recorder(loaded, adapter)
        state = recorder.start(
            now_us=now_us,
            instruments=instruments,
            subscriptions=subscriptions,
        )
        fences = {fence.request_id: fence for fence in state.fences}
        for item in callbacks:
            request_id, received_at_us, provider_at_us, callback_kind, payload = (
                _validated_replay_callback(item, configured_request_ids)
            )
            fence = fences.get(request_id)
            if fence is None:
                raise ValueError("replay callback request_id is not configured")
            recorder.receive(
                fence,
                MarketDataCallback(
                    callback_kind=callback_kind,
                    received_at_us=received_at_us,
                    provider_at_us=provider_at_us,
                    payload=payload,
                ),
            )
        projected = 0
        backlog = recorder.inbox.nonterminal_count()
        max_passes = (backlog + 9_999) // 10_000 + 2
        for pass_index in range(max_passes):
            before = recorder.inbox.nonterminal_count()
            if before == 0:
                break
            pass_now_us = now_us + 1 + pass_index
            recorder.inbox.reclaim_expired_leases(
                now_us=pass_now_us,
                authority=recorder._authority(),
            )
            processed = recorder.drain(
                now_us=pass_now_us,
                limit=min(10_000, max(1, backlog)),
            )
            projected += processed
            after = recorder.inbox.nonterminal_count()
            if after >= before:
                raise ReplayBlockedError(
                    f"replay made no progress: nonterminal_count={after}, pass={pass_index}"
                )
        if recorder.inbox.nonterminal_count() != 0:
            raise ReplayBlockedError("replay exceeded its deterministic processing bound")
        recorder.stop(now_us=now_us + 2)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, TypeError) as error:
        if recorder is not None and recorder.state is not None:
            try:
                recorder.stop(now_us=now_us + 2)
            except Exception:
                if adapter is not None:
                    with suppress(Exception):
                        adapter.disconnect()
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
