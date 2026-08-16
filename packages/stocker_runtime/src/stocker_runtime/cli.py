"""Machine-readable database maintenance CLI for Stocker V2."""

from __future__ import annotations

import json
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from zoneinfo import ZoneInfo

import typer

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ingestion import (
    AdmissionResult,
    CallbackFence,
    IBKRMarketData,
    IBKRSubscription,
    InstrumentSpec,
    MarketDataCallback,
    MarketDataStatus,
    Recorder,
    SubscriptionSpec,
    load_recorder_config,
    market_data_input_hash,
    validate_market_data_inputs,
)
from stocker_runtime.ingestion.ibkr_api import (
    OfficialIBKRApiProvenanceError,
    evaluate_official_ibkr_api_update,
    fetch_latest_official_ibkr_api_release,
    inspect_official_ibkr_api_archive,
    load_official_ibkr_api_provenance,
    require_official_ibkr_api,
    write_immutable_official_ibkr_api_provenance,
    write_official_ibkr_api_update_status,
)
from stocker_runtime.ingestion.lifecycle import recover_fatal_generation
from stocker_runtime.storage import (
    BackupError,
    BackupPolicy,
    LegacyImportError,
    RetentionManager,
    SchemaError,
    create_backup,
    import_legacy_database,
    initialize_database,
    migrate_database,
    record_backup_failure,
    restore_backup,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Initialize, verify, migrate, and retain an isolated Stocker V2 database.",
)
web_app = typer.Typer(help="Run the bounded read-only Stocker V2 web process.")
backup_app = typer.Typer(help="Create and restore checked compressed Stocker V2 backups.")
recorder_app = typer.Typer(help="Run the sole market-data-only Stocker V2 recorder.")
legacy_app = typer.Typer(help="Perform the one-way import from a stopped Stocker V1 database.")
ibkr_api_app = typer.Typer(help="Verify official IBKR API provenance and check for updates.")
app.add_typer(web_app, name="web")
app.add_typer(backup_app, name="backup")
app.add_typer(recorder_app, name="recorder")
app.add_typer(legacy_app, name="legacy")
app.add_typer(ibkr_api_app, name="ibkr-api")

MAX_REPLAY_FILE_BYTES = 8 * 1024 * 1024
MAX_REPLAY_INSTRUMENTS = 10_000
MAX_REPLAY_SUBSCRIPTIONS = 10_000
MAX_REPLAY_CALLBACKS = 50_000
MAX_REPLAY_CALLBACK_BYTES = 65_536
RECORDER_IDLE_WAIT_SECONDS = 1.0
RECORDER_HEALTH_INTERVAL_US = 1_000_000
RECORDER_MAINTENANCE_INTERVAL_US = 10_000_000
_NEW_YORK = ZoneInfo("America/New_York")


@lru_cache(maxsize=32)
def _xnys_session_window_us(session_date: date) -> tuple[int, int] | None:
    import pandas_market_calendars as market_calendars

    schedule = market_calendars.get_calendar("NYSE").schedule(
        start_date=session_date.isoformat(),
        end_date=session_date.isoformat(),
    )
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    market_open = cast(Any, row["market_open"])
    market_close = cast(Any, row["market_close"])
    return (
        int(market_open.timestamp() * 1_000_000),
        int(market_close.timestamp() * 1_000_000),
    )


def _market_data_expected_since_us(now_us: int) -> int | None:
    session_date = datetime.fromtimestamp(now_us / 1_000_000, UTC).astimezone(_NEW_YORK).date()
    window = _xnys_session_window_us(session_date)
    if window is None:
        return None
    opened_at_us, closed_at_us = window
    return opened_at_us if opened_at_us <= now_us < closed_at_us else None


def _recorder_health_tick(recorder: Recorder, *, now_us: int) -> None:
    expected_since_us = _market_data_expected_since_us(now_us)
    recorder.mark_stale(
        now_us=now_us,
        market_data_expected=expected_since_us is not None,
        expected_since_us=expected_since_us,
    )
    recorder.recover_connection(now_us=now_us)
    recorder.recover_subscriptions(now_us=now_us)


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


@ibkr_api_app.command("verify")
def ibkr_api_verify_command(
    provenance: Annotated[Path, typer.Option("--provenance", exists=True, dir_okay=False)],
) -> None:
    """Verify the installed client tree against immutable official provenance."""

    try:
        require_official_ibkr_api(provenance)
        record = load_official_ibkr_api_provenance(provenance)
    except (OfficialIBKRApiProvenanceError, RuntimeError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=78) from error
    _emit({"provenance": record.model_dump(mode="json"), "status": "ok"})


@ibkr_api_app.command("register")
def ibkr_api_register_command(
    archive: Annotated[Path, typer.Option("--archive", exists=True, dir_okay=False)],
    installed_package_root: Annotated[
        Path, typer.Option("--installed-package-root", exists=True, file_okay=False)
    ],
    provenance: Annotated[Path, typer.Option("--provenance", dir_okay=False)],
    operator: Annotated[str, typer.Option("--operator")],
) -> None:
    """Register a matching installed tree; never install broker code."""

    try:
        checked_at = datetime.now(UTC)
        record = inspect_official_ibkr_api_archive(
            archive,
            installed_package_root=installed_package_root,
            release=fetch_latest_official_ibkr_api_release(),
            registered_by=operator,
            checked_at=checked_at,
        )
        write_immutable_official_ibkr_api_provenance(provenance, record)
    except OfficialIBKRApiProvenanceError as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=78) from error
    _emit({"provenance": record.model_dump(mode="json"), "status": "ok"})


@ibkr_api_app.command("check-update")
def ibkr_api_check_update_command(
    provenance: Annotated[Path, typer.Option("--provenance", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Record whether a newer official archive exists; never install it."""

    try:
        installed = load_official_ibkr_api_provenance(provenance)
        status = evaluate_official_ibkr_api_update(
            installed,
            fetch_latest_official_ibkr_api_release(),
            checked_at=datetime.now(UTC),
        )
        write_official_ibkr_api_update_status(output, status)
    except OfficialIBKRApiProvenanceError as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=78) from error
    _emit({"status": "ok", "update": status.model_dump(mode="json")})


@legacy_app.command("import")
def legacy_import_command(
    source: Annotated[Path, typer.Option("--source", dir_okay=False)],
    target: Annotated[Path, typer.Option("--target", dir_okay=False)],
    started_at_us: Annotated[int | None, typer.Option("--started-at-us", min=0)] = None,
    accept_quiescent_unclean_generations: Annotated[
        bool,
        typer.Option(
            "--accept-quiescent-unclean-generations",
            help=(
                "Attended cutover assertion for an immutable source with unclosed legacy "
                "generations; leases and SQLite sidecars still fail closed."
            ),
        ),
    ] = False,
) -> None:
    """Import an immutable, quiescent V1 database into one new V2 target."""

    try:
        result = import_legacy_database(
            source,
            target,
            started_at_us=started_at_us,
            accept_quiescent_unclean_generations=accept_quiescent_unclean_generations,
        )
    except LegacyImportError as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "imported_row_count": result.imported_row_count,
            "migration_id": result.migration_id,
            "omitted_row_count": result.omitted_row_count,
            "reconciliation_path": str(result.reconciliation_path),
            "source_database_hash": result.source_database_hash,
            "source_row_count": result.source_row_count,
            "source_schema_digest": result.source_schema_digest,
            "status": "ok",
            "target_digest": result.target_digest,
            "target_path": str(result.target_path),
            "verification_status": result.verification_status,
        }
    )


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
def validate_recorder_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    inputs: Annotated[Path, typer.Option("--inputs", exists=True, dir_okay=False)],
) -> None:
    """Validate recorder and desired market-data input without connecting to IBKR."""

    try:
        loaded = load_recorder_config(config)
        instruments, subscriptions = _load_recorder_inputs(inputs)
        validate_market_data_inputs(instruments, subscriptions)
        if len(subscriptions) > loaded.market_data_line_limit:
            raise ValueError(
                f"invalid market-data input {inputs}: subscriptions exceed recorder "
                "market_data_line_limit"
            )
    except (OSError, ValueError, RuntimeError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=78) from error
    _emit(
        {
            "host": loaded.host,
            "mode": loaded.mode,
            "read_only": loaded.read_only,
            "required_subscriptions": sum(not item.optional for item in subscriptions),
            "status": "ok",
            "subscriptions": len(subscriptions),
        }
    )


@recorder_app.command("recover-fatal-generation")
def recover_fatal_generation_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    inputs: Annotated[Path, typer.Option("--inputs", exists=True, dir_okay=False)],
    generation: Annotated[int, typer.Option("--generation", min=1)],
    fatal_code: Annotated[str, typer.Option("--fatal-code", min=1, max=256)],
    operator: Annotated[str, typer.Option("--operator", min=1, max=256)],
    reason: Annotated[str, typer.Option("--reason", min=1, max=1_024)],
    authorized_at_us: Annotated[int | None, typer.Option("--authorized-at-us", min=0)] = None,
) -> None:
    """Authorize one exact eligible fatal generation for audited same-run restart."""

    try:
        loaded = load_recorder_config(config)
        instruments, subscriptions = _load_recorder_inputs(inputs)
        timestamp = time.time_ns() // 1_000 if authorized_at_us is None else authorized_at_us
        recover_fatal_generation(
            database=loaded.database,
            run_id=loaded.run_id,
            generation=generation,
            mode=loaded.mode,
            config_hash=loaded.config_hash,
            input_hash=market_data_input_hash(instruments, subscriptions),
            fatal_code=fatal_code,
            operator=operator,
            reason=reason,
            authorized_at_us=timestamp,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error, SchemaError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "authorized_at_us": timestamp,
            "fatal_code": fatal_code,
            "generation": generation,
            "run_id": loaded.run_id,
            "status": "ok",
        }
    )


@backup_app.command("create")
def backup_create_command(
    database: Annotated[Path, typer.Option("--database", dir_okay=False)],
    destination: Annotated[Path, typer.Option("--destination", file_okay=False)],
    tier: Annotated[Literal["daily", "weekly"], typer.Option("--tier")],
    created_at_us: Annotated[int | None, typer.Option("--created-at-us", min=0)] = None,
    working_directory: Annotated[
        Path | None, typer.Option("--working-directory", file_okay=False)
    ] = None,
) -> None:
    """Create one checked online backup under the frozen retention policy."""

    try:
        artifact = create_backup(
            database,
            destination,
            tier=tier,
            created_at_us=created_at_us,
            policy=BackupPolicy(),
            working_directory=working_directory,
        )
    except (BackupError, OSError, ValueError, sqlite3.Error) as error:
        with suppress(Exception):
            record_backup_failure(
                destination,
                code=type(error).__name__[:96],
                checked_at_us=created_at_us,
            )
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "archive_filename": artifact.archive_path.name,
            "compressed_bytes": artifact.manifest.compressed_bytes,
            "manifest_filename": artifact.manifest_path.name,
            "status": "ok",
            "tier": artifact.manifest.tier,
        }
    )


@backup_app.command("restore")
def backup_restore_command(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    destination: Annotated[Path, typer.Option("--destination", dir_okay=False)],
) -> None:
    """Verify both hashes and restore into a path that does not exist."""

    try:
        result = restore_backup(manifest, destination)
    except (BackupError, OSError, ValueError, sqlite3.Error) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    _emit(
        {
            "destination": str(result.destination),
            "status": "ok",
            "uncompressed_bytes": result.uncompressed_bytes,
            "uncompressed_sha256": result.uncompressed_sha256,
        }
    )


def _bounded_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"recorder input {field} must be bounded non-empty text")
    return value


def _load_recorder_inputs(
    path: Path,
) -> tuple[tuple[InstrumentSpec, ...], tuple[SubscriptionSpec, ...]]:
    try:
        return _load_recorder_inputs_unscoped(path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise ValueError(f"invalid market-data input {path}: {error}") from error


def _load_recorder_inputs_unscoped(
    path: Path,
) -> tuple[tuple[InstrumentSpec, ...], tuple[SubscriptionSpec, ...]]:
    if path.stat().st_size > MAX_REPLAY_FILE_BYTES:
        raise ValueError("file exceeds the 8 MiB input limit")
    payload = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"instruments", "subscriptions"}:
        raise ValueError("requires exactly instruments and subscriptions")
    raw_instruments = payload["instruments"]
    raw_subscriptions = payload["subscriptions"]
    if not isinstance(raw_instruments, list) or len(raw_instruments) > MAX_REPLAY_INSTRUMENTS:
        raise ValueError("recorder instruments must be a bounded list")
    if not isinstance(raw_subscriptions, list) or len(raw_subscriptions) > MAX_REPLAY_SUBSCRIPTIONS:
        raise ValueError("recorder subscriptions must be a bounded list")
    instruments: list[InstrumentSpec] = []
    for item in raw_instruments:
        base_fields = {
            "instrument_id",
            "ibkr_con_id",
            "kind",
            "symbol",
            "exchange",
            "currency",
        }
        option_fields = {
            "option_expiry",
            "option_strike",
            "option_right",
            "option_multiplier",
        }
        if not isinstance(item, dict) or frozenset(item) not in {
            frozenset(base_fields),
            frozenset(base_fields | option_fields),
        }:
            raise ValueError("recorder instrument has invalid fields")
        con_id = item["ibkr_con_id"]
        if isinstance(con_id, bool) or not isinstance(con_id, int) or con_id <= 0:
            raise ValueError("recorder instrument ibkr_con_id must be a positive integer")
        instruments.append(
            InstrumentSpec(
                instrument_id=_bounded_nonempty_text(item["instrument_id"], field="instrument_id"),
                ibkr_con_id=con_id,
                kind=_bounded_nonempty_text(item["kind"], field="kind"),
                symbol=_bounded_nonempty_text(item["symbol"], field="symbol"),
                exchange=_bounded_nonempty_text(item["exchange"], field="exchange"),
                currency=_bounded_nonempty_text(item["currency"], field="currency"),
                option_expiry=(
                    None
                    if "option_expiry" not in item
                    else _bounded_nonempty_text(item["option_expiry"], field="option_expiry")
                ),
                option_strike=(
                    None
                    if "option_strike" not in item
                    else _bounded_nonempty_text(item["option_strike"], field="option_strike")
                ),
                option_right=(
                    None
                    if "option_right" not in item
                    else _bounded_nonempty_text(item["option_right"], field="option_right")
                ),
                option_multiplier=(
                    None
                    if "option_multiplier" not in item
                    else _bounded_nonempty_text(
                        item["option_multiplier"], field="option_multiplier"
                    )
                ),
            )
        )
    subscriptions: list[SubscriptionSpec] = []
    for item in raw_subscriptions:
        fields = {
            "name",
            "instrument_id",
            "feed_kind",
            "request_id",
            "continuity_required",
            "optional",
            "stale_after_us",
        }
        if not isinstance(item, dict) or frozenset(item) not in {
            frozenset(fields),
            frozenset(fields | {"snapshot"}),
        }:
            raise ValueError("recorder subscription has invalid fields")
        request_id = item["request_id"]
        stale_after_us = item["stale_after_us"]
        continuity_required = item["continuity_required"]
        optional = item["optional"]
        snapshot = item.get("snapshot", False)
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            raise ValueError("recorder subscription request_id must be a nonnegative integer")
        if (
            isinstance(stale_after_us, bool)
            or not isinstance(stale_after_us, int)
            or stale_after_us <= 0
        ):
            raise ValueError("recorder subscription stale_after_us must be a positive integer")
        if (
            not isinstance(continuity_required, bool)
            or not isinstance(optional, bool)
            or not isinstance(snapshot, bool)
        ):
            raise ValueError("recorder subscription flags must be booleans")
        subscriptions.append(
            SubscriptionSpec(
                name=_bounded_nonempty_text(item["name"], field="name"),
                instrument_id=_bounded_nonempty_text(item["instrument_id"], field="instrument_id"),
                feed_kind=_bounded_nonempty_text(item["feed_kind"], field="feed_kind"),
                request_id=request_id,
                continuity_required=continuity_required,
                optional=optional,
                stale_after_us=stale_after_us,
                snapshot=snapshot,
            )
        )
    result = (tuple(instruments), tuple(subscriptions))
    validate_market_data_inputs(*result)
    return result


@recorder_app.command("run")
def recorder_run_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    inputs: Annotated[Path, typer.Option("--inputs", exists=True, dir_okay=False)],
) -> None:
    """Run the market-data-only recorder until SIGTERM or SIGINT."""

    recorder: Recorder | None = None
    shutdown = threading.Event()
    termination = "requested"

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal termination
        termination = "sigterm" if signum == signal.SIGTERM else "sigint"
        shutdown.set()
        if recorder is not None:
            recorder.wake_pending_callback_drain()

    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    try:
        loaded = load_recorder_config(config)
        instruments, subscriptions = _load_recorder_inputs(inputs)
        adapter = IBKRMarketData.official(
            host=loaded.host,
            port=loaded.port,
            client_id=loaded.client_id,
            read_only=loaded.read_only,
            external_read_only_verified=loaded.external_read_only_verified,
        )
        recorder = Recorder(loaded, adapter)
        recorder.start(
            now_us=time.time_ns() // 1_000,
            instruments=instruments,
            subscriptions=subscriptions,
        )
        started_loop_at_us = time.time_ns() // 1_000
        next_health_at_us = started_loop_at_us + RECORDER_HEALTH_INTERVAL_US
        next_maintenance_at_us = started_loop_at_us + RECORDER_MAINTENANCE_INTERVAL_US
        while not shutdown.is_set():
            before_wait_us = time.time_ns() // 1_000
            wait_until_us = min(next_health_at_us, next_maintenance_at_us)
            recorder.wait_for_pending_callbacks(
                timeout=min(
                    RECORDER_IDLE_WAIT_SECONDS,
                    max(0, wait_until_us - before_wait_us) / 1_000_000,
                )
            )
            if shutdown.is_set():
                break
            recorder.prepare_pending_callback_drain()
            now_us = time.time_ns() // 1_000
            health_due = now_us >= next_health_at_us
            recorder.drain(now_us=now_us, run_downstream_when_idle=health_due)
            if health_due:
                _recorder_health_tick(recorder, now_us=now_us)
                next_health_at_us = now_us + RECORDER_HEALTH_INTERVAL_US
            if now_us >= next_maintenance_at_us:
                recorder.maintain(now_us=now_us)
                next_maintenance_at_us = now_us + RECORDER_MAINTENANCE_INTERVAL_US
        recorder.stop(now_us=time.time_ns() // 1_000)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, TypeError) as error:
        if recorder is not None and recorder.state is not None:
            recorder.abandon_unclean()
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=1) from error
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
    _emit({"status": "ok", "termination": termination})


@web_app.command("run")
def web_run_command(config: Annotated[Path, typer.Option("--config", exists=True)]) -> None:
    """Serve the three-view browser application on its configured loopback bind."""

    try:
        import uvicorn

        from stocker_runtime.web import WebConfig, create_web_app

        loaded = WebConfig.model_validate_json(config.read_text(encoding="utf-8"))
        application = create_web_app(loaded)
        forwarded = ",".join(loaded.trusted_proxy_ips) if loaded.trust_proxy_headers else ""
        uvicorn.run(
            application,
            host=loaded.host,
            port=loaded.port,
            proxy_headers=loaded.trust_proxy_headers,
            forwarded_allow_ips=forwarded,
            log_level="info",
        )
    except (OSError, RuntimeError, ValueError) as error:
        _emit({"error": type(error).__name__, "message": str(error), "status": "error"})
        raise typer.Exit(code=78) from error


class _ReplayMarketData:
    """Offline adapter used only by the explicit replay CLI."""

    capabilities = frozenset({"market_data"})

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

    def configure_subscriptions(self, _subscriptions: tuple[IBKRSubscription, ...]) -> None:
        return None

    def subscribe(self, _fence: CallbackFence) -> None:
        return None

    def retry_subscription(self, _fence: CallbackFence) -> None:
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
            drain_limit = min(10_000, max(1, backlog))
            processed = recorder.drain(
                now_us=pass_now_us,
                limit=drain_limit,
                defer_downstream_when_full=False,
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
