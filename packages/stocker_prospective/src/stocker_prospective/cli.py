"""Operator CLI for immutable bundles, replay, migrations, recorder, and web."""

from __future__ import annotations

import json
import os
import signal
import socket
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn
from zoneinfo import ZoneInfo

import typer
import uvicorn
import yaml

from stocker_prospective.backup import backup_database
from stocker_prospective.bundle import (
    BundleBuildSpec,
    BundleError,
    activate_bundle,
    build_bundle,
    install_bundle,
    list_installed_bundles,
    load_active_bundle,
    verify_bundle,
)
from stocker_prospective.config import (
    ProspectiveConfig,
    RuntimeSafetyError,
    load_prospective_config,
    validate_persistent_paths,
    validate_runtime_safety,
)
from stocker_prospective.context import (
    ContextValidationError,
    DailyContextUnsigned,
    SignedDailyContext,
    create_signed_context,
    import_signed_context,
    load_imported_context,
)
from stocker_prospective.database import LeaseRecord, ProspectiveRepository
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.frozen_artifacts import (
    FrozenArtifactReconstructionError,
    reconstruct_frozen_artifacts,
)
from stocker_prospective.ibkr import (
    IBKRConnectionConfig,
    IBKRMarketDataAdapter,
    OfficialIBKRDependencyError,
    require_ibkr_socket_loopback_only,
    require_official_ibkr_api,
)
from stocker_prospective.ibkr_api import (
    OfficialIBKRApiProvenanceError,
    evaluate_official_ibkr_api_update,
    fetch_latest_official_ibkr_api_release,
    inspect_official_ibkr_api_archive,
    load_official_ibkr_api_provenance,
    write_immutable_official_ibkr_api_provenance,
    write_official_ibkr_api_update_status,
)
from stocker_prospective.market_data import MarketDataBudget, MarketDataType
from stocker_prospective.operational_state import (
    GapIncident,
    RecorderOperationalRepository,
    stable_gap_id,
)
from stocker_prospective.parallel import (
    ParallelSourceCaptureService,
    build_parallel_eodhd_service,
)
from stocker_prospective.parity import FeatureParityError, load_feature_parity_report
from stocker_prospective.recorder import RecorderDeploymentIdentity
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
from stocker_prospective.scientific_inputs import (
    EODHDGroupOPreparationService,
    acquire_eodhd_historical_activity_baseline,
)
from stocker_prospective.universe import UniverseError, load_registered_universe
from stocker_prospective.web import create_web_app

app = typer.Typer(
    name="stocker-prospective",
    help="Record-only prospective evidence runtime. No order commands exist.",
    no_args_is_help=True,
)
bundle_app = typer.Typer(help="Build, verify, install, and activate immutable bundles.")
context_app = typer.Typer(help="Create and import exact signed daily context packages.")
database_app = typer.Typer(help="Manage the prospective database schema.")
replay_app = typer.Typer(help="Run the deterministic synthetic vertical slice.")
recorder_app = typer.Typer(help="Run the market-data recorder process.")
scientific_inputs_app = typer.Typer(help="Prepare immutable causal scientific inputs.")
web_app = typer.Typer(help="Run the read-only web process.")
ibkr_api_app = typer.Typer(help="Verify first-party IBKR API provenance and check for updates.")
app.add_typer(bundle_app, name="bundle")
app.add_typer(context_app, name="context")
app.add_typer(database_app, name="db")
app.add_typer(replay_app, name="replay")
app.add_typer(recorder_app, name="recorder")
app.add_typer(scientific_inputs_app, name="scientific-inputs")
app.add_typer(web_app, name="web")
app.add_typer(ibkr_api_app, name="ibkr-api")


class _ReplayMarketDataBoundary:
    """Marker used by the startup safety check; it has no order methods."""

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


def _emit(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _fatal(message: str, *, exit_code: int = 78) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(exit_code)


@ibkr_api_app.command("verify")
def ibkr_api_verify(
    provenance: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Verify the installed Python tree against official archive provenance."""

    try:
        require_official_ibkr_api(provenance)
        _emit(load_official_ibkr_api_provenance(provenance))
    except (OfficialIBKRDependencyError, OfficialIBKRApiProvenanceError) as exc:
        _fatal(str(exc))


@ibkr_api_app.command("register")
def ibkr_api_register(
    archive: Path = typer.Option(..., exists=True, dir_okay=False),
    installed_package_root: Path = typer.Option(..., exists=True, file_okay=False),
    provenance: Path = typer.Option(...),
    operator: str = typer.Option(...),
) -> None:
    """Register an installed tree only when it exactly matches IBKR's latest ZIP."""

    try:
        release = fetch_latest_official_ibkr_api_release()
        record = inspect_official_ibkr_api_archive(
            archive,
            installed_package_root=installed_package_root,
            release=release,
            registered_by=operator,
            checked_at=datetime.now(UTC),
        )
        write_immutable_official_ibkr_api_provenance(provenance, record)
        _emit(record)
    except OfficialIBKRApiProvenanceError as exc:
        _fatal(str(exc))


@ibkr_api_app.command("check-update")
def ibkr_api_check_update(
    provenance: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
) -> None:
    """Record whether IBKR advertises a newer first-party archive; never install it."""

    try:
        installed = load_official_ibkr_api_provenance(provenance)
        latest = fetch_latest_official_ibkr_api_release()
        status = evaluate_official_ibkr_api_update(
            installed,
            latest,
            checked_at=datetime.now(UTC),
        )
        write_official_ibkr_api_update_status(output, status)
        _emit(status)
    except OfficialIBKRApiProvenanceError as exc:
        _fatal(str(exc))


def _mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{path} must contain an object/mapping")
    return payload


def _resolve_spec_paths(spec_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for name in (
        "m0_artifact",
        "m1_artifact",
        "preprocessor",
        "feature_schema",
        "universe",
        "threshold_provenance",
    ):
        candidate = Path(str(result[name]))
        result[name] = candidate if candidate.is_absolute() else spec_path.parent / candidate
    for name in ("audit_references", "determinism_references"):
        result[name] = [
            candidate if candidate.is_absolute() else spec_path.parent / candidate
            for value in result.get(name, [])
            for candidate in (Path(str(value)),)
        ]
    runtime = result.get("feature_runtime")
    if isinstance(runtime, dict):
        result["feature_runtime"] = {
            name: (candidate if candidate.is_absolute() else spec_path.parent / candidate)
            for name, value in runtime.items()
            for candidate in (Path(str(value)),)
        }
    return result


@bundle_app.command("reconstruct")
def bundle_reconstruct(
    frozen_root: Path = typer.Option(..., exists=True, file_okay=False),
    universe: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    bundle_id: str = typer.Option(...),
    created_at_utc: str = typer.Option(..., help="Timezone-aware ISO-8601 timestamp."),
    operator: str = typer.Option(...),
    feature_runtime_registry: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Registered frozen H0/front-options feature-runtime contract.",
    ),
    repository_root: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Research repository root used only while copying registered artifacts.",
    ),
) -> None:
    """Reconstruct no-fit deployable artifacts from the audited frozen JSON."""

    try:
        created_at = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
        _emit(
            reconstruct_frozen_artifacts(
                frozen_root=frozen_root,
                universe_path=universe,
                output_directory=output,
                bundle_id=bundle_id,
                created_at_utc=created_at,
                operator=operator,
                feature_runtime_registry_path=feature_runtime_registry,
                repository_root=repository_root,
            )
        )
    except (FrozenArtifactReconstructionError, ValueError) as exc:
        _fatal(str(exc))


@bundle_app.command("build")
def bundle_build(
    spec: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
) -> None:
    """Build a self-contained bundle on the research machine."""

    try:
        build_spec = BundleBuildSpec.model_validate(_resolve_spec_paths(spec, _mapping(spec)))
        _emit(build_bundle(build_spec, output))
    except (BundleError, ValueError) as exc:
        _fatal(str(exc))


@bundle_app.command("inspect")
def bundle_inspect(path: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Inspect a bundle and return its strict manifest plus verification."""

    try:
        _emit(verify_bundle(path))
    except BundleError as exc:
        _fatal(str(exc))


@bundle_app.command("verify")
def bundle_verify(path: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Verify all hashes and contracts."""

    try:
        result = verify_bundle(path)
        _emit(result)
        if not result.verified:
            raise typer.Exit(78)
    except BundleError as exc:
        _fatal(str(exc))


@bundle_app.command("install")
def bundle_install(
    path: Path = typer.Argument(..., exists=True, file_okay=False),
    bundle_root: Path = typer.Option(...),
    operator: str = typer.Option(...),
) -> None:
    """Copy a verified bundle into the versioned server store."""

    try:
        installed = install_bundle(path, bundle_root, operator=operator)
        _emit({"installed": str(installed)})
    except BundleError as exc:
        _fatal(str(exc))


@bundle_app.command("list")
def bundle_list(bundle_root: Path = typer.Option(...)) -> None:
    """List verified installed bundles."""

    _emit([item.model_dump(mode="json") for item in list_installed_bundles(bundle_root)])


@bundle_app.command("activate")
def bundle_activate(
    bundle_id: str = typer.Argument(...),
    bundle_root: Path = typer.Option(...),
    operator: str = typer.Option(...),
    expected_current: str | None = typer.Option(
        None,
        help="Compare-and-swap current bundle ID; omit only for first activation.",
    ),
) -> None:
    """Atomically activate a verified bundle for a future run."""

    try:
        _emit(
            activate_bundle(
                bundle_id,
                bundle_root,
                operator=operator,
                expected_current_bundle_id=expected_current,
            )
        )
    except BundleError as exc:
        _fatal(str(exc))


@context_app.command("sign")
def context_sign(
    unsigned: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    secret_env: str = typer.Option(...),
) -> None:
    """Create an integrity-protected package; the secret is read only from env."""

    if output.exists():
        _fatal(f"refusing to overwrite {output}")
    secret = os.environ.get(secret_env)
    if not secret:
        _fatal(f"context signing environment variable {secret_env} is absent")
    try:
        source = DailyContextUnsigned.model_validate_json(unsigned.read_text(encoding="utf-8"))
        package = create_signed_context(source, secret.encode())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
        _emit(
            {
                "context_id": package.context_id,
                "session_date": package.session_date,
                "integrity_hash": package.integrity_hash,
                "output": str(output),
            }
        )
    except (ContextValidationError, ValueError) as exc:
        _fatal(str(exc))


@context_app.command("inspect")
def context_inspect(package: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Inspect a signed package without exposing its signing secret."""

    try:
        _emit(SignedDailyContext.model_validate_json(package.read_text(encoding="utf-8")))
    except ValueError as exc:
        _fatal(str(exc))


@context_app.command("import")
def context_import(
    package: Path = typer.Argument(..., exists=True, dir_okay=False),
    context_root: Path = typer.Option(...),
    current_session: str = typer.Option(..., help="Current US session date (YYYY-MM-DD)."),
    secret_env: str = typer.Option(...),
    operator: str = typer.Option(...),
    expected_schema_hash: str | None = typer.Option(None),
    expected_feature_hash: str | None = typer.Option(None),
) -> None:
    """Install and map a package to one exact current US session."""

    secret = os.environ.get(secret_env)
    if not secret:
        _fatal(f"context signing environment variable {secret_env} is absent")
    try:
        _emit(
            import_signed_context(
                package,
                context_root=context_root,
                current_session=date.fromisoformat(current_session),
                secret=secret.encode(),
                operator=operator,
                expected_schema_hash=expected_schema_hash,
                expected_feature_hash=expected_feature_hash,
            )
        )
    except ContextValidationError as exc:
        _fatal(str(exc))


@database_app.command("migrate")
def database_migrate(database: Path = typer.Option(...)) -> None:
    """Apply append-only schema migrations with WAL enabled."""

    repository = ProspectiveRepository(database)
    repository.migrate()
    _emit({"database": "migrated", "path_exposed_in_api": False})


@database_app.command("backup")
def database_backup(
    database: Path = typer.Option(..., exists=True, dir_okay=False),
    destination: Path = typer.Option(...),
) -> None:
    """Create a checked, immutable online backup; no evidence is pruned."""

    try:
        _emit(backup_database(database, destination))
    except Exception as exc:
        _fatal(str(exc))


def _replay_settings(
    config: ProspectiveConfig,
    *,
    owner_id: str | None = None,
) -> ReplaySettings:
    if config.runtime.run_id is None:
        raise ValueError("replay requires an explicit runtime.run_id")
    if config.paths.replay_universe is None:
        raise ValueError("replay requires paths.replay_universe")
    return ReplaySettings(
        database_path=config.paths.database,
        run_id=config.runtime.run_id,
        prospective_start_utc=config.runtime.prospective_start_utc,
        app_version=config.runtime.app_version,
        git_commit=config.runtime.git_commit,
        universe_path=config.paths.replay_universe,
        owner_id=config.runtime.instance_id if owner_id is None else owner_id,
        recorder_lease_stale_seconds=config.runtime.recorder_lease_stale_seconds,
    )


def _recorder_owner_id(config: ProspectiveConfig) -> str:
    """Return a process-unique identity even when two processes share one config."""

    return f"{config.runtime.instance_id}:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _ibkr_adapter(config: ProspectiveConfig) -> IBKRMarketDataAdapter:
    if config.ibkr.port is None:
        raise RuntimeSafetyError(
            "blocked_unsafe_runtime_configuration: IBKR port must be explicitly configured"
        )
    connection_config = IBKRConnectionConfig(
        host=config.ibkr.host,
        port=config.ibkr.port,
        client_id=config.ibkr.client_id,
        expected_environment=config.ibkr.expected_environment,
        connect_timeout_seconds=config.ibkr.connect_timeout_seconds,
        request_timeout_seconds=config.ibkr.request_timeout_seconds,
        quote_capture_timeout_seconds=config.ibkr.quote_capture_timeout_seconds,
        allowed_market_data_types=tuple(
            MarketDataType(value) for value in config.ibkr.allowed_market_data_types
        ),
    )
    return IBKRMarketDataAdapter(
        config=connection_config,
        budget=MarketDataBudget(
            line_limit=config.ibkr.market_data_line_budget,
            reserved_headroom=(
                config.ibkr.externally_reserved_lines
                + config.ibkr.reserved_future_trading_lines
                + config.ibkr.safety_margin_lines
            ),
            request_rate_limit=config.ibkr.request_rate_per_second,
        ),
        socket_preflight=require_ibkr_socket_loopback_only,
        require_durable_inbox_on_start=True,
    )


def _validate_ibkr_scoring_inputs(
    config: ProspectiveConfig,
) -> RecorderDeploymentIdentity:
    try:
        verification = load_active_bundle(config.paths.bundle_root)
    except BundleError as exc:
        if (
            config.runtime.mode != "record_only"
            or not str(exc).startswith("blocked_missing_verified_frozen_bundle")
            or config.paths.replay_universe is None
        ):
            raise
        universe = load_registered_universe(config.paths.replay_universe)
        return RecorderDeploymentIdentity.from_registered_universe(universe)
    identity = RecorderDeploymentIdentity.from_bundle(verification)
    if config.runtime.mode != "shadow":
        return identity
    parity = load_feature_parity_report(config.paths.feature_parity_report)
    parity.require_scoring_allowed()
    if config.paths.context_root is None:
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: context_root is absent"
        )
    secret = os.environ.get(config.context.hmac_secret_env)
    if not secret:
        raise ContextValidationError(
            "blocked_missing_previous_session_options_context: signing secret is absent"
        )
    current_session = datetime.now(ZoneInfo("America/New_York")).date()
    load_imported_context(
        context_root=config.paths.context_root,
        current_session=current_session,
        secret=secret.encode(),
        expected_schema_hash=verification.manifest.previous_session_context_schema_hash,
        expected_feature_hash=verification.manifest.previous_session_context_feature_hash,
        expected_symbols=tuple(verification.manifest.universe.symbols),
    )
    return identity


@scientific_inputs_app.command("build-activity-baseline")
def build_activity_baseline(
    config_path: Path = typer.Option(..., "--config", exists=True),
    from_session: str = typer.Option(..., "--from-session"),
    latest_authorised_session: str = typer.Option(..., "--latest-authorised-session"),
    minimum_sessions: int = typer.Option(10, min=10),
) -> None:
    """Acquire and create the exact 20-stock EODHD activity baseline once."""

    try:
        config = load_prospective_config(config_path)
        identity = _validate_ibkr_scoring_inputs(config)
        output = config.paths.historical_activity_bars
        if output is None:
            raise RuntimeSafetyError("historical_activity_bars path is required")
        result = acquire_eodhd_historical_activity_baseline(
            symbols=identity.symbols,
            from_session=date.fromisoformat(from_session),
            latest_authorised_session=date.fromisoformat(latest_authorised_session),
            output_path=output,
            minimum_sessions=minimum_sessions,
        )
        _emit(result)
    except Exception as exc:
        _fatal(str(exc))


@replay_app.command("run")
def replay_run(config_path: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Run the deterministic fixture and exit."""

    config: ProspectiveConfig | None = None
    owner_id: str | None = None
    try:
        config = load_prospective_config(config_path)
        validate_runtime_safety(config, _ReplayMarketDataBoundary())
        owner_id = _recorder_owner_id(config)
        _emit(
            run_deterministic_replay(
                _replay_settings(
                    config,
                    owner_id=owner_id,
                )
            )
        )
    except Exception as exc:
        _fatal(str(exc))
    finally:
        if (
            config is not None
            and config.runtime.run_id is not None
            and owner_id is not None
            and config.paths.database.is_file()
        ):
            ProspectiveRepository(config.paths.database).release_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=owner_id,
            )


def _record_interrupted_streaming_episode_gaps(
    *,
    repository: ProspectiveRepository,
    operational_repository: RecorderOperationalRepository,
    run_id: str,
    recorder_generation: int,
    detected_at: datetime,
) -> tuple[str, ...]:
    """Persist each process-discontinuous option episode exactly once."""

    restorable_schedules = FrozenRecorderRepository(repository).restorable_option_episode_schedules(
        run_id=run_id
    )
    interrupted_streams = tuple(
        row for row in restorable_schedules if str(row["status"]) == "streaming"
    )
    existing_gap_episodes = {
        episode_id
        for gap in operational_repository.active_gaps(run_id=run_id)
        if gap.cause_code == "PROCESS_RESTART_OPTION_CONTINUITY_LOST"
        for episode_id in gap.affected_episode_ids
    }
    for row in interrupted_streams:
        episode_id = str(row["episode_id"])
        if episode_id in existing_gap_episodes:
            continue
        gap_start = datetime.fromisoformat(str(row["updated_at_utc"]))
        if gap_start.tzinfo is None or gap_start.utcoffset() is None:
            raise RuntimeSafetyError(
                "blocked_option_episode_continuity_timestamp_not_timezone_aware"
            )
        gap_start = gap_start.astimezone(UTC)
        operational_repository.record_gap(
            GapIncident(
                gap_id=stable_gap_id(
                    run_id=run_id,
                    recorder_generation=recorder_generation,
                    symbol=str(row["symbol"]),
                    stream_kind="option_episode",
                    request_id=None,
                    connection_generation=0,
                    start_timestamp_utc=gap_start,
                    cause_code="PROCESS_RESTART_OPTION_CONTINUITY_LOST",
                    incident_identity=episode_id,
                ),
                run_id=run_id,
                recorder_generation=recorder_generation,
                symbol=str(row["symbol"]),
                stream_kind="option_episode",
                request_id=None,
                connection_generation=0,
                start_timestamp_utc=gap_start,
                detection_timestamp_utc=detected_at,
                cause_code="PROCESS_RESTART_OPTION_CONTINUITY_LOST",
                severity="scientific",
                recoverability="unrecoverable",
                affected_episode_ids=(episode_id,),
            )
        )
    return tuple(str(row["episode_id"]) for row in interrupted_streams)


def _fail_closed_on_unclean_recorder_takeover(
    *,
    lease: LeaseRecord,
    config: ProspectiveConfig,
    owner_id: str,
    repository: ProspectiveRepository,
    durable_inbox: DurableCallbackInbox,
    operational_repository: RecorderOperationalRepository,
) -> bool:
    """Latch uncertain continuity while allowing raw-only inbox recovery."""

    assert config.runtime.run_id is not None
    with repository._connect() as connection:
        prior_same_run_generation = (
            connection.execute(
                """
                SELECT 1
                FROM recorder_generation_v1
                WHERE run_id = ? AND recorder_generation < ?
                LIMIT 1
                """,
                (config.runtime.run_id, lease.generation),
            ).fetchone()
            is not None
        )
    same_run_stale_takeover = (
        lease.recovered_stale_owner and lease.previous_run_id == config.runtime.run_id
    )
    if not (same_run_stale_takeover or prior_same_run_generation):
        return False
    observed = datetime.now(UTC)
    operational_repository.start_generation(
        run_id=config.runtime.run_id,
        recorder_generation=lease.generation,
        owner_id=owner_id,
        started_at=observed,
        required_market_data_mode=(
            config.ibkr.allowed_market_data_types[0]
            if len(config.ibkr.allowed_market_data_types) == 1
            else None
        ),
        # The application replaces this placeholder with the exact expected
        # artifact count before it records any verification result.
        expected_artifact_count=1,
    )
    interrupted_episode_ids = _record_interrupted_streaming_episode_gaps(
        repository=repository,
        operational_repository=operational_repository,
        run_id=config.runtime.run_id,
        recorder_generation=lease.generation,
        detected_at=observed,
    )
    durable_inbox.record_incident(
        stable_error_code="RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN",
        component="recorder_startup",
        severity="fatal",
        occurred_at=observed,
        error_class="UncleanRecorderRestart",
        evidence_loss_possible=True,
        details={
            "recovered_stale_owner": lease.recovered_stale_owner,
            "previous_run_id": lease.previous_run_id,
            "previous_owner_id": lease.previous_owner_id,
            "previous_heartbeat_at_utc": (
                None
                if lease.previous_heartbeat_at_utc is None
                else lease.previous_heartbeat_at_utc.isoformat()
            ),
            "socket_opened_at_detection": False,
            "raw_inbox_recovery_continues": True,
            "scientific_scoring_blocked": True,
            "interrupted_streaming_episode_ids": interrupted_episode_ids,
            "operator_action": (
                "preserve raw recovery and explicitly resolve the fatal audit "
                "before any scientific use"
            ),
        },
    )
    durable_inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN",
        occurred_at=observed,
        error_class="UncleanRecorderRestart",
        evidence_loss_possible=True,
        first_possibly_lost_source_sequence=(durable_inbox.latest_source_sequence()),
    )
    return True


@recorder_app.command("run")
def recorder_run(
    config_path: Path = typer.Option(..., "--config", exists=True),
    release_directory: Path = typer.Option(..., exists=True, file_okay=False),
    once: bool = typer.Option(False, help="Exit after a deterministic replay pass."),
) -> None:
    """Own the single recorder lease and market-data loop."""

    adapter: IBKRMarketDataAdapter | None = None
    frozen_application: Any | None = None
    parallel_capture: ParallelSourceCaptureService | None = None
    group_o_preparer: EODHDGroupOPreparationService | None = None
    reported_group_o_error: str | None = None
    repository: ProspectiveRepository | None = None
    lease_owned = False
    config: ProspectiveConfig | None = None
    owner_id: str | None = None
    durable_inbox: DurableCallbackInbox | None = None
    operational_repository: RecorderOperationalRepository | None = None
    recorder_generation: int | None = None
    fatal_exit = False
    try:
        config = load_prospective_config(config_path)
        validate_persistent_paths(config, release_directory)
        if config.runtime.run_id is None:
            raise RuntimeSafetyError(
                "blocked_unsafe_runtime_configuration: runtime.run_id is required"
            )
        owner_id = _recorder_owner_id(config)
        if config.runtime.source == "replay":
            validate_runtime_safety(config, _ReplayMarketDataBoundary())
            result = run_deterministic_replay(
                _replay_settings(
                    config,
                    owner_id=owner_id,
                )
            )
            _emit(result)
            repository = ProspectiveRepository(config.paths.database)
            lease_owned = True
            repository.open_anchor()
            if once:
                repository.release_recorder_lease(
                    run_id=config.runtime.run_id,
                    owner_id=owner_id,
                )
                lease_owned = False
                return
        else:
            if config.paths.frozen_m1c_artifact_root is None:
                raise RuntimeSafetyError(
                    "blocked_unsafe_runtime_configuration: IBKR callback recording "
                    "requires the durable raw recorder; legacy memory-drain "
                    "diagnostic mode cannot open a socket"
                )
            adapter = _ibkr_adapter(config)
            validate_runtime_safety(config, adapter)
            ibkr_api_module = require_official_ibkr_api()
            deployment_identity = _validate_ibkr_scoring_inputs(config)
            repository = ProspectiveRepository(config.paths.database)
            repository.migrate()
            repository.open_anchor()
            lease = repository.acquire_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=owner_id,
                now=datetime.now(UTC),
                stale_after=timedelta(seconds=config.runtime.recorder_lease_stale_seconds),
            )
            recorder_generation = lease.generation
            lease_owned = True
            durable_inbox = DurableCallbackInbox(
                config.paths.database,
                max_unacknowledged=(config.runtime.callback_inbox_max_unacknowledged),
                run_id=config.runtime.run_id,
                recorder_generation=recorder_generation,
                owner_id=owner_id,
            )
            adapter.attach_durable_inbox(durable_inbox)
            operational_repository = RecorderOperationalRepository(config.paths.database)
            _fail_closed_on_unclean_recorder_takeover(
                lease=lease,
                config=config,
                owner_id=owner_id,
                repository=repository,
                durable_inbox=durable_inbox,
                operational_repository=operational_repository,
            )
            from stocker_prospective.ibkr_official import (
                create_official_callback_client,
                create_official_option_contract,
                create_official_stock_contract,
            )

            adapter.attach_official_client(create_official_callback_client(adapter))

            def heartbeat() -> object:
                lease_record = repository.heartbeat_recorder_lease(
                    run_id=config.runtime.run_id or "",
                    owner_id=owner_id or "",
                    now=datetime.now(UTC),
                )
                if (
                    operational_repository is not None
                    and recorder_generation is not None
                    and frozen_application is not None
                ):
                    health = adapter.connection.health()
                    operational_repository.touch(
                        run_id=config.runtime.run_id or "",
                        recorder_generation=recorder_generation,
                        owner_id=owner_id or "",
                        now=datetime.now(UTC),
                        ibkr_connection_state=health.state.value,
                        observed_market_data_mode=(
                            None
                            if health.market_data_type is None
                            else health.market_data_type.value
                        ),
                        broker_state_mutation_count=0,
                    )
                return lease_record

            adapter.start()
            from stocker_prospective.frozen_live_application import (
                build_frozen_prospective_application,
            )

            frozen_application = build_frozen_prospective_application(
                config=config,
                adapter=adapter,
                repository=repository,
                identity=deployment_identity,
                stock_contract_factory=create_official_stock_contract,
                option_contract_factory=(
                    lambda symbol, expiry, strike, right, multiplier, exchange, trading: (
                        create_official_option_contract(
                            symbol=symbol,
                            expiry=expiry,
                            strike=strike,
                            right=right,
                            multiplier=multiplier,
                            exchange=exchange,
                            trading_class=trading,
                        )
                    )
                ),
                ibkr_api_version=str(getattr(ibkr_api_module, "__version__", "unknown")),
                heartbeat=heartbeat,
                durable_inbox=durable_inbox,
                recorder_generation=recorder_generation,
                recorder_owner_id=owner_id,
            )
            if config.parallel_validation.enabled:
                parallel_capture = build_parallel_eodhd_service(
                    config=config,
                    repository=repository,
                    identity=deployment_identity,
                    heartbeat=lambda: repository.heartbeat_recorder_lease(
                        run_id=config.runtime.run_id or "",
                        owner_id=owner_id or "",
                        now=datetime.now(UTC),
                    ),
                    completion_sink=(
                        None
                        if frozen_application is None
                        else frozen_application.process_source_transfer
                    ),
                )
            if frozen_application is not None and config.paths.context_root is not None:
                group_o_artifacts = (
                    release_directory
                    / "research"
                    / "cross-market-context"
                    / "20260723-daily-stock-front-options-context-v01"
                    / "artifacts"
                    / "primary"
                )
                group_o_preparer = EODHDGroupOPreparationService(
                    symbols=deployment_identity.symbols,
                    context_root=config.paths.context_root,
                    cache_root=(Path(config.paths.context_root) / "source-cache" / "eodhd-group-o"),
                    feature_manifest_path=(
                        group_o_artifacts / "front_options_feature_manifest.json"
                    ),
                    regime_mapping_path=(group_o_artifacts / "front_options_regime_mapping.json"),
                    capture_delay_seconds=(config.parallel_validation.capture_delay_seconds),
                    heartbeat=heartbeat,
                )
                group_o_preparer.poll(now=datetime.now(UTC))

        assert repository is not None
        stopping = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        next_lease_heartbeat = time.monotonic()
        while not stopping:
            if frozen_application is not None:
                frozen_application.poll(now=datetime.now(UTC))
                if group_o_preparer is not None:
                    group_o_preparer.poll(now=datetime.now(UTC))
                    if (
                        group_o_preparer.last_error is not None
                        and group_o_preparer.last_error != reported_group_o_error
                    ):
                        reported_group_o_error = group_o_preparer.last_error
                        typer.echo(
                            f"Group O preparation deferred: {reported_group_o_error}",
                            err=True,
                        )
            if parallel_capture is not None:
                parallel_capture.poll(now=datetime.now(UTC))
            monotonic_now = time.monotonic()
            if monotonic_now >= next_lease_heartbeat:
                repository.heartbeat_recorder_lease(
                    run_id=config.runtime.run_id,
                    owner_id=owner_id,
                    now=datetime.now(UTC),
                )
                next_lease_heartbeat = monotonic_now + config.runtime.heartbeat_seconds
            time.sleep(config.ibkr.stream_poll_interval_seconds)
    except typer.Exit:
        raise
    except (
        BundleError,
        ContextValidationError,
        FeatureParityError,
        OfficialIBKRDependencyError,
        RuntimeSafetyError,
        UniverseError,
        ValueError,
    ) as exc:
        fatal_exit = True
        _fatal(str(exc), exit_code=78)
    except Exception as exc:
        fatal_exit = True
        _fatal(str(exc), exit_code=75)
    finally:
        clean_shutdown = not fatal_exit
        fatal_latched = False
        if durable_inbox is not None:
            try:
                fatal_latched = durable_inbox.has_active_fatal()
            except Exception as exc:
                fatal_latched = True
                clean_shutdown = False
                typer.echo(f"fatal latch read failed: {exc}", err=True)
        if (
            operational_repository is not None
            and recorder_generation is not None
            and config is not None
            and config.runtime.run_id is not None
            and owner_id is not None
            and not fatal_latched
        ):
            try:
                operational_repository.set_stopping(
                    run_id=config.runtime.run_id,
                    recorder_generation=recorder_generation,
                    owner_id=owner_id,
                    now=datetime.now(UTC),
                )
            except Exception as exc:
                clean_shutdown = False
                typer.echo(f"recorder stopping state failed: {exc}", err=True)
        if group_o_preparer is not None:
            try:
                group_o_preparer.shutdown()
            except Exception as exc:
                clean_shutdown = False
                typer.echo(f"Group O preparation cleanup failed: {exc}", err=True)
        if (
            frozen_application is not None
            and repository is not None
            and operational_repository is not None
            and durable_inbox is not None
            and recorder_generation is not None
            and config is not None
            and config.runtime.run_id is not None
        ):
            observed_shutdown = datetime.now(UTC)
            try:
                interrupted_episode_ids = _record_interrupted_streaming_episode_gaps(
                    repository=repository,
                    operational_repository=operational_repository,
                    run_id=config.runtime.run_id,
                    recorder_generation=recorder_generation,
                    detected_at=observed_shutdown,
                )
                if interrupted_episode_ids:
                    durable_inbox.record_incident(
                        stable_error_code=("RECORDER_SHUTDOWN_EPISODE_CONTINUITY_LOST"),
                        component="recorder_shutdown",
                        severity="fatal",
                        occurred_at=observed_shutdown,
                        error_class="InterruptedStreamingEpisode",
                        evidence_loss_possible=True,
                        source_sequence=durable_inbox.latest_source_sequence(),
                        details={
                            "interrupted_streaming_episode_ids": (interrupted_episode_ids),
                            "scientific_scoring_blocked": True,
                        },
                    )
                    durable_inbox.latch_fatal(
                        latch_kind="ingestion",
                        stable_error_code=("RECORDER_SHUTDOWN_EPISODE_CONTINUITY_LOST"),
                        occurred_at=observed_shutdown,
                        error_class="InterruptedStreamingEpisode",
                        evidence_loss_possible=True,
                        first_possibly_lost_source_sequence=(
                            durable_inbox.latest_source_sequence()
                        ),
                    )
                    fatal_latched = True
            except Exception as exc:
                # A failed continuity audit must never fall through to
                # STOPPED_CLEANLY, even if the remaining cleanup succeeds.
                fatal_latched = True
                clean_shutdown = False
                typer.echo(f"episode continuity audit failed: {exc}", err=True)
        if frozen_application is not None:
            try:
                frozen_application.shutdown(now=datetime.now(UTC))
            except Exception as exc:
                clean_shutdown = False
                typer.echo(f"frozen recorder shutdown cleanup failed: {exc}", err=True)
        if adapter is not None:
            try:
                adapter.stop()
            except Exception as exc:
                clean_shutdown = False
                typer.echo(f"IBKR adapter cleanup failed: {exc}", err=True)
        if (
            clean_shutdown
            and not fatal_latched
            and operational_repository is not None
            and recorder_generation is not None
            and config is not None
            and config.runtime.run_id is not None
            and owner_id is not None
        ):
            try:
                operational_repository.set_stopped_cleanly(
                    run_id=config.runtime.run_id,
                    recorder_generation=recorder_generation,
                    owner_id=owner_id,
                    now=datetime.now(UTC),
                    termination_reason="operator_stop",
                )
            except Exception as exc:
                typer.echo(f"recorder stopped state failed: {exc}", err=True)
        if (
            lease_owned
            and repository is not None
            and config is not None
            and config.runtime.run_id is not None
            and owner_id is not None
        ):
            repository.release_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=owner_id,
            )
        if repository is not None:
            repository.close_anchor()


@web_app.command("run")
def web_run(config_path: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Serve the browser UI and read-only API on the configured private bind."""

    try:
        config = load_prospective_config(config_path)
        validate_runtime_safety(config, object())
        application = create_web_app(config)
        forwarded = ",".join(config.web.trusted_proxy_ips) if config.web.trust_proxy_headers else ""
        uvicorn.run(
            application,
            host=config.web.host,
            port=config.web.port,
            proxy_headers=config.web.trust_proxy_headers,
            forwarded_allow_ips=forwarded,
            log_level="info",
        )
    except typer.Exit:
        raise
    except (RuntimeSafetyError, ValueError) as exc:
        _fatal(str(exc), exit_code=78)
    except Exception as exc:
        _fatal(str(exc), exit_code=75)
