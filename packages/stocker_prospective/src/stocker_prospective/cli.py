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
from stocker_prospective.database import ProspectiveRepository
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
from stocker_prospective.parity import FeatureParityError, load_feature_parity_report
from stocker_prospective.recorder import (
    IBKRDiagnosticRecorder,
    RecorderDeploymentIdentity,
)
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
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
web_app = typer.Typer(help="Run the read-only web process.")
ibkr_api_app = typer.Typer(help="Verify first-party IBKR API provenance and check for updates.")
app.add_typer(bundle_app, name="bundle")
app.add_typer(context_app, name="context")
app.add_typer(database_app, name="db")
app.add_typer(replay_app, name="replay")
app.add_typer(recorder_app, name="recorder")
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
    return result


@bundle_app.command("reconstruct")
def bundle_reconstruct(
    frozen_root: Path = typer.Option(..., exists=True, file_okay=False),
    universe: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    bundle_id: str = typer.Option(...),
    created_at_utc: str = typer.Option(..., help="Timezone-aware ISO-8601 timestamp."),
    operator: str = typer.Option(...),
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
            reserved_headroom=config.ibkr.reserved_line_headroom,
            request_rate_limit=config.ibkr.request_rate_per_second,
        ),
        socket_preflight=require_ibkr_socket_loopback_only,
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


@recorder_app.command("run")
def recorder_run(
    config_path: Path = typer.Option(..., "--config", exists=True),
    release_directory: Path = typer.Option(..., exists=True, file_okay=False),
    once: bool = typer.Option(False, help="Exit after a deterministic replay pass."),
) -> None:
    """Own the single recorder lease and market-data loop."""

    adapter: IBKRMarketDataAdapter | None = None
    diagnostic_recorder: Any | None = None
    repository: ProspectiveRepository | None = None
    lease_owned = False
    config: ProspectiveConfig | None = None
    owner_id: str | None = None
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
            adapter = _ibkr_adapter(config)
            validate_runtime_safety(config, adapter)
            require_official_ibkr_api()
            deployment_identity = _validate_ibkr_scoring_inputs(config)
            repository = ProspectiveRepository(config.paths.database)
            repository.migrate()
            repository.open_anchor()
            repository.acquire_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=owner_id,
                now=datetime.now(UTC),
                stale_after=timedelta(seconds=config.runtime.recorder_lease_stale_seconds),
            )
            lease_owned = True
            from stocker_prospective.ibkr_official import (
                create_official_callback_client,
                create_official_stock_contract,
            )

            adapter.attach_official_client(create_official_callback_client(adapter))
            adapter.start()
            diagnostic_recorder = IBKRDiagnosticRecorder(
                config=config,
                repository=repository,
                adapter=adapter,
                identity=deployment_identity,
                contract_factory=create_official_stock_contract,
                heartbeat=lambda: repository.heartbeat_recorder_lease(
                    run_id=config.runtime.run_id or "",
                    owner_id=owner_id or "",
                    now=datetime.now(UTC),
                ),
            )

        assert repository is not None
        stopping = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stopping:
            if diagnostic_recorder is not None:
                diagnostic_recorder.poll(now=datetime.now(UTC))
            repository.heartbeat_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=owner_id,
                now=datetime.now(UTC),
            )
            time.sleep(config.runtime.heartbeat_seconds)
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
        _fatal(str(exc), exit_code=78)
    except Exception as exc:
        _fatal(str(exc), exit_code=75)
    finally:
        if diagnostic_recorder is not None:
            try:
                diagnostic_recorder.shutdown(now=datetime.now(UTC))
            except Exception as exc:
                typer.echo(f"recorder shutdown cleanup failed: {exc}", err=True)
        if adapter is not None:
            try:
                adapter.stop()
            except Exception as exc:
                typer.echo(f"IBKR adapter cleanup failed: {exc}", err=True)
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
