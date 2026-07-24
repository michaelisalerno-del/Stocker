"""Operator CLI for immutable bundles, replay, migrations, recorder, and web."""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NoReturn

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
)
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.ibkr import require_official_ibkr_api
from stocker_prospective.parity import load_feature_parity_report
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
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
app.add_typer(bundle_app, name="bundle")
app.add_typer(context_app, name="context")
app.add_typer(database_app, name="db")
app.add_typer(replay_app, name="replay")
app.add_typer(recorder_app, name="recorder")
app.add_typer(web_app, name="web")


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


def _replay_settings(config: ProspectiveConfig) -> ReplaySettings:
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
        owner_id=config.runtime.instance_id,
    )


@replay_app.command("run")
def replay_run(config_path: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Run the deterministic fixture and exit."""

    try:
        config = load_prospective_config(config_path)
        validate_runtime_safety(config, _ReplayMarketDataBoundary())
        _emit(run_deterministic_replay(_replay_settings(config)))
    except Exception as exc:
        _fatal(str(exc))


@recorder_app.command("run")
def recorder_run(
    config_path: Path = typer.Option(..., "--config", exists=True),
    release_directory: Path = typer.Option(..., exists=True, file_okay=False),
    once: bool = typer.Option(False, help="Exit after a deterministic replay pass."),
) -> None:
    """Own the single recorder lease and market-data loop."""

    try:
        config = load_prospective_config(config_path)
        validate_runtime_safety(config, _ReplayMarketDataBoundary())
        validate_persistent_paths(config, release_directory)
        if config.runtime.source == "replay":
            result = run_deterministic_replay(_replay_settings(config))
            _emit(result)
            if once:
                return
        else:
            require_official_ibkr_api()
            load_active_bundle(config.paths.bundle_root)
            parity = load_feature_parity_report(config.paths.feature_parity_report)
            if config.runtime.mode == "shadow":
                parity.require_scoring_allowed()
            _fatal(
                "blocked_ibkr_connection: configure and attach the official callback client",
                exit_code=69,
            )

        if config.runtime.run_id is None:
            raise ValueError("runtime.run_id is required")
        repository = ProspectiveRepository(config.paths.database)
        stopping = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stopping:
            repository.heartbeat_recorder_lease(
                run_id=config.runtime.run_id,
                owner_id=config.runtime.instance_id,
                now=datetime.now(UTC),
            )
            time.sleep(config.runtime.heartbeat_seconds)
    except Exception as exc:
        _fatal(str(exc))


@web_app.command("run")
def web_run(config_path: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Serve the browser UI and read-only API on the configured private bind."""

    try:
        config = load_prospective_config(config_path)
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
    except Exception as exc:
        _fatal(str(exc))
