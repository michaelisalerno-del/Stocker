from __future__ import annotations

import json
import signal
from pathlib import Path

from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import CallbackFence, Recorder, load_recorder_config
from stocker_runtime.storage import connect_v2, initialize_database
from stocker_runtime.web import WebConfig

ROOT = Path(__file__).parents[1]
SYSTEMD = ROOT / "deploy/systemd"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def test_v2_services_have_distinct_least_privilege_filesystem_boundaries() -> None:
    recorder = _unit("stocker-v2-recorder.service")
    web = _unit("stocker-v2-web.service")
    daily = _unit("stocker-v2-backup-daily.service")
    weekly = _unit("stocker-v2-backup-weekly.service")

    assert "User=stocker-recorder" in recorder
    assert "User=stocker-web" in web
    assert "User=stocker-backup" in daily
    assert "User=stocker-backup" in weekly
    assert "ExecStart=/opt/stocker/v2-current/.venv/bin/stocker-runtime recorder run" in recorder
    assert "ExecStart=/opt/stocker/v2-current/.venv/bin/stocker-runtime web run" in web
    for unit in (recorder, web, daily, weekly):
        assert "/opt/stocker/current" not in unit
        assert "/opt/stocker/v2-current" in unit
    assert "--tier daily" in daily
    assert "--tier weekly" in weekly
    assert "--working-directory /var/cache/stocker-v2-backup-work" in daily
    assert "CacheDirectory=stocker-v2-backup-work" in daily
    assert "TimeoutStartSec=31min" in daily
    assert "TimeoutStartSec=31min" in weekly
    assert "ExecCondition=" not in daily
    assert "ExecCondition=" not in weekly

    assert "ReadWritePaths=/var/lib/stocker/v2" in recorder
    assert "ReadOnlyPaths=/var/lib/stocker/backups-v2" in recorder
    assert "ReadWritePaths=/var/lib/stocker/backups-v2" not in recorder
    assert "ReadOnlyPaths=/var/lib/stocker/v2" in web
    assert "ReadOnlyPaths=/var/lib/stocker/backups-v2" in web
    assert "ReadWritePaths=/var/lib/stocker/v2" not in web.splitlines()
    assert "ReadOnlyPaths=/var/lib/stocker/v2" in daily
    assert "ReadWritePaths=/var/lib/stocker/backups-v2" in daily
    assert "ReadWritePaths=/var/lib/stocker/v2" not in daily.splitlines()

    assert "IPAddressDeny=any" in recorder
    assert "IPAddressAllow=localhost" in recorder
    assert "IPAddressDeny=any" in web
    assert "IPAddressAllow=localhost" in web
    assert "IPAddressDeny=any" in daily
    assert "Restart=on-failure" in recorder
    assert "KillSignal=SIGTERM" in recorder
    assert "Restart=on-failure" in web
    assert "KillSignal=SIGTERM" in web


def test_daily_and_weekly_timers_are_bounded_and_not_implicitly_enabled() -> None:
    daily = _unit("stocker-v2-backup-daily.timer")
    weekly = _unit("stocker-v2-backup-weekly.timer")

    assert "OnCalendar=*-*-* 23:45:00 UTC" in daily
    assert "OnCalendar=Sun *-*-* 22:45:00 UTC" in weekly
    assert "Persistent=true" in daily
    assert "Persistent=true" in weekly
    assert "RandomizedDelaySec=" in daily
    assert "RandomizedDelaySec=" in weekly
    assert not (SYSTEMD / "stocker-backup.service").exists()
    assert not (SYSTEMD / "stocker-backup.timer").exists()


def test_cutover_release_contains_only_v2_stocker_application_services() -> None:
    for retired in (
        "stocker-recorder.service",
        "stocker-web.service",
        "stocker-backup.service",
        "stocker-backup.timer",
    ):
        assert not (SYSTEMD / retired).exists()
    assert not (ROOT / "deploy/stocker.env.example").exists()
    assert not (ROOT / "deploy/scripts/prepare-web-sqlite-boundary.py").exists()
    assert (ROOT / "docs/operations/stocker-v2-cutover.md").is_file()
    assert not (ROOT / "docs/operations/prospective-server-runbook.md").exists()

    active_surface = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "deploy").rglob("*") if path.is_file()
    ).lower()
    assert "stocker-prospective" not in active_surface
    assert "configs/prospective" not in active_surface


def test_v2_deployment_contains_no_legacy_vendor_transfer_or_execution_fields() -> None:
    deployment_files = [
        ROOT / "deploy/stocker-recorder.env.example",
        ROOT / "deploy/stocker-web.env.example",
        ROOT / "deploy/stocker-backup.env.example",
        ROOT / "configs/runtime/recorder.example.json",
        ROOT / "configs/runtime/market-data.example.json",
        ROOT / "configs/runtime/web.example.json",
        SYSTEMD / "stocker-v2-recorder.service",
        SYSTEMD / "stocker-v2-web.service",
        SYSTEMD / "stocker-v2-backup-daily.service",
        SYSTEMD / "stocker-v2-backup-weekly.service",
        SYSTEMD / "stocker-v2-backup-daily.timer",
        SYSTEMD / "stocker-v2-backup-weekly.timer",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in deployment_files).lower()

    for forbidden in (
        "eodhd",
        "source_transfer",
        "source-transfer",
        "parallel",
        "parquet",
        "report.zip",
        "place_order",
        "order service",
        "paper_enabled",
        "live_enabled",
    ):
        assert forbidden not in joined
    assert "ibkr_read_only" not in joined
    assert "credential" not in joined


def test_v2_deployment_examples_validate_with_only_record_shadow_authority() -> None:
    recorder = load_recorder_config(ROOT / "configs/runtime/recorder.example.json")
    web = WebConfig.model_validate_json(
        (ROOT / "configs/runtime/web.example.json").read_text(encoding="utf-8")
    )
    market_data = json.loads(
        (ROOT / "configs/runtime/market-data.example.json").read_text(encoding="utf-8")
    )

    assert recorder.mode == "prospective_record"
    assert recorder.host == "127.0.0.1"
    assert recorder.read_only is True
    assert recorder.external_read_only_verified is True
    assert str(recorder.database) == "/var/lib/stocker/v2/stocker-v2.sqlite3"
    assert web.host == "127.0.0.1"
    assert str(web.database) == "/var/lib/stocker/v2/stocker-v2.sqlite3"
    assert str(web.backup_directory) == "/var/lib/stocker/backups-v2"
    assert market_data == {"instruments": [], "subscriptions": []}


def test_gateway_units_are_conspicuously_market_data_only_without_capability_change() -> None:
    related = tuple(SYSTEMD.glob("stocker-ibgateway*.service")) + tuple(
        SYSTEMD.glob("stocker-ibgateway*.socket")
    )
    assert related
    for path in related:
        text = path.read_text(encoding="utf-8")
        assert "market-data-only" in text.lower(), path.name
    gateway = _unit("stocker-ibgateway.service")
    assert "ExecStart=/opt/ibgateway/current/ibgateway" in gateway
    assert "Restart=always" in gateway
    assert "EnvironmentFile=" not in gateway


def test_sqlite_boundary_preparation_targets_only_v2_and_backup_paths() -> None:
    script_path = ROOT / "deploy/scripts/prepare-v2-sqlite-boundary.py"
    source = script_path.read_text(encoding="utf-8")

    assert not (ROOT / "deploy/scripts/prepare-web-sqlite-boundary.py").exists()
    assert 'DATABASE_DIRECTORY_NAME = "v2"' in source
    assert 'BACKUP_DIRECTORY_NAME = "backups-v2"' in source
    assert 'DATABASE_NAME = "stocker-v2.sqlite3"' in source
    assert 'RECORDER_USER = "stocker-recorder"' in source
    assert 'WEB_USER = "stocker-web"' in source
    assert 'BACKUP_USER = "stocker-backup"' in source
    assert 'READER_GROUP = "stocker-readers"' in source
    assert "prospective.sqlite3" not in source
    assert "bundles" not in source


def test_cutover_precreates_reader_boundary_before_import_and_verifies_afterward() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")

    assert "groupadd --system stocker-readers" in runbook
    assert runbook.count("--gid stocker-readers") == 3
    assert "install -d -o root -g stocker-readers -m 0750 /var/lib/stocker" in runbook
    assert (
        "install -d -o stocker-recorder -g stocker-readers -m 2750 /var/lib/stocker/v2" in runbook
    )
    assert (
        "install -d -o stocker-backup -g stocker-readers -m 2750 "
        "/var/lib/stocker/backups-v2" in runbook
    )
    install_verifier = "sudo install -o root -g root -m 0755"
    assert install_verifier in runbook
    assert "/opt/stocker/v2-current/deploy/scripts/prepare-v2-sqlite-boundary.py" in runbook
    assert "/usr/local/libexec/stocker-prepare-v2-sqlite-boundary" in runbook
    importer = runbook.index("stocker-runtime legacy import")
    verifier = runbook.index("sudo /usr/local/libexec/stocker-prepare-v2-sqlite-boundary")
    assert runbook.index(install_verifier) < importer
    assert importer < verifier


def test_cutover_runbook_preserves_one_writer_and_two_distinct_rollback_paths() -> None:
    runbook = (ROOT / "docs/operations/stocker-v2-cutover.md").read_text(encoding="utf-8")
    lowered = runbook.lower()

    for required in (
        "market-closed",
        "owner approval",
        "read-only recovery set",
        "quick_check",
        "foreign_key_check",
        "source_row_count",
        "imported_row_count",
        "omitted_row_count",
        "query-only",
        "first callback",
        "zero order capability",
        "observation window",
        "checked v2 backup",
        "there is no dual write",
        "before first callback",
        "after first callback",
        "new v1 run and recorder generation",
        "explicit gap",
        "never reverse-import",
        "owner closes the rollback window",
        "owner-only rollback-window closure",
        "installed v1 units",
        "/opt/stocker/v2-current",
        "/opt/stocker/current",
    ):
        assert required in lowered
    assert lowered.index("start the v2 web") < lowered.index("start the v2 recorder")
    assert lowered.index("declare v2 admission operational") < lowered.index(
        "owner-only rollback-window closure"
    )
    assert "never remove the v1 units" in lowered
    assert "never repoint\n`/opt/stocker/current`, before the owner" in lowered
    assert "paper trading" not in lowered
    assert "live trading" not in lowered


def test_official_api_update_service_uses_the_v2_runtime_cli() -> None:
    unit = _unit("stocker-ibkr-api-update.service")

    assert "stocker-runtime ibkr-api check-update" in unit
    assert "stocker-prospective" not in unit
    assert "/opt/stocker/v2-current" in unit
    assert "/opt/stocker/current" not in unit
    assert "User=stocker-recorder" in unit
    assert "Group=stocker-readers" in unit


def test_installable_release_has_no_v1_runtime_or_entrypoint() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    check_script = (ROOT / "scripts/check.sh").read_text(encoding="utf-8")

    assert not tuple((ROOT / "packages/stocker_prospective").rglob("*.py"))
    assert not (ROOT / "configs/prospective").exists()
    assert "stocker-prospective" not in project
    assert "packages/stocker_prospective" not in project
    assert "check_prospective" not in workflow
    assert "stocker-prospective" not in workflow
    assert "check_prospective" not in check_script
    final_fixture = (
        ROOT
        / "tests/fixtures/legacy_prospective_migrations/0026_opening_leader_continuation_v0.sql"
    )
    assert final_fixture.is_file()


class _SignalMarketData:
    def __init__(self, on_connect: object) -> None:
        self.on_connect = on_connect
        self.callback = None

    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def set_disconnect_callback(self, _callback: object) -> None:
        return None

    def set_status_callback(self, _callback: object) -> None:
        return None

    def configure_subscriptions(self, _subscriptions: object) -> None:
        return None

    def connect(self) -> None:
        assert callable(self.on_connect)
        self.on_connect(signal.SIGTERM, None)

    def disconnect(self) -> None:
        return None

    def subscribe(self, _fence: CallbackFence) -> None:
        return None

    def cancel(self, _request_id: int) -> None:
        return None


def test_recorder_service_command_handles_sigterm_as_a_clean_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    (backup_directory / "backup-status.json").write_text(
        '{"checked_at_us":1,"code":"BackupCapacityError","format_version":1,'
        '"latest_manifest_filename":null,"state":"degraded"}\n',
        encoding="utf-8",
    )
    initialize_database(database)
    config = tmp_path / "recorder.json"
    inputs = tmp_path / "market-data.json"
    config.write_text(
        json.dumps(
            {
                "database": str(database),
                "run_id": "run-service",
                "owner_id": "owner-service",
                "mode": "prospective_record",
                "host": "127.0.0.1",
                "port": 4003,
                "client_id": 71,
                "read_only": True,
                "external_read_only_verified": True,
                "config_hash": "a" * 64,
                "git_commit": "0000000",
                "backup_directory": str(backup_directory),
            }
        ),
        encoding="utf-8",
    )
    inputs.write_text('{"instruments":[],"subscriptions":[]}', encoding="utf-8")
    handlers: dict[int, object] = {}

    def install_handler(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr("stocker_runtime.cli.signal.signal", install_handler)

    def create_adapter(**_kwargs: object) -> _SignalMarketData:
        return _SignalMarketData(lambda signum, frame: handlers[signum](signum, frame))

    monkeypatch.setattr("stocker_runtime.cli.IBKRMarketData.official", create_adapter)
    result = CliRunner().invoke(
        app,
        ["recorder", "run", "--config", str(config), "--inputs", str(inputs)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["termination"] == "sigterm"
    with connect_v2(database) as connection:
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-service'"
        ).fetchone()
        generation = connection.execute(
            "SELECT clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='run-service'"
        ).fetchone()
        incident = connection.execute(
            "SELECT scope, severity, code, details_json FROM incidents "
            "WHERE run_id='run-service' AND code='BACKUP_DEGRADED'"
        ).fetchone()
    assert tuple(run) == ("stopped", run["ended_at_us"])
    assert run["ended_at_us"] is not None
    assert tuple(generation) == (1, "CLEAN_STOP")
    assert tuple(incident) == (
        "storage",
        "degraded",
        "BACKUP_DEGRADED",
        '{"backup_status_code":"BackupCapacityError"}',
    )


def test_recorder_service_failure_leaves_an_unclean_generation_for_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "stocker-v2.sqlite3"
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    (backup_directory / "backup-status.json").write_text(
        '{"checked_at_us":1,"code":"BackupCapacityError","format_version":1,'
        '"latest_manifest_filename":null,"state":"degraded"}\n',
        encoding="utf-8",
    )
    initialize_database(database)
    config = tmp_path / "recorder.json"
    inputs = tmp_path / "market-data.json"
    config.write_text(
        json.dumps(
            {
                "database": str(database),
                "run_id": "run-restart",
                "owner_id": "owner-restart",
                "mode": "prospective_record",
                "host": "127.0.0.1",
                "port": 4003,
                "client_id": 71,
                "read_only": True,
                "external_read_only_verified": True,
                "config_hash": "b" * 64,
                "git_commit": "0000000",
                "backup_directory": str(backup_directory),
            }
        ),
        encoding="utf-8",
    )
    inputs.write_text('{"instruments":[],"subscriptions":[]}', encoding="utf-8")
    monkeypatch.setattr(
        "stocker_runtime.cli.signal.signal",
        lambda _signum, _handler: signal.SIG_DFL,
    )
    monkeypatch.setattr("stocker_runtime.cli.time.time_ns", lambda: 1_000_000_000)

    monkeypatch.setattr(
        "stocker_runtime.cli.IBKRMarketData.official",
        lambda **_kwargs: _SignalMarketData(lambda _signum, _frame: None),
    )
    real_drain = Recorder.drain

    def fail_drain(_recorder: Recorder, *, now_us: int) -> None:
        raise RuntimeError(f"temporary drain failure at {now_us}")

    monkeypatch.setattr(Recorder, "drain", fail_drain)
    failed = CliRunner().invoke(
        app,
        ["recorder", "run", "--config", str(config), "--inputs", str(inputs)],
    )

    assert failed.exit_code == 1
    with connect_v2(database) as connection:
        run = connection.execute(
            "SELECT status, ended_at_us FROM runs WHERE run_id='run-restart'"
        ).fetchone()
        generation = connection.execute(
            "SELECT ended_at_us, clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='run-restart' AND generation=1"
        ).fetchone()
    assert tuple(run) == ("running", None)
    assert tuple(generation) == (None, 0, None)

    monkeypatch.setattr(Recorder, "drain", real_drain)
    restarted = Recorder(
        load_recorder_config(config),
        _SignalMarketData(lambda _signum, _frame: None),
    )
    state = restarted.start(now_us=62_000_000, instruments=(), subscriptions=())
    assert state.recorder_generation == 2
    restarted.stop(now_us=63_000_000)
