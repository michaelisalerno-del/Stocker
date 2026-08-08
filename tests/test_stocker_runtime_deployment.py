from __future__ import annotations

import json
import signal
from pathlib import Path

from typer.testing import CliRunner

from stocker_runtime.cli import app
from stocker_runtime.ingestion import CallbackFence, load_recorder_config
from stocker_runtime.storage import connect_v2, initialize_database
from stocker_runtime.web import WebConfig

ROOT = Path(__file__).parents[1]
SYSTEMD = ROOT / "deploy/systemd"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def test_v2_services_have_distinct_least_privilege_filesystem_boundaries() -> None:
    recorder = _unit("stocker-recorder.service")
    web = _unit("stocker-web.service")
    daily = _unit("stocker-backup-daily.service")
    weekly = _unit("stocker-backup-weekly.service")

    assert "User=stocker-recorder" in recorder
    assert "User=stocker-web" in web
    assert "User=stocker-backup" in daily
    assert "User=stocker-backup" in weekly
    assert "ExecStart=/opt/stocker/current/.venv/bin/stocker-runtime recorder run" in recorder
    assert "ExecStart=/opt/stocker/current/.venv/bin/stocker-runtime web run" in web
    assert "--tier daily" in daily
    assert "--tier weekly" in weekly

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
    daily = _unit("stocker-backup-daily.timer")
    weekly = _unit("stocker-backup-weekly.timer")

    assert "OnCalendar=*-*-* 23:45:00 UTC" in daily
    assert "OnCalendar=Sun *-*-* 22:45:00 UTC" in weekly
    assert "Persistent=true" in daily
    assert "Persistent=true" in weekly
    assert "RandomizedDelaySec=" in daily
    assert "RandomizedDelaySec=" in weekly
    assert not (SYSTEMD / "stocker-backup.service").exists()
    assert not (SYSTEMD / "stocker-backup.timer").exists()


def test_v2_deployment_contains_no_legacy_vendor_transfer_or_execution_fields() -> None:
    deployment_files = [
        ROOT / "deploy/stocker-recorder.env.example",
        ROOT / "deploy/stocker-web.env.example",
        ROOT / "deploy/stocker-backup.env.example",
        ROOT / "configs/runtime/recorder.example.json",
        ROOT / "configs/runtime/market-data.example.json",
        ROOT / "configs/runtime/web.example.json",
        *SYSTEMD.glob("stocker-*.service"),
        *SYSTEMD.glob("stocker-*.timer"),
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in deployment_files).lower()

    assert not (ROOT / "deploy/stocker.env.example").exists()
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
