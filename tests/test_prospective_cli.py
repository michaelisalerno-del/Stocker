from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from typer.testing import CliRunner

import stocker_prospective.cli as cli_module
import stocker_prospective.ibkr_official as ibkr_official_module
from stocker_prospective.bundle import BundleError
from stocker_prospective.cli import app
from stocker_prospective.config import load_prospective_config
from stocker_prospective.ibkr_api import OfficialIBKRApiProvenance, OfficialIBKRApiRelease

ROOT = Path(__file__).parents[1]
RUNNER = CliRunner()


def write_replay_config(tmp_path: Path) -> Path:
    path = tmp_path / "replay.yaml"
    payload = {
        "paths": {
            "database": str(tmp_path / "shared/data/prospective.sqlite3"),
            "bundle_root": str(tmp_path / "shared/bundles"),
            "feature_parity_report": str(ROOT / "configs/prospective/feature-parity-m1.json"),
            "context_root": str(tmp_path / "shared/context"),
            "replay_universe": str(ROOT / "configs/prospective/anchor-frozen-20.json"),
        },
        "runtime": {
            "mode": "shadow",
            "source": "replay",
            "prospective_start_utc": "2026-07-24T13:00:00Z",
            "instance_id": "cli-test",
            "app_version": "0.1.0-test",
            "git_commit": "deadbeef",
            "run_id": "cli-replay-001",
        },
        "risk": {"trading_enabled": False},
        "web": {"host": "127.0.0.1", "port": 8765},
        "ibkr": {},
        "context": {
            "mode": "signed_import",
            "hmac_secret_env": "STOCKER_CONTEXT_SIGNING_SECRET",
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def write_ibkr_config(tmp_path: Path) -> Path:
    path = write_replay_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["runtime"]["source"] = "ibkr"
    payload["runtime"]["mode"] = "record_only"
    payload["ibkr"] = {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 71,
        "expected_environment": "paper",
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_cli_exposes_operational_commands_and_no_order_command() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "bundle" in result.stdout
    assert "recorder" in result.stdout
    assert "web" in result.stdout
    assert "replay" in result.stdout
    assert "ibkr-api" in result.stdout
    lowered = result.stdout.lower()
    assert "place-order" not in lowered
    assert "cancel-order" not in lowered
    assert "\n│ orders " not in lowered


def test_ibkr_api_register_writes_verified_immutable_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "twsapi_macunix.1048.01.zip"
    package_source = 'VERSION = {"major": 10, "minor": 48, "micro": 1}\n'
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("IBJts/API_VersionNum.txt", "API_Version=10.48.01\r\n")
        bundle.writestr(
            "IBJts/source/pythonclient/ibapi/__init__.py",
            package_source,
        )
    installed_package = tmp_path / "installed/ibapi"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text(package_source, encoding="utf-8")
    release = OfficialIBKRApiRelease(
        api_version="10.48",
        release_date=date(2026, 7, 7),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
    )
    monkeypatch.setattr(
        cli_module,
        "fetch_latest_official_ibkr_api_release",
        lambda: release,
        raising=False,
    )
    provenance = tmp_path / "provenance.json"

    result = RUNNER.invoke(
        app,
        [
            "ibkr-api",
            "register",
            "--archive",
            str(archive),
            "--installed-package-root",
            str(installed_package),
            "--provenance",
            str(provenance),
            "--operator",
            "test-operator",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    assert payload["api_version"] == "10.48.1"
    assert payload["archive_filename"] == archive.name
    assert payload["registered_by"] == "test-operator"


def test_ibkr_api_update_check_records_status_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree_hash = "2fb1a3296db30cc2ec0c21503856b06990ca7f0fc2cefcfe6f4cbf8c9c196a63"
    installed = OfficialIBKRApiProvenance(
        schema_version="1",
        source="interactive_brokers_official_tws_api",
        release_channel="latest",
        platform="mac_unix",
        api_version="10.48.1",
        release_date=date(2026, 7, 7),
        official_page_url="https://interactivebrokers.github.io/",
        official_page_checked_at_utc=datetime(2026, 7, 25, 9, tzinfo=UTC),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
        archive_filename="twsapi_macunix.1048.01.zip",
        archive_sha256="0" * 64,
        source_tree_sha256=tree_hash,
        installed_tree_sha256=tree_hash,
        registered_at_utc=datetime(2026, 7, 25, 9, 5, tzinfo=UTC),
        registered_by="test-operator",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(installed.model_dump_json(indent=2), encoding="utf-8")
    latest = OfficialIBKRApiRelease(
        api_version="10.49",
        release_date=date(2026, 8, 4),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1049.01.zip"),
    )
    monkeypatch.setattr(
        cli_module,
        "fetch_latest_official_ibkr_api_release",
        lambda: latest,
    )
    status_path = tmp_path / "update-status.json"

    result = RUNNER.invoke(
        app,
        [
            "ibkr-api",
            "check-update",
            "--provenance",
            str(provenance),
            "--output",
            str(status_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    status = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    assert status["update_available"] is True
    assert status["automatic_installation"] is False


def test_cli_replay_runs_and_db_migration_is_idempotent(tmp_path: Path) -> None:
    config = write_replay_config(tmp_path)

    first = RUNNER.invoke(app, ["replay", "run", "--config", str(config)])
    second = RUNNER.invoke(app, ["replay", "run", "--config", str(config)])
    migrate = RUNNER.invoke(
        app,
        [
            "db",
            "migrate",
            "--database",
            str(tmp_path / "shared/data/prospective.sqlite3"),
        ],
    )

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert migrate.exit_code == 0, migrate.stdout
    assert "synthetic_replay_not_frozen_m1" in first.stdout
    assert '"signal_episode_count": 2' in first.stdout


def test_recorder_process_owner_is_unique_for_the_same_config(tmp_path: Path) -> None:
    config = load_prospective_config(write_replay_config(tmp_path))

    first = cli_module._recorder_owner_id(config)
    second = cli_module._recorder_owner_id(config)

    assert first != second
    assert first.startswith("cli-test:")


def test_record_only_can_use_verified_registered_universe_when_bundle_is_missing(
    tmp_path: Path,
) -> None:
    config = load_prospective_config(write_ibkr_config(tmp_path))

    identity = cli_module._validate_ibkr_scoring_inputs(config)

    assert identity.bundle_verified is False
    assert identity.model_artifact_id == "blocked_missing_verified_frozen_bundle"
    assert len(identity.symbols) == 20


def test_example_config_requires_explicit_deployed_git_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = ROOT / "configs/prospective/replay.example.yaml"
    monkeypatch.delenv("STOCKER_GIT_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="STOCKER_GIT_COMMIT is absent"):
        load_prospective_config(example)

    monkeypatch.setenv("STOCKER_GIT_COMMIT", "abcdef1234567890")
    loaded = load_prospective_config(example)
    assert loaded.runtime.git_commit == "abcdef1234567890"


def test_transient_ibkr_failure_uses_restartable_exit_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_ibkr_config(tmp_path)
    release = tmp_path / "release"
    release.mkdir()

    class FakeAdapter:
        stopped = False

        def attach_official_client(self, client: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("blocked_ibkr_connection")

        def stop(self) -> None:
            self.stopped = True

    adapter = FakeAdapter()
    monkeypatch.setattr(cli_module, "_ibkr_adapter", lambda _config: adapter)
    monkeypatch.setattr(cli_module, "require_official_ibkr_api", lambda: object())
    monkeypatch.setattr(cli_module, "_validate_ibkr_scoring_inputs", lambda _config: None)
    monkeypatch.setattr(
        ibkr_official_module,
        "create_official_callback_client",
        lambda _adapter: object(),
    )

    result = RUNNER.invoke(
        app,
        [
            "recorder",
            "run",
            "--config",
            str(config),
            "--release-directory",
            str(release),
        ],
    )

    assert result.exit_code == 75
    assert "blocked_ibkr_connection" in result.stderr
    assert adapter.stopped is True
    with sqlite3.connect(tmp_path / "shared/data/prospective.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM recorder_lease").fetchone() == (0,)


def test_permanent_bundle_failure_uses_restart_preventing_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_ibkr_config(tmp_path)
    release = tmp_path / "release"
    release.mkdir()

    class FakeAdapter:
        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "_ibkr_adapter", lambda _config: FakeAdapter())
    monkeypatch.setattr(cli_module, "require_official_ibkr_api", lambda: object())
    monkeypatch.setattr(
        cli_module,
        "_validate_ibkr_scoring_inputs",
        lambda _config: (_ for _ in ()).throw(
            BundleError("blocked_missing_verified_frozen_bundle")
        ),
    )

    result = RUNNER.invoke(
        app,
        [
            "recorder",
            "run",
            "--config",
            str(config),
            "--release-directory",
            str(release),
        ],
    )

    assert result.exit_code == 78
    assert "blocked_missing_verified_frozen_bundle" in result.stderr
