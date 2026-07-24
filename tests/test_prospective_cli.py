from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from stocker_prospective.cli import app

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


def test_cli_exposes_operational_commands_and_no_order_command() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "bundle" in result.stdout
    assert "recorder" in result.stdout
    assert "web" in result.stdout
    assert "replay" in result.stdout
    lowered = result.stdout.lower()
    assert "place-order" not in lowered
    assert "cancel-order" not in lowered
    assert "\n│ orders " not in lowered


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
