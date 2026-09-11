from datetime import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.runs import Environment, RunConfig, RunWindow
from stocker_execution.runtime import build_paper_runtime


def _write_runtime_configs(tmp_path: Path, *, expected_account: bool) -> tuple[Path, Path]:
    runs = tmp_path / "runs.yaml"
    runs.write_text(
        """
universes:
  - universe_id: NASDAQ
    name: NASDAQ
    members:
      - {symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
runs:
  - run_id: paper-run
    enabled: true
    universe: NASDAQ
    strategy: TEST_EXECUTION
    environment: PAPER
    risk: {risk_per_trade: 0.001, max_concurrent_positions: 5}
    session:
      start: "09:30"
      end: "16:00"
      timezone: America/New_York
      calendar: XNYS
""",
        encoding="utf-8",
    )
    ibkr = tmp_path / "ibkr.yaml"
    account_line = "  expected_account: DU123456\n" if expected_account else ""
    ibkr.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
"""
        + account_line,
        encoding="utf-8",
    )
    return runs, ibkr


def test_run_enabled_defaults_true_and_environment_remains_per_run() -> None:
    run = RunConfig(
        run_id="future-live",
        universe="NASDAQ",
        strategy="TEST_EXECUTION",
        environment=Environment.LIVE,
    )

    assert run.enabled is True
    assert run.environment is Environment.LIVE


def test_run_window_has_explicit_exchange_calendar() -> None:
    window = RunWindow(
        start=time(9, 30),
        end=time(16),
        timezone="America/New_York",
        calendar="XNYS",
    )

    assert window.timezone == "America/New_York"
    assert window.calendar == "XNYS"


def test_production_runtime_requires_explicit_paper_account(tmp_path: Path) -> None:
    runs, ibkr = _write_runtime_configs(tmp_path, expected_account=False)

    with pytest.raises(ValueError, match="requires PAPER expected_account"):
        build_paper_runtime(
            runs_config_path=runs,
            ibkr_config_path=ibkr,
            database_path=tmp_path / "runtime.sqlite3",
        )


def test_manual_smoke_fails_closed_before_connect_without_expected_account(
    tmp_path: Path,
) -> None:
    runs, ibkr = _write_runtime_configs(tmp_path, expected_account=False)

    result = CliRunner().invoke(
        app,
        [
            "stage8-paper-smoke",
            "--runs-config",
            str(runs),
            "--ibkr-config",
            str(ibkr),
            "--database",
            str(tmp_path / "runtime.sqlite3"),
        ],
    )

    assert result.exit_code == 1
    assert "requires PAPER" in result.output
    assert "expected_account" in result.output
