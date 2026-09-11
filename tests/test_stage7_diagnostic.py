from pathlib import Path

import pytest
from click import unstyle
from typer import rich_utils
from typer.testing import CliRunner

from stocker_core.cli import app


def _base_arguments() -> list[str]:
    return [
        "stage7-paper-diagnostic",
        "--signal-id",
        "manual-diagnostic-1",
        "--symbol",
        "AAPL",
        "--entry-reference",
        "100",
        "--m-price",
        "2",
        "--risk-per-trade",
        "0.0001",
    ]


@pytest.mark.parametrize("force_color", [False, True])
def test_diagnostic_never_transmits_without_explicit_confirmation(monkeypatch, force_color) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", force_color)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "standard" if force_color else None)
    result = CliRunner().invoke(app, _base_arguments(), color=force_color)

    assert result.exit_code != 0
    output = unstyle(result.output)
    assert "--confirm-paper-order is required" in output
    assert "no order was transmitted" in output


def test_diagnostic_rejects_live_run_before_connecting(tmp_path: Path) -> None:
    run_path = tmp_path / "live-run.yaml"
    run_path.write_text(
        """
run_id: live-run
universe: NASDAQ
strategy: TEST_EXECUTION
environment: LIVE
""",
        encoding="utf-8",
    )
    ibkr_path = tmp_path / "ibkr.yaml"
    ibkr_path.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
  expected_account: DU123456
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            *_base_arguments(),
            "--confirm-paper-order",
            "--run-config",
            str(run_path),
            "--ibkr-config",
            str(ibkr_path),
        ],
    )

    assert result.exit_code != 0
    assert "LIVE_EXECUTION_DISABLED" in result.output


def test_diagnostic_requires_an_explicit_expected_paper_account(tmp_path: Path) -> None:
    run_path = tmp_path / "paper-run.yaml"
    run_path.write_text(
        """
run_id: paper-run
universe: NASDAQ
strategy: TEST_EXECUTION
environment: PAPER
""",
        encoding="utf-8",
    )
    ibkr_path = tmp_path / "ibkr.yaml"
    ibkr_path.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            *_base_arguments(),
            "--confirm-paper-order",
            "--run-config",
            str(run_path),
            "--ibkr-config",
            str(ibkr_path),
        ],
    )

    assert result.exit_code != 0
    assert "requires expected_account" in result.output
