import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import load_run_config
from stocker_core.runs import Environment, RunConfig


def test_loads_valid_paper_run(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run_id: nasdaq_session_hard_paper
universe: NASDAQ
strategy: TEST_EXECUTION
environment: PAPER
""",
        encoding="utf-8",
    )

    assert load_run_config(config_path) == RunConfig(
        run_id="nasdaq_session_hard_paper",
        universe="NASDAQ",
        strategy="TEST_EXECUTION",
        environment=Environment.PAPER,
    )


def test_start_reports_run_without_loading_execution(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run_id: nasdaq_session_hard_paper
universe: NASDAQ
strategy: TEST_EXECUTION
environment: PAPER
""",
        encoding="utf-8",
    )
    execution_modules = [name for name in sys.modules if name.startswith("stocker_execution")]
    for name in execution_modules:
        monkeypatch.delitem(sys.modules, name)

    result = CliRunner().invoke(app, ["start", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Stocker starting" in result.stdout
    assert "Run: nasdaq_session_hard_paper" in result.stdout
    assert "Universe: NASDAQ" in result.stdout
    assert "Strategy: TEST_EXECUTION" in result.stdout
    assert "Environment: PAPER" in result.stdout
    assert "Stage 1 runtime ready" in result.stdout
    assert not any(name.startswith("stocker_execution") for name in sys.modules)


def test_live_run_can_be_represented() -> None:
    run = RunConfig(
        run_id="nasdaq_session_hard_live",
        universe="NASDAQ",
        strategy="TEST_EXECUTION",
        environment=Environment.LIVE,
    )

    assert run.environment is Environment.LIVE


def test_missing_required_run_field_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
universe: NASDAQ
strategy: TEST_EXECUTION
environment: PAPER
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="run_id"):
        load_run_config(config_path)


@pytest.mark.parametrize("universe", ["NASDAQ", "FTSE350"])
def test_run_accepts_different_universe_names(universe: str) -> None:
    run = RunConfig(
        run_id=f"{universe.lower()}_session_hard_paper",
        universe=universe,
        strategy="TEST_EXECUTION",
        environment=Environment.PAPER,
    )

    assert run.universe == universe
