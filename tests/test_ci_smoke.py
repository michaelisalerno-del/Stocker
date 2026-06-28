import os
from pathlib import Path


def test_github_actions_ci_workflow_exists() -> None:
    workflow = Path(".github/workflows/ci.yml")

    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "uv sync --all-groups" in text
    assert "uv run pytest" in text
    assert "EODHD_API_TOKEN" not in text


def test_eodhd_local_smoke_script_is_safe_and_executable() -> None:
    script = Path("scripts/smoke_eodhd_local.sh")

    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "--dry-run" in text
    assert "fetch-eodhd-eod" in text
    assert "fetch-eodhd-intraday" in text
    assert "EODHD_API_TOKEN" in text
    assert "data_smoke" in text


def test_research_smoke_script_is_safe_and_executable() -> None:
    script = Path("scripts/research_smoke_local.sh")

    assert script.exists()
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "EODHD_API_TOKEN" in text
    assert "universes/manual/us_test_5.yaml" in text
    assert "moving_average_momentum.yaml" in text
    assert "research run-universe" in text
    assert "RESEARCH_SMOKE_MAX_SYMBOLS:-5" in text
    assert '--max-symbols "$MAX_SYMBOLS"' in text
    assert "classification_counts" in text
