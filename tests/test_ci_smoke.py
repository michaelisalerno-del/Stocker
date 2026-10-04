from pathlib import Path


def test_github_actions_ci_workflow_exists() -> None:
    workflow = Path(".github/workflows/ci.yml")

    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    installs = [line for line in text.splitlines() if "uv sync" in line]
    assert installs and all("--locked" in line for line in installs)
    assert "fail-fast: false" in text
    assert "bash scripts/check.sh" in text
    assert "continue-on-error" not in text
