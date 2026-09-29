import hashlib
import json
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
    assert "EODHD_API_TOKEN" not in text


def test_external_vendor_smoke_launchers_are_parked_with_recovery_evidence() -> None:
    root = Path("research/operational-history/parked-ibkr-runtime")
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ("scripts/smoke_eodhd_local.sh", "scripts/research_smoke_local.sh"):
        assert not Path(name).exists()
        archived = root / (name.replace("/", "__") + ".txt")
        row = next(r for r in manifest["files"] if r.get("original_path") == name)
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == row["sha256"]
