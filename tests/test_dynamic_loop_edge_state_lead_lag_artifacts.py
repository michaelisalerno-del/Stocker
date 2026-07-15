from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
ARTIFACT_ROOT = WORK / "artifacts/20260715-dynamic-loop-edge-state-lead-lag-v1"


def test_primary_and_exact_rerun_manifests_are_byte_identical() -> None:
    primary = (ARTIFACT_ROOT / "primary/artifact_manifest.json").read_bytes()
    exact = (ARTIFACT_ROOT / "exact_rerun/artifact_manifest.json").read_bytes()

    assert primary == exact
    manifest = json.loads(primary)
    assert manifest["research_only"] is True
    assert manifest["execution_enabled"] is False
    assert len(manifest["files"]) == 30


def test_independent_audit_passes_identity_and_safety_checks() -> None:
    primary = json.loads(
        (ARTIFACT_ROOT / "primary/independent_audit.json").read_text(encoding="utf-8")
    )
    exact = json.loads(
        (ARTIFACT_ROOT / "exact_rerun/independent_audit.json").read_text(encoding="utf-8")
    )
    checks = {item["name"]: item["passed"] for item in primary["checks"]}

    assert primary == exact
    assert primary["passed"] is True
    assert checks["primary_exact_machine_readable_and_plot_identity"] is True
    assert checks["no_runtime_or_execution_paths_modified"] is True


def test_no_broker_order_position_or_deployment_path_changed_since_frozen_v2() -> None:
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "ca3537a0f337097a9a75abf87ae4bf419fae6a5d",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    forbidden = (
        "packages/stocker_execution/",
        "apps/",
        "deployment/",
        "infra/",
    )

    assert not any(path.startswith(forbidden) for path in changed)
