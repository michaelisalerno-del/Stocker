from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/audit_exact_oriented_route_quality_screen_v1.py"
RUNNER = ROOT / "work/run_exact_oriented_route_quality_screen_v1.py"
ARTIFACT = Path(
    "/private/tmp/stocker_exact_oriented_route_quality_screen_v1_20260711"
)
SPEC = importlib.util.spec_from_file_location("exact_route_auditor", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_auditor_has_no_production_imports() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any(name.startswith("work.") for name in imported)


def test_auditor_self_test_passes() -> None:
    assert MODULE.self_test()["all_passed"] is True


def test_passing_audit_binds_frozen_sources() -> None:
    result = json.loads((ARTIFACT / "independent_audit.json").read_text())
    assert result["all_passed"] is True
    assert result["check_count"] == 17
    assert result["contract_sha256"] == MODULE.CONTRACT_SHA256
    assert result["runner_sha256"] == digest(RUNNER)
    assert result["auditor_source_sha256"] == digest(SOURCE)
    assert result["maximum_qexact_probability_replay_error"] == 0.0


def test_audit_confirms_zero_candidate_without_refinement() -> None:
    result = json.loads((ARTIFACT / "independent_audit.json").read_text())
    assert result["candidate_count"] == 0
    assert result["rejection_verified"] is True
    assert result["further_refinement_authorized"] is False


def test_audit_confirms_safety_and_phase_isolation() -> None:
    result = json.loads((ARTIFACT / "independent_audit.json").read_text())
    assert result["later_period_paths_resolved"] is False
    assert result["later_period_rows_read"] is False
    assert result["shadow_tree_read"] is False
    assert result["shadow_tree_written"] is False
    assert result["historical_volume_used"] is False
    assert result["research_only"] is True
    assert result["live_ordering_enabled"] is False
    assert result["order_placement"] == "disabled"

