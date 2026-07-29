from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/audit_causal_state_pattern_discovery_v1.py"
SPEC = importlib.util.spec_from_file_location("causal_state_pattern_audit", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_auditor_is_independent_of_runner_imports() -> None:
    source = SOURCE.read_text()
    assert "from work.run_causal_state_pattern_discovery_v1" not in source
    assert "import work.run_causal_state_pattern_discovery_v1" not in source


def test_auditor_binds_frozen_contract_runner_input_and_manifest() -> None:
    assert MODULE.sha256(MODULE.CONTRACT) == MODULE.EXPECTED["contract"]
    assert MODULE.sha256(MODULE.RUNNER) == MODULE.EXPECTED["runner"]
    assert MODULE.sha256(MODULE.ANCHOR) == MODULE.EXPECTED["anchor"]
    assert MODULE.sha256(MODULE.ROOT / "candidate_manifest.csv") == MODULE.EXPECTED["manifest"]


def test_auditor_has_exact_safety_labels() -> None:
    source = SOURCE.read_text()
    assert "research_only: true" in source
    assert "live_ordering_enabled: false" in source
    assert "order_placement: disabled" in source

