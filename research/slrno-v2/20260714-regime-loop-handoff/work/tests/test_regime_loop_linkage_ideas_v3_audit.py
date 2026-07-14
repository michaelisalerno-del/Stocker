from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/audit_regime_loop_linkage_ideas_v3.py"
SPEC = importlib.util.spec_from_file_location("regime_loop_linkage_audit", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_auditor_does_not_import_any_linkage_runner() -> None:
    source = SOURCE.read_text()
    assert "import work.run_regime_loop_linkage" not in source
    assert "from work.run_regime_loop_linkage" not in source


def test_auditor_binds_contract_runner_and_source_hashes() -> None:
    assert MODULE.sha256(MODULE.CONTRACT) == MODULE.EXPECTED["contract"]
    assert MODULE.sha256(MODULE.RUNNER) == MODULE.EXPECTED["runner"]
    assert MODULE.sha256(MODULE.FACTOR) == MODULE.EXPECTED["factor"]
    assert MODULE.sha256(MODULE.QUALITY) == MODULE.EXPECTED["quality"]


def test_auditor_fixed_blend_and_loss_are_exact() -> None:
    base = np.asarray([0.1, 0.5])
    full = np.asarray([0.3, 0.8])
    blended = MODULE.blend(base, full)
    assert np.all(blended > base)
    assert np.all(blended < full)
    ll, brier = MODULE.losses(np.asarray([0, 1]), np.asarray([0.25, 0.75]))
    assert np.allclose(ll, -np.log(0.75))
    assert np.allclose(brier, 0.0625)


def test_auditor_holm_handles_global_family() -> None:
    frame = pd.DataFrame({"p_value": [0.01, 0.04, 0.03]})
    adjusted = MODULE.holm(frame, [])
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 3


def test_auditor_has_exact_safety_labels() -> None:
    source = SOURCE.read_text()
    assert "research_only: true" in source
    assert "live_ordering_enabled: false" in source
    assert "order_placement: disabled" in source

