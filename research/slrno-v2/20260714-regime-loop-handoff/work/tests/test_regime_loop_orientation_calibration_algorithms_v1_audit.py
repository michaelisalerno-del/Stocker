from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[2]
AUDIT_PATH = (
    WORKSPACE
    / "work/audit_regime_loop_orientation_calibration_algorithms_v1.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("orientation_algorithm_v1_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auditor_does_not_import_the_scoring_runner() -> None:
    source = AUDIT_PATH.read_text()
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("work.run_") for name in imports)
    assert "import work.run_regime_loop_orientation" not in source
    assert "from work.run_regime_loop_orientation" not in source


def test_auditor_binds_every_frozen_input_hash() -> None:
    module = load_audit_module()
    assert module.sha256(module.CONTRACT) == module.EXPECTED["contract"]
    assert module.sha256(module.RUNNER) == module.EXPECTED["runner"]
    assert module.sha256(module.SOURCE) == module.EXPECTED["source"]
    assert module.sha256(module.SOURCE_AUDIT) == module.EXPECTED["source_audit"]


def test_contract_and_auditor_have_exact_research_safety_labels() -> None:
    module = load_audit_module()
    contract = json.loads(module.CONTRACT.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    source = AUDIT_PATH.read_text()
    assert "research_only: true" in source
    assert "live_ordering_enabled: false" in source
    assert "order_placement: disabled" in source


def test_orientation_feature_blocks_have_frozen_widths_and_scales() -> None:
    module = load_audit_module()
    frame = pd.DataFrame(
        {
            module.source_p("baseline", "absolute_return_bps", 6, "p75"): [0.2, 0.4],
            module.source_p("raw_full_link", "absolute_return_bps", 6, "p75"): [0.3, 0.5],
            "orientation_index": [0, 43],
            "orientation_clock_index": [0, 131],
        }
    )
    global_values, residual = module.global_features(
        frame, "absolute_return_bps", 6, "p75"
    )
    scaler = StandardScaler().fit(global_values)
    without_clock = module.orientation_features(
        frame, "absolute_return_bps", 6, "p75", scaler, 44, None
    ).toarray()
    with_clock = module.orientation_features(
        frame, "absolute_return_bps", 6, "p75", scaler, 44, 132
    ).toarray()
    assert without_clock.shape == (2, 90)
    assert with_clock.shape == (2, 354)
    assert np.allclose(without_clock[:, 2:46].sum(axis=1), 0.25)
    assert np.allclose(without_clock[:, 46:90].sum(axis=1), residual * 0.125)
    assert np.allclose(with_clock[:, 90:222].sum(axis=1), 0.125)
    assert np.allclose(with_clock[:, 222:354].sum(axis=1), residual * 0.0625)


def test_loss_calibration_bootstrap_and_holm_are_deterministic() -> None:
    module = load_audit_module()
    y = np.asarray([0, 0, 1, 1])
    p = np.asarray([0.05, 0.15, 0.85, 0.95])
    log_loss, brier = module.losses(y, p)
    assert np.isfinite(log_loss).all()
    assert np.allclose(brier, np.asarray([0.0025, 0.0225, 0.0225, 0.0025]))
    ece, maximum, bins = module.calibration(y, p, np.ones(4), minimum=1)
    assert np.isclose(ece, 0.1)
    assert np.isclose(maximum, 0.15)
    assert bins == 4
    daily = np.linspace(-0.1, 0.1, 20)
    assert module.bootstrap(daily, 17) == module.bootstrap(daily, 17)
    assert module.sign_flip(daily, 17) == module.sign_flip(daily, 17)
    adjusted = module.holm(pd.DataFrame({"p_value": [0.01, 0.04, 0.03]}), [])
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 3

