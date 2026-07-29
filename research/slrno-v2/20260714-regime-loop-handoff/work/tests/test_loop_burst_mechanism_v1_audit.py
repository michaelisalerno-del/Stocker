from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
AUDIT_PATH = WORKSPACE / "work/audit_loop_burst_mechanism_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("loop_burst_v1_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auditor_does_not_import_the_production_runner() -> None:
    source = AUDIT_PATH.read_text()
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("work.run_") for name in imports)
    assert "import work.run_loop_burst" not in source
    assert "from work.run_loop_burst" not in source


def test_auditor_pins_contract_runner_and_all_sources() -> None:
    module = load_module()
    paths = {
        "contract": module.CONTRACT,
        "runner": module.RUNNER,
        "oof_source": module.OOF_SOURCE,
        "oof_audit": module.OOF_AUDIT,
        "run_source": module.RUN_SOURCE,
        "cycle_source": module.CYCLE_SOURCE,
        "factor_contract": module.FACTOR_CONTRACT,
        "factor_runner": module.FACTOR_RUNNER,
    }
    assert {name: module.sha256(path) for name, path in paths.items()} == module.EXPECTED_HASHES


def test_auditor_has_exact_research_safety_labels() -> None:
    module = load_module()
    contract = json.loads(module.CONTRACT.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    source = AUDIT_PATH.read_text()
    assert "research_only: true" in source
    assert "live_ordering_enabled: false" in source
    assert "order_placement: disabled" in source


def test_independent_design_width_and_loss_are_exact() -> None:
    module = load_module()
    frame = pd.DataFrame(
        {
            **{feature: [0.0, 1.0] for feature in module.PHASE_FEATURES},
            "orientation_index": [0, 25],
        }
    )
    matrix, penalties = module.design(
        frame, np.zeros(5), np.ones(5), "qburst_orientation"
    )
    assert matrix.shape == (2, 156)
    assert penalties.shape == (156,)
    assert np.all(penalties[6:31] == 4.0)
    assert np.all(penalties[31:] == 8.0)
    log_loss, brier = module.losses(
        np.asarray([0, 1]), np.asarray([0.25, 0.75])
    )
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)


def test_frame_comparison_handles_nan_infinity_and_csv_types() -> None:
    module = load_module()
    calculated = pd.DataFrame(
        {"key": [1, 2], "label": ["", "x"], "value": [np.nan, np.inf]}
    )
    stored = pd.DataFrame(
        {"key": [1, 2], "label": [np.nan, "x"], "value": [np.nan, np.inf]}
    )
    passed, maximum, errors = module.compare_frames(calculated, stored, ["key"])
    assert passed is False
    assert errors == 1
    assert maximum == 0.0


def test_bootstrap_sign_flip_and_holm_are_deterministic() -> None:
    module = load_module()
    values = np.linspace(-0.02, 0.01, 20)
    assert module.bootstrap(values, 13) == module.bootstrap(values, 13)
    assert module.sign_flip(values, 13) == module.sign_flip(values, 13)
    frame = pd.DataFrame(
        {
            "baseline": ["a", "b", "a", "b"],
            "endpoint": ["log_loss", "log_loss", "brier", "brier"],
            "p_value": [0.01, 0.04, 0.03, 0.02],
        }
    )
    adjusted = module.holm(frame)
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 4

