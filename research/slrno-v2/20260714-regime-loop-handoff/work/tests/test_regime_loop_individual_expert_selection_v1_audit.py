from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
AUDIT_PATH = WORKSPACE / "work/audit_regime_loop_individual_expert_selection_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("individual_expert_v1_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auditor_does_not_import_the_production_runner() -> None:
    source = AUDIT_PATH.read_text()
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("work.run_") for name in imported)
    assert "import work.run_regime_loop_individual" not in source
    assert "from work.run_regime_loop_individual" not in source


def test_auditor_pins_all_contract_runner_and_source_hashes() -> None:
    module = load_module()
    paths = {
        "contract": module.CONTRACT,
        "runner": module.RUNNER,
        "source": module.SOURCE,
        "source_audit": module.SOURCE_AUDIT,
        "source_contract": module.SOURCE_CONTRACT,
        "source_runner": module.SOURCE_RUNNER,
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


def test_independent_loss_tie_break_and_key_functions_are_exact() -> None:
    module = load_module()
    y = np.asarray([0, 1])
    log_loss, brier = module.binary_losses(y, np.asarray([0.25, 0.75]))
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)
    equal = {expert: 1.0 for expert in module.EXPERTS}
    assert module.choose(equal) == "baseline"
    equal["dependency_stack"] = 0.8
    assert module.choose(equal) == "dependency_stack"
    assert module.key_text(("cycle_id", "current_state"), ("cycle_13", 5)) == (
        "cycle_id=cycle_13|current_state=5"
    )


def test_frame_comparison_handles_csv_empty_strings_and_exact_numbers() -> None:
    module = load_module()
    calculated = pd.DataFrame(
        {"key": [1, 2], "label": ["", "x"], "value": [0.5, np.inf]}
    )
    stored = pd.DataFrame(
        {"key": [1, 2], "label": [np.nan, "x"], "value": [0.5, np.inf]}
    )
    passed, maximum, errors = module.frame_comparison(calculated, stored, ["key"])
    assert passed is True
    assert maximum == 0.0
    assert errors == 0


def test_bootstrap_sign_flip_and_holm_are_deterministic() -> None:
    module = load_module()
    values = np.linspace(-0.02, 0.01, 20)
    assert module.bootstrap(values, 31) == module.bootstrap(values, 31)
    assert module.sign_flip(values, 31) == module.sign_flip(values, 31)
    frame = pd.DataFrame(
        {
            "selector": ["a", "b", "c"],
            "comparison": ["raw", "raw", "raw"],
            "endpoint": ["log_loss", "log_loss", "log_loss"],
            "p_value": [0.01, 0.04, 0.03],
        }
    )
    adjusted = module.holm(frame, ["comparison", "endpoint"])
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 3

