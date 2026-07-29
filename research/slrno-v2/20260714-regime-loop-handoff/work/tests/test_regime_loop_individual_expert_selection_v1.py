from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER_PATH = WORKSPACE / "work/run_regime_loop_individual_expert_selection_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("individual_expert_v1", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_frame(module, rows: int = 600) -> pd.DataFrame:
    target = np.arange(rows) % 2
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(rows),
            "symbol_norm": np.resize(np.asarray(["A", "B"]), rows),
            "session_date": np.resize(
                np.asarray(["2024-09-03", "2024-09-04"]), rows
            ),
            "start_timestamp": np.arange(rows),
            "month": "2024-09",
            "cycle_index": 0,
            "cycle_id": "cycle_01",
            "state": 1,
            "current_state": 1,
            "entry_clock_quartile": 0,
            "inverse_compatible_weight": 1.0,
        }
    )
    for movement_target in module.TARGETS:
        for horizon in module.HORIZONS:
            for tier in module.TIERS:
                frame[module.target_column(movement_target, horizon, tier)] = target
                for expert in module.EXPERTS:
                    probability = np.full(rows, 0.5)
                    if expert == "dependency_stack":
                        probability = np.where(target == 1, 0.9, 0.1)
                    frame[
                        module.expert_column(expert, movement_target, horizon, tier)
                    ] = probability
    return frame


def test_contract_hash_and_research_labels_are_exact() -> None:
    module = load_module()
    contract = json.loads(module.CONTRACT.read_text())
    assert module.sha256(module.CONTRACT) == module.EXPECTED_HASHES["contract"]
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["population_and_causality"]["later_period_paths_permitted"] is False
    assert (
        contract["population_and_causality"][
            "prospective_shadow_read_or_write_permitted"
        ]
        is False
    )


def test_expert_and_selector_sets_match_the_frozen_contract() -> None:
    module = load_module()
    contract = json.loads(module.CONTRACT.read_text())
    assert tuple(contract["experts"]["selectable"]) == module.EXPERTS
    assert tuple(contract["experts"]["conservative_tie_priority"]) == module.TIE_PRIORITY
    assert tuple(contract["decision"]["candidate_selectors"]) == module.CANDIDATE_SELECTORS
    assert module.SELECTORS[0] == "global_best"


def test_tie_break_is_conservative_and_loss_is_exact() -> None:
    module = load_module()
    equal = {expert: 1.0 for expert in module.EXPERTS}
    assert module.choose_expert(equal) == "baseline"
    equal["raw_full_link"] = 0.9
    assert module.choose_expert(equal) == "raw_full_link"
    y = np.asarray([0, 1])
    log_loss, brier = module.binary_losses(y, np.asarray([0.25, 0.75]))
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)


def test_flat_orientation_assignment_selects_earlier_month_winner() -> None:
    module = load_module()
    selection = synthetic_frame(module)
    validation = selection.iloc[:20].copy()
    mapping, ledger = module.flat_assignments(
        selection,
        validation,
        "2024-11",
        "loop_regime_best",
        "raw_full_link",
        guarded=False,
    )
    assert mapping[("cycle_01", 1)] == "dependency_stack"
    assert len(ledger) == 1
    assert ledger[0]["supported"] is True
    assert ledger[0]["selected_expert"] == "dependency_stack"
    assert ledger[0]["candidate_improvement_vs_parent"] > module.GUARD_MARGIN


def test_guarded_assignment_falls_back_when_margin_is_not_met() -> None:
    module = load_module()
    selection = synthetic_frame(module)
    for target in module.TARGETS:
        for horizon in module.HORIZONS:
            for tier in module.TIERS:
                selection[
                    module.expert_column("dependency_stack", target, horizon, tier)
                ] = selection[
                    module.expert_column("raw_full_link", target, horizon, tier)
                ]
    validation = selection.iloc[:20].copy()
    mapping, ledger = module.flat_assignments(
        selection,
        validation,
        "2024-11",
        "guarded_loop_regime_best",
        "raw_full_link",
        guarded=True,
    )
    assert mapping[("cycle_01", 1)] == "raw_full_link"
    assert ledger[0]["decision_reason"] == "margin_fallback"


def test_hierarchy_reaches_supported_clock_child_and_copies_probabilities() -> None:
    module = load_module()
    selection = synthetic_frame(module)
    validation = selection.iloc[:20].copy()
    mapping, ledger = module.hierarchical_assignments(
        selection, validation, "2024-11", "raw_full_link"
    )
    assert mapping[("cycle_01", 1, 0)] == "dependency_stack"
    assert [row["level"] for row in ledger] == [
        "loop",
        "loop_regime",
        "loop_regime_clock",
    ]
    selected = module.map_experts(
        validation,
        ("cycle_id", "current_state", "entry_clock_quartile"),
        mapping,
        "raw_full_link",
    )
    module.copy_selected_probabilities(validation, "hierarchical_clock_best", selected)
    column = module.selector_column(
        "hierarchical_clock_best", "absolute_return_bps", 6, "p75"
    )
    expected = validation[
        module.expert_column("dependency_stack", "absolute_return_bps", 6, "p75")
    ]
    assert np.array_equal(validation[column].to_numpy(), expected.to_numpy())


def test_bootstrap_sign_flip_and_holm_are_deterministic() -> None:
    module = load_module()
    values = np.linspace(-0.02, 0.01, 20)
    assert module.bootstrap_interval(values, 42) == module.bootstrap_interval(values, 42)
    assert module.sign_flip_p_value(values, 42) == module.sign_flip_p_value(values, 42)
    frame = pd.DataFrame(
        {
            "selector": ["a", "b", "c"],
            "comparison": ["raw", "raw", "raw"],
            "endpoint": ["log_loss", "log_loss", "log_loss"],
            "p_value": [0.01, 0.04, 0.03],
        }
    )
    adjusted = module.holm_adjust(frame, ["comparison", "endpoint"])
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 3

