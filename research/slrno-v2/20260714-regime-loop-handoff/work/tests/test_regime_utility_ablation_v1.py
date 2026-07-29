from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER = WORKSPACE / "work/run_regime_utility_ablation_v1.py"
ARTIFACT = Path("/private/tmp/stocker_regime_utility_ablation_v1_20260711")


def load_runner():
    spec = importlib.util.spec_from_file_location("regime_utility_ablation_v1", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_hash_and_research_boundary_are_frozen() -> None:
    module = load_runner()
    contract, pre_score = module.load_contract_and_pre_score()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["periods"]["2025_permitted"] is False
    assert contract["periods"]["2023_permitted"] is False
    assert contract["periods"]["2026_permitted"] is False
    assert pre_score["later_period_outcomes_permitted"] is False


def test_history_tokens_and_overlapping_cycle_rotations_are_exact() -> None:
    module = load_runner()
    tokens = module.history_tokens(
        np.asarray([8, 1, 1]),
        np.asarray([8, 2, 3]),
        np.asarray([0, 1, 1]),
    )
    assert tokens.tolist() == [640, 89, 97]
    core = (1, 2, 1, 3)
    assert module.oriented_paths(core, 1) == [
        (1, 2, 1, 3, 1),
        (1, 3, 1, 2, 1),
    ]


def test_bootstrap_is_deterministic_and_negative_values_remain_negative() -> None:
    module = load_runner()
    values = np.linspace(-0.02, -0.001, 128)
    first = module.moving_block_interval(values, 17, draws=250)
    second = module.moving_block_interval(values, 17, draws=250)
    assert first == second
    assert first[2] < 0.0


def test_artifact_cohort_layers_and_forbidden_fields() -> None:
    predictions = pd.read_parquet(ARTIFACT / "oof_predictions_2024.parquet")
    assert len(predictions) == 34169
    assert predictions["anchor_id"].is_unique
    assert predictions["month_key"].drop_duplicates().tolist() == [
        f"2024-{month:02d}" for month in range(7, 13)
    ]
    forbidden = (
        "direction",
        "signed_return",
        "pnl",
        "profit",
        "order",
        "broker",
        "strategy",
        "deployment",
    )
    assert not [
        column
        for column in predictions.columns
        if any(token in column.lower() for token in forbidden)
    ]
    for layer in ("context", "state", "history", "departure", "loops", "burst"):
        assert f"prediction__{layer}__absolute_return_bps__h6" in predictions
        assert f"prediction__{layer}__future_range_bps__h24" in predictions


def test_predeclared_decision_and_independent_audit_pass() -> None:
    decision = json.loads((ARTIFACT / "decision.json").read_text())
    audit = json.loads((ARTIFACT / "independent_audit.json").read_text())
    assert decision["regime_reliably_useful"] is True
    retained = {
        name: value["checks"]["retained"]
        for name, value in decision["incremental_layer_decisions"].items()
    }
    assert retained == {
        "state_vs_context": True,
        "history_vs_state": True,
        "departure_vs_history": False,
        "loops_vs_departure": False,
        "burst_vs_loops": False,
    }
    assert decision["final_stack_vs_context"]["magnitude_checks"]["pass"] is True
    assert audit["all_passed"] is True
    assert audit["passed"] == 23
    assert audit["failed"] == 0
