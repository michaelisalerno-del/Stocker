from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_exact_oriented_route_quality_screen_v1.py"
SPEC = importlib.util.spec_from_file_location("exact_route_screen", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen_and_research_only() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["period_and_phase_lock"]["fit_and_evaluate"] == "2024_only"
    assert contract["period_and_phase_lock"]["later_period_paths_permitted"] is False
    assert contract["decision_and_stop_rules"]["later_period_scoring"] is False


def test_real_frozen_cycles_expand_to_45_exact_routes() -> None:
    cycles = pd.read_csv(MODULE.CYCLES)
    manifest = MODULE.build_exact_route_manifest(cycles)
    assert len(manifest) == 45
    assert manifest["route_id"].nunique() == 45
    split = manifest.loc[manifest["route_kind"].eq("split_exact")]
    assert split["route_id"].tolist() == [
        "cycle_15@state_1__path_1_2_1_3_1",
        "cycle_15@state_1__path_1_3_1_2_1",
    ]
    assert not split["structural_probability_available"].any()


def test_path_occurrence_is_exact_and_directional() -> None:
    frame = pd.DataFrame(
        {
            "future_state_1": [2, 3, 2],
            "future_state_2": [1, 1, 1],
            "future_state_3": [3, 2, 3],
            "future_state_4": [1, 1, 0],
        }
    )
    first = MODULE.path_occurrence(frame, (1, 2, 1, 3, 1))
    second = MODULE.path_occurrence(frame, (1, 3, 1, 2, 1))
    assert first.tolist() == [True, False, False]
    assert second.tolist() == [False, True, False]
    assert not np.any(first & second)


def test_qexact_feature_width_and_scale() -> None:
    context = sparse.csr_matrix(np.ones((3, MODULE.CONTEXT_WIDTH)))
    matrix = MODULE.qexact_features(context, np.asarray([0, 12, 44]))
    assert matrix.shape == (3, MODULE.QEXACT_WIDTH)
    route = matrix[:, MODULE.CONTEXT_WIDTH :].toarray()
    assert np.allclose(route.sum(axis=1), MODULE.ROUTE_SCALE)


def test_support_gate_requires_both_quarters() -> None:
    contract = json.loads(MODULE.CONTRACT.read_text())
    frame = pd.DataFrame(
        {
            "exact_route_occurs": np.ones(250, dtype=np.int8),
            "symbol_norm": [f"S{index % 15:02d}" for index in range(250)],
            "quarter": ["2024_q3"] * 125 + ["2024_q4"] * 125,
        }
    )
    # Compatible-row support is deliberately 1,000, so replicate negatives.
    negative = pd.DataFrame(
        {
            "exact_route_occurs": np.zeros(750, dtype=np.int8),
            "symbol_norm": ["S00"] * 750,
            "quarter": ["2024_q3"] * 750,
        }
    )
    assert MODULE.support_gate(pd.concat([frame, negative]), contract)["pass"] is True
    one_quarter = pd.concat([frame.assign(quarter="2024_q3"), negative])
    assert MODULE.support_gate(one_quarter, contract)["pass"] is False


def test_holm_adjustment_is_monotone_within_tier() -> None:
    rows = pd.DataFrame(
        {
            "route_id": ["a", "b", "c"],
            "target": ["x"] * 3,
            "horizon": [6] * 3,
            "tier": ["p75"] * 3,
            "p_value": [0.001, 0.02, 0.04],
        }
    )
    result = MODULE.holm_table(rows).sort_values("p_value")
    adjusted = result["holm_adjusted_p"].to_numpy(float)
    assert np.all(np.diff(adjusted) >= -1e-15)
    assert result["family_size"].eq(3).all()


def test_sign_flip_is_deterministic() -> None:
    values = np.asarray([-0.2, -0.1, -0.3, -0.1, -0.2] * 3)
    left = MODULE.sign_flip_p_value(values, 17, draws=999)
    right = MODULE.sign_flip_p_value(values, 17, draws=999)
    assert left == right
    assert 0.0 < left <= 1.0


def test_runner_has_no_later_or_shadow_input_constants() -> None:
    source = SOURCE.read_text()
    assert "ANCHOR_2025" not in source
    assert "ANCHOR_2023" not in source
    assert "quality_scoring_2025" not in source
    assert "quality_scoring_2023" not in source
    assert "prediction_ledger" not in source

