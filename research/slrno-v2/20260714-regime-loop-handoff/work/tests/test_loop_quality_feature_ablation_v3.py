from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_loop_quality_feature_ablation_v3.py"
)
SPEC = importlib.util.spec_from_file_location("v3_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


def test_contract_hash_authorization_and_safety_are_frozen() -> None:
    assert v3.sha256(v3.CONTRACT) == v3.CONTRACT_SHA256
    contract = json.loads(v3.CONTRACT.read_text())
    assert contract["execution_authorization"]["authorized"] is True
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"


def test_unique_support_semantics_pass_exact_frozen_oof_cohort() -> None:
    _, mapping, _ = v3.load_cycles_and_mapping()
    oof = v3.prepare_oof(mapping)
    support = v3.v3_support(oof)
    assert support["total_effective_weight"] == 14167.0
    assert support["quarter_effective_weight"] == {
        "2024_q3": 7635.0,
        "2024_q4": 6532.0,
    }
    assert support["sessions"] == 128
    assert support["stocks"] == 22
    assert support["minimum_stock_effective_weight"] == 93.0
    assert support["realized_rows"] == 15584
    assert support["realized_rows_is_independent_support_gate"] is False
    assert support["support_pass"] is True


def test_topology_rotation_and_column_order_are_exact() -> None:
    assert len(v3.TOPOLOGY_COLUMNS) == 63
    assert v3.compatible_rotations((0, 1, 0, 1), 0) == (
        (0, 1, 0, 1, 0),
    )
    assert len(v3.compatible_rotations((1, 2, 1, 3), 1)) == 2
    centroids = np.arange(8 * 14, dtype=float).reshape(8, 14)
    vector, _ = v3.topology_vector((1, 2, 1, 3), 1, centroids)
    assert len(vector) == 63
    assert np.isclose(vector[:8].sum(), 1.0)
    assert np.isclose(vector[8:16].sum(), 1.0)
    assert vector[61] == 1.0


def test_three_interior_feature_widths_and_scales() -> None:
    frame = pd.DataFrame(
        {
            "cycle_index": [0, 14],
            "state": [1, 1],
            **{column: [0.0, 0.0] for column in v3.TOPOLOGY_COLUMNS},
        }
    )
    frame.loc[:, v3.NEXT_COLUMNS[0]] = 1.0
    frame.loc[:, v3.COMPOSITION_COLUMNS[0]] = 1.0
    frame.loc[:, "length_is_2"] = 1.0
    frame.loc[:, v3.NEXT_CENTROID_COLUMNS[0]] = 2.0
    context = sparse.csr_matrix(np.ones((2, 17)))
    matrices = v3.feature_matrices(frame, context)
    assert matrices["qroute_topology"].shape == (2, 80)
    assert matrices["qcycle_main"].shape == (2, 37)
    assert matrices["qcycle_state"].shape == (2, 197)
    assert matrices["qroute_topology"][0, 17 + 19] == 1.0


def test_probability_outputs_are_ordered_and_joint_chain_exact() -> None:
    frame = pd.DataFrame({"loop_probability": [0.2, 0.8]})
    probability = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.25, 0.65]])
    v3.add_probability_columns(
        frame, "qroute_topology", "absolute_return_bps", 6, probability
    )
    assert np.allclose(frame["qroute_topology__absolute_return_bps__h6__p75"], [0.5, 0.9])
    assert np.allclose(frame["qroute_topology__absolute_return_bps__h6__p90"], [0.2, 0.65])
    assert np.allclose(
        frame["joint__qroute_topology__absolute_return_bps__h6__p75"],
        [0.1, 0.72],
    )


def test_bootstrap_and_seed_mapping_are_deterministic() -> None:
    values = np.linspace(-0.02, 0.01, 30)
    assert v3.moving_block_interval(values, 123, draws=100) == v3.moving_block_interval(
        values, 123, draws=100
    )
    assert len(v3.COMPARISONS) == 5


def test_validate_only_reads_no_later_panel() -> None:
    result = v3.validate_only()
    assert result["status"] == "validated_without_fit"
    assert result["support"]["support_pass"] is True
    assert result["oof_rows"] == 216438
    assert result["training_rows"] == 32677


def test_scoring_lock_requires_fit_and_independent_audit(tmp_path: Path) -> None:
    original = v3.OUT
    try:
        v3.OUT = tmp_path
        with pytest.raises(FileNotFoundError):
            v3.validate_fit_and_pre_score_lock()
    finally:
        v3.OUT = original


def test_source_has_no_live_shadow_path_or_order_surface() -> None:
    source = MODULE_PATH.read_text()
    assert "shadow_validation" not in source
    assert "place_order" not in source
    assert "broker" not in source.lower()
