from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_loop_quality_feature_ablation_v2.py"
)
SPEC = importlib.util.spec_from_file_location("v2_support_stop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)


def test_frozen_contract_hash_and_safety_labels_are_exact() -> None:
    assert v2.sha256(v2.CONTRACT) == v2.CONTRACT_SHA256
    contract = json.loads(v2.CONTRACT.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["model_run_authorized_by_this_file"] is False
    assert (
        contract["calibration_bins_and_support"]
        ["pooled_oof_minimum_effective_conditional_weight"]
        == 20000
    )


def test_rotation_construction_is_causal_deduplicated_and_ambiguity_preserving() -> None:
    assert v2.compatible_rotations((0, 1, 0, 1), 0) == (
        (0, 1, 0, 1, 0),
    )
    assert v2.compatible_rotations((1, 2, 1, 3), 1) == (
        (1, 2, 1, 3, 1),
        (1, 3, 1, 2, 1),
    )
    assert v2.compatible_rotations((1, 2), 7) == ()


def test_topology_vector_has_frozen_order_width_and_unscaled_centroids() -> None:
    centroids = np.arange(v2.K * v2.CENTROID_WIDTH, dtype=float).reshape(
        v2.K, v2.CENTROID_WIDTH
    )
    vector, routes = v2.topology_vector((1, 2, 1, 3), 1, centroids)
    assert len(vector) == v2.TOPOLOGY_WIDTH == len(v2.TOPOLOGY_COLUMNS)
    assert len(routes) == 2
    assert np.isclose(vector[:8].sum(), 1.0)
    assert np.isclose(vector[8:16].sum(), 1.0)
    assert vector[16:19].tolist() == [0.0, 0.0, 1.0]
    expected_next = 0.5 * centroids[2] + 0.5 * centroids[3]
    assert np.allclose(vector[19:33], expected_next)
    assert vector[61] == 1.0
    assert vector[62] > 0.0


def test_support_uses_unique_weight_and_rejects_twelve_cell_repetition() -> None:
    design = pd.DataFrame(
        {
            "loop_occurs": [1, 1, 1, 0],
            "conditional_weight": [0.5, 0.5, 1.0, 0.0],
            "anchor_id": [1, 1, 2, 3],
            "symbol_norm": ["AAA", "AAA", "BBB", "CCC"],
            "quarter": ["2024_q3", "2024_q3", "2024_q4", "2024_q4"],
        }
    )
    table, summary = v2.support_audit(design)
    assert summary["pooled_oof_unique_effective_weight"] == 2.0
    assert summary["twelve_cell_repeated_weight"] == 24.0
    assert summary["double_counted_cell_weight_accepted"] is False
    assert summary["support_pass"] is False
    repeated = table.loc[
        table["measure"].eq("twelve_cell_repeated_weight_diagnostic_only")
    ].iloc[0]
    assert bool(repeated["pass"]) is False


def test_output_root_cannot_overlap_inputs_or_leave_private_tmp() -> None:
    with pytest.raises(ValueError):
        v2.validate_output_root(v2.PARENT_ROOT)
    with pytest.raises(ValueError):
        v2.validate_output_root(Path("/tmp/v2-output"))


def test_source_contains_no_model_fit_later_panel_or_live_shadow_runtime() -> None:
    source = MODULE_PATH.read_text()
    forbidden = (
        "sklearn",
        "LogisticRegression",
        ".fit(",
        "quality_scoring_2025",
        "quality_scoring_2023",
        "anchor_panel_2025",
        "anchor_panel_2023",
        "anchor_panel_2026",
        "shadow_validation",
    )
    assert not {token for token in forbidden if token in source}


@pytest.mark.skipif(
    not v2.PARENT_OOF.is_file(),
    reason="ephemeral frozen 2024 OOF artifacts are unavailable",
)
def test_end_to_end_support_stop_has_no_predictions_or_parent_change() -> None:
    output = Path(
        "/private/tmp/stocker_loop_quality_feature_ablation_v2_pytest"
    )
    parent_before = v2.hashes(v2.PARENT_DECISION_AND_SAVED_SNAPSHOT_FILES)
    summary = v2.run(output)
    parent_after = v2.hashes(v2.PARENT_DECISION_AND_SAVED_SNAPSHOT_FILES)

    assert parent_before == parent_after
    assert summary["status"] == "support_stop_verified"
    assert summary["support"]["pooled_oof_unique_effective_weight"] == 14167.0
    assert summary["support"]["pooled_oof_minimum_effective_weight"] == 20000.0
    assert summary["support"]["support_pass"] is False
    assert summary["model_fit_performed"] is False
    assert summary["prediction_generated"] is False
    assert summary["later_period_panel_read"] is False
    assert summary["parent_grade_changed"] is False

    fit_complete = json.loads((output / "fit_complete.json").read_text())
    assert fit_complete["status"] == "stopped_before_model_fit"
    assert fit_complete["later_scoring_authorized"] is False
    assert fit_complete["source_attribution_permitted"] is False
    assert not (output / "model_parameters.npz").exists()
    assert not (output / "oof_predictions_2024.parquet").exists()
    design = pd.read_parquet(
        output / "oof_design_rows_2024.parquet",
        columns=["loop_occurs", "conditional_weight"],
    )
    realised = design.loc[design["loop_occurs"].eq(1)]
    assert len(design) == 216438
    assert len(realised) == 15584
    assert realised["conditional_weight"].sum() == 14167.0
