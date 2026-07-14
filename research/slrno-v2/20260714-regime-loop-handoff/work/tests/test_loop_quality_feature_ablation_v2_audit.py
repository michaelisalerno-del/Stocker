from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "audit_loop_quality_feature_ablation_v2.py"
)
SPEC = importlib.util.spec_from_file_location("feature_ablation_v2_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_uniform_rotation_construction_deduplicates_repeated_pair() -> None:
    assert audit.compatible_rotations((1, 3, 1, 3), 1) == [
        (1, 3, 1, 3, 1)
    ]


def test_cycle15_state1_keeps_both_candidate_rotations_without_future_selection() -> None:
    assert audit.compatible_rotations((1, 2, 1, 3), 1) == [
        (1, 2, 1, 3, 1),
        (1, 3, 1, 2, 1),
    ]


def test_simple_route_topology_has_expected_next_state_and_composition() -> None:
    standardized, _, _ = audit.normalized_centroids()
    values, metadata = audit.topology_values((3, 6), 3, standardized)
    assert values.shape == (63,)
    assert metadata["compatible_rotation_count"] == 1
    assert metadata["next_state_distribution"] == [0, 0, 0, 0, 0, 0, 1, 0]
    assert metadata["future_route_state_composition"] == [
        0,
        0,
        0,
        0.5,
        0,
        0,
        0.5,
        0,
    ]
    assert metadata["next_state_entropy_normalized"] == 0.0
    assert values[16:19].tolist() == [1.0, 0.0, 0.0]


def test_ambiguous_route_uses_uniform_mixture_and_normalized_entropy() -> None:
    standardized, _, _ = audit.normalized_centroids()
    values, metadata = audit.topology_values((1, 2, 1, 3), 1, standardized)
    assert metadata["compatible_rotation_count"] == 2
    assert metadata["next_state_distribution"] == [0, 0, 0.5, 0.5, 0, 0, 0, 0]
    assert metadata["future_route_state_composition"] == [
        0,
        0.5,
        0.25,
        0.25,
        0,
        0,
        0,
        0,
    ]
    assert metadata["next_state_entropy_normalized"] == pytest.approx(1 / 3)
    assert values[-2] == 1.0
    assert values[-1] == pytest.approx(1 / 3)


def test_centroid_normalization_is_population_based_across_eight_states() -> None:
    standardized, _, scale = audit.normalized_centroids()
    assert standardized.shape == (8, 14)
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(
        standardized.std(axis=0, ddof=0)[scale > 0], 1.0, atol=1e-12
    )


def test_rotation_mapping_contains_44_units_and_one_ambiguous_unit() -> None:
    mapping, vectors = audit.build_rotation_mapping()
    assert len(mapping) == 44
    assert len(vectors) == 44
    assert mapping["compatible_rotation_count"].sum() == 45
    assert mapping.loc[
        mapping["compatible_rotation_count"].gt(1), "route_id"
    ].tolist() == ["cycle_15@state_1"]


def test_contract_freezes_all_five_comparisons_and_model_widths() -> None:
    contract = json.loads(audit.CONTRACT.read_text())
    assert tuple(
        contract["multiplicity_and_uncertainty"]["primary_comparison_family"]
    ) == audit.PRIMARY_COMPARISONS
    widths = {
        name: int(spec.get("width", spec.get("total_width")))
        for name, spec in contract["models"].items()
    }
    assert widths == audit.MODEL_WIDTHS
    assert contract["multiplicity_and_uncertainty"]["bootstrap_draws"] == 10000
    assert contract["multiplicity_and_uncertainty"]["block_length_sessions"] == 5


def test_unique_oof_effective_weight_deterministically_fails_support() -> None:
    result = audit.independent_support_and_reference()
    assert result["positive_rows"] == 15584
    assert result["unique_positive_anchors"] == 14167
    assert result["effective_conditional_weight"] == 14167.0
    assert result["all_positive_anchor_weights_equal_one"] is True
    assert result["support_threshold"] == 20000.0
    assert result["support_pass"] is False


def test_prefit_independent_audit_passes_but_never_authorizes_scoring() -> None:
    result = audit.prefit_audit()
    assert result["all_passed"] is True
    assert result["scoring_authorized"] is False
    assert result["source_attribution_permitted"] is False
    assert result["research_only"] is True
    assert result["live_ordering_enabled"] is False
    assert result["order_placement"] == "disabled"


def test_postfit_stop_audit_when_artifacts_exist() -> None:
    if not (audit.DEFAULT_ROOT / "fit_complete.json").exists():
        pytest.skip("V2 stop artifacts are not complete yet")
    result = audit.postfit_audit(audit.DEFAULT_ROOT)
    assert result["all_passed"] is True
    assert result["status"] == "support_stop_verified"
    assert result["scoring_authorized"] is False
    assert result["predictions_generated"] is False
    assert result["source_attribution_permitted"] is False
    assert result["later_periods_read"] is False
    assert result["live_shadows_read_by_audit"] is False
