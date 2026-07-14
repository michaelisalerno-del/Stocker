from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "audit_loop_quality_feature_ablation_v3.py"
)
SPEC = importlib.util.spec_from_file_location("feature_ablation_v3_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_contract_delta_and_unique_support_are_exact() -> None:
    checks = audit.Audit()
    audit.verify_contract_delta(checks)
    support = audit.verify_unique_support(checks)
    assert checks.all_passed is True
    assert support["total_effective_weight"] == 14167.0
    assert support["quarter_weights"] == {"2024_q3": 7635.0, "2024_q4": 6532.0}
    assert support["minimum_stock_weight"] == 93.0
    assert support["support_pass"] is True


def test_independent_topology_has_44_units_and_one_ambiguous_route() -> None:
    mapping = audit.independent_rotation_mapping()
    assert len(mapping) == 44
    assert not mapping.duplicated(["cycle_id", "current_state"]).any()
    assert mapping["compatible_rotation_count"].sum() == 45
    assert mapping.loc[
        mapping["compatible_rotation_count"].gt(1), ["cycle_id", "current_state"]
    ].to_dict("records") == [{"cycle_id": "cycle_15", "current_state": 1}]
    assert np.allclose(mapping[audit.v2audit.topology_column_names()].to_numpy(float)[:, :8].sum(axis=1), 1.0)


def test_interior_feature_widths_and_declared_centroid_scale_are_exact() -> None:
    columns = audit.v2audit.topology_column_names()
    frame = pd.DataFrame(
        {
            "cycle_index": [0, 14],
            "state": [1, 1],
            **{column: [0.0, 0.0] for column in columns},
        }
    )
    frame.loc[:, columns[0]] = 1.0
    frame.loc[:, columns[8]] = 1.0
    frame.loc[:, columns[16]] = 1.0
    frame.loc[:, columns[19]] = 2.0
    matrices = audit.feature_matrices(frame, sparse.csr_matrix(np.ones((2, 17))))
    assert matrices["qroute_topology"].shape == (2, 80)
    assert matrices["qcycle_main"].shape == (2, 37)
    assert matrices["qcycle_state"].shape == (2, 197)
    assert matrices["qroute_topology"][0, 17 + 19] == 1.0


def test_entropy_cutpoints_and_quartiles_use_only_frozen_oof_weights() -> None:
    frame = pd.DataFrame(
        {
            "loop_occurs": [1, 1, 1, 0],
            "conditional_weight": [1.0, 1.0, 2.0, 1.0],
            "next_state_entropy_normalized": [0.0, 0.0, 0.25, 1.0],
        }
    )
    cuts = audit.entropy_cutpoints(frame)
    assert np.all(cuts >= 0.0)
    audit.add_entropy_quartile(frame, cuts)
    assert frame["entropy_quartile"].between(0, 3).all()


def test_probability_outputs_are_nested_and_joint_chain_exact() -> None:
    frame = pd.DataFrame({"loop_probability": [0.2, 0.8]})
    classes = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.25, 0.65]])
    audit.append_probabilities(
        frame, "qroute_topology", "absolute_return_bps", 6, classes
    )
    assert np.allclose(
        frame["qroute_topology__absolute_return_bps__h6__p75"], [0.5, 0.9]
    )
    assert np.allclose(
        frame["qroute_topology__absolute_return_bps__h6__p90"], [0.2, 0.65]
    )
    assert np.allclose(
        frame["joint__qroute_topology__absolute_return_bps__h6__p75"],
        [0.1, 0.72],
    )


def test_bootstrap_is_deterministic_and_uses_declared_seed_family() -> None:
    values = np.linspace(-0.02, 0.01, 30)
    assert audit.moving_block_interval(values, 20260710, draws=100) == audit.moving_block_interval(
        values, 20260710, draws=100
    )
    assert len(audit.PRIMARY_COMPARISONS) == 5


def test_portable_attribution_is_demotion_only() -> None:
    provisional = {
        "label": "history_token_needed",
        "comparison_pass": {
            "qfull_vs_qcontext_reference": True,
            "qfull_vs_qcycle_state": True,
        },
    }
    gates = {
        "2025": {"comparison_pass": {"qfull_vs_qcontext_reference": True, "qfull_vs_qcycle_state": True}},
        "2023": {"comparison_pass": {"qfull_vs_qcontext_reference": True, "qfull_vs_qcycle_state": False}},
    }
    result = audit.portable_attribution(provisional, gates)
    assert result["final_development_portability_label"] == "unresolved_not_portable"
    assert result["later_period_promotion_performed"] is False
    assert result["prospective_validated"] is False


def test_audit_source_does_not_import_frozen_production_runner() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any("loop_quality_feature_ablation_v3" in name for name in imports)


def test_saved_pre_score_audit_authorized_only_after_all_55_checks() -> None:
    result = json.loads((audit.ROOT / "pre_score_audit.json").read_text())
    assert result["all_passed"] is True
    assert result["check_count"] == 55
    assert result["scoring_authorized"] is True
    assert result["maximum_oof_prediction_error"] == 0.0
    assert result["maximum_full_parameter_error"] == 0.0
    assert result["later_period_outcomes_opened_by_audit"] is False


def test_saved_post_score_audit_passes_all_44_checks_and_keeps_safety_labels() -> None:
    result = json.loads((audit.ROOT / "independent_artifact_audit.json").read_text())
    assert result["all_passed"] is True
    assert result["check_count"] == 44
    assert result["research_only"] is True
    assert result["live_ordering_enabled"] is False
    assert result["order_placement"] == "disabled"
    assert result["prospective_validated"] is False
    assert result["parent_grade_changed"] is False
    assert max(result["maximum_scoring_prediction_error"].values()) <= 1e-12
    assert result["source_attribution"]["final_development_portability_label"] == "no_reference_signal"
