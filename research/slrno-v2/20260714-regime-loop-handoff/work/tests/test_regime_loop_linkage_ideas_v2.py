from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_regime_loop_linkage_ideas_v2.py"
SPEC = importlib.util.spec_from_file_location("regime_loop_linkage_v2", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v2_binds_immutable_v1_contract_and_runner() -> None:
    assert MODULE.sha256(MODULE.PARENT_CONTRACT) == MODULE.PARENT_CONTRACT_SHA256
    assert MODULE.sha256(MODULE.PARENT_RUNNER) == MODULE.PARENT_RUNNER_SHA256


def test_v2_amends_only_structural_probability_identity() -> None:
    contract = MODULE.load_contract()
    assert contract["contract_id"] == "regime_loop_linkage_ideas_v2"
    assert contract["population_and_join"]["occurrence_probability"] == (
        "factor_occurrence_oof.qhistory"
    )
    assert contract["population_and_join"]["quality_parent_loop_probability_role"] == (
        "diagnostic_only_excluded_from_link_probabilities"
    )
    assert contract["global_variant_gate"] == MODULE.PARENT.load_contract()[
        "global_variant_gate"
    ]


def test_v2_failure_record_confirms_no_v1_linkage_result() -> None:
    contract = MODULE.load_contract()["v2_amendment"]
    stopped = contract["v1_fail_closed_record"]
    assert stopped["joint_scores_or_metrics_calculated"] is False
    assert stopped["meta_models_fitted"] is False
    assert stopped["movement_outcomes_or_linkage_results_used_to_make_amendment"] is False


def test_v2_preserves_exact_safety_boundary() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["decision"]["later_period_scoring"] is False
    assert contract["decision"]["named_loop_good_or_high_promotion_permitted"] is False

