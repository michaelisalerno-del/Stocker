import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run_oriented_route_movement_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("oriented_route_diagnostic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

CONTRACT = diagnostic.CONTRACT
LABELS = diagnostic.LABELS
build_route_manifest = diagnostic.build_route_manifest
deduplicated_oriented_paths = diagnostic.deduplicated_oriented_paths
minimum_grade = diagnostic.minimum_grade


QUALITY_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
DIAGNOSTIC_ROOT = Path(
    "/private/tmp/stocker_oriented_route_movement_diagnostic_20260710"
)


def test_simple_cycle_rotations_are_closed_and_exact() -> None:
    assert deduplicated_oriented_paths((3, 6), 3) == [(3, 6, 3)]
    assert deduplicated_oriented_paths((3, 6), 6) == [(6, 3, 6)]


def test_repeated_alternation_deduplicates_equivalent_rotations() -> None:
    assert deduplicated_oriented_paths((1, 3, 1, 3), 1) == [
        (1, 3, 1, 3, 1)
    ]


def test_cycle_15_state_1_preserves_ambiguous_union() -> None:
    assert deduplicated_oriented_paths((1, 2, 1, 3), 1) == [
        (1, 2, 1, 3, 1),
        (1, 3, 1, 2, 1),
    ]


def test_route_manifest_has_all_frozen_current_state_units() -> None:
    cycles = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv")
    manifest = build_route_manifest(cycles)
    assert len(manifest) == 44
    assert not manifest.duplicated(["cycle_id", "current_state"]).any()
    assert manifest.loc[
        manifest["route_kind"].eq("ambiguous_union"), "route_id"
    ].tolist() == ["cycle_15@state_1"]


def test_minimum_grade_never_promotes() -> None:
    assert minimum_grade([LABELS["high"], LABELS["good"]]) == LABELS["good"]
    assert minimum_grade([LABELS["high"], LABELS["failed"]]) == LABELS["failed"]
    assert (
        minimum_grade([LABELS["high"], LABELS["unsupported"]])
        == LABELS["unsupported"]
    )


def test_minimum_grade_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        minimum_grade([])
    with pytest.raises(ValueError):
        minimum_grade(["certified"])


def test_contract_is_explicitly_post_outcome_and_non_promotional() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["scientific_status"] == (
        "post_outcome_exploratory_development_diagnostic"
    )
    assert contract["post_outcome_route_selection"] is True
    assert contract["prospective_validation_claim_permitted"] is False
    assert contract["promotion_or_surface_permission"] is False
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"


def test_contract_preserves_both_shadows_and_forbids_refits() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["integrity"]["both_shadow_trees_must_match_before_and_after"]
    assert contract["integrity"]["both_shadow_ledgers_must_remain_empty"]
    forbidden = contract["forbidden_actions"]
    assert forbidden["state_refit"]
    assert forbidden["structural_loop_refit"]
    assert forbidden["quality_model_refit"]
    assert forbidden["threshold_or_temperature_change"]
    assert forbidden["shadow_prediction_or_ledger_change"]


def test_completed_diagnostic_kept_both_shadow_trees_byte_stable() -> None:
    before = json.loads((DIAGNOSTIC_ROOT / "protected_shadows_before.json").read_text())
    after = json.loads((DIAGNOSTIC_ROOT / "protected_shadows_after.json").read_text())
    assert before == after
    for shadow in after.values():
        assert shadow["ledger_size"] == 0
        assert shadow["ledger_lines"] == 0
        assert shadow["outcomes_opened"] is False
        assert shadow["research_only"] is True
        assert shadow["live_ordering_enabled"] is False
        assert shadow["order_placement"] == "disabled"


def test_completed_diagnostic_does_not_promote_a_route() -> None:
    summary = json.loads((DIAGNOSTIC_ROOT / "summary.json").read_text())
    assert summary["models_refit"] is False
    assert summary["thresholds_changed"] is False
    assert summary["diagnostic_good_or_high_global_routes"] == []
    assert summary["promotion_or_surface_permission"] is False
    assert summary["prospective_validated"] is False


def test_only_cycle_09_state_3_h6_has_a_nonfailed_period_horizon_grade() -> None:
    grades = pd.read_csv(DIAGNOSTIC_ROOT / "route_horizon_grades.csv")
    selected = grades.loc[
        ~grades["grade"].isin(
            [LABELS["failed"], LABELS["unsupported"]]
        ),
        ["period", "route_id", "horizon", "grade"],
    ]
    assert selected.to_dict("records") == [
        {
            "period": "2024_oof",
            "route_id": "cycle_09@state_3",
            "horizon": 6,
            "grade": LABELS["good"],
        }
    ]
