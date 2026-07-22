from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from stocker_research.broad_conflict_advance_hazard_v02 import (
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    DENSE_H1_FEATURES,
    ROUTE_FEATURES,
    advance_increment_passes,
    assign_frozen_route_states,
    broad_conflict_mechanism_passes,
    candidate_normalized_weights,
    choose_broad_conflict_decision,
    earliest_completion_lead,
    fixed_lead_labels,
    predecessor_surface_differences,
    prefix_proximity,
    remaining_required_transitions,
    route_bundle_permutation,
    session_bootstrap_multiplicities,
    theoretical_raw_population,
)
from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES as PREDECESSOR_H0_FEATURES,
)
from stocker_research.route_competition_hazard_v0 import (
    ROUTE_FEATURES as PREDECESSOR_ROUTE_FEATURES,
)
from stocker_research.route_competition_hazard_v0 import (
    fit_hazard_model,
    reject_protected_dates,
)


def test_dense_checkpoint_generation_is_the_preregistered_even_set() -> None:
    assert DENSE_CHECKPOINTS == (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34)


def test_theoretical_population_uses_raw_rows_before_advance_exclusions() -> None:
    assert theoretical_raw_population(eligible_sessions=160) == 48_000


@pytest.mark.parametrize(
    ("completion_ordinals", "expected"),
    [
        ([11, 12, 13], 1),
        ([12, 13], 2),
        ([13], 3),
        ([10, 14], 0),
    ],
    ids=("lead-one", "lead-two", "lead-three", "no-completion"),
)
def test_earliest_completion_lead(completion_ordinals: list[int], expected: int) -> None:
    assert earliest_completion_lead(10, completion_ordinals) == expected


@pytest.mark.parametrize(
    ("motif_type", "path", "progress", "declared", "expected"),
    [
        ("primitive", [0, 1, 0], 2, 1, 1),
        ("repeat", [0, 1, 0, 1, 0], 3, 2, 2),
        ("composite", [1, 0, 1, 0, 1, 2, 1], 4, 3, 3),
    ],
)
def test_prefix_remaining_transitions_use_canonical_paths(
    motif_type: str,
    path: list[int],
    progress: int,
    declared: int,
    expected: int,
) -> None:
    assert (
        remaining_required_transitions(
            progress_states=progress,
            canonical_oriented_path=path,
            motif_type=motif_type,
            declared_transitions_remaining=declared,
        )
        == expected
    )


def test_one_transition_away_detection_uses_all_current_prefixes() -> None:
    prefixes = pd.DataFrame(
        {
            "bar_ordinal": [10, 10, 10, 11],
            "semantic_loop_id": ["P", "R", "C", "FUTURE"],
            "orientation_id": ["p", "r", "c", "future"],
            "motif_type": ["primitive", "repeat", "composite", "primitive"],
            "progress_states": [2, 3, 4, 2],
            "transitions_remaining": [1, 2, 3, 1],
        }
    )
    paths = {
        ("P", "p"): [0, 1, 0],
        ("R", "r"): [0, 1, 0, 1, 0],
        ("C", "c"): [1, 0, 1, 0, 1, 2, 1],
    }
    assert prefix_proximity(prefixes, checkpoint=10, canonical_oriented_paths=paths) == {
        "any_prefix_one_transition_from_completion": 1,
        "minimum_remaining_transitions": 1.0,
        "number_of_one_transition_away_prefixes": 1,
    }


@pytest.mark.parametrize(
    ("lead", "one_away", "eligible", "target"),
    [
        (1, 0, 0, 0),
        (2, 1, 0, 1),
        (2, 0, 1, 1),
        (3, 0, 1, 1),
        (0, 0, 1, 0),
    ],
)
def test_clean_advance_excludes_lead_one_and_near_complete_prefixes(
    lead: int, one_away: int, eligible: int, target: int
) -> None:
    labels = fixed_lead_labels(
        first_completion_lead=lead,
        any_prefix_one_transition_from_completion=one_away,
    )
    assert labels["advance_eligible"] == eligible
    assert labels["completion_in_bars_2_or_3"] == target


def test_candidate_normalized_weighting_equalizes_stock_sessions() -> None:
    frame = pd.DataFrame(
        {
            "period": ["assessment"] * 5,
            "session": ["s1", "s1", "s1", "s1", "s2"],
            "symbol": ["A", "A", "B", "B", "A"],
            "advance_eligible": [1, 1, 1, 0, 1],
        }
    )
    weighted = candidate_normalized_weights(frame)
    assert weighted.loc[:2, "sequential_row_weight"].tolist() == [0.25, 0.25, 0.5]
    assert np.isnan(weighted.loc[3, "sequential_row_weight"])
    assert weighted.loc[4, "sequential_row_weight"] == 1.0
    totals = (
        weighted.loc[weighted["advance_eligible"].eq(1)]
        .groupby(["session", "symbol"])["sequential_row_weight"]
        .sum()
    )
    assert totals.to_dict() == {("s1", "A"): 0.5, ("s1", "B"): 0.5, ("s2", "A"): 1.0}


def test_shared_checkpoint_predecessor_equivalence_detects_feature_drift() -> None:
    reference = pd.DataFrame(
        {
            "row_id": ["A|2025-01-02|6", "B|2025-01-02|6"],
            "checkpoint_timestamp_utc": pd.to_datetime(
                ["2025-01-02T14:55:00Z", "2025-01-02T14:55:00Z"]
            ),
            "route_resolution_state": ["BROAD_CONFLICT", "OTHER"],
            "registered_completion_next_3_bars": [1, 0],
            "f0": [1.0, 2.0],
            "r0": [3.0, 4.0],
        }
    )
    exact = predecessor_surface_differences(
        reference, reference.copy(), feature_columns=("f0", "r0")
    )
    assert exact == {
        "row_identity_mismatches": 0,
        "checkpoint_timestamp_mismatches": 0,
        "target_mismatches": 0,
        "route_resolution_label_mismatches": 0,
        "maximum_shared_feature_difference": 0.0,
    }
    changed = reference.copy()
    changed.loc[1, "r0"] += 0.25
    assert predecessor_surface_differences(reference, changed, feature_columns=("f0", "r0"))[
        "maximum_shared_feature_difference"
    ] == pytest.approx(0.25)


def test_frozen_h0_and_route_feature_surfaces_are_exact() -> None:
    predecessor_non_clock = tuple(
        feature for feature in PREDECESSOR_H0_FEATURES if not feature.startswith("checkpoint_")
    )
    assert (
        *predecessor_non_clock,
        *(f"checkpoint_{checkpoint}" for checkpoint in DENSE_CHECKPOINTS),
    ) == DENSE_H0_FEATURES
    assert ROUTE_FEATURES == PREDECESSOR_ROUTE_FEATURES
    assert (*DENSE_H0_FEATURES, *ROUTE_FEATURES) == DENSE_H1_FEATURES


def test_frozen_route_resolution_labels_preserve_precedence() -> None:
    thresholds = {
        "prefix_family_entropy": [0.2, 0.5, 0.8],
        "top_minus_second_prefix_depth": [0.1, 0.2, 0.4],
        "top_prefix_depth_fraction": [0.2, 0.4, 0.7],
    }
    frame = pd.DataFrame(
        {
            "active_prefix_count": [8, 6, 5, 2, 4],
            "active_prefix_count_change_last_3_bars": [0, -1, 0, 0, 0],
            "top_prefix_depth_fraction": [0.3, 0.5, 0.8, 0.1, 0.4],
            "top_minus_second_prefix_depth": [0.0, 0.3, 0.5, 0.0, 0.2],
            "prefix_family_entropy": [0.9, 0.5, 0.2, 0.0, 0.5],
            "depth_margin_change_last_3_bars": [0.0, 0.1, 0.0, 0.0, 0.0],
        }
    )
    assert assign_frozen_route_states(frame, thresholds).tolist() == [
        "BROAD_CONFLICT",
        "NARROWING",
        "DOMINANT_ROUTE",
        "LOW_ROUTE_SUPPORT",
        "OTHER",
    ]


def test_development_only_scaling_uses_the_matching_model_population() -> None:
    rows = 8
    development = pd.DataFrame(
        {
            feature: np.linspace(index, index + 1, rows)
            for index, feature in enumerate(DENSE_H0_FEATURES)
        }
    )
    development["completion_in_bars_2_or_3"] = [0, 1] * 4
    development["row_weight"] = 1.0
    model = fit_hazard_model(
        development,
        features=DENSE_H0_FEATURES,
        target="completion_in_bars_2_or_3",
    )
    shifted = development.copy()
    shifted.loc[:, list(DENSE_H0_FEATURES)] += 1000.0
    model.predict_probability(shifted)
    assert np.allclose(model.scaler.mean_, development[list(DENSE_H0_FEATURES)].mean())


def test_session_bootstrap_and_route_bundle_permutation_preserve_slates() -> None:
    sessions = pd.Series(["s1", "s1", "s2", "s2"])
    draws = session_bootstrap_multiplicities(sessions, draws=15, seed=31)
    assert len(draws) == 15
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)

    frame = pd.DataFrame(
        {
            "period": ["development"] * 4,
            "session": ["s1"] * 4,
            "checkpoint": [6] * 4,
            "symbol": ["A", "B", "C", "D"],
            "route_a": [1, 2, 3, 4],
            "route_b": [10, 20, 30, 40],
            "baseline": [100, 200, 300, 400],
        }
    )
    permuted = route_bundle_permutation(
        frame,
        route_features=("route_a", "route_b"),
        strata=("period", "session", "checkpoint"),
        seed=9,
    )
    assert permuted["baseline"].tolist() == frame["baseline"].tolist()
    assert sorted(zip(permuted.route_a, permuted.route_b, strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
    ]


def test_protected_date_rejection() -> None:
    reject_protected_dates(pd.DataFrame({"session": ["2025-08-22"]}))
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def _passing_advance_gates() -> dict[str, object]:
    return {
        "log_loss_improvement": 0.01,
        "brier_improvement": 0.001,
        "auc_improvement": 0.0,
        "average_precision_improvement": 0.002,
        "bootstrap_80_log_loss_lower": 0.0,
        "bootstrap_80_brier_lower": 0.0,
        "bootstrap_80_average_precision_lower": 0.0,
        "positive_months": 5,
        "materially_adverse_checkpoint_groups": 0,
        "real_exceeds_all_nulls": True,
        "support_and_concentration_passed": True,
    }


def _passing_broad_conflict_gates() -> dict[str, object]:
    return {
        "assessment_minus_pooled": 0.01,
        "assessment_minus_low_route_support": 0.02,
        "bootstrap_80_pooled_difference_lower": 0.0,
        "bootstrap_80_low_route_difference_lower": 0.0,
        "development_minus_pooled": 0.005,
        "positive_assessment_months": 5,
        "materially_adverse_checkpoint_groups": 0,
        "assessment_rows": 3000,
        "assessment_positives": 100,
        "assessment_sessions": 100,
        "assessment_stocks": 15,
    }


def test_decision_logic_requires_model_and_broad_conflict_gates() -> None:
    assert advance_increment_passes(_passing_advance_gates())
    assert broad_conflict_mechanism_passes(_passing_broad_conflict_gates())
    assert (
        choose_broad_conflict_decision(
            blocker=None,
            advance_passed=True,
            broad_conflict_passed=True,
            broad_conflict_descriptively_enriched=True,
            baseline_meaningful=True,
        )
        == "broad_route_conflict_adds_clean_advance_warning"
    )
    assert (
        choose_broad_conflict_decision(
            blocker=None,
            advance_passed=True,
            broad_conflict_passed=False,
            broad_conflict_descriptively_enriched=True,
            baseline_meaningful=True,
        )
        == "route_competition_adds_clean_advance_warning_without_state_specificity"
    )
    assert (
        choose_broad_conflict_decision(
            blocker="blocked_insufficient_dense_advance_positive_support",
            advance_passed=True,
            broad_conflict_passed=True,
            broad_conflict_descriptively_enriched=True,
            baseline_meaningful=True,
        )
        == "blocked_insufficient_dense_advance_positive_support"
    )
    assert (
        choose_broad_conflict_decision(
            blocker=None,
            advance_passed=False,
            broad_conflict_passed=False,
            broad_conflict_descriptively_enriched=True,
            baseline_meaningful=True,
        )
        == "descriptive_broad_conflict_structure_only"
    )
    assert (
        choose_broad_conflict_decision(
            blocker=None,
            advance_passed=False,
            broad_conflict_passed=False,
            broad_conflict_descriptively_enriched=False,
            baseline_meaningful=True,
        )
        == "compressed_transition_baseline_only"
    )
    assert (
        choose_broad_conflict_decision(
            blocker=None,
            advance_passed=False,
            broad_conflict_passed=False,
            broad_conflict_descriptively_enriched=False,
            baseline_meaningful=False,
        )
        == "no_clean_advance_route_increment"
    )


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run_screen_v02.py"
    specification = importlib.util.spec_from_file_location("test_broad_conflict_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_auditor() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "audit_screen_v02.py"
    specification = importlib.util.spec_from_file_location("test_broad_conflict_auditor", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_blocked_report_does_not_require_model_metric_columns() -> None:
    runner = _load_runner()
    blocker = "blocked_insufficient_dense_advance_positive_support"
    report = runner.build_report(
        {
            "primary_decision": blocker,
            "blocker": blocker,
        },
        {
            "reconstructed_raw_assessment_rows": 47_847,
            "theoretical_raw_assessment_rows": 48_000,
            "raw_retention": 47_847 / 48_000,
            "advance_rows": 34_577,
            "advance_positive_outcomes": 399,
            "base_rate": 0.01,
        },
        pd.DataFrame({"stop_reason": [blocker]}),
        pd.DataFrame({"stop_reason": [blocker]}),
    )
    assert f"`{blocker}`" in report
    assert "before model fitting" in report


def test_audit_finalizer_fail_closes_missing_decision_and_report(tmp_path: Path) -> None:
    auditor = _load_auditor()
    output = tmp_path / "artifacts" / "primary"
    auditor.finalize_audited_report(output, {"passed": False})

    decision = auditor.read_json(output / "decision.json")
    assert decision["primary_decision"] == "blocked_reproducibility_or_audit_failure"
    assert decision["blocker"] == "blocked_reproducibility_or_audit_failure"
    assert decision["research_only"] is True
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "`blocked_reproducibility_or_audit_failure`" in report
    assert "Independent lightweight audit passed: False." in report
    assert (tmp_path / "reports" / "report.md").read_text(encoding="utf-8") == report


def test_audit_finalizer_overrides_support_blocker_after_discrepancy(tmp_path: Path) -> None:
    auditor = _load_auditor()
    output = tmp_path / "artifacts" / "primary"
    output.mkdir(parents=True)
    auditor.write_json(
        output / "decision.json",
        {
            "primary_decision": "blocked_insufficient_dense_advance_positive_support",
            "blocker": "blocked_insufficient_dense_advance_positive_support",
        },
    )
    (output / "report.md").write_text(
        "# Screen\n\n"
        "Primary decision: `blocked_insufficient_dense_advance_positive_support`.\n\n"
        "This blocked screen provides no evidence of trading utility.\n",
        encoding="utf-8",
    )

    auditor.finalize_audited_report(output, {"passed": False})

    decision = auditor.read_json(output / "decision.json")
    assert decision["primary_decision"] == "blocked_reproducibility_or_audit_failure"
    assert decision["blocker"] == "blocked_reproducibility_or_audit_failure"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "`blocked_reproducibility_or_audit_failure`" in report
    assert "blocked_insufficient_dense_advance_positive_support" not in report


def test_auditor_detects_serialized_gate_drift() -> None:
    auditor = _load_auditor()
    expected = {"increment": 0.01, "passed": True, "support": {"rows": 400}}
    assert auditor.compare_mappings(expected, expected)["passed"] is True

    forged = {"increment": 0.02, "passed": True, "support": {"rows": 400}}
    comparison = auditor.compare_mappings(forged, expected)
    assert comparison["passed"] is False
    assert comparison["mismatches"] == 1


def test_auditor_independently_rejects_protected_feature_timestamp() -> None:
    auditor = _load_auditor()
    panel = pd.DataFrame(
        {
            "session": ["2025-08-22"],
            "checkpoint_timestamp_utc": ["2025-08-22T15:00:00Z"],
            "feature_available_timestamp_utc": ["2025-08-22T15:05:00Z"],
        }
    )
    states = pd.DataFrame(
        {
            "session": ["2025-08-22"],
            "bar_complete_timestamp": ["2025-08-22T16:40:00Z"],
        }
    )
    ledger = pd.DataFrame(
        {
            "session": ["2025-08-22"],
            "available_timestamp_utc": ["2025-08-22T16:35:00Z"],
        }
    )
    protected = {
        "minimum_timestamp_read": "2025-08-22T16:40:00Z",
        "maximum_timestamp_read": "2025-08-22T16:40:00Z",
        "maximum_structural_timestamp_read": "2025-08-22T16:35:00Z",
        "maximum_checkpoint_timestamp": "2025-08-22T15:00:00Z",
        "protected_rows_by_source": {
            "causal_state_trace": 0,
            "structural_ledger": 0,
            "dense_checkpoint_panel": 0,
        },
        "protected_rows_materialised": 0,
        "passed": True,
    }
    source = {
        "minimum_timestamp_read": "2025-08-22T16:40:00Z",
        "maximum_timestamp_read": "2025-08-22T16:40:00Z",
        "protected_rows_materialised": 0,
    }
    assert (
        auditor.independent_boundary_audit(panel, states, ledger, protected, source)["passed"]
        is True
    )

    panel.loc[0, "feature_available_timestamp_utc"] = "2025-08-23T00:00:00Z"
    result = auditor.independent_boundary_audit(panel, states, ledger, protected, source)
    assert result["passed"] is False
    assert (
        result["protected_rows_by_independently_read_surface"]["feature_available_timestamp"] == 1
    )

    panel.loc[0, "feature_available_timestamp_utc"] = "2025-08-22T15:05:00Z"
    states.loc[0, "bar_complete_timestamp"] = "2023-12-29T16:40:00Z"
    protected["minimum_timestamp_read"] = "2023-12-29T16:40:00Z"
    protected["maximum_timestamp_read"] = "2023-12-29T16:40:00Z"
    source["minimum_timestamp_read"] = "2023-12-29T16:40:00Z"
    source["maximum_timestamp_read"] = "2023-12-29T16:40:00Z"
    assert (
        auditor.independent_boundary_audit(panel, states, ledger, protected, source)["passed"]
        is False
    )


def test_auditor_reconstructs_split_month_and_checkpoint_group() -> None:
    auditor = _load_auditor()
    panel = pd.DataFrame(
        {
            "session": ["2024-12-31", "2025-01-02", "2025-08-22"],
            "checkpoint": [6, 20, 34],
            "period": ["development", "assessment", "assessment"],
            "year_month": ["2024-12", "2025-01", "2025-08"],
            "checkpoint_group": ["early_6_14", "middle_16_24", "later_26_34"],
        }
    )
    corrected, audit = auditor.independently_reconstruct_chronology_labels(panel)
    assert audit["passed"] is True
    assert corrected[["period", "year_month", "checkpoint_group"]].equals(
        panel[["period", "year_month", "checkpoint_group"]].astype("string")
    )

    panel.loc[1, ["period", "year_month", "checkpoint_group"]] = [
        "development",
        "2025-02",
        "early_6_14",
    ]
    corrected, audit = auditor.independently_reconstruct_chronology_labels(panel)
    assert audit["passed"] is False
    assert audit["period_mismatches"] == 1
    assert audit["year_month_mismatches"] == 1
    assert audit["checkpoint_group_mismatches"] == 1
    assert corrected.loc[1, ["period", "year_month", "checkpoint_group"]].tolist() == [
        "assessment",
        "2025-01",
        "middle_16_24",
    ]


def test_audit_entrypoint_fail_closes_unexpected_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = _load_auditor()
    output = tmp_path / "artifacts" / "primary"
    output.mkdir(parents=True)
    (output / "decision.json").write_text("[]\n", encoding="utf-8")

    def raise_invalid_artifact(_: Path) -> dict[str, object]:
        raise KeyError("missing null row")

    monkeypatch.setattr(auditor, "run_audit", raise_invalid_artifact)
    audit, exit_code = auditor.audit_entrypoint(output)

    assert exit_code == 1
    assert audit["passed"] is False
    assert audit["exception_type"] == "KeyError"
    decision = auditor.read_json(output / "decision.json")
    assert decision["primary_decision"] == "blocked_reproducibility_or_audit_failure"
    assert decision["blocker"] == "blocked_reproducibility_or_audit_failure"
    assert auditor.read_json(output / "lightweight_audit.json")["passed"] is False
    assert "Independent lightweight audit passed: False." in (output / "report.md").read_text(
        encoding="utf-8"
    )


def test_audit_entrypoint_fail_closes_unreadable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = _load_auditor()
    output = tmp_path / "artifacts" / "primary"
    output.mkdir(parents=True)
    auditor.write_json(
        output / "decision.json",
        {
            **auditor.SAFETY_FLAGS,
            "primary_decision": "broad_route_conflict_adds_clean_advance_warning",
            "blocker": None,
        },
    )
    (output / "report.md").write_bytes(b"\xff\xfe")
    monkeypatch.setattr(
        auditor,
        "run_audit",
        lambda _: {**auditor.SAFETY_FLAGS, "checks": {"all": True}, "passed": True},
    )

    audit, exit_code = auditor.audit_entrypoint(output)

    assert exit_code == 1
    assert audit["passed"] is False
    assert audit["exception_type"] == "UnicodeDecodeError"
    decision = auditor.read_json(output / "decision.json")
    assert decision["primary_decision"] == "blocked_reproducibility_or_audit_failure"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Independent lightweight audit passed: False." in report
