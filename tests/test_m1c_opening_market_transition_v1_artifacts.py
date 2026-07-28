from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from stocker_prospective.opening_market_transition_v1 import (
    OpeningTransitionThresholdManifestV1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-opening-market-transition-v1"
    / "artifacts"
    / "primary"
)
REPORT = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-opening-market-transition-v1"
    / "reports"
    / "m1c_opening_market_transition_v1.md"
)
PRIOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-signed-market-shock-transition-v1"
    / "artifacts"
    / "primary"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_required_opening_experiment_artifacts_are_complete() -> None:
    required = {
        "prior_signed_shock_population_reconciliation_v1.csv",
        "checkpoint_6_timing_audit_v1.json",
        "canonical_vti_source_audit_v1.json",
        "frozen_2024_opening_threshold_manifest_v1.json",
        "predictor_only_calibration_v1.csv",
        "checkpoint_6_population_accounting_v1.csv",
        "episode_opening_transition_v1.parquet",
        "unique_opening_transition_events_v1.csv",
        "assessment_results_v1.csv",
        "stress_results_v1.csv",
        "market_following_results_v1.csv",
        "amplification_results_v1.csv",
        "resistance_results_v1.csv",
        "continuous_ranking_results_v1.csv",
        "transition_sign_stratification_v1.csv",
        "gap_open_alignment_diagnostics_v1.csv",
        "severe_vs_normal_comparison_v1.csv",
        "baseline_comparisons_v1.csv",
        "session_cluster_bootstrap_v1.parquet",
        "opening_transition_event_cluster_bootstrap_v1.parquet",
        "leave_one_month_out_v1.csv",
        "leave_one_stock_out_v1.csv",
        "leave_one_transition_event_out_v1.csv",
        "null_and_temporal_placebo_results_v1.json",
        "concentration_report_v1.csv",
        "summary_v1.json",
        "provenance_manifest_v1.json",
    }

    assert required.issubset({path.name for path in PRIMARY.iterdir()})
    assert REPORT.is_file()


def test_threshold_manifest_and_opened_population_are_frozen() -> None:
    manifest = OpeningTransitionThresholdManifestV1.model_validate_json(
        (PRIMARY / "frozen_2024_opening_threshold_manifest_v1.json").read_text()
    )
    population = pd.read_csv(
        PRIMARY / "checkpoint_6_population_accounting_v1.csv"
    )

    assert manifest.checkpoint_v1 == 6
    assert manifest.expected_opening_bar_count_v1 == 6
    assert manifest.thresholds.market_opening_return_support_v1 == 247
    fresh = population.loc[
        population["stage"].eq("canonical_fresh_first_entry_rows")
    ].set_index("period")["row_count"]
    assert fresh.to_dict() == {
        "development": 413,
        "assessment": 360,
        "stress": 442,
    }
    eligible = population.loc[
        population["stage"].eq("final_eligible_rows")
    ].set_index("period")["row_count"]
    assert eligible.to_dict() == {
        "development": 407,
        "assessment": 356,
        "stress": 437,
    }


def test_prior_population_reconciliation_is_exact_and_explanatory() -> None:
    reconciliation = pd.read_csv(
        PRIMARY / "prior_signed_shock_population_reconciliation_v1.csv"
    )

    assert len(reconciliation) == 49
    by_period = reconciliation.groupby("period").agg(
        tail=("included_in_tail_phase_diagnostics_v1", "sum"),
        primary=("included_in_primary_signed_shock_population_v1", "sum"),
    )
    assert by_period.to_dict(orient="index") == {
        "assessment": {"tail": 15, "primary": 9},
        "stress": {"tail": 34, "primary": 29},
    }
    excluded = reconciliation.loc[
        ~reconciliation[
            "included_in_primary_signed_shock_population_v1"
        ].astype(bool)
    ]
    assert len(excluded) == 11
    assert excluded[
        "minutes_since_prior_canonical_fresh_episode_v1"
    ].eq(20.0).all()
    assert excluded["inclusion_exclusion_reason_v1"].str.contains(
        "minimum_episode_spacing_not_met:20<30",
        regex=False,
    ).all()


def test_frozen_m1c_tail_phase_and_a1_fields_are_unchanged() -> None:
    opening = pd.read_parquet(
        PRIMARY / "episode_opening_transition_v1.parquet"
    )
    prior = pd.read_parquet(
        PRIOR_PRIMARY / "episode_market_state_response_v1.parquet",
        filters=[("checkpoint", "==", 6)],
    )
    fields = [
        "episode_id",
        "M1C_probability",
        "m1c_high_tail_v1",
        "m1c_tail_phase_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
    ]
    left = opening[["stock", "session", "checkpoint", *fields]]
    right = prior[["stock", "session", "checkpoint", *fields]]

    pd.testing.assert_frame_equal(
        left.sort_values(["stock", "session"]).reset_index(drop=True),
        right.sort_values(["stock", "session"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_protected_execution_and_prior_artifact_guards_hold() -> None:
    summary = json.loads((PRIMARY / "summary_v1.json").read_text())
    provenance = json.loads(
        (PRIMARY / "provenance_manifest_v1.json").read_text()
    )
    panel = pd.read_parquet(PRIMARY / "episode_opening_transition_v1.parquet")

    assert panel["session"].astype(str).lt("2026-01-01").all()
    assert summary["confirmations"] == {
        "a1_unchanged": True,
        "broker_accessed": False,
        "contaminated_fields_used": False,
        "engineering_transfer_used_for_recalibration": False,
        "fresh_episode_definition_unchanged": True,
        "future_bars_used_for_predictors": False,
        "m1c_unchanged": True,
        "option_profitability_tested": False,
        "order_placed": False,
        "order_routing_enabled": False,
        "peer_slate_normalisation_used": False,
        "prospective_recorder_logging_only_integrated": True,
        "protected_2026_outcomes_accessed": False,
        "tail_phase_v1_unchanged": True,
    }
    assert provenance["protected_data_confirmation"][
        "protected_data_opened"
    ] is False
    assert provenance["execution_confirmation"] == {
        "broker_access": False,
        "order_routing_enabled": False,
        "orders_submitted": False,
    }
    assert _sha256(
        PRIOR_PRIMARY / "episode_market_state_response_v1.parquet"
    ) == "5abcff67f6c17ec6ab99c8bbc9d37e769594eb9cc90513882d3dd1e0f7ecf9a7"
    report = REPORT.read_text()
    assert "**Not tested**" in report
    assert "executable option outcomes" in " ".join(report.split())


def test_recorded_random_seeds_match_the_frozen_contract_exactly() -> None:
    contract = json.loads(
        (
            REPO_ROOT
            / "research"
            / "directional-readiness"
            / "20260728-m1c-opening-market-transition-v1"
            / "contract.json"
        ).read_text()
    )
    provenance = json.loads(
        (PRIMARY / "provenance_manifest_v1.json").read_text()
    )
    session_draws = pd.read_parquet(
        PRIMARY / "session_cluster_bootstrap_v1.parquet"
    )
    event_draws = pd.read_parquet(
        PRIMARY / "opening_transition_event_cluster_bootstrap_v1.parquet"
    )
    null_draws = pd.read_parquet(PRIMARY / "primary_null_draws_v1.parquet")
    null_results = json.loads(
        (PRIMARY / "null_and_temporal_placebo_results_v1.json").read_text()
    )

    expected = {
        "session_cluster_bootstrap": contract["bootstrap"]["session_seed"],
        "opening_transition_event_cluster_bootstrap": contract["bootstrap"][
            "opening_transition_event_seed"
        ],
        "primary_null": contract["null"]["seed"],
    }
    assert provenance["random_seeds"] == expected
    assert set(session_draws["seed"]) == {expected["session_cluster_bootstrap"]}
    assert set(event_draws["seed"]) == {
        expected["opening_transition_event_cluster_bootstrap"]
    }
    assert set(null_draws["seed"]) == {expected["primary_null"]}
    assert {
        value["seed"]
        for value in null_results["primary_null"].values()
    } == {expected["primary_null"]}
