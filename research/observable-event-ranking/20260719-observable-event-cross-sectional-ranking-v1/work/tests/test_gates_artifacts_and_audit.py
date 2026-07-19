from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.observable_event_ranking_v1.artifacts import (
    ArtifactBinding,
    ArtifactWriter,
    compare_artifact_directories,
)
from stocker_research.observable_event_ranking_v1.audit import run_independent_audit
from stocker_research.observable_event_ranking_v1.contract import (
    PRIMARY_FEATURES,
    REQUIRED_SAFETY_FLAGS,
)
from stocker_research.observable_event_ranking_v1.decision import (
    evaluate_development_gate,
    evaluate_prospective_gate,
    evaluate_support_gate,
)
from stocker_research.observable_event_ranking_v1.economic import (
    EconomicContinuationDecision,
    EconomicEvaluationBlocked,
    EconomicEvaluationPrerequisites,
    evaluate_economic_continuation_gate,
    require_economic_prerequisites,
)
from stocker_research.observable_event_ranking_v1.pipeline import (
    StageDependencyError,
    build_targets_stage,
    run_preflight,
)
from stocker_research.observable_event_ranking_v1.prospective import (
    PredictionLedgerError,
    SettlementLedgerError,
    append_prediction,
    append_settlement,
    score_frozen_prediction,
)


def _binding() -> ArtifactBinding:
    return ArtifactBinding(
        git_sha="abc123",
        branch="agent/observable-event-ranking-v1",
        contract_hash="contract",
        implementation_hash="implementation",
        data_snapshot_hash="data",
        universe_hash="universe",
        sector_map_hash="sector",
        run_id="run-1",
        random_seeds={"bootstrap": 20260719},
        dependency_versions={"python": "3.12"},
        safety=REQUIRED_SAFETY_FLAGS,
    )


def test_support_gate_uses_all_otherwise_valid_scheduled_slates_as_denominator() -> None:
    slates = pd.DataFrame(
        {
            "slate_id": [f"slate-{number}" for number in range(1_666)],
            "session": pd.date_range("2024-01-01", periods=1_666, freq="D", tz="UTC"),
            "eligible_stock_count": 60,
            "candidate_count": [8] * 1_000 + [0] * 666,
            "otherwise_valid_scheduled_slate": True,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [f"event-{number}" for number in range(5_000)],
            "slate_id": [f"slate-{number % 1_000}" for number in range(5_000)],
            "symbol": [f"S{number % 100:03d}" for number in range(5_000)],
            "session": pd.date_range("2024-01-01", periods=5_000, freq="h", tz="UTC"),
        }
    )

    decision = evaluate_support_gate(
        events,
        slates,
        exact_event_rerun=True,
        outcome_free_audit=True,
    )

    assert decision.gates["candidate_support_fraction"] == pytest.approx(1_000 / 1_666)
    assert decision.passed
    assert decision.decision == "support_gate_passed"


def test_development_gate_cannot_be_rescued_when_candidate_loses_one_primary_metric() -> None:
    result = evaluate_development_gate(
        support_passed=True,
        candidate_mean_ic=0.03,
        baseline_mean_ic=0.02,
        candidate_top_two_minus_median=0.001,
        baseline_top_two_minus_median=0.002,
        positive_fold_fraction=0.75,
        max_stock_event_fraction=0.05,
        max_stock_top_two_fraction=0.10,
        exact_rerun=True,
        independent_audit=True,
    )

    assert result.decision == "historical_no_incremental_ranking_evidence"
    assert not result.authorises_prospective_freeze


def test_development_gate_fails_closed_until_exact_rerun_and_audit() -> None:
    result = evaluate_development_gate(
        support_passed=True,
        candidate_mean_ic=0.03,
        baseline_mean_ic=0.02,
        candidate_top_two_minus_median=0.003,
        baseline_top_two_minus_median=0.002,
        positive_fold_fraction=0.75,
        max_stock_event_fraction=0.05,
        max_stock_top_two_fraction=0.10,
        exact_rerun=False,
        independent_audit=False,
    )

    assert result.decision == "blocked_audit_or_reproducibility_failure"
    assert not result.authorises_prospective_freeze


def test_prospective_gate_refuses_historical_evidence_even_if_numbers_pass() -> None:
    result = evaluate_prospective_gate(
        evidence_type="historical_development",
        months=12,
        evaluable_supported_slates=2_000,
        event_rows=10_000,
        candidate_minus_baseline_ic=0.02,
        ic_ci_lower=0.01,
        candidate_top_two_minus_median=0.002,
        baseline_top_two_minus_median=0.001,
        top_two_difference_ci_lower=0.0001,
        positive_first_six_months=6,
        positive_leave_one_stock_out_fraction=1.0,
        max_stock_event_fraction=0.02,
        top_five_selection_fraction=0.10,
        exact_rerun=True,
        independent_audit=True,
    )

    assert not result.passed
    assert result.decision == "prospective_gate_not_evaluable_from_historical_data"


def test_prediction_is_append_only_and_settlement_refuses_early_outcome(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    settlement_root = tmp_path / "settlements"
    prediction = {
        "prediction_id": "p1",
        "event_id": "e1",
        "slate_id": "s1",
        "symbol": "AAPL",
        "decision_timestamp": "2025-07-07T14:00:00Z",
        "prediction_timestamp": "2025-07-07T14:00:00Z",
        "planned_entry_reference_time": "2025-07-07T14:05:00Z",
        "planned_exit_reference_time": "2025-07-07T15:05:00Z",
        "outcome_available_at": "2025-07-07T15:05:00Z",
        "source_provider": "EODHD",
        "source_dataset_id": "audited-5m-v1",
        "source_hash": "b" * 64,
        "score": 0.3,
        "frozen_baseline_score": 0.2,
        "bundle_hash": "a" * 64,
        "safety": REQUIRED_SAFETY_FLAGS,
    }

    append_prediction(prediction_root, prediction)

    with pytest.raises(PredictionLedgerError):
        append_prediction(prediction_root, prediction)
    with pytest.raises(PredictionLedgerError):
        append_prediction(
            prediction_root, {**prediction, "prediction_id": "p2", "future_return": 1.0}
        )
    with pytest.raises(PredictionLedgerError):
        append_prediction(
            prediction_root,
            {**prediction, "prediction_id": "../outside"},
        )
    with pytest.raises(PredictionLedgerError):
        append_prediction(
            prediction_root,
            {key: value for key, value in prediction.items() if key != "bundle_hash"},
        )
    with pytest.raises(SettlementLedgerError):
        append_settlement(
            settlement_root,
            prediction=prediction,
            settlement={
                "prediction_id": "p1",
                "future_return_60m": 0.01,
                "settlement_status": "settled",
            },
            settlement_time="2025-07-07T15:04:59Z",
        )
    with pytest.raises(SettlementLedgerError):
        append_settlement(
            settlement_root,
            prediction=prediction,
            settlement={
                "prediction_id": "different",
                "future_return_60m": 0.01,
                "settlement_status": "settled",
            },
            settlement_time="2025-07-07T15:05:00Z",
        )


def test_prospective_score_uses_serialized_model_and_frozen_baseline() -> None:
    feature_values = {feature: 1.0 for feature in PRIMARY_FEATURES}
    model = {
        "feature_names": list(PRIMARY_FEATURES),
        "preprocessor": {
            "medians": [0.0] * 12,
            "lower_clip": [-10.0] * 12,
            "upper_clip": [10.0] * 12,
            "means": [0.0] * 12,
            "scales": [1.0] * 12,
        },
        "intercept": 0.5,
        "coefficients": [0.25] + [0.0] * 11,
    }
    baseline = {
        "baseline_id": "B1_EVENT_STRENGTH",
        "kind": "direct_observable_feature",
        "source_feature": "event_strength",
    }

    scored = score_frozen_prediction(
        {
            "prediction_id": "p1",
            "event_id": "e1",
            "slate_id": "s1",
            "symbol": "AAPL",
            "decision_timestamp": "2025-07-07T14:00:00Z",
            "prediction_timestamp": "2025-07-07T14:00:01Z",
            "planned_entry_reference_time": "2025-07-07T14:05:00Z",
            "planned_exit_reference_time": "2025-07-07T15:05:00Z",
            "outcome_available_at": "2025-07-07T15:05:00Z",
            "source_provider": "EODHD",
            "source_dataset_id": "audited-5m-v1",
            "source_hash": "b" * 64,
            "features": feature_values,
        },
        model_parameters=model,
        baseline_parameters=baseline,
        bundle_hash="a" * 64,
    )

    assert scored["score"] == pytest.approx(0.75)
    assert scored["frozen_baseline_score"] == pytest.approx(1.0)
    assert scored["bundle_hash"] == "a" * 64
    assert scored["safety"] == REQUIRED_SAFETY_FLAGS


def test_economic_layer_is_sealed_until_every_prerequisite_is_explicit() -> None:
    with pytest.raises(EconomicEvaluationBlocked):
        require_economic_prerequisites(EconomicEvaluationPrerequisites())
    prerequisites = EconomicEvaluationPrerequisites(
        prospective_structural_gate_passed=True,
        complete_live_quote_coverage=0.80,
        entry_exit_timing_rule_passed=True,
        commission_fee_configuration_version="user-supplied-v1",
        currency_conversion_treatment="USD_only",
        economic_contract_hash="frozen-economic-contract",
    )
    decision = evaluate_economic_continuation_gate(
        prerequisites=prerequisites,
        net_top_two_bootstrap_lower=0.0001,
        cost_stress_1_5x_point_estimate=0.0001,
        max_stock_net_contribution_fraction=0.15,
        short_period_dominated=False,
        exact_rerun=True,
        independent_audit=True,
    )

    assert decision == EconomicContinuationDecision(
        decision="economic_quote_simulation_continuation_gate_passed",
        passed=True,
        quote_simulation_is_achieved_fill=False,
    )


def test_artifact_writer_is_deterministic_for_json_csv_and_parquet(tmp_path: Path) -> None:
    left = ArtifactWriter(tmp_path / "left", _binding())
    right = ArtifactWriter(tmp_path / "right", _binding())
    frame = pd.DataFrame({"b": [2, 1], "a": ["y", "x"]})
    for writer in (left, right):
        writer.json("sample.json", {"z": 1, "a": 2})
        writer.csv("sample.csv", frame, columns=("a", "b"), sort_by=("a",))
        writer.parquet("sample.parquet", frame, columns=("a", "b"), sort_by=("a",))
        writer.manifest()

    comparison = compare_artifact_directories(tmp_path / "left", tmp_path / "right")

    assert comparison.identical
    assert comparison.mismatches == ()


def test_preflight_does_not_open_protected_bars_and_writes_specific_sector_blocker(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    protected = (
        data_dir
        / "raw/source=eodhd/endpoint=intraday/symbol=AAA/interval=5m"
        / "2025-08-23_2026-06-30.json"
    )
    protected.parent.mkdir(parents=True)
    protected.write_text("this is intentionally not valid JSON", encoding="utf-8")
    safe = protected.with_name("2024-01-01_2025-08-23.json")
    safe.write_text("safe pre-cutoff bytes", encoding="utf-8")
    output = tmp_path / "primary"

    result = run_preflight(
        repository_root=Path.cwd(),
        data_dir=data_dir,
        output_dir=output,
        git_sha="test-sha",
        branch="agent/observable-event-ranking-v1",
    )

    decision = json.loads((output / "support_decision.json").read_text(encoding="utf-8"))
    sources = json.loads((output / "source_identity_manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "static_provenance_audit.json").read_text(encoding="utf-8"))
    assert result.decision == "blocked_missing_point_in_time_sector_membership"
    assert decision["decision"] == "blocked_missing_point_in_time_sector_membership"
    assert sources["protected_files_opened"] == 0
    assert sources["protected_file_count"] == 1
    assert (
        sources["safe_source_files"][0]["source_content_sha256"]
        == hashlib.sha256(b"safe pre-cutoff bytes").hexdigest()
    )
    assert sources["safe_source_files"][0]["content_hashed"] is True
    assert provenance["audit_passed"]
    assert provenance["forbidden_import_violations"] == []
    with pytest.raises(StageDependencyError):
        build_targets_stage(output)


def test_target_stage_guard_allows_an_explicitly_passed_support_decision(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "primary"
    artifact_dir.mkdir()
    (artifact_dir / "support_decision.json").write_text(
        json.dumps({"passed": True, "decision": "support_gate_passed"}),
        encoding="utf-8",
    )

    build_targets_stage(artifact_dir)


def test_sector_map_hash_binds_row_content_not_only_schema_and_count(tmp_path: Path) -> None:
    base = {
        "symbol": "AAA",
        "effective_from": pd.Timestamp("2020-01-01", tz="UTC"),
        "effective_to": pd.NaT,
        "known_at": pd.Timestamp("2019-12-01", tz="UTC"),
        "stable_source_id": "sector:AAA:2020",
        "source_provider": "trusted-pit-sector-source",
        "source_dataset_id": "sector-history-v1",
        "source_hash": "sector-source-hash",
    }
    paths: list[Path] = []
    for name, sector in (("left", "Technology"), ("right", "Industrials")):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame([{**base, "sector": sector}]).to_parquet(path, index=False)
        paths.append(path)
    left = run_preflight(
        repository_root=Path.cwd(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "left-output",
        git_sha="test-sha",
        branch="agent/observable-event-ranking-v1",
        sector_ledger_path=paths[0],
    )
    right = run_preflight(
        repository_root=Path.cwd(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "right-output",
        git_sha="test-sha",
        branch="agent/observable-event-ranking-v1",
        sector_ledger_path=paths[1],
    )

    assert left.binding.sector_map_hash != right.binding.sector_map_hash


def test_independent_auditor_and_exact_comparison_detect_intentional_corruption(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    for destination in (primary, exact):
        run_preflight(
            repository_root=Path.cwd(),
            data_dir=data_dir,
            output_dir=destination,
            git_sha="test-sha",
            branch="agent/observable-event-ranking-v1",
        )
    assert compare_artifact_directories(primary, exact).identical
    clean_audit = run_independent_audit(primary_dir=primary, exact_dir=exact)
    assert clean_audit["audit_passed"], clean_audit["failed_checks"]
    assert all(clean_audit["independent_formula_fixture_checks"].values())
    assert clean_audit["checks"]["independent_event_and_peer_formula_fixtures"]
    assert clean_audit["checks"]["independent_timing_rank_metric_fixtures"]
    assert clean_audit["checks"]["independent_weight_bootstrap_gate_fixtures"]

    payload = json.loads((primary / "support_decision.json").read_text(encoding="utf-8"))
    payload["decision"] = "support_gate_passed"
    (primary / "support_decision.json").write_text(json.dumps(payload), encoding="utf-8")
    corrupted = run_independent_audit(primary_dir=primary, exact_dir=exact)

    assert not corrupted["audit_passed"]
    assert "exact_rerun_identity" in corrupted["failed_checks"]
    assert "primary_artifact_manifest_entries_match_files" in corrupted["failed_checks"]


def test_independent_auditor_rehashes_implementation_sources(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    source = (
        repository_root
        / "packages/stocker_research/src/stocker_research/observable_event_ranking_v1/sample.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text('"""Audited source."""\n', encoding="utf-8")
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    for destination in (primary, exact):
        run_preflight(
            repository_root=repository_root,
            data_dir=tmp_path / "data",
            output_dir=destination,
            git_sha="test-sha",
            branch="agent/observable-event-ranking-v1",
        )
    assert run_independent_audit(
        primary_dir=primary,
        exact_dir=exact,
        repository_root=repository_root,
    )["audit_passed"]

    source.write_text('"""Source changed after the run."""\n', encoding="utf-8")
    corrupted = run_independent_audit(
        primary_dir=primary,
        exact_dir=exact,
        repository_root=repository_root,
    )

    assert not corrupted["audit_passed"]
    assert "implementation_source_files_match_manifest" in corrupted["failed_checks"]


def test_independent_auditor_rejects_retired_exact_event_columns(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    for destination in (primary, exact):
        run_preflight(
            repository_root=Path.cwd(),
            data_dir=tmp_path / "data",
            output_dir=destination,
            git_sha="test-sha",
            branch="agent/observable-event-ranking-v1",
        )
    pd.DataFrame({"state": pd.Series(dtype="string")}).to_parquet(
        primary / "event_ledger.parquet",
        index=False,
    )

    corrupted = run_independent_audit(primary_dir=primary, exact_dir=exact)

    assert not corrupted["audit_passed"]
    assert "event_ledger_outcome_free" in corrupted["failed_checks"]
