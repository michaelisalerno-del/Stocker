from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUNNER = EXPERIMENT_DIR / "run_screen_v0.py"
AUDITOR = EXPERIMENT_DIR / "audit_screen_v0.py"


def test_runner_fails_closed_when_all_local_one_minute_sources_are_missing(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    primary = tmp_path / "primary"
    exact = tmp_path / "exact_rerun"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(exact),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout.strip() == "blocked_one_minute_history_unavailable"

    decision = json.loads((primary / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "blocked_one_minute_history_unavailable"
    assert decision["models_fitted"] == 0
    assert decision["protected_rows_opened"] == 0
    for key, expected in {
        "research_only": True,
        "feasibility_screen": True,
        "observable_only": True,
        "one_minute_sequence_test": True,
        "execution_enabled": False,
        "order_placement": "disabled",
        "broker_integration_required": False,
        "strategy_promotion": False,
        "production_runtime_modified": False,
        "loops_regimes_states_and_structural_paths_forbidden": True,
    }.items():
        assert decision[key] == expected

    coverage = pd.read_csv(primary / "one_minute_availability_audit.csv", keep_default_na=False)
    assert len(coverage) == 8_240
    assert coverage["symbol"].nunique() == 20
    assert coverage["session"].nunique() == 412
    assert set(coverage["source_status"]) == {"missing_source_file"}
    for convention in ("bar_start", "bar_end"):
        assert coverage[f"{convention}_candidate_observed_minute_count"].eq(0).all()
        assert (
            coverage[f"{convention}_candidate_missing_minute_ordinals"]
            .isin({"0-389", "0-209"})
            .all()
        )

    frozen = json.loads(
        (primary / "frozen_population_reconstruction.json").read_text(encoding="utf-8")
    )
    assert frozen["passed"] is True
    assert frozen["development_admitted_rows"] == 1_239
    assert frozen["assessment_admitted_rows"] == 1_560
    assert frozen["assessment_sessions"] == 153
    assert frozen["assessment_stocks"] == 20

    rerun = json.loads((primary / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    assert rerun["passed"] is True
    assert rerun["decision"] == "blocked_one_minute_history_unavailable"
    assert (primary / "assessment_predictions.parquet").is_file()
    assert pd.read_parquet(primary / "assessment_predictions.parquet").empty
    assert (primary / "decision.json").read_bytes() == (exact / "decision.json").read_bytes()


def test_runner_reports_observed_minutes_from_a_partial_local_source(tmp_path: Path) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    source = provider_root / "symbol=AAL" / "timeframe=1m" / "data.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T14:31:00Z",
                    "2025-01-02T14:30:00Z",
                    "2025-08-25T13:30:00Z",
                ],
                utc=True,
            ),
            "open": [10.0, 10.1, 11.0, 99.0],
            "high": [10.2, 10.3, 11.2, 100.0],
            "low": [9.9, 10.0, 10.9, 98.0],
            "close": [10.1, 10.2, 11.1, 99.5],
            "volume": [100.0, 110.0, 120.0, 999.0],
        }
    ).to_parquet(source, index=False)
    primary = tmp_path / "primary"
    exact = tmp_path / "exact_rerun"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(exact),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout.strip() == "blocked_one_minute_history_unavailable"
    coverage = pd.read_csv(primary / "one_minute_availability_audit.csv")
    first = coverage.loc[coverage["symbol"].eq("AAL") & coverage["session"].eq("2024-01-02")].iloc[
        0
    ]
    assert first["source_status"] == "partial"
    assert first["bar_start_candidate_observed_minute_count"] == 2
    assert first["bar_start_candidate_observed_minute_ordinals"] == "0-1"
    assert first["bar_start_candidate_missing_minute_count"] == 388
    assert first["bar_start_candidate_missing_minute_ordinals"] == "2-389"

    manifest = json.loads((primary / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["one_minute_rows_materialised"] == 3
    assert manifest["complete_symbol_sessions"] == 0
    assert manifest["sources_present"] == 1
    audit = json.loads((primary / "independent_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["checks"]["availability"]["one_minute_rows_materialised"] == 3
    assert audit["checks"]["availability"]["source_files_present"] == 1


def test_availability_gate_accepts_a_complete_bar_end_candidate_session(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    source = provider_root / "symbol=AAL" / "timeframe=1m" / "data.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-02T14:31:00Z", periods=390, freq="1min", tz="UTC")}
    ).to_parquet(source, index=False)
    primary = tmp_path / "primary"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(tmp_path / "exact_rerun"),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    coverage = pd.read_csv(primary / "one_minute_availability_audit.csv", keep_default_na=False)
    session = coverage.loc[
        coverage["symbol"].eq("AAL") & coverage["session"].eq("2024-01-02")
    ].iloc[0]
    assert session["source_status"] == "complete_under_bar_end_candidate"
    assert session["bar_start_candidate_observed_minute_count"] == 389
    assert session["bar_start_candidate_missing_minute_ordinals"] == "0"
    assert session["bar_end_candidate_observed_minute_count"] == 390
    assert session["bar_end_candidate_missing_minute_ordinals"] == ""
    manifest = json.loads((primary / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete_symbol_sessions"] == 1
    assert manifest["bar_end_candidate_complete_symbol_sessions"] == 1
    assert manifest["availability_gate_passed"] is False
    audit = json.loads((primary / "independent_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True


def test_unreadable_present_source_remains_a_history_unavailable_blocker(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    source = provider_root / "symbol=AAL" / "timeframe=1m" / "data.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"wrong_column": [1]}).to_parquet(source, index=False)
    primary = tmp_path / "primary"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(tmp_path / "exact_rerun"),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout.strip() == "blocked_one_minute_history_unavailable"
    coverage = pd.read_csv(primary / "one_minute_availability_audit.csv")
    assert set(coverage.loc[coverage["symbol"].eq("AAL"), "source_status"]) == {"unreadable_source"}
    hashes = json.loads((primary / "input_artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["one_minute_source_artifacts_hashed"] == 0
    audit = json.loads((primary / "independent_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True


def test_runner_completes_an_independent_audit_for_both_exact_runs(tmp_path: Path) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    primary = tmp_path / "primary"
    exact = tmp_path / "exact_rerun"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(exact),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    primary_audit = json.loads((primary / "independent_audit.json").read_text(encoding="utf-8"))
    exact_audit = json.loads((exact / "independent_audit.json").read_text(encoding="utf-8"))
    assert primary_audit == exact_audit
    assert primary_audit["passed"] is True
    assert primary_audit["auditor_imported_runner"] is False
    assert primary_audit["checks"]["availability"]["rows_verified"] == 8_240
    assert primary_audit["checks"]["availability"]["one_minute_rows_materialised"] == 0
    assert primary_audit["checks"]["frozen_population"]["assessment_rows"] == 1_560
    assert primary_audit["checks"]["protected_boundary"]["protected_rows_opened"] == 0
    assert primary_audit["checks"]["downstream_not_opened"]["models_fitted"] == 0

    rerun = json.loads((primary / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    assert rerun["passed"] is True
    assert rerun["independent_audit_status"] == "passed"
    assert rerun["independent_audit_sha256"]


def test_independent_auditor_fails_closed_after_artifact_tampering(tmp_path: Path) -> None:
    provider_root = tmp_path / "processed" / "source=eodhd" / "instrument_type=stock"
    primary = tmp_path / "primary"
    exact = tmp_path / "exact_rerun"
    run = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--provider-root",
            str(provider_root),
            "--primary-output",
            str(primary),
            "--exact-output",
            str(exact),
            "--report-output",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    (primary / "report.md").write_text("tampered\n", encoding="utf-8")

    audit = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--artifacts",
            str(primary),
            "--provider-root",
            str(provider_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert audit.returncode == 1
    failure = json.loads((primary / "independent_audit.json").read_text(encoding="utf-8"))
    assert failure["passed"] is False
    assert failure["decision"] == "blocked_reproducibility_or_audit_failure"
