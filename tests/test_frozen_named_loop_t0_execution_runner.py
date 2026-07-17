from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

WORK = Path("research/slrno-v2/20260714-regime-loop-handoff/work")
RUNNER_PATH = WORK / "run_frozen_named_loop_t0_execution_reference_v1.py"
AUDITOR_PATH = WORK / "audit_frozen_named_loop_t0_execution_v1.py"


def load_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("frozen_t0_runner", RUNNER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    module = load_module()
    root = tmp_path_factory.mktemp("frozen-t0-reference")
    primary = root / "primary"
    exact = root / "exact"
    report = root / "report.md"
    module.run_reference(output=primary, report_path=report, exact_rerun_of=None)
    module.run_reference(output=exact, report_path=None, exact_rerun_of=primary)
    return primary, exact, report


def test_reference_runner_generates_required_machine_readable_artifacts(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, _, _ = run_outputs
    expected = {
        "frozen_experiment_contract.json",
        "frozen_pair_to_family_mapping.json",
        "historical_named_reference_ledger.parquet",
        "historical_control_reference_ledger.parquet",
        "trigger_reconstruction_ledger.parquet",
        "fill_evidence_classification_ledger.parquet",
        "payoff_envelope_ledger.parquet",
        "historical_reference_metrics.csv",
        "prospective_opportunity_ledger.parquet",
        "prospective_trigger_fill_append_ledger.parquet",
        "prospective_outcome_settlement_ledger.parquet",
        "data_quality_report.json",
        "named_family_metrics.csv",
        "control_family_metrics.csv",
        "named_versus_control_comparisons.csv",
        "execution_decay_curve.csv",
        "break_even_slippage_results.csv",
        "session_block_intervals.csv",
        "stock_and_month_breakdowns.csv",
        "null_test_results.csv",
        "concentration_results.csv",
        "missing_2023_archival_report.json",
        "prospective_completion_status.json",
        "run_metadata.json",
        "artifact_manifest.json",
    }

    assert expected <= {path.name for path in primary.iterdir() if path.is_file()}


def test_primary_and_exact_rerun_outputs_are_byte_identical(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, exact, _ = run_outputs
    exclusions = {"exact_rerun_identity.json", "independent_audit.json"}
    primary_files = {
        path.relative_to(primary): path
        for path in primary.rglob("*")
        if path.is_file() and path.name not in exclusions
    }
    exact_files = {
        path.relative_to(exact): path
        for path in exact.rglob("*")
        if path.is_file() and path.name not in exclusions
    }

    assert set(primary_files) == set(exact_files)
    assert all(
        primary_files[name].read_bytes() == exact_files[name].read_bytes() for name in primary_files
    )


def test_reference_artifacts_preserve_named_control_and_fill_separation(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, _, _ = run_outputs
    named = pd.read_csv(primary / "named_family_metrics.csv")
    controls = pd.read_csv(primary / "control_family_metrics.csv")
    evidence = pd.read_parquet(primary / "fill_evidence_classification_ledger.parquet")

    assert set(named["family"]) == {"cycle_04|state_4", "cycle_07|state_5"}
    assert set(controls["family"]) == {"cycle_04|state_2", "cycle_07|state_6"}
    assert set(evidence["fill_evidence_classification"]) == {
        "BOUNDED_BUT_NOT_EXACT",
        "GAP_FILL_OBSERVABLE",
    }
    assert not named["family"].isin(set(controls["family"])).any()


def test_incomplete_prospective_status_is_blinded_and_safe(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, _, _ = run_outputs
    status = json.loads((primary / "prospective_completion_status.json").read_text())
    flattened = json.dumps(status).lower()

    assert status["prospective_decision"] == "prospective_sample_incomplete"
    assert status["completion_rule_reached"] is False
    assert "payoff" not in flattened
    assert "profit" not in flattened
    assert "f10_" not in flattened


def test_timestamp_shift_falsification_never_replaces_primary_clock(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, _, _ = run_outputs
    nulls = pd.read_csv(primary / "null_test_results.csv")
    shifted = nulls.loc[nulls["null_name"].eq("timestamp_shift_plus_one_bar")]

    assert not shifted.empty
    assert shifted["may_replace_primary_clock"].eq(False).all()  # noqa: E712


def test_scientific_report_uses_allowed_historical_and_prospective_labels(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    _, _, report = run_outputs
    text = report.read_text()

    assert "reference_fill_not_provably_executable" in text
    assert "prospective_sample_incomplete" in text
    assert "No order was placed" in text


def test_independent_auditor_reconstructs_every_fill_and_aggregate(
    run_outputs: tuple[Path, Path, Path],
) -> None:
    primary, exact, _ = run_outputs
    specification = importlib.util.spec_from_file_location("frozen_t0_auditor", AUDITOR_PATH)
    assert specification is not None and specification.loader is not None
    auditor = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(auditor)
    output = primary / "independent_audit.json"
    result = auditor.run_audit(primary, exact, output)

    assert result["passed"] is True
    assert (
        "5,555 fills" in result["checks"]["signal_trigger_fill_terminal_cost_and_payoff"]["detail"]
    )
    assert result["checks"]["named_control_metrics_and_comparisons"]["passed"] is True
    assert result["checks"]["research_only_changed_paths"]["passed"] is True
