from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from stocker_research.observable_extreme_tail_replication_v1 import (
    EvidenceKind,
    ExposureEvidence,
    SafeStockSource,
    build_stock_outcome_exposure_ledger,
    canonical_stock_symbol,
    classify_stock_exposure,
    discover_outcome_evidence,
    discover_safe_stock_sources,
    exposure_gate_decision,
    find_forbidden_fields,
    scan_report_corpus,
    validate_market_timestamp,
    validate_symbol_disjointness,
)


def test_outcome_evidence_marks_stock_exposed() -> None:
    result = classify_stock_exposure(
        "TEST",
        [
            ExposureEvidence(
                logical_path="data/reports/research/test_payoff.csv",
                kind=EvidenceKind.PRICE_OR_PAYOFF_REPORT,
                detail="symbol column plus forward_24_bar_return",
            )
        ],
    )

    assert result.exposure_status == "outcome_exposed"
    assert result.eligible_for_cross_stock_holdout is False


def test_unknown_exposure_is_assumed_exposed() -> None:
    result = classify_stock_exposure("TEST", [])

    assert result.exposure_status == "unknown_assume_exposed"
    assert result.eligible_for_cross_stock_holdout is False
    assert result.exclusion_reason == "exposure_evidence_unresolved"


def test_qa_only_without_completed_corpus_scan_remains_unknown() -> None:
    result = classify_stock_exposure(
        "TEST",
        [
            ExposureEvidence(
                logical_path="data/reports/vendor_qa/TEST_5m_eodhd_qa.json",
                kind=EvidenceKind.DATA_QA_ONLY,
                detail="schema and vendor QA only",
            )
        ],
    )

    assert result.exposure_status == "unknown_assume_exposed"
    assert result.eligible_for_cross_stock_holdout is False


def test_ambiguous_report_mention_remains_unknown_and_preserves_path() -> None:
    source = SafeStockSource(
        symbol="ECHO",
        vendor_qa_logical_path="data/reports/vendor_qa/ECHO_5m_eodhd_qa.json",
        bar_audit_logical_path="data/reports/audits/ECHO_5m_audit.json",
        raw_file_logical_path="data/raw/ECHO/2024-01-01_2025-08-23.json",
        raw_file_sha256="abc",
        vendor_qa_status="missing",
        vendor_qa_issue_codes=("missing_vendor_qa",),
        bar_audit_passed=False,
        bar_audit_issue_codes=("missing_bar_audit",),
    )
    ambiguous_path = "data/reports/research/summary.md"

    [row] = build_stock_outcome_exposure_ledger(
        [source],
        {"ECHO": []},
        ambiguous_report_mentions={"ECHO": (ambiguous_path,)},
    )

    assert row["exposure_status"] == "unknown_assume_exposed"
    assert row["eligible_for_cross_stock_holdout"] is False
    assert ambiguous_path in str(row["exposure_evidence_paths"])


def test_machine_resolved_structural_only_research_is_not_outcome_exposure() -> None:
    result = classify_stock_exposure(
        "TEST",
        [
            ExposureEvidence(
                logical_path="data/reports/research/structural_states.json",
                kind=EvidenceKind.STRUCTURAL_NONPRICE_RESEARCH,
                detail="state-count schema without future price or payoff target",
            )
        ],
    )

    assert result.exposure_status == "clean_structural_nonprice_only"
    assert result.eligible_for_cross_stock_holdout is True


def test_development_and_assessment_stock_aliases_must_be_disjoint() -> None:
    assert canonical_stock_symbol("amat.us") == "AMAT"

    try:
        validate_symbol_disjointness(["AMAT"], ["AMAT.US"])
    except ValueError as error:
        assert str(error) == "development_assessment_symbol_overlap:AMAT"
    else:
        raise AssertionError("stock aliases were allowed to cross the holdout boundary")


def test_protected_dates_are_rejected() -> None:
    validate_market_timestamp("2025-08-22T23:59:59+00:00")

    for protected in ("2025-08-23T00:00:00+00:00", "2026-01-02T14:30:00+00:00"):
        try:
            validate_market_timestamp(protected)
        except ValueError as error:
            assert str(error) == "blocked_protected_boundary_failure"
        else:
            raise AssertionError(f"protected timestamp was accepted: {protected}")


def test_clean_holdout_support_gate_is_not_lowered() -> None:
    assert exposure_gate_decision(14) == "blocked_no_clean_cross_stock_holdout_remaining"
    assert exposure_gate_decision(15) is None


def test_forbidden_structural_and_broker_fields_are_detected() -> None:
    fields = [
        "observable_chain_score",
        "loop_probability",
        "regime_id",
        "closure_probability",
        "broker_order_id",
        "payoff_history",
    ]

    assert find_forbidden_fields(fields) == [
        "broker_order_id",
        "closure_probability",
        "loop_probability",
        "payoff_history",
        "regime_id",
    ]


def test_individual_payoff_report_is_discovered_through_vendor_alias(tmp_path: Path) -> None:
    report_root = tmp_path / "data" / "reports" / "research"
    report_root.mkdir(parents=True)
    report = report_root / "20260101_example_AMAT.US_5m.json"
    report.write_text(
        json.dumps({"symbol": "AMAT.US", "net_return": 0.01}),
        encoding="utf-8",
    )

    evidence = discover_outcome_evidence(report_root, ["AMAT", "CRCL"])

    assert [item.logical_path for item in evidence["AMAT"]] == [
        "data/reports/research/20260101_example_AMAT.US_5m.json"
    ]
    assert evidence["CRCL"] == []


def test_aggregate_symbol_ledger_with_forward_outcome_is_discovered(tmp_path: Path) -> None:
    report_root = tmp_path / "data" / "reports" / "research"
    report_root.mkdir(parents=True)
    ledger = report_root / "trade_outcomes.csv"
    ledger.write_text(
        "symbol,forward_24_bar_return,net_r\nBMNR,0.01,0.1\n",
        encoding="utf-8",
    )

    evidence = discover_outcome_evidence(report_root, ["BMNR", "CRCL"])

    assert [item.logical_path for item in evidence["BMNR"]] == [
        "data/reports/research/trade_outcomes.csv"
    ]
    assert evidence["CRCL"] == []


def test_safe_source_manifest_hashes_only_pre_boundary_raw_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    qa_root = data_root / "reports" / "vendor_qa"
    audit_root = data_root / "reports" / "audits"
    raw_file = (
        data_root
        / "raw"
        / "source=eodhd"
        / "endpoint=intraday"
        / "symbol=TEST"
        / "interval=5m"
        / "2024-01-01_2025-08-23.json"
    )
    qa_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("[]\n", encoding="utf-8")
    (qa_root / "TEST_5m_eodhd_qa.json").write_text(
        json.dumps(
            {
                "symbol": "TEST",
                "status": "pass",
                "issue_codes": [],
                "raw_files": {"first_raw_file": str(raw_file)},
            }
        ),
        encoding="utf-8",
    )
    (audit_root / "TEST_5m_audit.json").write_text(
        json.dumps({"symbol": "TEST", "passed": True, "issues": []}),
        encoding="utf-8",
    )

    [source] = discover_safe_stock_sources(data_root)

    assert source.symbol == "TEST"
    assert source.raw_file_logical_path.endswith("2024-01-01_2025-08-23.json")
    assert source.raw_file_sha256 == (
        "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
    )
    assert source.vendor_qa_status == "pass"
    assert source.bar_audit_passed is True


def test_raw_manifest_symbol_without_qa_is_enumerated_and_fails_data_qa(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    raw_file = (
        data_root
        / "raw"
        / "source=eodhd"
        / "endpoint=intraday"
        / "symbol=ECHO"
        / "interval=5m"
        / "2024-01-01_2025-08-23.json"
    )
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("[]", encoding="utf-8")

    [source] = discover_safe_stock_sources(data_root)
    [row] = build_stock_outcome_exposure_ledger([source], {"ECHO": []}, clean_scan_symbols={"ECHO"})

    assert source.symbol == "ECHO"
    assert source.vendor_qa_status == "missing"
    assert source.vendor_qa_issue_codes == ("missing_vendor_qa",)
    assert source.bar_audit_passed is False
    assert source.bar_audit_issue_codes == ("missing_bar_audit",)
    assert row["exposure_status"] == "clean_data_qa_only"
    assert row["eligible_for_cross_stock_holdout"] is False


def test_clean_exposure_and_data_eligibility_remain_separate() -> None:
    source = SafeStockSource(
        symbol="CRCL",
        vendor_qa_logical_path="data/reports/vendor_qa/CRCL_5m_eodhd_qa.json",
        bar_audit_logical_path="data/reports/audits/CRCL_5m_audit.json",
        raw_file_logical_path="data/raw/CRCL/2024-01-01_2025-08-23.json",
        raw_file_sha256="abc",
        vendor_qa_status="warning",
        vendor_qa_issue_codes=("validation_warnings",),
        bar_audit_passed=False,
        bar_audit_issue_codes=("large_price_jump",),
    )

    [row] = build_stock_outcome_exposure_ledger([source], {"CRCL": []}, clean_scan_symbols={"CRCL"})

    assert row["exposure_status"] == "clean_data_qa_only"
    assert row["data_qa_only"] is True
    assert row["eligible_for_cross_stock_holdout"] is False
    assert row["exclusion_reason"] == "vendor_qa_or_bar_audit_not_passed"


def test_report_corpus_scan_covers_text_and_complete_zip_members(tmp_path: Path) -> None:
    report_root = tmp_path / "data" / "reports" / "research"
    report_root.mkdir(parents=True)
    (report_root / "summary.md").write_text("AMAT had a future return.\n", encoding="utf-8")
    with zipfile.ZipFile(report_root / "archived.zip", "w") as archive:
        archive.writestr("nested/outcomes.csv", "symbol,net_return\nBMNR,0.1\n")
    (report_root / "archived.zip.part-000").write_bytes(b"incomplete")

    result = scan_report_corpus(report_root, ["AMAT", "BMNR", "CRCL"])

    assert result.text_files_scanned == 1
    assert result.zip_archives_scanned == 1
    assert result.zip_members_scanned == 1
    assert result.symbol_mention_paths["AMAT"] == ("data/reports/research/summary.md",)
    assert result.symbol_mention_paths["BMNR"] == (
        "data/reports/research/archived.zip!nested/outcomes.csv",
    )
    assert result.symbol_mention_paths["CRCL"] == ()
    assert result.ignored_incomplete_archive_parts == (
        "data/reports/research/archived.zip.part-000",
    )


def _write_synthetic_safe_source(data_root: Path) -> None:
    qa_root = data_root / "reports" / "vendor_qa"
    audit_root = data_root / "reports" / "audits"
    report_root = data_root / "reports" / "research"
    raw_file = (
        data_root
        / "raw"
        / "source=eodhd"
        / "endpoint=intraday"
        / "symbol=TEST"
        / "interval=5m"
        / "2024-01-01_2025-08-23.json"
    )
    qa_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("[]\n", encoding="utf-8")
    (qa_root / "TEST_5m_eodhd_qa.json").write_text(
        json.dumps(
            {
                "symbol": "TEST",
                "status": "pass",
                "issue_codes": [],
                "raw_files": {"first_raw_file": str(raw_file)},
            }
        ),
        encoding="utf-8",
    )
    (audit_root / "TEST_5m_audit.json").write_text(
        json.dumps({"symbol": "TEST", "passed": True, "issues": []}),
        encoding="utf-8",
    )


def _run_synthetic_freeze(data_root: Path, output: Path) -> None:
    experiment_dir = Path(__file__).resolve().parents[1]
    repo_root = experiment_dir.parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "packages" / "stocker_research" / "src"),
    }
    subprocess.run(
        [
            sys.executable,
            str(experiment_dir / "prepare_freeze.py"),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_phase_a_freeze_blocks_before_materialising_market_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "freeze"
    _write_synthetic_safe_source(data_root)

    _run_synthetic_freeze(data_root, output)

    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    boundary = json.loads((output / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "blocked_no_clean_cross_stock_holdout_remaining"
    assert decision["assessment_outcomes_read"] is False
    assert boundary["minimum_market_timestamp_read"] is None
    assert boundary["maximum_market_timestamp_read"] is None
    assert boundary["protected_rows_materialised"] == 0


def test_phase_a_freeze_is_byte_reproducible(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_synthetic_safe_source(data_root)

    _run_synthetic_freeze(data_root, first)
    _run_synthetic_freeze(data_root, second)

    manifest = json.loads((first / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name in (*manifest["artifacts"], "freeze_manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_smoke_exact_rerun_uses_sibling_output_and_is_labelled(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "smoke_primary"
    experiment_dir = Path(__file__).resolve().parents[1]
    repo_root = experiment_dir.parents[2]
    _write_synthetic_safe_source(data_root)

    subprocess.run(
        [
            sys.executable,
            str(experiment_dir / "run_replication.py"),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--max-symbols",
            "1",
            "--exact-rerun",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root / "packages" / "stocker_research" / "src"),
        },
    )

    exact_output = tmp_path / "smoke_primary_exact_rerun"
    primary_decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    exact_decision = json.loads((exact_output / "decision.json").read_text(encoding="utf-8"))
    assert primary_decision["non_scientific_smoke_test"] is True
    assert exact_decision["non_scientific_smoke_test"] is True
    assert exact_output.is_dir()
