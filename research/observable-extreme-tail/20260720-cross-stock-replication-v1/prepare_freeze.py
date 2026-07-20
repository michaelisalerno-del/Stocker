"""Prepare the outcome-exposure freeze before any assessment outcome is opened."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stocker_research.observable_extreme_tail_replication_v1 import (
    BLOCKED_NO_CLEAN_HOLDOUT,
    DEVELOPMENT_SYMBOLS,
    PREDECESSOR_EXPERIMENT,
    SAFETY_FLAGS,
    SafeStockSource,
    build_stock_outcome_exposure_ledger,
    canonical_json,
    discover_outcome_evidence,
    discover_safe_stock_sources,
    exposure_gate_decision,
    scan_report_corpus,
    sha256_file,
    stock_exposure_evidence,
    validate_symbol_disjointness,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "freeze"
DEFAULT_DATA_ROOT = Path.home() / "StockerLocal" / "data"
MODULE_LOGICAL_PATH = (
    "packages/stocker_research/src/stocker_research/observable_extreme_tail_replication_v1.py"
)
PREPARE_LOGICAL_PATH = (
    "research/observable-extreme-tail/20260720-cross-stock-replication-v1/prepare_freeze.py"
)

LEDGER_FIELDS = (
    "symbol",
    "present_in_safe_raw_manifest",
    "raw_file_hash",
    "data_qa_only",
    "used_in_observable_model_training",
    "used_in_feature_selection",
    "used_in_threshold_selection",
    "used_in_directional_outcome_analysis",
    "used_in_post_hoc_tail_inspection",
    "used_in_any_price_or_payoff_report",
    "used_only_in_structural_nonprice_research",
    "exposure_evidence_paths",
    "exposure_status",
    "eligible_for_cross_stock_holdout",
    "exclusion_reason",
    "vendor_qa_status",
    "vendor_qa_issue_codes",
    "bar_audit_passed",
    "bar_audit_issue_codes",
    "safe_raw_file",
)

DOWNSTREAM_NOT_PRODUCED = (
    "frozen_candidate_contract.json",
    "feature_manifest.json",
    "development_oof_slate_maxima.parquet",
    "admission_threshold.json",
    "assessment_decision_panel.parquet",
    "assessment_predictions.parquet",
    "selection_ledger.parquet",
    "baseline_selection_ledger.parquet",
    "primary_metrics.csv",
    "monthly_metrics.csv",
    "clock_metrics.csv",
    "bootstrap_metrics.csv",
    "permutation_null_metrics.csv",
    "leave_one_stock_out_metrics.csv",
    "best_session_deletion_metrics.csv",
    "concentration_metrics.csv",
    "timing_sensitivity.csv",
    "friction_sensitivity.csv",
)

FREEZE_OUTPUT_FILES = (
    "assessment_symbols.txt",
    "decision.json",
    "development_symbols.txt",
    "excluded_symbols.txt",
    "exposure_evidence_manifest.json",
    "input_artifact_hashes.json",
    "not_produced_artifacts.json",
    "predecessor_reconstruction_metrics.json",
    "protected_boundary_audit.json",
    "report.md",
    "source_manifest.json",
    "stock_outcome_exposure_ledger.csv",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _write_symbols(path: Path, symbols: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{symbol}\n" for symbol in sorted(symbols))
    path.write_text(content, encoding="utf-8")


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in LEDGER_FIELDS})


def _resolve_logical_path(logical_path: str, data_root: Path) -> Path:
    container_path = logical_path.split("!", maxsplit=1)[0]
    if container_path == "data/reports/research":
        return data_root / "reports" / "research"
    if container_path.startswith("data/"):
        return data_root / container_path.removeprefix("data/")
    return REPO_ROOT / container_path


def _input_hashes(
    sources: list[SafeStockSource],
    evidence_paths: set[str],
    data_root: Path,
) -> dict[str, str]:
    paths = {
        *(source.raw_file_logical_path for source in sources),
        *(source.vendor_qa_logical_path for source in sources),
        *(source.bar_audit_logical_path for source in sources),
        *(path.split("!", maxsplit=1)[0] for path in evidence_paths),
        MODULE_LOGICAL_PATH,
        PREPARE_LOGICAL_PATH,
    }
    result: dict[str, str] = {}
    for logical_path in sorted(paths):
        path = _resolve_logical_path(logical_path, data_root)
        if path.is_file():
            result[logical_path] = sha256_file(path)
    return result


def _load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise RuntimeError(f"contract_safety_flag_mismatch:{key}")
    return contract


def _report_text(summary: dict[str, Any]) -> str:
    clean = ", ".join(summary["clean_outcome_unexposed_symbols"]) or "none"
    eligible = ", ".join(summary["eligible_assessment_symbols"]) or "none"
    return f"""# Observable Extreme-Tail Cross-Stock Replication V1

## Decision

`{summary["decision"]}`

The mandatory stock outcome-exposure gate stopped the experiment before model
reconstruction and before any assessment market outcome was opened. The actual safe
EODHD stock manifest contains {summary["safe_raw_universe_count"]} symbols;
{summary["outcome_exposed_count"]} are outcome-exposed, {summary["unknown_count"]} are
unknown-assume-exposed, and only {summary["clean_outcome_unexposed_count"]} is
machine-evidenced as outcome-unexposed. The contract requires at least 15.

Outcome-unexposed symbols: {clean}.

QA-eligible cross-stock assessment symbols: {eligible}. The clean symbol is excluded
when its existing vendor QA or bar audit has not passed.

## Boundary

No raw market row was parsed or materialised. Assessment and development model rows
are both zero, the minimum/maximum market timestamps read are null, protected files
touched are zero, and `protected_rows_materialised=0`.

## Scientific scope

This is a retrospective, research-only blocker result. No candidate prediction,
admission threshold, slate, delayed return, baseline, bootstrap, permutation null,
execution result, or deployable edge was calculated. The predecessor was used only
to establish model/outcome exposure of its development stocks; reconstruction was
not reached because Phase A failed first.
"""


def prepare_freeze(
    *,
    data_root: Path,
    output: Path,
    max_symbols: int | None = None,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    """Create the deterministic Phase A freeze and apply its fail-closed gate."""

    _load_contract()
    smoke = max_symbols is not None or max_sessions is not None
    sources = discover_safe_stock_sources(data_root)
    if max_symbols is not None:
        if max_symbols < 1:
            raise ValueError("max_symbols_must_be_positive")
        sources = sources[:max_symbols]
    symbols = [source.symbol for source in sources]
    report_root = data_root / "reports" / "research"
    report_scan = scan_report_corpus(report_root, symbols)
    discovered = discover_outcome_evidence(report_root, symbols)
    clean_scan_symbols = {
        symbol for symbol, paths in report_scan.symbol_mention_paths.items() if not paths
    }
    ambiguous_report_mentions = {
        symbol: paths
        for symbol, paths in report_scan.symbol_mention_paths.items()
        if paths and not discovered.get(symbol) and symbol not in DEVELOPMENT_SYMBOLS
    }
    ledger = build_stock_outcome_exposure_ledger(
        sources,
        discovered,
        clean_scan_symbols=clean_scan_symbols,
        ambiguous_report_mentions=ambiguous_report_mentions,
    )
    evidence_by_symbol = {
        source.symbol: stock_exposure_evidence(
            source,
            discovered,
            clean_scan_symbols=clean_scan_symbols,
            ambiguous_report_mentions=ambiguous_report_mentions,
        )
        for source in sources
    }
    evidence_paths = {item.logical_path for items in evidence_by_symbol.values() for item in items}
    clean_symbols = sorted(
        str(row["symbol"]) for row in ledger if str(row["exposure_status"]).startswith("clean_")
    )
    eligible_symbols = sorted(
        str(row["symbol"]) for row in ledger if bool(row["eligible_for_cross_stock_holdout"])
    )
    excluded_symbols = sorted(set(symbols) - set(eligible_symbols))
    validate_symbol_disjointness(DEVELOPMENT_SYMBOLS, eligible_symbols)
    decision = exposure_gate_decision(len(clean_symbols))
    if decision is None:
        raise RuntimeError("phase_a_gate_passed_runner_not_invoked")
    if decision != BLOCKED_NO_CLEAN_HOLDOUT:
        raise RuntimeError("unexpected_phase_a_decision")

    exposure_counts = Counter(str(row["exposure_status"]) for row in ledger)
    summary: dict[str, Any] = {
        **SAFETY_FLAGS,
        "non_scientific_smoke_test": smoke,
        "decision": decision,
        "safe_raw_universe_count": len(ledger),
        "outcome_exposed_count": exposure_counts["outcome_exposed"],
        "clean_outcome_unexposed_count": len(clean_symbols),
        "clean_outcome_unexposed_symbols": clean_symbols,
        "unknown_count": exposure_counts["unknown_assume_exposed"],
        "unknown_symbols": sorted(
            str(row["symbol"])
            for row in ledger
            if row["exposure_status"] == "unknown_assume_exposed"
        ),
        "eligible_assessment_count": len(eligible_symbols),
        "eligible_assessment_symbols": eligible_symbols,
        "excluded_symbols": excluded_symbols,
        "required_clean_stocks": 15,
        "development_rows_materialised": 0,
        "assessment_rows_materialised": 0,
        "assessment_outcomes_read": False,
        "protected_rows_materialised": 0,
        "phase_reached": "phase_a_stock_outcome_exposure_ledger",
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_ledger(output / "stock_outcome_exposure_ledger.csv", ledger)
    _write_symbols(output / "development_symbols.txt", DEVELOPMENT_SYMBOLS)
    _write_symbols(output / "assessment_symbols.txt", eligible_symbols)
    _write_symbols(output / "excluded_symbols.txt", excluded_symbols)

    source_manifest = {
        **SAFETY_FLAGS,
        "non_scientific_smoke_test": smoke,
        "source": ("EODHD intraday pre-boundary raw-file manifest enriched by existing vendor QA"),
        "instrument_type": "stock",
        "timeframe": "5m",
        "safe_raw_universe_count": len(sources),
        "raw_market_rows_parsed": 0,
        "sources": [
            {
                "symbol": source.symbol,
                "safe_raw_file": source.raw_file_logical_path,
                "safe_raw_file_sha256": source.raw_file_sha256,
                "vendor_qa_file": source.vendor_qa_logical_path,
                "vendor_qa_status": source.vendor_qa_status,
                "vendor_qa_issue_codes": list(source.vendor_qa_issue_codes),
                "bar_audit_file": source.bar_audit_logical_path,
                "bar_audit_passed": source.bar_audit_passed,
                "bar_audit_issue_codes": list(source.bar_audit_issue_codes),
            }
            for source in sources
        ],
    }
    _write_json(output / "source_manifest.json", source_manifest)
    _write_json(
        output / "exposure_evidence_manifest.json",
        {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "classification_is_conservative": True,
            "unresolved_status": "unknown_assume_exposed",
            "report_corpus_root": "data/reports/research",
            "corpus_manifest_sha256": report_scan.corpus_manifest_sha256,
            "corpus_files_hashed": report_scan.files_hashed,
            "corpus_bytes_hashed": report_scan.bytes_hashed,
            "text_files_scanned": report_scan.text_files_scanned,
            "zip_archives_scanned": report_scan.zip_archives_scanned,
            "zip_members_scanned": report_scan.zip_members_scanned,
            "ignored_incomplete_archive_parts": list(report_scan.ignored_incomplete_archive_parts),
            "ignored_part_policy": (
                "hash incomplete ZIP parts but do not parse them when the complete archive exists"
            ),
            "symbols": {
                symbol: {
                    "report_mention_paths": list(report_scan.symbol_mention_paths.get(symbol, ())),
                    "classification_evidence": [
                        {
                            "logical_path": item.logical_path,
                            "kind": item.kind.value,
                            "detail": item.detail,
                        }
                        for item in evidence_by_symbol[symbol]
                    ],
                }
                for symbol in sorted(symbols)
            },
        },
    )
    _write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "development_allowed": "2024-01-01 through 2024-12-31",
            "assessment_allowed": "2025-01-01 through 2025-08-22 inclusive",
            "protected_start": "2025-08-23T00:00:00Z",
            "minimum_market_timestamp_read": None,
            "maximum_market_timestamp_read": None,
            "rows_by_symbol_year_month": {},
            "safe_source_file_hashes": {
                source.raw_file_logical_path: source.raw_file_sha256 for source in sources
            },
            "protected_files_touched": [],
            "protected_files_touched_count": 0,
            "raw_market_rows_parsed": 0,
            "market_rows_materialised": 0,
            "protected_rows_materialised": 0,
            "assessment_outcomes_read": False,
            "result": "passed_no_market_rows_opened_before_phase_a_blocker",
        },
    )
    input_hashes = _input_hashes(sources, evidence_paths, data_root)
    _write_json(output / "input_artifact_hashes.json", input_hashes)
    predecessor_paths = sorted(
        path for path in input_hashes if path.startswith(f"{PREDECESSOR_EXPERIMENT}/")
    )
    _write_json(
        output / "predecessor_reconstruction_metrics.json",
        {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "source_experiment": PREDECESSOR_EXPERIMENT,
            "candidate_models": ["P1", "P1_SIZE", "D0"],
            "candidate_score": (
                "p_move * (2 * p_up_given_move_observable - 1) * predicted_absolute_movement_bps"
            ),
            "required_tolerance": 1e-12,
            "artifacts_used_for_exposure_lineage_only": {
                path: input_hashes[path] for path in predecessor_paths
            },
            "predictions_reconstructed": False,
            "reference_rows_scored": 0,
            "result": "not_attempted_phase_a_blocker_precedes_reconstruction",
        },
    )
    _write_json(output / "decision.json", summary)
    _write_json(
        output / "not_produced_artifacts.json",
        {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "decision": decision,
            "reason": "mandatory Phase A stop before candidate reconstruction or outcomes",
            "artifacts": list(DOWNSTREAM_NOT_PRODUCED),
        },
    )
    (output / "report.md").write_text(_report_text(summary), encoding="utf-8")

    freeze_files = [output / name for name in FREEZE_OUTPUT_FILES]
    _write_json(
        output / "freeze_manifest.json",
        {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "decision": decision,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "artifacts": {path.name: sha256_file(path) for path in freeze_files},
            "cohort_file_hashes": {
                name: sha256_file(output / name)
                for name in (
                    "development_symbols.txt",
                    "assessment_symbols.txt",
                    "excluded_symbols.txt",
                )
            },
            "assessment_outcomes_read_before_freeze": False,
            "frozen_before_assessment_outcomes": True,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--max-sessions", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_freeze(
        data_root=args.data_root.expanduser().resolve(),
        output=args.output.resolve(),
        max_symbols=args.max_symbols,
        max_sessions=args.max_sessions,
    )
    print(canonical_json(result), end="")


if __name__ == "__main__":
    main()
