"""Independently audit the fail-closed cross-stock replication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_DATA_ROOT = Path.home() / "StockerLocal" / "data"

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "cross_stock_holdout": True,
    "forward_time_assessment": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_and_structural_paths_forbidden": True,
}
DEVELOPMENT_SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt", ".yaml", ".yml"})
OUTCOME_MARKERS = (
    b"net_return",
    b"gross_return",
    b"forward_",
    b"raw_return",
    b"net_r",
    b"gross_r",
    b"pnl",
    b"mfe",
    b"mae",
    b"entry_price",
    b"exit_price",
    b"target_hit",
    b"stop_hit",
)
CHUNK_BYTES = 1024 * 1024
FORBIDDEN_SCIENTIFIC_FIELD_FRAGMENTS = (
    "loop",
    "regime",
    "closure",
    "structural_path",
    "transition_burst",
    "excursion",
    "payoff_history",
    "broker",
    "account",
    "position",
    "order",
    "portfolio",
    "deployment",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_symbol(value: str) -> str:
    return value.strip().upper().removesuffix(".US")


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid_boolean:{value}")


def resolve_logical_path(logical_path: str, data_root: Path) -> Path:
    container = logical_path.split("!", maxsplit=1)[0]
    if container == "data/reports/research":
        return data_root / "reports" / "research"
    if container.startswith("data/"):
        return data_root / container.removeprefix("data/")
    return REPO_ROOT / container


def read_symbols(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Audit:
    checks: list[dict[str, object]] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(bool(item["passed"]) for item in self.checks)


def _stream_symbol_matches(handle: BinaryIO, patterns: dict[str, re.Pattern[bytes]]) -> set[str]:
    matches: set[str] = set()
    carry = b""
    while chunk := handle.read(CHUNK_BYTES):
        searchable = (carry + chunk).upper()
        for symbol, pattern in patterns.items():
            if pattern.search(searchable):
                matches.add(symbol)
        carry = searchable[-64:]
    return matches


def audit_report_corpus(
    report_root: Path,
    clean_symbols: list[str],
) -> dict[str, object]:
    patterns = {
        symbol: re.compile(
            rb"(?<![A-Z0-9])" + re.escape(symbol.encode()) + rb"(?:\.US)?(?![A-Z0-9])"
        )
        for symbol in clean_symbols
    }
    mentions: dict[str, list[str]] = {symbol: [] for symbol in clean_symbols}
    corpus_digest = hashlib.sha256()
    files_hashed = 0
    bytes_hashed = 0
    text_files = 0
    zip_archives = 0
    zip_members = 0
    ignored_parts: list[str] = []
    for path in sorted(candidate for candidate in report_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(report_root).as_posix()
        logical = f"data/reports/research/{relative}"
        digest = sha256_file(path)
        size = path.stat().st_size
        corpus_digest.update(f"{logical}\0{size}\0{digest}\n".encode())
        files_hashed += 1
        bytes_hashed += size
        if path.suffix.lower() in TEXT_SUFFIXES:
            with path.open("rb") as handle:
                found = _stream_symbol_matches(handle, patterns)
            for symbol in found:
                mentions[symbol].append(logical)
            text_files += 1
        elif path.suffix.lower() == ".zip" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    if member.is_dir() or Path(member.filename).suffix.lower() not in TEXT_SUFFIXES:
                        continue
                    with archive.open(member) as handle:
                        found = _stream_symbol_matches(handle, patterns)
                    for symbol in found:
                        mentions[symbol].append(f"{logical}!{member.filename}")
                    zip_members += 1
            zip_archives += 1
        elif ".zip.part-" in path.name:
            ignored_parts.append(logical)
    return {
        "corpus_manifest_sha256": corpus_digest.hexdigest(),
        "files_hashed": files_hashed,
        "bytes_hashed": bytes_hashed,
        "text_files_scanned": text_files,
        "zip_archives_scanned": zip_archives,
        "zip_members_scanned": zip_members,
        "ignored_incomplete_archive_parts": sorted(ignored_parts),
        "clean_symbol_mentions": {symbol: sorted(paths) for symbol, paths in mentions.items()},
    }


def path_contains_symbol_and_outcome(path: Path, symbol: str) -> bool:
    symbol_pattern = re.compile(
        rb"(?<![A-Z0-9])" + re.escape(symbol.encode()) + rb"(?:\.US)?(?![A-Z0-9])"
    )
    name_matches = symbol_pattern.search(path.name.upper().encode()) is not None
    carry = b""
    symbol_found = name_matches
    outcome_found = False
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            searchable = (carry + chunk).upper()
            symbol_found |= symbol_pattern.search(searchable) is not None
            outcome_found |= any(marker.upper() in searchable for marker in OUTCOME_MARKERS)
            carry = searchable[-64:]
    return symbol_found and outcome_found


def has_absolute_local_path(artifacts: Path) -> bool:
    for path in artifacts.iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            content = path.read_bytes()
            if b"/Users/" in content or b"file://" in content:
                return True
    return False


def run_audit(artifacts: Path, data_root: Path) -> dict[str, Any]:
    audit = Audit()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))
    evidence_manifest = json.loads(
        (artifacts / "exposure_evidence_manifest.json").read_text(encoding="utf-8")
    )
    boundary = json.loads((artifacts / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    reconstruction = json.loads(
        (artifacts / "predecessor_reconstruction_metrics.json").read_text(encoding="utf-8")
    )
    freeze_manifest = json.loads((artifacts / "freeze_manifest.json").read_text(encoding="utf-8"))
    not_produced = json.loads(
        (artifacts / "not_produced_artifacts.json").read_text(encoding="utf-8")
    )
    with (artifacts / "stock_outcome_exposure_ledger.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ledger = list(csv.DictReader(handle))

    for name, document in (
        ("contract", contract),
        ("decision", decision),
        ("source_manifest", source_manifest),
        ("exposure_evidence_manifest", evidence_manifest),
        ("protected_boundary_audit", boundary),
        ("predecessor_reconstruction_metrics", reconstruction),
        ("freeze_manifest", freeze_manifest),
        ("not_produced_artifacts", not_produced),
    ):
        audit.check(
            f"{name}_safety_flags",
            all(document.get(key) == value for key, value in SAFETY_FLAGS.items()),
        )

    artifact_hashes = freeze_manifest["artifacts"]
    audit.check(
        "freeze_artifact_hashes",
        all(
            (artifacts / name).is_file() and sha256_file(artifacts / name) == expected
            for name, expected in artifact_hashes.items()
        ),
    )
    audit.check(
        "contract_hash",
        freeze_manifest["contract_sha256"] == sha256_file(CONTRACT_PATH),
    )

    input_hashes = json.loads(
        (artifacts / "input_artifact_hashes.json").read_text(encoding="utf-8")
    )
    input_hashes_pass = True
    for logical_path, expected in input_hashes.items():
        resolved = resolve_logical_path(logical_path, data_root)
        input_hashes_pass &= resolved.is_file() and sha256_file(resolved) == expected
    audit.check("input_artifact_hashes", input_hashes_pass)

    symbols = [row["symbol"] for row in ledger]
    raw_manifest_paths = sorted(
        (data_root / "raw" / "source=eodhd" / "endpoint=intraday").glob(
            "symbol=*/interval=5m/2024-01-01_2025-08-23.json"
        )
    )
    raw_manifest_symbols = sorted(
        canonical_symbol(path.parent.parent.name.removeprefix("symbol="))
        for path in raw_manifest_paths
    )
    audit.check("safe_universe_unique_and_sorted", symbols == sorted(set(symbols)))
    audit.check("raw_manifest_union_complete", symbols == raw_manifest_symbols)
    audit.check("safe_universe_count", len(ledger) == int(decision["safe_raw_universe_count"]))
    audit.check(
        "all_rows_present_in_safe_raw_manifest",
        all(parse_bool(row["present_in_safe_raw_manifest"]) for row in ledger),
    )
    audit.check(
        "safe_raw_filenames",
        all(row["safe_raw_file"].endswith("/2024-01-01_2025-08-23.json") for row in ledger),
    )
    source_rows = {item["symbol"]: item for item in source_manifest["sources"]}
    audit.check("source_manifest_matches_ledger", set(source_rows) == set(symbols))
    raw_hash_pass = all(
        row["raw_file_hash"] == source_rows[row["symbol"]]["safe_raw_file_sha256"]
        and row["safe_raw_file"] == source_rows[row["symbol"]]["safe_raw_file"]
        for row in ledger
    )
    audit.check("ledger_raw_hashes_match_source_manifest", raw_hash_pass)

    clean_rows = [row for row in ledger if row["exposure_status"].startswith("clean_")]
    unknown_rows = [row for row in ledger if row["exposure_status"] == "unknown_assume_exposed"]
    exposed_rows = [row for row in ledger if row["exposure_status"] == "outcome_exposed"]
    audit.check(
        "clean_count_recalculated", len(clean_rows) == decision["clean_outcome_unexposed_count"]
    )
    audit.check("unknown_count_recalculated", len(unknown_rows) == decision["unknown_count"])
    audit.check(
        "outcome_exposed_count_recalculated", len(exposed_rows) == decision["outcome_exposed_count"]
    )
    audit.check(
        "unknown_assumed_exposed",
        all(
            not parse_bool(row["eligible_for_cross_stock_holdout"])
            and row["exclusion_reason"] == "exposure_evidence_unresolved"
            for row in unknown_rows
        ),
    )

    corpus = audit_report_corpus(
        data_root / "reports" / "research", [row["symbol"] for row in clean_rows]
    )
    audit.check(
        "report_corpus_manifest_recalculated",
        corpus["corpus_manifest_sha256"] == evidence_manifest["corpus_manifest_sha256"]
        and corpus["files_hashed"] == evidence_manifest["corpus_files_hashed"]
        and corpus["bytes_hashed"] == evidence_manifest["corpus_bytes_hashed"],
    )
    audit.check(
        "clean_symbols_absent_from_outcome_report_corpus",
        all(not paths for paths in corpus["clean_symbol_mentions"].values()),
    )

    exposed_evidence_pass = True
    for row in exposed_rows:
        symbol = row["symbol"]
        paths = row["exposure_evidence_paths"].split("|")
        research_paths = [
            path for path in paths if path.startswith("data/reports/research/") and "!" not in path
        ]
        predecessor_evidence = symbol in DEVELOPMENT_SYMBOLS and any(
            path.startswith("research/movement-regime-path/") for path in paths
        )
        report_evidence = any(
            resolve_logical_path(path, data_root).is_file()
            and path_contains_symbol_and_outcome(resolve_logical_path(path, data_root), symbol)
            for path in research_paths
        )
        exposed_evidence_pass &= predecessor_evidence or report_evidence
    audit.check("outcome_exposure_evidence_reconstructed", exposed_evidence_pass)

    development = read_symbols(artifacts / "development_symbols.txt")
    assessment = read_symbols(artifacts / "assessment_symbols.txt")
    excluded = read_symbols(artifacts / "excluded_symbols.txt")
    eligible = sorted(
        row["symbol"] for row in ledger if parse_bool(row["eligible_for_cross_stock_holdout"])
    )
    audit.check("development_symbols_exact_predecessor", tuple(development) == DEVELOPMENT_SYMBOLS)
    audit.check("development_assessment_disjoint", not (set(development) & set(assessment)))
    audit.check("assessment_symbols_recalculated", assessment == eligible)
    audit.check("excluded_symbols_recalculated", excluded == sorted(set(symbols) - set(eligible)))
    audit.check(
        "clean_data_qa_gate_recalculated",
        all(
            parse_bool(row["eligible_for_cross_stock_holdout"])
            == (
                row["exposure_status"].startswith("clean_")
                and row["vendor_qa_status"] == "pass"
                and parse_bool(row["bar_audit_passed"])
            )
            for row in ledger
        ),
    )

    expected_decision = (
        "blocked_no_clean_cross_stock_holdout_remaining" if len(clean_rows) < 15 else None
    )
    audit.check("phase_a_decision_logic", decision["decision"] == expected_decision)
    audit.check(
        "assessment_outcomes_unopened",
        decision["assessment_outcomes_read"] is False
        and decision["assessment_rows_materialised"] == 0
        and decision["development_rows_materialised"] == 0,
    )
    audit.check(
        "protected_boundary",
        boundary["minimum_market_timestamp_read"] is None
        and boundary["maximum_market_timestamp_read"] is None
        and boundary["protected_files_touched_count"] == 0
        and boundary["market_rows_materialised"] == 0
        and boundary["protected_rows_materialised"] == 0,
    )
    audit.check(
        "predecessor_reconstruction_not_reached",
        reconstruction["predictions_reconstructed"] is False
        and reconstruction["reference_rows_scored"] == 0
        and reconstruction["result"] == "not_attempted_phase_a_blocker_precedes_reconstruction",
    )
    audit.check(
        "downstream_artifacts_absent",
        all(not (artifacts / name).exists() for name in not_produced["artifacts"]),
    )
    scientific_field_names = set(ledger[0])
    for item in source_manifest["sources"]:
        scientific_field_names.update(item)
    audit.check(
        "forbidden_candidate_structural_and_broker_fields_absent",
        not any(
            fragment in field.lower()
            for field in scientific_field_names
            for fragment in FORBIDDEN_SCIENTIFIC_FIELD_FRAGMENTS
        ),
    )
    audit.check("no_local_absolute_paths", not has_absolute_local_path(artifacts))

    exact_manifest_path = artifacts / "exact_rerun_manifest.json"
    if exact_manifest_path.is_file():
        exact_manifest = json.loads(exact_manifest_path.read_text(encoding="utf-8"))
        audit.check(
            "exact_rerun_manifest_passed",
            exact_manifest.get("passed") is True
            and all(exact_manifest.get(key) == value for key, value in SAFETY_FLAGS.items()),
        )
        audit.check(
            "exact_rerun_implementation_hashes",
            all(
                (REPO_ROOT / logical_path).is_file()
                and sha256_file(REPO_ROOT / logical_path) == expected
                for logical_path, expected in exact_manifest.get(
                    "implementation_hashes", {}
                ).items()
            )
            and bool(exact_manifest.get("implementation_hashes")),
        )

    result = {
        **SAFETY_FLAGS,
        "auditor_imports_runner": False,
        "auditor_imports_experiment_module": False,
        "assessment_outcomes_read": False,
        "protected_rows_materialised": 0,
        "passed": audit.passed,
        "check_count": len(audit.checks),
        "failed_checks": [item["check"] for item in audit.checks if not item["passed"]],
        "checks": audit.checks,
    }
    (artifacts / "independent_audit.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_audit(
            artifacts=args.artifacts.resolve(),
            data_root=args.data_root.expanduser().resolve(),
        )
    except Exception as error:
        result = {
            **SAFETY_FLAGS,
            "auditor_imports_runner": False,
            "auditor_imports_experiment_module": False,
            "passed": False,
            "check_count": 0,
            "failed_checks": ["auditor_exception"],
            "exception_type": type(error).__name__,
            "exception": str(error),
        }
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "independent_audit.json").write_text(
            canonical_json(result), encoding="utf-8"
        )
    print(canonical_json(result), end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
