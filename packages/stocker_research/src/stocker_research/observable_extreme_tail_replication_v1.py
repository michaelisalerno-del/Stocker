"""Fail-closed primitives for Observable Extreme-Tail Replication V1.

The scientific run is allowed to advance beyond its exposure freeze only when a
machine-evidenced, outcome-unexposed stock cohort satisfies the preregistered support
gate.  This module deliberately contains no broker, execution, regime, loop, closure,
or structural-path integration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

PROTECTED_START = datetime(2025, 8, 23, tzinfo=UTC)
MINIMUM_CLEAN_HOLDOUT_STOCKS = 15
BLOCKED_NO_CLEAN_HOLDOUT = "blocked_no_clean_cross_stock_holdout_remaining"

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

PREDECESSOR_EXPERIMENT = (
    "research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0"
)

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

FORBIDDEN_FIELD_FRAGMENTS = (
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
    "order_id",
    "order_placement",
    "portfolio_size",
    "position_size",
    "deployment",
)

OUTCOME_REPORT_MARKERS = (
    b'"net_return"',
    b'"gross_return"',
    b'"pnl"',
    b'"forward_',
    b'"mfe"',
    b'"mae"',
    b'"best_trade"',
    b'"entry_price"',
    b'"exit_price"',
)
SYMBOL_REPORT_NAME = re.compile(r"_(?P<symbol>[A-Z][A-Z0-9.-]*)_(?:1d|5m)\.json$")
CSV_OUTCOME_FIELD_FRAGMENTS = (
    "forward_",
    "raw_return",
    "net_return",
    "gross_return",
    "net_r",
    "gross_r",
    "pnl",
    "mfe",
    "mae",
    "entry_price",
    "exit_price",
    "target_hit",
    "stop_hit",
    "favorable_excursion",
    "adverse_excursion",
)

REPORT_TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt", ".yaml", ".yml"})
REPORT_SCAN_CHUNK_BYTES = 1024 * 1024


class EvidenceKind(StrEnum):
    """Machine-readable evidence categories used by the stock ledger."""

    DATA_QA_ONLY = "data_qa_only"
    OBSERVABLE_MODEL_TRAINING = "observable_model_training"
    FEATURE_SELECTION = "feature_selection"
    THRESHOLD_SELECTION = "threshold_selection"
    DIRECTIONAL_OUTCOME_ANALYSIS = "directional_outcome_analysis"
    POST_HOC_TAIL_INSPECTION = "post_hoc_tail_inspection"
    PRICE_OR_PAYOFF_REPORT = "price_or_payoff_report"
    STRUCTURAL_NONPRICE_RESEARCH = "structural_nonprice_research"
    OUTCOME_CORPUS_NO_SYMBOL_MATCH = "outcome_corpus_no_symbol_match"
    AMBIGUOUS_REPORT_MENTION = "ambiguous_report_mention"


OUTCOME_EXPOSURE_KINDS = frozenset(
    {
        EvidenceKind.OBSERVABLE_MODEL_TRAINING,
        EvidenceKind.FEATURE_SELECTION,
        EvidenceKind.THRESHOLD_SELECTION,
        EvidenceKind.DIRECTIONAL_OUTCOME_ANALYSIS,
        EvidenceKind.POST_HOC_TAIL_INSPECTION,
        EvidenceKind.PRICE_OR_PAYOFF_REPORT,
    }
)


@dataclass(frozen=True)
class ExposureEvidence:
    """One exact logical path supporting a stock-exposure classification."""

    logical_path: str
    kind: EvidenceKind
    detail: str


@dataclass(frozen=True)
class ExposureClassification:
    """Fail-closed result for one stock in the safe raw manifest."""

    symbol: str
    exposure_status: str
    eligible_for_cross_stock_holdout: bool
    exclusion_reason: str


@dataclass(frozen=True)
class SafeStockSource:
    """One stock proven present in the pre-boundary EODHD raw manifest."""

    symbol: str
    vendor_qa_logical_path: str
    bar_audit_logical_path: str
    raw_file_logical_path: str
    raw_file_sha256: str
    vendor_qa_status: str
    vendor_qa_issue_codes: tuple[str, ...]
    bar_audit_passed: bool
    bar_audit_issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class ReportCorpusScan:
    """Deterministic full-text/ZIP scan of the existing research-report corpus."""

    corpus_manifest_sha256: str
    files_hashed: int
    bytes_hashed: int
    text_files_scanned: int
    zip_archives_scanned: int
    zip_members_scanned: int
    ignored_incomplete_archive_parts: tuple[str, ...]
    symbol_mention_paths: dict[str, tuple[str, ...]]


def canonical_json(value: Any) -> str:
    """Serialize a scientific JSON artifact deterministically."""

    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def canonical_stock_symbol(symbol: str) -> str:
    """Return the stock identity used for cross-report alias matching."""

    normalized = symbol.strip().upper()
    return normalized.removesuffix(".US")


def sha256_file(path: Path) -> str:
    """Hash a file without interpreting or materialising its rows."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_data_path(data_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError("source_path_outside_data_root") from error
    return f"data/{relative.as_posix()}"


def discover_safe_stock_sources(data_root: Path) -> list[SafeStockSource]:
    """Enumerate every symbol in the pre-boundary EODHD raw manifest."""

    qa_root = data_root / "reports" / "vendor_qa"
    audit_root = data_root / "reports" / "audits"
    raw_root = data_root / "raw" / "source=eodhd" / "endpoint=intraday"
    sources: list[SafeStockSource] = []
    raw_paths = sorted(raw_root.glob("symbol=*/interval=5m/2024-01-01_2025-08-23.json"))
    for raw_path in raw_paths:
        symbol_directory = raw_path.parent.parent.name
        if not symbol_directory.startswith("symbol="):
            raise ValueError("invalid_raw_manifest_symbol_path")
        symbol = canonical_stock_symbol(symbol_directory.removeprefix("symbol="))
        qa_path = qa_root / f"{symbol}_5m_eodhd_qa.json"
        qa: dict[str, Any]
        if qa_path.is_file():
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            if canonical_stock_symbol(str(qa["symbol"])) != symbol:
                raise ValueError("vendor_qa_symbol_mismatch")
            qa_raw_value = Path(str(qa["raw_files"]["first_raw_file"]))
            qa_raw_path = qa_raw_value if qa_raw_value.is_absolute() else data_root / qa_raw_value
            if qa_raw_path.resolve() != raw_path.resolve():
                raise ValueError("blocked_protected_boundary_failure")
        else:
            qa = {
                "status": "missing",
                "issue_codes": ["missing_vendor_qa"],
            }
        audit_path = audit_root / f"{symbol}_5m_audit.json"
        audit: dict[str, Any]
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            audit = {"passed": False, "issues": [{"code": "missing_bar_audit"}]}
        sources.append(
            SafeStockSource(
                symbol=symbol,
                vendor_qa_logical_path=_logical_data_path(data_root, qa_path),
                bar_audit_logical_path=_logical_data_path(data_root, audit_path),
                raw_file_logical_path=_logical_data_path(data_root, raw_path),
                raw_file_sha256=sha256_file(raw_path),
                vendor_qa_status=str(qa.get("status", "unknown")),
                vendor_qa_issue_codes=tuple(sorted(map(str, qa.get("issue_codes", [])))),
                bar_audit_passed=bool(audit.get("passed", False)),
                bar_audit_issue_codes=tuple(
                    sorted(str(item.get("code", "unknown")) for item in audit.get("issues", []))
                ),
            )
        )
    return sources


def validate_symbol_disjointness(
    development_symbols: list[str] | tuple[str, ...],
    assessment_symbols: list[str] | tuple[str, ...],
) -> None:
    """Reject direct or vendor-suffix alias overlap across cohorts."""

    development = {canonical_stock_symbol(symbol) for symbol in development_symbols}
    assessment = {canonical_stock_symbol(symbol) for symbol in assessment_symbols}
    overlap = sorted(development & assessment)
    if overlap:
        raise ValueError(f"development_assessment_symbol_overlap:{','.join(overlap)}")


def validate_market_timestamp(value: str | datetime) -> datetime:
    """Reject timestamps outside the preregistered safe assessment boundary."""

    timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if timestamp.tzinfo is None:
        raise ValueError("blocked_protected_boundary_failure")
    normalized = timestamp.astimezone(UTC)
    if normalized >= PROTECTED_START or normalized.year >= 2026:
        raise ValueError("blocked_protected_boundary_failure")
    return normalized


def exposure_gate_decision(clean_stock_count: int) -> str | None:
    """Return the preregistered Phase A blocker, without lowering support."""

    if clean_stock_count < MINIMUM_CLEAN_HOLDOUT_STOCKS:
        return BLOCKED_NO_CLEAN_HOLDOUT
    return None


def find_forbidden_fields(field_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return forbidden experiment fields in stable order."""

    return sorted(
        field
        for field in field_names
        if any(fragment in field.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS)
    )


def _logical_research_path(report_root: Path, path: Path) -> str:
    relative = path.relative_to(report_root).as_posix()
    return f"data/reports/research/{relative}"


def _symbol_pattern(symbols: list[str] | tuple[str, ...]) -> re.Pattern[bytes]:
    alternatives = b"|".join(
        re.escape(canonical_stock_symbol(symbol).encode("ascii"))
        for symbol in sorted(symbols, key=lambda value: (-len(value), value))
    )
    return re.compile(rb"(?<![A-Z0-9])(?P<symbol>" + alternatives + rb")(?:\.US)?(?![A-Z0-9])")


def _scan_binary_stream(
    handle: IO[bytes],
    pattern: re.Pattern[bytes],
) -> set[str]:
    mentions: set[str] = set()
    carry = b""
    while chunk := handle.read(REPORT_SCAN_CHUNK_BYTES):
        searchable = (carry + chunk).upper()
        mentions.update(
            match.group("symbol").decode("ascii") for match in pattern.finditer(searchable)
        )
        carry = searchable[-64:]
    return mentions


def scan_report_corpus(
    report_root: Path,
    symbols: list[str] | tuple[str, ...],
) -> ReportCorpusScan:
    """Hash and exhaustively search supported reports and complete ZIP archives."""

    canonical_symbols = tuple(sorted({canonical_stock_symbol(symbol) for symbol in symbols}))
    pattern = _symbol_pattern(canonical_symbols)
    mention_paths: dict[str, set[str]] = {symbol: set() for symbol in canonical_symbols}
    manifest_digest = hashlib.sha256()
    files_hashed = 0
    bytes_hashed = 0
    text_files_scanned = 0
    zip_archives_scanned = 0
    zip_members_scanned = 0
    ignored_parts: list[str] = []

    for path in sorted(candidate for candidate in report_root.rglob("*") if candidate.is_file()):
        logical_path = _logical_research_path(report_root, path)
        file_hash = sha256_file(path)
        size = path.stat().st_size
        manifest_digest.update(f"{logical_path}\0{size}\0{file_hash}\n".encode())
        files_hashed += 1
        bytes_hashed += size

        if path.suffix.lower() in REPORT_TEXT_SUFFIXES:
            with path.open("rb") as handle:
                found = _scan_binary_stream(handle, pattern)
            for symbol in found:
                mention_paths[symbol].add(logical_path)
            text_files_scanned += 1
            continue
        if path.suffix.lower() == ".zip" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    if (
                        member.is_dir()
                        or Path(member.filename).suffix.lower() not in REPORT_TEXT_SUFFIXES
                    ):
                        continue
                    with archive.open(member) as handle:
                        found = _scan_binary_stream(handle, pattern)
                    member_path = f"{logical_path}!{member.filename}"
                    for symbol in found:
                        mention_paths[symbol].add(member_path)
                    zip_members_scanned += 1
            zip_archives_scanned += 1
            continue
        if ".zip.part-" in path.name:
            ignored_parts.append(logical_path)

    return ReportCorpusScan(
        corpus_manifest_sha256=manifest_digest.hexdigest(),
        files_hashed=files_hashed,
        bytes_hashed=bytes_hashed,
        text_files_scanned=text_files_scanned,
        zip_archives_scanned=zip_archives_scanned,
        zip_members_scanned=zip_members_scanned,
        ignored_incomplete_archive_parts=tuple(ignored_parts),
        symbol_mention_paths={
            symbol: tuple(sorted(paths)) for symbol, paths in sorted(mention_paths.items())
        },
    )


def _first_outcome_marker(path: Path) -> str | None:
    with path.open("rb") as handle:
        prefix = handle.read(512 * 1024).lower()
    for marker in OUTCOME_REPORT_MARKERS:
        if marker in prefix:
            return marker.decode("ascii").strip('"')
    return None


def discover_outcome_evidence(
    report_root: Path, symbols: list[str] | tuple[str, ...]
) -> dict[str, list[ExposureEvidence]]:
    """Find one decisive outcome-bearing report for each requested stock."""

    canonical_to_requested = {
        canonical_stock_symbol(symbol): canonical_stock_symbol(symbol) for symbol in symbols
    }
    evidence: dict[str, list[ExposureEvidence]] = {
        canonical: [] for canonical in canonical_to_requested
    }
    if not report_root.is_dir():
        return evidence
    for path in sorted(report_root.rglob("*.json")):
        match = SYMBOL_REPORT_NAME.search(path.name)
        if match is None:
            continue
        canonical = canonical_stock_symbol(match.group("symbol"))
        if canonical not in evidence or evidence[canonical]:
            continue
        marker = _first_outcome_marker(path)
        if marker is None:
            continue
        evidence[canonical].append(
            ExposureEvidence(
                logical_path=_logical_research_path(report_root, path),
                kind=EvidenceKind.PRICE_OR_PAYOFF_REPORT,
                detail=f"individual report contains outcome field {marker}",
            )
        )
    unresolved = {symbol for symbol, items in evidence.items() if not items}
    for path in sorted(report_root.rglob("*.csv")):
        if not unresolved:
            break
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                field_names = reader.fieldnames or []
                symbol_fields = [
                    field
                    for field in field_names
                    if field.lower() in {"symbol", "stock_symbol", "ticker"}
                    or field.lower().endswith("_symbol")
                ]
                outcome_fields = [
                    field
                    for field in field_names
                    if any(fragment in field.lower() for fragment in CSV_OUTCOME_FIELD_FRAGMENTS)
                ]
                if not symbol_fields or not outcome_fields:
                    continue
                for row in reader:
                    row_symbols = {
                        canonical_stock_symbol(row.get(field, "")) for field in symbol_fields
                    }
                    for canonical in sorted(row_symbols & unresolved):
                        evidence[canonical].append(
                            ExposureEvidence(
                                logical_path=_logical_research_path(report_root, path),
                                kind=EvidenceKind.PRICE_OR_PAYOFF_REPORT,
                                detail=(
                                    "aggregate symbol ledger contains outcome fields "
                                    + ",".join(sorted(outcome_fields)[:5])
                                ),
                            )
                        )
                        unresolved.remove(canonical)
        except (OSError, csv.Error):
            continue
    return evidence


def _predecessor_evidence(symbol: str) -> list[ExposureEvidence]:
    if canonical_stock_symbol(symbol) not in DEVELOPMENT_SYMBOLS:
        return []
    return [
        ExposureEvidence(
            logical_path=f"{PREDECESSOR_EXPERIMENT}/artifacts/primary/model_coefficients.json",
            kind=EvidenceKind.OBSERVABLE_MODEL_TRAINING,
            detail="Movement V0 P1, P1_SIZE, and D0 development cohort",
        ),
        ExposureEvidence(
            logical_path=f"{PREDECESSOR_EXPERIMENT}/artifacts/primary/movement_thresholds.json",
            kind=EvidenceKind.THRESHOLD_SELECTION,
            detail="Movement V0 2024 meaningful-movement thresholds",
        ),
        ExposureEvidence(
            logical_path=f"{PREDECESSOR_EXPERIMENT}/artifacts/primary/direction_metrics.csv",
            kind=EvidenceKind.DIRECTIONAL_OUTCOME_ANALYSIS,
            detail="Movement V0 observable D0 directional-outcome analysis",
        ),
        ExposureEvidence(
            logical_path=f"{PREDECESSOR_EXPERIMENT}/artifacts/primary/chain_ranking_metrics.csv",
            kind=EvidenceKind.POST_HOC_TAIL_INSPECTION,
            detail="Movement V0 observable-chain top-one outcome diagnostic",
        ),
    ]


def stock_exposure_evidence(
    source: SafeStockSource,
    discovered_evidence: dict[str, list[ExposureEvidence]],
    *,
    clean_scan_symbols: set[str],
    ambiguous_report_mentions: dict[str, tuple[str, ...]] | None = None,
) -> list[ExposureEvidence]:
    """Assemble the exact evidence set used to classify one source symbol."""

    symbol = canonical_stock_symbol(source.symbol)
    evidence = [
        ExposureEvidence(
            logical_path=source.vendor_qa_logical_path,
            kind=EvidenceKind.DATA_QA_ONLY,
            detail=f"EODHD vendor QA status={source.vendor_qa_status}",
        ),
        ExposureEvidence(
            logical_path=source.bar_audit_logical_path,
            kind=EvidenceKind.DATA_QA_ONLY,
            detail=f"bar audit passed={source.bar_audit_passed}",
        ),
        *discovered_evidence.get(symbol, []),
        *_predecessor_evidence(symbol),
    ]
    if symbol in clean_scan_symbols:
        evidence.append(
            ExposureEvidence(
                logical_path="data/reports/research",
                kind=EvidenceKind.OUTCOME_CORPUS_NO_SYMBOL_MATCH,
                detail="complete supported-text and ZIP scan found no symbol occurrence",
            )
        )
    for logical_path in (ambiguous_report_mentions or {}).get(symbol, ()):
        evidence.append(
            ExposureEvidence(
                logical_path=logical_path,
                kind=EvidenceKind.AMBIGUOUS_REPORT_MENTION,
                detail=(
                    "symbol-token text match without machine-resolved stock identity "
                    "or outcome exposure"
                ),
            )
        )
    return evidence


def build_stock_outcome_exposure_ledger(
    sources: list[SafeStockSource],
    discovered_evidence: dict[str, list[ExposureEvidence]],
    *,
    clean_scan_symbols: set[str] | None = None,
    ambiguous_report_mentions: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, object]]:
    """Build complete fail-closed ledger rows in stable symbol order."""

    rows: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda item: item.symbol):
        symbol = canonical_stock_symbol(source.symbol)
        evidence = stock_exposure_evidence(
            source,
            discovered_evidence,
            clean_scan_symbols=clean_scan_symbols or set(),
            ambiguous_report_mentions=ambiguous_report_mentions,
        )
        classification = classify_stock_exposure(symbol, evidence)
        data_eligible = source.vendor_qa_status == "pass" and source.bar_audit_passed
        eligible = classification.eligible_for_cross_stock_holdout and data_eligible
        exclusion_reason = classification.exclusion_reason
        if classification.eligible_for_cross_stock_holdout and not data_eligible:
            exclusion_reason = "vendor_qa_or_bar_audit_not_passed"
        kinds = {item.kind for item in evidence}
        outcome_exposed = bool(kinds & OUTCOME_EXPOSURE_KINDS)
        rows.append(
            {
                "symbol": symbol,
                "present_in_safe_raw_manifest": True,
                "raw_file_hash": source.raw_file_sha256,
                "data_qa_only": classification.exposure_status == "clean_data_qa_only",
                "used_in_observable_model_training": (
                    EvidenceKind.OBSERVABLE_MODEL_TRAINING in kinds
                ),
                "used_in_feature_selection": EvidenceKind.FEATURE_SELECTION in kinds,
                "used_in_threshold_selection": EvidenceKind.THRESHOLD_SELECTION in kinds,
                "used_in_directional_outcome_analysis": bool(
                    {
                        EvidenceKind.DIRECTIONAL_OUTCOME_ANALYSIS,
                        EvidenceKind.PRICE_OR_PAYOFF_REPORT,
                    }
                    & kinds
                ),
                "used_in_post_hoc_tail_inspection": (
                    EvidenceKind.POST_HOC_TAIL_INSPECTION in kinds
                ),
                "used_in_any_price_or_payoff_report": outcome_exposed,
                "used_only_in_structural_nonprice_research": bool(
                    EvidenceKind.STRUCTURAL_NONPRICE_RESEARCH in kinds and not outcome_exposed
                ),
                "exposure_evidence_paths": "|".join(
                    sorted({item.logical_path for item in evidence})
                ),
                "exposure_status": classification.exposure_status,
                "eligible_for_cross_stock_holdout": eligible,
                "exclusion_reason": exclusion_reason,
                "vendor_qa_status": source.vendor_qa_status,
                "vendor_qa_issue_codes": "|".join(source.vendor_qa_issue_codes),
                "bar_audit_passed": source.bar_audit_passed,
                "bar_audit_issue_codes": "|".join(source.bar_audit_issue_codes),
                "safe_raw_file": source.raw_file_logical_path,
            }
        )
    return rows


def classify_stock_exposure(
    symbol: str, evidence: list[ExposureEvidence]
) -> ExposureClassification:
    """Classify a symbol without allowing absence of evidence to mean clean."""

    if any(item.kind in OUTCOME_EXPOSURE_KINDS for item in evidence):
        return ExposureClassification(
            symbol=symbol,
            exposure_status="outcome_exposed",
            eligible_for_cross_stock_holdout=False,
            exclusion_reason="prior_price_or_payoff_outcome_exposure",
        )
    resolved_nonoutcome_kinds = {
        EvidenceKind.DATA_QA_ONLY,
        EvidenceKind.STRUCTURAL_NONPRICE_RESEARCH,
        EvidenceKind.OUTCOME_CORPUS_NO_SYMBOL_MATCH,
    }
    if (
        evidence
        and all(item.kind in resolved_nonoutcome_kinds for item in evidence)
        and any(
            item.kind
            in {
                EvidenceKind.STRUCTURAL_NONPRICE_RESEARCH,
                EvidenceKind.OUTCOME_CORPUS_NO_SYMBOL_MATCH,
            }
            for item in evidence
        )
    ):
        status = (
            "clean_structural_nonprice_only"
            if any(item.kind is EvidenceKind.STRUCTURAL_NONPRICE_RESEARCH for item in evidence)
            else "clean_data_qa_only"
        )
        return ExposureClassification(
            symbol=symbol,
            exposure_status=status,
            eligible_for_cross_stock_holdout=True,
            exclusion_reason="",
        )
    return ExposureClassification(
        symbol=symbol,
        exposure_status="unknown_assume_exposed",
        eligible_for_cross_stock_holdout=False,
        exclusion_reason="exposure_evidence_unresolved",
    )
