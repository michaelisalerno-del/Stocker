"""Static and ledger-level retirement/provenance audits."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import FORBIDDEN_PRIMARY_COLUMNS

FORBIDDEN_IMPORT_FRAGMENTS: tuple[str, ...] = (
    ".regime",
    ".loops",
    ".loop_",
    ".excursion",
    ".posterior",
)


class ProvenanceViolation(RuntimeError):
    """Raised when retired or outcome-bearing inputs cross the primary boundary."""


def _forbidden_column(column: str) -> bool:
    normalized = column.lower()
    if normalized in FORBIDDEN_PRIMARY_COLUMNS:
        return True
    prefixes = (
        "future_",
        "target_",
        "mfe",
        "mae",
        "regime_",
        "loop_",
        "excursion_",
        "posterior_",
        "personality_",
        "slrno_",
        "pnl_",
        "cost_",
        "spread_",
        "slippage_",
    )
    return normalized.startswith(prefixes)


def assert_outcome_free_event_ledger(ledger: pd.DataFrame) -> None:
    """Fail if an outcome, economic, or retired field appears in an event ledger."""

    forbidden = sorted(column for column in ledger.columns if _forbidden_column(str(column)))
    if forbidden:
        raise ProvenanceViolation(f"forbidden event-ledger columns: {forbidden}")


def assert_primary_feature_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Fail if a retired field appears in the primary feature surface."""

    forbidden = sorted(column for column in columns if _forbidden_column(str(column)))
    if forbidden:
        raise ProvenanceViolation(f"forbidden feature columns: {forbidden}")


def audit_primary_imports(source_files: list[Path]) -> list[str]:
    """Parse imports without importing candidate modules and report retired dependencies."""

    violations: list[str] = []
    for path in sorted(source_files):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for module in imported:
            if any(fragment in f".{module}" for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
                violations.append(f"{path.name}:{module}")
    return sorted(violations)
