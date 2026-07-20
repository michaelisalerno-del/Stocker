"""Run the frozen replication boundary, independent audit, and exact rerun."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from stocker_research.observable_extreme_tail_replication_v1 import (
    BLOCKED_NO_CLEAN_HOLDOUT,
    SAFETY_FLAGS,
    canonical_json,
    sha256_file,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
PREPARE_PATH = EXPERIMENT_DIR / "prepare_freeze.py"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_replication.py"
DEFAULT_FREEZE = EXPERIMENT_DIR / "artifacts" / "freeze"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
DEFAULT_DATA_ROOT = Path.home() / "StockerLocal" / "data"
IMPLEMENTATION_LOGICAL_PATHS = (
    "packages/stocker_research/src/stocker_research/observable_extreme_tail_replication_v1.py",
    "research/observable-extreme-tail/20260720-cross-stock-replication-v1/prepare_freeze.py",
    "research/observable-extreme-tail/20260720-cross-stock-replication-v1/run_replication.py",
    "research/observable-extreme-tail/20260720-cross-stock-replication-v1/audit_replication.py",
    "research/observable-extreme-tail/20260720-cross-stock-replication-v1/contract.json",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _load_verified_freeze(freeze: Path) -> dict[str, Any]:
    manifest_path = freeze / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise RuntimeError("blocked_reproducibility_or_audit_failure:contract_hash")
    for key, expected in SAFETY_FLAGS.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"blocked_reproducibility_or_audit_failure:safety:{key}")
    for name, expected in manifest["artifacts"].items():
        path = freeze / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"blocked_reproducibility_or_audit_failure:freeze:{name}")
    if manifest.get("decision") != BLOCKED_NO_CLEAN_HOLDOUT:
        raise RuntimeError("unexpected_phase_a_result")
    return manifest


def _copy_freeze(freeze: Path, output: Path, manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in sorted(manifest["artifacts"]):
        shutil.copyfile(freeze / name, output / name)
    shutil.copyfile(freeze / "freeze_manifest.json", output / "freeze_manifest.json")


def _run_checked(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"blocked_reproducibility_or_audit_failure:{detail}")


def _prepare_command(
    *,
    data_root: Path,
    output: Path,
    max_symbols: int | None,
    max_sessions: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PREPARE_PATH),
        "--data-root",
        str(data_root),
        "--output",
        str(output),
    ]
    if max_symbols is not None:
        command.extend(("--max-symbols", str(max_symbols)))
    if max_sessions is not None:
        command.extend(("--max-sessions", str(max_sessions)))
    return command


def _audit_command(*, artifacts: Path, data_root: Path) -> list[str]:
    return [
        sys.executable,
        str(AUDITOR_PATH),
        "--artifacts",
        str(artifacts),
        "--data-root",
        str(data_root),
    ]


def _compare_artifacts(primary: Path, exact: Path, names: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in sorted(names):
        primary_path = primary / name
        exact_path = exact / name
        if not primary_path.is_file() or not exact_path.is_file():
            raise RuntimeError(f"blocked_reproducibility_or_audit_failure:missing:{name}")
        primary_hash = sha256_file(primary_path)
        if primary_hash != sha256_file(exact_path):
            raise RuntimeError(f"blocked_reproducibility_or_audit_failure:mismatch:{name}")
        hashes[name] = primary_hash
    return hashes


def run_replication(
    *,
    data_root: Path,
    freeze: Path,
    output: Path,
    audit: bool,
    exact_rerun: bool,
    max_symbols: int | None,
    max_sessions: int | None,
) -> dict[str, Any]:
    """Consume the freeze and stop at its preregistered scientific blocker."""

    smoke = max_symbols is not None or max_sessions is not None
    if smoke:
        _run_checked(
            _prepare_command(
                data_root=data_root,
                output=output,
                max_symbols=max_symbols,
                max_sessions=max_sessions,
            )
        )
        manifest = _load_verified_freeze(output)
    else:
        manifest = _load_verified_freeze(freeze)
        _copy_freeze(freeze, output, manifest)

    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    if decision.get("decision") != BLOCKED_NO_CLEAN_HOLDOUT:
        raise RuntimeError("phase_a_blocker_not_preserved")
    if bool(decision.get("non_scientific_smoke_test")) != smoke:
        raise RuntimeError("smoke_label_mismatch")

    exact_passed: bool | None = None
    audit_passed: bool | None = None
    if exact_rerun:
        exact_output = (
            DEFAULT_EXACT_OUTPUT
            if output == DEFAULT_OUTPUT and not smoke
            else output.parent / f"{output.name}_exact_rerun"
        )
        _run_checked(
            _prepare_command(
                data_root=data_root,
                output=exact_output,
                max_symbols=max_symbols,
                max_sessions=max_sessions,
            )
        )
        comparison_names = [*manifest["artifacts"], "freeze_manifest.json"]
        compared_hashes = _compare_artifacts(output, exact_output, comparison_names)
        exact_manifest = {
            **SAFETY_FLAGS,
            "non_scientific_smoke_test": smoke,
            "decision": BLOCKED_NO_CLEAN_HOLDOUT,
            "complete_run_scope": "phase_a_mandatory_blocker",
            "byte_identical": True,
            "strict_numeric_tolerance_used": False,
            "implementation_hashes": {
                path: sha256_file(REPO_ROOT / path) for path in IMPLEMENTATION_LOGICAL_PATHS
            },
            "compared_artifact_hashes": compared_hashes,
            "passed": True,
        }
        _write_json(output / "exact_rerun_manifest.json", exact_manifest)
        _write_json(exact_output / "exact_rerun_manifest.json", exact_manifest)
        if audit:
            _run_checked(_audit_command(artifacts=output, data_root=data_root))
            _run_checked(_audit_command(artifacts=exact_output, data_root=data_root))
            post_audit_names = [*comparison_names, "independent_audit.json"]
            exact_manifest["post_audit_compared_artifact_hashes"] = _compare_artifacts(
                output, exact_output, post_audit_names
            )
            _write_json(output / "exact_rerun_manifest.json", exact_manifest)
            _write_json(exact_output / "exact_rerun_manifest.json", exact_manifest)
            _compare_artifacts(output, exact_output, ["exact_rerun_manifest.json"])
            _run_checked(_audit_command(artifacts=output, data_root=data_root))
            _run_checked(_audit_command(artifacts=exact_output, data_root=data_root))
            final_audit_hashes = _compare_artifacts(
                output, exact_output, ["independent_audit.json"]
            )
            if (
                exact_manifest["post_audit_compared_artifact_hashes"]["independent_audit.json"]
                != final_audit_hashes["independent_audit.json"]
            ):
                raise RuntimeError("blocked_reproducibility_or_audit_failure:final_audit_hash")
            audit_document = json.loads(
                (output / "independent_audit.json").read_text(encoding="utf-8")
            )
            audit_passed = bool(audit_document["passed"])
        else:
            _compare_artifacts(output, exact_output, ["exact_rerun_manifest.json"])
        exact_passed = True
    elif audit:
        _run_checked(_audit_command(artifacts=output, data_root=data_root))
        audit_document = json.loads((output / "independent_audit.json").read_text(encoding="utf-8"))
        audit_passed = bool(audit_document["passed"])

    if output == DEFAULT_OUTPUT:
        shutil.copyfile(output / "report.md", EXPERIMENT_DIR / "reports" / "report.md")

    result = {
        **SAFETY_FLAGS,
        "non_scientific_smoke_test": smoke,
        "decision": BLOCKED_NO_CLEAN_HOLDOUT,
        "phase_reached": "phase_a_stock_outcome_exposure_ledger",
        "assessment_outcomes_read": False,
        "protected_rows_materialised": 0,
        "independent_audit_passed": audit_passed,
        "exact_rerun_passed": exact_passed,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--exact-rerun", action="store_true")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--max-sessions", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_replication(
        data_root=args.data_root.expanduser().resolve(),
        freeze=args.freeze.resolve(),
        output=args.output.resolve(),
        audit=args.audit,
        exact_rerun=args.exact_rerun,
        max_symbols=args.max_symbols,
        max_sessions=args.max_sessions,
    )
    print(canonical_json(result), end="")


if __name__ == "__main__":
    main()
