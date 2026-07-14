"""Offline, research-only runner for the dormant per-loop quality shadow.

There is intentionally no evaluation command.  The current frozen eligibility
set is empty, so ``issue`` stops before it reads a candidate prediction file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import per_loop_quality_shadow_core as core

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT_PATH = HERE / "contracts/20260710-frozen-loop-quality-shadow-v1.json"
SOURCE_MANIFEST_PATH = (
    HERE / "contracts/20260710-frozen-loop-quality-shadow-v1-manifest.json"
)
PROTECTED_SNAPSHOT_PATH = (
    HERE / "contracts/20260710-frozen-loop-quality-shadow-v1-protected-snapshot.json"
)
DEFAULT_RUNTIME = HERE / "shadow_validation/frozen_loop_quality_shadow_v1"
DEFAULT_QUALITY_ROOT = Path(
    "/private/tmp/stocker_per_loop_movement_quality_20260710"
)
AGGREGATE_SHADOW = HERE / "shadow_validation/frozen_loop_movement_shadow_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(core.safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "contract": root / "contract.json",
        "source_manifest": root / "source_freeze_manifest.json",
        "protected_snapshot": root / "protected_aggregate_shadow_snapshot.json",
        "bundle": root / "frozen_bundle",
        "ledger": root / "prediction_ledger.jsonl",
        "predictions": root / "prediction_batches",
        "metadata": root / "runtime_metadata.json",
    }


def resolve_source(item: dict[str, Any], quality_root: Path) -> Path:
    roots = {
        "workspace": WORKSPACE,
        "quality_artifact": quality_root,
    }
    return roots[item["root_role"]] / item["relative_path"]


def verify_sources(
    manifest: dict[str, Any], quality_root: Path
) -> list[dict[str, Any]]:
    if core.sha256_file(CONTRACT_PATH) != manifest["contract"]["sha256"]:
        raise AssertionError("quality shadow contract hash drift")
    verified = []
    for item in manifest["files"]:
        source = resolve_source(item, quality_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = core.sha256_file(source)
        if observed != item["sha256"]:
            raise AssertionError(f"frozen quality source drift: {item['name']}")
        verified.append({**item, "source": source})
    return verified


def verify_protected_aggregate() -> dict[str, Any]:
    expected = read_json(PROTECTED_SNAPSHOT_PATH)
    current = core.content_snapshot(AGGREGATE_SHADOW, WORKSPACE)
    if current != expected:
        raise AssertionError("existing aggregate movement shadow changed")
    return current


def verify_semantics(bundle: Path, contract: dict[str, Any]) -> dict[str, bool]:
    provisional = pd.read_csv(bundle / "artifacts/provisional_tiers_2024.csv")
    core.validate_provisional_tiers(provisional)
    final_tiers = pd.read_csv(bundle / "artifacts/final_cycle_tiers.csv")
    core.validate_final_tiers(final_tiers)
    fit_complete = read_json(bundle / "artifacts/fit_complete.json")
    gates = read_json(bundle / "artifacts/provisional_gates_2024.json")
    final_gates = read_json(bundle / "artifacts/final_gates.json")
    independent_audit = read_json(
        bundle / "artifacts/independent_artifact_audit.json"
    )
    audit_checks = independent_audit.get("checks", [])
    audit_named = {
        str(item.get("name")): item.get("pass") is True for item in audit_checks
    }
    checks = {
        "fit_research_only": fit_complete.get("research_only") is True,
        "fit_live_disabled": fit_complete.get("live_ordering_enabled") is False,
        "fit_orders_disabled": fit_complete.get("order_placement") == "disabled",
        "fit_scoring_outcomes_closed": fit_complete.get("scoring_outcomes_opened")
        is False,
        "provisional_research_only": gates.get("research_only") is True,
        "provisional_promotion_forbidden": gates.get("promotion_permitted") is False,
        "provisional_zero_good": gates.get("good_movement_quality_cycles") == 0,
        "provisional_zero_high": gates.get("high_movement_quality_cycles") == 0,
        "provisional_all_unqualified": gates.get("unqualified_cycles") == 20,
        "contract_zero_eligible": contract["eligibility_freeze"][
            "eligible_cycle_ids"
        ]
        == [],
        "final_all_periods_unqualified": all(
            final_tiers[column].astype(str).eq("unqualified").all()
            for column in (
                "provisional_2024_oof_grade",
                "development_2025_grade",
                "backward_2023_grade",
                "final_grade",
            )
        ),
        "final_zero_qualified": final_gates.get("qualified_good_or_high_cycles")
        == 0,
        "final_zero_high": final_gates.get("high_cycles") == 0,
        "final_minimum_rule": final_gates.get(
            "final_grade_is_minimum_of_2024_oof_2025_2023"
        )
        is True,
        "final_no_unqualified_surface": final_gates.get(
            "no_unqualified_cycle_may_surface"
        )
        is True,
        "final_research_only": final_gates.get("research_only") is True,
        "final_live_disabled": final_gates.get("live_ordering_enabled") is False,
        "final_orders_disabled": final_gates.get("order_placement") == "disabled",
        "independent_audit_all_passed": independent_audit.get("all_passed")
        is True,
        "independent_audit_48_checks": independent_audit.get("check_count") == 48
        and len(audit_checks) == 48
        and all(item.get("pass") is True for item in audit_checks),
        "independent_audit_research_only": independent_audit.get("research_only")
        is True,
        "independent_audit_live_disabled": independent_audit.get(
            "live_ordering_enabled"
        )
        is False,
        "independent_audit_orders_disabled": independent_audit.get(
            "order_placement"
        )
        == "disabled",
        "independent_audit_no_2026": independent_audit.get("no_2026_rows")
        is True,
        "independent_audit_final_all_unqualified": independent_audit.get(
            "final_decision", {}
        ).get("all_twenty_unqualified")
        is True,
        "independent_audit_final_zero_qualified": independent_audit.get(
            "final_decision", {}
        ).get("qualified_good_or_high_cycles")
        == 0,
        "independent_audit_final_zero_high": independent_audit.get(
            "final_decision", {}
        ).get("high_cycles")
        == 0,
        "independent_audit_shadow_closed": independent_audit.get(
            "prospective_shadow", {}
        ).get("ledger_lines")
        == 0
        and independent_audit.get("prospective_shadow", {}).get(
            "outcomes_opened"
        )
        is False,
        "independent_audit_shadow_hash_exact": independent_audit.get(
            "prospective_shadow", {}
        ).get("tree_sha256")
        == contract["integrity"]["broader_protected_path_snapshot_tree_sha256"],
        "independent_audit_final_artifact_hashes_exact": independent_audit.get(
            "scoring_artifact_hashes", {}
        ).get("final_cycle_tiers.csv")
        == contract["eligibility_freeze"]["final_cycle_tiers_sha256"]
        and independent_audit.get("scoring_artifact_hashes", {}).get("gates.json")
        == contract["eligibility_freeze"]["final_gates_sha256"],
        "independent_audit_no_execution_surface": audit_named.get(
            "no_execution_surface"
        )
        is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"dormant quality semantics failed: {failed}")
    return checks


def finalize_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Replace the empty initial freeze with sealed final-tier certification."""

    paths = runtime_paths(Path(args.runtime_root))
    if not paths["metadata"].is_file():
        raise FileNotFoundError("initialize the runtime before final certification")
    old_metadata = read_json(paths["metadata"])
    if old_metadata.get("outcomes_opened") is not False:
        raise AssertionError("cannot finalize after outcomes opened")
    if core.validate_ledger(paths["ledger"]):
        raise AssertionError("cannot finalize after any prediction issuance")
    protected_before = verify_protected_aggregate()
    contract = read_json(CONTRACT_PATH)
    core.validate_contract(contract)
    manifest = read_json(SOURCE_MANIFEST_PATH)
    verified = verify_sources(manifest, Path(args.quality_artifact_root))
    for item in verified:
        destination = paths["bundle"] / item["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], destination)
        if core.sha256_file(destination) != item["sha256"]:
            raise AssertionError(f"final bundle copy failed: {item['name']}")
    shutil.copy2(CONTRACT_PATH, paths["contract"])
    shutil.copy2(SOURCE_MANIFEST_PATH, paths["source_manifest"])
    shutil.copy2(PROTECTED_SNAPSHOT_PATH, paths["protected_snapshot"])
    semantics = verify_semantics(paths["bundle"], contract)
    protected_after = verify_protected_aggregate()
    if protected_after != protected_before:
        raise AssertionError("aggregate shadow changed during final certification")
    metadata = {
        "contract_id": contract["contract_id"],
        "initialized_at_utc": old_metadata["initialized_at_utc"],
        "final_certified_at_utc": pd.Timestamp.now(tz="UTC"),
        "final_certification_complete": True,
        "superseded_initial_source_manifest_sha256": old_metadata[
            "source_manifest_sha256"
        ],
        "activation_state": "dormant_no_eligible_cycles",
        "eligible_cycle_ids": [],
        "contract_sha256": core.sha256_file(paths["contract"]),
        "source_manifest_sha256": core.sha256_file(paths["source_manifest"]),
        "final_cycle_tiers_sha256": contract["eligibility_freeze"][
            "final_cycle_tiers_sha256"
        ],
        "final_gates_sha256": contract["eligibility_freeze"][
            "final_gates_sha256"
        ],
        "independent_artifact_audit_sha256": contract["final_certification"][
            "independent_post_score_audit_sha256"
        ],
        "independent_artifact_audit_check_count": 48,
        "independent_artifact_audit_all_passed": True,
        "protected_aggregate_files_only_snapshot_sha256": protected_after[
            "snapshot_sha256"
        ],
        "broader_protected_path_snapshot_tree_sha256": contract["integrity"][
            "broader_protected_path_snapshot_tree_sha256"
        ],
        "bundle_file_count": len(verified),
        "semantic_checks": semantics,
        "outcomes_opened": False,
        **core.safety_payload(),
    }
    write_json_atomic(paths["metadata"], metadata)
    return verify_runtime(paths["root"])


def init_runtime(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    core.validate_contract(contract)
    manifest = read_json(SOURCE_MANIFEST_PATH)
    verified = verify_sources(manifest, Path(args.quality_artifact_root))
    protected_before = verify_protected_aggregate()
    paths = runtime_paths(Path(args.runtime_root))
    if paths["metadata"].is_file():
        return verify_runtime(paths["root"])
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise AssertionError(f"runtime root is non-empty: {paths['root']}")
    paths["bundle"].mkdir(parents=True, exist_ok=True)
    paths["predictions"].mkdir(parents=True, exist_ok=True)
    for item in verified:
        destination = paths["bundle"] / item["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], destination)
        if core.sha256_file(destination) != item["sha256"]:
            raise AssertionError(f"bundle copy failed: {item['name']}")
    shutil.copy2(CONTRACT_PATH, paths["contract"])
    shutil.copy2(SOURCE_MANIFEST_PATH, paths["source_manifest"])
    shutil.copy2(PROTECTED_SNAPSHOT_PATH, paths["protected_snapshot"])
    paths["ledger"].write_text("")
    semantics = verify_semantics(paths["bundle"], contract)
    protected_after = verify_protected_aggregate()
    if protected_after != protected_before:
        raise AssertionError("aggregate shadow changed during quality initialization")
    metadata = {
        "contract_id": contract["contract_id"],
        "initialized_at_utc": pd.Timestamp.now(tz="UTC"),
        "activation_state": "dormant_no_eligible_cycles",
        "eligible_cycle_ids": [],
        "contract_sha256": core.sha256_file(paths["contract"]),
        "source_manifest_sha256": core.sha256_file(paths["source_manifest"]),
        "protected_aggregate_snapshot_sha256": protected_after["snapshot_sha256"],
        "bundle_file_count": len(verified),
        "semantic_checks": semantics,
        "outcomes_opened": False,
        **core.safety_payload(),
    }
    write_json_atomic(paths["metadata"], metadata)
    return verify_runtime(paths["root"])


def verify_runtime(root: Path) -> dict[str, Any]:
    paths = runtime_paths(root)
    if not paths["metadata"].is_file():
        raise FileNotFoundError(f"run init first: {paths['metadata']}")
    metadata = read_json(paths["metadata"])
    contract = read_json(paths["contract"])
    core.validate_contract(contract)
    if core.sha256_file(paths["contract"]) != metadata["contract_sha256"]:
        raise AssertionError("runtime contract drift")
    if (
        core.sha256_file(paths["source_manifest"])
        != metadata["source_manifest_sha256"]
    ):
        raise AssertionError("runtime source manifest drift")
    source_manifest = read_json(paths["source_manifest"])
    for item in source_manifest["files"]:
        bundled = paths["bundle"] / item["bundle_path"]
        if not bundled.is_file() or core.sha256_file(bundled) != item["sha256"]:
            raise AssertionError(f"runtime bundle drift: {item['name']}")
    expected_snapshot = read_json(paths["protected_snapshot"])
    current_snapshot = core.content_snapshot(AGGREGATE_SHADOW, WORKSPACE)
    if current_snapshot != expected_snapshot:
        raise AssertionError("protected aggregate shadow drift")
    protected_key = (
        "protected_aggregate_files_only_snapshot_sha256"
        if "protected_aggregate_files_only_snapshot_sha256" in metadata
        else "protected_aggregate_snapshot_sha256"
    )
    if current_snapshot["snapshot_sha256"] != metadata[protected_key]:
        raise AssertionError("protected aggregate snapshot metadata drift")
    semantics = verify_semantics(paths["bundle"], contract)
    records = core.validate_ledger(paths["ledger"])
    if records:
        raise AssertionError("dormant zero-eligible runtime must have an empty ledger")
    if metadata.get("outcomes_opened") is not False:
        raise AssertionError("outcome-open state drift")
    return {
        "runtime_root": str(root),
        "contract_id": contract["contract_id"],
        "activation_state": contract["eligibility_freeze"]["activation_state"],
        "eligible_cycle_ids": contract["eligibility_freeze"]["eligible_cycle_ids"],
        "eligible_cycle_count": 0,
        "issuance_permitted": False,
        "ledger_batches": 0,
        "ledger_sha256": core.sha256_file(paths["ledger"]),
        "outcomes_opened": False,
        "outcome_evaluator_present": False,
        "protected_aggregate_shadow_unchanged": True,
        "protected_aggregate_files_only_snapshot_sha256": current_snapshot[
            "snapshot_sha256"
        ],
        "broader_protected_path_snapshot_tree_sha256": contract["integrity"][
            "broader_protected_path_snapshot_tree_sha256"
        ],
        "final_certification_complete": metadata.get(
            "final_certification_complete", False
        ),
        "independent_artifact_audit_sha256": metadata.get(
            "independent_artifact_audit_sha256"
        ),
        "independent_artifact_audit_check_count": metadata.get(
            "independent_artifact_audit_check_count"
        ),
        "independent_artifact_audit_all_passed": metadata.get(
            "independent_artifact_audit_all_passed", False
        ),
        "semantic_checks": semantics,
        **core.safety_payload(),
    }


def issue(args: argparse.Namespace) -> dict[str, Any]:
    """Fail before reading the supplied file when the eligibility set is empty."""

    root = Path(args.runtime_root)
    verify_runtime(root)
    contract = read_json(runtime_paths(root)["contract"])
    if not contract["eligibility_freeze"]["eligible_cycle_ids"]:
        raise core.DormantNoEligibleCycles(
            "quality shadow is dormant: all twenty frozen cycles are globally "
            "unqualified, so no candidate batch was read"
        )
    # Unreachable for v1. Kept to make the schema validator explicit and tested.
    candidate = pd.read_parquet(Path(args.batch))
    validated = core.validate_prediction_batch(candidate, contract)
    return {"validated_rows": len(validated), **core.safety_payload()}


def self_test() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    checks = core.validate_contract(contract)
    snapshot = verify_protected_aggregate()
    return {
        "contract_checks": len(checks),
        "dormant": True,
        "eligible_cycle_count": 0,
        "aggregate_snapshot_sha256": snapshot["snapshot_sha256"],
        "outcomes_opened": False,
        **core.safety_payload(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    init.add_argument(
        "--quality-artifact-root", type=Path, default=DEFAULT_QUALITY_ROOT
    )
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    finalize.add_argument(
        "--quality-artifact-root", type=Path, default=DEFAULT_QUALITY_ROOT
    )
    status = subparsers.add_parser("status")
    status.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    issue_parser = subparsers.add_parser("issue")
    issue_parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    issue_parser.add_argument("--batch", type=Path, required=True)
    subparsers.add_parser("self-test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "init":
            result = init_runtime(args)
        elif args.command == "finalize":
            result = finalize_runtime(args)
        elif args.command == "status":
            result = verify_runtime(Path(args.runtime_root))
        elif args.command == "issue":
            result = issue(args)
        elif args.command == "self-test":
            result = self_test()
        else:
            raise AssertionError(args.command)
    except core.DormantNoEligibleCycles as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    print(json.dumps(core.safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
