"""Bounded stage orchestration for the clean-slate V1 lineage."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stocker_execution.ibkr_observability.config import IBKRObserverConfig
from stocker_execution.ibkr_observability.official_api import official_api_status
from stocker_research.observable_event_ranking_v1.artifacts import (
    ArtifactBinding,
    ArtifactWriter,
    sha256_file,
)
from stocker_research.observable_event_ranking_v1.contract import (
    DEVELOPMENT_CUTOFF,
    REQUIRED_SAFETY_FLAGS,
    canonical_hash,
    frozen_contract,
)
from stocker_research.observable_event_ranking_v1.provenance import audit_primary_imports
from stocker_research.observable_event_ranking_v1.sector_context import (
    validate_sector_membership_ledger,
)

_RAW_FILE_PATTERN = re.compile(r"(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.json$")


class StageDependencyError(RuntimeError):
    """Raised when a later scientific stage is not authorised."""


@dataclass(frozen=True)
class PreflightResult:
    """Outcome-free preflight and artifact identity."""

    decision: str
    output_dir: Path
    binding: ArtifactBinding


def _inventory_files(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = data_dir / "raw/source=eodhd/endpoint=intraday"
    safe: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    if not root.exists():
        return safe, protected
    for path in sorted(root.glob("symbol=*/interval=5m/*.json")):
        match = _RAW_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        relative = path.relative_to(data_dir).as_posix()
        identity = {
            "source_provider": "EODHD",
            "source_file_identity": relative,
            "from_date": match.group("start"),
            "to_date": match.group("end"),
            "size_bytes": path.stat().st_size,
            "market_rows_parsed": False,
        }
        is_safe = match.group("end") <= DEVELOPMENT_CUTOFF
        identity["source_content_sha256"] = sha256_file(path) if is_safe else None
        identity["content_hashed"] = is_safe
        identity["inventory_identity_hash"] = canonical_hash(identity)
        (safe if is_safe else protected).append(identity)
    return safe, protected


def _implementation_manifest(repository_root: Path) -> tuple[dict[str, Any], str]:
    roots = (
        repository_root
        / "packages/stocker_research/src/stocker_research/observable_event_ranking_v1",
        repository_root / "packages/stocker_execution/src/stocker_execution/ibkr_observability",
        repository_root / "research/observable-event-ranking/"
        "20260719-observable-event-cross-sectional-ranking-v1/work",
    )
    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts or "artifacts" in path.parts:
                continue
            entries.append(
                {
                    "path": path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    implementation_hash = canonical_hash(entries)
    return (
        {
            "manifest_version": "observable_event_ranking_v1_implementation_sources",
            "implementation_hash": implementation_hash,
            "source_files": entries,
        },
        implementation_hash,
    )


def _environment_manifest() -> dict[str, Any]:
    packages = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "pandas-market-calendars",
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return {
        "manifest_version": "observable_event_ranking_v1_environment",
        "python_implementation": platform.python_implementation(),
        "dependency_versions": versions,
        "determinism": {
            "worker_count": 1,
            "stable_row_order": True,
            "single_thread_numerical_policy": True,
            "python_hash_seed_required_for_full_run": "0",
            "numerical_thread_environment": {
                variable: os.getenv(variable, "not_set")
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "safety": REQUIRED_SAFETY_FLAGS,
    }


def _empty_frame(schema: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})


def _add_binding(payload: dict[str, Any], binding: ArtifactBinding) -> dict[str, Any]:
    return {**payload, "artifact_binding": binding.to_dict(), "safety": REQUIRED_SAFETY_FLAGS}


def run_preflight(
    *,
    repository_root: Path,
    data_dir: Path,
    output_dir: Path,
    git_sha: str,
    branch: str,
    sector_ledger_path: Path | None = None,
    max_symbols: int | None = None,
    max_sessions: int | None = None,
) -> PreflightResult:
    """Inventory only permitted files, fail closed, and write all pre-target artifacts."""

    contract = frozen_contract()
    contract_hash = canonical_hash(contract)
    implementation_manifest, implementation_hash = _implementation_manifest(repository_root)
    primary_package = (
        repository_root
        / "packages/stocker_research/src/stocker_research/observable_event_ranking_v1"
    )
    provenance_violations = audit_primary_imports(sorted(primary_package.glob("*.py")))
    environment = _environment_manifest()
    safe_files, protected_files = _inventory_files(data_dir)
    safe_symbols = sorted(
        {
            identity["source_file_identity"].split("symbol=", 1)[1].split("/", 1)[0]
            for identity in safe_files
        }
    )
    inventory_snapshot = {
        "safe_files": safe_files,
        "protected_files": protected_files,
        "development_cutoff": DEVELOPMENT_CUTOFF,
    }
    data_snapshot_hash = canonical_hash(inventory_snapshot)
    scientific_run = max_symbols is None and max_sessions is None

    sector_ledger = _empty_frame(
        {
            "symbol": "string",
            "sector": "string",
            "effective_from": "datetime64[ns, UTC]",
            "effective_to": "datetime64[ns, UTC]",
            "known_at": "datetime64[ns, UTC]",
            "stable_source_id": "string",
            "source_provider": "string",
            "source_dataset_id": "string",
            "source_hash": "string",
        }
    )
    sector_issues = ["missing_point_in_time_sector_membership"]
    if sector_ledger_path is not None and sector_ledger_path.exists():
        sector_ledger = pd.read_parquet(sector_ledger_path)
        sector_issues = validate_sector_membership_ledger(sector_ledger)
    universe_ledger = _empty_frame(
        {
            "effective_date": "datetime64[ns, UTC]",
            "symbol": "string",
            "stable_source_id": "string",
            "source_provider": "string",
            "source_dataset_id": "string",
            "source_hash": "string",
            "eligible": "bool",
            "qualification_reasons": "string",
        }
    )
    universe_hash = canonical_hash(
        {"schema": list(universe_ledger.columns), "rows": 0, "reason": "preflight_blocked"}
    )
    sector_input_hash = (
        sha256_file(sector_ledger_path)
        if sector_ledger_path is not None and sector_ledger_path.exists()
        else canonical_hash({"status": "missing_point_in_time_sector_membership"})
    )
    sector_hash = canonical_hash(
        {
            "schema": list(sector_ledger.columns),
            "rows": len(sector_ledger),
            "issues": sector_issues,
            "input_file_sha256": sector_input_hash,
        }
    )
    if provenance_violations:
        decision = "blocked_audit_or_reproducibility_failure"
    elif sector_issues:
        decision = "blocked_missing_point_in_time_sector_membership"
    elif len(safe_symbols) < 50:
        decision = "blocked_insufficient_historical_source_universe"
    else:
        decision = "blocked_data_or_chronology_failure"
    run_id = (
        "oer-v1-"
        + canonical_hash(
            {
                "contract_hash": contract_hash,
                "implementation_hash": implementation_hash,
                "data_snapshot_hash": data_snapshot_hash,
                "universe_hash": universe_hash,
                "sector_hash": sector_hash,
                "smoke": not scientific_run,
            }
        )[:20]
    )
    binding = ArtifactBinding(
        git_sha=git_sha,
        branch=branch,
        contract_hash=contract_hash,
        implementation_hash=implementation_hash,
        data_snapshot_hash=data_snapshot_hash,
        universe_hash=universe_hash,
        sector_map_hash=sector_hash,
        run_id=run_id,
        random_seeds={"bootstrap": 20260719, "random_baseline": 20260719},
        dependency_versions=environment["dependency_versions"],
        safety=REQUIRED_SAFETY_FLAGS,
    )
    writer = ArtifactWriter(output_dir, binding)
    writer.json("frozen_experiment_contract.json", contract)
    writer.json(
        "source_identity_manifest.json",
        _add_binding(
            {
                "manifest_version": "observable_event_ranking_v1_sources",
                "historical_provider": "EODHD",
                "safe_source_files": safe_files,
                "safe_file_count": len(safe_files),
                "safe_symbol_count": len(safe_symbols),
                "safe_symbols": safe_symbols,
                "protected_source_files": protected_files,
                "protected_file_count": len(protected_files),
                "protected_files_opened": 0,
                "processed_parquet_files_opened": 0,
                "source_content_hash_status": (
                    "safe_pre_cutoff_files_sha256_hashed_without_parsing_market_rows"
                ),
                "inventory_snapshot": inventory_snapshot,
                "data_snapshot_hash": data_snapshot_hash,
                "audit_metadata_file_count": len(
                    list((data_dir / "reports/audits").glob("*_5m_audit.json"))
                ),
                "vendor_qa_metadata_file_count": len(
                    list((data_dir / "reports/vendor_qa").glob("*_5m_eodhd_qa.json"))
                ),
                "volume_interpretation": "provider_reported_activity_proxy",
            },
            binding,
        ),
    )
    writer.json(
        "implementation_source_manifest.json",
        _add_binding(implementation_manifest, binding),
    )
    writer.json("environment_manifest.json", _add_binding(environment, binding))
    writer.parquet(
        "universe_ledger.parquet",
        universe_ledger,
        columns=tuple(universe_ledger.columns),
        sort_by=("effective_date", "symbol"),
    )
    writer.parquet(
        "sector_membership_ledger.parquet",
        sector_ledger,
        columns=tuple(sector_ledger.columns),
        sort_by=("effective_from", "symbol"),
    )
    calibration = _empty_frame(
        {
            "calibration_row_id": "string",
            "symbol": "string",
            "session": "datetime64[ns, UTC]",
            "bar_end": "datetime64[ns, UTC]",
            "recent_market_relative": "float64",
            "preceding_market_relative": "float64",
            "recent_sector_relative": "float64",
            "preceding_sector_relative": "float64",
            "market_relative_acceleration_z": "float64",
            "sector_relative_acceleration_z": "float64",
            "event_strength": "float64",
            "source_hash": "string",
        }
    )
    writer.parquet(
        "event_calibration_rows.parquet",
        calibration,
        columns=tuple(calibration.columns),
        sort_by=("session", "bar_end", "symbol"),
    )
    writer.json(
        "event_threshold.json",
        _add_binding(
            {
                "event_family": "E1_POSITIVE_RELATIVE_ACCELERATION",
                "threshold": None,
                "status": "not_fitted",
                "reason": decision,
                "outcomes_read": False,
            },
            binding,
        ),
    )
    event_ledger = _empty_frame(
        {
            "event_id": "string",
            "slate_id": "string",
            "symbol": "string",
            "session": "datetime64[ns, UTC]",
            "source_provider": "string",
            "source_dataset_id": "string",
            "source_hash": "string",
            "source_bar_start": "datetime64[ns, UTC]",
            "source_bar_end": "datetime64[ns, UTC]",
            "source_availability_timestamp": "datetime64[ns, UTC]",
            "feature_availability_time": "datetime64[ns, UTC]",
            "event_confirmation_time": "datetime64[ns, UTC]",
            "assigned_decision_time": "datetime64[ns, UTC]",
            "planned_entry_reference_time": "datetime64[ns, UTC]",
            "planned_exit_reference_time": "datetime64[ns, UTC]",
            "timezone": "string",
            "adjustment_status": "string",
            "corporate_action_status": "string",
            "gap_status": "string",
            "volume_is_provider_activity_proxy": "bool",
            "event_strength": "float64",
        }
    )
    writer.parquet(
        "event_ledger.parquet",
        event_ledger,
        columns=tuple(event_ledger.columns),
        sort_by=("assigned_decision_time", "symbol"),
    )
    slate_ledger = _empty_frame(
        {
            "slate_id": "string",
            "session": "datetime64[ns, UTC]",
            "decision_time": "datetime64[ns, UTC]",
            "eligible_stock_count": "int64",
            "candidate_count": "int64",
            "otherwise_valid_scheduled_slate": "bool",
            "supported_event_slate": "bool",
            "unavailable_reason": "string",
        }
    )
    writer.parquet(
        "slate_ledger.parquet",
        slate_ledger,
        columns=tuple(slate_ledger.columns),
        sort_by=("decision_time",),
    )
    dedup = pd.DataFrame(
        [
            {
                "raw_trigger_count": 0,
                "first_event_count": 0,
                "later_trigger_count": 0,
                "grid_assignment_count": 0,
                "rejected_count": 0,
                "unavailable_reason": decision,
            }
        ]
    )
    writer.csv(
        "event_deduplication_summary.csv",
        dedup,
        columns=tuple(dedup.columns),
    )
    for filename, dimension in (
        ("support_by_clock.csv", "decision_clock"),
        ("support_by_month.csv", "month"),
        ("support_by_stock.csv", "symbol"),
        ("support_by_sector.csv", "sector"),
    ):
        summary = _empty_frame({dimension: "string", "event_rows": "int64", "blocker": "string"})
        writer.csv(
            filename,
            summary,
            columns=(dimension, "event_rows", "blocker"),
            sort_by=(dimension,),
        )
    blocker_rows: list[dict[str, object]] = []
    if sector_issues:
        blocker_rows.append(
            {
                "stage": "preflight",
                "blocker": "blocked_missing_point_in_time_sector_membership",
                "count": len(sector_issues),
                "detail": (
                    "No trusted effective-dated sector membership was found; current snapshots "
                    "are not projected backward."
                ),
            }
        )
    if len(safe_symbols) < 50:
        blocker_rows.append(
            {
                "stage": "preflight",
                "blocker": "insufficient_historical_source_universe",
                "count": len(safe_symbols),
                "detail": "safe pre-cutoff source symbols; at least 50 are required per slate",
            }
        )
    blocker_rows.extend(
        [
            {
                "stage": "preflight",
                "blocker": "unproven_source_timestamp_convention",
                "count": 1,
                "detail": "bar start/end semantics are not proven for the scientific source",
            },
            {
                "stage": "preflight",
                "blocker": "unresolved_corporate_action_handling",
                "count": 1,
                "detail": "raw-close split and ticker handling is not resolved",
            },
        ]
    )
    blockers = pd.DataFrame(blocker_rows)
    writer.csv(
        "missingness_and_blockers.csv",
        blockers,
        columns=("stage", "blocker", "count", "detail"),
        sort_by=("stage", "blocker"),
    )
    support_decision = _add_binding(
        {
            "decision": decision,
            "passed": False,
            "scientific_run": scientific_run,
            "non_scientific_smoke_limits": {
                "max_symbols": max_symbols,
                "max_sessions": max_sessions,
            },
            "targets_permitted": False,
            "model_fit_permitted": False,
            "outcomes_read": False,
            "gates": {
                "point_in_time_sector_membership": not sector_issues,
                "safe_source_symbol_count": len(safe_symbols),
                "valid_slate_minimum_universe": 50,
                "bar_label_convention_proven": False,
                "corporate_action_handling_resolved": False,
                "static_provenance_audit": not provenance_violations,
            },
            "larger_unchanged_source_universe_may_support_new_run": True,
        },
        binding,
    )
    writer.json("support_decision.json", support_decision)
    writer.json(
        "outcome_free_event_audit.json",
        _add_binding(
            {
                "audit_passed": not provenance_violations,
                "event_rows": 0,
                "outcome_columns_present": False,
                "retired_columns_present": False,
                "result_scope": "schema_and_preflight_blocker_only",
            },
            binding,
        ),
    )
    writer.json(
        "static_provenance_audit.json",
        _add_binding(
            {
                "audit_passed": not provenance_violations,
                "primary_python_files_scanned": len(list(primary_package.glob("*.py"))),
                "forbidden_import_violations": provenance_violations,
                "forbidden_lineage_classes": [
                    "regime",
                    "loop",
                    "excursion",
                    "posterior",
                ],
            },
            binding,
        ),
    )
    _write_ibkr_preflight_artifacts(writer, binding)
    writer.manifest()
    return PreflightResult(decision=decision, output_dir=output_dir, binding=binding)


def _write_ibkr_preflight_artifacts(writer: ArtifactWriter, binding: ArtifactBinding) -> None:
    status = official_api_status()
    config = IBKRObserverConfig()
    writer.json(
        "ibkr_observability_contract.json",
        _add_binding(
            {
                "contract_version": "ibkr_observability_v1",
                "default_configuration": asdict(config),
                "official_api": asdict(status),
                "public_operations": [
                    "connect_local_session",
                    "disconnect",
                    "request_server_time",
                    "resolve_stock_contract_details",
                    "capture_top_of_book_snapshot",
                    "record_market_data_type_and_errors",
                    "cancel_observer_market_data_subscription",
                ],
                "orders_or_account_operations_exposed": False,
                "tws_read_only_api_mode_required": True,
            },
            binding,
        ),
    )
    contract_ledger = _empty_frame(
        {
            "research_symbol": "string",
            "source_provider_symbol": "string",
            "con_id": "Int64",
            "symbol": "string",
            "local_symbol": "string",
            "security_type": "string",
            "currency": "string",
            "routing_exchange": "string",
            "primary_exchange": "string",
            "trading_class": "string",
            "valid_exchanges": "string",
            "minimum_tick": "float64",
            "timezone_identifier": "string",
            "trading_hours": "string",
            "liquid_hours": "string",
            "contract_resolution_timestamp": "datetime64[ns, UTC]",
            "api_tws_version": "string",
            "resolution_status": "string",
            "resolution_error": "string",
        }
    )
    writer.parquet(
        "ibkr_contract_ledger.parquet",
        contract_ledger,
        columns=tuple(contract_ledger.columns),
        sort_by=("research_symbol",),
    )
    plan = _empty_frame(
        {
            "observation_id": "string",
            "event_id": "string",
            "decision_id": "string",
            "decision_timestamp": "datetime64[ns, UTC]",
            "planned_entry_reference_timestamp": "datetime64[ns, UTC]",
            "planned_exit_reference_timestamp": "datetime64[ns, UTC]",
            "planned_observation_timestamp": "datetime64[ns, UTC]",
            "symbol": "string",
            "con_id": "Int64",
            "required_observation_type": "string",
            "maximum_collection_delay_seconds": "float64",
            "completion_status": "string",
        }
    )
    writer.parquet(
        "ibkr_observation_plan.parquet",
        plan,
        columns=tuple(plan.columns),
        sort_by=("planned_observation_timestamp", "symbol"),
    )
    writer.json(
        "ibkr_quote_ledger_schema.json",
        _add_binding(
            {
                "schema_version": "ibkr_top_of_book_observation_v1",
                "append_only": True,
                "primary_observation": "first_complete_live_bid_ask_within_ten_seconds",
                "fields": [
                    "observation_id",
                    "event_id",
                    "decision_id",
                    "request_id",
                    "requested_timestamp",
                    "ibkr_server_time_observation",
                    "local_send_timestamp_utc",
                    "first_response_timestamp_utc",
                    "snapshot_completion_timestamp_utc",
                    "symbol",
                    "con_id",
                    "exchange",
                    "primary_exchange",
                    "bid",
                    "ask",
                    "bid_size",
                    "ask_size",
                    "last",
                    "last_size",
                    "market_data_type",
                    "classification",
                    "quote_age_or_timing_uncertainty_seconds",
                    "subscription_status",
                    "snapshot_complete",
                    "error_code",
                    "error_message",
                    "connection_status",
                    "api_tws_version",
                    "source_identifier",
                    "collector_version",
                    "collector_hash",
                    "reference_uncertainty",
                    "fill_claim",
                ],
            },
            binding,
        ),
    )
    writer.json(
        "ibkr_feasibility_report.json",
        _add_binding(
            {
                "official_api_installed": status.installed,
                "connection_attempted": False,
                "subscription_status": "not_observed",
                "contract_rows": 0,
                "quote_rows": 0,
                "blocker": status.blocker,
                "no_orders_sent": True,
                "current_contract_availability_used_for_historical_universe": False,
            },
            binding,
        ),
    )
    writer.json(
        "ibkr_observability_fake_dry_run.json",
        _add_binding(
            {
                "client": "FakeIBKRObservabilityClient",
                "network_connection_opened": False,
                "synthetic_contract_resolution": "resolved",
                "synthetic_quote_classification": "LIVE_TOP_OF_BOOK_OBSERVED",
                "market_data_type": "LIVE",
                "bid": 100.0,
                "ask": 100.1,
                "fill_claim": False,
                "account_requests": 0,
                "position_requests": 0,
                "order_requests": 0,
            },
            binding,
        ),
    )


def build_targets_stage(artifact_dir: Path) -> None:
    """Refuse target construction unless the frozen support decision passes."""

    decision_path = artifact_dir / "support_decision.json"
    if not decision_path.exists():
        raise StageDependencyError("support_decision.json is required before target construction")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not bool(decision.get("passed")):
        raise StageDependencyError(
            f"target construction refused: {decision.get('decision', 'unknown_support_decision')}"
        )
