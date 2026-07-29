"""Build the immutable M1C Opening Reversal V1.1 timing addendum.

This builder reads only the already-frozen V1 activation/configuration and
empty artifact schemas.  It does not read a 2026 outcome, connect to IBKR, or
expose any broker mutation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    load_activation_receipt_v1,
    load_frozen_experiment_config_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    build_activation_receipt_v1_1,
    build_frozen_timing_addendum_config_v1_1,
    load_activation_receipt_v1_1,
    load_frozen_timing_addendum_config_v1_1,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
BASE_V1_ROOT = ROOT / "research" / "prospective" / "20260729-m1c-prospective-opening-reversal-v1"
BASE_V1_ARTIFACT_ROOT = BASE_V1_ROOT / "artifacts" / "primary"
BASE_V1_ACTIVATION_PATH = BASE_V1_ARTIFACT_ROOT / "experiment_activation_receipt_v1.json"
BASE_V1_CONFIG_PATH = BASE_V1_ARTIFACT_ROOT / "frozen_experiment_configuration_v1.json"
BASE_V1_RULE_PATH = BASE_V1_ARTIFACT_ROOT / "frozen_rule_manifest_v1.json"
PUBLISHED_ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts" / "primary"
PUBLISHED_REPORT_ROOT = EXPERIMENT_ROOT / "reports"
ARTIFACT_ROOT = PUBLISHED_ARTIFACT_ROOT
REPORT_ROOT = PUBLISHED_REPORT_ROOT
ACTIVATION_PATH = ARTIFACT_ROOT / "experiment_activation_receipt_v1_1.json"
_GIT_STATE_OVERRIDE: tuple[str, str, str] | None = None

RUNTIME_TABLES_V1_1 = (
    "market_data_capacity_snapshot_v1_1.csv",
    "prediction_receipts_v1_1.csv",
    "eligible_episode_table_v1_1.csv",
    "promoted_episode_table_v1_1.csv",
    "non_promoted_eligible_episode_table_v1_1.csv",
    "underlying_outcomes_v1_1.csv",
    "primary_option_bid_ask_outcomes_v1_1.csv",
    "optional_comparison_outcomes_v1_1.csv",
    "optional_feed_degradation_events_v1_1.csv",
    "contract_discovery_audit_v1_1.csv",
    "baseline_comparisons_v1_1.csv",
    "a1_comparisons_v1_1.csv",
    "stock_response_descriptive_strata_v1_1.csv",
    "session_cluster_bootstrap_results_v1_1.csv",
    "event_cluster_bootstrap_results_v1_1.csv",
    "primary_null_results_v1_1.csv",
    "temporal_placebo_results_v1_1.csv",
    "leave_one_stock_out_results_v1_1.csv",
    "leave_one_session_out_results_v1_1.csv",
    "leave_one_event_out_results_v1_1.csv",
    "concentration_report_v1_1.csv",
)

CAUSAL_BARRIER_COLUMNS = (
    "experiment_id",
    "experiment_version",
    "activation_receipt_hash_v1_1",
    "session",
    "nominal_entry_timestamp_utc",
    "prediction_receipt_count",
    "prediction_receipt_hashes",
    "deferred_event_count",
    "first_deferred_event_received_at_utc",
    "entry_or_post_entry_data_admitted_before_receipts",
    "raw_event_archive_write_allowed",
    "core_recorder_continued",
    "barrier_status",
    "failure_reason",
    "release_authorized_at_utc",
    "audit_hash_v1_1",
)

ENGINEERING_COLUMNS = (
    "session",
    "cohort_phase",
    "prediction_receipt_count",
    "causal_barrier_status",
    "causal_barrier_audit_hash_v1_1",
    "bar_timing_pass",
    "six_bar_window_pass",
    "ibkr_eodhd_opening_return_agreement",
    "ibkr_eodhd_opening_range_agreement",
    "severe_state_agreement",
    "capacity_reserve_pass",
    "primary_option_pair_availability",
    "core_m1c_universe_uninterrupted",
    "no_order_guard_pass",
    "valid_transfer_session",
    "report_hash_v1_1",
)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(value),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_empty_csv(path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(columns)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["rtk", "proxy", "git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _base_v1_csv_columns(v1_1_name: str) -> tuple[str, ...]:
    v1_name = v1_1_name.replace("_v1_1.csv", "_v1.csv")
    path = BASE_V1_ARTIFACT_ROOT / v1_name
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), ())
    if not header:
        raise RuntimeError(f"base_v1_schema_missing:{v1_name}")
    return tuple(header)


def _pending_receipt(
    *,
    receipt_type: str,
    activation_timestamp: str,
    reason: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": "m1c-prospective-opening-reversal-v1",
        "experiment_version": "1.1",
        "receipt_type": receipt_type,
        "status": "pending",
        "activation_timestamp_utc": activation_timestamp,
        "reason": reason,
        "scientific_result_issued": False,
        "protected_outcomes_opened": False,
    }
    payload["receipt_hash_v1_1"] = _sha256_value(payload)
    return payload


def _source_hashes() -> dict[str, str]:
    paths = (
        "packages/stocker_prospective/src/stocker_prospective/"
        "m1c_prospective_opening_reversal_v1.py",
        "packages/stocker_prospective/src/stocker_prospective/"
        "m1c_prospective_opening_reversal_v1_1.py",
        "packages/stocker_prospective/src/stocker_prospective/m1c_opening_reversal_analysis_v1.py",
        "packages/stocker_prospective/src/stocker_prospective/recorder_v0.py",
        "packages/stocker_prospective/src/stocker_prospective/live_recorder.py",
        "packages/stocker_prospective/src/stocker_prospective/"
        "frozen_live_application.py",
        "packages/stocker_prospective/src/stocker_prospective/recorder_repository.py",
        "packages/stocker_prospective/src/stocker_prospective/option_discovery.py",
        "packages/stocker_prospective/src/stocker_prospective/config.py",
        "packages/stocker_prospective/src/stocker_prospective/"
        "migrations/0015_m1c_prospective_opening_reversal_v1_1.sql",
        "configs/prospective/server.example.yaml",
        "research/prospective/"
        "20260729-m1c-prospective-opening-reversal-v1-1/"
        "build_activation_artifacts.py",
        "research/prospective/"
        "20260729-m1c-prospective-opening-reversal-v1-1/README.md",
    )
    return {path: _sha256_file(ROOT / path) for path in paths}


def _build_activation_package() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    base_activation = load_activation_receipt_v1(str(BASE_V1_ACTIVATION_PATH))
    base_config = load_frozen_experiment_config_v1(str(BASE_V1_CONFIG_PATH))
    base_rule = json.loads(BASE_V1_RULE_PATH.read_text(encoding="utf-8"))
    if (
        base_activation.configuration_hash != base_config.configuration_hash
        or base_activation.frozen_rule_hash != str(base_rule["rule_hash"])
    ):
        raise RuntimeError("base_v1_activation_binding_invalid")

    activation_timestamp = datetime.now(UTC)
    if _GIT_STATE_OVERRIDE is None:
        branch = _git("branch", "--show-current")
        commit = _git("rev-parse", "HEAD")
        dirty_status = _git("status", "--porcelain=v1") or "clean"
    else:
        branch, commit, dirty_status = _GIT_STATE_OVERRIDE
    timing_config = build_frozen_timing_addendum_config_v1_1(
        superseded_activation_receipt_hash_v1=(base_activation.activation_receipt_hash),
        frozen_rule_hash_v1=base_activation.frozen_rule_hash,
        frozen_configuration_hash_v1=(base_activation.configuration_hash),
    )
    activation = build_activation_receipt_v1_1(
        activation_timestamp_utc=activation_timestamp,
        new_york_trading_date_at_activation=(
            activation_timestamp.astimezone(ZoneInfo("America/New_York")).date()
        ),
        branch=branch,
        commit=commit,
        dirty_working_tree_status=dirty_status,
        timing_addendum_config=timing_config,
        superseded_activation_receipt=base_activation,
        m1c_version=base_activation.m1c_version,
        tail_phase_version=base_activation.tail_phase_version,
        a1_version=base_activation.a1_version,
    )
    _write_json(
        ARTIFACT_ROOT / "frozen_timing_addendum_configuration_v1_1.json",
        timing_config,
    )
    _write_json(
        ARTIFACT_ROOT / "frozen_rule_manifest_v1_1.json",
        {
            "schema_version": ("m1c-prospective-opening-reversal-rule-manifest-v1.1"),
            "experiment_id": activation.experiment_id,
            "experiment_version": "1.1",
            "scientific_rule_source": str(BASE_V1_RULE_PATH.relative_to(ROOT)),
            "scientific_rule_hash": (base_activation.frozen_rule_hash),
            "scientific_rule_changed": False,
            "prediction": {
                "NEGATIVE_SEVERE_OPENING_TRANSITION": "CALL",
                "POSITIVE_SEVERE_OPENING_TRANSITION": "PUT",
                "every_other_state": "ABSTAIN",
                "formula": ("prediction_sign_v1=-opening_transition_sign_v1"),
            },
            "checkpoint": 6,
            "m1c_probability_threshold": 0.488333710794033,
            "tail_phase": "FIRST_ENTRY",
            "nominal_entry": "10:00 America/New_York",
            "primary_horizon_minutes": 15,
            "forbidden_action_inputs": [
                "A1",
                "RSI",
                "stock_amplification_or_resistance",
                "recent_momentum",
                "market_sector_features",
                "microstructure",
                "option_prices",
                "option_spreads",
                "chart_quality",
                "later_outcomes",
            ],
        },
    )
    _write_json(
        ARTIFACT_ROOT / "timing_addendum_manifest_v1_1.json",
        {
            "schema_version": ("m1c-opening-reversal-causal-timing-manifest-v1.1"),
            "receipt_contract": timing_config.receipt_contract,
            "sixth_bar_may_complete_at_nominal_entry": True,
            "raw_append_only_archival_before_receipt_allowed": True,
            "decision_surface_release_sequence": [
                "complete_sixth_predictor_bar",
                "score_frozen_20_stock_cohort",
                "persist_all_20_prediction_receipts",
                "persist_causal_barrier_audit",
                "release_buffered_entry_or_post_entry_data",
            ],
            "nominal_entry_actionable": False,
            "barrier_failure": ("fail_scientific_session_closed_and_continue_core_recorder"),
            "engineering_transfer_sessions_restart": 20,
            "fresh_run_required": True,
        },
    )
    _write_json(
        ARTIFACT_ROOT / "api_market_data_capacity_manifest_v1_1.json",
        {
            "schema_version": ("api-market-data-capacity-manifest-v1.1"),
            "base_manifest": str(
                (BASE_V1_ARTIFACT_ROOT / "api_market_data_capacity_manifest_v1.json").relative_to(
                    ROOT
                )
            ),
            "base_manifest_sha256": _sha256_file(
                BASE_V1_ARTIFACT_ROOT / "api_market_data_capacity_manifest_v1.json"
            ),
            "capacity_policy_changed": False,
            "reserved_position_monitoring_and_safety_lines": 12,
            "reserve_may_fund_optional_research": False,
            "maximum_promoted_underlyings": 1,
            "mandatory_option_streams": [
                "one_primary_1dte_call",
                "one_primary_1dte_put",
            ],
            "full_chain_live_subscription_created": False,
            "level_ii_required": False,
            "primary_pair_capacity_failure": ("option_economics_blocked_capacity"),
        },
    )
    _write_json(
        ARTIFACT_ROOT / "subscription_priority_manifest_v1_1.json",
        {
            "schema_version": "subscription-priority-manifest-v1.1",
            "capacity_policy_changed": False,
            "priority": [
                "critical_connection_session_clock_health",
                "VTI_and_required_critical_market_proxies",
                "frozen_20_stock_m1c_five_minute_bars",
                "frozen_m1c_scoring_and_episode_receipts",
                "one_promoted_episode_underlying",
                "primary_1dte_option_call_and_put",
                "optional_comparison_contracts",
                "single_promoted_underlying_tick_by_tick",
                "optional_additional_underlying_quote_detail",
                "level_ii_separate_authorisation_only",
                "neutral_controls_and_additional_expiries",
            ],
            "degradation_order": [
                "neutral_control",
                "additional_strike",
                "3_to_5_dte_comparison",
                "0dte_comparison",
                "tick_by_tick",
                "additional_underlying_diagnostic",
            ],
        },
    )

    _write_empty_csv(
        ARTIFACT_ROOT / "causal_barrier_audits_v1_1.csv",
        CAUSAL_BARRIER_COLUMNS,
    )
    _write_empty_csv(
        ARTIFACT_ROOT / "engineering_session_report_v1_1.csv",
        ENGINEERING_COLUMNS,
    )
    for filename in RUNTIME_TABLES_V1_1:
        _write_empty_csv(
            ARTIFACT_ROOT / filename,
            _base_v1_csv_columns(filename),
        )

    activation_iso = activation.activation_timestamp_utc.isoformat()
    pending_reason = "awaiting_20_post_v1_1_engineering_transfer_sessions"
    _write_json(
        ARTIFACT_ROOT / "transfer_report_v1_1.json",
        {
            "experiment_version": "1.1",
            "status": "pending",
            "cohort_phase": "engineering_transfer",
            "valid_sessions_required": 20,
            "valid_sessions_observed": 0,
            "source_mapping_frozen": False,
            "outcomes_opened": False,
            "reason": pending_reason,
        },
    )
    for filename, receipt_type in (
        ("transfer_decision_receipt_v1_1.json", "transfer_decision"),
        (
            "development_decision_receipt_v1_1.json",
            "development_decision",
        ),
        (
            "confirmation_start_receipt_v1_1.json",
            "confirmation_start",
        ),
        (
            "confirmation_decision_receipt_v1_1.json",
            "confirmation_decision",
        ),
        (
            "option_economics_decision_receipt_v1_1.json",
            "option_economics_decision",
        ),
    ):
        _write_json(
            ARTIFACT_ROOT / filename,
            _pending_receipt(
                receipt_type=receipt_type,
                activation_timestamp=activation_iso,
                reason=pending_reason,
            ),
        )

    routing = {
        "event_count_reconciliation": str(
            (
                BASE_V1_ARTIFACT_ROOT / "opening_transition_event_count_reconciliation_v1.csv"
            ).relative_to(ROOT)
        ),
        "corrected_retrospective_event_summary": str(
            (
                BASE_V1_ROOT
                / "reports"
                / "corrected_retrospective_opening_transition_event_summary_v2.md"
            ).relative_to(ROOT)
        ),
        "base_frozen_experiment_configuration": str(BASE_V1_CONFIG_PATH.relative_to(ROOT)),
        "base_frozen_rule_manifest": str(BASE_V1_RULE_PATH.relative_to(ROOT)),
        "v1_1_timing_configuration": ("frozen_timing_addendum_configuration_v1_1.json"),
        "v1_1_runtime_artifacts": [
            "causal_barrier_audits_v1_1.csv",
            "engineering_session_report_v1_1.csv",
            *RUNTIME_TABLES_V1_1,
        ],
    }
    _write_json(
        ARTIFACT_ROOT / "required_artifact_routing_v1_1.json",
        routing,
    )
    source_hashes = _source_hashes()
    provenance = {
        "schema_version": "m1c-opening-reversal-provenance-v1.1",
        "branch": branch,
        "commit": commit,
        "dirty_working_tree_status": dirty_status,
        "activation_timestamp_utc": activation_iso,
        "configuration_hash": (timing_config.configuration_hash_v1_1),
        "scientific_rule_hash": (base_activation.frozen_rule_hash),
        "base_v1_activation_receipt_hash": (base_activation.activation_receipt_hash),
        "vti_thresholds": {
            "opening_return_q10": -0.00288963733897,
            "opening_return_q90": 0.00225522676046,
            "opening_range_q75": 0.00384818171835,
        },
        "m1c_version": base_activation.m1c_version,
        "tail_phase_version": base_activation.tail_phase_version,
        "a1_version": base_activation.a1_version,
        "contract_selection_version": (base_activation.option_selection_version),
        "capacity_manager_version": (base_activation.capacity_manager_version),
        "recorder_schema_version": (activation.recorder_schema_version),
        "configured_line_budget": None,
        "reserved_line_count": 12,
        "data_sources": ["IBKR_live", "EODHD_transfer_comparison"],
        "transfer_status": "engineering_transfer_pending",
        "episode_count": 0,
        "event_count": 0,
        "capacity_denials": 0,
        "contract_discovery_failures": 0,
        "commands": [
            (
                "rtk uv run python research/prospective/"
                "20260729-m1c-prospective-opening-reversal-v1-1/"
                "build_activation_artifacts.py --activate"
            ),
            (
                "rtk uv run python research/prospective/"
                "20260729-m1c-prospective-opening-reversal-v1-1/"
                "build_activation_artifacts.py --verify"
            ),
        ],
        "seeds": {
            "primary_null": 2026072901,
            "session_cluster": 2026072902,
            "event_cluster": 2026072903,
        },
        "source_file_sha256": source_hashes,
        "protected_pre_activation_outcome_opened": False,
        "broker_order_routing_enabled": False,
        "broker_order_placed": False,
    }
    _write_json(
        ARTIFACT_ROOT / "provenance_manifest_v1_1.json",
        provenance,
    )
    summary = {
        "experiment_id": activation.experiment_id,
        "experiment_version": "1.1",
        "status": "activated_engineering_transfer_pending",
        "activation_timestamp_utc": activation_iso,
        "scientific_rule_changed": False,
        "timing_contract_changed": True,
        "exact_prediction_rule": ("prediction_sign_v1=-opening_transition_sign_v1"),
        "nominal_entry_actionable": False,
        "reserved_line_count": 12,
        "engineering_transfer_sessions_required": 20,
        "development_episode_count": 0,
        "confirmation_episode_count": 0,
        "scientific_result_issued": False,
        "option_profitability_claim_issued": False,
        "orders_placed": 0,
    }
    _write_json(
        ARTIFACT_ROOT / "summary_v1_1.json",
        summary,
    )
    report = f"""# M1C Prospective Opening Reversal V1.1

Activation: `{activation_iso}`

## Event accounting

V1's versioned reconciliation remains authoritative:
`event_label_ambiguity_corrected`. The discrepancy was a population-label
ambiguity, not an event-construction bug, and the prior interpretation remains
`blocked_insufficient_support`.

## Frozen scientific rule

V1.1 does not change the scientific rule. At checkpoint 6, a fresh high-M1C
`FIRST_ENTRY` episode opposes the severe VTI opening-transition sign:
negative severe → `CALL`; positive severe → `PUT`; otherwise `ABSTAIN`.

## Timing addendum

The sixth 09:55–10:00 bar must finish before the predictor exists. Therefore
V1.1 allows receipt creation just after the nominal 10:00 boundary only while
all entry/post-entry decision data remains behind a causal barrier. Raw
append-only archival may continue. All 20 receipts and the barrier audit must
be durable before buffered data is released. The nominal entry is not
actionable and no order may be placed.

## API and capacity operation

The 12-line reserve and the V1 subscription/degradation policy are unchanged.
VTI and all 20 M1C bar feeds remain mandatory. At most one promoted underlying
and one primary 1DTE call/put pair are streamed. Optional feeds fail closed
without borrowing reserve capacity.

## Engineering transfer

The engineering-transfer clock restarts at zero and requires 20 valid sessions
recorded after this activation. IBKR/EODHD predictor agreement, timing,
contract discovery, capacity, graceful degradation, recorder reliability, and
no-order safeguards may be inspected. Scientific outcomes remain unopened.

## Prospective development

Pending. No post-activation engineering session or scientific outcome was
available when this package was signed.

## Prospective confirmation

Pending and untouched.

## Option economics

Pending. Any later claim requires actual ask-entry/bid-exit evidence and the
separate frozen support contract. Midpoints are diagnostic only.

## Execution realism

This is research-only shadow recording. No actual fills, orders, or order
routing are present.
"""
    (REPORT_ROOT / "m1c_prospective_opening_reversal_v1_1.md").write_text(
        report,
        encoding="utf-8",
    )
    _write_json(ACTIVATION_PATH, activation)

    manifest_entries = {
        f"artifacts/primary/{path.relative_to(ARTIFACT_ROOT)}": (_sha256_file(path))
        for path in sorted(ARTIFACT_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_v1_1.json"
    }
    manifest_entries.update(
        {
            f"reports/{path.relative_to(REPORT_ROOT)}": (_sha256_file(path))
            for path in sorted(REPORT_ROOT.rglob("*"))
            if path.is_file()
        }
    )
    manifest_payload: dict[str, Any] = {
        "schema_version": ("m1c-opening-reversal-artifact-manifest-v1.1"),
        "experiment_id": activation.experiment_id,
        "experiment_version": "1.1",
        "activation_receipt_hash_v1_1": (activation.activation_receipt_hash_v1_1),
        "files": manifest_entries,
    }
    manifest_payload["manifest_hash_v1_1"] = _sha256_value(manifest_payload)
    _write_json(
        ARTIFACT_ROOT / "artifact_manifest_v1_1.json",
        manifest_payload,
    )


def _verify_package(
    *,
    artifact_root: Path,
    report_root: Path,
    emit: bool = True,
) -> dict[str, Any]:
    activation_path = artifact_root / "experiment_activation_receipt_v1_1.json"
    config_path = artifact_root / "frozen_timing_addendum_configuration_v1_1.json"
    manifest_path = artifact_root / "artifact_manifest_v1_1.json"
    for path in (
        activation_path,
        config_path,
        manifest_path,
        report_root / "m1c_prospective_opening_reversal_v1_1.md",
    ):
        if not path.is_file():
            raise RuntimeError(f"activation_package_file_missing:{path.name}")
    activation = load_activation_receipt_v1_1(str(activation_path))
    config = load_frozen_timing_addendum_config_v1_1(str(config_path))
    base = load_activation_receipt_v1(str(BASE_V1_ACTIVATION_PATH))
    if (
        activation.timing_addendum_configuration_hash_v1_1 != config.configuration_hash_v1_1
        or activation.superseded_activation_receipt_hash_v1 != base.activation_receipt_hash
        or activation.frozen_rule_hash != base.frozen_rule_hash
        or activation.frozen_configuration_hash_v1 != base.configuration_hash
    ):
        raise RuntimeError("activation_package_binding_mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_hash_v1_1", None)
    if manifest_hash != _sha256_value(manifest):
        raise RuntimeError("artifact_manifest_hash_mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("artifact_manifest_files_invalid")
    for relative, expected_hash in files.items():
        candidate = EXPERIMENT_ROOT / str(relative)
        if artifact_root != PUBLISHED_ARTIFACT_ROOT:
            relative_path = Path(str(relative))
            if relative_path.parts[:2] == ("artifacts", "primary"):
                candidate = artifact_root.joinpath(*relative_path.parts[2:])
            elif relative_path.parts[:1] == ("reports",):
                candidate = report_root.joinpath(*relative_path.parts[1:])
        if not candidate.is_file() or _sha256_file(candidate) != expected_hash:
            raise RuntimeError(f"artifact_hash_mismatch:{relative}")
    provenance = json.loads(
        (
            artifact_root / "provenance_manifest_v1_1.json"
        ).read_text(encoding="utf-8")
    )
    source_hashes = provenance.get("source_file_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("provenance_source_hashes_invalid")
    for relative, expected_hash in source_hashes.items():
        source = ROOT / str(relative)
        if (
            not source.is_file()
            or _sha256_file(source) != expected_hash
        ):
            raise RuntimeError(
                f"source_hash_mismatch:{relative}"
            )
    result = {
        "status": "verified",
        "experiment_version": "1.1",
        "activation_timestamp_utc": (activation.activation_timestamp_utc.isoformat()),
        "activation_receipt_hash_v1_1": (activation.activation_receipt_hash_v1_1),
        "configuration_hash_v1_1": (config.configuration_hash_v1_1),
        "scientific_rule_hash": activation.frozen_rule_hash,
        "file_count": len(files),
    }
    if emit:
        print(json.dumps(result, sort_keys=True))
    return result


def _copy_idempotent(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
            raise RuntimeError(f"activation package already exists:{destination}")
        return
    shutil.copy2(source, destination)


def _publish_staged_package(
    *,
    staged_artifact_root: Path,
    staged_report_root: Path,
    published_artifact_root: Path,
    published_report_root: Path,
    before_artifact_publish: Callable[[], None] | None = None,
) -> None:
    staged_activation = staged_artifact_root / "experiment_activation_receipt_v1_1.json"
    published_activation = published_artifact_root / "experiment_activation_receipt_v1_1.json"
    if published_activation.exists():
        raise RuntimeError("immutable V1.1 activation package already exists")
    for source in sorted(staged_artifact_root.rglob("*")):
        if not source.is_file() or source == staged_activation:
            continue
        _copy_idempotent(
            source,
            published_artifact_root / source.relative_to(staged_artifact_root),
        )
    for source in sorted(staged_report_root.rglob("*")):
        if source.is_file():
            _copy_idempotent(
                source,
                published_report_root / source.relative_to(staged_report_root),
            )
    if before_artifact_publish is not None:
        before_artifact_publish()
    _copy_idempotent(staged_activation, published_activation)


def _activate() -> None:
    if (PUBLISHED_ARTIFACT_ROOT / "experiment_activation_receipt_v1_1.json").exists():
        raise RuntimeError("immutable V1.1 activation package already exists")
    global ARTIFACT_ROOT, REPORT_ROOT, ACTIVATION_PATH
    global _GIT_STATE_OVERRIDE
    git_state = (
        _git("branch", "--show-current"),
        _git("rev-parse", "HEAD"),
        _git("status", "--porcelain=v1") or "clean",
    )
    with tempfile.TemporaryDirectory(
        prefix="m1c-opening-reversal-v1-1-",
    ) as temporary:
        staging = Path(temporary)
        staged_artifacts = staging / "artifacts" / "primary"
        staged_reports = staging / "reports"
        prior = (
            ARTIFACT_ROOT,
            REPORT_ROOT,
            ACTIVATION_PATH,
            _GIT_STATE_OVERRIDE,
        )
        ARTIFACT_ROOT = staged_artifacts
        REPORT_ROOT = staged_reports
        ACTIVATION_PATH = staged_artifacts / "experiment_activation_receipt_v1_1.json"
        _GIT_STATE_OVERRIDE = git_state
        try:
            _build_activation_package()
            _verify_package(
                artifact_root=staged_artifacts,
                report_root=staged_reports,
                emit=False,
            )
            _publish_staged_package(
                staged_artifact_root=staged_artifacts,
                staged_report_root=staged_reports,
                published_artifact_root=PUBLISHED_ARTIFACT_ROOT,
                published_report_root=PUBLISHED_REPORT_ROOT,
            )
        finally:
            (
                ARTIFACT_ROOT,
                REPORT_ROOT,
                ACTIVATION_PATH,
                _GIT_STATE_OVERRIDE,
            ) = prior
    _verify_package(
        artifact_root=PUBLISHED_ARTIFACT_ROOT,
        report_root=PUBLISHED_REPORT_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--activate", action="store_true")
    mode.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.activate:
        _activate()
    else:
        _verify_package(
            artifact_root=PUBLISHED_ARTIFACT_ROOT,
            report_root=PUBLISHED_REPORT_ROOT,
        )


if __name__ == "__main__":
    main()
