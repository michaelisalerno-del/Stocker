#!/usr/bin/env python3
"""Independently audit the blocked One-Minute Activity-Price Lead Screen V0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "observable-pressure-onset"
    / "20260720-high-movement-pressure-onset-screen-v0-1"
    / "artifacts"
    / "primary"
)
PREDECESSOR_PANEL = PREDECESSOR_PRIMARY / "compact_decision_panel.parquet"
EXPECTED_DECISION = "blocked_one_minute_history_unavailable"
DEVELOPMENT_START = pd.Timestamp("2024-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
SYMBOLS = (
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
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "observable_only": True,
    "one_minute_sequence_test": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}
EMPTY_CSV_ARTIFACTS = (
    "onset_metrics.csv",
    "direction_metrics.csv",
    "checkpoint_metrics.csv",
    "monthly_metrics.csv",
    "calibration_bins.csv",
    "feature_group_diagnostics.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "economic_reference_metrics.csv",
    "concentration_metrics.csv",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path.name}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a canonical audit object."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Return a file SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_path(provider_root: Path, symbol: str) -> Path:
    """Return the independently derived one-minute source path."""

    return provider_root / f"symbol={symbol}" / "timeframe=1m" / "data.parquet"


def compact_ordinal_ranges(values: list[int]) -> str:
    """Independently render stable inclusive integer ranges."""

    ordered = sorted(set(values))
    if not ordered:
        return ""
    output: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        output.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    output.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(output)


def read_safe_timestamps(path: Path) -> pd.Series:
    """Independently materialise only safe one-minute timestamp labels."""

    frame = pd.read_parquet(
        path,
        columns=["timestamp"],
        filters=[
            ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    timestamps = pd.Series(
        pd.to_datetime(frame["timestamp"], utc=True, errors="raise"),
        dtype="datetime64[ns, UTC]",
    ).sort_values(kind="mergesort", ignore_index=True)
    if timestamps.lt(DEVELOPMENT_START).any() or timestamps.ge(PROTECTED_START).any():
        raise AssertionError("protected timestamp materialised")
    return timestamps


def timestamp_hash(timestamps: pd.Series) -> str:
    """Independently hash safe timestamp labels."""

    canonical = "\n".join(value.isoformat() for value in timestamps.tolist())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidate_coverage(
    timestamps: pd.DatetimeIndex,
    *,
    market_open: pd.Timestamp,
    expected_minutes: int,
    convention: str,
) -> dict[str, Any]:
    """Independently map one explicit start/end convention candidate."""

    if convention not in {"bar_start", "bar_end"}:
        raise AssertionError("unknown timestamp convention candidate")
    minute_ns = pd.Timedelta(minutes=1).value
    candidate_starts = timestamps.asi8 - (minute_ns if convention == "bar_end" else 0)
    deltas_ns = candidate_starts - market_open.value
    on_grid = (
        (deltas_ns >= 0)
        & (deltas_ns < expected_minutes * minute_ns)
        & ((deltas_ns % minute_ns) == 0)
    )
    grid = (deltas_ns[on_grid] // minute_ns).astype(int).tolist()
    observed = sorted(set(grid))
    missing = sorted(set(range(expected_minutes)).difference(observed))
    duplicates = len(grid) - len(observed)
    return {
        "observed": observed,
        "missing": missing,
        "duplicate_count": duplicates,
        "off_grid_count": len(timestamps) - len(grid),
        "complete": len(observed) == expected_minutes and duplicates == 0,
    }


def assert_safety(payload: Mapping[str, Any]) -> None:
    """Verify every mandatory safety value."""

    for key, expected in SAFETY_FLAGS.items():
        if payload.get(key) != expected:
            raise AssertionError(f"safety flag differs: {key}")


def audit_availability(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    """Independently reconstruct local safe symbol/session/minute coverage."""

    coverage = pd.read_csv(
        artifacts / "one_minute_availability_audit.csv", keep_default_na=False
    ).sort_values(["symbol", "session"], kind="mergesort", ignore_index=True)
    schedule = mcal.get_calendar("XNYS").schedule(start_date="2024-01-01", end_date="2025-08-22")
    if len(schedule) != 412:
        raise AssertionError("independent XNYS session count differs")
    if len(coverage) != len(SYMBOLS) * len(schedule):
        raise AssertionError("availability row count differs")
    expected_rows: list[dict[str, Any]] = []
    source_checks: dict[str, dict[str, Any]] = {}
    safe_timestamp_values: list[pd.Timestamp] = []
    for symbol in SYMBOLS:
        path = provider_path(provider_root, symbol)
        exists = path.is_file()
        read_error_code: str | None = None
        if exists:
            try:
                timestamps = read_safe_timestamps(path)
            except AssertionError:
                raise
            except Exception as exc:  # noqa: BLE001 - mirror fail-closed ledger status.
                timestamps = pd.Series([], dtype="datetime64[ns, UTC]")
                read_error_code = type(exc).__name__
        else:
            timestamps = pd.Series([], dtype="datetime64[ns, UTC]")
        safe_timestamp_values.extend(timestamps.tolist())
        local_dates = timestamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        matched_rows = 0
        complete_sessions = 0
        bar_start_complete_sessions = 0
        bar_end_complete_sessions = 0
        for session, row in schedule.iterrows():
            session_text = pd.Timestamp(session).strftime("%Y-%m-%d")
            market_open = pd.Timestamp(row["market_open"])
            market_close = pd.Timestamp(row["market_close"])
            expected_minutes = int((market_close - market_open).total_seconds() // 60)
            values = pd.DatetimeIndex(timestamps.loc[local_dates.eq(session_text)])
            bar_start = candidate_coverage(
                values,
                market_open=market_open,
                expected_minutes=expected_minutes,
                convention="bar_start",
            )
            bar_end = candidate_coverage(
                values,
                market_open=market_open,
                expected_minutes=expected_minutes,
                convention="bar_end",
            )
            matched_rows += len(values)
            if not exists:
                status = "missing_source_file"
            elif read_error_code:
                status = "unreadable_source"
            elif bar_start["complete"] and bar_end["complete"]:
                status = "complete_under_both_candidates"
                complete_sessions += 1
                bar_start_complete_sessions += 1
                bar_end_complete_sessions += 1
            elif bar_start["complete"]:
                status = "complete_under_bar_start_candidate"
                complete_sessions += 1
                bar_start_complete_sessions += 1
            elif bar_end["complete"]:
                status = "complete_under_bar_end_candidate"
                complete_sessions += 1
                bar_end_complete_sessions += 1
            elif bar_start["observed"] or bar_end["observed"]:
                status = "partial"
            else:
                status = "missing_session"
            expected_rows.append(
                {
                    "symbol": symbol,
                    "session": session_text,
                    "expected_minute_count": expected_minutes,
                    "ordinal_basis": "dual_convention_candidates_pending_empirical_proof",
                    "bar_start_candidate_observed_minute_count": len(bar_start["observed"]),
                    "bar_start_candidate_observed_minute_ordinals": compact_ordinal_ranges(
                        bar_start["observed"]
                    ),
                    "bar_start_candidate_missing_minute_count": len(bar_start["missing"]),
                    "bar_start_candidate_missing_minute_ordinals": compact_ordinal_ranges(
                        bar_start["missing"]
                    ),
                    "bar_start_candidate_duplicate_minute_count": bar_start["duplicate_count"],
                    "bar_start_candidate_off_grid_minute_count": bar_start["off_grid_count"],
                    "bar_end_candidate_observed_minute_count": len(bar_end["observed"]),
                    "bar_end_candidate_observed_minute_ordinals": compact_ordinal_ranges(
                        bar_end["observed"]
                    ),
                    "bar_end_candidate_missing_minute_count": len(bar_end["missing"]),
                    "bar_end_candidate_missing_minute_ordinals": compact_ordinal_ranges(
                        bar_end["missing"]
                    ),
                    "bar_end_candidate_duplicate_minute_count": bar_end["duplicate_count"],
                    "bar_end_candidate_off_grid_minute_count": bar_end["off_grid_count"],
                    "source_status": status,
                }
            )
        source_checks[symbol] = {
            "source_file_present": exists,
            "source_read_error_code": read_error_code,
            "bounded_safe_rows_materialised": len(timestamps),
            "bounded_safe_timestamp_sha256": timestamp_hash(timestamps),
            "complete_regular_sessions": complete_sessions,
            "bar_start_candidate_complete_regular_sessions": bar_start_complete_sessions,
            "bar_end_candidate_complete_regular_sessions": bar_end_complete_sessions,
            "unmatched_safe_timestamp_count": len(timestamps) - matched_rows,
        }
    expected_frame = pd.DataFrame(expected_rows).sort_values(
        ["symbol", "session"], kind="mergesort", ignore_index=True
    )
    text_columns = (
        "symbol",
        "session",
        "ordinal_basis",
        "bar_start_candidate_observed_minute_ordinals",
        "bar_start_candidate_missing_minute_ordinals",
        "bar_end_candidate_observed_minute_ordinals",
        "bar_end_candidate_missing_minute_ordinals",
        "source_status",
    )
    integer_columns = (
        "expected_minute_count",
        "bar_start_candidate_observed_minute_count",
        "bar_start_candidate_missing_minute_count",
        "bar_start_candidate_duplicate_minute_count",
        "bar_start_candidate_off_grid_minute_count",
        "bar_end_candidate_observed_minute_count",
        "bar_end_candidate_missing_minute_count",
        "bar_end_candidate_duplicate_minute_count",
        "bar_end_candidate_off_grid_minute_count",
    )
    for column in text_columns:
        if not coverage[column].astype(str).eq(expected_frame[column].astype(str)).all():
            raise AssertionError(f"availability text column differs: {column}")
    for column in integer_columns:
        if not coverage[column].astype(int).eq(expected_frame[column].astype(int)).all():
            raise AssertionError(f"availability count column differs: {column}")
    manifest = read_json(artifacts / "source_manifest.json")
    assert_safety(manifest)
    if manifest["external_api_called"] or manifest["credentials_read"]:
        raise AssertionError("external access declaration differs")
    for source in manifest["sources"]:
        observed = {
            key: source[key]
            for key in (
                "source_file_present",
                "source_read_error_code",
                "bounded_safe_rows_materialised",
                "bounded_safe_timestamp_sha256",
                "complete_regular_sessions",
                "bar_start_candidate_complete_regular_sessions",
                "bar_end_candidate_complete_regular_sessions",
                "unmatched_safe_timestamp_count",
            )
        }
        if observed != source_checks[source["symbol"]]:
            raise AssertionError(f"source manifest coverage differs: {source['symbol']}")
    safe_rows = len(safe_timestamp_values)
    sources_present = sum(check["source_file_present"] for check in source_checks.values())
    complete = int(expected_frame["source_status"].str.startswith("complete_under_").sum())
    bar_start_complete = int(
        expected_frame["source_status"]
        .isin(
            {
                "complete_under_bar_start_candidate",
                "complete_under_both_candidates",
            }
        )
        .sum()
    )
    bar_end_complete = int(
        expected_frame["source_status"]
        .isin(
            {
                "complete_under_bar_end_candidate",
                "complete_under_both_candidates",
            }
        )
        .sum()
    )
    if manifest["one_minute_rows_materialised"] != safe_rows:
        raise AssertionError("source manifest safe row count differs")
    if manifest["sources_present"] != sources_present:
        raise AssertionError("source manifest present-source count differs")
    if manifest["complete_symbol_sessions"] != complete:
        raise AssertionError("source manifest complete-session count differs")
    if manifest["bar_start_candidate_complete_symbol_sessions"] != bar_start_complete:
        raise AssertionError("bar-start candidate complete-session count differs")
    if manifest["bar_end_candidate_complete_symbol_sessions"] != bar_end_complete:
        raise AssertionError("bar-end candidate complete-session count differs")
    return {
        "rows_verified": len(coverage),
        "symbols_verified": int(coverage["symbol"].nunique()),
        "sessions_verified": int(coverage["session"].nunique()),
        "regular_sessions_390_minutes": int(expected_frame["expected_minute_count"].eq(390).sum()),
        "early_close_sessions_210_minutes": int(
            expected_frame["expected_minute_count"].eq(210).sum()
        ),
        "one_minute_rows_materialised": safe_rows,
        "source_files_present": sources_present,
        "complete_symbol_sessions": complete,
        "bar_start_candidate_complete_symbol_sessions": bar_start_complete,
        "bar_end_candidate_complete_symbol_sessions": bar_end_complete,
    }


def audit_frozen_population(artifacts: Path) -> dict[str, Any]:
    """Independently compare the blocker panel with frozen V0.1 admissions."""

    source = pd.read_parquet(PREDECESSOR_PANEL)
    expected = source.loc[source["high_movement_admitted"].astype(bool)].copy()
    observed = pd.read_parquet(artifacts / "compact_decision_panel.parquet")
    keys = ["symbol", "session", "decision_ordinal"]
    expected = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    observed = observed.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(expected) != 2_799 or len(observed) != len(expected):
        raise AssertionError("frozen admitted row count differs")
    exact_columns = [
        "symbol",
        "session",
        "year",
        "year_month",
        "decision_ordinal",
        "decision_time_america_new_york",
        "p_large_remaining_move",
        "movement_admission_threshold",
        "high_movement_admitted",
        "parent_slate_id",
        "parent_slate_eligible",
        "parent_valid_stock_count",
        "admitted_stock_count",
        "support_status",
        "primary_eligible",
        "row_weight",
    ]
    for column in exact_columns:
        left = expected[column].reset_index(drop=True)
        right = observed[column].reset_index(drop=True)
        if pd.api.types.is_float_dtype(left.dtype):
            if not left.astype(float).eq(right.astype(float)).all():
                raise AssertionError(f"frozen float column differs: {column}")
        elif not left.astype(str).eq(right.astype(str)).all():
            raise AssertionError(f"frozen identity column differs: {column}")
    expected_timestamp = pd.to_datetime(expected["feature_available_timestamp_utc"], utc=True)
    observed_timestamp = pd.to_datetime(observed["decision_timestamp_utc"], utc=True)
    if not expected_timestamp.eq(observed_timestamp).all():
        raise AssertionError("decision timestamp differs")
    assessment = observed.loc[observed["year"].eq(2025)]
    if (len(assessment), assessment["session"].nunique(), assessment["symbol"].nunique()) != (
        1_560,
        153,
        20,
    ):
        raise AssertionError("assessment population differs")
    reconstruction = read_json(artifacts / "frozen_population_reconstruction.json")
    assert_safety(reconstruction)
    if reconstruction["source_panel_sha256"] != sha256_file(PREDECESSOR_PANEL):
        raise AssertionError("predecessor panel hash differs")
    return {
        "rows_verified": len(observed),
        "development_rows": int(observed["year"].eq(2024).sum()),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "checkpoint_rows": {
            str(key): int(value)
            for key, value in assessment.groupby("decision_ordinal", sort=True).size().items()
        },
        "maximum_decision_timestamp": str(observed_timestamp.max()),
    }


def audit_inputs(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    """Verify every immutable input hash and logical path."""

    manifest = read_json(artifacts / "input_artifact_hashes.json")
    assert_safety(manifest)
    verified = 0
    for record in manifest["artifacts"]:
        logical = str(record["logical_path"])
        candidate = (REPO_ROOT / logical).resolve()
        if REPO_ROOT.resolve() not in candidate.parents and candidate != REPO_ROOT.resolve():
            raise AssertionError("input logical path escaped repository")
        if sha256_file(candidate) != record["sha256"]:
            raise AssertionError(f"input hash differs: {logical}")
        verified += 1
    safe_source_hashes = manifest["one_minute_safe_timestamp_hashes"]
    for record in safe_source_hashes:
        logical = str(record["logical_path"])
        symbol_parts = [part for part in logical.split("/") if part.startswith("symbol=")]
        if len(symbol_parts) != 1:
            raise AssertionError("one-minute source logical path is malformed")
        symbol = symbol_parts[0].split("=", maxsplit=1)[1]
        timestamps = read_safe_timestamps(provider_path(provider_root, symbol))
        if len(timestamps) != record["bounded_safe_rows_materialised"]:
            raise AssertionError(f"one-minute safe row count differs: {symbol}")
        if timestamp_hash(timestamps) != record["bounded_safe_timestamp_sha256"]:
            raise AssertionError(f"one-minute safe timestamp hash differs: {symbol}")
    if manifest["one_minute_source_artifacts_hashed"] != len(safe_source_hashes):
        raise AssertionError("one-minute source hash count differs")
    return {
        "hashes_verified": verified,
        "one_minute_source_hashes": len(safe_source_hashes),
    }


def audit_protected_boundary(artifacts: Path) -> dict[str, Any]:
    """Verify that safe timestamp reads never materialised a protected row."""

    boundary = read_json(artifacts / "protected_boundary_audit.json")
    assert_safety(boundary)
    if not boundary["passed"]:
        raise AssertionError("protected boundary did not pass")
    if boundary["protected_rows_opened"] != 0:
        raise AssertionError("protected rows were opened")
    if boundary["protected_files_touched"]:
        raise AssertionError("protected files were touched")
    maximum = boundary["maximum_one_minute_timestamp_read"]
    if maximum is not None and pd.Timestamp(maximum) >= PROTECTED_START:
        raise AssertionError("maximum one-minute timestamp crossed the protected boundary")
    if boundary["parquet_predicate_maximum_exclusive"] != PROTECTED_START.isoformat():
        raise AssertionError("protected Parquet predicate differs")
    return {
        "protected_start": boundary["protected_start"],
        "protected_rows_opened": 0,
        "one_minute_source_files_opened": boundary["one_minute_source_files_opened"],
        "safe_one_minute_rows_materialised": boundary["one_minute_rows_materialised"],
        "maximum_frozen_timestamp": boundary["frozen_predecessor_maximum_timestamp"],
    }


def audit_downstream_not_opened(artifacts: Path) -> dict[str, Any]:
    """Prove that the first gate prevented all downstream scientific work."""

    models = read_json(artifacts / "model_configurations.json")
    coefficients = read_json(artifacts / "model_coefficients.json")
    barriers = read_json(artifacts / "onset_barriers.json")
    normalisation = read_json(artifacts / "normalisation_manifest.json")
    feature_manifest = read_json(artifacts / "feature_manifest.json")
    forbidden = read_json(artifacts / "forbidden_feature_audit.json")
    for payload in (models, coefficients, barriers, normalisation, feature_manifest, forbidden):
        assert_safety(payload)
    if models["fitted_model_count"] != 0 or coefficients["fitted_model_count"] != 0:
        raise AssertionError("models were fitted")
    if barriers["barriers_bps"] != {"12": None, "6": None}:
        raise AssertionError("onset barriers were unexpectedly calculated")
    if normalisation["normalisation_rows_fitted"] != 0:
        raise AssertionError("activity normalisation was fitted")
    if not forbidden["passed"] or forbidden["violations"]:
        raise AssertionError("forbidden predictor name present")
    if not pd.read_parquet(artifacts / "one_minute_sequence_ledger.parquet").empty:
        raise AssertionError("one-minute feature ledger is not empty")
    if not pd.read_parquet(artifacts / "onset_path_ledger.parquet").empty:
        raise AssertionError("onset path ledger is not empty")
    if not pd.read_parquet(artifacts / "assessment_predictions.parquet").empty:
        raise AssertionError("assessment predictions are not empty")
    for filename in EMPTY_CSV_ARTIFACTS:
        if not pd.read_csv(artifacts / filename).empty:
            raise AssertionError(f"downstream CSV is not empty: {filename}")
    plots = sorted(path.name for path in artifacts.glob("*.png"))
    if plots:
        raise AssertionError(f"plots were unexpectedly created: {plots}")
    return {
        "models_fitted": 0,
        "normalisation_rows_fitted": 0,
        "onset_path_rows": 0,
        "assessment_prediction_rows": 0,
        "bootstrap_draws": 0,
        "null_draws": 0,
        "economic_diagnostics": 0,
        "plots": 0,
        "unperformed_formula_checks": "not_applicable_due_to_first_gate_history_blocker",
    }


def audit_timestamp_semantics(artifacts: Path) -> dict[str, Any]:
    """Verify the correct gate precedence without claiming unproved semantics."""

    semantics = read_json(artifacts / "timestamp_semantics_audit.json")
    assert_safety(semantics)
    if semantics["passed"] or semantics["bar_start_or_end_proved"]:
        raise AssertionError("timestamp semantics were falsely claimed as proved")
    if semantics["decision_precedence"] != EXPECTED_DECISION:
        raise AssertionError("timestamp gate precedence differs")
    return {
        "status": semantics["status"],
        "bar_start_or_end_proved": False,
        "causal_window_materialised": False,
        "decision_precedence": EXPECTED_DECISION,
    }


def audit_decision_and_rerun(artifacts: Path) -> dict[str, Any]:
    """Verify fail-closed decision logic and pre-audit rerun hashes."""

    decision = read_json(artifacts / "decision.json")
    assert_safety(decision)
    if decision["decision"] != EXPECTED_DECISION:
        raise AssertionError("decision differs")
    source_manifest = read_json(artifacts / "source_manifest.json")
    if source_manifest["availability_gate_passed"]:
        raise AssertionError("history blocker was emitted after the availability gate passed")
    if (
        decision["complete_symbol_sessions"] != source_manifest["complete_symbol_sessions"]
        or decision["one_minute_rows_materialised"]
        != source_manifest["one_minute_rows_materialised"]
        or decision["models_fitted"] != 0
    ):
        raise AssertionError("blocked decision support differs")
    rerun = read_json(artifacts / "exact_rerun_manifest.json")
    assert_safety(rerun)
    if not rerun["passed"] or not all(row["passed"] for row in rerun["comparisons"]):
        raise AssertionError("exact rerun comparison failed")
    for row in rerun["comparisons"]:
        artifact_path = artifacts / str(row["artifact"])
        current_hash = sha256_file(artifact_path)
        if current_hash not in {row["primary_sha256"], row["exact_rerun_sha256"]}:
            raise AssertionError(f"artifact hash no longer matches rerun: {row['artifact']}")
    return {
        "decision": decision["decision"],
        "decision_reconstructed": True,
        "pre_audit_exact_comparisons_verified": len(rerun["comparisons"]),
    }


def run_audit(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    """Run all independent blocker checks."""

    source_text = Path(__file__).read_text(encoding="utf-8")
    syntax = ast.parse(source_text)
    imported_runner = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name.endswith("run_screen_v0") for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and str(node.module).endswith("run_screen_v0"))
        for node in ast.walk(syntax)
    )
    if imported_runner:
        raise AssertionError("auditor imported experiment runner")
    contract = read_json(artifacts / "contract.json")
    assert_safety(contract)
    checks = {
        "availability": audit_availability(artifacts, provider_root),
        "frozen_population": audit_frozen_population(artifacts),
        "input_hashes": audit_inputs(artifacts, provider_root),
        "protected_boundary": audit_protected_boundary(artifacts),
        "timestamp_semantics": audit_timestamp_semantics(artifacts),
        "downstream_not_opened": audit_downstream_not_opened(artifacts),
        "decision_and_exact_rerun": audit_decision_and_rerun(artifacts),
    }
    return {
        **SAFETY_FLAGS,
        "passed": True,
        "decision": EXPECTED_DECISION,
        "auditor_imported_runner": imported_runner,
        "auditor_imported_reusable_module": False,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    """Parse independent-auditor arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Audit one primary or exact-rerun artifact directory."""

    args = parse_args()
    output = args.artifacts / "independent_audit.json"
    try:
        result = run_audit(args.artifacts, args.provider_root)
    except Exception as exc:
        result = {
            **SAFETY_FLAGS,
            "passed": False,
            "decision": "blocked_reproducibility_or_audit_failure",
            "auditor_imported_runner": False,
            "error": str(exc),
        }
        write_json(output, result)
        return 1
    write_json(output, result)
    print("passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
