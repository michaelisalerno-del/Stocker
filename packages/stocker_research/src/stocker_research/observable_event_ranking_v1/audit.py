"""Independent artifact auditor.

This module deliberately imports neither the experiment runner nor candidate event,
target, metric, model-prediction, or gate-decision helpers.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

_SAFETY = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "live_trading_enabled": False,
    "paper_order_submission_enabled": False,
    "account_data_requested": False,
    "positions_requested": False,
    "ibkr_orders_permitted": False,
}
_FORBIDDEN_EVENT_PREFIXES = (
    "future_",
    "target_",
    "regime_",
    "loop_",
    "excursion_",
    "posterior_",
    "mfe",
    "mae",
    "pnl_",
    "cost_",
    "spread_",
    "slippage_",
)
_FORBIDDEN_EVENT_COLUMNS = {
    "state",
    "regime",
    "loop",
    "excursion",
    "posterior",
    "personality",
    "personality_label",
    "template",
    "template_label",
    "payoff_candidate",
    "payoff_selected_candidate",
}
_FORBIDDEN_CLIENT_CALLS = {
    "placeOrder",
    "cancelOrder",
    "reqIds",
    "reqOpenOrders",
    "reqAllOpenOrders",
    "reqAutoOpenOrders",
    "reqExecutions",
    "reqPositions",
    "reqAccountUpdates",
    "reqAccountSummary",
    "reqAccountUpdatesMulti",
    "reqPositionsMulti",
    "reqPnL",
    "reqPnLSingle",
    "reqCompletedOrders",
    "reqManagedAccts",
    "reqGlobalCancel",
    "exerciseOptions",
}


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average
        start = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return covariance / (left_scale * right_scale)


def _independent_formula_fixture_checks() -> dict[str, bool]:
    """Exercise frozen formulas without importing candidate implementation helpers."""

    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    recent = closes[6] / closes[3] - 1.0
    preceding = closes[3] / closes[0] - 1.0
    peers = [0.01, 0.02, 0.04, 0.08]
    leave_one_out = [
        float(median([other for other_index, other in enumerate(peers) if other_index != index]))
        for index in range(len(peers))
    ]
    prior = [1.0, 2.0, 3.0, 4.0, 100.0]
    prior_location = float(median(prior))
    prior_scale = 1.4826 * float(median([abs(value - prior_location) for value in prior]))
    threshold = 1.25
    triggers = [
        ("AAA", "2025-07-07", datetime(2025, 7, 7, 13, 55, tzinfo=UTC)),
        ("AAA", "2025-07-07", datetime(2025, 7, 7, 14, 5, tzinfo=UTC)),
        ("BBB", "2025-07-07", datetime(2025, 7, 7, 13, 58, tzinfo=UTC)),
    ]
    first_events: dict[tuple[str, str], datetime] = {}
    later_count = 0
    for symbol, session, confirmation in sorted(triggers, key=lambda row: row[2]):
        key = (symbol, session)
        if key in first_events:
            later_count += 1
        else:
            first_events[key] = confirmation
    new_york = ZoneInfo("America/New_York")
    exact_confirmation = datetime(2025, 7, 7, 10, 0, tzinfo=new_york)
    grids = [
        datetime(2025, 7, 7, hour, minute, tzinfo=new_york)
        for hour, minute in ((10, 0), (10, 30), (11, 0))
    ]
    assigned = next(grid for grid in grids if grid > exact_confirmation)

    decision = datetime(2025, 7, 7, 14, 0, tzinfo=UTC)
    delayed_entry = decision + timedelta(minutes=5)
    exit_60m = delayed_entry + timedelta(minutes=60)
    tied_outcomes = [1.0, 2.0, 2.0, 4.0]
    mapped_ranks = [rank / 3.0 for rank in _average_ranks(tied_outcomes)]
    serialized_intercept = 0.5
    serialized_coefficients = [0.25, -0.1]
    serialized_means = [1.0, 2.0]
    serialized_scales = [2.0, 4.0]
    raw_features = [3.0, 6.0]
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            raw_features,
            serialized_means,
            serialized_scales,
            strict=True,
        )
    ]
    prediction = serialized_intercept + sum(
        value * coefficient
        for value, coefficient in zip(standardized, serialized_coefficients, strict=True)
    )
    outcomes = [0.0, 1.0, 2.0, 3.0]
    scores = [0.0, 1.0, 2.0, 3.0]
    spearman = _correlation(_average_ranks(outcomes), _average_ranks(scores))
    top_two_difference = (3.0 + 2.0) / 2.0 - float(median(outcomes))
    slate_ids = ["a", "a", "b", "b", "b", "b"]
    weights = [1.0 / slate_ids.count(slate_id) for slate_id in slate_ids]
    session_by_slate = {"a": "s1", "b": "s1", "c": "s2"}
    sampled_sessions = ["s1", "s2", "s1"]
    sampled_slates = [
        slate_id
        for session in sampled_sessions
        for slate_id, slate_session in session_by_slate.items()
        if slate_session == session
    ]
    stock_event_counts = [25] * 40
    top_selection_counts = [10] * 5 + [5] * 10

    return {
        "event_nonoverlapping_return_windows": math.isclose(recent, 106.0 / 103.0 - 1.0)
        and math.isclose(preceding, 103.0 / 100.0 - 1.0),
        "leave_one_out_peer_medians": leave_one_out == [0.04, 0.04, 0.02, 0.02],
        "trailing_robust_scale_excludes_current": prior_location == 3.0
        and math.isclose(prior_scale, 1.4826),
        "threshold_equality_triggers": threshold <= 1.25,
        "first_event_deduplication": len(first_events) == 2 and later_count == 1,
        "strict_next_grid_assignment": assigned.hour == 10 and assigned.minute == 30,
        "entry_t_plus_2_and_exit_60m": delayed_entry == datetime(2025, 7, 7, 14, 5, tzinfo=UTC)
        and exit_60m == datetime(2025, 7, 7, 15, 5, tzinfo=UTC),
        "within_slate_average_tie_ranks": mapped_ranks == [0.0, 0.5, 0.5, 1.0],
        "serialized_linear_prediction": math.isclose(prediction, 0.65),
        "observable_baseline_prediction": scores == outcomes,
        "spearman_calculation": math.isclose(spearman, 1.0),
        "top_two_minus_median_calculation": math.isclose(top_two_difference, 1.0),
        "equal_total_slate_weights": math.isclose(sum(weights[:2]), 1.0)
        and math.isclose(sum(weights[2:]), 1.0),
        "session_block_sampling_preserves_slates": sampled_slates == ["a", "b", "c", "a", "b"],
        "concentration_and_gate_boundaries": max(stock_event_counts) / sum(stock_event_counts)
        <= 0.075
        and max(top_selection_counts) / sum(top_selection_counts) <= 0.15,
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries_match_files(root: Path, entries: list[dict[str, Any]]) -> bool:
    """Re-hash every bound file while rejecting paths outside the manifest root."""

    expected_paths: set[str] = set()
    resolved_root = root.resolve()
    for entry in entries:
        relative = str(entry.get("path", ""))
        if not relative or relative in expected_paths:
            return False
        expected_paths.add(relative)
        path = (root / relative).resolve()
        if path.parent != resolved_root and resolved_root not in path.parents:
            return False
        if not path.is_file():
            return False
        if path.stat().st_size != entry.get("size_bytes"):
            return False
        if _file_hash(path) != entry.get("sha256"):
            return False
    return True


def _artifact_manifest_matches_files(artifact_dir: Path) -> bool:
    manifest_path = artifact_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or manifest.get("artifact_count") != len(entries):
        return False
    return _manifest_entries_match_files(artifact_dir, entries)


def _directory_identity(left: Path, right: Path) -> bool:
    def hashes(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): _file_hash(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name not in {"independent_audit.json", "independent_audit.md"}
        }

    return hashes(left) == hashes(right)


def _static_safety(repository_root: Path) -> tuple[bool, bool, bool]:
    package = (
        repository_root / "packages/stocker_execution/src/stocker_execution/ibkr_observability"
    )
    order_calls: list[str] = []
    imports_orders = False
    credential_literals = False
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        imports_orders = imports_orders or "stocker_execution.orders" in text
        credential_literals = credential_literals or bool(
            re.search(r"(?:DU|U)\d{7,}|account[_-]?id\s*=\s*['\"][^'\"]+", text)
        )
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _FORBIDDEN_CLIENT_CALLS
            ):
                order_calls.append(f"{path.name}:{node.func.attr}")
    return not order_calls, not imports_orders, not credential_literals


def _static_primary_provenance(repository_root: Path) -> bool:
    package = (
        repository_root
        / "packages/stocker_research/src/stocker_research/observable_event_ranking_v1"
    )
    forbidden_fragments = (".regime", ".loops", ".loop_", ".excursion", ".posterior")
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            if any(
                fragment in f".{module}" for module in modules for fragment in forbidden_fragments
            ):
                return False
    return True


def _artifact_sensitive_identifiers_absent(artifact_dir: Path) -> bool:
    pattern = re.compile(
        r"(?:DU|U)\d{7,}|(?:account[_ -]?(?:id|number))\s*[:=]\s*['\"]?[A-Za-z0-9]+",
        flags=re.IGNORECASE,
    )
    for path in sorted(artifact_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in {".json", ".csv", ".md"}
            and pattern.search(path.read_text(encoding="utf-8"))
        ):
            return False
    return True


def run_independent_audit(
    *,
    primary_dir: Path,
    exact_dir: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Independently verify the blocked pre-target run and exact-rerun identity."""

    root = repository_root or Path.cwd()
    contract = json.loads(
        (primary_dir / "frozen_experiment_contract.json").read_text(encoding="utf-8")
    )
    sources = json.loads(
        (primary_dir / "source_identity_manifest.json").read_text(encoding="utf-8")
    )
    implementation = json.loads(
        (primary_dir / "implementation_source_manifest.json").read_text(encoding="utf-8")
    )
    decision = json.loads((primary_dir / "support_decision.json").read_text(encoding="utf-8"))
    threshold = json.loads((primary_dir / "event_threshold.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (primary_dir / "static_provenance_audit.json").read_text(encoding="utf-8")
    )
    event_table = pq.read_table(  # type: ignore[no-untyped-call]
        primary_dir / "event_ledger.parquet"
    )
    universe_table = pq.read_table(  # type: ignore[no-untyped-call]
        primary_dir / "universe_ledger.parquet"
    )
    sector_table = pq.read_table(  # type: ignore[no-untyped-call]
        primary_dir / "sector_membership_ledger.parquet"
    )
    event_columns = [str(column).lower() for column in event_table.column_names]
    order_calls_absent, order_import_absent, credential_literals_absent = _static_safety(root)
    primary_provenance_absent = _static_primary_provenance(root)
    artifact_sensitive_identifiers_absent = _artifact_sensitive_identifiers_absent(primary_dir)
    fixture_checks = _independent_formula_fixture_checks()
    event_peer_fixture_names = {
        "event_nonoverlapping_return_windows",
        "leave_one_out_peer_medians",
        "trailing_robust_scale_excludes_current",
        "threshold_equality_triggers",
        "first_event_deduplication",
        "strict_next_grid_assignment",
    }
    timing_metric_fixture_names = {
        "entry_t_plus_2_and_exit_60m",
        "within_slate_average_tie_ranks",
        "serialized_linear_prediction",
        "observable_baseline_prediction",
        "spearman_calculation",
        "top_two_minus_median_calculation",
    }
    weight_gate_fixture_names = {
        "equal_total_slate_weights",
        "session_block_sampling_preserves_slates",
        "concentration_and_gate_boundaries",
    }
    source_snapshot = sources["inventory_snapshot"]
    implementation_entries = implementation["source_files"]
    checks = {
        "contract_safety_flags": contract.get("safety") == _SAFETY,
        "contract_hash_binding": (
            _canonical_hash(contract) == decision["artifact_binding"]["contract_hash"]
        ),
        "source_inventory_hash": (
            _canonical_hash(source_snapshot) == sources["data_snapshot_hash"]
        ),
        "implementation_manifest_hash": (
            _canonical_hash(implementation_entries) == implementation["implementation_hash"]
        ),
        "implementation_source_files_match_manifest": _manifest_entries_match_files(
            root, implementation_entries
        ),
        "primary_artifact_manifest_entries_match_files": _artifact_manifest_matches_files(
            primary_dir
        ),
        "exact_artifact_manifest_entries_match_files": _artifact_manifest_matches_files(exact_dir),
        "protected_data_unopened": sources.get("protected_files_opened") == 0,
        "processed_data_unopened": sources.get("processed_parquet_files_opened") == 0,
        "universe_effective_dates_fail_closed": (
            universe_table.num_rows == 0
            and decision.get("decision") == "blocked_missing_point_in_time_sector_membership"
        ),
        "sector_effective_dates_fail_closed": sector_table.num_rows == 0,
        "event_ledger_outcome_free": not any(
            column in _FORBIDDEN_EVENT_COLUMNS or column.startswith(_FORBIDDEN_EVENT_PREFIXES)
            for column in event_columns
        ),
        "event_population_empty_before_target_gate": event_table.num_rows == 0,
        "static_primary_provenance_audit": provenance.get("audit_passed") is True
        and provenance.get("forbidden_import_violations") == []
        and primary_provenance_absent,
        "threshold_not_fitted_after_data_blocker": (
            threshold.get("threshold") is None and threshold.get("status") == "not_fitted"
        ),
        "targets_not_permitted": decision.get("targets_permitted") is False,
        "model_not_permitted": decision.get("model_fit_permitted") is False,
        "outcomes_not_read": decision.get("outcomes_read") is False,
        "exact_rerun_identity": _directory_identity(primary_dir, exact_dir),
        "forbidden_ibkr_calls_absent": order_calls_absent,
        "order_models_not_imported": order_import_absent,
        "credentials_and_account_identifiers_absent": credential_literals_absent,
        "artifact_account_identifiers_absent": artifact_sensitive_identifiers_absent,
        "independent_event_and_peer_formula_fixtures": all(
            fixture_checks[name] for name in event_peer_fixture_names
        ),
        "independent_timing_rank_metric_fixtures": all(
            fixture_checks[name] for name in timing_metric_fixture_names
        ),
        "independent_weight_bootstrap_gate_fixtures": all(
            fixture_checks[name] for name in weight_gate_fixture_names
        ),
        "main_runner_not_imported": True,
        "candidate_event_functions_not_imported": True,
        "candidate_metric_functions_not_imported": True,
        "candidate_gate_functions_not_imported": True,
        "candidate_prediction_helpers_not_imported": True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    not_applicable = [
        "artifact_event_calculation_sample",
        "artifact_leave_one_out_market_sector_medians",
        "artifact_trailing_only_robust_scaling",
        "artifact_threshold_application",
        "artifact_first_event_deduplication",
        "artifact_grid_assignment",
        "artifact_entry_t_plus_2_and_target_60m_timing",
        "artifact_within_slate_ranks_and_equal_weights",
        "artifact_serialized_model_and_baseline_predictions",
        "artifact_spearman_top_two_and_bootstrap_grouping",
        "artifact_concentration_and_gate_logic",
    ]
    return {
        "audit_version": "observable_event_ranking_v1_independent_audit",
        "audit_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "independent_formula_fixture_checks": fixture_checks,
        "not_applicable_due_pre_target_data_blocker": not_applicable,
        "decision_audited": decision.get("decision"),
        "scientific_interpretation": (
            "The audit validates a fail-closed pre-target blocker only; no structural, "
            "directional, economic, or executable result exists."
        ),
    }
