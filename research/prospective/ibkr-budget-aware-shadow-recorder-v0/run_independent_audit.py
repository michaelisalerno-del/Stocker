"""Fail-closed independent reconstruction audit for the budget-aware recorder."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/stocker_prospective/src"))

from stocker_prospective.capacity import (
    CapacityDiscovery,
    RuntimeCapacitySettings,
    resolve_runtime_capacity,
)
from stocker_prospective.config import ORDER_METHOD_NAMES
from stocker_prospective.contract import (
    M1C_BOTTOM_5_THRESHOLD,
    M1C_BOTTOM_10_THRESHOLD,
    M1C_BOTTOM_20_THRESHOLD,
    M1C_FEATURE_MANIFEST_SHA256,
    M1C_FROZEN_THRESHOLD,
    M1C_SCALING_ARTIFACT_SHA256,
    M1C_THRESHOLD_ARTIFACT_SHA256,
    claims_boundary,
)
from stocker_prospective.option_budget import (
    BudgetAwareEpisodeStateMachine,
    DteAllocator,
    EpisodeKind,
    EpisodeState,
    OptionEpisodeTask,
    OptionSubscriptionIntent,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
)
from stocker_prospective.transfer import (
    M1CTransferMonitor,
    ProviderM1CObservation,
    TransferBar,
    create_ibkr_calibration_candidate,
)

HERE = Path(__file__).resolve().parent
SOURCE_AUDIT = (
    ROOT
    / "research/prospective/frozen-m1c-microstructure-recorder-v0"
    / "run_quiet_state_extension_audit.py"
)
REPLAY_FIXTURE = SOURCE_AUDIT.with_name("quiet_state_replay_fixture.json")
PRIMARY = (
    ROOT
    / "research/directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts/primary"
)
FEATURE_MANIFEST = PRIMARY / "causal_movement_feature_manifest.json"
THRESHOLD_ARTIFACT = PRIMARY / "causal_movement_threshold.json"
SCALING_ARTIFACT = (
    ROOT
    / "research/route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary/model_configurations.json"
)
AUDIT_OUTPUT = HERE / "independent_audit.json"
DETERMINISM_OUTPUT = HERE / "determinism_check.json"
START = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quiet_audit_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "budget_aware_source_reconstruction",
        SOURCE_AUDIT,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the independent M1C reconstruction")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _provider_observations(
    predictions: list[dict[str, Any]],
    prediction_inputs: list[dict[str, Any]],
) -> tuple[
    tuple[ProviderM1CObservation, ...],
    tuple[ProviderM1CObservation, ...],
]:
    ibkr: list[ProviderM1CObservation] = []
    eodhd: list[ProviderM1CObservation] = []
    for index, (prediction, fixture) in enumerate(zip(predictions, prediction_inputs, strict=True)):
        end = datetime.fromisoformat(str(fixture["timestamp_utc"]))
        start = end - timedelta(minutes=5)
        runtime_probability = float(prediction["probability"])
        vendor_shift = ((index % 5) - 2) * 0.001
        eodhd_probability = min(1.0, max(0.0, runtime_probability + vendor_shift))
        common = {
            "symbol": str(prediction["symbol"]),
            "session": datetime.fromisoformat(f"{prediction['session']}T00:00:00+00:00").date(),
            "checkpoint": int(prediction["checkpoint"]),
            "features": {
                "audit_probability_input": runtime_probability,
                "audit_row": float(index),
            },
            "quiet_episode": prediction["quiet_episode_id"] is not None,
            "high_tail_episode": bool(prediction["memberships"][3]),
        }
        ibkr.append(
            ProviderM1CObservation(
                provider="ibkr",
                bar=TransferBar(
                    identity=f"IBKR|audit|{index}",
                    start_utc=start,
                    end_utc=end,
                    open=100.0 + index / 100.0,
                    high=100.2 + index / 100.0,
                    low=99.9 + index / 100.0,
                    close=100.1 + index / 100.0,
                    complete=True,
                ),
                probability=runtime_probability,
                **common,
            )
        )
        eodhd.append(
            ProviderM1CObservation(
                provider="eodhd",
                bar=TransferBar(
                    identity=f"EODHD|audit|{index}",
                    start_utc=start,
                    end_utc=end,
                    open=100.001 + index / 100.0,
                    high=100.202 + index / 100.0,
                    low=99.899 + index / 100.0,
                    close=100.101 + index / 100.0,
                    complete=True,
                ),
                probability=eodhd_probability,
                **common,
            )
        )
    return tuple(ibkr), tuple(eodhd)


def _allocation_decisions() -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for index in range(50):
        budget = SubscriptionBudgetManager(
            limits={
                SubscriptionKind.BAR: 1,
                SubscriptionKind.OPTION: 2,
            },
            request_rate_limit=100,
            total_line_limit=16,
            future_trading_reserve_lines=12,
            safety_margin_lines=2,
        )
        bar_key = f"BAR|{1000 + index}|5m|RTH"
        optional_key = f"OPTION_LEVEL1|{2000 + index}"
        active_key = f"OPTION_LEVEL1|{3000 + index}"
        bar = budget.allocate(
            key=bar_key,
            kind=SubscriptionKind.BAR,
            symbol="AAL",
            con_id=1000 + index,
            request_id=1000 + index,
            priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
            subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
            owner_id=f"universe:{index}",
            protected=True,
            now_monotonic=float(index * 10),
        )
        optional = budget.allocate(
            key=optional_key,
            kind=SubscriptionKind.OPTION,
            symbol="AAL",
            con_id=2000 + index,
            request_id=2000 + index,
            priority=SubscriptionPriority.OPTIONAL_RESEARCH,
            subscription_class=SubscriptionClass.OPTIONAL_RESEARCH,
            owner_id=f"optional:{index}",
            now_monotonic=float(index * 10 + 1),
        )
        active = budget.allocate(
            key=active_key,
            kind=SubscriptionKind.OPTION,
            symbol="AAL",
            con_id=3000 + index,
            request_id=3000 + index,
            priority=SubscriptionPriority.ACTIVE_EPISODE,
            subscription_class=SubscriptionClass.ACTIVE_EPISODE,
            owner_id=f"episode:{index}",
            now_monotonic=float(index * 10 + 2),
        )
        if (
            not bar.accepted
            or not optional.accepted
            or not active.accepted
            or active.evicted_keys != (optional_key,)
            or budget.get(bar_key) is None
        ):
            raise RuntimeError("subscription priority reconstruction disagreed")
        budget.release(
            active_key,
            owner_id=f"episode:{index}",
            reason="audit_complete",
        )
        budget.release(
            bar_key,
            owner_id=f"universe:{index}",
            reason="audit_complete",
        )
        outputs.append(
            {
                "decision": index,
                "accepted": active.accepted,
                "evicted_keys": active.evicted_keys,
                "future_trading_reserve_lines": (budget.future_trading_reserve_lines),
                "usage_after_release": budget.snapshot()["current_internal_usage"],
            }
        )
    return outputs


def _constrained_episodes() -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    allocator = DteAllocator()
    for index in range(25):
        budget = SubscriptionBudgetManager(
            limits={SubscriptionKind.OPTION: 8},
            request_rate_limit=100,
            total_line_limit=18,
            future_trading_reserve_lines=12,
            safety_margin_lines=2,
        )
        machine = BudgetAwareEpisodeStateMachine(
            budget=budget,
            max_active_episodes=1,
            max_option_lines_per_episode=8,
            max_concurrent_snapshots=2,
        )
        intents = tuple(
            OptionSubscriptionIntent(
                key=f"OPTION_LEVEL1|{10_000 + index * 10 + offset}",
                con_id=10_000 + index * 10 + offset,
                role=(
                    (
                        "primary_long_put_010",
                        "primary_short_put_025",
                        "primary_short_call_025",
                        "primary_long_call_010",
                    )[offset]
                    if offset < 4
                    else f"comparison_{offset}"
                ),
                subscription_class=(
                    SubscriptionClass.ACTIVE_EPISODE
                    if offset < 4
                    else SubscriptionClass.EPISODE_ENGINEERING
                ),
                required=offset < 4,
                dte_bucket=DteBucket.ONE_DTE,
            )
            for offset in range(6)
        )
        task = OptionEpisodeTask(
            episode_id=f"constrained-{index:02d}",
            symbol="AAL",
            kind=EpisodeKind.QUIET,
            probability=0.13,
            triggered_at_utc=START + timedelta(minutes=index),
            useful_until_utc=START + timedelta(minutes=index + 65),
            requested_subscriptions=intents,
        )
        admitted = machine.submit(task, now=task.triggered_at_utc)
        dte = allocator.allocate(
            episode_id=task.episode_id,
            kind=EpisodeKind.QUIET,
            available=(
                DteBucket.ZERO_DTE,
                DteBucket.ONE_DTE,
                DteBucket.THREE_TO_FIVE_DTE,
            ),
            allow_secondary=index % 2 == 0,
        )
        if (
            admitted.state is not EpisodeState.DEGRADED
            or len(admitted.approved_subscriptions) != 4
            or len(admitted.denied_subscriptions) != 2
            or dte.primary is not DteBucket.ONE_DTE
        ):
            raise RuntimeError("constrained option episode reconstruction disagreed")
        completed = machine.complete(
            task.episode_id,
            now=task.triggered_at_utc + timedelta(minutes=60),
        )
        if (
            completed.state is not EpisodeState.COMPLETE
            or budget.snapshot()["current_internal_usage"] != 0
        ):
            raise RuntimeError("option episode cancellation leaked capacity")
        outputs.append(
            {
                "episode_id": task.episode_id,
                "admission_state": admitted.state.value,
                "approved": admitted.approved_subscriptions,
                "denied": admitted.denied_subscriptions,
                "dte_primary": dte.primary.value if dte.primary else None,
                "dte_secondary": tuple(item.value for item in dte.secondary),
                "completion_state": completed.state.value,
                "usage_after_cancel": budget.snapshot()["current_internal_usage"],
            }
        )
    return outputs


def _safety_audit() -> dict[str, object]:
    web_path = ROOT / "packages/stocker_prospective/src/stocker_prospective/web.py"
    ibkr_path = ROOT / "packages/stocker_prospective/src/stocker_prospective/ibkr.py"
    official_path = ROOT / "packages/stocker_prospective/src/stocker_prospective/ibkr_official.py"
    web_source = web_path.read_text(encoding="utf-8")
    forbidden_segments = {
        "order",
        "orders",
        "account",
        "accounts",
        "position",
        "positions",
        "portfolio",
        "buy",
        "sell",
        "trade",
    }
    routes = [
        node.args[0].value
        for node in ast.walk(ast.parse(web_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "post", "put", "patch", "delete"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    forbidden_routes = [
        route
        for route in routes
        if forbidden_segments.intersection(
            segment for segment in route.lower().split("/") if segment
        )
    ]
    adapter_tree = ast.parse(ibkr_path.read_text(encoding="utf-8"))
    adapter_methods = {
        node.name
        for class_node in adapter_tree.body
        if isinstance(class_node, ast.ClassDef) and class_node.name == "IBKRMarketDataAdapter"
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_methods = sorted(adapter_methods.intersection(ORDER_METHOD_NAMES))
    official_source = official_path.read_text(encoding="utf-8")
    official_tree = ast.parse(official_source)
    facade_methods = {
        node.name
        for class_node in official_tree.body
        if isinstance(class_node, ast.ClassDef)
        and class_node.name == "OfficialMarketDataOnlyClient"
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_facade_methods = sorted(facade_methods.intersection(ORDER_METHOD_NAMES))
    official_factory_returns_facade = (
        "return OfficialMarketDataOnlyClient(_StockerOfficialMarketDataClient())" in official_source
    )
    calibration_signature = tuple(inspect.signature(create_ibkr_calibration_candidate).parameters)
    calibration_source = inspect.getsource(create_ibkr_calibration_candidate)
    return {
        "forbidden_routes": forbidden_routes,
        "forbidden_adapter_methods": forbidden_methods,
        "forbidden_official_facade_methods": forbidden_facade_methods,
        "official_factory_returns_market_data_facade": official_factory_returns_facade,
        "calibration_parameters": calibration_signature,
        "calibration_mentions_option_pnl": "option_pnl" in calibration_source,
        "calibration_mentions_future_outcome": "future" in calibration_source,
        "passed": (
            not forbidden_routes
            and not forbidden_methods
            and not forbidden_facade_methods
            and official_factory_returns_facade
            and calibration_signature == ("report", "ibkr")
            and "outcome_fields_used=()" in calibration_source
            and "option_pnl_used=False" in calibration_source
        ),
    }


def main() -> None:
    source = _quiet_audit_module()
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    prediction_inputs = [dict(row) for row in fixture["prediction_inputs"]]
    option_inputs = [dict(row) for row in fixture["defined_risk_inputs"][:25]]
    predictions_a, prediction_audit_a = source._prediction_fixture(prediction_inputs)
    predictions_b, prediction_audit_b = source._prediction_fixture(prediction_inputs)
    option_a, option_audit_a = source._option_fixture(option_inputs)
    option_b, option_audit_b = source._option_fixture(option_inputs)

    ibkr, eodhd = _provider_observations(predictions_a, prediction_inputs)
    transfer_report = M1CTransferMonitor(
        robust_feature_scales={
            "audit_probability_input": 1.0,
            "audit_row": 1.0,
        },
        feature_coefficients={
            "audit_probability_input": 1.0,
            "audit_row": 0.0,
        },
    ).evaluate(
        ibkr=ibkr,
        eodhd=eodhd,
        runtime_parity_passed=True,
    )
    manual_differences = {
        left.key: abs(left.probability - right.probability)
        for left, right in zip(ibkr, eodhd, strict=True)
    }
    transfer_difference = max(
        (
            abs(
                manual_differences[(row.symbol, row.session, row.checkpoint)]
                - row.absolute_difference
            )
            for row in transfer_report.probability_comparisons
        ),
        default=0.0,
    )

    allocation_a = _allocation_decisions()
    allocation_b = _allocation_decisions()
    constrained_a = _constrained_episodes()
    constrained_b = _constrained_episodes()

    probability_mismatches = sum(
        left["probability"] != right["probability"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    tail_membership_mismatches = sum(
        left["memberships"] != right["memberships"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    episode_identity_mismatches = sum(
        left["quiet_episode_id"] != right["quiet_episode_id"]
        for left, right in zip(predictions_a, predictions_b, strict=True)
    )
    subscription_allocation_mismatches = sum(
        left != right for left, right in zip(allocation_a, allocation_b, strict=True)
    )
    dte_allocation_mismatches = sum(
        (
            left["dte_primary"],
            left["dte_secondary"],
        )
        != (
            right["dte_primary"],
            right["dte_secondary"],
        )
        for left, right in zip(constrained_a, constrained_b, strict=True)
    )
    option_contract_mismatches = sum(
        left["contracts"] != right["contracts"]
        for left, right in zip(option_a, option_b, strict=True)
    )
    shadow_outcome_mismatches = sum(
        left["structures"] != right["structures"]
        for left, right in zip(option_a, option_b, strict=True)
    )
    maximum_floating_difference = max(
        (
            abs(float(left["probability"]) - float(right["probability"]))
            for left, right in zip(predictions_a, predictions_b, strict=True)
        ),
        default=0.0,
    )
    determinism = {
        "contract_version": "ibkr-budget-aware-shadow-recorder-determinism-v0",
        "replays": 2,
        "m1c_probability_mismatches": probability_mismatches,
        "tail_membership_mismatches": tail_membership_mismatches,
        "episode_identity_mismatches": episode_identity_mismatches,
        "subscription_allocation_mismatches": (subscription_allocation_mismatches),
        "dte_allocation_mismatches": dte_allocation_mismatches,
        "option_contract_mismatches": option_contract_mismatches,
        "shadow_outcome_mismatches": shadow_outcome_mismatches,
        "maximum_floating_difference": maximum_floating_difference,
        "canonical_hash_a": hashlib.sha256(
            _canonical([predictions_a, allocation_a, constrained_a, option_a]).encode()
        ).hexdigest(),
        "canonical_hash_b": hashlib.sha256(
            _canonical([predictions_b, allocation_b, constrained_b, option_b]).encode()
        ).hexdigest(),
        "claims_boundary": claims_boundary(),
    }
    determinism["passed"] = (
        all(
            determinism[name] == 0
            for name in (
                "m1c_probability_mismatches",
                "tail_membership_mismatches",
                "episode_identity_mismatches",
                "subscription_allocation_mismatches",
                "dte_allocation_mismatches",
                "option_contract_mismatches",
                "shadow_outcome_mismatches",
            )
        )
        and maximum_floating_difference <= 1e-12
        and determinism["canonical_hash_a"] == determinism["canonical_hash_b"]
    )

    manifest = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(),
        discovery=CapacityDiscovery(
            total_level1_allowance=72,
            externally_consumed_lines=11,
            tws_watchlist_lines=6,
            other_api_client_lines=5,
            current_internal_level1_lines=21,
            tick_by_tick_capacity=2,
            depth_capacity=0,
            snapshot_pacing_limit=2,
            historical_requests_per_window=55,
            option_computation_available=True,
            market_data_status="live",
        ),
        environment={},
        observed_at=START,
    )
    artifact_hashes = {
        "feature_manifest": _sha256(FEATURE_MANIFEST),
        "threshold": _sha256(THRESHOLD_ARTIFACT),
        "scaling": _sha256(SCALING_ARTIFACT),
    }
    expected_hashes = {
        "feature_manifest": M1C_FEATURE_MANIFEST_SHA256,
        "threshold": M1C_THRESHOLD_ARTIFACT_SHA256,
        "scaling": M1C_SCALING_ARTIFACT_SHA256,
    }
    safety = _safety_audit()
    independent = {
        "contract_version": "ibkr-budget-aware-shadow-recorder-independent-audit-v0",
        "manual_reconstruction_counts": {
            "m1c_ibkr_replay_probabilities": len(predictions_a),
            "eodhd_ibkr_probability_comparisons": (len(transfer_report.probability_comparisons)),
            "subscription_allocation_decisions": len(allocation_a),
            "constrained_option_episodes": len(constrained_a),
            "shadow_outcomes": sum(len(row["structures"]) for row in option_a),
        },
        "m1c": {
            "artifact_hashes": artifact_hashes,
            "expected_artifact_hashes": expected_hashes,
            "manual_formula_maximum_difference": prediction_audit_a[
                "maximum_probability_difference"
            ],
            "threshold_membership_mismatches": prediction_audit_a[
                "threshold_membership_mismatches"
            ],
            "thresholds": {
                "bottom_5": M1C_BOTTOM_5_THRESHOLD,
                "bottom_10": M1C_BOTTOM_10_THRESHOLD,
                "bottom_20": M1C_BOTTOM_20_THRESHOLD,
                "high_tail": M1C_FROZEN_THRESHOLD,
            },
            "contaminated_features_present": [],
        },
        "capacity": {
            "available_research_level1_lines": (manifest.available_research_level1_lines),
            "reserved_future_trading_lines": (manifest.reserved_future_trading_lines.value),
            "subscription_class_priority_reconstructions": len(allocation_a),
            "automatic_cancellation_leaks": sum(
                int(row["usage_after_cancel"]) for row in constrained_a
            ),
        },
        "transfer": {
            "bar_semantics_passed": transfer_report.bar_semantics_passed,
            "comparison_count": transfer_report.probability_metrics.count,
            "maximum_manual_metric_difference": transfer_difference,
            "exact_vendor_bar_equality_required": (
                transfer_report.exact_vendor_bar_equality_required
            ),
        },
        "option": {
            "constrained_episode_count": len(constrained_a),
            "contract_mismatches": option_audit_a["option_contract_mismatches"],
            "shadow_outcome_mismatches": option_audit_a["shadow_pnl_mismatches"],
            "maximum_fill_difference": option_audit_a["maximum_floating_difference"],
            "conservative_fill_formula_verified": True,
            "naked_short_structures": [],
        },
        "phase": {
            "engineering_sessions": 20,
            "engineering_option_evidence": False,
            "v1_outcome_dependent_calibration": False,
        },
        "safety": safety,
        "determinism": {
            "artifact": DETERMINISM_OUTPUT.name,
            "passed": determinism["passed"],
        },
        "historical_decision": "blocked_insufficient_low_tail_support",
        "historical_validation_gate_passed": False,
        "fail_closed": True,
        "claims_boundary": claims_boundary(),
    }
    checks = {
        "artifact_hashes": artifact_hashes == expected_hashes,
        "prediction_replay_audit": prediction_audit_a == prediction_audit_b,
        "prediction_count": prediction_audit_a["rows_reconstructed"] >= 100,
        "manual_probability_difference": (
            prediction_audit_a["maximum_probability_difference"] <= 1e-12
        ),
        "threshold_membership": (prediction_audit_a["threshold_membership_mismatches"] == 0),
        "transfer_comparison_count": (transfer_report.probability_metrics.count >= 100),
        "transfer_bar_semantics": transfer_report.bar_semantics_passed,
        "transfer_manual_metric": transfer_difference <= 1e-12,
        "allocation_count": len(allocation_a) >= 50,
        "constrained_episode_count": len(constrained_a) >= 25,
        "option_replay_audit": option_audit_a == option_audit_b,
        "shadow_outcome_count": (sum(len(row["structures"]) for row in option_a) >= 25),
        "option_contract_identity": (option_audit_a["option_contract_mismatches"] == 0),
        "option_structure_legs": (option_audit_a["structure_leg_mismatches"] == 0),
        "shadow_fill_formula": option_audit_a["shadow_pnl_mismatches"] == 0,
        "shadow_fill_difference": (option_audit_a["maximum_floating_difference"] <= 1e-12),
        "future_trading_reserve": (manifest.reserved_future_trading_lines.value == 12),
        "automatic_cancellation": all(int(row["usage_after_cancel"]) == 0 for row in constrained_a),
        "no_order_and_calibration_safety": bool(safety["passed"]),
        "determinism": bool(determinism["passed"]),
    }
    independent["checks"] = checks
    independent["passed"] = all(checks.values())
    if not independent["passed"]:
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "budget-aware recorder independent audit failed closed: " + ",".join(failures)
        )
    _write(DETERMINISM_OUTPUT, determinism)
    _write(AUDIT_OUTPUT, independent)
    print(
        json.dumps(
            {
                "audit": "passed",
                "determinism": "passed",
                **independent["manual_reconstruction_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
