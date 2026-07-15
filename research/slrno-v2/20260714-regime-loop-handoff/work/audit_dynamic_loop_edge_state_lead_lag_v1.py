#!/usr/bin/env python3
"""Independent audit for the Dynamic Loop Edge State Lead-Lag V1 run.

Critical calendar joins, paired losses, and the V2 policy shift are rebuilt
here without importing the experiment runner or its reusable calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260715-dynamic-loop-edge-state-lead-lag-v1.json"
V2_RUNNER = WORK / "run_dynamic_loop_edge_state_v2.py"
V2_MODULE_ROOT = REPO / "packages/stocker_research/src/stocker_research/dynamic_loop_edge_state"
DEFAULT_ROOT = WORK / "artifacts/20260715-dynamic-loop-edge-state-lead-lag-v1"
CONTROL = "hierarchical_payoff_history_change_point"
FULL = "hierarchical_change_point"
LEADS = (0, 1, 2, 3, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(results: list[dict[str, object]], name: str, condition: bool, detail: str) -> None:
    results.append({"name": name, "passed": bool(condition), "detail": detail})


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"missing {label} columns: {missing}")


def resolve_v2(contract: Mapping[str, Any], key: str) -> Path:
    return (CONTRACT_PATH.parent / str(contract["inputs"][key])).resolve()


def independently_rebuild_calendar_targets(
    forecasts: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    sessions = (
        features[["period", "score_session"]]
        .drop_duplicates()
        .sort_values(["period", "score_session"], kind="stable")
    )
    maps: list[pd.DataFrame] = []
    for lead in LEADS:
        shifted = sessions.copy()
        shifted["target_lead_sessions"] = lead
        shifted["audited_target_session"] = shifted.groupby("period", observed=True)[
            "score_session"
        ].shift(-lead)
        maps.append(shifted)
    mapping = pd.concat(maps, ignore_index=True)
    return forecasts[["forecast_id", "period", "score_session"]].merge(
        mapping,
        on=["period", "score_session"],
        how="left",
        validate="many_to_many",
    )


def independently_rebuild_paired_brier(joins: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "period",
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
        "target_lead_sessions",
        "target_session",
    ]
    columns = [
        *keys,
        "target_outcome_id",
        "target_payoff_available",
        "target_payoff_positive",
        "target_robust_net_bps",
        "p_next_payoff_positive",
        "edge_state",
    ]
    control = joins.loc[joins["model_name"].eq(CONTROL), columns]
    full = joins.loc[joins["model_name"].eq(FULL), columns]
    paired = control.merge(
        full,
        on=keys,
        suffixes=("_control", "_full"),
        how="inner",
        validate="one_to_one",
    )
    if (
        not paired["target_outcome_id_control"]
        .astype("string")
        .equals(paired["target_outcome_id_full"].astype("string"))
    ):
        raise AssertionError("independent paired outcome identity mismatch")
    paired = paired.loc[paired["target_payoff_available_control"].eq(True)].copy()  # noqa: E712
    y = paired["target_payoff_positive_control"].astype(float)
    paired["brier_improvement"] = (paired["p_next_payoff_positive_control"] - y) ** 2 - (
        paired["p_next_payoff_positive_full"] - y
    ) ** 2
    paired["economic_increment"] = paired["target_robust_net_bps_control"] * (
        paired["edge_state_full"].eq("active").astype(int)
        - paired["edge_state_control"].eq("active").astype(int)
    )
    return (
        paired.groupby("target_lead_sessions", observed=True, sort=True)
        .agg(
            observable_targets=("brier_improvement", "size"),
            brier_improvement=("brier_improvement", "mean"),
            economic_increment_bps=("economic_increment", "sum"),
        )
        .reset_index()
    )


def independently_reconstruct_delay(decisions: pd.DataFrame) -> pd.DataFrame:
    cell = ["period", "loop_id", "orientation", "horizon"]
    keys = [*cell, "score_session"]
    policy = decisions[[*keys, "accepted"]].drop_duplicates(keys).sort_values(keys, kind="stable")
    policy["audited_delayed"] = policy.groupby(cell, observed=True)["accepted"].shift(
        1, fill_value=False
    )
    rebuilt = decisions.merge(
        policy[[*keys, "audited_delayed"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    rebuilt["immediate"] = rebuilt["accepted"].eq(True)  # noqa: E712
    rebuilt["delayed"] = rebuilt["audited_delayed"].eq(True)  # noqa: E712
    rebuilt["category"] = np.select(
        [
            rebuilt["immediate"] & rebuilt["delayed"],
            rebuilt["immediate"] & ~rebuilt["delayed"],
            ~rebuilt["immediate"] & rebuilt["delayed"],
        ],
        ["retained", "dropped", "introduced"],
        default="rejected_both",
    )
    return rebuilt


def comparable_files(root: Path) -> dict[str, str]:
    excluded = {"artifact_manifest.json", "independent_audit.json", "research_report.md"}
    return {
        path.name: sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in excluded
    }


def audit(primary: Path, exact: Path, primary_report: Path, exact_report: Path) -> dict[str, Any]:
    results: list[dict[str, object]] = []
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = sha256(CONTRACT_PATH)
    check(
        results,
        "contract_identity",
        contract["contract_id"] == "20260715-dynamic-loop-edge-state-lead-lag-v1",
        contract_hash,
    )
    check(
        results,
        "lead_family_registration",
        tuple(contract["lead_family"]["registered_leads_sessions"]) == LEADS
        and contract["lead_family"]["primary_lead_sessions"] == 1,
        "registered=(0,1,2,3,5); primary=1",
    )
    check(
        results,
        "frozen_v2_runner_hash",
        sha256(V2_RUNNER) == contract["frozen_lineage"]["v2_runner_sha256"],
        sha256(V2_RUNNER),
    )
    module_ok = all(
        sha256(V2_MODULE_ROOT / name) == expected
        for name, expected in contract["frozen_lineage"]["v2_module_sha256"].items()
    )
    check(results, "frozen_v2_module_hashes", module_ok, "four reusable V2 modules")

    primary_files = comparable_files(primary)
    exact_files = comparable_files(exact)
    check(
        results,
        "primary_exact_machine_readable_and_plot_identity",
        primary_files == exact_files,
        f"primary={len(primary_files)} files; exact={len(exact_files)} files",
    )
    check(
        results,
        "primary_exact_report_identity",
        sha256(primary_report) == sha256(exact_report),
        sha256(primary_report),
    )

    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    check(
        results,
        "git_and_configuration_metadata",
        metadata["contract_hash"] == contract_hash
        and metadata["data_snapshot_hash"] == contract["frozen_lineage"]["v2_data_snapshot_sha256"],
        f"git={metadata['git_sha']}; contract={metadata['contract_hash']}",
    )
    safety = metadata["safety"]
    check(
        results,
        "research_only_safety_flags",
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["broker_connection_enabled"] is False
        and safety["deployment_enabled"] is False,
        "broker/order/deployment/execution disabled",
    )

    forecasts = pd.read_parquet(primary / "frozen_forecast_ledger.parquet")
    outcomes = pd.read_parquet(primary / "settled_outcome_ledger.parquet")
    joins = pd.read_parquet(primary / "forecast_to_target_lead_joins.parquet")
    paired = pd.read_parquet(primary / "full_vs_no_feature_paired_predictions.parquet")
    v2_primary = resolve_v2(contract, "v2_primary_root")
    v2_forecasts = pd.read_parquet(v2_primary / "causal_edge_state_forecasts.parquet")
    v2_features = pd.read_parquet(v2_primary / "causal_feature_panel.parquet")
    v2_panel = pd.read_parquet(v2_primary / "session_payoff_panel.parquet")
    v2_decisions = pd.read_parquet(v2_primary / "trade_decisions.parquet")

    check(results, "forecast_unique_ids", forecasts["forecast_id"].is_unique, str(len(forecasts)))
    check(results, "outcome_unique_ids", outcomes["outcome_id"].is_unique, str(len(outcomes)))
    freeze = pd.to_datetime(forecasts["forecast_freeze_timestamp"], utc=True)
    creation = pd.to_datetime(forecasts["forecast_creation_timestamp"], utc=True)
    feature_time = pd.to_datetime(
        forecasts["feature_max_availability_timestamp"], utc=True, errors="coerce"
    )
    training_time = pd.to_datetime(
        forecasts["training_latest_availability_timestamp"], utc=True, errors="coerce"
    )
    check(
        results,
        "forecast_freeze_and_feature_availability",
        freeze.equals(creation)
        and not (feature_time.notna() & feature_time.ge(freeze)).any()
        and not (training_time.notna() & training_time.ge(freeze)).any(),
        "features and settled training strictly precede freeze",
    )
    check(
        results,
        "forecast_not_revised_after_outcome",
        bool(forecasts["source_run_id"].eq(contract["frozen_lineage"]["v2_run_id"]).all()),
        "copied from immutable frozen V2 run",
    )

    rebuilt_calendar = independently_rebuild_calendar_targets(forecasts, v2_features)
    audited = joins[["forecast_id", "target_lead_sessions", "target_session"]].merge(
        rebuilt_calendar,
        on=["forecast_id", "target_lead_sessions"],
        validate="one_to_one",
    )
    calendar_equal = (
        audited["target_session"]
        .astype("string")
        .equals(audited["audited_target_session"].astype("string"))
    )
    check(
        results,
        "target_lead_calendar_joins",
        calendar_equal and set(joins["target_lead_sessions"].unique()) == set(LEADS),
        "explicit within-period trading sessions",
    )
    cross_period = joins.loc[joins["target_session"].notna(), ["period", "target_session"]].assign(
        target_year=lambda frame: frame["target_session"].astype(str).str[:4].astype(int)
    )
    check(
        results,
        "period_boundaries",
        bool(cross_period["period"].eq(cross_period["target_year"]).all()),
        "no 2023-to-2025 join",
    )
    missing_zero = joins.loc[~joins["target_payoff_available"]]
    check(
        results,
        "missing_targets_not_zero",
        bool(missing_zero["target_robust_net_bps"].isna().all()),
        f"missing={len(missing_zero)}",
    )

    target_keys = ["period", "session", "loop_id", "orientation", "horizon"]
    audited_targets = joins.loc[joins["target_payoff_available"]].merge(
        v2_panel[target_keys + ["source_data_id", "robust_net_payoff_bps"]].rename(
            columns={"session": "target_session"}
        ),
        on=["period", "target_session", "loop_id", "orientation", "horizon"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_source"),
    )
    check(
        results,
        "settled_target_identity",
        np.allclose(
            audited_targets["target_robust_net_bps"],
            audited_targets["robust_net_payoff_bps"],
            rtol=0.0,
            atol=1e-12,
        ),
        "joined directly to frozen V2 source_data_id",
    )

    paired_keys = [
        "period",
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
        "target_lead_sessions",
        "target_session",
    ]
    check(
        results,
        "full_no_feature_paired_population_identity",
        not paired.duplicated(paired_keys).any() and len(paired) == 41_800,
        f"paired={len(paired)}",
    )
    training_fields = contract["paired_primary_models"]["required_identical_training_fields"]
    source_pair = v2_forecasts.loc[
        v2_forecasts["model_name"].isin([CONTROL, FULL]),
        ["model_name", *paired_keys[:5], *training_fields],
    ]
    control = source_pair.loc[source_pair["model_name"].eq(CONTROL)].drop(columns="model_name")
    full = source_pair.loc[source_pair["model_name"].eq(FULL)].drop(columns="model_name")
    training = control.merge(
        full,
        on=paired_keys[:5],
        how="outer",
        suffixes=("_control", "_full"),
        indicator=True,
    )
    training_ok = training["_merge"].eq("both").all()
    for field in training_fields:
        left = training[f"{field}_control"]
        right = training[f"{field}_full"]
        training_ok = (
            training_ok
            and (
                left.astype("string").eq(right.astype("string")) | (left.isna() & right.isna())
            ).all()
        )
    check(
        results,
        "full_no_feature_identical_training_state",
        bool(training_ok),
        "only frozen feature overlay differs",
    )

    independent_metrics = independently_rebuild_paired_brier(joins)
    exported = pd.read_csv(primary / "paired_lead_metrics.csv")
    exported = exported.loc[exported["scope"].eq("all")]
    compared = independent_metrics.merge(exported, on="target_lead_sessions", validate="one_to_one")
    metrics_ok = np.allclose(
        compared["brier_improvement"], compared["paired_brier_improvement"], atol=1e-12
    ) and np.allclose(
        compared["economic_increment_bps"],
        compared["paired_economic_increment_bps"],
        atol=1e-9,
    )
    check(results, "independent_paired_metrics", metrics_ok, "Brier and economics rebuilt")
    primary_row = compared.loc[compared["target_lead_sessions"].eq(1)].iloc[0]
    check(
        results,
        "primary_endpoint_registration_and_value",
        float(primary_row["brier_improvement"]) < 0.0
        and int(primary_row["observable_targets"]) == 2_787,
        f"lead1_brier={primary_row['brier_improvement']:.12f}",
    )

    full_decisions = v2_decisions.loc[v2_decisions["model_name"].eq(FULL)]
    delay = independently_reconstruct_delay(full_decisions)
    immediate = delay.loc[delay["immediate"] & delay["status"].eq("filled")]
    shifted = delay.loc[delay["delayed"] & delay["status"].eq("filled")]
    categories = delay.groupby("category", observed=True)["primary_net_payoff_bps"].sum()
    delay_ok = (
        len(immediate) == 275
        and len(shifted) == 259
        and np.isclose(immediate["primary_net_payoff_bps"].sum(), -9_416.383037515745)
        and np.isclose(shifted["primary_net_payoff_bps"].sum(), 9_014.535299458603)
        and np.isclose(categories["introduced"] - categories["dropped"], 18_430.918336974348)
    )
    check(
        results,
        "original_v2_delay_reconstruction",
        delay_ok,
        "275/-9416.383 immediate; 259/+9014.535 shifted",
    )
    cost_ok = bool(
        np.allclose(
            full_decisions.loc[full_decisions["status"].eq("filled"), "primary_net_payoff_bps"],
            full_decisions.loc[full_decisions["status"].eq("filled"), "gross_payoff_bps"]
            - full_decisions.loc[full_decisions["status"].eq("filled"), "primary_total_cost_bps"],
            atol=1e-12,
        )
    )
    check(results, "cost_calculations", cost_ok, "gross - entry/exit/all applicable costs")

    matches = pd.read_parquet(primary / "same_setup_match_classification.parquet")
    restarted = pd.read_parquet(primary / "restarted_horizon_delayed_trades.parquet")
    check(
        results,
        "exact_same_setup_and_no_replacement",
        bool(
            not matches["exact_match"].astype(bool).any()
            and restarted["match_basis"].eq("structural_lineage_diagnostic").all()
        ),
        "same-loop/lineage candidates never promoted to exact",
    )
    constant = pd.read_parquet(primary / "constant_terminal_time_delayed_trades.parquet")
    check(
        results,
        "restarted_and_constant_terminal_separate",
        set(restarted.columns) != set(constant.columns)
        and not constant["constant_terminal_available"].astype(bool).any(),
        "original terminal precedes next-session entry",
    )

    feature_columns = [str(column) for column in forecasts if str(column).startswith("z__")]
    check(
        results,
        "episode_label_isolation",
        not any("episode" in column or "hindsight" in column for column in feature_columns)
        and forecasts["frozen_feature_values_json"]
        .astype(str)
        .str.contains("episode|hindsight", case=False, regex=True)
        .sum()
        == 0,
        "hindsight attached only after freeze",
    )
    loo = pd.read_csv(primary / "leave_one_stock_out_results.csv")
    check(
        results,
        "leave_one_stock_out_rebuilding",
        bool(
            loo["all_stock_dependent_inputs_rebuilt"].astype(bool).all()
            and loo["excluded_stock"].nunique() == 20
        ),
        "20 stocks; panels/features/shared/cell states and targets rebuilt",
    )

    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            contract["frozen_lineage"]["final_scored_v2_implementation_commit"],
            "HEAD",
        ],
        cwd=REPO,
        text=True,
    ).splitlines()
    forbidden_prefixes = (
        "packages/stocker_execution/",
        "apps/",
        "deployment/",
        "infra/",
    )
    check(
        results,
        "no_runtime_or_execution_paths_modified",
        not any(path.startswith(forbidden_prefixes) for path in changed),
        f"changed_paths_since_v2={len(changed)}",
    )

    passed = all(bool(item["passed"]) for item in results)
    file_hashes = {name: value for name, value in sorted(primary_files.items())}
    return {
        "audit_id": "dynamic_loop_edge_state_lead_lag_v1_independent_audit",
        "audit_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "auditor_sha256": sha256(Path(__file__)),
        "passed": passed,
        "research_only": True,
        "execution_enabled": False,
        "contract_hash": contract_hash,
        "checks": results,
        "machine_readable_and_plot_hashes": file_hashes,
        "primary_report_hash": sha256(primary_report),
        "exact_report_hash": sha256(exact_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=DEFAULT_ROOT / "primary")
    parser.add_argument("--exact", type=Path, default=DEFAULT_ROOT / "exact_rerun")
    parser.add_argument(
        "--primary-report",
        type=Path,
        default=WORK / "reports/20260715-dynamic-loop-edge-state-lead-lag-v1.md",
    )
    parser.add_argument(
        "--exact-report", type=Path, default=DEFAULT_ROOT / "exact_rerun/research_report.md"
    )
    args = parser.parse_args()
    result = audit(
        args.primary.resolve(),
        args.exact.resolve(),
        args.primary_report.resolve(),
        args.exact_report.resolve(),
    )
    write_json(args.primary.resolve() / "independent_audit.json", result)
    write_json(args.exact.resolve() / "independent_audit.json", result)
    if not result["passed"]:
        failed = [item["name"] for item in result["checks"] if not item["passed"]]
        raise SystemExit(f"independent audit failed: {failed}")
    print(f"independent audit passed: {len(result['checks'])} checks")


if __name__ == "__main__":
    main()
