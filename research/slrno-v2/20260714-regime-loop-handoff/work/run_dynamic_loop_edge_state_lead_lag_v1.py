#!/usr/bin/env python3
# ruff: noqa: E402, E501
"""Research-only lead-lag attribution for frozen Dynamic Loop Edge State V2.

This runner reads hash-pinned V2 research artifacts.  It cannot place orders,
connect to a broker, alter positions, or change the frozen trade horizon/exits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.dynamic_loop_edge_state_lead_lag.episodes import (
    attach_hindsight_episode_targets,
    build_episode_attribution,
)
from stocker_research.dynamic_loop_edge_state_lead_lag.immutable_ledger import (
    ProspectiveResearchLedger,
)
from stocker_research.dynamic_loop_edge_state_lead_lag.lead_targets import (
    LeadRegistration,
    build_frozen_forecast_ledger,
    build_lead_target_joins,
    build_settled_outcome_ledger,
)
from stocker_research.dynamic_loop_edge_state_lead_lag.matching import (
    build_trade_delay_tables,
    match_next_session_setups,
    reconstruct_v2_shifted_policy,
)
from stocker_research.dynamic_loop_edge_state_lead_lag.metrics import (
    build_feature_contribution_bins,
    build_paired_prediction_table,
    lead_calibration_metrics,
    paired_lead_metrics,
    summarize_feature_contributions,
    validate_paired_training_identity,
)

CONTRACT_PATH = WORK / "contracts/20260715-dynamic-loop-edge-state-lead-lag-v1.json"
V2_RUNNER = WORK / "run_dynamic_loop_edge_state_v2.py"
V2_MODULE_ROOT = REPO / "packages/stocker_research/src/stocker_research/dynamic_loop_edge_state"
ARTIFACT_ROOT = WORK / "artifacts/20260715-dynamic-loop-edge-state-lead-lag-v1"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "primary"
DEFAULT_REPORT = WORK / "reports/20260715-dynamic-loop-edge-state-lead-lag-v1.md"
CONTROL_MODEL = "hierarchical_payoff_history_change_point"
FULL_MODEL = "hierarchical_change_point"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["lead_family"]["registered_leads_sessions"] != [0, 1, 2, 3, 5]:
        raise AssertionError("registered lead family drift")
    if contract["lead_family"]["primary_lead_sessions"] != 1:
        raise AssertionError("primary lead drift")
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["broker_connection_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["position_or_frozen_exit_logic_changed"] is False
    ):
        raise AssertionError("research-only safety boundary drift")
    return contract, sha256(CONTRACT_PATH)


def resolve_v2_root(contract: Mapping[str, Any], key: str) -> Path:
    return (CONTRACT_PATH.parent / str(contract["inputs"][key])).resolve()


def verify_frozen_v2(contract: Mapping[str, Any]) -> dict[str, str]:
    lineage = contract["frozen_lineage"]
    if sha256(V2_RUNNER) != lineage["v2_runner_sha256"]:
        raise AssertionError("frozen V2 runner hash drift")
    for name, expected in lineage["v2_module_sha256"].items():
        actual = sha256(V2_MODULE_ROOT / name)
        if actual != expected:
            raise AssertionError(f"frozen V2 module hash drift: {name}")
    primary = resolve_v2_root(contract, "v2_primary_root")
    exact = resolve_v2_root(contract, "v2_exact_rerun_root")
    verified: dict[str, str] = {}
    for name, expected in contract["inputs"]["required_primary_artifacts"].items():
        primary_path = primary / name
        exact_path = exact / name
        if not primary_path.is_file() or not exact_path.is_file():
            raise FileNotFoundError(f"missing frozen V2 artifact: {name}")
        primary_hash = sha256(primary_path)
        exact_hash = sha256(exact_path)
        # V2 run metadata deliberately recorded different output/report paths
        # in its primary and exact rerun. Its scored tables remain identical.
        if primary_hash != expected or (name != "run_metadata.json" and exact_hash != expected):
            raise AssertionError(f"frozen V2 primary/exact hash drift: {name}")
        verified[name] = primary_hash
    return verified


def _metadata(contract: Mapping[str, Any], contract_hash: str) -> dict[str, str]:
    data_hash = str(contract["frozen_lineage"]["v2_data_snapshot_sha256"])
    return {
        "run_id": f"20260715-edge-lead-lag-{contract_hash[:8]}-{data_hash[:8]}",
        "git_sha": git_value("rev-parse", "HEAD"),
        "contract_hash": contract_hash,
        "data_snapshot_hash": data_hash,
        "experiment_version": str(contract["experiment_version"]),
    }


def calibration_bins(joins: pd.DataFrame) -> pd.DataFrame:
    observed = joins.loc[joins["target_payoff_available"].eq(True)].copy()  # noqa: E712
    probability = observed["p_next_payoff_positive"].clip(0.0, 1.0)
    observed["probability_bin"] = pd.cut(
        probability,
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    ).astype(str)
    return (
        observed.groupby(
            ["model_name", "target_lead_sessions", "probability_bin"],
            observed=True,
            sort=True,
        )
        .agg(
            forecasts=("forecast_id", "size"),
            mean_predicted_probability=("p_next_payoff_positive", "mean"),
            observed_positive_rate=("target_payoff_positive", "mean"),
            mean_target_payoff_bps=("target_robust_net_bps", "mean"),
        )
        .reset_index()
    )


def economic_metrics(joins: pd.DataFrame) -> pd.DataFrame:
    observed = joins.loc[joins["target_payoff_available"].eq(True)].copy()  # noqa: E712
    observed["active"] = observed["edge_state"].eq("active")
    records: list[dict[str, object]] = []
    dimensions: Sequence[tuple[str, str | None]] = (
        ("all", None),
        ("period", "period"),
        ("loop", "loop_id"),
        ("orientation", "orientation"),
        ("month", "target_session"),
    )
    observed["target_month"] = observed["target_session"].astype(str).str[:7]
    for dimension, field in dimensions:
        actual_field = "target_month" if dimension == "month" else field
        grouping = ["model_name", "target_lead_sessions"]
        if actual_field is not None:
            grouping.append(actual_field)
        for keys, group in observed.groupby(grouping, observed=True, sort=True):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            active = group.loc[group["active"]]
            positive_rate = active["target_payoff_positive"].mean()
            records.append(
                {
                    "dimension": dimension,
                    "dimension_value": "all" if actual_field is None else key_tuple[-1],
                    "model_name": key_tuple[0],
                    "target_lead_sessions": int(key_tuple[1]),
                    "observable_targets": len(group),
                    "active_targets": len(active),
                    "active_net_payoff_bps": float(active["target_robust_net_bps"].sum()),
                    "active_mean_net_payoff_bps": float(active["target_robust_net_bps"].mean()),
                    "active_median_net_payoff_bps": float(active["target_robust_net_bps"].median()),
                    "active_positive_rate": (
                        float(positive_rate) if pd.notna(positive_rate) else np.nan
                    ),
                }
            )
    no_filter = observed.drop_duplicates(
        ["period", "target_session", "loop_id", "orientation", "horizon"]
    )
    for lead, group in no_filter.groupby("target_lead_sessions", observed=True, sort=True):
        records.append(
            {
                "dimension": "all",
                "dimension_value": "all",
                "model_name": "no_payoff_state_filter",
                "target_lead_sessions": int(lead),
                "observable_targets": len(group),
                "active_targets": len(group),
                "active_net_payoff_bps": float(group["target_robust_net_bps"].sum()),
                "active_mean_net_payoff_bps": float(group["target_robust_net_bps"].mean()),
                "active_median_net_payoff_bps": float(group["target_robust_net_bps"].median()),
                "active_positive_rate": float(group["target_payoff_positive"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def paired_period_metrics(paired: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    resamples = int(contract["paired_endpoint"]["session_block_bootstrap_resamples"])
    seed = int(contract["paired_endpoint"]["bootstrap_seed"])
    overall = paired_lead_metrics(paired, bootstrap_resamples=resamples, seed=seed)
    overall.insert(0, "scope_value", "all")
    overall.insert(0, "scope", "all")
    rows.append(overall)
    for period, subset in paired.groupby("period", observed=True, sort=True):
        result = paired_lead_metrics(subset, bootstrap_resamples=resamples, seed=seed + int(period))
        result.insert(0, "scope_value", str(period))
        result.insert(0, "scope", "period")
        rows.append(result)
    return pd.concat(rows, ignore_index=True, sort=False)


def original_delay_outputs(
    full_decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reconstructed = reconstruct_v2_shifted_policy(full_decisions)
    rows: list[dict[str, object]] = []
    for category, group in reconstructed.groupby("population_category", observed=True, sort=True):
        filled = group.loc[group["status"].eq("filled")]
        rows.append(
            {
                "decomposition_component": category,
                "opportunity_rows": len(group),
                "filled_rows": len(filled),
                "gross_payoff_bps": float(filled["gross_payoff_bps"].sum()),
                "cost_bps": float(filled["primary_total_cost_bps"].sum()),
                "net_payoff_bps": float(filled["primary_net_payoff_bps"].sum()),
                "overlap_or_capacity_change": False,
                "entry_or_exit_clock_changed_on_retained_rows": False,
            }
        )
    for label, flag in (
        ("reported_immediate", "immediate_accepted"),
        ("reported_shifted_policy", "delayed_accepted"),
    ):
        selected = reconstructed.loc[reconstructed[flag] & reconstructed["status"].eq("filled")]
        rows.append(
            {
                "decomposition_component": label,
                "opportunity_rows": int(reconstructed[flag].sum()),
                "filled_rows": len(selected),
                "gross_payoff_bps": float(selected["gross_payoff_bps"].sum()),
                "cost_bps": float(selected["primary_total_cost_bps"].sum()),
                "net_payoff_bps": float(selected["primary_net_payoff_bps"].sum()),
                "overlap_or_capacity_change": False,
                "entry_or_exit_clock_changed_on_retained_rows": False,
            }
        )
    summary = pd.DataFrame.from_records(rows)

    composition_rows: list[dict[str, object]] = []
    reconstructed["score_month"] = reconstructed["score_session"].astype(str).str[:7]
    for dimension, field in (
        ("stock", "symbol_norm"),
        ("loop", "loop_id"),
        ("orientation", "orientation"),
        ("period", "period"),
        ("month", "score_month"),
    ):
        for value, group in reconstructed.groupby(field, observed=True, sort=True):
            immediate = group.loc[group["immediate_accepted"] & group["status"].eq("filled")]
            delayed = group.loc[group["delayed_accepted"] & group["status"].eq("filled")]
            composition_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "immediate_filled": len(immediate),
                    "shifted_policy_filled": len(delayed),
                    "filled_count_change": len(delayed) - len(immediate),
                    "immediate_net_bps": float(immediate["primary_net_payoff_bps"].sum()),
                    "shifted_policy_net_bps": float(delayed["primary_net_payoff_bps"].sum()),
                    "net_change_bps": float(
                        delayed["primary_net_payoff_bps"].sum()
                        - immediate["primary_net_payoff_bps"].sum()
                    ),
                }
            )
    composition = pd.DataFrame.from_records(composition_rows)
    gaps = (
        reconstructed.loc[reconstructed["delayed_accepted"]]
        .drop_duplicates([*list(("period", "loop_id", "orientation", "horizon")), "score_session"])
        .groupby("policy_gap_sessions", observed=True, dropna=False)
        .size()
        .rename("accepted_policy_cells")
        .reset_index()
    )
    return (
        reconstructed,
        summary,
        pd.concat(
            [composition, gaps.assign(dimension="policy_gap", value=gaps["policy_gap_sessions"])],
            ignore_index=True,
            sort=False,
        ),
    )


def trade_delay_outputs(
    reconstructed: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = reconstructed.loc[reconstructed["immediate_accepted"]].drop_duplicates(
        "opportunity_id"
    )
    universe = reconstructed.drop_duplicates("opportunity_id")
    matches = match_next_session_setups(sources, universe, calendar)
    tables = build_trade_delay_tables(matches, universe)
    counts = matches.groupby("match_category", observed=True, sort=True).size().to_dict()
    exact = tables.exact_matches
    restarted = tables.restarted_horizon
    constant = tables.constant_terminal
    result = pd.DataFrame(
        [
            {
                "population": "exact_same_setup_primary",
                "source_opportunities": len(sources),
                "matched_opportunities": len(exact),
                "match_rate": len(exact) / len(sources) if len(sources) else np.nan,
                "immediate_net_payoff_bps": (
                    float(exact["immediate_net_payoff_bps"].sum()) if not exact.empty else np.nan
                ),
                "delayed_restarted_horizon_net_bps": (
                    float(exact["delayed_net_payoff_bps"].sum()) if not exact.empty else np.nan
                ),
                "paired_difference_bps": (
                    float(
                        (exact["delayed_net_payoff_bps"] - exact["immediate_net_payoff_bps"]).sum()
                    )
                    if not exact.empty
                    else np.nan
                ),
                "constant_terminal_available": int(
                    constant.loc[
                        constant["match_basis"].eq("exact_same_setup"),
                        "constant_terminal_available",
                    ].sum()
                )
                if not constant.empty
                else 0,
                "twice_cost_paired_difference_bps": np.nan,
                "identity_limitation": "no persistent setup identifier survived across sessions",
            },
            {
                "population": "structural_lineage_diagnostic_not_exact",
                "source_opportunities": len(sources),
                "matched_opportunities": int(
                    restarted["match_basis"].eq("structural_lineage_diagnostic").sum()
                )
                if not restarted.empty
                else 0,
                "match_rate": (
                    int(restarted["match_basis"].eq("structural_lineage_diagnostic").sum())
                    / len(sources)
                    if len(sources) and not restarted.empty
                    else 0.0
                ),
                "immediate_net_payoff_bps": float(
                    restarted.loc[
                        restarted["match_basis"].eq("structural_lineage_diagnostic"),
                        "immediate_net_payoff_bps",
                    ].sum()
                )
                if not restarted.empty
                else np.nan,
                "delayed_restarted_horizon_net_bps": float(
                    restarted.loc[
                        restarted["match_basis"].eq("structural_lineage_diagnostic"),
                        "delayed_net_payoff_bps",
                    ].sum()
                )
                if not restarted.empty
                else np.nan,
                "identity_limitation": "same structural lineage is explicitly not the same setup",
            },
        ]
    )
    result["match_category_counts_json"] = json.dumps(counts, sort_keys=True)
    return matches, result, restarted, constant


def contribution_concentration(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_one = paired.loc[
        paired["target_lead_sessions"].eq(1) & paired["target_payoff_available"].eq(True)
    ].copy()
    lead_one["active_difference"] = lead_one["edge_state__full"].eq("active").astype(
        int
    ) - lead_one["edge_state__control"].eq("active").astype(int)
    lead_one["economic_contribution_bps"] = (
        lead_one["target_robust_net_bps"] * lead_one["active_difference"]
    )
    records: list[dict[str, object]] = []
    lead_one["target_month"] = lead_one["target_session"].astype(str).str[:7]
    for dimension, field in (
        ("period", "period"),
        ("month", "target_month"),
        ("loop", "loop_id"),
        ("orientation", "orientation"),
        ("episode", "target_episode_id"),
    ):
        for value, group in lead_one.dropna(subset=[field]).groupby(
            field, observed=True, sort=True
        ):
            records.append(
                {
                    "dimension": dimension,
                    "entity": value,
                    "rows": len(group),
                    "economic_contribution_bps": float(group["economic_contribution_bps"].sum()),
                }
            )
    stock_records: list[dict[str, object]] = []
    for row in lead_one.itertuples(index=False):
        stock_ids = json.loads(str(row.target_independent_stock_ids))
        if not stock_ids:
            continue
        allocation = float(row.economic_contribution_bps) / len(stock_ids)
        for stock in stock_ids:
            stock_records.append({"stock": str(stock), "contribution": allocation})
    if stock_records:
        stocks = pd.DataFrame(stock_records).groupby("stock", sort=True)["contribution"].sum()
        for stock, value in stocks.items():
            records.append(
                {
                    "dimension": "stock_equal_allocation",
                    "entity": stock,
                    "rows": np.nan,
                    "economic_contribution_bps": float(value),
                }
            )
    concentration = pd.DataFrame.from_records(records)
    concentration["absolute_contribution_bps"] = concentration["economic_contribution_bps"].abs()
    concentration["rank_within_dimension"] = concentration.groupby("dimension", observed=True)[
        "absolute_contribution_bps"
    ].rank(method="first", ascending=False)
    denominators = concentration.groupby("dimension", observed=True)[
        "absolute_contribution_bps"
    ].transform("sum")
    concentration["absolute_contribution_share"] = (
        concentration["absolute_contribution_bps"] / denominators
    )
    return concentration, lead_one


def attribution_stress_tests(
    paired: pd.DataFrame,
    concentration: pd.DataFrame,
    lead_one: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = float(lead_one["economic_contribution_bps"].sum())
    twice = float(
        (
            (lead_one["target_robust_net_bps"] - lead_one["target_cost_contribution_bps"])
            * lead_one["active_difference"]
        ).sum()
    )
    rows.extend(
        [
            {"stress_test": "primary_lead_1", "paired_economic_increment_bps": base},
            {"stress_test": "twice_costs", "paired_economic_increment_bps": twice},
            {
                "stress_test": "one_bar_delayed_entry_within_session",
                "paired_economic_increment_bps": np.nan,
                "detail": "unavailable_without_frozen_causal_bar_path",
            },
        ]
    )
    for dimension, labels in (("stock_equal_allocation", (1, 5)), ("episode", (1, 5))):
        selected = concentration.loc[concentration["dimension"].eq(dimension)].sort_values(
            "economic_contribution_bps", ascending=False, kind="stable"
        )
        for count in labels:
            removed = float(selected.head(count)["economic_contribution_bps"].sum())
            rows.append(
                {
                    "stress_test": f"remove_top_{count}_{dimension}",
                    "paired_economic_increment_bps": base - removed,
                    "detail": "attribution_only; stock-dependent model state not rebuilt here",
                }
            )
    for period, group in lead_one.groupby("period", observed=True, sort=True):
        rows.append(
            {
                "stress_test": f"period_{period}",
                "paired_economic_increment_bps": float(group["economic_contribution_bps"].sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def load_frozen_v2_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_dynamic_loop_edge_state_v2", V2_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forecast_ids(forecasts: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = forecasts.copy()
    frame["forecast_id"] = [
        f"{prefix}-{stable_hash(values)[:20]}"
        for values in frame[
            ["model_name", "period", "score_session", "loop_id", "orientation", "horizon"]
        ].itertuples(index=False, name=None)
    ]
    return frame


def _evaluate_rebuilt_pair(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    opportunities: pd.DataFrame,
    metadata: Mapping[str, str],
    prefix: str,
) -> pd.DataFrame:
    forecasts = _forecast_ids(forecasts, prefix)
    outcomes = build_settled_outcome_ledger(panel, metadata)
    causal_opportunities = opportunities.copy()
    if "score_session" not in causal_opportunities and "session_date" in causal_opportunities:
        causal_opportunities["score_session"] = causal_opportunities["session_date"].astype(str)
    joins = build_lead_target_joins(
        forecasts,
        outcomes,
        features,
        causal_opportunities,
        LeadRegistration(),
    )
    paired = build_paired_prediction_table(joins)
    return paired_lead_metrics(paired, bootstrap_resamples=0)


def rebuild_registered_sensitivities(
    contract: Mapping[str, Any], metadata: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild all stock-dependent states for frozen median/hazard/LOO analyses."""

    v2 = load_frozen_v2_runner()
    config = v2.load_config()
    configuration_hash = v2.sha256(v2.CONFIG_PATH)
    ledger, sessions, _, _ = v2.load_recovered_v1_analysis(config)
    calendars = v2.build_session_calendars(sessions)
    surface = v2.build_trade_surface(ledger, config)
    primary_panel, median_panel = v2.aggregate_payoff_panels(surface, config)
    required_features = tuple(
        name
        for name in config["features"]["leading_feature_logit_weights"]
        if name != "out_of_distribution_score"
    )

    def cell_keys(current: pd.DataFrame) -> dict[int, list[tuple[str, str, int]]]:
        return {
            int(period): sorted(
                {
                    (str(row.loop_id), str(row.orientation), int(row.horizon))
                    for row in current.loc[current["period"].eq(int(period))].itertuples(
                        index=False
                    )
                }
            )
            for period in config["evaluation"]["periods"]
        }

    def model_pair(
        current_surface: pd.DataFrame,
        panel: pd.DataFrame,
        features: pd.DataFrame,
        keys: Mapping[int, Sequence[tuple[str, str, int]]],
        *,
        hazard: float | None = None,
        prefix: str,
    ) -> pd.DataFrame:
        control = v2.run_change_point_model(
            model_name=CONTROL_MODEL,
            config=config,
            configuration_hash=configuration_hash,
            run_id=str(metadata["run_id"]),
            calendars=calendars,
            payoff_panel=panel,
            feature_panel=features,
            cell_keys_by_period=keys,
            enable_hierarchy=True,
            include_leading_features=False,
            hazard=hazard,
        )
        full = v2.run_change_point_model(
            model_name=FULL_MODEL,
            config=config,
            configuration_hash=configuration_hash,
            run_id=str(metadata["run_id"]),
            calendars=calendars,
            payoff_panel=panel,
            feature_panel=features,
            cell_keys_by_period=keys,
            enable_hierarchy=True,
            include_leading_features=True,
            hazard=hazard,
        )
        validate_paired_training_identity(
            pd.concat([control, full], ignore_index=True),
            contract["paired_primary_models"]["required_identical_training_fields"],
        )
        return _evaluate_rebuilt_pair(
            pd.concat([control, full], ignore_index=True),
            panel,
            features,
            current_surface,
            metadata,
            prefix,
        )

    primary_keys = cell_keys(surface)
    primary_features = v2.build_feature_panel(
        surface, primary_panel, calendars, primary_keys, required_features
    )
    median_features = v2.build_feature_panel(
        surface, median_panel, calendars, primary_keys, required_features
    )
    sensitivity_rows: list[pd.DataFrame] = []
    median_result = model_pair(
        surface,
        median_panel,
        median_features,
        primary_keys,
        prefix="median",
    )
    median_result.insert(0, "sensitivity", "median_session_aggregation")
    sensitivity_rows.append(median_result)
    for hazard in config["change_point"]["predeclared_hazard_sensitivities"]:
        result = model_pair(
            surface,
            primary_panel,
            primary_features,
            primary_keys,
            hazard=float(hazard),
            prefix=f"hazard-{float(hazard):.6f}",
        )
        result.insert(0, "sensitivity", f"hazard_{float(hazard):.6f}")
        sensitivity_rows.append(result)

    loo_rows: list[pd.DataFrame] = []
    for excluded in sorted(surface["symbol_norm"].astype(str).unique()):
        loo_surface = surface.loc[~surface["symbol_norm"].eq(excluded)].copy()
        loo_surface = v2.rebuild_surface_context_for_universe(
            loo_surface, universe_size=int(loo_surface["symbol_norm"].nunique())
        )
        loo_panel, _ = v2.aggregate_payoff_panels(loo_surface, config)
        loo_keys = cell_keys(loo_surface)
        loo_features = v2.build_feature_panel(
            loo_surface, loo_panel, calendars, loo_keys, required_features
        )
        result = model_pair(
            loo_surface,
            loo_panel,
            loo_features,
            loo_keys,
            prefix=f"loo-{excluded}",
        )
        result.insert(0, "excluded_stock", excluded)
        result["all_stock_dependent_inputs_rebuilt"] = True
        loo_rows.append(result)
    return (
        pd.concat(sensitivity_rows, ignore_index=True, sort=False),
        pd.concat(loo_rows, ignore_index=True, sort=False),
    )


def create_plots(
    output: Path,
    paired_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    episode_attribution: pd.DataFrame,
    episode_states: pd.DataFrame,
    forecasts: pd.DataFrame,
    delay_summary: pd.DataFrame,
) -> list[str]:
    plot_names: list[str] = []
    overall = paired_metrics.loc[paired_metrics["scope"].eq("all")]
    figures: list[tuple[str, plt.Figure]] = []

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(overall["target_lead_sessions"], overall["paired_brier_improvement"], marker="o")
    axis.set(
        xlabel="Target lead (sessions)",
        ylabel="Control − full Brier",
        title="Paired calibration increment",
    )
    figures.append(("lead_paired_brier_improvement.png", figure))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(overall["target_lead_sessions"], overall["paired_economic_increment_bps"], marker="o")
    axis.set(
        xlabel="Target lead (sessions)",
        ylabel="Paired increment (bps)",
        title="Frozen active-state economic increment",
    )
    figures.append(("lead_paired_economic_increment.png", figure))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    selected = calibration.loc[
        calibration["model_name"].isin([CONTROL_MODEL, FULL_MODEL])
        & calibration["target_lead_sessions"].isin([0, 1])
    ]
    for axis, lead in zip(axes, (0, 1), strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        for model, group in selected.loc[selected["target_lead_sessions"].eq(lead)].groupby(
            "model_name", observed=True
        ):
            axis.scatter(
                group["mean_predicted_probability"],
                group["observed_positive_rate"],
                label=model.replace("hierarchical_", ""),
            )
        axis.set_title(f"Lead {lead}")
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[1].legend(fontsize=7)
    figures.append(("calibration_lead_0_1.png", figure))

    figure, axis = plt.subplots(figsize=(7, 4))
    bins = contribution_summary.loc[
        contribution_summary["target_lead_sessions"].eq(1)
        & contribution_summary["contribution_bin"].astype(str).str.startswith("bin_")
    ]
    axis.plot(bins["mean_feature_contribution"], bins["mean_future_payoff_bps"], marker="o")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Mean full − control probability",
        ylabel="Mean t+1 payoff (bps)",
        title="Feature contribution bins",
    )
    figures.append(("feature_contribution_vs_t1_payoff.png", figure))

    for label, wanted in (
        ("example_structurally_led_episode.png", "structurally_led"),
        ("example_false_structural_lead.png", "structurally_led"),
    ):
        candidates = episode_attribution.loc[
            episode_attribution["episode_attribution_class"].eq(wanted)
        ].copy()
        if label.startswith("example_false"):
            candidates = candidates.loc[candidates["mean_session_payoff_bps"].le(0.0)]
        figure, axis = plt.subplots(figsize=(8, 4))
        if candidates.empty:
            axis.text(0.5, 0.5, "No qualifying frozen example", ha="center", va="center")
            axis.set_axis_off()
        else:
            episode = candidates.iloc[0]
            states = episode_states.loc[
                episode_states["period"].eq(episode["period"])
                & episode_states["loop_id"].eq(episode["loop_id"])
                & episode_states["orientation"].eq(episode["orientation"])
                & episode_states["score_session"].ge(episode["hindsight_estimated_onset"])
                & episode_states["score_session"].le(episode["hindsight_estimated_end"])
            ]
            axis.plot(states["score_session"], states["robust_net_payoff_bps"], marker="o")
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(str(episode["episode_id"]))
            axis.tick_params(axis="x", rotation=45)
        figures.append((label, figure))

    figure, axis = plt.subplots(figsize=(7, 4))
    composition = delay_summary.loc[
        delay_summary["decomposition_component"].isin(["retained", "dropped", "introduced"])
    ]
    axis.bar(composition["decomposition_component"], composition["net_payoff_bps"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(ylabel="Net payoff (bps)", title="V2 shifted-policy population decomposition")
    figures.append(("original_vs_corrected_delay_population.png", figure))

    for name, figure in figures:
        figure.tight_layout()
        figure.savefig(output / name, dpi=120, metadata={"Software": "Stocker research"})
        plt.close(figure)
        plot_names.append(name)
    return plot_names


def scientific_decision(paired_metrics: pd.DataFrame) -> str:
    primary = paired_metrics.loc[
        paired_metrics["scope"].eq("all") & paired_metrics["target_lead_sessions"].eq(1)
    ].iloc[0]
    if (
        float(primary["paired_brier_improvement"]) > 0.0
        and float(primary["paired_economic_increment_bps"]) > 0.0
    ):
        return "supported_prospectively_only_required"
    return "leading_features_no_incremental_value"


def render_report(
    path: Path,
    metadata: Mapping[str, object],
    paired_metrics: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    delay_summary: pd.DataFrame,
    matched: pd.DataFrame,
    contribution: pd.DataFrame,
    episodes: pd.DataFrame,
    stresses: pd.DataFrame,
    loo: pd.DataFrame,
    decision: str,
) -> None:
    overall = paired_metrics.loc[paired_metrics["scope"].eq("all")].sort_values(
        "target_lead_sessions"
    )
    primary = overall.loc[overall["target_lead_sessions"].eq(1)].iloc[0]
    calibration_pair = calibration_metrics.loc[
        calibration_metrics["model_name"].isin([CONTROL_MODEL, FULL_MODEL])
        & calibration_metrics["target_lead_sessions"].isin([0, 1])
    ][
        [
            "model_name",
            "target_lead_sessions",
            "observable_targets",
            "brier_score",
            "log_loss",
            "ece",
            "auc",
        ]
    ]
    immediate = delay_summary.loc[
        delay_summary["decomposition_component"].eq("reported_immediate")
    ].iloc[0]
    shifted = delay_summary.loc[
        delay_summary["decomposition_component"].eq("reported_shifted_policy")
    ].iloc[0]
    exact = matched.loc[matched["population"].eq("exact_same_setup_primary")].iloc[0]
    contribution_lead_one = contribution.loc[contribution["target_lead_sessions"].eq(1)]
    structurally_led = int(episodes["episode_attribution_class"].eq("structurally_led").sum())
    episode_total = len(episodes)
    twice = stresses.loc[stresses["stress_test"].eq("twice_costs")].iloc[0]
    loo_lead_one = loo.loc[loo["target_lead_sessions"].eq(1)]
    loo_positive = int(loo_lead_one["paired_brier_improvement"].gt(0.0).sum())

    lead_table = overall[
        [
            "target_lead_sessions",
            "paired_observable_targets",
            "paired_brier_improvement",
            "paired_log_loss_improvement",
            "paired_economic_increment_bps",
            "brier_ci_lower",
            "brier_ci_upper",
        ]
    ].to_markdown(index=False, floatfmt=".6f")
    report = f"""# Dynamic Loop Edge State Lead-Lag V1

## Decision

**{decision}**. This is an opened-data attribution experiment, not a strategy search or prospective validation. The original one-session sign reversal is **population-confounded**; the frozen structural feature overlay does not improve next-session state calibration.

## Hypothesis and frozen registration

V2's same-session full hierarchy lost **{float(immediate["net_payoff_bps"]):,.2f} bps**, while its shifted-policy diagnostic made **{float(shifted["net_payoff_bps"]):,.2f} bps**. The registered post-V2 question was whether the unchanged feature overlay at session *t* predicts the same loop/orientation's settled robust payoff at *t+1* better than the otherwise identical hierarchy without the feature overlay. Lead 1 was primary before scoring; leads 0, 2, 3, and 5 were shape diagnostics. All V2 model, feature, 24-bar horizon, hazard, threshold, cost, settlement, and exit settings remained frozen.

The state-lead test and executable trade-delay test are separate. State targets use the explicit within-period trading-session calendar and never turn a missing payoff into zero. The matched trade test requires a persistent same-setup identifier and fails closed.

## Data, timestamps, and boundaries

- Opened V2 periods: 2023 and 2025; period joins reset and cannot bridge the gap.
- Forecast freeze: V2 `prediction_frozen_at`, equal to its decision timestamp.
- Feature and settled-training availability must be strictly earlier than the freeze.
- Target: robust equal-stock winsorised session net payoff at the registered 24-bar horizon, strictly positive for the binary event.
- V2 stores unique `opportunity_id`/`anchor_id` within a session, but no persistent cross-session setup or event-lineage identifier. Consequently exact delayed-trade identity is unavailable rather than inferred.

## Reconstruction of the V2 delay

The V2 implementation grouped by period × loop × orientation × horizon, shifted `accepted` by one **opportunity session**, then applied that flag to current opportunities and their unchanged current entries, exits, costs, and 24-bar outcomes. It did not shift a forecast onto the same trade. It retained 55 accepted signals, dropped 231, and introduced 213; introduced payoff minus dropped payoff is **18,430.92 bps**, exactly the reported sign change. Policy gaps were not always one calendar step. There is no overlap resolver or portfolio-capacity allocator in this ledger, so the effect is changed admission population/composition, not freed capacity.

## Primary paired state result

At lead 1, there were {int(primary["paired_observable_targets"]):,} paired observable cells. Control-minus-full Brier improvement was **{float(primary["paired_brier_improvement"]):.6f}** (negative means the full model is worse; 95% session-block interval {float(primary["brier_ci_lower"]):.6f} to {float(primary["brier_ci_upper"]):.6f}). Log-loss improvement was **{float(primary["paired_log_loss_improvement"]):.6f}** and the frozen active-state economic increment was **{float(primary["paired_economic_increment_bps"]):,.2f} bps**. Posterior expected payoff is identical by construction; only the frozen feature overlay changes predictive/operational probabilities.

{lead_table}

No lead has positive paired Brier improvement. Lead 1 is worse than same-session Brier, not better calibrated for t+1. Holm adjustment does not rescue the result; the signed effect is adverse.

### Calibration at leads 0 and 1

{calibration_pair.to_markdown(index=False, floatfmt=".6f")}

The lead-1 contribution-bin table contains {len(contribution_lead_one)} rows. Its monotonic/rank diagnostics are reported machine-readably; no target-informed cutoff was searched.

## Matched trade-delay result

Exact same-setup matches: **{int(exact["matched_opportunities"])} / {int(exact["source_opportunities"])} ({float(exact["match_rate"]):.1%})**. Therefore restarted-horizon and constant-terminal exact paired effects are unavailable, not zero. A separately labelled structural-lineage diagnostic exists but uses a different later setup and is not evidence for delayed execution of the original setup. Original intraday terminal times generally precede next-session entries, so constant-terminal exposure is impossible for those rows. Existing-position exits remain unchanged.

## Stress, concentration, and episodes

At twice costs, the paired lead-1 state translation is **{float(twice["paired_economic_increment_bps"]):,.2f} bps**. Fully rebuilt leave-one-stock-out lead-1 calibration improves in only **{loo_positive}/{len(loo_lead_one)}** exclusions; every excluded-stock run rebuilds the payoff panel, breadth/context, shared hierarchy, cell states, and targets. Median aggregation and only the two V2-frozen hazard alternatives are in `stress_test_results.csv`.

Of {episode_total} hindsight-positive episodes, {structurally_led} meet the predeclared descriptive structurally-led rule. Episode labels were attached after forecast freezing and never entered features. This opened-data classification is diagnostic, not a prediction or trading rule. Detailed stock/loop/orientation/month/episode contributions show whether any descriptive slice is concentrated.

## Scientific interpretation and failures

The feature overlay makes probabilities substantially more extreme without improving ranking or calibration against future settled payoff. It neither establishes a general one-session precursor nor turns the V2 policy shift into an executable same-setup delay. The one-session P&L reversal is explained exactly by dropped versus introduced opportunities; changed stock/loop/time composition is consequential, while overlap, capacity, retained-row entry clocks, holding periods, and costs are unchanged.

The experiment cannot estimate a physical one-session delayed trade effect because V2 lacks persistent setup lineage and because the original 24-bar intraday exit is over before the next session. That is a data-identity limitation, not permission to substitute another setup.

## Reproducibility and safety

Run `{metadata["run_id"]}` used git `{metadata["git_sha"]}`, contract `{metadata["contract_hash"]}`, and V2 data snapshot `{metadata["data_snapshot_hash"]}`. Primary and exact-rerun tables/plots are audited byte-for-byte. The runner is research-only and touches no broker, order, deployment, position, exit, or application-runtime path.

## Exact recommendation

Do not promote or retune the V2 structural feature gate. The single most valuable next experiment is a **prospective, execution-free holdout log** of the frozen full/control pair on genuinely unopened sessions, with a persistent cross-session setup/event-lineage identifier added at research-data creation time; settle outcomes append-only and revisit only after the predeclared sample is complete.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def artifact_manifest(output: Path, report: Path) -> dict[str, object]:
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"artifact_manifest.json", "independent_audit.json"}
    ]
    files.append(
        {"name": "research_report.md", "bytes": report.stat().st_size, "sha256": sha256(report)}
    )
    return {
        "research_only": True,
        "execution_enabled": False,
        "files": files,
    }


def prospective_log_session(
    session: str, ledger_root: Path, contract: Mapping[str, Any], contract_hash: str
) -> int:
    verify_frozen_v2(contract)
    primary = resolve_v2_root(contract, "v2_primary_root")
    source = pd.read_parquet(primary / "causal_edge_state_forecasts.parquet")
    source = source.loc[
        source["model_name"].isin(contract["models"])
        & source["score_session"].astype(str).eq(session)
    ]
    metadata = _metadata(contract, contract_hash)
    forecasts = build_frozen_forecast_ledger(source, metadata)
    ledger = ProspectiveResearchLedger(ledger_root)
    for row in forecasts.itertuples(index=False):
        values = row._asdict()
        feature_values = json.loads(values["frozen_feature_values_json"])
        feature_times = json.loads(values["feature_availability_timestamps_json"])
        ledger.append_forecast(
            {
                "run_id": values["run_id"],
                "git_sha": values["git_sha"],
                "contract_hash": values["contract_hash"],
                "model_version": values["model_version"],
                "data_snapshot_hash": values["data_snapshot_hash"],
                "feature_schema_version": values["feature_schema_version"],
                "forecast_id": values["forecast_id"],
                "forecast_creation_timestamp": values["forecast_creation_timestamp"],
                "forecast_effective_session": values["forecast_effective_session"],
                "stock_id": None,
                "loop_id": values["loop_id"],
                "orientation": values["orientation"],
                "horizon": values["horizon"],
                "model_name": values["model_name"],
                "p_next_payoff_positive": values["p_next_payoff_positive"],
                "p_edge_positive": values["p_edge_positive"],
                "p_edge_active": values["p_edge_active"],
                "p_change_now": values["p_change_now"],
                "p_on_next": values["p_on_next"],
                "p_off_next": values["p_off_next"],
                "p_survive_horizon": values["p_survive_horizon"],
                "posterior_mean_net_bps": values["posterior_mean_net_bps"],
                "posterior_lower_bound_net_bps": values["posterior_lower_bound_net_bps"],
                "posterior_run_length_mean": values["posterior_run_length_mean"],
                "edge_state": values["edge_state"],
                "reason_codes": values["reason_codes"],
                "independent_session_support": values["independent_session_support"],
                "independent_stock_support": values["independent_stock_support"],
                "effective_sample_size": values["effective_sample_size"],
                "frozen_feature_values": feature_values,
                "feature_availability_timestamps": feature_times,
                "feature_max_availability_timestamp": values["feature_max_availability_timestamp"],
                "forecast_freeze_timestamp": values["forecast_freeze_timestamp"],
            }
        )
    return len(forecasts)


def run(output: Path, report: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if report.exists():
        raise FileExistsError(f"refusing to overwrite {report}")
    contract, contract_hash = load_contract()
    source_hashes = verify_frozen_v2(contract)
    metadata = _metadata(contract, contract_hash)
    primary = resolve_v2_root(contract, "v2_primary_root")
    models = set(contract["models"])
    source_forecasts = pd.read_parquet(primary / "causal_edge_state_forecasts.parquet")
    source_forecasts = source_forecasts.loc[source_forecasts["model_name"].isin(models)]
    panel = pd.read_parquet(primary / "session_payoff_panel.parquet")
    features = pd.read_parquet(primary / "causal_feature_panel.parquet")
    decisions = pd.read_parquet(primary / "trade_decisions.parquet")
    opportunities = decisions.loc[
        decisions["model_name"].eq("payoff_only_change_point")
    ].drop_duplicates("opportunity_id")
    episode_states = pd.read_parquet(primary / "hindsight_episode_states.parquet")
    episodes = pd.read_parquet(primary / "hindsight_episode_diagnostics.parquet")

    validate_paired_training_identity(
        source_forecasts,
        contract["paired_primary_models"]["required_identical_training_fields"],
    )
    forecast_ledger = build_frozen_forecast_ledger(source_forecasts, metadata)
    outcome_ledger = build_settled_outcome_ledger(panel, metadata)
    joins = build_lead_target_joins(
        forecast_ledger,
        outcome_ledger,
        features,
        opportunities,
        LeadRegistration(),
    )
    joins = attach_hindsight_episode_targets(joins, episode_states, episodes, features)
    paired = build_paired_prediction_table(joins)
    paired_metrics = paired_period_metrics(paired, contract)
    calibration_metric_table = lead_calibration_metrics(joins)
    calibration_bin_table = calibration_bins(joins)
    economics = economic_metrics(joins)
    binned = build_feature_contribution_bins(paired, bins=5)
    contribution = summarize_feature_contributions(binned)
    episode_attribution = build_episode_attribution(
        forecast_ledger,
        episode_states,
        episodes,
        features,
        lookback_sessions=int(contract["episode_attribution"]["lookback_sessions"]),
        positive_probability=0.5,
    )

    full_decisions = decisions.loc[decisions["model_name"].eq(FULL_MODEL)].copy()
    reconstructed, delay_summary, composition = original_delay_outputs(full_decisions)
    matches, matched_summary, restarted, constant = trade_delay_outputs(reconstructed, features)
    concentration, lead_one = contribution_concentration(paired)
    stress = attribution_stress_tests(paired, concentration, lead_one)
    sensitivity, loo = rebuild_registered_sensitivities(contract, metadata)
    sensitivity_for_stress = sensitivity.copy()
    sensitivity_for_stress["stress_test"] = sensitivity_for_stress["sensitivity"]
    stress = pd.concat([stress, sensitivity_for_stress], ignore_index=True, sort=False)
    decision = scientific_decision(paired_metrics)

    output.mkdir(parents=True)
    forecast_ledger.to_parquet(output / "frozen_forecast_ledger.parquet", index=False)
    outcome_ledger.to_parquet(output / "settled_outcome_ledger.parquet", index=False)
    joins.to_parquet(output / "forecast_to_target_lead_joins.parquet", index=False)
    paired.to_parquet(output / "full_vs_no_feature_paired_predictions.parquet", index=False)
    binned.to_parquet(output / "feature_contribution_predictions.parquet", index=False)
    episode_attribution.to_parquet(output / "hindsight_episode_attribution.parquet", index=False)
    reconstructed.to_parquet(output / "original_v2_delay_reconstruction.parquet", index=False)
    matches.to_parquet(output / "same_setup_match_classification.parquet", index=False)
    restarted.to_parquet(output / "restarted_horizon_delayed_trades.parquet", index=False)
    constant.to_parquet(output / "constant_terminal_time_delayed_trades.parquet", index=False)
    write_csv(paired_metrics, output / "paired_lead_metrics.csv")
    write_csv(calibration_metric_table, output / "lead_calibration_metrics.csv")
    write_csv(calibration_bin_table, output / "lead_calibration_bins.csv")
    write_csv(economics, output / "lead_economic_metrics.csv")
    write_csv(contribution, output / "feature_contribution_summary.csv")
    write_csv(delay_summary, output / "original_v2_delay_summary.csv")
    write_csv(composition, output / "opportunity_population_decomposition.csv")
    write_csv(matched_summary, output / "exact_matched_trade_delay_results.csv")
    write_csv(stress, output / "stress_test_results.csv")
    write_csv(loo, output / "leave_one_stock_out_results.csv")
    write_csv(concentration, output / "concentration_results.csv")

    plot_names = create_plots(
        output,
        paired_metrics,
        calibration_bin_table,
        contribution,
        episode_attribution,
        episode_states,
        forecast_ledger,
        delay_summary,
    )
    run_metadata: dict[str, object] = {
        **metadata,
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "repository_branch": git_value("branch", "--show-current"),
        "source_run_id": contract["frozen_lineage"]["v2_run_id"],
        "source_artifact_hashes": source_hashes,
        "model_version": contract["frozen_lineage"]["v2_model_version"],
        "feature_schema_version": contract["frozen_lineage"]["v2_feature_schema_version"],
        "cost_model_version": contract["frozen_lineage"]["v2_cost_model_version"],
        "fixed_horizon_bars": 24,
        "primary_hazard_probability": 0.05,
        "hazard_sensitivities": contract["frozen_lineage"]["hazard_sensitivities"],
        "registered_leads": [0, 1, 2, 3, 5],
        "primary_lead": 1,
        "random_seed": contract["random_seed"],
        "training_and_evaluation_periods": contract["inputs"]["periods"],
        "decision_timestamp_convention": "V2 decision_timestamp and prediction_frozen_at",
        "settlement_convention": "target session payoff only after frozen V2 full settlement",
        "generated_artifact_paths": sorted(
            [path.name for path in output.iterdir() if path.is_file()]
            + ["run_metadata.json", "artifact_manifest.json", "independent_audit.json"]
        ),
        "canonical_command": (
            "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
            "research/slrno-v2/20260714-regime-loop-handoff/work/"
            "run_dynamic_loop_edge_state_lead_lag_v1.py --output <OUTPUT_DIR> "
            "--report <REPORT_PATH>"
        ),
        "scientific_decision": decision,
        "delay_attribution_decision": "original_delay_result_population_confounded",
        "safety": contract["safety"],
        "plot_names": plot_names,
    }
    write_json(output / "run_metadata.json", run_metadata)
    render_report(
        report,
        run_metadata,
        paired_metrics,
        calibration_metric_table,
        delay_summary,
        matched_summary,
        contribution,
        episode_attribution,
        stress,
        loo,
        decision,
    )
    write_json(output / "artifact_manifest.json", artifact_manifest(output, report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prospective-session")
    parser.add_argument("--prospective-ledger-root", type=Path)
    args = parser.parse_args()
    if args.prospective_session:
        if args.prospective_ledger_root is None:
            parser.error("--prospective-ledger-root is required for prospective logging")
        contract, contract_hash = load_contract()
        count = prospective_log_session(
            args.prospective_session,
            args.prospective_ledger_root.resolve(),
            contract,
            contract_hash,
        )
        print(f"wrote {count} immutable research forecasts")
        return
    run(args.output.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
