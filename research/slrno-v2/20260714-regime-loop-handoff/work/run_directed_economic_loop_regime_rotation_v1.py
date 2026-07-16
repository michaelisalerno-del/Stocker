# ruff: noqa: E402, E501
"""Research-only directed economic loop-regime rotation experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stocker-mplconfig")

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.directed_economic_rotation import (
    ActivationRegistration,
    FamilyTaxonomy,
    GraphSettings,
    MaturedRotationExample,
    PastOnlyRotationGraph,
    PrequentialSettings,
    ProspectiveRotationLedger,
    activation_metrics,
    aggregate_family_states,
    apply_cost_stress,
    build_activation_targets,
    build_family_episode_intervals,
    build_family_payoff_support,
    calibration_table,
    derive_source_events,
    paired_model_comparison,
    permute_source_events,
    run_prequential_rotation,
    shift_source_events,
    shrink_pair_probability,
    system_activation_metrics,
    translate_predictions_to_opportunities,
)

CONTRACT_PATH = WORK / "contracts/20260716-directed-economic-loop-regime-rotation-v1.json"
MAPPING_PATH = (
    WORK / "contracts/20260716-directed-economic-loop-regime-rotation-v1-family-mapping.json"
)
DEFAULT_OUTPUT = WORK / "artifacts/20260716-directed-economic-loop-regime-rotation-v1/primary"
DEFAULT_REPORT = WORK / "reports/20260716-directed-economic-loop-regime-rotation-v1.md"
MODEL_VERSION = "directed_economic_loop_regime_rotation_v1.0.0"
RUN_TIMESTAMP = "2026-07-16T00:00:00+00:00"
OUTCOME_COLUMNS = {
    "activation_target",
    "target_available",
    "target_status",
    "label_availability_timestamp",
    "first_activation_session",
    "target_episode_ids",
    "observed_activation_count",
    "multiple_activation_flag",
    "no_activation_flag",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object, *, length: int = 24) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def safe_json(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(safe_json(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def ensure_new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")


def _resolved_contract_path(value: str) -> Path:
    return (CONTRACT_PATH.parent / value).resolve()


def input_paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    return {
        "family_mapping": _resolved_contract_path(inputs["family_mapping"]["path"]),
        "pair_states": _resolved_contract_path(inputs["causal_edge_state_forecasts"]["path"]),
        "session_panel": _resolved_contract_path(inputs["session_payoff_panel"]["path"]),
        "episode_states": _resolved_contract_path(inputs["hindsight_episode_states"]["path"]),
        "episode_diagnostics": _resolved_contract_path(
            inputs["hindsight_episode_diagnostics"]["path"]
        ),
        "trade_decisions": _resolved_contract_path(inputs["trade_decisions"]["path"]),
        "one_bar_delay": _resolved_contract_path(inputs["one_bar_delay_outcomes"]["path"]),
        "cycle_dictionary": _resolved_contract_path(inputs["cycle_dictionary"]["path"]),
        "v2_rebuild_runner": _resolved_contract_path(inputs["v2_rebuild_runner"]["path"]),
        "v2_rebuild_config": _resolved_contract_path(inputs["v2_rebuild_config"]["path"]),
    }


def verify_contract_and_inputs() -> tuple[dict[str, Any], dict[str, str], str]:
    contract: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    safety = contract["safety"]
    required_safe = {
        "research_only": True,
        "live_ordering_enabled": False,
        "broker_connection_enabled": False,
        "ig_integration_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "deployment_enabled": False,
        "application_runtime_changed": False,
        "position_management_changed": False,
        "existing_exit_logic_changed": False,
        "api_keys_or_secrets_used": False,
    }
    drift = [name for name, expected in required_safe.items() if safety.get(name) is not expected]
    if drift:
        raise AssertionError(f"research-only safety contract drift: {drift}")
    if contract.get("registered_before_scoring") is not True:
        raise AssertionError("contract was not registered before scoring")
    if int(contract["activation_target"]["primary_window_sessions"]) != 3:
        raise AssertionError("primary target window drift")
    paths = input_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "family_mapping": contract["inputs"]["family_mapping"]["sha256"],
        "pair_states": contract["inputs"]["causal_edge_state_forecasts"]["sha256"],
        "session_panel": contract["inputs"]["session_payoff_panel"]["sha256"],
        "episode_states": contract["inputs"]["hindsight_episode_states"]["sha256"],
        "episode_diagnostics": contract["inputs"]["hindsight_episode_diagnostics"]["sha256"],
        "trade_decisions": contract["inputs"]["trade_decisions"]["sha256"],
        "cycle_dictionary": contract["inputs"]["cycle_dictionary"]["sha256"],
        "v2_rebuild_runner": contract["inputs"]["v2_rebuild_runner"]["sha256"],
        "v2_rebuild_config": contract["inputs"]["v2_rebuild_config"]["sha256"],
    }
    mismatches = sorted(name for name, value in expected.items() if hashes[name] != value)
    if mismatches:
        raise AssertionError(f"frozen input hash mismatch: {mismatches}")
    sequential_root = paths["one_bar_delay"].parent
    manifest_path = sequential_root / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("sequential-veto artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get(paths["one_bar_delay"].name) != hashes["one_bar_delay"]:
        raise AssertionError("one-bar-delay source does not match its frozen manifest")
    snapshot_payload = {"contract": sha256(CONTRACT_PATH), **hashes}
    snapshot = hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract, hashes, snapshot


def split_forecast_and_outcome_ledgers(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_columns = ["forecast_id", *sorted(OUTCOME_COLUMNS & set(scored.columns))]
    outcomes = scored.loc[:, outcome_columns].copy()
    forecasts = scored.drop(columns=sorted(OUTCOME_COLUMNS & set(scored.columns))).copy()
    if any(column in forecasts for column in OUTCOME_COLUMNS):
        raise AssertionError("outcome leaked into immutable forecast ledger")
    if forecasts["forecast_id"].duplicated().any() or outcomes["forecast_id"].duplicated().any():
        raise AssertionError("forecast or outcome IDs are not unique")
    return forecasts, outcomes


def verify_exact_rerun(output: Path, primary: Path) -> dict[str, object]:
    suffixes = {".parquet", ".csv", ".json"}
    excluded = {"artifact_manifest.json", "exact_rerun_identity.json", "independent_audit.json"}
    output_files = {
        path.name: path
        for path in output.iterdir()
        if path.is_file() and path.suffix in suffixes and path.name not in excluded
    }
    primary_files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.suffix in suffixes and path.name not in excluded
    }
    missing = sorted(set(primary_files) - set(output_files))
    extra = sorted(set(output_files) - set(primary_files))
    mismatches = sorted(
        name
        for name in set(output_files) & set(primary_files)
        if sha256(output_files[name]) != sha256(primary_files[name])
    )
    return {
        "primary_path": str(primary),
        "compared_machine_readable_files": len(primary_files),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
        "byte_identical": not missing and not extra and not mismatches,
    }


def scientific_decision(
    comparisons: pd.DataFrame,
    *,
    economic_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    loo_results: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    primary = comparisons.loc[comparisons["comparison"].eq("M3_vs_M1")]
    secondary = comparisons.loc[comparisons["comparison"].eq("M3_vs_M2")]
    if "target_window_sessions" in comparisons:
        primary = primary.loc[primary["target_window_sessions"].eq(3)]
        secondary = secondary.loc[secondary["target_window_sessions"].eq(3)]
    if "period_slice" in comparisons:
        primary = primary.loc[primary["period_slice"].eq("all")]
        secondary = secondary.loc[secondary["period_slice"].eq("all")]
    if primary.empty or secondary.empty:
        return "experiment_blocked_by_missing_state_or_family_identity"
    primary_row = primary.iloc[0]
    secondary_row = secondary.iloc[0]
    primary_positive = bool(
        float(primary_row["brier_improvement"]) > 0.0
        and float(primary_row["log_loss_improvement"]) > 0.0
    )
    secondary_positive = bool(
        float(secondary_row["brier_improvement"]) > 0.0
        and float(secondary_row["log_loss_improvement"]) > 0.0
    )
    if not primary_positive:
        return "destination_own_history_sufficient"
    if not secondary_positive:
        return "undirected_transition_pressure_only"
    if economic_metrics.empty:
        return "activation_rotation_predictive_tradeability_unknown"
    m3_economic = economic_metrics.loc[
        economic_metrics["model_name"].eq("M3_directed_family_rotation")
    ]
    if m3_economic.empty or float(m3_economic.iloc[0]["total_net_payoff_bps"]) <= 0.0:
        return "activation_prediction_not_economically_useful"
    if not null_metrics.empty and bool(null_metrics["brier_improvement"].gt(0.0).all()):
        return "no_directed_rotation"
    if not loo_results.empty and float(loo_results["brier_improvement"].gt(0.0).mean()) < 0.5:
        return "pair_transition_sparse_or_concentrated"
    if not concentration.empty and float(concentration["absolute_contribution_share"].max()) > 0.5:
        return "pair_transition_sparse_or_concentrated"
    return "directed_family_rotation_supported_prospectively_only_required"


def artifact_manifest(output: Path) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(output.iterdir(), key=lambda value: value.name)
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _annotate(
    frame: pd.DataFrame,
    *,
    run_id: str,
    contract_hash: str,
    data_snapshot: str,
) -> pd.DataFrame:
    result = frame.copy()
    for name, value in (
        ("experiment_run_id", run_id),
        ("contract_hash", contract_hash),
        ("data_snapshot_id", data_snapshot),
        ("experiment_model_version", MODEL_VERSION),
    ):
        if name not in result:
            result.insert(0, name, value)
    return result


def load_frozen_surfaces(
    contract: Mapping[str, Any],
) -> dict[str, pd.DataFrame | FamilyTaxonomy]:
    paths = input_paths(contract)
    taxonomy = FamilyTaxonomy.from_json(paths["family_mapping"])
    all_pair_forecasts = pd.read_parquet(paths["pair_states"])
    pair_states = all_pair_forecasts.copy()
    pair_states = pair_states.loc[
        pair_states["model_name"].eq(contract["economic_state"]["pair_source_model"])
        & pair_states["horizon"].eq(int(contract["frozen_lineage"]["fixed_horizon_bars"]))
    ].copy()
    pair_states = pair_states.sort_values(
        ["period", "score_session", "loop_id", "orientation"], kind="stable"
    ).reset_index(drop=True)
    if pair_states.empty:
        raise AssertionError("frozen no-leading-feature pair-state population is empty")
    if pair_states["prediction_frozen_at"].gt(pair_states["decision_timestamp"]).any():
        raise AssertionError("pair forecast was not frozen by its decision timestamp")
    session_panel = pd.read_parquet(paths["session_panel"])
    episode_diagnostics = pd.read_parquet(paths["episode_diagnostics"])
    trade_decisions = pd.read_parquet(paths["trade_decisions"])
    no_filter = trade_decisions.loc[
        trade_decisions["model_name"].eq("no_payoff_state_filter")
        & trade_decisions["horizon"].eq(24)
    ].copy()
    if no_filter.empty:
        raise AssertionError("no-filter opportunity comparator is empty")
    no_filter = taxonomy.map_pairs(no_filter)
    no_filter = no_filter.loc[no_filter["family_mapping_status"].eq("mapped")].copy()
    if no_filter["opportunity_id"].duplicated().any():
        raise AssertionError("base opportunity population is not immutable/unique")
    return {
        "taxonomy": taxonomy,
        "all_pair_forecasts": all_pair_forecasts,
        "pair_states": pair_states,
        "session_panel": session_panel,
        "episode_diagnostics": episode_diagnostics,
        "trade_decisions": no_filter,
        "one_bar_delay": pd.read_parquet(paths["one_bar_delay"]),
    }


def build_core_state_targets(
    *,
    pair_states: pd.DataFrame,
    session_panel: pd.DataFrame,
    pair_episodes: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
) -> dict[str, pd.DataFrame]:
    family_states = aggregate_family_states(pair_states, taxonomy)
    source_events = derive_source_events(family_states)
    calendar = (
        family_states[["period", "score_session", "forecast_freeze_timestamp"]]
        .drop_duplicates(["period", "score_session"])
        .sort_values(["period", "score_session"], kind="stable")
        .reset_index(drop=True)
    )
    if calendar.duplicated(["period", "score_session"]).any():
        raise AssertionError("session calendar is not unique")
    support = build_family_payoff_support(session_panel, taxonomy)
    episodes = build_family_episode_intervals(
        pair_episodes,
        taxonomy,
        calendar,
        support,
    )
    targets = build_activation_targets(
        forecast_states=family_states,
        calendar=calendar,
        episode_intervals=episodes,
        payoff_support=support,
        registration=ActivationRegistration(),
    )
    if targets.loc[targets["target_available"], "activation_target"].isna().any():
        raise AssertionError("available targets contain missing activation labels")
    return {
        "family_states": family_states,
        "source_events": source_events,
        "calendar": calendar,
        "family_payoff_support": support,
        "family_episodes": episodes,
        "targets": targets,
    }


def prequential_settings(
    contract: Mapping[str, Any],
    *,
    run_id: str,
    window: int,
    pooling_strength: float | None = None,
) -> PrequentialSettings:
    graph = contract["directed_graph"]
    model = contract["models"]
    rule = contract["prediction_rule"]
    return PrequentialSettings(
        run_id=run_id,
        target_window_sessions=window,
        learning_rate=float(model["learning_rate"]),
        ridge_penalty=float(model["ridge_penalty"]),
        feature_clip=float(model["feature_clip"]),
        coefficient_clip=float(model["coefficient_clip"]),
        minimum_training_rows=int(model["minimum_training_rows_per_destination"]),
        minimum_training_activations=int(model["minimum_training_activations_per_destination"]),
        minimum_lift_over_base=float(rule["minimum_predicted_lift_over_base"]),
        maximum_interval_width=float(rule["maximum_probability_interval_width"]),
        graph=GraphSettings(
            base_alpha=float(graph["beta_prior_alpha"]),
            base_beta=9.0,
            edge_alpha=float(graph["beta_prior_alpha"]),
            edge_beta=float(graph["beta_prior_beta"]),
            pooling_strength=float(
                pooling_strength
                if pooling_strength is not None
                else graph["empirical_bayes_pooling_strength"]
            ),
            minimum_source_event_sessions=int(graph["minimum_source_event_sessions"]),
            log_lift_clip=tuple(float(value) for value in graph["log_lift_clip"]),
        ),
    )


def run_registered_models(
    state: Mapping[str, pd.DataFrame],
    contract: Mapping[str, Any],
    *,
    run_id: str,
    events: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    pooling_strength: float | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window in (1, 3, 5):
        rows.append(
            run_prequential_rotation(
                state["family_states"],
                state["source_events"] if events is None else events,
                state["targets"] if targets is None else targets,
                settings=prequential_settings(
                    contract,
                    run_id=run_id,
                    window=window,
                    pooling_strength=pooling_strength,
                ),
            )
        )
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "period",
                "forecast_session",
                "target_window_sessions",
                "destination_family",
                "model_name",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _source_vector(rows: pd.DataFrame) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for row in rows.to_dict(orient="records"):
        events: set[str] = set()
        if bool(row.get("source_active", False)):
            events.add("active")
        if bool(row.get("newly_decaying", False)):
            events.add("newly_decaying")
        if bool(row.get("newly_retired", False)):
            events.add("newly_retired")
        if events:
            result[str(row["destination_family"])] = frozenset(events)
    return result


def build_past_only_graph_ledger(
    family_states: pd.DataFrame,
    source_events: pd.DataFrame,
    targets: pd.DataFrame,
    settings: GraphSettings,
) -> pd.DataFrame:
    """Replay graph snapshots without importing runner forecast calculations."""

    target = targets.loc[targets["target_window_sessions"].eq(3)].copy()
    lookup = {
        (int(str(row.period)), str(row.forecast_session), str(row.destination_family)): row
        for row in target.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for period, states in family_states.groupby("period", sort=True, observed=True):
        graph = PastOnlyRotationGraph(settings)
        pending: list[MaturedRotationExample] = []
        families = sorted(states["destination_family"].astype(str).unique())
        period_events = source_events.loc[source_events["period"].eq(period)].copy()
        for session in sorted(states["score_session"].astype(str).unique()):
            current = states.loc[states["score_session"].astype(str).eq(session)]
            freeze = pd.Timestamp(current["forecast_freeze_timestamp"].iloc[0])
            graph.update_matured(pending, as_of=freeze)
            snapshot = graph.rows(families)
            snapshot.insert(0, "graph_as_of_session", session)
            snapshot.insert(0, "graph_freeze_timestamp", freeze)
            snapshot.insert(0, "period", int(str(period)))
            snapshot["latest_label_availability_timestamp"] = graph.latest_label_availability
            records.extend(snapshot.to_dict(orient="records"))
            current_events = period_events.loc[
                period_events["score_session"].astype(str).eq(session)
            ]
            source_vector = _source_vector(current_events)
            for family in families:
                row = lookup.get((int(str(period)), session, family))
                if row is None or not bool(row.target_available):
                    continue
                pending.append(
                    MaturedRotationExample(
                        example_id=f"graph-{stable_hash((period, session, family, 3))}",
                        period=int(str(period)),
                        forecast_session=session,
                        destination_family=family,
                        activation_target=bool(row.activation_target),
                        label_availability_timestamp=pd.Timestamp(row.label_availability_timestamp),
                        source_events=source_vector,
                    )
                )
    return pd.DataFrame.from_records(records)


def comparator_forecasts(
    all_pair_forecasts: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    target_lookup = targets.loc[targets["target_window_sessions"].isin([1, 3, 5])].copy()
    outputs: list[pd.DataFrame] = []
    for source_model, name in (
        ("payoff_only_change_point", "M5_payoff_only_BOCPD"),
        ("v1_60_session_selector", "M6_v1_60_session_selector"),
    ):
        selected = all_pair_forecasts.loc[
            all_pair_forecasts["model_name"].eq(source_model) & all_pair_forecasts["horizon"].eq(24)
        ].copy()
        if selected.empty:
            continue
        family = aggregate_family_states(selected, taxonomy)
        for window in (1, 3, 5):
            frame = family.merge(
                target_lookup.loc[target_lookup["target_window_sessions"].eq(window)],
                left_on=["period", "score_session", "destination_family"],
                right_on=["period", "forecast_session", "destination_family"],
                how="left",
                validate="one_to_one",
                suffixes=("", "_target"),
            )
            one_step = np.clip(frame["max_p_on_next"].fillna(0.5).to_numpy(float), 1e-6, 1 - 1e-6)
            frame["predicted_activation_probability"] = 1.0 - np.power(1.0 - one_step, window)
            frame["activation_base_rate"] = np.nan
            frame["prediction_state"] = np.where(
                frame["operational_state"].eq("active"), "abstain", "nominated"
            )
            frame["model_name"] = name
            frame["target_window_sessions"] = window
            frame["forecast_session"] = frame["score_session"]
            frame["forecast_id"] = [
                f"comparator-{stable_hash((name, period, session, family_name, window))}"
                for period, session, family_name in frame[
                    ["period", "score_session", "destination_family"]
                ].itertuples(index=False, name=None)
            ]
            frame["forecast_timestamp"] = frame["forecast_freeze_timestamp"]
            frame["reason_codes"] = np.where(
                frame["operational_state"].eq("active"), "destination_already_active", ""
            )
            outputs.append(frame)
    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()


def build_pair_targets(
    pair_states: pd.DataFrame,
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    state = pair_states.copy()
    state["destination_family"] = (
        state["loop_id"].astype(str) + "|" + state["orientation"].astype(str)
    )
    state = state.rename(
        columns={
            "edge_state": "operational_state",
            "prediction_frozen_at": "forecast_freeze_timestamp",
        }
    )
    state = state[
        [
            "period",
            "score_session",
            "destination_family",
            "operational_state",
            "forecast_freeze_timestamp",
        ]
    ]
    pair_support = panel.copy()
    pair_support["destination_family"] = (
        pair_support["loop_id"].astype(str) + "|" + pair_support["orientation"].astype(str)
    )
    pair_support = pair_support.rename(columns={"session": "session"})
    pair_episodes = episodes.copy()
    pair_episodes["destination_family"] = (
        pair_episodes["loop_id"].astype(str) + "|" + pair_episodes["orientation"].astype(str)
    )
    pair_episodes = pair_episodes.loc[
        pair_episodes["hindsight_estimated_onset"].notna()
        & pair_episodes["hindsight_estimated_end"].notna()
    ].copy()
    pair_episodes["episode_onset_session"] = pair_episodes["hindsight_estimated_onset"].astype(str)
    pair_episodes["episode_end_session"] = pair_episodes["hindsight_estimated_end"].astype(str)
    pair_episodes["label_availability_timestamp"] = pd.to_datetime(
        pair_episodes["episode_end_session"] + "T23:59:59Z", utc=True
    )
    return build_activation_targets(
        forecast_states=state,
        calendar=calendar,
        episode_intervals=pair_episodes[
            [
                "period",
                "destination_family",
                "episode_id",
                "episode_onset_session",
                "episode_end_session",
                "label_availability_timestamp",
            ]
        ],
        payoff_support=pair_support[
            [
                "period",
                "session",
                "destination_family",
                "data_availability_timestamp",
            ]
        ],
        registration=ActivationRegistration(),
    )


def run_pair_refinement(
    family_forecasts: pd.DataFrame,
    pair_states: pd.DataFrame,
    pair_targets: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
    *,
    run_id: str,
    pooling_strength: float = 20.0,
) -> pd.DataFrame:
    family = family_forecasts.loc[
        family_forecasts["model_name"].eq("M3_directed_family_rotation")
        & family_forecasts["target_window_sessions"].eq(3)
    ].copy()
    state = taxonomy.map_pairs(pair_states)
    state["destination_pair"] = (
        state["loop_id"].astype(str) + "|" + state["orientation"].astype(str)
    )
    target_lookup = {
        (int(str(row.period)), str(row.forecast_session), str(row.destination_family)): row
        for row in pair_targets.loc[pair_targets["target_window_sessions"].eq(3)].itertuples(
            index=False
        )
    }
    records: list[dict[str, object]] = []
    for period, period_states in state.groupby("period", sort=True, observed=True):
        counts: dict[str, list[int]] = {}
        pending: list[dict[str, object]] = []
        for session in sorted(period_states["score_session"].astype(str).unique()):
            current = period_states.loc[
                period_states["score_session"].astype(str).eq(session)
            ].sort_values(["loop_id", "orientation"], kind="stable")
            freeze = pd.Timestamp(current["prediction_frozen_at"].iloc[0])
            for example in pending:
                if (
                    bool(example["trained"])
                    or pd.Timestamp(example["label_availability_timestamp"]) >= freeze
                ):
                    continue
                pair = str(example["destination_pair"])
                pair_counts = counts.setdefault(pair, [0, 0])
                pair_counts[0] += 1
                pair_counts[1] += int(bool(example["activation_target"]))
                example["trained"] = True
            for row in current.to_dict(orient="records"):
                destination_pair = str(row["destination_pair"])
                destination_family = str(row["destination_family"])
                family_row = family.loc[
                    family["period"].eq(period)
                    & family["forecast_session"].astype(str).eq(session)
                    & family["destination_family"].eq(destination_family)
                ]
                if family_row.empty:
                    continue
                family_record = family_row.iloc[0]
                support, activations = counts.get(destination_pair, [0, 0])
                probability = shrink_pair_probability(
                    pair_activations=activations,
                    pair_support=support,
                    family_probability=float(family_record["predicted_activation_probability"]),
                    pooling_strength=pooling_strength,
                )
                target = target_lookup.get((int(str(period)), session, destination_pair))
                target_available = bool(target.target_available) if target is not None else False
                status = (
                    "supported" if support >= 20 and activations >= 4 else "unknown_pair_effect"
                )
                records.append(
                    {
                        "run_id": run_id,
                        "forecast_id": f"pair-forecast-{stable_hash((run_id, period, session, destination_pair))}",
                        "period": int(str(period)),
                        "forecast_session": session,
                        "forecast_timestamp": freeze,
                        "forecast_freeze_timestamp": freeze,
                        "target_window_sessions": 3,
                        "destination_family": destination_family,
                        "destination_pair": destination_pair,
                        "loop_id": row["loop_id"],
                        "orientation": row["orientation"],
                        "model_name": "M4_directed_pair_rotation",
                        "predicted_activation_probability": probability,
                        "family_probability": float(
                            family_record["predicted_activation_probability"]
                        ),
                        "pair_training_rows": support,
                        "pair_training_activations": activations,
                        "pair_support_status": status,
                        "prediction_state": (
                            "nominated"
                            if status == "supported"
                            and str(row["edge_state"]) != "active"
                            and probability >= float(family_record["activation_base_rate"]) * 1.25
                            else "abstain"
                        ),
                        "target_available": target_available,
                        "activation_target": (
                            bool(target.activation_target)
                            if target is not None and target_available
                            else pd.NA
                        ),
                        "target_status": target.target_status if target is not None else "missing",
                        "label_availability_timestamp": (
                            target.label_availability_timestamp if target is not None else pd.NaT
                        ),
                        "target_episode_ids": (
                            target.target_episode_ids if target is not None else ""
                        ),
                    }
                )
                if target is not None and target_available:
                    pending.append(
                        {
                            "destination_pair": destination_pair,
                            "activation_target": bool(target.activation_target),
                            "label_availability_timestamp": pd.Timestamp(
                                target.label_availability_timestamp
                            ),
                            "trained": False,
                        }
                    )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["activation_target"] = result["activation_target"].astype("boolean")
    return result


def build_metrics(
    scored_forecasts: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []
    system_rows: list[dict[str, object]] = []
    models = sorted(scored_forecasts["model_name"].astype(str).unique())
    for model_name in models:
        for window in sorted(
            scored_forecasts.loc[
                scored_forecasts["model_name"].eq(model_name),
                "target_window_sessions",
            ]
            .dropna()
            .unique()
        ):
            subset = scored_forecasts.loc[
                scored_forecasts["model_name"].eq(model_name)
                & scored_forecasts["target_window_sessions"].eq(window)
            ]
            for period_slice, frame in [
                ("all", subset),
                *[
                    (str(period), subset.loc[subset["period"].eq(period)])
                    for period in sorted(subset["period"].unique())
                ],
            ]:
                values = activation_metrics(frame)
                metric_rows.append(
                    {
                        "model_name": model_name,
                        "target_window_sessions": int(window),
                        "period_slice": period_slice,
                        **values,
                    }
                )
            table = calibration_table(
                subset,
                bins=int(contract["evaluation"]["calibration_bins"]),
            )
            table.insert(0, "target_window_sessions", int(window))
            table.insert(0, "model_name", model_name)
            calibration_rows.append(table)
            if {
                "probability_no_activation",
                "probability_multiple_activation",
            }.issubset(subset.columns):
                system_rows.append(
                    {
                        "model_name": model_name,
                        "target_window_sessions": int(window),
                        **system_activation_metrics(subset),
                    }
                )
    comparison_rows: list[dict[str, object]] = []
    for window in (1, 3, 5):
        window_frame = scored_forecasts.loc[scored_forecasts["target_window_sessions"].eq(window)]
        for treatment, control, label in (
            (
                "M3_directed_family_rotation",
                "M1_destination_own_history",
                "M3_vs_M1",
            ),
            (
                "M3_directed_family_rotation",
                "M2_undirected_system_state",
                "M3_vs_M2",
            ),
        ):
            comparison_rows.append(
                {
                    "comparison": label,
                    "target_window_sessions": window,
                    "period_slice": "all",
                    **paired_model_comparison(
                        window_frame,
                        treatment=treatment,
                        control=control,
                        bootstrap_resamples=int(contract["evaluation"]["bootstrap_resamples"]),
                        seed=int(contract["evaluation"]["bootstrap_seed"]),
                    ),
                }
            )
            for period in sorted(window_frame["period"].unique()):
                comparison_rows.append(
                    {
                        "comparison": label,
                        "target_window_sessions": window,
                        "period_slice": str(period),
                        **paired_model_comparison(
                            window_frame.loc[window_frame["period"].eq(period)],
                            treatment=treatment,
                            control=control,
                            bootstrap_resamples=500,
                            seed=int(contract["evaluation"]["bootstrap_seed"]),
                        ),
                    }
                )
    primary = scored_forecasts.loc[
        scored_forecasts["target_window_sessions"].eq(3)
        & scored_forecasts["model_name"].eq("M3_directed_family_rotation")
        & scored_forecasts["target_available"]
    ].copy()
    ranking_rows: list[dict[str, object]] = []
    for (period, session), group in primary.groupby(
        ["period", "forecast_session"], sort=True, observed=True
    ):
        ranked = group.sort_values(
            "predicted_activation_probability", ascending=False, kind="stable"
        ).reset_index(drop=True)
        positives = np.flatnonzero(ranked["activation_target"].fillna(False).to_numpy(bool))
        ranking_rows.append(
            {
                "period": int(str(period)),
                "forecast_session": str(session),
                "observed_activations": int(len(positives)),
                "top_one_correct": bool(len(positives) and positives[0] == 0),
                "top_three_recall": float(np.sum(positives < 3) / len(positives))
                if len(positives)
                else math.nan,
                "mean_reciprocal_rank": float(1.0 / (positives[0] + 1)) if len(positives) else 0.0,
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(calibration_rows, ignore_index=True),
        pd.DataFrame(comparison_rows),
        pd.concat(
            [pd.DataFrame(system_rows), pd.DataFrame(ranking_rows)],
            ignore_index=True,
            sort=False,
        ),
    )


def build_activation_timing(
    scored_forecasts: pd.DataFrame,
    family_episodes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    index = {
        (int(str(period)), str(session)): item
        for period, group in calendar.groupby("period", sort=True, observed=True)
        for item, session in enumerate(
            group["score_session"].astype(str).drop_duplicates().sort_values()
        )
    }
    episode_lookup = family_episodes.set_index("episode_id", drop=False).to_dict("index")
    rows: list[dict[str, object]] = []
    selected = scored_forecasts.loc[
        scored_forecasts["target_window_sessions"].eq(3)
        & scored_forecasts["prediction_state"].eq("nominated")
    ]
    for record in selected.to_dict(orient="records"):
        episode_ids = [
            value for value in str(record.get("target_episode_ids", "")).split("|") if value
        ]
        if not episode_ids:
            rows.append(
                {
                    "forecast_id": record["forecast_id"],
                    "model_name": record["model_name"],
                    "period": record["period"],
                    "forecast_session": record["forecast_session"],
                    "destination_family": record["destination_family"],
                    "activation_observed": False,
                    "episode_id": pd.NA,
                    "sessions_to_activation": np.nan,
                    "episode_duration_sessions": np.nan,
                    "episode_payoff_available_after_forecast_bps": np.nan,
                    "fraction_episode_remaining": np.nan,
                }
            )
            continue
        for episode_id in episode_ids:
            episode = episode_lookup.get(episode_id)
            if episode is None:
                continue
            period = int(str(record["period"]))
            forecast_index = index[(period, str(record["forecast_session"]))]
            onset_index = index[(period, str(episode["episode_onset_session"]))]
            lead = onset_index - forecast_index
            rows.append(
                {
                    "forecast_id": record["forecast_id"],
                    "model_name": record["model_name"],
                    "period": period,
                    "forecast_session": record["forecast_session"],
                    "destination_family": record["destination_family"],
                    "activation_observed": True,
                    "episode_id": episode_id,
                    "sessions_to_activation": lead,
                    "episode_duration_sessions": episode["duration_sessions"],
                    "episode_payoff_available_after_forecast_bps": episode[
                        "total_episode_payoff_bps"
                    ],
                    "fraction_episode_remaining": 1.0 if lead > 0 else 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def build_economic_translation(
    scored_forecasts: pd.DataFrame,
    opportunities: pd.DataFrame,
    calendar: pd.DataFrame,
    one_bar_delay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_models = [
        "M0_activation_base_rate",
        "M1_destination_own_history",
        "M2_undirected_system_state",
        "M3_directed_family_rotation",
    ]
    selected = scored_forecasts.loc[
        scored_forecasts["target_window_sessions"].eq(3)
        & scored_forecasts["model_name"].isin(prediction_models)
    ].copy()
    translation = translate_predictions_to_opportunities(selected, opportunities, calendar)
    forecast_context = selected[
        [
            "forecast_id",
            "source_family_state_vector_json",
            "target_episode_ids",
            "activation_target",
            "target_status",
        ]
    ].drop_duplicates("forecast_id")
    translation = translation.merge(
        forecast_context,
        on="forecast_id",
        how="left",
        validate="many_to_one",
    )
    delay = one_bar_delay.loc[one_bar_delay["checkpoint_type"].eq("fixed_bar_1")].copy()
    delay = delay.sort_values(
        ["opportunity_id", "checkpoint_timestamp"], kind="stable"
    ).drop_duplicates("opportunity_id")
    translation = translation.merge(
        delay[
            [
                "opportunity_id",
                "outcome_status",
                "one_bar_delay_constant_terminal_net_bps",
            ]
        ],
        on="opportunity_id",
        how="left",
        validate="many_to_one",
    )
    translation["one_bar_delay_available"] = translation[
        "one_bar_delay_constant_terminal_net_bps"
    ].notna()
    stressed = apply_cost_stress(
        translation.loc[
            translation["economic_translation_status"].eq("eligible_opportunity")
        ].copy(),
        multiplier=2.0,
    )
    metric_rows: list[dict[str, object]] = []
    for model_name, group in translation.groupby("model_name", sort=True, observed=True):
        eligible = group.loc[group["economic_translation_status"].eq("eligible_opportunity")]
        payoff = pd.to_numeric(eligible["primary_net_payoff_bps"], errors="coerce")
        gross = pd.to_numeric(eligible["gross_payoff_bps"], errors="coerce")
        costs = pd.to_numeric(eligible["primary_total_cost_bps"], errors="coerce")
        delayed = pd.to_numeric(
            eligible["one_bar_delay_constant_terminal_net_bps"], errors="coerce"
        ).dropna()
        twice = stressed.loc[stressed["model_name"].eq(model_name)]
        metric_rows.append(
            {
                "model_name": str(model_name),
                "nominated_forecasts": int(group["forecast_id"].nunique()),
                "eligible_opportunities": int(len(eligible)),
                "no_tradeable_opportunity": int(
                    group["economic_translation_status"]
                    .eq("no_tradeable_destination_opportunity")
                    .sum()
                ),
                "total_gross_payoff_bps": float(gross.sum()),
                "total_cost_bps": float(costs.sum()),
                "total_net_payoff_bps": float(payoff.sum()),
                "mean_net_payoff_bps": float(payoff.mean()),
                "hit_rate": float(payoff.gt(0.0).mean()),
                "twice_cost_net_payoff_bps": float(twice["stressed_net_payoff_bps"].sum()),
                "one_bar_delay_observations": int(len(delayed)),
                "one_bar_delay_net_payoff_bps": float(delayed.sum()),
                "entry_exit_overlap_cost_rules_changed": False,
                "replacement_opportunity_used": False,
                "capacity_refill_used": False,
            }
        )
    return translation, pd.DataFrame(metric_rows), stressed


def build_concentration(translation: pd.DataFrame) -> pd.DataFrame:
    eligible = translation.loc[
        translation["model_name"].eq("M3_directed_family_rotation")
        & translation["economic_translation_status"].eq("eligible_opportunity")
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "key",
                "net_contribution_bps",
                "absolute_contribution_share",
                "rank",
            ]
        )
    eligible["source_edge_key"] = eligible["source_family_state_vector_json"].fillna("{}")
    eligible["episode_key"] = eligible["target_episode_ids"].replace("", "no_episode")
    dimensions = {
        "destination_family": "destination_family",
        "stock": "stock_id",
        "period": "period",
        "month": "month",
        "episode": "episode_key",
        "source_state_vector": "source_edge_key",
        "pair": "loop_id",
    }
    rows: list[dict[str, object]] = []
    for dimension, column in dimensions.items():
        contribution = (
            eligible.groupby(column, observed=True)["primary_net_payoff_bps"]
            .sum()
            .sort_values(ascending=False)
        )
        denominator = float(contribution.abs().sum())
        hhi = float(np.square(contribution.abs() / denominator).sum()) if denominator else 0.0
        for rank, (key, value) in enumerate(contribution.items(), start=1):
            rows.append(
                {
                    "dimension": dimension,
                    "key": str(key),
                    "net_contribution_bps": float(value),
                    "absolute_contribution_share": abs(float(value)) / denominator
                    if denominator
                    else 0.0,
                    "rank": rank,
                    "herfindahl": hhi,
                }
            )
    return pd.DataFrame.from_records(rows)


def build_deletion_stresses(
    translation: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    eligible = translation.loc[
        translation["model_name"].eq("M3_directed_family_rotation")
        & translation["economic_translation_status"].eq("eligible_opportunity")
    ].copy()
    rows: list[dict[str, object]] = []
    for dimension, column in (("stock", "stock_id"), ("episode", "target_episode_ids")):
        ranked = concentration.loc[concentration["dimension"].eq(dimension)].sort_values(
            "rank", kind="stable"
        )
        for count in (1, 5):
            removed = set(ranked.head(count)["key"].astype(str))
            subset = eligible.loc[~eligible[column].astype(str).isin(removed)]
            rows.append(
                {
                    "stress_test": f"remove_top_{count}_{dimension}",
                    "removed_keys": "|".join(sorted(removed)),
                    "trade_count": int(len(subset)),
                    "net_payoff_bps": float(subset["primary_net_payoff_bps"].sum()),
                    "all_stock_dependent_states_rebuilt": False,
                    "status": "economic_attribution_only",
                }
            )
    for period, subset in eligible.groupby("period", sort=True, observed=True):
        rows.append(
            {
                "stress_test": f"period_{int(str(period))}",
                "removed_keys": "",
                "trade_count": int(len(subset)),
                "net_payoff_bps": float(subset["primary_net_payoff_bps"].sum()),
                "all_stock_dependent_states_rebuilt": False,
                "status": "period_slice",
            }
        )
    return pd.DataFrame.from_records(rows)


def permute_destination_targets(targets: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    result = targets.copy()
    rng = np.random.default_rng(seed)
    for (_, _window), index in result.groupby(
        ["period", "target_window_sessions"], sort=True, observed=True
    ).groups.items():
        positions = list(index)
        available = [
            position for position in positions if bool(result.at[position, "target_available"])
        ]
        values = result.loc[available, "activation_target"].astype(bool).to_numpy()
        result.loc[available, "activation_target"] = rng.permutation(values)
    keys = ["period", "forecast_session", "target_window_sessions"]
    positives = result["target_available"] & result["activation_target"].fillna(False)
    result["observed_activation_count"] = (
        positives.astype(int).groupby([result[key] for key in keys], sort=False).transform("sum")
    )
    result["multiple_activation_flag"] = result["observed_activation_count"].gt(1)
    available_count = (
        result["target_available"]
        .astype(int)
        .groupby([result[key] for key in keys], sort=False)
        .transform("sum")
    )
    result["no_activation_flag"] = available_count.gt(0) & result["observed_activation_count"].eq(0)
    return result


def build_null_results(
    state: Mapping[str, pd.DataFrame],
    contract: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {
        "wrong_lag_10": (
            shift_source_events(state["source_events"], sessions=10),
            state["targets"],
        ),
        "source_permutation": (
            permute_source_events(state["source_events"], seed=int(contract["random_seed"])),
            state["targets"],
        ),
        "destination_label_permutation": (
            state["source_events"],
            permute_destination_targets(state["targets"], seed=int(contract["random_seed"])),
        ),
    }
    rows: list[dict[str, object]] = []
    forecasts: list[pd.DataFrame] = []
    for label, (events, targets) in variants.items():
        scored = run_prequential_rotation(
            state["family_states"],
            events,
            targets,
            settings=prequential_settings(contract, run_id=f"{run_id}-{label}", window=3),
        )
        scored["null_test"] = label
        forecasts.append(scored)
        comparison = paired_model_comparison(
            scored,
            treatment="M3_directed_family_rotation",
            control="M1_destination_own_history",
            bootstrap_resamples=500,
            seed=int(contract["random_seed"]),
        )
        rows.append({"null_test": label, **comparison})
    return pd.DataFrame(rows), pd.concat(forecasts, ignore_index=True)


def load_v2_rebuilder(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_rotation_v2_rebuilder", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_rebuild_context(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    paths = input_paths(contract)
    v2 = load_v2_rebuilder(paths["v2_rebuild_runner"])
    config = v2.load_config()
    ledger, sessions, _, _ = v2.load_recovered_v1_analysis(config)
    calendars = v2.build_session_calendars(sessions)
    surface = v2.build_trade_surface(ledger, config)
    required_features = tuple(
        name
        for name in config["features"]["leading_feature_logit_weights"]
        if name != "out_of_distribution_score"
    )
    return {
        "v2": v2,
        "config": config,
        "configuration_hash": sha256(paths["v2_rebuild_config"]),
        "calendars": calendars,
        "surface": surface,
        "required_features": required_features,
    }


def derive_pair_episodes_from_panel(
    panel: pd.DataFrame,
    calendars: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    warmup = int(config["support"]["warmup_completed_sessions"])
    neutral_band = float(config["evaluation"]["episode_neutral_band_bps"])
    smooth = int(config["evaluation"]["episode_smoothing_sessions"])
    calendar_index = pd.concat(calendars.values(), ignore_index=True)[
        ["period", "score_session", "session_index"]
    ]
    source = panel.rename(columns={"session": "score_session"}).merge(
        calendar_index,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    source = source.loc[source["session_index"].ge(warmup)].copy()
    rows: list[dict[str, object]] = []
    counter = 0
    for key, group in source.groupby(
        ["period", "loop_id", "orientation", "horizon"],
        sort=True,
        observed=True,
    ):
        group = group.sort_values("session_index", kind="stable").copy()
        smoothed = (
            group["robust_net_payoff_bps"]
            .rolling(smooth, center=True, min_periods=2)
            .mean()
            .to_numpy(float)
        )
        labels = np.where(
            smoothed > neutral_band,
            "positive",
            np.where(smoothed < -neutral_band, "negative", "neutral"),
        ).astype(object)
        for position in np.flatnonzero(labels == "positive"):
            future = smoothed[position + 1 : position + 4]
            if (
                len(future)
                and np.nanmean(future) < smoothed[position]
                and (future <= neutral_band).any()
            ):
                labels[position] = "decaying"
        mask = np.isin(labels, ["positive", "decaying"])
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        for start, end in zip(starts, ends, strict=True):
            segment = group.iloc[start : end + 1]
            if len(segment) < 2:
                continue
            counter += 1
            rows.append(
                {
                    "episode_id": f"rebuilt_episode_{counter:04d}",
                    "loop_id": str(key[1]),
                    "orientation": str(key[2]),
                    "period": int(str(key[0])),
                    "horizon": int(str(key[3])),
                    "hindsight_estimated_onset": str(segment["score_session"].iloc[0]),
                    "hindsight_estimated_end": str(segment["score_session"].iloc[-1]),
                    "duration_sessions": int(
                        segment["session_index"].iloc[-1] - segment["session_index"].iloc[0] + 1
                    ),
                    "observed_payoff_sessions": int(len(segment)),
                    "mean_session_payoff_bps": float(segment["robust_net_payoff_bps"].mean()),
                    "total_episode_payoff_bps": float(segment["robust_net_payoff_bps"].sum()),
                }
            )
    return pd.DataFrame.from_records(rows)


def rebuild_rotation_inputs(
    context: Mapping[str, Any],
    taxonomy: FamilyTaxonomy,
    contract: Mapping[str, Any],
    *,
    excluded_stocks: set[str] | None = None,
    aggregation: str = "winsorised_mean",
    run_id: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    v2 = context["v2"]
    config = context["config"]
    surface = context["surface"].copy()
    excluded = set(excluded_stocks or ())
    if excluded:
        surface = surface.loc[~surface["symbol_norm"].astype(str).isin(excluded)].copy()
        surface = v2.rebuild_surface_context_for_universe(
            surface,
            universe_size=int(surface["symbol_norm"].nunique()),
        )
    primary_panel, median_panel = v2.aggregate_payoff_panels(surface, config)
    panel = median_panel if aggregation == "median" else primary_panel
    cell_keys = {
        int(period): sorted(
            {
                (str(row.loop_id), str(row.orientation), int(row.horizon))
                for row in surface.loc[surface["period"].eq(int(period))].itertuples(index=False)
            }
        )
        for period in config["evaluation"]["periods"]
    }
    features = v2.build_feature_panel(
        surface,
        panel,
        context["calendars"],
        cell_keys,
        context["required_features"],
    )
    pair_states = v2.run_change_point_model(
        model_name="hierarchical_payoff_history_change_point",
        config=config,
        configuration_hash=context["configuration_hash"],
        run_id=run_id,
        calendars=context["calendars"],
        payoff_panel=panel,
        feature_panel=features,
        cell_keys_by_period=cell_keys,
        enable_hierarchy=True,
        include_leading_features=False,
    )
    pair_episodes = derive_pair_episodes_from_panel(panel, context["calendars"], config)
    state = build_core_state_targets(
        pair_states=pair_states,
        session_panel=panel,
        pair_episodes=pair_episodes,
        taxonomy=taxonomy,
    )
    forecasts = run_prequential_rotation(
        state["family_states"],
        state["source_events"],
        state["targets"],
        settings=prequential_settings(contract, run_id=run_id, window=3),
    )
    detail = {
        "excluded_stocks": sorted(excluded),
        "aggregation": aggregation,
        "surface_rows": int(len(surface)),
        "surface_stocks": int(surface["symbol_norm"].nunique()),
        "session_panel_rows": int(len(panel)),
        "pair_state_rows": int(len(pair_states)),
        "family_state_rows": int(len(state["family_states"])),
        "all_stock_dependent_states_rebuilt": True,
        "family_aggregates_rebuilt": True,
        "transition_graph_rebuilt": True,
        "activation_labels_rebuilt": True,
    }
    return state, forecasts, detail


def run_rebuilt_stresses(
    contract: Mapping[str, Any],
    taxonomy: FamilyTaxonomy,
    *,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = prepare_rebuild_context(contract)
    stress_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    _, median_forecasts, detail = rebuild_rotation_inputs(
        context,
        taxonomy,
        contract,
        aggregation="median",
        run_id=f"{run_id}-median",
    )
    median_comparison = paired_model_comparison(
        median_forecasts,
        treatment="M3_directed_family_rotation",
        control="M1_destination_own_history",
        bootstrap_resamples=500,
        seed=int(contract["random_seed"]),
    )
    stress_rows.append(
        {
            "stress_test": "median_economic_aggregation_fully_rebuilt",
            **median_comparison,
            "rebuild_detail_json": json.dumps(detail, sort_keys=True, separators=(",", ":")),
        }
    )
    symbols = sorted(context["surface"]["symbol_norm"].astype(str).unique())
    for index, stock in enumerate(symbols, start=1):
        _, forecasts, detail = rebuild_rotation_inputs(
            context,
            taxonomy,
            contract,
            excluded_stocks={stock},
            run_id=f"{run_id}-loo-{stock}",
        )
        comparison = paired_model_comparison(
            forecasts,
            treatment="M3_directed_family_rotation",
            control="M1_destination_own_history",
            bootstrap_resamples=300,
            seed=int(contract["random_seed"]),
        )
        loo_rows.append(
            {
                "excluded_stock": stock,
                **comparison,
                **detail,
            }
        )
        print(
            json.dumps(
                {
                    "rebuild": "leave_one_stock_out",
                    "stock": stock,
                    "index": index,
                    "total": len(symbols),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    loo = pd.DataFrame.from_records(loo_rows)
    stress_rows.append(
        {
            "stress_test": "leave_one_stock_out_fully_rebuilt",
            "paired_rows": int(loo["paired_rows"].min()) if not loo.empty else 0,
            "brier_improvement": float(loo["brier_improvement"].mean())
            if not loo.empty
            else math.nan,
            "log_loss_improvement": float(loo["log_loss_improvement"].mean())
            if not loo.empty
            else math.nan,
            "directionally_positive_brier_fraction": float(loo["brier_improvement"].gt(0.0).mean())
            if not loo.empty
            else math.nan,
            "rebuild_detail_json": json.dumps(
                {"stocks": len(symbols), "all_stock_dependent_states_rebuilt": True},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return pd.DataFrame.from_records(stress_rows), loo


def build_supported_pair_graph(
    pair_states: pd.DataFrame,
    pair_targets: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
) -> pd.DataFrame:
    states = taxonomy.map_pairs(pair_states).sort_values(
        ["period", "loop_id", "orientation", "score_session"], kind="stable"
    )
    states["pair_id"] = states["loop_id"].astype(str) + "|" + states["orientation"].astype(str)
    keys = ["period", "pair_id"]
    states["prior_state"] = states.groupby(keys, sort=False, observed=True)["edge_state"].shift()
    states["active"] = states["edge_state"].eq("active")
    states["newly_decaying"] = states["edge_state"].eq("decaying") & states["prior_state"].eq(
        "active"
    )
    states["newly_retired"] = states["edge_state"].eq("retired") & states["prior_state"].isin(
        ["active", "decaying"]
    )
    target = pair_targets.loc[
        pair_targets["target_window_sessions"].eq(3) & pair_targets["target_available"]
    ].copy()
    counts: dict[tuple[int, str, str, str], list[int]] = {}
    for destination in target.to_dict(orient="records"):
        period = int(destination["period"])
        session = str(destination["forecast_session"])
        destination_pair = str(destination["destination_family"])
        source_rows = states.loc[
            states["period"].eq(period) & states["score_session"].astype(str).eq(session)
        ]
        for source in source_rows.to_dict(orient="records"):
            source_pair = str(source["pair_id"])
            if source_pair == destination_pair:
                continue
            for event in ("active", "newly_decaying", "newly_retired"):
                if not bool(source[event]):
                    continue
                value = counts.setdefault((period, source_pair, event, destination_pair), [0, 0])
                value[0] += 1
                value[1] += int(bool(destination["activation_target"]))
    rows: list[dict[str, object]] = []
    for (period, source_pair, event, destination_pair), (support, activations) in sorted(
        counts.items()
    ):
        source_loop, source_orientation = source_pair.split("|", maxsplit=1)
        destination_loop, destination_orientation = destination_pair.split("|", maxsplit=1)
        source_family = taxonomy.family_for(source_loop, source_orientation)
        destination_family = taxonomy.family_for(destination_loop, destination_orientation)
        destination_rows = target.loc[
            target["period"].eq(period) & target["destination_family"].eq(destination_pair)
        ]
        base_support = int(len(destination_rows))
        base_activations = int(destination_rows["activation_target"].fillna(False).sum())
        base_rate = (base_activations + 1.0) / (base_support + 10.0)
        raw_rate = (activations + 1.0) / (support + 2.0)
        raw_lift = raw_rate / max(base_rate, 1e-12)
        weight = support / (support + 20.0)
        shrunk_lift = math.exp(
            weight * float(np.clip(math.log(raw_lift), -math.log(4), math.log(4)))
        )
        supported = support >= 20 and activations >= 4
        rows.append(
            {
                "period": period,
                "source_pair": source_pair,
                "source_family": source_family,
                "source_event": event,
                "destination_pair": destination_pair,
                "destination_family": destination_family,
                "source_event_sessions": support,
                "destination_activations": activations,
                "destination_base_support": base_support,
                "destination_base_activations": base_activations,
                "raw_transition_probability": raw_rate,
                "destination_base_rate": base_rate,
                "directed_lift": raw_lift,
                "shrunk_lift": shrunk_lift,
                "support_status": "supported" if supported else "unknown",
                "familywise_status": "secondary_holm_required"
                if supported
                else "unsupported_not_tested",
            }
        )
    return pd.DataFrame.from_records(rows)


def make_plots(
    output: Path,
    graph: pd.DataFrame,
    comparisons: pd.DataFrame,
    calibration: pd.DataFrame,
    timing: pd.DataFrame,
    nulls: pd.DataFrame,
    concentration: pd.DataFrame,
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    primary_graph = graph.loc[
        graph["graph_as_of_session"].eq(
            graph.groupby("period")["graph_as_of_session"].transform("max")
        )
        & graph["support_status"].eq("supported")
    ].copy()
    if not primary_graph.empty:
        heat = primary_graph.loc[primary_graph["source_event"].eq("newly_decaying")].pivot_table(
            index="source_family",
            columns="destination_family",
            values="shrunk_lift",
            aggfunc="mean",
        )
        fig, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(heat.fillna(1.0), cmap="coolwarm", vmin=0.75, vmax=1.25)
        axis.set_xticks(
            range(len(heat.columns)),
            [value.rsplit("_", 1)[-1] for value in heat.columns],
            rotation=45,
        )
        axis.set_yticks(range(len(heat.index)), [value.rsplit("_", 1)[-1] for value in heat.index])
        axis.set_title("Past-only decaying-source directed lift")
        fig.colorbar(image, ax=axis, label="shrunk lift")
        fig.tight_layout()
        path = output / "source_decay_destination_activation_heatmap.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    lead = comparisons.loc[
        comparisons["comparison"].eq("M3_vs_M1") & comparisons["period_slice"].eq("all")
    ].sort_values("target_window_sessions")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(lead["target_window_sessions"], lead["brier_improvement"], marker="o")
    axis.set_xlabel("Activation window (sessions)")
    axis.set_ylabel("M3 minus M1 Brier improvement")
    axis.set_title("Directed-rotation incremental calibration")
    fig.tight_layout()
    path = output / "directed_rotation_window_shape.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    selected_calibration = calibration.loc[
        calibration["model_name"].isin(
            ["M1_destination_own_history", "M3_directed_family_rotation"]
        )
        & calibration["target_window_sessions"].eq(3)
    ]
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    for model, group in selected_calibration.groupby("model_name", observed=True):
        axis.plot(group["mean_probability"], group["activation_rate"], marker="o", label=model[:2])
    axis.set_xlabel("Predicted activation probability")
    axis.set_ylabel("Observed activation rate")
    axis.legend()
    axis.set_title("Primary-window calibration")
    fig.tight_layout()
    path = output / "m1_vs_m3_calibration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    if not timing.empty:
        observed = timing.loc[timing["activation_observed"]]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.scatter(
            observed["sessions_to_activation"],
            observed["episode_payoff_available_after_forecast_bps"],
            alpha=0.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Sessions from forecast to activation")
        axis.set_ylabel("Episode payoff available (bps)")
        axis.set_title("Forecast lead versus episode payoff")
        fig.tight_layout()
        path = output / "forecast_lead_episode_payoff.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    if not nulls.empty:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(nulls["null_test"], nulls["brier_improvement"])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("M3 minus M1 Brier improvement")
        axis.set_title("Registered directed-rotation nulls")
        fig.tight_layout()
        path = output / "real_vs_null_increment.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    if not concentration.empty:
        edge = concentration.loc[concentration["dimension"].eq("source_state_vector")].head(10)
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(range(len(edge)), edge["net_contribution_bps"])
        axis.set_xticks(
            range(len(edge)), [f"edge {index + 1}" for index in range(len(edge))], rotation=45
        )
        axis.set_ylabel("Net contribution (bps)")
        axis.set_title("Economic concentration by source-state vector")
        fig.tight_layout()
        path = output / "economic_concentration_by_edge.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def build_episode_attribution(
    episodes: pd.DataFrame,
    scored_forecasts: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    result = episodes.copy()
    index = {
        (int(str(period)), str(session)): item
        for period, group in calendar.groupby("period", sort=True, observed=True)
        for item, session in enumerate(
            group["score_session"].astype(str).drop_duplicates().sort_values()
        )
    }
    m3 = scored_forecasts.loc[
        scored_forecasts["model_name"].eq("M3_directed_family_rotation")
        & scored_forecasts["target_window_sessions"].eq(3)
        & scored_forecasts["prediction_state"].eq("nominated")
    ].copy()
    first_ids: list[object] = []
    leads: list[float] = []
    for episode in result.to_dict(orient="records"):
        period = int(episode["period"])
        onset = str(episode["episode_onset_session"])
        candidates = m3.loc[
            m3["period"].eq(period)
            & m3["destination_family"].eq(episode["destination_family"])
            & m3["forecast_session"].astype(str).lt(onset)
            & m3["target_episode_ids"]
            .fillna("")
            .astype(str)
            .str.contains(str(episode["episode_id"]), regex=False)
        ].sort_values("forecast_session", kind="stable")
        if candidates.empty:
            first_ids.append(pd.NA)
            leads.append(math.nan)
            continue
        forecast = candidates.iloc[0]
        first_ids.append(forecast["forecast_id"])
        leads.append(
            float(index[(period, onset)] - index[(period, str(forecast["forecast_session"]))])
        )
    result["first_m3_forecast_id"] = first_ids
    result["forecast_lead_sessions"] = leads
    result["forecast_preceded_onset"] = result["forecast_lead_sessions"].gt(0.0)
    result["fraction_episode_remaining_after_forecast"] = np.where(
        result["forecast_preceded_onset"], 1.0, 0.0
    )
    return result


def write_report(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    mapping: Mapping[str, Any],
    source_events: pd.DataFrame,
    graph: pd.DataFrame,
    activation_metrics_table: pd.DataFrame,
    comparisons: pd.DataFrame,
    system_metrics: pd.DataFrame,
    economic_metrics: pd.DataFrame,
    null_results: pd.DataFrame,
    loo_results: pd.DataFrame,
    concentration: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    primary_comparison = comparisons.loc[
        comparisons["comparison"].eq("M3_vs_M1")
        & comparisons["target_window_sessions"].eq(3)
        & comparisons["period_slice"].eq("all")
    ].iloc[0]
    direction_comparison = comparisons.loc[
        comparisons["comparison"].eq("M3_vs_M2")
        & comparisons["target_window_sessions"].eq(3)
        & comparisons["period_slice"].eq("all")
    ].iloc[0]

    def model_row(name: str) -> pd.Series:
        rows = activation_metrics_table.loc[
            activation_metrics_table["model_name"].eq(name)
            & activation_metrics_table["target_window_sessions"].eq(3)
            & activation_metrics_table["period_slice"].eq("all")
        ]
        return rows.iloc[0]

    m0 = model_row("M0_activation_base_rate")
    m1 = model_row("M1_destination_own_history")
    m2 = model_row("M2_undirected_system_state")
    m3 = model_row("M3_directed_family_rotation")
    latest_graph = graph.loc[
        graph["graph_as_of_session"].eq(
            graph.groupby("period")["graph_as_of_session"].transform("max")
        )
    ]
    supported = latest_graph.loc[latest_graph["support_status"].eq("supported")].sort_values(
        ["support", "shrunk_lift"], ascending=[False, False], kind="stable"
    )
    edge_lines = (
        "\n".join(
            f"- {row.source_family} `{row.source_event}` → {row.destination_family}: "
            f"shrunk lift {row.shrunk_lift:.3f}, support {int(row.support)}, activations {int(row.activations)}"
            for row in supported.head(12).itertuples(index=False)
        )
        or "- No family edge met the frozen support threshold."
    )
    event_counts = (
        source_events.groupby("destination_family", observed=True)[
            ["newly_active", "newly_decaying", "newly_retired"]
        ]
        .sum()
        .reset_index()
    )
    event_lines = "\n".join(
        f"- {row.destination_family}: active onsets {int(row.newly_active)}, decays {int(row.newly_decaying)}, retirements {int(row.newly_retired)}"
        for row in event_counts.itertuples(index=False)
    )
    economic = economic_metrics.loc[
        economic_metrics["model_name"].eq("M3_directed_family_rotation")
    ]
    economic_text = (
        f"{float(economic.iloc[0].total_net_payoff_bps):.2f} bps over "
        f"{int(economic.iloc[0].eligible_opportunities)} eligible opportunities; twice-cost "
        f"{float(economic.iloc[0].twice_cost_net_payoff_bps):.2f} bps; one-bar-delay "
        f"{float(economic.iloc[0].one_bar_delay_net_payoff_bps):.2f} bps over "
        f"{int(economic.iloc[0].one_bar_delay_observations)} observable delayed rows"
        if not economic.empty
        else "No nominated M3 forecast had a tradeable destination opportunity."
    )
    loo_text = (
        f"mean Brier improvement {loo_results.brier_improvement.mean():.6f}; "
        f"positive in {loo_results.brier_improvement.gt(0).mean():.1%} of fully rebuilt exclusions"
        if not loo_results.empty
        else "not completed"
    )
    null_lines = "\n".join(
        f"- {row.null_test}: Brier improvement {row.brier_improvement:.6f}, log-loss improvement {row.log_loss_improvement:.6f}"
        for row in null_results.itertuples(index=False)
    )
    top_concentration = concentration.sort_values(
        "absolute_contribution_share", ascending=False, kind="stable"
    ).head(6)
    concentration_lines = (
        "\n".join(
            f"- {row.dimension} `{row.key}`: {row.absolute_contribution_share:.1%} of absolute contribution"
            for row in top_concentration.itertuples(index=False)
        )
        or "- No M3 economic contribution was available."
    )
    observed_timing = timing.loc[
        timing["model_name"].eq("M3_directed_family_rotation") & timing["activation_observed"]
    ]
    timing_text = (
        f"median lead {observed_timing.sessions_to_activation.median():.1f} sessions; "
        f"mean episode payoff available {observed_timing.episode_payoff_available_after_forecast_bps.mean():.2f} bps"
        if not observed_timing.empty
        else "no correctly nominated activation episode"
    )
    report = f"""# Directed Economic Loop–Regime Rotation V1

## 1–3. Hypothesis, prior boundary, and scientific status

The frozen question is whether the causal decay or retirement of one structural family improves prediction of a **different** family's new positive economic episode within three trading sessions. It is not same-pair persistence, structural loop recurrence, a rolling winner table, or the within-opportunity competitor veto.

The prior rolling selector estimated each pair's own recent payoff; V2 estimated each pair's own lifecycle; lead-lag tested same-pair structural features; recurrence predicted structural paths; and Sequential Competitor Veto eliminated simultaneously compatible loops inside one opportunity. None previously estimated `P(destination activates soon | different source decays/retires)`, separated no/one/multiple family activations, or used a past-only directed economic graph. These opened 2023/2025 surfaces are attribution data, not validation or trading approval.

## 4–8. Frozen taxonomy, state construction, target, and clock

The outcome-free taxonomy maps all 24 frozen two-transition return-cycle pairs to eight orientation families: {", ".join(mapping["destination_families"])}. No family was merged after payoff inspection. The source economic state is the frozen V2 `hierarchical_payoff_history_change_point`, with structural leading features disabled. Pair probabilities are support-weighted into family states; active takes precedence, then decaying, while mixed retired/unknown remains unknown.

The primary label is a new family-positive episode onset in the next **three explicit trading sessions**; one and five sessions are sensitivities. Current active episodes are excluded, multiple activations remain multi-label, and missing future payoff support is unavailable—not zero. Positive labels mature only after the full unioned family episode end. Forecast freeze is the V2 regular-session-open decision timestamp.

## 9–12. Walk-forward graph, source events, and destination base rates

At each session the runner first matures labels with availability strictly before the freeze, then updates beta-smoothed destination rates and cross-family graph counts, constructs M1/M2/M3 features, and freezes forecasts. Same-family edges are excluded. The graph uses alpha=beta=1 edge smoothing, pooling strength 20, and a minimum eight source-event sessions.

Source lifecycle census:

{event_lines}

Supported end-of-period family edges (descriptive graph table; not individually promoted):

{edge_lines}

## 13–18. M0–M3 and primary paired tests

| Model | Brier | Log loss | ECE | Coverage |
|---|---:|---:|---:|---:|
| M0 base rate | {m0.brier_score:.6f} | {m0.log_loss:.6f} | {m0.ece:.6f} | {m0.coverage:.1%} |
| M1 destination history | {m1.brier_score:.6f} | {m1.log_loss:.6f} | {m1.ece:.6f} | {m1.coverage:.1%} |
| M2 undirected system | {m2.brier_score:.6f} | {m2.log_loss:.6f} | {m2.ece:.6f} | {m2.coverage:.1%} |
| M3 directed family | {m3.brier_score:.6f} | {m3.log_loss:.6f} | {m3.ece:.6f} | {m3.coverage:.1%} |

Primary M3-versus-M1 paired Brier improvement: **{primary_comparison.brier_improvement:.6f}** (session-block 95% interval {primary_comparison.brier_interval_lower:.6f} to {primary_comparison.brier_interval_upper:.6f}); paired log-loss improvement: **{primary_comparison.log_loss_improvement:.6f}**. Directional M3-versus-M2 Brier improvement: **{direction_comparison.brier_improvement:.6f}**; log-loss improvement: **{direction_comparison.log_loss_improvement:.6f}**. Period and 1/5-session shapes are in `paired_model_metrics.csv`; no window was selected after scoring.

## 19–23. Pair refinement, system outcomes, timing, and economic translation

M4 is secondary: pair activation rates are shown only at frozen support (20 rows and four activations) and shrunk toward M3 family forecasts. Unsupported pair edges remain unknown. No-activation and multiple-activation probabilities are exported separately; multi-label scoring is not mixed with first-activation ranking.

Activation timing: {timing_text}. Because all primary onsets follow the forecast, the mapped episode payoff is entirely post-forecast at the family-label level; this does not imply that a qualifying trade exists.

Opportunity translation uses only later frozen no-filter V2 opportunities with the exact predicted destination family inside the three-session window. Missing families are not replaced and overlap/capacity is not refilled. M3 result: {economic_text}.

## 24–29. Cost, nulls, leave-one-stock-out, and concentration

Registered nulls:

{null_lines}

Fully rebuilt leave-one-stock-out: {loo_text}. The median session aggregation is fully rebuilt. The minimum-two-bar dwell and alternate taxonomy sensitivities are explicitly not applicable because neither was registered for this session-level source; primary states were not silently changed.

Concentration diagnostics:

{concentration_lines}

## 30–34. Failure cases, decision, and recommendation

Failure modes include sparse newly-decaying/retired edges, long positive family unions that suppress new-onset labels, family aggregation where one active pair masks another pair's retirement, calibrated episode prediction without a later eligible opportunity, and contribution concentration. A structured graph or one high-lift edge is not evidence of tradeability.

Scientific decision: **`{metadata["scientific_decision"]}`**.

This decision distinguishes historical description from prediction: every scored forecast was frozen before its target, but the periods were already opened. Independent audit and exact-rerun identity are separate machine-readable artifacts and are required before the result can justify even prospective logging.

Exact next recommendation: freeze this contract and prospectively log M1/M2/M3 family forecasts on a genuinely unopened data snapshot until enough new decay/retirement events mature; do not refine pairs or thresholds during collection.

## Reproducibility and assumptions

- Run ID: `{metadata["run_id"]}`
- Git SHA: `{metadata["git_sha"]}`
- Contract SHA-256: `{metadata["contract_hash"]}`
- Data snapshot: `{metadata["data_snapshot_id"]}`
- Fixed horizon: 24 bars; frozen cost translation: 5 bps per side.
- A family positive episode is the calendar union of overlapping or adjacent frozen pair-positive episodes. This union is an evaluation label, never a feature.
- Family taxonomy uses current frozen regime orientation because the repository had no prior outcome-free named branch taxonomy spanning all eligible pairs.
- The V2 decision timestamp is the regular-session-open family scoring freeze; within-session opportunities are later economic translations, not state-feature inputs.
- Existing entries, exits, overlap, positions, broker paths, and runtime behaviour remain unchanged.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def prospective_schema() -> dict[str, object]:
    return {
        "schema_version": "directed_economic_rotation_prospective_v1",
        "forecast_create_only": True,
        "outcome_create_only": True,
        "execution_enabled": False,
        "opened_historical_sources_rejected_in_holdout_mode": True,
        "new_data_snapshot_required_for_holdout": True,
        "required_forecast_fields": [
            "run_id",
            "git_sha",
            "contract_hash",
            "data_snapshot_hash",
            "forecast_id",
            "forecast_session",
            "forecast_timestamp",
            "destination_family",
            "source_family_state_vector_json",
            "predicted_activation_probability",
            "probability_no_activation",
            "probability_multiple_activation",
            "prediction_state",
            "feature_availability_timestamp",
            "training_cutoff",
            "forecast_freeze_timestamp",
        ],
    }


def run_historical(
    *,
    output: Path,
    report_path: Path,
    exact_rerun_of: Path | None,
    rebuild_stresses: bool,
) -> None:
    ensure_new_output(output)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    contract, input_hashes, data_snapshot = verify_contract_and_inputs()
    contract_hash = sha256(CONTRACT_PATH)
    git_sha = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    run_id = "directed-rotation-" + stable_hash(
        [contract_hash, data_snapshot, git_sha, MODEL_VERSION]
    )
    surfaces = load_frozen_surfaces(contract)
    taxonomy = surfaces["taxonomy"]
    assert isinstance(taxonomy, FamilyTaxonomy)
    pair_states = surfaces["pair_states"]
    session_panel = surfaces["session_panel"]
    pair_episodes = surfaces["episode_diagnostics"]
    all_pair_forecasts = surfaces["all_pair_forecasts"]
    opportunities = surfaces["trade_decisions"]
    one_bar_delay = surfaces["one_bar_delay"]
    assert isinstance(pair_states, pd.DataFrame)
    assert isinstance(session_panel, pd.DataFrame)
    assert isinstance(pair_episodes, pd.DataFrame)
    assert isinstance(all_pair_forecasts, pd.DataFrame)
    assert isinstance(opportunities, pd.DataFrame)
    assert isinstance(one_bar_delay, pd.DataFrame)
    state = build_core_state_targets(
        pair_states=pair_states,
        session_panel=session_panel,
        pair_episodes=pair_episodes,
        taxonomy=taxonomy,
    )
    scored_primary = run_registered_models(state, contract, run_id=run_id)
    contextual = comparator_forecasts(
        all_pair_forecasts,
        taxonomy,
        state["targets"],
    )
    scored_all = pd.concat([scored_primary, contextual], ignore_index=True, sort=False)
    pair_targets = build_pair_targets(
        pair_states,
        session_panel,
        pair_episodes,
        state["calendar"],
    )
    pair_refinement = run_pair_refinement(
        scored_primary,
        pair_states,
        pair_targets,
        taxonomy,
        run_id=run_id,
    )
    graph = build_past_only_graph_ledger(
        state["family_states"],
        state["source_events"],
        state["targets"],
        prequential_settings(contract, run_id=run_id, window=3).graph,
    )
    pair_graph = build_supported_pair_graph(pair_states, pair_targets, taxonomy)
    activation_table, calibration, comparisons, system_table = build_metrics(
        scored_all,
        contract,
    )
    immutable_forecasts, outcome_ledger = split_forecast_and_outcome_ledgers(scored_all)
    pair_forecasts, pair_outcomes = split_forecast_and_outcome_ledgers(pair_refinement)
    translation, economic_metrics, twice_cost = build_economic_translation(
        scored_primary,
        opportunities,
        state["calendar"],
        one_bar_delay,
    )
    concentration = build_concentration(translation)
    deletion_stresses = build_deletion_stresses(translation, concentration)
    timing = build_activation_timing(
        scored_primary,
        state["family_episodes"],
        state["calendar"],
    )
    episode_attribution = build_episode_attribution(
        state["family_episodes"],
        scored_primary,
        state["calendar"],
    )
    null_results, null_forecasts = build_null_results(
        state,
        contract,
        run_id=run_id,
    )
    stress_rows: list[dict[str, object]] = []
    for row in comparisons.loc[
        comparisons["period_slice"].eq("all") & comparisons["comparison"].eq("M3_vs_M1")
    ].to_dict(orient="records"):
        stress_rows.append(
            {
                "stress_test": f"activation_window_{int(row['target_window_sessions'])}_sessions",
                **row,
            }
        )
    for strength in contract["directed_graph"]["shrinkage_sensitivities"]:
        forecasts = run_prequential_rotation(
            state["family_states"],
            state["source_events"],
            state["targets"],
            settings=prequential_settings(
                contract,
                run_id=f"{run_id}-pool-{float(strength):g}",
                window=3,
                pooling_strength=float(strength),
            ),
        )
        comparison = paired_model_comparison(
            forecasts,
            treatment="M3_directed_family_rotation",
            control="M1_destination_own_history",
            bootstrap_resamples=500,
            seed=int(contract["random_seed"]),
        )
        stress_rows.append(
            {
                "stress_test": f"graph_pooling_strength_{float(strength):g}",
                **comparison,
            }
        )
    stress_rows.extend(deletion_stresses.to_dict(orient="records"))
    stress_rows.extend(
        [
            {
                "stress_test": "twice_costs",
                "trade_count": int(
                    twice_cost["model_name"].eq("M3_directed_family_rotation").sum()
                ),
                "net_payoff_bps": float(
                    twice_cost.loc[
                        twice_cost["model_name"].eq("M3_directed_family_rotation"),
                        "stressed_net_payoff_bps",
                    ].sum()
                ),
                "status": "existing_immutable_opportunities_repriced",
            },
            {
                "stress_test": "one_bar_execution_delay",
                "trade_count": int(
                    translation.loc[
                        translation["model_name"].eq("M3_directed_family_rotation"),
                        "one_bar_delay_available",
                    ].sum()
                ),
                "net_payoff_bps": float(
                    pd.to_numeric(
                        translation.loc[
                            translation["model_name"].eq("M3_directed_family_rotation"),
                            "one_bar_delay_constant_terminal_net_bps",
                        ],
                        errors="coerce",
                    ).sum()
                ),
                "status": "available_rows_only_missing_not_zero",
            },
            {
                "stress_test": "minimum_two_bar_state_dwell",
                "status": "not_applicable_frozen_session_level_state_source",
            },
            {
                "stress_test": "family_taxonomy_sensitivity",
                "status": "none_registered_before_scoring",
            },
        ]
    )
    if rebuild_stresses:
        rebuilt_stress, loo_results = run_rebuilt_stresses(
            contract,
            taxonomy,
            run_id=run_id,
        )
        stress_rows.extend(rebuilt_stress.to_dict(orient="records"))
    else:
        loo_results = pd.DataFrame()
        stress_rows.append(
            {
                "stress_test": "fully_rebuilt_sensitivities",
                "status": "skipped_by_explicit_command_flag",
            }
        )
    stress_results = pd.DataFrame.from_records(stress_rows)
    decision = scientific_decision(
        comparisons,
        economic_metrics=economic_metrics,
        null_metrics=null_results,
        loo_results=loo_results,
        concentration=concentration,
    )

    output.mkdir(parents=True)
    mapping_raw = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    write_json(output / "frozen_structural_family_mapping.json", mapping_raw)
    detailed = {
        "session_level_pair_economic_state_ledger.parquet": pair_states,
        "session_level_family_economic_state_ledger.parquet": state["family_states"],
        "source_lifecycle_transition_events.parquet": state["source_events"],
        "destination_activation_targets.parquet": state["targets"],
        "family_payoff_support.parquet": state["family_payoff_support"],
        "past_only_family_transition_graph.parquet": graph,
        "supported_pair_transition_graph.parquet": pair_graph,
        "frozen_destination_forecasts.parquet": immutable_forecasts,
        "settled_activation_outcome_ledger.parquet": outcome_ledger,
        "no_and_multiple_activation_forecasts.parquet": scored_primary[
            [
                "forecast_id",
                "period",
                "forecast_session",
                "target_window_sessions",
                "model_name",
                "probability_no_activation",
                "probability_multiple_activation",
                "predicted_activation_count",
                "observed_activation_count",
                "no_activation_flag",
                "multiple_activation_flag",
                "target_available",
            ]
        ],
        "M0_base_rate_predictions.parquet": immutable_forecasts.loc[
            immutable_forecasts["model_name"].eq("M0_activation_base_rate")
        ],
        "M1_destination_own_history_predictions.parquet": immutable_forecasts.loc[
            immutable_forecasts["model_name"].eq("M1_destination_own_history")
        ],
        "M2_undirected_system_predictions.parquet": immutable_forecasts.loc[
            immutable_forecasts["model_name"].eq("M2_undirected_system_state")
        ],
        "M3_directed_family_rotation_predictions.parquet": immutable_forecasts.loc[
            immutable_forecasts["model_name"].eq("M3_directed_family_rotation")
        ],
        "M4_supported_pair_rotation_predictions.parquet": pair_forecasts,
        "M4_supported_pair_rotation_outcomes.parquet": pair_outcomes,
        "comparator_predictions.parquet": immutable_forecasts.loc[
            immutable_forecasts["model_name"].isin(
                ["M5_payoff_only_BOCPD", "M6_v1_60_session_selector"]
            )
        ],
        "economic_opportunity_translation.parquet": translation,
        "twice_cost_economic_translation.parquet": twice_cost,
        "null_test_forecasts.parquet": null_forecasts,
        "activation_timing_results.parquet": timing,
        "episode_attribution.parquet": episode_attribution,
    }
    summaries = {
        "activation_prediction_metrics.csv": activation_table,
        "paired_model_metrics.csv": comparisons,
        "calibration_results.csv": calibration,
        "system_activation_metrics.csv": system_table,
        "economic_opportunity_metrics.csv": economic_metrics,
        "stress_test_results.csv": stress_results,
        "null_test_results.csv": null_results,
        "leave_one_stock_out_results.csv": loo_results,
        "concentration_results.csv": concentration,
        "supported_family_transition_graph.csv": graph.loc[
            graph["support_status"].eq("supported")
            & graph["graph_as_of_session"].eq(
                graph.groupby("period")["graph_as_of_session"].transform("max")
            )
        ],
    }
    for filename, frame in detailed.items():
        _write_parquet(
            output / filename,
            _annotate(
                frame,
                run_id=run_id,
                contract_hash=contract_hash,
                data_snapshot=data_snapshot,
            ),
        )
    for filename, frame in summaries.items():
        _write_csv(
            output / filename,
            _annotate(
                frame,
                run_id=run_id,
                contract_hash=contract_hash,
                data_snapshot=data_snapshot,
            ),
        )
    write_json(output / "prospective_immutable_forecast_ledger_schema.json", prospective_schema())
    plot_paths = make_plots(
        output,
        graph,
        comparisons,
        calibration,
        timing,
        null_results,
        concentration,
    )
    metadata = {
        "run_id": run_id,
        "git_sha": git_sha,
        "repository_branch": branch,
        "contract_id": contract["contract_id"],
        "contract_hash": contract_hash,
        "family_mapping_hash": sha256(MAPPING_PATH),
        "data_snapshot_id": data_snapshot,
        "source_hashes": input_hashes,
        "model_version": MODEL_VERSION,
        "feature_schema_version": "directed_rotation_destination_system_graph_v1",
        "fixed_horizon_bars": 24,
        "primary_target_window_sessions": 3,
        "cost_bps_per_side": 5.0,
        "random_seed": int(contract["random_seed"]),
        "scored_periods": [2023, 2025],
        "generated_at": RUN_TIMESTAMP,
        "scientific_status": contract["scientific_status"],
        "scientific_decision": decision,
        "forecast_timestamp_convention": "V2 regular-session-open decision timestamp",
        "label_settlement_convention": "strictly after target support and full positive episode end",
        "command": (
            "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
            "research/slrno-v2/20260714-regime-loop-handoff/work/"
            "run_directed_economic_loop_regime_rotation_v1.py --output <OUTPUT> --report <REPORT>"
        ),
        "rebuild_stresses_executed": rebuild_stresses,
        "safety": contract["safety"],
        "artifact_names": sorted([*detailed, *summaries]),
        "plot_names": sorted(path.name for path in plot_paths),
    }
    write_json(output / "run_metadata.json", metadata)
    if exact_rerun_of is not None:
        identity = verify_exact_rerun(output, exact_rerun_of)
        write_json(output / "exact_rerun_identity.json", identity)
        if not bool(identity["byte_identical"]):
            raise AssertionError(f"exact rerun identity failed: {identity}")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_report(
        report_path,
        metadata=metadata,
        mapping=mapping_raw,
        source_events=state["source_events"],
        graph=graph,
        activation_metrics_table=activation_table,
        comparisons=comparisons,
        system_metrics=system_table,
        economic_metrics=economic_metrics,
        null_results=null_results,
        loo_results=loo_results,
        concentration=concentration,
        timing=timing,
    )


def run_prospective(args: argparse.Namespace) -> None:
    if args.prospective_root is None or args.record_json is None:
        raise ValueError("prospective mode requires --prospective-root and --record-json")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    ledger = ProspectiveRotationLedger(
        Path(args.prospective_root),
        opened_periods=set(contract["opened_data_status"]["opened_periods"]),
    )
    record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
    if args.mode == "prospective-forecast":
        path = ledger.append_forecast(record, holdout=bool(args.holdout))
    else:
        path = ledger.append_outcome(record)
    print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["historical", "prospective-forecast", "prospective-settle"],
        default="historical",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--exact-rerun-of", type=Path)
    parser.add_argument("--skip-rebuild-stresses", action="store_true")
    parser.add_argument("--prospective-root", type=Path)
    parser.add_argument("--record-json", type=Path)
    parser.add_argument("--holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "historical":
        run_historical(
            output=Path(args.output),
            report_path=Path(args.report),
            exact_rerun_of=(Path(args.exact_rerun_of) if args.exact_rerun_of is not None else None),
            rebuild_stresses=not bool(args.skip_rebuild_stresses),
        )
    else:
        run_prospective(args)


if __name__ == "__main__":
    main()
