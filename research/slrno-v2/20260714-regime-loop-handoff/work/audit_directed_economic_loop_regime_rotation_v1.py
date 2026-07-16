# ruff: noqa: E501
"""Independent auditor for Directed Economic Loop-Regime Rotation V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260716-directed-economic-loop-regime-rotation-v1.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260716-directed-economic-loop-regime-rotation-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260716-directed-economic-loop-regime-rotation-v1/exact_rerun"
MACHINE_SUFFIXES = {".parquet", ".csv", ".json"}
IDENTITY_EXCLUSIONS = {
    "artifact_manifest.json",
    "exact_rerun_identity.json",
    "independent_audit.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_window(
    sessions: Sequence[str],
    forecast_session: str,
    window: int,
) -> list[str] | None:
    if window not in {1, 3, 5}:
        raise ValueError("unregistered target window")
    try:
        index = list(sessions).index(str(forecast_session))
    except ValueError:
        return None
    if index + window >= len(sessions):
        return None
    return list(sessions[index + 1 : index + window + 1])


def verify_exact_machine_identity(primary: Path, exact: Path) -> dict[str, object]:
    primary_files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file()
        and path.suffix in MACHINE_SUFFIXES
        and path.name not in IDENTITY_EXCLUSIONS
    }
    exact_files = {
        path.name: path
        for path in exact.iterdir()
        if path.is_file()
        and path.suffix in MACHINE_SUFFIXES
        and path.name not in IDENTITY_EXCLUSIONS
    }
    missing = sorted(set(primary_files) - set(exact_files))
    extra = sorted(set(exact_files) - set(primary_files))
    mismatches = sorted(
        name
        for name in set(primary_files) & set(exact_files)
        if sha256(primary_files[name]) != sha256(exact_files[name])
    )
    return {
        "byte_identical": not missing and not extra and not mismatches,
        "compared_files": len(primary_files),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
    }


def prohibited_changed_paths(paths: Iterable[str]) -> list[str]:
    allowed = (
        "research/slrno-v2/",
        "packages/stocker_research/",
        "tests/",
    )
    return sorted(path for path in paths if not path.startswith(allowed))


def _json_safe(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _resolved(value: str) -> Path:
    return (CONTRACT_PATH.parent / value).resolve()


def _input_hashes(contract: Mapping[str, Any]) -> dict[str, str]:
    inputs = contract["inputs"]
    paths = {
        "family_mapping": _resolved(inputs["family_mapping"]["path"]),
        "pair_states": _resolved(inputs["causal_edge_state_forecasts"]["path"]),
        "session_panel": _resolved(inputs["session_payoff_panel"]["path"]),
        "episode_states": _resolved(inputs["hindsight_episode_states"]["path"]),
        "episode_diagnostics": _resolved(inputs["hindsight_episode_diagnostics"]["path"]),
        "trade_decisions": _resolved(inputs["trade_decisions"]["path"]),
        "one_bar_delay": _resolved(inputs["one_bar_delay_outcomes"]["path"]),
        "cycle_dictionary": _resolved(inputs["cycle_dictionary"]["path"]),
        "v2_rebuild_runner": _resolved(inputs["v2_rebuild_runner"]["path"]),
        "v2_rebuild_config": _resolved(inputs["v2_rebuild_config"]["path"]),
    }
    return {name: sha256(path) for name, path in paths.items()}


def _data_snapshot(contract: Mapping[str, Any]) -> str:
    payload = {"contract": sha256(CONTRACT_PATH), **_input_hashes(contract)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mapping_lookup(mapping: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (str(row["loop_id"]), str(row["orientation"])): str(row["destination_family"])
        for row in mapping["pairs"]
    }


def _check_family_state_reconstruction(
    root: Path,
    mapping: Mapping[str, Any],
) -> tuple[bool, str]:
    pairs = pd.read_parquet(root / "session_level_pair_economic_state_ledger.parquet")
    families = pd.read_parquet(root / "session_level_family_economic_state_ledger.parquet")
    lookup = _mapping_lookup(mapping)
    pairs["destination_family"] = [
        lookup.get((str(loop), str(orientation)), "unknown_topology")
        for loop, orientation in pairs[["loop_id", "orientation"]].itertuples(
            index=False, name=None
        )
    ]
    expected: dict[tuple[int, str, str], tuple[str, float]] = {}
    for key, group in pairs.groupby(
        ["period", "score_session", "destination_family"],
        sort=True,
        observed=True,
    ):
        if str(key[2]) == "unknown_topology":
            continue
        states = set(group["edge_state"].astype(str))
        state = (
            "active"
            if "active" in states
            else "decaying"
            if "decaying" in states
            else "retired"
            if "retired" in states and "unknown" not in states
            else "unknown"
        )
        weights = np.clip(
            pd.to_numeric(group["effective_sample_size"], errors="coerce")
            .fillna(1.0)
            .to_numpy(float),
            1.0,
            50.0,
        )
        mean = float(
            np.average(
                pd.to_numeric(group["posterior_mean_net_bps"], errors="coerce").to_numpy(float),
                weights=weights,
            )
        )
        expected[(int(str(key[0])), str(key[1]), str(key[2]))] = (state, mean)
    for row in families.itertuples(index=False):
        value = expected.get(
            (int(str(row.period)), str(row.score_session), str(row.destination_family))
        )
        if value is None:
            return False, "family row has no mapped pair population"
        if str(row.operational_state) != value[0] or not math.isclose(
            float(str(row.posterior_mean_net_bps)),
            value[1],
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return False, "family state or support-weighted mean mismatch"
    return True, f"reconstructed {len(families)} family-state rows"


def _check_source_events(root: Path) -> tuple[bool, str]:
    states = pd.read_parquet(root / "session_level_family_economic_state_ledger.parquet")
    events = pd.read_parquet(root / "source_lifecycle_transition_events.parquet")
    states = states.sort_values(["period", "destination_family", "score_session"], kind="stable")
    prior = states.groupby(["period", "destination_family"], observed=True)[
        "operational_state"
    ].shift()
    expected_decay = states["operational_state"].eq("decaying") & prior.eq("active")
    expected_retire = states["operational_state"].eq("retired") & prior.isin(["active", "decaying"])
    expected_active = states["operational_state"].eq("active") & prior.fillna("unknown").ne(
        "active"
    )
    aligned = events.sort_values(["period", "destination_family", "score_session"], kind="stable")
    ok = (
        np.array_equal(expected_decay.to_numpy(bool), aligned["newly_decaying"].to_numpy(bool))
        and np.array_equal(expected_retire.to_numpy(bool), aligned["newly_retired"].to_numpy(bool))
        and np.array_equal(expected_active.to_numpy(bool), aligned["newly_active"].to_numpy(bool))
        and pd.to_datetime(aligned["source_event_timestamp"], utc=True)
        .eq(pd.to_datetime(aligned["forecast_freeze_timestamp"], utc=True))
        .all()
    )
    return bool(ok), f"reconstructed {len(aligned)} causal source-event rows"


def _check_targets(root: Path) -> tuple[bool, str]:
    states = pd.read_parquet(root / "session_level_family_economic_state_ledger.parquet")
    targets = pd.read_parquet(root / "destination_activation_targets.parquet")
    support = pd.read_parquet(root / "family_payoff_support.parquet")
    episodes = pd.read_parquet(root / "episode_attribution.parquet")
    calendars = {
        int(str(period)): group["score_session"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
        for period, group in states.groupby("period", sort=True, observed=True)
    }
    state_lookup = {
        (int(str(row.period)), str(row.score_session), str(row.destination_family)): str(
            row.operational_state
        )
        for row in states.itertuples(index=False)
    }
    failures: list[str] = []
    for row in targets.itertuples(index=False):
        period = int(str(row.period))
        window = int(str(row.target_window_sessions))
        sessions = target_window(calendars[period], str(row.forecast_session), window)
        if sessions is None:
            if str(row.target_status) != "period_boundary":
                failures.append("period boundary")
            continue
        if (
            str(row.target_start_session) != sessions[0]
            or str(row.target_end_session) != sessions[-1]
        ):
            failures.append("calendar join")
            continue
        key = (period, str(row.forecast_session), str(row.destination_family))
        if state_lookup[key] == "active":
            if str(row.target_status) != "current_active_not_candidate":
                failures.append("active continuation labelled as onset")
            continue
        observed_support = support.loc[
            support["period"].eq(period)
            & support["destination_family"].eq(str(row.destination_family))
            & support["session"].astype(str).isin(sessions)
        ]
        if observed_support.empty:
            if bool(row.target_available):
                failures.append("missing support became no activation")
            continue
        onsets = episodes.loc[
            episodes["period"].eq(period)
            & episodes["destination_family"].eq(str(row.destination_family))
            & episodes["episode_onset_session"].astype(str).isin(sessions)
        ]
        expected = not onsets.empty
        if not bool(row.target_available) or bool(row.activation_target) != expected:
            failures.append("activation label")
    keys = ["period", "forecast_session", "target_window_sessions"]
    for _, group in targets.groupby(keys, sort=True, observed=True):
        observed = int((group["target_available"] & group["activation_target"].fillna(False)).sum())
        if not group["observed_activation_count"].eq(observed).all():
            failures.append("multiple activation count")
        expected_no = bool(group["target_available"].any() and observed == 0)
        if not group["no_activation_flag"].eq(expected_no).all():
            failures.append("no activation flag")
    return not failures, f"checked {len(targets)} targets; failures={sorted(set(failures))}"


def _check_graph(root: Path) -> tuple[bool, str]:
    graph = pd.read_parquet(root / "past_only_family_transition_graph.parquet")
    targets = pd.read_parquet(root / "destination_activation_targets.parquet")
    events = pd.read_parquet(root / "source_lifecycle_transition_events.parquet")
    failures: list[str] = []
    checked = 0
    final = graph.loc[
        graph["graph_as_of_session"].eq(
            graph.groupby("period")["graph_as_of_session"].transform("max")
        )
    ]
    for row in final.itertuples(index=False):
        cutoff = pd.Timestamp(str(row.graph_freeze_timestamp))
        destination = targets.loc[
            targets["period"].eq(row.period)
            & targets["target_window_sessions"].eq(3)
            & targets["destination_family"].eq(row.destination_family)
            & targets["target_available"]
            & pd.to_datetime(targets["label_availability_timestamp"], utc=True).lt(cutoff)
        ].copy()
        base_support = len(destination)
        base_activations = int(destination["activation_target"].fillna(False).sum())
        source = events.loc[
            events["period"].eq(row.period) & events["destination_family"].eq(row.source_family)
        ][["score_session", row.source_event]].rename(
            columns={row.source_event: "source_event_present"}
        )
        conditioned = destination.merge(
            source,
            left_on="forecast_session",
            right_on="score_session",
            how="left",
            validate="many_to_one",
        )
        conditioned = conditioned.loc[conditioned["source_event_present"].fillna(False)]
        support = len(conditioned)
        activations = int(conditioned["activation_target"].fillna(False).sum())
        edge_rate = (activations + 1.0) / (support + 2.0)
        base_rate = (base_activations + 1.0) / (base_support + 10.0)
        weight = support / (support + 20.0)
        shrunk = math.exp(
            weight
            * float(
                np.clip(
                    math.log(edge_rate / max(base_rate, 1e-12)),
                    -math.log(4.0),
                    math.log(4.0),
                )
            )
        )
        if (
            int(str(row.support)) != support
            or int(str(row.activations)) != activations
            or not math.isclose(
                float(str(row.raw_transition_probability)), edge_rate, abs_tol=1e-12
            )
            or not math.isclose(float(str(row.destination_base_rate)), base_rate, abs_tol=1e-12)
            or not math.isclose(float(str(row.shrunk_lift)), shrunk, abs_tol=1e-12)
        ):
            failures.append(
                f"{row.period}|{row.source_family}|{row.source_event}|{row.destination_family}"
            )
        checked += 1
    return (
        not failures,
        f"independently reconstructed {checked} final graph edges; failures={failures[:3]}",
    )


def _check_forecasts_and_metrics(root: Path) -> tuple[bool, str]:
    forecasts = pd.read_parquet(root / "frozen_destination_forecasts.parquet")
    outcomes = pd.read_parquet(root / "settled_activation_outcome_ledger.parquet")
    forbidden = {
        "activation_target",
        "target_available",
        "target_status",
        "first_activation_session",
        "target_episode_ids",
    }
    if forbidden & set(forecasts.columns):
        return False, "immutable forecast ledger contains settled outcomes"
    if forecasts["forecast_id"].duplicated().any() or outcomes["forecast_id"].duplicated().any():
        return False, "forecast or outcome IDs are not unique"
    if not set(outcomes["forecast_id"]).issubset(set(forecasts["forecast_id"])):
        return False, "outcome references an unknown forecast"
    feature_time = pd.to_datetime(forecasts["feature_availability_timestamp"], utc=True)
    freeze = pd.to_datetime(forecasts["forecast_freeze_timestamp"], utc=True)
    training = pd.to_datetime(forecasts["training_cutoff"], utc=True)
    if (feature_time.notna() & feature_time.gt(freeze)).any() or (
        training.notna() & training.ge(freeze)
    ).any():
        return False, "future feature or target label entered a forecast"
    compared = forecasts.loc[
        forecasts["model_name"].isin(
            [
                "M1_destination_own_history",
                "M2_undirected_system_state",
                "M3_directed_family_rotation",
            ]
        )
    ]
    keys = ["period", "forecast_session", "destination_family", "target_window_sessions"]
    if not compared.groupby(keys, observed=True)["model_name"].nunique().eq(3).all():
        return False, "M1/M2/M3 populations differ"
    schemas = {
        row.model_name: set(json.loads(str(row.feature_schema_json)))
        for row in compared.drop_duplicates("model_name").itertuples(index=False)
    }
    if not (
        schemas["M1_destination_own_history"]
        < schemas["M2_undirected_system_state"]
        < schemas["M3_directed_family_rotation"]
    ):
        return False, "M3 differs from M1 through an unregistered schema"
    scored = forecasts.merge(outcomes, on="forecast_id", how="left", validate="one_to_one")
    primary = scored.loc[
        scored["model_name"].isin(["M1_destination_own_history", "M3_directed_family_rotation"])
        & scored["target_window_sessions"].eq(3)
        & scored["target_available"].fillna(False)
    ]
    pivot = primary.pivot(
        index=keys,
        columns="model_name",
        values="predicted_activation_probability",
    )
    target_series = (
        primary.loc[primary["model_name"].eq("M1_destination_own_history")]
        .set_index(keys)["activation_target"]
        .astype(float)
    )
    target = target_series.loc[pivot.index].to_numpy(float)
    m1 = pivot["M1_destination_own_history"].to_numpy(float)
    m3 = pivot["M3_directed_family_rotation"].to_numpy(float)
    brier = float(np.mean(np.square(m1 - target)) - np.mean(np.square(m3 - target)))
    clipped_m1 = np.clip(m1, 1e-12, 1 - 1e-12)
    clipped_m3 = np.clip(m3, 1e-12, 1 - 1e-12)
    loss1 = -np.mean(target * np.log(clipped_m1) + (1 - target) * np.log(1 - clipped_m1))
    loss3 = -np.mean(target * np.log(clipped_m3) + (1 - target) * np.log(1 - clipped_m3))
    metrics = pd.read_csv(root / "paired_model_metrics.csv")
    row = metrics.loc[
        metrics["comparison"].eq("M3_vs_M1")
        & metrics["target_window_sessions"].eq(3)
        & metrics["period_slice"].astype(str).eq("all")
    ].iloc[0]
    ok = math.isclose(float(row.brier_improvement), brier, abs_tol=1e-12) and math.isclose(
        float(row.log_loss_improvement), float(loss1 - loss3), abs_tol=1e-12
    )
    return bool(ok), f"reconstructed paired primary losses over {len(target)} rows"


def _check_economic_translation(root: Path) -> tuple[bool, str]:
    forecasts = pd.read_parquet(root / "frozen_destination_forecasts.parquet")
    translation = pd.read_parquet(root / "economic_opportunity_translation.parquet")
    eligible = translation.loc[
        translation["economic_translation_status"].eq("eligible_opportunity")
    ].copy()
    if eligible.empty:
        return True, "no eligible translated opportunities; missing rows preserved"
    joined = eligible.merge(
        forecasts[
            ["forecast_id", "destination_family", "forecast_session", "target_window_sessions"]
        ],
        on="forecast_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_forecast"),
    )
    same_family = joined["destination_family"].eq(joined["destination_family_forecast"]).all()
    later = (
        joined["score_session"]
        .astype(str)
        .gt(joined["forecast_session_forecast"].astype(str))
        .all()
    )
    costs = np.isclose(
        pd.to_numeric(joined["gross_payoff_bps"], errors="coerce")
        - pd.to_numeric(joined["primary_total_cost_bps"], errors="coerce"),
        pd.to_numeric(joined["primary_net_payoff_bps"], errors="coerce"),
        atol=1e-10,
        rtol=0.0,
    ).all()
    immutable = (
        ~joined["replacement_opportunity_used"].astype(bool).any()
        and ~joined["capacity_refill_used"].astype(bool).any()
        and not joined.duplicated(["model_name", "opportunity_id"]).any()
    )
    unchanged = joined["existing_position_action"].eq("unchanged_existing_exit_rule").all()
    return bool(same_family and later and costs and immutable and unchanged), (
        f"checked {len(joined)} same-family later opportunities, all costs and clocks"
    )


def _check_manifest(root: Path) -> tuple[bool, str]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    failures = [
        name
        for name, digest in manifest.items()
        if name != "independent_audit.json"
        and (not (root / name).exists() or sha256(root / name) != digest)
    ]
    plots = [name for name in manifest if name.endswith(".png")]
    return not failures and bool(
        plots
    ), f"verified {len(manifest)} hashes and {len(plots)} plots; failures={failures}"


def run_audit(primary: Path, exact: Path) -> dict[str, object]:
    contract: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mapping: dict[str, Any] = json.loads(
        (primary / "frozen_structural_family_mapping.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add(
        "contract_identity",
        metadata["contract_hash"] == sha256(CONTRACT_PATH)
        and contract["contract_id"] == metadata["contract_id"]
        and contract["registered_before_scoring"] is True,
        f"contract={metadata['contract_hash']}",
    )
    input_hashes = _input_hashes(contract)
    expected_inputs = metadata["source_hashes"]
    add(
        "data_snapshot_identity",
        input_hashes == expected_inputs
        and metadata["data_snapshot_id"] == _data_snapshot(contract),
        f"snapshot={metadata['data_snapshot_id']}",
    )
    add(
        "family_mapping_frozen_and_outcome_free",
        mapping["registered_before_scoring"] is True
        and mapping.get("payoff_used_to_define_mapping", False) is False
        and len(mapping["destination_families"]) == 8
        and len(mapping["pairs"]) == 24,
        f"mapping={mapping['mapping_id']}; pairs={len(mapping['pairs'])}",
    )
    state_ok, state_detail = _check_family_state_reconstruction(primary, mapping)
    add("economic_lifecycle_state_reconstruction", state_ok, state_detail)
    event_ok, event_detail = _check_source_events(primary)
    add("source_event_timestamp_reconstruction", event_ok, event_detail)
    target_ok, target_detail = _check_targets(primary)
    add("activation_target_and_calendar_reconstruction", target_ok, target_detail)
    graph_ok, graph_detail = _check_graph(primary)
    add("past_only_graph_and_shrinkage_reconstruction", graph_ok, graph_detail)
    forecast_ok, forecast_detail = _check_forecasts_and_metrics(primary)
    add("forecast_freeze_population_and_metric_reconstruction", forecast_ok, forecast_detail)
    economic_ok, economic_detail = _check_economic_translation(primary)
    add("economic_translation_and_cost_reconstruction", economic_ok, economic_detail)
    pair_graph = pd.read_parquet(primary / "supported_pair_transition_graph.parquet")
    add(
        "unsupported_pair_edges_remain_unknown",
        bool(
            pair_graph.loc[
                (pair_graph["source_event_sessions"] < 20)
                | (pair_graph["destination_activations"] < 4),
                "support_status",
            ]
            .eq("unknown")
            .all()
        ),
        f"pair edges={len(pair_graph)}; supported={int(pair_graph.support_status.eq('supported').sum())}",
    )
    loo = pd.read_csv(primary / "leave_one_stock_out_results.csv")
    stress = pd.read_csv(primary / "stress_test_results.csv")
    blocked_rebuild = stress.loc[
        stress["stress_test"].eq("fully_rebuilt_median_and_leave_one_stock_out")
        & stress["status"].eq("blocked_missing_hash_pinned_v2_rebuild_inputs")
        & stress["result_imputed"].eq(False)
    ]
    add(
        "leave_one_stock_out_rebuild_or_fail_closed_blocker",
        bool(
            (
                not loo.empty
                and loo["all_stock_dependent_states_rebuilt"].astype(bool).all()
                and loo["family_aggregates_rebuilt"].astype(bool).all()
                and loo["transition_graph_rebuilt"].astype(bool).all()
                and loo["activation_labels_rebuilt"].astype(bool).all()
            )
            or (loo.empty and not blocked_rebuild.empty)
        ),
        f"exclusions={len(loo)}; blocked_fail_closed={not blocked_rebuild.empty}",
    )
    nulls = pd.read_csv(primary / "null_test_results.csv")
    add(
        "registered_null_family",
        set(nulls["null_test"])
        == {"wrong_lag_10", "source_permutation", "destination_label_permutation"},
        f"nulls={sorted(nulls['null_test'].astype(str))}",
    )
    concentration = pd.read_csv(primary / "concentration_results.csv")
    concentration_ok = True
    if not concentration.empty:
        sums = concentration.groupby("dimension")["absolute_contribution_share"].sum()
        concentration_ok = bool(np.allclose(sums.to_numpy(float), 1.0, atol=1e-9))
    add("concentration_accounting", concentration_ok, f"rows={len(concentration)}")
    primary_manifest_ok, primary_manifest_detail = _check_manifest(primary)
    exact_manifest_ok, exact_manifest_detail = _check_manifest(exact)
    add("primary_machine_and_plot_hashes", primary_manifest_ok, primary_manifest_detail)
    add("exact_machine_and_plot_hashes", exact_manifest_ok, exact_manifest_detail)
    identity = verify_exact_machine_identity(primary, exact)
    recorded_identity = json.loads(
        (exact / "exact_rerun_identity.json").read_text(encoding="utf-8")
    )
    add(
        "primary_exact_rerun_identity",
        bool(identity["byte_identical"]) and bool(recorded_identity["byte_identical"]),
        json.dumps(identity, sort_keys=True),
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            f"{contract['frozen_lineage']['starting_commit']}..{metadata['git_sha']}",
        ],
        cwd=REPO,
        text=True,
    ).splitlines()
    prohibited = prohibited_changed_paths(changed)
    safety = contract["safety"]
    add(
        "git_identity_and_research_only_safety",
        head == metadata["git_sha"]
        and not prohibited
        and safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["broker_connection_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["position_management_changed"] is False
        and safety["existing_exit_logic_changed"] is False,
        f"git={head}; changed={len(changed)}; prohibited={prohibited}",
    )
    schema = json.loads(
        (primary / "prospective_immutable_forecast_ledger_schema.json").read_text(encoding="utf-8")
    )
    add(
        "prospective_logger_execution_free",
        schema["execution_enabled"] is False
        and schema["forecast_create_only"] is True
        and schema["outcome_create_only"] is True,
        f"schema={schema['schema_version']}",
    )
    failed = [str(row["name"]) for row in checks if not bool(row["passed"])]
    return {
        "audit_id": "20260716-directed-economic-loop-regime-rotation-v1-independent-audit",
        "status": "pass" if not failed else "fail",
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "independent_reconstruction": {
            "target_windows": True,
            "family_state_aggregation": True,
            "source_event_timestamps": True,
            "past_only_graph_counts": True,
            "beta_smoothing_and_shrinkage": True,
            "paired_primary_losses": True,
            "economic_cost_and_identity_joins": True,
        },
        "research_only": True,
    }


def write_audit_and_update_manifest(root: Path, audit: Mapping[str, object]) -> None:
    audit_path = root / "independent_audit.json"
    _write_json(audit_path, audit)
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[audit_path.name] = sha256(audit_path)
    _write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-rerun", type=Path, default=DEFAULT_EXACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = run_audit(Path(args.primary), Path(args.exact_rerun))
    write_audit_and_update_manifest(Path(args.primary), audit)
    write_audit_and_update_manifest(Path(args.exact_rerun), audit)
    print(json.dumps(_json_safe(audit), sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
