"""Independent audit for causal payoff-model path development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1-pre-score.json"
RUNNER_PATH = HERE / "run_causal_payoff_model_paths_v1.py"
SEED = 20260714
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
ROUND_TRIP_COST_BPS = 10.0
Z_90 = 1.6448536269514722
WARMUP_SESSIONS = 60
ROUTE_CLASSES = (
    "no_transition",
    "expected_leg_partial",
    "exact_parent_completion",
    "incompatible_first_transition",
    "expected_leg_then_diversion",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def source_path(name: str, contract: dict[str, Any]) -> Path:
    if name == "contract":
        return CONTRACT_PATH
    if name == "runner":
        return RUNNER_PATH
    if name == "auditor":
        return Path(__file__).resolve()
    if name == "tests":
        return HERE / "tests/test_causal_payoff_model_paths_v1.py"
    if name.startswith("provider_2024_"):
        symbol = name.removeprefix("provider_2024_")
        root = Path(contract["inputs"]["provider_root_2024"])
        return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
    mapping = {
        "base_loader_runner": "base_loader_runner",
        "accepted_setup_signals_2024": "accepted_setup_signals_2024",
        "anchor_panel_2024": "anchor_panel_2024",
        "state_runs_2024": "state_runs_2024",
        "fixed_cycles": "fixed_cycles",
        "parent_report": "parent_report",
        "parent_handoff": "parent_handoff",
        "prospective_log_contract_v2": "prospective_log_contract_v2",
    }
    return Path(contract["inputs"][mapping[name]])


def load_provider(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame = frame.loc[pd.to_datetime(frame["session_date"]).dt.year.eq(2024)].copy()
    return {
        str(session): group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for session, group in frame.groupby("session_date", sort=False)
    }


def load_runs(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    columns = ["symbol_norm", "session_date", "state", "start_pos", "end_pos"]
    frame = pd.read_csv(path, usecols=columns)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    for column in ("state", "start_pos", "end_pos"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return {
        (str(symbol), str(session)): group.sort_values("start_pos", kind="stable").reset_index(
            drop=True
        )
        for (symbol, session), group in frame.groupby(["symbol_norm", "session_date"], sort=False)
    }


def topology(transitions: list[tuple[int, int]], anchor: int, alternate: int) -> str:
    if not transitions:
        return "no_transition"
    if transitions[0][0] != alternate:
        return "incompatible_first_transition"
    if len(transitions) == 1:
        return "expected_leg_partial"
    if transitions[1][0] == anchor:
        return "exact_parent_completion"
    return "expected_leg_then_diversion"


def pre_entry_status(
    transitions: list[tuple[int, int]], entry: int, anchor: int, alternate: int
) -> str:
    known = [(state, position) for state, position in transitions if position < entry]
    label = topology(known, anchor, alternate)
    return {
        "no_transition": "orientation_intact",
        "expected_leg_partial": "expected_leg_active",
        "exact_parent_completion": "completed_before_entry",
        "incompatible_first_transition": "invalidated_before_entry",
        "expected_leg_then_diversion": "invalidated_before_entry",
    }[label]


def route_status(
    transitions: list[tuple[int, int]], checkpoint: int, anchor: int, alternate: int
) -> str:
    known = [(state, position) for state, position in transitions if position <= checkpoint]
    label = topology(known, anchor, alternate)
    return {
        "no_transition": "orientation_intact",
        "expected_leg_partial": "expected_leg_active",
        "exact_parent_completion": "exact_parent_completion_detected",
        "incompatible_first_transition": "incompatible_first_transition_detected",
        "expected_leg_then_diversion": "expected_leg_then_diversion_detected",
    }[label]


def gross_bps(direction: int, exit_price: float, entry_price: float) -> float:
    return 10000.0 * direction * (exit_price / entry_price - 1.0)


def moving_block_bootstrap(
    frame: pd.DataFrame, value_column: str, seed_offset: int
) -> dict[str, float]:
    sessions = sorted(frame["session_date"].astype(str).unique())
    by_session = {
        date: group[value_column].to_numpy(float)
        for date, group in frame.groupby("session_date", sort=False)
    }
    rng = np.random.default_rng(SEED + seed_offset)
    count = len(sessions)
    blocks = int(math.ceil(count / BOOTSTRAP_BLOCK))
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, count, size=blocks)
        sampled = [
            sessions[(int(start) + offset) % count]
            for start in starts
            for offset in range(BOOTSTRAP_BLOCK)
        ][:count]
        values[draw] = float(np.mean(np.concatenate([by_session[date] for date in sampled])))
    return {
        "observed_mean": float(frame[value_column].mean()),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.0)) / (len(values) + 1)),
    }


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def group_slice(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return frame if group == "pooled" else frame.loc[frame["candidate"].eq(group)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.artifact / "independent_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite {audit_path}")

    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    source_hashes = json.loads((args.artifact / "source_hashes.json").read_text())
    decision = json.loads((args.artifact / "decision.json").read_text())
    surface = pd.read_parquet(args.artifact / "research_surface.parquet")
    route = pd.read_parquet(args.artifact / "route_predictions.parquet")
    admission = pd.read_parquet(args.artifact / "admission_predictions.parquet")
    snapshots = pd.read_parquet(args.artifact / "causal_snapshots.parquet")
    snapshot_predictions = pd.read_parquet(args.artifact / "snapshot_predictions.parquet")
    policies = pd.read_parquet(args.artifact / "sequential_policy_rows.parquet")

    safety = contract["safety"]
    seal = contract["sealed_data_status"]
    safety_ok = bool(
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["application_code_modification_allowed"] is False
        and safety["repository_write_allowed"] is False
        and seal["genuinely_unseen_sessions_available"] is False
        and seal["validation_claim_allowed"] is False
        and seal["diversion_specific_hypothesis_test_allowed"] is False
        and decision["sealed_validation_performed"] is False
        and decision["economic_edge_claim"] is False
        and decision["strategy_promotion"] is False
    )

    current_hashes = {name: sha256(source_path(name, contract)) for name in pre_score["sha256"]}
    hashes_ok = current_hashes == pre_score["sha256"] == source_hashes["sha256"]

    expected_counts = {"cycle_04|state4": 259, "cycle_07|state5": 432}
    population_errors = sum(
        int(len(surface.loc[surface["candidate"].eq(candidate)]) != expected)
        for candidate, expected in expected_counts.items()
    )
    calendar = sorted(surface["session_date"].unique())
    score_dates = calendar[WARMUP_SESSIONS:]
    population_errors += int(len(calendar) != 128 or len(score_dates) != 68)

    causal_training_errors = 0
    for frame in (route, admission, snapshot_predictions):
        causal_training_errors += int(
            (
                ~(
                    pd.to_datetime(frame["train_last_session"])
                    < pd.to_datetime(frame["session_date"])
                )
            ).sum()
        )

    route_probability_columns = [f"route_probability__{name}" for name in ROUTE_CLASSES]
    route_prior_columns = [f"route_prior__{name}" for name in ROUTE_CLASSES]
    probability_errors = int(
        (~np.isclose(route[route_probability_columns].sum(axis=1), 1.0, atol=1e-12)).sum()
        + (~np.isclose(route[route_prior_columns].sum(axis=1), 1.0, atol=1e-12)).sum()
        + (route[route_probability_columns].lt(0.004 - 1e-12).any(axis=1)).sum()
        + (~route["actual_route"].isin(ROUTE_CLASSES)).sum()
    )

    admission_mean = admission["predicted_net_bps"].to_numpy(float)
    admission_std = admission["predictive_std_bps"].to_numpy(float)
    admission_expected_class = np.where(
        admission_mean - Z_90 * admission_std > 0,
        "positive",
        np.where(
            admission_mean + Z_90 * admission_std < 0,
            "negative",
            "unknown_abstain",
        ),
    )
    snapshot_mean = snapshot_predictions["predicted_hold_advantage_bps"].to_numpy(float)
    snapshot_std = snapshot_predictions["predictive_std_bps"].to_numpy(float)
    snapshot_expected_class = np.where(
        snapshot_mean - Z_90 * snapshot_std > 0,
        "positive_hold",
        np.where(
            snapshot_mean + Z_90 * snapshot_std < 0,
            "negative_exit",
            "unknown_abstain",
        ),
    )
    class_errors = int(
        np.sum(admission_expected_class != admission["uncertainty_class"].to_numpy(str))
        + np.sum(snapshot_expected_class != snapshot_predictions["uncertainty_class"].to_numpy(str))
        + np.sum((admission_mean > 0) != admission["point_positive"].to_numpy(bool))
        + np.sum((snapshot_mean < 0) != snapshot_predictions["point_negative_exit"].to_numpy(bool))
    )

    forbidden_tokens = (
        "gross",
        "net_return",
        "future",
        "mfe",
        "mae",
        "child",
        "morph",
        "topology_outcome",
    )
    allowed_features = [
        *contract["admission_features"]["numeric"],
        *contract["admission_features"]["categorical"],
    ]
    forbidden_feature_hits = [
        name
        for name in allowed_features
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    diversion_artifacts = [
        path.name
        for path in args.artifact.iterdir()
        if "diversion" in path.name.lower() and path.is_file()
    ]

    provider_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    root = Path(contract["inputs"]["provider_root_2024"])
    for symbol in contract["population"]["symbols"]:
        sessions = load_provider(root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet")
        for session, bars in sessions.items():
            provider_lookup[(symbol, session)] = bars
    runs = load_runs(Path(contract["inputs"]["state_runs_2024"]))

    route_errors = 0
    pre_entry_errors = 0
    replay_max_error = 0.0
    transition_cache: dict[str, list[tuple[int, int]]] = {}
    surface_by_signal = surface.set_index("signal_id")
    for row in surface.itertuples(index=False):
        key = (str(row.symbol_norm), str(row.session_date))
        bars = provider_lookup[key]
        state_runs = runs[key]
        transitions_frame = state_runs.loc[
            state_runs["start_pos"].gt(int(row.anchor_state_position))
            & state_runs["start_pos"].le(int(row.frozen_exit_state_position))
        ].sort_values("start_pos", kind="stable")
        transitions = [
            (int(item.state), int(item.start_pos))
            for item in transitions_frame.itertuples(index=False)
        ]
        transition_cache[str(row.signal_id)] = transitions
        route_errors += int(
            topology(
                transitions,
                int(row.anchor_state),
                int(row.expected_alternate_state),
            )
            != row.route_topology_outcome_only
        )
        pre_entry_errors += int(
            pre_entry_status(
                transitions,
                int(row.entry_state_position),
                int(row.anchor_state),
                int(row.expected_alternate_state),
            )
            != row.pre_entry_path_status
        )
        fixed_price = float(bars.iloc[int(row.frozen_exit_ordinal)]["close"])
        replay = gross_bps(int(row.direction), fixed_price, float(row.entry_price))
        replay_max_error = max(
            replay_max_error,
            abs(fixed_price - float(row.fixed_exit_price)),
            abs(replay - float(row.fixed_gross_bps)),
            abs(replay - ROUND_TRIP_COST_BPS - float(row.fixed_net_bps)),
        )

    snapshot_clock_errors = 0
    snapshot_route_errors = 0
    snapshot_replay_max_error = 0.0
    for row in snapshots.itertuples(index=False):
        signal = surface_by_signal.loc[str(row.signal_id)]
        bars = provider_lookup[(str(row.symbol_norm), str(row.session_date))]
        snapshot_clock_errors += int(
            int(row.next_open_ordinal) != int(row.checkpoint_ordinal) + 1
            or int(row.checkpoint_ordinal) <= int(signal.entry_ordinal)
            or int(row.next_open_ordinal) > int(signal.frozen_exit_ordinal)
        )
        checkpoint_position = int(signal.entry_state_position) + int(row.checkpoint_offset)
        expected_status = route_status(
            transition_cache[str(row.signal_id)],
            checkpoint_position,
            int(signal.anchor_state),
            int(signal.expected_alternate_state),
        )
        snapshot_route_errors += int(expected_status != row.causal_route_status)
        next_price = float(bars.iloc[int(row.next_open_ordinal)]["open"])
        replay = gross_bps(int(signal.direction), next_price, float(signal.entry_price))
        snapshot_replay_max_error = max(
            snapshot_replay_max_error,
            abs(next_price - float(row.next_open_price)),
            abs(replay - float(row.next_open_gross_bps)),
            abs(float(signal.fixed_gross_bps) - replay - float(row.hold_advantage_bps)),
        )

    policy_errors = 0
    prediction_groups = {
        (str(signal_id), str(model)): group.sort_values("checkpoint_ordinal", kind="stable")
        for (signal_id, model), group in snapshot_predictions.groupby(
            ["signal_id", "model"], sort=False
        )
    }
    for row in policies.itertuples(index=False):
        group = prediction_groups.get((str(row.signal_id), str(row.model)), pd.DataFrame())
        if row.policy_type == "uncertainty_aware":
            actions = (
                group.loc[group["uncertainty_class"].eq("negative_exit")]
                if not group.empty
                else group
            )
        else:
            actions = group.loc[group["point_negative_exit"].eq(True)] if not group.empty else group
        acted = not actions.empty
        expected_action_ordinal = int(actions.iloc[0]["next_open_ordinal"]) if acted else -1
        expected_gross = (
            float(actions.iloc[0]["next_open_gross_bps"]) if acted else float(row.fixed_gross_bps)
        )
        policy_errors += int(
            bool(row.action) != acted
            or int(row.action_next_open_ordinal) != expected_action_ordinal
            or not np.isclose(float(row.policy_gross_bps), expected_gross, atol=1e-12)
            or not np.isclose(
                float(row.policy_net_bps), expected_gross - ROUND_TRIP_COST_BPS, atol=1e-12
            )
            or not np.isclose(
                float(row.paired_difference_bps),
                expected_gross - ROUND_TRIP_COST_BPS - float(row.fixed_net_bps),
                atol=1e-12,
            )
        )

    aggregate_count_errors = 0
    aggregate_max_error = 0.0
    route_metrics = pd.read_csv(args.artifact / "route_forecast_metrics.csv")
    score_route = route.loc[route["calendar_index"].ge(WARMUP_SESSIONS)]
    for metric in route_metrics.itertuples(index=False):
        group = group_slice(score_route, str(metric.group))
        aggregate_count_errors += int(len(group) != int(metric.rows))
        aggregate_max_error = max(
            aggregate_max_error,
            abs(float(group["model_log_loss"].mean()) - float(metric.model_log_loss)),
            abs(float(group["prior_log_loss"].mean()) - float(metric.prior_log_loss)),
            abs(float(group["log_loss_improvement"].mean()) - float(metric.log_loss_improvement)),
        )

    admission_selectors = pd.read_csv(args.artifact / "admission_selector_metrics.csv")
    for metric in admission_selectors.itertuples(index=False):
        group = group_slice(admission.loc[admission["model"].eq(metric.model)], str(metric.group))
        selected = (
            group["uncertainty_class"].eq("positive")
            if metric.policy_type == "uncertainty_aware"
            else group["point_positive"].astype(bool)
        )
        selector_return = np.where(selected, group["fixed_net_bps"], 0.0)
        aggregate_count_errors += int(
            len(group) != int(metric.rows) or int(selected.sum()) != int(metric.selected_rows)
        )
        aggregate_max_error = max(
            aggregate_max_error,
            abs(float(selected.mean()) - float(metric.coverage)),
            abs(float(np.mean(selector_return)) - float(metric.selector_mean_per_opportunity_bps)),
            abs(
                float(np.mean(selector_return - group["fixed_net_bps"]))
                - float(metric.paired_difference_bps)
            ),
        )

    sequential_metrics = pd.read_csv(args.artifact / "sequential_policy_metrics.csv")
    for metric in sequential_metrics.itertuples(index=False):
        group = group_slice(
            policies.loc[
                policies["model"].eq(metric.model) & policies["policy_type"].eq(metric.policy_type)
            ],
            str(metric.group),
        )
        aggregate_count_errors += int(
            len(group) != int(metric.rows) or int(group["action"].sum()) != int(metric.action_rows)
        )
        aggregate_max_error = max(
            aggregate_max_error,
            abs(float(group["action"].mean()) - float(metric.action_coverage)),
            abs(float(group["policy_net_bps"].mean()) - float(metric.policy_mean_net_bps)),
            abs(float(group["paired_difference_bps"].mean()) - float(metric.paired_difference_bps)),
        )

    bootstrap_max_error = 0.0
    holm_max_error = 0.0
    route_boot = pd.read_csv(args.artifact / "route_bootstraps.csv")
    for index, metric in route_boot.reset_index(drop=True).iterrows():
        group = group_slice(score_route, str(metric["group"]))
        expected = moving_block_bootstrap(group, "log_loss_improvement", index)
        bootstrap_max_error = max(
            bootstrap_max_error,
            *(abs(expected[name] - float(metric[name])) for name in expected),
        )
    holm_max_error = max(
        holm_max_error,
        float(
            np.max(
                np.abs(
                    holm(route_boot["p_one_sided"].to_numpy(float))
                    - route_boot["holm_adjusted_p"].to_numpy(float)
                )
            )
        ),
    )

    admission_boot = pd.read_csv(args.artifact / "admission_bootstraps.csv")
    for family, seed_start in (
        ("direct_admission", 100),
        ("route_augmented_admission", 103),
    ):
        family_frame = admission_boot.loc[admission_boot["family"].eq(family)].reset_index(
            drop=True
        )
        model = (
            "admission_only"
            if family == "direct_admission"
            else "admission_plus_causal_route_probabilities"
        )
        model_frame = admission.loc[admission["model"].eq(model)].copy()
        model_frame["selected"] = model_frame["uncertainty_class"].eq("positive")
        model_frame["paired"] = (
            np.where(model_frame["selected"], model_frame["fixed_net_bps"], 0.0)
            - model_frame["fixed_net_bps"]
        )
        for index, metric in family_frame.iterrows():
            group = group_slice(model_frame, str(metric["group"]))
            expected = moving_block_bootstrap(group, "paired", seed_start + index)
            bootstrap_max_error = max(
                bootstrap_max_error,
                *(abs(expected[name] - float(metric[name])) for name in expected),
            )
        holm_max_error = max(
            holm_max_error,
            float(
                np.max(
                    np.abs(
                        holm(family_frame["p_one_sided"].to_numpy(float))
                        - family_frame["holm_adjusted_p"].to_numpy(float)
                    )
                )
            ),
        )

    sequential_boot = pd.read_csv(args.artifact / "sequential_bootstraps.csv")
    for family, seed_start in (
        ("sequential_route_plus_price", 200),
        ("sequential_route_only", 203),
    ):
        family_frame = sequential_boot.loc[sequential_boot["family"].eq(family)].reset_index(
            drop=True
        )
        model = (
            "sequential_route_plus_price_path"
            if family == "sequential_route_plus_price"
            else "sequential_route_state_only"
        )
        model_frame = policies.loc[
            policies["model"].eq(model) & policies["policy_type"].eq("uncertainty_aware")
        ]
        for index, metric in family_frame.iterrows():
            group = group_slice(model_frame, str(metric["group"]))
            expected = moving_block_bootstrap(group, "paired_difference_bps", seed_start + index)
            bootstrap_max_error = max(
                bootstrap_max_error,
                *(abs(expected[name] - float(metric[name])) for name in expected),
            )
        holm_max_error = max(
            holm_max_error,
            float(
                np.max(
                    np.abs(
                        holm(family_frame["p_one_sided"].to_numpy(float))
                        - family_frame["holm_adjusted_p"].to_numpy(float)
                    )
                )
            ),
        )

    manifest = json.loads((args.artifact / "artifact_manifest.json").read_text())
    manifest_errors = 0
    for item in manifest["files"]:
        path = args.artifact / item["name"]
        manifest_errors += int(
            not path.exists()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != item["sha256"]
        )

    checks = {
        "research_only_and_unsealed_status": safety_ok,
        "pre_score_and_artifact_source_hashes_match": hashes_ok,
        "frozen_population_and_calendar_match": population_errors == 0,
        "all_training_sessions_strictly_precede_scores": causal_training_errors == 0,
        "route_probabilities_valid_and_floored": probability_errors == 0,
        "uncertainty_and_point_classes_replay": class_errors == 0,
        "admission_feature_allowlist_has_no_outcomes": not forbidden_feature_hits,
        "diversion_specific_payoff_artifact_absent": not diversion_artifacts,
        "route_topology_and_pre_entry_clock_replay": route_errors == 0 and pre_entry_errors == 0,
        "fixed_payoffs_replay": replay_max_error <= 1e-8,
        "snapshot_next_open_clock_and_route_replay": snapshot_clock_errors == 0
        and snapshot_route_errors == 0,
        "snapshot_payoffs_replay": snapshot_replay_max_error <= 1e-8,
        "sequential_policies_use_earliest_causal_class": policy_errors == 0,
        "aggregate_metrics_replay": aggregate_count_errors == 0 and aggregate_max_error <= 1e-10,
        "bootstrap_and_holm_replay": bootstrap_max_error <= 1e-10 and holm_max_error <= 1e-10,
        "artifact_manifest_complete_and_valid": manifest_errors == 0,
    }
    payload = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "errors": {
            "population": population_errors,
            "causal_training": causal_training_errors,
            "probabilities": probability_errors,
            "classes": class_errors,
            "forbidden_feature_hits": forbidden_feature_hits,
            "diversion_artifacts": diversion_artifacts,
            "route": route_errors,
            "pre_entry": pre_entry_errors,
            "snapshot_clock": snapshot_clock_errors,
            "snapshot_route": snapshot_route_errors,
            "policies": policy_errors,
            "aggregate_counts": aggregate_count_errors,
            "manifest": manifest_errors,
        },
        "maximum_errors": {
            "fixed_payoff_replay": replay_max_error,
            "snapshot_payoff_replay": snapshot_replay_max_error,
            "aggregate_metric_replay": aggregate_max_error,
            "bootstrap_replay": bootstrap_max_error,
            "holm_replay": holm_max_error,
        },
    }
    write_json(audit_path, payload)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "passed": payload["passed"],
                "total": payload["total"],
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
