"""Independent audit for the research-only causal loop state-path artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-causal-loop-state-path-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260714-causal-loop-state-path-v1-pre-score.json"
RUNNER_PATH = HERE / "run_causal_loop_state_path_v1.py"
SEED = 20260714
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
ROUND_TRIP_COST = 10.0

FIXED_POLICY = "fixed_anchor_h24_close"
TERMINAL_POLICY = "terminal_route_event_next_open_else_fixed"
COMPLETION_POLICY = "exact_completion_next_open_else_fixed"
INVALIDATION_POLICY = "route_invalidation_next_open_else_fixed"
TRANSITION_POLICY = "first_post_entry_transition_next_open_else_fixed"
ALT_POLICIES = (TERMINAL_POLICY, COMPLETION_POLICY, INVALIDATION_POLICY, TRANSITION_POLICY)


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


def load_runs(path: Path) -> pd.DataFrame:
    columns = ["symbol_norm", "session_date", "state", "start_pos", "end_pos", "start_timestamp"]
    frame = (
        pd.read_parquet(path, columns=columns)
        if path.suffix == ".parquet"
        else pd.read_csv(path, usecols=columns)
    )
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    for column in ("state", "start_pos", "end_pos"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True, errors="raise")
    return frame.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)


def load_provider(path: Path, period: int) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path, columns=["timestamp", "open", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame = frame.loc[pd.to_datetime(frame["session_date"]).dt.year.eq(period)].copy()
    return {
        str(session): group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for session, group in frame.groupby("session_date", sort=False)
    }


def independent_topology(
    transitions: list[tuple[int, int]], anchor: int, alternate: int
) -> tuple[str, int | None, int | None, str | None, int | None]:
    if len(transitions) == 0:
        return "no_transition", None, None, None, None
    first_state, first_position = transitions[0]
    if first_state != alternate:
        return "incompatible_first_transition", None, first_position, "invalidation", first_position
    if len(transitions) == 1:
        return "expected_leg_partial", None, None, None, None
    second_state, second_position = transitions[1]
    if second_state == anchor:
        return "exact_parent_completion", second_position, None, "completion", second_position
    return "expected_leg_then_diversion", None, second_position, "invalidation", second_position


def independent_pre_entry_status(
    transitions: list[tuple[int, int]], entry_position: int, anchor: int, alternate: int
) -> str:
    known = [(state, position) for state, position in transitions if position < entry_position]
    topology = independent_topology(known, anchor, alternate)[0]
    if topology == "no_transition":
        return "orientation_intact"
    if topology == "expected_leg_partial":
        return "expected_leg_active"
    if topology == "exact_parent_completion":
        return "completed_before_entry"
    return "invalidated_before_entry"


def gross_bps(direction: int, price: float, entry: float) -> float:
    return 10000.0 * direction * (price / entry - 1.0)


def independent_policy(
    event_position: int | None,
    entry_position: int,
    anchor_position: int,
    tape_anchor: int,
    entry_ordinal: int,
    fixed_ordinal: int,
    bars: pd.DataFrame,
    direction: int,
    entry_price: float,
) -> dict[str, Any]:
    detection = None if event_position is None else tape_anchor + event_position - anchor_position
    next_open = None if detection is None else detection + 1
    actionable = bool(
        event_position is not None
        and event_position >= entry_position
        and detection is not None
        and detection >= entry_ordinal
        and next_open is not None
        and next_open <= fixed_ordinal
        and next_open < len(bars)
    )
    exit_ordinal = int(next_open) if actionable else fixed_ordinal
    price_column = "open" if actionable else "close"
    exit_price = float(bars.iloc[exit_ordinal][price_column])
    gross = gross_bps(direction, exit_price, entry_price)
    return {
        "detection": detection
        if event_position is not None and event_position >= entry_position
        else None,
        "next_open": next_open
        if event_position is not None and event_position >= entry_position
        else None,
        "actionable": actionable,
        "exit_ordinal": exit_ordinal,
        "exit_price": exit_price,
        "gross": gross,
        "net": gross - ROUND_TRIP_COST,
    }


def moving_block_bootstrap(group: pd.DataFrame, column: str, seed_offset: int) -> dict[str, float]:
    sessions = sorted(group["session_date"].astype(str).unique())
    by_session = {
        date: block[column].to_numpy(float)
        for date, block in group.groupby("session_date", sort=False)
    }
    rng = np.random.default_rng(SEED + seed_offset)
    n = len(sessions)
    block_count = int(math.ceil(n / BOOTSTRAP_BLOCK))
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, n, size=block_count)
        sampled = [
            sessions[(int(start) + offset) % n]
            for start in starts
            for offset in range(BOOTSTRAP_BLOCK)
        ][:n]
        values[draw] = float(np.mean(np.concatenate([by_session[date] for date in sampled])))
    return {
        "paired_mean_difference_bps": float(group[column].mean()),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.0)) / (len(values) + 1)),
    }


def holm_values(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def audit_runner_event_logic() -> dict[str, Any]:
    tree = ast.parse(RUNNER_PATH.read_text())
    functions = {
        node.name: ast.get_source_segment(RUNNER_PATH.read_text(), node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    required = {"topology_from_transitions", "pre_entry_status", "event_policy"}
    if not required.issubset(functions):
        raise AssertionError("event functions missing")
    structural_source = functions["topology_from_transitions"] + functions["pre_entry_status"]
    forbidden = ["mfe", "mae", "final_positive", "path_class", "child", "morph", "future"]
    hits = [token for token in forbidden if token in structural_source.lower()]
    event_args = {
        argument.arg
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "event_policy"
        for argument in node.args.args
    }
    forbidden_event_args = sorted(
        name
        for name in event_args
        if any(
            token in name.lower() for token in ("mfe", "mae", "payoff", "future", "child", "morph")
        )
    )
    return {
        "structural_forbidden_token_hits": hits,
        "forbidden_event_arguments": forbidden_event_args,
        "passes": not hits and not forbidden_event_args,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.artifact / "independent_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    contract = json.loads(CONTRACT_PATH.read_text())
    events = pd.read_parquet(args.artifact / "signal_path_events.parquet")
    source_hashes = json.loads((args.artifact / "source_hashes.json").read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    decision = json.loads((args.artifact / "decision.json").read_text())

    safety = contract["safety"]
    safety_ok = bool(
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["application_code_modification_allowed"] is False
        and safety["repository_write_allowed"] is False
        and decision["economic_edge_claim"] is False
        and decision["strategy_promotion"] is False
        and decision["sealed_validation_performed"] is False
    )

    current_hashes = {}
    for name, expected_hash in pre_score["sha256"].items():
        if name == "contract":
            path = CONTRACT_PATH
        elif name == "runner":
            path = RUNNER_PATH
        elif name == "auditor":
            path = Path(__file__).resolve()
        elif name == "tests":
            path = HERE / "tests/test_causal_loop_state_path_v1.py"
        elif name.startswith("provider_"):
            _, period_text, symbol = name.split("_", 2)
            root = Path(contract["inputs"]["provider_roots"][period_text])
            path = root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        else:
            mapping = {
                "base_loader_runner": contract["inputs"]["base_loader_runner"],
                "v2_signal_level_artifact": contract["inputs"]["v2_signal_level_artifact"],
                "v2_report": contract["inputs"]["v2_report"],
                "v2_handoff": contract["inputs"]["v2_handoff"],
                "v2_prospective_log_contract": contract["inputs"]["v2_prospective_log_contract"],
                "fixed_cycles": contract["inputs"]["fixed_cycles"],
                "runs_2023": contract["inputs"]["runs"]["2023"],
                "runs_2025": contract["inputs"]["runs"]["2025"],
            }
            path = Path(mapping[name])
        current_hashes[name] = sha256(path)
        if current_hashes[name] != expected_hash:
            raise AssertionError(f"frozen source changed: {name}")
    hashes_ok = current_hashes == source_hashes["sha256"] == pre_score["sha256"]

    runs_lookup: dict[tuple[int, str, str], pd.DataFrame] = {}
    for period in (2023, 2025):
        runs = load_runs(Path(contract["inputs"]["runs"][str(period)]))
        for (symbol, session), group in runs.groupby(["symbol_norm", "session_date"], sort=False):
            runs_lookup[(period, str(symbol), str(session))] = group.reset_index(drop=True)
    provider_lookup: dict[tuple[int, str, str], pd.DataFrame] = {}
    for period in (2023, 2025):
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            by_session = load_provider(
                root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet", period
            )
            for session, bars in by_session.items():
                provider_lookup[(period, symbol, session)] = bars

    topology_errors = 0
    pre_entry_errors = 0
    timestamp_errors = 0
    action_errors = 0
    same_bar_errors = 0
    pre_entry_action_errors = 0
    max_fixed_error = 0.0
    max_policy_price_error = 0.0
    max_policy_return_error = 0.0

    for row in events.itertuples(index=False):
        period = int(row.period)
        key = (period, str(row.symbol_norm), str(row.session_date))
        state_runs = runs_lookup[key]
        bars = provider_lookup[key]
        anchor_position = int(row.anchor_state_position)
        entry_position = int(row.entry_state_position)
        frozen_position = int(row.frozen_exit_state_position)
        transitions_frame = state_runs.loc[
            state_runs["start_pos"].gt(anchor_position)
            & state_runs["start_pos"].le(frozen_position)
        ].sort_values("start_pos", kind="stable")
        transitions = [
            (int(item.state), int(item.start_pos))
            for item in transitions_frame.itertuples(index=False)
        ]
        topology, completion, invalidation, terminal_type, terminal = independent_topology(
            transitions, int(row.anchor_state), int(row.expected_alternate_state)
        )
        if topology != row.path_topology or terminal_type != row.terminal_event_type:
            topology_errors += 1
        if (
            independent_pre_entry_status(
                transitions,
                entry_position,
                int(row.anchor_state),
                int(row.expected_alternate_state),
            )
            != row.pre_entry_path_status
        ):
            pre_entry_errors += 1
        first_post = next(
            (position for _, position in transitions if position >= entry_position), None
        )
        event_positions = {
            TERMINAL_POLICY: terminal,
            COMPLETION_POLICY: completion,
            INVALIDATION_POLICY: invalidation,
            TRANSITION_POLICY: first_post,
        }
        tape_anchor = int(row.tape_anchor_ordinal)
        entry_ordinal = int(row.entry_ordinal)
        fixed_ordinal = int(row.frozen_exit_ordinal)
        if pd.Timestamp(bars.iloc[tape_anchor]["timestamp"]) != pd.Timestamp(row.start_timestamp):
            timestamp_errors += 1
        fixed_price = float(bars.iloc[fixed_ordinal]["close"])
        fixed_gross = gross_bps(int(row.direction), fixed_price, float(row.entry_price))
        max_fixed_error = max(
            max_fixed_error,
            abs(fixed_price - float(row.fixed_exit_price)),
            abs(fixed_gross - float(row.fixed_gross_bps)),
            abs(fixed_gross - ROUND_TRIP_COST - float(row.fixed_net_bps)),
        )
        row_map = row._asdict()
        for policy, position in event_positions.items():
            replay = independent_policy(
                position,
                entry_position,
                anchor_position,
                tape_anchor,
                entry_ordinal,
                fixed_ordinal,
                bars,
                int(row.direction),
                float(row.entry_price),
            )
            prefix = policy
            stored_action = bool(row_map[f"{prefix}__actionable"])
            if stored_action != replay["actionable"]:
                action_errors += 1
            if stored_action:
                stored_detection = int(row_map[f"{prefix}__detection_ordinal"])
                stored_exit = int(row_map[f"{prefix}__exit_ordinal"])
                if stored_exit != stored_detection + 1:
                    same_bar_errors += 1
                if position is None or position < entry_position:
                    pre_entry_action_errors += 1
                run_match = state_runs.loc[state_runs["start_pos"].eq(position)]
                if len(run_match) != 1:
                    timestamp_errors += 1
                else:
                    run_timestamp = pd.Timestamp(run_match.iloc[0]["start_timestamp"])
                    bar_timestamp = pd.Timestamp(bars.iloc[stored_detection]["timestamp"])
                    if run_timestamp != bar_timestamp:
                        timestamp_errors += 1
            max_policy_price_error = max(
                max_policy_price_error,
                abs(float(row_map[f"{prefix}__exit_price"]) - replay["exit_price"]),
            )
            max_policy_return_error = max(
                max_policy_return_error,
                abs(float(row_map[f"{prefix}__gross_bps"]) - replay["gross"]),
                abs(float(row_map[f"{prefix}__net_bps"]) - replay["net"]),
                abs(
                    float(row_map[f"{prefix}__paired_difference_bps"])
                    - (replay["net"] - (fixed_gross - ROUND_TRIP_COST))
                ),
            )

    expected_counts = {
        (2023, "cycle_04|state4"): 132,
        (2023, "cycle_07|state5"): 722,
        (2025, "cycle_04|state4"): 96,
        (2025, "cycle_07|state5"): 713,
    }
    population_errors = 0
    for key, expected in expected_counts.items():
        actual = len(
            events.loc[
                events["period"].eq(key[0])
                & events["candidate"].eq(key[1])
                & events["role"].eq("primary_candidate")
            ]
        )
        population_errors += int(actual != expected)

    policy_metrics = pd.read_csv(args.artifact / "exit_policy_metrics.csv")
    max_aggregate_error = 0.0
    aggregate_count_errors = 0
    for metric in policy_metrics.itertuples(index=False):
        group = events.loc[
            events["period"].eq(int(metric.period))
            & events["candidate"].eq(metric.candidate)
            & events["role"].eq(metric.role)
        ]
        if metric.policy == FIXED_POLICY:
            net_col = "fixed_net_bps"
            action_rows = 0
            difference = np.zeros(len(group))
        else:
            net_col = f"{metric.policy}__net_bps"
            action_rows = int(group[f"{metric.policy}__actionable"].sum())
            difference = group[f"{metric.policy}__paired_difference_bps"].to_numpy(float)
        aggregate_count_errors += int(
            len(group) != int(metric.rows) or action_rows != int(metric.action_rows)
        )
        max_aggregate_error = max(
            max_aggregate_error,
            abs(float(group[net_col].mean()) - float(metric.mean_net_bps)),
            abs(float(np.mean(difference)) - float(metric.paired_mean_difference_bps)),
        )

    bootstrap_metrics = pd.read_csv(args.artifact / "paired_bootstraps.csv")
    expected_bootstrap: dict[tuple[int, str, str], dict[str, float]] = {}
    seed_offset = 0
    primary = events.loc[events["role"].eq("primary_candidate")]
    for (_, _, role), cell in events.groupby(["period", "candidate", "role"], sort=True):
        if role != "primary_candidate":
            continue
        for policy in ALT_POLICIES:
            result = moving_block_bootstrap(cell, f"{policy}__paired_difference_bps", seed_offset)
            expected_bootstrap[
                (int(cell.iloc[0]["period"]), str(cell.iloc[0]["candidate"]), policy)
            ] = result
            seed_offset += 1
    max_bootstrap_error = 0.0
    for metric in bootstrap_metrics.itertuples(index=False):
        expected = expected_bootstrap[(int(metric.period), metric.candidate, metric.policy)]
        max_bootstrap_error = max(
            max_bootstrap_error,
            abs(expected["paired_mean_difference_bps"] - float(metric.paired_mean_difference_bps)),
            abs(expected["ci_lower"] - float(metric.ci_lower)),
            abs(expected["ci_upper"] - float(metric.ci_upper)),
            abs(expected["p_one_sided"] - float(metric.p_one_sided)),
        )
    max_holm_error = 0.0
    for _family, group in bootstrap_metrics.groupby("family", sort=False):
        expected_adjusted = holm_values(group["p_one_sided"].to_numpy(float))
        max_holm_error = max(
            max_holm_error,
            float(np.max(np.abs(expected_adjusted - group["holm_adjusted_p"].to_numpy(float)))),
        )

    contrast_metrics = pd.read_csv(args.artifact / "policy_contrast_metrics.csv")
    max_contrast_bootstrap_error = 0.0
    for (period, candidate), cell in primary.groupby(["period", "candidate"], sort=True):
        contrast = cell.copy()
        contrast["contrast"] = (
            contrast[f"{TERMINAL_POLICY}__net_bps"] - contrast[f"{TRANSITION_POLICY}__net_bps"]
        )
        expected = moving_block_bootstrap(contrast, "contrast", seed_offset)
        seed_offset += 1
        metric = contrast_metrics.loc[
            contrast_metrics["period"].eq(period) & contrast_metrics["candidate"].eq(candidate)
        ].iloc[0]
        max_contrast_bootstrap_error = max(
            max_contrast_bootstrap_error,
            abs(expected["paired_mean_difference_bps"] - float(metric.paired_mean_difference_bps)),
            abs(expected["ci_lower"] - float(metric.ci_lower)),
            abs(expected["ci_upper"] - float(metric.ci_upper)),
            abs(expected["p_one_sided"] - float(metric.p_one_sided)),
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

    leakage = audit_runner_event_logic()
    checks = {
        "research_only_safety_boundary": safety_ok,
        "pre_score_and_artifact_source_hashes_match": hashes_ok,
        "primary_population_counts_match": population_errors == 0,
        "path_topology_rederived_exactly": topology_errors == 0,
        "pre_entry_clock_rederived_exactly": pre_entry_errors == 0,
        "state_event_provider_timestamps_match": timestamp_errors == 0,
        "event_actions_rederived_exactly": action_errors == 0,
        "every_action_exits_strictly_next_open": same_bar_errors == 0,
        "no_pre_entry_event_treated_as_post_entry_action": pre_entry_action_errors == 0,
        "fixed_and_event_payoffs_replay": max(
            max_fixed_error, max_policy_price_error, max_policy_return_error
        )
        <= 1e-8,
        "aggregate_policy_metrics_replay": aggregate_count_errors == 0
        and max_aggregate_error <= 1e-10,
        "bootstrap_and_holm_replay": max(
            max_bootstrap_error, max_holm_error, max_contrast_bootstrap_error
        )
        <= 1e-10,
        "event_logic_leakage_scan": leakage["passes"],
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
            "topology": topology_errors,
            "pre_entry": pre_entry_errors,
            "timestamps": timestamp_errors,
            "actions": action_errors,
            "same_bar": same_bar_errors,
            "pre_entry_action": pre_entry_action_errors,
            "aggregate_counts": aggregate_count_errors,
            "manifest": manifest_errors,
        },
        "maximum_errors": {
            "fixed_replay": max_fixed_error,
            "policy_price_replay": max_policy_price_error,
            "policy_return_replay": max_policy_return_error,
            "aggregate_metric_replay": max_aggregate_error,
            "bootstrap_replay": max_bootstrap_error,
            "holm_replay": max_holm_error,
            "contrast_bootstrap_replay": max_contrast_bootstrap_error,
        },
        "leakage_scan": leakage,
    }
    write_json(output_path, payload)
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
