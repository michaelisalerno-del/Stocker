"""Research-only causal parent-loop state-path investigation.

This script never places orders or changes application code. It replays four
predeclared, completed-bar state events as hypothetical next-open exits over an
already-open retrospective surface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-causal-loop-state-path-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260714-causal-loop-state-path-v1-pre-score.json"
BASE_PATH = HERE / "run_loop_payoff_phase_path_v1.py"
SPEC = importlib.util.spec_from_file_location("loop_phase_path_v1_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

PERIODS = (2023, 2025)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
SEED = 20260714
PRIMARY_COST_PER_SIDE = 5.0
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "auditor": HERE / "audit_causal_loop_state_path_v1.py",
        "tests": HERE / "tests/test_causal_loop_state_path_v1.py",
        "base_loader_runner": Path(inputs["base_loader_runner"]),
        "v2_signal_level_artifact": Path(inputs["v2_signal_level_artifact"]),
        "v2_report": Path(inputs["v2_report"]),
        "v2_handoff": Path(inputs["v2_handoff"]),
        "v2_prospective_log_contract": Path(inputs["v2_prospective_log_contract"]),
        "fixed_cycles": Path(inputs["fixed_cycles"]),
        "runs_2023": Path(inputs["runs"]["2023"]),
        "runs_2025": Path(inputs["runs"]["2025"]),
    }
    for period in PERIODS:
        root = Path(inputs["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = provider_path(root, symbol)
    return paths


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["broker_connection_enabled"] is False
        and safety["paper_or_demo_execution_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["position_or_order_functionality_allowed"] is False
        and safety["application_code_modification_allowed"] is False
        and safety["repository_write_allowed"] is False
        and contract["sealed_data_status"]["genuinely_unseen_sessions_available"] is False
        and contract["sealed_data_status"]["2023_or_2025_validation_claim_allowed"] is False
    ):
        raise AssertionError("research-only safety boundary drift")
    if contract["population"]["primary_cost_bps_per_side"] != PRIMARY_COST_PER_SIDE:
        raise AssertionError("primary cost drift")
    return contract


def freeze_manifest(contract: dict[str, Any]) -> None:
    if PRE_SCORE_PATH.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest {PRE_SCORE_PATH}")
    paths = source_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze; missing inputs: {missing}")
    write_json(
        PRE_SCORE_PATH,
        {
            "contract_id": contract["contract_id"],
            "frozen_before_path_conditioned_payoff_scoring": True,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sha256": {name: sha256(path) for name, path in sorted(paths.items())},
        },
    )


def verify_frozen_sources(contract: dict[str, Any]) -> dict[str, str]:
    expected = json.loads(PRE_SCORE_PATH.read_text())
    actual = {name: sha256(path) for name, path in sorted(source_paths(contract).items())}
    if actual != expected["sha256"]:
        changed = sorted(
            name
            for name in set(actual) | set(expected["sha256"])
            if actual.get(name) != expected["sha256"].get(name)
        )
        raise AssertionError(f"pre-score source hash mismatch: {changed}")
    return actual


def topology_from_transitions(
    transitions: list[tuple[int, int]],
    anchor_state: int,
    alternate_state: int,
) -> tuple[str, int | None, int | None, str | None, int | None]:
    """Return topology, completion pos, invalidation pos, terminal type/pos."""
    if not transitions:
        return "no_transition", None, None, None, None
    first_state, first_pos = transitions[0]
    if first_state != alternate_state:
        return "incompatible_first_transition", None, first_pos, "invalidation", first_pos
    if len(transitions) == 1:
        return "expected_leg_partial", None, None, None, None
    second_state, second_pos = transitions[1]
    if second_state == anchor_state:
        return "exact_parent_completion", second_pos, None, "completion", second_pos
    return "expected_leg_then_diversion", None, second_pos, "invalidation", second_pos


def pre_entry_status(
    transitions: list[tuple[int, int]],
    entry_state_position: int,
    anchor_state: int,
    alternate_state: int,
) -> str:
    known = [(state, pos) for state, pos in transitions if pos < entry_state_position]
    topology, _, _, _, _ = topology_from_transitions(known, anchor_state, alternate_state)
    return {
        "no_transition": "orientation_intact",
        "expected_leg_partial": "expected_leg_active",
        "exact_parent_completion": "completed_before_entry",
        "incompatible_first_transition": "invalidated_before_entry",
        "expected_leg_then_diversion": "invalidated_before_entry",
    }[topology]


def gross_return_bps(direction: int, exit_price: float, entry_price: float) -> float:
    if direction not in (-1, 1):
        raise AssertionError("invalid signal direction")
    return 10000.0 * direction * (exit_price / entry_price - 1.0)


def event_policy(
    event_position: int | None,
    entry_state_position: int,
    anchor_state_position: int,
    tape_anchor_ordinal: int,
    entry_ordinal: int,
    frozen_exit_ordinal: int,
    bars: pd.DataFrame,
    direction: int,
    entry_price: float,
    fixed_exit_price: float,
) -> dict[str, Any]:
    detection_ordinal: int | None = None
    next_open_ordinal: int | None = None
    actionable = False
    if event_position is not None and event_position >= entry_state_position:
        detection_ordinal = tape_anchor_ordinal + event_position - anchor_state_position
        next_open_ordinal = detection_ordinal + 1
        actionable = (
            detection_ordinal >= entry_ordinal
            and next_open_ordinal <= frozen_exit_ordinal
            and next_open_ordinal < len(bars)
        )
    if actionable:
        exit_ordinal = int(next_open_ordinal)
        exit_price = float(bars.iloc[exit_ordinal]["open"])
        exit_timestamp = pd.Timestamp(bars.iloc[exit_ordinal]["timestamp"])
    else:
        exit_ordinal = frozen_exit_ordinal
        exit_price = fixed_exit_price
        exit_timestamp = pd.Timestamp(bars.iloc[exit_ordinal]["timestamp"])
    gross = gross_return_bps(direction, exit_price, entry_price)
    return {
        "event_position": event_position,
        "detection_ordinal": detection_ordinal,
        "next_open_ordinal": next_open_ordinal,
        "actionable": actionable,
        "exit_ordinal": exit_ordinal,
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_COST,
        "detection_bars_after_entry": (
            detection_ordinal - entry_ordinal if detection_ordinal is not None else math.nan
        ),
    }


def event_timestamp_check(
    state_runs: pd.DataFrame,
    event_position: int | None,
    bars: pd.DataFrame,
    tape_anchor_ordinal: int,
    anchor_state_position: int,
) -> None:
    if event_position is None:
        return
    match = state_runs.loc[state_runs["start_pos"].eq(event_position)]
    if len(match) != 1:
        raise AssertionError("state event position does not identify one run")
    detection_ordinal = tape_anchor_ordinal + event_position - anchor_state_position
    state_timestamp = pd.Timestamp(match.iloc[0]["start_timestamp"])
    bar_timestamp = pd.Timestamp(bars.iloc[detection_ordinal]["timestamp"])
    if state_timestamp.tzinfo is None:
        state_timestamp = state_timestamp.tz_localize("UTC")
    else:
        state_timestamp = state_timestamp.tz_convert("UTC")
    if bar_timestamp.tzinfo is None:
        bar_timestamp = bar_timestamp.tz_localize("UTC")
    else:
        bar_timestamp = bar_timestamp.tz_convert("UTC")
    if state_timestamp != bar_timestamp:
        raise AssertionError("state-event/provider timestamp mismatch")


def enrich_signal(
    row: Any,
    spec: dict[str, Any],
    bars: pd.DataFrame,
    state_runs: pd.DataFrame,
) -> dict[str, Any]:
    anchor_state = int(spec["anchor_state"])
    alternate_state = int(spec["expected_alternate_state"])
    anchor_position = int(row.anchor_start_pos)
    entry_position = int(row.entry_state_position)
    frozen_position = int(row.frozen_exit_state_position)
    anchor_run = BASE.covering_run(state_runs, anchor_position)
    if int(anchor_run.state) != anchor_state or int(anchor_run.start_pos) != anchor_position:
        raise AssertionError("anchor does not match its frozen state-run start")
    within = state_runs.loc[
        state_runs["start_pos"].gt(anchor_position) & state_runs["start_pos"].le(frozen_position)
    ].sort_values("start_pos", kind="stable")
    transitions = [
        (int(item.state), int(item.start_pos)) for item in within.itertuples(index=False)
    ]
    topology, completion_pos, invalidation_pos, terminal_type, terminal_pos = (
        topology_from_transitions(transitions, anchor_state, alternate_state)
    )
    first_transition_pos = next((pos for _, pos in transitions if pos >= entry_position), None)

    tape_anchor_ordinal = int(row.tape_anchor_ordinal)
    entry_ordinal = int(row.entry_ordinal)
    frozen_exit_ordinal = tape_anchor_ordinal + 24
    if frozen_exit_ordinal >= len(bars):
        raise AssertionError("frozen exit outside provider tape")
    if pd.Timestamp(bars.iloc[tape_anchor_ordinal]["timestamp"]) != pd.Timestamp(
        row.start_timestamp
    ):
        raise AssertionError("provider anchor timestamp mismatch")
    if int(row.frozen_exit_state_position) != anchor_position + 24:
        raise AssertionError("frozen state-position horizon drift")
    fixed_exit_price = float(bars.iloc[frozen_exit_ordinal]["close"])
    fixed_gross = gross_return_bps(int(row.direction), fixed_exit_price, float(row.entry_price))
    if not np.isclose(fixed_gross, float(row.gross_return_bps), atol=1e-8, rtol=1e-8):
        raise AssertionError("fixed close replay mismatch")

    event_positions = {
        TERMINAL_POLICY: terminal_pos,
        COMPLETION_POLICY: completion_pos,
        INVALIDATION_POLICY: invalidation_pos,
        TRANSITION_POLICY: first_transition_pos,
    }
    for position in set(value for value in event_positions.values() if value is not None):
        event_timestamp_check(state_runs, position, bars, tape_anchor_ordinal, anchor_position)

    policies = {
        name: event_policy(
            position,
            entry_position,
            anchor_position,
            tape_anchor_ordinal,
            entry_ordinal,
            frozen_exit_ordinal,
            bars,
            int(row.direction),
            float(row.entry_price),
            fixed_exit_price,
        )
        for name, position in event_positions.items()
    }

    payload = {
        "period": int(row.period),
        "candidate": spec["candidate"],
        "role": spec["role"],
        "symbol_norm": str(row.symbol_norm),
        "session_date": str(row.session_date),
        "quarter": str(row.quarter),
        "anchor_id": row.anchor_id,
        "start_timestamp": pd.Timestamp(row.start_timestamp),
        "anchor_state": anchor_state,
        "expected_alternate_state": alternate_state,
        "direction": int(row.direction),
        "entry_price": float(row.entry_price),
        "entry_ordinal": entry_ordinal,
        "tape_anchor_ordinal": tape_anchor_ordinal,
        "frozen_exit_ordinal": frozen_exit_ordinal,
        "anchor_state_position": anchor_position,
        "entry_state_position": entry_position,
        "frozen_exit_state_position": frozen_position,
        "path_topology": topology,
        "pre_entry_path_status": pre_entry_status(
            transitions, entry_position, anchor_state, alternate_state
        ),
        "transition_count_through_frozen_close": len(transitions),
        "first_transition_state": transitions[0][0] if transitions else math.nan,
        "first_transition_position": transitions[0][1] if transitions else math.nan,
        "second_transition_state": transitions[1][0] if len(transitions) > 1 else math.nan,
        "second_transition_position": transitions[1][1] if len(transitions) > 1 else math.nan,
        "completion_event_position": completion_pos,
        "invalidation_event_position": invalidation_pos,
        "terminal_event_type": terminal_type,
        "terminal_event_position": terminal_pos,
        "first_post_entry_transition_position": first_transition_pos,
        "fixed_exit_price": fixed_exit_price,
        "fixed_gross_bps": fixed_gross,
        "fixed_net_bps": fixed_gross - ROUND_TRIP_COST,
        "final_positive": bool(row.final_positive),
        "v2_path_class": str(row.path_class),
        "mfe_bps": float(row.mfe_bps),
        "mae_bps": float(row.mae_bps),
        "mfe_atr": float(row.mfe_atr),
        "mae_atr": float(row.mae_atr),
        "time_to_mfe_bars": int(row.time_to_mfe_bars),
        "time_to_mae_bars": int(row.time_to_mae_bars),
    }
    for policy, result in policies.items():
        prefix = policy
        payload[f"{prefix}__event_position"] = result["event_position"]
        payload[f"{prefix}__detection_ordinal"] = result["detection_ordinal"]
        payload[f"{prefix}__next_open_ordinal"] = result["next_open_ordinal"]
        payload[f"{prefix}__actionable"] = result["actionable"]
        payload[f"{prefix}__exit_ordinal"] = result["exit_ordinal"]
        payload[f"{prefix}__exit_timestamp"] = result["exit_timestamp"]
        payload[f"{prefix}__exit_price"] = result["exit_price"]
        payload[f"{prefix}__gross_bps"] = result["gross_bps"]
        payload[f"{prefix}__net_bps"] = result["net_bps"]
        payload[f"{prefix}__paired_difference_bps"] = result["net_bps"] - payload["fixed_net_bps"]
        payload[f"{prefix}__detection_bars_after_entry"] = result["detection_bars_after_entry"]
    return payload


def moving_block_sample(sessions: list[str], rng: np.random.Generator) -> list[str]:
    n = len(sessions)
    blocks = int(math.ceil(n / BOOTSTRAP_BLOCK))
    starts = rng.integers(0, n, size=blocks)
    return [
        sessions[(int(start) + offset) % n] for start in starts for offset in range(BOOTSTRAP_BLOCK)
    ][:n]


def paired_bootstrap(
    group: pd.DataFrame, difference_column: str, seed_offset: int
) -> dict[str, Any]:
    sessions = sorted(group["session_date"].astype(str).unique())
    by_session = {
        date: block[difference_column].to_numpy(float)
        for date, block in group.groupby("session_date", sort=False)
    }
    observed = float(group[difference_column].mean())
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = moving_block_sample(sessions, rng)
        replay = np.concatenate([by_session[date] for date in sampled])
        values[draw] = float(np.mean(replay))
    return {
        "paired_mean_difference_bps": observed,
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.0)) / (len(values) + 1)),
        "draws_valid": int(len(values)),
        "sessions": len(sessions),
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    p = output["p_one_sided"].to_numpy(float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    output["holm_adjusted_p"] = adjusted
    output["passes_holm_0_05"] = output["holm_adjusted_p"].lt(0.05) & output["ci_lower"].gt(0.0)
    return output


def policy_columns(policy: str) -> tuple[str, str, str]:
    if policy == FIXED_POLICY:
        return "fixed_gross_bps", "fixed_net_bps", ""
    return (
        f"{policy}__gross_bps",
        f"{policy}__net_bps",
        f"{policy}__paired_difference_bps",
    )


def artifact_manifest(out: Path) -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(
                item
                for item in out.iterdir()
                if item.is_file()
                and item.name not in {"artifact_manifest.json", "independent_audit.json"}
            )
        ],
    }


def build_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in contract["population"]["primary_candidates"]:
        specs.append({**item, "role": "primary_candidate"})
    for item in contract["population"]["descriptive_matched_controls"]:
        specs.append({**item, "role": "matched_control"})
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-manifest", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    if args.freeze_manifest:
        if args.out is not None:
            raise ValueError("--out is incompatible with --freeze-manifest")
        freeze_manifest(contract)
        print(json.dumps({"frozen": str(PRE_SCORE_PATH)}, indent=2))
        return
    if args.out is None:
        raise ValueError("--out is required when scoring")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    source_hashes = verify_frozen_sources(contract)
    args.out.mkdir(parents=True)

    signals = pd.read_parquet(Path(contract["inputs"]["v2_signal_level_artifact"]))
    if not (
        signals["period"].isin(PERIODS).all()
        and signals["strategy"].eq(contract["population"]["source_strategy"]).all()
        and signals["horizon"].eq(24).all()
        and signals["status"].eq("filled").all()
    ):
        raise AssertionError("V2 population surface drift")
    signals["period"] = signals["period"].astype(int)
    signals["session_date"] = signals["session_date"].astype(str)

    specs = build_specs(contract)
    enriched_rows: list[dict[str, Any]] = []
    provider_coverage: list[dict[str, Any]] = []
    for period in PERIODS:
        tape, coverage = BASE.load_tape(
            Path(contract["inputs"]["provider_roots"][str(period)]),
            list(contract["population"]["symbols"]),
            period,
        )
        provider_coverage.extend(coverage)
        runs = BASE.load_runs(Path(contract["inputs"]["runs"][str(period)]), period)
        runs_lookup = BASE.run_lookup(runs)
        period_signals = signals.loc[signals["period"].eq(period)]
        for spec in specs:
            cell = period_signals.loc[
                period_signals["top_loop"].eq(spec["candidate"].split("|")[0])
                & period_signals["anchor_state"].eq(int(spec["anchor_state"]))
            ].sort_values(["session_date", "symbol_norm", "entry_ordinal"], kind="stable")
            if spec["role"] == "primary_candidate":
                expected = int(spec["expected_rows"][str(period)])
                if len(cell) != expected:
                    raise AssertionError(
                        f"primary population mismatch {period} {spec['candidate']}: {len(cell)}"
                    )
            for row in cell.itertuples(index=False):
                key = (str(row.symbol_norm), str(row.session_date))
                enriched_rows.append(enrich_signal(row, spec, tape[key], runs_lookup[key]))

    enriched = (
        pd.DataFrame(enriched_rows)
        .sort_values(
            ["period", "candidate", "session_date", "symbol_norm", "entry_ordinal"], kind="stable"
        )
        .reset_index(drop=True)
    )
    enriched.to_parquet(args.out / "signal_path_events.parquet", index=False)
    pd.DataFrame(provider_coverage).to_csv(args.out / "data_coverage.csv", index=False)

    topology_rows: list[dict[str, Any]] = []
    topology_class_rows: list[dict[str, Any]] = []
    event_coverage_rows: list[dict[str, Any]] = []
    event_timing_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    grouped_cells = enriched.groupby(["period", "candidate", "role"], sort=True)
    seed_offset = 0
    for (period, candidate, role), cell in grouped_cells:
        for topology, group in cell.groupby("path_topology", sort=True):
            topology_rows.append(
                {
                    "period": period,
                    "candidate": candidate,
                    "role": role,
                    "path_topology": topology,
                    "rows": int(len(group)),
                    "share": float(len(group) / len(cell)),
                    "fixed_mean_gross_bps": float(group["fixed_gross_bps"].mean()),
                    "fixed_mean_net_bps": float(group["fixed_net_bps"].mean()),
                    "fixed_positive_rate": float(group["fixed_net_bps"].gt(0).mean()),
                    "timing_failure_rate": float(
                        group["v2_path_class"].eq("timing_failure").mean()
                    ),
                    "no_usable_move_rate": float(
                        group["v2_path_class"].eq("no_usable_move").mean()
                    ),
                    "mean_mfe_bps": float(group["mfe_bps"].mean()),
                    "mean_mae_bps": float(group["mae_bps"].mean()),
                    "median_time_to_mfe_bars": float(group["time_to_mfe_bars"].median()),
                    "median_time_to_mae_bars": float(group["time_to_mae_bars"].median()),
                    "pre_entry_terminal_rate": float(
                        group["pre_entry_path_status"]
                        .isin(["completed_before_entry", "invalidated_before_entry"])
                        .mean()
                    ),
                }
            )
            for payoff_class, subgroup in group.groupby("v2_path_class", sort=True):
                topology_class_rows.append(
                    {
                        "period": period,
                        "candidate": candidate,
                        "role": role,
                        "path_topology": topology,
                        "payoff_class": payoff_class,
                        "rows": int(len(subgroup)),
                        "topology_share": float(len(subgroup) / len(group)),
                        "cell_share": float(len(subgroup) / len(cell)),
                        "fixed_mean_net_bps": float(subgroup["fixed_net_bps"].mean()),
                    }
                )

        for policy in (FIXED_POLICY, *ALT_POLICIES):
            gross_col, net_col, diff_col = policy_columns(policy)
            if policy == FIXED_POLICY:
                action = pd.Series(False, index=cell.index)
                difference = pd.Series(0.0, index=cell.index)
            else:
                action = cell[f"{policy}__actionable"].astype(bool)
                difference = cell[diff_col]
                event_coverage_rows.append(
                    {
                        "period": period,
                        "candidate": candidate,
                        "role": role,
                        "policy": policy,
                        "rows": int(len(cell)),
                        "actionable_event_rows": int(action.sum()),
                        "actionable_event_share": float(action.mean()),
                        "fallback_rows": int((~action).sum()),
                        "fallback_share": float((~action).mean()),
                        "median_detection_bars_after_entry": float(
                            cell.loc[action, f"{policy}__detection_bars_after_entry"].median()
                        )
                        if action.any()
                        else math.nan,
                    }
                )
                if action.any():
                    acted = cell.loc[action]
                    event_timing_rows.append(
                        {
                            "period": period,
                            "candidate": candidate,
                            "role": role,
                            "policy": policy,
                            "rows": int(len(acted)),
                            "mean_fixed_net_bps": float(acted["fixed_net_bps"].mean()),
                            "mean_event_exit_net_bps": float(acted[net_col].mean()),
                            "mean_paired_difference_bps": float(acted[diff_col].mean()),
                            "median_detection_bars_after_entry": float(
                                acted[f"{policy}__detection_bars_after_entry"].median()
                            ),
                            "median_exit_bars_after_entry": float(
                                (acted[f"{policy}__exit_ordinal"] - acted["entry_ordinal"]).median()
                            ),
                        }
                    )
            policy_rows.append(
                {
                    "period": period,
                    "candidate": candidate,
                    "role": role,
                    "policy": policy,
                    "rows": int(len(cell)),
                    "action_rows": int(action.sum()),
                    "action_share": float(action.mean()),
                    "mean_gross_bps": float(cell[gross_col].mean()),
                    "mean_net_bps": float(cell[net_col].mean()),
                    "positive_rate": float(cell[net_col].gt(0).mean()),
                    "fixed_mean_net_bps": float(cell["fixed_net_bps"].mean()),
                    "paired_mean_difference_bps": float(difference.mean()),
                    "paired_median_difference_bps": float(difference.median()),
                }
            )
            for cost in contract["evaluation"]["cost_sensitivity_bps_per_side"]:
                cost_rows.append(
                    {
                        "period": period,
                        "candidate": candidate,
                        "role": role,
                        "policy": policy,
                        "cost_bps_per_side": float(cost),
                        "mean_gross_bps": float(cell[gross_col].mean()),
                        "mean_net_bps": float(cell[gross_col].mean() - 2.0 * float(cost)),
                        "positive_mean_net": bool(cell[gross_col].mean() - 2.0 * float(cost) > 0),
                    }
                )
            if role == "primary_candidate" and policy != FIXED_POLICY:
                for quarter, group in cell.groupby("quarter", sort=True):
                    quarter_rows.append(
                        {
                            "period": period,
                            "candidate": candidate,
                            "policy": policy,
                            "quarter": quarter,
                            "rows": int(len(group)),
                            "action_rows": int(group[f"{policy}__actionable"].sum()),
                            "paired_mean_difference_bps": float(group[diff_col].mean()),
                            "policy_mean_net_bps": float(group[net_col].mean()),
                        }
                    )
                for deleted_symbol in contract["population"]["symbols"]:
                    group = cell.loc[~cell["symbol_norm"].eq(deleted_symbol)]
                    deletion_rows.append(
                        {
                            "period": period,
                            "candidate": candidate,
                            "policy": policy,
                            "deleted_symbol": deleted_symbol,
                            "rows": int(len(group)),
                            "action_rows": int(group[f"{policy}__actionable"].sum()),
                            "paired_mean_difference_bps": float(group[diff_col].mean()),
                            "policy_mean_net_bps": float(group[net_col].mean()),
                        }
                    )
                family = (
                    "primary_terminal" if policy == TERMINAL_POLICY else "secondary_exit_family"
                )
                bootstrap_rows.append(
                    {
                        "period": period,
                        "candidate": candidate,
                        "policy": policy,
                        "family": family,
                        "rows": int(len(cell)),
                        "action_rows": int(action.sum()),
                        **paired_bootstrap(cell, diff_col, seed_offset),
                    }
                )
                seed_offset += 1

    topology_metrics = pd.DataFrame(topology_rows).sort_values(
        ["role", "candidate", "period", "path_topology"], kind="stable"
    )
    topology_classes = pd.DataFrame(topology_class_rows).sort_values(
        ["role", "candidate", "period", "path_topology", "payoff_class"], kind="stable"
    )
    event_coverage = pd.DataFrame(event_coverage_rows).sort_values(
        ["role", "candidate", "period", "policy"], kind="stable"
    )
    event_timing = pd.DataFrame(event_timing_rows).sort_values(
        ["role", "candidate", "period", "policy"], kind="stable"
    )
    policy_metrics = pd.DataFrame(policy_rows).sort_values(
        ["role", "candidate", "period", "policy"], kind="stable"
    )
    quarters = pd.DataFrame(quarter_rows).sort_values(
        ["candidate", "period", "policy", "quarter"], kind="stable"
    )
    deletions = pd.DataFrame(deletion_rows).sort_values(
        ["candidate", "period", "policy", "deleted_symbol"], kind="stable"
    )
    costs = pd.DataFrame(cost_rows).sort_values(
        ["role", "candidate", "period", "policy", "cost_bps_per_side"], kind="stable"
    )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstraps = pd.concat(
        [
            holm_adjust(bootstrap_frame.loc[bootstrap_frame["family"].eq(family)])
            for family in ("primary_terminal", "secondary_exit_family")
        ],
        ignore_index=True,
    ).sort_values(["family", "candidate", "period", "policy"], kind="stable")

    contrast_rows: list[dict[str, Any]] = []
    for (period, candidate), cell in enriched.loc[enriched["role"].eq("primary_candidate")].groupby(
        ["period", "candidate"], sort=True
    ):
        contrast_col = "terminal_minus_first_transition_bps"
        contrast = cell.copy()
        contrast[contrast_col] = (
            contrast[f"{TERMINAL_POLICY}__net_bps"] - contrast[f"{TRANSITION_POLICY}__net_bps"]
        )
        result = paired_bootstrap(contrast, contrast_col, seed_offset)
        seed_offset += 1
        contrast_rows.append(
            {
                "period": period,
                "candidate": candidate,
                "rows": int(len(contrast)),
                "terminal_action_rows": int(contrast[f"{TERMINAL_POLICY}__actionable"].sum()),
                "first_transition_action_rows": int(
                    contrast[f"{TRANSITION_POLICY}__actionable"].sum()
                ),
                **result,
                "positive_quarters": int(
                    sum(
                        group[contrast_col].mean() > 0
                        for _, group in contrast.groupby("quarter", sort=True)
                    )
                ),
                "positive_stock_deletions": int(
                    sum(
                        contrast.loc[~contrast["symbol_norm"].eq(symbol), contrast_col].mean() > 0
                        for symbol in contract["population"]["symbols"]
                    )
                ),
            }
        )
    contrasts = pd.DataFrame(contrast_rows).sort_values(["candidate", "period"], kind="stable")

    topology_metrics.to_csv(args.out / "path_topology_metrics.csv", index=False)
    topology_classes.to_csv(args.out / "path_topology_payoff_class.csv", index=False)
    event_coverage.to_csv(args.out / "event_coverage.csv", index=False)
    event_timing.to_csv(args.out / "event_timing_metrics.csv", index=False)
    policy_metrics.to_csv(args.out / "exit_policy_metrics.csv", index=False)
    quarters.to_csv(args.out / "quarter_metrics.csv", index=False)
    deletions.to_csv(args.out / "stock_deletion_metrics.csv", index=False)
    costs.to_csv(args.out / "cost_sensitivity.csv", index=False)
    bootstraps.to_csv(args.out / "paired_bootstraps.csv", index=False)
    contrasts.to_csv(args.out / "policy_contrast_metrics.csv", index=False)

    primary_boot = bootstraps.loc[bootstraps["family"].eq("primary_terminal")]
    primary_metrics = policy_metrics.loc[
        policy_metrics["role"].eq("primary_candidate")
        & policy_metrics["policy"].eq(TERMINAL_POLICY)
    ]
    primary_quarters = (
        quarters.loc[quarters["policy"].eq(TERMINAL_POLICY)]
        .groupby(["period", "candidate"], as_index=False)
        .agg(
            positive_quarters=("paired_mean_difference_bps", lambda values: int((values > 0).sum()))
        )
    )
    primary_deletions = (
        deletions.loc[deletions["policy"].eq(TERMINAL_POLICY)]
        .groupby(["period", "candidate"], as_index=False)
        .agg(
            positive_stock_deletions=(
                "paired_mean_difference_bps",
                lambda values: int((values > 0).sum()),
            )
        )
    )
    primary_cost = costs.loc[
        costs["role"].eq("primary_candidate")
        & costs["policy"].eq(TERMINAL_POLICY)
        & costs["cost_bps_per_side"].eq(PRIMARY_COST_PER_SIDE)
    ]
    checks = {
        "four_primary_cells_present": len(primary_metrics) == 4,
        "candidate_support_at_least_50_all_four": bool(
            len(primary_metrics) == 4 and primary_metrics["rows"].ge(50).all()
        ),
        "actionable_terminal_events_at_least_30_all_four": bool(
            len(primary_metrics) == 4 and primary_metrics["action_rows"].ge(30).all()
        ),
        "paired_mean_positive_all_four": bool(
            len(primary_metrics) == 4 and primary_metrics["paired_mean_difference_bps"].gt(0).all()
        ),
        "block_interval_lower_positive_all_four": bool(
            len(primary_boot) == 4 and primary_boot["ci_lower"].gt(0).all()
        ),
        "all_four_primary_endpoints_pass_holm": bool(
            len(primary_boot) == 4 and primary_boot["passes_holm_0_05"].all()
        ),
        "at_least_three_positive_quarters_all_four": bool(
            len(primary_quarters) == 4 and primary_quarters["positive_quarters"].ge(3).all()
        ),
        "at_least_16_positive_stock_deletions_all_four": bool(
            len(primary_deletions) == 4
            and primary_deletions["positive_stock_deletions"].ge(16).all()
        ),
        "absolute_mean_net_positive_at_5bps_per_side_all_four": bool(
            len(primary_cost) == 4 and primary_cost["mean_net_bps"].gt(0).all()
        ),
    }
    primary_gate = all(checks.values())
    route_specific_observed = bool(
        len(contrasts) == 4 and contrasts["paired_mean_difference_bps"].gt(0).all()
    )
    route_specific_secure = bool(route_specific_observed and contrasts["ci_lower"].gt(0).all())
    if primary_gate and route_specific_observed:
        decision_name = "route_event_family_merits_prospective_logging_only"
    elif primary_gate:
        decision_name = "generic_transition_timing_only"
    else:
        decision_name = "path_event_family_not_supported"
    decision = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "application_modified": False,
        "sealed_validation_performed": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "primary_cost_bps_per_side": PRIMARY_COST_PER_SIDE,
        "checks": checks,
        "primary_gate_passed_on_opened_retrospective_data": primary_gate,
        "terminal_route_beats_first_transition_observed_all_four": route_specific_observed,
        "terminal_route_beats_first_transition_ci_lower_positive_all_four": route_specific_secure,
        "decision": decision_name,
        "maximum_allowed_action": "prospective immutable research logging only",
    }
    write_json(args.out / "decision.json", decision)
    write_json(
        args.out / "summary.json",
        {
            "contract_id": contract["contract_id"],
            "scientific_status": contract["scientific_status"],
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sealed_data_status": contract["sealed_data_status"],
            "provider_volume_label": contract["provenance"]["volume"],
            "quotes_or_ticks_used": False,
            "primary_candidate_topologies": topology_metrics.loc[
                topology_metrics["role"].eq("primary_candidate")
            ].to_dict("records"),
            "primary_policy_metrics": primary_metrics.to_dict("records"),
            "primary_bootstraps": primary_boot.to_dict("records"),
            "primary_quarter_checks": primary_quarters.to_dict("records"),
            "primary_stock_deletion_checks": primary_deletions.to_dict("records"),
            "route_vs_generic_transition": contrasts.to_dict("records"),
            "decision": decision,
        },
    )
    write_json(
        args.out / "source_hashes.json",
        {
            "contract_id": contract["contract_id"],
            "frozen_before_path_conditioned_payoff_scoring": True,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "sha256": source_hashes,
        },
    )
    write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(enriched),
                "primary_rows": int(enriched["role"].eq("primary_candidate").sum()),
                "decision": decision_name,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
