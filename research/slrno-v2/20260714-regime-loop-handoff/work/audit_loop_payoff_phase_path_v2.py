#!/usr/bin/env python3
"""Independent audit of corrected 190-session loop phase/path V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260713-loop-payoff-phase-path-v2.json"
PRE_SCORE = HERE / "contracts/20260713-loop-payoff-phase-path-v2-pre-score.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def sources(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract_v2": CONTRACT,
        "runner_v2": HERE / "run_loop_payoff_phase_path_v2.py",
        "auditor_v2": Path(__file__).resolve(),
        "contract_v1": Path(contract["inputs"]["base_v1_contract"]),
        "runner_v1": Path(contract["inputs"]["base_v1_runner"]),
        "anchor_2023": Path(contract["inputs"]["anchor_panels"]["2023"]),
        "anchor_2025": Path(contract["inputs"]["anchor_panels"]["2025"]),
        "ledger": Path(contract["inputs"]["accepted_signal_ledger"]),
        "fixed_cycles": Path(contract["inputs"]["fixed_cycles"]),
        "runs_2023": Path(contract["inputs"]["runs"]["2023"]),
        "runs_2024": Path(contract["inputs"]["runs"]["2024_hazard_fit"]),
        "runs_2025": Path(contract["inputs"]["runs"]["2025"]),
        "parent_report": Path(contract["inputs"]["parent_report"]),
        "parent_handoff": Path(contract["inputs"]["parent_handoff"]),
    }
    for period in (2023, 2025):
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = provider_path(root, symbol)
    return paths


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    ranks = rankdata(np.asarray(score, dtype=float), method="average")
    p = int(y.sum())
    n = int((~y).sum())
    return float((ranks[y].sum() - p * (p + 1) / 2) / (p * n))


def read_runs(path: Path) -> pd.DataFrame:
    columns = ["symbol_norm", "session_date", "state", "duration", "start_pos", "end_pos", "has_next_state"]
    frame = pd.read_parquet(path, columns=columns) if path.suffix == ".parquet" else pd.read_csv(path, usecols=columns)
    frame["session_date"] = frame["session_date"].astype(str)
    return frame


def rebuild_manifest(root: Path) -> None:
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(p for p in root.iterdir() if p.is_file() and p.name != "artifact_manifest.json")
    ]
    write_json(root / "artifact_manifest.json", {
        "research_only": True, "live_ordering_enabled": False, "order_placement": "disabled", "files": files,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    contract = json.loads(CONTRACT.read_text())
    pre = json.loads(PRE_SCORE.read_text())
    scored = pd.read_parquet(root / "signal_level_path_diagnostics.parquet")
    cohorts = pd.read_csv(root / "cohort_metrics.csv")
    paths = pd.read_csv(root / "path_class_metrics.csv")
    boots = pd.read_csv(root / "hazard_auc_bootstraps.csv")
    deletions = pd.read_csv(root / "stock_deletion_metrics.csv")
    calendar = json.loads((root / "score_calendar.json").read_text())
    decision = json.loads((root / "decision.json").read_text())
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    actual_sources = {name: digest(path) for name, path in sources(contract).items()}
    checks["source_hashes"] = actual_sources == pre["sha256"]
    checks["safety"] = bool(
        contract["research_only"] is True and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled" and contract["strategy_promotion_allowed"] is False
        and decision["research_only"] is True and decision["sealed_validation_performed"] is False
        and decision["economic_edge_claim"] is False and decision["strategy_promotion"] is False
    )
    checks["calendar_190_sessions"] = all(len(calendar[str(period)]) == 190 for period in (2023, 2025))
    checks["warmup_excluded"] = all(
        set(scored.loc[scored["period"].eq(period), "session_date"].astype(str)).issubset(set(calendar[str(period)]))
        for period in (2023, 2025)
    )
    expected_counts = contract["population"]["parent_expected_candidate_counts"]
    population_errors = 0
    for key, expected in expected_counts.items():
        period, loop, state = key.split("|")
        actual = len(scored.loc[
            scored["period"].eq(int(period)) & scored["top_loop"].eq(loop)
            & scored["anchor_state"].eq(int(state.removeprefix("state")))
        ])
        population_errors += abs(actual - int(expected))
    metrics["candidate_population_count_error"] = population_errors
    checks["parent_candidate_population_replay"] = population_errors == 0
    net_error = float(np.max(np.abs(scored["net_return_bps"] - (scored["gross_return_bps"] - 10.0))))
    metrics["maximum_net_identity_error_bps"] = net_error
    checks["net_identity"] = net_error == 0.0

    train = read_runs(Path(contract["inputs"]["runs"]["2024_hazard_fit"]))
    durations = {
        int(state): np.sort(group.loc[group["has_next_state"].eq(True), "duration"].to_numpy(int))
        for state, group in train.groupby("state", sort=True)
    }
    hazard_error = percentile_error = 0.0
    state_errors = 0
    for period in (2023, 2025):
        runs = read_runs(Path(contract["inputs"]["runs"][str(period)]))
        groups = {(str(s), str(d)): g for (s, d), g in runs.groupby(["symbol_norm", "session_date"], sort=False)}
        for row in scored.loc[scored["period"].eq(period)].itertuples(index=False):
            group = groups[(str(row.symbol_norm), str(row.session_date))]
            hit = group.loc[group["start_pos"].le(row.entry_state_position) & group["end_pos"].ge(row.entry_state_position)]
            if len(hit) != 1:
                state_errors += 1
                continue
            state_run = hit.iloc[0]
            age = int(row.entry_state_position) - int(state_run.start_pos) + 1
            required = int(row.frozen_exit_state_position) - int(state_run.start_pos) + 1
            sample = durations[int(state_run.state)]
            at_risk = sample[sample >= age]
            hazard_error = max(hazard_error, abs(float(np.mean(at_risk < required)) - float(row.exit_before_frozen_close_hazard_2024)))
            percentile_error = max(percentile_error, abs(float(np.mean(sample <= age)) - float(row.duration_percentile_2024)))
            if int(state_run.state) != int(row.admission_state) or age != int(row.admission_regime_age_bars):
                state_errors += 1
    metrics.update({"maximum_hazard_error": hazard_error, "maximum_percentile_error": percentile_error, "admission_state_errors": state_errors})
    checks["hazard_and_state_replay"] = state_errors == 0 and hazard_error == 0.0 and percentile_error == 0.0

    metric_error = path_count_error = deletion_error = 0.0
    candidate_rows = []
    for period, loop, state in [(2023, "cycle_04", 4), (2023, "cycle_07", 5), (2025, "cycle_04", 4), (2025, "cycle_07", 5)]:
        group = scored.loc[scored["period"].eq(period) & scored["top_loop"].eq(loop) & scored["anchor_state"].eq(state)]
        candidate = f"{loop}|state{state}"
        cohort = cohorts.loc[cohorts["period"].eq(period) & cohorts["cohort"].eq(candidate + "|candidate")].iloc[0]
        observed_auc = auc(group["final_positive"], -group["exit_before_frozen_close_hazard_2024"])
        recorded_auc = float(boots.loc[boots["period"].eq(period) & boots["candidate"].eq(candidate), "auc_negative_hazard"].iloc[0])
        metric_error = max(metric_error, abs(float(group["net_return_bps"].mean()) - float(cohort.mean_net_bps)), abs(observed_auc - recorded_auc))
        actual_counts = group["path_class"].value_counts().to_dict()
        recorded_counts = paths.loc[paths["period"].eq(period) & paths["candidate"].eq(candidate)].set_index("path_class")["rows"].to_dict()
        path_count_error += sum(abs(int(actual_counts.get(k, 0)) - int(recorded_counts.get(k, 0))) for k in set(actual_counts) | set(recorded_counts))
        candidate_rows.append(cohort)
    for row in deletions.itertuples(index=False):
        loop, state = str(row.candidate).split("|state")
        group = scored.loc[
            scored["period"].eq(int(row.period)) & scored["top_loop"].eq(loop)
            & scored["anchor_state"].eq(int(state)) & ~scored["symbol_norm"].eq(str(row.deleted_symbol))
        ]
        deletion_error += int(not np.isclose(auc(group["final_positive"], -group["exit_before_frozen_close_hazard_2024"]), row.negative_hazard_auc, atol=1e-12, rtol=0))
    metrics.update({"maximum_metric_error": metric_error, "path_count_error": path_count_error, "deletion_replay_errors": deletion_error})
    checks["candidate_and_path_metric_replay"] = metric_error <= 1e-12 and path_count_error == 0
    checks["stock_deletion_replay"] = deletion_error == 0

    candidates = pd.DataFrame(candidate_rows)
    deletion_counts = deletions.assign(ok=deletions["negative_hazard_auc"].gt(0.5)).groupby(["period", "candidate"])["ok"].sum()
    replay_checks = {
        "support_at_least_50_all_four_cells": bool(candidates["rows"].ge(50).all() and len(candidates) == 4),
        "negative_hazard_auc_above_half_all_four_cells": bool(boots["auc_negative_hazard"].gt(0.5).all()),
        "all_four_hazard_endpoints_pass_holm": bool(boots["passes_holm_0_05"].all()),
        "orientation_survival_difference_positive_all_four_cells": bool(candidates["survival_mean_net_difference_bps"].gt(0).all()),
        "hazard_leave_one_stock_out_at_least_16_all_four_cells": bool(deletion_counts.ge(16).all()),
    }
    checks["decision_replay"] = replay_checks == decision["checks"]
    audit = {
        "audit": "independent_loop_payoff_phase_path_v2", "research_only": True,
        "live_ordering_enabled": False, "order_placement": "disabled",
        "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "pass": all(checks.values()), "metrics": metrics,
    }
    write_json(root / "independent_audit.json", audit)
    rebuild_manifest(root)
    if not audit["pass"]:
        raise AssertionError(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
