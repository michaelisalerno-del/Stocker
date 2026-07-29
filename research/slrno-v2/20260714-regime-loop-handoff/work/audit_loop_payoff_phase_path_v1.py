#!/usr/bin/env python3
"""Independent audit for the research-only loop phase/path diagnostic."""

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
CONTRACT_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v1-pre-score.json"
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
FEATURE_COLUMNS = {
    "admission_state",
    "admission_regime_age_bars",
    "orientation_survived_to_admission",
    "duration_percentile_2024",
    "exit_before_frozen_close_hazard_2024",
    "hazard_support_2024",
    "atr14_prior",
}


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
    if isinstance(value, list):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": HERE / "run_loop_payoff_phase_path_v1.py",
        "auditor": Path(__file__).resolve(),
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


def read_runs(path: Path) -> pd.DataFrame:
    columns = ["symbol_norm", "session_date", "state", "duration", "start_pos", "end_pos", "has_next_state"]
    frame = pd.read_parquet(path, columns=columns) if path.suffix == ".parquet" else pd.read_csv(path, usecols=columns)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    return frame.sort_values(["symbol_norm", "session_date", "start_pos"], kind="stable").reset_index(drop=True)


def rebuild_manifest(out: Path) -> None:
    files = []
    for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "artifact_manifest.json"):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "artifact_manifest.json", {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": files,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.artifact_root
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    scored = pd.read_parquet(out / "signal_level_path_diagnostics.parquet")
    cohorts = pd.read_csv(out / "cohort_metrics.csv")
    paths = pd.read_csv(out / "path_class_metrics.csv")
    bootstraps = pd.read_csv(out / "hazard_auc_bootstraps.csv")
    deletions = pd.read_csv(out / "stock_deletion_metrics.csv")
    decision = json.loads((out / "decision.json").read_text())
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    actual_sources = {name: sha256(path) for name, path in source_paths(contract).items()}
    checks["source_hashes"] = actual_sources == pre_score["sha256"]
    checks["safety_contract"] = bool(
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["strategy_promotion_allowed"] is False
        and contract["application_code_modification_allowed"] is False
        and contract["sealed_data_status"]["genuinely_unseen_sessions_available"] is False
    )
    checks["safety_decision"] = bool(
        decision["research_only"] is True
        and decision["live_ordering_enabled"] is False
        and decision["order_placement"] == "disabled"
        and decision["sealed_validation_performed"] is False
        and decision["strategy_promotion"] is False
        and decision["economic_edge_claim"] is False
    )
    forbidden_feature_terms = ("future", "next_state", "realized", "gross_return", "net_return", "mfe", "mae", "path_class")
    checks["feature_leakage_scan"] = not any(any(term in column for term in forbidden_feature_terms) for column in FEATURE_COLUMNS)

    ledger = pd.read_parquet(Path(contract["inputs"]["accepted_signal_ledger"]))
    ledger["period"] = pd.to_numeric(ledger["period"], errors="raise").astype(int)
    expected = ledger.loc[
        ledger["period"].isin([2023, 2025])
        & ledger["strategy"].eq(contract["population"]["source_strategy"])
        & ledger["horizon"].eq(24)
        & ledger["status"].eq("filled")
    ]
    checks["frozen_population"] = len(expected) == len(scored)
    net_error = np.max(np.abs(scored["net_return_bps"].to_numpy(float) - (scored["gross_return_bps"].to_numpy(float) - 10.0)))
    metrics["maximum_net_identity_error_bps"] = float(net_error)
    checks["net_return_identity"] = bool(net_error <= 1e-12)

    cycles = pd.read_csv(Path(contract["inputs"]["fixed_cycles"]))
    maximum_loop_error = 0.0
    top_mismatch = 0
    for period in (2023, 2025):
        anchor = pd.read_parquet(Path(contract["inputs"]["anchor_panels"][str(period)]), columns=["anchor_id", *LOOP_COLUMNS])
        values = anchor.loc[:, LOOP_COLUMNS].to_numpy(float)
        index = np.argmax(values, axis=1)
        replay = pd.DataFrame({
            "anchor_id": anchor["anchor_id"].to_numpy(int),
            "replay_loop": cycles["cycle_id"].to_numpy(str)[index],
            "replay_probability": values[np.arange(len(anchor)), index],
        })
        compare = scored.loc[scored["period"].eq(period), ["anchor_id", "top_loop", "top_loop_probability"]].merge(replay, on="anchor_id", validate="many_to_one")
        top_mismatch += int((compare["top_loop"] != compare["replay_loop"]).sum())
        maximum_loop_error = max(maximum_loop_error, float(np.max(np.abs(compare["top_loop_probability"] - compare["replay_probability"]))))
    metrics["top_loop_mismatches"] = top_mismatch
    metrics["maximum_top_loop_probability_error"] = maximum_loop_error
    checks["top_loop_replay"] = top_mismatch == 0 and maximum_loop_error == 0.0

    train = read_runs(Path(contract["inputs"]["runs"]["2024_hazard_fit"]))
    duration_map = {
        int(state): np.sort(group.loc[group["has_next_state"].eq(True), "duration"].to_numpy(int))
        for state, group in train.groupby("state", sort=True)
    }
    hazard_error = 0.0
    percentile_error = 0.0
    support_error = 0
    state_errors = 0
    for period in (2023, 2025):
        runs = read_runs(Path(contract["inputs"]["runs"][str(period)]))
        run_groups = {
            (str(symbol), str(session)): group.reset_index(drop=True)
            for (symbol, session), group in runs.groupby(["symbol_norm", "session_date"], sort=False)
        }
        for row in scored.loc[scored["period"].eq(period)].itertuples(index=False):
            group = run_groups[(str(row.symbol_norm), str(row.session_date))]
            candidates = group.loc[group["start_pos"].le(int(row.entry_state_position)) & group["end_pos"].ge(int(row.entry_state_position))]
            if len(candidates) != 1:
                state_errors += 1
                continue
            admission = candidates.iloc[0]
            age = int(row.entry_state_position) - int(admission.start_pos) + 1
            required = int(row.frozen_exit_state_position) - int(admission.start_pos) + 1
            durations = duration_map[int(admission.state)]
            at_risk = durations[durations >= age]
            hazard = float(np.mean(at_risk < required))
            percentile = float(np.mean(durations <= age))
            hazard_error = max(hazard_error, abs(hazard - float(row.exit_before_frozen_close_hazard_2024)))
            percentile_error = max(percentile_error, abs(percentile - float(row.duration_percentile_2024)))
            support_error = max(support_error, abs(len(at_risk) - int(row.hazard_support_2024)))
            if int(admission.state) != int(row.admission_state) or age != int(row.admission_regime_age_bars):
                state_errors += 1
    metrics.update({
        "maximum_hazard_error": hazard_error,
        "maximum_duration_percentile_error": percentile_error,
        "maximum_hazard_support_error": support_error,
        "admission_state_errors": state_errors,
    })
    checks["admission_state_and_hazard_replay"] = state_errors == 0 and hazard_error == 0.0 and percentile_error == 0.0 and support_error == 0

    candidate_specs = [(2023, "cycle_04", 4), (2023, "cycle_07", 5), (2025, "cycle_04", 4), (2025, "cycle_07", 5)]
    cohort_error = 0.0
    auc_error = 0.0
    path_count_error = 0
    for period, loop, state in candidate_specs:
        group = scored.loc[scored["period"].eq(period) & scored["top_loop"].eq(loop) & scored["anchor_state"].eq(state)]
        name = f"{loop}|state{state}|candidate"
        recorded = cohorts.loc[cohorts["period"].eq(period) & cohorts["cohort"].eq(name)].iloc[0]
        cohort_error = max(cohort_error, abs(float(recorded.mean_net_bps) - float(group["net_return_bps"].mean())))
        auc = auc_score(group["final_positive"].to_numpy(bool), -group["exit_before_frozen_close_hazard_2024"].to_numpy(float))
        auc_recorded = float(bootstraps.loc[bootstraps["period"].eq(period) & bootstraps["candidate"].eq(f"{loop}|state{state}"), "auc_negative_hazard"].iloc[0])
        auc_error = max(auc_error, abs(auc - auc_recorded))
        counts = group["path_class"].value_counts().to_dict()
        recorded_counts = paths.loc[paths["period"].eq(period) & paths["candidate"].eq(f"{loop}|state{state}")].set_index("path_class")["rows"].to_dict()
        path_count_error += sum(abs(int(counts.get(key, 0)) - int(recorded_counts.get(key, 0))) for key in set(counts) | set(recorded_counts))
    metrics["maximum_candidate_mean_net_error_bps"] = cohort_error
    metrics["maximum_hazard_auc_error"] = auc_error
    metrics["path_class_count_error"] = path_count_error
    checks["candidate_metric_replay"] = cohort_error <= 1e-12 and auc_error <= 1e-12
    checks["path_class_replay"] = path_count_error == 0

    deletion_replay_errors = 0
    for row in deletions.itertuples(index=False):
        loop, state_text = str(row.candidate).split("|state")
        group = scored.loc[
            scored["period"].eq(int(row.period))
            & scored["top_loop"].eq(loop)
            & scored["anchor_state"].eq(int(state_text))
            & ~scored["symbol_norm"].eq(str(row.deleted_symbol))
        ]
        auc = auc_score(group["final_positive"].to_numpy(bool), -group["exit_before_frozen_close_hazard_2024"].to_numpy(float))
        if not np.isclose(auc, float(row.negative_hazard_auc), atol=1e-12, rtol=0.0):
            deletion_replay_errors += 1
    metrics["stock_deletion_replay_errors"] = deletion_replay_errors
    checks["stock_deletion_replay"] = deletion_replay_errors == 0

    candidate_cohorts = cohorts.loc[cohorts["cohort"].str.endswith("|candidate")]
    deletion_counts = deletions.assign(hazard_positive=deletions["negative_hazard_auc"].gt(0.5)).groupby(["period", "candidate"])["hazard_positive"].sum()
    replay_checks = {
        "support_at_least_50_all_four_cells": bool(candidate_cohorts["rows"].ge(50).all() and len(candidate_cohorts) == 4),
        "negative_hazard_auc_above_half_all_four_cells": bool(bootstraps["auc_negative_hazard"].gt(0.5).all() and len(bootstraps) == 4),
        "all_four_hazard_endpoints_pass_holm": bool(bootstraps["passes_holm_0_05"].all() and len(bootstraps) == 4),
        "orientation_survival_difference_positive_all_four_cells": bool(candidate_cohorts["survival_mean_net_difference_bps"].gt(0.0).all() and len(candidate_cohorts) == 4),
        "hazard_leave_one_stock_out_at_least_16_all_four_cells": bool(deletion_counts.ge(16).all() and len(deletion_counts) == 4),
    }
    checks["decision_replay"] = replay_checks == decision["checks"]

    audit = {
        "audit": "independent_loop_payoff_phase_path_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "pass": bool(all(checks.values())),
        "metrics": metrics,
    }
    write_json(out / "independent_audit.json", audit)
    rebuild_manifest(out)
    if not audit["pass"]:
        raise AssertionError(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
