#!/usr/bin/env python3
"""Corrected research-only loop payoff phase/path diagnostic (190 sessions)."""

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
CONTRACT_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v2.json"
PRE_SCORE_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v2-pre-score.json"
BASE_RUNNER = HERE / "run_loop_payoff_phase_path_v1.py"
SPEC = importlib.util.spec_from_file_location("loop_phase_path_v1_base", BASE_RUNNER)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
PERIODS = (2023, 2025)
SEED = 20260713
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5


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


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract_v2": CONTRACT_PATH,
        "runner_v2": Path(__file__).resolve(),
        "auditor_v2": HERE / "audit_loop_payoff_phase_path_v2.py",
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
    for period in PERIODS:
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = BASE.provider_path(root, symbol)
    return paths


def load_and_verify() -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["broker_connection_enabled"] is False
        and contract["paper_or_demo_execution_enabled"] is False
        and contract["deployment_enabled"] is False
        and contract["strategy_promotion_allowed"] is False
        and contract["application_code_modification_allowed"] is False
        and contract["sealed_data_status"]["genuinely_unseen_sessions_available"] is False
    ):
        raise AssertionError("research-only safety boundary drift")
    actual = {name: sha256(path) for name, path in source_paths(contract).items()}
    if actual != pre_score["sha256"]:
        changed = sorted(name for name in set(actual) | set(pre_score["sha256"]) if actual.get(name) != pre_score["sha256"].get(name))
        raise AssertionError(f"pre-score source hash mismatch: {changed}")
    return contract, actual


def fast_hazard_bootstrap(group: pd.DataFrame, seed_offset: int) -> dict[str, Any]:
    sessions = sorted(group["session_date"].astype(str).unique())
    labels = group["final_positive"].to_numpy(bool)
    scores = -group["exit_before_frozen_close_hazard_2024"].to_numpy(float)
    dates = group["session_date"].astype(str).to_numpy()
    indices = {date: np.flatnonzero(dates == date) for date in sessions}
    observed = BASE.auc_score(labels, scores)
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    n = len(sessions)
    blocks = int(math.ceil(n / BOOTSTRAP_BLOCK))
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, n, size=blocks)
        sampled = [sessions[(int(start) + offset) % n] for start in starts for offset in range(BOOTSTRAP_BLOCK)][:n]
        take = np.concatenate([indices[date] for date in sampled])
        values[draw] = BASE.auc_score(labels[take], scores[take])
    values = values[np.isfinite(values)]
    return {
        "auc_negative_hazard": observed,
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.5)) / (len(values) + 1)),
        "draws_valid": int(len(values)),
        "sessions": len(sessions),
    }


def artifact_manifest(out: Path) -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "artifact_manifest.json")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    contract, source_hashes = load_and_verify()
    args.out.mkdir(parents=True)
    cycles = BASE.load_cycles(Path(contract["inputs"]["fixed_cycles"]))
    anchors = pd.concat([
        BASE.load_anchor_panel(Path(contract["inputs"]["anchor_panels"][str(period)]), period, cycles)
        for period in PERIODS
    ], ignore_index=True)
    ledger = pd.read_parquet(Path(contract["inputs"]["accepted_signal_ledger"]))
    ledger["period"] = pd.to_numeric(ledger["period"], errors="raise").astype(int)
    ledger["session_date"] = ledger["session_date"].astype(str)
    source = ledger.loc[
        ledger["period"].isin(PERIODS)
        & ledger["strategy"].eq(contract["population"]["source_strategy"])
        & ledger["horizon"].eq(24)
        & ledger["status"].eq("filled")
    ].copy()
    source = source.merge(
        anchors,
        on=["period", "anchor_id", "symbol_norm", "session_date", "start_timestamp"],
        how="left",
        validate="many_to_one",
    )
    if source["top_loop"].isna().any():
        raise AssertionError("anchor context join failure")
    runs_2024 = BASE.load_runs(Path(contract["inputs"]["runs"]["2024_hazard_fit"]), 2024)
    hazard_fit = BASE.hazard_training(runs_2024)
    enriched: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    score_calendar: dict[str, list[str]] = {}
    warmup = int(contract["population"]["warmup_completed_sessions"])
    for period in PERIODS:
        symbols = list(contract["population"]["symbols"])
        tape, tape_audit = BASE.load_tape(Path(contract["inputs"]["provider_roots"][str(period)]), symbols, period)
        coverage.extend(tape_audit)
        all_sessions = sorted({session for _, session in tape})
        if len(all_sessions) != 250:
            raise AssertionError(f"provider session count drift {period}: {len(all_sessions)}")
        score_sessions = all_sessions[warmup:]
        if len(score_sessions) != 190:
            raise AssertionError("score session count drift")
        score_calendar[str(period)] = score_sessions
        period_source = source.loc[source["period"].eq(period) & source["session_date"].isin(score_sessions)].copy()
        runs = BASE.load_runs(Path(contract["inputs"]["runs"][str(period)]), period)
        enriched.append(BASE.enrich_signals(period_source, period, tape, BASE.run_lookup(runs), hazard_fit))
    scored = pd.concat(enriched, ignore_index=True).sort_values(
        ["period", "session_date", "symbol_norm", "bar_ordinal"], kind="stable"
    ).reset_index(drop=True)
    scored.to_parquet(args.out / "signal_level_path_diagnostics.parquet", index=False)
    pd.DataFrame(coverage).to_csv(args.out / "data_coverage.csv", index=False)
    write_json(args.out / "score_calendar.json", score_calendar)

    cohort_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    specs = [("cycle_04", 4, 2), ("cycle_07", 5, 6)]
    seed_offset = 0
    for period in PERIODS:
        period_frame = scored.loc[scored["period"].eq(period)]
        cohort_rows.append(BASE.summarize_subset("unfiltered_frozen_signal", period, period_frame))
        for loop, candidate_state, control_state in specs:
            loop_frame = period_frame.loc[period_frame["top_loop"].eq(loop)]
            candidate = period_frame.loc[BASE.candidate_mask(period_frame, loop, candidate_state)].copy()
            control = period_frame.loc[BASE.candidate_mask(period_frame, loop, control_state)].copy()
            cohort_rows.extend([
                BASE.summarize_subset(f"{loop}|loop_only", period, loop_frame),
                BASE.summarize_subset(f"{loop}|state{candidate_state}|candidate", period, candidate),
                BASE.summarize_subset(f"{loop}|state{control_state}|matched_control", period, control),
            ])
            expected_key = f"{period}|{loop}|state{candidate_state}"
            if len(candidate) != int(contract["population"]["parent_expected_candidate_counts"][expected_key]):
                raise AssertionError(f"parent candidate population mismatch {expected_key}: {len(candidate)}")
            for outcome, group in candidate.groupby("path_class", sort=True):
                path_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "path_class": outcome,
                    "rows": int(len(group)), "share": float(len(group) / len(candidate)),
                    "mean_net_bps": float(group["net_return_bps"].mean()), "mean_mfe_bps": float(group["mfe_bps"].mean()),
                    "mean_mae_bps": float(group["mae_bps"].mean()), "median_time_to_mfe_bars": float(group["time_to_mfe_bars"].median()),
                    "median_time_to_mae_bars": float(group["time_to_mae_bars"].median()), "mean_mfe_atr": float(group["mfe_atr"].mean()),
                    "mean_mae_atr": float(group["mae_atr"].mean()),
                })
            for profitable, group in candidate.groupby("final_positive", sort=True):
                feature_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "outcome": "positive" if profitable else "nonpositive",
                    "rows": int(len(group)), "mean_net_bps": float(group["net_return_bps"].mean()),
                    "median_regime_age_bars": float(group["admission_regime_age_bars"].median()),
                    "mean_duration_percentile": float(group["duration_percentile_2024"].mean()),
                    "mean_exit_hazard": float(group["exit_before_frozen_close_hazard_2024"].mean()),
                    "orientation_survival_rate": float(group["orientation_survived_to_admission"].mean()),
                    "mean_mfe_bps": float(group["mfe_bps"].mean()), "mean_mae_bps": float(group["mae_bps"].mean()),
                })
            for quarter, group in candidate.groupby("quarter", sort=True):
                quarter_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "quarter": quarter, "rows": int(len(group)),
                    "mean_net_bps": float(group["net_return_bps"].mean()),
                    "negative_hazard_auc": BASE.auc_score(group["final_positive"], -group["exit_before_frozen_close_hazard_2024"]),
                    "orientation_survival_rate": float(group["orientation_survived_to_admission"].mean()),
                })
            bootstrap_rows.append({
                "period": period, "candidate": f"{loop}|state{candidate_state}", "rows": len(candidate),
                **fast_hazard_bootstrap(candidate, seed_offset),
            })
            seed_offset += 1
            for deleted in contract["population"]["symbols"]:
                subset = candidate.loc[~candidate["symbol_norm"].eq(deleted)]
                survive = subset["orientation_survived_to_admission"].to_numpy(bool)
                survival_diff = math.nan
                if survive.any() and (~survive).any():
                    survival_diff = float(subset.loc[survive, "net_return_bps"].mean() - subset.loc[~survive, "net_return_bps"].mean())
                deletion_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "deleted_symbol": deleted, "rows": len(subset),
                    "negative_hazard_auc": BASE.auc_score(subset["final_positive"], -subset["exit_before_frozen_close_hazard_2024"]),
                    "orientation_survival_mean_net_difference_bps": survival_diff,
                })

    cohorts = pd.DataFrame(cohort_rows)
    features = pd.DataFrame(feature_rows)
    paths = pd.DataFrame(path_rows)
    quarters = pd.DataFrame(quarter_rows)
    bootstraps = BASE.holm_adjust(pd.DataFrame(bootstrap_rows))
    deletions = pd.DataFrame(deletion_rows)
    cohorts.to_csv(args.out / "cohort_metrics.csv", index=False)
    features.to_csv(args.out / "profitable_vs_losing_features.csv", index=False)
    paths.to_csv(args.out / "path_class_metrics.csv", index=False)
    quarters.to_csv(args.out / "quarter_metrics.csv", index=False)
    bootstraps.to_csv(args.out / "hazard_auc_bootstraps.csv", index=False)
    deletions.to_csv(args.out / "stock_deletion_metrics.csv", index=False)
    candidates = cohorts.loc[cohorts["cohort"].str.endswith("|candidate")]
    deletion_checks = deletions.assign(
        hazard_positive=deletions["negative_hazard_auc"].gt(0.5),
        survival_positive=deletions["orientation_survival_mean_net_difference_bps"].gt(0),
    ).groupby(["period", "candidate"], as_index=False).agg(
        hazard_positive_deletions=("hazard_positive", "sum"),
        survival_positive_deletions=("survival_positive", "sum"),
    )
    checks = {
        "support_at_least_50_all_four_cells": bool(candidates["rows"].ge(50).all() and len(candidates) == 4),
        "negative_hazard_auc_above_half_all_four_cells": bool(bootstraps["auc_negative_hazard"].gt(0.5).all() and len(bootstraps) == 4),
        "all_four_hazard_endpoints_pass_holm": bool(bootstraps["passes_holm_0_05"].all() and len(bootstraps) == 4),
        "orientation_survival_difference_positive_all_four_cells": bool(candidates["survival_mean_net_difference_bps"].gt(0).all() and len(candidates) == 4),
        "hazard_leave_one_stock_out_at_least_16_all_four_cells": bool(deletion_checks["hazard_positive_deletions"].ge(16).all() and len(deletion_checks) == 4),
    }
    decision = {
        "research_only": True, "live_ordering_enabled": False, "order_placement": "disabled",
        "application_modified": False, "sealed_validation_performed": False,
        "prospective_validation_claim": False, "economic_edge_claim": False, "strategy_promotion": False,
        "primary_cost_bps_per_side": 5, "checks": checks,
        "decision": "phase_hazard_features_supported_for_prospective_logging_only" if all(checks.values()) else "phase_hazard_features_not_supported_as_payoff_admission_discriminator",
    }
    write_json(args.out / "decision.json", decision)
    write_json(args.out / "summary.json", {
        "contract_id": contract["contract_id"], "scientific_status": contract["scientific_status"],
        "v1_correction": contract["correction_from_v1"], "sealed_data_status": contract["sealed_data_status"],
        "research_only": True, "live_ordering_enabled": False, "order_placement": "disabled",
        "provider_volume_label": contract["evidence_labels"]["volume"], "quotes_or_ticks_used": False,
        "score_rows": len(scored), "score_sessions_by_period": {str(k): len(v) for k, v in score_calendar.items()},
        "candidate_metrics": candidates.to_dict("records"), "hazard_bootstraps": bootstraps.to_dict("records"),
        "deletion_checks": deletion_checks.to_dict("records"), "path_metrics": paths.to_dict("records"), "decision": decision,
    })
    write_json(args.out / "source_hashes.json", {
        "contract_id": contract["contract_id"], "frozen_before_scoring": True,
        "research_only": True, "live_ordering_enabled": False, "order_placement": "disabled", "sha256": source_hashes,
    })
    write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
    print(json.dumps({"out": str(args.out), "decision": decision["decision"], "rows": len(scored), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
