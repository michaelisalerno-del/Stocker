#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent artifact audit for pair-closure history V1.

The auditor intentionally does not import the runner or the candidate research
module.  It reconstructs target, chronology, metrics, hashes, and safety checks
from serialized artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
EXPERIMENT_ID = "20260720-immediate-pair-closure-history-v1"
CONTRACT_PATH = WORK_DIR / "contracts" / f"{EXPERIMENT_ID}.json"
PRIMARY_DIR = WORK_DIR / "artifacts" / EXPERIMENT_ID / "primary"
EXACT_DIR = WORK_DIR / "artifacts" / EXPERIMENT_ID / "exact_rerun"
AUDIT_REPORT = WORK_DIR / "reports" / f"{EXPERIMENT_ID}-independent-audit.md"
EXPECTED_SAFETY = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
    "live_ordering_enabled": False,
    "protected_2026_opened": False,
    "promotable": False,
}
FORBIDDEN_CALLS = (
    "placeOrder",
    "reqOpenOrders",
    "reqAllOpenOrders",
    "reqExecutions",
    "reqPositions",
    "reqAccountUpdates",
    "reqAccountSummary",
    "globalCancel",
)
SOURCE_FILES = (
    REPO_ROOT / "packages/stocker_research/src/stocker_research/pair_closure_history_v1.py",
    WORK_DIR / "run_immediate_pair_closure_history_v1.py",
    WORK_DIR / "audit_immediate_pair_closure_history_v1.py",
    CONTRACT_PATH,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(
        json.dumps(payload, sort_keys=True, indent=2, default=str).encode("utf-8") + b"\n"
    )


def _log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(truth, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))


def _brier(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        np.mean(np.square(np.asarray(truth, dtype=float) - np.asarray(probability, dtype=float)))
    )


def _auc(truth: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(truth, dtype=int)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = pd.Series(probability).rank(method="average").to_numpy(dtype=float)
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _calibration_error(truth: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(truth, dtype=float)
    p = np.asarray(probability, dtype=float)
    indices = np.minimum((np.clip(p, 0.0, 1.0) * 10).astype(int), 9)
    result = 0.0
    for index in range(10):
        mask = indices == index
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def _top_decile_lift(group: pd.DataFrame) -> float:
    truth = group["target_pair_closure"].to_numpy(dtype=float)
    probability = group["probability"].to_numpy(dtype=float)
    count = max(1, int(math.ceil(0.10 * len(group))))
    order = np.lexsort((group["decision_id"].astype(str).to_numpy(), -probability))
    return float(truth[order[:count]].mean() - truth.mean())


def _artifact_hash_checks(directory: Path) -> list[dict[str, Any]]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for record in manifest["artifacts"]:
        path = directory / str(record["file"])
        actual = _sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "check": f"artifact_hash:{record['file']}",
                "passed": actual == record["sha256"]
                and path.stat().st_size == int(record["bytes"]),
                "expected": record["sha256"],
                "actual": actual,
            }
        )
    return rows


def _paired(predictions: pd.DataFrame) -> pd.DataFrame:
    names = ("M2_IMMEDIATE_PAIR", "M5_LAST_FIVE_STATES")
    source = predictions.loc[predictions["model"].isin(names)].copy()
    keys = [
        "decision_id",
        "representation",
        "evaluation_period",
        "symbol",
        "session",
        "decision_timestamp",
        "target_pair_closure",
    ]
    wide = source.pivot(index=keys, columns="model", values="probability").reset_index()
    truth = wide["target_pair_closure"].to_numpy(dtype=float)
    for model in names:
        p = np.clip(wide[model].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        wide[f"{model}_loss"] = -(truth * np.log(p) + (1.0 - truth) * np.log1p(-p))
        wide[f"{model}_brier"] = np.square(truth - p)
    wide["log_loss_improvement"] = wide["M2_IMMEDIATE_PAIR_loss"] - wide["M5_LAST_FIVE_STATES_loss"]
    wide["brier_improvement"] = wide["M2_IMMEDIATE_PAIR_brier"] - wide["M5_LAST_FIVE_STATES_brier"]
    return wide


def audit(directory: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, expected: Any, actual: Any) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "expected": expected, "actual": actual}
        )

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (directory / "source_identity_manifest.json").read_text(encoding="utf-8")
    )
    configuration = json.loads(
        (directory / "model_effective_configuration.json").read_text(encoding="utf-8")
    )
    for key, expected in EXPECTED_SAFETY.items():
        record(f"contract_safety:{key}", contract.get(key) == expected, expected, contract.get(key))
        record(f"metadata_safety:{key}", metadata.get(key) == expected, expected, metadata.get(key))
    record(
        "blocked_part_b_not_reopened",
        metadata.get("part_b_contract_reopened") is False,
        False,
        metadata.get("part_b_contract_reopened"),
    )
    record(
        "numeric_semantic_validity_not_claimed",
        metadata.get("numeric_state_semantic_validity_claimed") is False,
        False,
        metadata.get("numeric_state_semantic_validity_claimed"),
    )
    checks.extend(_artifact_hash_checks(directory))
    record(
        "contract_hash",
        metadata["contract_hash"] == _sha256_file(CONTRACT_PATH),
        _sha256_file(CONTRACT_PATH),
        metadata["contract_hash"],
    )
    implementation = hashlib.sha256()
    for path in SOURCE_FILES:
        relative = path.relative_to(REPO_ROOT)
        implementation.update(relative.as_posix().encode("utf-8"))
        implementation.update(b"\0")
        implementation.update(path.read_bytes())
    record(
        "implementation_hash",
        implementation.hexdigest() == metadata["implementation_hash"],
        implementation.hexdigest(),
        metadata["implementation_hash"],
    )
    record(
        "development_source_identity",
        source_manifest["development"]["data_snapshot_hash"]
        == "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
        and int(source_manifest["development"]["rows"]) == 424_583,
        "424583 rows at frozen 2024 snapshot",
        source_manifest["development"],
    )
    record(
        "assessment_source_identity",
        source_manifest["assessment"]["data_snapshot_hash"]
        == "29e82d6539810e5fcebc13e860d07474c38ee0349fe38aedce0378f9aefb67a4"
        and int(source_manifest["assessment"]["rows"]) == 424_827,
        "424827 rows at frozen 2025 snapshot",
        source_manifest["assessment"],
    )

    population = pd.read_parquet(directory / "pair_closure_population.parquet")
    population["decision_timestamp"] = pd.to_datetime(population["decision_timestamp"], utc=True)
    population["target_available_timestamp"] = pd.to_datetime(
        population["target_available_timestamp"], utc=True
    )
    record(
        "protected_2026_absent",
        int(population["decision_timestamp"].dt.year.max()) == 2025,
        2025,
        int(population["decision_timestamp"].dt.year.max()),
    )
    record(
        "stable_unique_decision_ids",
        not population["decision_id"].duplicated().any(),
        True,
        not population["decision_id"].duplicated().any(),
    )
    record(
        "compressed_pair_has_distinct_states",
        population["previous_state_1"].ne(population["current_state"]).all(),
        True,
        population["previous_state_1"].ne(population["current_state"]).all(),
    )
    available = population["target_available"].astype(bool)
    reconstructed_target = population.loc[available, "next_state"].eq(
        population.loc[available, "previous_state_1"]
    )
    record(
        "independent_pair_closure_target",
        np.array_equal(
            reconstructed_target.to_numpy(dtype=int),
            population.loc[available, "target_pair_closure"].to_numpy(dtype=int),
        ),
        True,
        "recomputed from next_state == previous_state_1",
    )
    record(
        "target_strictly_after_decision",
        (
            population.loc[available, "target_available_timestamp"]
            > population.loc[available, "decision_timestamp"]
        ).all(),
        True,
        "timestamp comparison",
    )
    forbidden_scientific_columns = {
        "future_return",
        "pnl",
        "profit",
        "mfe",
        "mae",
        "spread",
        "slippage",
        "price_target",
    }
    record(
        "no_economic_or_price_target_columns",
        not forbidden_scientific_columns.intersection(population.columns),
        [],
        sorted(forbidden_scientific_columns.intersection(population.columns)),
    )
    record(
        "source_artifacts_are_logical_not_local_paths",
        not population["source_artifact"].astype(str).str.startswith("/").any(),
        True,
        population["source_artifact"].astype(str).head(3).tolist(),
    )

    development_predictions = pd.read_parquet(directory / "development_oof_predictions.parquet")
    assessment_predictions = pd.read_parquet(directory / "assessment_predictions.parquet")
    predictions = pd.concat([development_predictions, assessment_predictions], ignore_index=True)
    prediction_population = population.loc[
        population["target_available"].astype(bool),
        ["decision_id", "target_pair_closure"],
    ]
    joined = predictions.merge(
        prediction_population,
        on="decision_id",
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    record(
        "prediction_targets_match_population",
        joined["target_pair_closure_source"].notna().all()
        and np.array_equal(
            joined["target_pair_closure"].to_numpy(dtype=int),
            joined["target_pair_closure_source"].to_numpy(dtype=int),
        ),
        True,
        "independent join",
    )
    expected_models = set(configuration["model_context_levels"])
    record(
        "frozen_model_ladder_complete",
        set(predictions["model"].astype(str)) == expected_models,
        sorted(expected_models),
        sorted(predictions["model"].astype(str).unique()),
    )
    chronology_ok = True
    development_available = population.loc[
        population["period"].eq("DEVELOPMENT_2024") & population["target_available"].astype(bool)
    ].copy()
    development_available["month"] = development_available["decision_timestamp"].dt.strftime(
        "%Y-%m"
    )
    for (representation, month), group in development_predictions.groupby(
        ["representation", "score_month"], sort=True
    ):
        expected_train = int(
            (
                development_available["representation"].eq(representation)
                & development_available["month"].lt(month)
            ).sum()
        )
        chronology_ok &= group["training_rows"].eq(expected_train).all()
    record("expanding_fold_training_counts", chronology_ok, True, chronology_ok)
    assessment_available = population.loc[
        population["period"].eq("ASSESSMENT_2025") & population["target_available"].astype(bool)
    ]
    assessment_training_ok = True
    for representation, group in assessment_predictions.groupby("representation", sort=True):
        expected_train = int((development_available["representation"].eq(representation)).sum())
        assessment_training_ok &= group["training_rows"].eq(expected_train).all()
    assessment_training_ok &= set(
        pd.to_datetime(assessment_predictions["decision_timestamp"], utc=True).dt.year.unique()
    ) == {2025}
    record(
        "assessment_fit_uses_2024_count_only",
        assessment_training_ok,
        True,
        assessment_training_ok,
    )
    record(
        "assessment_rows_complete",
        assessment_predictions["decision_id"].nunique() == len(assessment_available),
        len(assessment_available),
        assessment_predictions["decision_id"].nunique(),
    )

    stored_metrics = pd.read_csv(directory / "model_metrics.csv")
    metric_ok = True
    for keys, group in predictions.groupby(
        ["representation", "evaluation_period", "model"], sort=True
    ):
        stored = stored_metrics.loc[
            stored_metrics["representation"].eq(keys[0])
            & stored_metrics["evaluation_period"].eq(keys[1])
            & stored_metrics["model"].eq(keys[2])
        ]
        if len(stored) != 1:
            metric_ok = False
            continue
        row = stored.iloc[0]
        truth = group["target_pair_closure"].to_numpy(dtype=int)
        probability = group["probability"].to_numpy(dtype=float)
        metric_ok &= math.isclose(_log_loss(truth, probability), row["log_loss"], abs_tol=1e-11)
        metric_ok &= math.isclose(_brier(truth, probability), row["brier_score"], abs_tol=1e-11)
        metric_ok &= math.isclose(_auc(truth, probability), row["roc_auc"], abs_tol=1e-11)
        metric_ok &= math.isclose(
            _calibration_error(truth, probability), row["calibration_error"], abs_tol=1e-11
        )
        metric_ok &= math.isclose(_top_decile_lift(group), row["top_decile_lift"], abs_tol=1e-11)
    record("independent_model_metrics", metric_ok, True, metric_ok)

    bootstrap_summary = pd.read_csv(directory / "paired_session_bootstrap_summary.csv")
    paired_point_ok = True
    for keys, group in predictions.groupby(["representation", "evaluation_period"], sort=True):
        paired = _paired(group)
        stored = bootstrap_summary.loc[
            bootstrap_summary["representation"].eq(keys[0])
            & bootstrap_summary["evaluation_period"].eq(keys[1])
        ]
        if len(stored) != 1:
            paired_point_ok = False
            continue
        row = stored.iloc[0]
        paired_point_ok &= math.isclose(
            paired["log_loss_improvement"].mean(), row["log_loss_improvement"], abs_tol=1e-11
        )
        paired_point_ok &= math.isclose(
            paired["brier_improvement"].mean(), row["brier_improvement"], abs_tol=1e-11
        )
        paired_point_ok &= int(row["sessions"]) == paired["session"].nunique()
    record("paired_metric_point_estimates", paired_point_ok, True, paired_point_ok)

    pair_metrics = pd.read_csv(directory / "pair_orientation_metrics.csv")
    pair_point_ok = True
    for row in pair_metrics.itertuples(index=False):
        source = population.loc[
            population["period"].eq(row.period)
            & population["representation"].eq(row.representation)
            & population["target_available"].astype(bool)
        ]
        pair = source.loc[
            source["previous_state_1"].eq(row.previous_state)
            & source["current_state"].eq(row.current_state)
        ]
        baseline = source.loc[
            source["current_state"].eq(row.current_state)
            & source["previous_state_1"].ne(row.previous_state)
        ]
        pair_point_ok &= len(pair) == int(row.rows)
        pair_point_ok &= math.isclose(
            pair["target_pair_closure"].mean(), row.closure_rate, abs_tol=1e-11
        )
        if len(baseline):
            pair_point_ok &= math.isclose(
                pair["target_pair_closure"].mean() - baseline["target_pair_closure"].mean(),
                row.closure_lift,
                abs_tol=1e-11,
            )
    record("independent_pair_orientation_points", pair_point_ok, True, pair_point_ok)

    exact_path = directory / "exact_rerun_manifest.json"
    exact = json.loads(exact_path.read_text(encoding="utf-8")) if exact_path.is_file() else {}
    exact_hashes_ok = bool(exact.get("byte_identical"))
    for comparison in exact.get("comparisons", []):
        primary = PRIMARY_DIR / comparison["file"]
        rerun = EXACT_DIR / comparison["file"]
        exact_hashes_ok &= (
            _sha256_file(primary) == comparison["primary_sha256"]
            and _sha256_file(rerun) == comparison["exact_rerun_sha256"]
            and comparison["primary_sha256"] == comparison["exact_rerun_sha256"]
        )
    record("exact_rerun_hashes", exact_hashes_ok, True, exact_hashes_ok)

    implementation_text = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_FILES[:2])
    forbidden_found = [value for value in FORBIDDEN_CALLS if value in implementation_text]
    record("no_order_capable_calls", not forbidden_found, [], forbidden_found)
    credential_tokens = [
        value
        for value in ("account_number", "account_id", "api_token", "password=", "secret_key")
        if value in implementation_text.lower()
    ]
    record("no_credentials_or_account_identifiers", not credential_tokens, [], credential_tokens)

    audit_passed = all(bool(row["passed"]) for row in checks)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "audit_passed": audit_passed,
        "check_count": len(checks),
        "failed_checks": [row["check"] for row in checks if not bool(row["passed"])],
        "checks": checks,
        "auditor_independent_of_runner_import": True,
        "auditor_independent_of_candidate_module_import": True,
        **EXPECTED_SAFETY,
    }
    _write_json(directory / "independent_audit.json", payload)
    lines = [
        "# Immediate Pair-Closure History V1 — Independent Audit",
        "",
        f"Audit passed: `{audit_passed}`. Checks: `{len(checks)}`.",
        "",
        "The auditor independently reconstructed structural targets, chronology counts, prediction metrics, paired point estimates, pair-orientation point estimates, artifact hashes, exact-rerun identity, and safety scans.",
        "",
        "Failed checks:",
        "",
        *(f"- `{value}`" for value in payload["failed_checks"]),
    ]
    if not payload["failed_checks"]:
        lines.append("- None.")
    AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not audit_passed:
        raise RuntimeError(f"independent audit failed: {payload['failed_checks']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=PRIMARY_DIR)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.artifact_dir.resolve()),
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
