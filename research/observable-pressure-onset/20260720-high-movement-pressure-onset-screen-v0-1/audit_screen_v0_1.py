"""Independent audit for the V0.1 pressure-onset support-semantics repair.

This auditor does not import either experiment runner or the reusable research
module. It reuses only the immutable V0 *auditor* for the unchanged causal-bar,
feature, threshold, metric, and bootstrap checks.
"""

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
V0_DIR = EXPERIMENT_DIR.parent / "20260720-high-movement-pressure-onset-screen-v0"
V0_PRIMARY = V0_DIR / "artifacts" / "primary"
V0_EXACT = V0_DIR / "artifacts" / "exact_rerun"
V0_AUDITOR_PATH = V0_DIR / "audit_screen_v0.py"
V0_RUNNER_PATH = V0_DIR / "run_screen_v0.py"
V0_CONTRACT_PATH = V0_DIR / "contract.json"
V0_COMMIT = "8f3954fdcd1db480bff1a0c535b1ea6fa5f71f13"
V0_REUSABLE_LOGICAL_PATH = (
    "packages/stocker_research/src/stocker_research/pressure_onset_screen_v0.py"
)
V0_REUSABLE_BLOB_SHA256 = "4ca273edd426716b1917c875efbc56cc32d80ba7880b6585dde8bed3591be10f"
V0_EXACT_MANIFEST_SHA256 = "18647bfa3cc63eb15c8a3e16219a894e7b1bfd9b1703dc4452e0b3c1af5a539c"
V0_IMMUTABLE_FILE_HASHES = {
    V0_RUNNER_PATH: "01f7535f659e1dc03f982199371be243e46647667ac22c8a3972d3f1f498a058",
    V0_AUDITOR_PATH: "f1385ffdd01d9787f48624fcbc12794c245d99015b7804063a1f567cb7e5069a",
    V0_CONTRACT_PATH: "fca26a226b3c1f38bbe19815f49c9d948a55724221a2bf94976bc82c6a1f6f08",
    V0_PRIMARY / "exact_rerun_manifest.json": V0_EXACT_MANIFEST_SHA256,
    V0_EXACT / "exact_rerun_manifest.json": V0_EXACT_MANIFEST_SHA256,
}
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
SCIENTIFIC_STATUS = "opened_support_contract_repair_retrospective_feasibility_evidence"
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "observable_only": True,
    "support_contract_repair": True,
    "opened_support_counts_only": True,
    "model_results_previously_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}
EXTRA_REQUIRED = (
    "support_contract_repair.json",
    "parent_slate_accounting.csv",
    "admitted_slate_accounting.csv",
    "singleton_slate_metrics.csv",
    "multi_candidate_slate_metrics.csv",
    "weight_audit.csv",
    "largest_stock_deletion_metrics.csv",
    "leave_one_stock_out_diagnostics.csv",
    "v0_vs_v0_1_population_comparison.json",
    "v0_source_artifact_hashes.json",
)
MODEL_DEPENDENT_CONFIRMATION_COLUMNS = {
    "favourable_retracement_bps",
    "predicted_direction_remained_same",
}


class AuditFailure(RuntimeError):
    """A failed V0.1 integrity assertion."""


def load_v0_auditor() -> ModuleType:
    """Load the immutable independent V0 auditor, never its runner."""

    for path, expected in V0_IMMUTABLE_FILE_HASHES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if digest != expected:
            raise AuditFailure(f"immutable V0 lineage hash differs: {path.name}")
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != V0_REUSABLE_BLOB_SHA256:
        raise AuditFailure("immutable V0 reusable-module blob differs")
    name = "stocker_pressure_onset_v0_independent_auditor"
    specification = importlib.util.spec_from_file_location(name, V0_AUDITOR_PATH)
    if specification is None or specification.loader is None:
        raise AuditFailure("immutable V0 auditor could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


V0_AUDIT = load_v0_auditor()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AuditFailure(detail)


def maximum_error(actual: Sequence[float], expected: Sequence[float]) -> float:
    return float(V0_AUDIT.maximum_error(actual, expected))


def weighted_fit(
    frame: pd.DataFrame, labels: pd.Series, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Independently fit the frozen model using repaired precomputed weights."""

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    weights = frame["row_weight"].to_numpy(dtype=float)
    require(bool(np.isfinite(values).all()), "non-finite weighted-fit feature")
    require(bool(np.isfinite(weights).all() and (weights > 0.0).all()), "invalid row weight")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        labels.to_numpy(dtype=int),
        sample_weight=weights,
    )
    require(int(np.max(estimator.n_iter_)) < 250, "audited weighted fit did not converge")
    return means, scales, estimator


def fitted_predict(
    model: tuple[np.ndarray, np.ndarray, LogisticRegression],
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    means, scales, estimator = model
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    return np.asarray(estimator.predict_proba((values - means) / scales)[:, 1], dtype=float)


def t1_scoring_frame(frame: pd.DataFrame, d2_features: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for feature in d2_features:
        if feature in {"checkpoint_60m", "p_large_remaining_move"}:
            continue
        source = f"t1__{feature}"
        require(source in frame, f"t+1 source missing: {source}")
        output[feature] = frame[source]
    return output


def attach_confirmation(
    frame: pd.DataFrame, probability_t: np.ndarray, probability_t1: np.ndarray
) -> pd.DataFrame:
    """Independently reconstruct both model-dependent confirmation features."""

    output = frame.copy()
    direction_t = np.where(probability_t >= 0.5, 1.0, -1.0)
    direction_t1 = np.where(probability_t1 >= 0.5, 1.0, -1.0)
    output["predicted_direction_remained_same"] = (direction_t == direction_t1).astype(float)
    base_close = output["calc__current_close"].to_numpy(dtype=float)
    next_high = output["t1__current_high"].to_numpy(dtype=float)
    next_low = output["t1__current_low"].to_numpy(dtype=float)
    next_close = output["t1__current_close"].to_numpy(dtype=float)
    long_favourable = 10_000.0 * (next_high / base_close - 1.0)
    long_progress = 10_000.0 * (next_close / base_close - 1.0)
    short_favourable = 10_000.0 * (1.0 - next_low / base_close)
    short_progress = 10_000.0 * (1.0 - next_close / base_close)
    favourable = np.where(direction_t > 0.0, long_favourable, short_favourable)
    progress = np.where(direction_t > 0.0, long_progress, short_progress)
    output["favourable_retracement_bps"] = np.maximum(0.0, np.maximum(0.0, favourable) - progress)
    require(
        bool(
            pd.to_datetime(output["confirmation_available_timestamp"], utc=True)
            .le(pd.to_datetime(output["entry_timestamp"], utc=True))
            .all()
        ),
        "confirmation reaches beyond delayed entry",
    )
    return output


def fit_ladder(
    development: pd.DataFrame, configurations: Mapping[str, Any]
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, LogisticRegression]],
    pd.DataFrame,
]:
    features = configurations["models"]
    models: dict[str, tuple[np.ndarray, np.ndarray, LogisticRegression]] = {}
    for name in ("A0", "A1", "A2"):
        models[name] = weighted_fit(development, development["directional_onset"], features[name])
    direction = development.loc[development["directional_onset"].eq(1)].copy()
    for name in ("D0", "D1", "D2"):
        models[name] = weighted_fit(direction, direction["up_given_onset"], features[name])
    d2_t = fitted_predict(models["D2"], development, features["D2"])
    d2_t1 = fitted_predict(
        models["D2"], t1_scoring_frame(development, features["D2"]), features["D2"]
    )
    confirmed = attach_confirmation(development, d2_t, d2_t1)
    models["A3"] = weighted_fit(confirmed, confirmed["directional_onset"], features["A3"])
    confirmed_direction = confirmed.loc[confirmed["directional_onset"].eq(1)]
    models["D3"] = weighted_fit(
        confirmed_direction,
        confirmed_direction["up_given_onset"],
        features["D3"],
    )
    return models, confirmed


def score_ladder(
    frame: pd.DataFrame,
    models: Mapping[str, tuple[np.ndarray, np.ndarray, LogisticRegression]],
    configurations: Mapping[str, Any],
) -> pd.DataFrame:
    features = configurations["models"]
    output = frame.copy()
    for name in ("A0", "A1", "A2"):
        output[f"p_onset__{name}"] = fitted_predict(models[name], output, features[name])
    for name in ("D0", "D1", "D2"):
        output[f"p_up_given_onset__{name}"] = fitted_predict(models[name], output, features[name])
    d2_t1 = fitted_predict(models["D2"], t1_scoring_frame(output, features["D2"]), features["D2"])
    output = attach_confirmation(
        output, output["p_up_given_onset__D2"].to_numpy(dtype=float), d2_t1
    )
    output["p_onset__A3"] = fitted_predict(models["A3"], output, features["A3"])
    output["p_up_given_onset__D3"] = fitted_predict(models["D3"], output, features["D3"])
    for system, onset_name, direction_name in (
        ("readiness", "A1", "D1"),
        ("pressure", "A2", "D2"),
        ("confirmed", "A3", "D3"),
    ):
        onset = output[f"p_onset__{onset_name}"]
        upward = output[f"p_up_given_onset__{direction_name}"]
        output[f"signed_pressure_score__{system}"] = (
            onset * (2.0 * upward - 1.0) * output["p_large_remaining_move"]
        )
    return output


def loss_improvement(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    candidate: str,
    kind: str,
) -> float:
    labels = frame[target].to_numpy(dtype=float)
    before = frame[baseline].to_numpy(dtype=float)
    after = frame[candidate].to_numpy(dtype=float)
    if kind == "brier":
        return float(np.mean((labels - before) ** 2) - np.mean((labels - after) ** 2))
    before = np.clip(before, 1e-15, 1.0 - 1e-15)
    after = np.clip(after, 1e-15, 1.0 - 1e-15)
    before_loss = -np.mean(labels * np.log(before) + (1.0 - labels) * np.log(1.0 - before))
    after_loss = -np.mean(labels * np.log(after) + (1.0 - labels) * np.log(1.0 - after))
    return float(before_loss - after_loss)


def increments(primary: pd.DataFrame) -> dict[str, float]:
    direction = primary.loc[primary["directional_onset"].eq(1)]
    output: dict[str, float] = {}
    for candidate, baseline, frame, target, prefix in (
        ("A2", "A1", primary, "directional_onset", "A2_minus_A1"),
        ("D2", "D1", direction, "up_given_onset", "D2_minus_D1"),
        ("A3", "A2", primary, "directional_onset", "A3_minus_A2"),
        ("D3", "D2", direction, "up_given_onset", "D3_minus_D2"),
    ):
        probability_prefix = "p_onset" if candidate.startswith("A") else "p_up_given_onset"
        for kind in ("brier", "log_loss"):
            output[f"{prefix}_{kind}"] = loss_improvement(
                frame,
                target=target,
                baseline=f"{probability_prefix}__{baseline}",
                candidate=f"{probability_prefix}__{candidate}",
                kind=kind,
            )
    return output


def verify_v0_lineage(artifacts: Path, panel: pd.DataFrame) -> dict[str, Any]:
    """Verify source hashes and every frozen pre-model value against immutable V0."""

    source_hashes = read_json(artifacts / "v0_source_artifact_hashes.json")
    expected_immutable_hashes = [
        {"logical_path": str(path.relative_to(REPO_ROOT)), "sha256": digest}
        for path, digest in V0_IMMUTABLE_FILE_HASHES.items()
    ] + [
        {
            "logical_path": f"git:{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}",
            "sha256": V0_REUSABLE_BLOB_SHA256,
        }
    ]
    require(
        source_hashes.get("immutable_lineage_hashes") == expected_immutable_hashes,
        "immutable V0 lineage hash record differs",
    )
    roots = {"primary": V0_PRIMARY, "exact_rerun": V0_EXACT}
    manifest = read_json(V0_PRIMARY / "exact_rerun_manifest.json")
    require(bool(manifest.get("passed")), "hard-anchored V0 exact rerun did not pass")
    expected_artifacts: dict[str, str] = {}
    for comparison in manifest["comparisons"]:
        artifact = str(comparison["artifact"])
        primary_hash = str(comparison["primary_sha256"])
        exact_hash = str(comparison["exact_rerun_sha256"])
        require(
            bool(comparison["passed"]) and primary_hash == exact_hash,
            f"hard-anchored V0 comparison differs: {artifact}",
        )
        expected_artifacts[artifact] = primary_hash
    expected_artifacts["independent_audit.json"] = str(manifest["independent_audit_sha256"])
    expected_artifacts["exact_rerun_manifest.json"] = V0_EXACT_MANIFEST_SHA256
    require(
        set(source_hashes.get("artifact_roots", {})) == set(roots),
        "V0 source artifact roots differ",
    )
    checked = 0
    for root_name, root in roots.items():
        records = source_hashes["artifact_roots"][root_name]
        require(
            {str(record["artifact"]) for record in records} == set(expected_artifacts),
            f"V0 source artifact set differs: {root_name}",
        )
        require(
            {path.name for path in root.iterdir() if path.is_file()} == set(expected_artifacts),
            f"V0 on-disk artifact set differs: {root_name}",
        )
        for record in records:
            path = root / str(record["artifact"])
            require(path.is_file(), f"V0 source artifact missing: {path.name}")
            require(
                V0_AUDIT.sha256_file(path) == record["sha256"] == expected_artifacts[path.name],
                f"V0 source artifact hash differs: {path.name}",
            )
            checked += 1
    v0_panel = pd.read_parquet(V0_PRIMARY / "compact_decision_panel.parquet")
    keys = ["symbol", "session", "decision_ordinal"]
    left = v0_panel.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = panel.sort_values(keys, kind="mergesort").reset_index(drop=True)
    columns = [
        name
        for name in left.columns
        if name not in {"screen_status", *MODEL_DEPENDENT_CONFIRMATION_COLUMNS}
    ]
    pd.testing.assert_frame_equal(
        left.loc[:, columns],
        right.loc[:, columns],
        check_exact=True,
        check_dtype=True,
        check_like=False,
    )
    require(
        read_json(artifacts / "movement_admission_thresholds.json")
        == read_json(V0_PRIMARY / "movement_admission_thresholds.json"),
        "movement thresholds changed from V0",
    )
    require(
        read_json(artifacts / "onset_barriers.json")
        == read_json(V0_PRIMARY / "onset_barriers.json"),
        "onset barriers changed from V0",
    )
    comparison = read_json(artifacts / "v0_vs_v0_1_population_comparison.json")
    require(bool(comparison["passed"]), "saved V0 population comparison failed")
    for name in (
        "movement_probability_maximum_error",
        "admission_flag_differences",
        "onset_label_differences",
        "feature_value_differences",
        "timestamp_differences",
        "qa_exclusion_differences",
    ):
        require(float(comparison[name]) == 0.0, f"frozen comparison differs: {name}")
    require(not bool(comparison["rows_added_from_new_market_data"]), "new market rows added")
    return {
        "v0_artifact_hashes_verified": checked,
        "immutable_lineage_hashes_verified": len(expected_immutable_hashes),
        "frozen_rows_verified": len(right),
        "frozen_columns_verified": len(columns),
        "thresholds_barriers_admissions_labels_and_features_unchanged": True,
    }


def reconstruct_parent_eligibility(predecessor: pd.DataFrame, provider_root: Path) -> pd.DataFrame:
    """Independently reconstruct pre-admission support for every predecessor slate."""

    columns = ["symbol", "session", "year", "year_month", "decision_ordinal", "slate_id"]
    available_parts: list[pd.DataFrame] = []
    for symbol in sorted(predecessor["symbol"].astype(str).unique()):
        bars = V0_AUDIT.prepare_readiness_audit_bars(
            provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        )
        valid_sessions = set(bars["session"].astype(str))
        available_parts.append(
            predecessor.loc[
                predecessor["symbol"].astype(str).eq(symbol)
                & predecessor["session"].astype(str).isin(valid_sessions),
                columns,
            ].copy()
        )
    available = pd.concat(available_parts, ignore_index=True).sort_values(
        ["symbol", "decision_ordinal", "session"], kind="mergesort"
    )
    available["raw_source_stock_count"] = available.groupby("slate_id", sort=True)[
        "symbol"
    ].transform("nunique")
    available["raw_parent_gate_passed"] = available["raw_source_stock_count"].ge(15)
    available["prior_valid_parent_history_count"] = available.groupby(
        ["symbol", "decision_ordinal"], sort=False
    )["raw_parent_gate_passed"].transform(
        lambda values: values.astype(int).cumsum().shift(1, fill_value=0)
    )
    available["complete_causal_feature_history"] = available["prior_valid_parent_history_count"].ge(
        10
    )
    counts = (
        available.groupby("slate_id", sort=True)
        .agg(
            raw_source_stock_count=("symbol", "nunique"),
            history_complete_stock_count=("complete_causal_feature_history", "sum"),
        )
        .reset_index()
    )
    expected = (
        predecessor.loc[:, ["slate_id", "session", "year", "year_month", "decision_ordinal"]]
        .drop_duplicates("slate_id", keep="first")
        .merge(counts, on="slate_id", how="left", validate="one_to_one", sort=False)
    )
    for column in ("raw_source_stock_count", "history_complete_stock_count"):
        expected[column] = expected[column].fillna(0).astype(int)
    return expected.sort_values("slate_id", kind="mergesort").reset_index(drop=True)


def verify_support_hierarchy(
    artifacts: Path,
    panel: pd.DataFrame,
    assessment: pd.DataFrame,
    provider_root: Path,
) -> dict[str, Any]:
    parent = pd.read_csv(artifacts / "parent_slate_accounting.csv")
    admitted = pd.read_csv(artifacts / "admitted_slate_accounting.csv")
    weights = pd.read_csv(artifacts / "weight_audit.csv")
    predecessor = pd.read_parquet(V0_AUDIT.PREDECESSOR_PANEL)
    reconstructed = reconstruct_parent_eligibility(predecessor, provider_root).set_index("slate_id")
    expected_slates = set(predecessor["slate_id"].astype(str).unique())
    require(set(parent["slate_id"].astype(str)) == expected_slates, "parent slates differ")
    require(parent["slate_id"].is_unique, "duplicate parent accounting row")
    require(admitted["slate_id"].is_unique, "duplicate admitted accounting row")
    require(
        parent["slate_id"].astype(str).tolist() == admitted["slate_id"].astype(str).tolist(),
        "parent and admitted accounting keys differ",
    )
    grouped = panel.groupby("slate_id", sort=True)
    group_sizes = grouped.size().astype(int).to_dict()
    group_admitted = grouped["high_movement_admitted"].sum().astype(int).to_dict()
    for row in admitted.itertuples(index=False):
        slate_id = str(row.slate_id)
        reconstructed_row = reconstructed.loc[slate_id]
        require(
            int(row.raw_source_stock_count) == int(reconstructed_row["raw_source_stock_count"]),
            "raw parent source count differs",
        )
        require(
            int(row.history_complete_stock_count)
            == int(reconstructed_row["history_complete_stock_count"]),
            "history-complete parent count differs",
        )
        expected_parent = int(
            group_sizes.get(slate_id, int(reconstructed_row["history_complete_stock_count"]))
        )
        expected_admitted = int(group_admitted.get(slate_id, 0))
        require(int(row.parent_valid_stock_count) == expected_parent, "parent count differs")
        require(int(row.admitted_stock_count) == expected_admitted, "admitted count differs")
        eligible = expected_parent >= 15
        require(bool(row.parent_slate_eligible) == eligible, "parent eligibility differs")
        expected_primary = expected_admitted if eligible else 0
        require(int(row.primary_row_count) == expected_primary, "primary row count differs")
        require(
            bool(row.singleton_admitted_slate) == (eligible and expected_admitted == 1),
            "singleton flag differs",
        )
        require(
            bool(row.multi_candidate_admitted_slate) == (eligible and expected_admitted >= 2),
            "multi flag differs",
        )
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)].copy()
    require(len(primary) == 1_560, "repaired primary row count differs")
    require(primary["session"].nunique() == 153, "repaired primary sessions differ")
    require(primary["symbol"].nunique() == 20, "repaired primary stocks differ")
    labels = primary["onset_label"].value_counts().to_dict()
    require(
        labels == {"NO_ONSET": 879, "UP_ONSET": 345, "DOWN_ONSET": 336},
        "repaired onset support differs",
    )
    expected_weight_rows = panel.loc[
        panel["parent_slate_eligible"].astype(bool) & panel["high_movement_admitted"].astype(bool)
    ].copy()
    weight_keys = ["symbol", "session", "decision_ordinal"]
    require(len(weights) == len(expected_weight_rows), "weight-audit rows differ")
    merged = expected_weight_rows.merge(
        weights,
        on=weight_keys,
        how="inner",
        suffixes=("_panel", "_audit"),
        validate="one_to_one",
    )
    require(len(merged) == len(weights), "weight-audit keys differ")
    require(
        maximum_error(
            merged["row_weight_panel"],
            1.0 / merged["admitted_stock_count_panel"].to_numpy(dtype=float),
        )
        <= 1e-15,
        "row weight denominator differs",
    )
    require(
        maximum_error(merged["row_weight_panel"], merged["row_weight_audit"]) <= 5e-12,
        "saved row weights differ",
    )
    totals = weights.groupby("parent_slate_id", sort=True)["row_weight"].sum()
    require(maximum_error(totals, np.ones(len(totals))) <= 5e-12, "slate weights differ")
    assessment_admitted = admitted.loc[admitted["year"].eq(2025)]
    singleton = int(assessment_admitted["singleton_admitted_slate"].sum())
    multi = int(assessment_admitted["multi_candidate_admitted_slate"].sum())
    no_admission = int(
        (
            assessment_admitted["parent_slate_eligible"].astype(bool)
            & assessment_admitted["admitted_stock_count"].eq(0)
        ).sum()
    )
    require(singleton == 33 and multi == 266 and no_admission == 15, "slate types differ")
    invalid_counts = (
        assessment_admitted.loc[~assessment_admitted["parent_slate_eligible"].astype(bool)][
            "parent_valid_stock_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    require(invalid_counts == {13: 2}, "assessment invalid parent-slate counts differ")
    return {
        "parent_slates": len(parent),
        "assessment_valid_parent_slates": int(
            assessment_admitted["parent_slate_eligible"].astype(bool).sum()
        ),
        "assessment_invalid_parent_slates": int(
            (~assessment_admitted["parent_slate_eligible"].astype(bool)).sum()
        ),
        "assessment_zero_admission_valid_slates": no_admission,
        "assessment_invalid_parent_stock_distribution": {
            str(key): int(value) for key, value in invalid_counts.items()
        },
        "assessment_singleton_slates": singleton,
        "assessment_multi_candidate_slates": multi,
        "weight_rows_verified": len(weights),
        "primary_rows": len(primary),
    }


def verify_weighted_models(
    compact: pd.DataFrame,
    assessment: pd.DataFrame,
    configurations: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    development = compact.loc[
        compact["year"].eq(2024)
        & compact["parent_slate_eligible"].astype(bool)
        & compact["high_movement_admitted"].astype(bool)
    ].copy()
    models, confirmed = fit_ladder(development, configurations)
    scored = score_ladder(assessment, models, configurations)
    errors: dict[str, float] = {}
    for name, model in models.items():
        means, scales, estimator = model
        frozen = coefficients["models"][name]
        errors[f"{name}_means"] = maximum_error(means, frozen["means"])
        errors[f"{name}_scales"] = maximum_error(scales, frozen["scales"])
        errors[f"{name}_coefficients"] = maximum_error(estimator.coef_[0], frozen["coefficients"])
        errors[f"{name}_intercept"] = abs(
            float(estimator.intercept_[0]) - float(frozen["intercept"])
        )
        probability = f"p_onset__{name}" if name.startswith("A") else f"p_up_given_onset__{name}"
        errors[f"{name}_assessment_predictions"] = maximum_error(
            scored[probability], assessment[probability]
        )
    errors["development_confirmation_direction"] = maximum_error(
        confirmed["predicted_direction_remained_same"],
        compact.loc[development.index, "predicted_direction_remained_same"],
    )
    errors["development_confirmation_retracement"] = maximum_error(
        confirmed["favourable_retracement_bps"],
        compact.loc[development.index, "favourable_retracement_bps"],
    )
    errors["assessment_confirmation_direction"] = maximum_error(
        scored["predicted_direction_remained_same"],
        assessment["predicted_direction_remained_same"],
    )
    errors["assessment_confirmation_retracement"] = maximum_error(
        scored["favourable_retracement_bps"], assessment["favourable_retracement_bps"]
    )
    require(max(errors.values()) <= 1e-12, f"weighted model reconstruction differs: {errors}")
    return scored, {
        "eight_models_refit": True,
        "development_rows": len(development),
        "maximum_model_or_prediction_error": max(errors.values()),
    }


def compare_economic_frame(expected: pd.DataFrame, actual: pd.DataFrame) -> float:
    keys = ["candidate", "horizon", "friction_bps"]
    left = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    require(
        left[["candidate", "horizon"]]
        .astype(str)
        .equals(right[["candidate", "horizon"]].astype(str)),
        "economic categorical keys differ",
    )
    require(
        maximum_error(left["friction_bps"], right["friction_bps"]) == 0.0,
        "economic friction keys differ",
    )
    numeric = [
        "mean_signed_gross_return_bps",
        "mean_signed_return_after_friction_bps",
        "median_signed_return_after_friction_bps",
        "positive_after_friction_rate",
        "mean_signed_cohort_relative_return_bps",
    ]
    error = max(maximum_error(left[name], right[name]) for name in numeric)
    require(error <= 1e-8, "economic metrics differ")
    require(
        left[["selected_rows", "sessions", "stocks"]]
        .astype(int)
        .equals(right[["selected_rows", "sessions", "stocks"]].astype(int)),
        "economic support differs",
    )
    return error


def verify_economic_and_slate_types(
    artifacts: Path, primary: pd.DataFrame, admitted: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selections = V0_AUDIT.economic_selections(primary)
    expected = V0_AUDIT.economic_aggregate(selections)
    error = compare_economic_frame(
        expected, pd.read_csv(artifacts / "economic_reference_metrics.csv")
    )
    counts = admitted.set_index("slate_id")["admitted_stock_count"].astype(int).to_dict()
    selections["admitted_stock_count"] = selections["slate_id"].astype(str).map(counts).astype(int)
    selections["admitted_slate_type"] = np.where(
        selections["admitted_stock_count"].eq(1), "singleton", "multi_candidate"
    )
    selections["within_admitted_comparison_status"] = np.where(
        selections["admitted_stock_count"].eq(1),
        "degenerate_singleton",
        "competitive_multi_candidate",
    )
    errors = [error]
    total_slates = int(admitted["admitted_stock_count"].ge(1).sum())
    for slate_type, filename in (
        ("singleton", "singleton_slate_metrics.csv"),
        ("multi_candidate", "multi_candidate_slate_metrics.csv"),
    ):
        subset = selections.loc[selections["admitted_slate_type"].eq(slate_type)]
        group_expected = V0_AUDIT.economic_aggregate(subset)
        actual = pd.read_csv(artifacts / filename)
        errors.append(compare_economic_frame(group_expected, actual))
        slates = int(subset["slate_id"].nunique())
        require(actual["slates"].astype(int).eq(slates).all(), "slate metric count differs")
        require(
            maximum_error(actual["slate_fraction"], np.repeat(slates / total_slates, len(actual)))
            <= 5e-12,
            "slate metric fraction differs",
        )
    singleton = selections.loc[selections["admitted_slate_type"].eq("singleton")]
    require(
        singleton["within_admitted_comparison_status"].eq("degenerate_singleton").all(),
        "singleton comparison not labelled degenerate",
    )
    return selections, {
        "selections_verified": len(selections),
        "singleton_selections": len(singleton),
        "multi_candidate_selections": int(
            selections["admitted_slate_type"].eq("multi_candidate").sum()
        ),
        "maximum_economic_error": max(errors),
    }


def reconstruct_concentration(primary: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_shares(
        frame: pd.DataFrame,
        *,
        population: str,
        scope_type: str,
        scope_value: str,
        candidate: str = "not_applicable",
        onset_class: str = "not_applicable",
        limit: float | None = None,
    ) -> None:
        counts = frame["symbol"].astype(str).value_counts(sort=False)
        for symbol, count in counts.items():
            share = float(count / len(frame))
            rows.append(
                {
                    "population": population,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "candidate": candidate,
                    "onset_class": onset_class,
                    "symbol": str(symbol),
                    "rows": int(count),
                    "share": share,
                    "absolute_contribution_bps": math.nan,
                    "maximum_allowed_share": math.nan if limit is None else limit,
                    "passes": True if limit is None else share <= limit + 1e-15,
                }
            )

    add_shares(
        primary,
        population="primary_high_movement_rows",
        scope_type="pooled",
        scope_value="all",
        limit=0.10,
    )
    for checkpoint, frame in primary.groupby("decision_ordinal", sort=True):
        add_shares(
            frame,
            population="primary_high_movement_rows",
            scope_type="checkpoint",
            scope_value=str(int(checkpoint)),
        )
    for month, frame in primary.groupby("year_month", sort=True):
        add_shares(
            frame,
            population="primary_high_movement_rows",
            scope_type="month",
            scope_value=str(month),
        )
    for onset_class, frame in primary.groupby("onset_label", sort=True):
        add_shares(
            frame,
            population="outcome_class_rows",
            scope_type="onset_class",
            scope_value=str(onset_class),
            onset_class=str(onset_class),
        )
    for candidate, frame in selections.groupby("candidate", sort=True):
        add_shares(
            frame,
            population="selected_economic_reference_rows",
            scope_type="pooled",
            scope_value="all",
            candidate=str(candidate),
            limit=0.20,
        )
        contribution = (
            frame.assign(_absolute=(frame["signed_gross_return_bps_30m"] - 20.0).abs())
            .groupby("symbol", sort=True)["_absolute"]
            .sum()
        )
        total = float(contribution.sum())
        for symbol, value in contribution.items():
            share = float(value / total) if total > 0.0 else 0.0
            rows.append(
                {
                    "population": "economic_absolute_contribution_after_20bps",
                    "scope_type": "pooled",
                    "scope_value": "all",
                    "candidate": str(candidate),
                    "onset_class": "not_applicable",
                    "symbol": str(symbol),
                    "rows": int(frame["symbol"].astype(str).eq(str(symbol)).sum()),
                    "share": share,
                    "absolute_contribution_bps": float(value),
                    "maximum_allowed_share": 0.20,
                    "passes": share <= 0.20 + 1e-15,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["population", "scope_type", "scope_value", "candidate", "onset_class", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def verify_concentration(
    artifacts: Path,
    primary: pd.DataFrame,
    selections: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    expected = reconstruct_concentration(primary, selections)
    actual = pd.read_csv(artifacts / "concentration_metrics.csv")
    keys = ["population", "scope_type", "scope_value", "candidate", "onset_class", "symbol"]
    actual = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    require(
        expected[keys].astype(str).equals(actual[keys].astype(str)), "concentration keys differ"
    )
    require(
        expected["rows"].astype(int).equals(actual["rows"].astype(int)), "concentration rows differ"
    )
    share_error = maximum_error(expected["share"], actual["share"])
    contribution_error = maximum_error(
        expected["absolute_contribution_bps"], actual["absolute_contribution_bps"]
    )
    limit_error = maximum_error(expected["maximum_allowed_share"], actual["maximum_allowed_share"])
    require(
        share_error <= 5e-12 and contribution_error <= 1e-7 and limit_error == 0.0,
        "concentration values differ",
    )
    require(
        expected["passes"].astype(bool).equals(actual["passes"].astype(bool)),
        "concentration pass flags differ",
    )
    counts = primary["symbol"].astype(str).value_counts(sort=False)
    leader = (
        counts.rename_axis("symbol")
        .reset_index(name="rows")
        .sort_values(["rows", "symbol"], ascending=[False, True], kind="mergesort")
        .iloc[0]
    )
    summary = decision["concentration"]
    require(
        str(summary["largest_admitted_row_stock"]) == str(leader.symbol), "largest stock differs"
    )
    require(
        abs(float(summary["maximum_primary_row_stock_share"]) - float(leader.rows / len(primary)))
        <= 1e-15,
        "maximum admitted-row share differs",
    )
    pooled = expected.loc[
        expected["population"].eq("primary_high_movement_rows")
        & expected["scope_type"].eq("pooled")
    ]
    selected = expected.loc[expected["population"].eq("selected_economic_reference_rows")]
    contribution = expected.loc[
        expected["population"].eq("economic_absolute_contribution_after_20bps")
    ]
    model_candidates = {"readiness", "pressure", "confirmed"}
    largest_selected = selected.loc[
        selected["candidate"].isin(model_candidates)
        & selected["symbol"].astype(str).eq(str(leader.symbol))
    ]
    largest_contribution = contribution.loc[
        contribution["candidate"].isin(model_candidates)
        & contribution["symbol"].astype(str).eq(str(leader.symbol))
    ]
    largest_selected_max = (
        float(largest_selected["share"].max()) if not largest_selected.empty else 0.0
    )
    largest_contribution_max = (
        float(largest_contribution["share"].max()) if not largest_contribution.empty else 0.0
    )
    primary_passes = bool(pooled["passes"].all())
    selected_passes = bool(selected["passes"].all())
    economic_not_dominated = bool(
        largest_selected_max <= 0.20 + 1e-15 and largest_contribution_max <= 0.20 + 1e-15
    )
    require(
        bool(summary["primary_row_concentration_passes"]) == primary_passes,
        "saved primary concentration gate differs",
    )
    require(
        bool(summary["selected_concentration_passes"]) == selected_passes,
        "saved selection concentration gate differs",
    )
    require(
        bool(summary["economic_not_dominated_by_largest_stock"]) == economic_not_dominated,
        "saved economic dominance gate differs",
    )
    return {
        "ledger_rows_verified": len(actual),
        "largest_stock": str(leader.symbol),
        "largest_stock_rows": int(leader.rows),
        "maximum_stock_row_share": float(leader.rows / len(primary)),
        "primary_row_concentration_passes": primary_passes,
        "selected_concentration_passes": selected_passes,
        "largest_stock_model_selection_share": largest_selected_max,
        "largest_stock_model_contribution_share": largest_contribution_max,
        "economic_not_dominated_by_largest_stock": economic_not_dominated,
        "maximum_share_error": share_error,
        "maximum_absolute_contribution_bps_error": contribution_error,
    }


def permute_bundle(frame: pd.DataFrame, features: Sequence[str], *, seed: int) -> pd.DataFrame:
    output = frame.copy()
    source = frame.copy()
    rng = np.random.default_rng(seed)
    for _, index in output.groupby("parent_slate_id", sort=True).groups.items():
        positions = np.asarray(list(index))
        permutation = rng.permutation(len(positions))
        output.loc[positions, list(features)] = source.loc[positions, list(features)].to_numpy(
            copy=True
        )[permutation]
    return output


def verify_null(
    compact: pd.DataFrame,
    assessment: pd.DataFrame,
    configurations: Mapping[str, Any],
    artifact: pd.DataFrame,
    selections: pd.DataFrame,
) -> dict[str, Any]:
    development = (
        compact.loc[
            compact["year"].eq(2024)
            & compact["parent_slate_eligible"].astype(bool)
            & compact["high_movement_admitted"].astype(bool)
        ]
        .copy()
        .reset_index(drop=True)
    )
    primary = (
        assessment.loc[assessment["high_movement_admitted"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )
    pressure = [
        name
        for name in configurations["models"]["A2"]
        if name not in configurations["models"]["A1"]
    ]
    readiness_mean = float(
        selections.loc[
            selections["candidate"].eq("readiness"), "signed_gross_return_bps_30m"
        ].mean()
    )
    expected: dict[str, list[float]] = {
        "A2_minus_A1_brier_improvement": [],
        "D2_minus_D1_brier_improvement": [],
        "pressure_minus_readiness_economic_30m": [],
    }
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    for draw in range(50):
        null_development = permute_bundle(development, pressure, seed=20260721 + draw)
        null_assessment = permute_bundle(primary, pressure, seed=20260721 + draw + 100_000)
        model_a = weighted_fit(
            null_development,
            null_development["directional_onset"],
            configurations["models"]["A2"],
        )
        direction_development = null_development.loc[null_development["directional_onset"].eq(1)]
        model_d = weighted_fit(
            direction_development,
            direction_development["up_given_onset"],
            configurations["models"]["D2"],
        )
        p_a = fitted_predict(model_a, null_assessment, configurations["models"]["A2"])
        p_d = fitted_predict(model_d, null_assessment, configurations["models"]["D2"])
        expected["A2_minus_A1_brier_improvement"].append(
            V0_AUDIT.brier_improvement(
                null_assessment["directional_onset"],
                null_assessment["p_onset__A1"].to_numpy(dtype=float),
                p_a,
            )
        )
        mask = null_assessment["directional_onset"].eq(1)
        expected["D2_minus_D1_brier_improvement"].append(
            V0_AUDIT.brier_improvement(
                null_assessment.loc[mask, "up_given_onset"],
                null_assessment.loc[mask, "p_up_given_onset__D1"].to_numpy(dtype=float),
                p_d[mask.to_numpy()],
            )
        )
        null_assessment["signed_pressure_score__pressure"] = (
            p_a
            * (2.0 * p_d - 1.0)
            * null_assessment["p_large_remaining_move"].to_numpy(dtype=float)
        )
        null_selection = V0_AUDIT.pressure_only_selection(null_assessment)
        expected["pressure_minus_readiness_economic_30m"].append(
            float(null_selection["signed_gross_return_bps_30m"].mean()) - readiness_mean
        )
    errors: list[float] = []
    for metric, values in expected.items():
        draws = artifact.loc[
            artifact["record_type"].eq("draw") & artifact["metric"].eq(metric)
        ].sort_values("draw")
        errors.append(maximum_error(draws["null_value"], values))
        summary = artifact.loc[
            artifact["record_type"].eq("summary") & artifact["metric"].eq(metric)
        ].iloc[0]
        array = np.asarray(values, dtype=float)
        errors.extend(
            [
                abs(float(summary.null_q90) - float(np.quantile(array, 0.90))),
                abs(
                    float(summary.real_percentile)
                    - float(np.mean(array < float(summary.real_value)))
                ),
            ]
        )
        require(
            draws["null_interpretation"].eq("admitted_bundle_within_valid_parent_slate").all(),
            "null interpretation differs",
        )
    require(max(errors) <= 1e-8, "repaired within-parent-slate null differs")
    return {
        "draws_verified": 50,
        "bundle_feature_count": len(pressure),
        "admission_preserved": True,
        "maximum_null_error": max(errors),
    }


def verify_deletion_and_loso(
    artifacts: Path,
    compact: pd.DataFrame,
    assessment: pd.DataFrame,
    configurations: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)].copy()
    counts = primary["symbol"].astype(str).value_counts(sort=False)
    largest = str(
        counts.rename_axis("symbol")
        .reset_index(name="rows")
        .sort_values(["rows", "symbol"], ascending=[False, True], kind="mergesort")
        .iloc[0]["symbol"]
    )
    require(
        largest == str(decision["largest_stock_deletion"]["deleted_symbol"]),
        "deletion stock differs",
    )
    development = compact.loc[
        compact["year"].eq(2024)
        & compact["parent_slate_eligible"].astype(bool)
        & compact["high_movement_admitted"].astype(bool)
        & ~compact["symbol"].astype(str).eq(largest)
    ].copy()
    counts_after = development.groupby("parent_slate_id", sort=True)["symbol"].transform("size")
    development["row_weight"] = 1.0 / counts_after.to_numpy(dtype=float)
    deletion_assessment = assessment.loc[~assessment["symbol"].astype(str).eq(largest)].copy()
    models, _ = fit_ladder(development, configurations)
    scored = score_ladder(deletion_assessment, models, configurations)
    deletion_primary = scored.loc[scored["high_movement_admitted"].astype(bool)].copy()
    actual_increments = increments(deletion_primary)
    saved = decision["largest_stock_deletion"]
    increment_error = max(
        abs(actual_increments[name] - float(saved["deletion_increments"][name]))
        for name in actual_increments
    )
    require(increment_error <= 1e-12, "largest-stock deletion increments differ")
    deletion_artifact = pd.read_csv(artifacts / "largest_stock_deletion_metrics.csv")
    increment_rows = deletion_artifact.loc[deletion_artifact["record_type"].eq("increment")]
    require(
        len(increment_rows) == len(actual_increments)
        and set(increment_rows["comparison"].astype(str)) == set(actual_increments)
        and increment_rows["comparison"].astype(str).is_unique,
        "deletion increment coverage differs",
    )
    for row in increment_rows.itertuples(index=False):
        require(
            abs(float(row.deletion_value) - actual_increments[str(row.comparison)]) <= 1e-12,
            "deletion increment artifact differs",
        )
    checkpoint_rows = deletion_artifact.loc[
        deletion_artifact["record_type"].eq("checkpoint_increment")
    ].copy()
    checkpoint_keys = {
        (str(row.comparison), str(row.metric), str(int(float(row.scope_value))))
        for row in checkpoint_rows.itertuples(index=False)
    }
    expected_checkpoint_keys = {
        (comparison, metric, checkpoint)
        for comparison in ("A2_minus_A1", "D2_minus_D1")
        for metric in ("brier_score", "log_loss")
        for checkpoint in ("6", "12")
    }
    require(
        len(checkpoint_rows) == len(expected_checkpoint_keys)
        and checkpoint_keys == expected_checkpoint_keys,
        "deletion checkpoint coverage differs",
    )
    checkpoint_values: list[float] = []
    for row in checkpoint_rows.itertuples(index=False):
        checkpoint = int(float(row.scope_value))
        candidate, _, baseline = str(row.comparison).partition("_minus_")
        subset = deletion_primary.loc[deletion_primary["decision_ordinal"].eq(checkpoint)]
        target = "directional_onset"
        prefix = "p_onset"
        if candidate.startswith("D"):
            subset = subset.loc[subset["directional_onset"].eq(1)]
            target = "up_given_onset"
            prefix = "p_up_given_onset"
        value = loss_improvement(
            subset,
            target=target,
            baseline=f"{prefix}__{baseline}",
            candidate=f"{prefix}__{candidate}",
            kind="brier" if str(row.metric) == "brier_score" else "log_loss",
        )
        require(abs(value - float(row.deletion_value)) <= 1e-12, "deletion checkpoint differs")
        checkpoint_values.append(value)
    expected_no_adversity = all(value >= -0.001 for value in checkpoint_values)
    require(
        bool(saved["no_materially_adverse_checkpoint"]) == expected_no_adversity,
        "deletion checkpoint decision differs",
    )
    main_increments = increments(primary)
    principal = (
        "A2_minus_A1_brier",
        "A2_minus_A1_log_loss",
        "D2_minus_D1_brier",
        "D2_minus_D1_log_loss",
    )
    same_signed = all(
        np.sign(main_increments[name]) == np.sign(actual_increments[name]) for name in principal
    )
    principal_non_negative = all(actual_increments[name] >= 0.0 for name in principal)
    require(
        bool(saved["same_signed_predictive_conclusions"]) == same_signed,
        "saved deletion sign conclusion differs",
    )
    require(
        bool(saved["principal_increments_non_negative"]) == principal_non_negative,
        "saved deletion principal gate differs",
    )
    loso_expected: list[dict[str, Any]] = []
    for symbol in sorted(primary["symbol"].astype(str).unique()):
        subset = primary.loc[~primary["symbol"].astype(str).eq(symbol)]
        values = increments(subset)
        for comparison, value in values.items():
            loso_expected.append(
                {
                    "deleted_symbol": symbol,
                    "comparison": comparison,
                    "rows": len(subset),
                    "sessions": int(subset["session"].nunique()),
                    "stocks": int(subset["symbol"].nunique()),
                    "full_increment": main_increments[comparison],
                    "leave_one_out_increment": value,
                    "same_sign_as_full": bool(
                        np.sign(main_increments[comparison]) == np.sign(value)
                    ),
                }
            )
    expected_loso = (
        pd.DataFrame(loso_expected)
        .sort_values(["deleted_symbol", "comparison"], kind="mergesort")
        .reset_index(drop=True)
    )
    actual_loso = (
        pd.read_csv(artifacts / "leave_one_stock_out_diagnostics.csv")
        .sort_values(["deleted_symbol", "comparison"], kind="mergesort")
        .reset_index(drop=True)
    )
    require(
        expected_loso[["deleted_symbol", "comparison", "rows", "sessions", "stocks"]].equals(
            actual_loso[["deleted_symbol", "comparison", "rows", "sessions", "stocks"]]
        ),
        "leave-one-stock-out support differs",
    )
    loso_error = max(
        maximum_error(expected_loso[name], actual_loso[name])
        for name in ("full_increment", "leave_one_out_increment")
    )
    require(loso_error <= 1e-12, "leave-one-stock-out increments differ")
    require(
        expected_loso["same_sign_as_full"]
        .astype(bool)
        .equals(actual_loso["same_sign_as_full"].astype(bool)),
        "leave-one-stock-out signs differ",
    )
    return {
        "deleted_symbol": largest,
        "development_rows_after_deletion": len(development),
        "assessment_primary_rows_after_deletion": len(deletion_primary),
        "maximum_deletion_increment_error": increment_error,
        "same_signed_predictive_conclusions": same_signed,
        "principal_increments_non_negative": principal_non_negative,
        "no_materially_adverse_checkpoint": expected_no_adversity,
        "main_increments": main_increments,
        "deletion_increments": actual_increments,
        "leave_one_stock_out_rows_verified": len(actual_loso),
        "maximum_leave_one_out_error": loso_error,
    }


def metric_lookup(frame: pd.DataFrame, model: str, metric: str) -> float:
    row = frame.loc[
        frame["population"].eq("primary_high_movement")
        & frame["scope_type"].eq("pooled")
        & frame["model"].eq(model)
    ]
    require(len(row) == 1, f"decision metric lookup differs: {model}/{metric}")
    return float(row.iloc[0][metric])


def slice_improvements(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    metric: str,
    expected_scopes: set[str],
) -> dict[str, float]:
    primary = frame.loc[frame["population"].eq("primary_high_movement")]
    baseline_frame = primary.loc[primary["model"].eq(baseline)].copy()
    candidate_frame = primary.loc[primary["model"].eq(candidate)].copy()
    for model, model_frame in ((baseline, baseline_frame), (candidate, candidate_frame)):
        scopes = model_frame["scope_value"].astype(str)
        require(scopes.is_unique, f"duplicate decision slice: {model}")
        require(set(scopes) == expected_scopes, f"decision slice coverage differs: {model}")
    baseline_rows = baseline_frame.assign(
        scope_value=baseline_frame["scope_value"].astype(str)
    ).set_index("scope_value")
    candidate_rows = candidate_frame.assign(
        scope_value=candidate_frame["scope_value"].astype(str)
    ).set_index("scope_value")
    return {
        str(value): float(baseline_rows.loc[value, metric] - candidate_rows.loc[value, metric])
        for value in sorted(expected_scopes)
    }


def summary_value(frame: pd.DataFrame, metric: str, column: str) -> float:
    row = frame.loc[frame["record_type"].eq("summary") & frame["metric"].eq(metric)]
    require(len(row) == 1, f"decision summary lookup differs: {metric}/{column}")
    return float(row.iloc[0][column])


def economic_value(frame: pd.DataFrame, candidate: str) -> float:
    row = frame.loc[
        frame["candidate"].eq(candidate)
        & frame["horizon"].eq("primary_30m_close_t_plus_8")
        & frame["friction_bps"].eq(20.0)
    ]
    require(len(row) == 1, f"decision economic lookup differs: {candidate}")
    return float(row.iloc[0]["mean_signed_return_after_friction_bps"])


def require_float_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, float], detail: str
) -> None:
    require(set(actual) == set(expected), f"{detail} keys differ")
    require(
        all(abs(float(actual[key]) - expected[key]) <= 1e-12 for key in expected),
        f"{detail} values differ",
    )


def verify_decision(
    decision: Mapping[str, Any],
    primary: pd.DataFrame,
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    concentration: Mapping[str, Any],
    deletion: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive every decision gate from independently verified scientific artifacts."""

    support = decision["support"]
    labels = primary["onset_label"].value_counts().to_dict()
    require(int(support["rows"]) == len(primary), "decision support rows differ")
    require(int(support["sessions"]) == primary["session"].nunique(), "support sessions differ")
    require(int(support["stocks"]) == primary["symbol"].nunique(), "support stocks differ")
    require(int(support["up_onsets"]) == labels.get("UP_ONSET", 0), "UP support differs")
    require(int(support["down_onsets"]) == labels.get("DOWN_ONSET", 0), "DOWN support differs")
    require(int(support["no_onsets"]) == labels.get("NO_ONSET", 0), "NO support differs")
    aggregate_support = {
        "admitted_rows_at_least_1200": len(primary) >= 1_200,
        "sessions_at_least_100": primary["session"].nunique() >= 100,
        "stocks_at_least_15": primary["symbol"].nunique() >= 15,
        "directional_onset_rows_at_least_250": int(primary["directional_onset"].sum()) >= 250,
        "up_onsets_at_least_100": int(labels.get("UP_ONSET", 0)) >= 100,
        "down_onsets_at_least_100": int(labels.get("DOWN_ONSET", 0)) >= 100,
        "months_at_least_6": primary["year_month"].nunique() >= 6,
        "valid_parent_stocks_at_least_15": int(primary["parent_valid_stock_count"].min()) >= 15,
    }
    require(
        support["aggregate_support_gates"] == aggregate_support,
        "aggregate support gates differ",
    )
    require(
        bool(support["aggregate_support_passes"]) == all(aggregate_support.values()),
        "aggregate support aggregation differs",
    )
    conditional_support = bool(
        int(primary["directional_onset"].sum()) >= 250
        and int(labels.get("UP_ONSET", 0)) >= 100
        and int(labels.get("DOWN_ONSET", 0)) >= 100
    )
    require(
        bool(support["conditional_direction_support_passes"]) == conditional_support,
        "conditional direction support differs",
    )

    computed_increments = {
        "A2_minus_A1_brier": metric_lookup(onset, "A1", "brier_score")
        - metric_lookup(onset, "A2", "brier_score"),
        "A2_minus_A1_log_loss": metric_lookup(onset, "A1", "log_loss")
        - metric_lookup(onset, "A2", "log_loss"),
        "D2_minus_D1_brier": metric_lookup(direction, "D1", "brier_score")
        - metric_lookup(direction, "D2", "brier_score"),
        "D2_minus_D1_log_loss": metric_lookup(direction, "D1", "log_loss")
        - metric_lookup(direction, "D2", "log_loss"),
        "A3_minus_A2_brier": metric_lookup(onset, "A2", "brier_score")
        - metric_lookup(onset, "A3", "brier_score"),
        "A3_minus_A2_log_loss": metric_lookup(onset, "A2", "log_loss")
        - metric_lookup(onset, "A3", "log_loss"),
        "D3_minus_D2_brier": metric_lookup(direction, "D2", "brier_score")
        - metric_lookup(direction, "D3", "brier_score"),
        "D3_minus_D2_log_loss": metric_lookup(direction, "D2", "log_loss")
        - metric_lookup(direction, "D3", "log_loss"),
    }
    require_float_mapping(decision["increments"], computed_increments, "decision increments")

    represented_months = set(primary["year_month"].astype(str).unique())
    exact_checkpoints = {"6", "12"}
    monthly_improvements = {
        "A2_minus_A1": slice_improvements(
            monthly,
            baseline="A1",
            candidate="A2",
            metric="brier_score",
            expected_scopes=represented_months,
        ),
        "D2_minus_D1": slice_improvements(
            monthly,
            baseline="D1",
            candidate="D2",
            metric="brier_score",
            expected_scopes=represented_months,
        ),
        "A3_minus_A2": slice_improvements(
            monthly,
            baseline="A2",
            candidate="A3",
            metric="brier_score",
            expected_scopes=represented_months,
        ),
        "D3_minus_D2": slice_improvements(
            monthly,
            baseline="D2",
            candidate="D3",
            metric="brier_score",
            expected_scopes=represented_months,
        ),
    }
    for comparison, expected in monthly_improvements.items():
        require_float_mapping(
            decision["monthly_brier_improvements"][comparison],
            expected,
            f"monthly decision improvements {comparison}",
        )
    checkpoint_improvements = {
        "A2_minus_A1_brier": slice_improvements(
            checkpoint,
            baseline="A1",
            candidate="A2",
            metric="brier_score",
            expected_scopes=exact_checkpoints,
        ),
        "A2_minus_A1_log_loss": slice_improvements(
            checkpoint,
            baseline="A1",
            candidate="A2",
            metric="log_loss",
            expected_scopes=exact_checkpoints,
        ),
        "D2_minus_D1_brier": slice_improvements(
            checkpoint,
            baseline="D1",
            candidate="D2",
            metric="brier_score",
            expected_scopes=exact_checkpoints,
        ),
        "D2_minus_D1_log_loss": slice_improvements(
            checkpoint,
            baseline="D1",
            candidate="D2",
            metric="log_loss",
            expected_scopes=exact_checkpoints,
        ),
    }
    for comparison, expected in checkpoint_improvements.items():
        require_float_mapping(
            decision["checkpoint_improvements"][comparison],
            expected,
            f"checkpoint decision improvements {comparison}",
        )

    occurrence_gates = {
        "brier_improvement_positive": computed_increments["A2_minus_A1_brier"] > 0.0,
        "log_loss_improvement_positive": computed_increments["A2_minus_A1_log_loss"] > 0.0,
        "bootstrap_90_lower_brier_non_negative": summary_value(
            bootstrap, "A2_minus_A1_brier_improvement", "lower_90"
        )
        >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": summary_value(
            bootstrap, "A2_minus_A1_log_loss_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in monthly_improvements["A2_minus_A1"].values()
        )
        >= 5,
        "real_brier_increment_exceeds_null_q90": computed_increments["A2_minus_A1_brier"]
        > summary_value(nulls, "A2_minus_A1_brier_improvement", "null_q90"),
        "neither_checkpoint_materially_adverse": all(
            value >= -0.001
            for name in ("A2_minus_A1_brier", "A2_minus_A1_log_loss")
            for value in checkpoint_improvements[name].values()
        ),
        "concentration_gates_pass": True,
    }
    direction_gates = {
        "brier_improvement_positive": computed_increments["D2_minus_D1_brier"] > 0.0,
        "log_loss_improvement_positive": computed_increments["D2_minus_D1_log_loss"] > 0.0,
        "auc_not_reduced": metric_lookup(direction, "D2", "auc")
        >= metric_lookup(direction, "D1", "auc"),
        "bootstrap_90_lower_brier_non_negative": summary_value(
            bootstrap, "D2_minus_D1_brier_improvement", "lower_90"
        )
        >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": summary_value(
            bootstrap, "D2_minus_D1_log_loss_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in monthly_improvements["D2_minus_D1"].values()
        )
        >= 5,
        "real_brier_increment_exceeds_null_q90": computed_increments["D2_minus_D1_brier"]
        > summary_value(nulls, "D2_minus_D1_brier_improvement", "null_q90"),
        "neither_checkpoint_materially_adverse": all(
            value >= -0.001
            for name in ("D2_minus_D1_brier", "D2_minus_D1_log_loss")
            for value in checkpoint_improvements[name].values()
        ),
        "concentration_gates_pass": True,
    }
    confirmation_economic_not_worsened = economic_value(economic, "confirmed") >= economic_value(
        economic, "pressure"
    )
    confirmation_occurrence_gates = {
        "brier_improvement_positive": computed_increments["A3_minus_A2_brier"] > 0.0,
        "log_loss_improvement_positive": computed_increments["A3_minus_A2_log_loss"] > 0.0,
        "bootstrap_90_lower_brier_non_negative": summary_value(
            bootstrap, "A3_minus_A2_brier_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in monthly_improvements["A3_minus_A2"].values()
        )
        >= 5,
        "delayed_economic_result_not_worsened": confirmation_economic_not_worsened,
        "concentration_gates_pass": True,
    }
    confirmation_direction_gates = {
        "brier_improvement_positive": computed_increments["D3_minus_D2_brier"] > 0.0,
        "log_loss_improvement_positive": computed_increments["D3_minus_D2_log_loss"] > 0.0,
        "bootstrap_90_lower_brier_non_negative": summary_value(
            bootstrap, "D3_minus_D2_brier_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in monthly_improvements["D3_minus_D2"].values()
        )
        >= 5,
        "delayed_economic_result_not_worsened": confirmation_economic_not_worsened,
        "concentration_gates_pass": True,
    }
    for name, expected in (
        ("occurrence_gates", occurrence_gates),
        ("direction_gates", direction_gates),
        ("confirmation_occurrence_gates", confirmation_occurrence_gates),
        ("confirmation_direction_gates", confirmation_direction_gates),
    ):
        require(decision[name] == expected, f"{name} differs")

    occurrence_passes = all(occurrence_gates.values())
    direction_passes = bool(conditional_support and all(direction_gates.values()))
    confirmation_occurrence_passes = bool(
        not occurrence_passes and all(confirmation_occurrence_gates.values())
    )
    confirmation_direction_passes = bool(
        not direction_passes and conditional_support and all(confirmation_direction_gates.values())
    )
    evidence = {
        "occurrence_passes": occurrence_passes,
        "direction_passes": direction_passes,
        "confirmation_occurrence_passes": confirmation_occurrence_passes,
        "confirmation_direction_passes": confirmation_direction_passes,
        "readiness_useful": bool(
            metric_lookup(onset, "A1", "brier_score") < metric_lookup(onset, "A0", "brier_score")
            and metric_lookup(onset, "A1", "log_loss") < metric_lookup(onset, "A0", "log_loss")
        ),
    }
    require(decision["evidence"] == evidence, "decision evidence differs")
    if evidence["occurrence_passes"] and evidence["direction_passes"]:
        base = "pressure_onset_and_direction_increment_observed"
    elif evidence["occurrence_passes"]:
        base = "pressure_onset_occurrence_only"
    elif evidence["direction_passes"]:
        base = "directional_pressure_only"
    elif evidence["confirmation_occurrence_passes"] or evidence["confirmation_direction_passes"]:
        base = "one_bar_confirmation_required"
    elif evidence["readiness_useful"]:
        base = "movement_readiness_but_direction_unresolved"
    else:
        base = "no_pressure_onset_increment"
    require(decision["predictive_decision_before_concentration"] == base, "base decision differs")
    conditions = {
        "raw_ten_percent_gate_passes": bool(concentration["primary_row_concentration_passes"]),
        "delete_largest_same_signed_conclusions": bool(
            deletion["same_signed_predictive_conclusions"]
        ),
        "delete_largest_principal_increments_non_negative": bool(
            deletion["principal_increments_non_negative"]
        ),
        "delete_largest_no_material_adversity": bool(deletion["no_materially_adverse_checkpoint"]),
        "economic_not_dominated_by_largest_stock": bool(
            concentration["economic_not_dominated_by_largest_stock"]
        ),
    }
    require(
        decision["concentration_decision_conditions"] == conditions,
        "concentration decision conditions differ",
    )
    positive = {
        "pressure_onset_and_direction_increment_observed",
        "pressure_onset_occurrence_only",
        "directional_pressure_only",
        "one_bar_confirmation_required",
    }
    final = base
    if base in positive and not conditions["raw_ten_percent_gate_passes"]:
        stress = all(
            (
                conditions["delete_largest_same_signed_conclusions"],
                conditions["delete_largest_principal_increments_non_negative"],
                conditions["delete_largest_no_material_adversity"],
                conditions["economic_not_dominated_by_largest_stock"],
            )
        )
        if not stress:
            final = "pressure_signal_observed_but_concentration_gate_failed"
    require(decision["decision"] == final, "final concentration-aware decision differs")
    return {
        "predictive_decision": base,
        "final_decision": final,
        "decision_reconstructed_from_verified_artifacts": True,
    }


def run_audit(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    required = (*V0_AUDIT.REQUIRED, *EXTRA_REQUIRED)
    missing = [name for name in required if not (artifacts / name).is_file()]
    require(not missing, f"required artifacts missing: {missing}")
    contract = read_json(artifacts / "contract.json")
    decision = read_json(artifacts / "decision.json")
    repair = read_json(artifacts / "support_contract_repair.json")
    for key, expected in SAFETY_FLAGS.items():
        require(contract.get(key) == expected, f"contract safety differs: {key}")
        require(contract.get("safety", {}).get(key) == expected, f"nested safety differs: {key}")
        require(decision.get(key) == expected, f"decision safety differs: {key}")
        require(repair.get(key) == expected, f"repair safety differs: {key}")
    require(contract.get("scientific_status") == SCIENTIFIC_STATUS, "scientific status differs")
    expected_contract_lineage = {
        str(path.relative_to(V0_DIR)): digest for path, digest in V0_IMMUTABLE_FILE_HASHES.items()
    }
    expected_contract_lineage[f"git:{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}"] = (
        V0_REUSABLE_BLOB_SHA256
    )
    require(
        contract.get("v0_source", {}).get("immutable_sha256") == expected_contract_lineage,
        "contract immutable V0 lineage differs",
    )
    expected_repair = {
        "old_incorrect_interpretation": "minimum candidate count applied after movement admission",
        "correct_interpretation": (
            "minimum valid-stock count applied to parent slate before movement admission"
        ),
        "movement_thresholds_changed": False,
        "onset_barriers_changed": False,
        "features_changed": False,
        "models_changed": False,
        "labels_changed": False,
        "dates_changed": False,
        "rows_added_from_new_market_data": False,
    }
    for key, expected in expected_repair.items():
        require(repair.get(key) == expected, f"support repair declaration differs: {key}")
    rerun = read_json(artifacts / "exact_rerun_manifest.json")
    require(bool(rerun["passed"]), "exact rerun failed")
    require(all(bool(row["passed"]) for row in rerun["comparisons"]), "rerun comparison failed")
    input_hashes = read_json(artifacts / "input_artifact_hashes.json")
    for record in input_hashes["artifacts"]:
        path = REPO_ROOT / str(record["logical_path"])
        require(path.is_file(), f"input artifact missing: {record['logical_path']}")
        require(V0_AUDIT.sha256_file(path) == record["sha256"], "input artifact hash differs")
    protected = read_json(artifacts / "protected_boundary_audit.json")
    require(int(protected["protected_rows_materialised"]) == 0, "protected rows materialised")
    require(protected["protected_files_touched"] == [], "protected files touched")
    source = V0_AUDIT.verify_sources(read_json(artifacts / "source_manifest.json"), provider_root)
    require(
        pd.Timestamp(source["maximum_timestamp_read"]) < PROTECTED_START, "protected source read"
    )
    panel = pd.read_parquet(artifacts / "compact_decision_panel.parquet")
    ledger = pd.read_parquet(artifacts / "onset_path_ledger.parquet")
    oof = pd.read_parquet(artifacts / "development_oof_predictions.parquet")
    assessment = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    require(len(panel) <= 20_000, "compact row limit exceeded")
    forbidden = sorted(
        name
        for name in panel.columns
        if any(fragment in name.lower() for fragment in V0_AUDIT.FORBIDDEN)
    )
    require(not forbidden, f"forbidden compact columns: {forbidden}")
    require(
        pd.to_datetime(panel["decision_available_timestamp"], utc=True).lt(PROTECTED_START).all(),
        "protected compact timestamp",
    )
    windows = V0_AUDIT.verify_windows_and_paths(panel, ledger)
    feature_manifest = read_json(artifacts / "feature_manifest.json")
    features = V0_AUDIT.verify_features(panel, feature_manifest)
    readiness = V0_AUDIT.verify_readiness_from_bounded_bars(panel, provider_root)
    thresholds = V0_AUDIT.verify_thresholds_and_labels(
        panel,
        oof,
        read_json(artifacts / "movement_admission_thresholds.json"),
        read_json(artifacts / "onset_barriers.json"),
        read_json(artifacts / "movement_oof_fold_manifest.json"),
    )
    predecessor = read_json(artifacts / "predecessor_reconstruction.json")
    require(bool(predecessor["passed"]), "predecessor reconstruction failed")
    require(float(predecessor["maximum_prediction_absolute_error"]) <= 1e-12, "M1 differs")
    lineage = verify_v0_lineage(artifacts, panel)
    hierarchy = verify_support_hierarchy(artifacts, panel, assessment, provider_root)
    configurations = read_json(artifacts / "model_configurations.json")
    coefficients = read_json(artifacts / "model_coefficients.json")
    scored, model_result = verify_weighted_models(panel, assessment, configurations, coefficients)
    primary = scored.loc[scored["high_movement_admitted"].astype(bool)].copy()
    onset = pd.read_csv(artifacts / "onset_metrics.csv", dtype={"scope_value": str})
    direction = pd.read_csv(artifacts / "direction_metrics.csv", dtype={"scope_value": str})
    monthly = pd.read_csv(artifacts / "monthly_metrics.csv", dtype={"scope_value": str})
    checkpoint = pd.read_csv(artifacts / "checkpoint_metrics.csv", dtype={"scope_value": str})
    metric_frames = [onset, direction, monthly, checkpoint]
    metrics = V0_AUDIT.verify_metrics(
        assessment,
        metric_frames,
        pd.read_csv(artifacts / "calibration_bins.csv", dtype={"scope_value": str}),
    )
    admitted = pd.read_csv(artifacts / "admitted_slate_accounting.csv")
    admitted = admitted.loc[admitted["year"].eq(2025)].copy()
    selections, economic = verify_economic_and_slate_types(artifacts, primary, admitted)
    concentration = verify_concentration(artifacts, primary, selections, decision)
    bootstrap_frame = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    bootstrap = V0_AUDIT.verify_bootstrap(primary, selections, bootstrap_frame)
    null_frame = pd.read_csv(artifacts / "null_metrics.csv")
    null = verify_null(
        panel,
        assessment,
        configurations,
        null_frame,
        selections,
    )
    deletion = verify_deletion_and_loso(artifacts, panel, assessment, configurations, decision)
    economic_frame = pd.read_csv(artifacts / "economic_reference_metrics.csv")
    decision_result = verify_decision(
        decision,
        primary,
        onset,
        direction,
        monthly,
        checkpoint,
        bootstrap_frame,
        null_frame,
        economic_frame,
        concentration,
        deletion,
    )
    json_artifacts = [
        read_json(path)
        for path in artifacts.glob("*.json")
        if path.name != "independent_audit.json"
    ]
    absolute_strings = [
        value
        for payload in json_artifacts
        for value in V0_AUDIT.recursive_strings(payload)
        if value.startswith("/Users/") or value.startswith(str(REPO_ROOT))
    ]
    require(not absolute_strings, "local absolute path found in JSON artifact")
    return {
        **SAFETY_FLAGS,
        "scientific_status": SCIENTIFIC_STATUS,
        "auditor_imported_experiment_runner": False,
        "auditor_imported_reusable_module": False,
        "passed": True,
        "checks": {
            "safety_and_repair_declaration": True,
            "source_hashes_and_counts": source,
            "protected_boundary": True,
            "absence_of_forbidden_columns": True,
            "fixed_windows_and_cohort_paths": windows,
            "chronology_thresholds_barriers_and_labels": thresholds,
            "readiness_from_bounded_causal_bars": readiness,
            "feature_formulas_and_confirmation": features,
            "immutable_v0_lineage": lineage,
            "parent_and_admitted_support_hierarchy": hierarchy,
            "weighted_models_and_manual_predictions": model_result,
            "probability_metrics_and_calibration": metrics,
            "session_block_bootstrap": bootstrap,
            "within_parent_slate_bundle_permutation": null,
            "economic_reference_and_singleton_semantics": economic,
            "concentration": concentration,
            "largest_stock_deletion_and_leave_one_out": deletion,
            "support_and_decision_logic": decision_result,
            "exact_rerun_manifest": True,
            "local_absolute_paths_absent": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=(
            Path.home()
            / "StockerLocal"
            / "data"
            / "processed"
            / "source=eodhd"
            / "instrument_type=stock"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.artifacts / "independent_audit.json"
    try:
        result = run_audit(args.artifacts, args.provider_root)
        output.write_text(canonical_json(result), encoding="utf-8")
        return 0
    except Exception as exc:
        failure = {
            **SAFETY_FLAGS,
            "scientific_status": SCIENTIFIC_STATUS,
            "auditor_imported_experiment_runner": False,
            "auditor_imported_reusable_module": False,
            "passed": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
        output.write_text(canonical_json(failure), encoding="utf-8")
        print(failure["failure"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
