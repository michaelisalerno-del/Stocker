"""Run High-Movement Pressure-Onset Screen V0.1 support-semantics repair."""

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
from typing import Any, cast

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
from sklearn.exceptions import ConvergenceWarning

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_SRC = REPO_ROOT / "packages" / "stocker_research" / "src"
V0_DIR = EXPERIMENT_DIR.parent / "20260720-high-movement-pressure-onset-screen-v0"
V0_PRIMARY = V0_DIR / "artifacts" / "primary"
V0_EXACT = V0_DIR / "artifacts" / "exact_rerun"
V0_RUNNER_PATH = V0_DIR / "run_screen_v0.py"
V0_AUDITOR_PATH = V0_DIR / "audit_screen_v0.py"
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frozen_v0_reusable_blob() -> bytes:
    """Read and verify the reusable module blob from the declared V0 commit."""

    result = subprocess.run(
        ["git", "cat-file", "blob", f"{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("immutable V0 reusable-module blob is unavailable")
    if hashlib.sha256(result.stdout).hexdigest() != V0_REUSABLE_BLOB_SHA256:
        raise RuntimeError("immutable V0 reusable-module blob hash differs")
    return result.stdout


def verify_immutable_v0_lineage_files() -> bytes:
    """Anchor every imported frozen engine file before importing V0 code."""

    for path, expected in V0_IMMUTABLE_FILE_HASHES.items():
        if not path.is_file() or sha256_path(path) != expected:
            raise RuntimeError(f"immutable V0 lineage hash differs: {path.name}")
    return read_frozen_v0_reusable_blob()


FROZEN_V0_REUSABLE_BLOB = verify_immutable_v0_lineage_files()
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from stocker_research.pressure_onset_screen_v0 import (
    FrozenLogisticModel,
    annotate_economic_selection_semantics,
    apply_support_contract_repair,
    concentration_aware_decision,
    fit_fixed_logistic,
    largest_admitted_stock,
    permute_feature_bundle_within_slates,
)

DEFAULT_PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0_1.py"
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"

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
SCIENTIFIC_STATUS = "opened_support_contract_repair_retrospective_feasibility_evidence"
MODEL_DEPENDENT_CONFIRMATION = (
    "favourable_retracement_bps",
    "predicted_direction_remained_same",
)
EXTRA_SCIENTIFIC_ARTIFACTS = (
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


def load_v0_runner() -> ModuleType:
    """Load the immutable V0 implementation as the frozen scientific engine."""

    reusable_name = "stocker_research.pressure_onset_screen_v0"
    current_reusable = sys.modules[reusable_name]
    frozen_reusable = ModuleType(reusable_name)
    frozen_reusable.__file__ = f"git:{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}"
    frozen_reusable.__package__ = "stocker_research"
    name = "stocker_pressure_onset_v0_frozen_runner"
    specification = importlib.util.spec_from_file_location(name, V0_RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("V0 runner could not be loaded")
    module = importlib.util.module_from_spec(specification)
    try:
        sys.modules[reusable_name] = frozen_reusable
        exec(
            compile(
                FROZEN_V0_REUSABLE_BLOB.decode("utf-8"),
                frozen_reusable.__file__,
                "exec",
            ),
            frozen_reusable.__dict__,
        )
        sys.modules[name] = module
        specification.loader.exec_module(module)
    finally:
        sys.modules[reusable_name] = current_reusable
    return module


V0 = load_v0_runner()


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise V0.ScreenBlocker(
                "blocked_chronology_or_leakage_failure", f"V0.1 contract safety differs: {key}"
            )
    if contract.get("scientific_status") != SCIENTIFIC_STATUS:
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "V0.1 scientific status differs"
        )
    return contract


def validate_v0_artifact_roots() -> dict[str, str]:
    """Validate every V0 artifact against its hard-anchored exact-rerun manifest."""

    manifest = json.loads((V0_PRIMARY / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    if not bool(manifest.get("passed")):
        raise RuntimeError("immutable V0 exact-rerun manifest did not pass")
    expected: dict[str, str] = {}
    for comparison in manifest["comparisons"]:
        artifact = str(comparison["artifact"])
        primary_hash = str(comparison["primary_sha256"])
        exact_hash = str(comparison["exact_rerun_sha256"])
        if not bool(comparison["passed"]) or primary_hash != exact_hash:
            raise RuntimeError(f"immutable V0 comparison differs: {artifact}")
        expected[artifact] = primary_hash
    expected["independent_audit.json"] = str(manifest["independent_audit_sha256"])
    expected["exact_rerun_manifest.json"] = V0_EXACT_MANIFEST_SHA256
    for root in (V0_PRIMARY, V0_EXACT):
        actual_names = {path.name for path in root.iterdir() if path.is_file()}
        if actual_names != set(expected):
            raise RuntimeError(f"immutable V0 artifact set differs: {root.name}")
        for artifact, expected_hash in expected.items():
            if sha256_path(root / artifact) != expected_hash:
                raise RuntimeError(f"immutable V0 artifact hash differs: {artifact}")
    return expected


def frozen_v0_hashes() -> dict[str, Any]:
    """Hash both immutable V0 artifact roots before fitting any V0.1 model."""

    verify_immutable_v0_lineage_files()
    validate_v0_artifact_roots()
    roots: dict[str, list[dict[str, Any]]] = {}
    for label, root in (("primary", V0_PRIMARY), ("exact_rerun", V0_EXACT)):
        records = []
        for path in sorted(item for item in root.iterdir() if item.is_file()):
            records.append(
                {
                    "artifact": path.name,
                    "sha256": V0.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        roots[label] = records
    return {
        **SAFETY_FLAGS,
        "v0_commit": V0_COMMIT,
        "v0_experiment": str(V0_DIR.relative_to(REPO_ROOT)),
        "immutable_lineage_hashes": [
            {
                "logical_path": str(path.relative_to(REPO_ROOT)),
                "sha256": expected,
            }
            for path, expected in V0_IMMUTABLE_FILE_HASHES.items()
        ]
        + [
            {
                "logical_path": f"git:{V0_COMMIT}:{V0_REUSABLE_LOGICAL_PATH}",
                "sha256": V0_REUSABLE_BLOB_SHA256,
            }
        ],
        "artifact_roots": roots,
    }


def compare_frozen_population(
    compact: pd.DataFrame,
    *,
    thresholds: Mapping[int, float],
    barriers: Mapping[str, float],
    source_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every V0 row and pre-fit value is byte-for-value unchanged."""

    v0_panel = pd.read_parquet(V0_PRIMARY / "compact_decision_panel.parquet")
    keys = ["symbol", "session", "decision_ordinal"]
    left = v0_panel.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = compact.sort_values(keys, kind="mergesort").reset_index(drop=True)
    frozen_columns = [column for column in left.columns if column != "screen_status"]
    missing = sorted(set(frozen_columns).difference(right.columns))
    if missing:
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"V0.1 frozen columns missing: {missing}"
        )
    try:
        pd.testing.assert_frame_equal(
            left.loc[:, frozen_columns],
            right.loc[:, frozen_columns],
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"V0 frozen panel differs: {exc}"
        ) from exc
    v0_thresholds = json.loads(
        (V0_PRIMARY / "movement_admission_thresholds.json").read_text(encoding="utf-8")
    )["thresholds"]
    v0_barriers = json.loads((V0_PRIMARY / "onset_barriers.json").read_text(encoding="utf-8"))[
        "barriers_bps"
    ]
    threshold_errors = {
        str(ordinal): abs(float(thresholds[ordinal]) - float(v0_thresholds[str(ordinal)]))
        for ordinal in (6, 12)
    }
    barrier_errors = {
        str(ordinal): abs(float(barriers[str(ordinal)]) - float(v0_barriers[str(ordinal)]))
        for ordinal in (6, 12)
    }
    if max(threshold_errors.values()) > 0.0 or max(barrier_errors.values()) > 0.0:
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "threshold or onset barrier changed"
        )
    v0_source = json.loads((V0_PRIMARY / "source_manifest.json").read_text(encoding="utf-8"))
    v0_source_hashes = {
        str(record["symbol"]): str(record["bounded_safe_hash"]) for record in v0_source["sources"]
    }
    current_source_hashes = {
        str(record["symbol"]): str(record["bounded_safe_hash"])
        for record in source_context["sources"]
    }
    if v0_source_hashes != current_source_hashes:
        raise V0.ScreenBlocker("blocked_chronology_or_leakage_failure", "V0 source hashes changed")
    assessment = right.loc[right["year"].eq(2025)]
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)]
    labels = primary["onset_label"].value_counts().to_dict()
    return {
        **SAFETY_FLAGS,
        "passed": True,
        "comparison_mode": "exact_frame_identity_before_model_dependent_confirmation",
        "v0_compact_panel_sha256": V0.sha256_file(V0_PRIMARY / "compact_decision_panel.parquet"),
        "frozen_columns_compared": len(frozen_columns),
        "frozen_rows_compared": len(right),
        "development_rows": int(right["year"].eq(2024).sum()),
        "assessment_rows": len(assessment),
        "primary_rows": len(primary),
        "primary_sessions": int(primary["session"].nunique()),
        "primary_stocks": int(primary["symbol"].nunique()),
        "up_onsets": int(labels.get("UP_ONSET", 0)),
        "down_onsets": int(labels.get("DOWN_ONSET", 0)),
        "no_onsets": int(labels.get("NO_ONSET", 0)),
        "movement_probability_maximum_error": 0.0,
        "admission_flag_differences": 0,
        "onset_label_differences": 0,
        "feature_value_differences": 0,
        "timestamp_differences": 0,
        "qa_exclusion_differences": 0,
        "threshold_errors": threshold_errors,
        "barrier_errors": barrier_errors,
        "static_confirmation_feature_values_changed": False,
        "confirmation_feature_definitions_changed": False,
        "model_dependent_confirmation_previously_unavailable": sorted(MODEL_DEPENDENT_CONFIRMATION),
        "rows_added_from_new_market_data": False,
    }


def reconstruct_parent_eligibility(
    predecessor: pd.DataFrame, *, provider_root: Path
) -> pd.DataFrame:
    """Reconstruct pre-admission valid-stock counts, including failed parent slates."""

    columns = ["symbol", "session", "year", "year_month", "decision_ordinal", "slate_id"]
    available_parts: list[pd.DataFrame] = []
    for symbol in V0.SYMBOLS:
        raw = V0.bounded_source(V0.provider_path(provider_root, symbol))
        bars, _ = V0.prepare_symbol_bars(raw, symbol=symbol)
        valid_sessions = set(bars["session"].astype(str))
        requested = predecessor.loc[
            predecessor["symbol"].eq(symbol)
            & predecessor["session"].astype(str).isin(valid_sessions),
            columns,
        ].copy()
        available_parts.append(requested)
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


def attach_support_hierarchy(
    compact: pd.DataFrame, parent_eligibility: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach repaired metadata and account for unavailable parent slates."""

    repaired = apply_support_contract_repair(compact)
    annotated = repaired.annotated_rows.copy()
    if (
        not annotated["source_slate_size"]
        .astype(int)
        .equals(annotated["parent_valid_stock_count"].astype(int))
    ):
        raise V0.ScreenBlocker(
            "blocked_parent_slate_support_failure", "parent valid-stock counts differ"
        )
    raw_map = parent_eligibility.set_index("slate_id")["raw_source_stock_count"]
    history_map = parent_eligibility.set_index("slate_id")["history_complete_stock_count"]
    annotated["parent_raw_source_stock_count"] = annotated["slate_id"].map(raw_map).astype(int)
    annotated["parent_history_complete_stock_count"] = (
        annotated["slate_id"].map(history_map).astype(int)
    )
    compact_parent_counts = annotated.groupby("slate_id", sort=True)["symbol"].nunique()
    compact_admitted_counts = annotated.groupby("slate_id", sort=True)[
        "high_movement_admitted"
    ].sum()
    parent = parent_eligibility.copy()
    parent["parent_valid_stock_count"] = parent["history_complete_stock_count"]
    present = parent["slate_id"].isin(compact_parent_counts.index)
    parent.loc[present, "parent_valid_stock_count"] = (
        parent.loc[present, "slate_id"].map(compact_parent_counts).astype(int)
    )
    parent["parent_valid_stock_count"] = parent["parent_valid_stock_count"].astype(int)
    missing = ~present
    if parent.loc[missing, "parent_valid_stock_count"].ge(15).any():
        raise V0.ScreenBlocker(
            "blocked_parent_slate_support_failure",
            "a reconstructed eligible parent slate is absent from the frozen panel",
        )
    parent["parent_slate_eligible"] = parent["parent_valid_stock_count"].ge(15)
    parent["admitted_stock_count"] = (
        parent["slate_id"].map(compact_admitted_counts).fillna(0).astype(int)
    )
    parent["support_status"] = np.select(
        [
            ~parent["parent_slate_eligible"],
            parent["admitted_stock_count"].eq(0),
            parent["admitted_stock_count"].eq(1),
        ],
        [
            "parent_slate_insufficient_valid_stocks",
            "no_high_movement_admission",
            "valid_singleton_admission",
        ],
        default="valid_multi_candidate_admission",
    )
    admitted = parent.copy()
    admitted["primary_row_count"] = np.where(
        admitted["parent_slate_eligible"], admitted["admitted_stock_count"], 0
    ).astype(int)
    admitted["singleton_admitted_slate"] = admitted["parent_slate_eligible"] & admitted[
        "admitted_stock_count"
    ].eq(1)
    admitted["multi_candidate_admitted_slate"] = admitted["parent_slate_eligible"] & admitted[
        "admitted_stock_count"
    ].ge(2)
    parent = parent.drop(columns=["admitted_stock_count"])
    parent = parent.sort_values("slate_id", kind="mergesort").reset_index(drop=True)
    admitted = admitted.sort_values("slate_id", kind="mergesort").reset_index(drop=True)
    primary = annotated.loc[annotated["primary_eligible"]].copy()
    return annotated, parent, admitted, primary


def repaired_support_summary(primary: pd.DataFrame) -> dict[str, Any]:
    """Apply aggregate gates without using admitted-count or concentration as a stop."""

    labels = primary["onset_label"].value_counts().to_dict()
    gates = {
        "admitted_rows_at_least_1200": bool(len(primary) >= 1_200),
        "sessions_at_least_100": bool(primary["session"].nunique() >= 100),
        "stocks_at_least_15": bool(primary["symbol"].nunique() >= 15),
        "directional_onset_rows_at_least_250": bool(int(primary["directional_onset"].sum()) >= 250),
        "up_onsets_at_least_100": bool(int(labels.get("UP_ONSET", 0)) >= 100),
        "down_onsets_at_least_100": bool(int(labels.get("DOWN_ONSET", 0)) >= 100),
        "months_at_least_6": bool(primary["year_month"].nunique() >= 6),
        "valid_parent_stocks_at_least_15": bool(primary["parent_valid_stock_count"].min() >= 15),
    }
    return {
        "rows": len(primary),
        "sessions": int(primary["session"].nunique()),
        "stocks": int(primary["symbol"].nunique()),
        "months": int(primary["year_month"].nunique()),
        "slates": int(primary["parent_slate_id"].nunique()),
        "directional_onset_rows": int(primary["directional_onset"].sum()),
        "up_onsets": int(labels.get("UP_ONSET", 0)),
        "down_onsets": int(labels.get("DOWN_ONSET", 0)),
        "no_onsets": int(labels.get("NO_ONSET", 0)),
        "minimum_valid_parent_stocks": int(primary["parent_valid_stock_count"].min()),
        "minimum_admitted_stocks": int(primary["admitted_stock_count"].min()),
        "maximum_stock_row_share": float(primary["symbol"].value_counts(normalize=True).max()),
        "aggregate_support_gates": gates,
        "aggregate_support_passes": bool(all(gates.values())),
        "conditional_direction_support_passes": bool(
            int(primary["directional_onset"].sum()) >= 250
            and int(labels.get("UP_ONSET", 0)) >= 100
            and int(labels.get("DOWN_ONSET", 0)) >= 100
        ),
        "concentration_does_not_block_fitting": True,
        "admitted_candidate_minimum_does_not_apply": True,
    }


def fit_repaired_ladder(
    development: pd.DataFrame,
) -> tuple[dict[str, FrozenLogisticModel], pd.DataFrame, list[dict[str, Any]]]:
    """Fit the frozen eight-model ladder with precomputed admitted-slate weights."""

    models: dict[str, FrozenLogisticModel] = {}
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    try:
        for name in ("A0", "A1", "A2"):
            models[name] = fit_fixed_logistic(
                development,
                development["directional_onset"],
                features=V0.MODEL_FEATURES[name],
                slate_column="parent_slate_id",
                model_id=name,
                sample_weight_column="row_weight",
            )
        direction = development.loc[development["directional_onset"].eq(1)].copy()
        for name in ("D0", "D1", "D2"):
            models[name] = fit_fixed_logistic(
                direction,
                direction["up_given_onset"],
                features=V0.MODEL_FEATURES[name],
                slate_column="parent_slate_id",
                model_id=name,
                sample_weight_column="row_weight",
            )
        confirmed = V0._attach_confirmation_from_probabilities(
            development,
            models["D2"].predict(development),
            models["D2"].predict(V0.t1_scoring_frame(development)),
        )
        models["A3"] = fit_fixed_logistic(
            confirmed,
            confirmed["directional_onset"],
            features=V0.MODEL_FEATURES["A3"],
            slate_column="parent_slate_id",
            model_id="A3",
            sample_weight_column="row_weight",
        )
        confirmed_direction = confirmed.loc[confirmed["directional_onset"].eq(1)].copy()
        models["D3"] = fit_fixed_logistic(
            confirmed_direction,
            confirmed_direction["up_given_onset"],
            features=V0.MODEL_FEATURES["D3"],
            slate_column="parent_slate_id",
            model_id="D3",
            sample_weight_column="row_weight",
        )
    except ConvergenceWarning as exc:
        raise V0.ScreenBlocker(
            "blocked_model_convergence_failure", f"model convergence warning: {exc}"
        ) from exc
    except RuntimeError as exc:
        raise V0.ScreenBlocker("blocked_model_convergence_failure", str(exc)) from exc
    if len(models) != 8 or not all(model.converged for model in models.values()):
        raise V0.ScreenBlocker(
            "blocked_model_convergence_failure", "eight converged models were not produced"
        )
    manifest = [
        {
            "feature": "predicted_direction_remained_same",
            "source_model": "D2",
            "source_model_fit": "single_fixed_development_fit",
            "additional_model_specifications": 0,
            "available_at": "completed_t_plus_1_before_open_t_plus_2",
            "support_repair_weight": "1/admitted_stock_count",
        }
    ]
    return models, confirmed, manifest


def repaired_null_metrics(
    development: pd.DataFrame,
    assessment_primary: pd.DataFrame,
    models: Mapping[str, FrozenLogisticModel],
    real_selections: pd.DataFrame,
) -> pd.DataFrame:
    """Run the frozen 50-draw null while preserving admission and repaired weights."""

    real_direction = assessment_primary.loc[assessment_primary["directional_onset"].eq(1)]
    real_economic = (
        real_selections.loc[
            real_selections["candidate"].eq("pressure"), "signed_gross_return_bps_30m"
        ].mean()
        - real_selections.loc[
            real_selections["candidate"].eq("readiness"), "signed_gross_return_bps_30m"
        ].mean()
    )
    real_values = {
        "A2_minus_A1_brier_improvement": V0._brier_improvement(
            assessment_primary["directional_onset"],
            assessment_primary["p_onset__A1"].to_numpy(dtype=float),
            assessment_primary["p_onset__A2"].to_numpy(dtype=float),
        ),
        "D2_minus_D1_brier_improvement": V0._brier_improvement(
            real_direction["up_given_onset"],
            real_direction["p_up_given_onset__D1"].to_numpy(dtype=float),
            real_direction["p_up_given_onset__D2"].to_numpy(dtype=float),
        ),
        "pressure_minus_readiness_economic_30m": float(real_economic),
    }
    values: dict[str, list[float]] = {name: [] for name in real_values}
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    for draw in range(V0.NULL_DRAWS):
        null_development = permute_feature_bundle_within_slates(
            development.reset_index(drop=True),
            V0.PRESSURE_FEATURES,
            seed=V0.NULL_SEED + draw,
            slate_column="parent_slate_id",
        )
        null_assessment = permute_feature_bundle_within_slates(
            assessment_primary.reset_index(drop=True),
            V0.PRESSURE_FEATURES,
            seed=V0.NULL_SEED + draw + 100_000,
            slate_column="parent_slate_id",
        )
        null_a2 = fit_fixed_logistic(
            null_development,
            null_development["directional_onset"],
            features=V0.MODEL_FEATURES["A2"],
            slate_column="parent_slate_id",
            model_id=f"null_A2_{draw}",
            sample_weight_column="row_weight",
        )
        null_direction_development = null_development.loc[
            null_development["directional_onset"].eq(1)
        ]
        null_d2 = fit_fixed_logistic(
            null_direction_development,
            null_direction_development["up_given_onset"],
            features=V0.MODEL_FEATURES["D2"],
            slate_column="parent_slate_id",
            model_id=f"null_D2_{draw}",
            sample_weight_column="row_weight",
        )
        p_a2 = null_a2.predict(null_assessment)
        p_d2 = null_d2.predict(null_assessment)
        null_a = V0._brier_improvement(
            null_assessment["directional_onset"], models["A1"].predict(null_assessment), p_a2
        )
        direction_mask = null_assessment["directional_onset"].eq(1)
        null_d = V0._brier_improvement(
            null_assessment.loc[direction_mask, "up_given_onset"],
            models["D1"].predict(null_assessment.loc[direction_mask]),
            p_d2[direction_mask.to_numpy()],
        )
        null_assessment["p_onset__A2"] = p_a2
        null_assessment["p_up_given_onset__D2"] = p_d2
        null_assessment["signed_pressure_score__pressure"] = (
            p_a2
            * (2.0 * p_d2 - 1.0)
            * null_assessment["p_large_remaining_move"].to_numpy(dtype=float)
        )
        null_selection = V0.economic_selections(null_assessment, candidates=("pressure",))
        null_economic = float(
            null_selection["signed_gross_return_bps_30m"].mean()
            - real_selections.loc[
                real_selections["candidate"].eq("readiness"),
                "signed_gross_return_bps_30m",
            ].mean()
        )
        draw_values = {
            "A2_minus_A1_brier_improvement": null_a,
            "D2_minus_D1_brier_improvement": null_d,
            "pressure_minus_readiness_economic_30m": null_economic,
        }
        for metric, value in draw_values.items():
            values[metric].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "null_value": value,
                    "real_value": real_values[metric],
                    "null_q90": math.nan,
                    "real_percentile": math.nan,
                    "null_interpretation": "admitted_bundle_within_valid_parent_slate",
                }
            )
    for metric, draw_values in values.items():
        array = np.asarray(draw_values, dtype=float)
        real = real_values[metric]
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "null_value": float(array.mean()),
                "real_value": real,
                "null_q90": float(np.quantile(array, 0.90)),
                "real_percentile": float(np.mean(array < real)),
                "null_interpretation": "admitted_bundle_within_valid_parent_slate",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["record_type", "metric", "draw"], kind="mergesort")
        .reset_index(drop=True)
    )


def _loss_improvement(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    candidate: str,
    kind: str,
) -> float:
    labels = frame[target].to_numpy(dtype=float)
    base = frame[baseline].to_numpy(dtype=float)
    contender = frame[candidate].to_numpy(dtype=float)
    if kind == "brier":
        return float(np.mean((labels - base) ** 2) - np.mean((labels - contender) ** 2))
    clipped_base = np.clip(base, 1e-15, 1.0 - 1e-15)
    clipped_contender = np.clip(contender, 1e-15, 1.0 - 1e-15)
    base_loss = -np.mean(
        labels * np.log(clipped_base) + (1.0 - labels) * np.log(1.0 - clipped_base)
    )
    contender_loss = -np.mean(
        labels * np.log(clipped_contender) + (1.0 - labels) * np.log(1.0 - clipped_contender)
    )
    return float(base_loss - contender_loss)


def primary_increments(scored_primary: pd.DataFrame) -> dict[str, float]:
    """Calculate all eight frozen primary loss increments directly from predictions."""

    direction = scored_primary.loc[scored_primary["directional_onset"].eq(1)]
    return {
        "A2_minus_A1_brier": _loss_improvement(
            scored_primary,
            target="directional_onset",
            baseline="p_onset__A1",
            candidate="p_onset__A2",
            kind="brier",
        ),
        "A2_minus_A1_log_loss": _loss_improvement(
            scored_primary,
            target="directional_onset",
            baseline="p_onset__A1",
            candidate="p_onset__A2",
            kind="log_loss",
        ),
        "D2_minus_D1_brier": _loss_improvement(
            direction,
            target="up_given_onset",
            baseline="p_up_given_onset__D1",
            candidate="p_up_given_onset__D2",
            kind="brier",
        ),
        "D2_minus_D1_log_loss": _loss_improvement(
            direction,
            target="up_given_onset",
            baseline="p_up_given_onset__D1",
            candidate="p_up_given_onset__D2",
            kind="log_loss",
        ),
        "A3_minus_A2_brier": _loss_improvement(
            scored_primary,
            target="directional_onset",
            baseline="p_onset__A2",
            candidate="p_onset__A3",
            kind="brier",
        ),
        "A3_minus_A2_log_loss": _loss_improvement(
            scored_primary,
            target="directional_onset",
            baseline="p_onset__A2",
            candidate="p_onset__A3",
            kind="log_loss",
        ),
        "D3_minus_D2_brier": _loss_improvement(
            direction,
            target="up_given_onset",
            baseline="p_up_given_onset__D2",
            candidate="p_up_given_onset__D3",
            kind="brier",
        ),
        "D3_minus_D2_log_loss": _loss_improvement(
            direction,
            target="up_given_onset",
            baseline="p_up_given_onset__D2",
            candidate="p_up_given_onset__D3",
            kind="log_loss",
        ),
    }


def economic_slate_metrics(
    selections: pd.DataFrame, admitted_accounting: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate economic references separately for singleton and multi slates."""

    counts = admitted_accounting.set_index("slate_id")["admitted_stock_count"].astype(int).to_dict()
    annotated = annotate_economic_selection_semantics(selections, counts)
    eligible = admitted_accounting.loc[admitted_accounting["admitted_stock_count"].ge(1)]
    singleton_slates = int(eligible["admitted_stock_count"].eq(1).sum())
    multi_slates = int(eligible["admitted_stock_count"].ge(2).sum())
    total = singleton_slates + multi_slates
    singleton = V0.economic_metrics(annotated.loc[annotated["admitted_slate_type"].eq("singleton")])
    singleton.insert(0, "admitted_slate_type", "singleton")
    singleton["slates"] = singleton_slates
    singleton["slate_fraction"] = singleton_slates / total
    multi = V0.economic_metrics(
        annotated.loc[annotated["admitted_slate_type"].eq("multi_candidate")]
    )
    multi.insert(0, "admitted_slate_type", "multi_candidate")
    multi["slates"] = multi_slates
    multi["slate_fraction"] = multi_slates / total
    return (
        annotated,
        singleton,
        {
            "multi_metrics": multi,
            "singleton_slates": singleton_slates,
            "multi_candidate_slates": multi_slates,
            "admitted_slates": total,
            "singleton_fraction": singleton_slates / total,
            "multi_candidate_fraction": multi_slates / total,
        },
    )


def extended_concentration_metrics(
    primary: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    largest_symbol: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Report row, slice, class, selection, and economic-contribution concentration."""

    columns = [
        "population",
        "scope_type",
        "scope_value",
        "candidate",
        "onset_class",
        "symbol",
        "rows",
        "share",
        "absolute_contribution_bps",
        "maximum_allowed_share",
        "passes",
    ]
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
            frame.assign(_absolute_contribution=(frame["signed_gross_return_bps_30m"] - 20.0).abs())
            .groupby("symbol", sort=True)["_absolute_contribution"]
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
    ledger = pd.DataFrame(rows, columns=columns).sort_values(
        ["population", "scope_type", "scope_value", "candidate", "onset_class", "symbol"],
        kind="mergesort",
    )
    overall = ledger.loc[
        ledger["population"].eq("primary_high_movement_rows") & ledger["scope_type"].eq("pooled")
    ]
    checkpoint_max = (
        ledger.loc[ledger["scope_type"].eq("checkpoint")]
        .groupby("scope_value", sort=True)["share"]
        .max()
        .to_dict()
    )
    month_max = (
        ledger.loc[ledger["scope_type"].eq("month")]
        .groupby("scope_value", sort=True)["share"]
        .max()
        .to_dict()
    )
    selected = ledger.loc[ledger["population"].eq("selected_economic_reference_rows")]
    contribution = ledger.loc[ledger["population"].eq("economic_absolute_contribution_after_20bps")]
    model_candidates = {"readiness", "pressure", "confirmed"}
    largest_selected = selected.loc[
        selected["candidate"].isin(model_candidates) & selected["symbol"].eq(largest_symbol)
    ]
    largest_contribution = contribution.loc[
        contribution["candidate"].isin(model_candidates) & contribution["symbol"].eq(largest_symbol)
    ]
    largest_selected_max = (
        float(largest_selected["share"].max()) if not largest_selected.empty else 0.0
    )
    largest_contribution_max = (
        float(largest_contribution["share"].max()) if not largest_contribution.empty else 0.0
    )
    summary = {
        "largest_admitted_row_stock": largest_symbol,
        "maximum_primary_row_stock_share": float(overall["share"].max()),
        "primary_row_concentration_passes": bool(overall["passes"].all()),
        "maximum_stock_share_by_checkpoint": {
            str(key): float(value) for key, value in checkpoint_max.items()
        },
        "maximum_stock_share_by_month": {
            str(key): float(value) for key, value in month_max.items()
        },
        "maximum_selected_stock_share": float(selected["share"].max()),
        "maximum_economic_contribution_share": float(contribution["share"].max()),
        "largest_stock_model_selection_share": largest_selected_max,
        "largest_stock_model_contribution_share": largest_contribution_max,
        "selected_concentration_passes": bool(selected["passes"].all()),
        "economic_not_dominated_by_largest_stock": bool(
            largest_selected_max <= 0.20 + 1e-15 and largest_contribution_max <= 0.20 + 1e-15
        ),
    }
    summary["all_concentration_gates_pass"] = bool(
        summary["primary_row_concentration_passes"] and summary["selected_concentration_passes"]
    )
    return ledger.reset_index(drop=True), summary


def _reweight_after_deletion(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    counts = output.groupby("parent_slate_id", sort=True)["symbol"].transform("size").astype(int)
    output["admitted_stock_count"] = counts
    output["row_weight"] = 1.0 / counts.to_numpy(dtype=float)
    totals = output.groupby("parent_slate_id", sort=True)["row_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise AssertionError("deletion-refit slate weights differ")
    return output


def largest_stock_deletion_diagnostic(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    main_primary: pd.DataFrame,
    *,
    deleted_symbol: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit downstream models after the concentration-selected stock deletion."""

    deletion_development = _reweight_after_deletion(
        development.loc[~development["symbol"].astype(str).eq(deleted_symbol)].copy()
    )
    deletion_assessment = assessment.loc[
        ~assessment["symbol"].astype(str).eq(deleted_symbol)
    ].copy()
    models, _, _ = fit_repaired_ladder(deletion_development)
    scored = V0.score_model_ladder(deletion_assessment, models)
    primary = scored.loc[scored["high_movement_admitted"].astype(bool)].copy()
    deletion_increments = primary_increments(primary)
    main_increments = primary_increments(main_primary)
    onset, direction, _, checkpoint, _ = V0.evaluate_model_ladder(scored)
    selections = V0.economic_selections(primary)
    economic = V0.economic_metrics(selections)
    rows: list[dict[str, Any]] = []
    for metric_frame in (onset, direction):
        for record in metric_frame.loc[
            metric_frame["population"].eq("primary_high_movement")
        ].itertuples(index=False):
            for metric in ("brier_score", "log_loss", "auc"):
                rows.append(
                    {
                        "record_type": "model_metric",
                        "deleted_symbol": deleted_symbol,
                        "model": str(record.model),
                        "comparison": "not_applicable",
                        "metric": metric,
                        "scope_type": "pooled",
                        "scope_value": "all",
                        "main_value": math.nan,
                        "deletion_value": float(getattr(record, metric)),
                        "same_sign": True,
                        "passes": True,
                    }
                )
    for name, value in deletion_increments.items():
        main = main_increments[name]
        rows.append(
            {
                "record_type": "increment",
                "deleted_symbol": deleted_symbol,
                "model": "not_applicable",
                "comparison": name,
                "metric": name.rsplit("_", 1)[-1],
                "scope_type": "pooled",
                "scope_value": "all",
                "main_value": main,
                "deletion_value": value,
                "same_sign": bool(np.sign(main) == np.sign(value)),
                "passes": bool(np.sign(main) == np.sign(value)),
            }
        )
    checkpoint_improvements: list[float] = []
    for baseline, candidate in (("A1", "A2"), ("D1", "D2")):
        for metric in ("brier_score", "log_loss"):
            values = V0._slice_improvements(
                checkpoint, baseline=baseline, candidate=candidate, metric=metric
            )
            for checkpoint_name, value in values.items():
                checkpoint_improvements.append(value)
                rows.append(
                    {
                        "record_type": "checkpoint_increment",
                        "deleted_symbol": deleted_symbol,
                        "model": "not_applicable",
                        "comparison": f"{candidate}_minus_{baseline}",
                        "metric": metric,
                        "scope_type": "checkpoint",
                        "scope_value": checkpoint_name,
                        "main_value": math.nan,
                        "deletion_value": value,
                        "same_sign": True,
                        "passes": value >= -0.001,
                    }
                )
    for record in economic.loc[
        economic["horizon"].eq("primary_30m_close_t_plus_8")
        & economic["candidate"].isin(["readiness", "pressure", "confirmed"])
    ].itertuples(index=False):
        rows.append(
            {
                "record_type": "economic_reference",
                "deleted_symbol": deleted_symbol,
                "model": str(record.candidate),
                "comparison": "not_applicable",
                "metric": "mean_signed_return_after_friction_bps",
                "scope_type": "friction_bps",
                "scope_value": str(int(record.friction_bps)),
                "main_value": math.nan,
                "deletion_value": float(record.mean_signed_return_after_friction_bps),
                "same_sign": True,
                "passes": True,
            }
        )
    principal = (
        "A2_minus_A1_brier",
        "A2_minus_A1_log_loss",
        "D2_minus_D1_brier",
        "D2_minus_D1_log_loss",
    )
    same_signed = all(
        np.sign(main_increments[name]) == np.sign(deletion_increments[name]) for name in principal
    )
    principal_non_negative = all(deletion_increments[name] >= 0.0 for name in principal)
    no_material_adversity = all(value >= -0.001 for value in checkpoint_improvements)
    summary = {
        "deleted_symbol": deleted_symbol,
        "selection_basis": "largest_admitted_row_share_only",
        "development_rows_after_deletion": len(deletion_development),
        "assessment_primary_rows_after_deletion": len(primary),
        "same_signed_predictive_conclusions": bool(same_signed),
        "principal_increments_non_negative": bool(principal_non_negative),
        "no_materially_adverse_checkpoint": bool(no_material_adversity),
        "main_increments": main_increments,
        "deletion_increments": deletion_increments,
    }
    rows.append(
        {
            "record_type": "stress_summary",
            "deleted_symbol": deleted_symbol,
            "model": "not_applicable",
            "comparison": "all_principal_increments",
            "metric": "stress_passes",
            "scope_type": "pooled",
            "scope_value": "all",
            "main_value": math.nan,
            "deletion_value": float(
                same_signed and principal_non_negative and no_material_adversity
            ),
            "same_sign": bool(same_signed),
            "passes": bool(same_signed and principal_non_negative and no_material_adversity),
        }
    )
    return pd.DataFrame(rows).sort_values(
        ["record_type", "comparison", "metric", "scope_type", "scope_value", "model"],
        kind="mergesort",
    ).reset_index(drop=True), summary


def leave_one_stock_out_diagnostics(primary: pd.DataFrame) -> pd.DataFrame:
    """Calculate no-refit primary increments after deleting each assessment stock."""

    full = primary_increments(primary)
    rows: list[dict[str, Any]] = []
    for symbol in sorted(primary["symbol"].astype(str).unique()):
        subset = primary.loc[~primary["symbol"].astype(str).eq(symbol)]
        values = primary_increments(subset)
        for comparison, value in values.items():
            rows.append(
                {
                    "deleted_symbol": symbol,
                    "comparison": comparison,
                    "rows": len(subset),
                    "sessions": int(subset["session"].nunique()),
                    "stocks": int(subset["symbol"].nunique()),
                    "full_increment": full[comparison],
                    "leave_one_out_increment": value,
                    "same_sign_as_full": bool(np.sign(full[comparison]) == np.sign(value)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["deleted_symbol", "comparison"], kind="mergesort")
        .reset_index(drop=True)
    )


def repaired_decision(
    *,
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    support: Mapping[str, Any],
    concentration: Mapping[str, Any],
    deletion: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen predictive gates, then the repaired post-fit concentration rule."""

    predictive = V0.derive_decision(
        onset,
        direction,
        monthly,
        checkpoint,
        bootstrap,
        nulls,
        economic,
        support,
        {
            "all_concentration_gates_pass": True,
            "interpretation": "concentration deferred until after fitting in V0.1",
        },
    )
    base_decision = str(predictive["decision"])
    economic_not_dominated = bool(concentration["economic_not_dominated_by_largest_stock"])
    final_decision = concentration_aware_decision(
        base_decision,
        maximum_admitted_row_share=float(concentration["maximum_primary_row_stock_share"]),
        deletion_same_signed_conclusions=bool(deletion["same_signed_predictive_conclusions"]),
        principal_increments_non_negative=bool(deletion["principal_increments_non_negative"]),
        no_material_adversity=bool(deletion["no_materially_adverse_checkpoint"]),
        economic_not_dominated=economic_not_dominated,
    )
    output = dict(predictive)
    output.update(SAFETY_FLAGS)
    output["scientific_status"] = SCIENTIFIC_STATUS
    output["decision"] = final_decision
    output["predictive_decision_before_concentration"] = base_decision
    output["support_contract_repair_details"] = {
        "parent_gate_applied_before_admission": True,
        "minimum_admitted_candidate_gate_removed": True,
        "concentration_did_not_prevent_fitting": True,
    }
    output["concentration"] = dict(concentration)
    output["largest_stock_deletion"] = dict(deletion)
    output["concentration_decision_conditions"] = {
        "raw_ten_percent_gate_passes": bool(concentration["primary_row_concentration_passes"]),
        "delete_largest_same_signed_conclusions": bool(
            deletion["same_signed_predictive_conclusions"]
        ),
        "delete_largest_principal_increments_non_negative": bool(
            deletion["principal_increments_non_negative"]
        ),
        "delete_largest_no_material_adversity": bool(deletion["no_materially_adverse_checkpoint"]),
        "economic_not_dominated_by_largest_stock": economic_not_dominated,
    }
    output["economic_reference_cannot_override_probability_gates"] = True
    return output


def _metric_table(frame: pd.DataFrame, models: Sequence[str]) -> str:
    rows = [
        "| Model | Brier | Log loss | AUC | Rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        record = frame.loc[
            frame["population"].eq("primary_high_movement")
            & frame["scope_type"].eq("pooled")
            & frame["model"].eq(model)
        ].iloc[0]
        rows.append(
            f"| {model} | {record['brier_score']:.9f} | {record['log_loss']:.9f} "
            f"| {record['auc']:.9f} | {int(record['rows'])} |"
        )
    return "\n".join(rows)


def render_report(
    *,
    predecessor: Mapping[str, Any],
    comparison: Mapping[str, Any],
    support: Mapping[str, Any],
    parent_accounting: pd.DataFrame,
    admitted_accounting: pd.DataFrame,
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    singleton_metrics: pd.DataFrame,
    multi_metrics: pd.DataFrame,
    concentration: Mapping[str, Any],
    deletion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    """Render the direct V0.1 support-repair report."""

    assessment_parent = parent_accounting.loc[parent_accounting["year"].eq(2025)]
    assessment_admitted = admitted_accounting.loc[admitted_accounting["year"].eq(2025)]
    parent_distribution = (
        assessment_parent["parent_valid_stock_count"].value_counts().sort_index().to_dict()
    )
    admitted_distribution = (
        assessment_admitted["admitted_stock_count"].value_counts().sort_index().to_dict()
    )
    bootstrap_summary = bootstrap.loc[bootstrap["record_type"].eq("summary")]
    null_summary = nulls.loc[nulls["record_type"].eq("summary")]
    economic_primary = economic.loc[economic["horizon"].eq("primary_30m_close_t_plus_8")]
    lines = [
        "# High-Movement Pressure-Onset Screen V0.1 — Support-Semantics Repair",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        f"Scientific status: `{SCIENTIFIC_STATUS}`.",
        "",
        "This is retrospective, research-only, observable-only feasibility evidence. It is "
        "not prospective validation, a strategy, achieved P&L, or executable edge.",
        "",
        "## Repair integrity",
        "",
        "- The 15-stock support gate is applied to the parent fixed-clock slate before "
        "movement admission.",
        "- Singleton admitted slates are retained and receive row weight `1.0`.",
        "- Concentration is evaluated after fitting and cannot hide model results.",
        f"- Frozen V0 panel identity passed across `{comparison['frozen_rows_compared']}` rows "
        f"and `{comparison['frozen_columns_compared']}` pre-fit columns.",
        f"- Frozen predecessor reconstruction passed: `{predecessor['passed']}`; maximum "
        f"probability error `{predecessor['maximum_prediction_absolute_error']:.3g}`.",
        "- Protected rows materialised: `0`.",
        "",
        "## Population",
        "",
        f"- Primary rows / sessions / stocks: `{support['rows']}` / `{support['sessions']}` / "
        f"`{support['stocks']}`.",
        f"- UP / DOWN / NO_ONSET: `{support['up_onsets']}` / `{support['down_onsets']}` / "
        f"`{support['no_onsets']}`.",
        f"- Assessment parent-stock distribution: `{parent_distribution}`.",
        f"- Assessment admitted-stock distribution: `{admitted_distribution}`.",
        f"- Singleton / multi-candidate admitted slates: "
        f"`{int(assessment_admitted['singleton_admitted_slate'].sum())}` / "
        f"`{int(assessment_admitted['multi_candidate_admitted_slate'].sum())}`.",
        "",
        "## Onset occurrence models",
        "",
        _metric_table(onset, ("A0", "A1", "A2", "A3")),
        "",
        "## Direction conditional on actual onset",
        "",
        _metric_table(direction, ("D0", "D1", "D2", "D3")),
        "",
        "## Frozen comparisons",
        "",
    ]
    for name, value in cast(Mapping[str, float], decision["increments"]).items():
        lines.append(f"- `{name}`: `{value:.12g}`")
    lines.extend(["", "## Session-block bootstrap", ""])
    for row in bootstrap_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: 90% `[{row.lower_90:.12g}, {row.upper_90:.12g}]`; "
            f"95% `[{row.lower_95:.12g}, {row.upper_95:.12g}]`."
        )
    lines.extend(["", "## Within-parent-slate bundled null", ""])
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: real `{row.real_value:.12g}`, null q90 "
            f"`{row.null_q90:.12g}`, percentile `{row.real_percentile:.3f}`."
        )
    lines.extend(["", "## Delayed economic reference", ""])
    for candidate in ("readiness", "pressure", "confirmed"):
        candidate_rows = economic_primary.loc[economic_primary["candidate"].eq(candidate)]
        values = {
            int(row.friction_bps): float(row.mean_signed_return_after_friction_bps)
            for row in candidate_rows.itertuples(index=False)
        }
        lines.append(
            f"- `{candidate}`: 0 / 10 / 20 bps = `{values[0]:.6f}` / "
            f"`{values[10]:.6f}` / `{values[20]:.6f}` bps."
        )
    lines.extend(
        [
            f"- Singleton metric rows: `{len(singleton_metrics)}`; multi-candidate metric "
            f"rows: `{len(multi_metrics)}`.",
            "",
            "## Concentration stress",
            "",
            f"- Maximum admitted-row stock share: "
            f"`{float(concentration['maximum_primary_row_stock_share']):.9%}`.",
            f"- Largest stock: `{concentration['largest_admitted_row_stock']}`.",
            f"- Delete-largest same signed conclusions: "
            f"`{deletion['same_signed_predictive_conclusions']}`.",
            f"- Deleted principal increments non-negative: "
            f"`{deletion['principal_increments_non_negative']}`.",
            f"- Economic result not dominated by largest stock: "
            f"`{concentration['economic_not_dominated_by_largest_stock']}`.",
            "",
            "The economic reference is synthetic and gross. It cannot rescue failed "
            "probability gates and does not model borrow, spread, or market impact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_support_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    predecessor_result: Mapping[str, Any],
    predecessor_model: Mapping[str, Any],
    movement_oof: pd.DataFrame,
    movement_folds: Sequence[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    compact: pd.DataFrame,
    ledger: pd.DataFrame,
    assessment: pd.DataFrame,
    models: Mapping[str, FrozenLogisticModel],
    confirmation_manifest: Sequence[Mapping[str, Any]],
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    concentration_ledger: pd.DataFrame,
    support: Mapping[str, Any],
    concentration: Mapping[str, Any],
    decision: Mapping[str, Any],
    source_context: Mapping[str, Any],
    support_repair: Mapping[str, Any],
    parent_accounting: pd.DataFrame,
    admitted_accounting: pd.DataFrame,
    singleton_metrics: pd.DataFrame,
    multi_metrics: pd.DataFrame,
    weight_audit: pd.DataFrame,
    deletion_metrics: pd.DataFrame,
    loso: pd.DataFrame,
    population_comparison: Mapping[str, Any],
    v0_hashes: Mapping[str, Any],
    report: str,
) -> None:
    """Write the unchanged V0 artifact family plus all V0.1 repair artifacts."""

    V0.write_run_artifacts(
        output,
        contract=contract,
        predecessor_result=predecessor_result,
        predecessor_model=predecessor_model,
        movement_oof=movement_oof,
        movement_folds=movement_folds,
        thresholds=thresholds,
        compact=compact,
        ledger=ledger,
        assessment=assessment,
        models=models,
        confirmation_manifest=confirmation_manifest,
        onset=onset,
        direction=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        calibration=calibration,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        concentration=concentration_ledger,
        support=support,
        concentration_summary=concentration,
        decision=decision,
        source_context=source_context,
    )
    configurations = json.loads((output / "model_configurations.json").read_text(encoding="utf-8"))
    configurations.update(SAFETY_FLAGS)
    configurations["scientific_status"] = SCIENTIFIC_STATUS
    configurations["configuration"]["row_weight"] = (
        "precomputed 1 / admitted_stock_count in valid parent slate"
    )
    configurations["configuration"]["direction_stage_denominator"] = (
        "all admitted rows in parent slate, retained after conditional-onset filtering"
    )
    configurations["null_interpretation"] = (
        "full pressure bundle permuted among admitted rows within valid parent slate"
    )
    V0.write_json(output / "model_configurations.json", configurations)
    V0.write_json(output / "support_contract_repair.json", support_repair)
    V0.write_csv(output / "parent_slate_accounting.csv", parent_accounting)
    V0.write_csv(output / "admitted_slate_accounting.csv", admitted_accounting)
    V0.write_csv(output / "singleton_slate_metrics.csv", singleton_metrics)
    V0.write_csv(output / "multi_candidate_slate_metrics.csv", multi_metrics)
    V0.write_csv(output / "weight_audit.csv", weight_audit)
    V0.write_csv(output / "largest_stock_deletion_metrics.csv", deletion_metrics)
    V0.write_csv(output / "leave_one_stock_out_diagnostics.csv", loso)
    V0.write_json(output / "v0_vs_v0_1_population_comparison.json", population_comparison)
    V0.write_json(output / "v0_source_artifact_hashes.json", v0_hashes)
    (output / "report.md").write_text(report, encoding="utf-8")


def execute_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Execute one deterministic V0.1 support-repair run."""

    contract = load_contract()
    v0_hashes = frozen_v0_hashes()
    predecessor = pd.read_parquet(V0.PREDECESSOR_PANEL)
    archived_assessment = pd.read_parquet(V0.PREDECESSOR_PREDICTIONS)
    reconstruction, frozen_model, frozen_probabilities = V0.predecessor_reconstruction(
        predecessor, archived_assessment
    )
    predecessor, movement_oof, movement_folds, thresholds = V0.prepare_movement_probabilities(
        predecessor, frozen_model, frozen_probabilities
    )
    compact, ledger, source_context = V0.build_compact_panel(
        predecessor, provider_root=provider_root
    )
    population_comparison = compare_frozen_population(
        compact,
        thresholds=thresholds,
        barriers=cast(Mapping[str, float], source_context["onset_barriers"]),
        source_context=source_context,
    )
    parent_eligibility = reconstruct_parent_eligibility(predecessor, provider_root=provider_root)
    compact, parent_accounting, admitted_accounting, all_primary = attach_support_hierarchy(
        compact, parent_eligibility
    )
    development = all_primary.loc[all_primary["year"].eq(2024)].copy()
    assessment = compact.loc[compact["year"].eq(2025)].copy()
    primary = all_primary.loc[all_primary["year"].eq(2025)].copy()
    support = repaired_support_summary(primary)
    expected_support = {
        "rows": 1560,
        "sessions": 153,
        "stocks": 20,
        "up_onsets": 345,
        "down_onsets": 336,
        "no_onsets": 879,
    }
    if any(int(support[key]) != value for key, value in expected_support.items()):
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "V0 primary population did not reproduce"
        )
    if not support["aggregate_support_passes"]:
        raise V0.ScreenBlocker(
            "blocked_insufficient_pressure_onset_support", "repaired aggregate support failed"
        )
    models, confirmed_development, confirmation_manifest = fit_repaired_ladder(development)
    scored_assessment = V0.score_model_ladder(assessment, models)
    scored_primary = scored_assessment.loc[
        scored_assessment["high_movement_admitted"].astype(bool)
    ].copy()
    onset, direction, monthly, checkpoint, calibration = V0.evaluate_model_ladder(scored_assessment)
    selections = V0.economic_selections(scored_primary)
    assessment_admitted_accounting = admitted_accounting.loc[
        admitted_accounting["year"].eq(2025)
    ].copy()
    selections, singleton_metrics, slate_metric_context = economic_slate_metrics(
        selections, assessment_admitted_accounting
    )
    multi_metrics = cast(pd.DataFrame, slate_metric_context.pop("multi_metrics"))
    economic = V0.economic_metrics(selections)
    leader = largest_admitted_stock(scored_primary)
    concentration_ledger, concentration = extended_concentration_metrics(
        scored_primary, selections, largest_symbol=leader.symbol
    )
    deletion_metrics, deletion = largest_stock_deletion_diagnostic(
        development,
        assessment,
        scored_primary,
        deleted_symbol=leader.symbol,
    )
    loso = leave_one_stock_out_diagnostics(scored_primary)
    bootstrap = V0.bootstrap_metrics(scored_primary, selections)
    nulls = repaired_null_metrics(
        confirmed_development,
        scored_primary,
        models,
        selections,
    )
    decision = repaired_decision(
        onset=onset,
        direction=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        support=support,
        concentration=concentration,
        deletion=deletion,
    )
    keys = ["symbol", "session", "decision_ordinal"]
    confirmation_values = pd.concat(
        [
            confirmed_development.loc[:, [*keys, *MODEL_DEPENDENT_CONFIRMATION]],
            scored_assessment.loc[:, [*keys, *MODEL_DEPENDENT_CONFIRMATION]],
        ],
        ignore_index=True,
    ).drop_duplicates(keys, keep="last")
    compact_with_confirmation = (
        compact.drop(columns=list(MODEL_DEPENDENT_CONFIRMATION))
        .merge(
            confirmation_values,
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    weight_audit = all_primary.loc[
        :,
        [
            "symbol",
            "session",
            "year",
            "year_month",
            "decision_ordinal",
            "parent_slate_id",
            "parent_raw_source_stock_count",
            "parent_history_complete_stock_count",
            "parent_valid_stock_count",
            "admitted_stock_count",
            "row_weight",
        ],
    ].copy()
    weight_audit["slate_weight_sum"] = weight_audit.groupby("parent_slate_id", sort=True)[
        "row_weight"
    ].transform("sum")
    if not np.allclose(weight_audit["slate_weight_sum"], 1.0, atol=1e-12):
        raise V0.ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "admitted-slate weight audit failed"
        )
    support_repair = {
        **SAFETY_FLAGS,
        "scientific_status": SCIENTIFIC_STATUS,
        "old_incorrect_interpretation": (
            "minimum candidate count applied after movement admission"
        ),
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
        "singleton_admitted_slates_retained": True,
        "concentration_prevents_fitting": False,
        "parent_count_reconstruction": (
            "bounded complete source sessions plus frozen ten-prior-valid-parent "
            "same-clock history; final compact counts used where materialized"
        ),
        "invalid_parent_slates_retained_in_accounting": True,
        "null_interpretation": (
            "admission preserved; bundled pressure features permuted among admitted rows "
            "within the same valid parent slate"
        ),
        "assessment_slate_counts": dict(slate_metric_context),
    }
    report = render_report(
        predecessor=reconstruction,
        comparison=population_comparison,
        support=support,
        parent_accounting=parent_accounting,
        admitted_accounting=admitted_accounting,
        onset=onset,
        direction=direction,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        singleton_metrics=singleton_metrics,
        multi_metrics=multi_metrics,
        concentration=concentration,
        deletion=deletion,
        decision=decision,
    )
    write_support_artifacts(
        output,
        contract=contract,
        predecessor_result=reconstruction,
        predecessor_model=frozen_model,
        movement_oof=movement_oof,
        movement_folds=movement_folds,
        thresholds=thresholds,
        compact=compact_with_confirmation,
        ledger=ledger,
        assessment=scored_assessment,
        models=models,
        confirmation_manifest=confirmation_manifest,
        onset=onset,
        direction=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        calibration=calibration,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        concentration_ledger=concentration_ledger,
        support=support,
        concentration=concentration,
        decision=decision,
        source_context=source_context,
        support_repair=support_repair,
        parent_accounting=parent_accounting,
        admitted_accounting=admitted_accounting,
        singleton_metrics=singleton_metrics,
        multi_metrics=multi_metrics,
        weight_audit=weight_audit,
        deletion_metrics=deletion_metrics,
        loso=loso,
        population_comparison=population_comparison,
        v0_hashes=v0_hashes,
        report=report,
    )
    return {
        "decision": decision["decision"],
        "predictive_decision_before_concentration": decision[
            "predictive_decision_before_concentration"
        ],
        "support": support,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "barriers": source_context["onset_barriers"],
        "parent_slate_distribution_2025": parent_accounting.loc[
            parent_accounting["year"].eq(2025), "parent_valid_stock_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
        "admitted_slate_distribution_2025": admitted_accounting.loc[
            admitted_accounting["year"].eq(2025), "admitted_stock_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
        "singleton_and_multi": slate_metric_context,
        "largest_stock": {
            "symbol": leader.symbol,
            "rows": leader.rows,
            "share": leader.share,
        },
        "predecessor_reconstruction": reconstruction,
        "population_identity": population_comparison["passed"],
        "minimum_timestamp_read": source_context["minimum_timestamp_read"],
        "maximum_timestamp_read": source_context["maximum_timestamp_read"],
    }


def compare_exact_runs(primary: Path, exact: Path) -> dict[str, Any]:
    """Compare every original and support-repair scientific artifact."""

    names = (*V0.SCIENTIFIC_ARTIFACTS, *EXTRA_SCIENTIFIC_ARTIFACTS)
    comparisons: list[dict[str, Any]] = []
    for name in names:
        primary_path = primary / name
        exact_path = exact / name
        if not primary_path.is_file() or not exact_path.is_file():
            raise V0.ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", f"rerun artifact missing: {name}"
            )
        primary_hash = V0.sha256_file(primary_path)
        exact_hash = V0.sha256_file(exact_path)
        mode = "byte_hash"
        passed = primary_hash == exact_hash
        if not passed and name.endswith(".parquet"):
            mode = "strict_numeric_and_value_comparison"
            try:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(primary_path),
                    pd.read_parquet(exact_path),
                    check_exact=True,
                    check_dtype=True,
                    check_like=False,
                )
                passed = True
            except AssertionError:
                passed = False
        comparisons.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "comparison_mode": mode,
                "passed": passed,
            }
        )
    result = {
        **SAFETY_FLAGS,
        "scientific_status": SCIENTIFIC_STATUS,
        "fixed_seeds": {
            "bootstrap": V0.BOOTSTRAP_SEED,
            "null": V0.NULL_SEED,
            "economic_random": V0.RANDOM_SEED,
        },
        "stable_sorting": True,
        "canonical_json": True,
        "deterministic_models": True,
        "comparisons": comparisons,
        "passed": all(record["passed"] for record in comparisons),
    }
    if not result["passed"]:
        failed = [record["artifact"] for record in comparisons if not record["passed"]]
        raise V0.ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", f"exact rerun differs: {failed}"
        )
    return result


def run_independent_auditor(artifacts: Path, *, provider_root: Path) -> None:
    command = [
        sys.executable,
        str(AUDITOR_PATH),
        "--artifacts",
        str(artifacts),
        "--provider-root",
        str(provider_root),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise V0.ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"independent audit failed: {detail}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--primary-output", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-output", type=Path, default=DEFAULT_EXACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary = args.primary_output.resolve()
    exact = args.exact_output.resolve()
    try:
        primary_summary = execute_run(primary, provider_root=args.provider_root)
        exact_summary = execute_run(exact, provider_root=args.provider_root)
        if primary_summary != exact_summary:
            raise V0.ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", "rerun summaries differ"
            )
        rerun = compare_exact_runs(primary, exact)
        rerun["independent_audit_status"] = "pending"
        V0.write_json(primary / "exact_rerun_manifest.json", rerun)
        V0.write_json(exact / "exact_rerun_manifest.json", rerun)
        run_independent_auditor(primary, provider_root=args.provider_root)
        run_independent_auditor(exact, provider_root=args.provider_root)
        primary_audit_hash = V0.sha256_file(primary / "independent_audit.json")
        exact_audit_hash = V0.sha256_file(exact / "independent_audit.json")
        if primary_audit_hash != exact_audit_hash:
            raise V0.ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "independent audit artifacts differ across exact rerun",
            )
        rerun["independent_audit_sha256"] = primary_audit_hash
        rerun["independent_audit_status"] = "passed"
        V0.write_json(primary / "exact_rerun_manifest.json", rerun)
        V0.write_json(exact / "exact_rerun_manifest.json", rerun)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(
            (primary / "report.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(V0.canonical_json({**primary_summary, "exact_rerun": True, "audit": True}))
        return 0
    except V0.ScreenBlocker as exc:
        primary.mkdir(parents=True, exist_ok=True)
        blocked = {
            **SAFETY_FLAGS,
            "scientific_status": SCIENTIFIC_STATUS,
            "decision": exc.code,
            "blocker": exc.detail,
        }
        V0.write_json(primary / "decision.json", blocked)
        print(V0.canonical_json(blocked), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
