#!/usr/bin/env python3
"""Independent audit for loop-quality feature-ablation V3.

The audit imports only the pinned V2 *audit* utilities. It never imports the
V3 production runner. V3 is reconstructed from frozen inputs, sklearn model
specification, stored predictions, and declared gate formulas.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
CONTRACT = WORKSPACE / "work/contracts/20260710-loop-quality-feature-ablation-v3.json"
V2_CONTRACT = WORKSPACE / "work/contracts/20260710-loop-quality-feature-ablation-v2.json"
V2_AUDIT_PATH = WORKSPACE / "work/audit_loop_quality_feature_ablation_v2.py"
RUNNER = WORKSPACE / "work/run_loop_quality_feature_ablation_v3.py"
ROOT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710")
QUALITY_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
V2_ROOT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v2_20260710")

V3_CONTRACT_SHA256 = "221a016e78c353a70261fe724cdfc4d312e355febfc353449844b31b8862702d"
V2_AUDIT_SHA256 = "4fd23383c1223d77812f6114e70de8d8545002e5fc263edd9f3148b9639b7d9e"
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
NEW_MODELS = ("qroute_topology", "qcycle_main", "qcycle_state")
ALL_MODELS = ("qcontext", *NEW_MODELS, "qfull")
MODEL_WIDTHS = {
    "qcontext": 17,
    "qroute_topology": 80,
    "qcycle_main": 37,
    "qcycle_state": 197,
    "qfull": 13157,
}
NUMERIC_CONTROLS = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
OOF_MONTHS = (
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
)
PRIMARY_COMPARISONS = (
    ("qroute_topology_vs_qcontext", "qroute_topology", "qcontext", "superiority"),
    (
        "qroute_topology_noninferiority_vs_qfull",
        "qroute_topology",
        "qfull",
        "retention",
    ),
    ("qcycle_main_vs_qroute_topology", "qcycle_main", "qroute_topology", "superiority"),
    ("qcycle_state_vs_qcycle_main", "qcycle_state", "qcycle_main", "superiority"),
    ("qfull_vs_qcycle_state", "qfull", "qcycle_state", "superiority"),
)
REFERENCE_COMPARISON = (
    "qfull_vs_qcontext_reference",
    "qfull",
    "qcontext",
    "reference",
)
EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if sha256(V2_AUDIT_PATH) != V2_AUDIT_SHA256:
    raise AssertionError("pinned V2 independent audit source changed")
_V2_SPEC = importlib.util.spec_from_file_location("pinned_v2_ablation_audit", V2_AUDIT_PATH)
assert _V2_SPEC is not None and _V2_SPEC.loader is not None
v2audit = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = v2audit
_V2_SPEC.loader.exec_module(v2audit)


def json_safe(value: Any) -> Any:
    return v2audit.json_safe(value)


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, passed: bool, details: Any = None) -> None:
        self.checks.append(
            {"name": name, "pass": bool(passed), "details": json_safe(details)}
        )

    @property
    def all_passed(self) -> bool:
        return all(row["pass"] for row in self.checks)


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def normalized_contract_for_delta(contract: dict[str, Any], version: str) -> dict[str, Any]:
    value = copy.deepcopy(contract)
    for key in (
        "contract_id",
        "contract_frozen_at_utc",
        "scientific_status",
        "model_run_authorized_by_this_file",
        "execution_authorization",
    ):
        value.pop(key, None)
    lineage = value["frozen_lineage"]
    lineage.get("source_pins", {}).pop("v2_contract_sha256", None)
    lineage.pop("v2_support_stop", None)
    support = value["calibration_bins_and_support"]
    support.pop("pooled_oof_minimum_effective_conditional_weight", None)
    support.pop("unique_oof_cohort_support", None)
    safety = value["integrity_and_safety"]
    safety.pop("v3_support_change_scope", None)
    planned = value["planned_artifacts_if_execution_is_separately_authorized"]
    planned["root"] = "<VERSIONED_ROOT>"
    planned["runner"] = "<VERSIONED_RUNNER>"
    planned["independent_audit"] = "<VERSIONED_AUDIT>"
    value["stop_rules"] = [
        rule.replace("separately frozen V3 contract", "separately frozen V<N> contract")
        .replace("separately frozen V4 contract", "separately frozen V<N> contract")
        for rule in value["stop_rules"]
    ]
    return _normalize_numbers(value)


def verify_contract_delta(audit: Audit) -> dict[str, Any]:
    v2 = json.loads(V2_CONTRACT.read_text())
    v3 = json.loads(CONTRACT.read_text())
    audit.check("v3_contract_sha256_exact", sha256(CONTRACT) == V3_CONTRACT_SHA256)
    audit.check(
        "v3_differs_from_v2_only_in_declared_scope",
        normalized_contract_for_delta(v2, "v2")
        == normalized_contract_for_delta(v3, "v3"),
    )
    support = v3["calibration_bins_and_support"]["unique_oof_cohort_support"]
    expected_support = {
        "minimum_total_effective_inverse_overlap_weight": 10000,
        "minimum_effective_inverse_overlap_weight_each_required_quarter": 5000,
        "required_quarters": ["2024_q3", "2024_q4"],
        "minimum_sessions": 100,
        "minimum_stocks": 18,
        "minimum_effective_inverse_overlap_weight_each_stock": 50,
        "minimum_realized_anchor_cycle_rows_reconstruction_integrity": 15000,
        "realized_row_count_is_independent_support_gate": False,
        "repeated_target_horizon_tier_weight_permitted": False,
        "outcome_performance_used_to_choose_support": False,
    }
    actual = {key: support.get(key) for key in expected_support}
    audit.check("v3_unique_cohort_support_semantics_exact", actual == expected_support, actual)
    execution = v3["execution_authorization"]
    audit.check(
        "v3_execution_authorization_is_research_only",
        execution.get("authorized") is True
        and execution.get("live_or_order_authorization") is False
        and v3.get("research_only") is True
        and v3.get("live_ordering_enabled") is False
        and v3.get("order_placement") == "disabled",
    )
    pins = v3["frozen_lineage"]["v2_support_stop"]
    pinned_paths = {
        "v2_contract_sha256": V2_CONTRACT,
        "v2_stop_runner_sha256": WORKSPACE / "work/run_loop_quality_feature_ablation_v2.py",
        "v2_independent_audit_source_sha256": V2_AUDIT_PATH,
        "v2_fit_complete_sha256": V2_ROOT / "fit_complete.json",
        "v2_stop_reason_sha256": V2_ROOT / "stop_reason.json",
        "v2_support_audit_sha256": V2_ROOT / "support_audit.json",
        "v2_pre_score_audit_sha256": V2_ROOT / "pre_score_audit.json",
        "v2_independent_artifact_audit_sha256": V2_ROOT / "independent_artifact_audit.json",
    }
    errors = {
        key: {"contract": pins.get(key), "actual": sha256(path)}
        for key, path in pinned_paths.items()
        if not path.exists() or pins.get(key) != sha256(path)
    }
    audit.check("v3_v2_stop_lineage_hashes_exact", not errors, errors)
    return {"support": support, "v2_lineage": pins}


def reconstruct_unique_support(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    if frame is None:
        frame = pd.read_parquet(
            QUALITY_ROOT / "oof_predictions_2024.parquet",
            columns=[
                "anchor_id",
                "symbol_norm",
                "session_date",
                "quarter",
                "loop_occurs",
                "conditional_weight",
            ],
        )
    positives = frame.loc[frame["loop_occurs"].eq(1)].copy()
    per_anchor = positives.groupby("anchor_id", sort=False)["conditional_weight"].sum()
    quarter = positives.groupby("quarter", sort=True)["conditional_weight"].sum()
    stock = positives.groupby("symbol_norm", sort=True)["conditional_weight"].sum()
    return {
        "compatible_rows": len(frame),
        "realized_rows": len(positives),
        "unique_realized_anchors": int(positives["anchor_id"].nunique()),
        "total_effective_weight": float(positives["conditional_weight"].sum()),
        "quarter_weights": {str(key): float(value) for key, value in quarter.items()},
        "sessions": int(positives["session_date"].nunique()),
        "stocks": int(positives["symbol_norm"].nunique()),
        "minimum_stock_weight": float(stock.min()),
        "all_anchor_weights_equal_one": bool(
            np.allclose(per_anchor.to_numpy(float), 1.0)
        ),
        "support_pass": bool(
            positives["conditional_weight"].sum() >= 10000
            and all(quarter.get(key, 0.0) >= 5000 for key in ("2024_q3", "2024_q4"))
            and positives["session_date"].nunique() >= 100
            and positives["symbol_norm"].nunique() >= 18
            and stock.min() >= 50
            and len(positives) >= 15000
        ),
    }


def verify_unique_support(audit: Audit) -> dict[str, Any]:
    result = reconstruct_unique_support()
    expected = {
        "compatible_rows": 216438,
        "realized_rows": 15584,
        "unique_realized_anchors": 14167,
        "total_effective_weight": 14167.0,
        "quarter_weights": {"2024_q3": 7635.0, "2024_q4": 6532.0},
        "sessions": 128,
        "stocks": 22,
        "minimum_stock_weight": 93.0,
        "all_anchor_weights_equal_one": True,
        "support_pass": True,
    }
    audit.check("v3_unique_cohort_support_reconstructed_exact", result == expected, result)
    return result


def preartifact_audit() -> dict[str, Any]:
    audit = Audit()
    delta = verify_contract_delta(audit)
    support = verify_unique_support(audit)
    mapping, _ = v2audit.build_rotation_mapping()
    audit.check(
        "v3_inherits_exact_44_unit_uniform_topology",
        len(mapping) == 44
        and mapping["compatible_rotation_count"].sum() == 45
        and mapping.loc[mapping["compatible_rotation_count"].gt(1), "route_id"].tolist()
        == ["cycle_15@state_1"],
    )
    parent = v2audit.prefit_audit()
    audit.check(
        "all_pinned_parent_inputs_and_saved_integrity_still_exact",
        parent["all_passed"] is True,
        {"checks": parent["check_count"]},
    )
    return {
        "phase": "v3_preartifact_contract_support",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "contract_delta": delta,
        "support": support,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": False,
    }


def numeric_medians() -> dict[str, float]:
    manifest = json.loads((QUALITY_ROOT / "feature_manifest.json").read_text())
    values = {name: float(manifest["numeric_medians"][name]) for name in NUMERIC_CONTROLS}
    if set(values) != set(NUMERIC_CONTROLS):
        raise AssertionError("numeric median columns changed")
    return values


def sparse_one_hot(indices: np.ndarray, width: int, scale: float) -> sparse.csr_matrix:
    indices = np.asarray(indices, dtype=int)
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= width:
        raise AssertionError("one-hot index outside width")
    return sparse.csr_matrix(
        (
            np.full(len(indices), float(scale), dtype=float),
            (np.arange(len(indices)), indices),
        ),
        shape=(len(indices), width),
    )


def raw_context(
    frame: pd.DataFrame, medians: Mapping[str, float] | None = None
) -> sparse.csr_matrix:
    numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(pd.Series(numeric_medians() if medians is None else medians))
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite context values")
    state = frame["state"].to_numpy(dtype=int)
    state_block = sparse.csr_matrix(np.eye(8, dtype=float)[state])
    result = sparse.hstack((state_block, sparse.csr_matrix(values)), format="csr")
    if result.shape[1] != 17:
        raise AssertionError("context width drift")
    return result


def fit_scaler(raw: sparse.csr_matrix, weights: np.ndarray) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw, sample_weight=np.asarray(weights, dtype=float))
    return scaler


def feature_matrices(
    frame: pd.DataFrame, scaled_context: sparse.csr_matrix
) -> dict[str, sparse.csr_matrix]:
    cycle = frame["cycle_index"].to_numpy(dtype=int)
    state = frame["state"].to_numpy(dtype=int)
    topology = frame.loc[:, v2audit.topology_column_names()].to_numpy(dtype=float).copy()
    topology[:, 19:61] *= 0.5
    cycle_main = sparse_one_hot(cycle, 20, 1.0)
    cycle_state = sparse_one_hot(cycle * 8 + state, 160, 0.5)
    matrices = {
        "qroute_topology": sparse.hstack(
            (scaled_context, sparse.csr_matrix(topology)), format="csr"
        ),
        "qcycle_main": sparse.hstack((scaled_context, cycle_main), format="csr"),
    }
    matrices["qcycle_state"] = sparse.hstack(
        (matrices["qcycle_main"], cycle_state), format="csr"
    )
    for name, matrix in matrices.items():
        if matrix.shape[1] != MODEL_WIDTHS[name]:
            raise AssertionError(f"{name} feature width drift")
    return matrices


def fit_ordered_model(
    matrix: sparse.csr_matrix, target: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=1000,
        random_state=20260710,
    )
    model.fit(matrix, np.asarray(target, dtype=int), sample_weight=np.asarray(weights, float))
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("ordered model classes changed")
    if int(model.n_iter_[0]) >= 1000:
        raise AssertionError("ordered model did not converge")
    return model


def add_topology_independent(frame: pd.DataFrame) -> pd.DataFrame:
    mapping, vectors = v2audit.build_rotation_mapping()
    expected = np.vstack(
        [
            vectors[(str(cycle), int(state))]
            for cycle, state in zip(frame["cycle_id"], frame["state"], strict=True)
        ]
    )
    output = frame.copy()
    output.loc[:, v2audit.topology_column_names()] = expected
    metadata = mapping.set_index(["cycle_id", "current_state"])
    keys = list(zip(output["cycle_id"].astype(str), output["state"].astype(int), strict=True))
    output["current_state"] = output["state"].to_numpy(dtype=int)
    output["compatible_rotation_count"] = np.asarray(
        [int(metadata.loc[key, "compatible_rotation_count"]) for key in keys],
        dtype=int,
    )
    output["compatible_rotations"] = [
        json.dumps(metadata.loc[key, "compatible_rotations"], separators=(",", ":"))
        for key in keys
    ]
    return output


def load_training_frame() -> pd.DataFrame:
    frame = pd.read_parquet(QUALITY_ROOT / "training_long_2024.parquet")
    if not frame["loop_occurs"].eq(1).all():
        raise AssertionError("training frame contains non-realized loop rows")
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {2024}:
        raise AssertionError("training frame crosses year boundary")
    return add_topology_independent(frame)


def model_key(model: str, target: str, horizon: int) -> str:
    return f"{model}__{target}__h{horizon}"


def class_probabilities(model: LogisticRegression, matrix: sparse.csr_matrix) -> np.ndarray:
    probability = np.asarray(model.predict_proba(matrix), dtype=float)
    if probability.shape != (matrix.shape[0], 3):
        raise AssertionError("class probability width drift")
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-12
    ):
        raise AssertionError("invalid ordered probabilities")
    return probability


def compare_array(
    audit: Audit, name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-12
) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        audit.check(name, False, {"actual": actual.shape, "expected": expected.shape})
        return math.inf
    if actual.dtype.kind in "OUS" or expected.dtype.kind in "OUS":
        passed = np.array_equal(actual.astype(str), expected.astype(str))
        audit.check(name, passed)
        return 0.0 if passed else math.inf
    error = float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0.0))
    audit.check(name, error <= tolerance, error)
    return error


def probability_columns(model: str) -> list[str]:
    columns: list[str] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                key = f"{model}__{target}__h{horizon}__{tier}"
                columns.extend((key, f"joint__{key}"))
    return columns


def append_probabilities(
    frame: pd.DataFrame,
    model: str,
    target: str,
    horizon: int,
    class_probability: np.ndarray,
) -> None:
    probability = np.asarray(class_probability, dtype=float)
    if probability.shape != (len(frame), 3):
        raise AssertionError("ordered probability shape changed")
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-12
    ):
        raise AssertionError("invalid ordered probability")
    p75 = probability[:, 1:].sum(axis=1)
    p90 = probability[:, 2]
    if np.any(p90 > p75 + 1e-12) or np.any(p90 < 0.0) or np.any(p75 > 1.0):
        raise AssertionError("ordered probability nesting changed")
    prefix = f"{model}__{target}__h{horizon}"
    frame[f"{prefix}__p75"] = p75
    frame[f"{prefix}__p90"] = p90
    structural = frame["loop_probability"].to_numpy(dtype=float)
    frame[f"joint__{prefix}__p75"] = structural * p75
    frame[f"joint__{prefix}__p90"] = structural * p90


def replay_parent_probabilities(frame: pd.DataFrame) -> None:
    for target in TARGETS:
        for horizon in HORIZONS:
            for model, sealed_name in (("qcontext", "qcontext"), ("qfull", "qcycle")):
                for tier in TIERS:
                    sealed = f"{sealed_name}__{target}__h{horizon}__{tier}"
                    current = f"{model}__{target}__h{horizon}__{tier}"
                    frame[current] = frame[sealed].to_numpy(dtype=float)
                    frame[f"joint__{current}"] = frame[f"joint__{sealed}"].to_numpy(
                        dtype=float
                    )


def merge_causal_controls(frame: pd.DataFrame, anchor_path: Path) -> pd.DataFrame:
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "state", "history_token", *NUMERIC_CONTROLS],
    )
    if anchors["anchor_id"].duplicated().any():
        raise AssertionError("anchor controls are not unique")
    merged = frame.merge(
        anchors,
        on="anchor_id",
        how="left",
        sort=False,
        suffixes=("", "__anchor"),
        validate="many_to_one",
    )
    if len(merged) != len(frame) or merged[list(NUMERIC_CONTROLS)].isna().any().any():
        raise AssertionError("causal-control merge failed")
    for name in ("state", "history_token"):
        if not np.array_equal(
            merged[name].to_numpy(), merged[f"{name}__anchor"].to_numpy()
        ):
            raise AssertionError(f"{name} disagrees with frozen anchor panel")
    return merged.drop(columns=["state__anchor", "history_token__anchor"])


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    order = np.argsort(value, kind="stable")
    value = value[order]
    weight = weight[order]
    locations = (np.cumsum(weight) - 0.5 * weight) / weight.sum()
    return np.interp(np.asarray(quantiles, dtype=float), locations, value)


def entropy_cutpoints(frame: pd.DataFrame) -> np.ndarray:
    realized = frame.loc[frame["loop_occurs"].eq(1)]
    return weighted_quantiles(
        realized["next_state_entropy_normalized"].to_numpy(dtype=float),
        realized["conditional_weight"].to_numpy(dtype=float),
        (0.25, 0.5, 0.75),
    )


def add_entropy_quartile(frame: pd.DataFrame, cutpoints: np.ndarray) -> None:
    frame["entropy_quartile"] = np.searchsorted(
        np.asarray(cutpoints, dtype=float),
        frame["next_state_entropy_normalized"].to_numpy(dtype=float),
        side="left",
    ).astype(np.int8)


def prepare_oof_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = v2audit.load_parent_oof().reset_index(drop=True)
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
    frame = merge_causal_controls(frame, PRICE_ROOT / "anchor_panel_train_2024.parquet")
    frame = add_topology_independent(frame)
    frame = frame.sort_values("source_row", kind="stable").reset_index(drop=True)
    replay_parent_probabilities(frame)
    cuts = entropy_cutpoints(frame)
    add_entropy_quartile(frame, cuts)
    if len(frame) != 216438 or not np.array_equal(
        frame["source_row"].to_numpy(), np.arange(len(frame))
    ):
        raise AssertionError("independent OOF row alignment failed")
    return frame, cuts


def prepare_training_frame() -> pd.DataFrame:
    frame = load_training_frame().reset_index(drop=True)
    frame["month_key"] = pd.to_datetime(
        frame["session_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    if len(frame) != 32677:
        raise AssertionError("training cohort changed")
    return frame


def reconstruct_oof_models(
    training: pd.DataFrame,
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = oof.copy()
    predictions = {
        (model, target, horizon): np.full((len(output), 3), np.nan, dtype=float)
        for model in NEW_MODELS
        for target in TARGETS
        for horizon in HORIZONS
    }
    training_raw = raw_context(training)
    output_raw = raw_context(output)
    training_weight = training["conditional_weight"].to_numpy(dtype=float)
    output_month = pd.to_datetime(
        output["session_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    fold_rows: list[dict[str, Any]] = []
    for fold, month in enumerate(OOF_MONTHS, start=1):
        train_position = np.flatnonzero(training["month_key"].lt(month).to_numpy())
        validation_position = np.flatnonzero(output_month.eq(month).to_numpy())
        if len(train_position) == 0 or len(validation_position) == 0:
            raise AssertionError("empty causal OOF fold")
        weights = training_weight[train_position]
        scaler = fit_scaler(training_raw[train_position], weights)
        training_context = scaler.transform(training_raw[train_position]).tocsr()
        validation_context = scaler.transform(output_raw[validation_position]).tocsr()
        training_fold = training.iloc[train_position].reset_index(drop=True)
        validation_fold = output.iloc[validation_position].reset_index(drop=True)
        training_matrices = feature_matrices(training_fold, training_context)
        validation_matrices = feature_matrices(validation_fold, validation_context)
        for model in NEW_MODELS:
            for target in TARGETS:
                for horizon in HORIZONS:
                    observed = training_fold[
                        f"quality_class__{target}__h{horizon}"
                    ].to_numpy(dtype=int)
                    fitted = fit_ordered_model(training_matrices[model], observed, weights)
                    predictions[(model, target, horizon)][validation_position] = (
                        class_probabilities(fitted, validation_matrices[model])
                    )
                    fold_rows.append(
                        {
                            "fold": fold,
                            "validation_month": month,
                            "training_rows": len(train_position),
                            "training_weight": float(weights.sum()),
                            "validation_compatible_rows": len(validation_position),
                            "model": model,
                            "target": target,
                            "horizon": horizon,
                            "feature_width": training_matrices[model].shape[1],
                            "n_iter": int(fitted.n_iter_[0]),
                            "temperature": 1.0,
                        }
                    )
    for (model, target, horizon), probability in predictions.items():
        if not np.isfinite(probability).all():
            raise AssertionError("OOF replay left uncovered rows")
        append_probabilities(output, model, target, horizon, probability)
    return output, pd.DataFrame(fold_rows)


def reconstruct_full_models(
    training: pd.DataFrame,
    entropy_cuts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = training["conditional_weight"].to_numpy(dtype=float)
    raw = raw_context(training)
    scaler = fit_scaler(raw, weights)
    matrices = feature_matrices(training, scaler.transform(raw).tocsr())
    _, centroid_center, centroid_scale = v2audit.normalized_centroids()
    medians = numeric_medians()
    parameters: dict[str, np.ndarray] = {
        "context_scaler_scale": scaler.scale_.copy(),
        "context_scaler_mean": scaler.mean_.copy(),
        "context_scaler_var": scaler.var_.copy(),
        "context_numeric_medians": np.asarray(
            [medians[column] for column in NUMERIC_CONTROLS], dtype=float
        ),
        "centroid_column_center": np.asarray(centroid_center, dtype=float),
        "centroid_column_scale": np.asarray(centroid_scale, dtype=float),
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts, dtype=float),
    }
    model_audit: dict[str, Any] = {"models": {}}
    for model in NEW_MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model, target, horizon)
                observed = training[
                    f"quality_class__{target}__h{horizon}"
                ].to_numpy(dtype=int)
                fitted = fit_ordered_model(matrices[model], observed, weights)
                parameters[f"{key}__classes"] = fitted.classes_.copy()
                parameters[f"{key}__coef"] = fitted.coef_.copy()
                parameters[f"{key}__intercept"] = fitted.intercept_.copy()
                parameters[f"{key}__n_iter"] = fitted.n_iter_.copy()
                parameters[f"{key}__temperature"] = np.asarray([1.0])
                model_audit["models"][key] = {
                    "feature_width": matrices[model].shape[1],
                    "n_iter": int(fitted.n_iter_[0]),
                    "temperature": 1.0,
                }
    return parameters, model_audit


def binary_losses(observed: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    outcome = np.asarray(observed, dtype=float)
    forecast = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(outcome * np.log(forecast) + (1.0 - outcome) * np.log(1.0 - forecast)),
        "brier": np.square(forecast - outcome),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if len(value) == 0 or weight.sum() <= 0.0:
        return math.nan
    return float(np.average(value, weights=weight))


def weighted_group_mean(
    groups: Iterable[Any], values: np.ndarray, weights: np.ndarray
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "group": pd.Series(groups).astype(str).to_numpy(),
            "weighted_value": np.asarray(values, dtype=float)
            * np.asarray(weights, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    )
    sums = frame.groupby("group", sort=True).sum()
    return sums["weighted_value"] / sums["weight"]


def moving_block_interval(
    values: np.ndarray,
    seed: int,
    confidence: float = 0.99,
    draws: int = 10000,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    block_length = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + block_length] for start in range(len(clean) - block_length + 1)]
    )
    required = int(math.ceil(len(clean) / block_length))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(blocks), size=required)
        samples[draw] = blocks[indices].reshape(-1)[: len(clean)].mean()
    alpha = (1.0 - confidence) / 2.0
    return (
        float(clean.mean()),
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_bin_rows: int,
) -> tuple[float, float]:
    outcome = np.asarray(observed, dtype=float)
    forecast = np.asarray(probability, dtype=float)
    weight = np.asarray(weights, dtype=float)
    bins = np.minimum((forecast * 10.0).astype(int), 9)
    total = float(weight.sum())
    if total <= 0.0:
        return math.nan, math.nan
    ece = 0.0
    supported: list[float] = []
    for bin_number in range(10):
        member = bins == bin_number
        if not member.any():
            continue
        bin_weight = float(weight[member].sum())
        error = abs(
            weighted_mean(outcome[member], weight[member])
            - weighted_mean(forecast[member], weight[member])
        )
        ece += bin_weight / total * error
        if int(member.sum()) >= minimum_bin_rows:
            supported.append(error)
    return float(ece), max(supported) if supported else math.nan


def surface_frame(panel: pd.DataFrame, surface: str) -> tuple[pd.DataFrame, np.ndarray]:
    if surface == "conditional":
        frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
        return frame, frame["conditional_weight"].to_numpy(dtype=float)
    if surface == "joint":
        frame = panel.reset_index(drop=True)
        return frame, np.ones(len(frame), dtype=float)
    raise ValueError(surface)


def cell_observed(
    frame: pd.DataFrame, surface: str, target: str, horizon: int, tier: str
) -> np.ndarray:
    if surface == "conditional":
        ordered = frame[f"quality_class__{target}__h{horizon}"].to_numpy(dtype=int)
        boundary = 1 if tier == "p75" else 2
        return (ordered >= boundary).astype(int)
    label = "good" if tier == "p75" else "high"
    return frame[f"joint_{label}_target__{target}__h{horizon}"].to_numpy(dtype=int)


def probability_column(
    model: str, target: str, horizon: int, tier: str, surface: str
) -> str:
    conditional = f"{model}__{target}__h{horizon}__{tier}"
    return conditional if surface == "conditional" else f"joint__{conditional}"


def build_cell_metrics(panel: pd.DataFrame, period: str, mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface in ("conditional", "joint"):
        frame, weights = surface_frame(panel, surface)
        minimum = {
            ("oof", "conditional"): 50,
            ("oof", "joint"): 250,
            ("score", "conditional"): 100,
            ("score", "joint"): 500,
            ("scoring", "conditional"): 100,
            ("scoring", "joint"): 500,
        }[(mode, surface)]
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    observed = cell_observed(frame, surface, target, horizon, tier)
                    for model in ALL_MODELS:
                        probability = frame[
                            probability_column(model, target, horizon, tier, surface)
                        ].to_numpy(dtype=float)
                        losses = binary_losses(observed, probability)
                        ece, maximum = calibration_summary(
                            observed, probability, weights, minimum
                        )
                        rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "model": model,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "weight": float(weights.sum()),
                                "positives": int(observed.sum()),
                                "weighted_prevalence": weighted_mean(observed, weights),
                                "log_loss": weighted_mean(losses["log_loss"], weights),
                                "brier": weighted_mean(losses["brier"], weights),
                                "ece": ece,
                                "maximum_supported_bin_error": maximum,
                            }
                        )
    return pd.DataFrame(rows)


def cell_difference(
    frame: pd.DataFrame,
    surface: str,
    candidate: str,
    baseline: str,
    target: str,
    horizon: int,
    tier: str,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = cell_observed(frame, surface, target, horizon, tier)
    candidate_probability = frame[
        probability_column(candidate, target, horizon, tier, surface)
    ].to_numpy(dtype=float)
    baseline_probability = frame[
        probability_column(baseline, target, horizon, tier, surface)
    ].to_numpy(dtype=float)
    candidate_loss = binary_losses(observed, candidate_probability)[loss]
    baseline_loss = binary_losses(observed, baseline_probability)[loss]
    return candidate_loss - baseline_loss, candidate_loss, baseline_loss


def pooled_model_loss(panel: pd.DataFrame, surface: str, model: str, loss: str) -> float:
    frame, weights = surface_frame(panel, surface)
    values: list[float] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = cell_observed(frame, surface, target, horizon, tier)
                probability = frame[
                    probability_column(model, target, horizon, tier, surface)
                ].to_numpy(dtype=float)
                values.append(weighted_mean(binary_losses(observed, probability)[loss], weights))
    return float(np.mean(values))


def comparison_surface(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    period: str,
    surface: str,
    comparison_index: int,
    name: str,
    candidate: str,
    baseline: str,
    kind: str,
) -> dict[str, Any]:
    frame, weights = surface_frame(panel, surface)
    cell_differences: dict[tuple[str, int, str, str], np.ndarray] = {}
    cell_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                for loss in ("log_loss", "brier"):
                    difference, _, baseline_loss = cell_difference(
                        frame,
                        surface,
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    cell_differences[(target, horizon, tier, loss)] = difference
                    if loss == "log_loss":
                        selector = (
                            metrics["surface"].eq(surface)
                            & metrics["target"].eq(target)
                            & metrics["horizon"].eq(horizon)
                            & metrics["tier"].eq(tier)
                        )
                        candidate_metric = metrics.loc[
                            selector & metrics["model"].eq(candidate)
                        ].iloc[0]
                        baseline_metric = metrics.loc[
                            selector & metrics["model"].eq(baseline)
                        ].iloc[0]
                        mean_difference = weighted_mean(difference, weights)
                        baseline_mean = weighted_mean(baseline_loss, weights)
                        cell_rows.append(
                            {
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "log_loss_difference": mean_difference,
                                "relative_log_loss_degradation": mean_difference / baseline_mean,
                                "candidate_ece": float(candidate_metric["ece"]),
                                "baseline_ece": float(baseline_metric["ece"]),
                                "ece_difference": float(
                                    candidate_metric["ece"] - baseline_metric["ece"]
                                ),
                                "candidate_maximum_supported_bin_error": float(
                                    candidate_metric["maximum_supported_bin_error"]
                                ),
                                "baseline_maximum_supported_bin_error": float(
                                    baseline_metric["maximum_supported_bin_error"]
                                ),
                                "maximum_supported_bin_error_difference": float(
                                    candidate_metric["maximum_supported_bin_error"]
                                    - baseline_metric["maximum_supported_bin_error"]
                                ),
                            }
                        )

    losses_payload: dict[str, Any] = {}
    for loss_index, loss in enumerate(("log_loss", "brier")):
        cell_means: list[float] = []
        baseline_means: list[float] = []
        daily_columns: list[pd.Series] = []
        quarter_columns: list[pd.Series] = []
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    difference = cell_differences[(target, horizon, tier, loss)]
                    _, _, baseline_loss = cell_difference(
                        frame,
                        surface,
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    cell_means.append(weighted_mean(difference, weights))
                    baseline_means.append(weighted_mean(baseline_loss, weights))
                    column_name = f"{target}_{horizon}_{tier}"
                    daily_columns.append(
                        weighted_group_mean(
                            frame["session_date"], difference, weights
                        ).rename(column_name)
                    )
                    quarter_columns.append(
                        weighted_group_mean(frame["quarter"], difference, weights).rename(
                            column_name
                        )
                    )
        pooled_difference = float(np.mean(cell_means))
        pooled_baseline = float(np.mean(baseline_means))
        daily = pd.concat(daily_columns, axis=1).mean(axis=1)
        quarters = pd.concat(quarter_columns, axis=1).mean(axis=1)
        seed = (
            20260710
            + comparison_index * 100
            + (0 if surface == "conditional" else 10)
            + loss_index
        )
        interval = moving_block_interval(daily.to_numpy(dtype=float), seed)
        deletions: dict[str, float] = {}
        symbols = frame["symbol_norm"].astype(str).to_numpy()
        for symbol in sorted(np.unique(symbols)):
            keep = symbols != symbol
            deletion_cells = [
                weighted_mean(values[keep], weights[keep])
                for key, values in cell_differences.items()
                if key[3] == loss
            ]
            deletions[str(symbol)] = float(np.mean(deletion_cells))
        losses_payload[loss] = {
            "pooled_candidate_minus_baseline": pooled_difference,
            "pooled_baseline": pooled_baseline,
            "relative_improvement": -pooled_difference / pooled_baseline,
            "daily_mean": interval[0],
            "daily_ci_low": interval[1],
            "daily_ci_high": interval[2],
            "quarter_differences": quarters.to_dict(),
            "leave_one_stock_out_differences": deletions,
            "maximum_leave_one_stock_out_difference": max(deletions.values()),
        }

    target_aggregates: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        target_aggregates[target] = {
            loss: float(
                np.mean(
                    [
                        weighted_mean(
                            cell_differences[(target, horizon, tier, loss)], weights
                        )
                        for horizon in HORIZONS
                        for tier in TIERS
                    ]
                )
            )
            for loss in ("log_loss", "brier")
        }
    horizon_aggregates: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        horizon_aggregates[str(horizon)] = {
            loss: float(
                np.mean(
                    [
                        weighted_mean(
                            cell_differences[(target, horizon, tier, loss)], weights
                        )
                        for target in TARGETS
                        for tier in TIERS
                    ]
                )
            )
            for loss in ("log_loss", "brier")
        }
    ece_tolerance = 0.005 if surface == "conditional" else 0.0025
    maximum_tolerance = 0.01 if surface == "conditional" else 0.005
    calibration_pass = all(
        row["ece_difference"] <= ece_tolerance
        and row["maximum_supported_bin_error_difference"] <= maximum_tolerance
        for row in cell_rows
    )
    checks = {
        "minimum_relative_log_loss_improvement": losses_payload["log_loss"]["relative_improvement"] >= 0.0025,
        "pooled_brier_difference_below_zero": losses_payload["brier"]["pooled_candidate_minus_baseline"] < 0.0,
        "daily_log_loss_upper_below_zero": losses_payload["log_loss"]["daily_ci_high"] < 0.0,
        "daily_brier_upper_below_zero": losses_payload["brier"]["daily_ci_high"] < 0.0,
        "every_quarter_log_loss_below_zero": all(value < 0.0 for value in losses_payload["log_loss"]["quarter_differences"].values()),
        "every_quarter_brier_below_zero": all(value < 0.0 for value in losses_payload["brier"]["quarter_differences"].values()),
        "every_stock_deletion_log_loss_below_zero": losses_payload["log_loss"]["maximum_leave_one_stock_out_difference"] < 0.0,
        "every_stock_deletion_brier_below_zero": losses_payload["brier"]["maximum_leave_one_stock_out_difference"] < 0.0,
        "cell_maximum_relative_log_loss_degradation": max(row["relative_log_loss_degradation"] for row in cell_rows) <= 0.0025,
        "target_aggregates_no_worse": all(value <= 0.0 for payload in target_aggregates.values() for value in payload.values()),
        "horizon_aggregates_no_worse": all(value <= 0.0 for payload in horizon_aggregates.values() for value in payload.values()),
        "calibration_noninferiority": calibration_pass,
    }
    return {
        "period": period,
        "surface": surface,
        "comparison": name,
        "candidate": candidate,
        "baseline": baseline,
        "kind": kind,
        "losses": losses_payload,
        "cell_diagnostics": cell_rows,
        "target_aggregates": target_aggregates,
        "horizon_aggregates": horizon_aggregates,
        "common_checks": checks,
        "common_pass": bool(all(checks.values())),
    }


def retention_gate(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    surface_payload: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    gains: dict[str, dict[str, float]] = {}
    for loss in ("log_loss", "brier"):
        context_loss = pooled_model_loss(panel, surface, "qcontext", loss)
        full_loss = pooled_model_loss(panel, surface, "qfull", loss)
        route_loss = pooled_model_loss(panel, surface, "qroute_topology", loss)
        full_gain = context_loss - full_loss
        route_gain = context_loss - route_loss
        gains[loss] = {
            "context_loss": context_loss,
            "full_loss": full_loss,
            "route_loss": route_loss,
            "full_gain": full_gain,
            "route_gain": route_gain,
            "retention": route_gain / full_gain if full_gain > 0.0 else math.nan,
            "noninferiority_margin": 0.1 * full_gain if full_gain > 0.0 else math.nan,
        }
    cell_retention: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                selector = (
                    metrics["surface"].eq(surface)
                    & metrics["target"].eq(target)
                    & metrics["horizon"].eq(horizon)
                    & metrics["tier"].eq(tier)
                )
                indexed = metrics.loc[selector].set_index("model")
                context_loss = float(indexed.loc["qcontext", "log_loss"])
                full_loss = float(indexed.loc["qfull", "log_loss"])
                route_loss = float(indexed.loc["qroute_topology", "log_loss"])
                full_gain = context_loss - full_loss
                route_gain = context_loss - route_loss
                retention = route_gain / full_gain if full_gain > 0.0 else math.nan
                cell_retention.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "tier": tier,
                        "full_gain": full_gain,
                        "route_gain": route_gain,
                        "retention": retention,
                        "pass": bool(full_gain > 0.0 and retention >= 0.75),
                    }
                )
    loss_payload = surface_payload["losses"]
    log_margin = gains["log_loss"]["noninferiority_margin"]
    brier_margin = gains["brier"]["noninferiority_margin"]
    checks = {
        "conditional_or_joint_log_loss_retention_at_least_90pct": gains["log_loss"]["retention"] >= 0.90,
        "conditional_or_joint_brier_retention_at_least_80pct": gains["brier"]["retention"] >= 0.80,
        "log_loss_99pct_upper_within_margin": loss_payload["log_loss"]["daily_ci_high"] <= log_margin,
        "brier_99pct_upper_within_margin": loss_payload["brier"]["daily_ci_high"] <= brier_margin,
        "every_quarter_log_loss_within_margin": all(value <= log_margin for value in loss_payload["log_loss"]["quarter_differences"].values()),
        "every_quarter_brier_within_margin": all(value <= brier_margin for value in loss_payload["brier"]["quarter_differences"].values()),
        "every_stock_log_loss_within_margin": loss_payload["log_loss"]["maximum_leave_one_stock_out_difference"] <= log_margin,
        "every_stock_brier_within_margin": loss_payload["brier"]["maximum_leave_one_stock_out_difference"] <= brier_margin,
        "each_positive_full_gain_cell_retains_75pct": all(row["pass"] for row in cell_retention),
        "calibration_noninferiority_to_full": surface_payload["common_checks"]["calibration_noninferiority"],
    }
    return {
        "gains": gains,
        "cell_retention": cell_retention,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def rotation_diagnostics(
    panel: pd.DataFrame,
    period: str,
    mode: str,
    comparison_name: str,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
    weights = frame["conditional_weight"].to_numpy(dtype=float)
    group_specs = {
        "cycle_current_state": frame["cycle_id"].astype(str)
        + "@"
        + frame["state"].astype(str),
        "compatible_rotation_count": frame["compatible_rotation_count"].astype(str),
        "next_state_entropy_quartile": frame["entropy_quartile"].astype(str),
    }
    cache: dict[str, list[np.ndarray]] = {"log_loss": [], "brier": []}
    for loss in ("log_loss", "brier"):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    difference, _, _ = cell_difference(
                        frame,
                        "conditional",
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    cache[loss].append(difference)
    rows: list[dict[str, Any]] = []
    for group_type, values in group_specs.items():
        for group_value in sorted(values.unique()):
            member = values.eq(group_value).to_numpy()
            subset = frame.loc[member]
            required_rows = 100 if mode == "oof" else 200
            required_quarters = 2 if mode == "oof" else 4
            supported = bool(
                int(member.sum()) >= required_rows
                and subset["symbol_norm"].nunique() >= 10
                and subset["quarter"].nunique() == required_quarters
            )
            pooled = {
                loss: float(
                    np.mean(
                        [
                            weighted_mean(difference[member], weights[member])
                            for difference in cache[loss]
                        ]
                    )
                )
                for loss in ("log_loss", "brier")
            }
            rows.append(
                {
                    "period": period,
                    "comparison": comparison_name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "group_type": group_type,
                    "group_value": group_value,
                    "rows": int(member.sum()),
                    "weight": float(weights[member].sum()),
                    "stocks": int(subset["symbol_norm"].nunique()),
                    "quarters": int(subset["quarter"].nunique()),
                    "supported": supported,
                    "pooled_log_loss_difference": pooled["log_loss"],
                    "pooled_brier_difference": pooled["brier"],
                    "sign_reversal": bool(
                        supported
                        and (
                            pooled["log_loss"] > 0.0
                            or pooled["brier"] > 0.0
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def provisional_attribution(comparison_pass: Mapping[str, bool]) -> dict[str, Any]:
    reference = bool(comparison_pass.get("qfull_vs_qcontext_reference", False))
    topology = bool(comparison_pass.get("qroute_topology_vs_qcontext", False))
    retention = bool(
        comparison_pass.get("qroute_topology_noninferiority_vs_qfull", False)
    )
    identity = bool(comparison_pass.get("qcycle_main_vs_qroute_topology", False))
    state = bool(comparison_pass.get("qcycle_state_vs_qcycle_main", False))
    history = bool(comparison_pass.get("qfull_vs_qcycle_state", False))
    residual: list[str] = []
    if identity:
        residual.append("cycle_identity_representation_needed")
    if state:
        residual.append("current_state_rotation_needed")
    if history:
        residual.append("history_token_needed")
    if not reference:
        label = "no_reference_signal"
    elif topology and retention and not residual:
        label = "topology_sufficient"
    elif topology and retention:
        label = "topology_dominant_with_residual_detail"
    elif identity:
        label = "cycle_identity_representation_needed"
    elif state:
        label = "current_state_rotation_needed"
    elif history:
        label = "history_token_needed"
    else:
        label = "unresolved"
    return {
        "label": label,
        "reference_signal_pass": reference,
        "topology_signal_pass": topology,
        "topology_retention_pass": retention,
        "supported_residual_components": residual,
        "comparison_pass": dict(comparison_pass),
        "prospective_validated": False,
        "frozen_parent_grade_changed": False,
    }


def evaluate_period(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    metrics = build_cell_metrics(panel, period, mode)
    comparison_payload: dict[str, Any] = {}
    comparison_pass: dict[str, bool] = {}
    summary_rows: list[dict[str, Any]] = []
    rotation_frames: list[pd.DataFrame] = []
    for index, (name, candidate, baseline, kind) in enumerate(
        (*PRIMARY_COMPARISONS, REFERENCE_COMPARISON)
    ):
        surfaces: dict[str, Any] = {}
        for surface in ("conditional", "joint"):
            payload = comparison_surface(
                panel,
                metrics,
                period,
                surface,
                index,
                name,
                candidate,
                baseline,
                kind,
            )
            if kind == "retention":
                payload["retention_gate"] = retention_gate(
                    panel, metrics, payload, surface
                )
                payload["surface_pass"] = payload["retention_gate"]["pass"]
            else:
                payload["surface_pass"] = payload["common_pass"]
            surfaces[surface] = payload
        rotations = rotation_diagnostics(
            panel, period, mode, name, candidate, baseline
        )
        rotation_pass = bool(
            not rotations.loc[rotations["supported"], "sign_reversal"].any()
        )
        overall = bool(
            surfaces["conditional"]["surface_pass"]
            and surfaces["joint"]["surface_pass"]
            and rotation_pass
        )
        comparison_pass[name] = overall
        comparison_payload[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "kind": kind,
            "surfaces": surfaces,
            "rotation_no_sign_reversal_pass": rotation_pass,
            "pass": overall,
        }
        summary_rows.append(
            {
                "period": period,
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "kind": kind,
                "conditional_pass": surfaces["conditional"]["surface_pass"],
                "joint_pass": surfaces["joint"]["surface_pass"],
                "rotation_no_sign_reversal_pass": rotation_pass,
                "pass": overall,
                "conditional_relative_log_loss_improvement": surfaces["conditional"]["losses"]["log_loss"]["relative_improvement"],
                "joint_relative_log_loss_improvement": surfaces["joint"]["losses"]["log_loss"]["relative_improvement"],
            }
        )
        rotation_frames.append(rotations)
    attribution = provisional_attribution(comparison_pass)
    gates = {
        "period": period,
        "mode": mode,
        "comparisons": comparison_payload,
        "comparison_pass": comparison_pass,
        "source_attribution": attribution,
    }
    return (
        metrics,
        pd.DataFrame(summary_rows),
        gates,
        pd.concat(rotation_frames, ignore_index=True),
        attribution,
    )


def compare_frame(
    audit: Audit,
    name: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: Sequence[str],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    missing = sorted(set(expected.columns).difference(actual.columns))
    extra = sorted(set(actual.columns).difference(expected.columns))
    if missing or extra:
        audit.check(
            name,
            False,
            {"missing": missing, "extra": extra, "actual_rows": len(actual), "expected_rows": len(expected)},
        )
        return {"maximum_numeric_error": math.inf, "categorical_errors": missing + extra}
    left = actual.sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = expected.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        audit.check(name, False, {"actual_rows": len(left), "expected_rows": len(right)})
        return {"maximum_numeric_error": math.inf, "categorical_errors": ["row_count"]}
    maximum = 0.0
    categorical: list[str] = []
    for column in expected.columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            same_nan = np.isnan(av) == np.isnan(bv)
            if not same_nan.all():
                categorical.append(f"{column}:nan")
                continue
            finite = np.isfinite(av) & np.isfinite(bv)
            error = float(np.max(np.abs(av[finite] - bv[finite]), initial=0.0))
            maximum = max(maximum, error)
            if error > tolerance:
                categorical.append(f"{column}:{error}")
        else:
            equal = (a.astype(str).to_numpy() == b.astype(str).to_numpy()) | (
                a.isna().to_numpy() & b.isna().to_numpy()
            )
            if not equal.all():
                categorical.append(column)
    details = {
        "rows": len(left),
        "maximum_numeric_error": maximum,
        "categorical_errors": categorical,
    }
    audit.check(name, maximum <= tolerance and not categorical, details)
    return details


def json_differences(
    actual: Any,
    expected: Any,
    path: str = "$",
    tolerance: float = 1e-12,
) -> tuple[list[str], float]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}:type"], math.inf
        differences: list[str] = []
        maximum = 0.0
        if set(actual) != set(expected):
            differences.append(
                f"{path}:keys(actual_only={sorted(set(actual)-set(expected))},expected_only={sorted(set(expected)-set(actual))})"
            )
        for key in sorted(set(actual).intersection(expected)):
            nested, error = json_differences(actual[key], expected[key], f"{path}.{key}", tolerance)
            differences.extend(nested)
            maximum = max(maximum, error)
        return differences, maximum
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return [f"{path}:sequence"], math.inf
        differences: list[str] = []
        maximum = 0.0
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            nested, error = json_differences(left, right, f"{path}[{index}]", tolerance)
            differences.extend(nested)
            maximum = max(maximum, error)
        return differences, maximum
    if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        if actual is None and (isinstance(expected, float) and not np.isfinite(expected)):
            return [], 0.0
        try:
            left = float(actual)
            right = float(expected)
        except (TypeError, ValueError):
            return [f"{path}:number"], math.inf
        if not np.isfinite(left) or not np.isfinite(right):
            return ([] if left == right else [f"{path}:nonfinite"]), (0.0 if left == right else math.inf)
        error = abs(left - right)
        return ([] if error <= tolerance else [f"{path}:{error}"]), error
    return ([] if actual == expected else [f"{path}:{actual!r}!={expected!r}"]), 0.0


def compare_json(
    audit: Audit,
    name: str,
    actual: Any,
    expected: Any,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    differences, maximum = json_differences(actual, expected, tolerance=tolerance)
    details = {
        "maximum_numeric_error": maximum,
        "differences": differences[:30],
        "difference_count": len(differences),
    }
    audit.check(name, not differences and maximum <= tolerance, details)
    return details


def independent_rotation_mapping() -> pd.DataFrame:
    mapping, vectors = v2audit.build_rotation_mapping()
    rows: list[dict[str, Any]] = []
    for row in mapping.itertuples(index=False):
        value: dict[str, Any] = {
            "cycle_index": int(row.cycle_index),
            "cycle_id": str(row.cycle_id),
            "cycle": str(row.cycle),
            "transition_length": int(row.transition_length),
            "current_state": int(row.current_state),
            "compatible_rotation_count": int(row.compatible_rotation_count),
            "compatible_rotations": json.dumps(row.compatible_rotations, separators=(",", ":")),
        }
        value.update(
            dict(
                zip(
                    v2audit.topology_column_names(),
                    vectors[(str(row.cycle_id), int(row.current_state))],
                    strict=True,
                )
            )
        )
        rows.append(value)
    return pd.DataFrame(rows)


def independent_topology_manifest(entropy_cuts: np.ndarray) -> dict[str, Any]:
    _, center, scale = v2audit.normalized_centroids()
    return {
        "topology_width": 63,
        "topology_columns": v2audit.topology_column_names(),
        "blocks": [
            {"name": "candidate_next_state_distribution", "start": 0, "stop": 8, "scale": 1.0},
            {"name": "future_route_state_composition", "start": 8, "stop": 16, "scale": 1.0},
            {"name": "transition_length_one_hot", "start": 16, "stop": 19, "scale": 1.0},
            {"name": "next_centroid_expectation", "start": 19, "stop": 33, "scale": 0.5},
            {"name": "route_centroid_expectation", "start": 33, "stop": 47, "scale": 0.5},
            {"name": "next_minus_current_centroid", "start": 47, "stop": 61, "scale": 0.5},
            {"name": "rotation_ambiguity", "start": 61, "stop": 63, "scale": 1.0},
        ],
        "design_rows_store_unscaled_normalized_expectations": True,
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts, dtype=float).tolist(),
        "entropy_quantile_weighting": "2024 OOF realized-loop inverse-overlap weight",
        "future_realized_feature_used": False,
        "stock_identity_feature_used": False,
        "centroid_column_center": np.asarray(center, dtype=float).tolist(),
        "centroid_column_scale": np.asarray(scale, dtype=float).tolist(),
        "topology_columns": v2audit.topology_column_names(),
    }


def independent_feature_manifest(entropy_cuts: np.ndarray) -> dict[str, Any]:
    return {
        "models": {
            "qcontext": {"width": 17, "sealed_reuse": True},
            "qroute_topology": {"width": 80, "fitted": True},
            "qcycle_main": {"width": 37, "fitted": True},
            "qcycle_state": {"width": 197, "fitted": True},
            "qfull": {"width": 13157, "sealed_reuse": True},
        },
        "numeric_controls": list(NUMERIC_CONTROLS),
        "numeric_medians": numeric_medians(),
        "topology_columns": v2audit.topology_column_names(),
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts, dtype=float).tolist(),
        "temperature": 1.0,
        "C": 0.2,
        "solver": "lbfgs",
        "seed": 20260710,
        "stock_identity_feature_used": False,
        "future_realized_feature_used": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def independent_support_payload(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.loc[frame["loop_occurs"].eq(1)]
    quarter = positive.groupby("quarter")["conditional_weight"].sum().to_dict()
    stock = positive.groupby("symbol_norm")["conditional_weight"].sum().to_dict()
    checks = {
        "total_effective_weight": float(positive["conditional_weight"].sum()) >= 10000.0,
        "each_required_quarter_weight": all(float(quarter.get(value, 0.0)) >= 5000.0 for value in ("2024_q3", "2024_q4")),
        "sessions": positive["session_date"].nunique() >= 100,
        "stocks": positive["symbol_norm"].nunique() >= 18,
        "each_stock_effective_weight": bool(stock and min(stock.values()) >= 50.0),
        "realized_rows_reconstruction_integrity": len(positive) >= 15000,
    }
    return {
        "compatible_rows": len(frame),
        "realized_rows": len(positive),
        "unique_realized_anchors": int(positive["anchor_id"].nunique()),
        "total_effective_weight": float(positive["conditional_weight"].sum()),
        "quarter_effective_weight": quarter,
        "sessions": int(positive["session_date"].nunique()),
        "stocks": int(positive["symbol_norm"].nunique()),
        "minimum_stock_effective_weight": float(min(stock.values())),
        "checks": checks,
        "support_pass": bool(all(checks.values())),
        "realized_rows_is_independent_support_gate": False,
    }


FIT_ARTIFACTS = (
    "topology_feature_manifest.json",
    "feature_manifest.json",
    "rotation_mapping.csv",
    "oof_predictions_2024.parquet",
    "model_parameters.npz",
    "fold_audit_2024.csv",
    "support_2024.json",
    "cell_diagnostics_2024.csv",
    "comparison_summary_2024.csv",
    "paired_pooled_gates_2024.json",
    "rotation_diagnostics_2024.csv",
    "two_axis_cycle_diagnostics_2024.csv",
    "provisional_source_attribution.json",
    "fit_source_hashes.json",
    "full_fit_audit.json",
)


def fit_source_paths() -> dict[str, Path]:
    return {
        "per_loop_contract.json": WORKSPACE / "work/contracts/20260710-per-loop-movement-quality-v1.json",
        "v2_contract.json": V2_CONTRACT,
        "v3_contract.json": CONTRACT,
        "runner.py": RUNNER,
        "fixed_cycles.csv": QUALITY_ROOT / "fixed_cycles.csv",
        "frozen_semimarkov_parameters.npz": STATE_ROOT / "frozen_semimarkov_parameters.npz",
        "quality_thresholds_2024.json": QUALITY_ROOT / "quality_thresholds_2024.json",
        "quality_feature_manifest.json": QUALITY_ROOT / "feature_manifest.json",
        "quality_fit_manifest.json": QUALITY_ROOT / "fit_manifest.json",
        "parent_oof_predictions_2024.parquet": QUALITY_ROOT / "oof_predictions_2024.parquet",
        "parent_training_long_2024.parquet": QUALITY_ROOT / "training_long_2024.parquet",
        "anchor_panel_train_2024.parquet": PRICE_ROOT / "anchor_panel_train_2024.parquet",
        "parent_final_cycle_tiers.csv": QUALITY_ROOT / "final_cycle_tiers.csv",
        "parent_gates.json": QUALITY_ROOT / "gates.json",
        "parent_summary.json": QUALITY_ROOT / "summary.json",
    }


def verify_runner_fit_call_graph(audit: Audit) -> dict[str, Any]:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "run_fit_only" not in functions:
        audit.check("v3_runner_has_fit_only_entrypoint", False)
        return {}
    reachable: set[str] = set()
    pending = ["run_fit_only"]
    while pending:
        name = pending.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions and node.func.id not in reachable:
                    pending.append(node.func.id)
    strings = [
        node.value
        for name in reachable
        for node in ast.walk(functions[name])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    forbidden_tokens = (
        "quality_scoring_2025.parquet",
        "quality_scoring_2023.parquet",
        "anchor_panel_2025.parquet",
        "anchor_panel_2023.parquet",
        "shadow_validation",
        "prediction_ledger",
        "broker",
        "order_placement_enabled",
    )
    hits = sorted(
        {
            token
            for token in forbidden_tokens
            if any(token in value for value in strings)
        }
    )
    audit.check(
        "v3_fit_call_graph_has_no_later_outcome_shadow_or_execution_path",
        not hits,
        {"reachable_functions": sorted(reachable), "forbidden_hits": hits},
    )
    lower_source = source.lower()
    shadow_hits = [
        token
        for token in ("shadow_validation/", "prediction_ledger.jsonl", "live_ordering_enabled = true")
        if token in lower_source
    ]
    audit.check("v3_source_has_no_live_shadow_or_order_surface", not shadow_hits, shadow_hits)
    imported_modules = [
        alias.name
        for node in ast.walk(ast.parse(Path(__file__).read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    audit.check(
        "v3_audit_does_not_import_production_runner",
        not any("loop_quality_feature_ablation_v3" in name for name in imported_modules),
        imported_modules,
    )
    return {"runner_sha256": sha256(RUNNER), "fit_reachable_functions": sorted(reachable)}


def verify_fit_freeze(audit: Audit) -> dict[str, Any]:
    marker_path = ROOT / "fit_complete.json"
    required = [*FIT_ARTIFACTS, "fit_source_hashes_pre_fit.json", "fit_complete.json"]
    missing = [name for name in required if not (ROOT / name).is_file()]
    audit.check("v3_fit_artifacts_complete", not missing, missing)
    if missing:
        return {}
    marker = json.loads(marker_path.read_text())
    marker_checks = {
        "status": marker.get("status") == "fit_frozen_pending_independent_pre_score_audit",
        "safety": marker.get("research_only") is True
        and marker.get("live_ordering_enabled") is False
        and marker.get("order_placement") == "disabled",
        "contract": marker.get("contract_sha256") == V3_CONTRACT_SHA256,
        "runner": marker.get("runner_sha256") == sha256(RUNNER),
        "support": marker.get("support_pass") is True,
        "fit": marker.get("model_fit_performed") is True
        and marker.get("oof_predictions_generated") is True,
        "later_closed": marker.get("later_period_panels_read") is False
        and marker.get("later_scoring_authorized") is False
        and marker.get("scoring_authorized") is False,
        "parent_and_shadows": marker.get("parent_grade_changed") is False
        and marker.get("live_shadow_tree_read") is False
        and marker.get("live_shadow_tree_written") is False,
    }
    audit.check("v3_fit_freeze_marker_exact", all(marker_checks.values()), marker_checks)
    expected_hashes = marker.get("artifact_hashes", {})
    actual_hashes = {name: sha256(ROOT / name) for name in FIT_ARTIFACTS}
    audit.check(
        "v3_fit_artifact_hashes_exact",
        expected_hashes == actual_hashes,
        {"stored": expected_hashes, "actual": actual_hashes},
    )
    saved_sources = json.loads((ROOT / "fit_source_hashes.json").read_text())
    pre_sources = json.loads((ROOT / "fit_source_hashes_pre_fit.json").read_text())
    actual_sources = {name: sha256(path) for name, path in fit_source_paths().items()}
    audit.check(
        "v3_pre_and_post_fit_source_hashes_exact",
        saved_sources == pre_sources == actual_sources,
        {"stored": saved_sources, "actual": actual_sources},
    )
    unexpected = sorted(
        path.name
        for path in ROOT.iterdir()
        if any(token in path.name for token in ("scoring_", "period_transfer", "evaluation_source_hashes"))
    )
    audit.check("v3_no_later_scoring_artifact_before_authorization", not unexpected, unexpected)
    audit.check(
        "v3_no_2026_artifact",
        not any("2026" in path.name and "20260710" not in path.name for path in ROOT.iterdir()),
    )
    return marker


def verify_parent_replay(audit: Audit, artifact: pd.DataFrame, parent: pd.DataFrame) -> None:
    key_error = float(
        np.max(
            np.abs(
                artifact["source_row"].to_numpy(dtype=float)
                - np.arange(len(artifact), dtype=float)
            ),
            initial=0.0,
        )
    )
    audit.check(
        "v3_oof_parent_row_ids_exact",
        len(artifact) == len(parent) == 216438
        and key_error == 0.0
        and np.array_equal(artifact["anchor_id"].astype(str), parent["anchor_id"].astype(str))
        and np.array_equal(artifact["cycle_index"].to_numpy(), parent["cycle_index"].to_numpy()),
        {"rows": len(artifact), "source_row_error": key_error},
    )
    maximum = 0.0
    for target in TARGETS:
        for horizon in HORIZONS:
            for model, sealed in (("qcontext", "qcontext"), ("qfull", "qcycle")):
                for tier in TIERS:
                    for prefix in ("", "joint__"):
                        actual = artifact[
                            f"{prefix}{model}__{target}__h{horizon}__{tier}"
                        ].to_numpy(dtype=float)
                        expected = parent[
                            f"{prefix}{sealed}__{target}__h{horizon}__{tier}"
                        ].to_numpy(dtype=float)
                        maximum = max(
                            maximum,
                            float(np.max(np.abs(actual - expected), initial=0.0)),
                        )
    audit.check("v3_sealed_qcontext_qfull_oof_replay_exact", maximum <= 1e-12, maximum)


def verify_probability_integrity(audit: Audit, frame: pd.DataFrame, name: str) -> dict[str, float]:
    maximum_nesting = 0.0
    maximum_chain = 0.0
    maximum_bound = 0.0
    structural = frame["loop_probability"].to_numpy(dtype=float)
    for model in ALL_MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                p75 = frame[f"{model}__{target}__h{horizon}__p75"].to_numpy(dtype=float)
                p90 = frame[f"{model}__{target}__h{horizon}__p90"].to_numpy(dtype=float)
                maximum_nesting = max(
                    maximum_nesting, float(np.max(np.maximum(p90 - p75, 0.0), initial=0.0))
                )
                maximum_bound = max(
                    maximum_bound,
                    float(np.max(np.maximum(-p90, 0.0), initial=0.0)),
                    float(np.max(np.maximum(p75 - 1.0, 0.0), initial=0.0)),
                )
                for tier, conditional in (("p75", p75), ("p90", p90)):
                    joint = frame[f"joint__{model}__{target}__h{horizon}__{tier}"].to_numpy(dtype=float)
                    maximum_chain = max(
                        maximum_chain,
                        float(np.max(np.abs(joint - structural * conditional), initial=0.0)),
                    )
    details = {
        "maximum_nesting_error": maximum_nesting,
        "maximum_chain_error": maximum_chain,
        "maximum_bound_error": maximum_bound,
    }
    audit.check(
        name,
        all(value <= 1e-12 for value in details.values()),
        details,
    )
    return details


def cycle_axis_diagnostics(panel: pd.DataFrame, period: str) -> pd.DataFrame:
    positive = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
    grades = pd.read_csv(QUALITY_ROOT / "final_cycle_tiers.csv").set_index("cycle_id")
    rows: list[dict[str, Any]] = []
    for cycle_id, frame in positive.groupby("cycle_id", sort=True):
        weights = frame["conditional_weight"].to_numpy(dtype=float)
        observed_rates: list[float] = []
        mean_qfull: list[float] = []
        high_cells = 0
        for target in TARGETS:
            for horizon in HORIZONS:
                observed = (
                    frame[f"quality_class__{target}__h{horizon}"].to_numpy(dtype=int)
                    >= 1
                ).astype(int)
                rate = weighted_mean(observed, weights)
                probability = frame[
                    f"qfull__{target}__h{horizon}__p75"
                ].to_numpy(dtype=float)
                mean_probability = weighted_mean(probability, weights)
                observed_rates.append(rate)
                mean_qfull.append(mean_probability)
                high_cells += int(rate >= 0.35 and mean_probability >= 0.35)
        differences: dict[str, float] = {}
        for name, candidate, baseline in (
            ("qfull_vs_qcontext", "qfull", "qcontext"),
            ("qroute_vs_qcontext", "qroute_topology", "qcontext"),
            ("qmain_vs_qroute", "qcycle_main", "qroute_topology"),
            ("qstate_vs_qmain", "qcycle_state", "qcycle_main"),
            ("qfull_vs_qstate", "qfull", "qcycle_state"),
        ):
            cells: list[float] = []
            for target in TARGETS:
                for horizon in HORIZONS:
                    for tier in TIERS:
                        observed = cell_observed(
                            frame, "conditional", target, horizon, tier
                        )
                        candidate_probability = frame[
                            probability_column(
                                candidate, target, horizon, tier, "conditional"
                            )
                        ].to_numpy(dtype=float)
                        baseline_probability = frame[
                            probability_column(
                                baseline, target, horizon, tier, "conditional"
                            )
                        ].to_numpy(dtype=float)
                        cells.append(
                            weighted_mean(
                                binary_losses(observed, candidate_probability)["log_loss"]
                                - binary_losses(observed, baseline_probability)["log_loss"],
                                weights,
                            )
                        )
            differences[name] = float(np.mean(cells))
        rows.append(
            {
                "period": period,
                "cycle_id": str(cycle_id),
                "cycle": str(frame["cycle"].iloc[0]) if "cycle" in frame else np.nan,
                "realized_rows": len(frame),
                "weight": float(weights.sum()),
                "p75_high_level_cells_of_6": high_cells,
                "minimum_p75_observed_rate": min(observed_rates),
                "minimum_p75_mean_qfull": min(mean_qfull),
                "absolute_high_period": high_cells == 6,
                **{
                    f"pooled_log_loss_difference__{name}": value
                    for name, value in differences.items()
                },
                "frozen_parent_grade": grades.loc[cycle_id, "final_grade"],
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    return pd.DataFrame(rows)


def pre_score_audit() -> dict[str, Any]:
    audit = Audit()
    verify_contract_delta(audit)
    verify_unique_support(audit)
    contract = json.loads(CONTRACT.read_text())
    v2audit.verify_static_contract(audit, contract)
    runner = verify_runner_fit_call_graph(audit)
    parent_hashes = v2audit.verify_parent_decisions(audit)
    marker = verify_fit_freeze(audit)
    if not marker:
        return {
            "phase": "v3_independent_pre_score",
            "all_passed": False,
            "check_count": len(audit.checks),
            "checks": audit.checks,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "scoring_authorized": False,
        }

    independent_oof, entropy_cuts = prepare_oof_frame()
    training = prepare_training_frame()
    dates = pd.to_datetime(independent_oof["session_date"], errors="raise")
    audit.check(
        "v3_oof_period_is_exactly_july_december_2024",
        sorted(dates.dt.strftime("%Y-%m").unique().tolist()) == list(OOF_MONTHS)
        and set(dates.dt.year.unique()) == {2024},
        sorted(dates.dt.strftime("%Y-%m").unique().tolist()),
    )
    compare_json(
        audit,
        "v3_support_artifact_independently_exact",
        json.loads((ROOT / "support_2024.json").read_text()),
        independent_support_payload(independent_oof),
    )
    compare_json(
        audit,
        "v3_topology_manifest_independently_exact",
        json.loads((ROOT / "topology_feature_manifest.json").read_text()),
        independent_topology_manifest(entropy_cuts),
    )
    compare_json(
        audit,
        "v3_feature_manifest_independently_exact",
        json.loads((ROOT / "feature_manifest.json").read_text()),
        independent_feature_manifest(entropy_cuts),
    )
    compare_frame(
        audit,
        "v3_rotation_mapping_independently_exact",
        pd.read_csv(ROOT / "rotation_mapping.csv"),
        independent_rotation_mapping(),
        ["cycle_id", "current_state"],
    )

    stored_oof = pd.read_parquet(ROOT / "oof_predictions_2024.parquet")
    verify_parent_replay(audit, stored_oof, independent_oof)
    topology_columns = v2audit.topology_column_names()
    topology_error = float(
        np.max(
            np.abs(
                stored_oof[topology_columns].to_numpy(dtype=float)
                - independent_oof[topology_columns].to_numpy(dtype=float)
            ),
            initial=0.0,
        )
    )
    audit.check("v3_oof_topology_values_independently_exact", topology_error <= 1e-12, topology_error)
    metadata_errors = []
    for column in ("current_state", "compatible_rotation_count", "compatible_rotations", "entropy_quartile"):
        if not np.array_equal(stored_oof[column].astype(str), independent_oof[column].astype(str)):
            metadata_errors.append(column)
    audit.check("v3_oof_rotation_metadata_independently_exact", not metadata_errors, metadata_errors)

    reconstructed_oof, reconstructed_folds = reconstruct_oof_models(training, independent_oof)
    model_columns = [column for model in NEW_MODELS for column in probability_columns(model)]
    maximum_oof_error = float(
        np.max(
            np.abs(
                stored_oof[model_columns].to_numpy(dtype=float)
                - reconstructed_oof[model_columns].to_numpy(dtype=float)
            ),
            initial=0.0,
        )
    )
    audit.check(
        "v3_all_108_causal_oof_model_predictions_independently_exact",
        maximum_oof_error <= 1e-12,
        {"maximum_error": maximum_oof_error, "models": 108},
    )
    compare_frame(
        audit,
        "v3_all_oof_fold_rows_and_convergence_independently_exact",
        pd.read_csv(ROOT / "fold_audit_2024.csv"),
        reconstructed_folds,
        ["fold", "model", "target", "horizon"],
    )
    verify_probability_integrity(audit, stored_oof, "v3_oof_probability_nesting_bounds_and_chain_exact")

    reconstructed_parameters, reconstructed_full_audit = reconstruct_full_models(
        training, entropy_cuts
    )
    with np.load(ROOT / "model_parameters.npz") as stored_parameters:
        stored_keys = set(stored_parameters.files)
        expected_keys = set(reconstructed_parameters)
        audit.check(
            "v3_full_model_parameter_keys_exact",
            stored_keys == expected_keys,
            {"stored_only": sorted(stored_keys - expected_keys), "expected_only": sorted(expected_keys - stored_keys)},
        )
        maximum_parameter_error = 0.0
        for key in sorted(stored_keys.intersection(expected_keys)):
            actual = np.asarray(stored_parameters[key])
            expected = np.asarray(reconstructed_parameters[key])
            if actual.shape != expected.shape:
                maximum_parameter_error = math.inf
                break
            maximum_parameter_error = max(
                maximum_parameter_error,
                float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0.0)),
            )
    audit.check(
        "v3_all_18_full_models_scaler_and_parameters_independently_exact",
        maximum_parameter_error <= 1e-12,
        {"maximum_error": maximum_parameter_error, "models": 18},
    )
    compare_json(
        audit,
        "v3_full_fit_convergence_audit_independently_exact",
        json.loads((ROOT / "full_fit_audit.json").read_text()),
        reconstructed_full_audit,
    )

    metrics, comparison_summary, gates, rotations, attribution = evaluate_period(
        reconstructed_oof, "2024_oof", "oof"
    )
    compare_frame(
        audit,
        "v3_2024_cell_diagnostics_independently_exact",
        pd.read_csv(ROOT / "cell_diagnostics_2024.csv"),
        metrics,
        ["surface", "model", "target", "horizon", "tier"],
    )
    compare_frame(
        audit,
        "v3_2024_comparison_summary_independently_exact",
        pd.read_csv(ROOT / "comparison_summary_2024.csv"),
        comparison_summary,
        ["comparison"],
    )
    compare_json(
        audit,
        "v3_2024_paired_bootstrap_and_gate_payload_independently_exact",
        json.loads((ROOT / "paired_pooled_gates_2024.json").read_text()),
        gates,
    )
    compare_frame(
        audit,
        "v3_2024_rotation_diagnostics_independently_exact",
        pd.read_csv(ROOT / "rotation_diagnostics_2024.csv"),
        rotations,
        ["comparison", "group_type", "group_value"],
    )
    compare_frame(
        audit,
        "v3_2024_two_axis_cycle_diagnostics_independently_exact",
        pd.read_csv(ROOT / "two_axis_cycle_diagnostics_2024.csv"),
        cycle_axis_diagnostics(reconstructed_oof, "2024_oof"),
        ["cycle_id"],
    )
    compare_json(
        audit,
        "v3_provisional_source_attribution_independently_exact",
        json.loads((ROOT / "provisional_source_attribution.json").read_text()),
        attribution,
    )
    audit.check(
        "v3_provisional_attribution_cannot_promote_parent_grades",
        attribution.get("frozen_parent_grade_changed") is False
        and attribution.get("prospective_validated") is False,
        attribution,
    )

    result = {
        "phase": "v3_independent_pre_score",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": audit.all_passed,
        "later_period_outcomes_opened_by_audit": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
        "runner": runner,
        "parent_decision_hashes": parent_hashes,
        "fit_marker_sha256": sha256(ROOT / "fit_complete.json"),
        "maximum_oof_prediction_error": maximum_oof_error,
        "maximum_full_parameter_error": maximum_parameter_error,
        "provisional_source_attribution": attribution,
    }
    write_json(ROOT / "pre_score_audit.json", result)
    return result


def parent_label_columns() -> list[str]:
    columns: list[str] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.extend(
                (
                    f"quality_class__{target}__h{horizon}",
                    f"joint_good_target__{target}__h{horizon}",
                    f"joint_high_target__{target}__h{horizon}",
                )
            )
    return columns


def sealed_parent_columns() -> list[str]:
    columns: list[str] = []
    for parent_model in ("qcontext", "qcycle"):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    key = f"{parent_model}__{target}__h{horizon}__{tier}"
                    columns.extend((key, f"joint__{key}"))
    return columns


def load_npz_parameters() -> dict[str, np.ndarray]:
    with np.load(ROOT / "model_parameters.npz") as stored:
        return {key: stored[key].copy() for key in stored.files}


def parameter_scaler(parameters: Mapping[str, np.ndarray]) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = np.asarray(parameters["context_scaler_scale"], dtype=float).copy()
    scaler.mean_ = np.asarray(parameters["context_scaler_mean"], dtype=float).copy()
    scaler.var_ = np.asarray(parameters["context_scaler_var"], dtype=float).copy()
    scaler.n_features_in_ = len(scaler.scale_)
    scaler.n_samples_seen_ = 1
    return scaler


def parameter_probabilities(
    matrix: sparse.csr_matrix,
    parameters: Mapping[str, np.ndarray],
    key: str,
) -> np.ndarray:
    coefficient = np.asarray(parameters[f"{key}__coef"], dtype=float)
    intercept = np.asarray(parameters[f"{key}__intercept"], dtype=float)
    logits = np.asarray(matrix @ coefficient.T) + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def prepare_scoring_frame(
    period: str,
    parameters: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    parent_path = QUALITY_ROOT / f"quality_scoring_{period}.parquet"
    anchor_path = PRICE_ROOT / f"anchor_panel_{period}.parquet"
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "cycle",
        "transition_length",
        "state",
        "history_token",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
        *parent_label_columns(),
        *sealed_parent_columns(),
    ]
    panel = pd.read_parquet(parent_path, columns=columns).reset_index(drop=True)
    panel.insert(0, "source_row", np.arange(len(panel), dtype=np.int64))
    panel = merge_causal_controls(panel, anchor_path)
    panel = add_topology_independent(panel)
    panel = panel.sort_values("source_row", kind="stable").reset_index(drop=True)
    replay_parent_probabilities(panel)
    cuts = np.asarray(parameters["entropy_quartile_cutpoints"], dtype=float)
    add_entropy_quartile(panel, cuts)
    medians = {
        column: float(parameters["context_numeric_medians"][index])
        for index, column in enumerate(NUMERIC_CONTROLS)
    }
    raw = raw_context(panel, medians)
    scaled = parameter_scaler(parameters).transform(raw).tocsr()
    matrices = feature_matrices(panel, scaled)
    for model in NEW_MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model, target, horizon)
                append_probabilities(
                    panel,
                    model,
                    target,
                    horizon,
                    parameter_probabilities(matrices[model], parameters, key),
                )
    return panel


def scoring_support(panel: pd.DataFrame) -> dict[str, Any]:
    positive = panel.loc[panel["loop_occurs"].eq(1)]
    weight = float(positive["conditional_weight"].sum())
    return {
        "compatible_rows": len(panel),
        "realized_rows": len(positive),
        "effective_weight": weight,
        "minimum_effective_weight": 25000.0,
        "support_pass": weight >= 25000.0,
        "quarters": sorted(positive["quarter"].astype(str).unique()),
        "stocks": int(positive["symbol_norm"].nunique()),
    }


def portable_attribution(
    provisional: Mapping[str, Any], period_gates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    provisional_pass = provisional["comparison_pass"]
    portable = {
        name: bool(
            passed
            and period_gates["2025"]["comparison_pass"].get(name, False)
            and period_gates["2023"]["comparison_pass"].get(name, False)
        )
        for name, passed in provisional_pass.items()
    }
    label = provisional["label"]
    required = {
        "no_reference_signal": (),
        "topology_sufficient": (
            "qfull_vs_qcontext_reference",
            "qroute_topology_vs_qcontext",
            "qroute_topology_noninferiority_vs_qfull",
        ),
        "topology_dominant_with_residual_detail": (
            "qfull_vs_qcontext_reference",
            "qroute_topology_vs_qcontext",
            "qroute_topology_noninferiority_vs_qfull",
        ),
        "cycle_identity_representation_needed": (
            "qfull_vs_qcontext_reference",
            "qcycle_main_vs_qroute_topology",
        ),
        "current_state_rotation_needed": (
            "qfull_vs_qcontext_reference",
            "qcycle_state_vs_qcycle_main",
        ),
        "history_token_needed": (
            "qfull_vs_qcontext_reference",
            "qfull_vs_qcycle_state",
        ),
        "unresolved": ("qfull_vs_qcontext_reference",),
    }[label]
    retained = bool(all(portable.get(name, False) for name in required))
    final_label = label if label == "no_reference_signal" or retained else "unresolved_not_portable"
    return {
        "provisional_2024_label": label,
        "final_development_portability_label": final_label,
        "provisional_label_retained": retained,
        "portable_comparison_pass": portable,
        "later_period_promotion_performed": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
    }


SCORING_ARTIFACTS = tuple(
    name
    for period in ("2025", "2023")
    for name in (
        f"scoring_predictions_{period}.parquet",
        f"cell_diagnostics_{period}.csv",
        f"comparison_summary_{period}.csv",
        f"rotation_diagnostics_{period}.csv",
        f"paired_pooled_gates_{period}.json",
        f"support_{period}.json",
    )
) + (
    "period_transfer_gates.json",
    "source_attribution.json",
    "two_axis_cycle_diagnostics.csv",
    "evaluation_source_hashes.json",
)


def evaluation_source_paths() -> dict[str, Path]:
    return {
        "quality_scoring_2025.parquet": QUALITY_ROOT / "quality_scoring_2025.parquet",
        "quality_scoring_2023.parquet": QUALITY_ROOT / "quality_scoring_2023.parquet",
        "anchor_panel_2025.parquet": PRICE_ROOT / "anchor_panel_2025.parquet",
        "anchor_panel_2023.parquet": PRICE_ROOT / "anchor_panel_2023.parquet",
    }


def verify_scoring_freeze(audit: Audit) -> tuple[dict[str, Any], dict[str, Any]]:
    required = [*SCORING_ARTIFACTS, "scoring_complete.json", "summary.json"]
    missing = [name for name in required if not (ROOT / name).is_file()]
    audit.check("v3_scoring_artifacts_complete", not missing, missing)
    if missing:
        return {}, {}
    fit_marker = json.loads((ROOT / "fit_complete.json").read_text())
    preaudit = json.loads((ROOT / "pre_score_audit.json").read_text())
    marker = json.loads((ROOT / "scoring_complete.json").read_text())
    checks = {
        "status": marker.get("status")
        == "scoring_complete_development_and_backward_portability_only",
        "safety": marker.get("research_only") is True
        and marker.get("live_ordering_enabled") is False
        and marker.get("order_placement") == "disabled",
        "contract": marker.get("contract_sha256") == V3_CONTRACT_SHA256,
        "runner_frozen": marker.get("runner_sha256")
        == fit_marker.get("runner_sha256")
        == sha256(RUNNER),
        "preaudit": preaudit.get("all_passed") is True
        and preaudit.get("scoring_authorized") is True
        and marker.get("pre_score_audit_passed") is True
        and marker.get("pre_score_audit_sha256") == sha256(ROOT / "pre_score_audit.json"),
        "period_status": marker.get("later_periods_are_prospective") is False,
        "parent_and_shadows": marker.get("parent_grade_changed") is False
        and marker.get("live_shadow_tree_read") is False
        and marker.get("live_shadow_tree_written") is False,
    }
    audit.check("v3_scoring_freeze_marker_exact", all(checks.values()), checks)
    stored_hashes = marker.get("artifact_hashes", {})
    actual_hashes = {name: sha256(ROOT / name) for name in SCORING_ARTIFACTS}
    audit.check(
        "v3_scoring_artifact_hashes_exact",
        stored_hashes == actual_hashes,
        {"stored": stored_hashes, "actual": actual_hashes},
    )
    evaluation_hashes = json.loads((ROOT / "evaluation_source_hashes.json").read_text())
    actual_evaluation = {
        name: sha256(path) for name, path in evaluation_source_paths().items()
    }
    audit.check(
        "v3_evaluation_source_hashes_exact",
        evaluation_hashes == actual_evaluation,
        {"stored": evaluation_hashes, "actual": actual_evaluation},
    )
    fit_hashes = json.loads((ROOT / "fit_source_hashes.json").read_text())
    audit.check(
        "v3_fit_sources_and_artifacts_unchanged_through_scoring",
        fit_hashes == {name: sha256(path) for name, path in fit_source_paths().items()}
        and fit_marker.get("artifact_hashes")
        == {name: sha256(ROOT / name) for name in FIT_ARTIFACTS},
    )
    tree = ast.parse(RUNNER.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    score_node = functions.get("run_scoring")
    lock_node = functions.get("validate_fit_and_pre_score_lock")
    score_calls = {
        node.func.id: node.lineno
        for node in ast.walk(score_node) if score_node is not None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    lock_text = ast.get_source_segment(RUNNER.read_text(), lock_node) if lock_node else ""
    audit.check(
        "v3_scoring_source_enforces_pre_score_lock_before_panel_load",
        score_calls.get("validate_fit_and_pre_score_lock", math.inf)
        < score_calls.get("scoring_source_paths", math.inf)
        and "pre_score_audit.json" in (lock_text or "")
        and "scoring_authorized" in (lock_text or ""),
        score_calls,
    )
    return marker, preaudit


def verify_scoring_parent_replay(
    audit: Audit,
    stored: pd.DataFrame,
    independent: pd.DataFrame,
    period: str,
) -> None:
    ids_exact = (
        len(stored) == len(independent)
        and np.array_equal(stored["source_row"].to_numpy(), np.arange(len(stored)))
        and np.array_equal(stored["anchor_id"].astype(str), independent["anchor_id"].astype(str))
        and np.array_equal(stored["cycle_index"].to_numpy(), independent["cycle_index"].to_numpy())
    )
    audit.check(f"v3_{period}_parent_row_ids_exact", ids_exact, len(stored))
    maximum = 0.0
    for target in TARGETS:
        for horizon in HORIZONS:
            for model, sealed in (("qcontext", "qcontext"), ("qfull", "qcycle")):
                for tier in TIERS:
                    for prefix in ("", "joint__"):
                        expected_name = f"{prefix}{sealed}__{target}__h{horizon}__{tier}"
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    np.abs(
                                        stored[f"{prefix}{model}__{target}__h{horizon}__{tier}"].to_numpy(dtype=float)
                                        - independent[expected_name].to_numpy(dtype=float)
                                    ),
                                    initial=0.0,
                                )
                            ),
                        )
    audit.check(f"v3_{period}_sealed_parent_probability_replay_exact", maximum <= 1e-12, maximum)


def post_score_audit() -> dict[str, Any]:
    audit = Audit()
    marker, preaudit = verify_scoring_freeze(audit)
    v2audit.verify_parent_decisions(audit)
    if not marker:
        result = {
            "phase": "v3_independent_post_score",
            "all_passed": False,
            "check_count": len(audit.checks),
            "checks": audit.checks,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        }
        return result
    parameters = load_npz_parameters()
    period_gates: dict[str, dict[str, Any]] = {}
    period_axes: list[pd.DataFrame] = [
        pd.read_csv(ROOT / "two_axis_cycle_diagnostics_2024.csv")
    ]
    maximum_prediction_error: dict[str, float] = {}
    for period in ("2025", "2023"):
        independent = prepare_scoring_frame(period, parameters)
        artifact_path = ROOT / f"scoring_predictions_{period}.parquet"
        schema_columns = pq.ParquetFile(artifact_path).schema_arrow.names
        dates = pd.to_datetime(independent["session_date"], errors="raise")
        audit.check(
            f"v3_{period}_period_exact_and_no_2026",
            set(dates.dt.year.unique()) == {int(period)} and not dates.dt.year.eq(2026).any(),
            sorted(dates.dt.year.unique().tolist()),
        )
        audit.check(
            f"v3_{period}_scoring_schema_exact",
            set(schema_columns) == set(independent.columns),
            {
                "stored_only": sorted(set(schema_columns) - set(independent.columns)),
                "expected_only": sorted(set(independent.columns) - set(schema_columns)),
            },
        )
        sealed_output_columns = [
            column
            for model in ("qcontext", "qfull")
            for column in probability_columns(model)
        ]
        stored_parent = pd.read_parquet(
            artifact_path,
            columns=["source_row", "anchor_id", "cycle_index", *sealed_output_columns],
        )
        verify_scoring_parent_replay(audit, stored_parent, independent, period)
        topology = v2audit.topology_column_names()
        stored_topology = pd.read_parquet(
            artifact_path,
            columns=[*topology, "current_state", "compatible_rotation_count", "compatible_rotations", "entropy_quartile"],
        )
        topology_error = float(
            np.max(
                np.abs(
                    stored_topology[topology].to_numpy(dtype=float)
                    - independent[topology].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
        audit.check(f"v3_{period}_topology_independently_exact", topology_error <= 1e-12, topology_error)
        metadata_errors = [
            column
            for column in ("current_state", "compatible_rotation_count", "compatible_rotations", "entropy_quartile")
            if not np.array_equal(stored_topology[column].astype(str), independent[column].astype(str))
        ]
        audit.check(f"v3_{period}_rotation_metadata_independently_exact", not metadata_errors, metadata_errors)
        model_columns = [column for model in NEW_MODELS for column in probability_columns(model)]
        stored_models = pd.read_parquet(artifact_path, columns=model_columns)
        prediction_error = float(
            np.max(
                np.abs(
                    stored_models[model_columns].to_numpy(dtype=float)
                    - independent[model_columns].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
        maximum_prediction_error[period] = prediction_error
        audit.check(
            f"v3_{period}_all_new_scoring_predictions_independently_exact",
            prediction_error <= 1e-12,
            prediction_error,
        )
        verify_probability_integrity(
            audit, independent, f"v3_{period}_probability_nesting_bounds_and_chain_exact"
        )
        compare_json(
            audit,
            f"v3_{period}_support_independently_exact",
            json.loads((ROOT / f"support_{period}.json").read_text()),
            scoring_support(independent),
        )
        metrics, comparisons, gates, rotations, _ = evaluate_period(
            independent, period, "scoring"
        )
        period_gates[period] = gates
        compare_frame(
            audit,
            f"v3_{period}_cell_diagnostics_independently_exact",
            pd.read_csv(ROOT / f"cell_diagnostics_{period}.csv"),
            metrics,
            ["surface", "model", "target", "horizon", "tier"],
        )
        compare_frame(
            audit,
            f"v3_{period}_comparison_summary_independently_exact",
            pd.read_csv(ROOT / f"comparison_summary_{period}.csv"),
            comparisons,
            ["comparison"],
        )
        compare_json(
            audit,
            f"v3_{period}_paired_bootstrap_and_gates_independently_exact",
            json.loads((ROOT / f"paired_pooled_gates_{period}.json").read_text()),
            gates,
        )
        compare_frame(
            audit,
            f"v3_{period}_rotation_diagnostics_independently_exact",
            pd.read_csv(ROOT / f"rotation_diagnostics_{period}.csv"),
            rotations,
            ["comparison", "group_type", "group_value"],
        )
        period_axes.append(cycle_axis_diagnostics(independent, period))
        del independent, stored_parent, stored_topology, stored_models, metrics, comparisons, rotations
        gc.collect()

    provisional = json.loads((ROOT / "provisional_source_attribution.json").read_text())
    transfer = portable_attribution(provisional, period_gates)
    compare_json(
        audit,
        "v3_period_transfer_gate_independently_exact",
        json.loads((ROOT / "period_transfer_gates.json").read_text()),
        transfer,
    )
    compare_json(
        audit,
        "v3_final_source_attribution_independently_exact",
        json.loads((ROOT / "source_attribution.json").read_text()),
        transfer,
    )
    compare_frame(
        audit,
        "v3_all_period_two_axis_cycle_diagnostics_independently_exact",
        pd.read_csv(ROOT / "two_axis_cycle_diagnostics.csv"),
        pd.concat(period_axes, ignore_index=True),
        ["period", "cycle_id"],
    )
    audit.check(
        "v3_transfer_is_demotion_only_and_not_prospective",
        transfer.get("later_period_promotion_performed") is False
        and transfer.get("prospective_validated") is False
        and transfer.get("parent_grade_changed") is False,
        transfer,
    )
    summary = json.loads((ROOT / "summary.json").read_text())
    audit.check(
        "v3_summary_safety_and_lineage_exact",
        summary.get("research_only") is True
        and summary.get("live_ordering_enabled") is False
        and summary.get("order_placement") == "disabled"
        and summary.get("fit_complete_sha256") == sha256(ROOT / "fit_complete.json")
        and summary.get("source_attribution") == transfer
        and summary.get("parent_grade_changed") is False
        and summary.get("later_periods_are_prospective") is False,
        summary.get("interpretation"),
    )
    result = {
        "phase": "v3_independent_post_score",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
        "maximum_scoring_prediction_error": maximum_prediction_error,
        "source_attribution": transfer,
        "pre_score_audit_sha256": sha256(ROOT / "pre_score_audit.json"),
        "scoring_complete_sha256": sha256(ROOT / "scoring_complete.json"),
    }
    write_json(ROOT / "independent_artifact_audit.json", result)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preartifact-only", action="store_true")
    parser.add_argument("--pre-score-only", action="store_true")
    parser.add_argument("--post-score", action="store_true")
    arguments = parser.parse_args()
    selected = (
        int(arguments.preartifact_only)
        + int(arguments.pre_score_only)
        + int(arguments.post_score)
    )
    if selected != 1:
        raise SystemExit(
            "select exactly one of --preartifact-only, --pre-score-only, or --post-score"
        )
    if arguments.preartifact_only:
        result = preartifact_audit()
    elif arguments.pre_score_only:
        result = pre_score_audit()
    else:
        result = post_score_audit()
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
