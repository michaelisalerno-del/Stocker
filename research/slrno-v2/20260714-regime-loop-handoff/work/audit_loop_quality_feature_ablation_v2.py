#!/usr/bin/env python3
"""Independent audit for the stopped V2 loop-quality feature ablation.

This audit deliberately does not import ``run_loop_quality_feature_ablation_v2``.
It reconstructs the frozen rotation topology and verifies the deterministic
OOF-support stop before any V2 model, later-period score, or attribution can be
created.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WORKSPACE = Path(__file__).resolve().parents[1]
CONTRACT = WORKSPACE / "work/contracts/20260710-loop-quality-feature-ablation-v2.json"
RUNNER = WORKSPACE / "work/run_loop_quality_feature_ablation_v2.py"
QUALITY_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
DEFAULT_ROOT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v2_20260710")

PARENT_OOF = QUALITY_ROOT / "oof_predictions_2024.parquet"
PARENT_FIT_COMPLETE = QUALITY_ROOT / "fit_complete.json"
PARENT_AUDIT = QUALITY_ROOT / "independent_artifact_audit.json"

TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
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
MODEL_WIDTHS = {
    "qcontext": 17,
    "qroute_topology": 80,
    "qcycle_main": 37,
    "qcycle_state": 197,
    "qfull": 13157,
}
PRIMARY_COMPARISONS = (
    "qroute_topology versus qcontext",
    "qroute_topology non-inferiority versus qfull",
    "qcycle_main versus qroute_topology",
    "qcycle_state versus qcycle_main",
    "qfull versus qcycle_state",
)
EXPECTED_PINNED_HASHES = {
    "per_loop_movement_quality_v1_contract_sha256": (
        WORKSPACE / "work/contracts/20260710-per-loop-movement-quality-v1.json",
        "67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e",
    ),
    "fixed_cycles_csv_sha256": (
        QUALITY_ROOT / "fixed_cycles.csv",
        "bf9292fa51de1e545e5a319fa2e2faf2088926acd5315b9106597b1da318b253",
    ),
    "frozen_semimarkov_parameters_npz_sha256": (
        STATE_ROOT / "frozen_semimarkov_parameters.npz",
        "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251",
    ),
    "quality_thresholds_2024_json_sha256": (
        QUALITY_ROOT / "quality_thresholds_2024.json",
        "f9e2355e36dae28e4279dfabe74645cb3a363b706d95d4179955093b80015b72",
    ),
    "quality_feature_manifest_json_sha256": (
        QUALITY_ROOT / "feature_manifest.json",
        "b3db72b43ad15f89ac8fedd182d2c5eee3931786dc600a339dc8714dad89ddd6",
    ),
    "quality_fit_manifest_json_sha256": (
        QUALITY_ROOT / "fit_manifest.json",
        "0a911d631fcc98445d60fa098a219313fabf3d88e78aa1137ab8b684c0e9ee58",
    ),
}
PARENT_DECISION_HASHES = {
    "final_cycle_tiers.csv": "2d4e4bd2ef26db396244fe7cd20a8485aba1814eaeacf5326916823225d7c598",
    "provisional_tiers_2024.csv": "2a46df0499d9abd2bdf8b144d8b4353c6e90560b5b4f1c64971238630d6bf973",
    "gates.json": "0e64c9a9dee02b1860117078a811387f64ec6324e7edb2ec2b4b2104ee3b7637",
    "summary.json": "5c409a890e47b3b0fd86ae9a0052c905498ac6b1d84a32414811f7b4acee3930",
    "prospective_shadow_pre_content_snapshot.json": (
        "3f7dd2899b1005d5852924ae8f4318f974a2a3d786fa09845aa9d06ef4455b21"
    ),
    "prospective_shadow_post_content_snapshot.json": (
        "3f7dd2899b1005d5852924ae8f4318f974a2a3d786fa09845aa9d06ef4455b21"
    ),
}
PARENT_OOF_SHA256 = "689b8853ec482c07a46faea48f49665df8c92612ef28bc9934fe2df2e97e7d30"
FORBIDDEN_EXECUTION_OUTPUTS = (
    "oof_predictions_2024.parquet",
    "model_parameters.npz",
    "fold_audit_2024.csv",
    "cell_metrics_2024.csv",
    "paired_pooled_gates_2024.json",
    "comparison_summary_2024.csv",
    "rotation_diagnostics_2024.csv",
    "scoring_predictions_2025.parquet",
    "scoring_predictions_2023.parquet",
    "cell_metrics_scoring.csv",
    "scoring_diagnostics_2025.csv",
    "scoring_diagnostics_2023.csv",
    "period_transfer_gates.json",
    "source_attribution.json",
    "evaluation_source_hashes.json",
    "scoring_complete.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


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


def compatible_rotations(
    core: tuple[int, ...], current_state: int
) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current_state),)
            for index, state in enumerate(core)
            if int(state) == int(current_state)
        }
    )


def normalized_centroids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters = np.load(STATE_ROOT / "frozen_semimarkov_parameters.npz")
    means = np.asarray(parameters["means"], dtype=float)
    if means.shape != (8, 14):
        raise AssertionError(f"invalid centroid shape {means.shape}")
    center = means.mean(axis=0)
    scale = means.std(axis=0, ddof=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (means - center) / safe_scale
    standardized[:, scale == 0.0] = 0.0
    return standardized, center, scale


def topology_column_names() -> list[str]:
    return (
        [f"next_state_p_{state}" for state in range(8)]
        + [f"route_state_fraction_{state}" for state in range(8)]
        + [f"length_is_{length}" for length in (2, 3, 4)]
        + [f"next_centroid_z_{index:02d}" for index in range(14)]
        + [f"route_centroid_z_{index:02d}" for index in range(14)]
        + [f"next_minus_current_centroid_z_{index:02d}" for index in range(14)]
        + ["rotation_count_minus_one", "next_state_entropy_normalized"]
    )


def topology_values(
    core: tuple[int, ...], current_state: int, standardized: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    rotations = compatible_rotations(core, current_state)
    if not rotations:
        raise AssertionError("cycle/current-state pair has no compatible rotation")
    length = len(core)
    next_distribution = np.zeros(8, dtype=float)
    composition = np.zeros(8, dtype=float)
    for route in rotations:
        next_distribution[int(route[1])] += 1.0 / len(rotations)
        for state in route[1:]:
            composition[int(state)] += 1.0 / (len(rotations) * length)
    length_one_hot = np.asarray(
        [float(length == value) for value in (2, 3, 4)], dtype=float
    )
    next_centroid = next_distribution @ standardized
    route_centroid = composition @ standardized
    delta = next_centroid - standardized[int(current_state)]
    positive = next_distribution[next_distribution > 0.0]
    entropy = (
        float(-(positive * np.log(positive)).sum() / math.log(8.0))
        if len(positive) > 1
        else 0.0
    )
    values = np.concatenate(
        [
            next_distribution,
            composition,
            length_one_hot,
            next_centroid,
            route_centroid,
            delta,
            np.asarray([len(rotations) - 1.0, entropy]),
        ]
    )
    if values.shape != (63,):
        raise AssertionError(f"topology width drift: {values.shape}")
    metadata = {
        "compatible_rotations": [
            "->".join(str(value) for value in route) for route in rotations
        ],
        "compatible_rotation_count": len(rotations),
        "transition_length": length,
        "next_state_distribution": next_distribution.tolist(),
        "future_route_state_composition": composition.tolist(),
        "next_state_entropy_normalized": entropy,
    }
    return values, metadata


def build_rotation_mapping() -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    standardized, _, _ = normalized_centroids()
    cycles = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv")
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, int], np.ndarray] = {}
    for cycle in cycles.itertuples(index=False):
        closed = tuple(int(value) for value in str(cycle.cycle).split("->"))
        if closed[0] != closed[-1]:
            raise AssertionError(f"open frozen cycle {cycle.cycle}")
        core = closed[:-1]
        for current_state in sorted(set(core)):
            vector, metadata = topology_values(core, current_state, standardized)
            vectors[(str(cycle.cycle_id), int(current_state))] = vector
            rows.append(
                {
                    "route_id": f"{cycle.cycle_id}@state_{current_state}",
                    "cycle_id": str(cycle.cycle_id),
                    "cycle_index": int(cycle.cycle_index),
                    "current_state": int(current_state),
                    "cycle": str(cycle.cycle),
                    **metadata,
                }
            )
    mapping = pd.DataFrame(rows).sort_values(
        ["cycle_id", "current_state"], kind="stable"
    )
    if len(mapping) != 44 or mapping.duplicated(["cycle_id", "current_state"]).any():
        raise AssertionError("independent rotation mapping is not 44 unique units")
    return mapping.reset_index(drop=True), vectors


def parent_oof_columns() -> list[str]:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "state",
        "history_token",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
    ]
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.extend(
                [
                    f"quality_class__{target}__h{horizon}",
                    f"joint_good_target__{target}__h{horizon}",
                    f"joint_high_target__{target}__h{horizon}",
                ]
            )
            for parent_model in ("qcontext", "qcycle"):
                for tier in TIERS:
                    columns.extend(
                        [
                            f"{parent_model}__{target}__h{horizon}__{tier}",
                            f"joint__{parent_model}__{target}__h{horizon}__{tier}",
                        ]
                    )
    return columns


def load_parent_oof() -> pd.DataFrame:
    return pd.read_parquet(PARENT_OOF, columns=parent_oof_columns())


def independent_support_and_reference() -> dict[str, Any]:
    frame = load_parent_oof()
    positives = frame.loc[frame["loop_occurs"].eq(1)].copy()
    per_anchor = positives.groupby("anchor_id", sort=False)["conditional_weight"].sum()
    weight = float(positives["conditional_weight"].sum())
    maximum_chain_error = 0.0
    maximum_nesting_error = 0.0
    maximum_probability_violation = 0.0
    for target in TARGETS:
        for horizon in HORIZONS:
            for parent_model in ("qcontext", "qcycle"):
                p75 = frame[
                    f"{parent_model}__{target}__h{horizon}__p75"
                ].to_numpy(float)
                p90 = frame[
                    f"{parent_model}__{target}__h{horizon}__p90"
                ].to_numpy(float)
                maximum_nesting_error = max(
                    maximum_nesting_error, float(np.maximum(p90 - p75, 0.0).max())
                )
                maximum_probability_violation = max(
                    maximum_probability_violation,
                    float(np.maximum(-p90, 0.0).max()),
                    float(np.maximum(p75 - 1.0, 0.0).max()),
                )
                for tier, probability in (("p75", p75), ("p90", p90)):
                    joint = frame[
                        f"joint__{parent_model}__{target}__h{horizon}__{tier}"
                    ].to_numpy(float)
                    error = np.max(
                        np.abs(
                            joint
                            - frame["loop_probability"].to_numpy(float) * probability
                        )
                    )
                    maximum_chain_error = max(maximum_chain_error, float(error))
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    return {
        "compatible_rows": len(frame),
        "positive_rows": len(positives),
        "unique_positive_anchors": int(positives["anchor_id"].nunique()),
        "effective_conditional_weight": weight,
        "all_positive_anchor_weights_equal_one": bool(
            np.allclose(per_anchor.to_numpy(float), 1.0)
        ),
        "oof_months": sorted(dates.dt.strftime("%Y-%m").unique().tolist()),
        "maximum_chain_rule_error": maximum_chain_error,
        "maximum_probability_nesting_error": maximum_nesting_error,
        "maximum_probability_bound_violation": maximum_probability_violation,
        "support_threshold": 20000.0,
        "support_pass": weight >= 20000.0,
    }


def _extract_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [piece.strip() for piece in value.split("||") if piece.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    raise AssertionError(f"cannot parse rotation list {value!r}")


def verify_design_rows(
    audit: Audit, root: Path, vectors: dict[tuple[str, int], np.ndarray]
) -> dict[str, Any]:
    design_path = root / "oof_design_rows_2024.parquet"
    if not design_path.exists():
        audit.check("oof_design_rows_present", False, str(design_path))
        return {}
    schema = pq.ParquetFile(design_path).schema.names
    required = {
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "state",
        "history_token",
        "loop_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
        *NUMERIC_CONTROLS,
    }
    missing = sorted(required.difference(schema))
    audit.check("oof_design_rows_required_columns", not missing, missing)
    if missing:
        return {}
    design = pd.read_parquet(design_path)
    parent = load_parent_oof()
    key = ["anchor_id", "cycle_index"]
    design = design.sort_values(key, kind="stable").reset_index(drop=True)
    parent = parent.sort_values(key, kind="stable").reset_index(drop=True)
    audit.check(
        "oof_design_row_count_and_unique_keys",
        len(design) == len(parent)
        and not design.duplicated(key).any()
        and not parent.duplicated(key).any(),
        {"design": len(design), "parent": len(parent)},
    )
    common = [
        "anchor_id",
        "cycle_index",
        "cycle_id",
        "state",
        "history_token",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
        "loop_probability",
        "symbol_norm",
        "session_date",
        "quarter",
    ]
    errors: dict[str, Any] = {}
    for column in common:
        left = design[column].to_numpy()
        right = parent[column].to_numpy()
        if np.issubdtype(design[column].dtype, np.number):
            error = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            if error > 1e-12:
                errors[column] = error
        elif not np.array_equal(left.astype(str), right.astype(str)):
            errors[column] = "categorical_mismatch"
    for target in TARGETS:
        for horizon in HORIZONS:
            for column in (
                f"quality_class__{target}__h{horizon}",
                f"joint_good_target__{target}__h{horizon}",
                f"joint_high_target__{target}__h{horizon}",
            ):
                if column not in design:
                    errors[column] = "missing"
                elif not np.array_equal(
                    design[column].to_numpy(), parent[column].to_numpy()
                ):
                    errors[column] = "label_mismatch"
    audit.check("oof_design_parent_ids_labels_weights_exact", not errors, errors)

    dates = pd.to_datetime(design["session_date"], errors="raise")
    audit.check(
        "oof_design_causal_months_and_no_2026",
        sorted(dates.dt.strftime("%Y-%m").unique().tolist()) == list(OOF_MONTHS)
        and set(dates.dt.year.unique()) == {2024},
        sorted(dates.dt.strftime("%Y-%m").unique().tolist()),
    )
    forbidden_feature_columns = [
        column
        for column in schema
        if column in {"stock_identity", "realized_rotation_selector"}
        or column.startswith(("future_state_", "future_duration", "run_end"))
    ]
    audit.check(
        "oof_design_has_no_future_selector_or_stock_feature",
        not forbidden_feature_columns,
        forbidden_feature_columns,
    )

    feature_manifest_path = root / "topology_feature_manifest.json"
    if not feature_manifest_path.exists():
        audit.check("topology_feature_manifest_present", False)
        return {"rows": len(design)}
    manifest = json.loads(feature_manifest_path.read_text())
    topology_columns = manifest.get("topology_columns", topology_column_names())
    audit.check(
        "topology_manifest_ordered_width_63",
        isinstance(topology_columns, list)
        and len(topology_columns) == 63
        and len(set(topology_columns)) == 63,
        topology_columns,
    )
    missing_topology = sorted(set(topology_columns).difference(design.columns))
    audit.check("oof_design_topology_columns_present", not missing_topology, missing_topology)
    max_error = math.inf
    if not missing_topology and len(topology_columns) == 63:
        expected = np.vstack(
            [
                vectors[(str(cycle), int(state))]
                for cycle, state in zip(
                    design["cycle_id"], design["state"], strict=True
                )
            ]
        )
        actual = design[topology_columns].to_numpy(float)
        # A manifest may explicitly say that the stored centroid block is scaled.
        stored_centroid_scale = float(manifest.get("stored_centroid_scale", 1.0))
        if stored_centroid_scale != 1.0:
            expected[:, 19:61] *= stored_centroid_scale
        max_error = float(np.max(np.abs(expected - actual)))
    audit.check("oof_design_topology_values_exact", max_error <= 1e-12, max_error)
    positives = design.loc[design["loop_occurs"].eq(1)]
    positive_weight = float(positives["conditional_weight"].sum())
    per_anchor = positives.groupby("anchor_id")["conditional_weight"].sum()
    audit.check(
        "oof_design_unique_weight_support_stop_exact",
        len(positives) == 15584
        and positives["anchor_id"].nunique() == 14167
        and positive_weight == 14167.0
        and np.allclose(per_anchor.to_numpy(float), 1.0),
        {
            "positive_rows": len(positives),
            "unique_positive_anchors": positives["anchor_id"].nunique(),
            "effective_weight": positive_weight,
        },
    )
    return {
        "rows": len(design),
        "positive_rows": len(positives),
        "effective_weight": positive_weight,
        "maximum_topology_error": max_error,
    }


def verify_rotation_mapping(
    audit: Audit, root: Path, expected: pd.DataFrame
) -> dict[str, Any]:
    path = root / "rotation_mapping.csv"
    if not path.exists():
        audit.check("rotation_mapping_present", False)
        return {}
    actual = pd.read_csv(path).sort_values(
        ["cycle_id", "current_state"], kind="stable"
    ).reset_index(drop=True)
    expected = expected.sort_values(
        ["cycle_id", "current_state"], kind="stable"
    ).reset_index(drop=True)
    audit.check(
        "rotation_mapping_44_unique_units",
        len(actual) == 44
        and not actual.duplicated(["cycle_id", "current_state"]).any(),
        len(actual),
    )
    key_match = len(actual) == len(expected) and np.array_equal(
        actual[["cycle_id", "current_state"]].astype(str).to_numpy(),
        expected[["cycle_id", "current_state"]].astype(str).to_numpy(),
    )
    audit.check("rotation_mapping_cycle_state_keys_exact", key_match)
    errors: list[str] = []
    if key_match:
        for index in range(len(actual)):
            row = actual.iloc[index]
            exp = expected.iloc[index]
            rotation_column = next(
                (
                    column
                    for column in (
                        "compatible_rotations",
                        "compatible_rotations_json",
                        "oriented_routes",
                    )
                    if column in actual.columns
                ),
                None,
            )
            if rotation_column is None:
                errors.append("missing compatible rotations column")
                break
            if _extract_json_list(row[rotation_column]) != exp["compatible_rotations"]:
                errors.append(str(exp["route_id"]))
            if "compatible_rotation_count" in actual and int(
                row["compatible_rotation_count"]
            ) != int(exp["compatible_rotation_count"]):
                errors.append(f"count:{exp['route_id']}")
    audit.check("rotation_mapping_dedup_uniform_routes_exact", not errors, errors)
    ambiguous = expected.loc[
        expected["compatible_rotation_count"].gt(1), "route_id"
    ].tolist()
    audit.check(
        "rotation_ambiguity_is_only_cycle15_state1",
        ambiguous == ["cycle_15@state_1"],
        ambiguous,
    )
    return {"rows": len(actual), "ambiguous_routes": ambiguous}


def verify_static_contract(audit: Audit, contract: dict[str, Any]) -> None:
    safety = (
        contract.get("research_only") is True
        and contract.get("live_ordering_enabled") is False
        and contract.get("order_placement") == "disabled"
        and contract.get("economic_edge_claim") is False
        and contract.get("deployment_enabled") is False
        and contract["periods"]["partial_2026_permitted"] is False
        and contract["periods"]["prospective_claim_permitted"] is False
    )
    audit.check("contract_safety_and_period_status_exact", safety)
    audit.check(
        "contract_causal_oof_schedule_exact",
        tuple(contract["periods"]["causal_oof_months"]) == OOF_MONTHS
        and contract["periods"]["oof_schedule"].startswith(
            "For each July-December month"
        ),
    )
    widths = {name: int(spec["width"] if "width" in spec else spec["total_width"])
              for name, spec in contract["models"].items()}
    audit.check("contract_model_feature_widths_exact", widths == MODEL_WIDTHS, widths)
    topology_blocks = contract["models"]["qroute_topology"]["topology_blocks"]
    topology_width = sum(int(block["width"]) for block in topology_blocks)
    audit.check(
        "contract_topology_blocks_width_and_scales_exact",
        topology_width == 63
        and [int(block["width"]) for block in topology_blocks] == [8, 8, 3, 42, 2]
        and [float(block["feature_scale"]) for block in topology_blocks]
        == [1.0, 1.0, 1.0, 0.5, 1.0],
        topology_blocks,
    )
    audit.check(
        "contract_sealed_qcontext_qfull_replay_and_no_refit",
        contract["models"]["qcontext"]["implementation"].startswith("reuse")
        and contract["models"]["qfull"]["implementation"].startswith("reuse")
        and float(contract["fair_comparison_rules"]["qcontext_and_qfull_replay_tolerance"])
        == 1e-12
        and contract["fair_comparison_rules"]["new_models_to_fit_if_execution_is_later_authorized"]
        == ["qroute_topology", "qcycle_main", "qcycle_state"],
    )
    comparisons = tuple(contract["multiplicity_and_uncertainty"]["primary_comparison_family"])
    multiplicity = contract["multiplicity_and_uncertainty"]
    audit.check("five_primary_comparisons_exact", comparisons == PRIMARY_COMPARISONS, comparisons)
    audit.check(
        "bonferroni_99pct_multiplicity_plan_exact",
        float(multiplicity["familywise_alpha"]) == 0.05
        and int(multiplicity["block_length_sessions"]) == 5
        and int(multiplicity["bootstrap_draws"]) == 10000
        and int(multiplicity["seed"]) == 20260710
        and multiplicity["same_resampled_block_indices_for_every_model_in_a_comparison"]
        is True,
    )
    superiority = contract["common_superiority_gate"]
    audit.check(
        "superiority_gate_all_required_components_frozen",
        float(superiority["minimum_pooled_relative_log_loss_improvement"]) == 0.0025
        and float(superiority["each_of_twelve_cells_maximum_relative_log_loss_degradation"])
        == 0.0025
        and superiority["both_conditional_and_joint_surfaces_must_pass"] is True
        and superiority["every_required_quarter_pooled_log_loss_and_brier_difference_below_zero"]
        is True
        and superiority["every_leave_one_stock_out_pooled_log_loss_and_brier_difference_below_zero"]
        is True,
    )
    retention = contract["topology_signal_retention_gate"]
    audit.check(
        "topology_retention_gate_exact",
        float(retention["conditional_log_loss_gain_retained_minimum"]) == 0.9
        and float(retention["joint_log_loss_gain_retained_minimum"]) == 0.9
        and float(retention["conditional_brier_gain_retained_minimum"]) == 0.8
        and float(retention["joint_brier_gain_retained_minimum"]) == 0.8
        and float(retention["each_cell_maximum_full_gain_loss"]) == 0.25
        and retention["both_surfaces_must_pass"] is True,
    )
    rotation = contract["rotation_level_diagnostics"]
    audit.check(
        "rotation_slices_and_no_sign_reversal_rule_exact",
        rotation["causal_slices"]
        == [
            "cycle_id by current_state",
            "candidate compatible_rotation_count",
            "candidate-next-state entropy quartile using 2024-frozen cut points",
        ]
        and int(rotation["oof_minimum_realized_rows"]) == 100
        and int(rotation["scoring_minimum_realized_rows"]) == 200
        and int(rotation["minimum_stocks"]) == 10
        and "no supported causal current-state or rotation-ambiguity slice"
        in rotation["source_attribution_no_sign_reversal_rule"],
    )
    audit.check(
        "five_comparison_bootstrap_seed_mapping_frozen",
        [
            20260710 + comparison * 100 + surface * 10 + loss
            for comparison in range(5)
            for surface in range(2)
            for loss in range(2)
        ]
        == sorted(
            {
                20260710 + comparison * 100 + surface * 10 + loss
                for comparison in range(5)
                for surface in range(2)
                for loss in range(2)
            }
        ),
    )


def verify_parent_decisions(audit: Audit) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in PARENT_DECISION_HASHES.items():
        path = QUALITY_ROOT / name
        actual = sha256(path) if path.exists() else "missing"
        hashes[name] = actual
        audit.check(f"parent_{name}_unchanged", actual == expected, actual)
    before = json.loads(
        (QUALITY_ROOT / "prospective_shadow_pre_content_snapshot.json").read_text()
    )
    after = json.loads(
        (QUALITY_ROOT / "prospective_shadow_post_content_snapshot.json").read_text()
    )
    audit.check(
        "parent_saved_shadow_snapshots_equal_without_live_shadow_read",
        before == after
        and before.get("runtime_outcomes_opened") is False
        and before.get("ledger_size") == 0
        and before.get("ledger_lines") == 0,
        {
            "tree_sha256": before.get("tree_sha256"),
            "ledger_size": before.get("ledger_size"),
            "outcomes_opened": before.get("runtime_outcomes_opened"),
        },
    )
    return hashes


def verify_runner_static_safety(audit: Audit) -> dict[str, Any]:
    if not RUNNER.exists():
        audit.check("v2_runner_present_for_static_safety_audit", False)
        return {}
    source = RUNNER.read_text()
    forbidden = [
        value
        for value in (
            "quality_scoring_2025.parquet",
            "quality_scoring_2023.parquet",
            "anchor_panel_2025",
            "anchor_panel_2023",
            "shadow_validation/",
            "prediction_ledger.jsonl",
        )
        if value in source
    ]
    audit.check("runner_has_no_later_panel_or_live_shadow_path", not forbidden, forbidden)
    audit.check(
        "runner_source_contains_explicit_support_stop_labels",
        all(
            value in source
            for value in (
                "stopped_before_model_fit",
                "later_scoring_authorized",
                "source_attribution_permitted",
            )
        ),
    )
    return {"sha256": sha256(RUNNER), "forbidden_path_hits": forbidden}


def verify_fit_stop(audit: Audit, root: Path) -> dict[str, Any]:
    path = root / "fit_complete.json"
    if not path.exists():
        audit.check("fit_complete_stop_marker_present", False, str(path))
        return {}
    marker = json.loads(path.read_text())
    checks = {
        "status": marker.get("status") == "stopped_before_model_fit",
        "weight": float(marker.get("pooled_oof_unique_effective_weight", math.nan))
        == 14167.0,
        "threshold": float(marker.get("pooled_oof_minimum_effective_weight", math.nan))
        == 20000.0,
        "support": marker.get("support_pass") is False,
        "later": marker.get("later_scoring_authorized") is False,
        "predictions": marker.get("predictions_generated") is False,
        "attribution": marker.get("source_attribution_permitted") is False,
        "safety": marker.get("research_only") is True
        and marker.get("live_ordering_enabled") is False
        and marker.get("order_placement") == "disabled",
    }
    audit.check("fit_complete_deterministic_support_stop_exact", all(checks.values()), checks)
    present_forbidden = [name for name in FORBIDDEN_EXECUTION_OUTPUTS if (root / name).exists()]
    audit.check(
        "no_model_prediction_gate_scoring_or_attribution_outputs_created",
        not present_forbidden,
        present_forbidden,
    )
    return marker


def verify_source_hashes(audit: Audit, root: Path) -> dict[str, Any]:
    path = root / "fit_source_hashes.json"
    if not path.exists():
        audit.check("fit_source_hashes_present", False)
        return {}
    manifest = json.loads(path.read_text())
    hashes = manifest.get("sources", manifest)
    mismatches: dict[str, Any] = {}
    aliases = {
        "per_loop_movement_quality_v1_contract_sha256": "per_loop_contract.json",
        "fixed_cycles_csv_sha256": "fixed_cycles.csv",
        "frozen_semimarkov_parameters_npz_sha256": "frozen_semimarkov_parameters.npz",
        "quality_thresholds_2024_json_sha256": "quality_thresholds_2024.json",
        "quality_feature_manifest_json_sha256": "quality_feature_manifest.json",
        "quality_fit_manifest_json_sha256": "quality_fit_manifest.json",
    }
    for name, (_, expected) in EXPECTED_PINNED_HASHES.items():
        value: Any = hashes.get(aliases[name])
        if isinstance(value, dict):
            value = value.get("sha256")
        if value != expected:
            mismatches[name] = {"expected": expected, "actual": value}
    audit.check("fit_source_hashes_include_all_contract_pins", not mismatches, mismatches)
    runner_hash = hashes.get("v2_stop_runner.py")
    audit.check(
        "fit_source_runner_hash_matches_current_runner",
        runner_hash == sha256(RUNNER),
        {"manifest": runner_hash, "current": sha256(RUNNER)},
    )
    return manifest


def prefit_audit() -> dict[str, Any]:
    audit = Audit()
    contract = json.loads(CONTRACT.read_text())
    verify_static_contract(audit, contract)
    contract_pins = contract["frozen_lineage"]["source_pins"]
    pin_results: dict[str, Any] = {}
    for name, (path, expected) in EXPECTED_PINNED_HASHES.items():
        actual = sha256(path)
        pin_results[name] = actual
        audit.check(
            f"pinned_source_{name}_exact",
            actual == expected and contract_pins.get(name) == expected,
            {"actual": actual, "contract": contract_pins.get(name)},
        )
    parent_fit = json.loads(PARENT_FIT_COMPLETE.read_text())
    audit.check(
        "parent_oof_artifact_hash_exact",
        sha256(PARENT_OOF) == PARENT_OOF_SHA256
        and parent_fit["artifact_hashes"]["oof_predictions_2024.parquet"]
        == PARENT_OOF_SHA256,
        sha256(PARENT_OOF),
    )
    mapping, _ = build_rotation_mapping()
    audit.check(
        "independent_uniform_rotation_topology_constructed",
        len(mapping) == 44
        and mapping["compatible_rotation_count"].sum() == 45
        and mapping.loc[
            mapping["compatible_rotation_count"].gt(1), "route_id"
        ].tolist()
        == ["cycle_15@state_1"],
        {"routes": len(mapping), "rotation_total": int(mapping["compatible_rotation_count"].sum())},
    )
    standardized, center, scale = normalized_centroids()
    audit.check(
        "centroid_population_normalization_exact",
        standardized.shape == (8, 14)
        and np.allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
        and np.allclose(
            standardized.std(axis=0, ddof=0)[scale > 0], 1.0, atol=1e-12
        ),
        {"center": center.tolist(), "scale": scale.tolist()},
    )
    support = independent_support_and_reference()
    audit.check(
        "sealed_qcontext_qfull_oof_reference_finite_nested_chain_exact",
        support["maximum_chain_rule_error"] <= 1e-12
        and support["maximum_probability_nesting_error"] <= 1e-12
        and support["maximum_probability_bound_violation"] <= 1e-12,
        support,
    )
    audit.check(
        "unique_oof_effective_weight_is_14167_and_below_20000",
        support["effective_conditional_weight"] == 14167.0
        and support["unique_positive_anchors"] == 14167
        and support["positive_rows"] == 15584
        and support["all_positive_anchor_weights_equal_one"]
        and support["support_pass"] is False,
        support,
    )
    parent_hashes = verify_parent_decisions(audit)
    return {
        "audit_phase": "pre_fit_support_and_contract",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "pinned_source_hashes": pin_results,
        "parent_decision_hashes": parent_hashes,
        "support_reconstruction": support,
        "rotation_units": len(mapping),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": False,
        "source_attribution_permitted": False,
    }


def postfit_audit(root: Path) -> dict[str, Any]:
    prefit = prefit_audit()
    audit = Audit(checks=list(prefit["checks"]))
    expected_mapping, vectors = build_rotation_mapping()
    rotation_result = verify_rotation_mapping(audit, root, expected_mapping)
    design_result = verify_design_rows(audit, root, vectors)
    runner_result = verify_runner_static_safety(audit)
    source_result = verify_source_hashes(audit, root)
    fit_marker = verify_fit_stop(audit, root)
    artifact_hash_errors: dict[str, Any] = {}
    for name, expected_hash in fit_marker.get("artifact_hashes", {}).items():
        path = root / name
        actual_hash = sha256(path) if path.exists() else "missing"
        if actual_hash != expected_hash:
            artifact_hash_errors[name] = {
                "expected": expected_hash,
                "actual": actual_hash,
            }
    audit.check(
        "fit_complete_artifact_hashes_exact",
        not artifact_hash_errors,
        artifact_hash_errors,
    )
    integrity_path = root / "parent_integrity_snapshot.json"
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text())
        audit.check(
            "parent_integrity_snapshot_before_after_exact_without_live_shadow_read",
            integrity.get("hashes_match") is True
            and integrity.get("parent_grade_changed") is False
            and integrity.get("live_shadow_tree_read") is False
            and integrity.get("live_shadow_tree_written") is False
            and integrity.get("parent_decision_and_saved_snapshot_hashes_before")
            == integrity.get("parent_decision_and_saved_snapshot_hashes_after")
            and integrity.get("parent_final_grade_counts") == {"unqualified": 20},
        )
    else:
        audit.check("parent_integrity_snapshot_present", False)
    planned_gate_path = root / "planned_gate_manifest.json"
    if planned_gate_path.exists():
        planned = json.loads(planned_gate_path.read_text())
        expected_seeds = [
            20260710 + comparison * 100 + surface * 10 + loss
            for comparison in range(5)
            for surface in range(2)
            for loss in range(2)
        ]
        actual_seeds = [int(row["seed"]) for row in planned.get("bootstrap_seed_mapping", [])]
        audit.check(
            "planned_five_comparison_multiplicity_manifest_exact_but_unexecuted",
            actual_seeds == expected_seeds
            and planned.get("gate_evaluation_performed") is False
            and planned.get("model_fit_performed") is False
            and planned.get("prediction_generated") is False
            and planned.get("status")
            == "planned_but_not_executed_due_prefit_support_stop",
            actual_seeds,
        )
    else:
        audit.check("planned_gate_manifest_present", False)
    stop_summary_path = root / "summary.json"
    if stop_summary_path.exists():
        stop_summary = json.loads(stop_summary_path.read_text())
        audit.check(
            "stop_summary_safety_parent_and_later_period_status_exact",
            stop_summary.get("status") == "support_stop_verified"
            and stop_summary.get("model_fit_performed") is False
            and stop_summary.get("prediction_generated") is False
            and stop_summary.get("later_period_panel_read") is False
            and stop_summary.get("source_attribution")
            == "not_permitted_due_frozen_support_stop"
            and stop_summary.get("parent_grade_changed") is False
            and stop_summary.get("live_shadow_tree_read") is False
            and stop_summary.get("live_shadow_tree_written") is False,
        )
    else:
        audit.check("support_stop_summary_present", False)
    feature_path = root / "topology_feature_manifest.json"
    if feature_path.exists():
        manifest = json.loads(feature_path.read_text())
        blocks = manifest.get("blocks", [])
        block_signature = [
            (
                block.get("name"),
                int(block.get("start", -1)),
                int(block.get("stop", -1)),
                float(block.get("feature_scale_if_later_fit", math.nan)),
            )
            for block in blocks
        ]
        expected_signature = [
            ("candidate_next_state_distribution", 0, 8, 1.0),
            ("future_route_state_composition", 8, 16, 1.0),
            ("transition_length_one_hot", 16, 19, 1.0),
            ("next_centroid_expectation", 19, 33, 0.5),
            ("route_centroid_expectation", 33, 47, 0.5),
            ("next_minus_current_centroid", 47, 61, 0.5),
            ("rotation_ambiguity", 61, 63, 1.0),
        ]
        audit.check(
            "artifact_feature_manifest_topology_width_blocks_and_scales_exact",
            manifest.get("topology_width") == 63
            and manifest.get("mapping_rows") == 44
            and block_signature == expected_signature,
            block_signature,
        )
        audit.check(
            "artifact_feature_manifest_no_future_stock_or_realized_rotation_feature",
            manifest.get("future_realized_feature_used") is False
            and manifest.get("stock_identity_feature_used") is False
            and manifest.get("rotation_aggregation")
            == "uniform over deduplicated compatible rotations beginning at current filtered state",
        )
    else:
        audit.check("topology_feature_manifest_present_for_postfit", False)
    # There can be no fold audit after a stop before model fitting. Causality is
    # therefore audited from the frozen schedule and the absence of fitted folds.
    audit.check(
        "fold_causality_preserved_by_pre_fit_stop",
        not (root / "fold_audit_2024.csv").exists()
        and tuple(json.loads(CONTRACT.read_text())["periods"]["causal_oof_months"])
        == OOF_MONTHS,
    )
    audit.check(
        "all_five_comparison_gates_and_multiplicity_not_executed_after_support_stop",
        all(not (root / name).exists() for name in (
            "paired_pooled_gates_2024.json",
            "comparison_summary_2024.csv",
            "cell_metrics_2024.csv",
            "rotation_diagnostics_2024.csv",
        ))
        and fit_marker.get("support_pass") is False,
    )
    audit.check(
        "no_source_attribution_or_later_period_promotion",
        not (root / "source_attribution.json").exists()
        and not (root / "period_transfer_gates.json").exists()
        and fit_marker.get("source_attribution_permitted") is False
        and fit_marker.get("later_scoring_authorized") is False,
    )
    result = {
        "audit_phase": "post_fit_deterministic_support_stop",
        "status": "support_stop_verified" if audit.all_passed else "audit_failed",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "support_reconstruction": prefit["support_reconstruction"],
        "rotation_reconstruction": rotation_result,
        "design_reconstruction": design_result,
        "runner_static_safety": runner_result,
        "fit_source_hashes": source_result,
        "fit_complete": fit_marker,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": False,
        "predictions_generated": False,
        "source_attribution_permitted": False,
        "later_periods_read": False,
        "parent_grades_unchanged": True if audit.all_passed else False,
        "live_shadows_read_by_audit": False,
        "interpretation": (
            "The V2 experiment stopped before model fit because the unique OOF "
            "effective conditional weight was 14,167, below the frozen 20,000 "
            "minimum. No feature source attribution is permitted."
        ),
    }
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pre-fit-only", action="store_true")
    parser.add_argument("--pre-score-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.pre_fit_only:
        result = prefit_audit()
    else:
        result = postfit_audit(arguments.root)
        output = (
            arguments.root / "pre_score_audit.json"
            if arguments.pre_score_only
            else arguments.root / "independent_artifact_audit.json"
        )
        write_result(output, result)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
