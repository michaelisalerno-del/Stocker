"""Deterministic pre-fit support stop for the frozen V2 feature ablation.

The frozen V2 contract requires at least 20,000 units of *unique* inverse-
overlap OOF conditional weight.  The July--December cohort contains only
14,167.  This runner reconstructs and audits that cohort plus the static route-
topology design, writes a sealed stop bundle, and exits before any model fit,
prediction, later-period panel read, or source attribution.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-loop-quality-feature-ablation-v2.json"
CONTRACT_SHA256 = "33d109a1bcc7ee58fb5ee65a5a5c1075a233baa07d50b1219db8358af22f4728"

PARENT_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
DEFAULT_OUTPUT_ROOT = Path(
    "/private/tmp/stocker_loop_quality_feature_ablation_v2_20260710"
)

PARENT_OOF = PARENT_ROOT / "oof_predictions_2024.parquet"
PARENT_TRAINING = PARENT_ROOT / "training_long_2024.parquet"
ANCHOR_PANEL = PRICE_ROOT / "anchor_panel_train_2024.parquet"
FIXED_CYCLES = PARENT_ROOT / "fixed_cycles.csv"
STATE_PARAMETERS = STATE_ROOT / "frozen_semimarkov_parameters.npz"

K = 8
CENTROID_WIDTH = 14
TOPOLOGY_WIDTH = 63
OOF_EFFECTIVE_WEIGHT_MINIMUM = 20000.0
CELL_COUNT = 12
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

NEXT_COLUMNS = tuple(f"next_state_p_{state}" for state in range(K))
COMPOSITION_COLUMNS = tuple(f"route_state_fraction_{state}" for state in range(K))
LENGTH_COLUMNS = ("length_is_2", "length_is_3", "length_is_4")
NEXT_CENTROID_COLUMNS = tuple(
    f"next_centroid_z_{index:02d}" for index in range(CENTROID_WIDTH)
)
ROUTE_CENTROID_COLUMNS = tuple(
    f"route_centroid_z_{index:02d}" for index in range(CENTROID_WIDTH)
)
DELTA_CENTROID_COLUMNS = tuple(
    f"next_minus_current_centroid_z_{index:02d}"
    for index in range(CENTROID_WIDTH)
)
AMBIGUITY_COLUMNS = (
    "rotation_count_minus_one",
    "next_state_entropy_normalized",
)
TOPOLOGY_COLUMNS = (
    *NEXT_COLUMNS,
    *COMPOSITION_COLUMNS,
    *LENGTH_COLUMNS,
    *NEXT_CENTROID_COLUMNS,
    *ROUTE_CENTROID_COLUMNS,
    *DELTA_CENTROID_COLUMNS,
    *AMBIGUITY_COLUMNS,
)

PINNED_INPUTS = {
    "per_loop_contract.json": HERE
    / "contracts/20260710-per-loop-movement-quality-v1.json",
    "v2_contract.json": CONTRACT,
    "fixed_cycles.csv": FIXED_CYCLES,
    "frozen_semimarkov_parameters.npz": STATE_PARAMETERS,
    "quality_thresholds_2024.json": PARENT_ROOT / "quality_thresholds_2024.json",
    "quality_feature_manifest.json": PARENT_ROOT / "feature_manifest.json",
    "quality_fit_manifest.json": PARENT_ROOT / "fit_manifest.json",
    "parent_oof_predictions_2024.parquet": PARENT_OOF,
    "parent_training_long_2024.parquet": PARENT_TRAINING,
    "anchor_panel_train_2024.parquet": ANCHOR_PANEL,
}

PARENT_DECISION_AND_SAVED_SNAPSHOT_FILES = {
    "parent_final_cycle_tiers.csv": PARENT_ROOT / "final_cycle_tiers.csv",
    "parent_provisional_tiers_2024.csv": PARENT_ROOT / "provisional_tiers_2024.csv",
    "parent_gates.json": PARENT_ROOT / "gates.json",
    "parent_summary.json": PARENT_ROOT / "summary.json",
    "saved_aggregate_shadow_snapshot_pre.json": PARENT_ROOT
    / "prospective_shadow_pre_content_snapshot.json",
    "saved_aggregate_shadow_snapshot_post.json": PARENT_ROOT
    / "prospective_shadow_post_content_snapshot.json",
    "saved_quality_shadow_contract.json": HERE
    / "contracts/20260710-frozen-loop-quality-shadow-v1.json",
    "saved_quality_shadow_manifest.json": HERE
    / "contracts/20260710-frozen-loop-quality-shadow-v1-manifest.json",
    "saved_quality_shadow_protected_snapshot.json": HERE
    / "contracts/20260710-frozen-loop-quality-shadow-v1-protected-snapshot.json",
}

EXPECTED_PIN_HASHES = {
    "per_loop_contract.json": "67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e",
    "v2_contract.json": CONTRACT_SHA256,
    "fixed_cycles.csv": "bf9292fa51de1e545e5a319fa2e2faf2088926acd5315b9106597b1da318b253",
    "frozen_semimarkov_parameters.npz": "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251",
    "quality_thresholds_2024.json": "f9e2355e36dae28e4279dfabe74645cb3a363b706d95d4179955093b80015b72",
    "quality_feature_manifest.json": "b3db72b43ad15f89ac8fedd182d2c5eee3931786dc600a339dc8714dad89ddd6",
    "quality_fit_manifest.json": "0a911d631fcc98445d60fa098a219313fabf3d88e78aa1137ab8b684c0e9ee58",
}


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def validate_output_root(output_root: Path) -> None:
    resolved = output_root.resolve()
    private_tmp = Path("/private/tmp").resolve()
    if (
        resolved.parent != private_tmp
        or not resolved.name.startswith(
            "stocker_loop_quality_feature_ablation_v2_"
        )
    ):
        raise ValueError("V2 stop output must be a dedicated /private/tmp directory")
    frozen_roots = (PARENT_ROOT.resolve(), STATE_ROOT.resolve(), PRICE_ROOT.resolve())
    if any(
        resolved == root or root in resolved.parents or resolved in root.parents
        for root in frozen_roots
    ):
        raise ValueError("V2 stop output overlaps a frozen input root")


def validate_contract_and_pins() -> dict[str, Any]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("frozen V2 contract hash changed")
    contract = json.loads(CONTRACT.read_text())
    if contract["research_only"] is not True:
        raise AssertionError("research-only label changed")
    if contract["live_ordering_enabled"] is not False:
        raise AssertionError("live ordering label changed")
    if contract["order_placement"] != "disabled":
        raise AssertionError("order placement label changed")
    if contract["models"]["qroute_topology"]["topology_width"] != TOPOLOGY_WIDTH:
        raise AssertionError("topology width changed")
    if (
        float(
            contract["calibration_bins_and_support"]
            ["pooled_oof_minimum_effective_conditional_weight"]
        )
        != OOF_EFFECTIVE_WEIGHT_MINIMUM
    ):
        raise AssertionError("OOF support threshold changed")
    actual = hashes(PINNED_INPUTS)
    for name, expected in EXPECTED_PIN_HASHES.items():
        if actual[name] != expected:
            raise AssertionError(f"pinned input hash changed: {name}")
    return {"contract": contract, "input_hashes": actual}


def cycle_core(cycle: str) -> tuple[int, ...]:
    states = tuple(int(value) for value in str(cycle).split("->"))
    if len(states) < 3 or states[0] != states[-1]:
        raise AssertionError(f"invalid closed cycle: {cycle}")
    return states[:-1]


def compatible_rotations(core: Iterable[int], current_state: int) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in core)
    routes = {
        values[index:] + values[:index] + (int(current_state),)
        for index, state in enumerate(values)
        if state == int(current_state)
    }
    return tuple(sorted(routes))


def standardized_centroids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(STATE_PARAMETERS) as parameters:
        means = parameters["means"].astype(float)
        semantic_new = parameters["semantic_new_state"].astype(int)
    if means.shape != (K, CENTROID_WIDTH):
        raise AssertionError("frozen centroid shape changed")
    if not np.array_equal(np.sort(semantic_new), np.arange(K)):
        raise AssertionError("semantic state index changed")
    center = means.mean(axis=0)
    scale = means.std(axis=0, ddof=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (means - center) / safe_scale
    standardized[:, scale == 0.0] = 0.0
    if not np.isfinite(standardized).all():
        raise AssertionError("non-finite standardized centroid")
    return standardized, center, safe_scale


def topology_vector(
    core: Iterable[int], current_state: int, centroids: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    routes = compatible_rotations(core, current_state)
    if not routes:
        raise AssertionError("cycle is incompatible with current state")
    length = len(routes[0]) - 1
    if length not in (2, 3, 4) or any(len(route) != length + 1 for route in routes):
        raise AssertionError("unsupported transition length")
    next_probability = np.zeros(K, dtype=float)
    composition = np.zeros(K, dtype=float)
    for route in routes:
        next_probability[route[1]] += 1.0 / len(routes)
        for state in route[1:]:
            composition[state] += 1.0 / (len(routes) * length)
    length_one_hot = np.asarray([length == value for value in (2, 3, 4)], float)
    next_centroid = next_probability @ centroids
    route_centroid = composition @ centroids
    delta = next_centroid - centroids[int(current_state)]
    positive = next_probability[next_probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(K)
    ambiguity = np.asarray([len(routes) - 1, entropy], dtype=float)
    vector = np.concatenate(
        (
            next_probability,
            composition,
            length_one_hot,
            next_centroid,
            route_centroid,
            delta,
            ambiguity,
        )
    )
    if len(vector) != TOPOLOGY_WIDTH or not np.isfinite(vector).all():
        raise AssertionError("invalid topology vector")
    if not math.isclose(float(next_probability.sum()), 1.0, abs_tol=1e-12):
        raise AssertionError("next-state distribution does not sum to one")
    if not math.isclose(float(composition.sum()), 1.0, abs_tol=1e-12):
        raise AssertionError("route composition does not sum to one")
    return vector, routes


def build_rotation_mapping(cycles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    centroids, centroid_center, centroid_scale = standardized_centroids()
    rows: list[dict[str, Any]] = []
    for _, cycle_row in cycles.sort_values("cycle_index").iterrows():
        core = cycle_core(cycle_row["cycle"])
        for current_state in sorted(set(core)):
            vector, routes = topology_vector(core, current_state, centroids)
            row: dict[str, Any] = {
                "cycle_index": int(cycle_row["cycle_index"]),
                "cycle_id": str(cycle_row["cycle_id"]),
                "cycle": str(cycle_row["cycle"]),
                "transition_length": int(cycle_row["transition_length"]),
                "current_state": int(current_state),
                "compatible_rotation_count": len(routes),
                "compatible_rotations": json.dumps(
                    ["->".join(map(str, route)) for route in routes],
                    separators=(",", ":"),
                ),
            }
            row.update(dict(zip(TOPOLOGY_COLUMNS, vector, strict=True)))
            rows.append(row)
    mapping = pd.DataFrame(rows)
    if mapping.duplicated(["cycle_id", "current_state"]).any():
        raise AssertionError("duplicate cycle-state topology mapping")
    metadata = {
        "centroid_column_center": centroid_center.tolist(),
        "centroid_column_scale": centroid_scale.tolist(),
        "centroid_expectations_in_design_rows": "raw standardized expectations; not multiplied by the declared 0.5 model feature scale",
    }
    return mapping, metadata


def load_and_validate_cycles() -> pd.DataFrame:
    cycles = pd.read_csv(FIXED_CYCLES)
    if len(cycles) != 20:
        raise AssertionError("frozen cycle count changed")
    if cycles["cycle_id"].nunique() != 20 or cycles["cycle_index"].nunique() != 20:
        raise AssertionError("cycle identifiers are not unique")
    for _, row in cycles.iterrows():
        if len(cycle_core(row["cycle"])) != int(row["transition_length"]):
            raise AssertionError("cycle transition length mismatch")
    return cycles


def merge_oof_design(mapping: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        f"quality_class__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
    ]
    joint_columns = [
        f"joint_{label}_target__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
        for label in ("good", "high")
    ]
    parent_columns = [
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
        *label_columns,
        *joint_columns,
    ]
    oof = pd.read_parquet(PARENT_OOF, columns=parent_columns).reset_index(drop=True)
    oof.insert(0, "parent_oof_row", np.arange(len(oof), dtype=np.int64))
    anchors = pd.read_parquet(
        ANCHOR_PANEL,
        columns=["anchor_id", "state", "history_token", *NUMERIC_CONTROLS],
    )
    if anchors["anchor_id"].duplicated().any():
        raise AssertionError("anchor controls are not unique")
    design = oof.merge(
        anchors,
        on="anchor_id",
        how="left",
        suffixes=("", "__anchor"),
        validate="many_to_one",
        sort=False,
    )
    if len(design) != len(oof) or design[list(NUMERIC_CONTROLS)].isna().any().any():
        raise AssertionError("OOF control merge failed")
    if not design["state"].equals(design["state__anchor"]):
        raise AssertionError("OOF state disagrees with frozen anchor")
    if not design["history_token"].equals(design["history_token__anchor"]):
        raise AssertionError("OOF history token disagrees with frozen anchor")
    design = design.drop(columns=["state__anchor", "history_token__anchor"])
    topology_columns = [
        "cycle_id",
        "current_state",
        "compatible_rotation_count",
        "compatible_rotations",
        *TOPOLOGY_COLUMNS,
    ]
    design = design.merge(
        mapping[topology_columns],
        left_on=["cycle_id", "state"],
        right_on=["cycle_id", "current_state"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(design) != len(oof) or design[list(TOPOLOGY_COLUMNS)].isna().any().any():
        raise AssertionError("OOF topology merge failed")
    design = design.sort_values("parent_oof_row", kind="stable").reset_index(drop=True)
    if not np.array_equal(design["parent_oof_row"], np.arange(len(design))):
        raise AssertionError("OOF row order changed")
    return design


def support_audit(design: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    realised = design.loc[design["loop_occurs"].eq(1)]
    unique_weight = float(realised["conditional_weight"].sum())
    repeated_weight = unique_weight * CELL_COUNT
    support_pass = bool(unique_weight >= OOF_EFFECTIVE_WEIGHT_MINIMUM)
    rows = [
        {
            "measure": "compatible_anchor_cycle_rows",
            "observed": float(len(design)),
            "threshold": math.nan,
            "pass": True,
            "counting_semantics": "each frozen OOF anchor-cycle row once",
        },
        {
            "measure": "realised_anchor_cycle_rows",
            "observed": float(len(realised)),
            "threshold": math.nan,
            "pass": True,
            "counting_semantics": "each realized frozen OOF anchor-cycle row once",
        },
        {
            "measure": "unique_inverse_overlap_effective_weight",
            "observed": unique_weight,
            "threshold": OOF_EFFECTIVE_WEIGHT_MINIMUM,
            "pass": support_pass,
            "counting_semantics": "sum conditional_weight once per unique realized anchor-cycle row",
        },
        {
            "measure": "twelve_cell_repeated_weight_diagnostic_only",
            "observed": repeated_weight,
            "threshold": OOF_EFFECTIVE_WEIGHT_MINIMUM,
            "pass": False,
            "counting_semantics": "rejected for support because it repeats the same cohort across twelve outcome cells",
        },
    ]
    payload = {
        "period": "2024_oof",
        "compatible_anchor_cycle_rows": len(design),
        "realised_anchor_cycle_rows": len(realised),
        "unique_realised_anchors": int(realised["anchor_id"].nunique()),
        "unique_symbols": int(realised["symbol_norm"].nunique()),
        "quarters": sorted(realised["quarter"].astype(str).unique()),
        "pooled_oof_unique_effective_weight": unique_weight,
        "pooled_oof_minimum_effective_weight": OOF_EFFECTIVE_WEIGHT_MINIMUM,
        "support_pass": support_pass,
        "twelve_cell_repeated_weight": repeated_weight,
        "double_counted_cell_weight_accepted": False,
        "interpretation": "Unique inverse-overlap cohort information is the support unit; repeating outcomes does not create new observations.",
    }
    return pd.DataFrame(rows), payload


def topology_manifest(metadata: dict[str, Any], mapping: pd.DataFrame) -> dict[str, Any]:
    blocks = [
        {"name": "candidate_next_state_distribution", "start": 0, "stop": 8, "feature_scale_if_later_fit": 1.0},
        {"name": "future_route_state_composition", "start": 8, "stop": 16, "feature_scale_if_later_fit": 1.0},
        {"name": "transition_length_one_hot", "start": 16, "stop": 19, "feature_scale_if_later_fit": 1.0},
        {"name": "next_centroid_expectation", "start": 19, "stop": 33, "feature_scale_if_later_fit": 0.5},
        {"name": "route_centroid_expectation", "start": 33, "stop": 47, "feature_scale_if_later_fit": 0.5},
        {"name": "next_minus_current_centroid", "start": 47, "stop": 61, "feature_scale_if_later_fit": 0.5},
        {"name": "rotation_ambiguity", "start": 61, "stop": 63, "feature_scale_if_later_fit": 1.0},
    ]
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "status": "static_topology_reconstructed_before_support_stop",
        "topology_width": TOPOLOGY_WIDTH,
        "topology_columns": list(TOPOLOGY_COLUMNS),
        "blocks": blocks,
        "mapping_rows": len(mapping),
        "rotation_aggregation": "uniform over deduplicated compatible rotations beginning at current filtered state",
        "future_realized_feature_used": False,
        "stock_identity_feature_used": False,
        "centroid_source_sha256": EXPECTED_PIN_HASHES[
            "frozen_semimarkov_parameters.npz"
        ],
        **metadata,
    }


def planned_gate_manifest() -> dict[str, Any]:
    comparisons = [
        "qroute_topology_vs_qcontext",
        "qroute_topology_noninferiority_vs_qfull",
        "qcycle_main_vs_qroute_topology",
        "qcycle_state_vs_qcycle_main",
        "qfull_vs_qcycle_state",
    ]
    seed_rows = []
    for comparison_index, comparison in enumerate(comparisons):
        for surface_index, surface in enumerate(("conditional", "joint")):
            for loss_index, loss in enumerate(("log_loss", "brier")):
                seed_rows.append(
                    {
                        "comparison": comparison,
                        "surface": surface,
                        "loss": loss,
                        "seed": 20260710
                        + comparison_index * 100
                        + surface_index * 10
                        + loss_index,
                    }
                )
    return {
        "status": "planned_but_not_executed_due_prefit_support_stop",
        "bootstrap_seed_mapping": seed_rows,
        "model_fit_performed": False,
        "prediction_generated": False,
        "gate_evaluation_performed": False,
    }


def self_tests() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    periodic = compatible_rotations((0, 1, 0, 1), 0)
    checks["periodic_rotations_deduplicate"] = periodic == ((0, 1, 0, 1, 0),)
    ambiguous = compatible_rotations((1, 2, 1, 3), 1)
    checks["ambiguous_rotations_retained"] = len(ambiguous) == 2
    synthetic = np.arange(K * CENTROID_WIDTH, dtype=float).reshape(K, CENTROID_WIDTH)
    vector, _ = topology_vector((1, 2, 1, 3), 1, synthetic)
    checks["topology_width"] = len(vector) == TOPOLOGY_WIDTH
    checks["next_distribution_normalized"] = math.isclose(vector[:8].sum(), 1.0)
    checks["composition_normalized"] = math.isclose(vector[8:16].sum(), 1.0)
    checks["twelve_cell_repetition_rejected"] = (
        14167.0 < OOF_EFFECTIVE_WEIGHT_MINIMUM
        and 14167.0 * CELL_COUNT > OOF_EFFECTIVE_WEIGHT_MINIMUM
    )
    checks["no_forbidden_topology_column_name"] = not any(
        token in column.lower()
        for column in TOPOLOGY_COLUMNS
        for token in ("future_realized", "outcome", "price", "stock")
    )
    if not all(checks.values()):
        raise AssertionError(f"V2 stop self-test failed: {checks}")
    return {"checks": checks, "passed": len(checks), "total": len(checks)}


def output_hashes(output_root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256(output_root / name) for name in names}


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    validate_output_root(output_root)
    validation = validate_contract_and_pins()
    parent_before = hashes(PARENT_DECISION_AND_SAVED_SNAPSHOT_FILES)
    source_before = hashes(PINNED_INPUTS)
    source_before["v2_stop_runner.py"] = sha256(Path(__file__))

    parent_tiers = pd.read_csv(PARENT_ROOT / "final_cycle_tiers.csv")
    if len(parent_tiers) != 20 or not parent_tiers["final_grade"].eq("unqualified").all():
        raise AssertionError("frozen parent grade decision changed")
    aggregate_pre = json.loads(
        (PARENT_ROOT / "prospective_shadow_pre_content_snapshot.json").read_text()
    )
    aggregate_post = json.loads(
        (PARENT_ROOT / "prospective_shadow_post_content_snapshot.json").read_text()
    )
    if aggregate_pre != aggregate_post:
        raise AssertionError("saved aggregate shadow snapshots disagree")
    if aggregate_post.get("runtime_outcomes_opened") is not False:
        raise AssertionError("saved aggregate shadow outcome flag is not false")
    if aggregate_post.get("ledger_lines") != 0:
        raise AssertionError("saved aggregate shadow ledger was not empty")

    tests = self_tests()
    cycles = load_and_validate_cycles()
    mapping, centroid_metadata = build_rotation_mapping(cycles)
    design = merge_oof_design(mapping)
    support_table, support = support_audit(design)
    if support["support_pass"]:
        raise AssertionError("frozen deterministic support stop unexpectedly passed")

    output_root.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_root / "rotation_mapping.csv", index=False)
    design.to_parquet(output_root / "oof_design_rows_2024.parquet", index=False)
    support_table.to_csv(output_root / "support_audit.csv", index=False)
    write_json(output_root / "support_audit.json", support)
    write_json(
        output_root / "topology_feature_manifest.json",
        topology_manifest(centroid_metadata, mapping),
    )
    write_json(output_root / "planned_gate_manifest.json", planned_gate_manifest())
    write_json(output_root / "self_tests.json", tests)

    parent_after = hashes(PARENT_DECISION_AND_SAVED_SNAPSHOT_FILES)
    input_after = hashes(PINNED_INPUTS)
    if parent_before != parent_after:
        raise AssertionError("parent decision or saved snapshot changed")
    for name, value in input_after.items():
        if source_before[name] != value:
            raise AssertionError(f"frozen input changed during V2 stop: {name}")

    parent_snapshot = {
        "parent_decision_and_saved_snapshot_hashes_before": parent_before,
        "parent_decision_and_saved_snapshot_hashes_after": parent_after,
        "hashes_match": parent_before == parent_after,
        "parent_final_grade_counts": parent_tiers["final_grade"].value_counts().to_dict(),
        "parent_grade_changed": False,
        "live_shadow_tree_read": False,
        "live_shadow_tree_written": False,
        "saved_aggregate_shadow_tree_sha256": aggregate_post["tree_sha256"],
        "saved_aggregate_shadow_ledger_lines": aggregate_post["ledger_lines"],
        "saved_aggregate_shadow_outcomes_opened": aggregate_post[
            "runtime_outcomes_opened"
        ],
    }
    write_json(output_root / "parent_integrity_snapshot.json", parent_snapshot)
    write_json(output_root / "fit_source_hashes.json", source_before)

    stop_reason = {
        "status": "stopped_before_model_fit",
        "stop_stage": "prefit_unique_oof_support",
        "stop_reason": "pooled_oof_unique_effective_weight_below_frozen_minimum",
        **support,
        "model_fit_performed": False,
        "prediction_generated": False,
        "source_attribution_performed": False,
        "later_period_panel_read": False,
        "later_scoring_authorized": False,
        "contract_change_required_to_continue": True,
        "required_next_contract": "separately frozen V3 using unique-cohort support semantics",
    }
    write_json(output_root / "stop_reason.json", stop_reason)

    preliminary_names = (
        "rotation_mapping.csv",
        "oof_design_rows_2024.parquet",
        "support_audit.csv",
        "support_audit.json",
        "topology_feature_manifest.json",
        "planned_gate_manifest.json",
        "self_tests.json",
        "parent_integrity_snapshot.json",
        "fit_source_hashes.json",
        "stop_reason.json",
    )
    fit_complete = {
        "status": "stopped_before_model_fit",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "pooled_oof_unique_effective_weight": support[
            "pooled_oof_unique_effective_weight"
        ],
        "pooled_oof_minimum_effective_weight": OOF_EFFECTIVE_WEIGHT_MINIMUM,
        "support_pass": False,
        "later_scoring_authorized": False,
        "predictions_generated": False,
        "model_fit_performed": False,
        "source_attribution_permitted": False,
        "parent_grade_changed": False,
        "live_shadow_tree_read": False,
        "live_shadow_tree_written": False,
        "artifact_hashes": output_hashes(output_root, preliminary_names),
    }
    write_json(output_root / "fit_complete.json", fit_complete)

    summary = {
        "experiment": "loop_quality_feature_ablation_v2",
        "status": "support_stop_verified",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "support": support,
        "model_fit_performed": False,
        "prediction_generated": False,
        "later_period_panel_read": False,
        "source_attribution": "not_permitted_due_frozen_support_stop",
        "frozen_parent_grades": {"unqualified": 20},
        "parent_grade_changed": False,
        "live_shadow_tree_read": False,
        "live_shadow_tree_written": False,
        "next_step": "Any support correction requires a separately frozen V3 before fitting; the V2 contract remains byte-identical.",
        "output_root": str(output_root),
        "fit_complete_sha256": sha256(output_root / "fit_complete.json"),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.output_root)
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
