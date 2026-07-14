#!/usr/bin/env python3
"""Post-outcome diagnostic of frozen movement quality by oriented cycle route.

This module does not fit or recalibrate a model.  It partitions the already
frozen cycle-quality predictions by ``(cycle_id, current_state)`` and applies
the parent experiment's support, proper-loss, calibration, robustness, lift,
and structural gates as closely as possible.  Its labels are exploratory and
cannot enter either prospective shadow.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from work.run_per_loop_movement_quality import (  # noqa: E402
    HORIZONS,
    SEED,
    TARGETS,
    TIERS,
    _base_support_gate,
    _quality_cell_gate,
    _structural_gate,
    safe,
)


CONTRACT = WORKSPACE / "work/contracts/20260710-oriented-route-movement-diagnostic-v1.json"
PARENT_CONTRACT = WORKSPACE / "work/contracts/20260710-per-loop-movement-quality-v1.json"
PARENT_RUNNER = WORKSPACE / "work/run_per_loop_movement_quality.py"
QUALITY_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
DEFAULT_OUT = Path("/private/tmp/stocker_oriented_route_movement_diagnostic_20260710")

INPUTS = {
    "2024_oof": (QUALITY_ROOT / "oof_predictions_2024.parquet", "oof"),
    "2025": (QUALITY_ROOT / "quality_scoring_2025.parquet", "scoring"),
    "2023": (QUALITY_ROOT / "quality_scoring_2023.parquet", "scoring"),
}

SHADOWS = {
    "aggregate_movement": WORKSPACE
    / "work/shadow_validation/frozen_loop_movement_shadow_v1",
    "per_loop_quality": WORKSPACE
    / "work/shadow_validation/frozen_loop_quality_shadow_v1",
}

LABELS = {
    "high": "diagnostic_high_candidate",
    "good": "diagnostic_good_candidate",
    "failed": "diagnostic_unqualified",
    "unsupported": "diagnostic_not_supported",
}
GRADE_RANK = {
    LABELS["unsupported"]: 0,
    LABELS["failed"]: 1,
    LABELS["good"]: 2,
    LABELS["high"]: 3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def deduplicated_oriented_paths(
    core: tuple[int, ...], current_state: int
) -> list[tuple[int, ...]]:
    """Return exact closed rotations compatible with ``current_state``."""

    return sorted(
        {
            core[index:] + core[:index] + (int(current_state),)
            for index, state in enumerate(core)
            if int(state) == int(current_state)
        }
    )


def minimum_grade(values: Iterable[str]) -> str:
    grades = list(values)
    if not grades:
        raise ValueError("cannot grade an empty collection")
    unknown = sorted(set(grades).difference(GRADE_RANK))
    if unknown:
        raise ValueError(f"unknown diagnostic grade(s): {unknown}")
    return min(grades, key=lambda value: GRADE_RANK[value])


def _state_centroids() -> tuple[np.ndarray, np.ndarray]:
    features = pd.read_csv(STATE_ROOT / "frozen_emission_preprocessing.csv")[
        "feature"
    ].astype(str).tolist()
    parameters = np.load(STATE_ROOT / "frozen_semimarkov_parameters.npz")
    means = parameters["means"]
    range_index = features.index("regime_log_bar_range")
    activity_index = features.index("regime_log_activity_12")
    return means[:, range_index], means[:, activity_index]


def build_route_manifest(cycles: pd.DataFrame) -> pd.DataFrame:
    range_centroid, activity_centroid = _state_centroids()
    rows: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        closed = tuple(int(value) for value in str(cycle.cycle).split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid frozen cycle {cycle.cycle}")
        core = closed[:-1]
        unique_states = sorted(set(core))
        for current_state in unique_states:
            paths = deduplicated_oriented_paths(core, current_state)
            if not paths:
                raise AssertionError("route construction lost the current state")
            path_strings = ["->".join(str(value) for value in path) for path in paths]
            other_states = sorted(set(core).difference({current_state}))
            destination_peak = max(
                (float(range_centroid[state]) for state in other_states),
                default=float(range_centroid[current_state]),
            )
            route_kind = "exact_route" if len(paths) == 1 else "ambiguous_union"
            rows.append(
                {
                    "route_id": f"{cycle.cycle_id}@state_{current_state}",
                    "cycle_id": str(cycle.cycle_id),
                    "current_state": int(current_state),
                    "cycle": str(cycle.cycle),
                    "transition_length": int(cycle.transition_length),
                    "route_kind": route_kind,
                    "oriented_route": " || ".join(path_strings),
                    "oriented_route_count": len(paths),
                    "current_state_log_bar_range_centroid": float(
                        range_centroid[current_state]
                    ),
                    "current_state_log_activity_12_centroid": float(
                        activity_centroid[current_state]
                    ),
                    "maximum_cycle_log_bar_range_centroid": float(
                        max(range_centroid[list(set(core))])
                    ),
                    "upward_excursion_range_contrast": float(
                        destination_peak - range_centroid[current_state]
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["cycle_id", "current_state"], kind="stable"
    )
    if result.duplicated(["cycle_id", "current_state"]).any():
        raise AssertionError("duplicate oriented route")
    if len(result) != 44:
        raise AssertionError(f"expected 44 oriented route units, found {len(result)}")
    ambiguous = result.loc[result["route_kind"].eq("ambiguous_union"), "route_id"]
    if ambiguous.tolist() != ["cycle_15@state_1"]:
        raise AssertionError(f"unexpected ambiguous routes: {ambiguous.tolist()}")
    return result.reset_index(drop=True)


def snapshot_shadow(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        if item.is_file():
            rows.append(
                {
                    "path": str(item.relative_to(path)),
                    "size": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    runtime = json.loads((path / "runtime_metadata.json").read_text())
    ledger = path / "prediction_ledger.jsonl"
    result = {
        "path": str(path.relative_to(WORKSPACE)),
        "files": rows,
        "tree_sha256": canonical_hash(rows),
        "file_count": len(rows),
        "ledger_size": ledger.stat().st_size,
        "ledger_lines": len(ledger.read_text().splitlines()),
        "ledger_sha256": sha256(ledger),
        "outcomes_opened": runtime.get("outcomes_opened"),
        "research_only": runtime.get("research_only"),
        "live_ordering_enabled": runtime.get("live_ordering_enabled"),
        "order_placement": runtime.get("order_placement"),
    }
    if result["ledger_size"] != 0 or result["ledger_lines"] != 0:
        raise AssertionError(f"shadow ledger is not empty: {path}")
    if result["outcomes_opened"] is not False:
        raise AssertionError(f"shadow outcomes were opened: {path}")
    if result["research_only"] is not True:
        raise AssertionError(f"shadow research label changed: {path}")
    if result["live_ordering_enabled"] is not False:
        raise AssertionError(f"shadow live label changed: {path}")
    if result["order_placement"] != "disabled":
        raise AssertionError(f"shadow order label changed: {path}")
    return result


def source_manifest() -> dict[str, Any]:
    sources = {
        "diagnostic_contract": CONTRACT,
        "diagnostic_runner": Path(__file__),
        "parent_contract": PARENT_CONTRACT,
        "parent_runner": PARENT_RUNNER,
        "fixed_cycles": QUALITY_ROOT / "fixed_cycles.csv",
        "oof_predictions_2024": INPUTS["2024_oof"][0],
        "quality_scoring_2025": INPUTS["2025"][0],
        "quality_scoring_2023": INPUTS["2023"][0],
        "state_parameters": STATE_ROOT / "frozen_semimarkov_parameters.npz",
        "state_feature_manifest": STATE_ROOT / "frozen_emission_preprocessing.csv",
    }
    rows = {
        name: {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
        for name, path in sources.items()
    }
    return {"sources": rows, "manifest_sha256": canonical_hash(rows)}


def required_columns() -> list[str]:
    columns = [
        "cycle_id",
        "state",
        "loop_occurs",
        "conditional_weight",
        "symbol_norm",
        "session_date",
        "quarter",
        "loop_probability",
        "first_order_probability",
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
            for model in ("qcontext", "qcycle"):
                for tier in TIERS:
                    columns.extend(
                        [
                            f"{model}__{target}__h{horizon}__{tier}",
                            f"joint__{model}__{target}__h{horizon}__{tier}",
                        ]
                    )
    return sorted(set(columns))


def validate_frame(frame: pd.DataFrame, period: str) -> None:
    years = pd.to_datetime(frame["session_date"], errors="raise").dt.year.unique()
    expected_year = 2024 if period == "2024_oof" else int(period)
    if set(years) != {expected_year} or expected_year >= 2026:
        raise AssertionError(f"{period} year boundary failure")
    if not frame["state"].between(0, 7).all():
        raise AssertionError(f"{period} state outside frozen range")
    if not frame["loop_occurs"].isin([0, 1]).all():
        raise AssertionError(f"{period} invalid loop labels")
    probability_columns = [
        column
        for column in frame.columns
        if "probability" in column or column.endswith(("__p75", "__p90"))
    ]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise AssertionError(f"{period} contains non-finite probabilities")
    if probabilities.min(initial=0.0) < -1e-12 or probabilities.max(initial=0.0) > 1 + 1e-12:
        raise AssertionError(f"{period} probability outside [0,1]")
    for target in TARGETS:
        for horizon in HORIZONS:
            quality = frame[f"quality_class__{target}__h{horizon}"]
            if not quality.isin([0, 1, 2]).all():
                raise AssertionError("invalid ordered quality class")
            for model in ("qcontext", "qcycle"):
                p75 = frame[f"{model}__{target}__h{horizon}__p75"].to_numpy(float)
                p90 = frame[f"{model}__{target}__h{horizon}__p90"].to_numpy(float)
                if np.any(p90 > p75 + 1e-12):
                    raise AssertionError("ordered probability nesting failure")
                structural = frame["loop_probability"].to_numpy(float)
                for tier, probability in (("p75", p75), ("p90", p90)):
                    joint = frame[
                        f"joint__{model}__{target}__h{horizon}__{tier}"
                    ].to_numpy(float)
                    if not np.allclose(joint, structural * probability, atol=1e-12):
                        raise AssertionError("joint chain-rule identity failure")


def _auc(observed: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(observed)) != 2:
        return math.nan
    return float(roc_auc_score(observed, probability))


def evaluate_period(
    frame: pd.DataFrame,
    period: str,
    mode: str,
    manifest: pd.DataFrame,
    parent_contract: dict[str, Any],
    full_fit_eligible: dict[str, bool],
) -> dict[str, pd.DataFrame]:
    support_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    if mode == "oof":
        required_quarters = ["2024_q3", "2024_q4"]
        structural_bin = int(
            parent_contract["common_quality_gates"]["calibration"][
                "oof_minimum_supported_joint_bin_rows"
            ]
        )
    else:
        required_quarters = [f"{period}_q{quarter}" for quarter in range(1, 5)]
        structural_bin = int(
            parent_contract["common_quality_gates"]["calibration"][
                "scoring_minimum_supported_joint_bin_rows"
            ]
        )
    tolerance = float(
        parent_contract["structural_reliability_gate_each_cycle_and_scoring_period"]
        ["maximum_supported_bin_error_tolerance"]
    )
    route_lookup = manifest.set_index(["cycle_id", "current_state"])

    for route_index, ((cycle_id, current_state), route_frame) in enumerate(
        frame.groupby(["cycle_id", "state"], sort=True)
    ):
        route_frame = route_frame.reset_index(drop=True)
        descriptor = route_lookup.loc[(str(cycle_id), int(current_state))]
        route_id = str(descriptor["route_id"])
        support = _base_support_gate(route_frame, mode, parent_contract)
        fit_eligible = bool(full_fit_eligible[str(cycle_id)])
        combined_support = bool(support["pass"] and fit_eligible)
        support_rows.append(
            {
                "period": period,
                "route_id": route_id,
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "route_kind": descriptor["route_kind"],
                "oriented_route": descriptor["oriented_route"],
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
                "full_2024_cycle_fit_eligible": fit_eligible,
                "combined_support_pass": combined_support,
            }
        )
        structural = _structural_gate(route_frame, structural_bin, tolerance)
        structural_observed = route_frame["loop_occurs"].to_numpy(dtype=int)
        structural_rows.append(
            {
                "period": period,
                "route_id": route_id,
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "compatible_rows": len(route_frame),
                "realised_rows": int(structural_observed.sum()),
                "occurrence_rate": float(structural_observed.mean()),
                "history_auc": _auc(
                    structural_observed,
                    route_frame["loop_probability"].to_numpy(dtype=float),
                ),
                "first_order_auc": _auc(
                    structural_observed,
                    route_frame["first_order_probability"].to_numpy(dtype=float),
                ),
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )

        if not combined_support:
            for horizon in HORIZONS:
                horizon_rows.append(
                    {
                        "period": period,
                        "route_id": route_id,
                        "cycle_id": cycle_id,
                        "current_state": int(current_state),
                        "horizon": horizon,
                        "grade": LABELS["unsupported"],
                        "support_pass": False,
                        "structural_pass": bool(structural["pass"]),
                        "structural_required_for_grade": mode != "oof",
                        "both_targets_good_pass": False,
                        "both_targets_high_p75_rate_pass": False,
                        "both_targets_high_p90_pass": False,
                    }
                )
            period_rows.append(
                {
                    "period": period,
                    "route_id": route_id,
                    "cycle_id": cycle_id,
                    "current_state": int(current_state),
                    "h6_grade": LABELS["unsupported"],
                    "h12_grade": LABELS["unsupported"],
                    "h24_grade": LABELS["unsupported"],
                    "global_grade": LABELS["unsupported"],
                    "prospective_validated": False,
                    "promotion_or_surface_permission": False,
                }
            )
            continue

        cell_results: dict[tuple[str, int, str], dict[str, Any]] = {}
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(TIERS):
                    result = _quality_cell_gate(
                        route_frame,
                        target,
                        horizon,
                        tier,
                        mode,
                        required_quarters,
                        parent_contract,
                        SEED
                        + (0 if mode == "oof" else 50000)
                        + route_index * 1000
                        + target_index * 200
                        + horizon * 5
                        + tier_index,
                    )
                    cell_results[(target, horizon, tier)] = result
                    cell_rows.append(
                        {
                            "period": period,
                            "route_id": route_id,
                            "cycle_id": cycle_id,
                            "current_state": int(current_state),
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "pass": bool(result["pass"]),
                            "positive_rows": result["positive_rows"],
                            "negative_rows": result["negative_rows"],
                            "observed_rate": result["observed_rate"],
                            "mean_qcontext_probability": result[
                                "mean_qcontext_probability"
                            ],
                            "mean_qcycle_probability": result[
                                "mean_qcycle_probability"
                            ],
                            "qcycle_probability_divided_by_qcontext": (
                                result["mean_qcycle_probability"]
                                / result["mean_qcontext_probability"]
                                if result["mean_qcontext_probability"] > 0
                                else math.nan
                            ),
                            "observed_rate_divided_by_mean_qcontext": result[
                                "observed_rate_divided_by_mean_qcontext"
                            ],
                            "daily_residual_ci_low": result["daily_residual_ci_low"],
                            "conditional_relative_log_loss_improvement": result[
                                "conditional_gate"
                            ]["relative_log_loss_improvement"],
                            "joint_relative_log_loss_improvement": result["joint_gate"]
                            ["relative_log_loss_improvement"],
                            "gate_detail": json.dumps(safe(result), sort_keys=True),
                        }
                    )

        horizon_grades: list[str] = []
        for horizon in HORIZONS:
            good_cells = [
                bool(cell_results[(target, horizon, "p75")]["pass"])
                for target in TARGETS
            ]
            high_p90_cells = [
                bool(cell_results[(target, horizon, "p90")]["pass"])
                for target in TARGETS
            ]
            high_rule = parent_contract["tier_rules_each_cycle_and_horizon"]["high"]
            high_p75_rate = all(
                cell_results[(target, horizon, "p75")]["observed_rate"]
                >= float(high_rule["minimum_p75_observed_conditional_exceedance_rate"])
                and cell_results[(target, horizon, "p75")][
                    "mean_qcycle_probability"
                ]
                >= float(
                    high_rule["minimum_p75_mean_calibrated_qcycle_probability"]
                )
                for target in TARGETS
            )
            structural_required = True if mode == "oof" else bool(structural["pass"])
            common_required = bool(combined_support and structural_required)
            good_pass = bool(common_required and all(good_cells))
            high_pass = bool(good_pass and high_p75_rate and all(high_p90_cells))
            grade = (
                LABELS["high"]
                if high_pass
                else LABELS["good"]
                if good_pass
                else LABELS["failed"]
            )
            horizon_grades.append(grade)
            horizon_rows.append(
                {
                    "period": period,
                    "route_id": route_id,
                    "cycle_id": cycle_id,
                    "current_state": int(current_state),
                    "horizon": horizon,
                    "grade": grade,
                    "support_pass": combined_support,
                    "structural_pass": bool(structural["pass"]),
                    "structural_required_for_grade": mode != "oof",
                    "both_targets_good_pass": bool(all(good_cells)),
                    "both_targets_high_p75_rate_pass": bool(high_p75_rate),
                    "both_targets_high_p90_pass": bool(all(high_p90_cells)),
                }
            )
        if all(grade == LABELS["high"] for grade in horizon_grades):
            global_grade = LABELS["high"]
        elif all(GRADE_RANK[grade] >= GRADE_RANK[LABELS["good"]] for grade in horizon_grades):
            global_grade = LABELS["good"]
        else:
            global_grade = LABELS["failed"]
        period_rows.append(
            {
                "period": period,
                "route_id": route_id,
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "h6_grade": horizon_grades[0],
                "h12_grade": horizon_grades[1],
                "h24_grade": horizon_grades[2],
                "global_grade": global_grade,
                "prospective_validated": False,
                "promotion_or_surface_permission": False,
            }
        )

    return {
        "support": pd.DataFrame(support_rows),
        "structural": pd.DataFrame(structural_rows),
        "cells": pd.DataFrame(cell_rows),
        "horizons": pd.DataFrame(horizon_rows),
        "periods": pd.DataFrame(period_rows),
    }


def cross_period_grades(
    period_grades: pd.DataFrame, horizon_grades: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_rows: list[dict[str, Any]] = []
    for (route_id, cycle_id, current_state, horizon), group in horizon_grades.groupby(
        ["route_id", "cycle_id", "current_state", "horizon"], sort=True
    ):
        by_period = group.set_index("period")["grade"].to_dict()
        grade = minimum_grade(by_period.values())
        horizon_rows.append(
            {
                "route_id": route_id,
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "horizon": int(horizon),
                "grade_2024_oof": by_period["2024_oof"],
                "grade_2025": by_period["2025"],
                "grade_2023": by_period["2023"],
                "cross_period_grade": grade,
                "prospective_validated": False,
                "promotion_or_surface_permission": False,
            }
        )
    horizon_result = pd.DataFrame(horizon_rows)

    route_rows: list[dict[str, Any]] = []
    for (route_id, cycle_id, current_state), group in period_grades.groupby(
        ["route_id", "cycle_id", "current_state"], sort=True
    ):
        by_period = group.set_index("period")["global_grade"].to_dict()
        grade = minimum_grade(by_period.values())
        route_rows.append(
            {
                "route_id": route_id,
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "grade_2024_oof": by_period["2024_oof"],
                "grade_2025": by_period["2025"],
                "grade_2023": by_period["2023"],
                "cross_period_global_grade": grade,
                "hypothesis_only": True,
                "prospective_validated": False,
                "promotion_or_surface_permission": False,
            }
        )
    return horizon_result, pd.DataFrame(route_rows)


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    parent_contract = json.loads(PARENT_CONTRACT.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["post_outcome_route_selection"] is True
        and contract["promotion_or_surface_permission"] is False
    ):
        raise AssertionError("diagnostic safety contract failure")

    before = {name: snapshot_shadow(path) for name, path in SHADOWS.items()}
    write_json(output / "protected_shadows_before.json", before)
    sources = source_manifest()
    write_json(output / "source_hashes.json", sources)

    cycles = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv")
    manifest = build_route_manifest(cycles)
    manifest.to_csv(output / "route_manifest.csv", index=False)
    eligibility = pd.read_csv(QUALITY_ROOT / "provisional_support_2024.csv").set_index(
        "cycle_id"
    )["full_2024_fit_eligible"].astype(bool).to_dict()

    results: dict[str, list[pd.DataFrame]] = {
        key: [] for key in ("support", "structural", "cells", "horizons", "periods")
    }
    columns = required_columns()
    for period, (path, mode) in INPUTS.items():
        frame = pd.read_parquet(path, columns=columns)
        validate_frame(frame, period)
        evaluated = evaluate_period(
            frame, period, mode, manifest, parent_contract, eligibility
        )
        for key in results:
            results[key].append(evaluated[key])
        del frame

    combined = {key: pd.concat(value, ignore_index=True) for key, value in results.items()}
    combined["support"].to_csv(output / "route_support.csv", index=False)
    combined["structural"].to_csv(output / "route_structural.csv", index=False)
    combined["cells"].to_csv(output / "route_quality_cells.csv", index=False)
    combined["horizons"].to_csv(output / "route_horizon_grades.csv", index=False)
    combined["periods"].to_csv(output / "route_period_grades.csv", index=False)

    cross_horizon, cross_route = cross_period_grades(
        combined["periods"], combined["horizons"]
    )
    cross_horizon.to_csv(output / "route_cross_period_horizon_grades.csv", index=False)
    cross_route.to_csv(output / "route_cross_period_global_grades.csv", index=False)

    after = {name: snapshot_shadow(path) for name, path in SHADOWS.items()}
    write_json(output / "protected_shadows_after.json", after)
    if before != after:
        raise AssertionError("a protected shadow changed during the diagnostic")

    supported_counts = (
        combined["support"].groupby("period")["combined_support_pass"].sum().astype(int)
    )
    summary = {
        "algorithm": "oriented_route_movement_diagnostic_v1",
        "scientific_status": "post_outcome_exploratory_development_diagnostic",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim": False,
        "prospective_validated": False,
        "promotion_or_surface_permission": False,
        "models_refit": False,
        "thresholds_changed": False,
        "total_route_units": len(manifest),
        "ambiguous_union_route_units": int(
            manifest["route_kind"].eq("ambiguous_union").sum()
        ),
        "supported_route_units_by_period": supported_counts.to_dict(),
        "evaluated_quality_cells": len(combined["cells"]),
        "cross_period_horizon_grade_counts": cross_horizon[
            "cross_period_grade"
        ].value_counts().to_dict(),
        "cross_period_global_grade_counts": cross_route[
            "cross_period_global_grade"
        ].value_counts().to_dict(),
        "diagnostic_good_or_high_global_routes": cross_route.loc[
            cross_route["cross_period_global_grade"].isin(
                [LABELS["good"], LABELS["high"]]
            ),
            "route_id",
        ].tolist(),
        "shadow_trees_unchanged": True,
        "source_manifest_sha256": sources["manifest_sha256"],
        "contract_sha256": sha256(CONTRACT),
        "interpretation": (
            "Movement magnitude/range only. Results were selected after outcomes were "
            "available and are hypotheses, not certified tiers, direction, P&L, economic "
            "edge, tradability, or shadow eligibility."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
