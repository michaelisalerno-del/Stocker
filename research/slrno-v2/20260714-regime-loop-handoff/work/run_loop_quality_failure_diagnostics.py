"""Read-only attribution diagnostics for the frozen loop-quality experiment.

This module does not fit or recalibrate a model.  It reads the already sealed
per-loop quality artifacts, decomposes their frozen gates, and writes a new
diagnostic bundle under /private/tmp.  The input artifact tree is hashed before
and after the analysis.  No prospective-shadow path is read or written.

The labels in this package concern absolute movement magnitude and future
range only.  They are not direction, signed-return, P&L, tradability, or live
execution labels.
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


DEFAULT_ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_per_loop_movement_quality_20260710"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/private/tmp/stocker_loop_quality_failure_diagnostics_20260710"
)

PERIODS = ("2024_oof", "2025", "2023")
PERIOD_FILE_STEMS = {
    "2024_oof": "oof_predictions_2024.parquet",
    "2025": "quality_scoring_2025.parquet",
    "2023": "quality_scoring_2023.parquet",
}
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
CYCLE_COUNT = 20

# Frozen contract values used only to describe diagnostic axes.  The original
# pass/fail gates are always read from gate_detail; they are never recomputed or
# changed here.
P75_GOOD_LEVEL = 0.30
P75_HIGH_LEVEL = 0.35
P90_HIGH_LEVEL = 0.15

REQUIRED_FILES = (
    "provisional_quality_cells_2024.csv",
    "quality_period_cells.csv",
    "provisional_support_2024.csv",
    "quality_period_support.csv",
    "provisional_structural_2024.csv",
    "quality_period_structural.csv",
    "provisional_tiers_2024.csv",
    "quality_period_cycle_grades.csv",
    "final_cycle_tiers.csv",
    "fixed_cycles.csv",
    "gates.json",
    "summary.json",
    *PERIOD_FILE_STEMS.values(),
)


def safe(value: Any) -> Any:
    """Convert numpy/pandas values into deterministic JSON values."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def validate_roots(artifact_root: Path, output_root: Path) -> None:
    artifact = artifact_root.resolve()
    output = output_root.resolve()
    private_tmp = Path("/private/tmp").resolve()
    if artifact != DEFAULT_ARTIFACT_ROOT.resolve():
        raise ValueError("diagnostics may read only the sealed per-loop artifact root")
    if (
        output.parent != private_tmp
        or not output.name.startswith("stocker_loop_quality_failure_diagnostics_")
    ):
        raise ValueError("diagnostic output must be a dedicated /private/tmp directory")
    if artifact == output or artifact in output.parents or output in artifact.parents:
        raise ValueError("diagnostic output must be separate from the frozen inputs")
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    missing = [name for name in REQUIRED_FILES if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen artifacts: {missing}")


def input_hashes(artifact_root: Path) -> dict[str, str]:
    return {name: sha256(artifact_root / name) for name in REQUIRED_FILES}


def _canonical_period(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["period"] = result["period"].astype(str)
    unexpected = set(result["period"]) - set(PERIODS)
    if unexpected:
        raise AssertionError(f"unexpected periods: {sorted(unexpected)}")
    return result


def load_frozen_tables(artifact_root: Path) -> dict[str, pd.DataFrame]:
    cells = _canonical_period(
        pd.concat(
            [
                pd.read_csv(artifact_root / "provisional_quality_cells_2024.csv"),
                pd.read_csv(artifact_root / "quality_period_cells.csv"),
            ],
            ignore_index=True,
        )
    )
    support = _canonical_period(
        pd.concat(
            [
                pd.read_csv(artifact_root / "provisional_support_2024.csv"),
                pd.read_csv(artifact_root / "quality_period_support.csv"),
            ],
            ignore_index=True,
        )
    )
    structural = _canonical_period(
        pd.concat(
            [
                pd.read_csv(artifact_root / "provisional_structural_2024.csv"),
                pd.read_csv(artifact_root / "quality_period_structural.csv"),
            ],
            ignore_index=True,
        )
    )
    grades = _canonical_period(
        pd.concat(
            [
                pd.read_csv(artifact_root / "provisional_tiers_2024.csv"),
                pd.read_csv(artifact_root / "quality_period_cycle_grades.csv"),
            ],
            ignore_index=True,
        )
    )
    final_tiers = pd.read_csv(artifact_root / "final_cycle_tiers.csv")
    cycles = pd.read_csv(artifact_root / "fixed_cycles.csv")

    expected_cells = len(PERIODS) * CYCLE_COUNT * len(TARGETS) * len(HORIZONS) * len(TIERS)
    if len(cells) != expected_cells:
        raise AssertionError(f"expected {expected_cells} frozen quality cells")
    cell_key = ["period", "cycle_id", "target", "horizon", "tier"]
    if cells.duplicated(cell_key).any():
        raise AssertionError("duplicate frozen quality cell")
    if len(support) != len(PERIODS) * CYCLE_COUNT:
        raise AssertionError("unexpected support row count")
    if len(structural) != len(PERIODS) * CYCLE_COUNT:
        raise AssertionError("unexpected structural row count")
    if len(grades) != len(PERIODS) * CYCLE_COUNT:
        raise AssertionError("unexpected grade row count")
    if len(final_tiers) != CYCLE_COUNT or len(cycles) != CYCLE_COUNT:
        raise AssertionError("frozen cycle count changed")
    if not final_tiers["final_grade"].eq("unqualified").all():
        raise AssertionError("this diagnostic expects the sealed all-unqualified result")
    if final_tiers["prospective_validated"].astype(bool).any():
        raise AssertionError("prospective validation must remain false")
    return {
        "cells": cells,
        "support": support,
        "structural": structural,
        "grades": grades,
        "final_tiers": final_tiers,
        "cycles": cycles,
    }


def gate_family_flags(gate_detail: str | Mapping[str, Any]) -> dict[str, bool]:
    """Map a frozen cell gate into separately reported diagnostic families."""

    detail = json.loads(gate_detail) if isinstance(gate_detail, str) else gate_detail
    checks = detail["checks"]
    conditional = detail["conditional_gate"]["checks"]
    joint = detail["joint_gate"]["checks"]
    return {
        "event_support": bool(checks["support"]),
        "rate_level": bool(checks["observed_rate"] and checks["mean_qcycle"]),
        "rate_vs_context": bool(checks["observed_over_qcontext"]),
        "residual_ci": bool(checks["lift_interval"]),
        "conditional_core": bool(
            conditional["relative_log_loss"] and conditional["brier_difference"]
        ),
        "conditional_daily_ci": bool(conditional["daily_intervals"]),
        "conditional_robustness": bool(
            conditional["quarter_and_stock_robustness"]
        ),
        "conditional_calibration": bool(
            conditional["ece_no_worse"]
            and conditional["maximum_supported_bin_error"]
        ),
        "joint_core": bool(joint["relative_log_loss"] and joint["brier_difference"]),
        "joint_daily_ci": bool(joint["daily_intervals"]),
        "joint_robustness": bool(joint["quarter_and_stock_robustness"]),
        "joint_calibration": bool(
            joint["ece_no_worse"] and joint["maximum_supported_bin_error"]
        ),
    }


def enrich_cells(cells: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(
        [gate_family_flags(value) for value in cells["gate_detail"]],
        index=cells.index,
    ).add_prefix("gate__")
    result = pd.concat([cells.copy(), flags], axis=1)
    result["p75_high_rate_level"] = (
        result["tier"].eq("p75")
        & result["observed_rate"].ge(P75_HIGH_LEVEL)
        & result["mean_qcycle_probability"].ge(P75_HIGH_LEVEL)
    )
    result["incremental_ratio_and_residual"] = (
        result["gate__rate_vs_context"] & result["gate__residual_ci"]
    )
    return result


def gate_family_counts(cells: pd.DataFrame) -> pd.DataFrame:
    families = [column for column in cells if column.startswith("gate__")]
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        for tier in TIERS:
            subset = cells.loc[
                cells["period"].eq(period) & cells["tier"].eq(tier)
            ]
            rows.append(
                {
                    "period": period,
                    "tier": tier,
                    "gate_family": "full_cell",
                    "pass_count": int(subset["pass"].sum()),
                    "fail_count": int((~subset["pass"]).sum()),
                    "total_cells": len(subset),
                    "pass_rate": float(subset["pass"].mean()),
                }
            )
            for column in families:
                passed = int(subset[column].sum())
                rows.append(
                    {
                        "period": period,
                        "tier": tier,
                        "gate_family": column.removeprefix("gate__"),
                        "pass_count": passed,
                        "fail_count": len(subset) - passed,
                        "total_cells": len(subset),
                        "pass_rate": passed / len(subset),
                    }
                )
    return pd.DataFrame(rows)


def _period_axis_row(
    period: str,
    cycle_id: str,
    subset: pd.DataFrame,
    support_pass: bool,
    structural_pass: bool,
    global_grade: str,
) -> dict[str, Any]:
    p75 = subset.loc[subset["tier"].eq("p75")]
    p90 = subset.loc[subset["tier"].eq("p90")]
    good_level = int(p75["gate__rate_level"].sum())
    high_level = int(p75["p75_high_rate_level"].sum())
    incremental = int(p75["incremental_ratio_and_residual"].sum())
    full = int(p75["pass"].sum())
    if high_level == 6:
        absolute_axis = "absolute_high_period"
    elif good_level == 6:
        absolute_axis = "absolute_good_period"
    elif good_level:
        absolute_axis = "partial_absolute_level"
    else:
        absolute_axis = "no_absolute_level_evidence"
    if full == 6:
        incremental_axis = "incremental_all_cells"
    elif incremental == 6:
        incremental_axis = "incremental_lift_but_other_gates_fail"
    elif full:
        incremental_axis = "incremental_partial"
    elif incremental:
        incremental_axis = "incremental_lift_partial"
    else:
        incremental_axis = "incremental_unconfirmed"
    return {
        "period": period,
        "cycle_id": cycle_id,
        "base_support_pass": bool(support_pass),
        "structural_pass": bool(structural_pass),
        "p75_good_level_cells_of_6": good_level,
        "p75_high_level_cells_of_6": high_level,
        "p75_incremental_ratio_and_residual_cells_of_6": incremental,
        "p75_full_pass_cells_of_6": full,
        "p75_conditional_core_cells_of_6": int(p75["gate__conditional_core"].sum()),
        "p75_joint_core_cells_of_6": int(p75["gate__joint_core"].sum()),
        "p75_both_robustness_cells_of_6": int(
            (p75["gate__conditional_robustness"] & p75["gate__joint_robustness"]).sum()
        ),
        "p90_full_pass_cells_of_6": int(p90["pass"].sum()),
        "absolute_movement_level_axis": absolute_axis,
        "incremental_vs_context_axis": incremental_axis,
        "frozen_global_grade": global_grade,
    }


def build_two_axis_table(
    cells: pd.DataFrame,
    support: pd.DataFrame,
    structural: pd.DataFrame,
    grades: pd.DataFrame,
) -> pd.DataFrame:
    support_map = support.set_index(["period", "cycle_id"])["combined_support_pass"]
    structural_map = structural.set_index(["period", "cycle_id"])["pass"]
    grade_map = grades.set_index(["period", "cycle_id"])["global_grade"]
    rows: list[dict[str, Any]] = []
    for (period, cycle_id), subset in cells.groupby(["period", "cycle_id"], sort=True):
        rows.append(
            _period_axis_row(
                str(period),
                str(cycle_id),
                subset,
                bool(support_map.loc[(period, cycle_id)]),
                bool(structural_map.loc[(period, cycle_id)]),
                str(grade_map.loc[(period, cycle_id)]),
            )
        )
    result = pd.DataFrame(rows)
    result["period"] = pd.Categorical(result["period"], PERIODS, ordered=True)
    return result.sort_values(["cycle_id", "period"]).reset_index(drop=True)


def build_all_cycle_decomposition(
    two_axis: pd.DataFrame,
    cycles: pd.DataFrame,
    final_tiers: pd.DataFrame,
) -> pd.DataFrame:
    result = cycles[["cycle_id", "cycle", "transition_length"]].copy()
    fields = (
        "base_support_pass",
        "structural_pass",
        "p75_full_pass_cells_of_6",
        "p75_good_level_cells_of_6",
        "p75_high_level_cells_of_6",
        "p75_incremental_ratio_and_residual_cells_of_6",
        "p90_full_pass_cells_of_6",
        "absolute_movement_level_axis",
        "incremental_vs_context_axis",
        "frozen_global_grade",
    )
    for period in PERIODS:
        label = period.replace("2024_oof", "oof_2024")
        period_frame = two_axis.loc[two_axis["period"].astype(str).eq(period)]
        period_frame = period_frame.set_index("cycle_id")
        for field in fields:
            result[f"{label}__{field}"] = result["cycle_id"].map(period_frame[field])
    result = result.merge(
        final_tiers[["cycle_id", "final_grade"]], on="cycle_id", how="left", validate="one_to_one"
    )
    result["frozen_grade_changed"] = False
    return result


def correlation_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Cross-period Pearson and Spearman correlations on matched frozen cells."""

    keys = ["cycle_id", "target", "horizon", "tier"]
    metrics = (
        "observed_rate",
        "mean_qcycle_probability",
        "observed_rate_divided_by_mean_qcontext",
        "daily_residual_ci_low",
        "conditional_relative_log_loss_improvement",
        "joint_relative_log_loss_improvement",
    )
    pairs = (("2024_oof", "2025"), ("2024_oof", "2023"), ("2025", "2023"))
    rows: list[dict[str, Any]] = []
    for tier in TIERS:
        subset = cells.loc[cells["tier"].eq(tier)]
        for metric in metrics:
            wide = subset.pivot(index=keys, columns="period", values=metric)
            for left, right in pairs:
                pair = wide[[left, right]].dropna()
                rows.append(
                    {
                        "tier": tier,
                        "metric": metric,
                        "left_period": left,
                        "right_period": right,
                        "matched_cells": len(pair),
                        "pearson": float(pair[left].corr(pair[right], method="pearson")),
                        "spearman": float(pair[left].corr(pair[right], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def weighted_rate(observed: Iterable[float], weights: Iterable[float]) -> float:
    values = np.asarray(observed, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if len(values) != len(weight) or not len(values):
        raise ValueError("observed and weights must have equal non-zero length")
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive finite mass")
    return float(np.dot(values, weight) / total)


def _raw_rate_row(
    frame: pd.DataFrame,
    observed: np.ndarray,
    key: dict[str, Any],
    slice_type: str,
    slice_value: str,
) -> dict[str, Any]:
    weight = frame["conditional_weight"].to_numpy(dtype=float)
    return {
        **key,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "rows": len(frame),
        "weight": float(weight.sum()),
        "positive_rows": int(observed.sum()),
        "positive_weight": float(np.dot(observed, weight)),
        "observed_rate": weighted_rate(observed, weight),
    }


def raw_rate_stability(
    artifact_root: Path, cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_columns = [
        f"quality_class__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
    ]
    base_columns = [
        "cycle_id",
        "symbol_norm",
        "quarter",
        "loop_occurs",
        "conditional_weight",
    ]
    quarter_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cell_rates = cells.set_index(["period", "cycle_id", "target", "horizon", "tier"])[
        "observed_rate"
    ]
    for period in PERIODS:
        frame = pd.read_parquet(
            artifact_root / PERIOD_FILE_STEMS[period],
            columns=base_columns + class_columns,
        )
        frame = frame.loc[frame["loop_occurs"].eq(1)].reset_index(drop=True)
        for cycle_id, cycle_frame in frame.groupby("cycle_id", sort=True):
            cycle_frame = cycle_frame.reset_index(drop=True)
            quarter_groups = {
                str(name): np.asarray(index, dtype=int)
                for name, index in cycle_frame.groupby("quarter", sort=True).indices.items()
            }
            symbols = sorted(cycle_frame["symbol_norm"].astype(str).unique())
            symbol_values = cycle_frame["symbol_norm"].astype(str).to_numpy()
            for target in TARGETS:
                for horizon in HORIZONS:
                    classes = cycle_frame[
                        f"quality_class__{target}__h{horizon}"
                    ].to_numpy(dtype=int)
                    for tier, threshold_class in (("p75", 1), ("p90", 2)):
                        observed = (classes >= threshold_class).astype(float)
                        key = {
                            "period": period,
                            "cycle_id": str(cycle_id),
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                        }
                        full_rate = weighted_rate(
                            observed,
                            cycle_frame["conditional_weight"].to_numpy(dtype=float),
                        )
                        recorded = float(
                            cell_rates.loc[(period, cycle_id, target, horizon, tier)]
                        )
                        if not math.isclose(full_rate, recorded, rel_tol=0.0, abs_tol=1e-12):
                            raise AssertionError("raw-rate reconstruction disagrees with frozen cell")
                        local_quarters = []
                        for quarter, indices in quarter_groups.items():
                            row = _raw_rate_row(
                                cycle_frame.iloc[indices].reset_index(drop=True),
                                observed[indices],
                                key,
                                "quarter",
                                quarter,
                            )
                            quarter_rows.append(row)
                            local_quarters.append(row["observed_rate"])
                        local_deletions = []
                        for symbol in symbols:
                            keep = symbol_values != symbol
                            row = _raw_rate_row(
                                cycle_frame.loc[keep].reset_index(drop=True),
                                observed[keep],
                                key,
                                "leave_one_stock_out",
                                symbol,
                            )
                            deletion_rows.append(row)
                            local_deletions.append(row["observed_rate"])
                        level = P75_GOOD_LEVEL if tier == "p75" else P90_HIGH_LEVEL
                        summary_rows.append(
                            {
                                **key,
                                "full_observed_rate": full_rate,
                                "quarter_count": len(local_quarters),
                                "minimum_quarter_rate": min(local_quarters),
                                "maximum_quarter_rate": max(local_quarters),
                                "quarter_rate_range": max(local_quarters) - min(local_quarters),
                                "stock_deletion_count": len(local_deletions),
                                "minimum_leave_one_stock_out_rate": min(local_deletions),
                                "maximum_leave_one_stock_out_rate": max(local_deletions),
                                "leave_one_stock_out_rate_range": max(local_deletions)
                                - min(local_deletions),
                                "contract_level_threshold": level,
                                "all_quarters_clear_contract_level": bool(
                                    min(local_quarters) >= level
                                ),
                                "all_leave_one_stock_out_clear_contract_level": bool(
                                    min(local_deletions) >= level
                                ),
                                "p75_high_level_threshold": P75_HIGH_LEVEL
                                if tier == "p75"
                                else math.nan,
                                "all_quarters_clear_p75_high_level": bool(
                                    tier == "p75"
                                    and min(local_quarters) >= P75_HIGH_LEVEL
                                ),
                                "all_leave_one_stock_out_clear_p75_high_level": bool(
                                    tier == "p75"
                                    and min(local_deletions) >= P75_HIGH_LEVEL
                                ),
                            }
                        )
    return (
        pd.DataFrame(quarter_rows),
        pd.DataFrame(deletion_rows),
        pd.DataFrame(summary_rows),
    )


def structural_reliability(
    structural: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    route = cycles.set_index("cycle_id")["cycle"]
    for _, row in structural.iterrows():
        checks = json.loads(row["checks"])
        rows.append(
            {
                "period": row["period"],
                "cycle_id": row["cycle_id"],
                "cycle": route.loc[row["cycle_id"]],
                "pass": bool(row["pass"]),
                "log_loss_lower": bool(checks["log_loss_lower"]),
                "brier_lower": bool(checks["brier_lower"]),
                "ece_no_worse": bool(checks["ece_no_worse"]),
                "maximum_supported_bin_error_pass": bool(
                    checks["maximum_supported_bin_error"]
                ),
                "history_minus_first_order_log_loss": float(
                    row["history_log_loss"] - row["first_order_log_loss"]
                ),
                "history_minus_first_order_brier": float(
                    row["history_brier"] - row["first_order_brier"]
                ),
                "history_minus_first_order_ece": float(
                    row["history_ece"] - row["first_order_ece"]
                ),
                "history_minus_first_order_maximum_supported_bin_error": float(
                    row["history_maximum_supported_bin_error"]
                    - row["first_order_maximum_supported_bin_error"]
                ),
                "oof_structural_is_diagnostic_only": row["period"] == "2024_oof",
            }
        )
    return pd.DataFrame(rows)


def selective_diagnostics(
    two_axis: pd.DataFrame,
    stability: pd.DataFrame,
    structural: pd.DataFrame,
    cycles: pd.DataFrame,
    final_tiers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    p75_stability = stability.loc[stability["tier"].eq("p75")]
    candidate_cycles: list[str] = []
    for cycle_id, axis in two_axis.groupby("cycle_id", sort=True):
        axis_high = bool(axis["p75_high_level_cells_of_6"].eq(6).all())
        slices = p75_stability.loc[p75_stability["cycle_id"].eq(cycle_id)]
        slice_high = bool(
            len(slices) == len(PERIODS) * len(TARGETS) * len(HORIZONS)
            and slices["all_quarters_clear_p75_high_level"].all()
            and slices["all_leave_one_stock_out_clear_p75_high_level"].all()
        )
        if axis_high and slice_high:
            candidate_cycles.append(str(cycle_id))
    if candidate_cycles != ["cycle_07", "cycle_13"]:
        raise AssertionError(
            f"unexpected exploratory absolute-high candidates: {candidate_cycles}"
        )

    structural_map = structural.groupby("cycle_id")["pass"].all()
    structurally_reliable = [
        cycle_id for cycle_id in candidate_cycles if bool(structural_map.loc[cycle_id])
    ]
    if structurally_reliable != ["cycle_07"]:
        raise AssertionError(
            f"unexpected all-period structural subset: {structurally_reliable}"
        )

    incremental = (
        two_axis.groupby("cycle_id")
        .agg(
            total_p75_full_pass_cells=("p75_full_pass_cells_of_6", "sum"),
            total_p75_incremental_cells=(
                "p75_incremental_ratio_and_residual_cells_of_6",
                "sum",
            ),
        )
        .sort_values(
            ["total_p75_full_pass_cells", "total_p75_incremental_cells"],
            ascending=False,
        )
    )
    strongest_incremental = str(incremental.index[0])
    if strongest_incremental != "cycle_09":
        raise AssertionError(
            f"unexpected strongest incremental candidate: {strongest_incremental}"
        )

    cycle_route = cycles.set_index("cycle_id")["cycle"]
    final_grade = final_tiers.set_index("cycle_id")["final_grade"]
    selected = ("cycle_06", "cycle_07", "cycle_09", "cycle_13")
    rows: list[dict[str, Any]] = []
    roles = {
        "cycle_06": "absolute_high_period_levels_but_quarter_instability",
        "cycle_07": "exploratory_absolute_high_candidate_with_structural_reliability",
        "cycle_09": "strongest_incremental_candidate_but_not_cross_period_robust",
        "cycle_13": "exploratory_absolute_high_candidate_without_all_period_structural_reliability",
    }
    for cycle_id in selected:
        axis = two_axis.loc[two_axis["cycle_id"].eq(cycle_id)]
        slices = p75_stability.loc[p75_stability["cycle_id"].eq(cycle_id)]
        rows.append(
            {
                "cycle_id": cycle_id,
                "cycle": cycle_route.loc[cycle_id],
                "diagnostic_role": roles[cycle_id],
                "all_periods_have_six_p75_high_level_cells": bool(
                    axis["p75_high_level_cells_of_6"].eq(6).all()
                ),
                "all_quarters_clear_p75_high_level": bool(
                    slices["all_quarters_clear_p75_high_level"].all()
                ),
                "all_leave_one_stock_out_clear_p75_high_level": bool(
                    slices["all_leave_one_stock_out_clear_p75_high_level"].all()
                ),
                "structural_reliability_all_periods": bool(
                    structural_map.loc[cycle_id]
                ),
                "total_p75_full_pass_cells_of_18": int(
                    axis["p75_full_pass_cells_of_6"].sum()
                ),
                "total_p75_incremental_ratio_and_residual_cells_of_18": int(
                    axis["p75_incremental_ratio_and_residual_cells_of_6"].sum()
                ),
                "minimum_p75_quarter_rate": float(slices["minimum_quarter_rate"].min()),
                "minimum_p75_leave_one_stock_out_rate": float(
                    slices["minimum_leave_one_stock_out_rate"].min()
                ),
                "frozen_final_grade": final_grade.loc[cycle_id],
                "frozen_grade_changed": False,
                "prospective_validated": False,
            }
        )
    payload = {
        "exploratory_absolute_high_candidates": candidate_cycles,
        "exploratory_absolute_high_candidates_with_all_period_structural_reliability": structurally_reliable,
        "strongest_incremental_candidate": strongest_incremental,
        "strongest_incremental_candidate_cross_period_robust": False,
        "interpretation": (
            "Absolute-high and incremental-vs-context are separate diagnostic axes. "
            "No diagnostic label changes a frozen movement-quality grade."
        ),
    }
    return pd.DataFrame(rows), payload


def base_support_summary(support: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for period in PERIODS:
        subset = support.loc[support["period"].eq(period)]
        result[period] = {
            "pass_count": int(subset["combined_support_pass"].sum()),
            "total_cycles": len(subset),
            "failed_cycles": sorted(
                subset.loc[~subset["combined_support_pass"], "cycle_id"].astype(str)
            ),
        }
    return result


def output_hashes(output_root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256(output_root / name) for name in names}


def run(artifact_root: Path, output_root: Path) -> dict[str, Any]:
    validate_roots(artifact_root, output_root)
    before_hashes = input_hashes(artifact_root)
    tables = load_frozen_tables(artifact_root)
    cells = enrich_cells(tables["cells"])

    gate_counts = gate_family_counts(cells)
    two_axis = build_two_axis_table(
        cells, tables["support"], tables["structural"], tables["grades"]
    )
    all_cycles = build_all_cycle_decomposition(
        two_axis, tables["cycles"], tables["final_tiers"]
    )
    correlations = correlation_table(cells)
    quarter, deletions, stability = raw_rate_stability(artifact_root, cells)
    structural = structural_reliability(tables["structural"], tables["cycles"])
    selected, selected_summary = selective_diagnostics(
        two_axis,
        stability,
        structural,
        tables["cycles"],
        tables["final_tiers"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gate_family_counts.csv": gate_counts,
        "all_cycle_decomposition.csv": all_cycles,
        "cross_period_correlations.csv": correlations,
        "two_axis_movement_vs_incremental.csv": two_axis,
        "raw_rate_quarter_stability.csv": quarter,
        "raw_rate_leave_one_stock_out_stability.csv": deletions,
        "raw_rate_stability_summary.csv": stability,
        "structural_reliability.csv": structural,
        "selective_loop_diagnostics.csv": selected,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_root / name, index=False)

    after_hashes = input_hashes(artifact_root)
    if before_hashes != after_hashes:
        raise AssertionError("frozen input artifact changed during read-only diagnostics")
    csv_hashes = output_hashes(output_root, outputs)
    summary = {
        "analysis_id": "loop_quality_failure_attribution_20260710",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "diagnostic_only": True,
        "model_refit_performed": False,
        "recalibration_performed": False,
        "frozen_grade_changed": False,
        "prospective_validated": False,
        "input_artifact_root": str(artifact_root),
        "output_artifact_root": str(output_root),
        "input_hashes_before_and_after_match": True,
        "input_hashes": before_hashes,
        "diagnostic_output_hashes": csv_hashes,
        "rows": {name: len(frame) for name, frame in outputs.items()},
        "base_support": base_support_summary(tables["support"]),
        "structural_pass_counts": {
            period: int(
                structural.loc[structural["period"].eq(period), "pass"].sum()
            )
            for period in PERIODS
        },
        "selective_diagnostics": selected_summary,
        "frozen_final_grade_counts": tables["final_tiers"]["final_grade"]
        .value_counts()
        .to_dict(),
        "conclusion": (
            "The all-horizon failure is not primarily base support. Absolute "
            "movement level and incremental-vs-context evidence separate, and "
            "named-cycle lift is not stable across periods and robustness slices."
        ),
        "interpretation": (
            "Movement magnitude and future range only; no direction, signed "
            "return, P&L, economic edge, tradability, order, or deployment claim."
        ),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.artifact_root, args.output_root)
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
