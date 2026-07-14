"""Causal individual-expert selection for regime-loop movement linkage.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-individual-expert-selection-v1.json"
SOURCE = Path(
    "/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711/"
    "linkage_predictions_2024_sep_dec.parquet"
)
SOURCE_AUDIT = Path(
    "/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711/independent_audit.json"
)
SOURCE_CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v3.json"
SOURCE_RUNNER = HERE / "run_regime_loop_linkage_ideas_v3.py"
ROOT = Path(
    "/private/tmp/stocker_regime_loop_individual_expert_selection_v1_20260711"
)

EXPECTED_HASHES = {
    "contract": "d34dade298518eb37a1e710c838d59d7e53875c9dc33c0054934b53902966267",
    "source": "99374428d372711b233cf6dfbe59a18f5667e032ef3039b2ae05df13400cd660",
    "source_audit": "92343c008f0cd585c3e02c1e3c60905aa4f9cde7e9c9b99111076a2cd8be300f",
    "source_contract": "88a60956857e6ccb4fb5e74beb9085e46765e55b31763b26927dc496822ce947",
    "source_runner": "c0e8786670fd51e3d93290ecd56ba51322ebe6ace0fd7e521803f2fd8c1ce72e",
}

MONTHS = ("2024-09", "2024-10", "2024-11", "2024-12")
VALIDATION_MONTHS = ("2024-11", "2024-12")
SELECTION_MONTHS = {
    "2024-11": ("2024-09", "2024-10"),
    "2024-12": ("2024-09", "2024-10", "2024-11"),
}
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
EXPERTS = (
    "baseline",
    "minimal_time_topology",
    "raw_full_link",
    "partial_full_link",
    "calibrated_raw_product",
    "dependency_stack",
)
TIE_PRIORITY = (
    "baseline",
    "partial_full_link",
    "minimal_time_topology",
    "raw_full_link",
    "calibrated_raw_product",
    "dependency_stack",
)
SELECTORS = (
    "global_best",
    "regime_best",
    "loop_best",
    "loop_regime_best",
    "guarded_loop_regime_best",
    "hierarchical_clock_best",
)
CANDIDATE_SELECTORS = SELECTORS[1:]
COMPARISONS = ("baseline", "raw_full_link", "global_best")
GROUP_COLUMNS = {
    "regime_best": ("current_state",),
    "loop_best": ("cycle_id",),
    "loop_regime_best": ("cycle_id", "current_state"),
    "guarded_loop_regime_best": ("cycle_id", "current_state"),
}
SUPPORT = {
    "regime": (500, 30),
    "loop": (500, 30),
    "loop_regime": (300, 24),
    "loop_regime_clock": (150, 12),
}
SUPPORT_BY_SELECTOR = {
    "regime_best": SUPPORT["regime"],
    "loop_best": SUPPORT["loop"],
    "loop_regime_best": SUPPORT["loop_regime"],
    "guarded_loop_regime_best": SUPPORT["loop_regime"],
}
GUARD_MARGIN = 0.0005
TIE_TOLERANCE = 1e-12
LOSS_EPSILON = 1e-12
SEED = 20260711
BOOTSTRAP_DRAWS = 4999
SIGN_FLIP_DRAWS = 9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def target_column(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def expert_column(expert: str, target: str, horizon: int, tier: str) -> str:
    return f"link__{expert}__{target}__h{horizon}__{tier}"


def selector_column(selector: str, target: str, horizon: int, tier: str) -> str:
    return f"selector__{selector}__{target}__h{horizon}__{tier}"


def selected_expert_column(selector: str) -> str:
    return f"selected_expert__{selector}"


def source_columns() -> list[str]:
    return [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month",
        "cycle_index",
        "cycle_id",
        "state",
        "current_state",
        "entry_clock_quartile",
        "inverse_compatible_weight",
        *[
            target_column(target, horizon, tier)
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
        *[
            expert_column(expert, target, horizon, tier)
            for expert in EXPERTS
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
    ]


def verify_contract_and_sources() -> tuple[dict[str, Any], dict[str, str]]:
    hashes = {
        "contract": sha256(CONTRACT),
        "source": sha256(SOURCE),
        "source_audit": sha256(SOURCE_AUDIT),
        "source_contract": sha256(SOURCE_CONTRACT),
        "source_runner": sha256(SOURCE_RUNNER),
    }
    if hashes != EXPECTED_HASHES:
        raise AssertionError(f"frozen hash mismatch: {hashes}")
    contract = json.loads(CONTRACT.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["economic_edge_claim"] is False
        and contract["population_and_causality"]["later_period_paths_permitted"]
        is False
        and contract["population_and_causality"][
            "prospective_shadow_read_or_write_permitted"
        ]
        is False
    ):
        raise AssertionError("contract safety semantics changed")
    if tuple(contract["experts"]["selectable"]) != EXPERTS:
        raise AssertionError("expert set changed")
    if tuple(contract["decision"]["candidate_selectors"]) != CANDIDATE_SELECTORS:
        raise AssertionError("selector set changed")
    source_audit = json.loads(SOURCE_AUDIT.read_text())
    if not (
        source_audit["all_passed"] is True
        and source_audit["checks_passed"] == 19
        and source_audit["checks_total"] == 19
    ):
        raise AssertionError("source audit is not fully passing")
    return contract, hashes


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(SOURCE, columns=source_columns())
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    if len(frame) != 130672 or tuple(sorted(frame["month"].unique())) != MONTHS:
        raise AssertionError("source population changed")
    if frame[source_columns()].isna().any().any():
        raise AssertionError("source has missing values")
    probability_columns = [
        expert_column(expert, target, horizon, tier)
        for expert in EXPERTS
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    values = frame[probability_columns].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("invalid expert probabilities")
    audit = {
        "rows": len(frame),
        "anchors": int(frame["anchor_id"].nunique()),
        "sessions": int(frame["session_date"].nunique()),
        "stocks": int(frame["symbol_norm"].nunique()),
        "cycles": int(frame["cycle_id"].nunique()),
        "states": sorted(frame["current_state"].unique().astype(int).tolist()),
        "validation_rows": int(frame["month"].isin(VALIDATION_MONTHS).sum()),
        "validation_anchors": int(
            frame.loc[frame["month"].isin(VALIDATION_MONTHS), "anchor_id"].nunique()
        ),
        "validation_sessions": int(
            frame.loc[
                frame["month"].isin(VALIDATION_MONTHS), "session_date"
            ].nunique()
        ),
    }
    audit["support_checks"] = {
        "validation_rows": audit["validation_rows"] >= 50000,
        "validation_anchors": audit["validation_anchors"] >= 8000,
        "validation_sessions": audit["validation_sessions"] >= 40,
        "stocks": audit["stocks"] == 22,
        "cycles": audit["cycles"] == 20,
        "states": audit["states"] == list(range(8)),
    }
    if not all(audit["support_checks"].values()):
        raise AssertionError(f"source support failed: {audit}")
    return frame, audit


def binary_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return math.nan
    return float(np.average(values, weights=weights))


def expert_row_log_loss(frame: pd.DataFrame, expert: str) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[target_column(target, horizon, tier)].to_numpy(int)
                p = frame[expert_column(expert, target, horizon, tier)].to_numpy(float)
                result += binary_losses(y, p)[0] / 12.0
    return result


def expert_losses(frame: pd.DataFrame) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    return {
        expert: weighted_mean(expert_row_log_loss(frame, expert), weights)
        for expert in EXPERTS
    }


def choose_expert(losses: dict[str, float]) -> str:
    finite = {expert: value for expert, value in losses.items() if np.isfinite(value)}
    if not finite:
        raise AssertionError("no finite expert objective")
    best = min(finite.values())
    tied = {
        expert
        for expert, value in finite.items()
        if value <= best + TIE_TOLERANCE
    }
    return next(expert for expert in TIE_PRIORITY if expert in tied)


def joint_positives(frame: pd.DataFrame) -> int:
    return int(
        sum(
            frame[target_column(target, horizon, tier)].sum()
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        )
    )


def unit_key(columns: Sequence[str], values: Sequence[Any]) -> str:
    return "|".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))


def normalized_key(values: Any) -> tuple[Any, ...]:
    return values if isinstance(values, tuple) else (values,)


def flat_assignments(
    selection: pd.DataFrame,
    validation: pd.DataFrame,
    validation_month: str,
    selector: str,
    global_expert: str,
    guarded: bool,
) -> tuple[dict[tuple[Any, ...], str], list[dict[str, Any]]]:
    columns = GROUP_COLUMNS[selector]
    minimum_rows, minimum_positives = SUPPORT_BY_SELECTOR[selector]
    training_groups = {
        normalized_key(key): group
        for key, group in selection.groupby(list(columns), sort=True)
    }
    validation_keys = sorted(
        {
            tuple(row)
            for row in validation.loc[:, list(columns)].itertuples(index=False, name=None)
        },
        key=lambda item: tuple(str(value) for value in item),
    )
    mapping: dict[tuple[Any, ...], str] = {}
    ledger: list[dict[str, Any]] = []
    for key in validation_keys:
        group = training_groups.get(key)
        if group is None:
            rows = positives = 0
            objectives = {expert: math.nan for expert in EXPERTS}
            candidate = selected = global_expert
            reason = "unseen_unit"
        else:
            rows = len(group)
            positives = joint_positives(group)
            objectives = expert_losses(group)
            candidate = choose_expert(objectives)
            supported = rows >= minimum_rows and positives >= minimum_positives
            if not supported:
                selected = global_expert
                reason = "support_fallback"
            elif guarded and (
                objectives[global_expert] - objectives[candidate] < GUARD_MARGIN
            ):
                selected = global_expert
                reason = "margin_fallback"
            else:
                selected = candidate
                reason = "unit_selected"
        mapping[key] = selected
        ledger.append(
            {
                "validation_month": validation_month,
                "selection_months_json": json.dumps(SELECTION_MONTHS[validation_month]),
                "selector": selector,
                "level": selector,
                "unit_key": unit_key(columns, key),
                "cycle_id": key[columns.index("cycle_id")] if "cycle_id" in columns else "",
                "current_state": key[columns.index("current_state")] if "current_state" in columns else -1,
                "entry_clock_quartile": -1,
                "selection_rows": rows,
                "joint_positives_across_12_cells": positives,
                "supported": bool(rows >= minimum_rows and positives >= minimum_positives),
                "global_expert": global_expert,
                "parent_expert": global_expert,
                "candidate_expert": candidate,
                "selected_expert": selected,
                "candidate_log_loss": objectives[candidate],
                "parent_log_loss_on_unit": objectives[global_expert],
                "candidate_improvement_vs_parent": objectives[global_expert]
                - objectives[candidate],
                "decision_reason": reason,
                "is_final_assignment": True,
                **{f"objective__{expert}": objectives[expert] for expert in EXPERTS},
            }
        )
    return mapping, ledger


def grouped_training_frames(
    selection: pd.DataFrame, columns: Sequence[str]
) -> dict[tuple[Any, ...], pd.DataFrame]:
    return {
        normalized_key(key): group
        for key, group in selection.groupby(list(columns), sort=True)
    }


def hierarchical_assignments(
    selection: pd.DataFrame,
    validation: pd.DataFrame,
    validation_month: str,
    global_expert: str,
) -> tuple[dict[tuple[Any, ...], str], list[dict[str, Any]]]:
    levels = (
        ("loop", ("cycle_id",), SUPPORT["loop"]),
        ("loop_regime", ("cycle_id", "current_state"), SUPPORT["loop_regime"]),
        (
            "loop_regime_clock",
            ("cycle_id", "current_state", "entry_clock_quartile"),
            SUPPORT["loop_regime_clock"],
        ),
    )
    maps: dict[str, dict[tuple[Any, ...], str]] = {}
    ledger: list[dict[str, Any]] = []
    for level_index, (level, columns, support) in enumerate(levels):
        training_groups = grouped_training_frames(selection, columns)
        validation_keys = sorted(
            {
                tuple(row)
                for row in validation.loc[:, list(columns)].itertuples(index=False, name=None)
            },
            key=lambda item: tuple(str(value) for value in item),
        )
        mapping: dict[tuple[Any, ...], str] = {}
        for key in validation_keys:
            if level_index == 0:
                parent_expert = global_expert
            else:
                parent_columns = levels[level_index - 1][1]
                parent_key = tuple(key[columns.index(column)] for column in parent_columns)
                parent_expert = maps[levels[level_index - 1][0]].get(
                    parent_key, global_expert
                )
            group = training_groups.get(key)
            if group is None:
                rows = positives = 0
                objectives = {expert: math.nan for expert in EXPERTS}
                candidate = selected = parent_expert
                reason = "unseen_unit"
            else:
                rows = len(group)
                positives = joint_positives(group)
                objectives = expert_losses(group)
                candidate = choose_expert(objectives)
                supported = rows >= support[0] and positives >= support[1]
                if not supported:
                    selected = parent_expert
                    reason = "support_fallback"
                elif objectives[parent_expert] - objectives[candidate] < GUARD_MARGIN:
                    selected = parent_expert
                    reason = "margin_fallback"
                else:
                    selected = candidate
                    reason = "child_selected"
            mapping[key] = selected
            ledger.append(
                {
                    "validation_month": validation_month,
                    "selection_months_json": json.dumps(SELECTION_MONTHS[validation_month]),
                    "selector": "hierarchical_clock_best",
                    "level": level,
                    "unit_key": unit_key(columns, key),
                    "cycle_id": key[columns.index("cycle_id")],
                    "current_state": key[columns.index("current_state")]
                    if "current_state" in columns
                    else -1,
                    "entry_clock_quartile": key[columns.index("entry_clock_quartile")]
                    if "entry_clock_quartile" in columns
                    else -1,
                    "selection_rows": rows,
                    "joint_positives_across_12_cells": positives,
                    "supported": bool(rows >= support[0] and positives >= support[1]),
                    "global_expert": global_expert,
                    "parent_expert": parent_expert,
                    "candidate_expert": candidate,
                    "selected_expert": selected,
                    "candidate_log_loss": objectives[candidate],
                    "parent_log_loss_on_unit": objectives[parent_expert],
                    "candidate_improvement_vs_parent": objectives[parent_expert]
                    - objectives[candidate],
                    "decision_reason": reason,
                    "is_final_assignment": level == "loop_regime_clock",
                    **{f"objective__{expert}": objectives[expert] for expert in EXPERTS},
                }
            )
        maps[level] = mapping
    return maps["loop_regime_clock"], ledger


def map_experts(
    frame: pd.DataFrame,
    columns: Sequence[str],
    mapping: dict[tuple[Any, ...], str],
    fallback: str,
) -> np.ndarray:
    keys = frame.loc[:, list(columns)].itertuples(index=False, name=None)
    return np.asarray([mapping.get(tuple(key), fallback) for key in keys], dtype=object)


def copy_selected_probabilities(
    validation: pd.DataFrame, selector: str, selected: np.ndarray
) -> None:
    validation[selected_expert_column(selector)] = selected
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                output = np.empty(len(validation), dtype=float)
                for expert in EXPERTS:
                    mask = selected == expert
                    if mask.any():
                        output[mask] = validation.loc[
                            mask, expert_column(expert, target, horizon, tier)
                        ].to_numpy(float)
                validation[selector_column(selector, target, horizon, tier)] = output


def build_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    ledger_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for validation_month in VALIDATION_MONTHS:
        selection = frame.loc[frame["month"].isin(SELECTION_MONTHS[validation_month])]
        validation = frame.loc[frame["month"].eq(validation_month)].copy()
        global_objectives = expert_losses(selection)
        global_expert = choose_expert(global_objectives)
        global_rows.append(
            {
                "validation_month": validation_month,
                "selection_months_json": json.dumps(SELECTION_MONTHS[validation_month]),
                "selection_rows": len(selection),
                "joint_positives_across_12_cells": joint_positives(selection),
                "selected_expert": global_expert,
                **{
                    f"objective__{expert}": global_objectives[expert]
                    for expert in EXPERTS
                },
            }
        )
        global_selected = np.full(len(validation), global_expert, dtype=object)
        copy_selected_probabilities(validation, "global_best", global_selected)
        ledger_rows.append(
            {
                "validation_month": validation_month,
                "selection_months_json": json.dumps(SELECTION_MONTHS[validation_month]),
                "selector": "global_best",
                "level": "global",
                "unit_key": "global",
                "cycle_id": "",
                "current_state": -1,
                "entry_clock_quartile": -1,
                "selection_rows": len(selection),
                "joint_positives_across_12_cells": joint_positives(selection),
                "supported": True,
                "global_expert": global_expert,
                "parent_expert": global_expert,
                "candidate_expert": global_expert,
                "selected_expert": global_expert,
                "candidate_log_loss": global_objectives[global_expert],
                "parent_log_loss_on_unit": global_objectives[global_expert],
                "candidate_improvement_vs_parent": 0.0,
                "decision_reason": "global_selected",
                "is_final_assignment": True,
                **{
                    f"objective__{expert}": global_objectives[expert]
                    for expert in EXPERTS
                },
            }
        )
        for selector in (
            "regime_best",
            "loop_best",
            "loop_regime_best",
            "guarded_loop_regime_best",
        ):
            mapping, rows = flat_assignments(
                selection,
                validation,
                validation_month,
                selector,
                global_expert,
                guarded=selector == "guarded_loop_regime_best",
            )
            selected = map_experts(
                validation, GROUP_COLUMNS[selector], mapping, global_expert
            )
            copy_selected_probabilities(validation, selector, selected)
            ledger_rows.extend(rows)
        hierarchical, rows = hierarchical_assignments(
            selection, validation, validation_month, global_expert
        )
        selected = map_experts(
            validation,
            ("cycle_id", "current_state", "entry_clock_quartile"),
            hierarchical,
            global_expert,
        )
        copy_selected_probabilities(
            validation, "hierarchical_clock_best", selected
        )
        ledger_rows.extend(rows)
        outputs.append(validation)
    predictions = pd.concat(outputs, ignore_index=True)
    if len(predictions) != 51235:
        raise AssertionError("validation prediction surface changed")
    selector_probabilities = [
        selector_column(selector, target, horizon, tier)
        for selector in SELECTORS
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    values = predictions[selector_probabilities].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("invalid selector probabilities")
    return predictions, pd.DataFrame(ledger_rows), pd.DataFrame(global_rows)


def probability_for(
    frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str
) -> np.ndarray:
    if model in SELECTORS:
        column = selector_column(model, target, horizon, tier)
    else:
        column = expert_column(model, target, horizon, tier)
    return frame[column].to_numpy(float)


def calibration_summary(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, minimum_rows: int
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.sum() < minimum_rows or weights[selected].sum() <= 0:
            continue
        observed = weighted_mean(y[selected], weights[selected])
        predicted = weighted_mean(p[selected], weights[selected])
        supported.append((float(weights[selected].sum()), abs(observed - predicted)))
    if not supported:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in supported)
    ece = sum(weight * error for weight, error in supported) / total
    return float(ece), float(max(error for _, error in supported)), len(supported)


def cell_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weight_surface, weights in (
        ("inverse_compatible", frame["inverse_compatible_weight"].to_numpy(float)),
        ("unweighted", np.ones(len(frame))),
    ):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    y = frame[target_column(target, horizon, tier)].to_numpy(int)
                    references: dict[str, tuple[float, float, float, float, int]] = {}
                    for reference in ("baseline", "raw_full_link", "global_best"):
                        p = probability_for(frame, reference, target, horizon, tier)
                        log_loss, brier = binary_losses(y, p)
                        ece, maximum, bins = calibration_summary(y, p, weights, 250)
                        references[reference] = (
                            weighted_mean(log_loss, weights),
                            weighted_mean(brier, weights),
                            ece,
                            maximum,
                            bins,
                        )
                    for selector in SELECTORS:
                        p = probability_for(frame, selector, target, horizon, tier)
                        log_loss, brier = binary_losses(y, p)
                        ece, maximum, bins = calibration_summary(y, p, weights, 250)
                        ll = weighted_mean(log_loss, weights)
                        br = weighted_mean(brier, weights)
                        rows.append(
                            {
                                "weight_surface": weight_surface,
                                "selector": selector,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "positives": int(y.sum()),
                                "weight_sum": float(weights.sum()),
                                "log_loss": ll,
                                "baseline_log_loss": references["baseline"][0],
                                "raw_log_loss": references["raw_full_link"][0],
                                "global_log_loss": references["global_best"][0],
                                "log_loss_difference_vs_baseline": ll
                                - references["baseline"][0],
                                "log_loss_difference_vs_raw": ll
                                - references["raw_full_link"][0],
                                "log_loss_difference_vs_global": ll
                                - references["global_best"][0],
                                "brier": br,
                                "baseline_brier": references["baseline"][1],
                                "raw_brier": references["raw_full_link"][1],
                                "global_brier": references["global_best"][1],
                                "brier_difference_vs_baseline": br
                                - references["baseline"][1],
                                "brier_difference_vs_raw": br
                                - references["raw_full_link"][1],
                                "brier_difference_vs_global": br
                                - references["global_best"][1],
                                "ece": ece,
                                "raw_ece": references["raw_full_link"][2],
                                "maximum_supported_bin_error": maximum,
                                "raw_maximum_supported_bin_error": references[
                                    "raw_full_link"
                                ][3],
                                "supported_bins": bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled_metrics(frame: pd.DataFrame, selector: str) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    storage = {
        key: []
        for key in (
            "baseline_ll",
            "raw_ll",
            "global_ll",
            "candidate_ll",
            "baseline_br",
            "raw_br",
            "global_br",
            "candidate_br",
        )
    }
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[target_column(target, horizon, tier)].to_numpy(int)
                for model, prefix in (
                    ("baseline", "baseline"),
                    ("raw_full_link", "raw"),
                    ("global_best", "global"),
                    (selector, "candidate"),
                ):
                    ll, br = binary_losses(
                        y, probability_for(frame, model, target, horizon, tier)
                    )
                    storage[f"{prefix}_ll"].append(weighted_mean(ll, weights))
                    storage[f"{prefix}_br"].append(weighted_mean(br, weights))
    values = {key: float(np.mean(value)) for key, value in storage.items()}
    return {
        "baseline_log_loss": values["baseline_ll"],
        "raw_log_loss": values["raw_ll"],
        "global_log_loss": values["global_ll"],
        "log_loss": values["candidate_ll"],
        "relative_log_loss_improvement_vs_baseline": (
            values["baseline_ll"] - values["candidate_ll"]
        )
        / values["baseline_ll"],
        "relative_log_loss_improvement_vs_raw": (
            values["raw_ll"] - values["candidate_ll"]
        )
        / values["raw_ll"],
        "log_loss_difference_vs_baseline": values["candidate_ll"]
        - values["baseline_ll"],
        "log_loss_difference_vs_raw": values["candidate_ll"] - values["raw_ll"],
        "log_loss_difference_vs_global": values["candidate_ll"]
        - values["global_ll"],
        "baseline_brier": values["baseline_br"],
        "raw_brier": values["raw_br"],
        "global_brier": values["global_br"],
        "brier": values["candidate_br"],
        "brier_difference_vs_baseline": values["candidate_br"]
        - values["baseline_br"],
        "brier_difference_vs_raw": values["candidate_br"] - values["raw_br"],
        "brier_difference_vs_global": values["candidate_br"]
        - values["global_br"],
    }


def row_loss_difference(
    frame: pd.DataFrame, selector: str, comparison: str, endpoint: str
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[target_column(target, horizon, tier)].to_numpy(int)
                reference = binary_losses(
                    y, probability_for(frame, comparison, target, horizon, tier)
                )[0 if endpoint == "log_loss" else 1]
                candidate = binary_losses(
                    y, probability_for(frame, selector, target, horizon, tier)
                )[0 if endpoint == "log_loss" else 1]
                result += (candidate - reference) / 12.0
    return result


def daily_values(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame(
        {
            "session_date": frame["session_date"].to_numpy(),
            "weighted": values * weights,
            "weight": weights,
        }
    ).groupby("session_date", sort=True).sum()
    return (grouped["weighted"] / grouped["weight"]).to_numpy(float)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray(
        [
            values[index : index + 5].mean()
            for index in range(0, len(values), 5)
            if len(values[index : index + 5]) == 5
        ]
    )
    sampled = np.random.default_rng(seed).choice(
        blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip_p_value(values: np.ndarray, seed: int) -> float:
    null = (
        np.random.default_rng(seed).choice(
            np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))
        )
        @ values
    ) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm_adjust(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    families = output.groupby(list(groups)).groups
    for _, positions in families.items():
        ordered = sorted(
            list(positions), key=lambda position: output.loc[position, "p_value"]
        )
        running = 0.0
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(
                1.0,
                max(
                    running,
                    (len(ordered) - rank + 1)
                    * float(output.loc[position, "p_value"]),
                ),
            )
            running = adjusted
            output.loc[position, "holm_adjusted_p"] = adjusted
            output.loc[position, "holm_pass"] = adjusted <= 0.05
            output.loc[position, "holm_rank"] = rank
            output.loc[position, "family_size"] = len(ordered)
    output["holm_rank"] = output["holm_rank"].astype(int)
    output["family_size"] = output["family_size"].astype(int)
    return output


def top_three_recall(
    frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str
) -> float:
    selected = frame[["anchor_id", target_column(target, horizon, tier)]].copy()
    selected["probability"] = probability_for(frame, model, target, horizon, tier)
    selected = selected.sort_values(
        ["anchor_id", "probability"], ascending=[True, False], kind="stable"
    )
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    y = selected[target_column(target, horizon, tier)].to_numpy(int)
    return float(
        ((selected["rank"].to_numpy(int) <= 3) & (y == 1)).sum() / y.sum()
    )


def assignment_summary(
    predictions: pd.DataFrame, ledger: pd.DataFrame, globals_frame: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_by_month = globals_frame.set_index("validation_month")[
        "selected_expert"
    ].to_dict()
    final = ledger.loc[ledger["is_final_assignment"]].copy()
    for selector in SELECTORS:
        selector_ledger = final.loc[final["selector"].eq(selector)]
        month_maps: dict[str, dict[str, str]] = {}
        for month in VALIDATION_MONTHS:
            mapping = selector_ledger.loc[
                selector_ledger["validation_month"].eq(month)
            ].set_index("unit_key")["selected_expert"].to_dict()
            month_maps[month] = mapping
        shared = sorted(set(month_maps["2024-11"]) & set(month_maps["2024-12"]))
        stability = (
            float(
                np.mean(
                    [
                        month_maps["2024-11"][key]
                        == month_maps["2024-12"][key]
                        for key in shared
                    ]
                )
            )
            if shared
            else math.nan
        )
        selected_values = predictions[selected_expert_column(selector)].astype(str)
        global_values = predictions["month"].map(global_by_month).astype(str)
        counts = selected_values.value_counts().to_dict()
        rows.append(
            {
                "selector": selector,
                "distinct_experts_used": int(selected_values.nunique()),
                "individualized_row_fraction_vs_global": float(
                    (selected_values != global_values).mean()
                ),
                "shared_units": len(shared),
                "assignment_stability": stability,
                "final_assignment_units": len(selector_ledger),
                **{f"rows__{expert}": int(counts.get(expert, 0)) for expert in EXPERTS},
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    predictions: pd.DataFrame, ledger: pd.DataFrame, globals_frame: pd.DataFrame
) -> dict[str, Any]:
    cells = cell_metrics(predictions)
    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for selector_index, selector in enumerate(SELECTORS):
        pooled = {
            "selector": selector,
            "rows": len(predictions),
            "anchors": int(predictions["anchor_id"].nunique()),
            **pooled_metrics(predictions, selector),
        }
        for comparison_index, comparison in enumerate(COMPARISONS):
            for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
                values = daily_values(
                    predictions,
                    row_loss_difference(predictions, selector, comparison, endpoint),
                )
                seed = (
                    SEED
                    + selector_index * 1000
                    + comparison_index * 100
                    + endpoint_index
                )
                lower, upper = bootstrap_interval(values, seed)
                p_value = sign_flip_p_value(values, seed + 10)
                prefix = f"{comparison}__{endpoint}"
                pooled[f"{prefix}__daily_sessions"] = len(values)
                pooled[f"{prefix}__bootstrap_lower"] = lower
                pooled[f"{prefix}__bootstrap_upper"] = upper
                pooled[f"{prefix}__p_value"] = p_value
                if selector in CANDIDATE_SELECTORS:
                    multiplicity_rows.append(
                        {
                            "selector": selector,
                            "comparison": comparison,
                            "endpoint": endpoint,
                            "p_value": p_value,
                        }
                    )
        pooled_rows.append(pooled)
    pooled_frame = pd.DataFrame(pooled_rows)
    multiplicity = holm_adjust(
        pd.DataFrame(multiplicity_rows), ["comparison", "endpoint"]
    )

    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for month in VALIDATION_MONTHS:
            temporal_rows.append(
                {
                    "selector": selector,
                    "month": month,
                    **pooled_metrics(
                        predictions.loc[predictions["month"].eq(month)], selector
                    ),
                }
            )
        for symbol in sorted(predictions["symbol_norm"].unique()):
            stock_rows.append(
                {
                    "selector": selector,
                    "deleted_symbol": symbol,
                    **pooled_metrics(
                        predictions.loc[predictions["symbol_norm"].ne(symbol)],
                        selector,
                    ),
                }
            )
        for (cycle_id, current_state), selected in predictions.groupby(
            ["cycle_id", "current_state"], sort=True
        ):
            positives = joint_positives(selected)
            orientation_rows.append(
                {
                    "selector": selector,
                    "cycle_id": cycle_id,
                    "current_state": int(current_state),
                    "rows": len(selected),
                    "joint_positives_across_12_cells": positives,
                    "supported": len(selected) >= 250 and positives >= 15,
                    **pooled_metrics(selected, selector),
                }
            )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    raw = top_three_recall(
                        predictions, "raw_full_link", target, horizon, tier
                    )
                    value = top_three_recall(
                        predictions, selector, target, horizon, tier
                    )
                    ranking_rows.append(
                        {
                            "selector": selector,
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "top_three_recall": value,
                            "raw_top_three_recall": raw,
                            "gain_vs_raw": value - raw,
                        }
                    )
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    assignments = assignment_summary(predictions, ledger, globals_frame)

    primary_cells = cells.loc[cells["weight_surface"].eq("inverse_compatible")]
    gates: dict[str, Any] = {}
    for selector in CANDIDATE_SELECTORS:
        pool = pooled_frame.set_index("selector").loc[selector]
        cell = primary_cells.loc[primary_cells["selector"].eq(selector)]
        time = temporal.loc[temporal["selector"].eq(selector)]
        stock = stocks.loc[stocks["selector"].eq(selector)]
        orientation = orientations.loc[
            orientations["selector"].eq(selector) & orientations["supported"]
        ]
        rank = ranking.loc[ranking["selector"].eq(selector)]
        assignment = assignments.set_index("selector").loc[selector]
        multi = multiplicity.loc[multiplicity["selector"].eq(selector)]
        raw_multi = multi.loc[multi["comparison"].eq("raw_full_link")]
        checks = {
            "pooled_no_worse_than_raw": bool(
                pool["log_loss_difference_vs_raw"] <= 0
                and pool["brier_difference_vs_raw"] <= 0
            ),
            "bootstrap_vs_baseline_and_raw": bool(
                all(
                    pool[f"{comparison}__{endpoint}__bootstrap_upper"] <= 0
                    for comparison in ("baseline", "raw_full_link")
                    for endpoint in ("log_loss", "brier")
                )
            ),
            "Holm_both_raw_endpoints": bool(raw_multi["holm_pass"].all()),
            "all_cell_losses_vs_raw": bool(
                (cell["log_loss_difference_vs_raw"] <= 0).all()
                and (cell["brier_difference_vs_raw"] <= 0).all()
            ),
            "all_cell_calibration": bool(
                (cell["ece"] <= cell["raw_ece"]).all()
                and (cell["maximum_supported_bin_error"] <= 0.02).all()
            ),
            "both_months_vs_raw": bool(
                (time["log_loss_difference_vs_raw"] <= 0).all()
                and (time["brier_difference_vs_raw"] <= 0).all()
            ),
            "every_stock_vs_raw": bool(
                (stock["log_loss_difference_vs_raw"] <= 0).all()
                and (stock["brier_difference_vs_raw"] <= 0).all()
            ),
            "zero_orientation_reversals_vs_baseline": bool(
                (orientation["log_loss_difference_vs_baseline"] <= 0).all()
                and (orientation["brier_difference_vs_baseline"] <= 0).all()
            ),
            "ranking_vs_raw": bool((rank["gain_vs_raw"] >= 0).all()),
            "at_least_two_experts_used": bool(
                assignment["distinct_experts_used"] >= 2
            ),
            "assignment_stability": bool(
                assignment["assignment_stability"] >= 0.5
            ),
        }
        gates[selector] = {
            "pass": bool(all(checks.values())),
            "checks": checks,
            "cell_loss_reversals_vs_raw": int(
                (
                    (cell["log_loss_difference_vs_raw"] > 0)
                    | (cell["brier_difference_vs_raw"] > 0)
                ).sum()
            ),
            "calibration_failures": int(
                (
                    (cell["ece"] > cell["raw_ece"])
                    | (cell["maximum_supported_bin_error"] > 0.02)
                ).sum()
            ),
            "supported_orientations": len(orientation),
            "orientation_log_loss_reversals_vs_baseline": int(
                (orientation["log_loss_difference_vs_baseline"] > 0).sum()
            ),
            "orientation_brier_reversals_vs_baseline": int(
                (orientation["brier_difference_vs_baseline"] > 0).sum()
            ),
            "ranking_degradations_vs_raw": int((rank["gain_vs_raw"] < 0).sum()),
            "distinct_experts_used": int(assignment["distinct_experts_used"]),
            "individualized_row_fraction_vs_global": float(
                assignment["individualized_row_fraction_vs_global"]
            ),
            "assignment_stability": float(assignment["assignment_stability"]),
        }
    passing = [selector for selector in CANDIDATE_SELECTORS if gates[selector]["pass"]]
    priority = (
        "guarded_loop_regime_best",
        "hierarchical_clock_best",
        "loop_regime_best",
        "loop_best",
        "regime_best",
    )
    selected_selector = next(
        (selector for selector in priority if selector in passing), None
    )
    decision = {
        "label": (
            "development_individual_expert_selector_pending_unseen_validation"
            if selected_selector
            else "individual_expert_selectors_rejected_or_unconfirmed"
        ),
        "passing_selectors": passing,
        "selected_selector": selected_selector,
        "named_loop_good_or_high_promoted": False,
        "raw_link_retained_as_diagnostic_only": selected_selector is None,
        "later_period_scoring_performed": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return {
        "cells": cells,
        "pooled": pooled_frame,
        "multiplicity": multiplicity,
        "temporal": temporal,
        "stocks": stocks,
        "orientations": orientations,
        "ranking": ranking,
        "assignments": assignments,
        "gates": gates,
        "decision": decision,
    }


def artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    return {
        "files": {
            name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
            for name in sorted(names)
        }
    }


def run() -> None:
    contract, hashes = verify_contract_and_sources()
    frame, source_audit = load_source()
    predictions, ledger, globals_frame = build_predictions(frame)
    results = evaluate(predictions, ledger, globals_frame)
    ROOT.mkdir(parents=True, exist_ok=False)
    write_json(ROOT / "source_audit.json", source_audit)
    predictions.to_parquet(ROOT / "selector_predictions_2024_nov_dec.parquet", index=False)
    ledger.to_csv(ROOT / "assignment_ledger.csv", index=False)
    globals_frame.to_csv(ROOT / "global_objectives.csv", index=False)
    results["cells"].to_csv(ROOT / "cell_metrics.csv", index=False)
    results["pooled"].to_csv(ROOT / "pooled_metrics.csv", index=False)
    results["multiplicity"].to_csv(ROOT / "multiplicity.csv", index=False)
    results["temporal"].to_csv(ROOT / "temporal_slices.csv", index=False)
    results["stocks"].to_csv(ROOT / "stock_deletions.csv", index=False)
    results["orientations"].to_csv(ROOT / "orientation_slices.csv", index=False)
    results["ranking"].to_csv(ROOT / "ranking.csv", index=False)
    results["assignments"].to_csv(ROOT / "assignment_summary.csv", index=False)
    write_json(ROOT / "selector_gates.json", results["gates"])
    write_json(ROOT / "decision.json", results["decision"])
    summary = {
        "contract_id": contract["contract_id"],
        "contract_sha256": hashes["contract"],
        "runner_sha256": sha256(Path(__file__)),
        "scientific_status": contract["scientific_status"],
        "source_hashes": hashes,
        "source_audit": source_audit,
        "validation_rows": len(predictions),
        "validation_anchors": int(predictions["anchor_id"].nunique()),
        "validation_months": list(VALIDATION_MONTHS),
        "expert_count": len(EXPERTS),
        "selector_count": len(SELECTORS),
        "candidate_selector_count": len(CANDIDATE_SELECTORS),
        "selector_pass": {
            selector: results["gates"][selector]["pass"]
            for selector in CANDIDATE_SELECTORS
        },
        "decision": results["decision"],
        "direct_volume_fields_used": [],
        "volume_label": "historical_volume_not_used",
        "direction_or_signed_return_used": False,
        "later_period_scoring_performed": False,
        "prospective_shadow_read_or_write_performed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(ROOT / "summary.json", summary)
    names = [
        "source_audit.json",
        "selector_predictions_2024_nov_dec.parquet",
        "assignment_ledger.csv",
        "global_objectives.csv",
        "cell_metrics.csv",
        "pooled_metrics.csv",
        "multiplicity.csv",
        "temporal_slices.csv",
        "stock_deletions.csv",
        "orientation_slices.csv",
        "ranking.csv",
        "assignment_summary.csv",
        "selector_gates.json",
        "decision.json",
        "summary.json",
    ]
    write_json(ROOT / "artifact_manifest.json", artifact_manifest(ROOT, names))
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


def self_test() -> None:
    y = np.asarray([0, 1])
    ll, brier = binary_losses(y, np.asarray([0.25, 0.75]))
    assert np.allclose(ll, -np.log(0.75))
    assert np.allclose(brier, 0.0625)
    assert choose_expert({expert: 1.0 for expert in EXPERTS}) == "baseline"
    losses = {expert: 1.0 for expert in EXPERTS}
    losses["raw_full_link"] = 0.9
    assert choose_expert(losses) == "raw_full_link"
    assert unit_key(("cycle_id", "current_state"), ("cycle_01", 1)) == (
        "cycle_id=cycle_01|current_state=1"
    )


if __name__ == "__main__":
    self_test()
    run()
