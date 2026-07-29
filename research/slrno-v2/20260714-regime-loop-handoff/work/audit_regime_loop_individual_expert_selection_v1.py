"""Independent replay audit for individual regime-loop expert selection V1.

This auditor does not import the production runner.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-individual-expert-selection-v1.json"
RUNNER = HERE / "run_regime_loop_individual_expert_selection_v1.py"
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
    "runner": "7f95e30126cc9c207a589dd00ad64c1ca02cc8d52be5ce2ffff2d73e87a44ad5",
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
CANDIDATES = SELECTORS[1:]
COMPARISONS = ("baseline", "raw_full_link", "global_best")
GROUP_COLUMNS = {
    "regime_best": ("current_state",),
    "loop_best": ("cycle_id",),
    "loop_regime_best": ("cycle_id", "current_state"),
    "guarded_loop_regime_best": ("cycle_id", "current_state"),
}
SUPPORT = {
    "regime_best": (500, 30),
    "loop_best": (500, 30),
    "loop_regime_best": (300, 24),
    "guarded_loop_regime_best": (300, 24),
    "loop": (500, 30),
    "loop_regime": (300, 24),
    "loop_regime_clock": (150, 12),
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


def ycol(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def ecol(expert: str, target: str, horizon: int, tier: str) -> str:
    return f"link__{expert}__{target}__h{horizon}__{tier}"


def scol(selector: str, target: str, horizon: int, tier: str) -> str:
    return f"selector__{selector}__{target}__h{horizon}__{tier}"


def xcol(selector: str) -> str:
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
        *[ycol(t, h, q) for t in TARGETS for h in HORIZONS for q in TIERS],
        *[
            ecol(e, t, h, q)
            for e in EXPERTS
            for t in TARGETS
            for h in HORIZONS
            for q in TIERS
        ],
    ]


def binary_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def load_source() -> pd.DataFrame:
    frame = pd.read_parquet(SOURCE, columns=source_columns())
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    for expert in EXPERTS:
        values = np.zeros(len(frame))
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    values += binary_losses(
                        frame[ycol(target, horizon, tier)].to_numpy(int),
                        frame[ecol(expert, target, horizon, tier)].to_numpy(float),
                    )[0] / 12.0
        frame[f"__expert_loss__{expert}"] = values
    return frame


def objectives(frame: pd.DataFrame) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    return {
        expert: wmean(frame[f"__expert_loss__{expert}"].to_numpy(float), weights)
        for expert in EXPERTS
    }


def choose(losses: dict[str, float]) -> str:
    best = min(value for value in losses.values() if np.isfinite(value))
    tied = {
        expert
        for expert, value in losses.items()
        if np.isfinite(value) and value <= best + TIE_TOLERANCE
    }
    return next(expert for expert in TIE_PRIORITY if expert in tied)


def positives(frame: pd.DataFrame) -> int:
    return int(sum(frame[ycol(t, h, q)].sum() for t in TARGETS for h in HORIZONS for q in TIERS))


def key_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def key_text(columns: Sequence[str], values: Sequence[Any]) -> str:
    return "|".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))


def flat_map(
    selection: pd.DataFrame,
    validation: pd.DataFrame,
    month: str,
    selector: str,
    global_expert: str,
) -> tuple[dict[tuple[Any, ...], str], list[dict[str, Any]]]:
    columns = GROUP_COLUMNS[selector]
    minimum = SUPPORT[selector]
    training = {
        key_tuple(key): group
        for key, group in selection.groupby(list(columns), sort=True)
    }
    validation_keys = sorted(
        {tuple(row) for row in validation[list(columns)].itertuples(index=False, name=None)},
        key=lambda row: tuple(str(value) for value in row),
    )
    mapping: dict[tuple[Any, ...], str] = {}
    rows: list[dict[str, Any]] = []
    for key in validation_keys:
        group = training.get(key)
        if group is None:
            count = positive_count = 0
            losses = {expert: math.nan for expert in EXPERTS}
            candidate = selected = global_expert
            reason = "unseen_unit"
        else:
            count, positive_count = len(group), positives(group)
            losses = objectives(group)
            candidate = choose(losses)
            if count < minimum[0] or positive_count < minimum[1]:
                selected, reason = global_expert, "support_fallback"
            elif selector == "guarded_loop_regime_best" and losses[global_expert] - losses[candidate] < GUARD_MARGIN:
                selected, reason = global_expert, "margin_fallback"
            else:
                selected, reason = candidate, "unit_selected"
        mapping[key] = selected
        rows.append(
            assignment_row(
                month,
                selector,
                selector,
                columns,
                key,
                count,
                positive_count,
                count >= minimum[0] and positive_count >= minimum[1],
                global_expert,
                global_expert,
                candidate,
                selected,
                losses,
                reason,
                True,
            )
        )
    return mapping, rows


def assignment_row(
    month: str,
    selector: str,
    level: str,
    columns: Sequence[str],
    key: Sequence[Any],
    rows: int,
    positive_count: int,
    supported: bool,
    global_expert: str,
    parent: str,
    candidate: str,
    selected: str,
    losses: dict[str, float],
    reason: str,
    final: bool,
) -> dict[str, Any]:
    return {
        "validation_month": month,
        "selection_months_json": json.dumps(SELECTION_MONTHS[month]),
        "selector": selector,
        "level": level,
        "unit_key": key_text(columns, key),
        "cycle_id": key[columns.index("cycle_id")] if "cycle_id" in columns else "",
        "current_state": key[columns.index("current_state")] if "current_state" in columns else -1,
        "entry_clock_quartile": key[columns.index("entry_clock_quartile")] if "entry_clock_quartile" in columns else -1,
        "selection_rows": rows,
        "joint_positives_across_12_cells": positive_count,
        "supported": bool(supported),
        "global_expert": global_expert,
        "parent_expert": parent,
        "candidate_expert": candidate,
        "selected_expert": selected,
        "candidate_log_loss": losses[candidate],
        "parent_log_loss_on_unit": losses[parent],
        "candidate_improvement_vs_parent": losses[parent] - losses[candidate],
        "decision_reason": reason,
        "is_final_assignment": bool(final),
        **{f"objective__{expert}": losses[expert] for expert in EXPERTS},
    }


def hierarchical_map(
    selection: pd.DataFrame,
    validation: pd.DataFrame,
    month: str,
    global_expert: str,
) -> tuple[dict[tuple[Any, ...], str], list[dict[str, Any]]]:
    levels = (
        ("loop", ("cycle_id",)),
        ("loop_regime", ("cycle_id", "current_state")),
        ("loop_regime_clock", ("cycle_id", "current_state", "entry_clock_quartile")),
    )
    maps: dict[str, dict[tuple[Any, ...], str]] = {}
    rows: list[dict[str, Any]] = []
    for index, (level, columns) in enumerate(levels):
        training = {
            key_tuple(key): group
            for key, group in selection.groupby(list(columns), sort=True)
        }
        validation_keys = sorted(
            {tuple(row) for row in validation[list(columns)].itertuples(index=False, name=None)},
            key=lambda row: tuple(str(value) for value in row),
        )
        mapping: dict[tuple[Any, ...], str] = {}
        minimum = SUPPORT[level]
        for key in validation_keys:
            if index == 0:
                parent = global_expert
            else:
                parent_level, parent_columns = levels[index - 1]
                parent_key = tuple(key[columns.index(column)] for column in parent_columns)
                parent = maps[parent_level].get(parent_key, global_expert)
            group = training.get(key)
            if group is None:
                count = positive_count = 0
                losses = {expert: math.nan for expert in EXPERTS}
                candidate = selected = parent
                reason = "unseen_unit"
            else:
                count, positive_count = len(group), positives(group)
                losses = objectives(group)
                candidate = choose(losses)
                if count < minimum[0] or positive_count < minimum[1]:
                    selected, reason = parent, "support_fallback"
                elif losses[parent] - losses[candidate] < GUARD_MARGIN:
                    selected, reason = parent, "margin_fallback"
                else:
                    selected, reason = candidate, "child_selected"
            mapping[key] = selected
            rows.append(
                assignment_row(
                    month,
                    "hierarchical_clock_best",
                    level,
                    columns,
                    key,
                    count,
                    positive_count,
                    count >= minimum[0] and positive_count >= minimum[1],
                    global_expert,
                    parent,
                    candidate,
                    selected,
                    losses,
                    reason,
                    level == "loop_regime_clock",
                )
            )
        maps[level] = mapping
    return maps["loop_regime_clock"], rows


def map_rows(
    frame: pd.DataFrame,
    columns: Sequence[str],
    mapping: dict[tuple[Any, ...], str],
    fallback: str,
) -> np.ndarray:
    return np.asarray(
        [
            mapping.get(tuple(row), fallback)
            for row in frame[list(columns)].itertuples(index=False, name=None)
        ],
        dtype=object,
    )


def replay_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    assignment_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for month in VALIDATION_MONTHS:
        selection = frame.loc[frame["month"].isin(SELECTION_MONTHS[month])]
        validation = frame.loc[frame["month"].eq(month)].copy()
        global_losses = objectives(selection)
        global_expert = choose(global_losses)
        global_rows.append(
            {
                "validation_month": month,
                "selection_months_json": json.dumps(SELECTION_MONTHS[month]),
                "selection_rows": len(selection),
                "joint_positives_across_12_cells": positives(selection),
                "selected_expert": global_expert,
                **{f"objective__{expert}": global_losses[expert] for expert in EXPERTS},
            }
        )
        selected_by_selector: dict[str, np.ndarray] = {
            "global_best": np.full(len(validation), global_expert, dtype=object)
        }
        assignment_rows.append(
            assignment_row(
                month,
                "global_best",
                "global",
                (),
                (),
                len(selection),
                positives(selection),
                True,
                global_expert,
                global_expert,
                global_expert,
                global_expert,
                global_losses,
                "global_selected",
                True,
            )
            | {"unit_key": "global"}
        )
        for selector in (
            "regime_best",
            "loop_best",
            "loop_regime_best",
            "guarded_loop_regime_best",
        ):
            mapping, rows = flat_map(selection, validation, month, selector, global_expert)
            selected_by_selector[selector] = map_rows(
                validation, GROUP_COLUMNS[selector], mapping, global_expert
            )
            assignment_rows.extend(rows)
        mapping, rows = hierarchical_map(selection, validation, month, global_expert)
        selected_by_selector["hierarchical_clock_best"] = map_rows(
            validation,
            ("cycle_id", "current_state", "entry_clock_quartile"),
            mapping,
            global_expert,
        )
        assignment_rows.extend(rows)
        for selector, selected in selected_by_selector.items():
            validation[xcol(selector)] = selected
            for target in TARGETS:
                for horizon in HORIZONS:
                    for tier in TIERS:
                        values = np.empty(len(validation))
                        for expert in EXPERTS:
                            mask = selected == expert
                            values[mask] = validation.loc[
                                mask, ecol(expert, target, horizon, tier)
                            ].to_numpy(float)
                        validation[scol(selector, target, horizon, tier)] = values
        outputs.append(validation)
    return (
        pd.concat(outputs, ignore_index=True),
        pd.DataFrame(assignment_rows),
        pd.DataFrame(global_rows),
    )


def probability(frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str) -> np.ndarray:
    column = scol(model, target, horizon, tier) if model in SELECTORS else ecol(model, target, horizon, tier)
    return frame[column].to_numpy(float)


def calibration(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for index in range(10):
        chosen = bins == index
        if chosen.sum() < 250 or weights[chosen].sum() <= 0:
            continue
        error = abs(wmean(y[chosen], weights[chosen]) - wmean(p[chosen], weights[chosen]))
        supported.append((float(weights[chosen].sum()), error))
    if not supported:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in supported)
    return (
        float(sum(weight * error for weight, error in supported) / total),
        float(max(error for _, error in supported)),
        len(supported),
    )


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface, weights in (
        ("inverse_compatible", frame["inverse_compatible_weight"].to_numpy(float)),
        ("unweighted", np.ones(len(frame))),
    ):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    y = frame[ycol(target, horizon, tier)].to_numpy(int)
                    refs: dict[str, tuple[float, float, float, float, int]] = {}
                    for reference in ("baseline", "raw_full_link", "global_best"):
                        p = probability(frame, reference, target, horizon, tier)
                        ll, br = binary_losses(y, p)
                        ece, maximum, bins = calibration(y, p, weights)
                        refs[reference] = (wmean(ll, weights), wmean(br, weights), ece, maximum, bins)
                    for selector in SELECTORS:
                        p = probability(frame, selector, target, horizon, tier)
                        ll_values, br_values = binary_losses(y, p)
                        ece, maximum, bins = calibration(y, p, weights)
                        ll, br = wmean(ll_values, weights), wmean(br_values, weights)
                        rows.append(
                            {
                                "weight_surface": surface,
                                "selector": selector,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "positives": int(y.sum()),
                                "weight_sum": float(weights.sum()),
                                "log_loss": ll,
                                "baseline_log_loss": refs["baseline"][0],
                                "raw_log_loss": refs["raw_full_link"][0],
                                "global_log_loss": refs["global_best"][0],
                                "log_loss_difference_vs_baseline": ll - refs["baseline"][0],
                                "log_loss_difference_vs_raw": ll - refs["raw_full_link"][0],
                                "log_loss_difference_vs_global": ll - refs["global_best"][0],
                                "brier": br,
                                "baseline_brier": refs["baseline"][1],
                                "raw_brier": refs["raw_full_link"][1],
                                "global_brier": refs["global_best"][1],
                                "brier_difference_vs_baseline": br - refs["baseline"][1],
                                "brier_difference_vs_raw": br - refs["raw_full_link"][1],
                                "brier_difference_vs_global": br - refs["global_best"][1],
                                "ece": ece,
                                "raw_ece": refs["raw_full_link"][2],
                                "maximum_supported_bin_error": maximum,
                                "raw_maximum_supported_bin_error": refs["raw_full_link"][3],
                                "supported_bins": bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled(frame: pd.DataFrame, selector: str) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    values = {name: [] for name in ("base_ll", "raw_ll", "global_ll", "cand_ll", "base_br", "raw_br", "global_br", "cand_br")}
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                for model, prefix in (("baseline", "base"), ("raw_full_link", "raw"), ("global_best", "global"), (selector, "cand")):
                    ll, br = binary_losses(y, probability(frame, model, target, horizon, tier))
                    values[f"{prefix}_ll"].append(wmean(ll, weights))
                    values[f"{prefix}_br"].append(wmean(br, weights))
    mean = {name: float(np.mean(items)) for name, items in values.items()}
    return {
        "baseline_log_loss": mean["base_ll"],
        "raw_log_loss": mean["raw_ll"],
        "global_log_loss": mean["global_ll"],
        "log_loss": mean["cand_ll"],
        "relative_log_loss_improvement_vs_baseline": (mean["base_ll"] - mean["cand_ll"]) / mean["base_ll"],
        "relative_log_loss_improvement_vs_raw": (mean["raw_ll"] - mean["cand_ll"]) / mean["raw_ll"],
        "log_loss_difference_vs_baseline": mean["cand_ll"] - mean["base_ll"],
        "log_loss_difference_vs_raw": mean["cand_ll"] - mean["raw_ll"],
        "log_loss_difference_vs_global": mean["cand_ll"] - mean["global_ll"],
        "baseline_brier": mean["base_br"],
        "raw_brier": mean["raw_br"],
        "global_brier": mean["global_br"],
        "brier": mean["cand_br"],
        "brier_difference_vs_baseline": mean["cand_br"] - mean["base_br"],
        "brier_difference_vs_raw": mean["cand_br"] - mean["raw_br"],
        "brier_difference_vs_global": mean["cand_br"] - mean["global_br"],
    }


def row_difference(frame: pd.DataFrame, selector: str, comparison: str, endpoint: str) -> np.ndarray:
    values = np.zeros(len(frame))
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                reference = binary_losses(y, probability(frame, comparison, target, horizon, tier))[0 if endpoint == "log_loss" else 1]
                candidate = binary_losses(y, probability(frame, selector, target, horizon, tier))[0 if endpoint == "log_loss" else 1]
                values += (candidate - reference) / 12.0
    return values


def daily(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame({"date": frame["session_date"], "weighted": values * weights, "weight": weights}).groupby("date", sort=True).sum()
    return (grouped["weighted"] / grouped["weight"]).to_numpy(float)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray([values[index:index + 5].mean() for index in range(0, len(values), 5) if len(values[index:index + 5]) == 5])
    sampled = np.random.default_rng(seed).choice(blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    null = (np.random.default_rng(seed).choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))) @ values) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    for _, positions in output.groupby(list(groups)).groups.items():
        ordered = sorted(list(positions), key=lambda position: output.loc[position, "p_value"])
        running = 0.0
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(1.0, max(running, (len(ordered) - rank + 1) * output.loc[position, "p_value"]))
            running = adjusted
            output.loc[position, "holm_adjusted_p"] = adjusted
            output.loc[position, "holm_pass"] = adjusted <= 0.05
            output.loc[position, "holm_rank"] = rank
            output.loc[position, "family_size"] = len(ordered)
    output["holm_rank"] = output["holm_rank"].astype(int)
    output["family_size"] = output["family_size"].astype(int)
    return output


def recall(frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str) -> float:
    selected = frame[["anchor_id", ycol(target, horizon, tier)]].copy()
    selected["p"] = probability(frame, model, target, horizon, tier)
    selected = selected.sort_values(["anchor_id", "p"], ascending=[True, False], kind="stable")
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    y = selected[ycol(target, horizon, tier)].to_numpy(int)
    return float(((selected["rank"].to_numpy(int) <= 3) & (y == 1)).sum() / y.sum())


def assignment_summary(predictions: pd.DataFrame, ledger: pd.DataFrame, globals_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_by_month = globals_frame.set_index("validation_month")["selected_expert"].to_dict()
    final = ledger.loc[ledger["is_final_assignment"]]
    for selector in SELECTORS:
        selected_ledger = final.loc[final["selector"].eq(selector)]
        maps = {
            month: selected_ledger.loc[selected_ledger["validation_month"].eq(month)].set_index("unit_key")["selected_expert"].to_dict()
            for month in VALIDATION_MONTHS
        }
        shared = sorted(set(maps["2024-11"]) & set(maps["2024-12"]))
        stability = float(np.mean([maps["2024-11"][key] == maps["2024-12"][key] for key in shared])) if shared else math.nan
        selected = predictions[xcol(selector)].astype(str)
        global_values = predictions["month"].map(global_by_month).astype(str)
        counts = selected.value_counts().to_dict()
        rows.append(
            {
                "selector": selector,
                "distinct_experts_used": int(selected.nunique()),
                "individualized_row_fraction_vs_global": float((selected != global_values).mean()),
                "shared_units": len(shared),
                "assignment_stability": stability,
                "final_assignment_units": len(selected_ledger),
                **{f"rows__{expert}": int(counts.get(expert, 0)) for expert in EXPERTS},
            }
        )
    return pd.DataFrame(rows)


def frame_comparison(calculated: pd.DataFrame, stored: pd.DataFrame, keys: Sequence[str], tolerance: float = 2e-11) -> tuple[bool, float, int]:
    joined = calculated.merge(stored, on=list(keys), suffixes=("__c", "__s"), validate="one_to_one")
    if len(joined) != len(calculated) or len(joined) != len(stored):
        return False, math.inf, 1
    maximum, errors = 0.0, 0
    common = [column for column in calculated.columns if column not in keys and column in stored.columns]
    for column in common:
        left, right = joined[f"{column}__c"], joined[f"{column}__s"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
            errors += int((~(finite | same_inf)).sum())
            if finite.any():
                difference = np.abs(a[finite] - b[finite])
                maximum = max(maximum, float(difference.max()))
                errors += int((difference > tolerance).sum())
        else:
            errors += int((left.fillna("").astype(str) != right.fillna("").astype(str)).sum())
    return errors == 0, maximum, errors


def run_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    hashes = {
        "contract": sha256(CONTRACT),
        "runner": sha256(RUNNER),
        "source": sha256(SOURCE),
        "source_audit": sha256(SOURCE_AUDIT),
        "source_contract": sha256(SOURCE_CONTRACT),
        "source_runner": sha256(SOURCE_RUNNER),
    }
    record("frozen_hashes", hashes == EXPECTED_HASHES, hashes)
    contract = json.loads(CONTRACT.read_text())
    record(
        "safety_labels",
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled",
        {key: contract[key] for key in ("research_only", "live_ordering_enabled", "order_placement")},
    )
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())["files"]
    mismatches = {
        name: sha256(ROOT / name)
        for name, descriptor in manifest.items()
        if sha256(ROOT / name) != descriptor["sha256"]
    }
    record("artifact_hashes", not mismatches, mismatches)

    source = load_source()
    replay, assignments, globals_frame = replay_predictions(source)
    stored = pd.read_parquet(ROOT / "selector_predictions_2024_nov_dec.parquet")
    key_equal = replay[["anchor_id", "cycle_index"]].equals(stored[["anchor_id", "cycle_index"]])
    record("prediction_surface_keys", key_equal and len(stored) == 51235, {"keys": key_equal, "rows": len(stored)})
    expert_errors = []
    for selector in SELECTORS:
        expert_errors.append(int((replay[xcol(selector)].astype(str) != stored[xcol(selector)].astype(str)).sum()))
    probability_error = max(
        float(np.max(np.abs(replay[scol(selector, target, horizon, tier)].to_numpy(float) - stored[scol(selector, target, horizon, tier)].to_numpy(float))))
        for selector in SELECTORS
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    )
    record("expert_assignment_prediction_replay", sum(expert_errors) == 0 and probability_error == 0, {"expert_errors": sum(expert_errors), "probability_error": probability_error})

    stored_assignments = pd.read_csv(ROOT / "assignment_ledger.csv")
    passed, maximum, errors = frame_comparison(assignments, stored_assignments, ["validation_month", "selector", "level", "unit_key"])
    record("assignment_ledger_replay", passed, {"max": maximum, "errors": errors})
    stored_globals = pd.read_csv(ROOT / "global_objectives.csv")
    passed, maximum, errors = frame_comparison(globals_frame, stored_globals, ["validation_month"])
    record("global_objective_replay", passed, {"max": maximum, "errors": errors})
    chronology = all(
        all(selection_month < row.validation_month for selection_month in json.loads(row.selection_months_json))
        for row in assignments.itertuples(index=False)
    )
    record("strict_selection_chronology", chronology, None)

    calculated_cells = cell_table(stored)
    stored_cells = pd.read_csv(ROOT / "cell_metrics.csv")
    passed, maximum, errors = frame_comparison(calculated_cells, stored_cells, ["weight_surface", "selector", "target", "horizon", "tier"])
    record("cell_metric_replay", passed, {"max": maximum, "errors": errors})

    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for selector_index, selector in enumerate(SELECTORS):
        row = {"selector": selector, "rows": len(stored), "anchors": stored["anchor_id"].nunique(), **pooled(stored, selector)}
        for comparison_index, comparison in enumerate(COMPARISONS):
            for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
                values = daily(stored, row_difference(stored, selector, comparison, endpoint))
                seed = SEED + selector_index * 1000 + comparison_index * 100 + endpoint_index
                lower, upper = bootstrap(values, seed)
                p_value = sign_flip(values, seed + 10)
                prefix = f"{comparison}__{endpoint}"
                row[f"{prefix}__daily_sessions"] = len(values)
                row[f"{prefix}__bootstrap_lower"] = lower
                row[f"{prefix}__bootstrap_upper"] = upper
                row[f"{prefix}__p_value"] = p_value
                if selector in CANDIDATES:
                    multiplicity_rows.append({"selector": selector, "comparison": comparison, "endpoint": endpoint, "p_value": p_value})
        pooled_rows.append(row)
    calculated_pooled = pd.DataFrame(pooled_rows)
    stored_pooled = pd.read_csv(ROOT / "pooled_metrics.csv")
    passed, maximum, errors = frame_comparison(calculated_pooled, stored_pooled, ["selector"])
    record("pooled_bootstrap_replay", passed, {"max": maximum, "errors": errors})
    calculated_holm = holm(pd.DataFrame(multiplicity_rows), ["comparison", "endpoint"])
    stored_holm = pd.read_csv(ROOT / "multiplicity.csv")
    passed, maximum, errors = frame_comparison(calculated_holm, stored_holm, ["selector", "comparison", "endpoint"])
    record("Holm_replay", passed, {"max": maximum, "errors": errors})

    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for month in VALIDATION_MONTHS:
            temporal_rows.append({"selector": selector, "month": month, **pooled(stored.loc[stored["month"].eq(month)], selector)})
        for symbol in sorted(stored["symbol_norm"].unique()):
            stock_rows.append({"selector": selector, "deleted_symbol": symbol, **pooled(stored.loc[stored["symbol_norm"].ne(symbol)], selector)})
        for (cycle, state), selected in stored.groupby(["cycle_id", "current_state"], sort=True):
            positive_count = positives(selected)
            orientation_rows.append({"selector": selector, "cycle_id": cycle, "current_state": int(state), "rows": len(selected), "joint_positives_across_12_cells": positive_count, "supported": len(selected) >= 250 and positive_count >= 15, **pooled(selected, selector)})
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    raw = recall(stored, "raw_full_link", target, horizon, tier)
                    value = recall(stored, selector, target, horizon, tier)
                    ranking_rows.append({"selector": selector, "target": target, "horizon": horizon, "tier": tier, "top_three_recall": value, "raw_top_three_recall": raw, "gain_vs_raw": value - raw})
    tables = [
        ("temporal_replay", pd.DataFrame(temporal_rows), pd.read_csv(ROOT / "temporal_slices.csv"), ["selector", "month"]),
        ("stock_replay", pd.DataFrame(stock_rows), pd.read_csv(ROOT / "stock_deletions.csv"), ["selector", "deleted_symbol"]),
        ("orientation_replay", pd.DataFrame(orientation_rows), pd.read_csv(ROOT / "orientation_slices.csv"), ["selector", "cycle_id", "current_state"]),
        ("ranking_replay", pd.DataFrame(ranking_rows), pd.read_csv(ROOT / "ranking.csv"), ["selector", "target", "horizon", "tier"]),
    ]
    for name, calculated, saved, keys in tables:
        passed, maximum, errors = frame_comparison(calculated, saved, keys)
        record(name, passed, {"max": maximum, "errors": errors})

    calculated_assignments = assignment_summary(stored, assignments, globals_frame)
    stored_assignment_summary = pd.read_csv(ROOT / "assignment_summary.csv")
    passed, maximum, errors = frame_comparison(calculated_assignments, stored_assignment_summary, ["selector"])
    record("assignment_summary_replay", passed, {"max": maximum, "errors": errors})

    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    primary_cells = calculated_cells.loc[calculated_cells["weight_surface"].eq("inverse_compatible")]
    saved_gates = json.loads((ROOT / "selector_gates.json").read_text())
    gate_errors: list[str] = []
    for selector in CANDIDATES:
        pool = calculated_pooled.set_index("selector").loc[selector]
        cell = primary_cells.loc[primary_cells["selector"].eq(selector)]
        time = temporal.loc[temporal["selector"].eq(selector)]
        stock = stocks.loc[stocks["selector"].eq(selector)]
        orientation = orientations.loc[orientations["selector"].eq(selector) & orientations["supported"]]
        rank = ranking.loc[ranking["selector"].eq(selector)]
        assignment = calculated_assignments.set_index("selector").loc[selector]
        raw_multi = calculated_holm.loc[(calculated_holm["selector"].eq(selector)) & (calculated_holm["comparison"].eq("raw_full_link"))]
        gate_checks = {
            "pooled_no_worse_than_raw": bool(pool["log_loss_difference_vs_raw"] <= 0 and pool["brier_difference_vs_raw"] <= 0),
            "bootstrap_vs_baseline_and_raw": bool(all(pool[f"{comparison}__{endpoint}__bootstrap_upper"] <= 0 for comparison in ("baseline", "raw_full_link") for endpoint in ("log_loss", "brier"))),
            "Holm_both_raw_endpoints": bool(raw_multi["holm_pass"].all()),
            "all_cell_losses_vs_raw": bool((cell["log_loss_difference_vs_raw"] <= 0).all() and (cell["brier_difference_vs_raw"] <= 0).all()),
            "all_cell_calibration": bool((cell["ece"] <= cell["raw_ece"]).all() and (cell["maximum_supported_bin_error"] <= 0.02).all()),
            "both_months_vs_raw": bool((time["log_loss_difference_vs_raw"] <= 0).all() and (time["brier_difference_vs_raw"] <= 0).all()),
            "every_stock_vs_raw": bool((stock["log_loss_difference_vs_raw"] <= 0).all() and (stock["brier_difference_vs_raw"] <= 0).all()),
            "zero_orientation_reversals_vs_baseline": bool((orientation["log_loss_difference_vs_baseline"] <= 0).all() and (orientation["brier_difference_vs_baseline"] <= 0).all()),
            "ranking_vs_raw": bool((rank["gain_vs_raw"] >= 0).all()),
            "at_least_two_experts_used": bool(assignment["distinct_experts_used"] >= 2),
            "assignment_stability": bool(assignment["assignment_stability"] >= 0.5),
        }
        expected = saved_gates[selector]
        counters = {
            "cell_loss_reversals_vs_raw": int(((cell["log_loss_difference_vs_raw"] > 0) | (cell["brier_difference_vs_raw"] > 0)).sum()),
            "calibration_failures": int(((cell["ece"] > cell["raw_ece"]) | (cell["maximum_supported_bin_error"] > 0.02)).sum()),
            "supported_orientations": len(orientation),
            "orientation_log_loss_reversals_vs_baseline": int((orientation["log_loss_difference_vs_baseline"] > 0).sum()),
            "orientation_brier_reversals_vs_baseline": int((orientation["brier_difference_vs_baseline"] > 0).sum()),
            "ranking_degradations_vs_raw": int((rank["gain_vs_raw"] < 0).sum()),
            "distinct_experts_used": int(assignment["distinct_experts_used"]),
        }
        if gate_checks != expected["checks"] or bool(all(gate_checks.values())) != expected["pass"] or any(counters[key] != expected[key] for key in counters):
            gate_errors.append(selector)
    record("selector_gate_replay", not gate_errors, gate_errors)

    decision = json.loads((ROOT / "decision.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    record(
        "decision_replay",
        all(not saved_gates[selector]["pass"] for selector in CANDIDATES)
        and decision["passing_selectors"] == []
        and decision["selected_selector"] is None
        and decision["named_loop_good_or_high_promoted"] is False,
        decision,
    )
    record(
        "summary_reconciliation",
        summary["decision"] == decision
        and summary["validation_rows"] == len(stored)
        and summary["direct_volume_fields_used"] == []
        and summary["direction_or_signed_return_used"] is False
        and summary["later_period_scoring_performed"] is False,
        {"rows": summary["validation_rows"], "selectors": summary["selector_count"]},
    )
    all_passed = all(check["passed"] for check in checks.values())
    result = {
        "audit_id": "regime_loop_individual_expert_selection_v1_independent_audit",
        "all_passed": all_passed,
        "checks_passed": sum(check["passed"] for check in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "auditor_imported_runner": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ROOT / "independent_audit.json").write_text(json.dumps(safe(result), indent=2, sort_keys=True) + "\n")
    if not all_passed:
        raise AssertionError(
            "individual expert audit failed: "
            + str([name for name, check in checks.items() if not check["passed"]])
        )
    print(json.dumps(safe(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_audit()
