"""Predeclared stress and null analyses for frozen exploratory signatures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.directional_signature_atlas.analysis import (
    one_sided_session_p_value,
    signature_from_dict,
    signature_metrics,
)
from stocker_research.directional_signature_atlas.historical import (
    recompute_cross_sectional_after_stock_deletion,
)
from stocker_research.directional_signature_atlas.outcomes import classify_terminal_move
from stocker_research.directional_signature_atlas.signatures import (
    Condition,
    Signature,
    apply_multiple_testing,
    apply_signature,
)


def _metric_row(
    frame: pd.DataFrame,
    signature: Signature,
    *,
    stress: str,
    period: int,
    removed: str = "",
) -> dict[str, Any]:
    metrics = signature_metrics(frame, signature)
    keep = {
        key: metrics[key]
        for key in (
            "rows",
            "sessions",
            "stocks",
            "mean_directional_net_bps",
            "median_directional_net_bps",
            "directional_lift",
            "positive_payoff_rate",
            "profit_factor",
            "maximum_drawdown_bps",
            "positive_month_fraction",
            "twice_cost_mean_net_bps",
            "top_stock_absolute_contribution_share",
            "top_month_absolute_contribution_share",
        )
    }
    return {
        "signature_id": signature.signature_id,
        "direction": signature.direction,
        "period": period,
        "stress": stress,
        "removed": removed,
        **keep,
    }


def _replace_outcomes(frame: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "opportunity_id",
        "target",
        "net_long_return_bps",
        "net_short_return_bps",
        "gross_long_return_bps",
    ]
    base = frame.drop(
        columns=[
            "target",
            "long_net_bps",
            "short_net_bps",
            "net_long_return_bps",
            "net_short_return_bps",
            "gross_long_return_bps",
        ],
        errors="ignore",
    )
    output = base.merge(outcomes[columns], on="opportunity_id", how="left", validate="one_to_one")
    output["long_net_bps"] = output["net_long_return_bps"]
    output["short_net_bps"] = output["net_short_return_bps"]
    return output.loc[output["target"].ne("UNAVAILABLE")]


def stress_signature_library(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    delayed_outcomes: pd.DataFrame,
    *,
    ordered_bins: dict[str, list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply every frozen stress without changing a signature."""

    delayed = _replace_outcomes(frame, delayed_outcomes)
    stress_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        payoff_column = "long_net_bps" if signature.direction == "LONG" else "short_net_bps"
        for period in (2024, 2025, 2026):
            period_frame = frame.loc[frame["period"].eq(period)].copy()
            if period_frame.empty:
                continue
            stress_rows.append(
                _metric_row(period_frame, signature, stress="primary_cost", period=period)
            )
            twice = period_frame.copy()
            twice["long_net_bps"] -= twice["round_trip_cost_bps"]
            twice["short_net_bps"] -= twice["round_trip_cost_bps"]
            stress_rows.append(_metric_row(twice, signature, stress="twice_cost", period=period))
            delayed_period = delayed.loc[delayed["period"].eq(period)]
            stress_rows.append(
                _metric_row(
                    delayed_period,
                    signature,
                    stress="one_bar_execution_delay_same_terminal",
                    period=period,
                )
            )
            for multiple in (1.0, 3.0):
                dead_band = period_frame.copy()
                dead_band["target"] = [
                    classify_terminal_move(float(value), 10.0, multiple)
                    for value in dead_band["gross_long_return_bps"]
                ]
                stress_rows.append(
                    _metric_row(
                        dead_band,
                        signature,
                        stress=f"dead_band_{int(multiple)}x_cost",
                        period=period,
                    )
                )
            permitted = period_frame.loc[period_frame["movement_permission"].astype(bool)]
            stress_rows.append(
                _metric_row(
                    permitted,
                    signature,
                    stress="movement_permission_on",
                    period=period,
                )
            )
            not_permitted = period_frame.loc[~period_frame["movement_permission"].astype(bool)]
            stress_rows.append(
                _metric_row(
                    not_permitted,
                    signature,
                    stress="movement_permission_off",
                    period=period,
                )
            )
            for condition_index, condition in enumerate(signature.conditions):
                if condition.feature.startswith("state_motif_"):
                    stress_rows.append(
                        _metric_row(
                            period_frame,
                            signature,
                            stress=f"motif_length_{condition.feature.rsplit('_', maxsplit=1)[-1]}",
                            period=period,
                        )
                    )
                levels = ordered_bins.get(condition.feature, [])
                if condition.value not in levels:
                    continue
                level_index = levels.index(condition.value)
                for neighbour_index in (level_index - 1, level_index + 1):
                    if not 0 <= neighbour_index < len(levels):
                        continue
                    neighbour_conditions = list(signature.conditions)
                    neighbour_conditions[condition_index] = Condition(
                        feature=condition.feature,
                        operator=condition.operator,
                        value=levels[neighbour_index],
                        family=condition.family,
                    )
                    neighbour = Signature(
                        signature_id=signature.signature_id,
                        direction=signature.direction,
                        conditions=tuple(neighbour_conditions),
                        source="threshold_neighbour_stress",
                    )
                    stress_rows.append(
                        _metric_row(
                            period_frame,
                            neighbour,
                            stress="adjacent_threshold_neighbour",
                            period=period,
                            removed=(
                                f"{condition.feature}:{condition.value}->{levels[neighbour_index]}"
                            ),
                        )
                    )
            weakest_removed = period_frame.loc[period_frame["historical_activity_bin"].ne("low")]
            stress_rows.append(
                _metric_row(
                    weakest_removed,
                    signature,
                    stress="exclude_weakest_historical_activity_cohort",
                    period=period,
                )
            )
            selected = period_frame.loc[apply_signature(period_frame, signature)].copy()
            selected["month"] = selected["session"].astype(str).str[:7]
            selected["hindsight_episode"] = (
                selected["session"].astype(str) + "|" + selected["decision_clock"].astype(str)
            )
            for dimension, label, count in (
                ("symbol", "remove_best_stock", 1),
                ("symbol", "remove_top_five_stocks", 5),
                ("month", "remove_best_month", 1),
                ("hindsight_episode", "remove_best_episode", 1),
                ("hindsight_episode", "remove_top_five_episodes", 5),
            ):
                if selected.empty:
                    removed_values: list[str] = []
                else:
                    removed_values = (
                        selected.groupby(dimension, sort=False)[payoff_column]
                        .sum()
                        .sort_values(ascending=False)
                        .head(count)
                        .index.astype(str)
                        .tolist()
                    )
                source_dimension = (
                    period_frame["session"].astype(str).str[:7]
                    if dimension == "month"
                    else (
                        period_frame["session"].astype(str)
                        + "|"
                        + period_frame["decision_clock"].astype(str)
                        if dimension == "hindsight_episode"
                        else period_frame[dimension].astype(str)
                    )
                )
                reduced = period_frame.loc[~source_dimension.isin(removed_values)]
                stress_rows.append(
                    _metric_row(
                        reduced,
                        signature,
                        stress=label,
                        period=period,
                        removed="|".join(removed_values),
                    )
                )
            for clock, clock_frame in period_frame.groupby("decision_clock", sort=True):
                stress_rows.append(
                    _metric_row(
                        clock_frame,
                        signature,
                        stress="coarse_clock_bin",
                        period=period,
                        removed=f"kept:{clock}",
                    )
                )
            for symbol in sorted(period_frame["symbol"].astype(str).unique()):
                recomputed = recompute_cross_sectional_after_stock_deletion(period_frame, symbol)
                row = _metric_row(
                    recomputed,
                    signature,
                    stress="leave_one_stock_out_recomputed",
                    period=period,
                    removed=symbol,
                )
                leave_one_out_rows.append(row)
    return pd.DataFrame(stress_rows), pd.DataFrame(leave_one_out_rows)


def _persistent_count(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    library: list[dict[str, Any]],
) -> tuple[int, float]:
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    p_values: list[float] = []
    validation_effects: list[float] = []
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        discovery_metrics = signature_metrics(discovery, signature)
        validation_metrics = signature_metrics(validation, signature)
        validation_effects.append(float(validation_metrics["mean_directional_net_bps"]))
        evaluated.append((discovery_metrics, validation_metrics))
        p_values.append(one_sided_session_p_value(validation, signature))
    adjusted = apply_multiple_testing(
        p_values,
        method="holm",
    )
    count = 0
    for (discovery_metrics, validation_metrics), adjusted_p_value in zip(
        evaluated, adjusted, strict=True
    ):
        if all(
            (
                float(discovery_metrics["mean_directional_net_bps"]) > 0.0,
                float(validation_metrics["mean_directional_net_bps"]) > 0.0,
                float(discovery_metrics["directional_lift"]) > 0.0,
                float(validation_metrics["directional_lift"]) > 0.0,
                float(validation_metrics["twice_cost_mean_net_bps"]) > 0.0,
                int(validation_metrics["rows"]) >= 80,
                int(validation_metrics["sessions"]) >= 30,
                int(validation_metrics["stocks"]) >= 8,
                int(validation_metrics["months"]) >= 3,
                float(validation_metrics["top_stock_absolute_contribution_share"]) <= 0.25,
                float(validation_metrics["positive_month_fraction"]) > 0.5,
                adjusted_p_value <= 0.10,
            )
        ):
            count += 1
    return count, max(validation_effects, default=math.nan)


def null_test_results(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    random_signatures: list[Signature],
    atlas_decisions: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Run the seven frozen null families without replacement-rule discovery."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    def record(name: str, transformed: pd.DataFrame, tested: list[dict[str, Any]]) -> None:
        discovery = transformed.loc[transformed["period"].eq(2024)]
        validation = transformed.loc[transformed["period"].eq(2025)]
        persistent, maximum = _persistent_count(discovery, validation, tested)
        rows.append(
            {
                "null": name,
                "tested_signatures": len(tested),
                "persistent_positive_count": persistent,
                "maximum_validation_mean_net_bps": maximum,
                "similar_persistent_validation_performance": persistent > 0,
            }
        )

    permuted = frame.copy()
    outcome_columns = ["target", "long_net_bps", "short_net_bps"]
    for _, positions in permuted.groupby("period", sort=False).groups.items():
        location = np.asarray(list(positions))
        permuted.loc[location, outcome_columns] = permuted.loc[
            location, outcome_columns
        ].to_numpy()[rng.permutation(len(location))]
    record("outcome_labels_permuted_within_period", permuted, library)

    shifted = frame.copy()
    condition_features = sorted(
        {
            condition["feature"]
            for entry in library
            for condition in entry["signature"]["conditions"]
        }
    )
    shifted = shifted.sort_values(["symbol", "decision_clock", "session"], kind="mergesort")
    shifted[condition_features] = shifted.groupby(["symbol", "decision_clock"], sort=False)[
        condition_features
    ].shift(1)
    record("feature_rows_wrong_session_lag", shifted, library)

    flipped = frame.copy()
    flipped["target"] = flipped["target"].replace({"LONG": "SHORT", "SHORT": "LONG"})
    flipped[["long_net_bps", "short_net_bps"]] = flipped[
        ["short_net_bps", "long_net_bps"]
    ].to_numpy()
    record("direction_labels_flipped", flipped, library)

    stock_permuted = frame.copy()
    cross_columns = [
        str(column)
        for column in stock_permuted
        if (
            "cross_sectional" in str(column)
            or "versus_universe" in str(column)
            or str(column).startswith("stock_vs_")
            or str(column) == "universe_breadth_bin"
        )
        and not str(column).endswith("__available_at")
    ]
    for _, positions in stock_permuted.groupby("decision_timestamp", sort=False).groups.items():
        location = np.asarray(list(positions))
        stock_permuted.loc[location, cross_columns] = stock_permuted.loc[
            location, cross_columns
        ].to_numpy()[rng.permutation(len(location))]
    cross_sectional_library = [
        entry
        for entry in library
        if any(
            condition["family"] in {"cross_sectional", "cross_sectional_environment"}
            for condition in entry["signature"]["conditions"]
        )
    ]
    record(
        "stock_identity_permuted_within_timestamp",
        stock_permuted,
        cross_sectional_library,
    )

    motif_permuted = frame.copy()
    motif_columns = ["state_motif_2", "state_motif_3", "state_motif_4"]
    for _, positions in motif_permuted.groupby("clock_phase", sort=False).groups.items():
        location = np.asarray(list(positions))
        motif_permuted.loc[location, motif_columns] = motif_permuted.loc[
            location, motif_columns
        ].to_numpy()[rng.permutation(len(location))]
    motif_library = [
        entry
        for entry in library
        if any(
            condition["family"] == "state_history" for condition in entry["signature"]["conditions"]
        )
    ]
    record("state_history_permuted_within_clock_phase", motif_permuted, motif_library)

    sampled = random_signatures[: len(library)]
    random_library = [
        {
            "signature": signature.to_dict(),
            "discovery_metrics": {},
        }
        for signature in sampled
    ]
    record("random_signatures_support_complexity_matched", frame, random_library)

    validation_decisions = atlas_decisions.loc[atlas_decisions["period"].eq(2025)]
    coverage = int(validation_decisions["predicted_state"].isin(["LONG", "SHORT"]).sum())
    validation = frame.loc[frame["period"].eq(2025)]
    random_payoff = 0.0
    if coverage > 0 and len(validation):
        sampled_positions = rng.choice(
            len(validation), size=min(coverage, len(validation)), replace=False
        )
        directions = rng.choice(["LONG", "SHORT"], size=len(sampled_positions))
        long = validation.iloc[sampled_positions]["long_net_bps"].to_numpy(float)
        short = validation.iloc[sampled_positions]["short_net_bps"].to_numpy(float)
        random_payoff = float(np.where(directions == "LONG", long, short).mean())
    rows.append(
        {
            "null": "random_atlas_controller_coverage_matched",
            "tested_signatures": 0,
            "persistent_positive_count": 0,
            "maximum_validation_mean_net_bps": random_payoff,
            "similar_persistent_validation_performance": False,
        }
    )
    return pd.DataFrame(rows)
