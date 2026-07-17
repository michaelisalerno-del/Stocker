"""Predeclared stress and null analyses for frozen exploratory signatures."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.directional_signature_atlas.analysis import (
    neutral_veto_metrics,
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
    status: str = "available",
    chronology_stage: str = "",
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
        "chronology_stage": chronology_stage
        or (
            str(frame["chronology_stage"].iloc[0])
            if len(frame) and "chronology_stage" in frame
            else ""
        ),
        "stress": stress,
        "removed": removed,
        "status": status,
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


def _motif_length_variants(
    frame: pd.DataFrame,
    signature: Signature,
    condition_index: int,
) -> list[tuple[int, str, Signature]]:
    condition = signature.conditions[condition_index]
    if not condition.feature.startswith("state_motif_"):
        return []
    current_parts = str(condition.value).split(">")
    variants: list[tuple[int, str, Signature]] = []
    for target_length in (2, 3, 4):
        target_feature = f"state_motif_{target_length}"
        if target_feature not in frame or target_length == len(current_parts):
            continue
        if target_length < len(current_parts):
            tokens = [">".join(current_parts[-target_length:])]
        else:
            tokens = sorted(
                {
                    str(value)
                    for value in frame[target_feature].dropna().unique()
                    if str(value).split(">")[-len(current_parts) :] == current_parts
                }
            )
        for token in tokens:
            conditions = list(signature.conditions)
            conditions[condition_index] = Condition(
                feature=target_feature,
                operator="==",
                value=token,
                family=condition.family,
            )
            variants.append(
                (
                    target_length,
                    token,
                    Signature(
                        signature_id=signature.signature_id,
                        direction=signature.direction,
                        conditions=tuple(conditions),
                        source=signature.source,
                    ),
                )
            )
    return variants


def stress_signature_library(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    delayed_outcomes: pd.DataFrame,
    *,
    ordered_bins: dict[str, list[Any]],
    causal_population: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply every frozen stress without changing a signature."""

    delayed = _replace_outcomes(frame, delayed_outcomes)
    causal_source = frame if causal_population is None else causal_population
    stress_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    stage_values = (
        sorted(frame["chronology_stage"].dropna().astype(str).unique())
        if "chronology_stage" in frame
        else [str(value) for value in sorted(frame["period"].unique())]
    )
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        peer_dependent_structural_context = any(
            condition.family
            in {
                "structural_state",
                "state_history",
                "loop",
                "movement_permission",
            }
            for condition in signature.conditions
        )
        payoff_column = "long_net_bps" if signature.direction == "LONG" else "short_net_bps"
        for stage in stage_values:
            period_frame = frame.loc[
                frame.get("chronology_stage", frame["period"].astype(str)).astype(str).eq(stage)
            ].copy()
            if period_frame.empty:
                continue
            period = int(period_frame["period"].iloc[0])
            stress_rows.append(
                _metric_row(period_frame, signature, stress="primary_cost", period=period)
            )
            twice = period_frame.copy()
            twice["long_net_bps"] -= twice["round_trip_cost_bps"]
            twice["short_net_bps"] -= twice["round_trip_cost_bps"]
            stress_rows.append(_metric_row(twice, signature, stress="twice_cost", period=period))
            delayed_stage = delayed.get("chronology_stage", delayed["period"].astype(str))
            delayed_period = delayed.loc[delayed_stage.astype(str).eq(stage)]
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
                    classify_terminal_move(float(value), float(cost), multiple)
                    for value, cost in zip(
                        dead_band["gross_long_return_bps"],
                        dead_band["round_trip_cost_bps"],
                        strict=True,
                    )
                ]
                stress_rows.append(
                    _metric_row(
                        dead_band,
                        signature,
                        stress=f"dead_band_{int(multiple)}x_cost",
                        period=period,
                    )
                )
            permitted = period_frame.loc[period_frame["movement_permission"].eq(True).fillna(False)]
            stress_rows.append(
                _metric_row(
                    permitted,
                    signature,
                    stress="movement_permission_on",
                    period=period,
                )
            )
            not_permitted = period_frame.loc[
                period_frame["movement_permission"].eq(False).fillna(False)
            ]
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
                    for target_length, token, variant in _motif_length_variants(
                        period_frame, signature, condition_index
                    ):
                        stress_rows.append(
                            _metric_row(
                                period_frame,
                                variant,
                                stress=f"motif_length_{target_length}",
                                period=period,
                                removed=f"{variant.conditions[condition_index].feature}=={token}",
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
            for label in ("remove_best_episode", "remove_top_five_episodes"):
                stress_rows.append(
                    _metric_row(
                        period_frame.iloc[0:0],
                        signature,
                        stress=label,
                        period=period,
                        removed="unavailable_no_exact_episode_identity",
                        status="unavailable_no_exact_episode_identity",
                        chronology_stage=stage,
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
                causal_stage = causal_source.loc[
                    causal_source.get(
                        "chronology_stage", causal_source["period"].astype(str)
                    )
                    .astype(str)
                    .eq(stage)
                ]
                recomputed = recompute_cross_sectional_after_stock_deletion(
                    causal_stage, symbol
                )
                if "target" in recomputed:
                    recomputed = recomputed.loc[recomputed["target"].ne("UNAVAILABLE")]
                row = _metric_row(
                    recomputed,
                    signature,
                    stress="leave_one_stock_out_direct_cross_section_recomputed",
                    period=period,
                    removed=symbol,
                    status=(
                        "partial_downstream_structural_context_frozen"
                        if peer_dependent_structural_context
                        else "available_direct_cross_section_recomputed"
                    ),
                )
                row["direct_cross_sectional_features_recomputed"] = True
                row["downstream_state_loop_movement_recomputed"] = False
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
        stage = transformed.get("chronology_stage", transformed["period"].astype(str))
        discovery = transformed.loc[stage.astype(str).eq("discovery")]
        validation = transformed.loc[stage.astype(str).eq("validation")]
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
    permutation_group = "chronology_stage" if "chronology_stage" in permuted else "period"
    for _, positions in permuted.groupby(permutation_group, sort=False).groups.items():
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
    lag_invariant_features = {
        "decision_clock",
        "clock_phase",
        "scheduled_bars_remaining_bin",
        "minutes_since_open_bin",
        "minutes_until_close_bin",
    }
    lag_affected_library = [
        entry
        for entry in library
        if any(
            condition["feature"] not in lag_invariant_features
            for condition in entry["signature"]["conditions"]
        )
    ]
    record("feature_rows_wrong_session_lag", shifted, lag_affected_library)

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
            "cross_sectional_rank" in str(column)
            or "versus_universe" in str(column)
            or str(column).startswith("stock_vs_")
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
            condition["feature"] in cross_columns
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
            condition["feature"] in motif_columns
            for condition in entry["signature"]["conditions"]
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

    decision_stage = atlas_decisions.get(
        "chronology_stage", atlas_decisions["period"].astype(str)
    ).astype(str)
    frame_stage = frame.get("chronology_stage", frame["period"].astype(str)).astype(str)
    observed: dict[str, dict[str, Any]] = {}
    for stage_name in ("discovery", "validation"):
        decisions = atlas_decisions.loc[decision_stage.eq(stage_name)].copy()
        outcomes = frame.loc[frame_stage.eq(stage_name)].set_index("opportunity_id")
        directional = decisions["predicted_state"].isin(["LONG", "SHORT"])
        selected = decisions.loc[directional]
        payoffs = [
            float(
                cast(Any, outcomes.loc[str(row.opportunity_id), "long_net_bps"])
                if row.predicted_state == "LONG"
                else cast(Any, outcomes.loc[str(row.opportunity_id), "short_net_bps"])
            )
            for row in selected.itertuples(index=False)
            if str(row.opportunity_id) in outcomes.index
        ]
        observed[stage_name] = {
            "coverage": len(payoffs),
            "long_fraction": float(selected["predicted_state"].eq("LONG").mean())
            if len(selected)
            else 0.5,
            "mean": float(np.mean(payoffs)) if payoffs else math.nan,
        }
    draws = 1000
    validation_means: list[float] = []
    persistent_draws = 0
    for _ in range(draws):
        draw_means: dict[str, float] = {}
        for stage_name in ("discovery", "validation"):
            stage_frame = frame.loc[frame_stage.eq(stage_name)]
            count = min(int(observed[stage_name]["coverage"]), len(stage_frame))
            if count == 0:
                draw_means[stage_name] = math.nan
                continue
            sample_positions = rng.choice(len(stage_frame), size=count, replace=False)
            long_count = int(round(count * float(observed[stage_name]["long_fraction"])))
            directions = np.asarray(["LONG"] * long_count + ["SHORT"] * (count - long_count))
            rng.shuffle(directions)
            selected = stage_frame.iloc[sample_positions]
            payoff = np.where(
                directions == "LONG",
                selected["long_net_bps"].to_numpy(float),
                selected["short_net_bps"].to_numpy(float),
            )
            draw_means[stage_name] = float(np.mean(payoff))
        validation_means.append(draw_means["validation"])
        if all(math.isfinite(draw_means[name]) and draw_means[name] > 0.0 for name in draw_means):
            persistent_draws += 1
    observed_validation = float(observed["validation"]["mean"])
    finite_validation = np.asarray(
        [value for value in validation_means if math.isfinite(value)], dtype=float
    )
    similarity_rate = (
        float(np.mean(finite_validation >= observed_validation))
        if len(finite_validation) and math.isfinite(observed_validation)
        else 0.0
    )
    rows.append(
        {
            "null": "random_atlas_controller_coverage_matched",
            "tested_signatures": 0,
            "draws": draws,
            "persistent_positive_count": persistent_draws,
            "maximum_validation_mean_net_bps": float(np.max(finite_validation))
            if len(finite_validation)
            else math.nan,
            "observed_validation_mean_net_bps": observed_validation,
            "empirical_similarity_rate": similarity_rate,
            "similar_persistent_validation_performance": similarity_rate >= 0.05,
        }
    )
    return pd.DataFrame(rows)


def stress_neutral_veto_library(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    delayed_outcomes: pd.DataFrame,
    *,
    ordered_bins: dict[str, list[Any]],
) -> pd.DataFrame:
    """Apply the frozen stability battery to first-class neutral vetoes."""

    delayed = _replace_outcomes(frame, delayed_outcomes)
    rows: list[dict[str, Any]] = []
    stage_values = sorted(frame["chronology_stage"].dropna().astype(str).unique())

    def add(
        source: pd.DataFrame,
        signature: Signature,
        veto_id: str,
        stage: str,
        stress: str,
        *,
        removed: str = "",
        status: str = "available",
    ) -> None:
        metrics = neutral_veto_metrics(source, signature)
        rows.append(
            {
                "neutral_veto_id": veto_id,
                "chronology_stage": stage,
                "period": int(source["period"].iloc[0]) if len(source) else 0,
                "stress": stress,
                "removed": removed,
                "status": status,
                **metrics,
            }
        )

    for entry in library:
        signature = signature_from_dict(entry["signature"])
        veto_id = str(entry["neutral_veto_id"])
        for stage in stage_values:
            source = frame.loc[frame["chronology_stage"].astype(str).eq(stage)].copy()
            if source.empty:
                continue
            add(source, signature, veto_id, stage, "primary")
            twice = source.copy()
            twice["long_net_bps"] -= twice["round_trip_cost_bps"]
            twice["short_net_bps"] -= twice["round_trip_cost_bps"]
            add(twice, signature, veto_id, stage, "twice_cost")
            delayed_stage = delayed.loc[delayed["chronology_stage"].astype(str).eq(stage)]
            add(
                delayed_stage,
                signature,
                veto_id,
                stage,
                "one_bar_execution_delay_same_terminal",
            )
            for movement_value, label in ((True, "on"), (False, "off")):
                subset = source.loc[source["movement_permission"].eq(movement_value).fillna(False)]
                add(subset, signature, veto_id, stage, f"movement_permission_{label}")
            selected = source.loc[apply_signature(source, signature)].copy()
            selected["month"] = selected["session"].astype(str).str[:7]
            for dimension, label, count in (
                ("symbol", "remove_best_stock", 1),
                ("symbol", "remove_top_five_stocks", 5),
                ("month", "remove_best_month", 1),
            ):
                contribution = (
                    selected.assign(
                        neutral_excess=selected["target"].eq("NEUTRAL").astype(float)
                        - float(source["target"].eq("NEUTRAL").mean())
                    )
                    .groupby(dimension, sort=False)["neutral_excess"]
                    .sum()
                    .sort_values(ascending=False)
                )
                removed_values = contribution.head(count).index.astype(str).tolist()
                source_dimension = (
                    source["session"].astype(str).str[:7]
                    if dimension == "month"
                    else source[dimension].astype(str)
                )
                add(
                    source.loc[~source_dimension.isin(removed_values)],
                    signature,
                    veto_id,
                    stage,
                    label,
                    removed="|".join(removed_values),
                )
            for label in ("remove_best_episode", "remove_top_five_episodes"):
                add(
                    source.iloc[0:0],
                    signature,
                    veto_id,
                    stage,
                    label,
                    removed="unavailable_no_exact_episode_identity",
                    status="unavailable_no_exact_episode_identity",
                )
            for condition_index, condition in enumerate(signature.conditions):
                if condition.feature.startswith("state_motif_"):
                    for length, token, variant in _motif_length_variants(
                        source, signature, condition_index
                    ):
                        add(
                            source,
                            variant,
                            veto_id,
                            stage,
                            f"motif_length_{length}",
                            removed=token,
                        )
                levels = ordered_bins.get(condition.feature, [])
                if condition.value not in levels:
                    continue
                index = levels.index(condition.value)
                for neighbour in (index - 1, index + 1):
                    if not 0 <= neighbour < len(levels):
                        continue
                    conditions = list(signature.conditions)
                    conditions[condition_index] = Condition(
                        feature=condition.feature,
                        operator=condition.operator,
                        value=levels[neighbour],
                        family=condition.family,
                    )
                    variant = Signature(
                        signature_id=signature.signature_id,
                        direction=signature.direction,
                        conditions=tuple(conditions),
                        source="neutral_threshold_neighbour_stress",
                    )
                    add(
                        source,
                        variant,
                        veto_id,
                        stage,
                        "adjacent_threshold_neighbour",
                        removed=f"{condition.value}->{levels[neighbour]}",
                    )
    return pd.DataFrame(rows)
