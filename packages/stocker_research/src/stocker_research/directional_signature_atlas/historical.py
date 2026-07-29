"""Historical adapters for the frozen fixed-clock atlas population."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.directional_signature_atlas.features import (
    add_cross_sectional_features,
    assert_causal_feature_ledger,
)
from stocker_research.directional_signature_atlas.outcomes import (
    build_economic_outcome,
    movement_permission,
)


def load_frozen_movement_bundle(bundle_root: Path, core: ModuleType) -> dict[str, Any]:
    """Load the exact registered state, path, cycle, and movement bundle."""

    artifacts = bundle_root / "artifacts"
    return {
        "preprocessing": pd.read_csv(artifacts / "state/frozen_emission_preprocessing.csv"),
        "state_parameters": dict(np.load(artifacts / "state/frozen_semimarkov_parameters.npz")),
        "cycles": core.load_cycles(artifacts / "state/fixed_cycle_shuffled_nulls.csv"),
        "path_parameters": dict(np.load(artifacts / "path/model_parameters.npz")),
        "feature_manifest": json.loads(
            (artifacts / "price/feature_manifest.json").read_text(encoding="utf-8")
        ),
        "outcome_parameters": dict(np.load(artifacts / "price/outcome_model_parameters.npz")),
    }


def _state_anchor_rows(panel: pd.DataFrame, decision_ordinals: set[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = panel.groupby(["symbol_norm", "session_date"], sort=False)
    for (symbol, session), group in grouped:
        run_states: list[int] = []
        run_lengths: list[int] = []
        for tuple_row in group.itertuples(index=False):
            row: Any = tuple_row
            state = int(row.state)
            if not run_states or state != run_states[-1]:
                run_states.append(state)
                run_lengths.append(1)
            else:
                run_lengths[-1] += 1
            ordinal = int(row.bar_index_in_session)
            if ordinal not in decision_ordinals:
                continue
            motifs: dict[str, str | None] = {}
            completed_states = run_states[:-1]
            for length in (2, 3, 4):
                motifs[f"state_motif_{length}"] = (
                    ">".join(str(value) for value in completed_states[-length:])
                    if len(completed_states) >= length
                    else None
                )
            repeat = 0
            cursor = len(run_states) - 1
            while cursor >= 2 and run_states[cursor] == run_states[cursor - 2]:
                repeat += 1
                cursor -= 2
            raw = row._asdict()
            rows.append(
                {
                    **raw,
                    "symbol": str(symbol),
                    "session": str(session),
                    "decision_ordinal": ordinal,
                    "current_state": state,
                    "previous_state_1": run_states[-2] if len(run_states) >= 2 else 8,
                    "previous_state_2": run_states[-3] if len(run_states) >= 3 else 8,
                    "previous_state": run_states[-2] if len(run_states) >= 2 else np.nan,
                    "prior_completed_state_dwell_bars": run_lengths[-2]
                    if len(run_lengths) >= 2
                    else np.nan,
                    # No exact transition-duration source exists at this fixed-clock
                    # boundary.  Dwell is not a valid substitute, so fail closed.
                    "prior_completed_transition_duration_bars": np.nan,
                    "same_orientation_repeat_count": repeat,
                    "state_run_entry_at_decision": int(row.age) == 1,
                    **motifs,
                }
            )
    anchors = pd.DataFrame(rows)
    if anchors.empty:
        raise AssertionError("no fixed-clock state anchors reconstructed")
    return anchors


def _orientation(core_states: tuple[int, ...], previous: int, current: int) -> str:
    if previous == 8 or len(core_states) == 2:
        return "bidirectional" if len(core_states) == 2 else "unavailable"
    forward = any(
        core_states[index] == previous and core_states[(index + 1) % len(core_states)] == current
        for index in range(len(core_states))
    )
    reverse = any(
        core_states[index] == previous and core_states[(index - 1) % len(core_states)] == current
        for index in range(len(core_states))
    )
    if forward and not reverse:
        return "forward"
    if reverse and not forward:
        return "reverse"
    return "ambiguous" if forward or reverse else "incompatible_transition"


def _add_loop_and_movement(
    anchors: pd.DataFrame,
    core: ModuleType,
    frozen: dict[str, Any],
) -> pd.DataFrame:
    output = anchors.copy()
    loop_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
    for column in loop_columns:
        output[column] = np.nan
    prediction_columns = (
        "predicted_future_range_bps",
        "predicted_absolute_movement_bps",
        "state_context_future_range_bps",
        "state_context_absolute_movement_bps",
    )
    for column in prediction_columns:
        output[column] = np.nan
    eligible = output["state_run_entry_at_decision"].astype(bool) & output[
        "frozen_state_inputs_complete"
    ].astype(bool)
    if not eligible.any():
        return output
    selected = output.loc[eligible].copy()
    selected["state"] = selected["current_state"].astype(int)
    selected["b0_entry_numeric"] = pd.to_numeric(selected["b0_state_numeric"], errors="coerce")
    selected["b0_entry_high_stress"] = pd.to_numeric(selected["b0_high_stress"], errors="coerce")
    local = pd.to_datetime(selected["timestamp"], utc=True).dt.tz_convert("America/New_York")
    entry_minutes = local.dt.hour * 60.0 + local.dt.minute - 570.0
    phase = 2.0 * np.pi * entry_minutes / 390.0
    selected["entry_time_sin"] = np.sin(phase)
    selected["entry_time_cos"] = np.cos(phase)
    selected = core.add_loop_scores(selected, frozen["cycles"], frozen["path_parameters"])
    selected = core.movement_predictions(
        selected,
        frozen["feature_manifest"],
        frozen["outcome_parameters"],
    )
    for column in loop_columns:
        output.loc[eligible, column] = selected[column].to_numpy(float)
    mapping = {
        "predicted_future_range_bps": "loop_scores__future_range_bps_prediction_24",
        "predicted_absolute_movement_bps": "loop_scores__absolute_return_bps_prediction_24",
        "state_context_future_range_bps": "state_context__future_range_bps_prediction_24",
        "state_context_absolute_movement_bps": "state_context__absolute_return_bps_prediction_24",
    }
    for destination, source in mapping.items():
        output.loc[eligible, destination] = selected[source].to_numpy(float)
    return output


def _add_loop_summaries(
    anchors: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    output = anchors.copy()
    loop_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
    score_matrix = output[loop_columns].to_numpy(float)
    available = np.isfinite(score_matrix).all(axis=1)
    output["top_parent_loop"] = pd.NA
    output["top_loop_orientation"] = pd.NA
    output["parent_loop_family"] = pd.NA
    output["top_loop_score"] = np.nan
    output["top_second_margin"] = np.nan
    output["compatibility_mass"] = np.nan
    output["compatibility_entropy"] = np.nan
    output["compatible_loop_count"] = np.nan
    cycle_cores = [tuple(int(value) for value in core) for core in cycles["core"]]
    for position in np.flatnonzero(available):
        scores = score_matrix[position]
        order = np.argsort(-scores, kind="mergesort")
        top = int(order[0])
        mass = float(scores.sum())
        probabilities = scores / mass if mass > 0.0 else np.zeros_like(scores)
        positive = probabilities[probabilities > 0.0]
        entropy = (
            float(-(positive * np.log(positive)).sum() / np.log(len(scores)))
            if len(positive) and len(scores) > 1
            else 0.0
        )
        core_states = cycle_cores[top]
        output.at[position, "top_parent_loop"] = f"cycle_{top + 1:02d}"
        output.at[position, "top_loop_orientation"] = _orientation(
            core_states,
            int(cast(Any, output.at[position, "previous_state_1"])),
            int(cast(Any, output.at[position, "current_state"])),
        )
        output.at[position, "parent_loop_family"] = f"transition_length_{len(core_states)}"
        output.at[position, "top_loop_score"] = float(scores[top])
        output.at[position, "top_second_margin"] = float(scores[top] - scores[order[1]])
        output.at[position, "compatibility_mass"] = mass
        output.at[position, "compatibility_entropy"] = entropy
        output.at[position, "compatible_loop_count"] = int(np.count_nonzero(scores > 0.0))
    return output.drop(columns=loop_columns)


def build_causal_anchor_panel(
    core: ModuleType,
    frozen: dict[str, Any],
    *,
    provider_root: Path,
    symbols: list[str],
    as_of: pd.Timestamp,
    decision_ordinals: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct causal state features once, then retain fixed decision rows."""

    start = pd.Timestamp("2024-01-01", tz="UTC")
    parts = [core.prepare_symbol_bars(symbol, provider_root, start, as_of) for symbol in symbols]
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.loc[
        ~(
            panel["symbol_norm"].eq("AAL")
            & panel["session_date"].astype(str).str.startswith("2026-")
        )
    ].copy()
    panel = panel.sort_values(["symbol_norm", "timestamp"], kind="mergesort").reset_index(drop=True)
    vti = core.prepare_symbol_bars("VTI", provider_root, start, as_of)
    vti_max_timestamp = pd.to_datetime(vti["timestamp"], utc=True).max()
    panel = core.add_market_features(panel, vti)
    panel["frozen_state_inputs_complete"] = pd.to_datetime(panel["timestamp"], utc=True).le(
        vti_max_timestamp
    )
    b0 = core.build_causal_b0(symbols, provider_root, start, as_of)
    panel = panel.merge(
        b0[
            [
                "session_date",
                "causal_slow_b0",
                "b0_direction_score",
                "b0_stress_score",
                "b0_stress_box",
            ]
        ],
        on="session_date",
        how="left",
        validate="many_to_one",
    )
    panel["b0_state_numeric"] = panel["causal_slow_b0"].map(
        {"weak_broad_tape": -1.0, "neutral_broad_tape": 0.0, "strong_broad_tape": 1.0}
    )
    panel["b0_high_stress"] = panel["b0_stress_box"].eq("high_stress").astype(float)
    panel = core.add_emission_features(panel)
    panel = panel.sort_values(["symbol_norm", "session_date", "timestamp"], kind="mergesort")
    panel = panel.reset_index(drop=True)
    panel = core.assign_session_states(
        panel,
        frozen["preprocessing"],
        frozen["state_parameters"],
    )
    grouped = panel.groupby(["symbol_norm", "session_date"], sort=False)
    rolling_high = grouped["high"].transform(lambda values: values.rolling(12, min_periods=1).max())
    rolling_low = grouped["low"].transform(lambda values: values.rolling(12, min_periods=1).min())
    panel["rolling_high_low_location"] = (panel["close"] - rolling_low) / (
        rolling_high - rolling_low
    ).replace(0.0, np.nan)
    anchors = _state_anchor_rows(panel, set(decision_ordinals))
    duration_hazard = frozen["state_parameters"]["duration_hazard"]
    anchors["current_departure_probability"] = [
        float(duration_hazard[int(state), min(int(age), 24) - 1]) if complete else np.nan
        for state, age, complete in zip(
            anchors["current_state"],
            anchors["age"],
            anchors["frozen_state_inputs_complete"],
            strict=True,
        )
    ]
    anchors = _add_loop_and_movement(anchors, core, frozen)
    anchors = _add_loop_summaries(anchors, frozen["cycles"])
    audit = {
        "panel_rows": len(panel),
        "anchor_rows": len(anchors),
        "state_run_entry_anchor_rows": int(anchors["state_run_entry_at_decision"].sum()),
        "movement_prediction_rows": int(anchors["predicted_future_range_bps"].notna().sum()),
        "state_input_complete_anchor_rows": int(anchors["frozen_state_inputs_complete"].sum()),
        "state_input_unavailable_anchor_rows": int(
            (~anchors["frozen_state_inputs_complete"].astype(bool)).sum()
        ),
        "vti_max_timestamp": str(vti_max_timestamp),
        "panel_max_timestamp": str(pd.to_datetime(panel["timestamp"], utc=True).max()),
    }
    return anchors, audit


def _four_way_scale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        [-np.inf, -1.0, 0.0, 1.0, np.inf],
        labels=[
            "below_minus_one_scale",
            "minus_one_to_zero",
            "zero_to_plus_one",
            "above_plus_one_scale",
        ],
        right=False,
    ).astype("string")


def _cut(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        edges,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype("string")


def build_feature_ledger(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    feature_schema: dict[str, Any],
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    """Join outcome-free fixed-clock price anchors to causal structural features."""

    ledger = events.rename(
        columns={"event_id": "opportunity_id", "symbol_norm": "symbol", "session_date": "session"}
    ).copy()
    ledger["opportunity_id"] = (
        ledger["opportunity_id"].astype(str).str.replace("lsn|", "atlas|", n=1, regex=False)
    )
    anchor_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "timestamp",
        "current_state",
        "previous_state",
        "previous_state_1",
        "previous_state_2",
        "state_motif_2",
        "state_motif_3",
        "state_motif_4",
        "age",
        "prior_completed_state_dwell_bars",
        "prior_completed_transition_duration_bars",
        "same_orientation_repeat_count",
        "current_departure_probability",
        "state_posterior_probability",
        "state_run_entry_at_decision",
        "frozen_state_inputs_complete",
        "top_parent_loop",
        "top_loop_orientation",
        "parent_loop_family",
        "top_loop_score",
        "top_second_margin",
        "compatibility_mass",
        "compatibility_entropy",
        "compatible_loop_count",
        "predicted_future_range_bps",
        "predicted_absolute_movement_bps",
        "state_context_future_range_bps",
        "state_context_absolute_movement_bps",
        "return_sum_6",
        "return_sum_12",
        "return_std_12",
        "stock_minus_vti_return_6",
        "market_breadth_return_6_positive",
        "market_dispersion_return_6",
        "log_relative_historical_volume",
        "rolling_high_low_location",
    ]
    ledger = ledger.merge(
        anchors[anchor_columns],
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_state"),
    )
    # The reused state/path/movement bundle was fitted on the full 2024 period.
    # It is therefore only a causally frozen transform from 2025 onward.  Keep
    # 2024 in the fixed population, but fail every model-derived field closed.
    before_parameter_freeze = pd.to_datetime(ledger["session"]).lt(pd.Timestamp("2025-01-01"))
    missing_state_inputs = (
        ~ledger["frozen_state_inputs_complete"].fillna(False).astype(bool) | before_parameter_freeze
    )
    state_columns = [
        "current_state",
        "previous_state",
        "state_motif_2",
        "state_motif_3",
        "state_motif_4",
        "age",
        "prior_completed_state_dwell_bars",
        "prior_completed_transition_duration_bars",
        "same_orientation_repeat_count",
        "state_posterior_probability",
        "current_departure_probability",
    ]
    ledger.loc[missing_state_inputs, state_columns] = np.nan
    ledger.loc[missing_state_inputs, "state_run_entry_at_decision"] = False
    model_derived_columns = [
        "top_parent_loop",
        "top_loop_orientation",
        "parent_loop_family",
        "top_loop_score",
        "top_second_margin",
        "compatibility_mass",
        "compatibility_entropy",
        "compatible_loop_count",
        "predicted_future_range_bps",
        "predicted_absolute_movement_bps",
        "state_context_future_range_bps",
        "state_context_absolute_movement_bps",
    ]
    ledger.loc[before_parameter_freeze, model_derived_columns] = np.nan
    if ledger.loc[~missing_state_inputs, "current_state"].isna().any():
        missing = (
            ledger.loc[~missing_state_inputs & ledger["current_state"].isna(), "opportunity_id"]
            .head()
            .tolist()
        )
        raise AssertionError(f"state reconstruction missed exact-input opportunities: {missing}")
    ledger["timestamp"] = pd.to_datetime(ledger["decision_timestamp"], utc=True)
    ledger["return_6_bps"] = 10_000.0 * pd.to_numeric(ledger["return_sum_6"], errors="coerce")
    ledger["return_12_bps"] = 10_000.0 * pd.to_numeric(ledger["return_sum_12"], errors="coerce")
    ledger = add_cross_sectional_features(
        ledger,
        ["return_6_bps", "return_12_bps"],
        timestamp_column="timestamp",
        min_peers=10,
    )
    ledger["stock_vs_universe_return_scale"] = (
        ledger["return_6_bps_versus_universe"] / ledger["prior_scale_bps"]
    )
    ledger["stock_vs_market_return_scale"] = (
        10_000.0 * ledger["stock_minus_vti_return_6"] / ledger["prior_scale_bps"]
    )
    ledger["directional_displacement_scale"] = ledger["return_sum_6"] / ledger[
        "return_std_12"
    ].replace(0.0, np.nan)
    ledger["range_to_cost_ratio"] = ledger["predicted_future_range_bps"] / round_trip_cost_bps
    ledger["movement_permission"] = pd.array(
        [
            movement_permission(float(value), round_trip_cost_bps) if pd.notna(value) else pd.NA
            for value in ledger["predicted_future_range_bps"]
        ],
        dtype="boolean",
    )
    ledger["scheduled_bars_remaining"] = 77 - ledger["decision_ordinal"].astype(int)
    ledger["minutes_since_open"] = ledger["decision_ordinal"].astype(int) * 5
    ledger["minutes_until_close"] = ledger["scheduled_bars_remaining"] * 5
    ledger["clock_phase"] = np.where(ledger["decision_ordinal"].eq(12), "opening", "middle")
    ledger["day_of_week"] = pd.to_datetime(ledger["session"]).dt.day_name()

    ledger["state_age_bin"] = _cut(
        ledger["age"], [-np.inf, 2, 4, 7, np.inf], ["1", "2_3", "4_6", "7_plus"]
    )
    ledger["prior_completed_dwell_bin"] = _cut(
        ledger["prior_completed_state_dwell_bars"],
        [-np.inf, 1, 3, 7, np.inf],
        ["missing_or_zero", "1_2", "3_6", "7_plus"],
    ).fillna("missing")
    ledger["same_orientation_repeat_bin"] = _cut(
        ledger["same_orientation_repeat_count"], [-np.inf, 1, 2, np.inf], ["0", "1", "2_plus"]
    )
    ledger["scheduled_bars_remaining_bin"] = _cut(
        ledger["scheduled_bars_remaining"], [41, 42, 66], ["41", "65"]
    )
    ledger["departure_probability_bin"] = _cut(
        ledger["current_departure_probability"],
        [-np.inf, 0.10, 0.30, np.inf],
        ["low", "middle", "high"],
    )
    ledger["top_loop_score_bin"] = _cut(
        ledger["top_loop_score"],
        [-np.inf, 1e-15, 0.05, 0.20, np.inf],
        ["zero", "low", "middle", "high"],
    )
    ledger["top_second_margin_bin"] = _cut(
        ledger["top_second_margin"],
        [-np.inf, 1e-15, 0.02, 0.10, np.inf],
        ["zero", "low", "middle", "high"],
    )
    ledger["compatibility_mass_bin"] = _cut(
        ledger["compatibility_mass"], [-np.inf, 0.25, 0.75, np.inf], ["low", "middle", "high"]
    )
    ledger["compatibility_entropy_bin"] = _cut(
        ledger["compatibility_entropy"],
        [-np.inf, 0.33, 0.66, np.inf],
        ["concentrated", "mixed", "diffuse"],
    )
    ledger["compatible_loop_count_bin"] = _cut(
        ledger["compatible_loop_count"], [-np.inf, 3, 7, np.inf], ["0_2", "3_6", "7_plus"]
    )
    for lag in (1, 3, 6, 12):
        ledger[f"return_{lag}_bin"] = _four_way_scale(ledger[f"return_{lag}_scale"])
    ledger["session_return_bin"] = _four_way_scale(ledger["session_return_scale"])
    ledger["opening_range_position_bin"] = _cut(
        ledger["opening_range_position"],
        [-np.inf, 0.0, 0.5, 1.0, np.inf],
        ["below_range", "lower_half", "upper_half", "above_range"],
    )
    ledger["typical_price_distance_bin"] = _four_way_scale(ledger["session_mean_distance_scale"])
    ledger["directional_displacement_bin"] = _four_way_scale(
        ledger["directional_displacement_scale"]
    )
    ledger["close_location_bin"] = _cut(
        ledger["current_close_location"], [-np.inf, 1 / 3, 2 / 3, np.inf], ["low", "middle", "high"]
    )
    ledger["current_range_bin"] = _cut(
        ledger["current_range_scale"],
        [-np.inf, 0.75, 1.25, np.inf],
        ["low", "normal", "high"],
    )
    body_ratio = ledger["current_body_scale"] / ledger["current_range_scale"].replace(0.0, np.nan)
    ledger["body_to_range_bin"] = _cut(
        body_ratio, [-np.inf, -0.33, 0.33, np.inf], ["bearish", "small", "bullish"]
    )
    wick_asymmetry = ledger["current_lower_wick_fraction"] - ledger["current_upper_wick_fraction"]
    ledger["wick_asymmetry_bin"] = _cut(
        wick_asymmetry,
        [-np.inf, -0.20, 0.20, np.inf],
        ["upper_dominant", "balanced", "lower_dominant"],
    )
    ledger["upper_wick_ratio_bin"] = _cut(
        ledger["current_upper_wick_fraction"],
        [-np.inf, 0.20, 0.50, np.inf],
        ["low", "middle", "high"],
    )
    ledger["lower_wick_ratio_bin"] = _cut(
        ledger["current_lower_wick_fraction"],
        [-np.inf, 0.20, 0.50, np.inf],
        ["low", "middle", "high"],
    )
    ledger["compression_bin"] = _cut(
        ledger["compression_3_to_12"],
        [-np.inf, 0.75, 1.25, np.inf],
        ["compressed", "normal", "expanded"],
    )
    ledger["rolling_high_low_location_bin"] = _cut(
        ledger["rolling_high_low_location"],
        [-np.inf, 1 / 3, 2 / 3, np.inf],
        ["low", "middle", "high"],
    )
    ledger["distance_from_rolling_low_bin"] = _cut(
        ledger["rolling_high_low_location"],
        [-np.inf, 1 / 3, 2 / 3, np.inf],
        ["near", "middle", "far"],
    )
    ledger["distance_from_rolling_high_bin"] = _cut(
        1.0 - ledger["rolling_high_low_location"],
        [-np.inf, 1 / 3, 2 / 3, np.inf],
        ["near", "middle", "far"],
    )
    ledger["realised_volatility_bin"] = _cut(
        10_000.0 * ledger["return_std_12"], [-np.inf, 15.0, 35.0, np.inf], ["low", "middle", "high"]
    )
    ledger["historical_activity_bin"] = _cut(
        ledger["log_relative_historical_volume"],
        [-np.inf, 0.5, 1.0, np.inf],
        ["low", "middle", "high"],
    )
    ledger["predicted_future_range_bin"] = _cut(
        ledger["predicted_future_range_bps"],
        [-np.inf, 30.0, 60.0, np.inf],
        ["below_permission", "permitted", "high"],
    )
    ledger["predicted_absolute_movement_bin"] = _cut(
        ledger["predicted_absolute_movement_bps"],
        [-np.inf, 20.0, 40.0, np.inf],
        ["low", "middle", "high"],
    )
    ledger["range_to_cost_bin"] = _cut(
        ledger["range_to_cost_ratio"],
        [-np.inf, 3.0, 6.0, np.inf],
        ["below_permission", "permitted", "high"],
    )
    for source in ("return_6_bps", "return_12_bps"):
        ledger[f"{source.removesuffix('_bps')}_cross_sectional_rank_bin"] = _cut(
            ledger[f"{source}_cross_sectional_rank"],
            [-np.inf, 0.20, 0.80, np.inf],
            ["bottom_20", "middle_60", "top_20"],
        )
    ledger["stock_vs_universe_return_bin"] = _four_way_scale(
        ledger["stock_vs_universe_return_scale"]
    )
    ledger["stock_vs_market_return_bin"] = _four_way_scale(ledger["stock_vs_market_return_scale"])
    ledger["universe_breadth_bin"] = _cut(
        ledger["return_6_bps_breadth_positive"],
        [-np.inf, 0.40, 0.60, np.inf],
        ["weak", "mixed", "strong"],
    )
    ledger["cross_sectional_dispersion_bin"] = _cut(
        ledger["return_6_bps_dispersion"], [-np.inf, 20.0, 50.0, np.inf], ["low", "middle", "high"]
    )
    ledger["round_trip_cost_bps"] = round_trip_cost_bps
    enabled = [
        row["name"] for row in feature_schema["features"] if bool(row.get("condition_enabled"))
    ]
    unavailable_loop = {
        "top_parent_loop",
        "top_loop_orientation",
        "parent_loop_family",
        "top_loop_score_bin",
        "top_second_margin_bin",
        "compatibility_mass_bin",
        "compatibility_entropy_bin",
        "compatible_loop_count_bin",
        "predicted_future_range_bin",
        "predicted_absolute_movement_bin",
        "range_to_cost_bin",
    }
    decisions = pd.to_datetime(ledger["decision_timestamp"], utc=True)
    for feature in enabled:
        availability = decisions.copy()
        if feature in unavailable_loop:
            availability = availability.where(ledger[feature].notna())
        else:
            availability = availability.where(ledger[feature].notna())
        ledger[f"{feature}__available_at"] = availability
    assert_causal_feature_ledger(ledger, enabled)
    return ledger.sort_values(
        ["session", "symbol", "decision_ordinal"], kind="mergesort"
    ).reset_index(drop=True)


def build_outcome_ledgers(
    events: pd.DataFrame,
    prior_contract: dict[str, Any],
    prior_runner: ModuleType,
    *,
    round_trip_cost_bps: float,
    horizon_bars: int,
    entry_delay_bars: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build primary and first-touch ledgers while preserving unavailable rows."""

    rows: list[dict[str, Any]] = []
    coverage_parts: list[pd.DataFrame] = []
    for symbol in sorted(events["symbol_norm"].astype(str).unique()):
        tape, coverage = prior_runner.load_symbol_rows(prior_contract, symbol)
        coverage_parts.append(coverage)
        selected = events.loc[events["symbol_norm"].eq(symbol)]
        sessions = {
            (int(period), str(session)): group
            for (period, session), group in tape.groupby(["period", "session_date"], sort=False)
        }
        for tuple_event in selected.itertuples(index=False):
            event: Any = tuple_event
            session = sessions.get((int(event.period), str(event.session_date)))
            if session is None:
                outcome = {
                    "decision_ordinal": int(event.decision_ordinal),
                    "score_status": "missing_session",
                    "target": "UNAVAILABLE",
                    "first_touch_target": "UNAVAILABLE",
                    "first_touch_step": None,
                    "first_touch_barrier_bps": float(event.barrier_bps),
                }
            else:
                outcome = build_economic_outcome(
                    session,
                    int(event.decision_ordinal),
                    round_trip_cost_bps,
                    horizon_bars=horizon_bars,
                    entry_delay_bars=entry_delay_bars,
                    first_touch_barrier_bps=float(event.barrier_bps),
                )
            rows.append(
                {
                    "run_id": "directional-signature-atlas-v1",
                    "opportunity_id": str(event.event_id).replace("lsn|", "atlas|", 1),
                    "source_event_id": str(event.event_id),
                    "period": int(event.period),
                    "symbol": symbol,
                    "session": str(event.session_date),
                    "decision_clock": str(event.decision_clock),
                    **outcome,
                }
            )
    primary = (
        pd.DataFrame(rows)
        .sort_values(["session", "symbol", "decision_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )
    first_touch = primary[
        [
            "run_id",
            "opportunity_id",
            "period",
            "symbol",
            "session",
            "decision_clock",
            "decision_ordinal",
            "score_status",
            "first_touch_target",
            "first_touch_step",
            "first_touch_barrier_bps",
        ]
    ].copy()
    coverage = pd.concat(coverage_parts, ignore_index=True)
    return primary, first_touch, coverage


def feature_family_map(feature_schema: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["name"]): str(row["family"])
        for row in feature_schema["features"]
        if bool(row.get("condition_enabled"))
    }


def recompute_cross_sectional_after_stock_deletion(
    frame: pd.DataFrame,
    deleted_symbol: str,
) -> pd.DataFrame:
    """Delete one stock and recompute direct contemporaneous peer features.

    Frozen state/loop/movement context is intentionally not re-estimated here;
    callers must label that downstream portion unavailable or frozen.
    """

    output = frame.loc[frame["symbol"].ne(deleted_symbol)].copy()
    output = add_cross_sectional_features(
        output,
        ["return_6_bps", "return_12_bps"],
        timestamp_column="timestamp",
        min_peers=10,
    )
    output["stock_vs_universe_return_scale"] = (
        output["return_6_bps_versus_universe"] / output["prior_scale_bps"]
    )
    for source in ("return_6_bps", "return_12_bps"):
        output[f"{source.removesuffix('_bps')}_cross_sectional_rank_bin"] = _cut(
            output[f"{source}_cross_sectional_rank"],
            [-np.inf, 0.20, 0.80, np.inf],
            ["bottom_20", "middle_60", "top_20"],
        )
    output["stock_vs_universe_return_bin"] = _four_way_scale(
        output["stock_vs_universe_return_scale"]
    )
    output["universe_breadth_bin"] = _cut(
        output["return_6_bps_breadth_positive"],
        [-np.inf, 0.40, 0.60, np.inf],
        ["weak", "mixed", "strong"],
    )
    output["cross_sectional_dispersion_bin"] = _cut(
        output["return_6_bps_dispersion"],
        [-np.inf, 20.0, 50.0, np.inf],
        ["low", "middle", "high"],
    )
    return output
