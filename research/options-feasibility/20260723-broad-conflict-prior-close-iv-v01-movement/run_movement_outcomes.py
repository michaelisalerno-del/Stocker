#!/usr/bin/env python3
"""Run descriptive underlying movement outcomes for the fixed three-stock options probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
V0_DIR = EXPERIMENT_DIR.parent / "20260722-broad-conflict-prior-close-iv-v0"
PROBE_DIR = EXPERIMENT_DIR.parent / "20260722-broad-conflict-prior-close-iv-v01-probe"
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PAIR_RESULTS = PROBE_DIR / "artifacts" / "primary" / "contract_history_probe_results.csv"
PROBE_AUDIT = PROBE_DIR / "artifacts" / "primary" / "lightweight_audit.json"
PROTECTED_START = date(2025, 8, 23)
for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))
sys.path.insert(0, str(V0_DIR))

from run_screen_v0 import (  # noqa: E402
    DENSE_PANEL,
    TRACE_PANEL,
    load_clean_advance_panel,
    sha256_file,
    write_csv,
    write_json,
    write_parquet,
)

from stocker_research.broad_conflict_options_iv_screen_v0 import (  # noqa: E402
    SAFETY_FLAGS,
    add_iv_relative_outcomes,
    compute_underlying_movement_outcomes,
    previous_trading_session,
)


def _iso_date(value: object, *, name: str) -> str:
    """Normalize a date-like value to an ISO session identity."""

    try:
        parsed = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid date") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} is not a valid date")
    return parsed.date().isoformat()


def _normalise_pair_availability(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapping = {"true": True, "false": False}
    normalised = values.astype(str).str.casefold().map(mapping)
    if normalised.isna().any():
        raise ValueError("pair_available contains a non-boolean value")
    return normalised.astype(bool)


def build_movement_panels(
    structural: pd.DataFrame,
    bars: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build underlying-only and valid-pair IV-relative panels with an exact causal join."""

    pair_required = {
        "symbol",
        "signal_date",
        "required_options_date",
        "pair_available",
        "atm_iv",
    }
    if missing := sorted(pair_required.difference(pairs.columns)):
        raise ValueError(f"option-pair results missing columns: {missing}")
    pair_frame = pairs.copy()
    pair_frame["symbol"] = pair_frame["symbol"].astype(str)
    pair_frame["signal_date"] = pair_frame["signal_date"].map(
        lambda value: _iso_date(value, name="signal_date")
    )
    pair_frame["required_options_date"] = pair_frame["required_options_date"].map(
        lambda value: _iso_date(value, name="required_options_date")
    )
    pair_frame["pair_available"] = _normalise_pair_availability(pair_frame["pair_available"])
    if pair_frame.duplicated(["symbol", "signal_date"]).any():
        raise ValueError("option-pair results contain duplicate stock-sessions")
    for row in pair_frame.itertuples(index=False):
        signal = date.fromisoformat(str(row.signal_date))
        required = date.fromisoformat(str(row.required_options_date))
        if required != previous_trading_session(signal):
            raise ValueError("options date is not the exact previous trading session")

    structural_frame = structural.copy()
    bars_frame = bars.copy()
    structural_frame["symbol"] = structural_frame["symbol"].astype(str)
    structural_frame["session"] = structural_frame["session"].map(
        lambda value: _iso_date(value, name="structural session")
    )
    bars_frame["symbol"] = bars_frame["symbol"].astype(str)
    bars_frame["session"] = bars_frame["session"].map(
        lambda value: _iso_date(value, name="bar session")
    )
    sample_keys = pair_frame[["symbol", "signal_date"]].rename(columns={"signal_date": "session"})
    sample = structural_frame.merge(
        sample_keys,
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    if sample.empty:
        raise ValueError("no clean structural rows match the sampled option pairs")
    if sample["row_id"].duplicated().any():
        raise ValueError("sampled structural rows contain duplicate identities")
    movement = compute_underlying_movement_outcomes(sample, bars_frame)
    joined = movement.merge(
        pair_frame,
        left_on=["symbol", "session"],
        right_on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    if joined["pair_available"].isna().any():
        raise ValueError("movement row did not receive its exact sampled option-pair status")
    valid = joined.loc[joined["pair_available"].astype(bool)].copy()
    atm_iv = pd.to_numeric(valid["atm_iv"], errors="raise")
    if not np.isfinite(atm_iv.to_numpy(float)).all() or bool(atm_iv.le(0.0).any()):
        raise ValueError("valid sampled pair has invalid ATM IV")
    valid["iv_sigma_15m"] = atm_iv * math.sqrt(15 / (252 * 390))
    valid["iv_expected_absolute_15m"] = valid["iv_sigma_15m"] * math.sqrt(2 / math.pi)
    iv_relative = add_iv_relative_outcomes(valid)
    movement_sorted = joined.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    iv_sorted = iv_relative.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    return movement_sorted, iv_sorted


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError(f"weighted metric inputs are invalid for {column}")
    return float(np.average(values, weights=weights))


def _weighted_available(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    values = pd.to_numeric(frame[column], errors="raise")
    available = frame.loc[values.notna()]
    if available.empty:
        return 0, math.nan
    return len(available), _weighted_mean(available, column)


def _weighted_median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError(f"weighted median inputs are invalid for {column}")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, weights.sum() / 2.0, side="left"))
    return float(ordered_values[position])


def _state_metric_row(frame: pd.DataFrame, *, scope: str, route_state: str) -> dict[str, Any]:
    return_10_rows, mean_return_10 = _weighted_available(frame, "absolute_log_return_10m")
    return_30_rows, mean_return_30 = _weighted_available(frame, "absolute_log_return_30m")
    return_60_rows, mean_return_60 = _weighted_available(frame, "absolute_log_return_60m")
    completion = frame.loc[frame["registered_completion_in_bars_2_or_3"].eq(1)]
    before_rows, mean_before = _weighted_available(completion, "movement_before_completion")
    after_rows, mean_after = _weighted_available(
        completion, "movement_from_completion_to_horizon_end"
    )
    return {
        "scope": scope,
        "route_resolution_state": route_state,
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stock_sessions": int(frame[["symbol", "session"]].drop_duplicates().shape[0]),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(pd.to_datetime(frame["session"]).dt.to_period("M").nunique()),
        "weight_sum": float(pd.to_numeric(frame["row_weight"], errors="raise").sum()),
        "mean_absolute_log_return_15m": _weighted_mean(frame, "absolute_log_return_15m"),
        "median_absolute_log_return_15m": _weighted_median(frame, "absolute_log_return_15m"),
        "mean_iv_expected_absolute_15m": _weighted_mean(frame, "iv_expected_absolute_15m"),
        "mean_iv_absolute_residual_15m": _weighted_mean(frame, "iv_absolute_residual_15m"),
        "median_iv_absolute_residual_15m": _weighted_median(frame, "iv_absolute_residual_15m"),
        "mean_iv_sigma_ratio_15m": _weighted_mean(frame, "iv_sigma_ratio_15m"),
        "exceed_iv_expected_rate": _weighted_mean(frame, "movement_exceeds_iv_expected_absolute"),
        "exceed_one_iv_sigma_rate": _weighted_mean(frame, "movement_exceeds_one_iv_sigma"),
        "mean_realised_range_15m": _weighted_mean(frame, "realised_range_15m"),
        "mean_maximum_absolute_excursion_15m": _weighted_mean(
            frame, "maximum_absolute_excursion_15m"
        ),
        "mean_realised_variance_15m": _weighted_mean(frame, "realised_variance_15m"),
        "absolute_log_return_10m_available_rows": return_10_rows,
        "mean_absolute_log_return_10m": mean_return_10,
        "absolute_log_return_30m_available_rows": return_30_rows,
        "mean_absolute_log_return_30m": mean_return_30,
        "absolute_log_return_60m_available_rows": return_60_rows,
        "mean_absolute_log_return_60m": mean_return_60,
        "registered_completion_in_bars_2_or_3_rows": len(completion),
        "registered_completion_in_bars_2_or_3_rate": _weighted_mean(
            frame, "registered_completion_in_bars_2_or_3"
        ),
        "movement_before_completion_available_rows": before_rows,
        "mean_movement_before_completion": mean_before,
        "movement_from_completion_to_horizon_end_available_rows": after_rows,
        "mean_movement_from_completion_to_horizon_end": mean_after,
    }


def summarize_route_states(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize frozen route states and BROAD-minus-LOW descriptively."""

    states: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    scopes = {
        "pooled": panel,
        "development": panel.loc[panel["period"].eq("development")],
        "assessment": panel.loc[panel["period"].eq("assessment")],
    }
    for scope, scoped in scopes.items():
        if scoped.empty:
            continue
        rows_by_state: dict[str, dict[str, Any]] = {}
        for route_state, state_frame in scoped.groupby("route_resolution_state", sort=True):
            row = _state_metric_row(
                state_frame,
                scope=scope,
                route_state=str(route_state),
            )
            states.append(row)
            rows_by_state[str(route_state)] = row
        broad = rows_by_state.get("BROAD_CONFLICT")
        low = rows_by_state.get("LOW_ROUTE_SUPPORT")
        if broad is None or low is None:
            continue
        contrasts.append(
            {
                "scope": scope,
                "contrast": "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT",
                "broad_rows": broad["rows"],
                "low_rows": low["rows"],
                "absolute_log_return_15m_difference": (
                    broad["mean_absolute_log_return_15m"] - low["mean_absolute_log_return_15m"]
                ),
                "iv_absolute_residual_15m_difference": (
                    broad["mean_iv_absolute_residual_15m"] - low["mean_iv_absolute_residual_15m"]
                ),
                "iv_sigma_ratio_15m_difference": (
                    broad["mean_iv_sigma_ratio_15m"] - low["mean_iv_sigma_ratio_15m"]
                ),
                "exceed_iv_expected_rate_difference": (
                    broad["exceed_iv_expected_rate"] - low["exceed_iv_expected_rate"]
                ),
                "realised_range_15m_difference": (
                    broad["mean_realised_range_15m"] - low["mean_realised_range_15m"]
                ),
                "maximum_absolute_excursion_15m_difference": (
                    broad["mean_maximum_absolute_excursion_15m"]
                    - low["mean_maximum_absolute_excursion_15m"]
                ),
            }
        )
    return pd.DataFrame(states), pd.DataFrame(contrasts)


def _pooled_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, scoped in (
        ("pooled", panel),
        ("development", panel.loc[panel["period"].eq("development")]),
        ("assessment", panel.loc[panel["period"].eq("assessment")]),
    ):
        if scoped.empty:
            continue
        row = _state_metric_row(scoped, scope=scope, route_state="ALL")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_stock_dates(movement: pd.DataFrame, iv_relative: pd.DataFrame) -> pd.DataFrame:
    """Summarize every sampled stock-date without inventing IV data for failed pairs."""

    rows: list[dict[str, Any]] = []
    for (symbol, session), group in movement.groupby(["symbol", "session"], sort=True):
        symbol_text = str(symbol)
        session_text = str(session)
        valid = iv_relative.loc[
            iv_relative["symbol"].eq(symbol_text) & iv_relative["session"].eq(session_text)
        ]
        pair_available = bool(group["pair_available"].iloc[0])
        if pair_available != (not valid.empty):
            raise ValueError("stock-date pair availability differs between movement panels")
        row: dict[str, Any] = {
            "symbol": symbol_text,
            "session": session_text,
            "required_options_date": str(group["required_options_date"].iloc[0]),
            "period": str(group["period"].iloc[0]),
            "clean_rows": len(group),
            "valid_pair_rows": len(valid),
            "pair_available": pair_available,
            "mean_absolute_log_return_15m": _weighted_mean(group, "absolute_log_return_15m"),
            "mean_realised_range_15m": _weighted_mean(group, "realised_range_15m"),
            "mean_maximum_absolute_excursion_15m": _weighted_mean(
                group, "maximum_absolute_excursion_15m"
            ),
        }
        if not valid.empty:
            row.update(
                {
                    "front_dte": int(float(valid["front_dte"].iloc[0])),
                    "atm_iv": float(valid["atm_iv"].iloc[0]),
                    "mean_iv_expected_absolute_15m": _weighted_mean(
                        valid, "iv_expected_absolute_15m"
                    ),
                    "mean_iv_absolute_residual_15m": _weighted_mean(
                        valid, "iv_absolute_residual_15m"
                    ),
                    "mean_iv_sigma_ratio_15m": _weighted_mean(valid, "iv_sigma_ratio_15m"),
                    "exceed_iv_expected_rate": _weighted_mean(
                        valid, "movement_exceeds_iv_expected_absolute"
                    ),
                }
            )
        else:
            row.update(
                {
                    "front_dte": None,
                    "atm_iv": None,
                    "mean_iv_expected_absolute_15m": None,
                    "mean_iv_absolute_residual_15m": None,
                    "mean_iv_sigma_ratio_15m": None,
                    "exceed_iv_expected_rate": None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_required_fields(
    expected: pd.DataFrame, observed: pd.DataFrame, columns: list[str]
) -> dict[str, int | float]:
    """Compare every required cached-rerun field, including joins and timestamps."""

    missing_columns = len(
        set(columns).difference(expected.columns) | set(columns).difference(observed.columns)
    )
    expected_frame = expected.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    observed_frame = observed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    expected_ids = list(expected_frame["row_id"].astype(str))
    observed_ids = list(observed_frame["row_id"].astype(str))
    row_identity_mismatches = len(set(expected_ids).symmetric_difference(observed_ids))
    if missing_columns or expected_ids != observed_ids:
        return {
            "missing_columns": missing_columns,
            "row_identity_mismatches": row_identity_mismatches,
            "field_mismatches": missing_columns + row_identity_mismatches,
            "maximum_numeric_difference": math.inf,
        }
    field_mismatches = 0
    maximum_numeric_difference = 0.0
    for column in columns:
        left = expected_frame[column]
        right = observed_frame[column]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            left_values = pd.to_numeric(left, errors="raise").to_numpy(float)
            right_values = pd.to_numeric(right, errors="raise").to_numpy(float)
            both_missing = np.isnan(left_values) & np.isnan(right_values)
            difference = np.abs(left_values - right_values)
            difference[both_missing] = 0.0
            field_mismatches += int((difference > 0.0).sum())
            field_mismatches += int(
                np.logical_xor(np.isnan(left_values), np.isnan(right_values)).sum()
            )
            finite = difference[np.isfinite(difference)]
            if finite.size:
                maximum_numeric_difference = max(maximum_numeric_difference, float(finite.max()))
            continue
        left_missing = left.isna()
        right_missing = right.isna()
        field_mismatches += int(left_missing.ne(right_missing).sum())
        comparable = ~(left_missing | right_missing)
        field_mismatches += int(
            left.loc[comparable].astype(str).ne(right.loc[comparable].astype(str)).sum()
        )
    return {
        "missing_columns": missing_columns,
        "row_identity_mismatches": row_identity_mismatches,
        "field_mismatches": field_mismatches,
        "maximum_numeric_difference": maximum_numeric_difference,
    }


def _report(
    *,
    movement: pd.DataFrame,
    iv_relative: pd.DataFrame,
    pooled: pd.DataFrame,
    route_states: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    assessment = pooled.loc[pooled["scope"].eq("assessment")].iloc[0]
    broad = route_states.loc[
        route_states["scope"].eq("assessment")
        & route_states["route_resolution_state"].eq("BROAD_CONFLICT")
    ].iloc[0]
    low = route_states.loc[
        route_states["scope"].eq("assessment")
        & route_states["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")
    ].iloc[0]
    contrast = contrasts.loc[contrasts["scope"].eq("assessment")].iloc[0]
    return f"""# Three-stock prior-close IV movement outcomes V0.1

Decision: `descriptive_options_movement_structure_only`

The fixed sample produced {len(movement)} clean underlying-movement rows; {len(iv_relative)} rows
had a valid exact-previous-session ATM option pair and therefore support IV-relative outcomes.
Assessment support is {int(assessment["rows"])} rows from one session and three stocks.

Assessment weighted 15-minute absolute movement was
{float(assessment["mean_absolute_log_return_15m"]):.8f}; expected absolute movement from
previous-close ATM IV was {float(assessment["mean_iv_expected_absolute_15m"]):.8f}; and the mean
IV residual was {float(assessment["mean_iv_absolute_residual_15m"]):.8f}. The exceed-IV rate was
{float(assessment["exceed_iv_expected_rate"]):.4%}.

Assessment weighted absolute movement was
{float(assessment["mean_absolute_log_return_10m"]):.8f} at 10 minutes,
{float(assessment["mean_absolute_log_return_30m"]):.8f} at 30 minutes
({int(assessment["absolute_log_return_30m_available_rows"])} rows), and
{float(assessment["mean_absolute_log_return_60m"]):.8f} at 60 minutes
({int(assessment["absolute_log_return_60m_available_rows"])} rows). Registered completion occurred
in bars two or three for {int(assessment["registered_completion_in_bars_2_or_3_rows"])} assessment
rows.

`BROAD_CONFLICT` has {int(broad["rows"])} assessment rows and mean IV residual
{float(broad["mean_iv_absolute_residual_15m"]):.8f}. `LOW_ROUTE_SUPPORT` has
{int(low["rows"])} assessment rows and mean IV residual
{float(low["mean_iv_absolute_residual_15m"]):.8f}. Their descriptive residual difference is
{float(contrast["iv_absolute_residual_15m_difference"]):.8f}; their IV-sigma-ratio difference is
{float(contrast["iv_sigma_ratio_15m_difference"]):.8f}; and their exceed-IV-rate difference is
{float(contrast["exceed_iv_expected_rate_difference"]):.4%}.

This three-stock, three-date sample cannot pass the frozen coverage, stability, bootstrap,
matched-control, or model gates. No O0/O1/R0/R1 model was fit. The result does not establish the
binding broad-conflict hypothesis and makes no claim about option P&L, executable fills,
profitability, economic edge, prospective validation, or trading utility.
"""


def main() -> int:
    """Run the cached, no-network movement-outcome amendment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIR_RESULTS)
    if (
        set(pairs["symbol"].astype(str)) != {"AAL", "MSTR", "WULF"}
        or len(pairs) != 9
        or set(pairs["required_options_date"].astype(str))
        != {"2024-01-16", "2024-10-31", "2025-08-21"}
    ):
        raise RuntimeError("sampled pair scope changed")
    probe_audit = json.loads(PROBE_AUDIT.read_text(encoding="utf-8"))
    if not probe_audit.get("passed"):
        raise RuntimeError("cached option-pair probe audit did not pass")
    clean, reconstruction = load_clean_advance_panel()
    bars = pd.read_parquet(TRACE_PANEL)
    target_keys = pairs[["symbol", "signal_date"]].rename(columns={"signal_date": "session"})
    target_keys["session"] = target_keys["session"].astype(str)
    target_bars = bars.merge(
        target_keys,
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    movement, iv_relative = build_movement_panels(clean, target_bars, pairs)
    if (
        max(date.fromisoformat(value) for value in movement["session"].astype(str))
        >= PROTECTED_START
    ):
        raise RuntimeError("blocked_protected_boundary_failure")
    if len(movement) != 96 or len(iv_relative) != 86:
        raise RuntimeError("sampled movement support changed")

    route_states, contrasts = summarize_route_states(iv_relative)
    pooled = _pooled_metrics(iv_relative)
    stock_dates = summarize_stock_dates(movement, iv_relative)

    write_parquet(output / "movement_panel.parquet", movement)
    write_parquet(output / "iv_relative_movement_panel.parquet", iv_relative)

    reloaded_pairs = pd.read_csv(PAIR_RESULTS)
    reloaded_clean, _ = load_clean_advance_panel()
    reloaded_bars = pd.read_parquet(TRACE_PANEL)
    reloaded_keys = reloaded_pairs[["symbol", "signal_date"]].rename(
        columns={"signal_date": "session"}
    )
    reloaded_keys["session"] = reloaded_keys["session"].astype(str)
    reloaded_target_bars = reloaded_bars.merge(
        reloaded_keys,
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    repeated_movement, repeated_iv = build_movement_panels(
        reloaded_clean, reloaded_target_bars, reloaded_pairs
    )
    persisted_movement = pd.read_parquet(output / "movement_panel.parquet")
    persisted_iv = pd.read_parquet(output / "iv_relative_movement_panel.parquet")
    identity_and_join_columns = [
        "row_id",
        "symbol",
        "session",
        "checkpoint",
        "checkpoint_bar_ordinal_zero_based",
        "period",
        "route_resolution_state",
        "row_weight",
        "first_completion_lead",
        *[column for column in pairs.columns if column != "symbol"],
    ]
    movement_columns = [
        "entry_price",
        "absolute_log_return_10m",
        "absolute_log_return_15m",
        "absolute_log_return_30m",
        "absolute_log_return_60m",
        "realised_range_15m",
        "maximum_absolute_excursion_15m",
        "realised_variance_15m",
        "primary_horizon_last_bar_ordinal",
        "entry_bar_start_timestamp",
        "primary_horizon_last_bar_complete_timestamp",
        "registered_completion_in_bars_2_or_3",
        "movement_before_completion",
        "movement_from_completion_to_horizon_end",
    ]
    iv_columns = [
        "atm_iv",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
        "iv_absolute_residual_15m",
        "iv_sigma_ratio_15m",
        "movement_exceeds_iv_expected_absolute",
        "movement_exceeds_one_iv_sigma",
    ]
    movement_comparison = compare_required_fields(
        repeated_movement,
        persisted_movement,
        [*identity_and_join_columns, *movement_columns],
    )
    iv_comparison = compare_required_fields(
        repeated_iv,
        persisted_iv,
        [*identity_and_join_columns, *movement_columns, *iv_columns],
    )
    selected_contract_comparison = compare_required_fields(
        repeated_iv,
        persisted_iv,
        ["row_id", "call_contract_id", "put_contract_id"],
    )
    determinism = {
        "passed": movement_comparison["field_mismatches"] == 0
        and iv_comparison["field_mismatches"] == 0
        and movement_comparison["maximum_numeric_difference"] <= 1e-12
        and iv_comparison["maximum_numeric_difference"] <= 1e-12,
        "redownloaded": False,
        "network_requests_made": 0,
        "cached_inputs_reloaded": True,
        "output_panels_reloaded": True,
        "joined_row_mismatches": movement_comparison["row_identity_mismatches"],
        "iv_row_mismatches": iv_comparison["row_identity_mismatches"],
        "movement_field_mismatches": movement_comparison["field_mismatches"],
        "iv_field_mismatches": iv_comparison["field_mismatches"],
        "selected_contract_mismatches": selected_contract_comparison["field_mismatches"],
        "maximum_movement_difference": movement_comparison["maximum_numeric_difference"],
        "maximum_iv_relative_difference": iv_comparison["maximum_numeric_difference"],
        "models_refit": False,
    }
    if not determinism["passed"]:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")

    source_manifest = {
        "frozen_dense_panel": str(DENSE_PANEL.relative_to(REPO_ROOT)),
        "frozen_dense_panel_sha256": sha256_file(DENSE_PANEL),
        "frozen_trace_panel": str(TRACE_PANEL.relative_to(REPO_ROOT)),
        "frozen_trace_panel_sha256": sha256_file(TRACE_PANEL),
        "sampled_pair_results": str(PAIR_RESULTS.relative_to(REPO_ROOT)),
        "sampled_pair_results_sha256": sha256_file(PAIR_RESULTS),
        "sampled_pair_audit": str(PROBE_AUDIT.relative_to(REPO_ROOT)),
        "sampled_pair_audit_sha256": sha256_file(PROBE_AUDIT),
        "protected_rows_materialised": 0,
        "network_requests_made": 0,
    }
    join_audit = {
        "passed": True,
        "sampled_stock_dates": 9,
        "stock_dates_with_exact_previous_close_chain": 9,
        "stock_dates_with_valid_primary_pair": 8,
        "clean_underlying_movement_rows": len(movement),
        "valid_pair_iv_relative_rows": len(iv_relative),
        "development_iv_relative_rows": int(iv_relative["period"].eq("development").sum()),
        "assessment_iv_relative_rows": int(iv_relative["period"].eq("assessment").sum()),
        "assessment_sessions": int(
            iv_relative.loc[iv_relative["period"].eq("assessment"), "session"].nunique()
        ),
        "assessment_stocks": int(
            iv_relative.loc[iv_relative["period"].eq("assessment"), "symbol"].nunique()
        ),
        "structural_reconstruction_passed": bool(reconstruction["passed"]),
        "same_day_or_future_options_joins": 0,
        "protected_rows_materialised": 0,
    }
    decision = {
        **SAFETY_FLAGS,
        "decision": "descriptive_options_movement_structure_only",
        "options_download_status": "supported",
        "options_coverage_status": "insufficient_support",
        "iv_excess_model_status": "descriptive_only",
        "broad_conflict_movement_status": "descriptive_only",
        "matched_control_status": "insufficient_support",
        "models_fit": [],
        "binding_question_answered": False,
        "reason": "three_stock_three_date_sample_below_frozen_inference_gates",
    }
    outcome_manifest = {
        "entry_price": "open of first completed five-minute bar after checkpoint",
        "primary_horizon_minutes": 15,
        "primary_future_bars": 3,
        "primary_binary": "movement_exceeds_iv_expected_absolute",
        "primary_continuous": "iv_absolute_residual_15m",
        "iv_information_time": "exact previous trading day close",
        "annual_trading_minutes": 252 * 390,
        "secondary_horizons_minutes": [10, 30, 60],
        "option_pnl_calculated": False,
    }

    contract = json.loads((EXPERIMENT_DIR / "contract.json").read_text(encoding="utf-8"))
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "structural_panel_reconstruction.json", reconstruction)
    write_json(output / "movement_join_audit.json", join_audit)
    write_json(output / "outcome_manifest.json", outcome_manifest)
    write_json(output / "decision.json", decision)
    write_json(output / "determinism_check.json", determinism)
    write_csv(output / "pooled_movement_metrics.csv", pooled)
    write_csv(output / "route_state_movement_metrics.csv", route_states)
    write_csv(output / "route_state_contrasts.csv", contrasts)
    write_csv(output / "stock_date_movement_metrics.csv", stock_dates)
    (output / "report.md").write_text(
        _report(
            movement=movement,
            iv_relative=iv_relative,
            pooled=pooled,
            route_states=route_states,
            contrasts=contrasts,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
