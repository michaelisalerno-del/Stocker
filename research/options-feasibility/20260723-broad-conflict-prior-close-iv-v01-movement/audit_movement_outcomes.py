#!/usr/bin/env python3
"""Independently audit the cached three-stock movement-outcome amendment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as market_calendars

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PROTECTED_START = date(2025, 8, 23)
EXPECTED_SAFETY: dict[str, bool | str] = {
    "research_only": True,
    "options_feasibility_screen": True,
    "options_data_granularity": "end_of_day",
    "options_information_time": "previous_trading_day_close",
    "intraday_option_fill_simulated": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "economic_strategy_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _previous_nyse_session(signal: date) -> date:
    calendar = market_calendars.get_calendar("NYSE")
    sessions = calendar.valid_days(
        start_date=signal - timedelta(days=30),
        end_date=signal - timedelta(days=1),
        tz="America/New_York",
    )
    if len(sessions) == 0:
        raise AssertionError("NYSE calendar found no previous session")
    return cast(date, sessions[-1].date())


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> float:
    difference = abs(float(cast(Any, left)) - float(cast(Any, right)))
    if not math.isfinite(difference) or difference > tolerance:
        raise AssertionError(f"numeric mismatch: {left} versus {right}")
    return difference


def _positive(value: object, *, name: str) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number) or number <= 0.0:
        raise AssertionError(f"{name} is not finite and positive")
    return number


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise AssertionError("aggregate inputs are non-finite")
    return float(np.average(values, weights=weights))


def _weighted_available(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    available = frame.loc[pd.to_numeric(frame[column], errors="raise").notna()]
    if available.empty:
        return 0, math.nan
    return len(available), _weighted_mean(available, column)


def _weighted_median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, weights.sum() / 2.0, side="left"))
    return float(values[order][position])


def _scope(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "pooled":
        return frame
    return frame.loc[frame["period"].eq(name)]


def main() -> int:
    contract = json.loads((PRIMARY / "contract.json").read_text(encoding="utf-8"))
    decision = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    source = json.loads((PRIMARY / "source_manifest.json").read_text(encoding="utf-8"))
    join_audit = json.loads((PRIMARY / "movement_join_audit.json").read_text(encoding="utf-8"))
    if any(contract.get(key) != value for key, value in EXPECTED_SAFETY.items()):
        raise AssertionError("contract safety flags changed")
    if any(decision.get(key) != value for key, value in EXPECTED_SAFETY.items()):
        raise AssertionError("decision safety flags changed")
    if decision.get("decision") != "descriptive_options_movement_structure_only":
        raise AssertionError("three-stock decision exceeded its descriptive ceiling")
    if decision.get("models_fit") != [] or decision.get("binding_question_answered") is not False:
        raise AssertionError("descriptive run claims fitted models or binding inference")

    dense_path = REPO_ROOT / source["frozen_dense_panel"]
    trace_path = REPO_ROOT / source["frozen_trace_panel"]
    pair_path = REPO_ROOT / source["sampled_pair_results"]
    pair_audit_path = REPO_ROOT / source["sampled_pair_audit"]
    hash_mismatches = 0
    for path, expected in (
        (dense_path, source["frozen_dense_panel_sha256"]),
        (trace_path, source["frozen_trace_panel_sha256"]),
        (pair_path, source["sampled_pair_results_sha256"]),
        (pair_audit_path, source["sampled_pair_audit_sha256"]),
    ):
        hash_mismatches += int(_sha256(path) != expected)
    if hash_mismatches:
        raise AssertionError("frozen source hash mismatch")
    if not json.loads(pair_audit_path.read_text(encoding="utf-8")).get("passed"):
        raise AssertionError("option-pair source audit did not pass")

    movement = pd.read_parquet(PRIMARY / "movement_panel.parquet").sort_values(
        "row_id", kind="mergesort"
    )
    iv_panel = pd.read_parquet(PRIMARY / "iv_relative_movement_panel.parquet").sort_values(
        "row_id", kind="mergesort"
    )
    pairs = pd.read_csv(pair_path)
    dense = pd.read_parquet(dense_path)
    trace = pd.read_parquet(trace_path)
    pairs["signal_date"] = pairs["signal_date"].astype(str)
    pairs["required_options_date"] = pairs["required_options_date"].astype(str)
    pair_available = pairs["pair_available"].astype(str).str.casefold().eq("true")
    pairs["pair_available"] = pair_available
    if len(pairs) != 9 or int(pair_available.sum()) != 8:
        raise AssertionError("sampled option-pair support changed")
    chronology_mismatches = 0
    protected_mismatches = 0
    for pair_record in pairs.itertuples(index=False):
        signal = date.fromisoformat(str(pair_record.signal_date))
        required = date.fromisoformat(str(pair_record.required_options_date))
        chronology_mismatches += int(required != _previous_nyse_session(signal))
        protected_mismatches += int(signal >= PROTECTED_START or required >= PROTECTED_START)
    if chronology_mismatches or protected_mismatches:
        raise AssertionError("option chronology or protected boundary changed")

    explicit_clean = dense["registered_completion_next_1_bar"].fillna(0).astype(int).eq(0) & dense[
        "any_prefix_one_transition_from_completion"
    ].fillna(0).astype(int).eq(0)
    if int(explicit_clean.ne(dense["advance_eligible"].astype(int).eq(1)).sum()) != 0:
        raise AssertionError("clean advance reconstruction changed")
    key_frame = pairs[["symbol", "signal_date"]].rename(columns={"signal_date": "session"})
    expected_structural = dense.loc[explicit_clean].merge(
        key_frame,
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    expected_ids = set(expected_structural["row_id"].astype(str))
    movement_ids = set(movement["row_id"].astype(str))
    row_identity_mismatches = len(expected_ids.symmetric_difference(movement_ids))
    if row_identity_mismatches or len(movement) != 96 or movement["row_id"].duplicated().any():
        raise AssertionError("underlying movement row identities changed")
    valid_keys = set(
        zip(
            pairs.loc[pair_available, "symbol"].astype(str),
            pairs.loc[pair_available, "signal_date"].astype(str),
            strict=True,
        )
    )
    expected_iv_ids = set(
        movement.loc[
            [
                (str(symbol), str(session)) in valid_keys
                for symbol, session in zip(movement["symbol"], movement["session"], strict=True)
            ],
            "row_id",
        ].astype(str)
    )
    iv_ids = set(iv_panel["row_id"].astype(str))
    iv_row_mismatches = len(expected_iv_ids.symmetric_difference(iv_ids))
    if iv_row_mismatches or len(iv_panel) != 86 or iv_panel["row_id"].duplicated().any():
        raise AssertionError("IV-relative movement row identities changed")

    pairs_by_key = pairs.set_index(["symbol", "signal_date"], drop=False)
    option_join_field_mismatches = 0
    for row in movement.itertuples(index=False):
        pair_row = cast(pd.Series, pairs_by_key.loc[(str(row.symbol), str(row.session))])
        option_join_field_mismatches += int(
            str(row.required_options_date) != str(pair_row["required_options_date"])
            or bool(row.pair_available) != bool(pair_row["pair_available"])
            or str(row.call_contract_id) != str(pair_row["call_contract_id"])
            or str(row.put_contract_id) != str(pair_row["put_contract_id"])
        )
        if bool(pair_row["pair_available"]):
            option_join_field_mismatches += int(
                abs(float(cast(Any, row.atm_iv)) - float(cast(Any, pair_row["atm_iv"]))) > 1e-12
                or abs(float(cast(Any, row.front_dte)) - float(cast(Any, pair_row["front_dte"])))
                > 1e-12
            )
    if option_join_field_mismatches:
        raise AssertionError("cached option-pair join fields changed")

    expected_by_id = expected_structural.set_index("row_id", drop=False)
    trace_groups = {
        (str(symbol), str(session)): group.set_index("bar_ordinal", drop=False)
        for (symbol, session), group in trace.groupby(["symbol", "session"], sort=False)
    }
    maximum_movement_difference = 0.0
    timestamp_mismatches = 0
    structural_field_mismatches = 0
    for row in movement.itertuples(index=False):
        source_row = cast(pd.Series, expected_by_id.loc[str(row.row_id)])
        structural_field_mismatches += int(
            str(row.route_resolution_state) != str(source_row["route_resolution_state"])
            or int(cast(Any, row.checkpoint)) != int(cast(Any, source_row["checkpoint"]))
            or abs(float(cast(Any, row.row_weight)) - float(cast(Any, source_row["row_weight"])))
            > 1e-12
        )
        checkpoint = int(cast(Any, source_row["checkpoint_bar_ordinal_zero_based"]))
        indexed = trace_groups[(str(row.symbol), str(row.session))]
        future = indexed.loc[[checkpoint + 1, checkpoint + 2, checkpoint + 3]]
        entry = _positive(future.iloc[0]["open"], name="entry")
        closes = [_positive(value, name="close") for value in future["close"]]
        highs = [_positive(value, name="high") for value in future["high"]]
        lows = [_positive(value, name="low") for value in future["low"]]
        five_minute_returns = [
            math.log(closes[0] / entry),
            math.log(closes[1] / closes[0]),
            math.log(closes[2] / closes[1]),
        ]
        expected_values = {
            "entry_price": entry,
            "absolute_log_return_10m": abs(math.log(closes[1] / entry)),
            "absolute_log_return_15m": abs(math.log(closes[2] / entry)),
            "realised_range_15m": math.log(max(highs) / min(lows)),
            "maximum_absolute_excursion_15m": max(
                abs(math.log(max(highs) / entry)), abs(math.log(min(lows) / entry))
            ),
            "realised_variance_15m": sum(value * value for value in five_minute_returns),
        }
        for column, expected in expected_values.items():
            maximum_movement_difference = max(
                maximum_movement_difference, _close(getattr(row, column), expected)
            )
        for bars_forward, column in (
            (6, "absolute_log_return_30m"),
            (12, "absolute_log_return_60m"),
        ):
            ordinal = checkpoint + bars_forward
            if ordinal in indexed.index:
                expected = abs(
                    math.log(_positive(indexed.loc[ordinal, "close"], name=column) / entry)
                )
                maximum_movement_difference = max(
                    maximum_movement_difference, _close(getattr(row, column), expected)
                )
            elif not pd.isna(getattr(row, column)):
                raise AssertionError(f"{column} should be unavailable")
        timestamp_mismatches += int(
            pd.Timestamp(cast(Any, row.entry_bar_start_timestamp))
            != pd.Timestamp(cast(Any, future.iloc[0]["bar_start_timestamp"]))
            or pd.Timestamp(cast(Any, row.primary_horizon_last_bar_complete_timestamp))
            != pd.Timestamp(cast(Any, future.iloc[-1]["bar_complete_timestamp"]))
        )
        lead_value = source_row["first_completion_lead"]
        lead: int | None = None if pd.isna(lead_value) else int(cast(Any, lead_value))
        if int(cast(Any, row.registered_completion_in_bars_2_or_3)) != int(lead in {2, 3}):
            raise AssertionError("completion-within-horizon indicator changed")
        if lead in {2, 3}:
            completion_close = closes[lead - 1]
            maximum_movement_difference = max(
                maximum_movement_difference,
                _close(row.movement_before_completion, abs(math.log(completion_close / entry))),
                _close(
                    row.movement_from_completion_to_horizon_end,
                    abs(math.log(closes[2] / completion_close)),
                ),
            )
    if timestamp_mismatches or structural_field_mismatches:
        raise AssertionError("movement timing or structural surface changed")

    maximum_iv_difference = 0.0
    for row in iv_panel.itertuples(index=False):
        sigma = float(cast(Any, row.atm_iv)) * math.sqrt(15 / (252 * 390))
        expected = sigma * math.sqrt(2 / math.pi)
        movement_value = float(cast(Any, row.absolute_log_return_15m))
        maximum_iv_difference = max(
            maximum_iv_difference,
            _close(row.iv_sigma_15m, sigma),
            _close(row.iv_expected_absolute_15m, expected),
            _close(row.iv_absolute_residual_15m, movement_value - expected),
            _close(row.iv_sigma_ratio_15m, movement_value / sigma),
        )
        if int(cast(Any, row.movement_exceeds_iv_expected_absolute)) != int(
            movement_value > expected
        ):
            raise AssertionError("exceed-expected indicator changed")
        if int(cast(Any, row.movement_exceeds_one_iv_sigma)) != int(movement_value > sigma):
            raise AssertionError("exceed-sigma indicator changed")

    route_metrics = pd.read_csv(PRIMARY / "route_state_movement_metrics.csv")
    maximum_metric_difference = 0.0
    for metric in route_metrics.itertuples(index=False):
        scoped = _scope(iv_panel, str(metric.scope))
        state = scoped.loc[scoped["route_resolution_state"].eq(str(metric.route_resolution_state))]
        if len(state) != int(cast(Any, metric.rows)):
            raise AssertionError("route-state support count changed")
        expected_metrics = {
            "mean_absolute_log_return_15m": _weighted_mean(state, "absolute_log_return_15m"),
            "median_absolute_log_return_15m": _weighted_median(state, "absolute_log_return_15m"),
            "mean_iv_expected_absolute_15m": _weighted_mean(state, "iv_expected_absolute_15m"),
            "mean_iv_absolute_residual_15m": _weighted_mean(state, "iv_absolute_residual_15m"),
            "median_iv_absolute_residual_15m": _weighted_median(state, "iv_absolute_residual_15m"),
            "mean_iv_sigma_ratio_15m": _weighted_mean(state, "iv_sigma_ratio_15m"),
            "exceed_iv_expected_rate": _weighted_mean(
                state, "movement_exceeds_iv_expected_absolute"
            ),
            "registered_completion_in_bars_2_or_3_rate": _weighted_mean(
                state, "registered_completion_in_bars_2_or_3"
            ),
        }
        for horizon in (10, 30, 60):
            column = f"absolute_log_return_{horizon}m"
            available_rows, expected_mean = _weighted_available(state, column)
            if int(cast(Any, getattr(metric, f"{column}_available_rows"))) != available_rows:
                raise AssertionError("secondary-horizon support count changed")
            if available_rows:
                expected_metrics[f"mean_{column}"] = expected_mean
        completion = state.loc[state["registered_completion_in_bars_2_or_3"].eq(1)]
        if int(cast(Any, metric.registered_completion_in_bars_2_or_3_rows)) != len(completion):
            raise AssertionError("completion-relative support count changed")
        for source_column, output_column in (
            ("movement_before_completion", "mean_movement_before_completion"),
            (
                "movement_from_completion_to_horizon_end",
                "mean_movement_from_completion_to_horizon_end",
            ),
        ):
            available_rows, expected_mean = _weighted_available(completion, source_column)
            if available_rows:
                expected_metrics[output_column] = expected_mean
        for column, expected in expected_metrics.items():
            maximum_metric_difference = max(
                maximum_metric_difference, _close(getattr(metric, column), expected)
            )
    contrasts = pd.read_csv(PRIMARY / "route_state_contrasts.csv")
    for contrast in contrasts.itertuples(index=False):
        rows = route_metrics.loc[route_metrics["scope"].eq(str(contrast.scope))].set_index(
            "route_resolution_state"
        )
        broad = cast(pd.Series, rows.loc["BROAD_CONFLICT"])
        low = cast(pd.Series, rows.loc["LOW_ROUTE_SUPPORT"])
        for output_column, source_column in (
            ("iv_absolute_residual_15m_difference", "mean_iv_absolute_residual_15m"),
            ("iv_sigma_ratio_15m_difference", "mean_iv_sigma_ratio_15m"),
            ("exceed_iv_expected_rate_difference", "exceed_iv_expected_rate"),
        ):
            expected = float(cast(Any, broad[source_column])) - float(cast(Any, low[source_column]))
            maximum_metric_difference = max(
                maximum_metric_difference, _close(getattr(contrast, output_column), expected)
            )

    if not join_audit.get("passed") or join_audit.get("assessment_sessions") != 1:
        raise AssertionError("join audit support changed")
    determinism = json.loads((PRIMARY / "determinism_check.json").read_text(encoding="utf-8"))
    if (
        not determinism.get("passed")
        or not determinism.get("cached_inputs_reloaded")
        or not determinism.get("output_panels_reloaded")
        or determinism.get("movement_field_mismatches") != 0
        or determinism.get("iv_field_mismatches") != 0
        or determinism.get("selected_contract_mismatches") != 0
    ):
        raise AssertionError("cached determinism rerun did not pass")
    audit = {
        "passed": True,
        "status": "three_stock_movement_outcomes_audited",
        "safety_flags_verified": True,
        "source_hash_mismatches": hash_mismatches,
        "chronology_mismatches": chronology_mismatches,
        "protected_boundary_mismatches": protected_mismatches,
        "row_identity_mismatches": row_identity_mismatches,
        "iv_row_mismatches": iv_row_mismatches,
        "option_join_field_mismatches": option_join_field_mismatches,
        "structural_field_mismatches": structural_field_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
        "underlying_rows_verified": len(movement),
        "iv_relative_rows_verified": len(iv_panel),
        "maximum_movement_difference": maximum_movement_difference,
        "maximum_iv_relative_difference": maximum_iv_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "primary_horizon_future_bars": 3,
        "network_requests_made": 0,
        "models_verified_absent": True,
        "decision_logic_verified": True,
    }
    _write_json(PRIMARY / "lightweight_audit.json", audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
