"""Role-aware second-stage cutter for state event detector outputs.

This research-only lab consumes sparse ``state_event_detector_v0`` event rows
and scores simple filters according to the event state's intended role. Negative
forward returns are not treated as failures for blocker/short states.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.event_failure_cutter_v0 import (
    add_rolling_symbol_state_efficacy,
    find_latest_state_event_detector_run,
)

DEFAULT_INPUT_BASE_DIR = Path("data/reports/research/state_event_detector_v0")
DEFAULT_OUTPUT_DIR = Path("data/reports/research/role_aware_event_cutter_v0")

EVENT_STATE_ROLES: dict[str, dict[str, int | str]] = {
    "controlled_pullback_after_bullish_impulse": {
        "role": "long_candidate",
        "default_expected_direction": 1,
    },
    "liquidation_failed_low_reclaim": {
        "role": "long_reversal_candidate",
        "default_expected_direction": 1,
    },
    "slow_snapback_after_dip": {
        "role": "long_reversal_candidate",
        "default_expected_direction": 1,
    },
    "failed_bounce_active_liquidation": {
        "role": "long_blocker_or_short_candidate",
        "default_expected_direction": -1,
    },
    "failed_bullish_impulse_recoil": {
        "role": "long_blocker_or_short_candidate",
        "default_expected_direction": -1,
    },
    "failed_open_down_continuation": {
        "role": "long_blocker_or_short_candidate",
        "default_expected_direction": -1,
    },
    "dead_chop_blocker": {
        "role": "no_trade_filter",
        "default_expected_direction": 0,
    },
}

FILTER_FEATURES = (
    "distance_from_vwap_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_opening_range_high_pct",
    "distance_from_opening_range_low_pct",
    "distance_from_session_open_pct",
    "distance_from_session_high_pct",
    "distance_from_session_low_pct",
    "distance_from_recent_high_pct",
    "distance_from_recent_low_pct",
    "close_location_value",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",
    "bar_return",
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "return_deceleration_3_vs_6",
    "return_deceleration_6_vs_12",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "bar_index_in_session",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
)

REQUIRED_INPUT_FILES = (
    "event_rows.csv",
    "summary.json",
    "summary.md",
    "decision.json",
)


@dataclass(frozen=True)
class RoleAwareEventCutterConfig:
    """Configuration for role-aware event cutter research."""

    horizons: tuple[int, ...] = (6, 9, 12, 24)
    train_fraction: float = 0.60
    random_seed: int = 1337
    random_iterations: int = 50
    min_train_events: int = 30
    min_test_events: int = 20
    min_retained_events: int = 10
    min_retained_pct: float = 0.05
    max_retained_pct: float = 0.95
    min_oos_lift_bps: float = 0.0
    min_random_excess_bps: float = 0.0
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    low_movement_threshold: float = 0.001
    top_single_filters_for_pairs: int = 5
    max_candidates_per_state_horizon: int = 24
    rolling_windows: tuple[int, ...] = (20, 60)


@dataclass(frozen=True)
class RoleAwareEventCutterResult:
    """Paths and headline decision from a role-aware cutter run."""

    run_id: str
    input_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    role_aware_state_summary_csv_path: Path
    aligned_directional_results_csv_path: Path
    long_candidate_filter_results_csv_path: Path
    blocker_quality_results_csv_path: Path
    short_candidate_results_csv_path: Path
    no_trade_quality_results_csv_path: Path
    role_evidence_conflicts_csv_path: Path
    filter_oos_results_csv_path: Path
    random_role_baselines_csv_path: Path
    concentration_warnings_csv_path: Path
    selected_filters_csv_path: Path
    rejected_filters_csv_path: Path
    decision: str
    selected_filter_count: int


def _return_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_return"


def _mfe_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mfe"


def _mae_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mae"


def _role_info(event_state: str) -> dict[str, int | str]:
    return EVENT_STATE_ROLES.get(
        str(event_state),
        {"role": "unknown", "default_expected_direction": 0},
    )


def _role(event_state: str) -> str:
    return str(_role_info(event_state)["role"])


def _default_direction(event_state: str) -> int:
    return int(_role_info(event_state)["default_expected_direction"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _nanmedian_or_nan(values: Sequence[float]) -> float:
    finite = [value for value in values if not math.isnan(float(value))]
    return float(np.median(finite)) if finite else math.nan


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 12) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _validate_input_dir(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_INPUT_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required state-event detector files: {missing}")


def _split_train_test(
    rows: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return rows.copy(), rows.copy()
    data = rows.copy()
    data["_timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.sort_values(["_timestamp", "symbol", "event_state"], kind="mergesort")
    if len(data) < 2:
        return data.drop(columns=["_timestamp"]), data.iloc[0:0].drop(columns=["_timestamp"])
    train_count = max(1, min(len(data) - 1, int(len(data) * train_fraction)))
    train = data.iloc[:train_count].drop(columns=["_timestamp"])
    test = data.iloc[train_count:].drop(columns=["_timestamp"])
    return train, test


def _safe_share(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def _month_values(rows: pd.DataFrame) -> pd.Series:
    if "session_date" not in rows:
        return pd.Series(dtype=str)
    return pd.to_datetime(rows["session_date"], errors="coerce").dt.strftime("%Y-%m")


def _max_share(values: pd.Series) -> float:
    if values.empty:
        return math.nan
    normalized = values.dropna().astype(str)
    if normalized.empty:
        return math.nan
    return float(normalized.value_counts(normalize=True).max())


def _concentration_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "symbol_count": 0,
            "single_symbol_share": math.nan,
            "session_count": 0,
            "single_session_share": math.nan,
            "month_count": 0,
            "single_month_share": math.nan,
        }
    months = _month_values(rows)
    return {
        "symbol_count": int(rows["symbol"].nunique()) if "symbol" in rows else 0,
        "single_symbol_share": _max_share(rows["symbol"]) if "symbol" in rows else math.nan,
        "session_count": int(rows["session_date"].nunique()) if "session_date" in rows else 0,
        "single_session_share": _max_share(rows["session_date"])
        if "session_date" in rows
        else math.nan,
        "month_count": int(months.dropna().nunique()),
        "single_month_share": _max_share(months),
    }


def _has_concentration_warning(
    metrics: dict[str, Any],
    config: RoleAwareEventCutterConfig,
) -> bool:
    symbol_share = float(metrics.get("single_symbol_share", math.nan))
    session_share = float(metrics.get("single_session_share", math.nan))
    month_share = float(metrics.get("single_month_share", math.nan))
    return bool(
        (not math.isnan(symbol_share) and symbol_share > config.max_single_symbol_share)
        or (not math.isnan(session_share) and session_share > config.max_single_session_share)
        or (not math.isnan(month_share) and month_share > config.max_single_month_share)
    )


def _concentration_warning_rows(
    *,
    metrics: dict[str, Any],
    event_state: str,
    horizon: int,
    filter_id: str,
    config: RoleAwareEventCutterConfig,
) -> list[dict[str, Any]]:
    checks = (
        ("single_symbol_dominates", "single_symbol_share", config.max_single_symbol_share),
        ("single_session_dominates", "single_session_share", config.max_single_session_share),
        ("single_month_dominates", "single_month_share", config.max_single_month_share),
    )
    rows: list[dict[str, Any]] = []
    for warning, key, threshold in checks:
        value = float(metrics.get(key, math.nan))
        if not math.isnan(value) and value > threshold:
            rows.append(
                {
                    "event_state": event_state,
                    "horizon": int(horizon),
                    "filter_id": filter_id,
                    "warning": warning,
                    "value": value,
                    "threshold": threshold,
                }
            )
    return rows


def add_aligned_return_column(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
) -> pd.DataFrame:
    """Add an aligned return column for a role/horizon."""

    data = rows.copy()
    returns = pd.to_numeric(data[_return_col(horizon)], errors="coerce")
    column = f"aligned_{horizon}_bar_return"
    if expected_direction == 0:
        data[column] = -returns.abs()
    else:
        data[column] = returns * expected_direction
    return data


def estimate_role_direction(
    *,
    event_state: str,
    train_rows: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """Estimate train direction while preserving the explicit role default."""

    default = _default_direction(event_state)
    returns = pd.to_numeric(
        train_rows.get(_return_col(horizon), pd.Series(dtype=float)),
        errors="coerce",
    )
    train_median = float(returns.median()) if returns.notna().any() else math.nan
    inferred = 0
    if not math.isnan(train_median):
        if train_median > 0.0:
            inferred = 1
        elif train_median < 0.0:
            inferred = -1
    expected = 0 if default == 0 else default
    conflict = bool(default != 0 and inferred != 0 and inferred != default)
    return {
        "event_state": event_state,
        "role": _role(event_state),
        "horizon": int(horizon),
        "default_expected_direction": int(default),
        "train_inferred_direction": int(inferred),
        "expected_direction": int(expected),
        "train_median_forward_return": train_median,
        "role_evidence_conflict": conflict,
    }


def _available_filter_features(rows: pd.DataFrame) -> list[str]:
    features = [column for column in FILTER_FEATURES if column in rows.columns]
    rolling = [column for column in rows.columns if column.startswith("symbol_state_h")]
    return features + [column for column in rolling if column not in features]


def _add_derived_features(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    if {"prior_3_bar_return", "prior_6_bar_return"}.issubset(data.columns):
        data["return_deceleration_3_vs_6"] = (
            pd.to_numeric(data["prior_3_bar_return"], errors="coerce")
            - pd.to_numeric(data["prior_6_bar_return"], errors="coerce")
        )
    if {"prior_6_bar_return", "prior_12_bar_return"}.issubset(data.columns):
        data["return_deceleration_6_vs_12"] = (
            pd.to_numeric(data["prior_6_bar_return"], errors="coerce")
            - pd.to_numeric(data["prior_12_bar_return"], errors="coerce")
        )
    return data


def _feature_thresholds(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.nunique() < 2:
        return []
    thresholds = [float(value) for value in numeric.quantile([0.25, 0.50, 0.75]).tolist()]
    if float(numeric.min()) <= 0.0 <= float(numeric.max()):
        thresholds.append(0.0)
    output: list[float] = []
    for value in sorted(set(round(item, 12) for item in thresholds)):
        if math.isfinite(value):
            output.append(float(value))
    return output


def _compare_feature(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if operator == "<=":
        return values <= threshold
    if operator == ">=":
        return values >= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _apply_candidate_rule(rows: pd.DataFrame, candidate: pd.Series) -> pd.Series:
    first = _compare_feature(
        rows[str(candidate["feature_1"])],
        str(candidate["operator_1"]),
        float(candidate["threshold_1"]),
    )
    if str(candidate["rule_type"]) == "single":
        return first.fillna(False)
    second = _compare_feature(
        rows[str(candidate["feature_2"])],
        str(candidate["operator_2"]),
        float(candidate["threshold_2"]),
    )
    if str(candidate["logical_operator"]) == "AND":
        return (first & second).fillna(False)
    return (first | second).fillna(False)


def _objective_metric(metrics: dict[str, Any], role: str) -> float:
    if role == "no_trade_filter":
        return float(metrics.get("no_trade_quality_score_after", math.nan))
    if role == "long_blocker_or_short_candidate":
        blocker = float(metrics.get("blocker_net_value_bps", math.nan)) / 10_000
        aligned = float(metrics.get("aligned_median_return_after", math.nan))
        if not math.isnan(blocker) and blocker > 0.0:
            return blocker
        return aligned
    return float(metrics.get("aligned_median_return_after", math.nan))


def _objective_before(metrics: dict[str, Any], role: str) -> float:
    if role == "no_trade_filter":
        return float(metrics.get("no_trade_quality_score_before", math.nan))
    return float(metrics.get("aligned_median_return_before", math.nan))


def evaluate_role_rows(
    rows: pd.DataFrame,
    retained_mask: pd.Series,
    *,
    event_state: str,
    horizon: int,
    expected_direction: int,
    role: str,
    low_movement_threshold: float = 0.001,
) -> dict[str, Any]:
    """Evaluate retained rows according to the event-state role."""

    mask = retained_mask.reindex(rows.index).fillna(False).astype(bool)
    retained = rows[mask]
    returns = pd.to_numeric(rows.get(_return_col(horizon), pd.Series(dtype=float)), errors="coerce")
    retained_returns = pd.to_numeric(
        retained.get(_return_col(horizon), pd.Series(dtype=float)),
        errors="coerce",
    )
    aligned_all = add_aligned_return_column(
        rows,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    aligned_retained = add_aligned_return_column(
        retained,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    aligned = pd.to_numeric(aligned_all[f"aligned_{horizon}_bar_return"], errors="coerce")
    retained_aligned = pd.to_numeric(
        aligned_retained[f"aligned_{horizon}_bar_return"],
        errors="coerce",
    )
    metrics = {
        "event_state": event_state,
        "horizon": int(horizon),
        "role": role,
        "expected_direction": int(expected_direction),
        "event_count": int(len(rows)),
        "retained_count": int(len(retained)),
        "dropped_count": int(len(rows) - len(retained)),
        "retained_pct": _safe_share(len(retained), len(rows)),
        "raw_median_forward_return_before": float(returns.median())
        if returns.notna().any()
        else math.nan,
        "raw_median_forward_return_after": float(retained_returns.median())
        if retained_returns.notna().any()
        else math.nan,
        "aligned_median_return_before": float(aligned.median())
        if aligned.notna().any()
        else math.nan,
        "aligned_median_return_after": float(retained_aligned.median())
        if retained_aligned.notna().any()
        else math.nan,
        "aligned_win_rate_before": float((aligned > 0.0).mean())
        if aligned.notna().any()
        else math.nan,
        "aligned_win_rate_after": float((retained_aligned > 0.0).mean())
        if retained_aligned.notna().any()
        else math.nan,
        "directional_consistency_after": float((retained_aligned > 0.0).mean())
        if retained_aligned.notna().any()
        else math.nan,
        "wrong_way_rate_after": float((retained_aligned < 0.0).mean())
        if retained_aligned.notna().any()
        else math.nan,
    }
    mfe = pd.to_numeric(
        retained.get(_mfe_col(horizon), pd.Series(np.nan, index=retained.index)),
        errors="coerce",
    )
    mae = pd.to_numeric(
        retained.get(_mae_col(horizon), pd.Series(np.nan, index=retained.index)),
        errors="coerce",
    )
    metrics.update(_concentration_metrics(retained))
    if role in {"long_candidate", "long_reversal_candidate"}:
        metrics.update(
            {
                "long_median_forward_return": metrics["raw_median_forward_return_after"],
                "long_win_rate": float((retained_returns > 0.0).mean())
                if retained_returns.notna().any()
                else math.nan,
                "long_mfe": float(mfe.median()) if mfe.notna().any() else math.nan,
                "long_mae": float(mae.median()) if mae.notna().any() else math.nan,
            }
        )
    if role == "long_blocker_or_short_candidate":
        bad_long = returns <= 0.0
        good_long = returns > 0.0
        retained_bad = mask & bad_long
        retained_good = mask & good_long
        avoided = float((-returns[retained_bad]).median() * 10_000) if retained_bad.any() else 0.0
        missed = float(returns[retained_good].median() * 10_000) if retained_good.any() else 0.0
        metrics.update(
            {
                "bad_long_capture_rate": _safe_share(int(retained_bad.sum()), int(bad_long.sum())),
                "good_long_false_block_rate": _safe_share(
                    int(retained_good.sum()),
                    int(good_long.sum()),
                ),
                "avoided_long_loss_bps": avoided,
                "missed_long_profit_bps": missed,
                "blocker_net_value_bps": avoided - missed,
                "short_median_return_after": float((-retained_returns).median())
                if retained_returns.notna().any()
                else math.nan,
                "short_win_rate_after": float((retained_returns < 0.0).mean())
                if retained_returns.notna().any()
                else math.nan,
                "short_mfe": float((-mae).median()) if mae.notna().any() else math.nan,
                "short_mae": float((-mfe).median()) if mfe.notna().any() else math.nan,
                "short_directional_accuracy": float((retained_returns < 0.0).mean())
                if retained_returns.notna().any()
                else math.nan,
            }
        )
    if role == "no_trade_filter":
        abs_returns = returns.abs()
        retained_abs = retained_returns.abs()
        low_before = (
            float((abs_returns <= low_movement_threshold).mean())
            if abs_returns.notna().any()
            else math.nan
        )
        low_after = (
            float((retained_abs <= low_movement_threshold).mean())
            if retained_abs.notna().any()
            else math.nan
        )
        big_before = (
            float((abs_returns > low_movement_threshold).mean())
            if abs_returns.notna().any()
            else math.nan
        )
        big_after = (
            float((retained_abs > low_movement_threshold).mean())
            if retained_abs.notna().any()
            else math.nan
        )
        metrics.update(
            {
                "median_abs_forward_return_before": float(abs_returns.median())
                if abs_returns.notna().any()
                else math.nan,
                "median_abs_forward_return_after": float(retained_abs.median())
                if retained_abs.notna().any()
                else math.nan,
                "median_mfe_after": float(mfe.median()) if mfe.notna().any() else math.nan,
                "median_mae_after": float(mae.median()) if mae.notna().any() else math.nan,
                "low_movement_rate_before": low_before,
                "low_movement_rate_after": low_after,
                "false_block_big_move_rate_before": big_before,
                "false_block_big_move_rate_after": big_after,
                "no_trade_quality_score_before": low_before - big_before
                if not math.isnan(low_before) and not math.isnan(big_before)
                else math.nan,
                "no_trade_quality_score_after": low_after - big_after
                if not math.isnan(low_after) and not math.isnan(big_after)
                else math.nan,
            }
        )
    objective_before = _objective_before(metrics, role)
    objective_after = _objective_metric(metrics, role)
    metrics["role_objective_before"] = objective_before
    metrics["role_objective_after"] = objective_after
    metrics["role_objective_lift_bps"] = (
        (objective_after - objective_before) * 10_000
        if not math.isnan(objective_after) and not math.isnan(objective_before)
        else math.nan
    )
    return metrics


def _candidate_record(
    *,
    event_state: str,
    horizon: int,
    role: str,
    expected_direction: int,
    rule_type: str,
    feature_1: str,
    operator_1: str,
    threshold_1: float,
    metrics: dict[str, Any],
    direction: dict[str, Any],
    feature_2: str | None = None,
    operator_2: str | None = None,
    threshold_2: float | None = None,
    logical_operator: str | None = None,
) -> dict[str, Any]:
    if rule_type == "single":
        expression = f"{feature_1} {operator_1} {threshold_1:g}"
    else:
        expression = (
            f"({feature_1} {operator_1} {threshold_1:g}) "
            f"{logical_operator} ({feature_2} {operator_2} {threshold_2:g})"
        )
    return {
        "filter_id": "",
        "event_state": event_state,
        "horizon": int(horizon),
        "role": role,
        "expected_direction": int(expected_direction),
        "default_expected_direction": int(direction["default_expected_direction"]),
        "train_inferred_direction": int(direction["train_inferred_direction"]),
        "role_evidence_conflict": bool(direction["role_evidence_conflict"]),
        "rule_type": rule_type,
        "filter_expression": expression,
        "feature_1": feature_1,
        "operator_1": operator_1,
        "threshold_1": float(threshold_1),
        "feature_2": feature_2 or "",
        "operator_2": operator_2 or "",
        "threshold_2": math.nan if threshold_2 is None else float(threshold_2),
        "logical_operator": logical_operator or "",
        **{f"train_{key}": value for key, value in metrics.items()},
    }


def build_candidate_filters(
    train_rows: pd.DataFrame,
    *,
    event_state: str,
    horizon: int,
    config: RoleAwareEventCutterConfig,
) -> pd.DataFrame:
    """Build train-selected single and two-feature rules for one state/horizon."""

    return_column = _return_col(horizon)
    if return_column not in train_rows:
        return pd.DataFrame()
    group = train_rows[train_rows["event_state"].astype(str).eq(event_state)].copy()
    group = _add_derived_features(group)
    group = group[pd.to_numeric(group[return_column], errors="coerce").notna()]
    if len(group) < config.min_train_events:
        return pd.DataFrame()
    role = _role(event_state)
    direction = estimate_role_direction(
        event_state=event_state,
        train_rows=group,
        horizon=horizon,
    )
    expected_direction = int(direction["expected_direction"])
    records: list[dict[str, Any]] = []
    for feature in _available_filter_features(group):
        for threshold in _feature_thresholds(group[feature]):
            for operator in ("<=", ">="):
                mask = _compare_feature(group[feature], operator, threshold).fillna(False)
                metrics = evaluate_role_rows(
                    group,
                    mask,
                    event_state=event_state,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                    low_movement_threshold=config.low_movement_threshold,
                )
                retained_pct = float(metrics["retained_pct"])
                if int(metrics["retained_count"]) < config.min_retained_events:
                    continue
                if retained_pct < config.min_retained_pct or retained_pct > config.max_retained_pct:
                    continue
                records.append(
                    _candidate_record(
                        event_state=event_state,
                        horizon=horizon,
                        role=role,
                        expected_direction=expected_direction,
                        rule_type="single",
                        feature_1=feature,
                        operator_1=operator,
                        threshold_1=threshold,
                        metrics=metrics,
                        direction=direction,
                    )
                )
    singles = pd.DataFrame(records)
    if singles.empty:
        return singles
    singles = singles.sort_values(
        ["train_role_objective_lift_bps", "train_retained_count"],
        ascending=[False, False],
    ).head(config.max_candidates_per_state_horizon)
    top_records = singles.head(config.top_single_filters_for_pairs).to_dict("records")
    combo_records: list[dict[str, Any]] = []
    for left_index, left in enumerate(top_records):
        for right in top_records[left_index + 1 :]:
            if left["feature_1"] == right["feature_1"]:
                continue
            left_mask = _apply_candidate_rule(group, pd.Series(left))
            right_mask = _apply_candidate_rule(group, pd.Series(right))
            for logical_operator, combo_mask in (
                ("AND", left_mask & right_mask),
                ("OR", left_mask | right_mask),
            ):
                metrics = evaluate_role_rows(
                    group,
                    combo_mask,
                    event_state=event_state,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                    low_movement_threshold=config.low_movement_threshold,
                )
                retained_pct = float(metrics["retained_pct"])
                if int(metrics["retained_count"]) < config.min_retained_events:
                    continue
                if retained_pct < config.min_retained_pct or retained_pct > config.max_retained_pct:
                    continue
                combo_records.append(
                    _candidate_record(
                        event_state=event_state,
                        horizon=horizon,
                        role=role,
                        expected_direction=expected_direction,
                        rule_type=f"two_feature_{logical_operator.lower()}",
                        feature_1=str(left["feature_1"]),
                        operator_1=str(left["operator_1"]),
                        threshold_1=float(left["threshold_1"]),
                        feature_2=str(right["feature_1"]),
                        operator_2=str(right["operator_1"]),
                        threshold_2=float(right["threshold_1"]),
                        logical_operator=logical_operator,
                        metrics=metrics,
                        direction=direction,
                    )
                )
    candidates = pd.concat([singles, pd.DataFrame(combo_records)], ignore_index=True)
    candidates = candidates.sort_values(
        ["train_role_objective_lift_bps", "train_retained_count"],
        ascending=[False, False],
    ).head(config.max_candidates_per_state_horizon)
    candidates = candidates.reset_index(drop=True)
    candidates["filter_id"] = [
        f"{event_state}|h{horizon}|f{index + 1:03d}" for index in range(len(candidates))
    ]
    return candidates


def run_random_role_baseline(
    *,
    test_rows: pd.DataFrame,
    retained_rows: pd.DataFrame,
    horizon: int,
    expected_direction: int,
    role: str,
    seed: int,
    iterations: int,
    baseline: str = "random_same_count",
) -> pd.Series:
    """Random same-count baseline scored by role."""

    retained_count = int(len(retained_rows))
    if test_rows.empty or retained_count <= 0:
        return pd.Series(
            {
                "baseline": baseline,
                "retained_count": retained_count,
                "aligned_median_return_after": math.nan,
                "role_objective_after": math.nan,
                "blocker_net_value_bps": math.nan,
                "no_trade_quality_score_after": math.nan,
            }
        )
    rng = np.random.default_rng(seed)
    indices = np.array(test_rows.index.tolist())
    sample_size = min(retained_count, len(test_rows))
    values: dict[str, list[float]] = {
        "aligned_median_return_after": [],
        "role_objective_after": [],
        "blocker_net_value_bps": [],
        "short_median_return_after": [],
        "no_trade_quality_score_after": [],
    }
    for _ in range(max(1, iterations)):
        sample_indices = rng.choice(indices, size=sample_size, replace=False)
        mask = pd.Series(test_rows.index.isin(sample_indices), index=test_rows.index)
        metrics = evaluate_role_rows(
            test_rows,
            mask,
            event_state="random",
            horizon=horizon,
            expected_direction=expected_direction,
            role=role,
        )
        for key in values:
            values[key].append(float(metrics.get(key, math.nan)))
    return pd.Series(
        {
            "baseline": baseline,
            "retained_count": retained_count,
            **{key: _nanmedian_or_nan(items) for key, items in values.items()},
        }
    )


def _same_symbol_random_baseline(
    *,
    test_rows: pd.DataFrame,
    retained_rows: pd.DataFrame,
    horizon: int,
    expected_direction: int,
    role: str,
    seed: int,
    iterations: int,
) -> pd.Series:
    if test_rows.empty or retained_rows.empty or "symbol" not in test_rows:
        return run_random_role_baseline(
            test_rows=test_rows,
            retained_rows=retained_rows,
            horizon=horizon,
            expected_direction=expected_direction,
            role=role,
            seed=seed,
            iterations=iterations,
            baseline="same_symbol_random_same_count",
        )
    rng = np.random.default_rng(seed)
    retained_counts = retained_rows["symbol"].astype(str).value_counts().to_dict()
    sampled_metrics: list[dict[str, Any]] = []
    for _ in range(max(1, iterations)):
        sample_indices: list[Any] = []
        for symbol, count in retained_counts.items():
            pool = test_rows[test_rows["symbol"].astype(str).eq(symbol)]
            if pool.empty:
                continue
            chosen = rng.choice(
                np.array(pool.index.tolist()),
                size=min(int(count), len(pool)),
                replace=False,
            )
            sample_indices.extend(chosen.tolist())
        mask = pd.Series(test_rows.index.isin(sample_indices), index=test_rows.index)
        sampled_metrics.append(
            evaluate_role_rows(
                test_rows,
                mask,
                event_state="same_symbol_random",
                horizon=horizon,
                expected_direction=expected_direction,
                role=role,
            )
        )
    keys = (
        "aligned_median_return_after",
        "role_objective_after",
        "blocker_net_value_bps",
        "short_median_return_after",
        "no_trade_quality_score_after",
    )
    return pd.Series(
        {
            "baseline": "same_symbol_random_same_count",
            "retained_count": int(len(retained_rows)),
            **{
                key: _nanmedian_or_nan(
                    [float(row.get(key, math.nan)) for row in sampled_metrics]
                )
                for key in keys
            },
        }
    )


def _different_event_same_bucket_baseline(
    *,
    test_pool: pd.DataFrame,
    source_rows: pd.DataFrame,
    retained_rows: pd.DataFrame,
    horizon: int,
    expected_direction: int,
    role: str,
) -> pd.Series:
    if retained_rows.empty or test_pool.empty or "time_of_day_bucket" not in test_pool:
        return run_random_role_baseline(
            test_rows=source_rows,
            retained_rows=retained_rows,
            horizon=horizon,
            expected_direction=expected_direction,
            role=role,
            seed=11,
            iterations=1,
            baseline="different_event_same_time_bucket",
        )
    samples: list[pd.DataFrame] = []
    for bucket, bucket_rows in retained_rows.groupby("time_of_day_bucket"):
        pool = test_pool[
            test_pool["time_of_day_bucket"].astype(str).eq(str(bucket))
            & ~test_pool["event_state"].astype(str).isin(bucket_rows["event_state"].unique())
        ]
        if pool.empty:
            continue
        samples.append(
            pool.sample(
                n=min(len(bucket_rows), len(pool)),
                random_state=19,
            )
        )
    if not samples:
        return pd.Series(
            {
                "baseline": "different_event_same_time_bucket",
                "retained_count": int(len(retained_rows)),
                "aligned_median_return_after": math.nan,
                "role_objective_after": math.nan,
                "blocker_net_value_bps": math.nan,
                "short_median_return_after": math.nan,
                "no_trade_quality_score_after": math.nan,
            }
        )
    sample = pd.concat(samples, ignore_index=False)
    mask = pd.Series(test_pool.index.isin(sample.index), index=test_pool.index)
    metrics = evaluate_role_rows(
        test_pool,
        mask,
        event_state="different_event_same_time_bucket",
        horizon=horizon,
        expected_direction=expected_direction,
        role=role,
    )
    return pd.Series(
        {
            "baseline": "different_event_same_time_bucket",
            "retained_count": int(len(sample)),
            "aligned_median_return_after": metrics["aligned_median_return_after"],
            "role_objective_after": metrics["role_objective_after"],
            "blocker_net_value_bps": metrics.get("blocker_net_value_bps", math.nan),
            "short_median_return_after": metrics.get("short_median_return_after", math.nan),
            "no_trade_quality_score_after": metrics.get(
                "no_trade_quality_score_after",
                math.nan,
            ),
        }
    )


def _state_summary_rows(
    rows: pd.DataFrame,
    *,
    config: RoleAwareEventCutterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_records: list[dict[str, Any]] = []
    oos_records: list[dict[str, Any]] = []
    data = rows[rows["event_state"].astype(str).isin(EVENT_STATE_ROLES)].copy()
    for event_state, state_rows in data.groupby("event_state"):
        train_state, test_state = _split_train_test(state_rows, config.train_fraction)
        role = _role(str(event_state))
        for horizon in config.horizons:
            return_column = _return_col(horizon)
            if return_column not in state_rows:
                continue
            train_rows = train_state[
                pd.to_numeric(train_state[return_column], errors="coerce").notna()
            ]
            test_rows = test_state[
                pd.to_numeric(test_state[return_column], errors="coerce").notna()
            ]
            all_rows = state_rows[
                pd.to_numeric(state_rows[return_column], errors="coerce").notna()
            ]
            if all_rows.empty:
                continue
            direction = estimate_role_direction(
                event_state=str(event_state),
                train_rows=train_rows,
                horizon=horizon,
            )
            expected_direction = int(direction["expected_direction"])
            all_metrics = evaluate_role_rows(
                all_rows,
                pd.Series(True, index=all_rows.index),
                event_state=str(event_state),
                horizon=horizon,
                expected_direction=expected_direction,
                role=role,
                low_movement_threshold=config.low_movement_threshold,
            )
            test_metrics = evaluate_role_rows(
                test_rows,
                pd.Series(True, index=test_rows.index),
                event_state=str(event_state),
                horizon=horizon,
                expected_direction=expected_direction,
                role=role,
                low_movement_threshold=config.low_movement_threshold,
            )
            state_records.append({**direction, **all_metrics})
            oos_records.append(
                {
                    **direction,
                    "train_event_count": int(len(train_rows)),
                    "test_event_count": int(len(test_rows)),
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )
    return pd.DataFrame(state_records), pd.DataFrame(oos_records)


def _selected_decision_for_role(role: str) -> str:
    if role in {"long_candidate", "long_reversal_candidate"}:
        return "continue_research_long_candidate"
    if role == "long_blocker_or_short_candidate":
        return "continue_research_short_candidate"
    if role == "no_trade_filter":
        return "continue_research_no_trade_filter"
    return "reject_no_directional_consistency"


def _rejection_reason(
    *,
    train_count: int,
    test_count: int,
    retained_count: int,
    oos_lift: float,
    random_beaten: bool,
    same_symbol_random_beaten: bool,
    pre_concentration_gate: bool,
    concentration_warning: bool,
    config: RoleAwareEventCutterConfig,
) -> str:
    if pre_concentration_gate and concentration_warning:
        return "reject_concentrated"
    if train_count < config.min_train_events or test_count < config.min_test_events:
        return "reject_low_sample"
    if retained_count < config.min_retained_events:
        return "reject_low_sample"
    if math.isnan(oos_lift) or oos_lift < config.min_oos_lift_bps:
        return "reject_no_oos_lift"
    if not random_beaten or not same_symbol_random_beaten:
        return "reject_random_baseline_better"
    if concentration_warning:
        return "reject_concentrated"
    return "reject_no_directional_consistency"


def _filter_search(
    event_rows: pd.DataFrame,
    *,
    config: RoleAwareEventCutterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_rolling_symbol_state_efficacy(
        event_rows,
        horizons=config.horizons,
        windows=config.rolling_windows,
    )
    data = _add_derived_features(data)
    data = data[data["event_state"].astype(str).isin(EVENT_STATE_ROLES)].copy()
    train_all, test_all = _split_train_test(data, config.train_fraction)
    candidates: list[pd.DataFrame] = []
    oos_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    for event_state, state_rows in data.groupby("event_state"):
        role = _role(str(event_state))
        train_state = train_all[train_all["event_state"].astype(str).eq(str(event_state))]
        test_state = test_all[test_all["event_state"].astype(str).eq(str(event_state))]
        for horizon in config.horizons:
            return_column = _return_col(horizon)
            if return_column not in state_rows:
                continue
            train_rows = train_state[
                pd.to_numeric(train_state[return_column], errors="coerce").notna()
            ]
            test_rows = test_state[
                pd.to_numeric(test_state[return_column], errors="coerce").notna()
            ]
            if len(test_rows) < config.min_test_events:
                continue
            state_candidates = build_candidate_filters(
                train_rows,
                event_state=str(event_state),
                horizon=horizon,
                config=config,
            )
            if state_candidates.empty:
                continue
            candidates.append(state_candidates)
            for _, candidate in state_candidates.iterrows():
                expected_direction = int(candidate["expected_direction"])
                retained_mask = _apply_candidate_rule(test_rows, candidate)
                retained_rows = test_rows[retained_mask]
                metrics = evaluate_role_rows(
                    test_rows,
                    retained_mask,
                    event_state=str(event_state),
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                    low_movement_threshold=config.low_movement_threshold,
                )
                random = run_random_role_baseline(
                    test_rows=test_rows,
                    retained_rows=retained_rows,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                    seed=config.random_seed + len(oos_records),
                    iterations=config.random_iterations,
                )
                same_symbol = _same_symbol_random_baseline(
                    test_rows=test_rows,
                    retained_rows=retained_rows,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                    seed=config.random_seed + len(oos_records) + 10_000,
                    iterations=config.random_iterations,
                )
                different_event = _different_event_same_bucket_baseline(
                    test_pool=test_all[
                        pd.to_numeric(test_all[return_column], errors="coerce").notna()
                    ],
                    source_rows=test_rows,
                    retained_rows=retained_rows,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    role=role,
                )
                baselines = (random, same_symbol, different_event)
                for baseline in baselines:
                    baseline_records.append(
                        {
                            "filter_id": candidate["filter_id"],
                            "event_state": str(event_state),
                            "horizon": int(horizon),
                            "role": role,
                            **baseline.to_dict(),
                        }
                    )
                objective_after = float(metrics["role_objective_after"])
                objective_before = float(metrics["role_objective_before"])
                random_objective = float(random["role_objective_after"])
                same_symbol_objective = float(same_symbol["role_objective_after"])
                random_excess = (
                    (objective_after - random_objective) * 10_000
                    if not math.isnan(objective_after) and not math.isnan(random_objective)
                    else math.nan
                )
                same_symbol_excess = (
                    (objective_after - same_symbol_objective) * 10_000
                    if not math.isnan(objective_after)
                    and not math.isnan(same_symbol_objective)
                    else math.nan
                )
                oos_lift = (
                    (objective_after - objective_before) * 10_000
                    if not math.isnan(objective_after) and not math.isnan(objective_before)
                    else math.nan
                )
                concentration_warning = _has_concentration_warning(metrics, config)
                random_beaten = bool(
                    not math.isnan(random_excess)
                    and random_excess >= config.min_random_excess_bps
                )
                same_symbol_random_beaten = bool(
                    not math.isnan(same_symbol_excess)
                    and same_symbol_excess >= config.min_random_excess_bps
                )
                pre_concentration_gate = bool(
                    len(train_rows) >= config.min_train_events
                    and len(test_rows) >= config.min_test_events
                    and int(metrics["retained_count"]) >= config.min_retained_events
                    and not math.isnan(oos_lift)
                    and oos_lift >= config.min_oos_lift_bps
                    and random_beaten
                    and same_symbol_random_beaten
                )
                gate_passed = bool(pre_concentration_gate and not concentration_warning)
                selected_decision = _selected_decision_for_role(role)
                rejection_reason = _rejection_reason(
                    train_count=len(train_rows),
                    test_count=len(test_rows),
                    retained_count=int(metrics["retained_count"]),
                    oos_lift=oos_lift,
                    random_beaten=random_beaten,
                    same_symbol_random_beaten=same_symbol_random_beaten,
                    pre_concentration_gate=pre_concentration_gate,
                    concentration_warning=concentration_warning,
                    config=config,
                )
                oos_records.append(
                    {
                        "filter_id": candidate["filter_id"],
                        "event_state": str(event_state),
                        "horizon": int(horizon),
                        "role": role,
                        "filter_expression": candidate["filter_expression"],
                        "train_event_count": int(len(train_rows)),
                        "test_event_count": int(len(test_rows)),
                        "role_evidence_conflict": bool(candidate["role_evidence_conflict"]),
                        **metrics,
                        "oos_role_objective_lift_bps": oos_lift,
                        "random_same_count_role_objective_after": random_objective,
                        "random_role_excess_bps": random_excess,
                        "same_symbol_random_role_objective_after": same_symbol_objective,
                        "same_symbol_random_role_excess_bps": same_symbol_excess,
                        "random_beaten": random_beaten,
                        "same_symbol_random_beaten": same_symbol_random_beaten,
                        "pre_concentration_gate_passed": pre_concentration_gate,
                        "concentration_warning": concentration_warning,
                        "gate_passed": gate_passed,
                        "selected_decision": selected_decision
                        if gate_passed
                        else "reject_filter",
                        "rejection_reason": ""
                        if gate_passed
                        else rejection_reason,
                    }
                )
                concentration_rows.extend(
                    _concentration_warning_rows(
                        metrics=metrics,
                        event_state=str(event_state),
                        horizon=horizon,
                        filter_id=str(candidate["filter_id"]),
                        config=config,
                    )
                )
    candidate_frame = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return (
        candidate_frame,
        pd.DataFrame(oos_records),
        pd.DataFrame(baseline_records),
        pd.DataFrame(concentration_rows),
    )


def build_decision(
    selected_filters: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> dict[str, Any]:
    """Build the lab-level role-aware cutter decision."""

    if selected_filters.empty:
        return {
            "decision": "reject_low_sample",
            "decision_reasons": ["no candidate filters had enough sample"],
            "selected_filter_count": 0,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
        }
    gate_passed = selected_filters["gate_passed"].fillna(False).astype(bool)
    pre_concentration = selected_filters.get(
        "pre_concentration_gate_passed",
        pd.Series(False, index=selected_filters.index),
    ).fillna(False).astype(bool)
    passed = selected_filters[gate_passed]
    warning_filter_ids = (
        set(concentration_warnings["filter_id"].astype(str))
        if "filter_id" in concentration_warnings
        else set()
    )
    passed_concentrated = (
        not passed.empty
        and (
            passed["concentration_warning"].fillna(False).astype(bool).any()
            or passed["filter_id"].astype(str).isin(warning_filter_ids).any()
        )
    )
    pre_concentration_only = selected_filters[pre_concentration & ~gate_passed]
    pre_concentration_rejected = (
        not pre_concentration_only.empty
        and (
            pre_concentration_only["concentration_warning"].fillna(False).astype(bool).any()
            or pre_concentration_only["filter_id"].astype(str).isin(warning_filter_ids).any()
        )
    )
    if passed_concentrated:
        decision = "reject_concentrated"
        reasons = ["one or more otherwise passing role-aware filters were concentration dominated"]
    elif not passed.empty:
        decisions = set(passed["selected_decision"].astype(str))
        decision = (
            "continue_research_mixed_roles" if len(decisions) > 1 else next(iter(decisions))
        )
        reasons = ["at least one role-aware filter passed OOS, random, and concentration gates"]
    elif pre_concentration_rejected:
        decision = "reject_concentrated"
        reasons = ["one or more otherwise passing role-aware filters were concentration dominated"]
    elif not selected_filters.get(
        "random_beaten",
        pd.Series(False, index=selected_filters.index),
    ).fillna(False).astype(bool).any():
        decision = "reject_random_baseline_better"
        reasons = ["candidate filters did not beat random same-count baselines"]
    elif not (
        selected_filters["oos_role_objective_lift_bps"].fillna(-np.inf) > 0.0
    ).any():
        decision = "reject_no_oos_lift"
        reasons = ["candidate filters did not improve held-out role-aware objective"]
    else:
        decision = "reject_no_directional_consistency"
        reasons = ["candidate filters did not pass role-aware consistency gates"]
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "selected_filter_count": int(len(passed)),
        "selected_filters": passed.to_dict("records"),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }


def _slice_filter_outputs(filter_oos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if filter_oos.empty:
        empty = pd.DataFrame()
        return {
            "long": empty,
            "blocker": empty,
            "short": empty,
            "no_trade": empty,
            "selected": empty,
            "rejected": empty,
        }
    role = filter_oos["role"].astype(str)
    long = filter_oos[role.isin(["long_candidate", "long_reversal_candidate"])].copy()
    blocker = filter_oos[role.eq("long_blocker_or_short_candidate")].copy()
    short = blocker.copy()
    no_trade = filter_oos[role.eq("no_trade_filter")].copy()
    selected = filter_oos[filter_oos["gate_passed"].fillna(False).astype(bool)].copy()
    rejected = filter_oos[~filter_oos["gate_passed"].fillna(False).astype(bool)].copy()
    return {
        "long": long,
        "blocker": blocker,
        "short": short,
        "no_trade": no_trade,
        "selected": selected,
        "rejected": rejected,
    }


def _summary_markdown(
    *,
    input_dir: Path,
    summary: dict[str, Any],
    state_summary: pd.DataFrame,
    filter_oos: pd.DataFrame,
    selected: pd.DataFrame,
    rejected: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> str:
    decision = summary["decision"]["decision"]
    long_states = [
        state for state, info in EVENT_STATE_ROLES.items() if str(info["role"]).startswith("long")
    ]
    short_states = [
        state
        for state, info in EVENT_STATE_ROLES.items()
        if info["role"] == "long_blocker_or_short_candidate"
    ]
    no_trade_states = [
        state for state, info in EVENT_STATE_ROLES.items() if info["role"] == "no_trade_filter"
    ]
    negative_useful = state_summary[
        state_summary["expected_direction"].eq(-1)
        & (state_summary["raw_median_forward_return_after"] < 0.0)
        & (state_summary["aligned_median_return_after"] > 0.0)
    ] if not state_summary.empty else pd.DataFrame()
    beat_random = (
        filter_oos[filter_oos["random_beaten"].fillna(False).astype(bool)]
        if not filter_oos.empty
        else pd.DataFrame()
    )
    survived_oos = (
        filter_oos[filter_oos["oos_role_objective_lift_bps"].fillna(-np.inf) > 0.0]
        if not filter_oos.empty
        else pd.DataFrame()
    )
    continue_text = (
        "Continue research only on the selected non-concentrated role-aware filters."
        if str(decision).startswith("continue_research")
        else "Kill or revise this cutter path before using it as a second-stage filter."
    )
    state_columns = [
        "event_state",
        "horizon",
        "role",
        "expected_direction",
        "raw_median_forward_return_after",
        "aligned_median_return_after",
        "aligned_win_rate_after",
        "role_evidence_conflict",
    ]
    filter_columns = [
        "filter_id",
        "event_state",
        "horizon",
        "role",
        "filter_expression",
        "retained_count",
        "oos_role_objective_lift_bps",
        "random_role_excess_bps",
        "concentration_warning",
    ]
    warning_columns = ["event_state", "horizon", "filter_id", "warning", "value", "threshold"]
    return f"""# Role-Aware Event Cutter V0

Research-only diagnostic. No edge is claimed. Order placement is disabled.

input_report: {input_dir}
decision: {decision}
total_input_event_rows: {summary["total_input_event_rows"]}

## Role-Aware Interpretation

Long candidates: {", ".join(long_states)}

Short candidates / long-blockers: {", ".join(short_states)}

No-trade filters: {", ".join(no_trade_states)}

Negative-response states that are useful rather than failures:

{_markdown_table(negative_useful, state_columns)}

## Filters That Beat Random

{_markdown_table(beat_random, filter_columns)}

## Filters That Survived OOS

{_markdown_table(survived_oos, filter_columns)}

## Selected Filters

{_markdown_table(selected, filter_columns)}

## Rejected Filters

{_markdown_table(rejected, filter_columns)}

## Concentration Warnings

{_markdown_table(concentration_warnings, warning_columns)}

## Decision

{continue_text}
"""


def run_role_aware_event_cutter_lab(
    *,
    input_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_INPUT_BASE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: RoleAwareEventCutterConfig | None = None,
) -> RoleAwareEventCutterResult:
    """Run the role-aware event cutter from a state-event-detector report."""

    cfg = config or RoleAwareEventCutterConfig()
    resolved_input = input_dir or find_latest_state_event_detector_run(input_base_dir)
    _validate_input_dir(resolved_input)
    run_id = "role_aware_event_cutter_v0_" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    event_rows = pd.read_csv(resolved_input / "event_rows.csv")
    feature_rows = _add_derived_features(event_rows)
    state_summary, aligned_directional = _state_summary_rows(feature_rows, config=cfg)
    candidate_filters, filter_oos, baselines, concentration_warnings = _filter_search(
        feature_rows,
        config=cfg,
    )
    outputs = _slice_filter_outputs(filter_oos)
    decision = build_decision(filter_oos, concentration_warnings)

    paths = {
        "summary_md": run_dir / "summary.md",
        "summary_json": run_dir / "summary.json",
        "decision": run_dir / "decision.json",
        "role_aware_state_summary": run_dir / "role_aware_state_summary.csv",
        "aligned_directional_results": run_dir / "aligned_directional_results.csv",
        "long_candidate_filter_results": run_dir / "long_candidate_filter_results.csv",
        "blocker_quality_results": run_dir / "blocker_quality_results.csv",
        "short_candidate_results": run_dir / "short_candidate_results.csv",
        "no_trade_quality_results": run_dir / "no_trade_quality_results.csv",
        "role_evidence_conflicts": run_dir / "role_evidence_conflicts.csv",
        "filter_oos_results": run_dir / "filter_oos_results.csv",
        "random_role_baselines": run_dir / "random_role_baselines.csv",
        "concentration_warnings": run_dir / "concentration_warnings.csv",
        "selected_filters": run_dir / "selected_filters.csv",
        "rejected_filters": run_dir / "rejected_filters.csv",
    }
    role_conflicts = (
        state_summary[state_summary["role_evidence_conflict"].fillna(False).astype(bool)]
        if not state_summary.empty
        else pd.DataFrame()
    )
    summary = {
        "run_id": run_id,
        "input_dir": str(resolved_input),
        "output_dir": str(run_dir),
        "config": asdict(cfg),
        "total_input_event_rows": int(len(event_rows)),
        "candidate_filter_count": int(len(candidate_filters)),
        "oos_filter_result_count": int(len(filter_oos)),
        "selected_filter_count": int(len(outputs["selected"])),
        "decision": decision,
        "role_mapping": EVENT_STATE_ROLES,
        "files": {key: str(value) for key, value in paths.items()},
    }
    _write_csv(paths["role_aware_state_summary"], state_summary)
    _write_csv(paths["aligned_directional_results"], aligned_directional)
    _write_csv(paths["long_candidate_filter_results"], outputs["long"])
    _write_csv(paths["blocker_quality_results"], outputs["blocker"])
    _write_csv(paths["short_candidate_results"], outputs["short"])
    _write_csv(paths["no_trade_quality_results"], outputs["no_trade"])
    _write_csv(paths["role_evidence_conflicts"], role_conflicts)
    _write_csv(paths["filter_oos_results"], filter_oos)
    _write_csv(paths["random_role_baselines"], baselines)
    _write_csv(paths["concentration_warnings"], concentration_warnings)
    _write_csv(paths["selected_filters"], outputs["selected"])
    _write_csv(paths["rejected_filters"], outputs["rejected"])
    _write_json(paths["decision"], decision)
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(
        _summary_markdown(
            input_dir=resolved_input,
            summary=summary,
            state_summary=state_summary,
            filter_oos=filter_oos,
            selected=outputs["selected"],
            rejected=outputs["rejected"],
            concentration_warnings=concentration_warnings,
        ),
        encoding="utf-8",
    )
    return RoleAwareEventCutterResult(
        run_id=run_id,
        input_dir=resolved_input,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision"],
        role_aware_state_summary_csv_path=paths["role_aware_state_summary"],
        aligned_directional_results_csv_path=paths["aligned_directional_results"],
        long_candidate_filter_results_csv_path=paths["long_candidate_filter_results"],
        blocker_quality_results_csv_path=paths["blocker_quality_results"],
        short_candidate_results_csv_path=paths["short_candidate_results"],
        no_trade_quality_results_csv_path=paths["no_trade_quality_results"],
        role_evidence_conflicts_csv_path=paths["role_evidence_conflicts"],
        filter_oos_results_csv_path=paths["filter_oos_results"],
        random_role_baselines_csv_path=paths["random_role_baselines"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        selected_filters_csv_path=paths["selected_filters"],
        rejected_filters_csv_path=paths["rejected_filters"],
        decision=str(decision["decision"]),
        selected_filter_count=int(len(outputs["selected"])),
    )


__all__ = [
    "EVENT_STATE_ROLES",
    "RoleAwareEventCutterConfig",
    "RoleAwareEventCutterResult",
    "add_aligned_return_column",
    "add_rolling_symbol_state_efficacy",
    "build_candidate_filters",
    "build_decision",
    "estimate_role_direction",
    "evaluate_role_rows",
    "run_random_role_baseline",
    "run_role_aware_event_cutter_lab",
]
