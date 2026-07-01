"""Research-only second-stage cutter for state event failures.

This lab consumes sparse ``state_event_detector_v0`` event rows and asks whether
simple, interpretable filters can separate good forward responses from bad ones
using only features known at the event bar plus leakage-safe prior-session
symbol/state efficacy features.
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

DEFAULT_INPUT_BASE_DIR = Path("data/reports/research/state_event_detector_v0")
DEFAULT_OUTPUT_DIR = Path("data/reports/research/event_failure_cutter_v0")

REQUIRED_INPUT_FILES = (
    "event_rows.csv",
    "manual_state_audit.csv",
    "event_state_summary.csv",
    "same_event_cross_symbol_similarity.csv",
    "random_baseline.csv",
    "oos_event_response.csv",
    "concentration_warnings.csv",
    "summary.json",
    "summary.md",
    "decision.json",
)

LONG_ENTRY_STATES = {
    "controlled_pullback_after_bullish_impulse",
    "liquidation_failed_low_reclaim",
    "slow_snapback_after_dip",
}
BLOCKER_STATES = {
    "failed_bounce_active_liquidation",
    "failed_bullish_impulse_recoil",
    "failed_open_down_continuation",
}
DEAD_CHOP_STATE = "dead_chop_blocker"
FOCUS_EVENT_STATES = (
    "controlled_pullback_after_bullish_impulse",
    "dead_chop_blocker",
    "liquidation_failed_low_reclaim",
    "failed_bounce_active_liquidation",
    "failed_bullish_impulse_recoil",
    "failed_open_down_continuation",
    "slow_snapback_after_dip",
)

BASE_FILTER_FEATURES = (
    "bar_index_in_session",
    "distance_from_vwap_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_opening_range_high_pct",
    "distance_from_opening_range_low_pct",
    "distance_from_session_open_pct",
    "distance_from_session_high_pct",
    "distance_from_session_low_pct",
    "distance_from_recent_high_pct",
    "distance_from_recent_low_pct",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",
    "close_location_value",
    "bar_return",
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "pullback_depth_from_recent_high",
    "impulse_volume_ratio",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "rolling_intraday_range_pct",
    "opening_range_width",
    "range_zscore",
    "compression_zscore",
    "return_zscore",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "reclaim_from_recent_low",
    "impulse_return_12",
)


@dataclass(frozen=True)
class EventFailureCutterConfig:
    """Configuration for the focused bad-trade cutter lab."""

    horizons: tuple[int, ...] = (6, 9, 12, 24)
    train_fraction: float = 0.60
    random_seed: int = 1337
    random_iterations: int = 50
    min_train_events: int = 30
    min_test_events: int = 20
    min_retained_events: int = 10
    min_retained_pct: float = 0.05
    max_retained_pct: float = 0.95
    min_oos_objective_lift_bps: float = 0.0
    min_random_excess_bps: float = 0.0
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    top_single_filters_for_pairs: int = 5
    max_candidates_per_state_horizon: int = 24
    rolling_windows: tuple[int, ...] = (20, 60)


@dataclass(frozen=True)
class EventFailureCutterResult:
    """Paths and headline decision from an event-failure-cutter run."""

    run_id: str
    input_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    failure_attribution_summary_csv_path: Path
    candidate_bad_trade_filters_csv_path: Path
    filter_oos_results_csv_path: Path
    random_filter_baseline_csv_path: Path
    blocker_quality_summary_csv_path: Path
    state_failure_examples_csv_path: Path
    feature_distribution_good_vs_bad_csv_path: Path
    concentration_warnings_csv_path: Path
    decision: str
    best_filter_count: int


def _return_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_return"


def _mfe_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mfe"


def _mae_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mae"


def _objective_mode(event_state: str) -> str:
    if event_state in LONG_ENTRY_STATES:
        return "long"
    if event_state == DEAD_CHOP_STATE:
        return "dead_chop"
    return "blocker"


def _objective_values(rows: pd.DataFrame, *, horizon: int, objective_mode: str) -> pd.Series:
    returns = pd.to_numeric(rows[_return_col(horizon)], errors="coerce")
    if objective_mode == "long":
        return returns
    if objective_mode == "dead_chop":
        return -returns.abs()
    return -returns


def _good_response_mask(rows: pd.DataFrame, *, horizon: int, objective_mode: str) -> pd.Series:
    returns = pd.to_numeric(rows[_return_col(horizon)], errors="coerce")
    if objective_mode == "long":
        return returns > 0.0
    if objective_mode == "dead_chop":
        threshold = float(returns.abs().median()) if returns.notna().any() else math.nan
        return returns.abs() <= threshold
    return returns <= 0.0


def _safe_share(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def _single_symbol_share(rows: pd.DataFrame) -> float:
    if rows.empty or "symbol" not in rows:
        return math.nan
    return float(rows["symbol"].astype(str).value_counts(normalize=True).max())


def _single_session_share(rows: pd.DataFrame) -> float:
    if rows.empty or "session_date" not in rows:
        return math.nan
    return float(rows["session_date"].astype(str).value_counts(normalize=True).max())


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


def find_latest_state_event_detector_run(
    base_dir: Path = DEFAULT_INPUT_BASE_DIR,
) -> Path:
    """Return the newest complete state-event-detector run directory."""

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing state_event_detector_v0 base directory: {base_dir}")
    candidates: list[Path] = []
    for path in base_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("state_event_detector_v0_"):
            continue
        if all((path / name).exists() for name in REQUIRED_INPUT_FILES):
            candidates.append(path)
    if not candidates:
        expected = base_dir / "state_event_detector_v0_<run_id>"
        raise FileNotFoundError(
            f"No complete state_event_detector_v0 run found; expected {expected}"
        )
    return sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name))[-1]


def _validate_input_dir(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_INPUT_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Input directory is missing required state_event_detector_v0 files: {missing}"
        )


def add_rolling_symbol_state_efficacy(
    event_rows: pd.DataFrame,
    *,
    horizons: Sequence[int],
    windows: Sequence[int] = (20, 60),
) -> pd.DataFrame:
    """Add prior-session symbol/state efficacy features without current-session leakage."""

    data = event_rows.copy()
    if data.empty:
        return data
    data["session_date"] = data["session_date"].astype(str)
    for horizon in horizons:
        return_column = _return_col(horizon)
        if return_column not in data:
            continue
        session = (
            data.assign(_return=pd.to_numeric(data[return_column], errors="coerce"))
            .groupby(["symbol", "event_state", "session_date"], as_index=False)
            .agg(
                session_median_return=("_return", "median"),
                session_win_rate=("_return", lambda values: float((values > 0.0).mean())),
            )
        )
        session["_session_ts"] = pd.to_datetime(session["session_date"], errors="coerce")
        session = session.sort_values(["symbol", "event_state", "_session_ts", "session_date"])
        grouped = session.groupby(["symbol", "event_state"], group_keys=False)
        for window in windows:
            shifted_return = grouped["session_median_return"].shift(1)
            shifted_win_rate = grouped["session_win_rate"].shift(1)
            session[f"symbol_state_h{horizon}_prior_{window}_session_median_return"] = (
                shifted_return.groupby([session["symbol"], session["event_state"]])
                .rolling(window=window, min_periods=1)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )
            session[f"symbol_state_h{horizon}_prior_{window}_session_win_rate"] = (
                shifted_win_rate.groupby([session["symbol"], session["event_state"]])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        merge_columns = [
            "symbol",
            "event_state",
            "session_date",
            *[
                f"symbol_state_h{horizon}_prior_{window}_session_median_return"
                for window in windows
            ],
            *[
                f"symbol_state_h{horizon}_prior_{window}_session_win_rate"
                for window in windows
            ],
        ]
        data = data.merge(
            session[merge_columns],
            on=["symbol", "event_state", "session_date"],
            how="left",
        )
    return data


def _available_filter_features(rows: pd.DataFrame) -> list[str]:
    rolling = [column for column in rows.columns if column.startswith("symbol_state_h")]
    features = [column for column in BASE_FILTER_FEATURES if column in rows.columns]
    return features + [column for column in rolling if column not in features]


def _feature_thresholds(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.nunique() < 2:
        return []
    thresholds = [float(value) for value in numeric.quantile([0.25, 0.50, 0.75]).tolist()]
    if float(numeric.min()) <= 0.0 <= float(numeric.max()):
        thresholds.append(0.0)
    deduped: list[float] = []
    for value in sorted(set(round(item, 12) for item in thresholds)):
        if math.isfinite(value):
            deduped.append(float(value))
    return deduped


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


def _evaluate_rows(
    rows: pd.DataFrame,
    retained_mask: pd.Series,
    *,
    horizon: int,
    objective_mode: str,
) -> dict[str, float | int | bool]:
    if rows.empty:
        return {
            "event_count": 0,
            "retained_count": 0,
            "dropped_count": 0,
            "retained_pct": math.nan,
            "median_return_before": math.nan,
            "median_return_after": math.nan,
            "median_return_lift_bps": math.nan,
            "win_rate_before": math.nan,
            "win_rate_after": math.nan,
            "win_rate_lift": math.nan,
            "median_mfe_before": math.nan,
            "median_mfe_after": math.nan,
            "median_mae_before": math.nan,
            "median_mae_after": math.nan,
            "drawdown_proxy_before": math.nan,
            "drawdown_proxy_after": math.nan,
            "objective_before": math.nan,
            "objective_after": math.nan,
            "objective_lift_bps": math.nan,
            "single_symbol_share": math.nan,
            "single_session_share": math.nan,
            "concentration_warning": False,
        }
    mask = retained_mask.reindex(rows.index).fillna(False).astype(bool)
    retained = rows[mask]
    returns = pd.to_numeric(rows[_return_col(horizon)], errors="coerce")
    retained_returns = pd.to_numeric(retained[_return_col(horizon)], errors="coerce")
    objective = _objective_values(rows, horizon=horizon, objective_mode=objective_mode)
    retained_objective = _objective_values(
        retained,
        horizon=horizon,
        objective_mode=objective_mode,
    )
    mfe = pd.to_numeric(
        rows.get(_mfe_col(horizon), pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    )
    retained_mfe = pd.to_numeric(
        retained.get(_mfe_col(horizon), pd.Series(np.nan, index=retained.index)),
        errors="coerce",
    )
    mae = pd.to_numeric(
        rows.get(_mae_col(horizon), pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    )
    retained_mae = pd.to_numeric(
        retained.get(_mae_col(horizon), pd.Series(np.nan, index=retained.index)),
        errors="coerce",
    )
    before_return = float(returns.median()) if returns.notna().any() else math.nan
    after_return = float(retained_returns.median()) if retained_returns.notna().any() else math.nan
    before_objective = float(objective.median()) if objective.notna().any() else math.nan
    after_objective = (
        float(retained_objective.median()) if retained_objective.notna().any() else math.nan
    )
    symbol_share = _single_symbol_share(retained)
    session_share = _single_session_share(retained)
    return {
        "event_count": int(len(rows)),
        "retained_count": int(len(retained)),
        "dropped_count": int(len(rows) - len(retained)),
        "retained_pct": _safe_share(len(retained), len(rows)),
        "median_return_before": before_return,
        "median_return_after": after_return,
        "median_return_lift_bps": (after_return - before_return) * 10_000
        if not math.isnan(after_return) and not math.isnan(before_return)
        else math.nan,
        "win_rate_before": float((returns > 0.0).mean()) if returns.notna().any() else math.nan,
        "win_rate_after": float((retained_returns > 0.0).mean())
        if retained_returns.notna().any()
        else math.nan,
        "win_rate_lift": float((retained_returns > 0.0).mean() - (returns > 0.0).mean())
        if retained_returns.notna().any() and returns.notna().any()
        else math.nan,
        "median_mfe_before": float(mfe.median()) if mfe.notna().any() else math.nan,
        "median_mfe_after": float(retained_mfe.median())
        if retained_mfe.notna().any()
        else math.nan,
        "median_mae_before": float(mae.median()) if mae.notna().any() else math.nan,
        "median_mae_after": float(retained_mae.median())
        if retained_mae.notna().any()
        else math.nan,
        "drawdown_proxy_before": float(mae.quantile(0.10)) if mae.notna().any() else math.nan,
        "drawdown_proxy_after": float(retained_mae.quantile(0.10))
        if retained_mae.notna().any()
        else math.nan,
        "objective_before": before_objective,
        "objective_after": after_objective,
        "objective_lift_bps": (after_objective - before_objective) * 10_000
        if not math.isnan(after_objective) and not math.isnan(before_objective)
        else math.nan,
        "single_symbol_share": symbol_share,
        "single_session_share": session_share,
        "concentration_warning": bool(symbol_share > 0.50 or session_share > 0.20),
    }


def _candidate_record(
    *,
    event_state: str,
    horizon: int,
    objective_mode: str,
    rule_type: str,
    feature_1: str,
    operator_1: str,
    threshold_1: float,
    metrics: dict[str, Any],
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
        "objective_mode": objective_mode,
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
    config: EventFailureCutterConfig,
) -> pd.DataFrame:
    """Build train-selected single and two-feature candidate filters."""

    return_column = _return_col(horizon)
    if return_column not in train_rows:
        return pd.DataFrame()
    group = train_rows[train_rows["event_state"].astype(str).eq(event_state)].copy()
    group = group[pd.to_numeric(group[return_column], errors="coerce").notna()]
    if len(group) < config.min_train_events:
        return pd.DataFrame()
    objective_mode = _objective_mode(event_state)
    records: list[dict[str, Any]] = []
    for feature in _available_filter_features(group):
        for threshold in _feature_thresholds(group[feature]):
            for operator in ("<=", ">="):
                mask = _compare_feature(group[feature], operator, threshold).fillna(False)
                metrics = _evaluate_rows(
                    group,
                    mask,
                    horizon=horizon,
                    objective_mode=objective_mode,
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
                        objective_mode=objective_mode,
                        rule_type="single",
                        feature_1=feature,
                        operator_1=operator,
                        threshold_1=threshold,
                        metrics=metrics,
                    )
                )
    singles = pd.DataFrame(records)
    if singles.empty:
        return singles
    singles = singles.sort_values(
        ["train_objective_lift_bps", "train_retained_count"],
        ascending=[False, False],
    ).head(config.max_candidates_per_state_horizon)
    top = singles.head(config.top_single_filters_for_pairs)
    combo_records: list[dict[str, Any]] = []
    top_records = top.to_dict("records")
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
                metrics = _evaluate_rows(
                    group,
                    combo_mask,
                    horizon=horizon,
                    objective_mode=objective_mode,
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
                        objective_mode=objective_mode,
                        rule_type=f"two_feature_{logical_operator.lower()}",
                        feature_1=str(left["feature_1"]),
                        operator_1=str(left["operator_1"]),
                        threshold_1=float(left["threshold_1"]),
                        feature_2=str(right["feature_1"]),
                        operator_2=str(right["operator_1"]),
                        threshold_2=float(right["threshold_1"]),
                        logical_operator=logical_operator,
                        metrics=metrics,
                    )
                )
    candidates = pd.concat([singles, pd.DataFrame(combo_records)], ignore_index=True)
    candidates = candidates.sort_values(
        ["train_objective_lift_bps", "train_retained_count"],
        ascending=[False, False],
    ).head(config.max_candidates_per_state_horizon)
    candidates = candidates.reset_index(drop=True)
    candidates["filter_id"] = [
        f"{event_state}|h{horizon}|f{index + 1:03d}" for index in range(len(candidates))
    ]
    return candidates


def run_random_same_count_baseline(
    *,
    test_rows: pd.DataFrame,
    retained_rows: pd.DataFrame,
    horizon: int,
    objective_mode: str,
    seed: int,
    iterations: int,
    baseline: str = "random_same_count",
) -> pd.Series:
    """Random same-count retained set baseline for one candidate filter."""

    retained_count = int(len(retained_rows))
    if test_rows.empty or retained_count <= 0:
        return pd.Series(
            {
                "baseline": baseline,
                "retained_count": retained_count,
                "median_return_after": math.nan,
                "median_objective_after": math.nan,
                "win_rate_after": math.nan,
            }
        )
    rng = np.random.default_rng(seed)
    sample_size = min(retained_count, len(test_rows))
    returns: list[float] = []
    objectives: list[float] = []
    win_rates: list[float] = []
    indices = np.array(test_rows.index.tolist())
    for _ in range(max(1, iterations)):
        sample_indices = rng.choice(indices, size=sample_size, replace=False)
        sample = test_rows.loc[sample_indices]
        sample_returns = pd.to_numeric(sample[_return_col(horizon)], errors="coerce")
        sample_objective = _objective_values(sample, horizon=horizon, objective_mode=objective_mode)
        returns.append(float(sample_returns.median()))
        objectives.append(float(sample_objective.median()))
        win_rates.append(float((sample_returns > 0.0).mean()))
    return pd.Series(
        {
            "baseline": baseline,
            "retained_count": retained_count,
            "median_return_after": float(np.nanmedian(returns)),
            "median_objective_after": float(np.nanmedian(objectives)),
            "win_rate_after": float(np.nanmedian(win_rates)),
        }
    )


def _run_same_symbol_random_baseline(
    *,
    test_rows: pd.DataFrame,
    retained_rows: pd.DataFrame,
    horizon: int,
    objective_mode: str,
    seed: int,
    iterations: int,
) -> pd.Series:
    if test_rows.empty or retained_rows.empty or "symbol" not in test_rows:
        return run_random_same_count_baseline(
            test_rows=test_rows,
            retained_rows=retained_rows,
            horizon=horizon,
            objective_mode=objective_mode,
            seed=seed,
            iterations=iterations,
            baseline="same_symbol_random_same_count",
        )
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    objectives: list[float] = []
    win_rates: list[float] = []
    retained_counts = retained_rows["symbol"].astype(str).value_counts().to_dict()
    for _ in range(max(1, iterations)):
        sampled_parts: list[pd.DataFrame] = []
        for symbol, count in retained_counts.items():
            pool = test_rows[test_rows["symbol"].astype(str).eq(symbol)]
            if pool.empty:
                continue
            sample_size = min(int(count), len(pool))
            sample_indices = rng.choice(
                np.array(pool.index.tolist()),
                size=sample_size,
                replace=False,
            )
            sampled_parts.append(pool.loc[sample_indices])
        if not sampled_parts:
            continue
        sample = pd.concat(sampled_parts, ignore_index=False)
        sample_returns = pd.to_numeric(sample[_return_col(horizon)], errors="coerce")
        sample_objective = _objective_values(sample, horizon=horizon, objective_mode=objective_mode)
        returns.append(float(sample_returns.median()))
        objectives.append(float(sample_objective.median()))
        win_rates.append(float((sample_returns > 0.0).mean()))
    return pd.Series(
        {
            "baseline": "same_symbol_random_same_count",
            "retained_count": int(len(retained_rows)),
            "median_return_after": float(np.nanmedian(returns)) if returns else math.nan,
            "median_objective_after": float(np.nanmedian(objectives)) if objectives else math.nan,
            "win_rate_after": float(np.nanmedian(win_rates)) if win_rates else math.nan,
        }
    )


def calculate_blocker_quality(
    rows: pd.DataFrame,
    retained_mask: pd.Series,
    *,
    horizon: int,
) -> dict[str, float | int]:
    """Calculate blocker capture and false-block rates against generic long returns."""

    if rows.empty:
        return {
            "retained_count": 0,
            "bad_long_capture_rate": math.nan,
            "good_long_false_block_rate": math.nan,
            "net_improvement_vs_generic_long_exposure": math.nan,
        }
    mask = retained_mask.reindex(rows.index).fillna(False).astype(bool)
    returns = pd.to_numeric(rows[_return_col(horizon)], errors="coerce")
    bad_long = returns <= 0.0
    good_long = returns > 0.0
    retained_bad = int((mask & bad_long).sum())
    retained_good = int((mask & good_long).sum())
    retained_returns = returns[mask]
    before_median = float(returns.median()) if returns.notna().any() else math.nan
    after_median = float(retained_returns.median()) if retained_returns.notna().any() else math.nan
    return {
        "retained_count": int(mask.sum()),
        "bad_long_capture_rate": _safe_share(retained_bad, int(bad_long.sum())),
        "good_long_false_block_rate": _safe_share(retained_good, int(good_long.sum())),
        "net_improvement_vs_generic_long_exposure": before_median - after_median
        if not math.isnan(before_median) and not math.isnan(after_median)
        else math.nan,
    }


def _train_test_split(
    rows: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = rows.copy()
    data["_timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.sort_values(["_timestamp", "symbol", "event_state"], kind="mergesort")
    if len(data) < 2:
        return data, data.iloc[0:0]
    train_count = max(1, min(len(data) - 1, int(len(data) * train_fraction)))
    return data.iloc[:train_count].drop(columns=["_timestamp"]), data.iloc[train_count:].drop(
        columns=["_timestamp"]
    )


def _feature_distribution_good_vs_bad(
    rows: pd.DataFrame,
    *,
    config: EventFailureCutterConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    data = rows[rows["event_state"].astype(str).isin(FOCUS_EVENT_STATES)].copy()
    for event_state in sorted(data["event_state"].dropna().astype(str).unique()):
        state_rows = data[data["event_state"].astype(str).eq(event_state)]
        objective_mode = _objective_mode(event_state)
        for horizon in config.horizons:
            if _return_col(horizon) not in state_rows:
                continue
            valid_returns = pd.to_numeric(
                state_rows[_return_col(horizon)],
                errors="coerce",
            ).notna()
            group = state_rows[valid_returns]
            if group.empty:
                continue
            good = _good_response_mask(group, horizon=horizon, objective_mode=objective_mode)
            for feature in _available_filter_features(group):
                values = pd.to_numeric(group[feature], errors="coerce")
                good_values = values[good]
                bad_values = values[~good]
                if good_values.dropna().empty or bad_values.dropna().empty:
                    continue
                good_median = float(good_values.median())
                bad_median = float(bad_values.median())
                iqr = float(values.quantile(0.75) - values.quantile(0.25))
                records.append(
                    {
                        "event_state": event_state,
                        "horizon": horizon,
                        "feature": feature,
                        "good_count": int(good.sum()),
                        "bad_count": int((~good).sum()),
                        "good_median": good_median,
                        "bad_median": bad_median,
                        "median_difference_bad_minus_good": bad_median - good_median,
                        "abs_standardized_difference": abs(bad_median - good_median)
                        / iqr
                        if iqr
                        else math.nan,
                    }
                )
    return pd.DataFrame(records)


def _failure_attribution_summary(
    rows: pd.DataFrame,
    feature_distribution: pd.DataFrame,
    *,
    config: EventFailureCutterConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    data = rows[rows["event_state"].astype(str).isin(FOCUS_EVENT_STATES)].copy()
    for event_state in sorted(data["event_state"].dropna().astype(str).unique()):
        state_rows = data[data["event_state"].astype(str).eq(event_state)]
        objective_mode = _objective_mode(event_state)
        for horizon in config.horizons:
            return_column = _return_col(horizon)
            if return_column not in state_rows:
                continue
            group = state_rows[pd.to_numeric(state_rows[return_column], errors="coerce").notna()]
            if group.empty:
                continue
            good = _good_response_mask(group, horizon=horizon, objective_mode=objective_mode)
            fd_group = feature_distribution[
                feature_distribution["event_state"].astype(str).eq(event_state)
                & feature_distribution["horizon"].eq(horizon)
            ]
            if fd_group.empty:
                top_feature = ""
                top_separation = math.nan
            else:
                top_row = fd_group.sort_values(
                    "abs_standardized_difference",
                    ascending=False,
                ).iloc[0]
                top_feature = str(top_row["feature"])
                top_separation = float(top_row["abs_standardized_difference"])
            returns = pd.to_numeric(group[return_column], errors="coerce")
            records.append(
                {
                    "event_state": event_state,
                    "horizon": horizon,
                    "objective_mode": objective_mode,
                    "event_count": int(len(group)),
                    "good_count": int(good.sum()),
                    "bad_count": int((~good).sum()),
                    "bad_rate": float((~good).mean()),
                    "median_forward_return": float(returns.median()),
                    "win_rate": float((returns > 0.0).mean()),
                    "top_failure_separator_feature": top_feature,
                    "top_failure_separator_abs_std_diff": top_separation,
                }
            )
    return pd.DataFrame(records)


def _state_failure_examples(
    rows: pd.DataFrame,
    *,
    config: EventFailureCutterConfig,
    examples_per_group: int = 5,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    data = rows[rows["event_state"].astype(str).isin(FOCUS_EVENT_STATES)].copy()
    example_features = [
        "distance_from_vwap_pct",
        "distance_from_opening_range_mid_pct",
        "upper_wick_pct_of_range",
        "close_location_value",
        "bar_return",
        "prior_3_bar_return",
        "pullback_depth_from_recent_high",
        "impulse_volume_ratio",
    ]
    for event_state in sorted(data["event_state"].dropna().astype(str).unique()):
        state_rows = data[data["event_state"].astype(str).eq(event_state)]
        objective_mode = _objective_mode(event_state)
        for horizon in config.horizons:
            return_column = _return_col(horizon)
            if return_column not in state_rows:
                continue
            valid_returns = pd.to_numeric(
                state_rows[return_column],
                errors="coerce",
            ).notna()
            group = state_rows[valid_returns].copy()
            if group.empty:
                continue
            returns = pd.to_numeric(group[return_column], errors="coerce")
            if objective_mode == "long":
                selected = group.assign(_sort=returns).sort_values("_sort").head(examples_per_group)
            elif objective_mode == "dead_chop":
                selected = (
                    group.assign(_sort=returns.abs())
                    .sort_values("_sort", ascending=False)
                    .head(examples_per_group)
                )
            else:
                selected = group.assign(_sort=returns).sort_values("_sort", ascending=False).head(
                    examples_per_group
                )
            for _, row in selected.iterrows():
                record = {
                    "symbol": row.get("symbol", ""),
                    "timestamp": row.get("timestamp", ""),
                    "session_date": row.get("session_date", ""),
                    "event_state": event_state,
                    "horizon": horizon,
                    "forward_return": row.get(return_column, math.nan),
                    "forward_mfe": row.get(_mfe_col(horizon), math.nan),
                    "forward_mae": row.get(_mae_col(horizon), math.nan),
                    "failure_type": "bad_long"
                    if objective_mode == "long"
                    else "false_block_or_missed_move",
                }
                for feature in example_features:
                    if feature in row:
                        record[feature] = row[feature]
                records.append(record)
    return pd.DataFrame(records)


def _same_symbol_share_warning(
    rows: pd.DataFrame,
    *,
    event_state: str,
    horizon: int,
    filter_id: str,
    config: EventFailureCutterConfig,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    symbol_share = _single_symbol_share(rows)
    session_share = _single_session_share(rows)
    if not math.isnan(symbol_share) and symbol_share > config.max_single_symbol_share:
        warnings.append(
            {
                "event_state": event_state,
                "horizon": horizon,
                "filter_id": filter_id,
                "warning": "single_symbol_dominates",
                "value": symbol_share,
                "threshold": config.max_single_symbol_share,
            }
        )
    if not math.isnan(session_share) and session_share > config.max_single_session_share:
        warnings.append(
            {
                "event_state": event_state,
                "horizon": horizon,
                "filter_id": filter_id,
                "warning": "single_session_dominates",
                "value": session_share,
                "threshold": config.max_single_session_share,
            }
        )
    return warnings


def _run_filter_search(
    event_rows: pd.DataFrame,
    *,
    config: EventFailureCutterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_rolling_symbol_state_efficacy(
        event_rows,
        horizons=config.horizons,
        windows=config.rolling_windows,
    )
    data = data[data["event_state"].astype(str).isin(FOCUS_EVENT_STATES)].copy()
    candidate_frames: list[pd.DataFrame] = []
    oos_records: list[dict[str, Any]] = []
    random_records: list[dict[str, Any]] = []
    blocker_records: list[dict[str, Any]] = []
    concentration_records: list[dict[str, Any]] = []
    for event_state in sorted(data["event_state"].dropna().astype(str).unique()):
        state_rows = data[data["event_state"].astype(str).eq(event_state)]
        train_state, test_state = _train_test_split(state_rows, config.train_fraction)
        objective_mode = _objective_mode(event_state)
        for horizon in config.horizons:
            return_column = _return_col(horizon)
            if return_column not in state_rows:
                continue
            train_valid = pd.to_numeric(train_state[return_column], errors="coerce").notna()
            test_valid = pd.to_numeric(test_state[return_column], errors="coerce").notna()
            train_rows = train_state[train_valid]
            test_rows = test_state[test_valid]
            candidates = build_candidate_filters(
                train_rows,
                event_state=event_state,
                horizon=horizon,
                config=config,
            )
            if candidates.empty:
                continue
            candidate_frames.append(candidates)
            for _, candidate in candidates.iterrows():
                retained_mask = _apply_candidate_rule(test_rows, candidate)
                retained_rows = test_rows[retained_mask]
                test_metrics = _evaluate_rows(
                    test_rows,
                    retained_mask,
                    horizon=horizon,
                    objective_mode=objective_mode,
                )
                random_baseline = run_random_same_count_baseline(
                    test_rows=test_rows,
                    retained_rows=retained_rows,
                    horizon=horizon,
                    objective_mode=objective_mode,
                    seed=config.random_seed + len(oos_records),
                    iterations=config.random_iterations,
                )
                same_symbol_baseline = _run_same_symbol_random_baseline(
                    test_rows=test_rows,
                    retained_rows=retained_rows,
                    horizon=horizon,
                    objective_mode=objective_mode,
                    seed=config.random_seed + len(oos_records) + 10_000,
                    iterations=config.random_iterations,
                )
                for baseline_row in (random_baseline, same_symbol_baseline):
                    random_records.append(
                        {
                            "filter_id": candidate["filter_id"],
                            "event_state": event_state,
                            "horizon": horizon,
                            **baseline_row.to_dict(),
                        }
                    )
                random_objective = float(random_baseline["median_objective_after"])
                objective_after = float(test_metrics["objective_after"])
                objective_lift = float(test_metrics["objective_lift_bps"])
                random_excess_bps = (
                    (objective_after - random_objective) * 10_000
                    if not math.isnan(objective_after) and not math.isnan(random_objective)
                    else math.nan
                )
                concentration_warning = bool(
                    float(test_metrics["single_symbol_share"]) > config.max_single_symbol_share
                    if not math.isnan(float(test_metrics["single_symbol_share"]))
                    else False
                ) or bool(
                    float(test_metrics["single_session_share"]) > config.max_single_session_share
                    if not math.isnan(float(test_metrics["single_session_share"]))
                    else False
                )
                random_beaten = bool(
                    not math.isnan(random_excess_bps)
                    and random_excess_bps >= config.min_random_excess_bps
                )
                gate_passed = bool(
                    len(train_rows) >= config.min_train_events
                    and len(test_rows) >= config.min_test_events
                    and int(test_metrics["retained_count"]) >= config.min_retained_events
                    and objective_lift >= config.min_oos_objective_lift_bps
                    and random_beaten
                    and not concentration_warning
                )
                oos_records.append(
                    {
                        "filter_id": candidate["filter_id"],
                        "event_state": event_state,
                        "horizon": horizon,
                        "objective_mode": objective_mode,
                        "filter_expression": candidate["filter_expression"],
                        "train_event_count": int(len(train_rows)),
                        "test_event_count": int(len(test_rows)),
                        "retained_count": int(test_metrics["retained_count"]),
                        "dropped_count": int(test_metrics["dropped_count"]),
                        "retained_percentage": float(test_metrics["retained_pct"]),
                        "median_return_before_filter": float(test_metrics["median_return_before"]),
                        "median_return_after_filter": float(test_metrics["median_return_after"]),
                        "median_return_lift_bps": float(test_metrics["median_return_lift_bps"]),
                        "win_rate_before": float(test_metrics["win_rate_before"]),
                        "win_rate_after": float(test_metrics["win_rate_after"]),
                        "win_rate_lift": float(test_metrics["win_rate_lift"]),
                        "median_mfe_before": float(test_metrics["median_mfe_before"]),
                        "median_mfe_after": float(test_metrics["median_mfe_after"]),
                        "median_mae_before": float(test_metrics["median_mae_before"]),
                        "median_mae_after": float(test_metrics["median_mae_after"]),
                        "drawdown_proxy_before": float(test_metrics["drawdown_proxy_before"]),
                        "drawdown_proxy_after": float(test_metrics["drawdown_proxy_after"]),
                        "oos_objective_before": float(test_metrics["objective_before"]),
                        "oos_objective_after": objective_after,
                        "oos_objective_lift_bps": objective_lift,
                        "single_symbol_share": float(test_metrics["single_symbol_share"]),
                        "single_session_share": float(test_metrics["single_session_share"]),
                        "random_same_count_median_objective_after": random_objective,
                        "random_same_count_median_return_after": float(
                            random_baseline["median_return_after"]
                        ),
                        "random_excess_objective_bps": random_excess_bps,
                        "random_beaten": random_beaten,
                        "concentration_warning": concentration_warning,
                        "gate_passed": gate_passed,
                        "verdict": "continue_research" if gate_passed else "reject_filter",
                    }
                )
                concentration_records.extend(
                    _same_symbol_share_warning(
                        retained_rows,
                        event_state=event_state,
                        horizon=horizon,
                        filter_id=str(candidate["filter_id"]),
                        config=config,
                    )
                )
                if objective_mode in {"blocker", "dead_chop"}:
                    blocker_quality = calculate_blocker_quality(
                        test_rows,
                        retained_mask,
                        horizon=horizon,
                    )
                    random_retained = test_rows.sample(
                        n=min(len(retained_rows), len(test_rows)),
                        random_state=config.random_seed + len(blocker_records),
                    )
                    random_quality = calculate_blocker_quality(
                        test_rows,
                        pd.Series(
                            test_rows.index.isin(random_retained.index),
                            index=test_rows.index,
                        ),
                        horizon=horizon,
                    )
                    blocker_records.append(
                        {
                            "filter_id": candidate["filter_id"],
                            "event_state": event_state,
                            "horizon": horizon,
                            "filter_expression": candidate["filter_expression"],
                            **blocker_quality,
                            "random_bad_long_capture_rate": random_quality["bad_long_capture_rate"],
                            "random_good_long_false_block_rate": random_quality[
                                "good_long_false_block_rate"
                            ],
                            "random_block_net_improvement": random_quality[
                                "net_improvement_vs_generic_long_exposure"
                            ],
                        }
                    )
    candidate_filters = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    return (
        candidate_filters,
        pd.DataFrame(oos_records),
        pd.DataFrame(random_records),
        pd.concat([pd.DataFrame(blocker_records), pd.DataFrame()], ignore_index=True),
        pd.DataFrame(concentration_records),
    )


def build_decision(
    filter_oos_results: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> dict[str, Any]:
    """Build the lab-level decision from OOS filter evidence."""

    if filter_oos_results.empty:
        decision = "reject_no_failure_discriminator"
        reasons = ["no candidate filter had enough train/test sample"]
    else:
        oos_positive = filter_oos_results["oos_objective_lift_bps"].fillna(-np.inf) > 0.0
        random_beaten = filter_oos_results["random_beaten"].fillna(False).astype(bool)
        gate_passed = filter_oos_results["gate_passed"].fillna(False).astype(bool)
        concentrated = (
            bool(concentration_warnings.shape[0])
            or filter_oos_results["concentration_warning"].fillna(False).astype(bool).any()
        )
        if oos_positive.any() and not random_beaten.any():
            decision = "reject_random_filter_beats_state_filter"
            reasons = ["candidate filters with OOS lift did not beat random same-count filters"]
        elif not oos_positive.any():
            decision = "reject_no_oos_lift"
            reasons = ["no candidate filter improved OOS objective versus no filter"]
        elif concentrated:
            decision = "reject_concentrated"
            reasons = ["one or more retained filter groups are concentration dominated"]
        elif not gate_passed.any():
            decision = "reject_no_failure_discriminator"
            reasons = ["no candidate filter passed sample, OOS, random, and concentration gates"]
        else:
            decision = "continue_research"
            reasons = ["at least one simple filter improved OOS objective and beat random baseline"]
    passed = (
        filter_oos_results[filter_oos_results["gate_passed"].fillna(False).astype(bool)]
        if not filter_oos_results.empty
        else pd.DataFrame()
    )
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "best_filter_count": int(len(passed)),
        "best_filters": passed[
            [
                "filter_id",
                "event_state",
                "horizon",
                "filter_expression",
                "oos_objective_lift_bps",
            ]
        ].to_dict("records")
        if not passed.empty
        else [],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }


def _summary_markdown(
    *,
    summary: dict[str, Any],
    failure_attribution: pd.DataFrame,
    feature_distribution: pd.DataFrame,
    filter_oos_results: pd.DataFrame,
    blocker_quality: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> str:
    decision = summary["decision"]["decision"]
    passed = filter_oos_results[
        filter_oos_results["gate_passed"].fillna(False).astype(bool)
    ] if not filter_oos_results.empty else pd.DataFrame()
    failed = filter_oos_results[
        ~filter_oos_results["gate_passed"].fillna(False).astype(bool)
    ] if not filter_oos_results.empty else pd.DataFrame()
    failed_states = failure_attribution[
        ~failure_attribution["event_state"].isin(passed["event_state"].unique())
    ] if not failure_attribution.empty and not passed.empty else failure_attribution
    top_features = feature_distribution.sort_values(
        "abs_standardized_difference",
        ascending=False,
    ).head(12) if not feature_distribution.empty else pd.DataFrame()
    continue_text = (
        "Continue research only on the passing filter/event/horizon pairs."
        if decision == "continue_research"
        else "Kill or revise this cutter path before using it as a second-stage filter."
    )
    failed_state_columns = [
        "event_state",
        "horizon",
        "event_count",
        "bad_rate",
        "top_failure_separator_feature",
    ]
    feature_columns = [
        "event_state",
        "horizon",
        "feature",
        "good_median",
        "bad_median",
        "abs_standardized_difference",
    ]
    passed_columns = [
        "filter_id",
        "event_state",
        "horizon",
        "filter_expression",
        "retained_count",
        "oos_objective_lift_bps",
        "random_excess_objective_bps",
    ]
    failed_columns = [
        "filter_id",
        "event_state",
        "horizon",
        "filter_expression",
        "oos_objective_lift_bps",
        "random_excess_objective_bps",
        "verdict",
    ]
    blocker_columns = [
        "filter_id",
        "event_state",
        "horizon",
        "bad_long_capture_rate",
        "good_long_false_block_rate",
        "net_improvement_vs_generic_long_exposure",
    ]
    concentration_columns = [
        "event_state",
        "horizon",
        "filter_id",
        "warning",
        "value",
        "threshold",
    ]
    return f"""# Event Failure Cutter V0

Research-only diagnostic. No edge is claimed. No order placement is enabled.

input_report: {summary["input_dir"]}
decision: {decision}
total_input_event_rows: {summary["total_input_event_rows"]}

## Which States Failed?

{_markdown_table(failed_states, failed_state_columns)}

## What Separates Good From Bad?

{_markdown_table(top_features, feature_columns)}

## Filters That Cut Bad Trades

{_markdown_table(passed, passed_columns)}

## Filters That Failed

{_markdown_table(failed, failed_columns)}

## Blocker Quality

{_markdown_table(blocker_quality, blocker_columns)}

## Random Same-Count Test

Did any filter beat random same-count filtering? {bool(summary["decision"]["best_filter_count"])}

## OOS Survival

Did any filter survive OOS? {bool(summary["decision"]["best_filter_count"])}

## Continue Or Kill

{continue_text}

## Concentration Warnings

{_markdown_table(concentration_warnings, concentration_columns)}
"""


def run_event_failure_cutter_lab(
    *,
    input_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_INPUT_BASE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: EventFailureCutterConfig | None = None,
) -> EventFailureCutterResult:
    """Run the event-failure-cutter lab from state-event-detector output."""

    cfg = config or EventFailureCutterConfig()
    resolved_input_dir = input_dir or find_latest_state_event_detector_run(input_base_dir)
    _validate_input_dir(resolved_input_dir)
    run_id = "event_failure_cutter_v0_" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event_rows = pd.read_csv(resolved_input_dir / "event_rows.csv")
    feature_rows = add_rolling_symbol_state_efficacy(
        event_rows,
        horizons=cfg.horizons,
        windows=cfg.rolling_windows,
    )
    feature_distribution = _feature_distribution_good_vs_bad(feature_rows, config=cfg)
    failure_attribution = _failure_attribution_summary(
        feature_rows,
        feature_distribution,
        config=cfg,
    )
    failure_examples = _state_failure_examples(feature_rows, config=cfg)
    (
        candidate_filters,
        filter_oos_results,
        random_baseline,
        blocker_quality,
        concentration_warnings,
    ) = _run_filter_search(event_rows, config=cfg)
    decision = build_decision(filter_oos_results, concentration_warnings)

    paths = {
        "summary_md": run_dir / "summary.md",
        "summary_json": run_dir / "summary.json",
        "decision": run_dir / "decision.json",
        "failure_attribution_summary": run_dir / "failure_attribution_summary.csv",
        "candidate_bad_trade_filters": run_dir / "candidate_bad_trade_filters.csv",
        "filter_oos_results": run_dir / "filter_oos_results.csv",
        "random_filter_baseline": run_dir / "random_filter_baseline.csv",
        "blocker_quality_summary": run_dir / "blocker_quality_summary.csv",
        "state_failure_examples": run_dir / "state_failure_examples.csv",
        "feature_distribution_good_vs_bad": run_dir / "feature_distribution_good_vs_bad.csv",
        "concentration_warnings": run_dir / "concentration_warnings.csv",
    }
    summary = {
        "run_id": run_id,
        "input_dir": str(resolved_input_dir),
        "output_dir": str(run_dir),
        "config": asdict(cfg),
        "total_input_event_rows": int(len(event_rows)),
        "candidate_filter_count": int(len(candidate_filters)),
        "oos_filter_result_count": int(len(filter_oos_results)),
        "decision": decision,
        "files": {key: str(value) for key, value in paths.items()},
    }
    _write_csv(paths["failure_attribution_summary"], failure_attribution)
    _write_csv(paths["candidate_bad_trade_filters"], candidate_filters)
    _write_csv(paths["filter_oos_results"], filter_oos_results)
    _write_csv(paths["random_filter_baseline"], random_baseline)
    _write_csv(paths["blocker_quality_summary"], blocker_quality)
    _write_csv(paths["state_failure_examples"], failure_examples)
    _write_csv(paths["feature_distribution_good_vs_bad"], feature_distribution)
    _write_csv(paths["concentration_warnings"], concentration_warnings)
    _write_json(paths["summary_json"], summary)
    _write_json(paths["decision"], decision)
    paths["summary_md"].write_text(
        _summary_markdown(
            summary=summary,
            failure_attribution=failure_attribution,
            feature_distribution=feature_distribution,
            filter_oos_results=filter_oos_results,
            blocker_quality=blocker_quality,
            concentration_warnings=concentration_warnings,
        ),
        encoding="utf-8",
    )
    return EventFailureCutterResult(
        run_id=run_id,
        input_dir=resolved_input_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision"],
        failure_attribution_summary_csv_path=paths["failure_attribution_summary"],
        candidate_bad_trade_filters_csv_path=paths["candidate_bad_trade_filters"],
        filter_oos_results_csv_path=paths["filter_oos_results"],
        random_filter_baseline_csv_path=paths["random_filter_baseline"],
        blocker_quality_summary_csv_path=paths["blocker_quality_summary"],
        state_failure_examples_csv_path=paths["state_failure_examples"],
        feature_distribution_good_vs_bad_csv_path=paths["feature_distribution_good_vs_bad"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        decision=str(decision["decision"]),
        best_filter_count=int(decision["best_filter_count"]),
    )


__all__ = [
    "EventFailureCutterConfig",
    "EventFailureCutterResult",
    "add_rolling_symbol_state_efficacy",
    "build_candidate_filters",
    "build_decision",
    "calculate_blocker_quality",
    "find_latest_state_event_detector_run",
    "run_event_failure_cutter_lab",
    "run_random_same_count_baseline",
]
