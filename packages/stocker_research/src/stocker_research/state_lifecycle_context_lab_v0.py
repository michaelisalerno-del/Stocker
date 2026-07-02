"""Research-only state lifecycle context lab.

This lab tests whether selected personality trades behave differently after
specific prior-regime mixes and prior sparse event clusters. It consumes local
research outputs and local OHLCV parquet only. It does not fetch vendor data,
touch broker execution, paper trading, live trading, or order placement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_data.storage import DatasetKey, dataset_path, read_parquet
from stocker_research.behavioral_state_similarity import (
    BehavioralStateConfig,
    build_behavioral_state_frame,
)
from stocker_research.personality_discovery_v0 import EVENT_STATE_PERSONALITY
from stocker_research.personality_live_replay_v0 import _add_missing_discovery_features

DEFAULT_OUTPUT_DIR = Path("data/reports/research/state_lifecycle_context_lab_v0")

REGIME_COLUMNS: tuple[str, ...] = (
    "vwap_side_regime",
    "opening_mid_side_regime",
    "session_open_side_regime",
    "range_regime",
    "compression_regime",
    "efficiency_regime",
    "relative_volume_regime",
    "time_regime",
    "vwap_x_efficiency_regime",
    "vwap_x_range_regime",
    "compression_x_efficiency_regime",
    "opening_mid_x_range_regime",
    "time_x_vwap_regime",
    "volume_x_vwap_regime",
)

NUMERIC_PRIOR_COLUMNS: tuple[str, ...] = (
    "distance_from_vwap_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_session_open_pct",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "range_zscore",
    "return_zscore",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "bar_return",
    "bar_range_pct",
    "close_location_value",
)
NUMERIC_THRESHOLD_EPSILON = 1e-12

UP_ATTEMPT_STATES = {
    "controlled_pullback_after_bullish_impulse",
    "liquidation_failed_low_reclaim",
    "slow_snapback_after_dip",
}
DOWN_PRESSURE_STATES = {
    "failed_bounce_active_liquidation",
    "failed_open_down_continuation",
    "failed_bullish_impulse_recoil",
}
CHOP_STATES = {"dead_chop_blocker"}


@dataclass(frozen=True)
class StateLifecycleContextConfig:
    """Configuration for the state lifecycle context lab."""

    lookback_bars: tuple[int, ...] = (6, 12, 24, 36)
    source: str = "eodhd"
    instrument_type: str = "stock"
    timeframe: str = "5m"
    market_calendar: str | None = "XNYS"
    min_train_count: int = 8
    min_oos_count: int = 5
    min_train_mean_lift_r: float = 0.0
    min_oos_mean_lift_r: float = 0.0
    max_selected_per_family: int = 40
    dense_placeholder_event_state: str = "dead_chop_blocker"


@dataclass(frozen=True)
class StateLifecycleContextResult:
    """Paths and headline result for one lifecycle context run."""

    run_id: str
    input_expression_report_dir: Path
    input_event_dir: Path
    data_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    trade_context_features_csv_path: Path
    base_summary_csv_path: Path
    prior_regime_numeric_scan_csv_path: Path
    prior_regime_categorical_scan_csv_path: Path
    prior_event_cluster_scan_csv_path: Path
    selected_context_candidates_csv_path: Path
    decision: str
    selected_candidate_count: int


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


def _trade_stats(rows: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    net = pd.to_numeric(rows.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        f"{prefix}count": int(len(rows)),
        f"{prefix}net_r": float(net.sum()) if len(net) else 0.0,
        f"{prefix}mean_r": float(net.mean()) if len(net) else math.nan,
        f"{prefix}win_rate": float((net > 0.0).mean()) if len(net) else math.nan,
        f"{prefix}symbols": int(rows["symbol"].nunique()) if len(rows) and "symbol" in rows else 0,
        f"{prefix}months": int(rows["month"].nunique()) if len(rows) and "month" in rows else 0,
    }


def _load_trades(input_expression_report_dir: Path) -> pd.DataFrame:
    train_path = input_expression_report_dir / "train_trades.csv"
    test_path = input_expression_report_dir / "test_trades.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expression trade files: {missing}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["split"] = "train"
    test["split"] = "test"
    trades = pd.concat([train, test], ignore_index=True)
    required = {
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "personality",
        "net_r",
    }
    missing_columns = sorted(required - set(trades.columns))
    if missing_columns:
        raise ValueError(f"Expression trades missing required columns: {missing_columns}")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.strftime("%Y-%m-%d")
    trades["bar_index_in_session"] = pd.to_numeric(
        trades["bar_index_in_session"], errors="coerce"
    )
    if "month" not in trades:
        trades["month"] = trades["timestamp"].dt.strftime("%Y-%m")
    else:
        trades["month"] = trades["month"].astype(str)
    return trades.reset_index(drop=True)


def _load_event_rows(input_event_dir: Path) -> pd.DataFrame:
    path = input_event_dir / "event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event rows: {path}")
    events = pd.read_csv(path)
    if events.empty:
        return events
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events["session_date"] = pd.to_datetime(events["session_date"]).dt.strftime("%Y-%m-%d")
    events["bar_index_in_session"] = pd.to_numeric(
        events["bar_index_in_session"], errors="coerce"
    )
    mapped = events["event_state"].astype(str).map(EVENT_STATE_PERSONALITY)
    events["event_personality"] = mapped.map(
        lambda value: value[0] if isinstance(value, tuple) else "unknown"
    )
    events["_mapped_direction"] = mapped.map(
        lambda value: value[2] if isinstance(value, tuple) else 0
    )
    return events.sort_values(
        ["symbol", "session_date", "bar_index_in_session", "timestamp", "event_state"],
        kind="mergesort",
    ).reset_index(drop=True)


def _dataset_for_symbol(
    *,
    data_dir: Path,
    symbol: str,
    config: StateLifecycleContextConfig,
) -> Path:
    path = cast(
        Path,
        dataset_path(
            DatasetKey(
                source=config.source,
                instrument_type=config.instrument_type,
                symbol=symbol.upper(),
                timeframe=config.timeframe,
            ),
            data_dir=data_dir,
        ),
    )
    if path.exists():
        return path
    fallback_symbol = (
        symbol.upper().removesuffix(".US")
        if symbol.upper().endswith(".US")
        else f"{symbol.upper()}.US"
    )
    fallback = cast(
        Path,
        dataset_path(
            DatasetKey(
                source=config.source,
                instrument_type=config.instrument_type,
                symbol=fallback_symbol,
                timeframe=config.timeframe,
            ),
            data_dir=data_dir,
        ),
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing local parquet for {symbol}: {path}")


def _load_dense_symbol_frame(
    *,
    data_dir: Path,
    symbol: str,
    config: StateLifecycleContextConfig,
) -> pd.DataFrame:
    raw = read_parquet(_dataset_for_symbol(data_dir=data_dir, symbol=symbol, config=config))
    required_existing = {"timestamp", "session_date", "bar_index_in_session", *REGIME_COLUMNS}
    if required_existing.issubset(raw.columns):
        dense = raw.copy()
    else:
        features = build_behavioral_state_frame(
            raw,
            symbol=symbol,
            config=BehavioralStateConfig(
                timeframe=config.timeframe,
                market_calendar=config.market_calendar,
            ),
        )
        features["event_state"] = config.dense_placeholder_event_state
        dense = _add_missing_discovery_features(features)
    dense["symbol"] = symbol.upper()
    dense["timestamp"] = pd.to_datetime(dense["timestamp"], utc=True, errors="coerce")
    dense["session_date"] = pd.to_datetime(dense["session_date"]).dt.strftime("%Y-%m-%d")
    dense["bar_index_in_session"] = pd.to_numeric(
        dense["bar_index_in_session"], errors="coerce"
    )
    return cast(
        pd.DataFrame,
        dense.sort_values(["session_date", "bar_index_in_session"], kind="mergesort"),
    )


def _add_prior_regime_features(
    trades: pd.DataFrame,
    *,
    data_dir: Path,
    config: StateLifecycleContextConfig,
) -> pd.DataFrame:
    symbols = sorted(trades["symbol"].astype(str).str.upper().unique())
    dense_groups: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in symbols:
        dense = _load_dense_symbol_frame(data_dir=data_dir, symbol=symbol, config=config)
        for session_date, group in dense.groupby("session_date", sort=False):
            dense_groups[(symbol, str(session_date))] = group

    rows: list[dict[str, Any]] = []
    for trade_index, trade in trades.iterrows():
        symbol = str(trade["symbol"]).upper()
        session_date = str(trade["session_date"])
        bar_index = int(trade["bar_index_in_session"])
        group = dense_groups.get((symbol, session_date), pd.DataFrame())
        feature_row: dict[str, Any] = {"_trade_index": int(cast(int, trade_index))}
        for window in config.lookback_bars:
            if group.empty:
                prior = group
            else:
                indices = pd.to_numeric(group["bar_index_in_session"], errors="coerce")
                prior = group[(indices < bar_index) & (indices >= bar_index - window)].copy()
            feature_row[f"prev_regime_bar_count_{window}"] = int(len(prior))
            for column in REGIME_COLUMNS:
                if column not in prior:
                    continue
                current_value = str(trade.get(column, ""))
                values = prior[column].astype(str) if not prior.empty else pd.Series(dtype=str)
                prefix = f"prev_{window}_{column}"
                if values.empty:
                    feature_row[f"{prefix}_dominant"] = "none"
                    feature_row[f"{prefix}_dominant_share"] = math.nan
                    feature_row[f"{prefix}_current_share"] = math.nan
                    feature_row[f"{prefix}_last"] = "none"
                    feature_row[f"{prefix}_last_matches_current"] = 0
                    feature_row[f"{prefix}_unique_count"] = 0
                    continue
                counts = values.value_counts(normalize=True)
                feature_row[f"{prefix}_dominant"] = str(counts.index[0])
                feature_row[f"{prefix}_dominant_share"] = float(counts.iloc[0])
                feature_row[f"{prefix}_current_share"] = float(values.eq(current_value).mean())
                feature_row[f"{prefix}_last"] = str(values.iloc[-1])
                feature_row[f"{prefix}_last_matches_current"] = int(
                    str(values.iloc[-1]) == current_value
                )
                feature_row[f"{prefix}_unique_count"] = int(values.nunique())
            for column in NUMERIC_PRIOR_COLUMNS:
                if column not in prior:
                    continue
                values = pd.to_numeric(prior[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                value_array = values.to_numpy(dtype=float)
                finite_values = value_array[np.isfinite(value_array)]
                prefix = f"prev_{window}_{column}"
                feature_row[f"{prefix}_mean"] = (
                    float(finite_values.mean()) if len(finite_values) else math.nan
                )
                feature_row[f"{prefix}_last"] = (
                    float(value_array[-1]) if len(value_array) else math.nan
                )
                feature_row[f"{prefix}_std"] = (
                    float(finite_values.std(ddof=1)) if len(finite_values) > 1 else math.nan
                )
        rows.append(feature_row)
    features = pd.DataFrame(rows).set_index("_trade_index")
    return trades.join(features, how="left")


def _direction_label_from_value(value: Any) -> str:
    if isinstance(value, str) and value.lower() in {"up", "down"}:
        return value.lower()
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "flat"
    if float(numeric) > 0:
        return "up"
    if float(numeric) < 0:
        return "down"
    return "flat"


def _add_prior_event_cluster_features(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: StateLifecycleContextConfig,
) -> pd.DataFrame:
    if events.empty:
        return trades
    event_groups = {
        (str(symbol).upper(), str(session_date)): group
        for (symbol, session_date), group in events.groupby(
            ["symbol", "session_date"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for trade_index, trade in trades.iterrows():
        symbol = str(trade["symbol"]).upper()
        session_date = str(trade["session_date"])
        bar_index = int(trade["bar_index_in_session"])
        current_state = str(trade.get("event_state", ""))
        current_personality = str(trade["personality"])
        current_direction = _direction_label_from_value(trade.get("expected_direction", 0))
        group = event_groups.get((symbol, session_date), pd.DataFrame())
        feature_row: dict[str, Any] = {"_trade_index": int(cast(int, trade_index))}
        for window in config.lookback_bars:
            if group.empty:
                prior = group
            else:
                indices = pd.to_numeric(group["bar_index_in_session"], errors="coerce")
                prior = group[(indices < bar_index) & (indices >= bar_index - window)].copy()
            states = prior["event_state"].astype(str) if not prior.empty else pd.Series(dtype=str)
            personalities = (
                prior["event_personality"].astype(str) if not prior.empty else pd.Series(dtype=str)
            )
            if "event_direction" in prior:
                directions = prior["event_direction"].map(_direction_label_from_value)
            else:
                directions = prior["_mapped_direction"].map(_direction_label_from_value)
            feature_row[f"cluster_any_count_{window}"] = int(len(prior))
            feature_row[f"cluster_unique_state_count_{window}"] = (
                int(states.nunique()) if len(prior) else 0
            )
            feature_row[f"cluster_same_state_count_{window}"] = int(
                states.eq(current_state).sum()
            )
            feature_row[f"cluster_same_personality_count_{window}"] = int(
                personalities.eq(current_personality).sum()
            )
            feature_row[f"cluster_same_direction_count_{window}"] = int(
                directions.eq(current_direction).sum()
            )
            feature_row[f"cluster_opposite_direction_count_{window}"] = int(
                (directions.isin(["up", "down"]) & ~directions.eq(current_direction)).sum()
            )
            feature_row[f"cluster_up_attempt_count_{window}"] = int(
                states.isin(UP_ATTEMPT_STATES).sum()
            )
            feature_row[f"cluster_down_pressure_count_{window}"] = int(
                states.isin(DOWN_PRESSURE_STATES).sum()
            )
            feature_row[f"cluster_chop_count_{window}"] = int(states.isin(CHOP_STATES).sum())
            if current_direction == "down":
                failed_attempt = states.isin(UP_ATTEMPT_STATES | {"failed_bullish_impulse_recoil"})
            elif current_direction == "up":
                failed_attempt = states.isin(
                    DOWN_PRESSURE_STATES | {"liquidation_failed_low_reclaim"}
                )
            else:
                failed_attempt = pd.Series(False, index=states.index)
            feature_row[f"cluster_failed_attempt_count_{window}"] = int(failed_attempt.sum())
            if prior.empty:
                feature_row[f"cluster_span_bars_{window}"] = math.nan
                feature_row[f"cluster_last_gap_bars_{window}"] = math.nan
                feature_row[f"cluster_mean_confidence_{window}"] = math.nan
            else:
                bars = pd.to_numeric(prior["bar_index_in_session"], errors="coerce")
                feature_row[f"cluster_span_bars_{window}"] = float(bar_index - bars.min())
                feature_row[f"cluster_last_gap_bars_{window}"] = float(bar_index - bars.max())
                confidence = pd.to_numeric(
                    prior.get("event_confidence_score", pd.Series(np.nan, index=prior.index)),
                    errors="coerce",
                )
                feature_row[f"cluster_mean_confidence_{window}"] = float(confidence.mean())
        rows.append(feature_row)
    features = pd.DataFrame(rows).set_index("_trade_index")
    return trades.join(features, how="left")


def _classify_scan(result: pd.DataFrame, config: StateLifecycleContextConfig) -> pd.DataFrame:
    if result.empty:
        return result
    result = result.copy()
    result["train_supported"] = (
        (result["train_count"] >= config.min_train_count)
        & (result["train_net_r"] > 0.0)
        & (result["train_mean_lift_r"] > config.min_train_mean_lift_r)
    )
    result["oos_supported"] = (
        (result["test_count"] >= config.min_oos_count)
        & (result["test_net_r"] > 0.0)
        & (result["test_mean_lift_r"] > config.min_oos_mean_lift_r)
    )
    result["classification"] = np.select(
        [
            result["train_supported"] & result["oos_supported"],
            result["train_supported"],
            result["oos_supported"],
        ],
        ["train_and_oos_candidate", "train_only_candidate", "oos_only_candidate"],
        default="not_supported",
    )
    return result.sort_values(
        ["test_mean_lift_r", "test_net_r", "train_mean_lift_r"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def _scan_numeric_features(
    trades: pd.DataFrame,
    *,
    feature_columns: list[str],
    family: str,
    config: StateLifecycleContextConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_all = trades[trades["split"].eq("train")]
    for personality, train_personality in train_all.groupby("personality", sort=False):
        test_personality = trades[
            trades["split"].eq("test") & trades["personality"].eq(personality)
        ]
        base_train = _trade_stats(train_personality, "base_train_")
        base_test = _trade_stats(test_personality, "base_test_")
        for feature in feature_columns:
            if feature not in train_personality:
                continue
            values = pd.to_numeric(train_personality[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            values = values.dropna()
            if values.nunique() < 2:
                continue
            if values.min() >= 0 and values.max() <= 20 and values.nunique() <= 12:
                thresholds = sorted({float(value) for value in values.unique()})
            else:
                thresholds = sorted(
                    {
                        float(values.quantile(q))
                        for q in (0.2, 0.33, 0.5, 0.67, 0.8)
                        if pd.notna(values.quantile(q))
                    }
                )
            for threshold in thresholds:
                for operator in ("<=", ">="):
                    train_values = pd.to_numeric(train_personality[feature], errors="coerce")
                    test_values = pd.to_numeric(test_personality[feature], errors="coerce")
                    train_mask = _numeric_threshold_mask(
                        train_values, operator=operator, threshold=threshold
                    )
                    test_mask = _numeric_threshold_mask(
                        test_values, operator=operator, threshold=threshold
                    )
                    train_keep = train_personality[train_mask]
                    test_keep = test_personality[test_mask]
                    if len(train_keep) < config.min_train_count:
                        continue
                    train_stats = _trade_stats(train_keep, "train_")
                    test_stats = _trade_stats(test_keep, "test_")
                    rows.append(
                        {
                            "family": family,
                            "personality": personality,
                            "feature": feature,
                            "operator": operator,
                            "threshold": threshold,
                            **base_train,
                            **train_stats,
                            **base_test,
                            **test_stats,
                            "train_mean_lift_r": train_stats["train_mean_r"]
                            - base_train["base_train_mean_r"],
                            "train_win_lift": train_stats["train_win_rate"]
                            - base_train["base_train_win_rate"],
                            "test_mean_lift_r": (
                                test_stats["test_mean_r"] - base_test["base_test_mean_r"]
                                if len(test_keep)
                                else math.nan
                            ),
                            "test_win_lift": (
                                test_stats["test_win_rate"] - base_test["base_test_win_rate"]
                                if len(test_keep)
                                else math.nan
                            ),
                        }
                    )
    return _classify_scan(pd.DataFrame(rows), config)


def _numeric_threshold_mask(
    values: pd.Series,
    *,
    operator: str,
    threshold: float,
) -> pd.Series:
    if operator == "<=":
        return values <= threshold + NUMERIC_THRESHOLD_EPSILON
    if operator == ">=":
        return values >= threshold - NUMERIC_THRESHOLD_EPSILON
    raise ValueError(f"Unsupported numeric threshold operator: {operator}")


def _scan_categorical_features(
    trades: pd.DataFrame,
    *,
    feature_columns: list[str],
    family: str,
    config: StateLifecycleContextConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_all = trades[trades["split"].eq("train")]
    for personality, train_personality in train_all.groupby("personality", sort=False):
        test_personality = trades[
            trades["split"].eq("test") & trades["personality"].eq(personality)
        ]
        base_train = _trade_stats(train_personality, "base_train_")
        base_test = _trade_stats(test_personality, "base_test_")
        for feature in feature_columns:
            if feature not in train_personality:
                continue
            for value, train_keep in train_personality.groupby(feature, dropna=False):
                if len(train_keep) < config.min_train_count:
                    continue
                test_keep = test_personality[test_personality[feature].astype(str).eq(str(value))]
                train_stats = _trade_stats(train_keep, "train_")
                test_stats = _trade_stats(test_keep, "test_")
                rows.append(
                    {
                        "family": family,
                        "personality": personality,
                        "feature": feature,
                        "value": str(value),
                        **base_train,
                        **train_stats,
                        **base_test,
                        **test_stats,
                        "train_mean_lift_r": train_stats["train_mean_r"]
                        - base_train["base_train_mean_r"],
                        "train_win_lift": train_stats["train_win_rate"]
                        - base_train["base_train_win_rate"],
                        "test_mean_lift_r": (
                            test_stats["test_mean_r"] - base_test["base_test_mean_r"]
                            if len(test_keep)
                            else math.nan
                        ),
                        "test_win_lift": (
                            test_stats["test_win_rate"] - base_test["base_test_win_rate"]
                            if len(test_keep)
                            else math.nan
                        ),
                    }
                )
    return _classify_scan(pd.DataFrame(rows), config)


def _prior_regime_numeric_columns(trades: pd.DataFrame) -> list[str]:
    return [
        column
        for column in trades.columns
        if column.startswith("prev_")
        and (
            column.endswith("_share")
            or column.endswith("_unique_count")
            or column.endswith("_mean")
            or column.endswith("_std")
            or column.endswith("_matches_current")
            or column.startswith("prev_regime_bar_count_")
        )
    ]


def _prior_regime_categorical_columns(trades: pd.DataFrame) -> list[str]:
    return [
        column
        for column in trades.columns
        if column.startswith("prev_")
        and any(
            column.endswith(f"{regime_column}_dominant")
            or column.endswith(f"{regime_column}_last")
            for regime_column in REGIME_COLUMNS
        )
    ]


def _cluster_numeric_columns(trades: pd.DataFrame) -> list[str]:
    return [column for column in trades.columns if column.startswith("cluster_")]


def _base_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"split": split, "personality": personality, **_trade_stats(group)}
        for (split, personality), group in trades.groupby(["split", "personality"], sort=False)
    ]
    return pd.DataFrame(rows)


def _selected_candidates(
    frames: list[pd.DataFrame],
    *,
    max_per_family: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty or "classification" not in frame:
            continue
        supported = frame[frame["classification"].eq("train_and_oos_candidate")].copy()
        if supported.empty:
            continue
        selected.append(
            supported.sort_values(
                ["test_mean_lift_r", "test_net_r", "train_mean_lift_r"],
                ascending=[False, False, False],
                kind="mergesort",
            ).head(max_per_family)
        )
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True, sort=False)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 16) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in shown.iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.5g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_md(
    path: Path,
    *,
    payload: dict[str, Any],
    base: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    numeric_top = (
        numeric[numeric["classification"].eq("train_and_oos_candidate")]
        if not numeric.empty and "classification" in numeric
        else pd.DataFrame()
    )
    categorical_top = (
        categorical[categorical["classification"].eq("train_and_oos_candidate")]
        if not categorical.empty and "classification" in categorical
        else pd.DataFrame()
    )
    cluster_top = (
        clusters[clusters["classification"].eq("train_and_oos_candidate")]
        if not clusters.empty and "classification" in clusters
        else pd.DataFrame()
    )
    lines = [
        "# State Lifecycle Context Lab V0",
        "",
        (
            "Research-only test of prior dense regime mix and prior sparse event clusters "
            "before selected personality trades. No broker, live, paper, vendor fetching, "
            "or order placement. No edge is claimed."
        ),
        "",
        f"Decision: `{payload['decision']}`",
        f"Pipeline hypothesis: `{payload['pipeline_hypothesis']}`",
        f"Lookback bars: `{', '.join(str(item) for item in payload['lookback_bars'])}`",
        f"Lookback minutes: `{', '.join(str(item) for item in payload['lookback_minutes'])}`",
        f"Volume label: `{payload['volume_label']}`",
        "",
        "## Base",
        "",
        _markdown_table(
            base,
            ["split", "personality", "count", "net_r", "mean_r", "win_rate", "symbols", "months"],
        ),
        "",
        "## Prior Regime Numeric Candidates",
        "",
        _markdown_table(
            numeric_top,
            [
                "personality",
                "feature",
                "operator",
                "threshold",
                "train_count",
                "train_net_r",
                "train_mean_lift_r",
                "test_count",
                "test_net_r",
                "test_mean_lift_r",
                "classification",
            ],
        ),
        "",
        "## Prior Regime Categorical Candidates",
        "",
        _markdown_table(
            categorical_top,
            [
                "personality",
                "feature",
                "value",
                "train_count",
                "train_net_r",
                "train_mean_lift_r",
                "test_count",
                "test_net_r",
                "test_mean_lift_r",
                "classification",
            ],
        ),
        "",
        "## Prior Event Cluster Candidates",
        "",
        _markdown_table(
            cluster_top,
            [
                "personality",
                "feature",
                "operator",
                "threshold",
                "train_count",
                "train_net_r",
                "train_mean_lift_r",
                "test_count",
                "test_net_r",
                "test_mean_lift_r",
                "classification",
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_state_lifecycle_context_lab(
    *,
    input_expression_report_dir: Path,
    input_event_dir: Path,
    data_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: StateLifecycleContextConfig = StateLifecycleContextConfig(),
) -> StateLifecycleContextResult:
    """Run a research-only state lifecycle context lab."""

    trades = _load_trades(input_expression_report_dir)
    events = _load_event_rows(input_event_dir)
    trades = _add_prior_regime_features(trades, data_dir=data_dir, config=config)
    trades = _add_prior_event_cluster_features(trades, events, config=config)
    from stocker_research.shadow_candidate_trigger_audit_v0 import (
        add_shadow_candidate_trigger_features,
    )

    trades = add_shadow_candidate_trigger_features(trades)

    base = _base_summary(trades)
    numeric = _scan_numeric_features(
        trades,
        feature_columns=_prior_regime_numeric_columns(trades),
        family="prior_regime_numeric",
        config=config,
    )
    categorical = _scan_categorical_features(
        trades,
        feature_columns=_prior_regime_categorical_columns(trades),
        family="prior_regime_categorical",
        config=config,
    )
    clusters = _scan_numeric_features(
        trades,
        feature_columns=_cluster_numeric_columns(trades),
        family="prior_event_cluster",
        config=config,
    )
    selected = _selected_candidates(
        [numeric, categorical, clusters],
        max_per_family=config.max_selected_per_family,
    )
    decision = (
        "continue_research_state_lifecycle_context"
        if not selected.empty
        else "reject_no_train_and_oos_context_candidates"
    )

    run_id = "state_lifecycle_context_lab_v0_" + datetime.now(tz=UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision": run_dir / "decision.json",
        "trade_context_features": run_dir / "trade_context_features.csv",
        "base_summary": run_dir / "base_summary.csv",
        "prior_regime_numeric_scan": run_dir / "prior_regime_numeric_scan.csv",
        "prior_regime_categorical_scan": run_dir / "prior_regime_categorical_scan.csv",
        "prior_event_cluster_scan": run_dir / "prior_event_cluster_scan.csv",
        "selected_context_candidates": run_dir / "selected_context_candidates.csv",
    }
    payload: dict[str, Any] = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "decision": decision,
        "pipeline_hypothesis": (
            "prior_regime_mix + prior_personality_cluster + current_personality + exit"
        ),
        "input_expression_report_dir": input_expression_report_dir,
        "input_event_dir": input_event_dir,
        "data_dir": data_dir,
        "lookback_bars": list(config.lookback_bars),
        "lookback_minutes": [int(window) * 5 for window in config.lookback_bars],
        "volume_label": "historical_volume from existing local 5m OHLCV parquet",
        "train_trade_count": int(trades["split"].eq("train").sum()),
        "test_trade_count": int(trades["split"].eq("test").sum()),
        "prior_regime_numeric_candidate_count": int(
            numeric["classification"].eq("train_and_oos_candidate").sum()
        )
        if not numeric.empty and "classification" in numeric
        else 0,
        "prior_regime_categorical_candidate_count": int(
            categorical["classification"].eq("train_and_oos_candidate").sum()
        )
        if not categorical.empty and "classification" in categorical
        else 0,
        "prior_event_cluster_candidate_count": int(
            clusters["classification"].eq("train_and_oos_candidate").sum()
        )
        if not clusters.empty and "classification" in clusters
        else 0,
        "selected_context_candidate_count": int(len(selected)),
    }
    decision_payload = {
        "decision": decision,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }
    _write_csv(paths["trade_context_features"], trades)
    _write_csv(paths["base_summary"], base)
    _write_csv(paths["prior_regime_numeric_scan"], numeric)
    _write_csv(paths["prior_regime_categorical_scan"], categorical)
    _write_csv(paths["prior_event_cluster_scan"], clusters)
    _write_csv(paths["selected_context_candidates"], selected)
    _write_json(paths["summary_json"], payload)
    _write_json(paths["decision"], decision_payload)
    _write_summary_md(
        paths["summary_md"],
        payload=payload,
        base=base,
        numeric=numeric,
        categorical=categorical,
        clusters=clusters,
    )
    return StateLifecycleContextResult(
        run_id=run_id,
        input_expression_report_dir=input_expression_report_dir,
        input_event_dir=input_event_dir,
        data_dir=data_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision"],
        trade_context_features_csv_path=paths["trade_context_features"],
        base_summary_csv_path=paths["base_summary"],
        prior_regime_numeric_scan_csv_path=paths["prior_regime_numeric_scan"],
        prior_regime_categorical_scan_csv_path=paths["prior_regime_categorical_scan"],
        prior_event_cluster_scan_csv_path=paths["prior_event_cluster_scan"],
        selected_context_candidates_csv_path=paths["selected_context_candidates"],
        decision=decision,
        selected_candidate_count=int(len(selected)),
    )


__all__ = [
    "StateLifecycleContextConfig",
    "StateLifecycleContextResult",
    "run_state_lifecycle_context_lab",
]
