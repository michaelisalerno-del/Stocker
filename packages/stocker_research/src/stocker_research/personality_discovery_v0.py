"""YAML-driven personality discovery over sparse state-event rows.

This research-only lab keeps the personality set explicit and versioned while
letting the scan search regimes, simple filters, and caveats per personality.
It does not touch execution, broker, paper trading, live trading, order
placement, or vendor fetching paths.
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
import yaml

from stocker_research.event_failure_cutter_v0 import find_latest_state_event_detector_run

DEFAULT_INPUT_BASE_DIR = Path("data/reports/research/state_event_detector_v0")
DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_discovery_v0")
DEFAULT_SPEC_DIR = Path("configs/research/personalities/v0")

EVENT_STATE_PERSONALITY: dict[str, tuple[str, str, int]] = {
    "controlled_pullback_after_bullish_impulse": (
        "pullback_continuation",
        "long_continuation",
        1,
    ),
    "liquidation_failed_low_reclaim": ("reclaim_reversal", "long_reversal", 1),
    "slow_snapback_after_dip": ("slow_repair", "long_reversal", 1),
    "failed_bounce_active_liquidation": (
        "active_liquidation",
        "short_or_long_blocker",
        -1,
    ),
    "failed_bullish_impulse_recoil": (
        "impulse_recoil",
        "short_or_long_blocker",
        -1,
    ),
    "failed_open_down_continuation": (
        "open_down_pressure",
        "short_or_long_blocker",
        -1,
    ),
    "dead_chop_blocker": ("dead_chop_noise", "no_trade_filter", 0),
    "exhaustion_extension": ("exhaustion_extension", "mean_reversion_or_no_chase", 0),
}

REGIME_FIELDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "vwap_side": ("vwap_side_regime",),
    "opening_mid_side": ("opening_mid_side_regime",),
    "session_open_side": ("session_open_side_regime",),
    "range": ("range_regime",),
    "compression": ("compression_regime",),
    "efficiency": ("efficiency_regime",),
    "relative_volume": ("relative_volume_regime",),
    "time_of_day": ("time_regime", "bar_index_bucket"),
    "vwap_x_efficiency": ("vwap_x_efficiency_regime",),
    "vwap_x_range": ("vwap_x_range_regime",),
    "compression_x_efficiency": ("compression_x_efficiency_regime",),
    "opening_mid_x_range": ("opening_mid_x_range_regime",),
    "time_x_vwap": ("time_x_vwap_regime",),
    "volume_x_vwap": ("volume_x_vwap_regime",),
    "auction_location": (
        "auction_current_location",
        "auction_opening_mid_location",
        "auction_session_open_location",
        "auction_prior_close_side",
        "auction_gap_bucket",
        "auction_prior_range_zone",
        "auction_prior_level_interaction",
    ),
    "cross_stock_alignment": (
        "cross_stock_same_direction_bucket",
        "cross_stock_same_personality_bucket",
    ),
    "event_quality": ("event_quality_regime",),
}

FILTER_FEATURES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "vwap_distance": ("distance_from_vwap_pct", "abs_distance_from_vwap_pct"),
    "opening_range": (
        "distance_from_opening_range_mid_pct",
        "distance_from_opening_range_high_pct",
        "distance_from_opening_range_low_pct",
    ),
    "session_location": (
        "distance_from_session_open_pct",
        "distance_from_session_high_pct",
        "distance_from_session_low_pct",
    ),
    "recent_structure": (
        "distance_from_recent_high_pct",
        "distance_from_recent_low_pct",
        "reclaim_from_recent_low",
        "pullback_depth_from_recent_high",
    ),
    "wick_quality": (
        "upper_wick_pct_of_range",
        "lower_wick_pct_of_range",
        "close_location_value",
        "role_rejection_wick",
        "role_close_rejection",
    ),
    "prior_return": (
        "bar_return",
        "prior_3_bar_return",
        "prior_6_bar_return",
        "prior_12_bar_return",
        "role_bar_reversal",
        "role_prior_3_turn",
    ),
    "efficiency": ("directional_efficiency_6", "directional_efficiency_12"),
    "range_volatility": (
        "rolling_intraday_range_pct",
        "compression_zscore",
        "range_zscore",
        "range_zscore_20",
    ),
    "cross_counts": ("vwap_cross_count_12", "range_cross_count_12"),
    "volume": ("relative_volume_at_bar_index", "relative_cumulative_volume"),
    "time": ("bar_index_in_session",),
    "auction_location": (
        "distance_from_prior_high_pct",
        "distance_from_prior_low_pct",
        "distance_from_prior_close_pct",
        "gap_from_prior_close_pct",
        "prior_range_position",
    ),
    "cross_stock_alignment": (
        "same_direction_other_symbol_count_15m",
        "same_personality_other_symbol_count_15m",
        "same_direction_other_symbol_count_30m",
        "same_personality_other_symbol_count_30m",
    ),
    "event_quality": (
        "event_quality_score",
        "role_rejection_wick",
        "role_close_rejection",
        "role_bar_reversal",
    ),
}

CAVEAT_FIELDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "auction_location": REGIME_FIELDS_BY_FAMILY["auction_location"]
    + FILTER_FEATURES_BY_FAMILY["auction_location"],
    "event_quality": REGIME_FIELDS_BY_FAMILY["event_quality"]
    + FILTER_FEATURES_BY_FAMILY["event_quality"],
    "time_of_day": REGIME_FIELDS_BY_FAMILY["time_of_day"]
    + FILTER_FEATURES_BY_FAMILY["time"],
    "cross_stock_alignment": REGIME_FIELDS_BY_FAMILY["cross_stock_alignment"]
    + FILTER_FEATURES_BY_FAMILY["cross_stock_alignment"],
    "freshness": (
        "personality_event_index_in_session",
        "same_personality_event_count_so_far",
    ),
}


@dataclass(frozen=True)
class PersonalitySpec:
    """Versioned YAML spec for one personality discovery surface."""

    personality: str
    version: str
    role: str
    default_expected_direction: int
    base_event_states: tuple[str, ...]
    preferred_horizons: tuple[int, ...]
    regime_families: tuple[str, ...]
    filter_families: tuple[str, ...]
    caveat_families: tuple[str, ...]
    train_count: int
    test_count: int
    retained_count: int
    symbol_count: int
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    candidate: bool = False


@dataclass(frozen=True)
class PersonalityDiscoveryConfig:
    """Configuration for YAML-driven personality discovery."""

    horizons: tuple[int, ...] = (6, 9, 12, 24)
    train_fraction: float = 0.60
    random_seed: int = 1337
    random_iterations: int = 100
    default_min_train_events: int = 30
    default_min_test_events: int = 12
    default_min_retained_events: int = 8
    default_min_symbols: int = 3
    min_train_lift: float = 0.03
    min_train_median_lift: float = 0.0
    max_filters_per_personality_horizon: int = 80
    max_selected_rows_per_personality_horizon: int = 40
    max_pair_seed_filters: int = 12


@dataclass(frozen=True)
class PersonalityDiscoveryResult:
    """Paths and headline result for a personality discovery run."""

    run_id: str
    input_dir: Path
    spec_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    loaded_specs_csv_path: Path
    personality_base_summary_csv_path: Path
    candidate_rules_csv_path: Path
    selected_rules_csv_path: Path
    passed_rules_csv_path: Path
    rejected_rules_csv_path: Path
    random_baseline_csv_path: Path
    concentration_warnings_csv_path: Path
    examples_csv_path: Path
    decision_matrix_csv_path: Path
    decision: str
    passed_rule_count: int


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


def _as_tuple(raw: Any) -> tuple[Any, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return (raw,)


def load_personality_specs(spec_dir: Path) -> list[PersonalitySpec]:
    """Load personality YAML specs from a directory."""

    if not spec_dir.exists():
        raise FileNotFoundError(f"Personality spec directory not found: {spec_dir}")
    specs: list[PersonalitySpec] = []
    for path in sorted(spec_dir.glob("*.yaml")) + sorted(spec_dir.glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        minimums = payload.get("minimums") or {}
        personality = str(payload["personality"])
        specs.append(
            PersonalitySpec(
                personality=personality,
                version=str(payload.get("version", "v0")),
                role=str(payload["role"]),
                default_expected_direction=int(payload.get("default_expected_direction", 0)),
                base_event_states=tuple(
                    str(item) for item in _as_tuple(payload.get("base_event_states"))
                ),
                preferred_horizons=tuple(
                    int(item)
                    for item in _as_tuple(payload.get("preferred_horizons", (6, 9, 12, 24)))
                ),
                regime_families=tuple(
                    str(item) for item in _as_tuple(payload.get("regime_families"))
                ),
                filter_families=tuple(
                    str(item) for item in _as_tuple(payload.get("filter_families"))
                ),
                caveat_families=tuple(
                    str(item) for item in _as_tuple(payload.get("caveat_families"))
                ),
                train_count=int(minimums.get("train_count", 30)),
                test_count=int(minimums.get("test_count", 12)),
                retained_count=int(minimums.get("retained_count", 8)),
                symbol_count=int(minimums.get("symbol_count", 3)),
                max_single_symbol_share=float(payload.get("max_single_symbol_share", 0.50)),
                max_single_session_share=float(payload.get("max_single_session_share", 0.20)),
                max_single_month_share=float(payload.get("max_single_month_share", 0.50)),
                candidate=bool(payload.get("candidate", False)),
            )
        )
    if not specs:
        raise FileNotFoundError(f"No personality YAML specs found in {spec_dir}")
    return specs


def _side_bucket(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([series > 0.001, series < -0.001], ["above", "below"], default="near"),
        index=series.index,
    )


def _bucket(
    series: pd.Series,
    *,
    low: float,
    high: float,
    low_name: str,
    mid_name: str,
    high_name: str,
) -> pd.Series:
    return pd.Series(
        np.select([series <= low, series >= high], [low_name, high_name], default=mid_name),
        index=series.index,
    )


def _relative_volume_bucket(series: pd.Series) -> pd.Series:
    result = pd.Series("normal_relative_volume", index=series.index, dtype="object")
    result[series.isna()] = "unknown"
    result[series <= 0.8] = "low_relative_volume"
    result[series >= 1.2] = "high_relative_volume"
    return result


def _map_state_personality(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    mapped = frame["event_state"].astype(str).map(EVENT_STATE_PERSONALITY)
    frame["personality"] = mapped.map(lambda value: value[0] if isinstance(value, tuple) else None)
    frame["role"] = mapped.map(lambda value: value[1] if isinstance(value, tuple) else None)
    frame["default_expected_direction"] = mapped.map(
        lambda value: value[2] if isinstance(value, tuple) else 0
    )
    return frame


def _add_cross_stock_alignment_features(frame: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    if "timestamp" not in frame or "symbol" not in frame:
        return frame
    data = frame.copy()
    bucket_col = f"cross_stock_bucket_{minutes}m"
    data[bucket_col] = data["timestamp"].dt.floor(f"{minutes}min")
    key = ["session_date", bucket_col]
    if "default_expected_direction" not in data:
        data["default_expected_direction"] = 0

    personality_symbols = (
        data.groupby(key + ["personality"])["symbol"]
        .agg(lambda series: set(series.dropna()))
        .rename("_same_personality_symbols")
        .reset_index()
    )
    direction_symbols = (
        data.groupby(key + ["default_expected_direction"])["symbol"]
        .agg(lambda series: set(series.dropna()))
        .rename("_same_direction_symbols")
        .reset_index()
    )
    data = data.merge(personality_symbols, on=key + ["personality"], how="left")
    data = data.merge(direction_symbols, on=key + ["default_expected_direction"], how="left")
    data[f"same_personality_other_symbol_count_{minutes}m"] = data.apply(
        lambda row: len((row["_same_personality_symbols"] or set()) - {row["symbol"]}),
        axis=1,
    )
    data[f"same_direction_other_symbol_count_{minutes}m"] = data.apply(
        lambda row: len((row["_same_direction_symbols"] or set()) - {row["symbol"]}),
        axis=1,
    )
    data["cross_stock_same_personality_bucket"] = np.where(
        data[f"same_personality_other_symbol_count_{minutes}m"] >= 1,
        "same_personality_elsewhere",
        "no_same_personality_elsewhere",
    )
    data["cross_stock_same_direction_bucket"] = np.where(
        data[f"same_direction_other_symbol_count_{minutes}m"] >= 1,
        "same_direction_elsewhere",
        "no_same_direction_elsewhere",
    )
    return data.drop(columns=["_same_personality_symbols", "_same_direction_symbols"])


def add_discovery_features(event_rows: pd.DataFrame) -> pd.DataFrame:
    """Add reusable regime, filter, and caveat features to event rows."""

    data = event_rows.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["session_date"] = pd.to_datetime(data["session_date"]).dt.strftime("%Y-%m-%d")
    data["month"] = data["timestamp"].dt.strftime("%Y-%m")
    data = _map_state_personality(data)

    data["vwap_side_regime"] = _side_bucket(
        data.get("distance_from_vwap_pct", pd.Series(0, index=data.index))
    )
    data["opening_mid_side_regime"] = _side_bucket(
        data.get("distance_from_opening_range_mid_pct", pd.Series(0, index=data.index))
    )
    data["session_open_side_regime"] = _side_bucket(
        data.get("distance_from_session_open_pct", pd.Series(0, index=data.index))
    )
    data["range_regime"] = _bucket(
        data.get("rolling_intraday_range_pct", pd.Series(np.nan, index=data.index)),
        low=0.01079666,
        high=0.01711841,
        low_name="low_range",
        mid_name="mid_range",
        high_name="high_range",
    )
    data["compression_regime"] = _bucket(
        data.get("compression_zscore", pd.Series(np.nan, index=data.index)),
        low=-0.5,
        high=0.5,
        low_name="compressed",
        mid_name="normal_compression",
        high_name="expanded",
    )
    data["efficiency_regime"] = _bucket(
        data.get("directional_efficiency_12", pd.Series(np.nan, index=data.index)),
        low=0.25,
        high=0.55,
        low_name="choppy_efficiency",
        mid_name="mixed_efficiency",
        high_name="directional_efficiency",
    )
    data["relative_volume_regime"] = _relative_volume_bucket(
        data.get("relative_volume_at_bar_index", pd.Series(np.nan, index=data.index))
    )
    if "time_of_day_bucket" in data:
        data["time_regime"] = data["time_of_day_bucket"].astype(str)
    else:
        index = data.get("bar_index_in_session", pd.Series(0, index=data.index))
        data["time_regime"] = np.select(
            [index <= 12, index <= 24, index <= 60],
            ["post_open", "morning", "midday"],
            default="late_day",
        )
    if "bar_index_bucket" not in data:
        index = data.get("bar_index_in_session", pd.Series(0, index=data.index))
        data["bar_index_bucket"] = np.select(
            [index <= 12, index <= 24, index <= 60],
            ["post_open", "morning", "midday"],
            default="late_day",
        )

    data["vwap_x_efficiency_regime"] = data["vwap_side_regime"] + "|" + data["efficiency_regime"]
    data["vwap_x_range_regime"] = data["vwap_side_regime"] + "|" + data["range_regime"]
    data["compression_x_efficiency_regime"] = (
        data["compression_regime"] + "|" + data["efficiency_regime"]
    )
    data["opening_mid_x_range_regime"] = (
        data["opening_mid_side_regime"] + "|" + data["range_regime"]
    )
    data["time_x_vwap_regime"] = data["time_regime"] + "|" + data["vwap_side_regime"]
    data["volume_x_vwap_regime"] = data["relative_volume_regime"] + "|" + data["vwap_side_regime"]

    data["abs_distance_from_vwap_pct"] = data.get(
        "distance_from_vwap_pct", pd.Series(0, index=data.index)
    ).abs()
    direction = data["default_expected_direction"].replace(0, np.nan).fillna(1)
    data["role_rejection_wick"] = np.where(
        direction < 0,
        data.get("upper_wick_pct_of_range", pd.Series(0, index=data.index)),
        data.get("lower_wick_pct_of_range", pd.Series(0, index=data.index)),
    )
    data["role_close_rejection"] = np.where(
        direction < 0,
        1 - data.get("close_location_value", pd.Series(0.5, index=data.index)),
        data.get("close_location_value", pd.Series(0.5, index=data.index)),
    )
    data["role_bar_reversal"] = direction * data.get("bar_return", pd.Series(0, index=data.index))
    data["role_prior_3_turn"] = direction * data.get(
        "prior_3_bar_return", pd.Series(0, index=data.index)
    )
    data["event_quality_score"] = (
        data["role_rejection_wick"].fillna(0)
        + data["role_close_rejection"].fillna(0)
        + data.get("directional_efficiency_6", pd.Series(0, index=data.index)).fillna(0)
    ) / 3
    data["event_quality_regime"] = _bucket(
        data["event_quality_score"],
        low=0.35,
        high=0.60,
        low_name="low_event_quality",
        mid_name="mixed_event_quality",
        high_name="high_event_quality",
    )

    high_dist = data.get("distance_from_session_high_pct", pd.Series(np.nan, index=data.index))
    low_dist = data.get("distance_from_session_low_pct", pd.Series(np.nan, index=data.index))
    data["auction_current_location"] = pd.Series(
        np.select(
            [high_dist >= -0.005, low_dist <= 0.005],
            ["near_session_high_to_date", "near_session_low_to_date"],
            default="middle_session_range_to_date",
        ),
        index=data.index,
    )
    data["auction_opening_mid_location"] = data["opening_mid_side_regime"]
    data["auction_session_open_location"] = data["session_open_side_regime"]
    if "distance_from_prior_close_pct" in data:
        data["auction_prior_close_side"] = _side_bucket(data["distance_from_prior_close_pct"])
    else:
        data["auction_prior_close_side"] = "unknown_prior_close"
    if "gap_from_prior_close_pct" in data:
        gap = data["gap_from_prior_close_pct"]
        data["auction_gap_bucket"] = pd.Series(
            np.select(
                [gap >= 0.01, gap >= 0.003, gap <= -0.01, gap <= -0.003],
                ["large_gap_up", "gap_up", "large_gap_down", "gap_down"],
                default="flat_gap",
            ),
            index=data.index,
        )
    else:
        data["auction_gap_bucket"] = "unknown_gap"
    if "prior_range_position" in data:
        pos = data["prior_range_position"]
        data["auction_prior_range_zone"] = pd.Series(
            np.select(
                [pos < 0, pos <= 0.2, pos <= 0.8, pos <= 1.0],
                [
                    "below_prior_range",
                    "lower_prior_range",
                    "middle_prior_range",
                    "upper_prior_range",
                ],
                default="above_prior_range",
            ),
            index=data.index,
        )
    else:
        data["auction_prior_range_zone"] = "unknown_prior_range"
    if "prior_level_interaction" not in data:
        data["auction_prior_level_interaction"] = "unknown_prior_level_interaction"
    else:
        data["auction_prior_level_interaction"] = data["prior_level_interaction"].astype(str)

    data = data.sort_values(["symbol", "session_date", "timestamp"]).reset_index(drop=True)
    data["personality_event_index_in_session"] = (
        data.groupby(["symbol", "session_date", "personality"]).cumcount() + 1
    )
    data["same_personality_event_count_so_far"] = data["personality_event_index_in_session"]
    data = _add_cross_stock_alignment_features(data, minutes=15)
    return _add_cross_stock_alignment_features(data, minutes=30)


def _return_column(horizon: int) -> str:
    return f"forward_{horizon}_bar_return"


def _abs_return_column(horizon: int) -> str:
    return f"forward_{horizon}_bar_abs_return"


def _score_rows(rows: pd.DataFrame, *, horizon: int, expected_direction: int) -> pd.DataFrame:
    data = rows.copy()
    ret_col = _return_column(horizon)
    if expected_direction == 0:
        abs_col = _abs_return_column(horizon)
        if abs_col in data:
            threshold = float(data[abs_col].median())
            data["role_score"] = threshold - data[abs_col]
        else:
            threshold = float(data[ret_col].abs().median())
            data["role_score"] = threshold - data[ret_col].abs()
    else:
        data["role_score"] = expected_direction * data[ret_col]
    data["same_result"] = data["role_score"] > 0
    return data


def _family_fields(
    families: Sequence[str],
    mapping: dict[str, tuple[str, ...]],
    available_columns: set[str],
) -> tuple[str, ...]:
    fields: list[str] = []
    for family in families:
        for field in mapping.get(family, ()):
            if field in available_columns and field not in fields:
                fields.append(field)
    return tuple(fields)


def build_filter_candidates(
    train_rows: pd.DataFrame,
    *,
    spec: PersonalitySpec,
    horizon: int,
    config: PersonalityDiscoveryConfig,
) -> pd.DataFrame:
    """Build train-only single-feature and two-feature filter candidates."""

    available = set(train_rows.columns)
    features: list[str] = []
    for field in _family_fields(spec.filter_families, FILTER_FEATURES_BY_FAMILY, available):
        if field not in features:
            features.append(field)
    for field in _family_fields(spec.caveat_families, CAVEAT_FIELDS_BY_FAMILY, available):
        if field not in features:
            features.append(field)
    feature_thresholds: dict[str, dict[float, float]] = {}
    quantiles = (0.50, 0.25, 0.33, 0.67, 0.75)
    for feature in features:
        if feature not in train_rows:
            continue
        series = train_rows[feature]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = series.replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 2:
            continue
        feature_thresholds[feature] = {
            quantile: float(values.quantile(quantile)) for quantile in quantiles
        }

    rows: list[dict[str, Any]] = []
    feature_single_rows: dict[str, list[dict[str, Any]]] = {}
    for quantile in quantiles:
        for feature in features:
            if feature not in feature_thresholds:
                continue
            threshold = feature_thresholds[feature][quantile]
            feature_rows = feature_single_rows.setdefault(feature, [])
            for operator in ("<=", ">="):
                row = {
                    "personality": spec.personality,
                    "horizon": horizon,
                    "rule_kind": "single",
                    "feature": feature,
                    "operator": operator,
                    "threshold": threshold,
                    "feature_b": "",
                    "operator_b": "",
                    "threshold_b": math.nan,
                    "filter_rule": f"{feature} {operator} {threshold:.6g}",
                }
                feature_rows.append(row)
                rows.append(row)
    single_candidates = pd.DataFrame(rows)
    if single_candidates.empty:
        return single_candidates
    single_candidates = single_candidates.drop_duplicates(["feature", "operator", "threshold"])

    max_total = max(1, config.max_filters_per_personality_horizon)
    single_limit = min(len(single_candidates), max(24, int(max_total * 0.55)), max_total)
    single_candidates = single_candidates.head(single_limit)

    pair_seed: list[dict[str, Any]] = []
    for _feature, feature_rows in feature_single_rows.items():
        if len(pair_seed) >= config.max_pair_seed_filters:
            break
        if feature_rows:
            pair_seed.append(feature_rows[0])

    pair_rows: list[dict[str, Any]] = []
    remaining = max_total - len(single_candidates)
    if remaining > 0:
        for left_index, left in enumerate(pair_seed):
            for right in pair_seed[left_index + 1 :]:
                if left["feature"] == right["feature"]:
                    continue
                for rule_kind, joiner in (("and", "AND"), ("or", "OR")):
                    pair_rows.append(
                        {
                            "personality": spec.personality,
                            "horizon": horizon,
                            "rule_kind": rule_kind,
                            "feature": left["feature"],
                            "operator": left["operator"],
                            "threshold": left["threshold"],
                            "feature_b": right["feature"],
                            "operator_b": right["operator"],
                            "threshold_b": right["threshold"],
                            "filter_rule": (
                                f"({left['feature']} {left['operator']} {left['threshold']:.6g}) "
                                f"{joiner} "
                                f"({right['feature']} {right['operator']} {right['threshold']:.6g})"
                            ),
                        }
                    )
                    if len(pair_rows) >= remaining:
                        break
                if len(pair_rows) >= remaining:
                    break
            if len(pair_rows) >= remaining:
                break

    candidates = pd.concat([single_candidates, pd.DataFrame(pair_rows)], ignore_index=True)
    if candidates.empty:
        return candidates
    return candidates.head(max_total)


def _op_mask(series: pd.Series, operator: str, threshold: Any) -> pd.Series:
    if operator == "==":
        return series.astype(str) == str(threshold)
    numeric_threshold = float(threshold)
    if operator == "<=":
        return series <= numeric_threshold
    if operator == ">=":
        return series >= numeric_threshold
    if operator == "<":
        return series < numeric_threshold
    if operator == ">":
        return series > numeric_threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _candidate_mask(rows: pd.DataFrame, filter_row: pd.Series) -> pd.Series:
    feature = str(filter_row["feature"])
    if feature not in rows:
        return pd.Series(False, index=rows.index)
    left = _op_mask(
        rows[feature],
        str(filter_row["operator"]),
        filter_row["threshold"],
    ).fillna(False)
    rule_kind = str(filter_row.get("rule_kind", "single"))
    if rule_kind == "single":
        return left
    feature_b = str(filter_row.get("feature_b", ""))
    if not feature_b or feature_b not in rows:
        return pd.Series(False, index=rows.index)
    right = _op_mask(
        rows[feature_b],
        str(filter_row["operator_b"]),
        filter_row["threshold_b"],
    ).fillna(False)
    if rule_kind == "and":
        return left & right
    if rule_kind == "or":
        return left | right
    raise ValueError(f"Unsupported rule kind: {rule_kind}")


def _random_baseline(
    pool: pd.DataFrame,
    *,
    count: int,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    if count <= 0 or len(pool) < count:
        return {
            "random_same_count_mean_rate": math.nan,
            "random_same_count_p95_rate": math.nan,
            "random_same_count_median_score": math.nan,
        }
    rng = np.random.default_rng(seed)
    same = pool["same_result"].astype(float).to_numpy()
    score = pool["role_score"].astype(float).to_numpy()
    rates: list[float] = []
    medians: list[float] = []
    for _ in range(iterations):
        sample = rng.choice(len(pool), size=count, replace=False)
        rates.append(float(np.nanmean(same[sample])))
        medians.append(float(np.nanmedian(score[sample])))
    return {
        "random_same_count_mean_rate": float(np.nanmean(rates)),
        "random_same_count_p95_rate": float(np.nanquantile(rates, 0.95)),
        "random_same_count_median_score": float(np.nanmedian(medians)),
    }


def _concentration(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {
            "symbol_count": 0,
            "single_symbol_share": math.nan,
            "session_count": 0,
            "single_session_share": math.nan,
            "month_count": 0,
            "single_month_share": math.nan,
        }
    symbol_counts = rows["symbol"].value_counts()
    session_counts = (
        rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1).value_counts()
    )
    month_counts = rows["month"].value_counts()
    return {
        "symbol_count": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "session_count": int(session_counts.size),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "month_count": int(month_counts.size),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
    }


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _format_bps(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{10000 * float(value):.2f} bps"


def _latest_input(input_dir: Path | None, input_base_dir: Path) -> Path:
    return (
        input_dir
        if input_dir is not None
        else find_latest_state_event_detector_run(input_base_dir)
    )


def _decision(passed: pd.DataFrame, specs: Sequence[PersonalitySpec]) -> str:
    if passed.empty:
        return "reject_no_personality_discovery_oos_lift"
    stable = {spec.personality for spec in specs if not spec.candidate}
    passed_stable = set(passed["personality"]) & stable
    if len(passed_stable) >= 3:
        return "continue_research_personality_discovery_multi_personality"
    if passed_stable:
        return "continue_research_personality_discovery_narrow"
    return "continue_research_candidate_personality_only"


def run_personality_discovery_lab(
    *,
    input_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_INPUT_BASE_DIR,
    spec_dir: Path = DEFAULT_SPEC_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityDiscoveryConfig = PersonalityDiscoveryConfig(),
) -> PersonalityDiscoveryResult:
    """Run YAML-driven personality discovery over a state-event report."""

    resolved_input = _latest_input(input_dir, input_base_dir)
    event_rows_path = resolved_input / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    specs = load_personality_specs(spec_dir)
    events = add_discovery_features(pd.read_csv(event_rows_path))
    cutoff = events["timestamp"].sort_values().iloc[int(len(events) * config.train_fraction)]
    events["split"] = np.where(events["timestamp"] <= cutoff, "train", "test")

    run_id = f"personality_discovery_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    loaded_specs = pd.DataFrame([asdict(spec) for spec in specs])
    base_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(specs):
        spec_events = events[events["event_state"].astype(str).isin(spec.base_event_states)].copy()
        if spec_events.empty:
            continue
        available = set(spec_events.columns)
        regime_fields = _family_fields(
            (*spec.regime_families, *spec.caveat_families),
            REGIME_FIELDS_BY_FAMILY,
            available,
        )
        if not regime_fields:
            regime_fields = ("personality",)
            spec_events["personality"] = spec.personality
        for horizon in (h for h in config.horizons if h in spec.preferred_horizons):
            ret_col = _return_column(horizon)
            if ret_col not in spec_events:
                continue
            horizon_events = spec_events.dropna(subset=[ret_col]).copy()
            if horizon_events.empty:
                continue
            horizon_events = _score_rows(
                horizon_events,
                horizon=horizon,
                expected_direction=spec.default_expected_direction,
            )
            train_all = horizon_events[horizon_events["split"] == "train"].copy()
            test_all = horizon_events[horizon_events["split"] == "test"].copy()
            if len(train_all) < spec.train_count or len(test_all) < spec.test_count:
                continue
            base_train_rate = float(train_all["same_result"].mean())
            base_test_rate = float(test_all["same_result"].mean())
            base_train_median = float(train_all["role_score"].median())
            base_test_median = float(test_all["role_score"].median())
            base_rows.append(
                {
                    "personality": spec.personality,
                    "role": spec.role,
                    "candidate": spec.candidate,
                    "horizon": horizon,
                    "train_count": int(len(train_all)),
                    "test_count": int(len(test_all)),
                    "base_train_same_result_rate": base_train_rate,
                    "base_test_same_result_rate": base_test_rate,
                    "base_train_median_score": base_train_median,
                    "base_test_median_score": base_test_median,
                }
            )

            filter_candidates = build_filter_candidates(
                train_all,
                spec=spec,
                horizon=horizon,
                config=config,
            )
            if filter_candidates.empty:
                continue
            for regime_field in regime_fields:
                if regime_field not in train_all:
                    continue
                regime_counts = train_all[regime_field].dropna().value_counts()
                for regime_value, train_count in regime_counts.items():
                    if int(train_count) < spec.train_count:
                        continue
                    train_regime = train_all[
                        train_all[regime_field].astype(str) == str(regime_value)
                    ]
                    test_regime = test_all[
                        test_all[regime_field].astype(str) == str(regime_value)
                    ]
                    if train_regime.empty:
                        continue
                    regime_train_rate = float(train_regime["same_result"].mean())
                    regime_train_median = float(train_regime["role_score"].median())
                    regime_test_rate = (
                        float(test_regime["same_result"].mean())
                        if not test_regime.empty
                        else math.nan
                    )
                    regime_test_median = (
                        float(test_regime["role_score"].median())
                        if not test_regime.empty
                        else math.nan
                    )
                    for _, filter_row in filter_candidates.iterrows():
                        feature = str(filter_row["feature"])
                        if feature not in train_regime:
                            continue
                        train_filter = train_regime[_candidate_mask(train_regime, filter_row)]
                        if len(train_filter) < spec.retained_count:
                            continue
                        filter_train_rate = float(train_filter["same_result"].mean())
                        filter_train_median = float(train_filter["role_score"].median())
                        candidate = {
                            "personality": spec.personality,
                            "role": spec.role,
                            "candidate": spec.candidate,
                            "horizon": horizon,
                            "regime_field": regime_field,
                            "regime_value": regime_value,
                            "filter_rule": filter_row["filter_rule"],
                            "rule_kind": filter_row.get("rule_kind", "single"),
                            "feature": feature,
                            "operator": filter_row["operator"],
                            "threshold": filter_row["threshold"],
                            "feature_b": filter_row.get("feature_b", ""),
                            "operator_b": filter_row.get("operator_b", ""),
                            "threshold_b": filter_row.get("threshold_b", math.nan),
                            "base_train_count": int(len(train_all)),
                            "regime_train_count": int(len(train_regime)),
                            "retained_train_count": int(len(train_filter)),
                            "base_train_same_result_rate": base_train_rate,
                            "regime_train_same_result_rate": regime_train_rate,
                            "filtered_train_same_result_rate": filter_train_rate,
                            "train_lift_vs_personality": filter_train_rate - base_train_rate,
                            "train_lift_vs_regime": filter_train_rate - regime_train_rate,
                            "base_train_median_score": base_train_median,
                            "regime_train_median_score": regime_train_median,
                            "filtered_train_median_score": filter_train_median,
                            "train_median_lift_vs_personality": (
                                filter_train_median - base_train_median
                            ),
                            "train_median_lift_vs_regime": (
                                filter_train_median - regime_train_median
                            ),
                        }
                        candidate_rows.append(candidate)
                        if (
                            filter_train_rate < regime_train_rate + config.min_train_lift
                            or filter_train_median
                            <= regime_train_median + config.min_train_median_lift
                        ):
                            continue
                        test_filter = test_regime[_candidate_mask(test_regime, filter_row)].copy()
                        random_result = _random_baseline(
                            test_regime if not test_regime.empty else test_all,
                            count=len(test_filter),
                            seed=config.random_seed + spec_index + horizon,
                            iterations=config.random_iterations,
                        )
                        concentration = _concentration(test_filter)
                        test_rate = (
                            float(test_filter["same_result"].mean())
                            if not test_filter.empty
                            else math.nan
                        )
                        test_median = (
                            float(test_filter["role_score"].median())
                            if not test_filter.empty
                            else math.nan
                        )
                        selected = {
                            **candidate,
                            **random_result,
                            **concentration,
                            "max_single_symbol_share": spec.max_single_symbol_share,
                            "max_single_session_share": spec.max_single_session_share,
                            "max_single_month_share": spec.max_single_month_share,
                            "base_test_count": int(len(test_all)),
                            "regime_test_count": int(len(test_regime)),
                            "retained_test_count": int(len(test_filter)),
                            "base_test_same_result_rate": base_test_rate,
                            "regime_test_same_result_rate": regime_test_rate,
                            "filtered_test_same_result_rate": test_rate,
                            "test_lift_vs_personality": test_rate - base_test_rate
                            if not math.isnan(test_rate)
                            else math.nan,
                            "test_lift_vs_regime": test_rate - regime_test_rate
                            if not math.isnan(test_rate) and not math.isnan(regime_test_rate)
                            else math.nan,
                            "base_test_median_score": base_test_median,
                            "regime_test_median_score": regime_test_median,
                            "filtered_test_median_score": test_median,
                            "test_median_lift_vs_personality": test_median - base_test_median
                            if not math.isnan(test_median)
                            else math.nan,
                            "test_median_lift_vs_regime": test_median - regime_test_median
                            if not math.isnan(test_median) and not math.isnan(regime_test_median)
                            else math.nan,
                        }
                        reasons: list[str] = []
                        if len(test_filter) < spec.test_count:
                            reasons.append("low_test_count")
                        if len(test_filter) < spec.retained_count:
                            reasons.append("low_retained_count")
                        if concentration["symbol_count"] < spec.symbol_count:
                            reasons.append("low_symbol_count")
                        if (
                            not math.isnan(float(concentration["single_symbol_share"]))
                            and concentration["single_symbol_share"] > spec.max_single_symbol_share
                        ):
                            reasons.append("single_symbol_dominated")
                        if (
                            not math.isnan(float(concentration["single_session_share"]))
                            and concentration["single_session_share"]
                            > spec.max_single_session_share
                        ):
                            reasons.append("single_session_dominated")
                        if (
                            not math.isnan(float(concentration["single_month_share"]))
                            and concentration["single_month_share"] > spec.max_single_month_share
                        ):
                            reasons.append("single_month_dominated")
                        if math.isnan(test_rate) or test_rate <= base_test_rate:
                            reasons.append("no_oos_lift_vs_personality")
                        if not math.isnan(regime_test_rate) and test_rate <= regime_test_rate:
                            reasons.append("no_oos_lift_vs_regime")
                        if (
                            not math.isnan(random_result["random_same_count_p95_rate"])
                            and test_rate <= random_result["random_same_count_p95_rate"]
                        ):
                            reasons.append("random_p95_not_beaten")
                        if math.isnan(test_median) or test_median <= base_test_median:
                            reasons.append("no_oos_median_lift_vs_personality")
                        selected["verdict"] = (
                            "pass_personality_discovery" if not reasons else "reject"
                        )
                        selected["reject_reasons"] = ";".join(reasons)
                        selected_rows.append(selected)
                        random_rows.append(
                            {
                                **candidate,
                                **random_result,
                                "retained_test_count": len(test_filter),
                            }
                        )
                        if not test_filter.empty:
                            for _, example in test_filter.head(12).iterrows():
                                examples.append(
                                    {
                                        "personality": spec.personality,
                                        "horizon": horizon,
                                        "regime_field": regime_field,
                                        "regime_value": regime_value,
                                        "filter_rule": filter_row["filter_rule"],
                                        "symbol": example.get("symbol"),
                                        "timestamp": example.get("timestamp"),
                                        "session_date": example.get("session_date"),
                                        "event_state": example.get("event_state"),
                                        "role_score": example.get("role_score"),
                                        "same_result": example.get("same_result"),
                                    }
                                )

    base = pd.DataFrame(base_rows)
    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    random_baseline = pd.DataFrame(random_rows)
    examples_df = pd.DataFrame(examples)
    passed = (
        selected[selected["verdict"].eq("pass_personality_discovery")].copy()
        if not selected.empty
        else pd.DataFrame()
    )
    rejected = (
        selected[~selected["verdict"].eq("pass_personality_discovery")].copy()
        if not selected.empty
        else pd.DataFrame()
    )
    concentration_rows = []
    if not selected.empty:
        for _, row in selected.iterrows():
            warnings = []
            if row.get("single_symbol_share", 0) > row.get("max_single_symbol_share", 0.50):
                warnings.append("single_symbol_dominated")
            if row.get("single_session_share", 0) > 0.20:
                warnings.append("single_session_dominated")
            if row.get("single_month_share", 0) > 0.50:
                warnings.append("single_month_dominated")
            if warnings:
                concentration_rows.append({**row.to_dict(), "warnings": ";".join(warnings)})
    concentration_warnings = pd.DataFrame(concentration_rows)

    matrix_rows = []
    for spec in specs:
        spec_passed = (
            passed[passed["personality"].eq(spec.personality)]
            if not passed.empty
            else pd.DataFrame()
        )
        best = (
            spec_passed.sort_values(
                ["test_lift_vs_personality", "filtered_test_same_result_rate"],
                ascending=False,
            ).iloc[0]
            if not spec_passed.empty
            else None
        )
        matrix_rows.append(
            {
                "personality": spec.personality,
                "role": spec.role,
                "candidate": spec.candidate,
                "passed_rule_count": int(len(spec_passed)),
                "decision": "continue_research" if best is not None else "reject_no_oos_rule",
                "best_horizon": int(best["horizon"]) if best is not None else math.nan,
                "best_regime": (
                    f"{best['regime_field']}={best['regime_value']}"
                    if best is not None
                    else ""
                ),
                "best_filter": best["filter_rule"] if best is not None else "",
                "best_test_count": int(best["retained_test_count"]) if best is not None else 0,
                "best_same_result_rate": float(best["filtered_test_same_result_rate"])
                if best is not None
                else math.nan,
                "best_lift_vs_personality": float(best["test_lift_vs_personality"])
                if best is not None
                else math.nan,
                "best_random_p95": float(best["random_same_count_p95_rate"])
                if best is not None
                else math.nan,
            }
        )
    decision_matrix = pd.DataFrame(matrix_rows)
    decision = _decision(passed, specs)

    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "loaded_specs": run_dir / "loaded_personality_specs.csv",
        "base": run_dir / "personality_base_summary.csv",
        "candidates": run_dir / "candidate_personality_rules.csv",
        "selected": run_dir / "selected_personality_rules.csv",
        "passed": run_dir / "passed_personality_rules.csv",
        "rejected": run_dir / "rejected_personality_rules.csv",
        "random": run_dir / "random_personality_baseline.csv",
        "concentration": run_dir / "concentration_warnings.csv",
        "examples": run_dir / "personality_discovery_examples.csv",
        "matrix": run_dir / "personality_decision_matrix.csv",
    }

    for path, frame in [
        (paths["loaded_specs"], loaded_specs),
        (paths["base"], base),
        (paths["candidates"], candidates),
        (paths["selected"], selected),
        (paths["passed"], passed),
        (paths["rejected"], rejected),
        (paths["random"], random_baseline),
        (paths["concentration"], concentration_warnings),
        (paths["examples"], examples_df),
        (paths["matrix"], decision_matrix),
    ]:
        _write_csv(path, frame)

    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_dir": str(resolved_input),
        "spec_dir": str(spec_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "loaded_spec_count": int(len(specs)),
        "event_rows": int(len(events)),
        "base_rows": int(len(base)),
        "candidate_rule_rows": int(len(candidates)),
        "selected_rule_rows": int(len(selected)),
        "passed_rule_rows": int(len(passed)),
        "passed_personalities": sorted(passed["personality"].unique().tolist())
        if not passed.empty
        else [],
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)

    lines = [
        "# Personality Discovery V0",
        "",
        (
            "Research-only diagnostic. No broker, IG, live trading, paper trading, "
            "vendor fetching, or order placement touched. No edge is claimed."
        ),
        "",
        f"Input report: `{resolved_input}`",
        f"Spec directory: `{spec_dir}`",
        "",
        f"Decision: `{decision}`",
        "",
        "## Counts",
        "",
        f"- Loaded specs: `{len(specs)}`",
        f"- Event rows: `{len(events)}`",
        f"- Candidate rule rows: `{len(candidates)}`",
        f"- Train-selected rule rows: `{len(selected)}`",
        f"- Passed rule rows: `{len(passed)}`",
        "",
        "## Personality Decision Matrix",
        "",
        (
            "| personality | role | passed | best_h | best_regime | best_filter | "
            "best_n | same_result | lift_vs_personality | rand_p95 |"
        ),
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in decision_matrix.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["personality"]),
                    str(row["role"]),
                    str(row["passed_rule_count"]),
                    "" if pd.isna(row["best_horizon"]) else str(int(row["best_horizon"])),
                    str(row["best_regime"]),
                    str(row["best_filter"]),
                    str(row["best_test_count"]),
                    _format_pct(row["best_same_result_rate"]),
                    _format_pct(row["best_lift_vs_personality"]),
                    _format_pct(row["best_random_p95"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Plain Interpretation", ""])
    if passed.empty:
        lines.append(
            "No personality/regime/filter/caveat combination survived the "
            "OOS/random/concentration gates."
        )
    else:
        lines.append(
            "The YAML-driven discovery process found personality-specific regimes, "
            "filters, and caveats that survived the gates. Treat these as research "
            "hypotheses, not execution rules."
        )
    lines.extend(["", "## Files", ""])
    for file_name in [
        "summary.json",
        "decision.json",
        "loaded_personality_specs.csv",
        "personality_base_summary.csv",
        "candidate_personality_rules.csv",
        "selected_personality_rules.csv",
        "passed_personality_rules.csv",
        "rejected_personality_rules.csv",
        "random_personality_baseline.csv",
        "concentration_warnings.csv",
        "personality_discovery_examples.csv",
        "personality_decision_matrix.csv",
    ]:
        lines.append(f"- `{file_name}`")
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PersonalityDiscoveryResult(
        run_id=run_id,
        input_dir=resolved_input,
        spec_dir=spec_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        loaded_specs_csv_path=paths["loaded_specs"],
        personality_base_summary_csv_path=paths["base"],
        candidate_rules_csv_path=paths["candidates"],
        selected_rules_csv_path=paths["selected"],
        passed_rules_csv_path=paths["passed"],
        rejected_rules_csv_path=paths["rejected"],
        random_baseline_csv_path=paths["random"],
        concentration_warnings_csv_path=paths["concentration"],
        examples_csv_path=paths["examples"],
        decision_matrix_csv_path=paths["matrix"],
        decision=decision,
        passed_rule_count=int(len(passed)),
    )
