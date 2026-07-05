"""Clean-slate research-only template discovery system.

This module rebuilds template-discovery context from existing local event rows.
It does not read saved rules, prior discovered-container reports, broker paths,
vendor fetchers, paper/live trading paths, deployment state, or YAML registries.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stocker_research.personality_discovery_v0 import add_discovery_features
from stocker_research.personality_stop_validation_v0 import (
    _risk_bps_for_model,
    score_stop_model_events,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/template_discovery_system_v0")
DEFAULT_PERIODS: dict[str, tuple[str, str]] = {
    "fresh_year": ("2024-07", "2025-06"),
    "saved_year": ("2025-07", "2026-06"),
    "full_available": ("2024-01", "2026-06"),
}
SUPPORTED_MODES = {
    "container-routing",
    "nested-container-routing",
    "family-directional-readout",
    "r-replay",
    "family-r-replay",
    "frozen-combo-replay",
    "frozen-template-transfer-replay",
    "template-component-selection",
}
FROZEN_COMBO_COMPONENT_FILES = (
    "fixed_next_consistent_trades.csv",
    "omitted_saved_loop_consistent_trades.csv",
    "strict_exact_frozen_addon_consistent_trades.csv",
)
FROZEN_COMPONENT_PRIORITY = {
    "current_fixed_next": 0,
    "omitted_saved_loop": 1,
    "strict_other_loop_addon": 2,
}
DEFAULT_FAMILY_STOP_MODELS = (
    "fixed_25bps",
    "fixed_50bps",
    "fixed_75bps",
    "fixed_100bps",
    "structure_session_extreme_10bps",
    "structure_recent_extreme_10bps",
    "structure_opening_range_extreme_10bps",
)
DEFAULT_FAMILY_TARGET_R_MULTIPLES = (1.0, 1.5, 2.0)
FAMILY_LOOP1 = (
    "controlled_pullback_after_bullish_impulse"
    "__to__"
    "failed_bullish_impulse_recoil"
)
FAMILY_LOOP2 = (
    "failed_bounce_active_liquidation"
    "__to__"
    "failed_bullish_impulse_recoil"
)
EVENT_STATE_PRIORITY = {
    state: index
    for index, state in enumerate(
        (
            "controlled_pullback_after_bullish_impulse",
            "failed_bullish_impulse_recoil",
            "liquidation_failed_low_reclaim",
            "failed_bounce_active_liquidation",
            "failed_open_down_continuation",
            "slow_snapback_after_dip",
            "dead_chop_blocker",
        )
    )
}
LOOP_SOURCE_REGIME_FEATURES = (
    "source_event_quality_regime",
    "source_relative_volume_regime",
    "source_vwap_side_regime",
    "source_range_regime",
    "source_compression_regime",
    "source_efficiency_regime",
    "source_bar_index_bucket",
    "source_auction_current_location",
    "source_auction_opening_mid_location",
    "source_auction_session_open_location",
)
LOOP_SOURCE_MIXED_REGIME_FEATURES = (
    "source_compression_x_efficiency_regime",
    "source_volume_x_vwap_regime",
    "source_time_x_vwap_regime",
    "source_vwap_x_range_regime",
    "source_opening_mid_x_range_regime",
)
LOOP_TRANSITION_REGIME_FEATURES = (
    *LOOP_SOURCE_REGIME_FEATURES,
    *LOOP_SOURCE_MIXED_REGIME_FEATURES,
)
NEXT_START_CATEGORICAL_FEATURES = (
    "event_quality_regime",
    "volume_x_vwap_regime",
    "vwap_x_range_regime",
    "opening_mid_x_range_regime",
    "auction_current_location",
    "compression_x_efficiency_regime",
    "time_x_vwap_regime",
    "bar_index_bucket",
    "cross_stock_same_direction_bucket",
)
NEXT_START_NUMERIC_FEATURES = (
    "minutes_between_source_and_next_event",
    "event_quality_score",
    "relative_volume_at_bar_index",
    "close_location_value",
)
PRE_SNAPSHOT_CATEGORICAL_FEATURES = (
    "source_compression_x_efficiency_regime",
    "source_volume_x_vwap_regime",
    "source_time_x_vwap_regime",
    "source_vwap_x_range_regime",
    "source_opening_mid_x_range_regime",
    "source_bar_index_bucket",
    "source_event_quality_regime",
    "source_auction_current_location",
    "source_cross_stock_same_direction_bucket",
    "source_relative_volume_regime",
    "source_vwap_side_regime",
    "source_range_regime",
    "source_compression_regime",
    "source_efficiency_regime",
    "source_auction_opening_mid_location",
    "source_auction_session_open_location",
)
PRE_SNAPSHOT_NUMERIC_FEATURES = (
    "minutes_between_source_and_next_event",
    "source_event_confidence_score",
    "source_event_quality_score",
    "source_bar_index_in_session",
    "source_close_location_value",
    "source_upper_wick_pct_of_range",
    "source_lower_wick_pct_of_range",
    "source_bar_return",
    "source_prior_3_bar_return",
    "source_prior_6_bar_return",
    "source_prior_12_bar_return",
    "source_directional_efficiency_3",
    "source_directional_efficiency_6",
    "source_directional_efficiency_12",
    "source_distance_from_vwap_pct",
    "source_abs_distance_from_vwap_pct",
    "source_distance_from_opening_range_mid_pct",
    "source_distance_from_opening_range_high_pct",
    "source_distance_from_opening_range_low_pct",
    "source_distance_from_session_open_pct",
    "source_distance_from_session_high_pct",
    "source_distance_from_session_low_pct",
    "source_distance_from_recent_high_pct",
    "source_distance_from_recent_low_pct",
    "source_vwap_cross_count_12",
    "source_range_cross_count_12",
    "source_rolling_intraday_range_pct",
    "source_compression_zscore",
    "source_range_zscore",
    "source_return_zscore",
    "source_relative_volume_at_bar_index",
    "source_relative_cumulative_volume",
    "source_pullback_depth_from_recent_high",
    "source_reclaim_from_recent_low",
)


@dataclass(frozen=True)
class TemplateDiscoveryEventInput:
    """One clean-slate event-row CSV input."""

    label: str
    event_rows_path: Path


@dataclass(frozen=True)
class DiscoveryAtom:
    """Generated source-visible atom."""

    atom_id: str
    axis: str
    feature: str
    operator: str
    value: str | float
    expression: str
    visibility: str = "source_visible"
    source: str = "generated"


@dataclass(frozen=True)
class ContainerSpec:
    """Generated container or nested container specification."""

    container_id: str
    level: Literal["C0", "C1", "regime_anchor"]
    expression: str
    atoms: tuple[str, ...]
    parent_id: str | None = None
    source: str = "generated"


@dataclass(frozen=True)
class LoopSpec:
    """Observed clean-slate loop transition."""

    source_event_state: str
    next_event_state: str
    loop_id: str


@dataclass(frozen=True)
class DirectionalReadout:
    """Directional inside/outside route readout."""

    candidate_id: str
    container_id: str
    loop_id: str
    directional_side: Literal["long", "short"]
    horizon: int
    stable_positive_route: bool
    stable_negative_route: bool
    mean_signed_lift: float | None
    mean_abs_lift: float | None


@dataclass(frozen=True)
class AdmissionCandidate:
    """Candidate-only admission finding, separate from blockers."""

    candidate_id: str
    container_id: str
    loop_id: str
    directional_side: str
    horizon: int


@dataclass(frozen=True)
class BlockerCandidate:
    """No-trade/blocker finding, separate from admissions."""

    candidate_id: str
    container_id: str
    loop_id: str
    directional_side: str
    horizon: int


@dataclass(frozen=True)
class ContextRefinementCandidate:
    """Loop-specific source/next-start refinement, separate from containers."""

    candidate_id: str
    loop_id: str
    visibility: Literal["source_visible", "next_event_start"]
    expression: str
    directional_side: Literal["long", "short"]
    horizon: int
    candidate_kind: Literal["admission_refinement", "blocker_refinement", "diagnostic"]


@dataclass(frozen=True)
class ReplaySpec:
    """Research-only R replay configuration for one supported route."""

    candidate_id: str
    container_id: str
    loop_id: str
    directional_side: Literal["long", "short"]
    horizon: int
    stop_model: str
    target_r: float
    cost_bps: float


@dataclass(frozen=True)
class ReplayReadout:
    """Research-only R replay output row."""

    candidate_id: str
    total_net_r: float
    mean_r: float
    median_r: float
    win_rate: float


@dataclass(frozen=True)
class FrozenTemplateTransferSpec:
    """Recovered research template to replay from event rows."""

    component: str
    rule_id: str
    row_source: Literal["source_context", "transitions", "events"]
    loop_id: str | None
    event_states: tuple[str, ...]
    expression: str
    expected_direction: int
    horizon: int
    stop_model: str
    target_r: float
    cost_bps: float
    exact: bool = True


FROZEN_TEMPLATE_TRANSFER_SPECS = (
    FrozenTemplateTransferSpec(
        component="current_fixed_next",
        rule_id="fixed_next_confirmation_choppy_open_down_source_h6",
        row_source="source_context",
        loop_id=(
            "failed_open_down_continuation__to__"
            "failed_bounce_active_liquidation"
        ),
        event_states=(),
        expression=(
            "NOT (source_compression_zscore <= -0.408248) "
            "AND NOT (b0_raw_state == strong_high_vol_broad_tape)"
        ),
        expected_direction=-1,
        horizon=6,
        stop_model="fixed_75bps",
        target_r=2.0,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="current_fixed_next",
        rule_id="fixed_next_confirmation_fast_failed_reclaim_short_source_h6",
        row_source="source_context",
        loop_id=(
            "failed_open_down_continuation__to__"
            "liquidation_failed_low_reclaim"
        ),
        event_states=(),
        expression="NOT (broad_failed_recoil_event_share_prior <= 0.484635)",
        expected_direction=-1,
        horizon=6,
        stop_model="fixed_100bps",
        target_r=2.0,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="omitted_saved_loop",
        rule_id="extended_directional_impulse_pullback_confirmation_note_v0",
        row_source="transitions",
        loop_id=(
            "failed_bullish_impulse_recoil__to__"
            "controlled_pullback_after_bullish_impulse"
        ),
        event_states=(),
        expression=(
            "source_compression_x_efficiency_regime == expanded|directional_efficiency "
            "AND source_distance_from_vwap_pct >= 0.0172204 "
            "AND source_bar_index_in_session <= 50 "
            "AND b0_stress_score_raw <= 0.477977 "
            "AND NOT (target_time_x_vwap_regime == late_day|above)"
        ),
        expected_direction=1,
        horizon=24,
        stop_model="fixed_150bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="omitted_saved_loop",
        rule_id="liquidation_reclaim_failed_bounce_confirmation_note_v0",
        row_source="transitions",
        loop_id="liquidation_failed_low_reclaim__to__failed_bounce_active_liquidation",
        event_states=(),
        expression=(
            "source_event_quality_score >= 0.688316 "
            "AND source_range_zscore <= -0.975675 "
            "AND NOT (source_opening_mid_x_range_regime == below|mid_range) "
            "AND broad_breadth_20d_up_prior <= 0.227273 "
            "AND target_opening_mid_x_range_regime == below|high_range"
        ),
        expected_direction=-1,
        horizon=24,
        stop_model="fixed_100bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="omitted_saved_loop",
        rule_id="low_volume_reclaim_repair_behavior_loop_split_v0",
        row_source="transitions",
        loop_id="liquidation_failed_low_reclaim__to__slow_snapback_after_dip",
        event_states=(),
        expression=(
            "source_volume_x_vwap_regime == low_relative_volume|above "
            "AND vwap_x_range_regime == above|high_range "
            "AND NOT (source_compression_regime == expanded) "
            "AND broad_median_ret_20d_prior <= -0.15482 "
            "AND NOT (target_compression_x_efficiency_regime == expanded|mixed_efficiency)"
        ),
        expected_direction=1,
        horizon=12,
        stop_model="fixed_75bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="strict_other_loop_addon",
        rule_id=(
            "controlled_pullback_after_bullish_impulse__to__"
            "failed_bounce_active_liquidation"
        ),
        row_source="transitions",
        loop_id=(
            "controlled_pullback_after_bullish_impulse__to__"
            "failed_bounce_active_liquidation"
        ),
        event_states=(),
        expression="relative_volume_at_bar_index <= 0.666793",
        expected_direction=-1,
        horizon=12,
        stop_model="fixed_75bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="strict_other_loop_addon",
        rule_id="failed_bullish_impulse_recoil__to__failed_bullish_impulse_recoil",
        row_source="transitions",
        loop_id="failed_bullish_impulse_recoil__to__failed_bullish_impulse_recoil",
        event_states=(),
        expression="distance_from_vwap_pct >= 0.04722",
        expected_direction=1,
        horizon=9,
        stop_model="fixed_100bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="strict_other_loop_addon",
        rule_id="liquidation_failed_low_reclaim__to__failed_open_down_continuation",
        row_source="transitions",
        loop_id="liquidation_failed_low_reclaim__to__failed_open_down_continuation",
        event_states=(),
        expression="relative_volume_at_bar_index <= 0.906369",
        expected_direction=1,
        horizon=12,
        stop_model="fixed_100bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="strict_other_loop_addon",
        rule_id="slow_snapback_after_dip__to__failed_open_down_continuation",
        row_source="transitions",
        loop_id="slow_snapback_after_dip__to__failed_open_down_continuation",
        event_states=(),
        expression="NOT bar_index_bucket == morning",
        expected_direction=-1,
        horizon=24,
        stop_model="fixed_75bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
    FrozenTemplateTransferSpec(
        component="strict_other_loop_addon",
        rule_id="failed_bullish_impulse_recoil__to__failed_open_down_continuation",
        row_source="transitions",
        loop_id="failed_bullish_impulse_recoil__to__failed_open_down_continuation",
        event_states=(),
        expression="NOT bar_index_bucket == morning",
        expected_direction=1,
        horizon=9,
        stop_model="fixed_125bps",
        target_r=2.5,
        cost_bps=10.0,
    ),
)


@dataclass(frozen=True)
class TemplateDiscoverySystemConfig:
    """Config for clean-slate template discovery."""

    mode: str = "container-routing"
    universe_profile: str = "liquid_midcap"
    periods: dict[str, tuple[str, str]] | None = None
    horizons: tuple[int, ...] = (6, 9, 12, 24)
    behavior_loop_discovery_period: str = "saved_year"
    min_behavior_loop_rows: int = 500
    min_behavior_loop_transition_rate: float = 0.20
    min_behavior_loop_symbols: int = 20
    min_behavior_loop_months: int = 10
    min_behavior_loop_split_rows: int = 100
    min_loop_regime_rows: int = 30
    c0_session_open_location: str = "below"
    c0_relative_cumulative_volume_max: float = 0.5631
    b0_smooth_window: int = 15
    b0_confirm_sessions: int = 5
    b0_min_hold_sessions: int = 15
    b0_weak_threshold: float = -0.12
    b0_strong_threshold: float = 0.12
    min_loop_refinement_rows: int = 80
    max_loop_refinement_terms_per_loop: int = 3
    state_change_only: bool = True
    min_atom_rows: int = 100
    max_atoms: int = 32
    min_container_rows: int = 120
    min_loop_inside_rows: int = 8
    min_loop_outside_rows: int = 8
    route_lift_bar: float = 0.05
    max_containers_to_route: int = 5
    max_single_symbol_share: float = 0.35
    stop_models: tuple[str, ...] = ("fixed_50bps",)
    target_r_multiples: tuple[float, ...] = (1.0,)
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0)
    frozen_combo_dir: Path | None = None
    frozen_component_paths: tuple[Path, ...] = ()
    frozen_candidate_book_path: Path | None = None
    component_candidate_dir: Path | None = None
    component_candidate_paths: tuple[Path, ...] = ()
    min_component_candidate_rows: int = 10
    min_component_total_r: float = 0.0
    max_component_negative_months: int = 8
    max_component_single_symbol_share: float = 0.35
    max_component_candidates_per_family: int = 20


@dataclass(frozen=True)
class TemplateDiscoverySystemResult:
    """Paths and headline result for one clean-slate run."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    behavior_loop_scorecard_csv_path: Path
    loop_regime_occupancy_csv_path: Path
    loop_mixed_regime_occupancy_csv_path: Path
    loop_transition_regime_occupancy_csv_path: Path
    c0_parent_readout_csv_path: Path
    b0_state_summary_csv_path: Path
    b0_route_detail_csv_path: Path
    loop_context_refinement_csv_path: Path
    loop_context_admissions_csv_path: Path
    loop_context_blockers_csv_path: Path
    atom_scorecard_csv_path: Path
    container_scorecard_csv_path: Path
    loop_routing_detail_csv_path: Path
    family_test_detail_csv_path: Path
    concentration_warnings_csv_path: Path
    admission_candidates_csv_path: Path
    blocker_candidates_csv_path: Path
    replay_results_csv_path: Path
    decision: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _frozen_component_paths(config: TemplateDiscoverySystemConfig) -> tuple[Path, ...]:
    if config.frozen_component_paths:
        return tuple(Path(path) for path in config.frozen_component_paths)
    if config.frozen_combo_dir is None:
        raise ValueError(
            "frozen-combo-replay requires --frozen-combo-dir or --frozen-component-rows."
        )
    combo_dir = Path(config.frozen_combo_dir)
    return tuple(combo_dir / filename for filename in FROZEN_COMBO_COMPONENT_FILES)


def _frozen_candidate_book_path(config: TemplateDiscoverySystemConfig) -> Path | None:
    if config.frozen_candidate_book_path is not None:
        return Path(config.frozen_candidate_book_path)
    if config.frozen_combo_dir is None:
        return None
    path = Path(config.frozen_combo_dir) / "frozen_candidate_book.csv"
    return path if path.exists() else None


def _load_frozen_component_rows(config: TemplateDiscoverySystemConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _frozen_component_paths(config):
        if not path.exists():
            raise FileNotFoundError(f"Frozen component rows not found: {path}")
        rows = pd.read_csv(path)
        rows["_frozen_source_file"] = path.name
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "component" not in combined:
        combined["component"] = "unknown"
    if "timestamp" not in combined or "symbol" not in combined:
        raise ValueError("Frozen component rows must include symbol and timestamp columns.")
    return combined


def _frozen_r_column(rows: pd.DataFrame) -> str:
    for column in (
        "net_r",
        "final_close_r_after_cost",
        "final_r_conservative",
        "target_capped_r_after_cost",
        "target_capped_r",
    ):
        if column in rows:
            return column
    raise ValueError("Frozen component rows must include a replay R column such as net_r.")


def _boolish(rows: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in rows:
            values = rows[column]
            if values.dtype == bool:
                return values.fillna(False).astype(bool)
            normalized = values.astype("string").fillna("").str.lower()
            return normalized.isin({"true", "1", "yes", "y"})
    return pd.Series(False, index=rows.index)


def _ensure_frozen_month(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "month" in out and out["month"].notna().any():
        out["month"] = out["month"].astype(str).str.slice(0, 7)
        return out
    out["month"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m")
    return out


def _dedupe_frozen_components(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if rows.empty:
        return rows.copy(), 0
    out = _ensure_frozen_month(rows)
    out["_component_priority"] = (
        out["component"].astype(str).map(FROZEN_COMPONENT_PRIORITY).fillna(99).astype(int)
    )
    out["_input_order"] = np.arange(len(out))
    out["_timestamp_key"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out = out.sort_values(
        ["symbol", "_timestamp_key", "_component_priority", "_input_order"],
        kind="mergesort",
    ).copy()
    deduped = out.drop_duplicates(["symbol", "_timestamp_key"], keep="first").copy()
    overlap_count = int(len(out) - len(deduped))
    deduped["timestamp"] = deduped["_timestamp_key"]
    return (
        deduped.drop(
            columns=["_component_priority", "_input_order", "_timestamp_key"],
            errors="ignore",
        ).reset_index(drop=True),
        overlap_count,
    )


def _summarize_frozen_rows(
    rows: pd.DataFrame,
    *,
    group_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_values = group_values or {}
    base: dict[str, Any] = dict(group_values)
    if rows.empty:
        return {
            **base,
            "rows": 0,
            "total_r": 0.0,
            "mean_r": None,
            "median_r": None,
            "win_rate": None,
            "positive_months": 0,
            "negative_months": 0,
            "worst_month_r": None,
            "best_month_r": None,
            "symbols": 0,
            "single_symbol_share": None,
            "stop_hit_rate": None,
            "target_hit_rate": None,
            "ambiguous_rate": None,
        }
    r_values = pd.to_numeric(rows[_frozen_r_column(rows)], errors="coerce").dropna()
    month_totals = (
        rows.loc[r_values.index].assign(_r=r_values).groupby("month", sort=True)["_r"].sum()
    )
    symbol_counts = rows.loc[r_values.index, "symbol"].astype(str).value_counts()
    return {
        **base,
        "rows": int(len(r_values)),
        "total_r": float(r_values.sum()),
        "mean_r": float(r_values.mean()),
        "median_r": float(r_values.median()),
        "win_rate": float((r_values > 0.0).mean()),
        "positive_months": int((month_totals > 0.0).sum()),
        "negative_months": int((month_totals < 0.0).sum()),
        "worst_month_r": float(month_totals.min()) if not month_totals.empty else None,
        "best_month_r": float(month_totals.max()) if not month_totals.empty else None,
        "symbols": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(r_values))
        if len(r_values)
        else None,
        "stop_hit_rate": float(_boolish(rows.loc[r_values.index], "stop_hit").mean()),
        "target_hit_rate": float(_boolish(rows.loc[r_values.index], "target_hit").mean()),
        "ambiguous_rate": float(
            _boolish(
                rows.loc[r_values.index],
                "ambiguous",
                "target_stop_order_ambiguous",
            ).mean()
        ),
    }


def _grouped_frozen_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for values, group in rows.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        records.append(
            _summarize_frozen_rows(
                group,
                group_values=dict(zip(group_cols, values, strict=False)),
            )
        )
    return pd.DataFrame(records)


def _frozen_monthly_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for month, group in rows.groupby("month", sort=True):
        records.append(_summarize_frozen_rows(group, group_values={"month": month}))
    return pd.DataFrame(records)


def _frozen_transfer_detector_id() -> str:
    return "b0_dirvol_s10_confirm5_hold12"


def _raw_frozen_transfer_b0_state(direction: float, stress: float) -> str:
    if pd.isna(direction):
        return "unknown"
    if direction <= -0.15:
        direction_state = "weak"
    elif direction >= 0.15:
        direction_state = "strong"
    else:
        direction_state = "neutral"
    if direction_state == "neutral" or pd.isna(stress):
        return f"{direction_state}_broad_tape"
    stress_state = "high_vol" if stress >= 0.50 else "normal_vol"
    return f"{direction_state}_{stress_state}_broad_tape"


def _frozen_transfer_market_context_columns() -> tuple[str, ...]:
    return (
        "detector_id",
        "detector_label",
        "b0_state",
        "b0_raw_state",
        "b0_direction_score",
        "b0_stress_score",
        "b0_direction_score_raw",
        "b0_stress_score_raw",
        "broad_median_ret_20d_prior",
        "broad_breadth_20d_up_prior",
        "broad_breadth_above_20d_ma_prior",
        "broad_median_drawdown_20d_prior",
        "broad_median_realized_vol_20d_prior",
        "broad_event_pressure_prior",
        "broad_failed_recoil_event_share_prior",
    )


def _empty_frozen_transfer_market_context(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "source_timestamp" in out:
        out["source_date"] = pd.to_datetime(
            out["source_timestamp"],
            utc=True,
        ).dt.normalize()
    elif "source_date" not in out:
        out["source_date"] = pd.NaT
    if "detector_id" not in out:
        out["detector_id"] = _frozen_transfer_detector_id()
    if "detector_label" not in out:
        out["detector_label"] = ""
    if "b0_state" not in out:
        out["b0_state"] = "unknown"
    if "b0_raw_state" not in out:
        out["b0_raw_state"] = "unknown"
    for column in _frozen_transfer_market_context_columns():
        if column not in out:
            out[column] = np.nan
    out["detector_id"] = out["detector_id"].fillna(_frozen_transfer_detector_id())
    out["detector_label"] = out["detector_label"].fillna("")
    out["b0_state"] = out["b0_state"].fillna("unknown").astype(str)
    out["b0_raw_state"] = out["b0_raw_state"].fillna("unknown").astype(str)
    return out


def _frozen_transfer_market_states(events: pd.DataFrame) -> pd.DataFrame:
    daily = _daily_symbol_rows_for_b0(events)
    market = _market_daily_for_b0(daily)
    if market.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for _, surface_rows in market.groupby("surface", sort=False):
        rows = surface_rows.sort_values("date").copy()
        rows["score_ret20"] = _rank_score(rows["broad_median_ret_20d_prior"])
        rows["score_breadth_up"] = _rank_score(rows["broad_breadth_20d_up_prior"])
        rows["score_above_ma"] = _rank_score(rows["broad_breadth_above_20d_ma_prior"])
        rows["score_drawdown"] = _rank_score(rows["broad_median_drawdown_20d_prior"])
        rows["score_vol"] = _rank_score(rows["broad_median_realized_vol_20d_prior"])
        rows["b0_direction_score_raw"] = rows[
            ["score_ret20", "score_breadth_up", "score_above_ma", "score_drawdown"]
        ].mean(axis=1)
        rows["b0_stress_score_raw"] = rows["score_vol"]
        rows["b0_direction_score"] = rows["b0_direction_score_raw"].rolling(
            10,
            min_periods=5,
        ).mean()
        rows["b0_stress_score"] = rows["b0_stress_score_raw"].rolling(
            10,
            min_periods=5,
        ).mean()
        rows["b0_raw_state"] = [
            _raw_frozen_transfer_b0_state(direction, stress)
            for direction, stress in zip(
                rows["b0_direction_score"],
                rows["b0_stress_score"],
                strict=False,
            )
        ]
        rows["b0_state"] = _confirm_b0_states(
            rows["b0_raw_state"].tolist(),
            confirm_sessions=5,
            min_hold_sessions=12,
        )
        rows["detector_id"] = _frozen_transfer_detector_id()
        rows["detector_label"] = (
            "direction + volatility, 10-session smooth, 5-confirm, 12-hold"
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _enrich_transitions_with_frozen_transfer_market_context(
    *,
    events: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    if transitions.empty:
        return _empty_frozen_transfer_market_context(transitions)

    out = transitions.copy()
    out["source_date"] = pd.to_datetime(out["source_timestamp"], utc=True).dt.normalize()
    market = _frozen_transfer_market_states(events)
    if market.empty:
        return _empty_frozen_transfer_market_context(out)

    keep = [
        "surface",
        "date",
        *_frozen_transfer_market_context_columns(),
    ]
    state = market[[column for column in keep if column in market]].copy()
    enriched = out.merge(
        state,
        left_on=["surface", "source_date"],
        right_on=["surface", "date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["date"], errors="ignore")
    return _empty_frozen_transfer_market_context(enriched)


def _frozen_transfer_feature_name(name: str, rows: pd.DataFrame) -> str | None:
    if name in rows:
        return name
    if name.startswith("target_"):
        alternate = name.removeprefix("target_")
        if alternate in rows:
            return alternate
    return None


def _frozen_transfer_term_mask(
    rows: pd.DataFrame,
    term: str,
    *,
    skip_missing: bool,
) -> tuple[pd.Series, list[str]]:
    term = term.strip()
    negated = False
    if term.startswith("NOT "):
        negated = True
        term = term[4:].strip()
    elif term.startswith("NOT(") and term.endswith(")"):
        negated = True
        term = term[3:].strip()
    if term.startswith("(") and term.endswith(")"):
        term = term[1:-1].strip()

    match = re.match(r"^([A-Za-z0-9_]+)\s*(<=|>=|==|!=|<|>)\s*(.+)$", term)
    if not match:
        raise ValueError(f"Unsupported frozen template expression term: {term!r}")
    feature, op, raw_value = match.groups()
    column = _frozen_transfer_feature_name(feature, rows)
    if column is None:
        missing_mask = pd.Series(bool(skip_missing), index=rows.index)
        return missing_mask, [feature]

    value = raw_value.strip().strip("'\"")
    if op in {"==", "!="}:
        mask = rows[column].fillna("").astype(str).eq(value)
        if op == "!=":
            mask = ~mask
    else:
        numeric = pd.to_numeric(rows[column], errors="coerce")
        threshold = float(value)
        if op == "<=":
            mask = numeric <= threshold
        elif op == ">=":
            mask = numeric >= threshold
        elif op == "<":
            mask = numeric < threshold
        else:
            mask = numeric > threshold
        mask = mask.fillna(False)
    if negated:
        mask = ~mask
    return mask.fillna(False), []


def _frozen_transfer_expression_mask(
    rows: pd.DataFrame,
    expression: str,
    *,
    skip_missing: bool = False,
) -> tuple[pd.Series, list[str]]:
    if not expression or expression.startswith("APPROX:"):
        return pd.Series(True, index=rows.index), []
    mask = pd.Series(True, index=rows.index)
    missing: list[str] = []
    for term in re.split(r"\s+AND\s+", expression):
        term_mask, term_missing = _frozen_transfer_term_mask(
            rows,
            term,
            skip_missing=skip_missing,
        )
        mask &= term_mask
        missing.extend(term_missing)
    return mask.fillna(False), sorted(set(missing))


def _frozen_transfer_period_label(month: str, config: TemplateDiscoverySystemConfig) -> str:
    for period, (start, end) in _periods(config).items():
        if start <= str(month) <= end:
            return period
    return "outside"


def _score_frozen_transfer_rows(
    rows: pd.DataFrame,
    spec: FrozenTemplateTransferSpec,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    forward_col = f"forward_{spec.horizon}_bar_return"
    if forward_col not in rows:
        return pd.DataFrame()
    available = rows.dropna(subset=[forward_col]).copy()
    if available.empty:
        return pd.DataFrame()

    risk = _risk_bps_for_model(
        available,
        model_name=spec.stop_model,
        expected_direction=spec.expected_direction,
        structure_buffer_bps=10.0,
        min_structure_stop_bps=5.0,
    )
    scored = score_stop_model_events(
        available,
        horizon=spec.horizon,
        expected_direction=spec.expected_direction,
        risk_bps=risk,
        target_r=spec.target_r,
    )
    risk_series = pd.to_numeric(scored["risk_bps"], errors="coerce")
    aligned = pd.to_numeric(scored["aligned_return_bps"], errors="coerce")
    stop_hit = pd.Series(scored["stop_hit"]).astype(bool)
    target_hit = pd.Series(scored["target_hit"]).astype(bool)
    raw_after_cost = (aligned - spec.cost_bps) / risk_series
    stop_after_cost = -1.0 - (spec.cost_bps / risk_series)
    final_close = raw_after_cost.where(~stop_hit, stop_after_cost)
    target_capped = raw_after_cost.clip(upper=spec.target_r).where(
        ~target_hit,
        spec.target_r,
    )
    target_capped = target_capped.where(~stop_hit, stop_after_cost)

    scored["component"] = spec.component
    scored["rule_id"] = spec.rule_id
    scored["expected_direction"] = spec.expected_direction
    scored["horizon"] = spec.horizon
    scored["stop_model"] = spec.stop_model
    scored["target_r"] = float(spec.target_r)
    scored["cost_bps"] = float(spec.cost_bps)
    scored["final_close_r_after_cost"] = final_close
    scored["target_capped_r_after_cost"] = target_capped
    scored["net_r"] = final_close
    scored["target_capped_r"] = target_capped
    scored["month"] = pd.to_datetime(scored["timestamp"], utc=True).dt.strftime("%Y-%m")
    scored["research_period"] = scored["month"].map(
        lambda month: _frozen_transfer_period_label(month, config)
    )
    return scored


def _rematerialize_frozen_transfer_source_events(
    *,
    selected_transitions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    if selected_transitions.empty:
        return pd.DataFrame()

    selected = selected_transitions.copy()
    selected["_source_ts_key"] = pd.to_datetime(
        selected["source_timestamp"],
        utc=True,
        errors="coerce",
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    event_rows = events.copy()
    event_rows["_source_ts_key"] = pd.to_datetime(
        event_rows["timestamp"],
        utc=True,
        errors="coerce",
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    event_rows["_source_event_state_key"] = (
        event_rows["event_state"].fillna("").astype(str)
    )

    context_columns = [
        "surface",
        "symbol",
        "_source_ts_key",
        "timestamp",
        "loop_id",
        "source_event_state",
        "target_event_state",
        "minutes_between_source_and_next_event",
        "event_rows_between_source_and_next_event",
        *_frozen_transfer_market_context_columns(),
    ]
    context = selected[
        [column for column in context_columns if column in selected.columns]
    ].copy()
    context["_source_event_state_key"] = context["source_event_state"].fillna("").astype(str)
    context = context.rename(
        columns={
            "timestamp": "fixed_next_confirmation_timestamp",
            "loop_id": "fixed_next_loop_id",
            "target_event_state": "fixed_next_target_event_state",
        }
    )
    join_keys = ["surface", "symbol", "_source_ts_key", "_source_event_state_key"]
    context = context.drop_duplicates(join_keys, keep="first")
    rematerialized = event_rows.merge(
        context,
        on=join_keys,
        how="inner",
        validate="many_to_one",
    )
    rematerialized["fixed_next_source_dt"] = rematerialized["_source_ts_key"]
    rematerialized = rematerialized.drop_duplicates(join_keys, keep="first")
    return rematerialized.drop(
        columns=["_source_ts_key", "_source_event_state_key"],
        errors="ignore",
    )


def _run_frozen_template_transfer_specs(
    *,
    events: pd.DataFrame,
    transitions: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
    skip_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for spec_order, spec in enumerate(FROZEN_TEMPLATE_TRANSFER_SPECS):
        source = events if spec.row_source == "events" else transitions
        rows = source.copy()
        if spec.loop_id is not None and "loop_id" in rows:
            rows = rows[rows["loop_id"].astype(str).eq(spec.loop_id)].copy()
        elif spec.loop_id is not None:
            rows = pd.DataFrame(index=rows.index)
        if spec.event_states:
            if "event_state" in rows:
                rows = rows[rows["event_state"].astype(str).isin(spec.event_states)].copy()
            else:
                rows = pd.DataFrame(index=rows.index)

        mask, missing = _frozen_transfer_expression_mask(
            rows,
            spec.expression,
            skip_missing=skip_missing,
        )
        selected = rows[mask].copy()
        score_source = (
            _rematerialize_frozen_transfer_source_events(
                selected_transitions=selected,
                events=events,
            )
            if spec.row_source == "source_context"
            else selected
        )
        scored = _score_frozen_transfer_rows(score_source, spec, config)
        if not scored.empty:
            scored["spec_order"] = spec_order
            scored["template_expression"] = spec.expression
            scored["template_exact"] = spec.exact
            scored["missing_features_dropped"] = (
                ",".join(missing) if skip_missing else ""
            )
            detail.append(scored)

        audit_rows.append(
            {
                "component": spec.component,
                "rule_id": spec.rule_id,
                "row_source": spec.row_source,
                "loop_id": spec.loop_id,
                "expression": spec.expression,
                "template_exact": spec.exact,
                "missing_features": ",".join(missing),
                "candidate_rows_before_score": int(len(selected)),
                "score_source_rows": int(len(score_source)),
                "scored_rows": int(len(scored)),
                "missing_policy": "drop_missing_terms"
                if skip_missing
                else "require_all_terms",
            }
        )

    all_detail = (
        pd.concat(detail, ignore_index=True, sort=False)
        if detail
        else pd.DataFrame()
    )
    return all_detail, pd.DataFrame(audit_rows)


def _write_frozen_transfer_summary_md(
    path: Path,
    payload: dict[str, Any],
    transfer_summary: pd.DataFrame,
    component_summary_before_dedupe: pd.DataFrame,
    component_summary_after_dedupe: pd.DataFrame,
    rule_summary: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame[[column for column in cols if column in frame]].head(limit)
        lines = [
            "| " + " | ".join(shown.columns.astype(str)) + " |",
            "| " + " | ".join("---" for _ in shown.columns) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in shown.columns) + " |")
        return "\n".join(lines)

    summary_cols = ["rows", "total_r", "mean_r", "win_rate", "negative_months"]
    component_cols = ["component", "rows", "total_r", "mean_r", "win_rate"]
    rule_cols = ["component", "rule_id", "rows", "total_r", "mean_r", "win_rate"]
    lines = [
        "# Frozen Template Transfer Replay",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Decision: `{payload['decision']}`",
        "- Research-only: true",
        "- Saved rules used: true",
        "- Edge claimed: false",
        "",
        "## Summary",
        "",
        table(transfer_summary, summary_cols),
        "",
        "## Components Before Dedupe",
        "",
        table(component_summary_before_dedupe, component_cols),
        "",
        "## Components After Dedupe",
        "",
        table(component_summary_after_dedupe, component_cols),
        "",
        "## Rules After Dedupe",
        "",
        table(rule_summary, rule_cols),
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in payload["reports"].values())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frozen_combo_summary_md(
    path: Path,
    payload: dict[str, Any],
    combo_summary: pd.DataFrame,
    component_summary_before_dedupe: pd.DataFrame,
    component_summary_after_dedupe: pd.DataFrame,
    rule_summary: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame[cols].head(limit)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
        return "\n".join(lines)

    summary_cols = [
        "rows",
        "total_r",
        "mean_r",
        "median_r",
        "win_rate",
        "positive_months",
        "negative_months",
        "worst_month_r",
        "best_month_r",
        "symbols",
        "single_symbol_share",
    ]
    component_cols = ["component", *summary_cols]
    rule_cols = ["component", "rule_id", *summary_cols]
    lines = [
        "# Template Discovery System V0",
        "",
        "Frozen-template combo replay baseline from explicit frozen component rows.",
        "",
        f"Decision: `{payload['decision']}`",
        f"Mode: `{payload['mode']}`",
        "",
        "## Safety",
        "",
        "- `research_only: true`",
        "- `live_ordering_enabled: false`",
        "- `order_placement: disabled`",
        "- `edge_claimed: false`",
        "",
        "## Clean-Slate Boundary",
        "",
        "- `clean_slate: false`",
        "- `saved_rules_used: true`",
        "- `seed_report_used: true`",
        "- `yaml_rules_saved: false`",
        "",
        "## Main Combo Summary",
        "",
        table(combo_summary, summary_cols),
        "",
        "## Component Summary Before Exact-Time Dedupe",
        "",
        table(component_summary_before_dedupe, component_cols),
        "",
        "## Component Summary After Exact-Time Dedupe",
        "",
        table(component_summary_after_dedupe, component_cols),
        "",
        "## Rule Summary",
        "",
        table(rule_summary, rule_cols),
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in payload["reports"].values())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _component_candidate_paths(config: TemplateDiscoverySystemConfig) -> tuple[Path, ...]:
    if config.component_candidate_paths:
        return tuple(Path(path) for path in config.component_candidate_paths)
    if config.component_candidate_dir is None:
        if config.frozen_combo_dir is not None:
            return _frozen_component_paths(config)
        raise ValueError(
            "template-component-selection requires --component-candidate-dir "
            "or --component-candidate-rows."
        )
    candidate_dir = Path(config.component_candidate_dir)
    preferred = tuple(
        candidate_dir / filename
        for filename in FROZEN_COMBO_COMPONENT_FILES
        if (candidate_dir / filename).exists()
    )
    if preferred:
        return preferred
    blocked_name_parts = (
        "summary",
        "scorecard",
        "candidate_book",
        "main_combo",
        "dedupe",
        "monthly",
    )
    candidates = [
        path
        for path in sorted(candidate_dir.glob("*.csv"))
        if not any(part in path.name for part in blocked_name_parts)
    ]
    if not candidates:
        raise FileNotFoundError(f"No component candidate CSV files found in {candidate_dir}")
    return tuple(candidates)


def _infer_component_from_path(path: Path) -> str:
    name = path.name
    if "fixed_next" in name:
        return "current_fixed_next"
    if "omitted_saved_loop" in name:
        return "omitted_saved_loop"
    if "strict" in name or "addon" in name:
        return "strict_other_loop_addon"
    return _slug(path.stem)


def _first_nonempty_series(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    out = pd.Series("", index=rows.index, dtype="object")
    for column in columns:
        if column not in rows:
            continue
        values = rows[column].fillna("").astype(str)
        out = out.mask(out.astype(str).eq(""), values)
    return out


def _load_component_candidate_rows(config: TemplateDiscoverySystemConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    next_order = 0
    for path in _component_candidate_paths(config):
        if not path.exists():
            raise FileNotFoundError(f"Component candidate rows not found: {path}")
        rows = pd.read_csv(path)
        if rows.empty:
            continue
        rows = rows.copy()
        if "symbol" not in rows or "timestamp" not in rows:
            raise ValueError(
                "Component candidate rows must include symbol and timestamp columns: "
                f"{path}"
            )
        if "component" not in rows:
            rows["component"] = _infer_component_from_path(path)
        else:
            component = rows["component"].fillna("").astype(str)
            rows["component"] = component.mask(
                component.eq(""),
                _infer_component_from_path(path),
            )
        if "rule_id" not in rows:
            rows["rule_id"] = _first_nonempty_series(
                rows,
                ("candidate_id", "sig", "candidate_loop", "loop_id", "condition"),
            )
        rule_id = rows["rule_id"].fillna("").astype(str)
        rows["rule_id"] = rule_id.mask(rule_id.eq(""), _slug(path.stem))
        for column in ("candidate_loop", "condition", "direction_name"):
            if column not in rows:
                rows[column] = ""
        rows["selection_id"] = _first_nonempty_series(
            rows,
            ("sig", "candidate_id", "rule_id"),
        )
        selection_id = rows["selection_id"].fillna("").astype(str)
        rows["selection_id"] = selection_id.mask(selection_id.eq(""), rows["rule_id"])
        rows["candidate_source_file"] = path.name
        rows["candidate_source_path"] = str(path)
        rows["candidate_input_order"] = np.arange(next_order, next_order + len(rows))
        next_order += len(rows)
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    _frozen_r_column(combined)
    return _ensure_frozen_month(combined)


def _first_group_value(rows: pd.DataFrame, column: str) -> Any:
    if column not in rows:
        return None
    values = rows[column].dropna()
    if values.empty:
        return None
    for value in values.astype(str):
        if value:
            return value
    return values.iloc[0]


def _component_selection_gate_reasons(
    summary: dict[str, Any],
    config: TemplateDiscoverySystemConfig,
) -> list[str]:
    reasons: list[str] = []
    if int(summary["rows"]) < config.min_component_candidate_rows:
        reasons.append("rows_below_min")
    if float(summary["total_r"]) < config.min_component_total_r:
        reasons.append("total_r_below_min")
    if int(summary["negative_months"]) > config.max_component_negative_months:
        reasons.append("negative_months_above_max")
    single_symbol_share = summary.get("single_symbol_share")
    if (
        single_symbol_share is not None
        and not pd.isna(single_symbol_share)
        and float(single_symbol_share) > config.max_component_single_symbol_share
    ):
        reasons.append("single_symbol_share_above_max")
    return reasons


def _score_component_candidates(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    group_cols = ["component", "selection_id"]
    for values, group in rows.groupby(group_cols, dropna=False, sort=False):
        component, selection_id = values
        summary = _summarize_frozen_rows(
            group,
            group_values={
                "component": component,
                "selection_id": selection_id,
            },
        )
        summary.update(
            {
                "rule_id": _first_group_value(group, "rule_id"),
                "candidate_loop": _first_group_value(group, "candidate_loop"),
                "condition": _first_group_value(group, "condition"),
                "direction_name": _first_group_value(group, "direction_name"),
                "expected_direction": _first_group_value(group, "expected_direction"),
                "horizon": _first_group_value(group, "horizon"),
                "stop_model": _first_group_value(group, "stop_model"),
                "target_r": _first_group_value(group, "target_r"),
                "cost_bps": _first_group_value(group, "cost_bps"),
                "candidate_source_files": ",".join(
                    sorted(set(group["candidate_source_file"].astype(str)))
                ),
            }
        )
        reasons = _component_selection_gate_reasons(summary, config)
        summary["selected"] = not reasons
        summary["selection_reason"] = (
            "passes_component_selection_gates" if not reasons else ""
        )
        summary["reject_reason"] = ";".join(reasons)
        records.append(summary)

    scorecard = pd.DataFrame(records)
    if scorecard.empty:
        return scorecard
    scorecard["_component_priority"] = (
        scorecard["component"].astype(str).map(FROZEN_COMPONENT_PRIORITY).fillna(99)
    )
    scorecard = scorecard.sort_values(
        ["_component_priority", "total_r", "rows"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    if config.max_component_candidates_per_family > 0:
        selected = scorecard["selected"].astype(bool)
        selected_ranks = (
            scorecard.loc[selected]
            .groupby("component", sort=False)
            .cumcount()
            .add(1)
        )
        scorecard.loc[selected, "component_selection_rank"] = selected_ranks
        over_rank = (
            scorecard["component_selection_rank"].fillna(0).astype(int)
            > config.max_component_candidates_per_family
        )
        scorecard.loc[over_rank, "selected"] = False
        scorecard.loc[over_rank, "reject_reason"] = scorecard.loc[
            over_rank,
            "reject_reason",
        ].mask(
            scorecard.loc[over_rank, "reject_reason"].astype(str).eq(""),
            "component_candidate_rank_above_max",
        )
    scorecard["component_selection_rank"] = scorecard[
        "component_selection_rank"
    ].fillna(0).astype(int)
    return scorecard.drop(columns=["_component_priority"], errors="ignore")


def _selected_component_rows(
    rows: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty or scorecard.empty:
        return pd.DataFrame(columns=rows.columns)
    keys = scorecard.loc[
        scorecard["selected"].astype(bool),
        ["component", "selection_id"],
    ]
    if keys.empty:
        return pd.DataFrame(columns=rows.columns)
    selected = rows.merge(keys, on=["component", "selection_id"], how="inner")
    return selected.sort_values("candidate_input_order", kind="mergesort").reset_index(
        drop=True
    )


def _component_selection_book(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    columns = [
        "component",
        "component_selection_rank",
        "selection_id",
        "rule_id",
        "candidate_loop",
        "condition",
        "direction_name",
        "expected_direction",
        "horizon",
        "stop_model",
        "target_r",
        "cost_bps",
        "rows",
        "total_r",
        "mean_r",
        "median_r",
        "win_rate",
        "positive_months",
        "negative_months",
        "worst_month_r",
        "best_month_r",
        "symbols",
        "single_symbol_share",
        "candidate_source_files",
        "selection_reason",
    ]
    selected = scorecard[scorecard["selected"].astype(bool)].copy()
    return selected.reindex(columns=columns)


def _write_component_selection_summary_md(
    path: Path,
    payload: dict[str, Any],
    combo_summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    selected_book: pd.DataFrame,
    rule_summary: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame.reindex(columns=cols).head(limit)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
        return "\n".join(lines)

    summary_cols = [
        "rows",
        "total_r",
        "mean_r",
        "median_r",
        "win_rate",
        "positive_months",
        "negative_months",
        "symbols",
        "single_symbol_share",
    ]
    candidate_cols = [
        "component",
        "selection_id",
        "rule_id",
        "rows",
        "total_r",
        "negative_months",
        "single_symbol_share",
        "selected",
        "reject_reason",
    ]
    book_cols = [
        "component",
        "component_selection_rank",
        "rule_id",
        "rows",
        "total_r",
        "negative_months",
        "selection_reason",
    ]
    rule_cols = ["component", "rule_id", *summary_cols]
    lines = [
        "# Template Discovery System V0",
        "",
        "Automated component selection from research candidate row artifacts.",
        "",
        f"Decision: `{payload['decision']}`",
        f"Mode: `{payload['mode']}`",
        "",
        "## Safety",
        "",
        "- `research_only: true`",
        "- `live_ordering_enabled: false`",
        "- `order_placement: disabled`",
        "- `edge_claimed: false`",
        "",
        "## Clean-Slate Boundary",
        "",
        "- `clean_slate: false`",
        "- `seed_report_used: true`",
        "- `yaml_rules_saved: false`",
        "",
        "## Selected Combo Summary",
        "",
        table(combo_summary, summary_cols),
        "",
        "## Candidate Scorecard",
        "",
        table(scorecard, candidate_cols),
        "",
        "## Selected Candidate Book",
        "",
        table(selected_book, book_cols),
        "",
        "## Selected Rule Summary After Exact-Time Dedupe",
        "",
        table(rule_summary, rule_cols),
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in payload["reports"].values())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()[:120] or "item"


def _periods(config: TemplateDiscoverySystemConfig) -> dict[str, tuple[str, str]]:
    return config.periods or DEFAULT_PERIODS


def _period_mask(
    rows: pd.DataFrame,
    period: str,
    config: TemplateDiscoverySystemConfig,
) -> pd.Series:
    periods = _periods(config)
    start, end = periods[period]
    months = rows["month"].astype(str)
    return months.ge(start) & months.le(end)


def _month_period_mask(
    months: pd.Series,
    period: str,
    config: TemplateDiscoverySystemConfig,
) -> pd.Series:
    start, end = _periods(config)[period]
    month_values = months.astype(str)
    return month_values.ge(start) & month_values.le(end)


def _half_label(month: str) -> str:
    year, month_num = str(month).split("-", 1)
    return f"{year}-H1" if int(month_num) <= 6 else f"{year}-H2"


def _num(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series(np.nan, index=rows.index)
    return pd.to_numeric(rows[column], errors="coerce")


def _str(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series("", index=rows.index)
    return rows[column].fillna("").astype(str)


def _term_mask(rows: pd.DataFrame, atom: DiscoveryAtom) -> pd.Series:
    if atom.operator == "==":
        return _str(rows, atom.feature).eq(str(atom.value)).fillna(False)
    values = _num(rows, atom.feature)
    threshold = float(atom.value)
    if atom.operator == "<=":
        return values.le(threshold).fillna(False)
    if atom.operator == ">=":
        return values.ge(threshold).fillna(False)
    raise ValueError(f"Unsupported atom operator: {atom.operator}")


def _stats(
    rows: pd.DataFrame,
    *,
    horizon: int | None = None,
    side: str | None = None,
) -> dict[str, Any]:
    if rows.empty:
        return {
            "rows": 0,
            "win_rate": None,
            "median_return_bps": None,
            "mean_return_bps": None,
            "symbols": 0,
            "months": 0,
            "single_symbol_share": None,
            "single_month_share": None,
        }
    values: pd.Series
    if horizon is not None and side is not None:
        direction = 1.0 if side == "long" else -1.0
        values = pd.to_numeric(rows[f"forward_{horizon}_bar_return"], errors="coerce") * direction
    else:
        values = pd.to_numeric(
            rows.get("forward_6_bar_return", pd.Series(0, index=rows.index)),
            errors="coerce",
        )
    values = values.dropna()
    if values.empty:
        return {
            "rows": 0,
            "win_rate": None,
            "median_return_bps": None,
            "mean_return_bps": None,
            "symbols": 0,
            "months": 0,
            "single_symbol_share": None,
            "single_month_share": None,
        }
    symbols = rows.loc[values.index, "symbol"].astype(str)
    months = rows.loc[values.index, "month"].astype(str)
    symbol_counts = symbols.value_counts()
    month_counts = months.value_counts()
    return {
        "rows": int(len(values)),
        "win_rate": float((values > 0.0).mean()),
        "median_return_bps": float((values * 10000.0).median()),
        "mean_return_bps": float((values * 10000.0).mean()),
        "symbols": int(symbol_counts.size),
        "months": int(month_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(values)) if len(values) else None,
        "single_month_share": float(month_counts.iloc[0] / len(values)) if len(values) else None,
    }


def _load_event_rows(inputs: tuple[TemplateDiscoveryEventInput, ...]) -> pd.DataFrame:
    labelled_frames: list[tuple[str, pd.DataFrame]] = []
    for item in inputs:
        rows = pd.read_csv(item.event_rows_path)
        rows = add_discovery_features(rows)
        rows["surface"] = item.label
        rows["input_event_rows_path"] = str(item.event_rows_path)
        labelled_frames.append((item.label, rows))
    if not labelled_frames:
        return pd.DataFrame()

    symbols_by_surface = {
        label: set(rows["symbol"].astype(str))
        for label, rows in labelled_frames
        if "symbol" in rows.columns
    }

    frames: list[pd.DataFrame] = []
    for label, rows in labelled_frames:
        if label.startswith("residual_ex_") and "symbol" in rows.columns:
            excluded_surface = label.removeprefix("residual_ex_")
            excluded_symbols = symbols_by_surface.get(excluded_surface)
            if excluded_symbols:
                rows = rows.loc[
                    ~rows["symbol"].astype(str).isin(excluded_symbols)
                ].copy()
        frames.append(rows)

    return pd.concat(frames, ignore_index=True)


def _build_transitions(events: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for surface, surface_rows in events.groupby("surface", sort=False):
        rows = surface_rows.copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
        rows = rows.sort_values(
            ["symbol", "session_date", "timestamp", "event_state"]
        ).reset_index(drop=True)
        rows["_source_row_id"] = np.arange(len(rows))
        timestamp_heads = (
            rows[["symbol", "session_date", "timestamp"]]
            .drop_duplicates()
            .sort_values(["symbol", "session_date", "timestamp"])
            .reset_index(drop=True)
        )
        timestamp_heads["_next_timestamp"] = timestamp_heads.groupby(
            ["symbol", "session_date"], sort=False
        )["timestamp"].shift(-1)
        sources = rows.merge(
            timestamp_heads[["symbol", "session_date", "timestamp", "_next_timestamp"]],
            on=["symbol", "session_date", "timestamp"],
            how="left",
            validate="many_to_one",
        )
        sources = sources[sources["_next_timestamp"].notna()].copy()
        target_rows = rows.drop(columns=["_source_row_id"]).copy()
        target_rows["_target_priority"] = (
            target_rows["event_state"]
            .astype(str)
            .map(EVENT_STATE_PRIORITY)
            .fillna(999)
            .astype(int)
        )
        target_rows = (
            target_rows.sort_values(
                ["symbol", "session_date", "timestamp", "_target_priority", "event_state"]
            )
            .drop_duplicates(["symbol", "session_date", "timestamp"], keep="first")
            .drop(columns=["_target_priority"])
        )
        target_columns = list(dict.fromkeys(target_rows.columns.tolist()))
        targets = target_rows[target_columns].rename(
            columns={
                column: f"target_{column}"
                for column in target_columns
                if column not in {"symbol", "session_date"}
            }
        )
        merged = sources.merge(
            targets,
            left_on=["symbol", "session_date", "_next_timestamp"],
            right_on=["symbol", "session_date", "target_timestamp"],
            how="inner",
            validate="many_to_many",
        )
        merged = merged[
            merged["event_state"].astype(str).ne(merged["target_event_state"].astype(str))
        ].copy()
        if merged.empty:
            continue
        merged = merged.sort_values(
            ["symbol", "timestamp", "target_timestamp", "target_event_state"]
        ).copy()
        source_columns = [
            column
            for column in rows.columns
            if column
            not in {
                "_source_row_id",
                "symbol",
                "session_date",
                "surface",
                "input_event_rows_path",
            }
        ]
        source_prefixed = merged[source_columns].copy()
        source_prefixed.columns = [f"source_{column}" for column in source_columns]
        target_update_columns = [
            column for column in target_columns if f"target_{column}" in merged
        ]
        target_values = merged[[f"target_{column}" for column in target_update_columns]].copy()
        target_values.columns = target_update_columns
        merged = pd.concat(
            [
                merged.drop(
                    columns=[column for column in target_update_columns if column in merged]
                ),
                source_prefixed,
                target_values,
            ],
            axis=1,
        ).copy()
        derived = pd.DataFrame(index=merged.index)
        derived["surface"] = surface
        derived["source_timestamp"] = pd.to_datetime(merged["source_timestamp"], utc=True)
        derived["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        derived["minutes_between_source_and_next_event"] = (
            derived["timestamp"] - derived["source_timestamp"]
        ).dt.total_seconds() / 60.0
        derived["event_rows_between_source_and_next_event"] = 1
        derived["source_current_event_state"] = merged["source_event_state"].astype(str)
        derived["next_event_state"] = merged["event_state"].astype(str)
        derived["loop_id"] = (
            derived["source_current_event_state"] + "__to__" + derived["next_event_state"]
        )
        derived["month"] = derived["timestamp"].dt.strftime("%Y-%m")
        merged = pd.concat(
            [
                merged.drop(
                    columns=[
                        "surface",
                        "source_timestamp",
                        "timestamp",
                        "source_current_event_state",
                        "next_event_state",
                        "loop_id",
                        "month",
                    ],
                    errors="ignore",
                ),
                derived,
            ],
            axis=1,
        ).copy()
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["surface", "symbol", "timestamp", "loop_id"]
    ).reset_index(drop=True)


def _build_family_replay_transitions(
    events: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    """Build old family-replay transition rows without global target de-duplication."""

    frames: list[pd.DataFrame] = []
    loop_ids = {FAMILY_LOOP1, FAMILY_LOOP2}
    replay_horizons = tuple(sorted(set(config.horizons) | {6, 9, 12, 24}))
    target_feature_candidates = (
        "event_state",
        *NEXT_START_CATEGORICAL_FEATURES,
        *NEXT_START_NUMERIC_FEATURES,
    )
    for surface, surface_rows in events.groupby("surface", sort=False):
        rows = surface_rows.copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
        rows["session_date"] = rows["session_date"].astype(str)
        rows = rows.sort_values(["symbol", "session_date", "timestamp"]).reset_index(
            drop=True
        )
        rows["_source_row_id"] = rows.index
        timestamp_heads = (
            rows[["symbol", "session_date", "timestamp"]]
            .drop_duplicates()
            .sort_values(["symbol", "session_date", "timestamp"])
            .reset_index(drop=True)
        )
        timestamp_heads["_next_timestamp"] = timestamp_heads.groupby(
            ["symbol", "session_date"],
            sort=False,
        )["timestamp"].shift(-1)
        sources = rows.merge(
            timestamp_heads[["symbol", "session_date", "timestamp", "_next_timestamp"]],
            on=["symbol", "session_date", "timestamp"],
            how="left",
            validate="many_to_one",
        )
        sources = sources[sources["_next_timestamp"].notna()].copy()
        if sources.empty:
            continue
        target_keep = [
            "symbol",
            "session_date",
            "timestamp",
            *[feature for feature in target_feature_candidates if feature in rows],
            *[
                f"forward_{horizon}_bar_return"
                for horizon in replay_horizons
                if f"forward_{horizon}_bar_return" in rows
            ],
        ]
        target_keep = list(dict.fromkeys(target_keep))
        targets = rows[target_keep].rename(
            columns={
                column: f"target_{column}"
                for column in target_keep
                if column not in {"symbol", "session_date"}
            }
        )
        base = sources.merge(
            targets,
            left_on=["symbol", "session_date", "_next_timestamp"],
            right_on=["symbol", "session_date", "target_timestamp"],
            how="inner",
            validate="many_to_many",
        )
        if base.empty:
            continue
        base["next_timestamp"] = pd.to_datetime(base["target_timestamp"], utc=True)
        base["next_event_state"] = base["target_event_state"].astype(str)
        base = base[
            base["next_timestamp"].notna()
            & base["next_event_state"].notna()
            & base["event_state"].astype(str).ne(base["next_event_state"].astype(str))
        ].copy()
        base["source_current_event_state"] = base["event_state"].astype(str)
        base["loop_id"] = (
            base["source_current_event_state"] + "__to__" + base["next_event_state"]
        )
        base = base[base["loop_id"].isin(loop_ids)].copy()
        if base.empty:
            continue
        base["_target_priority"] = (
            base["next_event_state"]
            .map(EVENT_STATE_PRIORITY)
            .fillna(999)
            .astype(int)
        )
        base = (
            base.sort_values(["_source_row_id", "_target_priority", "next_event_state"])
            .drop_duplicates("_source_row_id", keep="first")
            .copy()
        )
        source_columns = [
            column
            for column in rows.columns
            if column
            not in {
                "_source_row_id",
                "symbol",
                "session_date",
                "surface",
                "input_event_rows_path",
            }
        ]
        source_prefixed = base[source_columns].copy()
        source_prefixed.columns = [f"source_{column}" for column in source_columns]
        base = pd.concat([base, source_prefixed], axis=1).copy()
        base["source_timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        base["timestamp"] = base["next_timestamp"]
        base["event_state"] = base["next_event_state"]
        for feature in target_feature_candidates:
            target_column = f"target_{feature}"
            if target_column in base:
                base[feature] = base[target_column].to_numpy()
        for horizon in replay_horizons:
            target_column = f"target_forward_{horizon}_bar_return"
            if target_column in base:
                base[f"forward_{horizon}_bar_return"] = base[target_column].to_numpy()
        base["surface"] = surface
        base["month"] = pd.to_datetime(base["timestamp"], utc=True).dt.strftime("%Y-%m")
        base["minutes_between_source_and_next_event"] = (
            pd.to_datetime(base["timestamp"], utc=True)
            - pd.to_datetime(base["source_timestamp"], utc=True)
        ).dt.total_seconds() / 60.0
        frames.append(base)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["surface", "symbol", "timestamp", "loop_id"]
    ).reset_index(drop=True)


def _behavior_loop_columns() -> list[str]:
    return [
        "loop_id",
        "current_event_state",
        "next_event_state",
        "loop_kind",
        "loop_rows",
        "current_event_raw_rows",
        "current_event_transition_rows",
        "current_event_no_next_later_share",
        "loop_rate_transition_denominator",
        "loop_rate_raw_denominator",
        "symbols",
        "months",
        "sessions",
        "single_symbol_share",
        "single_month_share",
        "single_session_share",
        "split_labels",
        "min_split_loop_rows",
        "candidate_behavior_loop",
    ]


def _primary_surface(rows: pd.DataFrame) -> str:
    surfaces = rows["surface"].dropna().astype(str)
    if surfaces.empty:
        return ""
    return "smid24" if "smid24" in set(surfaces) else str(surfaces.iloc[0])


def _discover_behavior_loops(
    events: pd.DataFrame,
    transitions: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(columns=_behavior_loop_columns())

    primary_surface = _primary_surface(transitions)
    if not primary_surface:
        return pd.DataFrame(columns=_behavior_loop_columns())

    period = config.behavior_loop_discovery_period
    if period not in _periods(config):
        raise ValueError(f"Unsupported behavior loop discovery period: {period}")

    primary = transitions[transitions["surface"].astype(str).eq(primary_surface)].copy()
    primary = primary[_period_mask(primary, period, config)].copy()
    if primary.empty:
        return pd.DataFrame(columns=_behavior_loop_columns())

    event_months = pd.to_datetime(events["timestamp"], utc=True).dt.strftime("%Y-%m")
    event_primary = events[
        events["surface"].astype(str).eq(primary_surface)
        & _month_period_mask(event_months, period, config)
    ].copy()
    event_primary["_month"] = event_months.loc[event_primary.index].to_numpy()

    transition_denominators = primary["source_current_event_state"].astype(str).value_counts()
    raw_denominators = event_primary["event_state"].astype(str).value_counts()
    records: list[dict[str, Any]] = []
    for loop_id, loop_rows in primary.groupby("loop_id", sort=False):
        if loop_rows.empty:
            continue
        source_state = str(loop_rows["source_current_event_state"].iloc[0])
        next_state = str(loop_rows["next_event_state"].iloc[0])
        current_transition_rows = int(transition_denominators.get(source_state, 0))
        current_raw_rows = int(raw_denominators.get(source_state, 0))
        loop_count = int(len(loop_rows))
        transition_rate = (
            float(loop_count / current_transition_rows) if current_transition_rows else 0.0
        )
        raw_rate = float(loop_count / current_raw_rows) if current_raw_rows else 0.0
        no_next_later_share = (
            max(0.0, 1.0 - float(current_transition_rows / current_raw_rows))
            if current_raw_rows
            else None
        )
        symbols = loop_rows["symbol"].astype(str)
        months = loop_rows["month"].astype(str)
        sessions = loop_rows["session_date"].astype(str)
        symbol_counts = symbols.value_counts()
        month_counts = months.value_counts()
        session_counts = sessions.value_counts()
        split_counts = months.map(_half_label).value_counts()
        min_split_rows = int(split_counts.min()) if not split_counts.empty else 0
        candidate = (
            loop_count >= config.min_behavior_loop_rows
            and transition_rate >= config.min_behavior_loop_transition_rate
            and int(symbol_counts.size) >= config.min_behavior_loop_symbols
            and int(month_counts.size) >= config.min_behavior_loop_months
            and min_split_rows >= config.min_behavior_loop_split_rows
            and (not config.state_change_only or source_state != next_state)
        )
        records.append(
            {
                "loop_id": str(loop_id),
                "current_event_state": source_state,
                "next_event_state": next_state,
                "loop_kind": "state_change" if source_state != next_state else "same_state",
                "loop_rows": loop_count,
                "current_event_raw_rows": current_raw_rows,
                "current_event_transition_rows": current_transition_rows,
                "current_event_no_next_later_share": no_next_later_share,
                "loop_rate_transition_denominator": transition_rate,
                "loop_rate_raw_denominator": raw_rate,
                "symbols": int(symbol_counts.size),
                "months": int(month_counts.size),
                "sessions": int(session_counts.size),
                "single_symbol_share": float(symbol_counts.iloc[0] / loop_count),
                "single_month_share": float(month_counts.iloc[0] / loop_count),
                "single_session_share": float(session_counts.iloc[0] / loop_count),
                "split_labels": json.dumps(sorted(split_counts.index.astype(str).tolist())),
                "min_split_loop_rows": min_split_rows,
                "candidate_behavior_loop": bool(candidate),
            }
        )
    frame = pd.DataFrame(records, columns=_behavior_loop_columns())
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "candidate_behavior_loop",
            "loop_rows",
            "loop_rate_transition_denominator",
            "loop_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _loop_regime_report_columns() -> list[str]:
    return [
        "surface",
        "loop_id",
        "regime_kind",
        "regime_feature",
        "regime_value",
        "rows",
        "loop_rows",
        "surface_rows",
        "surface_regime_rows",
        "loop_share",
        "surface_share",
        "share_delta",
        "representation_ratio",
        "symbols",
        "months",
        "single_symbol_share",
        "single_month_share",
    ]


def _valid_text(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    return text.where(~text.str.lower().isin({"", "nan", "none", "<na>"}), "")


def _loop_regime_records(
    rows: pd.DataFrame,
    *,
    features: tuple[str, ...],
    regime_kind: str,
    config: TemplateDiscoverySystemConfig,
    transition: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame(columns=_loop_regime_report_columns())
    for surface, surface_rows in rows.groupby("surface", sort=False):
        surface_rows = surface_rows.copy()
        surface_count = int(len(surface_rows))
        if surface_count == 0:
            continue
        for feature in features:
            if transition:
                target_feature = feature.removeprefix("source_")
                if feature not in surface_rows or target_feature not in surface_rows:
                    continue
                source_values = _valid_text(surface_rows[feature])
                target_values = _valid_text(surface_rows[target_feature])
                values = source_values + " -> " + target_values
                feature_label = f"{feature}->{target_feature}"
                usable_mask = source_values.ne("") & target_values.ne("")
            else:
                if feature not in surface_rows:
                    continue
                values = _valid_text(surface_rows[feature])
                feature_label = feature
                usable_mask = values.ne("")
            usable = surface_rows[usable_mask].copy()
            if usable.empty:
                continue
            usable["_regime_value"] = values.loc[usable.index].to_numpy()
            surface_counts = usable["_regime_value"].value_counts()
            for loop_id, loop_rows in usable.groupby("loop_id", sort=False):
                loop_count = int(len(loop_rows))
                if loop_count == 0:
                    continue
                for value, group in loop_rows.groupby("_regime_value", sort=False):
                    row_count = int(len(group))
                    if row_count < config.min_loop_regime_rows:
                        continue
                    surface_value_count = int(surface_counts.get(value, 0))
                    loop_share = float(row_count / loop_count)
                    surface_share = float(surface_value_count / len(usable))
                    representation_ratio = (
                        float(loop_share / surface_share) if surface_share else None
                    )
                    symbol_counts = group["symbol"].astype(str).value_counts()
                    month_counts = group["month"].astype(str).value_counts()
                    records.append(
                        {
                            "surface": str(surface),
                            "loop_id": str(loop_id),
                            "regime_kind": regime_kind,
                            "regime_feature": feature_label,
                            "regime_value": str(value),
                            "rows": row_count,
                            "loop_rows": loop_count,
                            "surface_rows": int(len(usable)),
                            "surface_regime_rows": surface_value_count,
                            "loop_share": loop_share,
                            "surface_share": surface_share,
                            "share_delta": float(loop_share - surface_share),
                            "representation_ratio": representation_ratio,
                            "symbols": int(symbol_counts.size),
                            "months": int(month_counts.size),
                            "single_symbol_share": float(symbol_counts.iloc[0] / row_count)
                            if row_count
                            else None,
                            "single_month_share": float(month_counts.iloc[0] / row_count)
                            if row_count
                            else None,
                        }
                    )
    frame = pd.DataFrame(records, columns=_loop_regime_report_columns())
    if frame.empty:
        return frame
    frame["_abs_share_delta"] = pd.to_numeric(
        frame["share_delta"],
        errors="coerce",
    ).abs()
    frame = frame.sort_values(
        [
            "surface",
            "loop_id",
            "rows",
            "_abs_share_delta",
            "representation_ratio",
            "regime_feature",
            "regime_value",
        ],
        ascending=[True, True, False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_abs_share_delta"])
    return frame.reset_index(drop=True)


def _loop_regime_occupancy_reports(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe where discovered behaviour loops live across regime surfaces."""

    source_regimes = _loop_regime_records(
        rows,
        features=LOOP_SOURCE_REGIME_FEATURES,
        regime_kind="source_regime",
        config=config,
    )
    mixed_regimes = _loop_regime_records(
        rows,
        features=LOOP_SOURCE_MIXED_REGIME_FEATURES,
        regime_kind="source_mixed_regime",
        config=config,
    )
    transition_regimes = _loop_regime_records(
        rows,
        features=LOOP_TRANSITION_REGIME_FEATURES,
        regime_kind="transition_regime",
        config=config,
        transition=True,
    )
    return source_regimes, mixed_regimes, transition_regimes


def _fixed_c0_expression(config: TemplateDiscoverySystemConfig) -> str:
    threshold = f"{config.c0_relative_cumulative_volume_max:g}"
    return (
        "source_auction_session_open_location == "
        f"{config.c0_session_open_location} AND "
        f"source_relative_cumulative_volume <= {threshold}"
    )


def _fixed_c0_mask(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.Series:
    location = _str(rows, "source_auction_session_open_location").eq(
        config.c0_session_open_location
    )
    participation = _num(rows, "source_relative_cumulative_volume").le(
        config.c0_relative_cumulative_volume_max
    )
    return (location & participation).fillna(False)


def _fixed_c0_candidate(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> dict[str, Any]:
    return {
        "container_id": "fixed_c0:below_session_open_low_cum",
        "expression": _fixed_c0_expression(config),
        "level": "C0",
        "atom_ids": "[]",
        "mask": _fixed_c0_mask(rows, config),
        "source": "fixed_clean_slate_c0",
    }


def _summarize_named_route_candidates(
    rows: pd.DataFrame,
    candidates: list[dict[str, Any]],
    routes: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "container_id",
        "parent_id",
        "level",
        "expression",
        "b0_state",
        "rows",
        "parent_rows",
        "row_share_of_parent",
        "symbols",
        "months",
        "single_symbol_share",
        "stable_route_count",
        "stable_route_loop_count",
        "stable_positive_count",
        "stable_negative_count",
        "mean_stable_abs_lift",
    ]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        mask = candidate["mask"].reindex(rows.index).fillna(False)
        parent_mask = candidate.get("parent_mask", pd.Series(True, index=rows.index))
        parent_mask = parent_mask.reindex(rows.index).fillna(False)
        selected = rows[mask].copy()
        parent = rows[parent_mask].copy()
        group = (
            routes[routes["container_id"].astype(str).eq(str(candidate["container_id"]))]
            if not routes.empty and "container_id" in routes
            else pd.DataFrame()
        )
        stable = group[group["stable_abs_route"].fillna(False)].copy()
        positive = group[group["stable_positive_route"].fillna(False)].copy()
        negative = group[group["stable_negative_route"].fillna(False)].copy()
        symbol_counts = selected["symbol"].astype(str).value_counts()
        records.append(
            {
                "container_id": candidate["container_id"],
                "parent_id": candidate.get("parent_id", ""),
                "level": candidate.get("level", ""),
                "expression": candidate.get("expression", ""),
                "b0_state": candidate.get("b0_state", ""),
                "rows": int(len(selected)),
                "parent_rows": int(len(parent)),
                "row_share_of_parent": float(len(selected) / len(parent))
                if len(parent)
                else None,
                "symbols": int(selected["symbol"].nunique()) if not selected.empty else 0,
                "months": int(selected["month"].nunique()) if not selected.empty else 0,
                "single_symbol_share": float(symbol_counts.iloc[0] / len(selected))
                if len(selected)
                else None,
                "stable_route_count": int(len(stable)),
                "stable_route_loop_count": int(stable["loop_id"].nunique())
                if not stable.empty
                else 0,
                "stable_positive_count": int(len(positive)),
                "stable_negative_count": int(len(negative)),
                "mean_stable_abs_lift": float(stable["mean_abs_lift"].mean())
                if not stable.empty
                else None,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.sort_values(
        [
            "stable_route_count",
            "stable_route_loop_count",
            "mean_stable_abs_lift",
            "rows",
        ],
        ascending=[False, False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _fixed_c0_parent_readout(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    candidate = _fixed_c0_candidate(rows, config)
    readout = _evaluate_routes(rows, [candidate], config)
    if readout.empty:
        return readout
    readout.insert(0, "stage", "fixed_c0_parent")
    return readout


def _event_direction(events: pd.DataFrame) -> pd.Series:
    if "event_direction" in events:
        return _str(events, "event_direction")
    states = _str(events, "event_state").str.lower()
    direction = pd.Series("", index=events.index)
    direction.loc[states.str.contains("bull|reclaim|snapback", regex=True)] = "bullish"
    direction.loc[states.str.contains("failed|liquidation|recoil", regex=True)] = "bearish"
    return direction


def _daily_symbol_rows_for_b0(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = events.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["session_date"] = rows["session_date"].astype(str)
    for column in ("open", "high", "low", "close", "volume"):
        rows[column] = _num(rows, column)
    direction = _event_direction(rows)
    rows["_event_is_bullish"] = direction.eq("bullish").astype(float)
    rows["_event_is_bearish"] = direction.eq("bearish").astype(float)
    rows["_event_failed_recoil"] = (
        _str(rows, "event_state")
        .str.contains("failed|recoil|liquidation", case=False, regex=True)
        .astype(float)
    )
    grouped = rows.groupby(["surface", "symbol", "session_date"], sort=False)
    daily = grouped.agg(
        date=("timestamp", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        event_volume=("volume", "sum"),
        event_count=("event_state", "size"),
        bullish_event_share=("_event_is_bullish", "mean"),
        bearish_event_share=("_event_is_bearish", "mean"),
        failed_recoil_event_share=("_event_failed_recoil", "mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
    daily = daily.sort_values(["surface", "symbol", "date"]).reset_index(drop=True)
    grouped_symbol = daily.groupby(["surface", "symbol"], sort=False)
    daily["daily_return"] = grouped_symbol["close"].pct_change(fill_method=None)
    daily["ret_20d"] = grouped_symbol["close"].pct_change(20, fill_method=None)
    daily["ma_20d"] = grouped_symbol["close"].transform(
        lambda series: series.rolling(20, min_periods=10).mean()
    )
    daily["above_20d_ma"] = daily["close"].gt(daily["ma_20d"])
    daily["rolling_20d_high"] = grouped_symbol["close"].transform(
        lambda series: series.rolling(20, min_periods=10).max()
    )
    daily["drawdown_20d"] = daily["close"] / daily["rolling_20d_high"] - 1.0
    daily["realized_vol_20d"] = grouped_symbol["daily_return"].transform(
        lambda series: series.rolling(20, min_periods=10).std()
    )
    return daily


def _market_daily_for_b0(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    records: list[pd.DataFrame] = []
    for surface, surface_rows in daily.groupby("surface", sort=False):
        grouped = surface_rows.groupby("date", sort=True)
        market = grouped.agg(
            broad_symbol_count=("symbol", "nunique"),
            broad_median_ret_20d=("ret_20d", "median"),
            broad_breadth_20d_up=(
                "ret_20d",
                lambda series: float((pd.to_numeric(series, errors="coerce") > 0).mean()),
            ),
            broad_breadth_above_20d_ma=("above_20d_ma", "mean"),
            broad_median_drawdown_20d=("drawdown_20d", "median"),
            broad_median_realized_vol_20d=("realized_vol_20d", "median"),
            broad_bullish_event_share=("bullish_event_share", "mean"),
            broad_bearish_event_share=("bearish_event_share", "mean"),
            broad_failed_recoil_event_share=("failed_recoil_event_share", "mean"),
            broad_event_count=("event_count", "sum"),
        ).reset_index()
        market["surface"] = surface
        market = market.sort_values("date").reset_index(drop=True)
        market["broad_event_pressure"] = (
            market["broad_bullish_event_share"] - market["broad_bearish_event_share"]
        )
        for column in [
            "broad_median_ret_20d",
            "broad_breadth_20d_up",
            "broad_breadth_above_20d_ma",
            "broad_median_drawdown_20d",
            "broad_median_realized_vol_20d",
            "broad_event_pressure",
            "broad_failed_recoil_event_share",
        ]:
            market[column] = pd.to_numeric(market[column], errors="coerce")
            market[f"{column}_prior"] = market[column].shift(1)
        records.append(market)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _rank_score(series: pd.Series, *, higher_is_stronger: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    score = (values.rank(pct=True) - 0.5) * 2.0
    return score if higher_is_stronger else -score


def _raw_b0_state(direction: float, config: TemplateDiscoverySystemConfig) -> str:
    if pd.isna(direction):
        return "unknown"
    if direction <= config.b0_weak_threshold:
        return "weak_broad_tape"
    if direction >= config.b0_strong_threshold:
        return "strong_broad_tape"
    return "neutral_broad_tape"


def _confirm_b0_states(
    raw_states: list[str],
    *,
    confirm_sessions: int,
    min_hold_sessions: int,
) -> list[str]:
    current = "unknown"
    held = 0
    pending: str | None = None
    pending_count = 0
    output: list[str] = []
    for raw_state in raw_states:
        raw = raw_state or "unknown"
        if current == "unknown" and raw != "unknown":
            current = raw
            held = 1
            pending = None
            pending_count = 0
            output.append(current)
            continue
        if raw == current or raw == "unknown":
            held += 1
            pending = None
            pending_count = 0
            output.append(current)
            continue
        if held < min_hold_sessions:
            held += 1
            output.append(current)
            continue
        if pending == raw:
            pending_count += 1
        else:
            pending = raw
            pending_count = 1
        if pending_count >= confirm_sessions:
            current = raw
            held = 1
            pending = None
            pending_count = 0
        else:
            held += 1
        output.append(current)
    return output


def _b0_detector_id(config: TemplateDiscoverySystemConfig) -> str:
    return (
        f"b0_dir_s{config.b0_smooth_window}_confirm"
        f"{config.b0_confirm_sessions}_hold{config.b0_min_hold_sessions}"
    )


def _b0_market_states(
    events: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    daily = _daily_symbol_rows_for_b0(events)
    market = _market_daily_for_b0(daily)
    if market.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    detector_id = _b0_detector_id(config)
    for _, surface_rows in market.groupby("surface", sort=False):
        rows = surface_rows.sort_values("date").copy()
        rows["score_ret20"] = _rank_score(rows["broad_median_ret_20d_prior"])
        rows["score_breadth_up"] = _rank_score(rows["broad_breadth_20d_up_prior"])
        rows["score_above_ma"] = _rank_score(rows["broad_breadth_above_20d_ma_prior"])
        rows["score_drawdown"] = _rank_score(rows["broad_median_drawdown_20d_prior"])
        rows["b0_direction_score_raw"] = rows[
            ["score_ret20", "score_breadth_up", "score_above_ma", "score_drawdown"]
        ].mean(axis=1)
        rows["b0_direction_score"] = rows["b0_direction_score_raw"].rolling(
            config.b0_smooth_window,
            min_periods=max(2, config.b0_smooth_window // 2),
        ).mean()
        rows["b0_raw_state"] = [
            _raw_b0_state(direction, config)
            for direction in rows["b0_direction_score"].tolist()
        ]
        rows["b0_state"] = _confirm_b0_states(
            rows["b0_raw_state"].tolist(),
            confirm_sessions=config.b0_confirm_sessions,
            min_hold_sessions=config.b0_min_hold_sessions,
        )
        rows["detector_id"] = detector_id
        rows["detector_label"] = (
            "direction score, "
            f"{config.b0_smooth_window}-session smooth, "
            f"{config.b0_confirm_sessions}-confirm, "
            f"{config.b0_min_hold_sessions}-hold"
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_b0_to_transitions(
    rows: pd.DataFrame,
    market_states: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    out = rows.copy()
    out["source_date"] = pd.to_datetime(out["source_timestamp"], utc=True).dt.normalize()
    if market_states.empty:
        out["detector_id"] = _b0_detector_id(config)
        out["detector_label"] = ""
        out["b0_state"] = "unknown"
        out["b0_raw_state"] = "unknown"
        out["b0_direction_score"] = np.nan
        return out
    keep = [
        "surface",
        "date",
        "detector_id",
        "detector_label",
        "b0_state",
        "b0_raw_state",
        "b0_direction_score",
    ]
    state = market_states[keep].copy()
    joined = out.merge(
        state,
        left_on=["surface", "source_date"],
        right_on=["surface", "date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["date"], errors="ignore")
    joined["detector_id"] = joined["detector_id"].fillna(_b0_detector_id(config))
    joined["detector_label"] = joined["detector_label"].fillna("")
    joined["b0_state"] = joined["b0_state"].fillna("unknown").astype(str)
    joined["b0_raw_state"] = joined["b0_raw_state"].fillna("unknown").astype(str)
    return joined


def _b0_route_reports(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return rows.copy(), pd.DataFrame(), pd.DataFrame()
    market_states = _b0_market_states(events, config)
    enriched = _join_b0_to_transitions(rows, market_states, config)
    c0_mask = _fixed_c0_mask(enriched, config)
    candidates: list[dict[str, Any]] = []
    states = sorted(
        state
        for state in enriched["b0_state"].dropna().astype(str).unique()
        if state and state != "unknown"
    )
    for state in states:
        mask = c0_mask & enriched["b0_state"].astype(str).eq(state)
        if not mask.any():
            continue
        candidates.append(
            {
                "container_id": f"b0:{_slug(state)}__fixed_c0",
                "parent_id": "fixed_c0:below_session_open_low_cum",
                "level": "B0+C0",
                "expression": f"b0_state == {state} AND {_fixed_c0_expression(config)}",
                "atom_ids": "[]",
                "mask": mask.fillna(False),
                "parent_mask": c0_mask,
                "b0_state": state,
                "source": "event_row_derived_b0_overlay",
            }
        )
    detail = _evaluate_routes(enriched, candidates, config) if candidates else pd.DataFrame()
    if not detail.empty:
        detail.insert(0, "stage", "b0_c0_overlay")
    summary = _summarize_named_route_candidates(enriched, candidates, detail)
    return enriched, summary, detail


@dataclass(frozen=True)
class _RefinementTerm:
    visibility: Literal["source_visible", "next_event_start"]
    feature: str
    operator: str
    value_label: str
    threshold: float | None
    expression: str
    discovery_rows: int
    mask: pd.Series


def _loop_refinement_columns() -> list[str]:
    return [
        "candidate_id",
        "candidate_kind",
        "loop_id",
        "visibility",
        "feature",
        "operator",
        "value",
        "threshold",
        "expression",
        "directional_side",
        "horizon",
        "discovery_rows",
        "stable_positive_refinement",
        "stable_negative_refinement",
        "mean_signed_lift",
        "mean_abs_lift",
        "route_direction",
    ]


def _term_base_rows(
    loop_rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    primary_surface = _primary_surface(loop_rows)
    base = loop_rows[loop_rows["surface"].astype(str).eq(primary_surface)].copy()
    if config.behavior_loop_discovery_period in _periods(config) and not base.empty:
        base = base[_period_mask(base, config.behavior_loop_discovery_period, config)].copy()
    return base if not base.empty else loop_rows.copy()


def _generate_loop_refinement_terms(
    loop_rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> list[_RefinementTerm]:
    base = _term_base_rows(loop_rows, config)
    terms: list[_RefinementTerm] = []
    categorical_groups = (
        ("source_visible", PRE_SNAPSHOT_CATEGORICAL_FEATURES),
        ("next_event_start", NEXT_START_CATEGORICAL_FEATURES),
    )
    numeric_groups = (
        ("source_visible", PRE_SNAPSHOT_NUMERIC_FEATURES),
        ("next_event_start", NEXT_START_NUMERIC_FEATURES),
    )
    for visibility, features in categorical_groups:
        for feature in features:
            if feature not in loop_rows or feature not in base:
                continue
            counts = _str(base, feature).value_counts()
            for value, count in counts.items():
                value_text = str(value)
                if (
                    int(count) < config.min_loop_refinement_rows
                    or not value_text
                    or value_text.lower() in {"nan", "none", "<na>"}
                ):
                    continue
                mask = _str(loop_rows, feature).eq(value_text)
                terms.append(
                    _RefinementTerm(
                        visibility=visibility,  # type: ignore[arg-type]
                        feature=feature,
                        operator="==",
                        value_label=value_text,
                        threshold=None,
                        expression=f"{feature} == {value_text}",
                        discovery_rows=int(count),
                        mask=mask.fillna(False),
                    )
                )
    for visibility, features in numeric_groups:
        for feature in features:
            if feature not in loop_rows or feature not in base:
                continue
            values = _num(base, feature).dropna()
            if len(values) < config.min_loop_refinement_rows or values.nunique() < 4:
                continue
            seen: set[tuple[str, float]] = set()
            for quantile, threshold_value in values.quantile([0.25, 0.50, 0.75]).items():
                threshold = round(float(threshold_value), 12)
                for operator in ("<=", ">="):
                    key = (operator, threshold)
                    if key in seen:
                        continue
                    seen.add(key)
                    base_mask = (
                        _num(base, feature).le(threshold)
                        if operator == "<="
                        else _num(base, feature).ge(threshold)
                    )
                    discovery_rows = int(base_mask.fillna(False).sum())
                    if discovery_rows < config.min_loop_refinement_rows:
                        continue
                    mask = (
                        _num(loop_rows, feature).le(threshold)
                        if operator == "<="
                        else _num(loop_rows, feature).ge(threshold)
                    )
                    terms.append(
                        _RefinementTerm(
                            visibility=visibility,  # type: ignore[arg-type]
                            feature=feature,
                            operator=operator,
                            value_label=f"p{int(quantile * 100)}",
                            threshold=threshold,
                            expression=f"{feature} {operator} {threshold:g}",
                            discovery_rows=discovery_rows,
                            mask=mask.fillna(False),
                        )
                    )
    terms = sorted(
        terms,
        key=lambda term: (term.discovery_rows, term.visibility, term.feature, term.expression),
        reverse=True,
    )
    return terms[: config.max_loop_refinement_terms_per_loop]


def _evaluate_loop_context_refinement(
    rows: pd.DataFrame,
    candidate_loop_ids: set[str],
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if rows.empty or not candidate_loop_ids:
        return pd.DataFrame(columns=_loop_refinement_columns())
    period_specs = _period_specs(rows)
    records: list[dict[str, Any]] = []
    for loop_id in sorted(candidate_loop_ids):
        loop_rows = rows[rows["loop_id"].astype(str).eq(loop_id)].copy()
        if loop_rows.empty:
            continue
        for term in _generate_loop_refinement_terms(loop_rows, config):
            for side in ("long", "short"):
                for horizon in config.horizons:
                    if f"forward_{horizon}_bar_return" not in loop_rows:
                        continue
                    record: dict[str, Any] = {
                        "candidate_id": (
                            f"{loop_id}|{term.visibility}|{_slug(term.expression)}|"
                            f"{side}|h{horizon}"
                        ),
                        "loop_id": loop_id,
                        "visibility": term.visibility,
                        "feature": term.feature,
                        "operator": term.operator,
                        "value": term.value_label,
                        "threshold": term.threshold,
                        "expression": term.expression,
                        "directional_side": side,
                        "horizon": f"h{horizon}",
                        "discovery_rows": term.discovery_rows,
                    }
                    lifts: list[float] = []
                    eligible = True
                    for surface, period in period_specs:
                        period_rows = loop_rows.loc[
                            loop_rows["surface"].eq(surface)
                            & _period_mask(loop_rows, period, config)
                        ]
                        inside = period_rows.loc[
                            term.mask.reindex(period_rows.index).fillna(False)
                        ]
                        outside = period_rows.loc[
                            ~term.mask.reindex(period_rows.index).fillna(False)
                        ]
                        inside_stats = _stats(inside, horizon=horizon, side=side)
                        outside_stats = _stats(outside, horizon=horizon, side=side)
                        prefix = f"{surface}_{period}"
                        lift = (
                            inside_stats["win_rate"] - outside_stats["win_rate"]
                            if inside_stats["win_rate"] is not None
                            and outside_stats["win_rate"] is not None
                            else None
                        )
                        record[f"{prefix}_inside_rows"] = inside_stats["rows"]
                        record[f"{prefix}_inside_win_rate"] = inside_stats["win_rate"]
                        record[f"{prefix}_outside_rows"] = outside_stats["rows"]
                        record[f"{prefix}_outside_win_rate"] = outside_stats["win_rate"]
                        record[f"{prefix}_lift"] = lift
                        record[f"{prefix}_single_symbol_share"] = inside_stats[
                            "single_symbol_share"
                        ]
                        if (
                            inside_stats["rows"] < config.min_loop_inside_rows
                            or outside_stats["rows"] < config.min_loop_outside_rows
                        ):
                            eligible = False
                        if lift is not None and not pd.isna(lift):
                            lifts.append(float(lift))
                    positive = False
                    negative = False
                    if eligible and len(lifts) == len(period_specs):
                        first_two = lifts[:2] if len(lifts) >= 2 else lifts
                        transfer = lifts[2:] if len(lifts) > 2 else []
                        positive = all(
                            value >= config.route_lift_bar for value in first_two
                        ) and all(value >= -0.02 for value in transfer)
                        negative = all(
                            value <= -config.route_lift_bar for value in first_two
                        ) and all(value <= 0.02 for value in transfer)
                        record["mean_signed_lift"] = float(np.mean(lifts))
                        record["mean_abs_lift"] = float(
                            np.mean([abs(value) for value in lifts])
                        )
                    else:
                        record["mean_signed_lift"] = None
                        record["mean_abs_lift"] = None
                    record["stable_positive_refinement"] = bool(positive)
                    record["stable_negative_refinement"] = bool(negative)
                    record["route_direction"] = (
                        "positive" if positive else "negative" if negative else ""
                    )
                    record["candidate_kind"] = (
                        "admission_refinement"
                        if positive
                        else "blocker_refinement"
                        if negative
                        else "diagnostic"
                    )
                    records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=_loop_refinement_columns())
    return frame.sort_values(
        ["candidate_kind", "mean_abs_lift", "discovery_rows", "candidate_id"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _candidate_atom_features() -> dict[str, tuple[str, ...]]:
    return {
        "location": (
            "source_vwap_x_range_regime",
            "source_opening_mid_x_range_regime",
            "source_auction_current_location",
            "source_vwap_side_regime",
            "source_range_regime",
            "source_auction_session_open_location",
            "source_auction_opening_mid_location",
        ),
        "participation": (
            "source_volume_x_vwap_regime",
            "source_relative_volume_regime",
            "source_cross_stock_same_direction_bucket",
        ),
        "tempo": (
            "source_bar_index_bucket",
            "source_time_x_vwap_regime",
        ),
        "structure": (
            "source_compression_regime",
            "source_efficiency_regime",
            "source_compression_x_efficiency_regime",
            "source_event_quality_regime",
        ),
    }


def _numeric_atom_specs() -> tuple[tuple[str, str, str, str, float], ...]:
    return (
        ("location", "near_session_low", "source_distance_from_session_low_pct", "<=", 0.0016),
        ("location", "near_session_high", "source_distance_from_session_high_pct", "<=", 0.0016),
        ("location", "away_from_session_low", "source_distance_from_session_low_pct", ">=", 0.0115),
        (
            "location",
            "away_from_opening_range_low",
            "source_distance_from_opening_range_low_pct",
            ">=",
            0.0112,
        ),
        (
            "location",
            "near_opening_range_low",
            "source_distance_from_opening_range_low_pct",
            "<=",
            0.0022,
        ),
        (
            "location",
            "above_opening_mid",
            "source_distance_from_opening_range_mid_pct",
            ">=",
            0.0,
        ),
        (
            "location",
            "below_opening_mid",
            "source_distance_from_opening_range_mid_pct",
            "<=",
            0.0,
        ),
        (
            "participation",
            "low_cumulative_volume",
            "source_relative_cumulative_volume",
            "<=",
            0.5631,
        ),
        (
            "participation",
            "normal_or_low_cumulative_volume",
            "source_relative_cumulative_volume",
            "<=",
            0.6648,
        ),
        (
            "participation",
            "high_index_relative_volume",
            "source_relative_volume_at_bar_index",
            ">=",
            1.40,
        ),
        (
            "participation",
            "low_index_relative_volume",
            "source_relative_volume_at_bar_index",
            "<=",
            0.80,
        ),
        (
            "structure",
            "tight_intraday_range",
            "source_rolling_intraday_range_pct",
            "<=",
            0.0161,
        ),
        (
            "structure",
            "contained_intraday_range",
            "source_rolling_intraday_range_pct",
            "<=",
            0.0228,
        ),
        ("structure", "compressed_range_z", "source_range_zscore", "<=", -0.6904),
        ("structure", "very_compressed_range_z", "source_range_zscore", "<=", -0.8079),
        ("structure", "expanded_range_z", "source_range_zscore", ">=", 0.75),
        ("structure", "compressed_z", "source_compression_zscore", "<=", -0.75),
        (
            "structure",
            "choppy_efficiency6",
            "source_directional_efficiency_6",
            "<=",
            0.5895,
        ),
        (
            "structure",
            "low_efficiency12",
            "source_directional_efficiency_12",
            "<=",
            0.211,
        ),
        (
            "structure",
            "directional_efficiency12",
            "source_directional_efficiency_12",
            ">=",
            0.48,
        ),
        (
            "structure",
            "upper_wick_rejection",
            "source_upper_wick_pct_of_range",
            ">=",
            0.3333,
        ),
        (
            "structure",
            "lower_wick_rejection",
            "source_lower_wick_pct_of_range",
            ">=",
            0.3333,
        ),
        ("structure", "no_lower_wick", "source_lower_wick_pct_of_range", "<=", 0.0178),
        ("structure", "weak_close_location", "source_close_location_value", "<=", 0.10),
        ("structure", "strong_close_location", "source_close_location_value", ">=", 0.90),
        ("tempo", "early_source_bar", "source_bar_index_in_session", "<=", 8),
        ("tempo", "late_source_bar", "source_bar_index_in_session", ">=", 16),
    )


def _generate_atoms(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> tuple[list[DiscoveryAtom], pd.DataFrame]:
    atoms: list[DiscoveryAtom] = []
    primary = rows[rows["surface"].eq("smid24")].copy()
    if primary.empty:
        primary = rows.copy()
    for axis, features in _candidate_atom_features().items():
        for feature in features:
            if feature not in rows:
                continue
            counts = _str(primary, feature).value_counts()
            for value, count in counts.items():
                if not value or value.lower() in {"nan", "none", "<na>"}:
                    continue
                if float(count) / max(1, len(primary)) > 0.85:
                    continue
                if int(count) < config.min_atom_rows:
                    continue
                atom_id = f"{feature}=={value}"
                atoms.append(
                    DiscoveryAtom(
                        atom_id=atom_id,
                        axis=axis,
                        feature=feature,
                        operator="==",
                        value=str(value),
                        expression=f"{feature} == {value}",
                    )
                )
    for axis, label, feature, op, threshold in _numeric_atom_specs():
        if feature not in rows:
            continue
        values = _num(primary, feature)
        mask = values.le(threshold) if op == "<=" else values.ge(threshold)
        selected_count = int(mask.fillna(False).sum())
        if selected_count < config.min_atom_rows:
            continue
        if float(selected_count) / max(1, len(primary)) > 0.85:
            continue
        threshold_text = f"{threshold:g}"
        atom_id = f"{label}:{feature}{op}{threshold_text}"
        atoms.append(
            DiscoveryAtom(
                atom_id=atom_id,
                axis=axis,
                feature=feature,
                operator=op,
                value=round(float(threshold), 10),
                expression=f"{label} ({feature} {op} {threshold_text})",
            )
        )
    scored: list[dict[str, Any]] = []
    dedup: dict[str, DiscoveryAtom] = {}
    for atom in atoms:
        dedup.setdefault(atom.atom_id, atom)
    for atom in dedup.values():
        mask = _term_mask(rows, atom)
        primary_selected = primary.loc[_term_mask(primary, atom)]
        selected = rows.loc[mask]
        scored.append(
            {
                "atom": atom,
                "atom_id": atom.atom_id,
                "source": atom.source,
                "axis": atom.axis,
                "feature": atom.feature,
                "operator": atom.operator,
                "value": atom.value,
                "expression": atom.expression,
                "visibility": atom.visibility,
                "rows": int(len(selected)),
                "loops": int(selected["loop_id"].nunique()) if not selected.empty else 0,
                "surfaces": int(selected["surface"].nunique()) if not selected.empty else 0,
                "primary_rows": int(len(primary_selected)),
                "primary_loops": int(primary_selected["loop_id"].nunique())
                if not primary_selected.empty
                else 0,
            }
        )
    scored = sorted(
        scored,
        key=lambda item: (
            int(item["primary_loops"]),
            int(item["primary_rows"]),
            str(item["axis"]),
            str(item["atom_id"]),
        ),
        reverse=True,
    )
    kept: list[DiscoveryAtom] = []
    axis_counts: dict[str, int] = {}
    for item in scored:
        atom = item["atom"]
        if axis_counts.get(atom.axis, 0) >= 9:
            continue
        kept.append(atom)
        axis_counts[atom.axis] = axis_counts.get(atom.axis, 0) + 1
        if len(kept) >= config.max_atoms:
            break
    kept_ids = {atom.atom_id for atom in kept}
    scorecard = pd.DataFrame(
        [{key: value for key, value in item.items() if key != "atom"} for item in scored]
    )
    if not scorecard.empty:
        scorecard["selected_for_generation"] = scorecard["atom_id"].isin(kept_ids)
        scorecard = scorecard.sort_values(
            [
                "selected_for_generation",
                "primary_loops",
                "primary_rows",
                "axis",
                "atom_id",
            ],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return kept, scorecard


def _candidate_masks(
    rows: pd.DataFrame,
    atoms: list[DiscoveryAtom],
    config: TemplateDiscoverySystemConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    atom_by_id = {atom.atom_id: atom for atom in atoms}

    def discovery_priority(spec: ContainerSpec) -> float:
        axes = {atom_by_id[atom_id].axis for atom_id in spec.atoms if atom_id in atom_by_id}
        expression = spec.expression.lower()
        priority = 0.0
        if spec.level == "C1":
            priority += 10.0
        if "participation" in axes and "location" in axes:
            priority += 100.0
        if "participation" in axes and "tempo" in axes:
            priority += 90.0
        if "participation" in axes and "structure" in axes:
            priority += 70.0
        if axes == {"location"}:
            priority += 35.0
        if "cumulative_volume" in expression:
            priority += 80.0
        elif "low_relative_volume" in expression:
            priority += 35.0
        return priority

    def routing_family(expression: str, axes: str) -> str:
        text = expression.lower()
        if "normal_or_low_cumulative_volume" in text:
            return "normal_or_low_cumulative_volume"
        if "low_cumulative_volume" in text:
            return "low_cumulative_volume"
        if "low_index_relative_volume" in text:
            return "low_index_relative_volume"
        if "low_relative_volume" in text:
            return "low_relative_volume"
        if "high_index_relative_volume" in text:
            return "high_index_relative_volume"
        if "high_relative_volume" in text:
            return "high_relative_volume"
        if "participation" in axes:
            return f"participation:{axes}"
        return axes or "unclassified"

    def add_candidate(spec: ContainerSpec, mask: pd.Series) -> None:
        selected = rows.loc[mask]
        if len(selected) < config.min_container_rows:
            return
        symbol_share = (
            float(selected["symbol"].astype(str).value_counts(normalize=True).iloc[0])
            if not selected.empty
            else None
        )
        if symbol_share is not None and symbol_share > config.max_single_symbol_share:
            return
        axes = sorted(
            {atom_by_id[atom_id].axis for atom_id in spec.atoms if atom_id in atom_by_id}
        )
        record: dict[str, Any] = {
            "container_id": spec.container_id,
            "source": spec.source,
            "level": spec.level,
            "expression": spec.expression,
            "atom_ids": json.dumps(list(spec.atoms), sort_keys=True),
            "atom_count": len(spec.atoms),
            "axes": "+".join(axes),
            "parent_id": spec.parent_id,
            "rows": int(len(selected)),
            "loops": int(selected["loop_id"].nunique()) if not selected.empty else 0,
            "symbols": int(selected["symbol"].nunique()) if not selected.empty else 0,
            "months": int(selected["month"].nunique()) if not selected.empty else 0,
            "single_symbol_share": symbol_share,
        }
        for surface in sorted(rows["surface"].dropna().astype(str).unique()):
            surface_rows = selected[selected["surface"].eq(surface)]
            record[f"{surface}_rows"] = int(len(surface_rows))
            record[f"{surface}_loops"] = (
                int(surface_rows["loop_id"].nunique()) if not surface_rows.empty else 0
            )
        record["routing_shell_score"] = (
            record["loops"] * 5.0
            + min(float(record["rows"]), 5000.0) / 500.0
            + len([key for key in record if key.endswith("_rows") and record[key] > 0]) * 2.0
            - float(symbol_share or 1.0) * 3.0
        )
        record["routing_admission_priority"] = discovery_priority(spec)
        record["routing_family"] = routing_family(record["expression"], record["axes"])
        candidates.append({**record, "mask": mask.fillna(False), "spec": spec})
        shell_rows.append(record)

    atom_masks = {atom.atom_id: _term_mask(rows, atom) for atom in atoms}
    for atom in atoms:
        add_candidate(
            ContainerSpec(
                container_id=f"generated:{_slug(atom.atom_id)}",
                level="C0",
                expression=atom.expression,
                atoms=(atom.atom_id,),
            ),
            atom_masks[atom.atom_id],
        )
    for left, right in itertools.combinations(atoms, 2):
        if left.axis == right.axis:
            continue
        mask = atom_masks[left.atom_id] & atom_masks[right.atom_id]
        add_candidate(
            ContainerSpec(
                container_id=f"generated:{_slug(left.atom_id)}__and__{_slug(right.atom_id)}",
                level="C1",
                expression=f"{left.expression} AND {right.expression}",
                atoms=(left.atom_id, right.atom_id),
                parent_id=f"generated:{_slug(left.atom_id)}",
            ),
            mask,
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item["routing_admission_priority"]),
            float(item["routing_shell_score"]),
            int(item["loops"]),
            int(item["rows"]),
        ),
        reverse=True,
    )
    routed = []
    seen_families: set[str] = set()
    route_limit = max(1, config.max_containers_to_route)
    for item in ranked:
        family = str(item["routing_family"])
        if family in seen_families:
            continue
        routed.append(item)
        seen_families.add(family)
        if len(routed) >= route_limit:
            break
    if len(routed) < route_limit:
        selected_container_ids = {str(item["container_id"]) for item in routed}
        for item in ranked:
            if str(item["container_id"]) in selected_container_ids:
                continue
            routed.append(item)
            if len(routed) >= route_limit:
                break
    selected_ids = {str(item["container_id"]) for item in routed}
    shell_frame = pd.DataFrame(shell_rows)
    if not shell_frame.empty:
        shell_frame["selected_for_routing"] = shell_frame["container_id"].astype(str).isin(
            selected_ids
        )
        shell_frame = shell_frame.sort_values(
            [
                "selected_for_routing",
                "routing_admission_priority",
                "routing_shell_score",
                "loops",
                "rows",
            ],
            ascending=[False, False, False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return routed, shell_frame


def _period_specs(rows: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    surfaces = set(rows["surface"].dropna().astype(str))
    specs: list[tuple[str, str]] = []
    if "smid24" in surfaces:
        specs.extend([("smid24", "fresh_year"), ("smid24", "saved_year")])
    for surface in sorted(surfaces - {"smid24"}):
        specs.append((surface, "full_available"))
    if not specs:
        specs.append((str(rows["surface"].iloc[0]), "full_available"))
    return tuple(specs)


def _evaluate_routes(
    rows: pd.DataFrame,
    candidates: list[dict[str, Any]],
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    route_records: list[dict[str, Any]] = []
    period_specs = _period_specs(rows)
    for candidate in candidates:
        mask = candidate["mask"]
        parent_mask = candidate.get("parent_mask", pd.Series(True, index=rows.index))
        parent_mask = parent_mask.reindex(rows.index).fillna(False)
        if config.mode == "nested-container-routing" and candidate["level"] == "C1":
            atoms = json.loads(str(candidate["atom_ids"]))
            if atoms:
                parent_atom = atoms[0]
                matching_parent = next(
                    (
                        item["mask"]
                        for item in candidates
                        if json.loads(str(item["atom_ids"])) == [parent_atom]
                    ),
                    parent_mask,
                )
                parent_mask = matching_parent.reindex(rows.index).fillna(False)
        for loop_id, loop_rows in rows.groupby("loop_id", sort=False):
            if loop_rows.empty:
                continue
            for side in ("long", "short"):
                for horizon in config.horizons:
                    if f"forward_{horizon}_bar_return" not in loop_rows:
                        continue
                    record: dict[str, Any] = {
                        "candidate_id": (
                            f"{candidate['container_id']}|{loop_id}|{side}|h{horizon}"
                        ),
                        "container_id": candidate["container_id"],
                        "container_expression": candidate["expression"],
                        "level": candidate["level"],
                        "loop_id": loop_id,
                        "directional_side": side,
                        "horizon": f"h{horizon}",
                    }
                    lifts: list[float] = []
                    eligible = True
                    for surface, period in period_specs:
                        period_rows = loop_rows.loc[
                            loop_rows["surface"].eq(surface)
                            & _period_mask(loop_rows, period, config)
                        ]
                        inside = period_rows.loc[
                            mask.reindex(period_rows.index).fillna(False)
                        ]
                        parent_part = parent_mask.reindex(period_rows.index).fillna(False)
                        candidate_part = mask.reindex(period_rows.index).fillna(False)
                        outside_mask = parent_part & ~candidate_part
                        outside = period_rows.loc[outside_mask]
                        inside_stats = _stats(inside, horizon=horizon, side=side)
                        outside_stats = _stats(outside, horizon=horizon, side=side)
                        prefix = f"{surface}_{period}"
                        lift = (
                            inside_stats["win_rate"] - outside_stats["win_rate"]
                            if inside_stats["win_rate"] is not None
                            and outside_stats["win_rate"] is not None
                            else None
                        )
                        record[f"{prefix}_inside_rows"] = inside_stats["rows"]
                        record[f"{prefix}_inside_win_rate"] = inside_stats["win_rate"]
                        record[f"{prefix}_outside_rows"] = outside_stats["rows"]
                        record[f"{prefix}_outside_win_rate"] = outside_stats["win_rate"]
                        record[f"{prefix}_lift"] = lift
                        record[f"{prefix}_single_symbol_share"] = inside_stats[
                            "single_symbol_share"
                        ]
                        if (
                            inside_stats["rows"] < config.min_loop_inside_rows
                            or outside_stats["rows"] < config.min_loop_outside_rows
                        ):
                            eligible = False
                        if lift is not None and not pd.isna(lift):
                            lifts.append(float(lift))
                    if eligible and len(lifts) == len(period_specs):
                        first_two = lifts[:2] if len(lifts) >= 2 else lifts
                        transfer = lifts[2:] if len(lifts) > 2 else []
                        positive = all(
                            value >= config.route_lift_bar for value in first_two
                        ) and all(value >= -0.02 for value in transfer)
                        negative = all(
                            value <= -config.route_lift_bar for value in first_two
                        ) and all(value <= 0.02 for value in transfer)
                        consistent = (
                            all(value > 0 for value in first_two)
                            and all(value >= 0 for value in transfer)
                        ) or (
                            all(value < 0 for value in first_two)
                            and all(value <= 0 for value in transfer)
                        )
                        record["stable_positive_route"] = bool(positive)
                        record["stable_negative_route"] = bool(negative)
                        record["stable_abs_route"] = bool(positive or negative)
                        record["consistent_sign_route"] = bool(consistent)
                        record["route_direction"] = (
                            "positive" if positive else "negative" if negative else ""
                        )
                        record["mean_signed_lift"] = float(np.mean(lifts))
                        record["mean_abs_lift"] = float(np.mean([abs(value) for value in lifts]))
                    else:
                        record["stable_positive_route"] = False
                        record["stable_negative_route"] = False
                        record["stable_abs_route"] = False
                        record["consistent_sign_route"] = False
                        record["route_direction"] = ""
                        record["mean_signed_lift"] = None
                        record["mean_abs_lift"] = None
                    route_records.append(record)
    frame = pd.DataFrame(route_records)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["stable_abs_route", "mean_abs_lift", "candidate_id"],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _score_container_routes(containers: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    if containers.empty:
        return containers.copy()
    scored = containers.copy()
    defaults: dict[str, Any] = {
        "eligible_route_count": 0,
        "eligible_route_loop_count": 0,
        "stable_route_count": 0,
        "stable_route_loop_count": 0,
        "consistent_sign_route_count": 0,
        "positive_route_count": 0,
        "negative_route_count": 0,
        "stable_route_mean_abs_lift": 0.0,
        "eligible_route_mean_abs_lift": 0.0,
        "top_routed_loops": "[]",
    }
    for key, value in defaults.items():
        scored[key] = value

    if not routes.empty and "container_id" in routes.columns:
        records: list[dict[str, Any]] = []
        for container_id, group in routes.groupby("container_id", sort=False):
            eligible = group[group["mean_abs_lift"].notna()].copy()
            stable = group[group["stable_abs_route"].fillna(False)].copy()
            consistent = group[group["consistent_sign_route"].fillna(False)].copy()
            positive = group[group["stable_positive_route"].fillna(False)].copy()
            negative = group[group["stable_negative_route"].fillna(False)].copy()
            top_loops: list[dict[str, Any]] = []
            if not stable.empty:
                view = stable.sort_values(
                    ["mean_abs_lift", "loop_id", "directional_side", "horizon"],
                    ascending=[False, True, True, True],
                    na_position="last",
                ).head(5)
                for _, row in view.iterrows():
                    top_loops.append(
                        {
                            "loop_id": row.get("loop_id"),
                            "directional_side": row.get("directional_side"),
                            "horizon": row.get("horizon"),
                            "route_direction": row.get("route_direction"),
                            "mean_abs_lift": row.get("mean_abs_lift"),
                        }
                    )
            records.append(
                {
                    "container_id": container_id,
                    "eligible_route_count": int(len(eligible)),
                    "eligible_route_loop_count": int(eligible["loop_id"].nunique())
                    if not eligible.empty
                    else 0,
                    "stable_route_count": int(len(stable)),
                    "stable_route_loop_count": int(stable["loop_id"].nunique())
                    if not stable.empty
                    else 0,
                    "consistent_sign_route_count": int(len(consistent)),
                    "positive_route_count": int(len(positive)),
                    "negative_route_count": int(len(negative)),
                    "stable_route_mean_abs_lift": float(stable["mean_abs_lift"].mean())
                    if not stable.empty
                    else 0.0,
                    "eligible_route_mean_abs_lift": float(eligible["mean_abs_lift"].mean())
                    if not eligible.empty
                    else 0.0,
                    "top_routed_loops": json.dumps(top_loops, sort_keys=True),
                }
            )
        if records:
            route_frame = pd.DataFrame(records)
            scored = scored.drop(columns=list(defaults), errors="ignore").merge(
                route_frame,
                on="container_id",
                how="left",
            )
            for key, value in defaults.items():
                scored[key] = scored[key].fillna(value)

    scored["routing_score"] = (
        pd.to_numeric(scored["stable_route_count"], errors="coerce").fillna(0) * 10.0
        + pd.to_numeric(scored["stable_route_loop_count"], errors="coerce").fillna(0) * 5.0
        + pd.to_numeric(scored["consistent_sign_route_count"], errors="coerce").fillna(0)
        * 2.0
        + pd.to_numeric(scored["stable_route_mean_abs_lift"], errors="coerce").fillna(0.0)
        * 100.0
        + pd.to_numeric(scored["eligible_route_mean_abs_lift"], errors="coerce").fillna(0.0)
        * 20.0
        + np.minimum(pd.to_numeric(scored["loops"], errors="coerce").fillna(0), 8)
        - pd.to_numeric(scored["single_symbol_share"], errors="coerce").fillna(1.0) * 4.0
    )
    scored["recommended_container_candidate"] = (
        scored["selected_for_routing"].fillna(False).astype(bool)
        & (pd.to_numeric(scored["stable_route_loop_count"], errors="coerce").fillna(0) >= 2)
        & (pd.to_numeric(scored["eligible_route_loop_count"], errors="coerce").fillna(0) >= 3)
    )
    return scored.sort_values(
        [
            "recommended_container_candidate",
            "routing_score",
            "stable_route_count",
            "eligible_route_mean_abs_lift",
            "routing_shell_score",
            "rows",
        ],
        ascending=[False, False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _candidate_outputs(routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if routes.empty:
        columns = [
            "candidate_id",
            "candidate_kind",
            "container_id",
            "loop_id",
            "directional_side",
            "horizon",
            "route_direction",
            "mean_abs_lift",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), pd.DataFrame()
    stable = routes[routes["stable_abs_route"].fillna(False)].copy()
    admissions = stable[stable["stable_positive_route"].fillna(False)].copy()
    blockers = stable[stable["stable_negative_route"].fillna(False)].copy()
    keep = [
        "candidate_id",
        "container_id",
        "container_expression",
        "loop_id",
        "directional_side",
        "horizon",
        "route_direction",
        "mean_abs_lift",
        "mean_signed_lift",
    ]
    admissions = admissions[keep].copy() if not admissions.empty else pd.DataFrame(columns=keep)
    blockers = blockers[keep].copy() if not blockers.empty else pd.DataFrame(columns=keep)
    admissions["candidate_kind"] = "admission_candidate"
    blockers["candidate_kind"] = "blocker_candidate"
    warnings = []
    for _, row in stable.iterrows():
        share_cols = [col for col in stable.columns if col.endswith("_single_symbol_share")]
        shares = [
            float(row[col])
            for col in share_cols
            if row.get(col) is not None and not pd.isna(row[col])
        ]
        max_share = max(shares or [0.0])
        if max_share > 0.35:
            warnings.append(
                {
                    "candidate_id": row["candidate_id"],
                    "warning": "single_symbol_concentration",
                    "max_single_symbol_share": max_share,
                }
            )
    return admissions, blockers, pd.DataFrame(warnings)


def _apply_cost_to_r(scored: pd.DataFrame, *, cost_bps: float, target_r: float) -> pd.Series:
    risk = pd.to_numeric(scored["risk_bps"], errors="coerce")
    aligned = pd.to_numeric(scored["aligned_return_bps"], errors="coerce")
    stop_hit = pd.Series(scored["stop_hit"]).astype(bool)
    target_hit = pd.Series(scored["target_hit"]).astype(bool)
    raw_after_cost = (aligned - float(cost_bps)) / risk
    stop_after_cost = -1.0 - (float(cost_bps) / risk)
    target_capped = raw_after_cost.clip(upper=float(target_r)).where(~target_hit, float(target_r))
    return target_capped.where(~stop_hit, stop_after_cost)


def _run_replay(
    rows: pd.DataFrame,
    candidates: list[dict[str, Any]],
    admissions: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> pd.DataFrame:
    if admissions.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "surface",
                "period",
                "loop_id",
                "directional_side",
                "horizon",
                "stop_model",
                "target_r",
                "cost_bps",
                "rows",
                "total_net_r",
                "mean_r",
                "median_r",
                "win_rate",
                "positive_months",
                "months",
                "symbols",
                "single_symbol_share",
                "single_month_share",
            ]
        )
    mask_by_container = {candidate["container_id"]: candidate["mask"] for candidate in candidates}
    period_specs = _period_specs(rows)
    replay_rows: list[dict[str, Any]] = []
    for _, admission in admissions.iterrows():
        container_id = str(admission["container_id"])
        loop_id = str(admission["loop_id"])
        side = str(admission["directional_side"])
        horizon = int(str(admission["horizon"]).lstrip("h"))
        direction = 1 if side == "long" else -1
        selected = rows[
            rows["loop_id"].astype(str).eq(loop_id)
            & mask_by_container[container_id].reindex(rows.index).fillna(False)
        ].copy()
        if selected.empty:
            continue
        for surface, period in period_specs:
            period_rows = selected[
                selected["surface"].eq(surface) & _period_mask(selected, period, config)
            ].copy()
            available = period_rows.dropna(subset=[f"forward_{horizon}_bar_return"]).copy()
            for stop_model in config.stop_models:
                if available.empty:
                    continue
                risk = _risk_bps_for_model(
                    available,
                    model_name=stop_model,
                    expected_direction=direction,
                )
                for target_r in config.target_r_multiples:
                    scored = score_stop_model_events(
                        available,
                        horizon=horizon,
                        expected_direction=direction,
                        risk_bps=risk,
                        target_r=target_r,
                    )
                    symbol_counts = scored["symbol"].astype(str).value_counts()
                    month_counts = scored["month"].astype(str).value_counts()
                    for cost_bps in config.cost_bps_values:
                        r_values = _apply_cost_to_r(
                            scored,
                            cost_bps=cost_bps,
                            target_r=target_r,
                        ).dropna()
                        if r_values.empty:
                            continue
                        replay_months = scored.loc[r_values.index, "month"].astype(str)
                        month_r = r_values.groupby(replay_months).sum()
                        replay_rows.append(
                            {
                                "candidate_id": admission["candidate_id"],
                                "surface": surface,
                                "period": period,
                                "loop_id": loop_id,
                                "directional_side": side,
                                "horizon": f"h{horizon}",
                                "stop_model": stop_model,
                                "target_r": float(target_r),
                                "cost_bps": float(cost_bps),
                                "rows": int(len(r_values)),
                                "total_net_r": float(r_values.sum()),
                                "mean_r": float(r_values.mean()),
                                "median_r": float(r_values.median()),
                                "win_rate": float((r_values > 0.0).mean()),
                                "positive_months": int((month_r > 0.0).sum()),
                                "months": int(month_counts.size),
                                "symbols": int(symbol_counts.size),
                                "single_symbol_share": float(symbol_counts.iloc[0] / len(scored)),
                                "single_month_share": float(month_counts.iloc[0] / len(scored)),
                                "stop_hit_rate": float(
                                    pd.Series(scored["stop_hit"]).astype(bool).mean()
                                ),
                                "target_hit_rate": float(
                                    pd.Series(scored["target_hit"]).astype(bool).mean()
                                ),
                                "ambiguous_stop_target_rate": float(
                                    pd.Series(scored["target_stop_order_ambiguous"]).astype(bool).mean()
                                ),
                                "median_max_favorable_r": float(
                                    pd.to_numeric(
                                        scored["max_favorable_r"],
                                        errors="coerce",
                                    ).median()
                                ),
                                "median_max_adverse_r": float(
                                    pd.to_numeric(scored["max_adverse_r"], errors="coerce").median()
                                ),
                            }
                        )
    return pd.DataFrame(replay_rows)


def _family_replay_summary_columns() -> list[str]:
    return [
        "candidate_id",
        "loop_id",
        "surface",
        "period",
        "horizon",
        "stop_model",
        "target_r",
        "cost_bps",
        "rows",
        "final_close_total_r",
        "final_close_mean_r",
        "final_close_median_r",
        "final_close_win_rate",
        "target_capped_total_r",
        "target_capped_mean_r",
        "target_capped_median_r",
        "target_capped_win_rate",
        "stop_hit_rate",
        "target_hit_rate",
        "ambiguous_stop_target_rate",
        "median_risk_bps",
        "median_aligned_return_bps",
        "median_max_favorable_r",
        "median_max_adverse_r",
        "symbols",
        "months",
        "sessions",
        "single_symbol_share",
        "single_month_share",
        "single_session_share",
    ]


def _family_replay_scorecard_columns() -> list[str]:
    return [
        "candidate_id",
        "loop_id",
        "horizon",
        "visibility",
        "expression",
        "stop_model",
        "target_r",
        "smid_fresh_rows",
        "smid_fresh_final_close_total_r",
        "smid_fresh_final_close_mean_r",
        "smid_saved_rows",
        "smid_saved_final_close_total_r",
        "smid_saved_final_close_mean_r",
        "smid_full_rows",
        "smid_full_final_close_total_r",
        "smid_full_final_close_mean_r",
        "residual_full_rows",
        "residual_full_final_close_total_r",
        "residual_full_final_close_mean_r",
        "smid_full_stop_hit_rate",
        "smid_full_target_hit_rate",
        "smid_full_ambiguous_rate",
        "median_risk_bps",
        "min_period_final_close_mean_r",
        "combined_final_close_total_r",
    ]


def _family_selected_event_columns() -> list[str]:
    return [
        "candidate_id",
        "candidate_expression",
        "candidate_visibility",
        "loop_id",
        "surface",
        "symbol",
        "session_date",
        "timestamp",
        "month",
    ]


def _family_compressed_repair_mask(rows: pd.DataFrame) -> pd.Series:
    compressed = _str(rows, "source_compression_x_efficiency_regime").isin(
        ["compressed|mixed_efficiency", "compressed|choppy_efficiency"]
    )
    return compressed.fillna(False)


def _family_candidate_specs(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> dict[str, dict[str, Any]]:
    container = _fixed_c0_mask(rows, config)
    compact_source_range = _num(rows, "source_rolling_intraday_range_pct").le(0.0213415)
    compressed_next = _str(rows, "compression_x_efficiency_regime").eq(
        "compressed|choppy_efficiency"
    )
    source_compression = _str(rows, "source_compression_regime").eq("compressed")
    return {
        "l1_container_h6": {
            "loop_id": FAMILY_LOOP1,
            "horizon": 6,
            "expected_direction": -1,
            "visibility": "source_visible",
            "expression": "selected_discovered_container",
            "mask": container,
        },
        "l1_container_source_compression_h6": {
            "loop_id": FAMILY_LOOP1,
            "horizon": 6,
            "expected_direction": -1,
            "visibility": "source_visible",
            "expression": (
                "selected_discovered_container AND "
                "source_compression_regime == compressed"
            ),
            "mask": container & source_compression,
        },
        "l1_container_legacy_compressed_repair_h6": {
            "loop_id": FAMILY_LOOP1,
            "horizon": 6,
            "expected_direction": -1,
            "visibility": "source_visible_or_source_derived",
            "expression": "selected_discovered_container AND legacy compressed_repair",
            "mask": container & _family_compressed_repair_mask(rows),
        },
        "l2_container_h9": {
            "loop_id": FAMILY_LOOP2,
            "horizon": 9,
            "expected_direction": -1,
            "visibility": "source_visible",
            "expression": "selected_discovered_container",
            "mask": container,
        },
        "l2_compact_next_compressed_choppy_h9": {
            "loop_id": FAMILY_LOOP2,
            "horizon": 9,
            "expected_direction": -1,
            "visibility": "next_event_start",
            "expression": (
                "selected_discovered_container AND "
                "source_rolling_intraday_range_pct <= 0.0213415 AND "
                "compression_x_efficiency_regime == compressed|choppy_efficiency"
            ),
            "mask": container & compact_source_range & compressed_next,
        },
    }


def _family_score_events(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
    stop_model: str,
    target_r: float,
    cost_bps: float,
) -> pd.DataFrame:
    score_rows = rows.copy()
    for suffix in ("mfe", "mae"):
        source_col = f"source_forward_{horizon}_bar_{suffix}"
        replay_col = f"forward_{horizon}_bar_{suffix}"
        if source_col in score_rows:
            score_rows[replay_col] = score_rows[source_col]
    risk = _risk_bps_for_model(
        score_rows,
        model_name=stop_model,
        expected_direction=expected_direction,
        structure_buffer_bps=10.0,
        min_structure_stop_bps=5.0,
    )
    scored = score_stop_model_events(
        score_rows,
        horizon=horizon,
        expected_direction=expected_direction,
        risk_bps=risk,
        target_r=target_r,
    )
    risk_series = pd.to_numeric(scored["risk_bps"], errors="coerce")
    aligned = pd.to_numeric(scored["aligned_return_bps"], errors="coerce")
    stop_hit = pd.Series(scored["stop_hit"]).astype(bool)
    target_hit = pd.Series(scored["target_hit"]).astype(bool)
    raw_after_cost = (aligned - float(cost_bps)) / risk_series
    stop_after_cost = -1.0 - (float(cost_bps) / risk_series)
    final_close = raw_after_cost.where(~stop_hit, stop_after_cost)
    target_capped = raw_after_cost.clip(upper=float(target_r)).where(
        ~target_hit,
        float(target_r),
    )
    target_capped = target_capped.where(~stop_hit, stop_after_cost)
    scored["stop_model"] = stop_model
    scored["target_r"] = float(target_r)
    scored["cost_bps"] = float(cost_bps)
    scored["final_close_r_after_cost"] = final_close
    scored["target_capped_r_after_cost"] = target_capped
    scored["win_final_close_after_cost"] = final_close > 0.0
    scored["win_target_capped_after_cost"] = target_capped > 0.0
    return scored


def _family_concentration(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "symbols": 0,
            "months": 0,
            "sessions": 0,
            "single_symbol_share": None,
            "single_month_share": None,
            "single_session_share": None,
        }
    symbol_counts = rows["symbol"].astype(str).value_counts()
    month_counts = rows["month"].astype(str).value_counts()
    session_counts = (
        rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1).value_counts()
    )
    return {
        "symbols": int(symbol_counts.size),
        "months": int(month_counts.size),
        "sessions": int(session_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
    }


def _summarize_family_replay(
    rows: pd.DataFrame,
    *,
    candidate_id: str,
    loop_id: str,
    surface: str,
    period: str,
    horizon: int,
    stop_model: str,
    target_r: float,
    cost_bps: float,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_id": candidate_id,
        "loop_id": loop_id,
        "surface": surface,
        "period": period,
        "horizon": f"h{horizon}",
        "stop_model": stop_model,
        "target_r": float(target_r),
        "cost_bps": float(cost_bps),
        "rows": int(len(rows)),
    }
    if rows.empty:
        return {**base, **dict.fromkeys(_family_replay_summary_columns()[9:], None)}
    final_close = pd.to_numeric(rows["final_close_r_after_cost"], errors="coerce")
    target_capped = pd.to_numeric(rows["target_capped_r_after_cost"], errors="coerce")
    return {
        **base,
        "final_close_total_r": float(final_close.sum()),
        "final_close_mean_r": float(final_close.mean()),
        "final_close_median_r": float(final_close.median()),
        "final_close_win_rate": float((final_close > 0.0).mean()),
        "target_capped_total_r": float(target_capped.sum()),
        "target_capped_mean_r": float(target_capped.mean()),
        "target_capped_median_r": float(target_capped.median()),
        "target_capped_win_rate": float((target_capped > 0.0).mean()),
        "stop_hit_rate": float(pd.Series(rows["stop_hit"]).astype(bool).mean()),
        "target_hit_rate": float(pd.Series(rows["target_hit"]).astype(bool).mean()),
        "ambiguous_stop_target_rate": float(
            pd.Series(rows["target_stop_order_ambiguous"]).astype(bool).mean()
        ),
        "median_risk_bps": float(pd.to_numeric(rows["risk_bps"], errors="coerce").median()),
        "median_aligned_return_bps": float(
            pd.to_numeric(rows["aligned_return_bps"], errors="coerce").median()
        ),
        "median_max_favorable_r": float(
            pd.to_numeric(rows["max_favorable_r"], errors="coerce").median()
        ),
        "median_max_adverse_r": float(
            pd.to_numeric(rows["max_adverse_r"], errors="coerce").median()
        ),
        **_family_concentration(rows),
    }


def _family_stop_models(config: TemplateDiscoverySystemConfig) -> tuple[str, ...]:
    if config.stop_models == ("fixed_50bps",) and config.mode == "family-r-replay":
        return DEFAULT_FAMILY_STOP_MODELS
    return config.stop_models


def _family_target_r_multiples(config: TemplateDiscoverySystemConfig) -> tuple[float, ...]:
    if config.target_r_multiples == (1.0,) and config.mode == "family-r-replay":
        return DEFAULT_FAMILY_TARGET_R_MULTIPLES
    return config.target_r_multiples


def _run_family_r_replay(
    rows: pd.DataFrame,
    config: TemplateDiscoverySystemConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = _family_candidate_specs(rows, config)
    selected_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    periods = tuple(_periods(config))
    for candidate_id, spec in specs.items():
        selected = rows[
            rows["loop_id"].astype(str).eq(str(spec["loop_id"]))
            & spec["mask"].reindex(rows.index).fillna(False)
        ].copy()
        if not selected.empty:
            selected["candidate_id"] = candidate_id
            selected["candidate_expression"] = spec["expression"]
            selected["candidate_visibility"] = spec["visibility"]
            selected_frames.append(selected)
        for surface, surface_rows in selected.groupby("surface", sort=False):
            for period in periods:
                period_rows = surface_rows[_period_mask(surface_rows, period, config)].copy()
                forward_col = f"forward_{spec['horizon']}_bar_return"
                available = (
                    period_rows.dropna(subset=[forward_col]).copy()
                    if forward_col in period_rows
                    else pd.DataFrame()
                )
                for stop_model in _family_stop_models(config):
                    for target_r in _family_target_r_multiples(config):
                        for cost_bps in config.cost_bps_values:
                            if available.empty:
                                summary_rows.append(
                                    _summarize_family_replay(
                                        pd.DataFrame(),
                                        candidate_id=candidate_id,
                                        loop_id=spec["loop_id"],
                                        surface=surface,
                                        period=period,
                                        horizon=spec["horizon"],
                                        stop_model=stop_model,
                                        target_r=target_r,
                                        cost_bps=cost_bps,
                                    )
                                )
                                continue
                            scored = _family_score_events(
                                available,
                                horizon=spec["horizon"],
                                expected_direction=spec["expected_direction"],
                                stop_model=stop_model,
                                target_r=target_r,
                                cost_bps=cost_bps,
                            )
                            scored["candidate_id"] = candidate_id
                            scored["candidate_expression"] = spec["expression"]
                            scored["candidate_visibility"] = spec["visibility"]
                            scored["surface"] = surface
                            scored["period"] = period
                            scored["horizon"] = f"h{spec['horizon']}"
                            summary_rows.append(
                                _summarize_family_replay(
                                    scored,
                                    candidate_id=candidate_id,
                                    loop_id=spec["loop_id"],
                                    surface=surface,
                                    period=period,
                                    horizon=spec["horizon"],
                                    stop_model=stop_model,
                                    target_r=target_r,
                                    cost_bps=cost_bps,
                                )
                            )
    selected_events = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame(columns=_family_selected_event_columns())
    )
    summary = (
        pd.DataFrame(summary_rows)
        if summary_rows
        else pd.DataFrame(columns=_family_replay_summary_columns())
    )
    if not summary.empty:
        summary = summary.reindex(columns=_family_replay_summary_columns())
    scorecard = _family_scorecard(summary, specs)
    cost_focus = _family_cost_focus(summary, scorecard)
    return summary, scorecard, cost_focus, selected_events


def _family_scorecard(
    summary: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=_family_replay_scorecard_columns())
    primary = summary[summary["cost_bps"].eq(0.0)].copy()
    records: list[dict[str, Any]] = []
    for (candidate_id, stop_model, target_r), group in primary.groupby(
        ["candidate_id", "stop_model", "target_r"],
        sort=False,
    ):
        def get(
            surface: str,
            period: str,
            col: str,
            group: pd.DataFrame = group,
        ) -> Any:
            row = group[group["surface"].eq(surface) & group["period"].eq(period)]
            if row.empty:
                return None
            return row.iloc[0].get(col)

        means = [
            get("smid24", "fresh_year", "final_close_mean_r"),
            get("smid24", "saved_year", "final_close_mean_r"),
            get("residual_ex_smid24", "full_available", "final_close_mean_r"),
        ]
        valid_means = [value for value in means if value is not None and not pd.isna(value)]
        spec = specs[str(candidate_id)]
        record = {
            "candidate_id": candidate_id,
            "loop_id": spec["loop_id"],
            "horizon": f"h{spec['horizon']}",
            "visibility": spec["visibility"],
            "expression": spec["expression"],
            "stop_model": stop_model,
            "target_r": float(target_r),
            "smid_fresh_rows": get("smid24", "fresh_year", "rows"),
            "smid_fresh_final_close_total_r": get(
                "smid24",
                "fresh_year",
                "final_close_total_r",
            ),
            "smid_fresh_final_close_mean_r": get(
                "smid24",
                "fresh_year",
                "final_close_mean_r",
            ),
            "smid_saved_rows": get("smid24", "saved_year", "rows"),
            "smid_saved_final_close_total_r": get(
                "smid24",
                "saved_year",
                "final_close_total_r",
            ),
            "smid_saved_final_close_mean_r": get(
                "smid24",
                "saved_year",
                "final_close_mean_r",
            ),
            "smid_full_rows": get("smid24", "full_available", "rows"),
            "smid_full_final_close_total_r": get(
                "smid24",
                "full_available",
                "final_close_total_r",
            ),
            "smid_full_final_close_mean_r": get(
                "smid24",
                "full_available",
                "final_close_mean_r",
            ),
            "residual_full_rows": get("residual_ex_smid24", "full_available", "rows"),
            "residual_full_final_close_total_r": get(
                "residual_ex_smid24",
                "full_available",
                "final_close_total_r",
            ),
            "residual_full_final_close_mean_r": get(
                "residual_ex_smid24",
                "full_available",
                "final_close_mean_r",
            ),
            "smid_full_stop_hit_rate": get("smid24", "full_available", "stop_hit_rate"),
            "smid_full_target_hit_rate": get("smid24", "full_available", "target_hit_rate"),
            "smid_full_ambiguous_rate": get(
                "smid24",
                "full_available",
                "ambiguous_stop_target_rate",
            ),
            "median_risk_bps": get("smid24", "full_available", "median_risk_bps"),
            "min_period_final_close_mean_r": min(valid_means) if valid_means else None,
        }
        record["combined_final_close_total_r"] = sum(
            value
            for value in [
                record["smid_full_final_close_total_r"],
                record["residual_full_final_close_total_r"],
            ]
            if value is not None and not pd.isna(value)
        )
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=_family_replay_scorecard_columns())
    return frame.reindex(columns=_family_replay_scorecard_columns()).sort_values(
        [
            "min_period_final_close_mean_r",
            "combined_final_close_total_r",
            "smid_full_final_close_total_r",
        ],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _family_cost_focus(summary: pd.DataFrame, scorecard: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or scorecard.empty:
        return pd.DataFrame(columns=_family_replay_summary_columns())
    best_ids = scorecard.groupby("candidate_id", sort=False).head(1)[
        ["candidate_id", "stop_model", "target_r"]
    ]
    frames: list[pd.DataFrame] = []
    for _, row in best_ids.iterrows():
        frames.append(
            summary[
                summary["candidate_id"].eq(row["candidate_id"])
                & summary["stop_model"].eq(row["stop_model"])
                & summary["target_r"].eq(row["target_r"])
                & summary["period"].isin(["fresh_year", "saved_year", "full_available"])
            ].copy()
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=_family_replay_summary_columns()
    )


def _decision(admissions: pd.DataFrame, blockers: pd.DataFrame, replay: pd.DataFrame) -> str:
    if not admissions.empty:
        if not replay.empty and float(replay["total_net_r"].sum()) < 0.0:
            return "reject_negative_r_after_exit"
        return "continue_research_directional_supported"
    if not blockers.empty:
        return "continue_research_needs_blocker"
    return "reject_unstable_directional_split"


def _family_replay_decision(scorecard: pd.DataFrame) -> str:
    if scorecard.empty:
        return "reject_no_family_replay_candidates"
    means = pd.to_numeric(scorecard["min_period_final_close_mean_r"], errors="coerce").dropna()
    if means.empty:
        return "continue_research_family_replay_diagnostic_only"
    if float(means.max()) > 0.0:
        return "continue_research_family_replay_supported"
    return "reject_family_replay_no_positive_transfer"


def _write_summary_md(
    path: Path,
    payload: dict[str, Any],
    behavior_loops: pd.DataFrame,
    loop_regimes: pd.DataFrame,
    mixed_regimes: pd.DataFrame,
    transition_regimes: pd.DataFrame,
    c0_parent_readout: pd.DataFrame,
    b0_summary: pd.DataFrame,
    b0_routes: pd.DataFrame,
    containers: pd.DataFrame,
    routes: pd.DataFrame,
    context_admissions: pd.DataFrame,
    context_blockers: pd.DataFrame,
    admissions: pd.DataFrame,
    blockers: pd.DataFrame,
    family_scorecard: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame[cols].head(limit)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
        return "\n".join(lines)

    route_cols = [
        "candidate_id",
        "loop_id",
        "directional_side",
        "horizon",
        "route_direction",
        "mean_abs_lift",
    ]
    loop_cols = [
        "loop_id",
        "loop_rows",
        "loop_rate_transition_denominator",
        "symbols",
        "months",
        "candidate_behavior_loop",
    ]
    container_cols = [
        "container_id",
        "level",
        "expression",
        "rows",
        "loops",
        "stable_route_count",
        "routing_score",
        "recommended_container_candidate",
    ]
    regime_cols = [
        "loop_id",
        "regime_feature",
        "regime_value",
        "rows",
        "loop_share",
        "surface_share",
        "representation_ratio",
    ]
    b0_cols = [
        "container_id",
        "b0_state",
        "rows",
        "parent_rows",
        "stable_route_count",
        "stable_positive_count",
        "stable_negative_count",
        "mean_stable_abs_lift",
    ]
    context_cols = [
        "candidate_id",
        "loop_id",
        "visibility",
        "directional_side",
        "horizon",
        "route_direction",
        "mean_abs_lift",
    ]
    family_cols = [
        "candidate_id",
        "horizon",
        "visibility",
        "stop_model",
        "target_r",
        "smid_fresh_rows",
        "smid_fresh_final_close_total_r",
        "smid_saved_rows",
        "smid_saved_final_close_total_r",
        "residual_full_rows",
        "residual_full_final_close_total_r",
        "min_period_final_close_mean_r",
    ]
    lines = [
        "# Template Discovery System V0",
        "",
        "Clean-slate research-only template discovery from existing local event rows.",
        "",
        f"Decision: `{payload['decision']}`",
        f"Mode: `{payload['mode']}`",
        "",
        "## Safety",
        "",
        "- `research_only: true`",
        "- `live_ordering_enabled: false`",
        "- `order_placement: disabled`",
        "- `edge_claimed: false`",
        "",
        "## Clean-Slate Boundary",
        "",
        "- `saved_rules_used: false`",
        "- `seed_report_used: false`",
        "- `yaml_rules_saved: false`",
        "",
        "## Behavior Loop Discovery",
        "",
        table(behavior_loops, loop_cols),
        "",
        "## Loop Regime Occupancy",
        "",
        table(loop_regimes, regime_cols),
        "",
        "## Mixed Regime Occupancy",
        "",
        table(mixed_regimes, regime_cols),
        "",
        "## Transition Regime Occupancy",
        "",
        table(transition_regimes, regime_cols),
        "",
        "## Fixed C0 Parent Readout",
        "",
        table(c0_parent_readout, route_cols),
        "",
        "## B0 + C0 Overlay",
        "",
        table(b0_summary, b0_cols),
        "",
        "## B0 Route Detail",
        "",
        table(b0_routes, route_cols),
        "",
        "## Top Containers",
        "",
        table(containers, container_cols),
        "",
        "## Loop Context Admission Refinements",
        "",
        table(context_admissions, context_cols),
        "",
        "## Loop Context Blocker Refinements",
        "",
        table(context_blockers, context_cols),
        "",
        "## Stable Admissions",
        "",
        table(admissions, route_cols),
        "",
        "## Stable Blockers",
        "",
        table(blockers, route_cols),
        "",
        "## Family R Replay Scorecard",
        "",
        table(family_scorecard, family_cols),
        "",
        "## Strongest Routes",
        "",
        table(routes, route_cols),
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in payload["reports"].values())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empty_standard_report_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "behavior_loop_scorecard": run_dir / "behavior_loop_scorecard.csv",
        "loop_regime_occupancy": run_dir / "loop_regime_occupancy.csv",
        "loop_mixed_regime_occupancy": run_dir / "loop_mixed_regime_occupancy.csv",
        "loop_transition_regime_occupancy": run_dir
        / "loop_transition_regime_occupancy.csv",
        "c0_parent_readout": run_dir / "c0_parent_readout.csv",
        "b0_state_summary": run_dir / "b0_state_summary.csv",
        "b0_route_detail": run_dir / "b0_route_detail.csv",
        "loop_context_refinement": run_dir / "loop_context_refinement.csv",
        "loop_context_admissions": run_dir / "loop_context_admissions.csv",
        "loop_context_blockers": run_dir / "loop_context_blockers.csv",
        "atom_scorecard": run_dir / "atom_scorecard.csv",
        "container_scorecard": run_dir / "container_scorecard.csv",
        "loop_routing_detail": run_dir / "loop_routing_detail.csv",
        "family_test_detail": run_dir / "family_test_detail.csv",
        "concentration_warnings": run_dir / "concentration_warnings.csv",
        "admission_candidates": run_dir / "admission_candidates.csv",
        "blocker_candidates": run_dir / "blocker_candidates.csv",
        "replay_results": run_dir / "replay_results.csv",
        "family_r_replay_summary": run_dir / "family_r_replay_summary.csv",
        "family_r_replay_scorecard": run_dir / "family_r_replay_scorecard.csv",
        "family_r_replay_cost_sensitivity": run_dir
        / "family_r_replay_cost_sensitivity.csv",
        "family_r_replay_selected_events": run_dir
        / "family_r_replay_selected_events.csv",
    }


def _standard_report_names() -> dict[str, str]:
    return {
        "behavior_loop_scorecard": "behavior_loop_scorecard.csv",
        "loop_regime_occupancy": "loop_regime_occupancy.csv",
        "loop_mixed_regime_occupancy": "loop_mixed_regime_occupancy.csv",
        "loop_transition_regime_occupancy": "loop_transition_regime_occupancy.csv",
        "c0_parent_readout": "c0_parent_readout.csv",
        "b0_state_summary": "b0_state_summary.csv",
        "b0_route_detail": "b0_route_detail.csv",
        "loop_context_refinement": "loop_context_refinement.csv",
        "loop_context_admissions": "loop_context_admissions.csv",
        "loop_context_blockers": "loop_context_blockers.csv",
        "atom_scorecard": "atom_scorecard.csv",
        "container_scorecard": "container_scorecard.csv",
        "loop_routing_detail": "loop_routing_detail.csv",
        "family_test_detail": "family_test_detail.csv",
        "concentration_warnings": "concentration_warnings.csv",
        "admission_candidates": "admission_candidates.csv",
        "blocker_candidates": "blocker_candidates.csv",
        "replay_results": "replay_results.csv",
        "family_r_replay_summary": "family_r_replay_summary.csv",
        "family_r_replay_scorecard": "family_r_replay_scorecard.csv",
        "family_r_replay_cost_sensitivity": "family_r_replay_cost_sensitivity.csv",
        "family_r_replay_selected_events": "family_r_replay_selected_events.csv",
    }


def _run_frozen_template_transfer_replay_mode(
    *,
    input_event_rows: tuple[TemplateDiscoveryEventInput, ...],
    run_id: str,
    run_dir: Path,
    config: TemplateDiscoverySystemConfig,
) -> TemplateDiscoverySystemResult:
    events = _load_event_rows(input_event_rows)
    transitions = _enrich_transitions_with_frozen_transfer_market_context(
        events=events,
        transitions=_build_transitions(events),
    )
    events["month"] = pd.to_datetime(events["timestamp"], utc=True).dt.strftime("%Y-%m")

    all_rows, audit = _run_frozen_template_transfer_specs(
        events=events,
        transitions=transitions,
        config=config,
        skip_missing=False,
    )
    all_rows = _ensure_frozen_month(all_rows) if not all_rows.empty else all_rows
    deduped_trades, overlap_count = _dedupe_frozen_components(all_rows)
    transfer_summary = pd.DataFrame([_summarize_frozen_rows(deduped_trades)])
    component_summary_before_dedupe = _grouped_frozen_summary(all_rows, ["component"])
    component_summary_after_dedupe = _grouped_frozen_summary(deduped_trades, ["component"])
    rule_summary = _grouped_frozen_summary(deduped_trades, ["component", "rule_id"])
    period_summary = _grouped_frozen_summary(deduped_trades, ["research_period"])
    monthly_summary = _frozen_monthly_summary(deduped_trades)

    standard_paths = _empty_standard_report_paths(run_dir)
    paths: dict[str, Path] = {
        "summary_json": run_dir / "summary.json",
        "summary_markdown": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        **standard_paths,
        "frozen_template_transfer_all_rows": run_dir
        / "frozen_template_transfer_all_rows.csv",
        "frozen_template_transfer_exact_dedupe_trades": run_dir
        / "frozen_template_transfer_exact_dedupe_trades.csv",
        "frozen_template_transfer_template_audit": run_dir
        / "frozen_template_transfer_template_audit.csv",
        "frozen_template_transfer_summary": run_dir
        / "frozen_template_transfer_summary.csv",
        "frozen_template_transfer_component_summary": run_dir
        / "frozen_template_transfer_component_summary.csv",
        "frozen_template_transfer_component_summary_after_dedupe": run_dir
        / "frozen_template_transfer_component_summary_after_dedupe.csv",
        "frozen_template_transfer_rule_summary": run_dir
        / "frozen_template_transfer_rule_summary.csv",
        "frozen_template_transfer_period_summary": run_dir
        / "frozen_template_transfer_period_summary.csv",
        "frozen_template_transfer_monthly_summary": run_dir
        / "frozen_template_transfer_monthly_summary.csv",
    }

    for path in standard_paths.values():
        _write_csv(path, pd.DataFrame())
    _write_csv(paths["frozen_template_transfer_all_rows"], all_rows)
    _write_csv(paths["frozen_template_transfer_exact_dedupe_trades"], deduped_trades)
    _write_csv(paths["frozen_template_transfer_template_audit"], audit)
    _write_csv(paths["frozen_template_transfer_summary"], transfer_summary)
    _write_csv(
        paths["frozen_template_transfer_component_summary"],
        component_summary_before_dedupe,
    )
    _write_csv(
        paths["frozen_template_transfer_component_summary_after_dedupe"],
        component_summary_after_dedupe,
    )
    _write_csv(paths["frozen_template_transfer_rule_summary"], rule_summary)
    _write_csv(paths["frozen_template_transfer_period_summary"], period_summary)
    _write_csv(paths["frozen_template_transfer_monthly_summary"], monthly_summary)

    decision = (
        "continue_research_frozen_template_transfer_replayed"
        if not deduped_trades.empty
        else "reject_no_frozen_template_transfer_rows"
    )
    total_r = (
        float(transfer_summary.iloc[0]["total_r"])
        if not transfer_summary.empty and "total_r" in transfer_summary
        else 0.0
    )
    reports = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
        "decision_json": "decision.json",
        **_standard_report_names(),
        "frozen_template_transfer_all_rows": (
            "frozen_template_transfer_all_rows.csv"
        ),
        "frozen_template_transfer_exact_dedupe_trades": (
            "frozen_template_transfer_exact_dedupe_trades.csv"
        ),
        "frozen_template_transfer_template_audit": (
            "frozen_template_transfer_template_audit.csv"
        ),
        "frozen_template_transfer_summary": "frozen_template_transfer_summary.csv",
        "frozen_template_transfer_component_summary": (
            "frozen_template_transfer_component_summary.csv"
        ),
        "frozen_template_transfer_component_summary_after_dedupe": (
            "frozen_template_transfer_component_summary_after_dedupe.csv"
        ),
        "frozen_template_transfer_rule_summary": (
            "frozen_template_transfer_rule_summary.csv"
        ),
        "frozen_template_transfer_period_summary": (
            "frozen_template_transfer_period_summary.csv"
        ),
        "frozen_template_transfer_monthly_summary": (
            "frozen_template_transfer_monthly_summary.csv"
        ),
    }
    payload = {
        "run_id": run_id,
        "mode": config.mode,
        "universe_profile": config.universe_profile,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "clean_slate": False,
        "saved_rules_used": True,
        "seed_report_used": False,
        "yaml_rules_saved": False,
        "data_source": "existing event rows plus built-in frozen research template specs",
        "volume_label": "historical_volume from existing local 5m OHLCV event reports",
        "decision": decision,
        "input_event_rows": [
            {"label": item.label, "event_rows_path": str(item.event_rows_path)}
            for item in input_event_rows
        ],
        "event_rows": int(len(events)),
        "transition_rows": int(len(transitions)),
        "template_count": int(len(FROZEN_TEMPLATE_TRANSFER_SPECS)),
        "market_context_detector": _frozen_transfer_detector_id(),
        "discovery_ladder": [
            "existing_event_rows",
            "same_session_behavior_loop_transitions",
            "broad_and_b0_context_rebuilt_from_event_rows",
            "frozen_template_expression_selection",
            "source_context_templates_rematerialized_to_source_event_rows",
            "exact_symbol_timestamp_priority_dedupe",
            "frozen_template_transfer_r_summary",
        ],
        "notes": [
            (
                "Current fixed-next templates are selected on source-context "
                "transitions and rematerialized to the original source event row "
                "before R replay."
            ),
            (
                "Broad participation fields and B0 dirvol context are rebuilt "
                "from the supplied event rows."
            ),
            (
                "The later consistency cut that reduced fixed-next rows in the "
                "corrected frozen combo is row-set evidence, not a fully "
                "portable predicate."
            ),
        ],
        "frozen_template_transfer_all_row_count": int(len(all_rows)),
        "frozen_template_transfer_component_row_count": int(len(all_rows)),
        "frozen_template_transfer_trade_count": int(len(deduped_trades)),
        "frozen_template_transfer_exact_overlap_count": overlap_count,
        "frozen_template_transfer_total_r": total_r,
        "frozen_template_transfer_component_count": int(
            deduped_trades["component"].nunique()
        )
        if "component" in deduped_trades
        else 0,
        "frozen_template_transfer_rule_count": int(deduped_trades["rule_id"].nunique())
        if "rule_id" in deduped_trades
        else 0,
        "reports": reports,
        "config": {
            **config.__dict__,
            "periods": _periods(config),
        },
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
            "clean_slate": False,
            "saved_rules_used": True,
            "seed_report_used": False,
            "yaml_rules_saved": False,
            "market_context_detector": _frozen_transfer_detector_id(),
        },
    )
    _write_frozen_transfer_summary_md(
        paths["summary_markdown"],
        payload,
        transfer_summary,
        component_summary_before_dedupe,
        component_summary_after_dedupe,
        rule_summary,
    )

    return TemplateDiscoverySystemResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_markdown"],
        decision_json_path=paths["decision_json"],
        behavior_loop_scorecard_csv_path=paths["behavior_loop_scorecard"],
        loop_regime_occupancy_csv_path=paths["loop_regime_occupancy"],
        loop_mixed_regime_occupancy_csv_path=paths["loop_mixed_regime_occupancy"],
        loop_transition_regime_occupancy_csv_path=paths[
            "loop_transition_regime_occupancy"
        ],
        c0_parent_readout_csv_path=paths["c0_parent_readout"],
        b0_state_summary_csv_path=paths["b0_state_summary"],
        b0_route_detail_csv_path=paths["b0_route_detail"],
        loop_context_refinement_csv_path=paths["loop_context_refinement"],
        loop_context_admissions_csv_path=paths["loop_context_admissions"],
        loop_context_blockers_csv_path=paths["loop_context_blockers"],
        atom_scorecard_csv_path=paths["atom_scorecard"],
        container_scorecard_csv_path=paths["container_scorecard"],
        loop_routing_detail_csv_path=paths["loop_routing_detail"],
        family_test_detail_csv_path=paths["family_test_detail"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        admission_candidates_csv_path=paths["admission_candidates"],
        blocker_candidates_csv_path=paths["blocker_candidates"],
        replay_results_csv_path=paths["replay_results"],
        decision=decision,
    )


def _run_frozen_combo_replay_mode(
    *,
    input_event_rows: tuple[TemplateDiscoveryEventInput, ...],
    run_id: str,
    run_dir: Path,
    config: TemplateDiscoverySystemConfig,
) -> TemplateDiscoverySystemResult:
    component_rows = _ensure_frozen_month(_load_frozen_component_rows(config))
    deduped_trades, overlap_count = _dedupe_frozen_components(component_rows)
    combo_summary = pd.DataFrame([_summarize_frozen_rows(deduped_trades)])
    component_summary_before_dedupe = _grouped_frozen_summary(component_rows, ["component"])
    component_summary_after_dedupe = _grouped_frozen_summary(deduped_trades, ["component"])
    rule_summary = _grouped_frozen_summary(deduped_trades, ["component", "rule_id"])
    monthly_summary = _frozen_monthly_summary(deduped_trades)

    candidate_book_path = _frozen_candidate_book_path(config)
    candidate_book = (
        pd.read_csv(candidate_book_path)
        if candidate_book_path is not None and candidate_book_path.exists()
        else pd.DataFrame()
    )

    standard_paths = _empty_standard_report_paths(run_dir)
    paths: dict[str, Path] = {
        "summary_json": run_dir / "summary.json",
        "summary_markdown": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        **standard_paths,
        "frozen_candidate_book": run_dir / "frozen_candidate_book.csv",
        "frozen_combo_all_components": run_dir / "frozen_combo_all_components.csv",
        "frozen_combo_exact_dedupe_trades": run_dir
        / "frozen_combo_exact_dedupe_trades.csv",
        "frozen_combo_summary": run_dir / "frozen_combo_summary.csv",
        "frozen_combo_component_summary": run_dir / "frozen_combo_component_summary.csv",
        "frozen_combo_component_summary_after_dedupe": run_dir
        / "frozen_combo_component_summary_after_dedupe.csv",
        "frozen_combo_rule_summary": run_dir / "frozen_combo_rule_summary.csv",
        "frozen_combo_monthly_summary": run_dir / "frozen_combo_monthly_summary.csv",
    }

    for path in standard_paths.values():
        _write_csv(path, pd.DataFrame())
    _write_csv(paths["frozen_candidate_book"], candidate_book)
    _write_csv(paths["frozen_combo_all_components"], component_rows)
    _write_csv(paths["frozen_combo_exact_dedupe_trades"], deduped_trades)
    _write_csv(paths["frozen_combo_summary"], combo_summary)
    _write_csv(paths["frozen_combo_component_summary"], component_summary_before_dedupe)
    _write_csv(
        paths["frozen_combo_component_summary_after_dedupe"],
        component_summary_after_dedupe,
    )
    _write_csv(paths["frozen_combo_rule_summary"], rule_summary)
    _write_csv(paths["frozen_combo_monthly_summary"], monthly_summary)

    decision = (
        "continue_research_frozen_combo_baseline_reproduced"
        if not deduped_trades.empty
        else "reject_no_frozen_combo_rows"
    )
    total_r = (
        float(combo_summary.iloc[0]["total_r"])
        if not combo_summary.empty and "total_r" in combo_summary
        else 0.0
    )
    reports = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
        "decision_json": "decision.json",
        **_standard_report_names(),
        "frozen_candidate_book": "frozen_candidate_book.csv",
        "frozen_combo_all_components": "frozen_combo_all_components.csv",
        "frozen_combo_exact_dedupe_trades": "frozen_combo_exact_dedupe_trades.csv",
        "frozen_combo_summary": "frozen_combo_summary.csv",
        "frozen_combo_component_summary": "frozen_combo_component_summary.csv",
        "frozen_combo_component_summary_after_dedupe": (
            "frozen_combo_component_summary_after_dedupe.csv"
        ),
        "frozen_combo_rule_summary": "frozen_combo_rule_summary.csv",
        "frozen_combo_monthly_summary": "frozen_combo_monthly_summary.csv",
    }
    payload = {
        "run_id": run_id,
        "mode": config.mode,
        "universe_profile": config.universe_profile,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "clean_slate": False,
        "saved_rules_used": True,
        "seed_report_used": True,
        "yaml_rules_saved": False,
        "data_source": "explicit frozen research component rows",
        "volume_label": "historical_volume from existing local 5m OHLCV event reports",
        "decision": decision,
        "input_event_rows": [
            {"label": item.label, "event_rows_path": str(item.event_rows_path)}
            for item in input_event_rows
        ],
        "frozen_component_paths": [str(path) for path in _frozen_component_paths(config)],
        "frozen_candidate_book_path": str(candidate_book_path)
        if candidate_book_path is not None
        else None,
        "discovery_ladder": [
            "frozen_candidate_book",
            "frozen_component_trade_rows",
            "exact_symbol_timestamp_priority_dedupe",
            "frozen_combo_r_summary",
        ],
        "frozen_combo_component_row_count": int(len(component_rows)),
        "frozen_combo_trade_count": int(len(deduped_trades)),
        "frozen_combo_exact_overlap_count": overlap_count,
        "frozen_combo_total_r": total_r,
        "frozen_combo_component_count": int(deduped_trades["component"].nunique())
        if "component" in deduped_trades
        else 0,
        "frozen_combo_rule_count": int(deduped_trades["rule_id"].nunique())
        if "rule_id" in deduped_trades
        else 0,
        "reports": reports,
        "config": {
            **config.__dict__,
            "periods": _periods(config),
        },
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
            "clean_slate": False,
            "saved_rules_used": True,
            "seed_report_used": True,
            "yaml_rules_saved": False,
        },
    )
    _write_frozen_combo_summary_md(
        paths["summary_markdown"],
        payload,
        combo_summary,
        component_summary_before_dedupe,
        component_summary_after_dedupe,
        rule_summary,
    )

    return TemplateDiscoverySystemResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_markdown"],
        decision_json_path=paths["decision_json"],
        behavior_loop_scorecard_csv_path=paths["behavior_loop_scorecard"],
        loop_regime_occupancy_csv_path=paths["loop_regime_occupancy"],
        loop_mixed_regime_occupancy_csv_path=paths["loop_mixed_regime_occupancy"],
        loop_transition_regime_occupancy_csv_path=paths[
            "loop_transition_regime_occupancy"
        ],
        c0_parent_readout_csv_path=paths["c0_parent_readout"],
        b0_state_summary_csv_path=paths["b0_state_summary"],
        b0_route_detail_csv_path=paths["b0_route_detail"],
        loop_context_refinement_csv_path=paths["loop_context_refinement"],
        loop_context_admissions_csv_path=paths["loop_context_admissions"],
        loop_context_blockers_csv_path=paths["loop_context_blockers"],
        atom_scorecard_csv_path=paths["atom_scorecard"],
        container_scorecard_csv_path=paths["container_scorecard"],
        loop_routing_detail_csv_path=paths["loop_routing_detail"],
        family_test_detail_csv_path=paths["family_test_detail"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        admission_candidates_csv_path=paths["admission_candidates"],
        blocker_candidates_csv_path=paths["blocker_candidates"],
        replay_results_csv_path=paths["replay_results"],
        decision=decision,
    )


def _run_template_component_selection_mode(
    *,
    input_event_rows: tuple[TemplateDiscoveryEventInput, ...],
    run_id: str,
    run_dir: Path,
    config: TemplateDiscoverySystemConfig,
) -> TemplateDiscoverySystemResult:
    candidate_paths = _component_candidate_paths(config)
    candidate_rows = _load_component_candidate_rows(config)
    scorecard = _score_component_candidates(candidate_rows, config)
    selected_book = _component_selection_book(scorecard)
    rejected = (
        scorecard[~scorecard["selected"].astype(bool)].copy()
        if not scorecard.empty
        else pd.DataFrame()
    )
    selected_rows = _selected_component_rows(candidate_rows, scorecard)
    deduped_trades, overlap_count = _dedupe_frozen_components(selected_rows)
    combo_summary = pd.DataFrame([_summarize_frozen_rows(deduped_trades)])
    component_summary_before_dedupe = _grouped_frozen_summary(selected_rows, ["component"])
    component_summary_after_dedupe = _grouped_frozen_summary(deduped_trades, ["component"])
    rule_summary = _grouped_frozen_summary(deduped_trades, ["component", "rule_id"])
    monthly_summary = _frozen_monthly_summary(deduped_trades)

    standard_paths = _empty_standard_report_paths(run_dir)
    paths: dict[str, Path] = {
        "summary_json": run_dir / "summary.json",
        "summary_markdown": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        **standard_paths,
        "component_candidate_rows": run_dir / "component_candidate_rows.csv",
        "component_candidate_scorecard": run_dir
        / "component_candidate_scorecard.csv",
        "selected_candidate_book": run_dir / "selected_candidate_book.csv",
        "rejected_component_candidates": run_dir
        / "rejected_component_candidates.csv",
        "selected_component_rows": run_dir / "selected_component_rows.csv",
        "selected_combo_exact_dedupe_trades": run_dir
        / "selected_combo_exact_dedupe_trades.csv",
        "selected_combo_summary": run_dir / "selected_combo_summary.csv",
        "selected_combo_component_summary": run_dir
        / "selected_combo_component_summary.csv",
        "selected_combo_component_summary_after_dedupe": run_dir
        / "selected_combo_component_summary_after_dedupe.csv",
        "selected_combo_rule_summary": run_dir / "selected_combo_rule_summary.csv",
        "selected_combo_monthly_summary": run_dir
        / "selected_combo_monthly_summary.csv",
    }

    for path in standard_paths.values():
        _write_csv(path, pd.DataFrame())
    _write_csv(paths["component_candidate_rows"], candidate_rows)
    _write_csv(paths["component_candidate_scorecard"], scorecard)
    _write_csv(paths["selected_candidate_book"], selected_book)
    _write_csv(paths["rejected_component_candidates"], rejected)
    _write_csv(paths["selected_component_rows"], selected_rows)
    _write_csv(paths["selected_combo_exact_dedupe_trades"], deduped_trades)
    _write_csv(paths["selected_combo_summary"], combo_summary)
    _write_csv(paths["selected_combo_component_summary"], component_summary_before_dedupe)
    _write_csv(
        paths["selected_combo_component_summary_after_dedupe"],
        component_summary_after_dedupe,
    )
    _write_csv(paths["selected_combo_rule_summary"], rule_summary)
    _write_csv(paths["selected_combo_monthly_summary"], monthly_summary)

    decision = (
        "continue_research_component_selection_replayed"
        if not selected_book.empty and not deduped_trades.empty
        else "reject_no_component_candidates_selected"
    )
    total_r = (
        float(combo_summary.iloc[0]["total_r"])
        if not combo_summary.empty and "total_r" in combo_summary
        else 0.0
    )
    reports = {
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
        "decision_json": "decision.json",
        **_standard_report_names(),
        "component_candidate_rows": "component_candidate_rows.csv",
        "component_candidate_scorecard": "component_candidate_scorecard.csv",
        "selected_candidate_book": "selected_candidate_book.csv",
        "rejected_component_candidates": "rejected_component_candidates.csv",
        "selected_component_rows": "selected_component_rows.csv",
        "selected_combo_exact_dedupe_trades": (
            "selected_combo_exact_dedupe_trades.csv"
        ),
        "selected_combo_summary": "selected_combo_summary.csv",
        "selected_combo_component_summary": "selected_combo_component_summary.csv",
        "selected_combo_component_summary_after_dedupe": (
            "selected_combo_component_summary_after_dedupe.csv"
        ),
        "selected_combo_rule_summary": "selected_combo_rule_summary.csv",
        "selected_combo_monthly_summary": "selected_combo_monthly_summary.csv",
    }
    saved_rules_used = any(
        path.name in FROZEN_COMBO_COMPONENT_FILES or "frozen" in str(path)
        for path in candidate_paths
    )
    payload = {
        "run_id": run_id,
        "mode": config.mode,
        "universe_profile": config.universe_profile,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "clean_slate": False,
        "saved_rules_used": saved_rules_used,
        "seed_report_used": True,
        "yaml_rules_saved": False,
        "automated_component_selection": True,
        "data_source": "research component candidate row artifacts",
        "volume_label": "historical_volume from existing local 5m OHLCV event reports",
        "decision": decision,
        "input_event_rows": [
            {"label": item.label, "event_rows_path": str(item.event_rows_path)}
            for item in input_event_rows
        ],
        "component_candidate_paths": [str(path) for path in candidate_paths],
        "discovery_ladder": [
            "component_candidate_rows",
            "component_candidate_scorecard",
            "selection_gates",
            "selected_candidate_book",
            "exact_symbol_timestamp_priority_dedupe",
            "selected_combo_r_summary",
        ],
        "component_candidate_row_count": int(len(candidate_rows)),
        "component_candidate_count": int(len(scorecard)),
        "selected_component_candidate_count": int(len(selected_book)),
        "rejected_component_candidate_count": int(len(rejected)),
        "selected_component_row_count_before_dedupe": int(len(selected_rows)),
        "selected_combo_trade_count": int(len(deduped_trades)),
        "selected_combo_exact_overlap_count": overlap_count,
        "selected_combo_total_r": total_r,
        "selected_combo_component_count": int(deduped_trades["component"].nunique())
        if "component" in deduped_trades
        else 0,
        "selected_combo_rule_count": int(deduped_trades["rule_id"].nunique())
        if "rule_id" in deduped_trades
        else 0,
        "reports": reports,
        "config": {
            **config.__dict__,
            "periods": _periods(config),
        },
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
            "clean_slate": False,
            "saved_rules_used": saved_rules_used,
            "seed_report_used": True,
            "yaml_rules_saved": False,
            "automated_component_selection": True,
        },
    )
    _write_component_selection_summary_md(
        paths["summary_markdown"],
        payload,
        combo_summary,
        scorecard,
        selected_book,
        rule_summary,
    )

    return TemplateDiscoverySystemResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_markdown"],
        decision_json_path=paths["decision_json"],
        behavior_loop_scorecard_csv_path=paths["behavior_loop_scorecard"],
        loop_regime_occupancy_csv_path=paths["loop_regime_occupancy"],
        loop_mixed_regime_occupancy_csv_path=paths["loop_mixed_regime_occupancy"],
        loop_transition_regime_occupancy_csv_path=paths[
            "loop_transition_regime_occupancy"
        ],
        c0_parent_readout_csv_path=paths["c0_parent_readout"],
        b0_state_summary_csv_path=paths["b0_state_summary"],
        b0_route_detail_csv_path=paths["b0_route_detail"],
        loop_context_refinement_csv_path=paths["loop_context_refinement"],
        loop_context_admissions_csv_path=paths["loop_context_admissions"],
        loop_context_blockers_csv_path=paths["loop_context_blockers"],
        atom_scorecard_csv_path=paths["atom_scorecard"],
        container_scorecard_csv_path=paths["container_scorecard"],
        loop_routing_detail_csv_path=paths["loop_routing_detail"],
        family_test_detail_csv_path=paths["family_test_detail"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        admission_candidates_csv_path=paths["admission_candidates"],
        blocker_candidates_csv_path=paths["blocker_candidates"],
        replay_results_csv_path=paths["replay_results"],
        decision=decision,
    )


def run_template_discovery_system_lab(
    *,
    input_event_rows: tuple[TemplateDiscoveryEventInput, ...],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: TemplateDiscoverySystemConfig | None = None,
) -> TemplateDiscoverySystemResult:
    """Run clean-slate template discovery from existing local event rows."""

    config = config or TemplateDiscoverySystemConfig()
    if config.mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported template discovery mode: {config.mode}")
    no_event_row_modes = {"frozen-combo-replay", "template-component-selection"}
    if config.mode not in no_event_row_modes and not input_event_rows:
        raise ValueError("Supply at least one input event_rows.csv.")

    run_id = "template_discovery_system_v0_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "frozen-combo-replay":
        return _run_frozen_combo_replay_mode(
            input_event_rows=input_event_rows,
            run_id=run_id,
            run_dir=run_dir,
            config=config,
        )
    if config.mode == "frozen-template-transfer-replay":
        return _run_frozen_template_transfer_replay_mode(
            input_event_rows=input_event_rows,
            run_id=run_id,
            run_dir=run_dir,
            config=config,
        )
    if config.mode == "template-component-selection":
        return _run_template_component_selection_mode(
            input_event_rows=input_event_rows,
            run_id=run_id,
            run_dir=run_dir,
            config=config,
        )

    events = _load_event_rows(input_event_rows)
    transitions = _build_transitions(events)
    behavior_loop_scorecard = _discover_behavior_loops(events, transitions, config)
    candidate_loop_ids = set(
        behavior_loop_scorecard.loc[
            behavior_loop_scorecard["candidate_behavior_loop"].fillna(False),
            "loop_id",
        ].astype(str)
    )
    routed_transitions = transitions[
        transitions["loop_id"].astype(str).isin(candidate_loop_ids)
    ].copy()
    loop_regimes, mixed_regimes, transition_regimes = _loop_regime_occupancy_reports(
        routed_transitions,
        config,
    )
    c0_parent_readout = _fixed_c0_parent_readout(routed_transitions, config)
    b0_transitions, b0_summary, b0_route_detail = _b0_route_reports(
        events,
        routed_transitions,
        config,
    )
    loop_context_refinement = _evaluate_loop_context_refinement(
        b0_transitions,
        candidate_loop_ids,
        config,
    )
    loop_context_admissions = loop_context_refinement[
        loop_context_refinement["candidate_kind"].eq("admission_refinement")
    ].copy() if not loop_context_refinement.empty else pd.DataFrame(
        columns=_loop_refinement_columns()
    )
    loop_context_blockers = loop_context_refinement[
        loop_context_refinement["candidate_kind"].eq("blocker_refinement")
    ].copy() if not loop_context_refinement.empty else pd.DataFrame(
        columns=_loop_refinement_columns()
    )
    atoms, atom_scorecard = _generate_atoms(routed_transitions, config)
    candidates, container_scorecard = _candidate_masks(routed_transitions, atoms, config)
    routes = _evaluate_routes(routed_transitions, candidates, config)
    container_scorecard = _score_container_routes(container_scorecard, routes)
    admissions, blockers, concentration_warnings = _candidate_outputs(routes)
    replay = (
        _run_replay(routed_transitions, candidates, admissions, config)
        if config.mode == "r-replay" and not admissions.empty
        else _run_replay(routed_transitions, candidates, pd.DataFrame(), config)
    )
    if config.mode == "family-r-replay":
        family_replay_transitions = _build_family_replay_transitions(events, config)
        (
            family_replay_summary,
            family_replay_scorecard,
            family_replay_cost_focus,
            family_selected,
        ) = _run_family_r_replay(
            family_replay_transitions,
            config,
        )
    else:
        family_replay_transitions = pd.DataFrame()
        family_replay_summary = pd.DataFrame(columns=_family_replay_summary_columns())
        family_replay_scorecard = pd.DataFrame(columns=_family_replay_scorecard_columns())
        family_replay_cost_focus = pd.DataFrame(columns=_family_replay_summary_columns())
        family_selected = pd.DataFrame(columns=_family_selected_event_columns())
    decision = (
        _family_replay_decision(family_replay_scorecard)
        if config.mode == "family-r-replay"
        else _decision(
            admissions,
            blockers,
            replay if config.mode == "r-replay" else pd.DataFrame(),
        )
    )

    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_markdown": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "behavior_loop_scorecard": run_dir / "behavior_loop_scorecard.csv",
        "loop_regime_occupancy": run_dir / "loop_regime_occupancy.csv",
        "loop_mixed_regime_occupancy": run_dir / "loop_mixed_regime_occupancy.csv",
        "loop_transition_regime_occupancy": run_dir
        / "loop_transition_regime_occupancy.csv",
        "c0_parent_readout": run_dir / "c0_parent_readout.csv",
        "b0_state_summary": run_dir / "b0_state_summary.csv",
        "b0_route_detail": run_dir / "b0_route_detail.csv",
        "loop_context_refinement": run_dir / "loop_context_refinement.csv",
        "loop_context_admissions": run_dir / "loop_context_admissions.csv",
        "loop_context_blockers": run_dir / "loop_context_blockers.csv",
        "atom_scorecard": run_dir / "atom_scorecard.csv",
        "container_scorecard": run_dir / "container_scorecard.csv",
        "loop_routing_detail": run_dir / "loop_routing_detail.csv",
        "family_test_detail": run_dir / "family_test_detail.csv",
        "concentration_warnings": run_dir / "concentration_warnings.csv",
        "admission_candidates": run_dir / "admission_candidates.csv",
        "blocker_candidates": run_dir / "blocker_candidates.csv",
        "replay_results": run_dir / "replay_results.csv",
        "family_r_replay_summary": run_dir / "family_r_replay_summary.csv",
        "family_r_replay_scorecard": run_dir / "family_r_replay_scorecard.csv",
        "family_r_replay_cost_sensitivity": run_dir
        / "family_r_replay_cost_sensitivity.csv",
        "family_r_replay_selected_events": run_dir
        / "family_r_replay_selected_events.csv",
    }

    _write_csv(paths["behavior_loop_scorecard"], behavior_loop_scorecard)
    _write_csv(paths["loop_regime_occupancy"], loop_regimes)
    _write_csv(paths["loop_mixed_regime_occupancy"], mixed_regimes)
    _write_csv(paths["loop_transition_regime_occupancy"], transition_regimes)
    _write_csv(paths["c0_parent_readout"], c0_parent_readout)
    _write_csv(paths["b0_state_summary"], b0_summary)
    _write_csv(paths["b0_route_detail"], b0_route_detail)
    _write_csv(paths["loop_context_refinement"], loop_context_refinement)
    _write_csv(paths["loop_context_admissions"], loop_context_admissions)
    _write_csv(paths["loop_context_blockers"], loop_context_blockers)
    _write_csv(paths["atom_scorecard"], atom_scorecard)
    _write_csv(paths["container_scorecard"], container_scorecard)
    _write_csv(paths["loop_routing_detail"], routes)
    _write_csv(paths["family_test_detail"], routes)
    _write_csv(paths["concentration_warnings"], concentration_warnings)
    _write_csv(paths["admission_candidates"], admissions)
    _write_csv(paths["blocker_candidates"], blockers)
    _write_csv(paths["replay_results"], replay)
    _write_csv(paths["family_r_replay_summary"], family_replay_summary)
    _write_csv(paths["family_r_replay_scorecard"], family_replay_scorecard)
    _write_csv(paths["family_r_replay_cost_sensitivity"], family_replay_cost_focus)
    _write_csv(paths["family_r_replay_selected_events"], family_selected)

    payload = {
        "run_id": run_id,
        "mode": config.mode,
        "universe_profile": config.universe_profile,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "clean_slate": True,
        "saved_rules_used": False,
        "seed_report_used": False,
        "yaml_rules_saved": False,
        "data_source": "existing local event_rows.csv only",
        "volume_label": "historical_volume from existing local 5m OHLCV event reports",
        "decision": decision,
        "input_event_rows": [
            {"label": item.label, "event_rows_path": str(item.event_rows_path)}
            for item in input_event_rows
        ],
        "discovery_ladder": [
            "behavior_loop_discovery",
            "loop_regime_occupancy",
            "fixed_c0_parent_readout",
            "b0_broad_tape_overlay",
            "loop_context_refinement",
            "source_visible_container_generation",
            "container_loop_routing",
            "admission_blocker_split",
            "optional_r_replay",
            "optional_family_r_replay",
        ],
        "transition_surface_rows": transitions["surface"].value_counts().sort_index().to_dict()
        if not transitions.empty
        else {},
        "surface_rows": routed_transitions["surface"].value_counts().sort_index().to_dict()
        if not routed_transitions.empty
        else {},
        "behavior_loop_count": int(len(behavior_loop_scorecard)),
        "candidate_behavior_loop_count": int(len(candidate_loop_ids)),
        "loop_regime_occupancy_count": int(len(loop_regimes)),
        "loop_mixed_regime_occupancy_count": int(len(mixed_regimes)),
        "loop_transition_regime_occupancy_count": int(len(transition_regimes)),
        "c0_parent_readout_count": int(len(c0_parent_readout)),
        "b0_state_summary_count": int(len(b0_summary)),
        "b0_route_detail_count": int(len(b0_route_detail)),
        "loop_context_refinement_count": int(len(loop_context_refinement)),
        "loop_context_admission_count": int(len(loop_context_admissions)),
        "loop_context_blocker_count": int(len(loop_context_blockers)),
        "atom_count": int(atom_scorecard["selected_for_generation"].sum())
        if "selected_for_generation" in atom_scorecard
        else int(len(atom_scorecard)),
        "atom_scorecard_row_count": int(len(atom_scorecard)),
        "container_count": int(len(container_scorecard)),
        "routed_container_count": int(container_scorecard["selected_for_routing"].sum())
        if "selected_for_routing" in container_scorecard
        else 0,
        "recommended_container_count": int(
            container_scorecard["recommended_container_candidate"].sum()
        )
        if "recommended_container_candidate" in container_scorecard
        else 0,
        "route_detail_count": int(len(routes)),
        "admission_candidate_count": int(len(admissions)),
        "blocker_candidate_count": int(len(blockers)),
        "replay_result_count": int(len(replay)),
        "family_r_replay_summary_count": int(len(family_replay_summary)),
        "family_r_replay_scorecard_count": int(len(family_replay_scorecard)),
        "family_r_replay_cost_sensitivity_count": int(len(family_replay_cost_focus)),
        "family_r_replay_selected_event_count": int(len(family_selected)),
        "family_r_replay_transition_count": int(len(family_replay_transitions)),
        "family_r_replay_stop_models": list(_family_stop_models(config)),
        "family_r_replay_target_r_multiples": list(_family_target_r_multiples(config)),
        "reports": {
            "summary_json": "summary.json",
            "summary_markdown": "summary.md",
            "decision_json": "decision.json",
            "behavior_loop_scorecard": "behavior_loop_scorecard.csv",
            "loop_regime_occupancy": "loop_regime_occupancy.csv",
            "loop_mixed_regime_occupancy": "loop_mixed_regime_occupancy.csv",
            "loop_transition_regime_occupancy": "loop_transition_regime_occupancy.csv",
            "c0_parent_readout": "c0_parent_readout.csv",
            "b0_state_summary": "b0_state_summary.csv",
            "b0_route_detail": "b0_route_detail.csv",
            "loop_context_refinement": "loop_context_refinement.csv",
            "loop_context_admissions": "loop_context_admissions.csv",
            "loop_context_blockers": "loop_context_blockers.csv",
            "atom_scorecard": "atom_scorecard.csv",
            "container_scorecard": "container_scorecard.csv",
            "loop_routing_detail": "loop_routing_detail.csv",
            "family_test_detail": "family_test_detail.csv",
            "concentration_warnings": "concentration_warnings.csv",
            "admission_candidates": "admission_candidates.csv",
            "blocker_candidates": "blocker_candidates.csv",
            "replay_results": "replay_results.csv",
            "family_r_replay_summary": "family_r_replay_summary.csv",
            "family_r_replay_scorecard": "family_r_replay_scorecard.csv",
            "family_r_replay_cost_sensitivity": "family_r_replay_cost_sensitivity.csv",
            "family_r_replay_selected_events": "family_r_replay_selected_events.csv",
        },
        "config": {
            **config.__dict__,
            "periods": _periods(config),
        },
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
            "clean_slate": True,
            "saved_rules_used": False,
            "seed_report_used": False,
            "yaml_rules_saved": False,
        },
    )
    _write_summary_md(
        paths["summary_markdown"],
        payload,
        behavior_loop_scorecard,
        loop_regimes,
        mixed_regimes,
        transition_regimes,
        c0_parent_readout,
        b0_summary,
        b0_route_detail,
        container_scorecard,
        routes,
        loop_context_admissions,
        loop_context_blockers,
        admissions,
        blockers,
        family_replay_scorecard,
    )

    return TemplateDiscoverySystemResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_markdown"],
        decision_json_path=paths["decision_json"],
        behavior_loop_scorecard_csv_path=paths["behavior_loop_scorecard"],
        loop_regime_occupancy_csv_path=paths["loop_regime_occupancy"],
        loop_mixed_regime_occupancy_csv_path=paths["loop_mixed_regime_occupancy"],
        loop_transition_regime_occupancy_csv_path=paths[
            "loop_transition_regime_occupancy"
        ],
        c0_parent_readout_csv_path=paths["c0_parent_readout"],
        b0_state_summary_csv_path=paths["b0_state_summary"],
        b0_route_detail_csv_path=paths["b0_route_detail"],
        loop_context_refinement_csv_path=paths["loop_context_refinement"],
        loop_context_admissions_csv_path=paths["loop_context_admissions"],
        loop_context_blockers_csv_path=paths["loop_context_blockers"],
        atom_scorecard_csv_path=paths["atom_scorecard"],
        container_scorecard_csv_path=paths["container_scorecard"],
        loop_routing_detail_csv_path=paths["loop_routing_detail"],
        family_test_detail_csv_path=paths["family_test_detail"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        admission_candidates_csv_path=paths["admission_candidates"],
        blocker_candidates_csv_path=paths["blocker_candidates"],
        replay_results_csv_path=paths["replay_results"],
        decision=decision,
    )


__all__ = [
    "AdmissionCandidate",
    "BlockerCandidate",
    "ContainerSpec",
    "ContextRefinementCandidate",
    "DirectionalReadout",
    "DiscoveryAtom",
    "LoopSpec",
    "ReplayReadout",
    "ReplaySpec",
    "SUPPORTED_MODES",
    "TemplateDiscoveryEventInput",
    "TemplateDiscoverySystemConfig",
    "TemplateDiscoverySystemResult",
    "run_template_discovery_system_lab",
]
