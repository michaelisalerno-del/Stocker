"""Walk-forward staged personality -> mixed-regime -> filter -> caveat -> exit lab.

This research-only layer consumes local Discovery output, sparse event rows, and
an optional caveat report. It freezes mixed-regime personality/filter rows from
Discovery, applies supported caveats before exit scoring, then reuses the
existing conservative stop/target replay mechanics. It does not fetch data,
touch broker/execution paths, or place orders.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.bad_trade_sequence_caveat_v0 import attach_prior_event_context
from stocker_research.personality_discovery_v0 import EVENT_STATE_PERSONALITY
from stocker_research.personality_live_replay_v0 import _add_missing_discovery_features
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    WalkForwardSelectedFilterExitConfig,
    _apply_filter_candidate,
    _concentration,
    _concentration_warnings,
    _daily_pnl,
    _dedupe_trades,
    _exit_summary,
    _materialize_combo,
    _month_bounds,
    _personality_summary,
    _random_month_baseline,
    _score_exit_model,
    _select_frozen_monthly_candidates,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/walk_forward_staged_mixed_regime_caveat_exit_v0")

STRICT_TRAIN_AND_OOS_SUPPORTED = "strict_train_and_oos_supported"
TRAIN_SELECTED_STAGED_SUPPORTED = "train_selected_staged_supported"

DEFAULT_COMBINED_REGIME_FIELDS: tuple[str, ...] = (
    "vwap_x_efficiency_regime",
    "vwap_x_range_regime",
    "compression_x_efficiency_regime",
    "opening_mid_x_range_regime",
    "time_x_vwap_regime",
    "volume_x_vwap_regime",
)

STAGED_NUMERIC_CAVEAT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("tight risk", "risk_bps", "<="),
    ("weak close", "close_location_value", "<="),
    ("weak current return zscore", "return_zscore", "<="),
    ("weak current bar return", "bar_return", "<="),
    ("already extended over prior 6 bars", "prior_6_bar_return", ">="),
    (
        "weak cross-stock same-direction confirmation",
        "same_direction_other_symbol_count_15m",
        "<=",
    ),
    (
        "weak cross-stock same-personality confirmation",
        "same_personality_other_symbol_count_15m",
        "<=",
    ),
    (
        "weak 30m cross-stock same-direction confirmation",
        "same_direction_other_symbol_count_30m",
        "<=",
    ),
    (
        "crowded 30m cross-stock same-direction confirmation",
        "same_direction_other_symbol_count_30m",
        ">=",
    ),
    (
        "weak 30m cross-stock same-personality confirmation",
        "same_personality_other_symbol_count_30m",
        "<=",
    ),
    ("high historical relative volume at bar index", "relative_volume_at_bar_index", ">="),
    ("high historical cumulative relative volume", "relative_cumulative_volume", ">="),
)

PERSONALITY_EVENT_STATE: dict[str, str] = {
    personality: event_state
    for event_state, (personality, _role, direction) in EVENT_STATE_PERSONALITY.items()
    if direction != 0
}
PERSONALITY_DIRECTION: dict[str, int] = {
    personality: direction
    for _event_state, (personality, _role, direction) in EVENT_STATE_PERSONALITY.items()
    if direction != 0
}


@dataclass(frozen=True)
class StagedMixedRegimeCaveatExitConfig:
    """Configuration for the staged mixed-regime/caveat/exit replay."""

    warmup_months: tuple[str, ...] = ()
    replay_months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    )
    combined_regime_fields: tuple[str, ...] = DEFAULT_COMBINED_REGIME_FIELDS
    mixed_regime_value_contains: tuple[str, ...] = ()
    allowed_caveat_statuses: tuple[str, ...] = (STRICT_TRAIN_AND_OOS_SUPPORTED,)
    stop_models: tuple[str, ...] = (
        "fixed_50bps",
        "fixed_75bps",
        "fixed_100bps",
        "structure_session_extreme_10bps",
        "structure_recent_extreme_10bps",
        "structure_opening_range_extreme_10bps",
    )
    target_r_multiples: tuple[float, ...] = (1.0, 1.5, 2.0)
    cost_bps: float = 10.0
    max_filters_per_personality: int = 4
    max_exit_candidates_per_month: int = 48
    max_selected_per_month: int = 18
    max_selected_per_personality_month: int = 3
    max_caveat_rules: int = 12
    max_staged_caveat_rules_per_month: int = 2
    min_train_events: int = 35
    min_train_symbols: int = 4
    min_train_months: int = 4
    min_symbol_train_events: int = 3
    min_symbol_train_total_net_r: float = 0.0
    min_symbol_train_win_rate: float = 0.0
    min_eligible_symbols: int = 1
    enable_personality_acceptance: bool = True
    min_personality_train_trades: int = 3
    min_personality_train_total_net_r: float = 0.0
    min_personality_train_win_rate: float = 0.0
    enable_prior_replay_personality_acceptance: bool = False
    min_prior_replay_personality_trades: int = 15
    min_prior_replay_personality_total_net_r: float = -1.0
    min_prior_replay_personality_win_rate: float = 0.0
    min_staged_caveat_train_trades: int = 35
    min_staged_caveat_flagged_trades: int = 5
    min_replay_signals: int = 1
    min_total_trades: int = 30
    min_positive_months: int = 1
    allow_sparse_quality_decision: bool = False
    min_sparse_total_trades: int = 15
    min_sparse_positive_months: int = 4
    min_sparse_win_rate: float = 0.65
    min_sparse_mean_net_r: float = 0.20
    max_sparse_single_month_share: float = 0.75
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    random_iterations: int = 100
    random_seed: int = 1337
    enable_staged_train_caveats: bool = True
    staged_caveat_numeric_quantiles: tuple[float, ...] = (0.20, 0.33, 0.50, 0.67, 0.80)


@dataclass(frozen=True)
class StagedMixedRegimeCaveatExitResult:
    """Paths and headline result for one staged mixed-regime run."""

    run_id: str
    input_event_dir: Path
    input_personality_discovery_dir: Path
    input_caveat_report_dir: Path | None
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    mixed_regime_filter_book_csv_path: Path
    caveat_rule_book_csv_path: Path
    personality_acceptance_csv_path: Path
    entry_policy_diagnostics_csv_path: Path
    monthly_exit_sweep_csv_path: Path
    selected_monthly_candidates_csv_path: Path
    monthly_summary_csv_path: Path
    random_monthly_baseline_csv_path: Path
    signals_csv_path: Path
    caveated_signals_csv_path: Path
    trades_csv_path: Path
    missed_signals_csv_path: Path
    daily_pnl_csv_path: Path
    personality_summary_csv_path: Path
    concentration_warnings_csv_path: Path
    decision: str
    trade_count: int


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


def _selected_exit_config(
    config: StagedMixedRegimeCaveatExitConfig,
) -> WalkForwardSelectedFilterExitConfig:
    return WalkForwardSelectedFilterExitConfig(
        replay_months=config.replay_months,
        stop_models=config.stop_models,
        target_r_multiples=config.target_r_multiples,
        cost_bps=config.cost_bps,
        max_exit_candidates_per_month=config.max_exit_candidates_per_month,
        max_selected_per_month=config.max_selected_per_month,
        max_selected_per_personality_month=config.max_selected_per_personality_month,
        min_train_events=config.min_train_events,
        min_train_symbols=config.min_train_symbols,
        min_train_months=config.min_train_months,
        min_replay_signals=config.min_replay_signals,
        min_total_trades=config.min_total_trades,
        min_positive_months=config.min_positive_months,
        max_single_symbol_share=config.max_single_symbol_share,
        max_single_session_share=config.max_single_session_share,
        max_single_month_share=config.max_single_month_share,
        random_iterations=config.random_iterations,
        random_seed=config.random_seed,
    )


def _sort_score_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "selection_score",
            "test_lift_vs_personality",
            "filtered_test_same_result_rate",
            "retained_test_count",
        ]
        if column in frame.columns
    ]


def _column_or_default(data: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in data:
        return data[column]
    return pd.Series(default, index=data.index)


def _load_mixed_regime_filter_book(
    input_personality_discovery_dir: Path,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
) -> pd.DataFrame:
    path = input_personality_discovery_dir / "passed_personality_rules.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing passed personality rules: {path}")
    data = pd.read_csv(path)
    required = {
        "personality",
        "horizon",
        "regime_field",
        "regime_value",
        "filter_rule",
        "feature",
        "operator",
        "threshold",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"passed_personality_rules.csv missing required columns: {missing}")
    if data.empty:
        return _empty_filter_book()

    if config.combined_regime_fields:
        data = data[
            data["regime_field"].astype(str).isin(set(config.combined_regime_fields))
        ].copy()
    if config.mixed_regime_value_contains:
        value_text = data["regime_value"].astype(str)
        mixed_mask = pd.Series(False, index=data.index)
        for needle in config.mixed_regime_value_contains:
            mixed_mask = mixed_mask | value_text.str.contains(re.escape(needle), na=False)
        data = data[mixed_mask].copy()
    data = data[data["personality"].astype(str).isin(PERSONALITY_EVENT_STATE)].copy()
    if data.empty:
        return _empty_filter_book()

    if "selection_score" not in data:
        data["selection_score"] = (
            pd.to_numeric(
                _column_or_default(data, "test_lift_vs_personality", 0.0),
                errors="coerce",
            ).fillna(0.0)
            * 100.0
            + pd.to_numeric(
                _column_or_default(data, "filtered_test_same_result_rate", 0.0),
                errors="coerce",
            ).fillna(0.0)
            * 10.0
            + pd.to_numeric(
                _column_or_default(data, "retained_test_count", 0),
                errors="coerce",
            ).fillna(0.0)
            * 0.001
        )

    output = pd.DataFrame(
        {
            "personality": data["personality"].astype(str),
            "event_state": data["personality"].astype(str).map(PERSONALITY_EVENT_STATE),
            "horizon": pd.to_numeric(data["horizon"], errors="coerce").astype("Int64"),
            "regime_field": data["regime_field"].astype(str),
            "regime_value": data["regime_value"].astype(str),
            "filter_feature": data["feature"].astype(str),
            "filter_operator": data["operator"].astype(str),
            "filter_threshold": pd.to_numeric(data["threshold"], errors="coerce"),
            "filter_rule": data["filter_rule"].astype(str),
            "rule_kind": _column_or_default(data, "rule_kind", "single").astype(str),
            "feature_b": _column_or_default(data, "feature_b", "").astype(str),
            "operator_b": _column_or_default(data, "operator_b", "").astype(str),
            "threshold_b": pd.to_numeric(
                _column_or_default(data, "threshold_b", np.nan),
                errors="coerce",
            ),
            "selection_score": pd.to_numeric(data["selection_score"], errors="coerce").fillna(0.0),
            "source_filtered_test_same_result_rate": pd.to_numeric(
                _column_or_default(data, "filtered_test_same_result_rate", np.nan),
                errors="coerce",
            ),
            "source_test_lift_vs_personality": pd.to_numeric(
                _column_or_default(data, "test_lift_vs_personality", np.nan),
                errors="coerce",
            ),
            "source_test_lift_vs_regime": pd.to_numeric(
                _column_or_default(data, "test_lift_vs_regime", np.nan),
                errors="coerce",
            ),
            "source_retained_test_count": pd.to_numeric(
                _column_or_default(data, "retained_test_count", np.nan),
                errors="coerce",
            ),
        }
    )
    output["expected_direction"] = output["personality"].map(PERSONALITY_DIRECTION).astype(int)
    output = output.dropna(subset=["horizon", "filter_threshold"])
    output["horizon"] = output["horizon"].astype(int)
    sort_columns = _sort_score_columns(output)
    if sort_columns:
        output = output.sort_values(sort_columns, ascending=False, kind="mergesort")
    output = (
        output.groupby("personality", as_index=False)
        .head(config.max_filters_per_personality)
        .reset_index(drop=True)
    )
    return output.loc[:, _filter_book_columns()]


def _filter_book_columns() -> list[str]:
    return [
        "personality",
        "event_state",
        "horizon",
        "expected_direction",
        "regime_field",
        "regime_value",
        "filter_feature",
        "filter_operator",
        "filter_threshold",
        "filter_rule",
        "rule_kind",
        "feature_b",
        "operator_b",
        "threshold_b",
        "selection_score",
        "source_filtered_test_same_result_rate",
        "source_test_lift_vs_personality",
        "source_test_lift_vs_regime",
        "source_retained_test_count",
    ]


def _empty_filter_book() -> pd.DataFrame:
    return pd.DataFrame(columns=_filter_book_columns())


def _load_caveat_rule_book(
    input_caveat_report_dir: Path | None,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
) -> pd.DataFrame:
    columns = [
        "caveat_rule_id",
        "rule_name",
        "rule_family",
        "strict_status",
        "current_personality",
        "prior_personality",
        "prior2_personality",
        "condition_feature",
        "condition_operator",
        "condition_value",
        "feature",
        "operator",
        "selected_threshold",
        "test_kept_lift_vs_base_r",
        "test_excess_vs_random_median_r",
    ]
    if input_caveat_report_dir is None:
        return pd.DataFrame(columns=columns)
    path = input_caveat_report_dir / "strict_validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing strict caveat results: {path}")
    data = pd.read_csv(path)
    required = {"rule_name", "rule_family", "strict_status"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"strict_validation_results.csv missing required columns: {missing}")
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data[data["strict_status"].astype(str).isin(config.allowed_caveat_statuses)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    for column in [
        "current_personality",
        "prior_personality",
        "prior2_personality",
        "condition_feature",
        "condition_operator",
        "condition_value",
        "feature",
        "operator",
    ]:
        if column not in data:
            data[column] = ""
    for column in [
        "selected_threshold",
        "test_kept_lift_vs_base_r",
        "test_excess_vs_random_median_r",
    ]:
        if column not in data:
            data[column] = math.nan
    data = data.sort_values(
        ["test_kept_lift_vs_base_r", "test_excess_vs_random_median_r"],
        ascending=False,
        kind="mergesort",
    ).head(config.max_caveat_rules)
    output = pd.DataFrame(
        {
            "caveat_rule_id": np.arange(len(data), dtype=int),
            "rule_name": data["rule_name"].astype(str),
            "rule_family": data["rule_family"].astype(str),
            "strict_status": data["strict_status"].astype(str),
            "current_personality": data["current_personality"].fillna("").astype(str),
            "prior_personality": data["prior_personality"].fillna("").astype(str),
            "prior2_personality": data["prior2_personality"].fillna("").astype(str),
            "condition_feature": data["condition_feature"].fillna("").astype(str),
            "condition_operator": data["condition_operator"].fillna("").astype(str),
            "condition_value": data["condition_value"].fillna("").astype(str),
            "feature": data["feature"].fillna("").astype(str),
            "operator": data["operator"].fillna("").astype(str),
            "selected_threshold": pd.to_numeric(data["selected_threshold"], errors="coerce"),
            "test_kept_lift_vs_base_r": pd.to_numeric(
                data["test_kept_lift_vs_base_r"],
                errors="coerce",
            ),
            "test_excess_vs_random_median_r": pd.to_numeric(
                data["test_excess_vs_random_median_r"],
                errors="coerce",
            ),
        }
    )
    return output.loc[:, columns].reset_index(drop=True)


def _caveat_book_columns() -> list[str]:
    return [
        "caveat_rule_id",
        "month",
        "rule_name",
        "rule_family",
        "strict_status",
        "current_personality",
        "prior_personality",
        "prior2_personality",
        "condition_feature",
        "condition_operator",
        "condition_value",
        "feature",
        "operator",
        "selected_threshold",
        "train_flagged_count",
        "train_flagged_total_net_r",
        "train_kept_total_net_r",
        "train_kept_lift_vs_base_r",
        "test_kept_lift_vs_base_r",
        "test_excess_vs_random_median_r",
        "caveat_source",
    ]


def _empty_caveat_book() -> pd.DataFrame:
    return pd.DataFrame(columns=_caveat_book_columns())


def _personality_acceptance_columns() -> list[str]:
    return [
        "month",
        "acceptance_source",
        "personality",
        "selected_candidate_count",
        "train_trade_count",
        "train_total_net_r",
        "train_mean_net_r",
        "train_win_rate",
        "accepted",
        "rejection_reason",
        "min_personality_train_trades",
        "min_personality_train_total_net_r",
        "min_personality_train_win_rate",
    ]


def _empty_personality_acceptance() -> pd.DataFrame:
    return pd.DataFrame(columns=_personality_acceptance_columns())


def _personality_acceptance_book(
    selected: pd.DataFrame,
    train_trades: pd.DataFrame,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
    month: str,
) -> pd.DataFrame:
    """Classify staged personalities using only the prior train replay."""

    if selected.empty or "personality" not in selected:
        return _empty_personality_acceptance()

    selected_counts = {
        str(personality): int(count)
        for personality, count in selected["personality"].astype(str).value_counts().items()
    }
    selected_personalities = sorted(selected_counts)
    stats: dict[str, dict[str, float]] = {}
    if not train_trades.empty and {"personality", "net_r"}.issubset(train_trades.columns):
        rows = train_trades.copy()
        rows["_net_r"] = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0)
        grouped = rows.groupby(rows["personality"].astype(str), dropna=False)["_net_r"]
        for group_personality, values in grouped:
            stats[str(group_personality)] = {
                "count": float(len(values)),
                "total": float(values.sum()),
                "mean": float(values.mean()) if len(values) else math.nan,
                "win_rate": float((values > 0.0).mean()) if len(values) else math.nan,
            }

    rows_out: list[dict[str, Any]] = []
    for personality in selected_personalities:
        row_stats = stats.get(personality, {})
        train_count = int(row_stats.get("count", 0.0))
        train_total = float(row_stats.get("total", 0.0))
        train_mean = float(row_stats.get("mean", math.nan))
        train_win_rate = float(row_stats.get("win_rate", math.nan))
        reasons: list[str] = []
        if config.enable_personality_acceptance:
            if train_count < config.min_personality_train_trades:
                reasons.append("low_train_trade_count")
            if train_total < config.min_personality_train_total_net_r:
                reasons.append("train_total_net_r_below_min")
            if (
                train_count >= config.min_personality_train_trades
                and not math.isnan(train_win_rate)
                and train_win_rate < config.min_personality_train_win_rate
            ):
                reasons.append("train_win_rate_below_min")
        rows_out.append(
            {
                "month": month,
                "acceptance_source": "train",
                "personality": personality,
                "selected_candidate_count": int(selected_counts[personality]),
                "train_trade_count": train_count,
                "train_total_net_r": train_total,
                "train_mean_net_r": train_mean,
                "train_win_rate": train_win_rate,
                "accepted": not reasons,
                "rejection_reason": "|".join(reasons),
                "min_personality_train_trades": int(config.min_personality_train_trades),
                "min_personality_train_total_net_r": float(
                    config.min_personality_train_total_net_r
                ),
                "min_personality_train_win_rate": float(config.min_personality_train_win_rate),
            }
        )
    return pd.DataFrame(rows_out, columns=_personality_acceptance_columns())


def _prior_replay_personality_acceptance_book(
    selected: pd.DataFrame,
    prior_replay_trades: pd.DataFrame,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
    month: str,
) -> pd.DataFrame:
    """Classify personalities using only earlier replay-month trades."""

    if selected.empty or "personality" not in selected:
        return _empty_personality_acceptance()

    selected_counts = {
        str(personality): int(count)
        for personality, count in selected["personality"].astype(str).value_counts().items()
    }
    selected_personalities = sorted(selected_counts)
    stats: dict[str, dict[str, float]] = {}
    if not prior_replay_trades.empty and {
        "personality",
        "net_r",
    }.issubset(prior_replay_trades.columns):
        rows = prior_replay_trades.copy()
        rows["_net_r"] = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0)
        grouped = rows.groupby(rows["personality"].astype(str), dropna=False)["_net_r"]
        for group_personality, values in grouped:
            stats[str(group_personality)] = {
                "count": float(len(values)),
                "total": float(values.sum()),
                "mean": float(values.mean()) if len(values) else math.nan,
                "win_rate": float((values > 0.0).mean()) if len(values) else math.nan,
            }

    warmup = not stats
    rows_out: list[dict[str, Any]] = []
    for personality in selected_personalities:
        row_stats = stats.get(personality, {})
        trade_count = int(row_stats.get("count", 0.0))
        total = float(row_stats.get("total", 0.0))
        mean = float(row_stats.get("mean", math.nan))
        win_rate = float(row_stats.get("win_rate", math.nan))
        reasons: list[str] = []
        accepted = True
        if not warmup:
            if trade_count < config.min_prior_replay_personality_trades:
                reasons.append("insufficient_prior_replay_sample_fallback_train")
            else:
                if total < config.min_prior_replay_personality_total_net_r:
                    reasons.append("prior_replay_total_net_r_below_min")
                    accepted = False
                if (
                    not math.isnan(win_rate)
                    and win_rate < config.min_prior_replay_personality_win_rate
                ):
                    reasons.append("prior_replay_win_rate_below_min")
                    accepted = False
        rows_out.append(
            {
                "month": month,
                "acceptance_source": "prior_replay_warmup" if warmup else "prior_replay",
                "personality": personality,
                "selected_candidate_count": int(selected_counts[personality]),
                "train_trade_count": trade_count,
                "train_total_net_r": total,
                "train_mean_net_r": mean,
                "train_win_rate": win_rate,
                "accepted": accepted,
                "rejection_reason": "|".join(reasons),
                "min_personality_train_trades": int(
                    config.min_prior_replay_personality_trades
                ),
                "min_personality_train_total_net_r": float(
                    config.min_prior_replay_personality_total_net_r
                ),
                "min_personality_train_win_rate": float(
                    config.min_prior_replay_personality_win_rate
                ),
            }
        )
    return pd.DataFrame(rows_out, columns=_personality_acceptance_columns())


def _apply_personality_acceptance(
    selected: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    if acceptance.empty or "accepted" not in acceptance or "personality" not in acceptance:
        return selected.iloc[0:0].copy()
    accepted = set(
        acceptance.loc[acceptance["accepted"].astype(bool), "personality"].astype(str)
    )
    return selected[selected["personality"].astype(str).isin(accepted)].copy()


def _numeric_mask(rows: pd.DataFrame, feature: str, operator: str, threshold: float) -> pd.Series:
    if feature not in rows:
        return pd.Series(False, index=rows.index)
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return (values <= threshold).fillna(False)
    if operator == ">=":
        return (values >= threshold).fillna(False)
    raise ValueError(f"Unsupported caveat operator: {operator}")


def _train_caveat_thresholds(
    rows: pd.DataFrame,
    feature: str,
    quantiles: tuple[float, ...],
) -> list[tuple[float, float]]:
    values = pd.to_numeric(rows[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.nunique() < 2:
        return []
    return sorted(
        {(float(values.quantile(quantile)), float(quantile)) for quantile in quantiles},
        key=lambda item: item[0],
    )


def _select_staged_train_caveat_book(
    train_trades: pd.DataFrame,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
    month: str,
) -> pd.DataFrame:
    if len(train_trades) < config.min_staged_caveat_train_trades or "net_r" not in train_trades:
        return _empty_caveat_book()
    rows = train_trades.copy()
    net_r = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0)
    base_total = float(net_r.sum())
    candidates: list[dict[str, Any]] = []

    def add_candidate(
        *,
        spec_index: int,
        label: str,
        feature: str,
        operator: str,
        threshold: float,
        quantile: float,
        mask: pd.Series,
        current_personality: str,
        scope_rank: int,
    ) -> None:
        flagged_count = int(mask.sum())
        if flagged_count < config.min_staged_caveat_flagged_trades:
            return
        flagged_total = float(net_r[mask].sum())
        kept_total = float(net_r[~mask].sum())
        kept_lift = kept_total - base_total
        if kept_lift <= 0.0 or flagged_total >= 0.0:
            return
        family = (
            "train_selected_personality_numeric"
            if current_personality
            else "train_selected_numeric"
        )
        prefix = f"{current_personality} " if current_personality else ""
        candidates.append(
            {
                "_mask": mask,
                "_spec_index": spec_index,
                "_scope_rank": scope_rank,
                "_selected_train_quantile": quantile,
                "rule_name": f"{prefix}{label}: {feature} {operator} {threshold:.6g}",
                "rule_family": family,
                "strict_status": TRAIN_SELECTED_STAGED_SUPPORTED,
                "current_personality": current_personality,
                "prior_personality": "",
                "prior2_personality": "",
                "feature": feature,
                "operator": operator,
                "selected_threshold": float(threshold),
                "train_flagged_count": flagged_count,
                "train_flagged_total_net_r": flagged_total,
                "train_kept_total_net_r": kept_total,
                "train_kept_lift_vs_base_r": kept_lift,
                "test_kept_lift_vs_base_r": math.nan,
                "test_excess_vs_random_median_r": math.nan,
                "caveat_source": (
                    "staged_train_personality" if current_personality else "staged_train"
                ),
                "month": month,
            }
        )

    for spec_index, (label, feature, operator) in enumerate(STAGED_NUMERIC_CAVEAT_SPECS):
        if feature not in rows:
            continue
        for threshold, quantile in _train_caveat_thresholds(
            rows,
            feature,
            config.staged_caveat_numeric_quantiles,
        ):
            add_candidate(
                spec_index=spec_index,
                label=label,
                feature=feature,
                operator=operator,
                threshold=threshold,
                quantile=quantile,
                mask=_numeric_mask(rows, feature, operator, threshold),
                current_personality="",
                scope_rank=0,
            )
        if "personality" not in rows or rows["personality"].astype(str).nunique() <= 1:
            continue
        for personality, personality_rows in rows.groupby(rows["personality"].astype(str)):
            if len(personality_rows) < config.min_staged_caveat_flagged_trades:
                continue
            for threshold, quantile in _train_caveat_thresholds(
                personality_rows,
                feature,
                config.staged_caveat_numeric_quantiles,
            ):
                numeric = _numeric_mask(rows, feature, operator, threshold)
                personality_mask = rows["personality"].astype(str).eq(str(personality))
                add_candidate(
                    spec_index=spec_index,
                    label=label,
                    feature=feature,
                    operator=operator,
                    threshold=threshold,
                    quantile=quantile,
                    mask=numeric & personality_mask,
                    current_personality=str(personality),
                    scope_rank=1,
                )
    if not candidates:
        return _empty_caveat_book()

    selected_rows: list[dict[str, Any]] = []
    already_blocked = pd.Series(False, index=rows.index)
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -float(item["train_kept_lift_vs_base_r"]),
            float(item["train_flagged_total_net_r"]),
            -int(item["train_flagged_count"]),
            int(item["_scope_rank"]),
            int(item["_spec_index"]),
        ),
    ):
        mask = candidate["_mask"] & ~already_blocked
        marginal_count = int(mask.sum())
        marginal_total = float(net_r[mask].sum())
        if marginal_count < config.min_staged_caveat_flagged_trades or marginal_total >= 0.0:
            continue
        row = {key: value for key, value in candidate.items() if not key.startswith("_")}
        row["train_marginal_flagged_count"] = marginal_count
        row["train_marginal_flagged_total_net_r"] = marginal_total
        selected_rows.append(row)
        already_blocked = already_blocked | candidate["_mask"]
        if len(selected_rows) >= config.max_staged_caveat_rules_per_month:
            break
    if not selected_rows:
        return _empty_caveat_book()
    result = pd.DataFrame(selected_rows)
    result["caveat_rule_id"] = np.arange(len(result), dtype=int)
    return result.reindex(columns=_caveat_book_columns() + [
        "train_marginal_flagged_count",
        "train_marginal_flagged_total_net_r",
    ])


def _combine_caveat_books(
    external: pd.DataFrame,
    staged: pd.DataFrame,
    *,
    month: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not external.empty:
        ext = external.copy()
        ext["month"] = month
        ext["caveat_source"] = "input_report"
        frames.append(ext)
    if not staged.empty:
        frames.append(staged.copy())
    if not frames:
        return _empty_caveat_book()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["caveat_rule_id"] = np.arange(len(combined), dtype=int)
    return combined


def _eligible_symbols_from_scored(
    scored: pd.DataFrame,
    *,
    config: StagedMixedRegimeCaveatExitConfig,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    stats = (
        scored.groupby("symbol", dropna=False)
        .agg(
            symbol_train_count=("net_r", "size"),
            symbol_train_total_net_r=("net_r", "sum"),
            symbol_train_win_rate=("net_r", lambda values: float((values > 0.0).mean())),
        )
        .reset_index()
    )
    return stats[
        (stats["symbol_train_count"] >= config.min_symbol_train_events)
        & (stats["symbol_train_total_net_r"] >= config.min_symbol_train_total_net_r)
        & (stats["symbol_train_win_rate"] >= config.min_symbol_train_win_rate)
    ].copy()


def _sort_exit_sweep_candidates(exit_sweep: pd.DataFrame) -> pd.DataFrame:
    return exit_sweep.sort_values(
        ["exit_selection_score", "train_exit_total_net_r"],
        ascending=[False, False],
        kind="mergesort",
    )


def _cap_exit_sweep_with_personality_floor(
    exit_sweep: pd.DataFrame,
    *,
    config: WalkForwardSelectedFilterExitConfig,
) -> pd.DataFrame:
    if exit_sweep.empty:
        return exit_sweep.copy()
    ranked = _sort_exit_sweep_candidates(exit_sweep)
    cap = int(config.max_exit_candidates_per_month)
    if len(ranked) <= cap:
        return ranked.reset_index(drop=True)

    floor_per_personality = max(1, int(config.max_selected_per_personality_month))
    priority = (
        ranked.groupby("personality", sort=False, group_keys=False)
        .head(floor_per_personality)
        .copy()
    )
    if len(priority) >= cap:
        return _sort_exit_sweep_candidates(priority).head(cap).reset_index(drop=True)

    remaining = ranked.drop(index=priority.index)
    capped = pd.concat([priority, remaining.head(cap - len(priority))], ignore_index=False)
    return _sort_exit_sweep_candidates(capped).reset_index(drop=True)


def _build_staged_exit_sweep(
    selected_filter_book: pd.DataFrame,
    train_events: pd.DataFrame,
    *,
    month: str,
    config: StagedMixedRegimeCaveatExitConfig,
    exit_config: WalkForwardSelectedFilterExitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selected_filter_rank, (_source_index, candidate) in enumerate(
        selected_filter_book.iterrows()
    ):
        combo_train = _materialize_combo(train_events, candidate)
        retained_train = _apply_filter_candidate(combo_train, candidate)
        if len(retained_train) < exit_config.min_train_events:
            continue
        conc = _concentration(retained_train)
        if conc["symbol_count"] < exit_config.min_train_symbols:
            continue
        if conc["month_count"] < exit_config.min_train_months:
            continue
        if conc["single_symbol_share"] > exit_config.max_single_symbol_share:
            continue
        if conc["single_session_share"] > exit_config.max_single_session_share:
            continue
        if conc["single_month_share"] > exit_config.max_single_month_share:
            continue
        train_end = pd.to_datetime(retained_train["timestamp"], utc=True, errors="coerce").max()
        for stop_model in exit_config.stop_models:
            for target_r in exit_config.target_r_multiples:
                scored = _score_exit_model(
                    retained_train,
                    horizon=int(candidate["horizon"]),
                    expected_direction=int(candidate["expected_direction"]),
                    stop_model=stop_model,
                    target_r=float(target_r),
                    cost_bps=exit_config.cost_bps,
                )
                stats = _exit_summary(scored, "train_exit")
                if stats["train_exit_count"] < exit_config.min_train_events:
                    continue
                if stats["train_exit_mean_net_r"] <= 0.0 or stats["train_exit_total_net_r"] <= 0.0:
                    continue
                if stats["train_exit_win_rate"] <= 0.50:
                    continue
                eligible = _eligible_symbols_from_scored(scored, config=config)
                if len(eligible) < config.min_eligible_symbols:
                    continue
                score = (
                    float(stats["train_exit_mean_net_r"])
                    + 0.01 * float(stats["train_exit_total_net_r"])
                    + 0.25 * (float(stats["train_exit_win_rate"]) - 0.50)
                    + 0.001 * float(candidate["selection_score"])
                    + 0.005 * float(len(eligible))
                )
                rows.append(
                    {
                        "month": month,
                        "selected_filter_rank": int(selected_filter_rank),
                        "personality": candidate["personality"],
                        "event_state": candidate["event_state"],
                        "horizon": int(candidate["horizon"]),
                        "expected_direction": int(candidate["expected_direction"]),
                        "regime_field": candidate["regime_field"],
                        "regime_value": candidate["regime_value"],
                        "filter_feature": candidate["filter_feature"],
                        "filter_operator": candidate["filter_operator"],
                        "filter_threshold": float(candidate["filter_threshold"]),
                        "filter_rule": candidate["filter_rule"],
                        "filter_selection_score": float(candidate["selection_score"]),
                        "train_end_timestamp": train_end.isoformat()
                        if pd.notna(train_end)
                        else "",
                        "stop_model": stop_model,
                        "target_r": float(target_r),
                        "eligible_symbols": "|".join(
                            sorted(eligible["symbol"].astype(str).unique())
                        ),
                        "eligible_symbol_count": int(eligible["symbol"].nunique()),
                        **stats,
                        "exit_selection_score": score,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return _cap_exit_sweep_with_personality_floor(result, config=exit_config)


def _split_personalities(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    text = str(value)
    if text in {"", "nan", "<NA>", "None"}:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


def _numeric_caveat_mask(rows: pd.DataFrame, caveat: pd.Series) -> pd.Series:
    feature = str(caveat.get("feature", ""))
    operator = str(caveat.get("operator", ""))
    threshold = float(caveat.get("selected_threshold", math.nan))
    if not feature or feature not in rows or math.isnan(threshold):
        return pd.Series(False, index=rows.index)
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return (values <= threshold).fillna(False)
    if operator == ">=":
        return (values >= threshold).fillna(False)
    return pd.Series(False, index=rows.index)


def _conditional_context_caveat_mask(rows: pd.DataFrame, caveat: pd.Series) -> pd.Series:
    condition_feature = str(caveat.get("condition_feature", "") or "")
    condition_operator = str(caveat.get("condition_operator", "==") or "==")
    condition_value = str(caveat.get("condition_value", "") or "")
    if not condition_feature or condition_feature not in rows:
        return pd.Series(False, index=rows.index)
    if condition_operator != "==":
        return pd.Series(False, index=rows.index)
    condition_mask = rows[condition_feature].astype(str).eq(condition_value)
    return condition_mask & _numeric_caveat_mask(rows, caveat)


def _composite_sequence_parts(rule_name: str) -> tuple[str, str, str] | None:
    match = re.match(r"sequence_caveat:\s*([^ ]+)\s+OR\s+([^-]+)->(.+)$", rule_name)
    if match is None:
        return None
    return (
        match.group(1).strip(),
        match.group(2).strip(),
        match.group(3).strip(),
    )


def _caveat_mask(rows: pd.DataFrame, caveat: pd.Series) -> pd.Series:
    family = str(caveat.get("rule_family", ""))
    rule_name = str(caveat.get("rule_name", ""))
    current_values = _split_personalities(caveat.get("current_personality", ""))
    prior = str(caveat.get("prior_personality", "") or "")
    prior2 = str(caveat.get("prior2_personality", "") or "")

    if family in {"train_selected_numeric", "train_selected_personality_numeric"}:
        mask = _numeric_caveat_mask(rows, caveat)
        if current_values:
            mask = mask & rows["personality"].astype(str).isin(current_values)
        return mask
    if family == "train_selected_conditional_context_numeric":
        mask = _conditional_context_caveat_mask(rows, caveat)
        if current_values:
            mask = mask & rows["personality"].astype(str).isin(current_values)
        return mask
    if family in {"fixed_personality_block", "fixed_current_group"}:
        if not current_values:
            return pd.Series(False, index=rows.index)
        return rows["personality"].astype(str).isin(current_values)
    if family == "fixed_prior_personality_sequence":
        if not prior or not current_values or "prev_event_personality" not in rows:
            return pd.Series(False, index=rows.index)
        return rows["prev_event_personality"].astype(str).eq(prior) & rows["personality"].astype(
            str
        ).isin(current_values)
    if family == "fixed_prior_two_personality_sequence":
        if (
            not prior
            or not prior2
            or not current_values
            or "prev_event_personality" not in rows
            or "prev2_event_personality" not in rows
        ):
            return pd.Series(False, index=rows.index)
        return (
            rows["prev2_event_personality"].astype(str).eq(prior2)
            & rows["prev_event_personality"].astype(str).eq(prior)
            & rows["personality"].astype(str).isin(current_values)
        )
    if family == "composite_sequence":
        parts = _composite_sequence_parts(rule_name)
        if parts is None or "prev_event_personality" not in rows:
            return pd.Series(False, index=rows.index)
        current_blocker, sequence_prior, sequence_current = parts
        return rows["personality"].astype(str).eq(current_blocker) | (
            rows["prev_event_personality"].astype(str).eq(sequence_prior)
            & rows["personality"].astype(str).eq(sequence_current)
        )
    return pd.Series(False, index=rows.index)


def _apply_caveat_rules(
    rows: pd.DataFrame,
    caveat_book: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty or caveat_book.empty:
        return rows.copy(), rows.iloc[0:0].copy()
    working = rows.copy()
    working["_caveat_blocked"] = False
    working["caveat_rule_id"] = math.nan
    working["caveat_rule_name"] = ""
    working["caveat_rule_family"] = ""
    working["caveat_strict_status"] = ""
    for _, caveat in caveat_book.iterrows():
        unmatched = ~working["_caveat_blocked"].astype(bool)
        mask = _caveat_mask(working, caveat) & unmatched
        if not bool(mask.any()):
            continue
        working.loc[mask, "_caveat_blocked"] = True
        working.loc[mask, "caveat_rule_id"] = int(caveat["caveat_rule_id"])
        working.loc[mask, "caveat_rule_name"] = str(caveat["rule_name"])
        working.loc[mask, "caveat_rule_family"] = str(caveat["rule_family"])
        working.loc[mask, "caveat_strict_status"] = str(caveat["strict_status"])
    blocked = working[working["_caveat_blocked"].astype(bool)].copy()
    passed = working[~working["_caveat_blocked"].astype(bool)].copy()
    passed = passed.drop(columns=["_caveat_blocked"])
    blocked = blocked.drop(columns=["_caveat_blocked"])
    return passed.reset_index(drop=True), blocked.reset_index(drop=True)


def _apply_symbol_eligibility(rows: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    eligible_symbols = _split_personalities(candidate.get("eligible_symbols", ""))
    if not eligible_symbols:
        return rows.copy()
    return rows[rows["symbol"].astype(str).isin(eligible_symbols)].copy()


def _apply_staged_monthly_candidates(
    selected: pd.DataFrame,
    selected_filter_book: pd.DataFrame,
    replay_events: pd.DataFrame,
    caveat_book: pd.DataFrame,
    *,
    config: WalkForwardSelectedFilterExitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_frames: list[pd.DataFrame] = []
    caveated_frames: list[pd.DataFrame] = []
    for _, candidate in selected.iterrows():
        selected_filter = selected_filter_book.iloc[int(candidate["selected_filter_rank"])]
        combo_replay = _materialize_combo(replay_events, selected_filter)
        retained = _apply_filter_candidate(combo_replay, selected_filter)
        retained = _apply_symbol_eligibility(retained, candidate)
        if len(retained) < config.min_replay_signals:
            continue
        event_context = replay_events.drop(
            columns=["personality", "role", "default_expected_direction"],
            errors="ignore",
        )
        enriched = attach_prior_event_context(retained, event_context)
        passed, caveated = _apply_caveat_rules(enriched, caveat_book)
        caveated_scored = pd.DataFrame()
        if not caveated.empty:
            caveated_scored = _score_exit_model(
                caveated,
                horizon=int(candidate["horizon"]),
                expected_direction=int(candidate["expected_direction"]),
                stop_model=str(candidate["stop_model"]),
                target_r=float(candidate["target_r"]),
                cost_bps=config.cost_bps,
            )
        scored = _score_exit_model(
            passed,
            horizon=int(candidate["horizon"]),
            expected_direction=int(candidate["expected_direction"]),
            stop_model=str(candidate["stop_model"]),
            target_r=float(candidate["target_r"]),
            cost_bps=config.cost_bps,
        )
        for frame in [scored, caveated_scored]:
            if frame.empty:
                continue
            frame["monthly_candidate_rank"] = int(candidate["monthly_candidate_rank"])
            frame["selected_filter_rank"] = int(candidate["selected_filter_rank"])
            for column in [
                "month",
                "personality",
                "regime_field",
                "regime_value",
                "filter_feature",
                "filter_operator",
                "filter_threshold",
                "filter_rule",
                "filter_selection_score",
                "exit_selection_score",
                "train_exit_count",
                "train_exit_total_net_r",
                "train_exit_mean_net_r",
                "train_exit_win_rate",
            ]:
                frame[column] = candidate[column]
        if not scored.empty:
            signal_frames.append(scored)
        if not caveated_scored.empty:
            caveated_scored["blocked_by_caveat"] = True
            caveated_frames.append(caveated_scored)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    caveated_signals = (
        pd.concat(caveated_frames, ignore_index=True) if caveated_frames else pd.DataFrame()
    )
    return signals, caveated_signals


def _one_per_symbol_session(rows: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    data = rows.copy()
    data["_ts"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if policy == "first_signal_per_symbol_session":
        ordered = data.sort_values(
            ["_ts", "symbol", "exit_selection_score"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        ordered = ordered.drop_duplicates(["symbol", "timestamp", "session_date"], keep="first")
        result = ordered.drop_duplicates(["symbol", "session_date"], keep="first")
    elif policy == "highest_prior_score_per_symbol_session":
        result = data.sort_values(
            ["symbol", "session_date", "exit_selection_score", "_ts"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).drop_duplicates(["symbol", "session_date"], keep="first")
    elif policy == "last_signal_per_symbol_session":
        result = data.sort_values(
            ["symbol", "session_date", "_ts", "exit_selection_score"],
            ascending=[True, True, True, False],
            kind="mergesort",
        ).drop_duplicates(["symbol", "session_date"], keep="last")
    else:
        raise ValueError(f"Unsupported entry policy: {policy}")
    return result.drop(columns=["_ts"]).reset_index(drop=True)


def _entry_policy_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "entry_policy",
        "trade_count",
        "symbol_count",
        "session_count",
        "month_count",
        "total_net_r",
        "median_net_r",
        "mean_net_r",
        "win_rate",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for policy in [
        "first_signal_per_symbol_session",
        "highest_prior_score_per_symbol_session",
        "last_signal_per_symbol_session",
    ]:
        selected = _one_per_symbol_session(signals, policy=policy)
        net_r = pd.to_numeric(selected["net_r"], errors="coerce")
        rows.append(
            {
                "entry_policy": policy,
                "trade_count": int(len(selected)),
                "symbol_count": int(selected["symbol"].nunique()) if "symbol" in selected else 0,
                "session_count": int(selected["session_date"].nunique())
                if "session_date" in selected
                else 0,
                "month_count": int(selected["month"].nunique()) if "month" in selected else 0,
                "total_net_r": float(net_r.sum()) if not selected.empty else 0.0,
                "median_net_r": float(net_r.median()) if not selected.empty else math.nan,
                "mean_net_r": float(net_r.mean()) if not selected.empty else math.nan,
                "win_rate": float((net_r > 0.0).mean()) if not selected.empty else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _dedupe_staged_trades(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return cast(tuple[pd.DataFrame, pd.DataFrame], _dedupe_trades(signals))


def _sparse_quality_reasons(
    *,
    trades: pd.DataFrame,
    total_net_r: float,
    positive_months: int,
    random_month_sum: float,
    concentration: dict[str, float],
    reject_reasons: list[str],
    config: StagedMixedRegimeCaveatExitConfig,
) -> tuple[bool, list[str]]:
    if not config.allow_sparse_quality_decision:
        return False, []
    if not reject_reasons or not set(reject_reasons).issubset(
        {"low_trade_count", "month_concentrated"}
    ):
        return False, []
    net_r = pd.to_numeric(trades["net_r"], errors="coerce").fillna(0.0)
    trade_count = int(len(trades))
    win_rate = float((net_r > 0.0).mean()) if trade_count else math.nan
    mean_net_r = float(net_r.mean()) if trade_count else math.nan
    failures: list[str] = []
    if trade_count < config.min_sparse_total_trades:
        failures.append("sparse_trade_count_below_min")
    if total_net_r <= 0.0:
        failures.append("sparse_total_net_r_not_positive")
    if math.isnan(random_month_sum):
        failures.append("sparse_random_baseline_missing")
    elif total_net_r <= random_month_sum:
        failures.append("sparse_random_baseline_not_beaten")
    if positive_months < config.min_sparse_positive_months:
        failures.append("sparse_positive_months_below_min")
    if math.isnan(win_rate) or win_rate < config.min_sparse_win_rate:
        failures.append("sparse_win_rate_below_min")
    if math.isnan(mean_net_r) or mean_net_r < config.min_sparse_mean_net_r:
        failures.append("sparse_mean_net_r_below_min")
    if concentration["single_month_share"] > config.max_sparse_single_month_share:
        failures.append("sparse_month_concentration_above_max")
    if failures:
        return False, failures
    warnings = [
        f"{reason}_sparse_allowed"
        for reason in reject_reasons
        if reason in {"low_trade_count", "month_concentrated"}
    ]
    return True, warnings


def _decision(
    *,
    trades: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    random_month_sum: float,
    config: StagedMixedRegimeCaveatExitConfig,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    total_net_r = float(trades["net_r"].sum()) if not trades.empty else 0.0
    positive_months = (
        int((monthly_summary["total_net_r"] > 0.0).sum()) if not monthly_summary.empty else 0
    )
    conc = _concentration(trades)
    if len(trades) < config.min_total_trades:
        reasons.append("low_trade_count")
    if total_net_r <= 0.0:
        reasons.append("negative_total_r")
    if positive_months < config.min_positive_months:
        reasons.append("too_few_positive_months")
    if not math.isnan(random_month_sum) and total_net_r <= random_month_sum:
        reasons.append("random_monthly_baseline_not_beaten")
    if conc["single_symbol_share"] > config.max_single_symbol_share:
        reasons.append("symbol_concentrated")
    if conc["single_session_share"] > config.max_single_session_share:
        reasons.append("session_concentrated")
    if conc["single_month_share"] > config.max_single_month_share:
        reasons.append("month_concentrated")
    if reasons:
        sparse_passed, sparse_reasons = _sparse_quality_reasons(
            trades=trades,
            total_net_r=total_net_r,
            positive_months=positive_months,
            random_month_sum=random_month_sum,
            concentration=conc,
            reject_reasons=reasons,
            config=config,
        )
        if sparse_passed:
            return "continue_research_sparse_high_quality", sparse_reasons
        reasons.extend(sparse_reasons)
        return f"reject_{reasons[0]}", reasons
    return "continue_research_staged_mixed_regime_caveat_exit", reasons


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows)
    lines = [
        "| " + " | ".join(str(column) for column in shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for _, row in shown.iterrows():
        values: list[str] = []
        for column in shown.columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_md(
    path: Path,
    *,
    payload: dict[str, Any],
    monthly_summary: pd.DataFrame,
    personality_summary: pd.DataFrame,
    filter_book: pd.DataFrame,
    caveat_book: pd.DataFrame,
) -> None:
    filter_cols = [
        column
        for column in [
            "personality",
            "event_state",
            "horizon",
            "regime_field",
            "regime_value",
            "filter_rule",
            "selection_score",
        ]
        if column in filter_book
    ]
    caveat_cols = [
        column
        for column in [
            "rule_name",
            "rule_family",
            "strict_status",
            "test_kept_lift_vs_base_r",
            "test_excess_vs_random_median_r",
        ]
        if column in caveat_book
    ]
    lines = [
        "# Walk-Forward Staged Mixed Regime Caveat Exit V0",
        "",
        (
            "Research-only staged replay: personality, then mixed regime, then filter, "
            "then caveat, then exit. Inputs are existing local report outputs and sparse "
            "event rows. No broker, IG, live trading, paper trading, vendor fetching, "
            "or order placement. No edge is claimed."
        ),
        "",
        f"Input event report: `{payload['input_event_dir']}`",
        f"Input personality discovery report: `{payload['input_personality_discovery_dir']}`",
        f"Input caveat report: `{payload.get('input_caveat_report_dir') or 'none'}`",
        f"Decision: `{payload['decision']}`",
        "Pipeline: `personality -> mixed_regime -> filter -> caveat -> exit`",
        f"Combined-regime fields: `{', '.join(payload['combined_regime_fields'])}`",
        (
            f"Optional regime-value match terms: "
            f"`{', '.join(payload['mixed_regime_value_contains'])}`"
        )
        if payload["mixed_regime_value_contains"]
        else "Optional regime-value match terms: `none`",
        f"Allowed caveat statuses: `{', '.join(payload['allowed_caveat_statuses'])}`",
        f"Train-only personality acceptance enabled: `{payload['enable_personality_acceptance']}`",
        (
            "Prior-replay personality acceptance enabled: "
            f"`{payload['enable_prior_replay_personality_acceptance']}`"
        ),
        f"Sparse-quality decision enabled: `{payload['allow_sparse_quality_decision']}`",
        f"Staged train caveats enabled: `{payload['enable_staged_train_caveats']}`",
        "Stop/target ordering: `conservative_stop_first_when_both_touched`",
        "Volume label: `historical_volume from existing local 5m OHLCV event report`",
        "",
        "## Headline",
        "",
        f"- Warmup months: `{', '.join(payload['warmup_months']) or 'none'}`",
        f"- Replay months: `{', '.join(payload['months'])}`",
        f"- Mixed-regime filter rows: `{payload['mixed_regime_filter_count']}`",
        (
            f"- Accepted/rejected personality-months: "
            f"`{payload['accepted_personality_count']}/{payload['rejected_personality_count']}`"
        ),
        f"- Caveat rules: `{payload['caveat_rule_count']}`",
        f"- Signals after caveat: `{payload['signal_count']}`",
        f"- Caveated signals: `{payload['caveated_signal_count']}`",
        f"- Trades: `{payload['trade_count']}`",
        f"- Total net R: `{payload['total_net_r']:.2f}`",
        f"- Win rate: `{payload['win_rate']:.1%}`"
        if not math.isnan(payload["win_rate"])
        else "- Win rate: `n/a`",
        f"- Positive months: `{payload['positive_month_count']}/{payload['month_count']}`",
        (
            "- Sum of monthly random median total R: "
            f"`{payload['random_monthly_median_total_net_r_sum']:.2f}`"
        )
        if not math.isnan(payload["random_monthly_median_total_net_r_sum"])
        else "- Sum of monthly random median total R: `n/a`",
        "",
        "## Monthly Summary",
        "",
        _markdown_table(monthly_summary, max_rows=24),
        "",
        "## Personality Summary",
        "",
        _markdown_table(personality_summary, max_rows=24),
        "",
        "## Mixed-Regime Filter Book",
        "",
        _markdown_table(filter_book[filter_cols] if filter_cols else filter_book, max_rows=60),
        "",
        "## Caveat Rule Book",
        "",
        _markdown_table(caveat_book[caveat_cols] if caveat_cols else caveat_book, max_rows=30),
        "",
        "## Caveat",
        "",
        (
            "This report is still an event-row replay using forward MFE/MAE targets. "
            "Caveats are research gates only, and by default only strict train+OOS "
            "supported caveats are applied."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_staged_mixed_regime_caveat_exit_lab(
    *,
    input_event_dir: Path,
    input_personality_discovery_dir: Path,
    input_caveat_report_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: StagedMixedRegimeCaveatExitConfig = StagedMixedRegimeCaveatExitConfig(),
) -> StagedMixedRegimeCaveatExitResult:
    """Run the staged mixed-regime/caveat/exit research replay."""

    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    event_rows = pd.read_csv(event_rows_path)
    events = _add_missing_discovery_features(event_rows).copy()
    events["_wf_timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    selected_filter_book = _load_mixed_regime_filter_book(
        input_personality_discovery_dir,
        config=config,
    )
    input_caveat_book = _load_caveat_rule_book(input_caveat_report_dir, config=config)
    exit_config = _selected_exit_config(config)

    all_exit_sweeps: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_caveat_books: list[pd.DataFrame] = []
    all_personality_acceptance: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_caveated_signals: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    prior_replay_trade_history: list[pd.DataFrame] = []
    all_missed: list[pd.DataFrame] = []
    all_random: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []

    processing_months = tuple(config.warmup_months) + tuple(config.replay_months)
    warmup_month_count = len(config.warmup_months)
    for month_index, month in enumerate(processing_months):
        is_warmup_month = month_index < warmup_month_count
        start, end = _month_bounds(month)
        train_events = events[events["_wf_timestamp"] < start].drop(columns=["_wf_timestamp"])
        replay_events = events[
            (events["_wf_timestamp"] >= start) & (events["_wf_timestamp"] < end)
        ].drop(columns=["_wf_timestamp"])
        exit_sweep = _build_staged_exit_sweep(
            selected_filter_book,
            train_events,
            month=month,
            config=config,
            exit_config=exit_config,
        )
        if not exit_sweep.empty:
            all_exit_sweeps.append(exit_sweep)
        selected = _select_frozen_monthly_candidates(exit_sweep, config=exit_config)
        initial_personality_count = (
            int(selected["personality"].nunique()) if not selected.empty else 0
        )
        month_acceptance_frames: list[pd.DataFrame] = []
        if config.enable_personality_acceptance and not selected.empty:
            train_signals_for_gate, _train_gate_caveated = _apply_staged_monthly_candidates(
                selected,
                selected_filter_book,
                train_events,
                _empty_caveat_book(),
                config=exit_config,
            )
            train_trades_for_gate, _train_gate_missed = _dedupe_staged_trades(
                train_signals_for_gate
            )
            train_acceptance = _personality_acceptance_book(
                selected,
                train_trades_for_gate,
                config=config,
                month=month,
            )
            selected = _apply_personality_acceptance(selected, train_acceptance)
            month_acceptance_frames.append(train_acceptance)
        if config.enable_prior_replay_personality_acceptance and not selected.empty:
            prior_replay_trades = (
                pd.concat(prior_replay_trade_history, ignore_index=True)
                if prior_replay_trade_history
                else pd.DataFrame()
            )
            prior_replay_acceptance = _prior_replay_personality_acceptance_book(
                selected,
                prior_replay_trades,
                config=config,
                month=month,
            )
            selected = _apply_personality_acceptance(selected, prior_replay_acceptance)
            month_acceptance_frames.append(prior_replay_acceptance)
        personality_acceptance = (
            pd.concat(month_acceptance_frames, ignore_index=True)
            if month_acceptance_frames
            else _empty_personality_acceptance()
        )
        if not personality_acceptance.empty:
            all_personality_acceptance.append(personality_acceptance)
        if not selected.empty and not is_warmup_month:
            all_selected.append(selected)
        staged_caveat_book = _empty_caveat_book()
        if config.enable_staged_train_caveats and not selected.empty:
            train_signals_for_caveats, _train_caveated = _apply_staged_monthly_candidates(
                selected,
                selected_filter_book,
                train_events,
                _empty_caveat_book(),
                config=exit_config,
            )
            train_trades_for_caveats, _train_missed = _dedupe_staged_trades(
                train_signals_for_caveats
            )
            staged_caveat_book = _select_staged_train_caveat_book(
                train_trades_for_caveats,
                config=config,
                month=month,
            )
        month_caveat_book = _combine_caveat_books(
            input_caveat_book,
            staged_caveat_book,
            month=month,
        )
        if not month_caveat_book.empty and not is_warmup_month:
            all_caveat_books.append(month_caveat_book)
        signals, caveated_signals = _apply_staged_monthly_candidates(
            selected,
            selected_filter_book,
            replay_events,
            month_caveat_book,
            config=exit_config,
        )
        trades, missed = _dedupe_staged_trades(signals)
        if not trades.empty:
            prior_replay_trade_history.append(trades)
        if is_warmup_month:
            continue
        random_baseline = _random_month_baseline(
            replay_events,
            trades,
            config=exit_config,
            seed=config.random_seed + (month_index - warmup_month_count) * 1009,
        )
        if not random_baseline.empty:
            random_baseline["month"] = month
            all_random.append(random_baseline)
        if not signals.empty:
            all_signals.append(signals)
        if not caveated_signals.empty:
            all_caveated_signals.append(caveated_signals)
        if not trades.empty:
            all_trades.append(trades)
        if not missed.empty:
            all_missed.append(missed)
        random_total = (
            float(random_baseline["total_net_r"].median())
            if not random_baseline.empty
            else math.nan
        )
        month_rows.append(
            {
                "month": month,
                "train_event_rows": int(len(train_events)),
                "replay_event_rows": int(len(replay_events)),
                "selected_candidate_count": int(len(selected)),
                "accepted_personality_count": int(selected["personality"].nunique())
                if not selected.empty
                else 0,
                "rejected_personality_count": max(
                    0,
                    initial_personality_count
                    - (int(selected["personality"].nunique()) if not selected.empty else 0),
                ),
                "active_caveat_rule_count": int(len(month_caveat_book)),
                "signal_count": int(len(signals)),
                "caveated_signal_count": int(len(caveated_signals)),
                "trade_count": int(len(trades)),
                "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
                "session_count": int(trades["session_date"].nunique()) if not trades.empty else 0,
                "total_net_r": float(trades["net_r"].sum()) if not trades.empty else 0.0,
                "median_net_r": float(trades["net_r"].median()) if not trades.empty else math.nan,
                "mean_net_r": float(trades["net_r"].mean()) if not trades.empty else math.nan,
                "win_rate": float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan,
                "random_median_total_net_r": random_total,
                "excess_vs_random_total_net_r": float(trades["net_r"].sum()) - random_total
                if not math.isnan(random_total)
                else math.nan,
            }
        )

    exit_sweep_frame = (
        pd.concat(all_exit_sweeps, ignore_index=True) if all_exit_sweeps else pd.DataFrame()
    )
    selected_frame = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    caveat_book = (
        pd.concat(all_caveat_books, ignore_index=True) if all_caveat_books else pd.DataFrame()
    )
    personality_acceptance = (
        pd.concat(all_personality_acceptance, ignore_index=True)
        if all_personality_acceptance
        else _empty_personality_acceptance()
    )
    signal_frame = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    caveated_signal_frame = (
        pd.concat(all_caveated_signals, ignore_index=True)
        if all_caveated_signals
        else pd.DataFrame()
    )
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    missed_frame = pd.concat(all_missed, ignore_index=True) if all_missed else pd.DataFrame()
    random_frame = pd.concat(all_random, ignore_index=True) if all_random else pd.DataFrame()
    monthly_summary = pd.DataFrame(month_rows)
    daily = _daily_pnl(trade_frame)
    personality = _personality_summary(trade_frame)
    concentration_warnings = _concentration_warnings(trade_frame, exit_config)
    entry_policy_diagnostics = _entry_policy_diagnostics(signal_frame)

    random_month_sum = (
        float(monthly_summary["random_median_total_net_r"].sum())
        if "random_median_total_net_r" in monthly_summary
        else math.nan
    )
    decision, decision_reasons = _decision(
        trades=trade_frame,
        monthly_summary=monthly_summary,
        random_month_sum=random_month_sum,
        config=config,
    )
    total_net_r = float(trade_frame["net_r"].sum()) if not trade_frame.empty else 0.0
    win_rate = float((trade_frame["net_r"] > 0.0).mean()) if not trade_frame.empty else math.nan
    positive_month_count = (
        int((monthly_summary["total_net_r"] > 0.0).sum()) if not monthly_summary.empty else 0
    )

    run_id = (
        "walk_forward_staged_mixed_regime_caveat_exit_v0_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "filter_book": run_dir / "mixed_regime_filter_book.csv",
        "caveat_book": run_dir / "caveat_rule_book.csv",
        "personality_acceptance": run_dir / "personality_acceptance.csv",
        "entry_policy_diagnostics": run_dir / "entry_policy_diagnostics.csv",
        "monthly_exit_sweep": run_dir / "monthly_exit_sweep.csv",
        "selected_monthly_candidates": run_dir / "selected_monthly_candidates.csv",
        "monthly_summary": run_dir / "monthly_summary.csv",
        "random_monthly_baseline": run_dir / "random_monthly_baseline.csv",
        "signals": run_dir / "signals.csv",
        "caveated_signals": run_dir / "caveated_signals.csv",
        "trades": run_dir / "trades.csv",
        "missed_signals": run_dir / "missed_signals.csv",
        "daily": run_dir / "daily_pnl.csv",
        "personality": run_dir / "personality_summary.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["filter_book"], selected_filter_book),
        (paths["caveat_book"], caveat_book),
        (paths["personality_acceptance"], personality_acceptance),
        (paths["entry_policy_diagnostics"], entry_policy_diagnostics),
        (paths["monthly_exit_sweep"], exit_sweep_frame),
        (paths["selected_monthly_candidates"], selected_frame),
        (paths["monthly_summary"], monthly_summary),
        (paths["random_monthly_baseline"], random_frame),
        (paths["signals"], signal_frame),
        (paths["caveated_signals"], caveated_signal_frame),
        (paths["trades"], trade_frame),
        (paths["missed_signals"], missed_frame),
        (paths["daily"], daily),
        (paths["personality"], personality),
        (paths["concentration"], concentration_warnings),
    ]:
        _write_csv(path, frame)

    payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "volume_label": "historical_volume from existing local 5m OHLCV event report",
        "input_event_dir": str(input_event_dir),
        "input_personality_discovery_dir": str(input_personality_discovery_dir),
        "input_caveat_report_dir": str(input_caveat_report_dir)
        if input_caveat_report_dir is not None
        else None,
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "pipeline": "personality -> mixed_regime -> filter -> caveat -> exit",
        "warmup_months": list(config.warmup_months),
        "months": list(config.replay_months),
        "combined_regime_fields": list(config.combined_regime_fields),
        "mixed_regime_value_contains": list(config.mixed_regime_value_contains),
        "allowed_caveat_statuses": list(config.allowed_caveat_statuses),
        "enable_personality_acceptance": bool(config.enable_personality_acceptance),
        "min_personality_train_trades": int(config.min_personality_train_trades),
        "min_personality_train_total_net_r": float(config.min_personality_train_total_net_r),
        "min_personality_train_win_rate": float(config.min_personality_train_win_rate),
        "enable_prior_replay_personality_acceptance": bool(
            config.enable_prior_replay_personality_acceptance
        ),
        "min_prior_replay_personality_trades": int(
            config.min_prior_replay_personality_trades
        ),
        "min_prior_replay_personality_total_net_r": float(
            config.min_prior_replay_personality_total_net_r
        ),
        "min_prior_replay_personality_win_rate": float(
            config.min_prior_replay_personality_win_rate
        ),
        "allow_sparse_quality_decision": bool(config.allow_sparse_quality_decision),
        "min_sparse_total_trades": int(config.min_sparse_total_trades),
        "min_sparse_positive_months": int(config.min_sparse_positive_months),
        "min_sparse_win_rate": float(config.min_sparse_win_rate),
        "min_sparse_mean_net_r": float(config.min_sparse_mean_net_r),
        "max_sparse_single_month_share": float(config.max_sparse_single_month_share),
        "enable_staged_train_caveats": bool(config.enable_staged_train_caveats),
        "mixed_regime_filter_count": int(len(selected_filter_book)),
        "caveat_rule_count": int(len(caveat_book)),
        "accepted_personality_count": int(
            monthly_summary["accepted_personality_count"].sum()
        )
        if "accepted_personality_count" in monthly_summary
        else 0,
        "rejected_personality_count": int(
            monthly_summary["rejected_personality_count"].sum()
        )
        if "rejected_personality_count" in monthly_summary
        else 0,
        "selected_candidate_count": int(len(selected_frame)),
        "signal_count": int(len(signal_frame)),
        "caveated_signal_count": int(len(caveated_signal_frame)),
        "trade_count": int(len(trade_frame)),
        "total_net_r": total_net_r,
        "win_rate": win_rate,
        "positive_month_count": positive_month_count,
        "month_count": int(len(monthly_summary)),
        "random_monthly_median_total_net_r_sum": random_month_sum,
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "decision_reasons": decision_reasons,
            "research_only": True,
            "edge_claimed": False,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "pipeline": payload["pipeline"],
            "allow_sparse_quality_decision": payload["allow_sparse_quality_decision"],
            "sparse_quality_thresholds": {
                "min_sparse_total_trades": payload["min_sparse_total_trades"],
                "min_sparse_positive_months": payload["min_sparse_positive_months"],
                "min_sparse_win_rate": payload["min_sparse_win_rate"],
                "min_sparse_mean_net_r": payload["min_sparse_mean_net_r"],
                "max_sparse_single_month_share": payload["max_sparse_single_month_share"],
            },
        },
    )
    _write_summary_md(
        paths["summary_md"],
        payload=payload,
        monthly_summary=monthly_summary,
        personality_summary=personality,
        filter_book=selected_filter_book,
        caveat_book=caveat_book,
    )

    return StagedMixedRegimeCaveatExitResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_personality_discovery_dir=input_personality_discovery_dir,
        input_caveat_report_dir=input_caveat_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        mixed_regime_filter_book_csv_path=paths["filter_book"],
        caveat_rule_book_csv_path=paths["caveat_book"],
        personality_acceptance_csv_path=paths["personality_acceptance"],
        entry_policy_diagnostics_csv_path=paths["entry_policy_diagnostics"],
        monthly_exit_sweep_csv_path=paths["monthly_exit_sweep"],
        selected_monthly_candidates_csv_path=paths["selected_monthly_candidates"],
        monthly_summary_csv_path=paths["monthly_summary"],
        random_monthly_baseline_csv_path=paths["random_monthly_baseline"],
        signals_csv_path=paths["signals"],
        caveated_signals_csv_path=paths["caveated_signals"],
        trades_csv_path=paths["trades"],
        missed_signals_csv_path=paths["missed_signals"],
        daily_pnl_csv_path=paths["daily"],
        personality_summary_csv_path=paths["personality"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        trade_count=int(len(trade_frame)),
    )
