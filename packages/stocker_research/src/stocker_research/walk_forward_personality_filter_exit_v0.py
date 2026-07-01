"""Walk-forward personality/regime/filter rediscovery with exit replay.

This research-only layer consumes sparse state-event rows and a prior combined
regime report. For each replay month it selects filter thresholds and
stop/target parameters from rows strictly before that month, then applies the
frozen candidates to the replay month. It does not fetch data, touch broker or
execution paths, or place orders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.personality_discovery_v0 import _return_column
from stocker_research.personality_live_replay_v0 import _add_missing_discovery_features
from stocker_research.personality_stop_validation_v0 import _risk_bps_for_model

DEFAULT_OUTPUT_DIR = Path("data/reports/research/walk_forward_personality_filter_exit_v0")
DEFAULT_SELECTED_OUTPUT_DIR = Path("data/reports/research/walk_forward_selected_filter_exit_v0")

EVENT_DIRECTIONS: dict[str, int] = {
    "controlled_pullback_after_bullish_impulse": 1,
    "liquidation_failed_low_reclaim": 1,
    "slow_snapback_after_dip": 1,
    "failed_bounce_active_liquidation": -1,
    "failed_bullish_impulse_recoil": -1,
    "failed_open_down_continuation": -1,
}

DEFAULT_FILTER_FEATURES: tuple[str, ...] = (
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
    "body_pct_of_range",
    "bar_return",
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "range_zscore",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "bar_index_in_session",
    "same_personality_other_symbol_count_15m",
    "same_direction_other_symbol_count_15m",
)


@dataclass(frozen=True)
class WalkForwardPersonalityFilterExitConfig:
    """Configuration for walk-forward rediscovery and exit replay."""

    replay_months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    )
    filter_features: tuple[str, ...] = DEFAULT_FILTER_FEATURES
    stop_models: tuple[str, ...] = (
        "fixed_50bps",
        "fixed_75bps",
        "fixed_100bps",
        "structure_session_extreme_10bps",
        "structure_recent_extreme_10bps",
        "structure_opening_range_extreme_10bps",
    )
    target_r_multiples: tuple[float, ...] = (1.0, 1.5, 2.0)
    quantiles: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)
    cost_bps: float = 10.0
    top_combos_per_personality: int = 5
    max_filter_candidates_per_combo: int = 4
    max_exit_candidates_per_month: int = 48
    max_selected_per_month: int = 12
    max_selected_per_personality_month: int = 3
    min_train_events: int = 35
    min_train_symbols: int = 4
    min_train_months: int = 4
    min_replay_signals: int = 1
    min_total_trades: int = 30
    min_positive_months: int = 1
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    random_iterations: int = 100
    random_seed: int = 1337


@dataclass(frozen=True)
class WalkForwardPersonalityFilterExitResult:
    """Paths and headline result for one walk-forward run."""

    run_id: str
    input_event_dir: Path
    input_combined_regime_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    monthly_filter_candidates_csv_path: Path
    monthly_exit_sweep_csv_path: Path
    selected_monthly_candidates_csv_path: Path
    monthly_summary_csv_path: Path
    random_monthly_baseline_csv_path: Path
    signals_csv_path: Path
    trades_csv_path: Path
    missed_signals_csv_path: Path
    daily_pnl_csv_path: Path
    personality_summary_csv_path: Path
    concentration_warnings_csv_path: Path
    decision: str
    trade_count: int


@dataclass(frozen=True)
class WalkForwardSelectedFilterExitConfig:
    """Configuration for replaying a frozen selected-filter caveat book."""

    replay_months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    )
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
    max_exit_candidates_per_month: int = 48
    max_selected_per_month: int = 18
    max_selected_per_personality_month: int = 3
    max_blocker_rules: int = 12
    min_train_events: int = 35
    min_train_symbols: int = 4
    min_train_months: int = 4
    min_replay_signals: int = 1
    min_total_trades: int = 30
    min_positive_months: int = 1
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    random_iterations: int = 100
    random_seed: int = 1337


@dataclass(frozen=True)
class WalkForwardSelectedFilterExitResult:
    """Paths and headline result for selected-filter walk-forward exit replay."""

    run_id: str
    input_event_dir: Path
    input_filter_report_dir: Path
    input_blocker_report_dir: Path | None
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    selected_filter_book_csv_path: Path
    monthly_exit_sweep_csv_path: Path
    selected_monthly_candidates_csv_path: Path
    monthly_summary_csv_path: Path
    random_monthly_baseline_csv_path: Path
    signals_csv_path: Path
    blocked_signals_csv_path: Path
    trades_csv_path: Path
    missed_signals_csv_path: Path
    daily_pnl_csv_path: Path
    personality_summary_csv_path: Path
    blocker_caveat_summary_csv_path: Path
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


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_combo_universe(
    input_combined_regime_dir: Path,
    *,
    top_per_personality: int,
) -> pd.DataFrame:
    path = input_combined_regime_dir / "best_combined_regimes_by_personality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing combined regime rows: {path}")
    data = pd.read_csv(path)
    if "combined_regime_pass" in data:
        data = data[_as_bool(data["combined_regime_pass"])].copy()
    if data.empty:
        return data
    sort_cols = [
        column
        for column in ("personality", "combined_score", "test_event_count")
        if column in data
    ]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    data = data.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return (
        data.groupby("personality", as_index=False)
        .head(top_per_personality)
        .reset_index(drop=True)
    )


def _filter_mask(
    rows: pd.DataFrame,
    *,
    feature: str,
    operator: str,
    threshold: float,
) -> pd.Series:
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return values <= threshold
    if operator == "<":
        return values < threshold
    if operator == ">=":
        return values >= threshold
    if operator == ">":
        return values > threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _candidate_thresholds(
    train: pd.DataFrame,
    feature: str,
    *,
    quantiles: tuple[float, ...],
) -> list[float]:
    values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.nunique() < 2:
        return []
    return sorted({float(values.quantile(q)) for q in quantiles})


def _month_series(rows: pd.DataFrame) -> pd.Series:
    if "month" in rows:
        return rows["month"].astype(str)
    if "session_date" in rows:
        return rows["session_date"].astype(str).str.slice(0, 7)
    return pd.Series("", index=rows.index)


def _concentration(rows: pd.DataFrame) -> dict[str, int | float]:
    if rows.empty:
        return {
            "symbol_count": 0,
            "single_symbol_share": math.nan,
            "session_count": 0,
            "single_session_share": math.nan,
            "month_count": 0,
            "single_month_share": math.nan,
        }
    symbol_counts = rows["symbol"].astype(str).value_counts()
    session_counts = rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1)
    session_counts = session_counts.value_counts()
    month_counts = _month_series(rows).value_counts()
    return {
        "symbol_count": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "session_count": int(session_counts.size),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "month_count": int(month_counts.size),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
    }


def _materialize_combo(events: pd.DataFrame, combo: pd.Series) -> pd.DataFrame:
    event_state = str(combo["event_state"])
    horizon = int(combo["horizon"])
    direction = int(EVENT_DIRECTIONS.get(event_state, int(combo.get("direction", 0))))
    regime_field = str(combo["regime_field"])
    regime_value = str(combo["regime_value"])
    ret_col = _return_column(horizon)
    if direction == 0 or regime_field not in events or ret_col not in events:
        return pd.DataFrame()
    rows = events[
        events["event_state"].astype(str).eq(event_state)
        & events[regime_field].astype(str).eq(regime_value)
        & events[ret_col].notna()
    ].copy()
    if rows.empty:
        return rows
    rows["horizon"] = horizon
    rows["expected_direction"] = direction
    rows["raw_return_bps"] = pd.to_numeric(rows[ret_col], errors="coerce") * 10000.0
    rows["aligned_return_bps"] = rows["raw_return_bps"] * direction
    rows["personality"] = str(combo["personality"])
    rows["regime_field"] = regime_field
    rows["regime_value"] = regime_value
    rows["combo_id"] = (
        str(combo["personality"])
        + "|"
        + event_state
        + "|"
        + str(horizon)
        + "|"
        + regime_field
        + "="
        + regime_value
    )
    return rows.reset_index(drop=True)


def _filter_summary(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if rows.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_median_aligned_return_bps": math.nan,
            f"{prefix}_mean_aligned_return_bps": math.nan,
            f"{prefix}_aligned_win_rate": math.nan,
            **{f"{prefix}_{key}": value for key, value in _concentration(rows).items()},
        }
    aligned = pd.to_numeric(rows["aligned_return_bps"], errors="coerce")
    return {
        f"{prefix}_count": int(len(rows)),
        f"{prefix}_median_aligned_return_bps": float(aligned.median()),
        f"{prefix}_mean_aligned_return_bps": float(aligned.mean()),
        f"{prefix}_aligned_win_rate": float((aligned > 0.0).mean()),
        **{f"{prefix}_{key}": value for key, value in _concentration(rows).items()},
    }


def _select_filter_candidates_for_combo(
    combo_train: pd.DataFrame,
    combo: pd.Series,
    *,
    month: str,
    config: WalkForwardPersonalityFilterExitConfig,
) -> pd.DataFrame:
    if len(combo_train) < config.min_train_events:
        return pd.DataFrame()
    combo_conc = _concentration(combo_train)
    if (
        combo_conc["symbol_count"] < config.min_train_symbols
        or combo_conc["month_count"] < config.min_train_months
    ):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    base = _filter_summary(combo_train, "base_train")
    train_end = pd.to_datetime(combo_train["timestamp"], utc=True, errors="coerce").max()
    for feature in config.filter_features:
        if feature not in combo_train:
            continue
        for threshold in _candidate_thresholds(combo_train, feature, quantiles=config.quantiles):
            for operator in ("<=", ">="):
                retained = combo_train[
                    _filter_mask(
                        combo_train,
                        feature=feature,
                        operator=operator,
                        threshold=threshold,
                    )
                ].copy()
                if len(retained) < config.min_train_events:
                    continue
                retained_conc = _concentration(retained)
                if retained_conc["symbol_count"] < config.min_train_symbols:
                    continue
                if retained_conc["month_count"] < config.min_train_months:
                    continue
                if retained_conc["single_symbol_share"] > config.max_single_symbol_share:
                    continue
                if retained_conc["single_session_share"] > config.max_single_session_share:
                    continue
                if retained_conc["single_month_share"] > config.max_single_month_share:
                    continue
                stats = _filter_summary(retained, "train")
                median_lift = (
                    stats["train_median_aligned_return_bps"]
                    - base["base_train_median_aligned_return_bps"]
                )
                win_lift = stats["train_aligned_win_rate"] - base["base_train_aligned_win_rate"]
                if median_lift <= 0.0 and win_lift <= 0.0:
                    continue
                score = (
                    0.02 * float(stats["train_mean_aligned_return_bps"])
                    + 0.03 * float(stats["train_median_aligned_return_bps"])
                    + 25.0 * float(stats["train_aligned_win_rate"] - 0.50)
                    + 0.01 * float(median_lift)
                )
                rows.append(
                    {
                        "month": month,
                        "personality": combo["personality"],
                        "event_state": combo["event_state"],
                        "horizon": int(combo["horizon"]),
                        "expected_direction": int(
                            EVENT_DIRECTIONS.get(
                                str(combo["event_state"]),
                                combo.get("direction", 0),
                            )
                        ),
                        "regime_field": combo["regime_field"],
                        "regime_value": combo["regime_value"],
                        "filter_feature": feature,
                        "filter_operator": operator,
                        "filter_threshold": threshold,
                        "filter_rule": f"{feature} {operator} {threshold:.6g}",
                        "train_end_timestamp": train_end.isoformat()
                        if pd.notna(train_end)
                        else "",
                        **base,
                        **stats,
                        "train_median_lift_bps": median_lift,
                        "train_win_rate_lift": win_lift,
                        "filter_selection_score": score,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["filter_selection_score", "train_count"],
        ascending=[False, False],
        kind="mergesort",
    ).head(config.max_filter_candidates_per_combo)


def _apply_filter_candidate(rows: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows[
        _filter_mask(
            rows,
            feature=str(candidate["filter_feature"]),
            operator=str(candidate["filter_operator"]),
            threshold=float(candidate["filter_threshold"]),
        )
    ].copy()


def _score_exit_model(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
    stop_model: str,
    target_r: float,
    cost_bps: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ret_col = _return_column(horizon)
    data = rows.copy()
    risk = _risk_bps_for_model(
        data,
        model_name=stop_model,
        expected_direction=expected_direction,
        structure_buffer_bps=10.0,
        min_structure_stop_bps=5.0,
    )
    risk = pd.to_numeric(risk, errors="coerce")
    risk = risk.where(risk > 0.0, np.nan)
    raw_return_bps = pd.to_numeric(data[ret_col], errors="coerce") * 10000.0
    mfe_bps = pd.to_numeric(data.get(f"forward_{horizon}_bar_mfe", np.nan), errors="coerce")
    mfe_bps = mfe_bps * 10000.0
    mae_bps = pd.to_numeric(data.get(f"forward_{horizon}_bar_mae", np.nan), errors="coerce")
    mae_bps = mae_bps * 10000.0
    favorable = mfe_bps if expected_direction > 0 else -mae_bps
    adverse = -mae_bps if expected_direction > 0 else mfe_bps
    aligned = raw_return_bps * expected_direction
    stop_hit = adverse >= risk
    target_hit = favorable >= risk * float(target_r)
    ambiguous = stop_hit & target_hit
    gross_r = aligned / risk
    gross_r = gross_r.where(~target_hit, float(target_r))
    gross_r = gross_r.where(~stop_hit, -1.0)
    gross_r = gross_r.where(~ambiguous, -1.0)
    cost_r = float(cost_bps) / risk
    data["stop_model"] = stop_model
    data["target_r"] = float(target_r)
    data["risk_bps"] = risk
    data["aligned_return_bps"] = aligned
    data["favorable_excursion_bps"] = favorable
    data["adverse_excursion_bps"] = adverse
    data["stop_hit"] = stop_hit.fillna(False).astype(bool)
    data["target_hit"] = target_hit.fillna(False).astype(bool)
    data["target_stop_order_ambiguous"] = ambiguous.fillna(False).astype(bool)
    data["gross_r"] = gross_r
    data["cost_bps"] = float(cost_bps)
    data["cost_r"] = cost_r
    data["net_r"] = gross_r - cost_r
    data["exit_reason"] = np.select(
        [ambiguous, stop_hit, target_hit],
        ["ambiguous_stop_first", "stop", "target"],
        default="time_exit",
    )
    net = pd.to_numeric(data["net_r"], errors="coerce")
    return data[np.isfinite(net)].reset_index(drop=True)


def _exit_summary(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if rows.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_total_net_r": 0.0,
            f"{prefix}_median_net_r": math.nan,
            f"{prefix}_mean_net_r": math.nan,
            f"{prefix}_win_rate": math.nan,
            f"{prefix}_stop_hit_rate": math.nan,
            f"{prefix}_target_hit_rate": math.nan,
            **{f"{prefix}_{key}": value for key, value in _concentration(rows).items()},
        }
    net_r = pd.to_numeric(rows["net_r"], errors="coerce")
    return {
        f"{prefix}_count": int(len(rows)),
        f"{prefix}_total_net_r": float(net_r.sum()),
        f"{prefix}_median_net_r": float(net_r.median()),
        f"{prefix}_mean_net_r": float(net_r.mean()),
        f"{prefix}_win_rate": float((net_r > 0.0).mean()),
        f"{prefix}_stop_hit_rate": float(rows["stop_hit"].astype(bool).mean()),
        f"{prefix}_target_hit_rate": float(rows["target_hit"].astype(bool).mean()),
        **{f"{prefix}_{key}": value for key, value in _concentration(rows).items()},
    }


def _build_exit_sweep(
    filter_candidates: pd.DataFrame,
    train_by_combo: dict[str, pd.DataFrame],
    *,
    config: WalkForwardPersonalityFilterExitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in filter_candidates.iterrows():
        combo_train = train_by_combo.get(str(candidate["combo_id"]), pd.DataFrame())
        retained_train = _apply_filter_candidate(combo_train, candidate)
        if len(retained_train) < config.min_train_events:
            continue
        horizon = int(candidate["horizon"])
        expected_direction = int(candidate["expected_direction"])
        for stop_model in config.stop_models:
            for target_r in config.target_r_multiples:
                scored = _score_exit_model(
                    retained_train,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    stop_model=stop_model,
                    target_r=float(target_r),
                    cost_bps=config.cost_bps,
                )
                stats = _exit_summary(scored, "train_exit")
                if stats["train_exit_count"] < config.min_train_events:
                    continue
                if stats["train_exit_mean_net_r"] <= 0.0 or stats["train_exit_total_net_r"] <= 0.0:
                    continue
                if stats["train_exit_win_rate"] <= 0.50:
                    continue
                score = (
                    float(stats["train_exit_mean_net_r"])
                    + 0.01 * float(stats["train_exit_total_net_r"])
                    + 0.25 * (float(stats["train_exit_win_rate"]) - 0.50)
                    + 0.001 * float(candidate["filter_selection_score"])
                )
                rows.append(
                    {
                        **candidate.to_dict(),
                        "stop_model": stop_model,
                        "target_r": float(target_r),
                        **stats,
                        "exit_selection_score": score,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["exit_selection_score", "train_exit_total_net_r"],
        ascending=[False, False],
        kind="mergesort",
    ).head(config.max_exit_candidates_per_month)


def _select_monthly_candidates(
    exit_sweep: pd.DataFrame,
    *,
    config: WalkForwardPersonalityFilterExitConfig,
) -> pd.DataFrame:
    if exit_sweep.empty:
        return exit_sweep.copy()
    selected = (
        exit_sweep.sort_values(
            ["personality", "exit_selection_score", "train_exit_count"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .groupby("personality", as_index=False)
        .head(config.max_selected_per_personality_month)
        .sort_values("exit_selection_score", ascending=False, kind="mergesort")
        .head(config.max_selected_per_month)
        .reset_index(drop=True)
    )
    selected["monthly_candidate_rank"] = np.arange(len(selected), dtype=int)
    return selected


def _apply_monthly_candidates(
    selected: pd.DataFrame,
    replay_by_combo: dict[str, pd.DataFrame],
    *,
    config: WalkForwardPersonalityFilterExitConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, candidate in selected.iterrows():
        replay_rows = replay_by_combo.get(str(candidate["combo_id"]), pd.DataFrame())
        retained = _apply_filter_candidate(replay_rows, candidate)
        if len(retained) < config.min_replay_signals:
            continue
        scored = _score_exit_model(
            retained,
            horizon=int(candidate["horizon"]),
            expected_direction=int(candidate["expected_direction"]),
            stop_model=str(candidate["stop_model"]),
            target_r=float(candidate["target_r"]),
            cost_bps=config.cost_bps,
        )
        if scored.empty:
            continue
        scored["monthly_candidate_rank"] = int(candidate["monthly_candidate_rank"])
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
            scored[column] = candidate[column]
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dedupe_trades(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return signals.copy(), signals.copy()
    ordered = signals.copy()
    ordered["_ts"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.sort_values(
        ["_ts", "symbol", "exit_selection_score"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    ordered = ordered.drop_duplicates(["symbol", "timestamp", "session_date"], keep="first")
    keep = ~ordered[["symbol", "session_date"]].astype(str).agg("|".join, axis=1).duplicated()
    trades = ordered[keep].drop(columns=["_ts"]).reset_index(drop=True)
    missed = ordered[~keep].drop(columns=["_ts"]).reset_index(drop=True)
    if not missed.empty:
        missed["miss_reason"] = "one_trade_per_symbol_session"
    return trades, missed


def _random_month_baseline(
    replay_events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    config: WalkForwardPersonalityFilterExitConfig,
    seed: int,
) -> pd.DataFrame:
    if replay_events.empty or trades.empty:
        return pd.DataFrame(
            columns=["iteration", "trade_count", "total_net_r", "median_net_r", "win_rate"]
        )
    rng = np.random.default_rng(seed)
    params = trades[["horizon", "expected_direction", "stop_model", "target_r"]].reset_index(
        drop=True
    )
    valid_pools = {
        horizon: replay_events[replay_events[_return_column(int(horizon))].notna()].copy()
        for horizon in sorted(params["horizon"].astype(int).unique())
        if _return_column(int(horizon)) in replay_events
    }
    rows: list[dict[str, Any]] = []
    for iteration in range(config.random_iterations):
        net_values: list[float] = []
        for _, param in params.iterrows():
            horizon = int(param["horizon"])
            pool = valid_pools.get(horizon)
            if pool is None or pool.empty:
                continue
            sample = pool.sample(
                n=1,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            scored = _score_exit_model(
                sample,
                horizon=horizon,
                expected_direction=int(param["expected_direction"]),
                stop_model=str(param["stop_model"]),
                target_r=float(param["target_r"]),
                cost_bps=config.cost_bps,
            )
            if not scored.empty:
                net_values.append(float(scored["net_r"].iloc[0]))
        if net_values:
            net = np.asarray(net_values, dtype=float)
            rows.append(
                {
                    "iteration": iteration,
                    "trade_count": int(len(net)),
                    "total_net_r": float(net.sum()),
                    "median_net_r": float(np.median(net)),
                    "win_rate": float((net > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _daily_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["session_date", "trade_count", "daily_net_r", "cumulative_net_r", "drawdown_r"]
        )
    daily = (
        trades.groupby("session_date")
        .agg(trade_count=("net_r", "size"), daily_net_r=("net_r", "sum"))
        .reset_index()
        .sort_values("session_date")
    )
    daily["cumulative_net_r"] = daily["daily_net_r"].cumsum()
    peak = daily["cumulative_net_r"].cummax().clip(lower=0.0)
    daily["drawdown_r"] = daily["cumulative_net_r"] - peak
    return daily


def _personality_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "personality",
                "trade_count",
                "symbol_count",
                "session_count",
                "total_net_r",
                "median_net_r",
                "mean_net_r",
                "win_rate",
                "stop_hit_rate",
                "target_hit_rate",
            ]
        )
    return (
        trades.groupby("personality")
        .agg(
            trade_count=("net_r", "size"),
            symbol_count=("symbol", "nunique"),
            session_count=("session_date", "nunique"),
            total_net_r=("net_r", "sum"),
            median_net_r=("net_r", "median"),
            mean_net_r=("net_r", "mean"),
            win_rate=("net_r", lambda values: float((values > 0.0).mean())),
            stop_hit_rate=("stop_hit", lambda values: float(pd.Series(values).astype(bool).mean())),
            target_hit_rate=(
                "target_hit",
                lambda values: float(pd.Series(values).astype(bool).mean()),
            ),
        )
        .reset_index()
        .sort_values("total_net_r", ascending=False, kind="mergesort")
    )


def _blocker_caveat_summary(blocked_signals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "blocker_rule_id",
        "blocker_personality",
        "blocker_filter_rule",
        "blocked_signal_count",
        "blocked_total_net_r",
        "blocked_median_net_r",
        "blocked_win_rate",
        "blocked_symbol_count",
        "blocked_single_symbol_share",
        "blocked_session_count",
        "blocked_single_session_share",
        "blocked_month_count",
        "blocked_single_month_share",
    ]
    if blocked_signals.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = blocked_signals.groupby(
        ["blocker_rule_id", "blocker_personality", "blocker_filter_rule"],
        dropna=False,
    )
    for key, group in grouped:
        net_r = pd.to_numeric(group["net_r"], errors="coerce")
        conc = _concentration(group)
        rows.append(
            {
                "blocker_rule_id": int(float(key[0])),
                "blocker_personality": str(key[1]),
                "blocker_filter_rule": str(key[2]),
                "blocked_signal_count": int(len(group)),
                "blocked_total_net_r": float(net_r.sum()),
                "blocked_median_net_r": float(net_r.median()),
                "blocked_win_rate": float((net_r > 0.0).mean()),
                "blocked_symbol_count": conc["symbol_count"],
                "blocked_single_symbol_share": conc["single_symbol_share"],
                "blocked_session_count": conc["session_count"],
                "blocked_single_session_share": conc["single_session_share"],
                "blocked_month_count": conc["month_count"],
                "blocked_single_month_share": conc["single_month_share"],
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["blocked_total_net_r", "blocked_signal_count"],
        ascending=[True, False],
        kind="mergesort",
    )


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


def _decision(
    *,
    trades: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    random_month_sum: float,
    config: WalkForwardPersonalityFilterExitConfig,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    total_net_r = float(trades["net_r"].sum()) if not trades.empty else 0.0
    positive_months = (
        int((monthly_summary["total_net_r"] > 0.0).sum())
        if not monthly_summary.empty
        else 0
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
        return f"reject_{reasons[0]}", reasons
    return "continue_research_walk_forward_filter_exit", reasons


def _concentration_warnings(
    trades: pd.DataFrame,
    config: WalkForwardPersonalityFilterExitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(columns=["scope", "key", "event_count", "warning"])
    for scope, columns, threshold in [
        ("symbol", ["symbol"], config.max_single_symbol_share),
        ("symbol_session", ["symbol", "session_date"], config.max_single_session_share),
        ("month", ["month"], config.max_single_month_share),
    ]:
        counts = trades[columns].astype(str).agg("|".join, axis=1).value_counts()
        if counts.empty:
            continue
        share = float(counts.iloc[0] / len(trades))
        if share > threshold:
            rows.append(
                {
                    "scope": scope,
                    "key": counts.index[0],
                    "event_count": int(counts.iloc[0]),
                    "share": share,
                    "threshold": threshold,
                    "warning": f"{scope}_concentration",
                }
            )
    return pd.DataFrame(rows)


def _write_summary_md(
    path: Path,
    *,
    input_event_dir: Path,
    input_combined_regime_dir: Path,
    decision: str,
    payload: dict[str, Any],
    monthly_summary: pd.DataFrame,
    personality_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    selected_cols = [
        column
        for column in [
            "month",
            "personality",
            "event_state",
            "horizon",
            "regime_field",
            "regime_value",
            "filter_rule",
            "stop_model",
            "target_r",
            "train_exit_count",
            "train_exit_total_net_r",
            "train_exit_win_rate",
            "exit_selection_score",
        ]
        if column in selected
    ]
    lines = [
        "# Walk-Forward Personality Filter Exit V0",
        "",
        (
            "Research-only month-by-month rediscovery and replay. For each replay month, "
            "filter thresholds and stop/target parameters are selected using only rows "
            "before that month, then applied to that month. No broker, IG, live trading, "
            "paper trading, vendor fetching, or order placement. No edge is claimed."
        ),
        "",
        f"Input event report: `{input_event_dir}`",
        f"Input combined-regime report: `{input_combined_regime_dir}`",
        f"Decision: `{decision}`",
        f"Cost: `{payload['cost_bps']}` bps",
        "Stop/target ordering: `conservative_stop_first_when_both_touched`",
        "Volume label: `historical_volume from existing local 5m OHLCV event report`",
        "",
        "## Headline",
        "",
        f"- Replay months: `{', '.join(payload['months'])}`",
        f"- Trades: `{payload['trade_count']}`",
        f"- Total net R: `{payload['total_net_r']:.2f}`",
        f"- Win rate: `{payload['win_rate']:.1%}`"
        if not math.isnan(payload["win_rate"])
        else "- Win rate: `n/a`",
        f"- Positive months: `{payload['positive_month_count']}/{payload['month_count']}`",
        f"- Max drawdown proxy: `{payload['max_drawdown_r']:.2f}R`"
        if not math.isnan(payload["max_drawdown_r"])
        else "- Max drawdown proxy: `n/a`",
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
        "## Selected Monthly Candidates",
        "",
        _markdown_table(selected[selected_cols] if selected_cols else selected, max_rows=120),
        "",
        "## Caveat",
        "",
        (
            "This is still an event-row replay using forward MFE/MAE targets. If target "
            "and stop are both touched inside the forward window, scoring is stop-first "
            "because intrabar ordering is unknown."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_selected_filter_book(input_filter_report_dir: Path) -> pd.DataFrame:
    path = input_filter_report_dir / "selected_filters.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected filters: {path}")
    data = pd.read_csv(path)
    required = {
        "personality",
        "event_state",
        "horizon",
        "regime_field",
        "regime_value",
        "filter_feature",
        "filter_operator",
        "filter_threshold",
        "filter_rule",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"selected_filters.csv missing required columns: {missing}")
    if "selection_score" not in data:
        data["selection_score"] = 0.0
    data["expected_direction"] = data["event_state"].astype(str).map(EVENT_DIRECTIONS).fillna(0)
    data = data[data["expected_direction"].astype(int).ne(0)].copy()
    return data.sort_values("selection_score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def _load_dead_chop_blocker_book(
    input_blocker_report_dir: Path | None,
    *,
    max_rules: int,
) -> pd.DataFrame:
    columns = [
        "blocker_rule_id",
        "blocker_personality",
        "blocker_event_state",
        "blocker_role",
        "blocker_horizon",
        "blocker_regime_field",
        "blocker_regime_value",
        "blocker_filter_feature",
        "blocker_filter_operator",
        "blocker_filter_threshold",
        "blocker_filter_rule",
        "blocker_selection_score",
    ]
    if input_blocker_report_dir is None:
        return pd.DataFrame(columns=columns)
    path = input_blocker_report_dir / "selected_sidelined_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sidelined selected candidates: {path}")
    data = pd.read_csv(path)
    if data.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "personality",
        "event_state",
        "regime_field",
        "regime_value",
        "feature",
        "operator",
        "threshold",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"selected_sidelined_candidates.csv missing required columns: {missing}")
    if "verdict" in data:
        data = data[data["verdict"].astype(str).eq("promote_for_retest")].copy()
    data = data[
        data["personality"].astype(str).eq("dead_chop_noise")
        & data["event_state"].astype(str).eq("dead_chop_blocker")
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    if "excess_vs_random_same_count" not in data:
        data["excess_vs_random_same_count"] = 0.0
    data = data.sort_values(
        ["excess_vs_random_same_count", "retained_test_count"]
        if "retained_test_count" in data
        else ["excess_vs_random_same_count"],
        ascending=False,
        kind="mergesort",
    ).head(max_rules)
    output = pd.DataFrame(
        {
            "blocker_rule_id": np.arange(len(data), dtype=int),
            "blocker_personality": data["personality"].astype(str).to_numpy(),
            "blocker_event_state": data["event_state"].astype(str).to_numpy(),
            "blocker_role": data.get("role", "no_trade_filter"),
            "blocker_horizon": data.get("horizon", 0),
            "blocker_regime_field": data["regime_field"].astype(str).to_numpy(),
            "blocker_regime_value": data["regime_value"].astype(str).to_numpy(),
            "blocker_filter_feature": data["feature"].astype(str).to_numpy(),
            "blocker_filter_operator": data["operator"].astype(str).to_numpy(),
            "blocker_filter_threshold": pd.to_numeric(
                data["threshold"],
                errors="coerce",
            ).to_numpy(),
            "blocker_filter_rule": data.get("filter_rule", "").astype(str).to_numpy()
            if "filter_rule" in data
            else data["feature"].astype(str).to_numpy(),
            "blocker_selection_score": pd.to_numeric(
                data["excess_vs_random_same_count"],
                errors="coerce",
            ).fillna(0.0).to_numpy(),
        }
    )
    return output.loc[:, columns].reset_index(drop=True)


def _signal_key(rows: pd.DataFrame) -> pd.Series:
    return rows["symbol"].astype(str) + "|" + rows["timestamp"].astype(str)


def _build_dead_chop_blocker_hits(
    replay_events: pd.DataFrame,
    blocker_book: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "_block_key",
        "blocker_rule_id",
        "blocker_personality",
        "blocker_event_state",
        "blocker_filter_rule",
        "blocker_regime_field",
        "blocker_regime_value",
        "blocker_match_scope",
    ]
    if replay_events.empty or blocker_book.empty:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    dead_chop = replay_events[replay_events["event_state"].astype(str).eq("dead_chop_blocker")]
    if dead_chop.empty:
        return pd.DataFrame(columns=columns)
    for _, blocker in blocker_book.iterrows():
        regime_field = str(blocker["blocker_regime_field"])
        feature = str(blocker["blocker_filter_feature"])
        if regime_field not in dead_chop or feature not in dead_chop:
            continue
        rows = dead_chop[
            dead_chop[regime_field].astype(str).eq(str(blocker["blocker_regime_value"]))
        ].copy()
        if rows.empty:
            continue
        rows = rows[
            _filter_mask(
                rows,
                feature=feature,
                operator=str(blocker["blocker_filter_operator"]),
                threshold=float(blocker["blocker_filter_threshold"]),
            )
        ].copy()
        if rows.empty:
            continue
        rows["_block_key"] = _signal_key(rows)
        rows["blocker_rule_id"] = int(blocker["blocker_rule_id"])
        rows["blocker_personality"] = str(blocker["blocker_personality"])
        rows["blocker_event_state"] = str(blocker["blocker_event_state"])
        rows["blocker_filter_rule"] = str(blocker["blocker_filter_rule"])
        rows["blocker_regime_field"] = regime_field
        rows["blocker_regime_value"] = str(blocker["blocker_regime_value"])
        rows["blocker_match_scope"] = "dead_chop_event_key"
        frames.append(rows.loc[:, columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    hits = pd.concat(frames, ignore_index=True)
    hits = hits.sort_values(["_block_key", "blocker_rule_id"], kind="mergesort")
    return hits.drop_duplicates("_block_key", keep="first").reset_index(drop=True)


def _build_signal_condition_blocker_hits(
    signal_rows: pd.DataFrame,
    blocker_book: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "_block_key",
        "blocker_rule_id",
        "blocker_personality",
        "blocker_event_state",
        "blocker_filter_rule",
        "blocker_regime_field",
        "blocker_regime_value",
        "blocker_match_scope",
    ]
    if signal_rows.empty or blocker_book.empty:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    for _, blocker in blocker_book.iterrows():
        regime_field = str(blocker["blocker_regime_field"])
        feature = str(blocker["blocker_filter_feature"])
        if regime_field not in signal_rows or feature not in signal_rows:
            continue
        rows = signal_rows[
            signal_rows[regime_field].astype(str).eq(str(blocker["blocker_regime_value"]))
        ].copy()
        if rows.empty:
            continue
        rows = rows[
            _filter_mask(
                rows,
                feature=feature,
                operator=str(blocker["blocker_filter_operator"]),
                threshold=float(blocker["blocker_filter_threshold"]),
            )
        ].copy()
        if rows.empty:
            continue
        rows["_block_key"] = _signal_key(rows)
        rows["blocker_rule_id"] = int(blocker["blocker_rule_id"])
        rows["blocker_personality"] = str(blocker["blocker_personality"])
        rows["blocker_event_state"] = str(blocker["blocker_event_state"])
        rows["blocker_filter_rule"] = str(blocker["blocker_filter_rule"])
        rows["blocker_regime_field"] = regime_field
        rows["blocker_regime_value"] = str(blocker["blocker_regime_value"])
        rows["blocker_match_scope"] = "signal_condition"
        frames.append(rows.loc[:, columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    hits = pd.concat(frames, ignore_index=True)
    hits = hits.sort_values(["_block_key", "blocker_rule_id"], kind="mergesort")
    return hits.drop_duplicates("_block_key", keep="first").reset_index(drop=True)


def _apply_dead_chop_blockers(
    rows: pd.DataFrame,
    replay_events: pd.DataFrame,
    blocker_book: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty or blocker_book.empty:
        return rows.copy(), rows.iloc[0:0].copy()
    strict_hits = _build_dead_chop_blocker_hits(replay_events, blocker_book)
    signal_hits = _build_signal_condition_blocker_hits(rows, blocker_book)
    hit_frames = [frame for frame in (strict_hits, signal_hits) if not frame.empty]
    hits = pd.concat(hit_frames, ignore_index=True) if hit_frames else pd.DataFrame()
    if hits.empty:
        return rows.copy(), rows.iloc[0:0].copy()
    hits = hits.sort_values(["_block_key", "blocker_rule_id"], kind="mergesort")
    hits = hits.drop_duplicates("_block_key", keep="first")
    working = rows.copy()
    working["_block_key"] = _signal_key(working)
    merged = working.merge(hits, on="_block_key", how="left")
    blocked = merged[merged["blocker_rule_id"].notna()].copy()
    passed = merged[merged["blocker_rule_id"].isna()].copy()
    drop_cols = [column for column in ["_block_key"] if column in passed]
    passed = passed.drop(columns=drop_cols)
    blocked = blocked.drop(columns=drop_cols)
    return passed.reset_index(drop=True), blocked.reset_index(drop=True)


def _build_selected_exit_sweep(
    selected_filter_book: pd.DataFrame,
    train_events: pd.DataFrame,
    *,
    month: str,
    config: WalkForwardSelectedFilterExitConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selected_filter_rank, candidate in selected_filter_book.iterrows():
        combo_train = _materialize_combo(train_events, candidate)
        retained_train = _apply_filter_candidate(combo_train, candidate)
        if len(retained_train) < config.min_train_events:
            continue
        conc = _concentration(retained_train)
        if conc["symbol_count"] < config.min_train_symbols:
            continue
        if conc["month_count"] < config.min_train_months:
            continue
        if conc["single_symbol_share"] > config.max_single_symbol_share:
            continue
        if conc["single_session_share"] > config.max_single_session_share:
            continue
        if conc["single_month_share"] > config.max_single_month_share:
            continue
        train_end = pd.to_datetime(retained_train["timestamp"], utc=True, errors="coerce").max()
        for stop_model in config.stop_models:
            for target_r in config.target_r_multiples:
                scored = _score_exit_model(
                    retained_train,
                    horizon=int(candidate["horizon"]),
                    expected_direction=int(candidate["expected_direction"]),
                    stop_model=stop_model,
                    target_r=float(target_r),
                    cost_bps=config.cost_bps,
                )
                stats = _exit_summary(scored, "train_exit")
                if stats["train_exit_count"] < config.min_train_events:
                    continue
                if stats["train_exit_mean_net_r"] <= 0.0 or stats["train_exit_total_net_r"] <= 0.0:
                    continue
                if stats["train_exit_win_rate"] <= 0.50:
                    continue
                score = (
                    float(stats["train_exit_mean_net_r"])
                    + 0.01 * float(stats["train_exit_total_net_r"])
                    + 0.25 * (float(stats["train_exit_win_rate"]) - 0.50)
                    + 0.001 * float(candidate["selection_score"])
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
                        **stats,
                        "exit_selection_score": score,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["exit_selection_score", "train_exit_total_net_r"],
        ascending=[False, False],
        kind="mergesort",
    ).head(config.max_exit_candidates_per_month)


def _select_frozen_monthly_candidates(
    exit_sweep: pd.DataFrame,
    *,
    config: WalkForwardSelectedFilterExitConfig,
) -> pd.DataFrame:
    if exit_sweep.empty:
        return exit_sweep.copy()
    selected = (
        exit_sweep.sort_values(
            ["personality", "exit_selection_score", "train_exit_count"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .groupby("personality", as_index=False)
        .head(config.max_selected_per_personality_month)
        .sort_values("exit_selection_score", ascending=False, kind="mergesort")
        .head(config.max_selected_per_month)
        .reset_index(drop=True)
    )
    selected["monthly_candidate_rank"] = np.arange(len(selected), dtype=int)
    return selected


def _apply_frozen_monthly_candidates(
    selected: pd.DataFrame,
    selected_filter_book: pd.DataFrame,
    replay_events: pd.DataFrame,
    *,
    blocker_book: pd.DataFrame,
    config: WalkForwardSelectedFilterExitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    blocked_frames: list[pd.DataFrame] = []
    for _, candidate in selected.iterrows():
        selected_filter = selected_filter_book.iloc[int(candidate["selected_filter_rank"])]
        combo_replay = _materialize_combo(replay_events, selected_filter)
        retained = _apply_filter_candidate(combo_replay, selected_filter)
        if len(retained) < config.min_replay_signals:
            continue
        retained, blocked = _apply_dead_chop_blockers(retained, replay_events, blocker_book)
        blocked_scored = pd.DataFrame()
        if not blocked.empty:
            blocked_scored = _score_exit_model(
                blocked,
                horizon=int(candidate["horizon"]),
                expected_direction=int(candidate["expected_direction"]),
                stop_model=str(candidate["stop_model"]),
                target_r=float(candidate["target_r"]),
                cost_bps=config.cost_bps,
            )
            if not blocked_scored.empty:
                blocked_scored["blocked_by_dead_chop"] = True
        scored = _score_exit_model(
            retained,
            horizon=int(candidate["horizon"]),
            expected_direction=int(candidate["expected_direction"]),
            stop_model=str(candidate["stop_model"]),
            target_r=float(candidate["target_r"]),
            cost_bps=config.cost_bps,
        )
        if not scored.empty:
            scored["monthly_candidate_rank"] = int(candidate["monthly_candidate_rank"])
            scored["selected_filter_rank"] = int(candidate["selected_filter_rank"])
        if scored.empty and blocked_scored.empty:
            continue
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
            if not scored.empty:
                scored[column] = candidate[column]
            if not blocked_scored.empty:
                blocked_scored[column] = candidate[column]
        if not scored.empty:
            frames.append(scored)
        if not blocked_scored.empty:
            blocked_scored["monthly_candidate_rank"] = int(candidate["monthly_candidate_rank"])
            blocked_scored["selected_filter_rank"] = int(candidate["selected_filter_rank"])
            blocked_frames.append(blocked_scored)
    signals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    blocked_signals = (
        pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()
    )
    return signals, blocked_signals


def _write_selected_summary_md(
    path: Path,
    *,
    input_event_dir: Path,
    input_filter_report_dir: Path,
    decision: str,
    payload: dict[str, Any],
    monthly_summary: pd.DataFrame,
    personality_summary: pd.DataFrame,
    blocker_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    selected_cols = [
        column
        for column in [
            "month",
            "personality",
            "event_state",
            "horizon",
            "regime_field",
            "regime_value",
            "filter_rule",
            "stop_model",
            "target_r",
            "train_exit_count",
            "train_exit_total_net_r",
            "train_exit_win_rate",
            "exit_selection_score",
        ]
        if column in selected
    ]
    lines = [
        "# Walk-Forward Selected Filter Exit V0",
        "",
        (
            "Research-only month-by-month replay over a frozen selected-filter caveat "
            "book. Filter rules are loaded from selected_filters.csv. For each replay "
            "month, only stop/target parameters are selected from rows before that "
            "month. No broker, IG, live trading, paper trading, vendor fetching, or "
            "order placement. No edge is claimed."
        ),
        "",
        f"Input event report: `{input_event_dir}`",
        f"Input selected-filter report: `{input_filter_report_dir}`",
        f"Input blocker report: `{payload.get('input_blocker_report_dir') or 'none'}`",
        f"Decision: `{decision}`",
        f"Cost: `{payload['cost_bps']}` bps",
        "Stop/target ordering: `conservative_stop_first_when_both_touched`",
        "Volume label: `historical_volume from existing local 5m OHLCV event report`",
        "",
        "## Headline",
        "",
        f"- Replay months: `{', '.join(payload['months'])}`",
        f"- Trades: `{payload['trade_count']}`",
        f"- Blocked signals: `{payload.get('blocked_signal_count', 0)}`",
        f"- Dead-chop blocker rules: `{payload.get('dead_chop_blocker_rule_count', 0)}`",
        f"- Total net R: `{payload['total_net_r']:.2f}`",
        f"- Win rate: `{payload['win_rate']:.1%}`"
        if not math.isnan(payload["win_rate"])
        else "- Win rate: `n/a`",
        f"- Positive months: `{payload['positive_month_count']}/{payload['month_count']}`",
        f"- Max drawdown proxy: `{payload['max_drawdown_r']:.2f}R`"
        if not math.isnan(payload["max_drawdown_r"])
        else "- Max drawdown proxy: `n/a`",
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
        "## Dead-Chop Blocker Summary",
        "",
        _markdown_table(blocker_summary, max_rows=24),
        "",
        "## Selected Monthly Candidates",
        "",
        _markdown_table(selected[selected_cols] if selected_cols else selected, max_rows=120),
        "",
        "## Caveat",
        "",
        (
            "This path deliberately keeps the selected filters fixed. It tests whether "
            "those curated caveats survive when exits are selected from prior data only. "
            "When a blocker report is supplied, dead-chop rows are used only as "
            "current-bar no-trade caveats and blocked rows are scored separately."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_walk_forward_selected_filter_exit_lab(
    *,
    input_event_dir: Path,
    input_filter_report_dir: Path,
    input_blocker_report_dir: Path | None = None,
    output_dir: Path = DEFAULT_SELECTED_OUTPUT_DIR,
    config: WalkForwardSelectedFilterExitConfig = WalkForwardSelectedFilterExitConfig(),
) -> WalkForwardSelectedFilterExitResult:
    """Replay a frozen selected-filter caveat book with prior-only exit selection."""

    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    event_rows = pd.read_csv(event_rows_path)
    events = _add_missing_discovery_features(event_rows).copy()
    events["_wf_timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    selected_filter_book = _load_selected_filter_book(input_filter_report_dir)
    blocker_book = _load_dead_chop_blocker_book(
        input_blocker_report_dir,
        max_rules=config.max_blocker_rules,
    )

    all_exit_sweeps: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_blocked_signals: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_missed: list[pd.DataFrame] = []
    all_random: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []

    for month_index, month in enumerate(config.replay_months):
        start, end = _month_bounds(month)
        train_events = events[events["_wf_timestamp"] < start].drop(columns=["_wf_timestamp"])
        replay_events = events[
            (events["_wf_timestamp"] >= start) & (events["_wf_timestamp"] < end)
        ].drop(columns=["_wf_timestamp"])
        exit_sweep = _build_selected_exit_sweep(
            selected_filter_book,
            train_events,
            month=month,
            config=config,
        )
        if not exit_sweep.empty:
            all_exit_sweeps.append(exit_sweep)
        selected = _select_frozen_monthly_candidates(exit_sweep, config=config)
        if not selected.empty:
            all_selected.append(selected)
        signals, blocked_signals = _apply_frozen_monthly_candidates(
            selected,
            selected_filter_book,
            replay_events,
            blocker_book=blocker_book,
            config=config,
        )
        trades, missed = _dedupe_trades(signals)
        random_baseline = _random_month_baseline(
            replay_events,
            trades,
            config=config,
            seed=config.random_seed + month_index * 1009,
        )
        if not random_baseline.empty:
            random_baseline["month"] = month
            all_random.append(random_baseline)
        if not signals.empty:
            all_signals.append(signals)
        if not blocked_signals.empty:
            all_blocked_signals.append(blocked_signals)
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
                "blocked_signal_count": int(len(blocked_signals)),
                "signal_count": int(len(signals)),
                "trade_count": int(len(trades)),
                "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
                "session_count": int(trades["session_date"].nunique()) if not trades.empty else 0,
                "total_net_r": float(trades["net_r"].sum()) if not trades.empty else 0.0,
                "median_net_r": float(trades["net_r"].median()) if not trades.empty else math.nan,
                "mean_net_r": float(trades["net_r"].mean()) if not trades.empty else math.nan,
                "win_rate": float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan,
                "stop_hit_rate": float(trades["stop_hit"].astype(bool).mean())
                if not trades.empty
                else math.nan,
                "target_hit_rate": float(trades["target_hit"].astype(bool).mean())
                if not trades.empty
                else math.nan,
                "random_median_total_net_r": random_total,
                "excess_vs_random_total_net_r": float(trades["net_r"].sum()) - random_total
                if not math.isnan(random_total)
                else math.nan,
            }
        )

    selected_book_frame = selected_filter_book.copy()
    exit_sweep_frame = (
        pd.concat(all_exit_sweeps, ignore_index=True) if all_exit_sweeps else pd.DataFrame()
    )
    selected_frame = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    signal_frame = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    blocked_signal_frame = (
        pd.concat(all_blocked_signals, ignore_index=True)
        if all_blocked_signals
        else pd.DataFrame()
    )
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    missed_frame = pd.concat(all_missed, ignore_index=True) if all_missed else pd.DataFrame()
    random_frame = pd.concat(all_random, ignore_index=True) if all_random else pd.DataFrame()
    monthly_summary = pd.DataFrame(month_rows)
    daily = _daily_pnl(trade_frame)
    personality = _personality_summary(trade_frame)
    blocker_summary = _blocker_caveat_summary(blocked_signal_frame)
    concentration_warnings = _concentration_warnings(trade_frame, config)

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
    if decision == "continue_research_walk_forward_filter_exit":
        decision = "continue_research_walk_forward_selected_filter_exit"
    total_net_r = float(trade_frame["net_r"].sum()) if not trade_frame.empty else 0.0
    win_rate = float((trade_frame["net_r"] > 0.0).mean()) if not trade_frame.empty else math.nan
    positive_month_count = (
        int((monthly_summary["total_net_r"] > 0.0).sum()) if not monthly_summary.empty else 0
    )
    max_drawdown = float(daily["drawdown_r"].min()) if not daily.empty else math.nan
    conc = _concentration(trade_frame)

    run_id = (
        "walk_forward_selected_filter_exit_v0_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "selected_filter_book": run_dir / "selected_filter_book.csv",
        "monthly_exit_sweep": run_dir / "monthly_exit_sweep.csv",
        "selected_monthly_candidates": run_dir / "selected_monthly_candidates.csv",
        "monthly_summary": run_dir / "monthly_summary.csv",
        "random_monthly_baseline": run_dir / "random_monthly_baseline.csv",
        "signals": run_dir / "signals.csv",
        "blocked_signals": run_dir / "blocked_signals.csv",
        "trades": run_dir / "trades.csv",
        "missed_signals": run_dir / "missed_signals.csv",
        "daily": run_dir / "daily_pnl.csv",
        "personality": run_dir / "personality_summary.csv",
        "blocker_summary": run_dir / "blocker_caveat_summary.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["selected_filter_book"], selected_book_frame),
        (paths["monthly_exit_sweep"], exit_sweep_frame),
        (paths["selected_monthly_candidates"], selected_frame),
        (paths["monthly_summary"], monthly_summary),
        (paths["random_monthly_baseline"], random_frame),
        (paths["signals"], signal_frame),
        (paths["blocked_signals"], blocked_signal_frame),
        (paths["trades"], trade_frame),
        (paths["missed_signals"], missed_frame),
        (paths["daily"], daily),
        (paths["personality"], personality),
        (paths["blocker_summary"], blocker_summary),
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
        "input_filter_report_dir": str(input_filter_report_dir),
        "input_blocker_report_dir": str(input_blocker_report_dir)
        if input_blocker_report_dir is not None
        else None,
        "run_id": run_id,
        "output_dir": str(run_dir),
        "months": list(config.replay_months),
        "cost_bps": float(config.cost_bps),
        "stop_target_ordering": "conservative_stop_first_when_both_touched",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "selected_filter_count": int(len(selected_book_frame)),
        "dead_chop_blocker_rule_count": int(len(blocker_book)),
        "exit_candidate_count": int(len(exit_sweep_frame)),
        "selected_candidate_count": int(len(selected_frame)),
        "signal_count": int(len(signal_frame)),
        "blocked_signal_count": int(len(blocked_signal_frame)),
        "trade_count": int(len(trade_frame)),
        "total_net_r": total_net_r,
        "win_rate": win_rate,
        "positive_month_count": positive_month_count,
        "month_count": int(len(monthly_summary)),
        "max_drawdown_r": max_drawdown,
        "random_monthly_median_total_net_r_sum": random_month_sum,
        **{f"aggregate_{key}": value for key, value in conc.items()},
    }
    _write_json(paths["summary_json"], payload)
    _write_json(paths["decision_json"], payload)
    _write_selected_summary_md(
        paths["summary_md"],
        input_event_dir=input_event_dir,
        input_filter_report_dir=input_filter_report_dir,
        decision=decision,
        payload=payload,
        monthly_summary=monthly_summary,
        personality_summary=personality,
        blocker_summary=blocker_summary,
        selected=selected_frame,
    )
    return WalkForwardSelectedFilterExitResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_filter_report_dir=input_filter_report_dir,
        input_blocker_report_dir=input_blocker_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        selected_filter_book_csv_path=paths["selected_filter_book"],
        monthly_exit_sweep_csv_path=paths["monthly_exit_sweep"],
        selected_monthly_candidates_csv_path=paths["selected_monthly_candidates"],
        monthly_summary_csv_path=paths["monthly_summary"],
        random_monthly_baseline_csv_path=paths["random_monthly_baseline"],
        signals_csv_path=paths["signals"],
        blocked_signals_csv_path=paths["blocked_signals"],
        trades_csv_path=paths["trades"],
        missed_signals_csv_path=paths["missed_signals"],
        daily_pnl_csv_path=paths["daily"],
        personality_summary_csv_path=paths["personality"],
        blocker_caveat_summary_csv_path=paths["blocker_summary"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        trade_count=int(len(trade_frame)),
    )


def run_walk_forward_personality_filter_exit_lab(
    *,
    input_event_dir: Path,
    input_combined_regime_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: WalkForwardPersonalityFilterExitConfig = WalkForwardPersonalityFilterExitConfig(),
) -> WalkForwardPersonalityFilterExitResult:
    """Run prior-only monthly filter rediscovery and conservative exit replay."""

    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    event_rows = pd.read_csv(event_rows_path)
    events = _add_missing_discovery_features(event_rows).copy()
    events["_wf_timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    combos = _load_combo_universe(
        input_combined_regime_dir,
        top_per_personality=config.top_combos_per_personality,
    )

    all_filter_candidates: list[pd.DataFrame] = []
    all_exit_sweeps: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_missed: list[pd.DataFrame] = []
    all_random: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []

    for month_index, month in enumerate(config.replay_months):
        start, end = _month_bounds(month)
        train_events = events[events["_wf_timestamp"] < start].drop(columns=["_wf_timestamp"])
        replay_events = events[
            (events["_wf_timestamp"] >= start) & (events["_wf_timestamp"] < end)
        ].drop(columns=["_wf_timestamp"])

        train_by_combo: dict[str, pd.DataFrame] = {}
        replay_by_combo: dict[str, pd.DataFrame] = {}
        filter_frames: list[pd.DataFrame] = []
        for _, combo in combos.iterrows():
            combo_train = _materialize_combo(train_events, combo)
            if combo_train.empty:
                continue
            combo_id = str(combo_train["combo_id"].iloc[0])
            train_by_combo[combo_id] = combo_train
            replay_by_combo[combo_id] = _materialize_combo(replay_events, combo)
            filters = _select_filter_candidates_for_combo(
                combo_train,
                combo,
                month=month,
                config=config,
            )
            if not filters.empty:
                filters["combo_id"] = combo_id
                filter_frames.append(filters)

        filter_candidates = (
            pd.concat(filter_frames, ignore_index=True) if filter_frames else pd.DataFrame()
        )
        if not filter_candidates.empty:
            all_filter_candidates.append(filter_candidates)
        exit_sweep = _build_exit_sweep(filter_candidates, train_by_combo, config=config)
        if not exit_sweep.empty:
            all_exit_sweeps.append(exit_sweep)
        selected = _select_monthly_candidates(exit_sweep, config=config)
        if not selected.empty:
            all_selected.append(selected)
        signals = _apply_monthly_candidates(selected, replay_by_combo, config=config)
        trades, missed = _dedupe_trades(signals)
        random_baseline = _random_month_baseline(
            replay_events,
            trades,
            config=config,
            seed=config.random_seed + month_index * 1009,
        )
        if not random_baseline.empty:
            random_baseline["month"] = month
            all_random.append(random_baseline)
        if not signals.empty:
            all_signals.append(signals)
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
                "filter_candidate_count": int(len(filter_candidates)),
                "exit_candidate_count": int(len(exit_sweep)),
                "selected_candidate_count": int(len(selected)),
                "signal_count": int(len(signals)),
                "trade_count": int(len(trades)),
                "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
                "session_count": int(trades["session_date"].nunique()) if not trades.empty else 0,
                "total_net_r": float(trades["net_r"].sum()) if not trades.empty else 0.0,
                "median_net_r": float(trades["net_r"].median()) if not trades.empty else math.nan,
                "mean_net_r": float(trades["net_r"].mean()) if not trades.empty else math.nan,
                "win_rate": float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan,
                "stop_hit_rate": float(trades["stop_hit"].astype(bool).mean())
                if not trades.empty
                else math.nan,
                "target_hit_rate": float(trades["target_hit"].astype(bool).mean())
                if not trades.empty
                else math.nan,
                "random_median_total_net_r": random_total,
                "excess_vs_random_total_net_r": float(trades["net_r"].sum()) - random_total
                if not math.isnan(random_total)
                else math.nan,
            }
        )

    filter_candidate_frame = (
        pd.concat(all_filter_candidates, ignore_index=True)
        if all_filter_candidates
        else pd.DataFrame()
    )
    exit_sweep_frame = (
        pd.concat(all_exit_sweeps, ignore_index=True) if all_exit_sweeps else pd.DataFrame()
    )
    selected_frame = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    signal_frame = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    missed_frame = pd.concat(all_missed, ignore_index=True) if all_missed else pd.DataFrame()
    random_frame = pd.concat(all_random, ignore_index=True) if all_random else pd.DataFrame()
    monthly_summary = pd.DataFrame(month_rows)
    daily = _daily_pnl(trade_frame)
    personality = _personality_summary(trade_frame)
    concentration_warnings = _concentration_warnings(trade_frame, config)

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
    max_drawdown = float(daily["drawdown_r"].min()) if not daily.empty else math.nan
    conc = _concentration(trade_frame)

    run_id = (
        "walk_forward_personality_filter_exit_v0_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "monthly_filter_candidates": run_dir / "monthly_filter_candidates.csv",
        "monthly_exit_sweep": run_dir / "monthly_exit_sweep.csv",
        "selected_monthly_candidates": run_dir / "selected_monthly_candidates.csv",
        "monthly_summary": run_dir / "monthly_summary.csv",
        "random_monthly_baseline": run_dir / "random_monthly_baseline.csv",
        "signals": run_dir / "signals.csv",
        "trades": run_dir / "trades.csv",
        "missed_signals": run_dir / "missed_signals.csv",
        "daily": run_dir / "daily_pnl.csv",
        "personality": run_dir / "personality_summary.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["monthly_filter_candidates"], filter_candidate_frame),
        (paths["monthly_exit_sweep"], exit_sweep_frame),
        (paths["selected_monthly_candidates"], selected_frame),
        (paths["monthly_summary"], monthly_summary),
        (paths["random_monthly_baseline"], random_frame),
        (paths["signals"], signal_frame),
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
        "input_combined_regime_dir": str(input_combined_regime_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "months": list(config.replay_months),
        "cost_bps": float(config.cost_bps),
        "stop_target_ordering": "conservative_stop_first_when_both_touched",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "filter_candidate_count": int(len(filter_candidate_frame)),
        "exit_candidate_count": int(len(exit_sweep_frame)),
        "selected_candidate_count": int(len(selected_frame)),
        "signal_count": int(len(signal_frame)),
        "trade_count": int(len(trade_frame)),
        "total_net_r": total_net_r,
        "win_rate": win_rate,
        "positive_month_count": positive_month_count,
        "month_count": int(len(monthly_summary)),
        "max_drawdown_r": max_drawdown,
        "random_monthly_median_total_net_r_sum": random_month_sum,
        **{f"aggregate_{key}": value for key, value in conc.items()},
    }
    _write_json(paths["summary_json"], payload)
    _write_json(paths["decision_json"], payload)
    _write_summary_md(
        paths["summary_md"],
        input_event_dir=input_event_dir,
        input_combined_regime_dir=input_combined_regime_dir,
        decision=decision,
        payload=payload,
        monthly_summary=monthly_summary,
        personality_summary=personality,
        selected=selected_frame,
    )
    return WalkForwardPersonalityFilterExitResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_combined_regime_dir=input_combined_regime_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        monthly_filter_candidates_csv_path=paths["monthly_filter_candidates"],
        monthly_exit_sweep_csv_path=paths["monthly_exit_sweep"],
        selected_monthly_candidates_csv_path=paths["selected_monthly_candidates"],
        monthly_summary_csv_path=paths["monthly_summary"],
        random_monthly_baseline_csv_path=paths["random_monthly_baseline"],
        signals_csv_path=paths["signals"],
        trades_csv_path=paths["trades"],
        missed_signals_csv_path=paths["missed_signals"],
        daily_pnl_csv_path=paths["daily"],
        personality_summary_csv_path=paths["personality"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        trade_count=int(len(trade_frame)),
    )


__all__ = [
    "WalkForwardPersonalityFilterExitConfig",
    "WalkForwardPersonalityFilterExitResult",
    "WalkForwardSelectedFilterExitConfig",
    "WalkForwardSelectedFilterExitResult",
    "run_walk_forward_personality_filter_exit_lab",
    "run_walk_forward_selected_filter_exit_lab",
]
