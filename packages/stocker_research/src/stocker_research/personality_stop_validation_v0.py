"""Stop-loss and R-multiple diagnostics for personality template rules.

This research-only layer consumes an existing ``personality_template_v0`` report,
rebuilds the event-level rows behind each selected caveat, and evaluates fixed
and structure-derived stop distances. It does not fetch vendor data, touch broker
or execution paths, or place orders.
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

from stocker_research.personality_discovery_v0 import (
    _candidate_mask,
    _return_column,
    add_discovery_features,
)
from stocker_research.personality_template_v0 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TEMPLATE_OUTPUT_DIR,
)
from stocker_research.personality_template_v0 import (
    PersonalityTemplate,
    _base_mask,
    _score_against_parent,
    _write_csv,
    _write_json,
    load_personality_templates,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_stop_validation_v0")


@dataclass(frozen=True)
class PersonalityStopValidationConfig:
    """Configuration for personality stop/R validation."""

    stop_loss_bps: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
    target_r_multiples: tuple[float, ...] = (1.0, 1.5, 2.0)
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0)
    random_iterations: int = 100
    random_seed: int = 1337
    train_fraction: float = 0.60
    max_candidate_book_rows: int = 12
    min_events: int = 8
    min_symbols: int = 3
    min_months: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    structure_buffer_bps: float = 10.0
    min_structure_stop_bps: float = 5.0


@dataclass(frozen=True)
class PersonalityStopValidationResult:
    """Paths and headline result for a stop/R validation run."""

    run_id: str
    input_template_dir: Path
    input_event_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    stop_model_results_csv_path: Path
    selected_stop_models_csv_path: Path
    rejected_stop_models_csv_path: Path
    random_stop_baseline_csv_path: Path
    oos_stop_results_csv_path: Path
    cost_sensitivity_csv_path: Path
    frequency_summary_csv_path: Path
    candidate_book_csv_path: Path
    concentration_warnings_csv_path: Path
    stop_event_examples_csv_path: Path
    decision: str
    selected_stop_model_count: int


def _latest_template_run(input_template_dir: Path | None, input_base_dir: Path) -> Path:
    if input_template_dir is not None:
        return input_template_dir
    candidates = [
        path
        for path in input_base_dir.iterdir()
        if path.is_dir() and (path / "selected_template_rules.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No personality_template_v0 report found under {input_base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _format_bps(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1f}"


def _format_r(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _month_series(rows: pd.DataFrame) -> pd.Series:
    if "month" in rows:
        return rows["month"].astype(str)
    if "session_date" in rows:
        return rows["session_date"].astype(str).str.slice(0, 7)
    return pd.Series("", index=rows.index)


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
    symbol_counts = rows["symbol"].astype(str).value_counts()
    sessions = rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1)
    session_counts = sessions.value_counts()
    month_counts = _month_series(rows).value_counts()
    return {
        "symbol_count": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "session_count": int(session_counts.size),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "month_count": int(month_counts.size),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
    }


def _split_train_test_by_time(
    rows: pd.DataFrame,
    *,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by timestamp so train always precedes test."""

    if rows.empty:
        return rows.copy(), rows.copy()
    fraction = min(max(float(train_fraction), 0.01), 0.99)
    ordered = rows.copy()
    timestamps = pd.to_datetime(ordered.get("timestamp"), utc=True, errors="coerce")
    ordered["_split_timestamp"] = timestamps
    sort_columns = [
        column
        for column in ("_split_timestamp", "symbol", "session_date")
        if column in ordered.columns
    ]
    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    split = max(1, min(len(ordered) - 1, int(len(ordered) * fraction)))
    train = ordered.iloc[:split].drop(columns=["_split_timestamp"])
    test = ordered.iloc[split:].drop(columns=["_split_timestamp"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _distance_risk_bps(
    rows: pd.DataFrame,
    *,
    long_column: str,
    short_column: str,
    expected_direction: int,
    buffer_bps: float,
    minimum_bps: float,
) -> pd.Series:
    if expected_direction > 0:
        raw = pd.to_numeric(rows.get(long_column, np.nan), errors="coerce") * 10000.0
    else:
        raw = -pd.to_numeric(rows.get(short_column, np.nan), errors="coerce") * 10000.0
    risk = raw + buffer_bps
    return risk.where(risk >= minimum_bps, minimum_bps)


def _risk_bps_for_model(
    rows: pd.DataFrame,
    *,
    model_name: str,
    expected_direction: int,
    structure_buffer_bps: float = 10.0,
    min_structure_stop_bps: float = 5.0,
) -> pd.Series:
    """Calculate event-bar stop distance for a named model.

    Structure models only use current/prior distance features already present on
    the event rows. They do not use forward return, MFE, or MAE columns.
    """

    if model_name.startswith("fixed_") and model_name.endswith("bps"):
        stop = float(model_name.removeprefix("fixed_").removesuffix("bps"))
        return pd.Series(stop, index=rows.index, dtype=float)
    if model_name == "structure_session_extreme_10bps":
        return _distance_risk_bps(
            rows,
            long_column="distance_from_session_low_pct",
            short_column="distance_from_session_high_pct",
            expected_direction=expected_direction,
            buffer_bps=structure_buffer_bps,
            minimum_bps=min_structure_stop_bps,
        )
    if model_name == "structure_recent_extreme_10bps":
        return _distance_risk_bps(
            rows,
            long_column="distance_from_recent_low_pct",
            short_column="distance_from_recent_high_pct",
            expected_direction=expected_direction,
            buffer_bps=structure_buffer_bps,
            minimum_bps=min_structure_stop_bps,
        )
    if model_name == "structure_opening_range_extreme_10bps":
        return _distance_risk_bps(
            rows,
            long_column="distance_from_opening_range_low_pct",
            short_column="distance_from_opening_range_high_pct",
            expected_direction=expected_direction,
            buffer_bps=structure_buffer_bps,
            minimum_bps=min_structure_stop_bps,
        )
    raise ValueError(f"Unsupported stop model: {model_name}")


def _stop_model_names(config: PersonalityStopValidationConfig) -> tuple[str, ...]:
    fixed = tuple(f"fixed_{int(stop)}bps" for stop in config.stop_loss_bps)
    structure = (
        "structure_session_extreme_10bps",
        "structure_recent_extreme_10bps",
        "structure_opening_range_extreme_10bps",
    )
    return fixed + structure


def score_stop_model_events(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
    risk_bps: pd.Series,
    target_r: float,
) -> pd.DataFrame:
    """Return event-level conservative R metrics for one stop model."""

    if expected_direction == 0:
        raise ValueError("R-multiple stop scoring requires a directional expected direction")
    ret_col = _return_column(horizon)
    mfe_col = f"forward_{horizon}_bar_mfe"
    mae_col = f"forward_{horizon}_bar_mae"
    data = rows.copy()
    risk = pd.to_numeric(risk_bps, errors="coerce").reindex(data.index)
    risk = risk.where(risk > 0.0, np.nan)
    forward_return_bps = pd.to_numeric(data[ret_col], errors="coerce") * 10000.0
    mfe_bps = pd.to_numeric(data.get(mfe_col, np.nan), errors="coerce") * 10000.0
    mae_bps = pd.to_numeric(data.get(mae_col, np.nan), errors="coerce") * 10000.0
    if expected_direction > 0:
        favorable_bps = mfe_bps
        adverse_bps = -mae_bps
    else:
        favorable_bps = -mae_bps
        adverse_bps = mfe_bps
    aligned_return_bps = expected_direction * forward_return_bps
    stop_hit = adverse_bps >= risk
    target_hit = favorable_bps >= risk * target_r
    data["risk_bps"] = risk
    data["aligned_return_bps"] = aligned_return_bps
    data["favorable_excursion_bps"] = favorable_bps
    data["adverse_excursion_bps"] = adverse_bps
    data["stop_hit"] = stop_hit.map(bool).astype(object)
    data["target_hit"] = target_hit.map(bool).astype(object)
    data["target_stop_order_ambiguous"] = (stop_hit & target_hit).map(bool).astype(object)
    data["final_r_raw"] = aligned_return_bps / risk
    data["max_favorable_r"] = favorable_bps / risk
    data["max_adverse_r"] = adverse_bps / risk
    data["final_r_conservative"] = data["final_r_raw"].where(~stop_hit, -1.0)
    return data


def _apply_cost_to_scored_events(rows: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Apply a simple all-in bps cost to already-scored directional events."""

    data = rows.copy()
    risk = pd.to_numeric(data["risk_bps"], errors="coerce")
    aligned = pd.to_numeric(data["aligned_return_bps"], errors="coerce")
    cost = float(cost_bps)
    stop_hit = pd.Series(data["stop_hit"]).astype(bool)
    raw_after_cost = (aligned - cost) / risk
    stopped_after_cost = -1.0 - (cost / risk)
    data["cost_bps"] = cost
    data["final_r_after_cost"] = raw_after_cost.where(~stop_hit, stopped_after_cost)
    data["win_after_cost"] = data["final_r_after_cost"] > 0.0
    return data


def _summarize_scored_events(
    rows: pd.DataFrame,
    *,
    target_r: float,
) -> dict[str, float | int]:
    if rows.empty:
        return {
            "event_count": 0,
            "median_final_r_conservative": math.nan,
            "mean_final_r_conservative": math.nan,
            "win_rate_after_stop": math.nan,
            "stop_hit_rate": math.nan,
            "target_hit_rate": math.nan,
            "target_stop_order_ambiguous_rate": math.nan,
            "median_aligned_return_bps": math.nan,
            "median_risk_bps": math.nan,
            "median_max_favorable_r": math.nan,
            "median_max_adverse_r": math.nan,
            "target_r": target_r,
        }
    final_r = pd.to_numeric(rows["final_r_conservative"], errors="coerce")
    return {
        "event_count": int(len(rows)),
        "median_final_r_conservative": float(final_r.median()),
        "mean_final_r_conservative": float(final_r.mean()),
        "win_rate_after_stop": float((final_r > 0.0).mean()),
        "stop_hit_rate": float(pd.Series(rows["stop_hit"]).astype(bool).mean()),
        "target_hit_rate": float(pd.Series(rows["target_hit"]).astype(bool).mean()),
        "target_stop_order_ambiguous_rate": float(
            pd.Series(rows["target_stop_order_ambiguous"]).astype(bool).mean()
        ),
        "median_aligned_return_bps": float(
            pd.to_numeric(rows["aligned_return_bps"], errors="coerce").median()
        ),
        "median_risk_bps": float(pd.to_numeric(rows["risk_bps"], errors="coerce").median()),
        "median_max_favorable_r": float(
            pd.to_numeric(rows["max_favorable_r"], errors="coerce").median()
        ),
        "median_max_adverse_r": float(
            pd.to_numeric(rows["max_adverse_r"], errors="coerce").median()
        ),
        "target_r": target_r,
    }


def _summarize_costed_events(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {
            "event_count": 0,
            "median_final_r_after_cost": math.nan,
            "mean_final_r_after_cost": math.nan,
            "win_rate_after_cost": math.nan,
        }
    final_r = pd.to_numeric(rows["final_r_after_cost"], errors="coerce")
    return {
        "event_count": int(len(rows)),
        "median_final_r_after_cost": float(final_r.median()),
        "mean_final_r_after_cost": float(final_r.mean()),
        "win_rate_after_cost": float((final_r > 0.0).mean()),
    }


def _random_stop_baseline_from_scored_pool(
    scored_pool: pd.DataFrame,
    *,
    count: int,
    config: PersonalityStopValidationConfig,
    seed: int,
) -> dict[str, float]:
    if count <= 0 or len(scored_pool) < count:
        return {
            "random_median_final_r_conservative": math.nan,
            "random_mean_final_r_conservative": math.nan,
            "random_win_rate_after_stop": math.nan,
            "random_stop_hit_rate": math.nan,
        }
    rng = np.random.default_rng(seed)
    final_r = pd.to_numeric(scored_pool["final_r_conservative"], errors="coerce").to_numpy()
    stop_hit = pd.Series(scored_pool["stop_hit"]).astype(float).to_numpy()
    medians: list[float] = []
    means: list[float] = []
    win_rates: list[float] = []
    stop_hits: list[float] = []
    for _ in range(config.random_iterations):
        sample = rng.choice(len(scored_pool), size=count, replace=False)
        sample_r = final_r[sample]
        medians.append(float(np.nanmedian(sample_r)))
        means.append(float(np.nanmean(sample_r)))
        win_rates.append(float(np.nanmean(sample_r > 0.0)))
        stop_hits.append(float(np.nanmean(stop_hit[sample])))
    return {
        "random_median_final_r_conservative": float(np.nanmedian(medians)),
        "random_mean_final_r_conservative": float(np.nanmean(means)),
        "random_win_rate_after_stop": float(np.nanmean(win_rates)),
        "random_stop_hit_rate": float(np.nanmean(stop_hits)),
    }


def _random_stop_baseline(
    pool: pd.DataFrame,
    *,
    count: int,
    horizon: int,
    expected_direction: int,
    model_name: str,
    target_r: float,
    config: PersonalityStopValidationConfig,
    seed: int,
) -> dict[str, float]:
    if count <= 0 or len(pool) < count or expected_direction == 0:
        return {
            "random_median_final_r_conservative": math.nan,
            "random_mean_final_r_conservative": math.nan,
            "random_win_rate_after_stop": math.nan,
            "random_stop_hit_rate": math.nan,
        }
    rng = np.random.default_rng(seed)
    medians: list[float] = []
    means: list[float] = []
    win_rates: list[float] = []
    stop_hits: list[float] = []
    for _ in range(config.random_iterations):
        sample_index = rng.choice(pool.index.to_numpy(), size=count, replace=False)
        sample = pool.loc[sample_index].copy()
        risk = _risk_bps_for_model(
            sample,
            model_name=model_name,
            expected_direction=expected_direction,
            structure_buffer_bps=config.structure_buffer_bps,
            min_structure_stop_bps=config.min_structure_stop_bps,
        )
        scored = score_stop_model_events(
            sample,
            horizon=horizon,
            expected_direction=expected_direction,
            risk_bps=risk,
            target_r=target_r,
        )
        summary = _summarize_scored_events(scored, target_r=target_r)
        medians.append(float(summary["median_final_r_conservative"]))
        means.append(float(summary["mean_final_r_conservative"]))
        win_rates.append(float(summary["win_rate_after_stop"]))
        stop_hits.append(float(summary["stop_hit_rate"]))
    return {
        "random_median_final_r_conservative": float(np.nanmedian(medians)),
        "random_mean_final_r_conservative": float(np.nanmean(means)),
        "random_win_rate_after_stop": float(np.nanmean(win_rates)),
        "random_stop_hit_rate": float(np.nanmean(stop_hits)),
    }


def _rule_retained_rows(
    base_scored: pd.DataFrame,
    rule: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_field = str(rule["regime_field"])
    regime_value = str(rule["regime_value"])
    regime_pool = base_scored[base_scored[regime_field].astype(str).eq(regime_value)].copy()
    retained = regime_pool[_candidate_mask(regime_pool, rule)].copy()
    return base_scored, regime_pool, retained


def _template_base_contexts(
    events: pd.DataFrame,
    selected_rules: pd.DataFrame,
    template_lookup: dict[str, PersonalityTemplate],
) -> dict[str, pd.DataFrame]:
    contexts: dict[str, pd.DataFrame] = {}
    for template_id in selected_rules["template_id"].dropna().astype(str).unique():
        template = template_lookup.get(template_id)
        if template is None:
            continue
        horizon = int(template.horizon)
        ret_col = _return_column(horizon)
        parent = events[events["event_state"].astype(str).eq(template.parent_event_state)].copy()
        parent = parent.dropna(subset=[ret_col])
        base = events[_base_mask(events, template)].copy().dropna(subset=[ret_col])
        contexts[template_id] = _score_against_parent(
            base,
            parent,
            horizon=horizon,
            expected_direction=template.expected_direction,
        )
    return contexts


def _result_verdict(
    row: dict[str, Any],
    config: PersonalityStopValidationConfig,
) -> tuple[str, str]:
    reasons: list[str] = []
    if int(row["event_count"]) < config.min_events:
        reasons.append("low_event_count")
    if int(row["symbol_count"]) < config.min_symbols:
        reasons.append("low_symbol_count")
    if int(row["month_count"]) < config.min_months:
        reasons.append("low_month_count")
    for column, threshold, reason in [
        ("single_symbol_share", config.max_single_symbol_share, "single_symbol_dominated"),
        ("single_session_share", config.max_single_session_share, "single_session_dominated"),
        ("single_month_share", config.max_single_month_share, "single_month_dominated"),
    ]:
        value = _safe_float(row.get(column))
        if not math.isnan(value) and value > threshold:
            reasons.append(reason)
    median_r = _safe_float(row.get("median_final_r_conservative"))
    random_median_r = _safe_float(row.get("random_median_final_r_conservative"))
    win_rate = _safe_float(row.get("win_rate_after_stop"))
    random_win_rate = _safe_float(row.get("random_win_rate_after_stop"))
    if math.isnan(median_r) or median_r <= 0:
        reasons.append("non_positive_median_r")
    if not math.isnan(random_median_r) and median_r <= random_median_r:
        reasons.append("random_median_r_not_beaten")
    if not math.isnan(random_win_rate) and win_rate <= random_win_rate:
        reasons.append("random_win_rate_not_beaten")
    if reasons:
        return "reject", ";".join(reasons)
    return "pass_stop_model", ""


def _frequency_row(
    rows: pd.DataFrame,
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    concentration = _concentration(rows)
    if rows.empty:
        first_timestamp = ""
        last_timestamp = ""
        calendar_days = 0
    else:
        timestamps = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        first_timestamp = timestamps.min().isoformat()
        last_timestamp = timestamps.max().isoformat()
        calendar_days = int(max(1, (timestamps.max() - timestamps.min()).days + 1))
    session_count = int(concentration["session_count"])
    symbol_count = int(concentration["symbol_count"])
    month_count = int(concentration["month_count"])
    event_count = int(len(rows))
    return {
        **base,
        "event_count": event_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "calendar_days": calendar_days,
        "events_per_session": float(event_count / session_count) if session_count else math.nan,
        "events_per_symbol": float(event_count / symbol_count) if symbol_count else math.nan,
        "events_per_month": float(event_count / month_count) if month_count else math.nan,
        **concentration,
    }


def _oos_row(
    *,
    base: dict[str, Any],
    scored: pd.DataFrame,
    scored_pool: pd.DataFrame,
    target_r: float,
    config: PersonalityStopValidationConfig,
    seed: int,
) -> dict[str, Any]:
    train, test = _split_train_test_by_time(scored, train_fraction=config.train_fraction)
    pool_train, pool_test = _split_train_test_by_time(
        scored_pool,
        train_fraction=config.train_fraction,
    )
    train_summary = _summarize_scored_events(train, target_r=target_r)
    test_summary = _summarize_scored_events(test, target_r=target_r)
    train_random = _random_stop_baseline_from_scored_pool(
        pool_train,
        count=len(train),
        config=config,
        seed=seed,
    )
    test_random = _random_stop_baseline_from_scored_pool(
        pool_test,
        count=len(test),
        config=config,
        seed=seed + 17,
    )
    test_median = _safe_float(test_summary["median_final_r_conservative"])
    random_test_median = _safe_float(test_random["random_median_final_r_conservative"])
    test_win = _safe_float(test_summary["win_rate_after_stop"])
    random_test_win = _safe_float(test_random["random_win_rate_after_stop"])
    reasons: list[str] = []
    if int(train_summary["event_count"]) < max(3, config.min_events // 2):
        reasons.append("low_train_count")
    if int(test_summary["event_count"]) < max(3, config.min_events // 2):
        reasons.append("low_test_count")
    if math.isnan(test_median) or test_median <= 0.0:
        reasons.append("non_positive_test_median_r")
    if not math.isnan(random_test_median) and test_median <= random_test_median:
        reasons.append("random_test_median_not_beaten")
    if not math.isnan(random_test_win) and test_win <= random_test_win:
        reasons.append("random_test_win_not_beaten")
    return {
        **base,
        "train_event_count": int(train_summary["event_count"]),
        "test_event_count": int(test_summary["event_count"]),
        "train_median_final_r_conservative": train_summary["median_final_r_conservative"],
        "test_median_final_r_conservative": test_summary["median_final_r_conservative"],
        "train_win_rate_after_stop": train_summary["win_rate_after_stop"],
        "test_win_rate_after_stop": test_summary["win_rate_after_stop"],
        "train_stop_hit_rate": train_summary["stop_hit_rate"],
        "test_stop_hit_rate": test_summary["stop_hit_rate"],
        "random_train_median_final_r_conservative": train_random[
            "random_median_final_r_conservative"
        ],
        "random_test_median_final_r_conservative": test_random[
            "random_median_final_r_conservative"
        ],
        "random_train_win_rate_after_stop": train_random["random_win_rate_after_stop"],
        "random_test_win_rate_after_stop": test_random["random_win_rate_after_stop"],
        "test_median_r_excess_vs_random": test_median - random_test_median
        if not math.isnan(random_test_median)
        else math.nan,
        "test_win_rate_excess_vs_random": test_win - random_test_win
        if not math.isnan(random_test_win)
        else math.nan,
        "oos_verdict": "pass_oos_stop_model" if not reasons else "reject",
        "oos_reject_reasons": ";".join(reasons),
    }


def _cost_rows(
    *,
    base: dict[str, Any],
    scored: pd.DataFrame,
    config: PersonalityStopValidationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cost in config.cost_bps:
        costed = _apply_cost_to_scored_events(scored, cost_bps=float(cost))
        rows.append(
            {
                **base,
                "cost_bps": float(cost),
                **_summarize_costed_events(costed),
            }
        )
    return rows


def _build_candidate_book(
    selected: pd.DataFrame,
    oos_results: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    frequency: pd.DataFrame,
    *,
    config: PersonalityStopValidationConfig,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "template_id",
                "role",
                "stop_model",
                "target_r",
                "candidate_status",
                "candidate_reject_reasons",
            ]
        )
    key = ["template_id", "filter_rule", "stop_model", "target_r"]
    candidates = selected.copy()
    if not oos_results.empty:
        candidates = candidates.merge(
            oos_results[
                key
                + [
                    "test_event_count",
                    "test_median_final_r_conservative",
                    "test_win_rate_after_stop",
                    "random_test_median_final_r_conservative",
                    "oos_verdict",
                    "oos_reject_reasons",
                ]
            ],
            on=key,
            how="left",
        )
    max_cost = max(config.cost_bps) if config.cost_bps else 0.0
    if not cost_sensitivity.empty:
        worst_cost = cost_sensitivity[cost_sensitivity["cost_bps"].eq(float(max_cost))]
        candidates = candidates.merge(
            worst_cost[key + ["median_final_r_after_cost", "win_rate_after_cost"]],
            on=key,
            how="left",
        )
    if not frequency.empty:
        candidates = candidates.merge(
            frequency[
                key
                + [
                    "events_per_session",
                    "events_per_symbol",
                    "events_per_month",
                    "calendar_days",
                ]
            ],
            on=key,
            how="left",
        )
    statuses: list[str] = []
    reasons_out: list[str] = []
    for _, row in candidates.iterrows():
        reasons: list[str] = []
        if str(row.get("oos_verdict", "")) != "pass_oos_stop_model":
            reasons.append("oos_not_passed")
        if _safe_float(row.get("median_final_r_after_cost")) <= 0.0:
            reasons.append("cost_adjusted_median_r_not_positive")
        if _safe_float(row.get("test_event_count"), 0.0) < max(3, config.min_events // 2):
            reasons.append("low_oos_count")
        statuses.append("candidate_continue_research" if not reasons else "candidate_reject")
        reasons_out.append(";".join(reasons))
    candidates["candidate_status"] = statuses
    candidates["candidate_reject_reasons"] = reasons_out
    return (
        candidates.sort_values(
            [
                "candidate_status",
                "test_median_final_r_conservative",
                "median_final_r_after_cost",
                "median_final_r_conservative",
            ],
            ascending=[True, False, False, False],
        )
        .head(config.max_candidate_book_rows)
        .reset_index(drop=True)
    )


def evaluate_personality_stop_models(
    event_rows: pd.DataFrame,
    selected_rules: pd.DataFrame,
    templates: tuple[PersonalityTemplate, ...],
    *,
    config: PersonalityStopValidationConfig = PersonalityStopValidationConfig(),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate stop models for selected personality caveats."""

    regime_columns = {
        str(column)
        for column in selected_rules.get("regime_field", pd.Series(dtype=str)).dropna().unique()
    }
    explicit_regimes = {
        column: event_rows[column].copy() for column in regime_columns if column in event_rows
    }
    events = add_discovery_features(event_rows)
    for column, values in explicit_regimes.items():
        events[column] = values
    if "month" not in events:
        events["month"] = _month_series(events)
    template_lookup = {template.template_id: template for template in templates}
    base_contexts = _template_base_contexts(events, selected_rules, template_lookup)
    result_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    cost_sensitivity_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    example_rows: list[pd.DataFrame] = []

    for rule_index, rule in selected_rules.iterrows():
        template_id = str(rule["template_id"])
        template = template_lookup.get(template_id)
        if template is None:
            continue
        base_scored = base_contexts.get(template_id)
        if base_scored is None or base_scored.empty:
            continue
        expected_direction = int(rule["expected_direction"])
        horizon = int(rule["horizon"])
        if expected_direction == 0:
            continue
        _, regime_pool, retained = _rule_retained_rows(base_scored, rule)
        if retained.empty:
            continue
        concentration = _concentration(retained)
        for model_index, model_name in enumerate(_stop_model_names(config)):
            for target_r in config.target_r_multiples:
                risk = _risk_bps_for_model(
                    retained,
                    model_name=model_name,
                    expected_direction=expected_direction,
                    structure_buffer_bps=config.structure_buffer_bps,
                    min_structure_stop_bps=config.min_structure_stop_bps,
                )
                scored = score_stop_model_events(
                    retained,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    risk_bps=risk,
                    target_r=target_r,
                )
                pool_risk = _risk_bps_for_model(
                    regime_pool,
                    model_name=model_name,
                    expected_direction=expected_direction,
                    structure_buffer_bps=config.structure_buffer_bps,
                    min_structure_stop_bps=config.min_structure_stop_bps,
                )
                scored_pool = score_stop_model_events(
                    regime_pool,
                    horizon=horizon,
                    expected_direction=expected_direction,
                    risk_bps=pool_risk,
                    target_r=target_r,
                )
                random_seed = config.random_seed + rule_index * 1009 + model_index * 37
                random_result = _random_stop_baseline_from_scored_pool(
                    scored_pool,
                    count=len(retained),
                    config=config,
                    seed=random_seed,
                )
                base_identity = {
                    "template_id": template_id,
                    "personality": str(rule.get("personality", template.personality)),
                    "parent_event_state": template.parent_event_state,
                    "role": str(rule["role"]),
                    "horizon": horizon,
                    "expected_direction": expected_direction,
                    "regime_field": str(rule["regime_field"]),
                    "regime_value": str(rule["regime_value"]),
                    "filter_rule": str(rule["filter_rule"]),
                    "stop_model": model_name,
                    "target_r": target_r,
                }
                row = {
                    **base_identity,
                    **_summarize_scored_events(scored, target_r=target_r),
                    **concentration,
                    **random_result,
                }
                verdict, reasons = _result_verdict(row, config)
                row["verdict"] = verdict
                row["reject_reasons"] = reasons
                result_rows.append(row)
                oos_rows.append(
                    _oos_row(
                        base=base_identity,
                        scored=scored,
                        scored_pool=scored_pool,
                        target_r=target_r,
                        config=config,
                        seed=random_seed + 101,
                    )
                )
                cost_sensitivity_rows.extend(
                    _cost_rows(base=base_identity, scored=scored, config=config)
                )
                frequency_rows.append(_frequency_row(scored, base=base_identity))
                random_rows.append(
                    {
                        "template_id": template_id,
                        "filter_rule": str(rule["filter_rule"]),
                        "stop_model": model_name,
                        "target_r": target_r,
                        "event_count": int(len(retained)),
                        **random_result,
                    }
                )
                if verdict == "pass_stop_model":
                    examples = scored.sort_values("final_r_conservative", ascending=False).head(3)
                    example = examples[
                        [
                            "symbol",
                            "timestamp",
                            "session_date",
                            "final_r_conservative",
                            "aligned_return_bps",
                            "risk_bps",
                            "stop_hit",
                            "target_hit",
                        ]
                    ].copy()
                    example.insert(0, "template_id", template_id)
                    example.insert(1, "stop_model", model_name)
                    example.insert(2, "target_r", target_r)
                    example_rows.append(example)

    results = pd.DataFrame(result_rows)
    selected = (
        results[results["verdict"].eq("pass_stop_model")].copy()
        if not results.empty
        else pd.DataFrame()
    )
    if not selected.empty:
        selected = (
            selected.sort_values(
                ["template_id", "median_final_r_conservative", "win_rate_after_stop"],
                ascending=[True, False, False],
            )
            .groupby("template_id", group_keys=False)
            .head(5)
            .reset_index(drop=True)
        )
    if selected.empty and not results.empty:
        selected = pd.DataFrame(columns=results.columns)
    rejected = (
        results[results["verdict"].ne("pass_stop_model")].copy()
        if not results.empty
        else pd.DataFrame(columns=results.columns)
    )
    examples = pd.concat(example_rows, ignore_index=True) if example_rows else pd.DataFrame()
    oos_results = pd.DataFrame(oos_rows)
    cost_sensitivity = pd.DataFrame(cost_sensitivity_rows)
    frequency = pd.DataFrame(frequency_rows)
    candidate_book = _build_candidate_book(
        selected,
        oos_results,
        cost_sensitivity,
        frequency,
        config=config,
    )
    return (
        results,
        selected,
        rejected,
        pd.DataFrame(random_rows),
        oos_results,
        cost_sensitivity,
        frequency,
        candidate_book,
        examples,
    )


def _decision(results: pd.DataFrame, selected: pd.DataFrame) -> str:
    if selected.empty:
        if not results.empty and results["reject_reasons"].astype(str).str.contains(
            "dominated", na=False
        ).all():
            return "reject_concentrated"
        return "reject_no_stop_model_improvement"
    return "continue_research_stop_model"


def _summary_lines(
    *,
    input_template_dir: Path,
    input_event_dir: Path,
    decision: str,
    results: pd.DataFrame,
    selected: pd.DataFrame,
    rejected: pd.DataFrame,
    oos_results: pd.DataFrame,
    candidate_book: pd.DataFrame,
) -> list[str]:
    lines = [
        "# Personality Stop Validation V0",
        "",
        (
            "Research-only stop-loss and R-multiple diagnostics for selected "
            "personality + regime + filter rules. No broker, no IG, no live trading, "
            "no paper trading, no vendor fetching, and no order placement. No edge is claimed."
        ),
        "",
        f"Input template report: `{input_template_dir}`",
        f"Input event report: `{input_event_dir}`",
        f"Decision: `{decision}`",
        "",
        "## Counts",
        "",
        f"- Stop model rows: `{len(results)}`",
        f"- Selected stop models: `{len(selected)}`",
        f"- Rejected stop models: `{len(rejected)}`",
        f"- OOS stop rows: `{len(oos_results)}`",
        f"- Candidate-book rows: `{len(candidate_book)}`",
        "",
        "## Selected Stop Models",
        "",
        (
            "| template | role | stop | target R | n | symbols | median R | "
            "win | stop hit | random median R | verdict |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not selected.empty:
        for _, row in selected.sort_values(
            ["median_final_r_conservative", "win_rate_after_stop"],
            ascending=[False, False],
        ).head(40).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["template_id"]),
                        str(row["role"]),
                        str(row["stop_model"]),
                        _format_r(row["target_r"]),
                        str(int(row["event_count"])),
                        str(int(row["symbol_count"])),
                        _format_r(row["median_final_r_conservative"]),
                        _format_pct(row["win_rate_after_stop"]),
                        _format_pct(row["stop_hit_rate"]),
                        _format_r(row["random_median_final_r_conservative"]),
                        str(row["verdict"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Candidate Book",
            "",
            (
                "| template | role | stop | target R | test median R | cost median R | "
                "events/session | status |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not candidate_book.empty:
        for _, row in candidate_book.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["template_id"]),
                        str(row["role"]),
                        str(row["stop_model"]),
                        _format_r(row["target_r"]),
                        _format_r(row.get("test_median_final_r_conservative")),
                        _format_r(row.get("median_final_r_after_cost")),
                        _format_r(row.get("events_per_session")),
                        str(row["candidate_status"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A stop model passes only when conservative median R is positive, "
                "random same-count median R is beaten, random same-count win rate is "
                "beaten, and concentration gates pass. MFE/MAE cannot prove whether a "
                "target or stop happened first, so rows where both are touched are "
                "marked as target/stop order ambiguous and scored conservatively as "
                "-1R when the stop is touched. Candidate-book rows additionally require "
                "the later test split and the configured highest cost assumption to remain "
                "positive."
            ),
        ]
    )
    return lines


def run_personality_stop_validation_lab(
    *,
    input_template_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_TEMPLATE_OUTPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityStopValidationConfig = PersonalityStopValidationConfig(),
) -> PersonalityStopValidationResult:
    """Run stop-loss and R-multiple validation for selected personality rules."""

    resolved_template_dir = _latest_template_run(input_template_dir, input_base_dir)
    selected_path = resolved_template_dir / "selected_template_rules.csv"
    summary_path = resolved_template_dir / "summary.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected template rules: {selected_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing personality template summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_event_dir = Path(summary["input_event_dir"])
    template_path = Path(summary["template_path"])
    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")

    selected_rules = pd.read_csv(selected_path)
    event_rows = pd.read_csv(event_rows_path)
    template_book = load_personality_templates(template_path)
    (
        results,
        selected,
        rejected,
        random_baseline,
        oos_results,
        cost_sensitivity,
        frequency,
        candidate_book,
        examples,
    ) = evaluate_personality_stop_models(
        event_rows,
        selected_rules,
        template_book.templates,
        config=config,
    )
    concentration_warnings = (
        results[results["reject_reasons"].astype(str).str.contains("dominated", na=False)]
        if not results.empty
        else pd.DataFrame()
    )
    decision = _decision(results, selected)
    run_id = f"personality_stop_validation_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "results": run_dir / "stop_model_results.csv",
        "selected": run_dir / "selected_stop_models.csv",
        "rejected": run_dir / "rejected_stop_models.csv",
        "random": run_dir / "random_stop_baseline.csv",
        "oos": run_dir / "oos_stop_results.csv",
        "cost": run_dir / "cost_sensitivity.csv",
        "frequency": run_dir / "frequency_summary.csv",
        "candidate_book": run_dir / "candidate_book.csv",
        "concentration": run_dir / "concentration_warnings.csv",
        "examples": run_dir / "stop_event_examples.csv",
    }
    for path, frame in [
        (paths["results"], results),
        (paths["selected"], selected),
        (paths["rejected"], rejected),
        (paths["random"], random_baseline),
        (paths["oos"], oos_results),
        (paths["cost"], cost_sensitivity),
        (paths["frequency"], frequency),
        (paths["candidate_book"], candidate_book),
        (paths["concentration"], concentration_warnings),
        (paths["examples"], examples),
    ]:
        _write_csv(path, frame)

    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_template_dir": str(resolved_template_dir),
        "input_event_dir": str(input_event_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "stop_model_result_count": int(len(results)),
        "selected_stop_model_count": int(len(selected)),
        "rejected_stop_model_count": int(len(rejected)),
        "oos_stop_result_count": int(len(oos_results)),
        "candidate_book_count": int(len(candidate_book)),
        "stop_loss_bps": [float(value) for value in config.stop_loss_bps],
        "target_r_multiples": [float(value) for value in config.target_r_multiples],
        "cost_bps": [float(value) for value in config.cost_bps],
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)
    lines = _summary_lines(
        input_template_dir=resolved_template_dir,
        input_event_dir=input_event_dir,
        decision=decision,
        results=results,
        selected=selected,
        rejected=rejected,
        oos_results=oos_results,
        candidate_book=candidate_book,
    )
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PersonalityStopValidationResult(
        run_id=run_id,
        input_template_dir=resolved_template_dir,
        input_event_dir=input_event_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        stop_model_results_csv_path=paths["results"],
        selected_stop_models_csv_path=paths["selected"],
        rejected_stop_models_csv_path=paths["rejected"],
        random_stop_baseline_csv_path=paths["random"],
        oos_stop_results_csv_path=paths["oos"],
        cost_sensitivity_csv_path=paths["cost"],
        frequency_summary_csv_path=paths["frequency"],
        candidate_book_csv_path=paths["candidate_book"],
        concentration_warnings_csv_path=paths["concentration"],
        stop_event_examples_csv_path=paths["examples"],
        decision=decision,
        selected_stop_model_count=int(len(selected)),
    )


__all__ = [
    "PersonalityStopValidationConfig",
    "PersonalityStopValidationResult",
    "evaluate_personality_stop_models",
    "run_personality_stop_validation_lab",
    "score_stop_model_events",
]
