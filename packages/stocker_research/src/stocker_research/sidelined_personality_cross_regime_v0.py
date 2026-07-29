"""Cross-regime diagnostic for sidelined personality ideas.

This research-only report asks whether personalities that are not already in
the selected-filter book deserve another look. It consumes existing local
state-event outputs and optional prior specialty reports; it does not fetch
data, touch broker/execution paths, or place orders.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.personality_discovery_v0 import add_discovery_features

DEFAULT_OUTPUT_DIR = Path("data/reports/research/sidelined_personality_cross_regime_v0")

DEFAULT_REGIME_FIELDS: tuple[str, ...] = (
    "compression_x_efficiency_regime",
    "opening_mid_x_range_regime",
    "time_x_vwap_regime",
    "volume_x_vwap_regime",
    "vwap_x_efficiency_regime",
    "vwap_x_range_regime",
)

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
    "bar_return",
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "range_zscore",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "bar_index_in_session",
)

NO_TRADE_ROLES = {"no_trade_filter", "mean_reversion_or_no_chase"}


@dataclass(frozen=True)
class SidelinedPersonalityCrossRegimeConfig:
    """Configuration for the sidelined-personality crossed-regime diagnostic."""

    horizons: tuple[int, ...] = (6, 9, 12, 24)
    train_fraction: float = 0.60
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS
    filter_features: tuple[str, ...] = DEFAULT_FILTER_FEATURES
    quantiles: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)
    min_train_events: int = 30
    min_test_events: int = 12
    min_retained_events: int = 8
    min_symbols: int = 3
    min_oos_lift: float = 0.02
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    low_movement_threshold: float = 0.0015
    max_regimes_per_personality_horizon: int = 6
    random_iterations: int = 50
    random_seed: int = 1337


@dataclass(frozen=True)
class SidelinedPersonalityCrossRegimeResult:
    """Paths and headline result for a sidelined-personality report."""

    run_id: str
    input_event_dir: Path
    input_selected_filter_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    sidelined_personality_summary_csv_path: Path
    crossed_regime_summary_csv_path: Path
    candidate_filter_results_csv_path: Path
    selected_sidelined_candidates_csv_path: Path
    rejected_sidelined_candidates_csv_path: Path
    external_report_evidence_csv_path: Path
    concentration_warnings_csv_path: Path
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


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 12) -> str:
    if frame.empty:
        return "(empty)"
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    headers = list(visible.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _return_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_return"


def _mfe_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mfe"


def _mae_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mae"


def _safe_share(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _max_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    counts = frame[column].astype(str).value_counts(dropna=False)
    return _safe_share(int(counts.iloc[0]), len(frame)) if not counts.empty else 0.0


def _month_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "month" not in frame:
        return 0
    return int(frame["month"].astype(str).nunique())


def _score_rows(
    rows: pd.DataFrame,
    *,
    role: str,
    direction: int,
    horizon: int,
    low_movement_threshold: float,
) -> dict[str, float | int]:
    ret_col = _return_col(horizon)
    mfe_col = _mfe_col(horizon)
    mae_col = _mae_col(horizon)
    values = pd.to_numeric(rows.get(ret_col, pd.Series(dtype=float)), errors="coerce").dropna()
    count = int(len(values))
    if count == 0:
        return {
            "event_count": 0,
            "same_result_rate": math.nan,
            "median_aligned_return": math.nan,
            "median_abs_return": math.nan,
            "median_mfe": math.nan,
            "median_mae": math.nan,
            "low_movement_rate": math.nan,
        }
    if role in NO_TRADE_ROLES or direction == 0:
        same = values.abs() <= low_movement_threshold
        aligned = -values.abs()
    else:
        aligned = direction * values
        same = aligned > 0
    mfe = pd.to_numeric(rows.get(mfe_col, pd.Series(dtype=float)), errors="coerce")
    mae = pd.to_numeric(rows.get(mae_col, pd.Series(dtype=float)), errors="coerce")
    return {
        "event_count": count,
        "same_result_rate": float(same.mean()),
        "median_aligned_return": float(np.nanmedian(aligned)),
        "median_abs_return": float(np.nanmedian(values.abs())),
        "median_mfe": float(np.nanmedian(mfe)) if mfe.notna().any() else math.nan,
        "median_mae": float(np.nanmedian(mae)) if mae.notna().any() else math.nan,
        "low_movement_rate": float((values.abs() <= low_movement_threshold).mean()),
    }


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


def _load_event_rows(
    input_event_dir: Path,
    *,
    config: SidelinedPersonalityCrossRegimeConfig,
) -> pd.DataFrame:
    path = input_event_dir / "event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event rows: {path}")
    required = {
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "time_of_day_bucket",
        "event_state",
        "distance_from_vwap_pct",
        "distance_from_opening_range_mid_pct",
        "distance_from_session_open_pct",
        "rolling_intraday_range_pct",
        "compression_zscore",
        "directional_efficiency_12",
        "relative_volume_at_bar_index",
        *config.filter_features,
    }
    for horizon in config.horizons:
        required.update({_return_col(horizon), _mfe_col(horizon), _mae_col(horizon)})
    rows = pd.read_csv(path, usecols=lambda column: column in required)
    if rows.empty:
        return rows
    return add_discovery_features(rows)


def _load_promoted_personalities(input_selected_filter_dir: Path) -> set[str]:
    path = input_selected_filter_dir / "selected_filters.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected filters: {path}")
    selected = pd.read_csv(path)
    if selected.empty or "personality" not in selected:
        return set()
    return set(selected["personality"].dropna().astype(str).tolist())


def _sidelined_sparse_rows(rows: pd.DataFrame, promoted: set[str]) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    data = rows.copy()
    if "personality" not in data:
        data = add_discovery_features(data)
    if "personality_role" not in data and "role" in data:
        data["personality_role"] = data["role"]
    data = data[~data["personality"].astype(str).isin(promoted)].copy()
    return data.reset_index(drop=True)


def _personality_summary(
    rows: pd.DataFrame,
    *,
    config: SidelinedPersonalityCrossRegimeConfig,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "personality",
                "event_state",
                "role",
                "default_expected_direction",
                "horizon",
                "train_event_count",
                "test_event_count",
                "test_low_movement_rate",
            ]
        )
    for (personality, event_state), group in rows.groupby(
        ["personality", "event_state"],
        sort=True,
    ):
        role = str(group["personality_role"].iloc[0])
        direction = int(group["default_expected_direction"].iloc[0])
        train, test = _split_train_test(group, config.train_fraction)
        for horizon in config.horizons:
            ret_col = _return_col(horizon)
            if ret_col not in group:
                continue
            train_valid = train[train[ret_col].notna()]
            test_valid = test[test[ret_col].notna()]
            train_score = _score_rows(
                train_valid,
                role=role,
                direction=direction,
                horizon=horizon,
                low_movement_threshold=config.low_movement_threshold,
            )
            test_score = _score_rows(
                test_valid,
                role=role,
                direction=direction,
                horizon=horizon,
                low_movement_threshold=config.low_movement_threshold,
            )
            output.append(
                {
                    "personality": personality,
                    "event_state": event_state,
                    "role": role,
                    "default_expected_direction": direction,
                    "horizon": horizon,
                    "train_event_count": train_score["event_count"],
                    "test_event_count": test_score["event_count"],
                    "train_same_result_rate": train_score["same_result_rate"],
                    "test_same_result_rate": test_score["same_result_rate"],
                    "train_median_aligned_return": train_score["median_aligned_return"],
                    "test_median_aligned_return": test_score["median_aligned_return"],
                    "train_median_abs_return": train_score["median_abs_return"],
                    "test_median_abs_return": test_score["median_abs_return"],
                    "train_low_movement_rate": train_score["low_movement_rate"],
                    "test_low_movement_rate": test_score["low_movement_rate"],
                    "symbol_count": int(group["symbol"].astype(str).nunique()),
                    "session_count": int(group["session_date"].astype(str).nunique()),
                    "month_count": _month_count(group),
                }
            )
    return pd.DataFrame(output)


def _crossed_regime_summary(
    rows: pd.DataFrame,
    *,
    config: SidelinedPersonalityCrossRegimeConfig,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame()
    for (personality, event_state), group in rows.groupby(
        ["personality", "event_state"],
        sort=True,
    ):
        role = str(group["personality_role"].iloc[0])
        direction = int(group["default_expected_direction"].iloc[0])
        train, test = _split_train_test(group, config.train_fraction)
        for horizon in config.horizons:
            ret_col = _return_col(horizon)
            if ret_col not in group:
                continue
            train_valid = train[train[ret_col].notna()]
            test_valid = test[test[ret_col].notna()]
            base_train = _score_rows(
                train_valid,
                role=role,
                direction=direction,
                horizon=horizon,
                low_movement_threshold=config.low_movement_threshold,
            )
            base_test = _score_rows(
                test_valid,
                role=role,
                direction=direction,
                horizon=horizon,
                low_movement_threshold=config.low_movement_threshold,
            )
            for regime_field in config.regime_fields:
                if regime_field not in group:
                    continue
                values = sorted(map(str, group[regime_field].dropna().unique().tolist()))
                for value in values:
                    regime_train = train_valid[train_valid[regime_field].astype(str).eq(value)]
                    regime_test = test_valid[test_valid[regime_field].astype(str).eq(value)]
                    if (
                        len(regime_train) < config.min_train_events
                        or len(regime_test) < config.min_test_events
                    ):
                        continue
                    train_score = _score_rows(
                        regime_train,
                        role=role,
                        direction=direction,
                        horizon=horizon,
                        low_movement_threshold=config.low_movement_threshold,
                    )
                    test_score = _score_rows(
                        regime_test,
                        role=role,
                        direction=direction,
                        horizon=horizon,
                        low_movement_threshold=config.low_movement_threshold,
                    )
                    output.append(
                        {
                            "personality": personality,
                            "event_state": event_state,
                            "role": role,
                            "default_expected_direction": direction,
                            "horizon": horizon,
                            "regime_field": regime_field,
                            "regime_value": value,
                            "base_train_count": base_train["event_count"],
                            "base_test_count": base_test["event_count"],
                            "regime_train_count": train_score["event_count"],
                            "regime_test_count": test_score["event_count"],
                            "base_train_same_result_rate": base_train["same_result_rate"],
                            "base_test_same_result_rate": base_test["same_result_rate"],
                            "regime_train_same_result_rate": train_score["same_result_rate"],
                            "regime_test_same_result_rate": test_score["same_result_rate"],
                            "train_lift_vs_personality": float(
                                train_score["same_result_rate"] - base_train["same_result_rate"]
                            ),
                            "test_lift_vs_personality": float(
                                test_score["same_result_rate"] - base_test["same_result_rate"]
                            ),
                            "regime_train_median_aligned_return": train_score[
                                "median_aligned_return"
                            ],
                            "regime_test_median_aligned_return": test_score[
                                "median_aligned_return"
                            ],
                            "regime_train_median_abs_return": train_score["median_abs_return"],
                            "regime_test_median_abs_return": test_score["median_abs_return"],
                        }
                    )
    return pd.DataFrame(output)


def _apply_filter(
    rows: pd.DataFrame,
    feature: str,
    operator: str,
    threshold: float,
) -> pd.DataFrame:
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return rows[values <= threshold]
    return rows[values >= threshold]


def _random_same_count_rate(
    rows: pd.DataFrame,
    *,
    count: int,
    role: str,
    direction: int,
    horizon: int,
    config: SidelinedPersonalityCrossRegimeConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if rows.empty or count <= 0:
        return math.nan, math.nan
    count = min(count, len(rows))
    rates: list[float] = []
    medians: list[float] = []
    for _ in range(config.random_iterations):
        sampled = rows.sample(n=count, replace=False, random_state=int(rng.integers(0, 2**32 - 1)))
        score = _score_rows(
            sampled,
            role=role,
            direction=direction,
            horizon=horizon,
            low_movement_threshold=config.low_movement_threshold,
        )
        rates.append(float(score["same_result_rate"]))
        medians.append(float(score["median_aligned_return"]))
    return float(np.nanmean(rates)), float(np.nanmedian(medians))


def _candidate_filter_results(
    rows: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    config: SidelinedPersonalityCrossRegimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output: list[dict[str, Any]] = []
    if rows.empty or regimes.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    rng = np.random.default_rng(config.random_seed)
    regimes_to_scan = regimes.sort_values(
        ["test_lift_vs_personality", "regime_test_count"],
        ascending=[False, False],
    )
    regimes_to_scan = (
        regimes_to_scan.groupby(["personality", "event_state", "horizon"], as_index=False)
        .head(config.max_regimes_per_personality_horizon)
        .reset_index(drop=True)
    )
    grouped = {
        (personality, event_state): group
        for (personality, event_state), group in rows.groupby(
            ["personality", "event_state"],
            sort=False,
        )
    }
    for _, regime in regimes_to_scan.iterrows():
        key = (str(regime["personality"]), str(regime["event_state"]))
        group = grouped.get(key)
        if group is None:
            continue
        role = str(regime["role"])
        direction = int(regime["default_expected_direction"])
        horizon = int(regime["horizon"])
        regime_field = str(regime["regime_field"])
        regime_value = str(regime["regime_value"])
        ret_col = _return_col(horizon)
        train, test = _split_train_test(group, config.train_fraction)
        train = train[train[ret_col].notna() & train[regime_field].astype(str).eq(regime_value)]
        test = test[test[ret_col].notna() & test[regime_field].astype(str).eq(regime_value)]
        if len(train) < config.min_train_events or len(test) < config.min_test_events:
            continue
        regime_test_score = _score_rows(
            test,
            role=role,
            direction=direction,
            horizon=horizon,
            low_movement_threshold=config.low_movement_threshold,
        )
        for feature in config.filter_features:
            if feature not in train or feature not in test:
                continue
            numeric = pd.to_numeric(train[feature], errors="coerce").dropna()
            if numeric.nunique() < 2:
                continue
            for quantile in config.quantiles:
                threshold = float(numeric.quantile(quantile))
                for operator in ("<=", ">="):
                    retained_train = _apply_filter(train, feature, operator, threshold)
                    retained_test = _apply_filter(test, feature, operator, threshold)
                    if (
                        len(retained_train) < config.min_retained_events
                        or len(retained_test) < config.min_retained_events
                    ):
                        continue
                    train_score = _score_rows(
                        retained_train,
                        role=role,
                        direction=direction,
                        horizon=horizon,
                        low_movement_threshold=config.low_movement_threshold,
                    )
                    test_score = _score_rows(
                        retained_test,
                        role=role,
                        direction=direction,
                        horizon=horizon,
                        low_movement_threshold=config.low_movement_threshold,
                    )
                    random_rate, random_median = _random_same_count_rate(
                        test,
                        count=len(retained_test),
                        role=role,
                        direction=direction,
                        horizon=horizon,
                        config=config,
                        rng=rng,
                    )
                    symbol_count = int(retained_test["symbol"].astype(str).nunique())
                    single_symbol_share = _max_share(retained_test, "symbol")
                    single_session_share = _max_share(retained_test, "session_date")
                    single_month_share = _max_share(retained_test, "month")
                    test_lift = float(
                        test_score["same_result_rate"] - regime_test_score["same_result_rate"]
                    )
                    excess_vs_random = float(test_score["same_result_rate"] - random_rate)
                    reject_reasons: list[str] = []
                    if symbol_count < config.min_symbols:
                        reject_reasons.append("low_symbol_count")
                    if single_symbol_share > config.max_single_symbol_share:
                        reject_reasons.append("single_symbol_dominated")
                    if single_session_share > config.max_single_session_share:
                        reject_reasons.append("single_session_dominated")
                    if single_month_share > config.max_single_month_share:
                        reject_reasons.append("single_month_dominated")
                    if test_lift < config.min_oos_lift:
                        reject_reasons.append("no_oos_lift_vs_regime")
                    if excess_vs_random <= 0:
                        reject_reasons.append("random_same_count_not_beaten")
                    verdict = (
                        "promote_for_retest"
                        if not reject_reasons
                        else "reject_sidelined_filter"
                    )
                    output.append(
                        {
                            "personality": key[0],
                            "event_state": key[1],
                            "role": role,
                            "role_objective": "low_movement"
                            if role in NO_TRADE_ROLES or direction == 0
                            else "aligned_direction",
                            "default_expected_direction": direction,
                            "horizon": horizon,
                            "regime_field": regime_field,
                            "regime_value": regime_value,
                            "filter_rule": f"{feature} {operator} {threshold:.8g}",
                            "feature": feature,
                            "operator": operator,
                            "threshold": threshold,
                            "retained_train_count": int(len(retained_train)),
                            "retained_test_count": int(len(retained_test)),
                            "regime_test_same_result_rate": regime_test_score[
                                "same_result_rate"
                            ],
                            "filtered_train_same_result_rate": train_score[
                                "same_result_rate"
                            ],
                            "filtered_test_same_result_rate": test_score["same_result_rate"],
                            "test_lift_vs_regime": test_lift,
                            "filtered_test_median_aligned_return": test_score[
                                "median_aligned_return"
                            ],
                            "filtered_test_median_abs_return": test_score["median_abs_return"],
                            "random_same_count_mean_rate": random_rate,
                            "random_same_count_median_aligned_return": random_median,
                            "excess_vs_random_same_count": excess_vs_random,
                            "symbol_count": symbol_count,
                            "single_symbol_share": single_symbol_share,
                            "session_count": int(
                                retained_test["session_date"].astype(str).nunique()
                            ),
                            "single_session_share": single_session_share,
                            "month_count": _month_count(retained_test),
                            "single_month_share": single_month_share,
                            "verdict": verdict,
                            "reject_reasons": ";".join(reject_reasons),
                        }
                    )
    candidates = pd.DataFrame(output)
    if candidates.empty:
        return candidates, candidates, candidates
    selected = candidates[candidates["verdict"].eq("promote_for_retest")].copy()
    rejected = candidates[~candidates["verdict"].eq("promote_for_retest")].copy()
    selected = selected.sort_values(
        ["excess_vs_random_same_count", "test_lift_vs_regime", "retained_test_count"],
        ascending=[False, False, False],
    )
    rejected = rejected.sort_values(
        ["personality", "horizon", "regime_field", "feature", "operator"],
        kind="mergesort",
    )
    return candidates, selected, rejected


def _external_report_evidence(external_report_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report_dir in external_report_dirs:
        selected_path = report_dir / "selected_exhaustion_regime_filter_results.csv"
        if not selected_path.exists():
            rows.append(
                {
                    "report_dir": str(report_dir),
                    "personality": "unknown",
                    "status": "missing_selected_exhaustion_regime_filter_results",
                }
            )
            continue
        summary_path = report_dir / "summary.json"
        decision = ""
        if summary_path.exists():
            try:
                decision = str(
                    json.loads(summary_path.read_text(encoding="utf-8")).get("decision", "")
                )
            except json.JSONDecodeError:
                decision = "summary_json_parse_failed"
        selected = pd.read_csv(selected_path)
        if selected.empty:
            rows.append(
                {
                    "report_dir": str(report_dir),
                    "personality": "exhaustion_extension",
                    "role": "mean_reversion_or_no_chase",
                    "status": "empty_external_report",
                    "external_decision": decision,
                }
            )
            continue
        if "verdict" in selected:
            verdicts = selected.get("verdict", pd.Series("", index=selected.index)).astype(str)
            passing = selected[verdicts.str.startswith("pass")]
        else:
            passing = selected
        source = passing if not passing.empty else selected.head(1)
        for _, row in source.head(12).iterrows():
            rows.append(
                {
                    "report_dir": str(report_dir),
                    "personality": "exhaustion_extension",
                    "role": "mean_reversion_or_no_chase",
                    "status": "external_evidence_loaded",
                    "external_decision": decision,
                    "horizon": row.get("horizon", ""),
                    "regime_field": row.get("regime_field", ""),
                    "regime_value": row.get("regime_value", ""),
                    "filter_rule": row.get("filter_rule", ""),
                    "retained_test_count": row.get("retained_test_count", math.nan),
                    "filtered_test_same_result_rate": row.get(
                        "filtered_test_same_result_rate",
                        math.nan,
                    ),
                    "test_lift_vs_regime": row.get("test_lift_vs_regime", math.nan),
                    "filtered_test_median_aligned_return": row.get(
                        "filtered_test_median_aligned_return",
                        math.nan,
                    ),
                    "symbol_count": row.get("symbol_count", math.nan),
                    "single_symbol_share": row.get("single_symbol_share", math.nan),
                    "verdict": row.get("verdict", ""),
                }
            )
    return pd.DataFrame(rows)


def _concentration_warnings(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["personality", "horizon", "warning", "value", "threshold"]
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    for _, row in candidates.iterrows():
        for column, threshold in (
            ("single_symbol_share", 0.50),
            ("single_session_share", 0.20),
            ("single_month_share", 0.50),
        ):
            value = float(row.get(column, 0.0))
            if value > threshold:
                rows.append(
                    {
                        "personality": row.get("personality", ""),
                        "horizon": row.get("horizon", ""),
                        "warning": column,
                        "value": value,
                        "threshold": threshold,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _decide(
    selected: pd.DataFrame,
    external: pd.DataFrame,
    *,
    sparse_sidelined_count: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    external_pass = False
    if not external.empty and "verdict" in external:
        external_pass = external["verdict"].astype(str).str.startswith("pass").any()
    if not selected.empty or external_pass:
        return "continue_research_sidelined_personality", reasons
    if sparse_sidelined_count == 0:
        reasons.append("no_sparse_sidelined_personalities_available")
    reasons.append("no_sidelined_filter_survived_oos_random_concentration_gates")
    return "reject_no_sidelined_personality_promoted", reasons


def _write_summary_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    personality_summary: pd.DataFrame,
    selected: pd.DataFrame,
    external: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    selected_cols = [
        "personality",
        "role",
        "horizon",
        "regime_field",
        "regime_value",
        "filter_rule",
        "retained_test_count",
        "filtered_test_same_result_rate",
        "excess_vs_random_same_count",
    ]
    personality_cols = [
        "personality",
        "event_state",
        "role",
        "horizon",
        "test_event_count",
        "test_same_result_rate",
        "test_low_movement_rate",
    ]
    external_cols = [
        "personality",
        "horizon",
        "regime_field",
        "regime_value",
        "filter_rule",
        "retained_test_count",
        "filtered_test_same_result_rate",
        "verdict",
    ]
    text = f"""# Sidelined Personality Cross-Regime V0

Decision: {summary["decision"]}

Research-only: {summary["research_only"]}
Live ordering enabled: {summary["live_ordering_enabled"]}
Order placement: {summary["order_placement"]}
Edge claimed: {summary["edge_claimed"]}
Volume label: {summary["volume_label"]}

## What Was Tested

Promoted selected-filter personalities were excluded first. The sparse
event rows then tested only personalities not already in the selected-filter
book. Optional external specialty reports were summarized separately.

Sparse sidelined personalities tested: {summary["sparse_sidelined_personality_count"]}
External report dirs loaded: {summary["external_report_dir_count"]}

## Sidelined Personality Summary

{_markdown_table(personality_summary, personality_cols)}

## Selected Sidelined Candidates

{_markdown_table(selected, selected_cols)}

## External Specialty Evidence

{_markdown_table(external, external_cols)}

## Concentration Warnings

{_markdown_table(concentration, ["personality", "horizon", "warning", "value", "threshold"])}

## Plain English

This report does not claim an edge. A selected row only means the sidelined
personality is worth retesting inside the role-aware selected-filter workflow.
No-trade personalities are scored by low movement, not by positive returns.
Directional personalities are scored by aligned direction.
"""
    path.write_text(text, encoding="utf-8")


def run_sidelined_personality_cross_regime_lab(
    *,
    input_event_dir: Path,
    input_selected_filter_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    external_report_dirs: Sequence[Path] = (),
    config: SidelinedPersonalityCrossRegimeConfig = SidelinedPersonalityCrossRegimeConfig(),
) -> SidelinedPersonalityCrossRegimeResult:
    """Run a research-only crossed-regime diagnostic for sidelined personalities."""

    events = _load_event_rows(input_event_dir, config=config)
    promoted = _load_promoted_personalities(input_selected_filter_dir)
    sidelined = _sidelined_sparse_rows(events, promoted)
    personality = _personality_summary(sidelined, config=config)
    regimes = _crossed_regime_summary(sidelined, config=config)
    candidates, selected, rejected = _candidate_filter_results(sidelined, regimes, config=config)
    external = _external_report_evidence(external_report_dirs)
    concentration = _concentration_warnings(candidates)
    decision, decision_reasons = _decide(
        selected,
        external,
        sparse_sidelined_count=(
            int(sidelined["personality"].nunique()) if not sidelined.empty else 0
        ),
    )

    run_id = "sidelined_personality_cross_regime_v0_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision": run_dir / "decision.json",
        "personality": run_dir / "sidelined_personality_summary.csv",
        "regimes": run_dir / "crossed_regime_summary.csv",
        "candidates": run_dir / "candidate_filter_results.csv",
        "selected": run_dir / "selected_sidelined_candidates.csv",
        "rejected": run_dir / "rejected_sidelined_candidates.csv",
        "external": run_dir / "external_report_evidence.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }

    summary = {
        "run_id": run_id,
        "input_event_dir": str(input_event_dir),
        "input_selected_filter_dir": str(input_selected_filter_dir),
        "external_report_dirs": [str(path) for path in external_report_dirs],
        "output_dir": str(run_dir),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "volume_label": "historical_volume from existing local 5m OHLCV event report",
        "promoted_personality_count": len(promoted),
        "sparse_sidelined_personality_count": int(personality["personality"].nunique())
        if not personality.empty
        else 0,
        "sparse_sidelined_event_rows": int(len(sidelined)),
        "crossed_regime_rows": int(len(regimes)),
        "candidate_filter_rows": int(len(candidates)),
        "selected_candidate_count": int(len(selected)),
        "external_report_dir_count": len(external_report_dirs),
        "external_evidence_rows": int(len(external)),
        "horizons": list(config.horizons),
        "regime_fields": list(config.regime_fields),
        "filter_features": list(config.filter_features),
        "max_regimes_per_personality_horizon": config.max_regimes_per_personality_horizon,
        "random_iterations": config.random_iterations,
        "low_movement_threshold": config.low_movement_threshold,
    }
    decision_payload = {
        "decision": decision,
        "decision_reasons": decision_reasons,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }

    for frame_path, frame in (
        (paths["personality"], personality),
        (paths["regimes"], regimes),
        (paths["candidates"], candidates),
        (paths["selected"], selected),
        (paths["rejected"], rejected),
        (paths["external"], external),
        (paths["concentration"], concentration),
    ):
        _write_csv(frame_path, frame)
    _write_json(paths["summary_json"], summary)
    _write_json(paths["decision"], decision_payload)
    _write_summary_markdown(
        paths["summary_md"],
        summary=summary,
        personality_summary=personality,
        selected=selected,
        external=external,
        concentration=concentration,
    )

    return SidelinedPersonalityCrossRegimeResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_selected_filter_dir=input_selected_filter_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision"],
        sidelined_personality_summary_csv_path=paths["personality"],
        crossed_regime_summary_csv_path=paths["regimes"],
        candidate_filter_results_csv_path=paths["candidates"],
        selected_sidelined_candidates_csv_path=paths["selected"],
        rejected_sidelined_candidates_csv_path=paths["rejected"],
        external_report_evidence_csv_path=paths["external"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        selected_candidate_count=int(len(selected)),
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "SidelinedPersonalityCrossRegimeConfig",
    "SidelinedPersonalityCrossRegimeResult",
    "run_sidelined_personality_cross_regime_lab",
]
