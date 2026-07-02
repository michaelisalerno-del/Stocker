"""Research-only conditional context caveat lab.

This lab scans vetoes shaped as:

IF categorical context == value THEN numeric threshold blocks the trade.

It consumes a local state-lifecycle context report, writes diagnostic artifacts,
and never touches broker execution, paper trading, live trading, order
placement, or vendor fetching.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DEFAULT_OUTPUT_DIR = Path("data/reports/research/conditional_context_caveat_v0")

STRICT_TRAIN_AND_OOS_SUPPORTED = "strict_train_and_oos_supported"
OOS_ONLY_NOT_TRAIN_SUPPORTED = "oos_only_not_train_supported"
TRAIN_ONLY_NOT_OOS_SUPPORTED = "train_only_not_oos_supported"
NOT_SUPPORTED = "not_supported"

EXCLUDED_CONDITION_COLUMNS: frozenset[str] = frozenset(
    {
        "symbol",
        "timestamp",
        "session_date",
        "month",
        "split",
        "net_r",
        "entry_timestamp",
        "exit_timestamp",
    }
)
EXCLUDED_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "net_r",
        "bar_index_in_session",
        "horizon",
        "target_r",
        "expected_direction",
    }
)


@dataclass(frozen=True)
class ConditionalContextCaveatConfig:
    """Configuration for conditional context caveat diagnostics."""

    train_months: tuple[str, ...] = ("2026-01", "2026-02", "2026-03", "2026-04")
    test_months: tuple[str, ...] = ("2026-05", "2026-06")
    condition_features: tuple[str, ...] = ()
    numeric_features: tuple[str, ...] = ()
    numeric_operators: tuple[str, ...] = ("<=", ">=")
    numeric_quantiles: tuple[float, ...] = (0.20, 0.33, 0.50, 0.67, 0.80)
    random_iterations: int = 3000
    random_seed: int = 1337
    min_train_condition_count: int = 8
    min_train_flagged_count: int = 3
    min_oos_flagged_count: int = 1
    max_condition_values: int = 30
    max_selected_rules: int = 40
    max_flag_rules: int = 40
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20


@dataclass(frozen=True)
class ConditionalContextCaveatResult:
    """Paths and headline result for one conditional context caveat run."""

    run_id: str
    input_context_report_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    conditional_caveat_results_csv_path: Path
    selected_conditional_caveats_csv_path: Path
    strict_validation_results_csv_path: Path
    trade_conditional_caveat_flags_csv_path: Path
    decision: str
    selected_caveat_count: int


@dataclass(frozen=True)
class _Candidate:
    rule_name: str
    condition_feature: str
    condition_value: str
    feature: str
    operator: str
    selected_threshold: float
    selected_train_quantile: float
    mask: pd.Series


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


def _load_context_trades(input_context_report_dir: Path) -> pd.DataFrame:
    path = input_context_report_dir / "trade_context_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing context features: {path}")
    data = pd.read_csv(path)
    required = {"symbol", "timestamp", "session_date", "net_r"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"trade_context_features.csv missing required columns: {missing}")
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data[data["timestamp"].notna()].copy()
    data["session_date"] = pd.to_datetime(data["session_date"]).dt.strftime("%Y-%m-%d")
    if "month" not in data:
        data["month"] = data["timestamp"].dt.strftime("%Y-%m")
    else:
        data["month"] = data["month"].astype(str)
    if "split" in data:
        data["split"] = data["split"].astype(str)
    data["net_r"] = pd.to_numeric(data["net_r"], errors="coerce")
    data = data[data["net_r"].notna()].reset_index(drop=True)
    return data


def _split_masks(
    rows: pd.DataFrame,
    *,
    config: ConditionalContextCaveatConfig,
) -> tuple[pd.Series, pd.Series, str]:
    if "split" in rows and rows["split"].astype(str).isin({"train", "test"}).any():
        split = rows["split"].astype(str)
        return split.eq("train"), split.eq("test"), "split_column"
    month = rows["month"].astype(str)
    return (
        month.isin(config.train_months),
        month.isin(config.test_months),
        "configured_months",
    )


def _summary_stats(rows: pd.DataFrame) -> dict[str, int | float]:
    if rows.empty:
        return {
            "count": 0,
            "total_net_r": 0.0,
            "mean_net_r": math.nan,
            "win_rate": math.nan,
            **_concentration(rows),
        }
    net_r = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0)
    return {
        "count": int(len(rows)),
        "total_net_r": float(net_r.sum()),
        "mean_net_r": float(net_r.mean()) if len(net_r) else math.nan,
        "win_rate": float((net_r > 0.0).mean()) if len(net_r) else math.nan,
        **_concentration(rows),
    }


def _concentration(rows: pd.DataFrame) -> dict[str, int | float]:
    if rows.empty:
        return {
            "symbol_count": 0,
            "session_count": 0,
            "month_count": 0,
            "single_symbol_share": math.nan,
            "single_session_share": math.nan,
            "single_month_share": math.nan,
        }
    symbol_counts = rows["symbol"].astype(str).value_counts() if "symbol" in rows else pd.Series()
    if {"symbol", "session_date"}.issubset(rows.columns):
        session_counts = rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1)
        session_counts = session_counts.value_counts()
    elif "session_date" in rows:
        session_counts = rows["session_date"].astype(str).value_counts()
    else:
        session_counts = pd.Series()
    month_counts = rows["month"].astype(str).value_counts() if "month" in rows else pd.Series()
    return {
        "symbol_count": int(symbol_counts.size),
        "session_count": int(session_counts.size),
        "month_count": int(month_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows))
        if not symbol_counts.empty
        else math.nan,
        "single_session_share": float(session_counts.iloc[0] / len(rows))
        if not session_counts.empty
        else math.nan,
        "single_month_share": float(month_counts.iloc[0] / len(rows))
        if not month_counts.empty
        else math.nan,
    }


def _has_concentration_warning(
    concentration: dict[str, int | float],
    *,
    config: ConditionalContextCaveatConfig,
) -> bool:
    symbol_share = float(concentration.get("single_symbol_share", math.nan))
    session_share = float(concentration.get("single_session_share", math.nan))
    return (
        not math.isnan(symbol_share)
        and symbol_share > config.max_single_symbol_share
        or not math.isnan(session_share)
        and session_share > config.max_single_session_share
    )


def _random_kept_baseline(
    test_rows: pd.DataFrame,
    flagged_count: int,
    *,
    config: ConditionalContextCaveatConfig,
    rule_name: str,
) -> dict[str, float | int]:
    if test_rows.empty:
        return {
            "random_kept_net_r_median": math.nan,
            "random_kept_net_r_p05": math.nan,
            "random_kept_net_r_p95": math.nan,
            "random_same_count_reps": int(config.random_iterations),
        }
    net_r = pd.to_numeric(test_rows["net_r"], errors="coerce").fillna(0.0).to_numpy()
    if flagged_count <= 0:
        kept_totals: NDArray[np.float64] = np.repeat(
            float(net_r.sum()),
            config.random_iterations,
        )
    elif flagged_count >= len(net_r):
        kept_totals = np.zeros(config.random_iterations)
    else:
        digest = hashlib.sha256(rule_name.encode("utf-8")).digest()
        seed_offset = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(config.random_seed + seed_offset)
        kept_totals = np.empty(config.random_iterations)
        all_indices = np.arange(len(net_r))
        total = float(net_r.sum())
        for index in range(config.random_iterations):
            blocked = rng.choice(all_indices, size=flagged_count, replace=False)
            kept_totals[index] = total - float(net_r[blocked].sum())
    return {
        "random_kept_net_r_median": float(np.median(kept_totals)),
        "random_kept_net_r_p05": float(np.quantile(kept_totals, 0.05)),
        "random_kept_net_r_p95": float(np.quantile(kept_totals, 0.95)),
        "random_same_count_reps": int(config.random_iterations),
    }


def _strict_status(train_supported: bool, oos_supported: bool) -> str:
    if train_supported and oos_supported:
        return STRICT_TRAIN_AND_OOS_SUPPORTED
    if oos_supported:
        return OOS_ONLY_NOT_TRAIN_SUPPORTED
    if train_supported:
        return TRAIN_ONLY_NOT_OOS_SUPPORTED
    return NOT_SUPPORTED


def _infer_condition_features(
    rows: pd.DataFrame,
    *,
    config: ConditionalContextCaveatConfig,
) -> list[str]:
    if config.condition_features:
        return [feature for feature in config.condition_features if feature in rows]
    features: list[str] = []
    for column in rows.columns:
        if column in EXCLUDED_CONDITION_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(rows[column]):
            continue
        non_null = rows[column].dropna()
        if non_null.empty:
            continue
        if int(non_null.astype(str).nunique()) <= config.max_condition_values:
            features.append(str(column))
    return features


def _infer_numeric_features(
    rows: pd.DataFrame,
    *,
    config: ConditionalContextCaveatConfig,
) -> list[str]:
    if config.numeric_features:
        return [feature for feature in config.numeric_features if feature in rows]
    features: list[str] = []
    for column in rows.columns:
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.dropna().nunique() >= 2:
            features.append(str(column))
    return features


def _numeric_thresholds(
    rows: pd.DataFrame,
    feature: str,
    quantiles: tuple[float, ...],
) -> list[tuple[float, float]]:
    values = pd.to_numeric(rows[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.nunique() < 2:
        return []
    thresholds: dict[float, float] = {}
    for quantile in quantiles:
        threshold = float(values.quantile(quantile))
        if not math.isnan(threshold):
            thresholds[threshold] = float(quantile)
    return [(threshold, thresholds[threshold]) for threshold in sorted(thresholds)]


def _numeric_mask(
    rows: pd.DataFrame,
    *,
    feature: str,
    operator: str,
    threshold: float,
) -> pd.Series:
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return (values <= threshold).fillna(False)
    if operator == ">=":
        return (values >= threshold).fillna(False)
    raise ValueError(f"Unsupported numeric operator: {operator}")


def _condition_mask(rows: pd.DataFrame, *, feature: str, value: str) -> pd.Series:
    if feature not in rows:
        return pd.Series(False, index=rows.index)
    return rows[feature].astype(str).eq(value)


def _build_candidates(
    rows: pd.DataFrame,
    *,
    config: ConditionalContextCaveatConfig,
    train_mask: pd.Series,
) -> list[_Candidate]:
    condition_features = _infer_condition_features(rows, config=config)
    numeric_features = _infer_numeric_features(rows, config=config)
    train_rows = rows[train_mask].copy()
    candidates: list[_Candidate] = []
    for condition_feature in condition_features:
        train_values = train_rows[condition_feature].dropna().astype(str)
        if train_values.empty:
            continue
        value_counts = train_values.value_counts()
        for condition_value, condition_count in value_counts.items():
            condition_text = str(condition_value)
            if int(condition_count) < config.min_train_condition_count:
                continue
            condition_train = train_rows[
                train_rows[condition_feature].astype(str).eq(condition_text)
            ]
            condition_all_mask = _condition_mask(
                rows,
                feature=condition_feature,
                value=condition_text,
            )
            for numeric_feature in numeric_features:
                if numeric_feature == condition_feature or numeric_feature not in condition_train:
                    continue
                for threshold, quantile in _numeric_thresholds(
                    condition_train,
                    numeric_feature,
                    config.numeric_quantiles,
                ):
                    for operator in config.numeric_operators:
                        numeric = _numeric_mask(
                            rows,
                            feature=numeric_feature,
                            operator=operator,
                            threshold=threshold,
                        )
                        rule_name = (
                            "conditional_context: IF "
                            f"{condition_feature} == {condition_value} THEN "
                            f"{numeric_feature} {operator} {threshold:.6g}"
                        )
                        candidates.append(
                            _Candidate(
                                rule_name=rule_name,
                                condition_feature=condition_feature,
                                condition_value=condition_text,
                                feature=numeric_feature,
                                operator=operator,
                                selected_threshold=threshold,
                                selected_train_quantile=quantile,
                                mask=condition_all_mask & numeric,
                            )
                        )
    return candidates


def _evaluate_candidate(
    rows: pd.DataFrame,
    candidate: _Candidate,
    *,
    config: ConditionalContextCaveatConfig,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> dict[str, Any]:
    mask = candidate.mask.reindex(rows.index, fill_value=False).fillna(False).astype(bool)
    base_train = rows[train_mask]
    base_test = rows[test_mask]
    flagged_train = rows[train_mask & mask]
    kept_train = rows[train_mask & ~mask]
    flagged_test = rows[test_mask & mask]
    kept_test = rows[test_mask & ~mask]
    base_train_stats = _summary_stats(base_train)
    base_test_stats = _summary_stats(base_test)
    train_flagged_total = float(flagged_train["net_r"].sum()) if not flagged_train.empty else 0.0
    test_flagged_total = float(flagged_test["net_r"].sum()) if not flagged_test.empty else 0.0
    train_kept_total = float(kept_train["net_r"].sum()) if not kept_train.empty else 0.0
    test_kept_total = float(kept_test["net_r"].sum()) if not kept_test.empty else 0.0
    train_kept_lift = train_kept_total - float(base_train_stats["total_net_r"])
    test_kept_lift = test_kept_total - float(base_test_stats["total_net_r"])
    test_concentration = _concentration(flagged_test)
    concentration_warning = _has_concentration_warning(test_concentration, config=config)
    random_baseline = _random_kept_baseline(
        base_test,
        int(len(flagged_test)),
        config=config,
        rule_name=candidate.rule_name,
    )
    test_excess = test_kept_total - float(random_baseline["random_kept_net_r_median"])
    strict_train_supported = (
        len(flagged_train) >= config.min_train_flagged_count
        and train_flagged_total < 0.0
        and train_kept_lift > 0.0
    )
    strict_oos_supported = (
        len(flagged_test) >= config.min_oos_flagged_count
        and test_flagged_total < 0.0
        and test_kept_lift > 0.0
        and test_excess > 0.0
        and not concentration_warning
    )
    return {
        "rule_name": candidate.rule_name,
        "rule_family": "train_selected_conditional_context_numeric",
        "condition_feature": candidate.condition_feature,
        "condition_operator": "==",
        "condition_value": candidate.condition_value,
        "feature": candidate.feature,
        "operator": candidate.operator,
        "selected_threshold": candidate.selected_threshold,
        "selected_train_quantile": candidate.selected_train_quantile,
        "train_selected": False,
        "base_train_count": int(base_train_stats["count"]),
        "base_train_total_net_r": float(base_train_stats["total_net_r"]),
        "base_test_count": int(base_test_stats["count"]),
        "base_test_total_net_r": float(base_test_stats["total_net_r"]),
        "flagged_count": int(mask.sum()),
        "train_flagged_count": int(len(flagged_train)),
        "train_flagged_total_net_r": train_flagged_total,
        "train_kept_count": int(len(kept_train)),
        "train_kept_total_net_r": train_kept_total,
        "train_kept_lift_vs_base_r": train_kept_lift,
        "test_flagged_count": int(len(flagged_test)),
        "test_flagged_total_net_r": test_flagged_total,
        "test_kept_count": int(len(kept_test)),
        "test_kept_total_net_r": test_kept_total,
        "test_kept_lift_vs_base_r": test_kept_lift,
        "test_excess_vs_random_median_r": test_excess,
        **random_baseline,
        "test_flagged_symbol_count": int(test_concentration["symbol_count"]),
        "test_flagged_session_count": int(test_concentration["session_count"]),
        "test_flagged_single_symbol_share": test_concentration["single_symbol_share"],
        "test_flagged_single_session_share": test_concentration["single_session_share"],
        "concentration_warning": concentration_warning,
        "strict_train_supported": strict_train_supported,
        "strict_oos_supported": strict_oos_supported,
        "strict_status": _strict_status(strict_train_supported, strict_oos_supported),
    }


def _sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    status_rank = {
        STRICT_TRAIN_AND_OOS_SUPPORTED: 0,
        TRAIN_ONLY_NOT_OOS_SUPPORTED: 1,
        OOS_ONLY_NOT_TRAIN_SUPPORTED: 2,
        NOT_SUPPORTED: 3,
    }
    data = frame.copy()
    data["_status_rank"] = data["strict_status"].map(status_rank).fillna(9).astype(int)
    data = data.sort_values(
        [
            "_status_rank",
            "train_selected",
            "test_kept_lift_vs_base_r",
            "test_excess_vs_random_median_r",
            "train_kept_lift_vs_base_r",
            "train_flagged_count",
        ],
        ascending=[True, False, False, False, False, False],
        kind="mergesort",
    )
    return data.drop(columns=["_status_rank"]).reset_index(drop=True)


def _mark_train_selected(
    results: pd.DataFrame,
    *,
    config: ConditionalContextCaveatConfig,
) -> pd.DataFrame:
    if results.empty:
        return results
    data = results.copy()
    data["train_selected"] = False
    train_supported = data[data["strict_train_supported"].astype(bool)].copy()
    if train_supported.empty:
        return _sort_results(data)
    groups = train_supported.groupby(
        ["condition_feature", "condition_value", "feature", "operator"],
        dropna=False,
    )
    selected_indices: list[int] = []
    for _, group in groups:
        selected_index = group.sort_values(
            [
                "train_kept_lift_vs_base_r",
                "train_flagged_total_net_r",
                "train_flagged_count",
                "selected_threshold",
            ],
            ascending=[False, True, False, True],
            kind="mergesort",
        ).index[0]
        selected_indices.append(int(selected_index))
    ranked = data.loc[selected_indices].sort_values(
        ["train_kept_lift_vs_base_r", "train_flagged_total_net_r", "train_flagged_count"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    data.loc[ranked.head(config.max_selected_rules).index, "train_selected"] = True
    return _sort_results(data)


def _decision(strict_results: pd.DataFrame) -> tuple[str, list[str]]:
    if strict_results.empty:
        return "continue_research_no_conditional_caveat_supported", ["no_train_selected_rules"]
    strict_count = int(strict_results["strict_status"].eq(STRICT_TRAIN_AND_OOS_SUPPORTED).sum())
    train_only_count = int(strict_results["strict_status"].eq(TRAIN_ONLY_NOT_OOS_SUPPORTED).sum())
    oos_only_count = int(strict_results["strict_status"].eq(OOS_ONLY_NOT_TRAIN_SUPPORTED).sum())
    if strict_count:
        return "continue_research_strict_conditional_caveat_supported", []
    if train_only_count:
        return (
            "continue_research_train_conditional_warning_not_oos_validated",
            ["train-selected conditional caveats did not hold in OOS"],
        )
    if oos_only_count:
        return (
            "continue_research_oos_conditional_warning_not_train_validated",
            ["conditional caveats worked only in OOS and should not be promoted"],
        )
    return "continue_research_no_conditional_caveat_supported", ["no strict caveat support"]


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug[:96] or "rule"


def _build_trade_flags(
    rows: pd.DataFrame,
    candidates: list[_Candidate],
) -> pd.DataFrame:
    referenced_columns: list[str] = []
    for candidate in candidates:
        for column in [candidate.condition_feature, candidate.feature]:
            if column in rows and column not in referenced_columns:
                referenced_columns.append(column)
    base_columns = [
        column
        for column in [
            "symbol",
            "timestamp",
            "session_date",
            "month",
            "split",
            "personality",
            "event_state",
            "net_r",
            *referenced_columns,
        ]
        if column in rows.columns
    ]
    flags = rows[base_columns].copy()
    flag_columns: dict[str, NDArray[np.bool_]] = {}
    used_names: set[str] = set()
    for candidate in candidates:
        column = f"flag_{_slug(candidate.rule_name)}"
        suffix = 2
        while column in used_names:
            column = f"flag_{_slug(candidate.rule_name)}_{suffix}"
            suffix += 1
        used_names.add(column)
        flag_columns[column] = (
            candidate.mask.reindex(rows.index, fill_value=False)
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
    if flag_columns:
        flags = pd.concat([flags, pd.DataFrame(flag_columns, index=rows.index)], axis=1)
    return flags


def _candidate_key(row: pd.Series) -> tuple[str, str, str, str, float]:
    return (
        str(row["condition_feature"]),
        str(row["condition_value"]),
        str(row["feature"]),
        str(row["operator"]),
        float(row["selected_threshold"]),
    )


def _selected_candidates(
    candidates: list[_Candidate],
    strict_results: pd.DataFrame,
    *,
    max_rules: int,
) -> list[_Candidate]:
    if strict_results.empty:
        return []
    selected_keys = {
        _candidate_key(row)
        for _, row in strict_results.head(max_rules).iterrows()
    }
    return [
        candidate
        for candidate in candidates
        if (
            candidate.condition_feature,
            candidate.condition_value,
            candidate.feature,
            candidate.operator,
            float(candidate.selected_threshold),
        )
        in selected_keys
    ]


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 24) -> str:
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
    strict_results: pd.DataFrame,
) -> None:
    columns = [
        column
        for column in [
            "rule_name",
            "strict_status",
            "train_flagged_count",
            "train_flagged_total_net_r",
            "train_kept_lift_vs_base_r",
            "test_flagged_count",
            "test_flagged_total_net_r",
            "test_kept_lift_vs_base_r",
            "test_excess_vs_random_median_r",
            "concentration_warning",
        ]
        if column in strict_results.columns
    ]
    lines = [
        "# Conditional Context Caveat V0",
        "",
        (
            "Research-only scan for IF context/state/regime equals a value THEN apply a "
            "numeric veto threshold. Thresholds are selected on train only and evaluated "
            "OOS. No broker, live, paper, vendor fetch, order placement, or edge claim."
        ),
        "",
        f"Decision: `{payload['decision']}`",
        f"Input context report: `{payload['input_context_report_dir']}`",
        f"Split method: `{payload['split_method']}`",
        f"Train trades: `{payload['base_train']['count']}`",
        f"Test trades: `{payload['base_test']['count']}`",
        f"Candidate count: `{payload['candidate_count']}`",
        f"Train-selected caveat count: `{payload['train_selected_caveat_count']}`",
        "",
        "## Train-Selected Strict Validation",
        "",
        _markdown_table(strict_results[columns] if columns else strict_results),
        "",
        "## Safety",
        "",
        "- research_only: true",
        "- live_ordering_enabled: false",
        "- order_placement: disabled",
        "- edge_claimed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_conditional_context_caveat_lab(
    *,
    input_context_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: ConditionalContextCaveatConfig = ConditionalContextCaveatConfig(),
) -> ConditionalContextCaveatResult:
    """Run a reusable research-only conditional context caveat scan."""

    rows = _load_context_trades(input_context_report_dir)
    train_mask, test_mask, split_method = _split_masks(rows, config=config)
    candidates = _build_candidates(rows, config=config, train_mask=train_mask)
    all_results = pd.DataFrame(
        [
            _evaluate_candidate(
                rows,
                candidate,
                config=config,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            for candidate in candidates
        ]
    )
    all_results = _mark_train_selected(all_results, config=config)
    strict_results = (
        all_results[all_results["train_selected"].astype(bool)].copy()
        if not all_results.empty
        else pd.DataFrame()
    )
    selected = (
        strict_results[~strict_results["strict_status"].eq(NOT_SUPPORTED)].copy()
        if not strict_results.empty
        else pd.DataFrame()
    )
    selected_candidates = _selected_candidates(
        candidates,
        selected,
        max_rules=config.max_flag_rules,
    )
    trade_flags = _build_trade_flags(rows, selected_candidates)
    decision, decision_reasons = _decision(strict_results)

    run_id = "conditional_context_caveat_v0_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "results": run_dir / "conditional_caveat_results.csv",
        "selected": run_dir / "selected_conditional_caveats.csv",
        "strict": run_dir / "strict_validation_results.csv",
        "flags": run_dir / "trade_conditional_caveat_flags.csv",
    }
    for path, frame in [
        (paths["results"], all_results),
        (paths["selected"], selected),
        (paths["strict"], strict_results),
        (paths["flags"], trade_flags),
    ]:
        _write_csv(path, frame)

    base_all = _summary_stats(rows)
    base_train = _summary_stats(rows[train_mask])
    base_test = _summary_stats(rows[test_mask])
    status_counts = (
        all_results["strict_status"].value_counts().to_dict() if not all_results.empty else {}
    )
    selected_status_counts = (
        strict_results["strict_status"].value_counts().to_dict()
        if not strict_results.empty
        else {}
    )
    payload: dict[str, Any] = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "pipeline": "personality -> mixed_regime -> filter -> caveat -> exit",
        "input_context_report_dir": str(input_context_report_dir),
        "output_dir": str(run_dir),
        "split_method": split_method,
        "train_months": list(config.train_months),
        "test_months": list(config.test_months),
        "condition_features": _infer_condition_features(rows, config=config),
        "numeric_features": _infer_numeric_features(rows, config=config),
        "numeric_operators": list(config.numeric_operators),
        "numeric_quantiles": list(config.numeric_quantiles),
        "random_iterations": int(config.random_iterations),
        "base_all": base_all,
        "base_train": base_train,
        "base_test": base_test,
        "decision": decision,
        "decision_reasons": decision_reasons,
        "candidate_count": int(len(all_results)),
        "train_selected_caveat_count": int(len(strict_results)),
        "selected_supported_caveat_count": int(len(selected)),
        "strict_train_and_oos_supported_count": int(
            status_counts.get(STRICT_TRAIN_AND_OOS_SUPPORTED, 0)
        ),
        "oos_only_not_train_supported_count": int(
            status_counts.get(OOS_ONLY_NOT_TRAIN_SUPPORTED, 0)
        ),
        "train_only_not_oos_supported_count": int(
            status_counts.get(TRAIN_ONLY_NOT_OOS_SUPPORTED, 0)
        ),
        "not_supported_count": int(status_counts.get(NOT_SUPPORTED, 0)),
        "train_selected_status_counts": selected_status_counts,
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
        },
    )
    _write_summary_md(paths["summary_md"], payload=payload, strict_results=strict_results)

    return ConditionalContextCaveatResult(
        run_id=run_id,
        input_context_report_dir=input_context_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        conditional_caveat_results_csv_path=paths["results"],
        selected_conditional_caveats_csv_path=paths["selected"],
        strict_validation_results_csv_path=paths["strict"],
        trade_conditional_caveat_flags_csv_path=paths["flags"],
        decision=decision,
        selected_caveat_count=int(len(strict_results)),
    )


__all__ = [
    "ConditionalContextCaveatConfig",
    "ConditionalContextCaveatResult",
    "run_conditional_context_caveat_lab",
]
