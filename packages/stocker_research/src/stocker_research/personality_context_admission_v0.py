"""Research-only personality-specific context admission lab.

This lab compares a gated staged report against a more permissive staged report,
then studies candidate-only trades as blocked personality cohorts. It learns
personality-specific context re-entry/admission rules on train months only and
evaluates those rules out of sample. It never touches broker execution, paper
trading, live trading, order placement, or vendor fetching.
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

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_context_admission_v0")

STRICT_TRAIN_AND_OOS_SUPPORTED = "strict_train_and_oos_supported"
OOS_ONLY_NOT_TRAIN_SUPPORTED = "oos_only_not_train_supported"
TRAIN_ONLY_NOT_OOS_SUPPORTED = "train_only_not_oos_supported"
NOT_SUPPORTED = "not_supported"

DEFAULT_TRADE_KEY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "personality",
    "stop_model",
    "target_r",
    "monthly_candidate_rank",
    "selected_filter_rank",
)

EXCLUDED_CONTEXT_COLUMNS: frozenset[str] = frozenset(
    {
        "symbol",
        "timestamp",
        "session_date",
        "month",
        "net_r",
        "entry_timestamp",
        "exit_timestamp",
        "caveat_rule_name",
        "caveat_rule_family",
        "caveat_strict_status",
        "admission_rule_name",
        "admission_rule_status",
    }
)


@dataclass(frozen=True)
class PersonalityContextAdmissionConfig:
    """Configuration for personality-specific context admission diagnostics."""

    train_months: tuple[str, ...] = ("2026-01", "2026-02", "2026-03", "2026-04")
    test_months: tuple[str, ...] = ("2026-05", "2026-06")
    target_personalities: tuple[str, ...] = ()
    context_features: tuple[str, ...] = ()
    trade_key_columns: tuple[str, ...] = DEFAULT_TRADE_KEY_COLUMNS
    random_iterations: int = 3000
    random_seed: int = 1337
    min_train_admitted_count: int = 3
    min_oos_admitted_count: int = 1
    max_context_values: int = 40
    max_selected_rules_per_personality: int = 3
    max_flag_rules: int = 40
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20


@dataclass(frozen=True)
class PersonalityContextAdmissionResult:
    """Paths and headline result for one personality context admission run."""

    run_id: str
    input_baseline_report_dir: Path
    input_candidate_report_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    admission_rule_results_csv_path: Path
    selected_admissions_csv_path: Path
    blocked_candidate_trades_csv_path: Path
    trade_admission_flags_csv_path: Path
    decision: str
    selected_admission_count: int


@dataclass(frozen=True)
class _AdmissionCandidate:
    rule_name: str
    blocker_rule_name: str
    personality: str
    context_feature: str
    context_value: str
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


def _load_report_trades(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing staged trades: {path}")
    data = pd.read_csv(path)
    required = {"symbol", "timestamp", "personality", "net_r"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"trades.csv missing required columns: {missing}")
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data[data["timestamp"].notna()].copy()
    if "session_date" not in data:
        data["session_date"] = data["timestamp"].dt.strftime("%Y-%m-%d")
    else:
        data["session_date"] = pd.to_datetime(data["session_date"]).dt.strftime("%Y-%m-%d")
    if "month" not in data:
        data["month"] = data["timestamp"].dt.strftime("%Y-%m")
    else:
        data["month"] = data["month"].astype(str)
    data["personality"] = data["personality"].astype(str)
    data["net_r"] = pd.to_numeric(data["net_r"], errors="coerce")
    data = data[data["net_r"].notna()].reset_index(drop=True)
    return data


def _available_trade_key_columns(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
) -> list[str]:
    available = [
        column
        for column in config.trade_key_columns
        if column in baseline.columns and column in candidate.columns
    ]
    required = ["symbol", "timestamp", "personality"]
    for column in required:
        if column not in available and column in baseline.columns and column in candidate.columns:
            available.append(column)
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"Unable to build stable trade keys; missing columns: {missing}")
    return available


def _key_series(rows: pd.DataFrame, columns: list[str]) -> pd.Series:
    keyed = rows[columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(keyed[column]):
            keyed[column] = pd.to_datetime(keyed[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        elif pd.api.types.is_float_dtype(keyed[column]):
            keyed[column] = pd.to_numeric(keyed[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.12g}"
            )
        else:
            keyed[column] = keyed[column].astype(str)
    return keyed.agg("\x1f".join, axis=1)


def _blocked_candidate_trades(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
) -> pd.DataFrame:
    key_columns = _available_trade_key_columns(baseline, candidate, config=config)
    baseline_keys = set(_key_series(baseline, key_columns))
    candidate_keys = _key_series(candidate, key_columns)
    blocked = candidate[~candidate_keys.isin(baseline_keys)].copy()
    blocked["candidate_trade_key"] = candidate_keys[~candidate_keys.isin(baseline_keys)].to_numpy()
    blocked["blocker_family"] = "candidate_only_vs_baseline"
    blocked["blocker_personality"] = blocked["personality"].astype(str)
    blocked["blocker_rule_name"] = (
        "personality_context_blocker:"
        + blocked["personality"].astype(str)
        + ":candidate_only_vs_baseline"
    )
    blocked["target_personality_in_scope"] = True
    if config.target_personalities:
        allowed = set(config.target_personalities)
        blocked["target_personality_in_scope"] = blocked["personality"].astype(str).isin(allowed)
    return blocked.reset_index(drop=True)


def _split_masks(
    rows: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
) -> tuple[pd.Series, pd.Series]:
    month = rows["month"].astype(str)
    return month.isin(config.train_months), month.isin(config.test_months)


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
    config: PersonalityContextAdmissionConfig,
) -> bool:
    symbol_share = float(concentration.get("single_symbol_share", math.nan))
    session_share = float(concentration.get("single_session_share", math.nan))
    return (
        not math.isnan(symbol_share)
        and symbol_share > config.max_single_symbol_share
        or not math.isnan(session_share)
        and session_share > config.max_single_session_share
    )


def _random_same_count_admission_baseline(
    rows: pd.DataFrame,
    admitted_count: int,
    *,
    config: PersonalityContextAdmissionConfig,
    rule_name: str,
    split_name: str,
) -> dict[str, float | int]:
    if rows.empty:
        return {
            f"{split_name}_random_admitted_net_r_median": math.nan,
            f"{split_name}_random_admitted_net_r_p05": math.nan,
            f"{split_name}_random_admitted_net_r_p95": math.nan,
            f"{split_name}_random_same_count_reps": int(config.random_iterations),
        }
    net_r = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0).to_numpy()
    if admitted_count <= 0:
        totals: NDArray[np.float64] = np.zeros(config.random_iterations)
    elif admitted_count >= len(net_r):
        totals = np.repeat(float(net_r.sum()), config.random_iterations)
    else:
        digest = hashlib.sha256(f"{rule_name}|{split_name}".encode()).digest()
        seed_offset = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(config.random_seed + seed_offset)
        totals = np.empty(config.random_iterations)
        all_indices = np.arange(len(net_r))
        for index in range(config.random_iterations):
            admitted = rng.choice(all_indices, size=admitted_count, replace=False)
            totals[index] = float(net_r[admitted].sum())
    return {
        f"{split_name}_random_admitted_net_r_median": float(np.median(totals)),
        f"{split_name}_random_admitted_net_r_p05": float(np.quantile(totals, 0.05)),
        f"{split_name}_random_admitted_net_r_p95": float(np.quantile(totals, 0.95)),
        f"{split_name}_random_same_count_reps": int(config.random_iterations),
    }


def _strict_status(train_supported: bool, oos_supported: bool) -> str:
    if train_supported and oos_supported:
        return STRICT_TRAIN_AND_OOS_SUPPORTED
    if oos_supported:
        return OOS_ONLY_NOT_TRAIN_SUPPORTED
    if train_supported:
        return TRAIN_ONLY_NOT_OOS_SUPPORTED
    return NOT_SUPPORTED


def _infer_context_features(
    rows: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
) -> list[str]:
    if config.context_features:
        return [feature for feature in config.context_features if feature in rows]
    features: list[str] = []
    for column in rows.columns:
        if column in EXCLUDED_CONTEXT_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(rows[column]):
            continue
        non_null = rows[column].dropna()
        if non_null.empty:
            continue
        if int(non_null.astype(str).nunique()) <= config.max_context_values:
            features.append(str(column))
    return features


def _build_candidates(
    rows: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
    train_mask: pd.Series,
) -> list[_AdmissionCandidate]:
    context_features = _infer_context_features(rows, config=config)
    train_rows = rows[train_mask].copy()
    candidates: list[_AdmissionCandidate] = []
    if train_rows.empty or "personality" not in train_rows:
        return candidates
    for personality, personality_train in train_rows.groupby(
        train_rows["personality"].astype(str),
        dropna=False,
    ):
        personality_text = str(personality)
        personality_mask = rows["personality"].astype(str).eq(personality_text)
        blocker_rule_name = f"personality_context_blocker:{personality_text}"
        for context_feature in context_features:
            train_values = personality_train[context_feature].dropna().astype(str)
            if train_values.empty:
                continue
            for context_value, count in train_values.value_counts().items():
                if int(count) < config.min_train_admitted_count:
                    continue
                context_text = str(context_value)
                context_mask = rows[context_feature].astype(str).eq(context_text)
                rule_name = (
                    "personality_context_admission: "
                    f"{personality_text} IF {context_feature} == {context_text}"
                )
                candidates.append(
                    _AdmissionCandidate(
                        rule_name=rule_name,
                        blocker_rule_name=blocker_rule_name,
                        personality=personality_text,
                        context_feature=context_feature,
                        context_value=context_text,
                        mask=personality_mask & context_mask,
                    )
                )
    return candidates


def _evaluate_candidate(
    rows: pd.DataFrame,
    candidate: _AdmissionCandidate,
    *,
    config: PersonalityContextAdmissionConfig,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> dict[str, Any]:
    mask = candidate.mask.reindex(rows.index, fill_value=False).fillna(False).astype(bool)
    personality_mask = rows["personality"].astype(str).eq(candidate.personality)
    base_train = rows[train_mask & personality_mask]
    base_test = rows[test_mask & personality_mask]
    admitted_train = rows[train_mask & mask]
    admitted_test = rows[test_mask & mask]
    rejected_train = rows[train_mask & personality_mask & ~mask]
    rejected_test = rows[test_mask & personality_mask & ~mask]

    train_total = float(admitted_train["net_r"].sum()) if not admitted_train.empty else 0.0
    test_total = float(admitted_test["net_r"].sum()) if not admitted_test.empty else 0.0
    train_random = _random_same_count_admission_baseline(
        base_train,
        int(len(admitted_train)),
        config=config,
        rule_name=candidate.rule_name,
        split_name="train",
    )
    test_random = _random_same_count_admission_baseline(
        base_test,
        int(len(admitted_test)),
        config=config,
        rule_name=candidate.rule_name,
        split_name="test",
    )
    train_excess = train_total - float(train_random["train_random_admitted_net_r_median"])
    test_excess = test_total - float(test_random["test_random_admitted_net_r_median"])
    train_concentration = _concentration(admitted_train)
    test_concentration = _concentration(admitted_test)
    train_concentration_warning = _has_concentration_warning(train_concentration, config=config)
    test_concentration_warning = _has_concentration_warning(test_concentration, config=config)
    train_supported = (
        len(admitted_train) >= config.min_train_admitted_count
        and train_total > 0.0
        and train_excess > 0.0
    )
    oos_supported = (
        len(admitted_test) >= config.min_oos_admitted_count
        and test_total > 0.0
        and test_excess > 0.0
        and not test_concentration_warning
    )
    return {
        "rule_name": candidate.rule_name,
        "rule_family": "train_selected_personality_context_admission",
        "blocker_rule_name": candidate.blocker_rule_name,
        "blocker_family": "candidate_only_vs_baseline",
        "personality": candidate.personality,
        "context_feature": candidate.context_feature,
        "context_operator": "==",
        "context_value": candidate.context_value,
        "train_selected": False,
        "base_train_blocked_count": int(len(base_train)),
        "base_train_blocked_total_net_r": float(base_train["net_r"].sum())
        if not base_train.empty
        else 0.0,
        "base_test_blocked_count": int(len(base_test)),
        "base_test_blocked_total_net_r": float(base_test["net_r"].sum())
        if not base_test.empty
        else 0.0,
        "train_admitted_count": int(len(admitted_train)),
        "train_admitted_total_net_r": train_total,
        "train_admitted_mean_net_r": float(admitted_train["net_r"].mean())
        if not admitted_train.empty
        else math.nan,
        "train_admitted_win_rate": float((admitted_train["net_r"] > 0.0).mean())
        if not admitted_train.empty
        else math.nan,
        "train_not_admitted_count": int(len(rejected_train)),
        "train_not_admitted_total_net_r": float(rejected_train["net_r"].sum())
        if not rejected_train.empty
        else 0.0,
        "test_admitted_count": int(len(admitted_test)),
        "test_admitted_total_net_r": test_total,
        "test_admitted_mean_net_r": float(admitted_test["net_r"].mean())
        if not admitted_test.empty
        else math.nan,
        "test_admitted_win_rate": float((admitted_test["net_r"] > 0.0).mean())
        if not admitted_test.empty
        else math.nan,
        "test_not_admitted_count": int(len(rejected_test)),
        "test_not_admitted_total_net_r": float(rejected_test["net_r"].sum())
        if not rejected_test.empty
        else 0.0,
        "train_excess_vs_random_median_r": train_excess,
        "test_excess_vs_random_median_r": test_excess,
        **train_random,
        **test_random,
        "train_admitted_symbol_count": int(train_concentration["symbol_count"]),
        "train_admitted_session_count": int(train_concentration["session_count"]),
        "train_admitted_month_count": int(train_concentration["month_count"]),
        "train_admitted_single_symbol_share": train_concentration["single_symbol_share"],
        "train_admitted_single_session_share": train_concentration["single_session_share"],
        "train_admitted_single_month_share": train_concentration["single_month_share"],
        "test_admitted_symbol_count": int(test_concentration["symbol_count"]),
        "test_admitted_session_count": int(test_concentration["session_count"]),
        "test_admitted_month_count": int(test_concentration["month_count"]),
        "test_admitted_single_symbol_share": test_concentration["single_symbol_share"],
        "test_admitted_single_session_share": test_concentration["single_session_share"],
        "test_admitted_single_month_share": test_concentration["single_month_share"],
        "train_concentration_warning": train_concentration_warning,
        "test_concentration_warning": test_concentration_warning,
        "strict_train_supported": train_supported,
        "strict_oos_supported": oos_supported,
        "strict_status": _strict_status(train_supported, oos_supported),
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
            "personality",
            "test_admitted_total_net_r",
            "test_excess_vs_random_median_r",
            "train_admitted_total_net_r",
            "train_excess_vs_random_median_r",
            "train_admitted_count",
        ],
        ascending=[True, False, True, False, False, False, False, False],
        kind="mergesort",
    )
    return data.drop(columns=["_status_rank"]).reset_index(drop=True)


def _mark_train_selected(
    results: pd.DataFrame,
    *,
    config: PersonalityContextAdmissionConfig,
) -> pd.DataFrame:
    if results.empty:
        return results
    data = results.copy()
    data["train_selected"] = False
    train_supported = data[data["strict_train_supported"].astype(bool)].copy()
    if train_supported.empty:
        return _sort_results(data)
    for _personality, group in train_supported.groupby("personality", dropna=False):
        ranked = group.sort_values(
            [
                "train_excess_vs_random_median_r",
                "train_admitted_total_net_r",
                "train_admitted_count",
                "test_excess_vs_random_median_r",
            ],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        data.loc[
            ranked.head(config.max_selected_rules_per_personality).index,
            "train_selected",
        ] = True
    return _sort_results(data)


def _decision(selected: pd.DataFrame, all_results: pd.DataFrame) -> tuple[str, list[str]]:
    if selected.empty:
        oos_only_count = (
            int(all_results["strict_status"].eq(OOS_ONLY_NOT_TRAIN_SUPPORTED).sum())
            if not all_results.empty
            else 0
        )
        if oos_only_count:
            return (
                "continue_research_oos_personality_context_admission_warning_not_train_validated",
                ["admission rules worked only OOS and were not train-selected"],
            )
        return (
            "continue_research_no_personality_context_admission_supported",
            ["no train-selected personality context admissions"],
        )
    strict_count = int(selected["strict_status"].eq(STRICT_TRAIN_AND_OOS_SUPPORTED).sum())
    train_only_count = int(selected["strict_status"].eq(TRAIN_ONLY_NOT_OOS_SUPPORTED).sum())
    if strict_count:
        return "continue_research_strict_personality_context_admission_supported", []
    if train_only_count:
        return (
            "continue_research_train_personality_context_admission_warning_not_oos_validated",
            ["train-selected personality context admissions did not hold OOS"],
        )
    return (
        "continue_research_no_personality_context_admission_supported",
        ["no supported train-selected personality context admissions"],
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug[:96] or "rule"


def _selected_candidates(
    candidates: list[_AdmissionCandidate],
    selected: pd.DataFrame,
    *,
    max_rules: int,
) -> list[_AdmissionCandidate]:
    if selected.empty:
        return []
    selected_keys = {
        (
            str(row["personality"]),
            str(row["context_feature"]),
            str(row["context_value"]),
        )
        for _, row in selected.head(max_rules).iterrows()
    }
    return [
        candidate
        for candidate in candidates
        if (candidate.personality, candidate.context_feature, candidate.context_value)
        in selected_keys
    ]


def _build_trade_flags(
    rows: pd.DataFrame,
    selected_candidates: list[_AdmissionCandidate],
) -> pd.DataFrame:
    base_columns = [
        column
        for column in [
            "symbol",
            "timestamp",
            "session_date",
            "month",
            "personality",
            "event_state",
            "net_r",
            "blocker_rule_name",
            "blocker_family",
            "target_personality_in_scope",
        ]
        if column in rows.columns
    ]
    flags = rows[base_columns].copy()
    flags["admitted_by_selected_rule"] = False
    flags["admission_rule_name"] = ""
    flags["admission_context_feature"] = ""
    flags["admission_context_value"] = ""
    flag_columns: dict[str, NDArray[np.bool_]] = {}
    used_names: set[str] = set()
    for candidate in selected_candidates:
        mask = candidate.mask.reindex(rows.index, fill_value=False).fillna(False).astype(bool)
        unmatched = ~flags["admitted_by_selected_rule"].astype(bool)
        first_match = mask & unmatched
        flags.loc[first_match, "admitted_by_selected_rule"] = True
        flags.loc[first_match, "admission_rule_name"] = candidate.rule_name
        flags.loc[first_match, "admission_context_feature"] = candidate.context_feature
        flags.loc[first_match, "admission_context_value"] = candidate.context_value
        column = f"flag_{_slug(candidate.rule_name)}"
        suffix = 2
        while column in used_names:
            column = f"flag_{_slug(candidate.rule_name)}_{suffix}"
            suffix += 1
        used_names.add(column)
        flag_columns[column] = mask.to_numpy()
    if flag_columns:
        flags = pd.concat([flags, pd.DataFrame(flag_columns, index=rows.index)], axis=1)
    return flags


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
    selected: pd.DataFrame,
) -> None:
    columns = [
        column
        for column in [
            "rule_name",
            "strict_status",
            "train_admitted_count",
            "train_admitted_total_net_r",
            "train_excess_vs_random_median_r",
            "test_admitted_count",
            "test_admitted_total_net_r",
            "test_excess_vs_random_median_r",
            "test_concentration_warning",
        ]
        if column in selected.columns
    ]
    lines = [
        "# Personality Context Admission V0",
        "",
        (
            "Research-only validation of personality-specific blocked-candidate "
            "admission rules. Rules are selected using train months only and then "
            "evaluated OOS against same-count random baselines. No broker, live, "
            "paper, vendor fetch, order placement, or edge claim."
        ),
        "",
        f"Decision: `{payload['decision']}`",
        f"Baseline report: `{payload['input_baseline_report_dir']}`",
        f"Candidate report: `{payload['input_candidate_report_dir']}`",
        f"Train months: `{', '.join(payload['train_months'])}`",
        f"Test months: `{', '.join(payload['test_months'])}`",
        f"Blocked candidate trades: `{payload['blocked_candidate_count']}`",
        f"In-scope blocked candidate trades: `{payload['in_scope_blocked_candidate_count']}`",
        f"Candidate rule count: `{payload['candidate_rule_count']}`",
        f"Train-selected admission count: `{payload['train_selected_admission_count']}`",
        "",
        "## Train-Selected Admissions",
        "",
        _markdown_table(selected[columns] if columns else selected),
        "",
        "## Safety",
        "",
        "- research_only: true",
        "- live_ordering_enabled: false",
        "- order_placement: disabled",
        "- edge_claimed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_personality_context_admission_lab(
    *,
    input_baseline_report_dir: Path,
    input_candidate_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityContextAdmissionConfig = PersonalityContextAdmissionConfig(),
) -> PersonalityContextAdmissionResult:
    """Run a research-only personality-specific context admission scan."""

    baseline = _load_report_trades(input_baseline_report_dir)
    candidate = _load_report_trades(input_candidate_report_dir)
    blocked_all = _blocked_candidate_trades(baseline, candidate, config=config)
    in_scope = blocked_all[blocked_all["target_personality_in_scope"].astype(bool)].copy()
    train_mask, test_mask = _split_masks(in_scope, config=config)
    candidates = _build_candidates(in_scope, config=config, train_mask=train_mask)
    all_results = pd.DataFrame(
        [
            _evaluate_candidate(
                in_scope,
                candidate_rule,
                config=config,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            for candidate_rule in candidates
        ]
    )
    all_results = _mark_train_selected(all_results, config=config)
    selected = (
        all_results[all_results["train_selected"].astype(bool)].copy()
        if not all_results.empty
        else pd.DataFrame()
    )
    selected_candidates = _selected_candidates(
        candidates,
        selected,
        max_rules=config.max_flag_rules,
    )
    trade_flags = _build_trade_flags(blocked_all, selected_candidates)
    decision, decision_reasons = _decision(selected, all_results)

    run_id = "personality_context_admission_v0_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "results": run_dir / "admission_rule_results.csv",
        "selected": run_dir / "selected_personality_context_admissions.csv",
        "blocked": run_dir / "blocked_candidate_trades.csv",
        "flags": run_dir / "trade_admission_flags.csv",
    }
    for path, frame in [
        (paths["results"], all_results),
        (paths["selected"], selected),
        (paths["blocked"], blocked_all),
        (paths["flags"], trade_flags),
    ]:
        _write_csv(path, frame)

    status_counts = (
        all_results["strict_status"].value_counts().to_dict() if not all_results.empty else {}
    )
    selected_status_counts = (
        selected["strict_status"].value_counts().to_dict() if not selected.empty else {}
    )
    payload: dict[str, Any] = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "pipeline": (
            "personality -> mixed_regime -> filter -> "
            "personality_context_admission_report_only -> exit"
        ),
        "data_source": "existing local staged report trades.csv outputs",
        "volume_label": (
            "historical_volume from existing local 5m OHLCV-derived staged reports "
            "when volume-derived context fields are used"
        ),
        "input_baseline_report_dir": str(input_baseline_report_dir),
        "input_candidate_report_dir": str(input_candidate_report_dir),
        "output_dir": str(run_dir),
        "train_months": list(config.train_months),
        "test_months": list(config.test_months),
        "target_personalities": list(config.target_personalities),
        "context_features": _infer_context_features(in_scope, config=config),
        "trade_key_columns": _available_trade_key_columns(baseline, candidate, config=config),
        "random_iterations": int(config.random_iterations),
        "baseline_trade_count": int(len(baseline)),
        "candidate_trade_count": int(len(candidate)),
        "blocked_candidate_count": int(len(blocked_all)),
        "in_scope_blocked_candidate_count": int(len(in_scope)),
        "base_all_blocked": _summary_stats(in_scope),
        "base_train_blocked": _summary_stats(in_scope[train_mask]),
        "base_test_blocked": _summary_stats(in_scope[test_mask]),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "candidate_rule_count": int(len(all_results)),
        "train_selected_admission_count": int(len(selected)),
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
    _write_summary_md(paths["summary_md"], payload=payload, selected=selected)

    return PersonalityContextAdmissionResult(
        run_id=run_id,
        input_baseline_report_dir=input_baseline_report_dir,
        input_candidate_report_dir=input_candidate_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        admission_rule_results_csv_path=paths["results"],
        selected_admissions_csv_path=paths["selected"],
        blocked_candidate_trades_csv_path=paths["blocked"],
        trade_admission_flags_csv_path=paths["flags"],
        decision=decision,
        selected_admission_count=int(len(selected)),
    )


__all__ = [
    "PersonalityContextAdmissionConfig",
    "PersonalityContextAdmissionResult",
    "run_personality_context_admission_lab",
]
