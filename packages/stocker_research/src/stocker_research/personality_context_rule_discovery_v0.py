"""Research-only blank-slate personality context rule discovery.

This lab searches simple personality-specific admission rules from staged report
pairs. It is intentionally confined to existing report CSVs: no broker
execution, live trading, paper trading, order placement, or vendor fetching.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.personality_context_admission_v0 import (
    DEFAULT_TRADE_KEY_COLUMNS,
    _available_trade_key_columns,
    _key_series,
    _load_report_trades,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_context_rule_discovery_v0")

SUPPORTED_CANDIDATE_ONLY = "supported_candidate_only_reentry"
SUPPORTED_NO_PRIOR_ONLY = "supported_no_prior_outcome_candidate_only_sparse"
NOT_SUPPORTED = "not_supported"

DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "prev_event_personality",
    "volume_x_vwap_regime",
    "time_x_vwap_regime",
    "vwap_x_efficiency_regime",
    "vwap_x_range_regime",
    "compression_x_efficiency_regime",
    "opening_mid_x_range_regime",
    "relative_volume_regime",
    "time_regime",
    "vwap_side_regime",
)

DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = (
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "distance_from_recent_high_pct",
    "distance_from_vwap_pct",
    "relative_cumulative_volume",
    "relative_volume_at_bar_index",
    "same_direction_other_symbol_count_15m",
    "same_personality_other_symbol_count_15m",
    "same_direction_other_symbol_count_30m",
    "same_personality_other_symbol_count_30m",
    "close_location_value",
    "body_pct_of_range",
    "lower_wick_pct_of_range",
    "upper_wick_pct_of_range",
    "directional_efficiency_3",
    "directional_efficiency_6",
    "compression_zscore",
    "range_zscore",
    "return_zscore",
)


@dataclass(frozen=True)
class ReportPair:
    """One gated baseline and permissive candidate staged-report pair."""

    label: str
    baseline_report_dir: Path
    candidate_report_dir: Path


@dataclass(frozen=True)
class PersonalityContextRuleDiscoveryConfig:
    """Configuration for blank-slate context admission rule discovery."""

    target_personalities: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    trade_key_columns: tuple[str, ...] = DEFAULT_TRADE_KEY_COLUMNS
    quantiles: tuple[float, ...] = (0.20, 0.33, 0.50, 0.67, 0.80)
    max_category_values: int = 30
    max_numeric_thresholds_per_feature: int = 14
    min_rule_trades: int = 3
    min_rule_windows: int = 2
    min_positive_windows: int = 2
    max_negative_windows: int = 0
    min_total_net_r: float = 0.0
    min_excess_vs_random_median_r: float = 0.0
    max_single_window_share: float = 0.65
    max_atomic_rules_per_personality: int = 800
    max_union_base_rules_per_personality: int = 24
    max_union_rules_per_personality: int = 250
    random_iterations: int = 3000
    random_seed: int = 1337


@dataclass(frozen=True)
class PersonalityContextRuleDiscoveryResult:
    """Paths and headline result for one discovery run."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    rule_results_csv_path: Path
    rule_window_results_csv_path: Path
    selected_rules_csv_path: Path
    candidate_only_trades_csv_path: Path
    decision: str
    selected_rule_count: int


@dataclass(frozen=True)
class _RuleTerm:
    feature: str
    operator: str
    value: str | float


@dataclass(frozen=True)
class _Rule:
    rule_name: str
    personality: str
    alternatives: tuple[tuple[_RuleTerm, ...], ...]
    rule_kind: str


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug[:96] or "rule"


def _candidate_only_trades(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> pd.DataFrame:
    key_columns = _available_trade_key_columns(baseline, candidate, config=config)  # type: ignore[arg-type]
    baseline_keys = set(_key_series(baseline, key_columns))
    candidate_keys = _key_series(candidate, key_columns)
    rows = candidate[~candidate_keys.isin(baseline_keys)].copy()
    rows["candidate_trade_key"] = candidate_keys[~candidate_keys.isin(baseline_keys)].to_numpy()
    return rows.reset_index(drop=True)


def _load_pair_rows(
    report_pairs: tuple[ReportPair, ...],
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    no_prior_frames: list[pd.DataFrame] = []
    candidate_only_frames: list[pd.DataFrame] = []
    portfolio_rows: list[dict[str, Any]] = []
    for pair in report_pairs:
        baseline = _load_report_trades(pair.baseline_report_dir)
        candidate = _load_report_trades(pair.candidate_report_dir)
        candidate_only = _candidate_only_trades(baseline, candidate, config=config)
        for frame in (baseline, candidate, candidate_only):
            frame["window_label"] = pair.label
        no_prior_frames.append(candidate)
        candidate_only_frames.append(candidate_only)
        portfolio_rows.append(
            {
                "window_label": pair.label,
                "baseline_trade_count": int(len(baseline)),
                "baseline_total_net_r": float(baseline["net_r"].sum())
                if not baseline.empty
                else 0.0,
                "candidate_trade_count": int(len(candidate)),
                "candidate_total_net_r": float(candidate["net_r"].sum())
                if not candidate.empty
                else 0.0,
                "candidate_only_trade_count": int(len(candidate_only)),
                "candidate_only_total_net_r": float(candidate_only["net_r"].sum())
                if not candidate_only.empty
                else 0.0,
            }
        )
    no_prior = (
        pd.concat(no_prior_frames, ignore_index=True) if no_prior_frames else pd.DataFrame()
    )
    candidate_only = (
        pd.concat(candidate_only_frames, ignore_index=True)
        if candidate_only_frames
        else pd.DataFrame()
    )
    portfolios = pd.DataFrame(portfolio_rows)
    for rows in (no_prior, candidate_only):
        if rows.empty:
            continue
        for feature in config.numeric_features:
            if feature in rows:
                rows[feature] = pd.to_numeric(rows[feature], errors="coerce")
    return no_prior, candidate_only, portfolios


def _target_rows(
    rows: pd.DataFrame,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> pd.DataFrame:
    if rows.empty or "personality" not in rows:
        return rows.copy()
    if not config.target_personalities:
        return rows.copy()
    allowed = set(config.target_personalities)
    return rows[rows["personality"].astype(str).isin(allowed)].copy()


def _term_expression(term: _RuleTerm) -> str:
    value = f"{term.value:.6g}" if isinstance(term.value, float) else str(term.value)
    return f"{term.feature} {term.operator} {value}"


def _rule_expression(rule: _Rule) -> str:
    parts = []
    for terms in rule.alternatives:
        parts.append(" AND ".join(_term_expression(term) for term in terms))
    return " OR ".join(f"({part})" for part in parts)


def _term_mask(rows: pd.DataFrame, term: _RuleTerm) -> pd.Series:
    if term.feature not in rows:
        return pd.Series(False, index=rows.index)
    if term.operator == "==":
        return rows[term.feature].astype(str).eq(str(term.value))
    if term.operator == "!=":
        return ~rows[term.feature].astype(str).eq(str(term.value))
    numeric = pd.to_numeric(rows[term.feature], errors="coerce")
    threshold = float(term.value)
    if term.operator == ">=":
        return numeric >= threshold
    if term.operator == "<=":
        return numeric <= threshold
    raise ValueError(f"Unsupported operator: {term.operator}")


def _evaluate_rule_mask(rows: pd.DataFrame, rule: _Rule) -> pd.Series:
    if rows.empty:
        return pd.Series(False, index=rows.index)
    personality_mask = rows["personality"].astype(str).eq(rule.personality)
    rule_mask = pd.Series(False, index=rows.index)
    for terms in rule.alternatives:
        clause_mask = pd.Series(True, index=rows.index)
        for term in terms:
            clause_mask &= _term_mask(rows, term)
        rule_mask |= clause_mask
    return personality_mask & rule_mask.fillna(False)


def _compatible_terms(terms: tuple[_RuleTerm, ...]) -> bool:
    seen_equalities: dict[str, str] = {}
    seen_not_equalities: set[tuple[str, str]] = set()
    for term in terms:
        value = str(term.value)
        if term.operator == "==":
            if term.feature in seen_equalities and seen_equalities[term.feature] != value:
                return False
            if (term.feature, value) in seen_not_equalities:
                return False
            seen_equalities[term.feature] = value
        elif term.operator == "!=":
            if seen_equalities.get(term.feature) == value:
                return False
            seen_not_equalities.add((term.feature, value))
    return True


def _numeric_thresholds(
    rows: pd.DataFrame,
    feature: str,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> list[float]:
    series = pd.to_numeric(rows[feature], errors="coerce").dropna()
    if series.empty:
        return []
    thresholds: set[float] = {
        float(value)
        for value in series.quantile(list(config.quantiles)).to_numpy()
        if not pd.isna(value)
    }
    unique = series.drop_duplicates().sort_values()
    if len(unique) <= config.max_numeric_thresholds_per_feature:
        thresholds.update(float(value) for value in unique.to_numpy())
    if "return" in feature or "distance" in feature or feature.endswith("zscore"):
        thresholds.add(0.0)
    if feature.startswith("same_"):
        thresholds.update({1.0, 2.0, 3.0})
    if feature == "relative_cumulative_volume":
        thresholds.update({0.5, 0.809534, 1.0})
    if feature == "close_location_value":
        thresholds.update({0.5, 0.614245, 0.75})
    return sorted(thresholds)[: config.max_numeric_thresholds_per_feature]


def _candidate_terms(
    rows: pd.DataFrame,
    personality: str,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> tuple[list[_RuleTerm], list[_RuleTerm]]:
    personality_rows = rows[rows["personality"].astype(str).eq(personality)]
    categorical_terms: list[_RuleTerm] = []
    numeric_terms: list[_RuleTerm] = []
    for feature in config.categorical_features:
        if feature not in personality_rows:
            continue
        values = personality_rows[feature].dropna().astype(str).value_counts()
        if values.empty or len(values) > config.max_category_values:
            continue
        for value, count in values.items():
            if int(count) < max(1, min(config.min_rule_trades, 3)):
                continue
            categorical_terms.append(_RuleTerm(feature, "==", str(value)))
            categorical_terms.append(_RuleTerm(feature, "!=", str(value)))
    for feature in config.numeric_features:
        if feature not in personality_rows:
            continue
        for threshold in _numeric_thresholds(personality_rows, feature, config=config):
            numeric_terms.append(_RuleTerm(feature, ">=", threshold))
            numeric_terms.append(_RuleTerm(feature, "<=", threshold))
    return categorical_terms, numeric_terms


def _rule_from_terms(personality: str, terms: tuple[_RuleTerm, ...], *, kind: str) -> _Rule:
    expression = " AND ".join(_term_expression(term) for term in terms)
    rule_name = f"personality_context_rule_discovery: {personality} IF {expression}"
    return _Rule(
        rule_name=rule_name,
        personality=personality,
        alternatives=(terms,),
        rule_kind=kind,
    )


def _generate_atomic_rules(
    rows: pd.DataFrame,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> list[_Rule]:
    personalities = (
        list(config.target_personalities)
        if config.target_personalities
        else sorted(rows["personality"].astype(str).unique().tolist())
    )
    rules: list[_Rule] = []
    for personality in personalities:
        categorical_terms, numeric_terms = _candidate_terms(rows, personality, config=config)
        personality_rules: list[_Rule] = []
        for term in [*categorical_terms, *numeric_terms]:
            personality_rules.append(_rule_from_terms(personality, (term,), kind="single"))
        for cat, num in itertools.product(categorical_terms, numeric_terms):
            terms = (cat, num)
            if _compatible_terms(terms):
                personality_rules.append(_rule_from_terms(personality, terms, kind="and2"))
        for left, right in itertools.combinations(categorical_terms, 2):
            terms = (left, right)
            if _compatible_terms(terms):
                personality_rules.append(_rule_from_terms(personality, terms, kind="and2"))
        for cat_pair in itertools.combinations(categorical_terms, 2):
            if not _compatible_terms(cat_pair):
                continue
            for num in numeric_terms:
                terms = (*cat_pair, num)
                if _compatible_terms(terms):
                    personality_rules.append(_rule_from_terms(personality, terms, kind="and3"))
        dedup: dict[str, _Rule] = {}
        for rule in personality_rules:
            mask = _evaluate_rule_mask(rows, rule)
            if int(mask.sum()) < config.min_rule_trades:
                continue
            dedup.setdefault(_rule_expression(rule), rule)
            if len(dedup) >= config.max_atomic_rules_per_personality:
                break
        rules.extend(dedup.values())
    return rules


def _random_same_count_median(
    rows: pd.DataFrame,
    count: int,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
    seed_key: str,
) -> float:
    if rows.empty:
        return math.nan
    net_r = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0).to_numpy()
    if count <= 0:
        return 0.0
    if count >= len(net_r):
        return float(net_r.sum())
    digest = hashlib.sha256(seed_key.encode()).digest()
    seed_offset = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(config.random_seed + seed_offset)
    all_indices = np.arange(len(net_r))
    totals = np.empty(config.random_iterations)
    for index in range(config.random_iterations):
        chosen = rng.choice(all_indices, size=count, replace=False)
        totals[index] = float(net_r[chosen].sum())
    return float(np.median(totals))


def _evaluate_rule_sample(
    rows: pd.DataFrame,
    rule: _Rule,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
    sample_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if rows.empty:
        return _empty_rule_metrics(sample_name), []
    personality_rows = rows[rows["personality"].astype(str).eq(rule.personality)].copy()
    mask = _evaluate_rule_mask(personality_rows, rule)
    window_rows: list[dict[str, Any]] = []
    for window, window_base in personality_rows.groupby("window_label", dropna=False):
        window_mask = mask.reindex(window_base.index, fill_value=False)
        admitted = window_base[window_mask]
        total = float(admitted["net_r"].sum()) if not admitted.empty else 0.0
        random_median = _random_same_count_median(
            window_base,
            int(len(admitted)),
            config=config,
            seed_key=f"{sample_name}|{rule.rule_name}|{window}",
        )
        window_rows.append(
            {
                "sample": sample_name,
                "window_label": str(window),
                "rule_name": rule.rule_name,
                "rule_expression": _rule_expression(rule),
                "personality": rule.personality,
                "base_count": int(len(window_base)),
                "base_total_net_r": float(window_base["net_r"].sum())
                if not window_base.empty
                else 0.0,
                "admitted_count": int(len(admitted)),
                "admitted_total_net_r": total,
                "admitted_win_rate": float((admitted["net_r"] > 0.0).mean())
                if not admitted.empty
                else math.nan,
                "same_count_random_median_r": random_median,
                "excess_vs_random_median_r": total - random_median,
            }
        )
    window_frame = pd.DataFrame(window_rows)
    active = window_frame[window_frame["admitted_count"] > 0]
    if active.empty:
        return _empty_rule_metrics(sample_name), window_rows
    total_r = float(active["admitted_total_net_r"].sum())
    max_share = (
        float(active["admitted_total_net_r"].max() / total_r) if total_r > 0.0 else math.nan
    )
    metrics = {
        f"{sample_name}_active_windows": int(len(active)),
        f"{sample_name}_positive_windows": int(
            (active["admitted_total_net_r"] > 0.0).sum()
        ),
        f"{sample_name}_negative_windows": int(
            (active["admitted_total_net_r"] < 0.0).sum()
        ),
        f"{sample_name}_admitted_count": int(active["admitted_count"].sum()),
        f"{sample_name}_admitted_total_net_r": total_r,
        f"{sample_name}_admitted_win_count": int(
            (personality_rows[mask]["net_r"] > 0.0).sum()
        ),
        f"{sample_name}_admitted_loss_count": int(
            (personality_rows[mask]["net_r"] <= 0.0).sum()
        ),
        f"{sample_name}_same_count_random_median_total_r": float(
            active["same_count_random_median_r"].sum()
        ),
        f"{sample_name}_excess_vs_random_median_r": float(
            active["excess_vs_random_median_r"].sum()
        ),
        f"{sample_name}_worst_window_net_r": float(
            active["admitted_total_net_r"].min()
        ),
        f"{sample_name}_best_window_net_r": float(active["admitted_total_net_r"].max()),
        f"{sample_name}_max_single_window_share": max_share,
    }
    return metrics, window_rows


def _empty_rule_metrics(sample_name: str) -> dict[str, Any]:
    return {
        f"{sample_name}_active_windows": 0,
        f"{sample_name}_positive_windows": 0,
        f"{sample_name}_negative_windows": 0,
        f"{sample_name}_admitted_count": 0,
        f"{sample_name}_admitted_total_net_r": 0.0,
        f"{sample_name}_admitted_win_count": 0,
        f"{sample_name}_admitted_loss_count": 0,
        f"{sample_name}_same_count_random_median_total_r": 0.0,
        f"{sample_name}_excess_vs_random_median_r": 0.0,
        f"{sample_name}_worst_window_net_r": math.nan,
        f"{sample_name}_best_window_net_r": math.nan,
        f"{sample_name}_max_single_window_share": math.nan,
    }


def _sample_supported(
    metrics: dict[str, Any],
    sample_name: str,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> bool:
    share = float(metrics[f"{sample_name}_max_single_window_share"])
    concentration_ok = math.isnan(share) or share <= config.max_single_window_share
    return (
        int(metrics[f"{sample_name}_admitted_count"]) >= config.min_rule_trades
        and int(metrics[f"{sample_name}_active_windows"]) >= config.min_rule_windows
        and int(metrics[f"{sample_name}_positive_windows"]) >= config.min_positive_windows
        and int(metrics[f"{sample_name}_negative_windows"]) <= config.max_negative_windows
        and float(metrics[f"{sample_name}_admitted_total_net_r"]) > config.min_total_net_r
        and float(metrics[f"{sample_name}_excess_vs_random_median_r"])
        > config.min_excess_vs_random_median_r
        and concentration_ok
    )


def _evaluate_rules(
    rules: list[_Rule],
    *,
    no_prior_rows: pd.DataFrame,
    candidate_only_rows: pd.DataFrame,
    config: PersonalityContextRuleDiscoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for rule in rules:
        no_prior_metrics, no_prior_windows = _evaluate_rule_sample(
            no_prior_rows,
            rule,
            config=config,
            sample_name="no_prior",
        )
        candidate_only_metrics, candidate_only_windows = _evaluate_rule_sample(
            candidate_only_rows,
            rule,
            config=config,
            sample_name="candidate_only",
        )
        candidate_only_supported = _sample_supported(
            candidate_only_metrics,
            "candidate_only",
            config=config,
        )
        no_prior_supported = _sample_supported(no_prior_metrics, "no_prior", config=config)
        status = NOT_SUPPORTED
        if candidate_only_supported:
            status = SUPPORTED_CANDIDATE_ONLY
        elif no_prior_supported:
            status = SUPPORTED_NO_PRIOR_ONLY
        result_rows.append(
            {
                "rule_name": rule.rule_name,
                "rule_expression": _rule_expression(rule),
                "rule_kind": rule.rule_kind,
                "personality": rule.personality,
                "support_status": status,
                "candidate_only_supported": candidate_only_supported,
                "no_prior_supported": no_prior_supported,
                **no_prior_metrics,
                **candidate_only_metrics,
            }
        )
        window_rows.extend(no_prior_windows)
        window_rows.extend(candidate_only_windows)
    return pd.DataFrame(result_rows), pd.DataFrame(window_rows)


def _sort_rule_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    status_rank = {
        SUPPORTED_CANDIDATE_ONLY: 0,
        SUPPORTED_NO_PRIOR_ONLY: 1,
        NOT_SUPPORTED: 2,
    }
    data = results.copy()
    data["_rank"] = data["support_status"].map(status_rank).fillna(9).astype(int)
    data = data.sort_values(
        [
            "_rank",
            "candidate_only_admitted_total_net_r",
            "candidate_only_excess_vs_random_median_r",
            "no_prior_admitted_total_net_r",
            "no_prior_excess_vs_random_median_r",
            "no_prior_active_windows",
        ],
        ascending=[True, False, False, False, False, False],
        kind="mergesort",
    )
    return data.drop(columns=["_rank"]).reset_index(drop=True)


def _union_rule(left: _Rule, right: _Rule) -> _Rule | None:
    if left.personality != right.personality:
        return None
    alternatives = (*left.alternatives, *right.alternatives)
    expression = " OR ".join(
        " AND ".join(_term_expression(term) for term in terms)
        for terms in alternatives
    )
    return _Rule(
        rule_name=f"personality_context_rule_discovery: {left.personality} IF {expression}",
        personality=left.personality,
        alternatives=alternatives,
        rule_kind="or2",
    )


def _generate_union_rules(
    atomic_rules: list[_Rule],
    atomic_results: pd.DataFrame,
    *,
    config: PersonalityContextRuleDiscoveryConfig,
) -> list[_Rule]:
    if atomic_results.empty:
        return []
    rule_by_name = {rule.rule_name: rule for rule in atomic_rules}
    unions: list[_Rule] = []
    for personality, group in atomic_results.groupby("personality", dropna=False):
        ranked = group.sort_values(
            [
                "no_prior_positive_windows",
                "no_prior_admitted_total_net_r",
                "no_prior_excess_vs_random_median_r",
            ],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(config.max_union_base_rules_per_personality)
        base_rules = [
            rule_by_name[name]
            for name in ranked["rule_name"].astype(str).tolist()
            if name in rule_by_name
        ]
        seen: set[str] = set()
        for left, right in itertools.combinations(base_rules, 2):
            union = _union_rule(left, right)
            if union is None:
                continue
            key = _rule_expression(union)
            if key in seen:
                continue
            seen.add(key)
            unions.append(union)
            if len([rule for rule in unions if rule.personality == str(personality)]) >= (
                config.max_union_rules_per_personality
            ):
                break
    return unions


def _decision(selected: pd.DataFrame) -> tuple[str, list[str]]:
    if selected.empty:
        return (
            "continue_research_no_blank_slate_context_rule_supported",
            ["no discovered context admission rules passed support checks"],
        )
    candidate_supported = selected["support_status"].eq(SUPPORTED_CANDIDATE_ONLY).sum()
    if int(candidate_supported):
        return "continue_research_blank_slate_context_rule_supported", []
    return (
        "continue_research_blank_slate_context_rule_warning_candidate_only_sparse",
        ["rules held on no-prior outcomes but not enough true candidate-only re-entry"],
    )


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows)
    lines = [
        "| " + " | ".join(str(column) for column in shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for _, row in shown.iterrows():
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_md(path: Path, payload: dict[str, Any], selected: pd.DataFrame) -> None:
    columns = [
        column
        for column in [
            "rule_expression",
            "support_status",
            "rule_kind",
            "personality",
            "candidate_only_admitted_count",
            "candidate_only_admitted_total_net_r",
            "candidate_only_positive_windows",
            "no_prior_admitted_count",
            "no_prior_admitted_total_net_r",
            "no_prior_positive_windows",
            "no_prior_max_single_window_share",
        ]
        if column in selected.columns
    ]
    lines = [
        "# Personality Context Rule Discovery V0",
        "",
        (
            "Research-only blank-slate discovery of personality-specific context "
            "admission rules from staged report pairs. Generated rules are "
            "evaluated across windows against same-count random baselines. No "
            "broker, live, paper, vendor fetch, order placement, or edge claim."
        ),
        "",
        f"Decision: `{payload['decision']}`",
        f"Report pair count: `{payload['report_pair_count']}`",
        f"Target personalities: `{', '.join(payload['target_personalities'])}`",
        f"Atomic rule count: `{payload['atomic_rule_count']}`",
        f"Union rule count: `{payload['union_rule_count']}`",
        f"Selected rule count: `{payload['selected_rule_count']}`",
        "",
        "## Selected Rules",
        "",
        _markdown_table(selected[columns] if columns else selected),
        "",
        "## Safety",
        "",
        "- `research_only: true`",
        "- `live_ordering_enabled: false`",
        "- `order_placement: disabled`",
        "- `edge_claimed: false`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_personality_context_rule_discovery_lab(
    *,
    report_pairs: tuple[ReportPair, ...],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityContextRuleDiscoveryConfig | None = None,
) -> PersonalityContextRuleDiscoveryResult:
    """Run blank-slate personality context rule discovery over paired reports."""

    if not report_pairs:
        raise ValueError("Supply at least one report pair.")
    config = config or PersonalityContextRuleDiscoveryConfig()
    run_id = "personality_context_rule_discovery_v0_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    no_prior, candidate_only, portfolios = _load_pair_rows(report_pairs, config=config)
    target_no_prior = _target_rows(no_prior, config=config)
    target_candidate_only = _target_rows(candidate_only, config=config)
    atomic_rules = _generate_atomic_rules(target_no_prior, config=config)
    atomic_results, atomic_window_results = _evaluate_rules(
        atomic_rules,
        no_prior_rows=target_no_prior,
        candidate_only_rows=target_candidate_only,
        config=config,
    )
    union_rules = _generate_union_rules(atomic_rules, atomic_results, config=config)
    union_results, union_window_results = _evaluate_rules(
        union_rules,
        no_prior_rows=target_no_prior,
        candidate_only_rows=target_candidate_only,
        config=config,
    )
    all_results = _sort_rule_results(pd.concat([atomic_results, union_results]))
    window_results = pd.concat([atomic_window_results, union_window_results]).reset_index(
        drop=True
    )
    selected = all_results[all_results["support_status"].ne(NOT_SUPPORTED)].copy()
    selected = _sort_rule_results(selected)
    decision, reasons = _decision(selected)

    summary_payload = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "decision": decision,
        "decision_reasons": reasons,
        "report_pair_count": int(len(report_pairs)),
        "report_pairs": [
            {
                "label": pair.label,
                "baseline_report_dir": str(pair.baseline_report_dir),
                "candidate_report_dir": str(pair.candidate_report_dir),
            }
            for pair in report_pairs
        ],
        "target_personalities": list(config.target_personalities),
        "categorical_features": list(config.categorical_features),
        "numeric_features": list(config.numeric_features),
        "atomic_rule_count": int(len(atomic_rules)),
        "union_rule_count": int(len(union_rules)),
        "selected_rule_count": int(len(selected)),
        "no_prior_target_trade_count": int(len(target_no_prior)),
        "candidate_only_target_trade_count": int(len(target_candidate_only)),
        "portfolio_window_summary": portfolios.to_dict(orient="records"),
        "config": config.__dict__,
    }

    summary_json_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    decision_json_path = run_dir / "decision.json"
    rule_results_csv_path = run_dir / "rule_results.csv"
    rule_window_results_csv_path = run_dir / "rule_window_results.csv"
    selected_rules_csv_path = run_dir / "selected_rules.csv"
    candidate_only_trades_csv_path = run_dir / "candidate_only_trades.csv"

    _write_json(summary_json_path, summary_payload)
    _write_json(
        decision_json_path,
        {
            "decision": decision,
            "decision_reasons": reasons,
            "selected_rule_count": int(len(selected)),
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
        },
    )
    _write_csv(rule_results_csv_path, all_results)
    _write_csv(rule_window_results_csv_path, window_results)
    _write_csv(selected_rules_csv_path, selected)
    _write_csv(candidate_only_trades_csv_path, target_candidate_only)
    _write_summary_md(summary_markdown_path, summary_payload, selected)

    return PersonalityContextRuleDiscoveryResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        decision_json_path=decision_json_path,
        rule_results_csv_path=rule_results_csv_path,
        rule_window_results_csv_path=rule_window_results_csv_path,
        selected_rules_csv_path=selected_rules_csv_path,
        candidate_only_trades_csv_path=candidate_only_trades_csv_path,
        decision=decision,
        selected_rule_count=int(len(selected)),
    )
