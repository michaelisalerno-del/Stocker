"""Research-only shadow candidate trigger audit.

This audit treats every row in ``trade_context_features.csv`` as an eligible
shadow candidate. That keeps the opportunity stream separate from executed
trades, so safety stops or caveats do not bias the rolling deterioration signal.

It never touches broker execution, paper trading, live trading, order placement,
or vendor fetching.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/reports/research/shadow_candidate_trigger_audit_v0")


@dataclass(frozen=True)
class ShadowCandidateTriggerConfig:
    """Configuration for the shadow candidate deterioration trigger audit."""

    shadow_window: int = 20
    min_prior_candidates: int = 8
    weak_context_max_score: int = 3
    weak_context_share_threshold: float = 0.75
    shadow_net_r_threshold: float = -1.0
    anti_stale_windows: tuple[int, ...] = (6, 12, 24, 36)
    anti_stale_feature_bases: tuple[str, ...] = ("time_regime", "time_x_vwap_regime")
    anti_stale_quantiles: tuple[float, ...] = (0.10, 0.20, 0.33, 0.50, 0.67)
    min_train_count: int = 30
    min_rule_keep_count: int = 8
    random_iterations: int = 1000
    random_seed: int = 1337


@dataclass(frozen=True)
class ShadowCandidateTriggerResult:
    """Paths and headline result for one shadow candidate trigger audit."""

    run_id: str
    input_context_report_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    shadow_candidate_features_csv_path: Path
    monthly_policy_results_csv_path: Path
    policy_summary_csv_path: Path
    trade_shadow_trigger_flags_csv_path: Path
    decision: str


@dataclass(frozen=True)
class _AntiStaleRule:
    feature: str
    threshold: float
    train_score: float
    train_keep_count: int
    train_keep_net_r: float
    train_keep_mean_r: float
    train_lift_mean_r: float

    @property
    def name(self) -> str:
        return f"all: {self.feature} <= {self.threshold:.6g}"


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
    rows = pd.read_csv(path)
    required = {"symbol", "timestamp", "session_date", "net_r"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"trade_context_features.csv missing required columns: {missing}")
    rows = rows.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows[rows["timestamp"].notna()].copy()
    rows["session_date"] = pd.to_datetime(rows["session_date"]).dt.strftime("%Y-%m-%d")
    if "month" not in rows:
        rows["month"] = rows["timestamp"].dt.strftime("%Y-%m")
    else:
        rows["month"] = rows["month"].astype(str)
    if "split" not in rows:
        rows["split"] = "unknown"
    else:
        rows["split"] = rows["split"].astype(str)
    rows["symbol"] = rows["symbol"].astype(str)
    rows["net_r"] = pd.to_numeric(rows["net_r"], errors="coerce")
    rows = rows[rows["net_r"].notna()].copy()
    return rows.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)


def _col(rows: pd.DataFrame, name: str) -> pd.Series:
    if name not in rows:
        return pd.Series("", index=rows.index)
    return rows[name].astype(str)


def _num_col(rows: pd.DataFrame, name: str) -> pd.Series:
    if name not in rows:
        return pd.Series(np.nan, index=rows.index)
    return pd.to_numeric(rows[name], errors="coerce")


def _bool_object(values: pd.Series) -> pd.Series:
    return pd.Series(
        [bool(value) for value in values.fillna(False)],
        index=values.index,
        dtype=object,
    )


def _truthy_series(rows: pd.DataFrame, name: str, *, default: bool) -> pd.Series:
    if name not in rows:
        return pd.Series(default, index=rows.index, dtype=object)
    values = rows[name]
    if pd.api.types.is_bool_dtype(values):
        return _bool_object(values.astype(bool))
    normalized = values.astype(str).str.strip().str.lower()
    truthy = normalized.isin({"1", "true", "yes", "y"})
    falsy = normalized.isin({"0", "false", "no", "n"})
    return pd.Series(
        [
            bool(default if not is_true and not is_false else is_true)
            for is_true, is_false in zip(truthy, falsy, strict=True)
        ],
        index=rows.index,
        dtype=object,
    )


def _constructive_score(rows: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=rows.index, dtype="int64")
    score += _col(rows, "efficiency_regime").isin(["mixed_efficiency", "directional_efficiency"])
    score += _col(rows, "vwap_side_regime").eq("above")
    score += _col(rows, "opening_mid_side_regime").eq("above")
    score += _col(rows, "time_regime").eq("late_day")
    score += _col(rows, "range_regime").eq("high_range")
    score += _num_col(rows, "prev_24_vwap_x_efficiency_regime_current_share").ge(0.25)
    score += _num_col(rows, "prev_36_opening_mid_side_regime_current_share").ge(0.50)
    score += _num_col(rows, "vwap_cross_count_12").le(1.0)
    return score.astype(int)


def add_shadow_candidate_trigger_features(
    rows: pd.DataFrame,
    *,
    config: ShadowCandidateTriggerConfig = ShadowCandidateTriggerConfig(),
) -> pd.DataFrame:
    """Attach causal shadow-candidate deterioration trigger features.

    The rolling fields are shifted by one candidate, so each row uses only
    earlier eligible opportunities.
    """

    if config.shadow_window < 1:
        raise ValueError("shadow_window must be at least 1")
    if config.min_prior_candidates < 0:
        raise ValueError("min_prior_candidates must be non-negative")

    out = rows.copy()
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("rows must include timestamp")
    if "symbol" not in out:
        raise ValueError("rows must include symbol")
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    out["shadow_candidate_index"] = np.arange(len(out), dtype=int)

    if "planned_exit_shadow_net_r" in out:
        out["planned_exit_shadow_net_r"] = pd.to_numeric(
            out["planned_exit_shadow_net_r"],
            errors="coerce",
        )
    else:
        out["planned_exit_shadow_net_r"] = pd.to_numeric(out["net_r"], errors="coerce")
    out["planned_exit_shadow_net_r"] = out["planned_exit_shadow_net_r"].fillna(0.0)

    if "candidate_block_reason" in out:
        out["candidate_block_reason"] = out["candidate_block_reason"].fillna("none").astype(str)
    else:
        out["candidate_block_reason"] = "none"

    if "would_be_taken_without_safety" in out:
        out["would_be_taken_without_safety"] = _truthy_series(
            out,
            "would_be_taken_without_safety",
            default=True,
        )
    else:
        safety_reasons = {"safety_stop", "safety_blocked", "max_daily_loss", "cooldown"}
        reason = out["candidate_block_reason"].astype(str).str.strip().str.lower()
        out["would_be_taken_without_safety"] = pd.Series(
            [item not in safety_reasons for item in reason],
            index=out.index,
            dtype=object,
        )

    out["weak_context_score"] = _constructive_score(out)
    weak_flag = out["weak_context_score"].le(config.weak_context_max_score)
    out["weak_context_flag"] = _bool_object(weak_flag)

    window = config.shadow_window
    shifted_weak = weak_flag.astype(float).shift(1)
    shifted_net = out["planned_exit_shadow_net_r"].shift(1)
    count_col = f"prior_{window}_candidate_count"
    weak_share_col = f"prior_{window}_weak_context_share"
    shadow_net_col = f"prior_{window}_shadow_candidate_net_r"
    out[count_col] = np.minimum(np.arange(len(out)), window)
    out[weak_share_col] = shifted_weak.rolling(window=window, min_periods=1).mean()
    out[shadow_net_col] = shifted_net.rolling(window=window, min_periods=1).sum()
    out[weak_share_col] = out[weak_share_col].fillna(0.0)
    out[shadow_net_col] = out[shadow_net_col].fillna(0.0)

    weak_cluster = (
        out[count_col].ge(config.min_prior_candidates)
        & out[weak_share_col].ge(config.weak_context_share_threshold)
    )
    trigger = weak_cluster & out[shadow_net_col].le(config.shadow_net_r_threshold)
    out["weak_context_cluster_trigger"] = _bool_object(weak_cluster)
    out["weak_cluster_shadow_deterioration_trigger"] = _bool_object(trigger)
    return out


def _summary_stats(rows: pd.DataFrame) -> dict[str, int | float]:
    if rows.empty:
        return {"count": 0, "net_r": 0.0, "mean_r": math.nan, "win_rate": math.nan}
    net = pd.to_numeric(rows["planned_exit_shadow_net_r"], errors="coerce").fillna(0.0)
    return {
        "count": int(len(rows)),
        "net_r": float(net.sum()),
        "mean_r": float(net.mean()),
        "win_rate": float((net > 0.0).mean()),
    }


def _anti_stale_features(rows: pd.DataFrame, config: ShadowCandidateTriggerConfig) -> list[str]:
    result: list[str] = []
    for window in config.anti_stale_windows:
        for base in config.anti_stale_feature_bases:
            feature = f"prev_{window}_{base}_current_share"
            if feature not in rows:
                continue
            values = _num_col(rows, feature)
            if values.dropna().nunique() >= 2:
                result.append(feature)
    return result


def _select_anti_stale_rule(
    train: pd.DataFrame,
    *,
    config: ShadowCandidateTriggerConfig,
) -> _AntiStaleRule | None:
    if len(train) < config.min_train_count:
        return None
    base_stats = _summary_stats(train)
    rules: list[_AntiStaleRule] = []
    for feature in _anti_stale_features(train, config):
        values = _num_col(train, feature).replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 2:
            continue
        thresholds = sorted(
            {
                float(values.quantile(quantile))
                for quantile in config.anti_stale_quantiles
                if pd.notna(values.quantile(quantile))
            }
        )
        for threshold in thresholds:
            keep_mask = _num_col(train, feature).le(threshold).fillna(False)
            keep = train[keep_mask].copy()
            keep_stats = _summary_stats(keep)
            if int(keep_stats["count"]) < config.min_rule_keep_count:
                continue
            lift = float(keep_stats["mean_r"]) - float(base_stats["mean_r"])
            if float(keep_stats["net_r"]) <= 0.0 or lift <= 0.0:
                continue
            score = lift * math.sqrt(float(keep_stats["count"])) + 0.01 * float(
                keep_stats["net_r"]
            )
            rules.append(
                _AntiStaleRule(
                    feature=feature,
                    threshold=threshold,
                    train_score=score,
                    train_keep_count=int(keep_stats["count"]),
                    train_keep_net_r=float(keep_stats["net_r"]),
                    train_keep_mean_r=float(keep_stats["mean_r"]),
                    train_lift_mean_r=lift,
                )
            )
    if not rules:
        return None
    return sorted(
        rules,
        key=lambda rule: (
            rule.train_score,
            rule.train_keep_net_r,
            rule.train_keep_count,
            -rule.threshold,
        ),
        reverse=True,
    )[0]


def _rule_keep_mask(rows: pd.DataFrame, rule: _AntiStaleRule | None) -> pd.Series:
    if rule is None:
        return pd.Series(True, index=rows.index)
    return _num_col(rows, rule.feature).le(rule.threshold).fillna(False)


def _random_same_count(
    rows: pd.DataFrame,
    *,
    kept_count: int,
    actual_net_r: float,
    seed_key: str,
    config: ShadowCandidateTriggerConfig,
) -> dict[str, float]:
    if rows.empty:
        return {
            "random_median_net_r": math.nan,
            "random_p95_net_r": math.nan,
            "random_percentile": math.nan,
        }
    net = pd.to_numeric(rows["planned_exit_shadow_net_r"], errors="coerce").fillna(0.0).to_numpy()
    if kept_count <= 0:
        totals = np.zeros(config.random_iterations)
    elif kept_count >= len(net):
        totals = np.repeat(float(net.sum()), config.random_iterations)
    else:
        digest = hashlib.sha256(f"{config.random_seed}:{seed_key}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        indices = np.arange(len(net))
        totals = np.empty(config.random_iterations)
        for idx in range(config.random_iterations):
            kept = rng.choice(indices, size=kept_count, replace=False)
            totals[idx] = float(net[kept].sum())
    return {
        "random_median_net_r": float(np.median(totals)),
        "random_p95_net_r": float(np.quantile(totals, 0.95)),
        "random_percentile": float((totals <= actual_net_r).mean()),
    }


def _month_policy_rows(
    rows: pd.DataFrame,
    *,
    config: ShadowCandidateTriggerConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for month in sorted(rows["month"].astype(str).unique()):
        train = rows[rows["month"].astype(str) < str(month)].copy()
        replay = rows[rows["month"].astype(str).eq(str(month))].copy()
        if len(train) < config.min_train_count or replay.empty:
            continue
        rule = _select_anti_stale_rule(train, config=config)
        anti_keep = _rule_keep_mask(replay, rule)
        weak_cluster = replay["weak_context_cluster_trigger"].astype(bool)
        frozen_trigger = replay["weak_cluster_shadow_deterioration_trigger"].astype(bool)
        masks = {
            "base": pd.Series(True, index=replay.index),
            "raw_antistale": anti_keep,
            "weak_cluster_only": anti_keep | ~weak_cluster,
            "weak_cluster_shadow_deterioration": anti_keep | ~frozen_trigger,
        }
        base_stats = _summary_stats(replay)
        for policy, keep_mask in masks.items():
            kept = replay[keep_mask].copy()
            skipped = replay[~keep_mask].copy()
            kept_stats = _summary_stats(kept)
            skipped_stats = _summary_stats(skipped)
            random = _random_same_count(
                replay,
                kept_count=int(kept_stats["count"]),
                actual_net_r=float(kept_stats["net_r"]),
                seed_key=f"{month}:{policy}",
                config=config,
            )
            records.append(
                {
                    "split": str(replay["split"].iloc[0]),
                    "month": month,
                    "policy": policy,
                    "train_count": int(len(train)),
                    "base_count": base_stats["count"],
                    "base_net_r": base_stats["net_r"],
                    "kept_count": kept_stats["count"],
                    "kept_net_r": kept_stats["net_r"],
                    "skipped_count": skipped_stats["count"],
                    "skipped_net_r": skipped_stats["net_r"],
                    "kept_lift_r": float(kept_stats["net_r"]) - float(base_stats["net_r"]),
                    "weak_cluster_active_count": int(weak_cluster.sum()),
                    "shadow_deterioration_active_count": int(frozen_trigger.sum()),
                    "rule_name": rule.name if rule is not None else "",
                    "rule_feature": rule.feature if rule is not None else "",
                    "rule_threshold": rule.threshold if rule is not None else math.nan,
                    **random,
                }
            )
    return pd.DataFrame(records)


def _policy_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if monthly.empty:
        return pd.DataFrame()
    for (split, policy), group in monthly.groupby(["split", "policy"], sort=False):
        base_net = float(group["base_net_r"].sum())
        kept_net = float(group["kept_net_r"].sum())
        records.append(
            {
                "split": split,
                "policy": policy,
                "months": int(group["month"].nunique()),
                "base_count": int(group["base_count"].sum()),
                "base_net_r": base_net,
                "kept_count": int(group["kept_count"].sum()),
                "kept_net_r": kept_net,
                "skipped_count": int(group["skipped_count"].sum()),
                "skipped_net_r": float(group["skipped_net_r"].sum()),
                "kept_lift_r": kept_net - base_net,
                "positive_months": int((group["kept_net_r"] > 0.0).sum()),
                "random_median_sum": float(group["random_median_net_r"].sum()),
                "random_p95_sum": float(group["random_p95_net_r"].sum()),
                "excess_vs_random_median_sum": kept_net
                - float(group["random_median_net_r"].sum()),
            }
        )
    return pd.DataFrame(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decision(policy_summary: pd.DataFrame) -> str:
    return "continue_research_shadow_trigger_audit"


def run_shadow_candidate_trigger_audit(
    *,
    input_context_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: ShadowCandidateTriggerConfig = ShadowCandidateTriggerConfig(),
) -> ShadowCandidateTriggerResult:
    """Run the research-only shadow candidate trigger audit."""

    raw_rows = _load_context_trades(input_context_report_dir)
    feature_rows = add_shadow_candidate_trigger_features(raw_rows, config=config)
    monthly = _month_policy_rows(feature_rows, config=config)
    summary = _policy_summary(monthly)
    decision = _decision(summary)

    run_id = "shadow_candidate_trigger_audit_v0_" + datetime.now(tz=UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    decision_json_path = run_dir / "decision.json"
    shadow_candidate_features_csv_path = run_dir / "shadow_candidate_features.csv"
    monthly_policy_results_csv_path = run_dir / "monthly_policy_results.csv"
    policy_summary_csv_path = run_dir / "policy_summary.csv"
    trade_shadow_trigger_flags_csv_path = run_dir / "trade_shadow_trigger_flags.csv"

    flag_columns = [
        "symbol",
        "timestamp",
        "session_date",
        "month",
        "split",
        "personality",
        "planned_exit_shadow_net_r",
        "candidate_block_reason",
        "would_be_taken_without_safety",
        "shadow_candidate_index",
        "weak_context_score",
        "weak_context_flag",
        f"prior_{config.shadow_window}_candidate_count",
        f"prior_{config.shadow_window}_weak_context_share",
        f"prior_{config.shadow_window}_shadow_candidate_net_r",
        "weak_context_cluster_trigger",
        "weak_cluster_shadow_deterioration_trigger",
    ]
    existing_flag_columns = [column for column in flag_columns if column in feature_rows]

    _write_csv(shadow_candidate_features_csv_path, feature_rows)
    _write_csv(monthly_policy_results_csv_path, monthly)
    _write_csv(policy_summary_csv_path, summary)
    _write_csv(trade_shadow_trigger_flags_csv_path, feature_rows[existing_flag_columns])

    payload: dict[str, Any] = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "decision": decision,
        "input_context_report_dir": input_context_report_dir,
        "data_source": "existing local trade_context_features.csv report",
        "volume_label": (
            "historical_volume available upstream; this audit uses derived context and "
            "planned-exit shadow net-R fields"
        ),
        "shadow_candidate_definition": (
            "Every eligible row in trade_context_features.csv is treated as a shadow "
            "candidate, whether or not a safety or caveat layer would execute it."
        ),
        "config": {
            "shadow_window": config.shadow_window,
            "min_prior_candidates": config.min_prior_candidates,
            "weak_context_max_score": config.weak_context_max_score,
            "weak_context_share_threshold": config.weak_context_share_threshold,
            "shadow_net_r_threshold": config.shadow_net_r_threshold,
            "anti_stale_windows": list(config.anti_stale_windows),
            "anti_stale_feature_bases": list(config.anti_stale_feature_bases),
            "anti_stale_quantiles": list(config.anti_stale_quantiles),
            "min_train_count": config.min_train_count,
            "min_rule_keep_count": config.min_rule_keep_count,
        },
        "policy_summary": summary.to_dict(orient="records"),
    }
    _write_json(summary_json_path, payload)
    _write_json(
        decision_json_path,
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
        },
    )
    markdown = [
        "# Shadow Candidate Trigger Audit V0",
        "",
        "Research-only. No edge is claimed.",
        "",
        "This audit separates the eligible shadow opportunity stream from executed trades.",
        "",
        "## Frozen Trigger",
        "",
        (
            f"Enable anti-staleness only when prior {config.shadow_window} shadow "
            f"candidates have weak-context share >= {config.weak_context_share_threshold:g} "
            f"and shadow net R <= {config.shadow_net_r_threshold:g}R."
        ),
        "",
        "## Policy Summary",
        "",
        _markdown_table(summary),
        "",
    ]
    summary_markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    return ShadowCandidateTriggerResult(
        run_id=run_id,
        input_context_report_dir=input_context_report_dir,
        output_dir=run_dir,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        decision_json_path=decision_json_path,
        shadow_candidate_features_csv_path=shadow_candidate_features_csv_path,
        monthly_policy_results_csv_path=monthly_policy_results_csv_path,
        policy_summary_csv_path=policy_summary_csv_path,
        trade_shadow_trigger_flags_csv_path=trade_shadow_trigger_flags_csv_path,
        decision=decision,
    )


__all__ = [
    "ShadowCandidateTriggerConfig",
    "ShadowCandidateTriggerResult",
    "add_shadow_candidate_trigger_features",
    "run_shadow_candidate_trigger_audit",
]
