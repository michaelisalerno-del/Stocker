"""Research-only personality expression replay lab.

This lab tests strict tradable expressions of broad personalities. It consumes
local state-event rows and local personality Discovery rule output, selects
expression/filter/exit rows on train months only, then evaluates the selected
expressions on held-out months. It does not fetch data, touch broker execution,
paper trading, live trading, or order placement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.personality_discovery_v0 import EVENT_STATE_PERSONALITY
from stocker_research.personality_live_replay_v0 import _add_missing_discovery_features
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    EVENT_DIRECTIONS,
    _apply_filter_candidate,
    _dedupe_trades,
    _materialize_combo,
    _score_exit_model,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_expression_lab_v0")

DEFAULT_ALLOWED_PERSONALITIES: tuple[str, ...] = (
    "active_liquidation",
    "impulse_recoil",
    "slow_repair",
)

PERSONALITY_EVENT_STATE: dict[str, str] = {
    personality: event_state
    for event_state, (personality, _role, direction) in EVENT_STATE_PERSONALITY.items()
    if direction != 0
}


@dataclass(frozen=True)
class PersonalityExpressionLabConfig:
    """Configuration for the personality expression lab."""

    train_months: tuple[str, ...] = ("2026-01", "2026-02", "2026-03", "2026-04")
    test_months: tuple[str, ...] = ("2026-05", "2026-06")
    allowed_personalities: tuple[str, ...] = DEFAULT_ALLOWED_PERSONALITIES
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
    max_rule_candidates_per_personality: int = 80
    max_expressions_per_personality: int = 1
    min_train_trades: int = 20
    min_train_months: int = 3
    min_train_total_net_r: float = 0.0
    min_train_win_rate: float = 0.55
    min_oos_trades: int = 1


@dataclass(frozen=True)
class PersonalityExpressionLabResult:
    """Paths and headline result for one expression lab run."""

    run_id: str
    input_event_dir: Path
    input_personality_discovery_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    expression_candidate_sweep_csv_path: Path
    selected_expressions_csv_path: Path
    train_signals_csv_path: Path
    train_trades_csv_path: Path
    test_signals_csv_path: Path
    test_trades_csv_path: Path
    personality_summary_csv_path: Path
    decision: str
    test_trade_count: int


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


def _load_events(input_event_dir: Path) -> pd.DataFrame:
    path = input_event_dir / "event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event rows: {path}")
    rows = pd.read_csv(path)
    events = _add_missing_discovery_features(rows).copy()
    events["month"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce").dt.strftime(
        "%Y-%m"
    )
    return cast(pd.DataFrame, events)


def _expression_columns() -> list[str]:
    return [
        "expression_id",
        "personality",
        "event_state",
        "horizon",
        "expected_direction",
        "regime_field",
        "regime_value",
        "filter_rule",
        "rule_kind",
        "filter_feature",
        "filter_operator",
        "filter_threshold",
        "feature_b",
        "operator_b",
        "threshold_b",
        "stop_model",
        "target_r",
        "selection_score",
        "train_count",
        "train_total_net_r",
        "train_mean_net_r",
        "train_win_rate",
        "train_month_count",
        "train_symbol_count",
        "test_count",
        "test_total_net_r",
        "test_mean_net_r",
        "test_win_rate",
        "test_month_count",
        "test_symbol_count",
    ]


def _empty_expression_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_expression_columns())


def _load_expression_rules(
    input_personality_discovery_dir: Path,
    *,
    config: PersonalityExpressionLabConfig,
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
    data = data[data["personality"].astype(str).isin(config.allowed_personalities)].copy()
    data = data[data["personality"].astype(str).isin(PERSONALITY_EVENT_STATE)].copy()
    if data.empty:
        return pd.DataFrame()
    for column, default in [
        ("rule_kind", "single"),
        ("feature_b", ""),
        ("operator_b", ""),
        ("threshold_b", math.nan),
        ("retained_test_count", 0),
        ("test_lift_vs_personality", 0.0),
        ("filtered_test_same_result_rate", 0.0),
    ]:
        if column not in data:
            data[column] = default
    data = data.sort_values(
        ["retained_test_count", "test_lift_vs_personality", "filtered_test_same_result_rate"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    data = data.drop_duplicates(
        [
            "personality",
            "horizon",
            "regime_field",
            "regime_value",
            "rule_kind",
            "feature",
            "operator",
            "threshold",
            "feature_b",
            "operator_b",
            "threshold_b",
        ],
        keep="first",
    )
    return (
        data.groupby("personality", as_index=False)
        .head(config.max_rule_candidates_per_personality)
        .reset_index(drop=True)
    )


def _rule_to_combo(rule: pd.Series) -> pd.Series:
    personality = str(rule["personality"])
    event_state = PERSONALITY_EVENT_STATE[personality]
    return pd.Series(
        {
            "personality": personality,
            "event_state": event_state,
            "horizon": int(rule["horizon"]),
            "direction": int(EVENT_DIRECTIONS[event_state]),
            "regime_field": str(rule["regime_field"]),
            "regime_value": str(rule["regime_value"]),
            "filter_rule": str(rule["filter_rule"]),
            "rule_kind": str(rule.get("rule_kind", "single") or "single"),
            "filter_feature": str(rule["feature"]),
            "filter_operator": str(rule["operator"]),
            "filter_threshold": float(rule["threshold"]),
            "feature_b": "" if pd.isna(rule.get("feature_b", "")) else str(rule["feature_b"]),
            "operator_b": "" if pd.isna(rule.get("operator_b", "")) else str(rule["operator_b"]),
            "threshold_b": pd.to_numeric(rule.get("threshold_b", math.nan), errors="coerce"),
        }
    )


def _split_months(rows: pd.DataFrame, months: tuple[str, ...]) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows[rows["month"].astype(str).isin(months)].copy()


def _stats(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if rows.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_total_net_r": 0.0,
            f"{prefix}_mean_net_r": math.nan,
            f"{prefix}_win_rate": math.nan,
            f"{prefix}_month_count": 0,
            f"{prefix}_symbol_count": 0,
        }
    net_r = pd.to_numeric(rows["net_r"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_count": int(len(rows)),
        f"{prefix}_total_net_r": float(net_r.sum()),
        f"{prefix}_mean_net_r": float(net_r.mean()),
        f"{prefix}_win_rate": float((net_r > 0.0).mean()),
        f"{prefix}_month_count": int(rows["month"].nunique()) if "month" in rows else 0,
        f"{prefix}_symbol_count": int(rows["symbol"].nunique()) if "symbol" in rows else 0,
    }


def _score_expression_split(
    rows: pd.DataFrame,
    *,
    combo: pd.Series,
    stop_model: str,
    target_r: float,
    cost_bps: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    scored = _score_exit_model(
        rows,
        horizon=int(combo["horizon"]),
        expected_direction=int(combo["direction"]),
        stop_model=stop_model,
        target_r=float(target_r),
        cost_bps=cost_bps,
    )
    scored["exit_selection_score"] = 0.0
    trades, _missed = _dedupe_trades(scored)
    return cast(pd.DataFrame, trades)


def _candidate_sweep(
    events: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    config: PersonalityExpressionLabConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule_position, (_rule_index, rule) in enumerate(rules.iterrows()):
        combo = _rule_to_combo(rule)
        materialized = _materialize_combo(events, combo)
        if materialized.empty:
            continue
        try:
            filtered = _apply_filter_candidate(materialized, combo)
        except (KeyError, ValueError, TypeError):
            continue
        if filtered.empty:
            continue
        train_rows = _split_months(filtered, config.train_months)
        test_rows = _split_months(filtered, config.test_months)
        if train_rows.empty:
            continue
        for stop_model in config.stop_models:
            for target_r in config.target_r_multiples:
                train_trades = _score_expression_split(
                    train_rows,
                    combo=combo,
                    stop_model=stop_model,
                    target_r=float(target_r),
                    cost_bps=config.cost_bps,
                )
                if train_trades.empty:
                    continue
                test_trades = _score_expression_split(
                    test_rows,
                    combo=combo,
                    stop_model=stop_model,
                    target_r=float(target_r),
                    cost_bps=config.cost_bps,
                )
                train_stats = _stats(train_trades, "train")
                test_stats = _stats(test_trades, "test")
                score = (
                    float(train_stats["train_total_net_r"])
                    + 2.0 * float(train_stats["train_mean_net_r"])
                    + 5.0 * (float(train_stats["train_win_rate"]) - 0.50)
                    + 0.01 * float(train_stats["train_count"])
                )
                rows.append(
                    {
                        "expression_id": rule_position,
                        "personality": combo["personality"],
                        "event_state": combo["event_state"],
                        "horizon": int(combo["horizon"]),
                        "expected_direction": int(combo["direction"]),
                        "regime_field": combo["regime_field"],
                        "regime_value": combo["regime_value"],
                        "filter_rule": combo["filter_rule"],
                        "rule_kind": combo["rule_kind"],
                        "filter_feature": combo["filter_feature"],
                        "filter_operator": combo["filter_operator"],
                        "filter_threshold": float(combo["filter_threshold"]),
                        "feature_b": combo["feature_b"],
                        "operator_b": combo["operator_b"],
                        "threshold_b": combo["threshold_b"],
                        "stop_model": stop_model,
                        "target_r": float(target_r),
                        "selection_score": score,
                        **train_stats,
                        **test_stats,
                    }
                )
    if not rows:
        return _empty_expression_frame()
    return pd.DataFrame(rows).loc[:, _expression_columns()]


def _select_expressions(
    sweep: pd.DataFrame,
    *,
    config: PersonalityExpressionLabConfig,
) -> pd.DataFrame:
    if sweep.empty:
        return _empty_expression_frame()
    eligible = sweep[
        (sweep["train_count"] >= config.min_train_trades)
        & (sweep["train_month_count"] >= config.min_train_months)
        & (sweep["train_total_net_r"] >= config.min_train_total_net_r)
        & (sweep["train_win_rate"] >= config.min_train_win_rate)
    ].copy()
    if eligible.empty:
        return _empty_expression_frame()
    selected = (
        eligible.sort_values(
            ["selection_score", "train_total_net_r", "train_count"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .groupby("personality", as_index=False)
        .head(config.max_expressions_per_personality)
        .reset_index(drop=True)
    )
    selected["expression_id"] = np.arange(len(selected), dtype=int)
    return selected.loc[:, _expression_columns()]


def _apply_selected_expressions(
    events: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    months: tuple[str, ...],
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    for _, expression in selected.iterrows():
        combo = pd.Series(
            {
                "personality": expression["personality"],
                "event_state": expression["event_state"],
                "horizon": int(expression["horizon"]),
                "direction": int(expression["expected_direction"]),
                "regime_field": expression["regime_field"],
                "regime_value": expression["regime_value"],
                "filter_rule": expression["filter_rule"],
                "rule_kind": expression["rule_kind"],
                "filter_feature": expression["filter_feature"],
                "filter_operator": expression["filter_operator"],
                "filter_threshold": float(expression["filter_threshold"]),
                "feature_b": expression["feature_b"],
                "operator_b": expression["operator_b"],
                "threshold_b": expression["threshold_b"],
            }
        )
        materialized = _materialize_combo(events, combo)
        filtered = _apply_filter_candidate(materialized, combo)
        split = _split_months(filtered, months)
        if split.empty:
            continue
        scored = _score_exit_model(
            split,
            horizon=int(expression["horizon"]),
            expected_direction=int(expression["expected_direction"]),
            stop_model=str(expression["stop_model"]),
            target_r=float(expression["target_r"]),
            cost_bps=cost_bps,
        )
        if scored.empty:
            continue
        scored["expression_id"] = int(expression["expression_id"])
        scored["filter_rule"] = expression["filter_rule"]
        scored["filter_feature"] = expression["filter_feature"]
        scored["filter_operator"] = expression["filter_operator"]
        scored["filter_threshold"] = float(expression["filter_threshold"])
        scored["selection_score"] = float(expression["selection_score"])
        scored["exit_selection_score"] = float(expression["selection_score"])
        frames.append(scored)
    signals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if signals.empty:
        trades = pd.DataFrame()
    else:
        trades, _missed = _dedupe_trades(signals)
    return signals, trades


def _personality_summary(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "personality",
        "trade_count",
        "symbol_count",
        "month_count",
        "total_net_r",
        "mean_net_r",
        "win_rate",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for personality, group in trades.groupby("personality", dropna=False):
        net_r = pd.to_numeric(group["net_r"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "personality": str(personality),
                "trade_count": int(len(group)),
                "symbol_count": int(group["symbol"].nunique()),
                "month_count": int(group["month"].nunique()),
                "total_net_r": float(net_r.sum()),
                "mean_net_r": float(net_r.mean()),
                "win_rate": float((net_r > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        "total_net_r",
        ascending=False,
        kind="mergesort",
    )


def _decision(
    selected: pd.DataFrame,
    test_trades: pd.DataFrame,
    config: PersonalityExpressionLabConfig,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if selected.empty:
        return "reject_no_selected_expressions", ["no_selected_expressions"]
    if len(test_trades) < config.min_oos_trades:
        reasons.append("low_oos_trade_count")
    total = float(test_trades["net_r"].sum()) if not test_trades.empty else 0.0
    if total <= 0.0:
        reasons.append("non_positive_oos_total_net_r")
    if reasons:
        return f"reject_{reasons[0]}", reasons
    return "continue_research_personality_expression_lab", reasons


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
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
    personality: pd.DataFrame,
) -> None:
    selected_cols = [
        column
        for column in [
            "personality",
            "horizon",
            "regime_field",
            "regime_value",
            "filter_rule",
            "stop_model",
            "target_r",
            "train_count",
            "train_total_net_r",
            "test_count",
            "test_total_net_r",
        ]
        if column in selected
    ]
    lines = [
        "# Personality Expression Lab V0",
        "",
        (
            "Research-only replay of strict personality expressions selected on train months "
            "and evaluated on held-out months. No broker, IG, live trading, paper trading, "
            "vendor fetching, or order placement. No edge is claimed."
        ),
        "",
        f"Decision: `{payload['decision']}`",
        "Pipeline: `personality -> expression regime/filter -> exit`",
        f"Allowed personalities: `{', '.join(payload['allowed_personalities'])}`",
        f"Train months: `{', '.join(payload['train_months'])}`",
        f"Test months: `{', '.join(payload['test_months'])}`",
        "Volume label: `historical_volume from existing local 5m OHLCV event report`",
        "",
        "## Headline",
        "",
        f"- Candidate sweep rows: `{payload['candidate_sweep_count']}`",
        f"- Selected expressions: `{payload['selected_expression_count']}`",
        f"- Train trades: `{payload['train_trade_count']}`",
        f"- Train total net R: `{payload['train_total_net_r']:.2f}`",
        f"- Test trades: `{payload['test_trade_count']}`",
        f"- Test total net R: `{payload['test_total_net_r']:.2f}`",
        "",
        "## Selected Expressions",
        "",
        _markdown_table(selected[selected_cols] if selected_cols else selected),
        "",
        "## Personality Summary",
        "",
        _markdown_table(personality),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_personality_expression_lab(
    *,
    input_event_dir: Path,
    input_personality_discovery_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityExpressionLabConfig = PersonalityExpressionLabConfig(),
) -> PersonalityExpressionLabResult:
    """Run a research-only personality expression replay."""

    events = _load_events(input_event_dir)
    rules = _load_expression_rules(input_personality_discovery_dir, config=config)
    sweep = (
        _candidate_sweep(events, rules, config=config)
        if not rules.empty
        else _empty_expression_frame()
    )
    selected = _select_expressions(sweep, config=config)
    train_signals, train_trades = _apply_selected_expressions(
        events,
        selected,
        months=config.train_months,
        cost_bps=config.cost_bps,
    )
    test_signals, test_trades = _apply_selected_expressions(
        events,
        selected,
        months=config.test_months,
        cost_bps=config.cost_bps,
    )
    personality = _personality_summary(test_trades)
    decision, decision_reasons = _decision(selected, test_trades, config)
    train_total = float(train_trades["net_r"].sum()) if not train_trades.empty else 0.0
    test_total = float(test_trades["net_r"].sum()) if not test_trades.empty else 0.0
    test_win_rate = (
        float((pd.to_numeric(test_trades["net_r"], errors="coerce") > 0.0).mean())
        if not test_trades.empty
        else math.nan
    )

    run_id = f"personality_expression_lab_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "candidate_sweep": run_dir / "expression_candidate_sweep.csv",
        "selected": run_dir / "selected_expressions.csv",
        "train_signals": run_dir / "train_signals.csv",
        "train_trades": run_dir / "train_trades.csv",
        "test_signals": run_dir / "test_signals.csv",
        "test_trades": run_dir / "test_trades.csv",
        "personality": run_dir / "personality_summary.csv",
    }
    for path, frame in [
        (paths["candidate_sweep"], sweep),
        (paths["selected"], selected),
        (paths["train_signals"], train_signals),
        (paths["train_trades"], train_trades),
        (paths["test_signals"], test_signals),
        (paths["test_trades"], test_trades),
        (paths["personality"], personality),
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
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "pipeline": "personality -> expression_regime_filter -> exit",
        "allowed_personalities": list(config.allowed_personalities),
        "train_months": list(config.train_months),
        "test_months": list(config.test_months),
        "candidate_sweep_count": int(len(sweep)),
        "selected_expression_count": int(len(selected)),
        "train_trade_count": int(len(train_trades)),
        "train_total_net_r": train_total,
        "test_trade_count": int(len(test_trades)),
        "test_total_net_r": test_total,
        "test_win_rate": test_win_rate,
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
    _write_summary_md(
        paths["summary_md"],
        payload=payload,
        selected=selected,
        personality=personality,
    )

    return PersonalityExpressionLabResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_personality_discovery_dir=input_personality_discovery_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        expression_candidate_sweep_csv_path=paths["candidate_sweep"],
        selected_expressions_csv_path=paths["selected"],
        train_signals_csv_path=paths["train_signals"],
        train_trades_csv_path=paths["train_trades"],
        test_signals_csv_path=paths["test_signals"],
        test_trades_csv_path=paths["test_trades"],
        personality_summary_csv_path=paths["personality"],
        decision=decision,
        test_trade_count=int(len(test_trades)),
    )
