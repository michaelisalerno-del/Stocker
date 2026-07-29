"""Replay sparse exhaustion-extension regime filters with conservative exits.

This research-only layer consumes the standalone sparse exhaustion-extension
event rows plus selected exhaustion regime/filter rows. It selects only
stop/target parameters from rows before each replay month, then applies the
frozen filters to the replay month. No broker, execution, vendor fetching, or
order-placement path is touched.
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
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    _concentration,
    _daily_pnl,
    _dedupe_trades,
    _exit_summary,
    _filter_mask,
    _month_bounds,
    _score_exit_model,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/exhaustion_extension_exit_replay_v0")


@dataclass(frozen=True)
class ExhaustionExtensionExitReplayConfig:
    """Configuration for sparse exhaustion-extension exit replay."""

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
    max_exit_candidates_per_month: int = 64
    max_selected_per_month: int = 8
    max_selected_per_regime_month: int = 2
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
    random_seed: int = 2701


@dataclass(frozen=True)
class ExhaustionExtensionExitReplayResult:
    """Paths and headline result for one exhaustion-extension exit replay."""

    run_id: str
    input_exhaustion_event_dir: Path
    input_filter_report_dir: Path
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
    trades_csv_path: Path
    missed_signals_csv_path: Path
    daily_pnl_csv_path: Path
    exhaustion_exit_summary_csv_path: Path
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


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
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


def _load_exhaustion_events(input_exhaustion_event_dir: Path) -> pd.DataFrame:
    path = input_exhaustion_event_dir / "exhaustion_event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing exhaustion event rows: {path}")
    rows = pd.read_csv(path)
    required = {"symbol", "timestamp", "session_date", "expected_direction"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"exhaustion_event_rows.csv missing required columns: {missing}")
    if "event_state" not in rows:
        rows["event_state"] = "exhaustion_extension"
    rows = rows[rows["event_state"].astype(str).eq("exhaustion_extension")].copy()
    rows["expected_direction"] = (
        pd.to_numeric(rows["expected_direction"], errors="coerce").fillna(0).astype(int)
    )
    rows = rows[rows["expected_direction"].isin([-1, 1])].copy()
    rows = _add_missing_discovery_features(rows).reset_index(drop=True)
    fallback_columns = {
        "distance_from_recent_high_pct": "distance_from_session_high_pct",
        "distance_from_recent_low_pct": "distance_from_session_low_pct",
        "distance_from_opening_range_high_pct": "distance_from_session_high_pct",
        "distance_from_opening_range_low_pct": "distance_from_session_low_pct",
        "distance_from_session_high_pct": None,
        "distance_from_session_low_pct": None,
    }
    for column, fallback in fallback_columns.items():
        if column in rows:
            continue
        if fallback is not None and fallback in rows:
            rows[column] = pd.to_numeric(rows[fallback], errors="coerce")
        else:
            rows[column] = 0.0
    return rows


def _infer_rule_direction(regime_value: str) -> int:
    value = str(regime_value)
    if "upside_exhaustion" in value:
        return -1
    if "downside_exhaustion" in value:
        return 1
    return 0


def _load_selected_exhaustion_filters(input_filter_report_dir: Path) -> pd.DataFrame:
    path = input_filter_report_dir / "selected_exhaustion_regime_filter_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected exhaustion filters: {path}")
    data = pd.read_csv(path)
    required = {
        "horizon",
        "regime_field",
        "regime_value",
        "feature",
        "operator",
        "threshold",
        "filter_rule",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "selected_exhaustion_regime_filter_results.csv missing required "
            f"columns: {missing}"
        )
    if "verdict" in data:
        data = data[data["verdict"].astype(str).eq("pass_exhaustion_regime_filter")].copy()
    if data.empty:
        return pd.DataFrame()
    if "test_median_lift_vs_exhaustion" in data:
        lift = pd.to_numeric(data["test_median_lift_vs_exhaustion"], errors="coerce").fillna(0.0)
    else:
        lift = pd.Series(0.0, index=data.index)
    rate = pd.to_numeric(data.get("filtered_test_same_result_rate", 0.0), errors="coerce")
    rate = rate.fillna(0.0)
    selected = pd.DataFrame(
        {
            "personality": "exhaustion_extension",
            "event_state": "exhaustion_extension",
            "horizon": pd.to_numeric(data["horizon"], errors="coerce").astype(int),
            "regime_field": data["regime_field"].astype(str),
            "regime_value": data["regime_value"].astype(str),
            "filter_feature": data["feature"].astype(str),
            "filter_operator": data["operator"].astype(str),
            "filter_threshold": pd.to_numeric(data["threshold"], errors="coerce"),
            "filter_rule": data["filter_rule"].astype(str),
            "rule_expected_direction": data["regime_value"].map(_infer_rule_direction).astype(int),
            "selection_score": lift * 10000.0 + (rate - 0.50) * 100.0,
        }
    )
    selected = selected.dropna(subset=["filter_threshold"]).copy()
    selected["selected_filter_rank"] = np.arange(len(selected), dtype=int)
    return selected.sort_values("selection_score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def _materialize_exhaustion_rule(events: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    horizon = int(rule["horizon"])
    ret_col = _return_column(horizon)
    regime_field = str(rule["regime_field"])
    feature = str(rule["filter_feature"])
    if ret_col not in events or regime_field not in events or feature not in events:
        return pd.DataFrame()
    rows = events[
        events["event_state"].astype(str).eq("exhaustion_extension")
        & events[regime_field].astype(str).eq(str(rule["regime_value"]))
        & events[ret_col].notna()
    ].copy()
    if rows.empty:
        return rows
    rule_direction = int(rule.get("rule_expected_direction", 0))
    if rule_direction in {-1, 1}:
        rows = rows[rows["expected_direction"].astype(int).eq(rule_direction)].copy()
    if rows.empty:
        return rows
    rows = rows[
        _filter_mask(
            rows,
            feature=feature,
            operator=str(rule["filter_operator"]),
            threshold=float(rule["filter_threshold"]),
        )
    ].copy()
    if rows.empty:
        return rows
    rows["horizon"] = horizon
    rows["raw_return_bps"] = pd.to_numeric(rows[ret_col], errors="coerce") * 10000.0
    rows["aligned_return_bps"] = rows["raw_return_bps"] * rows["expected_direction"].astype(int)
    rows["personality"] = "exhaustion_extension"
    rows["regime_field"] = regime_field
    rows["regime_value"] = str(rule["regime_value"])
    rows["filter_feature"] = feature
    rows["filter_operator"] = str(rule["filter_operator"])
    rows["filter_threshold"] = float(rule["filter_threshold"])
    rows["filter_rule"] = str(rule["filter_rule"])
    rows["combo_id"] = (
        "exhaustion_extension|"
        + str(horizon)
        + "|"
        + regime_field
        + "="
        + str(rule["regime_value"])
        + "|"
        + str(rule["filter_rule"])
    )
    return rows.reset_index(drop=True)


def _score_exit_model_by_direction(
    rows: pd.DataFrame,
    *,
    horizon: int,
    stop_model: str,
    target_r: float,
    cost_bps: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for direction, group in rows.groupby("expected_direction"):
        direction_int = int(direction)
        if direction_int not in {-1, 1}:
            continue
        scored = _score_exit_model(
            group.copy(),
            horizon=horizon,
            expected_direction=direction_int,
            stop_model=stop_model,
            target_r=target_r,
            cost_bps=cost_bps,
        )
        if not scored.empty:
            frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_exit_sweep(
    selected_filter_book: pd.DataFrame,
    train_events: pd.DataFrame,
    *,
    month: str,
    config: ExhaustionExtensionExitReplayConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selected_filter_rank, rule in selected_filter_book.iterrows():
        retained_train = _materialize_exhaustion_rule(train_events, rule)
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
                scored = _score_exit_model_by_direction(
                    retained_train,
                    horizon=int(rule["horizon"]),
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
                    + 0.001 * float(rule["selection_score"])
                )
                rows.append(
                    {
                        "month": month,
                        "selected_filter_rank": int(selected_filter_rank),
                        "personality": "exhaustion_extension",
                        "event_state": "exhaustion_extension",
                        "horizon": int(rule["horizon"]),
                        "rule_expected_direction": int(rule["rule_expected_direction"]),
                        "regime_field": rule["regime_field"],
                        "regime_value": rule["regime_value"],
                        "filter_feature": rule["filter_feature"],
                        "filter_operator": rule["filter_operator"],
                        "filter_threshold": float(rule["filter_threshold"]),
                        "filter_rule": rule["filter_rule"],
                        "filter_selection_score": float(rule["selection_score"]),
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


def _select_monthly_candidates(
    exit_sweep: pd.DataFrame,
    *,
    config: ExhaustionExtensionExitReplayConfig,
) -> pd.DataFrame:
    if exit_sweep.empty:
        return exit_sweep.copy()
    selected = (
        exit_sweep.sort_values(
            ["regime_field", "regime_value", "exit_selection_score", "train_exit_count"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
        .groupby(["regime_field", "regime_value"], as_index=False)
        .head(config.max_selected_per_regime_month)
        .sort_values("exit_selection_score", ascending=False, kind="mergesort")
        .head(config.max_selected_per_month)
        .reset_index(drop=True)
    )
    selected["monthly_candidate_rank"] = np.arange(len(selected), dtype=int)
    return selected


def _apply_monthly_candidates(
    selected: pd.DataFrame,
    selected_filter_book: pd.DataFrame,
    replay_events: pd.DataFrame,
    *,
    config: ExhaustionExtensionExitReplayConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, candidate in selected.iterrows():
        selected_filter = selected_filter_book.iloc[int(candidate["selected_filter_rank"])]
        retained = _materialize_exhaustion_rule(replay_events, selected_filter)
        if len(retained) < config.min_replay_signals:
            continue
        scored = _score_exit_model_by_direction(
            retained,
            horizon=int(candidate["horizon"]),
            stop_model=str(candidate["stop_model"]),
            target_r=float(candidate["target_r"]),
            cost_bps=config.cost_bps,
        )
        if scored.empty:
            continue
        scored["monthly_candidate_rank"] = int(candidate["monthly_candidate_rank"])
        scored["selected_filter_rank"] = int(candidate["selected_filter_rank"])
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


def _random_month_baseline(
    replay_events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    config: ExhaustionExtensionExitReplayConfig,
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
    rows: list[dict[str, Any]] = []
    for iteration in range(config.random_iterations):
        net_values: list[float] = []
        for _, param in params.iterrows():
            horizon = int(param["horizon"])
            ret_col = _return_column(horizon)
            if ret_col not in replay_events:
                continue
            direction = int(param["expected_direction"])
            pool = replay_events[
                replay_events[ret_col].notna()
                & replay_events["expected_direction"].astype(int).eq(direction)
            ]
            if pool.empty:
                continue
            sample = pool.sample(
                n=1,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            scored = _score_exit_model(
                sample,
                horizon=horizon,
                expected_direction=direction,
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


def _exhaustion_exit_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "regime_field",
                "regime_value",
                "filter_rule",
                "trade_count",
                "symbol_count",
                "session_count",
                "total_net_r",
                "median_net_r",
                "win_rate",
            ]
        )
    return (
        trades.groupby(["horizon", "regime_field", "regime_value", "filter_rule"], dropna=False)
        .agg(
            trade_count=("net_r", "size"),
            symbol_count=("symbol", "nunique"),
            session_count=("session_date", "nunique"),
            total_net_r=("net_r", "sum"),
            median_net_r=("net_r", "median"),
            win_rate=("net_r", lambda values: float((values > 0.0).mean())),
        )
        .reset_index()
        .sort_values("total_net_r", ascending=False, kind="mergesort")
    )


def _concentration_warnings(
    trades: pd.DataFrame,
    config: ExhaustionExtensionExitReplayConfig,
) -> pd.DataFrame:
    columns = ["scope", "share", "threshold", "warning"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    conc = _concentration(trades)
    rows: list[dict[str, Any]] = []
    thresholds = {
        "single_symbol_share": config.max_single_symbol_share,
        "single_session_share": config.max_single_session_share,
        "single_month_share": config.max_single_month_share,
    }
    for key, threshold in thresholds.items():
        share = float(conc[key])
        if share > threshold:
            rows.append(
                {
                    "scope": key,
                    "share": share,
                    "threshold": threshold,
                    "warning": f"{key} exceeds concentration threshold",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _decision(
    *,
    trades: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    random_month_sum: float,
    concentration_warnings: pd.DataFrame,
    config: ExhaustionExtensionExitReplayConfig,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    trade_count = int(len(trades))
    total_net_r = float(trades["net_r"].sum()) if not trades.empty else 0.0
    win_rate = float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan
    positive_months = (
        int((monthly_summary["total_net_r"] > 0.0).sum()) if not monthly_summary.empty else 0
    )
    if trade_count < config.min_total_trades:
        reasons.append("low sample after exit replay")
    if total_net_r <= 0.0:
        reasons.append("non-positive total net R after conservative exit replay")
    if math.isnan(win_rate) or win_rate <= 0.50:
        reasons.append("win rate did not clear 50 percent")
    if positive_months < config.min_positive_months:
        reasons.append("too few positive replay months")
    if not math.isnan(random_month_sum) and total_net_r <= random_month_sum:
        reasons.append("random same-direction exit baseline was not beaten")
    if not concentration_warnings.empty:
        reasons.append("trade set is concentration dominated")
    if "trade set is concentration dominated" in reasons:
        return "reject_concentrated", reasons
    if "low sample after exit replay" in reasons:
        return "reject_low_sample", reasons
    if "random same-direction exit baseline was not beaten" in reasons:
        return "reject_random_baseline_better", reasons
    if reasons:
        return "reject_no_exit_lift", reasons
    return "continue_research_exhaustion_exit_replay", []


def _write_summary_md(
    path: Path,
    *,
    payload: dict[str, Any],
    monthly_summary: pd.DataFrame,
    exit_summary: pd.DataFrame,
    selected: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> None:
    selected_cols = [
        column
        for column in [
            "month",
            "regime_field",
            "regime_value",
            "horizon",
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
        "# Exhaustion Extension Exit Replay V0",
        "",
        (
            "Research-only replay for sparse exhaustion-extension regime/filter rows. "
            "Filters are loaded from the prior exhaustion scan and only stop/target "
            "parameters are selected from rows before each replay month. No edge is "
            "claimed."
        ),
        "",
        f"Input exhaustion event report: `{payload['input_exhaustion_event_dir']}`",
        f"Input filter report: `{payload['input_filter_report_dir']}`",
        f"Decision: `{payload['decision']}`",
        f"Cost: `{payload['cost_bps']}` bps",
        "Stop/target ordering: `conservative_stop_first_when_both_touched`",
        "Volume label: `historical_volume from existing local candidate/event reports`",
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
        "## Exhaustion Exit Summary",
        "",
        _markdown_table(exit_summary, max_rows=50),
        "",
        "## Selected Monthly Candidates",
        "",
        _markdown_table(selected[selected_cols] if selected_cols else selected, max_rows=80),
        "",
        "## Concentration Warnings",
        "",
        _markdown_table(concentration_warnings, max_rows=20),
        "",
        "## Interpretation",
        "",
        (
            "This is still not a proven edge. It is a historical event-row replay using "
            "local forward MFE/MAE targets and conservative stop-first ordering when "
            "intrabar path is ambiguous. It is useful only as a research diagnostic."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exhaustion_extension_exit_replay_lab(
    *,
    input_exhaustion_event_dir: Path,
    input_filter_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: ExhaustionExtensionExitReplayConfig = ExhaustionExtensionExitReplayConfig(),
) -> ExhaustionExtensionExitReplayResult:
    """Run prior-only conservative exit replay for sparse exhaustion filters."""

    events = _load_exhaustion_events(input_exhaustion_event_dir)
    events["_wf_timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    selected_filter_book = _load_selected_exhaustion_filters(input_filter_report_dir)

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
        exit_sweep = _build_exit_sweep(
            selected_filter_book,
            train_events,
            month=month,
            config=config,
        )
        if not exit_sweep.empty:
            all_exit_sweeps.append(exit_sweep)
        selected = _select_monthly_candidates(exit_sweep, config=config)
        if not selected.empty:
            all_selected.append(selected)
        signals = _apply_monthly_candidates(
            selected,
            selected_filter_book,
            replay_events,
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
    exit_summary = _exhaustion_exit_summary(trade_frame)
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
        concentration_warnings=concentration_warnings,
        config=config,
    )
    total_net_r = float(trade_frame["net_r"].sum()) if not trade_frame.empty else 0.0
    win_rate = float((trade_frame["net_r"] > 0.0).mean()) if not trade_frame.empty else math.nan
    positive_month_count = (
        int((monthly_summary["total_net_r"] > 0.0).sum()) if not monthly_summary.empty else 0
    )
    max_drawdown = float(daily["drawdown_r"].min()) if not daily.empty else math.nan
    conc = _concentration(trade_frame)

    run_id = "exhaustion_extension_exit_replay_v0_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
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
        "trades": run_dir / "trades.csv",
        "missed_signals": run_dir / "missed_signals.csv",
        "daily": run_dir / "daily_pnl.csv",
        "exit_summary": run_dir / "exhaustion_exit_summary.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["selected_filter_book"], selected_filter_book),
        (paths["monthly_exit_sweep"], exit_sweep_frame),
        (paths["selected_monthly_candidates"], selected_frame),
        (paths["monthly_summary"], monthly_summary),
        (paths["random_monthly_baseline"], random_frame),
        (paths["signals"], signal_frame),
        (paths["trades"], trade_frame),
        (paths["missed_signals"], missed_frame),
        (paths["daily"], daily),
        (paths["exit_summary"], exit_summary),
        (paths["concentration"], concentration_warnings),
    ]:
        _write_csv(path, frame)

    payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "volume_label": (
            "historical_volume from existing local candidate/event reports; "
            "no new vendor fetch"
        ),
        "input_exhaustion_event_dir": str(input_exhaustion_event_dir),
        "input_filter_report_dir": str(input_filter_report_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "months": list(config.replay_months),
        "cost_bps": float(config.cost_bps),
        "stop_target_ordering": "conservative_stop_first_when_both_touched",
        "decision": decision,
        "decision_reasons": decision_reasons,
        "selected_filter_count": int(len(selected_filter_book)),
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
        payload=payload,
        monthly_summary=monthly_summary,
        exit_summary=exit_summary,
        selected=selected_frame,
        concentration_warnings=concentration_warnings,
    )
    return ExhaustionExtensionExitReplayResult(
        run_id=run_id,
        input_exhaustion_event_dir=input_exhaustion_event_dir,
        input_filter_report_dir=input_filter_report_dir,
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
        trades_csv_path=paths["trades"],
        missed_signals_csv_path=paths["missed_signals"],
        daily_pnl_csv_path=paths["daily"],
        exhaustion_exit_summary_csv_path=paths["exit_summary"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        trade_count=int(len(trade_frame)),
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "ExhaustionExtensionExitReplayConfig",
    "ExhaustionExtensionExitReplayResult",
    "run_exhaustion_extension_exit_replay_lab",
]
