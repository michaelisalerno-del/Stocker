"""Research-only historical live replay for frozen personality candidates.

This module replays a supplied candidate book over a date range as if the
candidates were frozen before the month. It consumes existing event rows only,
does not fetch vendor data, and does not touch broker, paper, live, or order
placement paths.
"""

from __future__ import annotations

import math
import re
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
from stocker_research.personality_stop_validation_v0 import _risk_bps_for_model
from stocker_research.personality_template_v0 import (
    DEFAULT_TEMPLATE_PATH,
    _base_mask,
    _write_csv,
    _write_json,
    load_personality_templates,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_live_replay_v0")


@dataclass(frozen=True)
class LiveReplayConfig:
    """Configuration for historical live-style replay."""

    cost_bps: float = 10.0
    random_iterations: int = 100
    random_seed: int = 1337
    one_trade_per_symbol_session: bool = True
    include_candidate_status: str = "candidate_continue_research"
    structure_buffer_bps: float = 10.0
    min_structure_stop_bps: float = 5.0


@dataclass(frozen=True)
class PersonalityLiveReplayResult:
    """Paths and headline result for a personality live replay run."""

    run_id: str
    input_event_dir: Path
    candidate_book_path: Path
    template_path: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    signals_csv_path: Path
    trades_csv_path: Path
    daily_pnl_csv_path: Path
    symbol_summary_csv_path: Path
    personality_summary_csv_path: Path
    random_live_baseline_csv_path: Path
    missed_candidates_csv_path: Path
    blocked_by_dead_chop_csv_path: Path
    decision: str
    trade_count: int


def _parse_filter_rule(filter_rule: str) -> dict[str, str | float]:
    """Parse a simple one-feature numeric candidate-book filter rule."""

    match = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*([-+0-9.eE]+)\s*", filter_rule)
    if not match:
        raise ValueError(f"Unsupported filter rule: {filter_rule}")
    feature, operator, threshold = match.groups()
    return {
        "rule_kind": "single",
        "feature": feature,
        "operator": operator,
        "threshold": float(threshold),
        "feature_b": "",
        "operator_b": "",
        "threshold_b": math.nan,
        "filter_rule": filter_rule,
    }


def _add_missing_discovery_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add discovery fields while keeping event-report columns as source of truth."""

    explicit = {column: events[column].copy() for column in events.columns}
    enriched = add_discovery_features(events)
    for column, values in explicit.items():
        enriched[column] = values
    return enriched


def _replay_window(events: pd.DataFrame, replay_start: str, replay_end: str) -> pd.DataFrame:
    data = events.copy()
    timestamps = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    start = pd.Timestamp(replay_start, tz="UTC")
    end = pd.Timestamp(replay_end, tz="UTC") + pd.Timedelta(days=1)
    return data[(timestamps >= start) & (timestamps < end)].copy()


def _candidate_to_filter_row(candidate: pd.Series) -> pd.Series:
    parsed = _parse_filter_rule(str(candidate["filter_rule"]))
    return pd.Series(parsed)


def _candidate_matches(
    replay_events: pd.DataFrame,
    candidate_book: pd.DataFrame,
    *,
    template_path: Path,
    config: LiveReplayConfig,
) -> pd.DataFrame:
    template_book = load_personality_templates(template_path)
    templates = {template.template_id: template for template in template_book.templates}
    signal_frames: list[pd.DataFrame] = []
    active_candidates = candidate_book.copy()
    if "candidate_status" in active_candidates:
        active_candidates = active_candidates[
            active_candidates["candidate_status"].astype(str).eq(config.include_candidate_status)
        ].copy()
    active_candidates = active_candidates.reset_index(drop=True)
    for candidate_rank, candidate in active_candidates.iterrows():
        template = templates.get(str(candidate["template_id"]))
        if template is None:
            continue
        horizon = int(candidate["horizon"])
        ret_col = _return_column(horizon)
        if ret_col not in replay_events:
            continue
        base = replay_events[_base_mask(replay_events, template)].copy()
        if base.empty:
            continue
        regime_field = str(candidate["regime_field"])
        if regime_field not in base:
            continue
        regime = base[base[regime_field].astype(str).eq(str(candidate["regime_value"]))].copy()
        if regime.empty:
            continue
        filter_row = _candidate_to_filter_row(candidate)
        matched = regime[_candidate_mask(regime, filter_row)].copy()
        if matched.empty:
            continue
        risk = _risk_bps_for_model(
            matched,
            model_name=str(candidate["stop_model"]),
            expected_direction=int(candidate["expected_direction"]),
            structure_buffer_bps=config.structure_buffer_bps,
            min_structure_stop_bps=config.min_structure_stop_bps,
        )
        matched["candidate_rank"] = int(candidate_rank)
        matched["template_id"] = str(candidate["template_id"])
        matched["personality"] = str(candidate.get("personality", candidate["template_id"]))
        matched["role"] = str(candidate["role"])
        matched["horizon"] = horizon
        matched["expected_direction"] = int(candidate["expected_direction"])
        matched["regime_field"] = regime_field
        matched["regime_value"] = str(candidate["regime_value"])
        matched["filter_rule"] = str(candidate["filter_rule"])
        matched["stop_model"] = str(candidate["stop_model"])
        matched["target_r"] = float(candidate["target_r"])
        matched["risk_bps"] = risk
        signal_frames.append(matched)
    if not signal_frames:
        return pd.DataFrame()
    signals = pd.concat(signal_frames, ignore_index=True)
    signals["_timestamp_sort"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
    signals = signals.sort_values(
        ["_timestamp_sort", "symbol", "candidate_rank"],
        kind="mergesort",
    )
    signal_key = ["symbol", "timestamp", "session_date"]
    signals = signals.drop_duplicates(signal_key, keep="first").drop(columns=["_timestamp_sort"])
    return signals.reset_index(drop=True)


def simulate_trade_outcome(
    row: pd.Series,
    *,
    horizon: int,
    expected_direction: int,
    risk_bps: float,
    target_r: float,
    cost_bps: float,
) -> dict[str, float | bool | str]:
    """Simulate a target/stop/time exit from forward target columns.

    If target and stop are both touched within the horizon, ordering is unknown
    from event-row MFE/MAE, so the replay scores the event as stop-first.
    """

    ret_col = _return_column(horizon)
    mfe_col = f"forward_{horizon}_bar_mfe"
    mae_col = f"forward_{horizon}_bar_mae"
    risk = float(risk_bps)
    aligned_return_bps = float(expected_direction) * float(row[ret_col]) * 10000.0
    mfe_bps = float(row.get(mfe_col, math.nan)) * 10000.0
    mae_bps = float(row.get(mae_col, math.nan)) * 10000.0
    favorable_bps = mfe_bps if expected_direction > 0 else -mae_bps
    adverse_bps = -mae_bps if expected_direction > 0 else mfe_bps
    stop_hit = bool(adverse_bps >= risk)
    target_hit = bool(favorable_bps >= risk * float(target_r))
    ambiguous = bool(stop_hit and target_hit)
    if ambiguous:
        gross_r = -1.0
        exit_reason = "ambiguous_stop_first"
    elif stop_hit:
        gross_r = -1.0
        exit_reason = "stop"
    elif target_hit:
        gross_r = float(target_r)
        exit_reason = "target"
    else:
        gross_r = aligned_return_bps / risk
        exit_reason = "time_exit"
    cost_r = float(cost_bps) / risk
    return {
        "risk_bps": risk,
        "aligned_return_bps": aligned_return_bps,
        "favorable_excursion_bps": favorable_bps,
        "adverse_excursion_bps": adverse_bps,
        "stop_hit": stop_hit,
        "target_hit": target_hit,
        "target_stop_order_ambiguous": ambiguous,
        "gross_r": gross_r,
        "cost_bps": float(cost_bps),
        "cost_r": cost_r,
        "net_r": gross_r - cost_r,
        "exit_reason": exit_reason,
    }


def _signals_to_trades(
    signals: pd.DataFrame,
    *,
    config: LiveReplayConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), signals.copy()
    ordered = signals.copy()
    ordered["_timestamp_sort"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.sort_values(["_timestamp_sort", "symbol"], kind="mergesort")
    if config.one_trade_per_symbol_session:
        trade_mask = ~ordered[["symbol", "session_date"]].astype(str).agg(
            "|".join,
            axis=1,
        ).duplicated()
    else:
        trade_mask = pd.Series(True, index=ordered.index)
    trades = ordered[trade_mask].copy()
    missed = ordered[~trade_mask].copy()
    if not missed.empty:
        missed["miss_reason"] = "one_trade_per_symbol_session"
    outcome_rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        outcome = simulate_trade_outcome(
            row,
            horizon=int(row["horizon"]),
            expected_direction=int(row["expected_direction"]),
            risk_bps=float(row["risk_bps"]),
            target_r=float(row["target_r"]),
            cost_bps=config.cost_bps,
        )
        outcome_rows.append(outcome)
    outcomes = pd.DataFrame(outcome_rows, index=trades.index)
    outcomes = outcomes.drop(
        columns=[column for column in outcomes.columns if column in trades.columns],
        errors="ignore",
    )
    trades = pd.concat([trades.drop(columns=["_timestamp_sort"]), outcomes], axis=1)
    missed = missed.drop(columns=["_timestamp_sort"])
    return trades.reset_index(drop=True), missed.reset_index(drop=True)


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


def _summary_by(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                by,
                "trade_count",
                "symbol_count",
                "session_count",
                "total_net_r",
                "median_net_r",
                "win_rate",
                "stop_hit_rate",
            ]
        )
    return (
        trades.groupby(by)
        .agg(
            trade_count=("net_r", "size"),
            symbol_count=("symbol", "nunique"),
            session_count=("session_date", "nunique"),
            total_net_r=("net_r", "sum"),
            median_net_r=("net_r", "median"),
            win_rate=("net_r", lambda values: float((values > 0).mean())),
            stop_hit_rate=("stop_hit", lambda values: float(pd.Series(values).astype(bool).mean())),
        )
        .reset_index()
        .sort_values("total_net_r", ascending=False)
    )


def _blocked_by_dead_chop(replay_events: pd.DataFrame) -> pd.DataFrame:
    blocked = replay_events[replay_events["event_state"].astype(str).eq("dead_chop_blocker")].copy()
    keep_columns = [
        column
        for column in [
            "symbol",
            "timestamp",
            "session_date",
            "bar_index_in_session",
            "event_state",
            "forward_12_bar_abs_return",
            "forward_24_bar_abs_return",
        ]
        if column in blocked.columns
    ]
    return blocked[keep_columns].reset_index(drop=True)


def _random_live_baseline(
    replay_events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    config: LiveReplayConfig,
) -> pd.DataFrame:
    if trades.empty or replay_events.empty:
        return pd.DataFrame(
            columns=[
                "iteration",
                "trade_count",
                "total_net_r",
                "median_net_r",
                "win_rate",
            ]
        )
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []
    replay_pool = replay_events.reset_index(drop=True)
    params = trades[
        ["horizon", "expected_direction", "stop_model", "target_r"]
    ].reset_index(drop=True)
    valid_indices_by_horizon = {
        horizon: replay_pool.index[replay_pool[_return_column(horizon)].notna()].to_numpy()
        for horizon in sorted(params["horizon"].astype(int).unique())
        if _return_column(horizon) in replay_pool
    }
    for iteration in range(config.random_iterations):
        net_r_values: list[float] = []
        for _, param in params.iterrows():
            horizon = int(param["horizon"])
            valid_indices = valid_indices_by_horizon.get(horizon, np.array([], dtype=int))
            if len(valid_indices) == 0:
                continue
            event = replay_pool.iloc[int(rng.choice(valid_indices))]
            risk = _risk_bps_for_model(
                pd.DataFrame([event]),
                model_name=str(param["stop_model"]),
                expected_direction=int(param["expected_direction"]),
                structure_buffer_bps=config.structure_buffer_bps,
                min_structure_stop_bps=config.min_structure_stop_bps,
            ).iloc[0]
            outcome = simulate_trade_outcome(
                event,
                horizon=horizon,
                expected_direction=int(param["expected_direction"]),
                risk_bps=float(risk),
                target_r=float(param["target_r"]),
                cost_bps=config.cost_bps,
            )
            net_r_values.append(float(outcome["net_r"]))
        if net_r_values:
            rows.append(
                {
                    "iteration": iteration,
                    "trade_count": len(net_r_values),
                    "total_net_r": float(np.sum(net_r_values)),
                    "median_net_r": float(np.median(net_r_values)),
                    "win_rate": float(np.mean(np.array(net_r_values) > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _decision(trades: pd.DataFrame, random_baseline: pd.DataFrame) -> str:
    if trades.empty:
        return "reject_no_replay_trades"
    total_net_r = float(trades["net_r"].sum())
    if total_net_r <= 0:
        return "reject_negative_live_replay"
    if not random_baseline.empty:
        random_median_total = float(random_baseline["total_net_r"].median())
        if total_net_r <= random_median_total:
            return "reject_random_live_baseline_better"
    if trades["symbol"].nunique() < 2:
        return "reject_concentrated"
    return "continue_research_live_replay"


def _write_summary(
    path: Path,
    *,
    input_event_dir: Path,
    candidate_book_path: Path,
    replay_start: str,
    replay_end: str,
    decision: str,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    random_baseline: pd.DataFrame,
) -> None:
    total_net_r = float(trades["net_r"].sum()) if not trades.empty else 0.0
    win_rate = float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan
    max_drawdown = float(daily["drawdown_r"].min()) if not daily.empty else math.nan
    random_total = (
        float(random_baseline["total_net_r"].median()) if not random_baseline.empty else math.nan
    )
    lines = [
        "# Personality Live Replay V0",
        "",
        (
            "Research-only historical live-style replay over frozen personality candidates. "
            "No broker, no IG, no live trading, no paper trading, no vendor fetching, "
            "and no order placement. No edge is claimed."
        ),
        "",
        f"Input event report: `{input_event_dir}`",
        f"Candidate book: `{candidate_book_path}`",
        f"Replay window: `{replay_start}` to `{replay_end}`",
        f"Decision: `{decision}`",
        "",
        "## Headline",
        "",
        f"- Signals: `{len(signals)}`",
        f"- Trades: `{len(trades)}`",
        f"- Total net R: `{total_net_r:.2f}`",
        f"- Win rate: `{win_rate:.1%}`" if not math.isnan(win_rate) else "- Win rate: `n/a`",
        f"- Max drawdown R: `{max_drawdown:.2f}`"
        if not math.isnan(max_drawdown)
        else "- Max drawdown R: `n/a`",
        f"- Random same-count median total R: `{random_total:.2f}`"
        if not math.isnan(random_total)
        else "- Random same-count median total R: `n/a`",
        "",
        "## Caveat",
        "",
        (
            "This replay assumes the supplied candidate book was frozen before the replay "
            "month. Signal detection uses event-bar features only; forward return, MFE, "
            "and MAE are used only after the signal to score the simulated outcome."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_personality_live_replay_lab(
    *,
    input_event_dir: Path,
    candidate_book_path: Path,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    replay_start: str,
    replay_end: str,
    config: LiveReplayConfig = LiveReplayConfig(),
) -> PersonalityLiveReplayResult:
    """Run a one-window historical live-style replay over frozen candidates."""

    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    if not candidate_book_path.exists():
        raise FileNotFoundError(f"Missing candidate book: {candidate_book_path}")
    event_rows = pd.read_csv(event_rows_path)
    candidate_book = pd.read_csv(candidate_book_path)
    events = _add_missing_discovery_features(event_rows)
    replay_events = _replay_window(events, replay_start, replay_end)
    signals = _candidate_matches(
        replay_events,
        candidate_book,
        template_path=template_path,
        config=config,
    )
    trades, missed = _signals_to_trades(signals, config=config)
    daily = _daily_pnl(trades)
    symbol_summary = _summary_by(trades, "symbol")
    personality_summary = _summary_by(trades, "template_id")
    random_baseline = _random_live_baseline(replay_events, trades, config=config)
    blocked = _blocked_by_dead_chop(replay_events)
    decision = _decision(trades, random_baseline)

    run_id = f"personality_live_replay_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "signals": run_dir / "signals.csv",
        "trades": run_dir / "trades.csv",
        "daily": run_dir / "daily_pnl.csv",
        "symbol": run_dir / "symbol_summary.csv",
        "personality": run_dir / "personality_summary.csv",
        "random": run_dir / "random_live_baseline.csv",
        "missed": run_dir / "missed_candidates.csv",
        "blocked": run_dir / "blocked_by_dead_chop.csv",
    }
    for path, frame in [
        (paths["signals"], signals),
        (paths["trades"], trades),
        (paths["daily"], daily),
        (paths["symbol"], symbol_summary),
        (paths["personality"], personality_summary),
        (paths["random"], random_baseline),
        (paths["missed"], missed),
        (paths["blocked"], blocked),
    ]:
        _write_csv(path, frame)

    total_net_r = float(trades["net_r"].sum()) if not trades.empty else 0.0
    win_rate = float((trades["net_r"] > 0.0).mean()) if not trades.empty else math.nan
    daily_mean = float(daily["daily_net_r"].mean()) if not daily.empty else math.nan
    max_drawdown = float(daily["drawdown_r"].min()) if not daily.empty else math.nan
    random_median_total = (
        float(random_baseline["total_net_r"].median()) if not random_baseline.empty else math.nan
    )
    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_event_dir": str(input_event_dir),
        "candidate_book_path": str(candidate_book_path),
        "template_path": str(template_path),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "replay_start": replay_start,
        "replay_end": replay_end,
        "decision": decision,
        "signal_count": int(len(signals)),
        "trade_count": int(len(trades)),
        "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
        "session_count": int(trades["session_date"].nunique()) if not trades.empty else 0,
        "total_net_r": total_net_r,
        "win_rate": win_rate,
        "average_daily_net_r": daily_mean,
        "max_drawdown_r": max_drawdown,
        "random_same_count_median_total_net_r": random_median_total,
        "cost_bps": float(config.cost_bps),
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)
    _write_summary(
        paths["summary_md"],
        input_event_dir=input_event_dir,
        candidate_book_path=candidate_book_path,
        replay_start=replay_start,
        replay_end=replay_end,
        decision=decision,
        signals=signals,
        trades=trades,
        daily=daily,
        random_baseline=random_baseline,
    )
    return PersonalityLiveReplayResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        candidate_book_path=candidate_book_path,
        template_path=template_path,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        signals_csv_path=paths["signals"],
        trades_csv_path=paths["trades"],
        daily_pnl_csv_path=paths["daily"],
        symbol_summary_csv_path=paths["symbol"],
        personality_summary_csv_path=paths["personality"],
        random_live_baseline_csv_path=paths["random"],
        missed_candidates_csv_path=paths["missed"],
        blocked_by_dead_chop_csv_path=paths["blocked"],
        decision=decision,
        trade_count=int(len(trades)),
    )


__all__ = [
    "LiveReplayConfig",
    "PersonalityLiveReplayResult",
    "run_personality_live_replay_lab",
    "simulate_trade_outcome",
]
