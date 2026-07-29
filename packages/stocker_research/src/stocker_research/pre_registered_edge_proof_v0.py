"""Pre-registered forward proof harness for a frozen research tuple.

This research-only harness freezes candidate rows from an existing staged
pipeline report, then evaluates named later months without retraining,
reselecting thresholds, or changing exits. It consumes local report artifacts
and local event rows only. It does not fetch data, touch broker execution,
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

from stocker_research.personality_live_replay_v0 import _add_missing_discovery_features
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    WalkForwardSelectedFilterExitConfig,
    _concentration_warnings,
    _daily_pnl,
    _month_bounds,
    _personality_summary,
    _random_month_baseline,
)
from stocker_research.walk_forward_staged_mixed_regime_caveat_exit_v0 import (
    _apply_staged_monthly_candidates,
    _caveat_book_columns,
    _dedupe_staged_trades,
    _empty_caveat_book,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/pre_registered_edge_proof_v0")


@dataclass(frozen=True)
class PreRegisteredEdgeProofConfig:
    """Configuration for frozen-tuple forward evaluation."""

    registration_cutoff_month: str
    evaluation_months: tuple[str, ...]
    source_month: str | None = None
    personality: str = "active_liquidation"
    max_candidates: int = 1
    cost_bps: float = 10.0
    min_replay_signals: int = 1
    min_forward_trades: int = 15
    random_iterations: int = 1000
    random_seed: int = 1337


@dataclass(frozen=True)
class PreRegisteredEdgeProofResult:
    """Paths and headline result for one pre-registered proof run."""

    run_id: str
    input_event_dir: Path
    input_staged_report_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    registration_json_path: Path
    frozen_candidates_csv_path: Path
    frozen_caveats_csv_path: Path
    evaluation_monthly_summary_csv_path: Path
    evaluation_random_baseline_csv_path: Path
    evaluation_signals_csv_path: Path
    evaluation_caveated_signals_csv_path: Path
    evaluation_trades_csv_path: Path
    daily_pnl_csv_path: Path
    personality_summary_csv_path: Path
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


def _load_events(input_event_dir: Path) -> pd.DataFrame:
    path = input_event_dir / "event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event rows: {path}")
    events = cast(pd.DataFrame, _add_missing_discovery_features(pd.read_csv(path))).copy()
    events["_wf_timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    if "month" not in events:
        events["month"] = events["_wf_timestamp"].dt.strftime("%Y-%m")
    else:
        events["month"] = events["month"].astype(str)
    return events


def _load_required_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required staged report file: {path}")
    return pd.read_csv(path)


def _source_month(selected: pd.DataFrame, config: PreRegisteredEdgeProofConfig) -> str:
    if config.source_month:
        return config.source_month
    if selected.empty or "month" not in selected:
        return config.registration_cutoff_month
    months = sorted(
        {
            str(month)
            for month in selected["month"].dropna().astype(str)
            if str(month) <= config.registration_cutoff_month
        }
    )
    return months[-1] if months else config.registration_cutoff_month


def _freeze_candidates(
    selected: pd.DataFrame,
    *,
    config: PreRegisteredEdgeProofConfig,
) -> tuple[pd.DataFrame, str]:
    source_month = _source_month(selected, config)
    if selected.empty:
        frozen = selected.copy()
        frozen["source_month"] = pd.Series(dtype=str)
        return frozen, source_month
    frozen = selected[selected["month"].astype(str).eq(source_month)].copy()
    if config.personality:
        frozen = frozen[frozen["personality"].astype(str).eq(config.personality)].copy()
    if "monthly_candidate_rank" in frozen:
        frozen = frozen.sort_values(
            ["monthly_candidate_rank", "exit_selection_score"],
            ascending=[True, False],
            kind="mergesort",
        )
    frozen = frozen.head(int(config.max_candidates)).reset_index(drop=True)
    frozen.insert(0, "source_month", source_month)
    return frozen, source_month


def _freeze_caveats(input_staged_report_dir: Path, source_month: str) -> pd.DataFrame:
    path = input_staged_report_dir / "caveat_rule_book.csv"
    if not path.exists():
        return cast(pd.DataFrame, _empty_caveat_book())
    caveats = pd.read_csv(path)
    if caveats.empty or "month" not in caveats:
        return cast(pd.DataFrame, _empty_caveat_book())
    frozen = caveats[caveats["month"].astype(str).eq(source_month)].copy()
    if frozen.empty:
        return cast(pd.DataFrame, _empty_caveat_book())
    for column in _caveat_book_columns():
        if column not in frozen:
            frozen[column] = np.nan
    frozen["caveat_rule_id"] = np.arange(len(frozen), dtype=int)
    return cast(pd.DataFrame, frozen.loc[:, _caveat_book_columns()].reset_index(drop=True))


def _exit_config(config: PreRegisteredEdgeProofConfig) -> WalkForwardSelectedFilterExitConfig:
    return WalkForwardSelectedFilterExitConfig(
        replay_months=config.evaluation_months,
        cost_bps=config.cost_bps,
        min_replay_signals=config.min_replay_signals,
        min_total_trades=1,
        random_iterations=config.random_iterations,
        random_seed=config.random_seed,
    )


def _empty_monthly_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "month",
            "source_month",
            "replay_event_rows",
            "selected_candidate_count",
            "active_caveat_rule_count",
            "signal_count",
            "caveated_signal_count",
            "trade_count",
            "symbol_count",
            "session_count",
            "total_net_r",
            "median_net_r",
            "mean_net_r",
            "win_rate",
            "random_median_total_net_r",
            "excess_vs_random_total_net_r",
        ]
    )


def _aggregate_random(random_frame: pd.DataFrame, actual_total: float) -> dict[str, Any]:
    if random_frame.empty or "iteration" not in random_frame or "total_net_r" not in random_frame:
        return {
            "random_aggregate_iterations": 0,
            "random_aggregate_median_net_r": math.nan,
            "random_aggregate_p95_net_r": math.nan,
            "random_aggregate_actual_percentile": math.nan,
        }
    aggregate = random_frame.groupby("iteration", dropna=False)["total_net_r"].sum()
    return {
        "random_aggregate_iterations": int(len(aggregate)),
        "random_aggregate_median_net_r": float(aggregate.median()),
        "random_aggregate_p95_net_r": float(aggregate.quantile(0.95)),
        "random_aggregate_actual_percentile": float((aggregate < actual_total).mean()),
    }


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 24) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for column in view.columns:
            value = row[column]
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... {len(frame) - max_rows} more rows |" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def _decision(
    *,
    replay_event_count: int,
    trade_count: int,
    total_net_r: float,
    positive_month_count: int,
    random_p95: float,
    config: PreRegisteredEdgeProofConfig,
) -> tuple[str, list[str]]:
    if replay_event_count == 0:
        return "registered_waiting_for_future_data", ["no_evaluation_month_event_rows"]
    if trade_count < config.min_forward_trades:
        return "continue_research_insufficient_forward_sample", ["forward_trade_count_below_min"]
    if not math.isnan(random_p95) and total_net_r > random_p95 and positive_month_count > 0:
        return "forward_oos_supported_not_edge_proven", []
    return (
        "forward_oos_not_supported_research_only",
        ["random_p95_not_beaten_or_no_positive_months"],
    )


def _write_summary_md(path: Path, payload: dict[str, Any], monthly: pd.DataFrame) -> None:
    lines = [
        "# Pre-Registered Edge Proof V0",
        "",
        "Research-only frozen tuple forward audit.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Pipeline: `{payload['pipeline']}`",
        f"- Registration cutoff month: `{payload['registration_cutoff_month']}`",
        f"- Source month: `{payload['source_month']}`",
        f"- Evaluation months: `{', '.join(payload['evaluation_months'])}`",
        f"- Frozen candidate count: `{payload['frozen_candidate_count']}`",
        f"- Frozen caveat count: `{payload['frozen_caveat_count']}`",
        f"- Trade count: `{payload['trade_count']}`",
        f"- Total net R: `{payload['total_net_r']:.4f}`",
        f"- Win rate: `{payload['win_rate']:.4f}`"
        if not math.isnan(payload["win_rate"])
        else "- Win rate: `n/a`",
        f"- Edge claimed: `{payload['edge_claimed']}`",
        "",
        "## Monthly Summary",
        "",
        _markdown_table(monthly),
        "",
        "## Safety",
        "",
        "- `research_only: true`",
        "- `live_ordering_enabled: false`",
        "- `order_placement: disabled`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pre_registered_edge_proof(
    *,
    input_event_dir: Path,
    input_staged_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PreRegisteredEdgeProofConfig,
) -> PreRegisteredEdgeProofResult:
    """Register and evaluate a frozen staged tuple on later named months."""

    events = _load_events(input_event_dir)
    filter_book = _load_required_csv(input_staged_report_dir, "mixed_regime_filter_book.csv")
    selected = _load_required_csv(input_staged_report_dir, "selected_monthly_candidates.csv")
    frozen_candidates, source_month = _freeze_candidates(selected, config=config)
    frozen_caveats = _freeze_caveats(input_staged_report_dir, source_month)
    exit_config = _exit_config(config)

    all_signals: list[pd.DataFrame] = []
    all_caveated: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_random: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []

    for month_index, month in enumerate(config.evaluation_months):
        start, end = _month_bounds(month)
        replay_events = events[
            (events["_wf_timestamp"] >= start) & (events["_wf_timestamp"] < end)
        ].drop(columns=["_wf_timestamp"])
        month_candidates = frozen_candidates.copy()
        if not month_candidates.empty:
            month_candidates["month"] = month
        signals, caveated = _apply_staged_monthly_candidates(
            month_candidates,
            filter_book,
            replay_events,
            frozen_caveats,
            config=exit_config,
        )
        trades, _missed = _dedupe_staged_trades(signals)
        random_baseline = _random_month_baseline(
            replay_events,
            trades,
            config=exit_config,
            seed=config.random_seed + month_index * 1009,
        )
        if not random_baseline.empty:
            random_baseline["month"] = month
            all_random.append(random_baseline)
        if not signals.empty:
            all_signals.append(signals)
        if not caveated.empty:
            all_caveated.append(caveated)
        if not trades.empty:
            all_trades.append(trades)
        random_total = (
            float(random_baseline["total_net_r"].median())
            if not random_baseline.empty
            else math.nan
        )
        net_r = pd.to_numeric(trades.get("net_r", pd.Series(dtype=float)), errors="coerce")
        month_rows.append(
            {
                "month": month,
                "source_month": source_month,
                "replay_event_rows": int(len(replay_events)),
                "selected_candidate_count": int(len(month_candidates)),
                "active_caveat_rule_count": int(len(frozen_caveats)),
                "signal_count": int(len(signals)),
                "caveated_signal_count": int(len(caveated)),
                "trade_count": int(len(trades)),
                "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
                "session_count": int(trades["session_date"].nunique())
                if not trades.empty
                else 0,
                "total_net_r": float(net_r.sum()) if not trades.empty else 0.0,
                "median_net_r": float(net_r.median()) if not trades.empty else math.nan,
                "mean_net_r": float(net_r.mean()) if not trades.empty else math.nan,
                "win_rate": float((net_r > 0.0).mean()) if not trades.empty else math.nan,
                "random_median_total_net_r": random_total,
                "excess_vs_random_total_net_r": float(net_r.sum()) - random_total
                if not math.isnan(random_total)
                else math.nan,
            }
        )

    signals_frame = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    caveated_frame = (
        pd.concat(all_caveated, ignore_index=True) if all_caveated else pd.DataFrame()
    )
    trades_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    random_frame = pd.concat(all_random, ignore_index=True) if all_random else pd.DataFrame()
    monthly = pd.DataFrame(month_rows) if month_rows else _empty_monthly_summary()
    daily = _daily_pnl(trades_frame)
    personality = _personality_summary(trades_frame)
    concentration = _concentration_warnings(trades_frame, exit_config)

    total_net_r = (
        float(pd.to_numeric(trades_frame["net_r"], errors="coerce").sum())
        if not trades_frame.empty
        else 0.0
    )
    win_rate = (
        float((pd.to_numeric(trades_frame["net_r"], errors="coerce") > 0.0).mean())
        if not trades_frame.empty
        else math.nan
    )
    replay_event_count = (
        int(monthly["replay_event_rows"].sum()) if "replay_event_rows" in monthly else 0
    )
    positive_month_count = (
        int((pd.to_numeric(monthly["total_net_r"], errors="coerce") > 0.0).sum())
        if "total_net_r" in monthly
        else 0
    )
    random_summary = _aggregate_random(random_frame, total_net_r)
    decision, decision_reasons = _decision(
        replay_event_count=replay_event_count,
        trade_count=int(len(trades_frame)),
        total_net_r=total_net_r,
        positive_month_count=positive_month_count,
        random_p95=float(random_summary["random_aggregate_p95_net_r"]),
        config=config,
    )

    run_id = f"pre_registered_edge_proof_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "registration_json": run_dir / "registration.json",
        "frozen_candidates": run_dir / "frozen_candidates.csv",
        "frozen_caveats": run_dir / "frozen_caveats.csv",
        "monthly": run_dir / "evaluation_monthly_summary.csv",
        "random": run_dir / "evaluation_random_baseline.csv",
        "signals": run_dir / "evaluation_signals.csv",
        "caveated": run_dir / "evaluation_caveated_signals.csv",
        "trades": run_dir / "evaluation_trades.csv",
        "daily": run_dir / "daily_pnl.csv",
        "personality": run_dir / "personality_summary.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }

    registration_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "pipeline": "personality -> mixed_regime -> filter -> caveat -> exit",
        "input_staged_report_dir": str(input_staged_report_dir),
        "registration_cutoff_month": config.registration_cutoff_month,
        "source_month": source_month,
        "evaluation_months": list(config.evaluation_months),
        "personality": config.personality,
        "frozen_candidate_count": int(len(frozen_candidates)),
        "frozen_caveat_count": int(len(frozen_caveats)),
        "registration_time_utc": datetime.now(UTC).isoformat(),
        "immutability_note": (
            "Future evaluations must reuse frozen_candidates.csv and frozen_caveats.csv "
            "without changing thresholds, exits, symbols, or caveat definitions."
        ),
    }
    payload = {
        **registration_payload,
        "input_event_dir": str(input_event_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "data_source": (
            "existing local state_event_detector event_rows.csv plus existing staged "
            "research report artifacts; no vendor fetch"
        ),
        "volume_label": "historical_volume from existing local 5m OHLCV event report",
        "trade_count": int(len(trades_frame)),
        "total_net_r": total_net_r,
        "win_rate": win_rate,
        "positive_month_count": positive_month_count,
        "evaluation_month_count": int(len(config.evaluation_months)),
        **random_summary,
    }

    for path, frame in [
        (paths["frozen_candidates"], frozen_candidates),
        (paths["frozen_caveats"], frozen_caveats),
        (paths["monthly"], monthly),
        (paths["random"], random_frame),
        (paths["signals"], signals_frame),
        (paths["caveated"], caveated_frame),
        (paths["trades"], trades_frame),
        (paths["daily"], daily),
        (paths["personality"], personality),
        (paths["concentration"], concentration),
    ]:
        _write_csv(path, frame)
    _write_json(paths["registration_json"], registration_payload)
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "decision_reasons": decision_reasons,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
        },
    )
    _write_summary_md(paths["summary_md"], payload, monthly)

    return PreRegisteredEdgeProofResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        input_staged_report_dir=input_staged_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        registration_json_path=paths["registration_json"],
        frozen_candidates_csv_path=paths["frozen_candidates"],
        frozen_caveats_csv_path=paths["frozen_caveats"],
        evaluation_monthly_summary_csv_path=paths["monthly"],
        evaluation_random_baseline_csv_path=paths["random"],
        evaluation_signals_csv_path=paths["signals"],
        evaluation_caveated_signals_csv_path=paths["caveated"],
        evaluation_trades_csv_path=paths["trades"],
        daily_pnl_csv_path=paths["daily"],
        personality_summary_csv_path=paths["personality"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        trade_count=int(len(trades_frame)),
    )
