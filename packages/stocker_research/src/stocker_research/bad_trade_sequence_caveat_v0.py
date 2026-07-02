"""Reusable bad-trade sequence caveat research lab.

This module consumes an existing selected-filter/exit report and sparse
state-event rows. It is research-only: it reads local report files, writes
diagnostic report artifacts, and never touches broker, paper, live, execution,
order-placement, or vendor-fetching paths.
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

from stocker_research.personality_discovery_v0 import EVENT_STATE_PERSONALITY

DEFAULT_OUTPUT_DIR = Path("data/reports/research/bad_trade_sequence_caveat_v0")

STRICT_TRAIN_AND_OOS_SUPPORTED = "strict_train_and_oos_supported"
OOS_ONLY_NOT_TRAIN_SUPPORTED = "oos_only_not_train_supported"
TRAIN_ONLY_NOT_OOS_SUPPORTED = "train_only_not_oos_supported"
NOT_SUPPORTED = "not_supported"


@dataclass(frozen=True)
class BadTradeSequenceCaveatConfig:
    """Configuration for reusable bad-trade caveat diagnostics."""

    train_months: tuple[str, ...] = ("2026-01", "2026-02", "2026-03", "2026-04")
    test_months: tuple[str, ...] = ("2026-05", "2026-06")
    numeric_quantiles: tuple[float, ...] = (0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80)
    random_iterations: int = 3000
    random_seed: int = 1337
    min_candidate_count: int = 1
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20


@dataclass(frozen=True)
class BadTradeSequenceCaveatResult:
    """Paths and headline result for one caveat run."""

    run_id: str
    input_selected_report_dir: Path
    input_event_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    sequence_caveat_results_csv_path: Path
    current_personality_caveats_csv_path: Path
    prior_sequence_caveats_csv_path: Path
    numeric_threshold_caveats_csv_path: Path
    strict_validation_results_csv_path: Path
    trade_caveat_flags_csv_path: Path
    decision: str
    caveat_count: int


@dataclass(frozen=True)
class _Candidate:
    rule_name: str
    rule_family: str
    notes: str
    mask: pd.Series
    train_selected: bool = True
    selected_train_quantile: float | None = None
    selected_threshold: float | None = None
    feature: str | None = None
    operator: str | None = None
    current_personality: str | None = None
    prior_personality: str | None = None
    prior2_personality: str | None = None


@dataclass(frozen=True)
class _NumericRuleSpec:
    feature: str
    operator: str
    label: str


_NUMERIC_RULES: tuple[_NumericRuleSpec, ...] = (
    _NumericRuleSpec("close_location_value", "<=", "weak close"),
    _NumericRuleSpec("return_zscore", "<=", "weak current return zscore"),
    _NumericRuleSpec("bar_return", "<=", "weak current bar return"),
    _NumericRuleSpec("prior_3_bar_return", ">=", "already extended over prior 3 bars"),
    _NumericRuleSpec("prior_6_bar_return", ">=", "already extended over prior 6 bars"),
    _NumericRuleSpec("prior_12_bar_return", ">=", "already extended over prior 12 bars"),
    _NumericRuleSpec("risk_bps", "<=", "tight risk"),
    _NumericRuleSpec(
        "relative_volume_at_bar_index",
        ">=",
        "high historical relative volume at bar index",
    ),
    _NumericRuleSpec(
        "relative_cumulative_volume",
        ">=",
        "high historical cumulative relative volume",
    ),
    _NumericRuleSpec(
        "same_direction_other_symbol_count_15m",
        "<=",
        "weak cross-stock same-direction confirmation",
    ),
    _NumericRuleSpec(
        "same_personality_other_symbol_count_15m",
        "<=",
        "weak cross-stock same-personality confirmation",
    ),
    _NumericRuleSpec(
        "same_direction_other_symbol_count_30m",
        "<=",
        "weak 30m cross-stock same-direction confirmation",
    ),
    _NumericRuleSpec(
        "same_personality_other_symbol_count_30m",
        "<=",
        "weak 30m cross-stock same-personality confirmation",
    ),
)


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


def _event_personality(event_state: object) -> str:
    details = EVENT_STATE_PERSONALITY.get(str(event_state))
    if details is None:
        return "unknown_personality"
    return str(details[0])


def _coerce_personality_columns(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    if "personality" not in data:
        data["personality"] = data["event_state"].map(_event_personality)
    else:
        mapped = data["event_state"].map(_event_personality) if "event_state" in data else ""
        data["personality"] = data["personality"].where(data["personality"].notna(), mapped)
    data["personality"] = data["personality"].fillna("unknown_personality").astype(str)
    return data


def _load_trades(input_selected_report_dir: Path) -> pd.DataFrame:
    path = input_selected_report_dir / "trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-filter final trades: {path}")
    data = pd.read_csv(path)
    required = {"symbol", "timestamp", "session_date", "event_state", "net_r"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"trades.csv missing required columns: {missing}")
    data = _coerce_personality_columns(data)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data[data["timestamp"].notna()].copy()
    if "month" not in data:
        data["month"] = data["timestamp"].dt.strftime("%Y-%m")
    else:
        data["month"] = data["month"].astype(str)
    data["net_r"] = pd.to_numeric(data["net_r"], errors="coerce")
    data = data[data["net_r"].notna()].reset_index(drop=True)
    return data


def _load_events(input_event_dir: Path) -> pd.DataFrame:
    path = input_event_dir / "event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sparse event rows: {path}")
    data = pd.read_csv(path)
    required = {"symbol", "timestamp", "session_date", "event_state"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"event_rows.csv missing required columns: {missing}")
    data = _coerce_personality_columns(data)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data[data["timestamp"].notna()].copy()
    if "bar_index_in_session" not in data:
        data["bar_index_in_session"] = np.nan
    data["bar_index_in_session"] = pd.to_numeric(
        data["bar_index_in_session"],
        errors="coerce",
    )
    return data.reset_index(drop=True)


def attach_prior_event_context(trades: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the prior one and two sparse events from the same symbol/session.

    Matching uses only events from earlier bars. Equal timestamps only qualify
    when the event bar index is strictly lower than the trade bar index.
    """

    trade_data = _coerce_personality_columns(trades)
    event_data = _coerce_personality_columns(event_rows)
    trade_data = trade_data.copy()
    event_data = event_data.copy()
    trade_data["timestamp"] = pd.to_datetime(trade_data["timestamp"], utc=True, errors="coerce")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"], utc=True, errors="coerce")
    if "bar_index_in_session" not in trade_data:
        trade_data["bar_index_in_session"] = np.nan
    if "bar_index_in_session" not in event_data:
        event_data["bar_index_in_session"] = np.nan
    trade_data["bar_index_in_session"] = pd.to_numeric(
        trade_data["bar_index_in_session"],
        errors="coerce",
    )
    event_data["bar_index_in_session"] = pd.to_numeric(
        event_data["bar_index_in_session"],
        errors="coerce",
    )

    grouped_events: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in event_data.groupby(["symbol", "session_date"], dropna=False):
        sorted_group = group.sort_values(
            ["timestamp", "bar_index_in_session"],
            kind="mergesort",
        ).reset_index(drop=True)
        grouped_events[(str(key[0]), str(key[1]))] = sorted_group

    prior_payload: list[dict[str, object]] = []
    for _, trade in trade_data.iterrows():
        key = (str(trade["symbol"]), str(trade["session_date"]))
        candidates = grouped_events.get(key)
        trade_timestamp = trade["timestamp"]
        trade_bar = float(trade.get("bar_index_in_session", math.nan))
        selected = pd.DataFrame()
        if candidates is not None and pd.notna(trade_timestamp):
            earlier_timestamp = candidates["timestamp"] < trade_timestamp
            if math.isnan(trade_bar):
                earlier_bar = pd.Series(False, index=candidates.index)
            else:
                earlier_bar = candidates["timestamp"].eq(trade_timestamp) & (
                    candidates["bar_index_in_session"] < trade_bar
                )
            selected = candidates[earlier_timestamp | earlier_bar].tail(2)
        prior_payload.append(_prior_payload(selected, trade_bar))
    prior_frame = pd.DataFrame(prior_payload, index=trade_data.index)
    trade_data = trade_data.drop(
        columns=[column for column in prior_frame.columns if column in trade_data.columns],
    )
    return pd.concat(
        [trade_data.reset_index(drop=True), prior_frame.reset_index(drop=True)],
        axis=1,
    )


def _prior_payload(selected: pd.DataFrame, trade_bar: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "prev_event_personality": math.nan,
        "prev_event_state": math.nan,
        "prev_event_timestamp": math.nan,
        "prev_event_bar_index_in_session": math.nan,
        "prev_event_close_location_value": math.nan,
        "prev_event_distance_from_vwap_pct": math.nan,
        "bars_since_prev_event": math.nan,
        "prev2_event_personality": math.nan,
        "prev2_event_state": math.nan,
        "prev2_event_timestamp": math.nan,
        "prev2_event_bar_index_in_session": math.nan,
        "bars_since_prev2_event": math.nan,
    }
    if selected.empty:
        return payload
    previous = selected.iloc[-1]
    _fill_prior_payload(payload, previous, prefix="prev", trade_bar=trade_bar)
    if len(selected) > 1:
        previous_two = selected.iloc[-2]
        _fill_prior_payload(payload, previous_two, prefix="prev2", trade_bar=trade_bar)
    return payload


def _fill_prior_payload(
    payload: dict[str, object],
    event: pd.Series,
    *,
    prefix: str,
    trade_bar: float,
) -> None:
    payload[f"{prefix}_event_personality"] = str(event.get("personality", "unknown_personality"))
    payload[f"{prefix}_event_state"] = str(event.get("event_state", "unknown_event_state"))
    timestamp = event.get("timestamp")
    payload[f"{prefix}_event_timestamp"] = (
        timestamp.isoformat() if pd.notna(timestamp) else math.nan
    )
    event_bar = float(event.get("bar_index_in_session", math.nan))
    payload[f"{prefix}_event_bar_index_in_session"] = event_bar
    if prefix == "prev":
        payload["prev_event_close_location_value"] = event.get("close_location_value", math.nan)
        payload["prev_event_distance_from_vwap_pct"] = event.get(
            "distance_from_vwap_pct",
            math.nan,
        )
    if not math.isnan(trade_bar) and not math.isnan(event_bar):
        payload[f"bars_since_{prefix}_event"] = trade_bar - event_bar


def _mask_for_candidate(mask: pd.Series, rows: pd.DataFrame) -> pd.Series:
    return mask.reindex(rows.index, fill_value=False).fillna(False).astype(bool)


def _numeric_mask(rows: pd.DataFrame, feature: str, operator: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(rows[feature], errors="coerce")
    if operator == "<=":
        return (values <= threshold).fillna(False)
    if operator == ">=":
        return (values >= threshold).fillna(False)
    raise ValueError(f"Unsupported numeric operator: {operator}")


def _candidate_thresholds(
    train: pd.DataFrame,
    feature: str,
    quantiles: tuple[float, ...],
) -> list[tuple[float, float]]:
    values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.empty:
        return []
    thresholds: dict[float, float] = {}
    for quantile in quantiles:
        threshold = float(values.quantile(quantile))
        thresholds[threshold] = float(quantile)
    return [(threshold, thresholds[threshold]) for threshold in sorted(thresholds)]


def _summary_stats(rows: pd.DataFrame) -> dict[str, int | float]:
    if rows.empty:
        return {
            "count": 0,
            "loss_count": 0,
            "loss_rate": math.nan,
            "total_net_r": 0.0,
            "median_net_r": math.nan,
            "win_rate": math.nan,
            **_concentration(rows),
        }
    net_r = pd.to_numeric(rows["net_r"], errors="coerce")
    return {
        "count": int(len(rows)),
        "loss_count": int((net_r < 0.0).sum()),
        "loss_rate": float((net_r < 0.0).mean()),
        "total_net_r": float(net_r.sum()),
        "median_net_r": float(net_r.median()),
        "win_rate": float((net_r > 0.0).mean()),
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
    symbol_counts = rows["symbol"].astype(str).value_counts()
    session_counts = rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1)
    session_counts = session_counts.value_counts()
    month_counts = rows["month"].astype(str).value_counts() if "month" in rows else pd.Series()
    return {
        "symbol_count": int(symbol_counts.size),
        "session_count": int(session_counts.size),
        "month_count": int(month_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "single_month_share": float(month_counts.iloc[0] / len(rows))
        if not month_counts.empty
        else math.nan,
    }


def _random_kept_baseline(
    test_rows: pd.DataFrame,
    flagged_count: int,
    *,
    config: BadTradeSequenceCaveatConfig,
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


def _evaluate_candidate(
    rows: pd.DataFrame,
    candidate: _Candidate,
    *,
    config: BadTradeSequenceCaveatConfig,
) -> dict[str, object]:
    mask = _mask_for_candidate(candidate.mask, rows)
    train_mask = rows["month"].astype(str).isin(config.train_months)
    test_mask = rows["month"].astype(str).isin(config.test_months)
    base_all = rows
    base_train = rows[train_mask]
    base_test = rows[test_mask]
    flagged_all = rows[mask]
    kept_all = rows[~mask]
    flagged_train = rows[train_mask & mask]
    kept_train = rows[train_mask & ~mask]
    flagged_test = rows[test_mask & mask]
    kept_test = rows[test_mask & ~mask]

    all_stats = _summary_stats(base_all)
    train_stats = _summary_stats(base_train)
    test_stats = _summary_stats(base_test)
    test_concentration = _concentration(flagged_test)
    random_baseline = _random_kept_baseline(
        base_test,
        len(flagged_test),
        config=config,
        rule_name=candidate.rule_name,
    )
    train_flagged_total = float(flagged_train["net_r"].sum()) if not flagged_train.empty else 0.0
    test_flagged_total = float(flagged_test["net_r"].sum()) if not flagged_test.empty else 0.0
    train_kept_total = float(kept_train["net_r"].sum()) if not kept_train.empty else 0.0
    test_kept_total = float(kept_test["net_r"].sum()) if not kept_test.empty else 0.0
    train_kept_lift = train_kept_total - float(train_stats["total_net_r"])
    test_kept_lift = test_kept_total - float(test_stats["total_net_r"])
    test_excess = test_kept_total - float(random_baseline["random_kept_net_r_median"])
    concentration_warning = _has_concentration_warning(test_concentration, config=config)
    strict_train_supported = (
        len(flagged_train) > 0 and train_flagged_total < 0.0 and train_kept_lift > 0.0
    )
    strict_oos_supported = (
        len(flagged_test) > 0
        and test_flagged_total < 0.0
        and test_kept_lift > 0.0
        and test_excess > 0.0
        and not concentration_warning
    )
    strict_status = _strict_status(strict_train_supported, strict_oos_supported)
    return {
        "rule_name": candidate.rule_name,
        "rule_family": candidate.rule_family,
        "train_selected": bool(candidate.train_selected),
        "notes": candidate.notes,
        "feature": candidate.feature,
        "operator": candidate.operator,
        "current_personality": candidate.current_personality,
        "prior_personality": candidate.prior_personality,
        "prior2_personality": candidate.prior2_personality,
        "all_count": int(all_stats["count"]),
        "all_total_net_r": float(all_stats["total_net_r"]),
        "all_loss_rate": all_stats["loss_rate"],
        "flagged_count": int(len(flagged_all)),
        "flagged_total_net_r": float(flagged_all["net_r"].sum()) if not flagged_all.empty else 0.0,
        "flagged_loss_rate": _loss_rate(flagged_all),
        "kept_count": int(len(kept_all)),
        "kept_total_net_r": float(kept_all["net_r"].sum()) if not kept_all.empty else 0.0,
        "kept_lift_vs_base_r": float(kept_all["net_r"].sum()) - float(all_stats["total_net_r"])
        if not kept_all.empty
        else -float(all_stats["total_net_r"]),
        "train_base_count": int(train_stats["count"]),
        "train_base_total_net_r": float(train_stats["total_net_r"]),
        "train_flagged_count": int(len(flagged_train)),
        "train_flagged_total_net_r": train_flagged_total,
        "train_flagged_loss_rate": _loss_rate(flagged_train),
        "train_kept_count": int(len(kept_train)),
        "train_kept_total_net_r": train_kept_total,
        "train_kept_lift_vs_base_r": train_kept_lift,
        "test_base_count": int(test_stats["count"]),
        "test_base_total_net_r": float(test_stats["total_net_r"]),
        "test_flagged_count": int(len(flagged_test)),
        "test_flagged_total_net_r": test_flagged_total,
        "test_flagged_loss_rate": _loss_rate(flagged_test),
        "test_kept_count": int(len(kept_test)),
        "test_kept_total_net_r": test_kept_total,
        "test_kept_lift_vs_base_r": test_kept_lift,
        "test_symbol_count": test_concentration["symbol_count"],
        "test_session_count": test_concentration["session_count"],
        "test_month_count": test_concentration["month_count"],
        "test_single_symbol_share": test_concentration["single_symbol_share"],
        "test_single_session_share": test_concentration["single_session_share"],
        "test_single_month_share": test_concentration["single_month_share"],
        **random_baseline,
        "test_excess_vs_random_median_r": test_excess,
        "oos_beats_base": bool(test_kept_lift > 0.0),
        "oos_beats_random": bool(test_excess > 0.0),
        "concentration_warning": bool(concentration_warning),
        "selected_train_quantile": candidate.selected_train_quantile,
        "selected_threshold": candidate.selected_threshold,
        "strict_status": strict_status,
        "strict_train_supported": bool(strict_train_supported),
        "strict_oos_supported": bool(strict_oos_supported),
    }


def _loss_rate(rows: pd.DataFrame) -> float:
    if rows.empty:
        return math.nan
    return float((pd.to_numeric(rows["net_r"], errors="coerce") < 0.0).mean())


def _has_concentration_warning(
    concentration: dict[str, int | float],
    *,
    config: BadTradeSequenceCaveatConfig,
) -> bool:
    symbol_share = float(concentration["single_symbol_share"])
    session_share = float(concentration["single_session_share"])
    if math.isnan(symbol_share) or math.isnan(session_share):
        return False
    return (
        symbol_share >= config.max_single_symbol_share
        or session_share >= config.max_single_session_share
    )


def _strict_status(train_supported: bool, oos_supported: bool) -> str:
    if train_supported and oos_supported:
        return STRICT_TRAIN_AND_OOS_SUPPORTED
    if oos_supported:
        return OOS_ONLY_NOT_TRAIN_SUPPORTED
    if train_supported:
        return TRAIN_ONLY_NOT_OOS_SUPPORTED
    return NOT_SUPPORTED


def _build_current_personality_candidates(
    rows: pd.DataFrame,
    *,
    config: BadTradeSequenceCaveatConfig,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for personality in sorted(rows["personality"].dropna().astype(str).unique()):
        mask = rows["personality"].astype(str).eq(personality)
        if int(mask.sum()) < config.min_candidate_count:
            continue
        candidates.append(
            _Candidate(
                rule_name=f"current {personality}",
                rule_family="fixed_personality_block",
                notes="Block or sideline the current detected personality.",
                mask=mask,
                current_personality=personality,
            )
        )
    return candidates


def _build_current_group_candidates(
    rows: pd.DataFrame,
    *,
    config: BadTradeSequenceCaveatConfig,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    candidates.extend(
        _group_candidates(rows, ("personality", "regime_field", "regime_value"), config)
    )
    candidates.extend(_group_candidates(rows, ("personality", "filter_rule"), config))
    candidates.extend(_group_candidates(rows, ("personality", "stop_model"), config))
    candidates.extend(_group_candidates(rows, ("personality", "filter_rule", "stop_model"), config))
    return candidates


def _group_candidates(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
    config: BadTradeSequenceCaveatConfig,
) -> list[_Candidate]:
    if not set(columns).issubset(rows.columns):
        return []
    candidates: list[_Candidate] = []
    grouped = rows.dropna(subset=list(columns)).groupby(list(columns), dropna=False)
    for values, group in grouped:
        value_tuple = values if isinstance(values, tuple) else (values,)
        if len(group) < config.min_candidate_count:
            continue
        mask = pd.Series(False, index=rows.index)
        mask.loc[group.index] = True
        personality = str(value_tuple[0])
        descriptor = _group_descriptor(columns, value_tuple)
        candidates.append(
            _Candidate(
                rule_name=f"{personality} + {descriptor}",
                rule_family="fixed_current_group",
                notes="Current personality grouped by regime, filter, or exit settings.",
                mask=mask,
                current_personality=personality,
            )
        )
    return candidates


def _group_descriptor(columns: tuple[str, ...], values: tuple[object, ...]) -> str:
    if columns == ("personality", "regime_field", "regime_value"):
        return f"{values[1]}={values[2]}"
    if columns == ("personality", "filter_rule"):
        return f"{values[1]} filter"
    if columns == ("personality", "stop_model"):
        return f"{values[1]} exit"
    if columns == ("personality", "filter_rule", "stop_model"):
        return f"{values[1]} filter + {values[2]} exit"
    return " + ".join(str(value) for value in values[1:])


def _build_prior_sequence_candidates(
    rows: pd.DataFrame,
    *,
    config: BadTradeSequenceCaveatConfig,
) -> tuple[list[_Candidate], list[_Candidate]]:
    prior_one: list[_Candidate] = []
    prior_two: list[_Candidate] = []
    if {"prev_event_personality", "personality"}.issubset(rows.columns):
        data = rows[rows["prev_event_personality"].notna()].copy()
        for (prior, current), group in data.groupby(["prev_event_personality", "personality"]):
            if len(group) < config.min_candidate_count:
                continue
            mask = pd.Series(False, index=rows.index)
            mask.loc[group.index] = True
            prior_one.append(
                _Candidate(
                    rule_name=f"prior {prior} -> current {current}",
                    rule_family="fixed_prior_personality_sequence",
                    notes="Prior detected personality followed by current personality.",
                    mask=mask,
                    current_personality=str(current),
                    prior_personality=str(prior),
                )
            )
    if {"prev2_event_personality", "prev_event_personality", "personality"}.issubset(rows.columns):
        data = rows[
            rows["prev2_event_personality"].notna() & rows["prev_event_personality"].notna()
        ].copy()
        grouped = data.groupby(["prev2_event_personality", "prev_event_personality", "personality"])
        for (prior2, prior, current), group in grouped:
            if len(group) < config.min_candidate_count:
                continue
            mask = pd.Series(False, index=rows.index)
            mask.loc[group.index] = True
            prior_two.append(
                _Candidate(
                    rule_name=f"prior {prior2} -> prior {prior} -> current {current}",
                    rule_family="fixed_prior_two_personality_sequence",
                    notes="Prior two-personality sequence followed by current personality.",
                    mask=mask,
                    current_personality=str(current),
                    prior_personality=str(prior),
                    prior2_personality=str(prior2),
                )
            )
    return prior_one, prior_two


def _build_composite_sequence_candidates(
    current_candidates: list[_Candidate],
    prior_one_candidates: list[_Candidate],
    *,
    config: BadTradeSequenceCaveatConfig,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for current in current_candidates:
        if current.current_personality is None:
            continue
        for prior in prior_one_candidates:
            if prior.prior_personality is None or prior.current_personality is None:
                continue
            mask = current.mask | prior.mask
            if int(mask.sum()) < config.min_candidate_count:
                continue
            candidates.append(
                _Candidate(
                    rule_name=(
                        "sequence_caveat: "
                        f"{current.current_personality} OR "
                        f"{prior.prior_personality}->{prior.current_personality}"
                    ),
                    rule_family="composite_sequence",
                    notes="Union of a current-personality blocker and a prior sequence caveat.",
                    mask=mask,
                    current_personality=(
                        f"{current.current_personality}|{prior.current_personality}"
                    ),
                    prior_personality=prior.prior_personality,
                )
            )
    return candidates


def _build_numeric_threshold_candidates(
    rows: pd.DataFrame,
    *,
    config: BadTradeSequenceCaveatConfig,
) -> list[_Candidate]:
    train = rows[rows["month"].astype(str).isin(config.train_months)]
    candidates: list[_Candidate] = []
    for spec in _NUMERIC_RULES:
        if spec.feature not in rows.columns or spec.feature not in train.columns:
            continue
        for threshold, quantile in _candidate_thresholds(
            train,
            spec.feature,
            config.numeric_quantiles,
        ):
            mask = _numeric_mask(rows, spec.feature, spec.operator, threshold)
            if int(mask.sum()) < config.min_candidate_count:
                continue
            candidates.append(
                _Candidate(
                    rule_name=f"{spec.label}: {spec.feature} {spec.operator} {threshold:.6g}",
                    rule_family="train_selected_numeric",
                    notes="Threshold candidate generated from train quantiles only.",
                    mask=mask,
                    train_selected=False,
                    selected_train_quantile=quantile,
                    selected_threshold=threshold,
                    feature=spec.feature,
                    operator=spec.operator,
                )
            )
    return candidates


def _evaluate_candidates(
    rows: pd.DataFrame,
    candidates: list[_Candidate],
    *,
    config: BadTradeSequenceCaveatConfig,
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    result = pd.DataFrame(
        [_evaluate_candidate(rows, candidate, config=config) for candidate in candidates]
    )
    return _sort_results(result)


def _mark_train_selected_numeric(numeric_results: pd.DataFrame) -> pd.DataFrame:
    if numeric_results.empty:
        return numeric_results
    data = numeric_results.copy()
    data["train_selected"] = False
    grouped = data.groupby(["feature", "operator"], dropna=False)
    for _, group in grouped:
        selected_index = group.sort_values(
            ["train_flagged_total_net_r", "train_flagged_count", "selected_threshold"],
            ascending=[True, False, True],
            kind="mergesort",
        ).index[0]
        data.loc[selected_index, "train_selected"] = True
    return _sort_results(data)


def _sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in [
            "strict_oos_supported",
            "test_kept_lift_vs_base_r",
            "test_excess_vs_random_median_r",
            "train_kept_lift_vs_base_r",
            "flagged_count",
        ]
        if column in frame
    ]
    ascending = [False, False, False, False, False][: len(sort_columns)]
    return frame.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(
        drop=True
    )


def _concat_result_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    columns: list[str] = []
    normalized: list[pd.DataFrame] = []
    for frame in non_empty:
        trimmed = frame.dropna(axis=1, how="all")
        normalized.append(trimmed)
        for column in trimmed.columns:
            if column not in columns:
                columns.append(column)
    return pd.concat(normalized, ignore_index=True).reindex(columns=columns)


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug[:96] or "rule"


def _build_trade_flags(
    rows: pd.DataFrame,
    candidates: list[_Candidate],
) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "symbol",
            "timestamp",
            "session_date",
            "month",
            "personality",
            "event_state",
            "net_r",
            "exit_reason",
            "bar_index_in_session",
            "time_x_vwap_regime",
            "vwap_x_range_regime",
            "compression_x_efficiency_regime",
            "filter_rule",
            "stop_model",
            "risk_bps",
            "close_location_value",
            "return_zscore",
            "same_direction_other_symbol_count_15m",
            "relative_volume_at_bar_index",
            "prev2_event_personality",
            "prev2_event_state",
            "prev_event_personality",
            "prev_event_state",
            "bars_since_prev_event",
        ]
        if column in rows.columns
    ]
    flags = rows[columns].copy()
    used_names: set[str] = set()
    flag_columns: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {}
    for candidate in candidates:
        slug = _slug(candidate.rule_name)
        column = f"flag_{slug}"
        suffix = 2
        while column in used_names:
            column = f"flag_{slug}_{suffix}"
            suffix += 1
        used_names.add(column)
        flag_columns[column] = _mask_for_candidate(candidate.mask, rows).to_numpy()
    if not flag_columns:
        return flags
    flag_frame = pd.DataFrame(flag_columns, index=rows.index)
    return pd.concat([flags, flag_frame], axis=1)


def _decision(strict_results: pd.DataFrame) -> tuple[str, list[str]]:
    if strict_results.empty:
        return "continue_research_no_sequence_caveat_supported", ["no_caveats_tested"]
    strict_count = int(strict_results["strict_status"].eq(STRICT_TRAIN_AND_OOS_SUPPORTED).sum())
    oos_only_count = int(strict_results["strict_status"].eq(OOS_ONLY_NOT_TRAIN_SUPPORTED).sum())
    train_only_count = int(strict_results["strict_status"].eq(TRAIN_ONLY_NOT_OOS_SUPPORTED).sum())
    if strict_count > 0:
        return "continue_research_strict_train_and_oos_supported", []
    if oos_only_count > 0:
        return (
            "continue_research_oos_warning_not_train_validated",
            ["one or more caveats worked in held-out test but were not supported by train"],
        )
    if train_only_count > 0:
        return (
            "continue_research_train_warning_not_oos_validated",
            ["one or more caveats were supported by train but not by held-out test"],
        )
    return "continue_research_no_sequence_caveat_supported", ["no strict caveat support"]


def _best_oos_rule(strict_results: pd.DataFrame) -> dict[str, Any] | None:
    if strict_results.empty:
        return None
    supported = strict_results[strict_results["strict_oos_supported"].astype(bool)].copy()
    source = supported if not supported.empty else strict_results
    row = source.sort_values(
        ["test_kept_lift_vs_base_r", "test_excess_vs_random_median_r"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    return {str(key): _json_safe(value) for key, value in row.to_dict().items()}


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
            "rule_family",
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
    best = payload.get("best_oos_rule") or {}
    lines = [
        "# Bad Trade Sequence Caveat V0",
        "",
        (
            "Research-only caveat test for the selected regime + personality + filter + "
            "exit system. No broker code, no order placement, no vendor fetching, no "
            "edge claim."
        ),
        "",
        "## Input",
        "",
        f"- Selected report: `{payload['input_selected_report_dir']}`",
        f"- Event report: `{payload['input_event_dir']}`",
        "- Data source: existing local report outputs and sparse local 5m OHLCV-derived events",
        f"- Train months: `{', '.join(payload['train_months'])}`",
        f"- Test months: `{', '.join(payload['test_months'])}`",
        "",
        "## Base",
        "",
        (
            f"- All trades: {payload['base_all']['count']}, total: "
            f"{payload['base_all']['total_net_r']:.4f}R"
        ),
        (
            f"- Train: {payload['base_train']['count']} trades, "
            f"{payload['base_train']['total_net_r']:.4f}R"
        ),
        (
            f"- Test: {payload['base_test']['count']} trades, "
            f"{payload['base_test']['total_net_r']:.4f}R"
        ),
        "",
        "## Best OOS Warning",
        "",
        f"- Rule: `{best.get('rule_name', 'none')}`",
        f"- Strict status: `{best.get('strict_status', 'none')}`",
        f"- Test kept lift: `{best.get('test_kept_lift_vs_base_r', math.nan):.4f}R`"
        if isinstance(best.get("test_kept_lift_vs_base_r"), float)
        else "- Test kept lift: `n/a`",
        "",
        "## Strict Validation Results",
        "",
        _markdown_table(strict_results[columns] if columns else strict_results, max_rows=30),
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Safety",
        "",
        "- research_only: true",
        "- live_ordering_enabled: false",
        "- order_placement: disabled",
        "- edge_claimed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bad_trade_sequence_caveat_lab(
    *,
    input_selected_report_dir: Path,
    input_event_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: BadTradeSequenceCaveatConfig = BadTradeSequenceCaveatConfig(),
) -> BadTradeSequenceCaveatResult:
    """Run reusable bad-trade sequence caveat diagnostics."""

    trades = _load_trades(input_selected_report_dir)
    event_rows = _load_events(input_event_dir)
    enriched = attach_prior_event_context(trades, event_rows)
    current_candidates = _build_current_personality_candidates(enriched, config=config)
    current_group_candidates = _build_current_group_candidates(enriched, config=config)
    prior_one_candidates, prior_two_candidates = _build_prior_sequence_candidates(
        enriched,
        config=config,
    )
    composite_candidates = _build_composite_sequence_candidates(
        current_candidates,
        prior_one_candidates,
        config=config,
    )
    fixed_candidates = (
        current_candidates
        + current_group_candidates
        + prior_one_candidates
        + prior_two_candidates
        + composite_candidates
    )
    fixed_results = _evaluate_candidates(enriched, fixed_candidates, config=config)
    numeric_candidates = _build_numeric_threshold_candidates(enriched, config=config)
    numeric_results = _mark_train_selected_numeric(
        _evaluate_candidates(enriched, numeric_candidates, config=config)
    )

    selected_numeric_names = (
        set(numeric_results.loc[numeric_results["train_selected"].astype(bool), "rule_name"])
        if not numeric_results.empty
        else set()
    )
    selected_numeric_candidates = [
        _Candidate(
            rule_name=candidate.rule_name,
            rule_family=candidate.rule_family,
            notes=candidate.notes,
            mask=candidate.mask,
            train_selected=True,
            selected_train_quantile=candidate.selected_train_quantile,
            selected_threshold=candidate.selected_threshold,
            feature=candidate.feature,
            operator=candidate.operator,
        )
        for candidate in numeric_candidates
        if candidate.rule_name in selected_numeric_names
    ]
    selected_numeric_results = (
        numeric_results[numeric_results["train_selected"].astype(bool)].copy()
        if not numeric_results.empty
        else pd.DataFrame()
    )
    strict_results = _sort_results(_concat_result_frames([fixed_results, selected_numeric_results]))
    current_results = (
        strict_results[
            strict_results["rule_family"].isin(["fixed_personality_block", "fixed_current_group"])
        ].copy()
        if not strict_results.empty
        else pd.DataFrame()
    )
    prior_results = (
        strict_results[
            strict_results["rule_family"].isin(
                [
                    "fixed_prior_personality_sequence",
                    "fixed_prior_two_personality_sequence",
                    "composite_sequence",
                ]
            )
        ].copy()
        if not strict_results.empty
        else pd.DataFrame()
    )
    flag_candidates = fixed_candidates + selected_numeric_candidates
    trade_flags = _build_trade_flags(enriched, flag_candidates)

    decision, decision_reasons = _decision(strict_results)
    run_id = "bad_trade_sequence_caveat_v0_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "sequence": run_dir / "sequence_caveat_results.csv",
        "current": run_dir / "current_personality_caveats.csv",
        "prior": run_dir / "prior_sequence_caveats.csv",
        "numeric": run_dir / "numeric_threshold_caveats.csv",
        "strict": run_dir / "strict_validation_results.csv",
        "flags": run_dir / "trade_caveat_flags.csv",
    }

    for path, frame in [
        (paths["sequence"], strict_results),
        (paths["current"], current_results),
        (paths["prior"], prior_results),
        (paths["numeric"], numeric_results),
        (paths["strict"], strict_results),
        (paths["flags"], trade_flags),
    ]:
        _write_csv(path, frame)

    base_all = _summary_stats(enriched)
    base_train = _summary_stats(enriched[enriched["month"].astype(str).isin(config.train_months)])
    base_test = _summary_stats(enriched[enriched["month"].astype(str).isin(config.test_months)])
    strict_counts = (
        strict_results["strict_status"].value_counts().to_dict() if not strict_results.empty else {}
    )
    payload = {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_selected_report_dir": str(input_selected_report_dir),
        "input_event_dir": str(input_event_dir),
        "data_source": (
            "existing local selected-filter/exit trades.csv plus local sparse "
            "state_event_detector event_rows.csv from 5m OHLCV; no vendor fetch"
        ),
        "volume_label": "historical_volume from existing local 5m OHLCV-derived report columns",
        "train_months": list(config.train_months),
        "test_months": list(config.test_months),
        "numeric_quantiles": list(config.numeric_quantiles),
        "random_iterations": int(config.random_iterations),
        "base_all": base_all,
        "base_train": base_train,
        "base_test": base_test,
        "decision": decision,
        "decision_reasons": decision_reasons,
        "strict_train_and_oos_supported_count": int(
            strict_counts.get(STRICT_TRAIN_AND_OOS_SUPPORTED, 0)
        ),
        "oos_only_not_train_supported_count": int(
            strict_counts.get(OOS_ONLY_NOT_TRAIN_SUPPORTED, 0)
        ),
        "train_only_not_oos_supported_count": int(
            strict_counts.get(TRAIN_ONLY_NOT_OOS_SUPPORTED, 0)
        ),
        "not_supported_count": int(strict_counts.get(NOT_SUPPORTED, 0)),
        "candidate_count": int(len(strict_results)),
        "numeric_threshold_candidate_count": int(len(numeric_results)),
        "best_oos_rule": _best_oos_rule(strict_results),
        "output_dir": str(run_dir),
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
        },
    )
    _write_summary_md(paths["summary_md"], payload=payload, strict_results=strict_results)

    return BadTradeSequenceCaveatResult(
        run_id=run_id,
        input_selected_report_dir=input_selected_report_dir,
        input_event_dir=input_event_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        sequence_caveat_results_csv_path=paths["sequence"],
        current_personality_caveats_csv_path=paths["current"],
        prior_sequence_caveats_csv_path=paths["prior"],
        numeric_threshold_caveats_csv_path=paths["numeric"],
        strict_validation_results_csv_path=paths["strict"],
        trade_caveat_flags_csv_path=paths["flags"],
        decision=decision,
        caveat_count=int(len(strict_results)),
    )


__all__ = [
    "BadTradeSequenceCaveatConfig",
    "BadTradeSequenceCaveatResult",
    "attach_prior_event_context",
    "run_bad_trade_sequence_caveat_lab",
]
