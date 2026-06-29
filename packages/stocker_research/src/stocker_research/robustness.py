"""Robustness helpers for research results.

These helpers summarize already-scored research outputs and define the
conservative intraday candidate robustness gate policy. They do not select
parameters or fetch data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class RobustnessGatePolicy:
    """Conservative official candidate robustness gates for intraday research."""

    require_cost_stress_for_intraday_candidate: bool = True
    cost_stress_candidate_multiplier: float = 1.5
    min_candidate_profit_factor: float = 1.10
    require_positive_median_trade: bool = True
    max_top_positive_split_share: float = 0.50
    max_top_5_winner_profit_share: float = 0.50


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clean_floats(values: Iterable[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            clean.append(result)
    return clean


def _safe_share(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _net_return_at_multiplier(
    rows: Sequence[Mapping[str, Any]],
    multiplier: float,
) -> float | None:
    for row in rows:
        row_multiplier = _maybe_float(row.get("cost_multiplier"))
        if row_multiplier is not None and math.isclose(row_multiplier, multiplier):
            return _maybe_float(row.get("net_return"))
    return None


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _profit_factor(returns: Sequence[float]) -> float:
    wins = sum(value for value in returns if value > 0)
    losses = sum(value for value in returns if value < 0)
    if losses < 0:
        return float(wins / abs(losses))
    if wins > 0:
        return float("inf")
    return 0.0


def _longest_losing_streak(returns: Sequence[float]) -> int:
    longest = 0
    current = 0
    for value in returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_trade_returns(
    trade_returns: Iterable[Any],
    *,
    top_counts: Sequence[int] = (5, 10),
) -> dict[str, float | int]:
    """Summarize round-trip trade returns and top-winner concentration."""

    values = _clean_floats(trade_returns)
    positives = sorted((value for value in values if value > 0), reverse=True)
    losses = sorted(value for value in values if value < 0)
    total_positive = float(sum(positives))
    total_loss = float(sum(losses))
    total_net = float(sum(values))
    count = len(values)

    summary: dict[str, float | int] = {
        "number_of_trades": count,
        "average_trade": float(total_net / count) if count else 0.0,
        "median_trade": float(median(values)) if values else 0.0,
        "win_rate": float(len(positives) / count) if count else 0.0,
        "profit_factor": _profit_factor(values),
        "total_trade_return_sum": total_net,
        "top5_winning_trades_contribution": float(sum(positives[:5])),
        "top5_losing_trades_contribution": float(sum(losses[:5])),
        "top5_winners_share_of_total_net": _safe_share(float(sum(positives[:5])), total_net),
        "top5_losers_share_of_total_loss": _safe_share(
            abs(float(sum(losses[:5]))),
            abs(total_loss),
        ),
        "longest_losing_streak": _longest_losing_streak(values),
    }
    for top_count in top_counts:
        top_sum = float(sum(positives[:top_count]))
        summary[f"top{top_count}_winners_share_of_positive_profit"] = _safe_share(
            top_sum,
            total_positive,
        )
    return summary


def summarize_split_returns(
    rows: Iterable[Mapping[str, Any]],
    *,
    split_key: str = "split_id",
    return_key: str = "test_net_return",
) -> dict[str, float | int | str | None]:
    """Summarize walk-forward split returns and split-level concentration."""

    parsed: list[tuple[str, float]] = []
    for index, row in enumerate(rows):
        split_id = str(row.get(split_key, f"split_{index + 1:03d}"))
        parsed.append((split_id, _finite_float(row.get(return_key))))
    if not parsed:
        return {
            "split_count": 0,
            "positive_split_count": 0,
            "profitable_split_pct": 0.0,
            "best_split_id": None,
            "best_split_return": 0.0,
            "worst_split_id": None,
            "worst_split_return": 0.0,
            "top_positive_split_share": 0.0,
        }

    positives = [value for _, value in parsed if value > 0]
    best_split_id, best_return = max(parsed, key=lambda item: item[1])
    worst_split_id, worst_return = min(parsed, key=lambda item: item[1])
    positive_total = float(sum(positives))
    top_positive = float(max(positives)) if positives else 0.0
    return {
        "split_count": len(parsed),
        "positive_split_count": len(positives),
        "profitable_split_pct": float(len(positives) / len(parsed)),
        "best_split_id": best_split_id,
        "best_split_return": float(best_return),
        "worst_split_id": worst_split_id,
        "worst_split_return": float(worst_return),
        "top_positive_split_share": _safe_share(top_positive, positive_total),
    }


def _is_costs_kill(row: Mapping[str, Any]) -> bool:
    classification = str(
        row.get(
            "classification_under_existing_gates",
            row.get("classification", ""),
        )
    )
    reasons = row.get("classification_reasons", row.get("reasons", []))
    if isinstance(reasons, str):
        reason_values = [reasons]
    elif isinstance(reasons, Iterable):
        reason_values = [str(reason) for reason in reasons]
    else:
        reason_values = []
    return classification == "rejected_costs_kill_edge" or "costs_kill_edge" in reason_values


def summarize_cost_stress(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | bool | None]:
    """Find the first cost multiplier where selected-params diagnostics fail."""

    parsed = sorted(
        (
            {
                "cost_multiplier": _finite_float(row.get("cost_multiplier")),
                "net_return": _finite_float(row.get("net_return")),
                "costs_kill": _is_costs_kill(row),
            }
            for row in rows
        ),
        key=lambda item: item["cost_multiplier"],
    )
    first_nonpositive = next(
        (
            float(row["cost_multiplier"])
            for row in parsed
            if float(row["cost_multiplier"]) > 0 and float(row["net_return"]) <= 0
        ),
        None,
    )
    first_costs_kill = next(
        (
            float(row["cost_multiplier"])
            for row in parsed
            if float(row["cost_multiplier"]) > 0 and bool(row["costs_kill"])
        ),
        None,
    )
    one_point_five = next(
        (row for row in parsed if math.isclose(float(row["cost_multiplier"]), 1.5)),
        None,
    )
    survives_1_5x = bool(
        one_point_five is not None
        and float(one_point_five["net_return"]) > 0
        and not bool(one_point_five["costs_kill"])
    )
    return {
        "first_nonpositive_net_multiplier": first_nonpositive,
        "first_costs_kill_multiplier": first_costs_kill,
        "survives_1_5x_costs": survives_1_5x,
    }


def build_intraday_robustness_diagnostics(
    *,
    cost_stress_rows: Iterable[Mapping[str, Any]] | None,
    trade_returns: Iterable[Any] | None,
    split_rows: Iterable[Mapping[str, Any]] | None,
    policy: RobustnessGatePolicy | None = None,
) -> dict[str, Any]:
    """Build official intraday robustness diagnostics for candidate gates."""

    active_policy = policy or RobustnessGatePolicy()
    cost_rows = list(cost_stress_rows or [])
    cost_summary = summarize_cost_stress(cost_rows)
    base_net_return = _net_return_at_multiplier(cost_rows, 1.0)
    net_return_1_5x = _net_return_at_multiplier(cost_rows, 1.5)
    net_return_2x = _net_return_at_multiplier(cost_rows, 2.0)
    net_return_3x = _net_return_at_multiplier(cost_rows, 3.0)
    missing_cost_stress = not cost_rows or net_return_1_5x is None
    survives_1_5x = bool(cost_summary["survives_1_5x_costs"]) and not missing_cost_stress

    missing_trade_reconstruction = trade_returns is None
    trade_values = [] if trade_returns is None else _clean_floats(trade_returns)
    trade_summary = summarize_trade_returns(trade_values)
    median_trade = float(trade_summary["median_trade"])
    profit_factor = float(trade_summary["profit_factor"])
    top5_winner_share = float(trade_summary["top5_winners_share_of_positive_profit"])
    negative_median_trade = missing_trade_reconstruction or median_trade <= 0.0
    top_winner_concentrated = (
        not missing_trade_reconstruction
        and top5_winner_share > active_policy.max_top_5_winner_profit_share
    )

    split_values = list(split_rows or [])
    split_summary = summarize_split_returns(split_values)
    missing_split_concentration = not split_values
    top_positive_split_share = float(split_summary["top_positive_split_share"])
    split_concentrated = (
        missing_split_concentration
        or top_positive_split_share > active_policy.max_top_positive_split_share
    )

    flags: list[str] = []
    if missing_cost_stress:
        flags.append("missing_cost_stress")
    if active_policy.require_cost_stress_for_intraday_candidate and not survives_1_5x:
        flags.append("fragile_costs")
    if missing_trade_reconstruction:
        flags.append("missing_trade_reconstruction")
    if negative_median_trade:
        flags.append("negative_median_trade")
    if split_concentrated:
        flags.append("split_concentrated")
    if top_winner_concentrated:
        flags.append("trade_concentrated")
    if profit_factor < active_policy.min_candidate_profit_factor:
        flags.append("weak_profit_factor")
    if missing_split_concentration:
        flags.append("missing_split_concentration")

    return {
        "cost_stress": {
            "base_net_return": base_net_return,
            "net_return_1_5x_costs": net_return_1_5x,
            "net_return_2x_costs": net_return_2x,
            "net_return_3x_costs": net_return_3x,
            "survives_1_5x_costs": survives_1_5x,
            "first_non_positive_cost_multiplier": cost_summary[
                "first_nonpositive_net_multiplier"
            ],
        },
        "trade_concentration": {
            "reconstructed_round_trips": int(trade_summary["number_of_trades"]),
            "median_trade_return": median_trade,
            "average_trade_return": float(trade_summary["average_trade"]),
            "win_rate": float(trade_summary["win_rate"]),
            "profit_factor": profit_factor,
            "top_5_winners_profit_share": top5_winner_share,
            "top_10_winners_profit_share": float(
                trade_summary["top10_winners_share_of_positive_profit"]
            ),
            "negative_median_trade": negative_median_trade,
            "top_winner_concentrated": top_winner_concentrated,
        },
        "split_concentration": {
            "top_positive_split_share": top_positive_split_share,
            "split_concentrated": split_concentrated,
            "best_split": {
                "split_id": split_summary["best_split_id"],
                "net_return": split_summary["best_split_return"],
            },
            "worst_split": {
                "split_id": split_summary["worst_split_id"],
                "net_return": split_summary["worst_split_return"],
            },
        },
        "robustness_flags": flags,
    }


def intraday_robustness_gate_failure_reasons(
    diagnostics: Mapping[str, Any] | None,
    *,
    policy: RobustnessGatePolicy | None = None,
) -> list[str]:
    """Return official intraday candidate gate failure reasons."""

    active_policy = policy or RobustnessGatePolicy()
    if diagnostics is None:
        return [
            "failed_cost_stress",
            "missing_trade_reconstruction",
            "negative_median_trade",
            "weak_profit_factor",
        ]

    cost_stress = diagnostics.get("cost_stress", {})
    trade_concentration = diagnostics.get("trade_concentration", {})
    split_concentration = diagnostics.get("split_concentration", {})
    raw_flags = diagnostics.get("robustness_flags", [])
    if isinstance(raw_flags, str):
        flags = {raw_flags}
    elif isinstance(raw_flags, Iterable):
        flags = {str(flag) for flag in raw_flags}
    else:
        flags = set()
    reasons: list[str] = []

    multiplier = active_policy.cost_stress_candidate_multiplier
    if active_policy.require_cost_stress_for_intraday_candidate:
        if math.isclose(multiplier, 1.5):
            survives_cost_stress = bool(cost_stress.get("survives_1_5x_costs", False))
        else:
            multiplier_key = str(multiplier).replace(".", "_")
            stressed_return = _maybe_float(cost_stress.get(f"net_return_{multiplier_key}x_costs"))
            survives_cost_stress = stressed_return is not None and stressed_return > 0
        if (
            not survives_cost_stress
            or "fragile_costs" in flags
            or "missing_cost_stress" in flags
        ):
            _append_once(reasons, "failed_cost_stress")

    median_trade = _maybe_float(trade_concentration.get("median_trade_return"))
    if "missing_trade_reconstruction" in flags:
        _append_once(reasons, "missing_trade_reconstruction")
    if active_policy.require_positive_median_trade and (
        median_trade is None
        or median_trade <= 0
        or bool(trade_concentration.get("negative_median_trade", False))
    ):
        _append_once(reasons, "negative_median_trade")

    top_split_share = _maybe_float(split_concentration.get("top_positive_split_share"))
    if (
        top_split_share is None
        or top_split_share > active_policy.max_top_positive_split_share
        or bool(split_concentration.get("split_concentrated", False))
    ):
        _append_once(reasons, "split_concentrated")

    top5_share = _maybe_float(trade_concentration.get("top_5_winners_profit_share"))
    if (
        top5_share is None
        or top5_share > active_policy.max_top_5_winner_profit_share
        or bool(trade_concentration.get("top_winner_concentrated", False))
    ):
        _append_once(reasons, "trade_concentrated")

    profit_factor = _maybe_float(trade_concentration.get("profit_factor"))
    if profit_factor is None or profit_factor < active_policy.min_candidate_profit_factor:
        _append_once(reasons, "weak_profit_factor")

    return reasons


def build_partial_pass_row(
    *,
    symbol: str,
    benchmark_pass: bool,
    null_pass: bool,
    net_return: float,
    cost_stress_survives_1_5x: bool,
    median_trade: float,
    top_positive_split_share: float,
    top_winner_share: float,
    stability_score: float,
    train_selection_succeeded: bool,
    session_flat_compliant: bool,
    stability_threshold: float = 0.5,
    concentration_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a one-symbol partial-pass matrix row."""

    return {
        "symbol": symbol,
        "benchmark_pass": bool(benchmark_pass),
        "null_pass": bool(null_pass),
        "positive_net_return": float(net_return) > 0,
        "net_return": float(net_return),
        "cost_stress_survives_1_5x": bool(cost_stress_survives_1_5x),
        "median_trade_positive": float(median_trade) > 0,
        "median_trade": float(median_trade),
        "top_split_concentration_ok": float(top_positive_split_share)
        <= concentration_threshold,
        "top_positive_split_share": float(top_positive_split_share),
        "top_trade_concentration_ok": float(top_winner_share) <= concentration_threshold,
        "top_winner_share": float(top_winner_share),
        "stability_score_at_least_threshold": float(stability_score) >= stability_threshold,
        "stability_score": float(stability_score),
        "train_selection_succeeded": bool(train_selection_succeeded),
        "session_flat_compliant": bool(session_flat_compliant),
    }


def build_robustness_flags(
    partial_pass_row: Mapping[str, Any],
    *,
    trade_count: int,
    min_trades: int = 20,
    session_quality_warning: bool = False,
    stable_low_return_threshold: float = 0.01,
) -> list[str]:
    """Convert a partial-pass row into conservative diagnostic flags."""

    flags: list[str] = []
    if not bool(partial_pass_row.get("cost_stress_survives_1_5x", False)):
        flags.append("fragile_costs")
    if not bool(partial_pass_row.get("top_split_concentration_ok", False)):
        flags.append("split_concentrated")
    if not bool(partial_pass_row.get("top_trade_concentration_ok", False)):
        flags.append("trade_concentrated")
    if not bool(partial_pass_row.get("median_trade_positive", False)):
        flags.append("negative_median_trade")
    if not bool(partial_pass_row.get("train_selection_succeeded", False)):
        flags.append("train_selection_failed")
    if not bool(partial_pass_row.get("benchmark_pass", False)):
        flags.append("benchmark_failed")
    if not bool(partial_pass_row.get("null_pass", False)):
        flags.append("null_failed")
    if int(trade_count) < min_trades:
        flags.append("too_few_trades")
    if session_quality_warning:
        flags.append("session_quality_warning")

    net_return = _finite_float(partial_pass_row.get("net_return"))
    stable = bool(partial_pass_row.get("stability_score_at_least_threshold", False))
    if stable and 0 < net_return < stable_low_return_threshold:
        flags.append("stable_but_low_return")

    core_keys = (
        "benchmark_pass",
        "null_pass",
        "positive_net_return",
        "cost_stress_survives_1_5x",
        "median_trade_positive",
        "top_split_concentration_ok",
        "top_trade_concentration_ok",
        "stability_score_at_least_threshold",
        "train_selection_succeeded",
        "session_flat_compliant",
    )
    if not flags and all(bool(partial_pass_row.get(key, False)) for key in core_keys):
        flags.append("robust_partial_pass")
    return flags
