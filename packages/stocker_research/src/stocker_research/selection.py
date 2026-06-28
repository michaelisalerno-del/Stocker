"""Train-side parameter selection for research experiments."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SelectionResult(BaseModel):
    """Chosen parameter set and diagnostics from honest train-side selection."""

    selected_parameter_set_id: str
    selected_result: dict[str, Any]
    selection_method: str
    rejected_parameter_set_ids: list[str]
    diagnostics: dict[str, Any]


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("parameter_set_id", ""))


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def _as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value is None:
        return default
    return int(value)


def _best_test_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda row: (-_as_float(row, "test_net_return"), _row_id(row)))[0]


def _passes_train_gates(
    row: dict[str, Any],
    *,
    minimum_train_trades: int,
    max_train_drawdown: float,
    minimum_train_profitable_split_pct: float,
) -> bool:
    train_drawdown = _as_float(row, "train_max_drawdown", 0.0)
    return (
        _as_float(row, "train_net_return") > 0.0
        and _as_int(row, "train_trade_count") >= minimum_train_trades
        and train_drawdown >= max_train_drawdown
        and _as_float(row, "train_profitable_split_pct") >= minimum_train_profitable_split_pct
    )


def _sort_train_candidates(row: dict[str, Any]) -> tuple[float, float, int, float, str]:
    return (
        -_as_float(row, "train_net_return"),
        -_as_float(row, "train_profitable_split_pct"),
        -_as_int(row, "train_trade_count"),
        -_as_float(row, "train_max_drawdown", 0.0),
        _row_id(row),
    )


def select_parameter_set(
    grid_results: list[dict[str, Any]],
    *,
    minimum_train_trades: int = 20,
    max_train_drawdown: float = -0.25,
    minimum_train_profitable_split_pct: float = 0.0,
) -> SelectionResult:
    """Select a parameter set using training-side evidence only.

    The best test performer is retained as a diagnostic, but it is never used for
    selection.
    """

    if not grid_results:
        return SelectionResult(
            selected_parameter_set_id="none",
            selected_result={
                "parameter_set_id": "none",
                "params": {},
                "train_net_return": 0.0,
                "test_net_return": 0.0,
                "test_trade_count": 0,
                "test_max_drawdown": 0.0,
                "test_profitable_split_pct": 0.0,
            },
            selection_method="fallback_for_reporting_only",
            rejected_parameter_set_ids=[],
            diagnostics={
                "best_test_parameter_set_id": "none",
                "best_test_net_return": 0.0,
                "train_gate_pass_count": 0,
                "fallback_for_reporting_only": True,
                "selection_reason": "no_grid_results",
            },
        )

    ordered_rows = sorted(grid_results, key=_row_id)
    best_test = _best_test_row(ordered_rows)
    candidates = [
        row
        for row in ordered_rows
        if _passes_train_gates(
            row,
            minimum_train_trades=minimum_train_trades,
            max_train_drawdown=max_train_drawdown,
            minimum_train_profitable_split_pct=minimum_train_profitable_split_pct,
        )
    ]
    candidate_ids = {_row_id(row) for row in candidates}
    rejected_ids = sorted(_row_id(row) for row in ordered_rows if _row_id(row) not in candidate_ids)

    diagnostics: dict[str, Any] = {
        "best_test_parameter_set_id": _row_id(best_test),
        "best_test_net_return": _as_float(best_test, "test_net_return"),
        "train_gate_pass_count": len(candidates),
        "fallback_for_reporting_only": False,
        "selection_reason": "train_positive_trades_drawdown",
        "non_isolated_train_pass": len(candidates) > 1,
    }
    if candidates:
        selected = sorted(candidates, key=_sort_train_candidates)[0]
        return SelectionResult(
            selected_parameter_set_id=_row_id(selected),
            selected_result=selected,
            selection_method="train_gated",
            rejected_parameter_set_ids=rejected_ids,
            diagnostics=diagnostics,
        )

    fallback = ordered_rows[0]
    diagnostics["fallback_for_reporting_only"] = True
    diagnostics["selection_reason"] = "no_parameter_set_passed_train_gates"
    return SelectionResult(
        selected_parameter_set_id=_row_id(fallback),
        selected_result=fallback,
        selection_method="fallback_for_reporting_only",
        rejected_parameter_set_ids=[_row_id(row) for row in ordered_rows],
        diagnostics=diagnostics,
    )
