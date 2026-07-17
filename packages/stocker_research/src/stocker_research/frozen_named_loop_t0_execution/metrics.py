"""Deterministic metrics for the frozen execution-realism experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from .execution import gross_payoff_bps

FAMILY_CONTROL_PAIRS: Final[dict[str, str]] = {
    "cycle_04|state_4": "cycle_04|state_2",
    "cycle_07|state_5": "cycle_07|state_6",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"missing metric column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def performance_metrics(
    frame: pd.DataFrame, *, value_column: str = "net_payoff_bps"
) -> dict[str, float | int]:
    """Summarise one deterministic payoff population in row order."""

    values = _numeric(frame, value_column)
    positive = float(values.loc[values.gt(0.0)].sum())
    negative = float(-values.loc[values.lt(0.0)].sum())
    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax() if len(cumulative) else pd.Series(dtype=float)
    return {
        "opportunities": int(len(values)),
        "total_net_payoff_bps": float(values.sum()),
        "mean_net_payoff_bps": float(values.mean()) if len(values) else float("nan"),
        "median_net_payoff_bps": float(values.median()) if len(values) else float("nan"),
        "positive_payoff_rate": float(values.gt(0.0).mean()) if len(values) else float("nan"),
        "profit_factor": positive / negative if negative > 0.0 else float("nan"),
        "maximum_drawdown_bps": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def family_metrics(payoffs: pd.DataFrame) -> pd.DataFrame:
    """Report every family and fill model separately."""

    required = {"family", "classification", "fill_model", "net_payoff_bps"}
    if missing := sorted(required - set(payoffs)):
        raise ValueError(f"missing family metric columns: {missing}")
    rows: list[dict[str, object]] = []
    for (family, classification, fill_model), group in payoffs.groupby(
        ["family", "classification", "fill_model"], sort=True, dropna=False
    ):
        row: dict[str, object] = {
            "family": str(family),
            "classification": str(classification),
            "fill_model": str(fill_model),
            **performance_metrics(group),
        }
        row["independent_stocks"] = int(group["symbol"].nunique()) if "symbol" in group else 0
        row["sessions"] = int(group["session"].nunique()) if "session" in group else 0
        row["months"] = int(group["month"].nunique()) if "month" in group else 0
        rows.append(row)
    return pd.DataFrame(rows)


def named_control_comparisons(payoffs: pd.DataFrame) -> pd.DataFrame:
    """Compare only the two predeclared same-parent-loop family pairs."""

    metrics = family_metrics(payoffs)
    rows: list[dict[str, object]] = []
    for named, control in FAMILY_CONTROL_PAIRS.items():
        for fill_model in sorted(payoffs["fill_model"].astype(str).unique()):
            named_rows = metrics.loc[
                metrics["family"].eq(named) & metrics["fill_model"].eq(fill_model)
            ]
            control_rows = metrics.loc[
                metrics["family"].eq(control) & metrics["fill_model"].eq(fill_model)
            ]
            if named_rows.empty or control_rows.empty:
                continue
            named_row = named_rows.iloc[0]
            control_row = control_rows.iloc[0]
            rows.append(
                {
                    "comparison": f"{named}-minus-{control}",
                    "named_family": named,
                    "control_family": control,
                    "fill_model": fill_model,
                    "named_opportunities": int(named_row["opportunities"]),
                    "control_opportunities": int(control_row["opportunities"]),
                    "named_mean_bps": float(named_row["mean_net_payoff_bps"]),
                    "control_mean_bps": float(control_row["mean_net_payoff_bps"]),
                    "mean_difference_bps": float(named_row["mean_net_payoff_bps"])
                    - float(control_row["mean_net_payoff_bps"]),
                }
            )
    for fill_model in sorted(payoffs["fill_model"].astype(str).unique()):
        selected = payoffs.loc[payoffs["fill_model"].eq(fill_model)]
        named_values = _numeric(
            selected.loc[selected["classification"].eq("named")], "net_payoff_bps"
        )
        control_values = _numeric(
            selected.loc[selected["classification"].eq("control")], "net_payoff_bps"
        )
        if len(named_values) and len(control_values):
            rows.append(
                {
                    "comparison": "combined_named-minus-combined_controls_secondary",
                    "named_family": "combined_named_secondary",
                    "control_family": "combined_controls_secondary",
                    "fill_model": fill_model,
                    "named_opportunities": len(named_values),
                    "control_opportunities": len(control_values),
                    "named_mean_bps": float(named_values.mean()),
                    "control_mean_bps": float(control_values.mean()),
                    "mean_difference_bps": float(named_values.mean() - control_values.mean()),
                }
            )
    return pd.DataFrame(rows)


def session_block_bootstrap(
    frame: pd.DataFrame,
    *,
    resamples: int,
    block_length: int,
    seed: int,
    value_column: str = "net_payoff_bps",
    session_column: str = "session",
) -> dict[str, float | int]:
    """Circular moving-block interval over chronological session means."""

    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive")
    if session_column not in frame:
        raise ValueError(f"missing session column: {session_column}")
    session = (
        frame.assign(_value=pd.to_numeric(frame[value_column], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby(session_column, sort=True)["_value"]
        .mean()
    )
    values = session.to_numpy(float)
    if not len(values):
        return {
            "sessions": 0,
            "observed_session_mean_bps": float("nan"),
            "sessions_positive_percentage": float("nan"),
            "bootstrap_lower_95_bps": float("nan"),
            "bootstrap_upper_95_bps": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    block_count = int(np.ceil(len(values) / block_length))
    for draw in range(resamples):
        starts = rng.integers(0, len(values), size=block_count)
        rebuilt = np.asarray(
            [
                values[(int(start) + offset) % len(values)]
                for start in starts
                for offset in range(block_length)
            ],
            dtype=float,
        )[: len(values)]
        draws[draw] = float(rebuilt.mean())
    return {
        "sessions": int(len(values)),
        "observed_session_mean_bps": float(values.mean()),
        "sessions_positive_percentage": float(100.0 * np.mean(values > 0.0)),
        "bootstrap_lower_95_bps": float(np.quantile(draws, 0.025)),
        "bootstrap_upper_95_bps": float(np.quantile(draws, 0.975)),
    }


def _mean_at_stress(frame: pd.DataFrame, adverse_bps: float) -> float:
    required = {
        "direction",
        "reference_entry_price",
        "terminal_price",
        "cost_bps",
    }
    if missing := sorted(required - set(frame)):
        raise ValueError(f"missing break-even fields: {missing}")
    direction = frame["direction"].to_numpy(int)
    entry = frame["reference_entry_price"].to_numpy(float)
    terminal = frame["terminal_price"].to_numpy(float)
    costs = frame["cost_bps"].to_numpy(float)
    factor = np.where(direction == 1, 1.0 + adverse_bps / 10_000.0, 1.0 - adverse_bps / 10_000.0)
    stressed = entry * factor
    net = 10_000.0 * direction * (terminal / stressed - 1.0) - costs
    return float(np.mean(net))


def break_even_adverse_slippage_bps(frame: pd.DataFrame, *, tolerance_bps: float = 1e-9) -> float:
    """Find the diagnostic zero-mean adverse entry stress by bisection."""

    if frame.empty or _mean_at_stress(frame, 0.0) <= 0.0:
        return 0.0
    low = 0.0
    high = 25.0
    while high < 9_000.0 and _mean_at_stress(frame, high) > 0.0:
        high *= 2.0
    if _mean_at_stress(frame, high) > 0.0:
        return high
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _mean_at_stress(frame, midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= tolerance_bps:
            break
    return (low + high) / 2.0


def session_block_break_even_bootstrap(
    frame: pd.DataFrame,
    *,
    resamples: int,
    block_length: int,
    seed: int,
    session_column: str = "session",
) -> dict[str, float | int]:
    """Five-session block uncertainty for diagnostic break-even slippage."""

    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive")
    if session_column not in frame:
        raise ValueError(f"missing session column: {session_column}")
    sessions = sorted(frame[session_column].astype(str).unique())
    point = break_even_adverse_slippage_bps(frame)
    if not sessions:
        return {
            "sessions": 0,
            "break_even_adverse_slippage_bps": point,
            "bootstrap_lower_95_bps": float("nan"),
            "bootstrap_upper_95_bps": float("nan"),
        }
    groups = {
        session: frame.loc[frame[session_column].astype(str).eq(session)].copy()
        for session in sessions
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    block_count = int(np.ceil(len(sessions) / block_length))
    for draw in range(resamples):
        starts = rng.integers(0, len(sessions), size=block_count)
        sampled_sessions = [
            sessions[(int(start) + offset) % len(sessions)]
            for start in starts
            for offset in range(block_length)
        ][: len(sessions)]
        sampled = pd.concat([groups[session] for session in sampled_sessions], ignore_index=True)
        draws[draw] = break_even_adverse_slippage_bps(sampled)
    return {
        "sessions": len(sessions),
        "break_even_adverse_slippage_bps": point,
        "bootstrap_lower_95_bps": float(np.quantile(draws, 0.025)),
        "bootstrap_upper_95_bps": float(np.quantile(draws, 0.975)),
    }


def concentration_summary(
    frame: pd.DataFrame,
    *,
    dimension: str,
    value_column: str = "net_payoff_bps",
) -> dict[str, object]:
    """Concentration of absolute group contributions for one dimension."""

    if dimension not in frame:
        raise ValueError(f"missing concentration dimension: {dimension}")
    grouped = (
        frame.assign(_value=pd.to_numeric(frame[value_column], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby(dimension, dropna=False, sort=True)["_value"]
        .sum()
    )
    absolute = grouped.abs().sort_values(ascending=False)
    total = float(absolute.sum())
    shares = absolute / total if total > 0.0 else absolute * 0.0
    top_one = float(shares.iloc[:1].sum())
    top_five = float(shares.iloc[:5].sum())
    hhi = float(np.square(shares.to_numpy(float)).sum())
    return {
        "dimension": dimension,
        "contributors": int(len(grouped)),
        "top_one_absolute_contribution_share": top_one,
        "top_five_absolute_contribution_share": top_five,
        "herfindahl_index": hhi,
        "dominant_contributor": str(absolute.index[0]) if len(absolute) else None,
        "concentrated_or_unstable": bool(top_one >= 0.5 or top_five >= 0.8),
    }


def remove_top_contributors(
    frame: pd.DataFrame,
    *,
    dimension: str,
    top_n: int,
    value_column: str = "net_payoff_bps",
) -> dict[str, object]:
    """Remove the highest positive contributors and recompute every metric."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if dimension not in frame:
        raise ValueError(f"missing deletion dimension: {dimension}")
    contributions = (
        frame.assign(_value=pd.to_numeric(frame[value_column], errors="coerce"))
        .groupby(dimension, dropna=False, sort=True)["_value"]
        .sum()
        .sort_values(ascending=False)
    )
    removed = [str(value) for value in contributions.index[:top_n]]
    keep = ~frame[dimension].astype(str).isin(removed)
    metrics = performance_metrics(frame.loc[keep], value_column=value_column)
    return {
        "dimension": dimension,
        "top_n": top_n,
        "removed": removed,
        "remaining_opportunities": metrics["opportunities"],
        "remaining_total_net_payoff_bps": metrics["total_net_payoff_bps"],
        "remaining_mean_net_payoff_bps": metrics["mean_net_payoff_bps"],
    }


def leave_one_stock_out(
    frame: pd.DataFrame, *, value_column: str = "net_payoff_bps"
) -> pd.DataFrame:
    """Deterministic row-deletion attribution; there is no model to retrain."""

    if "symbol" not in frame:
        raise ValueError("missing symbol column")
    rows: list[dict[str, object]] = []
    for symbol in sorted(frame["symbol"].astype(str).unique()):
        remaining = frame.loc[~frame["symbol"].astype(str).eq(symbol)]
        metrics = performance_metrics(remaining, value_column=value_column)
        rows.append(
            {
                "removed_symbol": symbol,
                "remaining_opportunities": metrics["opportunities"],
                "remaining_total_net_payoff_bps": metrics["total_net_payoff_bps"],
                "remaining_mean_net_payoff_bps": metrics["mean_net_payoff_bps"],
                "attribution_method": "deterministic_row_deletion_not_model_refit",
            }
        )
    return pd.DataFrame(rows)


def direction_flipped_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Flip direction only, preserving the same entry, terminal, and identity."""

    required = {
        "opportunity_id",
        "direction",
        "reference_entry_price",
        "terminal_timestamp",
        "terminal_price",
        "cost_bps",
    }
    if missing := sorted(required - set(frame)):
        raise ValueError(f"missing direction-flip fields: {missing}")
    rows: list[dict[str, object]] = []
    for raw in frame.to_dict(orient="records"):
        values = cast(dict[str, Any], raw)
        flipped = -int(values["direction"])
        gross = gross_payoff_bps(
            flipped,
            float(values["reference_entry_price"]),
            float(values["terminal_price"]),
        )
        rows.append(
            {
                "opportunity_id": str(values["opportunity_id"]),
                "direction": flipped,
                "reference_entry_price": float(values["reference_entry_price"]),
                "terminal_timestamp": pd.Timestamp(values["terminal_timestamp"]),
                "terminal_price": float(values["terminal_price"]),
                "cost_bps": float(values["cost_bps"]),
                "gross_payoff_bps": gross,
                "net_payoff_bps": gross - float(values["cost_bps"]),
                "diagnostic": "direction_flipped_same_fill_and_terminal",
            }
        )
    return pd.DataFrame(rows)


def add_identity_columns(frame: pd.DataFrame, identity: Mapping[str, object]) -> pd.DataFrame:
    """Attach deterministic run identity without changing metric calculations."""

    result = frame.copy()
    for key, value in identity.items():
        result[key] = cast(Any, value)
    return result
