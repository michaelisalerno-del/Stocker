"""Independent-stock session payoff aggregation for loop edge-state research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

GROUP_COLUMNS: Final = ["session", "loop_id", "orientation", "horizon"]
COST_COLUMNS: Final = [
    "entry_cost_bps",
    "exit_cost_bps",
    "spread_cost_bps",
    "slippage_cost_bps",
    "commission_cost_bps",
    "financing_cost_bps",
    "fx_cost_bps",
    "other_cost_bps",
]
FEATURE_COLUMNS: Final = [
    "structural_breadth",
    "top_loop_score",
    "top_second_margin",
    "loop_score_entropy",
    "transition_surprise",
    "market_return",
    "market_volatility",
    "liquidity_pressure",
]
PANEL_COLUMNS: Final = [
    *GROUP_COLUMNS,
    "decision_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "data_availability_timestamp",
    "robust_net_payoff_bps",
    "robust_gross_payoff_bps",
    "cost_contribution_bps",
    "independent_stock_count",
    "raw_fill_count",
    "effective_sample_size",
    "positive_stock_fraction",
    "median_stock_payoff_bps",
    "cross_stock_payoff_dispersion_bps",
    "downside_frequency",
    *FEATURE_COLUMNS,
    "independent_stock_ids",
    "source_fill_ids",
    "aggregation_method",
]


@dataclass(frozen=True)
class AggregationSettings:
    """Frozen settings for one-stock-per-session payoff evidence."""

    method: str = "equal_stock_winsorized_mean"
    winsor_fraction_each_tail: float = 0.1
    stock_contribution_cap_bps: float = 500.0

    def __post_init__(self) -> None:
        if self.method not in {"equal_stock_winsorized_mean", "median"}:
            raise ValueError(f"unsupported aggregation method: {self.method}")
        if not 0.0 <= self.winsor_fraction_each_tail < 0.5:
            raise ValueError("winsor fraction must be in [0, 0.5)")
        if self.stock_contribution_cap_bps <= 0.0:
            raise ValueError("stock contribution cap must be positive")


def settled_before(trades: pd.DataFrame, decision_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return outcomes fully known strictly before a decision timestamp."""

    if "settlement_timestamp" not in trades:
        raise ValueError("settlement_timestamp is required")
    decision = pd.Timestamp(decision_timestamp)
    if decision.tzinfo is None:
        raise ValueError("decision timestamp must be timezone-aware")
    settlements = pd.to_datetime(trades["settlement_timestamp"], utc=True, errors="raise")
    return trades.loc[settlements.lt(decision.tz_convert("UTC"))].copy()


def _robust_location(values: np.ndarray, settings: AggregationSettings) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    if settings.method == "median":
        return float(np.median(finite))
    lower, upper = np.quantile(
        finite,
        [settings.winsor_fraction_each_tail, 1.0 - settings.winsor_fraction_each_tail],
    )
    return float(np.clip(finite, lower, upper).mean())


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=PANEL_COLUMNS)


def _validate(trades: pd.DataFrame) -> pd.DataFrame:
    required = {
        *GROUP_COLUMNS,
        "stock_id",
        "fill_id",
        "decision_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "settlement_timestamp",
        "feature_availability_timestamp",
        "gross_payoff_bps",
        *COST_COLUMNS,
    }
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"missing session-payoff columns: {missing}")
    frame = trades.copy()
    for column in (
        "decision_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "settlement_timestamp",
        "feature_availability_timestamp",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame["feature_availability_timestamp"].gt(frame["decision_timestamp"]).any():
        raise ValueError("feature availability timestamp is after its decision timestamp")
    if frame["entry_timestamp"].lt(frame["decision_timestamp"]).any():
        raise ValueError("entry timestamp precedes decision timestamp")
    if frame["exit_timestamp"].lt(frame["entry_timestamp"]).any():
        raise ValueError("exit timestamp precedes entry timestamp")
    if frame["settlement_timestamp"].lt(frame["exit_timestamp"]).any():
        raise ValueError("settlement timestamp precedes exit timestamp")
    numeric = ["gross_payoff_bps", *COST_COLUMNS]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("non-finite payoff or cost")
    if (frame[COST_COLUMNS].to_numpy(float) < 0.0).any():
        raise ValueError("cost components must be non-negative")
    return frame


def aggregate_session_payoffs(
    trades: pd.DataFrame,
    settings: AggregationSettings,
) -> pd.DataFrame:
    """Aggregate fills to one robust observation per session/loop/orientation/horizon.

    Repeated fills first collapse to one equal-weighted, capped stock contribution.
    An absent group is never materialised, so no-opportunity sessions remain missing.
    """

    if trades.empty:
        return _empty_panel()
    frame = _validate(trades)
    frame["total_cost_bps"] = frame.loc[:, COST_COLUMNS].sum(axis=1)
    frame["net_payoff_bps"] = frame["gross_payoff_bps"].to_numpy(float) - frame[
        "total_cost_bps"
    ].to_numpy(float)
    stock_keys = [*GROUP_COLUMNS, "stock_id"]
    stock_rows: list[dict[str, object]] = []
    for key, group in frame.groupby(stock_keys, sort=True, observed=True):
        key_values = key if isinstance(key, tuple) else (key,)
        raw_net = float(group["net_payoff_bps"].mean())
        capped_net = float(
            np.clip(
                raw_net,
                -settings.stock_contribution_cap_bps,
                settings.stock_contribution_cap_bps,
            )
        )
        stock_cost = float(group["total_cost_bps"].mean())
        row: dict[str, object] = dict(zip(stock_keys, key_values, strict=True))
        row.update(
            {
                "stock_net_payoff_bps": capped_net,
                "stock_cost_bps": stock_cost,
                "stock_gross_payoff_bps": capped_net + stock_cost,
                "raw_fill_count": int(len(group)),
                "decision_timestamp": group["decision_timestamp"].min(),
                "entry_timestamp": group["entry_timestamp"].min(),
                "exit_timestamp": group["exit_timestamp"].max(),
                "data_availability_timestamp": group["settlement_timestamp"].max(),
                "source_fill_ids": sorted(group["fill_id"].astype(str).tolist()),
            }
        )
        for feature in FEATURE_COLUMNS:
            row[feature] = (
                float(pd.to_numeric(group[feature], errors="coerce").mean())
                if feature in group
                else float("nan")
            )
        stock_rows.append(row)
    stocks = pd.DataFrame(stock_rows)
    observations: list[dict[str, object]] = []
    for key, group in stocks.groupby(GROUP_COLUMNS, sort=True, observed=True):
        key_values = key if isinstance(key, tuple) else (key,)
        net = group["stock_net_payoff_bps"].to_numpy(float)
        cost_values = group["stock_cost_bps"].to_numpy(float)
        robust_net = _robust_location(net, settings)
        robust_cost = _robust_location(cost_values, settings)
        median = float(np.median(net))
        mad = float(1.4826 * np.median(np.abs(net - median)))
        stock_ids = sorted(group["stock_id"].astype(str).unique().tolist())
        fill_ids = sorted(fill_id for values in group["source_fill_ids"] for fill_id in values)
        row = dict(zip(GROUP_COLUMNS, key_values, strict=True))
        row.update(
            {
                "decision_timestamp": group["decision_timestamp"].min(),
                "entry_timestamp": group["entry_timestamp"].min(),
                "exit_timestamp": group["exit_timestamp"].max(),
                "data_availability_timestamp": group["data_availability_timestamp"].max(),
                "robust_net_payoff_bps": robust_net,
                "robust_gross_payoff_bps": robust_net + robust_cost,
                "cost_contribution_bps": robust_cost,
                "independent_stock_count": int(len(stock_ids)),
                "raw_fill_count": int(group["raw_fill_count"].sum()),
                "effective_sample_size": float(len(stock_ids)),
                "positive_stock_fraction": float((net > 0.0).mean()),
                "median_stock_payoff_bps": median,
                "cross_stock_payoff_dispersion_bps": mad,
                "downside_frequency": float((net <= 0.0).mean()),
                "independent_stock_ids": json.dumps(stock_ids, separators=(",", ":")),
                "source_fill_ids": json.dumps(fill_ids, separators=(",", ":")),
                "aggregation_method": settings.method,
            }
        )
        for feature in FEATURE_COLUMNS:
            row[feature] = float(group[feature].mean())
        observations.append(row)
    return (
        pd.DataFrame(observations, columns=PANEL_COLUMNS)
        .sort_values(GROUP_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )
