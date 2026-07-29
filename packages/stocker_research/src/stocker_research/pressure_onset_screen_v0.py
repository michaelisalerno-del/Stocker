"""Causal primitives for High-Movement Pressure-Onset Screen V0.

This module is research-only and cannot place orders or modify production runtime.
It deliberately contains no loop, regime, state, closure, excursion, transition,
posterior, or structural-path features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

DECISION_ORDINALS = (6, 12)
EXPECTED_SESSION_BARS = 78
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
FloatArray = NDArray[np.float64]
FORBIDDEN_FEATURE_FRAGMENTS = (
    "regime",
    "state",
    "loop",
    "closure",
    "excursion",
    "transition",
    "posterior",
    "structural",
)
DECISIONS = (
    "pressure_onset_and_direction_increment_observed",
    "pressure_onset_occurrence_only",
    "directional_pressure_only",
    "one_bar_confirmation_required",
    "movement_readiness_but_direction_unresolved",
    "no_pressure_onset_increment",
    "pressure_signal_observed_but_concentration_gate_failed",
    "blocked_parent_slate_support_failure",
    "blocked_null_semantics_failure",
    "blocked_observable_movement_model_not_reconstructable",
    "blocked_protected_boundary_failure",
    "blocked_chronology_or_leakage_failure",
    "blocked_insufficient_pressure_onset_support",
    "blocked_quick_pressure_screen_resource_limit",
    "blocked_model_convergence_failure",
    "blocked_reproducibility_or_audit_failure",
)


def decision_bar_start_ordinal(decision_ordinal: int) -> int:
    """Map completed-bar count to the repository's zero-based bar-start ordinal."""

    if decision_ordinal not in DECISION_ORDINALS:
        raise ValueError("decision ordinal must be 6 or 12")
    return decision_ordinal - 1


def decision_time_local(decision_ordinal: int) -> time:
    """Return the New York clock time when the decision bar is complete."""

    if decision_ordinal == 6:
        return time(10, 0)
    if decision_ordinal == 12:
        return time(10, 30)
    raise ValueError("decision ordinal must be 6 or 12")


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    """Fixed causal timestamps and prices around one decision checkpoint."""

    decision_bar_ordinal: int
    decision_available_timestamp: pd.Timestamp
    confirmation_bar_ordinal: int
    confirmation_available_timestamp: pd.Timestamp
    entry_bar_ordinal: int
    entry_timestamp: pd.Timestamp
    delayed_entry_open: float
    onset_bar_ordinals: tuple[int, int, int]
    onset_closes: tuple[float, float, float]
    continuation_exit_bar_ordinal: int
    continuation_exit_close: float
    terminal_bar_ordinal: int
    terminal_close: float


def extract_decision_window(bars: pd.DataFrame, *, decision_ordinal: int) -> DecisionWindow:
    """Validate one full regular session and return the preregistered fixed window."""

    required = {
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "session",
        "source_quality_passed",
        "corporate_action_passed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"decision window columns missing: {missing}")
    ordered = bars.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True).copy()
    ordinals = pd.to_numeric(ordered["bar_ordinal"], errors="raise").astype(int)
    starts = pd.to_datetime(ordered["bar_start_timestamp"], utc=True, errors="raise")
    completes = pd.to_datetime(ordered["bar_complete_timestamp"], utc=True, errors="raise")
    local = starts.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    prices = ordered[["open", "high", "low", "close"]].to_numpy(dtype=float)
    exact_grid = bool(
        len(ordered) == EXPECTED_SESSION_BARS
        and ordinals.tolist() == list(range(EXPECTED_SESSION_BARS))
        and minute.tolist() == list(range(570, 960, 5))
        and local.dt.second.eq(0).all()
        and local.dt.microsecond.eq(0).all()
        and completes.eq(starts + pd.Timedelta(minutes=5)).all()
    )
    if not exact_grid:
        raise ValueError("session is not aligned to the exact five-minute regular-session grid")
    if ordered["session"].astype(str).nunique() != 1:
        raise ValueError("decision window crosses a session boundary")
    if not ordered["source_quality_passed"].astype(bool).all():
        raise ValueError("source gap or quarantined source quality failure")
    if not ordered["corporate_action_passed"].astype(bool).all():
        raise ValueError("corporate-action quality failure")
    if not np.isfinite(prices).all() or not bool((prices > 0.0).all()):
        raise ValueError("decision window prices must be positive and finite")
    if starts.ge(PROTECTED_START).any():
        raise ValueError("protected market row")

    origin = decision_bar_start_ordinal(decision_ordinal)
    confirmation = origin + 1
    entry = origin + 2
    onset = (entry, entry + 1, entry + 2)
    continuation_exit = origin + 8
    terminal = EXPECTED_SESSION_BARS - 1
    by_ordinal = ordered.set_index("bar_ordinal")
    if not by_ordinal.index.is_unique:
        raise ValueError("decision window contains duplicate bar ordinals")

    def cell(ordinal: int, column: str) -> Any:
        return cast(Any, by_ordinal.loc[ordinal, column])

    decision_available = pd.Timestamp(cell(origin, "bar_complete_timestamp"))
    confirmation_available = pd.Timestamp(cell(confirmation, "bar_complete_timestamp"))
    entry_timestamp = pd.Timestamp(cell(entry, "bar_start_timestamp"))
    if confirmation_available > entry_timestamp:
        raise ValueError("confirmation uses information unavailable before delayed entry")
    if decision_available.tz_convert("America/New_York").time() != decision_time_local(
        decision_ordinal
    ):
        raise ValueError("decision timestamp does not match checkpoint")
    return DecisionWindow(
        decision_bar_ordinal=origin,
        decision_available_timestamp=decision_available,
        confirmation_bar_ordinal=confirmation,
        confirmation_available_timestamp=confirmation_available,
        entry_bar_ordinal=entry,
        entry_timestamp=entry_timestamp,
        delayed_entry_open=float(cell(entry, "open")),
        onset_bar_ordinals=onset,
        onset_closes=(
            float(cell(onset[0], "close")),
            float(cell(onset[1], "close")),
            float(cell(onset[2], "close")),
        ),
        continuation_exit_bar_ordinal=continuation_exit,
        continuation_exit_close=float(cell(continuation_exit, "close")),
        terminal_bar_ordinal=terminal,
        terminal_close=float(cell(terminal, "close")),
    )


def cohort_relative_cumulative_paths_bps(
    cumulative_returns_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the leave-one-stock-out median at each simultaneous path close."""

    values = np.asarray(cumulative_returns_bps, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError("cohort paths require at least two stocks and exactly three closes")
    if not np.isfinite(values).all():
        raise ValueError("cohort paths must be finite")
    medians = np.empty_like(values)
    for row in range(values.shape[0]):
        medians[row] = np.median(np.delete(values, row, axis=0), axis=0)
    return values - medians, medians


def development_onset_barriers(frame: pd.DataFrame) -> dict[int, float]:
    """Freeze checkpoint-specific q75 maximum absolute residual moves on 2024 only."""

    path_columns = (
        "residual_t_plus_2_bps",
        "residual_t_plus_3_bps",
        "residual_t_plus_4_bps",
    )
    required = {"year", "decision_ordinal", *path_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"onset barrier columns missing: {missing}")
    development = frame.loc[pd.to_numeric(frame["year"], errors="raise").eq(2024)].copy()
    output: dict[int, float] = {}
    for ordinal in DECISION_ORDINALS:
        subset = development.loc[development["decision_ordinal"].eq(ordinal), path_columns]
        values = subset.to_numpy(dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"invalid 2024 onset paths at checkpoint {ordinal}")
        maximum = np.max(np.abs(values), axis=1)
        output[ordinal] = float(pd.Series(maximum).quantile(0.75, interpolation="linear"))
    return output


def classify_onset(
    residual_path_bps: list[float] | tuple[float, ...] | np.ndarray, *, barrier_bps: float
) -> str:
    """Classify the first signed barrier crossing across exactly three completed closes."""

    values = np.asarray(residual_path_bps, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("onset classification requires three finite residual closes")
    if not np.isfinite(barrier_bps) or barrier_bps <= 0.0:
        raise ValueError("onset barrier must be positive and finite")
    for value in values:
        if value >= barrier_bps:
            return "UP_ONSET"
        if value <= -barrier_bps:
            return "DOWN_ONSET"
    return "NO_ONSET"


def relative_strength_acceleration(relative_last_3: float, relative_previous_3: float) -> float:
    """Return the preregistered change between adjacent relative-return windows."""

    values = np.asarray([relative_last_3, relative_previous_3], dtype=np.float64)
    if not np.isfinite(values).all():
        return float("nan")
    return float(relative_last_3 - relative_previous_3)


def activity_acceleration(
    activity_last_2: list[float] | tuple[float, ...] | np.ndarray,
    activity_previous_4: list[float] | tuple[float, ...] | np.ndarray,
) -> float:
    """Compare log1p means of two recent and four preceding activity proxies."""

    latest = np.asarray(activity_last_2, dtype=np.float64)
    previous = np.asarray(activity_previous_4, dtype=np.float64)
    if latest.shape != (2,) or previous.shape != (4,):
        raise ValueError("activity acceleration requires two recent and four prior bars")
    if not np.isfinite(latest).all() or not np.isfinite(previous).all():
        return float("nan")
    if bool((latest < 0.0).any()) or bool((previous < 0.0).any()):
        return float("nan")
    return float(np.log1p(latest.mean()) - np.log1p(previous.mean()))


def range_acceleration(
    range_last_2: list[float] | tuple[float, ...] | np.ndarray,
    range_previous_4: list[float] | tuple[float, ...] | np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return the safe ratio of mean recent range to mean preceding range."""

    latest = np.asarray(range_last_2, dtype=np.float64)
    previous = np.asarray(range_previous_4, dtype=np.float64)
    if latest.shape != (2,) or previous.shape != (4,):
        raise ValueError("range acceleration requires two recent and four prior bars")
    if not np.isfinite(latest).all() or not np.isfinite(previous).all():
        return float("nan")
    denominator = float(previous.mean())
    if denominator <= epsilon:
        return float("nan")
    return float(latest.mean() / denominator)


def directional_efficiency(
    returns: list[float] | tuple[float, ...] | np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[float, float]:
    """Return signed and absolute path efficiency or unavailable near zero."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size not in {3, 6} or not np.isfinite(values).all():
        return float("nan"), float("nan")
    denominator = float(np.abs(values).sum())
    if denominator <= epsilon:
        return float("nan"), float("nan")
    signed = float(values.sum() / denominator)
    return signed, abs(signed)


def progress_per_activity(
    cohort_relative_return_last_3: float,
    relative_activity_last_3: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Scale signed cohort-relative progress by positive relative activity."""

    values = np.asarray(
        [cohort_relative_return_last_3, relative_activity_last_3, epsilon],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or epsilon <= 0.0 or relative_activity_last_3 < 0.0:
        return float("nan")
    return float(cohort_relative_return_last_3 / max(relative_activity_last_3, epsilon))


def close_location_pressure(
    *,
    highs: list[float] | tuple[float, ...] | np.ndarray,
    lows: list[float] | tuple[float, ...] | np.ndarray,
    closes: list[float] | tuple[float, ...] | np.ndarray,
) -> dict[str, float]:
    """Summarize close location across the latest three completed bars."""

    high_values = np.asarray(highs, dtype=np.float64)
    low_values = np.asarray(lows, dtype=np.float64)
    close_values = np.asarray(closes, dtype=np.float64)
    if high_values.shape != (3,) or low_values.shape != (3,) or close_values.shape != (3,):
        raise ValueError("close-location pressure requires exactly three bars")
    widths = high_values - low_values
    if not np.isfinite(np.concatenate([high_values, low_values, close_values])).all() or bool(
        (widths <= 1e-12).any()
    ):
        return {
            "current_close_location": float("nan"),
            "mean_close_location_last_3": float("nan"),
            "upper_quartile_close_fraction_last_3": float("nan"),
            "lower_quartile_close_fraction_last_3": float("nan"),
        }
    locations = (close_values - low_values) / widths
    return {
        "current_close_location": float(locations[-1]),
        "mean_close_location_last_3": float(locations.mean()),
        "upper_quartile_close_fraction_last_3": float((locations >= 0.75).mean()),
        "lower_quartile_close_fraction_last_3": float((locations <= 0.25).mean()),
    }


def new_extreme_counts(
    highs: list[float] | tuple[float, ...] | np.ndarray,
    lows: list[float] | tuple[float, ...] | np.ndarray,
    *,
    latest: int = 3,
) -> tuple[int, int]:
    """Count new completed-bar highs and lows among the latest bars."""

    high_values = np.asarray(highs, dtype=np.float64)
    low_values = np.asarray(lows, dtype=np.float64)
    if (
        high_values.ndim != 1
        or high_values.shape != low_values.shape
        or len(high_values) <= latest
        or latest <= 0
        or not np.isfinite(np.concatenate([high_values, low_values])).all()
    ):
        raise ValueError("new-extreme counts require aligned finite opening histories")
    start = len(high_values) - latest
    new_highs = 0
    new_lows = 0
    for index in range(start, len(high_values)):
        if high_values[index] > np.max(high_values[:index]):
            new_highs += 1
        if low_values[index] < np.min(low_values[:index]):
            new_lows += 1
    return new_highs, new_lows


def opening_range_acceptance(
    *,
    closes: list[float] | tuple[float, ...] | np.ndarray,
    initial_high: float,
    initial_low: float,
) -> dict[str, float]:
    """Measure acceptance outside and return inside the initial three-bar range."""

    values = np.asarray(closes, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 3
        or not np.isfinite(values).all()
        or not np.isfinite(initial_high)
        or not np.isfinite(initial_low)
        or initial_high <= initial_low
    ):
        raise ValueError("opening-range acceptance requires a valid completed-close history")
    outside = (values > initial_high) | (values < initial_low)
    latest_inside = bool(initial_low <= values[-1] <= initial_high)
    prior_outside = bool(outside[:-1].any()) if len(values) > 1 else False
    return {
        "close_above_initial_3_high": float(values[-1] > initial_high),
        "close_below_initial_3_low": float(values[-1] < initial_low),
        "completed_closes_outside_initial_range": float(outside.sum()),
        "latest_close_returned_inside_initial_range": float(latest_inside and prior_outside),
    }


def confirmation_deltas(
    at_t: Mapping[str, float],
    at_t_plus_1: Mapping[str, float],
    *,
    new_high: bool,
    new_low: bool,
    favourable_retracement_bps: float,
    opening_range_acceptance_persisted: bool,
    predicted_direction_remained_same: bool,
) -> dict[str, float]:
    """Return the fixed compact changes known after completed bar t+1."""

    sources = (
        ("cohort_relative_return_bps", "change_cohort_relative_return_bps"),
        ("relative_strength_acceleration", "change_relative_strength_acceleration"),
        ("activity_shock", "change_activity_shock"),
        ("range_acceleration", "change_range_acceleration"),
        ("signed_efficiency_3", "change_signed_efficiency_3"),
        ("current_close_location", "change_close_location"),
    )
    missing = sorted(
        {source for source, _ in sources}.difference(at_t)
        | {source for source, _ in sources}.difference(at_t_plus_1)
    )
    if missing:
        raise ValueError(f"confirmation inputs missing: {missing}")
    output = {target: float(at_t_plus_1[source] - at_t[source]) for source, target in sources}
    output.update(
        {
            "new_high_at_t_plus_1": float(new_high),
            "new_low_at_t_plus_1": float(new_low),
            "favourable_retracement_bps": float(favourable_retracement_bps),
            "opening_range_acceptance_persisted": float(opening_range_acceptance_persisted),
            "predicted_direction_remained_same": float(predicted_direction_remained_same),
        }
    )
    if not np.isfinite(np.fromiter(output.values(), dtype=np.float64)).all():
        raise ValueError("confirmation deltas must be finite")
    return output


@dataclass(frozen=True, slots=True)
class SupportContractRepair:
    """Parent/admitted population hierarchy produced by the V0.1 repair."""

    annotated_rows: pd.DataFrame
    primary_rows: pd.DataFrame
    parent_slates: pd.DataFrame
    admitted_slates: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ConcentrationLeader:
    """The stock selected solely by admitted-row concentration."""

    symbol: str
    rows: int
    share: float


def apply_support_contract_repair(
    frame: pd.DataFrame,
    *,
    minimum_parent_stocks: int = 15,
    slate_column: str = "slate_id",
    symbol_column: str = "symbol",
    admission_column: str = "high_movement_admitted",
) -> SupportContractRepair:
    """Validate parent slates before admission and retain singleton admissions."""

    required = {slate_column, symbol_column, admission_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"support-repair columns missing: {missing}")
    if frame.empty or minimum_parent_stocks <= 0:
        raise ValueError("support repair requires rows and a positive parent minimum")
    rows = frame.copy()
    rows["parent_slate_id"] = rows[slate_column].astype(str)
    if rows[["parent_slate_id", symbol_column]].duplicated().any():
        raise ValueError("parent slate contains a duplicate stock")
    rows[admission_column] = rows[admission_column].astype(bool)
    grouped = rows.groupby("parent_slate_id", sort=True)
    rows["parent_valid_stock_count"] = grouped[symbol_column].transform("nunique").astype(int)
    rows["admitted_stock_count"] = grouped[admission_column].transform("sum").astype(int)
    rows["parent_slate_eligible"] = rows["parent_valid_stock_count"].ge(minimum_parent_stocks)
    rows["support_status"] = np.select(
        [
            ~rows["parent_slate_eligible"],
            rows["admitted_stock_count"].eq(0),
            rows["admitted_stock_count"].eq(1),
        ],
        [
            "parent_slate_insufficient_valid_stocks",
            "no_high_movement_admission",
            "valid_singleton_admission",
        ],
        default="valid_multi_candidate_admission",
    )
    rows["primary_eligible"] = rows["parent_slate_eligible"] & rows[admission_column]
    rows["row_weight"] = np.nan
    primary_mask = rows["primary_eligible"]
    rows.loc[primary_mask, "row_weight"] = 1.0 / rows.loc[
        primary_mask, "admitted_stock_count"
    ].to_numpy(dtype=np.float64)
    primary = rows.loc[rows["primary_eligible"]].copy()
    totals = primary.groupby("parent_slate_id", sort=True)["row_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=np.float64), 1.0, atol=1e-12):
        raise AssertionError("admitted-slate row weights do not sum to one")
    accounting_columns = [
        "parent_slate_id",
        "parent_valid_stock_count",
        "admitted_stock_count",
        "parent_slate_eligible",
        "support_status",
    ]
    for optional in ("session", "year", "year_month", "decision_ordinal"):
        if optional in rows.columns:
            accounting_columns.insert(-4, optional)
    accounting = (
        rows.loc[:, accounting_columns]
        .drop_duplicates("parent_slate_id", keep="first")
        .sort_values("parent_slate_id", kind="mergesort")
        .reset_index(drop=True)
    )
    parent_columns = [
        column for column in accounting.columns if column not in {"admitted_stock_count"}
    ]
    parent_slates = accounting.loc[:, parent_columns].copy()
    admitted_slates = accounting.copy()
    admitted_slates["primary_row_count"] = np.where(
        admitted_slates["parent_slate_eligible"],
        admitted_slates["admitted_stock_count"],
        0,
    ).astype(int)
    admitted_slates["singleton_admitted_slate"] = admitted_slates[
        "parent_slate_eligible"
    ] & admitted_slates["admitted_stock_count"].eq(1)
    admitted_slates["multi_candidate_admitted_slate"] = admitted_slates[
        "parent_slate_eligible"
    ] & admitted_slates["admitted_stock_count"].ge(2)
    return SupportContractRepair(
        annotated_rows=rows,
        primary_rows=primary,
        parent_slates=parent_slates,
        admitted_slates=admitted_slates,
    )


def annotate_economic_selection_semantics(
    selections: pd.DataFrame,
    admitted_counts: Mapping[str, int],
    *,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Label singleton selections without removing their realised result."""

    if slate_column not in selections:
        raise ValueError("economic selections lack parent-slate identity")
    output = selections.copy()
    output["admitted_stock_count"] = output[slate_column].astype(str).map(admitted_counts)
    if output["admitted_stock_count"].isna().any():
        raise ValueError("economic selection lacks admitted-stock accounting")
    output["admitted_stock_count"] = output["admitted_stock_count"].astype(int)
    if output["admitted_stock_count"].lt(1).any():
        raise ValueError("economic selection requires at least one admitted stock")
    output["admitted_slate_type"] = np.where(
        output["admitted_stock_count"].eq(1), "singleton", "multi_candidate"
    )
    output["within_admitted_comparison_status"] = np.where(
        output["admitted_stock_count"].eq(1),
        "degenerate_singleton",
        "competitive_multi_candidate",
    )
    return output


def largest_admitted_stock(
    frame: pd.DataFrame, *, symbol_column: str = "symbol"
) -> ConcentrationLeader:
    """Select the fixed deletion stock from admitted-row share only."""

    if frame.empty or symbol_column not in frame:
        raise ValueError("largest-stock selection requires admitted rows and symbols")
    counts = frame[symbol_column].astype(str).value_counts(sort=False)
    ordered = (
        counts.rename_axis("symbol")
        .reset_index(name="rows")
        .sort_values(["rows", "symbol"], ascending=[False, True], kind="mergesort")
    )
    leader = ordered.iloc[0]
    return ConcentrationLeader(
        symbol=str(leader["symbol"]),
        rows=int(leader["rows"]),
        share=float(int(leader["rows"]) / len(frame)),
    )


def concentration_aware_decision(
    base_decision: str,
    *,
    maximum_admitted_row_share: float,
    deletion_same_signed_conclusions: bool,
    principal_increments_non_negative: bool,
    no_material_adversity: bool,
    economic_not_dominated: bool,
) -> str:
    """Apply the repaired post-fit concentration rule without blocking fitting."""

    if base_decision not in DECISIONS:
        raise ValueError("unknown pressure-screen decision")
    share = float(maximum_admitted_row_share)
    if not np.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("maximum admitted-row share must lie in [0, 1]")
    positive = {
        "pressure_onset_and_direction_increment_observed",
        "pressure_onset_occurrence_only",
        "directional_pressure_only",
        "one_bar_confirmation_required",
    }
    if base_decision not in positive or share <= 0.10 + 1e-15:
        return base_decision
    stress_passes = all(
        (
            deletion_same_signed_conclusions,
            principal_increments_non_negative,
            no_material_adversity,
            economic_not_dominated,
        )
    )
    return (
        base_decision if stress_passes else "pressure_signal_observed_but_concentration_gate_failed"
    )


def equal_slate_weights(slate_ids: pd.Series) -> FloatArray:
    """Give every represented simultaneous slate total model weight one."""

    values = slate_ids.astype(str).reset_index(drop=True)
    if values.empty or values.isna().any():
        raise ValueError("slate weights require complete identifiers")
    sizes = values.groupby(values, sort=True).transform("size").to_numpy(dtype=np.float64)
    weights = np.asarray(1.0 / sizes, dtype=np.float64)
    totals = pd.Series(weights).groupby(values, sort=True).sum().to_numpy(dtype=np.float64)
    if not np.allclose(totals, 1.0, atol=1e-12):
        raise AssertionError("slate weights do not sum to one")
    return weights


@dataclass(frozen=True, slots=True)
class FrozenLogisticModel:
    """JSON-serializable standardized deterministic L2 logistic model."""

    model_id: str
    feature_names: tuple[str, ...]
    means: FloatArray
    scales: FloatArray
    coefficients: FloatArray
    intercept: float
    training_rows: int
    training_slates: int
    iterations: int
    converged: bool

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        values = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.model_id} prediction features are not finite")
        linear = self.intercept + ((values - self.means) / self.scales) @ self.coefficients
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": "logistic",
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "training_rows": self.training_rows,
            "training_slates": self.training_slates,
            "iterations": self.iterations,
            "converged": self.converged,
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        }


def fit_fixed_logistic(
    frame: pd.DataFrame,
    target: Sequence[int] | pd.Series,
    *,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
    sample_weight_column: str | None = None,
) -> FrozenLogisticModel:
    """Fit the preregistered fixed logistic model with equal-slate weights."""

    names = tuple(features)
    values = frame.loc[:, list(names)].to_numpy(dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_id} training features are not finite")
    if labels.shape != (len(frame),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model_id} requires both aligned binary classes")
    if sample_weight_column is None:
        sample_weights = equal_slate_weights(frame[slate_column])
    else:
        if sample_weight_column not in frame:
            raise ValueError(f"{model_id} sample-weight column is missing")
        sample_weights = frame[sample_weight_column].to_numpy(dtype=np.float64)
        if not np.isfinite(sample_weights).all() or bool((sample_weights <= 0.0).any()):
            raise ValueError(f"{model_id} sample weights must be positive and finite")
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        labels,
        sample_weight=sample_weights,
    )
    iterations = int(np.max(estimator.n_iter_))
    if iterations >= 250:
        raise RuntimeError(f"{model_id} failed to converge")
    return FrozenLogisticModel(
        model_id=model_id,
        feature_names=names,
        means=means,
        scales=scales,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        training_rows=len(frame),
        training_slates=int(frame[slate_column].astype(str).nunique()),
        iterations=iterations,
        converged=True,
    )


def manual_logistic_prediction(model: Mapping[str, Any], frame: pd.DataFrame) -> FloatArray:
    """Reconstruct probabilities directly from serialized model parameters."""

    names = [str(value) for value in model["feature_names"]]
    values = frame.loc[:, names].to_numpy(dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (
        values.shape[1] != len(means)
        or means.shape != scales.shape
        or means.shape != coefficients.shape
    ):
        raise ValueError("serialized logistic dimensions do not align")
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def expanding_monthly_oof_probabilities(
    frame: pd.DataFrame,
    *,
    target_column: str,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Score July onward with monthly expanding models trained only on earlier rows."""

    if "year_month" not in frame or target_column not in frame:
        raise ValueError("monthly OOF frame is missing required columns")
    months = sorted(frame["year_month"].astype(str).unique())
    score_months = [month for month in months if "2024-07" <= month <= "2024-12"]
    probabilities = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    manifest: list[dict[str, Any]] = []
    for score_month in score_months:
        training = frame.loc[frame["year_month"].astype(str).lt(score_month)].copy()
        scoring = frame.loc[frame["year_month"].astype(str).eq(score_month)].copy()
        if training.empty or scoring.empty or training["year_month"].astype(str).max() < "2024-06":
            raise ValueError(f"insufficient initial expanding history for {score_month}")
        model = fit_fixed_logistic(
            training,
            training[target_column],
            features=features,
            slate_column=slate_column,
            model_id=f"{model_id}__score_{score_month}",
        )
        probabilities.loc[scoring.index] = model.predict(scoring)
        manifest.append(
            {
                "score_month": score_month,
                "training_start_month": str(training["year_month"].astype(str).min()),
                "training_end_month": str(training["year_month"].astype(str).max()),
                "training_rows": len(training),
                "scored_rows": len(scoring),
                "model": model.as_dict(),
            }
        )
    return probabilities, manifest


def movement_admission_thresholds(frame: pd.DataFrame) -> dict[int, float]:
    """Freeze checkpoint q75 admission thresholds from finite 2024 OOF scores only."""

    required = {"year", "decision_ordinal", "p_large_remaining_move"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"movement admission columns missing: {missing}")
    development = frame.loc[pd.to_numeric(frame["year"], errors="raise").eq(2024)]
    output: dict[int, float] = {}
    for ordinal in DECISION_ORDINALS:
        values = pd.to_numeric(
            development.loc[development["decision_ordinal"].eq(ordinal), "p_large_remaining_move"],
            errors="coerce",
        ).dropna()
        if values.empty or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"no finite movement OOF scores at checkpoint {ordinal}")
        output[ordinal] = float(values.quantile(0.75, interpolation="linear"))
    return output


@dataclass(frozen=True, slots=True)
class SessionBootstrapDraw:
    """One deterministic whole-session bootstrap sample."""

    draw: int
    sampled_sessions: tuple[str, ...]


def session_block_bootstrap_draws(
    sessions: Sequence[str], *, draws: int, seed: int
) -> tuple[SessionBootstrapDraw, ...]:
    """Sample complete session identifiers with replacement."""

    unique = tuple(sorted({str(session) for session in sessions}))
    if not unique or draws <= 0:
        raise ValueError("session bootstrap requires sessions and positive draws")
    rng = np.random.default_rng(seed)
    output: list[SessionBootstrapDraw] = []
    for draw in range(draws):
        sampled = tuple(str(value) for value in rng.choice(unique, size=len(unique), replace=True))
        output.append(SessionBootstrapDraw(draw=draw, sampled_sessions=sampled))
    return tuple(output)


def permute_feature_bundle_within_slates(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    seed: int,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute a complete feature bundle together among stocks within each slate."""

    names = list(feature_names)
    missing = sorted({slate_column, *names}.difference(frame.columns))
    if missing or not names:
        raise ValueError(f"permutation columns missing: {missing}")
    rng = np.random.default_rng(seed)
    output = frame.copy()
    for _, index in output.groupby(slate_column, sort=True).groups.items():
        positions = np.asarray(list(index))
        permutation = rng.permutation(len(positions))
        source = frame.loc[positions, names].to_numpy(copy=True)[permutation]
        output.loc[positions, names] = source
    return output


def assert_safe_timestamps(timestamps: pd.Series | Sequence[object]) -> None:
    """Fail closed if any materialized market timestamp reaches the protected boundary."""

    values = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    if values.empty:
        raise ValueError("market timestamp audit is empty")
    if values.ge(PROTECTED_START).any():
        raise ValueError("protected market row materialised")


def assert_allowed_feature_names(feature_names: Sequence[str]) -> None:
    """Reject every retired or forbidden structural feature-name fragment."""

    forbidden = sorted(
        name
        for name in {str(value) for value in feature_names}
        if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"forbidden feature names: {forbidden}")


def decide_pressure_screen(evidence: Mapping[str, Any]) -> str:
    """Apply the preregistered pressure-onset decision-category precedence."""

    blocker = evidence.get("integrity_blocker")
    if blocker is not None:
        if blocker not in DECISIONS or not str(blocker).startswith("blocked_"):
            raise ValueError("unregistered integrity blocker")
        return str(blocker)
    occurrence = bool(evidence.get("occurrence_passes", False))
    direction = bool(evidence.get("direction_passes", False))
    if occurrence and direction:
        return "pressure_onset_and_direction_increment_observed"
    if occurrence:
        return "pressure_onset_occurrence_only"
    if direction:
        return "directional_pressure_only"
    if bool(evidence.get("confirmation_occurrence_passes", False)) or bool(
        evidence.get("confirmation_direction_passes", False)
    ):
        return "one_bar_confirmation_required"
    if bool(evidence.get("readiness_useful", False)):
        return "movement_readiness_but_direction_unresolved"
    return "no_pressure_onset_increment"
