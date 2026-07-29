"""Causal stock-local mechanics for Directional Archetype Screen V0.

The module contains retrospective research utilities only.  It has no broker,
order-placement, option-P&L, or production-runtime surface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

RESEARCH_ONLY: Final[bool] = True
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-01")
PRIMARY_DIRECTION_HORIZON_MINUTES: Final[int] = 10
MINIMUM_EPISODE_SPACING_MINUTES: Final[int] = 30
EPSILON: Final[float] = 1e-12
CHECKPOINT_GROUPS: Final[tuple[str, ...]] = ("early", "middle", "late")
BASELINE_FEATURES: Final[tuple[str, ...]] = (
    "b_stock_return_5m",
    "b_stock_return_10m",
    "b_market_return_10m",
    "b_relative_return_10m",
    "b_distance_from_vwap",
)
CONTINUATION_FEATURES: Final[tuple[str, ...]] = (
    "c_z_return_5m",
    "c_z_return_10m",
    "c_z_return_20m",
    "c_z_return_30m",
    "c_directional_efficiency_20m",
    "c_mean_clv_4",
    "c_directional_close_fraction_4",
    "c_signed_wick_asymmetry_4",
    "c_signed_vwap_slope_4",
    "c_signed_vwap_distance",
    "c_vwap_side_closes_3",
    "c_break_above_prior_six_high",
    "c_break_below_prior_six_low",
    "c_signed_boundary_distance",
    "c_signed_boundary_acceptance_count",
    "c_boundary_rejection",
    "c_relative_return_5m",
    "c_relative_return_10m",
    "c_relative_agreement",
)
ABSORPTION_FEATURES: Final[tuple[str, ...]] = (
    "a_attempt_return_abs",
    "a_attempt_path_length",
    "a_attempt_directional_efficiency",
    "a_response_followthrough",
    "a_reversal_efficiency_change",
    "a_wick_rejection",
    "a_close_location_recovery",
    "a_failure_close_near_extreme",
    "a_boundary_failure",
    "a_boundary_distance_inside",
    "a_boundary_maintenance_count",
    "a_vwap_reclaim_failure",
    "a_vwap_distance_after_failure",
    "a_attempt_price_impact",
    "a_response_price_impact",
    "a_price_impact_decline",
    "a_elevated_activity_weak_progress",
    "a_relative_recovery",
    "a_market_resilience",
)
RELATIVE_STRENGTH_FEATURES: Final[tuple[str, ...]] = (
    "r_residual_return_5m",
    "r_residual_return_10m",
    "r_residual_return_20m",
    "r_residual_slope",
    "r_residual_persistence",
    "r_change_in_residual_strength",
    "r_stock_flat_up_market_down",
    "r_stock_flat_down_market_up",
    "r_residual_volatility_score",
    "r_distance_from_normal_residual_range",
    "r_absolute_residual_direction_agreement",
    "r_improving_while_absolute_compressed",
)


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def transitive_descendants(
    graph: Mapping[str, Sequence[str]],
    roots: Sequence[str],
) -> tuple[str, ...]:
    """Return the deterministic transitive descendants of declared roots."""

    frontier = list(dict.fromkeys(str(root) for root in roots))
    visited = set(frontier)
    descendants: set[str] = set()
    while frontier:
        node = frontier.pop()
        for child_value in graph.get(node, ()):
            child = str(child_value)
            descendants.add(child)
            if child not in visited:
                visited.add(child)
                frontier.append(child)
    descendants.difference_update(str(root) for root in roots)
    return tuple(sorted(descendants))


def build_movement_dependency_audit(
    *,
    graph: Mapping[str, Sequence[str]],
    contaminated_roots: Sequence[str],
    group_i_features: Sequence[str],
    peer_normalised_features: Sequence[str],
) -> dict[str, object]:
    """Classify frozen Group-I inputs through the complete dependency graph."""

    contaminated = set(transitive_descendants(graph, contaminated_roots))
    group_i = tuple(str(feature) for feature in group_i_features)
    contaminated_group_i = sorted(contaminated.intersection(group_i))
    peer_group_i = sorted(
        set(str(value) for value in peer_normalised_features).intersection(group_i)
    )
    excluded = set(contaminated_group_i).union(peer_group_i)
    causal = sorted(set(group_i).difference(excluded))
    return {
        "research_only": True,
        "dependency_graph": {
            str(parent): sorted(str(child) for child in children)
            for parent, children in sorted(graph.items())
        },
        "contaminated_roots": sorted(str(root) for root in contaminated_roots),
        "transitive_contaminated_descendants": sorted(contaminated),
        "contaminated_group_i_features": contaminated_group_i,
        "peer_normalised_group_i_features": peer_group_i,
        "excluded_group_i_features": sorted(excluded),
        "causal_group_i_features": causal,
        "archived_m1_numerically_affected": bool(contaminated_group_i),
        "m1c_required": bool(contaminated_group_i),
        "archived_signed_pressure_excluded": "signed_pressure" in excluded,
        "cross_sectional_peer_normalisation": False,
    }


def weighted_quantile(
    values: Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray,
    quantile: float,
) -> float:
    """Return the frozen deterministic midpoint-CDF weighted quantile."""

    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 1
        or mass.ndim != 1
        or len(data) == 0
        or len(data) != len(mass)
        or not np.isfinite(data).all()
        or not np.isfinite(mass).all()
        or bool((mass <= 0.0).any())
        or not 0.0 <= quantile <= 1.0
    ):
        raise ValueError("weighted quantile requires finite values and positive aligned weights")
    order = np.argsort(data, kind="mergesort")
    sorted_values = data[order]
    sorted_weights = mass[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return float(
        np.interp(
            quantile,
            positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def assign_stock_local_session_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Weight checkpoints using only their own stock-session support."""

    required = {"stock", "session", "checkpoint"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"stock-local weight inputs missing: {missing}")
    if frame.empty:
        raise ValueError("stock-local weights need at least one checkpoint")
    if frame.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("stock-local weight checkpoint identities must be unique")
    output = frame.copy()
    counts = output.groupby(["stock", "session"], sort=False)["checkpoint"].transform("size")
    output["stock_local_checkpoints_in_session"] = counts.astype(int)
    output["row_weight"] = 1.0 / counts.astype(float)
    totals = output.groupby(["stock", "session"], sort=False)["row_weight"].sum().to_numpy(float)
    if not np.allclose(totals, 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("stock-local session weights do not sum to one")
    return output


def reject_protected_sessions(
    sessions: Sequence[object] | pd.Series,
    *,
    protected_start: pd.Timestamp = PROTECTED_START,
) -> None:
    """Fail closed when any supplied session reaches the protected boundary."""

    parsed = pd.to_datetime(pd.Series(sessions), errors="raise", utc=True)
    boundary = pd.Timestamp(protected_start)
    boundary = (
        boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    )
    if bool(parsed.ge(boundary).any()):
        raise ValueError("protected session must not be read or materialised")


def construct_fresh_episodes(
    checkpoint_rows: pd.DataFrame,
    *,
    threshold: float,
    probability_column: str = "movement_probability",
    minimum_spacing_minutes: int = MINIMUM_EPISODE_SPACING_MINUTES,
) -> pd.DataFrame:
    """Select threshold crossings and enforce fixed stock-session spacing."""

    required = {
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
        probability_column,
        "partition",
    }
    missing = sorted(required.difference(checkpoint_rows.columns))
    if missing:
        raise ValueError(f"episode inputs missing: {missing}")
    if not math.isfinite(threshold):
        raise ValueError("movement threshold must be finite")
    if minimum_spacing_minutes != MINIMUM_EPISODE_SPACING_MINUTES:
        raise ValueError("fresh episodes require thirty-minute spacing")
    reject_protected_sessions(checkpoint_rows["session"])

    ordered = checkpoint_rows.copy()
    ordered["signal_timestamp"] = pd.to_datetime(
        ordered["signal_timestamp"], utc=True, errors="raise"
    )
    ordered["prospective_entry_timestamp"] = pd.to_datetime(
        ordered["prospective_entry_timestamp"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    if ordered.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("checkpoint identity must be unique")
    probabilities = pd.to_numeric(ordered[probability_column], errors="raise").to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("movement probabilities must be finite")
    ordered["above_frozen_threshold"] = probabilities >= threshold
    ordered["previous_checkpoint_probability"] = ordered.groupby(["stock", "session"], sort=False)[
        probability_column
    ].shift()
    ordered["fresh_crossing"] = ordered["above_frozen_threshold"] & (
        ordered["previous_checkpoint_probability"].isna()
        | ordered["previous_checkpoint_probability"].lt(threshold)
    )

    selected: list[int] = []
    episode_numbers: dict[int, int] = {}
    minutes_since_previous: dict[int, float] = {}
    for _, group in ordered.loc[ordered["fresh_crossing"]].groupby(["stock", "session"], sort=True):
        previous_start: pd.Timestamp | None = None
        episode_number = 0
        for index, row in group.iterrows():
            current_start = _as_timestamp(row["signal_timestamp"])
            elapsed = (
                None
                if previous_start is None
                else (current_start - previous_start).total_seconds() / 60.0
            )
            if elapsed is not None and elapsed < minimum_spacing_minutes:
                continue
            integer_index = _as_int(index)
            selected.append(integer_index)
            episode_number += 1
            episode_numbers[integer_index] = episode_number
            minutes_since_previous[integer_index] = math.nan if elapsed is None else float(elapsed)
            previous_start = current_start

    episodes = ordered.loc[selected].copy()
    episodes["episode_number"] = [episode_numbers[index] for index in selected]
    episodes["minutes_since_previous_episode"] = [
        minutes_since_previous[index] for index in selected
    ]
    episodes["trigger_bar_ordinal"] = episodes["checkpoint"].astype(int) - 1
    episodes["marker_bar_ordinal"] = episodes["checkpoint"].astype(int) - 2
    episodes["direction_marker_bar"] = "T-1"
    episodes["trigger_bar_excluded_from_direction_features"] = True
    return episodes.reset_index(drop=True)


def checkpoint_group(checkpoint: int) -> str:
    """Map the frozen checkpoint grid to one broad intraday group."""

    value = int(checkpoint)
    if 6 <= value <= 14:
        return "early"
    if 16 <= value <= 24:
        return "middle"
    if 26 <= value <= 34:
        return "late"
    raise ValueError(f"checkpoint outside the frozen grid: {value}")


@dataclass(frozen=True)
class _RobustFit:
    median: float
    iqr: float
    clip_lower: float
    clip_upper: float
    missing_value: float
    zero_scale: bool
    support: int


def _fit_robust(values: pd.Series) -> _RobustFit | None:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = raw[np.isfinite(raw)]
    if not len(finite):
        return None
    median = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr = float(q75 - q25)
    zero_scale = not math.isfinite(iqr) or iqr <= EPSILON
    lower, upper = np.quantile(finite, [0.01, 0.99])
    return _RobustFit(
        median=median,
        iqr=1.0 if zero_scale else iqr,
        clip_lower=float(lower),
        clip_upper=float(upper),
        missing_value=median,
        zero_scale=zero_scale,
        support=int(len(finite)),
    )


def _parameter_row(
    *,
    feature: str,
    stock: str,
    checkpoint: int,
    group: str,
    fallback_level: str,
    source_group: str,
    fit: _RobustFit,
) -> dict[str, object]:
    return {
        "feature": feature,
        "stock": stock,
        "checkpoint": checkpoint,
        "checkpoint_group": group,
        "fallback_level": fallback_level,
        "source_checkpoint_group": source_group,
        "support": fit.support,
        "median": fit.median,
        "iqr": fit.iqr,
        "clip_lower": fit.clip_lower,
        "clip_upper": fit.clip_upper,
        "missing_value": fit.missing_value,
        "zero_scale": fit.zero_scale,
    }


def fit_stock_local_normalisation(
    development: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    minimum_support: int = 20,
    excluded_sessions: Sequence[str] = (),
) -> pd.DataFrame:
    """Fit development-only stock-local robust transforms."""

    required = {"stock", "session", "checkpoint", *feature_columns}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"normalisation inputs missing: {missing}")
    if minimum_support != 20:
        raise ValueError("minimum stock-local normalisation support is frozen at 20")
    excluded = tuple(str(value) for value in excluded_sessions)
    frame = development.loc[~development["session"].astype(str).isin(excluded)].copy()
    dates = pd.to_datetime(frame["session"], errors="raise")
    if len(frame) and not dates.dt.year.eq(2024).all():
        raise ValueError("stock-local normalisation must fit on 2024 only")
    if frame.empty:
        raise ValueError("stock-local normalisation needs development rows")
    frame["_checkpoint_group"] = frame["checkpoint"].map(lambda value: checkpoint_group(int(value)))
    positions = {name: index for index, name in enumerate(CHECKPOINT_GROUPS)}
    rows: list[dict[str, object]] = []
    stocks = sorted(frame["stock"].astype(str).unique())
    checkpoints = sorted(frame["checkpoint"].astype(int).unique())
    for feature in feature_columns:
        pooled_fit = _fit_robust(frame[feature])
        if pooled_fit is None or pooled_fit.support < minimum_support:
            raise ValueError(f"development pooled feature lacks support: {feature}")
        rows.append(
            _parameter_row(
                feature=str(feature),
                stock="__POOLED__",
                checkpoint=-1,
                group="pooled",
                fallback_level="development_pooled",
                source_group="pooled",
                fit=pooled_fit,
            )
        )
        for stock in stocks:
            stock_rows = frame.loc[frame["stock"].astype(str).eq(stock)]
            stock_fit = _fit_robust(stock_rows[feature])
            for checkpoint in checkpoints:
                target_group = checkpoint_group(checkpoint)
                exact = stock_rows.loc[stock_rows["checkpoint"].astype(int).eq(checkpoint)]
                fitted = _fit_robust(exact[feature])
                level = "stock_checkpoint"
                source_group = target_group
                if fitted is None or fitted.support < minimum_support:
                    adjacent_names = sorted(
                        (
                            name
                            for name in CHECKPOINT_GROUPS
                            if abs(positions[name] - positions[target_group]) == 1
                        ),
                        key=lambda name: (
                            abs(positions[name] - positions[target_group]),
                            positions[name],
                        ),
                    )
                    fitted = None
                    for adjacent_name in adjacent_names:
                        adjacent = stock_rows.loc[
                            stock_rows["_checkpoint_group"].astype(str).eq(adjacent_name)
                        ]
                        candidate = _fit_robust(adjacent[feature])
                        if candidate is not None and candidate.support >= minimum_support:
                            fitted = candidate
                            level = "stock_adjacent_checkpoint_group"
                            source_group = adjacent_name
                            break
                if fitted is None or fitted.support < minimum_support:
                    if stock_fit is not None and stock_fit.support >= minimum_support:
                        fitted = stock_fit
                        level = "stock_all_checkpoints"
                        source_group = "all"
                    else:
                        fitted = pooled_fit
                        level = "development_pooled"
                        source_group = "pooled"
                rows.append(
                    _parameter_row(
                        feature=str(feature),
                        stock=stock,
                        checkpoint=checkpoint,
                        group=target_group,
                        fallback_level=level,
                        source_group=source_group,
                        fit=fitted,
                    )
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "stock", "checkpoint"], kind="mergesort")
        .reset_index(drop=True)
    )


def apply_stock_local_normalisation(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen transforms and return a row-level fallback audit."""

    required = {
        "feature",
        "stock",
        "checkpoint",
        "fallback_level",
        "median",
        "iqr",
        "clip_lower",
        "clip_upper",
        "missing_value",
    }
    missing = sorted(required.difference(parameters.columns))
    if missing:
        raise ValueError(f"normalisation parameters missing: {missing}")
    output = frame.copy()
    exact = parameters.set_index(["feature", "stock", "checkpoint"], drop=False)
    pooled = parameters.loc[parameters["stock"].astype(str).eq("__POOLED__")].set_index(
        "feature", drop=False
    )
    audit_rows: list[dict[str, object]] = []
    for feature in feature_columns:
        transformed: list[float] = []
        for row_index, row in output.iterrows():
            key = (str(feature), str(row["stock"]), int(row["checkpoint"]))
            exact_locator = cast(Any, exact.loc)
            fitted = cast(
                pd.Series,
                exact_locator[key] if key in exact.index else pooled.loc[str(feature)],
            )
            raw = float(row[feature]) if pd.notna(row[feature]) else math.nan
            missing_value_used = not math.isfinite(raw)
            value = float(fitted["missing_value"]) if missing_value_used else raw
            clipped = float(
                np.clip(value, float(fitted["clip_lower"]), float(fitted["clip_upper"]))
            )
            transformed.append((clipped - float(fitted["median"])) / float(fitted["iqr"]))
            audit_rows.append(
                {
                    "row_index": str(row_index),
                    "stock": str(row["stock"]),
                    "session": str(row["session"]),
                    "checkpoint": int(row["checkpoint"]),
                    "feature": str(feature),
                    "fallback_level": str(fitted["fallback_level"]),
                    "source_checkpoint_group": str(fitted["source_checkpoint_group"]),
                    "support": int(fitted["support"]),
                    "missing_value_used": missing_value_used,
                    "clipped": bool(value != clipped),
                }
            )
        output[feature] = np.asarray(transformed, dtype=np.float64)
    return output, pd.DataFrame(audit_rows)


def fit_stock_market_betas(
    development_bars: pd.DataFrame,
    *,
    minimum_support: int = 20,
    excluded_sessions: Sequence[str] = (),
) -> pd.DataFrame:
    """Fit deterministic stock-specific OLS market sensitivity on 2024 bars."""

    required = {
        "stock",
        "session",
        "checkpoint_group",
        "stock_return",
        "market_return",
    }
    missing = sorted(required.difference(development_bars.columns))
    if missing:
        raise ValueError(f"beta inputs missing: {missing}")
    excluded = tuple(str(value) for value in excluded_sessions)
    frame = development_bars.loc[~development_bars["session"].astype(str).isin(excluded)].copy()
    dates = pd.to_datetime(frame["session"], errors="raise")
    if len(frame) and not dates.dt.year.eq(2024).all():
        raise ValueError("stock-market beta fitting must use 2024 only")
    if frame.empty:
        raise ValueError("stock-market beta fitting needs development bars")

    def fit(
        rows: pd.DataFrame,
    ) -> tuple[float, float, float, float, float, float, int] | None:
        stock_values = pd.to_numeric(rows["stock_return"], errors="coerce").to_numpy(float)
        market_values = pd.to_numeric(rows["market_return"], errors="coerce").to_numpy(float)
        valid = np.isfinite(stock_values) & np.isfinite(market_values)
        stock_values = stock_values[valid]
        market_values = market_values[valid]
        if len(stock_values) < minimum_support:
            return None
        design = np.column_stack([np.ones(len(market_values)), market_values])
        coefficients, _, _, _ = np.linalg.lstsq(design, stock_values, rcond=None)
        residuals = stock_values - design @ coefficients
        residual_scale = float(np.std(residuals, ddof=0))
        if not math.isfinite(residual_scale) or residual_scale <= EPSILON:
            residual_scale = 1.0
        low, high = np.quantile(residuals, [0.10, 0.90])
        return (
            float(coefficients[0]),
            float(coefficients[1]),
            residual_scale,
            float(low),
            float(high),
            float(np.median(np.abs(stock_values))),
            int(len(stock_values)),
        )

    pooled_fit = fit(frame)
    if pooled_fit is None:
        raise ValueError("pooled beta fit lacks development support")
    rows: list[dict[str, object]] = []
    for stock in sorted(frame["stock"].astype(str).unique()):
        stock_rows = frame.loc[frame["stock"].astype(str).eq(stock)]
        stock_fit = fit(stock_rows)
        for group in CHECKPOINT_GROUPS:
            group_rows = stock_rows.loc[stock_rows["checkpoint_group"].astype(str).eq(group)]
            fitted = fit(group_rows)
            level = "stock_checkpoint_group"
            if fitted is None:
                fitted = stock_fit
                level = "stock_all_checkpoints"
            if fitted is None:
                fitted = pooled_fit
                level = "development_pooled"
            alpha, beta, scale, low, high, stock_abs_median, support = fitted
            rows.append(
                {
                    "stock": stock,
                    "checkpoint_group": group,
                    "alpha": alpha,
                    "beta": beta,
                    "residual_scale": scale,
                    "residual_range_low": low,
                    "residual_range_high": high,
                    "stock_abs_return_median": stock_abs_median,
                    "support": support,
                    "fallback_level": level,
                    "development_start": str(dates.min().date()),
                    "development_end": str(dates.max().date()),
                    "excluded_sessions": ",".join(sorted(excluded)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["stock", "checkpoint_group"], kind="mergesort")
        .reset_index(drop=True)
    )


def directional_efficiency(returns: Sequence[float] | FloatArray) -> float:
    """Signed net movement divided by completed-bar path length."""

    values = np.asarray(returns, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return math.nan
    return float(np.sum(values) / (np.sum(np.abs(values)) + EPSILON))


def mirrored_wick_rejection(
    *,
    attempt_sign: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> float:
    """Return a mirrored bar-derived wick response; positive predicts UP."""

    if attempt_sign not in (-1, 0, 1):
        raise ValueError("attempt sign must be -1, 0, or 1")
    if not all(math.isfinite(value) for value in (open_price, high, low, close)):
        return math.nan
    if high < low:
        return math.nan
    lower = min(open_price, close) - low
    upper = high - max(open_price, close)
    return float((lower - upper) / (high - low + EPSILON))


def mirrored_boundary_failure(
    *,
    attempt_sign: int,
    attempted_extreme: float,
    boundary: float,
    response_close: float,
) -> float:
    """Encode downside reclaim as positive and upside rejection as negative."""

    if attempt_sign not in (-1, 0, 1):
        raise ValueError("attempt sign must be -1, 0, or 1")
    if not all(math.isfinite(value) for value in (attempted_extreme, boundary, response_close)):
        return math.nan
    scale = abs(boundary) + EPSILON
    if attempt_sign < 0 and attempted_extreme < boundary and response_close > boundary:
        return float((response_close - boundary) / scale)
    if attempt_sign > 0 and attempted_extreme > boundary and response_close < boundary:
        return float(-(boundary - response_close) / scale)
    return 0.0


def beta_adjusted_residual(
    *,
    stock_return: float,
    market_return: float,
    alpha: float,
    beta: float,
    bars: int,
) -> float:
    """Calculate a frozen-beta cumulative residual return."""

    if bars <= 0:
        raise ValueError("residual horizon must contain completed bars")
    if not all(math.isfinite(value) for value in (stock_return, market_return, alpha, beta)):
        return math.nan
    return float(stock_return - (bars * alpha + beta * market_return))


def activity_price_impact(
    absolute_return: float,
    mean_historical_relative_activity: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Scale absolute price response by the historical activity proxy."""

    if not all(
        math.isfinite(value) for value in (absolute_return, mean_historical_relative_activity)
    ):
        return math.nan
    if absolute_return < 0.0 or mean_historical_relative_activity < 0.0:
        raise ValueError("activity impact inputs must be non-negative")
    return float(absolute_return / (mean_historical_relative_activity + epsilon))


def residual_persistence(residual_returns: Sequence[float] | FloatArray) -> float:
    """Return mean residual sign; exact zero contributes zero."""

    values = np.asarray(residual_returns, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return math.nan
    return float(np.mean(np.sign(values)))


def _finite_sum(values: NDArray[np.float64]) -> float:
    return float(np.sum(values)) if len(values) and np.isfinite(values).all() else math.nan


def _ols_slope(values: NDArray[np.float64]) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return math.nan
    x_values = np.arange(len(values), dtype=np.float64)
    centered = x_values - float(np.mean(x_values))
    denominator = float(np.sum(centered**2))
    return (
        float(np.sum(centered * (values - np.mean(values))) / denominator)
        if denominator > 0.0
        else 0.0
    )


def prepare_completed_bars(completed_bars: pd.DataFrame) -> pd.DataFrame:
    """Construct causal stock-local primitives from completed five-minute bars."""

    required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "historical_relative_activity",
        "vti__bar_log_return",
    }
    missing = sorted(required.difference(completed_bars.columns))
    if missing:
        raise ValueError(f"completed-bar inputs missing: {missing}")
    reject_protected_sessions(completed_bars["session"])
    bars = completed_bars.copy()
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    bars = bars.sort_values(["stock", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ValueError("completed-bar identity must be unique")
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    previous_close = bars.groupby("stock", sort=False)["close"].shift()
    if "bar_log_return" in bars:
        source_return = pd.to_numeric(bars["bar_log_return"], errors="coerce")
        calculated = np.log(bars["close"] / previous_close)
        bars["_stock_return"] = source_return.where(source_return.notna(), calculated)
    else:
        bars["_stock_return"] = np.log(bars["close"] / previous_close)
    bars["_market_return"] = pd.to_numeric(bars["vti__bar_log_return"], errors="coerce")
    bars["_relative_return"] = bars["_stock_return"] - bars["_market_return"]
    bars["_normalised_range"] = (bars["high"] - bars["low"]) / previous_close
    denominator = bars["high"] - bars["low"] + EPSILON
    bars["_clv"] = (2.0 * bars["close"] - bars["high"] - bars["low"]) / denominator
    lower_wick = np.minimum(bars["open"], bars["close"]) - bars["low"]
    upper_wick = bars["high"] - np.maximum(bars["open"], bars["close"])
    bars["_wick_asymmetry"] = (lower_wick - upper_wick) / denominator
    bars["_activity_proxy"] = pd.to_numeric(bars["historical_relative_activity"], errors="coerce")
    bars["_vwap"] = math.nan
    for indices in bars.groupby(["stock", "session"], sort=False).groups.values():
        positions = np.asarray(indices, dtype=int)
        rows = bars.loc[positions]
        typical = (
            rows["high"].to_numpy(float)
            + rows["low"].to_numpy(float)
            + rows["close"].to_numpy(float)
        ) / 3.0
        volume = rows["volume"].to_numpy(float)
        volume = np.where(np.isfinite(volume) & (volume > 0.0), volume, 0.0)
        cumulative_volume = np.cumsum(volume)
        bars.loc[positions, "_vwap"] = np.divide(
            np.cumsum(typical * volume),
            cumulative_volume,
            out=np.full(len(rows), np.nan, dtype=float),
            where=cumulative_volume > 0.0,
        )
    bars["_vwap_log_distance"] = np.log(bars["close"] / bars["_vwap"])
    return bars


def _continuation_boundary(prefix: pd.DataFrame) -> dict[str, float]:
    candidate: tuple[int, int, float, float] | None = None
    start = max(6, len(prefix) - 4)
    for position in range(start, len(prefix)):
        prior = prefix.iloc[position - 6 : position]
        current = prefix.iloc[position]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        up_distance = max(0.0, float(current["high"]) - prior_high) / (abs(prior_high) + EPSILON)
        down_distance = max(0.0, prior_low - float(current["low"])) / (abs(prior_low) + EPSILON)
        if up_distance > 0.0 or down_distance > 0.0:
            direction = 1 if up_distance >= down_distance else -1
            distance = up_distance if direction > 0 else down_distance
            boundary = prior_high if direction > 0 else prior_low
            candidate = (position, direction, boundary, distance)
    if candidate is None:
        return {
            "break_above": 0.0,
            "break_below": 0.0,
            "signed_distance": 0.0,
            "acceptance_count": 0.0,
            "rejection": 0.0,
        }
    position, direction, boundary, breach_distance = candidate
    subsequent = prefix.iloc[position:]
    beyond = (
        subsequent["close"].to_numpy(float) > boundary
        if direction > 0
        else subsequent["close"].to_numpy(float) < boundary
    )
    current_close = float(prefix.iloc[-1]["close"])
    current_distance = (
        (current_close - boundary) / (abs(boundary) + EPSILON)
        if direction > 0
        else (boundary - current_close) / (abs(boundary) + EPSILON)
    )
    rejected = current_distance <= 0.0
    return {
        "break_above": float(direction > 0),
        "break_below": float(direction < 0),
        "signed_distance": float(
            direction * max(0.0, current_distance if not rejected else breach_distance)
        ),
        "acceptance_count": float(direction * int(np.sum(beyond))),
        "rejection": float(-direction if rejected else 0.0),
    }


def _attempt_boundary(
    prefix: pd.DataFrame,
    marker_position: int,
    attempt_sign: int,
) -> dict[str, float]:
    attempt_start = marker_position - 4
    prior = prefix.iloc[max(0, attempt_start - 6) : attempt_start]
    attempt = prefix.iloc[attempt_start : marker_position - 1]
    response = prefix.iloc[marker_position - 1 : marker_position + 1]
    if len(prior) != 6 or len(attempt) != 3 or len(response) != 2 or attempt_sign == 0:
        return {"failure": math.nan, "inside": math.nan, "maintained": math.nan}
    boundary = float(prior["low"].min()) if attempt_sign < 0 else float(prior["high"].max())
    extreme = float(attempt["low"].min()) if attempt_sign < 0 else float(attempt["high"].max())
    response_close = float(response.iloc[-1]["close"])
    failure = mirrored_boundary_failure(
        attempt_sign=attempt_sign,
        attempted_extreme=extreme,
        boundary=boundary,
        response_close=response_close,
    )
    inside = (
        max(0.0, response_close - boundary) / (abs(boundary) + EPSILON)
        if attempt_sign < 0
        else -max(0.0, boundary - response_close) / (abs(boundary) + EPSILON)
    )
    maintained = (
        int(np.sum(response["close"].to_numpy(float) > boundary))
        if attempt_sign < 0
        else -int(np.sum(response["close"].to_numpy(float) < boundary))
    )
    return {
        "failure": failure,
        "inside": float(inside),
        "maintained": float(maintained),
    }


def build_raw_archetype_features(
    checkpoints: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Build common, continuation, and reversal rows ending at marker T-1."""

    required = {"stock", "session", "checkpoint"}
    missing = sorted(required.difference(checkpoints.columns))
    if missing:
        raise ValueError(f"checkpoint inputs missing: {missing}")
    reject_protected_sessions(checkpoints["session"])
    bars = prepare_completed_bars(completed_bars)
    grouped = {
        (str(stock), str(session)): rows.reset_index(drop=True)
        for (stock, session), rows in bars.groupby(["stock", "session"], sort=False)
    }
    feature_rows: list[dict[str, object]] = []
    for checkpoint_row in checkpoints.itertuples(index=False):
        typed_checkpoint_row = cast(Any, checkpoint_row)
        stock = str(typed_checkpoint_row.stock)
        session = str(typed_checkpoint_row.session)
        checkpoint = int(typed_checkpoint_row.checkpoint)
        marker_ordinal = checkpoint - 2
        trigger_ordinal = checkpoint - 1
        session_bars = grouped.get((stock, session))
        if session_bars is None:
            raise ValueError(f"missing completed bars for {stock}|{session}")
        prefix = session_bars.loc[
            session_bars["bar_ordinal"].astype(int).le(marker_ordinal)
        ].sort_values("bar_ordinal", kind="mergesort")
        marker = prefix.loc[prefix["bar_ordinal"].astype(int).eq(marker_ordinal)]
        trigger = session_bars.loc[session_bars["bar_ordinal"].astype(int).eq(trigger_ordinal)]
        if len(marker) != 1 or len(trigger) != 1:
            raise ValueError(f"missing marker or trigger bar for {stock}|{session}|{checkpoint}")
        marker_timestamp = pd.Timestamp(marker.iloc[0]["bar_complete_timestamp"])
        trigger_timestamp = pd.Timestamp(trigger.iloc[0]["bar_complete_timestamp"])
        if not marker_timestamp < trigger_timestamp:
            raise ValueError("direction marker must precede the trigger bar")
        if "signal_timestamp" in checkpoints:
            expected = pd.Timestamp(cast(Any, typed_checkpoint_row.signal_timestamp))
            if trigger_timestamp != expected:
                raise ValueError("trigger timestamp differs from the gate checkpoint")
        if len(prefix) < 5:
            raise ValueError("archetype features need five completed pre-trigger bars")
        returns = prefix["_stock_return"].to_numpy(float)
        market = prefix["_market_return"].to_numpy(float)
        relative = prefix["_relative_return"].to_numpy(float)
        close = prefix["close"].to_numpy(float)
        vwap = prefix["_vwap"].to_numpy(float)
        clv = prefix["_clv"].to_numpy(float)
        wick = prefix["_wick_asymmetry"].to_numpy(float)
        activity = prefix["_activity_proxy"].to_numpy(float)

        stock_1 = _finite_sum(returns[-1:])
        stock_2 = _finite_sum(returns[-2:])
        stock_4 = _finite_sum(returns[-4:])
        stock_6 = _finite_sum(returns[-6:])
        market_1 = _finite_sum(market[-1:])
        market_2 = _finite_sum(market[-2:])
        market_4 = _finite_sum(market[-4:])
        relative_1 = _finite_sum(relative[-1:])
        relative_2 = _finite_sum(relative[-2:])
        net_sign = int(np.sign(stock_4)) if math.isfinite(stock_4) else 0
        continuation_boundary = _continuation_boundary(prefix)
        last_four_clv = clv[-4:]
        last_four_wick = wick[-4:]
        vwap_log = np.log(vwap[-4:]) if np.isfinite(vwap[-4:]).all() else np.full(4, np.nan)
        direction_closes = (
            np.diff(np.log(close[-5:])) * net_sign > 0.0
            if len(close) >= 5 and net_sign
            else np.zeros(4, dtype=bool)
        )
        vwap_side = (
            (close[-3:] - vwap[-3:]) * net_sign > 0.0
            if net_sign and np.isfinite(vwap[-3:]).all()
            else np.zeros(3, dtype=bool)
        )

        attempt_returns = returns[-5:-2]
        response_returns = returns[-2:]
        response_market = market[-2:]
        attempt_relative = relative[-5:-2]
        response_relative = relative[-2:]
        attempt_return = _finite_sum(attempt_returns)
        response_return = _finite_sum(response_returns)
        attempt_sign = int(np.sign(attempt_return)) if math.isfinite(attempt_return) else 0
        attempt_path = (
            float(np.sum(np.abs(attempt_returns)))
            if np.isfinite(attempt_returns).all()
            else math.nan
        )
        attempt_efficiency = (
            abs(attempt_return) / (attempt_path + EPSILON)
            if math.isfinite(attempt_return) and math.isfinite(attempt_path)
            else math.nan
        )
        response_efficiency_in_attempt = (
            attempt_sign * response_return / (float(np.sum(np.abs(response_returns))) + EPSILON)
            if attempt_sign and math.isfinite(response_return)
            else 0.0
        )
        response_activity = (
            float(np.mean(activity[-2:])) if np.isfinite(activity[-2:]).all() else math.nan
        )
        attempt_activity = (
            float(np.mean(activity[-5:-2])) if np.isfinite(activity[-5:-2]).all() else math.nan
        )
        attempt_impact = (
            activity_price_impact(abs(attempt_return), attempt_activity)
            if math.isfinite(attempt_return)
            and math.isfinite(attempt_activity)
            and attempt_activity >= 0.0
            else math.nan
        )
        response_impact = (
            activity_price_impact(abs(response_return), response_activity)
            if math.isfinite(response_return)
            and math.isfinite(response_activity)
            and response_activity >= 0.0
            else math.nan
        )
        attempted_response_progress = (
            max(0.0, attempt_sign * response_return) if attempt_sign else 0.0
        )
        attempted_response_impact = (
            attempted_response_progress / (response_activity + EPSILON)
            if math.isfinite(response_activity) and response_activity >= 0.0
            else math.nan
        )
        attempt_boundary = _attempt_boundary(prefix, len(prefix) - 1, attempt_sign)
        response_clv = float(np.mean(clv[-2:])) if np.isfinite(clv[-2:]).all() else math.nan
        response_wick = (
            float(np.mean(last_four_wick[-2:]))
            if np.isfinite(last_four_wick[-2:]).all()
            else math.nan
        )
        response_close_failure = (
            -attempt_sign * (1.0 - attempt_sign * response_clv) / 2.0
            if attempt_sign and math.isfinite(response_clv)
            else 0.0
        )
        vwap_reclaim = 0.0
        if (
            attempt_sign < 0
            and np.isfinite(vwap[-3:]).all()
            and close[-3] < vwap[-3]
            and close[-1] > vwap[-1]
        ):
            vwap_reclaim = 1.0
        elif (
            attempt_sign > 0
            and np.isfinite(vwap[-3:]).all()
            and close[-3] > vwap[-3]
            and close[-1] < vwap[-1]
        ):
            vwap_reclaim = -1.0
        marker_vwap_distance = (
            float(np.log(close[-1] / vwap[-1]))
            if np.isfinite(vwap[-1]) and vwap[-1] > 0.0
            else math.nan
        )
        opening = prefix.loc[prefix["bar_ordinal"].astype(int).between(0, 5)]
        opening_midpoint = (
            0.5 * (float(opening["high"].max()) + float(opening["low"].min()))
            if len(opening) == 6
            else math.nan
        )
        prior_six = prefix.iloc[-7:-1]
        prior_high = float(prior_six["high"].max()) if len(prior_six) == 6 else math.nan
        prior_low = float(prior_six["low"].min()) if len(prior_six) == 6 else math.nan
        values: dict[str, object] = {
            "marker_bar_ordinal": marker_ordinal,
            "trigger_bar_ordinal": trigger_ordinal,
            "direction_marker_timestamp": marker_timestamp,
            "maximum_direction_feature_timestamp": marker_timestamp,
            "trigger_bar_excluded": True,
            "checkpoint_group": checkpoint_group(checkpoint),
            "day_of_week": pd.Timestamp(session).day_name(),
            "one_bar_log_return": stock_1,
            "two_bar_log_return": stock_2,
            "four_bar_log_return": stock_4,
            "six_bar_log_return": stock_6,
            "path_length_20m": (
                float(np.sum(np.abs(returns[-4:]))) if np.isfinite(returns[-4:]).all() else math.nan
            ),
            "directional_efficiency_20m": directional_efficiency(returns[-4:]),
            "normalised_bar_range": float(prefix.iloc[-1]["_normalised_range"]),
            "close_location_value": float(clv[-1]),
            "wick_asymmetry": float(wick[-1]),
            "session_vwap_distance": marker_vwap_distance,
            "session_vwap_slope": _ols_slope(vwap_log),
            "distance_from_session_open": float(np.log(close[-1] / float(prefix.iloc[0]["open"]))),
            "distance_from_opening_range_midpoint": (
                float(np.log(close[-1] / opening_midpoint))
                if math.isfinite(opening_midpoint) and opening_midpoint > 0.0
                else math.nan
            ),
            "distance_from_prior_six_high": (
                float(np.log(close[-1] / prior_high))
                if math.isfinite(prior_high) and prior_high > 0.0
                else math.nan
            ),
            "distance_from_prior_six_low": (
                float(np.log(close[-1] / prior_low))
                if math.isfinite(prior_low) and prior_low > 0.0
                else math.nan
            ),
            "historical_relative_activity": float(activity[-1]),
            "market_return_5m": market_1,
            "market_return_10m": market_2,
            "market_return_20m": market_4,
            "stock_minus_market_return_5m": relative_1,
            "stock_minus_market_return_10m": relative_2,
            "b_stock_return_5m": stock_1,
            "b_stock_return_10m": stock_2,
            "b_market_return_10m": market_2,
            "b_relative_return_10m": relative_2,
            "b_distance_from_vwap": marker_vwap_distance,
            "c_z_return_5m": stock_1,
            "c_z_return_10m": stock_2,
            "c_z_return_20m": stock_4,
            "c_z_return_30m": stock_6,
            "c_directional_efficiency_20m": directional_efficiency(returns[-4:]),
            "c_mean_clv_4": (
                float(np.mean(last_four_clv)) if np.isfinite(last_four_clv).all() else math.nan
            ),
            "c_directional_close_fraction_4": float(net_sign * np.mean(direction_closes)),
            "c_signed_wick_asymmetry_4": (
                float(np.mean(last_four_wick)) if np.isfinite(last_four_wick).all() else math.nan
            ),
            "c_signed_vwap_slope_4": _ols_slope(vwap_log),
            "c_signed_vwap_distance": marker_vwap_distance,
            "c_vwap_side_closes_3": float(net_sign * np.sum(vwap_side)),
            "c_break_above_prior_six_high": continuation_boundary["break_above"],
            "c_break_below_prior_six_low": continuation_boundary["break_below"],
            "c_signed_boundary_distance": continuation_boundary["signed_distance"],
            "c_signed_boundary_acceptance_count": continuation_boundary["acceptance_count"],
            "c_boundary_rejection": continuation_boundary["rejection"],
            "c_relative_return_5m": relative_1,
            "c_relative_return_10m": relative_2,
            "c_relative_agreement": (
                float(np.sign(stock_2) * abs(relative_2))
                if math.isfinite(stock_2)
                and math.isfinite(relative_2)
                and np.sign(stock_2) == np.sign(relative_2)
                else 0.0
            ),
            "a_attempt_return_abs": abs(attempt_return),
            "a_attempt_path_length": attempt_path,
            "a_attempt_directional_efficiency": attempt_efficiency,
            "a_response_followthrough": response_return,
            "a_reversal_efficiency_change": float(
                -attempt_sign * (attempt_efficiency - response_efficiency_in_attempt)
            )
            if math.isfinite(attempt_efficiency)
            else math.nan,
            "a_wick_rejection": response_wick,
            "a_close_location_recovery": response_clv,
            "a_failure_close_near_extreme": response_close_failure,
            "a_boundary_failure": attempt_boundary["failure"],
            "a_boundary_distance_inside": attempt_boundary["inside"],
            "a_boundary_maintenance_count": attempt_boundary["maintained"],
            "a_vwap_reclaim_failure": vwap_reclaim,
            "a_vwap_distance_after_failure": (marker_vwap_distance if vwap_reclaim else 0.0),
            "a_attempt_price_impact": attempt_impact,
            "a_response_price_impact": response_impact,
            "a_price_impact_decline": (
                float(-attempt_sign * (attempt_impact - attempted_response_impact))
                if math.isfinite(attempt_impact) and math.isfinite(attempted_response_impact)
                else math.nan
            ),
            "a_elevated_activity_weak_progress": (
                float(
                    -attempt_sign
                    * response_activity
                    * max(0.0, attempt_efficiency - response_efficiency_in_attempt)
                )
                if math.isfinite(response_activity) and math.isfinite(attempt_efficiency)
                else math.nan
            ),
            "a_relative_recovery": (_finite_sum(response_relative) - _finite_sum(attempt_relative)),
            "a_market_resilience": float(
                -attempt_sign
                * max(0.0, attempt_sign * _finite_sum(response_market))
                * max(0.0, -attempt_sign * _finite_sum(response_relative))
            )
            if attempt_sign
            and math.isfinite(_finite_sum(response_market))
            and math.isfinite(_finite_sum(response_relative))
            else 0.0,
        }
        for lag in range(4):
            values[f"_stock_return_lag_{lag}"] = float(returns[-1 - lag])
            values[f"_market_return_lag_{lag}"] = float(market[-1 - lag])
        feature_rows.append(values)
    identity = checkpoints.reset_index(drop=True).copy()
    return pd.concat([identity, pd.DataFrame(feature_rows)], axis=1)


def add_relative_strength_features(
    raw_features: pd.DataFrame,
    beta_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Add the stock-specific frozen-beta archetype without peer inputs."""

    required_parameters = {
        "stock",
        "checkpoint_group",
        "alpha",
        "beta",
        "residual_scale",
        "residual_range_low",
        "residual_range_high",
        "stock_abs_return_median",
    }
    missing = sorted(required_parameters.difference(beta_parameters.columns))
    if missing:
        raise ValueError(f"beta parameters missing: {missing}")
    parameters = beta_parameters.set_index(["stock", "checkpoint_group"])
    output = raw_features.copy()
    rows: list[dict[str, float]] = []
    for _, row in output.iterrows():
        key = (str(row["stock"]), str(row["checkpoint_group"]))
        if key not in parameters.index:
            raise ValueError(f"beta parameters missing for {key}")
        fitted = cast(pd.Series, parameters.loc[key])
        alpha = float(fitted["alpha"])
        beta = float(fitted["beta"])
        stock_returns = np.asarray(
            [float(row[f"_stock_return_lag_{lag}"]) for lag in range(4)],
            dtype=float,
        )
        market_returns = np.asarray(
            [float(row[f"_market_return_lag_{lag}"]) for lag in range(4)],
            dtype=float,
        )
        residuals = stock_returns - (alpha + beta * market_returns)
        residual_5 = _finite_sum(residuals[:1])
        residual_10 = _finite_sum(residuals[:2])
        residual_20 = _finite_sum(residuals[:4])
        stock_20 = _finite_sum(stock_returns)
        market_20 = _finite_sum(market_returns)
        chronological_residuals = residuals[::-1]
        slope = _ols_slope(chronological_residuals)
        scale = float(fitted["residual_scale"])
        low = float(fitted["residual_range_low"]) * math.sqrt(4.0)
        high = float(fitted["residual_range_high"]) * math.sqrt(4.0)
        distance = (
            residual_20 - high
            if residual_20 > high
            else residual_20 - low
            if residual_20 < low
            else 0.0
        )
        compressed_limit = 4.0 * float(fitted["stock_abs_return_median"])
        rows.append(
            {
                "r_residual_return_5m": residual_5,
                "r_residual_return_10m": residual_10,
                "r_residual_return_20m": residual_20,
                "r_residual_slope": slope,
                "r_residual_persistence": residual_persistence(residuals),
                "r_change_in_residual_strength": float(
                    np.mean(residuals[:2]) - np.mean(residuals[2:])
                ),
                "r_stock_flat_up_market_down": (
                    abs(market_20) if stock_20 >= 0.0 and market_20 < 0.0 else 0.0
                ),
                "r_stock_flat_down_market_up": (
                    -abs(market_20) if stock_20 <= 0.0 and market_20 > 0.0 else 0.0
                ),
                "r_residual_volatility_score": residual_20 / (scale * math.sqrt(4.0) + EPSILON),
                "r_distance_from_normal_residual_range": distance,
                "r_absolute_residual_direction_agreement": (
                    float(np.sign(residual_20) * abs(residual_20))
                    if np.sign(stock_20) == np.sign(residual_20)
                    else 0.0
                ),
                "r_improving_while_absolute_compressed": (
                    slope if abs(stock_20) <= compressed_limit and math.isfinite(slope) else 0.0
                ),
            }
        )
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def apply_selective_policy(
    probabilities: Sequence[float] | FloatArray,
    boundary: float,
) -> NDArray[np.str_]:
    """Apply one symmetric development-frozen CALL/PUT/ABSTAIN boundary."""

    values = np.asarray(probabilities, dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or not 0.0 <= boundary <= 0.5
    ):
        raise ValueError("selective policy inputs are outside the frozen domain")
    actions = np.full(len(values), "ABSTAIN", dtype="<U7")
    actions[values >= 0.5 + boundary] = "CALL"
    actions[values <= 0.5 - boundary] = "PUT"
    return actions


def remaining_fraction(
    pre_entry_return: float,
    post_entry_return: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Return the absolute fraction of the move remaining after entry."""

    if not all(math.isfinite(value) for value in (pre_entry_return, post_entry_return)):
        return math.nan
    return float(
        abs(post_entry_return) / (abs(pre_entry_return) + abs(post_entry_return) + epsilon)
    )


def shift_features_to_next_episode(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Create the within-stock next-episode temporal placebo bundle."""

    required = {"stock", "session", "checkpoint", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"temporal placebo inputs missing: {missing}")
    ordered = frame.sort_values(["stock", "session", "checkpoint"], kind="mergesort").copy()
    ordered.loc[:, list(feature_columns)] = ordered.groupby("stock", sort=False)[
        list(feature_columns)
    ].shift(1)
    return ordered.sort_index(kind="mergesort")


def archetype_decision(evidence: Mapping[str, object]) -> str:
    """Apply every preregistered individual archetype support condition."""

    passed = all(
        (
            bool(evidence.get("log_loss_improves", False)),
            bool(evidence.get("brier_improves", False)),
            _as_float(evidence.get("auc", -math.inf)) >= 0.55,
            _as_float(evidence.get("balanced_accuracy", -math.inf)) > 0.52,
            0.20 <= _as_float(evidence.get("action_coverage", math.nan)) <= 0.50,
            _as_float(evidence.get("selective_accuracy", -math.inf)) >= 0.57,
            bool(evidence.get("beats_all_selective_baselines", False)),
            _as_float(evidence.get("mean_aligned_return", -math.inf)) > 0.0,
            _as_float(evidence.get("median_aligned_return", -math.inf)) > 0.0,
            _as_float(evidence.get("bootstrap_80_accuracy_lower", -math.inf)) > 0.50,
            _as_float(evidence.get("bootstrap_80_mean_return_lower", -math.inf)) >= 0.0,
            _as_int(evidence.get("positive_months", 0)) >= 6,
            _as_int(evidence.get("null_predictive_wins", 0)) >= 9,
            _as_int(evidence.get("null_return_wins", 0)) >= 9,
            bool(evidence.get("beats_temporal_placebo", False)),
            bool(evidence.get("selective_support_passed", False)),
            bool(evidence.get("concentration_passed", False)),
            not bool(evidence.get("late_direction_problem", True)),
        )
    )
    return "supported" if passed else "not_supported"


__all__ = [
    "ABSORPTION_FEATURES",
    "BASELINE_FEATURES",
    "CHECKPOINT_GROUPS",
    "CONTINUATION_FEATURES",
    "EPSILON",
    "MINIMUM_EPISODE_SPACING_MINUTES",
    "PRIMARY_DIRECTION_HORIZON_MINUTES",
    "PROTECTED_START",
    "RELATIVE_STRENGTH_FEATURES",
    "activity_price_impact",
    "add_relative_strength_features",
    "apply_selective_policy",
    "apply_stock_local_normalisation",
    "archetype_decision",
    "beta_adjusted_residual",
    "build_raw_archetype_features",
    "build_movement_dependency_audit",
    "checkpoint_group",
    "construct_fresh_episodes",
    "directional_efficiency",
    "fit_stock_local_normalisation",
    "fit_stock_market_betas",
    "mirrored_boundary_failure",
    "mirrored_wick_rejection",
    "prepare_completed_bars",
    "remaining_fraction",
    "residual_persistence",
    "reject_protected_sessions",
    "shift_features_to_next_episode",
    "transitive_descendants",
    "weighted_quantile",
]
