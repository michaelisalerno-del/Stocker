"""Frozen T-1 stock-local feature construction for A1, C1, and R1."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

EPSILON: Final[float] = 1e-12
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


class DirectionFeatureBar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    bar_ordinal: int
    bar_start_timestamp: datetime
    bar_complete_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    historical_relative_activity: float
    stock_log_return: float
    market_log_return: float
    finalised: bool

    @field_validator("bar_start_timestamp", "bar_complete_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


class DirectionFeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    checkpoint: int
    checkpoint_category: str
    checkpoint_group: str
    day_of_week: str
    marker_bar_ordinal: int
    trigger_bar_ordinal: int
    maximum_direction_feature_timestamp: datetime
    trigger_bar_excluded: bool
    raw_features: dict[str, float]
    feature_hash: str
    beta_artifact_hash: str


def checkpoint_group(checkpoint: int) -> str:
    if 6 <= checkpoint <= 14:
        return "early"
    if 16 <= checkpoint <= 24:
        return "middle"
    if 26 <= checkpoint <= 34:
        return "late"
    raise ValueError("checkpoint outside frozen direction grid")


def _finite_sum(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    return float(np.sum(values)) if len(values) and np.isfinite(values).all() else math.nan


def _ols_slope(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
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


def _directional_efficiency(
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    if not len(values) or not np.isfinite(values).all():
        return math.nan
    return float(np.sum(values) / (np.sum(np.abs(values)) + EPSILON))


def _activity_impact(absolute_return: float, activity: float) -> float:
    if not math.isfinite(absolute_return) or not math.isfinite(activity):
        return math.nan
    return float(absolute_return / (activity + EPSILON))


def _continuation_boundary(
    highs: np.ndarray[tuple[int], np.dtype[np.float64]],
    lows: np.ndarray[tuple[int], np.dtype[np.float64]],
    closes: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> dict[str, float]:
    candidate: tuple[int, int, float, float] | None = None
    start = max(6, len(highs) - 4)
    for position in range(start, len(highs)):
        prior_high = float(np.max(highs[position - 6 : position]))
        prior_low = float(np.min(lows[position - 6 : position]))
        up_distance = max(0.0, float(highs[position]) - prior_high) / (abs(prior_high) + EPSILON)
        down_distance = max(0.0, prior_low - float(lows[position])) / (abs(prior_low) + EPSILON)
        if up_distance > 0.0 or down_distance > 0.0:
            direction = 1 if up_distance >= down_distance else -1
            candidate = (
                position,
                direction,
                prior_high if direction > 0 else prior_low,
                up_distance if direction > 0 else down_distance,
            )
    if candidate is None:
        return {
            "break_above": 0.0,
            "break_below": 0.0,
            "signed_distance": 0.0,
            "acceptance_count": 0.0,
            "rejection": 0.0,
        }
    position, direction, boundary, breach_distance = candidate
    subsequent = closes[position:]
    beyond = subsequent > boundary if direction > 0 else subsequent < boundary
    current_close = float(closes[-1])
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


def _mirrored_boundary_failure(
    *,
    attempt_sign: int,
    attempted_extreme: float,
    boundary: float,
    response_close: float,
) -> float:
    scale = abs(boundary) + EPSILON
    if attempt_sign < 0 and attempted_extreme < boundary and response_close > boundary:
        return float((response_close - boundary) / scale)
    if attempt_sign > 0 and attempted_extreme > boundary and response_close < boundary:
        return float(-(boundary - response_close) / scale)
    return 0.0


def _attempt_boundary(
    *,
    highs: np.ndarray[tuple[int], np.dtype[np.float64]],
    lows: np.ndarray[tuple[int], np.dtype[np.float64]],
    closes: np.ndarray[tuple[int], np.dtype[np.float64]],
    marker_position: int,
    attempt_sign: int,
) -> dict[str, float]:
    attempt_start = marker_position - 4
    prior_highs = highs[max(0, attempt_start - 6) : attempt_start]
    prior_lows = lows[max(0, attempt_start - 6) : attempt_start]
    attempt_highs = highs[attempt_start : marker_position - 1]
    attempt_lows = lows[attempt_start : marker_position - 1]
    response_closes = closes[marker_position - 1 : marker_position + 1]
    if (
        len(prior_highs) != 6
        or len(attempt_highs) != 3
        or len(response_closes) != 2
        or attempt_sign == 0
    ):
        return {"failure": math.nan, "inside": math.nan, "maintained": math.nan}
    boundary = float(np.min(prior_lows)) if attempt_sign < 0 else float(np.max(prior_highs))
    extreme = float(np.min(attempt_lows)) if attempt_sign < 0 else float(np.max(attempt_highs))
    response_close = float(response_closes[-1])
    inside = (
        max(0.0, response_close - boundary) / (abs(boundary) + EPSILON)
        if attempt_sign < 0
        else -max(0.0, boundary - response_close) / (abs(boundary) + EPSILON)
    )
    maintained = (
        int(np.sum(response_closes > boundary))
        if attempt_sign < 0
        else -int(np.sum(response_closes < boundary))
    )
    return {
        "failure": _mirrored_boundary_failure(
            attempt_sign=attempt_sign,
            attempted_extreme=extreme,
            boundary=boundary,
            response_close=response_close,
        ),
        "inside": float(inside),
        "maintained": float(maintained),
    }


class FrozenDirectionFeatureBuilder:
    """Build all raw archetype fields from bars no later than marker T-1."""

    def __init__(
        self,
        *,
        beta_parameters: dict[tuple[str, str], dict[str, float]],
        beta_artifact_hash: str,
    ) -> None:
        self.beta_parameters = beta_parameters
        self.beta_artifact_hash = beta_artifact_hash

    @classmethod
    def from_beta_artifact(
        cls,
        path: str | Path,
    ) -> FrozenDirectionFeatureBuilder:
        artifact = Path(path)
        frame = pd.read_csv(artifact)
        full = frame.loc[frame["fit_scope"].astype(str).eq("full_2024")].copy()
        if len(full) != 60 or full.duplicated(["stock", "checkpoint_group"]).any():
            raise ValueError("frozen full-2024 beta parameters are incomplete")
        parameters = {
            (str(row.stock), str(row.checkpoint_group)): {
                name: float(getattr(row, name))
                for name in (
                    "alpha",
                    "beta",
                    "residual_scale",
                    "residual_range_low",
                    "residual_range_high",
                    "stock_abs_return_median",
                )
            }
            for row in full.itertuples(index=False)
        }
        return cls(
            beta_parameters=parameters,
            beta_artifact_hash=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )

    def build(
        self,
        *,
        symbol: str,
        checkpoint: int,
        completed_bars: tuple[DirectionFeatureBar, ...],
    ) -> DirectionFeatureResult:
        if len(completed_bars) != checkpoint:
            raise ValueError("direction features require exactly checkpoint completed bars")
        if checkpoint not in tuple(range(6, 35, 2)):
            raise ValueError("checkpoint outside frozen direction grid")
        if [item.bar_ordinal for item in completed_bars] != list(range(checkpoint)):
            raise ValueError("direction bars must be contiguous")
        if any(item.symbol != symbol for item in completed_bars):
            raise ValueError("direction bar symbol identity mismatch")
        if any(not item.finalised for item in completed_bars):
            raise ValueError("direction bars must be finalised")
        sessions = {item.session for item in completed_bars}
        if len(sessions) != 1:
            raise ValueError("direction bars must belong to one session")
        session = next(iter(sessions))
        marker_ordinal = checkpoint - 2
        trigger_ordinal = checkpoint - 1
        prefix = completed_bars[: marker_ordinal + 1]
        marker = prefix[-1]
        trigger = completed_bars[trigger_ordinal]
        if not marker.bar_complete_timestamp < trigger.bar_complete_timestamp:
            raise ValueError("direction marker must precede trigger bar")
        if len(prefix) < 5:
            raise ValueError("direction features require five pre-trigger bars")
        if any(
            not all(
                math.isfinite(value)
                for value in (
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.historical_relative_activity,
                    bar.stock_log_return,
                    bar.market_log_return,
                )
            )
            for bar in prefix
        ):
            raise ValueError("direction feature bar contains non-finite values")

        returns = np.asarray([item.stock_log_return for item in prefix], dtype=np.float64)
        market = np.asarray([item.market_log_return for item in prefix], dtype=np.float64)
        relative = returns - market
        close = np.asarray([item.close for item in prefix], dtype=np.float64)
        high = np.asarray([item.high for item in prefix], dtype=np.float64)
        low = np.asarray([item.low for item in prefix], dtype=np.float64)
        open_ = np.asarray([item.open for item in prefix], dtype=np.float64)
        volume = np.asarray([item.volume for item in prefix], dtype=np.float64)
        activity = np.asarray(
            [item.historical_relative_activity for item in prefix],
            dtype=np.float64,
        )
        typical = (high + low + close) / 3.0
        positive_volume = np.where(volume > 0.0, volume, 0.0)
        cumulative_volume = np.cumsum(positive_volume)
        vwap = np.divide(
            np.cumsum(typical * positive_volume),
            cumulative_volume,
            out=np.full(len(prefix), np.nan, dtype=np.float64),
            where=cumulative_volume > 0.0,
        )
        width = high - low
        clv = (2.0 * close - high - low) / (width + EPSILON)
        lower_wick = np.minimum(open_, close) - low
        upper_wick = high - np.maximum(open_, close)
        wick = (lower_wick - upper_wick) / (width + EPSILON)

        stock_1 = _finite_sum(returns[-1:])
        stock_2 = _finite_sum(returns[-2:])
        stock_4 = _finite_sum(returns[-4:])
        stock_6 = _finite_sum(returns[-6:])
        relative_1 = _finite_sum(relative[-1:])
        relative_2 = _finite_sum(relative[-2:])
        net_sign = int(np.sign(stock_4)) if math.isfinite(stock_4) else 0
        continuation = _continuation_boundary(high, low, close)
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
        response_efficiency = (
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
        attempt_impact = _activity_impact(abs(attempt_return), attempt_activity)
        response_impact = _activity_impact(abs(response_return), response_activity)
        attempted_response_progress = (
            max(0.0, attempt_sign * response_return) if attempt_sign else 0.0
        )
        attempted_response_impact = (
            attempted_response_progress / (response_activity + EPSILON)
            if math.isfinite(response_activity) and response_activity >= 0.0
            else math.nan
        )
        attempt_boundary = _attempt_boundary(
            highs=high,
            lows=low,
            closes=close,
            marker_position=len(prefix) - 1,
            attempt_sign=attempt_sign,
        )
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
        raw: dict[str, float] = {
            "c_z_return_5m": stock_1,
            "c_z_return_10m": stock_2,
            "c_z_return_20m": stock_4,
            "c_z_return_30m": stock_6,
            "c_directional_efficiency_20m": _directional_efficiency(returns[-4:]),
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
            "c_break_above_prior_six_high": continuation["break_above"],
            "c_break_below_prior_six_low": continuation["break_below"],
            "c_signed_boundary_distance": continuation["signed_distance"],
            "c_signed_boundary_acceptance_count": continuation["acceptance_count"],
            "c_boundary_rejection": continuation["rejection"],
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
            "a_reversal_efficiency_change": (
                float(-attempt_sign * (attempt_efficiency - response_efficiency))
                if math.isfinite(attempt_efficiency)
                else math.nan
            ),
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
                    * max(0.0, attempt_efficiency - response_efficiency)
                )
                if math.isfinite(response_activity) and math.isfinite(attempt_efficiency)
                else math.nan
            ),
            "a_relative_recovery": (_finite_sum(response_relative) - _finite_sum(attempt_relative)),
            "a_market_resilience": (
                float(
                    -attempt_sign
                    * max(0.0, attempt_sign * _finite_sum(response_market))
                    * max(0.0, -attempt_sign * _finite_sum(response_relative))
                )
                if attempt_sign
                and math.isfinite(_finite_sum(response_market))
                and math.isfinite(_finite_sum(response_relative))
                else 0.0
            ),
        }
        group = checkpoint_group(checkpoint)
        fitted = self.beta_parameters.get((symbol, group))
        if fitted is None:
            raise ValueError(f"frozen beta unavailable for {symbol}|{group}")
        stock_lags = np.asarray(
            [float(returns[-1 - lag]) for lag in range(4)],
            dtype=np.float64,
        )
        market_lags = np.asarray(
            [float(market[-1 - lag]) for lag in range(4)],
            dtype=np.float64,
        )
        residuals = stock_lags - (fitted["alpha"] + fitted["beta"] * market_lags)
        residual_5 = _finite_sum(residuals[:1])
        residual_10 = _finite_sum(residuals[:2])
        residual_20 = _finite_sum(residuals[:4])
        stock_20 = _finite_sum(stock_lags)
        market_20 = _finite_sum(market_lags)
        slope = _ols_slope(residuals[::-1])
        low_range = fitted["residual_range_low"] * math.sqrt(4.0)
        high_range = fitted["residual_range_high"] * math.sqrt(4.0)
        distance = (
            residual_20 - high_range
            if residual_20 > high_range
            else residual_20 - low_range
            if residual_20 < low_range
            else 0.0
        )
        compressed_limit = 4.0 * fitted["stock_abs_return_median"]
        raw.update(
            {
                "r_residual_return_5m": residual_5,
                "r_residual_return_10m": residual_10,
                "r_residual_return_20m": residual_20,
                "r_residual_slope": slope,
                "r_residual_persistence": float(np.mean(np.sign(residuals))),
                "r_change_in_residual_strength": float(
                    np.mean(residuals[:2]) - np.mean(residuals[2:])
                ),
                "r_stock_flat_up_market_down": (
                    abs(market_20) if stock_20 >= 0.0 and market_20 < 0.0 else 0.0
                ),
                "r_stock_flat_down_market_up": (
                    -abs(market_20) if stock_20 <= 0.0 and market_20 > 0.0 else 0.0
                ),
                "r_residual_volatility_score": residual_20
                / (fitted["residual_scale"] * math.sqrt(4.0) + EPSILON),
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
        expected = {
            *CONTINUATION_FEATURES,
            *ABSORPTION_FEATURES,
            *RELATIVE_STRENGTH_FEATURES,
        }
        if set(raw) != expected:
            raise RuntimeError("direction feature manifest drifted")
        payload = "|".join(f"{name}={raw[name]:.17g}" for name in sorted(raw))
        return DirectionFeatureResult(
            symbol=symbol,
            session=session,
            checkpoint=checkpoint,
            checkpoint_category=str(checkpoint),
            checkpoint_group=group,
            day_of_week=session.strftime("%A"),
            marker_bar_ordinal=marker_ordinal,
            trigger_bar_ordinal=trigger_ordinal,
            maximum_direction_feature_timestamp=marker.bar_complete_timestamp,
            trigger_bar_excluded=True,
            raw_features=raw,
            feature_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            beta_artifact_hash=self.beta_artifact_hash,
        )
