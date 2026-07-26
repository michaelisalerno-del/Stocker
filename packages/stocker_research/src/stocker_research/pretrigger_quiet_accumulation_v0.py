"""Causal primitives for Pre-Trigger Quiet Accumulation Direction Screen V0.

The frozen M1 movement model is only an eligibility gate.  This module builds
one symmetric, bar-derived direction marker whose latest input is the close of
the bar immediately preceding the M1 trigger bar.
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
StringArray = NDArray[np.str_]

M1_THRESHOLD: Final[float] = 0.49588519865576763
EPSILON: Final[float] = 1e-12
PRIMARY_WINDOW_BARS: Final[int] = 5
PRIMARY_WINDOW_MINUTES: Final[int] = 25
SECONDARY_WINDOW_BARS: Final[tuple[int, ...]] = (3, 9)
ATR_WINDOW_BARS: Final[int] = 14
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
TARGET_HORIZONS_MINUTES: Final[tuple[int, ...]] = (5, 10, 15, 30)
DEVELOPMENT_START: Final[str] = "2024-01-01"
DEVELOPMENT_END: Final[str] = "2024-12-31"
ASSESSMENT_START: Final[str] = "2025-01-01"
ASSESSMENT_END: Final[str] = "2025-08-22"

QUIET_SIGNED_COMPONENTS: Final[tuple[str, ...]] = (
    "pressure_sum_25",
    "pressure_persistence_25",
    "pressure_slope_25",
    "signed_absorption_divergence_25",
    "activity_without_displacement_25",
    "relative_resilience_25",
    "mean_clv_25",
    "mean_wick_asymmetry_25",
    "break_failure_asymmetry_25",
    "mean_vwap_distance_25",
    "vwap_side_balance_25",
    "vwap_reclaim_balance_25",
    "accumulation_sign_persistence_25",
)
GROUP_P: Final[tuple[str, ...]] = (
    "pressure_sum_25",
    "pressure_persistence_25",
    "pressure_slope_25",
    "signed_absorption_divergence_25",
    "accumulation_sign_persistence_25",
)
GROUP_A: Final[tuple[str, ...]] = (
    "activity_without_displacement_25",
    "mean_clv_25",
    "mean_wick_asymmetry_25",
    "break_failure_asymmetry_25",
)
GROUP_C: Final[tuple[str, ...]] = (
    "range_compression_25",
    "path_compression_25",
    "relative_resilience_25",
    "mean_vwap_distance_25",
    "vwap_side_balance_25",
    "vwap_reclaim_balance_25",
    "quietness_25",
    "quiet_absorption_score_25",
)
Q0_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "stock_return_5m_tminus1",
    "stock_return_10m_tminus1",
    "stock_return_20m_tminus1",
    "market_return_5m_tminus1",
    "market_return_10m_tminus1",
    "market_return_20m_tminus1",
    "stock_minus_market_return_5m_tminus1",
    "stock_minus_market_return_10m_tminus1",
    "distance_from_vwap_tminus1",
    "distance_from_session_open_tminus1",
    "distance_from_opening_range_midpoint_tminus1",
    "distance_from_previous_six_bar_high_tminus1",
    "distance_from_previous_six_bar_low_tminus1",
    "clv_tminus1",
    "wick_asymmetry_tminus1",
)
MODEL_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "stock",
    "checkpoint_indicator",
    "day_of_week_indicator",
)
QS_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "quiet_absorption_score_25",
    "quietness_25",
)
Q1_NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*Q0_NUMERIC_FEATURES, *GROUP_P, *GROUP_A, *GROUP_C))
)
PRIMARY_RAW_FEATURES: Final[tuple[str, ...]] = (
    "net_return_25",
    "path_length_25",
    "range_sum_25",
    "directional_efficiency_25",
    "pressure_sum_25",
    "pressure_persistence_25",
    "pressure_slope_25",
    "activity_without_displacement_25",
    "relative_resilience_25",
    "mean_clv_25",
    "mean_wick_asymmetry_25",
    "break_failure_asymmetry_25",
    "mean_vwap_distance_25",
    "vwap_side_balance_25",
    "vwap_reclaim_balance_25",
)
DECISION_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "pretrigger_quiet_accumulation_direction_candidate_supported",
        "persistent_pressure_direction_supported_absorption_not_supported",
        "absorption_response_promising_but_full_gate_not_met",
        "pretrigger_direction_present_but_too_late",
        "quiet_accumulation_score_descriptive_only",
        "pretrigger_quiet_accumulation_unstable",
        "no_incremental_pretrigger_directional_signal",
        "blocked_movement_episode_reconstruction_failure",
        "blocked_insufficient_pretrigger_history",
        "blocked_insufficient_direction_episode_support",
        "blocked_insufficient_selective_action_support",
        "blocked_chronology_or_leakage_failure",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def _finite_complete(values: Sequence[float] | FloatArray) -> FloatArray | None:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        return None
    return array


def _complete_sum(values: Sequence[float] | FloatArray) -> float:
    array = _finite_complete(values)
    return float(np.sum(array)) if array is not None else math.nan


def _complete_mean(values: Sequence[float] | FloatArray) -> float:
    array = _finite_complete(values)
    return float(np.mean(array)) if array is not None else math.nan


def bar_signed_return(close: float, previous_close: float) -> float:
    """Log return of one completed bar."""

    if (
        not math.isfinite(close)
        or not math.isfinite(previous_close)
        or close <= 0.0
        or previous_close <= 0.0
    ):
        return math.nan
    return math.log(close / previous_close)


def bar_relative_return(stock_return: float, market_return: float) -> float:
    """Stock log return minus the frozen market-proxy log return."""

    if not math.isfinite(stock_return) or not math.isfinite(market_return):
        return math.nan
    return stock_return - market_return


def bar_normalised_range(high: float, low: float, previous_close: float) -> float:
    """Completed-bar high/low range divided by its prior close."""

    if (
        not math.isfinite(high)
        or not math.isfinite(low)
        or not math.isfinite(previous_close)
        or previous_close <= 0.0
        or high < low
    ):
        return math.nan
    return (high - low) / previous_close


def bar_clv(
    high: float,
    low: float,
    close: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Close-location value with the contract epsilon."""

    if not all(math.isfinite(value) for value in (high, low, close)) or high < low:
        return math.nan
    return (2.0 * close - high - low) / (high - low + epsilon)


def bar_wick_asymmetry(
    open_price: float,
    high: float,
    low: float,
    close: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Lower-wick recovery minus upper-wick rejection, normalised by range."""

    if not all(math.isfinite(value) for value in (open_price, high, low, close)) or high < low:
        return math.nan
    lower_wick = min(open_price, close) - low
    upper_wick = high - max(open_price, close)
    return (lower_wick - upper_wick) / (high - low + epsilon)


def bar_break_failure_asymmetry(
    *,
    high: float,
    low: float,
    close: float,
    prior_highs: Sequence[float] | FloatArray,
    prior_lows: Sequence[float] | FloatArray,
    epsilon: float = EPSILON,
) -> float:
    """Mirrored reclaim/rejection response relative to exactly six prior bars."""

    highs = np.asarray(prior_highs, dtype=np.float64)
    lows = np.asarray(prior_lows, dtype=np.float64)
    if len(highs) != 6 or len(lows) != 6:
        raise ValueError("break-failure asymmetry requires exactly six prior bars")
    if (
        not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or not all(math.isfinite(value) for value in (high, low, close))
        or high < low
    ):
        return math.nan
    prior_low = float(np.min(lows))
    prior_high = float(np.max(highs))
    denominator = high - low + epsilon
    downside_reclaim = float(low < prior_low) * (close - low) / denominator
    upside_rejection = float(high > prior_high) * (high - close) / denominator
    return downside_reclaim - upside_rejection


def pressure_persistence(values: Sequence[float] | FloatArray) -> float:
    """Mean pressure sign; exact zero contributes zero."""

    array = _finite_complete(values)
    return float(np.mean(np.sign(array))) if array is not None else math.nan


def pressure_slope(values: Sequence[float] | FloatArray) -> float:
    """OLS slope of cumulative signed pressure across the supplied bars."""

    array = _finite_complete(values)
    if array is None or len(array) < 2:
        return math.nan
    x_values = np.arange(len(array), dtype=np.float64)
    cumulative = np.cumsum(array)
    centered_x = x_values - float(np.mean(x_values))
    denominator = float(np.sum(centered_x**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(centered_x * (cumulative - np.mean(cumulative))) / denominator)


def signed_absorption_divergence(
    *,
    pressure_sum: float,
    pressure_z: float,
    price_z: float,
) -> float:
    """One mirrored pressure/price displacement divergence."""

    if not all(math.isfinite(value) for value in (pressure_sum, pressure_z, price_z)):
        return math.nan
    return float(np.sign(pressure_sum) * (abs(pressure_z) - abs(price_z)))


def activity_without_displacement(
    *,
    pressure_sum: float,
    activity: Sequence[float] | FloatArray,
    net_return: float,
    path_length: float,
    epsilon: float = EPSILON,
) -> float:
    """Signed non-negative activity proxy moderated by displacement fraction."""

    values = _finite_complete(activity)
    if (
        values is None
        or not all(math.isfinite(value) for value in (pressure_sum, net_return, path_length))
        or path_length < 0.0
    ):
        return math.nan
    displacement_fraction = min(1.0, abs(net_return) / (path_length + epsilon))
    non_negative_activity = np.maximum(values, 0.0)
    return float(
        np.sign(pressure_sum) * np.mean(non_negative_activity) * (1.0 - displacement_fraction)
    )


def score_sign_persistence(values: Sequence[float] | FloatArray) -> float:
    """Mean sign of the five causal three-bar divergence readings."""

    array = _finite_complete(values)
    return float(np.mean(np.sign(array))) if array is not None else math.nan


@dataclass(frozen=True)
class RobustLocationScale:
    """Development-fitted imputation, robust centre, and IQR scale."""

    imputation: float
    center: float
    scale: float

    def transform(self, values: Sequence[float] | FloatArray) -> FloatArray:
        raw = np.asarray(values, dtype=np.float64)
        imputed = np.where(np.isfinite(raw), raw, self.imputation)
        return np.asarray((imputed - self.center) / self.scale, dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {
            "imputation": self.imputation,
            "center": self.center,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RobustLocationScale:
        return cls(
            imputation=_as_float(value["imputation"]),
            center=_as_float(value["center"]),
            scale=_as_float(value["scale"]),
        )


def fit_robust_location_scale(values: Sequence[float] | FloatArray) -> RobustLocationScale:
    """Fit deterministic median/IQR preprocessing to development values."""

    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    imputation = float(np.median(finite)) if len(finite) else 0.0
    imputed = np.where(np.isfinite(raw), raw, imputation)
    center = float(np.median(imputed)) if len(imputed) else 0.0
    if len(imputed):
        q25, q75 = np.quantile(imputed, [0.25, 0.75])
        scale = float(q75 - q25)
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= EPSILON:
        scale = 1.0
    return RobustLocationScale(imputation=imputation, center=center, scale=scale)


@dataclass(frozen=True)
class QuietScoreParameters:
    """All development-only preprocessing needed by the frozen composite."""

    fit_partition: str
    pressure_25: RobustLocationScale
    price_25: RobustLocationScale
    pressure_3bar: RobustLocationScale
    price_3bar: RobustLocationScale
    range_sum: RobustLocationScale
    path_length: RobustLocationScale
    component_parameters: dict[str, RobustLocationScale]
    component_clip_lower: float = -3.0
    component_clip_upper: float = 3.0

    def as_dict(self) -> dict[str, object]:
        return {
            "fit_partition": self.fit_partition,
            "pressure_25": self.pressure_25.as_dict(),
            "price_25": self.price_25.as_dict(),
            "pressure_3bar": self.pressure_3bar.as_dict(),
            "price_3bar": self.price_3bar.as_dict(),
            "range_sum": self.range_sum.as_dict(),
            "path_length": self.path_length.as_dict(),
            "component_parameters": {
                name: parameter.as_dict() for name, parameter in self.component_parameters.items()
            },
            "component_clip": [self.component_clip_lower, self.component_clip_upper],
            "component_weights": "equal",
            "epsilon": EPSILON,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QuietScoreParameters:
        component_mapping = cast(Mapping[str, Mapping[str, object]], value["component_parameters"])
        clip = cast(Sequence[object], value["component_clip"])
        return cls(
            fit_partition=str(value["fit_partition"]),
            pressure_25=RobustLocationScale.from_dict(
                cast(Mapping[str, object], value["pressure_25"])
            ),
            price_25=RobustLocationScale.from_dict(cast(Mapping[str, object], value["price_25"])),
            pressure_3bar=RobustLocationScale.from_dict(
                cast(Mapping[str, object], value["pressure_3bar"])
            ),
            price_3bar=RobustLocationScale.from_dict(
                cast(Mapping[str, object], value["price_3bar"])
            ),
            range_sum=RobustLocationScale.from_dict(cast(Mapping[str, object], value["range_sum"])),
            path_length=RobustLocationScale.from_dict(
                cast(Mapping[str, object], value["path_length"])
            ),
            component_parameters={
                str(name): RobustLocationScale.from_dict(parameter)
                for name, parameter in component_mapping.items()
            },
            component_clip_lower=_as_float(clip[0]),
            component_clip_upper=_as_float(clip[1]),
        )


def _three_bar_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"_{prefix}_3bar_position_{position}" for position in range(5))


def _derive_divergence_columns(
    raw: pd.DataFrame,
    *,
    pressure_25: RobustLocationScale,
    price_25: RobustLocationScale,
    pressure_3bar: RobustLocationScale,
    price_3bar: RobustLocationScale,
) -> pd.DataFrame:
    output = raw.copy()
    pressure_raw = pd.to_numeric(output["pressure_sum_25"], errors="coerce").to_numpy(float)
    price_raw = pd.to_numeric(output["net_return_25"], errors="coerce").to_numpy(float)
    pressure_z = pressure_25.transform(pressure_raw)
    price_z = price_25.transform(price_raw)
    output["pressure_z_25"] = pressure_z
    output["price_z_25"] = price_z
    valid_25 = np.isfinite(pressure_raw) & np.isfinite(price_raw)
    divergence = np.full(len(output), np.nan, dtype=np.float64)
    divergence[valid_25] = np.sign(pressure_raw[valid_25]) * (
        np.abs(pressure_z[valid_25]) - np.abs(price_z[valid_25])
    )
    output["signed_absorption_divergence_25"] = divergence

    pressure_columns = _three_bar_columns("pressure_sum")
    return_columns = _three_bar_columns("net_return")
    missing = sorted(set((*pressure_columns, *return_columns)).difference(output.columns))
    if missing:
        raise ValueError(f"three-bar divergence intermediates missing: {missing}")
    persistence_values = np.full((len(output), PRIMARY_WINDOW_BARS), np.nan, dtype=float)
    for position, (pressure_column, return_column) in enumerate(
        zip(pressure_columns, return_columns, strict=True)
    ):
        pressure_position = pd.to_numeric(output[pressure_column], errors="coerce").to_numpy(float)
        return_position = pd.to_numeric(output[return_column], errors="coerce").to_numpy(float)
        pressure_position_z = pressure_3bar.transform(pressure_position)
        return_position_z = price_3bar.transform(return_position)
        valid = np.isfinite(pressure_position) & np.isfinite(return_position)
        values = np.full(len(output), np.nan, dtype=float)
        values[valid] = np.sign(pressure_position[valid]) * (
            np.abs(pressure_position_z[valid]) - np.abs(return_position_z[valid])
        )
        output[f"_signed_absorption_divergence_3bar_position_{position}"] = values
        persistence_values[:, position] = values
    complete = np.isfinite(persistence_values).all(axis=1)
    persistence = np.full(len(output), np.nan, dtype=float)
    persistence[complete] = np.mean(np.sign(persistence_values[complete]), axis=1)
    output["accumulation_sign_persistence_25"] = persistence
    return output


def fit_quiet_score_parameters(development: pd.DataFrame) -> QuietScoreParameters:
    """Fit all quiet-score standardisation using development rows only."""

    if "partition" not in development.columns:
        raise ValueError("quiet-score preprocessing requires an explicit partition")
    if not development["partition"].astype(str).eq("development").all():
        raise ValueError("quiet-score preprocessing may fit on development rows only")
    required = {
        "pressure_sum_25",
        "net_return_25",
        "range_sum_25",
        "path_length_25",
        *set(QUIET_SIGNED_COMPONENTS).difference(
            {"signed_absorption_divergence_25", "accumulation_sign_persistence_25"}
        ),
        *_three_bar_columns("pressure_sum"),
        *_three_bar_columns("net_return"),
    }
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"quiet-score fit inputs missing: {missing}")
    pressure_25 = fit_robust_location_scale(
        pd.to_numeric(development["pressure_sum_25"], errors="coerce").to_numpy(float)
    )
    price_25 = fit_robust_location_scale(
        pd.to_numeric(development["net_return_25"], errors="coerce").to_numpy(float)
    )
    pressure_3_values = development.loc[:, _three_bar_columns("pressure_sum")].to_numpy(float)
    price_3_values = development.loc[:, _three_bar_columns("net_return")].to_numpy(float)
    pressure_3bar = fit_robust_location_scale(pressure_3_values.ravel())
    price_3bar = fit_robust_location_scale(price_3_values.ravel())
    enriched = _derive_divergence_columns(
        development,
        pressure_25=pressure_25,
        price_25=price_25,
        pressure_3bar=pressure_3bar,
        price_3bar=price_3bar,
    )
    components = {
        column: fit_robust_location_scale(
            pd.to_numeric(enriched[column], errors="coerce").to_numpy(float)
        )
        for column in QUIET_SIGNED_COMPONENTS
    }
    return QuietScoreParameters(
        fit_partition="development",
        pressure_25=pressure_25,
        price_25=price_25,
        pressure_3bar=pressure_3bar,
        price_3bar=price_3bar,
        range_sum=fit_robust_location_scale(
            pd.to_numeric(development["range_sum_25"], errors="coerce").to_numpy(float)
        ),
        path_length=fit_robust_location_scale(
            pd.to_numeric(development["path_length_25"], errors="coerce").to_numpy(float)
        ),
        component_parameters=components,
    )


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty(len(values), dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def apply_quiet_score_parameters(
    frame: pd.DataFrame,
    parameters: QuietScoreParameters,
) -> pd.DataFrame:
    """Apply one frozen development parameter set and equal-weight score."""

    if parameters.fit_partition != "development":
        raise ValueError("quiet-score parameters must be development-fitted")
    output = _derive_divergence_columns(
        frame,
        pressure_25=parameters.pressure_25,
        price_25=parameters.price_25,
        pressure_3bar=parameters.pressure_3bar,
        price_3bar=parameters.price_3bar,
    )
    range_z = parameters.range_sum.transform(
        pd.to_numeric(output["range_sum_25"], errors="coerce").to_numpy(float)
    )
    path_z = parameters.path_length.transform(
        pd.to_numeric(output["path_length_25"], errors="coerce").to_numpy(float)
    )
    output["range_compression_25"] = -range_z
    output["path_compression_25"] = -path_z
    output["quietness_25"] = _sigmoid(-range_z) * _sigmoid(-path_z)
    component_z_columns: list[str] = []
    for column in QUIET_SIGNED_COMPONENTS:
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
        standardized = parameters.component_parameters[column].transform(values)
        name = f"{column}__clipped_z"
        output[name] = np.clip(
            standardized,
            parameters.component_clip_lower,
            parameters.component_clip_upper,
        )
        component_z_columns.append(name)
    output["signed_accumulation_core_25"] = output.loc[:, component_z_columns].mean(axis=1)
    output["quiet_absorption_score_25"] = (
        output["quietness_25"] * output["signed_accumulation_core_25"]
    )
    return output


def _prepare_bar_primitives(completed_bars: pd.DataFrame) -> pd.DataFrame:
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
        "vti__bar_log_return",
        "historical_relative_activity",
        "signed_pressure",
    }
    missing = sorted(required.difference(completed_bars.columns))
    if missing:
        raise ValueError(f"completed-bar inputs missing: {missing}")
    bars = completed_bars.copy()
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    bars = bars.sort_values(
        ["stock", "bar_complete_timestamp", "session", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ValueError("completed-bar identity must be unique")
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    bars["_previous_close"] = bars.groupby("stock", sort=False)["close"].shift()
    bars["_r"] = np.log(bars["close"] / bars["_previous_close"])
    bars["_market_r"] = pd.to_numeric(bars["vti__bar_log_return"], errors="coerce")
    bars["_relative_r"] = bars["_r"] - bars["_market_r"]
    bars["_normalised_range"] = (bars["high"] - bars["low"]) / bars["_previous_close"]
    denominator = bars["high"] - bars["low"] + EPSILON
    bars["_clv"] = (2.0 * bars["close"] - bars["high"] - bars["low"]) / denominator
    lower_wick = np.minimum(bars["open"], bars["close"]) - bars["low"]
    upper_wick = bars["high"] - np.maximum(bars["open"], bars["close"])
    bars["_wick_asymmetry"] = (lower_wick - upper_wick) / denominator
    bars["_activity_proxy"] = pd.to_numeric(bars["historical_relative_activity"], errors="coerce")
    bars["_signed_pressure"] = pd.to_numeric(bars["signed_pressure"], errors="coerce")

    atr_values = np.full(len(bars), np.nan, dtype=float)
    for _, indices in bars.groupby("stock", sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        stock_rows = bars.loc[positions]
        previous = stock_rows["_previous_close"].to_numpy(float)
        high = stock_rows["high"].to_numpy(float)
        low = stock_rows["low"].to_numpy(float)
        true_range = np.maximum.reduce(
            [
                high - low,
                np.abs(high - previous),
                np.abs(low - previous),
            ]
        )
        prior_atr = (
            pd.Series(true_range)
            .shift(1)
            .rolling(ATR_WINDOW_BARS, min_periods=ATR_WINDOW_BARS)
            .mean()
            .to_numpy(float)
        )
        atr_values[positions] = prior_atr
    bars["_prior_completed_atr"] = atr_values

    bars["_vwap"] = math.nan
    for _, indices in bars.groupby(["stock", "session"], sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        session_rows = bars.loc[positions]
        typical = (
            session_rows["high"].to_numpy(float)
            + session_rows["low"].to_numpy(float)
            + session_rows["close"].to_numpy(float)
        ) / 3.0
        if "volume" in session_rows.columns:
            weights = pd.to_numeric(session_rows["volume"], errors="coerce").to_numpy(float)
        else:
            weights = session_rows["_activity_proxy"].to_numpy(float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        cumulative_weight = np.cumsum(weights)
        cumulative_value = np.cumsum(typical * weights)
        vwap = np.divide(
            cumulative_value,
            cumulative_weight,
            out=np.full(len(session_rows), np.nan, dtype=float),
            where=cumulative_weight > 0.0,
        )
        bars.loc[positions, "_vwap"] = vwap
    bars["_vwap_distance"] = (bars["close"] - bars["_vwap"]) / (
        bars["_prior_completed_atr"] + EPSILON
    )

    break_values = np.full(len(bars), np.nan, dtype=float)
    for _, indices in bars.groupby(["stock", "session"], sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        session_rows = bars.loc[positions].sort_values("bar_ordinal", kind="mergesort")
        ordered_positions = session_rows.index.to_numpy(int)
        highs = session_rows["high"].to_numpy(float)
        lows = session_rows["low"].to_numpy(float)
        closes = session_rows["close"].to_numpy(float)
        for offset in range(6, len(session_rows)):
            break_values[ordered_positions[offset]] = bar_break_failure_asymmetry(
                high=float(highs[offset]),
                low=float(lows[offset]),
                close=float(closes[offset]),
                prior_highs=highs[offset - 6 : offset],
                prior_lows=lows[offset - 6 : offset],
            )
    bars["_break_failure_asymmetry"] = break_values
    return bars


def _window_summary(
    prefix: pd.DataFrame,
    *,
    start_ordinal: int,
    end_ordinal: int,
    window_bars: int,
) -> dict[str, float]:
    window = prefix.loc[prefix["bar_ordinal"].between(start_ordinal, end_ordinal)].sort_values(
        "bar_ordinal", kind="mergesort"
    )
    suffix = str(window_bars * 5)
    if len(window) != window_bars:
        return {
            f"net_return_{suffix}": math.nan,
            f"path_length_{suffix}": math.nan,
            f"range_sum_{suffix}": math.nan,
            f"directional_efficiency_{suffix}": math.nan,
            f"pressure_sum_{suffix}": math.nan,
            f"pressure_persistence_{suffix}": math.nan,
            f"pressure_slope_{suffix}": math.nan,
            f"activity_without_displacement_{suffix}": math.nan,
            f"relative_resilience_{suffix}": math.nan,
            f"mean_clv_{suffix}": math.nan,
            f"mean_wick_asymmetry_{suffix}": math.nan,
            f"break_failure_asymmetry_{suffix}": math.nan,
            f"mean_vwap_distance_{suffix}": math.nan,
            f"vwap_side_balance_{suffix}": math.nan,
            f"vwap_reclaim_balance_{suffix}": math.nan,
        }
    returns = window["_r"].to_numpy(float)
    path = np.abs(returns)
    net_return = _complete_sum(returns)
    path_length = _complete_sum(path)
    ranges = window["_normalised_range"].to_numpy(float)
    pressure = window["_signed_pressure"].to_numpy(float)
    activity = window["_activity_proxy"].to_numpy(float)
    relative = window["_relative_r"].to_numpy(float)
    vwap_distance = window["_vwap_distance"].to_numpy(float)
    close = window["close"].to_numpy(float)
    vwap = window["_vwap"].to_numpy(float)
    if np.isfinite(close).all() and np.isfinite(vwap).all():
        above = close > vwap
        below = close < vwap
        above_count = int(np.count_nonzero(above))
        below_count = int(np.count_nonzero(below))
        side_balance = float((above_count - below_count) / window_bars)
        below_to_above = int(np.sum(below[:-1] & above[1:]))
        above_to_below = int(np.sum(above[:-1] & below[1:]))
        reclaim_balance = (
            float((below_to_above - above_to_below) / (window_bars - 1)) if window_bars > 1 else 0.0
        )
    else:
        side_balance = math.nan
        reclaim_balance = math.nan
    return {
        f"net_return_{suffix}": net_return,
        f"path_length_{suffix}": path_length,
        f"range_sum_{suffix}": _complete_sum(ranges),
        f"directional_efficiency_{suffix}": (
            net_return / (path_length + EPSILON)
            if math.isfinite(net_return) and math.isfinite(path_length)
            else math.nan
        ),
        f"pressure_sum_{suffix}": _complete_sum(pressure),
        f"pressure_persistence_{suffix}": pressure_persistence(pressure),
        f"pressure_slope_{suffix}": pressure_slope(pressure),
        f"activity_without_displacement_{suffix}": activity_without_displacement(
            pressure_sum=_complete_sum(pressure),
            activity=activity,
            net_return=net_return,
            path_length=path_length,
        ),
        f"relative_resilience_{suffix}": _complete_sum(relative),
        f"mean_clv_{suffix}": _complete_mean(window["_clv"].to_numpy(float)),
        f"mean_wick_asymmetry_{suffix}": _complete_mean(window["_wick_asymmetry"].to_numpy(float)),
        f"break_failure_asymmetry_{suffix}": _complete_sum(
            window["_break_failure_asymmetry"].to_numpy(float)
        ),
        f"mean_vwap_distance_{suffix}": _complete_mean(vwap_distance),
        f"vwap_side_balance_{suffix}": side_balance,
        f"vwap_reclaim_balance_{suffix}": reclaim_balance,
    }


def _trailing_sum(rows: pd.DataFrame, column: str, bars: int) -> float:
    values = rows[column].tail(bars).to_numpy(float)
    return _complete_sum(values) if len(values) == bars else math.nan


def _normalised_distance(value: float, reference: float, atr: float) -> float:
    if not all(math.isfinite(item) for item in (value, reference, atr)):
        return math.nan
    return (value - reference) / (atr + EPSILON)


def build_pretrigger_feature_rows(
    episodes: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Build the binding T-1 feature rows and secondary window diagnostics."""

    episode_required = {
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
    }
    missing_episode = sorted(episode_required.difference(episodes.columns))
    if missing_episode:
        raise ValueError(f"episode inputs missing: {missing_episode}")
    bars = _prepare_bar_primitives(completed_bars)
    output = episodes.copy().reset_index(drop=True)
    output["signal_timestamp"] = pd.to_datetime(
        output["signal_timestamp"], utc=True, errors="raise"
    )
    output["prospective_entry_timestamp"] = pd.to_datetime(
        output["prospective_entry_timestamp"], utc=True, errors="raise"
    )
    feature_rows: list[dict[str, object]] = []
    for episode in output.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = _as_int(episode.checkpoint)
        trigger_ordinal = checkpoint - 1
        marker_ordinal = checkpoint - 2
        session_bars = bars.loc[
            bars["stock"].astype(str).eq(stock) & bars["session"].astype(str).eq(session)
        ].sort_values("bar_ordinal", kind="mergesort")
        trigger = session_bars.loc[session_bars["bar_ordinal"].eq(trigger_ordinal)]
        marker = session_bars.loc[session_bars["bar_ordinal"].eq(marker_ordinal)]
        entry = session_bars.loc[session_bars["bar_ordinal"].eq(checkpoint)]
        if len(trigger) != 1 or len(marker) != 1 or len(entry) != 1:
            identity = f"{stock}|{session}|{checkpoint}"
            raise ValueError(f"missing trigger, marker, or entry for {identity}")
        trigger_row = trigger.iloc[0]
        marker_row = marker.iloc[0]
        entry_row = entry.iloc[0]
        trigger_timestamp = _as_timestamp(trigger_row["bar_complete_timestamp"])
        marker_timestamp = _as_timestamp(marker_row["bar_complete_timestamp"])
        if trigger_timestamp != _as_timestamp(episode.signal_timestamp):
            raise ValueError("frozen M1 signal timestamp is not the trigger-bar close")
        if _as_timestamp(entry_row["bar_start_timestamp"]) != _as_timestamp(
            episode.prospective_entry_timestamp
        ):
            raise ValueError("prospective entry is not the first post-trigger bar open")
        if not marker_timestamp < trigger_timestamp:
            raise ValueError("pre-trigger marker must precede the trigger-bar close")
        prefix = session_bars.loc[session_bars["bar_ordinal"].le(marker_ordinal)].copy()
        primary_start = marker_ordinal - PRIMARY_WINDOW_BARS + 1
        primary = prefix.loc[
            prefix["bar_ordinal"].between(primary_start, marker_ordinal)
        ].sort_values("bar_ordinal", kind="mergesort")
        if len(primary) != PRIMARY_WINDOW_BARS:
            raise ValueError(
                f"insufficient primary pre-trigger history for {stock}|{session}|{checkpoint}"
            )
        ordinals = primary["bar_ordinal"].astype(int).tolist()
        if ordinals != list(range(primary_start, marker_ordinal + 1)):
            raise ValueError("primary pre-trigger bars are not contiguous")

        values: dict[str, object] = {
            "trigger_bar_ordinal": trigger_ordinal,
            "marker_bar_ordinal": marker_ordinal,
            "trigger_timestamp": trigger_timestamp,
            "pretrigger_marker_timestamp": marker_timestamp,
            "maximum_direction_feature_timestamp": marker_timestamp,
            "primary_window_bar_ordinals": ",".join(str(value) for value in ordinals),
            "primary_window_bars_present": len(primary),
            "trigger_bar_excluded": True,
            "signed_pressure_bars_present_25": int(
                np.isfinite(primary["_signed_pressure"].to_numpy(float)).sum()
            ),
        }
        for window_bars in (3, 5, 9):
            values.update(
                _window_summary(
                    prefix,
                    start_ordinal=marker_ordinal - window_bars + 1,
                    end_ordinal=marker_ordinal,
                    window_bars=window_bars,
                )
            )

        for position, end_ordinal in enumerate(ordinals):
            three_bar = prefix.loc[
                prefix["bar_ordinal"].between(end_ordinal - 2, end_ordinal)
            ].sort_values("bar_ordinal", kind="mergesort")
            if len(three_bar) == 3:
                values[f"_pressure_sum_3bar_position_{position}"] = _complete_sum(
                    three_bar["_signed_pressure"].to_numpy(float)
                )
                values[f"_net_return_3bar_position_{position}"] = _complete_sum(
                    three_bar["_r"].to_numpy(float)
                )
            else:
                values[f"_pressure_sum_3bar_position_{position}"] = math.nan
                values[f"_net_return_3bar_position_{position}"] = math.nan

        marker_close = _as_float(marker_row["close"])
        marker_atr = _as_float(marker_row["_prior_completed_atr"])
        market_5 = _trailing_sum(prefix, "_market_r", 1)
        market_10 = _trailing_sum(prefix, "_market_r", 2)
        values.update(
            {
                "stock_return_5m_tminus1": _trailing_sum(prefix, "_r", 1),
                "stock_return_10m_tminus1": _trailing_sum(prefix, "_r", 2),
                "stock_return_20m_tminus1": _trailing_sum(prefix, "_r", 4),
                "market_return_5m_tminus1": market_5,
                "market_return_10m_tminus1": market_10,
                "market_return_20m_tminus1": _trailing_sum(prefix, "_market_r", 4),
                "stock_minus_market_return_5m_tminus1": (
                    _trailing_sum(prefix, "_r", 1) - market_5
                    if math.isfinite(_trailing_sum(prefix, "_r", 1)) and math.isfinite(market_5)
                    else math.nan
                ),
                "stock_minus_market_return_10m_tminus1": (
                    _trailing_sum(prefix, "_r", 2) - market_10
                    if math.isfinite(_trailing_sum(prefix, "_r", 2)) and math.isfinite(market_10)
                    else math.nan
                ),
                "distance_from_vwap_tminus1": _normalised_distance(
                    marker_close,
                    _as_float(marker_row["_vwap"]),
                    marker_atr,
                ),
                "distance_from_session_open_tminus1": _normalised_distance(
                    marker_close,
                    _as_float(session_bars.iloc[0]["open"]),
                    marker_atr,
                ),
                "clv_tminus1": _as_float(marker_row["_clv"]),
                "wick_asymmetry_tminus1": _as_float(marker_row["_wick_asymmetry"]),
                "checkpoint_indicator": str(checkpoint),
                "day_of_week_indicator": str(pd.Timestamp(session).day_name()),
                "_window_vwap_mean_reference": _complete_mean(primary["_vwap"].to_numpy(float)),
            }
        )
        first_six = prefix.loc[prefix["bar_ordinal"].between(0, 5)]
        if len(first_six) == 6:
            opening_midpoint = 0.5 * (
                float(first_six["high"].max()) + float(first_six["low"].min())
            )
        else:
            opening_midpoint = math.nan
        values["distance_from_opening_range_midpoint_tminus1"] = _normalised_distance(
            marker_close,
            opening_midpoint,
            marker_atr,
        )
        prior_six = prefix.loc[prefix["bar_ordinal"].lt(marker_ordinal)].tail(6)
        if len(prior_six) == 6:
            prior_high = float(prior_six["high"].max())
            prior_low = float(prior_six["low"].min())
        else:
            prior_high = math.nan
            prior_low = math.nan
        values["distance_from_previous_six_bar_high_tminus1"] = _normalised_distance(
            marker_close,
            prior_high,
            marker_atr,
        )
        values["distance_from_previous_six_bar_low_tminus1"] = _normalised_distance(
            marker_close,
            prior_low,
            marker_atr,
        )
        feature_rows.append(values)
    return pd.concat([output, pd.DataFrame(feature_rows, index=output.index)], axis=1)


def attach_pretrigger_direction_targets(
    episodes: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen post-trigger entry and underlying-stock outcomes."""

    episode_required = {
        "stock",
        "session",
        "checkpoint",
        "pretrigger_marker_timestamp",
        "prospective_entry_timestamp",
    }
    bar_required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
    }
    missing_episode = sorted(episode_required.difference(episodes.columns))
    missing_bar = sorted(bar_required.difference(completed_bars.columns))
    if missing_episode or missing_bar:
        raise ValueError(f"target inputs missing: episodes={missing_episode}, bars={missing_bar}")
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
    indexed = bars.set_index(["stock", "session", "bar_ordinal"])

    output = episodes.copy().reset_index(drop=True)
    output["pretrigger_marker_timestamp"] = pd.to_datetime(
        output["pretrigger_marker_timestamp"], utc=True, errors="raise"
    )
    output["prospective_entry_timestamp"] = pd.to_datetime(
        output["prospective_entry_timestamp"], utc=True, errors="raise"
    )
    target_rows: list[dict[str, object]] = []
    for episode in output.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = _as_int(episode.checkpoint)

        def bar_at(
            ordinal: int,
            *,
            stock_key: str = stock,
            session_key: str = session,
        ) -> pd.Series:
            try:
                result = cast(Any, indexed.loc)[(stock_key, session_key, ordinal)]
            except KeyError as error:
                raise ValueError(
                    f"missing target bar for {stock_key}|{session_key}|{ordinal}"
                ) from error
            if isinstance(result, pd.DataFrame):
                raise ValueError("completed-bar identity must be unique")
            return cast(pd.Series, result)

        marker = bar_at(checkpoint - 2)
        entry = bar_at(checkpoint)
        marker_timestamp = _as_timestamp(marker["bar_complete_timestamp"])
        entry_timestamp = _as_timestamp(entry["bar_start_timestamp"])
        if marker_timestamp != _as_timestamp(episode.pretrigger_marker_timestamp):
            raise ValueError("target marker timestamp drifted from T-1")
        if entry_timestamp != _as_timestamp(episode.prospective_entry_timestamp):
            raise ValueError("target entry drifted from first post-trigger bar open")
        marker_close = _as_float(marker["close"])
        entry_price = _as_float(entry["open"])
        if marker_close <= 0.0 or entry_price <= 0.0:
            raise ValueError("target prices must be positive")
        values: dict[str, object] = {
            "marker_close": marker_close,
            "entry_price": entry_price,
            "pre_entry_signed_return": math.log(entry_price / marker_close),
        }
        atm_iv = (
            _as_float(episode.atm_iv)
            if hasattr(episode, "atm_iv") and pd.notna(episode.atm_iv)
            else math.nan
        )
        for horizon in TARGET_HORIZONS_MINUTES:
            future_bars = horizon // 5
            close_bar = bar_at(checkpoint + future_bars - 1)
            close_price = _as_float(close_bar["close"])
            if close_price <= 0.0:
                raise ValueError("target horizon close must be positive")
            signed_return = math.log(close_price / entry_price)
            path = [bar_at(checkpoint + offset) for offset in range(future_bars)]
            highs = np.asarray([_as_float(row["high"]) for row in path], dtype=float)
            lows = np.asarray([_as_float(row["low"]) for row in path], dtype=float)
            favourable_call = float(max(0.0, np.max(np.log(highs / entry_price))))
            adverse_call = float(max(0.0, np.max(np.log(entry_price / lows))))
            values[f"close_{horizon}m"] = close_price
            values[f"signed_log_return_{horizon}m"] = signed_return
            values[f"absolute_log_return_{horizon}m"] = abs(signed_return)
            values[f"call_mfe_{horizon}m"] = favourable_call
            values[f"call_mae_{horizon}m"] = adverse_call
            values[f"put_mfe_{horizon}m"] = adverse_call
            values[f"put_mae_{horizon}m"] = favourable_call
            expectation = (
                atm_iv * math.sqrt(horizon / ANNUAL_TRADING_MINUTES) * math.sqrt(2.0 / math.pi)
                if math.isfinite(atm_iv) and atm_iv > 0.0
                else math.nan
            )
            values[f"iv_expected_absolute_{horizon}m"] = expectation
            values[f"iv_excess_{horizon}m"] = (
                int(abs(signed_return) > expectation) if math.isfinite(expectation) else math.nan
            )
        primary_return = _as_float(values["signed_log_return_10m"])
        values["zero_return_10m"] = int(primary_return == 0.0)
        values["direction_up_10m"] = (
            math.nan if primary_return == 0.0 else int(primary_return > 0.0)
        )
        for horizon in (10, 30):
            values[f"remaining_fraction_{horizon}m"] = remaining_fraction(
                _as_float(values["pre_entry_signed_return"]),
                _as_float(values[f"signed_log_return_{horizon}m"]),
            )
        target_rows.append(values)
    return pd.concat([output, pd.DataFrame(target_rows, index=output.index)], axis=1)


def freeze_confidence_boundary(
    probabilities: Sequence[float] | FloatArray,
    *,
    target_coverage: float = 0.35,
    minimum_actions: int = 100,
    weights: Sequence[float] | FloatArray | None = None,
) -> float:
    """Freeze one symmetric confidence boundary by deterministic weighted rank."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("OOF probabilities must be finite and non-empty")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    if minimum_actions < 1 or minimum_actions > len(values):
        raise ValueError("minimum actions must be supported by development rows")
    episode_weights = (
        np.ones(len(values), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        len(episode_weights) != len(values)
        or not np.isfinite(episode_weights).all()
        or np.any(episode_weights <= 0.0)
    ):
        raise ValueError("confidence weights must be finite, positive, and aligned")
    confidence = np.abs(values - 0.5)
    order = np.lexsort((np.arange(len(values)), -confidence))
    cumulative = np.cumsum(episode_weights[order])
    target_weight = target_coverage * float(np.sum(episode_weights))
    target_index = int(np.searchsorted(cumulative, target_weight, side="left"))
    minimum_index = minimum_actions - 1
    selected_index = min(len(values) - 1, max(target_index, minimum_index))
    return float(confidence[order[selected_index]])


def selective_actions(
    probabilities: Sequence[float] | FloatArray,
    boundary: float,
) -> StringArray:
    """Apply the frozen symmetric CALL/PUT/ABSTAIN rule."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or not 0.0 <= boundary <= 0.5:
        raise ValueError("selective policy inputs are invalid")
    actions = np.full(len(values), "ABSTAIN", dtype="<U7")
    actions[values >= 0.5 + boundary] = "CALL"
    actions[values <= 0.5 - boundary] = "PUT"
    return np.asarray(actions, dtype=np.str_)


def aligned_return(action: str, signed_return: float) -> float:
    """Underlying-stock return aligned to CALL (+1) or PUT (-1) direction."""

    if action == "CALL":
        side = 1.0
    elif action == "PUT":
        side = -1.0
    else:
        raise ValueError("aligned return requires CALL or PUT")
    return side * signed_return


def remaining_fraction(
    pre_entry_signed_return: float,
    post_entry_signed_return: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Absolute post-entry share of marker-to-entry plus post-entry movement."""

    if not all(
        math.isfinite(value) for value in (pre_entry_signed_return, post_entry_signed_return)
    ):
        return math.nan
    return abs(post_entry_signed_return) / (
        abs(pre_entry_signed_return) + abs(post_entry_signed_return) + epsilon
    )


def grouped_feature_permutation(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_columns: Sequence[str] = ("session", "checkpoint"),
    seed: int,
) -> pd.DataFrame:
    """Permute a complete feature bundle within session/checkpoint groups."""

    required = {*feature_columns, *group_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"grouped permutation inputs missing: {missing}")
    output = frame.copy()
    generator = np.random.default_rng(seed)
    for _, indices in frame.groupby(list(group_columns), sort=True, dropna=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        source = positions[generator.permutation(len(positions))]
        output.loc[positions, list(feature_columns)] = frame.loc[
            source, list(feature_columns)
        ].to_numpy()
    return output


def label_null_within_slates(
    frame: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> pd.DataFrame:
    """Permute labels among stocks within each development session/checkpoint slate."""

    required = {"session", "checkpoint", "stock", target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"label-null inputs missing: {missing}")
    output = frame.copy()
    generator = np.random.default_rng(seed)
    for _, indices in frame.groupby(
        ["session", "checkpoint"], sort=True, dropna=False
    ).groups.items():
        positions = np.asarray(indices, dtype=int)
        labels = frame.loc[positions, target_column].to_numpy(copy=True)
        output.loc[positions, target_column] = labels[generator.permutation(len(labels))]
    return output


def temporal_placebo_bundle(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Move each complete stock feature bundle to its next fresh episode."""

    required = {"stock", "pretrigger_marker_timestamp", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"temporal-placebo inputs missing: {missing}")
    output = frame.copy()
    ordering = frame.assign(
        _original_position=np.arange(len(frame), dtype=int),
        _timestamp=pd.to_datetime(frame["pretrigger_marker_timestamp"], utc=True, errors="raise"),
    ).sort_values(["stock", "_timestamp", "_original_position"], kind="mergesort")
    shifted = ordering.groupby("stock", sort=False)[list(feature_columns)].shift(1)
    output.loc[ordering["_original_position"].to_numpy(int), list(feature_columns)] = (
        shifted.to_numpy()
    )
    return output


def validate_authorized_sessions(sessions: Sequence[object] | pd.Series) -> None:
    """Reject excluded opened-holdout and protected sessions."""

    dates = pd.to_datetime(pd.Series(sessions), errors="raise").dt.normalize()
    if bool(dates.lt(pd.Timestamp(DEVELOPMENT_START)).any()):
        raise ValueError("a row predates the authorized development chronology")
    if bool(dates.ge(pd.Timestamp("2026-01-01")).any()):
        raise ValueError("protected outcomes are forbidden")
    if bool(dates.gt(pd.Timestamp(ASSESSMENT_END)).any()):
        raise ValueError("excluded opened-holdout outcomes are forbidden")


def decide_pretrigger_candidate(evidence: Mapping[str, object]) -> str:
    """Apply the frozen support and primary pass gates without relaxation."""

    blocker = evidence.get("blocker")
    if blocker is not None:
        decision = str(blocker)
        if decision not in DECISION_CATEGORIES or not decision.startswith("blocked_"):
            raise ValueError(f"unknown pre-trigger blocker: {decision}")
        return decision
    if not bool(evidence.get("development_support_passed", False)) or not bool(
        evidence.get("assessment_support_passed", False)
    ):
        return "blocked_insufficient_direction_episode_support"
    if not bool(evidence.get("selective_support_passed", False)):
        return "blocked_insufficient_selective_action_support"

    full_gate = all(
        (
            bool(evidence.get("concentration_gates_passed", False)),
            bool(evidence.get("q1_log_loss_improves", False)),
            bool(evidence.get("q1_brier_improves", False)),
            _as_float(evidence.get("q1_auc", -math.inf)) >= 0.55,
            _as_float(evidence.get("q1_balanced_accuracy", -math.inf)) > 0.52,
            0.20 <= _as_float(evidence.get("action_coverage", math.nan)) <= 0.50,
            _as_float(evidence.get("selective_accuracy", -math.inf)) >= 0.57,
            bool(evidence.get("beats_required_baselines", False)),
            _as_float(evidence.get("mean_aligned_return_10m", -math.inf)) > 0.0,
            _as_float(evidence.get("median_aligned_return_10m", -math.inf)) > 0.0,
            _as_float(evidence.get("bootstrap_80_accuracy_lower", -math.inf)) > 0.50,
            _as_float(evidence.get("bootstrap_80_mean_return_lower", -math.inf)) >= 0.0,
            _as_int(evidence.get("positive_month_groups", 0)) >= 6,
            bool(evidence.get("null_gate_passed", False)),
            bool(evidence.get("temporal_placebo_gate_passed", False)),
            bool(evidence.get("score_monotonic_direction_correct", False)),
            not bool(evidence.get("late_direction_problem", True)),
        )
    )
    if full_gate:
        return "pretrigger_quiet_accumulation_direction_candidate_supported"

    directional_strength = all(
        (
            bool(evidence.get("q1_log_loss_improves", False))
            or _as_float(evidence.get("q1_auc", -math.inf)) >= 0.55,
            _as_float(evidence.get("selective_accuracy", -math.inf)) >= 0.57,
            _as_float(evidence.get("mean_aligned_return_10m", -math.inf)) > 0.0,
        )
    )
    if directional_strength and bool(evidence.get("late_direction_problem", True)):
        return "pretrigger_direction_present_but_too_late"
    if bool(evidence.get("persistent_pressure_supported", False)) and not bool(
        evidence.get("absorption_response_supported", False)
    ):
        return "persistent_pressure_direction_supported_absorption_not_supported"
    if bool(evidence.get("absorption_response_promising", False)) or bool(
        evidence.get("absorption_response_supported", False)
    ):
        return "absorption_response_promising_but_full_gate_not_met"
    if bool(evidence.get("score_descriptive_only", False)):
        return "quiet_accumulation_score_descriptive_only"
    if bool(evidence.get("stability_failed", False)):
        return "pretrigger_quiet_accumulation_unstable"
    return "no_incremental_pretrigger_directional_signal"
