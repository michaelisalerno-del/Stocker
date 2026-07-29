"""Causal primitives for the Observable Behavioural-State Dimensions Screen V0.

The vocabulary is a participant-behaviour metaphor only.  Every value is a
continuous transformation of completed five-minute price or historical activity
rows.  This research-only module has no execution or production-runtime surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

EPSILON = 1e-12
FloatArray = NDArray[np.float64]
DIMENSION_FEATURES = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)
CONJUNCTION_FEATURES = (
    "active_conviction",
    "active_frustration",
    "pressurised_tension",
    "pressurised_exhaustion",
    "independent_pressure",
)
DESCRIPTIVE_LABELS = (
    "CALM",
    "TENSE",
    "CONFLICTED",
    "BULLISH_PRESSURE",
    "BEARISH_PRESSURE",
    "UPWARD_PRESSURE_EXHAUSTING",
    "DOWNWARD_PRESSURE_EXHAUSTING",
    "INDEPENDENT",
)
DECISION_CATEGORIES = (
    "behavioural_dimensions_add_movement_and_direction",
    "behavioural_dimensions_add_movement_only",
    "behavioural_dimensions_add_direction_only",
    "behavioural_conjunctions_only",
    "behavioural_descriptions_only_no_predictive_increment",
    "no_behavioural_state_increment",
    "blocked_observable_predecessor_not_reconstructable",
    "blocked_protected_boundary_failure",
    "blocked_chronology_or_leakage_failure",
    "blocked_insufficient_behavioural_support",
    "blocked_quick_behavioural_screen_resource_limit",
    "blocked_model_convergence_failure",
    "blocked_reproducibility_or_audit_failure",
)
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
FORBIDDEN_MODEL_FEATURE_FRAGMENTS = (
    "regime",
    "state",
    "loop",
    "closure",
    "excursion",
    "transition",
    "posterior",
    "structural_score",
    "future_price",
    "future_activity",
    "future_return",
    "mfe",
    "mae",
    "p&l",
    "pnl",
    "profit_history",
    "news",
    "bid_ask",
    "order_book",
    "broker",
    "symbol",
    "month",
    "behavioural_label",
)


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> FloatArray:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("component inputs must be finite")
    return values


def _checkpoint_ordinal(value: object) -> int:
    ordinal = int(cast(Any, value))
    if float(cast(Any, value)) != float(ordinal):
        raise ValueError(f"checkpoint must be integral: {value}")
    return ordinal


def bar_component_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate fixed causal components for completed opening bars.

    The first return and true-range denominator is the regular-session open, so
    the overnight gap is not introduced into opening-window effort.  Later bars
    use the previous completed close.  Provider volume enters only through the
    precomputed causal ``historical_relative_activity`` ratio.
    """

    required = ("open", "high", "low", "close", "historical_relative_activity")
    _finite_numeric(bars, required)
    if bars.empty:
        raise ValueError("at least one completed bar is required")
    output = bars.reset_index(drop=True).copy()
    prices = output.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if bool((prices <= 0.0).any()):
        raise ValueError("prices must be positive")
    if bool((output["historical_relative_activity"].to_numpy(dtype=float) < 0.0).any()):
        raise ValueError("historical relative activity must be non-negative")

    previous_close = output["close"].shift(1).to_numpy(dtype=np.float64)
    previous_close[0] = float(output.iloc[0]["open"])
    close = output["close"].to_numpy(dtype=np.float64)
    high = output["high"].to_numpy(dtype=np.float64)
    low = output["low"].to_numpy(dtype=np.float64)
    open_ = output["open"].to_numpy(dtype=np.float64)
    width = high - low
    if bool((width < 0.0).any()):
        raise ValueError("bar high must not be below bar low")

    output["return_bps"] = 10_000.0 * (close / previous_close - 1.0)
    output["true_range_bps"] = (
        10_000.0
        * np.maximum.reduce([width, np.abs(high - previous_close), np.abs(low - previous_close)])
        / previous_close
    )
    nonzero = width > EPSILON
    close_location = np.full(len(output), 0.5, dtype=np.float64)
    upper_wick = np.zeros(len(output), dtype=np.float64)
    lower_wick = np.zeros(len(output), dtype=np.float64)
    close_location[nonzero] = (close[nonzero] - low[nonzero]) / width[nonzero]
    upper_wick[nonzero] = (high[nonzero] - np.maximum(open_[nonzero], close[nonzero])) / width[
        nonzero
    ]
    lower_wick[nonzero] = (np.minimum(open_[nonzero], close[nonzero]) - low[nonzero]) / width[
        nonzero
    ]
    output["close_location"] = np.clip(close_location, 0.0, 1.0)
    output["upper_wick_fraction"] = np.clip(upper_wick, 0.0, 1.0)
    output["lower_wick_fraction"] = np.clip(lower_wick, 0.0, 1.0)
    return output


def _least_squares_slope(values: FloatArray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    centered_x = x - float(x.mean())
    denominator = float(centered_x @ centered_x)
    if denominator <= EPSILON:
        return 0.0
    return float(centered_x @ (values - float(values.mean())) / denominator)


def opening_raw_components(
    completed_bars: pd.DataFrame,
    *,
    trailing_opening_range_median_bps: float,
    signed_progress_bps: float,
    signed_progress_acceleration_bps: float,
    return_gap_bps: float,
) -> dict[str, float]:
    """Aggregate completed bar components into preregistered raw components."""

    required = (
        "open",
        "high",
        "low",
        "close",
        "historical_relative_activity",
        "return_bps",
        "true_range_bps",
        "close_location",
        "upper_wick_fraction",
        "lower_wick_fraction",
    )
    _finite_numeric(completed_bars, required)
    bars = completed_bars.reset_index(drop=True)
    count = len(bars)
    if count < 2 or count % 2:
        raise ValueError("opening window must contain an even number of completed bars")
    if not np.isfinite(trailing_opening_range_median_bps) or (
        trailing_opening_range_median_bps <= 0.0
    ):
        raise ValueError("trailing opening-range median must be positive and finite")

    returns = bars["return_bps"].to_numpy(dtype=np.float64)
    true_ranges = bars["true_range_bps"].to_numpy(dtype=np.float64)
    relative_activity = bars["historical_relative_activity"].to_numpy(dtype=np.float64)
    highs = bars["high"].to_numpy(dtype=np.float64)
    lows = bars["low"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    session_open = float(bars.iloc[0]["open"])
    opening_range = float(highs.max() - lows.min())
    opening_range_bps = 10_000.0 * opening_range / session_open

    previous_high = np.maximum.accumulate(highs)
    previous_low = np.minimum.accumulate(lows)
    new_high = np.ones(count, dtype=bool)
    new_low = np.ones(count, dtype=bool)
    new_high[1:] = highs[1:] > previous_high[:-1]
    new_low[1:] = lows[1:] < previous_low[:-1]
    up_rejection = float(bars.loc[new_high, "upper_wick_fraction"].mean())
    down_rejection = float(bars.loc[new_low, "lower_wick_fraction"].mean())

    signed_efficiency = float(returns.sum() / max(float(np.abs(returns).sum()), EPSILON))
    cumulative_return_bps = 10_000.0 * (closes[-1] / session_open - 1.0)
    if abs(cumulative_return_bps) <= EPSILON:
        persistence = 0.5
    else:
        direction = float(np.sign(cumulative_return_bps))
        persistence = float(np.mean(np.sign(returns) == direction))

    half = count // 2
    log_activity = np.log1p(relative_activity)
    activity_acceleration = float(log_activity[half:].mean() - log_activity[:half].mean())
    range_acceleration = float(true_ranges[half:].mean() - true_ranges[:half].mean())
    high_slope = _least_squares_slope(highs) / max(opening_range, EPSILON)
    low_slope = _least_squares_slope(lows) / max(opening_range, EPSILON)

    activity_effort = float(np.log1p(relative_activity.mean()))
    range_effort = float(np.log1p(true_ranges.sum()))
    travel_effort = float(np.log1p(np.abs(returns).sum()))
    return {
        "activity_effort": activity_effort,
        "range_effort": range_effort,
        "travel_effort": travel_effort,
        "signed_progress": float(signed_progress_bps),
        "absolute_progress": abs(float(signed_progress_bps)),
        "signed_efficiency": signed_efficiency,
        "absolute_efficiency": abs(signed_efficiency),
        "close_retention": abs(closes[-1] - session_open)
        / max(float(np.sum(highs - lows)), EPSILON),
        "directional_persistence": persistence,
        "new_high_fraction": float(new_high.mean()),
        "new_low_fraction": float(new_low.mean()),
        "up_extreme_rejection": up_rejection,
        "down_extreme_rejection": down_rejection,
        "extreme_rejection": 0.5 * (up_rejection + down_rejection),
        "opening_range_bps": opening_range_bps,
        "trailing_opening_range_median_bps": float(trailing_opening_range_median_bps),
        "range_ratio": opening_range_bps / trailing_opening_range_median_bps,
        "compression": -float(
            np.log(max(opening_range_bps / trailing_opening_range_median_bps, EPSILON))
        ),
        "normalised_high_slope": float(high_slope),
        "normalised_low_slope": float(low_slope),
        "boundary_slope": 0.5 * float(high_slope + low_slope),
        "activity_acceleration": activity_acceleration,
        "range_acceleration": range_acceleration,
        "effort_acceleration": 0.5 * (activity_acceleration + range_acceleration),
        "signed_progress_acceleration": float(signed_progress_acceleration_bps),
        "return_gap": float(return_gap_bps),
        "mean_close_location": float(bars["close_location"].mean()),
    }


@dataclass(frozen=True, slots=True)
class RobustComponentScale:
    """Frozen development-only median/IQR scaling record."""

    center: float
    scale: float
    clip_lower: float = -5.0
    clip_upper: float = 5.0

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "method": "median_iqr",
            "center": self.center,
            "scale": self.scale,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
        }


def fit_component_scaling(
    development: pd.DataFrame,
    *,
    components: Sequence[str],
    checkpoint_column: str = "decision_ordinal",
) -> dict[int, dict[str, RobustComponentScale]]:
    """Fit checkpoint-specific median/IQR parameters on development rows only."""

    if development.empty:
        raise ValueError("development scaling rows are required")
    _finite_numeric(development, (checkpoint_column, *components))
    fitted: dict[int, dict[str, RobustComponentScale]] = {}
    for checkpoint, rows in development.groupby(checkpoint_column, sort=True):
        ordinal = _checkpoint_ordinal(checkpoint)
        fitted[ordinal] = {}
        for component in components:
            values = rows[component].astype(float)
            center = float(values.median())
            lower = float(values.quantile(0.25, interpolation="linear"))
            upper = float(values.quantile(0.75, interpolation="linear"))
            scale = upper - lower
            if not np.isfinite(scale) or scale < EPSILON:
                scale = 1.0
            fitted[ordinal][component] = RobustComponentScale(center=center, scale=scale)
    return fitted


def apply_component_scaling(
    frame: pd.DataFrame,
    scaling: Mapping[int, Mapping[str, RobustComponentScale]],
    *,
    components: Sequence[str],
    checkpoint_column: str = "decision_ordinal",
) -> pd.DataFrame:
    """Apply frozen development parameters and clip every standardized value."""

    _finite_numeric(frame, (checkpoint_column, *components))
    output = frame.copy()
    for component in components:
        output[f"z_{component}"] = np.nan
    for checkpoint, indices in output.groupby(checkpoint_column, sort=True).groups.items():
        ordinal = _checkpoint_ordinal(checkpoint)
        if ordinal not in scaling:
            raise ValueError(f"scaling unavailable for checkpoint {ordinal}")
        index = list(indices)
        for component in components:
            if component not in scaling[ordinal]:
                raise ValueError(f"scaling unavailable for {ordinal}/{component}")
            frozen = scaling[ordinal][component]
            values = (output.loc[index, component].to_numpy(dtype=float) - frozen.center) / (
                frozen.scale
            )
            output.loc[index, f"z_{component}"] = np.clip(
                values, frozen.clip_lower, frozen.clip_upper
            )
    z_columns = [f"z_{component}" for component in components]
    if not np.isfinite(output.loc[:, z_columns].to_numpy(dtype=float)).all():
        raise ValueError("standardized components must be finite")
    return output


def _signed_pressure(frame: pd.DataFrame) -> FloatArray:
    columns = (
        "z_signed_progress",
        "z_signed_efficiency",
        "z_mean_close_location",
        "z_boundary_slope",
    )
    values = _finite_numeric(frame, columns)
    return np.asarray(values.mean(axis=1), dtype=np.float64)


def derive_exhaustion_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Align progress acceleration and rejection with the current pressure side."""

    columns = (
        "signed_pressure",
        "signed_progress_acceleration",
        "up_extreme_rejection",
        "down_extreme_rejection",
    )
    _finite_numeric(frame, columns)
    output = frame.copy()
    pressure = output["signed_pressure"].to_numpy(dtype=float)
    pressure_sign = np.sign(pressure)
    pressure_sign[np.abs(pressure) <= EPSILON] = 0.0
    progress = output["signed_progress_acceleration"].to_numpy(dtype=float)
    upward = output["up_extreme_rejection"].to_numpy(dtype=float)
    downward = output["down_extreme_rejection"].to_numpy(dtype=float)
    rejection = np.where(
        pressure_sign > 0.0,
        upward,
        np.where(pressure_sign < 0.0, downward, 0.5 * (upward + downward)),
    )
    output["pressure_sign"] = pressure_sign
    output["aligned_progress_acceleration"] = pressure_sign * progress
    output["directional_rejection"] = rejection
    return output


def derive_behavioural_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the ten fixed equal-weight continuous behavioural dimensions."""

    required = (
        "z_activity_effort",
        "z_range_effort",
        "z_travel_effort",
        "z_absolute_efficiency",
        "z_close_retention",
        "z_directional_persistence",
        "z_extreme_rejection",
        "z_absolute_progress",
        "z_compression",
        "z_signed_progress",
        "z_signed_efficiency",
        "z_mean_close_location",
        "z_boundary_slope",
        "z_effort_acceleration",
        "z_aligned_progress_acceleration",
        "z_directional_rejection",
        "z_return_gap",
        "z_activity_gap",
        "z_range_gap",
        "return_gap",
    )
    _finite_numeric(frame, required)
    output = pd.DataFrame(index=frame.index)
    output["arousal"] = frame[["z_activity_effort", "z_range_effort", "z_travel_effort"]].mean(
        axis=1
    )
    output["conviction"] = frame[
        ["z_absolute_efficiency", "z_close_retention", "z_directional_persistence"]
    ].mean(axis=1)
    output["frustration"] = frame[
        ["z_activity_effort", "z_travel_effort", "z_extreme_rejection"]
    ].mean(axis=1) - frame[["z_absolute_progress", "z_absolute_efficiency"]].mean(axis=1)
    output["tension"] = (
        frame[["z_activity_effort", "z_compression", "z_extreme_rejection"]].mean(axis=1)
        - frame["z_absolute_progress"]
    )
    output["signed_pressure"] = _signed_pressure(frame)
    output["pressure_magnitude"] = output["signed_pressure"].abs()
    output["exhaustion_magnitude"] = (
        frame["z_effort_acceleration"]
        - frame["z_aligned_progress_acceleration"]
        + frame["z_directional_rejection"]
    )
    pressure_sign = np.sign(output["signed_pressure"].to_numpy(dtype=float))
    pressure_sign[np.abs(output["signed_pressure"].to_numpy(dtype=float)) <= EPSILON] = 0.0
    output["signed_exhaustion"] = pressure_sign * output["exhaustion_magnitude"]
    output["independence"] = (
        frame[["z_return_gap", "z_activity_gap", "z_range_gap"]].abs().mean(axis=1)
    )
    output["signed_independence"] = np.sign(frame["return_gap"]) * output["independence"]
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("behavioural dimensions must be finite")
    return output


def derive_conjunctions(dimensions: pd.DataFrame) -> pd.DataFrame:
    """Create exactly the five preregistered continuous conjunctions."""

    required = (
        "arousal",
        "conviction",
        "frustration",
        "tension",
        "signed_pressure",
        "pressure_magnitude",
        "exhaustion_magnitude",
        "independence",
    )
    _finite_numeric(dimensions, required)
    output = pd.DataFrame(index=dimensions.index)
    output["active_conviction"] = dimensions["arousal"] * dimensions["conviction"]
    output["active_frustration"] = dimensions["arousal"] * dimensions["frustration"]
    output["pressurised_tension"] = dimensions["tension"] * dimensions["pressure_magnitude"]
    output["pressurised_exhaustion"] = (
        dimensions["exhaustion_magnitude"] * dimensions["pressure_magnitude"]
    )
    output["independent_pressure"] = dimensions["independence"] * dimensions["signed_pressure"]
    return output


def fit_conjunction_bounds(
    development: pd.DataFrame,
    *,
    checkpoint_column: str = "decision_ordinal",
) -> dict[int, dict[str, tuple[float, float]]]:
    """Freeze checkpoint-specific development 1st/99th percentile bounds."""

    _finite_numeric(development, (checkpoint_column, *CONJUNCTION_FEATURES))
    bounds: dict[int, dict[str, tuple[float, float]]] = {}
    for checkpoint, rows in development.groupby(checkpoint_column, sort=True):
        bounds[_checkpoint_ordinal(checkpoint)] = {
            feature: (
                float(rows[feature].quantile(0.01, interpolation="linear")),
                float(rows[feature].quantile(0.99, interpolation="linear")),
            )
            for feature in CONJUNCTION_FEATURES
        }
    return bounds


def apply_conjunction_bounds(
    frame: pd.DataFrame,
    bounds: Mapping[int, Mapping[str, tuple[float, float]]],
    *,
    checkpoint_column: str = "decision_ordinal",
) -> pd.DataFrame:
    """Clip conjunctions to frozen development-only checkpoint bounds."""

    _finite_numeric(frame, (checkpoint_column, *CONJUNCTION_FEATURES))
    output = frame.copy()
    for checkpoint, indices in output.groupby(checkpoint_column, sort=True).groups.items():
        ordinal = _checkpoint_ordinal(checkpoint)
        if ordinal not in bounds:
            raise ValueError(f"conjunction bounds unavailable for checkpoint {ordinal}")
        index = list(indices)
        for feature in CONJUNCTION_FEATURES:
            lower, upper = bounds[ordinal][feature]
            output.loc[index, feature] = output.loc[index, feature].clip(lower, upper)
    return output


def fit_label_thresholds(
    development: pd.DataFrame,
    *,
    checkpoint_column: str = "decision_ordinal",
) -> dict[int, dict[str, float]]:
    """Fit only the fixed descriptive-label quantiles on development rows."""

    dimensions = (
        "arousal",
        "conviction",
        "frustration",
        "tension",
        "signed_pressure",
        "exhaustion_magnitude",
        "independence",
    )
    _finite_numeric(development, (checkpoint_column, *dimensions))
    thresholds: dict[int, dict[str, float]] = {}
    for checkpoint, rows in development.groupby(checkpoint_column, sort=True):
        thresholds[_checkpoint_ordinal(checkpoint)] = {
            "arousal_q30": float(rows["arousal"].quantile(0.30, interpolation="linear")),
            "tension_q70": float(rows["tension"].quantile(0.70, interpolation="linear")),
            "frustration_q70": float(rows["frustration"].quantile(0.70, interpolation="linear")),
            "signed_pressure_q30": float(
                rows["signed_pressure"].quantile(0.30, interpolation="linear")
            ),
            "signed_pressure_q70": float(
                rows["signed_pressure"].quantile(0.70, interpolation="linear")
            ),
            "conviction_q60": float(rows["conviction"].quantile(0.60, interpolation="linear")),
            "exhaustion_magnitude_q70": float(
                rows["exhaustion_magnitude"].quantile(0.70, interpolation="linear")
            ),
            "independence_q70": float(rows["independence"].quantile(0.70, interpolation="linear")),
        }
    return thresholds


def assign_descriptive_labels(
    frame: pd.DataFrame,
    thresholds: Mapping[int, Mapping[str, float]],
    *,
    checkpoint_column: str = "decision_ordinal",
) -> pd.DataFrame:
    """Attach reporting-only labels; no label is suitable as a model feature."""

    required = (
        checkpoint_column,
        "arousal",
        "conviction",
        "frustration",
        "tension",
        "signed_pressure",
        "exhaustion_magnitude",
        "independence",
    )
    _finite_numeric(frame, required)
    output = frame.copy()
    for label in DESCRIPTIVE_LABELS:
        output[f"label__{label}"] = False
    for checkpoint, indices in output.groupby(checkpoint_column, sort=True).groups.items():
        ordinal = _checkpoint_ordinal(checkpoint)
        if ordinal not in thresholds:
            raise ValueError(f"label thresholds unavailable for checkpoint {ordinal}")
        frozen = thresholds[ordinal]
        index = list(indices)
        rows = output.loc[index]
        output.loc[index, "label__CALM"] = rows["arousal"].le(frozen["arousal_q30"])
        output.loc[index, "label__TENSE"] = rows["tension"].ge(frozen["tension_q70"])
        output.loc[index, "label__CONFLICTED"] = rows["frustration"].ge(frozen["frustration_q70"])
        convicted = rows["conviction"].ge(frozen["conviction_q60"])
        output.loc[index, "label__BULLISH_PRESSURE"] = (
            rows["signed_pressure"].ge(frozen["signed_pressure_q70"]) & convicted
        )
        output.loc[index, "label__BEARISH_PRESSURE"] = (
            rows["signed_pressure"].le(frozen["signed_pressure_q30"]) & convicted
        )
        exhausting = rows["exhaustion_magnitude"].ge(frozen["exhaustion_magnitude_q70"])
        output.loc[index, "label__UPWARD_PRESSURE_EXHAUSTING"] = (
            rows["signed_pressure"].gt(0.0) & exhausting
        )
        output.loc[index, "label__DOWNWARD_PRESSURE_EXHAUSTING"] = (
            rows["signed_pressure"].lt(0.0) & exhausting
        )
        output.loc[index, "label__INDEPENDENT"] = rows["independence"].ge(
            frozen["independence_q70"]
        )
    label_columns = [f"label__{label}" for label in DESCRIPTIVE_LABELS]
    output["behavioural_label_count"] = output[label_columns].sum(axis=1).astype(np.int8)
    output["behavioural_labels"] = [
        "|".join(label for label in DESCRIPTIVE_LABELS if bool(output.at[index, f"label__{label}"]))
        for index in output.index
    ]
    return output


def equal_slate_weights(slate_ids: pd.Series) -> FloatArray:
    """Give every represented simultaneous slate total model weight one."""

    values = slate_ids.astype(str).reset_index(drop=True)
    if values.empty or values.isna().any():
        raise ValueError("slate weights require complete slate identifiers")
    sizes = values.groupby(values, sort=True).transform("size").to_numpy(dtype=np.float64)
    weights = np.asarray(1.0 / sizes, dtype=np.float64)
    totals = pd.Series(weights).groupby(values, sort=True).sum().to_numpy(dtype=np.float64)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("slate weights do not sum to one")
    return weights


def assert_allowed_model_features(feature_names: Sequence[str]) -> None:
    """Fail closed on structural, future, identity, broker, or label fields."""

    forbidden: list[str] = []
    label_fragments = tuple(label.casefold() for label in DESCRIPTIVE_LABELS)
    for name in feature_names:
        normalised = str(name).casefold().replace("/", "_").replace(" ", "_")
        if (
            any(fragment in normalised for fragment in FORBIDDEN_MODEL_FEATURE_FRAGMENTS)
            or normalised.startswith("label__")
            or normalised in label_fragments
        ):
            forbidden.append(str(name))
    if forbidden:
        raise ValueError(f"forbidden model features: {sorted(forbidden)}")


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
) -> FrozenLogisticModel:
    """Fit the fixed C=1 liblinear model with equal total slate weight."""

    names = tuple(str(feature) for feature in features)
    assert_allowed_model_features(names)
    values = _finite_numeric(frame, names)
    labels = np.asarray(target, dtype=np.int64)
    if labels.shape != (len(frame),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model_id} requires both aligned binary classes")
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (scales >= EPSILON), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260721,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        labels,
        sample_weight=equal_slate_weights(frame[slate_column]),
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
    """Reconstruct probabilities using only serialized preprocessing and coefficients."""

    names = tuple(str(value) for value in model["feature_names"])
    values = _finite_numeric(frame, names)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (
        means.shape != scales.shape
        or means.shape != coefficients.shape
        or len(means) != values.shape[1]
    ):
        raise ValueError("serialized logistic dimensions do not align")
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


@dataclass(frozen=True, slots=True)
class SessionBootstrapDraw:
    """One resample of complete session blocks."""

    draw: int
    sampled_sessions: tuple[str, ...]
    row_indices: NDArray[np.int64]


def session_block_bootstrap_draws(
    sessions: pd.Series | Sequence[object],
    *,
    draws: int,
    seed: int,
) -> tuple[SessionBootstrapDraw, ...]:
    """Sample session identifiers with replacement and retain every row in each block."""

    values = pd.Series(sessions, copy=False).astype(str).reset_index(drop=True)
    unique = np.asarray(sorted(values.unique()), dtype=object)
    if draws < 1 or len(unique) < 2:
        raise ValueError("bootstrap requires positive draws and at least two sessions")
    positions = {
        session: np.flatnonzero(values.to_numpy(dtype=object) == session).astype(np.int64)
        for session in unique
    }
    rng = np.random.default_rng(seed)
    output: list[SessionBootstrapDraw] = []
    for draw in range(draws):
        sampled = tuple(str(value) for value in rng.choice(unique, size=len(unique), replace=True))
        row_indices = np.concatenate([positions[session] for session in sampled]).astype(np.int64)
        output.append(
            SessionBootstrapDraw(
                draw=draw,
                sampled_sessions=sampled,
                row_indices=row_indices,
            )
        )
    return tuple(output)


def permute_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    slate_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute a complete behavioural bundle only among stocks in the same slate."""

    names = tuple(features)
    _finite_numeric(frame, names)
    if slate_column not in frame:
        raise ValueError(f"slate column missing: {slate_column}")
    output = frame.copy()
    for indices in output.groupby(slate_column, sort=True).groups.values():
        index = list(indices)
        bundle = frame.loc[index, list(names)].to_numpy(dtype=np.float64)
        output.loc[index, list(names)] = bundle[rng.permutation(len(index))]
    return output


def assert_safe_timestamps(timestamps: pd.Series | Sequence[object]) -> None:
    """Reject protected 2025-08-23+ rows before any experiment transformation."""

    values = pd.to_datetime(pd.Series(timestamps, copy=False), utc=True, errors="raise")
    if values.ge(PROTECTED_START).any():
        raise ValueError("protected market row materialised")


def decide_behavioural_screen(
    *,
    movement_passes: bool,
    direction_passes: bool,
    conjunction_passes: bool,
    descriptive_differences: bool,
    blocker: str | None = None,
) -> str:
    """Map preregistered evidence to exactly one allowed decision category."""

    if blocker is not None:
        if blocker not in DECISION_CATEGORIES or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown blocker: {blocker}")
        return blocker
    if movement_passes and direction_passes:
        return "behavioural_dimensions_add_movement_and_direction"
    if movement_passes:
        return "behavioural_dimensions_add_movement_only"
    if direction_passes:
        return "behavioural_dimensions_add_direction_only"
    if conjunction_passes:
        return "behavioural_conjunctions_only"
    if descriptive_differences:
        return "behavioural_descriptions_only_no_predictive_increment"
    return "no_behavioural_state_increment"
