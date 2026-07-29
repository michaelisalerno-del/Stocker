"""Development-frozen robust dimensions and four-state soft daily regimes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.mixture import GaussianMixture

from stocker_research.daily_stock_options_context_v0 import (
    DAILY_OPTIONS_MISSING_INDICATORS,
    DAILY_OPTIONS_RAW_FEATURES,
    DAILY_STOCK_RAW_FEATURES,
)

FloatArray = npt.NDArray[np.float64]

DAILY_STOCK_DIMENSIONS: Final[tuple[str, ...]] = (
    "daily_compression",
    "daily_directional_efficiency",
    "daily_trend_persistence",
    "daily_extension",
    "daily_rejection",
    "daily_volatility_acceleration",
    "daily_relative_strength",
    "daily_activity_acceleration",
)
DAILY_OPTIONS_DIMENSIONS: Final[tuple[str, ...]] = (
    "options_implied_tension",
    "options_premium_richness",
    "options_downside_asymmetry",
    "options_front_urgency",
    "options_liquidity_stress",
    "options_positioning_concentration",
    "options_directional_positioning",
    "options_surface_disagreement",
)


@dataclass(frozen=True)
class RobustValueScale:
    """One development-only robust centre and scale."""

    center: float
    scale: float


@dataclass(frozen=True)
class FrozenDimensionParameters:
    """Parameters needed to reproduce a frozen dimension surface."""

    kind: str
    scales: Mapping[str, RobustValueScale]
    imputation_medians: Mapping[str, float]
    fitted_period: str = "development_2024_only"


@dataclass(frozen=True)
class FrozenSoftRegime:
    """One fitted four-component diagonal Gaussian mixture and stable ID map."""

    prefix: str
    input_columns: tuple[str, ...]
    dimensions: tuple[str, ...]
    missing_indicators: tuple[str, ...]
    input_medians: FloatArray
    estimator: GaussianMixture
    canonical_to_original: tuple[int, int, int, int]
    original_to_canonical: tuple[int, int, int, int]
    canonical_centroids: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]
    canonical_dimensions: tuple[str, ...]
    fitted_period: str = "development_2024_only"


def _require_development(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("development frame is empty")
    sessions = pd.to_datetime(frame["session"], errors="raise")
    if not bool(sessions.dt.year.eq(2024).all()):
        raise ValueError("daily scaling and regimes must be fitted on 2024 only")


def _robust_value_scale(values: pd.Series) -> RobustValueScale:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("robust scaling requires finite development support")
    center = float(finite.median())
    scale = float(finite.quantile(0.75) - finite.quantile(0.25))
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return RobustValueScale(center=center, scale=scale)


def _z(values: pd.Series, scale: RobustValueScale) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric - scale.center) / scale.scale


def fit_stock_dimension_parameters(development: pd.DataFrame) -> FrozenDimensionParameters:
    """Fit all stock dimension robust centres and scales on 2024."""

    _require_development(development)
    if missing := sorted(set(DAILY_STOCK_RAW_FEATURES).difference(development.columns)):
        raise ValueError(f"daily stock raw features missing: {missing}")
    scales = {
        feature: _robust_value_scale(development[feature]) for feature in DAILY_STOCK_RAW_FEATURES
    }
    absolute_extension = pd.to_numeric(development["daily_extension_20"], errors="coerce").abs()
    scales["abs_daily_extension_20"] = _robust_value_scale(absolute_extension)
    directional = 0.5 * (
        _z(development["daily_efficiency_5"], scales["daily_efficiency_5"])
        + _z(development["daily_efficiency_10"], scales["daily_efficiency_10"])
    )
    scales["daily_directional_efficiency"] = _robust_value_scale(directional)
    return FrozenDimensionParameters(
        kind="daily_stock",
        scales=scales,
        imputation_medians={},
    )


def apply_stock_dimensions(
    frame: pd.DataFrame, parameters: FrozenDimensionParameters
) -> pd.DataFrame:
    """Apply the eight frozen daily stock dimension equations."""

    if parameters.kind != "daily_stock":
        raise ValueError("stock dimensions require stock parameters")
    if missing := sorted(set(DAILY_STOCK_RAW_FEATURES).difference(frame.columns)):
        raise ValueError(f"daily stock raw features missing: {missing}")
    scale = parameters.scales
    output = frame.copy()
    z_range = _z(output["daily_range_5_to_20"], scale["daily_range_5_to_20"])
    z_rv = _z(output["daily_rv_5_to_20"], scale["daily_rv_5_to_20"])
    z_overlap = _z(output["daily_range_overlap_5"], scale["daily_range_overlap_5"])
    z_efficiency_5 = _z(output["daily_efficiency_5"], scale["daily_efficiency_5"])
    z_efficiency_10 = _z(output["daily_efficiency_10"], scale["daily_efficiency_10"])
    directional = 0.5 * (z_efficiency_5 + z_efficiency_10)
    z_directional = _z(directional, scale["daily_directional_efficiency"])
    z_sign = _z(output["daily_sign_persistence_5"], scale["daily_sign_persistence_5"])
    z_abs_extension = _z(
        pd.to_numeric(output["daily_extension_20"], errors="coerce").abs(),
        scale["abs_daily_extension_20"],
    )
    z_extension = _z(output["daily_extension_20"], scale["daily_extension_20"])
    z_wick = _z(output["daily_extreme_wick_3"], scale["daily_extreme_wick_3"])
    output["daily_compression"] = (-z_range - z_rv + z_overlap) / 3.0
    output["daily_directional_efficiency"] = directional
    output["daily_trend_persistence"] = 0.5 * (z_sign + z_abs_extension)
    output["daily_extension"] = z_extension
    output["daily_rejection"] = 0.5 * (z_wick - z_directional)
    output["daily_volatility_acceleration"] = z_rv
    output["daily_relative_strength"] = _z(
        output["daily_relative_return_5"], scale["daily_relative_return_5"]
    )
    output["daily_activity_acceleration"] = _z(
        output["daily_activity_5_to_20"], scale["daily_activity_5_to_20"]
    )
    return output


def fit_options_dimension_parameters(development: pd.DataFrame) -> FrozenDimensionParameters:
    """Fit development medians and robust scales for the options dimensions."""

    _require_development(development)
    if missing := sorted(set(DAILY_OPTIONS_RAW_FEATURES).difference(development.columns)):
        raise ValueError(f"daily options raw features missing: {missing}")
    medians: dict[str, float] = {}
    imputed = development.copy()
    for feature in DAILY_OPTIONS_RAW_FEATURES:
        finite = pd.to_numeric(imputed[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(finite.median())
        if not math.isfinite(median):
            raise ValueError(f"daily options feature lacks development support: {feature}")
        medians[feature] = median
        imputed[feature] = finite.fillna(median)
    scales = {
        feature: _robust_value_scale(imputed[feature]) for feature in DAILY_OPTIONS_RAW_FEATURES
    }
    scales["abs_call_put_iv_gap"] = _robust_value_scale(imputed["call_put_iv_gap"].abs())
    scales["abs_skew_25d"] = _robust_value_scale(imputed["skew_25d"].abs())
    return FrozenDimensionParameters(
        kind="daily_options",
        scales=scales,
        imputation_medians=medians,
    )


def apply_options_dimensions(
    frame: pd.DataFrame, parameters: FrozenDimensionParameters
) -> pd.DataFrame:
    """Impute causally and apply the eight frozen daily options dimensions."""

    if parameters.kind != "daily_options":
        raise ValueError("options dimensions require options parameters")
    if missing := sorted(set(DAILY_OPTIONS_RAW_FEATURES).difference(frame.columns)):
        raise ValueError(f"daily options raw features missing: {missing}")
    output = frame.copy()
    for feature, median in parameters.imputation_medians.items():
        values = pd.to_numeric(output[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        output[feature] = values.fillna(median)
    scale = parameters.scales
    z_atm = _z(output["atm_iv"], scale["atm_iv"])
    z_straddle = _z(output["straddle_mid_pct"], scale["straddle_mid_pct"])
    z_gap = _z(output["call_put_iv_gap"], scale["call_put_iv_gap"])
    z_skew = _z(output["skew_25d"], scale["skew_25d"])
    z_urgency = _z(output["front_term_urgency"], scale["front_term_urgency"])
    z_spread = _z(output["combined_relative_spread"], scale["combined_relative_spread"])
    z_iv_minus_rv = _z(output["iv_minus_realised_20d"], scale["iv_minus_realised_20d"])
    z_concentration = _z(output["near_spot_oi_concentration"], scale["near_spot_oi_concentration"])
    z_positioning = _z(output["call_put_oi_imbalance"], scale["call_put_oi_imbalance"])
    output["options_implied_tension"] = (z_atm + z_straddle + z_iv_minus_rv) / 3.0
    output["options_premium_richness"] = 0.5 * (z_straddle + z_iv_minus_rv)
    output["options_downside_asymmetry"] = 0.5 * (z_skew - z_gap)
    output["options_front_urgency"] = z_urgency
    output["options_liquidity_stress"] = z_spread
    output["options_positioning_concentration"] = z_concentration
    output["options_directional_positioning"] = z_positioning
    output["options_surface_disagreement"] = (
        _z(output["call_put_iv_gap"].abs(), scale["abs_call_put_iv_gap"])
        + _z(output["skew_25d"].abs(), scale["abs_skew_25d"])
        + z_spread
    ) / 3.0
    return output


def fit_soft_regime(
    development: pd.DataFrame,
    *,
    dimensions: Sequence[str],
    missing_indicators: Sequence[str],
    canonical_dimensions: Sequence[str],
    prefix: str,
    random_state: int = 20260723,
) -> FrozenSoftRegime:
    """Fit exactly one four-component diagonal GMM and freeze stable IDs."""

    _require_development(development)
    if development.duplicated(["symbol", "session"]).any():
        raise ValueError("regime fitting requires one row per stock-session")
    dimension_names = tuple(dimensions)
    indicator_names = tuple(missing_indicators)
    input_columns = (*dimension_names, *indicator_names)
    if missing := sorted(set(input_columns).difference(development.columns)):
        raise ValueError(f"regime inputs missing: {missing}")
    canonical_names = tuple(canonical_dimensions)
    if missing := sorted(set(canonical_names).difference(dimension_names)):
        raise ValueError(f"canonical dimensions absent from regime dimensions: {missing}")
    raw = development.loc[:, list(input_columns)].to_numpy(float)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    medians = np.nanmedian(finite, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("every regime input requires finite development support")
    values = np.where(np.isfinite(raw), raw, medians)
    estimator = GaussianMixture(
        n_components=4,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=5,
        max_iter=300,
        random_state=random_state,
    )
    estimator.fit(values)
    if not estimator.converged_ or int(estimator.n_iter_) >= 300:
        raise RuntimeError("blocked_daily_regime_failure")
    dimension_index = {name: index for index, name in enumerate(input_columns)}
    canonical_to_original = tuple(
        sorted(
            range(4),
            key=lambda component: tuple(
                float(estimator.means_[component, dimension_index[name]])
                for name in canonical_names
            ),
        )
    )
    if len(canonical_to_original) != 4:
        raise AssertionError("four canonical regimes required")
    original_to_canonical_values = [0, 0, 0, 0]
    for canonical, original in enumerate(canonical_to_original):
        original_to_canonical_values[original] = canonical
    original_to_canonical = tuple(original_to_canonical_values)
    centroids: list[dict[str, float]] = []
    for original in canonical_to_original:
        centroids.append(
            {
                name: float(estimator.means_[original, dimension_index[name]])
                for name in dimension_names
            }
        )
    return FrozenSoftRegime(
        prefix=prefix,
        input_columns=input_columns,
        dimensions=dimension_names,
        missing_indicators=indicator_names,
        input_medians=np.asarray(medians, dtype=np.float64),
        estimator=estimator,
        canonical_to_original=(
            int(canonical_to_original[0]),
            int(canonical_to_original[1]),
            int(canonical_to_original[2]),
            int(canonical_to_original[3]),
        ),
        original_to_canonical=(
            int(original_to_canonical[0]),
            int(original_to_canonical[1]),
            int(original_to_canonical[2]),
            int(original_to_canonical[3]),
        ),
        canonical_centroids=(centroids[0], centroids[1], centroids[2], centroids[3]),
        canonical_dimensions=canonical_names,
    )


def apply_soft_regime(frame: pd.DataFrame, fitted: FrozenSoftRegime) -> pd.DataFrame:
    """Assign canonical posteriors, uncertainty summaries, and nearest distance."""

    if missing := sorted(set(fitted.input_columns).difference(frame.columns)):
        raise ValueError(f"regime inputs missing: {missing}")
    raw = frame.loc[:, list(fitted.input_columns)].to_numpy(float)
    values = np.where(np.isfinite(raw), raw, fitted.input_medians)
    original_probabilities = fitted.estimator.predict_proba(values)
    probabilities = original_probabilities[:, list(fitted.canonical_to_original)]
    output = frame.copy()
    for regime in range(4):
        output[f"{fitted.prefix}_p_{regime}"] = probabilities[:, regime]
    clipped = np.clip(probabilities, 1e-15, 1.0)
    output[f"{fitted.prefix}_entropy"] = -np.sum(probabilities * np.log(clipped), axis=1)
    ordered = np.sort(probabilities, axis=1)
    output[f"{fitted.prefix}_top_probability"] = ordered[:, -1]
    output[f"{fitted.prefix}_margin"] = ordered[:, -1] - ordered[:, -2]
    output[fitted.prefix] = np.argmax(probabilities, axis=1).astype(int)
    squared_distances: list[FloatArray] = []
    for original in fitted.canonical_to_original:
        variance = np.maximum(fitted.estimator.covariances_[original], 1e-12)
        squared = np.sum(np.square(values - fitted.estimator.means_[original]) / variance, axis=1)
        squared_distances.append(np.asarray(squared, dtype=np.float64))
    output[f"{fitted.prefix}_mahalanobis_to_nearest_centroid"] = np.sqrt(
        np.min(np.column_stack(squared_distances), axis=1)
    )
    return output


__all__ = [
    "DAILY_OPTIONS_DIMENSIONS",
    "DAILY_OPTIONS_MISSING_INDICATORS",
    "DAILY_OPTIONS_RAW_FEATURES",
    "DAILY_STOCK_DIMENSIONS",
    "FrozenDimensionParameters",
    "FrozenSoftRegime",
    "RobustValueScale",
    "apply_options_dimensions",
    "apply_soft_regime",
    "apply_stock_dimensions",
    "fit_options_dimension_parameters",
    "fit_soft_regime",
    "fit_stock_dimension_parameters",
]
