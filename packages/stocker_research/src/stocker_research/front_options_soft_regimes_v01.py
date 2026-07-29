"""Development-frozen front-options dimensions and four-state soft regimes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd

from stocker_research.daily_soft_regimes_v0 import (
    FrozenDimensionParameters,
    FrozenSoftRegime,
    RobustValueScale,
    apply_soft_regime,
    fit_soft_regime,
)

FRONT_OPTIONS_RAW_FEATURES: Final[tuple[str, ...]] = (
    "atm_iv",
    "straddle_mid_pct",
    "call_put_iv_gap",
    "skew_25d",
    "combined_relative_spread",
    "iv_minus_realised_20d",
    "near_spot_oi_concentration",
    "call_put_oi_imbalance",
)
FRONT_OPTIONS_MISSING_INDICATORS: Final[tuple[str, ...]] = (
    "skew_25d_missing",
    "near_spot_oi_concentration_missing",
    "call_put_oi_imbalance_missing",
)
FRONT_OPTIONS_DIMENSIONS: Final[tuple[str, ...]] = (
    "front_options_implied_tension",
    "front_options_premium_richness",
    "front_options_downside_asymmetry",
    "front_options_liquidity_stress",
    "front_options_positioning_concentration",
    "front_options_directional_positioning",
    "front_options_surface_disagreement",
)
FRONT_OPTIONS_CANONICAL_DIMENSIONS: Final[tuple[str, ...]] = (
    "front_options_implied_tension",
    "front_options_premium_richness",
    "front_options_downside_asymmetry",
    "front_options_liquidity_stress",
    "front_options_positioning_concentration",
)


def _require_development(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("front-options development frame is empty")
    sessions = pd.to_datetime(frame["session"], errors="raise")
    if not bool(sessions.dt.year.eq(2024).all()):
        raise ValueError("front-options fitting accepts 2024 development rows only")


def _robust_scale(values: pd.Series) -> RobustValueScale:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("front-options robust scaling requires finite support")
    center = float(finite.median())
    scale = float(finite.quantile(0.75) - finite.quantile(0.25))
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return RobustValueScale(center=center, scale=scale)


def _z(values: pd.Series, scale: RobustValueScale) -> pd.Series:
    return (pd.to_numeric(values, errors="coerce") - scale.center) / scale.scale


def fit_front_options_dimension_parameters(
    development: pd.DataFrame,
) -> FrozenDimensionParameters:
    """Fit front-only medians and robust scales on development rows."""

    _require_development(development)
    if missing := sorted(set(FRONT_OPTIONS_RAW_FEATURES).difference(development.columns)):
        raise ValueError(f"front-options raw features missing: {missing}")
    imputed = development.copy()
    medians: dict[str, float] = {}
    for feature in FRONT_OPTIONS_RAW_FEATURES:
        values = pd.to_numeric(imputed[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median())
        if not math.isfinite(median):
            raise ValueError(f"front-options feature lacks development support: {feature}")
        medians[feature] = median
        imputed[feature] = values.fillna(median)
    scales = {feature: _robust_scale(imputed[feature]) for feature in FRONT_OPTIONS_RAW_FEATURES}
    scales["abs_call_put_iv_gap"] = _robust_scale(imputed["call_put_iv_gap"].abs())
    scales["abs_skew_25d"] = _robust_scale(imputed["skew_25d"].abs())
    return FrozenDimensionParameters(
        kind="front_options",
        scales=scales,
        imputation_medians=medians,
    )


def apply_front_options_dimensions(
    frame: pd.DataFrame,
    parameters: FrozenDimensionParameters,
) -> pd.DataFrame:
    """Apply the seven preregistered front-only dimension equations."""

    if parameters.kind != "front_options":
        raise ValueError("front-options dimensions require front-options parameters")
    if missing := sorted(set(FRONT_OPTIONS_RAW_FEATURES).difference(frame.columns)):
        raise ValueError(f"front-options raw features missing: {missing}")
    output = frame.copy()
    for feature, median in parameters.imputation_medians.items():
        values = pd.to_numeric(output[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        output[feature] = values.fillna(median)
    scale = parameters.scales
    z_atm = _z(output["atm_iv"], scale["atm_iv"])
    z_straddle = _z(output["straddle_mid_pct"], scale["straddle_mid_pct"])
    z_gap = _z(output["call_put_iv_gap"], scale["call_put_iv_gap"])
    z_skew = _z(output["skew_25d"], scale["skew_25d"])
    z_spread = _z(output["combined_relative_spread"], scale["combined_relative_spread"])
    z_iv_minus_rv = _z(output["iv_minus_realised_20d"], scale["iv_minus_realised_20d"])
    output["front_options_implied_tension"] = (z_atm + z_straddle + z_iv_minus_rv) / 3.0
    output["front_options_premium_richness"] = (z_straddle + z_iv_minus_rv) / 2.0
    output["front_options_downside_asymmetry"] = (z_skew - z_gap) / 2.0
    output["front_options_liquidity_stress"] = z_spread
    output["front_options_positioning_concentration"] = _z(
        output["near_spot_oi_concentration"],
        scale["near_spot_oi_concentration"],
    )
    output["front_options_directional_positioning"] = _z(
        output["call_put_oi_imbalance"],
        scale["call_put_oi_imbalance"],
    )
    output["front_options_surface_disagreement"] = (
        _z(output["call_put_iv_gap"].abs(), scale["abs_call_put_iv_gap"])
        + _z(output["skew_25d"].abs(), scale["abs_skew_25d"])
        + z_spread
    ) / 3.0
    return output


def fit_front_options_regime(development: pd.DataFrame) -> FrozenSoftRegime:
    """Fit the single frozen four-state diagonal front-options GMM."""

    _require_development(development)
    return fit_soft_regime(
        development,
        dimensions=FRONT_OPTIONS_DIMENSIONS,
        missing_indicators=FRONT_OPTIONS_MISSING_INDICATORS,
        canonical_dimensions=FRONT_OPTIONS_CANONICAL_DIMENSIONS,
        prefix="front_options_regime",
        random_state=20260723,
    )


def apply_front_options_regime(
    frame: pd.DataFrame,
    fitted: FrozenSoftRegime,
) -> pd.DataFrame:
    """Assign frozen canonical front-options posterior probabilities."""

    return apply_soft_regime(frame, fitted)


def front_options_regime_mapping(
    fitted: FrozenSoftRegime,
    *,
    safety_flags: Mapping[str, object],
) -> dict[str, object]:
    """Serialize every value required for independent posterior reconstruction."""

    canonical = list(fitted.canonical_to_original)
    return {
        **safety_flags,
        "prefix": fitted.prefix,
        "fitted_period": fitted.fitted_period,
        "n_components": 4,
        "covariance_type": "diag",
        "reg_covar": 1e-5,
        "n_init": 5,
        "max_iter": 300,
        "random_state": 20260723,
        "input_columns": list(fitted.input_columns),
        "canonical_dimensions": list(fitted.canonical_dimensions),
        "canonical_to_original": canonical,
        "original_to_canonical": list(fitted.original_to_canonical),
        "canonical_centroids": list(fitted.canonical_centroids),
        "canonical_input_means": fitted.estimator.means_[canonical].astype(float).tolist(),
        "input_medians": fitted.input_medians.astype(float).tolist(),
        "canonical_weights": fitted.estimator.weights_[canonical].astype(float).tolist(),
        "canonical_covariances": fitted.estimator.covariances_[canonical].astype(float).tolist(),
        "iterations": int(fitted.estimator.n_iter_),
        "converged": bool(fitted.estimator.converged_),
    }


def apply_serialized_diag_regime(
    frame: pd.DataFrame,
    mapping: Mapping[str, Any],
    *,
    prefix: str,
) -> pd.DataFrame:
    """Apply a serialized canonical diagonal GMM without refitting it."""

    input_columns = tuple(str(value) for value in mapping["input_columns"])
    if missing := sorted(set(input_columns).difference(frame.columns)):
        raise ValueError(f"serialized regime inputs missing: {missing}")
    medians = np.asarray(mapping["input_medians"], dtype=float)
    means = np.asarray(mapping["canonical_input_means"], dtype=float)
    covariances = np.asarray(mapping["canonical_covariances"], dtype=float)
    weights = np.asarray(mapping["canonical_weights"], dtype=float)
    if means.shape != covariances.shape or means.shape[0] != 4:
        raise ValueError("serialized regime must contain four aligned diagonal components")
    raw = frame.loc[:, list(input_columns)].to_numpy(float)
    values = np.where(np.isfinite(raw), raw, medians)
    log_density = np.empty((len(frame), 4), dtype=float)
    width = means.shape[1]
    for regime in range(4):
        variance = np.maximum(covariances[regime], 1e-12)
        delta = values - means[regime]
        log_density[:, regime] = math.log(float(weights[regime])) - 0.5 * (
            width * math.log(2.0 * math.pi)
            + float(np.log(variance).sum())
            + (np.square(delta) / variance).sum(axis=1)
        )
    log_density -= log_density.max(axis=1, keepdims=True)
    probabilities = np.exp(log_density)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    output = frame.copy()
    for regime in range(4):
        output[f"{prefix}_p_{regime}"] = probabilities[:, regime]
    clipped = np.clip(probabilities, 1e-15, 1.0)
    output[f"{prefix}_entropy"] = -np.sum(probabilities * np.log(clipped), axis=1)
    ordered = np.sort(probabilities, axis=1)
    output[f"{prefix}_top_probability"] = ordered[:, -1]
    output[f"{prefix}_margin"] = ordered[:, -1] - ordered[:, -2]
    output[prefix] = np.argmax(probabilities, axis=1).astype(int)
    squared = np.column_stack(
        [
            (np.square(values - means[regime]) / np.maximum(covariances[regime], 1e-12)).sum(axis=1)
            for regime in range(4)
        ]
    )
    output[f"{prefix}_mahalanobis_to_nearest_centroid"] = np.sqrt(np.min(squared, axis=1))
    return output


__all__ = [
    "FRONT_OPTIONS_CANONICAL_DIMENSIONS",
    "FRONT_OPTIONS_DIMENSIONS",
    "FRONT_OPTIONS_MISSING_INDICATORS",
    "FRONT_OPTIONS_RAW_FEATURES",
    "apply_front_options_dimensions",
    "apply_front_options_regime",
    "apply_serialized_diag_regime",
    "fit_front_options_dimension_parameters",
    "fit_front_options_regime",
    "front_options_regime_mapping",
]
