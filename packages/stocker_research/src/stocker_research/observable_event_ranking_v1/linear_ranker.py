"""Deterministic pooled regularized linear ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import PRIMARY_FEATURES


@dataclass(frozen=True)
class LinearPreprocessor:
    """Training-only imputation, clipping, and standardisation parameters."""

    medians: tuple[float, ...]
    lower_clip: tuple[float, ...]
    upper_clip: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Apply frozen training parameters in primary feature order."""

        values = frame.loc[:, PRIMARY_FEATURES].to_numpy(dtype="float64")
        medians = np.asarray(self.medians)
        values = np.where(np.isfinite(values), values, medians)
        values = np.clip(values, np.asarray(self.lower_clip), np.asarray(self.upper_clip))
        transformed = (values - np.asarray(self.means)) / np.asarray(self.scales)
        return cast(npt.NDArray[np.float64], transformed)


@dataclass(frozen=True)
class LinearRankerModel:
    """Serialized coefficient representation of M1."""

    alpha: float
    feature_names: tuple[str, ...]
    preprocessor: LinearPreprocessor
    intercept: float
    coefficients: tuple[float, ...]

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Score rows without any stock or sector identifier input."""

        transformed = self.preprocessor.transform(frame)
        scores = self.intercept + transformed @ np.asarray(self.coefficients)
        return cast(npt.NDArray[np.float64], scores)


def equal_slate_sample_weights(slate_ids: pd.Series) -> npt.NDArray[np.float64]:
    """Give every slate total weight one and every member weight 1/slate_size."""

    sizes = slate_ids.groupby(slate_ids, sort=True).transform("size").to_numpy(dtype="float64")
    return 1.0 / sizes


def _fit_preprocessor(frame: pd.DataFrame) -> LinearPreprocessor:
    values = frame.loc[:, PRIMARY_FEATURES].to_numpy(dtype="float64")
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(values), values, medians)
    lower = np.quantile(imputed, 0.005, axis=0, method="linear")
    upper = np.quantile(imputed, 0.995, axis=0, method="linear")
    clipped = np.clip(imputed, lower, upper)
    means = clipped.mean(axis=0)
    scales = clipped.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    return LinearPreprocessor(
        medians=tuple(float(value) for value in medians),
        lower_clip=tuple(float(value) for value in lower),
        upper_clip=tuple(float(value) for value in upper),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
    )


def fit_linear_ranker(
    features: pd.DataFrame,
    targets: np.ndarray,
    slate_ids: pd.Series,
    *,
    alpha: float = 1.0,
) -> LinearRankerModel:
    """Fit M1 with fixed alpha and no search or identifier features."""

    if alpha != 1.0:
        raise ValueError("V1 alpha is frozen at 1.0")
    if len(features) != len(targets) or len(features) != len(slate_ids):
        raise ValueError("features, targets, and slate ids must align")
    preprocessor = _fit_preprocessor(features)
    design = preprocessor.transform(features)
    augmented = np.column_stack([np.ones(len(design)), design])
    weights = equal_slate_sample_weights(slate_ids)
    weighted_design = augmented * np.sqrt(weights)[:, None]
    weighted_target = np.asarray(targets, dtype="float64") * np.sqrt(weights)
    penalty = np.eye(augmented.shape[1], dtype="float64") * alpha
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return LinearRankerModel(
        alpha=alpha,
        feature_names=PRIMARY_FEATURES,
        preprocessor=preprocessor,
        intercept=float(parameters[0]),
        coefficients=tuple(float(value) for value in parameters[1:]),
    )
