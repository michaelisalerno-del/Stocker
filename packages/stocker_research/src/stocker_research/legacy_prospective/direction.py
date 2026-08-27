"""No-fit live runtime for frozen A1, C1, and R1 research classifications."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

DirectionAction = Literal["CALL", "PUT", "ABSTAIN"]
ArchetypeId = Literal["A1", "C1", "R1"]
ARCHETYPE_IDS: tuple[ArchetypeId, ...] = ("A1", "C1", "R1")

_LABELS = {
    "A1": "prospective hypothesis — not validated",
    "C1": "comparison only — not validated",
    "R1": "comparison only — not validated",
}


def _mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frozen artifact must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class _Normalisation:
    median: float
    iqr: float
    clip_lower: float
    clip_upper: float
    missing_value: float
    fallback_level: str


@dataclass(frozen=True)
class _DirectionModel:
    model_id: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    medians: np.ndarray[Any, np.dtype[np.float64]]
    centers: np.ndarray[Any, np.dtype[np.float64]]
    scales: np.ndarray[Any, np.dtype[np.float64]]
    categorical_levels: dict[str, tuple[str, ...]]
    design_feature_names: tuple[str, ...]
    coefficients: np.ndarray[Any, np.dtype[np.float64]]
    intercept: float
    boundary: float


class DirectionClassification(BaseModel):
    """One immutable directional research classification."""

    model_config = ConfigDict(frozen=True)

    model_id: Literal["A1", "C1", "R1"]
    probability_up: float
    confidence: float
    action: DirectionAction
    boundary: float
    label: str
    model_hash: str
    preprocessing_hash: str
    normalised_features: dict[str, float]
    fallback_levels: dict[str, str]


class FrozenDirectionRuntime:
    """Apply frozen stock-local transforms and serialized direction models."""

    def __init__(
        self,
        *,
        models: dict[str, _DirectionModel],
        exact_normalisation: dict[tuple[str, str, int], _Normalisation],
        pooled_normalisation: dict[str, _Normalisation],
        model_hash: str,
        preprocessing_hash: str,
    ) -> None:
        self._models = models
        self._exact_normalisation = exact_normalisation
        self._pooled_normalisation = pooled_normalisation
        self.model_hash = model_hash
        self.preprocessing_hash = preprocessing_hash

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_configurations_path: str | Path,
        normalisation_path: str | Path,
        thresholds_path: str | Path,
    ) -> FrozenDirectionRuntime:
        model_path = Path(model_configurations_path)
        normalisation_file = Path(normalisation_path)
        threshold_file = Path(thresholds_path)
        configurations = _mapping(model_path)
        normalisation = _mapping(normalisation_file)
        thresholds = _mapping(threshold_file)
        if configurations.get("research_only") is not True:
            raise ValueError("direction model research-only flag differs")
        if configurations.get("models_combined") is not False:
            raise ValueError("A1, C1, and R1 must remain separate")
        if thresholds.get("research_only") is not True:
            raise ValueError("direction threshold research-only flag differs")
        full_models = configurations.get("full_models")
        if not isinstance(full_models, dict):
            raise ValueError("frozen full direction models are absent")
        models: dict[str, _DirectionModel] = {}
        for model_id in ARCHETYPE_IDS:
            specification = full_models.get(model_id)
            threshold = thresholds.get(model_id)
            if not isinstance(specification, dict) or not isinstance(threshold, dict):
                raise ValueError(f"frozen {model_id} artifact is absent")
            if specification.get("model_id") != model_id:
                raise ValueError(f"frozen {model_id} identity differs")
            numeric_features = tuple(str(value) for value in specification["numeric_features"])
            categorical_features = tuple(
                str(value) for value in specification["categorical_features"]
            )
            if categorical_features != ("stock", "checkpoint_category", "day_of_week"):
                raise ValueError(f"frozen {model_id} categorical controls differ")
            categorical_levels = {
                str(name): tuple(str(value) for value in values)
                for name, values in specification["categorical_levels"].items()
            }
            expected_design = [
                item for feature in numeric_features for item in (feature, f"{feature}__missing")
            ]
            for category in categorical_features:
                expected_design.extend(
                    f"{category}=={level}" for level in categorical_levels[category]
                )
            design_names = tuple(str(value) for value in specification["design_feature_names"])
            if design_names != tuple(expected_design):
                raise ValueError(f"frozen {model_id} design order differs")
            medians = np.asarray(
                [specification["medians"][name] for name in numeric_features],
                dtype=np.float64,
            )
            centers = np.asarray(
                [specification["robust_centers"][name] for name in numeric_features],
                dtype=np.float64,
            )
            scales = np.asarray(
                [specification["robust_scales"][name] for name in numeric_features],
                dtype=np.float64,
            )
            coefficients = np.asarray(specification["coefficients"], dtype=np.float64)
            if (
                not np.isfinite(medians).all()
                or not np.isfinite(centers).all()
                or not np.isfinite(scales).all()
                or not np.isfinite(coefficients).all()
                or bool((scales <= 0.0).any())
                or len(coefficients) != len(design_names)
            ):
                raise ValueError(f"frozen {model_id} parameters are invalid")
            boundary = float(threshold["boundary"])
            if not 0.0 <= boundary <= 0.5:
                raise ValueError(f"frozen {model_id} boundary is invalid")
            models[model_id] = _DirectionModel(
                model_id=model_id,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                medians=medians,
                centers=centers,
                scales=scales,
                categorical_levels=categorical_levels,
                design_feature_names=design_names,
                coefficients=coefficients,
                intercept=float(specification["intercept"]),
                boundary=boundary,
            )
        parameter_rows = normalisation.get("parameters")
        if (
            normalisation.get("fit_period") != "2024 only"
            or normalisation.get("minimum_support") != 20
            or not isinstance(parameter_rows, list)
        ):
            raise ValueError("frozen direction normalisation contract differs")
        exact: dict[tuple[str, str, int], _Normalisation] = {}
        pooled: dict[str, _Normalisation] = {}
        for item in parameter_rows:
            if not isinstance(item, dict):
                raise ValueError("direction normalisation row is invalid")
            fitted = _Normalisation(
                median=float(item["median"]),
                iqr=float(item["iqr"]),
                clip_lower=float(item["clip_lower"]),
                clip_upper=float(item["clip_upper"]),
                missing_value=float(item["missing_value"]),
                fallback_level=str(item["fallback_level"]),
            )
            if (
                not all(
                    math.isfinite(value)
                    for value in (
                        fitted.median,
                        fitted.iqr,
                        fitted.clip_lower,
                        fitted.clip_upper,
                        fitted.missing_value,
                    )
                )
                or fitted.iqr <= 0.0
                or fitted.clip_lower > fitted.clip_upper
            ):
                raise ValueError("direction normalisation parameters are invalid")
            feature = str(item["feature"])
            stock = str(item["stock"])
            checkpoint = int(item["checkpoint"])
            if stock == "__POOLED__":
                pooled[feature] = fitted
            else:
                exact[(feature, stock, checkpoint)] = fitted
        required_features = {
            feature for model in models.values() for feature in model.numeric_features
        }
        if not required_features.issubset(pooled):
            raise ValueError("direction pooled normalisation fallback is incomplete")
        return cls(
            models=models,
            exact_normalisation=exact,
            pooled_normalisation=pooled,
            model_hash=_sha256(model_path),
            preprocessing_hash=_sha256(normalisation_file),
        )

    def feature_names(self, model_id: str) -> tuple[str, ...]:
        return self._require_model(model_id).numeric_features

    def classify(
        self,
        *,
        raw_features: Mapping[str, object],
        symbol: str,
        checkpoint: int,
        checkpoint_category: str,
        day_of_week: str,
    ) -> dict[str, DirectionClassification]:
        return {
            model_id: self.classify_one(
                model_id=model_id,
                raw_features=raw_features,
                symbol=symbol,
                checkpoint=checkpoint,
                checkpoint_category=checkpoint_category,
                day_of_week=day_of_week,
            )
            for model_id in ARCHETYPE_IDS
        }

    def classify_one(
        self,
        *,
        model_id: ArchetypeId,
        raw_features: Mapping[str, object],
        symbol: str,
        checkpoint: int,
        checkpoint_category: str,
        day_of_week: str,
    ) -> DirectionClassification:
        model = self._require_model(model_id)
        normalised: dict[str, float] = {}
        fallbacks: dict[str, str] = {}
        for name in model.numeric_features:
            fitted = self._exact_normalisation.get(
                (name, symbol, int(checkpoint)),
                self._pooled_normalisation[name],
            )
            raw = _finite(raw_features.get(name))
            value = fitted.missing_value if raw is None else raw
            clipped = min(max(value, fitted.clip_lower), fitted.clip_upper)
            normalised[name] = (clipped - fitted.median) / fitted.iqr
            fallbacks[name] = fitted.fallback_level
        values = np.asarray(
            [normalised[name] for name in model.numeric_features],
            dtype=np.float64,
        )
        missing = ~np.isfinite(values)
        imputed = np.where(missing, model.medians, values)
        standardized = (imputed - model.centers) / model.scales
        pieces: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for index in range(len(model.numeric_features)):
            pieces.append(np.asarray([standardized[index], float(missing[index])]))
        categories = {
            "stock": symbol,
            "checkpoint_category": checkpoint_category,
            "day_of_week": day_of_week,
        }
        for name in model.categorical_features:
            levels = model.categorical_levels[name]
            observed = categories[name]
            if observed not in levels:
                observed = "__UNKNOWN__"
            pieces.append(
                np.asarray([float(observed == level) for level in levels], dtype=np.float64)
            )
        design = np.concatenate(pieces)
        if len(design) != len(model.design_feature_names) or not np.isfinite(design).all():
            raise ValueError(f"{model_id} live design construction failed")
        linear = float(design @ model.coefficients + model.intercept)
        if linear >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-linear))
        else:
            exponential = math.exp(linear)
            probability = exponential / (1.0 + exponential)
        action: DirectionAction = "ABSTAIN"
        if probability >= 0.5 + model.boundary:
            action = "CALL"
        elif probability <= 0.5 - model.boundary:
            action = "PUT"
        return DirectionClassification(
            model_id=model_id,
            probability_up=probability,
            confidence=abs(probability - 0.5),
            action=action,
            boundary=model.boundary,
            label=_LABELS[model_id],
            model_hash=self.model_hash,
            preprocessing_hash=self.preprocessing_hash,
            normalised_features=normalised,
            fallback_levels=fallbacks,
        )

    def _require_model(self, model_id: str) -> _DirectionModel:
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"unknown direction archetype: {model_id}")
        return model


__all__ = [
    "ARCHETYPE_IDS",
    "DirectionClassification",
    "FrozenDirectionRuntime",
]
