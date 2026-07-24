"""Verified artifact loader for the frozen M0/M1 runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stocker_prospective.bundle import (
    BundleError,
    BundleVerification,
    validate_feature_vector,
)


class FrozenScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    m0_probability: float = Field(ge=0.0, le=1.0)
    m1_probability: float = Field(ge=0.0, le=1.0)
    frozen_threshold: float = Field(ge=0.0, le=1.0)
    bundle_id: str
    manifest_sha256: str
    feature_schema_hash: str
    score_label: str = "frozen_m1_verified_bundle"


class VerifiedFrozenScorer:
    """Load only hash-verified, operator-activated serialized artifacts."""

    def __init__(
        self,
        *,
        verification: BundleVerification,
        root: Path,
        m0_model: Any,
        m1_model: Any,
        preprocessor: Any,
    ) -> None:
        self.verification = verification
        self.root = root
        self._m0 = m0_model
        self._m1 = m1_model
        self._preprocessor = preprocessor

    @classmethod
    def load(
        cls,
        verification: BundleVerification,
        *,
        installed_bundle_path: str | Path,
    ) -> VerifiedFrozenScorer:
        if not verification.verified:
            raise BundleError("blocked_frozen_artifact_hash_mismatch")
        root = Path(installed_bundle_path)
        manifest = verification.manifest
        try:
            m0 = joblib.load(root / manifest.m0_artifact.path)
            m1 = joblib.load(root / manifest.m1_artifact.path)
            preprocessor = joblib.load(root / manifest.preprocessor.path)
        except Exception as exc:
            raise BundleError(
                "blocked_missing_verified_frozen_bundle: serialized artifact cannot be loaded"
            ) from exc
        if not callable(getattr(preprocessor, "transform", None)):
            raise BundleError(
                "blocked_missing_verified_frozen_bundle: preprocessor lacks transform"
            )
        for name, model in (("M0", m0), ("M1", m1)):
            if not callable(getattr(model, "predict_proba", None)):
                raise BundleError(
                    f"blocked_missing_verified_frozen_bundle: {name} lacks predict_proba"
                )
        return cls(
            verification=verification,
            root=root,
            m0_model=m0,
            m1_model=m1,
            preprocessor=preprocessor,
        )

    def score(self, values: list[tuple[str, object]]) -> FrozenScore:
        manifest = self.verification.manifest
        validate_feature_vector(manifest, values)
        frame = pd.DataFrame(
            [[value for _name, value in values]],
            columns=[name for name, _value in values],
        )
        try:
            transformed = self._preprocessor.transform(frame)
            m0 = _positive_probability(self._m0.predict_proba(transformed), "M0")
            m1 = _positive_probability(self._m1.predict_proba(transformed), "M1")
        except BundleError:
            raise
        except Exception as exc:
            raise BundleError(
                "blocked_feature_schema_mismatch: frozen scoring invocation failed"
            ) from exc
        return FrozenScore(
            m0_probability=m0,
            m1_probability=m1,
            frozen_threshold=manifest.threshold.value,
            bundle_id=manifest.bundle_id,
            manifest_sha256=self.verification.manifest_sha256,
            feature_schema_hash=manifest.feature_schema.sha256,
        )


def _positive_probability(matrix: Any, name: str) -> float:
    try:
        value = float(matrix[0][1])
    except Exception as exc:
        raise BundleError(
            f"blocked_feature_schema_mismatch: {name} predict_proba shape is invalid"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise BundleError(f"blocked_feature_schema_mismatch: {name} probability is invalid")
    return value
