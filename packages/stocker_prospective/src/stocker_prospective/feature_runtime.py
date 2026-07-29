"""No-fit loader for the immutable frozen H0 and front-options feature runtime."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_prospective.bundle import BundleError, BundleVerification, verify_bundle
from stocker_research.daily_soft_regimes_v0 import (
    FrozenDimensionParameters,
    RobustValueScale,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
    FRONT_OPTIONS_RAW_FEATURES,
    apply_front_options_dimensions,
    apply_serialized_diag_regime,
)
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.regime_validity_v2 import (
    EmissionPreprocessing,
    SemiMarkovParameters,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"blocked_feature_schema_mismatch: invalid {role}") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"blocked_feature_schema_mismatch: invalid {role}")
    return payload


def _verify_implementation_references(registry: dict[str, Any]) -> None:
    references = registry.get("implementation_references")
    if not isinstance(references, dict) or not references:
        raise BundleError(
            "blocked_feature_schema_mismatch: implementation references are absent"
        )
    loaded_module = sys.modules.get(LoopDictionary.__module__)
    package_file = None if loaded_module is None else getattr(loaded_module, "__file__", None)
    if package_file is None:
        raise BundleError(
            "blocked_feature_schema_mismatch: stocker-research package path is absent"
        )
    package_root = Path(package_file).parent
    for relative_path, expected_hash in references.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise BundleError(
                "blocked_feature_schema_mismatch: implementation reference is invalid"
            )
        module_name = Path(relative_path).stem
        if not relative_path.startswith(
            "packages/stocker_research/src/stocker_research/"
        ):
            raise BundleError(
                "blocked_feature_schema_mismatch: implementation reference escaped contract"
            )
        source_path = package_root / f"{module_name}.py"
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise BundleError(
                "blocked_frozen_artifact_hash_mismatch: "
                f"feature implementation differs for {module_name}"
            )


def _load_h0(
    *,
    parameters_path: Path,
    preprocessing_path: Path,
    expected_model_hash: str,
    expected_features: tuple[str, ...],
) -> tuple[EmissionPreprocessing, SemiMarkovParameters, str]:
    frame = pd.read_csv(preprocessing_path)
    required = {"feature", "imputer_median", "scaler_center", "scaler_scale"}
    if not required.issubset(frame.columns):
        raise BundleError("blocked_feature_schema_mismatch: H0 preprocessing columns differ")
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(frame["feature"].astype(str)),
        medians=frame["imputer_median"].to_numpy(dtype=float),
        centers=frame["scaler_center"].to_numpy(dtype=float),
        scales=frame["scaler_scale"].to_numpy(dtype=float),
    )
    try:
        preprocessing.validate()
        with np.load(parameters_path) as stored:
            parameters = SemiMarkovParameters(
                means=np.asarray(stored["means"]).copy(),
                variances=np.asarray(stored["variances"]).copy(),
                duration_hazard=np.asarray(stored["duration_hazard"]).copy(),
                transitions=np.asarray(stored["transitions"]).copy(),
                initial=np.asarray(stored["initial"]).copy(),
                occupancy=np.asarray(stored["occupancy"]).copy(),
            )
            model_hash = str(np.asarray(stored["state_model_hash"]).item())
        parameters.validate()
    except Exception as exc:
        raise BundleError("blocked_feature_schema_mismatch: H0 artifact is invalid") from exc
    if preprocessing.feature_names != expected_features or model_hash != expected_model_hash:
        raise BundleError("blocked_feature_schema_mismatch: H0 artifact identity differs")
    return preprocessing, parameters, model_hash


def _load_loop_dictionary(
    path: Path,
    *,
    expected_version: str,
    expected_hash: str,
    expected_count: int,
) -> LoopDictionary:
    try:
        table = pd.read_csv(path)
        definitions = {}
        for row in table.itertuples(index=False):
            definition = decompose_closed_path(json.loads(str(row.canonical_orientation)))
            if definition.semantic_loop_id != str(row.semantic_loop_id):
                raise ValueError("semantic loop identity differs")
            definitions[definition.semantic_loop_id] = definition
        dictionary = LoopDictionary(
            definitions,
            (),
            version=str(table["dictionary_version"].iloc[0]),
        )
    except Exception as exc:
        raise BundleError("blocked_feature_schema_mismatch: loop dictionary is invalid") from exc
    if (
        dictionary.version != expected_version
        or dictionary.dictionary_hash != expected_hash
        or len(definitions) != expected_count
    ):
        raise BundleError("blocked_feature_schema_mismatch: loop dictionary identity differs")
    return dictionary


def _front_options_parameters(
    manifest: dict[str, Any],
) -> FrozenDimensionParameters:
    if (
        manifest.get("fitted_period") != "development_2024_only"
        or manifest.get("previous_close_options_only") is not True
        or manifest.get("development_only_scaling_and_imputation") is not True
        or tuple(manifest.get("raw_features", ())) != FRONT_OPTIONS_RAW_FEATURES
        or tuple(manifest.get("dimensions", ())) != FRONT_OPTIONS_DIMENSIONS
        or tuple(manifest.get("missing_indicators", ()))
        != FRONT_OPTIONS_MISSING_INDICATORS
    ):
        raise BundleError(
            "blocked_feature_schema_mismatch: front-options feature contract differs"
        )
    scales_payload = manifest.get("scales")
    medians_payload = manifest.get("imputation_medians")
    if not isinstance(scales_payload, dict) or not isinstance(medians_payload, dict):
        raise BundleError(
            "blocked_feature_schema_mismatch: front-options frozen parameters are absent"
        )
    try:
        scales = {
            str(name): RobustValueScale(
                center=float(value["center"]),
                scale=float(value["scale"]),
            )
            for name, value in scales_payload.items()
            if isinstance(value, dict)
        }
        medians = {str(name): float(value) for name, value in medians_payload.items()}
        parameters = FrozenDimensionParameters(
            kind="front_options",
            scales=scales,
            imputation_medians=medians,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(
            "blocked_feature_schema_mismatch: front-options frozen parameters are invalid"
        ) from exc
    if set(medians) != set(FRONT_OPTIONS_RAW_FEATURES):
        raise BundleError(
            "blocked_feature_schema_mismatch: front-options imputation order differs"
        )
    return parameters


@dataclass(frozen=True)
class FrozenFeatureRuntime:
    """Verified immutable transform state; it exposes no fitting or order path."""

    h0_preprocessing: EmissionPreprocessing
    h0_parameters: SemiMarkovParameters
    h0_model_hash: str
    loop_dictionary: LoopDictionary
    front_options_parameters: FrozenDimensionParameters
    front_options_regime_mapping: dict[str, Any]

    @classmethod
    def load(
        cls,
        verification: BundleVerification,
        *,
        bundle_root: str | Path,
    ) -> FrozenFeatureRuntime:
        """Load only a hash-verified v2 bundle and recheck embedded identities."""

        if not verification.verified:
            raise BundleError("blocked_frozen_artifact_hash_mismatch")
        root = Path(bundle_root)
        actual_verification = verify_bundle(root)
        if (
            not actual_verification.verified
            or actual_verification.manifest_sha256 != verification.manifest_sha256
            or actual_verification.manifest != verification.manifest
        ):
            raise BundleError("blocked_frozen_artifact_hash_mismatch")
        identity = actual_verification.manifest.feature_runtime
        if identity is None:
            raise BundleError("blocked_missing_verified_frozen_feature_runtime")
        registry = _load_json(root / identity.registry.path, role="feature-runtime registry")
        _verify_implementation_references(registry)
        preprocessing, parameters, model_hash = _load_h0(
            parameters_path=root / identity.h0_parameters.path,
            preprocessing_path=root / identity.h0_preprocessing.path,
            expected_model_hash=identity.h0_model_hash,
            expected_features=identity.h0_emission_features,
        )
        dictionary = _load_loop_dictionary(
            root / identity.loop_dictionary.path,
            expected_version=identity.loop_dictionary_version,
            expected_hash=identity.loop_dictionary_hash,
            expected_count=identity.loop_definition_count,
        )
        feature_manifest = _load_json(
            root / identity.front_options_feature_manifest.path,
            role="front-options feature manifest",
        )
        regime_mapping = _load_json(
            root / identity.front_options_regime_mapping.path,
            role="front-options regime mapping",
        )
        front_parameters = _front_options_parameters(feature_manifest)
        if (
            regime_mapping.get("fitted_period") != "development_2024_only"
            or regime_mapping.get("previous_close_options_only") is not True
            or tuple(regime_mapping.get("input_columns", ()))
            != (*FRONT_OPTIONS_DIMENSIONS, *FRONT_OPTIONS_MISSING_INDICATORS)
        ):
            raise BundleError(
                "blocked_feature_schema_mismatch: front-options regime mapping differs"
            )
        return cls(
            h0_preprocessing=preprocessing,
            h0_parameters=parameters,
            h0_model_hash=model_hash,
            loop_dictionary=dictionary,
            front_options_parameters=front_parameters,
            front_options_regime_mapping=regime_mapping,
        )

    def transform_previous_session_options(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen front-options transforms without fitting or outcome access."""

        missing = sorted(set(FRONT_OPTIONS_RAW_FEATURES).difference(frame.columns))
        if missing:
            raise BundleError(
                "blocked_feature_schema_mismatch: "
                f"previous-session context is missing {','.join(missing)}"
            )
        prepared = frame.copy()
        for name in FRONT_OPTIONS_MISSING_INDICATORS:
            raw_name = {
                "skew_25d_missing": "skew_25d",
                "near_spot_oi_concentration_missing": "near_spot_oi_concentration",
                "call_put_oi_imbalance_missing": "call_put_oi_imbalance",
            }[name]
            prepared[name] = (
                pd.to_numeric(prepared[raw_name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .isna()
                .astype(int)
            )
        dimensions = apply_front_options_dimensions(
            prepared,
            self.front_options_parameters,
        )
        output: pd.DataFrame = apply_serialized_diag_regime(
            dimensions,
            self.front_options_regime_mapping,
            prefix="front_options_regime",
        )
        return output


__all__ = ["FrozenFeatureRuntime"]
