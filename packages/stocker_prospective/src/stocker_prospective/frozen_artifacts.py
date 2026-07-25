"""Deterministic deployment reconstruction from the audited frozen JSON handoff."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

FROZEN_INPUT_FILES: Final = (
    "model_coefficients.json",
    "minimal_feature_manifest.json",
    "model_configurations.json",
    "pre_outcome_freeze_manifest.json",
    "frozen_tail_thresholds.json",
    "historical_model_reconstruction.json",
    "lightweight_audit.json",
    "determinism_check.json",
)
MODEL_NAMES: Final = ("M0", "M1")
CONTRACT_VERSION: Final = "minimal-intraday-iv-excess-holdout-v01-group-o-plus-group-i"
RECONSTRUCTION_METHOD: Final = "deterministic_frozen_json_no_refit"
AUTHORIZATION_SCOPE: Final = (
    "deterministic reconstruction of deployable M0/M1 artifacts from the audited "
    "frozen JSON, without refitting, threshold changes, protected-data research, "
    "or live/paper trading"
)
AUDITED_SOURCE_HASHES: Final[dict[str, str]] = {
    "model_coefficients.json": "49bd22e47b20274b1fe058ae15d899fb4a3a5e18feb418f990ef5139528161b9",
    "minimal_feature_manifest.json": (
        "5c8743f7fa424a2aeca3a9a64e7696099377bc79a4ad8a240de7d9efc4be5e34"
    ),
    "model_configurations.json": (
        "384bc6ad64f430481d62ca7a6986f91c1227ae3a52e3117a7a59d9efc2859656"
    ),
    "pre_outcome_freeze_manifest.json": (
        "461ef18a35636b78c7d2f1e7efd77ef9aadd38722afc74e795be6315bc75445e"
    ),
    "frozen_tail_thresholds.json": (
        "0f291a281410818591b64d3c4ed6bf668032a9dac229a3c50e7a1d961437c5c3"
    ),
    "historical_model_reconstruction.json": (
        "101430b6cd3c786611f34788a86082e95bd60686495dff7dda202a3eccf7a384"
    ),
    "lightweight_audit.json": (
        "65cabbe5d83c7b4d88f540b2eeb8245766113ae018ca5f7a83c4243ce484b4f9"
    ),
    "determinism_check.json": (
        "6f865c4950f84a4a3ab52839a53a0a24bc49af5fbadbd759fae3465532076f0f"
    ),
}
AUDITED_UNIVERSE_SHA256: Final = (
    "af2391ea47e0097b16979151e3c69f6c4335755033a323db418759573c3991e3"
)

REQUIRED_SAFETY_FLAGS: Final[dict[str, object]] = {
    "daily_stock_features_excluded": True,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "hand_built_mismatch_features_excluded": True,
    "intraday_option_quotes_used": False,
    "minimal_options_plus_intraday_h0_model": True,
    "mismatch_features_excluded": True,
    "option_pnl_calculated": False,
    "order_placement": "disabled",
    "partial_cohort_model_allowed": False,
    "previous_close_options_only": True,
    "route_competition_features_excluded": True,
    "route_state_features_excluded": True,
    "strategy_promotion": False,
    "top_5_percent_threshold_frozen": True,
}


class FrozenArtifactReconstructionError(RuntimeError):
    """The audited frozen handoff cannot be reconstructed without invention."""


class FrozenArtifactReconstruction(BaseModel):
    """Machine-readable evidence emitted with reconstructed artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = "1"
    bundle_id: str
    created_at_utc: datetime
    operator: str
    authorization_scope: str
    reconstruction_method: str
    fit_invocations: int = Field(ge=0)
    protected_observations_read: int = Field(ge=0)
    source_hashes: dict[str, str]
    output_hashes: dict[str, str]
    feature_schema_hash: str
    context_schema_hash: str
    context_feature_hash: str
    frozen_threshold: float
    universe_hash: str


@dataclass(frozen=True)
class OrderedFeatureFramePreprocessor:
    """Validate and pass through the frozen ordered raw feature frame."""

    feature_names: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return an isolated frame only when the ordered schema is exact."""

        if tuple(str(column) for column in frame.columns) != self.feature_names:
            raise ValueError("blocked_feature_schema_mismatch: feature ordering differs")
        return frame.copy()


@dataclass(frozen=True)
class ReconstructedFrozenLogisticModel:
    """No-fit logistic scorer containing only the frozen numerical handoff."""

    model_id: str
    numeric_features: tuple[str, ...]
    numeric_medians: tuple[float, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    stock_levels: tuple[str, ...]
    design_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the frozen preprocessing and return two-class probabilities."""

        expected_design = (
            *self.numeric_features,
            *(f"control_stock__{level}" for level in self.stock_levels[1:]),
        )
        if self.design_columns != expected_design:
            raise ValueError("blocked_feature_schema_mismatch: design columns differ")
        missing = sorted({*self.numeric_features, "stock"}.difference(frame.columns))
        if missing:
            raise ValueError(f"blocked_feature_schema_mismatch: missing features {missing}")
        raw = frame.loc[:, list(self.numeric_features)].to_numpy(dtype=np.float64)
        medians = np.asarray(self.numeric_medians, dtype=np.float64)
        means = np.asarray(self.numeric_means, dtype=np.float64)
        scales = np.asarray(self.numeric_scales, dtype=np.float64)
        numeric = np.where(np.isfinite(raw), raw, medians)
        parts = [np.asarray((numeric - means) / scales, dtype=np.float64)]
        observed = frame["stock"].astype(str).to_numpy()
        if not bool(np.isin(observed, np.asarray(self.stock_levels, dtype=object)).all()):
            raise ValueError("blocked_frozen_universe_mismatch: unknown stock control")
        for level in self.stock_levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
        design = np.concatenate(parts, axis=1)
        if design.shape[1] != len(self.design_columns):
            raise ValueError("blocked_feature_schema_mismatch: design width differs")
        linear = design @ np.asarray(self.coefficients, dtype=np.float64) + self.intercept
        positive = np.asarray(
            1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
            dtype=np.float64,
        )
        return np.column_stack((1.0 - positive, positive))


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactReconstructionError(
            f"blocked_missing_verified_frozen_bundle: cannot read {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {path.name} must contain an object"
        )
    return payload


def _require_flags(payload: dict[str, Any], source_name: str) -> None:
    mismatches = [
        name for name, expected in REQUIRED_SAFETY_FLAGS.items() if payload.get(name) != expected
    ]
    if mismatches:
        raise FrozenArtifactReconstructionError(
            "blocked_unsafe_runtime_configuration: "
            f"{source_name} safety flags differ: {', '.join(mismatches)}"
        )


def _require_hash(actual_path: Path, expected: object, role: str) -> None:
    if not isinstance(expected, str) or _sha256(actual_path) != expected:
        raise FrozenArtifactReconstructionError(
            f"blocked_frozen_artifact_hash_mismatch: {role}"
        )


def _finite_tuple(values: object, *, role: str) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {role} must be a list"
        )
    converted = tuple(float(value) for value in values)
    if not converted or not all(math.isfinite(value) for value in converted):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {role} contains non-finite values"
        )
    return converted


def _string_tuple(values: object, *, role: str) -> tuple[str, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) for value in values)
    ):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {role} must be non-empty strings"
        )
    result = tuple(values)
    if len(result) != len(set(result)):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {role} contains duplicates"
        )
    return result


def _model_from_specification(
    name: str,
    specification: object,
    *,
    expected_features: tuple[str, ...],
    expected_stocks: tuple[str, ...],
) -> ReconstructedFrozenLogisticModel:
    if not isinstance(specification, dict):
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {name} specification is absent"
        )
    numeric_features = _string_tuple(
        specification.get("numeric_features"),
        role=f"{name}.numeric_features",
    )
    category_controls = specification.get("category_controls")
    category_levels = specification.get("category_levels")
    stock_levels = (
        _string_tuple(category_levels.get("stock"), role=f"{name}.category_levels.stock")
        if isinstance(category_levels, dict)
        else ()
    )
    medians = _finite_tuple(specification.get("numeric_medians"), role=f"{name}.medians")
    means = _finite_tuple(specification.get("numeric_means"), role=f"{name}.means")
    scales = _finite_tuple(specification.get("numeric_scales"), role=f"{name}.scales")
    design_columns = _string_tuple(
        specification.get("design_columns"),
        role=f"{name}.design_columns",
    )
    coefficients = _finite_tuple(
        specification.get("coefficients"),
        role=f"{name}.coefficients",
    )
    try:
        intercept = float(specification["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {name}.intercept is invalid"
        ) from exc
    expected_design_width = len(numeric_features) + len(stock_levels) - 1
    expected_design_columns = (
        *numeric_features,
        *(f"control_stock__{level}" for level in stock_levels[1:]),
    )
    valid = (
        specification.get("model_id") == name
        and specification.get("kind") == "logistic"
        and category_controls == ["stock"]
        and numeric_features == expected_features
        and stock_levels == expected_stocks
        and len(medians) == len(numeric_features)
        and len(means) == len(numeric_features)
        and len(scales) == len(numeric_features)
        and all(scale > 0.0 for scale in scales)
        and design_columns == expected_design_columns
        and len(design_columns) == expected_design_width
        and len(coefficients) == expected_design_width
        and math.isfinite(intercept)
    )
    if not valid:
        raise FrozenArtifactReconstructionError(
            f"blocked_feature_schema_mismatch: {name} frozen numerical contract differs"
        )
    return ReconstructedFrozenLogisticModel(
        model_id=name,
        numeric_features=numeric_features,
        numeric_medians=medians,
        numeric_means=means,
        numeric_scales=scales,
        stock_levels=stock_levels,
        design_columns=design_columns,
        coefficients=coefficients,
        intercept=intercept,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload))


def _dump_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path, compress=0, protocol=5)


def _copy_reference(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def reconstruct_frozen_artifacts(
    *,
    frozen_root: str | Path,
    universe_path: str | Path,
    output_directory: str | Path,
    bundle_id: str,
    created_at_utc: datetime,
    operator: str,
) -> FrozenArtifactReconstruction:
    """Reconstruct deployable scorers without fitting or reading observations."""

    source_root = Path(frozen_root)
    universe_source = Path(universe_path)
    output = Path(output_directory)
    if output.exists():
        raise FrozenArtifactReconstructionError(f"refusing to overwrite {output}")
    if not operator.strip():
        raise FrozenArtifactReconstructionError("operator identity is required")
    if created_at_utc.tzinfo is None or created_at_utc.utcoffset() is None:
        raise FrozenArtifactReconstructionError("created_at_utc must be timezone-aware")
    sources = {name: source_root / name for name in FROZEN_INPUT_FILES}
    missing = [str(path) for path in (*sources.values(), universe_source) if not path.is_file()]
    if missing:
        raise FrozenArtifactReconstructionError(
            "blocked_missing_verified_frozen_bundle: " + ", ".join(missing)
        )
    for name, expected_hash in AUDITED_SOURCE_HASHES.items():
        _require_hash(sources[name], expected_hash, f"approved {name}")
    if _sha256(universe_source) != AUDITED_UNIVERSE_SHA256:
        raise FrozenArtifactReconstructionError(
            "blocked_frozen_universe_mismatch: approved universe identity differs"
        )

    payloads = {name: _load_object(path) for name, path in sources.items()}
    for name in (
        "model_coefficients.json",
        "minimal_feature_manifest.json",
        "model_configurations.json",
        "pre_outcome_freeze_manifest.json",
        "frozen_tail_thresholds.json",
    ):
        _require_flags(payloads[name], name)

    freeze = payloads["pre_outcome_freeze_manifest.json"]
    _require_hash(
        sources["model_coefficients.json"],
        freeze.get("model_coefficients_sha256"),
        "model coefficients",
    )
    _require_hash(
        sources["minimal_feature_manifest.json"],
        freeze.get("feature_manifest_sha256"),
        "feature manifest",
    )
    _require_hash(
        sources["model_configurations.json"],
        freeze.get("model_configurations_sha256"),
        "model configurations",
    )
    _require_hash(
        sources["frozen_tail_thresholds.json"],
        freeze.get("frozen_tail_thresholds_sha256"),
        "frozen threshold",
    )
    if (
        freeze.get("frozen") is not True
        or freeze.get("holdout_outcomes_read_before_freeze") is not False
        or freeze.get("coverage_preflight_passed_before_freeze") is not True
    ):
        raise FrozenArtifactReconstructionError(
            "blocked_unsafe_runtime_configuration: pre-outcome freeze is invalid"
        )
    reconstruction_audit = payloads["historical_model_reconstruction.json"]
    lightweight_audit = payloads["lightweight_audit.json"]
    determinism = payloads["determinism_check.json"]
    if (
        reconstruction_audit.get("passed") is not True
        or reconstruction_audit.get("M0_exactly_reproduced") is not True
        or reconstruction_audit.get("M1_changed_after_reference_inspection") is not False
        or lightweight_audit.get("passed") is not True
        or lightweight_audit.get("independent_audit_passed") is not True
        or determinism.get("passed") is not True
        or determinism.get("status") != "passed"
        or any(
            float(determinism.get(name, math.inf)) != 0.0
            for name in (
                "maximum_feature_difference",
                "maximum_probability_difference",
                "joined_row_mismatches",
                "selected_contract_mismatches",
                "tail_membership_mismatches",
            )
        )
    ):
        raise FrozenArtifactReconstructionError(
            "blocked_frozen_artifact_hash_mismatch: required audit evidence did not pass"
        )

    feature_manifest = payloads["minimal_feature_manifest.json"]
    configurations = payloads["model_configurations.json"]
    try:
        expected_features = {
            name: _string_tuple(
                feature_manifest["models"][name]["numeric_features"],
                role=f"feature_manifest.models.{name}",
            )
            for name in MODEL_NAMES
        }
    except (KeyError, TypeError) as exc:
        raise FrozenArtifactReconstructionError(
            "blocked_feature_schema_mismatch: frozen model feature manifest is invalid"
        ) from exc
    if (
        expected_features["M1"]
        != tuple(
            [
                *feature_manifest.get("group_O", {}).get("numeric_features", []),
                *feature_manifest.get("group_I", {}).get("numeric_features", []),
            ]
        )
        or expected_features["M0"]
        != tuple(feature_manifest.get("group_O", {}).get("numeric_features", []))
        or any(
            configurations.get(name, {}).get("numeric_features") != list(expected_features[name])
            for name in MODEL_NAMES
        )
    ):
        raise FrozenArtifactReconstructionError(
            "blocked_feature_schema_mismatch: Group O / Group I ordering differs"
        )

    universe = _load_object(universe_source)
    expected_stocks = _string_tuple(universe.get("symbols"), role="universe.symbols")
    coefficients = payloads["model_coefficients.json"]
    _require_hash(
        sources["model_coefficients.json"],
        universe.get("source_artifact_sha256"),
        "universe source",
    )
    canonical_stocks = json.dumps(
        list(expected_stocks),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    calculated_universe_hash = hashlib.sha256(
        (canonical_stocks + "\n").encode("utf-8")
    ).hexdigest()
    m1_levels = coefficients.get("M1", {}).get("category_levels", {})
    if (
        len(expected_stocks) != 20
        or any(stock != stock.upper() for stock in expected_stocks)
        or universe.get("cohort") != "anchor_frozen_20"
        or universe.get("source_field") != "M1.category_levels.stock"
        or universe.get("universe_hash") != calculated_universe_hash
        or not isinstance(m1_levels, dict)
        or m1_levels.get("stock") != list(expected_stocks)
    ):
        raise FrozenArtifactReconstructionError(
            "blocked_frozen_universe_mismatch: registered anchor differs from frozen M1"
        )
    models = {
        name: _model_from_specification(
            name,
            coefficients.get(name),
            expected_features=expected_features[name],
            expected_stocks=expected_stocks,
        )
        for name in MODEL_NAMES
    }
    threshold_payload = payloads["frozen_tail_thresholds.json"]
    try:
        threshold = float(threshold_payload["M1_top_5_percent_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactReconstructionError(
            "blocked_feature_schema_mismatch: M1 threshold is absent"
        ) from exc
    if (
        not math.isfinite(threshold)
        or threshold != freeze.get("threshold_values", {}).get("M1_top_5_percent_threshold")
        or threshold_payload.get("method") != "deterministic_midpoint_cdf_weighted_quantile"
        or threshold_payload.get("written_before_holdout_outcomes") is not True
    ):
        raise FrozenArtifactReconstructionError(
            "blocked_feature_schema_mismatch: threshold provenance differs"
        )

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        feature_definitions = [
            {"name": name, "dtype": "float64", "missing": "allow"}
            for name in expected_features["M1"]
        ]
        feature_definitions.append({"name": "stock", "dtype": "string", "missing": "reject"})
        feature_schema = {"schema_version": "1", "features": feature_definitions}
        context_features = [
            {"name": name, "dtype": "float64", "missing": "allow"}
            for name in expected_features["M0"]
        ]
        context_feature_hash = hashlib.sha256(_canonical_json(context_features)).hexdigest()
        context_schema = {
            "schema_version": "1",
            "session_semantics": "exact_previous_valid_xnys_session",
            "source_group": "group_O_previous_close_front_options_context",
            "features": context_features,
            "feature_hash": context_feature_hash,
        }
        threshold_provenance = {
            "model": "M1",
            "value": threshold,
            "source": "weighted_2024_development_predictions",
            "frozen_before_holdout_outcomes": True,
            "source_artifact": "frozen_tail_thresholds.json",
            "source_artifact_sha256": _sha256(sources["frozen_tail_thresholds.json"]),
        }
        preprocessor = OrderedFeatureFramePreprocessor(
            feature_names=tuple(item["name"] for item in feature_definitions)
        )
        _dump_joblib(temporary / "m0.joblib", models["M0"])
        _dump_joblib(temporary / "m1.joblib", models["M1"])
        _dump_joblib(temporary / "preprocessor.joblib", preprocessor)
        _write_json(temporary / "feature-schema.json", feature_schema)
        _write_json(temporary / "context-schema.json", context_schema)
        _write_json(temporary / "threshold-provenance.json", threshold_provenance)
        _copy_reference(universe_source, temporary / "universe.json")
        audit_paths = []
        for name in (
            "pre_outcome_freeze_manifest.json",
            "historical_model_reconstruction.json",
            "lightweight_audit.json",
        ):
            relative = Path("references/audit") / name
            _copy_reference(sources[name], temporary / relative)
            audit_paths.append(relative.as_posix())
        audit_paths.append("context-schema.json")
        determinism_paths = []
        for name in ("determinism_check.json",):
            relative = Path("references/determinism") / name
            _copy_reference(sources[name], temporary / relative)
            determinism_paths.append(relative.as_posix())

        source_hashes = {name: _sha256(path) for name, path in sources.items()}
        output_names = (
            "m0.joblib",
            "m1.joblib",
            "preprocessor.joblib",
            "feature-schema.json",
            "context-schema.json",
            "threshold-provenance.json",
            "universe.json",
            *audit_paths,
            *determinism_paths,
        )
        output_names = tuple(dict.fromkeys(output_names))
        output_hashes = {name: _sha256(temporary / name) for name in output_names}
        result = FrozenArtifactReconstruction(
            bundle_id=bundle_id,
            created_at_utc=created_at_utc.astimezone(UTC),
            operator=operator,
            authorization_scope=AUTHORIZATION_SCOPE,
            reconstruction_method=RECONSTRUCTION_METHOD,
            fit_invocations=0,
            protected_observations_read=0,
            source_hashes=source_hashes,
            output_hashes=output_hashes,
            feature_schema_hash=output_hashes["feature-schema.json"],
            context_schema_hash=output_hashes["context-schema.json"],
            context_feature_hash=context_feature_hash,
            frozen_threshold=threshold,
            universe_hash=str(universe.get("universe_hash", "")),
        )
        _write_json(
            temporary / "reconstruction-manifest.json",
            result.model_dump(mode="json"),
        )
        timestamp = created_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z")
        bundle_spec = {
            "bundle_id": bundle_id,
            "created_at_utc": timestamp,
            "m0_artifact": "m0.joblib",
            "m1_artifact": "m1.joblib",
            "preprocessor": "preprocessor.joblib",
            "feature_schema": "feature-schema.json",
            "universe": "universe.json",
            "threshold": threshold,
            "threshold_provenance": "threshold-provenance.json",
            "training_start": "2024-01-01",
            "training_end": "2024-12-31",
            "historical_reference_start": "2025-01-01",
            "historical_reference_end": "2025-08-22",
            "holdout_start": "2025-09-01",
            "holdout_end": "2025-12-31",
            "protected_start": "2026-01-01",
            "code_feature_contract_version": CONTRACT_VERSION,
            "previous_session_context_schema_hash": result.context_schema_hash,
            "previous_session_context_feature_hash": result.context_feature_hash,
            "audit_references": [*audit_paths, "reconstruction-manifest.json"],
            "determinism_references": determinism_paths,
        }
        (temporary / "bundle-spec.yaml").write_text(
            yaml.safe_dump(bundle_spec, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
