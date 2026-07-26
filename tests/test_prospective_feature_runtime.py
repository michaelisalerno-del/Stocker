from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stocker_prospective.bundle import (
    BundleBuildSpec,
    BundleError,
    FeatureRuntimeBuildSpec,
    build_bundle,
    verify_bundle,
)
from stocker_prospective.feature_runtime import FrozenFeatureRuntime

ROOT = Path(__file__).parents[1]
H0_PARAMETERS = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260719-right-censored-regime-refit-v2/primary/full_refit_parameters.npz"
)
H0_PREPROCESSING = H0_PARAMETERS.with_name("full_refit_preprocessing.csv")
LOOP_DICTIONARY = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260718-loop-event-semantics-v2/primary/semantic_loop_dictionary_v2.csv"
)
FRONT_ROOT = (
    ROOT
    / "research/cross-market-context"
    / "20260723-daily-stock-front-options-context-v01/artifacts/primary"
)
FRONT_MANIFEST = FRONT_ROOT / "front_options_feature_manifest.json"
FRONT_MAPPING = FRONT_ROOT / "front_options_regime_mapping.json"
REGISTRY = ROOT / "configs/prospective/frozen-feature-runtime-v1.json"


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _base_spec(tmp_path: Path, runtime: FeatureRuntimeBuildSpec) -> BundleBuildSpec:
    source = tmp_path / "model"
    symbols = [
        "AAL",
        "AAOI",
        "APLD",
        "ASTS",
        "CIFR",
        "HIMS",
        "IONQ",
        "IREN",
        "MARA",
        "MP",
        "MRNA",
        "MSTR",
        "NVTS",
        "QBTS",
        "RGTI",
        "RIOT",
        "RIVN",
        "SMCI",
        "SOFI",
        "WULF",
    ]
    universe = {
        "universe_id": "anchor-frozen-20-v1",
        "cohort": "anchor_frozen_20",
        "symbols": symbols,
        "universe_hash": hashlib.sha256(
            (json.dumps(symbols, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "source_artifact": "model_coefficients.json#M1.category_levels.stock",
        "source_artifact_sha256": "e" * 64,
    }
    schema = {
        "schema_version": "1",
        "features": [
            {"name": "front_options_implied_tension", "dtype": "float64", "missing": "allow"},
            {"name": "stock", "dtype": "string", "missing": "reject"},
        ],
    }
    threshold = {
        "model": "M1",
        "value": 0.49588519865576763,
        "source": "weighted_2024_development_predictions",
        "frozen_before_holdout_outcomes": True,
    }
    return BundleBuildSpec(
        bundle_id="m1-feature-runtime-test-v1",
        created_at_utc=datetime(2026, 7, 26, 12, tzinfo=UTC),
        m0_artifact=_write(source / "m0.joblib", b"m0"),
        m1_artifact=_write(source / "m1.joblib", b"m1"),
        preprocessor=_write(source / "preprocessor.joblib", b"preprocessor"),
        feature_schema=_write(source / "schema.json", json.dumps(schema)),
        universe=_write(source / "universe.json", json.dumps(universe)),
        threshold=threshold["value"],
        threshold_provenance=_write(source / "threshold.json", json.dumps(threshold)),
        training_start="2024-01-01",
        training_end="2024-12-31",
        historical_reference_start="2025-01-01",
        historical_reference_end="2025-08-22",
        holdout_start="2025-09-01",
        holdout_end="2025-12-31",
        code_feature_contract_version="m1-group-o-plus-group-i-v1",
        previous_session_context_schema_hash="c" * 64,
        previous_session_context_feature_hash="d" * 64,
        feature_runtime=runtime,
    )


def _runtime_spec() -> FeatureRuntimeBuildSpec:
    return FeatureRuntimeBuildSpec(
        registry=REGISTRY,
        h0_parameters=H0_PARAMETERS,
        h0_preprocessing=H0_PREPROCESSING,
        loop_dictionary=LOOP_DICTIONARY,
        front_options_feature_manifest=FRONT_MANIFEST,
        front_options_regime_mapping=FRONT_MAPPING,
    )


def test_feature_runtime_is_copied_verified_and_loadable_without_research_paths(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    spec = _base_spec(tmp_path, _runtime_spec())

    build_bundle(spec, bundle)
    verification = verify_bundle(bundle)
    runtime = FrozenFeatureRuntime.load(verification, bundle_root=bundle)

    assert verification.verified is True
    assert verification.manifest.manifest_version == "2"
    assert verification.manifest.feature_runtime is not None
    assert runtime.h0_model_hash == (
        "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
    )
    assert runtime.loop_dictionary.dictionary_hash == (
        "497142c8d0ab880e59385da123d9eb2189469e9e3a4a631e0f63eb6fc77030d3"
    )
    manifest_text = (bundle / "manifest.json").read_text(encoding="utf-8")
    assert str(ROOT) not in manifest_text

    transformed = runtime.transform_previous_session_options(
        pd.DataFrame(
            [
                {
                    "atm_iv": 0.92145,
                    "straddle_mid_pct": 0.11360898680072756,
                    "call_put_iv_gap": 0.011450000000000016,
                    "skew_25d": None,
                    "combined_relative_spread": 0.10526315789473689,
                    "iv_minus_realised_20d": 0.07092555926389191,
                    "near_spot_oi_concentration": 0.2343130769116039,
                    "call_put_oi_imbalance": 0.4562339983046745,
                }
            ]
        )
    )

    assert transformed.loc[0, "skew_25d_missing"] == 1
    probabilities = [
        transformed.loc[0, f"front_options_regime_p_{index}"] for index in range(4)
    ]
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)


def test_feature_runtime_registry_or_artifact_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "runtime"
    copied.mkdir()
    paths: dict[str, Path] = {}
    for role, source in {
        "registry": REGISTRY,
        "h0_parameters": H0_PARAMETERS,
        "h0_preprocessing": H0_PREPROCESSING,
        "loop_dictionary": LOOP_DICTIONARY,
        "front_options_feature_manifest": FRONT_MANIFEST,
        "front_options_regime_mapping": FRONT_MAPPING,
    }.items():
        paths[role] = _write(copied / source.name, source.read_bytes())
    paths["h0_parameters"].write_bytes(paths["h0_parameters"].read_bytes() + b"tampered")
    runtime = FeatureRuntimeBuildSpec(**paths)

    with pytest.raises(BundleError, match="blocked_frozen_artifact_hash_mismatch"):
        build_bundle(_base_spec(tmp_path, runtime), tmp_path / "bundle")


def test_feature_runtime_registry_identity_is_pinned_independently(
    tmp_path: Path,
) -> None:
    altered_registry = _write(
        tmp_path / "altered-registry.json",
        REGISTRY.read_text(encoding="utf-8") + "\n",
    )
    runtime = _runtime_spec().model_copy(update={"registry": altered_registry})

    with pytest.raises(
        BundleError,
        match="blocked_frozen_artifact_hash_mismatch: feature-runtime registry differs",
    ):
        build_bundle(_base_spec(tmp_path, runtime), tmp_path / "bundle")


def test_feature_runtime_loader_reverifies_the_exact_bundle_root(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    build_bundle(_base_spec(tmp_path, _runtime_spec()), bundle)
    verification = verify_bundle(bundle)
    tampered = tmp_path / "tampered"
    shutil.copytree(bundle, tampered)
    identity = verification.manifest.feature_runtime
    assert identity is not None
    artifact = tampered / identity.h0_preprocessing.path
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(BundleError, match="blocked_frozen_artifact_hash_mismatch"):
        FrozenFeatureRuntime.load(verification, bundle_root=tampered)
