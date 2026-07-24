from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stocker_prospective.bundle import (
    BundleBuildSpec,
    BundleError,
    activate_bundle,
    build_bundle,
    install_bundle,
    list_installed_bundles,
    validate_feature_vector,
    verify_bundle,
)

ANCHOR_SYMBOLS = [
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


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _build_spec(tmp_path: Path) -> BundleBuildSpec:
    source = tmp_path / "research-machine"
    universe = {
        "universe_id": "anchor-frozen-20-v1",
        "cohort": "anchor_frozen_20",
        "symbols": ANCHOR_SYMBOLS,
        "source_artifact": "model_coefficients.json#M1.category_levels.stock",
    }
    feature_schema = {
        "schema_version": "1",
        "features": [
            {"name": "front_options_implied_tension", "dtype": "float64", "missing": "reject"},
            {"name": "checkpoint_6", "dtype": "float64", "missing": "reject"},
            {"name": "arousal", "dtype": "float64", "missing": "reject"},
        ],
    }
    threshold_provenance = {
        "model": "M1",
        "value": 0.49588519865576763,
        "source": "weighted_2024_development_predictions",
        "frozen_before_holdout_outcomes": True,
    }
    return BundleBuildSpec(
        bundle_id="m1-frozen-test-v1",
        created_at_utc=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        m0_artifact=_write(source / "m0.joblib", b"frozen-m0"),
        m1_artifact=_write(source / "m1.joblib", b"frozen-m1"),
        preprocessor=_write(source / "preprocessor.joblib", b"frozen-preprocessor"),
        feature_schema=_write(
            source / "feature-schema.json", json.dumps(feature_schema, sort_keys=True)
        ),
        universe=_write(source / "universe.json", json.dumps(universe, sort_keys=True)),
        threshold=0.49588519865576763,
        threshold_provenance=_write(
            source / "threshold-provenance.json",
            json.dumps(threshold_provenance, sort_keys=True),
        ),
        training_start="2024-01-01",
        training_end="2024-12-31",
        historical_reference_start="2025-01-01",
        historical_reference_end="2025-08-22",
        holdout_start="2025-09-01",
        holdout_end="2025-12-31",
        protected_start="2026-01-01",
        code_feature_contract_version="m1-group-o-plus-group-i-v1",
        audit_references=[
            _write(source / "audit.json", '{"passed":true}'),
        ],
        determinism_references=[
            _write(source / "determinism.json", '{"passed":true}'),
        ],
    )


def test_bundle_is_verified_and_independent_of_research_machine(tmp_path: Path) -> None:
    spec = _build_spec(tmp_path)
    output = tmp_path / "built" / spec.bundle_id

    built = build_bundle(spec, output)
    source_root = spec.m0_artifact.parent
    for path in source_root.iterdir():
        path.unlink()
    source_root.rmdir()

    result = verify_bundle(output)

    assert built.bundle_id == spec.bundle_id
    assert result.verified is True
    assert result.blockers == []
    assert result.manifest.universe.cohort == "anchor_frozen_20"
    assert result.manifest.universe.symbol_count == 20
    assert str(source_root) not in (output / "manifest.json").read_text(encoding="utf-8")


def test_bundle_hash_mismatch_and_missing_artifact_fail_closed(tmp_path: Path) -> None:
    spec = _build_spec(tmp_path)
    output = tmp_path / "bundle"
    build_bundle(spec, output)
    (output / "artifacts" / "m1.joblib").write_bytes(b"tampered")

    result = verify_bundle(output)

    assert result.verified is False
    assert "blocked_frozen_artifact_hash_mismatch" in result.blockers

    missing_spec = _build_spec(tmp_path / "missing")
    missing_spec.m1_artifact.unlink()
    with pytest.raises(BundleError, match="blocked_missing_verified_frozen_bundle"):
        build_bundle(missing_spec, tmp_path / "never-built")


def test_feature_order_dtype_and_universe_are_strict(tmp_path: Path) -> None:
    spec = _build_spec(tmp_path)
    output = tmp_path / "bundle"
    build_bundle(spec, output)
    manifest = verify_bundle(output).manifest

    validate_feature_vector(
        manifest,
        [
            ("front_options_implied_tension", 0.1),
            ("checkpoint_6", 1.0),
            ("arousal", 0.2),
        ],
    )
    with pytest.raises(BundleError, match="blocked_feature_schema_mismatch"):
        validate_feature_vector(
            manifest,
            [
                ("checkpoint_6", 1.0),
                ("front_options_implied_tension", 0.1),
                ("arousal", 0.2),
            ],
        )
    with pytest.raises(BundleError, match="blocked_feature_schema_mismatch"):
        validate_feature_vector(
            manifest,
            [
                ("front_options_implied_tension", "0.1"),
                ("checkpoint_6", 1.0),
                ("arousal", 0.2),
            ],
        )

    invalid = _build_spec(tmp_path / "bad-universe")
    payload = json.loads(invalid.universe.read_text(encoding="utf-8"))
    payload["symbols"].pop()
    invalid.universe.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="blocked_frozen_universe_mismatch"):
        build_bundle(invalid, tmp_path / "bad-bundle")


def test_install_is_immutable_and_activation_is_atomic(tmp_path: Path) -> None:
    spec = _build_spec(tmp_path)
    built = tmp_path / "built"
    build_bundle(spec, built)
    bundle_root = tmp_path / "server-data" / "bundles"

    installed = install_bundle(built, bundle_root, operator="test-operator")
    active = activate_bundle(
        spec.bundle_id,
        bundle_root,
        operator="test-operator",
        expected_current_bundle_id=None,
    )

    assert installed == bundle_root / "installed" / spec.bundle_id
    assert [item.bundle_id for item in list_installed_bundles(bundle_root)] == [spec.bundle_id]
    assert active.bundle_id == spec.bundle_id
    assert not (bundle_root / ".active.tmp").exists()
    with pytest.raises(PermissionError):
        (installed / "artifacts" / "m1.joblib").write_bytes(b"mutated")
    with pytest.raises(BundleError, match="installed bundle already exists"):
        install_bundle(built, bundle_root, operator="test-operator")
    with pytest.raises(BundleError, match="active bundle changed"):
        activate_bundle(
            spec.bundle_id,
            bundle_root,
            operator="test-operator",
            expected_current_bundle_id="some-other-bundle",
        )
