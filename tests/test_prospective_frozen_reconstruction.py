from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from typer.testing import CliRunner

from stocker_prospective.bundle import (
    activate_bundle,
    install_bundle,
    verify_bundle,
)
from stocker_prospective.cli import app
from stocker_prospective.frozen_artifacts import (
    FrozenArtifactReconstructionError,
    reconstruct_frozen_artifacts,
)
from stocker_prospective.parity import FeatureParityError, load_feature_parity_report
from stocker_prospective.scoring import VerifiedFrozenScorer

ROOT = Path(__file__).parents[1]
FROZEN_ROOT = (
    ROOT
    / "research/options-feasibility"
    / "20260724-minimal-intraday-iv-excess-holdout-v01"
    / "artifacts/primary"
)
UNIVERSE = ROOT / "configs/prospective/anchor-frozen-20.json"
FEATURE_RUNTIME_REGISTRY = (
    ROOT / "configs/prospective/frozen-feature-runtime-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_json_reconstructs_deterministic_deployable_artifacts_without_refit(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = reconstruct_frozen_artifacts(
        frozen_root=FROZEN_ROOT,
        universe_path=UNIVERSE,
        output_directory=first,
        bundle_id="m1-frozen-20260724-v1",
        created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        operator="test-operator",
    )
    second_result = reconstruct_frozen_artifacts(
        frozen_root=FROZEN_ROOT,
        universe_path=UNIVERSE,
        output_directory=second,
        bundle_id="m1-frozen-20260724-v1",
        created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        operator="test-operator",
    )

    assert first_result.reconstruction_method == "deterministic_frozen_json_no_refit"
    assert second_result == first_result
    assert first_result.fit_invocations == 0
    assert first_result.protected_observations_read == 0
    for relative_path in (
        "m0.joblib",
        "m1.joblib",
        "preprocessor.joblib",
        "feature-schema.json",
        "context-schema.json",
        "threshold-provenance.json",
        "bundle-spec.yaml",
        "reconstruction-manifest.json",
    ):
        assert _sha256(first / relative_path) == _sha256(second / relative_path)

    coefficients = json.loads((FROZEN_ROOT / "model_coefficients.json").read_text())
    m1_spec = coefficients["M1"]
    values = {
        name: mean + scale * ((index % 7) - 3) / 4
        for index, (name, mean, scale) in enumerate(
            zip(
                m1_spec["numeric_features"],
                m1_spec["numeric_means"],
                m1_spec["numeric_scales"],
                strict=True,
            )
        )
    }
    values["stock"] = "WULF"
    ordered_names = [
        item["name"]
        for item in json.loads((first / "feature-schema.json").read_text())["features"]
    ]
    frame = joblib.load(first / "preprocessor.joblib").transform(
        pd.DataFrame(
            [[values[name] for name in ordered_names]],
            columns=ordered_names,
        )
    )

    m0_probability = float(joblib.load(first / "m0.joblib").predict_proba(frame)[0][1])
    m1_probability = float(joblib.load(first / "m1.joblib").predict_proba(frame)[0][1])

    assert m0_probability == pytest.approx(0.3559940260695174, abs=1e-15)
    assert m1_probability == pytest.approx(0.4150300063720971, abs=1e-15)


def test_reconstruction_rejects_a_frozen_source_hash_mismatch(tmp_path: Path) -> None:
    copied = tmp_path / "frozen"
    copied.mkdir()
    for name in (
        "model_coefficients.json",
        "minimal_feature_manifest.json",
        "model_configurations.json",
        "pre_outcome_freeze_manifest.json",
        "frozen_tail_thresholds.json",
        "historical_model_reconstruction.json",
        "lightweight_audit.json",
        "determinism_check.json",
    ):
        (copied / name).write_bytes((FROZEN_ROOT / name).read_bytes())
    coefficients = json.loads((copied / "model_coefficients.json").read_text())
    coefficients["M1"]["intercept"] += 0.01
    (copied / "model_coefficients.json").write_text(json.dumps(coefficients))

    with pytest.raises(
        FrozenArtifactReconstructionError,
        match="blocked_frozen_artifact_hash_mismatch",
    ):
        reconstruct_frozen_artifacts(
            frozen_root=copied,
            universe_path=UNIVERSE,
            output_directory=tmp_path / "output",
            bundle_id="m1-frozen-20260724-v1",
            created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
            operator="test-operator",
        )


def test_reconstruction_rejects_a_universe_not_bound_to_the_frozen_model(
    tmp_path: Path,
) -> None:
    universe = json.loads(UNIVERSE.read_text())
    universe["symbols"][-1] = "ZZZZ"
    changed = tmp_path / "changed-universe.json"
    changed.write_text(json.dumps(universe), encoding="utf-8")

    with pytest.raises(
        FrozenArtifactReconstructionError,
        match="blocked_frozen_universe_mismatch",
    ):
        reconstruct_frozen_artifacts(
            frozen_root=FROZEN_ROOT,
            universe_path=changed,
            output_directory=tmp_path / "output",
            bundle_id="m1-frozen-20260724-v1",
            created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
            operator="test-operator",
        )


def test_reconstruction_rejects_altered_or_failed_audit_evidence(tmp_path: Path) -> None:
    copied = tmp_path / "frozen"
    copied.mkdir()
    for name in (
        "model_coefficients.json",
        "minimal_feature_manifest.json",
        "model_configurations.json",
        "pre_outcome_freeze_manifest.json",
        "frozen_tail_thresholds.json",
        "historical_model_reconstruction.json",
        "lightweight_audit.json",
        "determinism_check.json",
    ):
        (copied / name).write_bytes((FROZEN_ROOT / name).read_bytes())
    determinism = json.loads((copied / "determinism_check.json").read_text())
    determinism["passed"] = False
    determinism["status"] = "failed"
    (copied / "determinism_check.json").write_text(json.dumps(determinism))

    with pytest.raises(
        FrozenArtifactReconstructionError,
        match="blocked_frozen_artifact_hash_mismatch",
    ):
        reconstruct_frozen_artifacts(
            frozen_root=copied,
            universe_path=UNIVERSE,
            output_directory=tmp_path / "output",
            bundle_id="m1-frozen-20260724-v1",
            created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
            operator="test-operator",
        )


def test_reconstructed_model_rejects_reordered_frozen_design_columns(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reconstructed"
    reconstruct_frozen_artifacts(
        frozen_root=FROZEN_ROOT,
        universe_path=UNIVERSE,
        output_directory=output,
        bundle_id="m1-frozen-20260724-v1",
        created_at_utc=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        operator="test-operator",
    )
    model = joblib.load(output / "m1.joblib")
    reordered = replace(
        model,
        design_columns=(
            model.design_columns[1],
            model.design_columns[0],
            *model.design_columns[2:],
        ),
    )
    schema = json.loads((output / "feature-schema.json").read_text())
    frame = pd.DataFrame(
        [[0.0 for _item in schema["features"][:-1]] + ["AAL"]],
        columns=[item["name"] for item in schema["features"]],
    )

    with pytest.raises(ValueError, match="blocked_feature_schema_mismatch"):
        reordered.predict_proba(frame)


def test_reconstruction_cli_builds_an_activatable_bundle_while_parity_stays_closed(
    tmp_path: Path,
) -> None:
    reconstructed = tmp_path / "reconstructed"
    bundle = tmp_path / "bundle"
    bundle_root = tmp_path / "server-bundles"
    runner = CliRunner()

    reconstruction = runner.invoke(
        app,
        [
            "bundle",
            "reconstruct",
            "--frozen-root",
            str(FROZEN_ROOT),
            "--universe",
            str(UNIVERSE),
            "--output",
            str(reconstructed),
            "--bundle-id",
            "m1-frozen-20260724-v1",
            "--created-at-utc",
            "2026-07-25T12:00:00Z",
            "--operator",
            "test-operator",
        ],
    )
    built = runner.invoke(
        app,
        [
            "bundle",
            "build",
            "--spec",
            str(reconstructed / "bundle-spec.yaml"),
            "--output",
            str(bundle),
        ],
    )

    assert reconstruction.exit_code == 0, reconstruction.stdout
    assert built.exit_code == 0, built.stdout
    assert verify_bundle(bundle).verified is True
    installed = install_bundle(bundle, bundle_root, operator="test-operator")
    activate_bundle(
        "m1-frozen-20260724-v1",
        bundle_root,
        operator="test-operator",
        expected_current_bundle_id=None,
    )
    scorer = VerifiedFrozenScorer.load(
        verify_bundle(installed),
        installed_bundle_path=installed,
    )
    assert scorer.verification.manifest.threshold.value == 0.49588519865576763

    parity = load_feature_parity_report(
        ROOT / "configs/prospective/feature-parity-m1.json"
    )
    with pytest.raises(
        FeatureParityError,
        match="blocked_feature_source_semantics_mismatch",
    ):
        parity.require_scoring_allowed()


def test_reconstruction_can_embed_registered_feature_runtime_without_refitting(
    tmp_path: Path,
) -> None:
    reconstructed = tmp_path / "reconstructed"
    bundle = tmp_path / "bundle"
    runner = CliRunner()

    reconstruction = runner.invoke(
        app,
        [
            "bundle",
            "reconstruct",
            "--frozen-root",
            str(FROZEN_ROOT),
            "--universe",
            str(UNIVERSE),
            "--feature-runtime-registry",
            str(FEATURE_RUNTIME_REGISTRY),
            "--repository-root",
            str(ROOT),
            "--output",
            str(reconstructed),
            "--bundle-id",
            "m1-frozen-feature-runtime-test-v1",
            "--created-at-utc",
            "2026-07-26T12:00:00Z",
            "--operator",
            "test-operator",
        ],
    )
    built = runner.invoke(
        app,
        [
            "bundle",
            "build",
            "--spec",
            str(reconstructed / "bundle-spec.yaml"),
            "--output",
            str(bundle),
        ],
    )

    assert reconstruction.exit_code == 0, reconstruction.stdout
    assert built.exit_code == 0, built.stdout
    verification = verify_bundle(bundle)
    assert verification.verified is True
    assert verification.manifest.manifest_version == "2"
    assert verification.manifest.feature_runtime is not None
    assert verification.manifest.feature_runtime.scoring_authorized_by_registry is False
