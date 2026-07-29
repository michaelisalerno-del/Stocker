from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "research"
    / "prospective"
    / "20260729-m1c-prospective-opening-reversal-v1-1"
    / "build_activation_artifacts.py"
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "m1c_opening_reversal_activation_builder_v1_1",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1_1_activation_package_is_verified_and_binds_exact_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    artifact_root = tmp_path / "staged" / "artifacts" / "primary"
    report_root = tmp_path / "staged" / "reports"
    monkeypatch.setattr(builder, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(builder, "REPORT_ROOT", report_root)
    monkeypatch.setattr(
        builder,
        "ACTIVATION_PATH",
        artifact_root / "experiment_activation_receipt_v1_1.json",
    )
    monkeypatch.setattr(
        builder,
        "_GIT_STATE_OVERRIDE",
        ("test/v1-1", "b" * 40, "clean"),
    )

    builder._build_activation_package()
    verified = builder._verify_package(
        artifact_root=artifact_root,
        report_root=report_root,
        emit=False,
    )

    assert verified["status"] == "verified"
    activation = builder.load_activation_receipt_v1_1(
        str(artifact_root / "experiment_activation_receipt_v1_1.json")
    )
    base = builder.load_activation_receipt_v1(str(builder.BASE_V1_ACTIVATION_PATH))
    assert activation.experiment_version == "1.1"
    assert activation.superseded_activation_receipt_hash_v1 == base.activation_receipt_hash
    assert not activation.scientific_rule_changed
    assert not activation.nominal_entry_changed
    assert not activation.capacity_policy_changed
    assert activation.configured_reserved_line_count == 12
    assert activation.order_routing_disabled
    assert not activation.order_methods_available

    summary = artifact_root / "summary_v1_1.json"
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact_hash_mismatch"):
        builder._verify_package(
            artifact_root=artifact_root,
            report_root=report_root,
            emit=False,
        )


def test_v1_1_activation_receipt_is_published_last(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    staged_artifact = tmp_path / "stage" / "artifacts" / "primary"
    staged_report = tmp_path / "stage" / "reports"
    published_artifact = tmp_path / "published" / "artifacts" / "primary"
    published_report = tmp_path / "published" / "reports"
    staged_artifact.mkdir(parents=True)
    staged_report.mkdir(parents=True)
    published_artifact.mkdir(parents=True)
    (staged_artifact / "experiment_activation_receipt_v1_1.json").write_text(
        '{"immutable":true}\n',
        encoding="utf-8",
    )
    (staged_artifact / "summary_v1_1.json").write_text(
        '{"complete":true}\n',
        encoding="utf-8",
    )
    (staged_report / "report.md").write_text("complete\n", encoding="utf-8")

    def interrupt() -> None:
        raise RuntimeError("simulated_interruption_before_receipt_publish")

    with pytest.raises(RuntimeError, match="simulated_interruption"):
        builder._publish_staged_package(
            staged_artifact_root=staged_artifact,
            staged_report_root=staged_report,
            published_artifact_root=published_artifact,
            published_report_root=published_report,
            before_artifact_publish=interrupt,
        )
    assert not (published_artifact / "experiment_activation_receipt_v1_1.json").exists()

    builder._publish_staged_package(
        staged_artifact_root=staged_artifact,
        staged_report_root=staged_report,
        published_artifact_root=published_artifact,
        published_report_root=published_report,
    )
    assert (published_artifact / "experiment_activation_receipt_v1_1.json").is_file()


def test_v1_1_activation_refuses_nonempty_existing_package(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    staged_artifact = tmp_path / "stage" / "artifacts" / "primary"
    staged_report = tmp_path / "stage" / "reports"
    published_artifact = tmp_path / "published" / "artifacts" / "primary"
    published_report = tmp_path / "published" / "reports"
    staged_artifact.mkdir(parents=True)
    staged_report.mkdir(parents=True)
    published_artifact.mkdir(parents=True)
    (staged_artifact / "experiment_activation_receipt_v1_1.json").write_text(
        '{"immutable":true}\n',
        encoding="utf-8",
    )
    existing = published_artifact / "experiment_activation_receipt_v1_1.json"
    existing.write_text('{"immutable":"existing"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        builder._publish_staged_package(
            staged_artifact_root=staged_artifact,
            staged_report_root=staged_report,
            published_artifact_root=published_artifact,
            published_report_root=published_report,
        )
    assert existing.read_text(encoding="utf-8") == ('{"immutable":"existing"}\n')
