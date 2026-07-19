from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.observable_event_ranking_v1.artifacts import (
    ArtifactBinding,
    ArtifactWriter,
    sha256_file,
)
from stocker_research.observable_event_ranking_v1.cli import (
    COMMANDS,
    _freeze_prospective,
    build_parser,
)
from stocker_research.observable_event_ranking_v1.contract import (
    REQUIRED_SAFETY_FLAGS,
    canonical_hash,
)
from stocker_research.observable_event_ranking_v1.pipeline import StageDependencyError


def test_cli_exposes_every_frozen_stage_without_a_protected_data_bypass() -> None:
    assert COMMANDS == (
        "preflight",
        "build-events",
        "audit-events",
        "build-targets",
        "run-development",
        "audit-development",
        "exact-rerun",
        "freeze-prospective",
        "score-prospective",
        "settle-prospective",
        "ibkr-resolve-contracts",
        "ibkr-capture-quotes",
        "ibkr-observability-dry-run",
    )
    parser = build_parser()
    help_text = parser.format_help()

    assert "allow-protected" not in help_text
    assert "bypass-cutoff" not in help_text


def test_required_standalone_runner_files_exist() -> None:
    work = Path(
        "research/observable-event-ranking/"
        "20260719-observable-event-cross-sectional-ranking-v1/work"
    )

    assert (work / "run_observable_event_ranking_v1.py").exists()
    assert (work / "audit_observable_event_ranking_v1.py").exists()
    assert (work / "run_exact_rerun_v1.py").exists()


def test_prospective_freeze_hash_binds_every_required_component(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    binding = ArtifactBinding(
        git_sha="abc",
        branch="agent/observable-event-ranking-v1",
        contract_hash="contract",
        implementation_hash="implementation",
        data_snapshot_hash="data",
        universe_hash="universe",
        sector_map_hash="sector",
        run_id="run",
        random_seeds={"bootstrap": 20260719},
        dependency_versions={"python": "3.12"},
        safety=REQUIRED_SAFETY_FLAGS,
    )
    writer = ArtifactWriter(primary, binding)
    for name in (
        "frozen_experiment_contract.json",
        "event_threshold.json",
        "feature_manifest.json",
        "target_contract.json",
        "prospective_model_parameters.json",
        "strongest_baseline_selection.json",
        "prospective_baseline_parameters.json",
        "source_identity_manifest.json",
        "implementation_source_manifest.json",
        "environment_manifest.json",
    ):
        writer.json(name, {"component": name})
    writer.parquet(
        "universe_ledger.parquet",
        pd.DataFrame({"symbol": pd.Series(dtype="string")}),
        columns=("symbol",),
    )
    writer.parquet(
        "sector_membership_ledger.parquet",
        pd.DataFrame({"sector": pd.Series(dtype="string")}),
        columns=("sector",),
    )
    writer.json(
        "development_decision.json",
        {
            "decision": "historical_incremental_ranking_evidence_supports_prospective_freeze",
            "passed": True,
            "authorises_prospective_freeze": True,
        },
    )
    writer.manifest()

    _freeze_prospective(argparse.Namespace(artifact_root=tmp_path))

    freeze = json.loads((primary / "prospective_freeze_manifest.json").read_text(encoding="utf-8"))
    unhashed = {key: value for key, value in freeze.items() if key != "bundle_hash"}
    assert freeze["bundle_hash"] == canonical_hash(unhashed)
    assert all(
        sha256_file(primary / component["path"]) == component["sha256"]
        for component in freeze["components"]
    )
    with pytest.raises(StageDependencyError):
        _freeze_prospective(argparse.Namespace(artifact_root=tmp_path))
