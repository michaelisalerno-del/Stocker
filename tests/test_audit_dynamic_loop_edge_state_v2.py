from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work"
    / "audit_dynamic_loop_edge_state_v2.py"
)
SPEC = importlib.util.spec_from_file_location("audit_dynamic_loop_edge_state_v2", AUDITOR_PATH)
assert SPEC and SPEC.loader
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDITOR
SPEC.loader.exec_module(AUDITOR)


def test_auditor_requires_strictly_prior_availability() -> None:
    frame = pd.DataFrame(
        {
            "availability": [
                pd.Timestamp("2025-01-02T14:29:59Z"),
                pd.NaT,
            ],
            "decision": [
                pd.Timestamp("2025-01-02T14:30:00Z"),
                pd.Timestamp("2025-01-03T14:30:00Z"),
            ],
        }
    )

    assert AUDITOR._timestamps_strictly_before(frame, "availability", "decision")
    frame.loc[0, "availability"] = frame.loc[0, "decision"]
    assert not AUDITOR._timestamps_strictly_before(frame, "availability", "decision")


def test_auditor_recomputes_manifest_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "result.csv"
    artifact.write_text("a\n1\n")
    manifest = {
        "files": [
            {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": AUDITOR.sha256(artifact),
            }
        ]
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    valid, _ = AUDITOR._manifest_hashes_are_valid(tmp_path)
    assert valid

    artifact.write_text("a\n2\n")
    valid, _ = AUDITOR._manifest_hashes_are_valid(tmp_path)
    assert not valid
