from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

WORK = Path(__file__).resolve().parents[1] / "research/slrno-v2/20260714-regime-loop-handoff/work"
AUDITOR = WORK / "audit_directed_economic_loop_regime_rotation_v1.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rotation_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_target_window_uses_trading_calendar_not_calendar_days() -> None:
    module = _load()
    sessions = ["2025-01-03", "2025-01-06", "2025-01-08", "2025-01-09"]

    assert module.target_window(sessions, "2025-01-03", 3) == [
        "2025-01-06",
        "2025-01-08",
        "2025-01-09",
    ]
    assert module.target_window(sessions, "2025-01-08", 3) is None


def test_exact_identity_detects_changed_machine_artifact(tmp_path: Path) -> None:
    module = _load()
    primary = tmp_path / "primary"
    rerun = tmp_path / "rerun"
    primary.mkdir()
    rerun.mkdir()
    (primary / "ledger.csv").write_bytes(b"id\n1\n")
    (rerun / "ledger.csv").write_bytes(b"id\n2\n")

    result = module.verify_exact_machine_identity(primary, rerun)

    assert result["byte_identical"] is False
    assert result["hash_mismatches"] == ["ledger.csv"]


def test_research_safety_rejects_runtime_paths() -> None:
    module = _load()

    assert (
        module.prohibited_changed_paths(
            [
                "research/slrno-v2/work/runner.py",
                "packages/stocker_research/src/stocker_research/model.py",
            ]
        )
        == []
    )
    assert module.prohibited_changed_paths(
        ["packages/stocker_core/src/stocker_core/broker/orders.py"]
    ) == ["packages/stocker_core/src/stocker_core/broker/orders.py"]


def test_graph_event_names_map_to_independent_ledger_columns() -> None:
    module = _load()

    assert module.source_event_column("active") == "source_active"
    assert module.source_event_column("newly_decaying") == "newly_decaying"
