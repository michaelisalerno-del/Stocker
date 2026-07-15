from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "research/slrno-v2/20260714-regime-loop-handoff/work"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_pins_named_families_checkpoints_and_no_replacements() -> None:
    contract = json.loads(
        (WORK / "contracts/20260715-sequential-loop-competitor-veto-v1.json").read_text()
    )

    assert contract["named_families"]["targets"] == [
        {"loop_id": "cycle_04", "orientation": "state_4", "cycle": "2->4->2"},
        {"loop_id": "cycle_07", "orientation": "state_5", "cycle": "5->6->5"},
    ]
    assert contract["checkpoints"]["fixed_bars_after_anchor"] == [1, 2, 3, 6]
    assert contract["population"]["replacement_opportunities_allowed"] is False
    assert contract["named_families"]["replacement_target_selection_allowed"] is False


def test_independent_auditor_rebuilds_prefix_compatibility_without_future_identity() -> None:
    auditor = _load(WORK / "audit_sequential_loop_competitor_veto_v1.py", "competitor_auditor")

    assert auditor._independent_status("4->2->4", 4, ()) == "compatible"
    assert auditor._independent_status("4->6->4", 4, (2,)) == "impossible"
    assert auditor._independent_status("4->2->4", 4, (2, 4)) == "completed"


def test_exact_rerun_verifier_detects_one_byte_machine_artifact_drift(tmp_path: Path) -> None:
    runner = _load(WORK / "run_sequential_loop_competitor_veto_v1.py", "competitor_runner")
    primary = tmp_path / "primary"
    exact = tmp_path / "exact"
    primary.mkdir()
    exact.mkdir()
    (primary / "metrics.csv").write_bytes(b"a,b\n1,2\n")
    (exact / "metrics.csv").write_bytes(b"a,b\n1,2\n")

    assert runner.verify_exact_rerun(exact, primary)["byte_identical"] is True
    (exact / "metrics.csv").write_bytes(b"a,b\n1,3\n")
    assert runner.verify_exact_rerun(exact, primary)["byte_identical"] is False


def test_research_files_do_not_import_execution_or_broker_modules() -> None:
    paths = [
        WORK / "run_sequential_loop_competitor_veto_v1.py",
        WORK / "audit_sequential_loop_competitor_veto_v1.py",
        *sorted(
            (
                REPO / "packages/stocker_research/src/stocker_research/"
                "sequential_loop_competitor_veto"
            ).glob("*.py")
        ),
    ]
    forbidden = (
        "stocker_execution",
        "ig_integration",
        "place_order",
        "paper_trading",
        "demo_trading",
    )

    for path in paths:
        tree = ast.parse(path.read_text())
        imports = [
            alias.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        ]
        assert not any(token in imported for token in forbidden for imported in imports), path


def test_reconstructed_2023_checkpoint_at_terminal_is_unavailable() -> None:
    runner = _load(WORK / "run_sequential_loop_competitor_veto_v1.py", "terminal_clock_runner")
    timestamp = "2023-04-03T16:25:00Z"
    opportunity = SimpleNamespace(
        status="filled",
        direction=1,
        period=2023,
        terminal_timestamp=timestamp,
        symbol_norm="A",
        session_date="2023-04-03",
    )
    checkpoint = SimpleNamespace(
        checkpoint_timestamp=timestamp,
        checkpoint_type="first_completed_transition",
    )

    result = runner._outcome_for_checkpoint(
        opportunity,
        checkpoint,
        {},
        {},
        {},
    )

    assert result["outcome_status"] == "too_late"
    assert result["constant_terminal_net_bps"] != 0.0


def test_competitor_census_keeps_missing_target_separate_from_losses() -> None:
    runner = _load(WORK / "run_sequential_loop_competitor_veto_v1.py", "census_runner")
    anchor_sets = pd.DataFrame(
        {
            "opportunity_id": ["winner", "missing"],
            "loop_id": ["cycle_06", "cycle_06"],
            "payoff_class": ["bad", "bad"],
            "initial_posterior_probability": [0.3, 0.3],
        }
    )
    eliminations = pd.DataFrame(
        {
            "opportunity_id": ["winner", "missing"],
            "loop_id": ["cycle_06", "cycle_06"],
            "checkpoint_type": ["first_completed_transition"] * 2,
            "checkpoint_timestamp": pd.to_datetime(
                ["2025-01-02T15:00:00Z", "2025-01-03T15:00:00Z"], utc=True
            ),
            "bars_consumed": [2, 2],
        }
    )
    opportunities = pd.DataFrame(
        {
            "opportunity_id": ["winner", "missing"],
            "population_role": ["named_target", "named_target"],
            "top_loop": ["cycle_04", "cycle_04"],
            "anchor_state": [4, 4],
            "bar_ordinal": [10, 10],
            "original_net_payoff_bps": [25.0, float("nan")],
        }
    )

    row = runner.build_competitor_census(anchor_sets, eliminations, opportunities).iloc[0]

    assert row["compatible_opportunities"] == 2
    assert row["frequency_profitable_target"] == 1
    assert row["frequency_losing_target"] == 0
    assert row["frequency_missing_target"] == 1
