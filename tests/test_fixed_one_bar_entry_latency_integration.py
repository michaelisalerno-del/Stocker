from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
RUNNER = WORK / "run_fixed_one_bar_entry_latency_v1.py"
CONTRACT = WORK / "contracts/20260716-fixed-one-bar-entry-latency-v1.json"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fixed_latency_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_one_primary_latency_and_t2_cannot_replace_it() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["t1_fixed_latency"]["primary_latency_bars"] == 1
    assert contract["t1_fixed_latency"]["decision_uses_intervening_bar_contents"] is False
    assert contract["secondary_diagnostics"]["t2_latency_bars"] == 2
    assert contract["secondary_diagnostics"]["t2_may_replace_t1"] is False


def test_exact_t0_t1_had_not_previously_been_tested() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["prior_test_boundary"]["exact_t0_t1_test_previously_existed"] is False
    assert (
        "entry_step ranging from 1 through 24"
        in contract["prior_test_boundary"]["clean_anchor_difference"]
    )


def test_runner_reconstructs_exact_named_and_control_populations() -> None:
    runner = _runner()
    contract, _, _ = runner.load_and_verify_contract()
    named = runner.build_source_population(contract, track="track_a_named_family")
    controls = runner.build_source_population(contract, track="track_b_prior_only")

    assert named.groupby(["period", "loop_id", "orientation"]).size().to_dict() == {
        (2023, "cycle_04", "state_4"): 132,
        (2023, "cycle_07", "state_5"): 722,
        (2025, "cycle_04", "state_4"): 96,
        (2025, "cycle_07", "state_5"): 713,
    }
    assert controls.groupby(["period", "loop_id", "orientation"]).size().to_dict() == {
        (2023, "cycle_04", "state_2"): 8,
        (2023, "cycle_07", "state_6"): 331,
        (2025, "cycle_04", "state_2"): 6,
        (2025, "cycle_07", "state_6"): 296,
    }
    assert set(named["population_role"]) == {"named_candidate"}
    assert set(controls["population_role"]) == {"neutral_control", "negative_control"}


def test_source_clocks_prove_clean_anchor_was_not_exact_post_fill_latency() -> None:
    runner = _runner()
    contract, _, _ = runner.load_and_verify_contract()
    named = runner.build_source_population(contract, track="track_a_named_family")
    available = named.loc[named["period"].eq(2025)].copy()
    anchor = pd.to_datetime(available["anchor_timestamp"], utc=True)
    t0 = pd.to_datetime(available["original_entry_timestamp"], utc=True)

    assert t0.eq(anchor + pd.to_timedelta(5 * available["entry_step"], unit="m")).all()
    assert int(available["entry_step"].gt(1).sum()) == 206
    assert int((anchor + pd.Timedelta(minutes=10)).ne(t0 + pd.Timedelta(minutes=5)).sum()) == 206


def test_2023_rows_remain_missing_when_no_manifest_hash_matches() -> None:
    runner = _runner()
    contract, _, _ = runner.load_and_verify_contract()
    named = runner.build_source_population(contract, track="track_a_named_family")
    expected, scored, restarted = runner.score_population(
        named.loc[named["period"].eq(2023)].head(2), {}, latency_bars=1
    )

    assert expected["t1_status"].eq("provider_2023_hash_pinned_tape_unavailable").all()
    assert scored["t1_status"].eq("provider_2023_hash_pinned_tape_unavailable").all()
    assert scored["t1_net_return_bps"].isna().all()
    assert restarted["restarted_net_return_bps"].isna().all()


def test_safety_contract_prohibits_runtime_changes_and_replacement() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["safety"]["research_only"] is True
    assert contract["safety"]["broker_connection_enabled"] is False
    assert contract["safety"]["live_ordering_enabled"] is False
    assert contract["safety"]["position_management_changed"] is False
    assert contract["safety"]["existing_exit_logic_changed"] is False
    assert contract["safety"]["deployment_enabled"] is False
    assert contract["populations"]["replacement_opportunities_allowed"] is False
    assert contract["populations"]["replacement_loop_selection_allowed"] is False
