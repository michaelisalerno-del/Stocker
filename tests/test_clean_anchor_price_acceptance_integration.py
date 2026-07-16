from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd

from stocker_research.clean_anchor_price_acceptance.variants import build_variant_decisions

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
RUNNER = WORK / "run_clean_anchor_price_acceptance_v1.py"
CONTRACT = WORK / "contracts/20260716-clean-anchor-price-acceptance-v1.json"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("clean_anchor_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_registers_only_one_checkpoint_and_primary_rule() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["checkpoint"]["registered_checkpoint_count"] == 1
    assert contract["checkpoint"]["bar_duration_minutes"] == 5
    assert contract["price_acceptance"]["primary_pass_rule"] == (
        "signed_close_return_bps > 0 and favourable_excursion_bps > adverse_excursion_bps"
    )
    assert contract["variants"]["primary_variant"] == ("D_anchor_veto_plus_price_acceptance")


def test_contract_freezes_named_families_without_replacement() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["populations"]["primary_named"] == [
        {"loop_id": "cycle_04", "orientation": "state_4", "cycle": "2->4->2"},
        {"loop_id": "cycle_07", "orientation": "state_5", "cycle": "5->6->5"},
    ]
    assert contract["populations"]["failed_named_family_replacement_allowed"] is False


def test_contract_keeps_range_variant_unavailable_without_substitute() -> None:
    contract = json.loads(CONTRACT.read_text())

    assert contract["range_permission"]["available"] is False
    assert contract["inputs"]["range_prediction_ledger"]["sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_exact_frozen_named_population_and_static_veto_reconstruct() -> None:
    runner = _runner()
    contract, _, _ = runner.load_and_verify_contract()
    source = runner.build_source_population(contract, track="track_a_named_family")

    assert len(source) == 1663
    assert source.groupby(["period", "loop_id", "orientation"]).size().to_dict() == {
        (2023, "cycle_04", "state_4"): 132,
        (2023, "cycle_07", "state_5"): 722,
        (2025, "cycle_04", "state_4"): 96,
        (2025, "cycle_07", "state_5"): 713,
    }
    expected = source["static_anchor_veto_score"].gt(1.0)
    assert expected.eq(source["static_anchor_veto_pass"]).all()
    assert {"original_entry_price", "original_exit_price"}.issubset(source.columns)
    assert "entry_price" not in source.columns


def test_frozen_control_loader_excludes_generalized_track_b_surface() -> None:
    runner = _runner()
    contract, _, _ = runner.load_and_verify_contract()
    controls = runner.build_source_population(contract, track="track_b_prior_only")

    assert len(controls) == 641
    assert set(controls["population_role"]) == {"negative_control", "neutral_control"}
    assert set(zip(controls["loop_id"], controls["orientation"], strict=True)) == {
        ("cycle_04", "state_2"),
        ("cycle_07", "state_6"),
    }


def _synthetic_source(
    payoffs: list[float], anchors: list[bool], acceptance: list[bool]
) -> pd.DataFrame:
    count = len(payoffs)
    return pd.DataFrame(
        {
            "opportunity_id": [f"opp-{index}" for index in range(count)],
            "source_available": [True] * count,
            "availability_status": ["available"] * count,
            "static_anchor_veto_pass": anchors,
            "price_acceptance_pass": acceptance,
            "range_permission_available": [False] * count,
            "range_permission_pass": [False] * count,
            "entry_timestamp": [pd.Timestamp("2025-01-02 14:40:00+00:00")] * count,
            "original_terminal_timestamp": [pd.Timestamp("2025-01-02 16:35:00+00:00")] * count,
            "net_payoff_bps": payoffs,
        }
    )


def test_synthetic_clean_anchor_and_acceptance_retains_positive_case() -> None:
    decisions = build_variant_decisions(
        _synthetic_source([-30.0, -20.0, 40.0, 50.0], [False, False, True, True], [True] * 4)
    )
    d = decisions.loc[decisions["variant"].eq("D_anchor_veto_plus_price_acceptance")]

    assert d["policy_net_payoff_bps"].sum() == 90.0
    assert d["admitted"].sum() == 2


def test_synthetic_null_does_not_manufacture_interaction() -> None:
    decisions = build_variant_decisions(
        _synthetic_source([10.0, 10.0, 10.0, 10.0], [False, False, True, True], [False, True] * 2)
    )
    d = decisions.loc[
        decisions["variant"].eq("D_anchor_veto_plus_price_acceptance"),
        "policy_net_payoff_bps",
    ].sum()
    a = decisions.loc[decisions["variant"].eq("A_same_clock_base"), "policy_net_payoff_bps"].sum()

    assert d < a


def test_contaminated_anchor_rejects_even_favourable_first_bar() -> None:
    decisions = build_variant_decisions(_synthetic_source([100.0], [False], [True]))
    d = decisions.loc[decisions["variant"].eq("D_anchor_veto_plus_price_acceptance")].iloc[0]

    assert bool(d["admitted"]) is False
    assert d["reason_codes"] == "static_anchor_veto_pass_failed"


def test_safety_contract_disables_every_runtime_surface() -> None:
    safety = json.loads(CONTRACT.read_text())["safety"]

    assert safety["research_only"] is True
    assert safety["broker_connection_enabled"] is False
    assert safety["live_ordering_enabled"] is False
    assert safety["position_management_changed"] is False
    assert safety["existing_exit_logic_changed"] is False
    assert safety["deployment_enabled"] is False
