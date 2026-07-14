from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_regime_loop_linkage_ideas_v3.py"
SPEC = importlib.util.spec_from_file_location("regime_loop_linkage_v3", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v3_binds_immutable_v2_semantics_and_adapter() -> None:
    assert MODULE.sha256(MODULE.V2_CONTRACT) == MODULE.V2_CONTRACT_SHA256
    assert MODULE.sha256(MODULE.V2_RUNNER) == MODULE.V2_RUNNER_SHA256


def test_v3_changes_no_statistical_rule() -> None:
    contract = MODULE.load_contract()
    amendment = contract["v3_amendment"]
    assert amendment["only_code_amendment"][
        "changes_probabilities_metrics_gates_or_slice_qualification"
    ] is False
    assert contract["global_variant_gate"] == MODULE.ADAPTER.load_contract()[
        "global_variant_gate"
    ]


def test_qualified_slice_serialization_handles_empty_and_nonempty() -> None:
    columns = [
        "cycle_id",
        "current_state",
        "entry_clock_quartile",
        "qualified_development_attraction_slice",
    ]
    assert MODULE.qualified_slice_ids(pd.DataFrame(columns=columns)) == []
    frame = pd.DataFrame(
        [
            {"cycle_id": "cycle_01", "current_state": 2, "entry_clock_quartile": 3, "qualified_development_attraction_slice": True},
            {"cycle_id": "cycle_02", "current_state": 4, "entry_clock_quartile": 1, "qualified_development_attraction_slice": False},
        ]
    )
    assert MODULE.qualified_slice_ids(frame) == ["cycle_01@state_2@clock_3"]


def test_v3_safety_boundary_is_exact() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["decision"]["later_period_scoring"] is False
    assert contract["decision"]["named_loop_good_or_high_promotion_permitted"] is False

