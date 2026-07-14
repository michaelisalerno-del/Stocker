from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/20260713-loop-payoff-phase-path-v2.json"


def test_corrected_population_is_the_parent_score_surface() -> None:
    contract = json.loads(CONTRACT.read_text())
    population = contract["population"]
    assert population["provider_sessions_each_period"] == 250
    assert population["warmup_completed_sessions"] == 60
    assert population["score_sessions_each_period"] == 190
    assert population["parent_expected_candidate_counts"] == {
        "2023|cycle_04|state4": 132,
        "2023|cycle_07|state5": 722,
        "2025|cycle_04|state4": 96,
        "2025|cycle_07|state5": 713,
    }


def test_v2_cannot_claim_validation_or_modify_app() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["sealed_data_status"]["genuinely_unseen_sessions_available"] is False
    assert contract["strategy_promotion_allowed"] is False
    assert contract["application_code_modification_allowed"] is False
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
