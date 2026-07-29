from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).resolve().parents[1] / "run_loop_payoff_phase_path_v1.py"
SPEC = importlib.util.spec_from_file_location("loop_phase_path", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_duration_features_are_conditional_on_survival() -> None:
    durations = np.array([1, 2, 3, 4, 8, 12])
    percentile, hazard, support = MODULE.duration_features(durations, age=3, required_age=8)
    assert percentile == 0.5
    assert hazard == 0.5
    assert support == 4


def test_auc_score_handles_ties() -> None:
    y = np.array([False, False, True, True])
    assert MODULE.auc_score(y, np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert MODULE.auc_score(y, np.ones(4)) == 0.5


def test_fixed_bins_match_contract_boundaries() -> None:
    assert [MODULE.age_bin(value) for value in [1, 2, 4, 7, 13]] == ["1", "2-3", "4-6", "7-12", "13+"]
    assert [MODULE.hazard_bin(value) for value in [0.49, 0.5, 0.79, 0.8]] == ["<0.50", "0.50-0.80", "0.50-0.80", ">=0.80"]


def test_session_ordinal_translates_to_global_state_position() -> None:
    assert MODULE.to_state_position(anchor_start_pos=78, anchor_bar_ordinal=0, session_ordinal=4) == 82
    assert MODULE.to_state_position(anchor_start_pos=103, anchor_bar_ordinal=25, session_ordinal=29) == 107


def test_execution_ordinals_advance_from_timestamp_matched_tape_row() -> None:
    assert MODULE.execution_ordinals(anchor_tape_ordinal=48, entry_step=1, horizon=24) == (49, 72)


def test_source_path_has_no_runtime_or_broker_input() -> None:
    text = RUNNER.read_text().lower()
    forbidden = ("place_order(", "submit_order(", "paper_order(", "broker.connect(")
    assert not any(term in text for term in forbidden)
