from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

WORK = Path(__file__).resolve().parents[1]
RUNNER_PATH = WORK / "run_first_class_profitable_move_detectors_v1.py"
CONTRACT_PATH = WORK / "contracts/20260714-first-class-profitable-move-detectors-v1.json"
SPEC = importlib.util.spec_from_file_location("detector_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_contract_is_research_only_and_not_validation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["safety"]["research_only"] is True
    assert contract["safety"]["live_ordering_enabled"] is False
    assert contract["safety"]["order_placement"] == "disabled"
    assert contract["safety"]["application_code_modification_allowed"] is False
    assert contract["opened_data_status"]["validation_claim_allowed"] is False
    assert contract["evaluation"]["promotion_allowed"] is False


def test_contract_freezes_five_long_detectors_and_excludes_aal_2026() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert [item["id"] for item in contract["detectors"]] == list(runner.DETECTORS)
    assert {item["direction"] for item in contract["detectors"]} == {"long"}
    assert "AAL" in contract["data"]["symbols_2025"]
    assert "AAL" not in contract["data"]["symbols_2026"]


def test_shared_detector_features_have_no_regime_or_loop_inputs() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    feature_text = json.dumps(contract["shared_features"]).lower()
    for forbidden in ("regime", "loop", "cycle", "child", "morph", "route"):
        assert forbidden not in feature_text


def test_detector_masks_replay_frozen_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "prior_scale_bps": [100.0, 100.0, 100.0, 100.0],
            "bar_range_bps": [130.0, 80.0, 80.0, 130.0],
            "signed_body_bps": [-80.0, 10.0, 10.0, -80.0],
            "close_location": [0.10, 0.60, 0.70, 0.10],
            "return_3_bps": [-120.0, -20.0, 5.0, -120.0],
            "prior_12_low": [10.0, 10.0, 10.0, 10.0],
            "low": [9.9, 9.98, 10.0, 9.9],
            "close": [9.91, 10.02, 10.2, 9.91],
            "lower_wick_fraction": [0.1, 0.5, 0.1, 0.1],
            "prior_bar_scale_bps": [100.0, 100.0, 100.0, 100.0],
            "prior_bar_range_bps": [80.0, 80.0, 130.0, 80.0],
            "prior_bar_body_bps": [0.0, 0.0, -80.0, 0.0],
            "prior_bar_close_location": [0.5, 0.5, 0.1, 0.5],
            "prior_bar_midpoint": [10.0, 10.0, 10.0, 10.0],
            "open": [10.0, 10.0, 10.05, 10.0],
            "bar_ordinal": [20, 20, 20, 20],
            "opening_range_low": [10.0, 10.0, 10.0, 10.0],
            "relative_activity_6": [1.05, 1.0, 1.0, 1.20],
        }
    )
    masks = runner.detector_masks(frame)
    assert masks[runner.DETECTORS[0]].tolist() == [True, False, False, True]
    assert masks[runner.DETECTORS[1]].tolist() == [False, True, False, False]
    assert masks[runner.DETECTORS[2]].tolist() == [False, False, True, False]
    assert masks[runner.DETECTORS[4]].tolist() == [False, False, False, True]


def test_cooldown_retains_first_event_then_requires_more_than_24_bars() -> None:
    candidates = pd.DataFrame(
        {
            "period": [2025] * 4,
            "symbol_norm": ["X"] * 4,
            "session_date": ["2025-01-02"] * 4,
            "detector": [runner.DETECTORS[0]] * 4,
            "segment_position": [12, 20, 36, 37],
            "timestamp": pd.date_range("2025-01-02 15:30", periods=4, freq="5min", tz="UTC"),
        }
    )
    selected = runner._cooldown_select(candidates)
    assert selected["segment_position"].tolist() == [12, 37]


def _path_group(*, dual_touch: bool = False, stop_first: bool = False) -> pd.DataFrame:
    rows = []
    for position in range(1, 25):
        high = 100.5
        low = 99.5
        if position == 2:
            if dual_touch:
                high, low = 101.2, 98.8
            elif stop_first:
                high, low = 100.5, 98.8
            else:
                high, low = 101.2, 99.5
        rows.append(
            {
                "segment_position": position,
                "open": 100.0,
                "high": high,
                "low": low,
                "close": 100.0,
            }
        )
    return pd.DataFrame(rows).set_index("segment_position", drop=False)


def _event_row() -> SimpleNamespace:
    return SimpleNamespace(
        event_id="event",
        detector=runner.DETECTORS[0],
        period=2025,
        symbol_norm="X",
        session_date="2025-01-02",
        month="2025-01",
        decision_timestamp=pd.Timestamp("2025-01-02 15:00", tz="UTC"),
        entry_timestamp=pd.Timestamp("2025-01-02 15:05", tz="UTC"),
        segment_index=0,
        decision_segment_position=0,
        entry_open=100.0,
        target_price=101.0,
        invalidation_price=99.0,
        risk_bps=100.0,
    )


def test_score_path_target_first_and_costs() -> None:
    groups = {(2025, "X", "2025-01-02", 0): _path_group()}
    result = runner.score_path(
        _event_row(), groups, id_column="event_id", stop_column="invalidation_price"
    )
    assert result["target_first"] is True
    assert result["hit_step"] == 2
    assert result["dynamic_gross_bps"] == pytest.approx(100.0)
    assert result["dynamic_net_bps"] == pytest.approx(90.0)


def test_score_path_dual_touch_is_conservative_stop() -> None:
    groups = {(2025, "X", "2025-01-02", 0): _path_group(dual_touch=True)}
    result = runner.score_path(
        _event_row(), groups, id_column="event_id", stop_column="invalidation_price"
    )
    assert result["target_first"] is False
    assert result["hit_type"] == "dual_touch_conservative_stop"
    assert result["dynamic_net_bps"] == pytest.approx(-110.0)


def test_score_path_stop_first_is_failure() -> None:
    groups = {(2025, "X", "2025-01-02", 0): _path_group(stop_first=True)}
    result = runner.score_path(
        _event_row(), groups, id_column="event_id", stop_column="invalidation_price"
    )
    assert result["target_first"] is False
    assert result["hit_type"] == "stop_first"


def test_holm_adjustment_is_monotonic_in_sorted_p_values() -> None:
    raw = np.array([0.04, 0.01, 0.03])
    adjusted = runner.holm_adjust(raw)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)


def test_pre_outcome_ledgers_contain_no_outcome_columns() -> None:
    forbidden = {
        "target_first",
        "hit_type",
        "mfe_bps",
        "mae_bps",
        "dynamic_net_bps",
        "fixed_h24_net_bps",
    }
    assert forbidden.isdisjoint(runner.EVENT_COLUMNS)
    assert forbidden.isdisjoint(runner.CONTROL_COLUMNS)
