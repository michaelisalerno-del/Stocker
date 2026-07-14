from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

WORK = Path(__file__).resolve().parents[1]
RUNNER_PATH = WORK / "run_selective_payoff_equations_v1.py"
SPEC = importlib.util.spec_from_file_location("selective_equation_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
research = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(research)


def tape(*, high: float = 100.2, low: float = 99.8, close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-01 13:30", periods=24, freq="5min", tz="UTC"),
            "open": np.full(24, 100.0),
            "high": np.full(24, high),
            "low": np.full(24, low),
            "close": np.full(24, close),
        }
    )


def outcome_row() -> SimpleNamespace:
    return SimpleNamespace(
        direction=1,
        entry_ordinal=0,
        entry_open=100.0,
        stop_price=99.0,
        target_price=101.0,
        base_risk_bps=100.0,
    )


def score(frame: pd.DataFrame) -> dict[str, object]:
    return research.score_path(
        outcome_row(),
        frame,
        entry_ordinal_field="entry_ordinal",
        entry_open_field="entry_open",
        stop_field="stop_price",
        target_field="target_price",
        risk_field="base_risk_bps",
    )


def test_contract_is_research_only_and_opened_development() -> None:
    contract = research.load_contract()
    assert contract["safety"]["research_only"] is True
    assert contract["safety"]["live_ordering_enabled"] is False
    assert contract["safety"]["order_placement"] == "disabled"
    assert contract["opened_data_status"]["opened"] is True
    assert contract["opened_data_status"]["sealed_validation_available"] is False


def test_full_loop_vector_is_present_only_in_loop_equations() -> None:
    assert len(research.LOOP_COLUMNS) == 20
    prediction_numeric, _ = research.DIRECT_MODELS["prediction_only"]
    context_numeric, _ = research.DIRECT_MODELS["context_only"]
    combined_numeric, _ = research.DIRECT_MODELS["context_plus_loop_mixture"]
    assert set(research.LOOP_COLUMNS).issubset(prediction_numeric)
    assert set(research.LOOP_COLUMNS).isdisjoint(context_numeric)
    assert set(research.LOOP_COLUMNS).issubset(combined_numeric)


def test_direct_identity_column_is_unique_in_prediction_contract() -> None:
    assert research.prediction_identity_columns("event_id") == ["event_id"]
    assert research.prediction_identity_columns("snapshot_id") == ["snapshot_id", "event_id"]


def test_feature_contract_excludes_future_and_outcome_fields() -> None:
    features = set(research.CONTEXT_NUMERIC + research.CONTEXT_CATEGORICAL)
    features.update(research.LOOP_NUMERIC)
    features.update(research.SEQUENTIAL_NUMERIC)
    forbidden = ("future", "outcome", "target_first", "gross_return", "exit_price")
    assert not any(token in feature.lower() for feature in features for token in forbidden)


def test_pre_outcome_guard_rejects_outcome_columns() -> None:
    research.assert_pre_outcome_columns(pd.DataFrame({"event_id": ["a"], "feature": [1.0]}))
    research.assert_pre_outcome_columns(
        pd.DataFrame({"snapshot_id": ["a"], "running_mfe_bps": [1.0]}),
        allow_completed_path_excursions=True,
    )
    with pytest.raises(AssertionError):
        research.assert_pre_outcome_columns(
            pd.DataFrame({"event_id": ["a"], "future_return_bps": [1.0]})
        )


def test_cooldown_keeps_nonoverlapping_decisions() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "symbol_norm": ["AAL"] * 4,
            "session_date": ["2024-07-01"] * 4,
            "decision_ordinal": [1, 10, 25, 49],
        }
    )
    kept = research.apply_cooldown(frame, 24)
    assert kept["event_id"].tolist() == ["a", "c", "d"]


def test_target_first_path_and_cost() -> None:
    frame = tape()
    frame.loc[2, "high"] = 101.1
    result = score(frame)
    assert result["target_first"] is True
    assert result["hit_type"] == "target_first"
    assert result["gross_bps"] == pytest.approx(100.0)
    assert result["net_bps"] == pytest.approx(90.0)


def test_stop_first_path_is_failure() -> None:
    frame = tape()
    frame.loc[1, "low"] = 98.9
    result = score(frame)
    assert result["target_first"] is False
    assert result["hit_type"] == "stop_first"
    assert result["net_bps"] == pytest.approx(-110.0)


def test_dual_touch_is_conservative_stop() -> None:
    frame = tape()
    frame.loc[3, ["high", "low"]] = [101.1, 98.9]
    result = score(frame)
    assert result["target_first"] is False
    assert result["hit_type"] == "dual_touch_conservative_stop"
    assert result["net_bps"] == pytest.approx(-110.0)


def test_stop_gap_uses_worse_open() -> None:
    frame = tape()
    frame.loc[1, ["open", "high", "low", "close"]] = [98.0, 98.5, 97.5, 98.2]
    result = score(frame)
    assert result["hit_type"] == "stop_gap_or_open_first"
    assert result["gross_bps"] == pytest.approx(-200.0)
    assert result["net_bps"] == pytest.approx(-210.0)


def test_calibration_requires_prior_support_and_uses_jeffreys_bound() -> None:
    insufficient = pd.DataFrame(
        {"raw_probability": np.full(199, 0.6), "target_first": np.ones(199, dtype=bool)}
    )
    mean, lower, support = research.calibration_values(
        0.6, insufficient, nearest_rows=500, minimum_support=200
    )
    assert np.isnan(mean)
    assert np.isnan(lower)
    assert support == 199
    history = pd.DataFrame(
        {
            "raw_probability": np.linspace(0.4, 0.8, 300),
            "target_first": np.array([True] * 180 + [False] * 120),
        }
    )
    mean, lower, support = research.calibration_values(
        0.6, history, nearest_rows=300, minimum_support=200
    )
    assert support == 300
    assert mean == pytest.approx((180.5) / 301.0)
    assert 0.0 < lower < mean


def test_holm_requires_positive_lower_bound() -> None:
    frame = pd.DataFrame(
        {
            "p_one_sided": [0.01, 0.02],
            "ci_lower": [0.1, -0.1],
        }
    )
    adjusted = research.holm(frame)
    assert adjusted.loc[0, "holm_adjusted_p"] == pytest.approx(0.02)
    assert bool(adjusted.loc[0, "passes_holm_0_05"])
    assert not bool(adjusted.loc[1, "passes_holm_0_05"])
