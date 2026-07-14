from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
RUNNER_PATH = WORK / "run_long_short_neutral_detector_v1.py"
SPEC = importlib.util.spec_from_file_location("long_short_neutral_v1", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def synthetic_session() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-02 14:30:00+00:00", periods=78, freq="5min")
    close = 100.0 + np.linspace(0.0, 0.77, 78)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.01,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": np.arange(1000.0, 1078.0),
            "period": 2025,
            "symbol_norm": "TEST",
            "session_date": "2025-01-02",
            "month": "2025-01",
            "bar_ordinal": np.arange(78, dtype=np.int16),
        }
    )
    previous = frame["close"].shift(1).fillna(frame["open"])
    true_range = np.maximum.reduce(
        [
            (frame["high"] - frame["low"]).to_numpy(),
            (frame["high"] - previous).abs().to_numpy(),
            (frame["low"] - previous).abs().to_numpy(),
        ]
    )
    frame["true_range_bps"] = 10000.0 * true_range / previous
    frame["range_bps"] = 10000.0 * (frame["high"] - frame["low"]) / frame["open"]
    return frame


def event(barrier_bps: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(
        event_id="lsn|2025|TEST|2025-01-02|12",
        period=2025,
        symbol_norm="TEST",
        session_date="2025-01-02",
        decision_ordinal=12,
        barrier_bps=barrier_bps,
    )


def path_session(kind: str) -> pd.DataFrame:
    frame = synthetic_session()
    frame.loc[13:, ["open", "high", "low", "close"]] = [100.0, 100.2, 99.8, 100.0]
    if kind == "long":
        frame.loc[14, "high"] = 101.1
    elif kind == "short":
        frame.loc[14, "low"] = 98.9
    elif kind == "ambiguous":
        frame.loc[13, ["high", "low"]] = [101.1, 98.9]
    elif kind != "neutral":
        raise ValueError(kind)
    return frame


def test_contract_is_research_only() -> None:
    contract = runner.load_contract()
    assert contract["safety"]["research_only"] is True
    assert contract["safety"]["live_ordering_enabled"] is False
    assert contract["safety"]["order_placement"] == "disabled"
    assert contract["population"]["no_loop_filter"] is True


def test_event_features_are_outcome_free() -> None:
    frame = synthetic_session()
    before = runner.event_features(2025, "TEST", "2025-01-02", frame, 12)
    changed = frame.copy()
    changed.loc[13:, ["open", "high", "low", "close", "volume"]] *= 7.0
    after = runner.event_features(2025, "TEST", "2025-01-02", changed, 12)
    assert before == after
    assert before is not None
    forbidden = [
        column
        for column in before
        if any(token in column.lower() for token in runner.FORBIDDEN_PRE_OUTCOME_TOKENS)
    ]
    assert forbidden == []


def test_fixed_decision_requires_contiguous_prior_bars() -> None:
    frame = synthetic_session().loc[lambda value: value["bar_ordinal"].ne(7)]
    assert runner.event_features(2025, "TEST", "2025-01-02", frame, 12) is None


def test_session_features_require_all_elapsed_bars() -> None:
    frame = synthetic_session().loc[lambda value: value["bar_ordinal"].ne(18)]
    assert runner.event_features(2025, "TEST", "2025-01-02", frame, 36) is None


def test_long_path_label() -> None:
    result = runner.outcome_for_event(event(), path_session("long"))
    assert result["actual_class"] == "long"
    assert result["first_touch_step"] == 2
    assert result["long_net_bps_5"] == 90.0
    assert result["short_net_bps_5"] == -110.0


def test_short_path_label() -> None:
    result = runner.outcome_for_event(event(), path_session("short"))
    assert result["actual_class"] == "short"
    assert result["short_net_bps_5"] == 90.0
    assert result["long_net_bps_5"] == -110.0


def test_no_touch_is_neutral_with_mark_to_market_payoff() -> None:
    result = runner.outcome_for_event(event(), path_session("neutral"))
    assert result["actual_class"] == "neutral"
    assert result["neutral_reason"] == "no_touch"
    assert result["long_net_bps_5"] == -10.0
    assert result["short_net_bps_5"] == -10.0


def test_same_bar_dual_touch_is_neutral_and_conservative() -> None:
    result = runner.outcome_for_event(event(), path_session("ambiguous"))
    assert result["actual_class"] == "neutral"
    assert result["neutral_reason"] == "same_bar_dual_touch"
    assert result["long_net_bps_5"] == -110.0
    assert result["short_net_bps_5"] == -110.0


def test_missing_path_is_retained_as_unscored() -> None:
    frame = path_session("neutral").loc[lambda value: value["bar_ordinal"].ne(20)]
    result = runner.outcome_for_event(event(), frame)
    assert result["score_status"] == "missing_exact_24_bar_path"


def test_cost_aware_state_abstains_or_selects_direction() -> None:
    assert runner.economic_state(0.40, 0.35, 100.0)[0] == "neutral"
    assert runner.economic_state(0.60, 0.20, 100.0)[0] == "long"
    assert runner.economic_state(0.20, 0.60, 100.0)[0] == "short"


def test_multinomial_pipeline_outputs_three_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "decision_clock": ["clock_12"] * 9,
            "barrier_bps": np.arange(9, dtype=float) + 40.0,
            "actual_class": ["long", "neutral", "short"] * 3,
        }
    )
    pipeline = runner.build_pipeline(("barrier_bps",), ("decision_clock",))
    pipeline.fit(frame[["barrier_bps", "decision_clock"]], frame["actual_class"])
    probability = pipeline.predict_proba(frame[["barrier_bps", "decision_clock"]])
    assert probability.shape == (9, 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_block_draws_preserve_session_count() -> None:
    sessions = [f"2025-01-{day:02d}" for day in range(1, 11)]
    counts = runner._block_draw_counts(sessions, draws=25, block=5, seed=7)
    assert counts.shape == (25, 10)
    assert np.all(counts.sum(axis=1) == 10)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.01, 0.04, 0.02]
    adjusted = runner._holm_adjust(raw)
    assert adjusted == [0.03, 0.04, 0.04]
