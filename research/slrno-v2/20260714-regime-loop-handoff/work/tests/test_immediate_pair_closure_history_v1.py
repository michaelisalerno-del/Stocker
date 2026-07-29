from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_research.pair_closure_history_v1 import (  # noqa: E402
    HierarchicalBinaryFrequencyModel,
    benjamini_hochberg,
    binary_log_loss,
    brier_score,
    build_pair_closure_population,
    expanding_month_predictions,
    roc_auc,
    safety_flags,
)


def _panel(states: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows = len(states)
    starts = pd.date_range("2024-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "session": "2024-01-02",
            "segment_id": "AAA::2024-01-02::segment_00",
            "segment_end_reason": ["continued"] * (rows - 1) + ["scheduled_session_end"],
            "bar_ordinal": np.arange(rows),
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "clock_phase": "morning",
            "source_artifact": "/local/source/symbol=AAA/timeframe=5m/data.parquet",
            "source_hash": "source-hash",
            "data_snapshot_hash": "snapshot-hash",
        }
    )
    probabilities = np.full((rows, 8), 0.01)
    probabilities[np.arange(rows), states] = 0.93
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return frame, probabilities


def _population(states: np.ndarray) -> pd.DataFrame:
    panel, probabilities = _panel(states)
    return build_pair_closure_population(
        panel,
        semantic_states=states,
        state_probabilities=probabilities,
        posterior_entropy=np.full(len(states), 0.2),
        departure_probability=np.full(len(states), 0.1),
        representation="TEST_CAUSAL",
    )


def test_pair_closure_target_and_terminal_membership() -> None:
    states = np.asarray([0, 0, 1, 1, 0, 2, 3, 2, 4, 5, 5, 6])
    population = _population(states)

    first = population.iloc[0]
    assert first["previous_state_1"] == 0
    assert first["current_state"] == 1
    assert first["next_state"] == 0
    assert first["target_pair_closure"] == 1
    assert first["loop_orientation"] == "0->1->0"
    assert first["loop_pair_id"] == "loop_p_0-1-0"
    assert first["decision_timestamp"] < first["target_available_timestamp"]

    terminal = population.iloc[-1]
    assert not terminal["target_available"]
    assert terminal["target_pair_closure"] == -1
    assert terminal["censor_reason"] == "RIGHT_CENSORED_SESSION_END"
    assert len(population) == len(np.flatnonzero(states[1:] != states[:-1]))


def test_future_mutation_does_not_change_decision_features() -> None:
    original = np.asarray([0, 0, 1, 1, 0, 2, 3, 2, 4])
    mutated = np.asarray([0, 0, 1, 1, 3, 2, 3, 2, 4])
    before = _population(original).iloc[0]
    after = _population(mutated).iloc[0]
    causal_fields = [
        "decision_id",
        "decision_timestamp",
        "current_state",
        "previous_state_1",
        "previous_state_2",
        "previous_duration_1",
        "posterior_top_two_margin",
        "clock_phase",
    ]
    assert before[causal_fields].to_dict() == after[causal_fields].to_dict()
    assert before["target_pair_closure"] != after["target_pair_closure"]


def test_history_and_previous_loop_are_strictly_trailing() -> None:
    states = np.asarray([0, 1, 0, 2, 0, 1, 0, 3])
    population = _population(states)
    decision = population.loc[population["current_state"].eq(1)].iloc[-1]

    assert decision["previous_state_1"] == 0
    assert decision["previous_state_2"] == 2
    assert decision["previous_state_3"] == 0
    assert decision["previous_state_4"] == 1
    assert decision["previous_loop_pair"] == "loop_p_0-2-0"
    assert decision["previous_loop_recency_runs"] == 1
    assert decision["target_pair_closure"] == 1


def test_frequency_model_shrinks_and_falls_back_to_parent() -> None:
    train = pd.DataFrame(
        {
            "current_state": [1, 1, 1, 2],
            "previous_state_1": [0, 0, 3, 0],
            "target_pair_closure": [1, 1, 0, 0],
        }
    )
    model = HierarchicalBinaryFrequencyModel.fit(
        train,
        levels=(("current_state",), ("previous_state_1", "current_state")),
        tau=4.0,
    )
    score = pd.DataFrame(
        {
            "current_state": [1, 1, 7],
            "previous_state_1": [0, 6, 6],
        }
    )
    probability = model.predict(score)

    assert probability[0] > probability[1]
    assert probability[1] != model.global_probability
    assert probability[2] == model.global_probability
    assert np.all((probability > 0.0) & (probability < 1.0))


def test_expanding_folds_use_only_strictly_earlier_months() -> None:
    rows = []
    for month in range(1, 6):
        for index in range(6):
            rows.append(
                {
                    "decision_id": f"d-{month}-{index}",
                    "representation": "TEST",
                    "symbol": "AAA",
                    "session": f"2024-{month:02d}-01",
                    "decision_timestamp": pd.Timestamp(f"2024-{month:02d}-01 15:00", tz="UTC"),
                    "target_available": True,
                    "target_pair_closure": index % 2,
                    "current_state": index % 2,
                    "previous_state_1": (index + 1) % 3,
                    "previous_state_2": 8,
                    "previous_state_3": 8,
                    "previous_state_4": 8,
                    "previous_duration_1_bucket": 1,
                    "previous_duration_2_bucket": 1,
                    "previous_loop_pair": "NO_PREVIOUS_CLOSURE",
                    "previous_loop_recency_bucket": -1,
                    "posterior_margin_bucket": 1,
                    "departure_probability_bucket": 1,
                    "clock_phase": "morning",
                }
            )
    population = pd.DataFrame(rows)
    predictions = expanding_month_predictions(
        population,
        model_names=("M0_GLOBAL", "M2_IMMEDIATE_PAIR"),
        minimum_train_months=3,
    )

    april = predictions.loc[predictions["score_month"].eq("2024-04")]
    may = predictions.loc[predictions["score_month"].eq("2024-05")]
    assert april["training_rows"].eq(18).all()
    assert may["training_rows"].eq(24).all()
    assert set(predictions["score_month"]) == {"2024-04", "2024-05"}


def test_known_binary_metrics_and_bh() -> None:
    truth = np.asarray([0, 0, 1, 1])
    perfect = np.asarray([0.01, 0.02, 0.98, 0.99])
    assert binary_log_loss(truth, perfect) < 0.03
    assert brier_score(truth, perfect) < 0.001
    assert roc_auc(truth, perfect) == 1.0

    q_values = benjamini_hochberg([0.01, 0.04, 0.03])
    np.testing.assert_allclose(q_values, [0.03, 0.04, 0.04])


def test_safety_flags_are_fail_closed() -> None:
    flags = safety_flags()
    assert flags["research_only"] is True
    assert flags["execution_enabled"] is False
    assert flags["order_placement"] == "disabled"
    assert flags["live_ordering_enabled"] is False
    assert flags["strategy_promotion"] is False
    assert flags["promotable"] is False
