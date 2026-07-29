from __future__ import annotations

import math

import pandas as pd

from stocker_research.sequential_loop_competitor_veto import (
    CensusConfig,
    RollingTrainingOnlyCensus,
    TrainingOnlyCensus,
)


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": 2023,
                "session_date": "2023-01-03",
                "symbol_norm": "A",
                "loop_id": "cycle_04",
                "orientation": "state_4",
                "current_state": 4,
                "bar_ordinal": 5,
                "loop_occurs": True,
                "first_transition_state": 2,
                "first_transition_lag": 3,
                "second_transition_state": 4,
                "second_transition_lag": 7,
            },
            {
                "period": 2025,
                "session_date": "2025-01-02",
                "symbol_norm": "A",
                "loop_id": "cycle_04",
                "orientation": "state_4",
                "current_state": 4,
                "bar_ordinal": 5,
                "loop_occurs": True,
                "first_transition_state": 2,
                "first_transition_lag": 2,
                "second_transition_state": 4,
                "second_transition_lag": 5,
            },
            {
                "period": 2025,
                "session_date": "2025-01-03",
                "symbol_norm": "B",
                "loop_id": "cycle_04",
                "orientation": "state_4",
                "current_state": 4,
                "bar_ordinal": 40,
                "loop_occurs": False,
                "first_transition_state": 6,
                "first_transition_lag": 4,
                "second_transition_state": pd.NA,
                "second_transition_lag": pd.NA,
            },
            {
                "period": 2025,
                "session_date": "2025-01-06",
                "symbol_norm": "C",
                "loop_id": "cycle_04",
                "orientation": "state_4",
                "current_state": 4,
                "bar_ordinal": 5,
                "loop_occurs": True,
                "first_transition_state": 2,
                "first_transition_lag": 1,
                "second_transition_state": 4,
                "second_transition_lag": 2,
            },
        ]
    )


def test_census_uses_only_strictly_prior_sessions_and_resets_periods() -> None:
    census = TrainingOnlyCensus.from_examples(
        _examples(), period=2025, score_session="2025-01-06", config=CensusConfig()
    )

    assert census.training_sessions == ("2025-01-02", "2025-01-03")
    assert census.training_rows == 2


def test_appending_future_rows_does_not_change_clock_or_timing_likelihood() -> None:
    base = _examples().iloc[:3].copy()
    appended = _examples().copy()
    before = TrainingOnlyCensus.from_examples(
        base, period=2025, score_session="2025-01-06", config=CensusConfig()
    )
    after = TrainingOnlyCensus.from_examples(
        appended, period=2025, score_session="2025-01-06", config=CensusConfig()
    )

    assert math.isclose(
        before.clock_lift("cycle_04", "state_4", 4, 5),
        after.clock_lift("cycle_04", "state_4", 4, 5),
    )
    assert math.isclose(
        before.timing_likelihood("cycle_04", "state_4", 4, (2,), 3),
        after.timing_likelihood("cycle_04", "state_4", 4, (2,), 3),
    )


def test_unseen_possible_transition_has_smoothed_nonzero_likelihood() -> None:
    census = TrainingOnlyCensus.from_examples(
        _examples(), period=2025, score_session="2025-01-06", config=CensusConfig()
    )

    assert census.timing_likelihood("cycle_99", "state_4", 4, (7,), 3) >= 0.05


def test_leave_one_stock_out_rebuilds_census_inputs() -> None:
    full = TrainingOnlyCensus.from_examples(
        _examples(), period=2025, score_session="2025-01-06", config=CensusConfig()
    )
    excluded = TrainingOnlyCensus.from_examples(
        _examples(),
        period=2025,
        score_session="2025-01-06",
        config=CensusConfig(),
        excluded_stocks={"A"},
    )

    assert full.training_rows == 2
    assert excluded.training_rows == 1


def test_rolling_census_matches_batch_equations_at_each_causal_cutoff() -> None:
    rolling = RollingTrainingOnlyCensus(_examples(), period=2025, config=CensusConfig())
    for session in ("2025-01-03", "2025-01-06", "2025-01-07"):
        rolling.advance_before(session)
        batch = TrainingOnlyCensus.from_examples(
            _examples(), period=2025, score_session=session, config=CensusConfig()
        )
        assert tuple(rolling.training_sessions) == batch.training_sessions
        assert rolling.training_rows == batch.training_rows
        assert math.isclose(
            rolling.clock_lift("cycle_04", "state_4", 4, 5),
            batch.clock_lift("cycle_04", "state_4", 4, 5),
        )
        assert math.isclose(
            rolling.timing_likelihood("cycle_04", "state_4", 4, (2,), 3),
            batch.timing_likelihood("cycle_04", "state_4", 4, (2,), 3),
        )
