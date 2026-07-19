from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.right_censored_duration_v2 import (
    DurationFitConfig,
    RunEndingStatus,
    classify_training_run_endings,
    estimate_right_censored_durations,
)


def _session(
    states: list[int],
    *,
    complete: bool = True,
    ordinals: list[int] | None = None,
) -> pd.DataFrame:
    if ordinals is None:
        ordinals = list(range(len(states)))
    start = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    segment = np.cumsum(np.r_[0, np.diff(ordinals) != 1])
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "session": "2024-01-02",
            "segment_id": [f"TEST::2024-01-02::{value}" for value in segment],
            "segment_index": segment,
            "bar_ordinal": ordinals,
            "bar_start_timestamp": [start + pd.Timedelta(minutes=5 * value) for value in ordinals],
            "state": states,
            "session_source_complete": complete,
            "expected_session_bars": max(ordinals) + 1 if complete else max(ordinals) + 2,
        }
    )


def test_observed_transition_is_exact_exit_and_terminal_is_censored() -> None:
    ledger = classify_training_run_endings(_session([1, 1, 2, 2]))

    assert ledger["ending_status"].tolist() == [
        RunEndingStatus.OBSERVED_STATE_EXIT.value,
        RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
    ]
    assert ledger["duration"].tolist() == [2, 2]
    assert ledger["primary_fit_eligible"].all()


def test_internal_gap_is_neither_exit_nor_session_censoring() -> None:
    frame = _session([1, 1, 1, 2], complete=False, ordinals=[0, 1, 3, 4])
    ledger = classify_training_run_endings(frame)

    assert RunEndingStatus.INVALIDATED_BY_SOURCE_GAP.value in set(ledger["ending_status"])
    invalid = ledger.loc[
        ledger["ending_status"].eq(RunEndingStatus.INVALIDATED_BY_SOURCE_GAP.value)
    ]
    assert invalid["primary_fit_eligible"].eq(False).all()


def test_incomplete_session_fails_closed_for_primary_fit() -> None:
    ledger = classify_training_run_endings(_session([1, 2], complete=False))

    assert ledger["ending_status"].tolist() == [
        RunEndingStatus.INCOMPLETE_OR_UNAVAILABLE_SESSION.value,
        RunEndingStatus.INCOMPLETE_OR_UNAVAILABLE_SESSION.value,
    ]
    assert ledger["primary_fit_eligible"].tolist() == [False, False]


@pytest.mark.parametrize("duration", [1, 24, 25, 78])
def test_exact_exit_remains_at_exact_age(duration: int) -> None:
    ledger = pd.DataFrame(
        {
            "state": [0],
            "duration": [duration],
            "ending_status": [RunEndingStatus.OBSERVED_STATE_EXIT.value],
            "primary_fit_eligible": [True],
        }
    )

    fit = estimate_right_censored_durations(
        ledger,
        state_count=1,
        config=DurationFitConfig(maximum_age=78, alpha=0.5, beta=0.5),
    )

    assert fit.exits[0, duration - 1] == 1
    assert fit.at_risk[0, duration - 1] == 1
    if duration < 78:
        assert fit.at_risk[0, duration] == 0


@pytest.mark.parametrize("duration", [1, 24, 78])
def test_censored_run_adds_exposure_but_no_exit(duration: int) -> None:
    ledger = pd.DataFrame(
        {
            "state": [0],
            "duration": [duration],
            "ending_status": [RunEndingStatus.RIGHT_CENSORED_SESSION_END.value],
            "primary_fit_eligible": [True],
        }
    )

    fit = estimate_right_censored_durations(
        ledger,
        state_count=1,
        config=DurationFitConfig(maximum_age=78, alpha=0.5, beta=0.5),
    )

    assert np.all(fit.at_risk[0, :duration] == 1)
    assert fit.exits.sum() == 0
    assert fit.censored[0, duration - 1] == 1


def test_no_age_is_forced_to_hazard_one_and_survival_is_monotone() -> None:
    ledger = pd.DataFrame(
        {
            "state": [0, 0, 0],
            "duration": [1, 24, 78],
            "ending_status": [
                RunEndingStatus.OBSERVED_STATE_EXIT.value,
                RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
                RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
            ],
            "primary_fit_eligible": [True, True, True],
        }
    )

    fit = estimate_right_censored_durations(
        ledger,
        state_count=1,
        config=DurationFitConfig(maximum_age=78, alpha=0.5, beta=0.5),
    )

    assert np.all((fit.hazard >= 0.0) & (fit.hazard < 1.0))
    assert np.all(np.diff(fit.survival[0]) <= 1e-15)
    assert fit.hazard[0, 23] < 1.0
    assert fit.hazard[0, 77] < 1.0


def test_sparse_tail_backoff_is_deterministic() -> None:
    ledger = pd.DataFrame(
        {
            "state": [0, 0, 1, 1],
            "duration": [1, 78, 2, 78],
            "ending_status": [
                RunEndingStatus.OBSERVED_STATE_EXIT.value,
                RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
                RunEndingStatus.OBSERVED_STATE_EXIT.value,
                RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
            ],
            "primary_fit_eligible": [True] * 4,
        }
    )
    config = DurationFitConfig(
        maximum_age=78,
        alpha=0.5,
        beta=0.5,
        minimum_state_at_risk=3,
        tail_prior_hazard=0.05,
    )

    first = estimate_right_censored_durations(ledger, state_count=2, config=config)
    second = estimate_right_censored_durations(ledger, state_count=2, config=config)

    np.testing.assert_array_equal(first.hazard, second.hazard)
    assert np.any(first.backoff_weight > 0.0)
    assert np.allclose(first.hazard, second.hazard)
