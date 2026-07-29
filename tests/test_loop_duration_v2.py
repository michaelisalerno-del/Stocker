from __future__ import annotations

import numpy as np
import pytest

from stocker_research.loop_duration_v2 import (
    DiscreteSurvivalDurationModel,
    DurationDistribution,
    DurationObservation,
    convolve_duration_distributions,
)


def _delta(duration: int, *, maximum: int = 78) -> DurationDistribution:
    pmf = np.zeros(maximum, dtype=float)
    pmf[duration - 1] = 1.0
    return DurationDistribution(exact_pmf=pmf, survival_tail=0.0)


def test_exact_durations_1_23_24_and_25_remain_distinct() -> None:
    observations = tuple(
        DurationObservation(state=0, duration=value, right_censored=False)
        for value in (1, 23, 24, 25)
    )
    model = DiscreteSurvivalDurationModel.fit(
        observations, state_count=1, maximum_duration=78, smoothing_strength=0.0
    )
    distribution = model.duration_distribution(0)

    assert distribution.exact_pmf[0] > 0.0
    assert distribution.exact_pmf[22] > 0.0
    assert distribution.exact_pmf[23] > 0.0
    assert distribution.exact_pmf[24] > 0.0
    assert distribution.exact_pmf[23] != distribution.exact_pmf[24] or (
        model.exit_counts[0, 23] == 1 and model.exit_counts[0, 24] == 1
    )


def test_duration_24_is_included_exactly_at_horizon_24() -> None:
    result = convolve_duration_distributions((_delta(24),), horizon=24)

    assert result.completion_pmf[24] == pytest.approx(1.0)
    assert result.completion_probability == pytest.approx(1.0)
    assert result.no_completion_probability == pytest.approx(0.0)


def test_duration_25_is_not_folded_into_duration_24() -> None:
    result = convolve_duration_distributions((_delta(25),), horizon=24)

    assert result.completion_pmf[24] == pytest.approx(0.0)
    assert result.completion_probability == pytest.approx(0.0)
    assert result.no_completion_probability == pytest.approx(1.0)


def test_run_is_not_forced_to_exit_at_age_24() -> None:
    model = DiscreteSurvivalDurationModel.fit(
        (DurationObservation(state=0, duration=25, right_censored=False),),
        state_count=1,
        maximum_duration=78,
        smoothing_strength=0.0,
    )

    assert model.hazard[0, 23] == pytest.approx(0.0)
    assert model.hazard[0, 24] == pytest.approx(1.0)
    assert model.duration_distribution(0).exact_pmf[24] == pytest.approx(1.0)


def test_right_censored_terminal_run_contributes_at_risk_not_an_exit() -> None:
    model = DiscreteSurvivalDurationModel.fit(
        (
            DurationObservation(state=0, duration=5, right_censored=False),
            DurationObservation(state=0, duration=7, right_censored=True),
        ),
        state_count=1,
        maximum_duration=78,
        smoothing_strength=0.0,
    )

    assert model.at_risk_counts[0, 4] == 2
    assert model.exit_counts[0, 4] == 1
    assert model.censored_counts[0, 6] == 1
    assert model.exit_counts[0, 6] == 0
    assert model.duration_distribution(0).survival_tail > 0.0


def test_remaining_session_time_truncates_completion_without_losing_mass() -> None:
    model = DiscreteSurvivalDurationModel.fit(
        (
            DurationObservation(state=0, duration=3, right_censored=False),
            DurationObservation(state=0, duration=6, right_censored=False),
        ),
        state_count=1,
        maximum_duration=78,
        smoothing_strength=1.0,
    )
    forecast = model.conditional_exit_distribution(
        state=0, current_age=1, horizon=5, bars_remaining_in_session=2
    )

    assert np.allclose(forecast.completion_pmf[3:], 0.0)
    assert forecast.session_terminal_probability == pytest.approx(
        forecast.no_completion_probability
    )
    assert forecast.completion_probability + forecast.no_completion_probability == pytest.approx(
        1.0
    )


def test_completion_exactly_on_conditional_horizon_is_counted() -> None:
    model = DiscreteSurvivalDurationModel.fit(
        (DurationObservation(state=0, duration=4, right_censored=False),),
        state_count=1,
        maximum_duration=78,
        smoothing_strength=0.0,
    )
    forecast = model.conditional_exit_distribution(
        state=0, current_age=1, horizon=4, bars_remaining_in_session=10
    )

    assert forecast.completion_pmf[4] == pytest.approx(1.0)
    assert forecast.no_completion_probability == pytest.approx(0.0)


def test_completion_after_horizon_remains_no_completion_mass() -> None:
    result = convolve_duration_distributions((_delta(10),), horizon=6)

    assert result.completion_pmf.sum() == pytest.approx(0.0)
    assert result.no_completion_probability == pytest.approx(1.0)


def test_route_convolution_conserves_probability() -> None:
    first = DurationDistribution(exact_pmf=np.asarray([0.5, 0.5] + [0.0] * 76), survival_tail=0.0)
    second = DurationDistribution(
        exact_pmf=np.asarray([0.25, 0.75] + [0.0] * 76), survival_tail=0.0
    )
    result = convolve_duration_distributions((first, second), horizon=3)

    assert result.completion_pmf[2] == pytest.approx(0.125)
    assert result.completion_pmf[3] == pytest.approx(0.5)
    assert result.completion_probability + result.no_completion_probability == pytest.approx(1.0)


def test_duration_distribution_rejects_nonconserving_mass() -> None:
    with pytest.raises(ValueError, match="normalize"):
        DurationDistribution(exact_pmf=np.asarray([0.8, 0.8]), survival_tail=0.0)
