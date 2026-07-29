"""Censored discrete-survival duration infrastructure for loop events V2.

Durations are exact through the full regular-session support.  There is no
``>=24`` bucket and no forced age-24 exit.  Terminal session runs enter the
risk set as right-censored observations, and remaining session time truncates
forecasts without discarding probability mass.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

REGULAR_SESSION_MAX_BARS = 78
RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


@dataclass(frozen=True, slots=True)
class DurationObservation:
    """Observed run length; terminal session runs are explicitly censored."""

    state: int
    duration: int
    right_censored: bool

    def __post_init__(self) -> None:
        if self.state < 0:
            raise ValueError("duration state must be nonnegative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")


@dataclass(frozen=True, slots=True)
class DurationDistribution:
    """Exact duration PMF plus explicit mass beyond represented support."""

    exact_pmf: np.ndarray
    survival_tail: float

    def __post_init__(self) -> None:
        values = np.asarray(self.exact_pmf, dtype=float)
        tail = float(self.survival_tail)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("exact_pmf must be a nonempty vector")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("duration PMF contains an invalid value")
        if not np.isfinite(tail) or tail < 0.0:
            raise ValueError("duration survival tail is invalid")
        if not np.isclose(values.sum() + tail, 1.0, atol=1e-12):
            raise ValueError("duration PMF and tail do not normalize")


@dataclass(frozen=True, slots=True)
class ConditionalDurationForecast:
    """Exit availability by future-bar offset, including no-completion mass."""

    completion_pmf: np.ndarray
    completion_probability: float
    no_completion_probability: float
    session_terminal_probability: float


@dataclass(frozen=True, slots=True)
class CompletionConvolution:
    """Route completion time by bar offset after duration convolution."""

    completion_pmf: np.ndarray
    completion_probability: float
    no_completion_probability: float


@dataclass(frozen=True, slots=True)
class DiscreteSurvivalDurationModel:
    """Hierarchically smoothed state-specific discrete hazards with censoring."""

    hazard: np.ndarray
    at_risk_counts: np.ndarray
    exit_counts: np.ndarray
    censored_counts: np.ndarray
    maximum_duration: int
    smoothing_strength: float

    @classmethod
    def fit(
        cls,
        observations: Iterable[DurationObservation],
        *,
        state_count: int,
        maximum_duration: int = REGULAR_SESSION_MAX_BARS,
        smoothing_strength: float = 8.0,
    ) -> DiscreteSurvivalDurationModel:
        records = tuple(observations)
        if state_count <= 0 or maximum_duration <= 0:
            raise ValueError("state_count and maximum_duration must be positive")
        if smoothing_strength < 0.0:
            raise ValueError("smoothing_strength cannot be negative")
        if not records:
            raise ValueError("duration fitting requires observations")
        if any(record.state >= state_count for record in records):
            raise ValueError("duration observation has an unknown state")
        if any(record.duration > maximum_duration for record in records):
            raise ValueError("duration exceeds declared regular-session support")

        at_risk = np.zeros((state_count, maximum_duration), dtype=np.int64)
        exits = np.zeros_like(at_risk)
        censored = np.zeros_like(at_risk)
        for record in records:
            at_risk[record.state, : record.duration] += 1
            if record.right_censored:
                censored[record.state, record.duration - 1] += 1
            else:
                exits[record.state, record.duration - 1] += 1

        pooled_risk = at_risk.sum(axis=0).astype(float)
        pooled_exits = exits.sum(axis=0).astype(float)
        pooled_hazard = np.divide(
            pooled_exits + 0.5,
            pooled_risk + 1.0,
            out=np.zeros(maximum_duration, dtype=float),
            where=pooled_risk > 0,
        )
        hazard = np.zeros_like(at_risk, dtype=float)
        for state in range(state_count):
            supported = at_risk[state] > 0
            if smoothing_strength == 0.0:
                hazard[state, supported] = exits[state, supported] / at_risk[state, supported]
            else:
                hazard[state, supported] = (
                    exits[state, supported] + smoothing_strength * pooled_hazard[supported]
                ) / (at_risk[state, supported] + smoothing_strength)
                backoff = ~supported & (pooled_risk > 0)
                hazard[state, backoff] = pooled_hazard[backoff]
        hazard = np.clip(hazard, 0.0, 1.0)
        return cls(
            hazard=hazard,
            at_risk_counts=at_risk,
            exit_counts=exits,
            censored_counts=censored,
            maximum_duration=maximum_duration,
            smoothing_strength=smoothing_strength,
        )

    def duration_distribution(self, state: int) -> DurationDistribution:
        """Return exact total-duration mass without forcing the final hazard."""

        if state < 0 or state >= len(self.hazard):
            raise ValueError("unknown duration state")
        pmf = np.zeros(self.maximum_duration, dtype=float)
        survival = 1.0
        for age in range(self.maximum_duration):
            pmf[age] = survival * self.hazard[state, age]
            survival *= 1.0 - self.hazard[state, age]
        total = float(pmf.sum() + survival)
        if not np.isclose(total, 1.0, atol=1e-12):
            raise AssertionError("duration distribution lost probability mass")
        return DurationDistribution(exact_pmf=pmf, survival_tail=float(survival))

    def conditional_exit_distribution(
        self,
        *,
        state: int,
        current_age: int,
        horizon: int,
        bars_remaining_in_session: int,
    ) -> ConditionalDurationForecast:
        """Forecast when the next state event becomes available after a decision."""

        if state < 0 or state >= len(self.hazard):
            raise ValueError("unknown duration state")
        if current_age <= 0 or horizon <= 0 or bars_remaining_in_session < 0:
            raise ValueError("invalid conditional duration bounds")
        pmf = np.zeros(horizon + 1, dtype=float)
        effective = min(horizon, bars_remaining_in_session)
        survival = 1.0
        for offset in range(1, effective + 1):
            total_duration = current_age + offset - 1
            hazard = (
                float(self.hazard[state, total_duration - 1])
                if total_duration <= self.maximum_duration
                else 0.0
            )
            pmf[offset] = survival * hazard
            survival *= 1.0 - hazard
        no_completion = float(max(0.0, 1.0 - pmf.sum()))
        terminal = no_completion if bars_remaining_in_session < horizon else 0.0
        if not np.isclose(pmf.sum() + no_completion, 1.0, atol=1e-12):
            raise AssertionError("conditional duration forecast lost mass")
        return ConditionalDurationForecast(
            completion_pmf=pmf,
            completion_probability=float(pmf.sum()),
            no_completion_probability=no_completion,
            session_terminal_probability=terminal,
        )


def convolve_duration_distributions(
    distributions: Sequence[DurationDistribution], *, horizon: int
) -> CompletionConvolution:
    """Convolve exact durations and retain all after-horizon/tail mass."""

    if not distributions:
        raise ValueError("duration convolution requires at least one transition")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    exact = np.asarray([1.0], dtype=float)
    for distribution in distributions:
        # Index zero means zero elapsed bars; exact_pmf index zero means a
        # one-bar duration, so prepend an explicit zero before convolution.
        kernel = np.zeros(len(distribution.exact_pmf) + 1, dtype=float)
        kernel[1:] = np.asarray(distribution.exact_pmf, dtype=float)
        exact = np.convolve(exact, kernel)
        if len(exact) > horizon + 1:
            exact = exact[: horizon + 1]
    completion = np.zeros(horizon + 1, dtype=float)
    completion[: len(exact)] = exact
    completion[0] = 0.0
    completion_probability = float(completion.sum())
    no_completion = float(max(0.0, 1.0 - completion_probability))
    if not np.isclose(completion_probability + no_completion, 1.0, atol=1e-12):
        raise AssertionError("duration convolution lost probability mass")
    return CompletionConvolution(
        completion_pmf=completion,
        completion_probability=completion_probability,
        no_completion_probability=no_completion,
    )


__all__ = [
    "CompletionConvolution",
    "ConditionalDurationForecast",
    "DiscreteSurvivalDurationModel",
    "DurationDistribution",
    "DurationObservation",
    "REGULAR_SESSION_MAX_BARS",
    "convolve_duration_distributions",
]
