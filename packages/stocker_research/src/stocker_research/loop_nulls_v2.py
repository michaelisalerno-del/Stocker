"""Structure-preserving nulls for research-only loop discovery V2.

The primary null simulates state and duration runs from a fitted semi-Markov
process while preserving each original session length and boundary.  A broad,
frozen opening/middle/late variant conditions transitions on clock phase.  No
economic outcome is accepted by any public type or fitting function.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stocker_research.loop_duration_v2 import (
    DiscreteSurvivalDurationModel,
    DurationObservation,
)

CLOCK_PHASE_BOUNDARIES = (0, 12, 60, 78)
CLOCK_PHASES = ("opening", "middle", "late")
RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


@dataclass(frozen=True, slots=True)
class SessionRunSequence:
    """Outcome-free hard-state run sequence for one regular session."""

    symbol: str
    session: str
    states: tuple[int, ...]
    durations: tuple[int, ...]
    terminal_right_censored: bool

    def __post_init__(self) -> None:
        if not self.states or len(self.states) != len(self.durations):
            raise ValueError("session states and durations must be nonempty and aligned")
        if any(state < 0 for state in self.states):
            raise ValueError("session contains an unknown state")
        if any(duration <= 0 for duration in self.durations):
            raise ValueError("session contains a nonpositive duration")
        if any(
            left == right for left, right in zip(self.states[:-1], self.states[1:], strict=True)
        ):
            raise ValueError("compressed session contains a self transition")


@dataclass(frozen=True, slots=True)
class SimulatedSession:
    states: tuple[int, ...]
    durations: tuple[int, ...]
    terminal_right_censored: bool
    phase_labels: tuple[str, ...]


class _NullSimulator(Protocol):
    def simulate_session(
        self, session_length: int, *, rng: np.random.Generator
    ) -> SimulatedSession: ...


def session_phase(bar_ordinal: int) -> str:
    """Map a zero-based regular-session bar to a frozen broad phase."""

    value = int(bar_ordinal)
    if value < CLOCK_PHASE_BOUNDARIES[0] or value >= CLOCK_PHASE_BOUNDARIES[-1]:
        raise ValueError("bar ordinal lies outside the regular session")
    if value < CLOCK_PHASE_BOUNDARIES[1]:
        return "opening"
    if value < CLOCK_PHASE_BOUNDARIES[2]:
        return "middle"
    return "late"


@dataclass(frozen=True, slots=True)
class SemiMarkovNull:
    """Fitted initial, transition, and censored-duration structural null."""

    initial_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    duration_model: DiscreteSurvivalDurationModel
    duration_cumulative_probabilities: np.ndarray
    state_count: int

    @classmethod
    def fit(
        cls,
        sessions: Iterable[SessionRunSequence],
        *,
        state_count: int,
        maximum_duration: int,
        smoothing_strength: float = 8.0,
    ) -> SemiMarkovNull:
        records = tuple(sessions)
        if not records:
            raise ValueError("semi-Markov fitting requires sessions")
        if state_count <= 1:
            raise ValueError("state_count must exceed one")
        if any(max(record.states) >= state_count for record in records):
            raise ValueError("session contains a state outside the model")
        initial = np.full(state_count, 0.5, dtype=float)
        transitions = np.full((state_count, state_count), 0.5, dtype=float)
        np.fill_diagonal(transitions, 0.0)
        duration_rows: list[DurationObservation] = []
        for record in records:
            initial[record.states[0]] += 1.0
            for origin, destination in zip(record.states[:-1], record.states[1:], strict=True):
                if origin == destination:
                    raise ValueError("compressed session contains a self transition")
                transitions[origin, destination] += 1.0
            for index, (state, duration) in enumerate(
                zip(record.states, record.durations, strict=True)
            ):
                duration_rows.append(
                    DurationObservation(
                        state=state,
                        duration=duration,
                        right_censored=(
                            index == len(record.states) - 1 and record.terminal_right_censored
                        ),
                    )
                )
        initial /= initial.sum()
        transitions /= transitions.sum(axis=1, keepdims=True)
        duration_model = DiscreteSurvivalDurationModel.fit(
            duration_rows,
            state_count=state_count,
            maximum_duration=maximum_duration,
            smoothing_strength=smoothing_strength,
        )
        duration_cumulative = np.vstack(
            [
                np.cumsum(
                    np.r_[
                        duration_model.duration_distribution(state).exact_pmf,
                        duration_model.duration_distribution(state).survival_tail,
                    ]
                )
                for state in range(state_count)
            ]
        )
        duration_cumulative[:, -1] = 1.0
        return cls(
            initial,
            transitions,
            duration_model,
            duration_cumulative,
            state_count,
        )

    def _sample_duration(self, state: int, *, rng: np.random.Generator, remaining: int) -> int:
        selected = int(
            np.searchsorted(
                self.duration_cumulative_probabilities[state],
                rng.random(),
                side="right",
            )
        )
        proposed = selected + 1
        return min(proposed, remaining)

    @staticmethod
    def _sample_state(probabilities: np.ndarray, *, rng: np.random.Generator) -> int:
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        return int(np.searchsorted(cumulative, rng.random(), side="right"))

    def simulate_session(
        self, session_length: int, *, rng: np.random.Generator
    ) -> SimulatedSession:
        if session_length <= 0:
            raise ValueError("session_length must be positive")
        state = self._sample_state(self.initial_probabilities, rng=rng)
        states: list[int] = []
        durations: list[int] = []
        elapsed = 0
        while elapsed < session_length:
            states.append(state)
            duration = self._sample_duration(state, rng=rng, remaining=session_length - elapsed)
            durations.append(duration)
            elapsed += duration
            if elapsed >= session_length:
                break
            state = self._sample_state(self.transition_probabilities[state], rng=rng)
        if sum(durations) != session_length:
            raise AssertionError("simulated session length drifted")
        return SimulatedSession(
            states=tuple(states),
            durations=tuple(durations),
            terminal_right_censored=True,
            phase_labels=tuple(
                session_phase(min(index, CLOCK_PHASE_BOUNDARIES[-1] - 1))
                for index in range(session_length)
            ),
        )


@dataclass(frozen=True, slots=True)
class ClockConditionedSemiMarkovNull:
    """Semi-Markov null with transitions conditioned on frozen broad phases."""

    base: SemiMarkovNull
    phase_transition_probabilities: np.ndarray
    phase_boundaries: tuple[int, int, int, int] = CLOCK_PHASE_BOUNDARIES

    @classmethod
    def fit(
        cls,
        sessions: Iterable[SessionRunSequence],
        *,
        state_count: int,
        maximum_duration: int,
        smoothing_strength: float = 8.0,
    ) -> ClockConditionedSemiMarkovNull:
        records = tuple(sessions)
        base = SemiMarkovNull.fit(
            records,
            state_count=state_count,
            maximum_duration=maximum_duration,
            smoothing_strength=smoothing_strength,
        )
        counts = np.zeros((len(CLOCK_PHASES), state_count, state_count), dtype=float)
        for phase_index in range(len(CLOCK_PHASES)):
            counts[phase_index] = 2.0 * base.transition_probabilities
        for record in records:
            elapsed = 0
            for index, duration in enumerate(record.durations[:-1]):
                elapsed += duration
                phase_index = CLOCK_PHASES.index(
                    session_phase(min(elapsed, CLOCK_PHASE_BOUNDARIES[-1] - 1))
                )
                origin = record.states[index]
                destination = record.states[index + 1]
                counts[phase_index, origin, destination] += 1.0
        for phase_index in range(len(CLOCK_PHASES)):
            np.fill_diagonal(counts[phase_index], 0.0)
        probabilities = counts / counts.sum(axis=2, keepdims=True)
        return cls(base=base, phase_transition_probabilities=probabilities)

    def simulate_session(
        self, session_length: int, *, rng: np.random.Generator
    ) -> SimulatedSession:
        if session_length <= 0:
            raise ValueError("session_length must be positive")
        state = self.base._sample_state(self.base.initial_probabilities, rng=rng)
        states: list[int] = []
        durations: list[int] = []
        elapsed = 0
        while elapsed < session_length:
            states.append(state)
            duration = self.base._sample_duration(
                state, rng=rng, remaining=session_length - elapsed
            )
            durations.append(duration)
            elapsed += duration
            if elapsed >= session_length:
                break
            phase = session_phase(min(elapsed, CLOCK_PHASE_BOUNDARIES[-1] - 1))
            phase_index = CLOCK_PHASES.index(phase)
            state = self.base._sample_state(
                self.phase_transition_probabilities[phase_index, state], rng=rng
            )
        return SimulatedSession(
            states=tuple(states),
            durations=tuple(durations),
            terminal_right_censored=True,
            phase_labels=tuple(
                session_phase(min(index, CLOCK_PHASE_BOUNDARIES[-1] - 1))
                for index in range(session_length)
            ),
        )


def circular_session_control(session: SessionRunSequence, *, offset: int) -> SessionRunSequence:
    """Rotate whole state-duration blocks without breaking their pairing."""

    width = len(session.states)
    if width < 2 or offset <= 0 or offset >= width:
        raise ValueError("circular offset must be a nonzero in-session boundary")
    pairs = list(zip(session.states, session.durations, strict=True))
    rotated = pairs[offset:] + pairs[:offset]
    merged: list[tuple[int, int]] = []
    for state, duration in rotated:
        if merged and merged[-1][0] == state:
            merged[-1] = (state, merged[-1][1] + duration)
        else:
            merged.append((state, duration))
    return SessionRunSequence(
        symbol=session.symbol,
        session=session.session,
        states=tuple(pair[0] for pair in merged),
        durations=tuple(pair[1] for pair in merged),
        terminal_right_censored=session.terminal_right_censored,
    )


def count_candidate_paths(
    sessions: Sequence[SessionRunSequence], candidates: Sequence[tuple[int, ...]]
) -> np.ndarray:
    """Count exact closed state-event paths inside session boundaries."""

    output = np.zeros(len(candidates), dtype=np.int64)
    candidate_lookup: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_lookup.setdefault(len(candidate), {}).setdefault(candidate, []).append(
            candidate_index
        )
    for session in sessions:
        for width, paths in candidate_lookup.items():
            for start in range(len(session.states) - width + 1):
                observed = tuple(session.states[start : start + width])
                for candidate_index in paths.get(observed, ()):
                    output[candidate_index] += 1
    return output


def first_order_expected_counts(
    sessions: Sequence[SessionRunSequence],
    model: SemiMarkovNull,
    candidates: Sequence[tuple[int, ...]],
) -> np.ndarray:
    """Conditional first-order motif expectation without full simulation."""

    expected = np.zeros(len(candidates), dtype=float)
    for candidate_index, candidate in enumerate(candidates):
        probability = 1.0
        for origin, destination in zip(candidate[:-1], candidate[1:], strict=True):
            probability *= model.transition_probabilities[origin, destination]
        anchors = sum(
            sum(
                session.states[start] == candidate[0]
                for start in range(len(session.states) - len(candidate) + 1)
            )
            for session in sessions
        )
        expected[candidate_index] = anchors * probability
    return expected


def simulate_null_counts(
    model: _NullSimulator,
    *,
    session_lengths: Sequence[int],
    candidates: Sequence[tuple[int, ...]],
    draws: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic empirical structural-null count draws."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    output = np.zeros((draws, len(candidates)), dtype=np.int64)
    for draw in range(draws):
        simulated = tuple(
            SessionRunSequence(
                symbol=f"null_{index}",
                session=f"draw_{draw}",
                states=result.states,
                durations=result.durations,
                terminal_right_censored=True,
            )
            for index, length in enumerate(session_lengths)
            for result in (model.simulate_session(int(length), rng=rng),)
        )
        output[draw] = count_candidate_paths(simulated, candidates)
    return output


def empirical_p_values(observed: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """One-sided empirical exceedance p-values with the plus-one correction."""

    values = np.asarray(observed)
    null = np.asarray(draws)
    if null.ndim != 2 or values.shape != (null.shape[1],):
        raise ValueError("observed and null count shapes differ")
    return np.asarray(
        (1.0 + np.sum(null >= values[None, :], axis=0)) / (null.shape[0] + 1.0),
        dtype=float,
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Deterministic Benjamini-Hochberg FDR adjustment."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("p-values must be a vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


__all__ = [
    "CLOCK_PHASE_BOUNDARIES",
    "ClockConditionedSemiMarkovNull",
    "SemiMarkovNull",
    "SessionRunSequence",
    "SimulatedSession",
    "benjamini_hochberg",
    "circular_session_control",
    "count_candidate_paths",
    "empirical_p_values",
    "first_order_expected_counts",
    "session_phase",
    "simulate_null_counts",
]
