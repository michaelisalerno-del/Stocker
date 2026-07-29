"""Strictly past-only clock and transition-timing census."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CensusConfig:
    smoothing_alpha: float = 1.0
    pooling_strength: float = 10.0
    minimum_likelihood: float = 0.05
    lift_minimum: float = 0.25
    lift_maximum: float = 4.0
    coarse_clock: bool = False


def clock_bin(bar_ordinal: int, *, coarse: bool = False) -> str:
    if coarse:
        return "early" if int(bar_ordinal) <= 23 else "late"
    if int(bar_ordinal) <= 11:
        return "open"
    if int(bar_ordinal) <= 35:
        return "middle"
    return "late"


def _beta_rate(successes: int, rows: int, alpha: float) -> float:
    return (float(successes) + alpha) / (float(rows) + 2.0 * alpha)


@dataclass(frozen=True)
class TrainingOnlyCensus:
    examples: pd.DataFrame
    config: CensusConfig
    training_sessions: tuple[str, ...]
    training_rows: int
    clock_counts: dict[tuple[str, str, int, str], tuple[int, int]]
    state_counts: dict[tuple[str, str, int], tuple[int, int]]
    occurrence_cells: dict[tuple[str, str, int], pd.DataFrame]
    occurrence_states: dict[int, pd.DataFrame]

    @classmethod
    def from_examples(
        cls,
        examples: pd.DataFrame,
        *,
        period: int,
        score_session: str,
        config: CensusConfig,
        excluded_stocks: Set[str] | None = None,
    ) -> TrainingOnlyCensus:
        required = {
            "period",
            "session_date",
            "symbol_norm",
            "loop_id",
            "orientation",
            "current_state",
            "bar_ordinal",
            "loop_occurs",
            "first_transition_state",
            "first_transition_lag",
            "second_transition_state",
            "second_transition_lag",
        }
        missing = required - set(examples.columns)
        if missing:
            raise ValueError(f"missing census example columns: {sorted(missing)}")
        frame = examples.loc[
            examples["period"].astype(int).eq(int(period))
            & examples["session_date"].astype(str).lt(str(score_session))
        ].copy()
        if excluded_stocks:
            frame = frame.loc[
                ~frame["symbol_norm"].astype(str).isin(set(map(str, excluded_stocks)))
            ].copy()
        frame["clock_bin"] = frame["bar_ordinal"].map(
            lambda value: clock_bin(int(str(value)), coarse=config.coarse_clock)
        )
        frame = frame.sort_values(
            ["session_date", "symbol_norm", "bar_ordinal", "loop_id"], kind="stable"
        ).reset_index(drop=True)
        sessions = tuple(sorted(frame["session_date"].astype(str).unique()))
        clock_counts = {
            (str(loop), str(orientation), int(str(state)), str(clock)): (
                len(group),
                int(group["loop_occurs"].fillna(False).astype(bool).sum()),
            )
            for (loop, orientation, state, clock), group in frame.groupby(
                ["loop_id", "orientation", "current_state", "clock_bin"], sort=False
            )
        }
        state_counts = {
            (str(loop), str(orientation), int(str(state))): (
                len(group),
                int(group["loop_occurs"].fillna(False).astype(bool).sum()),
            )
            for (loop, orientation, state), group in frame.groupby(
                ["loop_id", "orientation", "current_state"], sort=False
            )
        }
        occurrences = frame.loc[frame["loop_occurs"].fillna(False).astype(bool)]
        occurrence_cells = {
            (str(loop), str(orientation), int(str(state))): group.reset_index(drop=True)
            for (loop, orientation, state), group in occurrences.groupby(
                ["loop_id", "orientation", "current_state"], sort=False
            )
        }
        occurrence_states = {
            int(str(state)): group.reset_index(drop=True)
            for state, group in occurrences.groupby("current_state", sort=False)
        }
        return cls(
            frame,
            config,
            sessions,
            len(frame),
            clock_counts,
            state_counts,
            occurrence_cells,
            occurrence_states,
        )

    def clock_lift(
        self,
        loop_id: str,
        orientation: str,
        current_state: int,
        bar_ordinal: int,
    ) -> float:
        bin_name = clock_bin(bar_ordinal, coarse=self.config.coarse_clock)
        state_rows, state_successes = self.state_counts.get(
            (str(loop_id), str(orientation), int(current_state)), (0, 0)
        )
        clock_rows, clock_successes = self.clock_counts.get(
            (str(loop_id), str(orientation), int(current_state), bin_name), (0, 0)
        )
        base = _beta_rate(
            state_successes,
            state_rows,
            self.config.smoothing_alpha,
        )
        conditional = _beta_rate(
            clock_successes,
            clock_rows,
            self.config.smoothing_alpha,
        )
        raw_lift = conditional / base if base > 0.0 else 1.0
        weight = clock_rows / (clock_rows + self.config.pooling_strength)
        shrunk = weight * raw_lift + (1.0 - weight)
        return min(self.config.lift_maximum, max(self.config.lift_minimum, shrunk))

    @staticmethod
    def _evidence_matches(
        row: Any,
        observed_transitions: tuple[int, ...],
        bars_since_anchor: int,
    ) -> bool:
        first_lag = pd.to_numeric(row.first_transition_lag, errors="coerce")
        second_lag = pd.to_numeric(row.second_transition_lag, errors="coerce")
        if not observed_transitions:
            return bool(pd.isna(first_lag) or float(first_lag) > bars_since_anchor)
        first_state = pd.to_numeric(row.first_transition_state, errors="coerce")
        if pd.isna(first_state) or int(first_state) != int(observed_transitions[0]):
            return False
        if pd.isna(first_lag) or float(first_lag) > bars_since_anchor:
            return False
        if len(observed_transitions) == 1:
            return bool(pd.isna(second_lag) or float(second_lag) > bars_since_anchor)
        second_state = pd.to_numeric(row.second_transition_state, errors="coerce")
        return bool(
            not pd.isna(second_state)
            and int(second_state) == int(observed_transitions[1])
            and not pd.isna(second_lag)
            and float(second_lag) <= bars_since_anchor
        )

    def timing_likelihood(
        self,
        loop_id: str,
        orientation: str,
        current_state: int,
        observed_transitions: tuple[int, ...],
        bars_since_anchor: int,
    ) -> float:
        cell = self.occurrence_cells.get(
            (str(loop_id), str(orientation), int(current_state)), pd.DataFrame()
        )
        pooled = self.occurrence_states.get(int(current_state), pd.DataFrame())

        def rate(frame: pd.DataFrame) -> float:
            successes = sum(
                self._evidence_matches(row, observed_transitions, bars_since_anchor)
                for row in frame.itertuples(index=False)
            )
            return _beta_rate(successes, len(frame), self.config.smoothing_alpha)

        cell_rate = rate(cell)
        pooled_rate = rate(pooled)
        weight = len(cell) / (len(cell) + self.config.pooling_strength)
        probability = weight * cell_rate + (1.0 - weight) * pooled_rate
        return max(self.config.minimum_likelihood, min(1.0, probability))

    def export(self) -> pd.DataFrame:
        """Return the exact past-only rows used by this census."""

        return self.examples.copy()


class RollingTrainingOnlyCensus:
    """Chronological census that incorporates each completed session exactly once.

    The public likelihood equations match :class:`TrainingOnlyCensus`; the rolling
    form avoids repeatedly regrouping the full history at every score session.
    """

    def __init__(
        self,
        examples: pd.DataFrame,
        *,
        period: int,
        config: CensusConfig,
        excluded_stocks: Set[str] | None = None,
    ) -> None:
        required = {
            "period",
            "session_date",
            "symbol_norm",
            "loop_id",
            "orientation",
            "current_state",
            "bar_ordinal",
            "loop_occurs",
            "first_transition_state",
            "first_transition_lag",
            "second_transition_state",
            "second_transition_lag",
        }
        missing = required - set(examples.columns)
        if missing:
            raise ValueError(f"missing census example columns: {sorted(missing)}")
        frame = examples.loc[examples["period"].astype(int).eq(int(period))].copy()
        if excluded_stocks:
            frame = frame.loc[
                ~frame["symbol_norm"].astype(str).isin(set(map(str, excluded_stocks)))
            ].copy()
        frame["session_date"] = frame["session_date"].astype(str)
        frame["clock_bin"] = frame["bar_ordinal"].map(
            lambda value: clock_bin(int(str(value)), coarse=config.coarse_clock)
        )
        self.config = config
        self.period = int(period)
        self._sessions = tuple(sorted(frame["session_date"].unique()))
        self._session_rows = {
            str(session): group.reset_index(drop=True)
            for session, group in frame.groupby("session_date", sort=False)
        }
        self._cursor = 0
        self._last_score_session: str | None = None
        self.training_sessions: list[str] = []
        self.training_rows = 0
        self.clock_counts: dict[tuple[str, str, int, str], tuple[int, int]] = {}
        self.state_counts: dict[tuple[str, str, int], tuple[int, int]] = {}
        self._occurrence_cells: dict[
            tuple[str, str, int], list[tuple[float, float, float, float]]
        ] = defaultdict(list)
        self._occurrence_states: dict[int, list[tuple[float, float, float, float]]] = defaultdict(
            list
        )
        self._compiled_cells: dict[tuple[str, str, int], np.ndarray] = {}
        self._compiled_states: dict[int, np.ndarray] = {}

    @staticmethod
    def _increment(
        counts: dict[Any, tuple[int, int]],
        key: Any,
        success: bool,
    ) -> None:
        rows, successes = counts.get(key, (0, 0))
        counts[key] = (rows + 1, successes + int(success))

    @staticmethod
    def _numeric(value: Any) -> float:
        result = pd.to_numeric(value, errors="coerce")
        return float(result) if not pd.isna(result) else float("nan")

    def advance_before(self, score_session: str) -> None:
        """Advance to a score session without consuming that session's rows."""

        session = str(score_session)
        if self._last_score_session is not None and session < self._last_score_session:
            raise ValueError("rolling census score sessions must be chronological")
        changed = False
        while self._cursor < len(self._sessions) and self._sessions[self._cursor] < session:
            training_session = self._sessions[self._cursor]
            rows = self._session_rows[training_session]
            for row in rows.itertuples(index=False):
                loop = str(row.loop_id)
                orientation = str(row.orientation)
                state = int(str(row.current_state))
                occurred = bool(False if pd.isna(row.loop_occurs) else row.loop_occurs)
                self._increment(
                    self.clock_counts,
                    (loop, orientation, state, str(row.clock_bin)),
                    occurred,
                )
                self._increment(
                    self.state_counts,
                    (loop, orientation, state),
                    occurred,
                )
                if occurred:
                    evidence = (
                        self._numeric(row.first_transition_state),
                        self._numeric(row.first_transition_lag),
                        self._numeric(row.second_transition_state),
                        self._numeric(row.second_transition_lag),
                    )
                    self._occurrence_cells[(loop, orientation, state)].append(evidence)
                    self._occurrence_states[state].append(evidence)
            self.training_rows += len(rows)
            self.training_sessions.append(training_session)
            self._cursor += 1
            changed = True
        if changed:
            self._compiled_cells.clear()
            self._compiled_states.clear()
        self._last_score_session = session

    def clock_lift(
        self,
        loop_id: str,
        orientation: str,
        current_state: int,
        bar_ordinal: int,
    ) -> float:
        bin_name = clock_bin(bar_ordinal, coarse=self.config.coarse_clock)
        state_rows, state_successes = self.state_counts.get(
            (str(loop_id), str(orientation), int(current_state)), (0, 0)
        )
        clock_rows, clock_successes = self.clock_counts.get(
            (str(loop_id), str(orientation), int(current_state), bin_name), (0, 0)
        )
        base = _beta_rate(state_successes, state_rows, self.config.smoothing_alpha)
        conditional = _beta_rate(clock_successes, clock_rows, self.config.smoothing_alpha)
        raw_lift = conditional / base if base > 0.0 else 1.0
        weight = clock_rows / (clock_rows + self.config.pooling_strength)
        shrunk = weight * raw_lift + (1.0 - weight)
        return min(self.config.lift_maximum, max(self.config.lift_minimum, shrunk))

    @staticmethod
    def _compile(
        rows: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, 4), dtype=float)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _match_count(
        evidence: np.ndarray,
        observed_transitions: tuple[int, ...],
        bars_since_anchor: int,
    ) -> int:
        if len(evidence) == 0:
            return 0
        first_state, first_lag, second_state, second_lag = evidence.T
        bars = float(bars_since_anchor)
        if not observed_transitions:
            matches = np.isnan(first_lag) | (first_lag > bars)
        elif len(observed_transitions) == 1:
            matches = (
                (first_state == int(observed_transitions[0]))
                & (first_lag <= bars)
                & (np.isnan(second_lag) | (second_lag > bars))
            )
        else:
            matches = (
                (first_state == int(observed_transitions[0]))
                & (first_lag <= bars)
                & (second_state == int(observed_transitions[1]))
                & (second_lag <= bars)
            )
        return int(np.count_nonzero(matches))

    def timing_likelihood(
        self,
        loop_id: str,
        orientation: str,
        current_state: int,
        observed_transitions: tuple[int, ...],
        bars_since_anchor: int,
    ) -> float:
        cell_key = (str(loop_id), str(orientation), int(current_state))
        state_key = int(current_state)
        if cell_key not in self._compiled_cells:
            self._compiled_cells[cell_key] = self._compile(self._occurrence_cells.get(cell_key, []))
        if state_key not in self._compiled_states:
            self._compiled_states[state_key] = self._compile(
                self._occurrence_states.get(state_key, [])
            )
        cell = self._compiled_cells[cell_key]
        pooled = self._compiled_states[state_key]
        cell_rate = _beta_rate(
            self._match_count(cell, observed_transitions, bars_since_anchor),
            len(cell),
            self.config.smoothing_alpha,
        )
        pooled_rate = _beta_rate(
            self._match_count(pooled, observed_transitions, bars_since_anchor),
            len(pooled),
            self.config.smoothing_alpha,
        )
        weight = len(cell) / (len(cell) + self.config.pooling_strength)
        probability = weight * cell_rate + (1.0 - weight) * pooled_rate
        return max(self.config.minimum_likelihood, min(1.0, probability))
