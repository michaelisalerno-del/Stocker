"""Right-censored discrete state-duration fitting for regime research V2.

Terminal session runs contribute exposure but never a fabricated exit.
Source-gap and unavailable-session runs are excluded from the primary fit.
This module is structural research only and has no execution surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False
PART_B_INTERACTION_SCORING_ENABLED = False
SEMANTIC_DICTIONARY_PROMOTION_ENABLED = False


class RunEndingStatus(StrEnum):
    """Mutually exclusive observed ending status for a training state run."""

    OBSERVED_STATE_EXIT = "OBSERVED_STATE_EXIT"
    RIGHT_CENSORED_SESSION_END = "RIGHT_CENSORED_SESSION_END"
    INVALIDATED_BY_SOURCE_GAP = "INVALIDATED_BY_SOURCE_GAP"
    INCOMPLETE_OR_UNAVAILABLE_SESSION = "INCOMPLETE_OR_UNAVAILABLE_SESSION"
    DATA_ERROR = "DATA_ERROR"


@dataclass(frozen=True, slots=True)
class DurationFitConfig:
    """Frozen, outcome-independent duration smoothing and tail support."""

    maximum_age: int = 78
    alpha: float = 0.5
    beta: float = 0.5
    minimum_state_at_risk: int = 5
    tail_prior_hazard: float = 0.05

    def __post_init__(self) -> None:
        if self.maximum_age <= 0 or self.maximum_age > 78:
            raise ValueError("maximum_age must be in [1, 78]")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Beta smoothing parameters must be positive")
        if self.minimum_state_at_risk <= 0:
            raise ValueError("minimum_state_at_risk must be positive")
        if not 0.0 < self.tail_prior_hazard < 1.0:
            raise ValueError("tail_prior_hazard must lie strictly inside (0, 1)")


@dataclass(frozen=True, slots=True)
class RightCensoredDurationFit:
    """Complete count, effective-hazard, and survival surface."""

    at_risk: np.ndarray
    exits: np.ndarray
    censored: np.ndarray
    raw_hazard: np.ndarray
    hazard: np.ndarray
    survival: np.ndarray
    effective_sample_size: np.ndarray
    backoff_weight: np.ndarray
    support_status: np.ndarray
    config: DurationFitConfig

    def validate(self) -> None:
        shape = self.at_risk.shape
        arrays = (
            self.exits,
            self.censored,
            self.raw_hazard,
            self.hazard,
            self.survival,
            self.effective_sample_size,
            self.backoff_weight,
            self.support_status,
        )
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise AssertionError("duration-fit surfaces have inconsistent shapes")
        if shape[1] != self.config.maximum_age:
            raise AssertionError("duration-fit support differs from configuration")
        if np.any(self.at_risk < 0) or np.any(self.exits < 0) or np.any(self.censored < 0):
            raise AssertionError("duration counts cannot be negative")
        if np.any(self.exits > self.at_risk):
            raise AssertionError("exits exceed at-risk exposure")
        if not np.isfinite(self.hazard).all() or np.any((self.hazard < 0.0) | (self.hazard >= 1.0)):
            raise AssertionError("duration hazards must be finite in [0, 1)")
        if not np.isfinite(self.survival).all() or np.any(self.survival < 0.0):
            raise AssertionError("survival must be finite and nonnegative")
        if np.any(np.diff(self.survival, axis=1) > 1e-15):
            raise AssertionError("survival must be non-increasing")
        previous = np.c_[np.ones(shape[0]), self.survival[:, :-1]]
        exact_mass = previous * self.hazard
        total = exact_mass.sum(axis=1) + self.survival[:, -1]
        if not np.allclose(total, 1.0, atol=1e-12):
            raise AssertionError("duration probability mass does not normalize")

    def counts_frame(self) -> pd.DataFrame:
        """Return deterministic long-form duration evidence."""

        rows: list[dict[str, Any]] = []
        for state in range(self.at_risk.shape[0]):
            for age_index in range(self.at_risk.shape[1]):
                rows.append(
                    {
                        "state": state,
                        "age": age_index + 1,
                        "at_risk": int(self.at_risk[state, age_index]),
                        "exits": int(self.exits[state, age_index]),
                        "censored": int(self.censored[state, age_index]),
                        "effective_sample_size": float(
                            self.effective_sample_size[state, age_index]
                        ),
                        "raw_hazard": float(self.raw_hazard[state, age_index]),
                        "hazard": float(self.hazard[state, age_index]),
                        "survival": float(self.survival[state, age_index]),
                        "backoff_weight": float(self.backoff_weight[state, age_index]),
                        "support_status": str(self.support_status[state, age_index]),
                    }
                )
        return pd.DataFrame(rows)


def _required_run_columns(frame: pd.DataFrame) -> None:
    required = {
        "symbol",
        "session",
        "segment_id",
        "segment_index",
        "bar_ordinal",
        "bar_start_timestamp",
        "state",
        "session_source_complete",
        "expected_session_bars",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training bar frame lacks run columns: {missing}")


def classify_training_run_endings(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress state labels and classify every run ending without hindsight."""

    _required_run_columns(frame)
    if frame.empty:
        raise ValueError("training bar frame cannot be empty")
    bars = frame.copy()
    bars["symbol"] = bars["symbol"].astype(str)
    bars["session"] = bars["session"].astype(str)
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_ordinal"] = pd.to_numeric(bars["bar_ordinal"], errors="raise").astype(int)
    bars["state"] = pd.to_numeric(bars["state"], errors="raise").astype(int)
    bars = bars.sort_values(
        ["symbol", "session", "bar_start_timestamp", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    if bars[["symbol", "session", "bar_ordinal"]].duplicated().any():
        raise ValueError("duplicate training bar natural key")
    if bars["state"].lt(0).any():
        raise ValueError("state labels must be nonnegative")
    if bars["bar_ordinal"].lt(0).any() or bars["bar_ordinal"].gt(77).any():
        raise ValueError("impossible training bar ordinal")

    rows: list[dict[str, Any]] = []
    run_number = 0
    for (symbol, session), session_frame in bars.groupby(["symbol", "session"], sort=True):
        session_positions = session_frame.index.to_numpy(dtype=int)
        complete_values = session_frame["session_source_complete"].astype(bool).unique()
        expected_values = session_frame["expected_session_bars"].astype(int).unique()
        if len(complete_values) != 1 or len(expected_values) != 1:
            raise ValueError("inconsistent session completeness metadata")
        complete = bool(complete_values[0])
        expected_count = int(expected_values[0])
        if expected_count <= 0 or expected_count > 78:
            raise ValueError("impossible expected session duration")
        segment_count = int(session_frame["segment_id"].nunique())
        has_gap = segment_count > 1
        if complete and has_gap:
            raise ValueError("source-complete session cannot contain a gap")

        segment_groups = list(session_frame.groupby("segment_id", sort=False))
        for segment_ordinal, (segment_id, segment_frame) in enumerate(segment_groups):
            positions = segment_frame.index.to_numpy(dtype=int)
            local_states = bars.loc[positions, "state"].to_numpy(dtype=int)
            starts = np.r_[0, np.flatnonzero(local_states[1:] != local_states[:-1]) + 1]
            ends = np.r_[starts[1:], len(local_states)]
            for local_run_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                first = int(positions[int(start)])
                last = int(positions[int(end) - 1])
                duration = int(end - start)
                if duration <= 0 or duration > 78:
                    raise ValueError("impossible state-run duration")

                is_segment_terminal = int(end) == len(local_states)
                segment_starts_after_gap = segment_ordinal > 0
                left_boundary_unobserved = local_run_index == 0 and (
                    segment_starts_after_gap or int(cast(Any, bars.at[first, "bar_ordinal"])) > 0
                )
                segment_ends_at_gap = (
                    is_segment_terminal and segment_ordinal < len(segment_groups) - 1
                )
                if left_boundary_unobserved or segment_ends_at_gap:
                    status = RunEndingStatus.INVALIDATED_BY_SOURCE_GAP
                    eligible = False
                    reason = (
                        "left_truncated_by_source_gap_or_missing_open"
                        if left_boundary_unobserved
                        else "run_reaches_structural_source_gap"
                    )
                elif not complete:
                    status = RunEndingStatus.INCOMPLETE_OR_UNAVAILABLE_SESSION
                    eligible = False
                    reason = "session_completeness_not_proven"
                elif not is_segment_terminal:
                    status = RunEndingStatus.OBSERVED_STATE_EXIT
                    eligible = True
                    reason = None
                else:
                    status = RunEndingStatus.RIGHT_CENSORED_SESSION_END
                    eligible = True
                    reason = None

                rows.append(
                    {
                        "run_id": f"training_run_{run_number:08d}",
                        "symbol": str(symbol),
                        "session": str(session),
                        "segment_id": str(segment_id),
                        "segment_index": int(cast(Any, bars.at[first, "segment_index"])),
                        "run_index_in_segment": local_run_index,
                        "state": int(cast(Any, bars.at[first, "state"])),
                        "duration": duration,
                        "start_position": first,
                        "end_position": last,
                        "start_bar_ordinal": int(cast(Any, bars.at[first, "bar_ordinal"])),
                        "end_bar_ordinal": int(cast(Any, bars.at[last, "bar_ordinal"])),
                        "start_timestamp": bars.at[first, "bar_start_timestamp"],
                        "end_timestamp": bars.at[last, "bar_start_timestamp"],
                        "ending_status": status.value,
                        "primary_fit_eligible": eligible,
                        "exclusion_reason": reason,
                        "session_source_complete": complete,
                        "expected_session_bars": expected_count,
                        "source_gap_in_session": has_gap,
                        "session_position_count": len(session_positions),
                    }
                )
                run_number += 1
    output = pd.DataFrame(rows)
    if output.empty:
        raise AssertionError("run classification produced no runs")
    return output


def estimate_right_censored_durations(
    run_ledger: pd.DataFrame,
    *,
    state_count: int,
    config: DurationFitConfig,
) -> RightCensoredDurationFit:
    """Estimate exact ages 1 through maximum_age with deterministic tail backoff."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    required = {"state", "duration", "ending_status", "primary_fit_eligible"}
    missing = sorted(required.difference(run_ledger.columns))
    if missing:
        raise ValueError(f"run ledger lacks duration columns: {missing}")

    eligible = run_ledger.loc[run_ledger["primary_fit_eligible"].astype(bool)].copy()
    allowed = {
        RunEndingStatus.OBSERVED_STATE_EXIT.value,
        RunEndingStatus.RIGHT_CENSORED_SESSION_END.value,
    }
    if not set(eligible["ending_status"].astype(str)).issubset(allowed):
        raise ValueError("eligible duration row has a non-primary ending status")
    eligible["state"] = pd.to_numeric(eligible["state"], errors="raise").astype(int)
    eligible["duration"] = pd.to_numeric(eligible["duration"], errors="raise").astype(int)
    if eligible["state"].lt(0).any() or eligible["state"].ge(state_count).any():
        raise ValueError("duration state lies outside declared state count")
    if eligible["duration"].lt(1).any() or eligible["duration"].gt(config.maximum_age).any():
        raise ValueError("duration lies outside exact configured support")

    shape = (state_count, config.maximum_age)
    at_risk = np.zeros(shape, dtype=np.int64)
    exits = np.zeros(shape, dtype=np.int64)
    censored = np.zeros(shape, dtype=np.int64)
    for row in eligible.itertuples(index=False):
        state = int(cast(Any, row.state))
        duration = int(cast(Any, row.duration))
        at_risk[state, :duration] += 1
        if str(row.ending_status) == RunEndingStatus.OBSERVED_STATE_EXIT.value:
            exits[state, duration - 1] += 1
        else:
            censored[state, duration - 1] += 1

    raw_hazard = (exits + config.alpha) / (at_risk + config.alpha + config.beta)
    pooled_risk = at_risk.sum(axis=0)
    pooled_exits = exits.sum(axis=0)
    pooled_hazard = (pooled_exits + config.alpha) / (pooled_risk + config.alpha + config.beta)
    pooled_reliability = pooled_risk / (pooled_risk + float(config.minimum_state_at_risk))
    backoff_target = (
        pooled_reliability * pooled_hazard + (1.0 - pooled_reliability) * config.tail_prior_hazard
    )
    backoff_weight = np.clip(
        (config.minimum_state_at_risk - at_risk) / float(config.minimum_state_at_risk),
        0.0,
        1.0,
    )
    hazard = (1.0 - backoff_weight) * raw_hazard + backoff_weight * backoff_target[None, :]
    epsilon = np.finfo(np.float64).eps
    hazard = np.clip(hazard, 0.0, 1.0 - epsilon)
    survival = np.cumprod(1.0 - hazard, axis=1)
    effective_sample_size = at_risk.astype(float)
    support_status = np.full(shape, "state_supported", dtype=object)
    support_status[backoff_weight > 0.0] = "hierarchical_backoff"
    support_status[at_risk == 0] = "tail_prior_backoff"

    fit = RightCensoredDurationFit(
        at_risk=at_risk,
        exits=exits,
        censored=censored,
        raw_hazard=raw_hazard,
        hazard=hazard,
        survival=survival,
        effective_sample_size=effective_sample_size,
        backoff_weight=backoff_weight,
        support_status=support_status,
        config=config,
    )
    fit.validate()
    return fit


__all__ = [
    "DurationFitConfig",
    "RightCensoredDurationFit",
    "RunEndingStatus",
    "classify_training_run_endings",
    "estimate_right_censored_durations",
]
