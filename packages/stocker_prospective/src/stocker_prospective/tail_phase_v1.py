"""Stock-local, causal route-state bookkeeping for M1C Tail Phase V1."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from stocker_prospective.contract import M1C_FROZEN_THRESHOLD
from stocker_prospective.m1c_features import FROZEN_CHECKPOINTS

M1C_TAIL_PHASE_V1_VERSION: Final[str] = "m1c-tail-phase-v1"
MOVEMENT_CONSUMED_LOOKBACK_MINUTES_V1: Final[int] = 15
MOVEMENT_CONSUMED_MEDIAN_2024_V1: Final[float] = 1.3986941389121161
MOVEMENT_CONSUMED_MEDIAN_2024_OBSERVATIONS_V1: Final[int] = 18_596
FIVE_MINUTE_BAR_MINUTES: Final[int] = 5
CHECKPOINT_SPACING_MINUTES: Final[int] = 10
PROTECTED_HISTORICAL_START_V1: Final[date] = date(2026, 1, 1)
TAIL_PHASE_BOOTSTRAP_SEED_V1: Final[int] = 20260728
TAIL_PHASE_BOOTSTRAP_DRAWS_V1: Final[int] = 1000

TailPhaseV1 = Literal[
    "FIRST_ENTRY",
    "PERSISTENT",
    "RE_ENTRY",
    "OUTSIDE_TAIL",
    "UNKNOWN_INCOMPLETE",
]
MovementConsumedBucketV1 = Literal[
    "LOW_OR_EQUAL",
    "HIGH",
    "UNKNOWN_INCOMPLETE",
]

# This explicit deny-list is also emitted in the V1 provenance artifacts.  No
# value under any of these names is accepted by either public calculation.
FORBIDDEN_TAIL_PHASE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "signed_pressure",
        "tension",
        "future_filtered_peer_slate",
        "peer_normalised_feature",
        "peer_normalized_feature",
        "eligible_stocks_in_session",
        "future_dependent_checkpoint_membership",
        "sequential_row_weight",
    }
)


class TailPhaseStateV1(BaseModel):
    """One immutable phase assignment at a scheduled frozen M1C checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    m1c_high_tail_v1: bool | None
    m1c_tail_phase_v1: TailPhaseV1
    tail_entry_number_v1: int | None
    tail_run_length_checkpoints_v1: int | None
    tail_run_age_minutes_v1: float | None
    prior_tail_entries_v1: int | None
    previous_checkpoint_above_tail_v1: bool | None
    minutes_since_previous_tail_exit_v1: float | None
    phase_history_complete_v1: bool
    phase_missing_reason_v1: str | None


class MovementConsumedBarV1(BaseModel):
    """The minimal stock-local bar surface accepted by the consumed calculator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    bar_ordinal: int
    bar_start_timestamp: datetime
    bar_complete_timestamp: datetime
    high: float
    low: float
    finalised: bool

    @field_validator("bar_start_timestamp", "bar_complete_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("movement-consumed bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


class MovementConsumedStateV1(BaseModel):
    """One causal pre-trigger range divided by a prior-close IV expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    movement_consumed_v1: float | None
    movement_consumed_numerator_v1: float | None
    movement_consumed_denominator_v1: float | None
    movement_consumed_complete_v1: bool
    movement_consumed_missing_reason_v1: str | None


class TailPhaseDateRangeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date | None


class TailPhaseChronologyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development: TailPhaseDateRangeV1
    assessment: TailPhaseDateRangeV1
    stress: TailPhaseDateRangeV1
    protected: TailPhaseDateRangeV1


class MovementConsumedMedianProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date
    complete_observations: int
    predictor_values_only: bool


class TailPhaseModelIdentifiersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    movement: str
    direction: str


class TailPhaseBootstrapV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    draws: int
    confidence_level: float


class TailPhaseFrozenConfigV1(BaseModel):
    """Versioned predictor-only split and chronology consumed by live logging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-tail-phase-v1"]
    research_id: Literal["M1C Tail Phase V1"]
    m1c_threshold: float
    frozen_checkpoints: tuple[int, ...]
    underlying_bar_minutes: int
    movement_consumed_lookback_minutes: int
    movement_consumed_median_2024: float
    movement_consumed_median_provenance: MovementConsumedMedianProvenanceV1
    chronology: TailPhaseChronologyV1
    model_identifiers: TailPhaseModelIdentifiersV1
    bootstrap: TailPhaseBootstrapV1

    @model_validator(mode="after")
    def _binding_contract(self) -> TailPhaseFrozenConfigV1:
        expected_ranges = {
            "development": (date(2024, 1, 1), date(2024, 12, 31)),
            "assessment": (date(2025, 1, 1), date(2025, 8, 22)),
            "stress": (date(2025, 9, 1), date(2025, 12, 31)),
            "protected": (PROTECTED_HISTORICAL_START_V1, None),
        }
        observed_ranges = {
            name: (
                cast(TailPhaseDateRangeV1, getattr(self.chronology, name)).start,
                cast(TailPhaseDateRangeV1, getattr(self.chronology, name)).end,
            )
            for name in expected_ranges
        }
        if self.m1c_threshold != M1C_FROZEN_THRESHOLD:
            raise ValueError("frozen M1C threshold differs")
        if self.frozen_checkpoints != FROZEN_CHECKPOINTS:
            raise ValueError("frozen checkpoint grid differs")
        if (
            self.underlying_bar_minutes != FIVE_MINUTE_BAR_MINUTES
            or self.movement_consumed_lookback_minutes != MOVEMENT_CONSUMED_LOOKBACK_MINUTES_V1
        ):
            raise ValueError("movement-consumed window contract differs")
        if self.movement_consumed_median_2024 != MOVEMENT_CONSUMED_MEDIAN_2024_V1:
            raise ValueError("frozen movement-consumed median differs")
        provenance = self.movement_consumed_median_provenance
        if (
            provenance.start != date(2024, 1, 1)
            or provenance.end != date(2024, 12, 31)
            or provenance.complete_observations != MOVEMENT_CONSUMED_MEDIAN_2024_OBSERVATIONS_V1
            or not provenance.predictor_values_only
        ):
            raise ValueError("movement-consumed median provenance differs")
        if observed_ranges != expected_ranges:
            raise ValueError("tail-phase chronology differs")
        if (
            self.bootstrap.seed != TAIL_PHASE_BOOTSTRAP_SEED_V1
            or self.bootstrap.draws != TAIL_PHASE_BOOTSTRAP_DRAWS_V1
            or self.bootstrap.confidence_level != 0.95
        ):
            raise ValueError("tail-phase bootstrap contract differs")
        if not self.model_identifiers.movement.startswith("M1C/"):
            raise ValueError("M1C model identifier differs")
        if not self.model_identifiers.direction.startswith("A1/"):
            raise ValueError("A1 model identifier differs")
        return self


def load_tail_phase_frozen_config_v1(
    path: str | Path,
) -> TailPhaseFrozenConfigV1:
    """Load and verify the exact V1 configuration used by prospective logging."""

    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return TailPhaseFrozenConfigV1.model_validate(payload)


@dataclass(frozen=True)
class _CheckpointObservation:
    checkpoint: int
    timestamp: datetime
    probability: float | None
    valid: bool
    invalid_reason: str | None

    @property
    def above(self) -> bool | None:
        if self.probability is None:
            return None
        return self.probability >= M1C_FROZEN_THRESHOLD


class TailPhaseTrackerV1:
    """Assign V1 phase without bridging a missing or invalid scheduled checkpoint."""

    def __init__(self) -> None:
        self._observations: dict[
            tuple[str, date],
            dict[int, _CheckpointObservation],
        ] = {}
        self._results: dict[tuple[str, date, int], TailPhaseStateV1] = {}

    def evaluate(
        self,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
        causal_timestamp: datetime,
        probability: float | None,
        valid: bool = True,
        invalid_reason: str | None = None,
    ) -> TailPhaseStateV1:
        """Record and classify exactly one stock/session/checkpoint observation."""

        value = int(checkpoint)
        if value not in FROZEN_CHECKPOINTS:
            raise ValueError(f"checkpoint outside frozen M1C grid: {value}")
        if causal_timestamp.tzinfo is None or causal_timestamp.utcoffset() is None:
            raise ValueError("tail-phase causal timestamp must be timezone-aware")
        timestamp = causal_timestamp.astimezone(UTC)
        numeric_probability: float | None
        if probability is None:
            numeric_probability = None
            valid = False
            invalid_reason = invalid_reason or "m1c_probability_missing"
        else:
            numeric_probability = float(probability)
            if not math.isfinite(numeric_probability) or not 0.0 <= numeric_probability <= 1.0:
                raise ValueError("M1C probability must be finite and lie in [0, 1]")
        if not valid and not invalid_reason:
            invalid_reason = "checkpoint_invalid"

        observation = _CheckpointObservation(
            checkpoint=value,
            timestamp=timestamp,
            probability=numeric_probability,
            valid=bool(valid),
            invalid_reason=invalid_reason,
        )
        identity = (symbol, session, value)
        existing_result = self._results.get(identity)
        rows = self._observations.setdefault((symbol, session), {})
        if existing_result is not None:
            if rows[value] != observation:
                raise ValueError("immutable tail-phase checkpoint differs")
            return existing_result

        out_of_order = bool(rows and value < max(rows))
        timestamp_out_of_order = any(
            (prior_checkpoint < value and prior.timestamp >= timestamp)
            or (prior_checkpoint > value and prior.timestamp <= timestamp)
            for prior_checkpoint, prior in rows.items()
        )
        rows[value] = observation
        result = self._classify(
            rows=rows,
            checkpoint=value,
            out_of_order=out_of_order,
            timestamp_out_of_order=timestamp_out_of_order,
        )
        self._results[identity] = result
        return result

    @staticmethod
    def _classify(
        *,
        rows: dict[int, _CheckpointObservation],
        checkpoint: int,
        out_of_order: bool,
        timestamp_out_of_order: bool,
    ) -> TailPhaseStateV1:
        current = rows[checkpoint]
        scheduled = tuple(value for value in FROZEN_CHECKPOINTS if value <= checkpoint)
        prior_scheduled = scheduled[:-1]
        missing = tuple(value for value in prior_scheduled if value not in rows)
        invalid = tuple(
            value for value in prior_scheduled if value in rows and not rows[value].valid
        )
        timestamp_gaps = tuple(
            (left, right)
            for left, right in zip(scheduled, scheduled[1:], strict=False)
            if left in rows
            and right in rows
            and rows[left].valid
            and rows[right].valid
            and rows[right].timestamp - rows[left].timestamp
            != timedelta(minutes=CHECKPOINT_SPACING_MINUTES)
        )
        history_complete = bool(
            current.valid
            and not missing
            and not invalid
            and not timestamp_gaps
            and not out_of_order
            and not timestamp_out_of_order
        )
        above = current.above
        previous_checkpoint = (
            None
            if checkpoint == FROZEN_CHECKPOINTS[0]
            else FROZEN_CHECKPOINTS[FROZEN_CHECKPOINTS.index(checkpoint) - 1]
        )
        previous = None if previous_checkpoint is None else rows.get(previous_checkpoint)
        previous_above: bool | None
        if previous_checkpoint is None:
            previous_above = False
        elif previous is None or not previous.valid:
            previous_above = None
        else:
            previous_above = previous.above

        blocking_reason: str | None = None
        if out_of_order:
            blocking_reason = "checkpoint_out_of_order"
        elif timestamp_out_of_order:
            blocking_reason = "causal_timestamp_out_of_order"
        elif not current.valid:
            blocking_reason = "current_checkpoint_invalid:" + cast(str, current.invalid_reason)
        elif previous_checkpoint is not None and previous is None and above:
            blocking_reason = f"missing_immediately_preceding_checkpoint:{previous_checkpoint}"
        elif (
            previous_checkpoint is not None
            and previous is not None
            and not previous.valid
            and above
        ):
            blocking_reason = f"invalid_immediately_preceding_checkpoint:{previous_checkpoint}"
        elif (
            previous_checkpoint is not None
            and previous is not None
            and previous.valid
            and rows[checkpoint].timestamp - previous.timestamp
            != timedelta(minutes=CHECKPOINT_SPACING_MINUTES)
            and above
        ):
            blocking_reason = (
                f"non_contiguous_checkpoint_timestamps:{previous_checkpoint}-{checkpoint}"
            )

        phase: TailPhaseV1
        if blocking_reason is not None:
            phase = "UNKNOWN_INCOMPLETE"
        elif above is False:
            phase = "OUTSIDE_TAIL"
        elif previous_checkpoint is None:
            phase = "FIRST_ENTRY"
        elif previous_above is True:
            phase = "PERSISTENT"
        elif previous_above is False:
            earlier_high = any(
                rows[value].valid and rows[value].above is True
                for value in FROZEN_CHECKPOINTS
                if value < previous_checkpoint and value in rows
            )
            if earlier_high:
                phase = "RE_ENTRY"
            elif history_complete:
                phase = "FIRST_ENTRY"
            else:
                phase = "UNKNOWN_INCOMPLETE"
                incomplete = (*missing, *invalid)
                blocking_reason = "earlier_checkpoint_history_incomplete:" + ",".join(
                    str(value) for value in sorted(set(incomplete))
                )
        else:
            phase = "UNKNOWN_INCOMPLETE"
            blocking_reason = "immediately_preceding_checkpoint_state_unknown"

        general_missing_reason = blocking_reason
        if general_missing_reason is None and not history_complete:
            if missing:
                general_missing_reason = "earlier_checkpoint_history_incomplete:" + ",".join(
                    str(value) for value in missing
                )
            elif invalid:
                general_missing_reason = "earlier_checkpoint_history_invalid:" + ",".join(
                    str(value) for value in invalid
                )
            elif timestamp_gaps:
                general_missing_reason = "non_contiguous_checkpoint_timestamps:" + ",".join(
                    f"{left}-{right}" for left, right in timestamp_gaps
                )

        entry_number: int | None = None
        prior_entries: int | None = None
        run_length: int | None
        run_age: float | None
        if phase == "OUTSIDE_TAIL":
            run_length = 0
            run_age = 0.0
        elif phase == "UNKNOWN_INCOMPLETE":
            run_length = None
            run_age = None
        else:
            run_checkpoints = [checkpoint]
            cursor = checkpoint
            while cursor != FROZEN_CHECKPOINTS[0]:
                candidate = FROZEN_CHECKPOINTS[FROZEN_CHECKPOINTS.index(cursor) - 1]
                candidate_row = rows.get(candidate)
                cursor_row = rows[cursor]
                if (
                    candidate_row is None
                    or not candidate_row.valid
                    or candidate_row.above is not True
                    or cursor_row.timestamp - candidate_row.timestamp
                    != timedelta(minutes=CHECKPOINT_SPACING_MINUTES)
                ):
                    break
                run_checkpoints.append(candidate)
                cursor = candidate
            first_run = rows[min(run_checkpoints)]
            run_length = len(run_checkpoints)
            run_age = (current.timestamp - first_run.timestamp).total_seconds() / 60.0

        if history_complete:
            confirmed_entries = 0
            prior_state = False
            for value in scheduled:
                observed = rows[value]
                observed_above = observed.above is True
                if observed_above and not prior_state:
                    confirmed_entries += 1
                prior_state = observed_above
            if above is True:
                entry_number = confirmed_entries
                prior_entries = confirmed_entries - 1
            else:
                prior_entries = confirmed_entries

        last_exit: datetime | None = None
        for left, right in zip(scheduled, scheduled[1:], strict=False):
            before = rows.get(left)
            after = rows.get(right)
            if (
                before is not None
                and after is not None
                and before.valid
                and after.valid
                and before.above is True
                and after.above is False
                and after.timestamp - before.timestamp
                == timedelta(minutes=CHECKPOINT_SPACING_MINUTES)
            ):
                last_exit = after.timestamp
        minutes_since_exit = (
            None if last_exit is None else (current.timestamp - last_exit).total_seconds() / 60.0
        )
        if minutes_since_exit is not None and minutes_since_exit < 0.0:
            minutes_since_exit = None

        return TailPhaseStateV1(
            m1c_high_tail_v1=above,
            m1c_tail_phase_v1=phase,
            tail_entry_number_v1=entry_number,
            tail_run_length_checkpoints_v1=run_length,
            tail_run_age_minutes_v1=run_age,
            prior_tail_entries_v1=prior_entries,
            previous_checkpoint_above_tail_v1=previous_above,
            minutes_since_previous_tail_exit_v1=minutes_since_exit,
            phase_history_complete_v1=history_complete,
            phase_missing_reason_v1=general_missing_reason,
        )


def _incomplete_consumed(
    *,
    reason: str,
    denominator: float | None,
    numerator: float | None = None,
) -> MovementConsumedStateV1:
    return MovementConsumedStateV1(
        movement_consumed_v1=None,
        movement_consumed_numerator_v1=numerator,
        movement_consumed_denominator_v1=denominator,
        movement_consumed_complete_v1=False,
        movement_consumed_missing_reason_v1=reason,
    )


def calculate_movement_consumed_v1(
    *,
    symbol: str,
    session: date,
    checkpoint: int,
    completed_bars: tuple[MovementConsumedBarV1, ...],
    previous_close_implied_movement_15m: float | None,
) -> MovementConsumedStateV1:
    """Calculate a fixed, trailing three-bar range without consulting future bars."""

    value = int(checkpoint)
    if value not in FROZEN_CHECKPOINTS:
        raise ValueError(f"checkpoint outside frozen M1C grid: {value}")
    denominator: float | None
    if previous_close_implied_movement_15m is None:
        denominator = None
    else:
        candidate = float(previous_close_implied_movement_15m)
        denominator = candidate if math.isfinite(candidate) and candidate > 0.0 else None

    required = tuple(range(value - 3, value))
    relevant = [
        bar
        for bar in completed_bars
        if bar.symbol == symbol and bar.session == session and bar.bar_ordinal in required
    ]
    by_ordinal: dict[int, MovementConsumedBarV1] = {}
    for bar in relevant:
        if bar.bar_ordinal in by_ordinal:
            return _incomplete_consumed(
                reason=f"duplicate_pretrigger_bar:{bar.bar_ordinal}",
                denominator=denominator,
            )
        by_ordinal[bar.bar_ordinal] = bar
    missing = tuple(ordinal for ordinal in required if ordinal not in by_ordinal)
    if missing:
        return _incomplete_consumed(
            reason="incomplete_pretrigger_window:" + ",".join(str(value) for value in missing),
            denominator=denominator,
        )
    ordered = [by_ordinal[ordinal] for ordinal in required]
    invalid = [
        bar.bar_ordinal
        for bar in ordered
        if not bar.finalised
        or not all(math.isfinite(number) and number > 0.0 for number in (bar.high, bar.low))
        or bar.high < bar.low
        or bar.bar_complete_timestamp - bar.bar_start_timestamp
        != timedelta(minutes=FIVE_MINUTE_BAR_MINUTES)
    ]
    if invalid:
        return _incomplete_consumed(
            reason="invalid_pretrigger_bar:" + ",".join(str(value) for value in invalid),
            denominator=denominator,
        )
    if any(
        current.bar_start_timestamp != previous.bar_complete_timestamp
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        return _incomplete_consumed(
            reason="non_contiguous_pretrigger_timestamps",
            denominator=denominator,
        )
    if denominator is None:
        reason = (
            "previous_close_implied_movement_15m_missing"
            if previous_close_implied_movement_15m is None
            else "previous_close_implied_movement_15m_invalid"
        )
        maximum_high = max(bar.high for bar in ordered)
        minimum_low = min(bar.low for bar in ordered)
        numerator = math.log(maximum_high / minimum_low)
        return _incomplete_consumed(
            reason=reason,
            denominator=None,
            numerator=(numerator if math.isfinite(numerator) and numerator >= 0.0 else None),
        )

    maximum_high = max(bar.high for bar in ordered)
    minimum_low = min(bar.low for bar in ordered)
    numerator = math.log(maximum_high / minimum_low)
    consumed = numerator / denominator
    if not math.isfinite(numerator) or numerator < 0.0 or not math.isfinite(consumed):
        return _incomplete_consumed(
            reason="movement_consumed_calculation_invalid",
            denominator=denominator,
        )
    return MovementConsumedStateV1(
        movement_consumed_v1=consumed,
        movement_consumed_numerator_v1=numerator,
        movement_consumed_denominator_v1=denominator,
        movement_consumed_complete_v1=True,
        movement_consumed_missing_reason_v1=None,
    )


def assign_movement_consumed_bucket_v1(
    value: float | None,
    *,
    frozen_median: float | None,
) -> MovementConsumedBucketV1:
    """Apply one already-frozen 2024 median; never derive a replacement split."""

    if value is None or frozen_median is None:
        return "UNKNOWN_INCOMPLETE"
    observed = float(value)
    threshold = float(frozen_median)
    if (
        not math.isfinite(observed)
        or observed < 0.0
        or not math.isfinite(threshold)
        or threshold < 0.0
    ):
        return "UNKNOWN_INCOMPLETE"
    return "LOW_OR_EQUAL" if observed <= threshold else "HIGH"


def assert_tail_phase_unprotected_sessions(sessions: Iterable[object]) -> None:
    """Fail before outcome calculation if any materialised session reaches 2026."""

    for raw in sessions:
        if isinstance(raw, datetime):
            observed = raw.date()
        elif isinstance(raw, date):
            observed = raw
        else:
            observed = date.fromisoformat(str(raw)[:10])
        if observed >= PROTECTED_HISTORICAL_START_V1:
            raise ValueError("protected 2026 session must not be read or materialised")


__all__ = [
    "CHECKPOINT_SPACING_MINUTES",
    "FORBIDDEN_TAIL_PHASE_FEATURES",
    "M1C_TAIL_PHASE_V1_VERSION",
    "MOVEMENT_CONSUMED_LOOKBACK_MINUTES_V1",
    "MOVEMENT_CONSUMED_MEDIAN_2024_OBSERVATIONS_V1",
    "MOVEMENT_CONSUMED_MEDIAN_2024_V1",
    "MovementConsumedBarV1",
    "MovementConsumedBucketV1",
    "MovementConsumedStateV1",
    "TailPhaseStateV1",
    "TailPhaseFrozenConfigV1",
    "TailPhaseTrackerV1",
    "TailPhaseV1",
    "assert_tail_phase_unprotected_sessions",
    "assign_movement_consumed_bucket_v1",
    "calculate_movement_consumed_v1",
    "load_tail_phase_frozen_config_v1",
]
