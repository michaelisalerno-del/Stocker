"""Immutable chronological phases for quiet-state and control observations."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from stocker_prospective.contract import claims_boundary
from stocker_prospective.recorder_repository import FrozenRecorderRepository

QuietObservationKind = Literal[
    "quiet_bottom_10",
    "neutral_control",
    "high_tail_control",
]


class QuietObservationCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frozen_m1c_prediction_valid: bool
    underlying_entry_observed: bool
    underlying_quote_window_finalised: bool
    option_selection_attempted: bool
    required_option_quote_windows_finalised: bool
    data_gaps_accounted_for: bool

    @classmethod
    def all_valid(cls) -> QuietObservationCompletion:
        return cls(
            frozen_m1c_prediction_valid=True,
            underlying_entry_observed=True,
            underlying_quote_window_finalised=True,
            option_selection_attempted=True,
            required_option_quote_windows_finalised=True,
            data_gaps_accounted_for=True,
        )

    @property
    def complete(self) -> bool:
        return all(self.model_dump().values())


class QuietPhaseAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    observation_kind: QuietObservationKind
    occurred_at_utc: datetime
    complete_quiet_episode_ordinal: int | None
    phase: str
    target_dependent_selection_opened: bool
    scientific_evidence_claim_allowed: bool
    completion: QuietObservationCompletion
    claims_boundary: dict[str, bool | float | str]

    @field_validator("occurred_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quiet phase timestamp must be timezone-aware")
        return value.astimezone(UTC)


class QuietStatePhaseLedger:
    """Assign 30 shakedown, 150 development, and 150 unopened confirmation episodes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._assignments = self._load()

    def _load(self) -> list[QuietPhaseAssignment]:
        if not self.path.is_file():
            return []
        rows = [
            QuietPhaseAssignment.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            current.occurred_at_utc < previous.occurred_at_utc
            for previous, current in zip(rows, rows[1:], strict=False)
        ):
            raise ValueError("quiet phase ledger chronology is invalid")
        if len({row.observation_id for row in rows}) != len(rows):
            raise ValueError("quiet phase observation identity is not unique")
        return rows

    @staticmethod
    def _phase(ordinal: int) -> str:
        if ordinal <= 30:
            return "engineering_shakedown"
        if ordinal <= 180:
            return "quiet_state_development"
        if ordinal <= 330:
            return "quiet_state_confirmation"
        return "confirmation_complete_collection_continues"

    def record(
        self,
        *,
        observation_id: str,
        observation_kind: QuietObservationKind,
        occurred_at: datetime,
        completion: QuietObservationCompletion,
        cohort_phase: tuple[str, bool] | None = None,
    ) -> QuietPhaseAssignment:
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("quiet phase timestamp must be timezone-aware")
        observed = occurred_at.astimezone(UTC)
        existing = next(
            (row for row in self._assignments if row.observation_id == observation_id),
            None,
        )
        if existing is not None:
            if (
                existing.occurred_at_utc != observed
                or existing.observation_kind != observation_kind
                or existing.completion != completion
            ):
                raise ValueError("quiet phase assignment is immutable")
            return existing
        if self._assignments and observed < self._assignments[-1].occurred_at_utc:
            raise ValueError("quiet phase ledger must be chronological")
        if cohort_phase is not None:
            phase, evidence_allowed = cohort_phase
            if phase == "engineering_transfer" and evidence_allowed:
                raise ValueError("engineering-transfer observations cannot be scientific evidence")
            if (
                observation_kind == "quiet_bottom_10"
                and completion.complete
                and phase != "engineering_transfer"
            ):
                ordinal = (
                    sum(
                        row.complete_quiet_episode_ordinal is not None and row.phase == phase
                        for row in self._assignments
                    )
                    + 1
                )
            else:
                ordinal = None
        else:
            completed_quiet = sum(
                row.complete_quiet_episode_ordinal is not None for row in self._assignments
            )
            if observation_kind == "quiet_bottom_10" and completion.complete:
                ordinal = completed_quiet + 1
                phase = self._phase(ordinal)
            else:
                ordinal = None
                phase = self._phase(completed_quiet + 1)
            evidence_allowed = False
        assignment = QuietPhaseAssignment(
            observation_id=observation_id,
            observation_kind=observation_kind,
            occurred_at_utc=observed,
            complete_quiet_episode_ordinal=ordinal,
            phase=phase,
            target_dependent_selection_opened=False,
            scientific_evidence_claim_allowed=evidence_allowed,
            completion=completion,
            claims_boundary=claims_boundary(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    assignment.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._assignments.append(assignment)
        return assignment


class QuietStatePhaseManager:
    """Mirror one immutable quiet/control phase assignment into SQLite."""

    def __init__(
        self,
        *,
        ledger: QuietStatePhaseLedger,
        repository: FrozenRecorderRepository,
        phase_resolver: Callable[[datetime], tuple[str, bool]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.repository = repository
        self.phase_resolver = phase_resolver

    def finalise(
        self,
        *,
        observation_id: str,
        observation_kind: QuietObservationKind,
        occurred_at: datetime,
        completion: QuietObservationCompletion,
    ) -> QuietPhaseAssignment:
        assignment = self.ledger.record(
            observation_id=observation_id,
            observation_kind=observation_kind,
            occurred_at=occurred_at,
            completion=completion,
            cohort_phase=(
                None if self.phase_resolver is None else self.phase_resolver(occurred_at)
            ),
        )
        self.repository.finalise_quiet_observation(
            observation_id=observation_id,
            phase=assignment.phase,
            completion_status=("complete" if completion.complete else "incomplete"),
            completed_at_utc=occurred_at,
        )
        return assignment


__all__ = [
    "QuietObservationCompletion",
    "QuietObservationKind",
    "QuietPhaseAssignment",
    "QuietStatePhaseLedger",
    "QuietStatePhaseManager",
]
