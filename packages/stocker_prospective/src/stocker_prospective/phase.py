"""Chronological immutable prospective research-phase ledger."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from stocker_prospective.contract import claims_boundary
from stocker_prospective.recorder_repository import FrozenRecorderRepository


class EpisodeCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    m1c_features_valid: bool
    directional_outputs_valid: bool
    underlying_level1_quality_passed: bool
    prospective_entry_observed: bool
    ten_minute_outcome_observed: bool
    required_quote_windows_finalised: bool
    data_gaps_accounted_for: bool

    @classmethod
    def all_valid(cls) -> EpisodeCompletion:
        return cls(
            m1c_features_valid=True,
            directional_outputs_valid=True,
            underlying_level1_quality_passed=True,
            prospective_entry_observed=True,
            ten_minute_outcome_observed=True,
            required_quote_windows_finalised=True,
            data_gaps_accounted_for=True,
        )

    @property
    def complete(self) -> bool:
        return all(self.model_dump().values())


class PhaseAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    occurred_at_utc: datetime
    complete_episode_ordinal: int | None
    phase: str
    scientific_evidence_claim_allowed: bool
    target_dependent_selection_opened: bool
    completion: EpisodeCompletion
    claims_boundary: dict[str, bool | float | str]

    @field_validator("occurred_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("phase timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ProspectivePhaseLedger:
    """Assign 30 shakedown, 100 development, then 100 unopened confirmation episodes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._assignments = self._load()

    def _load(self) -> list[PhaseAssignment]:
        if not self.path.is_file():
            return []
        rows = [
            PhaseAssignment.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            current.occurred_at_utc < previous.occurred_at_utc
            for previous, current in zip(rows, rows[1:], strict=False)
        ):
            raise ValueError("phase ledger chronology is invalid")
        if len({row.episode_id for row in rows}) != len(rows):
            raise ValueError("phase ledger episode identity is not unique")
        return rows

    @staticmethod
    def _phase(ordinal: int) -> tuple[str, bool]:
        if ordinal <= 30:
            return "engineering_shakedown", False
        if ordinal <= 130:
            return "microstructure_development", False
        if ordinal <= 230:
            return "microstructure_confirmation", False
        return "confirmation_complete_collection_continues", False

    def record(
        self,
        *,
        episode_id: str,
        occurred_at: datetime,
        completion: EpisodeCompletion,
        cohort_phase: tuple[str, bool] | None = None,
    ) -> PhaseAssignment:
        existing = next(
            (row for row in self._assignments if row.episode_id == episode_id),
            None,
        )
        if existing is not None:
            proposed_time = occurred_at.astimezone(UTC)
            if existing.occurred_at_utc != proposed_time or existing.completion != completion:
                raise ValueError("phase assignment is immutable")
            return existing
        observed = occurred_at.astimezone(UTC)
        if self._assignments and observed < self._assignments[-1].occurred_at_utc:
            raise ValueError("phase ledger must be chronological")
        if cohort_phase is not None:
            phase, evidence_allowed = cohort_phase
            if phase == "engineering_transfer" and evidence_allowed:
                raise ValueError("engineering-transfer episodes cannot be scientific evidence")
            ordinal = (
                sum(
                    item.complete_episode_ordinal is not None and item.phase == phase
                    for item in self._assignments
                )
                + 1
                if completion.complete and phase != "engineering_transfer"
                else None
            )
            if not completion.complete:
                phase = "incomplete"
                evidence_allowed = False
        elif completion.complete:
            ordinal = (
                sum(item.complete_episode_ordinal is not None for item in self._assignments) + 1
            )
            phase, evidence_allowed = self._phase(ordinal)
        else:
            ordinal = None
            phase = "incomplete"
            evidence_allowed = False
        assignment = PhaseAssignment(
            episode_id=episode_id,
            occurred_at_utc=observed,
            complete_episode_ordinal=ordinal,
            phase=phase,
            scientific_evidence_claim_allowed=evidence_allowed,
            target_dependent_selection_opened=False,
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


class ProspectivePhaseManager:
    """Finalize an episode once and mirror its immutable phase into SQLite."""

    def __init__(
        self,
        *,
        ledger: ProspectivePhaseLedger,
        repository: FrozenRecorderRepository,
        phase_resolver: Callable[[datetime], tuple[str, bool]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.repository = repository
        self.phase_resolver = phase_resolver

    def finalise(
        self,
        *,
        episode_id: str,
        occurred_at: datetime,
        completion: EpisodeCompletion,
    ) -> PhaseAssignment:
        assignment = self.ledger.record(
            episode_id=episode_id,
            occurred_at=occurred_at,
            completion=completion,
            cohort_phase=(
                None if self.phase_resolver is None else self.phase_resolver(occurred_at)
            ),
        )
        self.repository.finalise_episode(
            episode_id=episode_id,
            phase=assignment.phase,
            completion_status=("complete" if assignment.completion.complete else "incomplete"),
            completed_at_utc=occurred_at,
        )
        return assignment
