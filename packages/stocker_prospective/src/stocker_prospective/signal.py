"""Deterministic, restart-idempotent threshold-crossing eventisation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from stocker_prospective.database import ProspectiveRepository, ScoreInput


class SignalResult(BaseModel):
    status: Literal[
        "rejected",
        "below_threshold",
        "startup_above_threshold",
        "crossing",
        "above_threshold",
    ]
    score_id: int
    episode_id: str | None


class SignalEventizer:
    """Turn eligible completed-bar scores into below-to-above episodes."""

    def __init__(self, repository: ProspectiveRepository) -> None:
        self.repository = repository

    def record(self, score: ScoreInput) -> SignalResult:
        stored = self.repository.record_score(score)
        if not score.eligibility or score.m1_probability is None:
            result = SignalResult(status="rejected", score_id=stored.id, episode_id=None)
            self.repository.record_eventization(stored, status=result.status, episode_id=None)
            return result
        above = score.m1_probability >= score.frozen_threshold
        previous = self.repository.previous_eligible_score(score)
        if not above:
            result = SignalResult(
                status="below_threshold",
                score_id=stored.id,
                episode_id=None,
            )
            self.repository.record_eventization(stored, status=result.status, episode_id=None)
            return result
        if previous is None:
            result = SignalResult(
                status="startup_above_threshold",
                score_id=stored.id,
                episode_id=None,
            )
            self.repository.record_eventization(stored, status=result.status, episode_id=None)
            return result
        previous_probability = previous["m1_probability"]
        previous_threshold = previous["frozen_threshold"]
        previous_above = previous_probability is not None and float(previous_probability) >= float(
            previous_threshold
        )
        if not previous_above:
            crossing_episode_id = self.repository.create_signal_episode(stored)
            result = SignalResult(
                status="crossing",
                score_id=stored.id,
                episode_id=crossing_episode_id,
            )
            self.repository.record_eventization(
                stored,
                status=result.status,
                episode_id=crossing_episode_id,
            )
            return result
        latest_episode_id = self.repository.latest_signal_episode(score)
        if latest_episode_id is not None:
            self.repository.add_signal_checkpoint(latest_episode_id, stored)
        result = SignalResult(
            status="above_threshold",
            score_id=stored.id,
            episode_id=latest_episode_id,
        )
        self.repository.record_eventization(
            stored,
            status=result.status,
            episode_id=latest_episode_id,
        )
        return result
