"""Frozen causal M1C quiet-state classification and prospective controls."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

BOTTOM_5_THRESHOLD: Final[float] = 0.115697407847643
BOTTOM_10_THRESHOLD: Final[float] = 0.135896965695626
BOTTOM_20_THRESHOLD: Final[float] = 0.167095528962669
HIGH_TAIL_THRESHOLD: Final[float] = 0.488333710794033
MINIMUM_EPISODE_SPACING_MINUTES: Final[int] = 30
NEUTRAL_CONTROL_SAMPLING_FRACTION: Final[float] = 0.10
NEUTRAL_CONTROL_SALT: Final[str] = "m1c-quiet-state-neutral-control-v0|20260727|frozen-10-percent"


def _probability(value: float) -> float:
    observed = float(value)
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise ValueError("M1C probability must lie in [0, 1]")
    return observed


class QuietStateSnapshot(BaseModel):
    """All frozen tail memberships for one eligible M1C checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probability: float
    previous_probability: float | None
    bottom_5: bool
    bottom_10: bool
    bottom_20: bool
    high_tail: bool
    distance_from_bottom_10: float
    model_hash: str
    feature_hash: str
    data_quality_status: str
    data_quality_flags: tuple[str, ...] = ()


def classify_quiet_state(
    *,
    probability: float,
    previous_probability: float | None,
    model_hash: str,
    feature_hash: str,
    data_quality_status: str,
    data_quality_flags: Iterable[str] = (),
) -> QuietStateSnapshot:
    """Classify a score without fitting, recalibrating, or changing thresholds."""

    observed = _probability(probability)
    previous = None if previous_probability is None else _probability(previous_probability)
    return QuietStateSnapshot(
        probability=observed,
        previous_probability=previous,
        bottom_5=observed <= BOTTOM_5_THRESHOLD,
        bottom_10=observed <= BOTTOM_10_THRESHOLD,
        bottom_20=observed <= BOTTOM_20_THRESHOLD,
        high_tail=observed >= HIGH_TAIL_THRESHOLD,
        distance_from_bottom_10=observed - BOTTOM_10_THRESHOLD,
        model_hash=model_hash,
        feature_hash=feature_hash,
        data_quality_status=data_quality_status,
        data_quality_flags=tuple(sorted(set(data_quality_flags))),
    )


class HighTailProximity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_60_minutes: bool
    following_60_minutes: bool
    any_within_60_minutes: bool


def high_tail_proximity(
    *,
    trigger_timestamp: datetime,
    high_tail_timestamps: Iterable[datetime],
) -> HighTailProximity:
    """Describe prior/following frozen high-tail observations within sixty minutes."""

    if trigger_timestamp.tzinfo is None or trigger_timestamp.utcoffset() is None:
        raise ValueError("quiet trigger timestamp must be timezone-aware")
    trigger = trigger_timestamp.astimezone(UTC)
    previous = False
    following = False
    for candidate in high_tail_timestamps:
        if candidate.tzinfo is None or candidate.utcoffset() is None:
            raise ValueError("high-tail timestamp must be timezone-aware")
        difference = (candidate.astimezone(UTC) - trigger).total_seconds() / 60.0
        previous = previous or -60.0 <= difference < 0.0
        following = following or 0.0 < difference <= 60.0
    return HighTailProximity(
        previous_60_minutes=previous,
        following_60_minutes=following,
        any_within_60_minutes=previous or following,
    )


class QuietEpisodeDecision(BaseModel):
    """One frozen bottom-10 crossing and optional fresh prospective episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    checkpoint: int
    probability: float
    previous_probability: float | None
    bottom_5: bool
    bottom_10: bool
    bottom_20: bool
    high_tail: bool
    fresh_episode: bool
    quiet_episode_id: str | None
    episode_number: int | None
    minutes_since_previous_episode: float | None
    trigger_timestamp: datetime
    prospective_entry_timestamp: datetime
    previous_high_tail_within_60_minutes: bool
    following_high_tail_within_60_minutes: bool
    data_quality_flags: tuple[str, ...]
    rejection_reason: str | None

    @field_validator("trigger_timestamp", "prospective_entry_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quiet episode timestamps must be timezone-aware")
        return value.astimezone(UTC)


class QuietEpisodeTracker:
    """Emit inclusive bottom-10 downward crossings with frozen 30-minute spacing."""

    def __init__(self) -> None:
        self._previous_eligible: dict[tuple[str, date], float] = {}
        self._previous_episode: dict[tuple[str, date], datetime] = {}
        self._episode_count: dict[tuple[str, date], int] = {}

    def restore_session(
        self,
        *,
        symbol: str,
        session: date,
        previous_eligible_probability: float | None,
        previous_episode_timestamp: datetime | None,
        episode_count: int,
    ) -> None:
        key = (symbol, session)
        if (
            key in self._previous_eligible
            or key in self._previous_episode
            or key in self._episode_count
        ):
            raise ValueError("quiet session state is already initialized")
        if previous_eligible_probability is not None:
            self._previous_eligible[key] = _probability(previous_eligible_probability)
        if previous_episode_timestamp is not None:
            if (
                previous_episode_timestamp.tzinfo is None
                or previous_episode_timestamp.utcoffset() is None
            ):
                raise ValueError("persisted quiet timestamp must be timezone-aware")
            self._previous_episode[key] = previous_episode_timestamp.astimezone(UTC)
        if episode_count < 0:
            raise ValueError("persisted quiet episode count is invalid")
        if episode_count:
            self._episode_count[key] = episode_count

    def evaluate(
        self,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
        trigger_bar_end: datetime,
        probability: float,
        eligible: bool = True,
        data_quality_flags: Iterable[str] = (),
        previous_high_tail_within_60_minutes: bool = False,
        rejection_reason: str | None = None,
    ) -> QuietEpisodeDecision:
        if trigger_bar_end.tzinfo is None or trigger_bar_end.utcoffset() is None:
            raise ValueError("quiet trigger timestamp must be timezone-aware")
        timestamp = trigger_bar_end.astimezone(UTC)
        observed = _probability(probability)
        key = (symbol, session)
        previous = self._previous_eligible.get(key)
        bottom_10 = observed <= BOTTOM_10_THRESHOLD
        crossing = eligible and bottom_10 and (previous is None or previous > BOTTOM_10_THRESHOLD)
        previous_episode = self._previous_episode.get(key)
        elapsed = (
            None
            if previous_episode is None
            else (timestamp - previous_episode).total_seconds() / 60.0
        )
        spacing_passed = elapsed is None or elapsed >= MINIMUM_EPISODE_SPACING_MINUTES
        fresh = crossing and spacing_passed
        reason = rejection_reason
        if crossing and not spacing_passed:
            reason = "minimum_episode_spacing_not_met"
        episode_id: str | None = None
        episode_number: int | None = None
        if fresh:
            episode_number = self._episode_count.get(key, 0) + 1
            raw_identity = "|".join(
                (
                    "M1C_QUIET_BOTTOM_10",
                    symbol,
                    session.isoformat(),
                    str(int(checkpoint)),
                    timestamp.isoformat(),
                )
            )
            episode_id = f"m1c-quiet-{hashlib.sha256(raw_identity.encode()).hexdigest()[:24]}"
            self._episode_count[key] = episode_number
            self._previous_episode[key] = timestamp
        if eligible:
            self._previous_eligible[key] = observed
        return QuietEpisodeDecision(
            symbol=symbol,
            session=session,
            checkpoint=int(checkpoint),
            probability=observed,
            previous_probability=previous,
            bottom_5=observed <= BOTTOM_5_THRESHOLD,
            bottom_10=bottom_10,
            bottom_20=observed <= BOTTOM_20_THRESHOLD,
            high_tail=observed >= HIGH_TAIL_THRESHOLD,
            fresh_episode=fresh,
            quiet_episode_id=episode_id,
            episode_number=episode_number,
            minutes_since_previous_episode=elapsed,
            trigger_timestamp=timestamp,
            prospective_entry_timestamp=timestamp,
            previous_high_tail_within_60_minutes=previous_high_tail_within_60_minutes,
            following_high_tail_within_60_minutes=False,
            data_quality_flags=tuple(sorted(set(data_quality_flags))),
            rejection_reason=reason,
        )


class NeutralControlDecision(BaseModel):
    """Deterministic membership decision for a neutral checkpoint control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: date
    symbol: str
    checkpoint: int
    probability: float
    model_hash: str
    population_eligible: bool
    hash_hex: str
    hash_fraction: float
    sampling_fraction: float
    selected: bool
    salt_id: str


class NeutralControlSampler:
    """Frozen ten-percent hash sampler over the neutral M1C population."""

    def __init__(
        self,
        *,
        salt: str = NEUTRAL_CONTROL_SALT,
        sampling_fraction: float = NEUTRAL_CONTROL_SAMPLING_FRACTION,
    ) -> None:
        if salt != NEUTRAL_CONTROL_SALT:
            raise ValueError("neutral-control salt is frozen")
        if sampling_fraction != NEUTRAL_CONTROL_SAMPLING_FRACTION:
            raise ValueError("neutral-control sampling fraction is frozen at ten percent")
        self.salt = salt
        self.sampling_fraction = sampling_fraction
        self.salt_id = hashlib.sha256(salt.encode()).hexdigest()

    def evaluate(
        self,
        *,
        session: date,
        symbol: str,
        checkpoint: int,
        model_hash: str,
        probability: float,
        eligible: bool,
    ) -> NeutralControlDecision:
        observed = _probability(probability)
        payload = "|".join(
            (
                self.salt,
                session.isoformat(),
                symbol,
                str(int(checkpoint)),
                model_hash,
            )
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        fraction = int(digest, 16) / float(1 << 256)
        population_eligible = (
            eligible and observed > BOTTOM_20_THRESHOLD and observed < HIGH_TAIL_THRESHOLD
        )
        return NeutralControlDecision(
            session=session,
            symbol=symbol,
            checkpoint=int(checkpoint),
            probability=observed,
            model_hash=model_hash,
            population_eligible=population_eligible,
            hash_hex=digest,
            hash_fraction=fraction,
            sampling_fraction=self.sampling_fraction,
            selected=population_eligible and fraction < self.sampling_fraction,
            salt_id=self.salt_id,
        )


__all__ = [
    "BOTTOM_5_THRESHOLD",
    "BOTTOM_10_THRESHOLD",
    "BOTTOM_20_THRESHOLD",
    "HIGH_TAIL_THRESHOLD",
    "NEUTRAL_CONTROL_SALT",
    "NEUTRAL_CONTROL_SAMPLING_FRACTION",
    "HighTailProximity",
    "NeutralControlDecision",
    "NeutralControlSampler",
    "QuietEpisodeDecision",
    "QuietEpisodeTracker",
    "QuietStateSnapshot",
    "classify_quiet_state",
    "high_tail_proximity",
]
