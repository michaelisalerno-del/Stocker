"""Frozen market-independent Session HARD candidate mathematics and recipe.

Inputs are normalized, finalized regular-session minute bars. Universe acquisition,
calendar resolution, persistence and trading qualification belong to their owners.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import isfinite
from typing import Protocol

import numpy as np
import pandas as pd

from stocker_core.markets import MarketId, get_market


class CandidateEvidenceStatus(StrEnum):
    VALIDATED_EXISTING_RESEARCH = "VALIDATED_EXISTING_RESEARCH"
    UNVALIDATED_TRANSFER = "UNVALIDATED_TRANSFER"


@dataclass(frozen=True, slots=True)
class CandidateStage:
    stage_id: str
    score_name: str
    minutes: int
    capacity: int
    feature: str
    direction: str = "HIGH"
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class SessionHardCandidateRecipe:
    recipe_id: str
    market_scope: str
    stages: tuple[CandidateStage, ...]
    research_hashes: tuple[str, ...]


SESSION_HARD_CANDIDATE_RECIPE = SessionHardCandidateRecipe(
    recipe_id="SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1",
    market_scope="US_DEVELOPMENT_EVIDENCE_OTHER_MARKETS_UNVALIDATED_TRANSFER",
    stages=(
        CandidateStage("RANGE250", "initial_range_5m_score", 5, 250, "RANGE"),
        CandidateStage("RV50", "preselector_rv_10m_score", 10, 50, "RV"),
        CandidateStage("RV30", "watchlist_rv_15m_score", 15, 30, "RV"),
    ),
    research_hashes=(
        "76b6202ce8ad5c6c57b44fe0fefe8611fda39d0bba48c8810690210878cc9bfa",
        "39f730eac22d5bc9a380a9dbf028c9c374716b1c2c51d1f1e1f80a482fc91aab",
        "aca8441a0190ca8d66d7668ad152e06c05a15d1859b696d089a67747d8eb0364",
    ),
)


def candidate_evidence_status(market_id: MarketId) -> CandidateEvidenceStatus:
    """Evidence applies to the US development population, not all possible universes."""
    return (
        CandidateEvidenceStatus.VALIDATED_EXISTING_RESEARCH
        if get_market(market_id).country == "US"
        else CandidateEvidenceStatus.UNVALIDATED_TRANSFER
    )


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    con_id: int
    symbol: str
    primary_exchange: str | None
    listing_exchange: str
    currency: str
    market: MarketId
    security_type: str
    routing_exchange: str = "SMART"

    def __post_init__(self) -> None:
        if self.con_id <= 0 or not self.symbol or not self.listing_exchange:
            raise ValueError("A normalized broker security identity is required")
        if self.security_type != "STK":
            raise ValueError("Candidate identities must be eligible equities")


class CandidateBar(Protocol):
    @property
    def timestamp(self) -> object: ...

    @property
    def open(self) -> float: ...

    @property
    def high(self) -> float: ...

    @property
    def low(self) -> float: ...

    @property
    def close(self) -> float: ...


@dataclass(frozen=True, slots=True)
class CandidateValue:
    value: float | None
    missing_reason: str = ""


@dataclass(frozen=True, slots=True)
class CandidateRank:
    identity: CandidateIdentity
    score_name: str
    value: float | None
    missing_reason: str
    rank: int
    selected: bool


def require_aware(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Candidate timestamps must be timezone-aware")
    return timestamp.astimezone(UTC)


def opening_prefix(
    regular_open: datetime, regular_close: datetime, minutes: int
) -> tuple[datetime, ...]:
    """Use the official session bounds supplied by the canonical calendar owner."""
    start, end = require_aware(regular_open), require_aware(regular_close)
    if minutes <= 0 or start + timedelta(minutes=minutes) > end:
        raise ValueError("Session cannot provide the required opening prefix")
    return tuple(start + timedelta(minutes=i) for i in range(minutes))


def candidate_value(
    stage: CandidateStage,
    bars: Sequence[CandidateBar],
    *,
    expected_prefix: Sequence[datetime],
    as_of: datetime,
) -> CandidateValue:
    """Calculate only a complete, unique, finalized first-minute prefix.

    Callers attest bar finality through their data source. Timestamp checks also
    prevent unfinished/future minutes from entering any feature calculation.
    NumPy log and pandas groupby sum match the frozen float64 research operations,
    including pandas' compensated sum; a different reduction can change RV bits.
    """
    now = require_aware(as_of)
    expected = tuple(require_aware(t) for t in expected_prefix)
    if len(expected) != stage.minutes or any(
        right - left < timedelta(minutes=1)
        for left, right in zip(expected, expected[1:], strict=False)
    ):
        raise ValueError("Expected prefix must contain ordered active regular-session minutes")
    if not expected or now < expected[-1] + timedelta(minutes=1):
        raise ValueError("Candidate stage is not due: required minutes are not completed")
    prefix: list[CandidateBar] = []
    for bar in bars:
        if not isinstance(bar.timestamp, datetime):
            return CandidateValue(None, "NON_CONSECUTIVE_PREFIX")
        timestamp = require_aware(bar.timestamp)
        if timestamp.replace(second=0, microsecond=0) in expected and timestamp not in expected:
            return CandidateValue(None, "NON_CONSECUTIVE_PREFIX")
        if timestamp in expected:
            prefix.append(bar)
    times = tuple(
        require_aware(bar.timestamp) for bar in prefix if isinstance(bar.timestamp, datetime)
    )
    if len(set(times)) != len(times) or tuple(sorted(times)) != times:
        return CandidateValue(None, "NON_CONSECUTIVE_PREFIX")
    if times != expected:
        return CandidateValue(
            None, "MISSING_BAR" if set(times) <= set(expected) else "NON_CONSECUTIVE_PREFIX"
        )
    first_open = prefix[0].open
    if not isfinite(first_open) or first_open <= 0:
        return CandidateValue(None, "INVALID_OPEN")
    if any(
        not all(isfinite(v) and v > 0 for v in (b.open, b.high, b.low, b.close))
        or b.high < max(b.open, b.low, b.close)
        or b.low > min(b.open, b.high, b.close)
        for b in prefix
    ):
        return CandidateValue(None, "INVALID_OHLC")
    if stage.feature == "RANGE":
        value = (max(b.high for b in prefix) - min(b.low for b in prefix)) / first_open
    elif stage.feature == "RV":
        closes = np.asarray([b.close for b in prefix], dtype=np.float64)
        previous = np.concatenate(([first_open], closes[:-1]))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            squares = np.log(closes / previous) ** 2
            total = pd.Series(squares).groupby(np.zeros(stage.minutes, dtype=int)).sum().iloc[0]
            value = float(np.sqrt(total))
    else:
        raise ValueError(f"Unsupported candidate feature: {stage.feature}")
    return (
        CandidateValue(float(value)) if isfinite(value) else CandidateValue(None, "NONFINITE_SCORE")
    )


def rank_candidates(
    stage: CandidateStage,
    identities: Sequence[CandidateIdentity],
    values: Mapping[int, CandidateValue],
    *,
    market: MarketId,
) -> tuple[CandidateRank, ...]:
    """Global HIGH ranking; missing last; frozen ascending SHA256(symbol) ties.

    conId only breaks an otherwise identical symbol/hash tie in normalized inputs;
    a ticker never replaces broker identity. Previous stage survivors are the only
    identities a caller may supply to the next stage.
    """
    if stage.direction != "HIGH" or not stage.enabled:
        raise ValueError("This frozen recipe requires enabled HIGH stages")
    if len({i.con_id for i in identities}) != len(identities):
        raise ValueError("Duplicate canonical candidate identity")
    if any(i.market != market for i in identities):
        raise ValueError("Candidate identity belongs to another market")
    scores = {
        i.con_id: values.get(i.con_id, CandidateValue(None, "MISSING_BAR")) for i in identities
    }
    scores = {
        con_id: CandidateValue(None, "NONFINITE_SCORE")
        if item.value is not None and not isfinite(item.value)
        else item
        for con_id, item in scores.items()
    }

    def key(identity: CandidateIdentity) -> tuple[bool, float, str, int]:
        value = scores[identity.con_id].value
        return (
            value is None,
            -value if value is not None else 0.0,
            hashlib.sha256(identity.symbol.encode()).hexdigest(),
            identity.con_id,
        )

    return tuple(
        CandidateRank(
            identity,
            stage.score_name,
            scores[identity.con_id].value,
            scores[identity.con_id].missing_reason,
            rank,
            rank <= stage.capacity,
        )
        for rank, identity in enumerate(sorted(identities, key=key), 1)
    )
