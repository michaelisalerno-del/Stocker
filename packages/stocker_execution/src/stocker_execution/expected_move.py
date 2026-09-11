"""Explicit expected-move producers consumed by Stage 5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import isfinite, sqrt
from typing import Literal, Protocol

from stocker_core.markets import MarketDefinition
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument

HV_EXPECTED_MOVE_CALCULATION_VERSION = "EXPECTED_MOVE_HV_V1"
IBKR_HISTORICAL_VOLATILITY_SOURCE = "IBKR_HISTORICAL_VOLATILITY_TICK_104"
NORMAL_MEDIAN_ABSOLUTE_MULTIPLIER = 0.67448975
HV_CURRENT_SNAPSHOT_MAX_AGE = timedelta(minutes=5)


class ExpectedMoveStatus(StrEnum):
    """Availability of one causal expected-move observation."""

    READY = "READY"
    NOT_READY = "NOT_READY"


@dataclass(frozen=True, slots=True)
class ExpectedMoveResult:
    """Minimal expected-return value plus auditable producer lineage."""

    status: ExpectedMoveStatus
    expected_absolute_return_15m: float | None
    source: str
    observation_timestamp: datetime | None
    calculation_version: str
    reason: str
    raw_historical_volatility: float | None = None
    historical_volatility: float | None = None
    market_regular_minutes: int | None = None


class ExpectedMoveService(Protocol):
    """The small Stage 5 boundary for the HV M producer."""

    async def get_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult: ...

    async def prepare_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class HistoricalVolatilityExpectedMoveCalculation:
    """Pure frozen conversion from annualized 30-day HV to 15-minute M."""

    historical_volatility: float
    market_regular_minutes: int
    sigma_15: float
    expected_absolute_return_15m: float
    calculation_version: str = HV_EXPECTED_MOVE_CALCULATION_VERSION


def normalize_historical_volatility(value: float, *, unit: Literal["DECIMAL", "PERCENT"]) -> float:
    """Normalize an IBKR HV value to decimal form without estimating missing data."""

    raw = float(value)
    if not isfinite(raw) or raw <= 0.0:
        raise ValueError("historical volatility must be finite and positive")
    if unit == "DECIMAL":
        normalized = raw
    elif unit == "PERCENT":
        normalized = raw / 100.0
    else:
        raise ValueError(f"unsupported historical volatility unit: {unit}")
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError("historical volatility must be finite and positive")
    return normalized


def calculate_hv_expected_move(
    historical_volatility: float,
    *,
    market_regular_minutes: int,
    unit: Literal["DECIMAL", "PERCENT"] = "DECIMAL",
) -> HistoricalVolatilityExpectedMoveCalculation:
    """Apply the frozen V1 market-session scaling with no time-of-day adjustment."""

    hv = normalize_historical_volatility(historical_volatility, unit=unit)
    if market_regular_minutes <= 0:
        raise ValueError("market regular minutes must be positive")
    sigma_15 = hv * sqrt(15 / (252 * market_regular_minutes))
    expected = sigma_15 * NORMAL_MEDIAN_ABSOLUTE_MULTIPLIER
    if not all(isfinite(item) and item > 0.0 for item in (sigma_15, expected)):
        raise ValueError("historical volatility conversion produced an invalid value")
    return HistoricalVolatilityExpectedMoveCalculation(
        historical_volatility=hv,
        market_regular_minutes=market_regular_minutes,
        sigma_15=sigma_15,
        expected_absolute_return_15m=expected,
    )


class IbkrHistoricalVolatilityExpectedMoveService:
    """Produce M from a fresh temporary IBKR stock generic-tick-104 snapshot."""

    def __init__(
        self,
        ibkr: IbkrConnection,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ibkr, IbkrConnection):
            raise TypeError("HV expected move requires the Stage 2 IbkrConnection")
        self._ibkr = ibkr
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._prepared: dict[tuple[int, str, date, datetime], ExpectedMoveResult] = {}

    async def prepare_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> None:
        key = _expected_move_key(instrument, market, session, t0)
        requested_at = _aware_utc(self._clock())
        self._prepared = {
            prepared_key: result
            for prepared_key, result in self._prepared.items()
            if prepared_key[3] >= requested_at
        }
        if key in self._prepared:
            return
        raw_hv: float | None = None
        observation: datetime | None = None
        checkpoint = _aware_utc(t0)
        try:
            if requested_at > checkpoint:
                raise ValueError("tick 104 was not requested by the checkpoint")
            snapshot = await self._ibkr.historical_volatility_snapshot(instrument)
            raw_hv = snapshot.raw_historical_volatility
            observation = _aware_utc(snapshot.observation_timestamp)
            if not requested_at <= observation <= checkpoint:
                raise ValueError("unverifiable observation timestamp")
            if checkpoint - observation > HV_CURRENT_SNAPSHOT_MAX_AGE:
                raise ValueError("stale observation timestamp")
            calculation = calculate_hv_expected_move(
                raw_hv,
                market_regular_minutes=market.active_regular_minutes,
                unit=snapshot.unit,
            )
            result = ExpectedMoveResult(
                status=ExpectedMoveStatus.READY,
                expected_absolute_return_15m=calculation.expected_absolute_return_15m,
                source=IBKR_HISTORICAL_VOLATILITY_SOURCE,
                observation_timestamp=observation,
                calculation_version=calculation.calculation_version,
                reason="",
                raw_historical_volatility=raw_hv,
                historical_volatility=calculation.historical_volatility,
                market_regular_minutes=calculation.market_regular_minutes,
            )
        except (IbkrError, ValueError) as exc:
            detail = str(exc)
            result = ExpectedMoveResult(
                status=ExpectedMoveStatus.NOT_READY,
                expected_absolute_return_15m=None,
                source=IBKR_HISTORICAL_VOLATILITY_SOURCE,
                observation_timestamp=observation,
                calculation_version=HV_EXPECTED_MOVE_CALCULATION_VERSION,
                reason=detail if detail.startswith("HV_NOT_READY") else f"HV_NOT_READY: {detail}",
                raw_historical_volatility=raw_hv,
                market_regular_minutes=market.active_regular_minutes,
            )
        self._prepared[key] = result

    async def get_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult:
        key = _expected_move_key(instrument, market, session, t0)
        prepared = self._prepared.get(key)
        if prepared is not None:
            return prepared
        return ExpectedMoveResult(
            status=ExpectedMoveStatus.NOT_READY,
            expected_absolute_return_15m=None,
            source=IBKR_HISTORICAL_VOLATILITY_SOURCE,
            observation_timestamp=None,
            calculation_version=HV_EXPECTED_MOVE_CALCULATION_VERSION,
            reason="HV_NOT_READY: no fresh tick 104 observation was captured by T0",
            market_regular_minutes=market.active_regular_minutes,
        )


def _expected_move_key(
    instrument: QualifiedInstrument,
    market: MarketDefinition,
    session: date,
    t0: datetime,
) -> tuple[int, str, date, datetime]:
    return instrument.con_id, market.market_id.value, session, _aware_utc(t0)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("expected-move timestamp must be timezone-aware")
    return value.astimezone(UTC)
