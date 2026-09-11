from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_core.markets import MarketId, get_market
from stocker_execution.expected_move import (
    HV_EXPECTED_MOVE_CALCULATION_VERSION,
    IBKR_HISTORICAL_VOLATILITY_SOURCE,
    ExpectedMoveResult,
    ExpectedMoveStatus,
    IbkrHistoricalVolatilityExpectedMoveService,
    calculate_hv_expected_move,
    normalize_historical_volatility,
)
from stocker_execution.ibkr import (
    HistoricalVolatilitySnapshot,
    IbkrConnection,
    QualifiedInstrument,
)

T0 = datetime(2026, 9, 3, 14, 0, tzinfo=UTC)


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def instrument() -> QualifiedInstrument:
    return QualifiedInstrument("AAPL", 265598, "SMART", "NASDAQ", "USD", "STK")


class HvBoundary(IbkrConnection):
    def __init__(self, raw_hv: float = 0.40, *, observed_at: datetime = T0) -> None:
        self.raw_hv = raw_hv
        self.observed_at = observed_at
        self.hv_requests = 0
        self.option_chain_requests = 0
        self.historical_requests = 0

    async def historical_volatility_snapshot(
        self, requested: QualifiedInstrument, **_kwargs: object
    ) -> HistoricalVolatilitySnapshot:
        self.hv_requests += 1
        return HistoricalVolatilitySnapshot(
            requested.symbol,
            requested.con_id,
            self.raw_hv,
            self.observed_at,
            1,
        )

    async def option_chains(self, requested: QualifiedInstrument) -> tuple[object, ...]:
        del requested
        self.option_chain_requests += 1
        raise AssertionError("HV must not request an option chain")

    async def historical_bars(self, requested: QualifiedInstrument, **_kwargs: object) -> tuple[()]:
        del requested
        self.historical_requests += 1
        raise AssertionError("HV expected move must not backfill historical bars")


def test_hv_expected_move_conversion_is_deterministic_and_has_frozen_lineage() -> None:
    calculation = calculate_hv_expected_move(0.40, market_regular_minutes=390)

    assert calculation.historical_volatility == 0.40
    assert calculation.market_regular_minutes == 390
    assert calculation.sigma_15 == pytest.approx(0.004941662111074008, abs=1e-18)
    assert calculation.expected_absolute_return_15m == pytest.approx(0.00333310044188278, abs=1e-18)
    assert calculation.calculation_version == HV_EXPECTED_MOVE_CALCULATION_VERSION


@pytest.mark.parametrize(
    ("market_id", "expected_minutes", "expected_move"),
    (
        (MarketId.US_ALL, 390, 0.00333310044188278),
        (MarketId.UK_LSE, 510, 0.0029147117829851233),
        (MarketId.AUSTRALIA_ASX, 360, 0.003469200931336463),
    ),
)
def test_hv_expected_move_uses_each_market_active_regular_minutes(
    market_id: MarketId, expected_minutes: int, expected_move: float
) -> None:
    market = get_market(market_id)

    assert market.active_regular_minutes == expected_minutes
    assert calculate_hv_expected_move(
        0.40, market_regular_minutes=market.active_regular_minutes
    ).expected_absolute_return_15m == pytest.approx(expected_move, abs=1e-18)


def test_hv_normalization_accepts_decimal_or_percent_form_and_rejects_invalid_values() -> None:
    assert normalize_historical_volatility(0.40, unit="DECIMAL") == 0.40
    assert normalize_historical_volatility(40.0, unit="PERCENT") == 0.40

    for value in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            normalize_historical_volatility(value, unit="DECIMAL")

    with pytest.raises(ValueError, match="unit"):
        normalize_historical_volatility(0.40, unit="UNKNOWN")  # type: ignore[arg-type]


def test_hv_expected_move_service_uses_tick_104_without_options_or_history() -> None:
    boundary = HvBoundary()
    clock = MutableClock(T0 - timedelta(seconds=1))
    service = IbkrHistoricalVolatilityExpectedMoveService(boundary, clock=clock)

    async def scenario() -> ExpectedMoveResult:
        await service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
        clock.now = T0 + timedelta(seconds=1)
        return await service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )

    result = asyncio.run(scenario())

    assert result.status is ExpectedMoveStatus.READY
    assert result.source == IBKR_HISTORICAL_VOLATILITY_SOURCE
    assert result.observation_timestamp == T0
    assert result.raw_historical_volatility == 0.40
    assert result.historical_volatility == 0.40
    assert result.market_regular_minutes == 390
    assert result.expected_absolute_return_15m == pytest.approx(0.00333310044188278)
    assert boundary.hv_requests == 1
    assert boundary.option_chain_requests == 0
    assert boundary.historical_requests == 0


@pytest.mark.parametrize("raw_hv", (0.0, -1.0, float("nan")))
def test_invalid_hv_rejects_one_expected_move_without_fallback(raw_hv: float) -> None:
    boundary = HvBoundary(raw_hv)
    clock = MutableClock(T0 - timedelta(seconds=1))
    service = IbkrHistoricalVolatilityExpectedMoveService(boundary, clock=clock)

    async def scenario() -> ExpectedMoveResult:
        await service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
        clock.now = T0 + timedelta(seconds=1)
        return await service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )

    result = asyncio.run(scenario())

    assert result.status is ExpectedMoveStatus.NOT_READY
    assert result.reason.startswith("HV_NOT_READY")
    assert result.expected_absolute_return_15m is None
    assert boundary.option_chain_requests == 0
    assert boundary.historical_requests == 0


def test_stale_or_future_hv_observation_is_rejected() -> None:
    stale = HvBoundary(observed_at=T0 - timedelta(minutes=6))
    stale_clock = MutableClock(T0 - timedelta(minutes=7))
    stale_service = IbkrHistoricalVolatilityExpectedMoveService(stale, clock=stale_clock)

    async def stale_scenario() -> ExpectedMoveResult:
        await stale_service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
        stale_clock.now = T0 + timedelta(seconds=1)
        return await stale_service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )

    stale_result = asyncio.run(stale_scenario())

    assert stale_result.status is ExpectedMoveStatus.NOT_READY
    assert "stale observation timestamp" in stale_result.reason

    future = HvBoundary(observed_at=T0 + timedelta(microseconds=1))
    future_clock = MutableClock(T0 - timedelta(seconds=1))
    future_service = IbkrHistoricalVolatilityExpectedMoveService(future, clock=future_clock)

    async def future_scenario() -> ExpectedMoveResult:
        await future_service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
        future_clock.now = T0 + timedelta(seconds=1)
        return await future_service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )

    result = asyncio.run(future_scenario())

    assert result.status is ExpectedMoveStatus.NOT_READY
    assert "unverifiable observation timestamp" in result.reason


def test_hv_without_a_pre_t0_capture_is_not_requested_late() -> None:
    boundary = HvBoundary(observed_at=T0 + timedelta(seconds=1))
    clock = MutableClock(T0 + timedelta(seconds=1))
    service = IbkrHistoricalVolatilityExpectedMoveService(boundary, clock=clock)

    result = asyncio.run(
        service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
    )

    assert result.status is ExpectedMoveStatus.NOT_READY
    assert "captured by T0" in result.reason
    assert boundary.hv_requests == 0


def test_nearby_checkpoint_preparation_does_not_discard_an_unconsumed_snapshot() -> None:
    boundary = HvBoundary(observed_at=T0 - timedelta(minutes=4))
    clock = MutableClock(T0 - timedelta(minutes=4))
    service = IbkrHistoricalVolatilityExpectedMoveService(boundary, clock=clock)
    later_t0 = T0 + timedelta(minutes=2)

    async def scenario() -> ExpectedMoveResult:
        await service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )
        clock.now = T0 - timedelta(minutes=3)
        boundary.observed_at = clock.now
        await service.prepare_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=later_t0,
        )
        clock.now = T0
        return await service.get_expected_move(
            instrument(),
            market=get_market(MarketId.US_ALL),
            session=date(2026, 9, 3),
            t0=T0,
        )

    result = asyncio.run(scenario())

    assert result.status is ExpectedMoveStatus.READY
    assert result.observation_timestamp == T0 - timedelta(minutes=4)


def test_only_hv_is_installed_and_default_strategy_is_hv() -> None:
    from stocker_core.strategies import SESSION_HARD_HV_METHOD, installed_strategies
    from stocker_execution.strategy_factory import create_strategy

    assert installed_strategies() == (SESSION_HARD_HV_METHOD,)
    strategy = create_strategy(
        SESSION_HARD_HV_METHOD.strategy_id, SESSION_HARD_HV_METHOD.strategy_version
    )
    assert strategy.strategy_id == SESSION_HARD_HV_METHOD.strategy_id
    assert strategy.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
    with pytest.raises(ValueError, match="Unsupported runtime strategy"):
        create_strategy(
            "SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D", "SESSION_HARD_STRUCTURE_D_V1"
        )
    assert not hasattr(IbkrConnection, "option_chains")
    assert not hasattr(IbkrConnection, "option_snapshots")
    assert not hasattr(IbkrConnection, "qualify_options")
