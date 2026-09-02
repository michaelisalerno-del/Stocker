from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_execution.ibkr import HistoricalBar, QualifiedInstrument
from stocker_execution.stage5 import CandidateStatus, calculate_stage5_feature

T0 = datetime(2025, 2, 20, 15, 0, tzinfo=UTC)


def bar(timestamp: datetime, open_price: float) -> HistoricalBar:
    return HistoricalBar(timestamp, open_price, open_price, open_price, open_price, 100.0)


def instrument() -> QualifiedInstrument:
    return QualifiedInstrument("HOOD", 123, "SMART", "NASDAQ", "USD", "STK")


def test_feature_uses_native_five_minute_p0_and_exact_one_minute_endpoints() -> None:
    result = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=0.01,
        five_minute_bars=(bar(T0 - timedelta(minutes=5), 99.0), bar(T0, 100.0)),
        one_minute_bars=(
            bar(T0 - timedelta(minutes=4), 88.0),
            bar(T0 - timedelta(minutes=3), 99.0),
            bar(T0 - timedelta(minutes=2), 77.0),
            bar(T0, 100.0),
        ),
    )

    assert result.status is CandidateStatus.READY
    assert result.p0 == 100.0
    assert result.raw_open_t0_minus_3m == 99.0
    assert result.raw_open_t0 == 100.0
    assert result.raw_pre_move_price == 1.0
    assert result.pre_move_m == 1.0


@pytest.mark.parametrize(
    ("one_minute_bars", "reason"),
    [
        ((bar(T0 - timedelta(minutes=4), 99.0), bar(T0, 100.0)), "T0-3m"),
        ((bar(T0 - timedelta(minutes=3), 99.0), bar(T0 - timedelta(minutes=1), 100.0)), "T0"),
    ],
)
def test_missing_exact_one_minute_endpoint_is_not_ready_without_interpolation(
    one_minute_bars: tuple[HistoricalBar, ...], reason: str
) -> None:
    result = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=0.01,
        five_minute_bars=(bar(T0, 100.0),),
        one_minute_bars=one_minute_bars,
    )

    assert result.status is CandidateStatus.PRE_MOVE_NOT_READY
    assert reason in result.exclusion_reason
    assert result.pre_move_m is None


def test_missing_stage4_context_is_not_ready() -> None:
    result = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=None,
        five_minute_bars=(bar(T0, 100.0),),
        one_minute_bars=(bar(T0 - timedelta(minutes=3), 99.0), bar(T0, 100.0)),
    )

    assert result.status is CandidateStatus.PRE_CONTEXT_NOT_READY
    assert result.pre_move_m is None


@pytest.mark.parametrize(
    "duplicate_timestamp",
    [T0 - timedelta(minutes=3), T0],
)
def test_duplicate_required_one_minute_endpoint_is_not_ready(
    duplicate_timestamp: datetime,
) -> None:
    one_minute_bars = [
        bar(T0 - timedelta(minutes=3), 99.0),
        bar(T0, 100.0),
        bar(duplicate_timestamp, 101.0),
    ]

    result = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=0.01,
        five_minute_bars=(bar(T0, 100.0),),
        one_minute_bars=one_minute_bars,
    )

    assert result.status is CandidateStatus.PRE_MOVE_NOT_READY
    assert "duplicate" in result.exclusion_reason
    assert result.pre_move_m is None


def test_future_bars_are_not_consumed() -> None:
    without_future = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=0.01,
        five_minute_bars=(bar(T0, 100.0),),
        one_minute_bars=(bar(T0 - timedelta(minutes=3), 99.0), bar(T0, 100.0)),
    )
    with_future = calculate_stage5_feature(
        instrument=instrument(),
        session=date(2025, 2, 20),
        t0=T0,
        expected_absolute_return_15m=0.01,
        five_minute_bars=(bar(T0, 100.0), bar(T0 + timedelta(minutes=5), 1.0)),
        one_minute_bars=(
            bar(T0 - timedelta(minutes=3), 99.0),
            bar(T0, 100.0),
            bar(T0 + timedelta(minutes=1), 1.0),
        ),
    )

    assert with_future == without_future
