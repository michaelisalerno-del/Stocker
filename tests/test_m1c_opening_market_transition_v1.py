from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_prospective.opening_market_transition_v1 import (
    EXPECTED_OPENING_BAR_COUNT_V1,
    OpeningTransitionThresholdsV1,
    calculate_opening_preentry_window_v1,
    calculate_stock_opening_response_v1,
    classify_opening_market_transition_v1,
)
from stocker_prospective.signed_market_shock_v1 import (
    MarketShockBarV1,
    frozen_material_move_v1,
    partition_material_endpoint_v1,
)

SESSION = date(2025, 1, 2)
PREVIOUS_SESSION = date(2024, 12, 31)
OPEN = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
SIGNAL = OPEN + timedelta(minutes=30)


def _bars(
    symbol: str,
    *,
    count: int = 6,
    close_step: float = 0.5,
) -> tuple[MarketShockBarV1, ...]:
    return tuple(
        MarketShockBarV1(
            symbol=symbol,
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=OPEN + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=OPEN + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + close_step * ordinal,
            high=100.8 + close_step * ordinal,
            low=99.6 + close_step * ordinal,
            close=100.4 + close_step * ordinal,
            finalised=True,
        )
        for ordinal in range(count)
    )


def _window(
    bars: tuple[MarketShockBarV1, ...] | None = None,
):
    return calculate_opening_preentry_window_v1(
        market_proxy="VTI",
        session=SESSION,
        previous_session=PREVIOUS_SESSION,
        session_open_timestamp=OPEN,
        signal_timestamp=SIGNAL,
        entry_timestamp=SIGNAL,
        completed_bars=_bars("VTI") if bars is None else bars,
        prior_regular_session_close=99.0,
    )


def _thresholds() -> OpeningTransitionThresholdsV1:
    return OpeningTransitionThresholdsV1(
        market_opening_return_q10_v1=-0.02,
        market_opening_return_q90_v1=0.02,
        market_opening_range_q75_v1=0.025,
        market_overnight_gap_q10_v1=-0.01,
        market_overnight_gap_q90_v1=0.01,
        market_total_transition_q10_v1=-0.025,
        market_total_transition_q90_v1=0.025,
        market_opening_return_support_v1=250,
        market_opening_range_support_v1=250,
        market_overnight_gap_support_v1=250,
        market_total_transition_support_v1=250,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )


def _state(
    *,
    opening_return: float,
    opening_range: float,
):
    window = _window().model_copy(
        update={
            "market_opening_return_v1": opening_return,
            "market_opening_range_v1": opening_range,
        }
    )
    return classify_opening_market_transition_v1(
        window=window,
        thresholds=_thresholds(),
    )


def _trending_stock_bars(step: float) -> tuple[MarketShockBarV1, ...]:
    return tuple(
        MarketShockBarV1(
            symbol="ABC",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=OPEN + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=OPEN + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + step * ordinal,
            high=max(100.0 + step * ordinal, 100.2 + step * ordinal) + 0.5,
            low=min(100.0 + step * ordinal, 100.2 + step * ordinal) - 0.5,
            close=100.2 + step * ordinal,
            finalised=True,
        )
        for ordinal in range(6)
    )


def test_checkpoint_six_timing_uses_six_complete_opening_bars() -> None:
    result = _window()

    assert EXPECTED_OPENING_BAR_COUNT_V1 == 6
    assert result.complete_v1
    assert result.opening_bar_ordinals_v1 == (0, 1, 2, 3, 4, 5)
    assert result.expected_opening_bar_count_v1 == 6
    assert result.observed_opening_bar_count_v1 == 6
    assert result.final_complete_pre_entry_bar_start_v1 == OPEN + timedelta(
        minutes=25
    )
    assert result.maximum_market_timestamp_v1 == SIGNAL
    assert result.entry_bar_ordinal_v1 == 6
    assert result.entry_bar_included_v1 is False


def test_opening_return_identity_uses_regular_open_and_previous_session_close() -> None:
    result = _window()
    last_close = _bars("VTI")[-1].close

    assert result.market_opening_return_v1 == pytest.approx(
        math.log(last_close / _bars("VTI")[0].open)
    )
    assert result.market_overnight_gap_v1 == pytest.approx(math.log(100.0 / 99.0))
    assert result.market_total_transition_v1 == pytest.approx(
        math.log(last_close / 99.0)
    )
    assert (
        result.market_overnight_gap_v1 + result.market_opening_return_v1
        == pytest.approx(result.market_total_transition_v1)
    )


def test_entry_and_future_bars_cannot_change_opening_measurements() -> None:
    baseline = _window()
    entry = MarketShockBarV1(
        symbol="VTI",
        session=SESSION,
        bar_ordinal=6,
        bar_start_timestamp=SIGNAL,
        bar_complete_timestamp=SIGNAL + timedelta(minutes=5),
        open=1.0,
        high=10_000.0,
        low=0.01,
        close=9_999.0,
        finalised=True,
    )
    changed = _window((*_bars("VTI"), entry))

    assert changed == baseline


@pytest.mark.parametrize("defect", ["missing", "partial", "non_contiguous"])
def test_incomplete_opening_bars_fail_closed(defect: str) -> None:
    bars = list(_bars("VTI"))
    if defect == "missing":
        bars.pop(3)
    elif defect == "partial":
        bars[5] = bars[5].model_copy(update={"finalised": False})
    else:
        bars[4] = bars[4].model_copy(
            update={
                "bar_start_timestamp": bars[4].bar_start_timestamp
                + timedelta(minutes=1),
                "bar_complete_timestamp": bars[4].bar_complete_timestamp
                + timedelta(minutes=1),
            }
        )

    result = _window(tuple(bars))

    assert not result.complete_v1
    assert result.market_opening_return_v1 is None
    assert result.missing_reasons_v1


def test_opening_window_never_uses_another_session() -> None:
    wrong = _bars("VTI")[0].model_copy(
        update={
            "session": PREVIOUS_SESSION,
            "bar_start_timestamp": OPEN - timedelta(days=2),
            "bar_complete_timestamp": OPEN - timedelta(days=2) + timedelta(minutes=5),
        }
    )
    result = _window((wrong, *_bars("VTI")[1:]))

    assert not result.complete_v1
    assert "missing_market_bar:0" in result.missing_reasons_v1


@pytest.mark.parametrize(
    ("opening_return", "opening_range", "expected_state", "expected_sign"),
    [
        (
            -0.02,
            0.025,
            "NEGATIVE_SEVERE_OPENING_TRANSITION",
            -1,
        ),
        (
            0.02,
            0.025,
            "POSITIVE_SEVERE_OPENING_TRANSITION",
            1,
        ),
        (
            0.0,
            0.025,
            "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
            None,
        ),
        (0.0, 0.024, "NORMAL_OPENING", None),
    ],
)
def test_opening_state_uses_fixed_inclusive_thresholds(
    opening_return: float,
    opening_range: float,
    expected_state: str,
    expected_sign: int | None,
) -> None:
    result = _state(
        opening_return=opening_return,
        opening_range=opening_range,
    )

    assert result.opening_market_transition_state_v1 == expected_state
    assert result.opening_transition_sign_v1 == expected_sign
    assert result.complete_v1


def test_incomplete_window_produces_unknown_state() -> None:
    result = classify_opening_market_transition_v1(
        window=_window(_bars("VTI", count=5)),
        thresholds=_thresholds(),
    )

    assert result.opening_market_transition_state_v1 == "UNKNOWN_INCOMPLETE"
    assert not result.complete_v1
    assert result.opening_transition_event_id_v1 is None


@pytest.mark.parametrize(
    ("transition_return", "stock_step", "expected_class", "relative_sign"),
    [
        (0.02, 1.0, "AMPLIFYING", 1),
        (-0.02, -1.0, "AMPLIFYING", 1),
        (0.02, 0.1, "RESISTING", -1),
        (-0.02, -0.1, "RESISTING", -1),
    ],
)
def test_stock_response_class_respects_signed_relative_move(
    transition_return: float,
    stock_step: float,
    expected_class: str,
    relative_sign: int,
) -> None:
    state = _state(
        opening_return=transition_return,
        opening_range=0.03,
    )
    result = calculate_stock_opening_response_v1(
        symbol="ABC",
        session=SESSION,
        session_open_timestamp=OPEN,
        signal_timestamp=SIGNAL,
        completed_stock_bars=_trending_stock_bars(stock_step),
        market_opening_return_v1=transition_return,
        opening_transition_state_v1=state,
        threshold_15m=0.01,
    )

    assert result.stock_opening_response_class_v1 == expected_class
    assert math.copysign(1, result.stock_relative_opening_response_v1) == relative_sign


def test_exact_stock_market_relative_neutrality_is_not_forced() -> None:
    stock_bars = _trending_stock_bars(0.5)
    stock_return = math.log(stock_bars[-1].close / stock_bars[0].open)
    state = _state(opening_return=0.02, opening_range=0.03)

    result = calculate_stock_opening_response_v1(
        symbol="ABC",
        session=SESSION,
        session_open_timestamp=OPEN,
        signal_timestamp=SIGNAL,
        completed_stock_bars=stock_bars,
        market_opening_return_v1=stock_return,
        opening_transition_state_v1=state,
        threshold_15m=0.01,
    )

    assert result.stock_relative_opening_response_v1 == 0.0
    assert result.stock_opening_response_class_v1 == "NEUTRAL_EXACT"


def test_invalid_iv_scale_fails_stock_response_closed() -> None:
    result = calculate_stock_opening_response_v1(
        symbol="ABC",
        session=SESSION,
        session_open_timestamp=OPEN,
        signal_timestamp=SIGNAL,
        completed_stock_bars=_trending_stock_bars(1.0),
        market_opening_return_v1=0.02,
        opening_transition_state_v1=_state(
            opening_return=0.02,
            opening_range=0.03,
        ),
        threshold_15m=0.0,
    )

    assert result.stock_opening_response_class_v1 == "UNKNOWN_INCOMPLETE"
    assert result.missing_reasons_v1 == ("threshold_15m_invalid",)


def test_future_stock_bar_and_other_stock_cannot_change_response() -> None:
    state = _state(opening_return=0.02, opening_range=0.03)
    kwargs = {
        "symbol": "ABC",
        "session": SESSION,
        "session_open_timestamp": OPEN,
        "signal_timestamp": SIGNAL,
        "market_opening_return_v1": 0.02,
        "opening_transition_state_v1": state,
        "threshold_15m": 0.01,
    }
    baseline = calculate_stock_opening_response_v1(
        completed_stock_bars=_trending_stock_bars(1.0),
        **kwargs,
    )
    future = _trending_stock_bars(1.0)[-1].model_copy(
        update={
            "bar_ordinal": 6,
            "bar_start_timestamp": SIGNAL,
            "bar_complete_timestamp": SIGNAL + timedelta(minutes=5),
            "close": 1_000.0,
            "high": 1_001.0,
        }
    )
    changed = calculate_stock_opening_response_v1(
        completed_stock_bars=(
            *_trending_stock_bars(1.0),
            future,
            *_bars("OTHER"),
        ),
        **kwargs,
    )

    assert changed == baseline


@pytest.mark.parametrize(
    ("signed_return", "threshold", "expected"),
    [
        (0.0100001, 0.01, "MATERIAL_UP"),
        (-0.0100001, 0.01, "MATERIAL_DOWN"),
        (0.01, 0.01, "NO_MATERIAL_MOVE"),
        (-0.01, 0.01, "NO_MATERIAL_MOVE"),
        (0.0, 0.01, "NO_MATERIAL_MOVE"),
    ],
)
def test_material_partition_preserves_strict_frozen_target(
    signed_return: float,
    threshold: float,
    expected: str,
) -> None:
    partition = partition_material_endpoint_v1(
        signed_return=signed_return,
        threshold_15m=threshold,
    )

    assert partition == expected
    assert (partition in {"MATERIAL_UP", "MATERIAL_DOWN"}) == frozen_material_move_v1(
        signed_return=signed_return,
        threshold_15m=threshold,
    )
