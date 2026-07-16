from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.clean_anchor_price_acceptance.checkpoint import (
    calculate_price_acceptance,
    select_first_post_anchor_bar,
)

ANCHOR = pd.Timestamp("2025-06-02 14:30:00+00:00")


def _bars(*minutes: int) -> pd.DataFrame:
    rows = []
    for minute in minutes:
        rows.append(
            {
                "timestamp": ANCHOR + pd.Timedelta(minutes=minute),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        )
    return pd.DataFrame(rows)


def test_first_bar_is_selected_by_exact_timestamp_not_row_position() -> None:
    bars = _bars(0, 10, 5, 15)

    result = select_first_post_anchor_bar(bars, anchor_timestamp=ANCHOR)

    assert result.status == "available"
    assert result.bar_start_timestamp == ANCHOR + pd.Timedelta(minutes=5)
    assert result.freeze_timestamp == ANCHOR + pd.Timedelta(minutes=10)


def test_missing_first_bar_does_not_shift_a_later_row_into_checkpoint() -> None:
    result = select_first_post_anchor_bar(_bars(0, 10, 15), anchor_timestamp=ANCHOR)

    assert result.status == "missing_first_post_anchor_bar"
    assert result.bar_start_timestamp is None
    assert result.freeze_timestamp is None


def test_duplicate_first_bar_fails_closed() -> None:
    bars = pd.concat([_bars(0, 5), _bars(5, 10)], ignore_index=True)

    result = select_first_post_anchor_bar(bars, anchor_timestamp=ANCHOR)

    assert result.status == "ambiguous_first_post_anchor_bar"


def test_appending_future_bars_does_not_change_frozen_checkpoint() -> None:
    original = select_first_post_anchor_bar(_bars(0, 5), anchor_timestamp=ANCHOR)
    appended = select_first_post_anchor_bar(_bars(0, 5, 10, 15, 20), anchor_timestamp=ANCHOR)

    assert appended == original


def test_long_direction_adjusted_price_acceptance_is_exact() -> None:
    checkpoint = select_first_post_anchor_bar(
        pd.DataFrame(
            [
                {
                    "timestamp": ANCHOR + pd.Timedelta(minutes=5),
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 101.0,
                }
            ]
        ),
        anchor_timestamp=ANCHOR,
    )

    result = calculate_price_acceptance(checkpoint, anchor_reference_price=100.0, direction=1)

    assert result.status == "available"
    assert result.signed_close_return_bps == pytest.approx(100.0)
    assert result.favourable_excursion_bps == pytest.approx(200.0)
    assert result.adverse_excursion_bps == pytest.approx(50.0)
    assert result.acceptance_balance_bps == pytest.approx(150.0)
    assert result.price_acceptance_pass is True


def test_short_direction_adjusted_price_acceptance_is_exact() -> None:
    checkpoint = select_first_post_anchor_bar(
        pd.DataFrame(
            [
                {
                    "timestamp": ANCHOR + pd.Timedelta(minutes=5),
                    "open": 100.0,
                    "high": 100.5,
                    "low": 98.0,
                    "close": 99.0,
                }
            ]
        ),
        anchor_timestamp=ANCHOR,
    )

    result = calculate_price_acceptance(checkpoint, anchor_reference_price=100.0, direction=-1)

    assert result.signed_close_return_bps == pytest.approx(100.0)
    assert result.favourable_excursion_bps == pytest.approx(200.0)
    assert result.adverse_excursion_bps == pytest.approx(50.0)
    assert result.acceptance_balance_bps == pytest.approx(150.0)
    assert result.price_acceptance_pass is True


def test_zero_signed_close_fails_primary_rule() -> None:
    checkpoint = select_first_post_anchor_bar(_bars(5), anchor_timestamp=ANCHOR)

    result = calculate_price_acceptance(checkpoint, anchor_reference_price=100.5, direction=1)

    assert result.signed_close_return_bps == 0.0
    assert result.price_acceptance_pass is False


def test_positive_close_with_larger_adverse_excursion_fails() -> None:
    checkpoint = select_first_post_anchor_bar(
        pd.DataFrame(
            [
                {
                    "timestamp": ANCHOR + pd.Timedelta(minutes=5),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 97.0,
                    "close": 100.5,
                }
            ]
        ),
        anchor_timestamp=ANCHOR,
    )

    result = calculate_price_acceptance(checkpoint, anchor_reference_price=100.0, direction=1)

    assert result.signed_close_return_bps == pytest.approx(50.0)
    assert result.favourable_excursion_bps == pytest.approx(100.0)
    assert result.adverse_excursion_bps == pytest.approx(300.0)
    assert result.price_acceptance_pass is False


def test_missing_or_ambiguous_direction_fails_closed() -> None:
    checkpoint = select_first_post_anchor_bar(_bars(5), anchor_timestamp=ANCHOR)

    result = calculate_price_acceptance(checkpoint, anchor_reference_price=100.0, direction=0)

    assert result.status == "ambiguous_direction"
    assert result.price_acceptance_pass is False
    assert result.signed_close_return_bps is None
