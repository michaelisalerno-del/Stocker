"""Golden arithmetic checks for the recovered research PRE_MOVE_M contract.

These fixtures are research references only. They are not a production data source and do not
implement the current-session Stage 5 feature pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from stocker_execution.stage5 import PRE_MOVE_M_THRESHOLD, calculate_pre_move


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    symbol: str
    session: str
    prior_close: float
    atm_iv: float
    p0: float
    t0: str
    pre_start: str
    raw_t_minus_3_open: float
    raw_t0_open: float
    expected_m: float
    expected_pre_move_m: float


ROWS = (
    ReferenceRow(
        "OKLO",
        "2025-01-03",
        21.844999,
        1.0816,
        24.719999,
        "2025-01-03T15:10:00+00:00",
        "2025-01-03T15:07:00+00:00",
        24.516,
        24.72,
        0.263553170186303,
        0.774037328419791,
    ),
    ReferenceRow(
        "OKLO",
        "2025-01-07",
        29.9801,
        1.34375,
        30.229999,
        "2025-01-07T15:00:00+00:00",
        "2025-01-07T14:57:00+00:00",
        30.19,
        30.23,
        0.400414436576294,
        0.0998964947888205,
    ),
    ReferenceRow(
        "AXON",
        "2025-02-20",
        593.309997,
        0.77035,
        540.525024,
        "2025-02-20T15:10:00+00:00",
        "2025-02-20T15:07:00+00:00",
        539.9613,
        540.525,
        4.10446927263021,
        0.137338103317716,
    ),
    ReferenceRow(
        "HOOD",
        "2025-02-20",
        59.220001,
        0.5879,
        55.549999,
        "2025-02-20T15:00:00+00:00",
        "2025-02-20T14:57:00+00:00",
        56.08,
        55.55,
        0.321914569451631,
        1.64639951326801,
    ),
    ReferenceRow(
        "IVZ",
        "2025-02-20",
        18.2,
        0.30575,
        17.975,
        "2025-02-20T15:10:00+00:00",
        "2025-02-20T15:07:00+00:00",
        18.05,
        17.975,
        0.0541736979379846,
        1.38443567367056,
    ),
    ReferenceRow(
        "VST",
        "2025-02-20",
        169.339996,
        0.9161,
        159.770004,
        "2025-02-20T15:00:00+00:00",
        "2025-02-20T14:57:00+00:00",
        160.6057,
        159.77,
        1.44275054261026,
        0.579240829402579,
    ),
)


def expected_absolute_return_15m(atm_iv: float) -> float:
    return atm_iv * math.sqrt(15.0 / (252.0 * 390.0)) * math.sqrt(2.0 / math.pi)


def movement_scale(row: ReferenceRow) -> float:
    return row.p0 * expected_absolute_return_15m(row.atm_iv)


def aligned_raw_pre_move(row: ReferenceRow) -> float:
    factor = row.p0 / row.raw_t0_open
    return abs(row.p0 - row.raw_t_minus_3_open * factor)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row.symbol}-{row.session}")
def test_recovered_m_and_pre_move_m_match_authoritative_rows(row: ReferenceRow) -> None:
    observed_m = movement_scale(row)
    observed_pre_move_m = aligned_raw_pre_move(row) / observed_m

    assert observed_m == pytest.approx(row.expected_m, abs=5e-15)
    assert observed_pre_move_m == pytest.approx(row.expected_pre_move_m, abs=2e-14)


def test_m_varies_cross_sectionally_with_row_specific_price_and_iv() -> None:
    same_session = [row for row in ROWS if row.session == "2025-02-20"]

    assert len(same_session) == 4
    assert len({movement_scale(row) for row in same_session}) == len(same_session)
    assert min(movement_scale(row) for row in same_session) < 0.06
    assert max(movement_scale(row) for row in same_session) > 4.0


def test_m_varies_for_one_stock_across_dates() -> None:
    oklo = [row for row in ROWS if row.symbol == "OKLO"]

    assert len(oklo) == 2
    assert movement_scale(oklo[0]) != movement_scale(oklo[1])


def test_current_p0_not_prior_close_scales_the_dollar_movement() -> None:
    for row in ROWS:
        incorrect_prior_close_scale = row.prior_close * expected_absolute_return_15m(row.atm_iv)
        assert movement_scale(row) != pytest.approx(incorrect_prior_close_scale, abs=1e-12)


def test_fixed_value_is_only_applied_after_pre_move_normalisation() -> None:
    observed_qualification = {
        (row.symbol, row.session): aligned_raw_pre_move(row) / movement_scale(row)
        > PRE_MOVE_M_THRESHOLD
        for row in ROWS
    }

    assert PRE_MOVE_M_THRESHOLD not in {row.expected_m for row in ROWS}
    assert observed_qualification[("HOOD", "2025-02-20")] is True
    assert observed_qualification[("AXON", "2025-02-20")] is False


def test_reference_pre_window_is_exactly_three_minutes() -> None:
    for row in ROWS:
        t0 = datetime.fromisoformat(row.t0)
        pre_start = datetime.fromisoformat(row.pre_start)
        assert t0.tzinfo is not None
        assert pre_start.tzinfo is not None
        assert t0 - pre_start == timedelta(minutes=3)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"production-{row.symbol}-{row.session}")
def test_production_calculator_matches_authoritative_rows(row: ReferenceRow) -> None:
    result = calculate_pre_move(
        expected_absolute_return_15m=expected_absolute_return_15m(row.atm_iv),
        p0=row.p0,
        raw_open_t0=row.raw_t0_open,
        raw_open_t0_minus_3m=row.raw_t_minus_3_open,
    )

    assert result.m_price == pytest.approx(row.expected_m, abs=5e-15)
    assert result.pre_move_m == pytest.approx(row.expected_pre_move_m, abs=2e-14)
    assert result.alignment_factor == row.p0 / row.raw_t0_open
    assert result.aligned_pre_open == row.raw_t_minus_3_open * result.alignment_factor
    assert result.raw_pre_move_price == abs(row.p0 - result.aligned_pre_open)
    assert result.passes_pre_move_threshold is (row.expected_pre_move_m > PRE_MOVE_M_THRESHOLD)


def test_exact_threshold_equality_does_not_qualify() -> None:
    result = calculate_pre_move(
        expected_absolute_return_15m=1.0,
        p0=1.0,
        raw_open_t0=1.0,
        raw_open_t0_minus_3m=1.0 - PRE_MOVE_M_THRESHOLD,
    )

    assert result.pre_move_m == PRE_MOVE_M_THRESHOLD
    assert result.passes_pre_move_threshold is False
