"""Deterministic bounded option-expiry and strike selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal


class DteBucket(StrEnum):
    ZERO_DTE = "0DTE"
    ONE_DTE = "1DTE"
    THREE_TO_FIVE_DTE = "3_TO_5_DTE"


@dataclass(frozen=True)
class ExpirySelection:
    bucket: DteBucket
    expiry: date | None
    calendar_day_dte: int | None
    reason: str | None


@dataclass(frozen=True)
class ExactOptionContractRequest:
    underlying_contract_id: int
    expiry: date
    strike: float
    right: Literal["C", "P"]
    exchange: str
    trading_class: str
    exact_qualification_required: bool = True


def select_expiries(
    session_date: date,
    available_expiries: Iterable[date],
) -> dict[DteBucket, ExpirySelection]:
    """Choose only expiries actually present in each calendar-day bucket."""

    unique = sorted(set(available_expiries))
    allowed: dict[DteBucket, tuple[int, ...]] = {
        DteBucket.ZERO_DTE: (0,),
        DteBucket.ONE_DTE: (1,),
        DteBucket.THREE_TO_FIVE_DTE: (3, 4, 5),
    }
    result: dict[DteBucket, ExpirySelection] = {}
    for bucket, dtes in allowed.items():
        candidates = [
            (expiry - session_date, expiry)
            for expiry in unique
            if (expiry - session_date).days in dtes
        ]
        if not candidates:
            result[bucket] = ExpirySelection(
                bucket=bucket,
                expiry=None,
                calendar_day_dte=None,
                reason="no_expiry_in_bucket",
            )
            continue
        _, selected = min(candidates, key=lambda item: (item[0].days, item[1]))
        result[bucket] = ExpirySelection(
            bucket=bucket,
            expiry=selected,
            calendar_day_dte=(selected - session_date).days,
            reason=None,
        )
    return result


def select_atm_strike(underlying_reference: float, strikes: Iterable[float]) -> float:
    """Choose nearest strike; an exact tie resolves to the lower strike."""

    available = sorted(set(strikes))
    if not available:
        raise ValueError("at least one strike is required")
    return min(available, key=lambda strike: (abs(strike - underlying_reference), strike))


def bounded_contract_requests(
    *,
    underlying_contract_id: int,
    expiry: date,
    strikes: Iterable[float],
    underlying_reference: float,
    strike_steps: int,
    exchange: str,
    trading_class: str,
) -> tuple[ExactOptionContractRequest, ...]:
    """Return at most ``(2 * steps + 1) * 2`` exact contracts.

    This consumes chain metadata only. It never requests or streams a complete
    option chain, and it never expands the bound after a qualification failure.
    """

    if strike_steps < 0:
        raise ValueError("strike_steps must be nonnegative")
    available = sorted(set(strikes))
    atm = select_atm_strike(underlying_reference, available)
    atm_index = available.index(atm)
    selected = available[
        max(0, atm_index - strike_steps) : min(len(available), atm_index + strike_steps + 1)
    ]
    rights: tuple[Literal["C", "P"], ...] = ("C", "P")
    return tuple(
        ExactOptionContractRequest(
            underlying_contract_id=underlying_contract_id,
            expiry=expiry,
            strike=strike,
            right=right,
            exchange=exchange,
            trading_class=trading_class,
        )
        for strike in selected
        for right in rights
    )
