"""Executable-side, record-only shadow structure accounting."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from stocker_prospective.market_data import MarketDataType


class OptionExecutableQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_id: int
    right: Literal["C", "P"]
    strike: float
    bid: float | None
    ask: float | None
    bid_size: float | None = None
    ask_size: float | None = None
    provider_timestamp: datetime
    receive_timestamp: datetime
    market_data_type: MarketDataType
    multiplier: int
    stale: bool


class ShadowStructureType(StrEnum):
    LONG_ATM_CALL = "long_atm_call"
    LONG_ATM_PUT = "long_atm_put"
    LONG_ATM_STRADDLE = "long_atm_straddle"
    CALL_DEBIT_SPREAD = "call_debit_spread"
    PUT_DEBIT_SPREAD = "put_debit_spread"


class ShadowLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_id: int
    side: Literal["long", "short"]
    right: Literal["C", "P"]
    strike: float
    multiplier: int
    entry_quote_timestamp: datetime
    exit_quote_timestamp: datetime
    entry_executable_price: float
    exit_executable_price: float


class ShadowValuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    structure_type: ShadowStructureType
    interpretation: Literal["bullish", "bearish", "non_directional"]
    legs: tuple[ShadowLeg, ...]
    entry_debit: float | None
    exit_credit: float | None
    gross_return_on_debit: float | None
    gross_pnl: float | None
    estimated_fees: float | None
    target_timestamp: datetime
    actual_quote_timestamp: datetime | None
    capture_lag_seconds: float | None
    market_data_type: MarketDataType | None
    spread_quality: Literal["valid_executable_market", "rejected"]
    rejection_reason: str | None


def value_shadow_structures(
    *,
    entry_quotes: dict[tuple[str, float], OptionExecutableQuote],
    exit_quotes: dict[tuple[str, float], OptionExecutableQuote],
    atm_strike: float,
    lower_strike: float,
    upper_strike: float,
    target_timestamp: datetime,
    maximum_capture_lag: timedelta,
    estimated_fee_per_contract: float = 0.0,
) -> tuple[ShadowValuation, ...]:
    """Value all fixed structures using asks to enter and bids to exit.

    Directional call and put structures are retained independently. No option
    return feeds back into M1, its threshold, or universe membership.
    """

    definitions: tuple[
        tuple[
            ShadowStructureType,
            Literal["bullish", "bearish", "non_directional"],
            tuple[tuple[Literal["long", "short"], Literal["C", "P"], float], ...],
        ],
        ...,
    ] = (
        (
            ShadowStructureType.LONG_ATM_CALL,
            "bullish",
            (("long", "C", atm_strike),),
        ),
        (
            ShadowStructureType.LONG_ATM_PUT,
            "bearish",
            (("long", "P", atm_strike),),
        ),
        (
            ShadowStructureType.LONG_ATM_STRADDLE,
            "non_directional",
            (("long", "C", atm_strike), ("long", "P", atm_strike)),
        ),
        (
            ShadowStructureType.CALL_DEBIT_SPREAD,
            "bullish",
            (("long", "C", atm_strike), ("short", "C", upper_strike)),
        ),
        (
            ShadowStructureType.PUT_DEBIT_SPREAD,
            "bearish",
            (("long", "P", atm_strike), ("short", "P", lower_strike)),
        ),
    )
    return tuple(
        _value_one(
            structure_type=structure_type,
            interpretation=interpretation,
            definition=definition,
            entry_quotes=entry_quotes,
            exit_quotes=exit_quotes,
            target_timestamp=target_timestamp,
            maximum_capture_lag=maximum_capture_lag,
            estimated_fee_per_contract=estimated_fee_per_contract,
        )
        for structure_type, interpretation, definition in definitions
    )


def _value_one(
    *,
    structure_type: ShadowStructureType,
    interpretation: Literal["bullish", "bearish", "non_directional"],
    definition: tuple[tuple[Literal["long", "short"], Literal["C", "P"], float], ...],
    entry_quotes: dict[tuple[str, float], OptionExecutableQuote],
    exit_quotes: dict[tuple[str, float], OptionExecutableQuote],
    target_timestamp: datetime,
    maximum_capture_lag: timedelta,
    estimated_fee_per_contract: float,
) -> ShadowValuation:
    pairs: list[
        tuple[
            Literal["long", "short"],
            OptionExecutableQuote,
            OptionExecutableQuote,
        ]
    ] = []
    for side, right, strike in definition:
        entry = entry_quotes.get((right, strike))
        exit_quote = exit_quotes.get((right, strike))
        if entry is None or exit_quote is None:
            return _rejected(structure_type, interpretation, target_timestamp, "missing_leg")
        pairs.append((side, entry, exit_quote))

    entry_reason = _quote_rejection([pair[1] for pair in pairs], target_timestamp=None)
    if entry_reason is not None:
        return _rejected(structure_type, interpretation, target_timestamp, entry_reason)

    entry_debit = sum(_entry_contribution(side, entry) for side, entry, _exit_quote in pairs)
    if entry_debit <= 0:
        return _rejected(
            structure_type,
            interpretation,
            target_timestamp,
            "invalid_or_nonpositive_debit",
        )

    exit_reason = _quote_rejection(
        [pair[2] for pair in pairs],
        target_timestamp=target_timestamp,
        maximum_capture_lag=maximum_capture_lag,
    )
    if exit_reason is not None:
        return _rejected(structure_type, interpretation, target_timestamp, exit_reason)

    multipliers = {entry.multiplier for _, entry, _ in pairs} | {
        exit_quote.multiplier for _, _, exit_quote in pairs
    }
    if len(multipliers) != 1:
        return _rejected(
            structure_type,
            interpretation,
            target_timestamp,
            "multiplier_mismatch",
        )
    multiplier = multipliers.pop()
    exit_credit = sum(_exit_contribution(side, exit_quote) for side, _entry, exit_quote in pairs)
    actual_timestamp = max(exit_quote.provider_timestamp for _, _, exit_quote in pairs)
    lag = (actual_timestamp - target_timestamp).total_seconds()
    legs = tuple(
        ShadowLeg(
            contract_id=entry.contract_id,
            side=side,
            right=entry.right,
            strike=entry.strike,
            multiplier=multiplier,
            entry_quote_timestamp=entry.provider_timestamp,
            exit_quote_timestamp=exit_quote.provider_timestamp,
            entry_executable_price=entry.ask if side == "long" else entry.bid,  # type: ignore[arg-type]
            exit_executable_price=exit_quote.bid if side == "long" else exit_quote.ask,  # type: ignore[arg-type]
        )
        for side, entry, exit_quote in pairs
    )
    gross_pnl = (exit_credit - entry_debit) * multiplier
    fees = estimated_fee_per_contract * len(legs) * 2
    return ShadowValuation(
        structure_type=structure_type,
        interpretation=interpretation,
        legs=legs,
        entry_debit=entry_debit,
        exit_credit=exit_credit,
        gross_return_on_debit=(exit_credit - entry_debit) / entry_debit,
        gross_pnl=gross_pnl,
        estimated_fees=fees,
        target_timestamp=target_timestamp,
        actual_quote_timestamp=actual_timestamp,
        capture_lag_seconds=lag,
        market_data_type=MarketDataType.LIVE,
        spread_quality="valid_executable_market",
        rejection_reason=None,
    )


def _quote_rejection(
    quotes: list[OptionExecutableQuote],
    *,
    target_timestamp: datetime | None,
    maximum_capture_lag: timedelta | None = None,
) -> str | None:
    if any(quote.bid is None or quote.ask is None for quote in quotes):
        return "missing_quote"
    if any(quote.bid < 0 or quote.ask <= 0 for quote in quotes):  # type: ignore[operator]
        return "invalid_market"
    if any(quote.bid > quote.ask for quote in quotes):  # type: ignore[operator]
        return "crossed_market"
    if any(quote.stale for quote in quotes):
        return "stale_quote"
    if any(not quote.market_data_type.primary_eligible for quote in quotes):
        return "blocked_non_live_market_data"
    if target_timestamp is not None and maximum_capture_lag is not None:
        actual = max(quote.provider_timestamp for quote in quotes)
        lag = actual - target_timestamp
        if lag < timedelta(0) or lag > maximum_capture_lag:
            return "capture_lag_exceeded"
    return None


def _entry_contribution(
    side: Literal["long", "short"],
    quote: OptionExecutableQuote,
) -> float:
    if side == "long":
        assert quote.ask is not None
        return quote.ask
    assert quote.bid is not None
    return -quote.bid


def _exit_contribution(
    side: Literal["long", "short"],
    quote: OptionExecutableQuote,
) -> float:
    if side == "long":
        assert quote.bid is not None
        return quote.bid
    assert quote.ask is not None
    return -quote.ask


def _rejected(
    structure_type: ShadowStructureType,
    interpretation: Literal["bullish", "bearish", "non_directional"],
    target_timestamp: datetime,
    reason: str,
) -> ShadowValuation:
    return ShadowValuation(
        structure_type=structure_type,
        interpretation=interpretation,
        legs=(),
        entry_debit=None,
        exit_credit=None,
        gross_return_on_debit=None,
        gross_pnl=None,
        estimated_fees=None,
        target_timestamp=target_timestamp,
        actual_quote_timestamp=None,
        capture_lag_seconds=None,
        market_data_type=None,
        spread_quality="rejected",
        rejection_reason=reason,
    )
