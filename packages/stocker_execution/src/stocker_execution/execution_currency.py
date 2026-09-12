"""Currency valuation for shared execution; never exchanges cash or changes prices."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Protocol

from stocker_execution.discovery import DiscoveryFx
from stocker_execution.execution_models import StockExecutionRules
from stocker_execution.ibkr import QualifiedInstrument

MAX_FX_AGE = timedelta(seconds=5)


class CurrencySource(Protocol):
    async def discovery_fx(self, currency: str) -> DiscoveryFx: ...

    async def stock_execution_rules(
        self, instrument: QualifiedInstrument
    ) -> StockExecutionRules: ...


@dataclass(frozen=True)
class ExecutionValuation:
    account_currency: str
    price_currency: str
    price_unit: float
    account_per_price_unit: float
    fx_observed_at: datetime | None
    fx_evidence: str
    minimum_quantity: int
    quantity_increment: int

    def validate(self, now: datetime) -> None:
        if not all(
            type(x) is int and x > 0 for x in (self.minimum_quantity, self.quantity_increment)
        ):
            raise ValueError("EXECUTION_CURRENCY_UNAVAILABLE: invalid order quantity rules")
        if not all(isfinite(x) and x > 0 for x in (self.price_unit, self.account_per_price_unit)):
            raise ValueError("EXECUTION_CURRENCY_UNAVAILABLE: invalid conversion")
        if self.fx_observed_at is not None and (
            self.fx_observed_at.tzinfo is None
            or not timedelta(0) <= now - self.fx_observed_at <= MAX_FX_AGE
        ):
            raise ValueError("EXECUTION_CURRENCY_UNAVAILABLE: stale or future FX quote")


async def execution_valuation(
    source: CurrencySource,
    instrument: QualifiedInstrument,
    account_currency: str | None,
    clock: Callable[[], datetime],
) -> ExecutionValuation:
    price_currency = instrument.currency
    if instrument.security_type != "STK" or not all(
        c is not None and len(c) == 3 and c.isascii() and c.isalpha() and c.isupper()
        for c in (account_currency, price_currency)
    ):
        raise ValueError(
            "EXECUTION_CURRENCY_UNAVAILABLE: concrete stock/account currencies required"
        )
    assert account_currency is not None
    # Currency labels alone do not establish quotation units or permitted lots.
    # Require matching broker metadata for every stock, including same-currency stocks.
    rules = await source.stock_execution_rules(instrument)
    unit = rules.price_unit
    evidence: list[DiscoveryFx] = []

    async def usd_sides(currency: str) -> tuple[float, float]:
        if currency == "USD":
            return 1.0, 1.0
        quote = await source.discovery_fx(currency)
        timestamp = datetime.fromisoformat(quote.observed_at)
        if (
            quote.currency != currency
            or quote.con_id <= 0
            or not all(isfinite(x) and x > 0 for x in (quote.bid, quote.ask))
            or quote.bid > quote.ask
            or timestamp.tzinfo is None
            or not timedelta(0) <= clock() - timestamp <= MAX_FX_AGE
        ):
            raise ValueError("EXECUTION_CURRENCY_UNAVAILABLE: invalid FX evidence")
        evidence.append(quote)
        if quote.symbol == currency + "USD":
            return quote.bid, quote.ask
        if quote.symbol == "USD" + currency:
            return 1 / quote.ask, 1 / quote.bid
        raise ValueError("EXECUTION_CURRENCY_UNAVAILABLE: FX pair does not match currency")

    rate = 1.0
    if price_currency != account_currency:
        _, source_ask = await usd_sides(price_currency)
        account_bid, _ = await usd_sides(account_currency)
        # Upper bound in account currency: never understate cost by using mid.
        rate = source_ask / account_bid
    valuation = ExecutionValuation(
        account_currency,
        price_currency,
        unit,
        unit * rate,
        min((datetime.fromisoformat(q.observed_at) for q in evidence), default=None),
        json.dumps([asdict(q) for q in evidence], sort_keys=True, allow_nan=False),
        rules.minimum_quantity,
        rules.quantity_increment,
    )
    valuation.validate(clock())
    return valuation
