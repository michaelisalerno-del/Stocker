"""Unavailable and ambiguous chains retain their actual rejection evidence."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from ib_async import Contract, RequestError

from stocker_execution.first4_broker import PaperBroker
from test_first4 import OPEN, broker
from test_first4_anchor import execution


def chain(**changes):
    return NS(
        **{
            "exchange": "SMART",
            "tradingClass": "ABC",
            "multiplier": "100",
            "underlyingConId": 1,
            "expirations": {"20250722"},
            "strikes": {97, 103},
            **changes,
        }
    )


@pytest.mark.parametrize(
    "rows,reason",
    [
        ([], "OPTION_CHAIN_EMPTY"),
        ([chain(tradingClass="ABC1")], "NO_PERMITTED_STANDARD_OPTION_CHAIN"),
        ([chain(multiplier="10")], "NO_PERMITTED_STANDARD_OPTION_CHAIN"),
        ([chain(exchange="CBOE")], "NO_PERMITTED_STANDARD_OPTION_CHAIN"),
        ([chain(), chain()], "AMBIGUOUS_STANDARD_OPTION_CHAIN"),
    ],
)
def test_chain_failures_are_distinct_and_retain_bounded_metadata(tmp_path, rows, reason):
    b = broker(tmp_path)
    b.chain = AsyncMock(return_value=rows)
    b.ib.reqContractDetailsAsync = AsyncMock()
    with pytest.raises(ValueError) as error:
        asyncio.run(b.contracts(Contract(conId=1, symbol="ABC"), 100, OPEN))
    assert str(error.value) == reason
    detail = error.value.detail["option_chain"]
    assert detail["returned"] == len(rows)
    assert detail["required"] == {"exchange": "SMART", "trading_class": "ABC", "multiplier": "100"}
    assert len(detail["rows"]) == len(rows)
    b.ib.reqContractDetailsAsync.assert_not_awaited()


def test_empty_chain_is_rejected_before_anchor_wait(monkeypatch):
    async def check():
        runtime, ib, clock, event, underlying = execution(monkeypatch)
        runtime.broker.standard_chain = PaperBroker.standard_chain.__get__(runtime.broker)
        await asyncio.wait_for(runtime.execute(event, underlying), 0.5)
        detail = runtime.store.outcome.call_args.args[2]
        assert detail["error"] == "OPTION_CHAIN_EMPTY"
        assert detail["stage"] == "OPTION_CHAIN"
        assert detail["option_chain"]["returned"] == 0
        runtime.broker.enter.assert_not_awaited()
        assert not ib.wrapper._futures and not ib.wrapper._results
        assert not ib.wrapper.reqId2Ticker

    asyncio.run(check())


def test_chain_diagnostics_are_bounded(tmp_path):
    b = broker(tmp_path)
    b.chain = AsyncMock(return_value=[chain(tradingClass="ABC1") for _ in range(1000)])
    with pytest.raises(ValueError) as error:
        asyncio.run(b.standard_chain(Contract(conId=1, symbol="ABC")))
    detail = error.value.detail["option_chain"]
    assert detail["returned"] == 1000 and detail["matching"] == 0
    assert detail["truncated"] and len(detail["rows"]) == 20


def test_standard_chain_reuses_cache_and_does_not_adopt_adjusted_alternative(tmp_path):
    b = broker(tmp_path)
    permitted = chain()
    b.ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[chain(tradingClass="ABC1"), permitted])
    underlying = Contract(conId=1, symbol="ABC")

    async def check():
        assert await b.standard_chain(underlying) is permitted
        assert await b.standard_chain(underlying) is permitted

    asyncio.run(check())
    b.ib.reqSecDefOptParamsAsync.assert_awaited_once()


def test_chain_request_failure_is_not_relabelled_as_valid_empty_response(tmp_path):
    b = broker(tmp_path)
    b.ib.reqSecDefOptParamsAsync = AsyncMock(side_effect=RequestError(12, 354, "No permissions"))
    with pytest.raises(RequestError) as error:
        asyncio.run(b.standard_chain(Contract(conId=1, symbol="ABC")))
    assert error.value.code == 354
    assert not b.chains
