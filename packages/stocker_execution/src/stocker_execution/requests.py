"""Bounded IB requests with cancellation cleanup and request-local quote evidence."""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from ib_async import IB, Ticker, util


class BrokerConnection(IB):
    RaiseRequestErrors = True
    client: Any
    wrapper: Any

    def finish_request(self, request_id: int) -> None:
        self.wrapper._endReq(request_id)
        # RequestError completes with an explicit result in 2.1, leaving this map.
        self.wrapper._results.pop(request_id, None)

    @contextmanager
    def market_data(
        self, contract: Any, update: Callable[[Any], None] | None = None
    ) -> Iterator[tuple[Ticker, asyncio.Future[Any]]]:
        """One temporary stream, using the wrapper's request/error lifecycle.

        startTicker reuses a ticker by contract hash, including cached prices and
        concurrent tick-by-tick updates. Give this request its own ticker so only
        this wire request can supply its evidence. Never replace a strategy ticker.
        """
        request_id = self.client.getReqId()
        future = self.wrapper.startReq(request_id, contract)
        ticker = Ticker(contract=contract, defaults=self.wrapper.defaults)
        ticker.marketDataType = 0  # Require a type response on this request.
        self.wrapper.reqId2Ticker[request_id] = ticker
        self.wrapper.ticker2ReqId["mktData"][ticker] = request_id
        if update is not None:
            ticker.updateEvent += update
        try:
            self.client.reqMktData(request_id, contract, "", False, False, [])
            yield ticker, future
        finally:
            if update is not None:
                ticker.updateEvent -= update
            try:
                if self.isConnected():
                    self.client.cancelMktData(request_id)
            finally:
                self.wrapper.endTicker(ticker, "mktData")
                self.wrapper.reqId2Ticker.pop(request_id, None)
                future.cancel()
                if not future.cancelled():
                    future.exception()
                self.finish_request(request_id)

    async def reqTickersAsync(
        self, *contracts: Any, regulatorySnapshot: bool = False
    ) -> list[Ticker]:
        requests = []
        try:
            for contract in contracts:
                request_id = self.client.getReqId()
                future = self.wrapper.startReq(request_id, contract)
                ticker = self.wrapper.startTicker(request_id, contract, "snapshot")
                requests.append((request_id, future, ticker))
                self.client.reqMktData(request_id, contract, "", True, regulatorySnapshot, [])
            await asyncio.gather(*(future for _, future, _ in requests))
            return [ticker for _, _, ticker in requests]
        finally:
            for request_id, future, ticker in requests:
                completed = future.done() and not future.cancelled() and future.exception() is None
                future.cancel()
                try:
                    if not completed and self.isConnected():
                        self.client.cancelMktData(request_id)
                finally:
                    self.wrapper.endTicker(ticker, "snapshot")
                    self.wrapper.reqId2Ticker.pop(request_id, None)
                    self.finish_request(request_id)
            await asyncio.gather(*(future for _, future, _ in requests), return_exceptions=True)

    async def history(self, contract: Any, end: datetime, duration: str) -> list[Any]:
        request_id = self.client.getReqId()
        bars: list[Any] = []
        future = self.wrapper.startReq(request_id, contract, bars)
        try:
            self.client.reqHistoricalData(
                request_id,
                contract,
                util.formatIBDatetime(end),
                duration,
                "1 min",
                "TRADES",
                False,
                2,
                False,
                [],
            )
            async with asyncio.timeout(15):
                await future
            return list(bars)
        finally:
            try:
                if self.isConnected():
                    self.client.cancelHistoricalData(request_id)
            finally:
                self.finish_request(request_id)

    async def reqContractDetailsAsync(self, contract: Any) -> list[Any]:
        request_id = self.client.getReqId()
        future = self.wrapper.startReq(request_id, contract)
        try:
            self.client.reqContractDetails(request_id, contract)
            async with asyncio.timeout(15):
                return list(await future)
        finally:
            # This API has no cancelContractDetails. Late replies are ignored by
            # the wrapper after removing the future; never retained as a retry.
            self.finish_request(request_id)

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list[Any]:
        request_id = self.client.getReqId()
        future = self.wrapper.startReq(request_id)
        try:
            self.client.reqSecDefOptParams(
                request_id, underlyingSymbol, futFopExchange, underlyingSecType, underlyingConId
            )
            async with asyncio.timeout(15):
                return list(await future)
        finally:
            self.finish_request(request_id)
