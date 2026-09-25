"""Cancellation cleanup for the pinned ib_async 2.1 request lifecycle.

Its history helper converts timeouts to empty data and its scanner helper only
cancels on success. Keep the same wire requests, but always release their state.
"""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, cast

from ib_async import IB, ScanDataList, TickAttribLast, Ticker, util

from stocker_execution.first4_flow_wire import FlowWire, LastLease


class First4IB(IB):
    RaiseRequestErrors = True
    # ib_async's request plumbing is untyped; confine that boundary here.
    client: Any
    wrapper: Any

    def __init__(self) -> None:
        super().__init__()
        # The pinned wrapper discards the wire trade timestamp in favour of
        # packet receipt time. Adapt only this instance, before any connection.
        self._original_trade_tick = self.wrapper.tickByTickAllLast
        self.wrapper.tickByTickAllLast = self.trade_tick
        self.flow_wire = FlowWire(self)

    def reqTickByTickData(
        self, contract: Any, tickType: str, numberOfTicks: int = 0, ignoreSize: bool = False
    ) -> Ticker:
        if tickType == "Last" and (lease := self.flow_wire.valid_last(contract.conId)):
            lease.entry = True
            return cast(Ticker, lease.ticker)
        ticker = super().reqTickByTickData(contract, tickType, numberOfTicks, ignoreSize)
        self.flow_wire.mark_requested(contract.conId)
        if tickType == "Last":
            req = self.wrapper.ticker2ReqId["Last"][ticker]
            self.flow_wire.lasts[contract.conId] = LastLease(ticker, req)
        return ticker

    def cancelTickByTickData(self, contract: Any, tickType: str) -> bool:
        if tickType == "Last" and (lease := self.flow_wire.lasts.get(contract.conId)):
            lease.entry = False
            if not lease.observer:
                self.flow_wire.end_last(contract.conId)
            return True
        return super().cancelTickByTickData(contract, tickType)

    def flow_last_retained(self, request_id: int) -> bool:
        return any(x.request_id == request_id and x.observer for x in self.flow_wire.lasts.values())

    def trade_tick(
        self,
        request_id: int,
        tick_type: int,
        timestamp: int,
        price: float,
        size: float,
        attributes: TickAttribLast,
        exchange: str,
        conditions: str,
    ) -> None:
        self.flow_wire.trade(
            request_id, tick_type, timestamp, price, size, attributes, exchange, conditions
        )
        self._original_trade_tick(
            request_id, tick_type, timestamp, price, size, attributes, exchange, conditions
        )
        ticker = self.wrapper.reqId2Ticker.get(request_id)
        if ticker is not None:
            # Decoder dispatch is synchronous; updateEvent is emitted only after
            # the packet. Keep ticker.time/lastTime as receipt times for quotes.
            ticker.tickByTicks[-1] = ticker.tickByTicks[-1]._replace(
                time=datetime.fromtimestamp(timestamp, UTC)
            )

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
                True,
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

    async def scanner(self, subscription: Any, filters: list[Any]) -> list[Any]:
        rows = cast(Any, ScanDataList)()
        rows.reqId = self.client.getReqId()
        rows.subscription = subscription
        rows.scannerSubscriptionOptions = []
        rows.scannerSubscriptionFilterOptions = filters
        future = self.wrapper.startReq(rows.reqId, container=rows)
        try:
            self.wrapper.startSubscription(rows.reqId, rows)
            self.client.reqScannerSubscription(rows.reqId, subscription, [], filters)
            async with asyncio.timeout(5):
                await future
            return list(rows)
        finally:
            try:
                if self.isConnected():
                    self.client.cancelScannerSubscription(rows.reqId)
            finally:
                self.wrapper.endSubscription(rows)
                self.finish_request(rows.reqId)

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
