"""Request-scoped FIRST4 observation taps for pinned ib_async 2.1.0.

This is deliberately not an event bus. Last has exactly two named owners; quote
requests have independent tickers and never enter the entry ticker's tickByTicks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ib_async import Ticker

Sink = Callable[..., None]


@dataclass
class LastLease:
    ticker: Any
    request_id: int
    entry: bool = True
    observer: bool = False


class FlowWire:
    def __init__(self, ib: Any):
        self.ib = ib
        self.lasts: dict[int, LastLease] = {}
        self.requested: dict[int, float] = {}
        self.started = time.monotonic()
        self.sinks: dict[int, Sink] = {}
        self.quotes: dict[int, tuple[Any, str]] = {}
        wrapper = ib.wrapper
        self.original_bidask = wrapper.tickByTickBidAsk
        self.original_price = wrapper.priceSizeTick
        self.original_size = wrapper.tickSize
        self.original_type = wrapper.marketDataType
        wrapper.tickByTickBidAsk = self.bidask
        wrapper.priceSizeTick = self.price
        wrapper.tickSize = self.size
        wrapper.marketDataType = self.data_type

    def valid_last(self, con_id: int) -> LastLease | None:
        lease = self.lasts.get(con_id)
        if lease and self.ib.wrapper.reqId2Ticker.get(lease.request_id) is lease.ticker:
            return lease
        self.lasts.pop(con_id, None)
        return None

    def mark_requested(self, con_id: int) -> None:
        current = time.monotonic()
        self.requested = {
            key: value for key, value in self.requested.items() if current - value < 15
        }
        self.requested[con_id] = current

    def remaining_pacing(self, con_id: int) -> float:
        # Startup quarantine covers unknown requests from the previous process.
        return max(0.0, 15.0 - (time.monotonic() - self.requested.get(con_id, self.started)))

    def retain_last(self, contract: Any, sink: Sink) -> int:
        lease = self.valid_last(contract.conId)
        if lease is None:
            if self.remaining_pacing(contract.conId):
                raise ValueError("TBT_PACING_WAIT")
            ticker = self.ib.reqTickByTickData(contract, "Last", 0, False)
            lease = self.lasts[contract.conId]
            lease.entry = False
            assert ticker is lease.ticker
        if lease.observer:
            raise ValueError("DUPLICATE_OBSERVER_OWNER")
        lease.observer = True
        self.sinks[lease.request_id] = sink
        return lease.request_id

    def release_last(self, con_id: int) -> None:
        lease = self.lasts.get(con_id)
        if lease is None:
            return
        self.sinks.pop(lease.request_id, None)
        lease.observer = False
        if not lease.entry:
            self.end_last(con_id)

    def end_last(self, con_id: int) -> None:
        lease = self.lasts.pop(con_id, None)
        if lease is None:
            return
        self.sinks.pop(lease.request_id, None)
        wrapper = self.ib.wrapper
        if wrapper.reqId2Ticker.get(lease.request_id) is not lease.ticker:
            return  # Old generation must not cancel a reused broker request ID.
        try:
            self.ib.client.cancelTickByTickData(lease.request_id)
        finally:
            wrapper.endTicker(lease.ticker, "Last")
            wrapper.reqId2Ticker.pop(lease.request_id, None)

    def quote_request(self, contract: Any, mode: str, sink: Sink) -> int:
        tbt = mode == "TBT_TRADES_TBT_QUOTES"
        if tbt and self.remaining_pacing(contract.conId):
            raise ValueError("TBT_PACING_WAIT")
        req = self.ib.client.getReqId()
        ticker = Ticker(contract=contract, defaults=self.ib.wrapper.defaults)
        self.ib.wrapper.reqId2Ticker[req] = ticker
        self.ib.wrapper._reqId2Contract[req] = contract
        self.quotes[req] = (ticker, "BidAsk" if tbt else "L1")
        self.sinks[req] = sink
        try:
            if tbt:
                self.mark_requested(contract.conId)
                self.ib.client.reqTickByTickData(req, contract, "BidAsk", 0, False)
            else:
                self.ib.client.reqMktData(req, contract, "", False, False, [])
        except Exception:
            self.release_quote(req)
            raise
        return int(req)

    def release_quote(self, req: int) -> None:
        self.sinks.pop(req, None)
        item = self.quotes.pop(req, None)
        if item is None:
            return
        ticker, kind = item
        wrapper = self.ib.wrapper
        if wrapper.reqId2Ticker.get(req) is not ticker:
            return
        try:
            if self.ib.isConnected():
                if kind == "BidAsk":
                    self.ib.client.cancelTickByTickData(req)
                else:
                    self.ib.client.cancelMktData(req)
        finally:
            wrapper.reqId2Ticker.pop(req, None)
            wrapper._reqId2Contract.pop(req, None)

    def trade(
        self,
        req: int,
        tick_type: int,
        stamp: int,
        price: float,
        size: float,
        attributes: Any,
        exchange: str,
        conditions: str,
    ) -> None:
        if sink := self.sinks.get(req):
            sink(
                req,
                "trade",
                broker_time=stamp,
                broker_precision="seconds",
                price=price,
                size=size,
                tick_type=tick_type,
                exchange=exchange,
                conditions=conditions,
                source="Last",
                attributes=tuple(
                    k for k in ("pastLimit", "unreported") if getattr(attributes, k, False)
                ),
            )

    def bidask(
        self,
        req: int,
        stamp: int,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        attributes: Any,
    ) -> None:
        if req not in self.quotes:
            self.original_bidask(req, stamp, bid, ask, bid_size, ask_size, attributes)
            return
        if sink := self.sinks.get(req):
            sink(
                req,
                "quote",
                broker_time=stamp,
                broker_precision="seconds",
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                source="BidAsk",
                attributes=tuple(
                    k for k in ("bidPastLow", "askPastHigh") if getattr(attributes, k, False)
                ),
            )

    def price(self, req: int, tick_type: int, price: float, size: float) -> None:
        if req not in self.quotes:
            self.original_price(req, tick_type, price, size)
            return
        if tick_type in (1, 2) and (sink := self.sinks.get(req)):
            name = "bid" if tick_type == 1 else "ask"
            sink(req, "quote", field_name=name, source="L1", tick_type=tick_type, **{name: price})
            sink(
                req,
                "quote",
                field_name=name + "_size",
                source="L1",
                tick_type=tick_type,
                **{name + "_size": size},
            )

    def size(self, req: int, tick_type: int, size: float) -> None:
        if req not in self.quotes:
            self.original_size(req, tick_type, size)
            return
        if tick_type in (0, 3) and (sink := self.sinks.get(req)):
            name = "bid_size" if tick_type == 0 else "ask_size"
            sink(req, "quote", field_name=name, source="L1", tick_type=tick_type, **{name: size})

    def data_type(self, req: int, data_type: int) -> None:
        self.original_type(req, data_type)
        if req in self.quotes and (sink := self.sinks.get(req)):
            sink(req, "market_data_type", tick_type=data_type, source="L1")
