"""Cancellation cleanup for the pinned ib_async 2.1 request lifecycle.

Its history helper converts timeouts to empty data and its scanner helper only
cancels on success. Keep the same wire requests, but always release their state.
"""

import asyncio
from datetime import datetime
from typing import Any, cast

from ib_async import IB, ScanDataList, util


class First4IB(IB):
    RaiseRequestErrors = True
    # ib_async's request plumbing is untyped; confine that boundary here.
    client: Any
    wrapper: Any

    def finish_request(self, request_id: int) -> None:
        self.wrapper._endReq(request_id)
        # RequestError completes with an explicit result in 2.1, leaving this map.
        self.wrapper._results.pop(request_id, None)

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
