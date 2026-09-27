"""Bounded IB requests with cancellation cleanup and request-local quote evidence."""

import asyncio
from collections.abc import Callable
from time import monotonic_ns
from typing import Any

from ib_async import IB, util
from ib_async.objects import BarDataList
from ib_async.wrapper import Wrapper

from stocker_execution.config import MarketDataConfig
from stocker_execution.pacing import CORE, EXPOSURE, PacedClient, drain, priority, traffic


class DepthWrapper(Wrapper):
    """Raw depth callbacks: avoid ib_async 2.1's dictionary overwrite row semantics."""

    def securityDefinitionOptionParameter(
        self,
        reqId: int,
        exchange: str,
        underlyingConId: int | str,
        tradingClass: str,
        multiplier: str,
        expirations: list[str],
        strikes: list[float],
    ) -> None:
        if reqId not in self._futures:
            return  # Ignore late rows after completion, cancellation or an invalid identity.
        # ib_async 2.1's decoder forwards this integer protocol field as text.
        identity = str(underlyingConId)
        if not identity.isascii() or not identity.isdecimal() or int(identity) <= 0:
            self._results.pop(reqId, None)
            error = ValueError("OPTION_CHAIN_UNDERLYING_ID_INVALID")
            self._endReq(reqId, error, success=False)  # type: ignore[no-untyped-call]
            return
        super().securityDefinitionOptionParameter(
            reqId, exchange, int(identity), tradingClass, multiplier, expirations, strikes
        )

    def updateMktDepth(
        self, reqId: int, position: int, operation: int, side: int, price: float, size: float
    ) -> None:
        self.updateMktDepthL2(reqId, position, "", operation, side, price, size)

    def updateMktDepthL2(
        self,
        reqId: int,
        position: int,
        marketMaker: str,
        operation: int,
        side: int,
        price: float,
        size: float,
        isSmartDepth: bool = False,
    ) -> None:
        callback = getattr(self, "depth_handlers", {}).get(reqId)
        if callback:
            callback(
                {
                    "kind": "ROW",
                    "position": position,
                    "operation": operation,
                    "side": side,
                    "price": price,
                    "size": float(size),
                    "market_maker": marketMaker[:64],
                    "smart": isSmartDepth,
                    "monotonic_ns": monotonic_ns(),
                }
            )

    def error(
        self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""
    ) -> None:
        if errorCode == 317:
            callback = getattr(self, "depth_handlers", {}).get(reqId)
            if callback:
                callback({"kind": "RESET", "reason": "IBKR_317", "monotonic_ns": monotonic_ns()})
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)


class BrokerConnection(IB):
    RaiseRequestErrors = True
    client: Any
    wrapper: Any

    def __init__(self, limits: MarketDataConfig | None = None):
        super().__init__()
        self.client.apiEnd -= self.disconnectedEvent
        self.wrapper = DepthWrapper(self, defaults=self.wrapper.defaults)
        self.wrapper.depth_handlers = {}
        self.client = PacedClient(self.wrapper, limits or MarketDataConfig())
        self.client.apiEnd += self.disconnectedEvent

    def finish_request(self, request_id: Any) -> None:
        self.wrapper._endReq(request_id)
        # RequestError completes with an explicit result in 2.1, leaving this map.
        self.wrapper._results.pop(request_id, None)

    async def bounded_request(
        self,
        send: Callable[[Any], None],
        contract: Any = None,
        container: Any = None,
        key: Any = None,
        historical: bool = False,
    ) -> Any:
        request_id = key if key is not None else self.client.getReqId()
        future = self.wrapper.startReq(request_id, contract, container)
        epoch = self.client.connection_epoch
        completed = False
        parent = traffic.get()
        deadline = min(
            asyncio.get_running_loop().time() + 15, parent.deadline if parent else float("inf")
        )
        try:
            with priority(parent.priority if parent else CORE, deadline):
                send(request_id)
                async with asyncio.timeout_at(deadline):
                    result = await future
                    completed = True
                    return result
        finally:
            future.cancel()
            same_connection = self.client.connection_epoch == epoch
            if same_connection and self.wrapper._futures.get(request_id) in (None, future):
                self.finish_request(request_id)
            if historical and not completed and same_connection and self.isConnected():
                with priority(EXPOSURE, asyncio.get_running_loop().time() + 5) as cancellation:
                    self.client.cancelHistoricalData(request_id)
                    await drain(cancellation)

    async def reqContractDetailsAsync(self, contract: Any) -> list[Any]:
        # No cancelContractDetails API; pending wire requests expire and late results are ignored.
        return list(
            await self.bounded_request(
                lambda rid: self.client.reqContractDetails(rid, contract), contract
            )
        )

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list[Any]:
        return list(
            await self.bounded_request(
                lambda rid: self.client.reqSecDefOptParams(
                    rid, underlyingSymbol, futFopExchange, underlyingSecType, underlyingConId
                )
            )
        )

    async def reqHistoricalDataAsync(
        self,
        contract: Any,
        endDateTime: Any,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int = 2,
        keepUpToDate: bool = False,
        chartOptions: Any = None,
        timeout: float = 15,
    ) -> Any:
        if keepUpToDate or barSizeSetting not in {"1 min", "1 day"} or whatToShow != "TRADES":
            raise ValueError("HISTORICAL_ENDPOINT_NOT_SUPPORTED_BY_PACING_POLICY")
        bars = BarDataList()  # type: ignore[no-untyped-call]
        return await self.bounded_request(
            lambda rid: self.client.reqHistoricalData(
                rid,
                contract,
                util.formatIBDatetime(endDateTime),
                durationStr,
                barSizeSetting,
                whatToShow,
                useRTH,
                formatDate,
                False,
                chartOptions or [],
            ),
            contract,
            bars,
            historical=True,
        )

    async def reqHistoricalScheduleAsync(
        self, contract: Any, numDays: int, endDateTime: Any = "", useRTH: bool = False
    ) -> Any:
        return await self.bounded_request(
            lambda rid: self.client.reqHistoricalData(
                rid,
                contract,
                util.formatIBDatetime(endDateTime),
                f"{numDays} D",
                "1 day",
                "SCHEDULE",
                useRTH,
                1,
                False,
                [],
            ),
            contract,
            historical=True,
        )

    async def reqMktDepthExchangesAsync(self) -> Any:
        return await self.bounded_request(
            lambda _: self.client.reqMktDepthExchanges(), key="mktDepthExchanges"
        )

    async def reqMarketRuleAsync(self, marketRuleId: int) -> Any:
        return await self.bounded_request(
            lambda _: self.client.reqMarketRule(marketRuleId), key=f"marketRule-{marketRuleId}"
        )
