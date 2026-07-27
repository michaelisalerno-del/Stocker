"""Official ``ibapi`` callback bridge, imported only on an IBKR server."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

from stocker_prospective.ibkr import IBKRMarketDataAdapter, require_official_ibkr_api
from stocker_prospective.market_data import MarketDataType, RealtimeBarUpdate


def create_official_stock_contract(symbol: str) -> Any:
    """Build the narrow STK contract used for exact qualification."""

    require_official_ibkr_api()
    from ibapi.contract import Contract

    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def create_official_option_contract(
    *,
    symbol: str,
    expiry: date,
    strike: float,
    right: str,
    multiplier: int,
    exchange: str,
    trading_class: str,
) -> Any:
    """Build one exact bounded OPT contract; no chain streaming is involved."""

    if right not in {"C", "P"}:
        raise ValueError("option right must be C or P")
    require_official_ibkr_api()
    from ibapi.contract import Contract

    contract = Contract()
    contract.symbol = symbol
    contract.secType = "OPT"
    contract.lastTradeDateOrContractMonth = expiry.strftime("%Y%m%d")
    contract.strike = strike
    contract.right = right
    contract.multiplier = str(multiplier)
    contract.exchange = exchange
    contract.currency = "USD"
    contract.tradingClass = trading_class
    return contract


def create_official_callback_client(adapter: IBKRMarketDataAdapter) -> Any:
    """Create the official EWrapper/EClient bridge without exposing it to HTTP."""

    require_official_ibkr_api()
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    class _StockerOfficialMarketDataClient(EWrapper, EClient):  # type: ignore[misc]
        def __init__(self) -> None:
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self._market_data_types: dict[int, str] = {}

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            # The official callback name contains "order", but the value also
            # seeds market-data request IDs. It never creates an order surface.
            adapter.request_ids.synchronise(orderId)
            # nextValidId completes the socket handshake; it does not confirm
            # the market-data type for any request.
            adapter.on_connected(None)

        def error(
            self,
            reqId: int,
            errorTime: int,
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ) -> None:
            # Current official TWS API releases include the broker timestamp
            # between the request ID and error code.
            adapter.on_error(reqId, errorCode, errorString)

        def connectionClosed(self) -> None:  # noqa: N802
            adapter.on_connection_closed()

        def marketDataType(self, reqId: int, marketDataType: int) -> None:  # noqa: N802
            mapped = {
                1: MarketDataType.LIVE,
                2: MarketDataType.FROZEN,
                3: MarketDataType.DELAYED,
                4: MarketDataType.DELAYED_FROZEN,
            }.get(marketDataType)
            if mapped is None:
                adapter.on_error(
                    reqId,
                    marketDataType,
                    "unknown_market_data_type",
                )
                return
            self._market_data_types[reqId] = mapped.value
            adapter.on_market_data_type(reqId, mapped)

        def currentTime(self, time: int) -> None:  # noqa: N802
            adapter.on_current_time(datetime.fromtimestamp(time, tz=UTC))

        def tickByTickBidAsk(  # noqa: N802
            self,
            reqId: int,
            time: int,
            bidPrice: float,
            askPrice: float,
            bidSize: Any,
            askSize: Any,
            tickAttribBidAsk: Any,
        ) -> None:
            adapter.on_tick_by_tick_bidask(
                reqId,
                {
                    "provider_timestamp_utc": datetime.fromtimestamp(time, tz=UTC).isoformat(),
                    "bid": float(bidPrice),
                    "ask": float(askPrice),
                    "bid_size": float(bidSize),
                    "ask_size": float(askSize),
                    "bid_past_low": getattr(tickAttribBidAsk, "bidPastLow", None),
                    "ask_past_high": getattr(tickAttribBidAsk, "askPastHigh", None),
                    "market_data_type": self._market_data_types.get(reqId),
                },
            )

        def tickByTickAllLast(  # noqa: N802
            self,
            reqId: int,
            tickType: int,
            time: int,
            price: float,
            size: Any,
            tickAttribLast: Any,
            exchange: str,
            specialConditions: str,
        ) -> None:
            adapter.on_tick_by_tick_trade(
                reqId,
                {
                    "provider_timestamp_utc": datetime.fromtimestamp(time, tz=UTC).isoformat(),
                    "tick_type": tickType,
                    "price": float(price),
                    "size": float(size),
                    "past_limit": getattr(tickAttribLast, "pastLimit", None),
                    "unreported": getattr(tickAttribLast, "unreported", None),
                    "exchange": exchange or None,
                    "conditions": tuple(item for item in specialConditions.split(",") if item),
                    "market_data_type": self._market_data_types.get(reqId),
                },
            )

        def updateMktDepth(  # noqa: N802
            self,
            reqId: int,
            position: int,
            operation: int,
            side: int,
            price: float,
            size: Any,
        ) -> None:
            adapter.on_depth_update(
                reqId,
                self._depth_payload(
                    position=position,
                    operation=operation,
                    side=side,
                    price=price,
                    size=size,
                    market_maker=None,
                    smart_depth=False,
                ),
            )

        def updateMktDepthL2(  # noqa: N802
            self,
            reqId: int,
            position: int,
            marketMaker: str,
            operation: int,
            side: int,
            price: float,
            size: Any,
            isSmartDepth: bool,
        ) -> None:
            adapter.on_depth_update(
                reqId,
                self._depth_payload(
                    position=position,
                    operation=operation,
                    side=side,
                    price=price,
                    size=size,
                    market_maker=marketMaker or None,
                    smart_depth=isSmartDepth,
                ),
            )

        @staticmethod
        def _depth_payload(
            *,
            position: int,
            operation: int,
            side: int,
            price: float,
            size: Any,
            market_maker: str | None,
            smart_depth: bool,
        ) -> dict[str, Any]:
            return {
                "position": position,
                "operation": {
                    0: "insert",
                    1: "update",
                    2: "remove",
                }.get(operation, f"unknown_{operation}"),
                "side": {0: "ask", 1: "bid"}.get(side, f"unknown_{side}"),
                "price": float(price),
                "size": float(size),
                "market_maker_or_exchange": market_maker,
                "smart_depth": smart_depth,
            }

        def mktDepthExchanges(self, descriptions: list[Any]) -> None:  # noqa: N802
            adapter.on_depth_exchanges(
                tuple(
                    {
                        "exchange": getattr(item, "exchange", None),
                        "security_type": getattr(item, "secType", None),
                        "listing_exchange": getattr(item, "listingExch", None),
                        "service_data_type": getattr(item, "serviceDataType", None),
                        "aggregated_group": getattr(item, "aggGroup", None),
                    }
                    for item in descriptions
                )
            )

        def securityDefinitionOptionParameter(  # noqa: N802
            self,
            reqId: int,
            exchange: str,
            underlyingConId: int,
            tradingClass: str,
            multiplier: str,
            expirations: set[str],
            strikes: set[float],
        ) -> None:
            adapter.on_option_parameter(
                reqId,
                {
                    "exchange": exchange,
                    "underlying_contract_id": underlyingConId,
                    "trading_class": tradingClass,
                    "multiplier": multiplier,
                    "expirations": sorted(expirations),
                    "strikes": sorted(strikes),
                },
            )

        def securityDefinitionOptionParameterEnd(self, reqId: int) -> None:  # noqa: N802
            adapter.on_option_parameter_end(reqId)

        def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
            adapter.on_contract_details(reqId, contractDetails)

        def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
            adapter.on_contract_details_end(reqId)

        def tickPrice(  # noqa: N802
            self,
            reqId: int,
            tickType: int,
            price: float,
            attrib: Any,
        ) -> None:
            adapter.on_quote_update(
                reqId,
                {
                    "field": {
                        1: "bid",
                        2: "ask",
                        4: "last",
                        9: "close",
                    }.get(tickType, f"price_tick_{tickType}"),
                    "value": (
                        None
                        if not math.isfinite(price) or price < 0 or abs(price) >= 1e307
                        else price
                    ),
                    "market_data_type": self._market_data_types.get(reqId),
                    "receive_timestamp_utc": datetime.now(UTC).isoformat(),
                    "attributes": {
                        "can_auto_execute": getattr(attrib, "canAutoExecute", None),
                        "past_limit": getattr(attrib, "pastLimit", None),
                        "pre_open": getattr(attrib, "preOpen", None),
                    },
                },
            )

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            numeric = float(size) if size is not None else None
            if numeric is not None and (
                not math.isfinite(numeric) or numeric < 0 or abs(numeric) >= 1e307
            ):
                numeric = None
            adapter.on_quote_update(
                reqId,
                {
                    "field": {
                        0: "bid_size",
                        3: "ask_size",
                        5: "last_size",
                        8: "volume",
                        27: "call_open_interest",
                        28: "put_open_interest",
                    }.get(tickType, f"size_tick_{tickType}"),
                    "value": numeric,
                    "market_data_type": self._market_data_types.get(reqId),
                    "receive_timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )

        def tickOptionComputation(  # noqa: N802
            self,
            reqId: int,
            tickType: int,
            tickAttrib: int,
            impliedVol: float,
            delta: float,
            optPrice: float,
            pvDividend: float,
            gamma: float,
            vega: float,
            theta: float,
            undPrice: float,
        ) -> None:
            source = {
                10: "bid",
                11: "ask",
                12: "last",
                13: "model",
                80: "bid",
                81: "ask",
                82: "last",
                83: "model",
            }.get(tickType, f"tick_{tickType}")

            def present(value: float) -> float | None:
                return (
                    None
                    if value is None or not math.isfinite(value) or abs(value) >= 1e307
                    else value
                )

            adapter.on_quote_update(
                reqId,
                {
                    "field": "option_computation",
                    "computation_source": source,
                    "implied_volatility": present(impliedVol),
                    "delta": present(delta),
                    "option_price": present(optPrice),
                    "present_value_dividend": present(pvDividend),
                    "gamma": present(gamma),
                    "vega": present(vega),
                    "theta": present(theta),
                    "underlying_reference_price": present(undPrice),
                    "market_data_type": self._market_data_types.get(reqId),
                    "receive_timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )

        def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
            adapter.callbacks.complete(reqId)

        def realtimeBar(  # noqa: N802
            self,
            reqId: int,
            time: int,
            open_: float,
            high: float,
            low: float,
            close: float,
            volume: Any,
            wap: Any,
            count: int,
        ) -> None:
            def numeric(value: Any) -> float | None:
                if value is None:
                    return None
                converted = float(value)
                return (
                    None if not math.isfinite(converted) or abs(converted) >= 1e307 else converted
                )

            adapter.on_realtime_bar(
                RealtimeBarUpdate(
                    request_id=reqId,
                    source_timestamp_utc=datetime.fromtimestamp(time, tz=UTC),
                    receive_timestamp_utc=datetime.now(UTC),
                    open=numeric(open_),
                    high=numeric(high),
                    low=numeric(low),
                    close=numeric(close),
                    volume=numeric(volume),
                    wap=numeric(wap),
                    trade_count=None if count < 0 else count,
                )
            )

        def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
            adapter.on_historical_bar(
                reqId,
                self._historical_bar_payload(bar),
                update=False,
            )

        def historicalDataUpdate(self, reqId: int, bar: Any) -> None:  # noqa: N802
            adapter.on_historical_bar(
                reqId,
                self._historical_bar_payload(bar),
                update=True,
            )

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
            adapter.on_historical_bar_end(reqId, start=start, end=end)

        @staticmethod
        def _historical_bar_payload(bar: Any) -> dict[str, Any]:
            raw_timestamp = str(getattr(bar, "date", ""))
            try:
                bar_start = datetime.fromtimestamp(int(raw_timestamp), tz=UTC)
            except ValueError:
                bar_start = None
            return {
                "provider_bar_timestamp": raw_timestamp,
                "bar_start_utc": (None if bar_start is None else bar_start.isoformat()),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "wap": float(bar.wap),
                "trade_count": int(bar.barCount),
                "source": "ibkr_historical_keep_up_to_date",
            }

    class _MarketDataClientFacade:
        """Expose only the official socket and market-data calls Stocker uses."""

        __slots__ = ("__client",)

        def __init__(self, client: Any) -> None:
            self.__client = client

        def connect(self, host: str, port: int, client_id: int) -> Any:
            return self.__client.connect(host, port, client_id)

        def disconnect(self) -> None:
            self.__client.disconnect()

        def run(self) -> None:
            self.__client.run()

        def reqMktData(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqMktData(*arguments)

        def cancelMktData(self, request_id: int) -> None:  # noqa: N802
            self.__client.cancelMktData(request_id)

        def reqSecDefOptParams(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqSecDefOptParams(*arguments)

        def reqContractDetails(self, request_id: int, contract: Any) -> None:  # noqa: N802
            self.__client.reqContractDetails(request_id, contract)

        def reqRealTimeBars(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqRealTimeBars(*arguments)

        def cancelRealTimeBars(self, request_id: int) -> None:  # noqa: N802
            self.__client.cancelRealTimeBars(request_id)

        def reqTickByTickData(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqTickByTickData(*arguments)

        def cancelTickByTickData(self, request_id: int) -> None:  # noqa: N802
            self.__client.cancelTickByTickData(request_id)

        def reqMktDepth(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqMktDepth(*arguments)

        def cancelMktDepth(  # noqa: N802
            self, request_id: int, smart_depth: bool
        ) -> None:
            self.__client.cancelMktDepth(request_id, smart_depth)

        def reqMktDepthExchanges(self) -> None:  # noqa: N802
            self.__client.reqMktDepthExchanges()

        def reqHistoricalData(self, *arguments: Any) -> None:  # noqa: N802
            self.__client.reqHistoricalData(*arguments)

        def cancelHistoricalData(self, request_id: int) -> None:  # noqa: N802
            self.__client.cancelHistoricalData(request_id)

        def reqCurrentTime(self) -> None:  # noqa: N802
            self.__client.reqCurrentTime()

        def reqMarketDataType(self, market_data_type: int) -> None:  # noqa: N802
            self.__client.reqMarketDataType(market_data_type)

        def serverVersion(self) -> int:  # noqa: N802
            return int(self.__client.serverVersion())

    return _MarketDataClientFacade(_StockerOfficialMarketDataClient())
