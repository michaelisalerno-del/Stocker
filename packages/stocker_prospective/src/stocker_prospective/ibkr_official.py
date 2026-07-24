"""Official ``ibapi`` callback bridge, imported only on an IBKR server."""

from __future__ import annotations

import math
from datetime import UTC, datetime
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
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ) -> None:
            adapter.on_error(reqId, errorCode, errorString)

        def connectionClosed(self) -> None:  # noqa: N802
            adapter.on_connection_closed()

        def marketDataType(self, reqId: int, marketDataType: int) -> None:  # noqa: N802
            mapped = {
                1: MarketDataType.LIVE.value,
                2: MarketDataType.FROZEN.value,
                3: MarketDataType.DELAYED.value,
                4: MarketDataType.DELAYED_FROZEN.value,
            }.get(marketDataType, f"unknown_{marketDataType}")
            self._market_data_types[reqId] = mapped
            adapter.on_quote_update(
                reqId,
                {
                    "field": "market_data_type",
                    "value": mapped,
                    "receive_timestamp_utc": datetime.now(UTC).isoformat(),
                },
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

    return _MarketDataClientFacade(_StockerOfficialMarketDataClient())
