"""Official ``ibapi`` callback bridge, imported only on an IBKR server."""

from __future__ import annotations

from typing import Any

from stocker_prospective.ibkr import IBKRMarketDataAdapter, require_official_ibkr_api
from stocker_prospective.market_data import MarketDataType


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
            adapter.connection.connected(MarketDataType.LIVE)

        def error(
            self,
            reqId: int,
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ) -> None:
            adapter.on_error(reqId, errorCode, errorString)

        def connectionClosed(self) -> None:  # noqa: N802
            adapter.connection.connection_lost(
                code=1100,
                message="official_socket_connection_closed",
            )

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
                {"field": "market_data_type", "value": mapped},
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
                    "value": None if price < 0 else price,
                    "market_data_type": self._market_data_types.get(reqId),
                    "attributes": {
                        "can_auto_execute": getattr(attrib, "canAutoExecute", None),
                        "past_limit": getattr(attrib, "pastLimit", None),
                        "pre_open": getattr(attrib, "preOpen", None),
                    },
                },
            )

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            numeric = float(size) if size is not None else None
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
            }.get(tickType, f"tick_{tickType}")

            def present(value: float) -> float | None:
                return None if value is None or value < -1e300 else value

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
                },
            )

        def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
            adapter.callbacks.complete(reqId)

    return _StockerOfficialMarketDataClient()
