import asyncio
from stocker_execution.requests import BrokerConnection


async def main():
    ib = BrokerConnection()
    pending = ib.wrapper.startReq(123)
    ib.client.decoder.interpret(
        ["75", "123", "CME", "876880607", "BTC", "5", "1", "20261030", "1", "85000"]
    )
    ib.client.decoder.interpret(["76", "123"])
    chains = await pending
    value = chains[0].underlyingConId
    selected = [
        c
        for c in chains
        if c.exchange == "CME"
        and c.tradingClass == "BTC"
        and c.underlyingConId == 876880607
        and float(c.multiplier) == 5
    ]
    print(
        {
            "decoded_underlying_id": value,
            "type": type(value).__name__,
            "matching_chains": len(selected),
        }
    )
    assert len(selected) == 1, (
        "Observed option chain incorrectly rejected by deployed identity comparison"
    )


asyncio.run(main())
