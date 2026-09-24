"""The approved option-volume universe restriction reaches the broker wire."""

import asyncio
from datetime import timedelta
from unittest.mock import Mock

import pytest
from ib_async import RequestError, TagValue

from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_runtime import Runtime
from test_first4 import CLOSE, OPEN, broker


@pytest.mark.parametrize("request_fails", [False, True])
def test_option_volume_filter_is_sent_and_never_removed_on_failure(tmp_path, request_fails):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    ib = First4IB()
    ib.RaiseRequestErrors = True  # PaperBroker enables this on its execution connection.
    ib.client.getReqId = Mock(return_value=42)
    b.ib.scanner = ib.scanner

    def send(message, *fields):
        if message == 22:
            if request_fails:
                ib.wrapper.error(42, 162, "Synthetic scanner request rejection", "")
            else:
                ib.wrapper.scannerDataEnd(42)

    ib.client.send = Mock(side_effect=send)

    async def check():
        scan = runtime.scan(
            "2025-07-21", OPEN, CLOSE, CLOSE - timedelta(days=3), OPEN + timedelta(minutes=15)
        )
        if request_fails:
            with pytest.raises(RequestError, match="Synthetic scanner request rejection"):
                await scan
            assert runtime.last_observation is None
        else:
            await scan
            assert runtime.last_observation == (OPEN + timedelta(minutes=15)).isoformat()
        assert not runtime.tasks

    asyncio.run(check())
    requests = [call.args for call in ib.client.send.call_args_list if call.args[0] == 22]
    assert len(requests) == 1
    wire = requests[0]
    assert wire[:6] == (22, 42, 25, "STK", "STK.US", "MOST_ACTIVE")
    # Pinned ib_async 2.1's scanner protocol serializes this native field at 20.
    assert wire[20] == 1
    assert wire[23] == [TagValue("changePercAbove", "5.5"), TagValue("priceBelow", "20")]
    assert not ib.wrapper._futures and not ib.wrapper._results
    assert not ib.wrapper.reqId2Subscriber
    b.ib.placeOrder.assert_not_called()
    assert not b.store.rows("events") and not b.store.rows("orders")
