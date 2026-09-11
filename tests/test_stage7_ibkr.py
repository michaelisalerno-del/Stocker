import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.execution_models import (
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument


class FakeOrderClient:
    def __init__(self, *, account: str = "DU123456") -> None:
        self.accounts = [account]
        self.connected = False
        self.connect_kwargs = {}
        self.next_order_id = 100
        self.summary_futures = {}
        self.summary_cancelled = []
        self.wrapper = SimpleNamespace(
            accountSummary=lambda *args: None,
            startReq=lambda request_id: self.summary_futures.setdefault(
                request_id, asyncio.get_running_loop().create_future()
            ),
            _endReq=lambda request_id: self.summary_futures.pop(request_id, None),
        )
        self.client = SimpleNamespace(
            getReqId=self.get_req_id,
            reqAccountSummary=self.request_summary,
            cancelAccountSummary=self.summary_cancelled.append,
        )
        self.placed = []
        self.cancelled = []
        self.account_values = [
            SimpleNamespace(account=account, tag="NetLiquidation", value="100000", currency="BASE"),
            SimpleNamespace(account=account, tag="BuyingPower", value="250000", currency="BASE"),
        ]
        self.contract_details = [SimpleNamespace(minTick=0.05)]
        self.open_orders = []
        self.position_values = []
        self.execution_values = []
        self.completed_orders = []
        self.open_order_requests = 0
        self.completed_order_requests = 0

    def request_summary(self, request_id, group, tags):
        for value in self.account_values:
            self.wrapper.accountSummary(
                request_id, value.account, value.tag, value.value, value.currency
            )
        self.summary_futures[request_id].set_result(None)

    def get_req_id(self) -> int:
        self.next_order_id += 1
        return self.next_order_id

    async def connectAsync(self, host: str, port: int, **kwargs: object) -> None:
        self.connect_kwargs = {"host": host, "port": port, **kwargs}
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return self.accounts

    async def accountSummaryAsync(self, account: str = "") -> list[object]:
        assert account == self.accounts[0]
        return self.account_values

    async def reqContractDetailsAsync(self, contract: object) -> list[object]:
        assert contract.conId == 265598
        return self.contract_details

    def placeOrder(self, contract: object, order: object) -> object:
        self.placed.append((contract, order))
        return SimpleNamespace(contract=contract, order=order)

    async def reqAllOpenOrdersAsync(self) -> list[object]:
        self.open_order_requests += 1
        return self.open_orders

    async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]:
        assert apiOnly is False
        self.completed_order_requests += 1
        return self.completed_orders

    def openTrades(self) -> list[object]:
        return self.open_orders

    def trades(self) -> list[object]:
        return [*self.open_orders, *self.completed_orders]

    async def reqPositionsAsync(self) -> list[object]:
        return self.position_values

    async def reqExecutionsAsync(self, execFilter: object = None) -> list[object]:
        assert execFilter is None
        return self.execution_values

    def cancelOrder(self, order: object, manualCancelOrderTime: str = "") -> object:
        self.cancelled.append(order)
        return SimpleNamespace(order=order)


def _config(environment: Environment = Environment.PAPER) -> IbkrConfig:
    return IbkrConfig(
        environment=environment,
        host="127.0.0.1",
        port=4002 if environment is Environment.PAPER else 4001,
        client_id=21,
        expected_account="DU123456" if environment is Environment.PAPER else "U123456",
    )


def _instrument() -> QualifiedInstrument:
    return QualifiedInstrument("AAPL", 265598, "SMART", "NASDAQ", "USD", "STK")


def _plan(environment: Environment = Environment.PAPER) -> OrderPlan:
    return OrderPlan(
        "plan-1",
        "run-1",
        "signal-1",
        "strategy",
        "v1",
        265598,
        "AAPL",
        OrderAction.SELL,
        10,
        EntryOrderType.MARKET,
        100.0,
        101.0,
        98.0,
        environment,
        datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
    )


def test_execution_enabled_paper_connection_is_not_readonly_and_reads_account_state() -> None:
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.account_state()

    state = asyncio.run(scenario())

    assert client.connect_kwargs["readonly"] is False
    assert connection.environment is Environment.PAPER
    assert connection.account == "DU123456"
    assert state.equity == 100_000.0
    assert state.buying_power == 250_000.0
    assert state.positions == ()


def test_contract_details_supply_the_required_minimum_tick() -> None:
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> float:
        await connection.connect()
        return await connection.minimum_tick(_instrument())

    assert asyncio.run(scenario()) == 0.05


def test_fresh_account_summary_ignores_cache_and_cleans_up():
    async def scenario():
        client = FakeOrderClient()
        connection = IbkrConnection(_config(), client=client, execution_enabled=True)
        await connection.connect()
        original = client.wrapper.accountSummary
        client.account_values = [
            SimpleNamespace(account="DU123456", tag=tag, value=value, currency="USD")
            for tag, value in [("NetLiquidation", "123000"), ("GrossPositionValue", "2000")]
        ]

        async def stale(account):
            raise AssertionError("admission must request a fresh summary")

        client.accountSummaryAsync = stale
        state = await connection.account_state(fresh=True)
        assert state.equity == 123000 and state.gross_position_value == 2000
        assert state.currency == "USD" and state.buying_power is None
        assert len(client.summary_cancelled) == 1 and not client.summary_futures
        assert client.wrapper.accountSummary is original

    asyncio.run(scenario())


def test_fresh_account_summary_cancellation_cleans_up():
    async def scenario():
        client = FakeOrderClient()
        entered = asyncio.Event()
        client.client.reqAccountSummary = lambda *args: entered.set()
        connection = IbkrConnection(_config(), client=client, execution_enabled=True)
        await connection.connect()
        original = client.wrapper.accountSummary
        pending = asyncio.create_task(connection.account_state(fresh=True))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert len(client.summary_cancelled) == 1 and not client.summary_futures
        assert client.wrapper.accountSummary is original

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "required,equity",
    [("1.7976931348623157e308", "1.7976931348623157e308"), ("nan", "100"), ("200", "100")],
)
def test_invalid_or_insufficient_credit_preview_rejects(required, equity):
    async def scenario():
        client = FakeOrderClient()

        async def preview(contract, order):
            assert order.whatIf and not client.placed
            return SimpleNamespace(
                initMarginAfter=required, equityWithLoanAfter=equity, warningText=""
            )

        client.whatIfOrderAsync = preview
        connection = IbkrConnection(_config(), client=client, execution_enabled=True)
        await connection.connect()
        with pytest.raises(IbkrError, match="sufficient capacity"):
            await connection.check_order_capacity(
                replace(_plan(), entry_limit_price=100), _instrument()
            )
        assert not client.placed

    asyncio.run(scenario())


def test_paper_submission_transmits_one_coherent_market_stop_limit_bracket() -> None:
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.submit_protected_order(_plan(), _instrument())

    ids = asyncio.run(scenario())
    orders = [order for _, order in client.placed]

    assert ids.parent == 101
    assert ids.target == 102
    assert ids.stop == 103
    assert [order.orderType for order in orders] == ["MKT", "LMT", "STP"]
    assert [order.action for order in orders] == ["SELL", "BUY", "BUY"]
    assert [order.totalQuantity for order in orders] == [10, 10, 10]
    assert orders[0].transmit is False
    assert orders[0].tif == "DAY"
    assert orders[1].parentId == ids.parent and orders[1].transmit is False
    assert orders[2].parentId == ids.parent and orders[2].transmit is True
    assert orders[1].tif == "GTC"
    assert orders[2].tif == "GTC"
    assert orders[1].lmtPrice == 98.0
    assert orders[2].auxPrice == 101.0
    assert {order.orderRef for order in orders} == {"plan-1"}
    assert {order.account for order in orders} == {"DU123456"}


def test_guarded_entry_uses_gtd_limit_and_keeps_protective_children_gtc():
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)
    expiry = datetime.now(tz=UTC) + timedelta(seconds=5)
    plan = replace(
        _plan(),
        entry_order_type=EntryOrderType.LIMIT,
        entry_limit_price=100.0,
        entry_expires_at=expiry,
    )

    async def scenario():
        await connection.connect()
        await connection.submit_protected_order(plan, _instrument())

    asyncio.run(scenario())
    parent, target, stop = [order for _, order in client.placed]
    assert parent.orderType == "LMT" and parent.lmtPrice == 100.0
    assert parent.tif == "GTD"
    assert parent.goodTillDate == expiry.strftime("%Y%m%d-%H:%M:%S")
    assert target.tif == stop.tif == "GTC"
    assert target.goodTillDate == stop.goodTillDate == ""
    assert target.parentId == stop.parentId == parent.orderId
    assert target.lmtPrice == 98 and stop.auxPrice == 101
    assert [parent.transmit, target.transmit, stop.transmit] == [False, False, True]


def test_expired_guarded_entry_is_not_sent_to_ibkr():
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)
    plan = replace(
        _plan(),
        entry_order_type=EntryOrderType.LIMIT,
        entry_limit_price=100.0,
        entry_expires_at=datetime.now(tz=UTC),
    )

    async def scenario():
        await connection.connect()
        with pytest.raises(IbkrError, match="expired"):
            await connection.submit_protected_order(plan, _instrument())

    asyncio.run(scenario())
    assert client.placed == []


@pytest.mark.parametrize("reported_type", [1, 2])
def test_entry_quote_releases_temporary_stream_on_success_or_cancellation(reported_type):
    class QuoteClient(FakeOrderClient):
        def reqMarketDataType(self, data_type):
            assert data_type == 1

        def reqMktData(self, contract, **kwargs):
            assert kwargs["snapshot"] is False
            return SimpleNamespace(
                time=datetime.now(tz=UTC), bid=100, ask=100.01, marketDataType=reported_type
            )

        def cancelMktData(self, contract):
            self.cancelled.append(contract)

    client = QuoteClient()
    connection = IbkrConnection(_config(), client=client)

    async def scenario():
        await connection.connect()
        if reported_type == 1:
            quote = await connection.entry_quote(_instrument())
            assert quote.bid == 100 and quote.market_data_type == 1
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(connection.entry_quote(_instrument()), timeout=0.01)

    asyncio.run(scenario())
    assert len(client.cancelled) == 1
    assert connection.resource_status().active_market_data_lines == 0


def test_explicitly_enabled_live_plan_uses_a_writable_live_session() -> None:
    client = FakeOrderClient(account="U123456")
    connection = IbkrConnection(_config(Environment.LIVE), client=client, execution_enabled=True)

    async def scenario() -> None:
        await connection.connect()
        assert client.connect_kwargs["readonly"] is False
        await connection.submit_protected_order(_plan(Environment.LIVE), _instrument())

    asyncio.run(scenario())
    assert len(client.placed) == 3
    assert {order.account for _, order in client.placed} == {"U123456"}


def test_ambiguous_nonbase_account_equity_is_unavailable() -> None:
    client = FakeOrderClient()
    client.account_values = [
        SimpleNamespace(account="DU123456", tag="NetLiquidation", value="100000", currency="USD"),
        SimpleNamespace(account="DU123456", tag="NetLiquidation", value="90000", currency="GBP"),
    ]
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.account_state()

    assert asyncio.run(scenario()).equity is None


def test_open_orders_positions_and_fills_are_normalized_without_callback_leakage() -> None:
    client = FakeOrderClient()
    contract = SimpleNamespace(conId=265598, symbol="AAPL")
    target = SimpleNamespace(
        orderId=102,
        orderRef="plan-1",
        account="DU123456",
        action="BUY",
        orderType="LMT",
        parentId=101,
    )
    client.open_orders = [
        SimpleNamespace(
            contract=contract,
            order=target,
            orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=10),
        )
    ]
    client.position_values = [
        SimpleNamespace(
            account="DU123456", contract=contract, position=Decimal("-4"), avgCost=100.25
        )
    ]
    client.execution_values = [
        SimpleNamespace(
            contract=contract,
            execution=SimpleNamespace(
                execId="exec-1",
                orderId=101,
                orderRef="plan-1",
                acctNumber="DU123456",
                side="SLD",
                shares=Decimal("4"),
                price=100.25,
                time=datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
            ),
            commissionReport=SimpleNamespace(execId="exec-1", commission=0.25),
        )
    ]
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> tuple[object, object, object]:
        await connection.connect()
        return (
            await connection.read_open_orders(),
            await connection.read_positions(),
            await connection.read_fills(),
        )

    orders, positions, fills = asyncio.run(scenario())

    assert orders[0].role is OrderRole.TARGET
    assert orders[0].status is OrderLifecycle.SUBMITTED
    assert positions[0].quantity == -4.0
    assert positions[0].average_price == 100.25
    assert fills[0].execution_id == "exec-1"
    assert fills[0].order_plan_id == "plan-1"
    assert fills[0].side is OrderAction.SELL
    assert fills[0].quantity == 4.0
    assert fills[0].commission == 0.25
    client.execution_values[0].commissionReport = SimpleNamespace(execId="", commission=0.0)
    assert asyncio.run(connection.read_fills())[0].commission is None
    client.execution_values[0].commissionReport = SimpleNamespace(execId="exec-1", commission=0.0)
    assert asyncio.run(connection.read_fills())[0].commission == 0.0


def test_broker_rejection_status_is_normalized() -> None:
    client = FakeOrderClient()
    client.completed_orders = [
        SimpleNamespace(
            contract=SimpleNamespace(conId=265598, symbol="AAPL"),
            order=SimpleNamespace(
                orderId=101,
                orderRef="plan-1",
                account="DU123456",
                action="SELL",
                orderType="MKT",
                parentId=0,
            ),
            orderStatus=SimpleNamespace(status="Inactive", filled=0, remaining=10),
            log=[SimpleNamespace(message="price precaution rejected")],
        )
    ]
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.read_order_statuses()

    statuses = asyncio.run(scenario())

    assert statuses[0].status is OrderLifecycle.REJECTED
    assert statuses[0].reason == "price precaution rejected"


def test_completed_order_request_timeout_fails_reconciliation_without_hanging() -> None:
    class HangingCompletedOrderClient(FakeOrderClient):
        async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]:
            assert apiOnly is False
            await asyncio.Event().wait()
            return []

    client = HangingCompletedOrderClient()
    config = _config().model_copy(update={"request_timeout_seconds": 0.01})
    connection = IbkrConnection(config, client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.read_order_statuses()

    with pytest.raises(IbkrError, match="order-status request timed out"):
        asyncio.run(asyncio.wait_for(scenario(), timeout=0.5))


def test_order_snapshots_are_requested_once_per_connection_then_read_from_live_cache() -> None:
    client = FakeOrderClient()
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> None:
        await connection.connect()
        await connection.read_open_orders()
        await connection.read_order_statuses()
        await connection.read_open_orders()
        await connection.read_order_statuses()
        connection.disconnect()
        await connection.connect()
        await connection.read_open_orders()
        await connection.read_order_statuses()

    asyncio.run(scenario())

    assert client.open_order_requests == 2
    assert client.completed_order_requests == 2


@pytest.mark.parametrize(
    ("connection_method", "client_method", "message"),
    (
        ("account_state", "accountSummaryAsync", "account state request timed out"),
        ("read_open_orders", "reqAllOpenOrdersAsync", "open-order request timed out"),
        ("read_fills", "reqExecutionsAsync", "execution request timed out"),
        ("read_positions", "reqPositionsAsync", "position request timed out"),
    ),
)
def test_execution_state_requests_use_the_configured_timeout(
    connection_method: str,
    client_method: str,
    message: str,
) -> None:
    client = FakeOrderClient()

    async def hang(*args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        await asyncio.Event().wait()
        return []

    setattr(client, client_method, hang)
    config = _config().model_copy(update={"request_timeout_seconds": 0.01})
    connection = IbkrConnection(config, client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        method = getattr(connection, connection_method)
        return await method()

    with pytest.raises(IbkrError, match=message):
        asyncio.run(asyncio.wait_for(scenario(), timeout=0.5))


def test_pending_cancel_remains_active_until_ibkr_confirms_cancellation() -> None:
    client = FakeOrderClient()
    client.open_orders = [
        SimpleNamespace(
            contract=SimpleNamespace(conId=265598, symbol="AAPL"),
            order=SimpleNamespace(
                orderId=102,
                orderRef="plan-1",
                account="DU123456",
                action="BUY",
                orderType="LMT",
                parentId=101,
            ),
            orderStatus=SimpleNamespace(status="PendingCancel", filled=0, remaining=10),
        )
    ]
    connection = IbkrConnection(_config(), client=client, execution_enabled=True)

    async def scenario() -> object:
        await connection.connect()
        return await connection.read_open_orders()

    assert asyncio.run(scenario())[0].status is OrderLifecycle.SUBMITTED


def test_minimum_tick_timeout_cancels_pending_request(tmp_path):
    finished = asyncio.Event()

    class Client(FakeOrderClient):
        async def reqContractDetailsAsync(self, contract):
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

    async def scenario():
        connection = IbkrConnection(
            _config().model_copy(update={"request_timeout_seconds": 0.01}), client=Client()
        )
        await connection.connect()
        with pytest.raises(IbkrError, match="minimum tick request timed out"):
            await connection.minimum_tick(_instrument())
        assert finished.is_set()
        connection.disconnect()

    asyncio.run(scenario())
