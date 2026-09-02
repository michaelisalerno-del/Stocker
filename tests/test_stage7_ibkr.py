import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.execution_models import (
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)
from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument


class FakeOrderClient:
    def __init__(self, *, account: str = "DU123456") -> None:
        self.accounts = [account]
        self.connected = False
        self.connect_kwargs = {}
        self.next_order_id = 100
        self.client = SimpleNamespace(getReqId=self.get_req_id)
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
        return self.open_orders

    async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]:
        assert apiOnly is False
        return self.completed_orders

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
            commissionReport=SimpleNamespace(commission=0.25),
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
