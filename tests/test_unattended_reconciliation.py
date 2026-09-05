"""Reproduce an unavailable IBKR completed-order endpoint at the real adapter seam."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_core.runs import Environment
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import BrokerFill, BrokerOrderIds, OrderAction
from stocker_execution.ibkr import IbkrConnection
from stocker_execution.stage7 import Stage7ExecutionService
from test_stage7_execution import _instrument, _intent, _run
from test_stage7_ibkr import FakeOrderClient, _config, _plan


class UnavailableHistoryClient(FakeOrderClient):
    def __init__(self, *, account: str = "DU123456") -> None:
        super().__init__(account=account)
        self.live_trades: list[object] = []

    async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]:
        self.completed_order_requests += 1
        await asyncio.Event().wait()
        return []

    def trades(self) -> list[object]:
        return [*super().trades(), *self.live_trades]


def service_for(
    path: Path, client: UnavailableHistoryClient, environment: Environment = Environment.PAPER
) -> tuple[IbkrConnection, Stage7ExecutionService]:
    config = _config(environment).model_copy(update={"request_timeout_seconds": 0.01})
    connection = IbkrConnection(config, client=client, execution_enabled=True)
    service = Stage7ExecutionService(
        run=_run(environment),
        expected_account=client.accounts[0],
        broker=connection,
        ledger=ExecutionLedger(path),
        clock=lambda: datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
    )
    return connection, service


@pytest.mark.parametrize("environment", (Environment.PAPER, Environment.LIVE))
def test_closed_trade_reconnect_does_not_require_completed_history(
    tmp_path: Path, environment: Environment
) -> None:
    account = "DU123456" if environment is Environment.PAPER else "U123456"
    path = tmp_path / "ledger.sqlite3"
    ledger = ExecutionLedger(path)
    plan = _plan(environment)
    assert ledger.reserve(plan, expected_account=account)
    ledger.record_submission(
        plan.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account=account
    )
    for execution_id, order_id, side, price in (
        ("entry", 101, OrderAction.SELL, 100.0),
        ("exit", 103, OrderAction.BUY, 98.0),
    ):
        ledger.record_fill(
            BrokerFill(
                execution_id,
                order_id,
                account,
                environment,
                265598,
                "AAPL",
                side,
                10,
                price,
                datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
                0.0,
            )
        )
    assert not ledger.active_records(environment, account)
    client = UnavailableHistoryClient(account=account)
    connection, service = service_for(path, client, environment)

    async def scenario() -> None:
        for _ in range(2):
            await connection.connect()
            result = await service.reconcile()
            assert result.ok, result.detail
            connection.disconnect()

    asyncio.run(scenario())
    assert client.open_order_requests == 2
    assert client.completed_order_requests == 0


def test_unfinished_sibling_plan_still_blocks_when_history_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = ExecutionLedger(path)
    # The plan belongs to run-1, while reconciliation is led by paper-run.
    assert ledger.reserve(_plan(), expected_account="DU123456")
    ledger.mark_submitting(_plan().order_plan_id)
    client = UnavailableHistoryClient()
    connection, service = service_for(path, client)

    async def scenario() -> None:
        await connection.connect()
        result = await service.reconcile()
        assert not result.ok
        assert "order-status request timed out" in result.detail

    asyncio.run(scenario())
    assert client.completed_order_requests == 1


def test_orders_placed_after_flat_recovery_use_live_status_until_disconnect(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    client = UnavailableHistoryClient()
    connection, service = service_for(path, client)

    async def scenario() -> None:
        await connection.connect()
        assert (await service.reconcile()).ok
        submitted = await service.execute(_intent(), _instrument())
        assert submitted.order_ids is not None
        assert submitted.order_plan is not None
        # A normal live rejection callback must still be persisted, without history.
        client.live_trades = [
            SimpleNamespace(
                order=SimpleNamespace(
                    orderId=submitted.order_ids.parent,
                    orderRef=submitted.order_plan.order_plan_id,
                    account="DU123456",
                ),
                orderStatus=SimpleNamespace(status="Inactive", filled=0, remaining=100),
                log=[SimpleNamespace(message="broker rejected order")],
            )
        ]
        result = await service.reconcile()
        assert result.ok, result.detail
        assert not ExecutionLedger(path).active_records(Environment.PAPER, "DU123456")

    asyncio.run(scenario())
    assert client.completed_order_requests == 0


def test_flat_local_ledger_does_not_hide_unknown_broker_position(tmp_path: Path) -> None:
    client = UnavailableHistoryClient()
    client.position_values = [
        SimpleNamespace(
            account="DU123456",
            contract=SimpleNamespace(conId=265598, symbol="AAPL"),
            position=-10,
            avgCost=100.0,
        )
    ]
    connection, service = service_for(tmp_path / "ledger.sqlite3", client)

    async def scenario() -> None:
        await connection.connect()
        result = await service.reconcile()
        assert not result.ok
        assert "unexpected broker position" in result.detail

    asyncio.run(scenario())
    assert client.completed_order_requests == 0


def test_disconnect_with_new_unfinished_order_requires_history_again(tmp_path: Path) -> None:
    client = UnavailableHistoryClient()
    connection, service = service_for(tmp_path / "ledger.sqlite3", client)

    async def scenario() -> None:
        await connection.connect()
        assert (await service.reconcile()).ok
        submitted = await service.execute(_intent(), _instrument())
        assert submitted.order_ids is not None
        connection.disconnect()
        await connection.connect()
        result = await service.reconcile()
        assert not result.ok
        assert "order-status request timed out" in result.detail

    asyncio.run(scenario())
    assert client.completed_order_requests == 1
