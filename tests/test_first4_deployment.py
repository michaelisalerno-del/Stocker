"""Keep independently deployed safeguards when installing the supervised runtime."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from ib_async import Contract, Order, OrderStatus, Trade

from stocker_dashboard.app import create_dashboard_app
from stocker_execution.first4_broker import PaperBroker
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store
from test_first4 import CLOSE, broker, fake_ib, prepare_exit


def test_competing_session_block_survives_reconciliation_and_restart(tmp_path):
    b = broker(tmp_path)
    b.error(7, 10197, "No market data during competing live session", Contract(symbol="F"))
    generation = b.data_generation
    assert b.entry_reason() == "MARKET_DATA_COMPETING_SESSION_10197"
    asyncio.run(b.reconcile())
    assert b.market_data_block
    restarted = PaperBroker(b.config, b.store, fake_ib())
    assert restarted.market_data_block["code"] == 10197
    b.error(8, 10197, "Repeated competing session")
    with pytest.raises(ValueError, match="MARKET_DATA_CHANGED"):
        b.confirm_market_data(generation, {})
    b.confirm_market_data(b.data_generation, {"checks": "complete option-access evidence"})
    assert not b.market_data_block and not b.store.get_meta("market_data_block")
    assert b.store.get_meta("market_data_10197")["code"] == 10197
    assert b.opening_verified_session is None


def test_competing_session_reporting_failure_blocks_without_sqlite(tmp_path, caplog):
    b = broker(tmp_path)
    b.store.db.close()
    b.error(7, 10197, "competing session")
    assert b.fatal_error and b.market_data_block and not b.entries_armed()
    assert "could not persist" in caplog.text


@pytest.mark.parametrize(
    "field,value", [("account", "other"), ("clientId", 99), ("orderId", 999), ("permId", 999)]
)
def test_wrong_identity_cannot_update_owned_status(tmp_path, field, value):
    b = broker(tmp_path)
    prepare_exit(b)
    entry = b.store.active_entries()[0]
    order = Order(
        account=b.config.expected_account,
        clientId=81,
        orderId=1,
        permId=11,
        orderRef=entry["reference"],
    )
    setattr(order, field, value)
    b.order_status(Trade(order=order, orderStatus=OrderStatus(status="Filled", permId=11)))
    assert b.store.active_entries()[0]["status"] == "Cancelled"
    assert not b.reconciled and b.problem == "OWNED_ORDER_IDENTITY_MISMATCH"


def test_execution_identity_and_stale_status_are_not_adopted(tmp_path):
    b = broker(tmp_path)
    prepare_exit(b)
    entry = b.store.active_entries()[0]
    b.fill(
        None,
        NS(
            contract=Contract(secType="OPT", conId=101),
            execution=NS(
                acctNumber=b.config.expected_account,
                orderRef=entry["reference"],
                clientId=99,
                orderId=1,
                permId=11,
                execId="foreign",
            ),
        ),
    )
    assert len(b.store.rows("fills")) == 2
    assert not b.reconciled and b.problem == "EXECUTION_IDENTITY_MISMATCH"
    b.order_status(
        Trade(
            order=Order(
                account=b.config.expected_account,
                clientId=81,
                orderId=1,
                orderRef=entry["reference"],
                permId=11,
            ),
            orderStatus=OrderStatus(status="Submitted", permId=11),
        )
    )
    assert b.store.active_entries()[0]["status"] == "Cancelled"
    assert b.problem == "OUT_OF_ORDER_STATUS_RECONCILIATION_REQUIRED"


def test_connection_change_cannot_finish_reconciliation_or_cache_chain(tmp_path):
    b = broker(tmp_path)

    async def positions():
        b.error(-1, 2110, "Upstream lost")
        return []

    b.ib.reqPositionsAsync = positions
    with pytest.raises(ValueError, match="MARKET_DATA_CHANGED"):
        asyncio.run(b.reconcile())
    assert not b.reconciled
    b.upstream_available = True

    async def chain(*args):
        b.disconnected()
        return ["stale"]

    b.ib.reqSecDefOptParamsAsync = chain
    with pytest.raises(ValueError, match="MARKET_DATA_CHANGED"):
        asyncio.run(b.chain(Contract(conId=42, symbol="F")))
    assert not b.chains


def test_working_exit_is_visible_without_duplicate_or_operator_failure(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    ref = b.store.reserve_order(event, "EXIT", 2, payload)
    b.ib.openTrades.return_value = [
        Trade(order=Order(orderRef=ref), orderStatus=OrderStatus(status="Submitted"))
    ]
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    assert b.store.event(event["session"], "A")["outcome"] == "EXIT_WORKING"
    assert not b.operator_exceptions and not b.ib.placeOrder.called


def test_conflicting_order_cannot_race_owned_exit(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b)

    async def quotes(*args, **kwargs):
        b.ib.openTrades.return_value = [
            Trade(
                contract=Contract(secType="OPT", conId=101),
                order=Order(orderRef="foreign"),
                orderStatus=OrderStatus(status="Submitted"),
            )
        ]
        return [
            {
                "bid": 0.8,
                "ask": 1,
                "bid_at": (CLOSE - timedelta(seconds=20)).isoformat(),
                "ask_at": (CLOSE - timedelta(seconds=20)).isoformat(),
            }
        ] * 2

    b.quotes = quotes
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert "CONFLICTING_WORKING_ORDER" in next(iter(b.operator_exceptions.values()))


def test_position_write_and_view_queries_remain_bounded(tmp_path):
    b = broker(tmp_path)
    prepare_exit(b)
    position = NS(
        account=b.config.expected_account, contract=Contract(conId=101), position=1, avgCost=100
    )
    b.position(position)
    changes = b.store.db.total_changes
    b.position(position)
    assert b.store.db.total_changes == changes
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.session = "2025-07-21"
    b.store.rows = Mock(wraps=b.store.rows)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            response = await client.get("/api/opportunities")
            assert response.status_code == 200
            assert "pnl" not in response.json()
            assert not b.store.rows.call_args_list
            assert "payload" not in response.text

    asyncio.run(check())


def test_active_release_schema_migrates_without_clearing_authority(tmp_path):
    store = Store(tmp_path / "deployed.sqlite")
    store.set_meta("opening_check:2026-09-24", {"status": "WAITING_FOR_OPEN"})
    store.set_meta("market_data_block", {"code": 10197})
    store.db.executescript(
        "CREATE INDEX first4_unresolved ON first4_orders(reference) WHERE role='ENTRY' "
        "AND coalesce(json_extract(payload,'$.management_resolved'),0)=0;"
    )
    store.db.close()
    migrated = Store(tmp_path / "deployed.sqlite")
    assert migrated.get_meta("opening_check:2026-09-24")["status"] == "WAITING_FOR_OPEN"
    assert migrated.get_meta("market_data_block")["code"] == 10197
