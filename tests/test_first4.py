import asyncio
import hashlib
import importlib.util
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pandas as pd
import pytest
from ib_async import Contract, LimitOrder, Order, OrderStatus, Trade
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_dashboard.app import create_dashboard_app
from stocker_execution.first4 import Q5, Bar, eligible, prior15
from stocker_execution.first4_broker import PaperBroker, listed_strike, size
from stocker_execution.first4_config import PAPER_ACCOUNT, First4Config
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store

FIXTURE = Path(__file__).parent / "fixtures/first4"
OPEN = datetime(2025, 7, 21, 13, 30, tzinfo=UTC)
CLOSE = OPEN + timedelta(minutes=390)


def candidate(symbol, rank, prior=5):
    return dict(symbol=symbol, con_id=rank, rank=rank, price=10, change_pct=5.5, prior15=prior)


def test_frozen_all_1293_appearances_and_80_first4_selections(tmp_path):
    spec = importlib.util.spec_from_file_location("frozen", FIXTURE / "original_functions.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    manifest = json.loads((FIXTURE / "source_manifest.json").read_text())
    for name, digest in manifest["fixture_hashes"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    events = pd.read_parquet(FIXTURE / "first_appearances.parquet").reset_index(drop=True)
    windows = np.load(FIXTURE / "prior_windows.npy")
    store = Store(tmp_path / "state.sqlite")
    records = []
    for i, r in enumerate(events.itertuples()):
        clock = datetime.fromisoformat(r.scanner_time)
        opened = clock - timedelta(minutes=int(r.scanner_j) + 1)
        array = windows[i]
        start = clock - timedelta(minutes=16)
        bars = [
            Bar(start + timedelta(minutes=k), *row)
            for k, row in enumerate(array)
            if start + timedelta(minutes=k) >= opened
        ]
        got = prior15(bars, opened, clock)
        if r.scanner_j >= 14:
            ref = array[-15, 0] if r.scanner_j == 14 else array[0, 3]
            _, original = source.excursion(array[-15:], ref)
            assert np.isclose(
                got if got is not None else np.nan, original, equal_nan=True, rtol=0, atol=1e-12
            )
        assert np.isclose(
            got if got is not None else np.nan,
            r.prior15_range_pct,
            equal_nan=True,
            rtol=0,
            atol=1e-12,
        )
        records.append(
            dict(
                session=r.date,
                clock=clock,
                close=opened + timedelta(minutes=r.session_minutes),
                symbol=r.symbol,
                con_id=int(r.event_id) + 1,
                rank=r.scanner_rank,
                price=r.scanner_anchor,
                change_pct=r.change_pct,
                prior15=got,
            )
        )
    selected = []
    for (day, clock), g in pd.DataFrame(records).groupby(["session", "clock"], sort=True):
        selected.extend(
            store.observe(
                day, clock.to_pydatetime(), g.close.iloc[0].to_pydatetime(), g.to_dict("records")
            )
        )
    expected = pd.read_parquet(FIXTURE / "policy_decisions.parquet")
    expected = expected[(expected.Policy == "FIRST4") & (expected.Status == "ACCEPT")]
    assert len(events) == 1293 and len(selected) == len(expected) == 80
    assert {(e["session"], e["symbol"], e["slot"]) for e in selected} == {
        (r.date, r.symbol, int(r.Trade)) for r in expected.itertuples()
    }
    for e in selected:
        r = expected[(expected.date == e["session"]) & (expected.symbol == e["symbol"])].iloc[0]
        assert datetime.fromisoformat(e["entry_at"]) == datetime.fromisoformat(
            r.scanner_time
        ) + timedelta(minutes=1)
        assert datetime.fromisoformat(e["expiry_at"]) - datetime.fromisoformat(
            e["entry_at"]
        ) == timedelta(minutes=2880)
    for _day, g in expected.groupby("date"):
        ordered = g.sort_values(["scanner_time", "scanner_rank"])
        original = source.select(
            [(r.event_id, r.scanner_clock) for r in ordered.itertuples()], 0, 4
        )
        assert all(status == "ACCEPT" for _, status, _ in original)


def test_boundaries_missing_and_completed_bars():
    assert eligible(20, 5.5, 6) != "Q5"
    assert eligible(19.999, 5.5, Q5) != "Q5"
    assert eligible(19.999, 5.5, np.nextafter(Q5, np.inf)) == "Q5"
    assert eligible(10, np.nextafter(5.5, -np.inf), 6) != "Q5"
    assert eligible(10, 6, None) == "PRIOR15_UNAVAILABLE"
    bars = [Bar(OPEN + timedelta(minutes=i), 10, 11, 9, 10) for i in range(16)]
    assert prior15(bars, OPEN, OPEN + timedelta(minutes=15)) == pytest.approx(20)
    assert prior15(bars, OPEN, OPEN + timedelta(minutes=14)) is None
    assert prior15(bars[:-2], OPEN, OPEN + timedelta(minutes=15)) is None
    assert prior15(bars + [bars[0]], OPEN, OPEN + timedelta(minutes=15)) is None
    assert prior15(bars, OPEN, OPEN + timedelta(minutes=15, seconds=1)) is None
    assert prior15(bars, OPEN, OPEN + timedelta(minutes=15)) == prior15(
        bars[:-1], OPEN, OPEN + timedelta(minutes=15)
    )


def test_slots_duplicates_failures_restart_and_response_order(tmp_path):
    path = tmp_path / "state.sqlite"
    store = Store(path)
    clock = OPEN + timedelta(minutes=15)
    selected = store.observe(
        "2025-07-21",
        clock,
        CLOSE,
        [candidate("B", 2), candidate("A", 1), candidate("REJECT", 3, None)],
    )
    assert [e["symbol"] for e in selected] == ["A", "B"]
    store.outcome(selected[0], "BROKER_REJECTED")
    store.db.close()
    store = Store(path)
    selected = store.observe(
        "2025-07-21",
        clock + timedelta(minutes=1),
        CLOSE,
        [
            candidate("REJECT", 1),
            candidate("A", 2),
            candidate("D", 4),
            candidate("C", 3),
            candidate("E", 5),
        ],
    )
    assert [(e["symbol"], e["slot"]) for e in selected] == [("C", 3), ("D", 4)]
    assert next(e for e in store.rows("events") if e["symbol"] == "E")["decision"] == "DAILY_CAP"
    assert (
        next(e for e in store.rows("events") if e["symbol"] == "REJECT")["decision"]
        == "PRIOR15_UNAVAILABLE"
    )
    with pytest.raises(ValueError, match="OUT_OF_ORDER"):
        store.observe("2025-07-21", clock, CLOSE, [])


def armed_config():
    return First4Config(
        armed=True,
        expiry_rule="EXACT_CALENDAR_DATE",
        strike_rule="OUTWARD",
        premium_budget_usd=100,
        fee_reserve_per_package_usd=2,
        entry_limit="SUM_OF_ASKS",
        quote_max_age_seconds=2,
        entry_deadline_seconds=10,
        exit_seconds_before_close=30,
        exit_order="MARKET",
    )


class Event:
    def __iadd__(self, callback):
        return self

    def __isub__(self, callback):
        return self


def fake_ib():
    ib = NS(
        **{
            n: Event()
            for n in [
                "disconnectedEvent",
                "execDetailsEvent",
                "commissionReportEvent",
                "orderStatusEvent",
                "errorEvent",
                "positionEvent",
            ]
        }
    )
    ib.isConnected = Mock(return_value=True)
    ib.managedAccounts = Mock(return_value=[PAPER_ACCOUNT])
    ib.placeOrder = Mock()
    ib.client = NS(getReqId=Mock(return_value=101))
    ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[])
    ib.reqCompletedOrdersAsync = AsyncMock(return_value=[])
    ib.reqExecutionsAsync = AsyncMock(return_value=[])
    ib.reqPositionsAsync = AsyncMock(return_value=[])
    ib.positions = Mock(return_value=[])
    ib.openTrades = Mock(return_value=[])
    return ib


def broker(tmp_path):
    b = PaperBroker(armed_config(), Store(tmp_path / "s.sqlite"), fake_ib())
    b.reconciled = True
    b.entry_blocker = ""
    return b


def test_paper_boundary_reconnect_and_required_settings(tmp_path):
    with pytest.raises(ValidationError):
        First4Config(environment="LIVE")
    with pytest.raises(ValidationError):
        First4Config(expected_account="U12345")
    with pytest.raises(ValidationError):
        First4Config(strategy="SESSION_HARD")
    with pytest.raises(ValueError, match="Missing"):
        First4Config(armed=True).require_execution()
    b = broker(tmp_path)
    b.ib.managedAccounts.return_value = ["U12345"]
    with pytest.raises(ValueError, match="PAPER"):
        b.guard()
    assert not b.reconciled
    b.ib.managedAccounts.return_value = [PAPER_ACCOUNT]
    with pytest.raises(ValueError, match="reconciliation"):
        b.guard()
    asyncio.run(b.reconcile())
    b.guard()
    assert b.ib.placeOrder.call_count == 0


def test_pause_persists_and_is_enforced_after_async_preparation(tmp_path, monkeypatch):
    b = broker(tmp_path)
    entry = datetime.now(UTC)
    event = dict(
        session="2026-09-23",
        symbol="A",
        slot=1,
        entry_at=entry.isoformat(),
        close_at=(entry + timedelta(hours=1)).isoformat(),
    )
    b.store.set_meta("paused", True)
    with pytest.raises(ValueError, match="PAUSED"):
        b.submit(event, Contract(), LimitOrder("BUY", 1, 1), {}, "ENTRY")
    assert not b.ib.placeOrder.called and b.store.rows("orders") == []
    runtime = Runtime(armed_config(), b.store)
    assert runtime.pause
    b.store.set_meta("paused", False)
    event["close_at"] = entry.isoformat()
    with pytest.raises(ValueError, match="HOLDING_WINDOW"):
        b.submit(event, Contract(), LimitOrder("BUY", 1, 1), {}, "ENTRY")


def test_sizing_uses_actual_multiplier_fee_and_increment():
    assert size(100, 1, 100, 1, 1) == 0
    assert size(203, 1, 100, 2, 1) == 1
    assert size(600, 1, 150, 0, 2) == 4
    assert size(599.99, 1, 150, 0, 2) == 2
    assert listed_strike([9, 10, 11], 10.5, "C", "NEAREST_TIES_OUTWARD") == 11
    assert listed_strike([9, 10, 11], 9.5, "P", "NEAREST_TIES_OUTWARD") == 9


def test_leg_fills_idempotent_and_combo_status_not_fills(tmp_path):
    b = broker(tmp_path)
    event = dict(session="2025-07-21", symbol="A", slot=1)
    ref = b.store.reserve_order(event, "ENTRY", 1, {})
    b.order_status(
        NS(order=NS(orderRef=ref, orderId=1), orderStatus=NS(status="Filled", permId=22))
    )
    assert b.owned_quantities() == {}
    execution = NS(
        acctNumber=PAPER_ACCOUNT,
        orderRef=ref,
        execId="LEG1",
        shares=1,
        price=2,
        side="BOT",
        time=OPEN,
    )
    f = NS(execution=execution, contract=NS(secType="OPT", conId=100, multiplier="100"))
    b.fill(None, f)
    b.fill(None, f)
    assert b.owned_quantities() == {100: 1}
    execution.execId = "BAG1"
    f.contract.secType = "BAG"
    b.fill(None, f)
    assert len(b.store.rows("fills")) == 1
    with pytest.raises(ValueError, match="POSITION_MISMATCH"):
        asyncio.run(b.reconcile())


def test_unknown_positions_block_entries_without_touching_orders(tmp_path):
    b = broker(tmp_path)
    b.ib.reqPositionsAsync.return_value = [
        NS(account=PAPER_ACCOUNT, position=1, contract=NS(conId=999))
    ]
    asyncio.run(b.reconcile())
    assert b.entry_blocker and b.reconciled and not b.ib.placeOrder.called


def test_completed_order_decoder_shape_recovers_by_permanent_identity(tmp_path):
    b = broker(tmp_path)
    ref = b.store.reserve_order(dict(session="2025-07-21", symbol="A", slot=1), "ENTRY", 101, {})
    completed = Trade(
        order=Order(account=PAPER_ACCOUNT, orderRef=ref, permId=9001),
        orderStatus=OrderStatus(status="Cancelled"),
    )
    assert completed.order.clientId == completed.order.orderId == completed.orderStatus.permId == 0
    b.ib.reqCompletedOrdersAsync.return_value = [completed]
    asyncio.run(b.reconcile())
    row = b.store.rows("orders")[0]
    assert (row["order_id"], row["perm_id"], row["status"]) == (101, 9001, "Cancelled")
    completed.order.permId = 9002
    with pytest.raises(ValueError, match="Completed order identity"):
        asyncio.run(b.reconcile())


def test_entry_uses_bag_market_tick_and_actual_leg_quantity_rules(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    baseline = datetime.fromisoformat(event["entry_at"])
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: baseline)
    legs = [
        NS(contract=Contract(secType="OPT", conId=i, multiplier="100"), sizeIncrement=1, minSize=1)
        for i in (101, 102)
    ]
    combo = Contract(secType="BAG", symbol="A", currency="USD", exchange="SMART")
    b.contracts = AsyncMock(return_value=(*legs, combo))
    b.quotes = AsyncMock(
        return_value=[
            dict(bid=0.2, ask=0.3, bid_at=baseline.isoformat(), ask_at=baseline.isoformat())
        ]
        * 2
    )
    b.ib.reqMktData = Mock(return_value=NS(minTick=0.05))
    b.ib.cancelMktData = Mock()
    b.ib.reqContractDetailsAsync = AsyncMock(
        side_effect=AssertionError("No BAG contract-details request")
    )
    b.ib.placeOrder.side_effect = lambda c, o: Trade(
        contract=c, order=o, orderStatus=OrderStatus(status="Submitted", permId=22)
    )
    asyncio.run(b.enter(event, Contract(conId=1), 10))
    contract, order = b.ib.placeOrder.call_args.args
    assert contract is combo and order.totalQuantity == 1 and order.lmtPrice == 0.6
    assert order.tif == "GTD" and order.account == PAPER_ACCOUNT
    b.ib.reqMktData.assert_called_once_with(combo, "", False, False)
    b.ib.cancelMktData.assert_called_once_with(combo)
    assert not b.ib.reqContractDetailsAsync.called


def test_submission_routes_real_combo_identity_and_persists_before_send(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event = dict(
        session="2025-07-21",
        symbol="A",
        slot=1,
        entry_at=OPEN.isoformat(),
        close_at=CLOSE.isoformat(),
    )
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    combo = Contract(secType="BAG", symbol="A", currency="USD", exchange="SMART")

    def sent(contract, order):
        assert contract is combo and order.account == PAPER_ACCOUNT
        assert b.store.rows("orders")[0]["status"] == "RESERVED"
        return NS(order=order, orderStatus=NS(status="Submitted", permId=72))

    b.ib.placeOrder.side_effect = sent
    b.submit(event, combo, LimitOrder("BUY", 1, 0.7), {}, "ENTRY")
    assert b.store.rows("orders")[0]["status"] == "Submitted"
    with pytest.raises(sqlite3.IntegrityError):
        b.submit(event, combo, LimitOrder("BUY", 1, 0.7), {}, "ENTRY")
    assert b.ib.placeOrder.call_count == 1


def prepare_exit(b, quantities=(1, 1)):
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    legs = [
        Contract(
            secType="OPT", conId=i, symbol="A", exchange="SMART", currency="USD", multiplier="100"
        )
        for i in (101, 102)
    ]
    payload = {
        "put": legs[0].dict(),
        "call": legs[1].dict(),
        "exit_at": (CLOSE - timedelta(seconds=30)).isoformat(),
    }
    ref = b.store.reserve_order(event, "ENTRY", 1, payload)
    with b.store.db:
        b.store.db.execute("UPDATE first4_orders SET status='Cancelled' WHERE reference=?", (ref,))
    for leg, quantity in zip(legs, quantities, strict=True):
        if quantity:
            b.fill(
                None,
                NS(
                    execution=NS(
                        acctNumber=PAPER_ACCOUNT,
                        orderRef=ref,
                        execId=str(leg.conId),
                        shares=quantity,
                        price=1,
                        side="BOT",
                        time=OPEN,
                    ),
                    contract=leg,
                ),
            )
    b.ib.positions.return_value = [
        NS(contract=x, position=q) for x, q in zip(legs, quantities, strict=True)
    ]
    b.ib.client.getReqId.side_effect = range(200, 210)
    b.ib.placeOrder.side_effect = lambda c, o: NS(
        order=o, orderStatus=NS(status="Submitted", permId=o.orderId)
    )
    return event, payload


def test_session_close_submits_owned_combo_not_account_cancellation(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    contract, order = b.ib.placeOrder.call_args.args
    assert order.action == "SELL" and order.totalQuantity == 1 and order.account == PAPER_ACCOUNT
    assert [x.conId for x in contract.comboLegs] == [101, 102]
    asyncio.run(b.close_due())
    assert b.ib.placeOrder.call_count == 1


def test_partial_leg_exit_resumes_only_unsent_leg(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b, (2, 1))
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    submit = b.submit
    failed = False

    def interrupted(event, contract, order, payload, role, suffix=""):
        nonlocal failed
        if suffix == "102" and not failed:
            failed = True
            raise RuntimeError("Disconnected before reservation")
        return submit(event, contract, order, payload, role, suffix)

    b.submit = interrupted
    with pytest.raises(RuntimeError):
        asyncio.run(b.close_due())
    asyncio.run(b.close_due())
    assert [call.args[0].conId for call in b.ib.placeOrder.call_args_list] == [101, 102]
    assert [call.args[1].totalQuantity for call in b.ib.placeOrder.call_args_list] == [2, 1]


def test_old_unfilled_allocation_cannot_close_newer_position(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    old_event = b.store.observe(
        "2025-07-18",
        OPEN - timedelta(days=3) + timedelta(minutes=15),
        CLOSE - timedelta(days=3),
        [candidate("A", 1)],
    )[0]
    old_payload = {**payload, "exit_at": (CLOSE - timedelta(days=3)).isoformat()}
    ref = b.store.reserve_order(old_event, "ENTRY", 3, old_payload)
    with b.store.db:
        b.store.db.execute("UPDATE first4_orders SET status='Cancelled' WHERE reference=?", (ref,))
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN + timedelta(hours=1))
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert (
        next(e for e in b.store.rows("events") if e["session"] == "2025-07-18")["outcome"]
        == "CLOSED"
    )


def test_missing_option_quote_is_visible_and_subscriptions_cancelled(tmp_path):
    b = broker(tmp_path)
    b.ib.reqMktData = Mock(return_value=NS(updateEvent=Event(), marketDataType=3))
    b.ib.cancelMktData = Mock()
    with pytest.raises(ValueError, match="QUOTES"):
        asyncio.run(
            b.quotes(
                [Contract(conId=1), Contract(conId=2)],
                datetime.now(UTC) + timedelta(milliseconds=1),
            )
        )
    assert b.ib.cancelMktData.call_count == 2 and not b.ib.placeOrder.called


def test_old_cli_and_api_routes_removed_and_dashboard_health(tmp_path):
    for command in ["stage10-run", "stage9-run", "stage8-paper-run", "start"]:
        result = CliRunner().invoke(app, [command, "--help"])
        assert result.exit_code != 0 and "No such command" in result.output
    runtime = Runtime(First4Config(), Store(tmp_path / "s.sqlite"))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            for path in ["/", "/api/system", "/api/overview", "/static/dashboard.js"]:
                assert (await client.get(path)).status_code == 200
            for path in ["/api/runs", "/api/universe-builder/options", "/api/runs/old/enable"]:
                assert (await client.get(path)).status_code == 404
            assert (await client.post("/api/universe-runs/live", json={})).status_code == 405
            assert (await client.post("/api/first4/pause")).status_code == 200
            assert runtime.store.get_meta("paused")

    asyncio.run(check())
