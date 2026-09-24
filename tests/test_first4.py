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
from stocker_execution.first4_broker import PaperBroker, listed_expiry, listed_strike
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
        expiry_rule="NEAREST_WITHIN_24H_LATER_TIE",
        strike_rule="NEAREST_STRICT_OTM_WITHIN_1PCT",
        premium_budget_usd=250,
        fee_reserve_per_package_usd=10,
        entry_limit="SUM_OF_ASKS",
        quote_max_age_seconds=5,
        entry_deadline_seconds=180,
        exit_seconds_before_close=120,
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
    ib.client = NS(getReqId=Mock(return_value=101), clientId=81)
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


def opening_runtime(tmp_path, monkeypatch):
    from datetime import date

    config = armed_config().model_copy(
        update={"armed": False, "arm_after_quote_check_on": date(2025, 7, 21)}
    )
    runtime = Runtime(config, Store(tmp_path / "opening.sqlite"))
    runtime.broker = PaperBroker(config, runtime.store, fake_ib())
    runtime.broker.reconciled = True
    runtime.broker.entry_blocker = ""
    runtime.session, runtime.problem = "2025-07-21", ""
    runtime.schedule = [(runtime.session, OPEN, CLOSE)]
    monkeypatch.setattr("stocker_execution.first4_runtime.now", lambda: OPEN)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    return runtime


def test_opening_check_arms_same_runtime_without_replay(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    report = {
        "checks": {
            "fresh_realtime_option_quotes": True,
            "qualified_usd_standard_multiplier": True,
            "combo_price_increment": True,
        },
        "blockers": [],
    }

    async def probe(*args):
        # Scanner continues to freeze observations while the diagnostic awaits data.
        runtime.store.observe("2025-07-21", OPEN + timedelta(minutes=1), CLOSE, [])
        return report

    monkeypatch.setattr("stocker_execution.first4_runtime.option_access", probe)
    assert not runtime.status()["armed"]
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert runtime.status()["armed"]
    assert runtime.config.armed is False
    assert runtime.store.rows("sessions")[0]["last_clock"]
    assert not runtime.store.rows("events")
    assert not runtime.store.rows("orders")
    runtime.broker.ib.placeOrder.assert_not_called()
    runtime.broker.disconnected()
    assert not runtime.broker.entries_armed()
    runtime.broker.reconciled = True
    assert not runtime.broker.entries_armed()
    # An audit record is not authority to resume entries in a restarted process.
    restarted = PaperBroker(runtime.config, runtime.store, fake_ib())
    restarted.reconciled = True
    assert not restarted.entries_armed()


@pytest.mark.parametrize("failure", ["quote", "pause", "continuity", "account", "deadline"])
def test_opening_check_failure_cannot_arm(tmp_path, monkeypatch, failure):
    runtime = opening_runtime(tmp_path, monkeypatch)

    async def probe(*args):
        if failure == "quote":
            raise ValueError("OPTION_QUOTES_INVALID_STALE_OR_UNAVAILABLE")
        if failure == "pause":
            runtime.pause = True
        if failure == "continuity":
            runtime.store.block(runtime.session, "SCANNER_MINUTE_MISSED")
        if failure == "account":
            runtime.broker.ib.managedAccounts.return_value = ["U12345"]
        if failure == "deadline":
            monkeypatch.setattr(
                "stocker_execution.first4_runtime.now", lambda: OPEN + timedelta(minutes=14)
            )
        return {
            "checks": {
                "fresh_realtime_option_quotes": True,
                "qualified_usd_standard_multiplier": True,
                "combo_price_increment": True,
            },
            "blockers": [],
        }

    monkeypatch.setattr("stocker_execution.first4_runtime.option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert not runtime.status()["armed"]
    assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "FAILED"
    runtime.broker.ib.placeOrder.assert_not_called()


def test_opening_authority_expires_and_blocked_session_cannot_submit(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    runtime.broker.opening_verified_session = "2025-07-21"
    assert runtime.broker.entries_armed()
    runtime.store.block("2025-07-21", "SCANNER_MINUTE_MISSED")
    assert not runtime.broker.entries_armed()
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN + timedelta(days=1))
    assert not runtime.broker.entries_armed()


def test_opening_check_and_manager_do_not_overlap_ib_request_keys(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    active = 0

    async def open_orders():
        nonlocal active
        active += 1
        assert active == 1, "IB fixed request key overwritten by concurrent reconciliation"
        await asyncio.sleep(0.15)
        active -= 1
        return []

    runtime.broker.ib.reqAllOpenOrdersAsync = AsyncMock(side_effect=open_orders)
    monkeypatch.setattr(
        "stocker_execution.first4_runtime.option_access",
        AsyncMock(
            return_value={
                "checks": {
                    "fresh_realtime_option_quotes": True,
                    "qualified_usd_standard_multiplier": True,
                    "combo_price_increment": True,
                },
                "blockers": [],
            }
        ),
    )

    async def run():
        opening = asyncio.create_task(runtime.check_opening("2025-07-21", OPEN))
        await asyncio.sleep(0)
        manager = asyncio.create_task(runtime.maintain_broker())
        await asyncio.wait_for(opening, 2)
        manager.cancel()
        await asyncio.gather(manager, return_exceptions=True)
        assert runtime.status()["armed"]
        # Other callers are serialized as well, not just the maintenance loop.
        await asyncio.gather(runtime.broker.reconcile(), runtime.broker.reconcile())

    asyncio.run(run())
    assert runtime.broker.ib.reqAllOpenOrdersAsync.await_count == 3


def test_opening_enabled_path_submits_only_persisted_paper_entry(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    b = runtime.broker
    b.opening_verified_session = "2025-07-21"
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: datetime.fromisoformat(event["entry_at"])
    )
    combo = Contract(secType="BAG", symbol="A", currency="USD", exchange="SMART")
    b.ib.placeOrder.side_effect = lambda c, o: Trade(
        contract=c, order=o, orderStatus=OrderStatus(status="Submitted", permId=22)
    )
    b.submit(event, combo, LimitOrder("BUY", 1, 0.7), {}, "ENTRY")
    assert b.ib.placeOrder.call_args.args[1].account == PAPER_ACCOUNT
    assert b.store.rows("orders")[0]["status"] == "Submitted"
    with pytest.raises(sqlite3.IntegrityError):
        b.submit(event, combo, LimitOrder("BUY", 1, 0.7), {}, "ENTRY")
    assert b.ib.placeOrder.call_count == 1
    b.disconnected()
    b.reconciled = True
    with pytest.raises(ValueError, match="unarmed"):
        b.submit(event, combo, LimitOrder("BUY", 1, 0.7), {}, "ENTRY")


def test_shared_option_access_checks_quotes_after_metadata_without_orders(tmp_path, monkeypatch):
    from stocker_execution.first4_readiness import option_access

    b = broker(tmp_path)
    monkeypatch.setattr("stocker_execution.first4_readiness.now", lambda: OPEN)
    b.ib.reqMarketDataType = Mock()
    b.ib.qualifyContractsAsync = AsyncMock(return_value=[Contract(conId=42, symbol="F")])
    b.ib.reqTickersAsync = AsyncMock(
        return_value=[NS(marketPrice=lambda: 13, marketDataType=1, time=OPEN)]
    )
    b.chain = AsyncMock(
        return_value=[
            NS(
                exchange="SMART",
                tradingClass="F",
                multiplier="100",
                expirations={"20250723"},
                strikes={13},
            )
        ]
    )

    def details(c):
        c.conId = 101 if c.right == "P" else 102
        c.localSymbol = f"F     250723{c.right}00013000"
        return [
            NS(
                contract=c,
                underConId=42,
                minSize=1,
                sizeIncrement=1,
                minTick=0.01,
                realExpirationDate="20250723",
                lastTradeTime="16:00:00",
                timeZoneId="US/Eastern",
                orderTypes="LMT,GTD",
            )
        ]

    b.ib.reqContractDetailsAsync = AsyncMock(side_effect=details)
    calls = []

    async def tick(*args):
        calls.append("metadata")
        return 0.01

    async def quotes(*args):
        calls.append("quotes")
        return [
            {"bid": 0.1, "ask": 0.2, "bid_at": OPEN.isoformat(), "ask_at": OPEN.isoformat()}
        ] * 2

    b.combo_tick, b.quotes = tick, quotes
    result = asyncio.run(option_access(b, OPEN + timedelta(minutes=14)))
    assert all(result["checks"].values())
    assert calls == ["metadata", "quotes"]
    b.ib.placeOrder.assert_not_called()
    assert not b.store.rows("events") and not b.store.rows("orders")


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


def test_strict_otm_nearest_tolerance_ties_and_expiry_calendar_days():
    assert listed_strike([101, 103, 100], 100, "C") == 103
    assert listed_strike([97, 99, 100], 100, "P") == 97
    with pytest.raises(ValueError):
        listed_strike([10, 11], 10, "C")
    target = OPEN + timedelta(days=2)
    assert (
        listed_expiry(
            {"20250722": target - timedelta(days=1), "20250724": target + timedelta(days=1)}, OPEN
        )
        == "20250724"
    )
    # A later date label can exceed the allowed 24 hours at its actual 16:00 ET expiry.
    assert (
        listed_expiry(
            {
                "20250722": target - timedelta(hours=17.5),
                "20250724": target + timedelta(hours=30.5),
            },
            OPEN,
        )
        == "20250722"
    )
    assert listed_expiry({"20250721": OPEN, "20250723": target}, OPEN) == "20250723"
    with pytest.raises(ValueError):
        listed_expiry(
            {"20250721": OPEN, "20250724": target + timedelta(days=1, microseconds=1)}, OPEN
        )


def test_leg_fills_idempotent_and_combo_status_not_fills(tmp_path):
    b = broker(tmp_path)
    event = dict(session="2025-07-21", symbol="A", slot=1)
    ref = b.store.reserve_order(
        event, "ENTRY", 1, {"put": {"conId": 100}, "call": {"conId": 101}, "quantity": 1}
    )
    b.order_status(
        NS(
            order=Order(account=PAPER_ACCOUNT, clientId=81, orderRef=ref, orderId=1, permId=22),
            orderStatus=NS(status="Filled", permId=22),
        )
    )
    assert b.owned_quantities() == {}
    execution = NS(
        acctNumber=PAPER_ACCOUNT,
        clientId=81,
        orderId=1,
        permId=22,
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
    with b.store.db:
        b.store.db.execute("UPDATE first4_orders SET perm_id=9001 WHERE reference=?", (ref,))
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
        NS(
            contract=Contract(
                secType="OPT", conId=i, multiplier="100", strike=9.8 if i == 101 else 10.2
            ),
            sizeIncrement=1,
            minSize=1,
            realExpirationDate="20250723",
            lastTradeTime="16:00:00",
            timeZoneId="US/Eastern",
        )
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
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: datetime.fromisoformat(event["entry_at"])
    )
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
        "exit_at": (CLOSE - timedelta(seconds=120)).isoformat(),
        "entry_deadline_at": (
            datetime.fromisoformat(event["entry_at"]) + timedelta(seconds=180)
        ).isoformat(),
        "quotes": [{"ask": 1}, {"ask": 1}],
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
                        clientId=81,
                        orderId=1,
                        permId=22,
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
    b.quotes = AsyncMock(
        return_value=[
            {
                "bid": 0.8,
                "ask": 1,
                "bid_at": (CLOSE - timedelta(seconds=20)).isoformat(),
                "ask_at": (CLOSE - timedelta(seconds=20)).isoformat(),
            }
        ]
        * 2
    )
    b.combo_tick = AsyncMock(return_value=0.01)
    b.ib.reqContractDetailsAsync = AsyncMock(
        return_value=[NS(validExchanges="SMART", marketRuleIds="32")]
    )
    b.ib.reqMarketRuleAsync = AsyncMock(return_value=[NS(lowEdge=0, increment=0.01)])
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
    b.quotes = AsyncMock(
        return_value=[
            {
                "bid": 0.8,
                "ask": 1,
                "bid_at": (CLOSE - timedelta(seconds=20)).isoformat(),
                "ask_at": (CLOSE - timedelta(seconds=20)).isoformat(),
            }
        ]
    )
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
        == "ENTRY_UNFILLED"
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


def test_contract_mapping_verifies_standard_roots_multiplier_and_expiry(tmp_path):
    b = broker(tmp_path)
    underlying = Contract(secType="STK", conId=1, symbol="ABC")
    b.chain = AsyncMock(
        return_value=[
            NS(
                exchange="SMART",
                tradingClass="ABC",
                multiplier="100",
                expirations={"20250722", "20250724"},
                strikes=[97, 99, 100, 101, 103],
            )
        ]
    )

    def qualified(c):
        c.conId = 101 if c.right == "P" else 102
        c.localSymbol = (
            f"ABC   {c.lastTradeDateOrContractMonth[2:]}{c.right}{round(c.strike * 1000):08d}"
        )
        return [
            NS(
                contract=c,
                underConId=1,
                realExpirationDate=c.lastTradeDateOrContractMonth,
                lastTradeTime="16:00:00",
                timeZoneId="US/Eastern",
                orderTypes="LMT,GTD",
            )
        ]

    b.contract_details = AsyncMock(side_effect=qualified)
    p, c, combo = asyncio.run(b.contracts(underlying, 100, OPEN))
    assert (p.contract.strike, c.contract.strike) == (97, 103)
    assert (
        p.contract.lastTradeDateOrContractMonth
        == c.contract.lastTradeDateOrContractMonth
        == "20250722"
    )
    assert [leg.conId for leg in combo.comboLegs] == [101, 102]

    def adjusted(c):
        result = qualified(c)
        c.localSymbol = "ABC1  250724P00097000"
        return result

    b.contract_details.side_effect = adjusted
    with pytest.raises(ValueError, match="NONSTANDARD"):
        asyncio.run(b.contracts(underlying, 100, OPEN))


@pytest.mark.parametrize(
    "change,allowed",
    [
        ({}, True),
        ({"bid": 0}, False),
        ({"bid": 2}, False),
        ({"askSize": 0}, False),
        ({"marketDataType": 2}, False),
        ({"marketDataType": 3}, False),
        ({"marketDataType": 4}, False),
        ({"age": 6}, False),
        ({"price_ticks": False}, False),
    ],
)
def test_quote_freshness_uses_price_observations_and_rejects_bad_market_data(
    tmp_path, change, allowed
):
    b = broker(tmp_path)

    class Updates:
        def __iadd__(self, callback):
            asyncio.get_running_loop().call_soon(callback, ticker)
            return self

        def __isub__(self, callback):
            return self

    stamp = datetime.now(UTC) - timedelta(seconds=change.get("age", 0))
    ticker = NS(
        bid=1,
        ask=1.1,
        bidSize=1,
        askSize=1,
        marketDataType=1,
        time=datetime.now(UTC),
        ticks=[
            NS(tickType=i, time=stamp)
            for i in ([1, 2] if change.get("price_ticks", True) else [0, 3])
        ],
        updateEvent=Updates(),
    )
    for k, v in change.items():
        setattr(ticker, k, v)
    b.ib.reqMktData = Mock(return_value=ticker)
    b.ib.cancelMktData = Mock()

    async def check():
        return await b.quotes([Contract(conId=101)], datetime.now(UTC) + timedelta(seconds=0.03))

    if allowed:
        assert asyncio.run(check())[0]["ask_size"] == 1
    else:
        with pytest.raises(ValueError, match="QUOTES"):
            asyncio.run(check())
    assert b.ib.cancelMktData.call_count == 1


def test_budget_reserves_pending_and_filled_entries_and_never_scales(tmp_path, monkeypatch):
    b = broker(tmp_path)
    events = b.store.observe(
        "2025-07-21",
        OPEN + timedelta(minutes=15),
        CLOSE,
        [candidate(str(i), i) for i in range(1, 5)],
    )
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: datetime.fromisoformat(events[0]["entry_at"])
    )
    b.ib.client.getReqId.side_effect = range(10, 20)
    b.ib.placeOrder.side_effect = lambda c, o: Trade(
        contract=c, order=o, orderStatus=OrderStatus(status="Submitted", permId=o.orderId)
    )
    combo = Contract(secType="BAG")
    with pytest.raises(ValueError, match="PREMIUM"):
        b.submit(events[0], combo, LimitOrder("BUY", 1, 2.51), {}, "ENTRY")
    with pytest.raises(ValueError, match="ONE_DEBIT"):
        b.submit(events[0], combo, LimitOrder("BUY", 2, 0.01), {}, "ENTRY")
    for e in events:
        b.submit(e, combo, LimitOrder("BUY", 1, 2.5), {}, "ENTRY")
    assert sum(json.loads(o["payload"])["allocation_usd"] for o in b.store.rows("orders")) == 1040
    b.store.db.execute("UPDATE first4_orders SET status='Filled'")
    with pytest.raises(ValueError, match="SESSION_ALLOCATION"):
        b.store.reserve_order(dict(session="2025-07-21", symbol="FIFTH", slot=5), "ENTRY", 99, {})
    b.store.db.close()
    reopened = Store(tmp_path / "s.sqlite")
    assert len(reopened.rows("orders")) == 4


def test_entry_deadline_cancel_waits_for_ack_and_reconciles_partial_legs(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b, (1, 0))
    reference = b.store.rows("orders")[0]["reference"]
    b.store.db.execute("UPDATE first4_orders SET status='Submitted'")
    trade = Trade(
        order=Order(account=PAPER_ACCOUNT, clientId=81, orderRef=reference, orderId=1),
        orderStatus=OrderStatus(status="Submitted", permId=22),
    )
    b.ib.openTrades.return_value = [trade]
    b.ib.cancelOrder = Mock()
    deadline = datetime.fromisoformat(payload["entry_deadline_at"])
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: deadline - timedelta(microseconds=1)
    )
    asyncio.run(b.cancel_due_entries())
    assert not b.ib.cancelOrder.called
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: deadline)
    asyncio.run(b.cancel_due_entries())
    assert b.ib.cancelOrder.call_count == 1 and b.entry_blocker
    trade.orderStatus.status = "PendingCancel"
    asyncio.run(b.cancel_due_entries())
    assert b.ib.cancelOrder.call_count == 1
    trade.orderStatus.status = "Cancelled"
    b.order_status(trade)
    b.ib.openTrades.return_value = []
    b.ib.reqPositionsAsync.return_value = [
        NS(account=PAPER_ACCOUNT, contract=Contract(conId=101), position=1, avgCost=100)
    ]
    asyncio.run(b.cancel_due_entries())
    assert b.reconciled and not b.entry_blocker
    assert json.loads(b.store.rows("orders")[0]["payload"])["deadline_reconciled"]
    assert b.store.rows("events")[0]["outcome"] == "ENTRY_RECONCILED_HELD"
    assert not b.ib.placeOrder.called


def test_exit_ignores_entry_budget_pause_and_arming_but_never_invents_fills(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b)
    b.config = b.config.model_copy(update={"armed": False})
    b.store.set_meta("paused", True)
    b.entry_blocker = "ENTRY_BUDGET_REACHED"
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    order = b.ib.placeOrder.call_args.args[1]
    assert order.orderType == "MKT"
    assert b.store.rows("events")[0]["outcome"] == "EXIT_SUBMITTED"
    assert len(b.store.rows("fills")) == 2
    asyncio.run(b.close_due())
    assert b.ib.placeOrder.call_count == 1
    assert b.store.rows("events")[0]["outcome"] != "CLOSED"


def test_actual_fees_are_separate_from_reserve_and_unset_fee_is_not_real(tmp_path):
    b = broker(tmp_path)
    prepare_exit(b)
    b.commission(None, None, NS(currency="USD", commission=1.7976931348623157e308, execId="101"))
    assert b.store.rows("fills")[0]["commission"] is None
    b.commission(None, None, NS(currency="USD", commission=0.65, execId="101"))
    assert b.store.rows("fills")[0]["commission"] == 0.65


def test_exit_rechecks_positions_after_async_quotes(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b)
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    quotes = b.quotes.return_value

    async def changed(*args, **kwargs):
        b.ib.positions.return_value = []
        return quotes

    b.quotes = changed
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert not b.reconciled
    assert "EXIT_POSITION_CHANGED" in b.problem
    assert b.store.rows("events")[0]["outcome"] != "CLOSED"


def test_terminal_second_order_cannot_clear_first_pending_cancellation(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b, (1, 0))
    first_ref = b.store.rows("orders")[0]["reference"]
    b.store.db.execute("UPDATE first4_orders SET status='PendingCancel'")
    second = b.store.observe(
        "2025-07-21", OPEN + timedelta(minutes=16), CLOSE, [candidate("B", 1)]
    )[0]
    second_ref = b.store.reserve_order(second, "ENTRY", 2, payload)
    b.store.db.execute(
        "UPDATE first4_orders SET status='Cancelled' WHERE reference=?", (second_ref,)
    )
    trade = Trade(
        order=Order(account=PAPER_ACCOUNT, clientId=81, orderRef=first_ref, orderId=1),
        orderStatus=OrderStatus(status="PendingCancel", permId=22),
    )
    b.ib.openTrades.return_value = [trade]
    b.ib.reqAllOpenOrdersAsync.return_value = [trade]
    b.ib.reqPositionsAsync.return_value = [
        NS(account=PAPER_ACCOUNT, contract=Contract(conId=101), position=1, avgCost=100)
    ]
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN + timedelta(minutes=25))
    asyncio.run(b.cancel_due_entries())
    assert (
        b.entry_blocker
        == b.deadline_blocker()
        == "ENTRY_DEADLINE_AWAITING_CANCEL_FILL_RECONCILIATION"
    )
    second_payload = json.loads(
        next(o for o in b.store.rows("orders") if o["reference"] == second_ref)["payload"]
    )
    assert second_payload["deadline_reconciled"]
