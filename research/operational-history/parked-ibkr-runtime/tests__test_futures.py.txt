"""Offline contracts and broker lifecycle fixtures; no connectivity test orders."""

import asyncio
import importlib.util
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from eventkit import Event
from ib_async import Contract, ContractDetails, Order, OrderStatus, Trade

from stocker_dashboard.app import create_dashboard_app
from stocker_execution.broker import PaperBroker
from stocker_execution.config import (
    MARKETS,
    PAPER_ACCOUNT,
    FuturesConfig,
    MarketDataConfig,
    ProductMapping,
)
from stocker_execution.contracts import (
    Calendar,
    Quote,
    budget,
    nearby_futures,
    select_future,
    verify_option,
)
from stocker_execution.requests import BrokerConnection
from stocker_execution.rules import (
    Bar,
    clocks,
    eligibility,
    frozen_strike,
    model_delta,
    opportunity,
    prior_rv,
)
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store

AT = datetime(2026, 9, 28, 13, tzinfo=UTC)


def mapping(**changes):
    return ProductMapping(
        **{
            **dict(
                product="GC",
                symbol="GC",
                exchange="COMEX",
                trading_class="OG",
                currency="USD",
                multiplier=100,
                price_unit_factor=1,
                price_magnifier=1,
                delta_tolerance=0.01,
                expiry_timezone="America/New_York",
                termination_time="16:00:00",
                settlement="FUTURES",
                fee_reserve_gbp=2,
                source="OFFLINE TEST FIXTURE",
                approval="OFFLINE TEST FIXTURE ONLY",
                strike_rule="NEAREST_FROZEN_MODEL_DELTA",
            ),
            **changes,
        }
    )


def plan(cid=100):
    return {
        **budget(0.1, 100, 1, 0.8, AT, AT, 2),
        "option": Contract(
            secType="FOP",
            conId=cid,
            symbol="GC",
            exchange="COMEX",
            currency="USD",
            multiplier="100",
            right="P",
        ).dict(),
        "currency": "USD",
        "multiplier": 100,
        "price_unit_factor": 1,
        "limit": 0.1,
        "increments": [[0, 0.01]],
        "expiry_at": (AT + timedelta(hours=7)).isoformat(),
        "exit_at": (AT + timedelta(hours=1)).isoformat(),
        "signal_at": AT.isoformat(),
        "bid": 0.09,
        "ask": 0.1,
        "bid_at": AT.isoformat(),
        "ask_at": AT.isoformat(),
        "mapping": mapping().model_dump(),
    }


class FakeIB:
    def __init__(self):
        for name in (
            "execDetailsEvent",
            "commissionReportEvent",
            "orderStatusEvent",
            "positionEvent",
            "disconnectedEvent",
            "errorEvent",
        ):
            setattr(self, name, Event(name))
        self.accounts = [PAPER_ACCOUNT]
        self.connected = True
        self.counter = 10
        self.client = NS(clientId=83, getReqId=self.next_id)
        from stocker_execution.requests import BrokerConnection

        self.wrapper = BrokerConnection().wrapper
        self.client.reqMktData = lambda *args: None
        self.client.cancelMktData = lambda *args: None
        self.wrapper.depth_handlers = {}
        self.trades = []
        self.executions = []
        self.positions = []
        self.cancels = []
        self.raise_after_send = False

    def next_id(self):
        self.counter += 1
        return self.counter

    def isConnected(self):
        return self.connected

    def managedAccounts(self):
        return self.accounts

    def disconnect(self):
        self.connected = False
        self.disconnectedEvent.emit()

    def placeOrder(self, contract, order):
        order.clientId = 83
        order.permId = 1000 + order.orderId
        t = Trade(
            contract,
            order,
            OrderStatus(
                orderId=order.orderId, status="Submitted", remaining=1, permId=order.permId
            ),
        )
        self.trades.append(t)
        self.orderStatusEvent.emit(t)
        if self.raise_after_send:
            raise TimeoutError("uncertain transport")
        return t

    def openTrades(self):
        return [t for t in self.trades if not t.isDone()]

    def cancelOrder(self, order):
        self.cancels.append(order.orderRef)

    async def reqAllOpenOrdersAsync(self):
        return self.openTrades()

    async def reqCompletedOrdersAsync(self, apiOnly):
        return [t for t in self.trades if t.isDone()]

    async def reqExecutionsAsync(self, query):
        return self.executions

    async def reqPositionsAsync(self):
        return self.positions


def setup(tmp_path, monkeypatch, armed=True):
    monkeypatch.setattr("stocker_execution.broker.now", lambda: AT)
    ib = FakeIB()
    store = Store(tmp_path / "futures.sqlite")
    broker = PaperBroker(
        FuturesConfig(
            armed=armed,
            mappings={"GC": mapping()},
            market_data=MarketDataConfig(
                allowance_status="CONFIGURED",
                allowance_source="OFFLINE TEST FIXTURE",
                cancel_drain_seconds=0.1,
            ),
        ),
        store,
        ib,
    )
    broker.reconciled = True
    broker.reconciled_at = AT
    broker.fx = (0.8, AT)
    return store, broker, ib


def record_entry(store, identity="x", cid=100):
    event = {**opportunity("GC", 1, AT), "id": identity}
    store.observe(event, "", {})
    assert store.reserve(identity, plan(cid)) == ""
    return store.prepare_order(
        identity,
        "ENTRY",
        10 + cid,
        (AT + timedelta(seconds=20)).isoformat(),
        {"con_id": cid, "quantity": 1, "limit": 0.1},
    )


def fill_record(ref, exec_id="exec.1", side="BOT", at=AT):
    return dict(
        exec_id=exec_id,
        reference=ref,
        con_id=100,
        quantity=1,
        price=0.1,
        side=side,
        at=at.isoformat(),
        fx=0.8,
        fx_at=at.isoformat(),
    )


def test_source_fixture_rv_clocks_and_original_exit_anchor():
    fixture = json.loads(Path("tests/fixtures/futures/frozen.json").read_text())
    for row in fixture["rows"]:
        at = datetime.fromisoformat(row["at"]).astimezone(UTC)
        bars = [
            Bar(at - timedelta(minutes=31 - i), p, p, p, p, 1, p)
            for i, p in enumerate(row["pre31"])
        ]
        assert prior_rv(bars, at) == pytest.approx(row["rv15"], rel=1e-10, abs=1e-14)
        event = opportunity(row["market"], row["con_id"], at)
        assert datetime.fromisoformat(event["exit_at"]) == at + timedelta(minutes=60)
        expiry = at.replace(hour=21)
        strike = frozen_strike(
            row["price"], row["rv15"], at, expiry, event["right"], event["target_delta"]
        )
        assert model_delta(
            row["price"], strike, row["rv15"], at, expiry, event["right"]
        ) == pytest.approx(event["target_delta"], abs=1e-10)
        # An incomplete current bar cannot contaminate the completed prefix.
        assert prior_rv(bars + [Bar(at, 999, 999, 999, 999, 1, 999)], at) == prior_rv(bars, at)
        with pytest.raises(ValueError, match="INCOMPLETE"):
            prior_rv(bars[:-1], at)


def test_dst_weekends_and_only_approved_veto():
    assert clocks(date(2026, 3, 9))[0].hour == 13  # US DST precedes UK
    assert clocks(date(2026, 3, 2))[0].hour == 14
    assert clocks(date(2026, 10, 26))[0].hour == 13  # UK reverts before US
    assert clocks(date(2026, 11, 2))[0].hour == 14
    assert clocks(date(2026, 9, 27)) == []
    for market in MARKETS:
        for at in clocks(date(2026, 9, 28)):
            event = opportunity(market, 1, at)
            assert bool(event["veto"]) == (market == "NG" and at.hour == 17)
    assert opportunity("GC", 1, AT + timedelta(hours=7))["veto"] == ""


@pytest.mark.parametrize(
    "price,fee,allowed", [(0.075, 2, True), (0.1, 2, True), (0.100125, 2, False), (0.125, 2, False)]
)
def test_integer_budget_includes_multiplier_fx_and_fees(price, fee, allowed):
    if allowed:
        b = budget(price, 100, 1, 0.8, AT, AT, fee)
        assert b["quantity"] == 1 and b["cash_pennies"] <= 1000
    else:
        with pytest.raises(ValueError, match="SKIP_BUDGET_TOO_SMALL"):
            budget(price, 100, 1, 0.8, AT, AT, fee)
    with pytest.raises(ValueError, match="FX_STALE"):
        budget(price, 100, 1, 0.8, AT - timedelta(seconds=31), AT, fee)


def test_atomic_simultaneous_capacity_and_midnight(tmp_path):
    p = tmp_path / "futures.sqlite"
    s = Store(p)
    for i in range(12):
        s.observe({**opportunity("GC", 1, AT), "id": str(i)}, "", {})

    def attempt(i):
        other = Store(p)
        try:
            return other.reserve(str(i), plan())
        finally:
            other.db.close()

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(attempt, range(12)))
    assert results.count("") == 4 and results.count("SKIP_CAPACITY_FULL") == 8
    assert s.capacity() == {"reserved_open_trades": 4, "allocation_pennies": 4000}
    s.db.close()
    s = Store(p)
    assert s.capacity()["reserved_open_trades"] == 4
    assert s.reserve("0", plan()) == "DUPLICATE_OPPORTUNITY" or results[0] != ""


def test_cancel_uncertainty_and_late_fill_keep_obligation(tmp_path):
    s = Store(tmp_path / "futures.sqlite")
    ref = record_entry(s)
    assert not s.confirm_closed("x", 0)
    with s.db:
        s.db.execute("UPDATE orders SET status='Cancelled' WHERE reference=?", (ref,))
    assert s.confirm_closed("x", 0)
    assert s.capacity()["reserved_open_trades"] == 0
    s.record_fill(fill_record(ref))
    assert s.capacity()["reserved_open_trades"] == 1
    assert not s.confirm_closed("x", 0)
    with s.db:
        s.db.execute("UPDATE orders SET filled=1 WHERE reference=?", (ref,))
    assert not s.confirm_closed("x", 1)


def test_missing_commission_is_provisional_and_exit_submission_does_not_release(tmp_path):
    s = Store(tmp_path / "futures.sqlite")
    ref = record_entry(s)
    s.record_fill(fill_record(ref))
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (ref,)
        )
    exit_ref = s.prepare_order("x", "EXIT", 900, AT.isoformat(), {"con_id": 100})
    assert not s.confirm_closed("x", 1)
    s.record_fill({**fill_record(exit_ref, "exit.1", "SLD"), "price": 0.15})
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (exit_ref,)
        )
    assert s.confirm_closed("x", 0)
    assert s.economics()["realised_net_gbp"] is None
    with s.db:
        s.db.execute("UPDATE fills SET commission=1,commission_currency='USD'")
    assert s.economics()["realised_net_gbp"] == pytest.approx(2.4)
    assert s.economics()["closed_with_complete_costs"] == 1
    # A terminal unfilled exit attempt has no commission to await.
    with s.db:
        s.db.execute(
            "INSERT INTO orders"
            "(reference,event_id,role,order_id,status,filled,remaining,deadline,payload) "
            "VALUES('cancelled-exit','x','EXIT',901,'Cancelled',0,0,?,'{}')",
            (AT.isoformat(),),
        )
    assert s.economics()["realised_net_gbp"] == pytest.approx(2.4)
    s.record_fill({**fill_record(ref, "exec.2"), "price": 0.11})
    assert s.exposure("x") == 0
    assert len(s.fills("x")) == 2


def test_real_order_path_and_uncertain_submission_no_retry(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)
    ib.raise_after_send = True
    e = opportunity("GC", 1, AT)
    s.observe(e, "", {})

    async def scenario():
        with pytest.raises(TimeoutError):
            await b.enter(e, plan())
        assert len(ib.trades) == 1 and s.capacity()["reserved_open_trades"] == 1
        await b.reconcile()
        assert await b.enter(e, plan()) == "DUPLICATE_OPPORTUNITY"

    asyncio.run(scenario())
    order = ib.trades[0].order
    assert (order.account, order.totalQuantity, order.action, order.orderType) == (
        PAPER_ACCOUNT,
        1,
        "BUY",
        "LMT",
    )
    assert order.transmit and order.lmtPrice == 0.1


def test_fx_revalidated_at_admission_without_strike_substitution(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)
    e = opportunity("GC", 1, AT)
    s.observe(e, "", {})
    b.fx = (0.81, AT)
    with pytest.raises(ValueError, match="SKIP_BUDGET_TOO_SMALL"):
        asyncio.run(b.enter(e, plan()))
    assert not ib.trades and s.capacity()["reserved_open_trades"] == 0


def test_listed_selection_quote_budget_path_never_searches_for_cheaper_contract(
    tmp_path, monkeypatch
):
    s, b, ib = setup(tmp_path, monkeypatch)
    event = opportunity("GC", 1, AT)
    future = Contract(secType="FUT", conId=1, symbol="GC", exchange="COMEX", currency="USD")
    expiry = AT.replace(hour=20)
    target = frozen_strike(100, 0.001, AT, expiry, "P", 0.1)
    strikes = [target - 0.01, target + 0.01, target + 1]
    requested = []
    ask = 0.1

    async def chain(*args):
        # Replay the real ib_async wire decoder: it delivers the underlying ID as text.
        connection = BrokerConnection()
        connection.client.getReqId = lambda: 91

        def reply(rid, *request):
            connection.client.decoder.interpret(
                [
                    "75",
                    str(rid),
                    "COMEX",
                    "1",
                    "OG",
                    "100",
                    "1",
                    "20260928",
                    str(len(strikes)),
                    *(str(s) for s in strikes),
                ]
            )
            connection.client.decoder.interpret(["76", str(rid)])

        connection.client.reqSecDefOptParams = reply
        return await connection.reqSecDefOptParamsAsync(*args)

    async def details(contract):
        requested.append(contract.strike)
        contract.conId = 100
        return [
            ContractDetails(
                contract=contract,
                underConId=1,
                realExpirationDate="20260928",
                lastTradeTime="16:00:00",
                timeZoneId="America/New_York",
                priceMagnifier=1,
                tradingHours="20260928:0800-20260928:1700",
            )
        ]

    async def quote(contract):
        return Quote(0.09, ask, AT, AT, 1)

    async def increments(detail):
        return [(0, 0.01)]

    ib.reqSecDefOptParamsAsync = chain
    ib.reqContractDetailsAsync = details
    b.quote, b.increments = quote, increments

    async def scenario():
        nonlocal ask
        result = await b.prepare(event, future, {"futures_price": 100, "rv15": 0.001})
        assert result["quantity"] == 1 and result["cash_pennies"] == 1000
        assert result["option"]["conId"] == 100 and result["signal_future"]["conId"] == 1
        ask = 0.11
        with pytest.raises(ValueError, match="SKIP_BUDGET_TOO_SMALL"):
            await b.prepare(event, future, {"futures_price": 100, "rv15": 0.001})
        assert len(requested) == 2 and requested[0] == requested[1]
        assert not ib.trades and s.capacity()["reserved_open_trades"] == 0

    asyncio.run(scenario())


def test_initial_broker_sync_serializes_reconciliation(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []

        async def connect(*args, **kwargs):
            entered.set()  # socket is ready; IB's initial account sync still pending
            await release.wait()

        async def opened():
            assert release.is_set()
            requests.append("openOrders")
            return []

        async def qualify(*args):
            return [Contract(conId=90, secType="CASH", exchange="IDEALPRO", currency="USD")]

        ib.connectAsync = connect
        ib.reqAllOpenOrdersAsync = opened
        ib.reqMarketDataType = lambda kind: None
        ib.qualifyContractsAsync = qualify
        ib.reqMktData = lambda *args: NS(marketDataType=0, updateEvent=Event())
        initializing = asyncio.create_task(b.connect())
        await entered.wait()
        reconciling = asyncio.create_task(b.reconcile())
        await asyncio.sleep(0)
        assert not b.synchronized and not requests
        release.set()
        await asyncio.gather(initializing, reconciling)
        assert b.synchronized and len(requests) == 2

    asyncio.run(scenario())


def test_closed_trade_releases_while_same_contract_remains_owned(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)
    for index in range(2):
        identity = f"same-{index}"
        e = {**opportunity("GC", 1, AT), "id": identity}
        s.observe(e, "", {})
        assert s.reserve(identity, plan()) == ""
        ref = s.prepare_order(identity, "ENTRY", 100 + index, AT.isoformat(), {"con_id": 100})
        s.record_fill(fill_record(ref, f"buy{index}.1"))
        if index == 0:
            ref = s.prepare_order(identity, "EXIT", 200, AT.isoformat(), {"con_id": 100})
            s.record_fill(fill_record(ref, "sell0.1", "SLD"))
    with s.db:
        s.db.execute("UPDATE orders SET status='Filled',filled=1,remaining=0")
    ib.positions = [
        NS(account=PAPER_ACCOUNT, contract=Contract(conId=100, secType="FOP"), position=1)
    ]
    asyncio.run(b.reconcile())
    assert b.reconciled and s.capacity()["reserved_open_trades"] == 1
    assert s.active()[0]["id"] == "same-1"


def test_wrong_account_unarmed_and_unknown_positions_block_entries(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch, False)
    assert b.entry_reason() == "EXECUTION_UNARMED"
    ib.accounts = ["U12345"]
    with pytest.raises(ValueError, match="PAPER_ACCOUNT"):
        b.guard()
    ib.accounts = [PAPER_ACCOUNT]
    ib.positions = [
        NS(account=PAPER_ACCOUNT, contract=Contract(conId=999, secType="FUT"), position=1)
    ]
    asyncio.run(b.reconcile())
    assert "UNEXPLAINED" in b.entry_reason()
    assert not ib.cancels and not ib.trades


def test_exit_management_while_entries_paused_and_unarmed(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch, False)
    ref = record_entry(s)
    s.record_fill(fill_record(ref))
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (ref,)
        )
        s.db.execute("INSERT INTO positions VALUES(100,1,'{}')")
    s.set_meta("paused", True)
    later = AT + timedelta(hours=1)
    monkeypatch.setattr("stocker_execution.broker.now", lambda: later)
    b.reconciled_at = later

    async def quote(c):
        return Quote(0.08, 0.1, later, later, 1)

    b.quote = quote
    asyncio.run(b.manage_one(s.active()[0]))
    assert ib.trades[0].order.action == "SELL"
    assert s.capacity()["reserved_open_trades"] == 1
    assert datetime.fromisoformat(s.active()[0]["exit_at"]) == later


@pytest.mark.parametrize("kind", [2, 3, 4])
def test_frozen_and_delayed_quotes_rejected(kind):
    with pytest.raises(ValueError, match="NOT_REALTIME"):
        Quote(1, 2, AT, AT, kind).validate(AT)
    with pytest.raises(ValueError, match="CROSSED"):
        Quote(2, 1, AT, AT, 1).validate(AT)
    with pytest.raises(ValueError, match="STALE"):
        Quote(1, 2, AT, AT, 1).validate(AT + timedelta(seconds=6))


def test_contract_identity_cutoffs_and_calendar():
    c = Contract(
        conId=100,
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        tradingClass="OG",
        currency="USD",
        multiplier="100",
        right="P",
    )
    d = ContractDetails(
        contract=c,
        underConId=1,
        realExpirationDate="20260928",
        lastTradeTime="16:00:00",
        timeZoneId="America/New_York",
        priceMagnifier=1,
        tradingHours="20260928:0800-20260928:1600",
    )
    expiry, cal = verify_option(d, 1, "GC", mapping(), AT, AT + timedelta(hours=1), "P")
    assert expiry.hour == 20 and cal.state(AT)[0] == "OPEN"
    with pytest.raises(ValueError, match="CUTOFF"):
        verify_option(d, 1, "GC", mapping(), AT, expiry, "P")
    with pytest.raises(ValueError, match="UNDERLYING"):
        verify_option(d, 2, "GC", mapping(), AT, AT + timedelta(hours=1), "P")
    d.realExpirationDate = "20260927"
    with pytest.raises(ValueError, match="NO_REAL_0DTE_MATCH"):
        verify_option(d, 1, "GC", mapping(), AT, AT + timedelta(hours=1), "P")
    maintenance = NS(
        tradingHours="20260928:0000-20260928:1600,1602-20260928:2359", timeZoneId="America/Chicago"
    )
    calendar = Calendar.from_details(maintenance, AT)
    assert calendar.state(AT.replace(hour=21, minute=1))[0] == "MAINTENANCE"
    assert calendar.state(AT + timedelta(days=3))[0] == "BLOCKED"


def test_roll_uses_prior_volume_no_price_splice():
    a = NS(contract=Contract(secType="FUT", conId=1, lastTradeDateOrContractMonth="20261201"))
    b = NS(contract=Contract(secType="FUT", conId=2, lastTradeDateOrContractMonth="20261101"))
    assert (
        select_future(
            [(a, date(2026, 9, 25), 10), (b, date(2026, 9, 25), 20)],
            date(2026, 9, 25),
            date(2026, 9, 28),
        )
        == b
    )
    with pytest.raises(ValueError, match="PRIOR_SESSION"):
        select_future([(a, date(2026, 9, 28), 100)], date(2026, 9, 25), date(2026, 9, 28))
    with pytest.raises(ValueError, match="PRIOR_SESSION"):
        select_future(
            [(a, date(2026, 9, 25), 100), (b, date(2026, 9, 25), float("nan"))],
            date(2026, 9, 25),
            date(2026, 9, 28),
        )
    expired = NS(contract=Contract(secType="FUT", conId=3, lastTradeDateOrContractMonth="20260925"))
    assert expired in nearby_futures([a, b, expired], date(2026, 9, 24))
    assert expired not in nearby_futures([a, b, expired], date(2026, 9, 28))


@pytest.mark.parametrize(
    "product,zone",
    [("BTC", "Europe/London"), ("MBT", "Europe/London"), ("BFF", "America/New_York")],
)
@pytest.mark.parametrize("day,hour,bff_expiry", [("20261026", 13, 20), ("20261102", 14, 21)])
def test_btc_real_product_cutoffs_through_uk_us_dst_gap(product, zone, day, hour, bff_expiry):
    at = datetime.strptime(day, "%Y%m%d").replace(hour=hour, tzinfo=UTC)
    expiry_hour = bff_expiry if product == "BFF" else 16
    product_mapping = mapping(
        product=product,
        symbol=product,
        exchange="CME",
        expiry_timezone=zone,
        termination_time="16:00:00",
    )
    detail = ContractDetails(
        contract=Contract(
            secType="FOP",
            conId=100,
            symbol=product,
            exchange="CME",
            tradingClass="OG",
            currency="USD",
            multiplier="100",
            right="C",
        ),
        underConId=1,
        realExpirationDate=day,
        lastTradeTime=f"{expiry_hour}:00:00",
        timeZoneId="UTC",
        priceMagnifier=1,
        tradingHours=f"{day}:0000-{day}:2359",
    )
    expiry, _ = verify_option(detail, 1, "BTC", product_mapping, at, at + timedelta(hours=1), "C")
    assert expiry.hour == expiry_hour
    with pytest.raises(ValueError, match="UNSUPPORTED_EXIT"):
        verify_option(detail, 1, "BTC", product_mapping, expiry - timedelta(hours=1), expiry, "C")
    with pytest.raises(ValueError, match="NO_REAL_0DTE_MATCH"):
        verify_option(detail, 1, "BTC", product_mapping, expiry, expiry + timedelta(hours=1), "C")


def test_inherited_feature_gate_rejects_flat_last_five_returns():
    prices = [100 + i * 0.01 for i in range(60)]
    references = [{9: {"rv15": 0.01, "range15": 0.01, "volume15": 150}} for _ in range(5)]

    def bars(values):
        return [
            Bar(AT - timedelta(minutes=60 - i), p, p + 0.01, p - 0.01, p, 10, p)
            for i, p in enumerate(values)
        ]

    assert eligibility(bars(prices), AT, date(2026, 12, 1), references)["rv15"] > 0
    prices[-6:] = [prices[-6]] * 6
    assert prior_rv(bars(prices), AT) > 0
    with pytest.raises(ValueError, match="FEATURE_AVAILABILITY"):
        eligibility(bars(prices), AT, date(2026, 12, 1), references)


def test_completed_order_wire_shape_reconciles_without_client_or_order_id(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)
    e = opportunity("GC", 1, AT)
    s.observe(e, "", {})

    async def scenario():
        await b.enter(e, plan())
        sent = ib.trades[0]
        execution = NS(
            orderRef=sent.order.orderRef,
            acctNumber=PAPER_ACCOUNT,
            clientId=83,
            orderId=sent.order.orderId,
            permId=sent.order.permId,
            execId="wire.1",
            shares=1,
            price=0.1,
            side="BOT",
            time=AT,
        )
        ib.executions = [
            NS(execution=execution, contract=sent.contract, commissionReport=NS(execId=""))
        ]
        ib.positions = [NS(account=PAPER_ACCOUNT, contract=sent.contract, position=1)]
        # Actual completedOrder decoder omits orderId/clientId and stores filledQuantity on Order.
        completed = Trade(
            sent.contract,
            Order(
                account=PAPER_ACCOUNT,
                orderRef=sent.order.orderRef,
                permId=sent.order.permId,
                action="BUY",
                totalQuantity=1,
                orderType="LMT",
                lmtPrice=0.1,
                filledQuantity=1,
            ),
            OrderStatus(status="Filled"),
        )
        ib.trades = []

        async def get_completed(apiOnly):
            return [completed]

        ib.reqCompletedOrdersAsync = get_completed
        await b.reconcile()
        assert b.reconciled
        assert s.exposure(e["id"]) == 1
        assert s.capacity()["reserved_open_trades"] == 1

    asyncio.run(scenario())


def test_exit_rechecks_position_change_during_quote(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch, False)
    ref = record_entry(s)
    s.record_fill(fill_record(ref))
    with s.db:
        s.db.execute(
            "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?", (ref,)
        )
        s.db.execute("INSERT INTO positions VALUES(100,1,'{}')")
    later = AT + timedelta(hours=1)
    monkeypatch.setattr("stocker_execution.broker.now", lambda: later)
    b.reconciled_at = later

    async def quote(c):
        b.position(NS(account=PAPER_ACCOUNT, contract=c, position=0))
        return Quote(0.08, 0.1, later, later, 1)

    b.quote = quote
    with pytest.raises(ValueError, match="EXPOSURE_CHANGED"):
        asyncio.run(b.manage_one(s.active()[0]))
    assert not ib.trades


def test_stream_tail_requires_next_bar_and_no_restart_replay(tmp_path, monkeypatch):
    s, b, ib = setup(tmp_path, monkeypatch)
    r = Runtime(b.config, s, b)
    state = r.markets["GC"]
    monkeypatch.setattr("stocker_execution.runtime.now", lambda: AT)
    state.detail = NS(contract=Contract(conId=1, lastTradeDateOrContractMonth="20261201"))
    state.stream = [
        NS(
            date=AT - timedelta(minutes=2 - i),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
            average=100,
        )
        for i in range(2)
    ]
    r.update_bars(state)
    assert state.last_update == AT - timedelta(minutes=1)
    state.stream.append(NS(date=AT, open=100, high=101, low=99, close=100, volume=10, average=100))
    r.update_bars(state)
    assert state.last_update == AT
    state.live_since = AT + timedelta(seconds=1)
    state.last_clock = AT
    asyncio.run(r.decisions())
    assert s.economics()["opportunities"] == 0


def test_gc_experimental_configuration_cannot_change_orders():
    from pydantic import ValidationError

    for key in ("g2", "gc16_exclusion", "volume_veto", "live_enabled"):
        with pytest.raises(ValidationError):
            FuturesConfig(**{key: True})


def test_six_cards_and_retired_runtime_absent(tmp_path):
    runtime = Runtime(
        FuturesConfig(),
        Store(tmp_path / "futures.sqlite"),
        broker=PaperBroker(FuturesConfig(), Store(tmp_path / "other.sqlite"), FakeIB()),
    )

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            d = (await client.get("/api/overview")).json()
            assert [m["market"] for m in d["markets"]] == list(MARKETS)
            assert all(not m["entry_enabled"] for m in d["markets"])
            assert "disabled" in d["markets"][2]["diagnostic"]
            assert (await client.post("/api/entries/pause")).json()["paused"]
            assert not (await client.post("/api/entries/resume")).json()["armed"]
            assert (await client.get("/api/scanner")).status_code == 404

    asyncio.run(check())
    retired = "first" + "4"
    assert importlib.util.find_spec("stocker_execution." + retired) is None
    for folder in ("packages", "apps", "scripts", "configs"):
        for p in Path(folder).rglob("*"):
            if p.suffix in {".py", ".js", ".yaml", ".html", ".md", ".sh"}:
                assert retired not in p.read_text().lower(), p
    bad = tmp_path / "legacy.sqlite"
    with sqlite3.connect(bad) as db:
        db.execute("CREATE TABLE old(x)")
    with pytest.raises(ValueError, match="FRESH_FUTURES_LEDGER"):
        Store(bad)
