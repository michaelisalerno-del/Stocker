"""Generated/sanitised Saxo fixtures: no network, broker authentication or orders."""

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from saxo_support import AT, FUTURE, OPTION, plan, quote, setup, signal
from stocker_dashboard.app import create_dashboard_app
from stocker_execution.config import FuturesConfig, SaxoSettings
from stocker_execution.contracts import key
from stocker_execution.recorder import Recorder, read_row
from stocker_execution.rules import Bar
from stocker_execution.runtime import Runtime
from stocker_execution.saxo_auth import OAuth, SaxoError, atomic_json
from stocker_execution.saxo_client import SaxoClient
from stocker_execution.saxo_data import DataService
from stocker_execution.saxo_history import completed_bars, history_range
from stocker_execution.saxo_stream import merge_board
from stocker_execution.store import Store


def test_depth_updates_and_heartbeats_do_not_refresh_quotes():
    p = quote(at=100)
    p.update({"MarketDepth": {"Bid": [1], "BidSize": [2]}}, "900", 106)
    assert p.receipt == 100 and p.depth(106)["status"] == "L2_AVAILABLE"
    p.update({"Quote": {"Ask": 0.01}}, "4", 107)
    assert p.receipt == 100
    with pytest.raises(ValueError, match="REPLAY_CONTENT_CONFLICT"):
        p.update({"Quote": {"Ask": 99}}, "4", 108)
    assert p.value is None and not p.depth(108)["bids"]


def test_merge_rebuilds_only_updated_groups_and_never_mutates_the_previous_state():
    """Equivalent to the former whole-state deep copy, without copying the state per tick."""
    import copy

    from saxo_support import book
    from stocker_execution.saxo_stream import PriceState

    def reference(previous, update):  # the former implementation, kept as the oracle
        if not isinstance(update, dict):
            return copy.deepcopy(update)
        result = copy.deepcopy(previous) if isinstance(previous, dict) else {}
        for k, v in update.items():
            result[k] = reference(result.get(k), v)
        if isinstance(result.get("MarketDepth"), dict):
            depth = result["MarketDepth"]
            for count, side in (("NoOfBids", "Bid"), ("NoOfOffers", "Ask")):
                if depth.get(count) == 0:
                    for suffix in ("", "Size", "Orders"):
                        depth[side + suffix] = []
        return result

    p, snapshot = PriceState(), book()
    p.snapshot(snapshot, "g", 1.0)
    expected = reference({}, snapshot)
    assert p.value == expected
    updates = [
        {"Quote": {"Ask": 70.02}},
        {"MarketDepth": {"BidSize": [5] * 10}},
        {"PriceInfoDetails": {"Volume": None}},
        {"MarketDepth": {"NoOfBids": 0}},
    ]
    for i, update in enumerate(updates):
        previous, frozen = p.value, copy.deepcopy(p.value)
        assert p.update(update, str(i), 2.0 + i)
        expected = reference(expected, update)
        assert p.value == expected
        assert previous == frozen  # shared groups are never mutated in place
    assert p.value["MarketDepth"]["Bid"] == [] and "Quote.Ask" in p.field_changes


def test_option_board_indexed_patches_and_explicit_nulls():
    original = {
        "Expiries": [
            {
                "Index": 0,
                "Expiry": "2026-09-28",
                "Strikes": [{"Index": 2, "Call": {"Bid": 2}, "Put": {"Bid": 3}}],
            }
        ]
    }
    merged = merge_board(
        original,
        {
            "Expiries": [
                {
                    "Index": 0,
                    "Strikes": [{"Index": 2, "Call": None}, {"Index": 3, "Put": {"Bid": 4}}],
                }
            ]
        },
    )
    rows = merged["Expiries"][0]["Strikes"]
    assert rows[0] == {"Index": 2, "Call": None, "Put": {"Bid": 3}}
    assert rows[1]["Index"] == 3
    assert merge_board(merged, {"Expiries": []})["Expiries"] == []


def test_session_downgrade_reset_and_provider_envelopes(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        data.subscriptions["session"] = {"kind": "SESSION", "target": "SESSION"}
        data.subscriptions["cl"] = {"kind": "PRICE", "target": "CL"}
        data.recorder.ingest(key(FUTURE), "SNAPSHOT", data.markets["CL"].price.value, time.time())
        msg = {
            "reference": "cl",
            "message_id": "900",
            "payload": {"Timestamp": "fixture-provider-time", "Data": {"Quote": {"Ask": 72}}},
        }
        await data.receive(msg)
        row = read_row(data.recorder.windows[key(FUTURE)].rows[-1][1])
        assert row["provider_message"] == msg
        with pytest.raises(ValueError, match="FRESH_SNAPSHOT"):
            await data.receive(
                {"reference": "session", "message_id": "2", "payload": {"TradeLevel": "OrdersOnly"}}
            )
        assert data.markets["CL"].price.value is None
        assert data.problem == "SESSION_DOWNGRADED_EXPLICIT_UPGRADE_REQUIRED"
        assert not data.client.calls
        with pytest.raises(ValueError, match="FRESH_SNAPSHOTS"):
            await data.receive(
                {"reference": "_resetsubscriptions", "message_id": "3", "payload": {}}
            )
        store.db.close()

    asyncio.run(scenario())


def test_real_time_click_takes_primary_session_and_renews_delayed_streams(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        data.session = {"TradeLevel": "OrdersOnly"}
        data.subscriptions = {
            "session": {"kind": "SESSION", "target": "SESSION"},
            "cl": {"kind": "PRICE", "target": "CL"},
            "board": {"kind": "BOARD", "target": "CL"},
            "balance": {"kind": "BALANCE", "target": "BALANCE"},
        }
        await data.take_primary_session()
        assert data.client.calls == [
            (
                "PATCH",
                "/root/v1/sessions/capabilities",
                {"body": {"TradeLevel": "FullTradingAndChat"}, "primary_session": True},
            ),
            ("GET", "/root/v1/sessions/capabilities", {}),
        ]
        assert data.session["TradeLevel"] == "FullTradingAndChat"
        assert data.reset_refs == {"cl", "board"}
        assert data.markets["CL"].price.value is None  # the delayed snapshot is not reused
        # The session stream reporting the same upgrade does not renew twice.
        data.reset_refs.clear()
        await data.receive(
            {
                "reference": "session",
                "message_id": "4",
                "payload": {"TradeLevel": "FullTradingAndChat"},
            }
        )
        assert not data.reset_refs
        data.connected = False
        with pytest.raises(ValueError, match="NOT_CONNECTED"):
            await data.take_primary_session()
        store.db.close()

    asyncio.run(scenario())


def test_history_completed_tail_missing_volume_and_gap_pagination(tmp_path):
    at = datetime(2026, 9, 28, 13, tzinfo=UTC)
    rows = [
        {
            "Time": (at + timedelta(minutes=i)).isoformat(),
            "Open": 70,
            "High": 71,
            "Low": 69,
            "Close": 70.5,
            "Volume": 20,
        }
        for i in range(4)
    ]
    del rows[1]["Volume"]
    bars = completed_bars({"Data": rows}, at + timedelta(hours=1))
    assert [b.at.minute for b in bars] == [0, 2]
    assert bars[0].average is None

    async def scenario():
        _, data, store = setup(tmp_path)

        async def get(*args, **kwargs):
            return {"Data": rows[:1], "DataVersion": 1}

        data.client.request = get
        with pytest.raises(ValueError, match="PAGINATION_DID_NOT_ADVANCE"):
            await history_range(data, 100, at, at + timedelta(hours=1))
        store.db.close()

    asyncio.run(scenario())


def test_internal_paper_never_transmits_and_capacity_releases_after_flat(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path)
        for i in range(2):
            event = signal(i)
            store.observe(event, "", {})
            assert await broker.enter(event, plan()) == ""
        assert not data.client.calls and store.capacity()["reserved_open_trades"] == 2
        broker.armed = False  # disarming entries must not abandon existing management
        with store.db:
            store.db.execute(
                "UPDATE signals SET exit_at=? WHERE id='fixture-0'",
                ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(),),
            )
        await broker.manage()
        assert store.exposure("fixture-0") == 0 and store.exposure("fixture-1") == 1
        assert store.capacity()["reserved_open_trades"] == 1
        assert not data.client.calls
        assert store.economics()["basis"] == "INTERNALLY_SIMULATED"
        store.db.close()

    asyncio.run(scenario())


def test_no_naked_short_futures_or_stale_size_fill(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path)
        with pytest.raises(ValueError, match="SELL_ONLY"):
            broker.validate_order(plan(), "EXIT", "missing")
        invalid = plan()
        invalid["option"] = FUTURE
        with pytest.raises(ValueError, match="ONLY_OWNED_LONG"):
            broker.validate_order(invalid, "ENTRY", "missing")
        event = signal(1)
        store.observe(event, "", {})
        data.options[101][1].size_times["Ask"] = time.time() - 10
        data.options[101][1].update({"PriceInfoDetails": {"BidSize": 99}}, "bid-size", time.time())
        await broker.enter(event, plan())
        assert not store.fills(event["id"])
        await broker.manage()
        assert store.capacity()["reserved_open_trades"] == 0
        store.db.close()

    asyncio.run(scenario())


def test_sim_ambiguous_order_keeps_reservation_and_is_never_retried(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        event = signal(1)
        store.observe(event, "", {})
        await broker.enter(event, plan())
        assert not broker.reconciled
        await broker.reconcile()
        assert broker.problem == "AMBIGUOUS_ORDER_REQUIRES_BROKER_AUDIT"
        assert store.capacity()["reserved_open_trades"] == 1
        assert sum(path == "/trade/v2/orders" for _, path, _ in data.client.calls) == 1
        with pytest.raises(ValueError):
            await broker.enter(event, plan())
        assert sum(path == "/trade/v2/orders" for _, path, _ in data.client.calls) == 1
        store.db.close()

    asyncio.run(scenario())


def test_preflight_is_non_transmitting_unknown_exposure_blocks(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        broker.armed = False
        data.client.positions = [
            {
                "PositionBase": {
                    "AccountId": "fixture-id",
                    "Uic": 999,
                    "AssetType": "ContractFutures",
                    "Amount": 1,
                }
            }
        ]
        result = await broker.preflight()
        assert result["non_transmitting"] and not result["reconciled"]
        assert all(method == "GET" for method, _, _ in data.client.calls)
        with pytest.raises(ValueError, match="PREFLIGHT"):
            broker.arm("ENABLE PAPER ONLY")
        store.db.close()

    asyncio.run(scenario())


def test_dashboard_defaults_do_not_leak_secrets_or_create_subscriptions(tmp_path):
    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            for _ in range(3):
                result = (await client.get("/api/overview")).json()
                assert [m["market"] for m in result["markets"]] == ["CL", "GC", "NG", "NQ", "SI"]
                assert result["system"]["live_orders_disabled"]
                assert not runtime.data.subscriptions
            system = (await client.get("/api/system")).text
            assert "client_secret" not in system and "access_token" not in system
            assert (await client.get("/api/history?market=BTC")).status_code == 422
            assert (
                await client.post("/api/paper/arm", json={"acknowledgement": "ENABLE PAPER ONLY"})
            ).status_code == 409
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_internal_fill_revalidates_fx_and_rejects_budget_overrun(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path)
        event = signal(1)
        store.observe(event, "", {})
        stale_plan = plan()
        stale_plan["fx_at"] = time.time() - 60
        data.fx = quote(0.009, 0.0095)  # real cost now exceeds £1,000; do not use old cheap FX
        await broker.enter(event, stale_plan)
        assert not store.fills(event["id"])
        assert (
            store.history(None, None, None)[0]["reason"] == "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET"
        )
        await broker.manage()
        assert store.capacity()["reserved_open_trades"] == 0
        store.db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["old_indicative", "delayed", "closed_session", "permission"])
def test_paper_fills_require_current_open_session_quote(tmp_path, failure):
    async def scenario():
        broker, data, store = setup(tmp_path)
        option, price = data.options[101]
        option = {**option}
        if failure == "old_indicative":
            price.update({"Quote": {"PriceTypeAsk": "OldIndicative"}}, "old", time.time())
        elif failure == "delayed":
            price.update({"Quote": {"DelayedByMinutes": 10}}, "delayed", time.time())
        elif failure == "closed_session":
            option["trading_sessions"] = {"Sessions": []}
        else:
            option["is_tradable"] = False
        data.options[101] = (option, price)
        event = signal(1)
        store.observe(event, "", {})
        await broker.enter(event, plan())
        assert not store.fills(event["id"])
        assert not data.client.calls
        assert (
            store.history(None, None, None)[0]["reason"]
            == {
                "old_indicative": "QUOTE_NOT_USABLE",
                "delayed": "QUOTE_DELAYED_OR_DELAY_UNKNOWN",
                "closed_session": "OPTION_CURRENT_SESSION_NOT_OPEN_OR_UNVERIFIED",
                "permission": "OPTION_TRADING_PERMISSION_UNVERIFIED",
            }[failure]
        )
        store.db.close()

    asyncio.run(scenario())


def test_real_time_indicative_quote_can_fill_internal_paper(tmp_path):
    """Indicative is Saxo's normal real-time price; fills stay at ask plus one tick."""

    async def scenario():
        broker, data, store = setup(tmp_path)
        _, price = data.options[101]
        price.update(
            {"Quote": {"PriceTypeBid": "Indicative", "PriceTypeAsk": "Indicative"}},
            "indicative",
            time.time(),
        )
        event = signal(1)
        store.observe(event, "", {})
        assert await broker.enter(event, plan()) == ""
        (fill,) = store.fills(event["id"])
        assert fill["side"] == "BOT" and fill["price"] == plan()["limit"]
        assert not data.client.calls
        store.db.close()

    asyncio.run(scenario())


def test_broker_audit_replay_is_idempotent_across_crash_boundary(tmp_path, monkeypatch):
    broker, _, store = setup(tmp_path, "SAXO_SIM")
    event = signal(1)
    store.observe(event, "", {})
    store.reserve(event["id"], plan())
    ref = store.prepare_order(event["id"], "ENTRY", 1, event["exit_at"], {"option": OPTION})
    original = store.orders(event["id"])[0]
    evidence = {
        "OrderId": "fixture",
        "LogId": "one",
        "Status": "FinalFill",
        "SubStatus": "Confirmed",
        "FilledAmount": 1,
        "AveragePrice": 0.01,
        "ActivityTime": event["signal_at"],
    }
    audit = store.audit

    def fail_after_fill(reference, kind, detail):
        if kind == "SAXO_SIM_ORDER_EVIDENCE":
            raise RuntimeError("SIMULATED_CRASH_BEFORE_COMMIT")
        audit(reference, kind, detail)

    monkeypatch.setattr(store, "audit", fail_after_fill)
    with pytest.raises(RuntimeError, match="SIMULATED_CRASH"):
        broker.apply_order_evidence(original, evidence)
    assert not store.fills(event["id"]) and store.orders(event["id"])[0]["filled"] == 0
    monkeypatch.setattr(store, "audit", audit)
    broker.apply_order_evidence(original, evidence)
    # Also recover ledgers interrupted at the old commit boundary.
    with store.db:
        store.db.execute("UPDATE orders SET filled=0,status='Submitted' WHERE reference=?", (ref,))
    broker.apply_order_evidence(original, evidence)
    assert len(store.fills(event["id"])) == 1 and store.orders(event["id"])[0]["filled"] == 1
    store.db.close()


def test_pending_snapshot_keeps_original_message_receipt(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        arrived = time.time() - 20
        message = {
            "reference": "early",
            "message_id": "opaque",
            "payload": {"Quote": {"Bid": 70, "Ask": 71}},
        }
        data.pending["early"] = []
        await data.receive(message, arrived)
        pending = data.pending.pop("early")
        data.subscriptions["early"] = {"kind": "PRICE", "target": "CL"}
        for queued in pending:
            await data.receive(queued["message"], queued["receipt"])
        assert data.markets["CL"].price.receipt == arrived
        recorded = read_row(data.recorder.windows[key(FUTURE)].rows[-1][1])
        assert recorded["receipt"] == arrived
        assert data.markets["CL"].price.depth(time.time())["status"] != "L2_AVAILABLE"
        store.db.close()

    asyncio.run(scenario())


def test_quote_refreshed_during_an_earlier_market_await_is_not_stale(tmp_path, monkeypatch):
    """Gates use the time at their own clock, not the time decisions() started."""
    import stocker_execution.runtime as module

    clock = [AT + timedelta(seconds=5)]
    monkeypatch.setattr(module, "now", lambda: clock[0])

    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        runtime.started_at = AT - timedelta(seconds=1)
        for market, uic in (("CL", 100), ("GC", 200)):
            state = runtime.markets[market]
            state.identity = {**FUTURE, "market": market, "uic": uic}
            state.problem = state.history_problem = ""
            state.references = [
                {9: {"rv15": 0.01, "range15": 0.01, "volume15": 150}} for _ in range(5)
            ]
            state.bars = [
                Bar(
                    AT - timedelta(minutes=60 - i),
                    70 + i * 0.01,
                    71 + i * 0.01,
                    69,
                    70 + i * 0.01,
                    10,
                )
                for i in range(60)
            ]
            state.price = quote(70, 71, clock[0].timestamp())
            runtime.recorder.register(key(state.identity), state.identity)

        async def select(event, state, inputs):
            # CL's broker I/O takes two seconds; GC's quote is refreshed meanwhile.
            clock[0] += timedelta(seconds=2)
            runtime.markets["GC"].price = quote(1900, 1901, clock[0].timestamp())
            raise KeyError("uic")

        monkeypatch.setattr(runtime.broker, "select_option", select)
        await runtime.decisions()
        reasons = {r["market"]: r["reason"] for r in runtime.store.history(None, None, None)}
        assert reasons == {"CL": "EXECUTION_DISABLED", "GC": "EXECUTION_DISABLED"}
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_preflight_waits_for_the_management_lock(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        async with broker.lock:
            task = asyncio.create_task(broker.preflight())
            await asyncio.sleep(0.05)
            assert not task.done() and not data.client.calls
        result = await task
        assert result["reconciled"] and data.client.calls
        store.db.close()

    asyncio.run(scenario())


def test_stream_loop_survives_a_lost_token(tmp_path):
    """Cleanup after a failed refresh must not raise out of run(): the runtime would stop."""

    async def scenario():
        credentials = tmp_path / "credentials.json"
        atomic_json(
            credentials,
            {
                "environment": "SAXO_SIM",
                "client_id": "fixture-key",
                "client_secret": "fixture-secret",
                "account_key": "fixture-account",
            },
        )
        transport = httpx.MockTransport(lambda request: httpx.Response(500))
        oauth = OAuth("SAXO_SIM", SaxoSettings(credentials_file=credentials), tmp_path, transport)
        config = FuturesConfig()
        client = SaxoClient(oauth, transport=transport)
        data = DataService(config, client, Recorder(config.recorder, tmp_path / "events"))
        data.subscriptions["s1"] = {
            "kind": "PRICE",
            "target": "CL",
            "path": "/trade/v1/prices/subscriptions",
        }
        task = asyncio.create_task(data.run())
        await asyncio.sleep(0.1)
        assert not task.done()
        assert data.problem == "RECONNECT_REQUIRED" and not data.subscriptions
        data.stopping = True
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await client.close()

    asyncio.run(scenario())


def test_token_renewal_reauthorises_the_open_stream(tmp_path, monkeypatch):
    """Saxo binds a renewed token to the open context; only a refusal reconnects."""

    class Socket:
        async def recv(self):
            await asyncio.sleep(0.01)
            return b""

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Socket()

    monkeypatch.setattr("stocker_execution.saxo_data.connect", connect)

    async def scenario():
        _, data, store = setup(tmp_path)
        data.recorder.register(key(OPTION), OPTION)
        data.client.oauth = SimpleNamespace(
            account_key="fixture-account",
            generation=1,
            urls={"stream": "wss://fixture"},
            access_token=AsyncMock(return_value="fixture-token"),
        )
        data.client.authorize_stream = AsyncMock()
        data.startup = AsyncMock()
        data.subscriptions = {
            "s1": {
                "kind": "PRICE",
                "target": "CL",
                "path": "/trade/v1/prices/subscriptions",
                "arguments": {},
                "contact": time.monotonic(),
                "timeout": 30,
            }
        }
        task = asyncio.create_task(data.run())
        await asyncio.sleep(0.05)
        assert data.connected and data.reconnects == 1
        data.client.oauth.generation = 2
        await asyncio.sleep(0.05)
        data.client.authorize_stream.assert_awaited_once_with(data.context)
        assert data.connected and data.reconnects == 1 and "s1" in data.subscriptions
        data.client.authorize_stream.side_effect = SaxoError("STREAM_REAUTHORISATION_HTTP_401")
        data.client.oauth.generation = 3
        await asyncio.sleep(0.05)
        assert not data.connected and data.problem == "STREAM_REAUTHORISATION_HTTP_401"
        assert not data.subscriptions
        data.stopping = True
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        store.db.close()

    asyncio.run(scenario())


def test_one_paused_or_reset_subscription_does_not_reset_the_others(tmp_path):
    """Saxo's per-subscription signals stay per-subscription; the socket is kept."""

    async def scenario():
        _, data, store = setup(tmp_path)
        gold = {**FUTURE, "market": "GC", "uic": 200}
        data.markets["GC"].identity = gold
        data.markets["GC"].price = quote(1900, 1901)
        data.recorder.register(key(gold), gold)
        for ref, market in (("cl", "CL"), ("gc", "GC")):
            data.subscriptions[ref] = {
                "kind": "PRICE",
                "target": market,
                "path": "/trade/v1/prices/subscriptions",
                "arguments": {"Uic": data.markets[market].identity["uic"]},
                "contact": time.monotonic(),
                "timeout": 30,
            }
        heartbeat = {"OriginatingReferenceId": "cl", "Reason": "SubscriptionTemporarilyDisabled"}
        await data.receive(
            {"reference": "_heartbeat", "message_id": "1", "payload": {"Heartbeats": [heartbeat]}}
        )
        assert data.markets["CL"].price.problem == "SUBSCRIPTION_TEMPORARILY_DISABLED"
        assert data.markets["CL"].price.value is not None and not data.reset_refs
        await data.receive(
            {"reference": "cl", "message_id": "2", "payload": {"Data": {"Quote": {"Ask": 72}}}}
        )
        assert data.markets["CL"].price.problem == ""
        await data.receive(
            {
                "reference": "_resetsubscriptions",
                "message_id": "3",
                "payload": {"TargetReferenceIds": ["gc", "obsolete"]},
            }
        )
        assert data.reset_refs == {"gc"}
        assert data.markets["GC"].price.value is None
        assert data.markets["CL"].price.value is not None
        data.subscribe = AsyncMock()
        await data.replace_reset_subscriptions()
        data.subscribe.assert_awaited_once_with("PRICE", {"Uic": 200}, "GC", old="gc")
        assert not data.reset_refs
        store.db.close()

    asyncio.run(scenario())


def test_subscription_posted_across_a_reconnect_is_not_registered(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        data.subscriptions.clear()
        data.context = "old"
        calls = []

        async def request(method, path, **kwargs):
            calls.append((method, path))
            data.context = "new"  # the socket reconnected while this POST was in flight
            return {"Snapshot": {}}

        data.client.request = request
        with pytest.raises(SaxoError, match="SUBSCRIPTION_CONTEXT_REPLACED"):
            await data.subscribe("SESSION", {}, "SESSION")
        assert [c[0] for c in calls] == ["POST", "DELETE"] and "/old/" in calls[1][1]
        assert not any(s["kind"] == "SESSION" for s in data.subscriptions.values())
        assert not data.pending
        store.db.close()

    asyncio.run(scenario())


def test_fx_instrument_ambiguity_is_named_in_gates_not_silent(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        pair = {"Symbol": "GBPUSD", "AssetType": "FxSpot"}
        data.client.request = AsyncMock(
            return_value={"Data": [{**pair, "Identifier": 1}, {**pair, "Identifier": 2}]}
        )
        await data.subscribe_fx()
        assert data.fx.problem == "GBPUSD_FX_INSTRUMENT_NOT_UNIQUE" and data.fx.value is None
        store.db.close()
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "gates.sqlite3"))
        runtime.data.fx.gap("GBPUSD_FX_INSTRUMENT_NOT_UNIQUE")
        gate = next(g for g in runtime.overview()["markets"][0]["gates"] if g["key"] == "fx")
        assert gate == {
            "key": "fx",
            "label": "GBP/USD rate",
            "ok": False,
            "detail": "GBPUSD_FX_INSTRUMENT_NOT_UNIQUE",
        }
        step = next(i for i in runtime.status()["setup"] if i["key"] == "fx")
        assert not step["done"] and step["detail"] == "GBPUSD_FX_INSTRUMENT_NOT_UNIQUE"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_internal_fill_is_labelled_internal_and_reconciliation_opens_the_trade(tmp_path):
    async def scenario():
        broker, _, store = setup(tmp_path)
        event = signal(1)
        store.observe(event, "", {})
        assert await broker.enter(event, plan()) == ""
        (row,) = store.active()
        assert row["state"] == "EXPOSURE_REQUIRES_RECONCILIATION"
        decision = store.db.execute(
            "SELECT decision FROM signals WHERE id=?", (event["id"],)
        ).fetchone()[0]
        assert decision == "INTERNALLY_SIMULATED_FILL"
        await broker.reconcile()
        assert broker.reconciled and store.active()[0]["state"] == "OPEN"
        store.db.close()

    asyncio.run(scenario())
