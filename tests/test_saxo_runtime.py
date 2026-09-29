"""Generated/sanitised Saxo fixtures: no network, broker authentication or orders."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from saxo_support import FUTURE, OPTION, plan, quote, setup, signal
from stocker_dashboard.app import create_dashboard_app
from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.recorder import read_row
from stocker_execution.runtime import Runtime
from stocker_execution.saxo_data import completed_bars
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
            await data.history_range(100, at, at + timedelta(hours=1))
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


def test_old_providers_are_not_network_capable():
    from stocker_data.vendors.eodhd import EODHDClient, EODHDError

    client = EODHDClient()
    with pytest.raises(EODHDError, match="EODHD_INACTIVE"):
        client._client.get("https://eodhd.com/api/eod/fixture")
    client._client.close()
    active = Path("packages/stocker_execution/src/stocker_execution")
    assert not any("ib_async" in p.read_text() for p in active.glob("*.py"))
    assert not any(p.exists() for p in [active / "depth.py", active / "subscriptions.py"])


def test_internal_fill_revalidates_fx_and_rejects_budget_overrun(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path)
        event = signal(1)
        store.observe(event, "", {})
        stale_plan = plan()
        stale_plan["fx_at"] = time.time() - 60
        data.fx = quote(0.19, 0.2)  # real cost now exceeds £50; do not use old cheap FX
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
