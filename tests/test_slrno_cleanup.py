"""Offline regression fixtures for allocation, clock coordination and display work."""

import asyncio
import json
from datetime import timedelta

import pytest

from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import budget, key
from stocker_execution.rules import Bar
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store
from test_futures_invariants import AT
from test_saxo_runtime import FUTURE, plan, quote, setup, signal


def test_new_allocation_is_fifty_all_in_and_four_unique_slots(tmp_path):
    broker, _, store = setup(tmp_path)
    for i in range(4):
        event = signal(i)
        store.observe(event, "", {})
        p = {**plan(), "cash_pennies": 5000}
        broker.validate_order(p, "ENTRY", event["id"])
        assert store.reserve(event["id"], p) == ""
        assert store.reserve(event["id"], p) == "DUPLICATE_OPPORTUNITY"
    assert store.capacity() == {"reserved_open_trades": 4, "allocation_pennies": 20000}
    fifth = signal(5)
    store.observe(fifth, "", {})
    assert store.reserve(fifth["id"], plan()) == "SKIP_CAPACITY_FULL"
    with pytest.raises(ValueError, match="EXCEEDS_BUDGET"):
        broker.validate_order({**plan(), "cash_pennies": 5001}, "ENTRY", fifth["id"])
    assert budget(plan()["option"], 0.049, 1, 1)["cash_pennies"] == 5000
    with pytest.raises(ValueError, match="EXCEEDS_BUDGET"):
        budget(plan()["option"], 0.04901, 1, 1)
    assert budget(plan()["option"], 0.009, 1, 1)["quantity"] == 1
    store.db.close()


@pytest.mark.parametrize("arrival", [10, 20, 21])
def test_boundary_bar_waits_only_inside_original_deadline(tmp_path, monkeypatch, arrival):
    import stocker_execution.runtime as module

    clock = [AT]
    monkeypatch.setattr(module, "now", lambda: clock[0])

    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite"))
        runtime.started_at = AT - timedelta(seconds=1)
        state = runtime.markets["CL"]
        state.identity = FUTURE
        state.problem = state.history_problem = ""
        state.references = [{9: {"rv15": 0.01, "range15": 0.01, "volume15": 150}} for _ in range(5)]
        bars = [
            Bar(
                AT - timedelta(minutes=60 - i),
                70 + i * 0.01,
                71 + i * 0.01,
                69 + i * 0.01,
                70 + i * 0.01,
                10,
            )
            for i in range(60)
        ]
        state.bars = bars[:-1]
        state.price = quote(70, 71, AT.timestamp())
        runtime.recorder.register(key(FUTURE), FUTURE)
        await runtime.decisions()
        assert not runtime.store.history(None, None, None)
        assert state.last_clock is None
        clock[0] = AT + timedelta(seconds=arrival)
        state.bars = bars + [Bar(AT, 999, 999, 999, 999, 999)]
        state.price = quote(70, 71, clock[0].timestamp())
        await runtime.decisions()
        await runtime.decisions()
        rows = runtime.store.history(None, None, None)
        assert len(rows) == 1
        assert rows[0]["reason"] == (
            "EXECUTION_DISABLED" if arrival < 20 else "STALE_SIGNAL_NO_REPLAY"
        )
        assert rows[0]["exit_at"] == (AT + timedelta(hours=1)).isoformat()
        assert len(runtime.recorder.event_ids) == 1
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_history_oldest_first_sorts_before_pagination(tmp_path):
    store = Store(tmp_path / "history.sqlite")
    for i in range(205):
        e = signal(i)
        e["signal_at"] = (AT + timedelta(minutes=i)).isoformat()
        store.observe(e, "fixture", {})
    first = store.history(None, None, None, sort="asc")
    second = store.history(None, None, None, offset=100, sort="asc")
    assert first[0]["id"] == "fixture-0"
    assert second[0]["id"] == "fixture-100"
    assert store.history(None, None, None)[0]["id"] == "fixture-204"
    store.db.close()


def test_migration_keeps_old_allocation_policy_orders_and_fills(tmp_path):
    import sqlite3

    from test_futures_invariants import fill_record, record_entry

    path = tmp_path / "old.sqlite"
    store = Store(path)
    ref = record_entry(store)
    store.record_fill(fill_record(ref))
    # Reconstruct the deployed six-column reservation schema, preserving its dependent rows.
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.executescript("""
      BEGIN IMMEDIATE;
      CREATE TABLE old_reservations (id TEXT PRIMARY KEY REFERENCES signals(id),
        allocation_pennies INTEGER NOT NULL CHECK(allocation_pennies=1000),
        active INTEGER NOT NULL CHECK(active IN (0,1)),state TEXT NOT NULL,
        plan TEXT NOT NULL,created_at TEXT NOT NULL);
      INSERT INTO old_reservations SELECT id,1000,active,state,plan,created_at FROM reservations;
      DROP TABLE reservations;
      ALTER TABLE old_reservations RENAME TO reservations;
      COMMIT;
    """)
    old = dict(store.db.execute("SELECT * FROM reservations").fetchone())
    store.db.close()
    migrated = Store(path)
    row = dict(migrated.db.execute("SELECT * FROM reservations").fetchone())
    assert row.pop("policy_pennies") == 1000
    assert row == old
    assert len(migrated.orders("x")) == len(migrated.fills("x")) == 1
    assert not list(migrated.db.execute("PRAGMA foreign_key_check"))
    assert migrated.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert migrated.capacity()["allocation_pennies"] == 1000
    with pytest.raises(sqlite3.IntegrityError), migrated.db:
        migrated.db.execute("UPDATE reservations SET allocation_pennies=5000 WHERE id='x'")
    event = signal("new")
    migrated.observe(event, "", {})
    assert migrated.reserve(event["id"], {**plan(), "cash_pennies": 5000}) == ""
    assert migrated.capacity() == {"reserved_open_trades": 2, "allocation_pennies": 6000}
    migrated.db.close()
    reopened = Store(path)
    assert reopened.capacity()["allocation_pennies"] == 6000
    reopened.db.close()


@pytest.mark.parametrize("environment", ["SAXO_SIM", "SAXO_LIVE"])
def test_balance_subscription_is_shared_scoped_and_never_arms(tmp_path, environment):
    import time
    from types import SimpleNamespace

    from stocker_execution.recorder import Recorder
    from stocker_execution.saxo_client import allowed
    from stocker_execution.saxo_data import DataService

    async def scenario():
        calls = []

        async def request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            return {
                "Snapshot": {
                    "TotalValue": 123.4,
                    "CashBalance": 0,
                    "Currency": "GBP",
                    "CalculationReliability": "Ok",
                },
                "InactivityTimeout": 30,
            }

        config = FuturesConfig(data_environment=environment)
        client = SimpleNamespace(request=request, oauth=SimpleNamespace(account_key="selected-key"))
        data = DataService(config, client, Recorder(config.recorder, tmp_path / environment))
        data.account_verified = data.connected = True
        data.account_id = "private-account-1234"
        data.account_currency = "GBP"
        assert data.balance_view(time.time())["status"] == "Unavailable"
        await asyncio.gather(*(data.ensure_balance_subscription() for _ in range(4)))
        assert len(calls) == 1
        assert calls[0][2]["body"]["Arguments"]["AccountKey"] == "selected-key"
        view = data.balance_view(time.time())
        assert view["status"] == "Current" and view["cash_balance"] == 0
        assert view["cash_available_for_trading"] is None
        assert view["environment"] == environment.removeprefix("SAXO_")
        assert ("Real-money balances are not connected" in view["connection_note"]) == (
            environment == "SAXO_SIM"
        )
        assert view["account"] == "••••1234" and "selected-key" not in json.dumps(view)
        ref = next(iter(data.subscriptions))
        await data.receive({"reference": ref, "message_id": "1", "payload": {"CashBalance": None}})
        assert data.balance_view(time.time())["cash_balance"] is None
        assert data.balance_view(time.time())["total_value"] == 123.4
        updated = data.balance_received_at
        data.receive_balance({"TotalValue": 999, "CalculationReliability": "Approximated"}, 999)
        assert data.balance_view(time.time())["status"] == "Stale"
        assert data.balance_view(time.time())["total_value"] == 123.4
        assert data.balance_received_at == updated
        data.receive_balance({"Currency": "USD", "TotalValue": 999}, 1000)
        assert data.balance_view(time.time())["currency"] == "GBP"
        data.receive_balance({"CalculationReliability": "Ok"}, updated)
        await data.receive(
            {
                "reference": "_heartbeat",
                "message_id": "2",
                "payload": [
                    {"Heartbeats": [{"OriginatingReferenceId": ref, "Reason": "NoNewData"}]}
                ],
            }
        )
        assert data.balance_received_at == updated
        data.connected = False
        assert data.balance_view(time.time())["status"] == "Stale"
        assert data.balance_view(time.time())["last_success_at"] == updated
        client.oauth.account_key = "another-account"
        assert data.balance_view(time.time())["total_value"] is None
        assert config.execution_mode == "DISABLED" and not config.armed
        assert allowed("POST", "/port/v1/balances/subscriptions")
        assert not allowed("POST", "/trade/v2/orders")

    asyncio.run(scenario())


def test_derived_flow_matches_raw_replay_and_accounting_is_cached(tmp_path, monkeypatch):
    from stocker_execution.book_flow import observe
    from stocker_execution.config import RecorderConfig
    from stocker_execution.recorder import Recorder, apply, packed, read_row
    from test_book_flow import IDENTITY, book

    recorder = Recorder(RecorderConfig(), tmp_path)
    recorder.register("future", IDENTITY)
    history = []
    for i in range(1000):
        value = book(bid_size=1 + i % 3)
        recorder.ingest("future", "SNAPSHOT" if i == 0 else "UPDATE", value, i, generation="one")
        point = observe(
            IDENTITY, value, i, {"valid_until": i + 5, "subscription_id": "one"}, history
        )
        assert recorder.windows["future"].flow == point
        history.append(point)
        history = history[-512:]
    w = recorder.windows["future"]
    assert len(w.flow_history) <= 62
    for name, size in w.state_bytes.items():
        assert size == 8 * len(packed(getattr(w, name)))
    rebuilt = w.checkpoint
    for _, blob in w.rows:
        rebuilt = apply(rebuilt, read_row(blob))
    assert rebuilt == w.current

    # Pure reads may not serialise or decompress retained state.
    def forbidden(*args, **kwargs):
        raise AssertionError("repeated encoding on read")

    monkeypatch.setattr("stocker_execution.recorder.packed", forbidden)
    monkeypatch.setattr("stocker_execution.recorder.read_row", forbidden)
    for _ in range(20):
        assert recorder.memory() < recorder.config.rolling_max_bytes
        recorder.book_flow_view("future", 999)


def test_four_hour_rotation_does_not_pin_all_old_candidates(tmp_path, monkeypatch):
    import gzip

    from stocker_execution.config import RecorderConfig
    from stocker_execution.recorder import Recorder
    from stocker_execution.saxo_stream import PriceState
    from test_saxo_runtime import OPTION

    async def scenario():
        _, data, store = setup(tmp_path)
        data.options.clear()
        data.subscriptions.clear()
        recorder = Recorder(
            RecorderConfig(
                persistent_capture=True, recording_permission_evidence="OFFLINE GENERATED FIXTURE"
            ),
            tmp_path / "capture",
        )
        data.recorder = recorder
        await recorder.start()
        recorder.register(key(FUTURE), FUTURE)
        recorder.ingest(key(FUTURE), "SNAPSHOT", {"Quote": {"Bid": 70}}, 0)

        async def request(*args, **kwargs):
            return {}

        data.client.request = request
        clock = [0]
        monkeypatch.setattr("stocker_execution.saxo_data.time.time", lambda: clock[0])
        segment = None
        for hour in range(4):
            clock[0] = hour * 3600
            uic = 200 + hour
            option = {**OPTION, "uic": uic}
            instrument = key(option)
            recorder.register(instrument, option)
            recorder.ingest(instrument, "SNAPSHOT", {"Quote": {"Bid": 0.01}}, clock[0])
            data.options[uic] = (option, PriceState())
            data.option_required_at[uic] = clock[0]
            data.subscriptions[str(uic)] = {
                "target": str(uic),
                "path": "/trade/v1/prices/subscriptions",
            }
            result = recorder.trigger(key(FUTURE), {"id": str(hour)}, clock[0], [instrument])
            assert segment in (None, result["segment"])
            segment = result["segment"]
            # Previous event's entire one-hour obligation is retained through its endpoint.
            await data.release_unused_options(set())
            if hour:
                assert 199 + hour in data.options
            clock[0] += 1
            await data.release_unused_options(set())
            assert set(data.options) == {uic}
            assert len(data.subscriptions) == 1
            await recorder.queue.join()
        clock[0] = 4 * 3600 + 1
        recorder.tick(clock[0])
        await data.release_unused_options(set())
        assert not data.options
        await recorder.queue.join()
        manifest = json.loads((recorder.directory / (segment + ".manifest.json")).read_text())
        assert manifest["state"] == "COMPLETE"
        assert len(manifest["instruments"]) == 5 and len(manifest["events"]) == 4
        rows = [
            json.loads(x)
            for x in gzip.decompress(
                (recorder.directory / (segment + ".jsonl.gz")).read_bytes()
            ).splitlines()
        ]
        assert {r.get("identity", {}).get("uic") for r in rows} >= {200, 201, 202, 203}
        assert manifest["gaps"] == 0 and recorder.gaps == 4 and not recorder.problem
        await recorder.close()
        store.db.close()

    asyncio.run(scenario())


def test_boundary_refresh_runs_before_routine_history_and_never_retries_expired(
    tmp_path, monkeypatch
):
    import stocker_execution.runtime as module

    clock = [AT + timedelta(seconds=1)]
    monkeypatch.setattr(module, "now", lambda: clock[0])

    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "priority.sqlite"))
        runtime.data.connected = True
        state = runtime.markets["SI"]
        state.boundary_clock = AT
        calls = []

        async def history(state, *, boundary=False):
            calls.append((state.market, boundary))

        async def warm(state):
            pass

        monkeypatch.setattr(runtime.data, "history", history)
        monkeypatch.setattr(runtime.data, "warm_candidates", warm)
        await runtime.refresh_histories()
        assert calls[0] == ("SI", True)
        calls.clear()
        await runtime.refresh_histories()
        assert ("SI", True) not in calls
        clock[0] = AT + timedelta(seconds=20)
        state.boundary_checked = float("-inf")
        await runtime.refresh_histories()
        assert ("SI", True) not in calls
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_awaited_precheck_cannot_extend_entry_deadline_or_ignore_pause(tmp_path, monkeypatch):
    import stocker_execution.broker as module

    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        event = signal("late")
        at = module.utc(event["signal_at"])
        clock = [at + timedelta(seconds=18)]
        monkeypatch.setattr(module, "now", lambda: clock[0])
        store.observe(event, "", {})
        requests = []

        async def request(method, path, **kwargs):
            requests.append(path)
            clock[0] = at + timedelta(seconds=21)
            return {"PreCheckResult": "Ok"}

        monkeypatch.setattr(data.client, "request", request)
        monkeypatch.setattr(broker, "check_broker_cost", lambda *args: None)
        monkeypatch.setattr(module, "executable_quote", lambda *args: {})
        await broker.enter(event, plan())
        assert requests == ["/trade/v2/orders/precheck"]
        order = store.orders(event["id"])[0]
        assert order["deadline"] == (at + timedelta(seconds=20)).isoformat()
        assert order["status"] == "Inactive"
        store.set_meta("paused", True)
        with pytest.raises(ValueError, match="ENTRIES_PAUSED"):
            broker.validate_order(plan(), "ENTRY", event["id"])
        store.db.close()

    asyncio.run(scenario())


def test_page_scopes_and_display_cache_never_drive_admission(tmp_path, monkeypatch):
    import httpx

    from stocker_dashboard.app import create_dashboard_app

    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "views.sqlite"))

        def forbidden(*args, **kwargs):
            raise AssertionError("full overview unexpectedly built")

        monkeypatch.setattr(runtime, "overview", forbidden)
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.get(path)
                    for path in (
                        "/api/history?sort=asc",
                        "/api/execution",
                        "/api/system",
                        "/api/market/NG",
                    )
                )
            )
            assert all(r.status_code == 200 for r in responses)
            assert [m["market"] for m in responses[-1].json()["markets"]] == ["NG"]
            assert responses[-1].json()["markets"][0]["details"] is None
            assert (await client.get("/api/history?sort=sql")).status_code == 422
            assert runtime.data.client.calls == 0
        initial = runtime.store.economics()
        assert runtime.store.economics() is initial
        event = signal("cache")
        runtime.store.observe(event, "fixture", {})
        assert runtime.store.economics()["opportunities"] == 1
        assert runtime.store.reserve(event["id"], plan()) == ""
        assert runtime.store.capacity()["allocation_pennies"] == 5000
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_legacy_entry_policy_remains_ten_at_broker_boundary(tmp_path):
    broker, _, store = setup(tmp_path)
    event = signal("legacy")
    store.observe(event, "", {})
    assert store.reserve(event["id"], plan()) == ""
    with store.db:
        store.db.execute(
            "UPDATE reservations SET allocation_pennies=1000,policy_pennies=1000 WHERE id=?",
            (event["id"],),
        )
    broker.validate_order({**plan(), "cash_pennies": 1000}, "ENTRY", event["id"])
    with pytest.raises(ValueError, match="EXCEEDS_BUDGET"):
        broker.validate_order({**plan(), "cash_pennies": 1001}, "ENTRY", event["id"])
    store.db.close()


def test_recording_burst_wakes_existing_writer_before_batch_delay(tmp_path):
    from stocker_execution.config import RecorderConfig
    from stocker_execution.recorder import Recorder

    async def scenario():
        recorder = Recorder(
            RecorderConfig(persistent_capture=True, recording_permission_evidence="OFFLINE TEST"),
            tmp_path,
        )
        written = []
        recorder.write_batch = lambda items: written.extend(items)
        for i in range(64):
            assert recorder.enqueue("fixture", {"sequence": i}, [b'{"fixture":true}\n'])
        assert recorder.write_ready.is_set()
        recorder.closed = True
        await recorder.write_loop()
        assert len(written) == 64
        assert recorder.queued_bytes == 0 and recorder.queue.empty() and not recorder.problem

    asyncio.run(scenario())
