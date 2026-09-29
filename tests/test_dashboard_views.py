"""Dashboard views show the runtime's decisions without over-fetching or stale assets."""

import asyncio
import json
from datetime import timedelta

import httpx

from stocker_dashboard.app import create_dashboard_app
from stocker_execution.config import FuturesConfig
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store
from test_futures_invariants import AT
from test_runtime_safety import GC_MAPPING


def observe(store, market, minute, context=None):
    identity = f"{market}-{minute}"
    event = {
        "id": identity,
        "market": market,
        "rule_version": "fixture",
        "signal_at": (AT + timedelta(minutes=minute)).isoformat(),
        "exit_at": (AT + timedelta(minutes=minute + 60)).isoformat(),
    }
    store.observe(event, "FIXTURE_SKIP", {"large": "x" * 2000})
    if context is not None:
        with store.db:
            row = store.db.execute("SELECT detail FROM signals WHERE id=?", (identity,)).fetchone()
            detail = {**json.loads(row[0]), "option_context": context}
            store.db.execute(
                "UPDATE signals SET detail=? WHERE id=?", (json.dumps(detail), identity)
            )
    return identity


def ready_runtime(tmp_path, market="GC"):
    config = FuturesConfig(execution_mode="INTERNAL_PAPER", mappings={"GC": GC_MAPPING})
    runtime = Runtime(config, Store(tmp_path / "ledger.sqlite3"))
    runtime.broker.armed = runtime.broker.reconciled = True
    runtime.broker.problem = ""
    runtime.data.connected = True
    runtime.data.session = {"TradeLevel": "FullTradingAndChat"}
    runtime.markets[market].problem = runtime.markets[market].history_problem = ""
    return runtime


def test_paused_ready_market_says_paused_on_every_page(tmp_path):
    runtime = ready_runtime(tmp_path)
    runtime.pause = True
    overview = {c["market"]: c for c in runtime.overview()["markets"]}["GC"]
    detail = runtime.market_detail("GC")["markets"][0]
    assert overview["block_reason"] == detail["block_reason"] == "ENTRIES_PAUSED"
    assert not overview["entry_enabled"] and not detail["entry_enabled"]
    runtime.pause = False
    assert runtime.market_detail("GC")["markets"][0]["entry_enabled"]
    runtime.store.db.close()


def test_market_detail_reads_only_its_recent_signals_and_latest_context(tmp_path):
    runtime = ready_runtime(tmp_path)
    for minute in range(12):
        observe(runtime.store, "GC", minute, {"selection_status": f"S{minute}"})
    observe(runtime.store, "CL", 30)
    card = runtime.market_detail("GC")["markets"][0]
    assert [s["id"] for s in card["signals"]] == [f"GC-{m}" for m in range(11, 3, -1)]
    assert card["option_context"]["latest_event"]["context"] == {"selection_status": "S11"}
    runtime.store.db.close()


def test_history_rows_are_summaries_and_evidence_stays_on_demand(tmp_path):
    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        identity = observe(runtime.store, "CL", 1)
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            (row,) = (await client.get("/api/history")).json()["rows"]
            assert set(row) == {
                "id",
                "market",
                "rule_version",
                "signal_at",
                "exit_at",
                "decision",
                "reason",
                "state",
            }
            detail = (await client.get("/api/detail", params={"identity": identity})).json()
            assert detail["inputs"]["inputs"] == {"large": "x" * 2000}
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_page_assets_revalidate_and_arm_rejects_malformed_bodies(tmp_path):
    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            for path in ("/", "/markets", "/static/dashboard.js", "/static/dashboard.css"):
                response = await client.get(path)
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-cache", path
            for body in ("not json", "[]", '"ENABLE PAPER ONLY"'):
                response = await client.post(
                    "/api/paper/arm", content=body, headers={"Content-Type": "application/json"}
                )
                assert response.status_code == 422, body
            assert not runtime.broker.armed
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())
