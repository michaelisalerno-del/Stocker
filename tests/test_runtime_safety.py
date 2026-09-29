"""Execution state is owned by the runtime; display failures and data gaps stay visible."""

import asyncio
import json
import sqlite3
from datetime import timedelta

import httpx

from saxo_support import AT, FUTURE, GC_MAPPING, quote
from stocker_dashboard.app import create_dashboard_app
from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.rules import Bar
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store


def test_dashboard_read_failure_is_503_and_never_changes_execution_state(tmp_path):
    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        runtime.broker.reconciled = True

        def locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        runtime.store.history = locked  # type: ignore[method-assign]
        app = create_dashboard_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.get("/api/history")
        assert response.status_code == 503
        assert response.json() == {"error": "LEDGER_UNAVAILABLE"}
        assert runtime.broker.fatal_error == ""
        assert runtime.broker.reconciled
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_approved_gc_is_labelled_like_any_approved_market(tmp_path):
    config = FuturesConfig(execution_mode="INTERNAL_PAPER", mappings={"GC": GC_MAPPING})
    runtime = Runtime(config, Store(tmp_path / "ledger.sqlite3"))
    runtime.broker.armed = runtime.broker.reconciled = True
    runtime.broker.problem = ""
    runtime.data.connected = True
    runtime.data.session = {"TradeLevel": "FullTradingAndChat"}
    runtime.markets["GC"].problem = runtime.markets["GC"].history_problem = ""
    overview = {c["market"]: c for c in runtime.overview()["markets"]}
    detail = runtime.market_detail("GC")["markets"][0]
    assert overview["GC"]["strategy_state"] == detail["strategy_state"] == "MONITORING"
    assert overview["GC"]["entry_enabled"] and detail["entry_enabled"]
    # Unapproved GC remains monitor-only on both pages.
    runtime.markets["GC"].problem = "REFERENCE_CONTRACT_SELECTION_REQUIRED"
    overview = {c["market"]: c for c in runtime.overview()["markets"]}
    detail = runtime.market_detail("GC")["markets"][0]
    assert overview["GC"]["strategy_state"] == detail["strategy_state"] == "MONITOR_ONLY"
    runtime.store.db.close()


def test_unexpected_missing_field_is_not_recorded_as_a_trading_reason(tmp_path, monkeypatch):
    import stocker_execution.runtime as module

    monkeypatch.setattr(module, "now", lambda: AT + timedelta(seconds=5))

    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        runtime.started_at = AT - timedelta(seconds=1)
        state = runtime.markets["CL"]
        state.identity = FUTURE
        state.problem = state.history_problem = ""
        state.references = [{9: {"rv15": 0.01, "range15": 0.01, "volume15": 150}} for _ in range(5)]
        state.bars = [
            Bar(AT - timedelta(minutes=60 - i), 70 + i * 0.01, 71 + i * 0.01, 69, 70 + i * 0.01, 10)
            for i in range(60)
        ]
        state.price = quote(70, 71, AT.timestamp() + 5)
        runtime.recorder.register(key(FUTURE), FUTURE)

        async def missing(*args):
            raise KeyError("uic")

        monkeypatch.setattr(runtime.broker, "select_option", missing)
        await runtime.decisions()
        (row,) = runtime.store.history(None, None, None)
        (detail,) = runtime.store.db.execute(
            "SELECT detail FROM signals WHERE id=?", (row["id"],)
        ).fetchone()
        context = json.loads(detail)["option_context"]
        assert context["reason"] == "UNEXPECTED_MISSING_FIELD"
        assert row["reason"] == "EXECUTION_DISABLED"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_fresh_ledger_is_created_with_the_current_reservation_schema(tmp_path):
    store = Store(tmp_path / "fresh.sqlite3")
    (sql,) = store.db.execute("SELECT sql FROM sqlite_master WHERE name='reservations'").fetchone()
    assert "policy_pennies" in sql and "allocation_pennies=1000" not in sql
    assert store.get_meta("allocation_schema") == 2
    assert not store.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='depth_capture_status'"
    ).fetchone()
    assert store.depth_summary("missing")["state"] == "NOT_RECORDED"
    store.db.close()
    reopened = Store(tmp_path / "fresh.sqlite3")  # reopening is a no-op
    assert reopened.db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    reopened.db.close()
