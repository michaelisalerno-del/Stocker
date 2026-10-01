"""Execution state is owned by the runtime; display failures and data gaps stay visible."""

import asyncio
import json
import sqlite3
import time
from datetime import timedelta

import httpx
import pytest

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
    # Unapproved GC is blocked on both pages, like any unapproved market.
    runtime.markets["GC"].problem = "REFERENCE_CONTRACT_SELECTION_REQUIRED"
    overview = {c["market"]: c for c in runtime.overview()["markets"]}
    detail = runtime.market_detail("GC")["markets"][0]
    assert overview["GC"]["strategy_state"] == detail["strategy_state"] == "BLOCKED"
    runtime.store.db.close()


def test_gc_warms_candidates_like_any_market(tmp_path):
    async def scenario():
        runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
        state = runtime.markets["GC"]
        state.identity = {**FUTURE, "market": "GC"}
        # Stale bars stop warming inside the ranking step, which GC used to skip entirely.
        state.bars = [Bar(AT - timedelta(minutes=5), 70, 71, 69, 70, 10)]
        await runtime.data.warm_candidates(state)
        assert state.candidate_problem == "CANDIDATE_UNDERLYING_HISTORY_STALE"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


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
        # The chain as seen at the clock is recorded with every observed opportunity.
        assert "option_chain" in json.loads(detail)
        assert row["reason"] == "EXECUTION_DISABLED"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_every_clock_records_the_book_and_volatility_observations(tmp_path, monkeypatch):
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
            Bar(AT - timedelta(minutes=70 - i), 70 + i * 0.01, 71 + i * 0.01, 69, 70 + i * 0.01, 10)
            for i in range(70)
        ]
        state.price = quote(70, 71, AT.timestamp() + 5)
        runtime.recorder.register(key(FUTURE), FUTURE)
        await runtime.decisions()
        (row,) = runtime.store.history(None, None, None)
        (detail,) = runtime.store.db.execute(
            "SELECT detail FROM signals WHERE id=?", (row["id"],)
        ).fetchone()
        detail = json.loads(detail)
        # No depth has arrived, so the book is honestly unavailable, never zero-filled.
        assert detail["book_flow"]["status"] == "UNAVAILABLE"
        seen = detail["observation"]
        assert seen["rv60"] > 0
        assert seen["hour_reference_rv15_median"] == 0.01
        # The look14 forecast is recorded too; no option was selected, so no price check.
        assert detail["forecast"]["status"] == "OBSERVED" and "option" not in detail["forecast"]
        # Recording is observation only: the entry decision is unchanged.
        assert row["reason"] == "EXECUTION_DISABLED"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_observation_never_bridges_a_gap_and_never_raises():
    from stocker_execution.rules import observation

    bars = [
        Bar(AT - timedelta(minutes=70 - i), 70, 71, 69, 70 + (i % 2) * 0.1, 10) for i in range(70)
    ]
    del bars[40]  # one missing minute inside the last hour
    seen = observation(bars, AT, [])
    assert seen["rv60"] is None
    assert seen["hour_reference_rv15_median"] is None
    # 08:00-08:59 New York less 08:30: 59 bars, 58 neighbours, one spanning the gap.
    assert seen["session_minutes_counted"] == 57


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


def test_option_session_gate_uses_saxo_session_states():
    """Saxo documents AutomatedTrading, not "Open"; every other state stays blocked."""
    from datetime import UTC, datetime, timedelta

    import pytest

    from saxo_support import OPTION, quote
    from stocker_execution.contracts import executable_quote, verified_cutoff

    price = quote()
    now = datetime.now(UTC)

    def option(state):
        session = {
            "StartTime": (now - timedelta(hours=1)).isoformat(),
            "EndTime": (now + timedelta(hours=3)).isoformat(),
            "State": state,
        }
        return {
            **OPTION,
            "trading_sessions": {"Sessions": [session]},
            "expiry_instant": (now + timedelta(hours=3)).isoformat(),
            "last_trade_at": (now + timedelta(hours=3)).isoformat(),
            "expiry": (now + timedelta(hours=3)).date().isoformat(),
        }

    assert executable_quote(option("AutomatedTrading"), price.value, price.receipt, now)
    assert verified_cutoff(option("AutomatedTrading"), now + timedelta(minutes=60))
    for state in ("Open", "Closed", "Break", "Halt", "OpeningAuction", "PreTrading", "Undefined"):
        with pytest.raises(ValueError, match="SESSION_NOT_OPEN"):
            executable_quote(option(state), price.value, price.receipt, now)
        with pytest.raises(ValueError, match="EXIT_SESSION_UNVERIFIED"):
            verified_cutoff(option(state), now + timedelta(minutes=60))


def test_decision_worker_error_stays_until_restart_and_blocks_arming(tmp_path, monkeypatch):
    async def scenario():
        config = FuturesConfig(execution_mode="INTERNAL_PAPER")
        runtime = Runtime(config, Store(tmp_path / "ledger.sqlite3"))
        runtime.broker.armed = True

        async def broken():
            raise RuntimeError("fixture")

        monkeypatch.setattr(runtime, "decisions", broken)
        await runtime.decision_pass()
        assert not runtime.broker.armed
        assert runtime.broker.entry_reason() == "DECISION_WORKER_ERROR_REVIEW_REQUIRED"
        await runtime.broker.reconcile()  # a routine pass must not clear it
        assert runtime.broker.entry_reason() == "DECISION_WORKER_ERROR_REVIEW_REQUIRED"
        assert "DECISION_WORKER_ERROR" in runtime.alerts.conditions(runtime, time.time())["fatal"]
        with pytest.raises(ValueError, match="DECISION_WORKER_ERROR"):
            runtime.broker.arm("ENABLE PAPER ONLY")
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())
