"""Broker polling, audit and log volume stay bounded while obligations keep full cadence."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from saxo_support import FUTURE, FakeClient, plan, setup, signal
from stocker_execution.config import FuturesConfig
from stocker_execution.recorder import Recorder
from stocker_execution.rules import Bar, prior_rv, reference_summary
from stocker_execution.runtime import Runtime, failure_code
from stocker_execution.saxo_data import DataService
from stocker_execution.store import Store

PORTFOLIO = {"/port/v1/positions/me", "/port/v1/orders/me"}


def portfolio_reads(client):
    return sum(path in PORTFOLIO for _, path, _ in client.calls)


def test_idle_sim_reconciles_on_interval_and_sim_entries_need_a_recent_pass(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        broker.reconciled = False
        await broker.manage()
        assert broker.reconciled and portfolio_reads(data.client) == 2
        for _ in range(5):
            await broker.manage()
        assert portfolio_reads(data.client) == 2
        broker.last_reconcile -= 31
        await broker.manage()
        assert portfolio_reads(data.client) == 4
        assert broker.entry_reason() == ""
        broker.last_reconcile -= 61
        assert broker.entry_reason() == "RECONCILIATION_STALE"
        store.db.close()

    asyncio.run(scenario())


def test_pending_order_is_reconciled_every_cycle(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        event = signal(1)
        store.observe(event, "", {})
        await broker.enter(event, plan())  # ambiguous response leaves a SUBMITTING order
        before = portfolio_reads(data.client)
        for _ in range(3):
            await broker.manage()
        assert portfolio_reads(data.client) == before + 6
        store.db.close()

    asyncio.run(scenario())


def test_internal_paper_cycle_with_unchanged_position_writes_nothing(tmp_path):
    async def scenario():
        broker, _, store = setup(tmp_path)
        event = signal(1)
        store.observe(event, "", {})
        assert await broker.enter(event, plan()) == ""
        await broker.manage()
        changes = store.db.total_changes
        for _ in range(3):
            await broker.manage()
        assert store.db.total_changes == changes
        assert store.exposure(event["id"]) == 1
        store.db.close()

    asyncio.run(scenario())


def test_repeated_management_exception_is_audited_once(tmp_path):
    async def scenario():
        broker, data, store = setup(tmp_path, "SAXO_SIM")
        event = signal(1)
        store.observe(event, "", {})
        assert store.reserve(event["id"], plan()) == ""
        data.client.positions = [
            {"PositionBase": {"Amount": 1, "AccountId": "other", "AssetType": "FuturesOption"}}
        ]
        broker.reconciled = False
        for _ in range(5):
            await broker.manage()
        rows = store.db.execute(
            "SELECT detail FROM lifecycle WHERE reference=? AND kind='DECISION'", (event["id"],)
        ).fetchall()
        assert len(rows) == 1
        assert broker.management_problems[event["id"]] == "UNACCOUNTED_BROKER_POSITION_OR_ACCOUNT"
        store.db.close()

    asyncio.run(scenario())


def test_runtime_log_has_timestamps_and_only_coded_reasons(tmp_path):
    runtime = Runtime(FuturesConfig(), Store(tmp_path / "ledger.sqlite3"))
    runtime.report_failure("management", ValueError("SIM_ACCOUNT_NOT_VERIFIED"))
    runtime.report_failure("stream", RuntimeError("https://host/?token=secret"))
    runtime.log_handler.flush()
    text = (tmp_path / "SAXO_SIM" / "slrno.log").read_text()
    assert "ERROR stocker_execution.runtime management failed: SIM_ACCOUNT_NOT_VERIFIED" in text
    assert "stream failed: RuntimeError" in text and "secret" not in text
    assert text[:4].isdigit()  # asctime prefix
    assert failure_code(ValueError("invalid literal for int()")) == "ValueError"
    logging.getLogger("stocker_execution.runtime").removeHandler(runtime.log_handler)
    runtime.log_handler.close()
    runtime.store.db.close()


def test_failed_reference_sessions_are_not_refetched_every_minute(tmp_path, monkeypatch):
    import stocker_execution.saxo_data as module

    at = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=10)
    rows = [
        {"Time": (at + timedelta(minutes=i)).isoformat(), "Open": 70, "High": 71}
        | {"Low": 69, "Close": 70.5, "Volume": 20}
        for i in range(5)
    ]
    loads = []

    def failing(*args):
        loads.append(args)
        raise ValueError("REFERENCE_SELECTION_AUDIT_INVALID")

    monkeypatch.setattr(module, "load_selections", failing)

    async def scenario():
        config = FuturesConfig(reference_selections_file=tmp_path / "selections.json")
        client = FakeClient()

        async def chart(*args, **kwargs):
            return {"Data": rows, "DataVersion": 1, "ChartInfo": {"DelayedByMinutes": 0}}

        client.request = chart
        data = DataService(config, client, Recorder(config.recorder, tmp_path / "events"))
        state = data.markets["CL"]
        state.identity = FUTURE
        for _ in range(3):
            await data.history(state)
            assert state.history_problem == "REFERENCE_SESSION_AUDIT_OR_SAXO_COVERAGE_UNVERIFIED"
            assert state.bars and not state.references
        assert len(loads) == 1
        state.reference_retry_at = 0
        await data.history(state)
        assert len(loads) == 2

    asyncio.run(scenario())


def test_indexed_reference_summary_matches_previous_quadratic_version():
    import math
    from statistics import median

    from stocker_execution.rules import NY

    def old_prior_rv(bars, at, count=15):
        prefix = {b.at: b for b in bars if b.at + timedelta(minutes=1) <= at}
        needed = [prefix.get(at - timedelta(minutes=i)) for i in range(count + 1, 0, -1)]
        if any(b is None or not b.valid() for b in needed):
            raise ValueError("INCOMPLETE_COMPLETED_HISTORY")
        closes = [b.close for b in needed]
        pairs = zip(closes, closes[1:], strict=False)
        return math.sqrt(sum(math.log(b / a) ** 2 for a, b in pairs))

    def old_summary(bars):
        result, by_time = {}, {b.at: b for b in bars}
        for hour in range(8, 17):
            samples = {k: [] for k in ("rv15", "range15", "volume15")}
            for bar in bars:
                if bar.at.astimezone(NY).hour != hour:
                    continue
                try:
                    rv = old_prior_rv(bars, bar.at)
                    prior = [by_time[bar.at - timedelta(minutes=i)] for i in range(1, 16)]
                except (KeyError, ValueError):
                    continue
                samples["rv15"].append(rv)
                samples["range15"].append(
                    (max(b.high for b in prior) - min(b.low for b in prior))
                    / by_time[bar.at - timedelta(minutes=1)].close
                )
                samples["volume15"].append(sum(b.volume for b in prior))
            result[hour] = {k: median(v) for k, v in samples.items() if v}
        return result

    start = datetime(2026, 9, 25, 12, tzinfo=UTC)  # 08:00 New York
    bars = [
        Bar(start + timedelta(minutes=i), 70 + i % 7, 71 + i % 7, 69 + i % 7, 70.5 + i % 7, 10 + i)
        for i in range(540)
        if i != 200  # one gap exercises the incomplete-history path
    ]
    assert reference_summary(bars) == old_summary(bars)
    at = start + timedelta(minutes=300)
    assert prior_rv(bars, at) == old_prior_rv(bars, at)


def test_unchanged_broker_evidence_is_audited_once(tmp_path):
    from saxo_support import OPTION

    broker, _, store = setup(tmp_path, "SAXO_SIM")
    event = signal(1)
    store.observe(event, "", {})
    store.reserve(event["id"], plan())
    store.prepare_order(event["id"], "ENTRY", 1, event["exit_at"], {"option": OPTION})
    evidence = {
        "OrderId": "fixture",
        "LogId": "one",
        "Status": "Placed",
        "SubStatus": "Confirmed",
        "FilledAmount": 0,
        "AveragePrice": None,
        "ActivityTime": event["signal_at"],
    }

    def audited():
        return store.db.execute(
            "SELECT COUNT(*) FROM lifecycle WHERE kind='SAXO_SIM_ORDER_EVIDENCE'"
        ).fetchone()[0]

    for _ in range(3):
        broker.apply_order_evidence(store.orders(event["id"])[0], evidence)
    assert audited() == 1
    filled = {**evidence, "LogId": "two", "Status": "FinalFill", "FilledAmount": 1}
    broker.apply_order_evidence(store.orders(event["id"])[0], {**filled, "AveragePrice": 0.01})
    assert audited() == 2 and store.orders(event["id"])[0]["status"] == "Filled"
    store.db.close()
