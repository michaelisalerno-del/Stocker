"""Streamed one-minute chart samples: the boundary bar, its REST confirmation, the tripwire."""

import asyncio
import json
from datetime import timedelta

from saxo_support import AT, FUTURE, quote, setup
from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.rules import Bar
from stocker_execution.runtime import Runtime
from stocker_execution.saxo_history import history, streamed_bar
from stocker_execution.saxo_stream import ChartState
from stocker_execution.store import Store


def sample(at, close=70.0, volume=10.0):
    return {
        "Time": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": volume,
        "Interest": 5.0,
        "MarketTradingState": "Automated",
    }


def from_bar(bar):
    return {
        "Time": bar.at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Open": bar.open,
        "High": bar.high,
        "Low": bar.low,
        "Close": bar.close,
        "Volume": bar.volume,
    }


def test_chart_state_completes_a_sample_only_once_the_next_one_starts():
    c = ChartState()
    minute = AT - timedelta(minutes=1)
    c.snapshot(
        {
            "DataVersion": 7,
            "ChartInfo": {"DelayedByMinutes": 0},
            "Data": [sample(minute - timedelta(minutes=1)), sample(minute)],
        },
        "g",
        AT.timestamp() - 30,
    )
    # The minute before the clock is still the newest sample: mutable, so not a bar yet.
    assert streamed_bar(c, minute, AT) is None
    # Saxo opens the next sample when the minute ends and sends the closed bar with it.
    c.update({"Data": [sample(minute, 71), sample(AT, 71.5, 0)]}, AT.timestamp() + 0.4)
    assert streamed_bar(c, minute, AT + timedelta(seconds=1)) == Bar(
        minute, 71, 72, 70, 71, 10, interest=5.0, state="Automated"
    )
    # An in-place revision of the open sample merges; a DataVersion change clears everything.
    c.update({"Data": [sample(AT, 72, 3)]}, AT.timestamp() + 1)
    assert c.samples[sample(AT)["Time"]]["Close"] == 72 and c.updates == 2
    c.update({"DataVersion": 8, "Data": [sample(AT)]}, AT.timestamp() + 2)
    assert c.problem == "CHART_DATA_VERSION_CHANGED" and not c.samples
    assert streamed_bar(c, minute, AT + timedelta(seconds=3)) is None
    # A quiet stream past its inactivity timeout, or a delayed snapshot, never supplies a bar.
    c.snapshot(
        {
            "DataVersion": 8,
            "ChartInfo": {"DelayedByMinutes": 0},
            "Data": [sample(minute), sample(AT)],
        },
        "h",
        AT.timestamp(),
    )
    assert streamed_bar(c, minute, AT + timedelta(seconds=1)) is not None
    assert streamed_bar(c, minute, AT + timedelta(seconds=31)) is None
    c.snapshot(
        {
            "DataVersion": 8,
            "ChartInfo": {"DelayedByMinutes": 10},
            "Data": [sample(minute), sample(AT)],
        },
        "i",
        AT.timestamp(),
    )
    assert c.problem == "SAXO_CHART_DATA_DELAYED"
    assert streamed_bar(c, minute, AT + timedelta(seconds=1)) is None
    assert c.view(AT.timestamp() + 1)["status"] == "UNAVAILABLE"


def test_clock_decides_on_the_streamed_boundary_bar_and_records_it(tmp_path, monkeypatch):
    import stocker_execution.runtime as module

    monkeypatch.setattr(module, "now", lambda: AT + timedelta(seconds=2))

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
        state.bars = bars[:-1]  # the REST read has not returned the final minute yet
        state.price = quote(70, 71, (AT + timedelta(seconds=2)).timestamp())
        runtime.recorder.register(key(FUTURE), FUTURE)
        state.chart.snapshot(
            {
                "DataVersion": 1,
                "ChartInfo": {"DelayedByMinutes": 0},
                "Data": [from_bar(bars[-1]), sample(AT, 70.6, 1)],
            },
            "g",
            AT.timestamp() + 1,
        )
        await runtime.decisions()
        rows = runtime.store.history(None, None, None)
        assert len(rows) == 1 and rows[0]["reason"] == "EXECUTION_DISABLED"  # decided, not waited
        detail = json.loads(runtime.store.db.execute("SELECT detail FROM signals").fetchone()[0])
        assert detail["boundary_bar"]["source"] == "CHART_STREAM"
        assert detail["boundary_bar"]["minute"] == bars[-1].at.isoformat()
        assert state.bars[-1].at == bars[-1].at and state.chart_used == 1
        assert list(state.stream_bars) == [bars[-1].at]
        assert state.boundary_clock is None
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_clock_takes_every_trailing_minute_the_rest_read_lacks_from_the_stream(
    tmp_path, monkeypatch
):
    """A REST read inside clock-2 ends at clock-3 (it drops its newest sample); the stream
    holds both missing minutes, so the clock decides on them (review 2026-10-04)."""
    import stocker_execution.runtime as module

    monkeypatch.setattr(module, "now", lambda: AT + timedelta(seconds=2))

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
        state.bars = bars[:-2]
        state.price = quote(70, 71, (AT + timedelta(seconds=2)).timestamp())
        runtime.recorder.register(key(FUTURE), FUTURE)
        state.chart.snapshot(
            {
                "DataVersion": 1,
                "ChartInfo": {"DelayedByMinutes": 0},
                "Data": [from_bar(bars[-2]), from_bar(bars[-1]), sample(AT, 70.6, 1)],
            },
            "g",
            AT.timestamp() + 1,
        )
        await runtime.decisions()
        rows = runtime.store.history(None, None, None)
        assert len(rows) == 1 and rows[0]["reason"] == "EXECUTION_DISABLED"
        detail = json.loads(runtime.store.db.execute("SELECT detail FROM signals").fetchone()[0])
        assert detail["boundary_bar"]["minutes"] == [b.at.isoformat() for b in bars[-2:]]
        assert detail["boundary_bar"]["minute"] == bars[-1].at.isoformat()
        assert [b.at for b in state.bars[-2:]] == [b.at for b in bars[-2:]]
        assert state.chart_used == 2 and list(state.stream_bars) == [b.at for b in bars[-2:]]
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_quiet_minutes_before_the_boundary_wait_for_the_rest_read(tmp_path, monkeypatch):
    """Two no-trade minutes before clock-1 are missing from REST and the stream alike. The
    boundary REST read will fill them (the <=5-minute rule) once it carries clock-1, so the
    clock waits instead of skipping; a sixth quiet minute is a gap and stays final."""
    import stocker_execution.runtime as module

    monkeypatch.setattr(module, "now", lambda: AT + timedelta(seconds=2))

    async def scenario(missing):
        runtime = Runtime(FuturesConfig(), Store(tmp_path / f"ledger-{missing}.sqlite"))
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
        state.bars = bars[:-missing]
        state.price = quote(70, 71, (AT + timedelta(seconds=2)).timestamp())
        runtime.recorder.register(key(FUTURE), FUTURE)
        state.chart.snapshot(
            {"DataVersion": 1, "ChartInfo": {"DelayedByMinutes": 0}, "Data": [sample(AT)]},
            "g",
            AT.timestamp() + 1,
        )
        await runtime.decisions()
        waiting = state.boundary_clock == AT
        reasons = [r["reason"] for r in runtime.store.history(None, None, None)]
        await runtime.stop()
        runtime.store.db.close()
        return waiting, reasons

    assert asyncio.run(scenario(3)) == (True, [])  # clock-1 plus two quiet minutes
    assert asyncio.run(scenario(6)) == (True, [])  # the longest run the fill rule covers
    assert asyncio.run(scenario(7)) == (False, ["INCOMPLETE_COMPLETED_HISTORY"])


def test_a_tripped_stream_leaves_the_clock_waiting_for_rest(tmp_path, monkeypatch):
    import stocker_execution.runtime as module

    monkeypatch.setattr(module, "now", lambda: AT + timedelta(seconds=2))

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
        state.price = quote(70, 71, (AT + timedelta(seconds=2)).timestamp())
        runtime.recorder.register(key(FUTURE), FUTURE)
        state.chart.snapshot(
            {
                "DataVersion": 1,
                "ChartInfo": {"DelayedByMinutes": 0},
                "Data": [from_bar(bars[-1]), sample(AT, 70.6, 1)],
            },
            "g",
            AT.timestamp() + 1,
        )
        runtime.data.chart_stream_problem = "CHART_STREAM_MISMATCH"
        await runtime.decisions()
        assert not runtime.store.history(None, None, None)
        assert state.boundary_clock == AT and state.chart_used == 0
        assert runtime.status()["chart_stream_problem"] == "CHART_STREAM_MISMATCH"
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())


def test_rest_read_confirms_or_trips_the_streamed_bar(tmp_path):
    async def scenario():
        _, data, store = setup(tmp_path)
        state = data.markets["CL"]
        minute = AT - timedelta(minutes=1)
        charts = {
            "Data": [sample(minute - timedelta(minutes=1)), sample(minute), sample(AT)],
            "DataVersion": 1,
            "ChartInfo": {"DelayedByMinutes": 0},
        }
        original = data.client.request

        async def request(method, path, **kwargs):
            if path == "/chart/v3/charts":
                return charts
            return await original(method, path, **kwargs)

        data.client.request = request
        # A REST read that has not reached the minute yet leaves the check pending.
        state.stream_bars[AT] = (Bar(AT, 70, 71, 69, 70, 10), "event-0")
        await history(data, state)
        assert list(state.stream_bars) == [AT] and data.chart_stream_problem == ""
        state.stream_bars.clear()
        # Agreement confirms; the bar leaves the pending list.
        state.stream_bars[minute] = (Bar(minute, 70, 71, 69, 70, 10), "event-1")
        await history(data, state)
        assert not state.stream_bars and state.chart_confirmed == 1
        assert data.chart_stream_problem == "" and not data.chart_audits
        # Any difference trips the stream for good and leaves evidence for the clock.
        state.stream_bars[minute] = (Bar(minute, 70, 71, 69, 70.5, 10), "event-2")
        await history(data, state)
        assert data.chart_stream_problem == "CHART_STREAM_MISMATCH"
        assert data.chart_mismatch["event_id"] == "event-2"
        assert (
            data.chart_mismatch["streamed"]["close"] == 70.5
            and data.chart_mismatch["rest"]["close"] == 70
        )
        audits = data.drain_chart_audits()
        assert audits[0][0] == "event-2" and not data.chart_audits
        assert data.capability_view(state)["chart_stream"]["tripped"] == "CHART_STREAM_MISMATCH"

    asyncio.run(scenario())


def test_receive_routes_chart_updates_heartbeats_and_resets(tmp_path):
    async def scenario():
        _, data, _ = setup(tmp_path)
        state = data.markets["CL"]
        data.subscriptions["chart"] = {
            "kind": "CHART",
            "target": "CL",
            "timeout": 30,
            "contact": 0,
            "path": "/chart/v3/charts/subscriptions",
            "arguments": {},
        }
        state.chart.snapshot(
            {
                "DataVersion": 1,
                "ChartInfo": {"DelayedByMinutes": 0},
                "Data": [sample(AT - timedelta(minutes=1))],
            },
            "chart",
            AT.timestamp(),
        )
        await data.receive(
            {"reference": "chart", "message_id": "1", "payload": {"Data": {"Data": [sample(AT)]}}},
            AT.timestamp() + 1,
        )
        assert len(state.chart.samples) == 2 and state.chart.receipt == AT.timestamp() + 1
        await data.receive(
            {
                "reference": "_heartbeat",
                "message_id": "2",
                "payload": {
                    "Heartbeats": [{"OriginatingReferenceId": "chart", "Reason": "NoNewData"}]
                },
            },
            AT.timestamp() + 2,
        )
        assert state.chart.last_contact == AT.timestamp() + 2 and not state.chart.problem
        await data.receive(
            {
                "reference": "_heartbeat",
                "message_id": "3",
                "payload": {
                    "Heartbeats": [
                        {
                            "OriginatingReferenceId": "chart",
                            "Reason": "SubscriptionTemporarilyDisabled",
                        }
                    ]
                },
            },
            AT.timestamp() + 3,
        )
        assert (
            state.chart.problem == "SUBSCRIPTION_TEMPORARILY_DISABLED"
            and "chart" in data.reset_refs
        )
        state.chart.problem = ""
        await data.receive(
            {
                "reference": "_resetsubscriptions",
                "message_id": "4",
                "payload": {"TargetReferenceIds": ["chart"]},
            },
            AT.timestamp() + 4,
        )
        assert state.chart.problem == "SUBSCRIPTION_RESET_BY_PROVIDER"

    asyncio.run(scenario())
