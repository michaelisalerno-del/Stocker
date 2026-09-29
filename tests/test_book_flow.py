"""Sanitised sampled-book fixtures, never an authenticated feed or execution tape."""

import asyncio
import copy
import gzip
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from saxo_support import IDENTITY, book
from stocker_execution.book_flow import VERSION, observe
from stocker_execution.config import (
    MARKETS,
    MAX_OPEN_POSITIONS,
    MAX_PREMIUM_RISK_GBP,
    FuturesConfig,
    RecorderConfig,
)
from stocker_execution.contracts import key
from stocker_execution.recorder import Recorder, apply, read_row
from stocker_execution.runtime import Runtime
from stocker_execution.saxo_client import allowed
from stocker_execution.saxo_data import DataService
from stocker_execution.saxo_stream import PriceState
from stocker_execution.store import Store


def point(value, at, history=(), generation="one", timeout=5):
    return observe(
        IDENTITY,
        value,
        at,
        {"subscription_id": generation, "valid_until": at + timeout},
        list(history),
    )


def test_fixed_formulas_grid_and_causal_time_weighted_persistence():
    history = []
    for t in range(61):
        p = point(book(bid_size=3 if t < 30 else 1, ask_size=1 if t < 30 else 3), t, history)
        history.append(p)
    first, last = history[0], history[-1]
    assert first["spread_ticks"] == 1
    assert first["weighted_midpoint"] == pytest.approx(70.0075)
    assert first["weighted_displacement_ticks"] == pytest.approx(0.25)
    for n in (1, 3, 5, 10):
        assert first["depth"][str(n)]["bid"] == n * 3
        assert first["depth"][str(n)]["imbalance"] == 0.5
        assert first["depth"][str(n)]["order_imbalance"] == pytest.approx(1 / 3)
    sixty = last["lookbacks"]["60"]["5"]
    assert sixty["bid_heavy_fraction"] == 0.5 and sixty["ask_heavy_fraction"] == 0.5
    assert sixty["bid_change"] == -10 and sixty["ask_change"] == 10
    assert first["lookbacks"]["5"]["1"]["status"] == "INSUFFICIENT_HISTORY"
    # A future observation is never eligible as the lookback baseline.
    assert point(book(), 1, [last])["lookbacks"]["5"]["1"]["status"] == "INSUFFICIENT_HISTORY"


def test_missing_levels_order_counts_zero_denominators_and_delay():
    value = book(3)
    value["MarketDepth"]["UsingOrders"] = False
    p = point(value, 0)
    assert p["depth"]["3"]["bid"] == 9
    assert p["depth"]["5"]["bid"] is None
    assert p["depth"]["1"]["order_imbalance"] is None
    del value["MarketDepth"]["AskSize"]
    assert point(value, 0)["depth"]["1"]["imbalance"] is None
    empty = point(book(bid_size=0, ask_size=0), 0)
    assert empty["weighted_midpoint"] is None and empty["depth"]["10"]["imbalance"] is None
    value["MarketDepth"] = None
    p = point(value, 0)
    assert "L2_UNAVAILABLE" in p["quality_flags"] and p["spread_ticks"] == 1
    value["Quote"]["DelayedByMinutes"] = 15
    assert point(value, 0)["status"] == "DELAYED"
    value["Quote"]["Bid"] = 70.0001
    assert point(value, 0)["status"] == "UNAVAILABLE"


def test_repeated_last_trade_and_volume_resets_never_create_tape():
    initial = point(book(), 0)
    repeated = point(book(), 1, [initial])
    assert repeated["latest_trade"] == initial["latest_trade"] == {"price": 70, "size": 2}
    assert repeated["volume"]["change"] is None
    reset = book()
    reset["PriceInfoDetails"]["Volume"] = 4
    corrected = point(reset, 2, [initial, repeated])
    assert corrected["volume"]["status"] == "RESET_OR_CORRECTION_UNCLASSIFIED"
    assert corrected["volume"]["change"] is None
    reset["PriceInfoDetails"]["Volume"] = 9
    assert point(reset, 3, [corrected])["volume"]["change"] is None
    assert (
        point(reset, 3, [corrected], generation="new")["volume"]["status"] == "SEMANTICS_UNVERIFIED"
    )
    assert "executions" not in repeated and "aggressor" not in repeated


def test_price_shifts_match_tick_prices_and_gaps_invalidate_lookbacks():
    history = [point(book(), t) for t in range(61)]
    shifted = point(book(shift=0.01), 61, history)
    d = shifted["lookbacks"]["5"]["5"]
    assert d["bid_change"] == 0 and d["matched_bid_change"] == 0
    assert d["matched_bid_levels"] == 4  # one price left the window; no cancellation inference
    assert "cancel" not in json.dumps(shifted).lower()
    history.append(point(None, 61, history))
    after = point(book(), 62, history)
    assert after["lookbacks"]["5"]["5"]["status"] == "INSUFFICIENT_HISTORY"
    assert (
        point(book(), 62, history[:1])["lookbacks"]["60"]["1"]["status"] == "INSUFFICIENT_HISTORY"
    )


def test_current_quote_sizes_and_health_are_independent_of_field_changes():
    p = PriceState()
    p.inactivity_timeout = 30
    raw = book()
    raw["PriceInfoDetails"].update(BidSize=999, AskSize=999)
    p.snapshot(raw, "ref", 100)
    p.update({"PriceInfoDetails": {"AskSize": 1000}}, "one", 101)
    assert p.sizes() == {"Bid": 3, "Ask": 1} and p.size_times["Ask"] == 100
    p.update({"Quote": {"AskSize": None}}, "two", 102)
    assert p.sizes()["Ask"] is None
    changed = p.last_field_change
    p.update({"Quote": {"AskSize": None}}, "three", 110)
    assert p.last_field_change == changed and p.last_receipt == 110
    p.last_contact = 130  # NoNewData heartbeat; old unchanged book is still reconstructed
    assert p.depth(150)["fresh"] and not p.depth(161)["fresh"]
    assert p.receipt == 100  # trading-critical quote-age rule unchanged


def test_malformed_optional_depth_retains_raw_without_interrupting_data(tmp_path):
    r = Recorder(RecorderConfig(), tmp_path)
    r.register("future", IDENTITY)
    raw = {**book(), "MarketDepth": ["unexpected"]}
    state = PriceState()
    state.snapshot(raw, "ref", 100)
    r.ingest("future", "SNAPSHOT", raw, 100)
    assert read_row(r.windows["future"].rows[-1][1])["payload"] == raw
    assert r.windows["future"].flow["status"] == "UNAVAILABLE"
    r.ingest("future", "SNAPSHOT", book(), 101)
    assert r.windows["future"].flow["status"] == "CURRENT"


def test_queued_delta_does_not_backdate_calculation_using_later_snapshot(tmp_path):
    r = Recorder(RecorderConfig(), tmp_path)
    r.register("future", IDENTITY)
    r.ingest("future", "SNAPSHOT", book(), 100)
    r.ingest("future", "UPDATE", {"Quote": {"AskSize": 2}}, 90)
    row = read_row(r.windows["future"].rows[-1][1])
    assert row["receipt"] == 90 and row["book_flow"]["at"] == 100
    assert row["book_flow"]["lookbacks"]["5"]["1"]["status"] == "INSUFFICIENT_HISTORY"


@pytest.mark.parametrize("heartbeat_array", [False, True])
def test_five_ordinary_subscriptions_shared_by_consumers_and_heartbeats(
    tmp_path, monkeypatch, heartbeat_array
):
    async def scenario():
        calls = []

        async def request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            assert "infoprices" not in path
            if method == "POST":
                return {"Snapshot": book(), "RefreshRate": 1500, "InactivityTimeout": 30}
            return {}

        client = SimpleNamespace(request=request, oauth=SimpleNamespace(account_key="fixture"))
        recorder = Recorder(RecorderConfig(), tmp_path)
        data = DataService(FuturesConfig(), client, recorder)

        async def discover(state):
            state.identity = {
                **IDENTITY,
                "market": state.market,
                "uic": 100 + MARKETS.index(state.market),
            }
            recorder.register(key(state.identity), state.identity)

        async def no_fx():
            pass

        monkeypatch.setattr(data, "discover", discover)
        monkeypatch.setattr(data, "subscribe_fx", no_fx)
        await data.startup()
        await data.startup()
        assert len(calls) == 6  # five ordinary prices plus session, no duplicate consumers
        prices = [(ref, s) for ref, s in data.subscriptions.items() if s["kind"] == "PRICE"]
        assert len(prices) == 5
        for _ref, s in prices:
            assert s["path"] == "/trade/v1/prices/subscriptions"
            assert {"Quote", "PriceInfo", "PriceInfoDetails", "MarketDepth"} <= set(
                s["arguments"]["FieldGroups"]
            )
            assert s["refresh_ms"] == 1500
            await asyncio.gather(
                *(data.subscribe("PRICE", s["arguments"], s["target"]) for _ in range(4))
            )
        assert len(calls) == 6
        assert not allowed("POST", "/trade/v1/infoprices/subscriptions")
        ref, s = prices[0]
        p = data.markets[s["target"]].price
        first = p.last_receipt
        assert first is not None
        heartbeat = {
            "reference": "_heartbeat",
            "message_id": "heartbeat",
            "payload": {"Heartbeats": [{"OriginatingReferenceId": ref, "Reason": "NoNewData"}]},
        }
        if heartbeat_array:
            heartbeat["payload"] = [heartbeat["payload"]]
        await data.receive(heartbeat, first + 20)
        assert p.depth(first + 20)["fresh"] and p.last_receipt == first
        assert p.last_field_change == first
        for i in range(2):
            await data.receive(
                {
                    "reference": ref,
                    "message_id": str(i),
                    "payload": {
                        "Data": {"Quote": {"Bid": 70}},
                        "Timestamp": "2026-09-28T10:00:00Z",
                    },
                },
                first + 21 + i,
            )
        flow = recorder.windows[key(data.markets[s["target"]].identity)].flow
        recorded = read_row(recorder.windows[key(data.markets[s["target"]].identity)].rows[-1][1])
        assert recorded["provider_message"]["payload"]["Timestamp"] == "2026-09-28T10:00:00Z"
        assert flow["feed"]["granted_refresh_ms"] == 1500
        assert flow["feed"]["observed_receipt_ms"]["samples"] == 2
        assert not recorder.active and not recorder.event_ids  # flow creates no events
        old = p.generation
        new = await data.subscribe("PRICE", s["arguments"], s["target"], old=old)
        assert old not in data.subscriptions and new in data.subscriptions
        before = copy.deepcopy(p.value)
        await data.receive(
            {"reference": old, "message_id": "late", "payload": {"Quote": {"Bid": 999}}}
        )
        assert p.value == before

    asyncio.run(scenario())


def test_same_recorder_checkpoint_event_overlap_limits_and_feature_manifest(tmp_path):
    async def scenario():
        r = Recorder(
            RecorderConfig(
                persistent_capture=True,
                recording_permission_evidence="OFFLINE FIXTURE",
                disk_reserve_bytes=0,
            ),
            tmp_path,
        )
        r.register("future", IDENTITY)
        await r.start()
        r.ingest("future", "SNAPSHOT", book(), 0, generation="one")
        for t in range(1, 1002):
            r.ingest(
                "future", "UPDATE", {"MarketDepth": {"BidSize": [t] * 10}}, t, generation="one"
            )
        w = r.windows["future"]
        assert w.coverage(1001) == 900
        rebuilt = w.checkpoint
        for _, blob, _ in w.rows:
            rebuilt = apply(rebuilt, read_row(blob))
        assert rebuilt == w.current and w.checkpoint_flow["version"] == VERSION
        assert r.memory() <= r.config.rolling_max_bytes
        event = {"id": "frozen-one", "skip_reason": "PAPER_DISARMED"}
        first = r.trigger("future", event, 1001)
        second = r.trigger("future", {"id": "frozen-two", "skip_reason": "MONITOR_ONLY"}, 1002)
        assert first["segment"] == second["segment"]
        assert r.trigger("future", event, 1003)["state"] == "DUPLICATE_SUPPRESSED"
        await r.queue.join()
        segment = first["segment"]
        manifest = json.loads((tmp_path / (segment + ".manifest.json")).read_text())
        assert manifest["book_flow"]["version"] == VERSION
        assert manifest["book_flow"]["coverage_seconds"] == 900
        rows = [
            json.loads(b)
            for b in gzip.decompress((tmp_path / (segment + ".jsonl.gz")).read_bytes()).splitlines()
        ]
        sequences = [row["local_sequence"] for row in rows if "local_sequence" in row]
        assert len(sequences) == len(set(sequences))
        assert rows[0]["book_flow"]["version"] == VERSION
        assert any(row.get("book_flow", {}).get("lookbacks", {}) for row in rows)
        # A second market never leaks into this event, while its linked option does.
        r.register("other", {**IDENTITY, "market": "GC", "uic": 999})
        r.register("option", {**IDENTITY, "asset_type": "FuturesOption", "uic": 124})
        r.ingest("option", "SNAPSHOT", {"Quote": {"Bid": 1}}, 1003)
        r.attach("frozen-one", "option", 1003)
        r.ingest("other", "SNAPSHOT", book(), 1004)
        r.ingest("option", "UPDATE", {"Quote": {"Bid": 2}}, 1004)
        await r.queue.join()
        path = tmp_path / (segment + ".jsonl.gz")
        saved = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        assert not any(row.get("identity", {}).get("uic") == 999 for row in saved)
        assert any(
            row.get("identity", {}).get("uic") == 124 and row["kind"] == "UPDATE" for row in saved
        )
        r.tick(5000)
        await r.queue.join()
        completed = path.read_bytes()
        r.ingest("future", "UPDATE", book(), 5001)
        await r.queue.join()
        assert path.read_bytes() == completed
        await r.close()
        limited = Recorder(
            RecorderConfig(
                persistent_capture=True,
                recording_permission_evidence="OFFLINE FIXTURE",
                archive_max_bytes=65536,
                disk_reserve_bytes=0,
            ),
            tmp_path / "limited",
        )
        limited.register("future", IDENTITY)
        await limited.start()
        limited.ingest("future", "SNAPSHOT", book(), 0)
        limited.trigger("future", event, 0)
        await limited.queue.join()
        assert limited.problem == "STORAGE_LIMIT_REACHED"
        failed = json.loads(next(limited.directory.glob("*.manifest.json")).read_text())
        assert failed["book_flow"]["version"] == VERSION and failed["state"] == "INCOMPLETE"
        limited.ingest("future", "UPDATE", book(), 1)
        assert limited.windows["future"].current and limited.windows["future"].flow
        await limited.close()

    asyncio.run(scenario())


def test_l2_fluctuations_do_not_change_frozen_decisions_or_create_events(tmp_path, monkeypatch):
    import stocker_execution.runtime as runtime_module

    at = datetime(2026, 9, 28, 13, tzinfo=UTC)
    monkeypatch.setattr(runtime_module, "now", lambda: at)
    # Keep the existing completed-bar eligibility seam fixed while varying only L2.
    monkeypatch.setattr(
        runtime_module, "eligibility", lambda *args: {"rv15": 0.01, "futures_price": 70}
    )

    async def scenario():
        decisions = []
        for i, depth in enumerate(
            (book(bid_size=100)["MarketDepth"], book(ask_size=100)["MarketDepth"], None)
        ):
            store = Store(tmp_path / f"ledger-{i}.sqlite3")
            runtime = Runtime(FuturesConfig(), store)
            runtime.started_at = at - timedelta(seconds=1)
            state = runtime.markets["CL"]
            state.identity = {**IDENTITY, "expiry": "2026-12-01"}
            state.problem = state.history_problem = ""
            state.price.snapshot({**book(), "MarketDepth": depth}, "fixture", at.timestamp())
            runtime.recorder.register(key(state.identity), state.identity)
            runtime.recorder.ingest(
                key(state.identity), "SNAPSHOT", state.price.value, at.timestamp()
            )
            assert not runtime.recorder.event_ids
            await runtime.decisions()
            await runtime.decisions()  # dashboard/quote refresh never emits another clock
            rows = store.history(None, None, None)
            assert len(rows) == 1 and len(runtime.recorder.event_ids) == 1
            decisions.append(
                (rows[0]["decision"], rows[0]["reason"], rows[0]["signal_at"], rows[0]["exit_at"])
            )
            assert not runtime.broker.armed and not store.orders(rows[0]["id"])
            assert MAX_OPEN_POSITIONS == 4 and MAX_PREMIUM_RISK_GBP == 50
            await runtime.stop()
            store.db.close()
        assert len(set(decisions)) == 1

    asyncio.run(scenario())
