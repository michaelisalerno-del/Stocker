"""Deterministic offline observation tests. No broker sockets or operational paths."""

import asyncio
import itertools
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from ib_async import Contract, TickAttribBidAsk, TickAttribLast
from pydantic import ValidationError

from stocker_execution import first4_flow_observer as observer_module
from stocker_execution.first4_config import First4Config, OrderFlowConfig
from stocker_execution.first4_flow import FlowEvent, FlowReducer, metrics
from stocker_execution.first4_flow_observer import FlowObserver
from stocker_execution.first4_flow_store import CaptureStart, FlowWriter, read_flow, replay
from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_store import Store

STAMP = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
MODE = "TBT_TRADES_TBT_QUOTES"


def event(sequence, kind="trade", **values):
    defaults = dict(
        capture_id="fixture",
        session="2026-09-25",
        con_id=1,
        request_id=42,
        generation=1,
        sequence=sequence,
        received_at=STAMP.isoformat(),
        monotonic_ns=sequence * 1_000_000,
        kind=kind,
        feed_mode=MODE,
        broker_time=int(STAMP.timestamp()),
        broker_precision="seconds",
        price=11.0,
        size=100.0,
        tick_type=1,
        source="Last",
    )
    defaults.update(values)
    return FlowEvent(**defaults)


def quote(sequence=1, **values):
    return event(
        sequence,
        "quote",
        **dict(bid=10.0, ask=11.0, bid_size=20.0, ask_size=30.0, source="BidAsk", **values),
    )


@pytest.mark.parametrize(
    "price,direction",
    [(11, "BUY_EST"), (10, "SELL_EST"), (10.5, "UNKNOWN"), (12, "UNKNOWN"), (9, "UNKNOWN")],
)
def test_classification_and_volume_identity(price, direction):
    reducer = FlowReducer()
    reducer.apply(quote())
    assert reducer.apply(event(2, price=price))["direction"] == direction
    totals = reducer.snapshot()["totals"]
    assert (
        totals["eligible_observed_volume"]
        == sum(totals[k] for k in ("buy_est_volume", "sell_est_volume", "unknown_volume"))
        == 100
    )
    assert totals["trade_count"] == 1
    assert totals["volume_delta"] == totals["buy_est_volume"] - totals["sell_est_volume"]


@pytest.mark.parametrize("bid,ask", [(10, 10), (11, 10), (0, 11), (-1, 11), (float("nan"), 11)])
def test_invalid_locked_crossed_quotes(bid, ask):
    reducer = FlowReducer()
    reducer.apply(replace(quote(), bid=bid, ask=ask))
    assert reducer.apply(event(2))["direction"] == "UNKNOWN"


def test_no_lookahead_missing_future_stale_generation_and_coarse_precision():
    reducer = FlowReducer()
    assert reducer.apply(event(1))["reason"] == "MISSING_QUOTE"
    reducer.apply(quote(2))
    result = reducer.apply(event(3))
    assert result["direction"] == "BUY_EST" and result["coarse_timestamp_ambiguity"]
    reducer.apply(replace(quote(4), broker_time=int(STAMP.timestamp()) + 1))
    assert reducer.apply(event(5))["reason"] == "FUTURE_QUOTE"
    reducer.apply(quote(6))
    assert reducer.apply(event(7, monotonic_ns=2_000_000_000))["direction"] == "UNKNOWN"
    assert reducer.apply(event(8, generation=2))["reason"] == "ORDER_OR_GENERATION"
    assert reducer.snapshot()["totals"]["buy_est_volume"] == 100


def test_separate_l1_field_ages_and_size_updates_never_refresh_prices():
    reducer = FlowReducer()
    reducer.apply(event(1, "market_data_type", tick_type=1))
    reducer.apply(replace(quote(2), field_name="bid", broker_time=None))
    reducer.apply(replace(quote(3), field_name="ask", broker_time=None))
    assert reducer.apply(event(4, feed_mode="TBT_TRADES_L1_QUOTES"))["direction"] == "BUY_EST"
    reducer.apply(replace(quote(5), field_name="bid_size", monotonic_ns=2_000_000_000))
    assert (
        reducer.apply(event(6, feed_mode="TBT_TRADES_L1_QUOTES", monotonic_ns=2_001_000_000))[
            "direction"
        ]
        == "UNKNOWN"
    )
    assert reducer.totals["trade_count"] == 2
    assert reducer.fields["bid"][1] == 2_000_000
    reducer.apply(event(7, "market_data_type", tick_type=3))
    assert reducer.apply(event(8, feed_mode="TBT_TRADES_L1_QUOTES"))["direction"] == "UNKNOWN"


@pytest.mark.parametrize(
    "values",
    [
        dict(conditions="UNINTERPRETED"),
        dict(attributes=("pastLimit",)),
        dict(attributes=("unreported",)),
        dict(tick_type=2),
        dict(broker_time=int(STAMP.timestamp()) - 10),
    ],
)
def test_nonstandard_and_late_prints_are_separate_excluded_volume(values):
    reducer = FlowReducer()
    reducer.apply(quote())
    assert reducer.apply(event(2, **values))["direction"] == "EXCLUDED"
    assert reducer.totals["excluded_volume"] == 100
    assert metrics(reducer.totals)["eligible_observed_volume"] == 0
    assert metrics(reducer.totals)["classified_volume_fraction"] is None


def test_identical_prints_survive_but_capture_sequence_cannot_be_reprocessed():
    reducer = FlowReducer()
    reducer.apply(quote())
    for seq in (2, 3):
        reducer.apply(event(seq))
    assert reducer.totals["buy_est_volume"] == 200
    with pytest.raises(ValueError, match="SEQUENCE"):
        reducer.apply(event(3))
    reducer.apply(event(4, "gap", reason="DISCONNECT"))
    assert reducer.apply(event(5))["direction"] == "UNKNOWN"
    assert reducer.snapshot()["gaps"]


def wired():
    ib = First4IB()
    ids = itertools.count(42)
    ib.client.getReqId = Mock(side_effect=lambda: next(ids))
    ib.client.reqTickByTickData = Mock()
    ib.client.cancelTickByTickData = Mock()
    ib.client.reqMktData = Mock()
    ib.client.cancelMktData = Mock()
    ib.isConnected = Mock(return_value=True)
    ib.flow_wire.started -= 20
    return ib


def test_same_packet_order_and_shared_last_ownership():
    ib = wired()
    stock = Contract(conId=1, symbol="FIX", secType="STK")
    ticker = ib.reqTickByTickData(stock, "Last", 0, False)
    records = []

    def sink(req, kind, **values):
        records.append(event(len(records) + 1, kind, request_id=req, **values))

    last = ib.flow_wire.retain_last(stock, sink)
    assert last == 42
    ib.flow_wire.requested[1] -= 16
    quotes = ib.flow_wire.quote_request(stock, MODE, sink)
    anchor_ticks = []
    ticker.updateEvent += lambda t: anchor_ticks.extend(t.tickByTicks)
    ib.wrapper.tcpDataArrived()
    ib.wrapper.tickByTickAllLast(42, 1, int(STAMP.timestamp()), 11, 100, TickAttribLast(), "X", "")
    ib.wrapper.tickByTickBidAsk(quotes, int(STAMP.timestamp()), 10, 11, 20, 30, TickAttribBidAsk())
    ib.wrapper.tickByTickAllLast(42, 1, int(STAMP.timestamp()), 11, 100, TickAttribLast(), "X", "")
    ib.wrapper.tcpDataProcessed()
    assert all(hasattr(t, "price") for t in anchor_ticks) and len(anchor_ticks) == 2
    reducer = FlowReducer()
    classifications = [r for e in records if (r := reducer.apply(e))]
    assert [r["direction"] for r in classifications] == ["UNKNOWN", "BUY_EST"]
    ib.cancelTickByTickData(stock, "Last")
    assert 42 in ib.wrapper.reqId2Ticker and ib.flow_last_retained(42)
    ib.client.cancelTickByTickData.assert_not_called()
    ib.flow_wire.release_last(1)
    ib.flow_wire.release_quote(quotes)
    assert not ib.wrapper.reqId2Ticker and not ib.flow_wire.sinks
    assert ib.client.reqTickByTickData.call_count == 2


def test_observer_cleanup_keeps_entry_and_stale_generations_cannot_cancel_new_ticker():
    ib = wired()
    stock = Contract(conId=1, secType="STK")
    ticker = ib.reqTickByTickData(stock, "Last")
    ib.flow_wire.retain_last(stock, Mock())
    ib.flow_wire.release_last(1)
    ib.client.cancelTickByTickData.assert_not_called()
    assert ib.wrapper.reqId2Ticker[42] is ticker
    ib.wrapper.reset()
    replacement = ib.wrapper.startTicker(42, stock, "Last")
    ib.flow_wire.end_last(1)
    assert ib.wrapper.reqId2Ticker[42] is replacement
    ib.client.cancelTickByTickData.assert_not_called()


def test_pacing_is_shared_across_stream_types_and_reset():
    ib = wired()
    stock = Contract(conId=1, secType="STK")
    ib.reqTickByTickData(stock, "Last")
    with pytest.raises(ValueError, match="PACING"):
        ib.flow_wire.quote_request(stock, MODE, Mock())
    ib.wrapper.reset()
    with pytest.raises(ValueError, match="PACING"):
        ib.flow_wire.retain_last(stock, Mock())
    assert ib.client.reqTickByTickData.call_count == 1


def metadata(cid="fixture"):
    return dict(
        capture_id=cid,
        session="2026-09-25",
        con_id=1,
        requested_at=STAMP.isoformat(),
        classification_version="FIRST4_QUOTE_MATCH_V1",
        config={"quote_age_ms": 1000},
    )


def test_raw_replay_summary_and_restart_segments(tmp_path):
    config = OrderFlowConfig(raw_path=tmp_path, min_free_bytes=1_000_000, flush_seconds=0.05)
    writer = FlowWriter(config)
    writer.start()
    writer.ready.wait(2)
    for cid in ("first", "restart"):
        assert writer.put(CaptureStart(cid, json.dumps(metadata(cid))))
        for e in (quote(), event(2), event(3), event(4, "end", reason="RESTART_GAP")):
            assert writer.put(replace(e, capture_id=cid))
    writer.close()
    assert not writer.error
    evidence = read_flow(tmp_path, "2026-09-25", 1)
    assert len(evidence["captures"]) == 2 and len(evidence["bars"]) == 2
    result = replay(tmp_path, "2026-09-25", 1)
    assert all(c["saved_totals_match"] for c in result["captures"])
    assert all(c["trade_records"] == 2 for c in result["captures"])
    assert all(c["replayed"]["gaps"] for c in result["captures"])
    assert all(
        c["replayed"]["bars"][0]["cumulative_volume_delta"] == 200 for c in result["captures"]
    )


def test_queue_overflow_is_bounded_and_terminal(tmp_path):
    writer = FlowWriter(OrderFlowConfig(raw_path=tmp_path, queue_events=16, batch_events=8))
    for i in range(16):
        assert writer.put(event(i + 1))
    assert not writer.put(event(17))
    assert writer.high_water == 16 and writer.dropped == 1
    assert "QUEUE_OVERFLOW" in writer.error
    assert not writer.put(event(18))


def test_storage_failure_never_touches_execution_ledger(tmp_path):
    root = tmp_path / "is-a-file"
    root.write_text("retained")
    writer = FlowWriter(OrderFlowConfig(raw_path=root))
    writer.start()
    writer.ready.wait(2)
    writer.close()
    assert "STORAGE_ERROR" in writer.error and root.read_text() == "retained"
    assert not writer.put(event(1))


@pytest.mark.parametrize(
    "updates",
    [
        dict(max_stocks=5),
        dict(order_authoritative=True),
        dict(may_submit_orders=True),
        dict(reserved_tbt=3),
        dict(quote_age_ms=0),
    ],
)
def test_config_cannot_acquire_order_authority_or_steal_execution_reserve(updates):
    with pytest.raises(ValidationError):
        OrderFlowConfig(**updates)
    assert not First4Config().order_flow.enabled


def test_capacity_deterministic_and_modes_explicit():
    assert OrderFlowConfig(available_tbt=5).stock_capacity() == 0
    assert OrderFlowConfig(available_tbt=8).stock_capacity() == 2
    assert OrderFlowConfig(available_tbt=12).stock_capacity() == 4
    assert (
        OrderFlowConfig(
            available_tbt=8, feed_mode="TBT_TRADES_L1_QUOTES", available_l1=20
        ).stock_capacity()
        == 4
    )


def observer(tmp_path, monkeypatch, capacity=8):
    ib = wired()
    config = First4Config(
        order_flow=OrderFlowConfig(
            enabled=True,
            available_tbt=capacity,
            raw_path=tmp_path / "flow",
            flush_seconds=0.05,
            min_free_bytes=1_000_000,
        )
    )
    store = Store(tmp_path / "ledger.sqlite3")
    b = NS(ib=ib, data_generation=0, upstream_available=True, market_data_block={}, data_problem="")
    runtime = NS(config=config, broker=b, store=store, session="2026-09-25", running=True)
    flow = FlowObserver(runtime)
    flow.session = runtime.session
    flow.writer = FlowWriter(config.order_flow)
    flow.writer.start()
    flow.writer.ready.wait(2)
    clock = [STAMP]
    monkeypatch.setattr(observer_module, "utc_now", lambda: clock[0])
    for slot in range(1, 5):
        data = dict(
            session="2026-09-25",
            con_id=slot,
            symbol=f"S{slot}",
            slot=slot,
            information_at=STAMP.isoformat(),
            entry_at=(STAMP + timedelta(minutes=1)).isoformat(),
            close_at=(STAMP + timedelta(minutes=30)).isoformat(),
            outcome="ENTRY_UNFILLED",
        )
        flow.allocate(data, Contract(conId=slot, symbol=f"S{slot}", secType="STK"))
    return flow, ib, clock


def test_lifecycle_failed_entries_early_close_capacity_reconnect_and_error_isolation(
    tmp_path, monkeypatch
):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)
        await flow.step()
        assert [c.state for c in flow.captures.values()][2:] == ["CAPACITY_LIMITED"] * 2
        assert ib.client.reqTickByTickData.call_count == 2
        for key in (1, 2):
            ib.flow_wire.requested[key] -= 16
        await flow.step()
        assert ib.client.reqTickByTickData.call_count == 4
        c = flow.captures[1]
        flow.error(c.quote_request, 10189, "Synthetic missing entitlement")
        await flow.step()
        assert c.state == "ENTITLEMENT_MISSING" and c.terminal
        assert flow.broker.market_data_block == {}
        count = ib.client.reqTickByTickData.call_count
        await flow.step()
        assert ib.client.reqTickByTickData.call_count == count
        c2 = flow.captures[2]
        flow.broker.data_generation += 1
        await flow.step()
        assert c2.ended_at and c2.state == "STALE_OR_DISCONNECTED"
        old = c2.sequence
        c2.emit(43, "trade", price=10, size=1)
        assert c2.sequence == old
        ib.flow_wire.requested[2] -= 16
        await flow.step()
        assert c2.segments == 2
        clock[0] += timedelta(minutes=30)
        await flow.step()
        assert c2.terminal and c2.reason == "REGULAR_SESSION_ENDED"
        assert not ib.flow_wire.sinks and not ib.wrapper.reqId2Ticker
        flow.writer.close()
        assert not flow.writer.error
        assert (
            read_flow(tmp_path / "flow", "2026-09-25", 3)["captures"][0]["end_reason"]
            == "EXECUTION_RESERVED_SLOT_BUDGET"
        )

    asyncio.run(check())


def test_existing_entry_fixture_is_identical_with_observer_and_keeps_capture(monkeypatch):
    from test_first4_anchor import BASELINE, deliver, execution

    async def run(enabled):
        runtime, ib, clock, allocated, stock = execution(monkeypatch)
        records = []

        async def chain(*args):
            if enabled:
                ib.flow_wire.retain_last(
                    stock, lambda *args, **kwargs: records.append((args, kwargs))
                )
                ib.flow_wire.requested[stock.conId] -= 16
                ib.client.getReqId.return_value = 43
                req = ib.flow_wire.quote_request(stock, MODE, lambda *args, **kwargs: None)
                ib.wrapper.tickByTickBidAsk(
                    req, int(BASELINE.timestamp()), 1, 1.25, 5, 5, TickAttribBidAsk()
                )
            deliver(ib, BASELINE, BASELINE)
            clock[0] = BASELINE
            return []

        runtime.broker.chain.side_effect = chain
        await runtime.execute(allocated, stock)
        assert runtime.broker.enter.call_args.args[2] == 1.25
        if enabled:
            assert records and ib.flow_last_retained(42)
            # Entry cleanup removed only its callback/future, leaving the canonical trade request.
            assert not ib.wrapper._futures and 42 in ib.wrapper.reqId2Ticker
            ib.flow_wire.release_last(stock.conId)
        return runtime.store.outcome.call_args_list, runtime.broker.enter.call_args_list

    disabled = asyncio.run(run(False))
    enabled = asyncio.run(run(True))
    assert disabled == enabled


def test_quote_only_minute_is_zero_observed_volume_and_missing_interval_is_absent():
    reducer = FlowReducer()
    reducer.apply(quote())
    reducer.apply(event(2, "quote", received_at=(STAMP + timedelta(minutes=2)).isoformat()))
    bars = reducer.snapshot()["bars"]
    assert len(bars) == 2 and all(b["trade_count"] == 0 for b in bars)
    assert all(b["classified_volume_fraction"] is None for b in bars)
    assert bars[1]["minute"] == (STAMP + timedelta(minutes=2)).isoformat()


def test_writer_quota_failure_keeps_only_durable_totals(tmp_path):
    config = OrderFlowConfig(
        raw_path=tmp_path, max_storage_bytes=1_000_000, min_free_bytes=1_000_000, flush_seconds=0.05
    )
    writer = FlowWriter(config)
    writer.start()
    writer.ready.wait(2)
    writer.put(CaptureStart("fixture", json.dumps(metadata())))
    writer.put(event(1))
    writer.close()
    assert "STORAGE_LIMIT" in writer.error and writer.uncommitted_events == 2
    assert not list(tmp_path.glob("*.jsonl"))
    assert not read_flow(tmp_path, "2026-09-25", 1)["captures"]


def test_replay_recovers_unindexed_raw_without_fabricating_summary(tmp_path):
    reducer = FlowReducer()
    e = event(1)
    (tmp_path / "fixture.jsonl").write_text(
        json.dumps({"capture": metadata()})
        + "\n"
        + json.dumps({"event": e.record(), "classification": reducer.apply(e)})
        + "\n"
    )
    result = replay(tmp_path, "2026-09-25", 1)["captures"][0]
    assert result["summary_missing"] and result["saved_totals_match"] is None
    assert result["replayed"]["totals"]["unknown_volume"] == 100


def test_shared_competing_session_error_still_blocks_existing_broker(tmp_path, monkeypatch):
    from stocker_execution.first4_broker import PaperBroker

    flow, ib, clock = observer(tmp_path, monkeypatch)
    broker = PaperBroker(First4Config(), flow.runtime.store, ib=ib)
    flow.broker = broker
    asyncio.run(flow.step())
    ib.errorEvent += flow.error
    request = flow.captures[1].trade_request
    ib.errorEvent.emit(request, 10197, "Synthetic competing live session", None)
    assert broker.market_data_block["code"] == 10197
    assert broker.data_generation > 0
    for capture in flow.captures.values():
        flow.finish(capture, "TEST_END")
    flow.writer.close()
    ib.errorEvent -= flow.error


def test_one_stock_request_failure_does_not_stop_other_captures(tmp_path, monkeypatch):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)

        def request(req, contract, *args):
            if contract.conId == 1:
                raise ValueError("synthetic per-stock request rejection")

        ib.client.reqTickByTickData.side_effect = request
        await flow.step()
        assert flow.captures[1].terminal
        assert flow.captures[2].trade_request is not None
        assert not flow.problem and not flow.writer.error
        assert 1 not in ib.flow_wire.lasts
        for c in flow.captures.values():
            flow.finish(c, "TEST_END")
        flow.writer.close()
        assert not flow.writer.error

    asyncio.run(check())


def test_gap_limit_is_per_stock_and_late_callbacks_cannot_reopen_capture(tmp_path, monkeypatch):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)
        await flow.step()
        c = flow.captures[1]
        for _ in range(64):
            c.emit(-1, "gap", reason="synthetic stale")
            c.emit(-1, "resume", reason="synthetic fresh")
        assert c.terminal and c.reason == "CAPTURE_GAP_LIMIT"
        seq = c.sequence
        c.emit(-1, "gap", reason="after end")
        assert c.sequence == seq
        await flow.step()
        assert flow.captures[2].trade_request is not None and not flow.writer.error
        for c in flow.captures.values():
            flow.finish(c, "TEST_END")
        flow.writer.close()
        assert not flow.writer.error

    asyncio.run(check())


def test_reconnect_segment_sort_uses_segment_time_and_never_relabels_old_totals(
    tmp_path, monkeypatch
):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)
        await flow.step()
        c = flow.captures[1]
        old_id = c.capture_id
        c.emit(c.trade_request, "trade", tick_type=1, price=10, size=17)
        flow.finish(c, "DISCONNECT")
        clock[0] += timedelta(minutes=1)
        ib.flow_wire.requested[1] -= 16
        await flow.start_capture(c)
        assert c.capture_id != old_id
        # First flush may still contain old capture evidence. Never relabel it.
        v = await flow.view("2026-09-25", 1)
        assert v.get("totals", {}).get("unknown_volume", 0) == 0
        c.emit(c.trade_request, "trade", tick_type=1, price=10, size=23)
        for item in flow.captures.values():
            flow.finish(item, "TEST_END")
        flow.writer.close()
        rows = read_flow(tmp_path / "flow", "2026-09-25", 1)["captures"]
        assert rows[-1]["capture_id"] == c.capture_id
        assert rows[-1]["totals"]["unknown_volume"] == 23
        assert rows[0]["requested_at"] == rows[-1]["requested_at"]
        assert all(
            row["saved_minutes_match"]
            for row in replay(tmp_path / "flow", "2026-09-25", 1)["captures"]
        )

    asyncio.run(check())


def test_partial_first_minute_and_inactive_duration_do_not_claim_full_coverage(
    tmp_path, monkeypatch
):
    config = First4Config(order_flow=OrderFlowConfig(enabled=True, raw_path=tmp_path))
    flow = FlowObserver(NS(config=config, broker=NS()))
    reducer = FlowReducer()
    received = STAMP + timedelta(seconds=59)
    reducer.apply(replace(quote(), received_at=received.isoformat()))
    state = dict(metadata(), **reducer.snapshot())
    bars = [dict(b, capture_id="fixture") for b in state.pop("bars")]
    monkeypatch.setattr(
        observer_module, "read_flow", lambda *args: {"captures": [state], "bars": bars}
    )
    monkeypatch.setattr(observer_module, "utc_now", lambda: STAMP + timedelta(minutes=1))
    result = asyncio.run(flow.view("2026-09-25", 1))
    assert result["rolling_1m"]["partial_coverage"]
    assert result["coverage_duration_seconds"] == 0
    monkeypatch.setattr(observer_module, "utc_now", lambda: STAMP + timedelta(hours=2))
    assert asyncio.run(flow.view("2026-09-25", 1))["coverage_duration_seconds"] == 0


def test_open_stale_gap_marks_every_minute_until_explicit_resume():
    reducer = FlowReducer()
    reducer.apply(quote())
    reducer.apply(
        event(2, "gap", received_at=(STAMP + timedelta(seconds=30)).isoformat(), reason="STALE")
    )
    for minute in range(1, 7):
        reducer.apply(
            event(minute + 2, received_at=(STAMP + timedelta(minutes=minute)).isoformat())
        )
    assert all(bar["has_gap"] for bar in reducer.snapshot()["bars"])
    assert reducer.gaps[0]["end_at"] is None
    resume_at = (STAMP + timedelta(minutes=7)).isoformat()
    reducer.apply(event(9, "resume", received_at=resume_at))
    reducer.apply(replace(quote(10), received_at=resume_at))
    assert reducer.gaps[0]["end_at"] == resume_at
    assert not reducer.snapshot()["bars"][-1]["has_gap"]


def test_gap_limit_reason_survives_real_freshness_step(tmp_path, monkeypatch):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)
        await flow.step()
        for con_id in (1, 2):
            ib.flow_wire.requested[con_id] -= 16
        await flow.step()
        c = flow.captures[1]
        c.gap_count = 63
        c.first_received_at = c.last_trade_at = c.last_quote_at = STAMP.isoformat()
        clock[0] += timedelta(seconds=31)
        await flow.step()
        assert c.terminal and c.reason == "CAPTURE_GAP_LIMIT"
        assert (await flow.view("2026-09-25", 1))["reason"] == "CAPTURE_GAP_LIMIT"
        for capture in flow.captures.values():
            flow.finish(capture, "TEST_END")
        flow.writer.close()
        assert not flow.writer.error

    asyncio.run(check())


def test_session_reset_releases_old_observer_subscriptions(tmp_path, monkeypatch):
    async def check():
        flow, ib, clock = observer(tmp_path, monkeypatch)
        await flow.step()
        flow.runtime.session = "2026-09-28"
        await flow.step()
        assert not flow.captures and not ib.flow_wire.sinks and not ib.flow_wire.lasts
        flow.writer.close()
        assert not flow.writer.error
        assert (
            read_flow(tmp_path / "flow", "2026-09-25", 1)["captures"][0]["end_reason"]
            == "SESSION_RESET"
        )

    asyncio.run(check())
