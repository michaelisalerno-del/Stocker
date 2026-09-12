"""Real ib_async decoder/wrapper, fake socket requests; never connect to a broker."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import count
from types import SimpleNamespace

import pytest

from stocker_execution.dual_feed import (
    ORDINARY,
    DualFeedConnection,
    DualFeedRecorder,
    ordinary_fields,
)
from stocker_execution.dual_feed_comparison import compare_pair, verdict
from stocker_execution.dual_feed_operator import export_report
from stocker_execution.ibkr import IbkrConnection, IbkrError, _to_ib_contract
from stocker_execution.session_hard_data import IbkrSessionDataSource
from test_ibkr_resources import config, stock
from test_method_package import candidate


@pytest.fixture
def harness(monkeypatch):
    from ib_async import IB, TickAttribLast

    # ib_async requires a loop even though this harness never opens a socket.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    method, initial = candidate()
    now = [initial.t0 - timedelta(seconds=1)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr("stocker_execution.ibkr.datetime", Clock)
    ib = IB()
    connected = [True]
    monkeypatch.setattr(ib, "isConnected", lambda: connected[0])
    ids = count(10)
    monkeypatch.setattr(ib.client, "getReqId", lambda: next(ids))
    calls = []
    for name in ("reqMktData", "reqTickByTickData", "cancelMktData", "cancelTickByTickData"):
        monkeypatch.setattr(ib.client, name, lambda *a, _name=name, **k: calls.append((_name, a)))
    orders = []
    for name in ("placeOrder", "whatIfOrderAsync"):
        monkeypatch.setattr(ib, name, lambda *a, _name=name, **k: orders.append(_name))
    broker = DualFeedConnection(config(), client=ib)
    broker._account_id = "DU123456"
    instrument = replace(stock(initial.underlying_con_id), symbol=initial.symbol)
    broker.prepare_trade_events(instrument)
    original_string = ib.wrapper.tickString
    recorder = DualFeedRecorder(broker, clock=lambda: now[0])
    recorder.attach_reference(instrument)
    recorder.acquire_ordinary(instrument)
    stream = broker._causal_trade_streams[instrument.con_id]
    reference_id = ib.wrapper.ticker2ReqId["Last"][stream[4]]
    ordinary_id = recorder.streams[(instrument.con_id, ORDINARY)].request_id

    def emit(ref=(), alt=(), received=None):
        now[0] = received or initial.t0 + timedelta(seconds=1)
        ib.wrapper.tcpDataArrived()
        ib.wrapper.lastTime = now[0]
        for timestamp, price, size in ref:
            ib.wrapper.tickByTickAllLast(
                reference_id,
                1,
                int(timestamp.timestamp()),
                price,
                size,
                TickAttribLast(),
                "NASDAQ",
                "",
            )
        for timestamp, price, size in alt:
            value = f"{price};{size};{int(timestamp.timestamp() * 1000)};1000;100;true"
            # Exercise the real decoder dispatch, not just our Python callback.
            ib.client.decoder.interpret(["46", "1", str(ordinary_id), "77", value])
        ib.wrapper.tcpDataProcessed()

    def compare():
        now[0] = initial.t0 + timedelta(minutes=5, seconds=2)
        recorder.close()
        return compare_pair(
            instrument.con_id,
            recorder.events,
            tuple(recorder.streams.values()),
            t0=initial.t0,
            end=now[0],
            initial=initial,
            integrity_errors=recorder.integrity_errors,
        )

    yield SimpleNamespace(**locals())
    recorder.close()
    loop.close()
    asyncio.set_event_loop(None)


def test_real_decoder_preserves_repeats_ties_and_production_source(harness):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    ref = [(at, h.initial.p0, 1), (at, h.initial.p0, 1), (at, h.initial.up_trigger, 2)]
    h.emit(ref, ref)
    events = h.broker.trade_events(h.instrument, t0=h.initial.t0)
    assert [e.price for e in events] == [p for _, p, _ in ref]
    assert [e.sequence for e in events] == [1, 2, 3]
    assert h.broker.trade_stream_status(h.instrument, t0=h.initial.t0) == "VALID_CAUSAL_STREAM"
    assert len(h.recorder.events) == 6
    reference_decision = h.method.observe_trades({h.instrument.con_id: events})[0]
    result = h.compare()
    assert result["classification"] == "EXACT_METHOD_MATCH"
    assert result["strict_pass"]
    assert result["ordinary"]["timestamp_ties"] == 2
    assert result["ordinary"]["repeated_adjacent_prices"] == 1
    assert result["reference_replay"]["entry_reference"] == reference_decision.entry_reference
    assert h.ib.wrapper.tickString == h.original_string
    assert h.broker.trade_events(h.instrument, t0=h.initial.t0) == events
    assert h.broker.trade_stream_status(h.instrument, t0=h.initial.t0) == "VALID_CAUSAL_STREAM"
    assert h.broker.resource_status().active_market_data_lines == 1  # caller still owns TBT
    assert not h.orders


@pytest.mark.parametrize(
    "case,expected",
    [
        ("aggregate", "EXACT_METHOD_MATCH"),
        ("missing", "REFERENCE_SIGNAL_MISSED"),
        ("extra", "ALTERNATIVE_ONLY_SIGNAL"),
        ("opposite", "OPPOSITE_DIRECTION"),
        ("timing", "SAME_METHOD_DECISION_DIFFERENT_EVENT_TIMING"),
        ("different_price", "SAME_DIRECTION_DIFFERENT_TRIGGER"),
        ("late", "REFERENCE_SIGNAL_MISSED"),
    ],
)
def test_replay_cases(harness, case, expected):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    quiet = (at, h.initial.p0, 1)
    up = (at, h.initial.up_trigger, 1)
    down = (at, h.initial.down_trigger, 1)
    reference, alternative = [quiet, up], [quiet, up]
    if case == "aggregate":
        alternative = [up]
    elif case == "missing":
        alternative = [quiet]
    elif case == "extra":
        reference = [quiet]
    elif case == "opposite":
        reference, alternative = [up, down], [down]
    elif case == "different_price":
        alternative = [quiet, (at, up[1] + 0.01, 1)]
    if case == "late":
        h.emit(reference, [quiet])
        h.emit([], [up], received=h.initial.t0 + timedelta(minutes=5, seconds=1))
    elif case == "timing":
        h.emit(reference, [quiet])
        h.emit([], [up], received=at + timedelta(milliseconds=100))
    else:
        h.emit(reference, alternative)
    result = h.compare()
    assert result["classification"] == expected
    assert not result["strict_pass"]
    assert verdict([result]) == "ORDINARY_FEED_NOT_SUITABLE"
    if case == "aggregate":
        assert len(result["events_only_in_tbt"]) == 1
        assert not result["decision_relevant_price_order_preserved"]
    if case == "opposite":
        assert result["ordinary_replay"]["entry_side"] == "SHORT"
    if case == "late":
        assert result["ordinary_events_received_at_or_after_expiry"] == 1
        assert result["ordinary_events_received_after_reference_decision"] == 1


def test_ordinary_only_opposite_print_does_not_reach_production(harness):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    h.emit([(at, h.initial.p0, 1)], [(at, h.initial.down_trigger, 1)])
    source = IbkrSessionDataSource(h.broker, None)
    event_map = h.loop.run_until_complete(
        source.trades_for(
            {h.instrument.con_id: h.instrument},
            (h.initial,),
        )
    )
    h.method.observe_trades(event_map)
    assert not h.method.signals[0].selected
    assert all(e.price == h.initial.p0 for e in event_map[h.instrument.con_id])
    assert not h.orders


def test_reuse_and_release_do_not_cancel_another_ordinary_owner(harness):
    h = harness
    key, _ = h.broker._acquire_market_data_stream(
        _to_ib_contract(h.instrument),
        generic_tick_list="375",
        market_data_type=1,
        purpose="ANOTHER_DIAGNOSTIC_CONSUMER",
    )
    h.recorder.acquire_ordinary(h.instrument)
    assert len([c for c in h.calls if c[0] == "reqMktData"]) == 1
    h.recorder.close()
    assert h.broker._active_market_data[key].consumer_count == 1
    assert not any(c[0] == "cancelMktData" for c in h.calls)
    h.broker._release_market_data_stream(key)
    assert len([c for c in h.calls if c[0] == "cancelMktData"]) == 1
    assert not any(c[0] == "cancelTickByTickData" for c in h.calls)


def test_five_reference_plus_thirty_ordinary_fit_budget_and_no_rotation(harness):
    h = harness
    # Existing fixture is one of the five paired instruments.
    for con_id in range(1, 5):
        h.broker.prepare_trade_events(stock(con_id))
        h.recorder.attach_reference(stock(con_id))
    for con_id in range(1, 30):
        assert not h.recorder.acquire_ordinary(stock(con_id)).rejection
    assert h.broker.resource_status().active_market_data_lines == 35
    with pytest.raises(IbkrError, match="TICK_BY_TICK_CAPACITY"):
        h.broker.prepare_trade_events(stock(9999))
    h.broker.config = h.broker.config.model_copy(update={"market_data_line_budget": 35})
    assert "CAPACITY" in h.recorder.acquire_ordinary(stock(9000)).rejection
    assert h.broker.resource_status().active_tick_by_tick_lines == 5
    h.recorder.close()
    h.broker.release_trade_events()
    assert h.broker.resource_status().active_market_data_lines == 0


def test_incompatible_request_rejected_without_cancelling(harness):
    h = harness
    h.broker._acquire_market_data_stream(
        _to_ib_contract(stock(900)), generic_tick_list="236", market_data_type=1, purpose="BORROW"
    )
    assert "INCOMPATIBLE" in h.recorder.acquire_ordinary(stock(900)).rejection
    h.recorder.close()
    assert h.broker.resource_status().active_market_data_lines == 2


@pytest.mark.parametrize("case", ["late_subscription", "zero", "epoch", "error", "overflow"])
def test_incomplete_evidence_fails_closed(harness, case):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    if case != "zero":
        h.emit([(at, h.initial.up_trigger, 1)], [(at, h.initial.up_trigger, 1)])
    if case == "late_subscription":
        h.recorder.streams[(h.instrument.con_id, ORDINARY)].recording_started_at = at
    if case == "epoch":
        h.broker._connection_epoch += 1
    if case == "error":
        h.ib.errorEvent.emit(h.ordinary_id, 354, "Not subscribed", _to_ib_contract(h.instrument))
    if case == "overflow":
        h.recorder.max_events = 1
        h.emit([(at, h.initial.p0, 1)], [])
        h.recorder.integrity_errors.append("TAPE_OVERFLOW")
    result = h.compare()
    assert result["classification"] == "INSUFFICIENT_DATA"
    assert not result["strict_pass"]


def test_quote_callbacks_never_fabricate_prints_and_missing_fields_stay_missing(harness):
    h = harness
    h.ib.wrapper.tickString(h.ordinary_id, 45, "1700000000")
    h.ib.wrapper.tcpDataProcessed()
    assert h.recorder.events == []
    h.ib.wrapper.tickString(h.ordinary_id, 77, "100;;;1000;100;false")
    assert h.recorder.events[-1].event_at is None
    assert h.recorder.events[-1].size is None
    assert ordinary_fields("bad")[3] == "MALFORMED_77_PAYLOAD"


def test_export_is_bounded_raw_evidence_and_honest_empty_report(harness, tmp_path):
    h = harness
    result = h.compare()
    export_report(
        tmp_path,
        {
            "release_sha": "abc",
            "session": "2026-09-12",
            "t0": "x",
            "pairs": [result],
            "verdict": verdict([result]),
        },
        h.recorder,
    )
    assert (tmp_path / "events.csv").read_text().startswith("con_id,symbol,feed")
    assert "INSUFFICIENT_DATA" in (tmp_path / "comparison.md").read_text()


def test_broker_and_production_receive_timestamp_are_distinct(harness):
    h = harness
    broker_at = h.initial.t0 + timedelta(seconds=1)
    received = broker_at + timedelta(seconds=2)
    h.emit(
        [(broker_at, h.initial.up_trigger, 1)],
        [(broker_at, h.initial.up_trigger, 1)],
        received=received,
    )
    reference = next(e for e in h.recorder.events if e.feed != ORDINARY)
    assert reference.broker_at == broker_at
    assert reference.event_at == received
    assert h.broker.trade_events(h.instrument, t0=h.initial.t0)[0].timestamp == received
    result = h.compare()
    assert result["classification"] == "EXACT_METHOD_MATCH"
    assert result["strict_pass"]
    assert result["ordinary_replay"]["method_decision_timestamp"] == received
    assert result["ordinary"]["first_broker_event_timestamp"] == broker_at


@pytest.mark.parametrize("broker_offset", [-86400, 86400])
def test_same_receipt_content_matches_despite_very_different_broker_time(harness, broker_offset):
    h = harness
    received = h.initial.t0 + timedelta(seconds=1)
    broker_at = received + timedelta(seconds=broker_offset)
    h.emit(
        [(received, h.initial.up_trigger, 1)],
        [(broker_at, h.initial.up_trigger, 1)],
        received=received,
    )
    evidence_before = tuple(h.recorder.events)
    result = h.compare()
    assert result["classification"] == "EXACT_METHOD_MATCH"
    assert result["strict_pass"]
    assert result["ordinary_replay"]["method_decision_timestamp"] == received
    assert result["ordinary"]["first_broker_event_timestamp"] == broker_at
    assert result["ordinary"]["first_method_timestamp"] == received
    # Broker-second alignment cannot identify these observations as one trade.
    assert not result["alignment_pairs"]
    assert len(result["events_only_in_tbt"]) == len(result["events_only_in_ordinary"]) == 1
    assert tuple(h.recorder.events) == evidence_before


@pytest.mark.parametrize(
    "broker_seconds,receipt_seconds,signal",
    [
        (1, -0.5, False),  # Broker crossed T0; local receipt did not.
        (-60, 1, True),  # Broker before window, packet inside.
        (301, 1, True),  # Broker after window, packet inside.
        (1, 301, False),  # Broker inside, packet after expiry cannot rescue entry.
    ],
)
def test_both_replays_follow_receipt_window_boundaries(
    harness,
    broker_seconds,
    receipt_seconds,
    signal,
):
    from stocker_execution.dual_feed_comparison import replay_method

    h = harness
    broker_at = h.initial.t0 + timedelta(seconds=broker_seconds)
    received = h.initial.t0 + timedelta(seconds=receipt_seconds)
    h.emit(
        [(broker_at, h.initial.up_trigger, 1)],
        [(broker_at, h.initial.up_trigger, 1)],
        received=received,
    )
    end = h.initial.t0 + timedelta(minutes=5, seconds=2)
    results = [
        replay_method(h.initial, [e for e in h.recorder.events if e.feed == feed], end=end)
        for feed in ("REFERENCE_TBT", ORDINARY)
    ]
    assert all(r["signal"] is signal for r in results)
    assert results[0]["method_decision_timestamp"] == results[1]["method_decision_timestamp"]
    if signal:
        assert results[1]["method_decision_timestamp"] == received


def test_repeated_packet_prints_replay_in_callback_order_not_broker_time(harness):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    prices = [h.initial.p0, h.initial.p0, h.initial.down_trigger, h.initial.up_trigger]
    ref = [(at, price, 1) for price in prices]
    alt = [(at + timedelta(milliseconds=300 - i * 100), price, 1) for i, price in enumerate(prices)]
    h.emit(ref, alt)
    result = h.compare()
    assert result["strict_pass"]
    assert result["ordinary_replay"]["entry_side"] == "SHORT"
    assert result["ordinary"]["repeated_adjacent_prices"] == 1
    assert result["ordinary"]["out_of_order_broker_times"] == 3
    assert result["ordinary"]["packet_receipt_timestamp_ties"] == 3
    assert result["ordinary"]["max_events_per_receipt_timestamp"] == 4
    assert result["ordinary"]["receipt_order_inversions"] == 0


def test_descriptive_broker_receipt_and_monotonic_metrics_remain_separate(harness):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    h.emit([(at, h.initial.up_trigger, 1)], [], received=at + timedelta(milliseconds=100))
    h.emit(
        [],
        [(at + timedelta(milliseconds=125), h.initial.up_trigger, 1)],
        received=at + timedelta(milliseconds=350),
    )
    ordinary = next(e for e in h.recorder.events if e.feed == ORDINARY)
    reference = next(e for e in h.recorder.events if e.feed != ORDINARY)
    result = h.compare()
    assert result["classification"] == "SAME_METHOD_DECISION_DIFFERENT_EVENT_TIMING"
    for stat in ("median", "p95", "maximum"):
        assert result["broker_event_lag_seconds_equal_price_aligned"][stat] == pytest.approx(0.125)
        assert result["receive_lag_seconds_equal_price_aligned"][stat] == pytest.approx(0.25)
        assert (
            result["monotonic_receipt_lag_seconds_equal_price_aligned"][stat]
            == (ordinary.received_monotonic_ns - reference.received_monotonic_ns) / 1_000_000_000
        )
    assert ordinary.event_at == ordinary.broker_at == at + timedelta(milliseconds=125)
    assert ordinary.raw["payload"].split(";")[2] == str(int(ordinary.broker_at.timestamp() * 1000))
    assert set(result["time_domains"]) == {
        "METHOD_TIME",
        "BROKER_EVENT_TIME",
        "MONOTONIC_RECEIPT_TIME",
    }


def test_criteria_hash_changes_with_timestamp_semantics():
    import hashlib

    from stocker_execution.acquisition_store import encoded
    from stocker_execution.dual_feed_comparison import CRITERIA

    digest = hashlib.sha256(encoded(CRITERIA).encode()).hexdigest()
    assert digest != "8a35d0fa4dd1628db0a6b75580aa5a7c4d62860084f212ccd0844cf7a55f5dff"
    old_conversion = dict(
        CRITERIA,
        conversion="One valid tickString 77 callback -> one TradeEvent; "
        "broker milliseconds; raw price; local receipt sequence; "
        "no deduplication, expansion, rounding or interpolation",
    )
    assert digest != hashlib.sha256(encoded(old_conversion).encode()).hexdigest()
    assert CRITERIA["version"] == "DUAL_FEED_V2_LOCAL_RECEIPT"
    assert "LOCAL PACKET RECEIPT" in CRITERIA["conversion"]


def test_out_of_order_broker_messages_and_raw_batch_values_retained(harness):
    h = harness
    early = h.initial.t0 + timedelta(seconds=1)
    late = early + timedelta(milliseconds=100)
    h.emit([], [(late, h.initial.p0, 1), (early, h.initial.p0, 2)])
    ordinary = [e for e in h.recorder.events if e.feed == ORDINARY]
    assert [e.broker_at for e in ordinary] == [late, early]
    assert [e.size for e in ordinary] == [1, 2]
    assert h.compare()["ordinary"]["out_of_order_broker_times"] == 1


def test_reconnect_close_never_releases_new_epoch_owner(harness):
    h = harness
    h.broker._cancel_all_market_data_streams()
    h.broker._connection_epoch += 1
    lease, _ = h.broker._acquire_market_data_stream(
        _to_ib_contract(h.instrument),
        generic_tick_list="375",
        market_data_type=1,
        purpose="NEW_EPOCH_OWNER",
    )
    before = len([c for c in h.calls if c[0] == "cancelMktData"])
    with pytest.raises(IbkrError, match="EPOCH"):
        h.recorder.acquire_ordinary(stock(900))
    h.recorder.close()
    assert lease in h.broker._active_market_data
    assert len([c for c in h.calls if c[0] == "cancelMktData"]) == before


def test_order_and_what_if_tripwire(harness):
    from stocker_execution.dual_feed_operator import block_order_methods

    attempts = []
    block_order_methods(harness.broker, attempts)
    for action in (
        harness.ib.placeOrder,
        harness.ib.whatIfOrderAsync,
        harness.ib.client.placeOrder,
    ):
        with pytest.raises(RuntimeError, match="forbidden"):
            action(None, None)
    assert len(attempts) == 3
    assert not harness.orders


@pytest.mark.parametrize("unsafe", ["unconfirmed", "enabled", "live", "live_port"])
def test_operator_rejects_unsafe_modes_before_connect(monkeypatch, unsafe):
    from stocker_core.runs import Environment
    from stocker_execution.dual_feed_operator import validate_operator
    from test_candidate_pipeline import setup_run

    _, instance, _ = setup_run(count=1)
    run = instance.config.model_copy(
        update={
            "enabled": unsafe == "enabled",
            "environment": Environment.LIVE if unsafe == "live" else Environment.PAPER,
        }
    )
    monkeypatch.setattr(
        "stocker_execution.dual_feed_operator.load_runs_config",
        lambda _: SimpleNamespace(runs=(run,)),
    )
    monkeypatch.setattr(
        "stocker_execution.dual_feed_operator.load_ibkr_config",
        lambda *a: config().model_copy(update={"port": 4001}),
    )
    args = SimpleNamespace(
        confirm_dedicated_paper_gateway=unsafe != "unconfirmed",
        runs_config=None,
        ibkr_config=None,
        run_id=run.run_id,
    )
    with pytest.raises(ValueError):
        validate_operator(args)
    assert run.enabled == (unsafe == "enabled")


def test_frozen_selection_is_read_only_and_does_not_change_top30(tmp_path):
    import json
    import sqlite3

    from stocker_execution.candidate_pipeline import CandidateStore
    from stocker_execution.dual_feed_operator import frozen_selection
    from stocker_execution.runtime import RuntimeStore
    from test_candidate_pipeline import Provider, setup_run

    _, instance, session = setup_run(count=30)
    path = tmp_path / "state.sqlite"
    store = CandidateStore(path)
    RuntimeStore(path)
    now = session.opens_at
    identities = asyncio.run(Provider().acquire(instance))[0]
    store.begin(instance, session, now)
    store.save_population(instance.config.run_id, session.session, identities, [], now)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE opening_candidate_sessions SET state='SESSION_HARD_ACTIVE'")
        for rank, identity in enumerate(identities, 1):
            db.execute(
                "INSERT INTO opening_candidate_stages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    instance.config.run_id,
                    str(session.session),
                    2,
                    identity.con_id,
                    "RV15",
                    1,
                    rank,
                    1,
                    "",
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps([]),
                ),
            )
    before = path.read_bytes()
    selected, cohort = frozen_selection(path, instance.config, session.session, now=now)
    assert len(selected) == 30 and not cohort
    assert path.read_bytes() == before


def test_context_json_retains_missing_nonfinite_without_imputation():
    from stocker_execution.acquisition_store import encoded
    from stocker_execution.dual_feed_operator import context_json

    assert encoded(context_json({"feature": float("nan")})) == '{"feature": {"nonfinite": "nan"}}'


def test_ordinary_entitlement_error_and_late_error_preserve_reference_owner(harness):
    h = harness
    at = h.initial.t0 + timedelta(seconds=1)
    h.emit([(at, h.initial.up_trigger, 1)], [(at, h.initial.down_trigger, 1)])
    before = h.broker.trade_events(h.instrument, t0=h.initial.t0)
    ready = h.broker.trade_stream_status(h.instrument, t0=h.initial.t0)
    h.ib.errorEvent.emit(h.ordinary_id, 354, "Not subscribed", _to_ib_contract(h.instrument))
    assert h.broker.trade_events(h.instrument, t0=h.initial.t0) == before
    assert h.broker.trade_stream_status(h.instrument, t0=h.initial.t0) == ready
    assert h.method.observe_trades({h.instrument.con_id: before})[0].side == "LONG"
    assert not any(c[0] == "cancelTickByTickData" for c in h.calls)
    assert not h.orders
    h.recorder.close()
    h.ib.errorEvent.emit(h.ordinary_id, 354, "Late rejection", _to_ib_contract(h.instrument))
    assert h.broker.trade_events(h.instrument, t0=h.initial.t0) == before
    # Actual reference errors still invalidate the causal prefix exactly as before.
    h.ib.errorEvent.emit(h.reference_id, 354, "Not subscribed", _to_ib_contract(h.instrument))
    with pytest.raises(IbkrError, match="PREFIX_UNAVAILABLE"):
        h.broker.trade_events(h.instrument, t0=h.initial.t0)


def test_production_feed_methods_are_inherited_unchanged_and_empty_is_not_promising(harness):
    for method in (
        "prepare_trade_events",
        "trade_events",
        "trade_stream_status",
        "release_trade_events",
        "submit_protected_order",
    ):
        assert getattr(DualFeedConnection, method) is getattr(IbkrConnection, method)
    assert verdict([]) == "DUAL_FEED_DIAGNOSTIC_NOT_RUN"
    assert verdict([harness.compare()]) == "DUAL_FEED_DIAGNOSTIC_NOT_RUN"


def test_missing_market_is_insufficient_not_us_default(harness):
    from stocker_execution.dual_feed_comparison import replay_method

    with pytest.raises(ValueError, match="METHOD_MARKET_UNAVAILABLE"):
        replay_method(
            replace(harness.initial, market_id=None),
            [],
            end=harness.initial.t0 + timedelta(minutes=5),
        )


def test_complete_operator_fake_session_exports_and_releases_all_lines(
    harness, monkeypatch, tmp_path
):
    import json

    from ib_async import TickAttribLast

    from stocker_execution import dual_feed_operator as operator

    h = harness
    h.recorder.close()
    h.broker.release_trade_events()
    h.now[0] = h.initial.t0 - timedelta(seconds=1)
    selected = (h.instrument, *(stock(i) for i in range(1, 30)))
    run = SimpleNamespace(run_id=h.initial.run_id, enabled=False)
    monkeypatch.setattr(operator, "datetime", h.Clock)
    monkeypatch.setattr(
        operator, "validate_operator", lambda args: (run, h.broker.config, h.initial.t0, 6)
    )
    monkeypatch.setattr(
        operator,
        "ExchangeSessionResolver",
        lambda: SimpleNamespace(resolve=lambda *args: SimpleNamespace(session=h.initial.session)),
    )
    monkeypatch.setattr(operator, "frozen_selection", lambda *a, **k: (selected, ()))
    monkeypatch.setattr(operator, "load_runs_config", lambda *a: SimpleNamespace(runs=(run,)))
    monkeypatch.setattr(operator, "DualFeedConnection", lambda *a, **k: h.broker)
    monkeypatch.setattr(
        operator,
        "DualFeedRecorder",
        lambda broker, **kwargs: DualFeedRecorder(broker, clock=lambda: h.now[0], **kwargs),
    )
    monkeypatch.setattr(h.ib.client, "reqMarketDataType", lambda *a: None)

    async def connect():
        return None

    def disconnect():
        h.broker.release_trade_events()
        h.broker._cancel_all_market_data_streams()
        h.connected[0] = False

    async def context(*args):
        args[-1][h.instrument.con_id] = h.initial

    original_sleep = asyncio.sleep

    async def sleep(delay):
        h.now[0] += timedelta(seconds=60)
        h.ib.wrapper.tcpDataArrived()
        h.ib.wrapper.lastTime = h.now[0]
        for request_id in tuple(h.ib.wrapper.ticker2ReqId["Last"].values()):
            h.ib.wrapper.tickByTickAllLast(
                request_id,
                1,
                int(h.now[0].timestamp()),
                h.initial.up_trigger,
                1,
                TickAttribLast(),
                "NASDAQ",
                "",
            )
        for request_id in tuple(h.ib.wrapper.ticker2ReqId["mktData"].values()):
            h.ib.wrapper.tickString(
                request_id,
                77,
                f"{h.initial.up_trigger};1;{int(h.now[0].timestamp() * 1000)};1;100;true",
            )
        h.ib.wrapper.tcpDataProcessed()
        await original_sleep(0)

    monkeypatch.setattr(h.broker, "connect", connect)
    monkeypatch.setattr(h.broker, "disconnect", disconnect)
    monkeypatch.setattr(operator, "prepare_contexts", context)
    monkeypatch.setattr(operator.asyncio, "sleep", sleep)
    output = h.loop.run_until_complete(
        operator.observe(
            SimpleNamespace(
                database=tmp_path / "unused",
                runs_config=None,
                ordinary_top30=True,
                output=tmp_path,
                max_events=10000,
            )
        )
    )
    report = json.loads((output / "comparison.json").read_text())
    assert report["resources_peak"]["active_market_data_lines"] == 35
    assert report["resources_after_release"]["active_market_data_lines"] == 0
    assert report["ordinary_top30_observed"]
    assert report["ORDINARY_TRADE_STREAM"]["succeeded"] == 30
    assert report["REFERENCE_TBT"]["succeeded"] == 5
    assert report["order_methods_invoked"] == 0
    assert not report["errors"]
    assert not run.enabled
    assert (output / "acceptance-criteria.json").exists()
    import hashlib

    from stocker_execution.acquisition_store import encoded
    from stocker_execution.dual_feed_comparison import CRITERIA

    criteria_bytes = (output / "acceptance-criteria.json").read_bytes()
    assert criteria_bytes == encoded(CRITERIA).encode()
    assert report["criteria_sha256"] == hashlib.sha256(criteria_bytes).hexdigest()
    assert set(report["time_domains"]) == {
        "METHOD_TIME",
        "BROKER_EVENT_TIME",
        "MONOTONIC_RECEIPT_TIME",
    }
