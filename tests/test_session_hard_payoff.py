from datetime import UTC, datetime, timedelta

import pytest

from stocker_execution.session_hard_payoff import (
    CompletedPayoff,
    CostAwareDecision,
    assess_pooled_payoff,
)

NOW = datetime(2025, 2, 20, 15, tzinfo=UTC)


def history(n=20, gross_r=0.8):
    return [CompletedPayoff(str(i), NOW - timedelta(minutes=1), gross_r) for i in range(n)]


def assess(observations, *, stop=101.0, cost=30.0, opportunity_id="current"):
    return assess_pooled_payoff(
        opportunity_id=opportunity_id,
        signal_timestamp=NOW,
        entry_reference_price=100.0,
        initial_stop_price=stop,
        estimated_round_trip_cost_bps=cost,
        observations=observations,
    )


@pytest.mark.parametrize(
    ("n", "gross_r", "cost", "expected", "net"),
    [
        (19, 0.8, 30, CostAwareDecision.WARMUP, 0.5),
        (20, 0.8, 30, CostAwareDecision.PASS, 0.5),
        (20, 0.4, 55, CostAwareDecision.FAIL, -0.15),
        (20, 0.5, 50, CostAwareDecision.FAIL, 0.0),
    ],
)
def test_frozen_hurdle(n, gross_r, cost, expected, net):
    result = assess(history(n, gross_r), cost=cost)
    assert result.decision is expected
    assert result.estimated_net_r == pytest.approx(net)
    assert result.completed_observation_count == n


def test_completion_boundary_and_overlapping_unsorted_history():
    previous = history(19)
    equal = CompletedPayoff("equal", NOW, 1000)
    later = CompletedPayoff("later", NOW + timedelta(minutes=2), -1000)
    before = CompletedPayoff("before", NOW - timedelta(microseconds=1), 0.8)
    result = assess([equal, later, *reversed(previous), before])
    assert result.completed_observation_count == 20
    assert result.estimated_gross_r == pytest.approx(0.8)
    assert result.take_trade
    assert assess([equal, later, *previous]).decision is CostAwareDecision.WARMUP


def test_current_and_future_outcomes_cannot_change_current_decision():
    expected = assess(history())
    for outcome in (-1e9, 1e9):
        contaminated = [
            CompletedPayoff("current", NOW - timedelta(days=1), outcome),
            CompletedPayoff("later", NOW + timedelta(seconds=1), outcome),
            CompletedPayoff("equal", NOW, outcome),
        ]
        assert assess([*history(), *contaminated]) == expected


def test_unknown_stock_and_stock_independence():
    # There is intentionally no symbol input and no per-stock history lookup.
    prior = [
        CompletedPayoff(f"{symbol}-{i}", NOW - timedelta(days=1), 0.8)
        for symbol in ("AAPL", "MSFT", "NVDA", "META")
        for i in range(25)
    ]
    xyz = assess(prior, opportunity_id="XYZ-new")
    aapl = assess(prior, opportunity_id="AAPL-new")
    assert xyz == aapl
    assert xyz.completed_observation_count == 100
    assert xyz.take_trade


def test_cost_depends_on_nominal_stop_distance():
    wide = assess(history(), stop=101, cost=30)
    tight = assess(history(), stop=100.25, cost=30)
    assert wide.cost_r == pytest.approx(0.3)
    assert tight.cost_r == pytest.approx(1.2)
    assert wide.take_trade
    assert not tight.take_trade


def test_duplicate_history_is_counted_once():
    prior = history(19)
    assert assess([*prior, *prior]).completed_observation_count == 19


def test_empty_history_has_no_fabricated_estimate():
    result = assess([])
    assert result.decision is CostAwareDecision.WARMUP
    assert result.estimated_gross_r is None
    assert result.estimated_net_r is None


def test_frozen_296_opportunity_ledger():
    import csv
    from pathlib import Path

    ledger = Path(__file__).parent / "fixtures/session_hard_pooled_payoff.csv"
    with ledger.open() as handle:
        rows = list(csv.DictReader(handle))
    observations = [
        CompletedPayoff(
            row["row_id"],
            datetime.fromisoformat(row["available_at"]),
            float(row["gross_R"]),
        )
        for row in rows
    ]
    taken = []
    for row in rows:
        assessment = assess_pooled_payoff(
            opportunity_id=row["row_id"],
            signal_timestamp=datetime.fromisoformat(row["signal_timestamp"]),
            entry_reference_price=float(row["entry_price"]),
            initial_stop_price=float(row["entry_price"]) + 0.5 * float(row["M_price"]),
            estimated_round_trip_cost_bps=10.0,
            observations=observations,
        )
        # The frozen research abstained during warmup. The user's revised
        # production rule trades baseline during warmup; ready decisions match.
        frozen_take = assessment.completed_observation_count >= 20 and assessment.take_trade
        assert frozen_take == (row["POOLED_PAYOFF_HURDLE"] == "True")
        if frozen_take:
            taken.append(float(row["net_R"]))
    assert len(rows) == 296
    assert len(taken) == 224
    assert sum(float(row["net_R"]) for row in rows) / len(rows) == pytest.approx(
        0.2352, abs=0.00005
    )
    assert sum(taken) / len(taken) == pytest.approx(0.3911, abs=0.00005)
    assert sum(taken) / len(rows) == pytest.approx(0.2959, abs=0.00005)
    assert sum(taken) == pytest.approx(87.60, abs=0.005)


def baseline_signal(*, t0=NOW, con_id=999, symbol="XYZ", run_id="RUN_A"):
    from dataclasses import replace

    from stocker_core.strategies import SESSION_HARD_HV_METHOD
    from stocker_execution.session_hard_structure_d import EntryBar
    from stocker_execution.strategy_factory import create_strategy
    from test_stage6_session_hard_strategy import ready_snapshot, strategy_context

    row = replace(
        ready_snapshot(pre_move_m=1.0, con_id=con_id),
        run_ids=(run_id,),
        symbol=symbol,
        t0=t0,
        session=t0.date(),
        calculation_version="STAGE5_PRE_MOVE_HV_V1",
    )
    strategy = create_strategy(
        SESSION_HARD_HV_METHOD.strategy_id,
        SESSION_HARD_HV_METHOD.strategy_version,
    )
    strategy.evaluate((row,), strategy_context((row,), {con_id: 1.0}, run_id=run_id))
    return strategy.observe_entry_bars({con_id: (EntryBar(t0, 100, 100, 99.7, 99.75),)})[0]


def register(store, signal, cost=10):
    from stocker_core.runs import Environment, RunConfig
    from stocker_execution.ibkr import QualifiedInstrument

    instrument = QualifiedInstrument(
        signal.symbol,
        signal.underlying_con_id,
        "SMART",
        "NASDAQ",
        "USD",
        "STK",
    )
    run = RunConfig(
        run_id=signal.run_id,
        universe=signal.universe_id,
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
    )
    store.register_payoff(signal, instrument, run, cost)
    return instrument, run


def seed_pool(store, *, count=20, gross_r=0.8, before=NOW):
    from stocker_execution.session_hard_payoff import pooled_opportunity_id

    for i in range(count):
        signal = baseline_signal(
            t0=before - timedelta(days=i + 1),
            con_id=10000 + i,
            symbol=f"OLD{i}",
        )
        register(store, signal)
        store.assess_payoff(signal.signal_id)
        store.complete_payoff(
            signal.signal_id,
            CompletedPayoff(
                pooled_opportunity_id(signal), signal.t0 + timedelta(minutes=2), gross_r
            ),
        )


def test_abstention_learns_and_restart_preserves_identical_decision(tmp_path):
    import json

    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_payoff import hypothetical_baseline_outcome
    from stocker_execution.session_hard_structure_d import EntryBar

    path = tmp_path / "runtime.sqlite3"
    store = RuntimeStore(path)
    seed_pool(store, count=20, gross_r=0.4)
    signal = baseline_signal()
    register(store, signal, cost=55)
    assessment = store.assess_payoff(signal.signal_id)
    assert assessment.decision is CostAwareDecision.FAIL
    outcome = hypothetical_baseline_outcome(
        signal,
        (
            EntryBar(NOW, 100, 100, 99.7, 99.75),
            EntryBar(NOW + timedelta(minutes=1), 99.7, 99.8, 98.7, 98.8),
        ),
        as_of=NOW + timedelta(minutes=2),
    )
    assert outcome is not None
    assert outcome.gross_r == pytest.approx(2.0)
    store.complete_payoff(signal.signal_id, outcome)
    later = baseline_signal(t0=NOW + timedelta(minutes=3), con_id=888, symbol="UNSEEN")
    register(store, later)
    expected = store.assess_payoff(later.signal_id)
    assert expected.completed_observation_count == 21
    assert expected.estimated_gross_r == pytest.approx((20 * 0.4 + 2) / 21)
    assert expected.take_trade
    restored = RuntimeStore(path)
    assert restored.assess_payoff(later.signal_id) == expected
    assert restored.assess_payoff(signal.signal_id) == assessment
    row = next(row for row in restored.payoff_audit() if row["signal_id"] == signal.signal_id)
    assert row["actually_executed"] == 0
    assert row["completion_timestamp"] == outcome.completion_timestamp.isoformat()
    assert json.loads(row["payload"])["baseline_eligible"] is True
    assert json.loads(row["assessment"])["decision"] == assessment.decision


def test_shadow_requires_fillable_signal_and_complete_causal_prefix():
    from dataclasses import replace

    from stocker_execution.session_hard_payoff import hypothetical_baseline_outcome
    from stocker_execution.session_hard_structure_d import EntryBar, SignalStatus

    signal = baseline_signal()
    entry = EntryBar(NOW, 100, 100, 99.7, 99.75)
    stop = EntryBar(NOW + timedelta(minutes=1), 100, 101, 98, 99)
    assert (
        hypothetical_baseline_outcome(
            signal,
            (stop,),
            as_of=NOW + timedelta(minutes=2),
        )
        is None
    )
    assert (
        hypothetical_baseline_outcome(
            signal,
            (entry, stop),
            as_of=NOW + timedelta(minutes=1),
        )
        is None
    )
    outcome = hypothetical_baseline_outcome(
        signal,
        (entry, stop),
        as_of=NOW + timedelta(minutes=2),
    )
    assert outcome.gross_r == pytest.approx(-1)  # both levels: frozen stop-first
    assert outcome.completion_timestamp == NOW + timedelta(minutes=2)
    for unfilled in (
        replace(signal, status=SignalStatus.EXPIRED),
        replace(signal, selected=False),
        replace(signal, direction="UP"),
    ):
        assert (
            hypothetical_baseline_outcome(
                unfilled,
                (entry, stop),
                as_of=NOW + timedelta(minutes=2),
            )
            is None
        )


def test_timeout_is_original_t0_plus_15_and_missing_close_never_fabricates():
    from stocker_execution.session_hard_payoff import hypothetical_baseline_outcome
    from stocker_execution.session_hard_structure_d import EntryBar

    signal = baseline_signal()
    bars = [EntryBar(NOW + timedelta(minutes=i), 99.8, 100, 99.7, 99.75) for i in range(15)]
    assert hypothetical_baseline_outcome(signal, bars, as_of=NOW + timedelta(minutes=14)) is None
    outcome = hypothetical_baseline_outcome(signal, bars, as_of=NOW + timedelta(minutes=15))
    assert outcome.completion_timestamp == NOW + timedelta(minutes=15)
    assert outcome.gross_r == pytest.approx(0.1)
    bars[-1] = EntryBar(NOW + timedelta(minutes=14), 99.8, 100, 99.7)
    assert hypothetical_baseline_outcome(signal, bars, as_of=NOW + timedelta(minutes=15)) is None


def test_original_session_hard_cannot_seed_hv_pool(tmp_path):
    from dataclasses import replace

    from stocker_core.strategies import SESSION_HARD_METHOD
    from stocker_execution.runtime import RuntimeStore

    signal = replace(
        baseline_signal(),
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
    )
    with pytest.raises(ValueError, match="only exact"):
        register(RuntimeStore(tmp_path / "runtime.sqlite3"), signal)


def test_multiple_runs_do_not_duplicate_pooled_observation(tmp_path):
    from dataclasses import replace

    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_payoff import pooled_opportunity_id

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    signal = baseline_signal()
    duplicate = replace(signal, signal_id="second-run-signal", run_id="SECOND")
    for item in (signal, duplicate):
        register(store, item)
        store.complete_payoff(
            item.signal_id,
            CompletedPayoff(
                pooled_opportunity_id(item),
                NOW + timedelta(minutes=2),
                0.8,
            ),
        )
    later = baseline_signal(t0=NOW + timedelta(minutes=3))
    register(store, later)
    assert store.assess_payoff(later.signal_id).completed_observation_count == 1


@pytest.mark.parametrize("future_low", [97.0, 99.7])
def test_runtime_abstains_without_orders_and_keeps_shadow_after_restart(tmp_path, future_low):
    import asyncio

    from stocker_core.strategies import SESSION_HARD_HV_METHOD
    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_structure_d import EntryBar
    from stocker_execution.stage5 import Stage5Analyzer
    from test_stage8_runtime import (
        CapturingLogger,
        FakeBroker,
        FakeFeatureService,
        MutableClock,
        TriggerContextProvider,
        _hv_run,
        _runtime,
    )

    t0 = datetime(2026, 9, 2, 14, tzinfo=UTC)
    seed_pool(RuntimeStore(tmp_path / "runtime.sqlite3"), gross_r=-0.5, before=t0)

    class Bars:
        async def bars_for(self, run, instruments, *, session, now, signals, fetch_missing=True):
            return {
                con_id: (
                    EntryBar(t0, 99.5, 99.7, 99.4, 99.6),
                    EntryBar(t0 + timedelta(minutes=1), 99.7, 99.8, future_low, 99.7),
                )
                for con_id in instruments
            }

    broker = FakeBroker()
    clock = MutableClock()
    logger = CapturingLogger()
    run = _hv_run()
    features = FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
    runtime = _runtime(
        tmp_path,
        broker,
        run,
        clock=clock,
        logger=logger,
        stage5_by_strategy={SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(features)},
        context_provider=TriggerContextProvider(),
        entry_source=Bars(),
    )
    asyncio.run(runtime.start())
    clock.now = t0 + timedelta(minutes=1)
    asyncio.run(runtime.poll_once())
    assert not broker.submitted
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    row = next(row for row in store.payoff_audit() if row["symbol"] == "AAPL")
    decision = store.assess_payoff(row["signal_id"])
    assert decision.decision is CostAwareDecision.FAIL
    assert row["completion_timestamp"] is None  # future bar supplied by fake cannot leak
    assert any(event == "session_hard_cost_admission" for event, _ in logger.events)
    asyncio.run(runtime.stop())

    # A changed/removed current universe cannot make a persisted shadow disappear.
    restarted = _runtime(tmp_path, FakeBroker(), clock=clock, entry_source=Bars())
    clock.now = t0 + timedelta(minutes=2)
    asyncio.run(restarted._advance_pending_payoffs(clock.now))
    reloaded = RuntimeStore(tmp_path / "runtime.sqlite3")
    assert reloaded.assess_payoff(row["signal_id"]) == decision
    completed = next(
        item["completion_timestamp"]
        for item in reloaded.payoff_audit()
        if item["signal_id"] == row["signal_id"]
    )
    assert bool(completed) == (future_low == 97.0)
    assert not broker.submitted


@pytest.mark.parametrize("n", [0, 1, 19])
def test_insufficient_history_keeps_baseline_trading(n):
    result = assess(history(n, -100), cost=500)
    assert result.decision is CostAwareDecision.WARMUP
    assert result.decision.value == "TRADE_BASELINE_WARMUP"
    assert result.take_trade


def test_runtime_with_no_history_still_submits_normal_hv_paper_order(tmp_path):
    import asyncio

    from stocker_core.strategies import SESSION_HARD_HV_METHOD
    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.stage5 import Stage5Analyzer
    from test_stage8_runtime import (
        FakeBroker,
        FakeFeatureService,
        MutableClock,
        TriggerContextProvider,
        TriggerEntrySource,
        _hv_run,
        _runtime,
    )

    broker = FakeBroker()
    clock = MutableClock()
    runtime = _runtime(
        tmp_path,
        broker,
        _hv_run(),
        clock=clock,
        stage5_by_strategy={
            SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(
                FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
            )
        },
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(runtime.poll_once())
    asyncio.run(runtime.poll_once())
    assert len(broker.submitted) == 1
    plan = broker.submitted[0]
    assert plan.strategy_id == SESSION_HARD_HV_METHOD.strategy_id
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    assert store.assess_payoff(plan.signal_id).decision is CostAwareDecision.WARMUP
    # Submission is not an actual fill and not a completed observation.
    row = store.payoff_audit()[0]
    assert row["actually_executed"] == 0
    assert row["completion_timestamp"] is None


def test_ibkr_shadow_cache_and_pre_hurdle_backfill_use_saved_identity(tmp_path):
    import asyncio
    import json
    from dataclasses import replace

    from stocker_core.runs import Environment, RunConfig
    from stocker_execution.history import IbkrHistoryCache
    from stocker_execution.ibkr import HistoricalBar, IbkrConnection, QualifiedInstrument
    from stocker_execution.runtime import IbkrSessionDataSource, RuntimeStore

    signal = baseline_signal()
    instrument = QualifiedInstrument("XYZ", 999, "SMART", "NASDAQ", "USD", "STK")
    run = RunConfig(
        run_id=signal.run_id,
        universe=signal.universe_id,
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
    )

    class Boundary(IbkrConnection):
        def __init__(self):
            self.requests = []

        async def historical_bars(self, requested, **kwargs):
            self.requests.append((requested, kwargs))
            return tuple(
                HistoricalBar(
                    timestamp=NOW + timedelta(minutes=i),
                    open=99.8,
                    high=100,
                    low=99.7,
                    close=99.75,
                    volume=100,
                )
                for i in range(15)
            )

    path = tmp_path / "runtime.sqlite3"
    cache = IbkrHistoryCache(path)
    boundary = Boundary()
    source = IbkrSessionDataSource(boundary, cache)
    args = dict(
        session=(NOW + timedelta(days=1)).date(),
        now=NOW + timedelta(days=1),
        signals=(replace(signal, baseline_eligible=True),),
    )
    cached = asyncio.run(source.bars_for(run, {999: instrument}, **args, fetch_missing=False))
    assert not cached[999]
    assert not boundary.requests
    loaded = asyncio.run(source.bars_for(run, {999: instrument}, **args))
    assert len(loaded[999]) == 15
    assert loaded[999][-1].close == 99.75
    assert boundary.requests[0][1]["end_time"] == NOW + timedelta(minutes=15)
    assert boundary.requests[0][1]["duration"] == "900 S"
    assert cache.qualified_instrument(999) == instrument

    store = RuntimeStore(path)
    store.save_signals((signal,), NOW)
    store.backfill_payoffs((run,), cache)
    store.backfill_payoffs((run,), cache)
    audit = store.payoff_audit()
    assert len(audit) == 1
    assert audit[0]["assessment"] is None
    payload = json.loads(audit[0]["payload"])
    assert payload["admission_decision"] == "TRADE_BASELINE_PRE_HURDLE"
    assert payload["estimated_round_trip_cost_bps"] is None
    restored_signal, restored_instrument, _ = store.pending_payoffs()[0]
    assert restored_signal.baseline_eligible
    assert restored_instrument == instrument


def test_old_shadow_download_never_blocks_normal_paper_submission(tmp_path):
    import asyncio

    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_structure_d import EntryBar
    from test_stage8_runtime import (
        FakeBroker,
        MutableClock,
        TriggerContextProvider,
        _runtime,
    )

    t0 = datetime(2026, 9, 2, 14, tzinfo=UTC)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    register(store, baseline_signal(t0=t0 - timedelta(days=1)))

    async def scenario():
        download_started = asyncio.Event()

        class Bars:
            async def bars_for(
                self,
                run,
                instruments,
                *,
                session,
                now,
                signals,
                fetch_missing=True,
            ):
                if any(signal.baseline_eligible for signal in signals):
                    if fetch_missing:
                        download_started.set()
                        await asyncio.Event().wait()  # permanently unavailable history
                    return {}
                return {con_id: (EntryBar(t0, 99.5, 99.7, 99.4),) for con_id in instruments}

        broker = FakeBroker()
        clock = MutableClock()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=Bars(),
        )
        await runtime.start()
        clock.now = t0 + timedelta(minutes=1)
        await asyncio.wait_for(runtime.poll_once(), timeout=1)
        assert len(broker.submitted) == 1
        await asyncio.wait_for(download_started.wait(), timeout=1)
        await asyncio.wait_for(runtime.stop(), timeout=1)

    asyncio.run(scenario())


def test_bad_shadow_record_does_not_prevent_other_outcomes(tmp_path, monkeypatch):
    import asyncio

    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_structure_d import EntryBar
    from test_stage8_runtime import FakeBroker, MutableClock, _runtime

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    bad = baseline_signal(con_id=101)
    good = baseline_signal(con_id=102)
    register(store, bad)
    register(store, good)
    with store._connect() as connection:
        connection.execute(
            "UPDATE runtime_session_hard_payoffs SET payload = ? WHERE signal_id = ?",
            ("invalid json", bad.signal_id),
        )

    class Bars:
        async def bars_for(
            self,
            run,
            instruments,
            *,
            session,
            now,
            signals,
            fetch_missing=True,
        ):
            return {
                con_id: (
                    EntryBar(NOW, 100, 100, 99.7, 99.75),
                    EntryBar(NOW + timedelta(minutes=1), 99.7, 99.8, 98.7, 98.8),
                )
                for con_id in instruments
            }

    runtime = _runtime(tmp_path, FakeBroker(), clock=MutableClock(), entry_source=Bars())
    asyncio.run(runtime._advance_pending_payoffs(NOW + timedelta(minutes=2)))
    audit = {row["signal_id"]: row for row in store.payoff_audit()}
    assert audit[good.signal_id]["gross_r"] == pytest.approx(2)
    assert audit[bad.signal_id]["completion_timestamp"] is None


def test_all_simultaneous_admissions_freeze_before_first_broker_await(tmp_path):
    import asyncio

    from stocker_core.strategies import SESSION_HARD_HV_METHOD
    from stocker_execution.execution_models import BrokerOrderIds
    from stocker_execution.runtime import RuntimeStore
    from stocker_execution.session_hard_payoff import pooled_opportunity_id
    from stocker_execution.stage5 import Stage5Analyzer
    from test_stage8_runtime import (
        FakeBroker,
        FakeFeatureService,
        MutableClock,
        TriggerContextProvider,
        TriggerEntrySource,
        _hv_run,
        _runtime,
    )

    t0 = datetime(2026, 9, 2, 14, tzinfo=UTC)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    seed_pool(store, count=19, before=t0)
    pending = baseline_signal(t0=t0 - timedelta(hours=1), con_id=12345)
    register(store, pending)

    class Broker(FakeBroker):
        async def submit_protected_order(self, plan, instrument):
            self.submitted.append(plan)
            # Simulate a prior background completion during the broker await.
            store.complete_payoff(
                pending.signal_id,
                CompletedPayoff(
                    pooled_opportunity_id(pending),
                    pending.t0 + timedelta(minutes=2),
                    -100,
                ),
            )
            order_id = len(self.submitted) * 100
            return BrokerOrderIds(order_id, order_id + 1, order_id + 2)

    broker = Broker()
    clock = MutableClock()
    runtime = _runtime(
        tmp_path,
        broker,
        _hv_run("a"),
        _hv_run("b"),
        clock=clock,
        stage5_by_strategy={
            SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(
                FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
            )
        },
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = t0 + timedelta(minutes=1)
    asyncio.run(runtime.poll_once())
    current = [row for row in store.payoff_audit() if row["symbol"] == "AAPL"]
    assert len(current) == 2
    assessments = [store.assess_payoff(row["signal_id"]) for row in current]
    assert {item.completed_observation_count for item in assessments} == {19}
    assert all(item.take_trade for item in assessments)
    assert len(broker.submitted) == 2
