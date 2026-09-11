"""Disconnected acceptance rehearsal using real runtime, method, admission and reads.

Only market/broker input seams are fake. The frozen MODEL_T0 fixture supplies an
eligible context; the no-trade case supplies valid prints that never break a level.
Virtual timings are deterministic workload assumptions, not measured IBKR latency.
"""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import pytest

from stocker_core.acquisition import ACQUISITION_EXPERIMENT_V1
from stocker_core.markets import MarketId, get_market
from stocker_core.runs import Environment, RunRiskConfig
from stocker_dashboard.read_service import DashboardReadService
from stocker_execution.acquisition_store import AcquisitionStore
from stocker_execution.candidate_pipeline import CandidatePipeline, CandidateStore
from stocker_execution.discovery import DiscoveryRow
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.expected_move import ExpectedMoveResult, ExpectedMoveStatus
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import HistoricalBar, IbkrConnection
from stocker_execution.runtime import RuntimeStore, StockerRuntime
from stocker_execution.scanner_acquisition import ScannerAcquisition, acquisition_scans
from stocker_execution.session_hard_method import TradeEvent
from stocker_execution.session_hard_structure_d import SignalStatus
from stocker_execution.stage5 import (
    STAGE5_HV_CALCULATION_VERSION,
    Stage5Analyzer,
    Stage5CurrentDataService,
    Stage5SnapshotStore,
)
from stocker_execution.strategy_factory import MethodServices
from test_candidate_pipeline import Source, setup_run
from test_scanner_acquisition import Broker, capabilities
from test_stage8_runtime import FakeBroker, MutableClock, TriggerContextProvider


@pytest.mark.parametrize("trade", [True, False], ids=["eligible-trade", "valid-no-break"])
def test_opening_to_dashboard_rehearsal(tmp_path, trade):
    async def scenario():
        config, instance, session = setup_run(MarketId.US_NASDAQ, count=7497)
        run = instance.config.model_copy(
            update={
                "risk": RunRiskConfig(
                    risk_per_trade=0.001,
                    max_concurrent_positions=5,
                    max_gross_notional=100000,
                )
            }
        )
        config = config.model_copy(update={"runs": (run,)})
        clock = MutableClock(session.opens_at - timedelta(minutes=1))
        path = tmp_path / "rehearsal.sqlite"
        all_scans = asyncio.Event()
        first_pair = asyncio.Barrier(2)
        plans = acquisition_scans(
            ACQUISITION_EXPERIMENT_V1,
            get_market(MarketId.US_NASDAQ),
            capabilities(MarketId.US_NASDAQ),
        )
        component_index = {plan.component_id: i for i, plan in enumerate(plans)}
        scanner_times = []

        class OpeningBroker(Broker):
            async def acquisition_scan(self, request, audit):
                self.requests.append(request)
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                try:
                    if len(self.requests) <= 2:
                        await first_pair.wait()
                    clock.now += timedelta(milliseconds=250)
                    scanner_times.append(clock.now)
                    if len(scanner_times) == 105:
                        all_scans.set()
                    start = component_index[request.component_id] * 50 + 1
                    rows = tuple(
                        DiscoveryRow(i, f"S{i}", "SMART", None, "USD", "STK", rank, {})
                        for rank, i in enumerate(range(start, start + 50))
                    )
                    audit.update(row_count=50, latency_ms=250)
                    return rows
                finally:
                    self.active -= 1

            async def qualify_discovery_candidate(self, row):
                # No qualification can finish until ALL three 35-component sweeps
                # have returned: holding a scanner slot here deterministically deadlocks.
                await all_scans.wait()
                clock.now += timedelta(milliseconds=10)
                return await super().qualify_discovery_candidate(row)

        class OpeningHistory(Source):
            async def prefix(self, identity, expected, due):
                clock.now += timedelta(milliseconds=5)
                return await super().prefix(identity, expected, due)

        async def wait(due):
            clock.now = max(clock.now, due)

        acquisition = AcquisitionStore(path)
        scanner = OpeningBroker(MarketId.US_NASDAQ)
        provider = ScannerAcquisition(scanner, acquisition, clock, wait)
        history = OpeningHistory()
        pipeline = CandidatePipeline(CandidateStore(path), provider, history, clock)
        t0 = session.opens_at + timedelta(minutes=30)

        class RequiredHistory(IbkrConnection):
            def __init__(self):
                self.calls = []

            async def historical_bars(
                self,
                requested,
                *,
                bar_size,
                duration,
                what_to_show,
                regular_trading_hours,
                end_time=None,
                minimum_bars=1,
            ):
                assert what_to_show == "TRADES" and regular_trading_hours
                self.calls.append((requested.con_id, bar_size))
                current = HistoricalBar(t0, 100, 100, 100, 100, 10)
                if bar_size == "5 mins":
                    return (current,)
                assert bar_size == "1 min"
                return (
                    HistoricalBar(t0 - timedelta(minutes=3), 101.6, 101.6, 101.6, 101.6, 10),
                    current,
                )

        class ExpectedMove:
            def __init__(self):
                self.prepared = set()

            async def prepare_expected_move(self, requested, *, market, session, t0):
                assert clock.now < t0
                self.prepared.add(requested.con_id)

            async def get_expected_move(self, requested, *, market, session, t0):
                assert requested.con_id in self.prepared
                return ExpectedMoveResult(
                    status=ExpectedMoveStatus.READY,
                    expected_absolute_return_15m=0.02,
                    source="IBKR_TEST_ONLY_HISTORY_FIXTURE",
                    observation_timestamp=t0 - timedelta(days=1),
                    calculation_version="TEST_ONLY_EXPECTED_MOVE",
                    reason="isolated fixture",
                )

        required_history, expected_move = RequiredHistory(), ExpectedMove()
        features = Stage5CurrentDataService(
            required_history,
            IbkrHistoryCache(tmp_path / "history.sqlite"),
            expected_move,
            calculation_version=STAGE5_HV_CALCULATION_VERSION,
            clock=clock,
        )

        class Context(TriggerContextProvider):
            async def context_for(self, run, rows, checkpoint, instruments, cohort_history):
                context = await super().context_for(
                    run, rows, checkpoint, instruments, cohort_history
                )
                return replace(context, required_history_ready=frozenset(r.con_id for r in rows))

        class Prints:
            trade_errors = {}

            def __init__(self):
                self.started = {}
                self.observed = set()
                self.break_con_id = None

            def prepare_trades(self, instrument):
                self.started.setdefault(instrument.con_id, clock.now)

            def release_unused_trades(self, retained):
                pass  # Test evidence retained; no physical resources exist.

            def trade_stream_status(self, instrument, *, t0):
                if self.started.get(instrument.con_id, t0 + timedelta(seconds=1)) > t0:
                    return "CAUSAL_TRADES_PREFIX_UNAVAILABLE"
                return (
                    "VALID_CAUSAL_STREAM"
                    if instrument.con_id in self.observed
                    else "CAUSAL_TRADES_PREFIX_UNAVAILABLE"
                )

            async def trades_for(self, instruments, signals):
                result = {}
                for signal in signals:
                    con_id = signal.underlying_con_id
                    assert self.started[con_id] < signal.t0
                    if clock.now <= signal.t0:
                        continue
                    self.observed.add(con_id)
                    price = (
                        signal.down_trigger if trade and con_id == self.break_con_id else signal.p0
                    )
                    result[con_id] = (
                        TradeEvent(signal.t0, signal.p0, 1),
                        TradeEvent(signal.t0 + timedelta(seconds=1), price, 2),
                    )
                return result

        prints, broker = Prints(), FakeBroker()
        broker.quote_clock = clock
        snapshots = Stage5SnapshotStore(path)
        analyzer = Stage5Analyzer(features, snapshot_store=snapshots)
        context = Context()
        services = MethodServices(
            analyzer,
            context,
            prints,
            lambda market: market.checkpoint_times(),
            qualify=pipeline.qualify,
            prepare_history_on_ready=True,
            universe_lifecycle=pipeline.advance,
            universe_ready=pipeline.ready,
            universe_status=pipeline.store.summary,
            stop_universe=pipeline.stop,
        )
        runtime = StockerRuntime(
            config=config,
            ledger=ExecutionLedger(path),
            store=RuntimeStore(path),
            qualify=pipeline.qualify,
            stage5=analyzer,
            context_provider=context,
            entry_source=prints,
            method_services={run.strategy_version: services},
            broker=broker,
            expected_account="DU123456",
            clock=clock,
        )
        await runtime.start()
        await asyncio.wait_for(asyncio.gather(*pipeline.tasks.values()), 60)
        await runtime.poll_once()
        acquired_at = clock.now
        summary = acquisition.summary(run.run_id, session.session)
        assert summary["broad_membership"] == 7497
        assert summary["scanner_components"] == summary["components_complete"] == 105
        assert summary["components_failed"] == 0
        assert summary["raw_hits"] == 5250
        assert summary["scanner_unique_conids"] == summary["eligible_union"] == 1750
        assert scanner.qualifications == 1750 and scanner.maximum == 2
        assert acquired_at < session.opens_at + timedelta(minutes=5)
        assert not expected_move.prepared and not prints.started and not required_history.calls
        stages = []
        for minute, expected_input, selected_count in [(5, 1750, 250), (10, 250, 50), (15, 50, 30)]:
            clock.now = session.opens_at + timedelta(minutes=minute)
            await runtime.poll_once()
            await asyncio.gather(*pipeline.tasks.values())
            await runtime.poll_once()
            selected = pipeline.result(instance, session.session).requests
            population = pipeline.store.population(run.run_id, session.session, len(stages))
            assert len(population) == selected_count
            assert sum(n == minute for _, n in history.calls) == expected_input
            deadline = session.opens_at + timedelta(minutes={5: 10, 10: 15, 15: 30}[minute])
            assert clock.now < deadline
            stages.append(
                {
                    "minute": minute,
                    "input": expected_input,
                    "selected": selected_count,
                    "completed": clock.now.isoformat(),
                    "deadline": deadline.isoformat(),
                }
            )
        selected_ids = {row.instrument.con_id for row in selected}
        assert len(selected_ids) == 30
        prints.break_con_id = min(selected_ids)
        await asyncio.gather(*(task for task, _ in runtime._expected_move_tasks.values()))
        assert expected_move.prepared == selected_ids
        clock.now = session.opens_at + timedelta(minutes=25)
        await runtime.poll_once()
        assert set(prints.started) == selected_ids
        clock.now = session.opens_at + timedelta(minutes=30)
        await runtime.poll_once()
        await asyncio.gather(*runtime._checkpoint_tasks.values())
        signals = runtime._strategies[run.run_id].signals
        assert len(signals) == 30
        assert {con_id for con_id, _ in required_history.calls} == selected_ids
        assert len(required_history.calls) == 60
        assert all(s.pre_move_m == pytest.approx(0.8) and s.m_price == 2 for s in signals)
        assert all(s.status is SignalStatus.WAITING_FOR_ENTRY and s.q1_eligible for s in signals)
        clock.now += timedelta(seconds=1)
        await runtime.poll_once()
        reads = DashboardReadService(
            config=config,
            runtime_status=runtime.status,
            stage5_store=snapshots,
            runtime_store=runtime._store,
            ledger=runtime._ledger,
            clock=clock,
        )
        coverage = reads.runs()[0]["feed_coverage"]
        assert (
            coverage["history_ready"]
            == coverage["evaluated"]
            == coverage["timely_valid_streams"]
            == 30
        )
        assert not coverage["skipped"]
        assert len(broker.submitted) == int(trade)
        if trade:
            plan = broker.submitted[0]
            assert plan.con_id == prints.break_con_id and plan.quantity == 100
            assert plan.stop_price == 100.6 and plan.target_price == 97.6
            record = runtime._ledger.get(plan.order_plan_id)
            assert record.status is OrderLifecycle.SUBMITTED
            assert reads.orders()["total"] == 1
            for fill_index, quantity in enumerate((40, 60)):
                fill = BrokerFill(
                    f"entry-{fill_index}",
                    101,
                    broker.account,
                    Environment.PAPER,
                    plan.con_id,
                    plan.symbol,
                    OrderAction.SELL,
                    quantity,
                    99.6,
                    clock.now,
                    0.4,
                )
                assert runtime.record_fill(fill)
                broker.positions = (
                    BrokerPosition(
                        broker.account,
                        plan.con_id,
                        plan.symbol,
                        -sum((40, 60)[: fill_index + 1]),
                        99.6,
                    ),
                )
            broker.open_orders = tuple(
                BrokerOpenOrder(
                    order_id,
                    plan.order_plan_id,
                    broker.account,
                    Environment.PAPER,
                    plan.con_id,
                    plan.symbol,
                    role,
                    OrderLifecycle.SUBMITTED,
                )
                for order_id, role in [
                    (102, OrderRole.STOP),
                    (103, OrderRole.TARGET),
                    (104, OrderRole.TIMEOUT),
                ]
            )
            broker.statuses = (
                BrokerOrderStatus(
                    101,
                    plan.order_plan_id,
                    broker.account,
                    Environment.PAPER,
                    OrderLifecycle.FILLED,
                    100,
                    0,
                ),
            )
            assert (await runtime._execution[run.run_id].reconcile()).ok
            assert len(reads.positions()) == 1
            assert runtime.record_fill(
                BrokerFill(
                    "exit",
                    103,
                    broker.account,
                    Environment.PAPER,
                    plan.con_id,
                    plan.symbol,
                    OrderAction.BUY,
                    100,
                    97.6,
                    clock.now,
                    0.4,
                )
            )
            broker.positions, broker.open_orders = (), ()
            broker.statuses = ()
            assert (await runtime._execution[run.run_id].reconcile()).ok
            assert not reads.positions()
            assert runtime._ledger.get(plan.order_plan_id).status is OrderLifecycle.CLOSED
            assert reads.trades(environment=None, start=None, end=None)["total"] == 1
        else:
            assert reads.orders()["total"] == 0 and not reads.positions()
            assert reads.trades(environment=None, start=None, end=None)["total"] == 0
            assert not runtime._ledger.active_records(Environment.PAPER, broker.account)
        print(
            "OPENING_REHEARSAL "
            + json.dumps(
                {
                    "path": "trade" if trade else "no_trade",
                    "requested_sweeps": 3,
                    "completed_sweeps": 3,
                    "components": 105,
                    "raw_candidates": 5250,
                    "qualified": 1750,
                    "acquisition_completed": acquired_at.isoformat(),
                    "acquisition_deadline": (session.opens_at + timedelta(minutes=5)).isoformat(),
                    "stages": stages,
                    "history_ready": 30,
                    "causal_ready": 30,
                    "evaluated": 30,
                    "failures": {},
                    "duplicate_hits": 3500,
                    "orders": len(broker.submitted),
                },
                sort_keys=True,
            )
        )
        await runtime.stop()

    asyncio.run(scenario())
