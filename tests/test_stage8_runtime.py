import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from execution_test_support import execution_method  # noqa: F401
from stocker_core.config import IbkrConfig, RunsConfig
from stocker_core.markets import CapBucket, MarketId, MarketUniverseSpec
from stocker_core.methods import content_hash
from stocker_core.runs import (
    CandidateScreen,
    Environment,
    RunConfig,
    RunRiskConfig,
    RunScreenConfig,
    RunWindow,
)
from stocker_core.strategies import SESSION_HARD_HV_METHOD
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.ibkr import BrokerSession, CurrentQuote, QualifiedInstrument
from stocker_execution.runtime import (
    ApplicationState,
    MarketSession,
    MarketSessionState,
    RunRuntimeState,
    RuntimeStore,
    StockerRuntime,
)
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_THRESHOLD,
    EntryBar,
    SessionHardAssessment,
    StrategyContext,
    StrategyOpportunityKey,
)
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5FeatureResult,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    Stage5Status,
)
from stocker_execution.stage7 import ExecutionDestination, ExecutionRouter

NOW = datetime(2026, 9, 2, 13, 55, tzinfo=UTC)
SESSION = date(2026, 9, 2)


pytestmark = pytest.mark.usefixtures("execution_method")


class FakeBroker:
    def __init__(
        self,
        *,
        account: str = "DU123456",
        connected: bool = False,
        connect_error: Exception | None = None,
        positions: tuple[BrokerPosition, ...] = (),
        open_orders: tuple[BrokerOpenOrder, ...] = (),
        fills: tuple[BrokerFill, ...] = (),
        statuses: tuple[BrokerOrderStatus, ...] = (),
        environment: Environment = Environment.PAPER,
    ) -> None:
        self.environment = environment
        self.configured_account = account
        self.account = account if connected else ""
        self.is_connected = connected
        self.connection_epoch = 1 if connected else 0
        self.connect_error = connect_error
        self.positions = positions
        self.open_orders = open_orders
        self.fills = fills
        self.statuses = statuses
        self.submitted: list[object] = []
        self.events: list[str] = []
        self.reconciliation_gate: asyncio.Event | None = None
        self.reconciliation_waiting: asyncio.Event | None = None
        self.quote_clock = lambda: NOW
        self.quote_bid = 99.6

    async def entry_quote(self, instrument: QualifiedInstrument) -> CurrentQuote:
        return CurrentQuote(
            instrument.symbol,
            instrument.con_id,
            self.quote_clock(),
            self.quote_bid,
            self.quote_bid + 0.01,
            None,
            None,
            1,
        )

    async def connect(self) -> BrokerSession:
        self.events.append("connect")
        if self.connect_error is not None:
            raise self.connect_error
        self.account = self.account or self.configured_account
        self.is_connected = True
        self.connection_epoch += 1
        return BrokerSession(self.environment, self.account, True)

    def disconnect(self) -> None:
        self.events.append("disconnect")
        self.is_connected = False
        self.account = ""
        self.connection_epoch += 1

    def reconfigure(self, config: IbkrConfig) -> None:
        assert not self.is_connected
        assert config.environment is self.environment
        self.events.append("reconfigure")

    async def account_state(self, *, fresh=False) -> BrokerAccountState:
        self.events.append("account_state")
        return BrokerAccountState(
            self.environment,
            self.account,
            100_000.0,
            200_000.0,
            self.is_connected,
            self.positions,
            currency="USD",
            gross_position_value=sum(abs(p.quantity * p.average_price) for p in self.positions),
        )

    async def check_order_capacity(self, plan, instrument):
        return None

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float:
        return 0.01

    async def stock_execution_rules(self, instrument):
        from stocker_execution.execution_models import StockExecutionRules

        return StockExecutionRules(1.0, 1, 1)

    async def shortable_quantity(self, instrument: QualifiedInstrument) -> float:
        return 1000000

    async def submit_protected_order(
        self, plan: object, instrument: QualifiedInstrument
    ) -> BrokerOrderIds:
        self.events.append("submit")
        self.submitted.append(plan)
        return BrokerOrderIds(101, 102, 103, 104 if getattr(plan, "deadline", None) else None)

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]:
        self.events.append("open_orders")
        if self.reconciliation_gate is not None:
            if self.reconciliation_waiting is not None:
                self.reconciliation_waiting.set()
            await self.reconciliation_gate.wait()
        return self.open_orders

    async def read_fills(self) -> tuple[BrokerFill, ...]:
        self.events.append("fills")
        return self.fills

    async def read_positions(self) -> tuple[BrokerPosition, ...]:
        self.events.append("positions")
        return self.positions

    async def read_order_statuses(
        self, *, include_completed: bool = True
    ) -> tuple[BrokerOrderStatus, ...]:
        self.events.append("statuses")
        return self.statuses


class CapturingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def warning(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def error(self, event: str, **values: object) -> None:
        self.events.append((event, values))


class FakeFeatureService:
    def __init__(
        self,
        *,
        session_offset: int = 0,
        calculation_version: str = "STAGE5_PRE_MOVE_HV_V1",
    ) -> None:
        self.calls = 0
        self.prepared: list[tuple[date, datetime, tuple[int, ...]]] = []
        self.session_offset = session_offset
        self.calculation_version = calculation_version

    async def prepare_expected_moves(
        self,
        requests: Sequence[Stage5QualifiedRequest],
        *,
        session: date,
        t0: datetime,
    ) -> None:
        self.prepared.append((session, t0, tuple(item.instrument.con_id for item in requests)))

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        self.calls += 1
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=date.fromordinal(session.toordinal() + self.session_offset),
            t0=t0,
            status=Stage5Status.READY,
            exclusion_reason="",
            p0=100.0,
            expected_absolute_return_15m=0.02,
            m_price=2.0,
            raw_open_t0_minus_3m=100.8,
            raw_open_t0=100.0,
            alignment_factor=1.0,
            aligned_pre_open=100.8,
            raw_pre_move_price=0.8,
            pre_move_m=0.8,
            calculation_version=self.calculation_version,
        )


class BlockingFeatureService(FakeFeatureService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        self.started.set()
        await self.release.wait()
        return await super().get_feature(instrument, session=session, t0=t0)


class FixedSessionResolver:
    def resolve(self, run: RunConfig, now: datetime) -> MarketSession:
        return MarketSession(
            session=SESSION,
            state=MarketSessionState.ACTIVE_SESSION,
            opens_at=datetime(2026, 9, 2, 13, 30, tzinfo=UTC),
            closes_at=datetime(2026, 9, 2, 20, 0, tzinfo=UTC),
        )


class RotatingSessionResolver:
    def resolve(self, run: RunConfig, now: datetime) -> MarketSession:
        opens_at = datetime.combine(now.date(), time(13, 30), tzinfo=UTC)
        closes_at = datetime.combine(now.date(), time(20), tzinfo=UTC)
        state = (
            MarketSessionState.BEFORE_SESSION
            if now < opens_at
            else MarketSessionState.ACTIVE_SESSION
            if now < closes_at
            else MarketSessionState.AFTER_SESSION
        )
        return MarketSession(
            session=now.date(),
            state=state,
            opens_at=opens_at,
            closes_at=closes_at,
            active_bar_starts=tuple(opens_at + timedelta(minutes=5 * index) for index in range(78)),
        )


class EmptyContextProvider:
    async def context_for(
        self,
        run: RunConfig,
        rows: object,
        checkpoint: int,
        instruments: object,
        cohort_history: object,
    ) -> StrategyContext:
        return StrategyContext(run_id=run.run_id, session_hard={})


class EmptyEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
        fetch_missing: bool = True,
    ) -> dict[int, tuple[object, ...]]:
        return {}


class TriggerContextProvider:
    def __init__(self, *, failing_run: str | None = None) -> None:
        self.failing_run = failing_run

    async def context_for(
        self,
        run: RunConfig,
        rows: object,
        checkpoint: int,
        instruments: object,
        cohort_history: object,
    ) -> StrategyContext:
        if run.run_id == self.failing_run:
            raise RuntimeError("score inputs unavailable")
        snapshots = tuple(rows)
        from test_method_package import examples

        example = examples()[0]["features"]
        values = {k: v if v is not None else float("nan") for k, v in example.items()}
        return StrategyContext(
            run_id=run.run_id,
            session_hard={
                StrategyOpportunityKey(row.con_id, row.session, row.t0): SessionHardAssessment(
                    SESSION_HARD_THRESHOLD, checkpoint
                )
                for row in snapshots
                if row.con_id is not None
            },
            whipsaw_features={
                StrategyOpportunityKey(row.con_id, row.session, row.t0): values
                for row in snapshots
                if row.con_id is not None
            },
        )


class CapacityContextProvider(TriggerContextProvider):
    async def context_for(
        self,
        run: RunConfig,
        rows: object,
        checkpoint: int,
        instruments: object,
        cohort_history: object,
    ) -> StrategyContext:
        del run, rows, checkpoint, instruments, cohort_history
        raise RuntimeError("IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE")


class TriggerEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
        fetch_missing: bool = True,
    ) -> dict[int, tuple[EntryBar, ...]]:
        t0 = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
        return {con_id: (EntryBar(t0, 99.5, 99.7, 99.4),) for con_id in instruments}


class RollingTriggerEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
        fetch_missing: bool = True,
    ) -> dict[int, tuple[EntryBar, ...]]:
        timestamp = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        return {con_id: (EntryBar(timestamp, 99.5, 99.7, 99.4),) for con_id in instruments}


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _runs(*runs: RunConfig) -> RunsConfig:
    universe_ids = {run.universe for run in runs}
    universes = tuple(
        UniverseDefinition(
            universe_id=universe_id,
            name=universe_id,
            members=(
                InstrumentReference(
                    symbol="AAPL", exchange="SMART", primary_exchange="NASDAQ", currency="USD"
                ),
            ),
            market_spec=next(
                (
                    MarketUniverseSpec(
                        market_id=run.market_id,
                        cap_bucket=run.cap_bucket,
                        cap_bucket_version=str(run.cap_bucket_version),
                    )
                    for run in runs
                    if run.universe == universe_id
                    and run.market_id is not None
                    and run.cap_bucket is not None
                ),
                None,
            ),
        )
        for universe_id in sorted(universe_ids)
    )
    return RunsConfig(universes=universes, runs=runs)


def _run(
    run_id: str = "paper-run",
    *,
    environment: Environment = Environment.PAPER,
    enabled: bool = True,
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        enabled=enabled,
        universe="NASDAQ",
        strategy="TEST_EXECUTION",
        environment=environment,
        risk=RunRiskConfig(
            max_gross_notional=1000000, risk_per_trade=0.001, max_concurrent_positions=5
        ),
        session=RunWindow(
            start=time(9, 30),
            end=time(16),
            timezone="America/New_York",
            calendar="XNYS",
        ),
    )


def _hv_run(run_id: str = "hv-run") -> RunConfig:
    spec = SESSION_HARD_HV_METHOD.specification(MarketId.US_NASDAQ)
    return _run(run_id).model_copy(
        update={
            "strategy": SESSION_HARD_HV_METHOD.config_name,
            "strategy_id": SESSION_HARD_HV_METHOD.strategy_id,
            "strategy_version": SESSION_HARD_HV_METHOD.strategy_version,
            "market_id": MarketId.US_NASDAQ,
            "cap_bucket": CapBucket.ALL,
            "cap_bucket_version": "CAP_BUCKETS_V1",
            "candidate_screen_id": "METHOD_REQUIRED_DATA",
            "candidate_screen_version": SESSION_HARD_HV_METHOD.strategy_version,
            "screen": None,
            "method_spec": spec,
            "method_spec_hash": content_hash(spec),
            "universe_snapshot": UniverseDefinition(
                universe_id="NASDAQ",
                name="NASDAQ",
                market_spec=MarketUniverseSpec(
                    market_id=MarketId.US_NASDAQ, cap_bucket=CapBucket.ALL
                ),
                members=(InstrumentReference(symbol="AAPL", exchange="SMART", currency="USD"),),
            ),
        }
    )


def _runtime(
    tmp_path: Path,
    broker: FakeBroker,
    *runs: RunConfig,
    clock: MutableClock | None = None,
    feature_service: FakeFeatureService | None = None,
    context_provider: object | None = None,
    entry_source: object | None = None,
    qualification_log: list[tuple[str, ...]] | None = None,
    live_broker: FakeBroker | None = None,
    logger: object | None = None,
    stage5_by_strategy: dict[str, Stage5Analyzer] | None = None,
) -> StockerRuntime:
    selected_runs = runs or (_run(),)
    features = feature_service or FakeFeatureService()
    clock = clock or MutableClock()
    broker.quote_clock = clock
    if live_broker is not None:
        live_broker.quote_clock = clock

    async def qualify(active_runs: object) -> Stage5QualificationResult:
        instances = tuple(active_runs)
        if qualification_log is not None:
            qualification_log.append(tuple(item.config.run_id for item in instances))
        memberships = tuple(
            Stage5Membership(instance.config.run_id, instance.config.universe)
            for instance in instances
        )
        return Stage5QualificationResult(
            (
                Stage5QualifiedRequest(
                    QualifiedInstrument("AAPL", 1000, "SMART", "NASDAQ", "USD", "STK"),
                    memberships,
                ),
            ),
            (),
        )

    broker_arguments: dict[str, object]
    if live_broker is None:
        broker_arguments = {"broker": broker, "expected_account": "DU123456"}
    else:
        broker_arguments = {
            "execution_router": ExecutionRouter(
                (
                    ExecutionDestination(Environment.PAPER, "DU123456", broker),
                    ExecutionDestination(Environment.LIVE, "U123456", live_broker),
                )
            )
        }
    return StockerRuntime(
        config=_runs(*selected_runs),
        ledger=ExecutionLedger(tmp_path / "execution.sqlite3"),
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        qualify=qualify,
        stage5=Stage5Analyzer(features),
        stage5_by_strategy=stage5_by_strategy,
        context_provider=context_provider or EmptyContextProvider(),
        entry_source=entry_source or EmptyEntrySource(),
        session_resolver=FixedSessionResolver(),
        clock=clock or MutableClock(),
        logger=logger,
        **broker_arguments,
    )


def test_entry_observation_precedes_bulk_preparation_and_refreshes_clock(tmp_path, monkeypatch):
    clock = MutableClock()
    runtime = _runtime(tmp_path, FakeBroker(), clock=clock)
    asyncio.run(runtime.start())
    events = []

    async def observe(now):
        events.append(("entries", now))

    async def prepare(now):
        events.append(("prepare", now))
        clock.now += timedelta(seconds=65)

    monkeypatch.setattr(runtime, "_observe_entries", observe)
    monkeypatch.setattr(runtime, "_prepare_upcoming_expected_moves", prepare)
    asyncio.run(runtime.poll_once())
    assert events[0] == ("entries", NOW)
    assert events[-1] == ("entries", NOW + timedelta(seconds=65))


def test_clean_startup_reconciles_before_reaching_ready(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    status = runtime.status()
    assert status.application is ApplicationState.READY
    assert status.runs[0].state is RunRuntimeState.ACTIVE
    paper = status.execution_environments[0]
    assert paper.equity == 100_000.0
    assert paper.buying_power == 200_000.0
    assert broker.events.index("connect") < broker.events.index("open_orders")
    assert broker.events.index("open_orders") < broker.events.index("positions")
    assert "submit" not in broker.events

    broker.disconnect()
    disconnected = runtime.status().execution_environments[0]
    assert disconnected.equity is None
    assert disconnected.buying_power is None


def test_slow_preparation_does_not_hold_run_controls_or_shutdown(tmp_path):
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class SlowHistory(FakeFeatureService):
            async def prepare_expected_moves(self, requests, *, session, t0):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        run = _hv_run()
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            run,
            clock=MutableClock(datetime(2026, 9, 2, 13, 56, tzinfo=UTC)),
            stage5_by_strategy={
                SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(SlowHistory())
            },
        )
        await runtime.start()
        await asyncio.wait_for(runtime.poll_once(), timeout=0.5)
        await asyncio.wait_for(started.wait(), timeout=0.5)
        disabled = run.model_copy(update={"enabled": False})
        await asyncio.wait_for(
            runtime.apply_runs_config(_runs(disabled), frozenset({run.run_id})), timeout=0.5
        )
        await asyncio.wait_for(runtime.stop(), timeout=0.5)
        assert cancelled.is_set()
        assert runtime.status().application is ApplicationState.STOPPED

    asyncio.run(scenario())


def test_activity_qualification_rotates_once_on_each_new_market_session(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = MutableClock(datetime(2026, 9, 2, 14, 0, tzinfo=UTC))
        calls: list[tuple[str, ...]] = []
        selected = _run().model_copy(
            update={
                "screen": RunScreenConfig(
                    method=CandidateScreen.ACTIVITY_SHORTLIST_V1,
                    max_results=50,
                    version="ACTIVITY_SHORTLIST_V1",
                    scheduled_active_minutes=15,
                ),
                "strategy": "SESSION_HARD_HV",
                "market_id": MarketId.US_NASDAQ,
                "cap_bucket": CapBucket.MID,
                "cap_bucket_version": "CAP_BUCKETS_V1",
                "strategy_id": "SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
                "strategy_version": "SESSION_HARD_HV_V1",
                "candidate_screen_id": "ACTIVITY_SHORTLIST_V1",
                "candidate_screen_version": "ACTIVITY_SHORTLIST_V1",
            }
        )
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            selected,
            clock=clock,
            qualification_log=calls,
        )
        runtime._session_resolver = RotatingSessionResolver()
        await runtime.start()
        assert calls == []
        assert runtime.status().runs[0].state is RunRuntimeState.DEGRADED

        clock.now = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
        await runtime.poll_once()
        await runtime.poll_once()
        assert calls == []

    asyncio.run(scenario())


def test_wrong_account_prevents_readiness(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeBroker(account="DU999999", connected=True))

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().runs[0].state is RunRuntimeState.DEGRADED
    assert "ACCOUNT" in runtime.status().runs[0].reason.upper()


@pytest.mark.parametrize("finish", ["disable", "deadline", "complete"])
def test_incremental_checkpoint_publishes_results_and_controls_survive_slow_history(
    tmp_path,
    finish,
):
    async def scenario():
        clock = MutableClock()
        stalled = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        class SlowTail(FakeFeatureService):
            async def get_feature(self, instrument, *, session, t0):
                if instrument.con_id == 1004:
                    stalled.set()
                    try:
                        await release.wait()
                    finally:
                        cancelled.set()
                return await super().get_feature(instrument, session=session, t0=t0)

        run = _hv_run()
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker, run, clock=clock, feature_service=SlowTail())
        runtime._default_method_services = replace(
            runtime._default_method_services, incremental_checkpoints=True
        )
        await runtime.start()
        original = runtime._qualification.requests[0]
        runtime._qualification = Stage5QualificationResult(
            tuple(
                replace(original, instrument=replace(original.instrument, con_id=1000 + i))
                for i in range(5)
            ),
            (),
        )
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await asyncio.wait_for(runtime.poll_once(), timeout=0.5)
        await asyncio.wait_for(stalled.wait(), timeout=2)
        assert len(runtime.store.load_signals(run.run_id)) == 4
        assert runtime.status().runs[0].evaluation_completed == 4
        await asyncio.wait_for(runtime.poll_once(), timeout=0.5)
        if finish == "disable":
            disabled = run.model_copy(update={"enabled": False})
            await asyncio.wait_for(
                runtime.apply_runs_config(_runs(disabled), frozenset({run.run_id})), timeout=0.5
            )
            await asyncio.wait_for(cancelled.wait(), timeout=0.5)
        elif finish == "deadline":
            clock.now = datetime(2026, 9, 2, 14, 5, tzinfo=UTC)
            await asyncio.wait_for(runtime.poll_once(), timeout=0.5)
            await asyncio.wait_for(cancelled.wait(), timeout=0.5)
        else:
            release.set()
        await asyncio.wait_for(asyncio.gather(*runtime._checkpoint_tasks.values()), timeout=1)
        progress = runtime.status().runs[0]
        assert progress.evaluation_state == ("COMPLETED" if finish == "complete" else "INCOMPLETE")
        assert progress.evaluation_completed == (5 if finish == "complete" else 4)
        assert runtime.store.counters(run.run_id, clock.now.date()).checkpoints_processed == (
            1 if finish == "complete" else 0
        )
        await asyncio.wait_for(runtime.stop(), timeout=0.5)
        assert cancelled.is_set()
        assert broker.submitted == []

    asyncio.run(scenario())


@pytest.mark.parametrize("hour", [12, 19])
def test_session_history_resumes_without_opening_entry_streams(tmp_path, hour):
    async def scenario():
        clock = MutableClock()
        clock.now = datetime(2026, 9, 2, hour, 0, tzinfo=UTC)
        for _restart in range(2):
            features = FakeFeatureService()
            broker = FakeBroker()
            streams = []

            class Source(EmptyEntrySource):
                def prepare_trades(self, instrument, captured=streams):
                    captured.append(instrument)

            runtime = _runtime(
                tmp_path,
                broker,
                _hv_run(),
                clock=clock,
                feature_service=features,
                entry_source=Source(),
            )
            runtime._default_method_services = replace(
                runtime._default_method_services, prepare_history_on_ready=True
            )
            await runtime.start()
            await runtime.poll_once()
            await asyncio.sleep(0)
            assert len(features.prepared) == 1
            await runtime.poll_once()
            await asyncio.sleep(0)
            assert len(features.prepared) == 1
            assert features.calls == 0  # No out-of-window qualification or entry.
            assert streams == []
            assert broker.submitted == []
            await runtime.stop()

    asyncio.run(scenario())


def test_idle_entry_poll_does_not_rewrite_unchanged_candidates(tmp_path, monkeypatch):
    async def scenario():
        clock = MutableClock()
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker, _hv_run(), clock=clock)
        await runtime.start()
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()
        saved = runtime.store.load_signals("hv-run")
        assert saved
        comparisons = []
        signal_type = type(saved[0])
        original_eq = signal_type.__eq__

        def count_same_comparison(left, right):
            if left is right:
                comparisons.append(left.signal_id)
            return original_eq(left, right)

        monkeypatch.setattr(signal_type, "__eq__", count_same_comparison)
        rewritten = []
        save = runtime.store.save_signals

        def record(signals, now):
            rewritten.extend(signals)
            save(signals, now)

        monkeypatch.setattr(runtime.store, "save_signals", record)
        await runtime.poll_once()
        assert rewritten == []
        assert comparisons == []
        assert runtime.store.load_signals("hv-run") == saved
        assert broker.submitted == []
        clock.now = datetime(2026, 9, 2, 19, 0, tzinfo=UTC)
        run_status = runtime.status().runs[0]
        assert run_status.next_checkpoint is None
        assert run_status.last_scheduled_checkpoint == datetime(2026, 9, 2, 16, 20, tzinfo=UTC)
        await runtime.stop()

    asyncio.run(scenario())


def test_broker_unavailable_leaves_execution_safely_unavailable(tmp_path: Path) -> None:
    broker = FakeBroker(connect_error=RuntimeError("gateway unavailable"))
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().broker_connected is False
    assert broker.submitted == []


def test_enabled_run_requires_explicit_market_session_before_broker_connect(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    run = RunConfig(
        run_id="missing-session",
        universe="NASDAQ",
        strategy="TEST_EXECUTION",
        environment=Environment.PAPER,
        risk=RunRiskConfig(
            max_gross_notional=1000000, risk_per_trade=0.001, max_concurrent_positions=5
        ),
    )
    runtime = _runtime(tmp_path, broker, run)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().runs[0].reason == "explicit market session required"
    assert broker.events == []


def test_multiple_runs_coexist_and_disabled_run_never_activates(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        _run("first"),
        _run("second"),
        _run("disabled", enabled=False),
    )

    asyncio.run(runtime.start())

    states = {run.run_id: run.state for run in runtime.status().runs}
    assert states == {
        "first": RunRuntimeState.ACTIVE,
        "second": RunRuntimeState.ACTIVE,
        "disabled": RunRuntimeState.DISABLED,
    }
    assert broker.events.count("account_state") == 1
    assert broker.events.count("open_orders") == 1
    assert broker.events.count("statuses") == 1
    assert broker.events.count("fills") == 1
    assert broker.events.count("positions") == 1


def test_multiple_runs_share_periodic_account_reconciliation(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            _run("first"),
            _run("second"),
            clock=clock,
        )
        await runtime.start()
        broker.events.clear()
        clock.now += timedelta(seconds=6)

        await runtime.poll_once()

        assert broker.events.count("account_state") == 1
        assert broker.events.count("open_orders") == 1
        assert broker.events.count("statuses") == 1
        assert broker.events.count("fills") == 1
        assert broker.events.count("positions") == 1
        assert all(run.state is RunRuntimeState.ACTIVE for run in runtime.status().runs)

    asyncio.run(scenario())


def test_hot_enable_prepares_run_without_replaying_missed_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        features = FakeFeatureService()
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            _run("existing"),
            _run("hot", enabled=False),
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        status = await runtime.apply_runs_config(
            _runs(_run("existing"), _run("hot", enabled=True)),
            frozenset({"hot"}),
        )
        await runtime.poll_once()

        states = {item.run_id: item.state for item in status.runs}
        assert states["hot"] is RunRuntimeState.ACTIVE
        assert features.calls == 1  # Existing run only; the newly enabled run is not replayed.
        assert (
            runtime.store.checkpoint_state("hot", SESSION, datetime(2026, 9, 2, 14, 0, tzinfo=UTC))
            is not None
        )

    asyncio.run(scenario())


def test_hot_disable_stops_future_evaluation_without_touching_broker_exposure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        features = FakeFeatureService()
        position = BrokerPosition("DU123456", 1000, "AAPL", -10, 100.0)
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        broker.positions = (position,)
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        status = await runtime.apply_runs_config(
            _runs(_run(enabled=False)),
            frozenset({"paper-run"}),
        )
        await runtime.poll_once()

        assert status.runs[0].state is RunRuntimeState.DISABLED
        assert features.calls == 0
        assert broker.positions == (position,)
        assert not hasattr(broker, "cancel_order")
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_hot_risk_change_drives_future_sizing_without_reconnecting(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=TriggerEntrySource(),
        )
        await runtime.start()
        changed = _run().model_copy(
            update={
                "risk": RunRiskConfig(
                    max_gross_notional=1000000,
                    risk_per_trade=0.0005,
                    max_concurrent_positions=2,
                )
            }
        )

        await runtime.apply_runs_config(_runs(changed), frozenset({"paper-run"}))
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()

        assert len(broker.submitted) == 1
        assert broker.submitted[0].quantity == 50
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_environment_hot_change_routes_only_future_orders_and_preserves_exposure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("switching"),
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=RollingTriggerEntrySource(),
            live_broker=live,
        )
        await runtime.start()
        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.LIVE,
                host="live-gateway.local",
                port=4001,
                client_id=42,
                expected_account="U123456",
            )
        )
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()
        paper_plan = paper.submitted[0]
        runtime.record_fill(
            BrokerFill(
                "paper-entry",
                101,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderAction.SELL,
                paper_plan.quantity,
                99.5,
                clock.now,
                0.0,
            )
        )
        paper.positions = (
            BrokerPosition(
                "DU123456", paper_plan.con_id, paper_plan.symbol, -paper_plan.quantity, 99.5
            ),
        )
        paper.open_orders = (
            BrokerOpenOrder(
                102,
                paper_plan.order_plan_id,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderRole.STOP,
                OrderLifecycle.SUBMITTED,
            ),
            BrokerOpenOrder(
                103,
                paper_plan.order_plan_id,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderRole.TARGET,
                OrderLifecycle.SUBMITTED,
            ),
        )
        clock.now = datetime(2026, 9, 2, 14, 2, tzinfo=UTC)

        live_status = await runtime.apply_runs_config(
            _runs(_run("switching", environment=Environment.LIVE)),
            frozenset({"switching"}),
        )
        clock.now = datetime(2026, 9, 2, 14, 11, tzinfo=UTC)
        await runtime.poll_once()

        record = ExecutionLedger(tmp_path / "execution.sqlite3").get(paper_plan.order_plan_id)
        assert record is not None and record.environment is Environment.PAPER
        assert live_status.runs[0].environment is Environment.LIVE
        assert len(paper.submitted) == 1
        assert len(live.submitted) == 1

        paper_status = await runtime.apply_runs_config(
            _runs(_run("switching", environment=Environment.PAPER)),
            frozenset({"switching"}),
        )
        record = ExecutionLedger(tmp_path / "execution.sqlite3").get(paper_plan.order_plan_id)
        assert record is not None and record.environment is Environment.PAPER
        assert paper_status.runs[0].environment is Environment.PAPER

    asyncio.run(scenario())


def test_run_loop_reconnects_legacy_environment_after_hot_change(tmp_path: Path) -> None:
    async def scenario() -> None:
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("switching"),
            live_broker=live,
        )
        loop = asyncio.create_task(runtime.run_forever(poll_interval_seconds=0.001))
        try:
            for _ in range(200):
                if paper.is_connected:
                    break
                await asyncio.sleep(0.001)
            assert paper.is_connected
            await runtime.replace_broker_config(
                IbkrConfig(
                    environment=Environment.LIVE,
                    host="live-gateway.local",
                    port=4001,
                    client_id=42,
                    expected_account="U123456",
                )
            )
            await runtime.apply_runs_config(
                _runs(_run("switching", environment=Environment.LIVE)),
                frozenset({"switching"}),
            )
            initial_connects = paper.events.count("connect")

            paper.disconnect()
            for _ in range(200):
                if paper.events.count("connect") > initial_connects:
                    break
                await asyncio.sleep(0.001)

            assert paper.events.count("connect") == initial_connects + 1
            assert paper.is_connected
        finally:
            await runtime.stop()
            await asyncio.wait_for(loop, timeout=1.0)

    asyncio.run(scenario())


def test_custom_universe_change_reprepares_only_affected_run_without_stage5_fetch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, ...]] = []
        features = FakeFeatureService()
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            _run("custom", enabled=True).model_copy(update={"universe": "CUSTOM_US"}),
            _run("other"),
            feature_service=features,
            qualification_log=calls,
        )
        await runtime.start()
        calls.clear()
        current = _runs(
            _run("custom", enabled=True).model_copy(update={"universe": "CUSTOM_US"}),
            _run("other"),
        )
        custom = next(item for item in current.universes if item.universe_id == "CUSTOM_US")
        updated_custom = custom.model_copy(
            update={
                "members": (
                    *custom.members,
                    InstrumentReference(
                        symbol="msft",
                        exchange="smart",
                        primary_exchange="nasdaq",
                        currency="usd",
                    ),
                )
            }
        )
        changed = RunsConfig(
            universes=tuple(
                updated_custom if item.universe_id == "CUSTOM_US" else item
                for item in current.universes
            ),
            runs=current.runs,
        )

        status = await runtime.apply_runs_config(changed, frozenset({"custom"}))

        assert next(item for item in status.runs if item.run_id == "custom").state is (
            RunRuntimeState.ACTIVE
        )
        assert calls == [("custom",)]
        assert features.calls == 0

    asyncio.run(scenario())


def test_custom_universe_change_batches_shared_instrument_preparation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, ...]] = []
        custom_first = _run("custom-first").model_copy(update={"universe": "CUSTOM_US"})
        custom_second = _run("custom-second").model_copy(update={"universe": "CUSTOM_US"})
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            custom_first,
            custom_second,
            _run("other"),
            qualification_log=calls,
        )
        await runtime.start()
        calls.clear()
        current = _runs(custom_first, custom_second, _run("other"))
        custom = next(item for item in current.universes if item.universe_id == "CUSTOM_US")
        changed = RunsConfig(
            universes=tuple(
                custom.model_copy(
                    update={
                        "members": (
                            *custom.members,
                            InstrumentReference(
                                symbol="MSFT",
                                exchange="SMART",
                                primary_exchange="NASDAQ",
                                currency="USD",
                            ),
                        )
                    }
                )
                if item.universe_id == "CUSTOM_US"
                else item
                for item in current.universes
            ),
            runs=current.runs,
        )

        await runtime.apply_runs_config(changed, frozenset({"custom-first", "custom-second"}))

        assert calls == [("custom-first", "custom-second")]

    asyncio.run(scenario())


def test_unconfigured_live_environment_is_never_routed_to_paper(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker, _run("future-live", environment=Environment.LIVE))

    asyncio.run(runtime.start())

    status = runtime.status()
    assert status.application is ApplicationState.DEGRADED
    assert status.runs[0].state is RunRuntimeState.DEGRADED
    assert status.runs[0].reason == "EXECUTION_ENVIRONMENT_UNAVAILABLE"
    assert broker.events == []
    assert broker.submitted == []


def test_stop_blocks_new_work_and_preserves_broker_orders(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker)
    asyncio.run(runtime.start())

    asyncio.run(runtime.stop())

    assert runtime.status().application is ApplicationState.STOPPED
    assert runtime.status().runs[0].state is RunRuntimeState.STOPPED
    assert broker.events[-1] == "disconnect"
    assert not hasattr(broker, "cancel_order")


def test_stop_waits_for_in_flight_cycle_before_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        clock = MutableClock()
        features = BlockingFeatureService()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        polling = asyncio.create_task(runtime.poll_once())
        await features.started.wait()
        stopping = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0)

        assert broker.is_connected is True
        features.release.set()
        await polling
        await stopping
        assert broker.is_connected is False

    asyncio.run(scenario())


def test_checkpoint_is_evaluated_once_and_overlapping_runs_share_feature_work(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    features = FakeFeatureService()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _run("first"),
        _run("second"),
        clock=clock,
        feature_service=features,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())
    asyncio.run(runtime.poll_once())

    assert features.calls == 1
    assert runtime.status().counters.checkpoints_processed == 2


def test_hv_expected_move_is_prepared_before_the_checkpoint(tmp_path: Path) -> None:
    clock = MutableClock()
    hv_features = FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _hv_run(),
        clock=clock,
        stage5_by_strategy={SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(hv_features)},
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 13, 56, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert hv_features.prepared == [
        (
            SESSION,
            datetime(2026, 9, 2, 14, 0, tzinfo=UTC),
            (1000,),
        )
    ]
    assert hv_features.calls == 0


def test_hv_signal_uses_normal_stage7_paper_path_and_duplicate_protection(tmp_path: Path) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    hv_features = FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
    runtime = _runtime(
        tmp_path,
        broker,
        _hv_run(),
        clock=clock,
        stage5_by_strategy={SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(hv_features)},
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
    assert plan.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
    assert plan.environment is Environment.PAPER


@pytest.mark.parametrize("restart_second, expected_orders", [(90, 1), (120, 0)])
def test_restart_restores_hv_signal_and_executes_only_while_fresh(
    tmp_path: Path,
    restart_second,
    expected_orders,
) -> None:
    clock = MutableClock()
    first_features = FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
    first = _runtime(
        tmp_path,
        FakeBroker(),
        _hv_run(),
        clock=clock,
        stage5_by_strategy={
            SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(first_features)
        },
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(first.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(first.poll_once())
    asyncio.run(first.stop())

    second_features = FakeFeatureService(calculation_version="STAGE5_PRE_MOVE_HV_V1")
    broker = FakeBroker()
    restored = _runtime(
        tmp_path,
        broker,
        _hv_run(),
        clock=clock,
        stage5_by_strategy={
            SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(second_features)
        },
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(restored.start())
    clock.now = datetime(2026, 9, 2, 14, 0, tzinfo=UTC) + timedelta(seconds=restart_second)
    asyncio.run(restored.poll_once())

    assert first_features.calls == 1
    assert second_features.calls == 0
    assert len(broker.submitted) == expected_orders
    if broker.submitted:
        assert broker.submitted[0].strategy_version == SESSION_HARD_HV_METHOD.strategy_version


def test_late_start_marks_missed_checkpoint_without_evaluating(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 9, 2, 14, 1, tzinfo=UTC))
    features = FakeFeatureService()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        clock=clock,
        feature_service=features,
        context_provider=TriggerContextProvider(),
    )

    asyncio.run(runtime.start())
    asyncio.run(runtime.poll_once())

    assert features.calls == 0
    assert (
        runtime.store.checkpoint_state(
            "paper-run", SESSION, datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
        ).value
        == "SKIPPED_MISSED"
    )


def test_one_run_context_failure_does_not_stop_unrelated_run(tmp_path: Path) -> None:
    clock = MutableClock()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _run("broken"),
        _run("healthy"),
        clock=clock,
        context_provider=TriggerContextProvider(failing_run="broken"),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    states = {run.run_id: run.state for run in runtime.status().runs}
    assert states["broken"] is RunRuntimeState.DEGRADED
    assert states["healthy"] is RunRuntimeState.ACTIVE


def test_session_mismatched_stage5_input_is_not_traded(tmp_path: Path) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        feature_service=FakeFeatureService(session_offset=-1),
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert broker.submitted == []
    assert runtime.status().runs[0].state is RunRuntimeState.DEGRADED


def test_unexpected_position_blocks_affected_execution(tmp_path: Path) -> None:
    broker = FakeBroker(positions=(BrokerPosition("DU123456", 999, "UNKNOWN", 10, 50.0),))
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert "unexpected broker position" in runtime.status().runs[0].reason
    assert broker.submitted == []


def test_unexpected_order_blocks_affected_execution(tmp_path: Path) -> None:
    unknown = BrokerOpenOrder(
        999,
        "external-plan",
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderRole.ENTRY,
        OrderLifecycle.SUBMITTED,
    )
    runtime = _runtime(tmp_path, FakeBroker(open_orders=(unknown,)))

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert "unexpected broker open order" in runtime.status().runs[0].reason


def test_disconnect_blocks_work_and_reconnect_requires_reconciliation(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=TriggerEntrySource(),
        )
        await runtime.start()
        broker.is_connected = False
        broker.account = ""
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        await runtime.poll_once()

        assert runtime.status().application is ApplicationState.DEGRADED
        assert broker.submitted == []
        gate = asyncio.Event()
        waiting = asyncio.Event()
        broker.reconciliation_gate = gate
        broker.reconciliation_waiting = waiting
        reconnecting = asyncio.create_task(runtime.reconnect())
        await waiting.wait()
        assert runtime.status().application is ApplicationState.RECONCILING
        await runtime.poll_once()
        assert broker.submitted == []
        gate.set()
        await reconnecting
        assert runtime.status().application is ApplicationState.READY
        assert broker.events.index("disconnect") < broker.events.index("connect", 1)
        assert broker.events[-4:] == ["open_orders", "statuses", "fills", "positions"]

    asyncio.run(scenario())


def test_reconnect_with_unknown_exposure_stays_degraded(tmp_path: Path) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        broker.is_connected = False
        broker.account = ""
        await runtime.poll_once()
        broker.positions = (BrokerPosition("DU123456", 999, "UNKNOWN", 1, 10.0),)

        await runtime.reconnect()

        assert runtime.status().application is ApplicationState.DEGRADED
        assert "unexpected broker position" in runtime.status().runs[0].reason

    asyncio.run(scenario())


def test_broker_config_change_reconnects_and_reconciles_affected_environment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        broker.events.clear()

        status = await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.PAPER,
                host="paper-gateway.local",
                port=4002,
                client_id=41,
                expected_account="DU123456",
            )
        )

        assert broker.events == [
            "disconnect",
            "reconfigure",
            "connect",
            "account_state",
            "open_orders",
            "statuses",
            "fills",
            "positions",
        ]
        paper = status.execution_environments[0]
        assert paper.environment is Environment.PAPER
        assert paper.connected is True
        assert paper.reconciled is True
        assert paper.ready is True

    asyncio.run(scenario())


def test_first_run_after_empty_start_marks_account_reconciled(tmp_path):
    async def scenario():
        broker = FakeBroker()
        disabled = _run().model_copy(update={"enabled": False})
        runtime = _runtime(tmp_path, broker, disabled)
        await runtime.start()
        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.PAPER,
                expected_account="DU123456",
                host="127.0.0.1",
                port=4002,
                client_id=21,
            )
        )
        enabled = disabled.model_copy(update={"enabled": True})
        status = await runtime.apply_runs_config(
            _runs(enabled), changed_run_ids=frozenset({enabled.run_id})
        )
        assert status.application is ApplicationState.READY
        assert status.execution_environments[0].reconciled
        assert status.execution_environments[0].ready
        assert broker.submitted == []

    asyncio.run(scenario())


def test_broker_config_changes_restart_only_the_selected_environment(tmp_path: Path) -> None:
    async def scenario() -> None:
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("paper"),
            _run("live", environment=Environment.LIVE),
            live_broker=live,
        )
        await runtime.start()
        live_events = tuple(live.events)

        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.PAPER,
                host="paper-gateway.local",
                port=4002,
                client_id=41,
                expected_account="DU123456",
            )
        )

        assert tuple(live.events) == live_events
        assert live.is_connected is True

        paper_events = tuple(paper.events)
        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.LIVE,
                host="live-gateway.local",
                port=4001,
                client_id=42,
                expected_account="U123456",
            )
        )
        assert tuple(paper.events) == paper_events
        assert paper.is_connected is True

    asyncio.run(scenario())


def test_broker_account_change_with_existing_exposure_is_rejected_without_disconnect(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        ExecutionLedger(tmp_path / "execution.sqlite3").replace_broker_snapshot(
            environment=Environment.PAPER,
            account="DU123456",
            positions=(BrokerPosition("DU123456", 1000, "AAPL", -10, 100.0),),
            open_orders=(),
            observed_at=NOW,
        )
        broker.events.clear()

        with pytest.raises(ValueError, match="while broker exposure exists"):
            await runtime.replace_broker_config(
                IbkrConfig(
                    environment=Environment.PAPER,
                    host="paper-gateway.local",
                    port=4002,
                    client_id=41,
                    expected_account="DU654321",
                )
            )

        assert broker.is_connected is True
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_deterministic_paper_flow_fills_position_and_protective_exit(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    logger = CapturingLogger()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
        logger=logger,
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert len(broker.submitted) == 1
    plan = broker.submitted[0]
    entry = BrokerFill(
        "entry-fill",
        101,
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderAction.SELL,
        plan.quantity,
        99.5,
        datetime(2026, 9, 2, 14, 1, 10, tzinfo=UTC),
        1.0,
    )
    exit_fill = BrokerFill(
        "target-fill",
        103,
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderAction.BUY,
        plan.quantity,
        97.5,
        datetime(2026, 9, 2, 14, 4, tzinfo=UTC),
        1.0,
    )

    assert runtime.record_fill(entry) is True
    assert (
        len(
            ExecutionLedger(tmp_path / "execution.sqlite3").positions(Environment.PAPER, "DU123456")
        )
        == 1
    )
    assert runtime.record_fill(exit_fill) is True
    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert record is not None
    assert record.status is OrderLifecycle.CLOSED
    assert record.realized_pnl == 2.0 * plan.quantity - 2.0
    assert (
        ExecutionLedger(tmp_path / "execution.sqlite3").positions(Environment.PAPER, "DU123456")
        == ()
    )
    assert runtime.status().counters.orders == 1
    assert runtime.status().counters.fills == 2
    account_values = [
        values["account"] for _event, values in logger.events if values.get("account") is not None
    ]
    assert "DU***456" in account_values
    assert "DU123456" not in account_values


def test_duplicate_runtime_poll_does_not_duplicate_order(tmp_path: Path) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())
    asyncio.run(runtime.poll_once())

    assert len(broker.submitted) == 1


def test_restart_restores_waiting_signal_without_reevaluating_checkpoint(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    first_broker = FakeBroker()
    first = _runtime(
        tmp_path,
        first_broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(first.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(first.poll_once())
    asyncio.run(first.stop())
    assert first_broker.submitted == []

    second_broker = FakeBroker()
    restored = _runtime(
        tmp_path,
        second_broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(restored.start())
    clock.now = datetime(2026, 9, 2, 14, 1, 30, tzinfo=UTC)
    asyncio.run(restored.poll_once())

    assert len(second_broker.submitted) == 1


def test_restart_does_not_replay_expired_waiting_signal(tmp_path: Path) -> None:
    clock = MutableClock()
    first = _runtime(
        tmp_path,
        FakeBroker(),
        clock=clock,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(first.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(first.poll_once())
    asyncio.run(first.stop())

    clock.now = datetime(2026, 9, 2, 14, 6, tzinfo=UTC)
    broker = FakeBroker()
    restored = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(restored.start())
    asyncio.run(restored.poll_once())

    assert broker.submitted == []


def _submitted_runtime(
    tmp_path: Path,
) -> tuple[StockerRuntime, FakeBroker, MutableClock, object]:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(runtime.poll_once())
    return runtime, broker, clock, broker.submitted[0]


def _open_order(plan: object, order_id: int, role: OrderRole) -> BrokerOpenOrder:
    return BrokerOpenOrder(
        order_id,
        plan.order_plan_id,
        "DU123456",
        Environment.PAPER,
        plan.con_id,
        plan.symbol,
        role,
        OrderLifecycle.SUBMITTED,
    )


def test_deadline_fill_and_closed_trade_commission_are_known_runtime_events(tmp_path):
    runtime, broker, _clock, plan = _submitted_runtime(tmp_path)
    ledger = ExecutionLedger(tmp_path / "execution.sqlite3")
    assert ledger.recover_open_order(_open_order(plan, 104, OrderRole.TIMEOUT))
    entry = _fill(
        plan,
        execution_id="entry",
        order_id=101,
        side=OrderAction.SELL,
        quantity=plan.quantity,
        price=plan.entry_reference,
    )
    close = _fill(
        plan,
        execution_id="deadline",
        order_id=104,
        side=OrderAction.BUY,
        quantity=plan.quantity,
        price=plan.entry_reference,
    )
    assert runtime.record_fill(entry)
    assert runtime.record_fill(close)
    assert runtime.status().application is ApplicationState.READY
    assert runtime.status().counters.fills == 2
    assert not runtime.record_fill(replace(close, commission=1.25))
    assert runtime.status().application is ApplicationState.READY
    assert runtime.status().counters.fills == 2
    record = ledger.get(plan.order_plan_id)
    assert record.status is OrderLifecycle.CLOSED
    assert record.exit_reason == "METHOD_DEADLINE"
    assert record.realized_pnl == -1.25
    assert len(broker.submitted) == 1


def _fill(
    plan: object,
    *,
    execution_id: str,
    order_id: int,
    side: OrderAction,
    quantity: float,
    price: float,
) -> BrokerFill:
    return BrokerFill(
        execution_id,
        order_id,
        "DU123456",
        Environment.PAPER,
        plan.con_id,
        plan.symbol,
        side,
        quantity,
        price,
        datetime(2026, 9, 2, 14, 2, tzinfo=UTC),
        0.0,
        order_plan_id=plan.order_plan_id,
    )


def test_clean_restart_with_no_exposure_reaches_ready(tmp_path: Path) -> None:
    first = _runtime(tmp_path, FakeBroker())
    asyncio.run(first.start())
    asyncio.run(first.stop())
    restarted = _runtime(tmp_path, FakeBroker())

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY


def test_restart_restores_known_pending_bracket(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    orders = (
        _open_order(plan, 101, OrderRole.ENTRY),
        _open_order(plan, 102, OrderRole.STOP),
        _open_order(plan, 103, OrderRole.TARGET),
    )
    restarted = _runtime(tmp_path, FakeBroker(open_orders=orders))

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY
    assert restarted.status().runs[0].open_positions == 0


def test_restart_restores_known_open_position_with_protection(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    assert first.record_fill(
        _fill(
            plan,
            execution_id="entry",
            order_id=101,
            side=OrderAction.SELL,
            quantity=plan.quantity,
            price=99.5,
        )
    )
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            positions=(BrokerPosition("DU123456", plan.con_id, plan.symbol, -plan.quantity, 99.5),),
            open_orders=(
                _open_order(plan, 102, OrderRole.STOP),
                _open_order(plan, 103, OrderRole.TARGET),
            ),
        ),
    )

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY
    assert restarted.status().runs[0].open_positions == 1


def test_candidate_capacity_exhaustion_preserves_known_position_and_protection(
    tmp_path: Path,
) -> None:
    first, _first_broker, clock, plan = _submitted_runtime(tmp_path)
    assert first.record_fill(
        _fill(
            plan,
            execution_id="entry",
            order_id=101,
            side=OrderAction.SELL,
            quantity=plan.quantity,
            price=99.5,
        )
    )
    asyncio.run(first.stop())
    position = BrokerPosition("DU123456", plan.con_id, plan.symbol, -plan.quantity, 99.5)
    protective_orders = (
        _open_order(plan, 102, OrderRole.STOP),
        _open_order(plan, 103, OrderRole.TARGET),
    )
    broker = FakeBroker(positions=(position,), open_orders=protective_orders)
    restarted = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=CapacityContextProvider(),
    )
    asyncio.run(restarted.start())
    broker.events.clear()
    clock.now = datetime(2026, 9, 2, 14, 11, tzinfo=UTC)

    asyncio.run(restarted.poll_once())

    assert "IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE" in restarted.status().runs[0].reason
    assert broker.positions == (position,)
    assert broker.open_orders == protective_orders
    assert broker.is_connected is True
    assert "disconnect" not in broker.events


def test_restart_restores_partially_filled_entry(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    partial = plan.quantity / 2
    assert first.record_fill(
        _fill(
            plan,
            execution_id="partial-entry",
            order_id=101,
            side=OrderAction.SELL,
            quantity=partial,
            price=99.5,
        )
    )
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            positions=(BrokerPosition("DU123456", plan.con_id, plan.symbol, -partial, 99.5),),
            open_orders=(
                _open_order(plan, 101, OrderRole.ENTRY),
                _open_order(plan, 102, OrderRole.STOP),
                _open_order(plan, 103, OrderRole.TARGET),
            ),
        ),
    )

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.PARTIALLY_FILLED


def test_restart_ingests_trade_closed_while_offline(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            fills=(
                _fill(
                    plan,
                    execution_id="offline-entry",
                    order_id=101,
                    side=OrderAction.SELL,
                    quantity=plan.quantity,
                    price=99.5,
                ),
                _fill(
                    plan,
                    execution_id="offline-target",
                    order_id=103,
                    side=OrderAction.BUY,
                    quantity=plan.quantity,
                    price=97.5,
                ),
            )
        ),
    )

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.CLOSED


def test_restart_ingests_broker_rejection_while_offline(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    rejection = BrokerOrderStatus(
        101,
        plan.order_plan_id,
        "DU123456",
        Environment.PAPER,
        OrderLifecycle.REJECTED,
        0.0,
        plan.quantity,
        "rejected while offline",
    )
    restarted = _runtime(tmp_path, FakeBroker(statuses=(rejection,)))

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED


def test_interrupted_checkpoint_is_not_replayed_after_restart(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    t0 = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    assert store.reserve_checkpoint("paper-run", SESSION, t0, 6, NOW)
    restarted = _runtime(tmp_path, FakeBroker())

    asyncio.run(restarted.start())

    assert restarted.store.checkpoint_state("paper-run", SESSION, t0) is not None
    assert restarted.store.checkpoint_state("paper-run", SESSION, t0).value == "SKIPPED_INTERRUPTED"


def test_status_has_text_and_json_ready_for_future_dashboard(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeBroker())
    asyncio.run(runtime.start())

    status = runtime.status()

    assert "Application: READY" in status.as_text()
    assert "IBKR PAPER: connected" in status.as_text()
    assert "DU***456" in status.as_text()
    assert "DU123456" not in status.as_text()
    assert status.as_dict()["ibkr"]["PAPER"]["account"] == "DU***456"
    assert status.as_dict()["runs"][0]["environment"] == "PAPER"


def test_no_signal_is_not_an_operational_failure(tmp_path: Path) -> None:
    clock = MutableClock()
    runtime = _runtime(tmp_path, FakeBroker(), clock=clock)
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert runtime.status().application is ApplicationState.READY
    assert runtime.status().runs[0].state is RunRuntimeState.ACTIVE
    assert runtime.status().runs[0].signals_today == 0


def test_preparation_does_not_block_cycles_and_late_result_cannot_undo_pause(tmp_path):
    async def scenario():
        clock = MutableClock()
        features = FakeFeatureService()
        a, b = _run("a", enabled=False), _run("b")
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            a,
            b,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        original = runtime._qualify
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(instances):
            entered.set()
            await release.wait()
            return await original(instances)

        runtime._qualify = held
        update = asyncio.create_task(
            runtime.apply_runs_config(
                _runs(a.model_copy(update={"enabled": True}), b), frozenset({"a"})
            )
        )
        await asyncio.wait_for(entered.wait(), 1)
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await asyncio.wait_for(runtime.poll_once(), 1)
        assert features.calls == 1
        paused = await asyncio.wait_for(runtime.apply_runs_config(_runs(a, b), frozenset({"a"})), 1)
        assert next(r for r in paused.runs if r.run_id == "a").state is RunRuntimeState.DISABLED
        release.set()
        await update
        assert (
            next(r for r in runtime.status().runs if r.run_id == "a").state
            is RunRuntimeState.DISABLED
        )
        assert all(m.run_id != "a" for r in runtime._qualification.requests for m in r.memberships)
        await runtime.stop()

    asyncio.run(scenario())


def test_pause_bypasses_command_and_scheduler_locks_and_survives_old_update(tmp_path):
    import yaml

    from stocker_core.config import load_runs_config
    from stocker_dashboard.controls import RunControlService

    async def scenario():
        run = _run("a")
        runtime = _runtime(tmp_path, FakeBroker(), run)
        await runtime.start()
        runs_path, broker_path = tmp_path / "runs.yaml", tmp_path / "ibkr.yaml"
        runs_path.write_text(yaml.safe_dump(_runs(run).model_dump(mode="json")))
        controls = RunControlService(runs_path, broker_path, runtime=runtime)
        async with controls._lock, runtime._cycle_lock:
            result = await asyncio.wait_for(controls.disable_run("a"), 1)
            assert result.persisted and result.runtime_applied
            assert not runtime._manager.get_run("a").config.enabled
            assert runtime.status().runs[0].state is RunRuntimeState.DISABLED
        # An older command's already-prepared enabled payload cannot undo pause.
        await runtime.apply_runs_config(_runs(run), frozenset({"a"}))
        controls._write_runs(_runs(run))
        assert not load_runs_config(runs_path).runs[0].enabled
        assert not runtime._manager.get_run("a").config.enabled
        assert runtime.status().runs[0].state is RunRuntimeState.DISABLED
        await runtime.stop()

    asyncio.run(scenario())


def test_queued_enable_cannot_undo_newer_pause(tmp_path):
    import yaml

    from stocker_dashboard.controls import RunControlService

    async def scenario():
        class ObservedLock(asyncio.Lock):
            entered = asyncio.Event()

            async def acquire(self):
                self.entered.set()
                return await super().acquire()

        run = _hv_run("a")
        runtime = _runtime(tmp_path, FakeBroker(), run)
        await runtime.start()
        runs_path = tmp_path / "runs.yaml"
        runs_path.write_text(yaml.safe_dump(_runs(run).model_dump(mode="json")))
        controls = RunControlService(runs_path, tmp_path / "ibkr.yaml", runtime=runtime)
        lock = ObservedLock()
        controls._lock = lock
        await lock.acquire()
        lock.entered.clear()
        queued = asyncio.create_task(controls.enable_run("a"))
        await lock.entered.wait()
        assert (await controls.disable_run("a")).persisted
        lock.release()
        with pytest.raises(ValueError, match="newer pause"):
            await queued
        assert not runtime._manager.get_run("a").config.enabled
        assert runtime.status().runs[0].state is RunRuntimeState.DISABLED
        # A genuinely later enable is still a supported deliberate operation.
        assert (await controls.enable_run("a")).persisted
        assert runtime._manager.get_run("a").config.enabled
        await runtime.stop()

    asyncio.run(scenario())


def test_old_preparation_cannot_replace_newer_enabled_configuration(tmp_path):
    async def scenario():
        run = _run("changing", enabled=False)
        runtime = _runtime(tmp_path, FakeBroker(), run)
        await runtime.start()
        original = runtime._qualify
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def qualify(instances):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return await original(instances)

        runtime._qualify = qualify
        enabled = run.model_copy(update={"enabled": True})
        old = asyncio.create_task(
            runtime.apply_runs_config(_runs(enabled), frozenset({run.run_id}))
        )
        await asyncio.wait_for(entered.wait(), 2)
        newer = enabled.model_copy(
            update={"risk": enabled.risk.model_copy(update={"risk_per_trade": 0.0002})}
        )
        await runtime.apply_runs_config(_runs(newer), frozenset({run.run_id}))
        applied = runtime._manager.get_run(run.run_id).config
        assert applied.risk == newer.risk and applied.enabled
        execution = runtime._execution[run.run_id]
        strategy = runtime._strategies[run.run_id]
        release.set()
        await old
        assert runtime._manager.get_run(run.run_id).config == applied
        assert runtime._execution[run.run_id] is execution
        assert execution.run_config.risk == newer.risk and execution.run_config.enabled
        assert runtime._strategies[run.run_id] is strategy
        assert runtime.status().runs[0].state is RunRuntimeState.ACTIVE
        await runtime.stop()

    asyncio.run(scenario())
