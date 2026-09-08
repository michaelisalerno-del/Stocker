import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from execution_test_support import execution_method  # noqa: F401
from stocker_core.methods import ARTIFACTS, SESSION_HARD, content_hash, get_method, verified_q1_spec
from stocker_core.runs import Environment, RunRiskConfig
from stocker_core.strategies import installed_strategies
from stocker_dashboard.universe_runs import UniverseRunBuilder
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.runtime import RuntimeStore
from stocker_execution.session_hard_method import FrozenWhipsawModel, SessionHardMethod, TradeEvent
from stocker_execution.session_hard_structure_d import (
    SignalStatus,
    StrategyOpportunityKey,
    nominal_exit_prices,
)
from test_stage6_session_hard_strategy import ready_snapshot, strategy_context


def examples():
    from pathlib import Path

    return json.loads((Path(__file__).parent / "fixtures/session_hard_model_t0.json").read_text())


def candidate(low=True, clock=None):
    example = examples()[0 if low else 1]
    values = {k: v if v is not None else float("nan") for k, v in example["features"].items()}
    row = ready_snapshot(pre_move_m=values["PRE_MOVE_M"])
    method = SessionHardMethod(clock=clock or (lambda: row.t0))
    # The pure model fixtures use the frozen missing-cohort preprocessing path.
    values["cohort_percentile"] = float("nan")
    context = strategy_context((row,), {row.con_id: values["score"]})
    key = StrategyOpportunityKey(row.con_id, row.session, row.t0)
    context = replace(context, whipsaw_features={key: values}, available_at=row.t0)
    signal = method.evaluate((row,), context)[0]
    return method, signal


def test_only_current_method_and_no_cap_control():
    assert [m.label for m in installed_strategies()] == ["Session HARD"]
    options = UniverseRunBuilder().options()
    assert "capitalisation" not in options
    assert len(options["strategies"]) == 1
    with pytest.raises(ValueError, match="historical-only"):
        get_method(SESSION_HARD.method_id, "SESSION_HARD_HV_V1")


def test_frozen_model_reproduces_saved_fit_predictions():
    model = FrozenWhipsawModel()
    for row in examples():
        values = {k: v if v is not None else float("nan") for k, v in row["features"].items()}
        assert model.score(values) == row["probability"]


def test_fit_cutoff_quantile_and_inclusive_equality():
    spec = verified_q1_spec()
    fit = json.loads((ARTIFACTS / "fit_score_distribution.json").read_text())
    cutoff = float(np.quantile([r["risk_score"] for r in fit], 0.20, method="linear"))
    assert cutoff == spec["q1_risk_cutoff"] == 0.22995371253992852
    assert len(fit) == 931
    model = FrozenWhipsawModel()
    assert model.admits(cutoff)
    assert not model.admits(np.nextafter(cutoff, np.inf))


def test_q1_veto_precedes_arming():
    _, signal = candidate(False)
    assert signal.q1_eligible is False
    assert signal.armed_at is None
    assert signal.status is SignalStatus.NOT_QUALIFIED
    assert signal.reason == "MODEL_T0_Q1_VETO"


def test_arming_timestamp_follows_model_scoring_completion(monkeypatch):
    now = [ready_snapshot(pre_move_m=1.0).t0]
    original = FrozenWhipsawModel.score

    def delayed_score(self, features):
        result = original(self, features)
        now[0] += timedelta(seconds=2)
        return result

    monkeypatch.setattr(FrozenWhipsawModel, "score", delayed_score)
    method, signal = candidate(clock=lambda: now[0])
    assert signal.armed_at == signal.t0 + timedelta(seconds=2)
    first_break = TradeEvent(signal.t0 + timedelta(seconds=1), signal.up_trigger, 1)
    result = method.observe_trades({signal.underlying_con_id: (first_break,)})[0]
    assert result.reason == "FIRST_BREAK_PRECEDED_ARMING"


@pytest.mark.parametrize("direction", [1, -1])
def test_first_actual_print_selects_direction_permanently(direction):
    method, signal = candidate()
    assert signal.status is SignalStatus.WAITING_FOR_ENTRY
    level = signal.up_trigger if direction == 1 else signal.down_trigger
    other = signal.down_trigger if direction == 1 else signal.up_trigger
    events = (
        TradeEvent(signal.t0 + timedelta(seconds=1), level, 1),
        TradeEvent(signal.t0 + timedelta(seconds=2), other, 2),
    )
    triggered = method.observe_trades({signal.underlying_con_id: events})[0]
    assert triggered.side == ("LONG" if direction == 1 else "SHORT")
    assert triggered.entry_reference == level
    assert method.observe_trades({signal.underlying_con_id: events}) == ()
    stop, target = nominal_exit_prices(triggered)
    assert stop == level - direction * 0.5 * signal.m_price
    assert target == level + direction * signal.m_price
    assert triggered.deadline == signal.t0 + timedelta(minutes=15)


def test_completed_minute_cannot_select_current_direction():
    method, _ = candidate()
    with pytest.raises(ValueError, match="ordered TRADES"):
        method.observe_entry_bars({})


def test_method_state_roundtrip_and_historical_payload(tmp_path):
    method, signal = candidate()
    store = RuntimeStore(tmp_path / "state.sqlite")
    store.save_signals((signal,), signal.t0)
    restored = store.load_signals(signal.run_id)
    assert restored == (signal,)
    resumed = SessionHardMethod()
    resumed.restore_signals(restored)
    event = TradeEvent(signal.t0 + timedelta(seconds=1), signal.up_trigger, 1)
    assert resumed.observe_trades({signal.underlying_con_id: (event,)})[0].side == "LONG"
    legacy = replace(
        signal,
        strategy_version="SESSION_HARD_HV_V1",
        signal_id="historical",
        run_id="historical",
        method_spec_hash=None,
        artifact_hashes={},
        deadline=None,
    )
    store.save_signals((legacy,), signal.t0)
    assert store.load_signals("historical")[0].strategy_version == "SESSION_HARD_HV_V1"


def test_method_owns_universe_and_run_provenance(tmp_path):
    from stocker_core.config import RunsConfig
    from stocker_core.universes import InstrumentReference, UniverseDefinition

    config = RunsConfig(
        universes=(
            UniverseDefinition(
                universe_id="US_ALL",
                name="US",
                members=(InstrumentReference(symbol="AAPL", exchange="SMART", currency="USD"),),
            ),
        )
    )
    config, run = UniverseRunBuilder().add(
        config,
        market_id="US_ALL",
        strategy_id=SESSION_HARD.method_id,
        strategy_version=SESSION_HARD.version,
        environment=Environment.PAPER,
        risk=RunRiskConfig(risk_per_trade=0.001),
    )
    assert run.screen is None
    assert run.method_spec["universe_search"]["cap_constraint"] is None
    assert run.method_spec_hash == content_hash(run.method_spec)
    assert run.method_spec["artifact_hashes"]["model"] == verified_q1_spec()["model_sha256"]
    _, signal = candidate()
    store = RuntimeStore(tmp_path / "state.sqlite")
    store.save_method_run(run, signal.t0)
    saved = store.method_run(run.run_id)
    assert saved["configuration"]["method_spec_hash"] == run.method_spec_hash
    assert RuntimeStore(store.path).method_run(run.run_id) == saved
    # Existing DB schema is additive and idempotent.
    ExecutionLedger(store.path)
    ExecutionLedger(store.path)


def test_missing_or_modified_model_fails_without_bypass(tmp_path, monkeypatch):
    import stocker_core.methods as methods

    monkeypatch.setattr(methods, "ARTIFACTS", tmp_path)
    with pytest.raises(FileNotFoundError):
        FrozenWhipsawModel()
    (tmp_path / "prospective_q1.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenWhipsawModel()


def test_nonfinite_predictors_use_exact_saved_missing_preprocessing():
    model = FrozenWhipsawModel()
    values = {k: v if v is not None else float("nan") for k, v in examples()[0]["features"].items()}
    values["cohort_percentile"] = float("nan")
    expected = model.score(values)
    values["cohort_percentile"] = float("inf")
    assert model.score(values) == expected


def test_first_break_before_veto_is_available_expires_without_retroactive_entry():
    method, signal = candidate()
    method._signals[signal.signal_id] = replace(signal, armed_at=signal.t0 + timedelta(seconds=2))
    events = (
        TradeEvent(signal.t0 + timedelta(seconds=1), signal.up_trigger, 1),
        TradeEvent(signal.t0 + timedelta(seconds=3), signal.down_trigger, 2),
    )
    result = method.observe_trades({signal.underlying_con_id: events})[0]
    assert result.reason == "FIRST_BREAK_PRECEDED_ARMING"
    assert not result.selected


def test_missing_event_prefix_expires_only_the_affected_candidate():
    method, signal = candidate()
    other = replace(signal, signal_id="other", underlying_con_id=999)
    method._signals[other.signal_id] = other
    method.expire_unobservable(signal.underlying_con_id, "CAUSAL_TRADES_PREFIX_UNAVAILABLE")
    assert method._signals[signal.signal_id].status is SignalStatus.EXPIRED
    assert method._signals["other"].status is SignalStatus.WAITING_FOR_ENTRY


def test_q1_veto_does_not_change_the_historical_cohort_population(tmp_path):
    from stocker_execution.session_hard_structure_d import EntryBar

    method, signal = candidate(False)
    assert signal.q1_eligible is False
    label = method.cohort_labels[signal.signal_id]
    level = label.p0 - 0.2 * label.m_price
    method.observe_cohort_bars(signal.signal_id, (EntryBar(label.t0, level, level, level),))
    opportunity = method.cohort_opportunities[0]
    store = RuntimeStore(tmp_path / "cohort.sqlite")
    store.save_cohort(signal.signal_id, opportunity)
    method.acknowledge_cohort_update(signal.signal_id)
    assert method.cohort_opportunities == ()
    assert store.cohort_history(signal.run_id) == (opportunity,)
    assert method._signals[signal.signal_id].q1_eligible is False


@pytest.mark.parametrize("long", [True, False])
def test_protected_timeout_order_uses_method_deadline_and_direction(long):
    import asyncio
    from datetime import UTC, datetime

    from stocker_execution.execution_models import OrderAction
    from stocker_execution.ibkr import IbkrConnection
    from test_stage7_ibkr import FakeOrderClient, _config, _instrument, _plan

    client = FakeOrderClient()
    broker = IbkrConnection(_config(), client=client, execution_enabled=True)
    asyncio.run(broker.connect())
    deadline = datetime.now(UTC) + timedelta(minutes=15)
    plan = replace(
        _plan(),
        deadline=deadline,
        side=OrderAction.BUY if long else OrderAction.SELL,
        stop_price=99 if long else 101,
        target_price=102 if long else 98,
    )
    ids = asyncio.run(broker.submit_protected_order(plan, _instrument()))
    orders = [r[1] for r in client.placed]
    assert len(orders) == 4 and ids.timeout == orders[-1].orderId
    assert orders[0].action == ("BUY" if long else "SELL")
    assert all(o.action == ("SELL" if long else "BUY") for o in orders[1:])
    assert [o.transmit for o in orders] == [False, False, False, True]
    assert all(o.ocaType == 2 and o.ocaGroup == plan.order_plan_id + "-exits" for o in orders[1:])
    assert orders[-1].conditions[0].time == deadline.strftime("%Y%m%d %H:%M:%S UTC")
    # All sends above terminate at this in-memory fake; no network transport exists.
    assert client.connected


def test_trade_stream_releases_shared_capacity_for_execution():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument
    from test_ibkr_resources import CallbackEvent, StreamClient
    from test_stage7_ibkr import _config

    class Client(StreamClient):
        def reqTickByTickData(self, contract, tick_type, number, ignore_size):
            assert tick_type == "Last"
            self.stream = SimpleNamespace(updateEvent=CallbackEvent(), tickByTicks=[])
            return self.stream

        def cancelTickByTickData(self, contract, tick_type):
            self.cancelled.append(contract.conId)

    client = Client()
    broker = IbkrConnection(
        _config().model_copy(update={"market_data_line_budget": 1}), client=client
    )
    broker._account_id = "DU123456"
    instrument = QualifiedInstrument("AAPL", 1, "SMART", "NASDAQ", "USD", "STK")
    broker.prepare_trade_events(instrument)
    t0 = datetime.now(UTC)
    client.stream.tickByTicks = [
        SimpleNamespace(time=t0, price=100.2),
        SimpleNamespace(time=t0, price=99.8),
    ]
    client.stream.updateEvent.emit(client.stream)
    events = broker.trade_events(instrument, t0=t0)
    assert [e.price for e in events] == [100.2, 99.8]
    assert [e.sequence for e in events] == [1, 2]
    broker.release_trade_events(instrument.con_id)
    assert client.cancelled == [1]
    assert not broker._causal_trade_streams


@pytest.mark.parametrize("exit_price,expected_pnl", [(102.0, 198.0), (99.0, -102.0)])
def test_long_position_and_profit_survive_ledger_reload(tmp_path, exit_price, expected_pnl):
    from stocker_execution.execution_models import BrokerOrderIds, OrderAction, OrderLifecycle
    from test_stage7_ledger import _fill, _plan

    path = tmp_path / "long.sqlite"
    ledger = ExecutionLedger(path)
    plan = replace(_plan(), side=OrderAction.BUY, stop_price=99, target_price=102)
    assert ledger.reserve(plan, expected_account="DU123456")
    ledger.record_submission(
        plan.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account="DU123456"
    )
    ledger.record_fill(
        _fill("long-entry", 101, 100, 100, side=OrderAction.BUY, minute=32, commission=1)
    )
    ledger = ExecutionLedger(path)
    assert ledger.positions(Environment.PAPER, "DU123456")[0].quantity == 100
    import asyncio

    from stocker_execution.execution_models import BrokerOpenOrder, BrokerPosition, OrderRole
    from test_stage7_execution import FakeExecutionBroker, _service

    broker = FakeExecutionBroker()
    broker._positions = (BrokerPosition("DU123456", plan.con_id, plan.symbol, 100, 100),)
    broker._open_orders = tuple(
        BrokerOpenOrder(
            order_id,
            plan.order_plan_id,
            "DU123456",
            Environment.PAPER,
            plan.con_id,
            plan.symbol,
            role,
            OrderLifecycle.SUBMITTED,
        )
        for order_id, role in ((102, OrderRole.STOP), (103, OrderRole.TARGET))
    )
    assert asyncio.run(_service(path, broker).reconcile()).ok
    ledger.record_fill(
        _fill(
            "long-exit",
            103 if exit_price > 100 else 102,
            100,
            exit_price,
            side=OrderAction.SELL,
            minute=40,
            commission=1,
        )
    )
    record = ledger.get(plan.order_plan_id)
    assert record.status is OrderLifecycle.CLOSED
    assert record.realized_pnl == expected_pnl
    assert ledger.positions(Environment.PAPER, "DU123456") == ()


def test_additive_method_columns_preserve_old_execution_rows(tmp_path):
    import sqlite3

    from test_stage7_ledger import _submitted_ledger

    ledger = _submitted_ledger(tmp_path)
    before = ledger.get("plan-1")
    with sqlite3.connect(ledger.path) as connection:
        for column in ("deadline", "timeout_order_id", "market_id", "method_spec_hash"):
            connection.execute(f"ALTER TABLE execution_plans DROP COLUMN {column}")
    restored = ExecutionLedger(ledger.path).get("plan-1")
    assert restored == before
    assert restored.method_spec_hash is None


def test_future_method_can_supply_absolute_exits_without_session_hard_M():
    from test_stage7_order_plan import _decision, _intent

    intent = replace(_intent(), m_price=None, stop_price=101, target_price=98)
    result = _decision(intent)
    assert result.approved and result.quantity == 100
    assert result.stop_price == 101 and result.target_price == 98


@pytest.mark.usefixtures("execution_method")
def test_failed_shortability_is_a_single_persisted_admission(tmp_path):
    import asyncio
    from datetime import UTC, datetime

    from test_stage8_runtime import (
        FakeBroker,
        MutableClock,
        TriggerContextProvider,
        TriggerEntrySource,
        _hv_run,
        _runtime,
    )

    class NoBorrow(FakeBroker):
        calls = 0

        async def shortable_quantity(self, instrument):
            self.calls += 1
            return 0

    t0 = datetime(2026, 9, 2, 14, tzinfo=UTC)
    broker, clock = NoBorrow(), MutableClock()
    runtime = _runtime(
        tmp_path,
        broker,
        _hv_run(),
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )

    async def scenario():
        await runtime.start()
        clock.now = t0 + timedelta(minutes=1)
        await runtime.poll_once()
        await runtime.poll_once()
        clock.now += timedelta(seconds=1)
        await runtime.poll_once()

    asyncio.run(scenario())
    assert broker.calls == 1
    assert broker.submitted == []
    assert runtime._ledger.attempted_signal_ids("hv-run")


def test_prior20_hv_uses_exact_prior_session_closes(tmp_path):
    import asyncio
    import math
    from datetime import UTC, date, datetime

    from stocker_core.markets import MarketId, get_market
    from stocker_data.calendars import get_market_calendar
    from stocker_execution.history import IbkrHistoryCache
    from stocker_execution.ibkr import HistoricalBar, IbkrConnection, QualifiedInstrument
    from stocker_execution.session_hard_data import SOURCE, PriorSessionExpectedMoveService

    session = date(2026, 9, 8)
    schedule = (
        get_market_calendar("XNYS")
        .schedule(start_date=session - timedelta(days=90), end_date=session - timedelta(days=1))
        .tail(21)
    )
    closes = {
        v.to_pydatetime(): 100 * math.exp(0.01 if index % 2 else 0)
        for index, v in enumerate(schedule.market_close)
    }

    class History(IbkrConnection):
        def __init__(self):
            self.requested = []

        async def historical_bars(self, instrument, **kwargs):
            end = kwargs["end_time"]
            assert kwargs["bar_size"] == "1 min"
            assert kwargs["duration"] == "60 S"
            assert kwargs["regular_trading_hours"] is True
            self.requested.append(end)
            price = closes[end]
            return (HistoricalBar(end - timedelta(minutes=1), price, price, price, price, 100),)

    broker = History()
    service = PriorSessionExpectedMoveService(broker, IbkrHistoryCache(tmp_path / "history.sqlite"))
    instrument = QualifiedInstrument("TEST", 1, "SMART", "NASDAQ", "USD", "STK")
    args = dict(
        market=get_market(MarketId.US_ALL), session=session, t0=datetime(2026, 9, 8, 14, tzinfo=UTC)
    )
    asyncio.run(service.prepare_expected_move(instrument, **args))
    result = asyncio.run(service.get_expected_move(instrument, **args))
    assert result.source == SOURCE
    assert result.historical_volatility == pytest.approx(0.01 * math.sqrt(20 / 19) * math.sqrt(252))
    assert broker.requested == list(closes)
    assert all(t.date() < session for t in broker.requested)


def test_direction_inputs_and_aggregation_exclude_trigger_minute():
    from datetime import UTC, datetime

    from stocker_execution.ibkr import HistoricalBar
    from stocker_execution.session_hard_data import completed_five_minute_bars, directional_inputs

    start = datetime(2026, 9, 8, 13, 30, tzinfo=UTC)
    bars = tuple(
        HistoricalBar(start + timedelta(minutes=i), 100 + i, 102 + i, 99 + i, 101 + i, 10)
        for i in range(30)
    )
    t0 = start + timedelta(minutes=30)
    five = completed_five_minute_bars(bars)
    assert (five[-1].open, five[-1].high, five[-1].low, five[-1].close, five[-1].volume) == (
        125,
        131,
        124,
        130,
        50,
    )
    values = directional_inputs(bars, t0, 6)
    assert values["completed_5m_body"] == 5 / 7
    assert values["return_3m"] == pytest.approx((130 / 127 - 1) * 10000)
    assert values["session_open_distance"] == pytest.approx(3000)
    future = HistoricalBar(t0, 1000, 1001, 999, 1000, 100)
    with pytest.raises(ValueError, match="exact causal"):
        directional_inputs((*bars, future), t0, 6)


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.usefixtures("execution_method")
def test_runtime_releases_rejected_or_expired_streams(tmp_path, qualified):
    import asyncio
    from datetime import UTC, datetime

    from test_stage8_runtime import (
        FakeBroker,
        MutableClock,
        TriggerContextProvider,
        TriggerEntrySource,
        _hv_run,
        _runtime,
    )

    class Events(TriggerEntrySource):
        trade_errors = {}

        def __init__(self):
            self.active = set()

        def prepare_trades(self, instrument):
            self.active.add(instrument.con_id)

        def release_unused_trades(self, retained):
            self.active.intersection_update(retained)

        async def trades_for(self, instruments, signals):
            return {}

    source, broker, clock = Events(), FakeBroker(), MutableClock()
    runtime = _runtime(
        tmp_path,
        broker,
        _hv_run(),
        clock=clock,
        entry_source=source,
        context_provider=TriggerContextProvider() if qualified else None,
    )

    async def scenario():
        await runtime.start()
        await runtime.poll_once()  # prefetch before the first checkpoint
        assert source.active == {1000}
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()
        assert source.active == ({1000} if qualified else set())
        if qualified:
            # Past the final checkpoint's entry window: no future prefetch needs a stream.
            clock.now = datetime(2026, 9, 2, 16, 26, tzinfo=UTC)
            await runtime.poll_once()
            assert source.active == set()

    asyncio.run(scenario())
    assert broker.submitted == []
