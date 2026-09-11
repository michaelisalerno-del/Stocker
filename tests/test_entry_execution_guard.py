"""Regression for HRB's historical trigger being submitted near its old target."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from stocker_core.runs import Environment, RunConfig, RunRiskConfig
from stocker_core.strategies import SESSION_HARD_HV_METHOD
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOrderStatus,
    OrderAction,
    OrderLifecycle,
)
from stocker_execution.ibkr import CurrentQuote, QualifiedInstrument
from stocker_execution.session_hard_structure_d import (
    EntryBar,
    SessionHardStructureDStrategy,
    SignalStatus,
)
from stocker_execution.stage7 import Stage7ExecutionService
from test_stage7_execution import FakeExecutionBroker, _intent

SIGNAL_TIME = datetime(2026, 9, 4, 15, 41, tzinfo=UTC)
INSTRUMENT = QualifiedInstrument("HRB", 8130, "SMART", "NYSE", "USD", "STK")


def hrb_intent():
    seed = replace(
        _intent(),
        symbol="HRB",
        underlying_con_id=8130,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version="SESSION_HARD_HV_V1",
        session=SIGNAL_TIME.date(),
        t0=SIGNAL_TIME - timedelta(minutes=1),
        status=SignalStatus.WAITING_FOR_ENTRY,
        selected=False,
        p0=50.01,
        m_price=0.30049193626068643,
        entry_reference=None,
        entry_timestamp=None,
        signal_timestamp=None,
    )
    strategy = SessionHardStructureDStrategy()
    strategy.restore_signals((seed,))
    return strategy.observe_entry_bars(
        {
            8130: (
                EntryBar(seed.t0, 50.01, 50.01, 49.70),
                EntryBar(SIGNAL_TIME, 49.68, 49.69, 49.59),
            )
        }
    )[0]


class QuoteBroker(FakeExecutionBroker):
    def __init__(self, quote, clock, quote_delay=0):
        super().__init__()
        self.quote = quote
        self.clock = clock
        self.quote_delay = quote_delay
        self.quote_reads = 0

    async def account_state(self, *, fresh=False):
        return BrokerAccountState(
            self.environment,
            self.account,
            254407,
            1000000,
            True,
            currency="USD",
            gross_position_value=0,
        )

    async def entry_quote(self, instrument):
        self.quote_reads += 1
        self.clock[0] += timedelta(seconds=self.quote_delay)
        if isinstance(self.quote, Exception):
            raise self.quote
        return self.quote


def attempt(
    tmp_path,
    *,
    age=1,
    bid=49.96,
    ask=49.97,
    quote_age=0,
    data_type=1,
    quote_delay=0,
    quote_error=None,
):
    now = [SIGNAL_TIME + timedelta(seconds=age)]
    quote = CurrentQuote(
        "HRB", 8130, now[0] - timedelta(seconds=quote_age), bid, ask, None, None, data_type
    )
    broker = QuoteBroker(quote_error or quote, now, quote_delay)
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    run = RunConfig(
        run_id="paper-run",
        universe="NASDAQ",
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
        risk=RunRiskConfig(max_gross_notional=1000000, risk_per_trade=0.001),
    )
    service = Stage7ExecutionService(
        run=run, expected_account=broker.account, broker=broker, ledger=ledger, clock=lambda: now[0]
    )

    async def scenario():
        assert (await service.reconcile()).ok
        return await service.execute(hrb_intent(), INSTRUMENT)

    return asyncio.run(scenario()), broker, ledger


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"age": 63, "bid": 49.62, "ask": 49.63}, "STALE_SIGNAL"),
        ({"bid": 49.70, "ask": 49.71}, "ENTRY_PRICE_MOVED"),
        ({"bid": 49.62, "ask": 49.63}, "ENTRY_PRICE_MOVED"),
        ({"bid": 50.09, "ask": 50.11}, "ENTRY_PRICE_MOVED"),
        ({"quote_age": 6}, "ENTRY_QUOTE_UNAVAILABLE"),
        ({"data_type": 2}, "ENTRY_QUOTE_UNAVAILABLE"),
        ({"bid": None}, "ENTRY_QUOTE_UNAVAILABLE"),
        ({"quote_error": RuntimeError("no entitlement")}, "ENTRY_QUOTE_UNAVAILABLE"),
        ({"quote_delay": 61}, "STALE_SIGNAL"),
    ],
)
def test_hrb_obsolete_or_unverifiable_entry_is_rejected(tmp_path, kwargs, reason):
    result, broker, ledger = attempt(tmp_path, **kwargs)
    assert result.code.value == reason
    assert broker.submitted == []
    assert not ledger.has_signal(hrb_intent().signal_id)


def test_fresh_entry_caps_fill_price_and_expires_without_moving_exits(tmp_path):
    result, broker, ledger = attempt(tmp_path)
    assert result.code.value == "SUBMITTED"
    plan = result.order_plan
    assert broker.quote_reads == 1
    assert plan.quantity == 1693
    assert plan.entry_order_type.value == "LIMIT"
    assert plan.entry_limit_price == 49.95
    assert plan.entry_limit_price >= plan.entry_reference
    assert plan.stop_price == 50.10
    assert plan.target_price == 49.65
    assert plan.entry_expires_at == SIGNAL_TIME + timedelta(seconds=6)
    stored = ledger.get(plan.order_plan_id)
    assert stored.entry_limit_price == plan.entry_limit_price
    assert stored.entry_expires_at == plan.entry_expires_at


def test_expiring_partial_entry_retains_exposure_and_duplicate_protection(tmp_path):
    result, broker, ledger = attempt(tmp_path)
    plan = result.order_plan
    ledger.record_fill(
        BrokerFill(
            "partial",
            result.order_ids.parent,
            broker.account,
            broker.environment,
            8130,
            "HRB",
            OrderAction.SELL,
            100,
            49.96,
            SIGNAL_TIME + timedelta(seconds=2),
        )
    )
    ledger.record_order_status(
        BrokerOrderStatus(
            result.order_ids.parent,
            plan.order_plan_id,
            broker.account,
            broker.environment,
            OrderLifecycle.CANCELLED,
            100,
            0,
            "GTD expired",
        )
    )
    restarted = ExecutionLedger(ledger.path)
    record = restarted.get(plan.order_plan_id)
    assert record.filled_quantity == 100
    assert record.status is OrderLifecycle.PARTIALLY_FILLED
    assert restarted.active_records(broker.environment, broker.account) == (record,)
    assert not restarted.reserve(plan, expected_account=broker.account)


def test_reconnect_during_quote_requires_reconciliation_before_entry(tmp_path, monkeypatch):
    original = QuoteBroker.entry_quote

    async def reconnect(self, instrument):
        quote = await original(self, instrument)
        self.connection_epoch += 1
        return quote

    monkeypatch.setattr(QuoteBroker, "entry_quote", reconnect)
    result, broker, ledger = attempt(tmp_path)
    assert result.code.value == "EXECUTION_RECONCILIATION_REQUIRED"
    assert broker.submitted == []
    assert not ledger.has_signal(hrb_intent().signal_id)
