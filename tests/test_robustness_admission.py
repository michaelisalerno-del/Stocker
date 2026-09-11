import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from stocker_core.runs import Environment
from stocker_execution.execution_ledger import AdmissionRejected, ExecutionLedger
from stocker_execution.execution_models import (
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.stage7 import ExecutionResultCode
from test_stage7_execution import FakeExecutionBroker, _instrument, _intent, _run, _service
from test_stage7_ledger import _fill, _plan


def test_fill_during_quote_preparation_invalidates_account_snapshot(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = ExecutionLedger(path)
    first = replace(_plan(), con_id=777)

    class Broker(FakeExecutionBroker):
        async def minimum_tick(self, instrument):
            ledger.record_fill(
                replace(_fill("fill", 101, 100, 100, side=OrderAction.SELL, minute=31), con_id=777)
            )
            self._positions = (BrokerPosition(self.account, 777, "OTHER", -100, 100),)
            return 0.01

    broker = Broker()
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok
    ledger.reserve(first, expected_account=broker.account)
    ledger.record_submission(
        first.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account=broker.account
    )
    result = asyncio.run(service.execute(replace(_intent(), signal_id="second"), _instrument()))
    assert result.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert not broker.submitted


def test_concurrent_credit_previews_do_not_share_unreserved_funding(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Broker(FakeExecutionBroker):
            previews = 0

            async def check_order_capacity(self, plan, instrument):
                self.previews += 1
                entered.set()
                await release.wait()

        broker = Broker()
        path = tmp_path / "ledger.sqlite3"
        first = _service(path, broker)
        second = _service(path, broker, run=_run().model_copy(update={"run_id": "second"}))
        assert (await first.reconcile()).ok
        assert (await second.reconcile()).ok
        active = asyncio.create_task(first.execute(_intent(), _instrument()))
        await asyncio.wait_for(entered.wait(), 1)
        result = await second.execute(
            replace(_intent(), run_id="second", signal_id="second", underlying_con_id=777),
            replace(_instrument(), con_id=777, symbol="OTHER"),
        )
        assert result.code is ExecutionResultCode.PENDING_ENTRY_CAPACITY_UNVERIFIED
        assert broker.previews == 1
        assert len(ExecutionLedger(path).active_records(Environment.PAPER, broker.account)) == 1
        release.set()
        assert (await active).code is ExecutionResultCode.SUBMITTED

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["same", "account", "environment"])
def test_simultaneous_same_instrument_respects_account_environment_scope(tmp_path, scope):
    path = tmp_path / "scope.sqlite"
    ledgers = [ExecutionLedger(path), ExecutionLedger(path)]
    barrier = Barrier(2)

    def admit(index):
        account = "DU_OTHER" if index and scope == "account" else "DU123456"
        environment = Environment.LIVE if index and scope == "environment" else Environment.PAPER
        plan = replace(
            _plan(),
            order_plan_id=f"p{index}",
            signal_id=f"s{index}",
            run_id=f"run{index}",
            environment=environment,
        )
        barrier.wait(timeout=5)
        try:
            return ledgers[index].reserve(plan, expected_account=account, max_positions=5)
        except AdmissionRejected as exc:
            assert str(exc) == "POSITION_ALREADY_OPEN"
            return False

    with ThreadPoolExecutor(2) as pool:
        results = sorted(pool.map(admit, (0, 1)))
    assert results == ([False, True] if scope == "same" else [True, True])


def test_multiple_recovered_pending_commitments_are_counted_before_admission(tmp_path):
    ledger = ExecutionLedger(tmp_path / "multiple.sqlite")
    for index in range(2):
        assert ledger.reserve(
            replace(_plan(), order_plan_id=f"p{index}", signal_id=f"s{index}", con_id=index + 1),
            expected_account="DU123456",
        )
    ledger = ExecutionLedger(ledger.path)
    third = replace(_plan(), order_plan_id="third", signal_id="third", con_id=3)
    with pytest.raises(AdmissionRejected, match="PENDING_ENTRY_CAPACITY_UNVERIFIED"):
        ledger.reserve(third, expected_account="DU123456", require_settled_entries=True)
    # The lower-level ledger can account for legacy simultaneous commitments:
    # two unfilled 100-share entries at $100 leave only $5,000.
    assert ledger.reserve(
        third, expected_account="DU123456", max_positions=3, max_gross_notional=25000
    )
    assert ledger.get("third").intended_quantity == 50
    assert len(ledger.active_records(Environment.PAPER, "DU123456")) == 3


@pytest.mark.parametrize("phase", ["quote", "credit"])
def test_risk_update_during_admission_cannot_submit_old_sizing(tmp_path, phase):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Broker(FakeExecutionBroker):
            async def entry_quote(self, instrument):
                if phase == "quote":
                    entered.set()
                    await release.wait()
                return await super().entry_quote(instrument)

            async def check_order_capacity(self, plan, instrument):
                if phase == "credit":
                    entered.set()
                    await release.wait()

        broker = Broker()
        service = _service(tmp_path / "risk-change.sqlite", broker)
        assert (await service.reconcile()).ok
        pending = asyncio.create_task(service.execute(_intent(), _instrument()))
        await asyncio.wait_for(entered.wait(), 2)
        old = service.run_config
        service.update_run_config(
            old.model_copy(update={"risk": old.risk.model_copy(update={"risk_per_trade": 0.0001})})
        )
        assert service.run_config.risk.risk_per_trade == 0.0001
        release.set()
        result = await pending
        assert result.code is not ExecutionResultCode.SUBMITTED
        assert not broker.submitted
        assert not ExecutionLedger(tmp_path / "risk-change.sqlite").active_records(
            Environment.PAPER, broker.account
        )

    asyncio.run(scenario())


def test_reconciliation_blocks_status_fill_gap_even_with_protective_orders(tmp_path):
    path = tmp_path / "status-gap.sqlite"
    ledger = ExecutionLedger(path)
    plan = _plan()
    ledger.reserve(plan, expected_account="DU123456")
    ledger.record_submission(
        plan.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account="DU123456"
    )
    status = BrokerOrderStatus(
        101, plan.order_plan_id, "DU123456", Environment.PAPER, OrderLifecycle.CANCELLED, 40, 60
    )
    broker = FakeExecutionBroker(
        statuses=(status,),
        open_orders=tuple(
            BrokerOpenOrder(
                i,
                plan.order_plan_id,
                "DU123456",
                Environment.PAPER,
                plan.con_id,
                plan.symbol,
                role,
                OrderLifecycle.SUBMITTED,
            )
            for i, role in [(102, OrderRole.STOP), (103, OrderRole.TARGET)]
        ),
    )
    service = _service(path, broker)
    result = asyncio.run(service.reconcile())
    assert not result.ok and "entry execution details unresolved" in result.detail
    broker._fills = (_fill("late", 101, 40, 100, side=OrderAction.SELL, minute=32),)
    broker._positions = (BrokerPosition("DU123456", plan.con_id, plan.symbol, -40, 100),)
    assert asyncio.run(service.reconcile()).ok
