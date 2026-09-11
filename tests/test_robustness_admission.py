import asyncio
from dataclasses import replace

from stocker_core.runs import Environment
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import BrokerOrderIds, BrokerPosition, OrderAction
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
