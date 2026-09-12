import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from stocker_core.runs import RunRiskConfig
from stocker_execution.discovery import DiscoveryFx
from stocker_execution.execution_currency import execution_valuation
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.stage7 import ExecutionResultCode
from test_stage7_execution import FakeExecutionBroker, _instrument, _intent, _run, _service


class CurrencyBroker(FakeExecutionBroker):
    async def account_state(self, *, fresh=False):
        return replace(await super().account_state(fresh=fresh), currency="GBP")

    async def discovery_fx(self, currency):
        # Broker-reported two-sided quotes, USD per unit of GBP or AUD.
        bid, ask = {"GBP": (1.25, 1.26), "AUD": (0.62, 0.625)}[currency]
        return DiscoveryFx(
            currency,
            2 / (bid + ask),
            1001,
            currency + "USD",
            bid,
            ask,
            datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC).isoformat(),
        )

    async def stock_execution_rules(self, instrument):
        from stocker_execution.execution_models import StockExecutionRules

        return StockExecutionRules(0.01 if instrument.currency == "GBP" else 1.0, 1, 1)


@pytest.mark.parametrize(
    "currency,cap,quantity",
    [
        ("USD", 1000000, 125),
        ("USD", 1000, 12),
        ("AUD", 1000000, 200),
        ("GBP", 1000000, 10000),
    ],
)
def test_account_currency_sizes_risk_and_notional_without_changing_prices(
    tmp_path, currency, cap, quantity
):
    broker = CurrencyBroker()
    run = _run().model_copy(
        update={
            "risk": RunRiskConfig(
                risk_per_trade=0.001, max_gross_notional=cap, max_concurrent_positions=3
            )
        }
    )
    path = tmp_path / "ledger.sqlite"
    service = _service(path, broker, run=run)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), replace(_instrument(), currency=currency)))
    assert result.code is ExecutionResultCode.SUBMITTED, result.detail
    plan = result.order_plan
    assert plan.quantity == quantity
    assert plan.initial_risk_budget == 100  # GBP
    assert (plan.entry_reference, plan.stop_price, plan.target_price) == (100, 101, 98)
    record = ExecutionLedger(path).get(plan.order_plan_id)
    assert record.account_currency == "GBP"
    assert record.price_currency == currency
    assert record.account_per_price_unit == pytest.approx(
        {"USD": 0.8, "AUD": 0.5, "GBP": 0.01}[currency]
    )
    assert len(broker.submitted) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"bid": 0},
        {"ask": float("nan")},
        {"bid": 2, "ask": 1},
        {"symbol": "EURUSD"},
        {"currency": "AUD"},
        {"con_id": 0},
        {"observed_at": "2026-09-02T14:30:55+00:00"},
        {"observed_at": "2026-09-02T14:31:02+00:00"},
        {"observed_at": "2026-09-02T14:31:01"},
    ],
)
def test_bad_fx_never_reserves_or_submits(tmp_path, change):
    class Broker(CurrencyBroker):
        async def discovery_fx(self, currency):
            return replace(await super().discovery_fx(currency), **change)

    broker = Broker()
    path = tmp_path / "ledger.sqlite"
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    assert result.code is ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE
    assert not broker.submitted
    assert not ExecutionLedger(path).active_records(broker.environment, broker.account)


def test_fx_expiring_during_preview_releases_unsubmitted_reservation(tmp_path):
    now = datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC)

    class Broker(CurrencyBroker):
        async def check_order_capacity(self, plan, instrument):
            nonlocal now
            now += timedelta(seconds=6)

    broker = Broker()
    path = tmp_path / "ledger.sqlite"
    service = _service(path, broker)
    service._clock = lambda: now
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    assert result.code is ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE
    assert not broker.submitted
    assert not ExecutionLedger(path).active_records(broker.environment, broker.account)


def test_pause_during_currency_lookup_remains_effective(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Broker(CurrencyBroker):
            async def discovery_fx(self, currency):
                entered.set()
                await release.wait()
                return await super().discovery_fx(currency)

        broker = Broker()
        service = _service(tmp_path / "ledger.sqlite", broker)
        assert (await service.reconcile()).ok
        work = asyncio.create_task(service.execute(_intent(), _instrument()))
        await entered.wait()
        service.update_run_config(_run().model_copy(update={"enabled": False}))
        release.set()
        result = await work
        assert result.code is not ExecutionResultCode.SUBMITTED
        assert not broker.submitted

    asyncio.run(scenario())


def test_currency_conversion_orientation_and_identity():
    class Broker(CurrencyBroker):
        async def discovery_fx(self, currency):
            if currency == "CAD":
                return DiscoveryFx(
                    "CAD", 1.35, 1002, "USDCAD", 1.3, 1.4, "2026-09-02T14:31:01+00:00"
                )
            return await super().discovery_fx(currency)

    def now():
        return datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC)

    value = asyncio.run(
        execution_valuation(Broker(), replace(_instrument(), currency="CAD"), "GBP", now)
    )
    assert value.account_per_price_unit == pytest.approx(1 / 1.3 / 1.25)
    identity = asyncio.run(execution_valuation(Broker(), _instrument(), "USD", now))
    assert identity.account_per_price_unit == 1
    assert identity.fx_observed_at is None
    assert identity.fx_evidence == "[]"


def test_mixed_currency_reservations_are_atomic_and_persist_conversion(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from test_stage7_ledger import _plan

    path = tmp_path / "ledger.sqlite"
    ledgers = [ExecutionLedger(path), ExecutionLedger(path)]
    barrier = Barrier(2)
    plans = [
        replace(
            _plan(),
            order_plan_id=str(i),
            signal_id=str(i),
            con_id=i + 1,
            quantity=10,
            account_currency="GBP",
            price_currency=currency,
            account_per_price_unit=rate,
        )
        for i, (currency, rate) in enumerate([("USD", 0.8), ("AUD", 0.5)])
    ]

    def reserve(i):
        barrier.wait(timeout=5)
        return ledgers[i].reserve(plans[i], expected_account="DU123456", max_gross_notional=1000)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(reserve, range(2))) == [True, True]
    records = ExecutionLedger(path).active_records(plans[0].environment, "DU123456")
    assert len(records) == 2
    assert (
        sum(r.intended_quantity * r.entry_reference * r.account_per_price_unit for r in records)
        <= 1000
    )
    assert {r.price_currency for r in records} == {"USD", "AUD"}
    assert all(r.account_currency == "GBP" for r in records)


def test_partial_fill_then_cancel_releases_only_remaining_converted_exposure(tmp_path):
    from stocker_execution.execution_ledger import AdmissionRejected
    from stocker_execution.execution_models import BrokerOrderIds, OrderAction
    from test_stage7_ledger import _fill, _plan

    ledger = ExecutionLedger(tmp_path / "ledger.sqlite")
    first = replace(
        _plan(),
        quantity=10,
        account_currency="GBP",
        price_currency="USD",
        account_per_price_unit=0.8,
    )
    assert ledger.reserve(first, expected_account="DU123456", max_gross_notional=1000)
    ledger.record_submission(
        first.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account="DU123456"
    )
    ledger.record_fill(_fill("partial", 101, 4, 100, side=OrderAction.SELL, minute=32))
    second = replace(
        first,
        order_plan_id="two",
        signal_id="two",
        con_id=2,
        price_currency="AUD",
        account_per_price_unit=0.5,
        quantity=100,
    )
    with pytest.raises(AdmissionRejected, match="PENDING_ENTRY_CAPACITY_UNVERIFIED"):
        ledger.reserve(
            second,
            expected_account="DU123456",
            max_gross_notional=1000,
            broker_gross_notional=320,
            require_settled_entries=True,
        )
    # Diagnostic arithmetic: 320 filled GBP + 480 pending GBP leaves 200 GBP.
    assert ledger.reserve(
        second, expected_account="DU123456", max_gross_notional=1000, broker_gross_notional=320
    )
    assert ledger.get("two").intended_quantity == 4
    ledger.record_rejection("two", "preview rejected")
    from stocker_execution.execution_models import BrokerOrderStatus, OrderLifecycle

    ledger.record_order_status(
        BrokerOrderStatus(
            101, first.order_plan_id, "DU123456", first.environment, OrderLifecycle.CANCELLED, 4, 6
        )
    )
    third = replace(second, order_plan_id="three", signal_id="three")
    assert ledger.reserve(
        third,
        expected_account="DU123456",
        max_gross_notional=1000,
        broker_gross_notional=320,
        require_settled_entries=True,
    )
    assert ledger.get("three").intended_quantity == 13


def test_account_base_currency_change_does_not_relabel_active_limits(tmp_path):
    from stocker_execution.execution_ledger import AdmissionRejected
    from test_stage7_ledger import _plan

    ledger = ExecutionLedger(tmp_path / "ledger.sqlite")
    first = replace(
        _plan(), account_currency="GBP", price_currency="USD", account_per_price_unit=0.8
    )
    assert ledger.reserve(first, expected_account="DU123456", max_gross_notional=100000)
    second = replace(first, order_plan_id="two", signal_id="two", con_id=2, account_currency="USD")
    with pytest.raises(AdmissionRejected, match="ACCOUNT_STATE_UNAVAILABLE"):
        ledger.reserve(second, expected_account="DU123456", max_gross_notional=100000)


def test_pence_fills_report_pnl_and_r_in_consistent_units(tmp_path):
    from stocker_execution.execution_models import BrokerOrderIds, OrderAction
    from test_stage7_ledger import _fill, _plan

    ledger = ExecutionLedger(tmp_path / "ledger.sqlite")
    plan = replace(
        _plan(),
        account_currency="GBP",
        price_currency="GBP",
        price_unit=0.01,
        account_per_price_unit=0.01,
        per_share_initial_risk=1,
    )
    assert ledger.reserve(plan, expected_account="DU123456")
    ledger.record_submission(
        plan.order_plan_id, BrokerOrderIds(101, 102, 103), actual_account="DU123456"
    )
    ledger.record_fill(
        _fill("entry", 101, 100, 100, side=OrderAction.SELL, minute=32, commission=0.1)
    )
    ledger.record_fill(_fill("exit", 103, 100, 98, side=OrderAction.BUY, minute=33, commission=0.1))
    record = ExecutionLedger(ledger.path).get(plan.order_plan_id)
    assert record.realized_pnl == pytest.approx(1.8)
    assert record.execution_metrics()["realized_execution_r"] == pytest.approx(1.8)


def test_unknown_foreign_currency_submission_survives_restart(tmp_path):
    from stocker_execution.execution_models import OrderLifecycle

    broker = CurrencyBroker(reject=True)
    path = tmp_path / "ledger.sqlite"
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    assert result.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    record = ExecutionLedger(path).get(result.order_plan.order_plan_id)
    assert record.status is OrderLifecycle.SUBMITTING
    assert record.account_currency == "GBP"
    assert record.account_per_price_unit == 0.8
    assert record.fx_evidence
    restarted_broker = CurrencyBroker()
    restarted = _service(path, restarted_broker)
    assert not asyncio.run(restarted.reconcile()).ok
    second = replace(_intent(), signal_id="second", underlying_con_id=2)
    attempt = asyncio.run(
        restarted.execute(second, replace(_instrument(), con_id=2, currency="AUD"))
    )
    assert attempt.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert not restarted_broker.submitted
    assert ExecutionLedger(path).get(record.order_plan_id).account_per_price_unit == 0.8
