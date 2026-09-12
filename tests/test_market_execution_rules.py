import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stocker_core.markets import MARKET_CATALOGUE
from stocker_core.runs import RunRiskConfig
from stocker_execution.discovery import DiscoveryFx
from stocker_execution.execution_ledger import AdmissionRejected, ExecutionLedger
from stocker_execution.execution_models import StockExecutionRules
from stocker_execution.stage7 import ExecutionResultCode
from test_execution_currency import CurrencyBroker
from test_stage7_execution import _instrument, _intent, _run, _service


@pytest.mark.parametrize("cap,expected", [(100000, 100), (8100, 100), (7900, None)])
def test_final_quantity_respects_broker_lots_after_account_currency_cap(tmp_path, cap, expected):
    class Broker(CurrencyBroker):
        preview_quantity = None

        async def stock_execution_rules(self, instrument):
            return SimpleNamespace(price_unit=1.0, minimum_quantity=100, quantity_increment=100)

        async def check_order_capacity(self, plan, instrument):
            self.preview_quantity = plan.quantity

    broker = Broker()
    run = _run().model_copy(
        update={
            "risk": RunRiskConfig(
                risk_per_trade=0.001, max_gross_notional=cap, max_concurrent_positions=3
            )
        }
    )
    service = _service(tmp_path / "ledger.sqlite", broker, run=run)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    if expected is None:
        assert result.code is ExecutionResultCode.CAPACITY_REACHED
        assert not broker.submitted
        assert broker.preview_quantity is None
    else:
        assert result.code is ExecutionResultCode.SUBMITTED, result.detail
        assert result.order_plan.quantity == expected
        assert broker.preview_quantity == expected
        assert result.order_plan.risk_derived_quantity == 125
        assert result.order_plan.quantity * 100 * 0.8 <= cap
        assert (
            result.order_plan.entry_reference,
            result.order_plan.stop_price,
            result.order_plan.target_price,
        ) == (100, 101, 98)
        saved = ExecutionLedger(tmp_path / "ledger.sqlite").get(result.order_plan.order_plan_id)
        assert saved.minimum_quantity == saved.quantity_increment == 100


def test_non_gbp_subunit_stock_uses_verified_price_unit(tmp_path):
    class Broker(CurrencyBroker):
        async def price_unit(self, instrument):
            return 0.01

        async def stock_execution_rules(self, instrument):
            return SimpleNamespace(price_unit=0.01, minimum_quantity=1, quantity_increment=1)

        async def discovery_fx(self, currency):
            if currency == "ZAR":
                return DiscoveryFx("ZAR", 1.0, 1003, "USDZAR", 1, 1, "2026-09-02T14:31:01+00:00")
            return await super().discovery_fx(currency)

    broker = Broker()
    service = _service(tmp_path / "ledger.sqlite", broker)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), replace(_instrument(), currency="ZAR")))
    assert result.code is ExecutionResultCode.SUBMITTED, result.detail
    assert result.order_plan.quantity == 12500
    assert result.order_plan.account_per_price_unit == pytest.approx(0.008)


@pytest.mark.parametrize("market", MARKET_CATALOGUE, ids=lambda market: market.market_id)
def test_every_selectable_market_requires_verified_contract_rules(tmp_path, market):
    class Broker(CurrencyBroker):
        calls = []

        async def stock_execution_rules(self, instrument):
            self.calls.append(instrument)
            raise ValueError("Required contract units unavailable")

    broker = Broker()
    service = _service(tmp_path / "ledger.sqlite", broker)
    assert asyncio.run(service.reconcile()).ok
    instrument = replace(_instrument(), currency=market.currency, exchange=market.scanner_location)
    result = asyncio.run(service.execute(_intent(), instrument))
    assert broker.calls == [instrument]
    assert result.code is ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE
    assert not broker.submitted
    assert not ExecutionLedger(tmp_path / "ledger.sqlite").active_records(
        broker.environment, broker.account
    )


def test_concurrent_lot_rounding_and_cap_reservation_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from test_stage7_ledger import _plan

    path = tmp_path / "ledger.sqlite"
    ledgers = [ExecutionLedger(path), ExecutionLedger(path)]
    barrier = Barrier(2)
    plans = [
        replace(
            _plan(),
            con_id=i + 1,
            order_plan_id=str(i),
            signal_id=str(i),
            quantity=500,
            minimum_quantity=100,
            quantity_increment=100,
            account_currency="GBP",
            price_currency="USD",
        )
        for i in range(2)
    ]

    def reserve(i):
        barrier.wait(timeout=5)
        try:
            return ledgers[i].reserve(
                plans[i], expected_account="DU123456", max_gross_notional=15000
            )
        except AdmissionRejected as error:
            assert str(error) == "CAPACITY_REACHED"
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(reserve, range(2))) == [False, True]
    records = ExecutionLedger(path).active_records(plans[0].environment, "DU123456")
    assert len(records) == 1
    assert records[0].intended_quantity == 100
    assert records[0].quantity_increment == 100


@pytest.mark.parametrize("minimum,increment", [(0, 1), (1, 0), (1, 1.5), (True, 1)])
def test_invalid_normalized_quantity_rules_never_reserve(tmp_path, minimum, increment):
    class Broker(CurrencyBroker):
        async def stock_execution_rules(self, instrument):
            return StockExecutionRules(1.0, minimum, increment)

    broker = Broker()
    service = _service(tmp_path / "ledger.sqlite", broker)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    assert result.code is ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE
    assert not broker.submitted
