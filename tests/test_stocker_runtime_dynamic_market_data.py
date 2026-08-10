from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from stocker_runtime.ideas import MarketDataInterest
from stocker_runtime.ingestion.dynamic_market_data import (
    ContractCandidate,
    InstrumentResolver,
    InterestResolutionRequest,
    MarketDataCapacity,
    MarketDataDemand,
    OptionParameterSet,
    SubscriptionApplyPlan,
    SubscriptionController,
    plan_market_data,
)
from stocker_runtime.ingestion.inbox import CallbackFence
from stocker_runtime.ingestion.recorder import InstrumentSpec


def _interest(**changes: object) -> MarketDataInterest:
    values: dict[str, object] = {
        "interest_key": "primary-1dte-call",
        "underlying_instrument_id": "AAPL",
        "asset_kind": "option",
        "minimum_days_to_expiry": 1,
        "maximum_days_to_expiry": 1,
        "option_right": "call",
        "strike_offset": 0,
        "reference_price": 101.0,
        "feed_kind": "quotes",
        "cadence": "snapshot",
        "as_of_at_us": 1_786_281_600_000_000,
        "expires_at_us": 1_786_285_200_000_000,
        "required": True,
        "priority": 100,
        "maximum_contracts": 1,
        "input_event_ids": ("event-1",),
    }
    values.update(changes)
    return MarketDataInterest.model_validate(values)


def test_market_data_interest_is_bounded_causal_and_option_only() -> None:
    assert _interest().maximum_contracts == 1
    with pytest.raises(ValidationError, match="expiry"):
        _interest(minimum_days_to_expiry=5, maximum_days_to_expiry=3)
    with pytest.raises(ValidationError, match="seven days"):
        _interest(expires_at_us=1_786_281_600_000_000 + 7 * 86_400_000_000 + 1)
    with pytest.raises(ValidationError, match="unique"):
        _interest(input_event_ids=("event-1", "event-1"))
    with pytest.raises(ValidationError):
        _interest(asset_kind="stock")


def test_market_data_planner_deduplicates_and_defers_without_exceeding_capacity() -> None:
    requirements = (MarketDataDemand("static:AAPL", "AAPL", "bars", True, 1_000, 15_000_000),)
    interests = (
        MarketDataDemand("interest:a", "OPT-1", "quotes", True, 100, 60_000_000, True),
        MarketDataDemand("interest:b", "OPT-1", "quotes", False, 10, 30_000_000, True),
        MarketDataDemand("interest:c", "OPT-2", "quotes", False, 5, 30_000_000, True),
    )

    plan = plan_market_data(requirements, interests, MarketDataCapacity(line_limit=2))

    assert tuple((item.instrument_id, item.feed_kind) for item in plan.subscriptions) == (
        ("AAPL", "bars"),
        ("OPT-1", "quotes"),
    )
    assert plan.subscriptions[1].source_ids == ("interest:a", "interest:b")
    assert plan.subscriptions[1].stale_after_us == 30_000_000
    assert plan.subscriptions[1].snapshot is True
    assert plan.deferred_source_ids == ("interest:c",)
    assert plan.required_complete is True
    assert plan.line_count == 2

    required_shortfall = plan_market_data(
        requirements,
        (*interests, MarketDataDemand("interest:d", "OPT-3", "quotes", True, 200, 1)),
        MarketDataCapacity(line_limit=2),
    )
    assert required_shortfall.required_complete is False
    assert "interest:a" in required_shortfall.deferred_source_ids
    with pytest.raises(ValueError, match="1..100"):
        MarketDataCapacity(line_limit=101)

    streamed = plan_market_data(
        (),
        (interests[0], replace(interests[1], snapshot=False)),
        MarketDataCapacity(line_limit=1),
    )
    assert streamed.subscriptions[0].snapshot is False


class _DiscoveryBackend:
    def __init__(self) -> None:
        self.parameter_calls = 0
        self.contract_calls: list[tuple[str, float, str]] = []

    def option_parameters(
        self, *, underlying_con_id: int, symbol: str
    ) -> tuple[OptionParameterSet, ...]:
        assert (underlying_con_id, symbol) == (265598, "AAPL")
        self.parameter_calls += 1
        return (
            OptionParameterSet(
                exchange="SMART",
                trading_class="AAPL",
                multiplier="100",
                expirations=("20260810", "20260814"),
                strikes=(95.0, 100.0, 102.0, 105.0),
            ),
        )

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]:
        del multiplier, trading_class
        self.contract_calls.append((expiry, strike, right))
        return (
            ContractCandidate(
                con_id=9001,
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                multiplier="100",
                exchange="SMART",
                currency="USD",
                trading_class="AAPL",
            ),
        )


def test_instrument_resolver_uses_metadata_then_one_exact_contract_query() -> None:
    backend = _DiscoveryBackend()
    resolver = InstrumentResolver(
        backend,
        underlyings={"AAPL": InstrumentSpec("AAPL", 265598, "stock", "AAPL", "SMART", "USD")},
        completed_at_us=lambda: 1_786_281_600_000_123,
    )
    request = InterestResolutionRequest("interest-1", "instance-1", _interest())

    receipt = resolver.resolve(request)

    assert receipt.status == "resolved"
    assert receipt.interest_id == "interest-1"
    assert receipt.interest_key == "primary-1dte-call"
    assert receipt.instrument_id == "ibkr-option-9001"
    assert receipt.expiry == "20260810"
    assert receipt.strike == 100.0
    assert receipt.option_right == "call"
    assert receipt.candidates_inspected == 1
    assert backend.parameter_calls == 1
    assert backend.contract_calls == [("20260810", 100.0, "C")]

    second = resolver.resolve(
        InterestResolutionRequest(
            "interest-2",
            "instance-1",
            _interest(interest_key="primary-1dte-put", option_right="put"),
        )
    )
    assert second.status == "resolved"
    assert backend.parameter_calls == 1
    assert backend.contract_calls[-1] == ("20260810", 100.0, "P")


def test_instrument_resolver_rejects_ambiguous_exact_identity() -> None:
    backend = _DiscoveryBackend()
    original = backend.option_contracts

    def ambiguous(**kwargs: object) -> tuple[ContractCandidate, ...]:
        first = original(**kwargs)  # type: ignore[arg-type]
        return (*first, replace(first[0], con_id=9002))

    backend.option_contracts = ambiguous  # type: ignore[method-assign]
    resolver = InstrumentResolver(
        backend,
        underlyings={"AAPL": InstrumentSpec("AAPL", 265598, "stock", "AAPL", "SMART", "USD")},
        completed_at_us=lambda: 1_786_281_600_000_123,
    )

    receipt = resolver.resolve(InterestResolutionRequest("interest-1", "instance-1", _interest()))

    assert receipt.status == "denied"
    assert receipt.reason_code == "AMBIGUOUS_CONTRACT_IDENTITY"
    assert receipt.instrument_id is None


class _LifecycleBackend:
    def __init__(self) -> None:
        self.actions: list[tuple[str, object]] = []

    def configure_subscriptions(self, subscriptions: tuple[object, ...]) -> None:
        self.actions.append(
            (
                "configure",
                tuple(cast(Any, item).request_id for item in subscriptions),
            )
        )

    def cancel(self, request_id: int) -> None:
        self.actions.append(("cancel", request_id))

    def subscribe(self, fence: CallbackFence) -> None:
        self.actions.append(("subscribe", fence.request_id))


def test_subscription_controller_configures_then_frees_capacity_then_subscribes() -> None:
    backend = _LifecycleBackend()
    controller = SubscriptionController(backend)
    configured_item = type("Configured", (), {"request_id": 2000001})()
    plan = SubscriptionApplyPlan(
        configured=(configured_item,),
        starts=(
            (
                object(),
                CallbackFence("run", 1, 1, 2000001, "new-subscription"),
            ),
        ),
        stops=(1000001,),
    )

    result = controller.apply(plan)

    assert result.started_request_ids == (2000001,)
    assert result.stopped_request_ids == (1000001,)
    assert backend.actions == [
        ("configure", (2000001,)),
        ("cancel", 1000001),
        ("subscribe", 2000001),
    ]


def test_subscription_controller_does_not_start_after_cancellation_failure() -> None:
    class FailingCancelBackend(_LifecycleBackend):
        def cancel(self, request_id: int) -> None:
            super().cancel(request_id)
            raise RuntimeError("cancel failed")

    backend = FailingCancelBackend()
    controller = SubscriptionController(backend)
    configured_item = type("Configured", (), {"request_id": 2_000_001})()

    result = controller.apply(
        SubscriptionApplyPlan(
            configured=(configured_item,),
            starts=((object(), CallbackFence("run", 1, 1, 2_000_001, "new")),),
            stops=(1_000_001,),
        )
    )

    assert result.started_request_ids == ()
    assert result.stopped_request_ids == ()
    assert result.failures == (("cancel", 1_000_001, "RuntimeError"),)
    assert backend.actions == [("configure", (2_000_001,)), ("cancel", 1_000_001)]


def test_subscription_apply_plan_is_bounded_and_rejects_duplicate_actions() -> None:
    configured_item = type("Configured", (), {"request_id": 2_000_001})()
    fence = CallbackFence("run", 1, 1, 2_000_001, "new")
    with pytest.raises(ValueError, match="bound"):
        SubscriptionApplyPlan(
            configured=tuple(configured_item for _ in range(101)),
            starts=(),
            stops=(),
        )
    with pytest.raises(ValueError, match="unique"):
        SubscriptionApplyPlan(
            configured=(configured_item,),
            starts=((object(), fence), (object(), fence)),
            stops=(),
        )
    with pytest.raises(ValueError, match="unique"):
        SubscriptionApplyPlan(
            configured=(configured_item,),
            starts=(),
            stops=(1_000_001, 1_000_001),
        )
