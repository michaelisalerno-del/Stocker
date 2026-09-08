from datetime import UTC, date, datetime

import pytest

from stocker_core.runs import Environment, RunRiskConfig
from stocker_execution.execution_models import (
    BrokerAccountState,
    EntryOrderType,
    OrderAction,
)
from stocker_execution.session_hard_structure_d import SignalStatus, StrategySignal
from stocker_execution.stage7 import Stage7RiskDecision, Stage7RiskEngine, build_order_plan


def _intent(*, m_price: float = 2.0) -> StrategySignal:
    timestamp = datetime(2026, 9, 2, 14, 31, tzinfo=UTC)
    return StrategySignal(
        strategy_id="TEST_EXECUTION",
        strategy_version="TEST_EXECUTION_V1",
        signal_id="deterministic-stage6-signal",
        run_id="paper-run",
        underlying_con_id=265598,
        symbol="AAPL",
        universe_id="NASDAQ",
        session=date(2026, 9, 2),
        t0=datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        status=SignalStatus.ENTRY_TRIGGERED,
        reason="STRUCTURE_D_DOWN_FIRST_TOUCH",
        pre_move_m=0.8,
        cohort_percentile=75.0,
        band=None,
        session_hard_score=0.9999,
        session_hard_checkpoint=6,
        session_hard_qualified=True,
        feature_calculation_version="STAGE5_PRE_MOVE_HV_V1",
        side="SHORT",
        direction="DOWN",
        candidate_rank=1,
        selected=True,
        p0=100.4,
        m_price=m_price,
        entry_level=100.0,
        entry_reference=100.0,
        entry_timestamp=timestamp,
        signal_timestamp=timestamp,
    )


def _decision(intent: StrategySignal) -> Stage7RiskDecision:
    return Stage7RiskEngine().evaluate(
        order_intent=intent,
        account_state=BrokerAccountState(Environment.PAPER, "DU123456", 100_000.0, 200_000.0, True),
        risk_config=RunRiskConfig(risk_per_trade=0.001),
    )


def test_plan_maps_stage6_short_intent_to_protected_market_parent() -> None:
    intent = _intent()
    plan = build_order_plan(
        order_intent=intent,
        risk_decision=_decision(intent),
        environment=Environment.PAPER,
        minimum_tick=0.01,
        created_at=datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
    )

    assert plan.run_id == intent.run_id
    assert plan.signal_id == intent.signal_id
    assert plan.strategy_id == intent.strategy_id
    assert plan.strategy_version == intent.strategy_version
    assert plan.con_id == 265598
    assert plan.symbol == "AAPL"
    assert plan.side is OrderAction.SELL
    assert plan.quantity == 100
    assert plan.entry_order_type is EntryOrderType.MARKET
    assert plan.entry_reference == 100.0
    assert plan.stop_price == 101.0
    assert plan.target_price == 98.0
    assert plan.environment is Environment.PAPER


def test_short_protection_tick_rounding_never_increases_intended_risk() -> None:
    intent = _intent(m_price=0.246)
    decision = _decision(intent)
    plan = build_order_plan(
        order_intent=intent,
        risk_decision=decision,
        environment=Environment.PAPER,
        minimum_tick=0.05,
        created_at=datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
    )

    assert decision.stop_price == pytest.approx(100.123)
    assert decision.target_price == pytest.approx(99.754)
    assert plan.stop_price == 100.10
    assert plan.target_price == 99.80
    assert plan.stop_price - plan.entry_reference <= decision.per_share_risk


def test_order_plan_identity_is_deterministic_across_replays() -> None:
    intent = _intent()
    first = build_order_plan(
        order_intent=intent,
        risk_decision=_decision(intent),
        environment=Environment.PAPER,
        minimum_tick=0.01,
        created_at=datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
    )
    second = build_order_plan(
        order_intent=intent,
        risk_decision=_decision(intent),
        environment=Environment.PAPER,
        minimum_tick=0.01,
        created_at=datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
    )

    assert first.order_plan_id == second.order_plan_id


@pytest.mark.parametrize("minimum_tick", [0.0, -0.01, float("nan")])
def test_invalid_minimum_tick_cannot_produce_an_executable_plan(minimum_tick: float) -> None:
    intent = _intent()
    with pytest.raises(ValueError, match="minimum tick"):
        build_order_plan(
            order_intent=intent,
            risk_decision=_decision(intent),
            environment=Environment.PAPER,
            minimum_tick=minimum_tick,
            created_at=datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
        )


def test_inverted_protection_cannot_produce_an_executable_plan() -> None:
    intent = _intent()
    invalid = Stage7RiskDecision(
        approved=True,
        reason="APPROVED",
        risk_budget=100.0,
        per_share_risk=1.0,
        quantity=100,
        entry_price=100.0,
        stop_price=99.0,
        target_price=101.0,
    )

    with pytest.raises(ValueError, match="wrong side of entry"):
        build_order_plan(
            order_intent=intent,
            risk_decision=invalid,
            environment=Environment.PAPER,
            minimum_tick=0.01,
            created_at=datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
        )
