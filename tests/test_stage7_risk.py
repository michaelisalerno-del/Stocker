from datetime import UTC, date, datetime

import pytest

from stocker_core.runs import Environment, RunRiskConfig
from stocker_execution.execution_models import BrokerAccountState, BrokerPosition
from stocker_execution.session_hard_structure_d import (
    SignalStatus,
    StrategySignal,
)
from stocker_execution.stage7 import RiskRejection, Stage7RiskEngine


def _intent(
    *,
    entry_reference: float | None = 100.0,
    m_price: float | None = 2.0,
) -> StrategySignal:
    timestamp = datetime(2026, 9, 2, 14, 31, tzinfo=UTC)
    return StrategySignal(
        strategy_id="SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
        strategy_version="SESSION_HARD_STRUCTURE_D_V1",
        signal_id="signal-1",
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
        feature_calculation_version="STAGE5_PRE_MOVE_V1",
        side="SHORT",
        direction="DOWN",
        candidate_rank=1,
        selected=True,
        p0=100.4,
        m_price=m_price,
        entry_level=100.0,
        entry_reference=entry_reference,
        entry_timestamp=timestamp,
        signal_timestamp=timestamp,
        stop_distance_m=0.50,
        target_distance_m=1.00,
    )


def _account(
    *, equity: float | None = 100_000.0, positions: tuple[BrokerPosition, ...] = ()
) -> BrokerAccountState:
    return BrokerAccountState(
        environment=Environment.PAPER,
        account="DU123456",
        equity=equity,
        buying_power=200_000.0,
        connected=True,
        positions=positions,
    )


def test_risk_budget_per_share_risk_and_quantity_are_deterministic() -> None:
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(),
        account_state=_account(),
        risk_config=RunRiskConfig(risk_per_trade=0.01),
    )

    assert decision.approved is True
    assert decision.reason == "APPROVED"
    assert decision.risk_budget == 1_000.0
    assert decision.per_share_risk == 1.0
    assert decision.quantity == 1_000
    assert decision.entry_price == 100.0
    assert decision.stop_price == 101.0
    assert decision.target_price == 98.0


def test_quantity_rounds_down_and_never_exceeds_risk_budget() -> None:
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(entry_reference=101.0, m_price=3.0),
        account_state=_account(equity=10_000.0),
        risk_config=RunRiskConfig(risk_per_trade=0.01),
    )

    assert decision.quantity == 66
    assert decision.quantity * decision.per_share_risk <= decision.risk_budget
    assert (decision.quantity + 1) * decision.per_share_risk > decision.risk_budget


@pytest.mark.parametrize(
    ("risk_fraction", "reason"),
    [(0.0, RiskRejection.INVALID_RISK_CONFIG), (-0.01, RiskRejection.INVALID_RISK_CONFIG)],
)
def test_nonpositive_explicit_risk_config_rejects_without_crashing(
    risk_fraction: float, reason: RiskRejection
) -> None:
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(),
        account_state=_account(),
        risk_config=RunRiskConfig(risk_per_trade=risk_fraction),
    )

    assert decision.approved is False
    assert decision.reason == reason


def test_missing_explicit_risk_config_rejects() -> None:
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(), account_state=_account(), risk_config=None
    )

    assert decision.reason == RiskRejection.INVALID_RISK_CONFIG


@pytest.mark.parametrize(
    ("entry_reference", "m_price", "equity", "reason"),
    [
        (0.0, 2.0, 100_000.0, RiskRejection.INVALID_STOP_DISTANCE),
        (100.0, 0.0, 100_000.0, RiskRejection.INVALID_STOP_DISTANCE),
        (100.0, -1.0, 100_000.0, RiskRejection.INVALID_STOP_DISTANCE),
        (100.0, 2.0, None, RiskRejection.ACCOUNT_STATE_UNAVAILABLE),
        (100.0, 2.0, 0.5, RiskRejection.ZERO_QUANTITY),
    ],
)
def test_invalid_sizing_inputs_return_concise_rejection(
    entry_reference: float,
    m_price: float,
    equity: float | None,
    reason: RiskRejection,
) -> None:
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(entry_reference=entry_reference, m_price=m_price),
        account_state=_account(equity=equity),
        risk_config=RunRiskConfig(risk_per_trade=0.01),
    )

    assert decision.approved is False
    assert decision.reason == reason


def test_existing_same_instrument_position_blocks_pyramiding() -> None:
    position = BrokerPosition("DU123456", 265598, "AAPL", -10.0, 100.0)
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(),
        account_state=_account(positions=(position,)),
        risk_config=RunRiskConfig(risk_per_trade=0.01),
    )

    assert decision.reason == RiskRejection.POSITION_ALREADY_OPEN


def test_configured_capacity_counts_actual_nonzero_broker_positions() -> None:
    position = BrokerPosition("DU123456", 999, "MSFT", 5.0, 250.0)
    decision = Stage7RiskEngine().evaluate(
        order_intent=_intent(),
        account_state=_account(positions=(position,)),
        risk_config=RunRiskConfig(risk_per_trade=0.01, max_concurrent_positions=1),
    )

    assert decision.reason == RiskRejection.CAPACITY_REACHED
