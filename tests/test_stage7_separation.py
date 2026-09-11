import inspect

from stocker_execution.session_hard_structure_d import SessionHardStructureDStrategy
from stocker_execution.stage7 import Stage7RiskEngine


def test_stage7_does_not_calculate_pre_move_or_strategy_qualification() -> None:
    source = inspect.getsource(Stage7RiskEngine)

    assert "PRE_MOVE" not in source
    assert "cohort" not in source.lower()
    assert "session_hard" not in source.lower()


def test_stage6_strategy_makes_no_broker_order_calls() -> None:
    source = inspect.getsource(SessionHardStructureDStrategy)

    assert "IbkrConnection" not in source
    assert "submit_protected_order" not in source
    assert "placeOrder" not in source


def test_risk_calculation_is_synchronous_and_broker_independent() -> None:
    source = inspect.getsource(Stage7RiskEngine.evaluate)

    assert inspect.iscoroutinefunction(Stage7RiskEngine.evaluate) is False
    assert "await " not in source
    assert "Ibkr" not in source
