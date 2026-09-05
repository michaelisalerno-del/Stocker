"""Synthetic execution identity used only by routing/control tests."""

import pytest

import stocker_core.strategies as original_registry
from stocker_core.strategies import StrategyDefinition
from stocker_execution.runtime import StockerRuntime as OriginalRuntime

TEST_METHOD = StrategyDefinition(
    "TEST_EXECUTION",
    "TEST_EXECUTION_V1",
    "TEST_EXECUTION",
    "Test execution",
    "TEST",
    ("PAPER", "LIVE"),
)


@pytest.fixture
def execution_method(monkeypatch):
    import stocker_core.strategies as registry
    from stocker_execution.runtime import StockerRuntime

    # CLI reload tests may leave already-collected test modules holding the old class.
    for item in {registry, original_registry}:
        monkeypatch.setattr(item, "_INSTALLED", (*item.installed_strategies(), TEST_METHOD))
    for runtime in {StockerRuntime, OriginalRuntime}:
        monkeypatch.setattr(
            runtime,
            "_SUPPORTED_STRATEGIES",
            {*runtime._SUPPORTED_STRATEGIES, TEST_METHOD.config_name},
        )
