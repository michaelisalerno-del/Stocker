"""Compatibility names; only the current method is selectable."""

from stocker_core.methods import (
    SESSION_HARD as SESSION_HARD_HV_METHOD,
)
from stocker_core.methods import (
    MethodDefinition as StrategyDefinition,
)
from stocker_core.methods import (
    get_method as get_strategy,
)
from stocker_core.methods import (
    installed_methods as installed_strategies,
)

__all__ = ["SESSION_HARD_HV_METHOD", "StrategyDefinition", "get_strategy", "installed_strategies"]
