"""Small registry of trading methods actually installed in Stocker V1."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    strategy_id: str
    strategy_version: str
    config_name: str
    label: str
    display_token: str
    environments: tuple[str, ...]


SESSION_HARD_METHOD = StrategyDefinition(
    strategy_id="SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
    strategy_version="SESSION_HARD_STRUCTURE_D_V1",
    config_name="SESSION_HARD",
    label="Session HARD",
    display_token="HARD",
    environments=("PAPER", "LIVE"),
)

SESSION_HARD_HV_METHOD = StrategyDefinition(
    strategy_id="SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
    strategy_version="SESSION_HARD_HV_V1",
    config_name="SESSION_HARD_HV",
    label="Session HARD · HV",
    display_token="HARD-HV",
    environments=("PAPER",),
)

_INSTALLED = (SESSION_HARD_METHOD, SESSION_HARD_HV_METHOD)


def installed_strategies() -> tuple[StrategyDefinition, ...]:
    return _INSTALLED


def get_strategy(strategy_id: str, strategy_version: str) -> StrategyDefinition:
    for item in _INSTALLED:
        if item.strategy_id == strategy_id and item.strategy_version == strategy_version:
            return item
    raise ValueError(f"Trading method is not installed: {strategy_id}/{strategy_version}")
