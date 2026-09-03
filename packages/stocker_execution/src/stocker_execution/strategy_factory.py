"""Explicit runtime construction for Stocker's two installed strategy identities."""

from stocker_core.strategies import SESSION_HARD_HV_METHOD, SESSION_HARD_METHOD
from stocker_execution.session_hard_structure_d import SessionHardStructureDStrategy


def create_strategy(strategy_id: str, strategy_version: str) -> SessionHardStructureDStrategy:
    """Create the shared Session HARD mechanics with the requested installed identity."""

    identity = (strategy_id, strategy_version)
    if identity == (SESSION_HARD_METHOD.strategy_id, SESSION_HARD_METHOD.strategy_version):
        return SessionHardStructureDStrategy()
    if identity == (SESSION_HARD_HV_METHOD.strategy_id, SESSION_HARD_HV_METHOD.strategy_version):
        return SessionHardStructureDStrategy(
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        )
    raise ValueError(f"Unsupported runtime strategy: {strategy_id}/{strategy_version}")
