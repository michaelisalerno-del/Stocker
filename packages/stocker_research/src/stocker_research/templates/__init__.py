"""Deterministic research templates."""

from stocker_research.templates.base import StrategyTemplate
from stocker_research.templates.breakout import VolatilityBreakoutTemplate
from stocker_research.templates.mean_reversion import MeanReversionTemplate
from stocker_research.templates.moving_average import MovingAverageMomentumTemplate

__all__ = [
    "MeanReversionTemplate",
    "MovingAverageMomentumTemplate",
    "StrategyTemplate",
    "VolatilityBreakoutTemplate",
]
