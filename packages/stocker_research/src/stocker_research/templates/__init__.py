"""Deterministic research templates."""

from stocker_research.templates.base import StrategyTemplate
from stocker_research.templates.breakout import VolatilityBreakoutTemplate
from stocker_research.templates.mean_reversion import MeanReversionTemplate
from stocker_research.templates.moving_average import MovingAverageMomentumTemplate
from stocker_research.templates.opening_range import OpeningRangeBreakoutTemplate
from stocker_research.templates.pullback import PullbackInUptrendTemplate
from stocker_research.templates.vwap import VWAPReclaimRejectionTemplate

TEMPLATE_REGISTRY: dict[str, type[StrategyTemplate]] = {
    MovingAverageMomentumTemplate.name: MovingAverageMomentumTemplate,
    PullbackInUptrendTemplate.name: PullbackInUptrendTemplate,
    MeanReversionTemplate.name: MeanReversionTemplate,
    VolatilityBreakoutTemplate.name: VolatilityBreakoutTemplate,
    OpeningRangeBreakoutTemplate.name: OpeningRangeBreakoutTemplate,
    VWAPReclaimRejectionTemplate.name: VWAPReclaimRejectionTemplate,
}


def get_template(name: str) -> StrategyTemplate:
    """Return a registered deterministic research template."""

    try:
        return TEMPLATE_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown research template: {name}") from exc


__all__ = [
    "MeanReversionTemplate",
    "MovingAverageMomentumTemplate",
    "OpeningRangeBreakoutTemplate",
    "PullbackInUptrendTemplate",
    "StrategyTemplate",
    "TEMPLATE_REGISTRY",
    "VolatilityBreakoutTemplate",
    "VWAPReclaimRejectionTemplate",
    "get_template",
]
