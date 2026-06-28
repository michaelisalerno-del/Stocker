"""Transaction-cost assumptions for research and backtests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Simple basis-point cost model.

    Each field is treated as a one-way estimate. Round-trip cost doubles the combined
    spread, commission, and slippage assumptions.
    """

    spread_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("spread_bps", self.spread_bps),
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def one_way_bps(self) -> float:
        """Return estimated one-way cost in basis points."""

        return self.spread_bps + self.commission_bps + self.slippage_bps

    def round_trip_bps(self) -> float:
        """Return estimated entry-plus-exit cost in basis points."""

        return self.one_way_bps() * 2

    def estimate_round_trip_cost(self, notional: float) -> float:
        """Estimate round-trip cash cost for a notional value."""

        if notional < 0:
            raise ValueError("notional must be non-negative")
        return notional * self.round_trip_bps() / 10_000
