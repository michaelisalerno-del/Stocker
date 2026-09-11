"""Prospective scanner acquisition definitions, separate from frozen candidate mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_core.markets import CAP_BUCKETS_V1, CapBucket


class AcquisitionRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_id: str = "SESSION_HARD_IBKR_ACQUISITION_EXPERIMENT_V1"
    evaluation_period: str = "PROSPECTIVE_V1"
    families: tuple[str, ...] = (
        "TOP_TRADE_RATE",
        "TOP_VOLUME_RATE",
        "HOT_BY_VOLUME",
        "OPENING_PERCENT_GAIN",
        "OPENING_PERCENT_LOSS",
    )
    cap_slices: tuple[str, ...] = (
        "UNCAPPED",
        "BELOW_MICRO",
        "MICRO",
        "SMALL",
        "MID",
        "LARGE",
        "MEGA",
    )
    sweep_active_seconds: tuple[int, ...] = (60, 180, 240)
    sweep_lateness_seconds: int = Field(default=20, ge=0, le=30)
    scanner_concurrency: int = Field(default=2, ge=1, le=10)
    rows_per_component: int = Field(default=50, ge=1, le=50)
    pool_capacity: int | None = Field(default=None, ge=250)
    pool_cap_policy: Literal["BEST_RANK_FIRST_SEEN_CONID"] = "BEST_RANK_FIRST_SEEN_CONID"
    allow_partial_components: bool = False
    oracle_after: Literal["REGULAR_SESSION_CLOSE"] = "REGULAR_SESSION_CLOSE"
    transport_parity: bool = False

    @model_validator(mode="after")
    def validate_experiment(self) -> AcquisitionRecipe:
        if not self.recipe_id or not self.evaluation_period:
            raise ValueError("Acquisition requires a version and evaluation period")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("Acquisition families must be nonempty and distinct")
        if "UNCAPPED" not in self.cap_slices:
            raise ValueError("An uncapped component is required for floorless/unknown-cap coverage")
        for cap in self.cap_slices:
            cap_bounds(cap)
        if len(set(self.cap_slices)) != len(self.cap_slices):
            raise ValueError("Duplicate cap slices")
        if (
            not self.sweep_active_seconds
            or tuple(sorted(set(self.sweep_active_seconds))) != self.sweep_active_seconds
            or self.sweep_active_seconds[0] <= 0
            or self.sweep_active_seconds[-1] >= 300
        ):
            raise ValueError("Declare distinct causal sweeps before OPEN+5")
        return self


def cap_bounds(name: str) -> tuple[int | None, int | None]:
    if name == "UNCAPPED":
        return None, None
    if name == "BELOW_MICRO":
        return None, CAP_BUCKETS_V1.definition(CapBucket.MICRO).minimum_usd
    return CAP_BUCKETS_V1.bounds(CapBucket(name))


ACQUISITION_EXPERIMENT_V1 = AcquisitionRecipe()


@dataclass(frozen=True)
class AcquisitionScan:
    component_id: str
    family: str
    cap_slice: str
    location: str
    instrument: str
    scan_code: str
    rows: int
    filters: tuple[tuple[str, str], ...] = ()
    unsupported_reason: str = ""


# These masks are declared before observation and reuse the same collected hits.
SHADOW_RECIPES = {
    "ACTIVITY_ONLY_V1": {"families": ["TOP_TRADE_RATE", "TOP_VOLUME_RATE", "HOT_BY_VOLUME"]},
    "OPENING_MOVEMENT_ONLY_V1": {"families": ["OPENING_PERCENT_GAIN", "OPENING_PERCENT_LOSS"]},
    "UNCAPPED_ONLY_V1": {"cap_slices": ["UNCAPPED"]},
    "PARTITIONED_ONLY_V1": {"exclude_cap_slices": ["UNCAPPED"]},
    "UNCAPPED_PARTITIONED_HYBRID_V1": {},
}
