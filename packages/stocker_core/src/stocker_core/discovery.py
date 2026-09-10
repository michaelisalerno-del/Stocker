"""Method-owned discovery configuration; resource selection is not trade ranking."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_core.markets import ActivityScanner, CapBucket


class UniverseSource(StrEnum):
    DYNAMIC_IBKR = "DYNAMIC_IBKR"
    AUTHORITATIVE_LISTINGS = "AUTHORITATIVE_LISTINGS"
    CACHED_MARKET_UNIVERSE = "CACHED_MARKET_UNIVERSE"
    FIXED = "FIXED"
    RESEARCH = "RESEARCH"


class DiscoveryProfile(BaseModel):
    """Immutable per-run operational settings, independently audited from trading rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cap_bands: tuple[CapBucket, ...]
    scanner: ActivityScanner
    stock_type_filter: str = "CORP"
    allowed_stock_types: tuple[str, ...] = ("COMMON", "CORP", "ADR", "REIT")
    results_per_band: int = Field(default=50, ge=1, le=50)
    merged_candidate_limit: int = Field(default=250, ge=1, le=1000)
    monitoring_limit: int = Field(default=150, ge=1, le=1000)
    scan_concurrency: int = Field(default=2, ge=1, le=5)
    minimum_price: float = Field(default=1, gt=0, allow_inf_nan=False)
    price_currency: Literal["LOCAL", "USD"] = "LOCAL"
    scanner_location: str | None = None
    minimum_volume: int = Field(default=1000, ge=1)
    minimum_average_volume: int = Field(default=100000, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> "DiscoveryProfile":
        if not self.cap_bands or len(set(self.cap_bands)) != len(self.cap_bands):
            raise ValueError("Discovery requires distinct cap bands")
        if CapBucket.ALL in self.cap_bands:
            raise ValueError("Discovery requires explicit canonical cap bands")
        if self.monitoring_limit > self.merged_candidate_limit:
            raise ValueError("Monitoring limit cannot exceed merged candidate limit")
        if not self.allowed_stock_types:
            raise ValueError("Discovery requires an explicit stock eligibility policy")
        return self


# This profile belongs to Session HARD, not to every method in the catalogue.
# CAP_BUCKETS_V1 in markets.py is the sole authority for numerical boundaries.
SESSION_HARD_DISCOVERY = DiscoveryProfile(
    profile_id="SESSION_HARD_DISCOVERY_V2",
    version="2",
    cap_bands=(CapBucket.MICRO, CapBucket.SMALL, CapBucket.MID, CapBucket.LARGE, CapBucket.MEGA),
    scanner=ActivityScanner.TOP_TRADE_RATE,
    price_currency="USD",
    scanner_location="STK.US.MAJOR",
)
