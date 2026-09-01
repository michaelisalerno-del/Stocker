"""Stage 1 run configuration."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Environment(StrEnum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class RunConfig(BaseModel):
    run_id: str = Field(min_length=1)
    universe: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    environment: Environment
