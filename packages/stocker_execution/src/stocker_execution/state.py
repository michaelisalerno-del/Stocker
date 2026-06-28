"""Execution state placeholders."""

from pydantic import BaseModel, Field


class ExecutionState(BaseModel):
    """Internal execution state used to detect future broker reconciliation issues."""

    positions: dict[str, float] = Field(default_factory=dict)
    cash: float = 0.0
    last_broker_sync_at: str | None = None
