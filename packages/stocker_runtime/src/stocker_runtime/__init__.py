"""Public contracts for Stocker's prospective-record and shadow runtime."""

from stocker_runtime.domain import (
    MAX_MARKET_EVENT_PAYLOAD_BYTES,
    IdeaOutput,
    JsonValue,
    MarketEvent,
    Observation,
    OutputKind,
    ProposedPosition,
    ProposedTrade,
    ProtectedDataClass,
    RuntimeMode,
    Signal,
    canonical_json_bytes,
)
from stocker_runtime.ideas import (
    MAX_EVENTS_PER_BATCH,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataRequirement,
)

__all__ = [
    "IdeaOutput",
    "IdeaActivation",
    "IdeaBatch",
    "IdeaEvaluation",
    "IdeaManifest",
    "IdeaPlugin",
    "JsonValue",
    "MAX_EVENTS_PER_BATCH",
    "MAX_MARKET_EVENT_PAYLOAD_BYTES",
    "MarketDataRequirement",
    "MarketEvent",
    "Observation",
    "OutputKind",
    "ProtectedDataClass",
    "ProposedPosition",
    "ProposedTrade",
    "RuntimeMode",
    "Signal",
    "canonical_json_bytes",
]
__version__ = "0.1.0"
