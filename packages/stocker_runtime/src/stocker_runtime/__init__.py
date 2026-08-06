"""Public contracts for Stocker's prospective-record and shadow runtime."""

from stocker_runtime.domain import (
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
