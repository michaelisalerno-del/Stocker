"""Research-only exact-clock entry-latency evaluation."""

from .immutable_ledger import ProspectiveLatencyLedger
from .timing import (
    BAR_DURATION,
    FixedLatencyResult,
    direction_adjusted_entry_move_bps,
    score_fixed_latency,
)

__all__ = [
    "BAR_DURATION",
    "FixedLatencyResult",
    "ProspectiveLatencyLedger",
    "direction_adjusted_entry_move_bps",
    "score_fixed_latency",
]
