"""Research-only frozen named-loop T0 execution-realism tools."""

from .execution import (
    BAR_DURATION,
    FILL_STRESSES_BPS,
    PRIMARY_FILL_MODEL,
    FillEvidence,
    FillPayoff,
    TriggerReconstruction,
    TriggerType,
    apply_adverse_entry_slippage,
    gross_payoff_bps,
    reconstruct_frozen_oco_trigger,
    score_fill_envelope,
)
from .families import (
    CONTROL_FAMILIES,
    FROZEN_FAMILIES,
    NAMED_FAMILIES,
    FamilySpec,
    family_spec,
)
from .immutable_ledger import (
    DuplicateRecordError,
    IntegrityError,
    ProspectiveExecutionLedger,
)
from .prospective import (
    append_payloads,
    collection_parameters,
    load_payloads,
    open_collection_ledger,
)

__all__ = [
    "BAR_DURATION",
    "CONTROL_FAMILIES",
    "FILL_STRESSES_BPS",
    "FROZEN_FAMILIES",
    "NAMED_FAMILIES",
    "PRIMARY_FILL_MODEL",
    "FamilySpec",
    "FillEvidence",
    "FillPayoff",
    "TriggerReconstruction",
    "TriggerType",
    "DuplicateRecordError",
    "IntegrityError",
    "ProspectiveExecutionLedger",
    "append_payloads",
    "collection_parameters",
    "load_payloads",
    "open_collection_ledger",
    "apply_adverse_entry_slippage",
    "family_spec",
    "gross_payoff_bps",
    "reconstruct_frozen_oco_trigger",
    "score_fill_envelope",
]
