"""Frozen prospective M1C opening-reversal experiment mechanics.

This module contains only research-shadow decisions and evidence models.  It
does not expose a broker order method and it never treats option observations
as inputs to the frozen direction rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from enum import Enum, StrEnum
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.option_budget import (
    EpisodeAllocationRecord,
    EpisodeState,
)
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionClass,
)

M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID: Final[str] = "m1c-prospective-opening-reversal-v1"
M1C_PROSPECTIVE_OPENING_REVERSAL_V1_VERSION: Final[str] = "1"
M1C_HIGH_TAIL_THRESHOLD_V1: Final[float] = 0.488333710794033
NEGATIVE_OPENING_RETURN_THRESHOLD_V1: Final[float] = -0.00288963733897
POSITIVE_OPENING_RETURN_THRESHOLD_V1: Final[float] = 0.00225522676046
OPENING_RANGE_THRESHOLD_V1: Final[float] = 0.00384818171835
DESCRIPTIVE_OVERNIGHT_GAP_Q10_V1: Final[float] = -0.00382056890751
DESCRIPTIVE_OVERNIGHT_GAP_Q90_V1: Final[float] = 0.0063796856309
DESCRIPTIVE_TOTAL_TRANSITION_Q10_V1: Final[float] = -0.00536060944383
DESCRIPTIVE_TOTAL_TRANSITION_Q90_V1: Final[float] = 0.00643755517767
RESERVED_MARKET_DATA_LINES_V1: Final[int] = 12
PRIMARY_HORIZON_MINUTES_V1: Final[int] = 15
ENGINEERING_TRANSFER_SESSION_COUNT_V1: Final[int] = 20

OpeningStateV1 = Literal[
    "NEGATIVE_SEVERE_OPENING_TRANSITION",
    "POSITIVE_SEVERE_OPENING_TRANSITION",
    "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
    "NORMAL_OPENING",
    "UNKNOWN_INCOMPLETE",
]
PredictionV1 = Literal["CALL", "PUT", "ABSTAIN"]
CohortPhaseV1 = Literal[
    "engineering_transfer",
    "prospective_development",
    "untouched_confirmation",
]
MaterialOutcomeV1 = Literal[
    "MATERIAL_UP",
    "MATERIAL_DOWN",
    "NO_MATERIAL_MOVE",
]
EventReconciliationOutcomeV1 = Literal[
    "event_accounting_fully_reconciled",
    "event_label_ambiguity_corrected",
    "event_aggregation_bug_corrected",
    "blocked_event_accounting",
]


def _canonical_value(value: object) -> object:
    """Recursively normalise hash inputs, including nested Pydantic models."""

    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, datetime):
        encoded = value.isoformat()
        return encoded[:-6] + "Z" if encoded.endswith("+00:00") else encoded
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


class FrozenOpeningReversalRuleV1(BaseModel):
    """The complete immutable scientific and operational V1 rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"] = (
        "m1c-prospective-opening-reversal-v1"
    )
    experiment_version: Literal["1"] = "1"
    checkpoint: Literal[6] = 6
    market_proxy: Literal["VTI"] = "VTI"
    m1c_probability_threshold: float = M1C_HIGH_TAIL_THRESHOLD_V1
    tail_phase_required: Literal["FIRST_ENTRY"] = "FIRST_ENTRY"
    opening_return_q10: float = NEGATIVE_OPENING_RETURN_THRESHOLD_V1
    opening_return_q90: float = POSITIVE_OPENING_RETURN_THRESHOLD_V1
    opening_range_q75: float = OPENING_RANGE_THRESHOLD_V1
    primary_horizon_minutes: Literal[15] = 15
    reserved_market_data_lines: Literal[12] = 12
    maximum_promoted_underlyings: Literal[1] = 1
    primary_expiry: Literal["1DTE"] = "1DTE"
    order_routing_enabled: Literal[False] = False
    order_methods_available: Literal[False] = False
    direction_formula: Literal["prediction_sign_v1=-opening_transition_sign_v1"] = (
        "prediction_sign_v1=-opening_transition_sign_v1"
    )
    optional_feed_degradation_order: tuple[str, ...] = (
        "neutral_control",
        "additional_strike",
        "3_to_5_dte_comparison",
        "0dte_comparison",
        "tick_by_tick",
        "additional_underlying_diagnostic",
    )

    @model_validator(mode="after")
    def _exact_numeric_identity(self) -> FrozenOpeningReversalRuleV1:
        if (
            self.m1c_probability_threshold,
            self.opening_return_q10,
            self.opening_return_q90,
            self.opening_range_q75,
        ) != (
            M1C_HIGH_TAIL_THRESHOLD_V1,
            NEGATIVE_OPENING_RETURN_THRESHOLD_V1,
            POSITIVE_OPENING_RETURN_THRESHOLD_V1,
            OPENING_RANGE_THRESHOLD_V1,
        ):
            raise ValueError("opening reversal frozen numeric rule differs")
        return self

    @property
    def rule_hash(self) -> str:
        return _sha256(self)


class OpeningReversalSupportGateV1(BaseModel):
    """The identical development and confirmation support floor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    complete_eligible_stock_episodes: Literal[150] = 150
    unique_severe_opening_events: Literal[40] = 40
    positive_transition_events: Literal[15] = 15
    negative_transition_events: Literal[15] = 15
    represented_stocks: Literal[12] = 12
    sessions: Literal[40] = 40
    maximum_stock_episode_fraction: float = 0.2
    maximum_event_episode_fraction: float = 0.15

    @model_validator(mode="after")
    def _exact_concentration_gates(self) -> OpeningReversalSupportGateV1:
        if (
            self.maximum_stock_episode_fraction,
            self.maximum_event_episode_fraction,
        ) != (0.2, 0.15):
            raise ValueError("opening reversal concentration gates differ")
        return self


class FrozenOpeningReversalExperimentConfigV1(BaseModel):
    """Machine-readable preregistration consumed by live and report code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-prospective-opening-reversal-config-v1"]
    rule: FrozenOpeningReversalRuleV1
    engineering_transfer_valid_sessions: Literal[20]
    development_support: OpeningReversalSupportGateV1
    confirmation_support: OpeningReversalSupportGateV1
    primary_null_replications: int = Field(ge=1000)
    cluster_bootstrap_replications: Literal[2000]
    bootstrap_confidence_level: float
    winsor_fraction: float
    primary_null_seed: int
    session_cluster_seed: int
    event_cluster_seed: int
    baselines: tuple[str, ...]
    forbidden_action_inputs: tuple[str, ...]
    optional_expiry_comparisons: tuple[str, ...]
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _exact_analysis_parameters(
        self,
    ) -> FrozenOpeningReversalExperimentConfigV1:
        if (
            self.bootstrap_confidence_level,
            self.winsor_fraction,
        ) != (0.95, 0.01):
            raise ValueError("opening reversal analysis parameters differ")
        return self


def build_frozen_experiment_config_v1(
    *,
    primary_null_seed: int = 2026072901,
    session_cluster_seed: int = 2026072902,
    event_cluster_seed: int = 2026072903,
) -> FrozenOpeningReversalExperimentConfigV1:
    """Build the only permitted V1 analysis and collection configuration."""

    payload: dict[str, object] = {
        "schema_version": "m1c-prospective-opening-reversal-config-v1",
        "rule": FrozenOpeningReversalRuleV1(),
        "engineering_transfer_valid_sessions": 20,
        "development_support": OpeningReversalSupportGateV1(),
        "confirmation_support": OpeningReversalSupportGateV1(),
        "primary_null_replications": 1000,
        "cluster_bootstrap_replications": 2000,
        "bootstrap_confidence_level": 0.95,
        "winsor_fraction": 0.01,
        "primary_null_seed": primary_null_seed,
        "session_cluster_seed": session_cluster_seed,
        "event_cluster_seed": event_cluster_seed,
        "baselines": (
            "follow_vti_severe_opening_sign",
            "oppose_vti_severe_opening_sign",
            "always_call",
            "always_put",
            "most_recent_completed_five_minute_stock_momentum",
            "complete_stock_opening_window_momentum",
            "existing_clean_market_direction_baseline",
            "frozen_a1",
            "frozen_historical_asymmetric_downside_score",
            "independently_frozen_microstructure_rule",
        ),
        "forbidden_action_inputs": (
            "A1",
            "RSI",
            "stock_amplification",
            "stock_resistance",
            "recent_momentum",
            "market_sector_features",
            "microstructure",
            "option_prices",
            "option_spreads",
            "apparent_chart_quality",
            "tail_phase_beyond_first_entry",
            "later_outcomes",
        ),
        "optional_expiry_comparisons": ("0DTE", "3_TO_5_DTE"),
    }
    payload["configuration_hash"] = _sha256(payload)
    return FrozenOpeningReversalExperimentConfigV1.model_validate(payload)


FrozenComparisonScalarV1 = str | float | int | bool | None


def _freeze_comparisons(
    value: object,
) -> tuple[tuple[str, FrozenComparisonScalarV1], ...]:
    """Represent descriptive comparisons as a deeply immutable sorted tuple."""

    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, (tuple, list)):
        items = tuple(value)
    else:
        raise TypeError("frozen comparisons must be a mapping or key/value sequence")
    frozen: list[tuple[str, FrozenComparisonScalarV1]] = []
    seen: set[str] = set()
    for raw_item in items:
        if not isinstance(raw_item, (tuple, list)) or len(raw_item) != 2:
            raise TypeError("frozen comparison entries must be key/value pairs")
        raw_key, raw_value = raw_item
        key = str(raw_key)
        if not key or key in seen:
            raise ValueError("frozen comparison keys must be unique and nonempty")
        if raw_value is not None and not isinstance(
            raw_value,
            (str, float, int, bool),
        ):
            raise TypeError("frozen comparison values must be scalar")
        seen.add(key)
        frozen.append((key, raw_value))
    return tuple(sorted(frozen, key=lambda item: item[0]))


class OpeningReversalActivationReceiptV1(BaseModel):
    """Immutable boundary created before any V1 outcome is eligible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1"]
    activation_timestamp_utc: datetime
    new_york_trading_date_at_activation: date
    branch: str
    commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    dirty_working_tree_status: str
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    m1c_version: str
    tail_phase_version: str
    a1_version: str
    vti_transition_rule_version: str
    option_selection_version: str
    recorder_schema_version: str
    capacity_manager_version: str
    configured_reserved_line_count: Literal[12]
    mandatory_feed_manifest: tuple[str, ...]
    optional_feed_priority_manifest: tuple[str, ...]
    order_routing_disabled: Literal[True]
    order_methods_available: Literal[False]
    retrospective_event_reconciliation_outcome: EventReconciliationOutcomeV1
    protected_pre_activation_outcomes_opened: Literal[False]
    activation_receipt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("activation_timestamp_utc")
    @classmethod
    def _activation_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="activation timestamp")

    @model_validator(mode="after")
    def _frozen_manifests(self) -> OpeningReversalActivationReceiptV1:
        required = {
            "VTI_5m",
            "frozen_20_stock_m1c_5m",
            "critical_connection_and_clock_health",
            "one_promoted_underlying_level1",
            "primary_1dte_atm_call",
            "primary_1dte_atm_put",
        }
        if not required.issubset(self.mandatory_feed_manifest):
            raise ValueError("activation mandatory feed manifest is incomplete")
        if self.optional_feed_priority_manifest != (
            "optional_comparison_contracts",
            "single_promoted_underlying_tick_by_tick",
            "optional_additional_underlying_quote_detail",
            "level_ii_separate_authorisation_only",
            "neutral_controls_and_additional_expiries",
        ):
            raise ValueError("activation optional feed priority differs from V1")
        return self


def build_activation_receipt_v1(
    *,
    activation_timestamp_utc: datetime,
    new_york_trading_date_at_activation: date,
    branch: str,
    commit: str,
    dirty_working_tree_status: str,
    configuration_hash: str,
    m1c_version: str,
    tail_phase_version: str,
    a1_version: str,
    recorder_schema_version: str = "0014_m1c_prospective_opening_reversal_v1",
    capacity_manager_version: str = "market_data_capacity_snapshot_v1",
) -> OpeningReversalActivationReceiptV1:
    rule = FrozenOpeningReversalRuleV1()
    payload: dict[str, object] = {
        "experiment_id": rule.experiment_id,
        "experiment_version": rule.experiment_version,
        "activation_timestamp_utc": _aware_utc(
            activation_timestamp_utc,
            label="activation timestamp",
        ),
        "new_york_trading_date_at_activation": (new_york_trading_date_at_activation),
        "branch": branch,
        "commit": commit,
        "dirty_working_tree_status": dirty_working_tree_status,
        "configuration_hash": configuration_hash,
        "frozen_rule_hash": rule.rule_hash,
        "m1c_version": m1c_version,
        "tail_phase_version": tail_phase_version,
        "a1_version": a1_version,
        "vti_transition_rule_version": "m1c-opening-market-transition-v1",
        "option_selection_version": "opening-reversal-primary-1dte-atm-v1",
        "recorder_schema_version": recorder_schema_version,
        "capacity_manager_version": capacity_manager_version,
        "configured_reserved_line_count": 12,
        "mandatory_feed_manifest": (
            "critical_connection_and_clock_health",
            "VTI_5m",
            "frozen_20_stock_m1c_5m",
            "one_promoted_underlying_level1",
            "primary_1dte_atm_call",
            "primary_1dte_atm_put",
        ),
        "optional_feed_priority_manifest": (
            "optional_comparison_contracts",
            "single_promoted_underlying_tick_by_tick",
            "optional_additional_underlying_quote_detail",
            "level_ii_separate_authorisation_only",
            "neutral_controls_and_additional_expiries",
        ),
        "order_routing_disabled": True,
        "order_methods_available": False,
        "retrospective_event_reconciliation_outcome": ("event_label_ambiguity_corrected"),
        "protected_pre_activation_outcomes_opened": False,
    }
    payload["activation_receipt_hash"] = _sha256(payload)
    return OpeningReversalActivationReceiptV1.model_validate(payload)


def load_activation_receipt_v1(path: str) -> OpeningReversalActivationReceiptV1:
    """Load and verify a signed-by-content activation artifact."""

    from pathlib import Path

    receipt = OpeningReversalActivationReceiptV1.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    payload = receipt.model_dump(mode="python", exclude={"activation_receipt_hash"})
    if _sha256(payload) != receipt.activation_receipt_hash:
        raise ValueError("opening reversal activation receipt hash differs")
    if receipt.frozen_rule_hash != FrozenOpeningReversalRuleV1().rule_hash:
        raise ValueError("opening reversal frozen rule hash differs")
    return receipt


def load_frozen_experiment_config_v1(
    path: str,
) -> FrozenOpeningReversalExperimentConfigV1:
    """Load the config and verify its hash without permitting defaults to drift."""

    from pathlib import Path

    config = FrozenOpeningReversalExperimentConfigV1.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    payload = config.model_dump(mode="python", exclude={"configuration_hash"})
    if _sha256(payload) != config.configuration_hash:
        raise ValueError("opening reversal configuration hash differs")
    if config.rule != FrozenOpeningReversalRuleV1():
        raise ValueError("opening reversal rule differs from the code contract")
    return config


class OpeningReversalPredictionTimingEvidenceV1_1(BaseModel):
    """Causal timing proof for the V1.1 shadow-only receipt amendment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timing_addendum_activation_receipt_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")
    rule_committed_at_utc: datetime
    causal_barrier_armed_at_utc: datetime
    predictor_window_completed_at_utc: datetime
    first_entry_or_post_entry_event_buffered_at_utc: datetime | None
    entry_or_post_entry_data_admitted_before_receipt: bool
    raw_event_archive_write_before_receipt: Literal[True]
    decision_surface_release_requires_durable_receipt: Literal[True]
    nominal_entry_actionable: Literal[False]
    receipt_latency_after_nominal_entry_seconds: float = Field(ge=0.0)

    @field_validator(
        "rule_committed_at_utc",
        "causal_barrier_armed_at_utc",
        "predictor_window_completed_at_utc",
        "first_entry_or_post_entry_event_buffered_at_utc",
    )
    @classmethod
    def _timing_timestamp_is_aware(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        return _aware_utc(value, label="V1.1 timing evidence timestamp")

    @model_validator(mode="after")
    def _commitment_precedes_predictor_completion(
        self,
    ) -> OpeningReversalPredictionTimingEvidenceV1_1:
        if (
            self.rule_committed_at_utc > self.causal_barrier_armed_at_utc
            or self.causal_barrier_armed_at_utc > self.predictor_window_completed_at_utc
            or not math.isfinite(self.receipt_latency_after_nominal_entry_seconds)
        ):
            raise ValueError("V1.1 timing evidence chronology is invalid")
        return self


class OpeningReversalPredictionInputV1(BaseModel):
    """All causal facts allowed to construct one immutable V1 receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_version: Literal["1", "1.1"] = "1"
    activation_timestamp_utc: datetime
    cohort_phase: CohortPhaseV1
    transfer_status: str
    session: date
    stock: str = Field(min_length=1)
    checkpoint: int
    signal_timestamp_utc: datetime
    entry_timestamp_utc: datetime
    receipt_created_at_utc: datetime
    m1c_probability: float | None
    m1c_probability_valid: bool
    high_tail_membership: bool
    fresh_episode_id: str | None
    canonical_fresh_episode: bool
    tail_phase_v1: str
    market_opening_return_v1: float | None
    market_opening_range_v1: float | None
    opening_market_transition_state_v1: OpeningStateV1
    opening_transition_sign_v1: Literal[-1, 1] | None
    opening_transition_event_id_v1: str | None
    vti_opening_transition_complete: bool
    stock_causal_data_complete: bool
    previous_close_atm_iv_scale_15m: float | None
    previous_close_atm_iv_scale_valid: bool
    data_source: str = Field(min_length=1)
    capacity_snapshot_id: str | None
    frozen_comparisons: tuple[
        tuple[str, FrozenComparisonScalarV1],
        ...,
    ] = ()
    timing_evidence_v1_1: OpeningReversalPredictionTimingEvidenceV1_1 | None = None

    @field_validator(
        "activation_timestamp_utc",
        "signal_timestamp_utc",
        "entry_timestamp_utc",
        "receipt_created_at_utc",
    )
    @classmethod
    def _timestamps_are_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="prediction timestamp")

    @field_validator("stock")
    @classmethod
    def _canonical_stock(cls, value: str) -> str:
        canonical = value.strip().upper()
        if not canonical:
            raise ValueError("stock is required")
        return canonical

    @field_validator("frozen_comparisons", mode="before")
    @classmethod
    def _comparisons_are_deeply_frozen(
        cls,
        value: object,
    ) -> tuple[tuple[str, FrozenComparisonScalarV1], ...]:
        return _freeze_comparisons(value)


class OpeningReversalPredictionReceiptV1(BaseModel):
    """Immutable preregistered prediction; later outcomes never update this row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1", "1.1"]
    session: date
    stock: str
    checkpoint: Literal[6]
    signal_timestamp_utc: datetime
    entry_timestamp_utc: datetime
    receipt_created_at_utc: datetime
    m1c_probability: float | None
    m1c_threshold: float
    high_tail_membership: bool
    fresh_episode_id: str | None
    tail_phase_v1: str
    market_proxy_v1: Literal["VTI"]
    market_opening_return_v1: float | None
    market_opening_range_v1: float | None
    opening_market_transition_state_v1: OpeningStateV1
    opening_transition_sign_v1: Literal[-1, 1] | None
    opening_transition_event_id_v1: str | None
    negative_opening_return_threshold_v1: float
    positive_opening_return_threshold_v1: float
    opening_range_threshold_v1: float
    data_source: str
    transfer_status: str
    cohort_phase: CohortPhaseV1
    prediction_v1: PredictionV1
    prediction_sign_v1: Literal[-1, 0, 1]
    eligibility_v1: bool
    ineligibility_reasons_v1: tuple[str, ...]
    completeness_status_v1: Literal["complete", "incomplete"]
    scientific_outcome_eligible_v1: bool
    scientific_exclusion_reason_v1: str | None
    capacity_snapshot_id: str | None
    previous_close_atm_iv_scale_15m: float | None
    frozen_comparisons: tuple[
        tuple[str, FrozenComparisonScalarV1],
        ...,
    ]
    timing_evidence_v1_1: OpeningReversalPredictionTimingEvidenceV1_1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    rule_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _receipt_uses_exact_thresholds(
        self,
    ) -> OpeningReversalPredictionReceiptV1:
        if (
            self.m1c_threshold,
            self.negative_opening_return_threshold_v1,
            self.positive_opening_return_threshold_v1,
            self.opening_range_threshold_v1,
        ) != (
            M1C_HIGH_TAIL_THRESHOLD_V1,
            NEGATIVE_OPENING_RETURN_THRESHOLD_V1,
            POSITIVE_OPENING_RETURN_THRESHOLD_V1,
            OPENING_RANGE_THRESHOLD_V1,
        ):
            raise ValueError("prediction receipt threshold drift")
        if self.rule_hash_v1 != FrozenOpeningReversalRuleV1().rule_hash:
            raise ValueError("prediction receipt rule hash mismatch")
        if (self.experiment_version == "1" and self.timing_evidence_v1_1 is not None) or (
            self.experiment_version == "1.1" and self.timing_evidence_v1_1 is None
        ):
            raise ValueError("prediction receipt timing protocol mismatch")
        payload = self.model_dump(mode="python", exclude={"receipt_hash_v1"})
        if self.receipt_hash_v1 != _sha256(payload):
            raise ValueError("prediction receipt hash mismatch")
        return self

    @field_validator("frozen_comparisons", mode="before")
    @classmethod
    def _receipt_comparisons_are_deeply_frozen(
        cls,
        value: object,
    ) -> tuple[tuple[str, FrozenComparisonScalarV1], ...]:
        return _freeze_comparisons(value)


def _severe_state_from_values(
    *,
    opening_return: float | None,
    opening_range: float | None,
) -> tuple[OpeningStateV1, Literal[-1, 1] | None]:
    if (
        opening_return is None
        or opening_range is None
        or not math.isfinite(opening_return)
        or not math.isfinite(opening_range)
    ):
        return "UNKNOWN_INCOMPLETE", None
    if (
        opening_return <= NEGATIVE_OPENING_RETURN_THRESHOLD_V1
        and opening_range >= OPENING_RANGE_THRESHOLD_V1
    ):
        return "NEGATIVE_SEVERE_OPENING_TRANSITION", -1
    if (
        opening_return >= POSITIVE_OPENING_RETURN_THRESHOLD_V1
        and opening_range >= OPENING_RANGE_THRESHOLD_V1
    ):
        return "POSITIVE_SEVERE_OPENING_TRANSITION", 1
    if opening_range >= OPENING_RANGE_THRESHOLD_V1:
        return "ELEVATED_OPENING_RANGE_NONDIRECTIONAL", None
    return "NORMAL_OPENING", None


def build_prediction_receipt_v1(
    item: OpeningReversalPredictionInputV1,
    *,
    rule: FrozenOpeningReversalRuleV1 | None = None,
) -> OpeningReversalPredictionReceiptV1:
    """Apply the frozen rule exactly once and return a self-hashing receipt."""

    source = OpeningReversalPredictionInputV1.model_validate(item.model_dump(mode="python"))
    frozen_rule = FrozenOpeningReversalRuleV1() if rule is None else rule
    reasons: list[str] = []

    if source.signal_timestamp_utc <= source.activation_timestamp_utc:
        reasons.append("session_not_after_activation")
    if source.receipt_created_at_utc <= source.activation_timestamp_utc:
        reasons.append("receipt_not_after_activation")
    if source.checkpoint != frozen_rule.checkpoint:
        reasons.append("checkpoint_not_6")
    probability = source.m1c_probability
    probability_valid = (
        source.m1c_probability_valid
        and probability is not None
        and math.isfinite(probability)
        and 0.0 <= probability <= 1.0
    )
    if not probability_valid:
        reasons.append("m1c_probability_invalid")
    elif probability is not None and probability < frozen_rule.m1c_probability_threshold:
        reasons.append("m1c_below_frozen_high_tail")
    expected_high_tail = bool(
        probability_valid
        and probability is not None
        and probability >= frozen_rule.m1c_probability_threshold
    )
    if source.high_tail_membership != expected_high_tail:
        reasons.append("high_tail_membership_invalid")
    if not source.canonical_fresh_episode or source.fresh_episode_id is None:
        reasons.append("canonical_fresh_episode_required")
    if source.tail_phase_v1 != frozen_rule.tail_phase_required:
        reasons.append("tail_phase_not_first_entry")
    if not source.vti_opening_transition_complete:
        reasons.append("vti_opening_transition_incomplete")
    if not source.stock_causal_data_complete:
        reasons.append("stock_causal_data_incomplete")
    scale = source.previous_close_atm_iv_scale_15m
    if (
        not source.previous_close_atm_iv_scale_valid
        or scale is None
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        reasons.append("previous_close_atm_iv_scale_invalid")
    if (
        not source.transfer_status.strip()
        or "UNKNOWN_INCOMPLETE" in source.transfer_status.upper()
        or source.transfer_status.lower() == "missing"
    ):
        reasons.append("transfer_status_incomplete")
    if source.capacity_snapshot_id is None:
        reasons.append("capacity_snapshot_missing")
    timing = source.timing_evidence_v1_1
    if source.experiment_version == "1":
        if timing is not None:
            reasons.append("v1_timing_addendum_not_permitted")
        if source.receipt_created_at_utc >= source.entry_timestamp_utc:
            reasons.append("receipt_not_completed_before_entry")
    elif timing is None:
        reasons.append("timing_addendum_evidence_missing")
    else:
        expected_latency = (
            source.receipt_created_at_utc - source.entry_timestamp_utc
        ).total_seconds()
        if timing.rule_committed_at_utc != source.activation_timestamp_utc:
            reasons.append("timing_rule_commitment_mismatch")
        if timing.causal_barrier_armed_at_utc > source.signal_timestamp_utc:
            reasons.append("causal_barrier_not_armed_before_signal")
        if (
            timing.predictor_window_completed_at_utc != source.signal_timestamp_utc
            or timing.predictor_window_completed_at_utc != source.entry_timestamp_utc
        ):
            reasons.append("predictor_window_completion_mismatch")
        if source.receipt_created_at_utc < timing.predictor_window_completed_at_utc:
            reasons.append("receipt_created_before_predictor_complete")
        if timing.entry_or_post_entry_data_admitted_before_receipt:
            reasons.append("entry_or_post_entry_data_admitted_before_receipt")
        buffered_at = timing.first_entry_or_post_entry_event_buffered_at_utc
        if buffered_at is not None and buffered_at < source.entry_timestamp_utc:
            reasons.append("entry_buffer_timestamp_before_nominal_entry")
        if (
            buffered_at is not None
            and buffered_at > source.receipt_created_at_utc
        ):
            reasons.append("entry_buffer_timestamp_after_receipt")
        if expected_latency < 0.0 or not math.isclose(
            timing.receipt_latency_after_nominal_entry_seconds,
            expected_latency,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            reasons.append("timing_receipt_latency_mismatch")
    if source.signal_timestamp_utc != source.entry_timestamp_utc:
        reasons.append("frozen_signal_entry_timestamp_mismatch")

    derived_state, derived_sign = _severe_state_from_values(
        opening_return=source.market_opening_return_v1,
        opening_range=source.market_opening_range_v1,
    )
    if (
        source.opening_market_transition_state_v1 != derived_state
        or source.opening_transition_sign_v1 != derived_sign
    ):
        reasons.append("opening_state_threshold_mismatch")
    if derived_sign is not None and source.opening_transition_event_id_v1 is None:
        reasons.append("opening_transition_event_id_missing")
    if derived_sign is None:
        reasons.append("opening_state_not_severe")

    unique_reasons = tuple(dict.fromkeys(reasons))
    eligible = not unique_reasons
    incomplete_reasons = {
        "m1c_probability_invalid",
        "high_tail_membership_invalid",
        "vti_opening_transition_incomplete",
        "stock_causal_data_incomplete",
        "previous_close_atm_iv_scale_invalid",
        "transfer_status_incomplete",
        "capacity_snapshot_missing",
        "receipt_not_completed_before_entry",
        "timing_addendum_evidence_missing",
        "timing_rule_commitment_mismatch",
        "causal_barrier_not_armed_before_signal",
        "predictor_window_completion_mismatch",
        "receipt_created_before_predictor_complete",
        "entry_or_post_entry_data_admitted_before_receipt",
        "entry_buffer_timestamp_before_nominal_entry",
        "entry_buffer_timestamp_after_receipt",
        "timing_receipt_latency_mismatch",
        "opening_state_threshold_mismatch",
        "opening_transition_event_id_missing",
    }
    complete = not incomplete_reasons.intersection(unique_reasons)
    prediction: PredictionV1 = "ABSTAIN"
    prediction_sign: Literal[-1, 0, 1] = 0
    if eligible and derived_sign == -1:
        prediction = "CALL"
        prediction_sign = 1
    elif eligible and derived_sign == 1:
        prediction = "PUT"
        prediction_sign = -1

    scientific = eligible and source.cohort_phase != "engineering_transfer"
    scientific_exclusion = (
        None
        if scientific
        else "engineering_transfer"
        if eligible and source.cohort_phase == "engineering_transfer"
        else "prediction_ineligible"
    )
    payload: dict[str, object] = {
        "experiment_id": frozen_rule.experiment_id,
        "experiment_version": source.experiment_version,
        "session": source.session,
        "stock": source.stock,
        "checkpoint": 6,
        "signal_timestamp_utc": source.signal_timestamp_utc,
        "entry_timestamp_utc": source.entry_timestamp_utc,
        "receipt_created_at_utc": source.receipt_created_at_utc,
        "m1c_probability": probability,
        "m1c_threshold": frozen_rule.m1c_probability_threshold,
        "high_tail_membership": source.high_tail_membership,
        "fresh_episode_id": source.fresh_episode_id,
        "tail_phase_v1": source.tail_phase_v1,
        "market_proxy_v1": frozen_rule.market_proxy,
        "market_opening_return_v1": source.market_opening_return_v1,
        "market_opening_range_v1": source.market_opening_range_v1,
        "opening_market_transition_state_v1": (source.opening_market_transition_state_v1),
        "opening_transition_sign_v1": source.opening_transition_sign_v1,
        "opening_transition_event_id_v1": source.opening_transition_event_id_v1,
        "negative_opening_return_threshold_v1": frozen_rule.opening_return_q10,
        "positive_opening_return_threshold_v1": frozen_rule.opening_return_q90,
        "opening_range_threshold_v1": frozen_rule.opening_range_q75,
        "data_source": source.data_source,
        "transfer_status": source.transfer_status,
        "cohort_phase": source.cohort_phase,
        "prediction_v1": prediction,
        "prediction_sign_v1": prediction_sign,
        "eligibility_v1": eligible,
        "ineligibility_reasons_v1": unique_reasons,
        "completeness_status_v1": "complete" if complete else "incomplete",
        "scientific_outcome_eligible_v1": scientific,
        "scientific_exclusion_reason_v1": scientific_exclusion,
        "capacity_snapshot_id": source.capacity_snapshot_id,
        "previous_close_atm_iv_scale_15m": scale,
        "frozen_comparisons": source.frozen_comparisons,
        "rule_hash_v1": frozen_rule.rule_hash,
    }
    if timing is not None:
        payload["timing_evidence_v1_1"] = timing
    payload["receipt_hash_v1"] = _sha256(payload)
    return OpeningReversalPredictionReceiptV1.model_validate(payload)


class NonPromotedEligiblePredictionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stock: str
    m1c_probability: float
    prediction_v1: PredictionV1
    opening_transition_state_v1: OpeningStateV1
    capacity_snapshot_id: str | None
    winning_promoted_stock: str
    reason_not_promoted_v1: Literal["lower_frozen_promotion_rank"]
    receipt_hash_v1: str


class PromotionSelectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    promoted: OpeningReversalPredictionReceiptV1 | None
    non_promoted: tuple[NonPromotedEligiblePredictionV1, ...]
    eligible_count: int
    maximum_promoted_count: Literal[1] = 1
    selection_rule: Literal["m1c_probability_desc_receipt_time_asc_ticker_asc"] = (
        "m1c_probability_desc_receipt_time_asc_ticker_asc"
    )


def select_promoted_prediction_v1(
    receipts: Iterable[OpeningReversalPredictionReceiptV1],
) -> PromotionSelectionV1:
    """Promote at most one eligible episode without looking at option data."""

    candidates = tuple(
        receipt
        for receipt in receipts
        if receipt.eligibility_v1 and receipt.prediction_v1 != "ABSTAIN"
    )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda value: (
                -cast(float, value.m1c_probability),
                value.receipt_created_at_utc,
                value.stock,
            ),
        )
    )
    if not ordered:
        return PromotionSelectionV1(
            promoted=None,
            non_promoted=(),
            eligible_count=0,
        )
    winner = ordered[0]
    return PromotionSelectionV1(
        promoted=winner,
        non_promoted=tuple(
            NonPromotedEligiblePredictionV1(
                stock=receipt.stock,
                m1c_probability=cast(float, receipt.m1c_probability),
                prediction_v1=receipt.prediction_v1,
                opening_transition_state_v1=(receipt.opening_market_transition_state_v1),
                capacity_snapshot_id=receipt.capacity_snapshot_id,
                winning_promoted_stock=winner.stock,
                reason_not_promoted_v1="lower_frozen_promotion_rank",
                receipt_hash_v1=receipt.receipt_hash_v1,
            )
            for receipt in ordered[1:]
        ),
        eligible_count=len(ordered),
    )


class PostEntryBarV1(BaseModel):
    """One complete bar in the fixed 10:00--10:15 outcome window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal: int
    bar_start_timestamp_utc: datetime
    bar_complete_timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    finalised: bool

    @field_validator("bar_start_timestamp_utc", "bar_complete_timestamp_utc")
    @classmethod
    def _bar_timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="outcome bar timestamp")

    @model_validator(mode="after")
    def _valid_ohlc(self) -> PostEntryBarV1:
        prices = (self.open, self.high, self.low, self.close)
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in prices)
            or self.high < max(self.open, self.close, self.low)
            or self.low > min(self.open, self.close, self.high)
        ):
            raise ValueError("post-entry bar OHLC is invalid")
        return self


class OpeningReversalUnderlyingOutcomeV1(BaseModel):
    """Immutable underlying result, linked by hash to its prediction receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1", "1.1"]
    prediction_receipt_hash_v1: str
    fresh_episode_id: str
    opening_transition_event_id_v1: str
    session: date
    stock: str
    prediction_v1: Literal["CALL", "PUT"]
    prediction_sign_v1: Literal[-1, 1]
    entry_timestamp_utc: datetime
    terminal_timestamp_utc: datetime
    r_15m: float | None
    absolute_return_15m: float | None
    threshold_15m: float
    outcome_state_v1: MaterialOutcomeV1 | None
    opening_reversal_aligned_return_v1: float | None
    correct_predicted_material_direction_v1: bool | None
    accuracy_counting_no_move_as_failure_v1: bool | None
    maximum_favourable_excursion_v1: float | None
    maximum_adverse_excursion_v1: float | None
    canonical_post_entry_local_range_share_v1: float | None
    iv_residual_v1: float | None
    exceed_iv_v1: bool | None
    outcome_completeness_v1: Literal["complete", "incomplete"]
    missing_reason_v1: str | None
    outcome_created_at_utc: datetime
    outcome_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _complete_or_explicitly_missing(
        self,
    ) -> OpeningReversalUnderlyingOutcomeV1:
        required_results = (
            self.r_15m,
            self.absolute_return_15m,
            self.outcome_state_v1,
            self.opening_reversal_aligned_return_v1,
            self.accuracy_counting_no_move_as_failure_v1,
            self.maximum_favourable_excursion_v1,
            self.maximum_adverse_excursion_v1,
            self.iv_residual_v1,
            self.exceed_iv_v1,
        )
        if self.outcome_completeness_v1 == "complete":
            if self.missing_reason_v1 is not None or any(
                value is None for value in required_results
            ):
                raise ValueError("complete opening reversal outcome is missing")
        elif not self.missing_reason_v1 or any(value is not None for value in required_results):
            raise ValueError("incomplete opening reversal outcome is not fail-closed")
        payload = self.model_dump(
            mode="python",
            exclude={"outcome_receipt_hash_v1"},
        )
        if self.outcome_receipt_hash_v1 != _sha256(payload):
            raise ValueError("underlying outcome receipt hash mismatch")
        return self


def partition_material_outcome_v1(
    signed_return: float,
    threshold: float,
) -> MaterialOutcomeV1:
    if signed_return > threshold:
        return "MATERIAL_UP"
    if signed_return < -threshold:
        return "MATERIAL_DOWN"
    return "NO_MATERIAL_MOVE"


def build_opening_reversal_outcome_v1(
    *,
    prediction_receipt: OpeningReversalPredictionReceiptV1,
    completed_post_entry_bars: Sequence[PostEntryBarV1],
    threshold_15m: float,
    outcome_created_at_utc: datetime,
    canonical_post_entry_local_range_share_v1: float | None = None,
) -> OpeningReversalUnderlyingOutcomeV1:
    """Calculate the fixed 15-minute outcome without changing the prediction."""

    receipt = OpeningReversalPredictionReceiptV1.model_validate(
        prediction_receipt.model_dump(mode="python")
    )
    if not receipt.scientific_outcome_eligible_v1:
        raise ValueError("engineering_or_ineligible_outcome_forbidden")
    if receipt.prediction_v1 == "ABSTAIN" or receipt.prediction_sign_v1 == 0:
        raise ValueError("abstention_has_no_direction_outcome")
    if (
        threshold_15m <= 0.0
        or not math.isfinite(threshold_15m)
        or receipt.previous_close_atm_iv_scale_15m != threshold_15m
    ):
        raise ValueError("threshold_15m_differs_from_frozen_receipt")
    bars = tuple(
        PostEntryBarV1.model_validate(bar.model_dump(mode="python"))
        for bar in completed_post_entry_bars
    )
    if len(bars) != 3 or tuple(bar.ordinal for bar in bars) != (0, 1, 2):
        raise ValueError("outcome_requires_exactly_three_post_entry_bars")
    for index, bar in enumerate(bars):
        expected_start = receipt.entry_timestamp_utc + timedelta(minutes=5 * index)
        if (
            not bar.finalised
            or bar.bar_start_timestamp_utc != expected_start
            or bar.bar_complete_timestamp_utc != expected_start + timedelta(minutes=5)
        ):
            raise ValueError("post_entry_bars_incomplete_or_non_contiguous")
    created = _aware_utc(outcome_created_at_utc, label="outcome creation timestamp")
    if created < bars[-1].bar_complete_timestamp_utc:
        raise ValueError("outcome_created_before_horizon_complete")

    entry = bars[0].open
    signed_return = math.log(bars[-1].close / entry)
    state = partition_material_outcome_v1(signed_return, threshold_15m)
    sign = receipt.prediction_sign_v1
    correct = (
        None
        if state == "NO_MATERIAL_MOVE"
        else (state == "MATERIAL_UP" and sign == 1) or (state == "MATERIAL_DOWN" and sign == -1)
    )
    log_high = max(math.log(bar.high / entry) for bar in bars)
    log_low = min(math.log(bar.low / entry) for bar in bars)
    if sign == 1:
        favourable = log_high
        adverse = log_low
    else:
        favourable = -log_low
        adverse = -log_high
    payload: dict[str, object] = {
        "experiment_id": receipt.experiment_id,
        "experiment_version": receipt.experiment_version,
        "prediction_receipt_hash_v1": receipt.receipt_hash_v1,
        "fresh_episode_id": cast(str, receipt.fresh_episode_id),
        "opening_transition_event_id_v1": cast(
            str,
            receipt.opening_transition_event_id_v1,
        ),
        "session": receipt.session,
        "stock": receipt.stock,
        "prediction_v1": receipt.prediction_v1,
        "prediction_sign_v1": sign,
        "entry_timestamp_utc": receipt.entry_timestamp_utc,
        "terminal_timestamp_utc": bars[-1].bar_complete_timestamp_utc,
        "r_15m": signed_return,
        "absolute_return_15m": abs(signed_return),
        "threshold_15m": threshold_15m,
        "outcome_state_v1": state,
        "opening_reversal_aligned_return_v1": sign * signed_return,
        "correct_predicted_material_direction_v1": correct,
        "accuracy_counting_no_move_as_failure_v1": bool(correct),
        "maximum_favourable_excursion_v1": favourable,
        "maximum_adverse_excursion_v1": adverse,
        "canonical_post_entry_local_range_share_v1": (canonical_post_entry_local_range_share_v1),
        "iv_residual_v1": abs(signed_return) - threshold_15m,
        "exceed_iv_v1": state in {"MATERIAL_UP", "MATERIAL_DOWN"},
        "outcome_completeness_v1": "complete",
        "missing_reason_v1": None,
        "outcome_created_at_utc": created,
    }
    payload["outcome_receipt_hash_v1"] = _sha256(payload)
    return OpeningReversalUnderlyingOutcomeV1.model_validate(payload)


def build_incomplete_opening_reversal_outcome_v1(
    *,
    prediction_receipt: OpeningReversalPredictionReceiptV1,
    missing_reason_v1: str,
    outcome_created_at_utc: datetime,
) -> OpeningReversalUnderlyingOutcomeV1:
    """Record missing outcome evidence without inventing a return."""

    receipt = OpeningReversalPredictionReceiptV1.model_validate(
        prediction_receipt.model_dump(mode="python")
    )
    if not receipt.scientific_outcome_eligible_v1:
        raise ValueError("engineering_or_ineligible_outcome_forbidden")
    if receipt.prediction_v1 == "ABSTAIN" or receipt.prediction_sign_v1 == 0:
        raise ValueError("abstention_has_no_direction_outcome")
    threshold = receipt.previous_close_atm_iv_scale_15m
    if threshold is None or threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError("frozen_threshold_unavailable")
    reason = missing_reason_v1.strip()
    if not reason:
        raise ValueError("missing outcome reason is required")
    created = _aware_utc(outcome_created_at_utc, label="outcome creation timestamp")
    payload: dict[str, object] = {
        "experiment_id": receipt.experiment_id,
        "experiment_version": receipt.experiment_version,
        "prediction_receipt_hash_v1": receipt.receipt_hash_v1,
        "fresh_episode_id": cast(str, receipt.fresh_episode_id),
        "opening_transition_event_id_v1": cast(
            str,
            receipt.opening_transition_event_id_v1,
        ),
        "session": receipt.session,
        "stock": receipt.stock,
        "prediction_v1": receipt.prediction_v1,
        "prediction_sign_v1": receipt.prediction_sign_v1,
        "entry_timestamp_utc": receipt.entry_timestamp_utc,
        "terminal_timestamp_utc": (receipt.entry_timestamp_utc + timedelta(minutes=15)),
        "r_15m": None,
        "absolute_return_15m": None,
        "threshold_15m": threshold,
        "outcome_state_v1": None,
        "opening_reversal_aligned_return_v1": None,
        "correct_predicted_material_direction_v1": None,
        "accuracy_counting_no_move_as_failure_v1": None,
        "maximum_favourable_excursion_v1": None,
        "maximum_adverse_excursion_v1": None,
        "canonical_post_entry_local_range_share_v1": None,
        "iv_residual_v1": None,
        "exceed_iv_v1": None,
        "outcome_completeness_v1": "incomplete",
        "missing_reason_v1": reason,
        "outcome_created_at_utc": created,
    }
    payload["outcome_receipt_hash_v1"] = _sha256(payload)
    return OpeningReversalUnderlyingOutcomeV1.model_validate(payload)


class OpeningEventAccountingRowV1(BaseModel):
    """One unique event/population membership row for the count audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    period: str
    session: date
    checkpoint: int
    vti_proxy: str
    opening_state: str
    opening_sign: Literal[-1, 1]
    event_id: str
    eligible_stock_count: int = Field(ge=0)
    acted_stock_count: int = Field(ge=0)
    included_in_total_event_count: bool
    included_in_positive_count: bool
    included_in_negative_count: bool
    exact_explanation: str


class OpeningEventAccountingReconciliationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: EventReconciliationOutcomeV1
    unique_signed_event_count: int
    negative_unique_event_count: int
    positive_unique_event_count: int
    errors: tuple[str, ...]


def reconcile_opening_event_accounting_v1(
    rows: Iterable[OpeningEventAccountingRowV1],
) -> OpeningEventAccountingReconciliationV1:
    """Enforce one sign per VTI event and one common count population."""

    items = tuple(
        OpeningEventAccountingRowV1.model_validate(row.model_dump(mode="python")) for row in rows
    )
    signs_by_event: dict[str, set[int]] = {}
    errors: list[str] = []
    for row in items:
        signs_by_event.setdefault(row.event_id, set()).add(row.opening_sign)
        if row.included_in_positive_count and row.included_in_negative_count:
            errors.append(f"event_in_both_sign_counts:{row.event_id}")
        expected_positive = row.included_in_total_event_count and row.opening_sign == 1
        expected_negative = row.included_in_total_event_count and row.opening_sign == -1
        if (
            row.included_in_positive_count != expected_positive
            or row.included_in_negative_count != expected_negative
        ):
            errors.append(f"event_population_membership_mismatch:{row.event_id}")
    errors.extend(
        f"event_has_multiple_transition_signs:{event_id}"
        for event_id, signs in sorted(signs_by_event.items())
        if len(signs) != 1
    )
    included = {row.event_id: row for row in items if row.included_in_total_event_count}
    negative = {row.event_id for row in items if row.included_in_negative_count}
    positive = {row.event_id for row in items if row.included_in_positive_count}
    if len(negative) + len(positive) != len(included):
        errors.append("positive_plus_negative_does_not_equal_total_signed_events")
    unique_errors = tuple(dict.fromkeys(errors))
    return OpeningEventAccountingReconciliationV1(
        outcome=(
            "blocked_event_accounting" if unique_errors else "event_accounting_fully_reconciled"
        ),
        unique_signed_event_count=len(included),
        negative_unique_event_count=len(negative),
        positive_unique_event_count=len(positive),
        errors=unique_errors,
    )


class OpeningTransferBarV1(BaseModel):
    """One predictor-only source bar used in the 20-session transfer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal: int
    bar_start_timestamp_utc: datetime
    bar_complete_timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    complete: bool

    @field_validator("bar_start_timestamp_utc", "bar_complete_timestamp_utc")
    @classmethod
    def _transfer_bar_timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="opening transfer bar timestamp")

    @model_validator(mode="after")
    def _transfer_bar_is_valid(self) -> OpeningTransferBarV1:
        prices = (self.open, self.high, self.low, self.close)
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in prices)
            or self.high < max(self.open, self.close, self.low)
            or self.low > min(self.open, self.close, self.high)
        ):
            raise ValueError("opening transfer OHLC is invalid")
        return self


class OpeningTransferOperationalEvidenceV1(BaseModel):
    """Outcome-free recorder safeguards required for one valid transfer day."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_receipt_count: int = Field(ge=0)
    expected_prediction_receipt_count: Literal[20] = 20
    prediction_receipt_timing_pass: bool
    prediction_receipt_immutability_pass: bool
    capacity_snapshot_count: int = Field(ge=0)
    capacity_snapshots_complete: bool
    reserved_twelve_lines_pass: bool
    promoted_episode_count: int = Field(ge=0, le=1)
    promoted_underlying_level1_pass: bool
    contract_discovery_audit_count: int = Field(ge=0)
    contract_discovery_complete: bool
    primary_option_pair_available_count: int = Field(ge=0)
    primary_option_pair_recording_pass: bool
    graceful_degradation_pass: bool
    cancellation_recovery_pass: bool
    m1c_universe_uninterrupted: bool
    recorder_reliability_pass: bool
    no_order_guard_pass: bool
    orders_placed: Literal[0] = 0
    critical_checks_pass: bool
    missing_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _critical_gate_is_derived(
        self,
    ) -> OpeningTransferOperationalEvidenceV1:
        checks = (
            self.prediction_receipt_count == self.expected_prediction_receipt_count,
            self.prediction_receipt_timing_pass,
            self.prediction_receipt_immutability_pass,
            self.capacity_snapshots_complete,
            self.reserved_twelve_lines_pass,
            self.promoted_underlying_level1_pass,
            self.contract_discovery_complete,
            self.primary_option_pair_recording_pass,
            self.graceful_degradation_pass,
            self.cancellation_recovery_pass,
            self.m1c_universe_uninterrupted,
            self.recorder_reliability_pass,
            self.no_order_guard_pass,
            self.orders_placed == 0,
        )
        expected = all(checks) and not self.missing_reasons
        if self.critical_checks_pass != expected:
            raise ValueError("opening transfer operational gate is inconsistent")
        return self


def missing_opening_transfer_operational_evidence_v1(
    reason: str = "engineering_operational_evidence_missing",
) -> OpeningTransferOperationalEvidenceV1:
    """Fail closed when the live engineering evidence has not been supplied."""

    return OpeningTransferOperationalEvidenceV1(
        prediction_receipt_count=0,
        prediction_receipt_timing_pass=False,
        prediction_receipt_immutability_pass=False,
        capacity_snapshot_count=0,
        capacity_snapshots_complete=False,
        reserved_twelve_lines_pass=False,
        promoted_episode_count=0,
        promoted_underlying_level1_pass=False,
        contract_discovery_audit_count=0,
        contract_discovery_complete=False,
        primary_option_pair_available_count=0,
        primary_option_pair_recording_pass=False,
        graceful_degradation_pass=False,
        cancellation_recovery_pass=False,
        m1c_universe_uninterrupted=False,
        recorder_reliability_pass=False,
        no_order_guard_pass=False,
        critical_checks_pass=False,
        missing_reasons=(reason,),
    )


class OpeningTransferSessionResultV1(BaseModel):
    """Engineering-only IBKR/EODHD checkpoint-6 opening comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: date
    decision: Literal[
        "opening_transfer_supported_without_recalibration",
        "opening_transfer_supported_with_predictor_only_mapping",
        "opening_transfer_mixed",
        "opening_transfer_not_supported",
        "opening_transfer_operational_failure",
    ]
    valid: bool
    ibkr_opening_return: float | None
    eodhd_opening_return: float | None
    opening_return_absolute_difference: float | None
    ibkr_opening_range: float | None
    eodhd_opening_range: float | None
    opening_range_absolute_difference: float | None
    ibkr_opening_state: OpeningStateV1
    eodhd_opening_state: OpeningStateV1
    severe_state_agreement: bool
    sign_agreement: bool
    bar_timestamp_alignment: bool
    missingness_agreement: bool
    threshold_boundary_disagreement: bool
    checkpoint_6_episode_identity_agreement: bool
    stock_probability_rank_comparison_available: bool
    operational_evidence: OpeningTransferOperationalEvidenceV1
    exact_ohlc_equality_required: Literal[False]
    predictor_only_mapping_used: Literal[False]
    future_stock_returns_accessed: Literal[False]
    direction_outcomes_accessed: Literal[False]
    m1c_outcomes_accessed: Literal[False]
    option_pnl_accessed: Literal[False]
    missing_reasons: tuple[str, ...]
    report_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _report_hash_matches(self) -> OpeningTransferSessionResultV1:
        payload = self.model_dump(mode="python", exclude={"report_hash_v1"})
        if self.report_hash_v1 != _sha256(payload):
            raise ValueError("opening transfer report hash mismatch")
        return self


def _opening_transfer_measurement(
    *,
    session: date,
    bars: Sequence[OpeningTransferBarV1],
) -> tuple[float | None, float | None, tuple[str, ...]]:
    rows = tuple(OpeningTransferBarV1.model_validate(bar.model_dump(mode="python")) for bar in bars)
    reasons: list[str] = []
    if len(rows) != 6 or tuple(row.ordinal for row in rows) != tuple(range(6)):
        reasons.append("opening_transfer_requires_exactly_six_bars")
    if rows:
        try:
            expected_start, _ = xnys_session_bounds(session)
        except ValueError:
            reasons.append("opening_transfer_invalid_xnys_session")
        else:
            expected_end = expected_start + timedelta(minutes=30)
            if (
                rows[0].bar_start_timestamp_utc != expected_start
                or rows[-1].bar_complete_timestamp_utc != expected_end
            ):
                reasons.append("opening_transfer_wrong_session_window")
        for index, row in enumerate(rows):
            if not row.complete:
                reasons.append(f"opening_transfer_partial_bar:{index}")
            if row.bar_complete_timestamp_utc != (
                row.bar_start_timestamp_utc + timedelta(minutes=5)
            ):
                reasons.append(f"opening_transfer_invalid_bar_duration:{index}")
            if (
                index > 0
                and row.bar_start_timestamp_utc != rows[index - 1].bar_complete_timestamp_utc
            ):
                reasons.append(f"opening_transfer_non_contiguous_bar:{index}")
    if reasons:
        return None, None, tuple(dict.fromkeys(reasons))
    opening_return = math.log(rows[-1].close / rows[0].open)
    opening_range = math.log(max(row.high for row in rows) / min(row.low for row in rows))
    return opening_return, opening_range, ()


def evaluate_opening_transfer_session_v1(
    *,
    session: date,
    ibkr_bars: Sequence[OpeningTransferBarV1],
    eodhd_bars: Sequence[OpeningTransferBarV1],
    checkpoint_6_episode_identity_agreement: bool,
    stock_probability_rank_comparison_available: bool,
    operational_evidence: OpeningTransferOperationalEvidenceV1 | None = None,
) -> OpeningTransferSessionResultV1:
    """Compare predictor values only; no outcome value is accepted by the API."""

    ibkr_return, ibkr_range, ibkr_reasons = _opening_transfer_measurement(
        session=session,
        bars=ibkr_bars,
    )
    eodhd_return, eodhd_range, eodhd_reasons = _opening_transfer_measurement(
        session=session,
        bars=eodhd_bars,
    )
    ibkr_state, ibkr_sign = _severe_state_from_values(
        opening_return=ibkr_return,
        opening_range=ibkr_range,
    )
    eodhd_state, eodhd_sign = _severe_state_from_values(
        opening_return=eodhd_return,
        opening_range=eodhd_range,
    )
    timestamp_alignment = len(ibkr_bars) == len(eodhd_bars) == 6 and all(
        left.ordinal == right.ordinal
        and left.bar_start_timestamp_utc == right.bar_start_timestamp_utc
        and left.bar_complete_timestamp_utc == right.bar_complete_timestamp_utc
        for left, right in zip(ibkr_bars, eodhd_bars, strict=True)
    )
    complete = (
        not ibkr_reasons
        and not eodhd_reasons
        and timestamp_alignment
        and ibkr_return is not None
        and ibkr_range is not None
        and eodhd_return is not None
        and eodhd_range is not None
    )
    state_agreement = complete and ibkr_state == eodhd_state
    sign_agreement = complete and ibkr_sign == eodhd_sign
    boundary_disagreement = complete and (ibkr_state != eodhd_state or ibkr_sign != eodhd_sign)
    # A source-complete session counts toward the fixed engineering cohort even
    # when it reveals disagreement.  Excluding disagreements would select the
    # transfer cohort on the transfer result itself.
    engineering = (
        missing_opening_transfer_operational_evidence_v1()
        if operational_evidence is None
        else OpeningTransferOperationalEvidenceV1.model_validate(
            operational_evidence.model_dump(mode="python")
        )
    )
    valid = (
        complete
        and stock_probability_rank_comparison_available
        and engineering.critical_checks_pass
    )
    if valid and state_agreement and sign_agreement and checkpoint_6_episode_identity_agreement:
        decision = "opening_transfer_supported_without_recalibration"
    elif (
        not complete
        or not stock_probability_rank_comparison_available
        or not engineering.critical_checks_pass
    ):
        decision = "opening_transfer_operational_failure"
    elif boundary_disagreement:
        decision = "opening_transfer_mixed"
    else:
        decision = "opening_transfer_not_supported"
    reasons = tuple(
        dict.fromkeys(
            (
                *ibkr_reasons,
                *eodhd_reasons,
                *engineering.missing_reasons,
                *(
                    ()
                    if stock_probability_rank_comparison_available
                    else ("stock_probability_rank_comparison_unavailable",)
                ),
            )
        )
    )
    payload: dict[str, object] = {
        "session": session,
        "decision": decision,
        "valid": valid,
        "ibkr_opening_return": ibkr_return,
        "eodhd_opening_return": eodhd_return,
        "opening_return_absolute_difference": (
            None if ibkr_return is None or eodhd_return is None else abs(ibkr_return - eodhd_return)
        ),
        "ibkr_opening_range": ibkr_range,
        "eodhd_opening_range": eodhd_range,
        "opening_range_absolute_difference": (
            None if ibkr_range is None or eodhd_range is None else abs(ibkr_range - eodhd_range)
        ),
        "ibkr_opening_state": ibkr_state,
        "eodhd_opening_state": eodhd_state,
        "severe_state_agreement": state_agreement,
        "sign_agreement": sign_agreement,
        "bar_timestamp_alignment": timestamp_alignment,
        "missingness_agreement": bool(ibkr_reasons) == bool(eodhd_reasons),
        "threshold_boundary_disagreement": boundary_disagreement,
        "checkpoint_6_episode_identity_agreement": (checkpoint_6_episode_identity_agreement),
        "stock_probability_rank_comparison_available": (
            stock_probability_rank_comparison_available
        ),
        "operational_evidence": engineering,
        "exact_ohlc_equality_required": False,
        "predictor_only_mapping_used": False,
        "future_stock_returns_accessed": False,
        "direction_outcomes_accessed": False,
        "m1c_outcomes_accessed": False,
        "option_pnl_accessed": False,
        "missing_reasons": reasons,
    }
    payload["report_hash_v1"] = _sha256(payload)
    return OpeningTransferSessionResultV1.model_validate(payload)


OpeningReversalDecisionKindV1 = Literal[
    "transfer",
    "development",
    "confirmation_start",
    "confirmation",
    "option_economics",
]

TRANSFER_DECISIONS_V1 = frozenset(
    {
        "opening_transfer_supported_without_recalibration",
        "opening_transfer_supported_with_predictor_only_mapping",
        "opening_transfer_mixed",
        "opening_transfer_not_supported",
        "opening_transfer_operational_failure",
    }
)
DEVELOPMENT_DECISIONS_V1 = frozenset(
    {
        "prospective_opening_reversal_development_supported",
        "prospective_opening_reversal_development_not_supported",
        "blocked_insufficient_prospective_development_support",
        "blocked_opening_transfer",
        "operational_failure",
    }
)
CONFIRMATION_DECISIONS_V1 = frozenset(
    {
        "prospective_opening_reversal_direction_supported",
        "prospective_opening_reversal_ranking_only",
        "prospective_opening_reversal_not_supported",
        "blocked_insufficient_confirmation_support",
        "blocked_opening_transfer",
        "operational_failure",
    }
)
OPTION_ECONOMICS_DECISIONS_V1 = frozenset(
    {
        "prospective_opening_reversal_option_economics_supported",
        "direction_supported_without_option_edge",
        "option_economics_blocked_insufficient_bid_ask_support",
        "option_economics_blocked_capacity",
        "option_economics_not_supported",
    }
)
COHORT_SUPPORT_COUNT_KEYS_V1 = frozenset(
    {
        "complete_eligible_stock_episodes",
        "unique_severe_opening_events",
        "positive_transition_events",
        "negative_transition_events",
        "represented_stocks",
        "sessions",
        "maximum_stock_episode_count",
        "maximum_event_episode_count",
    }
)
OPTION_SUPPORT_COUNT_KEYS_V1 = frozenset(
    {
        "complete_promoted_option_episodes",
        "call_option_episodes",
        "put_option_episodes",
        "unique_severe_opening_events",
        "represented_stocks",
        "represented_expiries",
        "maximum_stock_episode_count",
        "maximum_expiry_episode_count",
        "maximum_event_episode_count",
    }
)


class OpeningReversalDecisionReceiptV1(BaseModel):
    """Immutable phase boundary or decision, separate from prediction rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1", "1.1"]
    receipt_kind: OpeningReversalDecisionKindV1
    boundary_timestamp_utc: datetime
    decision: str = Field(min_length=1)
    cohort_first_session: date | None
    cohort_last_session: date | None
    source_receipt_hashes: tuple[str, ...]
    support_counts: tuple[tuple[str, int], ...]
    protected_outcome_fields_accessed: bool
    receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("boundary_timestamp_utc")
    @classmethod
    def _boundary_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="decision boundary timestamp")

    @model_validator(mode="after")
    def _immutable_decision_contract(self) -> OpeningReversalDecisionReceiptV1:
        if (
            self.receipt_kind in {"transfer", "confirmation_start"}
            and self.protected_outcome_fields_accessed
        ):
            raise ValueError(f"{self.receipt_kind} receipt cannot access protected outcomes")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_receipt_hashes
        ):
            raise ValueError("decision source receipt hash is invalid")
        if len(self.source_receipt_hashes) != len(set(self.source_receipt_hashes)):
            raise ValueError("decision source receipt hashes must be unique")
        if tuple(sorted(self.support_counts)) != self.support_counts:
            raise ValueError("decision support counts must be canonical")
        counts = dict(self.support_counts)
        if self.receipt_kind == "transfer":
            if (
                self.decision not in TRANSFER_DECISIONS_V1
                or len(self.source_receipt_hashes) != ENGINEERING_TRANSFER_SESSION_COUNT_V1
                or counts.get("valid_sessions") != ENGINEERING_TRANSFER_SESSION_COUNT_V1
                or counts.get("operational_sessions_passed")
                != ENGINEERING_TRANSFER_SESSION_COUNT_V1
                or self.cohort_first_session is None
            ):
                raise ValueError("transfer decision receipt is incomplete")
        elif self.receipt_kind in {"development", "confirmation"}:
            allowed = (
                DEVELOPMENT_DECISIONS_V1
                if self.receipt_kind == "development"
                else CONFIRMATION_DECISIONS_V1
            )
            if (
                self.decision not in allowed
                or set(counts) != COHORT_SUPPORT_COUNT_KEYS_V1
                or counts["complete_eligible_stock_episodes"] != len(self.source_receipt_hashes)
                or self.cohort_first_session is None
                or not self.protected_outcome_fields_accessed
            ):
                raise ValueError(f"{self.receipt_kind} decision receipt is incomplete")
            if self.decision in {
                "prospective_opening_reversal_development_supported",
                "prospective_opening_reversal_direction_supported",
            }:
                episode_count = counts["complete_eligible_stock_episodes"]
                supported = (
                    episode_count >= 150
                    and counts["unique_severe_opening_events"] >= 40
                    and counts["positive_transition_events"] >= 15
                    and counts["negative_transition_events"] >= 15
                    and counts["represented_stocks"] >= 12
                    and counts["sessions"] >= 40
                    and counts["maximum_stock_episode_count"] <= 0.20 * episode_count
                    and counts["maximum_event_episode_count"] <= 0.15 * episode_count
                )
                if not supported:
                    raise ValueError(f"{self.receipt_kind} supported decision fails support gates")
        elif self.receipt_kind == "confirmation_start":
            if (
                self.decision != "untouched_confirmation_started"
                or len(self.source_receipt_hashes) != 1
                or counts
                or self.cohort_first_session is None
                or self.cohort_first_session != self.cohort_last_session
            ):
                raise ValueError("confirmation-start receipt is incomplete")
        else:
            if (
                self.decision not in OPTION_ECONOMICS_DECISIONS_V1
                or self.cohort_first_session is None
                or not self.protected_outcome_fields_accessed
                or set(counts) != OPTION_SUPPORT_COUNT_KEYS_V1
                or len(self.source_receipt_hashes)
                != 2 * counts["complete_promoted_option_episodes"] + 1
            ):
                raise ValueError("option-economics decision receipt is incomplete")
            if self.decision == "prospective_opening_reversal_option_economics_supported":
                episode_count = counts["complete_promoted_option_episodes"]
                if not (
                    episode_count >= 100
                    and counts["call_option_episodes"] >= 30
                    and counts["put_option_episodes"] >= 30
                    and counts["unique_severe_opening_events"] >= 30
                    and counts["maximum_stock_episode_count"] <= 0.20 * episode_count
                    and counts["maximum_expiry_episode_count"] <= 0.20 * episode_count
                    and counts["maximum_event_episode_count"] <= 0.15 * episode_count
                ):
                    raise ValueError("option-economics supported decision fails support gates")
        payload = self.model_dump(mode="python", exclude={"receipt_hash_v1"})
        if self.receipt_hash_v1 != _sha256(payload):
            raise ValueError("decision receipt hash mismatch")
        return self


def build_opening_reversal_decision_receipt_v1(
    *,
    experiment_version: Literal["1", "1.1"] = "1",
    receipt_kind: OpeningReversalDecisionKindV1,
    boundary_timestamp_utc: datetime,
    decision: str,
    cohort_first_session: date | None,
    cohort_last_session: date | None,
    source_receipt_hashes: Sequence[str],
    support_counts: Mapping[str, int],
    protected_outcome_fields_accessed: bool,
) -> OpeningReversalDecisionReceiptV1:
    """Hash a decision after its kind-specific facts have been assembled.

    Persistence performs a second validation against the immutable source rows;
    callers cannot advance a phase using this hash builder alone.
    """

    if (cohort_first_session is None) != (cohort_last_session is None):
        raise ValueError("decision cohort boundaries must both be present or absent")
    if (
        cohort_first_session is not None
        and cohort_last_session is not None
        and cohort_last_session < cohort_first_session
    ):
        raise ValueError("decision cohort boundaries are reversed")
    hashes = tuple(source_receipt_hashes)
    if len(hashes) != len(set(hashes)):
        raise ValueError("decision source receipt hashes must be unique")
    counts = tuple(sorted((str(key), int(value)) for key, value in support_counts.items()))
    if any(value < 0 for _, value in counts):
        raise ValueError("decision support counts must be nonnegative")
    payload: dict[str, object] = {
        "experiment_id": M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
        "experiment_version": experiment_version,
        "receipt_kind": receipt_kind,
        "boundary_timestamp_utc": _aware_utc(
            boundary_timestamp_utc,
            label="decision boundary timestamp",
        ),
        "decision": decision,
        "cohort_first_session": cohort_first_session,
        "cohort_last_session": cohort_last_session,
        "source_receipt_hashes": hashes,
        "support_counts": counts,
        "protected_outcome_fields_accessed": protected_outcome_fields_accessed,
    }
    payload["receipt_hash_v1"] = _sha256(payload)
    return OpeningReversalDecisionReceiptV1.model_validate(payload)


def build_opening_reversal_confirmation_start_receipt_v1(
    *,
    development_receipt: OpeningReversalDecisionReceiptV1,
    boundary_timestamp_utc: datetime,
) -> OpeningReversalDecisionReceiptV1:
    """Start confirmation only from a supported immutable development receipt."""

    development = OpeningReversalDecisionReceiptV1.model_validate(
        development_receipt.model_dump(mode="python")
    )
    boundary = _aware_utc(
        boundary_timestamp_utc,
        label="confirmation-start boundary timestamp",
    )
    if (
        development.receipt_kind != "development"
        or development.decision != "prospective_opening_reversal_development_supported"
        or development.cohort_last_session is None
        or boundary <= development.boundary_timestamp_utc
    ):
        raise ValueError("confirmation requires a prior supported development receipt")
    return build_opening_reversal_decision_receipt_v1(
        experiment_version=development.experiment_version,
        receipt_kind="confirmation_start",
        boundary_timestamp_utc=boundary,
        decision="untouched_confirmation_started",
        cohort_first_session=development.cohort_last_session,
        cohort_last_session=development.cohort_last_session,
        source_receipt_hashes=(development.receipt_hash_v1,),
        support_counts={},
        protected_outcome_fields_accessed=False,
    )


def build_opening_transfer_decision_receipt_v1(
    *,
    sessions: Sequence[OpeningTransferSessionResultV1],
    boundary_timestamp_utc: datetime,
    experiment_version: Literal["1", "1.1"] = "1",
) -> OpeningReversalDecisionReceiptV1:
    """Freeze the aggregate result of the first 20 valid transfer sessions."""

    ordered = tuple(sorted(sessions, key=lambda item: item.session))
    if len(ordered) != ENGINEERING_TRANSFER_SESSION_COUNT_V1:
        raise ValueError("transfer decision requires exactly 20 valid sessions")
    if len({item.session for item in ordered}) != len(ordered):
        raise ValueError("transfer decision sessions must be unique")
    if not all(item.valid for item in ordered):
        raise ValueError("transfer decision cannot include an invalid session")
    if not all(item.operational_evidence.critical_checks_pass for item in ordered):
        raise ValueError("transfer decision requires all engineering safeguards")
    decisions = {item.decision for item in ordered}
    if decisions == {"opening_transfer_supported_without_recalibration"}:
        decision = "opening_transfer_supported_without_recalibration"
    elif "opening_transfer_not_supported" in decisions:
        decision = "opening_transfer_not_supported"
    elif "opening_transfer_mixed" in decisions:
        decision = "opening_transfer_mixed"
    else:
        decision = "opening_transfer_operational_failure"
    counts = {
        "valid_sessions": len(ordered),
        "supported_without_recalibration": sum(
            item.decision == "opening_transfer_supported_without_recalibration" for item in ordered
        ),
        "mixed_sessions": sum(item.decision == "opening_transfer_mixed" for item in ordered),
        "not_supported_sessions": sum(
            item.decision == "opening_transfer_not_supported" for item in ordered
        ),
        "operational_sessions_passed": sum(
            item.operational_evidence.critical_checks_pass for item in ordered
        ),
    }
    return build_opening_reversal_decision_receipt_v1(
        experiment_version=experiment_version,
        receipt_kind="transfer",
        boundary_timestamp_utc=boundary_timestamp_utc,
        decision=decision,
        cohort_first_session=ordered[0].session,
        cohort_last_session=ordered[-1].session,
        source_receipt_hashes=tuple(item.report_hash_v1 for item in ordered),
        support_counts=counts,
        protected_outcome_fields_accessed=False,
    )


class OptionalOpeningReversalFeedV1(StrEnum):
    NEUTRAL_CONTROL = "neutral_control"
    ADDITIONAL_STRIKE = "additional_strike"
    THREE_TO_FIVE_DTE_COMPARISON = "3_to_5_dte_comparison"
    ZERO_DTE_COMPARISON = "0dte_comparison"
    TICK_BY_TICK = "tick_by_tick"
    ADDITIONAL_UNDERLYING_DIAGNOSTIC = "additional_underlying_diagnostic"

    @property
    def drop_order(self) -> int:
        return FrozenOpeningReversalRuleV1().optional_feed_degradation_order.index(self.value) + 1


class CapacityFeedV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subscription_identifier: str
    subscription_priority: int
    owning_subsystem: str
    may_be_dropped: bool
    drop_order: int | None
    line_cost: int
    status: str


class MarketDataCapacitySnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["market_data_capacity_snapshot_v1"]
    timestamp_utc: datetime
    configured_budget: int
    reserved_lines: int
    mandatory_lines: int
    optional_lines: int
    pending_lines: int
    cancelled_lines: int
    lines_awaiting_acknowledgement_or_cleanup: int
    estimated_free_lines: int
    exact_broker_accounting_known: Literal[False]
    uncertainty: Literal["conservative_local_estimate"]
    active_subscriptions: tuple[CapacityFeedV1, ...]
    current_promoted_episode_id: str | None
    capacity_denial_reasons: tuple[str, ...]
    snapshot_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _snapshot_hash_matches(self) -> MarketDataCapacitySnapshotV1:
        payload = self.model_dump(mode="python", exclude={"snapshot_hash"})
        if self.snapshot_hash != _sha256(payload):
            raise ValueError("capacity snapshot hash mismatch")
        return self


class CapacityDegradationEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: datetime
    episode_id: str
    feed: OptionalOpeningReversalFeedV1 | None
    subscription_ids: tuple[str, ...]
    reason: str
    raw_capacity_reason: str | None = None
    primary_direction_evidence_remains_complete: Literal[True] = True
    primary_option_evidence_remains_complete: bool


def build_capacity_degradation_events_v1(
    record: EpisodeAllocationRecord,
) -> tuple[CapacityDegradationEventV1, ...]:
    """Classify only the subscriptions changed by one allocation transition."""

    if record.state not in {
        EpisodeState.EPISODE_QUEUED,
        EpisodeState.DEGRADED,
    }:
        return ()
    transition = record.transition_subscriptions or (
        record.queued_subscriptions
        if record.state is EpisodeState.EPISODE_QUEUED
        else record.denied_subscriptions
    )
    required = set(record.required_subscriptions)
    primary_affected = tuple(key for key in transition if key in required)
    if primary_affected:
        return (
            CapacityDegradationEventV1(
                timestamp_utc=record.updated_at_utc,
                episode_id=record.episode_id,
                feed=None,
                subscription_ids=primary_affected,
                reason="option_economics_blocked_capacity",
                raw_capacity_reason=record.degradation_reason,
                primary_option_evidence_remains_complete=False,
            ),
        )
    drop_orders = dict(record.optional_drop_orders)
    optional_by_order: dict[int, list[str]] = {}
    for key in transition:
        optional_by_order.setdefault(drop_orders.get(key, 0), []).append(key)
    feeds_by_order = {feed.drop_order: feed for feed in OptionalOpeningReversalFeedV1}
    return tuple(
        CapacityDegradationEventV1(
            timestamp_utc=record.updated_at_utc,
            episode_id=record.episode_id,
            feed=feeds_by_order.get(drop_order),
            subscription_ids=tuple(subscription_ids),
            reason="optional_feed_not_started_capacity_reserved",
            raw_capacity_reason=record.degradation_reason,
            primary_option_evidence_remains_complete=True,
        )
        for drop_order, subscription_ids in sorted(optional_by_order.items())
    )


class OpeningReversalCapacityCoordinatorV1:
    """Project the one live subscription manager into the frozen V1 schema."""

    def __init__(self, *, budget: SubscriptionBudgetManager) -> None:
        if budget.future_trading_reserve_lines < RESERVED_MARKET_DATA_LINES_V1:
            raise ValueError("opening reversal requires a minimum 12-line reserve")
        self.budget = budget

    def snapshot(
        self,
        *,
        observed_at_utc: datetime,
        promoted_episode_id: str | None,
    ) -> MarketDataCapacitySnapshotV1:
        active = tuple(record for record in self.budget.records.values() if record.active)
        feeds = tuple(
            CapacityFeedV1(
                subscription_identifier=record.key,
                subscription_priority=int(record.priority),
                owning_subsystem=(
                    ",".join(sorted(record.owners))
                    if record.owners
                    else "ownership_awaiting_cleanup"
                ),
                may_be_dropped=not record.protected,
                drop_order=(record.drop_order if not record.protected else None),
                line_cost=record.line_cost,
                status=record.status.value,
            )
            for record in sorted(active, key=lambda value: value.key)
        )
        mandatory_lines = sum(
            record.line_cost
            for record in active
            if record.protected or record.subscription_class <= SubscriptionClass.ACTIVE_EPISODE
        )
        optional_lines = sum(
            record.line_cost
            for record in active
            if not record.protected and record.subscription_class > SubscriptionClass.ACTIVE_EPISODE
        )
        pending = sum(record.line_cost for record in active if record.status.value == "pending")
        awaiting = sum(
            record.line_cost for record in active if record.status.value == "cancellation_requested"
        )
        cancelled = sum(
            record.line_cost for record in self.budget.records.values() if not record.active
        )
        payload: dict[str, object] = {
            "schema_version": "market_data_capacity_snapshot_v1",
            "timestamp_utc": _aware_utc(
                observed_at_utc,
                label="capacity snapshot timestamp",
            ),
            "configured_budget": self.budget.total_line_limit,
            "reserved_lines": (
                self.budget.externally_reserved_lines
                + self.budget.preexisting_internal_lines
                + self.budget.future_trading_reserve_lines
                + self.budget.safety_margin_lines
            ),
            "mandatory_lines": mandatory_lines,
            "optional_lines": optional_lines,
            "pending_lines": pending,
            "cancelled_lines": cancelled,
            "lines_awaiting_acknowledgement_or_cleanup": awaiting,
            "estimated_free_lines": max(
                0,
                self.budget.usable_research_lines - sum(record.line_cost for record in active),
            ),
            "exact_broker_accounting_known": False,
            "uncertainty": "conservative_local_estimate",
            "active_subscriptions": feeds,
            "current_promoted_episode_id": promoted_episode_id,
            "capacity_denial_reasons": tuple(
                str(value.get("reason")) for value in self.budget.capacity_denials
            ),
        }
        payload["snapshot_hash"] = _sha256(payload)
        return MarketDataCapacitySnapshotV1.model_validate(payload)


class OptionContractCandidateV1(BaseModel):
    """Contract-definition metadata; construction does not start a quote stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    con_id: int = Field(gt=0)
    underlying: str
    expiry: date
    strike: float = Field(gt=0.0)
    right: Literal["C", "P"]
    multiplier: int = Field(gt=0)
    exchange: str
    trading_class: str


class PrimaryOptionPairSelectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call: OptionContractCandidateV1
    put: OptionContractCandidateV1
    discovery_timestamp_utc: datetime
    contract_source: str
    cache_hit: bool
    candidates_inspected: int
    frozen_tie_break_rule: Literal[
        "1dte_common_nearest_atm_absolute_distance_then_lower_strike_then_con_id"
    ]
    metadata_request_ended: Literal[True]
    full_chain_live_subscription_created: Literal[False]
    live_market_data_lines_consumed: Literal[0]
    planned_live_market_data_lines: Literal[2]
    selection_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _selection_is_linked_and_hashed(self) -> PrimaryOptionPairSelectionV1:
        if (
            self.call.right != "C"
            or self.put.right != "P"
            or self.call.underlying != self.put.underlying
            or self.call.expiry != self.put.expiry
            or self.call.strike != self.put.strike
        ):
            raise ValueError("primary option pair selection is inconsistent")
        payload = self.model_dump(mode="python", exclude={"selection_hash"})
        if self.selection_hash != _sha256(payload):
            raise ValueError("primary option pair selection hash mismatch")
        return self


def select_primary_option_pair_v1(
    *,
    session: date,
    underlying_reference: float,
    candidates: Iterable[OptionContractCandidateV1],
    discovery_timestamp_utc: datetime,
    contract_source: str,
    cache_hit: bool,
    candidates_inspected: int | None = None,
) -> PrimaryOptionPairSelectionV1:
    """Choose one same-strike 1DTE call/put pair from metadata only."""

    if not math.isfinite(underlying_reference) or underlying_reference <= 0.0:
        raise ValueError("underlying_reference_invalid")
    rows = tuple(
        OptionContractCandidateV1.model_validate(value.model_dump(mode="python"))
        for value in candidates
    )
    inspected = len(rows) if candidates_inspected is None else candidates_inspected
    if inspected < len(rows):
        raise ValueError("candidates_inspected cannot be below supplied candidates")
    expiry = session + timedelta(days=1)
    calls: dict[float, OptionContractCandidateV1] = {}
    puts: dict[float, OptionContractCandidateV1] = {}
    for row in sorted(rows, key=lambda value: value.con_id):
        if row.expiry != expiry:
            continue
        target = calls if row.right == "C" else puts
        target.setdefault(row.strike, row)
    common = tuple(sorted(set(calls).intersection(puts)))
    if not common:
        raise ValueError("primary_1dte_option_pair_unavailable")
    strike = min(
        common,
        key=lambda value: (abs(value - underlying_reference), value),
    )
    call = calls[strike]
    put = puts[strike]
    if call.underlying != put.underlying:
        raise ValueError("primary_option_pair_underlying_mismatch")
    payload: dict[str, object] = {
        "call": call,
        "put": put,
        "discovery_timestamp_utc": _aware_utc(
            discovery_timestamp_utc,
            label="contract discovery timestamp",
        ),
        "contract_source": contract_source,
        "cache_hit": cache_hit,
        "candidates_inspected": inspected,
        "frozen_tie_break_rule": (
            "1dte_common_nearest_atm_absolute_distance_then_lower_strike_then_con_id"
        ),
        "metadata_request_ended": True,
        "full_chain_live_subscription_created": False,
        # Contract discovery is metadata-only.  Quote lines are consumed only
        # after a separate capacity approval and subscription start.
        "live_market_data_lines_consumed": 0,
        "planned_live_market_data_lines": 2,
    }
    payload["selection_hash"] = _sha256(payload)
    return PrimaryOptionPairSelectionV1.model_validate(payload)


class OptionTopOfBookV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: datetime
    bid: float | None
    ask: float | None
    quote_age_seconds: float | None
    locked_or_crossed: bool
    stale: bool
    missing_reason: str | None

    @field_validator("timestamp_utc")
    @classmethod
    def _quote_timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="option quote timestamp")


class PrimaryOptionBidAskOutcomeV1(BaseModel):
    """Conservative ask-entry/bid-exit evidence for one selected contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_receipt_hash_v1: str
    contract: OptionContractCandidateV1
    role: Literal["predicted_leg", "opposite_leg"]
    entry_timestamp_utc: datetime
    subscription_start_utc: datetime
    subscription_end_utc: datetime
    capacity_line_owner: str
    entry_quote: OptionTopOfBookV1
    exit_quote: OptionTopOfBookV1
    entry_midpoint_diagnostic: float | None
    exit_midpoint_diagnostic: float | None
    entry_spread: float | None
    exit_spread: float | None
    conservative_return_v1: float | None
    complete: bool
    missing_reason: str | None
    outcome_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator(
        "entry_timestamp_utc",
        "subscription_start_utc",
        "subscription_end_utc",
    )
    @classmethod
    def _outcome_timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="primary option outcome timestamp")

    @model_validator(mode="after")
    def _complete_quote_contract(self) -> PrimaryOptionBidAskOutcomeV1:
        if self.subscription_end_utc < self.subscription_start_utc:
            raise ValueError("option subscription interval is reversed")
        if self.complete:
            horizon = self.entry_timestamp_utc + timedelta(minutes=15)
            for label, quote in (
                ("entry", self.entry_quote),
                ("exit", self.exit_quote),
            ):
                if (
                    quote.bid is None
                    or quote.ask is None
                    or not math.isfinite(quote.bid)
                    or not math.isfinite(quote.ask)
                    or quote.bid < 0.0
                    or quote.ask <= quote.bid
                    or quote.locked_or_crossed
                    or quote.stale
                    or quote.missing_reason is not None
                ):
                    raise ValueError(f"complete option {label} quote quality is invalid")
            if not (
                self.subscription_start_utc
                <= self.entry_quote.timestamp_utc
                <= self.subscription_end_utc
                and self.subscription_start_utc
                <= self.exit_quote.timestamp_utc
                <= self.subscription_end_utc
                and self.entry_quote.timestamp_utc >= self.entry_timestamp_utc
                and self.exit_quote.timestamp_utc >= horizon
            ):
                raise ValueError("complete option quote chronology is invalid")
            expected_entry_age = (
                self.entry_quote.timestamp_utc - self.entry_timestamp_utc
            ).total_seconds()
            expected_exit_age = (self.exit_quote.timestamp_utc - horizon).total_seconds()
            if (
                self.entry_quote.quote_age_seconds is None
                or self.exit_quote.quote_age_seconds is None
                or not math.isclose(
                    self.entry_quote.quote_age_seconds,
                    expected_entry_age,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    self.exit_quote.quote_age_seconds,
                    expected_exit_age,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("complete option quote age is inconsistent")
            if (
                self.entry_spread is None
                or self.exit_spread is None
                or self.conservative_return_v1 is None
                or self.missing_reason is not None
            ):
                raise ValueError("complete option outcome fields are missing")
            entry_bid = cast(float, self.entry_quote.bid)
            entry_ask = cast(float, self.entry_quote.ask)
            exit_bid = cast(float, self.exit_quote.bid)
            exit_ask = cast(float, self.exit_quote.ask)
            expected_derived_values = (
                (
                    "entry midpoint",
                    self.entry_midpoint_diagnostic,
                    (entry_bid + entry_ask) / 2.0,
                ),
                (
                    "exit midpoint",
                    self.exit_midpoint_diagnostic,
                    (exit_bid + exit_ask) / 2.0,
                ),
                ("entry spread", self.entry_spread, entry_ask - entry_bid),
                ("exit spread", self.exit_spread, exit_ask - exit_bid),
                (
                    "conservative return",
                    self.conservative_return_v1,
                    (exit_bid - entry_ask) / entry_ask,
                ),
            )
            for label, observed, expected in expected_derived_values:
                if (
                    observed is None
                    or not math.isfinite(observed)
                    or not math.isclose(
                        observed,
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(f"complete option {label} is inconsistent")
        elif self.conservative_return_v1 is not None or not self.missing_reason:
            raise ValueError("incomplete option outcome must fail closed")
        payload = self.model_dump(mode="python", exclude={"outcome_hash_v1"})
        if self.outcome_hash_v1 != _sha256(payload):
            raise ValueError("primary option outcome hash mismatch")
        return self


def build_primary_option_bid_ask_outcome_v1(
    *,
    prediction_receipt_hash_v1: str,
    contract: OptionContractCandidateV1,
    role: Literal["predicted_leg", "opposite_leg"],
    entry_timestamp_utc: datetime,
    subscription_start_utc: datetime,
    subscription_end_utc: datetime,
    capacity_line_owner: str,
    entry_quote: OptionTopOfBookV1,
    exit_quote: OptionTopOfBookV1,
) -> PrimaryOptionBidAskOutcomeV1:
    """Use only the first valid ask at entry and first valid bid at +15m."""

    entry_timestamp = _aware_utc(
        entry_timestamp_utc,
        label="frozen option entry timestamp",
    )
    subscription_start = _aware_utc(
        subscription_start_utc,
        label="subscription start",
    )
    subscription_end = _aware_utc(
        subscription_end_utc,
        label="subscription end",
    )
    horizon = entry_timestamp + timedelta(minutes=15)
    entry_prices_valid = (
        entry_quote.bid is not None
        and entry_quote.ask is not None
        and math.isfinite(entry_quote.bid)
        and math.isfinite(entry_quote.ask)
        and entry_quote.bid >= 0.0
        and entry_quote.ask > entry_quote.bid
    )
    entry_age_valid = (
        entry_quote.quote_age_seconds is not None
        and math.isfinite(entry_quote.quote_age_seconds)
        and entry_quote.quote_age_seconds >= 0.0
        and math.isclose(
            entry_quote.quote_age_seconds,
            (entry_quote.timestamp_utc - entry_timestamp).total_seconds(),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    entry_chronology_valid = (
        subscription_start <= entry_quote.timestamp_utc <= subscription_end
        and entry_quote.timestamp_utc >= entry_timestamp
    )
    entry_valid = (
        entry_prices_valid
        and entry_age_valid
        and entry_chronology_valid
        and not entry_quote.locked_or_crossed
        and not entry_quote.stale
        and entry_quote.missing_reason is None
    )
    exit_prices_valid = (
        exit_quote.bid is not None
        and exit_quote.ask is not None
        and math.isfinite(exit_quote.bid)
        and math.isfinite(exit_quote.ask)
        and exit_quote.bid >= 0.0
        and exit_quote.ask > exit_quote.bid
    )
    exit_age_valid = (
        exit_quote.quote_age_seconds is not None
        and math.isfinite(exit_quote.quote_age_seconds)
        and exit_quote.quote_age_seconds >= 0.0
        and math.isclose(
            exit_quote.quote_age_seconds,
            (exit_quote.timestamp_utc - horizon).total_seconds(),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    exit_chronology_valid = (
        subscription_start <= exit_quote.timestamp_utc <= subscription_end
        and exit_quote.timestamp_utc >= horizon
    )
    exit_valid = (
        exit_prices_valid
        and exit_age_valid
        and exit_chronology_valid
        and not exit_quote.locked_or_crossed
        and not exit_quote.stale
        and exit_quote.missing_reason is None
    )
    complete = entry_valid and exit_valid
    if complete:
        missing = None
    elif not entry_valid:
        if entry_quote.missing_reason is not None:
            missing = entry_quote.missing_reason
        elif entry_quote.locked_or_crossed:
            missing = "entry_quote_locked_or_crossed"
        elif entry_quote.stale:
            missing = "entry_quote_stale"
        elif not entry_prices_valid:
            missing = "entry_bid_ask_invalid"
        elif not entry_age_valid:
            missing = "entry_quote_age_invalid"
        elif not entry_chronology_valid:
            missing = "entry_quote_chronology_invalid"
        else:
            missing = "entry_quote_invalid"
    else:
        if exit_quote.missing_reason is not None:
            missing = exit_quote.missing_reason
        elif exit_quote.locked_or_crossed:
            missing = "exit_quote_locked_or_crossed"
        elif exit_quote.stale:
            missing = "exit_quote_stale"
        elif not exit_prices_valid:
            missing = "exit_bid_ask_invalid"
        elif not exit_age_valid:
            missing = "exit_quote_age_invalid"
        elif not exit_chronology_valid:
            missing = "exit_quote_chronology_invalid"
        else:
            missing = "exit_quote_invalid"
    entry_mid = (
        None
        if entry_quote.bid is None or entry_quote.ask is None
        else (entry_quote.bid + entry_quote.ask) / 2.0
    )
    exit_mid = (
        None
        if exit_quote.bid is None or exit_quote.ask is None
        else (exit_quote.bid + exit_quote.ask) / 2.0
    )
    entry_spread = (
        None
        if entry_quote.bid is None or entry_quote.ask is None
        else entry_quote.ask - entry_quote.bid
    )
    exit_spread = (
        None
        if exit_quote.bid is None or exit_quote.ask is None
        else exit_quote.ask - exit_quote.bid
    )
    conservative_return = (
        None
        if not complete
        else (cast(float, exit_quote.bid) - cast(float, entry_quote.ask))
        / cast(float, entry_quote.ask)
    )
    payload: dict[str, object] = {
        "prediction_receipt_hash_v1": prediction_receipt_hash_v1,
        "contract": contract,
        "role": role,
        "entry_timestamp_utc": entry_timestamp,
        "subscription_start_utc": subscription_start,
        "subscription_end_utc": subscription_end,
        "capacity_line_owner": capacity_line_owner,
        "entry_quote": entry_quote,
        "exit_quote": exit_quote,
        "entry_midpoint_diagnostic": entry_mid,
        "exit_midpoint_diagnostic": exit_mid,
        "entry_spread": entry_spread,
        "exit_spread": exit_spread,
        "conservative_return_v1": conservative_return,
        "complete": complete,
        "missing_reason": missing,
    }
    payload["outcome_hash_v1"] = _sha256(payload)
    return PrimaryOptionBidAskOutcomeV1.model_validate(payload)


__all__ = [
    "DESCRIPTIVE_OVERNIGHT_GAP_Q10_V1",
    "DESCRIPTIVE_OVERNIGHT_GAP_Q90_V1",
    "DESCRIPTIVE_TOTAL_TRANSITION_Q10_V1",
    "DESCRIPTIVE_TOTAL_TRANSITION_Q90_V1",
    "ENGINEERING_TRANSFER_SESSION_COUNT_V1",
    "FrozenOpeningReversalRuleV1",
    "FrozenOpeningReversalExperimentConfigV1",
    "M1C_HIGH_TAIL_THRESHOLD_V1",
    "M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID",
    "M1C_PROSPECTIVE_OPENING_REVERSAL_V1_VERSION",
    "MarketDataCapacitySnapshotV1",
    "NEGATIVE_OPENING_RETURN_THRESHOLD_V1",
    "OPENING_RANGE_THRESHOLD_V1",
    "OpeningEventAccountingReconciliationV1",
    "OpeningEventAccountingRowV1",
    "OpeningReversalActivationReceiptV1",
    "OpeningReversalCapacityCoordinatorV1",
    "OpeningReversalDecisionReceiptV1",
    "OpeningReversalPredictionInputV1",
    "OpeningReversalPredictionReceiptV1",
    "OpeningReversalPredictionTimingEvidenceV1_1",
    "OpeningReversalUnderlyingOutcomeV1",
    "OpeningTransferBarV1",
    "OpeningTransferOperationalEvidenceV1",
    "OpeningTransferSessionResultV1",
    "OptionContractCandidateV1",
    "OptionTopOfBookV1",
    "OptionalOpeningReversalFeedV1",
    "POSITIVE_OPENING_RETURN_THRESHOLD_V1",
    "PostEntryBarV1",
    "PrimaryOptionBidAskOutcomeV1",
    "PrimaryOptionPairSelectionV1",
    "PromotionSelectionV1",
    "RESERVED_MARKET_DATA_LINES_V1",
    "build_opening_reversal_outcome_v1",
    "build_opening_reversal_confirmation_start_receipt_v1",
    "build_opening_reversal_decision_receipt_v1",
    "build_opening_transfer_decision_receipt_v1",
    "build_incomplete_opening_reversal_outcome_v1",
    "build_activation_receipt_v1",
    "build_capacity_degradation_events_v1",
    "build_frozen_experiment_config_v1",
    "build_prediction_receipt_v1",
    "build_primary_option_bid_ask_outcome_v1",
    "evaluate_opening_transfer_session_v1",
    "partition_material_outcome_v1",
    "load_activation_receipt_v1",
    "load_frozen_experiment_config_v1",
    "missing_opening_transfer_operational_evidence_v1",
    "reconcile_opening_event_accounting_v1",
    "select_primary_option_pair_v1",
    "select_promoted_prediction_v1",
]
