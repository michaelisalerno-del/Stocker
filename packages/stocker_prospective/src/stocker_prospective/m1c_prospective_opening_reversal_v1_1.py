"""Frozen V1.1 timing addendum for M1C opening-reversal shadow research.

V1.1 changes only the operational receipt-time contract.  The V1 scientific
rule, nominal 10:00 entry, outcome horizon, capacity policy, and no-order
boundary remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
    OpeningReversalActivationReceiptV1,
    OpeningReversalPredictionReceiptV1,
    OpeningReversalPredictionTimingEvidenceV1_1,
)

M1C_PROSPECTIVE_OPENING_REVERSAL_V1_1_VERSION = "1.1"


def _canonical_value(value: object) -> object:
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
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass
class OpeningReversalDecisionDataGateV1_1:
    """Buffer decision-surface entry data while raw archival remains active."""

    protected_symbols: frozenset[str]
    _released_audit_hashes: dict[date, str] = field(default_factory=dict)
    _compromised_reasons: dict[date, str] = field(default_factory=dict)
    _deferred_counts: dict[date, int] = field(default_factory=dict)
    _first_deferred_received_at: dict[date, datetime] = field(default_factory=dict)
    _seen_event_ids: dict[date, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = frozenset(symbol.strip().upper() for symbol in self.protected_symbols)
        if not canonical or "" in canonical:
            raise ValueError("V1.1 decision gate requires protected symbols")
        self.protected_symbols = canonical

    def observe(
        self,
        *,
        session: date,
        symbol: str,
        nominal_entry_timestamp_utc: datetime,
        event_ordering_timestamp_utc: datetime,
        event_received_timestamp_utc: datetime,
        event_id: str | None = None,
    ) -> Literal["admit", "buffer"]:
        nominal_entry = _aware_utc(
            nominal_entry_timestamp_utc,
            label="V1.1 gate nominal entry",
        )
        ordering = _aware_utc(
            event_ordering_timestamp_utc,
            label="V1.1 gate event ordering timestamp",
        )
        received = _aware_utc(
            event_received_timestamp_utc,
            label="V1.1 gate event received timestamp",
        )
        if (
            symbol.strip().upper() not in self.protected_symbols
            or ordering < nominal_entry
            or session in self._released_audit_hashes
            or session in self._compromised_reasons
        ):
            return "admit"
        if event_id is not None:
            seen = self._seen_event_ids.setdefault(session, set())
            if event_id in seen:
                return "buffer"
            seen.add(event_id)
        self._deferred_counts[session] = self._deferred_counts.get(session, 0) + 1
        first = self._first_deferred_received_at.get(session)
        if first is None or received < first:
            self._first_deferred_received_at[session] = received
        return "buffer"

    def authorize_release_after_durable_audit(
        self,
        *,
        session: date,
        audit_hash_v1_1: str,
    ) -> None:
        if len(audit_hash_v1_1) != 64 or any(
            character not in "0123456789abcdef" for character in audit_hash_v1_1
        ):
            raise ValueError("V1.1 gate requires a durable audit hash")
        if session in self._compromised_reasons:
            raise ValueError("V1.1 compromised gate cannot pass")
        existing = self._released_audit_hashes.get(session)
        if existing is not None and existing != audit_hash_v1_1:
            raise ValueError("V1.1 gate release is immutable")
        self._released_audit_hashes[session] = audit_hash_v1_1

    def fail_closed_for_science_and_continue_core(
        self,
        *,
        session: date,
        reason: str,
    ) -> None:
        canonical = reason.strip()
        if not canonical:
            raise ValueError("V1.1 gate failure reason is required")
        if session in self._released_audit_hashes:
            raise ValueError("V1.1 released gate cannot later fail")
        existing = self._compromised_reasons.get(session)
        if existing is not None and existing != canonical:
            raise ValueError("V1.1 gate failure is immutable")
        self._compromised_reasons[session] = canonical

    def deferred_event_count(self, session: date) -> int:
        return self._deferred_counts.get(session, 0)

    def first_deferred_event_received_at(
        self,
        session: date,
    ) -> datetime | None:
        return self._first_deferred_received_at.get(session)

    def scientific_barrier_compromised(self, session: date) -> bool:
        return session in self._compromised_reasons

    def released(self, session: date) -> bool:
        return session in self._released_audit_hashes or session in self._compromised_reasons


class FrozenOpeningReversalTimingAddendumConfigV1_1(BaseModel):
    """The only operational change permitted by the V1.1 amendment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-prospective-opening-reversal-timing-addendum-config-v1.1"]
    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1.1"]
    superseded_experiment_version: Literal["1"]
    superseded_activation_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_rule_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_configuration_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    nominal_signal_timestamp: Literal["10:00 America/New_York"]
    nominal_entry_timestamp: Literal["10:00 America/New_York"]
    primary_horizon_minutes: Literal[15]
    receipt_contract: Literal[
        "durable_before_entry_or_post_entry_data_admitted_to_decision_surface"
    ]
    predictor_window_must_be_complete: Literal[True]
    raw_append_only_archival_before_receipt_allowed: Literal[True]
    nominal_entry_actionable: Literal[False]
    research_shadow_only: Literal[True]
    engineering_transfer_sessions_restart: Literal[20]
    scientific_rule_changed: Literal[False]
    capacity_policy_changed: Literal[False]
    order_routing_enabled: Literal[False]
    configuration_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _hash_is_self_consistent(
        self,
    ) -> FrozenOpeningReversalTimingAddendumConfigV1_1:
        payload = self.model_dump(
            mode="python",
            exclude={"configuration_hash_v1_1"},
        )
        if self.configuration_hash_v1_1 != _sha256(payload):
            raise ValueError("V1.1 timing addendum configuration hash mismatch")
        return self


def build_frozen_timing_addendum_config_v1_1(
    *,
    superseded_activation_receipt_hash_v1: str,
    frozen_rule_hash_v1: str,
    frozen_configuration_hash_v1: str,
) -> FrozenOpeningReversalTimingAddendumConfigV1_1:
    payload: dict[str, object] = {
        "schema_version": ("m1c-prospective-opening-reversal-timing-addendum-config-v1.1"),
        "experiment_id": M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
        "experiment_version": "1.1",
        "superseded_experiment_version": "1",
        "superseded_activation_receipt_hash_v1": (superseded_activation_receipt_hash_v1),
        "frozen_rule_hash_v1": frozen_rule_hash_v1,
        "frozen_configuration_hash_v1": frozen_configuration_hash_v1,
        "nominal_signal_timestamp": "10:00 America/New_York",
        "nominal_entry_timestamp": "10:00 America/New_York",
        "primary_horizon_minutes": 15,
        "receipt_contract": (
            "durable_before_entry_or_post_entry_data_admitted_to_decision_surface"
        ),
        "predictor_window_must_be_complete": True,
        "raw_append_only_archival_before_receipt_allowed": True,
        "nominal_entry_actionable": False,
        "research_shadow_only": True,
        "engineering_transfer_sessions_restart": 20,
        "scientific_rule_changed": False,
        "capacity_policy_changed": False,
        "order_routing_enabled": False,
    }
    payload["configuration_hash_v1_1"] = _sha256(payload)
    return FrozenOpeningReversalTimingAddendumConfigV1_1.model_validate(payload)


class OpeningReversalActivationReceiptV1_1(BaseModel):
    """Immutable superseding boundary created before any V1.1 outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1.1"]
    activation_timestamp_utc: datetime
    new_york_trading_date_at_activation: date
    branch: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    dirty_working_tree_status: str
    timing_addendum_configuration_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")
    superseded_activation_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_configuration_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    m1c_version: str
    tail_phase_version: str
    a1_version: str
    recorder_schema_version: Literal["0015_m1c_prospective_opening_reversal_v1_1"]
    configured_reserved_line_count: Literal[12]
    scientific_rule_changed: Literal[False]
    nominal_entry_changed: Literal[False]
    primary_horizon_changed: Literal[False]
    capacity_policy_changed: Literal[False]
    engineering_transfer_sessions_restart: Literal[20]
    nominal_entry_actionable: Literal[False]
    research_shadow_only: Literal[True]
    protected_pre_activation_outcomes_opened: Literal[False]
    order_routing_disabled: Literal[True]
    order_methods_available: Literal[False]
    activation_receipt_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("activation_timestamp_utc")
    @classmethod
    def _activation_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="V1.1 activation timestamp")

    @model_validator(mode="after")
    def _hash_is_self_consistent(
        self,
    ) -> OpeningReversalActivationReceiptV1_1:
        payload = self.model_dump(
            mode="python",
            exclude={"activation_receipt_hash_v1_1"},
        )
        if self.activation_receipt_hash_v1_1 != _sha256(payload):
            raise ValueError("V1.1 activation receipt hash mismatch")
        return self


def build_activation_receipt_v1_1(
    *,
    activation_timestamp_utc: datetime,
    new_york_trading_date_at_activation: date,
    branch: str,
    commit: str,
    dirty_working_tree_status: str,
    timing_addendum_config: FrozenOpeningReversalTimingAddendumConfigV1_1,
    superseded_activation_receipt: OpeningReversalActivationReceiptV1,
    m1c_version: str,
    tail_phase_version: str,
    a1_version: str,
) -> OpeningReversalActivationReceiptV1_1:
    activation = _aware_utc(
        activation_timestamp_utc,
        label="V1.1 activation timestamp",
    )
    if activation <= superseded_activation_receipt.activation_timestamp_utc:
        raise ValueError("V1.1 activation must follow the V1 boundary")
    if (
        timing_addendum_config.superseded_activation_receipt_hash_v1
        != superseded_activation_receipt.activation_receipt_hash
        or timing_addendum_config.frozen_rule_hash_v1
        != superseded_activation_receipt.frozen_rule_hash
        or timing_addendum_config.frozen_configuration_hash_v1
        != superseded_activation_receipt.configuration_hash
    ):
        raise ValueError("V1.1 addendum does not bind the exact V1 activation")
    payload: dict[str, object] = {
        "experiment_id": M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
        "experiment_version": "1.1",
        "activation_timestamp_utc": activation,
        "new_york_trading_date_at_activation": (new_york_trading_date_at_activation),
        "branch": branch,
        "commit": commit,
        "dirty_working_tree_status": dirty_working_tree_status,
        "timing_addendum_configuration_hash_v1_1": (timing_addendum_config.configuration_hash_v1_1),
        "superseded_activation_receipt_hash_v1": (
            superseded_activation_receipt.activation_receipt_hash
        ),
        "frozen_configuration_hash_v1": (superseded_activation_receipt.configuration_hash),
        "frozen_rule_hash": superseded_activation_receipt.frozen_rule_hash,
        "m1c_version": m1c_version,
        "tail_phase_version": tail_phase_version,
        "a1_version": a1_version,
        "recorder_schema_version": ("0015_m1c_prospective_opening_reversal_v1_1"),
        "configured_reserved_line_count": 12,
        "scientific_rule_changed": False,
        "nominal_entry_changed": False,
        "primary_horizon_changed": False,
        "capacity_policy_changed": False,
        "engineering_transfer_sessions_restart": 20,
        "nominal_entry_actionable": False,
        "research_shadow_only": True,
        "protected_pre_activation_outcomes_opened": False,
        "order_routing_disabled": True,
        "order_methods_available": False,
    }
    payload["activation_receipt_hash_v1_1"] = _sha256(payload)
    return OpeningReversalActivationReceiptV1_1.model_validate(payload)


def load_frozen_timing_addendum_config_v1_1(
    path: str,
) -> FrozenOpeningReversalTimingAddendumConfigV1_1:
    return FrozenOpeningReversalTimingAddendumConfigV1_1.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_activation_receipt_v1_1(
    path: str,
) -> OpeningReversalActivationReceiptV1_1:
    return OpeningReversalActivationReceiptV1_1.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


class OpeningReversalCausalBarrierAuditV1_1(BaseModel):
    """Immutable proof that all 20 receipts preceded decision-data release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1.1"]
    activation_receipt_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")
    session: date
    nominal_entry_timestamp_utc: datetime
    prediction_receipt_count: int = Field(ge=0, le=20)
    prediction_receipt_hashes: tuple[str, ...]
    deferred_event_count: int = Field(ge=0)
    first_deferred_event_received_at_utc: datetime | None
    entry_or_post_entry_data_admitted_before_receipts: bool
    raw_event_archive_write_allowed: Literal[True]
    core_recorder_continued: Literal[True]
    barrier_status: Literal["passed", "failed_closed"]
    failure_reason: str | None
    release_authorized_at_utc: datetime
    audit_hash_v1_1: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator(
        "nominal_entry_timestamp_utc",
        "first_deferred_event_received_at_utc",
        "release_authorized_at_utc",
    )
    @classmethod
    def _audit_timestamp_is_aware(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        return _aware_utc(value, label="V1.1 causal barrier timestamp")

    @model_validator(mode="after")
    def _audit_is_self_consistent(
        self,
    ) -> OpeningReversalCausalBarrierAuditV1_1:
        if (
            self.prediction_receipt_hashes != tuple(sorted(self.prediction_receipt_hashes))
            or len(self.prediction_receipt_hashes) != len(set(self.prediction_receipt_hashes))
            or self.prediction_receipt_count != len(self.prediction_receipt_hashes)
            or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in self.prediction_receipt_hashes
            )
        ):
            raise ValueError("V1.1 barrier prediction receipt set is invalid")
        if (self.deferred_event_count == 0) != (self.first_deferred_event_received_at_utc is None):
            raise ValueError("V1.1 barrier deferred-event evidence differs")
        if self.release_authorized_at_utc < self.nominal_entry_timestamp_utc:
            raise ValueError("V1.1 barrier released before nominal entry")
        if self.barrier_status == "passed":
            if (
                self.prediction_receipt_count != 20
                or self.entry_or_post_entry_data_admitted_before_receipts
                or self.failure_reason is not None
            ):
                raise ValueError("V1.1 passing barrier lacks complete proof")
        elif not self.failure_reason:
            raise ValueError("V1.1 failed barrier requires a reason")
        payload = self.model_dump(mode="python", exclude={"audit_hash_v1_1"})
        if self.audit_hash_v1_1 != _sha256(payload):
            raise ValueError("V1.1 causal barrier audit hash mismatch")
        return self


def build_causal_barrier_audit_v1_1(
    *,
    activation_receipt_hash_v1_1: str,
    session: date,
    nominal_entry_timestamp_utc: datetime,
    prediction_receipts: Sequence[OpeningReversalPredictionReceiptV1],
    deferred_event_received_timestamps: Sequence[datetime],
    entry_or_post_entry_data_admitted_before_receipts: bool,
    release_authorized_at_utc: datetime,
    operational_failure_reason: str | None = None,
) -> OpeningReversalCausalBarrierAuditV1_1:
    receipts = tuple(
        OpeningReversalPredictionReceiptV1.model_validate(receipt.model_dump(mode="python"))
        for receipt in prediction_receipts
    )
    nominal_entry = _aware_utc(
        nominal_entry_timestamp_utc,
        label="V1.1 nominal entry timestamp",
    )
    release = _aware_utc(
        release_authorized_at_utc,
        label="V1.1 release authorization timestamp",
    )
    deferred = tuple(
        sorted(
            _aware_utc(value, label="V1.1 deferred event timestamp")
            for value in deferred_event_received_timestamps
        )
    )
    if any(
        receipt.experiment_version != "1.1"
        or receipt.session != session
        or receipt.entry_timestamp_utc != nominal_entry
        or receipt.timing_evidence_v1_1 is None
        or (
            receipt.timing_evidence_v1_1.timing_addendum_activation_receipt_hash_v1_1
            != activation_receipt_hash_v1_1
        )
        for receipt in receipts
    ):
        raise ValueError("V1.1 barrier receipts differ from one frozen session")
    if receipts and release < max(receipt.receipt_created_at_utc for receipt in receipts):
        raise ValueError("V1.1 barrier release precedes durable receipts")
    hashes = tuple(sorted(receipt.receipt_hash_v1 for receipt in receipts))
    explicit_failure = (
        None if operational_failure_reason is None else operational_failure_reason.strip()
    )
    if operational_failure_reason is not None and not explicit_failure:
        raise ValueError("V1.1 barrier operational failure reason is empty")
    failure_reason = explicit_failure or (
        "entry_or_post_entry_data_admitted_before_receipts"
        if entry_or_post_entry_data_admitted_before_receipts
        else "prediction_receipt_set_incomplete_before_release"
        if len(receipts) != 20
        else None
    )
    payload: dict[str, object] = {
        "experiment_id": M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
        "experiment_version": "1.1",
        "activation_receipt_hash_v1_1": activation_receipt_hash_v1_1,
        "session": session,
        "nominal_entry_timestamp_utc": nominal_entry,
        "prediction_receipt_count": len(receipts),
        "prediction_receipt_hashes": hashes,
        "deferred_event_count": len(deferred),
        "first_deferred_event_received_at_utc": (None if not deferred else deferred[0]),
        "entry_or_post_entry_data_admitted_before_receipts": (
            entry_or_post_entry_data_admitted_before_receipts
        ),
        "raw_event_archive_write_allowed": True,
        "core_recorder_continued": True,
        "barrier_status": "passed" if failure_reason is None else "failed_closed",
        "failure_reason": failure_reason,
        "release_authorized_at_utc": release,
    }
    payload["audit_hash_v1_1"] = _sha256(payload)
    return OpeningReversalCausalBarrierAuditV1_1.model_validate(payload)


__all__ = [
    "FrozenOpeningReversalTimingAddendumConfigV1_1",
    "M1C_PROSPECTIVE_OPENING_REVERSAL_V1_1_VERSION",
    "OpeningReversalActivationReceiptV1_1",
    "OpeningReversalCausalBarrierAuditV1_1",
    "OpeningReversalDecisionDataGateV1_1",
    "OpeningReversalPredictionTimingEvidenceV1_1",
    "build_activation_receipt_v1_1",
    "build_causal_barrier_audit_v1_1",
    "build_frozen_timing_addendum_config_v1_1",
    "load_activation_receipt_v1_1",
    "load_frozen_timing_addendum_config_v1_1",
]
