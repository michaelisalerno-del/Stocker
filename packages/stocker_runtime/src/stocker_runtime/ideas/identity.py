"""Canonical deterministic identity for every generic idea output writer."""

from __future__ import annotations

import hashlib
from typing import cast

from stocker_runtime.domain import JsonValue, ProposedTradeLeg, canonical_json_bytes


def deterministic_idea_output_id(
    *,
    instance_id: str,
    input_event_ids: tuple[str, ...],
    output_kind: str,
    output_ordinal: int,
    as_of_at_us: int,
    payload: JsonValue,
    legs: tuple[ProposedTradeLeg, ...] = (),
) -> str:
    """Bind one output to its ordered evidence, meaning, payload, and typed legs."""

    if not instance_id or not input_event_ids or len(set(input_event_ids)) != len(input_event_ids):
        raise ValueError("output identity requires an instance and unique ordered evidence")
    material = cast(
        JsonValue,
        {
            "instance_id": instance_id,
            "input_event_ids": input_event_ids,
            "output_kind": output_kind,
            "output_ordinal": output_ordinal,
            "as_of_at_us": as_of_at_us,
            "payload": payload,
            "legs": tuple(leg.model_dump(mode="json") for leg in legs),
        },
    )
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def deterministic_idea_output_content_hash(
    *,
    run_id: str,
    instance_id: str,
    output_kind: str,
    subject_instrument_id: str,
    emitted_at_us: int,
    as_of_at_us: int,
    valid_until_at_us: int | None,
    direction: str | None,
    strength: float | None,
    confidence: float | None,
    horizon_us: int | None,
    input_event_ids: tuple[str, ...],
    output_ordinal: int,
    payload: JsonValue,
    data_class: str,
    authority_status: str,
    legs: tuple[ProposedTradeLeg, ...] = (),
) -> str:
    """Hash every immutable persisted output field through one canonical contract."""

    if (
        not run_id
        or not instance_id
        or not subject_instrument_id
        or not input_event_ids
        or len(set(input_event_ids)) != len(input_event_ids)
    ):
        raise ValueError("output content requires run, instance, subject, and ordered evidence")
    content = cast(
        JsonValue,
        {
            "run_id": run_id,
            "instance_id": instance_id,
            "output_kind": output_kind,
            "subject_instrument_id": subject_instrument_id,
            "emitted_at_us": emitted_at_us,
            "as_of_at_us": as_of_at_us,
            "valid_until_at_us": valid_until_at_us,
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "horizon_us": horizon_us,
            "input_event_ids": input_event_ids,
            "output_ordinal": output_ordinal,
            "payload": payload,
            "data_class": data_class,
            "authority_status": authority_status,
            "legs": tuple(leg.model_dump(mode="json") for leg in legs),
        },
    )
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()
