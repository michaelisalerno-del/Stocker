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
