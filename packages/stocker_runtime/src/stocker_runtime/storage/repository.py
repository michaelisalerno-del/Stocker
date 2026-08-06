"""Small authoritative repository seams for bounded Stocker V2 evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes, ensure_authority_free_json
from stocker_runtime.storage.connection import connect_v2

MAX_EXTENSION_JSON_BYTES = 64 * 1024
MAX_IDEA_OUTPUT_JSON_BYTES = 16 * 1024
MAX_CALLBACK_PAYLOAD_BYTES = 64 * 1024
MAX_NONTERMINAL_CALLBACK_ROWS = 50_000


class JsonAdmissionError(ValueError):
    """Canonical JSON is invalid or exceeds its durable admission bound."""


class IdentityCollisionError(RuntimeError):
    """A deterministic identity already names different durable content."""


@dataclass(frozen=True)
class IdeaOutputRecord:
    """A generic idea output ready for durable, authority-free persistence."""

    run_id: str
    instance_id: str
    output_kind: Literal["observation", "signal", "proposed_position", "proposed_trade"]
    subject_instrument_id: str
    emitted_at_us: int
    as_of_at_us: int
    valid_until_at_us: int | None
    direction: str | None
    strength: float | None
    confidence: float | None
    horizon_us: int | None
    first_input_event_id: str
    last_input_event_id: str
    output_ordinal: int
    payload: JsonValue
    data_class: Literal["prospective_protected", "shadow_protected"]
    authority_status: Literal["recorded", "unapproved"]


@dataclass(frozen=True)
class StoreResult:
    output_id: str
    inserted: bool


def canonical_json_text(value: JsonValue, *, max_bytes: int) -> str:
    """Admit canonical JSON within an explicit positive byte ceiling."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise JsonAdmissionError(f"value is not canonical JSON: {error}") from error
    if len(encoded) > max_bytes:
        raise JsonAdmissionError(f"canonical JSON exceeds {max_bytes} bytes")
    return encoded.decode("utf-8")


def _payload_hash(record: IdeaOutputRecord) -> str:
    payload_text = canonical_json_text(record.payload, max_bytes=MAX_IDEA_OUTPUT_JSON_BYTES)
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def deterministic_output_id(record: IdeaOutputRecord) -> str:
    """Derive retry-stable output identity from the accepted Phase 1 contract fields."""

    identity = cast(
        JsonValue,
        {
            "instance_id": record.instance_id,
            "first_input_event_id": record.first_input_event_id,
            "last_input_event_id": record.last_input_event_id,
            "output_kind": record.output_kind,
            "output_ordinal": record.output_ordinal,
            "as_of_at_us": record.as_of_at_us,
            "payload_hash": _payload_hash(record),
        },
    )
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _content_hash(record: IdeaOutputRecord, payload_hash: str) -> str:
    content = cast(
        JsonValue,
        {
            "run_id": record.run_id,
            "instance_id": record.instance_id,
            "output_kind": record.output_kind,
            "subject_instrument_id": record.subject_instrument_id,
            "emitted_at_us": record.emitted_at_us,
            "as_of_at_us": record.as_of_at_us,
            "valid_until_at_us": record.valid_until_at_us,
            "direction": record.direction,
            "strength": record.strength,
            "confidence": record.confidence,
            "horizon_us": record.horizon_us,
            "first_input_event_id": record.first_input_event_id,
            "last_input_event_id": record.last_input_event_id,
            "output_ordinal": record.output_ordinal,
            "payload_hash": payload_hash,
            "data_class": record.data_class,
            "authority_status": record.authority_status,
        },
    )
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


class OperationalRepository:
    """One-writer repository for generic V2 records; no broker or legacy access."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def put_idea_output(self, record: IdeaOutputRecord) -> StoreResult:
        """Insert once, accept an exact retry, and fail closed on an identity collision."""

        proposal = record.output_kind in {"proposed_position", "proposed_trade"}
        expected_status = "unapproved" if proposal else "recorded"
        if record.authority_status != expected_status:
            raise ValueError(f"{record.output_kind} requires authority_status={expected_status}")
        if proposal:
            ensure_authority_free_json(record.payload)
        payload_json = canonical_json_text(
            record.payload,
            max_bytes=MAX_IDEA_OUTPUT_JSON_BYTES,
        )
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        output_id = deterministic_output_id(record)
        content_hash = _content_hash(record, payload_hash)
        connection = connect_v2(self.database_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO idea_outputs(
                    output_id, run_id, instance_id, output_kind, subject_instrument_id,
                    emitted_at_us, as_of_at_us, valid_until_at_us, direction, strength,
                    confidence, horizon_us, first_input_event_id, last_input_event_id,
                    output_ordinal, payload_json, payload_hash, content_hash, data_class,
                    authority_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(output_id) DO NOTHING
                """,
                (
                    output_id,
                    record.run_id,
                    record.instance_id,
                    record.output_kind,
                    record.subject_instrument_id,
                    record.emitted_at_us,
                    record.as_of_at_us,
                    record.valid_until_at_us,
                    record.direction,
                    record.strength,
                    record.confidence,
                    record.horizon_us,
                    record.first_input_event_id,
                    record.last_input_event_id,
                    record.output_ordinal,
                    payload_json,
                    payload_hash,
                    content_hash,
                    record.data_class,
                    record.authority_status,
                ),
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                existing = connection.execute(
                    "SELECT content_hash FROM idea_outputs WHERE output_id = ?",
                    (output_id,),
                ).fetchone()
                if existing is None or str(existing["content_hash"]) != content_hash:
                    raise IdentityCollisionError(
                        f"output identity {output_id} already names different content"
                    )
            connection.commit()
            return StoreResult(output_id=output_id, inserted=inserted)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
