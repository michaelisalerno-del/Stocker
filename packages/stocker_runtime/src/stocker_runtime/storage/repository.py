"""Small authoritative repository seams for bounded Stocker V2 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from stocker_runtime.domain import (
    FrozenJsonObject,
    JsonValue,
    ProposedTradeLeg,
    canonical_json_bytes,
    ensure_authority_free_json,
)
from stocker_runtime.storage.connection import connect_v2

MAX_EXTENSION_JSON_BYTES = 64 * 1024
MAX_IDEA_OUTPUT_JSON_BYTES = 16 * 1024
MAX_CALLBACK_PAYLOAD_BYTES = 64 * 1024
MAX_NONTERMINAL_CALLBACK_ROWS = 50_000


class JsonAdmissionError(ValueError):
    """Canonical JSON is invalid or exceeds its durable admission bound."""


class IdentityCollisionError(RuntimeError):
    """A deterministic identity already names different durable content."""


class ProvenanceError(RuntimeError):
    """A write attempts to cross a run, mode, instance, or protected-data boundary."""


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
    input_event_ids: tuple[str, ...] = ()
    legs: tuple[ProposedTradeLeg, ...] = ()


@dataclass(frozen=True)
class StoreResult:
    output_id: str
    inserted: bool


@dataclass(frozen=True)
class CallbackReceiptRecord:
    """Canonical receipt material used to verify and extend the callback evidence chain."""

    batch_id: str
    run_id: str
    first_source_sequence: int
    last_source_sequence: int
    callback_count: int
    first_received_at_us: int
    last_received_at_us: int
    kind_counts: JsonValue
    status_counts: JsonValue
    callback_rows_hash: str
    first_normalized_event_id: str | None
    last_normalized_event_id: str | None
    created_at_us: int
    prior_chain_hash: str


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


def _freeze_json(value: object) -> JsonValue:
    if isinstance(value, dict):
        return cast(JsonValue, FrozenJsonObject(value))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return cast(JsonValue, value)


def _snapshot_payload(value: JsonValue) -> tuple[str, JsonValue]:
    payload_json = canonical_json_text(value, max_bytes=MAX_IDEA_OUTPUT_JSON_BYTES)
    parsed = json.loads(payload_json)
    return payload_json, _freeze_json(parsed)


def _output_id_from_payload_hash(record: IdeaOutputRecord, payload_hash: str) -> str:
    input_event_ids = record.input_event_ids or (record.first_input_event_id,)
    legs = cast(JsonValue, [leg.model_dump(mode="json") for leg in record.legs])
    identity = cast(
        JsonValue,
        {
            "instance_id": record.instance_id,
            "first_input_event_id": record.first_input_event_id,
            "last_input_event_id": record.last_input_event_id,
            "input_event_ids": input_event_ids,
            "output_kind": record.output_kind,
            "output_ordinal": record.output_ordinal,
            "as_of_at_us": record.as_of_at_us,
            "payload_hash": payload_hash,
            "legs": legs,
        },
    )
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def deterministic_output_id(record: IdeaOutputRecord) -> str:
    """Derive retry-stable output identity from the accepted Phase 1 contract fields."""

    payload_json, _ = _snapshot_payload(record.payload)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return _output_id_from_payload_hash(record, payload_hash)


def receipt_chain_hash(record: CallbackReceiptRecord) -> str:
    """Hash every receipt field, including its explicit predecessor link."""

    material = cast(
        JsonValue,
        {
            "batch_id": record.batch_id,
            "run_id": record.run_id,
            "first_source_sequence": record.first_source_sequence,
            "last_source_sequence": record.last_source_sequence,
            "callback_count": record.callback_count,
            "first_received_at_us": record.first_received_at_us,
            "last_received_at_us": record.last_received_at_us,
            "kind_counts": record.kind_counts,
            "status_counts": record.status_counts,
            "callback_rows_hash": record.callback_rows_hash,
            "first_normalized_event_id": record.first_normalized_event_id,
            "last_normalized_event_id": record.last_normalized_event_id,
            "created_at_us": record.created_at_us,
            "prior_chain_hash": record.prior_chain_hash,
        },
    )
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def callback_rows_hash(rows: tuple[Mapping[str, object], ...]) -> str:
    """Bind a receipt to the authoritative identity and outcome of each callback row."""

    def integer(row: Mapping[str, object], field: str) -> int:
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"callback {field} must be an integer")
        return value

    material = cast(
        JsonValue,
        [
            {
                "event_uid": str(row["event_uid"]),
                "payload_sha256": str(row["payload_sha256"]),
                "run_id": str(row["run_id"]),
                "source_sequence": integer(row, "source_sequence"),
                "callback_kind": str(row["callback_kind"]),
                "lifecycle": str(row["lifecycle"]),
                "received_at_us": integer(row, "received_at_us"),
                "provider_at_us": (
                    None if row["provider_at_us"] is None else integer(row, "provider_at_us")
                ),
                "normalized_event_id": (
                    None if row["normalized_event_id"] is None else str(row["normalized_event_id"])
                ),
                "acknowledged_at_us": (
                    None
                    if row["acknowledged_at_us"] is None
                    else integer(row, "acknowledged_at_us")
                ),
                "failure_code": (None if row["failure_code"] is None else str(row["failure_code"])),
            }
            for row in sorted(rows, key=lambda item: integer(item, "source_sequence"))
        ],
    )
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _content_hash(record: IdeaOutputRecord, payload_hash: str) -> str:
    input_event_ids = record.input_event_ids or (record.first_input_event_id,)
    legs = cast(JsonValue, [leg.model_dump(mode="json") for leg in record.legs])
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
            "input_event_ids": input_event_ids,
            "output_ordinal": record.output_ordinal,
            "payload_hash": payload_hash,
            "data_class": record.data_class,
            "authority_status": record.authority_status,
            "legs": legs,
        },
    )
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


class OperationalRepository:
    """One-writer repository for generic V2 records; no broker or legacy access."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def put_idea_output(self, record: IdeaOutputRecord) -> StoreResult:
        """Insert once, accept an exact retry, and fail closed on an identity collision."""

        payload_json, payload_snapshot = _snapshot_payload(record.payload)
        proposal = record.output_kind in {"proposed_position", "proposed_trade"}
        expected_status = "unapproved" if proposal else "recorded"
        if record.authority_status != expected_status:
            raise ValueError(f"{record.output_kind} requires authority_status={expected_status}")
        if proposal:
            ensure_authority_free_json(payload_snapshot)
        input_event_ids = record.input_event_ids or (
            (record.first_input_event_id,)
            if record.first_input_event_id == record.last_input_event_id
            else ()
        )
        if not input_event_ids or len(input_event_ids) > 256:
            raise ValueError("idea output requires 1..256 ordered input event ids")
        if len(set(input_event_ids)) != len(input_event_ids):
            raise ValueError("idea output input event ids must be unique")
        if (
            input_event_ids[0] != record.first_input_event_id
            or input_event_ids[-1] != record.last_input_event_id
        ):
            raise ValueError("idea output input endpoints do not match full provenance")
        if record.output_kind == "proposed_trade" and not record.legs:
            raise ValueError("proposed trade requires typed legs")
        if record.output_kind != "proposed_trade" and record.legs:
            raise ValueError("only proposed trades may contain legs")
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        output_id = _output_id_from_payload_hash(record, payload_hash)
        content_hash = _content_hash(record, payload_hash)
        connection = connect_v2(self.database_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            provenance = connection.execute(
                """
                SELECT instance.run_id, instance.data_class, instance.mode,
                       run.mode AS run_mode, run.data_class AS run_data_class,
                       first_event.run_id AS first_event_run_id,
                       last_event.run_id AS last_event_run_id
                FROM idea_instances instance
                JOIN runs run ON run.run_id = instance.run_id
                LEFT JOIN market_events first_event
                    ON first_event.event_id = ?
                LEFT JOIN market_events last_event
                    ON last_event.event_id = ?
                WHERE instance.instance_id = ?
                """,
                (
                    record.first_input_event_id,
                    record.last_input_event_id,
                    record.instance_id,
                ),
            ).fetchone()
            if (
                provenance is None
                or str(provenance["run_id"]) != record.run_id
                or str(provenance["data_class"]) != record.data_class
                or str(provenance["mode"]) != str(provenance["run_mode"])
                or str(provenance["data_class"]) != str(provenance["run_data_class"])
                or str(provenance["first_event_run_id"]) != record.run_id
                or str(provenance["last_event_run_id"]) != record.run_id
            ):
                raise ProvenanceError("idea output provenance does not match its run and instance")
            placeholders = ",".join("?" for _ in input_event_ids)
            input_rows = connection.execute(
                f"SELECT event_id, run_id FROM market_events WHERE event_id IN ({placeholders})",  # noqa: S608
                input_event_ids,
            ).fetchall()
            if len(input_rows) != len(input_event_ids) or any(
                str(item["run_id"]) != record.run_id for item in input_rows
            ):
                raise ProvenanceError("full idea output provenance crosses its run")
            logical = connection.execute(
                "SELECT output_id, content_hash FROM idea_outputs WHERE instance_id=? "
                "AND input_watermark=? AND output_kind=? AND output_ordinal=? AND as_of_at_us=?",
                (
                    record.instance_id,
                    record.last_input_event_id,
                    record.output_kind,
                    record.output_ordinal,
                    record.as_of_at_us,
                ),
            ).fetchone()
            if logical is not None and (
                str(logical["output_id"]) != output_id
                or str(logical["content_hash"]) != content_hash
            ):
                raise IdentityCollisionError(
                    "logical output identity already names different content"
                )
            cursor = connection.execute(
                """
                INSERT INTO idea_outputs(
                    output_id, run_id, instance_id, output_kind, subject_instrument_id,
                    emitted_at_us, as_of_at_us, valid_until_at_us, direction, strength,
                    confidence, horizon_us, first_input_event_id, last_input_event_id,
                    input_watermark, input_events_hash, output_ordinal, payload_json,
                    payload_hash, content_hash, data_class, authority_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.last_input_event_id,
                    hashlib.sha256(
                        canonical_json_bytes(
                            cast(
                                JsonValue,
                                input_event_ids,
                            )
                        )
                    ).hexdigest(),
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
            for ordinal, event_id in enumerate(input_event_ids):
                connection.execute(
                    "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
                    "VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                    (output_id, event_id, ordinal),
                )
            for ordinal, leg in enumerate(record.legs):
                connection.execute(
                    "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                    "target, quantity_value, notional_value, currency, price_hint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    (
                        output_id,
                        ordinal,
                        leg.instrument_id,
                        leg.action,
                        leg.target,
                        leg.quantity_value,
                        leg.notional_value,
                        leg.currency,
                        leg.price_hint,
                    ),
                )
            connection.commit()
            return StoreResult(output_id=output_id, inserted=inserted)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
